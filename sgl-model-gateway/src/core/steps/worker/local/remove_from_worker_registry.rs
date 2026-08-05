//! Step to remove workers from worker registry.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    core::{steps::workflow_data::WorkerRemovalWorkflowData, WorkerRemovalOutcome},
    observability::metrics::Metrics,
};

/// Step to remove workers from the worker registry.
///
/// Removes each worker by URL from the central worker registry.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromWorkerRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let worker_urls = &context.data.worker_urls;

        debug!(
            "Removing {} worker(s) from worker registry",
            worker_urls.len()
        );

        let mut unique_configs = HashSet::new();
        for worker_url in worker_urls {
            if let Some(worker) = app_context.worker_registry.get_by_url(worker_url) {
                let metadata = worker.metadata();
                for model_id in worker.model_ids() {
                    unique_configs.insert((
                        metadata.worker_type.clone(),
                        metadata.connection_mode.clone(),
                        model_id.to_string(),
                    ));
                }
            }
        }

        let mut removed_count = 0;
        for worker_url in worker_urls.iter() {
            let outcome = app_context
                .worker_registry
                .remove_by_url(worker_url)
                .map_err(|error| WorkflowError::StepFailed {
                    step_id: StepId::new("remove_from_worker_registry"),
                    message: error.to_string(),
                })?;
            match outcome {
                WorkerRemovalOutcome::Removed(_) => removed_count += 1,
                WorkerRemovalOutcome::Draining { block, .. } => {
                    debug!(
                        "Unpublished worker {} while its PD generation drains retained ownership: {:?}",
                        worker_url, block
                    );
                    removed_count += 1;
                }
                WorkerRemovalOutcome::NotFound => {}
            }
        }

        // Log if some workers were already removed (e.g., by another process)
        if removed_count != worker_urls.len() {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process)",
                removed_count,
                worker_urls.len()
            );
        } else {
            debug!("Removed {} worker(s) from registry", removed_count);
        }

        // Update Layer 3 worker pool size metrics for unique configurations
        for (worker_type, connection_mode, model_id) in unique_configs {
            // Get labels before moving values into get_workers_filtered
            let worker_type_label = worker_type.as_metric_label();
            let connection_mode_label = connection_mode.as_metric_label();

            let pool_size = app_context
                .worker_registry
                .get_workers_filtered(
                    Some(&model_id),
                    Some(worker_type),
                    Some(connection_mode),
                    None,
                    false,
                )
                .len();

            Metrics::set_worker_pool_size(
                worker_type_label,
                connection_mode_label,
                &model_id,
                pool_size,
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
