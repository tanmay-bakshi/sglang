//! Unified worker registration step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{core::steps::workflow_data::WorkerRegistrationData, observability::metrics::Metrics};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct RegisterWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut worker_ids = Vec::with_capacity(workers.len());

        for worker in workers.iter() {
            let worker_id = app_context
                .worker_registry
                .register(Arc::clone(worker))
                .map_err(|error| WorkflowError::StepFailed {
                    step_id: StepId::new("register_workers"),
                    message: error.to_string(),
                })?;
            debug!(
                "Registered worker {} (models: {:?}) with ID {:?}",
                worker.url(),
                worker.model_ids(),
                worker_id
            );
            worker_ids.push(worker_id);
        }

        let mut unique_configs = HashSet::new();
        for worker in workers {
            let metadata = worker.metadata();
            for model_id in worker.model_ids() {
                unique_configs.insert((
                    metadata.worker_type.clone(),
                    metadata.connection_mode.clone(),
                    model_id.to_string(),
                ));
            }
        }

        // Update Layer 3 worker pool size metrics per unique type/connection/model
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

        // Note: worker_ids are stored for potential future use but not persisted
        // as they are internal registry identifiers
        debug!(
            "Registered {} workers with IDs: {:?}",
            worker_ids.len(),
            worker_ids
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
