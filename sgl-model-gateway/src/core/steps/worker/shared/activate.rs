//! Unified worker activation step.

use async_trait::async_trait;
use tracing::info;
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::core::steps::workflow_data::WorkerRegistrationData;

/// Unified step to activate workers by marking them as healthy.
///
/// This is the final step in any worker registration workflow.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        for worker in workers.iter() {
            app_context
                .worker_registry
                .activate(worker)
                .await
                .map_err(|error| WorkflowError::StepFailed {
                    step_id: StepId::new("activate_workers"),
                    message: error.to_string(),
                })?;
        }

        info!("Activated {} worker(s) (marked as healthy)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
