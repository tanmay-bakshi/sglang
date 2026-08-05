//! Step to remove workers from policy registry.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    core::{steps::workflow_data::WorkerRemovalWorkflowData, Worker},
    policies::PolicyRegistry,
};

fn remove_worker_model_policies(policy_registry: &PolicyRegistry, worker: &dyn Worker) {
    let worker_url = worker.url();
    for model_id in worker.model_ids() {
        policy_registry.remove_worker_from_cache_aware(model_id, worker_url);
        policy_registry.on_worker_removed(model_id);
    }
}

/// Step to remove workers from the policy registry.
///
/// Removes each worker from cache-aware policies and notifies
/// the policy registry of worker removal.
pub struct RemoveFromPolicyRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromPolicyRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers_to_remove = context
            .data
            .actual_workers_to_remove
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        debug!(
            "Removing {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        for worker in workers_to_remove.iter() {
            remove_worker_model_policies(&app_context.policy_registry, worker.as_ref());

            // PD mode keeps prefill/decode cache-aware policies separate from
            // model_policies, so also drop the worker from the matching pool's policy.
            app_context
                .policy_registry
                .remove_pd_worker_from_cache_aware(worker.as_ref());
        }

        debug!(
            "Removed {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::PolicyConfig,
        core::{BasicWorkerBuilder, ModelCard, RuntimeType},
    };

    #[test]
    fn multi_model_worker_removal_clears_every_model_policy() {
        let policy_registry = PolicyRegistry::new(PolicyConfig::Random);
        let worker = BasicWorkerBuilder::new("https://provider.test")
            .runtime_type(RuntimeType::External)
            .models(vec![
                ModelCard::new("model-a").with_alias("model-a-versioned"),
                ModelCard::new("model-b"),
            ])
            .build();
        for model_id in worker.model_ids() {
            policy_registry.on_worker_added(model_id, None);
        }

        remove_worker_model_policies(&policy_registry, &worker);

        assert!(policy_registry.get_worker_counts().is_empty());
        assert!(policy_registry.get_all_mappings().is_empty());
        for model_id in ["model-a", "model-a-versioned", "model-b"] {
            assert!(policy_registry.get_policy(model_id).is_none());
        }
    }
}
