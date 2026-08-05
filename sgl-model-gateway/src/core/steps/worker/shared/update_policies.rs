//! Unified policy update step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    core::{steps::workflow_data::WorkerRegistrationData, Worker, WorkerRegistry},
    policies::PolicyRegistry,
};

/// Unified step to update policy registry for registered workers.
///
/// Handles both local workers (same model, possibly DP-aware) and
/// external workers (different models per worker).
pub struct UpdatePoliciesStep;

impl UpdatePoliciesStep {
    /// Check for conflicts between prefill and decode worker configurations for a model.
    fn check_worker_conflicts(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        let prefill_workers: Vec<_> = workers
            .iter()
            .filter(|w| {
                w.metadata()
                    .labels
                    .get("disaggregation_mode")
                    .map(|s| s.as_str())
                    == Some("prefill")
            })
            .collect();

        let decode_workers: Vec<_> = workers
            .iter()
            .filter(|w| {
                w.metadata()
                    .labels
                    .get("disaggregation_mode")
                    .map(|s| s.as_str())
                    == Some("decode")
            })
            .collect();

        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return;
        }

        // Compare configurations of prefill vs decode workers
        if let (Some(pw), Some(dw)) = (prefill_workers.first(), decode_workers.first()) {
            let pl = &pw.metadata().labels;
            let dl = &dw.metadata().labels;

            // Define keys to check for equality
            let keys_to_check = ["tp_size", "dp_size", "load_balance_method"];

            for key in keys_to_check {
                let p_val = pl.get(key);
                let d_val = dl.get(key);
                if p_val != d_val {
                    warn!(
                        "Model {} has conflicting {}: prefill={:?}, decode={:?}",
                        model_id, key, p_val, d_val
                    );
                }
            }

            // Specific check for Data-Parallel consistency
            if let Some(dp_size) = pl.get("dp_size").and_then(|s| s.parse::<usize>().ok()) {
                if dp_size > 1 {
                    let plb = pl.get("load_balance_method").map(|s| s.as_str());
                    if plb != Some("follow_bootstrap_room") {
                        warn!(
                            "Model {} has dp_size > 1 but load_balance_method is not 'follow_bootstrap_room' on prefill workers. This may cause rank mismatch in disaggregated mode.",
                            model_id
                        );
                    }
                }
            }
        }
    }

    fn update_worker_models(
        &self,
        worker_registry: &WorkerRegistry,
        policy_registry: &PolicyRegistry,
        worker: &Arc<dyn Worker>,
        policy_hint: Option<&str>,
    ) -> Vec<String> {
        let mut updated_models = Vec::new();
        for model_id in worker.model_ids() {
            policy_registry.on_worker_added(model_id, policy_hint);

            let all_workers = worker_registry.get_by_model(model_id);
            self.check_worker_conflicts(model_id, &all_workers);
            if let Some(policy) = policy_registry.get_policy(model_id) {
                if policy.name() == "cache_aware" {
                    policy_registry.init_cache_aware_policy(model_id, &all_workers);
                }
            }

            updated_models.push(model_id.to_string());
        }
        updated_models
    }
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for UpdatePoliciesStep {
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

        let labels = context
            .data
            .get_labels()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("labels".to_string()))?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        let mut updated_models = HashSet::new();

        for worker in workers.iter() {
            updated_models.extend(self.update_worker_models(
                &app_context.worker_registry,
                &app_context.policy_registry,
                worker,
                policy_hint,
            ));
        }

        // Initialize bucket policies for prefill workers (local workers only)
        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        if !prefill_workers.is_empty() {
            let policy = app_context.policy_registry.get_prefill_policy();
            if policy.name() == "bucket" {
                app_context
                    .policy_registry
                    .init_pd_bucket_policies(&prefill_workers);
            }
        }

        // Initialize cache-aware policies for PD mode (prefill_policy / decode_policy
        // are separate instances from model_policies and are not touched by the
        // per-model loop above). `init_workers` is idempotent so re-running on each
        // registration is safe.
        let decode_workers = app_context.worker_registry.get_decode_workers();
        let prefill_is_cache_aware =
            app_context.policy_registry.get_prefill_policy().name() == "cache_aware";
        let decode_is_cache_aware =
            app_context.policy_registry.get_decode_policy().name() == "cache_aware";
        if (prefill_is_cache_aware && !prefill_workers.is_empty())
            || (decode_is_cache_aware && !decode_workers.is_empty())
        {
            app_context
                .policy_registry
                .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);
        }

        debug!(
            "Updated policies for {} workers across {} models",
            workers.len(),
            updated_models.len()
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
    fn multi_model_worker_registers_a_policy_for_every_identity() {
        let worker_registry = WorkerRegistry::new();
        let policy_registry = PolicyRegistry::new(PolicyConfig::Random);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://provider.test")
                .runtime_type(RuntimeType::External)
                .models(vec![
                    ModelCard::new("model-a").with_alias("model-a-versioned"),
                    ModelCard::new("model-b"),
                ])
                .build(),
        );
        worker_registry.register(Arc::clone(&worker)).unwrap();

        let updated_models = UpdatePoliciesStep.update_worker_models(
            &worker_registry,
            &policy_registry,
            &worker,
            None,
        );

        assert_eq!(
            updated_models.into_iter().collect::<HashSet<_>>(),
            HashSet::from([
                "model-a".to_string(),
                "model-a-versioned".to_string(),
                "model-b".to_string(),
            ])
        );
        assert_eq!(
            policy_registry.get_worker_counts(),
            std::collections::HashMap::from([
                ("model-a".to_string(), 1),
                ("model-a-versioned".to_string(), 1),
                ("model-b".to_string(), 1),
            ])
        );
    }
}
