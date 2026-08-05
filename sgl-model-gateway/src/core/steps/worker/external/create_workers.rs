//! External worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::info;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::core::{
    circuit_breaker::CircuitBreakerConfig,
    model_card::ModelCard,
    steps::workflow_data::{ExternalWorkerWorkflowData, WorkerList},
    worker::{HealthConfig, RuntimeType, WorkerType},
    BasicWorkerBuilder, ConnectionMode, Worker,
};

/// Normalize URL for external APIs (ensure https://).
fn normalize_external_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{}", url)
    }
}

fn build_external_process_worker(
    url: &str,
    models: Vec<ModelCard>,
    api_key: Option<&str>,
    labels: &HashMap<String, String>,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthConfig,
) -> Arc<dyn Worker> {
    let initially_healthy = health_config.disable_health_check;
    let mut builder = BasicWorkerBuilder::new(url)
        .models(models)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .runtime_type(RuntimeType::External)
        .labels(labels.clone())
        .circuit_breaker_config(circuit_breaker_config)
        .health_config(health_config)
        .initially_healthy(initially_healthy);

    if let Some(api_key) = api_key {
        builder = builder.api_key(api_key);
    }

    Arc::new(builder.build())
}

/// Step 2: Create one process worker containing every discovered model.
pub struct CreateExternalWorkersStep;

#[async_trait]
impl StepExecutor<ExternalWorkerWorkflowData> for CreateExternalWorkersStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<ExternalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let model_cards = &context.data.model_cards;

        // Build configs from router settings
        let circuit_breaker_config = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

        let health_config = {
            let cfg = &app_context.router_config.health_check;
            HealthConfig {
                timeout_secs: cfg.timeout_secs,
                check_interval_secs: cfg.check_interval_secs,
                endpoint: cfg.endpoint.clone(),
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                disable_health_check: cfg.disable_health_check || config.disable_health_check,
            }
        };

        // Build labels from config
        let mut labels: HashMap<String, String> = config.labels.clone();
        if let Some(priority) = config.priority {
            labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            labels.insert("cost".to_string(), cost.to_string());
        }

        // Normalize URL (ensure https:// for external APIs)
        let normalized_url = normalize_external_url(&config.url);

        let worker = build_external_process_worker(
            &normalized_url,
            model_cards.clone(),
            config.api_key.as_deref(),
            &labels,
            circuit_breaker_config,
            health_config,
        );

        if model_cards.is_empty() {
            info!(
                "Created wildcard worker at {} (accepts any model, user auth forwarded)",
                normalized_url
            );
        } else {
            info!(
                "Created one external process worker for {} models at {}",
                model_cards.len(),
                normalized_url
            );
        }
        let workers = vec![worker];

        // Store results in workflow data
        context.data.workers = Some(WorkerList::from_workers(&workers));
        context.data.actual_workers = Some(workers);
        context.data.labels = labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::UNKNOWN_MODEL_ID;

    #[test]
    fn external_process_worker_owns_all_discovered_models() {
        let models = vec![
            ModelCard::new("model-a").with_alias("model-a-versioned"),
            ModelCard::new("model-b"),
        ];

        let worker = build_external_process_worker(
            "https://provider.test",
            models,
            Some("secret"),
            &HashMap::new(),
            CircuitBreakerConfig::default(),
            HealthConfig::default(),
        );

        assert_eq!(worker.models().len(), 2);
        assert_eq!(
            worker.model_ids(),
            vec!["model-a", "model-a-versioned", "model-b"]
        );
        assert!(worker.supports_model("model-a"));
        assert!(worker.supports_model("model-a-versioned"));
        assert!(worker.supports_model("model-b"));
        assert_eq!(worker.api_key().as_deref(), Some("secret"));
        assert!(!worker.is_healthy());
    }

    #[test]
    fn wildcard_external_process_retains_wildcard_identity_and_health_policy() {
        let worker = build_external_process_worker(
            "https://provider.test",
            Vec::new(),
            None,
            &HashMap::new(),
            CircuitBreakerConfig::default(),
            HealthConfig {
                disable_health_check: true,
                ..HealthConfig::default()
            },
        );

        assert!(worker.models().is_empty());
        assert_eq!(worker.model_ids(), vec![UNKNOWN_MODEL_ID]);
        assert!(worker.supports_model("any-model"));
        assert!(worker.is_healthy());
    }
}
