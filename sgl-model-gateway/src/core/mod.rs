//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Model types and endpoint definitions
//! - Error types
//! - Circuit breaker for reliability
//! - Token buckets for rate limiting
//! - Workflow steps for multi-step operations
//! - Common utilities

// Re-export UNKNOWN_MODEL_ID from protocols for use throughout core
pub use crate::protocols::UNKNOWN_MODEL_ID;

pub mod circuit_breaker;
pub mod error;
pub mod http_origin;
pub mod job_queue;
pub mod metrics_aggregator;
pub mod model_card;
pub mod model_type;
pub mod pd_decoder_directory;
pub mod pd_decoder_grant;
pub mod pd_decoder_pool;
mod pd_discovery;
pub mod pd_process;
pub mod pd_request_session;
pub mod pd_topology;
pub mod retry;
pub mod steps;
pub mod token_bucket;
pub mod worker;
pub mod worker_builder;
pub mod worker_manager;
pub mod worker_registry;
pub mod worker_service;

// Re-export commonly used types for convenience
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use error::{WorkerError, WorkerResult};
pub use http_origin::{HttpOrigin, HttpOriginError};
pub use job_queue::{Job, JobQueue, JobQueueConfig};
pub use model_card::{ModelCard, ProviderType};
pub use pd_decoder_directory::PdGroupRequest;
pub use pd_process::{
    KvTransferProtocol, PdMetadataSchema, PdProcessAdvertisement, PdProcessAdvertisementError,
    PdProcessMetadata, PdProcessMetadataError, PdProcessRegistration, PdProcessRole,
    PrefillBootstrapAdvertisement, PrefillBootstrapEndpoint, PreparedGrantProtocol,
};
pub use pd_request_session::{PdRequestSession, PdRequestSessionError, PdReservedRequestSession};
pub use pd_topology::{
    PdDecoderSpec, PdGroupId, PdPrefillSpec, PdTopology, PdTopologyBootstrapEndpoint,
    PdTopologyError, PdTopologyGroup, PdTopologyProcessSpec, PdTopologyRegistrationError,
    PdTopologySchema,
};
pub use retry::{is_retryable_status, RetryExecutor};
pub use worker::{
    AttachedBody, BasicWorker, ConnectionMode, HealthConfig, RuntimeType, Worker, WorkerLoadGuard,
    WorkerType,
};
pub use worker_builder::{BasicWorkerBuilder, DPAwareWorkerBuilder};
pub use worker_manager::{LoadMonitor, WorkerManager};
pub use worker_registry::{
    HashRing, PdRetirementBlock, WorkerRegistry, WorkerRegistryError, WorkerRemovalOutcome,
};
pub use worker_service::WorkerService;
