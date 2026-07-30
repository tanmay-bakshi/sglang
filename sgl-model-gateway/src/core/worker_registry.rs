//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use parking_lot::Mutex;
use smg_mesh::OptionalMeshSyncManager;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    core::{
        circuit_breaker::CircuitState,
        pd_decoder_directory::{PdDirectoryError, PdProcessDirectory},
        pd_decoder_grant::{DecoderId, PrefillId},
        pd_decoder_pool::DecoderPoolError,
        worker::{HealthChecker, RuntimeType, WorkerType},
        ConnectionMode, PdProcessRole, Worker,
    },
    observability::metrics::Metrics,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index). This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that's consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of (ring_position, worker_url)
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses Arc<str> to share URL across all virtual nodes (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a list of workers.
    /// Creates VIRTUAL_NODES_PER_WORKER entries per worker for even distribution.
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // Create Arc<str> once per worker, share across all virtual nodes
            let url: Arc<str> = Arc::from(worker.url());
            let url_bytes = url.as_bytes();

            // Create multiple virtual nodes per worker
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let mut hasher = blake3::Hasher::new();
                hasher.update(url_bytes);
                hasher.update(b"#");
                hasher.update(&(vnode as u64).to_le_bytes());
                let hash = hasher.finalize();
                let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // Sort by ring position for binary search
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // Take first 8 bytes as u64
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    /// Find worker URL for a key using consistent hashing.
    /// Returns the first healthy worker URL at or after the key's position (clockwise).
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // Binary search to find first entry at or after key_pos
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around
        // Track visited URLs to avoid checking same worker multiple times (virtual nodes)
        let mut checked_urls =
            std::collections::HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // Skip if we already checked this worker (from another virtual node)
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Model index using immutable snapshots for lock-free reads.
/// Each model maps to an Arc'd slice of workers that can be read without locking.
/// Updates create new snapshots (copy-on-write semantics).
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

#[derive(Debug)]
pub enum WorkerRemovalOutcome {
    NotFound,
    Removed(Arc<dyn Worker>),
    Draining {
        worker: Arc<dyn Worker>,
        block: PdRetirementBlock,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdRetirementBlock {
    PrefillPoolInUse {
        request_chains: usize,
        assignments: usize,
        active_rooms: usize,
        quarantined_cohorts: usize,
    },
    DecoderInUse {
        decoder_id: String,
        active_cohorts: usize,
    },
}

#[derive(Debug, Error)]
pub enum WorkerRegistryError {
    #[error("invalid PD lifecycle transition for {worker_url}: {reason}")]
    InvalidPdLifecycleTransition { worker_url: String, reason: String },
    #[error("PD lifecycle operation failed for {worker_url}: {reason}")]
    PdLifecycle { worker_url: String, reason: String },
    #[error("PD retirement failed after {worker_url} was drained and unpublished: {reason}")]
    PdRetirementFailedAfterDrain { worker_url: String, reason: String },
}

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses Arc<[T]> instead of Arc<RwLock<Vec<T>>> for lock-free reads.
    model_index: ModelIndex,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Generation-aware authority for every PD process lifecycle.
    pd_process_directory: Arc<PdProcessDirectory>,

    /// Serializes directory transitions with generic index publication.
    lifecycle: Arc<Mutex<()>>,

    /// Optional mesh sync manager for state synchronization
    /// When None, the registry works independently without mesh synchronization
    /// Uses RwLock for thread-safe access when setting mesh_sync after initialization
    mesh_sync: Arc<RwLock<OptionalMeshSyncManager>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            pd_process_directory: Arc::new(PdProcessDirectory::default()),
            lifecycle: Arc::new(Mutex::new(())),
            mesh_sync: Arc::new(RwLock::new(None)),
        }
    }

    /// Rebuild the hash ring for a model based on current workers in the model index
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        }
    }

    /// Get the hash ring for a model (O(1) lookup)
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Set mesh sync manager (thread-safe, can be called after initialization)
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        *self.mesh_sync.write().unwrap() = mesh_sync;
    }

    fn remove_from_indexes(&self, worker_id: &WorkerId, worker: &Arc<dyn Worker>) {
        let model_id = worker.model_id().to_string();
        if let Some(mut entry) = self.model_index.get_mut(&model_id) {
            let workers: Vec<Arc<dyn Worker>> = entry
                .iter()
                .filter(|indexed| indexed.url() != worker.url())
                .cloned()
                .collect();
            *entry = Arc::from(workers.into_boxed_slice());
        }
        self.rebuild_hash_ring(&model_id);

        if let Some(mut workers) = self.type_workers.get_mut(worker.worker_type()) {
            workers.retain(|id| id != worker_id);
        }
        if let Some(mut workers) = self.connection_workers.get_mut(worker.connection_mode()) {
            workers.retain(|id| id != worker_id);
        }
    }

    /// Register a new worker.
    pub fn register(&self, worker: Arc<dyn Worker>) -> Result<WorkerId, WorkerRegistryError> {
        let _lifecycle = self.lifecycle.lock();
        self.retire_drained_locked()?;

        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            existing_id.clone()
        } else {
            WorkerId::new()
        };
        let previous = self.workers.get(&worker_id).map(|entry| entry.clone());
        if let Err(error) = self.prepare_pd_registration(&worker, previous.as_ref()) {
            if let Some(previous) = previous {
                Metrics::set_worker_health(previous.url(), previous.is_healthy());
            } else {
                Metrics::remove_worker_metrics(worker.url());
            }
            return Err(error);
        }
        self.publish_worker(worker_id.clone(), worker);
        Ok(worker_id)
    }

    fn publish_worker(&self, worker_id: WorkerId, worker: Arc<dyn Worker>) {
        if let Some(previous) = self.workers.get(&worker_id).map(|entry| entry.clone()) {
            self.remove_from_indexes(&worker_id, &previous);
        }
        self.workers.insert(worker_id.clone(), Arc::clone(&worker));
        Metrics::set_worker_health(worker.url(), worker.is_healthy());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index for O(1) lookups using copy-on-write
        // This creates a new immutable snapshot with the added worker
        let model_id = worker.model_id().to_string();
        self.model_index
            .entry(model_id.clone())
            .and_modify(|existing| {
                // Create new snapshot with the additional worker
                let mut new_workers: Vec<Arc<dyn Worker>> = existing.iter().cloned().collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker.clone()].into_boxed_slice()));

        // Rebuild hash ring for this model
        self.rebuild_hash_ring(&model_id);

        // Update type index (clone needed for DashMap key ownership)
        self.type_workers
            .entry(worker.worker_type().clone())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(worker.connection_mode().clone())
            .or_default()
            .push(worker_id.clone());

        // Sync to mesh if enabled (no-op if mesh is not enabled)
        if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
            mesh_sync.sync_worker_state(
                worker_id.as_str().to_string(),
                worker.model_id().to_string(),
                worker.url().to_string(),
                worker.is_healthy(),
                0.0, // TODO: Get actual load
            );
        }
    }

    fn prepare_pd_registration(
        &self,
        worker: &Arc<dyn Worker>,
        previous: Option<&Arc<dyn Worker>>,
    ) -> Result<(), WorkerRegistryError> {
        let role = pd_worker_role(worker);
        let previous_metadata = previous.and_then(|current| current.metadata().pd_process.as_ref());
        let metadata = worker.metadata().pd_process.as_ref();

        if previous_metadata.is_some() && role.is_none() {
            return Err(WorkerRegistryError::InvalidPdLifecycleTransition {
                worker_url: worker.url().to_string(),
                reason: "a PD process cannot become a regular worker without retirement"
                    .to_string(),
            });
        }
        if let (Some(previous_metadata), Some(metadata)) = (previous_metadata, metadata) {
            if previous_metadata.launch_instance_id() == metadata.launch_instance_id()
                && previous_metadata.role() != metadata.role()
            {
                return Err(WorkerRegistryError::InvalidPdLifecycleTransition {
                    worker_url: worker.url().to_string(),
                    reason: "one launch generation cannot change PD roles".to_string(),
                });
            }
        }

        let same_generation = previous_metadata
            .zip(metadata)
            .is_some_and(|(current, incoming)| {
                current.role() == incoming.role()
                    && current.launch_instance_id() == incoming.launch_instance_id()
            });
        let result = match (role, same_generation) {
            (Some(PdProcessRole::Prefill), true) => self
                .pd_process_directory
                .refresh_prefill(Arc::clone(worker))
                .map(|_| ()),
            (Some(PdProcessRole::Prefill), false) => self
                .pd_process_directory
                .admit_prefill(Arc::clone(worker))
                .map(|_| ()),
            (Some(PdProcessRole::Decode), true) => self
                .pd_process_directory
                .refresh_decoder(Arc::clone(worker))
                .map(|_| ()),
            (Some(PdProcessRole::Decode), false) => self
                .pd_process_directory
                .admit_decoder(Arc::clone(worker))
                .map(|_| ()),
            (None, _) => return Ok(()),
        };
        result.map_err(|error| WorkerRegistryError::PdLifecycle {
            worker_url: worker.url().to_string(),
            reason: error.to_string(),
        })
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    /// Uses atomic entry API to avoid race conditions between check and insert.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Best-effort lookup of the URL for a given worker ID.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Remove a worker by ID.
    pub fn remove(
        &self,
        worker_id: &WorkerId,
    ) -> Result<WorkerRemovalOutcome, WorkerRegistryError> {
        let _lifecycle = self.lifecycle.lock();
        self.retire_drained_locked()?;
        let Some(worker) = self.workers.get(worker_id).map(|entry| entry.clone()) else {
            return Ok(WorkerRemovalOutcome::NotFound);
        };
        let Some(role) = pd_worker_role(&worker) else {
            self.remove_from_indexes(worker_id, &worker);
            self.finalize_worker_removal(worker_id, &worker);
            return Ok(WorkerRemovalOutcome::Removed(worker));
        };

        let drain_result = match role {
            PdProcessRole::Prefill => {
                let id = prefill_id(&worker)?;
                self.pd_process_directory
                    .drain_prefill(&id)
                    .map_err(|error| WorkerRegistryError::PdLifecycle {
                        worker_url: worker.url().to_string(),
                        reason: error.to_string(),
                    })?;
                self.remove_from_indexes(worker_id, &worker);
                self.finalize_worker_removal(worker_id, &worker);
                self.pd_process_directory
                    .remove_drained_prefill(&id)
                    .map(|entry| Arc::clone(entry.worker()))
            }
            PdProcessRole::Decode => {
                let id = decoder_id(&worker)?;
                self.pd_process_directory
                    .drain_decoder(&id)
                    .map_err(|error| WorkerRegistryError::PdLifecycle {
                        worker_url: worker.url().to_string(),
                        reason: error.to_string(),
                    })?;
                self.remove_from_indexes(worker_id, &worker);
                self.finalize_worker_removal(worker_id, &worker);
                self.pd_process_directory
                    .remove_drained_decoder(&id)
                    .map(|entry| Arc::clone(entry.worker()))
            }
        };
        match drain_result {
            Ok(retired) => Ok(WorkerRemovalOutcome::Removed(retired)),
            Err(error) if retirement_block(&error).is_some() => {
                Ok(WorkerRemovalOutcome::Draining {
                    worker,
                    block: retirement_block(&error)
                        .expect("retirement block was classified in the match guard"),
                })
            }
            Err(error) => Err(WorkerRegistryError::PdRetirementFailedAfterDrain {
                worker_url: worker.url().to_string(),
                reason: error.to_string(),
            }),
        }
    }

    /// Remove a worker by URL.
    pub fn remove_by_url(&self, url: &str) -> Result<WorkerRemovalOutcome, WorkerRegistryError> {
        let Some(worker_id) = self.url_to_id.get(url).map(|entry| entry.clone()) else {
            return Ok(WorkerRemovalOutcome::NotFound);
        };
        self.remove(&worker_id)
    }

    pub fn retire_drained_pd_processes(&self) -> Result<usize, WorkerRegistryError> {
        let _lifecycle = self.lifecycle.lock();
        self.retire_drained_locked()
    }

    pub fn pd_process_directory(&self) -> &Arc<PdProcessDirectory> {
        &self.pd_process_directory
    }

    fn retire_drained_locked(&self) -> Result<usize, WorkerRegistryError> {
        let sweep = self.pd_process_directory.retire_drained();
        let retired_count = sweep.retired.len();
        if !sweep.failures.is_empty() {
            let failure_count = sweep.failures.len();
            let reasons = sweep
                .failures
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<String>>()
                .join("; ");
            return Err(WorkerRegistryError::PdLifecycle {
                worker_url: "<drained-generation-sweep>".to_string(),
                reason: format!(
                    "{failure_count} generation(s) failed retirement: {reasons}; \
                     {retired_count} other generation(s) retired successfully"
                ),
            });
        }
        Ok(retired_count)
    }

    fn finalize_worker_removal(&self, worker_id: &WorkerId, worker: &Arc<dyn Worker>) {
        let is_current = self
            .workers
            .get(worker_id)
            .is_some_and(|current| Arc::ptr_eq(&current, worker));
        if !is_current {
            return;
        }
        self.workers.remove(worker_id);
        if self.url_to_id.get(worker.url()).as_deref() == Some(worker_id) {
            self.url_to_id.remove(worker.url());
        }
        worker.set_healthy(false);
        Metrics::remove_worker_metrics(worker.url());
        if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
            mesh_sync.remove_worker_state(worker_id.as_str());
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Empty worker slice constant for returning when no workers found
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Get all workers for a model (O(1) optimized, lock-free)
    /// Returns an Arc to the immutable worker slice - just an atomic refcount bump.
    /// This is the fastest possible read path with zero contention.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Update worker health status and sync to mesh
    pub fn update_worker_health(&self, worker_id: &WorkerId, is_healthy: bool) {
        if let Some(worker) = self.workers.get(worker_id) {
            // Update worker health (if Worker trait has a method for this)
            // For now, we'll just sync to mesh

            // Sync to mesh if enabled (no-op if mesh is not enabled)
            if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
                mesh_sync.sync_worker_state(
                    worker_id.as_str().to_string(),
                    worker.model_id().to_string(),
                    worker.url().to_string(),
                    is_healthy,
                    0.0, // TODO: Get actual load
                );
            }
        }
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the number of workers in the registry
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// Get all model IDs with workers (lock-free)
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - runtime_type: Filter by runtime type (Sglang, Vllm, External)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers: Vec<Arc<dyn Worker>> = if let Some(model) = model_id {
            self.get_by_model(model).to_vec()
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified (using matches for flexible gRPC matching)
                if let Some(ref conn) = connection_mode {
                    if !w.connection_mode().matches(conn) {
                        return false;
                    }
                }

                // Check runtime_type if specified
                if let Some(ref rt) = runtime_type {
                    if w.metadata().runtime_type != *rt {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics (lock-free)
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc { .. } => grpc_count += 1,
            }

            match worker.circuit_breaker().state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Get counts of regular and PD workers efficiently (O(1))
    /// This avoids the overhead of get_all() which allocates memory and iterates all workers
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        // Get total workers count efficiently from DashMap
        let total_workers = self.workers.len();

        // PD workers are any workers that are not Regular
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub(crate) fn start_health_checker(&self, check_interval_secs: u64) -> HealthChecker {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();
        let pd_process_directory = Arc::clone(&self.pd_process_directory);
        let lifecycle = Arc::clone(&self.lifecycle);

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                let sweep = {
                    let _lifecycle = lifecycle.lock();
                    pd_process_directory.retire_drained()
                };
                if !sweep.retired.is_empty() {
                    tracing::debug!(
                        "Retired {} drained PD process generation(s)",
                        sweep.retired.len()
                    );
                }
                for error in sweep.failures {
                    tracing::error!("Failed to retire a drained PD process generation: {error}");
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks in parallel for better performance
                // This is especially important when there are many workers
                let health_futures: Vec<_> = workers
                    .iter()
                    .filter(|worker| !worker.metadata().health_config.disable_health_check)
                    .map(|worker| {
                        let worker = worker.clone();
                        async move {
                            let _ = worker.check_health_async().await;
                        }
                    })
                    .collect();
                futures::future::join_all(health_futures).await;
            }
        });

        HealthChecker::new(handle, shutdown)
    }
}

fn pd_worker_role(worker: &Arc<dyn Worker>) -> Option<PdProcessRole> {
    match worker.worker_type() {
        WorkerType::Prefill { .. } => Some(PdProcessRole::Prefill),
        WorkerType::Decode => Some(PdProcessRole::Decode),
        WorkerType::Regular => worker
            .metadata()
            .pd_process
            .as_ref()
            .map(|metadata| metadata.role()),
    }
}

fn prefill_id(worker: &Arc<dyn Worker>) -> Result<PrefillId, WorkerRegistryError> {
    let metadata = worker.metadata().pd_process.as_ref().ok_or_else(|| {
        WorkerRegistryError::InvalidPdLifecycleTransition {
            worker_url: worker.url().to_string(),
            reason: "prefill worker is missing typed process metadata".to_string(),
        }
    })?;
    PrefillId::new(worker.base_url(), metadata.launch_instance_id()).map_err(|error| {
        WorkerRegistryError::InvalidPdLifecycleTransition {
            worker_url: worker.url().to_string(),
            reason: error.to_string(),
        }
    })
}

fn decoder_id(worker: &Arc<dyn Worker>) -> Result<DecoderId, WorkerRegistryError> {
    let metadata = worker.metadata().pd_process.as_ref().ok_or_else(|| {
        WorkerRegistryError::InvalidPdLifecycleTransition {
            worker_url: worker.url().to_string(),
            reason: "decoder worker is missing typed process metadata".to_string(),
        }
    })?;
    DecoderId::new(worker.base_url(), metadata.launch_instance_id()).map_err(|error| {
        WorkerRegistryError::InvalidPdLifecycleTransition {
            worker_url: worker.url().to_string(),
            reason: error.to_string(),
        }
    })
}

fn retirement_block(error: &PdDirectoryError) -> Option<PdRetirementBlock> {
    match error {
        PdDirectoryError::Pool(DecoderPoolError::PrefillPoolInUse {
            request_chains,
            assignments,
            active_rooms,
            quarantined_cohorts,
        }) => Some(PdRetirementBlock::PrefillPoolInUse {
            request_chains: *request_chains,
            assignments: *assignments,
            active_rooms: *active_rooms,
            quarantined_cohorts: *quarantined_cohorts,
        }),
        PdDirectoryError::Pool(DecoderPoolError::DecoderInUse {
            decoder_id,
            active_cohorts,
        }) => Some(PdRetirementBlock::DecoderInUse {
            decoder_id: decoder_id.to_string(),
            active_cohorts: *active_cohorts,
        }),
        _ => None,
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Barrier},
        thread,
    };

    use uuid::Uuid;

    use super::*;
    use crate::core::{
        circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder, KvTransferProtocol,
        PdMetadataSchema, PdProcessMetadata, PrefillBootstrapEndpoint, PreparedGrantProtocol,
    };

    const MODEL_FINGERPRINT: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const KV_LAYOUT_FINGERPRINT: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn pd_metadata(
        role: PdProcessRole,
        tp_size: usize,
        launch_instance_id: Uuid,
    ) -> PdProcessMetadata {
        PdProcessMetadata::new(
            PdMetadataSchema::V1,
            launch_instance_id,
            role,
            tp_size,
            1,
            MODEL_FINGERPRINT,
            KV_LAYOUT_FINGERPRINT,
            "bf16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            match role {
                PdProcessRole::Prefill => {
                    Some(PrefillBootstrapEndpoint::new("prefill-transfer.test", 50_051).unwrap())
                }
                PdProcessRole::Decode => None,
            },
        )
        .unwrap()
    }

    fn pd_worker(
        url: &str,
        role: PdProcessRole,
        tp_size: usize,
        launch_instance_id: Uuid,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(match role {
                    PdProcessRole::Prefill => WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                    PdProcessRole::Decode => WorkerType::Decode,
                })
                .pd_process(pd_metadata(role, tp_size, launch_instance_id))
                .build(),
        )
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker)).unwrap();

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        assert!(matches!(
            registry.remove(&worker_id).unwrap(),
            WorkerRemovalOutcome::Removed(_)
        ));
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1)).unwrap();
        registry.register(Arc::from(worker2)).unwrap();
        registry.register(Arc::from(worker3)).unwrap();

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080").unwrap();
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    #[test]
    fn same_url_registration_replaces_every_selectable_index() {
        let registry = WorkerRegistry::new();
        let original: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker.test:8080")
                .worker_type(WorkerType::Regular)
                .label("model_id", "old-model")
                .build(),
        );
        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker.test:8080")
                .worker_type(WorkerType::Regular)
                .label("model_id", "new-model")
                .build(),
        );

        let original_id = registry.register(Arc::clone(&original)).unwrap();
        let replacement_id = registry.register(Arc::clone(&replacement)).unwrap();

        assert_eq!(original_id, replacement_id);
        assert!(Arc::ptr_eq(
            &registry.get(&replacement_id).unwrap(),
            &replacement
        ));
        assert!(registry.get_by_model("old-model").is_empty());
        let model_workers = registry.get_by_model("new-model");
        assert_eq!(model_workers.len(), 1);
        assert!(Arc::ptr_eq(&model_workers[0], &replacement));
        let regular_workers = registry.get_by_type(&WorkerType::Regular);
        assert_eq!(regular_workers.len(), 1);
        assert!(Arc::ptr_eq(&regular_workers[0], &replacement));
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);
    }

    #[test]
    fn pd_startup_registration_seeds_both_orders_and_arbitrary_decoder_counts() {
        for prefill_first in [false, true] {
            for prefill_tp in [2, 4] {
                let registry = WorkerRegistry::new();
                let prefill = pd_worker(
                    "http://prefill.test:30000",
                    PdProcessRole::Prefill,
                    prefill_tp,
                    Uuid::from_u128(1),
                );
                if prefill_first {
                    registry.register(Arc::clone(&prefill)).unwrap();
                }
                for index in 0..3 {
                    registry
                        .register(pd_worker(
                            &format!("http://decode-{index}.test:30001"),
                            PdProcessRole::Decode,
                            1,
                            Uuid::from_u128(10 + index),
                        ))
                        .unwrap();
                }
                if !prefill_first {
                    registry.register(prefill).unwrap();
                }

                let ready = registry.pd_process_directory().ready_prefills();
                assert_eq!(ready.len(), 1);
                assert_eq!(ready[0].pool().snapshot().replicas.len(), 3);
            }
        }
    }

    #[test]
    fn new_prefill_generation_publishes_after_old_generation_stops_admission() {
        let registry = WorkerRegistry::new();
        let old_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let worker_id = registry.register(Arc::clone(&old_worker)).unwrap();
        let old_id = PrefillId::new(old_worker.base_url(), Uuid::from_u128(1)).unwrap();
        let (old_entry, mut owner) = registry
            .pd_process_directory()
            .begin_prefill_request(&old_id, "owned-old-generation")
            .unwrap();

        let new_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            4,
            Uuid::from_u128(2),
        );
        let replacement_id = registry.register(Arc::clone(&new_worker)).unwrap();
        let new_id = PrefillId::new(new_worker.base_url(), Uuid::from_u128(2)).unwrap();

        assert_eq!(worker_id, replacement_id);
        assert!(Arc::ptr_eq(&registry.get(&worker_id).unwrap(), &new_worker));
        assert!(Arc::ptr_eq(
            registry
                .pd_process_directory()
                .prefill(&old_id)
                .unwrap()
                .worker(),
            &old_worker
        ));
        assert_eq!(
            registry
                .pd_process_directory()
                .prefill_availability(&old_id),
            Some(crate::core::pd_decoder_directory::ProcessAvailability::Draining)
        );
        assert!(registry
            .pd_process_directory()
            .begin_prefill_request(&old_id, "too-late")
            .is_err());
        assert!(Arc::ptr_eq(
            registry
                .pd_process_directory()
                .prefill(&new_id)
                .unwrap()
                .worker(),
            &new_worker
        ));
        assert_eq!(registry.retire_drained_pd_processes().unwrap(), 0);

        old_entry.pool().finalize_request(&mut owner).unwrap();
        assert_eq!(registry.retire_drained_pd_processes().unwrap(), 1);
        assert!(registry.pd_process_directory().prefill(&old_id).is_none());
        assert!(Arc::ptr_eq(&registry.get(&worker_id).unwrap(), &new_worker));
    }

    #[test]
    fn failed_new_generation_never_replaces_generic_selection() {
        let registry = WorkerRegistry::new();
        let old_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let worker_id = registry.register(Arc::clone(&old_worker)).unwrap();
        let invalid_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            3,
            Uuid::from_u128(2),
        );

        assert!(registry.register(invalid_worker).is_err());
        assert!(Arc::ptr_eq(&registry.get(&worker_id).unwrap(), &old_worker));
        let ready = registry.pd_process_directory().ready_prefills();
        assert_eq!(ready.len(), 1);
        assert!(Arc::ptr_eq(ready[0].worker(), &old_worker));
    }

    #[test]
    fn new_launch_can_repurpose_a_url_but_one_generation_cannot_change_roles() {
        let registry = WorkerRegistry::new();
        let prefill = pd_worker(
            "http://process.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let worker_id = registry.register(Arc::clone(&prefill)).unwrap();
        let prefill_id = PrefillId::new(prefill.base_url(), Uuid::from_u128(1)).unwrap();
        let same_generation_decode = pd_worker(
            "http://process.test:30000",
            PdProcessRole::Decode,
            1,
            Uuid::from_u128(1),
        );

        assert!(matches!(
            registry.register(same_generation_decode),
            Err(WorkerRegistryError::InvalidPdLifecycleTransition { .. })
        ));
        assert!(Arc::ptr_eq(&registry.get(&worker_id).unwrap(), &prefill));

        let new_generation_decode = pd_worker(
            "http://process.test:30000",
            PdProcessRole::Decode,
            1,
            Uuid::from_u128(2),
        );
        let replacement_id = registry
            .register(Arc::clone(&new_generation_decode))
            .unwrap();
        let decoder_id =
            DecoderId::new(new_generation_decode.base_url(), Uuid::from_u128(2)).unwrap();

        assert_eq!(worker_id, replacement_id);
        assert!(Arc::ptr_eq(
            &registry.get(&worker_id).unwrap(),
            &new_generation_decode
        ));
        assert_eq!(
            registry
                .pd_process_directory()
                .prefill_availability(&prefill_id),
            Some(crate::core::pd_decoder_directory::ProcessAvailability::Draining)
        );
        assert!(Arc::ptr_eq(
            registry
                .pd_process_directory()
                .decoder(&decoder_id)
                .unwrap()
                .worker(),
            &new_generation_decode
        ));
    }

    #[test]
    fn removal_unpublishes_before_owned_generation_retires_and_allows_restart() {
        let registry = WorkerRegistry::new();
        let old_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let old_worker_id = registry.register(Arc::clone(&old_worker)).unwrap();
        let old_id = PrefillId::new(old_worker.base_url(), Uuid::from_u128(1)).unwrap();
        let (old_entry, mut owner) = registry
            .pd_process_directory()
            .begin_prefill_request(&old_id, "owned-removal")
            .unwrap();

        let outcome = registry.remove(&old_worker_id).unwrap();
        assert!(matches!(
            outcome,
            WorkerRemovalOutcome::Draining {
                block: PdRetirementBlock::PrefillPoolInUse { .. },
                ..
            }
        ));
        assert!(registry.get(&old_worker_id).is_none());
        assert!(registry.get_by_url(old_worker.url()).is_none());
        assert!(registry.get_by_model(old_worker.model_id()).is_empty());

        let new_worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            4,
            Uuid::from_u128(2),
        );
        let new_worker_id = registry.register(Arc::clone(&new_worker)).unwrap();
        assert_ne!(old_worker_id, new_worker_id);
        assert!(Arc::ptr_eq(
            &registry.get(&new_worker_id).unwrap(),
            &new_worker
        ));

        old_entry.pool().finalize_request(&mut owner).unwrap();
        assert_eq!(registry.retire_drained_pd_processes().unwrap(), 1);
        assert!(registry.pd_process_directory().prefill(&old_id).is_none());
        assert!(Arc::ptr_eq(
            &registry.get(&new_worker_id).unwrap(),
            &new_worker
        ));
    }

    #[test]
    fn property_refresh_preserves_generation_metadata_and_pool_identity() {
        let registry = WorkerRegistry::new();
        let original = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let worker_id = registry.register(Arc::clone(&original)).unwrap();
        let process_id = PrefillId::new(original.base_url(), Uuid::from_u128(1)).unwrap();
        let original_entry = registry
            .pd_process_directory()
            .prefill(&process_id)
            .unwrap();
        let refreshed: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(original.url())
                .worker_type(original.worker_type().clone())
                .label("priority", "7")
                .pd_process(
                    original
                        .metadata()
                        .pd_process
                        .clone()
                        .expect("test worker has typed metadata"),
                )
                .build(),
        );

        let refreshed_id = registry.register(Arc::clone(&refreshed)).unwrap();
        let refreshed_entry = registry
            .pd_process_directory()
            .prefill(&process_id)
            .unwrap();
        assert_eq!(worker_id, refreshed_id);
        assert!(Arc::ptr_eq(&registry.get(&worker_id).unwrap(), &refreshed));
        assert!(Arc::ptr_eq(refreshed_entry.worker(), &refreshed));
        assert_eq!(
            original_entry.pool().snapshot(),
            refreshed_entry.pool().snapshot()
        );
        assert_eq!(registry.pd_process_directory().ready_prefills().len(), 1);
    }

    #[test]
    fn successful_pd_removal_erases_generic_and_directory_state() {
        let registry = WorkerRegistry::new();
        let worker = pd_worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            Uuid::from_u128(1),
        );
        let worker_id = registry.register(Arc::clone(&worker)).unwrap();
        let process_id = PrefillId::new(worker.base_url(), Uuid::from_u128(1)).unwrap();

        assert!(matches!(
            registry.remove(&worker_id).unwrap(),
            WorkerRemovalOutcome::Removed(_)
        ));
        assert!(registry.get(&worker_id).is_none());
        assert!(registry
            .pd_process_directory()
            .prefill(&process_id)
            .is_none());
    }

    #[test]
    fn registry_replacement_serializes_against_prefill_request_admission() {
        for iteration in 0..32 {
            let registry = Arc::new(WorkerRegistry::new());
            let old_worker = pd_worker(
                "http://prefill-race.test:30000",
                PdProcessRole::Prefill,
                2,
                Uuid::from_u128(iteration * 2 + 1),
            );
            registry.register(Arc::clone(&old_worker)).unwrap();
            let old_id =
                PrefillId::new(old_worker.base_url(), Uuid::from_u128(iteration * 2 + 1)).unwrap();
            let barrier = Arc::new(Barrier::new(2));

            let admission_registry = Arc::clone(&registry);
            let admission_barrier = Arc::clone(&barrier);
            let admission = thread::spawn(move || {
                admission_barrier.wait();
                if let Ok((entry, mut owner)) = admission_registry
                    .pd_process_directory()
                    .begin_prefill_request(&old_id, format!("request-{iteration}"))
                {
                    entry.pool().finalize_request(&mut owner).unwrap();
                }
            });

            let replacement_registry = Arc::clone(&registry);
            let replacement_barrier = Arc::clone(&barrier);
            let replacement = thread::spawn(move || {
                replacement_barrier.wait();
                let worker = pd_worker(
                    "http://prefill-race.test:30000",
                    PdProcessRole::Prefill,
                    4,
                    Uuid::from_u128(iteration * 2 + 2),
                );
                replacement_registry.register(Arc::clone(&worker)).unwrap();
                worker
            });

            admission.join().unwrap();
            let replacement = replacement.join().unwrap();
            assert!(Arc::ptr_eq(
                &registry.get_by_url(replacement.url()).unwrap(),
                &replacement
            ));
        }
    }
}
