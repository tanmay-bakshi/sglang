use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use parking_lot::RwLock;
use serde::Serialize;
use thiserror::Error;
use uuid::Uuid;

use super::{
    pd_decoder_grant::{DecoderId, DecoderRequestTemplate, PrefillId, ProcessIdentityError},
    pd_decoder_pool::{
        DecoderAvailability, DecoderPool, DecoderPoolError, DecoderReplicaMetadata,
        DecoderSchedulingHints, EngineCompatibilityMetadata, LogicalRequestOwner, PendingAdmission,
        PendingSchedulingCharge,
    },
};
use crate::core::{
    ConnectionMode, HttpOrigin, PdGroupId, PdProcessMetadata, PdProcessRegistration, PdProcessRole,
    PdTopology, PrefillBootstrapEndpoint, Worker, WorkerType,
};

/// Current authenticated process state for one topology prefill.
#[derive(Clone, Debug, Serialize)]
pub struct PdTopologyPrefillStatus {
    pub origin: HttpOrigin,
    pub tensor_parallel_size: usize,
    pub bootstrap_endpoint: crate::core::PdTopologyBootstrapEndpoint,
    pub launch_instance_id: Option<Uuid>,
    pub generation_ready: bool,
    pub healthy: bool,
    pub available: bool,
}

/// Current authenticated process and pool state for one topology decoder.
#[derive(Clone, Debug, Serialize)]
pub struct PdTopologyDecoderStatus {
    pub origin: HttpOrigin,
    pub tensor_parallel_size: usize,
    pub launch_instance_id: Option<Uuid>,
    pub generation_ready: bool,
    pub healthy: bool,
    pub available: bool,
    pub pending_admissions: usize,
    pub active_child_requests: usize,
}

/// Current admission and process state for one immutable topology group.
#[derive(Clone, Debug, Serialize)]
pub struct PdTopologyGroupStatus {
    pub id: PdGroupId,
    pub manifest_index: usize,
    pub eligible: bool,
    pub selection_count: u64,
    pub outstanding_logical_requests: usize,
    pub ready_decoder_count: usize,
    pub prefill: PdTopologyPrefillStatus,
    pub decoders: Vec<PdTopologyDecoderStatus>,
}

/// Read-only attestation of the immutable topology and observed process generations.
#[derive(Clone, Debug, Serialize)]
pub struct PdTopologyStatus {
    pub schema: String,
    pub topology_sha256: String,
    pub topology: PdTopology,
    pub fully_registered: bool,
    pub groups: Vec<PdTopologyGroupStatus>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessAvailability {
    Ready,
    Draining,
    Unavailable,
}

#[derive(Debug)]
pub struct DecoderDirectoryEntry {
    id: DecoderId,
    worker: Arc<dyn Worker>,
    metadata: PdProcessMetadata,
}

impl DecoderDirectoryEntry {
    pub fn id(&self) -> &DecoderId {
        &self.id
    }

    pub fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    pub fn metadata(&self) -> &PdProcessMetadata {
        &self.metadata
    }
}

#[derive(Debug)]
pub struct PrefillDirectoryEntry {
    id: PrefillId,
    worker: Arc<dyn Worker>,
    metadata: PdProcessMetadata,
    bootstrap_endpoint: PrefillBootstrapEndpoint,
    pool: DecoderPool,
}

impl PrefillDirectoryEntry {
    pub fn id(&self) -> &PrefillId {
        &self.id
    }

    pub fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    pub fn metadata(&self) -> &PdProcessMetadata {
        &self.metadata
    }

    pub fn bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.bootstrap_endpoint
    }

    pub(super) fn pool(&self) -> &DecoderPool {
        &self.pool
    }
}

#[derive(Debug)]
struct PrefillRecord {
    entry: Arc<PrefillDirectoryEntry>,
    availability: ProcessAvailability,
}

#[derive(Debug)]
struct DecoderRecord {
    entry: Arc<DecoderDirectoryEntry>,
    availability: ProcessAvailability,
    pool_memberships: HashSet<PrefillId>,
}

#[derive(Debug, Default)]
struct DirectoryState {
    prefills: HashMap<PrefillId, PrefillRecord>,
    current_prefill_by_origin: HashMap<HttpOrigin, PrefillId>,
    decoders: HashMap<DecoderId, DecoderRecord>,
    current_decoder_by_origin: HashMap<HttpOrigin, DecoderId>,
    last_selected_group_index: Option<usize>,
    group_selection_counts: HashMap<PdGroupId, u64>,
}

/// Non-cloneable authority proving that group choice and request charging were atomic.
#[derive(Debug)]
pub struct PdGroupRequest {
    group_id: PdGroupId,
    prefill: Arc<PrefillDirectoryEntry>,
    owner: LogicalRequestOwner,
}

impl PdGroupRequest {
    /// Return the immutable selected group identifier.
    pub fn group_id(&self) -> &PdGroupId {
        &self.group_id
    }

    /// Return the selected prefill generation.
    pub fn prefill(&self) -> &Arc<PrefillDirectoryEntry> {
        &self.prefill
    }

    pub(crate) fn into_parts(self) -> (PdGroupId, Arc<PrefillDirectoryEntry>, LogicalRequestOwner) {
        (self.group_id, self.prefill, self.owner)
    }
}

#[derive(Debug)]
pub(crate) struct PdRetirementSweep {
    pub(crate) retired: Vec<Arc<dyn Worker>>,
    pub(crate) failures: Vec<PdDirectoryError>,
}

#[derive(Debug)]
pub struct PdProcessDirectory {
    topology: Option<Arc<PdTopology>>,
    topology_sha256: Option<Arc<str>>,
    state: RwLock<DirectoryState>,
}

impl Default for PdProcessDirectory {
    fn default() -> Self {
        Self::new(None)
    }
}

impl PdProcessDirectory {
    /// Construct a process directory, optionally under immutable topology ownership.
    pub fn new(topology: Option<Arc<PdTopology>>) -> Self {
        let topology_sha256 = topology
            .as_ref()
            .map(|topology| Arc::<str>::from(topology.sha256()));
        Self {
            topology,
            topology_sha256,
            state: RwLock::new(DirectoryState::default()),
        }
    }

    fn owns_pair(&self, prefill: &HttpOrigin, decoder: &HttpOrigin) -> bool {
        let Some(topology) = self.topology.as_ref() else {
            return true;
        };
        match (
            topology.group_for_origin(prefill),
            topology.group_for_origin(decoder),
        ) {
            (Some(prefill_group), Some(decoder_group)) => prefill_group.id == decoder_group.id,
            _ => false,
        }
    }

    /// All directory mutations take this lock before touching a pool. Pool code
    /// never calls back into the directory, which keeps the lock order acyclic.
    pub(super) fn admit_prefill(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        let (origin, metadata) = required_process(&worker, PdProcessRole::Prefill)?;
        if !matches!(metadata.tensor_parallel_size(), 1 | 2 | 4) {
            return Err(PdDirectoryError::UnsupportedPrefillTp(
                metadata.tensor_parallel_size(),
            ));
        }
        let bootstrap_endpoint = metadata
            .prefill_bootstrap_endpoint()
            .expect("validated prefill metadata contains an endpoint")
            .clone();
        let id = PrefillId::new(origin, metadata.launch_instance_id())?;
        let compatibility = engine_compatibility(&metadata)?;
        let pool = DecoderPool::new(id.clone(), metadata.tensor_parallel_size(), compatibility)?;

        let mut state = self.state.write();
        if state.prefills.contains_key(&id) {
            return Err(PdDirectoryError::DuplicatePrefill(id));
        }
        let compatible_decoders: Vec<(DecoderId, Arc<DecoderDirectoryEntry>)> = state
            .decoders
            .iter()
            .filter(|(_, record)| {
                record.availability == ProcessAvailability::Ready
                    && record.entry.id().url() != id.url()
                    && self.owns_pair(id.origin(), record.entry.id().origin())
                    && metadata.is_compatible_with(record.entry.metadata())
            })
            .map(|(decoder_id, record)| (decoder_id.clone(), Arc::clone(&record.entry)))
            .collect();
        for (_, decoder) in &compatible_decoders {
            pool.register(pool_metadata(decoder)?)?;
        }

        if let Some(previous_id) = state.current_prefill_by_origin.get(id.origin()).cloned() {
            if let Some(previous) = state.prefills.get_mut(&previous_id) {
                previous.entry.pool().begin_draining();
                previous.availability = ProcessAvailability::Draining;
            }
        }
        if let Some(previous_id) = state.current_decoder_by_origin.get(id.origin()).cloned() {
            drain_decoder_locked(&mut state, &previous_id)?;
        }

        let entry = Arc::new(PrefillDirectoryEntry {
            id: id.clone(),
            worker,
            metadata,
            bootstrap_endpoint,
            pool,
        });
        state
            .current_prefill_by_origin
            .insert(id.origin().clone(), id.clone());
        state.prefills.insert(
            id.clone(),
            PrefillRecord {
                entry: Arc::clone(&entry),
                availability: ProcessAvailability::Ready,
            },
        );
        for (decoder_id, _) in compatible_decoders {
            state
                .decoders
                .get_mut(&decoder_id)
                .expect("decoder was selected under the same lock")
                .pool_memberships
                .insert(id.clone());
        }
        Ok(entry)
    }

    pub(super) fn admit_decoder(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
        let (origin, metadata) = required_process(&worker, PdProcessRole::Decode)?;
        if !matches!(metadata.tensor_parallel_size(), 1 | 2) {
            return Err(PdDirectoryError::UnsupportedDecodeTp(
                metadata.tensor_parallel_size(),
            ));
        }
        let id = DecoderId::new(origin, metadata.launch_instance_id())?;
        let mut state = self.state.write();
        if state.decoders.contains_key(&id) {
            return Err(PdDirectoryError::DuplicateDecoder(id));
        }
        let entry = Arc::new(DecoderDirectoryEntry {
            id: id.clone(),
            worker,
            metadata,
        });
        let compatible_pools: Vec<(PrefillId, DecoderPool)> = state
            .prefills
            .iter()
            .filter(|(_, record)| {
                record.availability == ProcessAvailability::Ready
                    && record.entry.id().url() != id.url()
                    && self.owns_pair(record.entry.id().origin(), id.origin())
                    && record.entry.metadata().is_compatible_with(entry.metadata())
            })
            .map(|(prefill_id, record)| (prefill_id.clone(), record.entry.pool().clone()))
            .collect();
        let replica_metadata = pool_metadata(&entry)?;
        let mut installed_pools = Vec::new();
        for (prefill_id, pool) in &compatible_pools {
            if let Err(error) = pool.register_unavailable(replica_metadata.clone()) {
                rollback_unavailable_decoder(&installed_pools, &id);
                return Err(error.into());
            }
            installed_pools.push((prefill_id.clone(), pool.clone()));
        }

        let previous_id = state.current_decoder_by_origin.get(id.origin()).cloned();
        let previous_memberships = previous_id
            .as_ref()
            .and_then(|previous_id| state.decoders.get(previous_id))
            .map(|record| record.pool_memberships.clone())
            .unwrap_or_default();
        let compatible_prefills: HashSet<PrefillId> = compatible_pools
            .iter()
            .map(|(prefill_id, _)| prefill_id.clone())
            .collect();
        if let Some(previous_prefill_id) = state.current_prefill_by_origin.get(id.origin()).cloned()
        {
            let previous = state
                .prefills
                .get_mut(&previous_prefill_id)
                .expect("current prefill generation must be retained");
            previous.entry.pool().begin_draining();
            previous.availability = ProcessAvailability::Draining;
        }
        state
            .current_decoder_by_origin
            .insert(id.origin().clone(), id.clone());
        state.decoders.insert(
            id.clone(),
            DecoderRecord {
                entry: Arc::clone(&entry),
                availability: ProcessAvailability::Unavailable,
                pool_memberships: compatible_prefills.clone(),
            },
        );

        for (prefill_id, pool) in compatible_pools {
            if let Some(previous_id) = previous_id
                .as_ref()
                .filter(|_| previous_memberships.contains(&prefill_id))
            {
                pool.activate_replacement(previous_id, &id)
                    .expect("directory-installed decoder replacement must remain in its pool");
            } else {
                pool.set_availability(&id, DecoderAvailability::Ready)
                    .expect("directory-installed decoder must remain in its pool");
            }
        }
        if let Some(previous_id) = &previous_id {
            for prefill_id in previous_memberships.difference(&compatible_prefills) {
                state
                    .prefills
                    .get(prefill_id)
                    .expect("decoder membership names a retained prefill")
                    .entry
                    .pool()
                    .set_availability(previous_id, DecoderAvailability::Draining)
                    .expect("current decoder generation must remain in every recorded pool");
            }
            state
                .decoders
                .get_mut(previous_id)
                .expect("current decoder generation is retained during replacement")
                .availability = ProcessAvailability::Draining;
        }
        state
            .decoders
            .get_mut(&id)
            .expect("new decoder was inserted under the same directory lock")
            .availability = ProcessAvailability::Ready;
        Ok(entry)
    }

    pub(super) fn refresh_prefill(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        let (origin, metadata) = required_process(&worker, PdProcessRole::Prefill)?;
        let id = PrefillId::new(origin, metadata.launch_instance_id())?;
        let mut state = self.state.write();
        if state.current_prefill_by_origin.get(id.origin()) != Some(&id) {
            return Err(PdDirectoryError::ProcessNotCurrent);
        }
        let record = state
            .prefills
            .get_mut(&id)
            .ok_or_else(|| PdDirectoryError::UnknownPrefill(id.clone()))?;
        if record.availability != ProcessAvailability::Ready {
            return Err(PdDirectoryError::ProcessNotReady);
        }
        if &metadata != record.entry.metadata() {
            return Err(PdDirectoryError::GenerationMetadataChanged);
        }

        let entry = Arc::new(PrefillDirectoryEntry {
            id,
            worker,
            metadata,
            bootstrap_endpoint: record.entry.bootstrap_endpoint().clone(),
            pool: record.entry.pool().clone(),
        });
        record.entry = Arc::clone(&entry);
        Ok(entry)
    }

    pub(super) fn refresh_decoder(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
        let (origin, metadata) = required_process(&worker, PdProcessRole::Decode)?;
        let id = DecoderId::new(origin, metadata.launch_instance_id())?;
        let mut state = self.state.write();
        if state.current_decoder_by_origin.get(id.origin()) != Some(&id) {
            return Err(PdDirectoryError::ProcessNotCurrent);
        }
        let record = state
            .decoders
            .get_mut(&id)
            .ok_or_else(|| PdDirectoryError::UnknownDecoder(id.clone()))?;
        if record.availability != ProcessAvailability::Ready {
            return Err(PdDirectoryError::ProcessNotReady);
        }
        if &metadata != record.entry.metadata() {
            return Err(PdDirectoryError::GenerationMetadataChanged);
        }

        let entry = Arc::new(DecoderDirectoryEntry {
            id,
            worker,
            metadata,
        });
        record.entry = Arc::clone(&entry);
        Ok(entry)
    }

    pub(super) fn drain_prefill(&self, id: &PrefillId) -> Result<(), PdDirectoryError> {
        let mut state = self.state.write();
        let record = state
            .prefills
            .get_mut(id)
            .ok_or_else(|| PdDirectoryError::UnknownPrefill(id.clone()))?;
        record.entry.pool().begin_draining();
        record.availability = ProcessAvailability::Draining;
        Ok(())
    }

    pub(super) fn remove_drained_prefill(
        &self,
        id: &PrefillId,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        remove_drained_prefill_locked(&mut self.state.write(), id)
    }

    pub(super) fn drain_decoder(&self, id: &DecoderId) -> Result<(), PdDirectoryError> {
        drain_decoder_locked(&mut self.state.write(), id)
    }

    pub(super) fn remove_drained_decoder(
        &self,
        id: &DecoderId,
    ) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
        remove_drained_decoder_locked(&mut self.state.write(), id)
    }

    pub(crate) fn retire_drained(&self) -> PdRetirementSweep {
        let mut state = self.state.write();
        let mut prefill_ids: Vec<PrefillId> = state
            .prefills
            .iter()
            .filter(|(_, record)| record.availability == ProcessAvailability::Draining)
            .map(|(id, _)| id.clone())
            .collect();
        prefill_ids.sort();
        let mut decoder_ids: Vec<DecoderId> = state
            .decoders
            .iter()
            .filter(|(_, record)| record.availability == ProcessAvailability::Draining)
            .map(|(id, _)| id.clone())
            .collect();
        decoder_ids.sort();

        let mut retired = Vec::new();
        let mut failures = Vec::new();
        for id in prefill_ids {
            match remove_drained_prefill_locked(&mut state, &id) {
                Ok(entry) => retired.push(Arc::clone(entry.worker())),
                Err(PdDirectoryError::Pool(DecoderPoolError::PrefillPoolInUse { .. })) => {}
                Err(error) => failures.push(error),
            }
        }
        for id in decoder_ids {
            match remove_drained_decoder_locked(&mut state, &id) {
                Ok(entry) => retired.push(Arc::clone(entry.worker())),
                Err(PdDirectoryError::Pool(DecoderPoolError::DecoderInUse { .. })) => {}
                Err(error) => failures.push(error),
            }
        }
        PdRetirementSweep { retired, failures }
    }

    pub fn prefill(&self, id: &PrefillId) -> Option<Arc<PrefillDirectoryEntry>> {
        self.state
            .read()
            .prefills
            .get(id)
            .map(|record| Arc::clone(&record.entry))
    }

    pub fn begin_prefill_request(
        &self,
        id: &PrefillId,
        request_id: impl Into<String>,
    ) -> Result<(Arc<PrefillDirectoryEntry>, LogicalRequestOwner), PdDirectoryError> {
        let state = self.state.read();
        let record = state
            .prefills
            .get(id)
            .ok_or_else(|| PdDirectoryError::UnknownPrefill(id.clone()))?;
        if record.availability != ProcessAvailability::Ready {
            return Err(PdDirectoryError::ProcessNotReady);
        }
        ensure_worker_available(record.entry.worker())?;
        let owner = record.entry.pool().begin_request(request_id)?;
        Ok((Arc::clone(&record.entry), owner))
    }

    /// Select one topology group and charge its logical-request load atomically.
    pub fn begin_group_request(
        &self,
        request_id: impl Into<String>,
        model_id: Option<&str>,
    ) -> Result<PdGroupRequest, PdDirectoryError> {
        let topology = self
            .topology
            .as_ref()
            .ok_or(PdDirectoryError::TopologyNotConfigured)?;
        let request_id = request_id.into();
        let mut state = self.state.write();
        let mut candidates = Vec::new();

        for (manifest_index, group) in topology.groups.iter().enumerate() {
            let Some(prefill_id) = state.current_prefill_by_origin.get(&group.prefill.origin)
            else {
                continue;
            };
            let Some(prefill_record) = state.prefills.get(prefill_id) else {
                continue;
            };
            if prefill_record.availability != ProcessAvailability::Ready
                || !prefill_record.entry.worker().is_available()
                || model_id
                    .is_some_and(|model_id| !prefill_record.entry.worker().supports_model(model_id))
            {
                continue;
            }

            let ready_decoder_count = group
                .decoders
                .iter()
                .filter(|decoder_spec| {
                    let Some(decoder_id) =
                        state.current_decoder_by_origin.get(&decoder_spec.origin)
                    else {
                        return false;
                    };
                    state
                        .decoders
                        .get(decoder_id)
                        .is_some_and(|decoder_record| {
                            decoder_record.availability == ProcessAvailability::Ready
                                && decoder_record.pool_memberships.contains(prefill_id)
                                && decoder_record.entry.worker().is_available()
                        })
                })
                .count();
            if ready_decoder_count == 0 {
                continue;
            }

            candidates.push((
                manifest_index,
                group.id.clone(),
                Arc::clone(&prefill_record.entry),
                prefill_record
                    .entry
                    .pool()
                    .snapshot()
                    .active_logical_requests,
                ready_decoder_count,
            ));
        }
        if candidates.is_empty() {
            return Err(PdDirectoryError::NoEligibleTopologyGroup);
        }

        let mut best = Vec::new();
        for candidate in candidates {
            match best.first() {
                None => best.push(candidate),
                Some(current) => {
                    let ordering = ((candidate.3 as u128) * (current.4 as u128))
                        .cmp(&((current.3 as u128) * (candidate.4 as u128)));
                    match ordering {
                        std::cmp::Ordering::Less => {
                            best.clear();
                            best.push(candidate);
                        }
                        std::cmp::Ordering::Equal => best.push(candidate),
                        std::cmp::Ordering::Greater => {}
                    }
                }
            }
        }

        let selected_index = state.last_selected_group_index.map_or(0, |last_index| {
            best.iter()
                .position(|candidate| candidate.0 > last_index)
                .unwrap_or(0)
        });
        let (manifest_index, group_id, prefill, _, _) = best.swap_remove(selected_index);
        let owner = prefill.pool().begin_request(request_id)?;
        state.last_selected_group_index = Some(manifest_index);
        *state
            .group_selection_counts
            .entry(group_id.clone())
            .or_default() += 1;
        Ok(PdGroupRequest {
            group_id,
            prefill,
            owner,
        })
    }

    /// Snapshot the immutable topology and current generation-aware process state.
    pub fn topology_status(&self) -> Option<PdTopologyStatus> {
        let topology = self.topology.as_ref()?;
        let topology_sha256 = self
            .topology_sha256
            .as_ref()
            .expect("a configured topology has a startup-computed digest");
        let state = self.state.read();
        let mut groups = Vec::with_capacity(topology.groups.len());

        for (manifest_index, group) in topology.groups.iter().enumerate() {
            let prefill_record = state
                .current_prefill_by_origin
                .get(&group.prefill.origin)
                .and_then(|prefill_id| state.prefills.get(prefill_id));
            let prefill_generation_ready = prefill_record
                .is_some_and(|record| record.availability == ProcessAvailability::Ready);
            let prefill_healthy =
                prefill_record.is_some_and(|record| record.entry.worker().is_healthy());
            let prefill_available = prefill_record.is_some_and(|record| {
                record.availability == ProcessAvailability::Ready
                    && record.entry.worker().is_available()
            });
            let pool_snapshot = prefill_record.map(|record| record.entry.pool().snapshot());

            let decoders = group
                .decoders
                .iter()
                .map(|decoder_spec| {
                    let decoder_record = state
                        .current_decoder_by_origin
                        .get(&decoder_spec.origin)
                        .and_then(|decoder_id| state.decoders.get(decoder_id));
                    let generation_ready = decoder_record
                        .is_some_and(|record| record.availability == ProcessAvailability::Ready);
                    let healthy =
                        decoder_record.is_some_and(|record| record.entry.worker().is_healthy());
                    let replica = decoder_record.and_then(|record| {
                        pool_snapshot.as_ref().and_then(|snapshot| {
                            snapshot
                                .replicas
                                .iter()
                                .find(|replica| &replica.id == record.entry.id())
                        })
                    });
                    let available = decoder_record.is_some_and(|record| {
                        generation_ready
                            && record.entry.worker().is_available()
                            && prefill_record.is_some_and(|prefill| {
                                record.pool_memberships.contains(prefill.entry.id())
                            })
                            && replica.is_some_and(|replica| {
                                replica.availability == DecoderAvailability::Ready
                            })
                    });
                    PdTopologyDecoderStatus {
                        origin: decoder_spec.origin.clone(),
                        tensor_parallel_size: decoder_spec.tensor_parallel_size,
                        launch_instance_id: decoder_record
                            .map(|record| record.entry.metadata().launch_instance_id()),
                        generation_ready,
                        healthy,
                        available,
                        pending_admissions: replica.map_or(0, |replica| replica.pending_admissions),
                        active_child_requests: replica
                            .map_or(0, |replica| replica.active_child_requests),
                    }
                })
                .collect::<Vec<_>>();
            let ready_decoder_count = decoders.iter().filter(|decoder| decoder.available).count();
            groups.push(PdTopologyGroupStatus {
                id: group.id.clone(),
                manifest_index,
                eligible: prefill_available && ready_decoder_count > 0,
                selection_count: state
                    .group_selection_counts
                    .get(&group.id)
                    .copied()
                    .unwrap_or(0),
                outstanding_logical_requests: pool_snapshot
                    .as_ref()
                    .map_or(0, |snapshot| snapshot.active_logical_requests),
                ready_decoder_count,
                prefill: PdTopologyPrefillStatus {
                    origin: group.prefill.origin.clone(),
                    tensor_parallel_size: group.prefill.tensor_parallel_size,
                    bootstrap_endpoint: group.prefill.bootstrap_endpoint.clone(),
                    launch_instance_id: prefill_record
                        .map(|record| record.entry.metadata().launch_instance_id()),
                    generation_ready: prefill_generation_ready,
                    healthy: prefill_healthy,
                    available: prefill_available,
                },
                decoders,
            });
        }

        let fully_registered = groups.iter().all(|group| {
            group.prefill.generation_ready
                && group.prefill.healthy
                && group.prefill.available
                && group.decoders.iter().all(|decoder| decoder.available)
        });
        Some(PdTopologyStatus {
            schema: "pd-topology-status-v1".to_string(),
            topology_sha256: topology_sha256.to_string(),
            topology: (**topology).clone(),
            fully_registered,
            groups,
        })
    }

    /// Return the startup-computed immutable topology digest.
    pub fn topology_sha256(&self) -> Option<&str> {
        self.topology_sha256.as_deref()
    }

    pub(crate) fn begin_admission(
        &self,
        prefill_id: &PrefillId,
        request: &LogicalRequestOwner,
        template: &DecoderRequestTemplate,
        prepared_ttl: Duration,
        charge: PendingSchedulingCharge,
    ) -> Result<PendingAdmission, PdDirectoryError> {
        let state = self.state.read();
        let prefill = state
            .prefills
            .get(prefill_id)
            .ok_or_else(|| PdDirectoryError::UnknownPrefill(prefill_id.clone()))?;
        if !matches!(
            prefill.availability,
            ProcessAvailability::Ready | ProcessAvailability::Draining
        ) {
            return Err(PdDirectoryError::ProcessNotReady);
        }

        let eligible_decoders: HashSet<DecoderId> = state
            .decoders
            .iter()
            .filter(|(_, decoder)| {
                decoder.availability == ProcessAvailability::Ready
                    && decoder.pool_memberships.contains(prefill_id)
                    && decoder.entry.worker().is_available()
            })
            .map(|(decoder_id, _)| decoder_id.clone())
            .collect();
        prefill
            .entry
            .pool()
            .begin_admission(
                request,
                &eligible_decoders,
                template,
                prefill.entry.bootstrap_endpoint().clone(),
                prepared_ttl,
                charge,
            )
            .map_err(Into::into)
    }

    pub fn decoder(&self, id: &DecoderId) -> Option<Arc<DecoderDirectoryEntry>> {
        self.state
            .read()
            .decoders
            .get(id)
            .map(|record| Arc::clone(&record.entry))
    }

    pub fn decoder_availability(&self, id: &DecoderId) -> Option<ProcessAvailability> {
        self.state
            .read()
            .decoders
            .get(id)
            .map(|record| record.availability)
    }

    pub fn prefill_availability(&self, id: &PrefillId) -> Option<ProcessAvailability> {
        self.state
            .read()
            .prefills
            .get(id)
            .map(|record| record.availability)
    }

    pub fn ready_prefills(&self) -> Vec<Arc<PrefillDirectoryEntry>> {
        self.ready_prefills_for_model(None)
    }

    pub(crate) fn ready_prefills_for_model(
        &self,
        model_id: Option<&str>,
    ) -> Vec<Arc<PrefillDirectoryEntry>> {
        let mut entries: Vec<Arc<PrefillDirectoryEntry>> = self
            .state
            .read()
            .prefills
            .values()
            .filter(|record| {
                record.availability == ProcessAvailability::Ready
                    && record.entry.worker().is_available()
                    && model_id
                        .map(|model_id| record.entry.worker().supports_model(model_id))
                        .unwrap_or(true)
            })
            .map(|record| Arc::clone(&record.entry))
            .collect();
        entries.sort_by(|left, right| left.id().cmp(right.id()));
        entries
    }
}

fn rollback_unavailable_decoder(pools: &[(PrefillId, DecoderPool)], id: &DecoderId) {
    for (_, pool) in pools {
        pool.set_availability(id, DecoderAvailability::Draining)
            .expect("unavailable decoder was installed under directory ownership");
        pool.remove(id)
            .expect("unavailable decoder cannot own a cohort");
    }
}

fn remove_drained_prefill_locked(
    state: &mut DirectoryState,
    id: &PrefillId,
) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
    let record = state
        .prefills
        .get(id)
        .ok_or_else(|| PdDirectoryError::UnknownPrefill(id.clone()))?;
    if record.availability != ProcessAvailability::Draining {
        return Err(PdDirectoryError::ProcessNotDraining);
    }
    record.entry.pool().ensure_retirable()?;

    let record = state
        .prefills
        .remove(id)
        .expect("prefill was retained while retirement was proven");
    if state.current_prefill_by_origin.get(id.origin()) == Some(id) {
        state.current_prefill_by_origin.remove(id.origin());
    }
    for decoder in state.decoders.values_mut() {
        decoder.pool_memberships.remove(id);
    }
    Ok(record.entry)
}

fn remove_drained_decoder_locked(
    state: &mut DirectoryState,
    id: &DecoderId,
) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
    let availability = state
        .decoders
        .get(id)
        .ok_or_else(|| PdDirectoryError::UnknownDecoder(id.clone()))?
        .availability;
    if availability != ProcessAvailability::Draining {
        return Err(PdDirectoryError::ProcessNotDraining);
    }

    let memberships: Vec<PrefillId> = state
        .decoders
        .get(id)
        .expect("decoder was checked under the same lock")
        .pool_memberships
        .iter()
        .cloned()
        .collect();
    for prefill_id in memberships {
        let result = state
            .prefills
            .get(&prefill_id)
            .expect("membership names a retained prefill")
            .entry
            .pool()
            .remove(id);
        match result {
            Ok(()) | Err(DecoderPoolError::UnknownDecoder(_)) => {
                state
                    .decoders
                    .get_mut(id)
                    .expect("decoder is retained until every pool releases it")
                    .pool_memberships
                    .remove(&prefill_id);
            }
            Err(error) => return Err(error.into()),
        }
    }

    let record = state
        .decoders
        .remove(id)
        .expect("decoder was retained while memberships were released");
    if state.current_decoder_by_origin.get(id.origin()) == Some(id) {
        state.current_decoder_by_origin.remove(id.origin());
    }
    Ok(record.entry)
}

fn drain_decoder_locked(
    state: &mut DirectoryState,
    id: &DecoderId,
) -> Result<(), PdDirectoryError> {
    let memberships: Vec<PrefillId> = state
        .decoders
        .get(id)
        .ok_or_else(|| PdDirectoryError::UnknownDecoder(id.clone()))?
        .pool_memberships
        .iter()
        .cloned()
        .collect();
    for prefill_id in memberships {
        state
            .prefills
            .get(&prefill_id)
            .expect("membership names a retained prefill")
            .entry
            .pool()
            .set_availability(id, DecoderAvailability::Draining)?;
    }
    state
        .decoders
        .get_mut(id)
        .expect("decoder was checked under the same lock")
        .availability = ProcessAvailability::Draining;
    Ok(())
}

fn ensure_worker_available(worker: &Arc<dyn Worker>) -> Result<(), PdDirectoryError> {
    if worker.is_available() {
        return Ok(());
    }
    Err(PdDirectoryError::ProcessUnavailable)
}

fn required_process(
    worker: &Arc<dyn Worker>,
    role: PdProcessRole,
) -> Result<(HttpOrigin, PdProcessMetadata), PdDirectoryError> {
    let worker_role_matches = match role {
        PdProcessRole::Prefill => matches!(worker.worker_type(), WorkerType::Prefill { .. }),
        PdProcessRole::Decode => worker.worker_type() == &WorkerType::Decode,
    };
    if !worker_role_matches {
        return Err(PdDirectoryError::WorkerRoleMismatch);
    }
    let registration: &PdProcessRegistration = worker
        .metadata()
        .pd_process
        .as_ref()
        .ok_or(PdDirectoryError::MissingProcessMetadata)?;
    if worker.connection_mode() != &ConnectionMode::Http {
        return Err(PdDirectoryError::UnsupportedProcessTransport);
    }
    if worker.url() != registration.origin().as_str()
        || worker.base_url() != registration.origin().as_str()
    {
        return Err(PdDirectoryError::ProcessOriginMismatch);
    }
    let metadata = registration.metadata();
    if metadata.role() != role {
        return Err(PdDirectoryError::WorkerRoleMismatch);
    }
    Ok((registration.origin().clone(), metadata.clone()))
}

fn engine_compatibility(
    metadata: &PdProcessMetadata,
) -> Result<EngineCompatibilityMetadata, DecoderPoolError> {
    EngineCompatibilityMetadata::new(
        metadata.model_fingerprint(),
        metadata.logical_kv_layout_fingerprint(),
        metadata.kv_dtype(),
        metadata.kv_transfer_protocol().as_str(),
        metadata.prepared_grant_protocol().as_str(),
        metadata.page_size(),
    )
}

fn pool_metadata(
    decoder: &DecoderDirectoryEntry,
) -> Result<DecoderReplicaMetadata, DecoderPoolError> {
    DecoderReplicaMetadata::new(
        decoder.id().clone(),
        decoder.metadata().tensor_parallel_size(),
        engine_compatibility(decoder.metadata())?,
        DecoderSchedulingHints::new(1, 1, 1)?,
    )
}

#[derive(Debug, Error)]
pub enum PdDirectoryError {
    #[error("PD worker is missing typed process metadata")]
    MissingProcessMetadata,
    #[error("worker role and typed PD role differ")]
    WorkerRoleMismatch,
    #[error("PD process registration requires canonical HTTP transport")]
    UnsupportedProcessTransport,
    #[error("PD worker transport and registered process origin differ")]
    ProcessOriginMismatch,
    #[error("prefill directory supports TP1, TP2, or TP4, received TP{0}")]
    UnsupportedPrefillTp(usize),
    #[error("decoder directory supports TP1 or TP2, received TP{0}")]
    UnsupportedDecodeTp(usize),
    #[error("prefill generation {0} is already admitted")]
    DuplicatePrefill(PrefillId),
    #[error("decoder generation {0} is already admitted")]
    DuplicateDecoder(DecoderId),
    #[error("prefill generation {0} is unknown")]
    UnknownPrefill(PrefillId),
    #[error("decoder generation {0} is unknown")]
    UnknownDecoder(DecoderId),
    #[error("process generation is not draining")]
    ProcessNotDraining,
    #[error("process generation is not ready for new admissions")]
    ProcessNotReady,
    #[error("process generation is not healthy and circuit-breaker available")]
    ProcessUnavailable,
    #[error("process generation is not the current generation for its URL")]
    ProcessNotCurrent,
    #[error("process-generation metadata changed without a new launch identity")]
    GenerationMetadataChanged,
    #[error("strict PD topology is not configured")]
    TopologyNotConfigured,
    #[error("no PD topology group has a ready prefill and owned decoder capacity")]
    NoEligibleTopologyGroup,
    #[error(transparent)]
    InvalidProcessIdentity(#[from] ProcessIdentityError),
    #[error(transparent)]
    Pool(#[from] DecoderPoolError),
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
        time::Duration,
    };

    use bytes::Bytes;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::core::{
        pd_decoder_grant::{
            issue_test_grant, issue_test_reserve_refusal_receipt, DecoderGrantChildAccounting,
            DecoderInferenceRoute, DecoderRequestTemplate, DecoderReserveRefusalDisposition,
            DecoderSlotGeneration,
        },
        pd_decoder_pool::{
            begin_test_pending_for_grant, PendingAdmission, PendingAdmissionDisposition,
            PendingSchedulingCharge,
        },
        BasicWorkerBuilder, KvTransferProtocol, PdMetadataSchema, PreparedGrantProtocol,
    };

    const MODEL_FINGERPRINT: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const KV_LAYOUT_FINGERPRINT: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn instance(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn metadata(
        role: PdProcessRole,
        tp_size: usize,
        instance_id: Uuid,
        model_fingerprint: &str,
        kv_layout_fingerprint: &str,
    ) -> PdProcessMetadata {
        PdProcessMetadata::new(
            PdMetadataSchema::V1,
            instance_id,
            role,
            tp_size,
            1,
            model_fingerprint,
            kv_layout_fingerprint,
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

    fn worker(
        url: &str,
        role: PdProcessRole,
        tp_size: usize,
        instance_id: Uuid,
    ) -> Arc<dyn Worker> {
        worker_with_compatibility(
            url,
            role,
            tp_size,
            instance_id,
            MODEL_FINGERPRINT,
            KV_LAYOUT_FINGERPRINT,
        )
    }

    fn worker_with_compatibility(
        url: &str,
        role: PdProcessRole,
        tp_size: usize,
        instance_id: Uuid,
        model_fingerprint: &str,
        kv_layout_fingerprint: &str,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(match role {
                    PdProcessRole::Prefill => WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                    PdProcessRole::Decode => WorkerType::Decode,
                })
                .pd_process(PdProcessRegistration::new(
                    HttpOrigin::parse(url).unwrap(),
                    metadata(
                        role,
                        tp_size,
                        instance_id,
                        model_fingerprint,
                        kv_layout_fingerprint,
                    ),
                ))
                .build(),
        )
    }

    fn scalar_template() -> DecoderRequestTemplate {
        DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            Bytes::from_static(br#"{"text":"test"}"#),
        )
        .unwrap()
    }

    fn begin_scalar_admission(
        directory: &PdProcessDirectory,
        prefill: &PrefillDirectoryEntry,
        owner: &LogicalRequestOwner,
    ) -> Result<PendingAdmission, PdDirectoryError> {
        directory.begin_admission(
            prefill.id(),
            owner,
            &scalar_template(),
            Duration::from_secs(2),
            PendingSchedulingCharge::new(1, 64, 16).unwrap(),
        )
    }

    fn strict_topology() -> PdTopology {
        PdTopology::from_json(
            &json!({
                "schema": "pd-topology-v1",
                "groups": [
                    {
                        "id": "group-0",
                        "prefill": {
                            "origin": "http://prefill-a.test:30000",
                            "tensor_parallel_size": 2,
                            "bootstrap_endpoint": {
                                "host": "prefill-transfer.test",
                                "port": 50_051
                            }
                        },
                        "decoders": [
                            {
                                "origin": "http://decode-a.test:30001",
                                "tensor_parallel_size": 1
                            }
                        ]
                    },
                    {
                        "id": "group-1",
                        "prefill": {
                            "origin": "http://prefill-b.test:30000",
                            "tensor_parallel_size": 2,
                            "bootstrap_endpoint": {
                                "host": "prefill-transfer.test",
                                "port": 50_051
                            }
                        },
                        "decoders": [
                            {
                                "origin": "http://decode-b0.test:30001",
                                "tensor_parallel_size": 1
                            },
                            {
                                "origin": "http://decode-b1.test:30001",
                                "tensor_parallel_size": 1
                            },
                            {
                                "origin": "http://decode-b2.test:30001",
                                "tensor_parallel_size": 1
                            }
                        ]
                    }
                ]
            })
            .to_string(),
        )
        .unwrap()
    }

    fn fully_populated_topology_directory() -> (PdProcessDirectory, Vec<Arc<dyn Worker>>) {
        let directory = PdProcessDirectory::new(Some(Arc::new(strict_topology())));
        let prefills = vec![
            worker(
                "http://prefill-a.test:30000",
                PdProcessRole::Prefill,
                2,
                instance(100),
            ),
            worker(
                "http://prefill-b.test:30000",
                PdProcessRole::Prefill,
                2,
                instance(200),
            ),
        ];
        let decoders = vec![
            worker(
                "http://decode-a.test:30001",
                PdProcessRole::Decode,
                1,
                instance(101),
            ),
            worker(
                "http://decode-b0.test:30001",
                PdProcessRole::Decode,
                1,
                instance(201),
            ),
            worker(
                "http://decode-b1.test:30001",
                PdProcessRole::Decode,
                1,
                instance(202),
            ),
            worker(
                "http://decode-b2.test:30001",
                PdProcessRole::Decode,
                1,
                instance(203),
            ),
        ];
        for prefill in &prefills {
            directory.admit_prefill(Arc::clone(prefill)).unwrap();
        }
        for decoder in &decoders {
            directory.admit_decoder(Arc::clone(decoder)).unwrap();
        }
        (directory, prefills)
    }

    #[test]
    fn rejects_non_http_pd_transport_before_directory_mutation() {
        let url = "http://decode.test:30001";
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc { port: Some(50_051) })
                .pd_process(PdProcessRegistration::new(
                    HttpOrigin::parse(url).unwrap(),
                    metadata(
                        PdProcessRole::Decode,
                        1,
                        instance(1),
                        MODEL_FINGERPRINT,
                        KV_LAYOUT_FINGERPRINT,
                    ),
                ))
                .build(),
        );
        let directory = PdProcessDirectory::default();

        assert!(matches!(
            directory.admit_decoder(worker),
            Err(PdDirectoryError::UnsupportedProcessTransport)
        ));
        assert!(directory.ready_prefills().is_empty());
    }

    #[test]
    fn topology_ownership_is_static_in_both_registration_orders() {
        for prefill_first in [false, true] {
            let directory = PdProcessDirectory::new(Some(Arc::new(strict_topology())));
            let prefills = [
                worker(
                    "http://prefill-a.test:30000",
                    PdProcessRole::Prefill,
                    2,
                    instance(100),
                ),
                worker(
                    "http://prefill-b.test:30000",
                    PdProcessRole::Prefill,
                    2,
                    instance(200),
                ),
            ];
            let decoders = [
                worker(
                    "http://decode-a.test:30001",
                    PdProcessRole::Decode,
                    1,
                    instance(101),
                ),
                worker(
                    "http://decode-b0.test:30001",
                    PdProcessRole::Decode,
                    1,
                    instance(201),
                ),
                worker(
                    "http://decode-b1.test:30001",
                    PdProcessRole::Decode,
                    1,
                    instance(202),
                ),
                worker(
                    "http://decode-b2.test:30001",
                    PdProcessRole::Decode,
                    1,
                    instance(203),
                ),
            ];
            if prefill_first {
                for prefill in &prefills {
                    directory.admit_prefill(Arc::clone(prefill)).unwrap();
                }
            }
            for decoder in &decoders {
                directory.admit_decoder(Arc::clone(decoder)).unwrap();
            }
            if !prefill_first {
                for prefill in &prefills {
                    directory.admit_prefill(Arc::clone(prefill)).unwrap();
                }
            }

            let ready = directory.ready_prefills();
            assert_eq!(ready.len(), 2);
            let first = ready
                .iter()
                .find(|prefill| prefill.id().url() == "http://prefill-a.test:30000")
                .unwrap();
            let second = ready
                .iter()
                .find(|prefill| prefill.id().url() == "http://prefill-b.test:30000")
                .unwrap();
            assert_eq!(first.pool().snapshot().replicas.len(), 1);
            assert_eq!(
                first.pool().snapshot().replicas[0].id.url(),
                "http://decode-a.test:30001"
            );
            let second_origins = second
                .pool()
                .snapshot()
                .replicas
                .into_iter()
                .map(|replica| replica.id.url().to_string())
                .collect::<HashSet<_>>();
            assert_eq!(
                second_origins,
                [
                    "http://decode-b0.test:30001".to_string(),
                    "http://decode-b1.test:30001".to_string(),
                    "http://decode-b2.test:30001".to_string(),
                ]
                .into_iter()
                .collect()
            );
            assert!(directory.topology_status().unwrap().fully_registered);
        }
    }

    #[test]
    fn concurrent_group_charging_is_atomic_and_capacity_normalized() {
        let (directory, prefills) = fully_populated_topology_directory();
        let directory = Arc::new(directory);
        let barrier = Arc::new(Barrier::new(41));
        let mut handles = Vec::new();
        for index in 0..40 {
            let directory = Arc::clone(&directory);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                directory
                    .begin_group_request(format!("request-{index}"), None)
                    .unwrap()
            }));
        }
        barrier.wait();
        let requests = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        let status = directory.topology_status().unwrap();
        assert_eq!(status.groups[0].selection_count, 10);
        assert_eq!(status.groups[0].outstanding_logical_requests, 10);
        assert_eq!(status.groups[1].selection_count, 30);
        assert_eq!(status.groups[1].outstanding_logical_requests, 30);
        assert_eq!(
            status.groups[0].prefill.launch_instance_id,
            Some(instance(100))
        );
        assert_eq!(status.groups[1].ready_decoder_count, 3);

        for request in requests {
            let (_, prefill, mut owner) = request.into_parts();
            prefill.pool().finalize_request(&mut owner).unwrap();
        }
        assert!(directory
            .topology_status()
            .unwrap()
            .groups
            .iter()
            .all(|group| group.outstanding_logical_requests == 0));

        prefills[0].set_healthy(false);
        let status = directory.topology_status().unwrap();
        assert!(!status.fully_registered);
        assert!(!status.groups[0].prefill.available);
    }

    #[test]
    fn retry_another_decoder_cannot_escape_the_selected_group() {
        let (directory, _) = fully_populated_topology_directory();
        let request = directory
            .begin_group_request("bounded-retry", None)
            .unwrap();
        assert_eq!(request.group_id().as_str(), "group-0");
        let (_, prefill, mut owner) = request.into_parts();
        let mut pending = begin_scalar_admission(&directory, &prefill, &owner).unwrap();
        assert_eq!(pending.decoder_id().url(), "http://decode-a.test:30001");
        let snapshot = prefill.pool().snapshot();
        let refusal = issue_test_reserve_refusal_receipt(
            snapshot.prefill_id,
            pending.decoder_id().clone(),
            owner.chain_id(),
            pending.reservation_attempt_id(),
            pending.reserve_attempt_digest(),
            DecoderReserveRefusalDisposition::RetryAnotherDecoder,
            true,
        );
        assert_eq!(
            prefill
                .pool()
                .install_reserve_refusal_proof(&mut pending, &refusal)
                .unwrap(),
            PendingAdmissionDisposition::RetryAnotherDecoder
        );
        assert!(matches!(
            begin_scalar_admission(&directory, &prefill, &owner),
            Err(PdDirectoryError::Pool(
                DecoderPoolError::RetryAlternativesExhausted
            ))
        ));
        prefill.pool().finalize_request(&mut owner).unwrap();
    }

    #[test]
    fn rejects_pd_transport_origin_divergence_before_directory_mutation() {
        let registered_origin = HttpOrigin::parse("http://decode.test:30001").unwrap();
        let mut worker = BasicWorkerBuilder::new(registered_origin.as_str())
            .worker_type(WorkerType::Decode)
            .pd_process(PdProcessRegistration::new(
                registered_origin.clone(),
                metadata(
                    PdProcessRole::Decode,
                    1,
                    instance(1),
                    MODEL_FINGERPRINT,
                    KV_LAYOUT_FINGERPRINT,
                ),
            ))
            .build();
        worker.metadata.url = "http://other-decode.test:30001".to_string();
        let directory = PdProcessDirectory::default();

        assert!(matches!(
            directory.admit_decoder(Arc::new(worker)),
            Err(PdDirectoryError::ProcessOriginMismatch)
        ));
        assert!(directory
            .decoder(&DecoderId::new(registered_origin, instance(1)).unwrap())
            .is_none());
    }

    #[test]
    fn seeds_prefills_from_supported_tp_and_arbitrary_current_tp1_replica_counts() {
        for prefill_tp in [1, 2, 4] {
            for replica_count in [1, 2, 3, 5] {
                let directory = PdProcessDirectory::default();
                let mut decoders = Vec::new();
                for index in 0..replica_count {
                    decoders.push(
                        directory
                            .admit_decoder(worker(
                                &format!("http://decode-{index}.test:30001"),
                                PdProcessRole::Decode,
                                1,
                                instance(100 + index as u128),
                            ))
                            .unwrap(),
                    );
                }
                if prefill_tp % 2 == 0 {
                    decoders.push(
                        directory
                            .admit_decoder(worker(
                                "http://decode-tp2.test:30001",
                                PdProcessRole::Decode,
                                2,
                                instance(200),
                            ))
                            .unwrap(),
                    );
                }

                let prefill = directory
                    .admit_prefill(worker(
                        &format!("http://prefill-tp{prefill_tp}.test:30000"),
                        PdProcessRole::Prefill,
                        prefill_tp,
                        instance(1_000 + prefill_tp as u128),
                    ))
                    .unwrap();

                assert_eq!(prefill.pool().snapshot().replicas.len(), decoders.len());
                assert_eq!(prefill.bootstrap_endpoint().host(), "prefill-transfer.test");
                for decoder in decoders {
                    assert!(prefill
                        .pool()
                        .snapshot()
                        .replicas
                        .iter()
                        .any(|replica| &replica.id == decoder.id()));
                }
            }
        }
    }

    #[test]
    fn new_decoders_seed_every_compatible_ready_prefill_pool() {
        let directory = PdProcessDirectory::default();
        let prefill_tp2 = directory
            .admit_prefill(worker(
                "http://prefill-a.test:30000",
                PdProcessRole::Prefill,
                2,
                instance(1),
            ))
            .unwrap();
        let prefill_tp4 = directory
            .admit_prefill(worker(
                "http://prefill-b.test:30000",
                PdProcessRole::Prefill,
                4,
                instance(2),
            ))
            .unwrap();

        for index in 0..5 {
            directory
                .admit_decoder(worker(
                    &format!("http://decode-{index}.test:30001"),
                    PdProcessRole::Decode,
                    if index == 4 { 2 } else { 1 },
                    instance(10 + index),
                ))
                .unwrap();
        }

        assert_eq!(prefill_tp2.pool().snapshot().replicas.len(), 5);
        assert_eq!(prefill_tp4.pool().snapshot().replicas.len(), 5);
    }

    #[test]
    fn compatibility_is_semantic_and_independent_of_physical_tp_partition() {
        let directory = PdProcessDirectory::default();
        let prefill = directory
            .admit_prefill(worker(
                "http://prefill.test:30000",
                PdProcessRole::Prefill,
                4,
                instance(1),
            ))
            .unwrap();
        directory
            .admit_decoder(worker(
                "http://decode-good.test:30001",
                PdProcessRole::Decode,
                1,
                instance(2),
            ))
            .unwrap();
        directory
            .admit_decoder(worker_with_compatibility(
                "http://decode-model.test:30001",
                PdProcessRole::Decode,
                1,
                instance(3),
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                KV_LAYOUT_FINGERPRINT,
            ))
            .unwrap();
        directory
            .admit_decoder(worker_with_compatibility(
                "http://decode-layout.test:30001",
                PdProcessRole::Decode,
                1,
                instance(4),
                MODEL_FINGERPRINT,
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            ))
            .unwrap();

        let replicas = prefill.pool().snapshot().replicas;
        assert_eq!(replicas.len(), 1);
        assert_eq!(replicas[0].id.url(), "http://decode-good.test:30001");
    }

    #[test]
    fn admission_boundary_excludes_unavailable_processes() {
        let directory = PdProcessDirectory::default();
        let prefill_worker = worker(
            "http://prefill-health.test:30000",
            PdProcessRole::Prefill,
            2,
            instance(1),
        );
        let prefill = directory
            .admit_prefill(Arc::clone(&prefill_worker))
            .unwrap();
        let unavailable_worker = worker(
            "http://decode-unavailable.test:30001",
            PdProcessRole::Decode,
            1,
            instance(2),
        );
        let unavailable = directory
            .admit_decoder(Arc::clone(&unavailable_worker))
            .unwrap();
        let ready_worker = worker(
            "http://decode-ready.test:30001",
            PdProcessRole::Decode,
            1,
            instance(3),
        );
        let ready = directory.admit_decoder(Arc::clone(&ready_worker)).unwrap();

        unavailable_worker.set_healthy(false);
        let (_, mut owner) = directory
            .begin_prefill_request(prefill.id(), "healthy-selection")
            .unwrap();
        let pending = begin_scalar_admission(&directory, &prefill, &owner).unwrap();
        assert_eq!(pending.decoder_id(), ready.id());
        assert_ne!(pending.decoder_id(), unavailable.id());
        drop(pending);
        prefill.pool().finalize_request(&mut owner).unwrap();

        prefill_worker.set_healthy(false);
        assert!(directory.ready_prefills().is_empty());
        assert!(matches!(
            directory.begin_prefill_request(prefill.id(), "unavailable-prefill"),
            Err(PdDirectoryError::ProcessUnavailable)
        ));

        prefill_worker.set_healthy(true);
        ready_worker.set_healthy(false);
        let (_, mut owner) = directory
            .begin_prefill_request(prefill.id(), "no-ready-decode")
            .unwrap();
        assert!(matches!(
            begin_scalar_admission(&directory, &prefill, &owner),
            Err(PdDirectoryError::Pool(DecoderPoolError::NoReadyDecoder))
        ));
        prefill.pool().finalize_request(&mut owner).unwrap();
    }

    #[test]
    fn same_url_decoder_replacement_drains_old_in_every_pool_and_preserves_arcs() {
        let directory = PdProcessDirectory::default();
        let prefills = [
            directory
                .admit_prefill(worker(
                    "http://prefill-a.test:30000",
                    PdProcessRole::Prefill,
                    2,
                    instance(1),
                ))
                .unwrap(),
            directory
                .admit_prefill(worker(
                    "http://prefill-b.test:30000",
                    PdProcessRole::Prefill,
                    4,
                    instance(2),
                ))
                .unwrap(),
        ];
        let old_worker = worker(
            "http://decode.test:30001",
            PdProcessRole::Decode,
            1,
            instance(10),
        );
        let old = directory.admit_decoder(Arc::clone(&old_worker)).unwrap();
        let replacement_worker = worker(
            "http://decode.test:30001",
            PdProcessRole::Decode,
            1,
            instance(11),
        );
        let replacement = directory
            .admit_decoder(Arc::clone(&replacement_worker))
            .unwrap();

        assert_eq!(
            directory.decoder_availability(old.id()),
            Some(ProcessAvailability::Draining)
        );
        assert_eq!(
            directory.decoder_availability(replacement.id()),
            Some(ProcessAvailability::Ready)
        );
        assert!(Arc::ptr_eq(old.worker(), &old_worker));
        assert!(Arc::ptr_eq(replacement.worker(), &replacement_worker));
        assert!(Arc::ptr_eq(
            directory.decoder(old.id()).unwrap().worker(),
            &old_worker
        ));
        for prefill in &prefills {
            let snapshot = prefill.pool().snapshot();
            assert_eq!(snapshot.replicas.len(), 2);
            assert_eq!(
                snapshot
                    .replicas
                    .iter()
                    .find(|replica| replica.id == *old.id())
                    .unwrap()
                    .availability,
                DecoderAvailability::Draining
            );
            let (_, mut owner) = directory
                .begin_prefill_request(prefill.id(), "replacement-visible")
                .unwrap();
            let pending = begin_scalar_admission(&directory, prefill, &owner).unwrap();
            assert_eq!(pending.decoder_id(), replacement.id());
            drop(pending);
            prefill.pool().finalize_request(&mut owner).unwrap();
        }

        let removed = directory.remove_drained_decoder(old.id()).unwrap();
        assert!(Arc::ptr_eq(removed.worker(), &old_worker));
        assert!(directory.decoder(old.id()).is_none());
        for prefill in &prefills {
            assert_eq!(prefill.pool().snapshot().replicas.len(), 1);
        }
    }

    #[test]
    fn same_url_prefill_replacement_retains_old_generation_and_exact_worker_arc() {
        let directory = PdProcessDirectory::default();
        let old_worker = worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            2,
            instance(1),
        );
        let old = directory.admit_prefill(Arc::clone(&old_worker)).unwrap();
        let replacement_worker = worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            4,
            instance(2),
        );
        let replacement = directory
            .admit_prefill(Arc::clone(&replacement_worker))
            .unwrap();

        assert!(Arc::ptr_eq(
            directory.prefill(old.id()).unwrap().worker(),
            &old_worker
        ));
        assert!(Arc::ptr_eq(replacement.worker(), &replacement_worker));
        let ready = directory.ready_prefills();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id(), replacement.id());
        assert_eq!(
            directory.prefill_availability(old.id()),
            Some(ProcessAvailability::Draining)
        );
        assert!(matches!(
            old.pool().begin_request("stale-selection"),
            Err(DecoderPoolError::PrefillPoolDraining)
        ));
        assert!(matches!(
            directory.begin_prefill_request(old.id(), "stale-directory-selection"),
            Err(PdDirectoryError::ProcessNotReady)
        ));
        let removed = directory.remove_drained_prefill(old.id()).unwrap();
        assert!(Arc::ptr_eq(removed.worker(), &old_worker));
        assert!(directory.prefill(old.id()).is_none());
    }

    #[test]
    fn drain_rejects_new_owners_but_allows_owned_request_retry_and_reconciliation() {
        let directory = PdProcessDirectory::default();
        let prefill_worker = worker(
            "http://prefill.test:30000",
            PdProcessRole::Prefill,
            4,
            instance(1),
        );
        let prefill = directory
            .admit_prefill(Arc::clone(&prefill_worker))
            .unwrap();
        let decoder = directory
            .admit_decoder(worker(
                "http://decode.test:30001",
                PdProcessRole::Decode,
                1,
                instance(2),
            ))
            .unwrap();
        let (_, mut pending_owner) = directory
            .begin_prefill_request(prefill.id(), "pending-request")
            .unwrap();
        let mut pending = begin_scalar_admission(&directory, &prefill, &pending_owner).unwrap();
        let (_, mut late_owner) = directory
            .begin_prefill_request(prefill.id(), "late-request")
            .unwrap();

        directory.drain_prefill(prefill.id()).unwrap();
        assert!(matches!(
            prefill.pool().begin_request("too-late"),
            Err(DecoderPoolError::PrefillPoolDraining)
        ));
        let retry_pending = begin_scalar_admission(&directory, &prefill, &late_owner).unwrap();
        drop(retry_pending);
        prefill.pool().finalize_request(&mut late_owner).unwrap();

        directory.drain_decoder(decoder.id()).unwrap();
        assert!(matches!(
            directory.remove_drained_prefill(prefill.id()),
            Err(PdDirectoryError::Pool(
                DecoderPoolError::PrefillPoolInUse { .. }
            ))
        ));
        assert!(matches!(
            directory.remove_drained_decoder(decoder.id()),
            Err(PdDirectoryError::Pool(DecoderPoolError::DecoderInUse {
                active_cohorts: 0,
                pending_admissions: 1,
                ..
            }))
        ));

        let snapshot = prefill.pool().snapshot();
        let receipt = issue_test_reserve_refusal_receipt(
            snapshot.prefill_id,
            pending.decoder_id().clone(),
            pending_owner.chain_id(),
            pending.reservation_attempt_id(),
            pending.reserve_attempt_digest(),
            DecoderReserveRefusalDisposition::Terminal,
            true,
        );
        assert_eq!(
            prefill
                .pool()
                .install_reserve_refusal_proof(&mut pending, &receipt)
                .unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        assert_eq!(prefill.pool().snapshot().replicas[0].pending_admissions, 0);
        prefill.pool().finalize_request(&mut pending_owner).unwrap();

        let removed = directory.remove_drained_prefill(prefill.id()).unwrap();
        assert!(Arc::ptr_eq(removed.worker(), &prefill_worker));
        assert!(directory.prefill(prefill.id()).is_none());
        directory.remove_drained_decoder(decoder.id()).unwrap();
    }

    #[test]
    fn active_assignment_blocks_prefill_and_decoder_retirement() {
        let directory = PdProcessDirectory::default();
        let prefill = directory
            .admit_prefill(worker(
                "http://prefill-active.test:30000",
                PdProcessRole::Prefill,
                4,
                instance(20),
            ))
            .unwrap();
        let decoder = directory
            .admit_decoder(worker(
                "http://decode-active.test:30001",
                PdProcessRole::Decode,
                1,
                instance(21),
            ))
            .unwrap();
        let (_, owner) = directory
            .begin_prefill_request(prefill.id(), "active-request")
            .unwrap();
        let mut grant = issue_test_grant(
            prefill.id().clone(),
            owner.chain_id(),
            4,
            decoder.id().clone(),
            instance(22),
            vec![DecoderSlotGeneration::new(instance(23))],
            vec![1],
            vec![DecoderGrantChildAccounting::new(64, 16)],
        )
        .unwrap();
        let mut pending = begin_test_pending_for_grant(prefill.pool(), &owner, &grant).unwrap();
        let _cohort = prefill.pool().bind_grant(&mut pending, &mut grant).unwrap();

        directory.drain_prefill(prefill.id()).unwrap();
        directory.drain_decoder(decoder.id()).unwrap();
        assert!(matches!(
            directory.remove_drained_prefill(prefill.id()),
            Err(PdDirectoryError::Pool(
                DecoderPoolError::PrefillPoolInUse { .. }
            ))
        ));
        assert!(matches!(
            directory.remove_drained_decoder(decoder.id()),
            Err(PdDirectoryError::Pool(DecoderPoolError::DecoderInUse {
                active_cohorts: 1,
                pending_admissions: 0,
                ..
            }))
        ));
    }

    #[test]
    fn prefill_replacement_serializes_against_atomic_admission() {
        for iteration in 0..64 {
            let directory = Arc::new(PdProcessDirectory::default());
            let old = directory
                .admit_prefill(worker(
                    "http://prefill-race.test:30000",
                    PdProcessRole::Prefill,
                    2,
                    instance(iteration * 2 + 1),
                ))
                .unwrap();
            directory
                .admit_decoder(worker(
                    &format!("http://decode-race-{iteration}.test:30001"),
                    PdProcessRole::Decode,
                    1,
                    instance(10_000 + iteration),
                ))
                .unwrap();
            let (_, mut owner) = directory
                .begin_prefill_request(old.id(), format!("request-{iteration}"))
                .unwrap();
            let barrier = Arc::new(Barrier::new(2));

            let admission_directory = Arc::clone(&directory);
            let admission_barrier = Arc::clone(&barrier);
            let admission_prefill = Arc::clone(&old);
            let admission = thread::spawn(move || {
                admission_barrier.wait();
                match begin_scalar_admission(&admission_directory, &admission_prefill, &owner) {
                    Ok(pending) => {
                        drop(pending);
                    }
                    Err(PdDirectoryError::ProcessNotReady) => {}
                    Err(error) => panic!("unexpected admission result: {error}"),
                }
                admission_prefill
                    .pool()
                    .finalize_request(&mut owner)
                    .unwrap();
            });

            let replacement_directory = Arc::clone(&directory);
            let replacement_barrier = Arc::clone(&barrier);
            let replacement = thread::spawn(move || {
                replacement_barrier.wait();
                replacement_directory
                    .admit_prefill(worker(
                        "http://prefill-race.test:30000",
                        PdProcessRole::Prefill,
                        4,
                        instance(iteration * 2 + 2),
                    ))
                    .unwrap()
            });

            admission.join().unwrap();
            let replacement = replacement.join().unwrap();
            assert_eq!(
                directory.prefill_availability(old.id()),
                Some(ProcessAvailability::Draining)
            );
            assert_eq!(directory.ready_prefills()[0].id(), replacement.id());
            assert!(matches!(
                old.pool().begin_request("after-replacement"),
                Err(DecoderPoolError::PrefillPoolDraining)
            ));
        }
    }

    #[test]
    fn directory_requires_typed_metadata_and_role_consistency() {
        let directory = PdProcessDirectory::default();
        let missing: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill.test:30000")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        );
        assert!(matches!(
            directory.admit_prefill(missing),
            Err(PdDirectoryError::MissingProcessMetadata)
        ));

        let wrong_role: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://decode.test:30001")
                .worker_type(WorkerType::Decode)
                .pd_process(PdProcessRegistration::new(
                    HttpOrigin::parse("http://decode.test:30001").unwrap(),
                    metadata(
                        PdProcessRole::Prefill,
                        2,
                        instance(8),
                        MODEL_FINGERPRINT,
                        KV_LAYOUT_FINGERPRINT,
                    ),
                ))
                .build(),
        );
        assert!(matches!(
            directory.admit_decoder(wrong_role),
            Err(PdDirectoryError::WorkerRoleMismatch)
        ));

        assert!(matches!(
            directory.admit_decoder(worker(
                "http://decode-tp3.test:30001",
                PdProcessRole::Decode,
                3,
                instance(9),
            )),
            Err(PdDirectoryError::UnsupportedDecodeTp(3))
        ));
    }
}
