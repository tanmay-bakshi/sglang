use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use parking_lot::RwLock;
use thiserror::Error;

use super::{
    pd_decoder_grant::{DecoderId, PrefillId, ProcessIdentityError},
    pd_decoder_pool::{
        DecoderAvailability, DecoderPool, DecoderPoolError, DecoderReplicaMetadata,
        DecoderSchedulingHints, EngineCompatibilityMetadata, LogicalRequestOwner,
    },
};
use crate::core::{PdProcessMetadata, PdProcessRole, PrefillBootstrapEndpoint, Worker, WorkerType};

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

    pub fn pool(&self) -> &DecoderPool {
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
    current_prefill_by_url: HashMap<String, PrefillId>,
    decoders: HashMap<DecoderId, DecoderRecord>,
    current_decoder_by_url: HashMap<String, DecoderId>,
}

#[derive(Debug)]
pub(crate) struct PdRetirementSweep {
    pub(crate) retired: Vec<Arc<dyn Worker>>,
    pub(crate) failures: Vec<PdDirectoryError>,
}

#[derive(Debug, Default)]
pub struct PdProcessDirectory {
    state: RwLock<DirectoryState>,
}

impl PdProcessDirectory {
    /// All directory mutations take this lock before touching a pool. Pool code
    /// never calls back into the directory, which keeps the lock order acyclic.
    pub fn admit_prefill(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        let metadata = required_metadata(&worker, PdProcessRole::Prefill)?.clone();
        if !matches!(metadata.tensor_parallel_size(), 2 | 4) {
            return Err(PdDirectoryError::UnsupportedPrefillTp(
                metadata.tensor_parallel_size(),
            ));
        }
        let bootstrap_endpoint = metadata
            .prefill_bootstrap_endpoint()
            .expect("validated prefill metadata contains an endpoint")
            .clone();
        let id = PrefillId::new(worker.base_url(), metadata.launch_instance_id())?;
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
                    && metadata.is_compatible_with(record.entry.metadata())
            })
            .map(|(decoder_id, record)| (decoder_id.clone(), Arc::clone(&record.entry)))
            .collect();
        for (_, decoder) in &compatible_decoders {
            pool.register(pool_metadata(decoder)?)?;
        }

        if let Some(previous_id) = state.current_prefill_by_url.get(id.url()).cloned() {
            if let Some(previous) = state.prefills.get_mut(&previous_id) {
                previous.entry.pool().begin_draining();
                previous.availability = ProcessAvailability::Draining;
            }
        }
        if let Some(previous_id) = state.current_decoder_by_url.get(id.url()).cloned() {
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
            .current_prefill_by_url
            .insert(id.url().to_string(), id.clone());
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

    pub fn admit_decoder(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
        let metadata = required_metadata(&worker, PdProcessRole::Decode)?.clone();
        if metadata.tensor_parallel_size() != 1 {
            return Err(PdDirectoryError::UnsupportedDecodeTp(
                metadata.tensor_parallel_size(),
            ));
        }
        let id = DecoderId::new(worker.base_url(), metadata.launch_instance_id())?;
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

        let previous_id = state.current_decoder_by_url.get(id.url()).cloned();
        let previous_memberships = previous_id
            .as_ref()
            .and_then(|previous_id| state.decoders.get(previous_id))
            .map(|record| record.pool_memberships.clone())
            .unwrap_or_default();
        let compatible_prefills: HashSet<PrefillId> = compatible_pools
            .iter()
            .map(|(prefill_id, _)| prefill_id.clone())
            .collect();
        if let Some(previous_prefill_id) = state.current_prefill_by_url.get(id.url()).cloned() {
            let previous = state
                .prefills
                .get_mut(&previous_prefill_id)
                .expect("current prefill generation must be retained");
            previous.entry.pool().begin_draining();
            previous.availability = ProcessAvailability::Draining;
        }
        state
            .current_decoder_by_url
            .insert(id.url().to_string(), id.clone());
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

    pub fn refresh_prefill(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        let metadata = required_metadata(&worker, PdProcessRole::Prefill)?.clone();
        let id = PrefillId::new(worker.base_url(), metadata.launch_instance_id())?;
        let mut state = self.state.write();
        if state.current_prefill_by_url.get(id.url()) != Some(&id) {
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

    pub fn refresh_decoder(
        &self,
        worker: Arc<dyn Worker>,
    ) -> Result<Arc<DecoderDirectoryEntry>, PdDirectoryError> {
        let metadata = required_metadata(&worker, PdProcessRole::Decode)?.clone();
        let id = DecoderId::new(worker.base_url(), metadata.launch_instance_id())?;
        let mut state = self.state.write();
        if state.current_decoder_by_url.get(id.url()) != Some(&id) {
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

    pub fn drain_prefill(&self, id: &PrefillId) -> Result<(), PdDirectoryError> {
        let mut state = self.state.write();
        let record = state
            .prefills
            .get_mut(id)
            .ok_or_else(|| PdDirectoryError::UnknownPrefill(id.clone()))?;
        record.entry.pool().begin_draining();
        record.availability = ProcessAvailability::Draining;
        Ok(())
    }

    pub fn remove_drained_prefill(
        &self,
        id: &PrefillId,
    ) -> Result<Arc<PrefillDirectoryEntry>, PdDirectoryError> {
        remove_drained_prefill_locked(&mut self.state.write(), id)
    }

    pub fn drain_decoder(&self, id: &DecoderId) -> Result<(), PdDirectoryError> {
        drain_decoder_locked(&mut self.state.write(), id)
    }

    pub fn remove_drained_decoder(
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
        let owner = record.entry.pool().begin_request(request_id)?;
        Ok((Arc::clone(&record.entry), owner))
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
        let mut entries: Vec<Arc<PrefillDirectoryEntry>> = self
            .state
            .read()
            .prefills
            .values()
            .filter(|record| record.availability == ProcessAvailability::Ready)
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
    if state.current_prefill_by_url.get(id.url()) == Some(id) {
        state.current_prefill_by_url.remove(id.url());
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
    if state.current_decoder_by_url.get(id.url()) == Some(id) {
        state.current_decoder_by_url.remove(id.url());
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

fn required_metadata(
    worker: &Arc<dyn Worker>,
    role: PdProcessRole,
) -> Result<&PdProcessMetadata, PdDirectoryError> {
    let worker_role_matches = match role {
        PdProcessRole::Prefill => matches!(worker.worker_type(), WorkerType::Prefill { .. }),
        PdProcessRole::Decode => worker.worker_type() == &WorkerType::Decode,
    };
    if !worker_role_matches {
        return Err(PdDirectoryError::WorkerRoleMismatch);
    }
    let metadata = worker
        .metadata()
        .pd_process
        .as_ref()
        .ok_or(PdDirectoryError::MissingProcessMetadata)?;
    if metadata.role() != role {
        return Err(PdDirectoryError::WorkerRoleMismatch);
    }
    Ok(metadata)
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
    #[error("prefill directory supports TP2 or TP4, received TP{0}")]
    UnsupportedPrefillTp(usize),
    #[error("decoder directory supports TP1, received TP{0}")]
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
    #[error("process generation is not the current generation for its URL")]
    ProcessNotCurrent,
    #[error("process-generation metadata changed without a new launch identity")]
    GenerationMetadataChanged,
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
    };

    use uuid::Uuid;

    use super::*;
    use crate::core::pd_decoder_grant::{
        issue_test_grant, issue_test_release_receipt, DecoderGrantChildAccounting,
        DecoderSlotGeneration, EngineReleaseKind,
    };
    use crate::core::pd_decoder_pool::RetryDisposition;
    use crate::core::{
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
                .pd_process(metadata(
                    role,
                    tp_size,
                    instance_id,
                    model_fingerprint,
                    kv_layout_fingerprint,
                ))
                .build(),
        )
    }

    #[test]
    fn seeds_tp2_and_tp4_prefills_from_arbitrary_current_tp1_replica_counts() {
        for prefill_tp in [2, 4] {
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

                let prefill = directory
                    .admit_prefill(worker(
                        &format!("http://prefill-tp{prefill_tp}.test:30000"),
                        PdProcessRole::Prefill,
                        prefill_tp,
                        instance(1_000 + prefill_tp as u128),
                    ))
                    .unwrap();

                assert_eq!(prefill.pool().snapshot().replicas.len(), replica_count);
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
                    1,
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
            let (_, owner) = directory
                .begin_prefill_request(prefill.id(), "replacement-visible")
                .unwrap();
            assert_eq!(
                prefill.pool().admission_candidates(&owner).unwrap(),
                vec![replacement.id().clone()]
            );
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
    fn prefill_retirement_waits_for_owned_cohorts_and_closes_new_requests() {
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
        let (_, mut owner) = directory
            .begin_prefill_request(prefill.id(), "active-request")
            .unwrap();
        let grant = issue_test_grant(
            prefill.id().clone(),
            owner.chain_id(),
            4,
            decoder.id().clone(),
            instance(3),
            vec![DecoderSlotGeneration::new(instance(4))],
            vec![1],
            vec![DecoderGrantChildAccounting::new(64, 16)],
        )
        .unwrap();
        let mut cohort = prefill.pool().bind_grant(&owner, &grant).unwrap();

        directory.drain_prefill(prefill.id()).unwrap();
        assert!(matches!(
            prefill.pool().begin_request("too-late"),
            Err(DecoderPoolError::PrefillPoolDraining)
        ));
        assert!(matches!(
            directory.remove_drained_prefill(prefill.id()),
            Err(PdDirectoryError::Pool(
                DecoderPoolError::PrefillPoolInUse { .. }
            ))
        ));

        let receipt = issue_test_release_receipt(
            cohort.assignment_id(),
            cohort.decoder_id().clone(),
            grant.binding().child_request_ids().collect(),
            grant.binding().prefill_bootstrap_endpoint().clone(),
            cohort.slot_generations().to_vec(),
            cohort.bootstrap_rooms().to_vec(),
            cohort.grant_digest(),
            EngineReleaseKind::PreparedCancelled,
            true,
        );
        prefill
            .pool()
            .finish_before_activation(&mut cohort, &receipt, RetryDisposition::Terminal)
            .unwrap();
        prefill.pool().finalize_request(&mut owner).unwrap();

        let removed = directory.remove_drained_prefill(prefill.id()).unwrap();
        assert!(Arc::ptr_eq(removed.worker(), &prefill_worker));
        assert!(directory.prefill(prefill.id()).is_none());
        directory.drain_decoder(decoder.id()).unwrap();
        directory.remove_drained_decoder(decoder.id()).unwrap();
    }

    #[test]
    fn prefill_replacement_serializes_against_new_chain_admission() {
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
            let barrier = Arc::new(Barrier::new(2));

            let admission_directory = Arc::clone(&directory);
            let admission_barrier = Arc::clone(&barrier);
            let old_id = old.id().clone();
            let admission = thread::spawn(move || {
                admission_barrier.wait();
                match admission_directory
                    .begin_prefill_request(&old_id, format!("request-{iteration}"))
                {
                    Ok((entry, mut owner)) => {
                        entry.pool().finalize_request(&mut owner).unwrap();
                    }
                    Err(PdDirectoryError::ProcessNotReady) => {}
                    Err(error) => panic!("unexpected admission result: {error}"),
                }
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
                .pd_process(metadata(
                    PdProcessRole::Prefill,
                    2,
                    instance(8),
                    MODEL_FINGERPRINT,
                    KV_LAYOUT_FINGERPRINT,
                ))
                .build(),
        );
        assert!(matches!(
            directory.admit_decoder(wrong_role),
            Err(PdDirectoryError::WorkerRoleMismatch)
        ));
    }
}
