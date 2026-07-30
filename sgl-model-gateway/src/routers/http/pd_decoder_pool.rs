//! Request-affine admission and lifecycle accounting for disaggregated decoders.
//!
//! A cohort binds an allocator-issued engine grant to one logical request,
//! decoder process generation, request-slot allocation generation, and ordered
//! room vector. Once its first external submission begins, every child remains
//! pinned to that binding until the engine proves the whole cohort complete or
//! terminally quiescent.
//!
//! The metadata checks in this module are eligibility checks only. They do not
//! prove asymmetric TP slicing, DMA lane selection, destination correctness, or
//! transfer quiescence. Configured scales are advisory and never override an
//! allocator grant. The caller must provide one authoritative routing process,
//! an engine reservation capability, and an exact engine terminal receipt
//! before invoking `confirm_quiesced`.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::Arc,
};

use parking_lot::Mutex;
use thiserror::Error;
use uuid::Uuid;

use super::pd_decoder_grant::{
    DecoderAllocationKey, DecoderGrantBinding, DecoderGrantDigest, DecoderId,
    DecoderSlotGeneration, EngineDecoderGrant, EngineReleaseKind, EngineReleaseReceipt, PrefillId,
};

/// Engine-declared fields used to reject obviously incompatible PD pairings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCompatibilityMetadata {
    model_fingerprint: Arc<str>,
    kv_layout_fingerprint: Arc<str>,
    kv_cache_dtype: Arc<str>,
    wire_protocol: Arc<str>,
    page_size: NonZeroUsize,
}

impl EngineCompatibilityMetadata {
    /// Construct immutable metadata reported by an engine process generation.
    pub fn new(
        model_fingerprint: impl Into<String>,
        kv_layout_fingerprint: impl Into<String>,
        kv_cache_dtype: impl Into<String>,
        wire_protocol: impl Into<String>,
        page_size: usize,
    ) -> Result<Self, DecoderPoolError> {
        let model_fingerprint = nonempty("model fingerprint", model_fingerprint.into())?;
        let kv_layout_fingerprint =
            nonempty("KV layout fingerprint", kv_layout_fingerprint.into())?;
        let kv_cache_dtype = nonempty("KV cache dtype", kv_cache_dtype.into())?;
        let wire_protocol = nonempty("wire protocol", wire_protocol.into())?;
        let page_size = NonZeroUsize::new(page_size).ok_or_else(|| {
            DecoderPoolError::InvalidConfiguration("page size must be nonzero".to_string())
        })?;

        Ok(Self {
            model_fingerprint: Arc::from(model_fingerprint),
            kv_layout_fingerprint: Arc::from(kv_layout_fingerprint),
            kv_cache_dtype: Arc::from(kv_cache_dtype),
            wire_protocol: Arc::from(wire_protocol),
            page_size,
        })
    }
}

fn nonempty(name: &str, value: String) -> Result<String, DecoderPoolError> {
    if value.trim().is_empty() {
        return Err(DecoderPoolError::InvalidConfiguration(format!(
            "{name} cannot be empty"
        )));
    }
    Ok(value)
}

/// Advisory load scales and relative service weight for one decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderSchedulingHints {
    child_request_scale: NonZeroUsize,
    kv_token_scale: NonZeroUsize,
    service_weight: NonZeroUsize,
}

impl DecoderSchedulingHints {
    /// Construct advisory scheduling scales.
    pub fn new(
        child_request_scale: usize,
        kv_token_scale: usize,
        service_weight: usize,
    ) -> Result<Self, DecoderPoolError> {
        Ok(Self {
            child_request_scale: NonZeroUsize::new(child_request_scale).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration(
                    "child request scheduling scale must be nonzero".to_string(),
                )
            })?,
            kv_token_scale: NonZeroUsize::new(kv_token_scale).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration(
                    "KV token scheduling scale must be nonzero".to_string(),
                )
            })?,
            service_weight: NonZeroUsize::new(service_weight).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration("service weight must be nonzero".to_string())
            })?,
        })
    }

    /// Advisory child-request normalization scale.
    pub fn child_request_scale(&self) -> usize {
        self.child_request_scale.get()
    }

    /// Advisory KV-token normalization scale.
    pub fn kv_token_scale(&self) -> usize {
        self.kv_token_scale.get()
    }
}

/// Engine-declared metadata and advisory scheduling scales for one TP1 decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReplicaMetadata {
    id: DecoderId,
    declared_decode_tp_size: NonZeroUsize,
    compatibility: EngineCompatibilityMetadata,
    scheduling: DecoderSchedulingHints,
}

impl DecoderReplicaMetadata {
    /// Construct decoder metadata without asserting transport correctness.
    pub fn new(
        id: DecoderId,
        declared_decode_tp_size: usize,
        compatibility: EngineCompatibilityMetadata,
        scheduling: DecoderSchedulingHints,
    ) -> Result<Self, DecoderPoolError> {
        let declared_decode_tp_size =
            NonZeroUsize::new(declared_decode_tp_size).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration(
                    "declared decode tensor parallel size must be nonzero".to_string(),
                )
            })?;
        Ok(Self {
            id,
            declared_decode_tp_size,
            compatibility,
            scheduling,
        })
    }

    /// Return the decoder process-generation identity.
    pub fn id(&self) -> &DecoderId {
        &self.id
    }
}

/// Whether a decoder accepts new cohorts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderAvailability {
    Ready,
    Draining,
    Unavailable,
}

/// Whether a terminalized cohort may be retried by its logical owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDisposition {
    Terminal,
    /// Retry without changing the selected decoder's eligibility.
    Retryable,
    /// Retry after excluding the selected decoder from this request chain.
    DecoderFailed,
}

/// Current lifecycle of an issued cohort.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CohortPhase {
    Reserved,
    Dispatched,
    Quiescing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestChainPhase {
    Open,
    Terminal,
}

/// One-owner logical request chain spanning zero or more retry cohorts.
#[derive(Debug)]
pub struct LogicalRequestOwner {
    inner: Arc<DecoderPoolInner>,
    chain_id: Uuid,
    request_id: Arc<str>,
    finalized: bool,
}

impl LogicalRequestOwner {
    /// Caller-visible request identity scoped by this owner generation.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Opaque identity for the retry chain.
    pub fn chain_id(&self) -> Uuid {
        self.chain_id
    }
}

impl Drop for LogicalRequestOwner {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        self.inner.drop_request_owner(self.chain_id);
        self.finalized = true;
    }
}

/// A non-cloneable capability owning every room in one decoder cohort.
///
/// Dropping this value without a terminal method leaves all cohort resources
/// reserved. That is conservative when any child transfer may still be active.
#[derive(Debug)]
pub struct DecoderAssignmentCohort {
    pool_id: Uuid,
    chain_id: Uuid,
    assignment_id: Uuid,
    binding: DecoderGrantBinding,
    phase: CohortPhase,
}

impl DecoderAssignmentCohort {
    /// Stable identity of the selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Ordered rooms corresponding one-to-one with the original child order.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        self.binding.bootstrap_rooms()
    }

    /// Ordered engine request-slot generations bound to this cohort.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        self.binding.slot_generations()
    }

    /// Digest that every transport permit and receipt must match.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Opaque identity for lifecycle telemetry and abort acknowledgement.
    pub fn assignment_id(&self) -> Uuid {
        self.assignment_id
    }
}

/// Immutable per-replica accounting snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReplicaSnapshot {
    pub id: DecoderId,
    pub availability: DecoderAvailability,
    pub active_cohorts: usize,
    pub active_child_requests: usize,
    pub quiescing_cohorts: usize,
    pub reserved_kv_tokens: usize,
    pub remaining_decode_tokens: usize,
    pub scheduling: DecoderSchedulingHints,
}

/// Immutable pool accounting snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderPoolSnapshot {
    pub prefill_id: PrefillId,
    pub declared_prefill_tp_size: usize,
    pub active_logical_requests: usize,
    pub replicas: Vec<DecoderReplicaSnapshot>,
}

/// Decoder-pool ownership, lifecycle, and admission failures.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecoderPoolError {
    #[error("invalid decoder-pool configuration: {0}")]
    InvalidConfiguration(String),
    #[error("invalid engine decoder grant: {0}")]
    InvalidGrant(String),
    #[error("decoder {0} is already registered")]
    DuplicateDecoder(DecoderId),
    #[error("decoder {0} is not registered")]
    UnknownDecoder(DecoderId),
    #[error(
        "decoder {decoder_id} must be draining before removal, current state is {availability:?}"
    )]
    DecoderNotDraining {
        decoder_id: DecoderId,
        availability: DecoderAvailability,
    },
    #[error("decoder {decoder_id} owns {active_cohorts} active cohorts")]
    DecoderInUse {
        decoder_id: DecoderId,
        active_cohorts: usize,
    },
    #[error("decoder {decoder_id} metadata is ineligible for this prefill pool: {reason}")]
    IneligibleDecoderMetadata {
        decoder_id: DecoderId,
        reason: String,
    },
    #[error("logical request identity {0} already has an owner")]
    RequestAlreadyOwned(String),
    #[error("logical request owner was issued by another decoder pool")]
    ForeignRequestOwner,
    #[error("logical request owner is already finalized")]
    RequestOwnerFinalized,
    #[error("logical request chain {0} is unknown")]
    UnknownRequestChain(Uuid),
    #[error("logical request {request_id} still owns assignment {assignment_id}")]
    RequestHasActiveCohort {
        request_id: String,
        assignment_id: Uuid,
    },
    #[error("logical request {0} is terminal")]
    RequestChainTerminal(String),
    #[error("no ready decoder replica is registered")]
    NoReadyDecoder,
    #[error("the logical request has exhausted every registered retry alternative")]
    RetryAlternativesExhausted,
    #[error("allocator grant targets prefill {actual}, expected {expected}")]
    GrantPrefillMismatch {
        expected: PrefillId,
        actual: PrefillId,
    },
    #[error("allocator grant targets logical chain {actual}, expected {expected}")]
    GrantRequestMismatch { expected: Uuid, actual: Uuid },
    #[error("allocator grant describes source TP{actual}, expected TP{expected}")]
    GrantSourceTpMismatch { expected: usize, actual: usize },
    #[error("allocator grant selected decoder {0}, which is not eligible for this request")]
    GrantDecoderIneligible(DecoderId),
    #[error("allocator grant selected decoder {0}, which no longer accepts admissions")]
    GrantDecoderUnavailable(DecoderId),
    #[error(
        "allocator grant has {room_count} rooms but its accounting describes {child_count} children"
    )]
    GrantChildCountMismatch {
        room_count: usize,
        child_count: usize,
    },
    #[error(
        "allocator child {child_index} slot generation {slot_generation} on decoder {decoder_id} was already bound by this request"
    )]
    GrantAlreadyBound {
        child_index: usize,
        decoder_id: DecoderId,
        slot_generation: Uuid,
    },
    #[error(
        "allocator child {child_index} slot generation {slot_generation} on decoder {decoder_id} changed binding digest"
    )]
    GrantGenerationRebound {
        child_index: usize,
        decoder_id: DecoderId,
        slot_generation: Uuid,
    },
    #[error(
        "allocator bootstrap room {room} on decoder {decoder_id} is already owned by an active cohort"
    )]
    GrantRoomInUse { decoder_id: DecoderId, room: u64 },
    #[error("allocator grant identity {0} is already active")]
    GrantAlreadyActive(Uuid),
    #[error("assignment capability was not issued by this decoder pool")]
    ForeignAssignment,
    #[error("assignment {0} is unknown or already terminal")]
    UnknownAssignment(Uuid),
    #[error("engine release receipt does not match assignment {assignment_id}: {reason}")]
    InvalidEngineReleaseReceipt {
        assignment_id: Uuid,
        reason: &'static str,
    },
    #[error("assignment {assignment_id} cannot transition from {actual} to {requested}")]
    InvalidTransition {
        assignment_id: Uuid,
        actual: &'static str,
        requested: &'static str,
    },
    #[error(
        "assignment {assignment_id} reported {generated_tokens} generated tokens with only {remaining_tokens} reserved"
    )]
    InvalidProgress {
        assignment_id: Uuid,
        generated_tokens: usize,
        remaining_tokens: usize,
    },
}

#[derive(Debug)]
struct ReplicaState {
    metadata: DecoderReplicaMetadata,
    availability: DecoderAvailability,
    active_cohorts: usize,
    active_child_requests: usize,
    quiescing_cohorts: usize,
    reserved_kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Debug)]
struct RequestChainRecord {
    request_id: Arc<str>,
    phase: RequestChainPhase,
    owner_alive: bool,
    active_assignment: Option<Uuid>,
    failed_decoders: HashSet<DecoderId>,
    used_grants: HashMap<DecoderAllocationKey, DecoderGrantDigest>,
}

#[derive(Debug)]
struct AssignmentRecord {
    chain_id: Uuid,
    binding: DecoderGrantBinding,
    phase: CohortPhase,
    child_count: usize,
    kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Debug)]
struct PoolState {
    prefill_id: PrefillId,
    declared_prefill_tp_size: NonZeroUsize,
    compatibility: EngineCompatibilityMetadata,
    replicas: HashMap<DecoderId, ReplicaState>,
    request_chains: HashMap<Uuid, RequestChainRecord>,
    active_request_ids: HashMap<Arc<str>, Uuid>,
    assignments: HashMap<Uuid, AssignmentRecord>,
    active_rooms: HashSet<(DecoderId, u64)>,
}

#[derive(Debug)]
struct DecoderPoolInner {
    pool_id: Uuid,
    state: Mutex<PoolState>,
}

impl DecoderPoolInner {
    fn drop_request_owner(&self, chain_id: Uuid) {
        let mut state = self.state.lock();
        let remove_now = match state.request_chains.get_mut(&chain_id) {
            Some(chain) if chain.active_assignment.is_none() => true,
            Some(chain) => {
                chain.owner_alive = false;
                false
            }
            None => false,
        };
        if remove_now {
            remove_request_chain(&mut state, chain_id);
        }
    }
}

/// Atomic cohort admission and lifecycle authority for a single prefill replica.
#[derive(Clone, Debug)]
pub struct DecoderPool {
    inner: Arc<DecoderPoolInner>,
}

impl DecoderPool {
    /// Construct an empty pool from declared prefill metadata.
    pub fn new(
        prefill_id: PrefillId,
        declared_prefill_tp_size: usize,
        compatibility: EngineCompatibilityMetadata,
    ) -> Result<Self, DecoderPoolError> {
        if declared_prefill_tp_size != 2 && declared_prefill_tp_size != 4 {
            return Err(DecoderPoolError::InvalidConfiguration(
                "declared prefill tensor parallel size must be 2 or 4".to_string(),
            ));
        }
        let declared_prefill_tp_size =
            NonZeroUsize::new(declared_prefill_tp_size).expect("2 and 4 are nonzero");
        Ok(Self {
            inner: Arc::new(DecoderPoolInner {
                pool_id: Uuid::new_v4(),
                state: Mutex::new(PoolState {
                    prefill_id,
                    declared_prefill_tp_size,
                    compatibility,
                    replicas: HashMap::new(),
                    request_chains: HashMap::new(),
                    active_request_ids: HashMap::new(),
                    assignments: HashMap::new(),
                    active_rooms: HashSet::new(),
                }),
            }),
        })
    }

    /// Register declared metadata for a decoder process generation.
    pub fn register(&self, metadata: DecoderReplicaMetadata) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        if state.replicas.contains_key(&metadata.id) {
            return Err(DecoderPoolError::DuplicateDecoder(metadata.id));
        }
        if metadata.declared_decode_tp_size.get() != 1 {
            return Err(DecoderPoolError::IneligibleDecoderMetadata {
                decoder_id: metadata.id,
                reason: format!(
                    "this pool requires declared TP1 decoders, received TP{}",
                    metadata.declared_decode_tp_size
                ),
            });
        }
        if !state
            .declared_prefill_tp_size
            .get()
            .is_multiple_of(metadata.declared_decode_tp_size.get())
        {
            return Err(DecoderPoolError::IneligibleDecoderMetadata {
                decoder_id: metadata.id,
                reason: format!(
                    "declared prefill TP{} is not divisible by declared decode TP{}",
                    state.declared_prefill_tp_size, metadata.declared_decode_tp_size
                ),
            });
        }
        if metadata.compatibility != state.compatibility {
            return Err(DecoderPoolError::IneligibleDecoderMetadata {
                decoder_id: metadata.id,
                reason: "model, KV layout, dtype, page size, or wire protocol metadata differs"
                    .to_string(),
            });
        }

        state.replicas.insert(
            metadata.id.clone(),
            ReplicaState {
                metadata,
                availability: DecoderAvailability::Ready,
                active_cohorts: 0,
                active_child_requests: 0,
                quiescing_cohorts: 0,
                reserved_kv_tokens: 0,
                remaining_decode_tokens: 0,
            },
        );
        Ok(())
    }

    /// Begin a unique logical request chain.
    pub fn begin_request(
        &self,
        request_id: impl Into<String>,
    ) -> Result<LogicalRequestOwner, DecoderPoolError> {
        let request_id = request_id.into();
        if request_id.trim().is_empty() {
            return Err(DecoderPoolError::InvalidGrant(
                "request identity cannot be empty".to_string(),
            ));
        }
        let request_id: Arc<str> = Arc::from(request_id);
        let mut state = self.inner.state.lock();
        if state.active_request_ids.contains_key(&request_id) {
            return Err(DecoderPoolError::RequestAlreadyOwned(
                request_id.to_string(),
            ));
        }
        let chain_id = Uuid::new_v4();
        state.request_chains.insert(
            chain_id,
            RequestChainRecord {
                request_id: Arc::clone(&request_id),
                phase: RequestChainPhase::Open,
                owner_alive: true,
                active_assignment: None,
                failed_decoders: HashSet::new(),
                used_grants: HashMap::new(),
            },
        );
        state
            .active_request_ids
            .insert(Arc::clone(&request_id), chain_id);
        Ok(LogicalRequestOwner {
            inner: Arc::clone(&self.inner),
            chain_id,
            request_id,
            finalized: false,
        })
    }

    /// Finalize a retry chain after success, cancellation, or retry exhaustion.
    pub fn finalize_request(
        &self,
        request: &mut LogicalRequestOwner,
    ) -> Result<(), DecoderPoolError> {
        self.validate_request_owner(request)?;
        let mut state = self.inner.state.lock();
        let chain = state
            .request_chains
            .get(&request.chain_id)
            .ok_or(DecoderPoolError::UnknownRequestChain(request.chain_id))?;
        if let Some(assignment_id) = chain.active_assignment {
            return Err(DecoderPoolError::RequestHasActiveCohort {
                request_id: chain.request_id.to_string(),
                assignment_id,
            });
        }
        remove_request_chain(&mut state, request.chain_id);
        request.finalized = true;
        Ok(())
    }

    /// Change whether a registered decoder accepts new cohorts.
    pub fn set_availability(
        &self,
        decoder_id: &DecoderId,
        availability: DecoderAvailability,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let replica = state
            .replicas
            .get_mut(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        replica.availability = availability;
        Ok(())
    }

    /// Update advisory load scales without changing allocator authority.
    pub fn update_scheduling_hints(
        &self,
        decoder_id: &DecoderId,
        scheduling: DecoderSchedulingHints,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let replica = state
            .replicas
            .get_mut(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        replica.metadata.scheduling = scheduling;
        Ok(())
    }

    /// Remove a draining process generation after every cohort is terminal.
    pub fn remove(&self, decoder_id: &DecoderId) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let replica = state
            .replicas
            .get(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        if replica.availability != DecoderAvailability::Draining {
            return Err(DecoderPoolError::DecoderNotDraining {
                decoder_id: decoder_id.clone(),
                availability: replica.availability,
            });
        }
        if replica.active_cohorts > 0 {
            return Err(DecoderPoolError::DecoderInUse {
                decoder_id: decoder_id.clone(),
                active_cohorts: replica.active_cohorts,
            });
        }
        state.replicas.remove(decoder_id);
        Ok(())
    }

    /// Return ordered decoder candidates for an allocator reservation attempt.
    pub fn admission_candidates(
        &self,
        request: &LogicalRequestOwner,
    ) -> Result<Vec<DecoderId>, DecoderPoolError> {
        self.validate_request_owner(request)?;
        let state = self.inner.state.lock();
        let failed_decoders = {
            let chain = state
                .request_chains
                .get(&request.chain_id)
                .ok_or(DecoderPoolError::UnknownRequestChain(request.chain_id))?;
            if let Some(assignment_id) = chain.active_assignment {
                return Err(DecoderPoolError::RequestHasActiveCohort {
                    request_id: chain.request_id.to_string(),
                    assignment_id,
                });
            }
            if chain.phase == RequestChainPhase::Terminal {
                return Err(DecoderPoolError::RequestChainTerminal(
                    chain.request_id.to_string(),
                ));
            }
            chain.failed_decoders.clone()
        };
        select_decoders(&state.replicas, &failed_decoders)
    }

    /// Atomically bind an allocator-issued grant to this logical request.
    pub fn bind_grant(
        &self,
        request: &LogicalRequestOwner,
        grant: &EngineDecoderGrant,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        self.validate_request_owner(request)?;
        let binding = grant.binding();
        let accounting = binding.accounting();
        if binding.bootstrap_rooms().len() != accounting.child_count() {
            return Err(DecoderPoolError::GrantChildCountMismatch {
                room_count: binding.bootstrap_rooms().len(),
                child_count: accounting.child_count(),
            });
        }

        let mut state = self.inner.state.lock();
        if binding.prefill_id() != &state.prefill_id {
            return Err(DecoderPoolError::GrantPrefillMismatch {
                expected: state.prefill_id.clone(),
                actual: binding.prefill_id().clone(),
            });
        }
        if binding.request_chain_id() != request.chain_id {
            return Err(DecoderPoolError::GrantRequestMismatch {
                expected: request.chain_id,
                actual: binding.request_chain_id(),
            });
        }
        if binding.source_tp_size() != state.declared_prefill_tp_size.get() {
            return Err(DecoderPoolError::GrantSourceTpMismatch {
                expected: state.declared_prefill_tp_size.get(),
                actual: binding.source_tp_size(),
            });
        }
        if state.assignments.contains_key(&binding.grant_id()) {
            return Err(DecoderPoolError::GrantAlreadyActive(binding.grant_id()));
        }
        let failed_decoders = {
            let chain = state
                .request_chains
                .get(&request.chain_id)
                .ok_or(DecoderPoolError::UnknownRequestChain(request.chain_id))?;
            if let Some(assignment_id) = chain.active_assignment {
                return Err(DecoderPoolError::RequestHasActiveCohort {
                    request_id: chain.request_id.to_string(),
                    assignment_id,
                });
            }
            if chain.phase == RequestChainPhase::Terminal {
                return Err(DecoderPoolError::RequestChainTerminal(
                    chain.request_id.to_string(),
                ));
            }
            for (child_index, allocation_key) in binding.allocation_keys().enumerate() {
                let Some(existing_digest) = chain.used_grants.get(&allocation_key).copied() else {
                    continue;
                };
                if existing_digest == binding.digest() {
                    return Err(DecoderPoolError::GrantAlreadyBound {
                        child_index,
                        decoder_id: binding.decoder_id().clone(),
                        slot_generation: binding.slot_generations()[child_index].as_uuid(),
                    });
                }
                return Err(DecoderPoolError::GrantGenerationRebound {
                    child_index,
                    decoder_id: binding.decoder_id().clone(),
                    slot_generation: binding.slot_generations()[child_index].as_uuid(),
                });
            }
            chain.failed_decoders.clone()
        };

        let candidates = select_decoders(&state.replicas, &failed_decoders)?;
        if !candidates.contains(binding.decoder_id()) {
            return Err(DecoderPoolError::GrantDecoderIneligible(
                binding.decoder_id().clone(),
            ));
        }
        let replica = state
            .replicas
            .get(binding.decoder_id())
            .expect("eligible grant decoder disappeared while pool lock was held");
        if replica.availability != DecoderAvailability::Ready {
            return Err(DecoderPoolError::GrantDecoderUnavailable(
                binding.decoder_id().clone(),
            ));
        }
        if let Some(room) = binding.bootstrap_rooms().iter().find(|room| {
            state
                .active_rooms
                .contains(&(binding.decoder_id().clone(), **room))
        }) {
            return Err(DecoderPoolError::GrantRoomInUse {
                decoder_id: binding.decoder_id().clone(),
                room: *room,
            });
        }

        let active_child_requests = replica
            .active_child_requests
            .checked_add(accounting.child_count())
            .ok_or_else(|| {
                DecoderPoolError::InvalidGrant(
                    "active child-request accounting overflows usize".to_string(),
                )
            })?;
        let reserved_kv_tokens = replica
            .reserved_kv_tokens
            .checked_add(accounting.total_reserved_kv_tokens())
            .ok_or_else(|| {
                DecoderPoolError::InvalidGrant(
                    "active KV-token accounting overflows usize".to_string(),
                )
            })?;
        let remaining_decode_tokens = replica
            .remaining_decode_tokens
            .checked_add(accounting.total_remaining_decode_tokens())
            .ok_or_else(|| {
                DecoderPoolError::InvalidGrant(
                    "active decode-token accounting overflows usize".to_string(),
                )
            })?;
        let assignment_id = binding.grant_id();

        let replica = state
            .replicas
            .get_mut(binding.decoder_id())
            .expect("eligible grant decoder disappeared while pool lock was held");
        replica.active_cohorts += 1;
        replica.active_child_requests = active_child_requests;
        replica.reserved_kv_tokens = reserved_kv_tokens;
        replica.remaining_decode_tokens = remaining_decode_tokens;
        state.active_rooms.extend(
            binding
                .bootstrap_rooms()
                .iter()
                .map(|room| (binding.decoder_id().clone(), *room)),
        );

        state.assignments.insert(
            assignment_id,
            AssignmentRecord {
                chain_id: request.chain_id,
                binding: binding.clone(),
                phase: CohortPhase::Reserved,
                child_count: accounting.child_count(),
                kv_tokens: accounting.total_reserved_kv_tokens(),
                remaining_decode_tokens: accounting.total_remaining_decode_tokens(),
            },
        );
        let chain = state
            .request_chains
            .get_mut(&request.chain_id)
            .expect("request chain disappeared while pool lock was held");
        for allocation_key in binding.allocation_keys() {
            chain.used_grants.insert(allocation_key, binding.digest());
        }
        chain.active_assignment = Some(assignment_id);

        Ok(DecoderAssignmentCohort {
            pool_id: self.inner.pool_id,
            chain_id: request.chain_id,
            assignment_id,
            binding: binding.clone(),
            phase: CohortPhase::Reserved,
        })
    }

    /// Mark that the cohort is crossing its first external submission boundary.
    pub fn mark_dispatched(
        &self,
        cohort: &mut DecoderAssignmentCohort,
    ) -> Result<(), DecoderPoolError> {
        self.transition(cohort, CohortPhase::Reserved, CohortPhase::Dispatched)
    }

    /// Reduce aggregate remaining work as child decode tokens are emitted.
    pub fn observe_decode_progress(
        &self,
        cohort: &DecoderAssignmentCohort,
        generated_tokens: usize,
    ) -> Result<(), DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let (decoder_id, remaining_tokens) = {
            let record = state
                .assignments
                .get(&cohort.assignment_id)
                .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
            validate_assignment_record(record, cohort)?;
            if record.phase != CohortPhase::Dispatched {
                return Err(invalid_transition(
                    cohort.assignment_id,
                    record.phase,
                    "record decode progress",
                ));
            }
            (
                record.binding.decoder_id().clone(),
                record.remaining_decode_tokens,
            )
        };
        if generated_tokens > remaining_tokens {
            return Err(DecoderPoolError::InvalidProgress {
                assignment_id: cohort.assignment_id,
                generated_tokens,
                remaining_tokens,
            });
        }
        state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment disappeared while pool lock was held")
            .remaining_decode_tokens -= generated_tokens;
        state
            .replicas
            .get_mut(&decoder_id)
            .expect("assigned decoder disappeared while pool lock was held")
            .remaining_decode_tokens -= generated_tokens;
        Ok(())
    }

    /// Atomically quarantine the whole cohort while any child may still transfer.
    pub fn begin_quiescence(
        &self,
        cohort: &mut DecoderAssignmentCohort,
    ) -> Result<(), DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let decoder_id = {
            let record = state
                .assignments
                .get(&cohort.assignment_id)
                .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
            validate_assignment_record(record, cohort)?;
            if cohort.phase != CohortPhase::Dispatched || record.phase != CohortPhase::Dispatched {
                return Err(invalid_transition(
                    cohort.assignment_id,
                    record.phase,
                    phase_name(CohortPhase::Quiescing),
                ));
            }
            record.binding.decoder_id().clone()
        };
        state
            .replicas
            .get_mut(&decoder_id)
            .expect("assigned decoder disappeared while pool lock was held")
            .quiescing_cohorts += 1;
        state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment disappeared while pool lock was held")
            .phase = CohortPhase::Quiescing;
        cohort.phase = CohortPhase::Quiescing;
        Ok(())
    }

    /// Terminalize a cohort that never crossed the dispatch boundary.
    pub fn finish_before_dispatch(
        &self,
        cohort: DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        validate_engine_release_receipt(&cohort, receipt, EngineReleaseKind::PreparedCancelled)?;
        self.release(cohort, CohortPhase::Reserved, disposition)
    }

    /// Terminalize a successfully completed child cohort.
    pub fn complete(
        &self,
        cohort: DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
    ) -> Result<(), DecoderPoolError> {
        validate_engine_release_receipt(&cohort, receipt, EngineReleaseKind::Completed)?;
        self.release(cohort, CohortPhase::Dispatched, RetryDisposition::Terminal)
    }

    /// Release every child after the engine attests exact terminal quiescence.
    pub fn confirm_quiesced(
        &self,
        cohort: DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        validate_engine_release_receipt(&cohort, receipt, EngineReleaseKind::Aborted)?;
        self.release(cohort, CohortPhase::Quiescing, disposition)
    }

    /// Return immutable accounting suitable for metrics and tests.
    pub fn snapshot(&self) -> DecoderPoolSnapshot {
        let state = self.inner.state.lock();
        let mut replicas: Vec<DecoderReplicaSnapshot> = state
            .replicas
            .values()
            .map(|replica| DecoderReplicaSnapshot {
                id: replica.metadata.id.clone(),
                availability: replica.availability,
                active_cohorts: replica.active_cohorts,
                active_child_requests: replica.active_child_requests,
                quiescing_cohorts: replica.quiescing_cohorts,
                reserved_kv_tokens: replica.reserved_kv_tokens,
                remaining_decode_tokens: replica.remaining_decode_tokens,
                scheduling: replica.metadata.scheduling,
            })
            .collect();
        replicas.sort_by(|left, right| left.id.cmp(&right.id));
        DecoderPoolSnapshot {
            prefill_id: state.prefill_id.clone(),
            declared_prefill_tp_size: state.declared_prefill_tp_size.get(),
            active_logical_requests: state.request_chains.len(),
            replicas,
        }
    }

    fn transition(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        expected: CohortPhase,
        next: CohortPhase,
    ) -> Result<(), DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        if cohort.phase != expected || record.phase != expected {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                phase_name(next),
            ));
        }
        cohort.phase = next;
        record.phase = next;
        Ok(())
    }

    fn release(
        &self,
        cohort: DecoderAssignmentCohort,
        expected: CohortPhase,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        self.validate_assignment_pool(&cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, &cohort)?;
        if cohort.phase != expected || record.phase != expected {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                "terminal",
            ));
        }

        let record = state
            .assignments
            .remove(&cohort.assignment_id)
            .expect("assignment disappeared while pool lock was held");
        let replica = state
            .replicas
            .get_mut(record.binding.decoder_id())
            .expect("assigned decoder disappeared while pool lock was held");
        replica.active_cohorts -= 1;
        replica.active_child_requests -= record.child_count;
        replica.reserved_kv_tokens -= record.kv_tokens;
        replica.remaining_decode_tokens -= record.remaining_decode_tokens;
        if expected == CohortPhase::Quiescing {
            replica.quiescing_cohorts -= 1;
        }
        for room in record.binding.bootstrap_rooms() {
            let removed = state
                .active_rooms
                .remove(&(record.binding.decoder_id().clone(), *room));
            debug_assert!(removed);
        }

        let remove_request = {
            let chain = state
                .request_chains
                .get_mut(&record.chain_id)
                .expect("request owner disappeared while cohort was active");
            assert_eq!(chain.active_assignment, Some(cohort.assignment_id));
            chain.active_assignment = None;
            match disposition {
                RetryDisposition::Terminal => {
                    chain.phase = RequestChainPhase::Terminal;
                    chain.failed_decoders.clear();
                }
                RetryDisposition::Retryable => {
                    debug_assert_eq!(chain.phase, RequestChainPhase::Open);
                }
                RetryDisposition::DecoderFailed => {
                    chain
                        .failed_decoders
                        .insert(record.binding.decoder_id().clone());
                }
            }
            !chain.owner_alive
        };
        if remove_request {
            remove_request_chain(&mut state, record.chain_id);
        }
        Ok(())
    }

    fn validate_request_owner(
        &self,
        request: &LogicalRequestOwner,
    ) -> Result<(), DecoderPoolError> {
        if !Arc::ptr_eq(&self.inner, &request.inner) {
            return Err(DecoderPoolError::ForeignRequestOwner);
        }
        if request.finalized {
            return Err(DecoderPoolError::RequestOwnerFinalized);
        }
        Ok(())
    }

    fn validate_assignment_pool(
        &self,
        cohort: &DecoderAssignmentCohort,
    ) -> Result<(), DecoderPoolError> {
        if cohort.pool_id != self.inner.pool_id {
            return Err(DecoderPoolError::ForeignAssignment);
        }
        Ok(())
    }
}

fn select_decoders(
    replicas: &HashMap<DecoderId, ReplicaState>,
    failed_decoders: &HashSet<DecoderId>,
) -> Result<Vec<DecoderId>, DecoderPoolError> {
    let unfailed: Vec<&ReplicaState> = replicas
        .values()
        .filter(|replica| !failed_decoders.contains(&replica.metadata.id))
        .collect();
    if !failed_decoders.is_empty() && unfailed.is_empty() {
        return Err(DecoderPoolError::RetryAlternativesExhausted);
    }
    let mut ready: Vec<&ReplicaState> = unfailed
        .into_iter()
        .filter(|replica| replica.availability == DecoderAvailability::Ready)
        .collect();
    if ready.is_empty() {
        return Err(DecoderPoolError::NoReadyDecoder);
    }
    ready.sort_by(|left, right| compare_current_load(left, right));
    Ok(ready
        .into_iter()
        .map(|replica| replica.metadata.id.clone())
        .collect())
}

fn remove_request_chain(state: &mut PoolState, chain_id: Uuid) {
    let Some(chain) = state.request_chains.remove(&chain_id) else {
        return;
    };
    let removed = state.active_request_ids.remove(&chain.request_id);
    debug_assert_eq!(removed, Some(chain_id));
}

fn validate_assignment_record(
    record: &AssignmentRecord,
    cohort: &DecoderAssignmentCohort,
) -> Result<(), DecoderPoolError> {
    if record.chain_id != cohort.chain_id || record.binding != cohort.binding {
        return Err(DecoderPoolError::ForeignAssignment);
    }
    Ok(())
}

fn validate_engine_release_receipt(
    cohort: &DecoderAssignmentCohort,
    receipt: &EngineReleaseReceipt,
    expected_kind: EngineReleaseKind,
) -> Result<(), DecoderPoolError> {
    let mismatch = if receipt.grant_id() != cohort.assignment_id {
        Some("assignment identity differs")
    } else if receipt.decoder_id() != cohort.binding.decoder_id() {
        Some("decoder process generation differs")
    } else if !cohort
        .binding
        .child_request_ids()
        .eq(receipt.child_request_ids().iter().copied())
    {
        Some("ordered child request identities differ")
    } else if receipt.prefill_bootstrap_endpoint() != cohort.binding.prefill_bootstrap_endpoint() {
        Some("prefill bootstrap endpoint differs")
    } else if receipt.slot_generations() != cohort.binding.slot_generations() {
        Some("ordered decoder slot generations differ")
    } else if receipt.bootstrap_rooms() != cohort.binding.bootstrap_rooms() {
        Some("ordered bootstrap rooms differ")
    } else if receipt.grant_digest() != cohort.binding.digest() {
        Some("grant digest differs")
    } else if receipt.kind() != expected_kind {
        Some("release kind differs")
    } else if !receipt.take_once() {
        Some("receipt does not attest take-once reconciliation")
    } else {
        None
    };
    if let Some(reason) = mismatch {
        return Err(DecoderPoolError::InvalidEngineReleaseReceipt {
            assignment_id: cohort.assignment_id,
            reason,
        });
    }
    Ok(())
}

fn compare_current_load(left: &ReplicaState, right: &ReplicaState) -> Ordering {
    compare_ratio(
        left.remaining_decode_tokens,
        left.metadata.scheduling.service_weight.get(),
        right.remaining_decode_tokens,
        right.metadata.scheduling.service_weight.get(),
    )
    .then_with(|| {
        compare_ratio(
            left.reserved_kv_tokens,
            left.metadata.scheduling.kv_token_scale.get(),
            right.reserved_kv_tokens,
            right.metadata.scheduling.kv_token_scale.get(),
        )
    })
    .then_with(|| {
        compare_ratio(
            left.active_child_requests,
            left.metadata.scheduling.child_request_scale.get(),
            right.active_child_requests,
            right.metadata.scheduling.child_request_scale.get(),
        )
    })
    .then_with(|| left.metadata.id.cmp(&right.metadata.id))
}

fn compare_ratio(
    left_numerator: usize,
    left_denominator: usize,
    right_numerator: usize,
    right_denominator: usize,
) -> Ordering {
    ((left_numerator as u128) * (right_denominator as u128))
        .cmp(&((right_numerator as u128) * (left_denominator as u128)))
}

fn invalid_transition(
    assignment_id: Uuid,
    actual: CohortPhase,
    requested: &'static str,
) -> DecoderPoolError {
    DecoderPoolError::InvalidTransition {
        assignment_id,
        actual: phase_name(actual),
        requested,
    }
}

fn phase_name(phase: CohortPhase) -> &'static str {
    match phase {
        CohortPhase::Reserved => "reserved",
        CohortPhase::Dispatched => "dispatched",
        CohortPhase::Quiescing => "quiescing",
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
        thread,
    };

    use super::*;
    use crate::routers::http::pd_decoder_grant::{
        issue_test_grant, issue_test_release_receipt, DecoderGrantChildAccounting,
    };

    static NEXT_ROOM: AtomicU64 = AtomicU64::new(1);

    fn stable_instance_id(name: &str) -> Uuid {
        let digest = blake3::hash(name.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest.as_bytes()[..16]);
        Uuid::from_bytes(bytes)
    }

    fn prefill_id(name: &str) -> PrefillId {
        PrefillId::new("http://prefill.test:30000", stable_instance_id(name)).unwrap()
    }

    fn decoder_id(name: &str) -> DecoderId {
        DecoderId::new("http://decode.test:30001", stable_instance_id(name)).unwrap()
    }

    fn compatibility(protocol: &str) -> EngineCompatibilityMetadata {
        EngineCompatibilityMetadata::new(
            "gemma-4-31b-nvfp4@sha256:model",
            "gemma4-full10-swa50@sha256:layout",
            "bfloat16",
            protocol,
            1,
        )
        .unwrap()
    }

    fn scheduling(child_scale: usize, kv_scale: usize) -> DecoderSchedulingHints {
        DecoderSchedulingHints::new(child_scale, kv_scale, 1).unwrap()
    }

    fn replica(name: &str, protocol: &str) -> DecoderReplicaMetadata {
        replica_with_id(decoder_id(name), protocol)
    }

    fn replica_with_id(id: DecoderId, protocol: &str) -> DecoderReplicaMetadata {
        DecoderReplicaMetadata::new(id, 1, compatibility(protocol), scheduling(32, 32_000)).unwrap()
    }

    fn pool_for_prefill(prefill_name: &str, declared_prefill_tp_size: usize) -> DecoderPool {
        DecoderPool::new(
            prefill_id(prefill_name),
            declared_prefill_tp_size,
            compatibility("packed-v1"),
        )
        .unwrap()
    }

    fn pool(declared_prefill_tp_size: usize) -> DecoderPool {
        pool_for_prefill("prefill-0@generation-1", declared_prefill_tp_size)
    }

    fn child_accounting() -> DecoderGrantChildAccounting {
        DecoderGrantChildAccounting::new(1_024, 128)
    }

    fn scalar_accounting() -> Vec<DecoderGrantChildAccounting> {
        vec![child_accounting()]
    }

    fn issue_grant(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        decoder_id: DecoderId,
        grant_id: Uuid,
        slot_generations: Vec<DecoderSlotGeneration>,
        rooms: Vec<u64>,
        accounting: Vec<DecoderGrantChildAccounting>,
    ) -> EngineDecoderGrant {
        let snapshot = pool.snapshot();
        issue_test_grant(
            snapshot.prefill_id,
            owner.chain_id(),
            snapshot.declared_prefill_tp_size,
            decoder_id,
            grant_id,
            slot_generations,
            rooms,
            accounting,
        )
        .expect("test allocator grant must be valid")
    }

    fn release_receipt(
        cohort: &DecoderAssignmentCohort,
        kind: EngineReleaseKind,
    ) -> EngineReleaseReceipt {
        issue_test_release_receipt(
            cohort.assignment_id(),
            cohort.decoder_id().clone(),
            cohort.slot_generations().to_vec(),
            cohort.bootstrap_rooms().to_vec(),
            cohort.grant_digest(),
            kind,
            true,
        )
    }

    fn release_before_dispatch(
        pool: &DecoderPool,
        cohort: DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        let receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
        pool.finish_before_dispatch(cohort, &receipt, disposition)
    }

    fn release_after_abort(
        pool: &DecoderPool,
        cohort: DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        let receipt = release_receipt(&cohort, EngineReleaseKind::Aborted);
        pool.confirm_quiesced(cohort, &receipt, disposition)
    }

    fn release_after_completion(
        pool: &DecoderPool,
        cohort: DecoderAssignmentCohort,
    ) -> Result<(), DecoderPoolError> {
        let receipt = release_receipt(&cohort, EngineReleaseKind::Completed);
        pool.complete(cohort, &receipt)
    }

    fn bind_next(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        accounting: Vec<DecoderGrantChildAccounting>,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        let decoder_id = pool
            .admission_candidates(owner)?
            .into_iter()
            .next()
            .expect("candidate list cannot be empty");
        let child_count = accounting.len();
        let first_room = NEXT_ROOM.fetch_add(child_count as u64, Ordering::Relaxed);
        let rooms = (0..child_count)
            .map(|offset| first_room + offset as u64)
            .collect();
        let grant = issue_grant(
            pool,
            owner,
            decoder_id,
            Uuid::new_v4(),
            (0..child_count)
                .map(|_| DecoderSlotGeneration::new(Uuid::new_v4()))
                .collect(),
            rooms,
            accounting,
        );
        pool.bind_grant(owner, &grant)
    }

    #[test]
    fn registers_declared_tp_metadata_without_claiming_transport_correctness() {
        for declared_prefill_tp_size in [2, 4] {
            let pool = pool(declared_prefill_tp_size);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            assert_eq!(
                pool.snapshot().declared_prefill_tp_size,
                declared_prefill_tp_size
            );
        }
    }

    #[test]
    fn rejects_unsupported_prefill_tensor_parallelism() {
        for declared_prefill_tp_size in [0, 1, 3, 8] {
            assert!(DecoderPool::new(
                prefill_id("prefill-invalid"),
                declared_prefill_tp_size,
                compatibility("packed-v1"),
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_ineligible_declared_decoder_metadata() {
        let pool = pool(4);
        let tp2 = DecoderReplicaMetadata::new(
            decoder_id("decode-tp2"),
            2,
            compatibility("packed-v1"),
            scheduling(32, 32_000),
        )
        .unwrap();
        assert!(matches!(
            pool.register(tp2),
            Err(DecoderPoolError::IneligibleDecoderMetadata { .. })
        ));
        assert!(matches!(
            pool.register(replica("decode-wrong-wire", "packed-v2")),
            Err(DecoderPoolError::IneligibleDecoderMetadata { .. })
        ));
    }

    #[test]
    fn balances_arbitrary_replica_count_by_projected_decode_work() {
        let pool = pool(2);
        for index in 0..3 {
            pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                .unwrap();
        }

        let mut requests = Vec::new();
        let mut cohorts = Vec::new();
        for index in 0..9 {
            let request = pool.begin_request(format!("request-{index}")).unwrap();
            let cohort = bind_next(&pool, &request, scalar_accounting()).unwrap();
            requests.push(request);
            cohorts.push(cohort);
        }
        assert_eq!(
            pool.snapshot()
                .replicas
                .iter()
                .map(|replica| replica.active_cohorts)
                .collect::<Vec<_>>(),
            vec![3, 3, 3]
        );

        for (mut request, cohort) in requests.into_iter().zip(cohorts) {
            release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();
            pool.finalize_request(&mut request).unwrap();
        }
    }

    #[test]
    fn batch_cohort_preserves_room_order_and_releases_all_children_together() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let request = pool.begin_request("batch").unwrap();
        let accounting = vec![
            DecoderGrantChildAccounting::new(100, 10),
            DecoderGrantChildAccounting::new(200, 20),
            DecoderGrantChildAccounting::new(300, 30),
        ];
        let decoder_id = pool.admission_candidates(&request).unwrap().remove(0);
        let grant = issue_grant(
            &pool,
            &request,
            decoder_id,
            Uuid::new_v4(),
            (0..3)
                .map(|_| DecoderSlotGeneration::new(Uuid::new_v4()))
                .collect(),
            vec![41, 43, 42],
            accounting,
        );
        let mut cohort = pool.bind_grant(&request, &grant).unwrap();
        assert_eq!(cohort.bootstrap_rooms(), &[41, 43, 42]);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].active_child_requests, 3);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 600);
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 60);

        pool.mark_dispatched(&mut cohort).unwrap();
        pool.begin_quiescence(&mut cohort).unwrap();
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 1);
        release_after_abort(&pool, cohort, RetryDisposition::Terminal).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 0);
        assert_eq!(snapshot.replicas[0].active_child_requests, 0);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
    }

    #[test]
    fn grant_cannot_cross_logical_request_owners() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = pool.admission_candidates(&first_owner).unwrap().remove(0);
        let grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id,
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![11],
            scalar_accounting(),
        );

        assert_eq!(
            pool.bind_grant(&second_owner, &grant).unwrap_err(),
            DecoderPoolError::GrantRequestMismatch {
                expected: second_owner.chain_id(),
                actual: first_owner.chain_id(),
            }
        );
        let cohort = pool.bind_grant(&first_owner, &grant).unwrap();
        release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn grant_cannot_cross_prefill_process_generations() {
        let pool = pool_for_prefill("prefill-0@generation-2", 2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let decoder_id = pool.admission_candidates(&owner).unwrap().remove(0);
        let grant = issue_test_grant(
            prefill_id("prefill-0@generation-1"),
            owner.chain_id(),
            2,
            decoder_id,
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![11],
            scalar_accounting(),
        )
        .unwrap();

        assert_eq!(
            pool.bind_grant(&owner, &grant).unwrap_err(),
            DecoderPoolError::GrantPrefillMismatch {
                expected: prefill_id("prefill-0@generation-2"),
                actual: prefill_id("prefill-0@generation-1"),
            }
        );
    }

    #[test]
    fn release_requires_an_exact_engine_receipt() {
        let expected_reasons = [
            "assignment identity differs",
            "decoder process generation differs",
            "ordered decoder slot generations differ",
            "ordered bootstrap rooms differ",
            "grant digest differs",
            "release kind differs",
            "receipt does not attest take-once reconciliation",
        ];
        for (mismatch, expected_reason) in expected_reasons.into_iter().enumerate() {
            let pool = pool(2);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            let owner = pool.begin_request(format!("request-{mismatch}")).unwrap();
            let selected_decoder = pool.admission_candidates(&owner).unwrap().remove(0);
            let grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![11],
                scalar_accounting(),
            );
            let cohort = pool.bind_grant(&owner, &grant).unwrap();
            let alternate_grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![12],
                scalar_accounting(),
            );
            let receipt = issue_test_release_receipt(
                if mismatch == 0 {
                    Uuid::new_v4()
                } else {
                    cohort.assignment_id()
                },
                if mismatch == 1 {
                    decoder_id("decode-other")
                } else {
                    selected_decoder
                },
                if mismatch == 2 {
                    vec![DecoderSlotGeneration::new(Uuid::new_v4())]
                } else {
                    cohort.slot_generations().to_vec()
                },
                if mismatch == 3 {
                    vec![12]
                } else {
                    cohort.bootstrap_rooms().to_vec()
                },
                if mismatch == 4 {
                    alternate_grant.grant_digest()
                } else {
                    cohort.grant_digest()
                },
                if mismatch == 5 {
                    EngineReleaseKind::Completed
                } else {
                    EngineReleaseKind::PreparedCancelled
                },
                mismatch != 6,
            );
            let assignment_id = cohort.assignment_id();

            assert_eq!(
                pool.finish_before_dispatch(cohort, &receipt, RetryDisposition::Terminal)
                    .unwrap_err(),
                DecoderPoolError::InvalidEngineReleaseReceipt {
                    assignment_id,
                    reason: expected_reason,
                }
            );
            assert_eq!(pool.snapshot().replicas[0].active_cohorts, 1);
        }
    }

    #[test]
    fn retry_cannot_replay_or_alter_a_decoder_slot_binding() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let decoder_id = pool.admission_candidates(&owner).unwrap().remove(0);
        let slot_generations = vec![
            DecoderSlotGeneration::new(Uuid::new_v4()),
            DecoderSlotGeneration::new(Uuid::new_v4()),
        ];
        let original = issue_grant(
            &pool,
            &owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            slot_generations.clone(),
            vec![101, 102],
            vec![child_accounting(), child_accounting()],
        );
        let first = pool.bind_grant(&owner, &original).unwrap();
        release_before_dispatch(&pool, first, RetryDisposition::Retryable).unwrap();

        assert_eq!(
            pool.bind_grant(&owner, &original).unwrap_err(),
            DecoderPoolError::GrantAlreadyBound {
                child_index: 0,
                decoder_id: decoder_id.clone(),
                slot_generation: slot_generations[0].as_uuid(),
            }
        );
        for rooms in [vec![101, 103], vec![102, 101]] {
            let altered = issue_grant(
                &pool,
                &owner,
                decoder_id.clone(),
                Uuid::new_v4(),
                slot_generations.clone(),
                rooms,
                vec![child_accounting(), child_accounting()],
            );
            assert_eq!(
                pool.bind_grant(&owner, &altered).unwrap_err(),
                DecoderPoolError::GrantGenerationRebound {
                    child_index: 0,
                    decoder_id: decoder_id.clone(),
                    slot_generation: slot_generations[0].as_uuid(),
                }
            );
        }
    }

    #[test]
    fn slot_generations_are_decoder_process_local() {
        let pool = pool(2);
        let first_decoder = decoder_id("decode-0");
        let second_decoder = decoder_id("decode-1");
        pool.register(replica_with_id(first_decoder.clone(), "packed-v1"))
            .unwrap();
        pool.register(replica_with_id(second_decoder.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let slot_generation = DecoderSlotGeneration::new(Uuid::new_v4());
        let first_grant = issue_grant(
            &pool,
            &owner,
            first_decoder,
            Uuid::new_v4(),
            vec![slot_generation],
            vec![101],
            scalar_accounting(),
        );
        let first = pool.bind_grant(&owner, &first_grant).unwrap();
        release_before_dispatch(&pool, first, RetryDisposition::DecoderFailed).unwrap();

        let second_grant = issue_grant(
            &pool,
            &owner,
            second_decoder.clone(),
            Uuid::new_v4(),
            vec![slot_generation],
            vec![101],
            scalar_accounting(),
        );
        let second = pool.bind_grant(&owner, &second_grant).unwrap();
        assert_eq!(second.decoder_id(), &second_decoder);
        release_before_dispatch(&pool, second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn active_bootstrap_room_cannot_be_owned_twice_on_one_decoder() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = pool.admission_candidates(&first_owner).unwrap().remove(0);
        let first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let first = pool.bind_grant(&first_owner, &first_grant).unwrap();
        let second_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        assert_eq!(
            pool.bind_grant(&second_owner, &second_grant).unwrap_err(),
            DecoderPoolError::GrantRoomInUse {
                decoder_id,
                room: 700,
            }
        );
        release_before_dispatch(&pool, first, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn equal_active_room_numbers_are_valid_on_separate_decoders() {
        let pool = pool(2);
        let first_decoder = decoder_id("decode-0");
        let second_decoder = decoder_id("decode-1");
        pool.register(replica_with_id(first_decoder.clone(), "packed-v1"))
            .unwrap();
        pool.register(replica_with_id(second_decoder.clone(), "packed-v1"))
            .unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let first_grant = issue_grant(
            &pool,
            &first_owner,
            first_decoder.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let second_grant = issue_grant(
            &pool,
            &second_owner,
            second_decoder.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );

        let first = pool.bind_grant(&first_owner, &first_grant).unwrap();
        let second = pool.bind_grant(&second_owner, &second_grant).unwrap();
        assert_eq!(first.decoder_id(), &first_decoder);
        assert_eq!(second.decoder_id(), &second_decoder);
        release_before_dispatch(&pool, first, RetryDisposition::Terminal).unwrap();
        release_before_dispatch(&pool, second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn request_owner_finalization_prevents_retry_history_bleed() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();

        let mut first_owner = pool.begin_request("reused-id").unwrap();
        let first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let failed_decoder = first.decoder_id().clone();
        release_before_dispatch(&pool, first, RetryDisposition::DecoderFailed).unwrap();
        pool.finalize_request(&mut first_owner).unwrap();

        let second_owner = pool.begin_request("reused-id").unwrap();
        let second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();
        assert_eq!(second.decoder_id(), &failed_decoder);
        release_before_dispatch(&pool, second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn terminal_release_closes_the_logical_request_chain() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();

        assert_eq!(
            pool.admission_candidates(&owner).unwrap_err(),
            DecoderPoolError::RequestChainTerminal("request".to_string())
        );
        pool.finalize_request(&mut owner).unwrap();
    }

    #[test]
    fn dropped_owner_is_reaped_after_its_active_cohort_terminates() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        drop(owner);
        assert_eq!(pool.snapshot().active_logical_requests, 1);
        release_before_dispatch(&pool, cohort, RetryDisposition::DecoderFailed).unwrap();
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        let replacement = pool.begin_request("request").unwrap();
        let replacement_cohort = bind_next(&pool, &replacement, scalar_accounting()).unwrap();
        release_before_dispatch(&pool, replacement_cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn retryable_non_decoder_failure_preserves_destination_eligibility() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let first = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let decoder_id = first.decoder_id().clone();
        release_before_dispatch(&pool, first, RetryDisposition::Retryable).unwrap();

        let retry = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        assert_eq!(retry.decoder_id(), &decoder_id);
        release_before_dispatch(&pool, retry, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn exhausted_retry_chain_is_explicit_and_finalizable() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        release_before_dispatch(&pool, cohort, RetryDisposition::DecoderFailed).unwrap();
        assert_eq!(
            pool.admission_candidates(&owner).unwrap_err(),
            DecoderPoolError::RetryAlternativesExhausted
        );
        pool.finalize_request(&mut owner).unwrap();

        let replacement = pool.begin_request("request").unwrap();
        let replacement_cohort = bind_next(&pool, &replacement, scalar_accounting()).unwrap();
        release_before_dispatch(&pool, replacement_cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn allocator_candidates_never_fall_back_to_a_failed_decoder() {
        let pool = pool(2);
        let small_scheduling_scale = scheduling(1, 2_048);
        for name in ["decode-0", "decode-1"] {
            pool.register(
                DecoderReplicaMetadata::new(
                    decoder_id(name),
                    1,
                    compatibility("packed-v1"),
                    small_scheduling_scale,
                )
                .unwrap(),
            )
            .unwrap();
        }

        let retry_owner = pool.begin_request("retry").unwrap();
        let failed = bind_next(&pool, &retry_owner, scalar_accounting()).unwrap();
        let failed_decoder = failed.decoder_id().clone();
        release_before_dispatch(&pool, failed, RetryDisposition::DecoderFailed).unwrap();
        pool.set_availability(&failed_decoder, DecoderAvailability::Draining)
            .unwrap();

        let blocker_owner = pool.begin_request("blocker").unwrap();
        let blocker = bind_next(&pool, &blocker_owner, scalar_accounting()).unwrap();
        assert_ne!(blocker.decoder_id(), &failed_decoder);
        assert_eq!(
            pool.admission_candidates(&retry_owner).unwrap(),
            vec![blocker.decoder_id().clone()]
        );

        let available_decoder = blocker.decoder_id().clone();
        release_before_dispatch(&pool, blocker, RetryDisposition::Terminal).unwrap();
        let retry = bind_next(&pool, &retry_owner, scalar_accounting()).unwrap();
        assert_eq!(retry.decoder_id(), &available_decoder);
        release_before_dispatch(&pool, retry, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn quiescing_phase_and_counter_change_as_one_operation() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        pool.mark_dispatched(&mut cohort).unwrap();
        pool.begin_quiescence(&mut cohort).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 1);
        release_after_abort(&pool, cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn removal_requires_draining_and_zero_owned_cohorts() {
        let pool = pool(4);
        let decoder_id = decoder_id("decode-0");
        pool.register(replica_with_id(decoder_id.clone(), "packed-v1"))
            .unwrap();
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderNotDraining { .. })
        ));

        let owner = pool.begin_request("request").unwrap();
        let cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        pool.set_availability(&decoder_id, DecoderAvailability::Draining)
            .unwrap();
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderInUse { .. })
        ));
        release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();
        pool.remove(&decoder_id).unwrap();
    }

    #[test]
    fn valid_allocator_grant_is_not_vetoed_by_advisory_scales() {
        let pool = pool(2);
        pool.register(
            DecoderReplicaMetadata::new(
                decoder_id("decode-0"),
                1,
                compatibility("packed-v1"),
                scheduling(1, 1),
            )
            .unwrap(),
        )
        .unwrap();
        let owner = pool.begin_request("batch").unwrap();
        let accounting = vec![
            DecoderGrantChildAccounting::new(10_000, 1),
            DecoderGrantChildAccounting::new(10_000, 1),
            DecoderGrantChildAccounting::new(10_000, 1),
        ];
        let cohort = bind_next(&pool, &owner, accounting).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_child_requests, 3);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 30_000);
        release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn decode_progress_updates_work_without_releasing_child_or_kv_ownership() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        pool.mark_dispatched(&mut cohort).unwrap();
        pool.observe_decode_progress(&cohort, 100).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 28);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 1_024);
        assert_eq!(snapshot.replicas[0].active_child_requests, 1);
        assert!(matches!(
            pool.observe_decode_progress(&cohort, 29),
            Err(DecoderPoolError::InvalidProgress { .. })
        ));
        release_after_completion(&pool, cohort).unwrap();
    }

    #[test]
    fn concurrent_grant_binding_never_reuses_active_rooms() {
        let pool = Arc::new(pool(2));
        pool.register(
            DecoderReplicaMetadata::new(
                decoder_id("decode-0"),
                1,
                compatibility("packed-v1"),
                scheduling(32, 320),
            )
            .unwrap(),
        )
        .unwrap();

        let handles: Vec<_> = (0..64)
            .map(|index| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    let owner = pool.begin_request(format!("request-{index}"))?;
                    let accounting = vec![DecoderGrantChildAccounting::new(10, 1)];
                    let cohort = bind_next(&pool, &owner, accounting)?;
                    Ok::<_, DecoderPoolError>((owner, cohort))
                })
            })
            .collect();
        let mut admitted = Vec::new();
        for handle in handles {
            match handle.join().unwrap() {
                Ok(pair) => admitted.push(pair),
                Err(error) => panic!("unexpected admission error: {error}"),
            }
        }

        assert_eq!(admitted.len(), 64);
        assert_eq!(
            admitted
                .iter()
                .flat_map(|(_, cohort)| cohort.bootstrap_rooms().iter().copied())
                .collect::<HashSet<_>>()
                .len(),
            admitted.len()
        );
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 64);
        assert_eq!(snapshot.replicas[0].active_child_requests, 64);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 640);

        for (mut owner, cohort) in admitted {
            release_before_dispatch(&pool, cohort, RetryDisposition::Terminal).unwrap();
            pool.finalize_request(&mut owner).unwrap();
        }
    }
}
