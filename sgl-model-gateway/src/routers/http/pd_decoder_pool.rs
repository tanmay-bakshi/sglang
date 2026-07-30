//! Request-affine admission and lifecycle accounting for disaggregated decoders.
//!
//! A cohort is one stock PDRouter request: it targets one decoder and owns one
//! bootstrap room per child request. Once its first external submission begins,
//! every child remains pinned to that decoder until the whole cohort completes or
//! every transfer operation is proven quiescent.
//!
//! The metadata checks in this module are eligibility checks only. They do not
//! prove asymmetric TP slicing, DMA lane selection, destination correctness, or
//! transfer quiescence. The caller must provide exact demand bounds, a single
//! authoritative routing process, and externally verified handle-bound
//! quiescence before invoking `confirm_quiesced`.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt,
    num::NonZeroUsize,
    sync::Arc,
};

use parking_lot::Mutex;
use thiserror::Error;
use uuid::Uuid;

const MAX_BOOTSTRAP_SEQUENCE: u64 = u32::MAX as u64;

/// Stable identity for one decoder process generation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DecoderId(Arc<str>);

impl DecoderId {
    /// Construct a decoder identity.
    pub fn new(value: impl Into<String>) -> Result<Self, DecoderPoolError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DecoderPoolError::InvalidConfiguration(
                "decoder identity cannot be empty".to_string(),
            ));
        }
        Ok(Self(Arc::from(value)))
    }

    /// Return the identity as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DecoderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

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

/// Hard child-request admission limits and relative service weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderCapacity {
    max_concurrent_child_requests: NonZeroUsize,
    max_kv_tokens: NonZeroUsize,
    service_weight: NonZeroUsize,
}

impl DecoderCapacity {
    /// Construct decoder capacity.
    pub fn new(
        max_concurrent_child_requests: usize,
        max_kv_tokens: usize,
        service_weight: usize,
    ) -> Result<Self, DecoderPoolError> {
        Ok(Self {
            max_concurrent_child_requests: NonZeroUsize::new(max_concurrent_child_requests)
                .ok_or_else(|| {
                    DecoderPoolError::InvalidConfiguration(
                        "max concurrent child requests must be nonzero".to_string(),
                    )
                })?,
            max_kv_tokens: NonZeroUsize::new(max_kv_tokens).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration("max KV tokens must be nonzero".to_string())
            })?,
            service_weight: NonZeroUsize::new(service_weight).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration("service weight must be nonzero".to_string())
            })?,
        })
    }

    /// Maximum concurrently owned child requests across all cohorts.
    pub fn max_concurrent_child_requests(&self) -> usize {
        self.max_concurrent_child_requests.get()
    }

    /// Maximum conservatively reserved KV tokens.
    pub fn max_kv_tokens(&self) -> usize {
        self.max_kv_tokens.get()
    }
}

/// Engine-declared metadata and configured admission limits for one TP1 decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReplicaMetadata {
    id: DecoderId,
    declared_decode_tp_size: NonZeroUsize,
    compatibility: EngineCompatibilityMetadata,
    capacity: DecoderCapacity,
}

impl DecoderReplicaMetadata {
    /// Construct decoder metadata without asserting transport correctness.
    pub fn new(
        id: DecoderId,
        declared_decode_tp_size: usize,
        compatibility: EngineCompatibilityMetadata,
        capacity: DecoderCapacity,
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
            capacity,
        })
    }

    /// Return the decoder process-generation identity.
    pub fn id(&self) -> &DecoderId {
        &self.id
    }
}

/// Conservative resource bound for one child request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderDemand {
    kv_tokens: NonZeroUsize,
    decode_tokens: NonZeroUsize,
}

impl DecoderDemand {
    /// Construct a caller-supplied child demand bound.
    pub fn new(kv_tokens: usize, decode_tokens: usize) -> Result<Self, DecoderPoolError> {
        Ok(Self {
            kv_tokens: NonZeroUsize::new(kv_tokens).ok_or_else(|| {
                DecoderPoolError::InvalidDemand("KV token demand must be nonzero".to_string())
            })?,
            decode_tokens: NonZeroUsize::new(decode_tokens).ok_or_else(|| {
                DecoderPoolError::InvalidDemand("decode token demand must be nonzero".to_string())
            })?,
        })
    }

    /// Conservatively reserved KV tokens.
    pub fn kv_tokens(&self) -> usize {
        self.kv_tokens.get()
    }

    /// Upper bound on generated tokens.
    pub fn decode_tokens(&self) -> usize {
        self.decode_tokens.get()
    }
}

/// Immutable aggregate demand for one PDRouter batch or scalar request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCohortDemand {
    children: Arc<[DecoderDemand]>,
    total_kv_tokens: usize,
    total_decode_tokens: usize,
}

impl DecoderCohortDemand {
    /// Construct aggregate demand while preserving child order.
    pub fn new(children: Vec<DecoderDemand>) -> Result<Self, DecoderPoolError> {
        if children.is_empty() {
            return Err(DecoderPoolError::InvalidDemand(
                "a decoder cohort must contain at least one child".to_string(),
            ));
        }
        let mut total_kv_tokens = 0usize;
        let mut total_decode_tokens = 0usize;
        for child in &children {
            total_kv_tokens = total_kv_tokens
                .checked_add(child.kv_tokens.get())
                .ok_or_else(|| {
                    DecoderPoolError::InvalidDemand(
                        "cohort KV token demand overflows usize".to_string(),
                    )
                })?;
            total_decode_tokens = total_decode_tokens
                .checked_add(child.decode_tokens.get())
                .ok_or_else(|| {
                    DecoderPoolError::InvalidDemand(
                        "cohort decode token demand overflows usize".to_string(),
                    )
                })?;
        }
        Ok(Self {
            children: Arc::from(children),
            total_kv_tokens,
            total_decode_tokens,
        })
    }

    /// Construct a scalar cohort.
    pub fn single(child: DecoderDemand) -> Self {
        Self {
            children: Arc::from([child]),
            total_kv_tokens: child.kv_tokens.get(),
            total_decode_tokens: child.decode_tokens.get(),
        }
    }

    /// Ordered child demands corresponding one-to-one with bootstrap rooms.
    pub fn children(&self) -> &[DecoderDemand] {
        &self.children
    }

    /// Number of child requests owned by the cohort.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Aggregate KV reservation.
    pub fn total_kv_tokens(&self) -> usize {
        self.total_kv_tokens
    }

    /// Aggregate upper bound on generated tokens.
    pub fn total_decode_tokens(&self) -> usize {
        self.total_decode_tokens
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
    decoder_id: DecoderId,
    bootstrap_rooms: Arc<[u64]>,
    phase: CohortPhase,
}

impl DecoderAssignmentCohort {
    /// Stable identity of the selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Ordered rooms corresponding one-to-one with the original child order.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
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
    pub capacity: DecoderCapacity,
}

/// Immutable pool accounting snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderPoolSnapshot {
    pub declared_prefill_tp_size: usize,
    pub active_logical_requests: usize,
    pub replicas: Vec<DecoderReplicaSnapshot>,
}

/// Decoder-pool ownership, lifecycle, and admission failures.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecoderPoolError {
    #[error("invalid decoder-pool configuration: {0}")]
    InvalidConfiguration(String),
    #[error("invalid decoder demand: {0}")]
    InvalidDemand(String),
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
    #[error("every ready decoder is at its configured admission limit")]
    AtCapacity,
    #[error("unfailed retry alternatives exist but are temporarily at capacity")]
    RetryAlternativesTemporarilyFull,
    #[error("the logical request has exhausted every registered retry alternative")]
    RetryAlternativesExhausted,
    #[error("bootstrap-room sequence exhausted for this pool process generation")]
    BootstrapRoomExhausted,
    #[error("assignment capability was not issued by this decoder pool")]
    ForeignAssignment,
    #[error("assignment {0} is unknown or already terminal")]
    UnknownAssignment(Uuid),
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
    #[error("decoder {decoder_id} capacity is below its current reservation")]
    CapacityBelowReservation { decoder_id: DecoderId },
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

impl ReplicaState {
    fn can_admit(&self, demand: &DecoderCohortDemand) -> bool {
        if self.availability != DecoderAvailability::Ready {
            return false;
        }
        let Some(active_child_requests) =
            self.active_child_requests.checked_add(demand.child_count())
        else {
            return false;
        };
        if active_child_requests > self.metadata.capacity.max_concurrent_child_requests.get() {
            return false;
        }
        if self
            .remaining_decode_tokens
            .checked_add(demand.total_decode_tokens())
            .is_none()
        {
            return false;
        }
        let Some(kv_tokens) = self
            .reserved_kv_tokens
            .checked_add(demand.total_kv_tokens())
        else {
            return false;
        };
        kv_tokens <= self.metadata.capacity.max_kv_tokens.get()
    }
}

#[derive(Debug)]
struct RequestChainRecord {
    request_id: Arc<str>,
    phase: RequestChainPhase,
    owner_alive: bool,
    active_assignment: Option<Uuid>,
    failed_decoders: HashSet<DecoderId>,
}

#[derive(Debug)]
struct AssignmentRecord {
    chain_id: Uuid,
    decoder_id: DecoderId,
    bootstrap_rooms: Arc<[u64]>,
    phase: CohortPhase,
    child_count: usize,
    kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Debug)]
struct PoolState {
    declared_prefill_tp_size: NonZeroUsize,
    compatibility: EngineCompatibilityMetadata,
    room_prefix: u64,
    next_room_sequence: u64,
    replicas: HashMap<DecoderId, ReplicaState>,
    request_chains: HashMap<Uuid, RequestChainRecord>,
    active_request_ids: HashMap<Arc<str>, Uuid>,
    assignments: HashMap<Uuid, AssignmentRecord>,
}

impl PoolState {
    fn allocate_bootstrap_rooms(
        &mut self,
        child_count: usize,
    ) -> Result<Arc<[u64]>, DecoderPoolError> {
        let child_count =
            u64::try_from(child_count).map_err(|_| DecoderPoolError::BootstrapRoomExhausted)?;
        let last_offset = child_count.checked_sub(1).ok_or_else(|| {
            DecoderPoolError::InvalidDemand(
                "a decoder cohort must contain at least one child".to_string(),
            )
        })?;
        let last_sequence = self
            .next_room_sequence
            .checked_add(last_offset)
            .ok_or(DecoderPoolError::BootstrapRoomExhausted)?;
        if last_sequence > MAX_BOOTSTRAP_SEQUENCE {
            return Err(DecoderPoolError::BootstrapRoomExhausted);
        }

        let capacity =
            usize::try_from(child_count).map_err(|_| DecoderPoolError::BootstrapRoomExhausted)?;
        let mut rooms = Vec::with_capacity(capacity);
        for sequence in self.next_room_sequence..=last_sequence {
            rooms.push((self.room_prefix << 32) | sequence);
        }
        self.next_room_sequence = last_sequence + 1;
        Ok(Arc::from(rooms))
    }
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
        declared_prefill_tp_size: usize,
        compatibility: EngineCompatibilityMetadata,
    ) -> Result<Self, DecoderPoolError> {
        let declared_prefill_tp_size =
            NonZeroUsize::new(declared_prefill_tp_size).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration(
                    "declared prefill tensor parallel size must be nonzero".to_string(),
                )
            })?;
        let room_prefix = u64::from(rand::random::<u32>() & 0x7fff_ffff);
        Ok(Self {
            inner: Arc::new(DecoderPoolInner {
                pool_id: Uuid::new_v4(),
                state: Mutex::new(PoolState {
                    declared_prefill_tp_size,
                    compatibility,
                    room_prefix,
                    next_room_sequence: 0,
                    replicas: HashMap::new(),
                    request_chains: HashMap::new(),
                    active_request_ids: HashMap::new(),
                    assignments: HashMap::new(),
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
            return Err(DecoderPoolError::InvalidDemand(
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

    /// Update hard limits after an authoritative decoder capacity report.
    pub fn update_capacity(
        &self,
        decoder_id: &DecoderId,
        capacity: DecoderCapacity,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let replica = state
            .replicas
            .get_mut(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        if replica.active_child_requests > capacity.max_concurrent_child_requests.get()
            || replica.reserved_kv_tokens > capacity.max_kv_tokens.get()
        {
            return Err(DecoderPoolError::CapacityBelowReservation {
                decoder_id: decoder_id.clone(),
            });
        }
        replica.metadata.capacity = capacity;
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

    /// Atomically select one decoder and reserve an immutable child cohort.
    pub fn reserve(
        &self,
        request: &LogicalRequestOwner,
        demand: DecoderCohortDemand,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        self.validate_request_owner(request)?;
        let mut state = self.inner.state.lock();
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

        let selected_id = select_decoder(&state.replicas, &failed_decoders, &demand)?;
        let bootstrap_rooms = state.allocate_bootstrap_rooms(demand.child_count())?;
        let assignment_id = Uuid::new_v4();

        let replica = state
            .replicas
            .get_mut(&selected_id)
            .expect("selected decoder disappeared while pool lock was held");
        replica.active_cohorts += 1;
        replica.active_child_requests += demand.child_count();
        replica.reserved_kv_tokens += demand.total_kv_tokens();
        replica.remaining_decode_tokens += demand.total_decode_tokens();

        state.assignments.insert(
            assignment_id,
            AssignmentRecord {
                chain_id: request.chain_id,
                decoder_id: selected_id.clone(),
                bootstrap_rooms: Arc::clone(&bootstrap_rooms),
                phase: CohortPhase::Reserved,
                child_count: demand.child_count(),
                kv_tokens: demand.total_kv_tokens(),
                remaining_decode_tokens: demand.total_decode_tokens(),
            },
        );
        state
            .request_chains
            .get_mut(&request.chain_id)
            .expect("request chain disappeared while pool lock was held")
            .active_assignment = Some(assignment_id);

        Ok(DecoderAssignmentCohort {
            pool_id: self.inner.pool_id,
            chain_id: request.chain_id,
            assignment_id,
            decoder_id: selected_id,
            bootstrap_rooms,
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
            (record.decoder_id.clone(), record.remaining_decode_tokens)
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
            record.decoder_id.clone()
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
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        self.release(cohort, CohortPhase::Reserved, disposition)
    }

    /// Terminalize a successfully completed child cohort.
    pub fn complete(&self, cohort: DecoderAssignmentCohort) -> Result<(), DecoderPoolError> {
        self.release(cohort, CohortPhase::Dispatched, RetryDisposition::Terminal)
    }

    /// Release every child after externally verified handle-bound quiescence.
    ///
    /// This state container does not create or verify the proof. Production
    /// integration must do so before calling this method.
    pub fn confirm_quiesced(
        &self,
        cohort: DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
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
                capacity: replica.metadata.capacity,
            })
            .collect();
        replicas.sort_by(|left, right| left.id.cmp(&right.id));
        DecoderPoolSnapshot {
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
            .get_mut(&record.decoder_id)
            .expect("assigned decoder disappeared while pool lock was held");
        replica.active_cohorts -= 1;
        replica.active_child_requests -= record.child_count;
        replica.reserved_kv_tokens -= record.kv_tokens;
        replica.remaining_decode_tokens -= record.remaining_decode_tokens;
        if expected == CohortPhase::Quiescing {
            replica.quiescing_cohorts -= 1;
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
                    chain.failed_decoders.insert(record.decoder_id);
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

fn select_decoder(
    replicas: &HashMap<DecoderId, ReplicaState>,
    failed_decoders: &HashSet<DecoderId>,
    demand: &DecoderCohortDemand,
) -> Result<DecoderId, DecoderPoolError> {
    if failed_decoders.is_empty() {
        let ready: Vec<&ReplicaState> = replicas
            .values()
            .filter(|replica| replica.availability == DecoderAvailability::Ready)
            .collect();
        if ready.is_empty() {
            return Err(DecoderPoolError::NoReadyDecoder);
        }
        return ready
            .into_iter()
            .filter(|replica| replica.can_admit(demand))
            .min_by(|left, right| compare_projected_load(left, right, demand))
            .map(|replica| replica.metadata.id.clone())
            .ok_or(DecoderPoolError::AtCapacity);
    }

    let unfailed: Vec<&ReplicaState> = replicas
        .values()
        .filter(|replica| !failed_decoders.contains(&replica.metadata.id))
        .collect();
    if unfailed.is_empty() {
        return Err(DecoderPoolError::RetryAlternativesExhausted);
    }
    let ready: Vec<&ReplicaState> = unfailed
        .into_iter()
        .filter(|replica| replica.availability == DecoderAvailability::Ready)
        .collect();
    if ready.is_empty() {
        return Err(DecoderPoolError::NoReadyDecoder);
    }
    ready
        .into_iter()
        .filter(|replica| replica.can_admit(demand))
        .min_by(|left, right| compare_projected_load(left, right, demand))
        .map(|replica| replica.metadata.id.clone())
        .ok_or(DecoderPoolError::RetryAlternativesTemporarilyFull)
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
    if record.chain_id != cohort.chain_id
        || record.decoder_id != cohort.decoder_id
        || record.bootstrap_rooms != cohort.bootstrap_rooms
    {
        return Err(DecoderPoolError::ForeignAssignment);
    }
    Ok(())
}

fn compare_projected_load(
    left: &ReplicaState,
    right: &ReplicaState,
    demand: &DecoderCohortDemand,
) -> Ordering {
    compare_ratio(
        left.remaining_decode_tokens + demand.total_decode_tokens(),
        left.metadata.capacity.service_weight.get(),
        right.remaining_decode_tokens + demand.total_decode_tokens(),
        right.metadata.capacity.service_weight.get(),
    )
    .then_with(|| {
        compare_ratio(
            left.reserved_kv_tokens + demand.total_kv_tokens(),
            left.metadata.capacity.max_kv_tokens.get(),
            right.reserved_kv_tokens + demand.total_kv_tokens(),
            right.metadata.capacity.max_kv_tokens.get(),
        )
    })
    .then_with(|| {
        compare_ratio(
            left.active_child_requests + demand.child_count(),
            left.metadata.capacity.max_concurrent_child_requests.get(),
            right.active_child_requests + demand.child_count(),
            right.metadata.capacity.max_concurrent_child_requests.get(),
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
    use std::{collections::HashSet, sync::Arc, thread};

    use super::*;

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

    fn capacity(max_children: usize, max_tokens: usize) -> DecoderCapacity {
        DecoderCapacity::new(max_children, max_tokens, 1).unwrap()
    }

    fn replica(name: &str, protocol: &str) -> DecoderReplicaMetadata {
        DecoderReplicaMetadata::new(
            DecoderId::new(name).unwrap(),
            1,
            compatibility(protocol),
            capacity(32, 32_000),
        )
        .unwrap()
    }

    fn pool(declared_prefill_tp_size: usize) -> DecoderPool {
        DecoderPool::new(declared_prefill_tp_size, compatibility("packed-v1")).unwrap()
    }

    fn child_demand() -> DecoderDemand {
        DecoderDemand::new(1_024, 128).unwrap()
    }

    fn scalar_demand() -> DecoderCohortDemand {
        DecoderCohortDemand::single(child_demand())
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
    fn rejects_ineligible_declared_decoder_metadata() {
        let pool = pool(4);
        let tp2 = DecoderReplicaMetadata::new(
            DecoderId::new("decode-tp2").unwrap(),
            2,
            compatibility("packed-v1"),
            capacity(32, 32_000),
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
            let cohort = pool.reserve(&request, scalar_demand()).unwrap();
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
            pool.finish_before_dispatch(cohort, RetryDisposition::Terminal)
                .unwrap();
            pool.finalize_request(&mut request).unwrap();
        }
    }

    #[test]
    fn batch_cohort_preserves_room_order_and_releases_all_children_together() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let request = pool.begin_request("batch").unwrap();
        let demand = DecoderCohortDemand::new(vec![
            DecoderDemand::new(100, 10).unwrap(),
            DecoderDemand::new(200, 20).unwrap(),
            DecoderDemand::new(300, 30).unwrap(),
        ])
        .unwrap();
        let mut cohort = pool.reserve(&request, demand).unwrap();
        assert_eq!(cohort.bootstrap_rooms().len(), 3);
        assert_eq!(cohort.bootstrap_rooms()[1], cohort.bootstrap_rooms()[0] + 1);
        assert_eq!(cohort.bootstrap_rooms()[2], cohort.bootstrap_rooms()[1] + 1);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].active_child_requests, 3);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 600);
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 60);

        pool.mark_dispatched(&mut cohort).unwrap();
        pool.begin_quiescence(&mut cohort).unwrap();
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 1);
        pool.confirm_quiesced(cohort, RetryDisposition::Terminal)
            .unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 0);
        assert_eq!(snapshot.replicas[0].active_child_requests, 0);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
    }

    #[test]
    fn request_owner_finalization_prevents_retry_history_bleed() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();

        let mut first_owner = pool.begin_request("reused-id").unwrap();
        let first = pool.reserve(&first_owner, scalar_demand()).unwrap();
        let failed_decoder = first.decoder_id().clone();
        pool.finish_before_dispatch(first, RetryDisposition::DecoderFailed)
            .unwrap();
        pool.finalize_request(&mut first_owner).unwrap();

        let second_owner = pool.begin_request("reused-id").unwrap();
        let second = pool.reserve(&second_owner, scalar_demand()).unwrap();
        assert_eq!(second.decoder_id(), &failed_decoder);
        pool.finish_before_dispatch(second, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn terminal_release_closes_the_logical_request_chain() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let cohort = pool.reserve(&owner, scalar_demand()).unwrap();
        pool.finish_before_dispatch(cohort, RetryDisposition::Terminal)
            .unwrap();

        assert_eq!(
            pool.reserve(&owner, scalar_demand()).unwrap_err(),
            DecoderPoolError::RequestChainTerminal("request".to_string())
        );
        pool.finalize_request(&mut owner).unwrap();
    }

    #[test]
    fn dropped_owner_is_reaped_after_its_active_cohort_terminates() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let cohort = pool.reserve(&owner, scalar_demand()).unwrap();
        drop(owner);
        assert_eq!(pool.snapshot().active_logical_requests, 1);
        pool.finish_before_dispatch(cohort, RetryDisposition::DecoderFailed)
            .unwrap();
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        let replacement = pool.begin_request("request").unwrap();
        let replacement_cohort = pool.reserve(&replacement, scalar_demand()).unwrap();
        pool.finish_before_dispatch(replacement_cohort, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn retryable_non_decoder_failure_preserves_destination_eligibility() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let first = pool.reserve(&owner, scalar_demand()).unwrap();
        let decoder_id = first.decoder_id().clone();
        pool.finish_before_dispatch(first, RetryDisposition::Retryable)
            .unwrap();

        let retry = pool.reserve(&owner, scalar_demand()).unwrap();
        assert_eq!(retry.decoder_id(), &decoder_id);
        pool.finish_before_dispatch(retry, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn exhausted_retry_chain_is_explicit_and_finalizable() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let cohort = pool.reserve(&owner, scalar_demand()).unwrap();
        pool.finish_before_dispatch(cohort, RetryDisposition::DecoderFailed)
            .unwrap();
        assert_eq!(
            pool.reserve(&owner, scalar_demand()).unwrap_err(),
            DecoderPoolError::RetryAlternativesExhausted
        );
        pool.finalize_request(&mut owner).unwrap();

        let replacement = pool.begin_request("request").unwrap();
        let replacement_cohort = pool.reserve(&replacement, scalar_demand()).unwrap();
        pool.finish_before_dispatch(replacement_cohort, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn temporarily_full_retry_alternative_never_falls_back_to_failed_decoder() {
        let pool = pool(2);
        let small_capacity = capacity(1, 2_048);
        for name in ["decode-0", "decode-1"] {
            pool.register(
                DecoderReplicaMetadata::new(
                    DecoderId::new(name).unwrap(),
                    1,
                    compatibility("packed-v1"),
                    small_capacity,
                )
                .unwrap(),
            )
            .unwrap();
        }

        let retry_owner = pool.begin_request("retry").unwrap();
        let failed = pool.reserve(&retry_owner, scalar_demand()).unwrap();
        let failed_decoder = failed.decoder_id().clone();
        pool.finish_before_dispatch(failed, RetryDisposition::DecoderFailed)
            .unwrap();
        pool.set_availability(&failed_decoder, DecoderAvailability::Draining)
            .unwrap();

        let blocker_owner = pool.begin_request("blocker").unwrap();
        let blocker = pool.reserve(&blocker_owner, scalar_demand()).unwrap();
        assert_ne!(blocker.decoder_id(), &failed_decoder);
        assert_eq!(
            pool.reserve(&retry_owner, scalar_demand()).unwrap_err(),
            DecoderPoolError::RetryAlternativesTemporarilyFull
        );

        let available_decoder = blocker.decoder_id().clone();
        pool.finish_before_dispatch(blocker, RetryDisposition::Terminal)
            .unwrap();
        let retry = pool.reserve(&retry_owner, scalar_demand()).unwrap();
        assert_eq!(retry.decoder_id(), &available_decoder);
        pool.finish_before_dispatch(retry, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn quiescing_phase_and_counter_change_as_one_operation() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = pool.reserve(&owner, scalar_demand()).unwrap();
        pool.mark_dispatched(&mut cohort).unwrap();
        pool.begin_quiescence(&mut cohort).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 1);
        pool.confirm_quiesced(cohort, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn removal_requires_draining_and_zero_owned_cohorts() {
        let pool = pool(4);
        let decoder_id = DecoderId::new("decode-0").unwrap();
        pool.register(replica(decoder_id.as_str(), "packed-v1"))
            .unwrap();
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderNotDraining { .. })
        ));

        let owner = pool.begin_request("request").unwrap();
        let cohort = pool.reserve(&owner, scalar_demand()).unwrap();
        pool.set_availability(&decoder_id, DecoderAvailability::Draining)
            .unwrap();
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderInUse { .. })
        ));
        pool.finish_before_dispatch(cohort, RetryDisposition::Terminal)
            .unwrap();
        pool.remove(&decoder_id).unwrap();
    }

    #[test]
    fn batch_admission_counts_children_against_capacity() {
        let pool = pool(2);
        pool.register(
            DecoderReplicaMetadata::new(
                DecoderId::new("decode-0").unwrap(),
                1,
                compatibility("packed-v1"),
                capacity(2, 10_000),
            )
            .unwrap(),
        )
        .unwrap();
        let owner = pool.begin_request("batch").unwrap();
        let demand = DecoderCohortDemand::new(vec![
            DecoderDemand::new(1, 1).unwrap(),
            DecoderDemand::new(1, 1).unwrap(),
            DecoderDemand::new(1, 1).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            pool.reserve(&owner, demand).unwrap_err(),
            DecoderPoolError::AtCapacity
        );
    }

    #[test]
    fn decode_progress_updates_work_without_releasing_child_or_kv_ownership() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = pool.reserve(&owner, scalar_demand()).unwrap();
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
        pool.complete(cohort).unwrap();
    }

    #[test]
    fn concurrent_cohort_admission_never_oversubscribes_or_reuses_rooms() {
        let pool = Arc::new(pool(2));
        pool.register(
            DecoderReplicaMetadata::new(
                DecoderId::new("decode-0").unwrap(),
                1,
                compatibility("packed-v1"),
                capacity(32, 320),
            )
            .unwrap(),
        )
        .unwrap();

        let handles: Vec<_> = (0..64)
            .map(|index| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    let owner = pool.begin_request(format!("request-{index}"))?;
                    let cohort = pool.reserve(
                        &owner,
                        DecoderCohortDemand::single(DecoderDemand::new(10, 1).unwrap()),
                    )?;
                    Ok::<_, DecoderPoolError>((owner, cohort))
                })
            })
            .collect();
        let mut admitted = Vec::new();
        let mut capacity_errors = 0;
        for handle in handles {
            match handle.join().unwrap() {
                Ok(pair) => admitted.push(pair),
                Err(DecoderPoolError::AtCapacity) => capacity_errors += 1,
                Err(error) => panic!("unexpected admission error: {error}"),
            }
        }

        assert_eq!(admitted.len(), 32);
        assert_eq!(capacity_errors, 32);
        assert_eq!(
            admitted
                .iter()
                .flat_map(|(_, cohort)| cohort.bootstrap_rooms().iter().copied())
                .collect::<HashSet<_>>()
                .len(),
            admitted.len()
        );
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 32);
        assert_eq!(snapshot.replicas[0].active_child_requests, 32);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 320);

        for (mut owner, cohort) in admitted {
            pool.finish_before_dispatch(cohort, RetryDisposition::Terminal)
                .unwrap();
            pool.finalize_request(&mut owner).unwrap();
        }
    }
}
