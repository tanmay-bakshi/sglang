//! Request-affine admission and lifecycle accounting for disaggregated decoders.
//!
//! A bootstrap room names one transfer attempt and one decode destination. Once
//! an attempt is dispatched, it cannot be moved to another decoder. A retry gets
//! a new room only after the prior attempt has reached an authoritative terminal
//! state or transfer quiescence has been confirmed.

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

/// Wire and KV-layout properties that must match across a PD pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferCompatibility {
    model_fingerprint: Arc<str>,
    kv_layout_fingerprint: Arc<str>,
    kv_cache_dtype: Arc<str>,
    wire_protocol: Arc<str>,
    page_size: NonZeroUsize,
}

impl TransferCompatibility {
    /// Construct a compatibility identity reported by both engines.
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

/// Hard decoder admission limits and relative service weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderCapacity {
    max_concurrent_requests: NonZeroUsize,
    max_kv_tokens: NonZeroUsize,
    service_weight: NonZeroUsize,
}

impl DecoderCapacity {
    /// Construct decoder capacity.
    pub fn new(
        max_concurrent_requests: usize,
        max_kv_tokens: usize,
        service_weight: usize,
    ) -> Result<Self, DecoderPoolError> {
        Ok(Self {
            max_concurrent_requests: NonZeroUsize::new(max_concurrent_requests).ok_or_else(
                || {
                    DecoderPoolError::InvalidConfiguration(
                        "max concurrent requests must be nonzero".to_string(),
                    )
                },
            )?,
            max_kv_tokens: NonZeroUsize::new(max_kv_tokens).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration("max KV tokens must be nonzero".to_string())
            })?,
            service_weight: NonZeroUsize::new(service_weight).ok_or_else(|| {
                DecoderPoolError::InvalidConfiguration("service weight must be nonzero".to_string())
            })?,
        })
    }

    /// Maximum concurrently owned request leases.
    pub fn max_concurrent_requests(&self) -> usize {
        self.max_concurrent_requests.get()
    }

    /// Maximum conservatively reserved KV tokens.
    pub fn max_kv_tokens(&self) -> usize {
        self.max_kv_tokens.get()
    }
}

/// Static configuration for one TP1 decoder replica.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReplicaConfig {
    id: DecoderId,
    decode_tp_size: NonZeroUsize,
    compatibility: TransferCompatibility,
    capacity: DecoderCapacity,
}

impl DecoderReplicaConfig {
    /// Construct a decoder replica configuration.
    pub fn new(
        id: DecoderId,
        decode_tp_size: usize,
        compatibility: TransferCompatibility,
        capacity: DecoderCapacity,
    ) -> Result<Self, DecoderPoolError> {
        let decode_tp_size = NonZeroUsize::new(decode_tp_size).ok_or_else(|| {
            DecoderPoolError::InvalidConfiguration(
                "decode tensor parallel size must be nonzero".to_string(),
            )
        })?;
        Ok(Self {
            id,
            decode_tp_size,
            compatibility,
            capacity,
        })
    }

    /// Return the decoder identity.
    pub fn id(&self) -> &DecoderId {
        &self.id
    }
}

/// Conservative resource bound for one decode attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderDemand {
    kv_tokens: NonZeroUsize,
    decode_tokens: NonZeroUsize,
}

impl DecoderDemand {
    /// Construct request demand from exact prompt and generation bounds.
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

    /// Upper bound on remaining generated tokens.
    pub fn decode_tokens(&self) -> usize {
        self.decode_tokens.get()
    }
}

/// Whether a decoder accepts new assignments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderAvailability {
    Ready,
    Draining,
    Unavailable,
}

/// Whether a terminalized attempt may be retried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDisposition {
    Terminal,
    Retryable,
}

/// Current lifecycle of an issued assignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssignmentPhase {
    Reserved,
    Dispatched,
    Quiescing,
}

/// An authenticated, one-owner assignment capability.
///
/// This value is intentionally not cloneable. Dropping it without using one of
/// the terminal methods leaves the pool reservation in place, which is the safe
/// behavior when transfer quiescence is unknown.
#[derive(Debug)]
pub struct DecoderAssignment {
    pool_id: Uuid,
    assignment_id: Uuid,
    request_id: Arc<str>,
    decoder_id: DecoderId,
    bootstrap_room: u64,
    phase: AssignmentPhase,
}

impl DecoderAssignment {
    /// Stable identity of the selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Bootstrap room unique within this pool process generation.
    pub fn bootstrap_room(&self) -> u64 {
        self.bootstrap_room
    }

    /// Opaque identity for lifecycle telemetry and quiescence acknowledgements.
    pub fn assignment_id(&self) -> Uuid {
        self.assignment_id
    }
}

/// Immutable per-replica accounting snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReplicaSnapshot {
    pub id: DecoderId,
    pub availability: DecoderAvailability,
    pub active_assignments: usize,
    pub quiescing_assignments: usize,
    pub reserved_kv_tokens: usize,
    pub remaining_decode_tokens: usize,
    pub capacity: DecoderCapacity,
}

/// Immutable pool accounting snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderPoolSnapshot {
    pub prefill_tp_size: usize,
    pub replicas: Vec<DecoderReplicaSnapshot>,
}

/// Decoder-pool lifecycle and admission failures.
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
    #[error("decoder {decoder_id} has {active_assignments} active assignments")]
    DecoderInUse {
        decoder_id: DecoderId,
        active_assignments: usize,
    },
    #[error("decoder {decoder_id} is incompatible with this prefill pool: {reason}")]
    IncompatibleDecoder {
        decoder_id: DecoderId,
        reason: String,
    },
    #[error("request {0} already owns an active decoder assignment")]
    RequestAlreadyAssigned(String),
    #[error("no ready decoder replica is registered")]
    NoReadyDecoder,
    #[error("every ready decoder is at its configured admission limit")]
    AtCapacity,
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
    config: DecoderReplicaConfig,
    availability: DecoderAvailability,
    active_assignments: usize,
    quiescing_assignments: usize,
    reserved_kv_tokens: usize,
    remaining_decode_tokens: usize,
}

impl ReplicaState {
    fn can_admit(&self, demand: DecoderDemand) -> bool {
        if self.availability != DecoderAvailability::Ready {
            return false;
        }
        let Some(active_assignments) = self.active_assignments.checked_add(1) else {
            return false;
        };
        if active_assignments > self.config.capacity.max_concurrent_requests.get() {
            return false;
        }
        if self
            .remaining_decode_tokens
            .checked_add(demand.decode_tokens.get())
            .is_none()
        {
            return false;
        }
        let Some(kv_tokens) = self.reserved_kv_tokens.checked_add(demand.kv_tokens.get()) else {
            return false;
        };
        kv_tokens <= self.config.capacity.max_kv_tokens.get()
    }
}

#[derive(Debug)]
struct AssignmentRecord {
    request_id: Arc<str>,
    decoder_id: DecoderId,
    bootstrap_room: u64,
    phase: AssignmentPhase,
    kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Debug)]
struct PoolState {
    prefill_tp_size: NonZeroUsize,
    compatibility: TransferCompatibility,
    room_prefix: u64,
    next_room_sequence: u64,
    replicas: HashMap<DecoderId, ReplicaState>,
    assignments: HashMap<Uuid, AssignmentRecord>,
    active_requests: HashMap<Arc<str>, Uuid>,
    failed_decoders: HashMap<Arc<str>, HashSet<DecoderId>>,
}

impl PoolState {
    fn allocate_bootstrap_room(&mut self) -> Result<u64, DecoderPoolError> {
        if self.next_room_sequence > MAX_BOOTSTRAP_SEQUENCE {
            return Err(DecoderPoolError::BootstrapRoomExhausted);
        }
        let room = (self.room_prefix << 32) | self.next_room_sequence;
        self.next_room_sequence += 1;
        Ok(room)
    }
}

/// Atomic admission and lifecycle authority for a single prefill replica.
#[derive(Debug)]
pub struct DecoderPool {
    pool_id: Uuid,
    state: Mutex<PoolState>,
}

impl DecoderPool {
    /// Construct an empty pool for one prefill replica.
    pub fn new(
        prefill_tp_size: usize,
        compatibility: TransferCompatibility,
    ) -> Result<Self, DecoderPoolError> {
        let prefill_tp_size = NonZeroUsize::new(prefill_tp_size).ok_or_else(|| {
            DecoderPoolError::InvalidConfiguration(
                "prefill tensor parallel size must be nonzero".to_string(),
            )
        })?;
        let room_prefix = u64::from(rand::random::<u32>() & 0x7fff_ffff);
        Ok(Self {
            pool_id: Uuid::new_v4(),
            state: Mutex::new(PoolState {
                prefill_tp_size,
                compatibility,
                room_prefix,
                next_room_sequence: 0,
                replicas: HashMap::new(),
                assignments: HashMap::new(),
                active_requests: HashMap::new(),
                failed_decoders: HashMap::new(),
            }),
        })
    }

    /// Add a process generation to the decoder pool.
    pub fn register(&self, config: DecoderReplicaConfig) -> Result<(), DecoderPoolError> {
        let mut state = self.state.lock();
        if state.replicas.contains_key(&config.id) {
            return Err(DecoderPoolError::DuplicateDecoder(config.id));
        }
        if config.decode_tp_size.get() != 1 {
            return Err(DecoderPoolError::IncompatibleDecoder {
                decoder_id: config.id,
                reason: format!(
                    "decoder pool requires TP1 replicas, received TP{}",
                    config.decode_tp_size
                ),
            });
        }
        if !state
            .prefill_tp_size
            .get()
            .is_multiple_of(config.decode_tp_size.get())
        {
            return Err(DecoderPoolError::IncompatibleDecoder {
                decoder_id: config.id,
                reason: format!(
                    "prefill TP{} is not divisible by decode TP{}",
                    state.prefill_tp_size, config.decode_tp_size
                ),
            });
        }
        if config.compatibility != state.compatibility {
            return Err(DecoderPoolError::IncompatibleDecoder {
                decoder_id: config.id,
                reason: "model, KV layout, dtype, page size, or wire protocol differs".to_string(),
            });
        }

        state.replicas.insert(
            config.id.clone(),
            ReplicaState {
                config,
                availability: DecoderAvailability::Ready,
                active_assignments: 0,
                quiescing_assignments: 0,
                reserved_kv_tokens: 0,
                remaining_decode_tokens: 0,
            },
        );
        Ok(())
    }

    /// Change whether a registered decoder accepts new assignments.
    pub fn set_availability(
        &self,
        decoder_id: &DecoderId,
        availability: DecoderAvailability,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.state.lock();
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
        let mut state = self.state.lock();
        let replica = state
            .replicas
            .get_mut(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        if replica.active_assignments > capacity.max_concurrent_requests.get()
            || replica.reserved_kv_tokens > capacity.max_kv_tokens.get()
        {
            return Err(DecoderPoolError::CapacityBelowReservation {
                decoder_id: decoder_id.clone(),
            });
        }
        replica.config.capacity = capacity;
        Ok(())
    }

    /// Remove a drained process generation after all assignments are terminal.
    pub fn remove(&self, decoder_id: &DecoderId) -> Result<(), DecoderPoolError> {
        let mut state = self.state.lock();
        let replica = state
            .replicas
            .get(decoder_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(decoder_id.clone()))?;
        if replica.active_assignments > 0 {
            return Err(DecoderPoolError::DecoderInUse {
                decoder_id: decoder_id.clone(),
                active_assignments: replica.active_assignments,
            });
        }
        state.replicas.remove(decoder_id);
        Ok(())
    }

    /// Atomically select a decoder and reserve its configured resources.
    pub fn reserve(
        &self,
        request_id: impl Into<String>,
        demand: DecoderDemand,
    ) -> Result<DecoderAssignment, DecoderPoolError> {
        let request_id = request_id.into();
        if request_id.trim().is_empty() {
            return Err(DecoderPoolError::InvalidDemand(
                "request identity cannot be empty".to_string(),
            ));
        }
        let request_id: Arc<str> = Arc::from(request_id);
        let mut state = self.state.lock();
        if state.active_requests.contains_key(&request_id) {
            return Err(DecoderPoolError::RequestAlreadyAssigned(
                request_id.to_string(),
            ));
        }

        let ready_count = state
            .replicas
            .values()
            .filter(|replica| replica.availability == DecoderAvailability::Ready)
            .count();
        if ready_count == 0 {
            return Err(DecoderPoolError::NoReadyDecoder);
        }

        let failed_decoders = state.failed_decoders.get(&request_id);
        let mut candidates: Vec<&ReplicaState> = state
            .replicas
            .values()
            .filter(|replica| replica.can_admit(demand))
            .collect();
        if candidates.is_empty() {
            return Err(DecoderPoolError::AtCapacity);
        }

        let has_unfailed_candidate = candidates.iter().any(|replica| {
            failed_decoders
                .map(|failed| !failed.contains(&replica.config.id))
                .unwrap_or(true)
        });
        if has_unfailed_candidate {
            candidates.retain(|replica| {
                failed_decoders
                    .map(|failed| !failed.contains(&replica.config.id))
                    .unwrap_or(true)
            });
        }

        let selected_id = candidates
            .into_iter()
            .min_by(|left, right| compare_projected_load(left, right, demand))
            .expect("candidate list was checked as nonempty")
            .config
            .id
            .clone();
        let bootstrap_room = state.allocate_bootstrap_room()?;
        let assignment_id = Uuid::new_v4();

        let replica = state
            .replicas
            .get_mut(&selected_id)
            .expect("selected decoder disappeared while pool lock was held");
        replica.active_assignments += 1;
        replica.reserved_kv_tokens += demand.kv_tokens.get();
        replica.remaining_decode_tokens += demand.decode_tokens.get();

        let record = AssignmentRecord {
            request_id: Arc::clone(&request_id),
            decoder_id: selected_id.clone(),
            bootstrap_room,
            phase: AssignmentPhase::Reserved,
            kv_tokens: demand.kv_tokens.get(),
            remaining_decode_tokens: demand.decode_tokens.get(),
        };
        state
            .active_requests
            .insert(Arc::clone(&request_id), assignment_id);
        state.assignments.insert(assignment_id, record);

        Ok(DecoderAssignment {
            pool_id: self.pool_id,
            assignment_id,
            request_id,
            decoder_id: selected_id,
            bootstrap_room,
            phase: AssignmentPhase::Reserved,
        })
    }

    /// Mark that the assignment is crossing its first external submission boundary.
    pub fn mark_dispatched(
        &self,
        assignment: &mut DecoderAssignment,
    ) -> Result<(), DecoderPoolError> {
        self.transition(
            assignment,
            AssignmentPhase::Reserved,
            AssignmentPhase::Dispatched,
        )
    }

    /// Reduce remaining work as decode tokens are emitted.
    pub fn observe_decode_progress(
        &self,
        assignment: &DecoderAssignment,
        generated_tokens: usize,
    ) -> Result<(), DecoderPoolError> {
        self.validate_pool(assignment)?;
        let mut state = self.state.lock();
        let (decoder_id, remaining_tokens) = {
            let record = state.assignments.get(&assignment.assignment_id).ok_or(
                DecoderPoolError::UnknownAssignment(assignment.assignment_id),
            )?;
            validate_record(record, assignment)?;
            if record.phase != AssignmentPhase::Dispatched {
                return Err(invalid_transition(
                    assignment.assignment_id,
                    record.phase,
                    "record decode progress",
                ));
            }
            (record.decoder_id.clone(), record.remaining_decode_tokens)
        };
        if generated_tokens > remaining_tokens {
            return Err(DecoderPoolError::InvalidProgress {
                assignment_id: assignment.assignment_id,
                generated_tokens,
                remaining_tokens,
            });
        }
        state
            .assignments
            .get_mut(&assignment.assignment_id)
            .expect("assignment disappeared while pool lock was held")
            .remaining_decode_tokens -= generated_tokens;
        state
            .replicas
            .get_mut(&decoder_id)
            .expect("assigned decoder disappeared while pool lock was held")
            .remaining_decode_tokens -= generated_tokens;
        Ok(())
    }

    /// Quarantine a dispatched attempt while transfer quiescence is ambiguous.
    pub fn begin_quiescence(
        &self,
        assignment: &mut DecoderAssignment,
    ) -> Result<(), DecoderPoolError> {
        self.transition(
            assignment,
            AssignmentPhase::Dispatched,
            AssignmentPhase::Quiescing,
        )?;
        let mut state = self.state.lock();
        let replica = state
            .replicas
            .get_mut(&assignment.decoder_id)
            .expect("assigned decoder disappeared while pool lock was held");
        replica.quiescing_assignments += 1;
        Ok(())
    }

    /// Terminalize an attempt that never crossed the dispatch boundary.
    pub fn finish_before_dispatch(
        &self,
        assignment: DecoderAssignment,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        self.release(assignment, AssignmentPhase::Reserved, disposition)
    }

    /// Terminalize a successfully completed decode attempt.
    pub fn complete(&self, assignment: DecoderAssignment) -> Result<(), DecoderPoolError> {
        self.release(
            assignment,
            AssignmentPhase::Dispatched,
            RetryDisposition::Terminal,
        )
    }

    /// Release a quarantined attempt after handle-bound quiescence proof.
    pub fn confirm_quiesced(
        &self,
        assignment: DecoderAssignment,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        self.release(assignment, AssignmentPhase::Quiescing, disposition)
    }

    /// Return immutable accounting suitable for metrics and tests.
    pub fn snapshot(&self) -> DecoderPoolSnapshot {
        let state = self.state.lock();
        let mut replicas: Vec<DecoderReplicaSnapshot> = state
            .replicas
            .values()
            .map(|replica| DecoderReplicaSnapshot {
                id: replica.config.id.clone(),
                availability: replica.availability,
                active_assignments: replica.active_assignments,
                quiescing_assignments: replica.quiescing_assignments,
                reserved_kv_tokens: replica.reserved_kv_tokens,
                remaining_decode_tokens: replica.remaining_decode_tokens,
                capacity: replica.config.capacity,
            })
            .collect();
        replicas.sort_by(|left, right| left.id.cmp(&right.id));
        DecoderPoolSnapshot {
            prefill_tp_size: state.prefill_tp_size.get(),
            replicas,
        }
    }

    fn transition(
        &self,
        assignment: &mut DecoderAssignment,
        expected: AssignmentPhase,
        next: AssignmentPhase,
    ) -> Result<(), DecoderPoolError> {
        self.validate_pool(assignment)?;
        let mut state = self.state.lock();
        let record = state.assignments.get_mut(&assignment.assignment_id).ok_or(
            DecoderPoolError::UnknownAssignment(assignment.assignment_id),
        )?;
        validate_record(record, assignment)?;
        if assignment.phase != expected || record.phase != expected {
            return Err(invalid_transition(
                assignment.assignment_id,
                record.phase,
                phase_name(next),
            ));
        }
        assignment.phase = next;
        record.phase = next;
        Ok(())
    }

    fn release(
        &self,
        assignment: DecoderAssignment,
        expected: AssignmentPhase,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        self.validate_pool(&assignment)?;
        let mut state = self.state.lock();
        let record = state.assignments.get(&assignment.assignment_id).ok_or(
            DecoderPoolError::UnknownAssignment(assignment.assignment_id),
        )?;
        validate_record(record, &assignment)?;
        if assignment.phase != expected || record.phase != expected {
            return Err(invalid_transition(
                assignment.assignment_id,
                record.phase,
                "terminal",
            ));
        }

        let record = state
            .assignments
            .remove(&assignment.assignment_id)
            .expect("assignment disappeared while pool lock was held");
        let replica = state
            .replicas
            .get_mut(&record.decoder_id)
            .expect("assigned decoder disappeared while pool lock was held");
        replica.active_assignments -= 1;
        replica.reserved_kv_tokens -= record.kv_tokens;
        replica.remaining_decode_tokens -= record.remaining_decode_tokens;
        if expected == AssignmentPhase::Quiescing {
            replica.quiescing_assignments -= 1;
        }
        state.active_requests.remove(&record.request_id);

        match disposition {
            RetryDisposition::Terminal => {
                state.failed_decoders.remove(&record.request_id);
            }
            RetryDisposition::Retryable => {
                state
                    .failed_decoders
                    .entry(record.request_id)
                    .or_default()
                    .insert(record.decoder_id);
            }
        }
        Ok(())
    }

    fn validate_pool(&self, assignment: &DecoderAssignment) -> Result<(), DecoderPoolError> {
        if assignment.pool_id != self.pool_id {
            return Err(DecoderPoolError::ForeignAssignment);
        }
        Ok(())
    }
}

fn validate_record(
    record: &AssignmentRecord,
    assignment: &DecoderAssignment,
) -> Result<(), DecoderPoolError> {
    if record.request_id != assignment.request_id
        || record.decoder_id != assignment.decoder_id
        || record.bootstrap_room != assignment.bootstrap_room
    {
        return Err(DecoderPoolError::ForeignAssignment);
    }
    Ok(())
}

fn compare_projected_load(
    left: &ReplicaState,
    right: &ReplicaState,
    demand: DecoderDemand,
) -> Ordering {
    compare_ratio(
        left.remaining_decode_tokens + demand.decode_tokens.get(),
        left.config.capacity.service_weight.get(),
        right.remaining_decode_tokens + demand.decode_tokens.get(),
        right.config.capacity.service_weight.get(),
    )
    .then_with(|| {
        compare_ratio(
            left.reserved_kv_tokens + demand.kv_tokens.get(),
            left.config.capacity.max_kv_tokens.get(),
            right.reserved_kv_tokens + demand.kv_tokens.get(),
            right.config.capacity.max_kv_tokens.get(),
        )
    })
    .then_with(|| {
        compare_ratio(
            left.active_assignments + 1,
            left.config.capacity.max_concurrent_requests.get(),
            right.active_assignments + 1,
            right.config.capacity.max_concurrent_requests.get(),
        )
    })
    .then_with(|| left.config.id.cmp(&right.config.id))
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
    actual: AssignmentPhase,
    requested: &'static str,
) -> DecoderPoolError {
    DecoderPoolError::InvalidTransition {
        assignment_id,
        actual: phase_name(actual),
        requested,
    }
}

fn phase_name(phase: AssignmentPhase) -> &'static str {
    match phase {
        AssignmentPhase::Reserved => "reserved",
        AssignmentPhase::Dispatched => "dispatched",
        AssignmentPhase::Quiescing => "quiescing",
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc, thread};

    use super::*;

    fn compatibility(protocol: &str) -> TransferCompatibility {
        TransferCompatibility::new(
            "gemma-4-31b-nvfp4@sha256:model",
            "gemma4-full10-swa50@sha256:layout",
            "bfloat16",
            protocol,
            1,
        )
        .unwrap()
    }

    fn capacity(max_requests: usize, max_tokens: usize) -> DecoderCapacity {
        DecoderCapacity::new(max_requests, max_tokens, 1).unwrap()
    }

    fn replica(name: &str, protocol: &str) -> DecoderReplicaConfig {
        DecoderReplicaConfig::new(
            DecoderId::new(name).unwrap(),
            1,
            compatibility(protocol),
            capacity(32, 32_000),
        )
        .unwrap()
    }

    fn pool(prefill_tp_size: usize) -> DecoderPool {
        DecoderPool::new(prefill_tp_size, compatibility("packed-v1")).unwrap()
    }

    fn demand() -> DecoderDemand {
        DecoderDemand::new(1_024, 128).unwrap()
    }

    #[test]
    fn accepts_tp2_and_tp4_prefill_with_tp1_decoders() {
        for prefill_tp_size in [2, 4] {
            let pool = pool(prefill_tp_size);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            assert_eq!(pool.snapshot().prefill_tp_size, prefill_tp_size);
        }
    }

    #[test]
    fn rejects_non_tp1_and_wire_incompatible_decoders() {
        let pool = pool(4);
        let tp2 = DecoderReplicaConfig::new(
            DecoderId::new("decode-tp2").unwrap(),
            2,
            compatibility("packed-v1"),
            capacity(32, 32_000),
        )
        .unwrap();
        assert!(matches!(
            pool.register(tp2),
            Err(DecoderPoolError::IncompatibleDecoder { .. })
        ));
        assert!(matches!(
            pool.register(replica("decode-wrong-wire", "packed-v2")),
            Err(DecoderPoolError::IncompatibleDecoder { .. })
        ));
    }

    #[test]
    fn balances_arbitrary_replica_count_by_projected_decode_work() {
        let pool = pool(2);
        for index in 0..3 {
            pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                .unwrap();
        }

        let mut assignments = Vec::new();
        for index in 0..9 {
            assignments.push(pool.reserve(format!("request-{index}"), demand()).unwrap());
        }
        let snapshot = pool.snapshot();
        assert_eq!(
            snapshot
                .replicas
                .iter()
                .map(|replica| replica.active_assignments)
                .collect::<Vec<_>>(),
            vec![3, 3, 3]
        );

        for assignment in assignments {
            pool.finish_before_dispatch(assignment, RetryDisposition::Terminal)
                .unwrap();
        }
    }

    #[test]
    fn retry_waits_for_quiescence_and_uses_a_new_room() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();

        let mut first = pool.reserve("request", demand()).unwrap();
        let first_decoder = first.decoder_id().clone();
        let first_room = first.bootstrap_room();
        pool.mark_dispatched(&mut first).unwrap();
        assert_eq!(
            pool.reserve("request", demand()).unwrap_err(),
            DecoderPoolError::RequestAlreadyAssigned("request".to_string())
        );
        pool.begin_quiescence(&mut first).unwrap();
        assert_eq!(
            pool.reserve("request", demand()).unwrap_err(),
            DecoderPoolError::RequestAlreadyAssigned("request".to_string())
        );
        pool.confirm_quiesced(first, RetryDisposition::Retryable)
            .unwrap();

        let retry = pool.reserve("request", demand()).unwrap();
        assert_ne!(retry.bootstrap_room(), first_room);
        assert_ne!(retry.decoder_id(), &first_decoder);
        pool.finish_before_dispatch(retry, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn draining_preserves_inflight_ownership_and_stops_admission() {
        let pool = pool(4);
        let decoder_id = DecoderId::new("decode-0").unwrap();
        pool.register(replica(decoder_id.as_str(), "packed-v1"))
            .unwrap();
        let assignment = pool.reserve("request", demand()).unwrap();
        pool.set_availability(&decoder_id, DecoderAvailability::Draining)
            .unwrap();
        assert_eq!(
            pool.reserve("other", demand()).unwrap_err(),
            DecoderPoolError::NoReadyDecoder
        );
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderInUse { .. })
        ));
        pool.finish_before_dispatch(assignment, RetryDisposition::Terminal)
            .unwrap();
        pool.remove(&decoder_id).unwrap();
    }

    #[test]
    fn admission_enforces_request_and_kv_limits() {
        let pool = pool(2);
        let config = DecoderReplicaConfig::new(
            DecoderId::new("decode-0").unwrap(),
            1,
            compatibility("packed-v1"),
            capacity(2, 2_048),
        )
        .unwrap();
        pool.register(config).unwrap();
        let first = pool.reserve("first", demand()).unwrap();
        let second = pool.reserve("second", demand()).unwrap();
        assert_eq!(
            pool.reserve("third", demand()).unwrap_err(),
            DecoderPoolError::AtCapacity
        );
        pool.finish_before_dispatch(first, RetryDisposition::Terminal)
            .unwrap();
        pool.finish_before_dispatch(second, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn decode_progress_updates_projected_work_without_releasing_kv() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut assignment = pool.reserve("request", demand()).unwrap();
        pool.mark_dispatched(&mut assignment).unwrap();
        pool.observe_decode_progress(&assignment, 100).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 28);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 1_024);
        assert!(matches!(
            pool.observe_decode_progress(&assignment, 29),
            Err(DecoderPoolError::InvalidProgress { .. })
        ));
        pool.complete(assignment).unwrap();
    }

    #[test]
    fn decode_work_accounting_rejects_overflow() {
        let pool = pool(2);
        let config = DecoderReplicaConfig::new(
            DecoderId::new("decode-0").unwrap(),
            1,
            compatibility("packed-v1"),
            capacity(2, 2),
        )
        .unwrap();
        pool.register(config).unwrap();
        let first = pool
            .reserve("first", DecoderDemand::new(1, usize::MAX).unwrap())
            .unwrap();
        assert_eq!(
            pool.reserve("second", DecoderDemand::new(1, 1).unwrap())
                .unwrap_err(),
            DecoderPoolError::AtCapacity
        );
        pool.finish_before_dispatch(first, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn foreign_pool_cannot_retire_an_assignment() {
        let first_pool = pool(2);
        let second_pool = pool(2);
        first_pool
            .register(replica("decode-0", "packed-v1"))
            .unwrap();
        second_pool
            .register(replica("decode-0", "packed-v1"))
            .unwrap();
        let assignment = first_pool.reserve("request", demand()).unwrap();
        assert_eq!(
            second_pool.finish_before_dispatch(assignment, RetryDisposition::Terminal),
            Err(DecoderPoolError::ForeignAssignment)
        );
        assert_eq!(first_pool.snapshot().replicas[0].active_assignments, 1);
    }

    #[test]
    fn concurrent_admission_never_oversubscribes_or_reuses_rooms() {
        let pool = Arc::new(pool(2));
        let config = DecoderReplicaConfig::new(
            DecoderId::new("decode-0").unwrap(),
            1,
            compatibility("packed-v1"),
            capacity(32, 320),
        )
        .unwrap();
        pool.register(config).unwrap();

        let handles: Vec<_> = (0..64)
            .map(|index| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    pool.reserve(
                        format!("request-{index}"),
                        DecoderDemand::new(10, 1).unwrap(),
                    )
                })
            })
            .collect();
        let mut assignments = Vec::new();
        let mut capacity_errors = 0;
        for handle in handles {
            match handle.join().unwrap() {
                Ok(assignment) => assignments.push(assignment),
                Err(DecoderPoolError::AtCapacity) => capacity_errors += 1,
                Err(error) => panic!("unexpected admission error: {error}"),
            }
        }

        assert_eq!(assignments.len(), 32);
        assert_eq!(capacity_errors, 32);
        assert_eq!(
            assignments
                .iter()
                .map(DecoderAssignment::bootstrap_room)
                .collect::<HashSet<_>>()
                .len(),
            assignments.len()
        );
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_assignments, 32);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 320);

        for assignment in assignments {
            pool.finish_before_dispatch(assignment, RetryDisposition::Terminal)
                .unwrap();
        }
    }
}
