//! Request-affine admission and lifecycle accounting for disaggregated decoders.
//!
//! A cohort binds an allocator-issued engine grant to one logical request,
//! decoder process generation, request-slot allocation generation, and ordered
//! room vector. Promotion is itself the irreversible activation/publication
//! boundary: the cohort becomes active before the promote request is polled and
//! remains pinned until the engine proves it complete or terminally quiescent.
//!
//! The metadata checks in this module are eligibility checks only. They do not
//! prove asymmetric TP slicing, DMA lane selection, destination correctness, or
//! transfer quiescence. Configured scales are advisory and never override an
//! allocator grant. The caller must provide one authoritative routing process,
//! an engine reservation capability, and an exact engine terminal receipt
//! before invoking a receipt-backed terminal transition.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt,
    num::NonZeroUsize,
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
use thiserror::Error;
use tracing::warn;
use uuid::Uuid;

use super::pd_decoder_grant::{
    AbortReconciliationGrant, BindReconciliationGrant, BoundPreparedGrant,
    CompletionReconciliationGrant, DecoderAllocationKey, DecoderControlAuthorization,
    DecoderGrantBinding, DecoderGrantControlClient, DecoderGrantDigest, DecoderGrantReservation,
    DecoderId, DecoderRequestTemplate, DecoderReserveAttemptDigest,
    DecoderReserveRefusalDisposition, DecoderReserveRefusalReceipt, DecoderSlotGeneration,
    EngineAbortOutcome, EngineCompletionOutcome, EngineGrantError, EngineQuarantineReceipt,
    EngineReleaseKind, EngineReleaseReceipt, PrefillId, PreparedCancellationReconciliationGrant,
    PreparedGrantCancellationReceipt, PreparedGrantCancellationTarget,
    PromotionReconciliationGrant, QuarantineReconciliationGrant, ReserveReconciliationGrant,
    RetainedEngineGrant, UnboundCancellationReconciliationGrant, UnboundPreparedGrant,
};
use crate::core::PrefillBootstrapEndpoint;

/// Engine-declared fields used to reject obviously incompatible PD pairings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCompatibilityMetadata {
    model_fingerprint: Arc<str>,
    kv_layout_fingerprint: Arc<str>,
    kv_cache_dtype: Arc<str>,
    wire_protocol: Arc<str>,
    prepared_grant_protocol: Arc<str>,
    page_size: NonZeroUsize,
}

impl EngineCompatibilityMetadata {
    /// Construct immutable metadata reported by an engine process generation.
    pub fn new(
        model_fingerprint: impl Into<String>,
        kv_layout_fingerprint: impl Into<String>,
        kv_cache_dtype: impl Into<String>,
        wire_protocol: impl Into<String>,
        prepared_grant_protocol: impl Into<String>,
        page_size: usize,
    ) -> Result<Self, DecoderPoolError> {
        let model_fingerprint = nonempty("model fingerprint", model_fingerprint.into())?;
        let kv_layout_fingerprint =
            nonempty("KV layout fingerprint", kv_layout_fingerprint.into())?;
        let kv_cache_dtype = nonempty("KV cache dtype", kv_cache_dtype.into())?;
        let wire_protocol = nonempty("wire protocol", wire_protocol.into())?;
        let prepared_grant_protocol =
            nonempty("prepared-grant protocol", prepared_grant_protocol.into())?;
        let page_size = NonZeroUsize::new(page_size).ok_or_else(|| {
            DecoderPoolError::InvalidConfiguration("page size must be nonzero".to_string())
        })?;

        Ok(Self {
            model_fingerprint: Arc::from(model_fingerprint),
            kv_layout_fingerprint: Arc::from(kv_layout_fingerprint),
            kv_cache_dtype: Arc::from(kv_cache_dtype),
            wire_protocol: Arc::from(wire_protocol),
            prepared_grant_protocol: Arc::from(prepared_grant_protocol),
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

/// Engine-declared metadata and advisory scheduling scales for one decoder.
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
    Cancelling,
    Active,
    Completing,
    Aborting,
    Quarantining,
    Quarantined,
    Terminal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AdmissionRetryConstraint {
    AnyEligible,
    SameDecoder(DecoderId),
}

#[derive(Debug)]
enum RequestChainState {
    IdleOpen(AdmissionRetryConstraint),
    Reserving(Box<PendingAdmissionRecord>),
    Assigned(Uuid),
    Quarantined(Uuid),
    Terminal,
}

/// Advisory scheduling charge held while one engine reservation is unresolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingSchedulingCharge {
    child_requests: NonZeroUsize,
    reserved_kv_tokens: usize,
    remaining_decode_tokens: usize,
}

impl PendingSchedulingCharge {
    /// Construct the load estimate charged before allocator I/O.
    pub fn new(
        child_requests: usize,
        reserved_kv_tokens: usize,
        remaining_decode_tokens: usize,
    ) -> Result<Self, DecoderPoolError> {
        let child_requests = NonZeroUsize::new(child_requests).ok_or_else(|| {
            DecoderPoolError::InvalidConfiguration(
                "pending admission must describe at least one child request".to_string(),
            )
        })?;
        Ok(Self {
            child_requests,
            reserved_kv_tokens,
            remaining_decode_tokens,
        })
    }

    /// Number of child requests included in the pending attempt.
    pub fn child_requests(&self) -> usize {
        self.child_requests.get()
    }

    /// Conservatively estimated KV tokens held while reserving.
    pub fn reserved_kv_tokens(&self) -> usize {
        self.reserved_kv_tokens
    }

    /// Estimated remaining decode tokens held while reserving.
    pub fn remaining_decode_tokens(&self) -> usize {
        self.remaining_decode_tokens
    }
}

/// Authoritative next step after a pending reservation is released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PendingAdmissionDisposition {
    RetrySameDecoder,
    RetryAnotherDecoder,
    Terminal,
}

/// Pool-bound reserve result after every authoritative refusal is applied.
#[derive(Debug)]
pub enum PendingReserveOutcome {
    Prepared(Box<UnboundPreparedGrant>),
    Refused(PendingAdmissionDisposition),
}

/// Failure while reconciling engine authority with pending pool ownership.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PendingReconciliationError {
    #[error(transparent)]
    Engine(#[from] EngineGrantError),
    #[error(transparent)]
    Pool(#[from] DecoderPoolError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PendingCancellationProof {
    Unbound(PreparedGrantCancellationReceipt),
    Bound(EngineReleaseReceipt),
}

#[derive(Debug)]
pub(super) struct PendingCancellationPin {
    target: PreparedGrantCancellationTarget,
}

impl PendingCancellationPin {
    fn new(target: PreparedGrantCancellationTarget) -> Self {
        Self { target }
    }

    pub(super) fn matches(&self, target: &PreparedGrantCancellationTarget) -> bool {
        &self.target == target
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingCancellationKind {
    Unbound,
    Bound,
}

#[derive(Debug)]
enum PendingCancellationEngine {
    Unbound(Box<UnboundCancellationReconciliationGrant>),
    Bound(Box<PreparedCancellationReconciliationGrant>),
}

#[derive(Debug)]
enum PendingAdmissionAuthority {
    ReserveReady,
    ReserveCheckedOut {
        operation_id: Uuid,
    },
    ReserveRetry(Box<ReserveReconciliationGrant>),
    PreparedIssued,
    CancellationCheckedOut {
        operation_id: Uuid,
        kind: PendingCancellationKind,
    },
    CancellationRetry(PendingCancellationEngine),
    ReserveProof(Box<DecoderReserveRefusalReceipt>),
    CancellationProof(Box<PendingCancellationProof>),
}

/// Exclusive lease over one exact pending reservation owned by the pool.
#[must_use = "pending admission ownership must be reserved, bound, or explicitly released"]
pub struct PendingAdmission {
    inner: Arc<DecoderPoolInner>,
    chain_id: Uuid,
    claim_id: Uuid,
    decoder_id: DecoderId,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    charge: PendingSchedulingCharge,
    retry_constraint: AdmissionRetryConstraint,
    resolved: bool,
}

impl fmt::Debug for PendingAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingAdmission")
            .field("chain_id", &self.chain_id)
            .field("claim_id", &self.claim_id)
            .field("decoder_id", &self.decoder_id)
            .field("reservation_attempt_id", &self.reservation_attempt_id)
            .field("reserve_attempt_digest", &self.reserve_attempt_digest)
            .field("charge", &self.charge)
            .field("retry_constraint", &self.retry_constraint)
            .field("resolved", &self.resolved)
            .finish()
    }
}

impl PendingAdmission {
    /// Exact decoder process generation selected for this attempt.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Stable idempotency identity for every reserve retry.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    /// Digest of the exact reserve transcript pinned before allocator I/O.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    /// Move the exact reservation transcript into pool-bound reconciliation.
    pub fn begin_reserve(
        &mut self,
        client: &DecoderGrantControlClient,
        authorization: DecoderControlAuthorization,
    ) -> Result<PendingReserveReconciliation<'_>, DecoderPoolError> {
        let operation_id = Uuid::new_v4();
        let engine =
            self.inner
                .checkout_initial_reserve(self, operation_id, client, authorization)?;
        Ok(PendingReserveReconciliation {
            pending: self,
            operation_id,
            engine: Some(engine),
            polled: false,
            complete: false,
        })
    }

    /// Resume the exact reserve authority recovered from a dropped poll.
    pub fn resume_reserve(&mut self) -> Result<PendingReserveReconciliation<'_>, DecoderPoolError> {
        let operation_id = Uuid::new_v4();
        let engine = self.inner.checkout_reserve_retry(self, operation_id)?;
        Ok(PendingReserveReconciliation {
            pending: self,
            operation_id,
            engine: Some(engine),
            polled: true,
            complete: false,
        })
    }
}

/// Pool-bound reserve retry whose authority returns to the pool on cancellation.
#[must_use = "reserve reconciliation must be polled or dropped to release an unstarted attempt"]
pub struct PendingReserveReconciliation<'a> {
    pending: &'a mut PendingAdmission,
    operation_id: Uuid,
    engine: Option<ReserveReconciliationGrant>,
    polled: bool,
    complete: bool,
}

impl fmt::Debug for PendingReserveReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingReserveReconciliation")
            .field(
                "reservation_attempt_id",
                &self.pending.reservation_attempt_id,
            )
            .field("decoder_id", &self.pending.decoder_id)
            .field("operation_id", &self.operation_id)
            .field("polled", &self.polled)
            .field("complete", &self.complete)
            .finish_non_exhaustive()
    }
}

impl PendingReserveReconciliation<'_> {
    /// Reconcile the same exact attempt and apply any allocator refusal in-pool.
    pub async fn reconcile_reserve(
        &mut self,
    ) -> Result<PendingReserveOutcome, PendingReconciliationError> {
        self.polled = true;
        let result = self
            .engine
            .as_mut()
            .expect("live reserve checkout must retain engine authority")
            .reconcile_reserve()
            .await;
        match result {
            Ok(grant) => {
                self.engine = None;
                self.pending.inner.complete_reserve_checkout(
                    self.pending,
                    self.operation_id,
                    PendingAdmissionAuthority::PreparedIssued,
                );
                self.complete = true;
                Ok(PendingReserveOutcome::Prepared(Box::new(grant)))
            }
            Err(EngineGrantError::AllocatorRefused(receipt)) => {
                self.engine = None;
                let inner = Arc::clone(&self.pending.inner);
                let disposition = inner.install_reserve_refusal_and_apply(
                    self.pending,
                    self.operation_id,
                    *receipt,
                );
                self.complete = true;
                match disposition {
                    Ok(disposition) => {
                        self.pending.resolved = true;
                        Ok(PendingReserveOutcome::Refused(disposition))
                    }
                    Err(error) => Err(error.into()),
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Gateway-issued idempotency identity reused by every reserve retry.
    pub fn reservation_attempt_id(&self) -> Result<Uuid, EngineGrantError> {
        self.engine
            .as_ref()
            .ok_or_else(|| {
                EngineGrantError::ProtocolViolation(
                    "reserve reconciliation is already complete".to_string(),
                )
            })?
            .reservation_attempt_id()
    }

    #[cfg(test)]
    fn mark_test_polled(&mut self) {
        self.polled = true;
    }
}

impl Drop for PendingReserveReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let engine = self
            .engine
            .take()
            .expect("incomplete reserve checkout must retain engine authority");
        let inner = Arc::clone(&self.pending.inner);
        if self.polled {
            inner.restore_reserve_checkout(self.pending, self.operation_id, engine);
            return;
        }
        inner.rollback_checked_out_reserve(self.pending, self.operation_id);
        self.pending.resolved = true;
    }
}

/// Pool-bound pending cancellation whose raw authority never escapes the pool.
#[must_use = "pending cancellation must be reconciled or dropped for exact retry"]
pub struct PendingCancellationReconciliation<'a> {
    pending: &'a mut PendingAdmission,
    operation_id: Uuid,
    engine: Option<PendingCancellationEngine>,
    complete: bool,
}

impl fmt::Debug for PendingCancellationReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingCancellationReconciliation")
            .field(
                "reservation_attempt_id",
                &self.pending.reservation_attempt_id,
            )
            .field("decoder_id", &self.pending.decoder_id)
            .field("operation_id", &self.operation_id)
            .field("complete", &self.complete)
            .finish_non_exhaustive()
    }
}

impl PendingCancellationReconciliation<'_> {
    /// Reconcile the exact cancellation and apply its proof before returning.
    pub async fn reconcile_cancellation(
        &mut self,
    ) -> Result<PendingAdmissionDisposition, PendingReconciliationError> {
        let result = match self
            .engine
            .as_mut()
            .expect("live cancellation checkout must retain engine authority")
        {
            PendingCancellationEngine::Unbound(engine) => engine
                .reconcile_cancellation()
                .await
                .map(PendingCancellationProof::Unbound),
            PendingCancellationEngine::Bound(engine) => engine
                .reconcile_cancellation()
                .await
                .map(PendingCancellationProof::Bound),
        };
        match result {
            Ok(proof) => self.install_proof(proof),
            Err(error) => Err(error.into()),
        }
    }

    fn install_proof(
        &mut self,
        proof: PendingCancellationProof,
    ) -> Result<PendingAdmissionDisposition, PendingReconciliationError> {
        self.engine = None;
        let inner = Arc::clone(&self.pending.inner);
        let disposition =
            inner.install_pending_cancellation_and_apply(self.pending, self.operation_id, proof);
        self.complete = true;
        match disposition {
            Ok(disposition) => {
                self.pending.resolved = true;
                Ok(disposition)
            }
            Err(error) => Err(error.into()),
        }
    }

    #[cfg(test)]
    fn install_test_proof(
        &mut self,
        proof: PendingCancellationProof,
    ) -> Result<PendingAdmissionDisposition, PendingReconciliationError> {
        match self
            .engine
            .as_mut()
            .expect("test cancellation checkout must retain engine authority")
        {
            PendingCancellationEngine::Unbound(engine) => {
                engine.assume_test_reconciled()?;
            }
            PendingCancellationEngine::Bound(engine) => {
                engine.assume_test_reconciled()?;
            }
        }
        self.install_proof(proof)
    }
}

impl Drop for PendingCancellationReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let engine = self
            .engine
            .take()
            .expect("incomplete cancellation checkout must retain engine authority");
        let inner = Arc::clone(&self.pending.inner);
        inner.restore_cancellation_checkout(self.pending, self.operation_id, engine);
    }
}

impl Drop for PendingAdmission {
    fn drop(&mut self) {
        if self.resolved {
            return;
        }
        let inner = Arc::clone(&self.inner);
        inner.release_pending_claim(self);
    }
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
    prepared_grant: Option<BoundPreparedGrant>,
}

/// Non-cloneable proof that the pool selected one exact grant transcript.
pub(super) struct DecoderGrantPoolBinding {
    grant_id: Uuid,
    grant_digest: DecoderGrantDigest,
}

impl DecoderGrantPoolBinding {
    fn new(binding: &DecoderGrantBinding) -> Self {
        Self {
            grant_id: binding.grant_id(),
            grant_digest: binding.digest(),
        }
    }

    pub(super) fn matches(&self, binding: &DecoderGrantBinding) -> bool {
        self.grant_id == binding.grant_id() && self.grant_digest == binding.digest()
    }
}

impl DecoderAssignmentCohort {
    /// Stable identity of the selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Ordered engine child request identities bound to the original inputs.
    pub fn child_request_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.binding.child_request_ids()
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
    pub pending_admissions: usize,
    pub pending_child_requests: usize,
    pub pending_reserved_kv_tokens: usize,
    pub pending_remaining_decode_tokens: usize,
    pub active_cohorts: usize,
    pub active_child_requests: usize,
    pub quiescing_cohorts: usize,
    pub quarantined_cohorts: usize,
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
    #[error(
        "decoder {decoder_id} owns {active_cohorts} active cohorts and {pending_admissions} pending admissions"
    )]
    DecoderInUse {
        decoder_id: DecoderId,
        active_cohorts: usize,
        pending_admissions: usize,
    },
    #[error(
        "prefill pool owns {request_chains} request chains, {pending_admissions} pending admissions, {assignments} assignments, {room_owners} rooms, {allocation_owners} allocations, and {quarantined_cohorts} quarantined cohorts"
    )]
    PrefillPoolInUse {
        request_chains: usize,
        pending_admissions: usize,
        assignments: usize,
        room_owners: usize,
        allocation_owners: usize,
        quarantined_cohorts: usize,
    },
    #[error("decoder {decoder_id} metadata is ineligible for this prefill pool: {reason}")]
    IneligibleDecoderMetadata {
        decoder_id: DecoderId,
        reason: String,
    },
    #[error("logical request identity {0} already has an owner")]
    RequestAlreadyOwned(String),
    #[error("prefill pool is draining and no longer accepts logical requests")]
    PrefillPoolDraining,
    #[error("logical request owner was issued by another decoder pool")]
    ForeignRequestOwner,
    #[error("logical request owner is already finalized")]
    RequestOwnerFinalized,
    #[error("logical request chain {0} no longer has a live owner")]
    RequestChainOwnerDropped(Uuid),
    #[error("logical request chain {0} is unknown")]
    UnknownRequestChain(Uuid),
    #[error("logical request {request_id} still owns assignment {assignment_id}")]
    RequestHasActiveCohort {
        request_id: String,
        assignment_id: Uuid,
    },
    #[error("logical request {request_id} still owns pending attempt {reservation_attempt_id}")]
    RequestHasPendingAdmission {
        request_id: String,
        reservation_attempt_id: Uuid,
    },
    #[error("logical request {0} is terminal")]
    RequestChainTerminal(String),
    #[error("no ready decoder replica is registered")]
    NoReadyDecoder,
    #[error("the logical request has exhausted every registered retry alternative")]
    RetryAlternativesExhausted,
    #[error("retry requires decoder generation {0}, which is not ready")]
    RetryDecoderUnavailable(DecoderId),
    #[error("pending reservation attempt {0} has already started")]
    PendingReservationAlreadyStarted(Uuid),
    #[error("pending reservation attempt {0} already has a live recovery claim")]
    PendingAdmissionAlreadyClaimed(Uuid),
    #[error("pending reservation attempt {0} still has a live request owner")]
    PendingAdmissionOwnerStillAlive(Uuid),
    #[error("pending admission capability was not issued by this decoder pool")]
    ForeignPendingAdmission,
    #[error("pending reservation attempt {0} is unknown or already bound")]
    UnknownPendingAdmission(Uuid),
    #[error("pending reservation attempt {0} has no authoritative release proof")]
    PendingAdmissionProofPending(Uuid),
    #[error("pending reservation attempt {0} has no pinned cancellation intent")]
    PendingCancellationNotPinned(Uuid),
    #[error("pending reservation attempt {0} already retains a conflicting release proof")]
    ConflictingPendingAdmissionProof(Uuid),
    #[error(
        "pending reservation attempt {reservation_attempt_id} received an invalid cancellation proof: {reason}"
    )]
    InvalidPendingCancellationProof {
        reservation_attempt_id: Uuid,
        reason: &'static str,
    },
    #[error(
        "decoder-pool accounting for pending reservation {reservation_attempt_id} is inconsistent: {reason}"
    )]
    InconsistentPendingAdmission {
        reservation_attempt_id: Uuid,
        reason: &'static str,
    },
    #[error(
        "pending reservation describes {pending_children} children but the request describes {request_children}"
    )]
    PendingChildCountMismatch {
        pending_children: usize,
        request_children: usize,
    },
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
    #[error(
        "allocator slot generation {slot_generation} on decoder {decoder_id} is already owned by an active cohort"
    )]
    GrantSlotGenerationInUse {
        decoder_id: DecoderId,
        slot_generation: Uuid,
    },
    #[error("allocator grant identity {0} is already active")]
    GrantAlreadyActive(Uuid),
    #[error("assignment capability was not issued by this decoder pool")]
    ForeignAssignment,
    #[error("assignment {0} is unknown or already terminal")]
    UnknownAssignment(Uuid),
    #[error("assignment {0} is still waiting for an authoritative terminal proof")]
    TerminalProofPending(Uuid),
    #[error("assignment {0} already retains a conflicting terminal proof")]
    ConflictingTerminalProof(Uuid),
    #[error("decoder-pool accounting for assignment {assignment_id} is inconsistent: {reason}")]
    InconsistentAssignment {
        assignment_id: Uuid,
        reason: &'static str,
    },
    #[error("engine release receipt does not match assignment {assignment_id}: {reason}")]
    InvalidEngineReleaseReceipt {
        assignment_id: Uuid,
        reason: &'static str,
    },
    #[error("engine quarantine receipt does not match assignment {assignment_id}: {reason}")]
    InvalidEngineQuarantineReceipt {
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

/// Failures while reconciling one pool-bound engine operation.
#[derive(Debug, Error)]
pub enum DecoderAssignmentReconciliationError {
    #[error(transparent)]
    Engine(#[from] EngineGrantError),
    #[error(transparent)]
    Pool(#[from] DecoderPoolError),
    #[error("{0} reconciliation is already complete")]
    AlreadyComplete(&'static str),
}

/// Pool-bound cancellation that can only terminalize its own reserved cohort.
#[must_use = "cancellation reconciliation must be driven to an authoritative outcome"]
pub struct PoolCancellationReconciliation<'a> {
    pool: DecoderPool,
    cohort: &'a mut DecoderAssignmentCohort,
    engine: PreparedCancellationReconciliationGrant,
    complete: bool,
}

impl fmt::Debug for PoolCancellationReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PoolCancellationReconciliation")
            .field("assignment_id", &self.cohort.assignment_id)
            .field("complete", &self.complete)
            .finish()
    }
}

/// Pool-bound completion that can only terminalize its own active cohort.
#[must_use = "completion reconciliation must be driven to an authoritative outcome"]
pub struct PoolCompletionReconciliation<'a> {
    pool: DecoderPool,
    cohort: &'a mut DecoderAssignmentCohort,
    engine: CompletionReconciliationGrant,
    complete: bool,
}

impl fmt::Debug for PoolCompletionReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PoolCompletionReconciliation")
            .field("assignment_id", &self.cohort.assignment_id)
            .field("complete", &self.complete)
            .finish()
    }
}

/// Pool-bound abort that applies either authoritative terminal outcome in place.
#[must_use = "abort reconciliation must be driven to an authoritative outcome"]
pub struct PoolAbortReconciliation<'a> {
    pool: DecoderPool,
    cohort: &'a mut DecoderAssignmentCohort,
    engine: AbortReconciliationGrant,
    complete: bool,
}

impl fmt::Debug for PoolAbortReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PoolAbortReconciliation")
            .field("assignment_id", &self.cohort.assignment_id)
            .field("complete", &self.complete)
            .finish()
    }
}

/// Pool-bound quarantine that can only retain its own exact cohort.
#[must_use = "quarantine reconciliation must be driven to an authoritative outcome"]
pub struct PoolQuarantineReconciliation<'a> {
    pool: DecoderPool,
    cohort: &'a mut DecoderAssignmentCohort,
    engine: QuarantineReconciliationGrant,
    complete: bool,
}

impl fmt::Debug for PoolQuarantineReconciliation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PoolQuarantineReconciliation")
            .field("assignment_id", &self.cohort.assignment_id)
            .field("complete", &self.complete)
            .finish()
    }
}

impl PoolCancellationReconciliation<'_> {
    /// Reconcile the pinned engine cancellation and its exact pool cohort.
    ///
    /// Once the engine returns a receipt, later retries only repeat pool
    /// application. Engine I/O is never repeated after authoritative release.
    pub async fn reconcile(&mut self) -> Result<(), DecoderAssignmentReconciliationError> {
        if self.complete {
            return Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "cancellation",
            ));
        }
        match self.pool.resume_terminal_reconciliation(self.cohort) {
            Ok(()) => {
                self.complete = true;
                return Ok(());
            }
            Err(DecoderPoolError::TerminalProofPending(assignment_id))
                if assignment_id == self.cohort.assignment_id => {}
            Err(error) => return Err(error.into()),
        }
        let receipt = self.engine.reconcile_cancellation().await?;
        self.pool.install_cancellation_proof(self.cohort, receipt)?;
        self.pool.resume_terminal_reconciliation(self.cohort)?;
        self.complete = true;
        Ok(())
    }
}

impl PoolCompletionReconciliation<'_> {
    /// Reconcile the pinned engine completion and its exact pool cohort.
    ///
    /// Once the engine returns a receipt, later retries only repeat pool
    /// application. Engine I/O is never repeated after authoritative release.
    pub async fn reconcile(&mut self) -> Result<(), DecoderAssignmentReconciliationError> {
        if self.complete {
            return Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "completion",
            ));
        }
        match self.pool.resume_terminal_reconciliation(self.cohort) {
            Ok(()) => {
                self.complete = true;
                return Ok(());
            }
            Err(DecoderPoolError::TerminalProofPending(assignment_id))
                if assignment_id == self.cohort.assignment_id => {}
            Err(error) => return Err(error.into()),
        }
        let outcome = self.engine.reconcile_completion().await?;
        self.pool.install_completion_proof(self.cohort, outcome)?;
        self.pool.resume_terminal_reconciliation(self.cohort)?;
        self.complete = true;
        Ok(())
    }
}

impl PoolAbortReconciliation<'_> {
    /// Reconcile the pinned engine abort and its exact pool cohort.
    ///
    /// The authoritative engine outcome is retained before pool application,
    /// including quarantine fallbacks, so a pool retry cannot repeat abort I/O.
    pub async fn reconcile(&mut self) -> Result<(), DecoderAssignmentReconciliationError> {
        if self.complete {
            return Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "abort",
            ));
        }
        match self.pool.resume_terminal_reconciliation(self.cohort) {
            Ok(()) => {
                self.complete = true;
                return Ok(());
            }
            Err(DecoderPoolError::TerminalProofPending(assignment_id))
                if assignment_id == self.cohort.assignment_id => {}
            Err(error) => return Err(error.into()),
        }
        let outcome = self.engine.reconcile_abort().await?;
        self.pool.install_abort_proof(self.cohort, outcome)?;
        self.pool.resume_terminal_reconciliation(self.cohort)?;
        self.complete = true;
        Ok(())
    }
}

impl PoolQuarantineReconciliation<'_> {
    /// Reconcile the pinned engine quarantine and its exact pool cohort.
    ///
    /// Once the engine returns a receipt, later retries only repeat pool
    /// application. Engine I/O is never repeated after authoritative retention.
    pub async fn reconcile(&mut self) -> Result<(), DecoderAssignmentReconciliationError> {
        if self.complete {
            return Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "quarantine",
            ));
        }
        match self.pool.resume_terminal_reconciliation(self.cohort) {
            Ok(()) => {
                self.complete = true;
                return Ok(());
            }
            Err(DecoderPoolError::TerminalProofPending(assignment_id))
                if assignment_id == self.cohort.assignment_id => {}
            Err(error) => return Err(error.into()),
        }
        let receipt = self.engine.reconcile_quarantine().await?;
        self.pool.install_quarantine_proof(self.cohort, receipt)?;
        self.pool.resume_terminal_reconciliation(self.cohort)?;
        self.complete = true;
        Ok(())
    }
}

impl Drop for PoolCancellationReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        warn!(
            assignment_id = %self.cohort.assignment_id,
            "Pool-bound cancellation reconciliation was dropped before pool terminalization"
        );
    }
}

impl Drop for PoolCompletionReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        warn!(
            assignment_id = %self.cohort.assignment_id,
            "Pool-bound completion reconciliation was dropped before pool terminalization"
        );
    }
}

impl Drop for PoolAbortReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        warn!(
            assignment_id = %self.cohort.assignment_id,
            "Pool-bound abort reconciliation was dropped before pool terminalization"
        );
    }
}

impl Drop for PoolQuarantineReconciliation<'_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        warn!(
            assignment_id = %self.cohort.assignment_id,
            "Pool-bound quarantine reconciliation was dropped before pool retention"
        );
    }
}

#[derive(Debug)]
struct ReplicaState {
    metadata: DecoderReplicaMetadata,
    availability: DecoderAvailability,
    pending_admissions: usize,
    pending_child_requests: usize,
    pending_reserved_kv_tokens: usize,
    pending_remaining_decode_tokens: usize,
    active_cohorts: usize,
    active_child_requests: usize,
    quiescing_cohorts: usize,
    quarantined_cohorts: usize,
    reserved_kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Debug)]
struct RequestChainRecord {
    request_id: Arc<str>,
    state: RequestChainState,
    owner_alive: bool,
    failed_decoders: HashSet<DecoderId>,
    used_grants: HashMap<DecoderAllocationKey, DecoderGrantDigest>,
    resolved_admissions: HashMap<Uuid, PendingAdmissionReconciliationRecord>,
}

#[derive(Debug)]
struct PendingAdmissionRecord {
    decoder_id: DecoderId,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    charge: PendingSchedulingCharge,
    retry_constraint: AdmissionRetryConstraint,
    reservation: Option<Arc<DecoderGrantReservation>>,
    claim_id: Option<Uuid>,
    authority: PendingAdmissionAuthority,
    reconciliation: Option<PendingAdmissionReconciliationRecord>,
}

enum PendingAuthorityProof {
    Reserve(DecoderReserveRefusalReceipt),
    Cancellation(PendingCancellationProof),
}

impl PendingAdmissionRecord {
    fn authority_proof(&self) -> Option<PendingAuthorityProof> {
        match &self.authority {
            PendingAdmissionAuthority::ReserveProof(receipt) => {
                Some(PendingAuthorityProof::Reserve(receipt.as_ref().clone()))
            }
            PendingAdmissionAuthority::CancellationProof(proof) => {
                Some(PendingAuthorityProof::Cancellation(proof.as_ref().clone()))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PendingAdmissionReconciliationRecord {
    Refusal(DecoderReserveRefusalReceipt),
    Cancellation {
        disposition: PendingAdmissionDisposition,
        target: PreparedGrantCancellationTarget,
        proof: Option<Box<PendingCancellationProof>>,
    },
}

#[derive(Debug)]
struct AssignmentRecord {
    chain_id: Uuid,
    binding: DecoderGrantBinding,
    phase: CohortPhase,
    terminal_reconciliation: Option<TerminalReconciliationRecord>,
    child_count: usize,
    kv_tokens: usize,
    remaining_decode_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TerminalReconciliationRecord {
    Cancellation {
        disposition: RetryDisposition,
        proof: Option<EngineReleaseReceipt>,
    },
    Completion {
        proof: Option<EngineCompletionOutcome>,
    },
    Abort {
        disposition: RetryDisposition,
        proof: Option<EngineAbortOutcome>,
    },
    Quarantine {
        proof: Option<EngineQuarantineReceipt>,
    },
}

enum TerminalApplication {
    Release {
        expected_phase: CohortPhase,
        disposition: RetryDisposition,
    },
    Quarantine {
        expected_phase: CohortPhase,
    },
}

struct LiveAssignmentLedger {
    decoder_id: DecoderId,
    chain_id: Uuid,
    child_count: usize,
    kv_tokens: usize,
    remaining_decode_tokens: usize,
    rooms: Vec<u64>,
    allocations: Vec<DecoderAllocationKey>,
}

#[derive(Debug)]
struct PoolState {
    prefill_id: PrefillId,
    declared_prefill_tp_size: NonZeroUsize,
    compatibility: EngineCompatibilityMetadata,
    accepting_requests: bool,
    last_scheduled_decoder: Option<DecoderId>,
    replicas: HashMap<DecoderId, ReplicaState>,
    request_chains: HashMap<Uuid, RequestChainRecord>,
    active_request_ids: HashMap<Arc<str>, Uuid>,
    assignments: HashMap<Uuid, AssignmentRecord>,
    room_owners: HashMap<(DecoderId, u64), Uuid>,
    allocation_owners: HashMap<DecoderAllocationKey, Uuid>,
}

#[derive(Debug)]
struct DecoderPoolInner {
    pool_id: Uuid,
    state: Mutex<PoolState>,
}

impl DecoderPoolInner {
    fn checkout_initial_reserve(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        client: &DecoderGrantControlClient,
        authorization: DecoderControlAuthorization,
    ) -> Result<ReserveReconciliationGrant, DecoderPoolError> {
        let mut state = self.state.lock();
        let record = pending_record_mut(&mut state, pending)?;
        if !matches!(record.authority, PendingAdmissionAuthority::ReserveReady) {
            return Err(DecoderPoolError::PendingReservationAlreadyStarted(
                pending.reservation_attempt_id,
            ));
        }
        let reservation = Arc::clone(record.reservation.as_ref().ok_or({
            DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id: pending.reservation_attempt_id,
                reason: "fresh pending attempt has no immutable reservation",
            }
        })?);
        record.authority = PendingAdmissionAuthority::ReserveCheckedOut { operation_id };
        Ok(client.begin_authorized_reserve(reservation, authorization))
    }

    fn checkout_reserve_retry(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
    ) -> Result<ReserveReconciliationGrant, DecoderPoolError> {
        let mut state = self.state.lock();
        let record = pending_record_mut(&mut state, pending)?;
        if !matches!(record.authority, PendingAdmissionAuthority::ReserveRetry(_)) {
            return Err(DecoderPoolError::PendingReservationAlreadyStarted(
                pending.reservation_attempt_id,
            ));
        }
        let authority = std::mem::replace(
            &mut record.authority,
            PendingAdmissionAuthority::ReserveCheckedOut { operation_id },
        );
        let PendingAdmissionAuthority::ReserveRetry(engine) = authority else {
            unreachable!("reserve retry authority changed while the pool mutex was held");
        };
        Ok(*engine)
    }

    fn complete_reserve_checkout(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        authority: PendingAdmissionAuthority,
    ) {
        let mut state = self.state.lock();
        let record = pending_record_mut(&mut state, pending)
            .expect("live reserve wrapper must match its pending admission");
        assert!(matches!(
            record.authority,
            PendingAdmissionAuthority::ReserveCheckedOut {
                operation_id: checked_out_id
            } if checked_out_id == operation_id
        ));
        record.authority = authority;
    }

    fn install_reserve_refusal_and_apply(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        receipt: DecoderReserveRefusalReceipt,
    ) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
        let mut state = self.state.lock();
        {
            let record = pending_record_mut(&mut state, pending)
                .expect("live reserve wrapper must match its pending admission");
            assert!(matches!(
                record.authority,
                PendingAdmissionAuthority::ReserveCheckedOut {
                    operation_id: checked_out_id
                } if checked_out_id == operation_id
            ));
            record.authority = PendingAdmissionAuthority::ReserveProof(Box::new(receipt));
        }
        let receipt = match &pending_record(&state, pending)?.authority {
            PendingAdmissionAuthority::ReserveProof(receipt) => receipt.as_ref().clone(),
            _ => unreachable!("reserve proof changed while the pool mutex was held"),
        };
        validate_reserve_refusal_receipt(&state, pending, &receipt)?;
        install_pending_reconciliation(
            &mut state,
            pending,
            PendingAdmissionReconciliationRecord::Refusal(receipt),
        )?;
        apply_pending_reconciliation(&mut state, pending)
    }

    fn restore_reserve_checkout(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        engine: ReserveReconciliationGrant,
    ) {
        let mut state = self.state.lock();
        let record = pending_record_mut(&mut state, pending)
            .expect("live reserve wrapper must match its pending admission");
        assert!(matches!(
            record.authority,
            PendingAdmissionAuthority::ReserveCheckedOut {
                operation_id: checked_out_id
            } if checked_out_id == operation_id
        ));
        record.authority = PendingAdmissionAuthority::ReserveRetry(Box::new(engine));
    }

    fn rollback_checked_out_reserve(&self, pending: &PendingAdmission, operation_id: Uuid) {
        let mut state = self.state.lock();
        {
            let record = pending_record(&state, pending)
                .expect("live reserve wrapper must match its pending admission");
            assert!(matches!(
                record.authority,
                PendingAdmissionAuthority::ReserveCheckedOut {
                    operation_id: checked_out_id
                } if checked_out_id == operation_id
            ));
        }
        rollback_pending_record(&mut state, pending);
    }

    fn restore_cancellation_checkout(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        engine: PendingCancellationEngine,
    ) {
        let mut state = self.state.lock();
        let record = pending_record_mut(&mut state, pending)
            .expect("live cancellation wrapper must match its pending admission");
        assert!(matches!(
            record.authority,
            PendingAdmissionAuthority::CancellationCheckedOut {
                operation_id: checked_out_id,
                ..
            } if checked_out_id == operation_id
        ));
        record.authority = PendingAdmissionAuthority::CancellationRetry(engine);
    }

    fn install_pending_cancellation_and_apply(
        &self,
        pending: &PendingAdmission,
        operation_id: Uuid,
        proof: PendingCancellationProof,
    ) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
        let mut state = self.state.lock();
        {
            let record = pending_record_mut(&mut state, pending)
                .expect("live cancellation wrapper must match its pending admission");
            let expected_kind = match &proof {
                PendingCancellationProof::Unbound(_) => PendingCancellationKind::Unbound,
                PendingCancellationProof::Bound(_) => PendingCancellationKind::Bound,
            };
            assert!(matches!(
                record.authority,
                PendingAdmissionAuthority::CancellationCheckedOut {
                    operation_id: checked_out_id,
                    kind,
                } if checked_out_id == operation_id && kind == expected_kind
            ));
            record.authority = PendingAdmissionAuthority::CancellationProof(Box::new(proof));
        }
        let proof = match &pending_record(&state, pending)?.authority {
            PendingAdmissionAuthority::CancellationProof(proof) => proof.as_ref().clone(),
            _ => unreachable!("cancellation proof changed while the pool mutex was held"),
        };
        install_pending_cancellation_proof(&mut state, pending, &proof)?;
        apply_pending_reconciliation(&mut state, pending)
    }

    fn release_pending_claim(&self, pending: &PendingAdmission) {
        let mut state = self.state.lock();
        let Some(chain) = state.request_chains.get(&pending.chain_id) else {
            return;
        };
        if chain
            .resolved_admissions
            .contains_key(&pending.reservation_attempt_id)
        {
            return;
        }
        let record = pending_record(&state, pending)
            .expect("live pending lease must match its pool-owned record");
        if matches!(record.authority, PendingAdmissionAuthority::ReserveReady) {
            rollback_pending_record(&mut state, pending);
            return;
        }
        assert!(!matches!(
            record.authority,
            PendingAdmissionAuthority::ReserveCheckedOut { .. }
                | PendingAdmissionAuthority::CancellationCheckedOut { .. }
        ));
        pending_record_mut(&mut state, pending)
            .expect("validated pending lease disappeared while the pool mutex was held")
            .claim_id = None;
    }

    fn drop_request_owner(&self, chain_id: Uuid) {
        let mut state = self.state.lock();
        let remove_now = match state.request_chains.get_mut(&chain_id) {
            Some(chain)
                if matches!(
                    &chain.state,
                    RequestChainState::IdleOpen(_) | RequestChainState::Terminal
                ) =>
            {
                true
            }
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

trait ActiveEngineGrant {
    fn grant_id(&self) -> Uuid;

    fn grant_digest(&self) -> DecoderGrantDigest;

    fn begin_abort(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError>;

    fn begin_quarantine(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError>;
}

impl ActiveEngineGrant for PromotionReconciliationGrant {
    fn grant_id(&self) -> Uuid {
        self.grant_id()
    }

    fn grant_digest(&self) -> DecoderGrantDigest {
        self.grant_digest()
    }

    fn begin_abort(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        PromotionReconciliationGrant::begin_abort(self, pool_binding, reason_code, diagnostic)
    }

    fn begin_quarantine(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        PromotionReconciliationGrant::begin_quarantine(self, pool_binding, reason_code, diagnostic)
    }
}

impl ActiveEngineGrant for RetainedEngineGrant {
    fn grant_id(&self) -> Uuid {
        self.grant_id()
    }

    fn grant_digest(&self) -> DecoderGrantDigest {
        self.grant_digest()
    }

    fn begin_abort(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        RetainedEngineGrant::begin_abort(self, pool_binding, reason_code, diagnostic)
    }

    fn begin_quarantine(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        RetainedEngineGrant::begin_quarantine(self, pool_binding, reason_code, diagnostic)
    }
}

impl DecoderPool {
    /// Construct an empty pool from declared prefill metadata.
    pub(super) fn new(
        prefill_id: PrefillId,
        declared_prefill_tp_size: usize,
        compatibility: EngineCompatibilityMetadata,
    ) -> Result<Self, DecoderPoolError> {
        if !matches!(declared_prefill_tp_size, 1 | 2 | 4) {
            return Err(DecoderPoolError::InvalidConfiguration(
                "declared prefill tensor parallel size must be 1, 2, or 4".to_string(),
            ));
        }
        let declared_prefill_tp_size =
            NonZeroUsize::new(declared_prefill_tp_size).expect("1, 2, and 4 are nonzero");
        Ok(Self {
            inner: Arc::new(DecoderPoolInner {
                pool_id: Uuid::new_v4(),
                state: Mutex::new(PoolState {
                    prefill_id,
                    declared_prefill_tp_size,
                    compatibility,
                    accepting_requests: true,
                    last_scheduled_decoder: None,
                    replicas: HashMap::new(),
                    request_chains: HashMap::new(),
                    active_request_ids: HashMap::new(),
                    assignments: HashMap::new(),
                    room_owners: HashMap::new(),
                    allocation_owners: HashMap::new(),
                }),
            }),
        })
    }

    /// Register declared metadata for a decoder process generation.
    pub(super) fn register(
        &self,
        metadata: DecoderReplicaMetadata,
    ) -> Result<(), DecoderPoolError> {
        self.register_with_availability(metadata, DecoderAvailability::Ready)
    }

    /// Install a decoder generation without making it admission-selectable.
    pub(super) fn register_unavailable(
        &self,
        metadata: DecoderReplicaMetadata,
    ) -> Result<(), DecoderPoolError> {
        self.register_with_availability(metadata, DecoderAvailability::Unavailable)
    }

    fn register_with_availability(
        &self,
        metadata: DecoderReplicaMetadata,
        availability: DecoderAvailability,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        if state.replicas.contains_key(&metadata.id) {
            return Err(DecoderPoolError::DuplicateDecoder(metadata.id));
        }
        if !matches!(metadata.declared_decode_tp_size.get(), 1 | 2) {
            return Err(DecoderPoolError::IneligibleDecoderMetadata {
                decoder_id: metadata.id,
                reason: format!(
                    "this pool supports declared TP1 or TP2 decoders, received TP{}",
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
                reason: "model, KV layout, dtype, page size, wire protocol, or prepared-grant protocol metadata differs".to_string(),
            });
        }

        state.replicas.insert(
            metadata.id.clone(),
            ReplicaState {
                metadata,
                availability,
                pending_admissions: 0,
                pending_child_requests: 0,
                pending_reserved_kv_tokens: 0,
                pending_remaining_decode_tokens: 0,
                active_cohorts: 0,
                active_child_requests: 0,
                quiescing_cohorts: 0,
                quarantined_cohorts: 0,
                reserved_kv_tokens: 0,
                remaining_decode_tokens: 0,
            },
        );
        Ok(())
    }

    /// Drain one generation and activate its replacement under one pool lock.
    pub(crate) fn activate_replacement(
        &self,
        draining_id: &DecoderId,
        replacement_id: &DecoderId,
    ) -> Result<(), DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let draining = state
            .replicas
            .get(draining_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(draining_id.clone()))?;
        let replacement = state
            .replicas
            .get(replacement_id)
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(replacement_id.clone()))?;
        if replacement.availability != DecoderAvailability::Unavailable {
            return Err(DecoderPoolError::InvalidConfiguration(format!(
                "replacement decoder {replacement_id} must be unavailable before activation"
            )));
        }
        if draining.availability == DecoderAvailability::Unavailable {
            return Err(DecoderPoolError::InvalidConfiguration(format!(
                "decoder {draining_id} cannot be replaced while unavailable"
            )));
        }

        state
            .replicas
            .get_mut(draining_id)
            .expect("draining decoder was validated under the same pool lock")
            .availability = DecoderAvailability::Draining;
        state
            .replicas
            .get_mut(replacement_id)
            .expect("replacement decoder was validated under the same pool lock")
            .availability = DecoderAvailability::Ready;
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
        if !state.accepting_requests {
            return Err(DecoderPoolError::PrefillPoolDraining);
        }
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
                state: RequestChainState::IdleOpen(AdmissionRetryConstraint::AnyEligible),
                owner_alive: true,
                failed_decoders: HashSet::new(),
                used_grants: HashMap::new(),
                resolved_admissions: HashMap::new(),
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
        match &chain.state {
            RequestChainState::Reserving(pending) => {
                return Err(DecoderPoolError::RequestHasPendingAdmission {
                    request_id: chain.request_id.to_string(),
                    reservation_attempt_id: pending.reservation_attempt_id,
                });
            }
            RequestChainState::Assigned(assignment_id)
            | RequestChainState::Quarantined(assignment_id) => {
                return Err(DecoderPoolError::RequestHasActiveCohort {
                    request_id: chain.request_id.to_string(),
                    assignment_id: *assignment_id,
                });
            }
            RequestChainState::IdleOpen(_) | RequestChainState::Terminal => {}
        }
        remove_request_chain(&mut state, request.chain_id);
        request.finalized = true;
        Ok(())
    }

    /// Change whether a registered decoder accepts new cohorts.
    pub(super) fn set_availability(
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

    /// Close this prefill generation to new logical request ownership.
    pub(crate) fn begin_draining(&self) {
        self.inner.state.lock().accepting_requests = false;
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

    /// Prove that this prefill pool retains no request or transfer ownership.
    pub(crate) fn ensure_retirable(&self) -> Result<(), DecoderPoolError> {
        let state = self.inner.state.lock();
        let pending_admissions = state
            .replicas
            .values()
            .map(|replica| replica.pending_admissions)
            .sum::<usize>();
        let quarantined_cohorts = state
            .replicas
            .values()
            .map(|replica| replica.quarantined_cohorts)
            .sum();
        if state.request_chains.is_empty()
            && state.assignments.is_empty()
            && state.room_owners.is_empty()
            && state.allocation_owners.is_empty()
            && pending_admissions == 0
            && quarantined_cohorts == 0
        {
            return Ok(());
        }
        Err(DecoderPoolError::PrefillPoolInUse {
            request_chains: state.request_chains.len(),
            pending_admissions,
            assignments: state.assignments.len(),
            room_owners: state.room_owners.len(),
            allocation_owners: state.allocation_owners.len(),
            quarantined_cohorts,
        })
    }

    /// Remove a draining process generation after every cohort is terminal.
    pub(super) fn remove(&self, decoder_id: &DecoderId) -> Result<(), DecoderPoolError> {
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
        if replica.active_cohorts > 0 || replica.pending_admissions > 0 {
            return Err(DecoderPoolError::DecoderInUse {
                decoder_id: decoder_id.clone(),
                active_cohorts: replica.active_cohorts,
                pending_admissions: replica.pending_admissions,
            });
        }
        state.replicas.remove(decoder_id);
        Ok(())
    }

    /// Atomically select, construct, and charge one exact reservation attempt.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_admission(
        &self,
        request: &LogicalRequestOwner,
        eligible_decoders: &HashSet<DecoderId>,
        template: &DecoderRequestTemplate,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        prepared_ttl: Duration,
        charge: PendingSchedulingCharge,
    ) -> Result<PendingAdmission, DecoderPoolError> {
        self.validate_request_owner(request)?;
        if charge.child_requests() != template.child_count() {
            return Err(DecoderPoolError::PendingChildCountMismatch {
                pending_children: charge.child_requests(),
                request_children: template.child_count(),
            });
        }

        let mut state = self.inner.state.lock();
        let (retry_constraint, failed_decoders) = {
            let chain = state
                .request_chains
                .get(&request.chain_id)
                .ok_or(DecoderPoolError::UnknownRequestChain(request.chain_id))?;
            let retry_constraint = match &chain.state {
                RequestChainState::IdleOpen(retry_constraint) => retry_constraint.clone(),
                RequestChainState::Reserving(pending) => {
                    return Err(DecoderPoolError::RequestHasPendingAdmission {
                        request_id: chain.request_id.to_string(),
                        reservation_attempt_id: pending.reservation_attempt_id,
                    });
                }
                RequestChainState::Assigned(assignment_id)
                | RequestChainState::Quarantined(assignment_id) => {
                    return Err(DecoderPoolError::RequestHasActiveCohort {
                        request_id: chain.request_id.to_string(),
                        assignment_id: *assignment_id,
                    });
                }
                RequestChainState::Terminal => {
                    return Err(DecoderPoolError::RequestChainTerminal(
                        chain.request_id.to_string(),
                    ));
                }
            };
            (retry_constraint, chain.failed_decoders.clone())
        };

        let decoder_id = match &retry_constraint {
            AdmissionRetryConstraint::AnyEligible => select_decoders(
                &state.replicas,
                &failed_decoders,
                eligible_decoders,
                state.last_scheduled_decoder.as_ref(),
            )?
            .into_iter()
            .next()
            .expect("successful decoder selection cannot be empty"),
            AdmissionRetryConstraint::SameDecoder(decoder_id) => {
                let ready = state.replicas.get(decoder_id).is_some_and(|replica| {
                    eligible_decoders.contains(decoder_id)
                        && replica.availability == DecoderAvailability::Ready
                        && !failed_decoders.contains(decoder_id)
                });
                if !ready {
                    return Err(DecoderPoolError::RetryDecoderUnavailable(
                        decoder_id.clone(),
                    ));
                }
                decoder_id.clone()
            }
        };

        let reservation = template
            .prepare_reservation(
                state.prefill_id.clone(),
                prefill_bootstrap_endpoint,
                decoder_id.clone(),
                request.chain_id,
                state.declared_prefill_tp_size.get(),
                prepared_ttl,
            )
            .map_err(|error| DecoderPoolError::InvalidGrant(error.to_string()))?;
        let reservation_attempt_id = reservation.reservation_attempt_id();
        let reserve_attempt_digest = reservation.reserve_attempt_digest();
        let reservation = Arc::new(reservation);
        let claim_id = Uuid::new_v4();

        install_pending_attempt(
            &mut state,
            request.chain_id,
            decoder_id.clone(),
            reservation_attempt_id,
            reserve_attempt_digest,
            charge,
            retry_constraint.clone(),
            Some(Arc::clone(&reservation)),
            claim_id,
            PendingAdmissionAuthority::ReserveReady,
        )?;

        Ok(PendingAdmission {
            inner: Arc::clone(&self.inner),
            chain_id: request.chain_id,
            claim_id,
            decoder_id,
            reservation_attempt_id,
            reserve_attempt_digest,
            charge,
            retry_constraint,
            resolved: false,
        })
    }

    /// Reclaim an unowned pending attempt for its live logical request.
    pub fn recover_pending_admission(
        &self,
        request: &LogicalRequestOwner,
        reservation_attempt_id: Uuid,
    ) -> Result<PendingAdmission, DecoderPoolError> {
        self.validate_request_owner(request)?;
        let mut state = self.inner.state.lock();
        let chain = state
            .request_chains
            .get_mut(&request.chain_id)
            .ok_or(DecoderPoolError::UnknownRequestChain(request.chain_id))?;
        let record = match &mut chain.state {
            RequestChainState::Reserving(record)
                if record.reservation_attempt_id == reservation_attempt_id =>
            {
                record
            }
            _ => {
                return Err(DecoderPoolError::UnknownPendingAdmission(
                    reservation_attempt_id,
                ));
            }
        };
        issue_pending_claim(Arc::clone(&self.inner), request.chain_id, record)
    }

    /// Reclaim an unowned pending attempt after its request owner has gone away.
    pub fn recover_orphaned_pending_admission(
        &self,
        reservation_attempt_id: Uuid,
    ) -> Result<PendingAdmission, DecoderPoolError> {
        let mut state = self.inner.state.lock();
        let chain_id = state
            .request_chains
            .iter()
            .find_map(|(chain_id, chain)| match &chain.state {
                RequestChainState::Reserving(record)
                    if record.reservation_attempt_id == reservation_attempt_id =>
                {
                    Some(*chain_id)
                }
                _ => None,
            })
            .ok_or(DecoderPoolError::UnknownPendingAdmission(
                reservation_attempt_id,
            ))?;
        let chain = state
            .request_chains
            .get_mut(&chain_id)
            .expect("located orphaned pending chain disappeared while the pool mutex was held");
        if chain.owner_alive {
            return Err(DecoderPoolError::PendingAdmissionOwnerStillAlive(
                reservation_attempt_id,
            ));
        }
        let RequestChainState::Reserving(record) = &mut chain.state else {
            unreachable!("located pending attempt changed while the pool mutex was held");
        };
        issue_pending_claim(Arc::clone(&self.inner), chain_id, record)
    }

    #[cfg(test)]
    pub(crate) fn install_reserve_refusal_proof(
        &self,
        pending: &mut PendingAdmission,
        receipt: &DecoderReserveRefusalReceipt,
    ) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        let mut state = self.inner.state.lock();
        if !pending_is_resolved(&state, pending) {
            pending_record_mut(&mut state, pending)?.authority =
                PendingAdmissionAuthority::ReserveProof(Box::new(receipt.clone()));
        }
        validate_reserve_refusal_receipt(&state, pending, receipt)?;
        let proof = PendingAdmissionReconciliationRecord::Refusal(receipt.clone());
        install_pending_reconciliation(&mut state, pending, proof)?;
        let disposition = apply_pending_reconciliation(&mut state, pending)?;
        pending.resolved = true;
        Ok(disposition)
    }

    #[cfg(test)]
    fn pin_pending_cancellation(
        &self,
        pending: &PendingAdmission,
        target: &PreparedGrantCancellationTarget,
        disposition: PendingAdmissionDisposition,
    ) -> Result<PendingCancellationPin, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        validate_pending_cancellation_target(pending, target)?;
        let mut state = self.inner.state.lock();
        install_pending_cancellation_intent(&mut state, pending, target, disposition)?;
        Ok(PendingCancellationPin::new(target.clone()))
    }

    /// Atomically pin and begin cancellation of an unbound PREPARED grant.
    pub fn begin_unbound_pending_cancellation<'a>(
        &self,
        pending: &'a mut PendingAdmission,
        grant: &mut UnboundPreparedGrant,
        disposition: PendingAdmissionDisposition,
    ) -> Result<PendingCancellationReconciliation<'a>, DecoderPoolError> {
        let target = grant
            .cancellation_target()
            .map_err(|error| DecoderPoolError::InvalidGrant(error.to_string()))?;
        self.validate_pending_pool(pending)?;
        validate_pending_cancellation_target(pending, &target)?;
        let operation_id = Uuid::new_v4();
        let mut state = self.inner.state.lock();
        ensure_prepared_pending_authority(&state, pending)?;
        install_pending_cancellation_intent(&mut state, pending, &target, disposition)?;
        let pin = PendingCancellationPin::new(target);
        let engine = grant
            .begin_pending_cancellation(pin)
            .expect("validated unbound cancellation target lost engine authority");
        pending_record_mut(&mut state, pending)?.authority =
            PendingAdmissionAuthority::CancellationCheckedOut {
                operation_id,
                kind: PendingCancellationKind::Unbound,
            };
        drop(state);
        Ok(PendingCancellationReconciliation {
            pending,
            operation_id,
            engine: Some(PendingCancellationEngine::Unbound(Box::new(engine))),
            complete: false,
        })
    }

    /// Atomically pin and begin cancellation after failed or ambiguous bind I/O.
    pub fn begin_bind_pending_cancellation<'a>(
        &self,
        pending: &'a mut PendingAdmission,
        grant: &mut BindReconciliationGrant,
        disposition: PendingAdmissionDisposition,
    ) -> Result<PendingCancellationReconciliation<'a>, DecoderPoolError> {
        let target = grant
            .cancellation_target()
            .map_err(|error| DecoderPoolError::InvalidGrant(error.to_string()))?;
        self.validate_pending_pool(pending)?;
        validate_pending_cancellation_target(pending, &target)?;
        let operation_id = Uuid::new_v4();
        let mut state = self.inner.state.lock();
        ensure_prepared_pending_authority(&state, pending)?;
        install_pending_cancellation_intent(&mut state, pending, &target, disposition)?;
        let pin = PendingCancellationPin::new(target);
        let engine = grant
            .begin_pending_cancellation(pin)
            .expect("validated bind cancellation target lost engine authority");
        pending_record_mut(&mut state, pending)?.authority =
            PendingAdmissionAuthority::CancellationCheckedOut {
                operation_id,
                kind: PendingCancellationKind::Unbound,
            };
        drop(state);
        Ok(PendingCancellationReconciliation {
            pending,
            operation_id,
            engine: Some(PendingCancellationEngine::Unbound(Box::new(engine))),
            complete: false,
        })
    }

    /// Atomically pin and begin cancellation of a bound PREPARED grant.
    pub fn begin_bound_pending_cancellation<'a>(
        &self,
        pending: &'a mut PendingAdmission,
        grant: &mut BoundPreparedGrant,
        disposition: PendingAdmissionDisposition,
    ) -> Result<PendingCancellationReconciliation<'a>, DecoderPoolError> {
        let target = grant
            .cancellation_target()
            .map_err(|error| DecoderPoolError::InvalidGrant(error.to_string()))?;
        self.validate_pending_pool(pending)?;
        validate_pending_cancellation_target(pending, &target)?;
        let operation_id = Uuid::new_v4();
        let mut state = self.inner.state.lock();
        ensure_prepared_pending_authority(&state, pending)?;
        install_pending_cancellation_intent(&mut state, pending, &target, disposition)?;
        let pin = PendingCancellationPin::new(target);
        let engine = grant
            .begin_pending_cancellation(pin)
            .expect("validated bound cancellation target lost engine authority");
        pending_record_mut(&mut state, pending)?.authority =
            PendingAdmissionAuthority::CancellationCheckedOut {
                operation_id,
                kind: PendingCancellationKind::Bound,
            };
        drop(state);
        Ok(PendingCancellationReconciliation {
            pending,
            operation_id,
            engine: Some(PendingCancellationEngine::Bound(Box::new(engine))),
            complete: false,
        })
    }

    /// Resume exact pending-cancellation authority recovered from a dropped task.
    pub fn resume_pending_cancellation<'a>(
        &self,
        pending: &'a mut PendingAdmission,
    ) -> Result<PendingCancellationReconciliation<'a>, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        let operation_id = Uuid::new_v4();
        let mut state = self.inner.state.lock();
        let record = pending_record_mut(&mut state, pending)?;
        if !matches!(
            record.authority,
            PendingAdmissionAuthority::CancellationRetry(_)
        ) {
            return Err(DecoderPoolError::PendingReservationAlreadyStarted(
                pending.reservation_attempt_id,
            ));
        }
        let authority = std::mem::replace(
            &mut record.authority,
            PendingAdmissionAuthority::CancellationCheckedOut {
                operation_id,
                kind: PendingCancellationKind::Unbound,
            },
        );
        let PendingAdmissionAuthority::CancellationRetry(engine) = authority else {
            unreachable!("cancellation retry authority changed while the pool mutex was held");
        };
        let kind = match &engine {
            PendingCancellationEngine::Unbound(_) => PendingCancellationKind::Unbound,
            PendingCancellationEngine::Bound(_) => PendingCancellationKind::Bound,
        };
        record.authority = PendingAdmissionAuthority::CancellationCheckedOut { operation_id, kind };
        drop(state);
        Ok(PendingCancellationReconciliation {
            pending,
            operation_id,
            engine: Some(engine),
            complete: false,
        })
    }

    #[cfg(test)]
    fn install_pending_cancellation_proof(
        &self,
        pending: &mut PendingAdmission,
        proof: &PendingCancellationProof,
    ) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        let mut state = self.inner.state.lock();
        if !pending_is_resolved(&state, pending) {
            pending_record_mut(&mut state, pending)?.authority =
                PendingAdmissionAuthority::CancellationProof(Box::new(proof.clone()));
        }
        install_pending_cancellation_proof(&mut state, pending, proof)?;
        let disposition = apply_pending_reconciliation(&mut state, pending)?;
        pending.resolved = true;
        Ok(disposition)
    }

    /// Retry pool application of an already installed authoritative proof.
    pub fn resume_pending_admission(
        &self,
        pending: &mut PendingAdmission,
    ) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        let mut state = self.inner.state.lock();
        if !pending_is_resolved(&state, pending) {
            let authority = pending_record(&state, pending)?.authority_proof();
            match authority {
                Some(PendingAuthorityProof::Reserve(receipt)) => {
                    validate_reserve_refusal_receipt(&state, pending, &receipt)?;
                    install_pending_reconciliation(
                        &mut state,
                        pending,
                        PendingAdmissionReconciliationRecord::Refusal(receipt),
                    )?;
                }
                Some(PendingAuthorityProof::Cancellation(proof)) => {
                    install_pending_cancellation_proof(&mut state, pending, &proof)?;
                }
                None => {}
            }
        }
        let disposition = apply_pending_reconciliation(&mut state, pending)?;
        pending.resolved = true;
        Ok(disposition)
    }

    /// Atomically convert an exact pending attempt into its allocator-issued grant.
    pub fn bind_grant(
        &self,
        pending: &mut PendingAdmission,
        grant: &mut BoundPreparedGrant,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        self.validate_pending_pool(pending)?;
        let binding = grant.binding().clone();
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
        if binding.request_chain_id() != pending.chain_id {
            return Err(DecoderPoolError::GrantRequestMismatch {
                expected: pending.chain_id,
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
        {
            let chain = state
                .request_chains
                .get(&pending.chain_id)
                .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
            let record = match &chain.state {
                RequestChainState::Reserving(record) => record,
                _ => {
                    return Err(DecoderPoolError::UnknownPendingAdmission(
                        pending.reservation_attempt_id,
                    ));
                }
            };
            validate_pending_record(record, pending)?;
            if !matches!(record.authority, PendingAdmissionAuthority::PreparedIssued) {
                return Err(DecoderPoolError::PendingAdmissionProofPending(
                    pending.reservation_attempt_id,
                ));
            }
            if record.reconciliation.is_some() {
                return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
                    pending.reservation_attempt_id,
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
        }

        if binding.decoder_id() != &pending.decoder_id
            || binding.reservation_attempt_id() != pending.reservation_attempt_id
            || binding.reserve_attempt_digest() != pending.reserve_attempt_digest
        {
            return Err(DecoderPoolError::GrantDecoderIneligible(
                binding.decoder_id().clone(),
            ));
        }
        if accounting.child_count() != pending.charge.child_requests() {
            return Err(DecoderPoolError::PendingChildCountMismatch {
                pending_children: pending.charge.child_requests(),
                request_children: accounting.child_count(),
            });
        }
        validate_pending_decoder_ledger(
            &state,
            binding.decoder_id(),
            pending.reservation_attempt_id,
        )?;
        let replica = state
            .replicas
            .get(binding.decoder_id())
            .ok_or_else(|| DecoderPoolError::UnknownDecoder(binding.decoder_id().clone()))?;
        if let Some(room) = binding.bootstrap_rooms().iter().find(|room| {
            state
                .room_owners
                .contains_key(&(binding.decoder_id().clone(), **room))
        }) {
            return Err(DecoderPoolError::GrantRoomInUse {
                decoder_id: binding.decoder_id().clone(),
                room: *room,
            });
        }
        if let Some(child_index) = binding
            .allocation_keys()
            .position(|key| state.allocation_owners.contains_key(&key))
        {
            return Err(DecoderPoolError::GrantSlotGenerationInUse {
                decoder_id: binding.decoder_id().clone(),
                slot_generation: binding.slot_generations()[child_index].as_uuid(),
            });
        }

        let active_cohorts = replica.active_cohorts.checked_add(1).ok_or_else(|| {
            DecoderPoolError::InvalidGrant("active cohort accounting overflows usize".to_string())
        })?;
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
        let pool_binding = DecoderGrantPoolBinding::new(&binding);
        let prepared_grant = grant.take_for_pool_binding(pool_binding).map_err(|_| {
            DecoderPoolError::InvalidGrant(
                "prepared grant has no pool-binding authority".to_string(),
            )
        })?;

        let replica = state
            .replicas
            .get_mut(binding.decoder_id())
            .expect("pending decoder disappeared while pool lock was held");
        replica.pending_admissions -= 1;
        replica.pending_child_requests -= pending.charge.child_requests();
        replica.pending_reserved_kv_tokens -= pending.charge.reserved_kv_tokens();
        replica.pending_remaining_decode_tokens -= pending.charge.remaining_decode_tokens();
        replica.active_cohorts = active_cohorts;
        replica.active_child_requests = active_child_requests;
        replica.reserved_kv_tokens = reserved_kv_tokens;
        replica.remaining_decode_tokens = remaining_decode_tokens;
        for room in binding.bootstrap_rooms() {
            let previous = state
                .room_owners
                .insert((binding.decoder_id().clone(), *room), assignment_id);
            debug_assert!(previous.is_none());
        }
        for allocation in binding.allocation_keys() {
            let previous = state.allocation_owners.insert(allocation, assignment_id);
            debug_assert!(previous.is_none());
        }

        state.assignments.insert(
            assignment_id,
            AssignmentRecord {
                chain_id: pending.chain_id,
                binding: binding.clone(),
                phase: CohortPhase::Reserved,
                terminal_reconciliation: None,
                child_count: accounting.child_count(),
                kv_tokens: accounting.total_reserved_kv_tokens(),
                remaining_decode_tokens: accounting.total_remaining_decode_tokens(),
            },
        );
        let chain = state
            .request_chains
            .get_mut(&pending.chain_id)
            .expect("request chain disappeared while pool lock was held");
        for allocation_key in binding.allocation_keys() {
            chain.used_grants.insert(allocation_key, binding.digest());
        }
        chain.state = RequestChainState::Assigned(assignment_id);
        pending.resolved = true;

        Ok(DecoderAssignmentCohort {
            pool_id: self.inner.pool_id,
            chain_id: pending.chain_id,
            assignment_id,
            binding: binding.clone(),
            phase: CohortPhase::Reserved,
            prepared_grant: Some(prepared_grant),
        })
    }

    /// Begin cancellation bound to one exact reserved cohort.
    pub fn begin_cancellation<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<PoolCancellationReconciliation<'a>, DecoderPoolError> {
        let engine = self.pin_cancellation(cohort, disposition)?;
        Ok(PoolCancellationReconciliation {
            pool: self.clone(),
            cohort,
            engine,
            complete: false,
        })
    }

    fn pin_cancellation(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<PreparedCancellationReconciliationGrant, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        preflight_live_assignment(
            &state,
            cohort,
            CohortPhase::Reserved,
            phase_name(CohortPhase::Cancelling),
        )?;
        let prepared_grant = cohort.prepared_grant.as_ref().ok_or_else(|| {
            DecoderPoolError::InvalidGrant(
                "reserved assignment has no prepared grant authority".to_string(),
            )
        })?;
        validate_engine_grant_identity(
            prepared_grant.grant_id(),
            prepared_grant.grant_digest(),
            cohort,
            "cancellation",
        )?;
        let prepared_grant = cohort
            .prepared_grant
            .as_mut()
            .expect("prepared grant authority was validated immediately before mutation");
        let pool_binding = DecoderGrantPoolBinding::new(&cohort.binding);
        let cancellation = prepared_grant
            .begin_pool_cancellation(pool_binding)
            .map_err(|_| {
                DecoderPoolError::InvalidGrant(
                    "matching prepared grant has no cancellation authority".to_string(),
                )
            })?;
        cohort.prepared_grant = None;
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment was preflighted under the same pool lock");
        record.phase = CohortPhase::Cancelling;
        record.terminal_reconciliation = Some(TerminalReconciliationRecord::Cancellation {
            disposition,
            proof: None,
        });
        cohort.phase = CohortPhase::Cancelling;
        Ok(cancellation)
    }

    /// Atomically activate a cohort and consume its exact grant into promotion.
    ///
    /// A mismatched cohort/grant pair or unavailable grant authority leaves both
    /// inputs reserved and unchanged. On success, prepared cancellation is no
    /// longer available and the returned capability is the only engine control
    /// authority.
    pub fn begin_promotion(
        &self,
        cohort: &mut DecoderAssignmentCohort,
    ) -> Result<PromotionReconciliationGrant, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let ledger = preflight_live_assignment(
            &state,
            cohort,
            CohortPhase::Reserved,
            phase_name(CohortPhase::Active),
        )?;
        if !state
            .request_chains
            .get(&ledger.chain_id)
            .expect("request chain was preflighted under the same pool lock")
            .owner_alive
        {
            return Err(DecoderPoolError::RequestChainOwnerDropped(ledger.chain_id));
        }
        let prepared_grant = cohort.prepared_grant.as_ref().ok_or_else(|| {
            DecoderPoolError::InvalidGrant(
                "reserved assignment has no prepared grant authority".to_string(),
            )
        })?;
        validate_engine_grant_identity(
            prepared_grant.grant_id(),
            prepared_grant.grant_digest(),
            cohort,
            "promotion",
        )?;
        let prepared_grant = cohort
            .prepared_grant
            .as_mut()
            .expect("prepared grant authority was validated immediately before mutation");
        let pool_binding = DecoderGrantPoolBinding::new(&cohort.binding);
        let promotion = prepared_grant.begin_promotion(pool_binding).map_err(|_| {
            DecoderPoolError::InvalidGrant(
                "matching prepared grant has no promotion authority".to_string(),
            )
        })?;
        cohort.prepared_grant = None;
        state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment was preflighted under the same pool lock")
            .phase = CohortPhase::Active;
        cohort.phase = CohortPhase::Active;
        Ok(promotion)
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
            if record.phase != CohortPhase::Active {
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

    /// Begin successful completion bound to one exact active cohort.
    pub fn begin_completion<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut RetainedEngineGrant,
    ) -> Result<PoolCompletionReconciliation<'a>, DecoderPoolError> {
        let engine = self.pin_completion(cohort, grant)?;
        Ok(PoolCompletionReconciliation {
            pool: self.clone(),
            cohort,
            engine,
            complete: false,
        })
    }

    fn pin_completion(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        grant: &mut RetainedEngineGrant,
    ) -> Result<CompletionReconciliationGrant, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        preflight_live_assignment(
            &state,
            cohort,
            CohortPhase::Active,
            phase_name(CohortPhase::Completing),
        )?;
        validate_engine_grant_identity(
            grant.grant_id(),
            grant.grant_digest(),
            cohort,
            "completion",
        )?;
        let pool_binding = DecoderGrantPoolBinding::new(&cohort.binding);
        let completion = grant.begin_completion(pool_binding).map_err(|_| {
            DecoderPoolError::InvalidGrant(
                "matching retained grant has no completion authority".to_string(),
            )
        })?;
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment was preflighted under the same pool lock");
        record.phase = CohortPhase::Completing;
        record.terminal_reconciliation =
            Some(TerminalReconciliationRecord::Completion { proof: None });
        cohort.phase = CohortPhase::Completing;
        Ok(completion)
    }

    /// Begin abort bound to an active cohort with ambiguous promotion.
    pub fn begin_abort_from_promotion<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut PromotionReconciliationGrant,
        reason_code: &str,
        diagnostic: Option<&str>,
        disposition: RetryDisposition,
    ) -> Result<PoolAbortReconciliation<'a>, DecoderPoolError> {
        self.begin_abort(cohort, grant, reason_code, diagnostic, disposition)
    }

    /// Begin abort bound to an active cohort with retained engine authority.
    pub fn begin_abort_from_retained<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut RetainedEngineGrant,
        reason_code: &str,
        diagnostic: Option<&str>,
        disposition: RetryDisposition,
    ) -> Result<PoolAbortReconciliation<'a>, DecoderPoolError> {
        self.begin_abort(cohort, grant, reason_code, diagnostic, disposition)
    }

    /// Begin quarantine bound to an active cohort with ambiguous promotion.
    pub fn begin_quarantine_from_promotion<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut PromotionReconciliationGrant,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<PoolQuarantineReconciliation<'a>, DecoderPoolError> {
        self.begin_quarantine(cohort, grant, reason_code, diagnostic)
    }

    /// Begin quarantine bound to an active cohort with retained engine authority.
    pub fn begin_quarantine_from_retained<'a>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut RetainedEngineGrant,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<PoolQuarantineReconciliation<'a>, DecoderPoolError> {
        self.begin_quarantine(cohort, grant, reason_code, diagnostic)
    }

    fn begin_abort<'a, G: ActiveEngineGrant>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut G,
        reason_code: &str,
        diagnostic: Option<&str>,
        disposition: RetryDisposition,
    ) -> Result<PoolAbortReconciliation<'a>, DecoderPoolError> {
        let engine = self.pin_abort(cohort, grant, reason_code, diagnostic, disposition)?;
        Ok(PoolAbortReconciliation {
            pool: self.clone(),
            cohort,
            engine,
            complete: false,
        })
    }

    fn pin_abort<G: ActiveEngineGrant>(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        grant: &mut G,
        reason_code: &str,
        diagnostic: Option<&str>,
        disposition: RetryDisposition,
    ) -> Result<AbortReconciliationGrant, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let ledger = preflight_live_assignment(
            &state,
            cohort,
            CohortPhase::Active,
            phase_name(CohortPhase::Aborting),
        )?;
        validate_engine_grant_identity(grant.grant_id(), grant.grant_digest(), cohort, "abort")?;
        let quiescing_cohorts = state
            .replicas
            .get(&ledger.decoder_id)
            .expect("assigned decoder was preflighted under the same pool lock")
            .quiescing_cohorts
            .checked_add(1)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "decoder quiescing-cohort accounting overflows usize",
            })?;
        let pool_binding = DecoderGrantPoolBinding::new(&cohort.binding);
        let abort = grant
            .begin_abort(pool_binding, reason_code, diagnostic)
            .map_err(|_| {
                DecoderPoolError::InvalidGrant(
                    "matching engine grant could not pin abort authority".to_string(),
                )
            })?;
        state
            .replicas
            .get_mut(&ledger.decoder_id)
            .expect("assigned decoder was preflighted under the same pool lock")
            .quiescing_cohorts = quiescing_cohorts;
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment was preflighted under the same pool lock");
        record.phase = CohortPhase::Aborting;
        record.terminal_reconciliation = Some(TerminalReconciliationRecord::Abort {
            disposition,
            proof: None,
        });
        cohort.phase = CohortPhase::Aborting;
        Ok(abort)
    }

    fn begin_quarantine<'a, G: ActiveEngineGrant>(
        &self,
        cohort: &'a mut DecoderAssignmentCohort,
        grant: &mut G,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<PoolQuarantineReconciliation<'a>, DecoderPoolError> {
        let engine = self.pin_quarantine(cohort, grant, reason_code, diagnostic)?;
        Ok(PoolQuarantineReconciliation {
            pool: self.clone(),
            cohort,
            engine,
            complete: false,
        })
    }

    fn pin_quarantine<G: ActiveEngineGrant>(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        grant: &mut G,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let ledger = preflight_live_assignment(
            &state,
            cohort,
            CohortPhase::Active,
            phase_name(CohortPhase::Quarantining),
        )?;
        validate_engine_grant_identity(
            grant.grant_id(),
            grant.grant_digest(),
            cohort,
            "quarantine",
        )?;
        let quiescing_cohorts = state
            .replicas
            .get(&ledger.decoder_id)
            .expect("assigned decoder was preflighted under the same pool lock")
            .quiescing_cohorts
            .checked_add(1)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "decoder quiescing-cohort accounting overflows usize",
            })?;
        let pool_binding = DecoderGrantPoolBinding::new(&cohort.binding);
        let quarantine = grant
            .begin_quarantine(pool_binding, reason_code, diagnostic)
            .map_err(|_| {
                DecoderPoolError::InvalidGrant(
                    "matching engine grant could not pin quarantine authority".to_string(),
                )
            })?;
        state
            .replicas
            .get_mut(&ledger.decoder_id)
            .expect("assigned decoder was preflighted under the same pool lock")
            .quiescing_cohorts = quiescing_cohorts;
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .expect("assignment was preflighted under the same pool lock");
        record.phase = CohortPhase::Quarantining;
        record.terminal_reconciliation =
            Some(TerminalReconciliationRecord::Quarantine { proof: None });
        cohort.phase = CohortPhase::Quarantining;
        Ok(quarantine)
    }

    fn install_cancellation_proof(
        &self,
        cohort: &DecoderAssignmentCohort,
        receipt: EngineReleaseReceipt,
    ) -> Result<(), DecoderPoolError> {
        validate_engine_release_receipt(cohort, &receipt, EngineReleaseKind::PreparedCancelled)?;
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        if cohort.phase != CohortPhase::Cancelling || record.phase != CohortPhase::Cancelling {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                "install cancellation proof",
            ));
        }
        match record.terminal_reconciliation.as_mut() {
            Some(TerminalReconciliationRecord::Cancellation { proof, .. }) => {
                install_proof_once(proof, receipt, cohort.assignment_id)
            }
            _ => Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "cancelling assignment has no matching terminal intent",
            }),
        }
    }

    fn install_completion_proof(
        &self,
        cohort: &DecoderAssignmentCohort,
        outcome: EngineCompletionOutcome,
    ) -> Result<(), DecoderPoolError> {
        match &outcome {
            EngineCompletionOutcome::Completed(receipt) => {
                validate_engine_release_receipt(cohort, receipt, EngineReleaseKind::Completed)?;
            }
            EngineCompletionOutcome::Quarantined(receipt) => {
                validate_engine_quarantine_receipt(cohort, receipt)?;
            }
        }
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        if cohort.phase != CohortPhase::Completing || record.phase != CohortPhase::Completing {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                "install completion proof",
            ));
        }
        match record.terminal_reconciliation.as_mut() {
            Some(TerminalReconciliationRecord::Completion { proof }) => {
                install_proof_once(proof, outcome, cohort.assignment_id)
            }
            _ => Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "completing assignment has no matching terminal intent",
            }),
        }
    }

    fn install_abort_proof(
        &self,
        cohort: &DecoderAssignmentCohort,
        outcome: EngineAbortOutcome,
    ) -> Result<(), DecoderPoolError> {
        match &outcome {
            EngineAbortOutcome::Aborted(receipt) => {
                validate_engine_release_receipt(cohort, receipt, EngineReleaseKind::Aborted)?;
            }
            EngineAbortOutcome::Quarantined(receipt) => {
                validate_engine_quarantine_receipt(cohort, receipt)?;
            }
        }
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        if cohort.phase != CohortPhase::Aborting || record.phase != CohortPhase::Aborting {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                "install abort proof",
            ));
        }
        match record.terminal_reconciliation.as_mut() {
            Some(TerminalReconciliationRecord::Abort { proof, .. }) => {
                install_proof_once(proof, outcome, cohort.assignment_id)
            }
            _ => Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "aborting assignment has no matching terminal intent",
            }),
        }
    }

    fn install_quarantine_proof(
        &self,
        cohort: &DecoderAssignmentCohort,
        receipt: EngineQuarantineReceipt,
    ) -> Result<(), DecoderPoolError> {
        validate_engine_quarantine_receipt(cohort, &receipt)?;
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        let record = state
            .assignments
            .get_mut(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        if cohort.phase != CohortPhase::Quarantining || record.phase != CohortPhase::Quarantining {
            return Err(invalid_transition(
                cohort.assignment_id,
                record.phase,
                "install quarantine proof",
            ));
        }
        match record.terminal_reconciliation.as_mut() {
            Some(TerminalReconciliationRecord::Quarantine { proof }) => {
                install_proof_once(proof, receipt, cohort.assignment_id)
            }
            _ => Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "quarantining assignment has no matching terminal intent",
            }),
        }
    }

    #[cfg(test)]
    fn apply_cancellation_receipt(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        validate_pinned_disposition(cohort, &self.inner, disposition, CohortPhase::Cancelling)?;
        self.install_cancellation_proof(cohort, receipt.clone())?;
        self.resume_terminal_reconciliation(cohort)
    }

    #[cfg(test)]
    fn apply_completion_receipt(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
    ) -> Result<(), DecoderPoolError> {
        self.install_completion_proof(cohort, EngineCompletionOutcome::Completed(receipt.clone()))?;
        self.resume_terminal_reconciliation(cohort)
    }

    #[cfg(test)]
    fn apply_abort_release_receipt(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        receipt: &EngineReleaseReceipt,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        validate_pinned_disposition(cohort, &self.inner, disposition, CohortPhase::Aborting)?;
        self.install_abort_proof(cohort, EngineAbortOutcome::Aborted(receipt.clone()))?;
        self.resume_terminal_reconciliation(cohort)
    }

    #[cfg(test)]
    fn apply_quarantine_receipt(
        &self,
        cohort: &mut DecoderAssignmentCohort,
        receipt: &EngineQuarantineReceipt,
    ) -> Result<(), DecoderPoolError> {
        self.install_quarantine_proof(cohort, receipt.clone())?;
        self.resume_terminal_reconciliation(cohort)
    }

    /// Apply a previously persisted authoritative terminal proof.
    ///
    /// This is idempotent for a cohort already released or quarantined. It
    /// performs no engine I/O and takes no caller-supplied retry policy.
    pub fn resume_terminal_reconciliation(
        &self,
        cohort: &mut DecoderAssignmentCohort,
    ) -> Result<(), DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let mut state = self.inner.state.lock();
        if cohort.phase == CohortPhase::Terminal {
            if state.assignments.contains_key(&cohort.assignment_id) {
                return Err(DecoderPoolError::InconsistentAssignment {
                    assignment_id: cohort.assignment_id,
                    reason: "terminal cohort remains in the assignment ledger",
                });
            }
            return Ok(());
        }
        let application = {
            let record = state
                .assignments
                .get(&cohort.assignment_id)
                .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
            validate_assignment_record(record, cohort)?;
            validate_terminal_reconciliation_phase(record, cohort.assignment_id)?;
            if cohort.phase == CohortPhase::Quarantined {
                return validate_completed_quarantine(record, cohort);
            }
            match record.terminal_reconciliation.as_ref() {
                Some(TerminalReconciliationRecord::Cancellation {
                    disposition,
                    proof: Some(receipt),
                }) => {
                    validate_engine_release_receipt(
                        cohort,
                        receipt,
                        EngineReleaseKind::PreparedCancelled,
                    )?;
                    TerminalApplication::Release {
                        expected_phase: CohortPhase::Cancelling,
                        disposition: *disposition,
                    }
                }
                Some(TerminalReconciliationRecord::Completion {
                    proof: Some(EngineCompletionOutcome::Completed(receipt)),
                }) => {
                    validate_engine_release_receipt(cohort, receipt, EngineReleaseKind::Completed)?;
                    TerminalApplication::Release {
                        expected_phase: CohortPhase::Completing,
                        disposition: RetryDisposition::Terminal,
                    }
                }
                Some(TerminalReconciliationRecord::Completion {
                    proof: Some(EngineCompletionOutcome::Quarantined(receipt)),
                }) => {
                    validate_engine_quarantine_receipt(cohort, receipt)?;
                    TerminalApplication::Quarantine {
                        expected_phase: CohortPhase::Completing,
                    }
                }
                Some(TerminalReconciliationRecord::Abort {
                    disposition,
                    proof: Some(EngineAbortOutcome::Aborted(receipt)),
                }) => {
                    validate_engine_release_receipt(cohort, receipt, EngineReleaseKind::Aborted)?;
                    TerminalApplication::Release {
                        expected_phase: CohortPhase::Aborting,
                        disposition: *disposition,
                    }
                }
                Some(TerminalReconciliationRecord::Abort {
                    proof: Some(EngineAbortOutcome::Quarantined(receipt)),
                    ..
                }) => {
                    validate_engine_quarantine_receipt(cohort, receipt)?;
                    TerminalApplication::Quarantine {
                        expected_phase: CohortPhase::Aborting,
                    }
                }
                Some(TerminalReconciliationRecord::Quarantine {
                    proof: Some(receipt),
                }) => {
                    validate_engine_quarantine_receipt(cohort, receipt)?;
                    TerminalApplication::Quarantine {
                        expected_phase: CohortPhase::Quarantining,
                    }
                }
                Some(
                    TerminalReconciliationRecord::Cancellation { proof: None, .. }
                    | TerminalReconciliationRecord::Completion { proof: None }
                    | TerminalReconciliationRecord::Abort { proof: None, .. }
                    | TerminalReconciliationRecord::Quarantine { proof: None },
                ) => {
                    return Err(DecoderPoolError::TerminalProofPending(cohort.assignment_id));
                }
                None => {
                    return Err(invalid_transition(
                        cohort.assignment_id,
                        record.phase,
                        "resume terminal reconciliation",
                    ));
                }
            }
        };
        let (expected_phase, requested_transition) = match application {
            TerminalApplication::Release { expected_phase, .. } => (expected_phase, "terminal"),
            TerminalApplication::Quarantine { expected_phase } => {
                (expected_phase, phase_name(CohortPhase::Quarantined))
            }
        };
        let ledger =
            preflight_live_assignment(&state, cohort, expected_phase, requested_transition)?;
        match application {
            TerminalApplication::Release { disposition, .. } => {
                release_assignment(&mut state, cohort, ledger, expected_phase, disposition)
            }
            TerminalApplication::Quarantine { expected_phase } => {
                quarantine_assignment(&mut state, cohort, ledger, expected_phase)
            }
        }
    }

    /// Return an authoritative quarantine proof retained for this cohort.
    pub fn quarantine_receipt(
        &self,
        cohort: &DecoderAssignmentCohort,
    ) -> Result<Option<EngineQuarantineReceipt>, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        let state = self.inner.state.lock();
        let record = state
            .assignments
            .get(&cohort.assignment_id)
            .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
        validate_assignment_record(record, cohort)?;
        let receipt = match record.terminal_reconciliation.as_ref() {
            Some(TerminalReconciliationRecord::Completion {
                proof: Some(EngineCompletionOutcome::Quarantined(receipt)),
            })
            | Some(TerminalReconciliationRecord::Abort {
                proof: Some(EngineAbortOutcome::Quarantined(receipt)),
                ..
            })
            | Some(TerminalReconciliationRecord::Quarantine {
                proof: Some(receipt),
            }) => Some(receipt.clone()),
            _ => None,
        };
        Ok(receipt)
    }

    /// Report whether terminal reconciliation retained this exact cohort.
    pub(crate) fn cohort_remains_quarantined(
        &self,
        cohort: &DecoderAssignmentCohort,
    ) -> Result<bool, DecoderPoolError> {
        self.validate_assignment_pool(cohort)?;
        Ok(cohort.phase == CohortPhase::Quarantined)
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
                pending_admissions: replica.pending_admissions,
                pending_child_requests: replica.pending_child_requests,
                pending_reserved_kv_tokens: replica.pending_reserved_kv_tokens,
                pending_remaining_decode_tokens: replica.pending_remaining_decode_tokens,
                active_cohorts: replica.active_cohorts,
                active_child_requests: replica.active_child_requests,
                quiescing_cohorts: replica.quiescing_cohorts,
                quarantined_cohorts: replica.quarantined_cohorts,
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

    fn validate_pending_pool(&self, pending: &PendingAdmission) -> Result<(), DecoderPoolError> {
        if !Arc::ptr_eq(&pending.inner, &self.inner) {
            return Err(DecoderPoolError::ForeignPendingAdmission);
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

#[cfg(test)]
pub(crate) fn begin_test_pending_for_grant(
    pool: &DecoderPool,
    owner: &LogicalRequestOwner,
    grant: &BoundPreparedGrant,
) -> Result<PendingAdmission, DecoderPoolError> {
    pool.validate_request_owner(owner)?;
    let binding = grant.binding();
    let charge = PendingSchedulingCharge::new(
        binding.accounting().child_count(),
        binding.accounting().total_reserved_kv_tokens(),
        binding.accounting().total_remaining_decode_tokens(),
    )?;
    let mut state = pool.inner.state.lock();
    let (retry_constraint, failed_decoders) = {
        let chain = state
            .request_chains
            .get(&owner.chain_id)
            .ok_or(DecoderPoolError::UnknownRequestChain(owner.chain_id))?;
        let RequestChainState::IdleOpen(retry_constraint) = &chain.state else {
            return Err(DecoderPoolError::InvalidGrant(
                "test pending admission requires an idle request chain".to_string(),
            ));
        };
        (retry_constraint.clone(), chain.failed_decoders.clone())
    };
    let eligible_decoders: HashSet<DecoderId> = state.replicas.keys().cloned().collect();
    let selected = match &retry_constraint {
        AdmissionRetryConstraint::AnyEligible => select_decoders(
            &state.replicas,
            &failed_decoders,
            &eligible_decoders,
            state.last_scheduled_decoder.as_ref(),
        )?
        .remove(0),
        AdmissionRetryConstraint::SameDecoder(decoder_id) => decoder_id.clone(),
    };
    if binding.decoder_id() != &selected {
        return Err(DecoderPoolError::GrantDecoderIneligible(
            binding.decoder_id().clone(),
        ));
    }
    let claim_id = Uuid::new_v4();
    install_pending_attempt(
        &mut state,
        owner.chain_id,
        selected.clone(),
        binding.reservation_attempt_id(),
        binding.reserve_attempt_digest(),
        charge,
        retry_constraint.clone(),
        None,
        claim_id,
        PendingAdmissionAuthority::PreparedIssued,
    )?;
    Ok(PendingAdmission {
        inner: Arc::clone(&pool.inner),
        chain_id: owner.chain_id,
        claim_id,
        decoder_id: selected,
        reservation_attempt_id: binding.reservation_attempt_id(),
        reserve_attempt_digest: binding.reserve_attempt_digest(),
        charge,
        retry_constraint,
        resolved: false,
    })
}

fn release_assignment(
    state: &mut PoolState,
    cohort: &mut DecoderAssignmentCohort,
    ledger: LiveAssignmentLedger,
    expected_phase: CohortPhase,
    disposition: RetryDisposition,
) -> Result<(), DecoderPoolError> {
    let _record = state
        .assignments
        .remove(&cohort.assignment_id)
        .expect("assignment was preflighted under the same pool lock");
    let replica = state
        .replicas
        .get_mut(&ledger.decoder_id)
        .expect("assigned decoder was preflighted under the same pool lock");
    replica.active_cohorts -= 1;
    replica.active_child_requests -= ledger.child_count;
    replica.reserved_kv_tokens -= ledger.kv_tokens;
    replica.remaining_decode_tokens -= ledger.remaining_decode_tokens;
    if phase_is_quiescing(expected_phase) {
        replica.quiescing_cohorts -= 1;
    }
    for room in ledger.rooms {
        let removed = state.room_owners.remove(&(ledger.decoder_id.clone(), room));
        debug_assert_eq!(removed, Some(cohort.assignment_id));
    }
    for allocation in ledger.allocations {
        let removed = state.allocation_owners.remove(&allocation);
        debug_assert_eq!(removed, Some(cohort.assignment_id));
    }

    let remove_request = {
        let chain = state
            .request_chains
            .get_mut(&ledger.chain_id)
            .expect("request chain was preflighted under the same pool lock");
        match disposition {
            RetryDisposition::Terminal => {
                chain.state = RequestChainState::Terminal;
                chain.failed_decoders.clear();
            }
            RetryDisposition::Retryable => {
                chain.state = RequestChainState::IdleOpen(AdmissionRetryConstraint::AnyEligible);
            }
            RetryDisposition::DecoderFailed => {
                chain.failed_decoders.insert(ledger.decoder_id);
                chain.state = RequestChainState::IdleOpen(AdmissionRetryConstraint::AnyEligible);
            }
        }
        !chain.owner_alive
    };
    if remove_request {
        remove_request_chain(state, ledger.chain_id);
    }
    cohort.phase = CohortPhase::Terminal;
    Ok(())
}

fn quarantine_assignment(
    state: &mut PoolState,
    cohort: &mut DecoderAssignmentCohort,
    ledger: LiveAssignmentLedger,
    expected_phase: CohortPhase,
) -> Result<(), DecoderPoolError> {
    let replica = state
        .replicas
        .get_mut(&ledger.decoder_id)
        .expect("assigned decoder was preflighted under the same pool lock");
    if phase_is_quiescing(expected_phase) {
        replica.quiescing_cohorts -= 1;
    }
    replica.quarantined_cohorts += 1;
    state
        .assignments
        .get_mut(&cohort.assignment_id)
        .expect("assignment was preflighted under the same pool lock")
        .phase = CohortPhase::Quarantined;
    let chain = state
        .request_chains
        .get_mut(&ledger.chain_id)
        .expect("request chain was preflighted under the same pool lock");
    chain.state = RequestChainState::Quarantined(cohort.assignment_id);
    chain.failed_decoders.clear();
    cohort.phase = CohortPhase::Quarantined;
    Ok(())
}

fn select_decoders(
    replicas: &HashMap<DecoderId, ReplicaState>,
    failed_decoders: &HashSet<DecoderId>,
    eligible_decoders: &HashSet<DecoderId>,
    last_scheduled_decoder: Option<&DecoderId>,
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
        .filter(|replica| {
            replica.availability == DecoderAvailability::Ready
                && eligible_decoders.contains(&replica.metadata.id)
        })
        .collect();
    if ready.is_empty() {
        return Err(DecoderPoolError::NoReadyDecoder);
    }
    ready.sort_by(|left, right| {
        compare_current_load(left, right).then_with(|| {
            compare_round_robin(
                &left.metadata.id,
                &right.metadata.id,
                last_scheduled_decoder,
            )
        })
    });
    Ok(ready
        .into_iter()
        .map(|replica| replica.metadata.id.clone())
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn install_pending_attempt(
    state: &mut PoolState,
    chain_id: Uuid,
    decoder_id: DecoderId,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    charge: PendingSchedulingCharge,
    retry_constraint: AdmissionRetryConstraint,
    reservation: Option<Arc<DecoderGrantReservation>>,
    claim_id: Uuid,
    authority: PendingAdmissionAuthority,
) -> Result<(), DecoderPoolError> {
    let advances_round_robin = matches!(retry_constraint, AdmissionRetryConstraint::AnyEligible);
    let replica = state
        .replicas
        .get(&decoder_id)
        .expect("selected decoder disappeared while pool lock was held");
    let pending_admissions = replica.pending_admissions.checked_add(1).ok_or_else(|| {
        DecoderPoolError::InvalidGrant("pending reservation accounting overflows usize".to_string())
    })?;
    let pending_child_requests = replica
        .pending_child_requests
        .checked_add(charge.child_requests())
        .ok_or_else(|| {
            DecoderPoolError::InvalidGrant(
                "pending child-request accounting overflows usize".to_string(),
            )
        })?;
    let pending_reserved_kv_tokens = replica
        .pending_reserved_kv_tokens
        .checked_add(charge.reserved_kv_tokens())
        .ok_or_else(|| {
            DecoderPoolError::InvalidGrant(
                "pending KV-token accounting overflows usize".to_string(),
            )
        })?;
    let pending_remaining_decode_tokens = replica
        .pending_remaining_decode_tokens
        .checked_add(charge.remaining_decode_tokens())
        .ok_or_else(|| {
            DecoderPoolError::InvalidGrant(
                "pending decode-token accounting overflows usize".to_string(),
            )
        })?;

    let replica = state
        .replicas
        .get_mut(&decoder_id)
        .expect("selected decoder disappeared while pool lock was held");
    replica.pending_admissions = pending_admissions;
    replica.pending_child_requests = pending_child_requests;
    replica.pending_reserved_kv_tokens = pending_reserved_kv_tokens;
    replica.pending_remaining_decode_tokens = pending_remaining_decode_tokens;
    state
        .request_chains
        .get_mut(&chain_id)
        .expect("request chain disappeared while pool lock was held")
        .state = RequestChainState::Reserving(Box::new(PendingAdmissionRecord {
        decoder_id: decoder_id.clone(),
        reservation_attempt_id,
        reserve_attempt_digest,
        charge,
        retry_constraint,
        reservation,
        claim_id: Some(claim_id),
        authority,
        reconciliation: None,
    }));
    if advances_round_robin {
        state.last_scheduled_decoder = Some(decoder_id);
    }
    Ok(())
}

fn rollback_pending_record(state: &mut PoolState, pending: &PendingAdmission) {
    let (decoder_id, charge, retry_constraint, owner_alive) = {
        let chain = state
            .request_chains
            .get(&pending.chain_id)
            .expect("pending lease chain must exist until its authority is resolved");
        let record = match &chain.state {
            RequestChainState::Reserving(record) => record,
            _ => panic!("pending lease must still own a reserving request chain"),
        };
        validate_pending_record(record, pending)
            .expect("pending lease must exactly match its pool-owned record");
        (
            record.decoder_id.clone(),
            record.charge,
            record.retry_constraint.clone(),
            chain.owner_alive,
        )
    };

    let replica = state
        .replicas
        .get_mut(&decoder_id)
        .expect("pending decoder must exist until its charge is released");
    replica.pending_admissions = replica
        .pending_admissions
        .checked_sub(1)
        .expect("pending admission accounting must include the checked-out attempt");
    replica.pending_child_requests = replica
        .pending_child_requests
        .checked_sub(charge.child_requests())
        .expect("pending child accounting must include the checked-out attempt");
    replica.pending_reserved_kv_tokens = replica
        .pending_reserved_kv_tokens
        .checked_sub(charge.reserved_kv_tokens())
        .expect("pending KV accounting must include the checked-out attempt");
    replica.pending_remaining_decode_tokens = replica
        .pending_remaining_decode_tokens
        .checked_sub(charge.remaining_decode_tokens())
        .expect("pending decode accounting must include the checked-out attempt");

    if owner_alive {
        state
            .request_chains
            .get_mut(&pending.chain_id)
            .expect("pending request chain disappeared while the pool mutex was held")
            .state = RequestChainState::IdleOpen(retry_constraint);
        return;
    }
    remove_request_chain(state, pending.chain_id);
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
        || record.binding.grant_id() != cohort.assignment_id
        || record.binding.digest() != cohort.binding.digest()
    {
        return Err(DecoderPoolError::ForeignAssignment);
    }
    Ok(())
}

fn validate_completed_quarantine(
    record: &AssignmentRecord,
    cohort: &DecoderAssignmentCohort,
) -> Result<(), DecoderPoolError> {
    if cohort.phase != CohortPhase::Quarantined || record.phase != CohortPhase::Quarantined {
        return Err(invalid_transition(
            cohort.assignment_id,
            record.phase,
            "resume terminal reconciliation",
        ));
    }
    let proof_receipt = match record.terminal_reconciliation.as_ref() {
        Some(TerminalReconciliationRecord::Completion {
            proof: Some(EngineCompletionOutcome::Quarantined(receipt)),
        })
        | Some(TerminalReconciliationRecord::Abort {
            proof: Some(EngineAbortOutcome::Quarantined(receipt)),
            ..
        })
        | Some(TerminalReconciliationRecord::Quarantine {
            proof: Some(receipt),
        }) => receipt,
        _ => {
            return Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "quarantined assignment has no matching terminal proof",
            });
        }
    };
    validate_engine_quarantine_receipt(cohort, proof_receipt)
}

fn validate_terminal_reconciliation_phase(
    record: &AssignmentRecord,
    assignment_id: Uuid,
) -> Result<(), DecoderPoolError> {
    let matches_phase = matches!(
        (&record.terminal_reconciliation, record.phase),
        (None, CohortPhase::Reserved | CohortPhase::Active)
            | (
                Some(TerminalReconciliationRecord::Cancellation { .. }),
                CohortPhase::Cancelling
            )
            | (
                Some(TerminalReconciliationRecord::Completion { .. }),
                CohortPhase::Completing
            )
            | (
                Some(TerminalReconciliationRecord::Abort { .. }),
                CohortPhase::Aborting
            )
            | (
                Some(TerminalReconciliationRecord::Quarantine { .. }),
                CohortPhase::Quarantining
            )
            | (
                Some(TerminalReconciliationRecord::Abort {
                    proof: Some(EngineAbortOutcome::Quarantined(_)),
                    ..
                }),
                CohortPhase::Quarantined,
            )
            | (
                Some(TerminalReconciliationRecord::Completion {
                    proof: Some(EngineCompletionOutcome::Quarantined(_)),
                }),
                CohortPhase::Quarantined,
            )
            | (
                Some(TerminalReconciliationRecord::Quarantine { proof: Some(_) }),
                CohortPhase::Quarantined,
            )
    );
    if !matches_phase {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id,
            reason: "assignment phase differs from its terminal reconciliation ledger",
        });
    }
    Ok(())
}

fn install_proof_once<T: Eq>(
    slot: &mut Option<T>,
    proof: T,
    assignment_id: Uuid,
) -> Result<(), DecoderPoolError> {
    if let Some(existing) = slot.as_ref() {
        if existing == &proof {
            return Ok(());
        }
        return Err(DecoderPoolError::ConflictingTerminalProof(assignment_id));
    }
    *slot = Some(proof);
    Ok(())
}

#[cfg(test)]
fn validate_pinned_disposition(
    cohort: &DecoderAssignmentCohort,
    inner: &DecoderPoolInner,
    disposition: RetryDisposition,
    expected_phase: CohortPhase,
) -> Result<(), DecoderPoolError> {
    if cohort.pool_id != inner.pool_id {
        return Err(DecoderPoolError::ForeignAssignment);
    }
    let state = inner.state.lock();
    let record = state
        .assignments
        .get(&cohort.assignment_id)
        .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
    validate_assignment_record(record, cohort)?;
    if cohort.phase != expected_phase || record.phase != expected_phase {
        return Err(invalid_transition(
            cohort.assignment_id,
            record.phase,
            "validate pinned retry disposition",
        ));
    }
    let pinned = match record.terminal_reconciliation.as_ref() {
        Some(TerminalReconciliationRecord::Cancellation { disposition, .. })
            if expected_phase == CohortPhase::Cancelling =>
        {
            *disposition
        }
        Some(TerminalReconciliationRecord::Abort { disposition, .. })
            if expected_phase == CohortPhase::Aborting =>
        {
            *disposition
        }
        _ => {
            return Err(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "assignment has no matching pinned retry disposition",
            });
        }
    };
    if pinned != disposition {
        return Err(DecoderPoolError::ConflictingTerminalProof(
            cohort.assignment_id,
        ));
    }
    Ok(())
}

fn validate_engine_grant_identity(
    grant_id: Uuid,
    grant_digest: DecoderGrantDigest,
    cohort: &DecoderAssignmentCohort,
    operation: &'static str,
) -> Result<(), DecoderPoolError> {
    if grant_id != cohort.assignment_id || grant_digest != cohort.binding.digest() {
        return Err(DecoderPoolError::InvalidGrant(format!(
            "{operation} grant does not exactly match the assignment"
        )));
    }
    Ok(())
}

fn issue_pending_claim(
    inner: Arc<DecoderPoolInner>,
    chain_id: Uuid,
    record: &mut PendingAdmissionRecord,
) -> Result<PendingAdmission, DecoderPoolError> {
    if record.claim_id.is_some() {
        return Err(DecoderPoolError::PendingAdmissionAlreadyClaimed(
            record.reservation_attempt_id,
        ));
    }
    if matches!(
        record.authority,
        PendingAdmissionAuthority::ReserveCheckedOut { .. }
            | PendingAdmissionAuthority::CancellationCheckedOut { .. }
    ) {
        return Err(DecoderPoolError::PendingReservationAlreadyStarted(
            record.reservation_attempt_id,
        ));
    }
    let claim_id = Uuid::new_v4();
    record.claim_id = Some(claim_id);
    Ok(PendingAdmission {
        inner,
        chain_id,
        claim_id,
        decoder_id: record.decoder_id.clone(),
        reservation_attempt_id: record.reservation_attempt_id,
        reserve_attempt_digest: record.reserve_attempt_digest,
        charge: record.charge,
        retry_constraint: record.retry_constraint.clone(),
        resolved: false,
    })
}

fn pending_record<'a>(
    state: &'a PoolState,
    pending: &PendingAdmission,
) -> Result<&'a PendingAdmissionRecord, DecoderPoolError> {
    let chain = state
        .request_chains
        .get(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    let record = match &chain.state {
        RequestChainState::Reserving(record) => record,
        _ => {
            return Err(DecoderPoolError::UnknownPendingAdmission(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_record(record, pending)?;
    Ok(record)
}

fn pending_is_resolved(state: &PoolState, pending: &PendingAdmission) -> bool {
    state
        .request_chains
        .get(&pending.chain_id)
        .is_some_and(|chain| {
            chain
                .resolved_admissions
                .contains_key(&pending.reservation_attempt_id)
        })
}

fn pending_record_mut<'a>(
    state: &'a mut PoolState,
    pending: &PendingAdmission,
) -> Result<&'a mut PendingAdmissionRecord, DecoderPoolError> {
    let chain = state
        .request_chains
        .get_mut(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    let record = match &mut chain.state {
        RequestChainState::Reserving(record) => record,
        _ => {
            return Err(DecoderPoolError::UnknownPendingAdmission(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_record(record, pending)?;
    Ok(record)
}

fn ensure_prepared_pending_authority(
    state: &PoolState,
    pending: &PendingAdmission,
) -> Result<(), DecoderPoolError> {
    let record = pending_record(state, pending)?;
    if matches!(record.authority, PendingAdmissionAuthority::PreparedIssued) {
        return Ok(());
    }
    Err(DecoderPoolError::PendingReservationAlreadyStarted(
        pending.reservation_attempt_id,
    ))
}

fn validate_pending_record(
    record: &PendingAdmissionRecord,
    pending: &PendingAdmission,
) -> Result<(), DecoderPoolError> {
    if record.decoder_id != pending.decoder_id
        || record.reservation_attempt_id != pending.reservation_attempt_id
        || record.reserve_attempt_digest != pending.reserve_attempt_digest
        || record.charge != pending.charge
        || record.retry_constraint != pending.retry_constraint
        || record.claim_id != Some(pending.claim_id)
    {
        return Err(DecoderPoolError::ForeignPendingAdmission);
    }
    Ok(())
}

fn validate_reserve_refusal_receipt(
    state: &PoolState,
    pending: &PendingAdmission,
    receipt: &DecoderReserveRefusalReceipt,
) -> Result<(), DecoderPoolError> {
    if receipt.prefill_id() != &state.prefill_id
        || receipt.logical_request_chain_id() != pending.chain_id
        || receipt.decoder_id() != &pending.decoder_id
        || receipt.reservation_attempt_id() != pending.reservation_attempt_id
        || receipt.reserve_attempt_digest() != pending.reserve_attempt_digest
        || !receipt.take_once()
    {
        return Err(DecoderPoolError::ForeignPendingAdmission);
    }
    Ok(())
}

fn validate_pending_cancellation_target(
    pending: &PendingAdmission,
    target: &PreparedGrantCancellationTarget,
) -> Result<(), DecoderPoolError> {
    if target.decoder_id() != &pending.decoder_id
        || target.reservation_attempt_id() != pending.reservation_attempt_id
        || target.reserve_attempt_digest() != pending.reserve_attempt_digest
    {
        return Err(DecoderPoolError::ForeignPendingAdmission);
    }
    Ok(())
}

fn validate_pending_cancellation_proof(
    reservation_attempt_id: Uuid,
    target: &PreparedGrantCancellationTarget,
    proof: &PendingCancellationProof,
) -> Result<(), DecoderPoolError> {
    let valid = match proof {
        PendingCancellationProof::Unbound(receipt) => target.matches_unbound_receipt(receipt),
        PendingCancellationProof::Bound(receipt) => target.matches_bound_receipt(receipt),
    };
    if !valid {
        return Err(DecoderPoolError::InvalidPendingCancellationProof {
            reservation_attempt_id,
            reason: "receipt does not match the pinned cancellation target",
        });
    }
    Ok(())
}

fn install_pending_reconciliation(
    state: &mut PoolState,
    pending: &PendingAdmission,
    proof: PendingAdmissionReconciliationRecord,
) -> Result<(), DecoderPoolError> {
    let chain = state
        .request_chains
        .get_mut(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    if let Some(existing) = chain
        .resolved_admissions
        .get(&pending.reservation_attempt_id)
    {
        if existing == &proof {
            return Ok(());
        }
        return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
            pending.reservation_attempt_id,
        ));
    }
    let record = match &mut chain.state {
        RequestChainState::Reserving(record) => record,
        _ => {
            return Err(DecoderPoolError::UnknownPendingAdmission(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_record(record, pending)?;
    if let Some(existing) = record.reconciliation.as_ref() {
        if existing == &proof {
            return Ok(());
        }
        return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
            pending.reservation_attempt_id,
        ));
    }
    record.reconciliation = Some(proof);
    Ok(())
}

fn install_pending_cancellation_intent(
    state: &mut PoolState,
    pending: &PendingAdmission,
    target: &PreparedGrantCancellationTarget,
    disposition: PendingAdmissionDisposition,
) -> Result<(), DecoderPoolError> {
    let chain = state
        .request_chains
        .get_mut(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    if let Some(existing) = chain
        .resolved_admissions
        .get(&pending.reservation_attempt_id)
    {
        if matches!(
            existing,
            PendingAdmissionReconciliationRecord::Cancellation {
                disposition: existing_disposition,
                target: existing_target,
                ..
            } if *existing_disposition == disposition && existing_target == target
        ) {
            return Ok(());
        }
        return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
            pending.reservation_attempt_id,
        ));
    }
    let record = match &mut chain.state {
        RequestChainState::Reserving(record) => record,
        _ => {
            return Err(DecoderPoolError::UnknownPendingAdmission(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_record(record, pending)?;
    match record.reconciliation.as_ref() {
        None => {
            record.reconciliation = Some(PendingAdmissionReconciliationRecord::Cancellation {
                disposition,
                target: target.clone(),
                proof: None,
            });
            Ok(())
        }
        Some(PendingAdmissionReconciliationRecord::Cancellation {
            disposition: existing_disposition,
            target: existing_target,
            ..
        }) if *existing_disposition == disposition && existing_target == target => Ok(()),
        Some(_) => Err(DecoderPoolError::ConflictingPendingAdmissionProof(
            pending.reservation_attempt_id,
        )),
    }
}

fn install_pending_cancellation_proof(
    state: &mut PoolState,
    pending: &PendingAdmission,
    proof: &PendingCancellationProof,
) -> Result<(), DecoderPoolError> {
    let chain = state
        .request_chains
        .get_mut(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    if let Some(existing) = chain
        .resolved_admissions
        .get(&pending.reservation_attempt_id)
    {
        return match existing {
            PendingAdmissionReconciliationRecord::Cancellation {
                proof: Some(existing_proof),
                ..
            } if existing_proof.as_ref() == proof => Ok(()),
            _ => Err(DecoderPoolError::ConflictingPendingAdmissionProof(
                pending.reservation_attempt_id,
            )),
        };
    }
    let record = match &mut chain.state {
        RequestChainState::Reserving(record) => record,
        _ => {
            return Err(DecoderPoolError::UnknownPendingAdmission(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_record(record, pending)?;
    let (target, proof_slot) = match record.reconciliation.as_mut() {
        Some(PendingAdmissionReconciliationRecord::Cancellation { target, proof, .. }) => {
            (target, proof)
        }
        Some(PendingAdmissionReconciliationRecord::Refusal(_)) => {
            return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
                pending.reservation_attempt_id,
            ));
        }
        None => {
            return Err(DecoderPoolError::PendingCancellationNotPinned(
                pending.reservation_attempt_id,
            ));
        }
    };
    validate_pending_cancellation_proof(pending.reservation_attempt_id, target, proof)?;
    if let Some(existing) = proof_slot.as_ref() {
        if existing.as_ref() == proof {
            return Ok(());
        }
        return Err(DecoderPoolError::ConflictingPendingAdmissionProof(
            pending.reservation_attempt_id,
        ));
    }
    *proof_slot = Some(Box::new(proof.clone()));
    Ok(())
}

fn apply_pending_reconciliation(
    state: &mut PoolState,
    pending: &PendingAdmission,
) -> Result<PendingAdmissionDisposition, DecoderPoolError> {
    let chain = state
        .request_chains
        .get(&pending.chain_id)
        .ok_or(DecoderPoolError::UnknownRequestChain(pending.chain_id))?;
    if let Some(resolved) = chain
        .resolved_admissions
        .get(&pending.reservation_attempt_id)
    {
        return Ok(pending_reconciliation_disposition(resolved));
    }
    let (decoder_id, reservation_attempt_id, charge, reconciliation, owner_alive) = {
        let record = match &chain.state {
            RequestChainState::Reserving(record) => record,
            _ => {
                return Err(DecoderPoolError::UnknownPendingAdmission(
                    pending.reservation_attempt_id,
                ));
            }
        };
        validate_pending_record(record, pending)?;
        (
            record.decoder_id.clone(),
            record.reservation_attempt_id,
            record.charge,
            record.reconciliation.clone(),
            chain.owner_alive,
        )
    };
    let reconciliation = reconciliation.ok_or(DecoderPoolError::PendingAdmissionProofPending(
        pending.reservation_attempt_id,
    ))?;
    if matches!(
        &reconciliation,
        PendingAdmissionReconciliationRecord::Cancellation { proof: None, .. }
    ) {
        return Err(DecoderPoolError::PendingAdmissionProofPending(
            pending.reservation_attempt_id,
        ));
    }
    validate_pending_decoder_ledger(state, &decoder_id, pending.reservation_attempt_id)?;
    let disposition = pending_reconciliation_disposition(&reconciliation);

    let replica = state
        .replicas
        .get_mut(&decoder_id)
        .expect("pending decoder was validated under the same pool lock");
    replica.pending_admissions -= 1;
    replica.pending_child_requests -= charge.child_requests();
    replica.pending_reserved_kv_tokens -= charge.reserved_kv_tokens();
    replica.pending_remaining_decode_tokens -= charge.remaining_decode_tokens();

    let chain = state
        .request_chains
        .get_mut(&pending.chain_id)
        .expect("request chain was validated under the same pool lock");
    let previous = chain
        .resolved_admissions
        .insert(reservation_attempt_id, reconciliation);
    debug_assert!(previous.is_none());
    match disposition {
        PendingAdmissionDisposition::RetrySameDecoder => {
            chain.state =
                RequestChainState::IdleOpen(AdmissionRetryConstraint::SameDecoder(decoder_id));
        }
        PendingAdmissionDisposition::RetryAnotherDecoder => {
            chain.failed_decoders.insert(decoder_id);
            chain.state = RequestChainState::IdleOpen(AdmissionRetryConstraint::AnyEligible);
        }
        PendingAdmissionDisposition::Terminal => {
            chain.failed_decoders.clear();
            chain.state = RequestChainState::Terminal;
        }
    }
    if !owner_alive {
        remove_request_chain(state, pending.chain_id);
    }
    Ok(disposition)
}

fn pending_reconciliation_disposition(
    reconciliation: &PendingAdmissionReconciliationRecord,
) -> PendingAdmissionDisposition {
    match reconciliation {
        PendingAdmissionReconciliationRecord::Refusal(receipt) => match receipt.disposition() {
            DecoderReserveRefusalDisposition::RetrySameDecoder => {
                PendingAdmissionDisposition::RetrySameDecoder
            }
            DecoderReserveRefusalDisposition::RetryAnotherDecoder => {
                PendingAdmissionDisposition::RetryAnotherDecoder
            }
            DecoderReserveRefusalDisposition::Terminal => PendingAdmissionDisposition::Terminal,
        },
        PendingAdmissionReconciliationRecord::Cancellation { disposition, .. } => *disposition,
    }
}

fn validate_pending_decoder_ledger(
    state: &PoolState,
    decoder_id: &DecoderId,
    reservation_attempt_id: Uuid,
) -> Result<(), DecoderPoolError> {
    let replica =
        state
            .replicas
            .get(decoder_id)
            .ok_or(DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id,
                reason: "pending decoder generation is missing",
            })?;
    let mut pending_admissions = 0usize;
    let mut pending_child_requests = 0usize;
    let mut pending_reserved_kv_tokens = 0usize;
    let mut pending_remaining_decode_tokens = 0usize;
    for record in state.request_chains.values().filter_map(|chain| {
        let RequestChainState::Reserving(record) = &chain.state else {
            return None;
        };
        (record.decoder_id == *decoder_id).then_some(record)
    }) {
        pending_admissions = pending_admissions.checked_add(1).ok_or(
            DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id,
                reason: "pending reservation accounting overflows usize",
            },
        )?;
        pending_child_requests = pending_child_requests
            .checked_add(record.charge.child_requests())
            .ok_or(DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id,
                reason: "pending child-request accounting overflows usize",
            })?;
        pending_reserved_kv_tokens = pending_reserved_kv_tokens
            .checked_add(record.charge.reserved_kv_tokens())
            .ok_or(DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id,
                reason: "pending KV-token accounting overflows usize",
            })?;
        pending_remaining_decode_tokens = pending_remaining_decode_tokens
            .checked_add(record.charge.remaining_decode_tokens())
            .ok_or(DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id,
                reason: "pending decode-token accounting overflows usize",
            })?;
    }
    if replica.pending_admissions != pending_admissions
        || replica.pending_child_requests != pending_child_requests
        || replica.pending_reserved_kv_tokens != pending_reserved_kv_tokens
        || replica.pending_remaining_decode_tokens != pending_remaining_decode_tokens
    {
        return Err(DecoderPoolError::InconsistentPendingAdmission {
            reservation_attempt_id,
            reason: "decoder pending accounting differs from request-chain reservations",
        });
    }
    Ok(())
}

fn validate_decoder_ledger(
    state: &PoolState,
    decoder_id: &DecoderId,
    assignment_id: Uuid,
) -> Result<(), DecoderPoolError> {
    let replica =
        state
            .replicas
            .get(decoder_id)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "assigned decoder is missing",
            })?;
    let mut active_cohorts = 0usize;
    let mut active_child_requests = 0usize;
    let mut quiescing_cohorts = 0usize;
    let mut quarantined_cohorts = 0usize;
    let mut reserved_kv_tokens = 0usize;
    let mut remaining_decode_tokens = 0usize;
    let mut rooms = HashMap::new();
    let mut allocations = HashMap::new();

    for (record_assignment_id, record) in state
        .assignments
        .iter()
        .filter(|(_, record)| record.binding.decoder_id() == decoder_id)
    {
        if record.phase == CohortPhase::Terminal {
            return Err(DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "assignment ledger retains a terminal record",
            });
        }
        validate_terminal_reconciliation_phase(record, assignment_id)?;

        active_cohorts += 1;
        active_child_requests = active_child_requests
            .checked_add(record.child_count)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "assignment child-request accounting overflows usize",
            })?;
        reserved_kv_tokens = reserved_kv_tokens.checked_add(record.kv_tokens).ok_or(
            DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "assignment KV-token accounting overflows usize",
            },
        )?;
        remaining_decode_tokens = remaining_decode_tokens
            .checked_add(record.remaining_decode_tokens)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "assignment decode-token accounting overflows usize",
            })?;
        if phase_is_quiescing(record.phase) {
            quiescing_cohorts += 1;
        }
        if record.phase == CohortPhase::Quarantined {
            quarantined_cohorts += 1;
        }
        for room in record.binding.bootstrap_rooms() {
            if rooms
                .insert((decoder_id.clone(), *room), *record_assignment_id)
                .is_some()
            {
                return Err(DecoderPoolError::InconsistentAssignment {
                    assignment_id,
                    reason: "assignment ledger repeats a decoder bootstrap room",
                });
            }
        }
        for allocation in record.binding.allocation_keys() {
            if allocations
                .insert(allocation, *record_assignment_id)
                .is_some()
            {
                return Err(DecoderPoolError::InconsistentAssignment {
                    assignment_id,
                    reason: "assignment ledger repeats a decoder slot generation",
                });
            }
        }
    }

    if replica.active_cohorts != active_cohorts
        || replica.active_child_requests != active_child_requests
        || replica.quiescing_cohorts != quiescing_cohorts
        || replica.quarantined_cohorts != quarantined_cohorts
        || replica.reserved_kv_tokens != reserved_kv_tokens
        || replica.remaining_decode_tokens != remaining_decode_tokens
    {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id,
            reason: "decoder accounting differs from the assignment ledger",
        });
    }
    let room_owners: HashMap<(DecoderId, u64), Uuid> = state
        .room_owners
        .iter()
        .filter(|((active_decoder_id, _), _)| active_decoder_id == decoder_id)
        .map(|(room, owner)| (room.clone(), *owner))
        .collect();
    if room_owners != rooms {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id,
            reason: "decoder room ownership differs from the assignment ledger",
        });
    }
    let allocation_owners: HashMap<DecoderAllocationKey, Uuid> = state
        .allocation_owners
        .iter()
        .filter(|(allocation, _)| allocation.decoder_id() == decoder_id)
        .map(|(allocation, owner)| (allocation.clone(), *owner))
        .collect();
    if allocation_owners != allocations {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id,
            reason: "decoder slot-generation ownership differs from the assignment ledger",
        });
    }
    Ok(())
}

fn preflight_live_assignment(
    state: &PoolState,
    cohort: &DecoderAssignmentCohort,
    expected_phase: CohortPhase,
    requested_transition: &'static str,
) -> Result<LiveAssignmentLedger, DecoderPoolError> {
    let record = state
        .assignments
        .get(&cohort.assignment_id)
        .ok_or(DecoderPoolError::UnknownAssignment(cohort.assignment_id))?;
    validate_assignment_record(record, cohort)?;
    if cohort.phase != expected_phase || record.phase != expected_phase {
        return Err(invalid_transition(
            cohort.assignment_id,
            record.phase,
            requested_transition,
        ));
    }
    validate_terminal_reconciliation_phase(record, cohort.assignment_id)?;

    let ledger = LiveAssignmentLedger {
        decoder_id: record.binding.decoder_id().clone(),
        chain_id: record.chain_id,
        child_count: record.child_count,
        kv_tokens: record.kv_tokens,
        remaining_decode_tokens: record.remaining_decode_tokens,
        rooms: record.binding.bootstrap_rooms().to_vec(),
        allocations: record.binding.allocation_keys().collect(),
    };
    if ledger.rooms.iter().any(|room| {
        state.room_owners.get(&(ledger.decoder_id.clone(), *room)) != Some(&cohort.assignment_id)
    }) {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "an assigned bootstrap room is not retained",
        });
    }
    if ledger
        .allocations
        .iter()
        .any(|allocation| state.allocation_owners.get(allocation) != Some(&cohort.assignment_id))
    {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "an assigned slot generation is not retained",
        });
    }
    let replica =
        state
            .replicas
            .get(&ledger.decoder_id)
            .ok_or(DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id,
                reason: "assigned decoder is missing",
            })?;
    if replica.active_cohorts == 0
        || replica.active_child_requests < ledger.child_count
        || replica.reserved_kv_tokens < ledger.kv_tokens
        || replica.remaining_decode_tokens < ledger.remaining_decode_tokens
    {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "decoder accounting is smaller than the assignment ledger",
        });
    }
    if phase_is_quiescing(expected_phase) && replica.quiescing_cohorts == 0 {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "decoder has no quiescing cohort for this assignment",
        });
    }
    let chain = state.request_chains.get(&ledger.chain_id).ok_or(
        DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "request chain is missing",
        },
    )?;
    let chain_assignment = match &chain.state {
        RequestChainState::Assigned(assignment_id)
        | RequestChainState::Quarantined(assignment_id) => Some(*assignment_id),
        RequestChainState::IdleOpen(_)
        | RequestChainState::Reserving(_)
        | RequestChainState::Terminal => None,
    };
    if chain_assignment != Some(cohort.assignment_id) {
        return Err(DecoderPoolError::InconsistentAssignment {
            assignment_id: cohort.assignment_id,
            reason: "request chain does not own this assignment",
        });
    }
    validate_decoder_ledger(state, &ledger.decoder_id, cohort.assignment_id)?;
    Ok(ledger)
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

fn validate_engine_quarantine_receipt(
    cohort: &DecoderAssignmentCohort,
    receipt: &EngineQuarantineReceipt,
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
    } else if !receipt.take_once() {
        Some("receipt does not attest take-once reconciliation")
    } else {
        None
    };
    if let Some(reason) = mismatch {
        return Err(DecoderPoolError::InvalidEngineQuarantineReceipt {
            assignment_id: cohort.assignment_id,
            reason,
        });
    }
    Ok(())
}

fn compare_current_load(left: &ReplicaState, right: &ReplicaState) -> Ordering {
    compare_ratio(
        left.remaining_decode_tokens as u128 + left.pending_remaining_decode_tokens as u128,
        left.metadata.scheduling.service_weight.get(),
        right.remaining_decode_tokens as u128 + right.pending_remaining_decode_tokens as u128,
        right.metadata.scheduling.service_weight.get(),
    )
    .then_with(|| {
        compare_ratio(
            left.reserved_kv_tokens as u128 + left.pending_reserved_kv_tokens as u128,
            left.metadata.scheduling.kv_token_scale.get(),
            right.reserved_kv_tokens as u128 + right.pending_reserved_kv_tokens as u128,
            right.metadata.scheduling.kv_token_scale.get(),
        )
    })
    .then_with(|| {
        compare_ratio(
            left.active_child_requests as u128 + left.pending_child_requests as u128,
            left.metadata.scheduling.child_request_scale.get(),
            right.active_child_requests as u128 + right.pending_child_requests as u128,
            right.metadata.scheduling.child_request_scale.get(),
        )
    })
}

fn compare_round_robin(
    left: &DecoderId,
    right: &DecoderId,
    last_scheduled_decoder: Option<&DecoderId>,
) -> Ordering {
    let Some(last_scheduled_decoder) = last_scheduled_decoder else {
        return left.cmp(right);
    };
    match (
        left > last_scheduled_decoder,
        right > last_scheduled_decoder,
    ) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => left.cmp(right),
    }
}

fn compare_ratio(
    mut left_numerator: u128,
    left_denominator: usize,
    mut right_numerator: u128,
    right_denominator: usize,
) -> Ordering {
    let mut left_denominator = left_denominator as u128;
    let mut right_denominator = right_denominator as u128;
    let mut inverted = false;
    loop {
        let ordering =
            (left_numerator / left_denominator).cmp(&(right_numerator / right_denominator));
        if ordering != Ordering::Equal {
            return if inverted {
                ordering.reverse()
            } else {
                ordering
            };
        }

        let left_remainder = left_numerator % left_denominator;
        let right_remainder = right_numerator % right_denominator;
        let ordering = match (left_remainder == 0, right_remainder == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                left_numerator = left_denominator;
                left_denominator = left_remainder;
                right_numerator = right_denominator;
                right_denominator = right_remainder;
                inverted = !inverted;
                continue;
            }
        };
        return if inverted {
            ordering.reverse()
        } else {
            ordering
        };
    }
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
        CohortPhase::Cancelling => "cancelling",
        CohortPhase::Active => "active",
        CohortPhase::Completing => "completing",
        CohortPhase::Aborting => "aborting",
        CohortPhase::Quarantining => "quarantining",
        CohortPhase::Quarantined => "quarantined",
        CohortPhase::Terminal => "terminal",
    }
}

fn phase_is_quiescing(phase: CohortPhase) -> bool {
    matches!(phase, CohortPhase::Aborting | CohortPhase::Quarantining)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        future,
        sync::{
            atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
            Arc, Mutex as StdMutex,
        },
        thread,
    };

    use axum::{
        extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router,
    };
    use serde_json::{json, Value};
    use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};

    use super::*;
    use crate::core::{
        pd_decoder_grant::{
            issue_test_grant, issue_test_grant_at_control_url,
            issue_test_prepared_cancellation_receipt, issue_test_quarantine_receipt,
            issue_test_release_receipt, issue_test_reserve_refusal_receipt,
            issue_test_unbound_cancellation_target, DecoderGrantChildAccounting,
            DecoderInferenceRoute, TestEngineReceiptBinding,
        },
        HttpOrigin,
    };

    static NEXT_ROOM: AtomicU64 = AtomicU64::new(1);

    #[derive(Clone, Default)]
    struct ReserveRecoveryServerState {
        calls: Arc<AtomicUsize>,
        attempts: Arc<StdMutex<Vec<String>>>,
        first_request_seen: Arc<Notify>,
    }

    async fn reserve_recovery_handler(
        State(state): State<ReserveRecoveryServerState>,
        Json(request): Json<Value>,
    ) -> impl IntoResponse {
        let call_index = state.calls.fetch_add(1, AtomicOrdering::SeqCst);
        state.attempts.lock().unwrap().push(
            request["reservation_attempt_id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
        if call_index == 0 {
            state.first_request_seen.notify_one();
            future::pending::<()>().await;
            unreachable!("the deliberately ambiguous first reserve request cannot complete");
        }
        let receipt = json!({
            "schema_version": request["schema_version"],
            "operation": "reserve",
            "state": "refused",
            "prefill_process": request["prefill_process"],
            "prefill_bootstrap_endpoint": request["prefill_bootstrap_endpoint"],
            "decoder_process": request["decoder_process"],
            "logical_request_chain_id": request["logical_request_chain_id"],
            "reservation_attempt_id": request["reservation_attempt_id"],
            "reserve_attempt_digest": request["reserve_attempt_digest"],
            "source_tp_size": request["source_tp_size"],
            "prepared_ttl_ms": request["prepared_ttl_ms"],
            "inference_route": request["inference_route"],
            "request_shape": request["request_shape"],
            "reason_code": "capacity_exhausted",
            "diagnostic": null,
            "disposition": "retry_another_decoder",
            "receipt_id": Uuid::new_v4(),
            "receipt_digest": "73".repeat(32),
            "take_once": true,
        });
        (StatusCode::CONFLICT, Json(receipt))
    }

    async fn start_reserve_recovery_server() -> (String, ReserveRecoveryServerState, JoinHandle<()>)
    {
        let state = ReserveRecoveryServerState::default();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let application = Router::new()
            .route(
                "/_internal/pd/v1/decode-reservations/reserve",
                post(reserve_recovery_handler),
            )
            .with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, application).await.unwrap();
        });
        (format!("http://{address}"), state, task)
    }

    #[derive(Clone, Default)]
    struct CancellationRecoveryServerState {
        calls: Arc<AtomicUsize>,
    }

    async fn cancellation_recovery_handler(
        State(state): State<CancellationRecoveryServerState>,
        Json(mut request): Json<Value>,
    ) -> impl IntoResponse {
        state.calls.fetch_add(1, AtomicOrdering::SeqCst);
        let request = request
            .as_object_mut()
            .expect("cancellation request must be a JSON object");
        request.insert("operation".to_string(), Value::String("cancel".to_string()));
        request.insert("state".to_string(), Value::String("cancelled".to_string()));
        request.insert(
            "receipt_id".to_string(),
            Value::String(Uuid::new_v4().to_string()),
        );
        request.insert("receipt_digest".to_string(), Value::String("74".repeat(32)));
        request.insert("take_once".to_string(), Value::Bool(true));
        (StatusCode::OK, Json(request.clone()))
    }

    async fn start_cancellation_recovery_server(
    ) -> (String, CancellationRecoveryServerState, JoinHandle<()>) {
        let state = CancellationRecoveryServerState::default();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let application = Router::new()
            .fallback(post(cancellation_recovery_handler))
            .with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, application).await.unwrap();
        });
        (format!("http://{address}"), state, task)
    }

    fn stable_instance_id(name: &str) -> Uuid {
        let digest = blake3::hash(name.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest.as_bytes()[..16]);
        Uuid::from_bytes(bytes)
    }

    fn prefill_id(name: &str) -> PrefillId {
        PrefillId::new(
            HttpOrigin::parse("http://prefill.test:30000").unwrap(),
            stable_instance_id(name),
        )
        .unwrap()
    }

    fn decoder_id(name: &str) -> DecoderId {
        DecoderId::new(
            HttpOrigin::parse("http://decode.test:30001").unwrap(),
            stable_instance_id(name),
        )
        .unwrap()
    }

    fn compatibility(protocol: &str) -> EngineCompatibilityMetadata {
        compatibility_with_grant_protocol(protocol, "control-v1")
    }

    fn compatibility_with_grant_protocol(
        wire_protocol: &str,
        prepared_grant_protocol: &str,
    ) -> EngineCompatibilityMetadata {
        EngineCompatibilityMetadata::new(
            "gemma-4-31b-nvfp4@sha256:model",
            "gemma4-full10-swa50@sha256:layout",
            "bfloat16",
            wire_protocol,
            prepared_grant_protocol,
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
        replica_with_tp_and_id(id, 1, protocol)
    }

    fn replica_with_tp(
        name: &str,
        declared_decode_tp_size: usize,
        protocol: &str,
    ) -> DecoderReplicaMetadata {
        replica_with_tp_and_id(decoder_id(name), declared_decode_tp_size, protocol)
    }

    fn replica_with_tp_and_id(
        id: DecoderId,
        declared_decode_tp_size: usize,
        protocol: &str,
    ) -> DecoderReplicaMetadata {
        DecoderReplicaMetadata::new(
            id,
            declared_decode_tp_size,
            compatibility(protocol),
            scheduling(32, 32_000),
        )
        .unwrap()
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

    fn pending_charge(accounting: &[DecoderGrantChildAccounting]) -> PendingSchedulingCharge {
        PendingSchedulingCharge::new(
            accounting.len(),
            accounting
                .iter()
                .map(DecoderGrantChildAccounting::reserved_kv_tokens)
                .sum(),
            accounting
                .iter()
                .map(DecoderGrantChildAccounting::remaining_decode_tokens)
                .sum(),
        )
        .unwrap()
    }

    fn scalar_template() -> DecoderRequestTemplate {
        DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            bytes::Bytes::from_static(br#"{"text":"test"}"#),
        )
        .unwrap()
    }

    fn begin_scalar_admission(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        accounting: DecoderGrantChildAccounting,
    ) -> Result<PendingAdmission, DecoderPoolError> {
        let eligible_decoders: HashSet<DecoderId> = pool
            .snapshot()
            .replicas
            .into_iter()
            .map(|replica| replica.id)
            .collect();
        pool.begin_admission(
            owner,
            &eligible_decoders,
            &scalar_template(),
            PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000).unwrap(),
            Duration::from_secs(2),
            pending_charge(&[accounting]),
        )
    }

    fn test_selected_decoder(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
    ) -> Result<DecoderId, DecoderPoolError> {
        pool.validate_request_owner(owner)?;
        let state = pool.inner.state.lock();
        let chain = state
            .request_chains
            .get(&owner.chain_id)
            .ok_or(DecoderPoolError::UnknownRequestChain(owner.chain_id))?;
        let constraint = match &chain.state {
            RequestChainState::IdleOpen(constraint) => constraint,
            RequestChainState::Reserving(record) => {
                return Err(DecoderPoolError::RequestHasPendingAdmission {
                    request_id: chain.request_id.to_string(),
                    reservation_attempt_id: record.reservation_attempt_id,
                });
            }
            RequestChainState::Assigned(assignment_id)
            | RequestChainState::Quarantined(assignment_id) => {
                return Err(DecoderPoolError::RequestHasActiveCohort {
                    request_id: chain.request_id.to_string(),
                    assignment_id: *assignment_id,
                });
            }
            RequestChainState::Terminal => {
                return Err(DecoderPoolError::RequestChainTerminal(
                    chain.request_id.to_string(),
                ));
            }
        };
        match constraint {
            AdmissionRetryConstraint::AnyEligible => {
                let eligible_decoders: HashSet<DecoderId> =
                    state.replicas.keys().cloned().collect();
                Ok(select_decoders(
                    &state.replicas,
                    &chain.failed_decoders,
                    &eligible_decoders,
                    state.last_scheduled_decoder.as_ref(),
                )?
                .remove(0))
            }
            AdmissionRetryConstraint::SameDecoder(decoder_id) => Ok(decoder_id.clone()),
        }
    }

    fn issue_grant(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        decoder_id: DecoderId,
        grant_id: Uuid,
        slot_generations: Vec<DecoderSlotGeneration>,
        rooms: Vec<u64>,
        accounting: Vec<DecoderGrantChildAccounting>,
    ) -> BoundPreparedGrant {
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

    fn receipt_binding(cohort: &DecoderAssignmentCohort) -> TestEngineReceiptBinding {
        TestEngineReceiptBinding {
            grant_id: cohort.assignment_id(),
            decoder_id: cohort.decoder_id().clone(),
            child_request_ids: cohort.binding.child_request_ids().collect(),
            prefill_bootstrap_endpoint: cohort.binding.prefill_bootstrap_endpoint().clone(),
            slot_generations: cohort.slot_generations().to_vec(),
            bootstrap_rooms: cohort.bootstrap_rooms().to_vec(),
            grant_digest: cohort.grant_digest(),
        }
    }

    fn grant_receipt_binding(grant: &BoundPreparedGrant) -> TestEngineReceiptBinding {
        TestEngineReceiptBinding {
            grant_id: grant.grant_id(),
            decoder_id: grant.decoder_id().clone(),
            child_request_ids: grant.binding().child_request_ids().collect::<Vec<_>>(),
            prefill_bootstrap_endpoint: grant.prefill_bootstrap_endpoint().clone(),
            slot_generations: grant.slot_generations().to_vec(),
            bootstrap_rooms: grant.bootstrap_rooms().to_vec(),
            grant_digest: grant.grant_digest(),
        }
    }

    fn refusal_receipt(
        pool: &DecoderPool,
        pending: &PendingAdmission,
        disposition: DecoderReserveRefusalDisposition,
    ) -> DecoderReserveRefusalReceipt {
        issue_test_reserve_refusal_receipt(
            pool.snapshot().prefill_id,
            pending.decoder_id().clone(),
            pending.chain_id,
            pending.reservation_attempt_id(),
            pending.reserve_attempt_digest(),
            disposition,
            true,
        )
    }

    fn release_receipt(
        cohort: &DecoderAssignmentCohort,
        kind: EngineReleaseKind,
    ) -> EngineReleaseReceipt {
        issue_test_release_receipt(receipt_binding(cohort), kind, true)
    }

    fn quarantine_receipt(cohort: &DecoderAssignmentCohort) -> EngineQuarantineReceipt {
        issue_test_quarantine_receipt(receipt_binding(cohort), true)
    }

    fn retain_after_promotion(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
    ) -> RetainedEngineGrant {
        let mut promotion = pool.begin_promotion(cohort).unwrap();
        promotion.assume_test_promoted().unwrap()
    }

    fn pin_abort_after_promotion(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
    ) -> AbortReconciliationGrant {
        let mut promotion = pool.begin_promotion(cohort).unwrap();
        pool.pin_abort(
            cohort,
            &mut promotion,
            "test_abort",
            None,
            RetryDisposition::Terminal,
        )
        .unwrap()
    }

    fn pin_quarantine_after_promotion(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
    ) -> QuarantineReconciliationGrant {
        let mut promotion = pool.begin_promotion(cohort).unwrap();
        pool.pin_quarantine(cohort, &mut promotion, "test_quarantine", None)
            .unwrap()
    }

    fn release_before_activation(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        let _cancellation = pool.pin_cancellation(cohort, disposition)?;
        let receipt = release_receipt(cohort, EngineReleaseKind::PreparedCancelled);
        pool.apply_cancellation_receipt(cohort, &receipt, disposition)
    }

    fn release_after_abort(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
        disposition: RetryDisposition,
    ) -> Result<(), DecoderPoolError> {
        let receipt = release_receipt(cohort, EngineReleaseKind::Aborted);
        pool.apply_abort_release_receipt(cohort, &receipt, disposition)
    }

    fn release_after_completion(
        pool: &DecoderPool,
        cohort: &mut DecoderAssignmentCohort,
        retained: &mut RetainedEngineGrant,
    ) -> Result<(), DecoderPoolError> {
        let _completion = pool.pin_completion(cohort, retained)?;
        let receipt = release_receipt(cohort, EngineReleaseKind::Completed);
        pool.apply_completion_receipt(cohort, &receipt)
    }

    fn issue_next_grant(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        accounting: Vec<DecoderGrantChildAccounting>,
    ) -> Result<BoundPreparedGrant, DecoderPoolError> {
        let decoder_id = test_selected_decoder(pool, owner)?;
        let child_count = accounting.len();
        let first_room = NEXT_ROOM.fetch_add(child_count as u64, AtomicOrdering::Relaxed);
        let rooms = (0..child_count)
            .map(|offset| first_room + offset as u64)
            .collect();
        Ok(issue_grant(
            pool,
            owner,
            decoder_id,
            Uuid::new_v4(),
            (0..child_count)
                .map(|_| DecoderSlotGeneration::new(Uuid::new_v4()))
                .collect(),
            rooms,
            accounting,
        ))
    }

    fn bind_next(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        accounting: Vec<DecoderGrantChildAccounting>,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        let mut grant = issue_next_grant(pool, owner, accounting)?;
        let mut pending = begin_test_pending_for_grant(pool, owner, &grant)?;
        pool.bind_grant(&mut pending, &mut grant)
    }

    fn bind_issued_grant(
        pool: &DecoderPool,
        owner: &LogicalRequestOwner,
        grant: &mut BoundPreparedGrant,
    ) -> Result<DecoderAssignmentCohort, DecoderPoolError> {
        let mut pending = begin_test_pending_for_grant(pool, owner, grant)?;
        pool.bind_grant(&mut pending, grant)
    }

    #[test]
    fn registers_declared_tp_metadata_without_claiming_transport_correctness() {
        for declared_prefill_tp_size in [1, 2, 4] {
            let pool = pool(declared_prefill_tp_size);
            pool.register(replica_with_tp("decode-tp1", 1, "packed-v1"))
                .unwrap();
            if declared_prefill_tp_size % 2 == 0 {
                pool.register(replica_with_tp("decode-tp2", 2, "packed-v1"))
                    .unwrap();
            }
            assert_eq!(
                pool.snapshot().declared_prefill_tp_size,
                declared_prefill_tp_size
            );
            assert_eq!(
                pool.snapshot().replicas.len(),
                if declared_prefill_tp_size % 2 == 0 {
                    2
                } else {
                    1
                }
            );
        }
    }

    #[test]
    fn rejects_unsupported_prefill_tensor_parallelism() {
        for declared_prefill_tp_size in [0, 3, 8] {
            assert!(DecoderPool::new(
                prefill_id("prefill-invalid"),
                declared_prefill_tp_size,
                compatibility("packed-v1"),
            )
            .is_err());
        }
    }

    #[test]
    fn ratio_comparison_is_exact_without_cross_multiplication_overflow() {
        let maximum = usize::MAX as u128;
        assert_eq!(
            compare_ratio(maximum * 2, usize::MAX, maximum * 2 - 1, usize::MAX,),
            Ordering::Greater
        );
        assert_eq!(
            compare_ratio(maximum * 2, usize::MAX, maximum * 2, usize::MAX),
            Ordering::Equal
        );
        assert_eq!(compare_ratio(1, 3, 2, 5), Ordering::Less);
        assert_eq!(compare_ratio(7, 3, 9, 4), Ordering::Greater);
    }

    #[test]
    fn rejects_ineligible_declared_decoder_metadata() {
        let pool = pool(4);
        let tp3 = DecoderReplicaMetadata::new(
            decoder_id("decode-tp3"),
            3,
            compatibility("packed-v1"),
            scheduling(32, 32_000),
        )
        .unwrap();
        assert!(matches!(
            pool.register(tp3),
            Err(DecoderPoolError::IneligibleDecoderMetadata { .. })
        ));
        assert!(matches!(
            pool.register(replica("decode-wrong-wire", "packed-v2")),
            Err(DecoderPoolError::IneligibleDecoderMetadata { .. })
        ));
        let wrong_grant_protocol = DecoderReplicaMetadata::new(
            decoder_id("decode-wrong-grant-protocol"),
            1,
            compatibility_with_grant_protocol("packed-v1", "control-v2"),
            scheduling(32, 32_000),
        )
        .unwrap();
        assert!(matches!(
            pool.register(wrong_grant_protocol),
            Err(DecoderPoolError::IneligibleDecoderMetadata { .. })
        ));
    }

    #[test]
    fn replacement_is_unavailable_until_atomic_activation() {
        let pool = pool(4);
        let draining_id = decoder_id("decode@generation-1");
        let replacement_id = decoder_id("decode@generation-2");
        pool.register(replica_with_id(draining_id.clone(), "packed-v1"))
            .unwrap();
        pool.register_unavailable(replica_with_id(replacement_id.clone(), "packed-v1"))
            .unwrap();

        let owner = pool.begin_request("before-activation").unwrap();
        let pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        assert_eq!(pending.decoder_id(), &draining_id);
        drop(pending);

        pool.activate_replacement(&draining_id, &replacement_id)
            .unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(
            snapshot
                .replicas
                .iter()
                .find(|replica| replica.id == draining_id)
                .unwrap()
                .availability,
            DecoderAvailability::Draining
        );
        assert_eq!(
            snapshot
                .replicas
                .iter()
                .find(|replica| replica.id == replacement_id)
                .unwrap()
                .availability,
            DecoderAvailability::Ready
        );

        let replacement_owner = pool.begin_request("after-activation").unwrap();
        let replacement =
            begin_scalar_admission(&pool, &replacement_owner, child_accounting()).unwrap();
        assert_eq!(replacement.decoder_id(), &replacement_id);
    }

    #[test]
    fn prefill_drain_rejects_new_owners_but_allows_existing_request_admission() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let existing = pool.begin_request("existing").unwrap();

        pool.begin_draining();

        assert_eq!(
            pool.begin_request("new").unwrap_err(),
            DecoderPoolError::PrefillPoolDraining
        );
        let pending = begin_scalar_admission(&pool, &existing, child_accounting()).unwrap();
        drop(pending);
    }

    #[test]
    fn pending_admission_reconciles_while_draining_and_blocks_retirement() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        let receipt = refusal_receipt(&pool, &pending, DecoderReserveRefusalDisposition::Terminal);

        pool.begin_draining();
        drop(owner);

        assert_eq!(
            pool.ensure_retirable().unwrap_err(),
            DecoderPoolError::PrefillPoolInUse {
                request_chains: 1,
                pending_admissions: 1,
                assignments: 0,
                room_owners: 0,
                allocation_owners: 0,
                quarantined_cohorts: 0,
            }
        );
        assert_eq!(
            pool.install_reserve_refusal_proof(&mut pending, &receipt)
                .unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        pool.ensure_retirable().unwrap();
    }

    #[test]
    fn balances_supported_prefill_tp_across_arbitrary_replica_counts() {
        for prefill_tp_size in [1, 2, 4] {
            for replica_count in [1, 2, 3, 5] {
                let pool = pool(prefill_tp_size);
                for index in 0..replica_count {
                    pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                        .unwrap();
                }

                let mut requests = Vec::new();
                let mut cohorts = Vec::new();
                for index in 0..(replica_count * 3) {
                    let request = pool
                        .begin_request(format!("tp{prefill_tp_size}-n{replica_count}-{index}"))
                        .unwrap();
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
                    vec![3; replica_count]
                );

                for (mut request, mut cohort) in requests.into_iter().zip(cohorts) {
                    release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal)
                        .unwrap();
                    pool.finalize_request(&mut request).unwrap();
                }
            }
        }
    }

    #[test]
    fn round_robins_serial_completions_across_arbitrary_replica_counts() {
        for prefill_tp_size in [1, 2, 4] {
            for replica_count in [1, 2, 3, 5] {
                let pool = pool(prefill_tp_size);
                for index in 0..replica_count {
                    pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                        .unwrap();
                }
                let expected_round: Vec<DecoderId> = pool
                    .snapshot()
                    .replicas
                    .into_iter()
                    .map(|replica| replica.id)
                    .collect();
                let mut observed = Vec::new();

                for index in 0..(replica_count * 3) {
                    let mut request = pool
                        .begin_request(format!(
                            "serial-tp{prefill_tp_size}-n{replica_count}-{index}"
                        ))
                        .unwrap();
                    let mut cohort = bind_next(&pool, &request, scalar_accounting()).unwrap();
                    observed.push(cohort.decoder_id().clone());
                    let mut retained = retain_after_promotion(&pool, &mut cohort);
                    release_after_completion(&pool, &mut cohort, &mut retained).unwrap();
                    pool.finalize_request(&mut request).unwrap();
                }

                for round in observed.chunks(replica_count) {
                    assert_eq!(round, expected_round.as_slice());
                }
            }
        }
    }

    #[test]
    fn balances_pending_supported_tp_bursts_across_arbitrary_replica_counts() {
        for prefill_tp_size in [1, 2, 4] {
            for replica_count in [1, 2, 3, 5] {
                let pool = pool(prefill_tp_size);
                for index in 0..replica_count {
                    pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                        .unwrap();
                }

                let mut owners = Vec::new();
                let mut pending = Vec::new();
                for index in 0..(replica_count * 3) {
                    let owner = pool
                        .begin_request(format!(
                            "pending-tp{prefill_tp_size}-n{replica_count}-{index}"
                        ))
                        .unwrap();
                    let admission =
                        begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
                    owners.push(owner);
                    pending.push(admission);
                }

                assert_eq!(
                    pool.snapshot()
                        .replicas
                        .iter()
                        .map(|replica| replica.pending_admissions)
                        .collect::<Vec<_>>(),
                    vec![3; replica_count]
                );
                drop(pending);
                assert!(pool
                    .snapshot()
                    .replicas
                    .iter()
                    .all(|replica| replica.pending_admissions == 0));
                drop(owners);
            }
        }
    }

    #[test]
    fn concurrent_begin_admission_has_exactly_one_winner() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();

        let results = thread::scope(|scope| {
            let first = scope.spawn(|| begin_scalar_admission(&pool, &owner, child_accounting()));
            let second = scope.spawn(|| begin_scalar_admission(&pool, &owner, child_accounting()));
            [first.join().unwrap(), second.join().unwrap()]
        });
        let mut winner = None;
        let mut rejected = 0usize;
        for result in results {
            match result {
                Ok(pending) => winner = Some(pending),
                Err(DecoderPoolError::RequestHasPendingAdmission { .. }) => rejected += 1,
                Err(error) => panic!("unexpected concurrent admission result: {error}"),
            }
        }
        assert!(winner.is_some());
        assert_eq!(rejected, 1);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 1);
        drop(winner);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
    }

    #[test]
    fn pending_load_changes_the_next_decoder_selection() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();

        let first = begin_scalar_admission(&pool, &first_owner, child_accounting()).unwrap();
        let second = begin_scalar_admission(&pool, &second_owner, child_accounting()).unwrap();
        assert_ne!(first.decoder_id(), second.decoder_id());
        assert_eq!(
            pool.snapshot()
                .replicas
                .iter()
                .map(|replica| replica.pending_admissions)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn dropping_an_unstarted_admission_rolls_back_exact_pending_load() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let pending =
            begin_scalar_admission(&pool, &owner, DecoderGrantChildAccounting::new(123, 45))
                .unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].pending_admissions, 1);
        assert_eq!(snapshot.replicas[0].pending_child_requests, 1);
        assert_eq!(snapshot.replicas[0].pending_reserved_kv_tokens, 123);
        assert_eq!(snapshot.replicas[0].pending_remaining_decode_tokens, 45);

        drop(pending);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].pending_admissions, 0);
        assert_eq!(snapshot.replicas[0].pending_child_requests, 0);
        assert_eq!(snapshot.replicas[0].pending_reserved_kv_tokens, 0);
        assert_eq!(snapshot.replicas[0].pending_remaining_decode_tokens, 0);
        let retry = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        drop(retry);
    }

    #[test]
    fn dropping_an_owner_then_its_unstarted_admission_reaps_the_chain() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        drop(owner);
        assert_eq!(pool.snapshot().active_logical_requests, 1);

        drop(pending);
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
    }

    #[test]
    fn dropping_an_unpolled_reserve_rolls_back_pending_ownership() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        let client = DecoderGrantControlClient::from_builder(reqwest::Client::builder()).unwrap();
        let reconciliation = pending
            .begin_reserve(
                &client,
                DecoderControlAuthorization::new("test-decoder-api-key").unwrap(),
            )
            .unwrap();

        drop(reconciliation);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
        let retry = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        drop(retry);
    }

    #[test]
    fn dropping_a_polled_reserve_retains_pending_ownership() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        let client = DecoderGrantControlClient::from_builder(reqwest::Client::builder()).unwrap();
        let mut reconciliation = pending
            .begin_reserve(
                &client,
                DecoderControlAuthorization::new("test-decoder-api-key").unwrap(),
            )
            .unwrap();
        reconciliation.mark_test_polled();
        drop(reconciliation);

        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 1);
        assert!(matches!(
            test_selected_decoder(&pool, &owner),
            Err(DecoderPoolError::RequestHasPendingAdmission { .. })
        ));
    }

    #[tokio::test]
    async fn dropped_reserve_future_resumes_the_exact_engine_attempt() {
        let (decoder_url, server, server_task) = start_reserve_recovery_server().await;
        let pool = pool(2);
        let decoder_id = DecoderId::new(
            HttpOrigin::parse(&decoder_url).unwrap(),
            stable_instance_id("decode-0@generation-1"),
        )
        .unwrap();
        pool.register(replica_with_id(decoder_id, "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        let reservation_attempt_id = pending.reservation_attempt_id();
        let client = DecoderGrantControlClient::from_builder(reqwest::Client::builder()).unwrap();
        let mut reconciliation = pending
            .begin_reserve(
                &client,
                DecoderControlAuthorization::new("test-decoder-api-key").unwrap(),
            )
            .unwrap();
        assert_eq!(
            reconciliation.reservation_attempt_id().unwrap(),
            reservation_attempt_id
        );
        assert_eq!(
            pool.recover_pending_admission(&owner, reservation_attempt_id)
                .unwrap_err(),
            DecoderPoolError::PendingAdmissionAlreadyClaimed(reservation_attempt_id)
        );

        let first_request_seen = server.first_request_seen.notified();
        let mut first_poll = Box::pin(reconciliation.reconcile_reserve());
        tokio::select! {
            _ = first_request_seen => {}
            result = &mut first_poll => {
                panic!("ambiguous reserve unexpectedly completed: {result:?}");
            }
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                panic!("first reserve request did not reach the control endpoint");
            }
        }
        drop(first_poll);
        drop(reconciliation);

        let mut resumed = pending.resume_reserve().unwrap();
        assert_eq!(
            resumed.reservation_attempt_id().unwrap(),
            reservation_attempt_id
        );
        assert!(matches!(
            resumed.reconcile_reserve().await.unwrap(),
            PendingReserveOutcome::Refused(PendingAdmissionDisposition::RetryAnotherDecoder)
        ));
        drop(resumed);

        assert_eq!(server.calls.load(AtomicOrdering::SeqCst), 2);
        let attempts = server.attempts.lock().unwrap();
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0], reservation_attempt_id.to_string());
        assert_eq!(attempts[1], reservation_attempt_id.to_string());
        server_task.abort();
    }

    #[tokio::test]
    async fn dropped_cancellation_and_failed_pool_apply_reuse_one_engine_receipt() {
        let (decoder_url, server, server_task) = start_cancellation_recovery_server().await;
        let pool = pool(4);
        let decoder_id = DecoderId::new(
            HttpOrigin::parse(&decoder_url).unwrap(),
            stable_instance_id("decode-0@generation-1"),
        )
        .unwrap();
        pool.register(replica_with_id(decoder_id.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let snapshot = pool.snapshot();
        let mut grant = issue_test_grant_at_control_url(
            snapshot.prefill_id,
            owner.chain_id(),
            snapshot.declared_prefill_tp_size,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![NEXT_ROOM.fetch_add(1, AtomicOrdering::Relaxed)],
            scalar_accounting(),
            &decoder_url,
        )
        .unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();

        let cancellation = pool
            .begin_bound_pending_cancellation(
                &mut pending,
                &mut grant,
                PendingAdmissionDisposition::Terminal,
            )
            .unwrap();
        drop(cancellation);
        assert_eq!(server.calls.load(AtomicOrdering::SeqCst), 0);

        let mut resumed = pool.resume_pending_cancellation(&mut pending).unwrap();
        pool.inner
            .state
            .lock()
            .replicas
            .get_mut(&decoder_id)
            .unwrap()
            .pending_child_requests = 0;
        assert_eq!(
            resumed.reconcile_cancellation().await.unwrap_err(),
            PendingReconciliationError::Pool(DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id: resumed.pending.reservation_attempt_id(),
                reason: "decoder pending accounting differs from request-chain reservations",
            })
        );
        pool.inner
            .state
            .lock()
            .replicas
            .get_mut(&decoder_id)
            .unwrap()
            .pending_child_requests = 1;
        drop(resumed);

        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        assert_eq!(server.calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
        server_task.abort();
    }

    #[test]
    fn pending_claims_are_exclusive_and_recoverable_after_owner_drop() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let reservation_attempt_id = pending.reservation_attempt_id();

        assert_eq!(
            pool.recover_pending_admission(&owner, reservation_attempt_id)
                .unwrap_err(),
            DecoderPoolError::PendingAdmissionAlreadyClaimed(reservation_attempt_id)
        );
        drop(pending);

        pending = pool
            .recover_pending_admission(&owner, reservation_attempt_id)
            .unwrap();
        assert_eq!(pending.reservation_attempt_id(), reservation_attempt_id);
        drop(owner);
        drop(pending);

        let mut orphaned = pool
            .recover_orphaned_pending_admission(reservation_attempt_id)
            .unwrap();
        let proof = PendingCancellationProof::Bound(issue_test_release_receipt(
            grant_receipt_binding(&grant),
            EngineReleaseKind::PreparedCancelled,
            true,
        ));
        let mut cancellation = pool
            .begin_bound_pending_cancellation(
                &mut orphaned,
                &mut grant,
                PendingAdmissionDisposition::Terminal,
            )
            .unwrap();
        assert_eq!(
            cancellation.install_test_proof(proof).unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        drop(cancellation);
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
    }

    #[test]
    fn reserve_refusal_proof_replays_exactly_and_rejects_an_altered_receipt() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(
            &pool,
            &pending,
            DecoderReserveRefusalDisposition::RetrySameDecoder,
        );

        pool.install_reserve_refusal_proof(&mut pending, &receipt)
            .unwrap();
        pool.install_reserve_refusal_proof(&mut pending, &receipt)
            .unwrap();
        let altered = refusal_receipt(
            &pool,
            &pending,
            DecoderReserveRefusalDisposition::RetrySameDecoder,
        );
        assert_eq!(
            pool.install_reserve_refusal_proof(&mut pending, &altered)
                .unwrap_err(),
            DecoderPoolError::ConflictingPendingAdmissionProof(pending.reservation_attempt_id())
        );
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::RetrySameDecoder
        );
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::RetrySameDecoder
        );
    }

    #[test]
    fn resolved_admission_replays_while_a_new_attempt_is_reserving() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut resolved = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(
            &pool,
            &resolved,
            DecoderReserveRefusalDisposition::RetrySameDecoder,
        );
        pool.install_reserve_refusal_proof(&mut resolved, &receipt)
            .unwrap();
        let current = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();

        assert_eq!(
            pool.resume_pending_admission(&mut resolved).unwrap(),
            PendingAdmissionDisposition::RetrySameDecoder
        );
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 1);
        drop(current);
    }

    #[test]
    fn retry_same_refusal_pins_the_exact_decoder_generation() {
        let pool = pool(2);
        let first_generation = decoder_id("decode@generation-1");
        pool.register(replica_with_id(first_generation.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(
            &pool,
            &pending,
            DecoderReserveRefusalDisposition::RetrySameDecoder,
        );
        assert_eq!(
            pool.install_reserve_refusal_proof(&mut pending, &receipt)
                .unwrap(),
            PendingAdmissionDisposition::RetrySameDecoder
        );

        pool.set_availability(&first_generation, DecoderAvailability::Draining)
            .unwrap();
        pool.register(replica_with_id(
            decoder_id("decode@generation-2"),
            "packed-v1",
        ))
        .unwrap();
        assert_eq!(
            begin_scalar_admission(&pool, &owner, child_accounting()).unwrap_err(),
            DecoderPoolError::RetryDecoderUnavailable(first_generation)
        );
    }

    #[test]
    fn retry_another_refusal_excludes_only_the_failed_generation() {
        let pool = pool(4);
        let first_generation = decoder_id("decode@generation-1");
        let replacement_generation = decoder_id("decode@generation-2");
        pool.register(replica_with_id(first_generation.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(
            &pool,
            &pending,
            DecoderReserveRefusalDisposition::RetryAnotherDecoder,
        );
        pool.install_reserve_refusal_proof(&mut pending, &receipt)
            .unwrap();
        pool.resume_pending_admission(&mut pending).unwrap();

        pool.set_availability(&first_generation, DecoderAvailability::Draining)
            .unwrap();
        pool.register(replica_with_id(replacement_generation.clone(), "packed-v1"))
            .unwrap();
        let retry = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        assert_eq!(retry.decoder_id(), &replacement_generation);
    }

    #[test]
    fn retry_another_refusal_visits_each_ready_decoder_once_before_exhaustion() {
        let pool = pool(4);
        for index in 0..7 {
            pool.register(replica(&format!("decode-{index}"), "packed-v1"))
                .unwrap();
        }
        let owner = pool.begin_request("request").unwrap();
        let mut attempted_decoders = HashSet::new();

        for _ in 0..7 {
            let mut pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
            assert!(attempted_decoders.insert(pending.decoder_id().clone()));
            let refusal = refusal_receipt(
                &pool,
                &pending,
                DecoderReserveRefusalDisposition::RetryAnotherDecoder,
            );
            assert_eq!(
                pool.install_reserve_refusal_proof(&mut pending, &refusal)
                    .unwrap(),
                PendingAdmissionDisposition::RetryAnotherDecoder
            );
        }

        assert_eq!(attempted_decoders.len(), 7);
        assert!(pool
            .snapshot()
            .replicas
            .iter()
            .all(|replica| replica.availability == DecoderAvailability::Ready));
        assert_eq!(
            begin_scalar_admission(&pool, &owner, child_accounting()).unwrap_err(),
            DecoderPoolError::RetryAlternativesExhausted
        );
    }

    #[test]
    fn terminal_reserve_refusal_closes_the_request_chain() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(&pool, &pending, DecoderReserveRefusalDisposition::Terminal);
        pool.install_reserve_refusal_proof(&mut pending, &receipt)
            .unwrap();
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        assert_eq!(
            begin_scalar_admission(&pool, &owner, child_accounting()).unwrap_err(),
            DecoderPoolError::RequestChainTerminal("request".to_string())
        );
    }

    #[test]
    fn decoder_removal_waits_for_pending_admission_ownership() {
        let pool = pool(4);
        let decoder_id = decoder_id("decode-0");
        pool.register(replica_with_id(decoder_id.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let pending = begin_scalar_admission(&pool, &owner, child_accounting()).unwrap();
        pool.set_availability(&decoder_id, DecoderAvailability::Draining)
            .unwrap();

        assert_eq!(
            pool.remove(&decoder_id).unwrap_err(),
            DecoderPoolError::DecoderInUse {
                decoder_id: decoder_id.clone(),
                active_cohorts: 0,
                pending_admissions: 1,
            }
        );
        drop(pending);
        pool.remove(&decoder_id).unwrap();
    }

    #[test]
    fn dropped_owner_is_reaped_only_after_receipt_backed_pending_resolution() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(&pool, &pending, DecoderReserveRefusalDisposition::Terminal);
        drop(owner);
        assert_eq!(pool.snapshot().active_logical_requests, 1);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 1);

        assert_eq!(
            pool.install_reserve_refusal_proof(&mut pending, &receipt)
                .unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
    }

    #[test]
    fn pending_cancellation_requires_a_pinned_intent() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let proof = PendingCancellationProof::Bound(issue_test_release_receipt(
            grant_receipt_binding(&grant),
            EngineReleaseKind::PreparedCancelled,
            true,
        ));

        assert_eq!(
            pool.install_pending_cancellation_proof(&mut pending, &proof)
                .unwrap_err(),
            DecoderPoolError::PendingCancellationNotPinned(pending.reservation_attempt_id())
        );
    }

    #[test]
    fn pending_cancellation_intent_is_exact_and_idempotent() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let target = grant.cancellation_target().unwrap();

        pool.pin_pending_cancellation(
            &pending,
            &target,
            PendingAdmissionDisposition::RetrySameDecoder,
        )
        .unwrap();
        pool.pin_pending_cancellation(
            &pending,
            &target,
            PendingAdmissionDisposition::RetrySameDecoder,
        )
        .unwrap();
        assert_eq!(
            pool.pin_pending_cancellation(
                &pending,
                &target,
                PendingAdmissionDisposition::Terminal,
            )
            .unwrap_err(),
            DecoderPoolError::ConflictingPendingAdmissionProof(
                pending.reservation_attempt_id()
            )
        );

        let altered_grant = issue_grant(
            &pool,
            &owner,
            pending.decoder_id().clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![999],
            scalar_accounting(),
        );
        let altered_target = altered_grant.cancellation_target().unwrap();
        assert_eq!(
            pool.pin_pending_cancellation(
                &pending,
                &altered_target,
                PendingAdmissionDisposition::RetrySameDecoder,
            )
            .unwrap_err(),
            DecoderPoolError::ConflictingPendingAdmissionProof(pending.reservation_attempt_id())
        );
    }

    #[test]
    fn bound_pending_cancellation_proof_replays_and_rejects_conflicts() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt_binding = grant_receipt_binding(&grant);
        let proof = PendingCancellationProof::Bound(issue_test_release_receipt(
            receipt_binding.clone(),
            EngineReleaseKind::PreparedCancelled,
            true,
        ));
        let mut cancellation = pool
            .begin_bound_pending_cancellation(
                &mut pending,
                &mut grant,
                PendingAdmissionDisposition::Terminal,
            )
            .unwrap();
        cancellation.install_test_proof(proof.clone()).unwrap();
        drop(cancellation);
        pool.install_pending_cancellation_proof(&mut pending, &proof)
            .unwrap();
        pool.install_pending_cancellation_proof(&mut pending, &proof)
            .unwrap();

        let altered = PendingCancellationProof::Bound(issue_test_release_receipt(
            receipt_binding,
            EngineReleaseKind::PreparedCancelled,
            true,
        ));
        assert_eq!(
            pool.install_pending_cancellation_proof(&mut pending, &altered)
                .unwrap_err(),
            DecoderPoolError::ConflictingPendingAdmissionProof(pending.reservation_attempt_id())
        );
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::Terminal
        );
        assert_eq!(pool.snapshot().replicas[0].pending_admissions, 0);
    }

    #[test]
    fn bound_pending_cancellation_rejects_the_wrong_release_kind() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let target = grant.cancellation_target().unwrap();
        pool.pin_pending_cancellation(&pending, &target, PendingAdmissionDisposition::Terminal)
            .unwrap();
        let proof = PendingCancellationProof::Bound(issue_test_release_receipt(
            grant_receipt_binding(&grant),
            EngineReleaseKind::Completed,
            true,
        ));

        assert_eq!(
            pool.install_pending_cancellation_proof(&mut pending, &proof)
                .unwrap_err(),
            DecoderPoolError::InvalidPendingCancellationProof {
                reservation_attempt_id: pending.reservation_attempt_id(),
                reason: "receipt does not match the pinned cancellation target",
            }
        );
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap_err(),
            DecoderPoolError::InvalidPendingCancellationProof {
                reservation_attempt_id: target.reservation_attempt_id(),
                reason: "receipt does not match the pinned cancellation target",
            }
        );
    }

    #[test]
    fn unbound_pending_cancellation_matches_the_exact_attempted_digest() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let first_grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &first_grant).unwrap();
        let target =
            issue_test_unbound_cancellation_target(&first_grant, Some(first_grant.grant_digest()));
        pool.pin_pending_cancellation(
            &pending,
            &target,
            PendingAdmissionDisposition::RetryAnotherDecoder,
        )
        .unwrap();

        let second_grant = issue_grant(
            &pool,
            &owner,
            pending.decoder_id().clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![987],
            scalar_accounting(),
        );
        let altered_target = issue_test_unbound_cancellation_target(
            &second_grant,
            Some(second_grant.grant_digest()),
        );
        let altered = PendingCancellationProof::Unbound(issue_test_prepared_cancellation_receipt(
            &altered_target,
            true,
        ));
        assert!(matches!(
            pool.install_pending_cancellation_proof(&mut pending, &altered),
            Err(DecoderPoolError::InvalidPendingCancellationProof { .. })
        ));

        let proof = PendingCancellationProof::Unbound(issue_test_prepared_cancellation_receipt(
            &target, true,
        ));
        pool.install_pending_cancellation_proof(&mut pending, &proof)
            .unwrap();
        assert_eq!(
            pool.resume_pending_admission(&mut pending).unwrap(),
            PendingAdmissionDisposition::RetryAnotherDecoder
        );
    }

    #[test]
    fn refusal_and_cancellation_intents_conflict_in_both_orders() {
        for refusal_first in [false, true] {
            let pool = pool(2);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            let owner = pool
                .begin_request(format!("request-{refusal_first}"))
                .unwrap();
            let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
            let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
            let target = grant.cancellation_target().unwrap();
            let refusal =
                refusal_receipt(&pool, &pending, DecoderReserveRefusalDisposition::Terminal);

            if refusal_first {
                pool.install_reserve_refusal_proof(&mut pending, &refusal)
                    .unwrap();
                assert!(matches!(
                    pool.pin_pending_cancellation(
                        &pending,
                        &target,
                        PendingAdmissionDisposition::Terminal,
                    ),
                    Err(DecoderPoolError::ConflictingPendingAdmissionProof(_))
                ));
            } else {
                pool.pin_pending_cancellation(
                    &pending,
                    &target,
                    PendingAdmissionDisposition::Terminal,
                )
                .unwrap();
                assert!(matches!(
                    pool.install_reserve_refusal_proof(&mut pending, &refusal),
                    Err(DecoderPoolError::ConflictingPendingAdmissionProof(_))
                ));
            }
        }
    }

    #[test]
    fn rejected_bound_grant_can_release_pending_ownership_with_a_bound_proof() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = test_selected_decoder(&pool, &first_owner).unwrap();
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        let mut rejected_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut pending =
            begin_test_pending_for_grant(&pool, &second_owner, &rejected_grant).unwrap();
        assert_eq!(
            pool.bind_grant(&mut pending, &mut rejected_grant)
                .unwrap_err(),
            DecoderPoolError::GrantRoomInUse {
                decoder_id,
                room: 700,
            }
        );

        let receipt_binding = grant_receipt_binding(&rejected_grant);
        let proof = PendingCancellationProof::Bound(issue_test_release_receipt(
            receipt_binding,
            EngineReleaseKind::PreparedCancelled,
            true,
        ));
        let mut cancellation = pool
            .begin_bound_pending_cancellation(
                &mut pending,
                &mut rejected_grant,
                PendingAdmissionDisposition::Terminal,
            )
            .unwrap();
        cancellation.install_test_proof(proof).unwrap();
        drop(cancellation);
        pool.resume_pending_admission(&mut pending).unwrap();

        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].pending_admissions, 0);
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        release_before_activation(&pool, &mut first, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn binding_atomically_converts_pending_load_into_active_accounting() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let accounting = vec![
            DecoderGrantChildAccounting::new(100, 10),
            DecoderGrantChildAccounting::new(200, 20),
        ];
        let mut grant = issue_next_grant(&pool, &owner, accounting).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let before = pool.snapshot();
        assert_eq!(before.replicas[0].pending_admissions, 1);
        assert_eq!(before.replicas[0].pending_child_requests, 2);
        assert_eq!(before.replicas[0].pending_reserved_kv_tokens, 300);
        assert_eq!(before.replicas[0].pending_remaining_decode_tokens, 30);
        assert_eq!(before.replicas[0].active_cohorts, 0);

        let mut cohort = pool.bind_grant(&mut pending, &mut grant).unwrap();
        let after = pool.snapshot();
        assert_eq!(after.replicas[0].pending_admissions, 0);
        assert_eq!(after.replicas[0].pending_child_requests, 0);
        assert_eq!(after.replicas[0].pending_reserved_kv_tokens, 0);
        assert_eq!(after.replicas[0].pending_remaining_decode_tokens, 0);
        assert_eq!(after.replicas[0].active_cohorts, 1);
        assert_eq!(after.replicas[0].active_child_requests, 2);
        assert_eq!(after.replicas[0].reserved_kv_tokens, 300);
        assert_eq!(after.replicas[0].remaining_decode_tokens, 30);
        release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn corrupted_pending_accounting_preserves_the_durable_release_proof() {
        let pool = pool(2);
        let decoder_id = decoder_id("decode-0");
        pool.register(replica_with_id(decoder_id.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        let receipt = refusal_receipt(&pool, &pending, DecoderReserveRefusalDisposition::Terminal);
        pool.inner
            .state
            .lock()
            .replicas
            .get_mut(&decoder_id)
            .unwrap()
            .pending_child_requests = 0;

        assert_eq!(
            pool.install_reserve_refusal_proof(&mut pending, &receipt)
                .unwrap_err(),
            DecoderPoolError::InconsistentPendingAdmission {
                reservation_attempt_id: pending.reservation_attempt_id(),
                reason: "decoder pending accounting differs from request-chain reservations",
            }
        );
        let retained = {
            let state = pool.inner.state.lock();
            let chain = state.request_chains.get(&owner.chain_id()).unwrap();
            let RequestChainState::Reserving(record) = &chain.state else {
                panic!("failed reconciliation must retain the pending attempt");
            };
            record.reconciliation.clone()
        };
        assert_eq!(
            retained,
            Some(PendingAdmissionReconciliationRecord::Refusal(
                receipt.clone()
            ))
        );

        pool.inner
            .state
            .lock()
            .replicas
            .get_mut(&decoder_id)
            .unwrap()
            .pending_child_requests = 1;
        pool.resume_pending_admission(&mut pending).unwrap();
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
        let decoder_id = test_selected_decoder(&pool, &request).unwrap();
        let mut grant = issue_grant(
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
        let mut cohort = bind_issued_grant(&pool, &request, &mut grant).unwrap();
        assert_eq!(cohort.bootstrap_rooms(), &[41, 43, 42]);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].active_child_requests, 3);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 600);
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 60);

        let _abort = pin_abort_after_promotion(&pool, &mut cohort);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 1);
        release_after_abort(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 0);
        assert_eq!(snapshot.replicas[0].active_child_requests, 0);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
    }

    #[test]
    fn pool_binding_moves_only_the_exact_prepared_authority() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut mismatched_grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();

        assert!(grant.begin_cancellation().is_err());
        let _mismatched_cancellation = mismatched_grant.begin_cancellation().unwrap();

        let mut retained = retain_after_promotion(&pool, &mut cohort);
        assert_eq!(
            pool.begin_promotion(&mut cohort).unwrap_err(),
            DecoderPoolError::InvalidTransition {
                assignment_id: cohort.assignment_id(),
                actual: "active",
                requested: "active",
            }
        );
        release_after_completion(&pool, &mut cohort, &mut retained).unwrap();
    }

    #[test]
    fn lifecycle_consumers_require_the_exact_pool_binding() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let mut first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let mut second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();

        let foreign_binding = DecoderGrantPoolBinding::new(&second.binding);
        let first_prepared = first.prepared_grant.as_mut().unwrap();
        assert_eq!(
            first_prepared.begin_promotion(foreign_binding).unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool promotion binding does not match the exact prepared grant"
                    .to_string(),
            )
        );

        let mut promotion = pool.begin_promotion(&mut first).unwrap();
        assert_eq!(
            promotion
                .begin_abort(
                    DecoderGrantPoolBinding::new(&second.binding),
                    "test_abort",
                    None,
                )
                .unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool abort binding does not match the exact promotion grant".to_string(),
            )
        );
        assert_eq!(
            promotion
                .begin_quarantine(
                    DecoderGrantPoolBinding::new(&second.binding),
                    "test_quarantine",
                    None,
                )
                .unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool quarantine binding does not match the exact promotion grant"
                    .to_string(),
            )
        );

        let mut retained = promotion.assume_test_promoted().unwrap();
        assert_eq!(
            retained
                .begin_completion(DecoderGrantPoolBinding::new(&second.binding))
                .unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool completion binding does not match the exact retained grant"
                    .to_string(),
            )
        );
        assert_eq!(
            retained
                .begin_abort(
                    DecoderGrantPoolBinding::new(&second.binding),
                    "test_abort",
                    None,
                )
                .unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool abort binding does not match the exact retained grant".to_string(),
            )
        );
        assert_eq!(
            retained
                .begin_quarantine(
                    DecoderGrantPoolBinding::new(&second.binding),
                    "test_quarantine",
                    None,
                )
                .unwrap_err(),
            EngineGrantError::ProtocolViolation(
                "decoder-pool quarantine binding does not match the exact retained grant"
                    .to_string(),
            )
        );

        release_after_completion(&pool, &mut first, &mut retained).unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn completion_cannot_consume_authority_from_another_active_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let mut first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let mut second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();
        let mut first_retained = retain_after_promotion(&pool, &mut first);
        let mut second_retained = retain_after_promotion(&pool, &mut second);

        assert_eq!(
            pool.begin_completion(&mut second, &mut first_retained)
                .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "completion grant does not exactly match the assignment".to_string(),
            )
        );
        assert_eq!(first.phase, CohortPhase::Active);
        assert_eq!(second.phase, CohortPhase::Active);

        release_after_completion(&pool, &mut first, &mut first_retained).unwrap();
        release_after_completion(&pool, &mut second, &mut second_retained).unwrap();
    }

    #[test]
    fn abort_cannot_consume_authority_from_another_active_cohort() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let mut first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let mut second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();
        let mut first_promotion = pool.begin_promotion(&mut first).unwrap();
        let mut second_promotion = pool.begin_promotion(&mut second).unwrap();

        assert_eq!(
            pool.begin_abort_from_promotion(
                &mut second,
                &mut first_promotion,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "abort grant does not exactly match the assignment".to_string(),
            )
        );
        assert_eq!(first.phase, CohortPhase::Active);
        assert_eq!(second.phase, CohortPhase::Active);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 0);

        let _first_abort = pool
            .pin_abort(
                &mut first,
                &mut first_promotion,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap();
        release_after_abort(&pool, &mut first, RetryDisposition::Terminal).unwrap();
        let _second_abort = pool
            .pin_abort(
                &mut second,
                &mut second_promotion,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap();
        release_after_abort(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn quarantine_cannot_consume_authority_from_another_active_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let mut first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let mut second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();
        let mut first_promotion = pool.begin_promotion(&mut first).unwrap();
        let mut second_promotion = pool.begin_promotion(&mut second).unwrap();

        assert_eq!(
            pool.begin_quarantine_from_promotion(
                &mut second,
                &mut first_promotion,
                "test_quarantine",
                None,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "quarantine grant does not exactly match the assignment".to_string(),
            )
        );
        assert_eq!(first.phase, CohortPhase::Active);
        assert_eq!(second.phase, CohortPhase::Active);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 0);

        let _first_quarantine = pool
            .pin_quarantine(&mut first, &mut first_promotion, "test_quarantine", None)
            .unwrap();
        let first_receipt = quarantine_receipt(&first);
        pool.apply_quarantine_receipt(&mut first, &first_receipt)
            .unwrap();
        let _second_quarantine = pool
            .pin_quarantine(&mut second, &mut second_promotion, "test_quarantine", None)
            .unwrap();
        let second_receipt = quarantine_receipt(&second);
        pool.apply_quarantine_receipt(&mut second, &second_receipt)
            .unwrap();
    }

    #[test]
    fn unavailable_completion_authority_does_not_change_the_active_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let mut consumed = retained.begin_test_completion().unwrap();

        assert_eq!(
            pool.begin_completion(&mut cohort, &mut retained)
                .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "matching retained grant has no completion authority".to_string(),
            )
        );
        assert_eq!(cohort.phase, CohortPhase::Active);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 0);
        consumed.assume_test_reconciled().unwrap();
    }

    #[test]
    fn unavailable_abort_authority_does_not_change_the_active_cohort() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut promotion = pool.begin_promotion(&mut cohort).unwrap();
        let mut consumed = promotion.begin_test_abort("test_abort", None).unwrap();

        assert_eq!(
            pool.begin_abort_from_promotion(
                &mut cohort,
                &mut promotion,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "matching engine grant could not pin abort authority".to_string(),
            )
        );
        assert_eq!(cohort.phase, CohortPhase::Active);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 0);
        consumed.assume_test_reconciled().unwrap();
    }

    #[test]
    fn unavailable_quarantine_authority_does_not_change_the_active_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let mut consumed = retained
            .begin_test_quarantine("test_quarantine", None)
            .unwrap();

        assert_eq!(
            pool.begin_quarantine_from_retained(
                &mut cohort,
                &mut retained,
                "test_quarantine",
                None,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "matching engine grant could not pin quarantine authority".to_string(),
            )
        );
        assert_eq!(cohort.phase, CohortPhase::Active);
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 0);
        consumed.assume_test_reconciled().unwrap();
    }

    #[test]
    fn promotion_preflights_the_complete_live_ledger_before_consuming_authority() {
        #[derive(Clone, Copy, Debug)]
        enum Corruption {
            ActiveCohorts,
            ActiveChildRequests,
            ReservedKvTokens,
            RemainingDecodeTokens,
            MissingRoom,
            MissingAllocation,
            MissingReplica,
            MissingChain,
            WrongChainAssignment,
            TerminalChain,
            PrematureTerminalReconciliation,
        }

        impl Corruption {
            fn expected_reason(self) -> &'static str {
                match self {
                    Self::ActiveCohorts
                    | Self::ActiveChildRequests
                    | Self::ReservedKvTokens
                    | Self::RemainingDecodeTokens => {
                        "decoder accounting is smaller than the assignment ledger"
                    }
                    Self::MissingRoom => "an assigned bootstrap room is not retained",
                    Self::MissingAllocation => "an assigned slot generation is not retained",
                    Self::MissingReplica => "assigned decoder is missing",
                    Self::MissingChain => "request chain is missing",
                    Self::WrongChainAssignment => "request chain does not own this assignment",
                    Self::TerminalChain => "request chain does not own this assignment",
                    Self::PrematureTerminalReconciliation => {
                        "assignment phase differs from its terminal reconciliation ledger"
                    }
                }
            }
        }

        for corruption in [
            Corruption::ActiveCohorts,
            Corruption::ActiveChildRequests,
            Corruption::ReservedKvTokens,
            Corruption::RemainingDecodeTokens,
            Corruption::MissingRoom,
            Corruption::MissingAllocation,
            Corruption::MissingReplica,
            Corruption::MissingChain,
            Corruption::WrongChainAssignment,
            Corruption::TerminalChain,
            Corruption::PrematureTerminalReconciliation,
        ] {
            let pool = pool(4);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            let owner = pool
                .begin_request(format!("request-{corruption:?}"))
                .unwrap();
            let mut grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
            let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();
            let decoder_id = cohort.decoder_id().clone();
            let assignment_id = cohort.assignment_id();
            let chain_id = owner.chain_id();
            let room = cohort.bootstrap_rooms()[0];
            let allocation = cohort
                .binding
                .allocation_keys()
                .next()
                .expect("scalar cohort must have one allocation");
            let baseline = pool
                .snapshot()
                .replicas
                .into_iter()
                .next()
                .expect("registered replica must be present");
            let injected_receipt = quarantine_receipt(&cohort);
            let mut removed_replica: Option<ReplicaState> = None;
            let mut removed_chain: Option<RequestChainRecord> = None;

            {
                let mut state = pool.inner.state.lock();
                match corruption {
                    Corruption::ActiveCohorts => {
                        state.replicas.get_mut(&decoder_id).unwrap().active_cohorts = 0;
                    }
                    Corruption::ActiveChildRequests => {
                        state
                            .replicas
                            .get_mut(&decoder_id)
                            .unwrap()
                            .active_child_requests = 0;
                    }
                    Corruption::ReservedKvTokens => {
                        state
                            .replicas
                            .get_mut(&decoder_id)
                            .unwrap()
                            .reserved_kv_tokens = 0;
                    }
                    Corruption::RemainingDecodeTokens => {
                        state
                            .replicas
                            .get_mut(&decoder_id)
                            .unwrap()
                            .remaining_decode_tokens = 0;
                    }
                    Corruption::MissingRoom => {
                        assert_eq!(
                            state.room_owners.remove(&(decoder_id.clone(), room)),
                            Some(assignment_id)
                        );
                    }
                    Corruption::MissingAllocation => {
                        assert_eq!(
                            state.allocation_owners.remove(&allocation),
                            Some(assignment_id)
                        );
                    }
                    Corruption::MissingReplica => {
                        removed_replica = state.replicas.remove(&decoder_id);
                    }
                    Corruption::MissingChain => {
                        removed_chain = state.request_chains.remove(&chain_id);
                    }
                    Corruption::WrongChainAssignment => {
                        state.request_chains.get_mut(&chain_id).unwrap().state =
                            RequestChainState::Assigned(Uuid::new_v4());
                    }
                    Corruption::TerminalChain => {
                        state.request_chains.get_mut(&chain_id).unwrap().state =
                            RequestChainState::Terminal;
                    }
                    Corruption::PrematureTerminalReconciliation => {
                        state
                            .assignments
                            .get_mut(&assignment_id)
                            .unwrap()
                            .terminal_reconciliation =
                            Some(TerminalReconciliationRecord::Quarantine {
                                proof: Some(injected_receipt.clone()),
                            });
                    }
                }
            }

            assert_eq!(
                pool.begin_promotion(&mut cohort).unwrap_err(),
                DecoderPoolError::InconsistentAssignment {
                    assignment_id,
                    reason: corruption.expected_reason(),
                },
                "{corruption:?}"
            );
            assert_eq!(cohort.phase, CohortPhase::Reserved, "{corruption:?}");
            assert_eq!(
                pool.inner
                    .state
                    .lock()
                    .assignments
                    .get(&assignment_id)
                    .unwrap()
                    .phase,
                CohortPhase::Reserved,
                "{corruption:?}"
            );

            {
                let mut state = pool.inner.state.lock();
                match corruption {
                    Corruption::ActiveCohorts
                    | Corruption::ActiveChildRequests
                    | Corruption::ReservedKvTokens
                    | Corruption::RemainingDecodeTokens => {
                        let replica = state.replicas.get_mut(&decoder_id).unwrap();
                        replica.active_cohorts = baseline.active_cohorts;
                        replica.active_child_requests = baseline.active_child_requests;
                        replica.reserved_kv_tokens = baseline.reserved_kv_tokens;
                        replica.remaining_decode_tokens = baseline.remaining_decode_tokens;
                    }
                    Corruption::MissingRoom => {
                        assert!(state
                            .room_owners
                            .insert((decoder_id.clone(), room), assignment_id)
                            .is_none());
                    }
                    Corruption::MissingAllocation => {
                        assert!(state
                            .allocation_owners
                            .insert(allocation.clone(), assignment_id)
                            .is_none());
                    }
                    Corruption::MissingReplica => {
                        assert!(state
                            .replicas
                            .insert(decoder_id.clone(), removed_replica.take().unwrap())
                            .is_none());
                    }
                    Corruption::MissingChain => {
                        assert!(state
                            .request_chains
                            .insert(chain_id, removed_chain.take().unwrap())
                            .is_none());
                    }
                    Corruption::WrongChainAssignment => {
                        state.request_chains.get_mut(&chain_id).unwrap().state =
                            RequestChainState::Assigned(assignment_id);
                    }
                    Corruption::TerminalChain => {
                        state.request_chains.get_mut(&chain_id).unwrap().state =
                            RequestChainState::Assigned(assignment_id);
                    }
                    Corruption::PrematureTerminalReconciliation => {
                        state
                            .assignments
                            .get_mut(&assignment_id)
                            .unwrap()
                            .terminal_reconciliation = None;
                    }
                }
            }

            let mut retained = retain_after_promotion(&pool, &mut cohort);
            release_after_completion(&pool, &mut cohort, &mut retained).unwrap();
        }
    }

    #[test]
    fn promotion_rejects_cross_cohort_aggregate_drift_before_consuming_authority() {
        let pool = pool(4);
        let decoder_id = decoder_id("decode-0");
        pool.register(replica_with_id(decoder_id.clone(), "packed-v1"))
            .unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut second_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![701],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        let mut second = bind_issued_grant(&pool, &second_owner, &mut second_grant).unwrap();
        let baseline = pool
            .snapshot()
            .replicas
            .into_iter()
            .next()
            .expect("registered replica must be present");
        {
            let mut state = pool.inner.state.lock();
            let replica = state.replicas.get_mut(&decoder_id).unwrap();
            replica.active_cohorts = 1;
            replica.active_child_requests = 1;
            replica.reserved_kv_tokens = child_accounting().reserved_kv_tokens();
            replica.remaining_decode_tokens = child_accounting().remaining_decode_tokens();
        }

        assert_eq!(
            pool.begin_promotion(&mut first).unwrap_err(),
            DecoderPoolError::InconsistentAssignment {
                assignment_id: first.assignment_id(),
                reason: "decoder accounting differs from the assignment ledger",
            }
        );
        assert_eq!(first.phase, CohortPhase::Reserved);
        assert_eq!(second.phase, CohortPhase::Reserved);

        {
            let mut state = pool.inner.state.lock();
            let replica = state.replicas.get_mut(&decoder_id).unwrap();
            replica.active_cohorts = baseline.active_cohorts;
            replica.active_child_requests = baseline.active_child_requests;
            replica.reserved_kv_tokens = baseline.reserved_kv_tokens;
            replica.remaining_decode_tokens = baseline.remaining_decode_tokens;
        }
        let mut retained = retain_after_promotion(&pool, &mut first);
        release_after_completion(&pool, &mut first, &mut retained).unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn unavailable_promotion_authority_leaves_the_cohort_reserved() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut grant = issue_next_grant(&pool, &owner, scalar_accounting()).unwrap();
        let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();
        let _cancellation = cohort
            .prepared_grant
            .as_mut()
            .unwrap()
            .begin_cancellation()
            .unwrap();

        assert_eq!(
            pool.begin_promotion(&mut cohort).unwrap_err(),
            DecoderPoolError::InvalidGrant(
                "matching prepared grant has no promotion authority".to_string(),
            )
        );
        assert_eq!(cohort.phase, CohortPhase::Reserved);
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 1);
    }

    #[test]
    fn grant_cannot_cross_logical_request_owners() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = test_selected_decoder(&pool, &first_owner).unwrap();
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![11],
            scalar_accounting(),
        );
        let mut second_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id,
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![12],
            scalar_accounting(),
        );
        let mut second_pending =
            begin_test_pending_for_grant(&pool, &second_owner, &second_grant).unwrap();

        assert_eq!(
            pool.bind_grant(&mut second_pending, &mut first_grant)
                .unwrap_err(),
            DecoderPoolError::GrantRequestMismatch {
                expected: second_owner.chain_id(),
                actual: first_owner.chain_id(),
            }
        );
        let mut second = pool
            .bind_grant(&mut second_pending, &mut second_grant)
            .unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        release_before_activation(&pool, &mut first, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn grant_cannot_cross_prefill_process_generations() {
        let pool = pool_for_prefill("prefill-0@generation-2", 2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let decoder_id = test_selected_decoder(&pool, &owner).unwrap();
        let mut grant = issue_test_grant(
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

        let mut pending = begin_test_pending_for_grant(&pool, &owner, &grant).unwrap();
        assert_eq!(
            pool.bind_grant(&mut pending, &mut grant).unwrap_err(),
            DecoderPoolError::GrantPrefillMismatch {
                expected: prefill_id("prefill-0@generation-2"),
                actual: prefill_id("prefill-0@generation-1"),
            }
        );
    }

    #[test]
    fn reserved_release_rejects_every_binding_mismatch_without_consuming_the_cohort() {
        let expected_reasons = [
            "assignment identity differs",
            "decoder process generation differs",
            "ordered child request identities differ",
            "prefill bootstrap endpoint differs",
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
            let selected_decoder = test_selected_decoder(&pool, &owner).unwrap();
            let mut grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![11],
                scalar_accounting(),
            );
            let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();
            let alternate_grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![12],
                scalar_accounting(),
            );
            let mut binding = receipt_binding(&cohort);
            match mismatch {
                0 => binding.grant_id = Uuid::new_v4(),
                1 => binding.decoder_id = decoder_id("decode-other"),
                2 => binding.child_request_ids = vec![Uuid::new_v4()],
                3 => {
                    binding.prefill_bootstrap_endpoint =
                        PrefillBootstrapEndpoint::new("other-prefill.test", 5001).unwrap();
                }
                4 => {
                    binding.slot_generations = vec![DecoderSlotGeneration::new(Uuid::new_v4())];
                }
                5 => binding.bootstrap_rooms = vec![12],
                6 => binding.grant_digest = alternate_grant.grant_digest(),
                7 | 8 => {}
                _ => unreachable!(),
            }
            let kind = if mismatch == 7 {
                EngineReleaseKind::Completed
            } else {
                EngineReleaseKind::PreparedCancelled
            };
            let receipt = issue_test_release_receipt(binding, kind, mismatch != 8);
            let assignment_id = cohort.assignment_id();
            let _cancellation = pool
                .pin_cancellation(&mut cohort, RetryDisposition::Terminal)
                .unwrap();

            assert_eq!(
                pool.apply_cancellation_receipt(&mut cohort, &receipt, RetryDisposition::Terminal,)
                    .unwrap_err(),
                DecoderPoolError::InvalidEngineReleaseReceipt {
                    assignment_id,
                    reason: expected_reason,
                }
            );
            assert_eq!(pool.snapshot().replicas[0].active_cohorts, 1);

            let valid_receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
            pool.apply_cancellation_receipt(
                &mut cohort,
                &valid_receipt,
                RetryDisposition::Terminal,
            )
            .unwrap();
            assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
        }
    }

    #[test]
    fn completing_cohort_rejects_an_invalid_receipt_without_consuming_the_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let _completion = pool.pin_completion(&mut cohort, &mut retained).unwrap();
        let invalid_receipt = release_receipt(&cohort, EngineReleaseKind::Aborted);
        let assignment_id = cohort.assignment_id();

        assert_eq!(
            pool.apply_completion_receipt(&mut cohort, &invalid_receipt)
                .unwrap_err(),
            DecoderPoolError::InvalidEngineReleaseReceipt {
                assignment_id,
                reason: "release kind differs",
            }
        );
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 1);

        let valid_receipt = release_receipt(&cohort, EngineReleaseKind::Completed);
        pool.apply_completion_receipt(&mut cohort, &valid_receipt)
            .unwrap();
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[test]
    fn aborting_cohort_rejects_an_invalid_receipt_without_consuming_the_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let _abort = pin_abort_after_promotion(&pool, &mut cohort);
        let invalid_receipt = release_receipt(&cohort, EngineReleaseKind::Completed);
        let assignment_id = cohort.assignment_id();

        assert_eq!(
            pool.apply_abort_release_receipt(
                &mut cohort,
                &invalid_receipt,
                RetryDisposition::Terminal,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidEngineReleaseReceipt {
                assignment_id,
                reason: "release kind differs",
            }
        );
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 1);

        release_after_abort(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
    }

    #[test]
    fn wrong_phase_release_can_be_retried_after_the_required_transition() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let receipt = release_receipt(&cohort, EngineReleaseKind::Completed);
        let assignment_id = cohort.assignment_id();

        assert_eq!(
            pool.apply_completion_receipt(&mut cohort, &receipt)
                .unwrap_err(),
            DecoderPoolError::InvalidTransition {
                assignment_id,
                actual: "reserved",
                requested: "install completion proof",
            }
        );
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 1);

        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let _completion = pool.pin_completion(&mut cohort, &mut retained).unwrap();
        pool.apply_completion_receipt(&mut cohort, &receipt)
            .unwrap();
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[test]
    fn foreign_pool_failure_preserves_the_issuing_pool_capability() {
        let issuing_pool = pool_for_prefill("prefill-issuing", 2);
        let foreign_pool = pool_for_prefill("prefill-foreign", 2);
        issuing_pool
            .register(replica("decode-0", "packed-v1"))
            .unwrap();
        let owner = issuing_pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&issuing_pool, &owner, scalar_accounting()).unwrap();
        let _cancellation = issuing_pool
            .pin_cancellation(&mut cohort, RetryDisposition::Terminal)
            .unwrap();
        let receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);

        assert_eq!(
            foreign_pool
                .apply_cancellation_receipt(&mut cohort, &receipt, RetryDisposition::Terminal,)
                .unwrap_err(),
            DecoderPoolError::ForeignAssignment
        );
        assert_eq!(issuing_pool.snapshot().replicas[0].active_cohorts, 1);

        issuing_pool
            .apply_cancellation_receipt(&mut cohort, &receipt, RetryDisposition::Terminal)
            .unwrap();
        assert_eq!(issuing_pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[test]
    fn inconsistent_release_preflight_can_be_repaired_and_retried() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
        let decoder_id = cohort.decoder_id().clone();
        let room = cohort.bootstrap_rooms()[0];
        let assignment_id = cohort.assignment_id();
        let _cancellation = pool
            .pin_cancellation(&mut cohort, RetryDisposition::Terminal)
            .unwrap();
        {
            let mut state = pool.inner.state.lock();
            assert_eq!(
                state.room_owners.remove(&(decoder_id.clone(), room)),
                Some(assignment_id)
            );
        }

        assert_eq!(
            pool.apply_cancellation_receipt(&mut cohort, &receipt, RetryDisposition::Terminal,)
                .unwrap_err(),
            DecoderPoolError::InconsistentAssignment {
                assignment_id,
                reason: "an assigned bootstrap room is not retained",
            }
        );
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].active_child_requests, 1);
        {
            let mut state = pool.inner.state.lock();
            assert!(state
                .room_owners
                .insert((decoder_id, room), assignment_id)
                .is_none());
        }

        pool.apply_cancellation_receipt(&mut cohort, &receipt, RetryDisposition::Terminal)
            .unwrap();
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[tokio::test]
    async fn pool_bound_cancellation_reuses_its_receipt_after_pool_preflight_failure() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
        let decoder_id = cohort.decoder_id().clone();
        let room = cohort.bootstrap_rooms()[0];
        let assignment_id = cohort.assignment_id();
        let mut reconciliation = pool
            .begin_cancellation(&mut cohort, RetryDisposition::Terminal)
            .unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_cancellation_proof(reconciliation.cohort, receipt)
            .unwrap();
        assert_eq!(
            pool.inner
                .state
                .lock()
                .room_owners
                .remove(&(decoder_id.clone(), room)),
            Some(assignment_id)
        );

        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::Pool(
                DecoderPoolError::InconsistentAssignment {
                    assignment_id: failed_assignment_id,
                    reason: "an assigned bootstrap room is not retained",
                }
            )) if failed_assignment_id == assignment_id
        ));
        drop(reconciliation);
        assert!(pool
            .inner
            .state
            .lock()
            .room_owners
            .insert((decoder_id, room), assignment_id)
            .is_none());

        pool.resume_terminal_reconciliation(&mut cohort).unwrap();
        pool.resume_terminal_reconciliation(&mut cohort).unwrap();
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[test]
    fn terminal_reconciliation_proof_install_is_exact_and_preserves_pinned_disposition() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let failed_decoder = cohort.decoder_id().clone();
        let assignment_id = cohort.assignment_id();
        let first_receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
        let conflicting_receipt = release_receipt(&cohort, EngineReleaseKind::PreparedCancelled);
        let mut cancellation = pool
            .pin_cancellation(&mut cohort, RetryDisposition::DecoderFailed)
            .unwrap();

        assert_eq!(
            pool.resume_terminal_reconciliation(&mut cohort)
                .unwrap_err(),
            DecoderPoolError::TerminalProofPending(assignment_id)
        );
        assert_eq!(
            pool.apply_cancellation_receipt(
                &mut cohort,
                &first_receipt,
                RetryDisposition::Retryable,
            )
            .unwrap_err(),
            DecoderPoolError::ConflictingTerminalProof(assignment_id)
        );
        cancellation.assume_test_reconciled().unwrap();
        pool.install_cancellation_proof(&cohort, first_receipt.clone())
            .unwrap();
        pool.install_cancellation_proof(&cohort, first_receipt)
            .unwrap();
        assert_eq!(
            pool.install_cancellation_proof(&cohort, conflicting_receipt)
                .unwrap_err(),
            DecoderPoolError::ConflictingTerminalProof(assignment_id)
        );

        pool.resume_terminal_reconciliation(&mut cohort).unwrap();
        assert_ne!(
            test_selected_decoder(&pool, &owner).unwrap(),
            failed_decoder
        );
    }

    #[tokio::test]
    async fn pool_bound_completion_terminalizes_only_its_borrowed_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let receipt = release_receipt(&cohort, EngineReleaseKind::Completed);
        let mut reconciliation = pool.begin_completion(&mut cohort, &mut retained).unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_completion_proof(
                reconciliation.cohort,
                EngineCompletionOutcome::Completed(receipt),
            )
            .unwrap();

        reconciliation.reconcile().await.unwrap();
        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "completion"
            ))
        ));
        drop(reconciliation);
        assert_eq!(pool.snapshot().replicas[0].active_cohorts, 0);
    }

    #[tokio::test]
    async fn pool_bound_completion_terminalizes_an_authoritative_quarantine() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let chain_id = owner.chain_id();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let assignment_id = cohort.assignment_id();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let receipt = quarantine_receipt(&cohort);
        let mut reconciliation = pool.begin_completion(&mut cohort, &mut retained).unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_completion_proof(
                reconciliation.cohort,
                EngineCompletionOutcome::Quarantined(receipt.clone()),
            )
            .unwrap();

        reconciliation.reconcile().await.unwrap();
        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "completion"
            ))
        ));
        drop(reconciliation);
        drop(owner);

        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quarantined_cohorts, 1);
        assert_eq!(pool.quarantine_receipt(&cohort).unwrap(), Some(receipt));
        let state = pool.inner.state.lock();
        let chain = state.request_chains.get(&chain_id).unwrap();
        assert!(!chain.owner_alive);
        assert!(matches!(
            &chain.state,
            RequestChainState::Quarantined(value) if *value == assignment_id
        ));
    }

    #[tokio::test]
    async fn pool_bound_abort_applies_an_authoritative_release() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut promotion = pool.begin_promotion(&mut cohort).unwrap();
        let receipt = release_receipt(&cohort, EngineReleaseKind::Aborted);
        let mut reconciliation = pool
            .begin_abort_from_promotion(
                &mut cohort,
                &mut promotion,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_abort_proof(reconciliation.cohort, EngineAbortOutcome::Aborted(receipt))
            .unwrap();

        reconciliation.reconcile().await.unwrap();
        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "abort"
            ))
        ));
        drop(reconciliation);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
    }

    #[tokio::test]
    async fn pool_bound_abort_applies_an_authoritative_quarantine_fallback() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        let receipt = quarantine_receipt(&cohort);
        let mut reconciliation = pool
            .begin_abort_from_retained(
                &mut cohort,
                &mut retained,
                "test_abort",
                None,
                RetryDisposition::Terminal,
            )
            .unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_abort_proof(
                reconciliation.cohort,
                EngineAbortOutcome::Quarantined(receipt.clone()),
            )
            .unwrap();

        reconciliation.reconcile().await.unwrap();
        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "abort"
            ))
        ));
        drop(reconciliation);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quarantined_cohorts, 1);
        assert_eq!(pool.quarantine_receipt(&cohort).unwrap(), Some(receipt));
    }

    #[tokio::test]
    async fn pool_bound_quarantine_retains_only_its_borrowed_cohort() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut promotion = pool.begin_promotion(&mut cohort).unwrap();
        let receipt = quarantine_receipt(&cohort);
        let mut reconciliation = pool
            .begin_quarantine_from_promotion(&mut cohort, &mut promotion, "test_quarantine", None)
            .unwrap();
        reconciliation.engine.assume_test_reconciled().unwrap();
        reconciliation
            .pool
            .install_quarantine_proof(reconciliation.cohort, receipt.clone())
            .unwrap();

        reconciliation.reconcile().await.unwrap();
        assert!(matches!(
            reconciliation.reconcile().await,
            Err(DecoderAssignmentReconciliationError::AlreadyComplete(
                "quarantine"
            ))
        ));
        drop(reconciliation);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quarantined_cohorts, 1);
        assert_eq!(pool.quarantine_receipt(&cohort).unwrap(), Some(receipt));
    }

    #[test]
    fn retry_cannot_replay_or_alter_a_decoder_slot_binding() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let decoder_id = test_selected_decoder(&pool, &owner).unwrap();
        let slot_generations = vec![
            DecoderSlotGeneration::new(Uuid::new_v4()),
            DecoderSlotGeneration::new(Uuid::new_v4()),
        ];
        let mut original = issue_grant(
            &pool,
            &owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            slot_generations.clone(),
            vec![101, 102],
            vec![child_accounting(), child_accounting()],
        );
        let mut first = bind_issued_grant(&pool, &owner, &mut original).unwrap();
        release_before_activation(&pool, &mut first, RetryDisposition::Retryable).unwrap();
        let mut pending = begin_test_pending_for_grant(&pool, &owner, &original).unwrap();

        assert_eq!(
            pool.bind_grant(&mut pending, &mut original).unwrap_err(),
            DecoderPoolError::GrantAlreadyBound {
                child_index: 0,
                decoder_id: decoder_id.clone(),
                slot_generation: slot_generations[0].as_uuid(),
            }
        );
        for rooms in [vec![101, 103], vec![102, 101]] {
            let mut altered = issue_grant(
                &pool,
                &owner,
                decoder_id.clone(),
                Uuid::new_v4(),
                slot_generations.clone(),
                rooms,
                vec![child_accounting(), child_accounting()],
            );
            assert_eq!(
                pool.bind_grant(&mut pending, &mut altered).unwrap_err(),
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
        let registered_first = decoder_id("decode-0");
        let registered_second = decoder_id("decode-1");
        pool.register(replica_with_id(registered_first.clone(), "packed-v1"))
            .unwrap();
        pool.register(replica_with_id(registered_second.clone(), "packed-v1"))
            .unwrap();
        let owner = pool.begin_request("request").unwrap();
        let first_decoder = test_selected_decoder(&pool, &owner).unwrap();
        let second_decoder = if first_decoder == registered_first {
            registered_second
        } else {
            registered_first
        };
        let slot_generation = DecoderSlotGeneration::new(Uuid::new_v4());
        let mut first_grant = issue_grant(
            &pool,
            &owner,
            first_decoder,
            Uuid::new_v4(),
            vec![slot_generation],
            vec![101],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &owner, &mut first_grant).unwrap();
        release_before_activation(&pool, &mut first, RetryDisposition::DecoderFailed).unwrap();

        let mut second_grant = issue_grant(
            &pool,
            &owner,
            second_decoder.clone(),
            Uuid::new_v4(),
            vec![slot_generation],
            vec![101],
            scalar_accounting(),
        );
        let mut second = bind_issued_grant(&pool, &owner, &mut second_grant).unwrap();
        assert_eq!(second.decoder_id(), &second_decoder);
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn active_bootstrap_room_cannot_be_owned_twice_on_one_decoder() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = test_selected_decoder(&pool, &first_owner).unwrap();
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        let mut second_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut second_pending =
            begin_test_pending_for_grant(&pool, &second_owner, &second_grant).unwrap();
        assert_eq!(
            pool.bind_grant(&mut second_pending, &mut second_grant)
                .unwrap_err(),
            DecoderPoolError::GrantRoomInUse {
                decoder_id,
                room: 700,
            }
        );
        release_before_activation(&pool, &mut first, RetryDisposition::Terminal).unwrap();
        let mut second = pool
            .bind_grant(&mut second_pending, &mut second_grant)
            .unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn active_slot_generation_cannot_be_owned_by_two_request_chains() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let decoder_id = test_selected_decoder(&pool, &first_owner).unwrap();
        let slot_generation = DecoderSlotGeneration::new(Uuid::new_v4());
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![slot_generation],
            vec![700],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        let mut second_grant = issue_grant(
            &pool,
            &second_owner,
            decoder_id.clone(),
            Uuid::new_v4(),
            vec![slot_generation],
            vec![701],
            scalar_accounting(),
        );
        let mut second_pending =
            begin_test_pending_for_grant(&pool, &second_owner, &second_grant).unwrap();

        assert_eq!(
            pool.bind_grant(&mut second_pending, &mut second_grant)
                .unwrap_err(),
            DecoderPoolError::GrantSlotGenerationInUse {
                decoder_id,
                slot_generation: slot_generation.as_uuid(),
            }
        );

        release_before_activation(&pool, &mut first, RetryDisposition::Terminal).unwrap();
        let mut second = pool
            .bind_grant(&mut second_pending, &mut second_grant)
            .unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn equal_active_room_numbers_are_valid_on_separate_decoders() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();
        let first_owner = pool.begin_request("first").unwrap();
        let second_owner = pool.begin_request("second").unwrap();
        let first_decoder = test_selected_decoder(&pool, &first_owner).unwrap();
        let mut first_grant = issue_grant(
            &pool,
            &first_owner,
            first_decoder.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );
        let mut first = bind_issued_grant(&pool, &first_owner, &mut first_grant).unwrap();
        let second_decoder = test_selected_decoder(&pool, &second_owner).unwrap();
        let mut second_grant = issue_grant(
            &pool,
            &second_owner,
            second_decoder.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![700],
            scalar_accounting(),
        );

        let mut second = bind_issued_grant(&pool, &second_owner, &mut second_grant).unwrap();
        assert_ne!(first_decoder, second_decoder);
        assert_eq!(first.decoder_id(), &first_decoder);
        assert_eq!(second.decoder_id(), &second_decoder);
        release_before_activation(&pool, &mut first, RetryDisposition::Terminal).unwrap();
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn request_owner_finalization_prevents_retry_history_bleed() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        pool.register(replica("decode-1", "packed-v1")).unwrap();

        let mut first_owner = pool.begin_request("reused-id").unwrap();
        let mut first = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        let failed_decoder = first.decoder_id().clone();
        release_before_activation(&pool, &mut first, RetryDisposition::DecoderFailed).unwrap();

        let mut retry = bind_next(&pool, &first_owner, scalar_accounting()).unwrap();
        assert_ne!(retry.decoder_id(), &failed_decoder);
        release_before_activation(&pool, &mut retry, RetryDisposition::DecoderFailed).unwrap();
        pool.finalize_request(&mut first_owner).unwrap();

        let second_owner = pool.begin_request("reused-id").unwrap();
        let mut second = bind_next(&pool, &second_owner, scalar_accounting()).unwrap();
        assert_eq!(second.decoder_id(), &failed_decoder);
        release_before_activation(&pool, &mut second, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn terminal_release_closes_the_logical_request_chain() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();

        assert_eq!(
            test_selected_decoder(&pool, &owner).unwrap_err(),
            DecoderPoolError::RequestChainTerminal("request".to_string())
        );
        pool.finalize_request(&mut owner).unwrap();
    }

    #[test]
    fn dropped_owner_is_reaped_after_its_active_cohort_terminates() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        drop(owner);
        assert_eq!(pool.snapshot().active_logical_requests, 1);
        release_before_activation(&pool, &mut cohort, RetryDisposition::DecoderFailed).unwrap();
        assert_eq!(pool.snapshot().active_logical_requests, 0);
        let replacement = pool.begin_request("request").unwrap();
        let mut replacement_cohort = bind_next(&pool, &replacement, scalar_accounting()).unwrap();
        release_before_activation(&pool, &mut replacement_cohort, RetryDisposition::Terminal)
            .unwrap();
    }

    #[test]
    fn dropped_owner_cannot_cross_promotion_but_can_still_cancel() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let chain_id = owner.chain_id();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        drop(owner);

        assert_eq!(
            pool.begin_promotion(&mut cohort).unwrap_err(),
            DecoderPoolError::RequestChainOwnerDropped(chain_id)
        );
        assert_eq!(cohort.phase, CohortPhase::Reserved);
        release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
        assert_eq!(pool.snapshot().active_logical_requests, 0);
    }

    #[test]
    fn retryable_non_decoder_failure_preserves_destination_eligibility() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut first = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let decoder_id = first.decoder_id().clone();
        release_before_activation(&pool, &mut first, RetryDisposition::Retryable).unwrap();

        let mut retry = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        assert_eq!(retry.decoder_id(), &decoder_id);
        release_before_activation(&pool, &mut retry, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn exhausted_retry_chain_is_explicit_and_finalizable() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        release_before_activation(&pool, &mut cohort, RetryDisposition::DecoderFailed).unwrap();
        assert_eq!(
            test_selected_decoder(&pool, &owner).unwrap_err(),
            DecoderPoolError::RetryAlternativesExhausted
        );
        pool.finalize_request(&mut owner).unwrap();

        let replacement = pool.begin_request("request").unwrap();
        let mut replacement_cohort = bind_next(&pool, &replacement, scalar_accounting()).unwrap();
        release_before_activation(&pool, &mut replacement_cohort, RetryDisposition::Terminal)
            .unwrap();
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
        let mut failed = bind_next(&pool, &retry_owner, scalar_accounting()).unwrap();
        let failed_decoder = failed.decoder_id().clone();
        release_before_activation(&pool, &mut failed, RetryDisposition::DecoderFailed).unwrap();
        pool.set_availability(&failed_decoder, DecoderAvailability::Draining)
            .unwrap();

        let blocker_owner = pool.begin_request("blocker").unwrap();
        let mut blocker = bind_next(&pool, &blocker_owner, scalar_accounting()).unwrap();
        assert_ne!(blocker.decoder_id(), &failed_decoder);
        assert_eq!(
            test_selected_decoder(&pool, &retry_owner).unwrap(),
            blocker.decoder_id().clone()
        );

        let available_decoder = blocker.decoder_id().clone();
        release_before_activation(&pool, &mut blocker, RetryDisposition::Terminal).unwrap();
        let mut retry = bind_next(&pool, &retry_owner, scalar_accounting()).unwrap();
        assert_eq!(retry.decoder_id(), &available_decoder);
        release_before_activation(&pool, &mut retry, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn abort_phase_and_quiescing_counter_change_as_one_operation() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let _abort = pin_abort_after_promotion(&pool, &mut cohort);
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 1);
        release_after_abort(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn quarantine_rejects_every_binding_mismatch_without_consuming_the_cohort() {
        let expected_reasons = [
            "assignment identity differs",
            "decoder process generation differs",
            "ordered child request identities differ",
            "prefill bootstrap endpoint differs",
            "ordered decoder slot generations differ",
            "ordered bootstrap rooms differ",
            "grant digest differs",
            "receipt does not attest take-once reconciliation",
        ];
        for (mismatch, expected_reason) in expected_reasons.into_iter().enumerate() {
            let pool = pool(4);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            let owner = pool.begin_request(format!("request-{mismatch}")).unwrap();
            let selected_decoder = test_selected_decoder(&pool, &owner).unwrap();
            let mut grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![11],
                scalar_accounting(),
            );
            let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();
            let _quarantine = pin_quarantine_after_promotion(&pool, &mut cohort);
            let alternate_grant = issue_grant(
                &pool,
                &owner,
                selected_decoder.clone(),
                Uuid::new_v4(),
                vec![DecoderSlotGeneration::new(Uuid::new_v4())],
                vec![12],
                scalar_accounting(),
            );
            let mut binding = receipt_binding(&cohort);
            match mismatch {
                0 => binding.grant_id = Uuid::new_v4(),
                1 => binding.decoder_id = decoder_id("decode-other"),
                2 => binding.child_request_ids = vec![Uuid::new_v4()],
                3 => {
                    binding.prefill_bootstrap_endpoint =
                        PrefillBootstrapEndpoint::new("other-prefill.test", 5001).unwrap();
                }
                4 => {
                    binding.slot_generations = vec![DecoderSlotGeneration::new(Uuid::new_v4())];
                }
                5 => binding.bootstrap_rooms = vec![12],
                6 => binding.grant_digest = alternate_grant.grant_digest(),
                7 => {}
                _ => unreachable!(),
            }
            let receipt = issue_test_quarantine_receipt(binding, mismatch != 7);
            let assignment_id = cohort.assignment_id();

            assert_eq!(
                pool.apply_quarantine_receipt(&mut cohort, &receipt)
                    .unwrap_err(),
                DecoderPoolError::InvalidEngineQuarantineReceipt {
                    assignment_id,
                    reason: expected_reason,
                }
            );
            let snapshot = pool.snapshot();
            assert_eq!(snapshot.replicas[0].quiescing_cohorts, 1);
            assert_eq!(snapshot.replicas[0].quarantined_cohorts, 0);

            let valid_receipt = quarantine_receipt(&cohort);
            pool.apply_quarantine_receipt(&mut cohort, &valid_receipt)
                .unwrap();
            assert_eq!(
                pool.quarantine_receipt(&cohort).unwrap(),
                Some(valid_receipt)
            );
        }
    }

    #[test]
    fn quarantine_rejects_missing_rooms_without_consuming_the_cohort() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let _quarantine = pin_quarantine_after_promotion(&pool, &mut cohort);
        let receipt = quarantine_receipt(&cohort);
        let room = cohort.bootstrap_rooms()[0];
        let decoder_id = cohort.decoder_id().clone();
        {
            let mut state = pool.inner.state.lock();
            assert_eq!(
                state.room_owners.remove(&(decoder_id.clone(), room)),
                Some(cohort.assignment_id())
            );
        }

        assert_eq!(
            pool.apply_quarantine_receipt(&mut cohort, &receipt)
                .unwrap_err(),
            DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id(),
                reason: "an assigned bootstrap room is not retained",
            }
        );
        assert_eq!(pool.snapshot().replicas[0].quiescing_cohorts, 1);
        assert_eq!(
            pool.quarantine_receipt(&cohort).unwrap(),
            Some(receipt.clone())
        );

        pool.inner
            .state
            .lock()
            .room_owners
            .insert((decoder_id, room), cohort.assignment_id());
        pool.apply_quarantine_receipt(&mut cohort, &receipt)
            .unwrap();
    }

    #[test]
    fn quarantine_rejects_undersized_accounting_without_consuming_the_cohort() {
        for field in [
            "active_cohorts",
            "active_child_requests",
            "reserved_kv_tokens",
            "remaining_decode_tokens",
        ] {
            let pool = pool(4);
            pool.register(replica("decode-0", "packed-v1")).unwrap();
            let owner = pool.begin_request(format!("request-{field}")).unwrap();
            let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
            let _quarantine = pin_quarantine_after_promotion(&pool, &mut cohort);
            let receipt = quarantine_receipt(&cohort);
            let decoder_id = cohort.decoder_id().clone();
            let baseline = pool
                .snapshot()
                .replicas
                .into_iter()
                .next()
                .expect("registered replica must be present");
            {
                let mut state = pool.inner.state.lock();
                let replica = state.replicas.get_mut(&decoder_id).unwrap();
                match field {
                    "active_cohorts" => replica.active_cohorts = 0,
                    "active_child_requests" => replica.active_child_requests = 0,
                    "reserved_kv_tokens" => replica.reserved_kv_tokens = 0,
                    "remaining_decode_tokens" => replica.remaining_decode_tokens = 0,
                    _ => unreachable!(),
                }
            }

            assert_eq!(
                pool.apply_quarantine_receipt(&mut cohort, &receipt)
                    .unwrap_err(),
                DecoderPoolError::InconsistentAssignment {
                    assignment_id: cohort.assignment_id(),
                    reason: "decoder accounting is smaller than the assignment ledger",
                }
            );
            assert_eq!(
                pool.quarantine_receipt(&cohort).unwrap(),
                Some(receipt.clone())
            );

            {
                let mut state = pool.inner.state.lock();
                let replica = state.replicas.get_mut(&decoder_id).unwrap();
                replica.active_cohorts = baseline.active_cohorts;
                replica.active_child_requests = baseline.active_child_requests;
                replica.reserved_kv_tokens = baseline.reserved_kv_tokens;
                replica.remaining_decode_tokens = baseline.remaining_decode_tokens;
            }
            pool.apply_quarantine_receipt(&mut cohort, &receipt)
                .unwrap();
        }
    }

    #[test]
    fn quarantine_rejects_missing_quiescing_accounting_without_consuming_the_cohort() {
        let pool = pool(4);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let _quarantine = pin_quarantine_after_promotion(&pool, &mut cohort);
        let receipt = quarantine_receipt(&cohort);
        let decoder_id = cohort.decoder_id().clone();
        {
            let mut state = pool.inner.state.lock();
            state
                .replicas
                .get_mut(&decoder_id)
                .unwrap()
                .quiescing_cohorts = 0;
        }

        assert_eq!(
            pool.apply_quarantine_receipt(&mut cohort, &receipt)
                .unwrap_err(),
            DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id(),
                reason: "decoder has no quiescing cohort for this assignment",
            }
        );
        assert_eq!(cohort.phase, CohortPhase::Quarantining);
        assert_eq!(
            pool.quarantine_receipt(&cohort).unwrap(),
            Some(receipt.clone())
        );

        pool.inner
            .state
            .lock()
            .replicas
            .get_mut(&decoder_id)
            .unwrap()
            .quiescing_cohorts = 1;
        pool.apply_quarantine_receipt(&mut cohort, &receipt)
            .unwrap();
    }

    #[test]
    fn quarantine_rejects_a_terminal_request_chain_without_consuming_the_cohort() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let _quarantine = pin_quarantine_after_promotion(&pool, &mut cohort);
        let receipt = quarantine_receipt(&cohort);
        {
            let mut state = pool.inner.state.lock();
            state
                .request_chains
                .get_mut(&owner.chain_id())
                .unwrap()
                .state = RequestChainState::Terminal;
        }

        assert_eq!(
            pool.apply_quarantine_receipt(&mut cohort, &receipt)
                .unwrap_err(),
            DecoderPoolError::InconsistentAssignment {
                assignment_id: cohort.assignment_id(),
                reason: "request chain does not own this assignment",
            }
        );
        assert_eq!(
            pool.quarantine_receipt(&cohort).unwrap(),
            Some(receipt.clone())
        );

        pool.inner
            .state
            .lock()
            .request_chains
            .get_mut(&owner.chain_id())
            .unwrap()
            .state = RequestChainState::Assigned(cohort.assignment_id());
        pool.apply_quarantine_receipt(&mut cohort, &receipt)
            .unwrap();
    }

    #[test]
    fn quarantine_retains_the_exact_receipt_and_every_owned_resource() {
        let pool = pool(4);
        let assigned_decoder = decoder_id("decode-0");
        pool.register(replica_with_id(assigned_decoder.clone(), "packed-v1"))
            .unwrap();
        let mut owner = pool.begin_request("request").unwrap();
        let mut grant = issue_grant(
            &pool,
            &owner,
            assigned_decoder.clone(),
            Uuid::new_v4(),
            vec![
                DecoderSlotGeneration::new(Uuid::new_v4()),
                DecoderSlotGeneration::new(Uuid::new_v4()),
            ],
            vec![41, 43],
            vec![
                DecoderGrantChildAccounting::new(100, 10),
                DecoderGrantChildAccounting::new(200, 20),
            ],
        );
        let mut cohort = bind_issued_grant(&pool, &owner, &mut grant).unwrap();
        let mut promotion = pool.begin_promotion(&mut cohort).unwrap();
        pool.observe_decode_progress(&cohort, 7).unwrap();
        let _quarantine = pool
            .pin_quarantine(&mut cohort, &mut promotion, "test_quarantine", None)
            .unwrap();
        let receipt = quarantine_receipt(&cohort);
        let assignment_id = cohort.assignment_id();

        pool.apply_quarantine_receipt(&mut cohort, &receipt)
            .unwrap();

        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_cohorts, 1);
        assert_eq!(snapshot.replicas[0].active_child_requests, 2);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 300);
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 23);
        assert_eq!(snapshot.replicas[0].quiescing_cohorts, 0);
        assert_eq!(snapshot.replicas[0].quarantined_cohorts, 1);
        assert_eq!(
            pool.quarantine_receipt(&cohort).unwrap(),
            Some(receipt.clone())
        );
        assert_eq!(
            test_selected_decoder(&pool, &owner).unwrap_err(),
            DecoderPoolError::RequestHasActiveCohort {
                request_id: "request".to_string(),
                assignment_id,
            }
        );
        assert_eq!(
            pool.finalize_request(&mut owner).unwrap_err(),
            DecoderPoolError::RequestHasActiveCohort {
                request_id: "request".to_string(),
                assignment_id,
            }
        );

        let second_owner = pool.begin_request("second").unwrap();
        let mut colliding_grant = issue_grant(
            &pool,
            &second_owner,
            assigned_decoder.clone(),
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![41],
            scalar_accounting(),
        );
        let mut colliding_pending =
            begin_test_pending_for_grant(&pool, &second_owner, &colliding_grant).unwrap();
        assert_eq!(
            pool.bind_grant(&mut colliding_pending, &mut colliding_grant)
                .unwrap_err(),
            DecoderPoolError::GrantRoomInUse {
                decoder_id: assigned_decoder.clone(),
                room: 41,
            }
        );

        pool.set_availability(&assigned_decoder, DecoderAvailability::Draining)
            .unwrap();
        assert_eq!(
            pool.remove(&assigned_decoder).unwrap_err(),
            DecoderPoolError::DecoderInUse {
                decoder_id: assigned_decoder,
                active_cohorts: 1,
                pending_admissions: 1,
            }
        );
        let release_receipt = release_receipt(&cohort, EngineReleaseKind::Aborted);
        assert_eq!(
            pool.apply_abort_release_receipt(
                &mut cohort,
                &release_receipt,
                RetryDisposition::Terminal,
            )
            .unwrap_err(),
            DecoderPoolError::InvalidTransition {
                assignment_id,
                actual: "quarantined",
                requested: "validate pinned retry disposition",
            }
        );
        assert_eq!(
            pool.apply_quarantine_receipt(&mut cohort, &receipt)
                .unwrap_err(),
            DecoderPoolError::InvalidTransition {
                assignment_id,
                actual: "quarantined",
                requested: "install quarantine proof",
            }
        );
        assert_eq!(pool.quarantine_receipt(&cohort).unwrap(), Some(receipt));
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
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        pool.set_availability(&decoder_id, DecoderAvailability::Draining)
            .unwrap();
        assert!(matches!(
            pool.remove(&decoder_id),
            Err(DecoderPoolError::DecoderInUse { .. })
        ));
        release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
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
        let mut cohort = bind_next(&pool, &owner, accounting).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].active_child_requests, 3);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 30_000);
        release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
    }

    #[test]
    fn decode_progress_updates_work_without_releasing_child_or_kv_ownership() {
        let pool = pool(2);
        pool.register(replica("decode-0", "packed-v1")).unwrap();
        let owner = pool.begin_request("request").unwrap();
        let mut cohort = bind_next(&pool, &owner, scalar_accounting()).unwrap();
        let mut retained = retain_after_promotion(&pool, &mut cohort);
        pool.observe_decode_progress(&cohort, 100).unwrap();
        let snapshot = pool.snapshot();
        assert_eq!(snapshot.replicas[0].remaining_decode_tokens, 28);
        assert_eq!(snapshot.replicas[0].reserved_kv_tokens, 1_024);
        assert_eq!(snapshot.replicas[0].active_child_requests, 1);
        assert!(matches!(
            pool.observe_decode_progress(&cohort, 29),
            Err(DecoderPoolError::InvalidProgress { .. })
        ));
        release_after_completion(&pool, &mut cohort, &mut retained).unwrap();
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

        for (mut owner, mut cohort) in admitted {
            release_before_activation(&pool, &mut cohort, RetryDisposition::Terminal).unwrap();
            pool.finalize_request(&mut owner).unwrap();
        }
    }
}
