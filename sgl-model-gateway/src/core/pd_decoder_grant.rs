//! Engine-backed decoder reservation capabilities.
//!
//! The decoder engine is the sole allocation authority. A prepared grant owns
//! an exact ordered child-allocation vector. Promotion is an asynchronous engine
//! transition that must finish before either inference request becomes pollable.
//! Terminal pool accounting is released only by an exact engine receipt.

use std::{collections::HashSet, fmt, sync::Arc};

use bytes::Bytes;
use thiserror::Error;
use tracing::warn;
use uuid::Uuid;

use super::pd_decoder_pool::{DecoderGrantPoolBinding, PendingCancellationPin};
use crate::core::{HttpOrigin, PrefillBootstrapEndpoint};

mod control;

pub use control::{
    DecoderControlAuthorization, DecoderGrantControlClient, DecoderGrantReservation,
    DecoderRequestTemplate, ReserveReconciliationGrant,
};

const GRANT_DIGEST_DOMAIN: &[u8] = b"sglang-pd-decoder-grant-v4";
const RESERVATION_DIGEST_DOMAIN: &[u8] = b"sglang-pd-decoder-reservation-v1";
const RESERVE_ATTEMPT_DIGEST_DOMAIN: &[u8] = b"sglang-pd-decoder-reserve-attempt-v1";

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProcessIdentity {
    origin: HttpOrigin,
    instance_id: Uuid,
}

impl ProcessIdentity {
    fn new(origin: HttpOrigin, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        if instance_id.is_nil() {
            return Err(ProcessIdentityError::NilInstanceId);
        }
        Ok(Self {
            origin,
            instance_id,
        })
    }

    fn url(&self) -> &str {
        self.origin.as_str()
    }

    fn origin(&self) -> &HttpOrigin {
        &self.origin
    }

    fn instance_id(&self) -> Uuid {
        self.instance_id
    }
}

/// Stable identity for one selected prefill process generation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefillId(ProcessIdentity);

impl PrefillId {
    /// Construct an identity from the selected worker URL and launch instance.
    pub fn new(origin: HttpOrigin, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        Ok(Self(ProcessIdentity::new(origin, instance_id)?))
    }

    /// Canonical selected worker base URL.
    pub fn url(&self) -> &str {
        self.0.url()
    }

    /// Canonical selected worker origin.
    pub fn origin(&self) -> &HttpOrigin {
        self.0.origin()
    }

    /// SGLang launch generation from ``PortArgs.instance_id``.
    pub fn instance_id(&self) -> Uuid {
        self.0.instance_id()
    }
}

impl fmt::Display for PrefillId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.url(), self.instance_id())
    }
}

/// Stable identity for one selected decoder process generation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DecoderId(ProcessIdentity);

impl DecoderId {
    /// Construct an identity from the selected worker URL and launch instance.
    pub fn new(origin: HttpOrigin, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        Ok(Self(ProcessIdentity::new(origin, instance_id)?))
    }

    /// Canonical selected worker base URL.
    pub fn url(&self) -> &str {
        self.0.url()
    }

    /// SGLang launch generation from ``PortArgs.instance_id``.
    /// Canonical selected worker origin.
    pub fn origin(&self) -> &HttpOrigin {
        self.0.origin()
    }

    pub fn instance_id(&self) -> Uuid {
        self.0.instance_id()
    }
}

impl fmt::Display for DecoderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.url(), self.instance_id())
    }
}

/// Invalid process-generation identity.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProcessIdentityError {
    #[error("process launch instance cannot be the nil UUID")]
    NilInstanceId,
}

/// Inference shape supported by the prepared decoder reservation protocol.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecoderInferenceRoute {
    Generate,
    ChatCompletions,
    Completions,
}

impl DecoderInferenceRoute {
    /// Exact HTTP route bound into the grant transcript.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generate => "/generate",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }
}

impl fmt::Display for DecoderInferenceRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// JSON representation of normalized child request fields.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecoderRequestShape {
    Scalar,
    Batch,
}

impl DecoderRequestShape {
    /// Canonical control-protocol representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Batch => "batch",
        }
    }
}

/// Engine allocation generation for one decoder request slot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecoderSlotGeneration(Uuid);

impl DecoderSlotGeneration {
    /// Wrap an engine-issued request-slot generation.
    pub fn new(value: Uuid) -> Self {
        Self(value)
    }

    /// Return the opaque allocation generation.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

/// Replay identity for one decoder-local request-slot allocation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DecoderAllocationKey {
    decoder_id: DecoderId,
    slot_generation: DecoderSlotGeneration,
}

impl DecoderAllocationKey {
    pub(crate) fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }
}

/// BLAKE3 digest of the exact gateway-issued reserve-attempt transcript.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecoderReserveAttemptDigest([u8; 32]);

impl DecoderReserveAttemptDigest {
    /// Return the digest bytes for reservation transcript chaining.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn to_hex(self) -> String {
        blake3::Hash::from(self.0).to_hex().to_string()
    }

    fn from_hex(value: &str) -> Result<Self, EngineGrantError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve-attempt digest must be exactly 64 lowercase hexadecimal characters"
                    .to_string(),
            ));
        }
        let digest = blake3::Hash::from_hex(value).map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "reserve-attempt digest is not 32-byte hexadecimal: {error}"
            ))
        })?;
        Ok(Self(*digest.as_bytes()))
    }
}

/// BLAKE3 digest of the exact provisional reservation transcript.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecoderReservationDigest([u8; 32]);

impl DecoderReservationDigest {
    /// Return the digest bytes for final-grant transcript chaining.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn to_hex(self) -> String {
        blake3::Hash::from(self.0).to_hex().to_string()
    }

    fn from_hex(value: &str) -> Result<Self, EngineGrantError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EngineGrantError::ProtocolViolation(
                "reservation digest must be exactly 64 lowercase hexadecimal characters"
                    .to_string(),
            ));
        }
        let digest = blake3::Hash::from_hex(value).map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "reservation digest is not 32-byte hexadecimal: {error}"
            ))
        })?;
        Ok(Self(*digest.as_bytes()))
    }
}

/// BLAKE3 digest of every authority-bearing grant field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecoderGrantDigest([u8; 32]);

impl DecoderGrantDigest {
    /// Return the digest bytes for transport-handle binding.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn to_hex(self) -> String {
        blake3::Hash::from(self.0).to_hex().to_string()
    }

    fn from_hex(value: &str) -> Result<Self, EngineGrantError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EngineGrantError::ProtocolViolation(
                "grant digest must be exactly 64 lowercase hexadecimal characters".to_string(),
            ));
        }
        let digest = blake3::Hash::from_hex(value).map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "grant digest is not 32-byte hexadecimal: {error}"
            ))
        })?;
        Ok(Self(*digest.as_bytes()))
    }
}

/// Exact PREPARED allocation identity pinned before cancellation authority moves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedGrantCancellationTarget {
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    decoder_id: DecoderId,
    kind: PreparedGrantCancellationTargetKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedGrantCancellationTargetKind {
    Unbound {
        reservation_digest: DecoderReservationDigest,
        attempted_grant_digest: Option<DecoderGrantDigest>,
    },
    Bound {
        grant_digest: DecoderGrantDigest,
    },
}

impl PreparedGrantCancellationTarget {
    fn unbound(
        binding: &UnboundGrantBinding,
        attempted_grant_digest: Option<DecoderGrantDigest>,
    ) -> Self {
        Self {
            grant_id: binding.grant_id(),
            reservation_attempt_id: binding.reservation_attempt_id(),
            reserve_attempt_digest: binding.reserve_attempt_digest(),
            decoder_id: binding.decoder_id().clone(),
            kind: PreparedGrantCancellationTargetKind::Unbound {
                reservation_digest: binding.digest(),
                attempted_grant_digest,
            },
        }
    }

    fn bound(binding: &DecoderGrantBinding) -> Self {
        Self {
            grant_id: binding.grant_id(),
            reservation_attempt_id: binding.reservation_attempt_id(),
            reserve_attempt_digest: binding.reserve_attempt_digest(),
            decoder_id: binding.decoder_id().clone(),
            kind: PreparedGrantCancellationTargetKind::Bound {
                grant_digest: binding.digest(),
            },
        }
    }

    /// Gateway-issued idempotency identity for the owning reserve attempt.
    pub(super) fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    /// Digest of the exact idempotent reserve-attempt request.
    pub(super) fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    /// Decoder process generation retaining the PREPARED allocation.
    pub(super) fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    pub(super) fn matches_unbound_receipt(
        &self,
        receipt: &PreparedGrantCancellationReceipt,
    ) -> bool {
        let PreparedGrantCancellationTargetKind::Unbound {
            reservation_digest,
            attempted_grant_digest,
        } = self.kind
        else {
            return false;
        };
        receipt.grant_id() == self.grant_id
            && receipt.reservation_attempt_id() == self.reservation_attempt_id
            && receipt.reserve_attempt_digest() == self.reserve_attempt_digest
            && receipt.decoder_id() == &self.decoder_id
            && receipt.reservation_digest() == reservation_digest
            && receipt.attempted_grant_digest() == attempted_grant_digest
            && receipt.take_once()
    }

    pub(super) fn matches_bound_receipt(&self, receipt: &EngineReleaseReceipt) -> bool {
        let PreparedGrantCancellationTargetKind::Bound { grant_digest } = self.kind else {
            return false;
        };
        receipt.grant_id() == self.grant_id
            && receipt.decoder_id() == &self.decoder_id
            && receipt.grant_digest() == grant_digest
            && receipt.kind() == EngineReleaseKind::PreparedCancelled
            && receipt.take_once()
    }
}

/// SHA-256 digest emitted by engine or transport authority.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AuthorityDigest([u8; 32]);

impl AuthorityDigest {
    /// Return the digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn from_hex(name: &str, value: &str) -> Result<Self, EngineGrantError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "{name} must contain exactly 64 lowercase hexadecimal characters"
            )));
        }
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            let offset = index * 2;
            *byte = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|error| {
                EngineGrantError::ProtocolViolation(format!(
                    "{name} is not hexadecimal at byte {index}: {error}"
                ))
            })?;
        }
        Ok(Self(bytes))
    }
}

/// Engine-derived advisory scheduling accounting for one normalized child.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderGrantChildAccounting {
    reserved_kv_tokens: usize,
    remaining_decode_tokens: usize,
}

impl DecoderGrantChildAccounting {
    /// Construct engine-derived advisory accounting.
    pub fn new(reserved_kv_tokens: usize, remaining_decode_tokens: usize) -> Self {
        Self {
            reserved_kv_tokens,
            remaining_decode_tokens,
        }
    }

    /// Conservatively reserved KV tokens.
    pub fn reserved_kv_tokens(&self) -> usize {
        self.reserved_kv_tokens
    }

    /// Remaining decode-token scheduling estimate.
    pub fn remaining_decode_tokens(&self) -> usize {
        self.remaining_decode_tokens
    }
}

/// Ordered aggregate accounting attached to one engine grant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderGrantAccounting {
    children: Arc<[DecoderGrantChildAccounting]>,
    total_reserved_kv_tokens: usize,
    total_remaining_decode_tokens: usize,
}

impl DecoderGrantAccounting {
    fn new(children: Vec<DecoderGrantChildAccounting>) -> Result<Self, EngineGrantError> {
        if children.is_empty() {
            return Err(EngineGrantError::InvalidGrant(
                "an engine grant must contain at least one child".to_string(),
            ));
        }
        let mut total_reserved_kv_tokens = 0usize;
        let mut total_remaining_decode_tokens = 0usize;
        for child in &children {
            total_reserved_kv_tokens = total_reserved_kv_tokens
                .checked_add(child.reserved_kv_tokens)
                .ok_or_else(|| {
                    EngineGrantError::InvalidGrant(
                        "grant KV-token accounting overflows usize".to_string(),
                    )
                })?;
            total_remaining_decode_tokens = total_remaining_decode_tokens
                .checked_add(child.remaining_decode_tokens)
                .ok_or_else(|| {
                    EngineGrantError::InvalidGrant(
                        "grant decode-token accounting overflows usize".to_string(),
                    )
                })?;
        }
        Ok(Self {
            children: Arc::from(children),
            total_reserved_kv_tokens,
            total_remaining_decode_tokens,
        })
    }

    /// Ordered child accounting corresponding one-to-one with grant children.
    pub fn children(&self) -> &[DecoderGrantChildAccounting] {
        &self.children
    }

    /// Number of normalized child requests.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Aggregate engine-derived KV reservation.
    pub fn total_reserved_kv_tokens(&self) -> usize {
        self.total_reserved_kv_tokens
    }

    /// Aggregate engine-derived remaining decode work.
    pub fn total_remaining_decode_tokens(&self) -> usize {
        self.total_remaining_decode_tokens
    }
}

/// Exact engine allocation bound to one normalized child request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderGrantChildBinding {
    child_request_id: Uuid,
    slot_generation: DecoderSlotGeneration,
    bootstrap_room: u64,
    request_slot: u64,
    request_generation: u64,
    writer_manifest_digest: AuthorityDigest,
    allocation_digest: AuthorityDigest,
    accounting: DecoderGrantChildAccounting,
}

impl DecoderGrantChildBinding {
    #[allow(clippy::too_many_arguments)]
    fn new(
        child_request_id: Uuid,
        slot_generation: DecoderSlotGeneration,
        bootstrap_room: u64,
        request_slot: u64,
        request_generation: u64,
        writer_manifest_digest: AuthorityDigest,
        allocation_digest: AuthorityDigest,
        accounting: DecoderGrantChildAccounting,
    ) -> Result<Self, EngineGrantError> {
        if child_request_id.is_nil() {
            return Err(EngineGrantError::InvalidGrant(
                "engine grant child request identity cannot be the nil UUID".to_string(),
            ));
        }
        Ok(Self {
            child_request_id,
            slot_generation,
            bootstrap_room,
            request_slot,
            request_generation,
            writer_manifest_digest,
            allocation_digest,
            accounting,
        })
    }

    /// Gateway-owned child request identity echoed by the engine.
    pub fn child_request_id(&self) -> Uuid {
        self.child_request_id
    }

    /// Decoder-local request-slot allocation generation.
    pub fn slot_generation(&self) -> DecoderSlotGeneration {
        self.slot_generation
    }

    /// Decoder-local bootstrap room.
    pub fn bootstrap_room(&self) -> u64 {
        self.bootstrap_room
    }

    /// Engine request-pool slot index.
    pub fn request_slot(&self) -> u64 {
        self.request_slot
    }

    /// Allocator-derived request-slot reuse generation.
    pub fn request_generation(&self) -> u64 {
        self.request_generation
    }

    /// Exact ordered writer-manifest digest.
    pub fn writer_manifest_digest(&self) -> AuthorityDigest {
        self.writer_manifest_digest
    }

    /// Exact engine allocation digest.
    pub fn allocation_digest(&self) -> AuthorityDigest {
        self.allocation_digest
    }

    /// Engine-derived advisory accounting.
    pub fn accounting(&self) -> DecoderGrantChildAccounting {
        self.accounting
    }
}

/// Immutable identity sealed inside an engine-issued reservation.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct UnboundGrantBinding {
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    base_request_body: Bytes,
    prefill_id: PrefillId,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    children: Arc<[DecoderGrantChildBinding]>,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    accounting: DecoderGrantAccounting,
    digest: DecoderReservationDigest,
}

impl fmt::Debug for UnboundGrantBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnboundGrantBinding")
            .field("grant_id", &self.grant_id)
            .field("reservation_attempt_id", &self.reservation_attempt_id)
            .field("reserve_attempt_digest", &self.reserve_attempt_digest)
            .field("inference_route", &self.inference_route)
            .field("request_shape", &self.request_shape)
            .field("prepared_ttl_ms", &self.prepared_ttl_ms)
            .field(
                "prepared_expires_at_unix_ms",
                &self.prepared_expires_at_unix_ms,
            )
            .field("base_request_body_bytes", &self.base_request_body.len())
            .field("prefill_id", &self.prefill_id)
            .field("request_chain_id", &self.request_chain_id)
            .field("source_tp_size", &self.source_tp_size)
            .field("decoder_id", &self.decoder_id)
            .field("child_count", &self.children.len())
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

impl UnboundGrantBinding {
    #[allow(clippy::too_many_arguments)]
    fn new(
        grant_id: Uuid,
        reservation_attempt_id: Uuid,
        reserve_attempt_digest: DecoderReserveAttemptDigest,
        inference_route: DecoderInferenceRoute,
        request_shape: DecoderRequestShape,
        prepared_ttl_ms: u64,
        prepared_expires_at_unix_ms: u64,
        base_request_body: Bytes,
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        request_chain_id: Uuid,
        source_tp_size: usize,
        decoder_id: DecoderId,
        children: Vec<DecoderGrantChildBinding>,
    ) -> Result<Self, EngineGrantError> {
        validate_grant_identity(grant_id, request_chain_id, source_tp_size)?;
        if reservation_attempt_id.is_nil() {
            return Err(EngineGrantError::InvalidGrant(
                "reservation attempt identity cannot be the nil UUID".to_string(),
            ));
        }
        validate_prepared_lease(prepared_ttl_ms, prepared_expires_at_unix_ms)?;
        let (slot_generations, bootstrap_rooms, accounting) = validate_child_bindings(&children)?;
        if request_shape == DecoderRequestShape::Scalar && children.len() != 1 {
            return Err(EngineGrantError::InvalidGrant(
                "a scalar engine grant must contain exactly one child".to_string(),
            ));
        }
        let digest = digest_reservation(
            grant_id,
            reserve_attempt_digest,
            prepared_expires_at_unix_ms,
            &children,
        );
        Ok(Self {
            grant_id,
            reservation_attempt_id,
            reserve_attempt_digest,
            inference_route,
            request_shape,
            prepared_ttl_ms,
            prepared_expires_at_unix_ms,
            base_request_body,
            prefill_id,
            prefill_bootstrap_endpoint,
            request_chain_id,
            source_tp_size,
            decoder_id,
            children: Arc::from(children),
            slot_generations: Arc::from(slot_generations),
            bootstrap_rooms: Arc::from(bootstrap_rooms),
            accounting,
            digest,
        })
    }

    fn bind(&self, request_body: Bytes) -> DecoderGrantBinding {
        let digest = digest_binding(self.digest, &request_body);
        DecoderGrantBinding {
            grant_id: self.grant_id,
            reservation_attempt_id: self.reservation_attempt_id,
            reserve_attempt_digest: self.reserve_attempt_digest,
            inference_route: self.inference_route,
            request_shape: self.request_shape,
            prepared_ttl_ms: self.prepared_ttl_ms,
            prepared_expires_at_unix_ms: self.prepared_expires_at_unix_ms,
            request_body,
            prefill_id: self.prefill_id.clone(),
            prefill_bootstrap_endpoint: self.prefill_bootstrap_endpoint.clone(),
            request_chain_id: self.request_chain_id,
            source_tp_size: self.source_tp_size,
            decoder_id: self.decoder_id.clone(),
            children: Arc::clone(&self.children),
            slot_generations: Arc::clone(&self.slot_generations),
            bootstrap_rooms: Arc::clone(&self.bootstrap_rooms),
            accounting: self.accounting.clone(),
            reservation_digest: self.digest,
            digest,
        }
    }

    pub(crate) fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    pub(crate) fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    pub(crate) fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    pub(crate) fn inference_route(&self) -> DecoderInferenceRoute {
        self.inference_route
    }

    pub(crate) fn request_shape(&self) -> DecoderRequestShape {
        self.request_shape
    }

    pub(crate) fn prepared_ttl_ms(&self) -> u64 {
        self.prepared_ttl_ms
    }

    pub(crate) fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.prepared_expires_at_unix_ms
    }

    pub(crate) fn base_request_body(&self) -> Bytes {
        self.base_request_body.clone()
    }

    pub(crate) fn prefill_id(&self) -> &PrefillId {
        &self.prefill_id
    }

    pub(crate) fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.prefill_bootstrap_endpoint
    }

    pub(crate) fn request_chain_id(&self) -> Uuid {
        self.request_chain_id
    }

    pub(crate) fn source_tp_size(&self) -> usize {
        self.source_tp_size
    }

    pub(crate) fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    pub(crate) fn children(&self) -> &[DecoderGrantChildBinding] {
        &self.children
    }

    pub(crate) fn child_request_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.children.iter().map(|child| child.child_request_id)
    }

    pub(crate) fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        &self.slot_generations
    }

    pub(crate) fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
    }

    pub(crate) fn digest(&self) -> DecoderReservationDigest {
        self.digest
    }
}

fn validate_grant_identity(
    grant_id: Uuid,
    request_chain_id: Uuid,
    source_tp_size: usize,
) -> Result<(), EngineGrantError> {
    if grant_id.is_nil() {
        return Err(EngineGrantError::InvalidGrant(
            "engine grant identity cannot be the nil UUID".to_string(),
        ));
    }
    if request_chain_id.is_nil() {
        return Err(EngineGrantError::InvalidGrant(
            "logical request-chain identity cannot be the nil UUID".to_string(),
        ));
    }
    if !matches!(source_tp_size, 1 | 2 | 4 | 8) {
        return Err(EngineGrantError::InvalidGrant(
            "source tensor-parallel size must be 1, 2, 4, or 8".to_string(),
        ));
    }
    Ok(())
}

fn validate_prepared_lease(
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
) -> Result<(), EngineGrantError> {
    if prepared_ttl_ms == 0 {
        return Err(EngineGrantError::InvalidGrant(
            "prepared reservation TTL must be nonzero".to_string(),
        ));
    }
    if prepared_expires_at_unix_ms == 0 {
        return Err(EngineGrantError::InvalidGrant(
            "prepared reservation expiry must be nonzero".to_string(),
        ));
    }
    Ok(())
}

fn validate_child_bindings(
    children: &[DecoderGrantChildBinding],
) -> Result<(Vec<DecoderSlotGeneration>, Vec<u64>, DecoderGrantAccounting), EngineGrantError> {
    if children.is_empty() {
        return Err(EngineGrantError::InvalidGrant(
            "an engine grant must contain at least one child".to_string(),
        ));
    }
    let request_ids: HashSet<Uuid> = children
        .iter()
        .map(DecoderGrantChildBinding::child_request_id)
        .collect();
    if request_ids.len() != children.len() {
        return Err(EngineGrantError::InvalidGrant(
            "an engine grant cannot repeat a child request identity".to_string(),
        ));
    }
    let slot_generations: Vec<DecoderSlotGeneration> =
        children.iter().map(|child| child.slot_generation).collect();
    if slot_generations
        .iter()
        .any(|generation| generation.as_uuid().is_nil())
    {
        return Err(EngineGrantError::InvalidGrant(
            "an engine grant cannot contain a nil decoder slot generation".to_string(),
        ));
    }
    let unique_slots: HashSet<DecoderSlotGeneration> = slot_generations.iter().copied().collect();
    if unique_slots.len() != children.len() {
        return Err(EngineGrantError::InvalidGrant(
            "an engine grant cannot repeat a decoder slot generation".to_string(),
        ));
    }
    let bootstrap_rooms: Vec<u64> = children
        .iter()
        .map(DecoderGrantChildBinding::bootstrap_room)
        .collect();
    let unique_rooms: HashSet<u64> = bootstrap_rooms.iter().copied().collect();
    if unique_rooms.len() != children.len() {
        return Err(EngineGrantError::InvalidGrant(
            "an engine grant cannot repeat a decoder-local bootstrap room".to_string(),
        ));
    }
    let accounting =
        DecoderGrantAccounting::new(children.iter().map(|child| child.accounting).collect())?;
    Ok((slot_generations, bootstrap_rooms, accounting))
}

/// Immutable identity sealed inside a bound engine-issued reservation.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct DecoderGrantBinding {
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    request_body: Bytes,
    prefill_id: PrefillId,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    children: Arc<[DecoderGrantChildBinding]>,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    accounting: DecoderGrantAccounting,
    reservation_digest: DecoderReservationDigest,
    digest: DecoderGrantDigest,
}

impl fmt::Debug for DecoderGrantBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecoderGrantBinding")
            .field("grant_id", &self.grant_id)
            .field("reservation_attempt_id", &self.reservation_attempt_id)
            .field("reserve_attempt_digest", &self.reserve_attempt_digest)
            .field("inference_route", &self.inference_route)
            .field("request_shape", &self.request_shape)
            .field("prepared_ttl_ms", &self.prepared_ttl_ms)
            .field(
                "prepared_expires_at_unix_ms",
                &self.prepared_expires_at_unix_ms,
            )
            .field("request_body_bytes", &self.request_body.len())
            .field("prefill_id", &self.prefill_id)
            .field("request_chain_id", &self.request_chain_id)
            .field("source_tp_size", &self.source_tp_size)
            .field("decoder_id", &self.decoder_id)
            .field("child_count", &self.children.len())
            .field("reservation_digest", &self.reservation_digest)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

impl DecoderGrantBinding {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        grant_id: Uuid,
        reservation_attempt_id: Uuid,
        inference_route: DecoderInferenceRoute,
        request_shape: DecoderRequestShape,
        prepared_ttl_ms: u64,
        prepared_expires_at_unix_ms: u64,
        base_request_body: Bytes,
        request_body: Bytes,
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        request_chain_id: Uuid,
        source_tp_size: usize,
        decoder_id: DecoderId,
        children: Vec<DecoderGrantChildBinding>,
    ) -> Result<Self, EngineGrantError> {
        let child_request_ids: Vec<Uuid> = children
            .iter()
            .map(DecoderGrantChildBinding::child_request_id)
            .collect();
        let reserve_attempt_digest = digest_reserve_attempt(
            reservation_attempt_id,
            inference_route,
            request_shape,
            prepared_ttl_ms,
            &base_request_body,
            &prefill_id,
            &prefill_bootstrap_endpoint,
            request_chain_id,
            source_tp_size,
            &decoder_id,
            &child_request_ids,
        );
        let unbound = UnboundGrantBinding::new(
            grant_id,
            reservation_attempt_id,
            reserve_attempt_digest,
            inference_route,
            request_shape,
            prepared_ttl_ms,
            prepared_expires_at_unix_ms,
            base_request_body,
            prefill_id,
            prefill_bootstrap_endpoint,
            request_chain_id,
            source_tp_size,
            decoder_id,
            children,
        )?;
        Ok(unbound.bind(request_body))
    }

    pub(crate) fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    pub(crate) fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    pub(crate) fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    pub(crate) fn inference_route(&self) -> DecoderInferenceRoute {
        self.inference_route
    }

    pub(crate) fn request_shape(&self) -> DecoderRequestShape {
        self.request_shape
    }

    pub(crate) fn prepared_ttl_ms(&self) -> u64 {
        self.prepared_ttl_ms
    }

    pub(crate) fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.prepared_expires_at_unix_ms
    }

    pub(crate) fn request_body(&self) -> Bytes {
        self.request_body.clone()
    }

    pub(crate) fn prefill_id(&self) -> &PrefillId {
        &self.prefill_id
    }

    pub(crate) fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.prefill_bootstrap_endpoint
    }

    pub(crate) fn request_chain_id(&self) -> Uuid {
        self.request_chain_id
    }

    pub(crate) fn source_tp_size(&self) -> usize {
        self.source_tp_size
    }

    pub(crate) fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    pub(crate) fn children(&self) -> &[DecoderGrantChildBinding] {
        &self.children
    }

    pub(crate) fn child_request_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.children.iter().map(|child| child.child_request_id)
    }

    pub(crate) fn allocation_keys(&self) -> impl Iterator<Item = DecoderAllocationKey> + '_ {
        self.children.iter().map(|child| DecoderAllocationKey {
            decoder_id: self.decoder_id.clone(),
            slot_generation: child.slot_generation,
        })
    }

    pub(crate) fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        &self.slot_generations
    }

    pub(crate) fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
    }

    pub(crate) fn accounting(&self) -> &DecoderGrantAccounting {
        &self.accounting
    }

    pub(crate) fn reservation_digest(&self) -> DecoderReservationDigest {
        self.reservation_digest
    }

    pub(crate) fn digest(&self) -> DecoderGrantDigest {
        self.digest
    }
}

#[allow(clippy::too_many_arguments)]
fn digest_reserve_attempt(
    reservation_attempt_id: Uuid,
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    prepared_ttl_ms: u64,
    base_request_body: &[u8],
    prefill_id: &PrefillId,
    prefill_bootstrap_endpoint: &PrefillBootstrapEndpoint,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: &DecoderId,
    child_request_ids: &[Uuid],
) -> DecoderReserveAttemptDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESERVE_ATTEMPT_DIGEST_DOMAIN);
    hasher.update(reservation_attempt_id.as_bytes());
    hash_text(&mut hasher, inference_route.as_str());
    hash_text(&mut hasher, request_shape.as_str());
    hasher.update(&prepared_ttl_ms.to_le_bytes());
    hash_bytes(&mut hasher, base_request_body);
    hash_process(&mut hasher, &prefill_id.0);
    hash_text(&mut hasher, prefill_bootstrap_endpoint.host());
    hasher.update(&u64::from(prefill_bootstrap_endpoint.port()).to_le_bytes());
    hasher.update(request_chain_id.as_bytes());
    hasher.update(&(source_tp_size as u64).to_le_bytes());
    hash_process(&mut hasher, &decoder_id.0);
    hasher.update(&(child_request_ids.len() as u64).to_le_bytes());
    for (index, child_request_id) in child_request_ids.iter().enumerate() {
        hasher.update(&(index as u64).to_le_bytes());
        hasher.update(child_request_id.as_bytes());
    }
    DecoderReserveAttemptDigest(*hasher.finalize().as_bytes())
}

fn digest_reservation(
    grant_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    prepared_expires_at_unix_ms: u64,
    children: &[DecoderGrantChildBinding],
) -> DecoderReservationDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESERVATION_DIGEST_DOMAIN);
    hasher.update(grant_id.as_bytes());
    hasher.update(reserve_attempt_digest.as_bytes());
    hasher.update(&prepared_expires_at_unix_ms.to_le_bytes());
    hasher.update(&(children.len() as u64).to_le_bytes());
    for (index, child) in children.iter().enumerate() {
        hasher.update(&(index as u64).to_le_bytes());
        hasher.update(child.child_request_id.as_bytes());
        hasher.update(child.slot_generation.0.as_bytes());
        hasher.update(&child.bootstrap_room.to_le_bytes());
        hasher.update(&child.request_slot.to_le_bytes());
        hasher.update(&child.request_generation.to_le_bytes());
        hasher.update(child.writer_manifest_digest.as_bytes());
        hasher.update(child.allocation_digest.as_bytes());
        hasher.update(&(child.accounting.reserved_kv_tokens as u64).to_le_bytes());
        hasher.update(&(child.accounting.remaining_decode_tokens as u64).to_le_bytes());
    }
    DecoderReservationDigest(*hasher.finalize().as_bytes())
}

fn digest_binding(
    reservation_digest: DecoderReservationDigest,
    request_body: &[u8],
) -> DecoderGrantDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(GRANT_DIGEST_DOMAIN);
    hasher.update(reservation_digest.as_bytes());
    hash_bytes(&mut hasher, request_body);
    DecoderGrantDigest(*hasher.finalize().as_bytes())
}

fn hash_process(hasher: &mut blake3::Hasher, process: &ProcessIdentity) {
    hash_text(hasher, process.url());
    hasher.update(process.instance_id().as_bytes());
}

fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Engine-issued PREPARED reservation whose final inference body is not yet bound.
pub struct UnboundPreparedGrant {
    binding: UnboundGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for UnboundPreparedGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnboundPreparedGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl UnboundPreparedGrant {
    fn from_control(binding: UnboundGrantBinding, control: control::PreparedGrantControl) -> Self {
        Self {
            binding,
            control: Some(control),
        }
    }

    /// Engine grant identity, also used as the eventual pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.binding.grant_id()
    }

    /// Gateway-issued idempotency identity for this reserve attempt.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.binding.reservation_attempt_id()
    }

    /// Digest of the exact idempotent reserve-attempt request.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.binding.reserve_attempt_digest()
    }

    /// Requested PREPARED lease duration in milliseconds.
    pub fn prepared_ttl_ms(&self) -> u64 {
        self.binding.prepared_ttl_ms()
    }

    /// Engine-issued absolute PREPARED expiry in Unix milliseconds.
    pub fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.binding.prepared_expires_at_unix_ms()
    }

    /// Whether normalized request fields use scalar or array JSON.
    pub fn request_shape(&self) -> DecoderRequestShape {
        self.binding.request_shape()
    }

    /// Selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Exact generation-scoped prefill endpoint consumed by decoder bootstrap.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        self.binding.prefill_bootstrap_endpoint()
    }

    /// Exact ordered normalized child allocations.
    pub fn children(&self) -> &[DecoderGrantChildBinding] {
        self.binding.children()
    }

    /// Exact ordered decoder slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        self.binding.slot_generations()
    }

    /// Exact ordered engine-issued decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        self.binding.bootstrap_rooms()
    }

    /// Exact RID-enriched request bytes supplied to provisional allocation.
    pub fn base_request_body(&self) -> Bytes {
        self.binding.base_request_body()
    }

    /// Digest of the exact provisional allocation transcript.
    pub fn reservation_digest(&self) -> DecoderReservationDigest {
        self.binding.digest()
    }

    pub(super) fn cancellation_target(
        &self,
    ) -> Result<PreparedGrantCancellationTarget, EngineGrantError> {
        if self.control.is_none() {
            return Err(EngineGrantError::ProtocolViolation(
                "unbound prepared grant has no concrete cancellation capability".to_string(),
            ));
        }
        Ok(PreparedGrantCancellationTarget::unbound(
            &self.binding,
            None,
        ))
    }

    /// Canonically enrich and pin the final inference body before any bind I/O.
    ///
    /// The returned reconciliation capability has no operation that accepts
    /// replacement bytes. Bind retry and cancellation therefore refer to the
    /// same exact transcript even after an ambiguous network outcome.
    pub fn begin_bind(&mut self) -> Result<BindReconciliationGrant, EngineGrantError> {
        let request_body = control::build_bound_request(&self.binding)?;
        let binding = self.binding.bind(request_body);
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "unbound prepared grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(BindReconciliationGrant {
            unbound_binding: self.binding.clone(),
            binding,
            control: Some(control),
        })
    }

    pub(super) fn begin_pending_cancellation(
        &mut self,
        pin: PendingCancellationPin,
    ) -> Result<UnboundCancellationReconciliationGrant, EngineGrantError> {
        let target = self.cancellation_target()?;
        if !pin.matches(&target) {
            return Err(EngineGrantError::ProtocolViolation(
                "pending cancellation pin does not match the unbound grant".to_string(),
            ));
        }
        self.take_cancellation()
    }

    fn take_cancellation(
        &mut self,
    ) -> Result<UnboundCancellationReconciliationGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "unbound prepared grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(UnboundCancellationReconciliationGrant {
            unbound_binding: self.binding.clone(),
            attempted_binding: None,
            control: Some(control),
        })
    }
}

impl Drop for UnboundPreparedGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Unbound PREPARED grant capability was dropped; engine-owned expiry remains authoritative"
        );
    }
}

/// One exact unbound PREPARED cancellation pinned before its first poll.
pub struct UnboundCancellationReconciliationGrant {
    unbound_binding: UnboundGrantBinding,
    attempted_binding: Option<DecoderGrantBinding>,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for UnboundCancellationReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnboundCancellationReconciliationGrant")
            .field("unbound_binding", &self.unbound_binding)
            .field("attempted_binding", &self.attempted_binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl UnboundCancellationReconciliationGrant {
    /// Reconcile the same cancellation without surrendering authority on error.
    pub async fn reconcile_cancellation(
        &mut self,
    ) -> Result<PreparedGrantCancellationReceipt, EngineGrantError> {
        let receipt = self
            .control()?
            .cancel_unbound(&self.unbound_binding, self.attempted_binding.as_ref())
            .await?;
        self.control = None;
        Ok(receipt)
    }

    #[cfg(test)]
    pub(crate) fn assume_test_reconciled(&mut self) -> Result<(), EngineGrantError> {
        self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test unbound cancellation has no concrete control capability".to_string(),
            )
        })?;
        Ok(())
    }

    fn control(&self) -> Result<&control::PreparedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "unbound cancellation reconciliation has no concrete control capability"
                    .to_string(),
            )
        })
    }
}

impl Drop for UnboundCancellationReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.unbound_binding.grant_id(),
            decoder_id = %self.unbound_binding.decoder_id(),
            "Unbound cancellation reconciliation capability was dropped; PREPARED engine allocation remains authoritative"
        );
    }
}

/// PREPARED capability with one exact final body pinned for idempotent binding.
pub struct BindReconciliationGrant {
    unbound_binding: UnboundGrantBinding,
    binding: DecoderGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for BindReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BindReconciliationGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl BindReconciliationGrant {
    /// Engine grant identity, also used as the eventual pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.binding.grant_id()
    }

    /// Gateway-issued idempotency identity for this reserve attempt.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.binding.reservation_attempt_id()
    }

    /// Digest of the exact idempotent reserve-attempt request.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.binding.reserve_attempt_digest()
    }

    /// Requested PREPARED lease duration in milliseconds.
    pub fn prepared_ttl_ms(&self) -> u64 {
        self.binding.prepared_ttl_ms()
    }

    /// Engine-issued absolute PREPARED expiry in Unix milliseconds.
    pub fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.binding.prepared_expires_at_unix_ms()
    }

    /// Cheap clone of the exact once-serialized body pinned before bind I/O.
    pub fn request_body(&self) -> Bytes {
        self.binding.request_body()
    }

    /// Digest covering both reserve input and the final bound transcript.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Digest of the exact provisional allocation transcript.
    pub fn reservation_digest(&self) -> DecoderReservationDigest {
        self.binding.reservation_digest()
    }

    pub(super) fn cancellation_target(
        &self,
    ) -> Result<PreparedGrantCancellationTarget, EngineGrantError> {
        if self.control.is_none() {
            return Err(EngineGrantError::ProtocolViolation(
                "bind reconciliation grant has no concrete cancellation capability".to_string(),
            ));
        }
        Ok(PreparedGrantCancellationTarget::unbound(
            &self.unbound_binding,
            Some(self.binding.digest()),
        ))
    }

    /// Bind or reconcile the same exact body without surrendering ownership on error.
    pub async fn reconcile_bind(&mut self) -> Result<BoundPreparedGrant, EngineGrantError> {
        self.control()?.bind(&self.binding).await?;
        let control = self
            .control
            .take()
            .expect("successful bind reconciliation lost its concrete control capability");
        Ok(BoundPreparedGrant::from_control(
            self.binding.clone(),
            control,
        ))
    }

    pub(super) fn begin_pending_cancellation(
        &mut self,
        pin: PendingCancellationPin,
    ) -> Result<UnboundCancellationReconciliationGrant, EngineGrantError> {
        let target = self.cancellation_target()?;
        if !pin.matches(&target) {
            return Err(EngineGrantError::ProtocolViolation(
                "pending cancellation pin does not match the attempted bind".to_string(),
            ));
        }
        self.take_cancellation()
    }

    #[cfg(test)]
    pub(crate) fn begin_cancellation(
        &mut self,
    ) -> Result<UnboundCancellationReconciliationGrant, EngineGrantError> {
        self.take_cancellation()
    }

    fn take_cancellation(
        &mut self,
    ) -> Result<UnboundCancellationReconciliationGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "bind reconciliation grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(UnboundCancellationReconciliationGrant {
            unbound_binding: self.unbound_binding.clone(),
            attempted_binding: Some(self.binding.clone()),
            control: Some(control),
        })
    }

    fn control(&self) -> Result<&control::PreparedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "bind reconciliation grant has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for BindReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Bind reconciliation capability was dropped; PREPARED engine allocation remains authoritative"
        );
    }
}

/// Concrete engine-issued PREPARED reservation bound to exact inference bytes.
pub struct BoundPreparedGrant {
    binding: DecoderGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for BoundPreparedGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundPreparedGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl BoundPreparedGrant {
    fn from_control(binding: DecoderGrantBinding, control: control::PreparedGrantControl) -> Self {
        Self {
            binding,
            control: Some(control),
        }
    }

    pub(crate) fn binding(&self) -> &DecoderGrantBinding {
        &self.binding
    }

    /// Engine grant identity, also used as the pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.binding.grant_id
    }

    /// Gateway-issued idempotency identity for this reserve attempt.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.binding.reservation_attempt_id()
    }

    /// Digest of the exact idempotent reserve-attempt request.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.binding.reserve_attempt_digest()
    }

    /// Requested PREPARED lease duration in milliseconds.
    pub fn prepared_ttl_ms(&self) -> u64 {
        self.binding.prepared_ttl_ms()
    }

    /// Engine-issued absolute PREPARED expiry in Unix milliseconds.
    pub fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.binding.prepared_expires_at_unix_ms()
    }

    /// Selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Exact generation-scoped prefill endpoint consumed by decoder bootstrap.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        self.binding.prefill_bootstrap_endpoint()
    }

    /// Exact ordered normalized child allocations.
    pub fn children(&self) -> &[DecoderGrantChildBinding] {
        self.binding.children()
    }

    /// Exact ordered decoder slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        self.binding.slot_generations()
    }

    /// Exact ordered decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        self.binding.bootstrap_rooms()
    }

    /// Cheap clone of the exact once-serialized body both inference sends reuse.
    pub fn request_body(&self) -> Bytes {
        self.binding.request_body()
    }

    /// Digest covering the request and every ordered child allocation.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Digest of the exact provisional allocation transcript.
    pub fn reservation_digest(&self) -> DecoderReservationDigest {
        self.binding.reservation_digest()
    }

    pub(super) fn cancellation_target(
        &self,
    ) -> Result<PreparedGrantCancellationTarget, EngineGrantError> {
        if self.control.is_none() {
            return Err(EngineGrantError::ProtocolViolation(
                "prepared grant has no concrete cancellation capability".to_string(),
            ));
        }
        Ok(PreparedGrantCancellationTarget::bound(&self.binding))
    }

    pub(super) fn take_for_pool_binding(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
    ) -> Result<Self, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool binding does not match the exact prepared grant".to_string(),
            ));
        }
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "pool binding has no concrete prepared control capability".to_string(),
            )
        })?;
        Ok(Self::from_control(self.binding.clone(), control))
    }

    pub(super) fn begin_pending_cancellation(
        &mut self,
        pin: PendingCancellationPin,
    ) -> Result<PreparedCancellationReconciliationGrant, EngineGrantError> {
        let target = self.cancellation_target()?;
        if !pin.matches(&target) {
            return Err(EngineGrantError::ProtocolViolation(
                "pending cancellation pin does not match the bound grant".to_string(),
            ));
        }
        self.take_cancellation()
    }

    pub(super) fn begin_pool_cancellation(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
    ) -> Result<PreparedCancellationReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool cancellation binding does not match the exact prepared grant"
                    .to_string(),
            ));
        }
        self.take_cancellation()
    }

    #[cfg(test)]
    pub(crate) fn begin_cancellation(
        &mut self,
    ) -> Result<PreparedCancellationReconciliationGrant, EngineGrantError> {
        self.take_cancellation()
    }

    fn take_cancellation(
        &mut self,
    ) -> Result<PreparedCancellationReconciliationGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "prepared cancellation has no concrete control capability".to_string(),
            )
        })?;
        Ok(PreparedCancellationReconciliationGrant {
            binding: self.binding.clone(),
            control: Some(control),
        })
    }

    pub(super) fn begin_promotion(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
    ) -> Result<PromotionReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool promotion binding does not match the exact prepared grant"
                    .to_string(),
            ));
        }
        self.take_for_promotion()
    }

    fn take_for_promotion(&mut self) -> Result<PromotionReconciliationGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "promotion has no concrete prepared control capability".to_string(),
            )
        })?;
        Ok(PromotionReconciliationGrant {
            binding: self.binding.clone(),
            control: Some(control),
        })
    }

    #[cfg(test)]
    pub(crate) fn begin_test_promotion(
        &mut self,
    ) -> Result<PromotionReconciliationGrant, EngineGrantError> {
        self.take_for_promotion()
    }
}

impl Drop for BoundPreparedGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Prepared decoder grant capability was dropped; engine-owned expiry remains authoritative"
        );
    }
}

/// One exact bound PREPARED cancellation pinned before its first poll.
pub struct PreparedCancellationReconciliationGrant {
    binding: DecoderGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for PreparedCancellationReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCancellationReconciliationGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl PreparedCancellationReconciliationGrant {
    /// Reconcile the same prepared cancellation without changing operation.
    pub async fn reconcile_cancellation(
        &mut self,
    ) -> Result<EngineReleaseReceipt, EngineGrantError> {
        let receipt = self.control()?.cancel(&self.binding).await?;
        self.control = None;
        Ok(receipt)
    }

    #[cfg(test)]
    pub(crate) fn assume_test_reconciled(&mut self) -> Result<(), EngineGrantError> {
        self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test prepared cancellation has no concrete control capability".to_string(),
            )
        })?;
        Ok(())
    }

    fn control(&self) -> Result<&control::PreparedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "prepared cancellation reconciliation has no concrete control capability"
                    .to_string(),
            )
        })
    }
}

impl Drop for PreparedCancellationReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Prepared cancellation reconciliation capability was dropped; PREPARED engine allocation remains authoritative"
        );
    }
}

/// Capability for a grant whose promotion may already have reached the engine.
///
/// This type intentionally has no prepared-cancel operation. Promotion retry,
/// abort, and quarantine all preserve the concrete capability until the engine
/// returns a validated authoritative receipt.
pub struct PromotionReconciliationGrant {
    binding: DecoderGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for PromotionReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PromotionReconciliationGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl PromotionReconciliationGrant {
    /// Engine grant identity, also used as the pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.binding.grant_id
    }

    /// Selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Exact generation-scoped prefill endpoint consumed by decoder bootstrap.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        self.binding.prefill_bootstrap_endpoint()
    }

    /// Exact ordered decoder slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        self.binding.slot_generations()
    }

    /// Exact ordered decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        self.binding.bootstrap_rooms()
    }

    /// Cheap clone of the exact once-serialized body both inference sends reuse.
    pub fn request_body(&self) -> Bytes {
        self.binding.request_body()
    }

    /// Digest covering the request and every ordered child allocation.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Retry promotion reconciliation without surrendering ownership on error.
    pub async fn reconcile_promotion(&mut self) -> Result<RetainedEngineGrant, EngineGrantError> {
        self.control()?.promote(&self.binding).await?;
        let control = self
            .control
            .take()
            .expect("promotion reconciliation lost its concrete control capability");
        Ok(RetainedEngineGrant {
            binding: self.binding.clone(),
            control: Some(control.into_retained()),
        })
    }

    #[cfg(test)]
    pub(crate) fn assume_test_promoted(&mut self) -> Result<RetainedEngineGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test promotion has no concrete prepared control capability".to_string(),
            )
        })?;
        Ok(RetainedEngineGrant {
            binding: self.binding.clone(),
            control: Some(control.into_retained()),
        })
    }

    /// Pin an abort of a potentially promoted grant.
    pub(super) fn begin_abort(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool abort binding does not match the exact promotion grant".to_string(),
            ));
        }
        self.take_for_abort(reason_code, diagnostic)
    }

    fn take_for_abort(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        let context = FailureContext::new(reason_code, diagnostic)?;
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "promotion reconciliation grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(AbortReconciliationGrant {
            binding: self.binding.clone(),
            context,
            control: Some(TerminalGrantControl::Prepared(control)),
        })
    }

    #[cfg(test)]
    pub(crate) fn begin_test_abort(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        self.take_for_abort(reason_code, diagnostic)
    }

    /// Pin quarantine of a potentially promoted grant.
    pub(super) fn begin_quarantine(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool quarantine binding does not match the exact promotion grant"
                    .to_string(),
            ));
        }
        self.take_for_quarantine(reason_code, diagnostic)
    }

    fn take_for_quarantine(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        let context = FailureContext::new(reason_code, diagnostic)?;
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "promotion reconciliation grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(QuarantineReconciliationGrant {
            binding: self.binding.clone(),
            context,
            control: Some(TerminalGrantControl::Prepared(control)),
        })
    }

    fn control(&self) -> Result<&control::PreparedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "promotion reconciliation grant has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for PromotionReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Promotion reconciliation capability was dropped; engine ownership remains retained"
        );
    }
}

/// Post-promotion engine reservation retained through response-body lifetime.
pub struct RetainedEngineGrant {
    binding: DecoderGrantBinding,
    control: Option<control::RetainedGrantControl>,
}

impl fmt::Debug for RetainedEngineGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedEngineGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl RetainedEngineGrant {
    /// Engine grant identity, also used as the pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.binding.grant_id
    }

    /// Selected decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        self.binding.decoder_id()
    }

    /// Exact generation-scoped prefill endpoint consumed by decoder bootstrap.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        self.binding.prefill_bootstrap_endpoint()
    }

    /// Exact ordered decoder slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        self.binding.slot_generations()
    }

    /// Exact ordered decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        self.binding.bootstrap_rooms()
    }

    /// Cheap clone of the exact once-serialized body both inference sends reuse.
    pub fn request_body(&self) -> Bytes {
        self.binding.request_body()
    }

    /// Digest covering the request and every ordered child allocation.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Pin successful completion before polling its terminal receipt.
    pub(super) fn begin_completion(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
    ) -> Result<CompletionReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool completion binding does not match the exact retained grant"
                    .to_string(),
            ));
        }
        self.take_for_completion()
    }

    fn take_for_completion(&mut self) -> Result<CompletionReconciliationGrant, EngineGrantError> {
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "retained engine grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(CompletionReconciliationGrant {
            binding: self.binding.clone(),
            control: Some(control),
        })
    }

    #[cfg(test)]
    pub(crate) fn begin_test_completion(
        &mut self,
    ) -> Result<CompletionReconciliationGrant, EngineGrantError> {
        self.take_for_completion()
    }

    /// Pin an abort of every retained child.
    ///
    /// The engine releases its allocations only when it can prove an exact
    /// all-child no-submit or terminal outcome. Otherwise the grant remains
    /// monotonically quarantined.
    pub(super) fn begin_abort(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool abort binding does not match the exact retained grant".to_string(),
            ));
        }
        self.take_for_abort(reason_code, diagnostic)
    }

    fn take_for_abort(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        let context = FailureContext::new(reason_code, diagnostic)?;
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "retained engine grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(AbortReconciliationGrant {
            binding: self.binding.clone(),
            context,
            control: Some(TerminalGrantControl::Retained(control)),
        })
    }

    #[cfg(test)]
    pub(crate) fn begin_test_abort(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<AbortReconciliationGrant, EngineGrantError> {
        self.take_for_abort(reason_code, diagnostic)
    }

    /// Pin quarantine of an ambiguous retained reservation.
    pub(super) fn begin_quarantine(
        &mut self,
        pool_binding: DecoderGrantPoolBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        if !pool_binding.matches(&self.binding) {
            return Err(EngineGrantError::ProtocolViolation(
                "decoder-pool quarantine binding does not match the exact retained grant"
                    .to_string(),
            ));
        }
        self.take_for_quarantine(reason_code, diagnostic)
    }

    fn take_for_quarantine(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        let context = FailureContext::new(reason_code, diagnostic)?;
        let control = self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "retained engine grant has no concrete control capability".to_string(),
            )
        })?;
        Ok(QuarantineReconciliationGrant {
            binding: self.binding.clone(),
            context,
            control: Some(TerminalGrantControl::Retained(control)),
        })
    }

    #[cfg(test)]
    pub(crate) fn begin_test_quarantine(
        &mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<QuarantineReconciliationGrant, EngineGrantError> {
        self.take_for_quarantine(reason_code, diagnostic)
    }
}

impl Drop for RetainedEngineGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Retained decoder grant capability was dropped; engine ownership remains retained"
        );
    }
}

struct FailureContext {
    reason_code: Arc<str>,
    diagnostic: Option<Arc<str>>,
}

impl FailureContext {
    fn new(reason_code: &str, diagnostic: Option<&str>) -> Result<Self, EngineGrantError> {
        control::validate_failure_context(reason_code, diagnostic)?;
        Ok(Self {
            reason_code: Arc::from(reason_code),
            diagnostic: diagnostic.map(Arc::from),
        })
    }
}

impl fmt::Debug for FailureContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FailureContext")
            .field("reason_code", &self.reason_code)
            .field("has_diagnostic", &self.diagnostic.is_some())
            .finish()
    }
}

enum TerminalGrantControl {
    Prepared(control::PreparedGrantControl),
    Retained(control::RetainedGrantControl),
}

impl TerminalGrantControl {
    async fn abort(
        &self,
        binding: &DecoderGrantBinding,
        context: &FailureContext,
    ) -> Result<EngineAbortOutcome, EngineGrantError> {
        match self {
            Self::Prepared(control) => {
                control
                    .abort(binding, &context.reason_code, context.diagnostic.as_deref())
                    .await
            }
            Self::Retained(control) => {
                control
                    .abort(binding, &context.reason_code, context.diagnostic.as_deref())
                    .await
            }
        }
    }

    async fn quarantine(
        &self,
        binding: &DecoderGrantBinding,
        context: &FailureContext,
    ) -> Result<EngineQuarantineReceipt, EngineGrantError> {
        match self {
            Self::Prepared(control) => {
                control
                    .quarantine(binding, &context.reason_code, context.diagnostic.as_deref())
                    .await
            }
            Self::Retained(control) => {
                control
                    .quarantine(binding, &context.reason_code, context.diagnostic.as_deref())
                    .await
            }
        }
    }
}

/// One exact completion pinned before its first control poll.
pub struct CompletionReconciliationGrant {
    binding: DecoderGrantBinding,
    control: Option<control::RetainedGrantControl>,
}

impl fmt::Debug for CompletionReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompletionReconciliationGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl CompletionReconciliationGrant {
    /// Reconcile only the pinned completion operation.
    pub async fn reconcile_completion(
        &mut self,
    ) -> Result<EngineCompletionOutcome, EngineGrantError> {
        let outcome = self.control()?.complete(&self.binding).await?;
        self.control = None;
        Ok(outcome)
    }

    #[cfg(test)]
    pub(crate) fn assume_test_reconciled(&mut self) -> Result<(), EngineGrantError> {
        self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test completion has no concrete control capability".to_string(),
            )
        })?;
        Ok(())
    }

    fn control(&self) -> Result<&control::RetainedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "completion reconciliation has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for CompletionReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Completion reconciliation capability was dropped; engine ownership remains retained"
        );
    }
}

/// One exact abort and failure context pinned before its first control poll.
pub struct AbortReconciliationGrant {
    binding: DecoderGrantBinding,
    context: FailureContext,
    control: Option<TerminalGrantControl>,
}

impl fmt::Debug for AbortReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AbortReconciliationGrant")
            .field("binding", &self.binding)
            .field("context", &self.context)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl AbortReconciliationGrant {
    /// Reconcile only the pinned abort operation and exact failure context.
    pub async fn reconcile_abort(&mut self) -> Result<EngineAbortOutcome, EngineGrantError> {
        let outcome = self.control()?.abort(&self.binding, &self.context).await?;
        self.control = None;
        Ok(outcome)
    }

    #[cfg(test)]
    pub(crate) fn assume_test_reconciled(&mut self) -> Result<(), EngineGrantError> {
        self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test abort has no concrete control capability".to_string(),
            )
        })?;
        Ok(())
    }

    fn control(&self) -> Result<&TerminalGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "abort reconciliation has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for AbortReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Abort reconciliation capability was dropped; engine ownership remains retained"
        );
    }
}

/// One exact quarantine and failure context pinned before its first control poll.
pub struct QuarantineReconciliationGrant {
    binding: DecoderGrantBinding,
    context: FailureContext,
    control: Option<TerminalGrantControl>,
}

impl fmt::Debug for QuarantineReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QuarantineReconciliationGrant")
            .field("binding", &self.binding)
            .field("context", &self.context)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl QuarantineReconciliationGrant {
    /// Reconcile only the pinned quarantine operation and exact failure context.
    pub async fn reconcile_quarantine(
        &mut self,
    ) -> Result<EngineQuarantineReceipt, EngineGrantError> {
        let receipt = self
            .control()?
            .quarantine(&self.binding, &self.context)
            .await?;
        self.control = None;
        Ok(receipt)
    }

    #[cfg(test)]
    pub(crate) fn assume_test_reconciled(&mut self) -> Result<(), EngineGrantError> {
        self.control.take().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "test quarantine has no concrete control capability".to_string(),
            )
        })?;
        Ok(())
    }

    fn control(&self) -> Result<&TerminalGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "quarantine reconciliation has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for QuarantineReconciliationGrant {
    fn drop(&mut self) {
        if self.control.is_none() {
            return;
        }
        warn!(
            grant_id = %self.binding.grant_id(),
            decoder_id = %self.binding.decoder_id(),
            "Quarantine reconciliation capability was dropped; engine ownership remains retained"
        );
    }
}

/// Engine receipt proving release of an unbound PREPARED allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedGrantCancellationReceipt {
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    decoder_id: DecoderId,
    child_request_ids: Arc<[Uuid]>,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    reservation_digest: DecoderReservationDigest,
    attempted_grant_digest: Option<DecoderGrantDigest>,
    receipt_id: Uuid,
    receipt_digest: AuthorityDigest,
    take_once: bool,
}

impl PreparedGrantCancellationReceipt {
    #[allow(clippy::too_many_arguments)]
    fn from_control(
        grant_id: Uuid,
        reservation_attempt_id: Uuid,
        reserve_attempt_digest: DecoderReserveAttemptDigest,
        decoder_id: DecoderId,
        child_request_ids: Vec<Uuid>,
        slot_generations: Vec<DecoderSlotGeneration>,
        bootstrap_rooms: Vec<u64>,
        prepared_ttl_ms: u64,
        prepared_expires_at_unix_ms: u64,
        reservation_digest: DecoderReservationDigest,
        attempted_grant_digest: Option<DecoderGrantDigest>,
        receipt_id: Uuid,
        receipt_digest: AuthorityDigest,
        take_once: bool,
    ) -> Self {
        Self {
            grant_id,
            reservation_attempt_id,
            reserve_attempt_digest,
            decoder_id,
            child_request_ids: Arc::from(child_request_ids),
            slot_generations: Arc::from(slot_generations),
            bootstrap_rooms: Arc::from(bootstrap_rooms),
            prepared_ttl_ms,
            prepared_expires_at_unix_ms,
            reservation_digest,
            attempted_grant_digest,
            receipt_id,
            receipt_digest,
            take_once,
        }
    }

    /// Engine-issued reservation identity.
    pub fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    /// Gateway-issued idempotency identity whose allocation was released.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    /// Exact idempotent reserve-attempt transcript released by the engine.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    /// Decoder generation whose provisional allocations were released.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Exact ordered gateway-owned child identities.
    pub fn child_request_ids(&self) -> &[Uuid] {
        &self.child_request_ids
    }

    /// Exact ordered decoder request-slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        &self.slot_generations
    }

    /// Exact ordered decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
    }

    /// Requested PREPARED lease duration in milliseconds.
    pub fn prepared_ttl_ms(&self) -> u64 {
        self.prepared_ttl_ms
    }

    /// Engine-issued absolute PREPARED expiry in Unix milliseconds.
    pub fn prepared_expires_at_unix_ms(&self) -> u64 {
        self.prepared_expires_at_unix_ms
    }

    /// Exact provisional reservation transcript released by the engine.
    pub fn reservation_digest(&self) -> DecoderReservationDigest {
        self.reservation_digest
    }

    /// Final digest pinned by a bind attempt, if binding was attempted.
    pub fn attempted_grant_digest(&self) -> Option<DecoderGrantDigest> {
        self.attempted_grant_digest
    }

    /// Immutable engine receipt identity.
    pub fn receipt_id(&self) -> Uuid {
        self.receipt_id
    }

    /// Immutable engine receipt digest.
    pub fn receipt_digest(&self) -> AuthorityDigest {
        self.receipt_digest
    }

    /// Whether engine reconciliation is take-once and retry-observable.
    pub fn take_once(&self) -> bool {
        self.take_once
    }
}

/// Authoritative terminal release kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineReleaseKind {
    PreparedCancelled,
    Completed,
    Aborted,
}

/// Engine-authoritative result of a completion request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineCompletionOutcome {
    /// Every child completed and its allocation was released.
    Completed(EngineReleaseReceipt),
    /// Completion could not prove safe release, so every allocation remains retained.
    Quarantined(EngineQuarantineReceipt),
}

/// Engine-authoritative result of an abort request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineAbortOutcome {
    /// Every child is terminal and its allocation was released.
    Aborted(EngineReleaseReceipt),
    /// Exact quiescence was not provable, so every allocation remains retained.
    Quarantined(EngineQuarantineReceipt),
}

/// Engine receipt proving that an ambiguous grant remains monotonically held.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineQuarantineReceipt {
    grant_id: Uuid,
    decoder_id: DecoderId,
    child_request_ids: Arc<[Uuid]>,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    grant_digest: DecoderGrantDigest,
    receipt_id: Uuid,
    receipt_digest: AuthorityDigest,
    take_once: bool,
}

impl EngineQuarantineReceipt {
    #[allow(clippy::too_many_arguments)]
    fn from_control(
        grant_id: Uuid,
        decoder_id: DecoderId,
        child_request_ids: Vec<Uuid>,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        slot_generations: Vec<DecoderSlotGeneration>,
        bootstrap_rooms: Vec<u64>,
        grant_digest: DecoderGrantDigest,
        receipt_id: Uuid,
        receipt_digest: AuthorityDigest,
        take_once: bool,
    ) -> Self {
        Self {
            grant_id,
            decoder_id,
            child_request_ids: Arc::from(child_request_ids),
            prefill_bootstrap_endpoint,
            slot_generations: Arc::from(slot_generations),
            bootstrap_rooms: Arc::from(bootstrap_rooms),
            grant_digest,
            receipt_id,
            receipt_digest,
            take_once,
        }
    }

    /// Engine grant and pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    /// Decoder generation retaining the allocations.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Exact ordered gateway-owned child identities.
    pub fn child_request_ids(&self) -> &[Uuid] {
        &self.child_request_ids
    }

    /// Exact prefill bootstrap endpoint bound into the grant.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.prefill_bootstrap_endpoint
    }

    /// Exact ordered decoder request-slot generations retained by the engine.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        &self.slot_generations
    }

    /// Exact ordered decoder-local bootstrap rooms retained by the engine.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
    }

    /// Exact grant digest echoed by the engine.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.grant_digest
    }

    /// Immutable engine receipt identity.
    pub fn receipt_id(&self) -> Uuid {
        self.receipt_id
    }

    /// Immutable engine receipt digest.
    pub fn receipt_digest(&self) -> AuthorityDigest {
        self.receipt_digest
    }

    /// Whether reconciliation is take-once and retry-observable.
    pub fn take_once(&self) -> bool {
        self.take_once
    }
}

/// Engine-returned release receipt for one exact batch grant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineReleaseReceipt {
    grant_id: Uuid,
    decoder_id: DecoderId,
    child_request_ids: Arc<[Uuid]>,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    grant_digest: DecoderGrantDigest,
    kind: EngineReleaseKind,
    receipt_id: Uuid,
    receipt_digest: AuthorityDigest,
    take_once: bool,
}

impl EngineReleaseReceipt {
    #[allow(clippy::too_many_arguments)]
    fn from_control(
        grant_id: Uuid,
        decoder_id: DecoderId,
        child_request_ids: Vec<Uuid>,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        slot_generations: Vec<DecoderSlotGeneration>,
        bootstrap_rooms: Vec<u64>,
        grant_digest: DecoderGrantDigest,
        kind: EngineReleaseKind,
        receipt_id: Uuid,
        receipt_digest: AuthorityDigest,
        take_once: bool,
    ) -> Self {
        Self {
            grant_id,
            decoder_id,
            child_request_ids: Arc::from(child_request_ids),
            prefill_bootstrap_endpoint,
            slot_generations: Arc::from(slot_generations),
            bootstrap_rooms: Arc::from(bootstrap_rooms),
            grant_digest,
            kind,
            receipt_id,
            receipt_digest,
            take_once,
        }
    }

    /// Engine grant and pool assignment identity.
    pub fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    /// Decoder process generation whose allocations were released.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Exact ordered gateway-owned child identities.
    pub fn child_request_ids(&self) -> &[Uuid] {
        &self.child_request_ids
    }

    /// Exact prefill bootstrap endpoint bound into the grant.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.prefill_bootstrap_endpoint
    }

    /// Exact ordered decoder request-slot generations.
    pub fn slot_generations(&self) -> &[DecoderSlotGeneration] {
        &self.slot_generations
    }

    /// Exact ordered decoder-local bootstrap rooms.
    pub fn bootstrap_rooms(&self) -> &[u64] {
        &self.bootstrap_rooms
    }

    /// Exact grant digest echoed by the engine.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.grant_digest
    }

    /// Whether release followed cancellation, completion, or abort.
    pub fn kind(&self) -> EngineReleaseKind {
        self.kind
    }

    /// Immutable engine receipt identity.
    pub fn receipt_id(&self) -> Uuid {
        self.receipt_id
    }

    /// Immutable engine receipt digest.
    pub fn receipt_digest(&self) -> AuthorityDigest {
        self.receipt_digest
    }

    /// Whether engine reconciliation is take-once and retry-observable.
    pub fn take_once(&self) -> bool {
        self.take_once
    }
}

/// Authoritative retry policy attached to one refused reserve attempt.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecoderReserveRefusalDisposition {
    /// Retry a fresh exact attempt against this decoder generation.
    RetrySameDecoder,
    /// Preserve the tombstone and select a different decoder generation.
    RetryAnotherDecoder,
    /// Finish admission without another decoder reservation attempt.
    Terminal,
}

impl DecoderReserveRefusalDisposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::RetrySameDecoder => "retry_same_decoder",
            Self::RetryAnotherDecoder => "retry_another_decoder",
            Self::Terminal => "terminal",
        }
    }
}

impl fmt::Display for DecoderReserveRefusalDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Engine tombstone proving that one exact reserve attempt allocated nothing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderReserveRefusalReceipt {
    prefill_id: PrefillId,
    decoder_id: DecoderId,
    logical_request_chain_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    reason_code: String,
    disposition: DecoderReserveRefusalDisposition,
    receipt_id: Uuid,
    receipt_digest: AuthorityDigest,
    take_once: bool,
}

impl fmt::Display for DecoderReserveRefusalReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({})", self.reason_code, self.disposition)
    }
}

impl DecoderReserveRefusalReceipt {
    /// Exact selected prefill process generation.
    pub fn prefill_id(&self) -> &PrefillId {
        &self.prefill_id
    }

    /// Logical request chain whose pending admission was refused.
    pub fn logical_request_chain_id(&self) -> Uuid {
        self.logical_request_chain_id
    }

    /// Gateway-issued identity of the tombstoned reserve attempt.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    /// Digest of the exact reserve transcript that allocated nothing.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
    }

    /// Exact decoder process generation that refused the attempt.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// Bounded machine-readable allocator reason code.
    pub fn reason_code(&self) -> &str {
        &self.reason_code
    }

    /// Authoritative retry policy, independent of status and reason text.
    pub fn disposition(&self) -> DecoderReserveRefusalDisposition {
        self.disposition
    }

    /// Immutable engine receipt identity.
    pub fn receipt_id(&self) -> Uuid {
        self.receipt_id
    }

    /// Immutable engine receipt digest.
    pub fn receipt_digest(&self) -> AuthorityDigest {
        self.receipt_digest
    }

    /// Whether refusal reconciliation is take-once and retry-observable.
    pub fn take_once(&self) -> bool {
        self.take_once
    }
}

/// Concrete decoder reservation and lifecycle failures.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum EngineGrantError {
    #[error(transparent)]
    InvalidProcessIdentity(#[from] ProcessIdentityError),
    #[error("invalid engine decoder grant: {0}")]
    InvalidGrant(String),
    #[error("decoder allocator refused the reservation: {0}")]
    AllocatorRefused(Box<DecoderReserveRefusalReceipt>),
    #[error("decoder reserve outcome is allocation-ambiguous: {0}")]
    AmbiguousReserve(String),
    #[error("decoder control outcome is ambiguous during {operation}: {message}")]
    AmbiguousControl {
        operation: &'static str,
        message: String,
    },
    #[error("decoder control protocol violation: {0}")]
    ProtocolViolation(String),
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn issue_test_grant(
    prefill_id: PrefillId,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    grant_id: Uuid,
    slot_generations: Vec<DecoderSlotGeneration>,
    bootstrap_rooms: Vec<u64>,
    accounting: Vec<DecoderGrantChildAccounting>,
) -> Result<BoundPreparedGrant, EngineGrantError> {
    issue_test_grant_with_control_url(
        prefill_id,
        request_chain_id,
        source_tp_size,
        decoder_id,
        grant_id,
        slot_generations,
        bootstrap_rooms,
        accounting,
        None,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn issue_test_grant_at_control_url(
    prefill_id: PrefillId,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    grant_id: Uuid,
    slot_generations: Vec<DecoderSlotGeneration>,
    bootstrap_rooms: Vec<u64>,
    accounting: Vec<DecoderGrantChildAccounting>,
    control_url: &str,
) -> Result<BoundPreparedGrant, EngineGrantError> {
    issue_test_grant_with_control_url(
        prefill_id,
        request_chain_id,
        source_tp_size,
        decoder_id,
        grant_id,
        slot_generations,
        bootstrap_rooms,
        accounting,
        Some(control_url),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn issue_test_grant_with_control_url(
    prefill_id: PrefillId,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    grant_id: Uuid,
    slot_generations: Vec<DecoderSlotGeneration>,
    bootstrap_rooms: Vec<u64>,
    accounting: Vec<DecoderGrantChildAccounting>,
    control_url: Option<&str>,
) -> Result<BoundPreparedGrant, EngineGrantError> {
    if slot_generations.len() != bootstrap_rooms.len() || slot_generations.len() != accounting.len()
    {
        return Err(EngineGrantError::InvalidGrant(
            "test grant vectors must have identical lengths".to_string(),
        ));
    }
    let request_shape = if slot_generations.len() == 1 {
        DecoderRequestShape::Scalar
    } else {
        DecoderRequestShape::Batch
    };
    let children = slot_generations
        .into_iter()
        .zip(bootstrap_rooms)
        .zip(accounting)
        .enumerate()
        .map(|(index, ((slot_generation, bootstrap_room), accounting))| {
            let digest_byte = u8::try_from(index % 255).expect("bounded digest test byte");
            DecoderGrantChildBinding::new(
                Uuid::from_u128(index as u128 + 1),
                slot_generation,
                bootstrap_room,
                index as u64,
                1,
                AuthorityDigest([digest_byte; 32]),
                AuthorityDigest([digest_byte.wrapping_add(1); 32]),
                accounting,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let binding = DecoderGrantBinding::new(
        grant_id,
        Uuid::from_u128(0xaaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa),
        DecoderInferenceRoute::Generate,
        request_shape,
        1_000,
        1_900_000_000_000,
        Bytes::from_static(b"{}"),
        Bytes::from_static(b"{}"),
        prefill_id,
        PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000)
            .map_err(|error| EngineGrantError::InvalidGrant(error.to_string()))?,
        request_chain_id,
        source_tp_size,
        decoder_id,
        children,
    )?;
    let control = match control_url {
        Some(control_url) => {
            control::test_prepared_grant_control_at(binding.grant_id(), control_url)
        }
        None => control::test_prepared_grant_control(binding.grant_id()),
    };
    Ok(BoundPreparedGrant {
        binding,
        control: Some(control),
    })
}

#[cfg(test)]
pub(super) fn issue_test_unbound_cancellation_target(
    grant: &BoundPreparedGrant,
    attempted_grant_digest: Option<DecoderGrantDigest>,
) -> PreparedGrantCancellationTarget {
    PreparedGrantCancellationTarget {
        grant_id: grant.binding.grant_id(),
        reservation_attempt_id: grant.binding.reservation_attempt_id(),
        reserve_attempt_digest: grant.binding.reserve_attempt_digest(),
        decoder_id: grant.binding.decoder_id().clone(),
        kind: PreparedGrantCancellationTargetKind::Unbound {
            reservation_digest: grant.binding.reservation_digest(),
            attempted_grant_digest,
        },
    }
}

#[cfg(test)]
pub(super) fn issue_test_prepared_cancellation_receipt(
    target: &PreparedGrantCancellationTarget,
    take_once: bool,
) -> PreparedGrantCancellationReceipt {
    let PreparedGrantCancellationTargetKind::Unbound {
        reservation_digest,
        attempted_grant_digest,
    } = target.kind
    else {
        panic!("test unbound cancellation receipt requires an unbound target");
    };
    PreparedGrantCancellationReceipt::from_control(
        target.grant_id,
        target.reservation_attempt_id,
        target.reserve_attempt_digest,
        target.decoder_id.clone(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        1_000,
        1_900_000_000_000,
        reservation_digest,
        attempted_grant_digest,
        Uuid::new_v4(),
        AuthorityDigest([9; 32]),
        take_once,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn issue_test_reserve_refusal_receipt(
    prefill_id: PrefillId,
    decoder_id: DecoderId,
    logical_request_chain_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    disposition: DecoderReserveRefusalDisposition,
    take_once: bool,
) -> DecoderReserveRefusalReceipt {
    DecoderReserveRefusalReceipt {
        prefill_id,
        decoder_id,
        logical_request_chain_id,
        reservation_attempt_id,
        reserve_attempt_digest,
        reason_code: "test_refusal".to_string(),
        disposition,
        receipt_id: Uuid::new_v4(),
        receipt_digest: AuthorityDigest([10; 32]),
        take_once,
    }
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct TestEngineReceiptBinding {
    pub(crate) grant_id: Uuid,
    pub(crate) decoder_id: DecoderId,
    pub(crate) child_request_ids: Vec<Uuid>,
    pub(crate) prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    pub(crate) slot_generations: Vec<DecoderSlotGeneration>,
    pub(crate) bootstrap_rooms: Vec<u64>,
    pub(crate) grant_digest: DecoderGrantDigest,
}

#[cfg(test)]
pub(crate) fn issue_test_release_receipt(
    binding: TestEngineReceiptBinding,
    kind: EngineReleaseKind,
    take_once: bool,
) -> EngineReleaseReceipt {
    EngineReleaseReceipt::from_control(
        binding.grant_id,
        binding.decoder_id,
        binding.child_request_ids,
        binding.prefill_bootstrap_endpoint,
        binding.slot_generations,
        binding.bootstrap_rooms,
        binding.grant_digest,
        kind,
        Uuid::new_v4(),
        AuthorityDigest([7; 32]),
        take_once,
    )
}

#[cfg(test)]
pub(crate) fn issue_test_quarantine_receipt(
    binding: TestEngineReceiptBinding,
    take_once: bool,
) -> EngineQuarantineReceipt {
    EngineQuarantineReceipt::from_control(
        binding.grant_id,
        binding.decoder_id,
        binding.child_request_ids,
        binding.prefill_bootstrap_endpoint,
        binding.slot_generations,
        binding.bootstrap_rooms,
        binding.grant_digest,
        Uuid::new_v4(),
        AuthorityDigest([8; 32]),
        take_once,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_ids() -> (PrefillId, DecoderId) {
        (
            PrefillId::new(
                HttpOrigin::parse("http://prefill:30000").unwrap(),
                Uuid::new_v4(),
            )
            .unwrap(),
            DecoderId::new(
                HttpOrigin::parse("http://decode:30001").unwrap(),
                Uuid::new_v4(),
            )
            .unwrap(),
        )
    }

    fn child(
        child_request_id: Uuid,
        slot_generation: Uuid,
        bootstrap_room: u64,
    ) -> DecoderGrantChildBinding {
        DecoderGrantChildBinding::new(
            child_request_id,
            DecoderSlotGeneration::new(slot_generation),
            bootstrap_room,
            1,
            1,
            AuthorityDigest([1; 32]),
            AuthorityDigest([2; 32]),
            DecoderGrantChildAccounting::new(100, 10),
        )
        .unwrap()
    }

    #[derive(Clone)]
    struct DigestFixture {
        grant_id: Uuid,
        reservation_attempt_id: Uuid,
        inference_route: DecoderInferenceRoute,
        request_shape: DecoderRequestShape,
        prepared_ttl_ms: u64,
        prepared_expires_at_unix_ms: u64,
        base_request_body: Bytes,
        request_body: Bytes,
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        request_chain_id: Uuid,
        source_tp_size: usize,
        decoder_id: DecoderId,
        children: Vec<DecoderGrantChildBinding>,
    }

    impl DigestFixture {
        fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
            digest_reserve_attempt(
                self.reservation_attempt_id,
                self.inference_route,
                self.request_shape,
                self.prepared_ttl_ms,
                &self.base_request_body,
                &self.prefill_id,
                &self.prefill_bootstrap_endpoint,
                self.request_chain_id,
                self.source_tp_size,
                &self.decoder_id,
                &self
                    .children
                    .iter()
                    .map(DecoderGrantChildBinding::child_request_id)
                    .collect::<Vec<_>>(),
            )
        }

        fn reservation_digest(&self) -> DecoderReservationDigest {
            digest_reservation(
                self.grant_id,
                self.reserve_attempt_digest(),
                self.prepared_expires_at_unix_ms,
                &self.children,
            )
        }

        fn grant_digest(&self) -> DecoderGrantDigest {
            digest_binding(self.reservation_digest(), &self.request_body)
        }

        fn try_binding(&self) -> Result<DecoderGrantBinding, EngineGrantError> {
            DecoderGrantBinding::new(
                self.grant_id,
                self.reservation_attempt_id,
                self.inference_route,
                self.request_shape,
                self.prepared_ttl_ms,
                self.prepared_expires_at_unix_ms,
                self.base_request_body.clone(),
                self.request_body.clone(),
                self.prefill_id.clone(),
                self.prefill_bootstrap_endpoint.clone(),
                self.request_chain_id,
                self.source_tp_size,
                self.decoder_id.clone(),
                self.children.clone(),
            )
        }

        fn binding(&self) -> DecoderGrantBinding {
            self.try_binding().unwrap()
        }
    }

    fn digest_fixture() -> DigestFixture {
        DigestFixture {
            grant_id: Uuid::parse_str("00112233-4455-4677-8899-aabbccddeeff").unwrap(),
            reservation_attempt_id: Uuid::parse_str(
                "fedcba98-7654-4321-8fed-cba987654321",
            )
            .unwrap(),
            inference_route: DecoderInferenceRoute::Generate,
            request_shape: DecoderRequestShape::Batch,
            prepared_ttl_ms: 2_500,
            prepared_expires_at_unix_ms: 1_900_000_000_123,
            base_request_body: Bytes::from_static(
                br#"{"input_ids":[[1,2,3],[4,5]],"rid":["01020304-0506-4708-890a-0b0c0d0e0f10","f0e0d0c0-b0a0-4908-8706-050403020100"]}"#,
            ),
            request_body: Bytes::from_static(
                br#"{"input_ids":[[1,2,3],[4,5]],"rid":["01020304-0506-4708-890a-0b0c0d0e0f10","f0e0d0c0-b0a0-4908-8706-050403020100"],"bootstrap_host":["10.20.30.40","10.20.30.40"],"bootstrap_port":[50051,50051],"bootstrap_room":[41,42]}"#,
            ),
            prefill_id: PrefillId::new(
                HttpOrigin::parse("https://prefill.example:8443").unwrap(),
                Uuid::parse_str("11111111-2222-4333-8444-555555555555").unwrap(),
            )
            .unwrap(),
            prefill_bootstrap_endpoint: PrefillBootstrapEndpoint::new(
                "10.20.30.40",
                50051,
            )
            .unwrap(),
            request_chain_id: Uuid::parse_str("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
                .unwrap(),
            source_tp_size: 4,
            decoder_id: DecoderId::new(
                HttpOrigin::parse("http://decode.example:30001").unwrap(),
                Uuid::parse_str("12345678-9abc-4def-8123-456789abcdef").unwrap(),
            )
            .unwrap(),
            children: vec![
                DecoderGrantChildBinding::new(
                    Uuid::parse_str("01020304-0506-4708-890a-0b0c0d0e0f10").unwrap(),
                    DecoderSlotGeneration::new(
                        Uuid::parse_str("10203040-5060-4780-8900-a0b0c0d0e0f0").unwrap(),
                    ),
                    41,
                    7,
                    3,
                    AuthorityDigest([0x11; 32]),
                    AuthorityDigest([0x22; 32]),
                    DecoderGrantChildAccounting::new(12_345, 321),
                )
                .unwrap(),
                DecoderGrantChildBinding::new(
                    Uuid::parse_str("f0e0d0c0-b0a0-4908-8706-050403020100").unwrap(),
                    DecoderSlotGeneration::new(
                        Uuid::parse_str("0f1e2d3c-4b5a-4978-8695-a4b3c2d1e0ff").unwrap(),
                    ),
                    42,
                    8,
                    9,
                    AuthorityDigest([0x33; 32]),
                    AuthorityDigest([0x44; 32]),
                    DecoderGrantChildAccounting::new(67_890, 654),
                )
                .unwrap(),
            ],
        }
    }

    #[test]
    fn process_identity_binds_url_and_launch_instance() {
        let instance_id = Uuid::new_v4();
        let first = DecoderId::new(
            HttpOrigin::parse("http://decode:30001/").unwrap(),
            instance_id,
        )
        .unwrap();
        let same = DecoderId::new(
            HttpOrigin::parse("http://decode:30001").unwrap(),
            instance_id,
        )
        .unwrap();
        let reused_url = DecoderId::new(
            HttpOrigin::parse("http://decode:30001").unwrap(),
            Uuid::new_v4(),
        )
        .unwrap();
        assert_eq!(first, same);
        assert_ne!(first, reused_url);
    }

    #[test]
    fn http_origin_rejects_non_origin_urls_and_process_identity_rejects_nil_generation() {
        for url in [
            "ftp://decode.test:30001",
            "http://user@decode.test:30001",
            "http://decode.test:30001/worker",
            "http://decode.test:30001?generation=1",
            "http://decode.test:30001#worker",
        ] {
            assert!(HttpOrigin::parse(url).is_err(), "{url}");
        }
        assert!(DecoderId::new(
            HttpOrigin::parse("http://decode.test:30001").unwrap(),
            Uuid::nil(),
        )
        .is_err());
    }

    #[test]
    fn bootstrap_endpoint_requires_exact_nonnil_coordinates() {
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1", 5000).is_ok());
        assert!(PrefillBootstrapEndpoint::new("", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new(" 10.0.0.1", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1\n", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1", 0).is_err());
        assert!(PrefillBootstrapEndpoint::new("localhost", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("127.0.0.1", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("[::1]", 5000).is_err());
    }

    #[test]
    fn grant_identity_accepts_supported_prefill_tp_and_rejects_other_sizes() {
        let fixture = digest_fixture();
        for source_tp_size in [1, 2, 4, 8] {
            let mut candidate = fixture.clone();
            candidate.source_tp_size = source_tp_size;
            assert!(candidate.try_binding().is_ok());
        }
        for source_tp_size in [0, 3, 6, 16] {
            let mut candidate = fixture.clone();
            candidate.source_tp_size = source_tp_size;
            assert!(matches!(
                candidate.try_binding(),
                Err(EngineGrantError::InvalidGrant(message))
                    if message == "source tensor-parallel size must be 1, 2, 4, or 8"
            ));
        }
    }

    #[test]
    fn digest_chain_binds_every_authority_field() {
        let fixture = digest_fixture();
        let reserve_digest = fixture.reserve_attempt_digest();
        let reservation_digest = fixture.reservation_digest();
        let grant_digest = fixture.grant_digest();

        let assert_reserve_change = |mutator: fn(&mut DigestFixture)| {
            let mut changed = fixture.clone();
            mutator(&mut changed);
            assert_ne!(changed.reserve_attempt_digest(), reserve_digest);
        };
        assert_reserve_change(|value| value.reservation_attempt_id = Uuid::new_v4());
        assert_reserve_change(|value| value.inference_route = DecoderInferenceRoute::Completions);
        assert_reserve_change(|value| value.request_shape = DecoderRequestShape::Scalar);
        assert_reserve_change(|value| value.prepared_ttl_ms += 1);
        assert_reserve_change(|value| {
            value.base_request_body = Bytes::from_static(b"{\"input_ids\":[[9],[4,5]]}")
        });
        assert_reserve_change(|value| {
            value.prefill_id = PrefillId::new(
                HttpOrigin::parse("https://other-prefill.example:8443").unwrap(),
                value.prefill_id.instance_id(),
            )
            .unwrap()
        });
        assert_reserve_change(|value| {
            value.prefill_id =
                PrefillId::new(value.prefill_id.origin().clone(), Uuid::new_v4()).unwrap()
        });
        assert_reserve_change(|value| {
            value.prefill_bootstrap_endpoint =
                PrefillBootstrapEndpoint::new("10.20.30.41", 50052).unwrap()
        });
        assert_reserve_change(|value| value.request_chain_id = Uuid::new_v4());
        assert_reserve_change(|value| value.source_tp_size = 2);
        assert_reserve_change(|value| {
            value.decoder_id = DecoderId::new(
                HttpOrigin::parse("http://other-decode.example:30001").unwrap(),
                value.decoder_id.instance_id(),
            )
            .unwrap()
        });
        assert_reserve_change(|value| {
            value.decoder_id =
                DecoderId::new(value.decoder_id.origin().clone(), Uuid::new_v4()).unwrap()
        });
        assert_reserve_change(|value| value.children.swap(0, 1));

        assert_ne!(
            digest_reservation(
                Uuid::new_v4(),
                reserve_digest,
                fixture.prepared_expires_at_unix_ms,
                &fixture.children,
            ),
            reservation_digest
        );
        assert_ne!(
            digest_reservation(
                fixture.grant_id,
                DecoderReserveAttemptDigest([0xAB; 32]),
                fixture.prepared_expires_at_unix_ms,
                &fixture.children,
            ),
            reservation_digest
        );
        assert_ne!(
            digest_reservation(
                fixture.grant_id,
                reserve_digest,
                fixture.prepared_expires_at_unix_ms + 1,
                &fixture.children,
            ),
            reservation_digest
        );

        let assert_allocation_change = |mutator: fn(&mut DecoderGrantChildBinding)| {
            let mut children = fixture.children.clone();
            mutator(&mut children[0]);
            assert_ne!(
                digest_reservation(
                    fixture.grant_id,
                    reserve_digest,
                    fixture.prepared_expires_at_unix_ms,
                    &children,
                ),
                reservation_digest
            );
        };
        assert_allocation_change(|child| child.child_request_id = Uuid::new_v4());
        assert_allocation_change(|child| {
            child.slot_generation = DecoderSlotGeneration::new(Uuid::new_v4())
        });
        assert_allocation_change(|child| child.bootstrap_room += 1);
        assert_allocation_change(|child| child.request_slot += 1);
        assert_allocation_change(|child| child.request_generation += 1);
        assert_allocation_change(|child| child.writer_manifest_digest = AuthorityDigest([5; 32]));
        assert_allocation_change(|child| child.allocation_digest = AuthorityDigest([6; 32]));
        assert_allocation_change(|child| child.accounting.reserved_kv_tokens += 1);
        assert_allocation_change(|child| child.accounting.remaining_decode_tokens += 1);
        let mut reordered_children = fixture.children.clone();
        reordered_children.swap(0, 1);
        assert_ne!(
            digest_reservation(
                fixture.grant_id,
                reserve_digest,
                fixture.prepared_expires_at_unix_ms,
                &reordered_children,
            ),
            reservation_digest
        );

        assert_ne!(
            digest_binding(DecoderReservationDigest([0xCD; 32]), &fixture.request_body,),
            grant_digest
        );
        assert_ne!(
            digest_binding(reservation_digest, b"{\"final\":\"changed\"}"),
            grant_digest
        );

        let binding = fixture.binding();
        assert_eq!(binding.reserve_attempt_digest(), reserve_digest);
        assert_eq!(binding.reservation_digest(), reservation_digest);
        assert_eq!(binding.digest(), grant_digest);
    }

    #[test]
    fn grant_rejects_invalid_or_duplicate_child_allocation_identity() {
        let slot = Uuid::new_v4();
        let first_child = Uuid::new_v4();
        let second_child = Uuid::new_v4();
        for children in [
            vec![child(first_child, Uuid::nil(), 11)],
            vec![
                child(first_child, Uuid::new_v4(), 11),
                child(first_child, Uuid::new_v4(), 12),
            ],
            vec![child(first_child, slot, 11), child(second_child, slot, 12)],
            vec![
                child(first_child, Uuid::new_v4(), 11),
                child(second_child, Uuid::new_v4(), 11),
            ],
        ] {
            let mut fixture = digest_fixture();
            fixture.request_shape = if children.len() == 1 {
                DecoderRequestShape::Scalar
            } else {
                DecoderRequestShape::Batch
            };
            fixture.children = children;
            assert!(fixture.try_binding().is_err());
        }
    }

    #[test]
    fn digest_domains_match_cross_language_golden_vectors() {
        let fixture = digest_fixture();
        assert_eq!(
            fixture.reserve_attempt_digest().to_hex(),
            "1673ccc0b56472cbcf512f2caa4fb2989ecb82729791f3438a677af1b2582c14"
        );
        assert_eq!(
            fixture.reservation_digest().to_hex(),
            "d0b0b05dea2236839cc9bef079325e2ff0be11d93bcf9c97aa4718cfe5de495a"
        );
        assert_eq!(
            fixture.grant_digest().to_hex(),
            "1a47879143d21f3e0945673cd4b207d2a347cc83d326897967d8937568d0cd73"
        );
    }

    #[test]
    fn prepared_test_grant_carries_test_only_control_capability() {
        let (prefill_id, decoder_id) = process_ids();
        let grant = issue_test_grant(
            prefill_id,
            Uuid::new_v4(),
            2,
            decoder_id,
            Uuid::new_v4(),
            vec![DecoderSlotGeneration::new(Uuid::new_v4())],
            vec![11],
            vec![DecoderGrantChildAccounting::new(100, 10)],
        )
        .unwrap();
        assert!(grant.control.is_some());
    }
}
