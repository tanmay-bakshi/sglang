//! Engine-backed decoder reservation capabilities.
//!
//! The decoder engine is the sole allocation authority. A prepared grant owns
//! an exact ordered child-allocation vector. Promotion is an asynchronous engine
//! transition that must finish before either inference request becomes pollable.
//! Terminal pool accounting is released only by an exact engine receipt.

use std::{collections::HashSet, fmt, sync::Arc};

use reqwest::Url;
use thiserror::Error;
use uuid::Uuid;

mod control;

pub use control::{DecoderGrantControlClient, DecoderGrantReservation};

const GRANT_DIGEST_DOMAIN: &[u8] = b"sglang-pd-decoder-grant-v3";

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProcessIdentity {
    url: Arc<str>,
    instance_id: Uuid,
}

impl ProcessIdentity {
    fn new(url: impl Into<String>, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        if instance_id.is_nil() {
            return Err(ProcessIdentityError::NilInstanceId);
        }
        let url = url.into();
        let parsed = Url::parse(&url)
            .map_err(|error| ProcessIdentityError::InvalidUrl(error.to_string()))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(ProcessIdentityError::InvalidUrl(
                "process URL must use http or https".to_string(),
            ));
        }
        if parsed.host_str().is_none() {
            return Err(ProcessIdentityError::InvalidUrl(
                "process URL must contain a host".to_string(),
            ));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(ProcessIdentityError::InvalidUrl(
                "process URL cannot contain credentials".to_string(),
            ));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(ProcessIdentityError::InvalidUrl(
                "process URL cannot contain a query or fragment".to_string(),
            ));
        }
        if parsed.path() != "/" {
            return Err(ProcessIdentityError::InvalidUrl(
                "process URL must be an origin without a path".to_string(),
            ));
        }
        let canonical_url = parsed.as_str().trim_end_matches('/').to_string();
        Ok(Self {
            url: Arc::from(canonical_url),
            instance_id,
        })
    }

    fn url(&self) -> &str {
        &self.url
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
    pub fn new(url: impl Into<String>, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        Ok(Self(ProcessIdentity::new(url, instance_id)?))
    }

    /// Canonical selected worker base URL.
    pub fn url(&self) -> &str {
        self.0.url()
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
    pub fn new(url: impl Into<String>, instance_id: Uuid) -> Result<Self, ProcessIdentityError> {
        Ok(Self(ProcessIdentity::new(url, instance_id)?))
    }

    /// Canonical selected worker base URL.
    pub fn url(&self) -> &str {
        self.0.url()
    }

    /// SGLang launch generation from ``PortArgs.instance_id``.
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
    #[error("invalid process URL: {0}")]
    InvalidUrl(String),
    #[error("process launch instance cannot be the nil UUID")]
    NilInstanceId,
}

/// Generation-scoped prefill transport endpoint advertised by engine capabilities.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefillBootstrapEndpoint {
    host: Arc<str>,
    port: u16,
}

impl PrefillBootstrapEndpoint {
    /// Construct the exact endpoint consumed by decoder-side pre-allocation.
    pub fn new(host: impl Into<String>, port: u16) -> Result<Self, EngineGrantError> {
        let host = host.into();
        if host.is_empty()
            || host.trim() != host
            || host.chars().any(|character| character.is_control())
        {
            return Err(EngineGrantError::InvalidGrant(
                "prefill bootstrap host must be nonempty and control-free".to_string(),
            ));
        }
        if port == 0 {
            return Err(EngineGrantError::InvalidGrant(
                "prefill bootstrap port must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            host: Arc::from(host),
            port,
        })
    }

    /// Exact engine-advertised bootstrap host.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Exact engine-advertised bootstrap port.
    pub fn port(&self) -> u16 {
        self.port
    }
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecoderGrantBinding {
    grant_id: Uuid,
    inference_route: DecoderInferenceRoute,
    prefill_id: PrefillId,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: DecoderId,
    children: Arc<[DecoderGrantChildBinding]>,
    slot_generations: Arc<[DecoderSlotGeneration]>,
    bootstrap_rooms: Arc<[u64]>,
    accounting: DecoderGrantAccounting,
    digest: DecoderGrantDigest,
}

impl DecoderGrantBinding {
    #[allow(clippy::too_many_arguments)]
    fn new(
        grant_id: Uuid,
        inference_route: DecoderInferenceRoute,
        request_body_json: &str,
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        request_chain_id: Uuid,
        source_tp_size: usize,
        decoder_id: DecoderId,
        children: Vec<DecoderGrantChildBinding>,
    ) -> Result<Self, EngineGrantError> {
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
        if source_tp_size != 2 && source_tp_size != 4 {
            return Err(EngineGrantError::InvalidGrant(
                "source tensor-parallel size must be 2 or 4".to_string(),
            ));
        }
        if children.is_empty() {
            return Err(EngineGrantError::InvalidGrant(
                "an engine grant must contain at least one child".to_string(),
            ));
        }

        let request_ids: HashSet<Uuid> = children
            .iter()
            .map(|child| child.child_request_id())
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
        let unique_slots: HashSet<DecoderSlotGeneration> =
            slot_generations.iter().copied().collect();
        if unique_slots.len() != children.len() {
            return Err(EngineGrantError::InvalidGrant(
                "an engine grant cannot repeat a decoder slot generation".to_string(),
            ));
        }
        let bootstrap_rooms: Vec<u64> = children.iter().map(|child| child.bootstrap_room).collect();
        let unique_rooms: HashSet<u64> = bootstrap_rooms.iter().copied().collect();
        if unique_rooms.len() != children.len() {
            return Err(EngineGrantError::InvalidGrant(
                "an engine grant cannot repeat a decoder-local bootstrap room".to_string(),
            ));
        }
        let accounting =
            DecoderGrantAccounting::new(children.iter().map(|child| child.accounting).collect())?;
        let digest = digest_binding(
            grant_id,
            inference_route,
            request_body_json,
            &prefill_id,
            &prefill_bootstrap_endpoint,
            request_chain_id,
            source_tp_size,
            &decoder_id,
            &children,
        );

        Ok(Self {
            grant_id,
            inference_route,
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

    pub(crate) fn grant_id(&self) -> Uuid {
        self.grant_id
    }

    pub(crate) fn inference_route(&self) -> DecoderInferenceRoute {
        self.inference_route
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

    pub(crate) fn digest(&self) -> DecoderGrantDigest {
        self.digest
    }
}

#[allow(clippy::too_many_arguments)]
fn digest_binding(
    grant_id: Uuid,
    inference_route: DecoderInferenceRoute,
    request_body_json: &str,
    prefill_id: &PrefillId,
    prefill_bootstrap_endpoint: &PrefillBootstrapEndpoint,
    request_chain_id: Uuid,
    source_tp_size: usize,
    decoder_id: &DecoderId,
    children: &[DecoderGrantChildBinding],
) -> DecoderGrantDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(GRANT_DIGEST_DOMAIN);
    hasher.update(grant_id.as_bytes());
    hash_text(&mut hasher, inference_route.as_str());
    hash_text(&mut hasher, request_body_json);
    hash_process(&mut hasher, &prefill_id.0);
    hash_text(&mut hasher, prefill_bootstrap_endpoint.host());
    hasher.update(&u64::from(prefill_bootstrap_endpoint.port()).to_le_bytes());
    hasher.update(request_chain_id.as_bytes());
    hasher.update(&(source_tp_size as u64).to_le_bytes());
    hash_process(&mut hasher, &decoder_id.0);
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
    DecoderGrantDigest(*hasher.finalize().as_bytes())
}

fn hash_process(hasher: &mut blake3::Hasher, process: &ProcessIdentity) {
    hash_text(hasher, process.url());
    hasher.update(process.instance_id().as_bytes());
}

fn hash_text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Concrete engine-issued prepared reservation.
pub struct EngineDecoderGrant {
    binding: DecoderGrantBinding,
    control: Option<control::PreparedGrantControl>,
}

impl fmt::Debug for EngineDecoderGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineDecoderGrant")
            .field("binding", &self.binding)
            .field("has_control", &self.control.is_some())
            .finish()
    }
}

impl EngineDecoderGrant {
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

    /// Digest covering the request and every ordered child allocation.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Explicitly cancel a reservation that never crossed the dispatch boundary.
    pub async fn cancel(mut self) -> Result<EngineReleaseReceipt, EngineGrantError> {
        let receipt = self.control()?.cancel(&self.binding).await?;
        self.control = None;
        Ok(receipt)
    }

    /// Irreversibly promote the reservation before any inference send is pollable.
    pub async fn promote(mut self) -> Result<RetainedEngineGrant, EngineGrantError> {
        if let Err(error) = self.control()?.promote(&self.binding).await {
            if let Some(control) = self.control.take() {
                control.best_effort_quarantine(
                    self.binding.clone(),
                    "promotion_ambiguous",
                    Some(error.to_string()),
                );
            }
            return Err(error);
        }
        let control = self
            .control
            .take()
            .expect("prepared engine grant lost its concrete control capability");
        Ok(RetainedEngineGrant {
            binding: self.binding.clone(),
            control: Some(control.into_retained()),
        })
    }

    fn control(&self) -> Result<&control::PreparedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "engine decoder grant has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for EngineDecoderGrant {
    fn drop(&mut self) {
        let Some(control) = self.control.take() else {
            return;
        };
        control.best_effort_cancel(self.binding.clone());
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

    /// Digest covering the request and every ordered child allocation.
    pub fn grant_digest(&self) -> DecoderGrantDigest {
        self.binding.digest()
    }

    /// Release after authoritative successful completion and teardown.
    pub async fn complete(mut self) -> Result<EngineReleaseReceipt, EngineGrantError> {
        let receipt = self.control()?.complete(&self.binding).await?;
        self.control = None;
        Ok(receipt)
    }

    /// Ask the engine to abort every child and prove terminal quiescence.
    ///
    /// The engine releases its allocations only when it can prove an exact
    /// all-child no-submit or terminal outcome. Otherwise the grant remains
    /// monotonically quarantined.
    pub async fn abort(
        mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineAbortOutcome, EngineGrantError> {
        let receipt = self
            .control()?
            .abort(&self.binding, reason_code, diagnostic)
            .await?;
        self.control = None;
        Ok(receipt)
    }

    /// Monotonically quarantine an ambiguous promoted reservation without release.
    pub async fn quarantine(
        mut self,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineQuarantineReceipt, EngineGrantError> {
        let receipt = self
            .control()?
            .quarantine(&self.binding, reason_code, diagnostic)
            .await?;
        self.control = None;
        Ok(receipt)
    }

    fn control(&self) -> Result<&control::RetainedGrantControl, EngineGrantError> {
        self.control.as_ref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "retained engine grant has no concrete control capability".to_string(),
            )
        })
    }
}

impl Drop for RetainedEngineGrant {
    fn drop(&mut self) {
        let Some(control) = self.control.take() else {
            return;
        };
        control.best_effort_quarantine(self.binding.clone(), "retained_grant_dropped", None);
    }
}

/// Authoritative terminal release kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineReleaseKind {
    PreparedCancelled,
    Completed,
    Aborted,
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

/// Concrete decoder reservation and lifecycle failures.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum EngineGrantError {
    #[error(transparent)]
    InvalidProcessIdentity(#[from] ProcessIdentityError),
    #[error("invalid engine decoder grant: {0}")]
    InvalidGrant(String),
    #[error("decoder allocator refused the reservation: {0}")]
    AllocatorRefused(String),
    #[error("decoder control request failed during {operation}: {message}")]
    ControlRequestFailed {
        operation: &'static str,
        message: String,
    },
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
) -> Result<EngineDecoderGrant, EngineGrantError> {
    if slot_generations.len() != bootstrap_rooms.len() || slot_generations.len() != accounting.len()
    {
        return Err(EngineGrantError::InvalidGrant(
            "test grant vectors must have identical lengths".to_string(),
        ));
    }
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
        DecoderInferenceRoute::Generate,
        "{}",
        prefill_id,
        PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000)?,
        request_chain_id,
        source_tp_size,
        decoder_id,
        children,
    )?;
    Ok(EngineDecoderGrant {
        binding,
        control: None,
    })
}

#[cfg(test)]
pub(crate) fn issue_test_release_receipt(
    grant_id: Uuid,
    decoder_id: DecoderId,
    slot_generations: Vec<DecoderSlotGeneration>,
    bootstrap_rooms: Vec<u64>,
    grant_digest: DecoderGrantDigest,
    kind: EngineReleaseKind,
    take_once: bool,
) -> EngineReleaseReceipt {
    EngineReleaseReceipt::from_control(
        grant_id,
        decoder_id,
        (0..slot_generations.len())
            .map(|index| Uuid::from_u128(index as u128 + 1))
            .collect(),
        PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000)
            .expect("fixed test bootstrap endpoint must be valid"),
        slot_generations,
        bootstrap_rooms,
        grant_digest,
        kind,
        Uuid::new_v4(),
        AuthorityDigest([7; 32]),
        take_once,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_ids() -> (PrefillId, DecoderId) {
        (
            PrefillId::new("http://prefill:30000", Uuid::new_v4()).unwrap(),
            DecoderId::new("http://decode:30001", Uuid::new_v4()).unwrap(),
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

    #[test]
    fn process_identity_binds_url_and_launch_instance() {
        let instance_id = Uuid::new_v4();
        let first = DecoderId::new("http://decode:30001/", instance_id).unwrap();
        let same = DecoderId::new("http://decode:30001", instance_id).unwrap();
        let reused_url = DecoderId::new("http://decode:30001", Uuid::new_v4()).unwrap();
        assert_eq!(first, same);
        assert_ne!(first, reused_url);
    }

    #[test]
    fn process_identity_accepts_only_nonnil_http_origins() {
        let instance_id = Uuid::new_v4();
        for url in [
            "ftp://decode.test:30001",
            "http://user@decode.test:30001",
            "http://decode.test:30001/worker",
            "http://decode.test:30001?generation=1",
            "http://decode.test:30001#worker",
        ] {
            assert!(DecoderId::new(url, instance_id).is_err(), "{url}");
        }
        assert!(DecoderId::new("http://decode.test:30001", Uuid::nil()).is_err());
    }

    #[test]
    fn bootstrap_endpoint_requires_exact_nonnil_coordinates() {
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1", 5000).is_ok());
        assert!(PrefillBootstrapEndpoint::new("", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new(" 10.0.0.1", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1\n", 5000).is_err());
        assert!(PrefillBootstrapEndpoint::new("10.0.0.1", 0).is_err());
    }

    #[test]
    fn grant_digest_binds_request_processes_and_exact_child_order() {
        let (prefill_id, decoder_id) = process_ids();
        let grant_id = Uuid::new_v4();
        let chain_id = Uuid::new_v4();
        let first_slot = Uuid::new_v4();
        let second_slot = Uuid::new_v4();
        let first_child = Uuid::new_v4();
        let second_child = Uuid::new_v4();
        let bootstrap_endpoint =
            PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000).unwrap();
        let base = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::Generate,
            r#"{"prompt":"hello","temperature":1.0}"#,
            prefill_id.clone(),
            bootstrap_endpoint.clone(),
            chain_id,
            2,
            decoder_id.clone(),
            vec![
                child(first_child, first_slot, 11),
                child(second_child, second_slot, 12),
            ],
        )
        .unwrap()
        .digest();
        let reordered = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::Generate,
            r#"{"prompt":"hello","temperature":1.0}"#,
            prefill_id.clone(),
            bootstrap_endpoint.clone(),
            chain_id,
            2,
            decoder_id.clone(),
            vec![
                child(second_child, second_slot, 12),
                child(first_child, first_slot, 11),
            ],
        )
        .unwrap()
        .digest();
        let changed_body = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::Generate,
            r#"{"prompt":"goodbye","temperature":1.0}"#,
            prefill_id.clone(),
            bootstrap_endpoint.clone(),
            chain_id,
            2,
            decoder_id.clone(),
            vec![
                child(first_child, first_slot, 11),
                child(second_child, second_slot, 12),
            ],
        )
        .unwrap()
        .digest();
        let changed_bootstrap = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::Generate,
            r#"{"prompt":"hello","temperature":1.0}"#,
            prefill_id,
            PrefillBootstrapEndpoint::new("other-bootstrap.test", 5001).unwrap(),
            chain_id,
            2,
            decoder_id,
            vec![
                child(first_child, first_slot, 11),
                child(second_child, second_slot, 12),
            ],
        )
        .unwrap()
        .digest();
        assert_ne!(base, reordered);
        assert_ne!(base, changed_body);
        assert_ne!(base, changed_bootstrap);
    }

    #[test]
    fn grant_rejects_invalid_or_duplicate_child_allocation_identity() {
        let (prefill_id, decoder_id) = process_ids();
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
            assert!(DecoderGrantBinding::new(
                Uuid::new_v4(),
                DecoderInferenceRoute::Generate,
                "{}",
                prefill_id.clone(),
                PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 5000).unwrap(),
                Uuid::new_v4(),
                2,
                decoder_id.clone(),
                children,
            )
            .is_err());
        }
    }

    #[test]
    fn grant_digest_v3_matches_cross_language_golden_vector() {
        let grant_id = Uuid::parse_str("00112233-4455-4677-8899-aabbccddeeff").unwrap();
        let prefill_id = PrefillId::new(
            "https://prefill.example:8443",
            Uuid::parse_str("11111111-2222-4333-8444-555555555555").unwrap(),
        )
        .unwrap();
        let bootstrap_endpoint = PrefillBootstrapEndpoint::new("10.20.30.40", 50051).unwrap();
        let chain_id = Uuid::parse_str("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee").unwrap();
        let decoder_id = DecoderId::new(
            "http://decode.example:30001",
            Uuid::parse_str("12345678-9abc-4def-8123-456789abcdef").unwrap(),
        )
        .unwrap();
        let children = vec![
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
        ];
        let binding = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::ChatCompletions,
            r#"{"model":"gemma","rid":["01020304-0506-4708-890a-0b0c0d0e0f10","f0e0d0c0-b0a0-4908-8706-050403020100"],"max_tokens":17}"#,
            prefill_id,
            bootstrap_endpoint,
            chain_id,
            4,
            decoder_id,
            children,
        )
        .unwrap();
        assert_eq!(
            binding.digest().to_hex(),
            "5294f13bfa1b2f9ca5e553b96212737a7fe993f88b138ed3b063f24ff39b9938"
        );
    }

    #[test]
    fn prepared_test_grant_has_no_forgeable_production_control() {
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
        assert!(grant.control.is_none());
    }
}
