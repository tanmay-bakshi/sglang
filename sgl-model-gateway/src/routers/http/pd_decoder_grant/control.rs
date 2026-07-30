use std::{fmt, sync::Arc, time::Duration};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::{
    AuthorityDigest, DecoderGrantBinding, DecoderGrantChildAccounting, DecoderGrantChildBinding,
    DecoderGrantDigest, DecoderId, DecoderInferenceRoute, DecoderSlotGeneration,
    EngineAbortOutcome, EngineDecoderGrant, EngineGrantError, EngineQuarantineReceipt,
    EngineReleaseKind, EngineReleaseReceipt, PrefillBootstrapEndpoint, PrefillId,
};

const SCHEMA_VERSION: u32 = 1;
const CONTROL_PATH: &str = "/_internal/pd/v1/decode-reservations";
const GRANT_TOKEN_BYTES: usize = 32;
const MAX_REASON_CODE_BYTES: usize = 64;
const MAX_DIAGNOSTIC_BYTES: usize = 512;

/// Exact immutable input to one batch-atomic decoder reservation.
#[derive(Clone, Debug)]
pub struct DecoderGrantReservation {
    prefill_id: PrefillId,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    decoder_id: DecoderId,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: DecoderInferenceRoute,
    request_body_json: Arc<str>,
    child_request_ids: Arc<[Uuid]>,
}

impl DecoderGrantReservation {
    /// Construct a reservation for one exact enriched inference request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        decoder_id: DecoderId,
        logical_request_chain_id: Uuid,
        source_tp_size: usize,
        prepared_ttl: Duration,
        inference_route: DecoderInferenceRoute,
        request_body_json: impl Into<String>,
        child_request_ids: Vec<Uuid>,
    ) -> Result<Self, EngineGrantError> {
        if logical_request_chain_id.is_nil() {
            return Err(EngineGrantError::InvalidGrant(
                "logical request-chain identity cannot be the nil UUID".to_string(),
            ));
        }
        if source_tp_size != 2 && source_tp_size != 4 {
            return Err(EngineGrantError::InvalidGrant(
                "source tensor-parallel size must be 2 or 4".to_string(),
            ));
        }
        let prepared_ttl_ms = u64::try_from(prepared_ttl.as_millis()).map_err(|_| {
            EngineGrantError::InvalidGrant(
                "prepared decoder grant TTL exceeds u64 milliseconds".to_string(),
            )
        })?;
        if prepared_ttl_ms == 0 {
            return Err(EngineGrantError::InvalidGrant(
                "prepared decoder grant TTL must be nonzero".to_string(),
            ));
        }
        validate_child_request_ids(&child_request_ids)?;

        let request_body_json = request_body_json.into();
        validate_request_rids(&request_body_json, &child_request_ids)?;

        Ok(Self {
            prefill_id,
            prefill_bootstrap_endpoint,
            decoder_id,
            logical_request_chain_id,
            source_tp_size,
            prepared_ttl_ms,
            inference_route,
            request_body_json: Arc::from(request_body_json),
            child_request_ids: Arc::from(child_request_ids),
        })
    }

    /// Selected prefill process generation.
    pub fn prefill_id(&self) -> &PrefillId {
        &self.prefill_id
    }

    /// Exact engine-advertised endpoint used by decoder pre-allocation.
    pub fn prefill_bootstrap_endpoint(&self) -> &PrefillBootstrapEndpoint {
        &self.prefill_bootstrap_endpoint
    }

    /// Candidate decoder process generation.
    pub fn decoder_id(&self) -> &DecoderId {
        &self.decoder_id
    }

    /// One-owner logical request chain.
    pub fn logical_request_chain_id(&self) -> Uuid {
        self.logical_request_chain_id
    }

    /// Exact prefill tensor-parallel width.
    pub fn source_tp_size(&self) -> usize {
        self.source_tp_size
    }

    /// Engine-clock-owned prepared lease TTL in milliseconds.
    pub fn prepared_ttl_ms(&self) -> u64 {
        self.prepared_ttl_ms
    }

    /// Closed inference route bound into the grant.
    pub fn inference_route(&self) -> DecoderInferenceRoute {
        self.inference_route
    }

    /// Exact once-serialized enriched request JSON bound by the grant digest.
    pub fn request_body_json(&self) -> &str {
        &self.request_body_json
    }

    /// Ordered gateway-owned child identities represented by ``rid``.
    pub fn child_request_ids(&self) -> &[Uuid] {
        &self.child_request_ids
    }
}

fn validate_child_request_ids(child_request_ids: &[Uuid]) -> Result<(), EngineGrantError> {
    if child_request_ids.is_empty() {
        return Err(EngineGrantError::InvalidGrant(
            "a decoder reservation must contain at least one child identity".to_string(),
        ));
    }
    if child_request_ids.iter().any(Uuid::is_nil) {
        return Err(EngineGrantError::InvalidGrant(
            "decoder reservation child identities cannot contain the nil UUID".to_string(),
        ));
    }
    let unique: std::collections::HashSet<Uuid> = child_request_ids.iter().copied().collect();
    if unique.len() != child_request_ids.len() {
        return Err(EngineGrantError::InvalidGrant(
            "decoder reservation child identities must be unique".to_string(),
        ));
    }
    Ok(())
}

fn validate_request_rids(
    request_body_json: &str,
    child_request_ids: &[Uuid],
) -> Result<(), EngineGrantError> {
    let request_body: Value = serde_json::from_str(request_body_json).map_err(|error| {
        EngineGrantError::InvalidGrant(format!(
            "reservation request body is not valid JSON: {error}"
        ))
    })?;
    let request_object = request_body.as_object().ok_or_else(|| {
        EngineGrantError::InvalidGrant("reservation request body must be a JSON object".to_string())
    })?;
    let rid = request_object.get("rid").ok_or_else(|| {
        EngineGrantError::InvalidGrant(
            "enriched reservation request body must contain rid".to_string(),
        )
    })?;

    if child_request_ids.len() == 1 {
        let expected = child_request_ids[0].to_string();
        if rid.as_str() != Some(expected.as_str()) {
            return Err(EngineGrantError::InvalidGrant(
                "scalar request rid must be the canonical gateway child UUID".to_string(),
            ));
        }
        return Ok(());
    }

    let rid_values = rid.as_array().ok_or_else(|| {
        EngineGrantError::InvalidGrant(
            "batched request rid must be an ordered UUID array".to_string(),
        )
    })?;
    if rid_values.len() != child_request_ids.len() {
        return Err(EngineGrantError::InvalidGrant(format!(
            "batched request rid length {} differs from child count {}",
            rid_values.len(),
            child_request_ids.len()
        )));
    }
    for (index, (rid_value, child_request_id)) in
        rid_values.iter().zip(child_request_ids).enumerate()
    {
        let expected = child_request_id.to_string();
        if rid_value.as_str() != Some(expected.as_str()) {
            return Err(EngineGrantError::InvalidGrant(format!(
                "batched request rid {index} is not its canonical gateway child UUID"
            )));
        }
    }
    Ok(())
}

/// Concrete HTTP client for decoder reservation lifecycle authority.
#[derive(Clone, Debug)]
pub struct DecoderGrantControlClient {
    client: Client,
}

impl DecoderGrantControlClient {
    /// Bind lifecycle operations to the gateway's configured HTTP client.
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Reserve one batch atomically on the exact candidate decoder generation.
    pub async fn reserve(
        &self,
        reservation: &DecoderGrantReservation,
    ) -> Result<EngineDecoderGrant, EngineGrantError> {
        let endpoint = format!("{}{CONTROL_PATH}/reserve", reservation.decoder_id.url());
        let request = ReserveRequest {
            schema_version: SCHEMA_VERSION,
            prefill_process: WireProcessIdentity::from_prefill(&reservation.prefill_id),
            prefill_bootstrap_endpoint: WireBootstrapEndpoint::from(
                &reservation.prefill_bootstrap_endpoint,
            ),
            decoder_process: WireProcessIdentity::from_decoder(&reservation.decoder_id),
            logical_request_chain_id: reservation.logical_request_chain_id,
            source_tp_size: reservation.source_tp_size,
            prepared_ttl_ms: reservation.prepared_ttl_ms,
            inference_route: reservation.inference_route.as_str(),
            request_body_json: reservation.request_body_json(),
            child_request_ids: reservation.child_request_ids(),
        };
        let response = self
            .client
            .post(endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|error| EngineGrantError::ControlRequestFailed {
                operation: "reserve",
                message: error.to_string(),
            })?;
        if response.status() == StatusCode::CONFLICT
            || response.status() == StatusCode::TOO_MANY_REQUESTS
        {
            return Err(EngineGrantError::AllocatorRefused(
                response_error_message(response).await,
            ));
        }
        if !response.status().is_success() {
            return Err(EngineGrantError::ControlRequestFailed {
                operation: "reserve",
                message: response_error_message(response).await,
            });
        }
        let response: ReserveResponse = response
            .json()
            .await
            .map_err(|error| EngineGrantError::ProtocolViolation(error.to_string()))?;
        self.validate_reserve_response(reservation, response)
    }

    fn validate_reserve_response(
        &self,
        reservation: &DecoderGrantReservation,
        response: ReserveResponse,
    ) -> Result<EngineDecoderGrant, EngineGrantError> {
        validate_schema(response.schema_version)?;
        if response.state != WireGrantState::Prepared {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "reserve returned state {:?}, expected prepared",
                response.state
            )));
        }
        validate_process(
            "prefill",
            &response.prefill_process,
            reservation.prefill_id.url(),
            reservation.prefill_id.instance_id(),
        )?;
        validate_bootstrap_endpoint(
            &response.prefill_bootstrap_endpoint,
            &reservation.prefill_bootstrap_endpoint,
        )?;
        validate_process(
            "decoder",
            &response.decoder_process,
            reservation.decoder_id.url(),
            reservation.decoder_id.instance_id(),
        )?;
        if response.logical_request_chain_id != reservation.logical_request_chain_id {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response changed the logical request chain".to_string(),
            ));
        }
        if response.grant_id.is_nil() {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response contains a nil grant identity".to_string(),
            ));
        }
        if response.prepared_expires_at_unix_ms == 0 {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response contains no prepared expiry".to_string(),
            ));
        }
        let token = response.grant_token;
        if response.allocations.len() != reservation.child_request_ids.len() {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "reserve returned {} allocations for {} gateway child identities",
                response.allocations.len(),
                reservation.child_request_ids.len()
            )));
        }

        let mut children = Vec::with_capacity(response.allocations.len());
        for (index, allocation) in response.allocations.into_iter().enumerate() {
            if allocation.child_request_id != reservation.child_request_ids[index] {
                return Err(EngineGrantError::ProtocolViolation(format!(
                    "allocation {index} changed its ordered gateway child identity"
                )));
            }
            if allocation.decoder_slot_generation.is_nil() {
                return Err(EngineGrantError::ProtocolViolation(format!(
                    "allocation {index} contains a nil decoder slot generation"
                )));
            }
            let reserved_kv_tokens =
                usize::try_from(allocation.reserved_kv_tokens).map_err(|_| {
                    EngineGrantError::ProtocolViolation(format!(
                        "allocation {index} KV accounting exceeds usize"
                    ))
                })?;
            let remaining_decode_tokens = usize::try_from(allocation.remaining_decode_tokens)
                .map_err(|_| {
                    EngineGrantError::ProtocolViolation(format!(
                        "allocation {index} decode accounting exceeds usize"
                    ))
                })?;
            children.push(DecoderGrantChildBinding::new(
                allocation.child_request_id,
                DecoderSlotGeneration::new(allocation.decoder_slot_generation),
                allocation.bootstrap_room,
                allocation.request_slot,
                allocation.request_generation,
                AuthorityDigest::from_hex(
                    "writer manifest digest",
                    &allocation.writer_manifest_digest,
                )?,
                AuthorityDigest::from_hex("allocation digest", &allocation.allocation_digest)?,
                DecoderGrantChildAccounting::new(reserved_kv_tokens, remaining_decode_tokens),
            )?);
        }
        let binding = DecoderGrantBinding::new(
            response.grant_id,
            reservation.inference_route,
            reservation.request_body_json(),
            reservation.prefill_id.clone(),
            reservation.prefill_bootstrap_endpoint.clone(),
            reservation.logical_request_chain_id,
            reservation.source_tp_size,
            reservation.decoder_id.clone(),
            children,
        )?;
        let engine_digest = DecoderGrantDigest::from_hex(&response.grant_digest)?;
        if engine_digest != binding.digest() {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "engine grant digest {} differs from gateway digest {}",
                engine_digest.to_hex(),
                binding.digest().to_hex()
            )));
        }
        let control = PreparedGrantControl {
            client: self.client.clone(),
            grant_url: Arc::from(format!(
                "{}{CONTROL_PATH}/{}",
                reservation.decoder_id.url(),
                response.grant_id
            )),
            token,
        };
        Ok(EngineDecoderGrant::from_control(binding, control))
    }
}

#[derive(Clone)]
struct SecretGrantToken(Arc<str>);

impl SecretGrantToken {
    fn new(value: String) -> Result<Self, EngineGrantError> {
        let decoded = URL_SAFE_NO_PAD.decode(&value).map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "grant token is not unpadded base64url: {error}"
            ))
        })?;
        if decoded.len() != GRANT_TOKEN_BYTES {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "grant token decodes to {} bytes, expected {GRANT_TOKEN_BYTES}",
                decoded.len()
            )));
        }
        if URL_SAFE_NO_PAD.encode(decoded) != value {
            return Err(EngineGrantError::ProtocolViolation(
                "grant token is not canonical unpadded base64url".to_string(),
            ));
        }
        Ok(Self(Arc::from(value)))
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SecretGrantToken {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for SecretGrantToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretGrantToken([REDACTED])")
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedGrantControl {
    client: Client,
    grant_url: Arc<str>,
    token: SecretGrantToken,
}

impl PreparedGrantControl {
    pub(super) async fn promote(
        &self,
        binding: &DecoderGrantBinding,
    ) -> Result<(), EngineGrantError> {
        let request = BindingControlRequest::new(binding);
        let receipt = send_control(
            &self.client,
            &self.grant_url,
            &self.token,
            "promote",
            &request,
        )
        .await?;
        validate_control_receipt(
            &receipt,
            WireOperation::Promote,
            WireGrantState::Promoted,
            binding,
        )?;
        Ok(())
    }

    pub(super) async fn cancel(
        &self,
        binding: &DecoderGrantBinding,
    ) -> Result<EngineReleaseReceipt, EngineGrantError> {
        let request = BindingControlRequest::new(binding);
        let receipt = send_control(
            &self.client,
            &self.grant_url,
            &self.token,
            "cancel",
            &request,
        )
        .await?;
        validate_control_receipt(
            &receipt,
            WireOperation::Cancel,
            WireGrantState::Cancelled,
            binding,
        )?;
        release_receipt(receipt, binding, EngineReleaseKind::PreparedCancelled)
    }

    pub(super) async fn abort(
        &self,
        binding: &DecoderGrantBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineAbortOutcome, EngineGrantError> {
        abort_control(
            &self.client,
            &self.grant_url,
            &self.token,
            binding,
            reason_code,
            diagnostic,
        )
        .await
    }

    pub(super) async fn quarantine(
        &self,
        binding: &DecoderGrantBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineQuarantineReceipt, EngineGrantError> {
        quarantine_control(
            &self.client,
            &self.grant_url,
            &self.token,
            binding,
            reason_code,
            diagnostic,
        )
        .await
    }

    pub(super) fn into_retained(self) -> RetainedGrantControl {
        RetainedGrantControl {
            client: self.client,
            grant_url: self.grant_url,
            token: self.token,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct RetainedGrantControl {
    client: Client,
    grant_url: Arc<str>,
    token: SecretGrantToken,
}

impl RetainedGrantControl {
    pub(super) async fn complete(
        &self,
        binding: &DecoderGrantBinding,
    ) -> Result<EngineReleaseReceipt, EngineGrantError> {
        let request = BindingControlRequest::new(binding);
        let receipt = send_control(
            &self.client,
            &self.grant_url,
            &self.token,
            "complete",
            &request,
        )
        .await?;
        validate_control_receipt(
            &receipt,
            WireOperation::Complete,
            WireGrantState::Completed,
            binding,
        )?;
        release_receipt(receipt, binding, EngineReleaseKind::Completed)
    }

    pub(super) async fn abort(
        &self,
        binding: &DecoderGrantBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineAbortOutcome, EngineGrantError> {
        abort_control(
            &self.client,
            &self.grant_url,
            &self.token,
            binding,
            reason_code,
            diagnostic,
        )
        .await
    }

    pub(super) async fn quarantine(
        &self,
        binding: &DecoderGrantBinding,
        reason_code: &str,
        diagnostic: Option<&str>,
    ) -> Result<EngineQuarantineReceipt, EngineGrantError> {
        quarantine_control(
            &self.client,
            &self.grant_url,
            &self.token,
            binding,
            reason_code,
            diagnostic,
        )
        .await
    }
}

async fn abort_control(
    client: &Client,
    grant_url: &str,
    token: &SecretGrantToken,
    binding: &DecoderGrantBinding,
    reason_code: &str,
    diagnostic: Option<&str>,
) -> Result<EngineAbortOutcome, EngineGrantError> {
    validate_failure_context(reason_code, diagnostic)?;
    let request = FailureControlRequest {
        binding: BindingControlRequest::new(binding),
        reason_code,
        diagnostic,
    };
    let receipt = send_control(client, grant_url, token, "abort", &request).await?;
    match receipt.state {
        WireGrantState::Aborted => {
            validate_control_receipt(
                &receipt,
                WireOperation::Abort,
                WireGrantState::Aborted,
                binding,
            )?;
            Ok(EngineAbortOutcome::Aborted(release_receipt(
                receipt,
                binding,
                EngineReleaseKind::Aborted,
            )?))
        }
        WireGrantState::Quarantined => {
            validate_control_receipt(
                &receipt,
                WireOperation::Abort,
                WireGrantState::Quarantined,
                binding,
            )?;
            Ok(EngineAbortOutcome::Quarantined(quarantine_receipt(
                receipt, binding,
            )?))
        }
        state => Err(EngineGrantError::ProtocolViolation(format!(
            "abort returned state {state:?}, expected aborted or quarantined"
        ))),
    }
}

async fn quarantine_control(
    client: &Client,
    grant_url: &str,
    token: &SecretGrantToken,
    binding: &DecoderGrantBinding,
    reason_code: &str,
    diagnostic: Option<&str>,
) -> Result<EngineQuarantineReceipt, EngineGrantError> {
    validate_failure_context(reason_code, diagnostic)?;
    let request = FailureControlRequest {
        binding: BindingControlRequest::new(binding),
        reason_code,
        diagnostic,
    };
    let receipt = send_control(client, grant_url, token, "quarantine", &request).await?;
    validate_control_receipt(
        &receipt,
        WireOperation::Quarantine,
        WireGrantState::Quarantined,
        binding,
    )?;
    quarantine_receipt(receipt, binding)
}

fn validate_failure_context(
    reason_code: &str,
    diagnostic: Option<&str>,
) -> Result<(), EngineGrantError> {
    if reason_code.is_empty()
        || reason_code.len() > MAX_REASON_CODE_BYTES
        || !reason_code
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(EngineGrantError::InvalidGrant(format!(
            "failure reason code must be 1..={MAX_REASON_CODE_BYTES} ASCII bytes from [A-Za-z0-9_.-]"
        )));
    }
    if let Some(value) = diagnostic {
        validate_diagnostic(value)?;
    }
    Ok(())
}

fn validate_diagnostic(diagnostic: &str) -> Result<(), EngineGrantError> {
    if diagnostic.is_empty()
        || diagnostic.len() > MAX_DIAGNOSTIC_BYTES
        || diagnostic.chars().any(char::is_control)
    {
        return Err(EngineGrantError::InvalidGrant(format!(
            "failure diagnostic must be 1..={MAX_DIAGNOSTIC_BYTES} control-free UTF-8 bytes"
        )));
    }
    Ok(())
}

async fn send_control<T: Serialize + ?Sized>(
    client: &Client,
    grant_url: &str,
    token: &SecretGrantToken,
    operation_path: &'static str,
    request: &T,
) -> Result<WireControlReceipt, EngineGrantError> {
    let response = client
        .post(format!("{grant_url}/{operation_path}"))
        .bearer_auth(token.expose())
        .json(request)
        .send()
        .await
        .map_err(|error| EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: error.to_string(),
        })?;
    if !response.status().is_success() {
        return Err(EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: response_error_message(response).await,
        });
    }
    response
        .json()
        .await
        .map_err(|error| EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: format!("engine returned an invalid receipt: {error}"),
        })
}

fn validate_control_receipt(
    receipt: &WireControlReceipt,
    expected_operation: WireOperation,
    expected_state: WireGrantState,
    binding: &DecoderGrantBinding,
) -> Result<(), EngineGrantError> {
    validate_schema(receipt.schema_version)?;
    if receipt.grant_id != binding.grant_id()
        || receipt.logical_request_chain_id != binding.request_chain_id()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed grant or request-chain identity".to_string(),
        ));
    }
    validate_process(
        "prefill",
        &receipt.prefill_process,
        binding.prefill_id().url(),
        binding.prefill_id().instance_id(),
    )?;
    validate_bootstrap_endpoint(
        &receipt.prefill_bootstrap_endpoint,
        binding.prefill_bootstrap_endpoint(),
    )?;
    validate_process(
        "decoder",
        &receipt.decoder_process,
        binding.decoder_id().url(),
        binding.decoder_id().instance_id(),
    )?;
    if receipt.inference_route != binding.inference_route().as_str()
        || receipt.source_tp_size != binding.source_tp_size()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed inference route or source tensor parallelism".to_string(),
        ));
    }
    let child_request_ids: Vec<Uuid> = binding.child_request_ids().collect();
    let slots: Vec<Uuid> = binding
        .slot_generations()
        .iter()
        .map(|generation| generation.as_uuid())
        .collect();
    if receipt.child_request_ids != child_request_ids
        || receipt.decoder_slot_generations != slots
        || receipt.bootstrap_rooms != binding.bootstrap_rooms()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed ordered child, slot-generation, or room bindings".to_string(),
        ));
    }
    if DecoderGrantDigest::from_hex(&receipt.grant_digest)? != binding.digest() {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed the exact grant digest".to_string(),
        ));
    }
    if receipt.operation != expected_operation || receipt.state != expected_state {
        return Err(EngineGrantError::ProtocolViolation(format!(
            "control receipt returned operation/state {:?}/{:?}, expected {:?}/{:?}",
            receipt.operation, receipt.state, expected_operation, expected_state
        )));
    }
    if receipt.receipt_id.is_nil() {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt contains a nil identity".to_string(),
        ));
    }
    AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?;
    if !receipt.take_once {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt does not attest take-once reconciliation".to_string(),
        ));
    }
    Ok(())
}

fn release_receipt(
    receipt: WireControlReceipt,
    binding: &DecoderGrantBinding,
    kind: EngineReleaseKind,
) -> Result<EngineReleaseReceipt, EngineGrantError> {
    Ok(EngineReleaseReceipt::from_control(
        binding.grant_id(),
        binding.decoder_id().clone(),
        binding.child_request_ids().collect(),
        binding.prefill_bootstrap_endpoint().clone(),
        binding.slot_generations().to_vec(),
        binding.bootstrap_rooms().to_vec(),
        binding.digest(),
        kind,
        receipt.receipt_id,
        AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?,
        receipt.take_once,
    ))
}

fn quarantine_receipt(
    receipt: WireControlReceipt,
    binding: &DecoderGrantBinding,
) -> Result<EngineQuarantineReceipt, EngineGrantError> {
    Ok(EngineQuarantineReceipt::from_control(
        binding.grant_id(),
        binding.decoder_id().clone(),
        binding.child_request_ids().collect(),
        binding.prefill_bootstrap_endpoint().clone(),
        binding.digest(),
        receipt.receipt_id,
        AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?,
        receipt.take_once,
    ))
}

fn validate_schema(schema_version: u32) -> Result<(), EngineGrantError> {
    if schema_version != SCHEMA_VERSION {
        return Err(EngineGrantError::ProtocolViolation(format!(
            "decoder control schema version {schema_version} differs from {SCHEMA_VERSION}"
        )));
    }
    Ok(())
}

fn validate_process(
    name: &str,
    process: &WireProcessIdentity,
    expected_url: &str,
    expected_instance_id: Uuid,
) -> Result<(), EngineGrantError> {
    if process.url != expected_url || process.instance_id != expected_instance_id {
        return Err(EngineGrantError::ProtocolViolation(format!(
            "{name} process identity differs from the selected worker generation"
        )));
    }
    Ok(())
}

fn validate_bootstrap_endpoint(
    endpoint: &WireBootstrapEndpoint,
    expected: &PrefillBootstrapEndpoint,
) -> Result<(), EngineGrantError> {
    if endpoint.host != expected.host() || endpoint.port != expected.port() {
        return Err(EngineGrantError::ProtocolViolation(
            "prefill bootstrap endpoint differs from the selected generation".to_string(),
        ));
    }
    Ok(())
}

async fn response_error_message(response: reqwest::Response) -> String {
    let status = response.status();
    let body = match response.bytes().await {
        Ok(bytes) => {
            let limit = bytes.len().min(512);
            String::from_utf8_lossy(&bytes[..limit]).into_owned()
        }
        Err(error) => format!("failed to read error response: {error}"),
    };
    format!("HTTP {status}: {body}")
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WireProcessIdentity {
    url: String,
    instance_id: Uuid,
}

impl WireProcessIdentity {
    fn from_prefill(prefill_id: &PrefillId) -> Self {
        Self {
            url: prefill_id.url().to_string(),
            instance_id: prefill_id.instance_id(),
        }
    }

    fn from_decoder(decoder_id: &DecoderId) -> Self {
        Self {
            url: decoder_id.url().to_string(),
            instance_id: decoder_id.instance_id(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WireBootstrapEndpoint {
    host: String,
    port: u16,
}

impl From<&PrefillBootstrapEndpoint> for WireBootstrapEndpoint {
    fn from(endpoint: &PrefillBootstrapEndpoint) -> Self {
        Self {
            host: endpoint.host().to_string(),
            port: endpoint.port(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ReserveRequest<'a> {
    schema_version: u32,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: &'a str,
    request_body_json: &'a str,
    child_request_ids: &'a [Uuid],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReserveResponse {
    schema_version: u32,
    state: WireGrantState,
    grant_id: Uuid,
    grant_token: SecretGrantToken,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    grant_digest: String,
    allocations: Vec<WireAllocation>,
    prepared_expires_at_unix_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAllocation {
    child_request_id: Uuid,
    decoder_slot_generation: Uuid,
    bootstrap_room: u64,
    request_slot: u64,
    request_generation: u64,
    writer_manifest_digest: String,
    allocation_digest: String,
    reserved_kv_tokens: u64,
    remaining_decode_tokens: u64,
}

#[derive(Debug, Serialize)]
struct BindingControlRequest<'a> {
    schema_version: u32,
    grant_id: Uuid,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: &'a str,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: &'a [u64],
    grant_digest: String,
}

impl<'a> BindingControlRequest<'a> {
    fn new(binding: &'a DecoderGrantBinding) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            grant_id: binding.grant_id(),
            prefill_process: WireProcessIdentity::from_prefill(binding.prefill_id()),
            prefill_bootstrap_endpoint: WireBootstrapEndpoint::from(
                binding.prefill_bootstrap_endpoint(),
            ),
            decoder_process: WireProcessIdentity::from_decoder(binding.decoder_id()),
            logical_request_chain_id: binding.request_chain_id(),
            source_tp_size: binding.source_tp_size(),
            inference_route: binding.inference_route().as_str(),
            child_request_ids: binding.child_request_ids().collect(),
            decoder_slot_generations: binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect(),
            bootstrap_rooms: binding.bootstrap_rooms(),
            grant_digest: binding.digest().to_hex(),
        }
    }
}

#[derive(Debug, Serialize)]
struct FailureControlRequest<'a> {
    #[serde(flatten)]
    binding: BindingControlRequest<'a>,
    reason_code: &'a str,
    diagnostic: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WireOperation {
    Promote,
    Cancel,
    Complete,
    Abort,
    Quarantine,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WireGrantState {
    Prepared,
    Promoted,
    Cancelled,
    Completed,
    Aborted,
    Quarantined,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireControlReceipt {
    schema_version: u32,
    grant_id: Uuid,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: String,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: Vec<u64>,
    grant_digest: String,
    operation: WireOperation,
    state: WireGrantState,
    receipt_id: Uuid,
    receipt_digest: String,
    take_once: bool,
}

#[cfg(test)]
fn hex_digest(digest: AuthorityDigest) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest.as_bytes() {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing into String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, VecDeque},
        sync::{Arc, Mutex},
    };

    use axum::{body::Body, extract::State, http::Request, response::Response, Router};
    use http_body_util::BodyExt;
    use serde_json::json;
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::*;

    #[derive(Clone)]
    struct PlannedResponse {
        status: StatusCode,
        body: Value,
    }

    #[derive(Debug)]
    struct CapturedRequest {
        path: String,
        authorization: Option<String>,
        body: Value,
    }

    #[derive(Clone, Default)]
    struct TestServerState {
        responses: Arc<Mutex<HashMap<String, VecDeque<PlannedResponse>>>>,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
    }

    async fn control_handler(
        State(state): State<TestServerState>,
        request: Request<Body>,
    ) -> Response<Body> {
        let path = request.uri().path().to_string();
        let authorization = request
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .map(|value| value.to_str().unwrap().to_string());
        let body_bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
        state.requests.lock().unwrap().push(CapturedRequest {
            path: path.clone(),
            authorization,
            body,
        });
        let planned = state
            .responses
            .lock()
            .unwrap()
            .get_mut(&path)
            .and_then(VecDeque::pop_front)
            .expect("test server received an unplanned control request");
        Response::builder()
            .status(planned.status)
            .header("content-type", "application/json")
            .body(Body::from(planned.body.to_string()))
            .unwrap()
    }

    async fn start_server() -> (String, TestServerState, JoinHandle<()>) {
        let state = TestServerState::default();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let application = Router::new()
            .fallback(control_handler)
            .with_state(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, application).await.unwrap();
        });
        (format!("http://{address}"), state, task)
    }

    fn install_plans(state: &TestServerState, plans: HashMap<String, VecDeque<PlannedResponse>>) {
        *state.responses.lock().unwrap() = plans;
    }

    fn uuid(value: &str) -> Uuid {
        Uuid::parse_str(value).unwrap()
    }

    struct Fixture {
        reservation: DecoderGrantReservation,
        binding: DecoderGrantBinding,
        reserve_response: Value,
        token: String,
    }

    fn fixture(decoder_url: &str, child_count: usize) -> Fixture {
        let prefill_id = PrefillId::new(
            "http://prefill.test:30000",
            uuid("10000000-0000-4000-8000-000000000001"),
        )
        .unwrap();
        let decoder_id =
            DecoderId::new(decoder_url, uuid("20000000-0000-4000-8000-000000000002")).unwrap();
        let bootstrap = PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 42000).unwrap();
        let chain_id = uuid("30000000-0000-4000-8000-000000000003");
        let grant_id = uuid("40000000-0000-4000-8000-000000000004");
        let child_request_ids: Vec<Uuid> = (0..child_count)
            .map(|index| Uuid::from_u128(0x50000000000040008000000000000000 + index as u128 + 5))
            .collect();
        let rid = if child_count == 1 {
            Value::String(child_request_ids[0].to_string())
        } else {
            Value::Array(
                child_request_ids
                    .iter()
                    .map(|value| Value::String(value.to_string()))
                    .collect(),
            )
        };
        let request_body_json =
            serde_json::to_string(&json!({"prompt": "hello", "rid": rid})).unwrap();
        let reservation = DecoderGrantReservation::new(
            prefill_id.clone(),
            bootstrap.clone(),
            decoder_id.clone(),
            chain_id,
            2,
            Duration::from_secs(2),
            DecoderInferenceRoute::Generate,
            request_body_json.clone(),
            child_request_ids.clone(),
        )
        .unwrap();

        let mut allocation_json = Vec::new();
        let mut children = Vec::new();
        for (index, child_request_id) in child_request_ids.iter().copied().enumerate() {
            let digest_byte = u8::try_from(index + 1).unwrap();
            let slot_generation =
                Uuid::from_u128(0x60000000000040008000000000000000 + index as u128 + 6);
            let writer_digest = AuthorityDigest([digest_byte; 32]);
            let allocation_digest = AuthorityDigest([digest_byte + 16; 32]);
            children.push(
                DecoderGrantChildBinding::new(
                    child_request_id,
                    DecoderSlotGeneration::new(slot_generation),
                    100 + index as u64,
                    200 + index as u64,
                    300 + index as u64,
                    writer_digest,
                    allocation_digest,
                    DecoderGrantChildAccounting::new(400 + index, 500 + index),
                )
                .unwrap(),
            );
            allocation_json.push(json!({
                "child_request_id": child_request_id,
                "decoder_slot_generation": slot_generation,
                "bootstrap_room": 100 + index as u64,
                "request_slot": 200 + index as u64,
                "request_generation": 300 + index as u64,
                "writer_manifest_digest": hex_digest(writer_digest),
                "allocation_digest": hex_digest(allocation_digest),
                "reserved_kv_tokens": 400 + index,
                "remaining_decode_tokens": 500 + index,
            }));
        }
        let binding = DecoderGrantBinding::new(
            grant_id,
            DecoderInferenceRoute::Generate,
            &request_body_json,
            prefill_id.clone(),
            bootstrap.clone(),
            chain_id,
            2,
            decoder_id.clone(),
            children,
        )
        .unwrap();
        let token = URL_SAFE_NO_PAD.encode([0xA5; GRANT_TOKEN_BYTES]);
        let reserve_response = json!({
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "grant_id": grant_id,
            "grant_token": token,
            "prefill_process": WireProcessIdentity::from_prefill(&prefill_id),
            "prefill_bootstrap_endpoint": WireBootstrapEndpoint::from(&bootstrap),
            "decoder_process": WireProcessIdentity::from_decoder(&decoder_id),
            "logical_request_chain_id": chain_id,
            "grant_digest": binding.digest().to_hex(),
            "allocations": allocation_json,
            "prepared_expires_at_unix_ms": 1_900_000_000_000u64,
        });
        Fixture {
            reservation,
            binding,
            reserve_response,
            token,
        }
    }

    fn receipt(
        binding: &DecoderGrantBinding,
        operation: WireOperation,
        state: WireGrantState,
    ) -> Value {
        json!({
            "schema_version": SCHEMA_VERSION,
            "grant_id": binding.grant_id(),
            "prefill_process": WireProcessIdentity::from_prefill(binding.prefill_id()),
            "prefill_bootstrap_endpoint":
                WireBootstrapEndpoint::from(binding.prefill_bootstrap_endpoint()),
            "decoder_process": WireProcessIdentity::from_decoder(binding.decoder_id()),
            "logical_request_chain_id": binding.request_chain_id(),
            "source_tp_size": binding.source_tp_size(),
            "inference_route": binding.inference_route().as_str(),
            "child_request_ids": binding.child_request_ids().collect::<Vec<_>>(),
            "decoder_slot_generations": binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect::<Vec<_>>(),
            "bootstrap_rooms": binding.bootstrap_rooms(),
            "grant_digest": binding.digest().to_hex(),
            "operation": operation,
            "state": state,
            "receipt_id": uuid("70000000-0000-4000-8000-000000000007"),
            "receipt_digest": hex_digest(AuthorityDigest([0x77; 32])),
            "take_once": true,
        })
    }

    fn response(body: Value) -> PlannedResponse {
        PlannedResponse {
            status: StatusCode::OK,
            body,
        }
    }

    fn plan(
        entries: impl IntoIterator<Item = (String, PlannedResponse)>,
    ) -> HashMap<String, VecDeque<PlannedResponse>> {
        let mut plans: HashMap<String, VecDeque<PlannedResponse>> = HashMap::new();
        for (path, response) in entries {
            plans.entry(path).or_default().push_back(response);
        }
        plans
    }

    #[test]
    fn reservation_requires_exact_scalar_and_batched_rids() {
        let prefill_id = PrefillId::new("http://prefill.test:30000", Uuid::new_v4()).unwrap();
        let decoder_id = DecoderId::new("http://decoder.test:30001", Uuid::new_v4()).unwrap();
        let bootstrap = PrefillBootstrapEndpoint::new("bootstrap.test", 40000).unwrap();
        let child_0 = Uuid::new_v4();
        let child_1 = Uuid::new_v4();
        let make = |body: &str, children: Vec<Uuid>| {
            DecoderGrantReservation::new(
                prefill_id.clone(),
                bootstrap.clone(),
                decoder_id.clone(),
                Uuid::new_v4(),
                4,
                Duration::from_secs(1),
                DecoderInferenceRoute::ChatCompletions,
                body,
                children,
            )
        };

        assert!(make(&format!(r#"{{"rid":"{child_0}"}}"#), vec![child_0]).is_ok());
        assert!(make(
            &format!(r#"{{"rid":["{child_0}","{child_1}"]}}"#),
            vec![child_0, child_1]
        )
        .is_ok());
        assert!(make(&format!(r#"{{"rid":["{child_0}"]}}"#), vec![child_0]).is_err());
        assert!(make(
            &format!(r#"{{"rid":["{child_1}","{child_0}"]}}"#),
            vec![child_0, child_1]
        )
        .is_err());
        assert!(make(r#"{"prompt":"missing"}"#, vec![child_0]).is_err());
    }

    #[tokio::test]
    async fn reserve_promote_complete_uses_secret_header_only() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 2);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let complete_path = format!("{CONTROL_PATH}/{grant_id}/complete");
        let plans = plan([
            (
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()),
            ),
            (
                promote_path.clone(),
                response(receipt(
                    &fixture.binding,
                    WireOperation::Promote,
                    WireGrantState::Promoted,
                )),
            ),
            (
                complete_path.clone(),
                response(receipt(
                    &fixture.binding,
                    WireOperation::Complete,
                    WireGrantState::Completed,
                )),
            ),
        ]);
        install_plans(&state, plans);

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let debug = format!("{grant:?}");
        assert!(!debug.contains(&fixture.token));
        let token_debug = format!(
            "{:?}",
            SecretGrantToken::new(fixture.token.clone()).unwrap()
        );
        assert_eq!(token_debug, "SecretGrantToken([REDACTED])");
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let release = retained.complete().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::Completed);

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].path, promote_path);
        assert_eq!(requests[2].path, complete_path);
        for request in &requests[1..] {
            assert_eq!(
                request.authorization.as_deref(),
                Some(format!("Bearer {}", fixture.token).as_str())
            );
            let serialized = serde_json::to_string(&request.body).unwrap();
            assert!(!serialized.contains(&fixture.token));
            assert!(request.body.get("grant_token").is_none());
        }
        task.abort();
    }

    #[tokio::test]
    async fn prepared_cancel_returns_exact_release_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let plans = plan([
            (
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()),
            ),
            (
                format!("{CONTROL_PATH}/{grant_id}/cancel"),
                response(receipt(
                    &fixture.binding,
                    WireOperation::Cancel,
                    WireGrantState::Cancelled,
                )),
            ),
        ]);
        install_plans(&state, plans);
        let mut grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let release = grant.cancel().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::PreparedCancelled);
        assert_eq!(
            release.child_request_ids(),
            fixture.reservation.child_request_ids()
        );
        task.abort();
    }

    #[tokio::test]
    async fn prepared_cancel_retries_after_malformed_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let cancel_path = format!("{CONTROL_PATH}/{grant_id}/cancel");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    cancel_path.clone(),
                    response(json!({"schema_version": SCHEMA_VERSION})),
                ),
                (
                    cancel_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Cancel,
                        WireGrantState::Cancelled,
                    )),
                ),
            ]),
        );

        let mut grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        assert!(matches!(
            grant.cancel().await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "cancel",
                ..
            })
        ));
        assert!(format!("{grant:?}").contains("has_control: true"));
        let release = grant.cancel().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::PreparedCancelled);
        assert_eq!(state.requests.lock().unwrap().len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn promotion_retries_after_malformed_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    promote_path.clone(),
                    response(json!({"schema_version": SCHEMA_VERSION})),
                ),
                (
                    promote_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
            ]),
        );

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        assert!(matches!(
            promotion.reconcile_promotion().await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "promote",
                ..
            })
        ));
        assert!(format!("{promotion:?}").contains("has_control: true"));
        let retained = promotion.reconcile_promotion().await.unwrap();
        assert_eq!(retained.grant_id(), fixture.binding.grant_id());
        assert_eq!(state.requests.lock().unwrap().len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn promotion_ambiguity_can_abort_but_cannot_cancel() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let abort_path = format!("{CONTROL_PATH}/{grant_id}/abort");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    promote_path.clone(),
                    response(json!({"schema_version": SCHEMA_VERSION})),
                ),
                (
                    abort_path.clone(),
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Abort,
                        WireGrantState::Aborted,
                    )),
                ),
            ]),
        );

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        assert!(promotion.reconcile_promotion().await.is_err());
        let outcome = promotion
            .abort(
                "promotion_receipt_lost",
                Some("promotion outcome requires quiescence"),
            )
            .await
            .unwrap();
        assert!(matches!(outcome, EngineAbortOutcome::Aborted(_)));
        let paths: Vec<String> = state
            .requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![format!("{CONTROL_PATH}/reserve"), promote_path, abort_path,]
        );
        task.abort();
    }

    async fn run_abort_state(state: WireGrantState) -> EngineAbortOutcome {
        let (server_url, server_state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let plans = plan([
            (
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()),
            ),
            (
                format!("{CONTROL_PATH}/{grant_id}/promote"),
                response(receipt(
                    &fixture.binding,
                    WireOperation::Promote,
                    WireGrantState::Promoted,
                )),
            ),
            (
                format!("{CONTROL_PATH}/{grant_id}/abort"),
                response(receipt(&fixture.binding, WireOperation::Abort, state)),
            ),
        ]);
        install_plans(&server_state, plans);
        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let outcome = retained
            .abort("decoder_response_failed", Some("upstream body failed"))
            .await
            .unwrap();
        task.abort();
        outcome
    }

    #[tokio::test]
    async fn abort_releases_only_with_authoritative_aborted_state() {
        let aborted = run_abort_state(WireGrantState::Aborted).await;
        assert!(matches!(aborted, EngineAbortOutcome::Aborted(_)));
        let quarantined = run_abort_state(WireGrantState::Quarantined).await;
        assert!(matches!(quarantined, EngineAbortOutcome::Quarantined(_)));
    }

    #[tokio::test]
    async fn explicit_quarantine_returns_retention_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    format!("{CONTROL_PATH}/{grant_id}/promote"),
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
                (
                    format!("{CONTROL_PATH}/{grant_id}/quarantine"),
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Quarantine,
                        WireGrantState::Quarantined,
                    )),
                ),
            ]),
        );
        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let quarantine = retained
            .quarantine("response_body_dropped", Some("client disconnected"))
            .await
            .unwrap();
        assert_eq!(quarantine.grant_id(), fixture.binding.grant_id());
        assert_eq!(
            quarantine.child_request_ids(),
            fixture.reservation.child_request_ids()
        );
        task.abort();
    }

    #[tokio::test]
    async fn retained_complete_retries_after_malformed_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let complete_path = format!("{CONTROL_PATH}/{grant_id}/complete");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    promote_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
                (
                    complete_path.clone(),
                    response(json!({"schema_version": SCHEMA_VERSION})),
                ),
                (
                    complete_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Complete,
                        WireGrantState::Completed,
                    )),
                ),
            ]),
        );

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        assert!(retained.complete().await.is_err());
        assert!(format!("{retained:?}").contains("has_control: true"));
        let release = retained.complete().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::Completed);
        assert_eq!(state.requests.lock().unwrap().len(), 4);
        task.abort();
    }

    #[tokio::test]
    async fn retained_abort_retries_after_invalid_receipt() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let abort_path = format!("{CONTROL_PATH}/{grant_id}/abort");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    promote_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
                (
                    abort_path.clone(),
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Complete,
                        WireGrantState::Completed,
                    )),
                ),
                (
                    abort_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Abort,
                        WireGrantState::Aborted,
                    )),
                ),
            ]),
        );

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        assert!(retained
            .abort("response_failed", Some("first receipt was invalid"))
            .await
            .is_err());
        assert!(format!("{retained:?}").contains("has_control: true"));
        let outcome = retained
            .abort("response_failed", Some("retry exact abort"))
            .await
            .unwrap();
        assert!(matches!(outcome, EngineAbortOutcome::Aborted(_)));
        assert_eq!(state.requests.lock().unwrap().len(), 4);
        task.abort();
    }

    #[tokio::test]
    async fn retained_quarantine_retries_after_http_ambiguity() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let quarantine_path = format!("{CONTROL_PATH}/{grant_id}/quarantine");
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    promote_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
                (
                    quarantine_path.clone(),
                    PlannedResponse {
                        status: StatusCode::SERVICE_UNAVAILABLE,
                        body: json!({"error": "receipt delivery failed"}),
                    },
                ),
                (
                    quarantine_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Quarantine,
                        WireGrantState::Quarantined,
                    )),
                ),
            ]),
        );

        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        assert!(matches!(
            retained
                .quarantine("response_dropped", Some("first response was ambiguous"))
                .await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "quarantine",
                ..
            })
        ));
        assert!(format!("{retained:?}").contains("has_control: true"));
        let quarantine = retained
            .quarantine("response_dropped", Some("retry exact quarantine"))
            .await
            .unwrap();
        assert_eq!(quarantine.grant_id(), fixture.binding.grant_id());
        assert_eq!(state.requests.lock().unwrap().len(), 4);
        task.abort();
    }

    #[tokio::test]
    async fn dropping_grant_capabilities_emits_no_control_request() {
        let (server_url, state, task) = start_server().await;
        let prepared_fixture = fixture(&server_url, 1);
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                response(prepared_fixture.reserve_response.clone()),
            )]),
        );
        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&prepared_fixture.reservation)
            .await
            .unwrap();
        drop(grant);
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 1);
        task.abort();

        let (server_url, state, task) = start_server().await;
        let promotion_fixture = fixture(&server_url, 1);
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                response(promotion_fixture.reserve_response.clone()),
            )]),
        );
        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&promotion_fixture.reservation)
            .await
            .unwrap();
        drop(grant.begin_promotion());
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 1);
        task.abort();

        let (server_url, state, task) = start_server().await;
        let retained_fixture = fixture(&server_url, 1);
        let grant_id = retained_fixture.binding.grant_id();
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(retained_fixture.reserve_response.clone()),
                ),
                (
                    format!("{CONTROL_PATH}/{grant_id}/promote"),
                    response(receipt(
                        &retained_fixture.binding,
                        WireOperation::Promote,
                        WireGrantState::Promoted,
                    )),
                ),
            ]),
        );
        let grant = DecoderGrantControlClient::new(Client::new())
            .reserve(&retained_fixture.reservation)
            .await
            .unwrap();
        let mut promotion = grant.begin_promotion();
        let retained = promotion.reconcile_promotion().await.unwrap();
        drop(retained);
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 2);
        task.abort();
    }

    #[tokio::test]
    async fn reserve_refusal_and_allocation_identity_mismatch_are_distinct() {
        let (server_url, state, task) = start_server().await;
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                PlannedResponse {
                    status: StatusCode::CONFLICT,
                    body: json!({"error": "capacity"}),
                },
            )]),
        );
        let refused_fixture = fixture(&server_url, 1);
        let refused = DecoderGrantControlClient::new(Client::new())
            .reserve(&refused_fixture.reservation)
            .await;
        assert!(matches!(
            refused,
            Err(EngineGrantError::AllocatorRefused(_))
        ));
        task.abort();

        let (server_url, state, task) = start_server().await;
        let mut mismatch_fixture = fixture(&server_url, 1);
        mismatch_fixture.reserve_response["allocations"][0]["child_request_id"] =
            json!(Uuid::new_v4());
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                response(mismatch_fixture.reserve_response.clone()),
            )]),
        );
        let mismatch = DecoderGrantControlClient::new(Client::new())
            .reserve(&mismatch_fixture.reservation)
            .await;
        assert!(matches!(
            mismatch,
            Err(EngineGrantError::ProtocolViolation(_))
        ));
        task.abort();
    }

    #[test]
    fn failure_context_is_bounded_and_control_free() {
        assert!(validate_failure_context("transport_failed", Some("peer closed")).is_ok());
        assert!(validate_failure_context("", None).is_err());
        assert!(validate_failure_context("not allowed", None).is_err());
        assert!(validate_failure_context(&"x".repeat(MAX_REASON_CODE_BYTES + 1), None).is_err());
        assert!(validate_failure_context("failed", Some("line\nbreak")).is_err());
        assert!(
            validate_failure_context("failed", Some(&"x".repeat(MAX_DIAGNOSTIC_BYTES + 1)))
                .is_err()
        );
    }

    #[test]
    fn control_receipt_cannot_change_child_or_bootstrap_binding() {
        let fixture = fixture("http://decoder.test:30001", 1);
        let mut changed_child = receipt(
            &fixture.binding,
            WireOperation::Promote,
            WireGrantState::Promoted,
        );
        changed_child["child_request_ids"][0] = json!(Uuid::new_v4());
        let changed_child: WireControlReceipt = serde_json::from_value(changed_child).unwrap();
        assert!(validate_control_receipt(
            &changed_child,
            WireOperation::Promote,
            WireGrantState::Promoted,
            &fixture.binding,
        )
        .is_err());

        let mut changed_bootstrap = receipt(
            &fixture.binding,
            WireOperation::Promote,
            WireGrantState::Promoted,
        );
        changed_bootstrap["prefill_bootstrap_endpoint"]["port"] = json!(42001);
        let changed_bootstrap: WireControlReceipt =
            serde_json::from_value(changed_bootstrap).unwrap();
        assert!(validate_control_receipt(
            &changed_bootstrap,
            WireOperation::Promote,
            WireGrantState::Promoted,
            &fixture.binding,
        )
        .is_err());
    }
}
