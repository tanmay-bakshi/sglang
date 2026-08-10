use std::{fmt, sync::Arc, time::Duration};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use bytes::{Bytes, BytesMut};
use reqwest::{
    header::{HeaderValue, ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
    redirect::Policy,
    Client, ClientBuilder, RequestBuilder, Response, StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::{
    digest_reserve_attempt, AuthorityDigest, DecoderGrantBinding, DecoderGrantChildAccounting,
    DecoderGrantChildBinding, DecoderGrantDigest, DecoderId, DecoderInferenceRoute,
    DecoderRequestShape, DecoderReservationDigest, DecoderReserveAttemptDigest,
    DecoderReserveRefusalDisposition, DecoderReserveRefusalReceipt, DecoderSlotGeneration,
    EngineAbortOutcome, EngineCompletionOutcome, EngineGrantError, EngineQuarantineReceipt,
    EngineReleaseKind, EngineReleaseReceipt, PrefillId, PreparedGrantCancellationReceipt,
    UnboundGrantBinding, UnboundPreparedGrant,
};
use crate::core::PrefillBootstrapEndpoint;

const SCHEMA_VERSION: u32 = 1;
const CONTROL_PATH: &str = "/_internal/pd/v1/decode-reservations";
const GRANT_TOKEN_BYTES: usize = 32;
const MAX_REASON_CODE_BYTES: usize = 64;
const MAX_DIAGNOSTIC_BYTES: usize = 512;
const MAX_CONTROL_RESPONSE_BYTES: usize = 512 * 1024;
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const IDENTITY_CONTENT_ENCODING: &str = "identity";
const RID_KEY: &str = "rid";
const BOOTSTRAP_HOST_KEY: &str = "bootstrap_host";
const BOOTSTRAP_PORT_KEY: &str = "bootstrap_port";
const BOOTSTRAP_ROOM_KEY: &str = "bootstrap_room";
const GATEWAY_OWNED_REQUEST_KEYS: [&str; 10] = [
    RID_KEY,
    BOOTSTRAP_HOST_KEY,
    BOOTSTRAP_PORT_KEY,
    BOOTSTRAP_ROOM_KEY,
    "bootstrap_pair_key",
    "decode_tp_size",
    "disagg_prefill_dp_rank",
    "routed_dp_rank",
    "data_parallel_rank",
    "http_worker_ipc",
];

/// Decoder process authorization used only for the initial reservation request.
#[derive(Clone)]
pub struct DecoderControlAuthorization(Arc<str>);

impl DecoderControlAuthorization {
    /// Validate one configured decoder API key without exposing it on failure.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineGrantError> {
        let value = value.into();
        if value.is_empty() || HeaderValue::from_str(&format!("Bearer {value}")).is_err() {
            return Err(EngineGrantError::InvalidGrant(
                "decoder_control_authorization_invalid".to_string(),
            ));
        }
        Ok(Self(Arc::from(value)))
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for DecoderControlAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DecoderControlAuthorization([REDACTED])")
    }
}

/// Validated client request before gateway identities are assigned.
#[derive(Clone)]
pub struct DecoderRequestTemplate {
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    input_count: usize,
    request_body: Map<String, Value>,
    original_body_bytes: usize,
}

impl fmt::Debug for DecoderRequestTemplate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecoderRequestTemplate")
            .field("inference_route", &self.inference_route)
            .field("request_shape", &self.request_shape)
            .field("input_count", &self.input_count)
            .field("original_body_bytes", &self.original_body_bytes)
            .finish_non_exhaustive()
    }
}

impl DecoderRequestTemplate {
    /// Parse and validate a client request without admitting topology metadata.
    pub fn new(
        inference_route: DecoderInferenceRoute,
        request_body: Bytes,
    ) -> Result<Self, EngineGrantError> {
        let original_body_bytes = request_body.len();
        let request_body_json: Value = serde_json::from_slice(&request_body).map_err(|error| {
            EngineGrantError::InvalidGrant(format!(
                "request template body is not valid JSON: {error}"
            ))
        })?;
        let request_body = request_body_json.as_object().cloned().ok_or_else(|| {
            EngineGrantError::InvalidGrant(
                "request template body must be a JSON object".to_string(),
            )
        })?;
        for key in GATEWAY_OWNED_REQUEST_KEYS {
            if request_body.contains_key(key) {
                return Err(EngineGrantError::InvalidGrant(format!(
                    "request template cannot contain gateway-owned field {key}"
                )));
            }
        }
        let (request_shape, input_count) = derive_request_shape(&request_body, inference_route)?;
        validate_no_parallel_sampling(&request_body, inference_route, request_shape, input_count)?;
        Ok(Self {
            inference_route,
            request_shape,
            input_count,
            request_body,
            original_body_bytes,
        })
    }

    /// Derived scalar or batch representation.
    pub fn request_shape(&self) -> DecoderRequestShape {
        self.request_shape
    }

    /// Closed inference route used to validate and dispatch this request.
    pub fn inference_route(&self) -> DecoderInferenceRoute {
        self.inference_route
    }

    /// Number of independently allocated child requests.
    pub fn child_count(&self) -> usize {
        self.input_count
    }

    /// Assign one exact attempt and inject its ordered child identities.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_reservation(
        &self,
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        decoder_id: DecoderId,
        logical_request_chain_id: Uuid,
        source_tp_size: usize,
        prepared_ttl: Duration,
    ) -> Result<DecoderGrantReservation, EngineGrantError> {
        let mut child_request_ids = Vec::with_capacity(self.input_count);
        let mut unique = std::collections::HashSet::with_capacity(self.input_count);
        while child_request_ids.len() < self.input_count {
            let child_request_id = Uuid::new_v4();
            if unique.insert(child_request_id) {
                child_request_ids.push(child_request_id);
            }
        }
        let rid = match self.request_shape {
            DecoderRequestShape::Scalar => Value::String(child_request_ids[0].to_string()),
            DecoderRequestShape::Batch => Value::Array(
                child_request_ids
                    .iter()
                    .map(|child_request_id| Value::String(child_request_id.to_string()))
                    .collect(),
            ),
        };
        let mut request_body = self.request_body.clone();
        request_body.insert(RID_KEY.to_string(), rid);
        let base_request_body = Bytes::from(
            serde_json::to_vec(&Value::Object(request_body)).map_err(|error| {
                EngineGrantError::InvalidGrant(format!(
                    "failed to serialize prepared request body: {error}"
                ))
            })?,
        );
        DecoderGrantReservation::from_prepared(
            prefill_id,
            prefill_bootstrap_endpoint,
            decoder_id,
            logical_request_chain_id,
            Uuid::new_v4(),
            source_tp_size,
            prepared_ttl,
            self.inference_route,
            self.request_shape,
            base_request_body,
            Arc::from(child_request_ids),
        )
    }
}

/// Exact immutable input to one batch-atomic decoder reservation.
pub struct DecoderGrantReservation {
    prefill_id: PrefillId,
    prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
    decoder_id: DecoderId,
    logical_request_chain_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: DecoderReserveAttemptDigest,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    base_request_body: Bytes,
    child_request_ids: Arc<[Uuid]>,
}

impl fmt::Debug for DecoderGrantReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecoderGrantReservation")
            .field("prefill_id", &self.prefill_id)
            .field("decoder_id", &self.decoder_id)
            .field("logical_request_chain_id", &self.logical_request_chain_id)
            .field("reservation_attempt_id", &self.reservation_attempt_id)
            .field("reserve_attempt_digest", &self.reserve_attempt_digest)
            .field("source_tp_size", &self.source_tp_size)
            .field("prepared_ttl_ms", &self.prepared_ttl_ms)
            .field("inference_route", &self.inference_route)
            .field("request_shape", &self.request_shape)
            .field("base_request_body_bytes", &self.base_request_body.len())
            .field("child_count", &self.child_request_ids.len())
            .finish_non_exhaustive()
    }
}

impl DecoderGrantReservation {
    #[allow(clippy::too_many_arguments)]
    fn from_prepared(
        prefill_id: PrefillId,
        prefill_bootstrap_endpoint: PrefillBootstrapEndpoint,
        decoder_id: DecoderId,
        logical_request_chain_id: Uuid,
        reservation_attempt_id: Uuid,
        source_tp_size: usize,
        prepared_ttl: Duration,
        inference_route: DecoderInferenceRoute,
        request_shape: DecoderRequestShape,
        base_request_body: Bytes,
        child_request_ids: Arc<[Uuid]>,
    ) -> Result<Self, EngineGrantError> {
        if logical_request_chain_id.is_nil() {
            return Err(EngineGrantError::InvalidGrant(
                "logical request-chain identity cannot be the nil UUID".to_string(),
            ));
        }
        if reservation_attempt_id.is_nil() {
            return Err(EngineGrantError::InvalidGrant(
                "reservation attempt identity cannot be the nil UUID".to_string(),
            ));
        }
        if !matches!(source_tp_size, 1 | 2 | 4 | 8) {
            return Err(EngineGrantError::InvalidGrant(
                "source tensor-parallel size must be 1, 2, 4, or 8".to_string(),
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
        let reserve_attempt_digest = digest_reserve_attempt(
            reservation_attempt_id,
            inference_route,
            request_shape,
            prepared_ttl_ms,
            &base_request_body,
            &prefill_id,
            &prefill_bootstrap_endpoint,
            logical_request_chain_id,
            source_tp_size,
            &decoder_id,
            &child_request_ids,
        );

        Ok(Self {
            prefill_id,
            prefill_bootstrap_endpoint,
            decoder_id,
            logical_request_chain_id,
            reservation_attempt_id,
            reserve_attempt_digest,
            source_tp_size,
            prepared_ttl_ms,
            inference_route,
            request_shape,
            base_request_body,
            child_request_ids,
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

    /// Stable idempotency identity for one allocator attempt.
    pub fn reservation_attempt_id(&self) -> Uuid {
        self.reservation_attempt_id
    }

    /// Digest of the exact idempotent reserve-attempt request.
    pub fn reserve_attempt_digest(&self) -> DecoderReserveAttemptDigest {
        self.reserve_attempt_digest
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

    /// Whether normalized request fields use scalar or array JSON.
    pub fn request_shape(&self) -> DecoderRequestShape {
        self.request_shape
    }

    /// Cheap clone of the exact RID-enriched provisional request bytes.
    pub fn base_request_body(&self) -> Bytes {
        self.base_request_body.clone()
    }

    fn base_request_body_json(&self) -> &str {
        std::str::from_utf8(&self.base_request_body)
            .expect("validated JSON request bodies are UTF-8")
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

fn derive_request_shape(
    request: &Map<String, Value>,
    inference_route: DecoderInferenceRoute,
) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    match inference_route {
        DecoderInferenceRoute::ChatCompletions => Ok((DecoderRequestShape::Scalar, 1)),
        DecoderInferenceRoute::Completions => {
            let prompt = request.get("prompt").ok_or_else(|| {
                EngineGrantError::InvalidGrant(
                    "completion request must contain prompt before reservation".to_string(),
                )
            })?;
            derive_text_or_token_shape(prompt, "completion prompt")
        }
        DecoderInferenceRoute::Generate => derive_generate_shape(request),
    }
}

fn derive_generate_shape(
    request: &Map<String, Value>,
) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    let inputs: Vec<(&str, &Value)> = ["text", "input_ids", "input_embeds"]
        .into_iter()
        .filter_map(|key| {
            request
                .get(key)
                .filter(|value| !value.is_null())
                .map(|value| (key, value))
        })
        .collect();
    if inputs.len() != 1 {
        return Err(EngineGrantError::InvalidGrant(
            "generate request must contain exactly one of text, input_ids, or input_embeds"
                .to_string(),
        ));
    }
    let (name, value) = inputs[0];
    match name {
        "text" => derive_generate_text_shape(value),
        "input_ids" => derive_token_ids_shape(value),
        "input_embeds" => derive_embedding_shape(value),
        _ => unreachable!("generate input names are closed above"),
    }
}

fn derive_generate_text_shape(
    value: &Value,
) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    if value.is_string() {
        return Ok((DecoderRequestShape::Scalar, 1));
    }
    let values = value.as_array().ok_or_else(|| {
        EngineGrantError::InvalidGrant(
            "generate text must be a string or a nonempty string batch".to_string(),
        )
    })?;
    if values.is_empty() || !values.iter().all(Value::is_string) {
        return Err(EngineGrantError::InvalidGrant(
            "generate text must be a string or a nonempty string batch".to_string(),
        ));
    }
    Ok((DecoderRequestShape::Batch, values.len()))
}

fn derive_token_ids_shape(value: &Value) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    let values = value.as_array().ok_or_else(|| {
        EngineGrantError::InvalidGrant(
            "input_ids must be a nonempty token array or homogeneous batch".to_string(),
        )
    })?;
    if values.is_empty() {
        return Err(EngineGrantError::InvalidGrant(
            "input_ids cannot be an empty array".to_string(),
        ));
    }
    if values.iter().all(is_token_id) {
        return Ok((DecoderRequestShape::Scalar, 1));
    }
    if values.iter().all(|entry| {
        entry
            .as_array()
            .is_some_and(|tokens| !tokens.is_empty() && tokens.iter().all(is_token_id))
    }) {
        return Ok((DecoderRequestShape::Batch, values.len()));
    }
    Err(EngineGrantError::InvalidGrant(
        "input_ids has a mixed or unsupported batch representation".to_string(),
    ))
}

fn derive_text_or_token_shape(
    value: &Value,
    name: &str,
) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    if value.is_string() {
        return Ok((DecoderRequestShape::Scalar, 1));
    }
    let values = value.as_array().ok_or_else(|| {
        EngineGrantError::InvalidGrant(format!(
            "{name} must be a string, token array, or homogeneous batch"
        ))
    })?;
    if values.is_empty() {
        return Err(EngineGrantError::InvalidGrant(format!(
            "{name} cannot be an empty array"
        )));
    }
    if values.iter().all(is_token_id) {
        return Ok((DecoderRequestShape::Scalar, 1));
    }
    if values.iter().all(Value::is_string)
        || values.iter().all(|entry| {
            entry
                .as_array()
                .is_some_and(|tokens| !tokens.is_empty() && tokens.iter().all(is_token_id))
        })
    {
        return Ok((DecoderRequestShape::Batch, values.len()));
    }
    Err(EngineGrantError::InvalidGrant(format!(
        "{name} has a mixed or unsupported batch representation"
    )))
}

fn is_token_id(value: &Value) -> bool {
    value.as_i64().is_some_and(|token_id| token_id >= 0)
}

fn derive_embedding_shape(value: &Value) -> Result<(DecoderRequestShape, usize), EngineGrantError> {
    let outer = value.as_array().ok_or_else(|| {
        EngineGrantError::InvalidGrant("input_embeds must be a nonempty array".to_string())
    })?;
    if outer.is_empty() {
        return Err(EngineGrantError::InvalidGrant(
            "input_embeds cannot be an empty array".to_string(),
        ));
    }
    let scalar = outer.iter().all(|token| {
        token.as_array().is_some_and(|embedding| {
            !embedding.is_empty() && embedding.iter().all(Value::is_number)
        })
    });
    if scalar {
        return Ok((DecoderRequestShape::Scalar, 1));
    }
    let batch = outer.iter().all(|sample| {
        sample.as_array().is_some_and(|tokens| {
            !tokens.is_empty()
                && tokens.iter().all(|token| {
                    token.as_array().is_some_and(|embedding| {
                        !embedding.is_empty() && embedding.iter().all(Value::is_number)
                    })
                })
        })
    });
    if batch {
        return Ok((DecoderRequestShape::Batch, outer.len()));
    }
    Err(EngineGrantError::InvalidGrant(
        "input_embeds has a mixed or unsupported batch representation".to_string(),
    ))
}

fn validate_no_parallel_sampling(
    request: &Map<String, Value>,
    inference_route: DecoderInferenceRoute,
    request_shape: DecoderRequestShape,
    input_count: usize,
) -> Result<(), EngineGrantError> {
    validate_sampling_count(request.get("n"))?;
    if inference_route == DecoderInferenceRoute::Completions {
        validate_sampling_count(request.get("best_of"))?;
    }
    if inference_route != DecoderInferenceRoute::Generate {
        return Ok(());
    }
    match request.get("sampling_params") {
        Some(Value::Object(parameters)) => validate_sampling_count(parameters.get("n"))?,
        Some(Value::Array(parameter_sets)) => {
            if request_shape != DecoderRequestShape::Batch {
                return Err(EngineGrantError::InvalidGrant(
                    "scalar generate requests cannot use per-input sampling_params".to_string(),
                ));
            }
            if parameter_sets.len() != input_count {
                return Err(EngineGrantError::InvalidGrant(format!(
                    "generate sampling_params count {} differs from input count {input_count}",
                    parameter_sets.len()
                )));
            }
            for parameters in parameter_sets {
                let parameters = parameters.as_object().ok_or_else(|| {
                    EngineGrantError::InvalidGrant(
                        "generate sampling_params batches must contain only objects".to_string(),
                    )
                })?;
                validate_sampling_count(parameters.get("n"))?;
            }
        }
        Some(Value::Null) | None => {}
        Some(_) => {
            return Err(EngineGrantError::InvalidGrant(
                "generate sampling_params must be an object or a per-input object batch"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_sampling_count(value: Option<&Value>) -> Result<(), EngineGrantError> {
    if value.is_none() || value == Some(&Value::Null) || value.and_then(Value::as_u64) == Some(1) {
        return Ok(());
    }
    Err(EngineGrantError::InvalidGrant(
        "prepared decoder grants currently require parallel sampling n=1".to_string(),
    ))
}

pub(super) fn build_bound_request(
    binding: &UnboundGrantBinding,
) -> Result<Bytes, EngineGrantError> {
    let mut request_body: Value =
        serde_json::from_slice(&binding.base_request_body()).map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "prepared request body is not valid JSON: {error}"
            ))
        })?;
    let request_object = request_body.as_object_mut().ok_or_else(|| {
        EngineGrantError::ProtocolViolation(
            "prepared request body is not a JSON object".to_string(),
        )
    })?;
    for key in [BOOTSTRAP_HOST_KEY, BOOTSTRAP_PORT_KEY, BOOTSTRAP_ROOM_KEY] {
        if request_object.contains_key(key) {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "prepared request body already contains gateway-owned field {key}"
            )));
        }
    }
    let child_count = binding.children().len();
    let host = shaped_value(
        binding.request_shape(),
        child_count,
        Value::String(binding.prefill_bootstrap_endpoint().host().to_string()),
    );
    let port = shaped_value(
        binding.request_shape(),
        child_count,
        Value::from(binding.prefill_bootstrap_endpoint().port()),
    );
    let rooms = match binding.request_shape() {
        DecoderRequestShape::Scalar => Value::from(binding.bootstrap_rooms()[0]),
        DecoderRequestShape::Batch => Value::Array(
            binding
                .bootstrap_rooms()
                .iter()
                .copied()
                .map(Value::from)
                .collect(),
        ),
    };
    for (key, value) in [
        (BOOTSTRAP_HOST_KEY, host),
        (BOOTSTRAP_PORT_KEY, port),
        (BOOTSTRAP_ROOM_KEY, rooms),
    ] {
        request_object.insert(key.to_string(), value);
    }
    serde_json::to_vec(&request_body)
        .map(Bytes::from)
        .map_err(|error| {
            EngineGrantError::ProtocolViolation(format!(
                "failed to serialize bound request body: {error}"
            ))
        })
}

fn shaped_value(shape: DecoderRequestShape, child_count: usize, value: Value) -> Value {
    match shape {
        DecoderRequestShape::Scalar => value,
        DecoderRequestShape::Batch => {
            Value::Array((0..child_count).map(|_| value.clone()).collect())
        }
    }
}

/// Concrete HTTP client for decoder reservation lifecycle authority.
#[derive(Clone, Debug)]
pub struct DecoderGrantControlClient {
    client: Client,
}

impl DecoderGrantControlClient {
    #[cfg(test)]
    /// Construct a dedicated control-plane client with safe test defaults.
    fn new() -> Result<Self, EngineGrantError> {
        Self::from_builder(Client::builder())
    }

    /// Construct a dedicated control-plane client from caller-supplied settings.
    ///
    /// Redirects and transparent response decompression are always disabled,
    /// regardless of the supplied builder. Control requests carry bearer
    /// authority and prompt-bearing transcripts, so these policies are
    /// invariants rather than caller choices.
    pub fn from_builder(builder: ClientBuilder) -> Result<Self, EngineGrantError> {
        Self::from_builder_with_timeout(builder, CONTROL_REQUEST_TIMEOUT)
    }

    fn from_builder_with_timeout(
        builder: ClientBuilder,
        request_timeout: Duration,
    ) -> Result<Self, EngineGrantError> {
        if request_timeout.is_zero() {
            return Err(EngineGrantError::InvalidGrant(
                "decoder_control_request_timeout_invalid".to_string(),
            ));
        }
        let client = builder
            .redirect(Policy::none())
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .timeout(request_timeout)
            .build()
            .map_err(|_| {
                EngineGrantError::InvalidGrant("decoder_control_client_build_failed".to_string())
            })?;
        Ok(Self { client })
    }

    /// Pin one exact reserve attempt before any allocator I/O.
    pub fn begin_authorized_reserve(
        &self,
        reservation: impl Into<Arc<DecoderGrantReservation>>,
        authorization: DecoderControlAuthorization,
    ) -> ReserveReconciliationGrant {
        ReserveReconciliationGrant {
            client: self.clone(),
            reservation: Some(reservation.into()),
            authorization,
            polled: false,
        }
    }

    #[cfg(test)]
    fn begin_reserve(
        &self,
        reservation: impl Into<Arc<DecoderGrantReservation>>,
    ) -> ReserveReconciliationGrant {
        self.begin_authorized_reserve(
            reservation,
            DecoderControlAuthorization::new("test-decoder-api-key")
                .expect("the test decoder API key must be a valid bearer value"),
        )
    }

    async fn reserve_once(
        &self,
        reservation: &DecoderGrantReservation,
        authorization: &DecoderControlAuthorization,
    ) -> Result<UnboundPreparedGrant, EngineGrantError> {
        let endpoint = format!("{}{CONTROL_PATH}/reserve", reservation.decoder_id.url());
        let request = ReserveRequest {
            schema_version: SCHEMA_VERSION,
            prefill_process: WireProcessIdentity::from_prefill(&reservation.prefill_id),
            prefill_bootstrap_endpoint: WireBootstrapEndpoint::from(
                &reservation.prefill_bootstrap_endpoint,
            ),
            decoder_process: WireProcessIdentity::from_decoder(&reservation.decoder_id),
            logical_request_chain_id: reservation.logical_request_chain_id,
            reservation_attempt_id: reservation.reservation_attempt_id,
            reserve_attempt_digest: reservation.reserve_attempt_digest.to_hex(),
            source_tp_size: reservation.source_tp_size,
            prepared_ttl_ms: reservation.prepared_ttl_ms,
            inference_route: reservation.inference_route.as_str(),
            request_shape: reservation.request_shape.as_str(),
            base_request_body_json: reservation.base_request_body_json(),
            child_request_ids: reservation.child_request_ids(),
        };
        let response = control_post(&self.client, endpoint)
            .bearer_auth(authorization.expose())
            .json(&request)
            .send()
            .await
            .map_err(|error| {
                EngineGrantError::AmbiguousReserve(request_failure_reason(&error).to_string())
            })?;
        let response = read_control_response(response)
            .await
            .map_err(|error| EngineGrantError::AmbiguousReserve(error.code().to_string()))?;
        if response.status() == StatusCode::CONFLICT
            || response.status() == StatusCode::TOO_MANY_REQUESTS
        {
            let receipt: WireReserveRefusalReceipt = serde_json::from_slice(response.body())
                .map_err(|_| {
                    EngineGrantError::AmbiguousReserve(
                        "invalid_reserve_refusal_receipt".to_string(),
                    )
                })?;
            let receipt = validate_reserve_refusal_receipt(receipt, reservation).map_err(|_| {
                EngineGrantError::AmbiguousReserve("invalid_reserve_refusal_receipt".to_string())
            })?;
            return Err(EngineGrantError::AllocatorRefused(Box::new(receipt)));
        }
        if !response.status().is_success() {
            return Err(EngineGrantError::AmbiguousReserve(format!(
                "http_status_{}_without_authoritative_receipt",
                response.status().as_u16()
            )));
        }
        let response: ReserveResponse = serde_json::from_slice(response.body()).map_err(|_| {
            EngineGrantError::AmbiguousReserve("invalid_prepared_receipt".to_string())
        })?;
        self.validate_reserve_response(reservation, response)
    }

    fn validate_reserve_response(
        &self,
        reservation: &DecoderGrantReservation,
        response: ReserveResponse,
    ) -> Result<UnboundPreparedGrant, EngineGrantError> {
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
        if response.reservation_attempt_id != reservation.reservation_attempt_id {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response changed the reservation attempt identity".to_string(),
            ));
        }
        if DecoderReserveAttemptDigest::from_hex(&response.reserve_attempt_digest)?
            != reservation.reserve_attempt_digest
        {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response changed the exact reserve-attempt digest".to_string(),
            ));
        }
        if response.source_tp_size != reservation.source_tp_size
            || response.prepared_ttl_ms != reservation.prepared_ttl_ms
            || response.inference_route != reservation.inference_route.as_str()
            || response.request_shape != reservation.request_shape.as_str()
        {
            return Err(EngineGrantError::ProtocolViolation(
                "reserve response changed route, shape, lease TTL, or source tensor parallelism"
                    .to_string(),
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
        let binding = UnboundGrantBinding::new(
            response.grant_id,
            reservation.reservation_attempt_id,
            reservation.reserve_attempt_digest,
            reservation.inference_route,
            reservation.request_shape,
            reservation.prepared_ttl_ms,
            response.prepared_expires_at_unix_ms,
            reservation.base_request_body(),
            reservation.prefill_id.clone(),
            reservation.prefill_bootstrap_endpoint.clone(),
            reservation.logical_request_chain_id,
            reservation.source_tp_size,
            reservation.decoder_id.clone(),
            children,
        )?;
        let engine_digest = DecoderReservationDigest::from_hex(&response.reservation_digest)?;
        if engine_digest != binding.digest() {
            return Err(EngineGrantError::ProtocolViolation(format!(
                "engine reservation digest {} differs from gateway digest {}",
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
        Ok(UnboundPreparedGrant::from_control(binding, control))
    }
}

fn validate_reserve_refusal_receipt(
    receipt: WireReserveRefusalReceipt,
    reservation: &DecoderGrantReservation,
) -> Result<DecoderReserveRefusalReceipt, EngineGrantError> {
    validate_schema(receipt.schema_version)?;
    if receipt.operation != WireOperation::Reserve || receipt.state != WireGrantState::Refused {
        return Err(EngineGrantError::ProtocolViolation(format!(
            "allocator refusal returned operation/state {:?}/{:?}, expected reserve/refused",
            receipt.operation, receipt.state
        )));
    }
    validate_process(
        "prefill",
        &receipt.prefill_process,
        reservation.prefill_id.url(),
        reservation.prefill_id.instance_id(),
    )?;
    validate_bootstrap_endpoint(
        &receipt.prefill_bootstrap_endpoint,
        &reservation.prefill_bootstrap_endpoint,
    )?;
    validate_process(
        "decoder",
        &receipt.decoder_process,
        reservation.decoder_id.url(),
        reservation.decoder_id.instance_id(),
    )?;
    if receipt.logical_request_chain_id != reservation.logical_request_chain_id
        || receipt.reservation_attempt_id != reservation.reservation_attempt_id
    {
        return Err(EngineGrantError::ProtocolViolation(
            "allocator refusal changed request-chain or reserve-attempt identity".to_string(),
        ));
    }
    if receipt.source_tp_size != reservation.source_tp_size
        || receipt.prepared_ttl_ms != reservation.prepared_ttl_ms
        || receipt.inference_route != reservation.inference_route.as_str()
        || receipt.request_shape != reservation.request_shape.as_str()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "allocator refusal changed route, shape, lease TTL, or source tensor parallelism"
                .to_string(),
        ));
    }
    if DecoderReserveAttemptDigest::from_hex(&receipt.reserve_attempt_digest)?
        != reservation.reserve_attempt_digest
    {
        return Err(EngineGrantError::ProtocolViolation(
            "allocator refusal changed the exact reserve-attempt digest".to_string(),
        ));
    }
    validate_failure_context(&receipt.reason_code, receipt.diagnostic.as_deref())?;
    if receipt.receipt_id.is_nil() {
        return Err(EngineGrantError::ProtocolViolation(
            "allocator refusal receipt contains a nil identity".to_string(),
        ));
    }
    let receipt_digest = AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?;
    if !receipt.take_once {
        return Err(EngineGrantError::ProtocolViolation(
            "allocator refusal does not attest a take-once attempt tombstone".to_string(),
        ));
    }
    let disposition = match receipt.disposition {
        WireReserveRefusalDisposition::RetrySameDecoder => {
            DecoderReserveRefusalDisposition::RetrySameDecoder
        }
        WireReserveRefusalDisposition::RetryAnotherDecoder => {
            DecoderReserveRefusalDisposition::RetryAnotherDecoder
        }
        WireReserveRefusalDisposition::Terminal => DecoderReserveRefusalDisposition::Terminal,
    };
    Ok(DecoderReserveRefusalReceipt {
        prefill_id: reservation.prefill_id.clone(),
        decoder_id: reservation.decoder_id.clone(),
        logical_request_chain_id: reservation.logical_request_chain_id,
        reservation_attempt_id: reservation.reservation_attempt_id,
        reserve_attempt_digest: reservation.reserve_attempt_digest,
        reason_code: receipt.reason_code,
        disposition,
        receipt_id: receipt.receipt_id,
        receipt_digest,
        take_once: receipt.take_once,
    })
}

/// Exact reserve attempt retained across every allocation-ambiguous outcome.
pub struct ReserveReconciliationGrant {
    client: DecoderGrantControlClient,
    reservation: Option<Arc<DecoderGrantReservation>>,
    authorization: DecoderControlAuthorization,
    polled: bool,
}

impl fmt::Debug for ReserveReconciliationGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReserveReconciliationGrant")
            .field("reservation", &self.reservation)
            .field("authorization", &self.authorization)
            .field("polled", &self.polled)
            .finish()
    }
}

impl ReserveReconciliationGrant {
    /// Gateway-issued idempotency identity reused by every reserve retry.
    pub fn reservation_attempt_id(&self) -> Result<Uuid, EngineGrantError> {
        Ok(self.reservation()?.reservation_attempt_id())
    }

    /// Reconcile the same exact reserve attempt without pivoting on ambiguity.
    pub async fn reconcile_reserve(&mut self) -> Result<UnboundPreparedGrant, EngineGrantError> {
        self.polled = true;
        let client = self.client.clone();
        let result = client
            .reserve_once(self.reservation()?, &self.authorization)
            .await;
        match result {
            Ok(grant) => {
                self.reservation = None;
                Ok(grant)
            }
            Err(error @ EngineGrantError::AllocatorRefused(_)) => {
                self.reservation = None;
                Err(error)
            }
            Err(error @ EngineGrantError::AmbiguousReserve(_)) => Err(error),
            Err(error) => Err(EngineGrantError::AmbiguousReserve(error.to_string())),
        }
    }

    fn reservation(&self) -> Result<&DecoderGrantReservation, EngineGrantError> {
        self.reservation.as_deref().ok_or_else(|| {
            EngineGrantError::ProtocolViolation(
                "reserve reconciliation has no exact attempt capability".to_string(),
            )
        })
    }
}

impl Drop for ReserveReconciliationGrant {
    fn drop(&mut self) {
        if !self.polled || self.reservation.is_none() {
            return;
        }
        let reservation = self
            .reservation
            .as_ref()
            .expect("checked reserve reconciliation ownership");
        tracing::warn!(
            reservation_attempt_id = %reservation.reservation_attempt_id(),
            decoder_id = %reservation.decoder_id(),
            "Reserve reconciliation capability was dropped after an allocation-ambiguous outcome"
        );
    }
}

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

#[derive(Debug)]
pub(super) struct PreparedGrantControl {
    client: Client,
    grant_url: Arc<str>,
    token: SecretGrantToken,
}

#[cfg(test)]
pub(super) fn test_prepared_grant_control(grant_id: Uuid) -> PreparedGrantControl {
    test_prepared_grant_control_at(grant_id, "http://127.0.0.1:9")
}

#[cfg(test)]
pub(super) fn test_prepared_grant_control_at(
    grant_id: Uuid,
    decoder_url: &str,
) -> PreparedGrantControl {
    let client = DecoderGrantControlClient::new()
        .expect("test decoder control client must use a valid transport configuration")
        .client;
    let token = SecretGrantToken::new(URL_SAFE_NO_PAD.encode([0xA5; GRANT_TOKEN_BYTES]))
        .expect("test decoder grant token must satisfy the production token contract");
    PreparedGrantControl {
        client,
        grant_url: Arc::from(format!("{decoder_url}{CONTROL_PATH}/{grant_id}")),
        token,
    }
}

impl PreparedGrantControl {
    pub(super) async fn bind(&self, binding: &DecoderGrantBinding) -> Result<(), EngineGrantError> {
        let response = control_post(&self.client, format!("{}/bind", self.grant_url))
            .bearer_auth(self.token.expose())
            .header(CONTENT_TYPE, "application/json")
            .body(binding.request_body())
            .send()
            .await
            .map_err(|error| EngineGrantError::AmbiguousControl {
                operation: "bind",
                message: request_failure_reason(&error).to_string(),
            })?;
        let response = read_control_response(response).await.map_err(|error| {
            EngineGrantError::AmbiguousControl {
                operation: "bind",
                message: error.code().to_string(),
            }
        })?;
        if !response.status().is_success() {
            return Err(EngineGrantError::AmbiguousControl {
                operation: "bind",
                message: http_status_reason(response.status()),
            });
        }
        let receipt: WireControlReceipt =
            serde_json::from_slice(response.body()).map_err(|_| {
                EngineGrantError::AmbiguousControl {
                    operation: "bind",
                    message: "invalid_control_receipt".to_string(),
                }
            })?;
        validate_control_receipt(
            &receipt,
            WireOperation::Bind,
            WireGrantState::Prepared,
            binding,
        )
    }

    pub(super) async fn cancel_unbound(
        &self,
        binding: &UnboundGrantBinding,
        attempted_binding: Option<&DecoderGrantBinding>,
    ) -> Result<PreparedGrantCancellationReceipt, EngineGrantError> {
        let request = UnboundCancellationRequest::new(binding, attempted_binding);
        let response = control_post(&self.client, format!("{}/cancel", self.grant_url))
            .bearer_auth(self.token.expose())
            .json(&request)
            .send()
            .await
            .map_err(|error| EngineGrantError::AmbiguousControl {
                operation: "cancel",
                message: request_failure_reason(&error).to_string(),
            })?;
        let response = read_control_response(response).await.map_err(|error| {
            EngineGrantError::AmbiguousControl {
                operation: "cancel",
                message: error.code().to_string(),
            }
        })?;
        if !response.status().is_success() {
            return Err(EngineGrantError::AmbiguousControl {
                operation: "cancel",
                message: http_status_reason(response.status()),
            });
        }
        let receipt: WireUnboundCancellationReceipt = serde_json::from_slice(response.body())
            .map_err(|_| EngineGrantError::AmbiguousControl {
                operation: "cancel",
                message: "invalid_control_receipt".to_string(),
            })?;
        validate_unbound_cancellation_receipt(&receipt, binding, attempted_binding)?;
        Ok(PreparedGrantCancellationReceipt::from_control(
            binding.grant_id(),
            binding.reservation_attempt_id(),
            binding.reserve_attempt_digest(),
            binding.decoder_id().clone(),
            binding.child_request_ids().collect(),
            binding.slot_generations().to_vec(),
            binding.bootstrap_rooms().to_vec(),
            binding.prepared_ttl_ms(),
            binding.prepared_expires_at_unix_ms(),
            binding.digest(),
            attempted_binding.map(DecoderGrantBinding::digest),
            receipt.receipt_id,
            AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?,
            receipt.take_once,
        ))
    }

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

#[derive(Debug)]
pub(super) struct RetainedGrantControl {
    client: Client,
    grant_url: Arc<str>,
    token: SecretGrantToken,
}

impl RetainedGrantControl {
    pub(super) async fn complete(
        &self,
        binding: &DecoderGrantBinding,
    ) -> Result<EngineCompletionOutcome, EngineGrantError> {
        let request = BindingControlRequest::new(binding);
        let receipt = send_control(
            &self.client,
            &self.grant_url,
            &self.token,
            "complete",
            &request,
        )
        .await?;
        match receipt.state {
            WireGrantState::Completed => {
                validate_control_receipt(
                    &receipt,
                    WireOperation::Complete,
                    WireGrantState::Completed,
                    binding,
                )?;
                Ok(EngineCompletionOutcome::Completed(release_receipt(
                    receipt,
                    binding,
                    EngineReleaseKind::Completed,
                )?))
            }
            WireGrantState::Quarantined => {
                validate_control_receipt(
                    &receipt,
                    WireOperation::Complete,
                    WireGrantState::Quarantined,
                    binding,
                )?;
                Ok(EngineCompletionOutcome::Quarantined(quarantine_receipt(
                    receipt, binding,
                )?))
            }
            state => Err(EngineGrantError::ProtocolViolation(format!(
                "complete returned state {state:?}, expected completed or quarantined"
            ))),
        }
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

pub(super) fn validate_failure_context(
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
    let response = control_post(client, format!("{grant_url}/{operation_path}"))
        .bearer_auth(token.expose())
        .json(request)
        .send()
        .await
        .map_err(|error| EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: request_failure_reason(&error).to_string(),
        })?;
    let response = read_control_response(response).await.map_err(|error| {
        EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: error.code().to_string(),
        }
    })?;
    if !response.status().is_success() {
        return Err(EngineGrantError::AmbiguousControl {
            operation: operation_path,
            message: http_status_reason(response.status()),
        });
    }
    serde_json::from_slice(response.body()).map_err(|_| EngineGrantError::AmbiguousControl {
        operation: operation_path,
        message: "invalid_control_receipt".to_string(),
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
        || receipt.reservation_attempt_id != binding.reservation_attempt_id()
        || receipt.logical_request_chain_id != binding.request_chain_id()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed grant, reservation-attempt, or request-chain identity"
                .to_string(),
        ));
    }
    if DecoderReserveAttemptDigest::from_hex(&receipt.reserve_attempt_digest)?
        != binding.reserve_attempt_digest()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed the exact reserve-attempt digest".to_string(),
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
        || receipt.request_shape != binding.request_shape().as_str()
        || receipt.source_tp_size != binding.source_tp_size()
        || receipt.prepared_ttl_ms != binding.prepared_ttl_ms()
        || receipt.prepared_expires_at_unix_ms != binding.prepared_expires_at_unix_ms()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed route, shape, lease, or source tensor parallelism".to_string(),
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
    if DecoderReservationDigest::from_hex(&receipt.reservation_digest)?
        != binding.reservation_digest()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "control receipt changed the exact reservation digest".to_string(),
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

fn validate_unbound_cancellation_receipt(
    receipt: &WireUnboundCancellationReceipt,
    binding: &UnboundGrantBinding,
    attempted_binding: Option<&DecoderGrantBinding>,
) -> Result<(), EngineGrantError> {
    validate_schema(receipt.schema_version)?;
    if receipt.grant_id != binding.grant_id()
        || receipt.reservation_attempt_id != binding.reservation_attempt_id()
        || receipt.logical_request_chain_id != binding.request_chain_id()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed grant, reservation-attempt, or request-chain identity"
                .to_string(),
        ));
    }
    if DecoderReserveAttemptDigest::from_hex(&receipt.reserve_attempt_digest)?
        != binding.reserve_attempt_digest()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed the exact reserve-attempt digest".to_string(),
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
        || receipt.request_shape != binding.request_shape().as_str()
        || receipt.source_tp_size != binding.source_tp_size()
        || receipt.prepared_ttl_ms != binding.prepared_ttl_ms()
        || receipt.prepared_expires_at_unix_ms != binding.prepared_expires_at_unix_ms()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed route, shape, lease, or source tensor parallelism"
                .to_string(),
        ));
    }
    let child_request_ids: Vec<Uuid> = binding.child_request_ids().collect();
    let slot_generations: Vec<Uuid> = binding
        .slot_generations()
        .iter()
        .map(|generation| generation.as_uuid())
        .collect();
    if receipt.child_request_ids != child_request_ids
        || receipt.decoder_slot_generations != slot_generations
        || receipt.bootstrap_rooms != binding.bootstrap_rooms()
    {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed ordered child, slot-generation, or room bindings"
                .to_string(),
        ));
    }
    if DecoderReservationDigest::from_hex(&receipt.reservation_digest)? != binding.digest() {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed the exact reservation digest".to_string(),
        ));
    }
    let expected_digest = attempted_binding.map(|value| value.digest().to_hex());
    if receipt.attempted_grant_digest != expected_digest {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt changed the attempted grant digest".to_string(),
        ));
    }
    if receipt.operation != WireOperation::Cancel || receipt.state != WireGrantState::Cancelled {
        return Err(EngineGrantError::ProtocolViolation(format!(
            "unbound cancellation receipt returned operation/state {:?}/{:?}, expected cancel/cancelled",
            receipt.operation, receipt.state
        )));
    }
    if receipt.receipt_id.is_nil() {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt contains a nil identity".to_string(),
        ));
    }
    AuthorityDigest::from_hex("receipt digest", &receipt.receipt_digest)?;
    if !receipt.take_once {
        return Err(EngineGrantError::ProtocolViolation(
            "unbound cancellation receipt does not attest take-once reconciliation".to_string(),
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
        binding.slot_generations().to_vec(),
        binding.bootstrap_rooms().to_vec(),
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

fn control_post(client: &Client, endpoint: String) -> RequestBuilder {
    client
        .post(endpoint)
        .header(ACCEPT_ENCODING, IDENTITY_CONTENT_ENCODING)
}

fn request_failure_reason(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        return "request_timeout";
    }
    if error.is_connect() {
        return "connection_failed";
    }
    if error.is_body() {
        return "request_body_failed";
    }
    "request_failed"
}

fn http_status_reason(status: StatusCode) -> String {
    format!("http_status_{}", status.as_u16())
}

struct ControlResponse {
    status: StatusCode,
    body: Bytes,
}

impl ControlResponse {
    fn status(&self) -> StatusCode {
        self.status
    }

    fn body(&self) -> &[u8] {
        &self.body
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlResponseError {
    UnsupportedContentEncoding,
    BodyTooLarge,
    BodyReadFailed,
}

impl ControlResponseError {
    fn code(self) -> &'static str {
        match self {
            Self::UnsupportedContentEncoding => "unsupported_content_encoding",
            Self::BodyTooLarge => "response_body_too_large",
            Self::BodyReadFailed => "response_body_read_failed",
        }
    }
}

async fn read_control_response(
    mut response: Response,
) -> Result<ControlResponse, ControlResponseError> {
    validate_content_encoding(&response)?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_CONTROL_RESPONSE_BYTES as u64)
    {
        return Err(ControlResponseError::BodyTooLarge);
    }

    let status = response.status();
    let initial_capacity = response
        .content_length()
        .unwrap_or(0)
        .min(MAX_CONTROL_RESPONSE_BYTES as u64) as usize;
    let mut body = BytesMut::with_capacity(initial_capacity);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| ControlResponseError::BodyReadFailed)?
    {
        if chunk.len() > MAX_CONTROL_RESPONSE_BYTES.saturating_sub(body.len()) {
            return Err(ControlResponseError::BodyTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(ControlResponse {
        status,
        body: body.freeze(),
    })
}

fn validate_content_encoding(response: &Response) -> Result<(), ControlResponseError> {
    let mut values = response.headers().get_all(CONTENT_ENCODING).iter();
    let Some(value) = values.next() else {
        return Ok(());
    };
    if values.next().is_some() {
        return Err(ControlResponseError::UnsupportedContentEncoding);
    }
    let encoding = value
        .to_str()
        .map_err(|_| ControlResponseError::UnsupportedContentEncoding)?;
    if !encoding.eq_ignore_ascii_case(IDENTITY_CONTENT_ENCODING) {
        return Err(ControlResponseError::UnsupportedContentEncoding);
    }
    Ok(())
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

#[derive(Serialize)]
struct ReserveRequest<'a> {
    schema_version: u32,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: &'a str,
    request_shape: &'a str,
    base_request_body_json: &'a str,
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
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: String,
    request_shape: String,
    reservation_digest: String,
    allocations: Vec<WireAllocation>,
    prepared_expires_at_unix_ms: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReserveRefusalReceipt {
    schema_version: u32,
    operation: WireOperation,
    state: WireGrantState,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    source_tp_size: usize,
    prepared_ttl_ms: u64,
    inference_route: String,
    request_shape: String,
    reason_code: String,
    diagnostic: Option<String>,
    disposition: WireReserveRefusalDisposition,
    receipt_id: Uuid,
    receipt_digest: String,
    take_once: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
enum WireReserveRefusalDisposition {
    RetrySameDecoder,
    RetryAnotherDecoder,
    Terminal,
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
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: &'a str,
    request_shape: &'a str,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: &'a [u64],
    reservation_digest: String,
    grant_digest: String,
}

impl<'a> BindingControlRequest<'a> {
    fn new(binding: &'a DecoderGrantBinding) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            grant_id: binding.grant_id(),
            reservation_attempt_id: binding.reservation_attempt_id(),
            reserve_attempt_digest: binding.reserve_attempt_digest().to_hex(),
            prefill_process: WireProcessIdentity::from_prefill(binding.prefill_id()),
            prefill_bootstrap_endpoint: WireBootstrapEndpoint::from(
                binding.prefill_bootstrap_endpoint(),
            ),
            decoder_process: WireProcessIdentity::from_decoder(binding.decoder_id()),
            logical_request_chain_id: binding.request_chain_id(),
            source_tp_size: binding.source_tp_size(),
            inference_route: binding.inference_route().as_str(),
            request_shape: binding.request_shape().as_str(),
            prepared_ttl_ms: binding.prepared_ttl_ms(),
            prepared_expires_at_unix_ms: binding.prepared_expires_at_unix_ms(),
            child_request_ids: binding.child_request_ids().collect(),
            decoder_slot_generations: binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect(),
            bootstrap_rooms: binding.bootstrap_rooms(),
            reservation_digest: binding.reservation_digest().to_hex(),
            grant_digest: binding.digest().to_hex(),
        }
    }
}

#[derive(Serialize)]
struct UnboundCancellationRequest<'a> {
    schema_version: u32,
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: &'a str,
    request_shape: &'a str,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: &'a [u64],
    reservation_digest: String,
    attempted_grant_digest: Option<String>,
}

impl<'a> UnboundCancellationRequest<'a> {
    fn new(
        binding: &'a UnboundGrantBinding,
        attempted_binding: Option<&'a DecoderGrantBinding>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            grant_id: binding.grant_id(),
            reservation_attempt_id: binding.reservation_attempt_id(),
            reserve_attempt_digest: binding.reserve_attempt_digest().to_hex(),
            prefill_process: WireProcessIdentity::from_prefill(binding.prefill_id()),
            prefill_bootstrap_endpoint: WireBootstrapEndpoint::from(
                binding.prefill_bootstrap_endpoint(),
            ),
            decoder_process: WireProcessIdentity::from_decoder(binding.decoder_id()),
            logical_request_chain_id: binding.request_chain_id(),
            source_tp_size: binding.source_tp_size(),
            inference_route: binding.inference_route().as_str(),
            request_shape: binding.request_shape().as_str(),
            prepared_ttl_ms: binding.prepared_ttl_ms(),
            prepared_expires_at_unix_ms: binding.prepared_expires_at_unix_ms(),
            child_request_ids: binding.child_request_ids().collect(),
            decoder_slot_generations: binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect(),
            bootstrap_rooms: binding.bootstrap_rooms(),
            reservation_digest: binding.digest().to_hex(),
            attempted_grant_digest: attempted_binding.map(|value| value.digest().to_hex()),
        }
    }
}

#[derive(Serialize)]
struct FailureControlRequest<'a> {
    #[serde(flatten)]
    binding: BindingControlRequest<'a>,
    reason_code: &'a str,
    diagnostic: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WireOperation {
    Reserve,
    Bind,
    Promote,
    Cancel,
    Complete,
    Abort,
    Quarantine,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WireGrantState {
    Refused,
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
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: String,
    request_shape: String,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: Vec<u64>,
    reservation_digest: String,
    grant_digest: String,
    operation: WireOperation,
    state: WireGrantState,
    receipt_id: Uuid,
    receipt_digest: String,
    take_once: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireUnboundCancellationReceipt {
    schema_version: u32,
    grant_id: Uuid,
    reservation_attempt_id: Uuid,
    reserve_attempt_digest: String,
    prefill_process: WireProcessIdentity,
    prefill_bootstrap_endpoint: WireBootstrapEndpoint,
    decoder_process: WireProcessIdentity,
    logical_request_chain_id: Uuid,
    source_tp_size: usize,
    inference_route: String,
    request_shape: String,
    prepared_ttl_ms: u64,
    prepared_expires_at_unix_ms: u64,
    child_request_ids: Vec<Uuid>,
    decoder_slot_generations: Vec<Uuid>,
    bootstrap_rooms: Vec<u64>,
    reservation_digest: String,
    attempted_grant_digest: Option<String>,
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
        cell::{Ref, RefCell},
        collections::{HashMap, VecDeque},
        convert::Infallible,
        future,
        sync::{Arc, Mutex},
    };

    use axum::{body::Body, extract::State, http::Request, response::Response, Router};
    use http_body_util::BodyExt;
    use serde_json::json;
    use tokio::{net::TcpListener, sync::Notify, task::JoinHandle};

    use super::{super::BoundPreparedGrant, *};
    use crate::{
        config::types::RetryConfig,
        core::{
            pd_decoder_directory::PdProcessDirectory, BasicWorkerBuilder, HttpOrigin,
            KvTransferProtocol, PdMetadataSchema, PdProcessMetadata, PdProcessRegistration,
            PdProcessRole, PdReservedRequestSession, PdTopology, PrefillBootstrapEndpoint,
            PreparedGrantProtocol, Worker, WorkerType,
        },
    };

    const PROMPT_SECRET: &str = "prompt-secret-never-log-7d33f2";

    #[derive(Clone)]
    enum PlannedBody {
        Full(Bytes),
        Chunked(Vec<Bytes>),
        Pending,
    }

    #[derive(Clone)]
    struct PlannedResponse {
        status: StatusCode,
        headers: Vec<(String, String)>,
        body: PlannedBody,
    }

    impl PlannedResponse {
        fn json(body: Value) -> Self {
            Self {
                status: StatusCode::OK,
                headers: vec![("content-type".to_string(), "application/json".to_string())],
                body: PlannedBody::Full(Bytes::from(body.to_string())),
            }
        }

        fn raw(body: Bytes) -> Self {
            Self {
                status: StatusCode::OK,
                headers: Vec::new(),
                body: PlannedBody::Full(body),
            }
        }

        fn chunked(chunks: Vec<Bytes>) -> Self {
            Self {
                status: StatusCode::OK,
                headers: Vec::new(),
                body: PlannedBody::Chunked(chunks),
            }
        }

        fn pending() -> Self {
            Self {
                status: StatusCode::OK,
                headers: Vec::new(),
                body: PlannedBody::Pending,
            }
        }

        fn with_status(mut self, status: StatusCode) -> Self {
            self.status = status;
            self
        }

        fn with_header(mut self, name: &str, value: &str) -> Self {
            self.headers.push((name.to_string(), value.to_string()));
            self
        }
    }

    #[derive(Debug)]
    struct CapturedRequest {
        path: String,
        authorization: Option<String>,
        accept_encoding: Option<String>,
        body_bytes: Bytes,
        body: Value,
    }

    struct SessionGrant {
        binding: DecoderGrantBinding,
        final_request_body: Bytes,
        reserve_response: Value,
    }

    struct SessionEngine {
        grant: Option<SessionGrant>,
        terminal: Arc<Notify>,
    }

    impl SessionEngine {
        fn new(terminal: Arc<Notify>) -> Self {
            Self {
                grant: None,
                terminal,
            }
        }

        fn handle(&mut self, path: &str, body_bytes: &Bytes, body: &Value) -> PlannedResponse {
            if path == format!("{CONTROL_PATH}/reserve") {
                if self.grant.is_none() {
                    self.grant = Some(build_session_grant(body));
                }
                return response(
                    self.grant
                        .as_ref()
                        .expect("session reserve installed a grant")
                        .reserve_response
                        .clone(),
                );
            }

            let grant = self
                .grant
                .as_ref()
                .expect("session control operation requires a reserved grant");
            let grant_id = grant.binding.grant_id();
            if path == format!("{CONTROL_PATH}/{grant_id}/bind") {
                assert_eq!(body_bytes, &grant.final_request_body);
                return response(receipt(
                    &grant.binding,
                    WireOperation::Bind,
                    WireGrantState::Prepared,
                ));
            }
            if path == format!("{CONTROL_PATH}/{grant_id}/promote") {
                return response(receipt(
                    &grant.binding,
                    WireOperation::Promote,
                    WireGrantState::Promoted,
                ));
            }
            if path == format!("{CONTROL_PATH}/{grant_id}/complete") {
                self.terminal.notify_one();
                return response(receipt(
                    &grant.binding,
                    WireOperation::Complete,
                    WireGrantState::Completed,
                ));
            }
            if path == format!("{CONTROL_PATH}/{grant_id}/abort") {
                self.terminal.notify_one();
                return response(receipt(
                    &grant.binding,
                    WireOperation::Abort,
                    WireGrantState::Aborted,
                ));
            }
            panic!("session engine received unexpected control path {path}");
        }
    }

    #[derive(Clone, Default)]
    struct TestServerState {
        responses: Arc<Mutex<HashMap<String, VecDeque<PlannedResponse>>>>,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        session_engine: Arc<Mutex<Option<SessionEngine>>>,
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
        let accept_encoding = request
            .headers()
            .get(ACCEPT_ENCODING)
            .map(|value| value.to_str().unwrap().to_string());
        let body_bytes = request.into_body().collect().await.unwrap().to_bytes();
        let body = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
        state.requests.lock().unwrap().push(CapturedRequest {
            path: path.clone(),
            authorization,
            accept_encoding,
            body_bytes: body_bytes.clone(),
            body: body.clone(),
        });
        let planned = state
            .responses
            .lock()
            .unwrap()
            .get_mut(&path)
            .and_then(VecDeque::pop_front)
            .or_else(|| {
                state
                    .session_engine
                    .lock()
                    .unwrap()
                    .as_mut()
                    .map(|engine| engine.handle(&path, &body_bytes, &body))
            })
            .expect("test server received an unplanned control request");
        let mut response = Response::builder().status(planned.status);
        for (name, value) in &planned.headers {
            response = response.header(name.as_str(), value.as_str());
        }
        let body = match planned.body {
            PlannedBody::Full(body) => Body::from(body),
            PlannedBody::Chunked(chunks) => Body::from_stream(tokio_stream::iter(
                chunks.into_iter().map(Ok::<Bytes, Infallible>),
            )),
            PlannedBody::Pending => future::pending::<Body>().await,
        };
        response.body(body).unwrap()
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

    fn build_session_grant(request: &Value) -> SessionGrant {
        let prefill_process: WireProcessIdentity =
            serde_json::from_value(request["prefill_process"].clone()).unwrap();
        let decoder_process: WireProcessIdentity =
            serde_json::from_value(request["decoder_process"].clone()).unwrap();
        let bootstrap: WireBootstrapEndpoint =
            serde_json::from_value(request["prefill_bootstrap_endpoint"].clone()).unwrap();
        let prefill_id = PrefillId::new(
            HttpOrigin::parse(&prefill_process.url).unwrap(),
            prefill_process.instance_id,
        )
        .unwrap();
        let decoder_id = DecoderId::new(
            HttpOrigin::parse(&decoder_process.url).unwrap(),
            decoder_process.instance_id,
        )
        .unwrap();
        let prefill_bootstrap_endpoint =
            PrefillBootstrapEndpoint::new(bootstrap.host.clone(), bootstrap.port).unwrap();
        let request_chain_id =
            serde_json::from_value(request["logical_request_chain_id"].clone()).unwrap();
        let reservation_attempt_id =
            serde_json::from_value(request["reservation_attempt_id"].clone()).unwrap();
        let reserve_attempt_digest = DecoderReserveAttemptDigest::from_hex(
            request["reserve_attempt_digest"].as_str().unwrap(),
        )
        .unwrap();
        let source_tp_size = request["source_tp_size"].as_u64().unwrap() as usize;
        let prepared_ttl_ms = request["prepared_ttl_ms"].as_u64().unwrap();
        let inference_route = match request["inference_route"].as_str().unwrap() {
            "/generate" => DecoderInferenceRoute::Generate,
            "/v1/chat/completions" => DecoderInferenceRoute::ChatCompletions,
            "/v1/completions" => DecoderInferenceRoute::Completions,
            route => panic!("unexpected session inference route {route}"),
        };
        let request_shape = match request["request_shape"].as_str().unwrap() {
            "scalar" => DecoderRequestShape::Scalar,
            "batch" => DecoderRequestShape::Batch,
            shape => panic!("unexpected session request shape {shape}"),
        };
        let base_request_body = Bytes::copy_from_slice(
            request["base_request_body_json"]
                .as_str()
                .unwrap()
                .as_bytes(),
        );
        let child_request_ids: Vec<Uuid> =
            serde_json::from_value(request["child_request_ids"].clone()).unwrap();
        let grant_id = Uuid::new_v4();
        let prepared_expires_at_unix_ms = 2_000_000_000_000u64;
        let mut allocations = Vec::with_capacity(child_request_ids.len());
        let mut children = Vec::with_capacity(child_request_ids.len());
        for (index, child_request_id) in child_request_ids.iter().copied().enumerate() {
            let digest_byte = u8::try_from(index + 1).unwrap();
            let slot_generation = Uuid::new_v4();
            let writer_manifest_digest = AuthorityDigest([digest_byte; 32]);
            let allocation_digest = AuthorityDigest([digest_byte + 16; 32]);
            let bootstrap_room = 10_000 + index as u64;
            let request_slot = 20_000 + index as u64;
            let request_generation = 30_000 + index as u64;
            children.push(
                DecoderGrantChildBinding::new(
                    child_request_id,
                    DecoderSlotGeneration::new(slot_generation),
                    bootstrap_room,
                    request_slot,
                    request_generation,
                    writer_manifest_digest,
                    allocation_digest,
                    DecoderGrantChildAccounting::new(512 + index, 64 + index),
                )
                .unwrap(),
            );
            allocations.push(json!({
                "child_request_id": child_request_id,
                "decoder_slot_generation": slot_generation,
                "bootstrap_room": bootstrap_room,
                "request_slot": request_slot,
                "request_generation": request_generation,
                "writer_manifest_digest": hex_digest(writer_manifest_digest),
                "allocation_digest": hex_digest(allocation_digest),
                "reserved_kv_tokens": 512 + index,
                "remaining_decode_tokens": 64 + index,
            }));
        }
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
        )
        .unwrap();
        let final_request_body = build_bound_request(&unbound).unwrap();
        let binding = unbound.bind(final_request_body.clone());
        let reserve_response = json!({
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "grant_id": grant_id,
            "grant_token": URL_SAFE_NO_PAD.encode([0x5a; GRANT_TOKEN_BYTES]),
            "prefill_process": request["prefill_process"],
            "prefill_bootstrap_endpoint": request["prefill_bootstrap_endpoint"],
            "decoder_process": request["decoder_process"],
            "logical_request_chain_id": request_chain_id,
            "reservation_attempt_id": reservation_attempt_id,
            "reserve_attempt_digest": reserve_attempt_digest.to_hex(),
            "source_tp_size": source_tp_size,
            "prepared_ttl_ms": prepared_ttl_ms,
            "inference_route": inference_route.as_str(),
            "request_shape": request_shape.as_str(),
            "reservation_digest": binding.reservation_digest().to_hex(),
            "allocations": allocations,
            "prepared_expires_at_unix_ms": prepared_expires_at_unix_ms,
        });
        SessionGrant {
            binding,
            final_request_body,
            reserve_response,
        }
    }

    const SESSION_MODEL_FINGERPRINT: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SESSION_KV_LAYOUT_FINGERPRINT: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SESSION_API_KEY: &str = "session-engine-api-key";

    fn session_process_metadata(
        role: PdProcessRole,
        tensor_parallel_size: usize,
        launch_instance_id: Uuid,
    ) -> PdProcessMetadata {
        PdProcessMetadata::new(
            PdMetadataSchema::V1,
            launch_instance_id,
            role,
            tensor_parallel_size,
            1,
            SESSION_MODEL_FINGERPRINT,
            SESSION_KV_LAYOUT_FINGERPRINT,
            "bf16",
            64,
            KvTransferProtocol::PackedV4,
            PreparedGrantProtocol::V1,
            (role == PdProcessRole::Prefill)
                .then(|| PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 42_000).unwrap()),
        )
        .unwrap()
    }

    fn session_worker(
        url: &str,
        role: PdProcessRole,
        tensor_parallel_size: usize,
        launch_instance_id: Uuid,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .api_key(SESSION_API_KEY)
                .worker_type(match role {
                    PdProcessRole::Prefill => WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                    PdProcessRole::Decode => WorkerType::Decode,
                })
                .pd_process(PdProcessRegistration::new(
                    HttpOrigin::parse(url).unwrap(),
                    session_process_metadata(role, tensor_parallel_size, launch_instance_id),
                ))
                .build(),
        )
    }

    fn session_directory(
        decoder_url: &str,
        prefill_tp_size: usize,
    ) -> (Arc<PdProcessDirectory>, PrefillId) {
        let directory = Arc::new(PdProcessDirectory::default());
        let prefill = directory
            .admit_prefill(session_worker(
                "http://prefill.test:30000",
                PdProcessRole::Prefill,
                prefill_tp_size,
                Uuid::new_v4(),
            ))
            .unwrap();
        directory
            .admit_decoder(session_worker(
                decoder_url,
                PdProcessRole::Decode,
                1,
                Uuid::new_v4(),
            ))
            .unwrap();
        (directory, prefill.id().clone())
    }

    fn topology_session_directory(decoder_url: &str) -> (Arc<PdProcessDirectory>, PrefillId) {
        let topology = PdTopology::from_json(
            &json!({
                "schema": "pd-topology-v1",
                "groups": [{
                    "id": "group-0",
                    "prefill": {
                        "origin": "http://prefill.test:30000",
                        "tensor_parallel_size": 2,
                        "bootstrap_endpoint": {
                            "host": "prefill-bootstrap.test",
                            "port": 42_000
                        }
                    },
                    "decoders": [{
                        "origin": decoder_url,
                        "tensor_parallel_size": 1
                    }]
                }]
            })
            .to_string(),
        )
        .unwrap();
        let directory = Arc::new(PdProcessDirectory::new(Some(Arc::new(topology))));
        let prefill = directory
            .admit_prefill(session_worker(
                "http://prefill.test:30000",
                PdProcessRole::Prefill,
                2,
                Uuid::new_v4(),
            ))
            .unwrap();
        directory
            .admit_decoder(session_worker(
                decoder_url,
                PdProcessRole::Decode,
                1,
                Uuid::new_v4(),
            ))
            .unwrap();
        (directory, prefill.id().clone())
    }

    fn session_template() -> DecoderRequestTemplate {
        DecoderRequestTemplate::new(
            DecoderInferenceRoute::ChatCompletions,
            Bytes::from_static(br#"{"messages":[{"role":"user","content":"production-shaped"}]}"#),
        )
        .unwrap()
    }

    async fn wait_for_session_cleanup(directory: &PdProcessDirectory, prefill_id: &PrefillId) {
        let cleanup = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let snapshot = directory.prefill(prefill_id).unwrap().pool().snapshot();
                let decoder = &snapshot.replicas[0];
                if snapshot.active_logical_requests == 0
                    && decoder.pending_admissions == 0
                    && decoder.active_cohorts == 0
                    && decoder.quiescing_cohorts == 0
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        if cleanup.is_err() {
            let snapshot = directory.prefill(prefill_id).unwrap().pool().snapshot();
            panic!("session actor did not release its pool ownership: {snapshot:?}");
        }
    }

    async fn wait_for_request_count(state: &TestServerState, expected: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if state.requests.lock().unwrap().len() >= expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("test server did not receive {expected} requests"));
    }

    struct Fixture {
        reservation: RefCell<Option<DecoderGrantReservation>>,
        binding: DecoderGrantBinding,
        final_request_body: Bytes,
        reserve_response: Value,
        token: String,
    }

    impl Fixture {
        fn reservation(&self) -> Ref<'_, DecoderGrantReservation> {
            Ref::map(self.reservation.borrow(), |reservation| {
                reservation
                    .as_ref()
                    .expect("fixture reservation was already consumed")
            })
        }

        fn take_reservation(&self) -> DecoderGrantReservation {
            self.reservation
                .borrow_mut()
                .take()
                .expect("fixture reservation was already consumed")
        }
    }

    fn fixture(decoder_url: &str, child_count: usize) -> Fixture {
        let request_shape = if child_count == 1 {
            DecoderRequestShape::Scalar
        } else {
            DecoderRequestShape::Batch
        };
        fixture_with_shape_and_tp(decoder_url, child_count, request_shape, 2)
    }

    fn fixture_with_shape(
        decoder_url: &str,
        child_count: usize,
        request_shape: DecoderRequestShape,
    ) -> Fixture {
        fixture_with_shape_and_tp(decoder_url, child_count, request_shape, 2)
    }

    fn fixture_with_tp(decoder_url: &str, child_count: usize, source_tp_size: usize) -> Fixture {
        let request_shape = if child_count == 1 {
            DecoderRequestShape::Scalar
        } else {
            DecoderRequestShape::Batch
        };
        fixture_with_shape_and_tp(decoder_url, child_count, request_shape, source_tp_size)
    }

    fn fixture_with_shape_and_tp(
        decoder_url: &str,
        child_count: usize,
        request_shape: DecoderRequestShape,
        source_tp_size: usize,
    ) -> Fixture {
        let prefill_id = PrefillId::new(
            HttpOrigin::parse("http://prefill.test:30000").unwrap(),
            uuid("10000000-0000-4000-8000-000000000001"),
        )
        .unwrap();
        let decoder_id = DecoderId::new(
            HttpOrigin::parse(decoder_url).unwrap(),
            uuid("20000000-0000-4000-8000-000000000002"),
        )
        .unwrap();
        let bootstrap = PrefillBootstrapEndpoint::new("prefill-bootstrap.test", 42000).unwrap();
        let chain_id = uuid("30000000-0000-4000-8000-000000000003");
        let grant_id = uuid("40000000-0000-4000-8000-000000000004");
        let prepared_ttl_ms = 2_000u64;
        let prepared_expires_at_unix_ms = 1_900_000_000_000u64;
        let text = match request_shape {
            DecoderRequestShape::Scalar => Value::String(PROMPT_SECRET.to_string()),
            DecoderRequestShape::Batch => Value::Array(
                (0..child_count)
                    .map(|_| Value::String(PROMPT_SECRET.to_string()))
                    .collect(),
            ),
        };
        let template = DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            Bytes::from(serde_json::to_vec(&json!({"text": text})).unwrap()),
        )
        .unwrap();
        assert_eq!(template.request_shape(), request_shape);
        assert_eq!(template.child_count(), child_count);
        let reservation = template
            .prepare_reservation(
                prefill_id.clone(),
                bootstrap.clone(),
                decoder_id.clone(),
                chain_id,
                source_tp_size,
                Duration::from_secs(2),
            )
            .unwrap();
        let reservation_attempt_id = reservation.reservation_attempt_id();
        let child_request_ids = reservation.child_request_ids().to_vec();
        let base_request_body = reservation.base_request_body();

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
        let unbound_binding = UnboundGrantBinding::new(
            grant_id,
            reservation_attempt_id,
            reservation.reserve_attempt_digest(),
            DecoderInferenceRoute::Generate,
            request_shape,
            prepared_ttl_ms,
            prepared_expires_at_unix_ms,
            base_request_body,
            prefill_id.clone(),
            bootstrap.clone(),
            chain_id,
            source_tp_size,
            decoder_id.clone(),
            children,
        )
        .unwrap();
        let final_request_body = build_bound_request(&unbound_binding).unwrap();
        let binding = unbound_binding.bind(final_request_body.clone());
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
            "reservation_attempt_id": reservation_attempt_id,
            "reserve_attempt_digest": binding.reserve_attempt_digest().to_hex(),
            "source_tp_size": source_tp_size,
            "prepared_ttl_ms": prepared_ttl_ms,
            "inference_route": DecoderInferenceRoute::Generate.as_str(),
            "request_shape": request_shape.as_str(),
            "reservation_digest": binding.reservation_digest().to_hex(),
            "allocations": allocation_json,
            "prepared_expires_at_unix_ms": prepared_expires_at_unix_ms,
        });
        Fixture {
            reservation: RefCell::new(Some(reservation)),
            binding,
            final_request_body,
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
            "reservation_attempt_id": binding.reservation_attempt_id(),
            "reserve_attempt_digest": binding.reserve_attempt_digest().to_hex(),
            "prefill_process": WireProcessIdentity::from_prefill(binding.prefill_id()),
            "prefill_bootstrap_endpoint":
                WireBootstrapEndpoint::from(binding.prefill_bootstrap_endpoint()),
            "decoder_process": WireProcessIdentity::from_decoder(binding.decoder_id()),
            "logical_request_chain_id": binding.request_chain_id(),
            "source_tp_size": binding.source_tp_size(),
            "inference_route": binding.inference_route().as_str(),
            "request_shape": binding.request_shape().as_str(),
            "prepared_ttl_ms": binding.prepared_ttl_ms(),
            "prepared_expires_at_unix_ms": binding.prepared_expires_at_unix_ms(),
            "child_request_ids": binding.child_request_ids().collect::<Vec<_>>(),
            "decoder_slot_generations": binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect::<Vec<_>>(),
            "bootstrap_rooms": binding.bootstrap_rooms(),
            "reservation_digest": binding.reservation_digest().to_hex(),
            "grant_digest": binding.digest().to_hex(),
            "operation": operation,
            "state": state,
            "receipt_id": uuid("70000000-0000-4000-8000-000000000007"),
            "receipt_digest": hex_digest(AuthorityDigest([0x77; 32])),
            "take_once": true,
        })
    }

    fn unbound_cancellation_receipt(
        binding: &DecoderGrantBinding,
        attempted_binding: bool,
    ) -> Value {
        json!({
            "schema_version": SCHEMA_VERSION,
            "grant_id": binding.grant_id(),
            "reservation_attempt_id": binding.reservation_attempt_id(),
            "reserve_attempt_digest": binding.reserve_attempt_digest().to_hex(),
            "prefill_process": WireProcessIdentity::from_prefill(binding.prefill_id()),
            "prefill_bootstrap_endpoint":
                WireBootstrapEndpoint::from(binding.prefill_bootstrap_endpoint()),
            "decoder_process": WireProcessIdentity::from_decoder(binding.decoder_id()),
            "logical_request_chain_id": binding.request_chain_id(),
            "source_tp_size": binding.source_tp_size(),
            "inference_route": binding.inference_route().as_str(),
            "request_shape": binding.request_shape().as_str(),
            "prepared_ttl_ms": binding.prepared_ttl_ms(),
            "prepared_expires_at_unix_ms": binding.prepared_expires_at_unix_ms(),
            "child_request_ids": binding.child_request_ids().collect::<Vec<_>>(),
            "decoder_slot_generations": binding
                .slot_generations()
                .iter()
                .map(|generation| generation.as_uuid())
                .collect::<Vec<_>>(),
            "bootstrap_rooms": binding.bootstrap_rooms(),
            "reservation_digest": binding.reservation_digest().to_hex(),
            "attempted_grant_digest":
                attempted_binding.then(|| binding.digest().to_hex()),
            "operation": WireOperation::Cancel,
            "state": WireGrantState::Cancelled,
            "receipt_id": uuid("72000000-0000-4000-8000-000000000007"),
            "receipt_digest": hex_digest(AuthorityDigest([0x72; 32])),
            "take_once": true,
        })
    }

    fn reserve_refusal_receipt(reservation: &DecoderGrantReservation) -> Value {
        json!({
            "schema_version": SCHEMA_VERSION,
            "operation": WireOperation::Reserve,
            "state": WireGrantState::Refused,
            "prefill_process": WireProcessIdentity::from_prefill(reservation.prefill_id()),
            "prefill_bootstrap_endpoint":
                WireBootstrapEndpoint::from(reservation.prefill_bootstrap_endpoint()),
            "decoder_process": WireProcessIdentity::from_decoder(reservation.decoder_id()),
            "logical_request_chain_id": reservation.logical_request_chain_id(),
            "reservation_attempt_id": reservation.reservation_attempt_id(),
            "reserve_attempt_digest": reservation.reserve_attempt_digest().to_hex(),
            "source_tp_size": reservation.source_tp_size(),
            "prepared_ttl_ms": reservation.prepared_ttl_ms(),
            "inference_route": reservation.inference_route().as_str(),
            "request_shape": reservation.request_shape().as_str(),
            "reason_code": "capacity_exhausted",
            "diagnostic": PROMPT_SECRET,
            "disposition": WireReserveRefusalDisposition::RetryAnotherDecoder,
            "receipt_id": uuid("73000000-0000-4000-8000-000000000007"),
            "receipt_digest": hex_digest(AuthorityDigest([0x73; 32])),
            "take_once": true,
        })
    }

    fn response(body: Value) -> PlannedResponse {
        PlannedResponse::json(body)
    }

    fn assert_debug_redacted(value: &impl fmt::Debug, secrets: &[&str]) {
        let debug = format!("{value:?}");
        for secret in secrets {
            assert!(!debug.contains(*secret));
        }
    }

    fn assert_error_redacted(error: &EngineGrantError, secrets: &[&str]) {
        let debug = format!("{error:?}");
        let display = error.to_string();
        for secret in secrets {
            assert!(!debug.contains(*secret));
            assert!(!display.contains(*secret));
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

    async fn reserve_bound(state: &TestServerState, fixture: &Fixture) -> BoundPreparedGrant {
        let bind_path = format!("{CONTROL_PATH}/{}/bind", fixture.binding.grant_id());
        {
            let mut responses = state.responses.lock().unwrap();
            let bind_responses = responses.entry(bind_path).or_default();
            if bind_responses.is_empty() {
                bind_responses.push_back(response(receipt(
                    &fixture.binding,
                    WireOperation::Bind,
                    WireGrantState::Prepared,
                )));
            }
        }

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let mut unbound = reserve.reconcile_reserve().await.unwrap();
        let mut binding = unbound.begin_bind().unwrap();
        binding.reconcile_bind().await.unwrap()
    }

    #[test]
    fn reservation_derives_route_shape_and_requires_exact_rids() {
        let prefill_id = PrefillId::new(
            HttpOrigin::parse("http://prefill.test:30000").unwrap(),
            Uuid::new_v4(),
        )
        .unwrap();
        let decoder_id = DecoderId::new(
            HttpOrigin::parse("http://decoder.test:30001").unwrap(),
            Uuid::new_v4(),
        )
        .unwrap();
        let bootstrap = PrefillBootstrapEndpoint::new("bootstrap.test", 40000).unwrap();
        let prepare = |route: DecoderInferenceRoute, body: Value| {
            DecoderRequestTemplate::new(route, Bytes::from(serde_json::to_vec(&body).unwrap()))?
                .prepare_reservation(
                    prefill_id.clone(),
                    bootstrap.clone(),
                    decoder_id.clone(),
                    Uuid::new_v4(),
                    4,
                    Duration::from_secs(1),
                )
        };

        let chat = prepare(
            DecoderInferenceRoute::ChatCompletions,
            json!({"messages": [{"role": "user", "content": PROMPT_SECRET}]}),
        )
        .unwrap();
        assert_eq!(chat.request_shape(), DecoderRequestShape::Scalar);
        assert_eq!(chat.child_request_ids().len(), 1);
        let chat_body: Value = serde_json::from_slice(&chat.base_request_body()).unwrap();
        assert_eq!(
            chat_body[RID_KEY],
            Value::String(chat.child_request_ids()[0].to_string())
        );

        let completion_tokens = prepare(
            DecoderInferenceRoute::Completions,
            json!({"prompt": [1, 2, 3]}),
        )
        .unwrap();
        assert_eq!(
            completion_tokens.request_shape(),
            DecoderRequestShape::Scalar
        );

        let completion_batch = prepare(
            DecoderInferenceRoute::Completions,
            json!({"prompt": [PROMPT_SECRET]}),
        )
        .unwrap();
        assert_eq!(completion_batch.request_shape(), DecoderRequestShape::Batch);
        assert_eq!(completion_batch.child_request_ids().len(), 1);
        let completion_body: Value =
            serde_json::from_slice(&completion_batch.base_request_body()).unwrap();
        assert_eq!(
            completion_body[RID_KEY],
            json!([completion_batch.child_request_ids()[0].to_string()])
        );

        let completion_token_batch = prepare(
            DecoderInferenceRoute::Completions,
            json!({"prompt": [[1, 2], [3]]}),
        )
        .unwrap();
        assert_eq!(
            completion_token_batch.request_shape(),
            DecoderRequestShape::Batch
        );
        assert_eq!(completion_token_batch.child_request_ids().len(), 2);

        let generate_text_batch = prepare(
            DecoderInferenceRoute::Generate,
            json!({"text": [PROMPT_SECRET]}),
        )
        .unwrap();
        assert_eq!(
            generate_text_batch.request_shape(),
            DecoderRequestShape::Batch
        );

        let generate_tokens = prepare(
            DecoderInferenceRoute::Generate,
            json!({"text": Value::Null, "input_ids": [1, 2, 3]}),
        )
        .unwrap();
        assert_eq!(generate_tokens.request_shape(), DecoderRequestShape::Scalar);

        let generate_token_batch = prepare(
            DecoderInferenceRoute::Generate,
            json!({"input_ids": [[1, 2], [3]]}),
        )
        .unwrap();
        assert_eq!(
            generate_token_batch.request_shape(),
            DecoderRequestShape::Batch
        );
        assert_eq!(generate_token_batch.child_request_ids().len(), 2);

        let generate_embedding_batch = prepare(
            DecoderInferenceRoute::Generate,
            json!({"input_embeds": [[[0.1, 0.2]], [[0.3, 0.4]]]}),
        )
        .unwrap();
        assert_eq!(
            generate_embedding_batch.request_shape(),
            DecoderRequestShape::Batch
        );

        assert!(prepare(DecoderInferenceRoute::Generate, json!({"text": [1]}),).is_err());
        assert!(prepare(
            DecoderInferenceRoute::Generate,
            json!({"input_ids": "invalid"}),
        )
        .is_err());
        assert!(prepare(DecoderInferenceRoute::Generate, json!({"input_ids": [-1]}),).is_err());
        assert!(prepare(DecoderInferenceRoute::Generate, json!({"input_ids": [1.5]}),).is_err());
        assert!(prepare(
            DecoderInferenceRoute::Generate,
            json!({"input_ids": [u64::MAX]}),
        )
        .is_err());
        assert!(prepare(
            DecoderInferenceRoute::Generate,
            json!({"text": PROMPT_SECRET, "input_ids": [1, 2]}),
        )
        .is_err());

        for key in GATEWAY_OWNED_REQUEST_KEYS {
            let error = prepare(
                DecoderInferenceRoute::Generate,
                json!({
                    "text": PROMPT_SECRET,
                    (key): Value::Null,
                }),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                EngineGrantError::InvalidGrant(message)
                    if message.contains("gateway-owned field")
            ));
        }

        let template = DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            Bytes::from_static(br#"{"text":"retry"}"#),
        )
        .unwrap();
        let first = template
            .prepare_reservation(
                prefill_id.clone(),
                bootstrap.clone(),
                decoder_id.clone(),
                Uuid::new_v4(),
                4,
                Duration::from_secs(1),
            )
            .unwrap();
        let second = template
            .prepare_reservation(
                prefill_id,
                bootstrap,
                decoder_id,
                Uuid::new_v4(),
                4,
                Duration::from_secs(1),
            )
            .unwrap();
        assert_ne!(
            first.reservation_attempt_id(),
            second.reservation_attempt_id()
        );
        assert_ne!(first.child_request_ids(), second.child_request_ids());
    }

    #[test]
    fn reservation_accepts_supported_prefill_tp_and_rejects_other_sizes() {
        let prefill_id = PrefillId::new(
            HttpOrigin::parse("http://prefill.test:30000").unwrap(),
            Uuid::new_v4(),
        )
        .unwrap();
        let decoder_id = DecoderId::new(
            HttpOrigin::parse("http://decoder.test:30001").unwrap(),
            Uuid::new_v4(),
        )
        .unwrap();
        let bootstrap = PrefillBootstrapEndpoint::new("bootstrap.test", 40000).unwrap();
        let template = DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            Bytes::from_static(br#"{"text":"test"}"#),
        )
        .unwrap();

        for source_tp_size in [1, 2, 4, 8] {
            assert!(template
                .prepare_reservation(
                    prefill_id.clone(),
                    bootstrap.clone(),
                    decoder_id.clone(),
                    Uuid::new_v4(),
                    source_tp_size,
                    Duration::from_secs(1),
                )
                .is_ok());
        }
        for source_tp_size in [0, 3, 6, 16] {
            assert!(matches!(
                template.prepare_reservation(
                    prefill_id.clone(),
                    bootstrap.clone(),
                    decoder_id.clone(),
                    Uuid::new_v4(),
                    source_tp_size,
                    Duration::from_secs(1),
                ),
                Err(EngineGrantError::InvalidGrant(message))
                    if message == "source tensor-parallel size must be 1, 2, 4, or 8"
            ));
        }
    }

    #[test]
    fn reservation_rejects_parallel_sampling_for_scalar_and_batch_bodies() {
        let make = |route: DecoderInferenceRoute, body: Value| {
            DecoderRequestTemplate::new(route, Bytes::from(serde_json::to_vec(&body).unwrap()))
        };

        assert!(make(
            DecoderInferenceRoute::ChatCompletions,
            json!({
                "messages": [{"role": "user", "content": PROMPT_SECRET}],
                "n": 2,
            }),
        )
        .is_err());
        assert!(make(
            DecoderInferenceRoute::Completions,
            json!({
                "prompt": PROMPT_SECRET,
                "best_of": 2,
            }),
        )
        .is_err());
        assert!(make(
            DecoderInferenceRoute::Generate,
            json!({
                "text": PROMPT_SECRET,
                "sampling_params": [{"n": 1}],
            }),
        )
        .is_err());
        assert!(make(
            DecoderInferenceRoute::Generate,
            json!({
                "text": [PROMPT_SECRET],
                "sampling_params": [{"n": 1}],
            }),
        )
        .is_ok());
        assert!(make(
            DecoderInferenceRoute::Generate,
            json!({
                "text": [PROMPT_SECRET, PROMPT_SECRET],
                "sampling_params": [{"n": 1}, {"n": 2}],
            }),
        )
        .is_err());
        assert!(make(
            DecoderInferenceRoute::Generate,
            json!({
                "text": [PROMPT_SECRET, PROMPT_SECRET],
                "sampling_params": [{"n": 1}],
            }),
        )
        .is_err());
        assert!(make(
            DecoderInferenceRoute::Generate,
            json!({
                "text": [PROMPT_SECRET, PROMPT_SECRET],
                "sampling_params": [{"n": 1}, Value::Null],
            }),
        )
        .is_err());
    }

    #[tokio::test]
    async fn reserve_promote_complete_uses_secret_header_only() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 2);
        let grant_id = fixture.binding.grant_id();
        let bind_path = format!("{CONTROL_PATH}/{grant_id}/bind");
        let promote_path = format!("{CONTROL_PATH}/{grant_id}/promote");
        let complete_path = format!("{CONTROL_PATH}/{grant_id}/complete");
        let plans = plan([
            (
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()),
            ),
            (
                bind_path.clone(),
                response(receipt(
                    &fixture.binding,
                    WireOperation::Bind,
                    WireGrantState::Prepared,
                )),
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

        let template = DecoderRequestTemplate::new(
            DecoderInferenceRoute::Generate,
            Bytes::from(json!({"text": PROMPT_SECRET}).to_string()),
        )
        .unwrap();
        assert_debug_redacted(&template, &[PROMPT_SECRET, &fixture.token]);
        let reservation_debug = format!("{:?}", fixture.reservation());
        assert!(!reservation_debug.contains(PROMPT_SECRET));
        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        assert_debug_redacted(
            &reserve,
            &[PROMPT_SECRET, &fixture.token, "test-decoder-api-key"],
        );
        let mut unbound = reserve.reconcile_reserve().await.unwrap();
        assert_debug_redacted(&unbound, &[PROMPT_SECRET, &fixture.token]);
        let mut binding = unbound.begin_bind().unwrap();
        assert_debug_redacted(&binding, &[PROMPT_SECRET, &fixture.token]);
        let mut grant = binding.reconcile_bind().await.unwrap();
        assert_debug_redacted(&grant, &[PROMPT_SECRET, &fixture.token]);
        let token_debug = format!(
            "{:?}",
            SecretGrantToken::new(fixture.token.clone()).unwrap()
        );
        assert_eq!(token_debug, "SecretGrantToken([REDACTED])");
        let mut promotion = grant.begin_test_promotion().unwrap();
        assert_debug_redacted(&promotion, &[PROMPT_SECRET, &fixture.token]);
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        assert_debug_redacted(&retained, &[PROMPT_SECRET, &fixture.token]);
        let mut completion = retained.begin_test_completion().unwrap();
        assert_debug_redacted(&completion, &[PROMPT_SECRET, &fixture.token]);
        let outcome = completion.reconcile_completion().await.unwrap();
        assert!(matches!(
            outcome,
            EngineCompletionOutcome::Completed(ref receipt)
                if receipt.kind() == EngineReleaseKind::Completed
        ));

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[1].path, bind_path);
        assert_eq!(requests[1].body_bytes, fixture.final_request_body);
        assert_eq!(
            requests[1].body,
            serde_json::from_slice::<Value>(&fixture.final_request_body).unwrap()
        );
        assert!(requests[1].body.get("base_request_body_json").is_none());
        assert!(requests[1].body.get("request_body_json").is_none());
        assert_eq!(requests[2].path, promote_path);
        assert_eq!(requests[3].path, complete_path);
        for request in requests.iter() {
            assert_eq!(
                request.accept_encoding.as_deref(),
                Some(IDENTITY_CONTENT_ENCODING)
            );
        }
        assert_eq!(
            requests[0].authorization.as_deref(),
            Some("Bearer test-decoder-api-key")
        );
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
    async fn reserve_ambiguity_retries_the_same_attempt_transcript() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(json!({"error": "receipt delivery ambiguous"}))
                        .with_status(StatusCode::SERVICE_UNAVAILABLE),
                ),
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
            ]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let attempt_id = fixture.binding.reservation_attempt_id();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        assert!(matches!(
            reserve.reconcile_reserve().await,
            Err(EngineGrantError::AmbiguousReserve(_))
        ));
        assert_eq!(reserve.reservation_attempt_id().unwrap(), attempt_id);
        let grant = reserve.reconcile_reserve().await.unwrap();
        assert_eq!(grant.reservation_attempt_id(), attempt_id);

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].body_bytes, requests[1].body_bytes);
        assert_eq!(
            requests[0].body["reservation_attempt_id"],
            attempt_id.to_string()
        );
        task.abort();
    }

    #[tokio::test]
    async fn reserve_timeout_retries_the_same_attempt_transcript() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    PlannedResponse::pending(),
                ),
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
            ]),
        );

        let client = DecoderGrantControlClient::from_builder_with_timeout(
            Client::builder(),
            Duration::from_millis(20),
        )
        .unwrap();
        let attempt_id = fixture.binding.reservation_attempt_id();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        assert!(matches!(
            reserve.reconcile_reserve().await,
            Err(EngineGrantError::AmbiguousReserve(reason)) if reason == "request_timeout"
        ));
        assert_eq!(reserve.reservation_attempt_id().unwrap(), attempt_id);
        let grant = reserve.reconcile_reserve().await.unwrap();
        assert_eq!(grant.reservation_attempt_id(), attempt_id);

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].body_bytes, requests[1].body_bytes);
        task.abort();
    }

    #[tokio::test]
    async fn production_session_runs_for_every_supported_prefill_tp() {
        for prefill_tp_size in [1, 2, 4, 8] {
            let (server_url, state, task) = start_server().await;
            let terminal = Arc::new(Notify::new());
            *state.session_engine.lock().unwrap() = Some(SessionEngine::new(Arc::clone(&terminal)));
            let (directory, prefill_id) = session_directory(&server_url, prefill_tp_size);
            let client = DecoderGrantControlClient::new().unwrap();

            let reserved = tokio::time::timeout(
                Duration::from_secs(2),
                PdReservedRequestSession::establish(
                    Arc::clone(&directory),
                    &prefill_id,
                    format!("tp{prefill_tp_size}-complete"),
                    None,
                    session_template(),
                    &client,
                    &RetryConfig::default(),
                ),
            )
            .await
            .expect("session establishment timed out")
            .unwrap();
            {
                let requests = state.requests.lock().unwrap();
                let paths: Vec<&str> = requests
                    .iter()
                    .map(|request| request.path.as_str())
                    .collect();
                assert_eq!(paths.len(), 2);
                assert_eq!(paths[0], format!("{CONTROL_PATH}/reserve"));
                assert!(paths[1].ends_with("/bind"));
            }
            let session = tokio::time::timeout(Duration::from_secs(2), reserved.promote())
                .await
                .expect("session promotion timed out")
                .unwrap();
            let request_body: Value = serde_json::from_slice(&session.request_body()).unwrap();
            assert!(request_body[RID_KEY].is_string());
            assert_eq!(request_body[BOOTSTRAP_HOST_KEY], "prefill-bootstrap.test");
            assert_eq!(request_body[BOOTSTRAP_PORT_KEY], 42_000);
            assert_eq!(request_body[BOOTSTRAP_ROOM_KEY], 10_000);

            tokio::time::timeout(Duration::from_secs(2), session.complete())
                .await
                .expect("session completion timed out")
                .unwrap();
            wait_for_session_cleanup(&directory, &prefill_id).await;

            let requests = state.requests.lock().unwrap();
            let paths: Vec<&str> = requests
                .iter()
                .map(|request| request.path.as_str())
                .collect();
            assert_eq!(paths.len(), 4);
            assert_eq!(paths[0], format!("{CONTROL_PATH}/reserve"));
            assert!(paths[1].ends_with("/bind"));
            assert!(paths[2].ends_with("/promote"));
            assert!(paths[3].ends_with("/complete"));
            assert_eq!(requests[0].body["source_tp_size"], prefill_tp_size);
            drop(requests);
            task.abort();
        }
    }

    #[tokio::test]
    async fn topology_precharge_runs_one_complete_lifecycle_without_leaking_ownership() {
        let (server_url, state, task) = start_server().await;
        let terminal = Arc::new(Notify::new());
        *state.session_engine.lock().unwrap() = Some(SessionEngine::new(terminal));
        let (directory, prefill_id) = topology_session_directory(&server_url);
        let client = DecoderGrantControlClient::new().unwrap();
        let group_request = directory
            .begin_group_request("topology-production-session", None)
            .unwrap();
        assert_eq!(group_request.group_id().as_str(), "group-0");
        assert_eq!(
            directory
                .prefill(&prefill_id)
                .unwrap()
                .pool()
                .snapshot()
                .active_logical_requests,
            1
        );

        let reserved = PdReservedRequestSession::establish_group(
            Arc::clone(&directory),
            group_request,
            None,
            session_template(),
            &client,
            &RetryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(reserved.group_id().unwrap().as_str(), "group-0");
        let session = reserved.promote().await.unwrap();
        session.complete().await.unwrap();
        wait_for_session_cleanup(&directory, &prefill_id).await;

        let requests = state.requests.lock().unwrap();
        let paths = requests
            .iter()
            .map(|request| request.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 4);
        assert_eq!(paths[0], format!("{CONTROL_PATH}/reserve"));
        assert!(paths[1].ends_with("/bind"));
        assert!(paths[2].ends_with("/promote"));
        assert!(paths[3].ends_with("/complete"));
        drop(requests);
        task.abort();
    }

    #[tokio::test]
    async fn dropping_production_session_aborts_engine_and_releases_pool_ownership() {
        let (server_url, state, task) = start_server().await;
        let terminal = Arc::new(Notify::new());
        *state.session_engine.lock().unwrap() = Some(SessionEngine::new(Arc::clone(&terminal)));
        let (directory, prefill_id) = session_directory(&server_url, 2);
        let client = DecoderGrantControlClient::new().unwrap();
        let session = PdReservedRequestSession::establish(
            Arc::clone(&directory),
            &prefill_id,
            "dropped-production-session",
            None,
            session_template(),
            &client,
            &RetryConfig::default(),
        )
        .await
        .unwrap()
        .promote()
        .await
        .unwrap();

        drop(session);
        tokio::time::timeout(Duration::from_secs(2), terminal.notified())
            .await
            .expect("dropped session did not dispatch engine abort");
        wait_for_session_cleanup(&directory, &prefill_id).await;

        let requests = state.requests.lock().unwrap();
        let paths: Vec<&str> = requests
            .iter()
            .map(|request| request.path.as_str())
            .collect();
        assert_eq!(paths.len(), 4);
        assert_eq!(paths[0], format!("{CONTROL_PATH}/reserve"));
        assert!(paths[1].ends_with("/bind"));
        assert!(paths[2].ends_with("/promote"));
        assert!(paths[3].ends_with("/abort"));
        drop(requests);
        task.abort();
    }

    #[tokio::test]
    async fn cancelling_promotion_waiter_aborts_engine_and_releases_pool_ownership() {
        let (server_url, state, task) = start_server().await;
        let terminal = Arc::new(Notify::new());
        *state.session_engine.lock().unwrap() = Some(SessionEngine::new(Arc::clone(&terminal)));
        let (directory, prefill_id) = session_directory(&server_url, 4);
        let client = DecoderGrantControlClient::new().unwrap();
        let reserved = PdReservedRequestSession::establish(
            Arc::clone(&directory),
            &prefill_id,
            "cancelled-production-promotion",
            None,
            session_template(),
            &client,
            &RetryConfig::default(),
        )
        .await
        .unwrap();
        let grant_id = state
            .session_engine
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .grant
            .as_ref()
            .unwrap()
            .binding
            .grant_id();
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/{grant_id}/promote"),
                PlannedResponse::pending(),
            )]),
        );

        let promotion = tokio::spawn(reserved.promote());
        wait_for_request_count(&state, 3).await;
        promotion.abort();
        assert!(promotion.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), terminal.notified())
            .await
            .expect("cancelled promotion did not dispatch engine abort");
        wait_for_session_cleanup(&directory, &prefill_id).await;

        let requests = state.requests.lock().unwrap();
        let paths: Vec<&str> = requests
            .iter()
            .map(|request| request.path.as_str())
            .collect();
        assert_eq!(paths.len(), 4);
        assert_eq!(paths[0], format!("{CONTROL_PATH}/reserve"));
        assert!(paths[1].ends_with("/bind"));
        assert!(paths[2].ends_with("/promote"));
        assert!(paths[3].ends_with("/abort"));
        drop(requests);
        task.abort();
    }

    #[tokio::test]
    async fn bind_ambiguity_retries_the_same_raw_body() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let bind_path = format!("{CONTROL_PATH}/{}/bind", fixture.binding.grant_id());
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    bind_path.clone(),
                    response(json!({"error": "bind receipt delivery ambiguous"}))
                        .with_status(StatusCode::SERVICE_UNAVAILABLE),
                ),
                (
                    bind_path,
                    response(receipt(
                        &fixture.binding,
                        WireOperation::Bind,
                        WireGrantState::Prepared,
                    )),
                ),
            ]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let mut unbound = reserve.reconcile_reserve().await.unwrap();
        let mut binding = unbound.begin_bind().unwrap();
        assert!(matches!(
            binding.reconcile_bind().await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "bind",
                ..
            })
        ));
        let bound = binding.reconcile_bind().await.unwrap();
        assert_eq!(bound.request_body(), fixture.final_request_body);

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].body_bytes, fixture.final_request_body);
        assert_eq!(requests[2].body_bytes, fixture.final_request_body);
        task.abort();
    }

    #[tokio::test]
    async fn redirect_responses_never_replay_prompt_or_bearer_authority() {
        let redirect_statuses = [
            StatusCode::MOVED_PERMANENTLY,
            StatusCode::FOUND,
            StatusCode::SEE_OTHER,
            StatusCode::TEMPORARY_REDIRECT,
            StatusCode::PERMANENT_REDIRECT,
        ];
        for status in redirect_statuses {
            let (target_url, target_state, target_task) = start_server().await;
            install_plans(
                &target_state,
                plan([(
                    "/redirect-target".to_string(),
                    response(json!({"unexpected": "redirect followed"})),
                )]),
            );
            let (source_url, source_state, source_task) = start_server().await;
            let fixture = fixture(&source_url, 1);
            let bind_path = format!("{CONTROL_PATH}/{}/bind", fixture.binding.grant_id());
            let location = format!("{target_url}/redirect-target");
            install_plans(
                &source_state,
                plan([
                    (
                        format!("{CONTROL_PATH}/reserve"),
                        response(fixture.reserve_response.clone()),
                    ),
                    (
                        bind_path,
                        response(json!({"redirect": true}))
                            .with_status(status)
                            .with_header("location", &location),
                    ),
                ]),
            );

            let client = DecoderGrantControlClient::from_builder(
                Client::builder().redirect(Policy::limited(10)),
            )
            .unwrap();
            let mut reserve = client.begin_reserve(fixture.take_reservation());
            let mut unbound = reserve.reconcile_reserve().await.unwrap();
            let mut binding = unbound.begin_bind().unwrap();
            let error = match binding.reconcile_bind().await {
                Ok(_) => panic!("redirect response unexpectedly produced a bound grant"),
                Err(error) => error,
            };
            assert!(matches!(
                &error,
                EngineGrantError::AmbiguousControl {
                    operation: "bind",
                    message,
                } if message == &http_status_reason(status)
            ));
            assert_error_redacted(&error, &[PROMPT_SECRET, &fixture.token]);

            {
                let requests = source_state.requests.lock().unwrap();
                assert_eq!(requests.len(), 2);
                assert_eq!(requests[1].body_bytes, fixture.final_request_body);
                assert_eq!(
                    requests[1].authorization.as_deref(),
                    Some(format!("Bearer {}", fixture.token).as_str())
                );
            }
            assert!(target_state.requests.lock().unwrap().is_empty());
            source_task.abort();
            target_task.abort();
        }
    }

    #[tokio::test]
    async fn oversized_control_response_is_rejected_without_exposing_its_body() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let oversized_body = Bytes::from(format!(
            "{PROMPT_SECRET}:{}:{}",
            fixture.token,
            "x".repeat(MAX_CONTROL_RESPONSE_BYTES)
        ));
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                PlannedResponse::raw(oversized_body),
            )]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let error = match reserve.reconcile_reserve().await {
            Ok(_) => panic!("oversized response unexpectedly produced a grant"),
            Err(error) => error,
        };
        assert!(matches!(
            &error,
            EngineGrantError::AmbiguousReserve(message)
                if message == ControlResponseError::BodyTooLarge.code()
        ));
        assert_error_redacted(&error, &[PROMPT_SECRET, &fixture.token]);
        task.abort();
    }

    #[tokio::test]
    async fn chunked_oversized_control_response_is_rejected_without_exposing_its_body() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let secret_chunk = Bytes::from(format!("{PROMPT_SECRET}:{}", fixture.token));
        let oversized_chunk = Bytes::from(vec![b'x'; MAX_CONTROL_RESPONSE_BYTES]);
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                PlannedResponse::chunked(vec![secret_chunk, oversized_chunk]),
            )]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let error = match reserve.reconcile_reserve().await {
            Ok(_) => panic!("oversized response unexpectedly produced a grant"),
            Err(error) => error,
        };
        assert!(matches!(
            &error,
            EngineGrantError::AmbiguousReserve(message)
                if message == ControlResponseError::BodyTooLarge.code()
        ));
        assert_error_redacted(&error, &[PROMPT_SECRET, &fixture.token]);
        task.abort();
    }

    #[tokio::test]
    async fn encoded_control_response_is_rejected_before_json_parsing() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()).with_header("content-encoding", "gzip"),
            )]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let error = match reserve.reconcile_reserve().await {
            Ok(_) => panic!("encoded response unexpectedly produced a grant"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            EngineGrantError::AmbiguousReserve(message)
                if message == ControlResponseError::UnsupportedContentEncoding.code()
        ));
        task.abort();
    }

    #[tokio::test]
    async fn peer_error_and_parse_bodies_are_never_returned_in_control_errors() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let bind_path = format!("{CONTROL_PATH}/{}/bind", fixture.binding.grant_id());
        let peer_body = Bytes::from(format!("{PROMPT_SECRET}:{}", fixture.token));
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(fixture.reserve_response.clone()),
                ),
                (
                    bind_path.clone(),
                    PlannedResponse::raw(peer_body.clone())
                        .with_status(StatusCode::SERVICE_UNAVAILABLE),
                ),
                (bind_path, PlannedResponse::raw(peer_body)),
            ]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let mut unbound = reserve.reconcile_reserve().await.unwrap();
        let mut binding = unbound.begin_bind().unwrap();
        let status_error = match binding.reconcile_bind().await {
            Ok(_) => panic!("HTTP error unexpectedly produced a bound grant"),
            Err(error) => error,
        };
        let parse_error = match binding.reconcile_bind().await {
            Ok(_) => panic!("malformed receipt unexpectedly produced a bound grant"),
            Err(error) => error,
        };
        for error in [&status_error, &parse_error] {
            assert_error_redacted(error, &[PROMPT_SECRET, &fixture.token]);
        }
        assert!(matches!(
            status_error,
            EngineGrantError::AmbiguousControl {
                operation: "bind",
                message,
            } if message == http_status_reason(StatusCode::SERVICE_UNAVAILABLE)
        ));
        assert!(matches!(
            parse_error,
            EngineGrantError::AmbiguousControl {
                operation: "bind",
                message,
            } if message == "invalid_control_receipt"
        ));
        task.abort();
    }

    #[tokio::test]
    async fn bind_failure_cancels_prepared_with_exact_attempt_receipt() {
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
                    format!("{CONTROL_PATH}/{grant_id}/bind"),
                    response(json!({"error": "bind outcome ambiguous"}))
                        .with_status(StatusCode::SERVICE_UNAVAILABLE),
                ),
                (
                    format!("{CONTROL_PATH}/{grant_id}/cancel"),
                    response(unbound_cancellation_receipt(&fixture.binding, true)),
                ),
            ]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(fixture.take_reservation());
        let mut unbound = reserve.reconcile_reserve().await.unwrap();
        let mut binding = unbound.begin_bind().unwrap();
        assert!(binding.reconcile_bind().await.is_err());
        let mut cancellation = binding.begin_cancellation().unwrap();
        assert_debug_redacted(&cancellation, &[PROMPT_SECRET, &fixture.token]);
        let cancellation = cancellation.reconcile_cancellation().await.unwrap();
        assert_eq!(
            cancellation.reservation_attempt_id(),
            fixture.binding.reservation_attempt_id()
        );
        assert_eq!(
            cancellation.reservation_digest(),
            fixture.binding.reservation_digest()
        );
        assert_eq!(
            cancellation.attempted_grant_digest(),
            Some(fixture.binding.digest())
        );

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].body_bytes, fixture.final_request_body);
        assert_eq!(
            requests[2].body["attempted_grant_digest"],
            fixture.binding.digest().to_hex()
        );
        task.abort();
    }

    #[tokio::test]
    async fn one_element_batch_preserves_array_rid_and_bootstrap_shape() {
        let (server_url, state, task) = start_server().await;
        let fixture = fixture_with_shape(&server_url, 1, DecoderRequestShape::Batch);
        install_plans(
            &state,
            plan([(
                format!("{CONTROL_PATH}/reserve"),
                response(fixture.reserve_response.clone()),
            )]),
        );
        let bound = reserve_bound(&state, &fixture).await;
        let body: Value = serde_json::from_slice(&bound.request_body()).unwrap();
        assert!(body["rid"].is_array());
        assert!(body[BOOTSTRAP_HOST_KEY].is_array());
        assert!(body[BOOTSTRAP_PORT_KEY].is_array());
        assert!(body[BOOTSTRAP_ROOM_KEY].is_array());
        task.abort();
    }

    #[tokio::test]
    async fn supported_tp_control_lifecycles_preserve_exact_source_topology() {
        for source_tp_size in [1, 2, 4, 8] {
            for child_count in [1, 3] {
                let (server_url, state, task) = start_server().await;
                let fixture = fixture_with_tp(&server_url, child_count, source_tp_size);
                let grant_id = fixture.binding.grant_id();
                install_plans(
                    &state,
                    plan([
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
                    ]),
                );

                let mut grant = reserve_bound(&state, &fixture).await;
                let mut cancellation = grant.begin_cancellation().unwrap();
                cancellation.reconcile_cancellation().await.unwrap();

                let requests = state.requests.lock().unwrap();
                assert_eq!(requests.len(), 3);
                assert_eq!(requests[0].body["source_tp_size"], source_tp_size);
                assert_eq!(requests[2].body["source_tp_size"], source_tp_size);
                task.abort();
            }
        }
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
        let mut grant = reserve_bound(&state, &fixture).await;
        let mut cancellation = grant.begin_cancellation().unwrap();
        let release = cancellation.reconcile_cancellation().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::PreparedCancelled);
        assert_eq!(
            release.child_request_ids(),
            fixture
                .binding
                .child_request_ids()
                .collect::<Vec<_>>()
                .as_slice()
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut cancellation = grant.begin_cancellation().unwrap();
        assert!(matches!(
            cancellation.reconcile_cancellation().await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "cancel",
                ..
            })
        ));
        assert!(format!("{cancellation:?}").contains("has_control: true"));
        let release = cancellation.reconcile_cancellation().await.unwrap();
        assert_eq!(release.kind(), EngineReleaseKind::PreparedCancelled);
        assert_eq!(state.requests.lock().unwrap().len(), 4);
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
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
        assert_eq!(state.requests.lock().unwrap().len(), 4);
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        assert!(promotion.reconcile_promotion().await.is_err());
        let mut abort = promotion
            .begin_test_abort(
                "promotion_receipt_lost",
                Some("promotion outcome requires quiescence"),
            )
            .unwrap();
        let outcome = abort.reconcile_abort().await.unwrap();
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
            vec![
                format!("{CONTROL_PATH}/reserve"),
                format!("{CONTROL_PATH}/{grant_id}/bind"),
                promote_path,
                abort_path,
            ]
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
        let mut grant = reserve_bound(&server_state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut abort = retained
            .begin_test_abort("decoder_response_failed", Some("upstream body failed"))
            .unwrap();
        let outcome = abort.reconcile_abort().await.unwrap();
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

    async fn run_completion_state(state: WireGrantState) -> EngineCompletionOutcome {
        let (server_url, server_state, task) = start_server().await;
        let fixture = fixture(&server_url, 1);
        let grant_id = fixture.binding.grant_id();
        install_plans(
            &server_state,
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
                    format!("{CONTROL_PATH}/{grant_id}/complete"),
                    response(receipt(&fixture.binding, WireOperation::Complete, state)),
                ),
            ]),
        );
        let mut grant = reserve_bound(&server_state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut completion = retained.begin_test_completion().unwrap();
        let outcome = completion.reconcile_completion().await.unwrap();
        assert_eq!(server_state.requests.lock().unwrap().len(), 4);
        task.abort();
        outcome
    }

    #[tokio::test]
    async fn complete_accepts_only_authoritative_completed_or_quarantined_states() {
        let completed = run_completion_state(WireGrantState::Completed).await;
        assert!(matches!(completed, EngineCompletionOutcome::Completed(_)));
        let quarantined = run_completion_state(WireGrantState::Quarantined).await;
        assert!(matches!(
            quarantined,
            EngineCompletionOutcome::Quarantined(_)
        ));
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
        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut quarantine = retained
            .begin_test_quarantine("response_body_dropped", Some("client disconnected"))
            .unwrap();
        let quarantine = quarantine.reconcile_quarantine().await.unwrap();
        assert_eq!(quarantine.grant_id(), fixture.binding.grant_id());
        assert_eq!(
            quarantine.child_request_ids(),
            fixture
                .binding
                .child_request_ids()
                .collect::<Vec<_>>()
                .as_slice()
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut completion = retained.begin_test_completion().unwrap();
        assert!(completion.reconcile_completion().await.is_err());
        assert!(format!("{completion:?}").contains("has_control: true"));
        let outcome = completion.reconcile_completion().await.unwrap();
        assert!(matches!(
            outcome,
            EngineCompletionOutcome::Completed(ref receipt)
                if receipt.kind() == EngineReleaseKind::Completed
        ));
        assert_eq!(state.requests.lock().unwrap().len(), 5);
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut abort = retained
            .begin_test_abort("response_failed", Some("pinned abort diagnostic"))
            .unwrap();
        assert!(!format!("{abort:?}").contains("pinned abort diagnostic"));
        assert!(abort.reconcile_abort().await.is_err());
        assert!(format!("{abort:?}").contains("has_control: true"));
        let outcome = abort.reconcile_abort().await.unwrap();
        assert!(matches!(outcome, EngineAbortOutcome::Aborted(_)));
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 5);
        assert_eq!(requests[3].body_bytes, requests[4].body_bytes);
        assert_eq!(requests[3].body["diagnostic"], "pinned abort diagnostic");
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
                    response(json!({"error": "receipt delivery failed"}))
                        .with_status(StatusCode::SERVICE_UNAVAILABLE),
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

        let mut grant = reserve_bound(&state, &fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let mut retained = promotion.reconcile_promotion().await.unwrap();
        let mut quarantine = retained
            .begin_test_quarantine("response_dropped", Some("pinned quarantine diagnostic"))
            .unwrap();
        assert!(!format!("{quarantine:?}").contains("pinned quarantine diagnostic"));
        assert!(matches!(
            quarantine.reconcile_quarantine().await,
            Err(EngineGrantError::AmbiguousControl {
                operation: "quarantine",
                ..
            })
        ));
        assert!(format!("{quarantine:?}").contains("has_control: true"));
        let quarantine_receipt = quarantine.reconcile_quarantine().await.unwrap();
        assert_eq!(quarantine_receipt.grant_id(), fixture.binding.grant_id());
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 5);
        assert_eq!(requests[3].body_bytes, requests[4].body_bytes);
        assert_eq!(
            requests[3].body["diagnostic"],
            "pinned quarantine diagnostic"
        );
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
        let grant = reserve_bound(&state, &prepared_fixture).await;
        drop(grant);
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 2);
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
        let mut grant = reserve_bound(&state, &promotion_fixture).await;
        drop(grant.begin_test_promotion().unwrap());
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 2);
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
        let mut grant = reserve_bound(&state, &retained_fixture).await;
        let mut promotion = grant.begin_test_promotion().unwrap();
        let retained = promotion.reconcile_promotion().await.unwrap();
        drop(retained);
        tokio::task::yield_now().await;
        assert_eq!(state.requests.lock().unwrap().len(), 3);
        task.abort();
    }

    #[tokio::test]
    async fn reserve_refusal_and_allocation_identity_mismatch_are_distinct() {
        let (server_url, state, task) = start_server().await;
        let refused_fixture = fixture(&server_url, 1);
        let refusal_receipt = reserve_refusal_receipt(&refused_fixture.reservation());
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(json!({"error": "capacity"}))
                        .with_status(StatusCode::TOO_MANY_REQUESTS),
                ),
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(json!({"schema_version": SCHEMA_VERSION}))
                        .with_status(StatusCode::TOO_MANY_REQUESTS),
                ),
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(refusal_receipt).with_status(StatusCode::CONFLICT),
                ),
            ]),
        );
        let client = DecoderGrantControlClient::new().unwrap();
        let attempt_id = refused_fixture.binding.reservation_attempt_id();
        let mut reserve = client.begin_reserve(refused_fixture.take_reservation());
        assert!(matches!(
            reserve.reconcile_reserve().await,
            Err(EngineGrantError::AmbiguousReserve(_))
        ));
        assert_eq!(reserve.reservation_attempt_id().unwrap(), attempt_id);
        assert!(matches!(
            reserve.reconcile_reserve().await,
            Err(EngineGrantError::AmbiguousReserve(_))
        ));
        assert_eq!(reserve.reservation_attempt_id().unwrap(), attempt_id);
        let refused = reserve.reconcile_reserve().await;
        let Err(error) = &refused else {
            panic!("validated refusal unexpectedly produced a grant");
        };
        assert_error_redacted(error, &[PROMPT_SECRET, &refused_fixture.token]);
        let Err(EngineGrantError::AllocatorRefused(receipt)) = refused else {
            panic!("validated refusal must return its authoritative tombstone");
        };
        assert_eq!(receipt.prefill_id(), refused_fixture.binding.prefill_id());
        assert_eq!(receipt.decoder_id(), refused_fixture.binding.decoder_id());
        assert_eq!(
            receipt.logical_request_chain_id(),
            refused_fixture.binding.request_chain_id()
        );
        assert_eq!(receipt.reservation_attempt_id(), attempt_id);
        assert_eq!(
            receipt.reserve_attempt_digest(),
            refused_fixture.binding.reserve_attempt_digest()
        );
        assert_eq!(receipt.reason_code(), "capacity_exhausted");
        assert_eq!(
            receipt.disposition(),
            DecoderReserveRefusalDisposition::RetryAnotherDecoder
        );
        assert_eq!(
            receipt.receipt_id(),
            uuid("73000000-0000-4000-8000-000000000007")
        );
        assert_eq!(receipt.receipt_digest(), AuthorityDigest([0x73; 32]));
        assert!(receipt.take_once());
        assert!(reserve.reservation_attempt_id().is_err());
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
        let client = DecoderGrantControlClient::new().unwrap();
        let mut reserve = client.begin_reserve(mismatch_fixture.take_reservation());
        let mismatch = reserve.reconcile_reserve().await;
        assert!(matches!(
            mismatch,
            Err(EngineGrantError::AmbiguousReserve(_))
        ));
        task.abort();
    }

    #[tokio::test]
    async fn authoritative_refusal_requires_a_fresh_reserve_attempt() {
        let (server_url, state, task) = start_server().await;
        let first_fixture = fixture(&server_url, 1);
        let second_fixture = fixture(&server_url, 1);
        let first_attempt_id = first_fixture.binding.reservation_attempt_id();
        let second_attempt_id = second_fixture.binding.reservation_attempt_id();
        assert_ne!(first_attempt_id, second_attempt_id);
        install_plans(
            &state,
            plan([
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(reserve_refusal_receipt(&first_fixture.reservation()))
                        .with_status(StatusCode::TOO_MANY_REQUESTS),
                ),
                (
                    format!("{CONTROL_PATH}/reserve"),
                    response(second_fixture.reserve_response.clone()),
                ),
            ]),
        );

        let client = DecoderGrantControlClient::new().unwrap();
        let mut first = client.begin_reserve(first_fixture.take_reservation());
        let Err(EngineGrantError::AllocatorRefused(refusal)) = first.reconcile_reserve().await
        else {
            panic!("first attempt must return its authoritative refusal tombstone");
        };
        assert_eq!(refusal.reservation_attempt_id(), first_attempt_id);
        assert!(first.reservation_attempt_id().is_err());

        let mut second = client.begin_reserve(second_fixture.take_reservation());
        let unbound = second.reconcile_reserve().await.unwrap();
        assert_eq!(unbound.reservation_attempt_id(), second_attempt_id);

        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].body["reservation_attempt_id"],
            first_attempt_id.to_string()
        );
        assert_eq!(
            requests[1].body["reservation_attempt_id"],
            second_attempt_id.to_string()
        );
        assert_ne!(requests[0].body_bytes, requests[1].body_bytes);
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
    fn reserve_refusal_disposition_is_a_closed_wire_enum() {
        let fixture = fixture("http://decoder.test:30001", 1);
        let mut refusal = reserve_refusal_receipt(&fixture.reservation());
        refusal["disposition"] = json!("retry_somewhere_new");
        assert!(serde_json::from_value::<WireReserveRefusalReceipt>(refusal).is_err());
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
