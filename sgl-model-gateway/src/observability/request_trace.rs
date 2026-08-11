//! Opt-in request-correlated events on the host's monotonic clock.

use std::{
    env,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

const REQUEST_TRACE_ENV: &str = "SGLANG_REQUEST_TRACE";
const REQUEST_TRACE_PREFIX: &str = "SGLANG_REQUEST_TRACE ";
const REQUEST_TRACE_SCHEMA_VERSION: u8 = 1;

static REQUEST_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static REQUEST_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Immutable inputs for one gateway routing decision.
pub struct PdRouteTraceInput<'a> {
    pub logical_request_id: &'a str,
    pub child_request_ids: &'a [Uuid],
    pub bootstrap_request_body: &'a [u8],
    pub group_id: Option<&'a str>,
    pub prefill_url: &'a str,
    pub decoder_url: &'a str,
    pub prefill_load_before_dispatch: usize,
    pub decoder_load_before_dispatch: usize,
}

#[derive(Debug, Error)]
pub enum RequestTraceError {
    #[error("request trace could not read CLOCK_MONOTONIC: {0}")]
    Clock(std::io::Error),
    #[error("request trace could not parse the gateway bootstrap body: {0}")]
    RequestBody(String),
    #[error("request trace serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug, Serialize)]
struct PdRouteTraceRecord<'a> {
    schema_version: u8,
    event: &'static str,
    role: &'static str,
    monotonic_ns: u64,
    sequence: u64,
    hostname: String,
    process_id: u32,
    logical_request_id: &'a str,
    request_ids: Vec<String>,
    bootstrap_rooms: Vec<u64>,
    request_generations: Vec<String>,
    group_id: Option<&'a str>,
    prefill_url: &'a str,
    decoder_url: &'a str,
    prefill_load_before_dispatch: usize,
    decoder_load_before_dispatch: usize,
}

/// Return whether the process was explicitly launched with request tracing.
pub fn enabled() -> bool {
    *REQUEST_TRACE_ENABLED.get_or_init(|| {
        let Some(value) = env::var_os(REQUEST_TRACE_ENV) else {
            return false;
        };
        let normalized = value.to_string_lossy().trim().to_ascii_lowercase();
        match normalized.as_str() {
            "1" | "true" | "yes" | "on" => true,
            "" | "0" | "false" | "no" | "off" => false,
            _ => panic!("{REQUEST_TRACE_ENV} must be one of 1/0, true/false, yes/no, or on/off"),
        }
    })
}

/// Emit one routing decision without changing inference behavior on trace failure.
pub fn emit_pd_route(input: PdRouteTraceInput<'_>) -> Result<(), RequestTraceError> {
    if !enabled() {
        return Ok(());
    }
    let record = build_pd_route_record(
        input,
        monotonic_ns()?,
        REQUEST_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    )?;
    tracing::info!(
        target: "sglang_request_trace",
        "{REQUEST_TRACE_PREFIX}{}",
        serde_json::to_string(&record)?
    );
    Ok(())
}

fn build_pd_route_record(
    input: PdRouteTraceInput<'_>,
    monotonic_ns: u64,
    sequence: u64,
) -> Result<PdRouteTraceRecord<'_>, RequestTraceError> {
    let bootstrap_rooms = parse_bootstrap_rooms(input.bootstrap_request_body)?;
    if bootstrap_rooms.len() != input.child_request_ids.len() {
        return Err(RequestTraceError::RequestBody(format!(
            "{} child request ids but {} bootstrap rooms",
            input.child_request_ids.len(),
            bootstrap_rooms.len()
        )));
    }
    Ok(PdRouteTraceRecord {
        schema_version: REQUEST_TRACE_SCHEMA_VERSION,
        event: "gateway_route_selected",
        role: "gateway",
        monotonic_ns,
        sequence,
        hostname: env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        process_id: std::process::id(),
        logical_request_id: input.logical_request_id,
        request_ids: input
            .child_request_ids
            .iter()
            .map(Uuid::to_string)
            .collect(),
        bootstrap_rooms,
        request_generations: Vec::new(),
        group_id: input.group_id,
        prefill_url: input.prefill_url,
        decoder_url: input.decoder_url,
        prefill_load_before_dispatch: input.prefill_load_before_dispatch,
        decoder_load_before_dispatch: input.decoder_load_before_dispatch,
    })
}

fn parse_bootstrap_rooms(body: &[u8]) -> Result<Vec<u64>, RequestTraceError> {
    let value: Value = serde_json::from_slice(body)?;
    let room_value = value
        .get("bootstrap_room")
        .ok_or_else(|| RequestTraceError::RequestBody("bootstrap_room is absent".to_string()))?;
    match room_value {
        Value::Number(number) => number.as_u64().map(|room| vec![room]).ok_or_else(|| {
            RequestTraceError::RequestBody("bootstrap_room must be an unsigned integer".to_string())
        }),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    RequestTraceError::RequestBody(
                        "bootstrap_room array values must be unsigned integers".to_string(),
                    )
                })
            })
            .collect(),
        _ => Err(RequestTraceError::RequestBody(
            "bootstrap_room must be an integer or array".to_string(),
        )),
    }
}

fn monotonic_ns() -> Result<u64, RequestTraceError> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // CLOCK_MONOTONIC is the one Linux clock shared across processes on this
    // machine. Instant is intentionally unsuitable because its epoch is opaque.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) };
    if result != 0 {
        return Err(RequestTraceError::Clock(std::io::Error::last_os_error()));
    }
    Ok(timestamp.tv_sec as u64 * 1_000_000_000 + timestamp.tv_nsec as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(body: &'a [u8], request_ids: &'a [Uuid]) -> PdRouteTraceInput<'a> {
        PdRouteTraceInput {
            logical_request_id: "logical-request",
            child_request_ids: request_ids,
            bootstrap_request_body: body,
            group_id: Some("prefill-a"),
            prefill_url: "http://127.0.0.1:30001",
            decoder_url: "http://127.0.0.1:31001",
            prefill_load_before_dispatch: 2,
            decoder_load_before_dispatch: 3,
        }
    }

    #[test]
    fn route_record_binds_scalar_room_to_child_request() {
        let request_ids = [Uuid::parse_str("01020304-0506-4708-890a-0b0c0d0e0f10").unwrap()];
        let record =
            build_pd_route_record(input(br#"{"bootstrap_room":41}"#, &request_ids), 123_456, 7)
                .unwrap();

        assert_eq!(record.monotonic_ns, 123_456);
        assert_eq!(record.sequence, 7);
        assert_eq!(record.bootstrap_rooms, vec![41]);
        assert_eq!(record.request_ids, vec![request_ids[0].to_string()]);
        assert_eq!(record.decoder_load_before_dispatch, 3);
    }

    #[test]
    fn route_record_preserves_ordered_batched_rooms() {
        let request_ids = [Uuid::new_v4(), Uuid::new_v4()];
        let record =
            build_pd_route_record(input(br#"{"bootstrap_room":[41,42]}"#, &request_ids), 1, 0)
                .unwrap();

        assert_eq!(record.bootstrap_rooms, vec![41, 42]);
        assert_eq!(
            record.request_ids,
            request_ids.iter().map(Uuid::to_string).collect::<Vec<_>>()
        );
    }

    #[test]
    fn route_record_rejects_unaligned_identity_vectors() {
        let request_ids = [Uuid::new_v4(), Uuid::new_v4()];
        let error = build_pd_route_record(input(br#"{"bootstrap_room":41}"#, &request_ids), 1, 0)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("2 child request ids but 1 bootstrap rooms"));
    }
}
