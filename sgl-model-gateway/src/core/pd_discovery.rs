//! Strict process-generation discovery for registered PD workers.

use std::time::Duration;

use once_cell::sync::Lazy;
use reqwest::{redirect::Policy, Client, StatusCode};
use serde::Deserialize;
use thiserror::Error;

use crate::core::{
    HttpOrigin, HttpOriginError, PdProcessAdvertisement, PdProcessAdvertisementError,
    PdProcessRegistration,
};

static DISCOVERY_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .redirect(Policy::none())
        .build()
        .expect("PD discovery HTTP client configuration must be valid")
});

#[derive(Deserialize)]
struct PdServerInfo {
    pd_process: Option<PdProcessAdvertisement>,
}
pub(crate) struct DiscoveredPdProcess {
    pub(crate) registration: PdProcessRegistration,
    pub(crate) advertisement: PdProcessAdvertisement,
}

/// Fetch and validate the process-generation advertisement at one canonical origin.
pub(crate) async fn discover_pd_process(
    origin: &HttpOrigin,
    api_key: &str,
    timeout: Duration,
) -> Result<DiscoveredPdProcess, PdDiscoveryError> {
    if api_key.is_empty() {
        return Err(PdDiscoveryError::MissingApiKey);
    }

    let endpoint = origin.endpoint("/server_info")?;
    let response = DISCOVERY_CLIENT
        .get(endpoint)
        .bearer_auth(api_key)
        .timeout(timeout)
        .send()
        .await
        .map_err(|error| PdDiscoveryError::Request(error.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(PdDiscoveryError::Status(status));
    }

    let server_info = response
        .json::<PdServerInfo>()
        .await
        .map_err(|error| PdDiscoveryError::InvalidResponse(error.to_string()))?;
    let advertisement = server_info
        .pd_process
        .ok_or(PdDiscoveryError::MissingAdvertisement)?;
    let metadata = advertisement.validate()?;
    Ok(DiscoveredPdProcess {
        registration: PdProcessRegistration::new(origin.clone(), metadata),
        advertisement,
    })
}

/// Failure to establish a current, authenticated PD process generation.
#[derive(Debug, Error)]
pub(crate) enum PdDiscoveryError {
    #[error("PD discovery requires a nonempty API key")]
    MissingApiKey,
    #[error("failed to construct the canonical PD discovery endpoint")]
    InvalidEndpoint(#[from] HttpOriginError),
    #[error("PD discovery request failed: {0}")]
    Request(String),
    #[error("PD discovery returned HTTP status {0}")]
    Status(StatusCode),
    #[error("PD discovery returned an invalid response: {0}")]
    InvalidResponse(String),
    #[error("PD discovery response omitted pd_process")]
    MissingAdvertisement,
    #[error("PD discovery returned invalid process metadata")]
    InvalidAdvertisement(#[from] PdProcessAdvertisementError),
}

#[cfg(test)]
mod tests {
    use axum::{
        http::{header::AUTHORIZATION, HeaderMap},
        response::IntoResponse,
        routing::get,
        Json, Router,
    };
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::*;

    async fn server_info(headers: HeaderMap) -> impl IntoResponse {
        if headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            != Some("Bearer secret")
        {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        Json(json!({
            "pd_process": {
                "schema": "v1",
                "launch_instance_id": "10000000-0000-4000-8000-000000000001",
                "role": "decode",
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "model_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "logical_kv_layout_fingerprint": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "kv_dtype": "bf16",
                "page_size": 64,
                "kv_transfer_protocol": "packed-v4",
                "prepared_grant_protocol": "control-v1",
                "prefill_bootstrap_endpoint": null
            }
        }))
        .into_response()
    }

    #[tokio::test]
    async fn discovers_an_authenticated_generation_at_the_canonical_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/server_info", get(server_info)),
            )
            .await
            .unwrap();
        });
        let origin = HttpOrigin::parse(&format!("http://{address}")).unwrap();

        let registration = discover_pd_process(&origin, "secret", Duration::from_secs(2))
            .await
            .unwrap();

        assert_eq!(registration.registration.origin(), &origin);
        assert_eq!(
            registration.registration.metadata().role(),
            crate::core::PdProcessRole::Decode
        );
        server.abort();
    }

    #[tokio::test]
    async fn refuses_redirects_instead_of_changing_discovery_authority() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route(
                    "/server_info",
                    get(|| async {
                        (
                            StatusCode::TEMPORARY_REDIRECT,
                            [("location", "http://other.test/server_info")],
                        )
                    }),
                ),
            )
            .await
            .unwrap();
        });
        let origin = HttpOrigin::parse(&format!("http://{address}")).unwrap();

        assert!(matches!(
            discover_pd_process(&origin, "secret", Duration::from_secs(2)).await,
            Err(PdDiscoveryError::Status(StatusCode::TEMPORARY_REDIRECT))
        ));
        server.abort();
    }
}
