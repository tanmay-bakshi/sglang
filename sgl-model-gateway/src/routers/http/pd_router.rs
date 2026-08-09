use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::{
    pd_sse::{SseParser, SseProgress},
    pd_types::api_path,
};
use crate::{
    config::types::RetryConfig,
    core::{
        pd_decoder_directory::PrefillDirectoryEntry,
        pd_decoder_grant::{
            DecoderGrantControlClient, DecoderInferenceRoute, DecoderRequestTemplate,
        },
        HashRing, PdRequestSession, PdReservedRequestSession, Worker, WorkerLoadGuard,
        WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::{GenerationRequest, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub decoder_control: DecoderGrantControlClient,
    pub enable_igw: bool,
    max_response_bytes: usize,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

struct PdTopologyRoutingReceipt {
    topology_sha256: String,
    request_id: String,
    group_id: String,
    prefill_origin: String,
    prefill_launch_instance_id: String,
    decoder_origin: String,
    decoder_launch_instance_id: String,
    kv_transfer_protocol: String,
    prepared_grant_protocol: String,
}

impl PdTopologyRoutingReceipt {
    fn from_session(
        topology_sha256: &str,
        session: &PdReservedRequestSession,
    ) -> Result<Self, String> {
        let group_id = session
            .group_id()
            .ok_or_else(|| "strict topology session is missing its group authority".to_string())?;
        let prefill = session
            .prefill_worker()
            .metadata()
            .pd_process
            .as_ref()
            .ok_or_else(|| "strict topology prefill is missing process metadata".to_string())?;
        let decoder = session
            .decoder_worker()
            .metadata()
            .pd_process
            .as_ref()
            .ok_or_else(|| "strict topology decoder is missing process metadata".to_string())?;
        if prefill.metadata().kv_transfer_protocol() != decoder.metadata().kv_transfer_protocol()
            || prefill.metadata().prepared_grant_protocol()
                != decoder.metadata().prepared_grant_protocol()
        {
            return Err("strict topology pair changed its authenticated PD protocols".to_string());
        }

        Ok(Self {
            topology_sha256: topology_sha256.to_string(),
            request_id: session.request_id().to_string(),
            group_id: group_id.as_str().to_string(),
            prefill_origin: prefill.origin().to_string(),
            prefill_launch_instance_id: prefill.metadata().launch_instance_id().to_string(),
            decoder_origin: decoder.origin().to_string(),
            decoder_launch_instance_id: decoder.metadata().launch_instance_id().to_string(),
            kv_transfer_protocol: prefill
                .metadata()
                .kv_transfer_protocol()
                .as_str()
                .to_string(),
            prepared_grant_protocol: prefill
                .metadata()
                .prepared_grant_protocol()
                .as_str()
                .to_string(),
        })
    }

    fn insert_into(self, response: &mut Response) {
        let headers = response.headers_mut();
        for (name, value) in [
            ("x-sglang-pd-topology-sha256", self.topology_sha256),
            ("x-sglang-pd-request-id", self.request_id),
            ("x-sglang-pd-group-id", self.group_id),
            ("x-sglang-pd-prefill-origin", self.prefill_origin),
            (
                "x-sglang-pd-prefill-launch-instance-id",
                self.prefill_launch_instance_id,
            ),
            ("x-sglang-pd-decoder-origin", self.decoder_origin),
            (
                "x-sglang-pd-decoder-launch-instance-id",
                self.decoder_launch_instance_id,
            ),
            (
                "x-sglang-pd-kv-transfer-protocol",
                self.kv_transfer_protocol,
            ),
            (
                "x-sglang-pd-prepared-grant-protocol",
                self.prepared_grant_protocol,
            ),
        ] {
            headers.insert(
                HeaderName::from_static(name),
                HeaderValue::from_str(&value)
                    .expect("validated topology receipt values are legal HTTP header values"),
            );
        }
    }
}

impl PDRouter {
    fn worker_endpoint_url(worker: &dyn Worker, endpoint: &str) -> String {
        api_path(worker.base_url(), endpoint)
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();

        if let Some(worker) = workers.first() {
            self.proxy_to_worker(worker.as_ref(), endpoint, headers)
                .await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker: &dyn Worker,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = Self::worker_endpoint_url(worker, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let decoder_control = DecoderGrantControlClient::from_builder(Client::builder())
            .map_err(|error| error.to_string())?;
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            decoder_control,
            enable_igw: ctx.router_config.enable_igw,
            max_response_bytes: ctx.router_config.max_payload_size,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    async fn execute_pd_session_dispatch<T: Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        inference_route: DecoderInferenceRoute,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(context.route);
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );

        let request_body = match serde_json::to_vec(original_request) {
            Ok(body) => bytes::Bytes::from(body),
            Err(error) => return Self::handle_serialization_error(error),
        };
        let template = match DecoderRequestTemplate::new(inference_route, request_body) {
            Ok(template) => template,
            Err(error) => {
                return error::bad_request("invalid_pd_request", error.to_string());
            }
        };
        let directory = Arc::clone(self.worker_registry.pd_process_directory());
        let request_id = Uuid::new_v4().to_string();
        let topology_sha256 = self.worker_registry.pd_topology_sha256();
        let session_result = if topology_sha256.is_some() {
            match directory.begin_group_request(&request_id, context.model_id) {
                Ok(group_request) => {
                    PdReservedRequestSession::establish_group(
                        directory,
                        group_request,
                        context.model_id,
                        template,
                        &self.decoder_control,
                        &self.retry_config,
                    )
                    .await
                }
                Err(error) => Err(error.into()),
            }
        } else {
            let selected_prefill = match self
                .select_session_prefill(
                    context.request_text.as_deref(),
                    context.model_id,
                    context.headers.as_ref(),
                )
                .await
            {
                Ok(prefill) => prefill,
                Err(error) => return Self::handle_server_selection_error(error),
            };
            PdReservedRequestSession::establish(
                directory,
                selected_prefill.id(),
                request_id,
                context.model_id,
                template,
                &self.decoder_control,
                &self.retry_config,
            )
            .await
        };
        let session = match session_result {
            Ok(session) => session,
            Err(error) => {
                error!(error = %error, "Failed to establish PD request session");
                return error::service_unavailable(
                    "pd_session_establishment_failed",
                    error.to_string(),
                );
            }
        };

        let topology_receipt = match topology_sha256 {
            Some(topology_sha256) => {
                match PdTopologyRoutingReceipt::from_session(topology_sha256, &session) {
                    Ok(receipt) => Some(receipt),
                    Err(error) => {
                        error!(error, "Failed to construct strict topology routing receipt");
                        return error::internal_error(
                            "pd_topology_receipt_failed",
                            "Strict topology routing receipt could not be constructed",
                        );
                    }
                }
            }
            None => None,
        };

        let mut response = self
            .execute_pd_session_internal(headers, context, session)
            .await;
        if let Some(receipt) = topology_receipt {
            receipt.insert_into(&mut response);
        }
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }
        response
    }

    async fn execute_pd_session_internal(
        &self,
        headers: Option<&HeaderMap>,
        context: PDRequestContext<'_>,
        session: PdReservedRequestSession,
    ) -> Response {
        let prefill = Arc::clone(session.prefill_worker());
        let decode = Arc::clone(session.decoder_worker());
        let request_body = session.request_body();
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(Arc::clone(&prefill), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(Arc::clone(&decode), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);
        let prefill_url = Self::worker_endpoint_url(prefill.as_ref(), context.route);
        let decode_url = Self::worker_endpoint_url(decode.as_ref(), context.route);
        let prefill_request = self.build_post_bytes_with_headers(
            prefill.as_ref(),
            &prefill_url,
            request_body.clone(),
            headers,
        );
        let decode_request =
            self.build_post_bytes_with_headers(decode.as_ref(), &decode_url, request_body, headers);

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        // Promotion installs lifetime accounting before its engine request is polled.
        // Prefill may overlap that round trip, but decode requires confirmed engine authority.
        let mut prefill_fut = Box::pin(prefill_request.send());
        let mut promotion_fut = Box::pin(session.promote());
        let mut prefill_early: Option<Result<reqwest::Response, reqwest::Error>> = None;
        let session = loop {
            tokio::select! {
                biased;
                promotion = &mut promotion_fut => break match promotion {
                    Ok(session) => session,
                    Err(error_value) => {
                        if let Some(prefill_result) = &prefill_early {
                            prefill.record_outcome(prefill_result.as_ref().is_ok_and(|response| {
                                response.status().is_success()
                                    || response.status().is_client_error()
                            }));
                        }
                        drop(prefill_fut);
                        error!(error = %error_value, "Failed to promote PD request session");
                        return error::service_unavailable(
                            "pd_session_establishment_failed",
                            error_value.to_string(),
                        );
                    }
                },
                result = &mut prefill_fut, if prefill_early.is_none() => {
                    prefill_early = Some(result);
                }
            }
        };

        let mut decode_fut = Box::pin(decode_request.send());
        let mut decode_early = None;
        let prefill_result = match prefill_early {
            Some(result) => result,
            None => loop {
                tokio::select! {
                    biased;
                    result = &mut prefill_fut => break result,
                    result = &mut decode_fut, if decode_early.is_none() => {
                        decode_early = Some(result);
                    }
                }
            },
        };

        let prefill_failed = match &prefill_result {
            Ok(response) => !response.status().is_success(),
            Err(_) => true,
        };
        if prefill_failed {
            drop(decode_fut);
            prefill.record_outcome(
                prefill_result
                    .as_ref()
                    .is_ok_and(|response| response.status().is_client_error()),
            );
            if let Err(error) = session.abort("prefill_dispatch_failed").await {
                error!(error = %error, "Failed to terminalize PD session after prefill failure");
                return error::bad_gateway(
                    "pd_session_abort_failed",
                    "Failed to terminalize decoder reservation",
                );
            }
            return match self
                .process_prefill_response(prefill_result, prefill.url(), false)
                .await
            {
                Err(response) => response,
                Ok(_) => error::bad_gateway(
                    "prefill_server_error",
                    "Prefill reported failure but returned a success response",
                ),
            };
        }

        let decode_result = match decode_early {
            Some(result) => result,
            None => decode_fut.await,
        };
        events::RequestReceivedEvent {}.emit();
        let prefill_body = match self
            .process_prefill_response(prefill_result, prefill.url(), context.return_logprob)
            .await
        {
            Ok((_, body)) => body,
            Err(response) => {
                let _ = session.abort("prefill_response_failed").await;
                return response;
            }
        };
        prefill.record_outcome(true);

        let response = match decode_result {
            Ok(response) => response,
            Err(error_value) => {
                decode.record_outcome(false);
                if let Err(error) = session.abort("decode_dispatch_failed").await {
                    error!(error = %error, "Failed to terminalize PD session after decode failure");
                    return error::bad_gateway(
                        "pd_session_abort_failed",
                        "Failed to terminalize decoder reservation",
                    );
                }
                return error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {error_value}"),
                );
            }
        };
        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        if !status.is_success() {
            if !context.is_stream {
                decode.record_outcome(status.is_client_error());
            }
            if let Err(error) = session.abort("decode_response_failed").await {
                error!(error = %error, "Failed to terminalize PD session after decode response failure");
                return error::bad_gateway(
                    "pd_session_abort_failed",
                    "Failed to terminalize decoder reservation",
                );
            }
            return self
                .handle_decode_error_response(response, &context, prefill, decode)
                .await;
        }

        if context.is_stream {
            let prefill_logprobs = if context.return_logprob {
                prefill_body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .and_then(|json| json.pointer("/meta_info/input_token_logprobs").cloned())
            } else {
                None
            };
            let response_headers = header_utils::preserve_response_headers(response.headers());
            return self.create_session_streaming_response(
                response.bytes_stream(),
                status,
                prefill_logprobs,
                context.return_logprob,
                Some(response_headers),
                prefill,
                decode,
                session,
            );
        }

        let response_headers = header_utils::preserve_response_headers(response.headers());
        let decode_body = match response.bytes().await {
            Ok(body) => body,
            Err(error_value) => {
                decode.record_outcome(false);
                let _ = session.abort("decode_body_failed").await;
                return error::bad_gateway(
                    "read_response_failed",
                    format!("Failed to read decode response: {error_value}"),
                );
            }
        };
        decode.record_outcome(true);
        if let Err(error) = session.complete().await {
            error!(error = %error, "Failed to complete PD request session");
            return error::bad_gateway(
                "pd_session_completion_failed",
                "Failed to release decoder reservation",
            );
        }

        let body = if context.return_logprob {
            Self::merge_non_streaming_logprobs(prefill_body.as_ref(), decode_body)
        } else {
            decode_body
        };
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = status;
        *response.headers_mut() = response_headers;
        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    Ok(error_json) => {
                        json!({ "message": error_json, "status": status.as_u16() })
                    }
                    Err(parse_err) => {
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        let preview: String = body_text.chars().take(256).collect();
                        tracing::warn!(
                            "Failed to parse decode error body as JSON from {}: {} \
                             (status={}, body preview: {:?})",
                            decode.url(),
                            parse_err,
                            status.as_u16(),
                            preview
                        );
                        json!({ "message": body_text, "status": status.as_u16() })
                    }
                },
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    /// Builds the text used for cache-aware routing of a chat request.
    ///
    /// This must reflect the *full* conversation (system prompt, prior turns,
    /// the current message and tool context) so that KV-cache prefix matching
    /// routes to the worker that actually shares the most prefix. Using only the
    /// first message ignores the conversation history that drives KV reuse in
    /// multi-turn chats. See https://github.com/sgl-project/sglang/issues/26263.
    ///
    /// Returns `None` when the conversation has no text to route on, preserving
    /// the prior behavior of not feeding an empty key into prefix matching.
    fn build_chat_request_text(body: &ChatCompletionRequest) -> Option<String> {
        // `extract_text_for_routing` walks every message (system, prior turns,
        // current message, tool content) and is the same routing text the regular
        // (non-PD) router uses, keeping cache-aware routing consistent across both.
        let text = body.extract_text_for_routing();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
        )
        .await?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
        .await?;

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    async fn select_session_prefill(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<Arc<PrefillDirectoryEntry>, String> {
        let entries = self
            .worker_registry
            .pd_process_directory()
            .ready_prefills_for_model(model_id);
        if entries.is_empty() {
            return Err("No model-compatible ready prefill process is available".to_string());
        }
        let workers: Vec<Arc<dyn Worker>> = entries
            .iter()
            .map(|entry| Arc::clone(entry.worker()))
            .collect();
        let policy = self.policy_registry.get_prefill_policy();
        let hash_ring = self
            .worker_registry
            .get_hash_ring(model_id.unwrap_or(UNKNOWN_MODEL_ID));
        let selected_index = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None,
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a model-compatible prefill process",
                    policy.name()
                )
            })?;
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy.name(),
        );
        Ok(Arc::clone(&entries[selected_index]))
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Uses select! to race stream.next() against tx.closed() so that
        // when the client disconnects the upstream HTTP connection is dropped
        // promptly, allowing the engine to abort the request.
        // `biased;` drains a ready upstream chunk before observing client
        // disconnect, so a chunk already produced by reqwest reaches the
        // client (and the logprob merger) before we tear the loop down.
        //
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // decode worker's circuit breaker is updated once on drop: success
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // stream cleanly to None and record a spurious success.
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        if !(status.is_success() || status.is_client_error()) {
            tracked.mark_errored();
        }
        let decode_for_log = decode.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        .unwrap_or(chunk)
                                } else {
                                    chunk
                                };

                                // Mark the wrapper completed before the client
                                // send: upstream finished cleanly regardless of
                                // whether the client is still listening, and
                                // the worker deserves the success tick either
                                // way. `mark_completed` is a no-op once Errored
                                // is set, so the synthetic-error path is unaffected.
                                if is_done {
                                    tracked.mark_completed();
                                }

                                if tx.send(Ok(result)).is_err() {
                                    tracing::debug!(
                                        "Receiver dropped (likely client disconnect), \
                                        cancelling upstream PD stream"
                                    );
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                // BreakerTrackedStream already logged the error
                                // and marked the terminal state as Errored so
                                // the worker's circuit breaker will tick on drop.
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, cancelling upstream PD stream from {}",
                            decode_for_log.url()
                        );
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_session_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        session: PdRequestSession,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        let decode_for_log = Arc::clone(&decode);
        let max_response_bytes = self.max_response_bytes;
        tokio::spawn(async move {
            let mut session = Some(session);
            let mut parser = SseParser::new(max_response_bytes)
                .expect("validated router payload limit must be nonzero");
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let progress = match parser.push(&chunk) {
                                    Ok(progress) => progress,
                                    Err(error_value) => {
                                        tracked.mark_errored();
                                        if let Some(active_session) = session.take() {
                                            let _ = active_session
                                                .abort("decode_stream_protocol_failed")
                                                .await;
                                        }
                                        let _ = tx.send(Err(format!(
                                            "Invalid PD decode stream: {error_value}"
                                        )));
                                        break;
                                    }
                                };

                                let (events, terminal) = match progress {
                                    SseProgress::InProgress { events } => (events, false),
                                    SseProgress::Terminal {
                                        events,
                                        consumed_bytes,
                                    } => {
                                        if consumed_bytes != chunk.len() {
                                            tracked.mark_errored();
                                            if let Some(active_session) = session.take() {
                                                let _ = active_session
                                                    .abort("decode_stream_protocol_failed")
                                                    .await;
                                            }
                                            let _ = tx.send(Err(
                                                "Invalid PD decode stream: bytes follow data: [DONE]"
                                                    .to_string(),
                                            ));
                                            break;
                                        }
                                        (events, true)
                                    }
                                };

                                let mut client_connected = true;
                                for payload in events {
                                    let event = Self::encode_session_sse_event(
                                        payload,
                                        prefill_logprobs.as_ref(),
                                        return_logprob,
                                    );
                                    if tx.send(Ok(event)).is_err() {
                                        client_connected = false;
                                        break;
                                    }
                                }
                                if !client_connected {
                                    if let Some(active_session) = session.take() {
                                        let _ = active_session.abort("client_disconnected").await;
                                    }
                                    break;
                                }

                                if terminal {
                                    let active_session = session.take().expect(
                                        "streaming PD session terminalized more than once",
                                    );
                                    if let Err(error_value) = active_session.complete().await {
                                        tracked.mark_errored();
                                        let _ = tx.send(Err(format!(
                                            "PD completion reconciliation failed: {error_value}"
                                        )));
                                        break;
                                    }
                                    tracked.mark_completed();
                                    let _ = tx.send(Ok(bytes::Bytes::from_static(
                                        b"data: [DONE]\n\n",
                                    )));
                                    break;
                                }
                            }
                            Some(Err(error_value)) => {
                                if let Some(active_session) = session.take() {
                                    let _ = active_session.abort("decode_stream_failed").await;
                                }
                                let _ = tx.send(Err(format!("Stream error: {error_value}")));
                                break;
                            }
                            None => {
                                tracked.mark_errored();
                                let stream_error = parser.finish().unwrap_err();
                                if let Some(active_session) = session.take() {
                                    let _ = active_session
                                        .abort("decode_stream_eof_before_done")
                                        .await;
                                }
                                let _ = tx.send(Err(format!(
                                    "PD decode stream ended before data: [DONE]: {stream_error}"
                                )));
                                break;
                            }
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, aborting PD request session from {}",
                            decode_for_log.url()
                        );
                        if let Some(active_session) = session.take() {
                            let _ = active_session.abort("client_disconnected").await;
                        }
                        break;
                    }
                }
            }
        });

        let body = Body::from_stream(UnboundedReceiverStream::new(rx));
        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];
        let mut response = Response::new(body);
        *response.status_mut() = status;
        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;
        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_bytes_with_headers(
        &self,
        worker: &dyn Worker,
        endpoint_url: &str,
        request_body: bytes::Bytes,
        headers: Option<&HeaderMap>,
    ) -> reqwest::RequestBuilder {
        let mut request = self
            .client
            .post(endpoint_url)
            .header(CONTENT_TYPE, "application/json")
            .body(request_body);
        if let Some(authorization) =
            header_utils::extract_auth_header(headers, worker.api_key()).as_ref()
        {
            request = request.header("Authorization", authorization);
        }
        if let Some(headers) = headers {
            for (name, value) in headers {
                if name.as_str().eq_ignore_ascii_case("authorization") {
                    continue;
                }
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(value) = value.to_str() {
                        request = request.header(name, value);
                    }
                }
            }
        }
        request
    }

    fn encode_session_sse_event(
        payload: Vec<u8>,
        prefill_logprobs: Option<&Value>,
        return_logprob: bool,
    ) -> bytes::Bytes {
        let mut event = Vec::with_capacity("data: ".len() + payload.len() + 2);
        event.extend_from_slice(b"data: ");
        event.extend_from_slice(&payload);
        event.extend_from_slice(b"\n\n");
        let event = bytes::Bytes::from(event);
        if !return_logprob || prefill_logprobs.is_none() {
            return event;
        }
        Self::merge_streaming_logprobs(prefill_logprobs.cloned(), &event).unwrap_or(event)
    }

    fn merge_non_streaming_logprobs(
        prefill_body: Option<&bytes::Bytes>,
        decode_body: bytes::Bytes,
    ) -> bytes::Bytes {
        let Some(prefill_body) = prefill_body else {
            return decode_body;
        };
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse PD responses for logprob merging");
            return decode_body;
        };
        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);
        serde_json::to_vec(&decode_json)
            .map(bytes::Bytes::from)
            .unwrap_or(decode_body)
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = Self::worker_endpoint_url(prefill.as_ref(), "health_generate");
        let decode_url = Self::worker_endpoint_url(decode.as_ref(), "health_generate");
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client.get(&decode_url).send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/generate",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_pd_session_dispatch(headers, body, DecoderInferenceRoute::Generate, context)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            Self::build_chat_request_text(body)
        } else {
            None
        };

        // Calculate batch size
        let context = PDRequestContext {
            route: "/v1/chat/completions",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_pd_session_dispatch(
            headers,
            body,
            DecoderInferenceRoute::ChatCompletions,
            context,
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let context = PDRequestContext {
            route: "/v1/completions",
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_pd_session_dispatch(headers, body, DecoderInferenceRoute::Completions, context)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/rerank; returning bad request");
        error::bad_request(
            "pd_unsupported_rerank",
            "PD mode does not support /v1/rerank",
        )
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::core::{
        BasicWorkerBuilder, DPAwareWorkerBuilder, HttpOrigin, KvTransferProtocol, PdMetadataSchema,
        PdProcessMetadata, PdProcessRegistration, PdProcessRole, PrefillBootstrapEndpoint,
        PreparedGrantProtocol, WorkerType,
    };

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            decoder_control: DecoderGrantControlClient::from_builder(Client::builder()).unwrap(),
            enable_igw: false,
            max_response_bytes: 1024 * 1024,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url.clone()).worker_type(worker_type.clone());
        let role = match worker_type {
            WorkerType::Prefill { .. } => Some(PdProcessRole::Prefill),
            WorkerType::Decode => Some(PdProcessRole::Decode),
            WorkerType::Regular => None,
        };
        if let Some(role) = role {
            let metadata = PdProcessMetadata::new(
                PdMetadataSchema::V1,
                Uuid::new_v4(),
                role,
                match role {
                    PdProcessRole::Prefill => 2,
                    PdProcessRole::Decode => 1,
                },
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "bf16",
                64,
                KvTransferProtocol::PackedV4,
                PreparedGrantProtocol::V1,
                match role {
                    PdProcessRole::Prefill => Some(
                        PrefillBootstrapEndpoint::new("prefill-transfer.test", 50_051).unwrap(),
                    ),
                    PdProcessRole::Decode => None,
                },
            )
            .unwrap();
            builder = builder.pd_process(PdProcessRegistration::new(
                HttpOrigin::parse(&url).unwrap(),
                metadata,
            ));
        }
        let worker = builder.build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[test]
    fn topology_receipt_emits_the_frozen_response_headers() {
        let mut response = Response::new(Body::empty());
        PdTopologyRoutingReceipt {
            topology_sha256: "24afeedde0264f2874a77f18266ad57b096e397c43bc65e16bd46334b52df5a2"
                .to_string(),
            request_id: "request-0".to_string(),
            group_id: "group-0".to_string(),
            prefill_origin: "http://prefill.test:30000".to_string(),
            prefill_launch_instance_id: "00000000-0000-0000-0000-000000000001".to_string(),
            decoder_origin: "http://decode.test:30001".to_string(),
            decoder_launch_instance_id: "00000000-0000-0000-0000-000000000002".to_string(),
            kv_transfer_protocol: "packed-v4".to_string(),
            prepared_grant_protocol: "control-v1".to_string(),
        }
        .insert_into(&mut response);

        for (name, expected) in [
            (
                "x-sglang-pd-topology-sha256",
                "24afeedde0264f2874a77f18266ad57b096e397c43bc65e16bd46334b52df5a2",
            ),
            ("x-sglang-pd-request-id", "request-0"),
            ("x-sglang-pd-group-id", "group-0"),
            ("x-sglang-pd-prefill-origin", "http://prefill.test:30000"),
            (
                "x-sglang-pd-prefill-launch-instance-id",
                "00000000-0000-0000-0000-000000000001",
            ),
            ("x-sglang-pd-decoder-origin", "http://decode.test:30001"),
            (
                "x-sglang-pd-decoder-launch-instance-id",
                "00000000-0000-0000-0000-000000000002",
            ),
            ("x-sglang-pd-kv-transfer-protocol", "packed-v4"),
            ("x-sglang-pd-prepared-grant-protocol", "control-v1"),
        ] {
            assert_eq!(response.headers().get(name).unwrap(), expected);
        }
    }

    #[test]
    fn test_chat_request_text_uses_full_conversation() {
        // Regression test for https://github.com/sgl-project/sglang/issues/26263
        // Cache-aware routing must build its text from the full conversation, not
        // just the first message, so that KV-cache prefix matching reflects what
        // the worker will actually process in a multi-turn chat.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "First question about apples."},
                {"role": "assistant", "content": "Apples are red."},
                {"role": "user", "content": "Follow up question about oranges."}
            ]
        }))
        .expect("valid chat request");

        let text = PDRouter::build_chat_request_text(&body)
            .expect("multi-message chat should produce routing text");

        assert!(
            text.contains("apples"),
            "routing text must include earlier turns, got: {text:?}"
        );
        assert!(
            text.contains("oranges"),
            "routing text must include later turns (not only the first message), got: {text:?}"
        );
    }

    #[test]
    fn test_chat_request_text_none_when_no_text() {
        // When the conversation carries no text content, no routing text should
        // be produced (None) rather than an empty string, preserving the prior
        // PD behavior. See https://github.com/sgl-project/sglang/issues/26263.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": ""}
            ]
        }))
        .expect("valid chat request");

        assert!(
            PDRouter::build_chat_request_text(&body).is_none(),
            "empty conversation text should produce None, not Some(\"\")"
        );
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register(Arc::from(unhealthy_worker))
            .unwrap();
        router
            .worker_registry
            .register(Arc::from(healthy_worker))
            .unwrap();
        router
            .worker_registry
            .register(Arc::from(decode_worker))
            .unwrap();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_endpoint_url_uses_base_url_for_dp_aware_worker() {
        let worker = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();

        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "health_generate"),
            "http://prefill:30000/health_generate"
        );
        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "/v1/models"),
            "http://prefill:30000/v1/models"
        );
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router
            .worker_registry
            .register(Arc::from(prefill_worker))
            .unwrap();
        router
            .worker_registry
            .register(Arc::from(decode_worker))
            .unwrap();

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}
