"""HTTP boundary for decoder reservation control and inference attachment."""

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from pydantic import TypeAdapter, ValidationError

from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAttempt,
    DecodeReservationAuthenticationError,
    DecodeReservationConflictError,
    DecodeReservationError,
    DecodeReservationNotFoundError,
    DecodeReservationOperation,
    DecodeReservationValidationError,
    authenticate_process_bearer,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    DecodeReservationSchedulerError,
    TokenizerManager,
)
from sglang.srt.utils.auth import AuthLevel, auth_level

_CONTROL_ROOT = "/_internal/pd/v1/decode-reservations"
_GENERATE_REQUEST_ADAPTER = TypeAdapter(GenerateReqInput)


def install_decode_reservation_routes(
    app: FastAPI,
    get_tokenizer_manager: Callable[[], TokenizerManager],
) -> None:
    """Install authenticated decoder reservation control routes.

    :param app: Inference FastAPI application.
    :param get_tokenizer_manager: Runtime tokenizer manager resolver.
    """

    @app.exception_handler(DecodeReservationAuthenticationError)
    async def handle_decode_authentication_error(
        request: Request,
        error: DecodeReservationAuthenticationError,
    ) -> ORJSONResponse:
        del request, error
        return _error_response(401, "Unauthorized")

    @app.exception_handler(DecodeReservationNotFoundError)
    async def handle_decode_not_found_error(
        request: Request,
        error: DecodeReservationNotFoundError,
    ) -> ORJSONResponse:
        del request
        return _error_response(404, str(error))

    @app.exception_handler(DecodeReservationValidationError)
    async def handle_decode_validation_error(
        request: Request,
        error: DecodeReservationValidationError,
    ) -> ORJSONResponse:
        del request
        return _error_response(400, str(error))

    @app.exception_handler(DecodeReservationConflictError)
    async def handle_decode_conflict_error(
        request: Request,
        error: DecodeReservationConflictError,
    ) -> ORJSONResponse:
        del request
        return _error_response(409, str(error))

    @app.exception_handler(DecodeReservationSchedulerError)
    async def handle_decode_scheduler_error(
        request: Request,
        error: DecodeReservationSchedulerError,
    ) -> ORJSONResponse:
        del request
        return _error_response(_scheduler_error_status(error.error_type), str(error))

    @app.exception_handler(DecodeReservationError)
    async def handle_decode_reservation_error(
        request: Request,
        error: DecodeReservationError,
    ) -> ORJSONResponse:
        del request
        return _error_response(409, str(error))

    @app.post(f"{_CONTROL_ROOT}/reserve")
    async def reserve_decode_reservation(request: Request) -> ORJSONResponse:
        manager = get_tokenizer_manager()
        authenticate_process_bearer(
            request.headers.get("Authorization"),
            manager.server_args.api_key,
        )
        attempt_body = await request.body()
        attempt_wire = _parse_json_mapping(attempt_body, "reserve request")
        attempt = DecodeReservationAttempt.from_value(attempt_wire)
        normalized_request = _normalize_reserve_request(request, attempt)
        response = await manager.reserve_decode_reservation(
            attempt=attempt,
            attempt_wire=attempt_wire,
            obj=normalized_request,
        )
        if response.get("state") != "refused":
            return ORJSONResponse(content=response)
        disposition = response.get("disposition")
        status_code = 409 if disposition == "terminal" else 429
        return ORJSONResponse(status_code=status_code, content=response)

    @app.post(f"{_CONTROL_ROOT}/{{grant_id}}/{{operation}}")
    @auth_level(AuthLevel.ENDPOINT)
    async def mutate_decode_reservation(
        grant_id: uuid.UUID,
        operation: DecodeReservationOperation,
        request: Request,
    ) -> ORJSONResponse:
        manager = get_tokenizer_manager()
        authorization_header = request.headers.get("Authorization")
        request_body = await request.body()
        if operation is DecodeReservationOperation.BIND:
            response = await manager.bind_decode_reservation(
                grant_id,
                authorization_header,
                request_body,
            )
            return ORJSONResponse(content=response)

        transcript = _parse_json_mapping(request_body, f"{operation.value} request")
        if (
            operation is DecodeReservationOperation.CANCEL
            and "grant_digest" not in transcript
        ):
            response = await manager.cancel_unbound_decode_reservation(
                grant_id,
                authorization_header,
                transcript,
            )
        else:
            response = await manager.transition_decode_reservation(
                grant_id,
                authorization_header,
                operation,
                transcript,
            )
        return ORJSONResponse(content=response)


async def attach_native_generate_request(
    tokenizer_manager: TokenizerManager,
    raw_request: Request,
    parsed_request: GenerateReqInput,
) -> GenerateReqInput:
    """Resolve a native request to its prepared object when exact bytes match.

    :param tokenizer_manager: Owning tokenizer manager.
    :param raw_request: Actual inference HTTP request.
    :param parsed_request: Normally parsed non-PD request.
    :returns: Prepared or normally parsed request.
    """

    attached = await tokenizer_manager.attach_decode_inference(
        "/generate",
        await raw_request.body(),
    )
    return parsed_request if attached is None else attached


async def handle_attached_openai_request(
    serving: OpenAIServingChat | OpenAIServingCompletion,
    protocol_request: ChatCompletionRequest | CompletionRequest,
    raw_request: Request,
    inference_route: str,
) -> tuple[bool, Any]:
    """Bypass conversion and tokenization for one exact promoted OpenAI request.

    :param serving: Matching OpenAI serving implementation.
    :param protocol_request: Final request used for response formatting.
    :param raw_request: Actual inference HTTP request.
    :param inference_route: Exact route bound into the grant.
    :returns: Whether an attachment matched and its response.
    """

    attached = await serving.tokenizer_manager.attach_decode_inference(
        inference_route,
        await raw_request.body(),
    )
    if attached is None:
        return False, None
    if protocol_request.stream:
        response = await serving._handle_streaming_request(
            attached,
            protocol_request,
            raw_request,
        )
    else:
        response = await serving._handle_non_streaming_request(
            attached,
            protocol_request,
            raw_request,
        )
    return True, response


def _normalize_reserve_request(
    request: Request,
    attempt: DecodeReservationAttempt,
) -> GenerateReqInput:
    try:
        if attempt.inference_route == "/generate":
            return _GENERATE_REQUEST_ADAPTER.validate_json(attempt.base_request_body)
        if attempt.inference_route == "/v1/completions":
            protocol_request = CompletionRequest.model_validate_json(
                attempt.base_request_body
            )
            serving = request.app.state.openai_serving_completion
        elif attempt.inference_route == "/v1/chat/completions":
            protocol_request = ChatCompletionRequest.model_validate_json(
                attempt.base_request_body
            )
            serving = request.app.state.openai_serving_chat
        else:
            raise DecodeReservationValidationError(
                "reserve request contains an unsupported inference route"
            )
        validation_error = serving._validate_request(protocol_request)
        if validation_error is not None:
            raise DecodeReservationValidationError(validation_error)
        normalized_request, _ = serving._convert_to_internal_request(
            protocol_request,
            None,
        )
        return normalized_request
    except ValidationError as error:
        raise DecodeReservationValidationError(
            "base inference request failed protocol validation"
        ) from error
    except ValueError as error:
        raise DecodeReservationValidationError(str(error)) from error


def _parse_json_mapping(request_body: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(request_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DecodeReservationValidationError(
            f"{name} must contain valid UTF-8 JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise DecodeReservationValidationError(f"{name} must be a JSON object")
    return dict(value)


def _scheduler_error_status(error_type: str) -> int:
    if error_type.endswith("AuthenticationError"):
        return 401
    if error_type.endswith("ValidationError"):
        return 400
    if error_type.endswith("NotFoundError"):
        return 404
    if (
        error_type.endswith("ConflictError")
        or error_type.endswith("ExpiredError")
        or error_type.endswith("InFlightError")
    ):
        return 409
    return 503


def _error_response(status_code: int, message: str) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status_code,
        content={"error": message},
    )
