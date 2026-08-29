import types
import unittest
from collections.abc import AsyncIterator
from http import HTTPStatus
from unittest.mock import AsyncMock, patch

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.session.errors import (
    STREAMING_SESSION_CONFLICT_ERROR_TYPE,
    StreamingSessionConflictError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_CORRELATION_ID = "conflict-request-rid"
_CONFLICT_MESSAGE = (
    "Streaming session expected_tip conflict for session session-a: "
    "expected 120, current tip is 128."
)


class _FailingTokenizerManager:
    """Minimal tokenizer-manager surface that fails generation deterministically."""

    error: Exception

    def __init__(self, error: Exception) -> None:
        """Initialize the failing manager.

        :param error: Exception raised when generation is consumed.
        """
        self.error = error

    async def attach_decode_inference(
        self,
        inference_route: str,
        request_body: bytes,
    ) -> GenerateReqInput | None:
        """Decline decode-reservation attachment.

        :param inference_route: Native inference route.
        :param request_body: Exact incoming request body.
        :returns: No attached request.
        """
        del inference_route, request_body
        return None

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Request,
    ) -> AsyncIterator[dict[str, object]]:
        """Raise the configured transport error.

        :param obj: Parsed native generation request.
        :param request: Incoming HTTP request.
        :yields: No values because generation fails before output.
        """
        del obj, request
        raise self.error
        yield {}

    def create_abort_task(self, obj: GenerateReqInput) -> None:
        """Return no disconnect cleanup for the synthetic request.

        :param obj: Parsed native generation request.
        """
        del obj


async def _invoke_generate(
    error: Exception,
    *,
    stream: bool,
) -> tuple[Response, list[bytes]]:
    """Invoke the native HTTP handler against a deterministic failure.

    :param error: Error raised by the tokenizer-manager test double.
    :param stream: Whether to exercise the SSE response path.
    :returns: HTTP response and any emitted body chunks.
    """
    manager = _FailingTokenizerManager(error)
    request = AsyncMock(spec=Request)
    request.body.return_value = b"{}"
    request.is_disconnected.return_value = False
    obj = GenerateReqInput(
        input_ids=[1],
        rid=_CORRELATION_ID,
        stream=stream,
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
    try:
        with patch.object(
            http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
            "get",
            return_value=False,
        ):
            response = await http_server.generate_request(obj, request)
        chunks: list[bytes] = []
        if isinstance(response, StreamingResponse):
            chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks
    finally:
        http_server.set_global_state(prior_state)


class StreamingSessionConflictReconstructionTest(unittest.IsolatedAsyncioTestCase):
    """Tokenizer-manager reconstruction for the typed scheduler conflict."""

    async def test_typed_conflict_reconstructs_with_correlation_id(self) -> None:
        """Reconstruct the public conflict from the typed finish reason."""
        manager = TokenizerManager.__new__(TokenizerManager)
        state = types.SimpleNamespace(
            obj=GenerateReqInput(input_ids=[1], rid=_CORRELATION_ID)
        )
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": _CONFLICT_MESSAGE,
                    "status_code": HTTPStatus.CONFLICT,
                    "err_type": STREAMING_SESSION_CONFLICT_ERROR_TYPE,
                }
            }
        }

        with self.assertRaises(StreamingSessionConflictError) as raised:
            await manager._handle_abort_finish_reason(out, state, is_stream=True)

        self.assertEqual(str(raised.exception), _CONFLICT_MESSAGE)
        self.assertEqual(raised.exception.correlation_id, _CORRELATION_ID)

    async def test_unmarked_conflict_is_not_promoted(self) -> None:
        """Leave unrelated 409 aborts on the pre-existing generic path."""
        manager = TokenizerManager.__new__(TokenizerManager)
        state = types.SimpleNamespace(
            obj=GenerateReqInput(input_ids=[1], rid=_CORRELATION_ID)
        )
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": "another conflict",
                    "status_code": HTTPStatus.CONFLICT,
                    "err_type": "AnotherConflictError",
                }
            }
        }

        result = await manager._handle_abort_finish_reason(out, state, is_stream=True)

        self.assertIsNone(result)


class StreamingSessionConflictHttpTest(unittest.IsolatedAsyncioTestCase):
    """Native JSON and SSE conflict-envelope behavior."""

    def _expected_payload(self) -> dict[str, dict[str, object]]:
        """Build the frozen public conflict payload.

        :returns: Expected payload shared by both transports.
        """
        return {
            "error": {
                "message": _CONFLICT_MESSAGE,
                "type": "streaming_session_conflict",
                "code": HTTPStatus.CONFLICT.value,
                "retryable": False,
                "correlation_id": _CORRELATION_ID,
            }
        }

    async def test_non_stream_conflict_returns_http_409(self) -> None:
        """Return an actual HTTP 409 and the stable public envelope."""
        response, chunks = await _invoke_generate(
            StreamingSessionConflictError(_CONFLICT_MESSAGE, _CORRELATION_ID),
            stream=False,
        )

        self.assertEqual(response.status_code, HTTPStatus.CONFLICT)
        self.assertEqual(
            response.body,
            http_server.dumps_json(self._expected_payload()),
        )
        self.assertEqual(chunks, [])

    async def test_stream_conflict_emits_only_error_then_done(self) -> None:
        """Keep HTTP 200 while emitting one 409-coded SSE event and DONE."""
        response, chunks = await _invoke_generate(
            StreamingSessionConflictError(_CONFLICT_MESSAGE, _CORRELATION_ID),
            stream=True,
        )

        expected_event = (
            b"data: " + http_server.dumps_json(self._expected_payload()) + b"\n\n"
        )
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(chunks, [expected_event, b"data: [DONE]\n\n"])

    async def test_existing_non_stream_400_body_is_unchanged(self) -> None:
        """Snapshot the pre-existing non-stream ValueError response bytes."""
        response, chunks = await _invoke_generate(
            ValueError("old bad request"),
            stream=False,
        )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(response.body, b'{"error":{"message":"old bad request"}}')
        self.assertEqual(chunks, [])

    async def test_existing_stream_400_body_is_unchanged(self) -> None:
        """Snapshot the pre-existing SSE ValueError response bytes."""
        response, chunks = await _invoke_generate(
            ValueError("old bad request"),
            stream=True,
        )

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(
            chunks,
            [
                (
                    b'data: {"error":{"message":"old bad request","type":'
                    b'"invalid_request_error","code":400,"retryable":false}}\n\n'
                ),
                b"data: [DONE]\n\n",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
