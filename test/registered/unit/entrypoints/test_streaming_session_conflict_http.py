import asyncio
import types
import unittest
from collections.abc import AsyncIterator
from http import HTTPStatus
from unittest.mock import AsyncMock, patch

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput
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
_OBSERVED_DIGEST = "sha256:v1:observed"
_OBSERVED_TIP = 128


class _DeterministicTokenizerManager:
    """Minimal tokenizer-manager surface with a deterministic terminal outcome."""

    error: Exception | None
    terminal_completion: asyncio.Event | None

    def __init__(self, error: Exception | None) -> None:
        """Initialize the deterministic manager.

        :param error: Optional exception raised when generation is consumed.
        """
        self.error = error
        self.terminal_completion = None

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Request,
    ) -> AsyncIterator[dict[str, object]]:
        """Raise the configured error or emit one terminal response.

        :param obj: Parsed native generation request.
        :param request: Incoming HTTP request.
        :yields: One terminal output when no exception was configured.
        """
        del obj, request
        if self.error is not None:
            raise self.error
        yield {
            "output_ids": [1],
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    def create_abort_task(
        self,
        obj: GenerateReqInput,
        terminal_completion: asyncio.Event | None = None,
    ) -> None:
        """Capture the response-local completion signal.

        :param obj: Parsed native generation request.
        :param terminal_completion: Completion signal owned by the HTTP response.
        """
        del obj
        self.terminal_completion = terminal_completion


async def _start_generate(
    error: Exception | None,
    *,
    stream: bool,
    disconnected: bool = False,
) -> tuple[Response, _DeterministicTokenizerManager, object]:
    """Start the native HTTP handler against a deterministic outcome.

    :param error: Optional error raised by the tokenizer-manager test double.
    :param stream: Whether to exercise the SSE response path.
    :param disconnected: Whether the request reports a client disconnect.
    :returns: HTTP response, lifecycle-aware manager, and prior global state.
    """
    manager = _DeterministicTokenizerManager(error)
    request = AsyncMock(spec=Request)
    request.body.return_value = b"{}"
    request.is_disconnected.return_value = disconnected
    obj = GenerateReqInput(
        input_ids=[1],
        rid=_CORRELATION_ID,
        stream=stream,
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
    with patch.object(
        http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
        "get",
        return_value=False,
    ):
        response = await http_server.generate_request(obj, request)
    return response, manager, prior_state


async def _invoke_generate(
    error: Exception | None,
    *,
    stream: bool,
    disconnected: bool = False,
) -> tuple[Response, list[bytes], _DeterministicTokenizerManager]:
    """Consume a deterministic native HTTP response.

    :param error: Optional error raised by the tokenizer-manager test double.
    :param stream: Whether to exercise the SSE response path.
    :param disconnected: Whether the request reports a client disconnect.
    :returns: HTTP response, emitted body chunks, and lifecycle-aware manager.
    """
    response, manager, prior_state = await _start_generate(
        error,
        stream=stream,
        disconnected=disconnected,
    )
    try:
        chunks: list[bytes] = []
        if isinstance(response, StreamingResponse):
            chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks, manager
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
                    "observed_tip": _OBSERVED_TIP,
                    "observed_digest": _OBSERVED_DIGEST,
                }
            }
        }

        with self.assertRaises(StreamingSessionConflictError) as raised:
            await manager._handle_abort_finish_reason(out, state, is_stream=True)

        self.assertEqual(str(raised.exception), _CONFLICT_MESSAGE)
        self.assertEqual(raised.exception.correlation_id, _CORRELATION_ID)
        self.assertEqual(raised.exception.observed_tip, _OBSERVED_TIP)
        self.assertEqual(raised.exception.observed_digest, _OBSERVED_DIGEST)

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
                "observed_tip": _OBSERVED_TIP,
                "observed_digest": _OBSERVED_DIGEST,
            }
        }

    async def test_non_stream_conflict_returns_http_409(self) -> None:
        """Return an actual HTTP 409 and the stable public envelope."""
        response, chunks, _ = await _invoke_generate(
            StreamingSessionConflictError(
                _CONFLICT_MESSAGE,
                _CORRELATION_ID,
                _OBSERVED_TIP,
                _OBSERVED_DIGEST,
            ),
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
        response, chunks, manager = await _invoke_generate(
            StreamingSessionConflictError(
                _CONFLICT_MESSAGE,
                _CORRELATION_ID,
                _OBSERVED_TIP,
                _OBSERVED_DIGEST,
            ),
            stream=True,
        )

        expected_event = (
            b"data: " + http_server.dumps_json(self._expected_payload()) + b"\n\n"
        )
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(chunks, [expected_event, b"data: [DONE]\n\n"])
        self.assertIsNotNone(manager.terminal_completion)
        self.assertTrue(manager.terminal_completion.is_set())

    async def test_existing_non_stream_400_body_is_unchanged(self) -> None:
        """Snapshot the pre-existing non-stream ValueError response bytes."""
        response, chunks, _ = await _invoke_generate(
            ValueError("old bad request"),
            stream=False,
        )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(response.body, b'{"error":{"message":"old bad request"}}')
        self.assertEqual(chunks, [])

    async def test_existing_stream_400_body_is_unchanged(self) -> None:
        """Snapshot the pre-existing SSE ValueError response bytes."""
        response, chunks, manager = await _invoke_generate(
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
        self.assertIsNotNone(manager.terminal_completion)
        self.assertTrue(manager.terminal_completion.is_set())

    async def test_terminal_output_marks_completion_before_delivery_finishes(
        self,
    ) -> None:
        """Suppress cleanup as soon as a terminal scheduler chunk is observed."""
        response, manager, prior_state = await _start_generate(None, stream=True)
        try:
            self.assertIsInstance(response, StreamingResponse)
            first_chunk = await anext(response.body_iterator)

            self.assertIn(b'"finish_reason":{"type":"stop"}', first_chunk)
            self.assertIsNotNone(manager.terminal_completion)
            self.assertTrue(manager.terminal_completion.is_set())
            await response.body_iterator.aclose()
        finally:
            http_server.set_global_state(prior_state)

    async def test_client_disconnect_does_not_mark_terminal_completion(self) -> None:
        """Leave genuine disconnect cleanup armed when generation is unfinished."""
        response, chunks, manager = await _invoke_generate(
            ValueError("client disconnected"),
            stream=True,
            disconnected=True,
        )

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(chunks, [])
        self.assertIsNotNone(manager.terminal_completion)
        self.assertFalse(manager.terminal_completion.is_set())

    async def test_terminal_server_error_marks_completion_before_propagating(
        self,
    ) -> None:
        """Suppress delayed cleanup for scheduler-terminal server errors."""
        error = http_server.HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="scheduler terminalized request",
        )
        response, manager, prior_state = await _start_generate(error, stream=True)
        try:
            self.assertIsInstance(response, StreamingResponse)
            with self.assertRaises(http_server.HTTPException) as raised:
                await anext(response.body_iterator)

            self.assertIs(raised.exception, error)
            self.assertIsNotNone(manager.terminal_completion)
            self.assertTrue(manager.terminal_completion.is_set())
        finally:
            http_server.set_global_state(prior_state)


class StreamingDisconnectAbortGuardTest(unittest.IsolatedAsyncioTestCase):
    """Completion-aware delayed abort behavior in multi-tokenizer mode."""

    @staticmethod
    def _manager() -> tuple[TokenizerManager, list[AbortReq]]:
        """Build the minimum real manager surface used by delayed cleanup.

        :returns: Tokenizer manager and its dispatched abort messages.
        """
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = types.SimpleNamespace(tokenizer_worker_num=2)
        manager.rid_to_state = {}
        manager.enable_metrics = False
        dispatched: list[AbortReq] = []
        manager._dispatch_to_scheduler = dispatched.append
        return manager, dispatched

    @staticmethod
    def _request() -> GenerateReqInput:
        """Build one normalized single-request shape.

        :returns: Request with the tokenizer-derived single-request marker.
        """
        request = GenerateReqInput(
            input_ids=[1],
            rid=_CORRELATION_ID,
            stream=True,
        )
        request.is_single = True
        return request

    async def test_terminal_conflict_cannot_abort_corrected_same_rid_retry(
        self,
    ) -> None:
        """Do not let terminal cleanup cross into a newer use of the same rid."""
        manager, dispatched = self._manager()
        completion = asyncio.Event()
        request = self._request()
        background = manager.create_abort_task(
            request,
            terminal_completion=completion,
        )

        completion.set()
        manager.rid_to_state[_CORRELATION_ID] = object()
        sleep = AsyncMock()
        with patch.object(asyncio, "sleep", sleep):
            await background()

        sleep.assert_not_awaited()
        self.assertEqual(dispatched, [])

    async def test_completion_during_grace_period_suppresses_abort(self) -> None:
        """Recheck completion after the disconnect grace period."""
        manager, dispatched = self._manager()
        completion = asyncio.Event()
        request = self._request()
        background = manager.create_abort_task(
            request,
            terminal_completion=completion,
        )
        manager.rid_to_state[_CORRELATION_ID] = object()

        async def complete_during_sleep(delay: float) -> None:
            """Mark terminal completion during the cleanup grace period.

            :param delay: Configured grace period.
            """
            self.assertEqual(delay, 2)
            completion.set()

        with patch.object(asyncio, "sleep", complete_during_sleep):
            await background()

        self.assertEqual(dispatched, [])

    async def test_unfinished_disconnect_still_dispatches_abort(self) -> None:
        """Preserve delayed aborts for a genuinely unfinished response."""
        manager, dispatched = self._manager()
        completion = asyncio.Event()
        request = self._request()
        background = manager.create_abort_task(
            request,
            terminal_completion=completion,
        )

        with (
            patch.object(asyncio, "sleep", AsyncMock()),
            patch(
                "sglang.srt.managers.tokenizer_manager.get_serving",
                return_value=manager.server_args,
            ),
        ):
            await background()

        self.assertEqual(len(dispatched), 1)
        self.assertIsInstance(dispatched[0], AbortReq)
        self.assertEqual(dispatched[0].rid, _CORRELATION_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
