import asyncio
import types
import unittest
from collections.abc import AsyncIterator
from http import HTTPStatus
from unittest.mock import patch

import orjson
from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.session.event_journal import (
    SessionEventCursor,
    SessionEventJournal,
)
from sglang.srt.session.fencing import SessionFencingRegister
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SESSION_ID = "session-a"
_REQUEST_ID = "request-a"
_DIGEST = "sha256:v1:current"
_DONE_EVENT = b"data: [DONE]\n\n"


def _request(last_event_id: str | None = None) -> Request:
    """Build a native HTTP request with an optional SSE cursor.

    :param last_event_id: Resume cursor header value.
    :returns: Starlette request suitable for the native handler.
    """
    headers: list[tuple[bytes, bytes]] = []
    if last_event_id is not None:
        headers.append((b"last-event-id", last_event_id.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "headers": headers,
        }
    )


def _generate_input() -> GenerateReqInput:
    """Build one stable streaming-session generation identity.

    :returns: Native streaming session request.
    """
    return GenerateReqInput(
        input_ids=[7],
        rid=_REQUEST_ID,
        stream=True,
        session_params={"id": _SESSION_ID},
    )


def _output(completion_tokens: int) -> dict[str, object]:
    """Build scheduler-derived metadata for one cumulative SSE output.

    :param completion_tokens: Cumulative generated token count.
    :returns: Tokenizer-manager output dictionary.
    """
    return {
        "output_ids": list(range(completion_tokens)),
        "meta_info": {
            "finish_reason": None,
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "_session_lineage_generation": 3,
        },
    }


class _JournalTokenizerManager:
    """Deterministic journal-aware tokenizer-manager test surface."""

    session_event_journal: SessionEventJournal
    generate_calls: int
    generation_finished: asyncio.Event
    release_second: asyncio.Event | None

    def __init__(
        self,
        outputs: list[dict[str, object]],
        *,
        journal_size: int = 8,
        pause_before_second: bool = False,
    ) -> None:
        """Initialize deterministic outputs and recovery state.

        :param outputs: Cumulative native generation outputs.
        :param journal_size: Retained data-event count.
        :param pause_before_second: Whether the second output awaits release.
        """
        self.session_event_journal = SessionEventJournal(journal_size)
        self.session_fencing_register = SessionFencingRegister()
        self.generate_calls = 0
        self.generation_finished = asyncio.Event()
        self.release_second = asyncio.Event() if pause_before_second else None
        self._outputs = outputs

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Request | None,
    ) -> AsyncIterator[dict[str, object]]:
        """Emit configured outputs independently of an HTTP connection.

        :param obj: Native generation request.
        :param request: Deliberately absent for journal-owned production.
        :yields: Configured cumulative output dictionaries.
        """
        del obj
        self.generate_calls += 1
        if request is not None:
            raise AssertionError("Session journal producer must own generation.")
        try:
            for index, output in enumerate(self._outputs):
                if index == 1 and self.release_second is not None:
                    await self.release_second.wait()
                yield output
                await asyncio.sleep(0)
        finally:
            self.generation_finished.set()

    async def get_session_info(self, session_id: str) -> types.SimpleNamespace:
        """Return durable coordinates for typed reconciliation errors.

        :param session_id: Requested session identity.
        :returns: Current tip and lineage digest.
        """
        if session_id != _SESSION_ID:
            raise AssertionError(f"Unexpected session {session_id}.")
        return types.SimpleNamespace(tip=103, lineage_digest=_DIGEST)


async def _invoke(
    manager: _JournalTokenizerManager,
    *,
    last_event_id: str | None = None,
) -> tuple[object, list[bytes]]:
    """Invoke and fully consume a native session stream.

    :param manager: Journal-aware manager test double.
    :param last_event_id: Optional resume cursor.
    :returns: Response and emitted bytes.
    """
    prior_state = http_server.get_global_state()
    http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
    try:
        with patch.object(
            http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
            "get",
            return_value=False,
        ):
            response = await http_server.generate_request(
                _generate_input(),
                _request(last_event_id),
            )
        chunks: list[bytes] = []
        if isinstance(response, StreamingResponse):
            chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks
    finally:
        http_server.set_global_state(prior_state)


class StreamingSessionResumeHttpTest(unittest.IsolatedAsyncioTestCase):
    """Native stable SSE identity and resume behavior."""

    async def test_initial_stream_has_stable_absolute_event_ids(self) -> None:
        """Identify every data event with lineage, request, and token interval."""
        manager = _JournalTokenizerManager([_output(1), _output(3)])

        response, chunks = await _invoke(manager)

        expected_ids = (
            SessionEventCursor(3, _REQUEST_ID, 100, 101).encode(),
            SessionEventCursor(3, _REQUEST_ID, 101, 103).encode(),
        )
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].startswith(f"id: {expected_ids[0]}\n".encode()))
        self.assertTrue(chunks[1].startswith(f"id: {expected_ids[1]}\n".encode()))
        self.assertEqual(chunks[2], _DONE_EVENT)
        self.assertEqual(manager.generate_calls, 1)

    async def test_resume_replays_byte_identical_suffix_without_generation(
        self,
    ) -> None:
        """Serve retained bytes and never dispatch a duplicate request."""
        manager = _JournalTokenizerManager([_output(1), _output(3)])
        _, initial = await _invoke(manager)
        first_id = SessionEventCursor(3, _REQUEST_ID, 100, 101).encode()

        response, replay = await _invoke(manager, last_event_id=first_id)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(replay, initial[1:])
        self.assertEqual(manager.generate_calls, 1)

    async def test_behind_cursor_returns_typed_reconciliation_conflict(self) -> None:
        """Return current durable state when the replay cursor is too old."""
        manager = _JournalTokenizerManager(
            [_output(1), _output(2), _output(3)],
            journal_size=1,
        )
        await _invoke(manager)
        stale_cursor = SessionEventCursor(3, _REQUEST_ID, 100, 101).encode()

        response, chunks = await _invoke(manager, last_event_id=stale_cursor)

        self.assertEqual(response.status_code, HTTPStatus.CONFLICT)
        self.assertEqual(chunks, [])
        self.assertEqual(
            orjson.loads(response.body),
            {
                "error": {
                    "message": (
                        "Resume cursor is behind the retained session journal."
                    ),
                    "type": "journal_behind",
                    "code": HTTPStatus.CONFLICT.value,
                    "retryable": False,
                    "current_tip": 103,
                    "current_digest": _DIGEST,
                    "required_action": "full_state_reconciliation",
                }
            },
        )
        self.assertEqual(manager.generate_calls, 1)

    async def test_initial_disconnect_does_not_cancel_journal_producer(self) -> None:
        """Continue generation after the original response iterator closes."""
        manager = _JournalTokenizerManager(
            [_output(1), _output(2)],
            pause_before_second=True,
        )
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            with patch.object(
                http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
                "get",
                return_value=False,
            ):
                response = await http_server.generate_request(
                    _generate_input(),
                    _request(),
                )
            self.assertIsInstance(response, StreamingResponse)
            first = await anext(response.body_iterator)
            await response.body_iterator.aclose()
            assert manager.release_second is not None
            manager.release_second.set()
            await asyncio.wait_for(manager.generation_finished.wait(), timeout=1)

            first_id = SessionEventCursor(3, _REQUEST_ID, 100, 101).encode()
            with patch.object(
                http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
                "get",
                return_value=False,
            ):
                replay_response = await http_server.generate_request(
                    _generate_input(),
                    _request(first_id),
                )
            self.assertIsInstance(replay_response, StreamingResponse)
            replay = [chunk async for chunk in replay_response.body_iterator]
        finally:
            http_server.set_global_state(prior_state)

        self.assertTrue(first.startswith(b"id: "))
        self.assertEqual(len(replay), 2)
        self.assertTrue(replay[0].startswith(b"id: "))
        self.assertEqual(replay[1], _DONE_EVENT)
        self.assertEqual(manager.generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
