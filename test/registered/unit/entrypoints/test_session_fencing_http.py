import types
import unittest
from collections.abc import AsyncIterator
from http import HTTPStatus
from unittest.mock import patch

import orjson
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    GenerateReqInput,
    InstallSessionFencingReqInput,
    OpenSessionReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.session.errors import (
    STREAMING_SESSION_STALE_EPOCH_ERROR_TYPE,
    StreamingSessionStaleEpochError,
)
from sglang.srt.session.event_journal import SessionEventCursor, SessionEventJournal
from sglang.srt.session.fencing import (
    CLUSTER_INCARNATION_HEADER,
    FENCING_EPOCH_HEADER,
    SessionFencingRegister,
    SessionFencingState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _request(epoch: str | None = None) -> Request:
    """Build an HTTP request with an optional fencing epoch.

    :param epoch: Raw epoch header value.
    :returns: Starlette request for a native session endpoint.
    """
    headers: list[tuple[bytes, bytes]] = []
    if epoch is not None:
        headers.append((FENCING_EPOCH_HEADER.lower().encode(), epoch.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "headers": headers,
        }
    )


class _FencedTokenizerManager:
    """Deterministic manager enforcing one installed register."""

    def __init__(self) -> None:
        """Install the test register and initialize mutation observations."""
        self.session_fencing_register = SessionFencingRegister()
        self.session_fencing_register.install(5, 9)
        self.session_event_journal = SessionEventJournal(8)
        self.generate_epochs: list[int | None] = []
        self.opened = False
        self.closed = False

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Request | None,
    ) -> AsyncIterator[dict[str, object]]:
        """Validate the transported header and emit one terminal output.

        :param obj: Native generation request carrying internal session params.
        :param request: Original request, absent for journal-owned production.
        :yields: One deterministic terminal output after fencing.
        """
        del request
        assert obj.session_params is not None
        epoch = obj.session_params["epoch"]
        self.generate_epochs.append(epoch)
        self.session_fencing_register.validate(
            epoch,
            lineage_generation=2,
            observed_tip=40,
        )
        yield {
            "output_ids": [1],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 40,
                "completion_tokens": 1,
                "_session_lineage_generation": 2,
            },
        }

    async def open_session(
        self,
        obj: OpenSessionReqInput,
        request: Request,
    ) -> str:
        """Validate before recording an open mutation.

        :param obj: Session open request.
        :param request: Incoming HTTP request.
        :returns: Configured session identity.
        """
        del request
        self.session_fencing_register.validate(obj.epoch)
        self.opened = True
        assert obj.session_id is not None
        return obj.session_id

    async def close_session(
        self,
        obj: CloseSessionReqInput,
        request: Request,
    ) -> None:
        """Validate before recording a close mutation.

        :param obj: Session close request.
        :param request: Incoming HTTP request.
        """
        del request
        self.session_fencing_register.validate(obj.epoch)
        self.closed = True

    async def install_session_fencing_register(
        self,
        obj: InstallSessionFencingReqInput,
    ) -> SessionFencingState:
        """Install the exact administrative pair.

        :param obj: Requested register value.
        :returns: Installed immutable register state.
        """
        return self.session_fencing_register.install(
            obj.epoch,
            obj.cluster_incarnation,
        )


class SessionFencingReconstructionTest(unittest.IsolatedAsyncioTestCase):
    """Scheduler finish-reason reconstruction at the tokenizer boundary."""

    async def test_stale_epoch_finish_reason_is_typed(self) -> None:
        """Reconstruct every register and stream-coordinate field."""
        manager = TokenizerManager.__new__(TokenizerManager)
        state = types.SimpleNamespace(obj=GenerateReqInput(input_ids=[1], rid="stale"))
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": "stale",
                    "status_code": HTTPStatus.CONFLICT,
                    "err_type": STREAMING_SESSION_STALE_EPOCH_ERROR_TYPE,
                    "request_epoch": 4,
                    "registered_epoch": 5,
                    "cluster_incarnation": 9,
                    "lineage_generation": 2,
                    "observed_tip": 40,
                }
            }
        }

        with self.assertRaises(StreamingSessionStaleEpochError) as raised:
            await manager._handle_abort_finish_reason(out, state, is_stream=True)

        error = raised.exception
        self.assertEqual(error.request_epoch, 4)
        self.assertEqual(error.registered_epoch, 5)
        self.assertEqual(error.cluster_incarnation, 9)
        self.assertEqual(error.lineage_generation, 2)
        self.assertEqual(error.observed_tip, 40)


class SessionFencingHttpTest(unittest.IsolatedAsyncioTestCase):
    """Header transport, typed errors, and response register echoes."""

    async def asyncSetUp(self) -> None:
        """Install a deterministic manager as HTTP global state."""
        self.manager = _FencedTokenizerManager()
        self.prior_state = http_server.get_global_state()
        http_server.set_global_state(
            types.SimpleNamespace(tokenizer_manager=self.manager)
        )
        self.header_patch = patch.object(
            http_server.envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES,
            "get",
            return_value=False,
        )
        self.header_patch.start()

    async def asyncTearDown(self) -> None:
        """Restore HTTP global state and environment access."""
        self.header_patch.stop()
        http_server.set_global_state(self.prior_state)

    def _assert_register_headers(self, response: Response) -> None:
        """Assert the response echoes the installed test register.

        :param response: Native session API response.
        """
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "5")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "9")

    async def test_non_stream_stale_epoch_is_typed_and_header_wins(self) -> None:
        """Reject the header epoch even if a body value attempts to override it."""
        obj = GenerateReqInput(
            input_ids=[1],
            rid="stale",
            stream=False,
            session_params={"id": "session", "epoch": 99},
        )

        response = await http_server.generate_request(obj, _request("4"))

        self.assertEqual(response.status_code, HTTPStatus.CONFLICT)
        self._assert_register_headers(response)
        self.assertEqual(self.manager.generate_epochs, [4])
        self.assertEqual(
            orjson.loads(response.body)["error"],
            {
                "message": "Stale session epoch 4: engine fencing register is (5, 9).",
                "type": "stale_epoch",
                "code": HTTPStatus.CONFLICT.value,
                "retryable": False,
                "request_epoch": 4,
                "registered_epoch": 5,
                "cluster_incarnation": 9,
            },
        )

    async def test_higher_epoch_passes_without_advancing_register(self) -> None:
        """Allow a higher request while leaving installation unchanged."""
        obj = GenerateReqInput(
            input_ids=[1],
            rid="higher",
            stream=False,
            session_params={"id": "session"},
        )

        response = await http_server.generate_request(obj, _request("8"))

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self._assert_register_headers(response)
        self.assertEqual(self.manager.generate_epochs, [8])
        self.assertEqual(self.manager.session_fencing_register.state.epoch, 5)

    async def test_stale_sse_event_has_stable_identity(self) -> None:
        """Journal a typed stale rejection at the current absolute tip."""
        obj = GenerateReqInput(
            input_ids=[1],
            rid="stale-stream",
            stream=True,
            session_params={"id": "session"},
        )

        response = await http_server.generate_request(obj, _request("4"))
        self.assertIsInstance(response, StreamingResponse)
        chunks = [chunk async for chunk in response.body_iterator]

        expected_id = SessionEventCursor(
            lineage_generation=2,
            request_id="stale-stream",
            start=40,
            end=40,
        ).encode()
        self._assert_register_headers(response)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith(f"id: {expected_id}\n".encode()))
        self.assertIn(b'"type":"stale_epoch"', chunks[0])
        self.assertEqual(chunks[1], b"data: [DONE]\n\n")

    async def test_open_and_close_reject_before_mutation(self) -> None:
        """Apply the same header fence to explicit session lifecycle mutations."""
        open_response = await http_server.open_session(
            OpenSessionReqInput(
                capacity_of_str_len=0,
                session_id="session",
                streaming=True,
            ),
            _request("4"),
        )
        close_response = await http_server.close_session(
            CloseSessionReqInput(session_id="session"),
            _request("4"),
        )

        self.assertEqual(open_response.status_code, HTTPStatus.CONFLICT)
        self.assertEqual(close_response.status_code, HTTPStatus.CONFLICT)
        self._assert_register_headers(open_response)
        self._assert_register_headers(close_response)
        self.assertFalse(self.manager.opened)
        self.assertFalse(self.manager.closed)

    async def test_install_endpoint_moves_and_echoes_exact_register(self) -> None:
        """Confirm the administrative installation before returning."""
        response = await http_server.install_session_fencing_register(
            InstallSessionFencingReqInput(epoch=12, cluster_incarnation=34)
        )

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(
            orjson.loads(response.body),
            {"epoch": 12, "cluster_incarnation": 34},
        )
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "12")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "34")

    async def test_invalid_epoch_header_returns_400_with_register(self) -> None:
        """Reject malformed epoch headers without entering the manager."""
        response = await http_server.open_session(
            OpenSessionReqInput(
                capacity_of_str_len=0,
                session_id="session",
                streaming=True,
            ),
            _request("not-an-integer"),
        )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self._assert_register_headers(response)
        self.assertFalse(self.manager.opened)


if __name__ == "__main__":
    unittest.main()
