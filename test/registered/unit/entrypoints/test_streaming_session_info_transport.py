import asyncio
import json
import types
import unittest

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import (
    GetSessionInfoReqErrorOutput,
    GetSessionInfoReqInput,
    GetSessionInfoReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.session.errors import StreamingSessionInfoUnavailableError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _session_info_output(
    correlation_id: str,
    *,
    tip: int,
) -> GetSessionInfoReqOutput:
    """Build one deterministic scheduler response.

    :param correlation_id: Internal control-plane request identity.
    :param tip: Durable session tip represented by the response.
    :returns: Complete session-info IPC response.
    """
    return GetSessionInfoReqOutput(
        correlation_id=correlation_id,
        exists=True,
        tip=tip,
        floor=tip,
        protected=0,
        inflight=False,
        held_tokens=tip,
        last_rid=f"rid-{tip}",
    )


class _SessionInfoTokenizerManager:
    """Minimal tokenizer-manager surface for the HTTP endpoint."""

    response: GetSessionInfoReqOutput
    requested_session_id: str | None

    def __init__(self, response: GetSessionInfoReqOutput) -> None:
        """Initialize the deterministic manager.

        :param response: Snapshot returned by every query.
        """
        self.response = response
        self.requested_session_id = None

    async def get_session_info(self, session_id: str) -> GetSessionInfoReqOutput:
        """Return the configured snapshot.

        :param session_id: Public session identifier.
        :returns: Configured scheduler response.
        """
        self.requested_session_id = session_id
        return self.response


class _UnavailableSessionInfoTokenizerManager:
    """Minimal tokenizer manager that rejects a non-streaming session."""

    async def get_session_info(self, session_id: str) -> GetSessionInfoReqOutput:
        """Reject introspection for an ordinary session.

        :param session_id: Public session identifier.
        :raises StreamingSessionInfoUnavailableError: Always.
        :returns: Never returns.
        """
        raise StreamingSessionInfoUnavailableError(
            f"Session {session_id} is not a streaming session."
        )


class StreamingSessionInfoTransportTest(unittest.IsolatedAsyncioTestCase):
    """Recovery endpoint and correlated IPC behavior."""

    async def test_http_body_contains_only_the_frozen_public_fields(self) -> None:
        """Keep the internal correlation identity off the public response."""
        output = _session_info_output("internal-correlation", tip=128)
        manager = _SessionInfoTokenizerManager(output)
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.session_info("session-a")
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(manager.requested_session_id, "session-a")
        self.assertEqual(
            json.loads(response.body),
            {
                "exists": True,
                "tip": 128,
                "floor": 128,
                "protected": 0,
                "inflight": False,
                "held_tokens": 128,
                "last_rid": "rid-128",
            },
        )

    async def test_non_streaming_session_returns_public_http_400(self) -> None:
        """Reject an ordinary session instead of fabricating zero cursors."""
        manager = _UnavailableSessionInfoTokenizerManager()
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.session_info("ordinary-session")
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            json.loads(response.body),
            {
                "error": {
                    "message": ("Session ordinary-session is not a streaming session.")
                }
            },
        )

    async def test_concurrent_reads_complete_their_exact_waiters(self) -> None:
        """Correlate overlapping reads of the same session independently."""
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.session_info_futures = {}
        manager.auto_create_handle_loop = lambda: None
        dispatched: list[GetSessionInfoReqInput] = []
        manager._dispatch_to_scheduler = dispatched.append

        tasks = [
            asyncio.create_task(manager.get_session_info("shared-session"))
            for _ in range(32)
        ]
        await asyncio.sleep(0)

        self.assertEqual(len(dispatched), len(tasks))
        self.assertEqual(
            len({request.correlation_id for request in dispatched}),
            len(tasks),
        )
        self.assertTrue(
            all(request.session_id == "shared-session" for request in dispatched)
        )

        for tip, request in reversed(list(enumerate(dispatched))):
            manager._handle_get_session_info_req_output(
                _session_info_output(request.correlation_id, tip=tip)
            )

        results = await asyncio.gather(*tasks)
        self.assertEqual([result.tip for result in results], list(range(len(tasks))))
        self.assertEqual(manager.session_info_futures, {})

    async def test_cancelled_read_cleans_up_and_ignores_late_output(self) -> None:
        """Remove cancelled waiters without reviving them on a late response."""
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.session_info_futures = {}
        manager.auto_create_handle_loop = lambda: None
        dispatched: list[GetSessionInfoReqInput] = []
        manager._dispatch_to_scheduler = dispatched.append

        task = asyncio.create_task(manager.get_session_info("cancelled-session"))
        await asyncio.sleep(0)
        self.assertEqual(len(dispatched), 1)
        correlation_id = dispatched[0].correlation_id

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(manager.session_info_futures, {})

        manager._handle_get_session_info_req_output(
            _session_info_output(correlation_id, tip=64)
        )
        self.assertEqual(manager.session_info_futures, {})

    async def test_typed_scheduler_error_reconstructs_at_tokenizer(self) -> None:
        """Carry the non-streaming rejection across IPC as a typed response."""
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.session_info_futures = {}
        manager.auto_create_handle_loop = lambda: None
        dispatched: list[GetSessionInfoReqInput] = []
        manager._dispatch_to_scheduler = dispatched.append

        task = asyncio.create_task(manager.get_session_info("ordinary-session"))
        await asyncio.sleep(0)
        self.assertEqual(len(dispatched), 1)
        manager._handle_get_session_info_req_output(
            GetSessionInfoReqErrorOutput(
                correlation_id=dispatched[0].correlation_id,
                message="Session ordinary-session is not a streaming session.",
            )
        )

        with self.assertRaisesRegex(
            StreamingSessionInfoUnavailableError,
            "Session ordinary-session is not a streaming session",
        ):
            await task
        self.assertEqual(manager.session_info_futures, {})

    async def test_session_info_ipc_round_trips_exact_value_domains(self) -> None:
        """Preserve correlation and public value domains over msgpack IPC."""
        request = GetSessionInfoReqInput(
            correlation_id="correlation-a",
            session_id="session-a",
        )
        output = GetSessionInfoReqOutput(
            correlation_id="correlation-a",
            exists=False,
            tip=0,
            floor=0,
            protected=0,
            inflight=False,
            held_tokens=0,
            last_rid=None,
        )
        error = GetSessionInfoReqErrorOutput(
            correlation_id="correlation-a",
            message="not a streaming session",
        )

        decoded_request = msgpack_decode(msgpack_encode(request))
        decoded_output = msgpack_decode(msgpack_encode(output))
        decoded_error = msgpack_decode(msgpack_encode(error))

        self.assertIs(type(decoded_request), GetSessionInfoReqInput)
        self.assertEqual(decoded_request, request)
        self.assertIs(type(decoded_request.correlation_id), str)
        self.assertIs(type(decoded_output), GetSessionInfoReqOutput)
        self.assertEqual(decoded_output, output)
        self.assertIs(type(decoded_output.exists), bool)
        self.assertIs(type(decoded_output.tip), int)
        self.assertIs(type(decoded_output.floor), int)
        self.assertIs(type(decoded_output.protected), int)
        self.assertIs(type(decoded_output.inflight), bool)
        self.assertIs(type(decoded_output.held_tokens), int)
        self.assertIsNone(decoded_output.last_rid)
        self.assertIs(type(decoded_error), GetSessionInfoReqErrorOutput)
        self.assertEqual(decoded_error, error)
        self.assertIs(type(decoded_error.correlation_id), str)
        self.assertIs(type(decoded_error.message), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
