import asyncio
import json
import types
import unittest

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import (
    DemoteSessionReqInput,
    DemoteSessionReqOutput,
    GetSessionInfoReqErrorOutput,
    GetSessionInfoReqInput,
    GetSessionInfoReqOutput,
    ListSessionsReqInput,
    ListSessionsReqOutput,
    SessionInventoryOutput,
    SessionKVResidencyOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.session.errors import StreamingSessionInfoUnavailableError
from sglang.srt.session.fencing import (
    CLUSTER_INCARNATION_HEADER,
    FENCING_EPOCH_HEADER,
    SessionFencingRegister,
)
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
        lineage_digest=f"sha256:v1:digest-{tip}",
        floor=tip,
        protected=0,
        inflight=False,
        held_tokens=tip,
        last_rid=f"rid-{tip}",
    )


def _list_sessions_output(correlation_id: str) -> ListSessionsReqOutput:
    """Build a deterministic inventory response.

    :param correlation_id: Internal control-plane request identity.
    :returns: Complete session-inventory IPC response.
    """
    return ListSessionsReqOutput(
        correlation_id=correlation_id,
        sessions=[
            SessionInventoryOutput(
                session_id="session-a",
                lineage_generation=3,
                tip=128,
                lineage_digest="sha256:v1:digest-128",
                floor=64,
                full=SessionKVResidencyOutput(
                    device_pages=8,
                    host_backed_pages=4,
                ),
                swa=SessionKVResidencyOutput(
                    device_pages=2,
                    host_backed_pages=1,
                ),
            )
        ],
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
        self.session_fencing_register = SessionFencingRegister()

    async def get_session_info(self, session_id: str) -> GetSessionInfoReqOutput:
        """Return the configured snapshot.

        :param session_id: Public session identifier.
        :returns: Configured scheduler response.
        """
        self.requested_session_id = session_id
        return self.response


class _UnavailableSessionInfoTokenizerManager:
    """Minimal tokenizer manager that rejects a non-streaming session."""

    session_fencing_register = SessionFencingRegister()

    async def get_session_info(self, session_id: str) -> GetSessionInfoReqOutput:
        """Reject introspection for an ordinary session.

        :param session_id: Public session identifier.
        :raises StreamingSessionInfoUnavailableError: Always.
        :returns: Never returns.
        """
        raise StreamingSessionInfoUnavailableError(
            f"Session {session_id} is not a streaming session."
        )


class _InventoryTokenizerManager:
    """Minimal tokenizer manager for the inventory HTTP endpoint."""

    engine_incarnation_id: str = "engine-incarnation-a"
    session_fencing_register = SessionFencingRegister()

    async def list_sessions(self) -> ListSessionsReqOutput:
        """Return one deterministic inventory entry.

        :returns: Configured session inventory.
        """
        return _list_sessions_output("internal-correlation")


class _DemotionTokenizerManager:
    """Minimal tokenizer manager for the demotion HTTP endpoint."""

    response: DemoteSessionReqOutput
    requested: DemoteSessionReqInput | None

    def __init__(self, response: DemoteSessionReqOutput) -> None:
        """Initialize the deterministic manager.

        :param response: Transaction result returned by every request.
        """
        self.response = response
        self.requested = None
        self.session_fencing_register = SessionFencingRegister()

    async def demote_session(
        self,
        obj: DemoteSessionReqInput,
        request: object,
    ) -> DemoteSessionReqOutput:
        """Return the configured transaction result.

        :param obj: Typed public request.
        :param request: Originating HTTP request.
        :returns: Configured scheduler response.
        """
        self.requested = obj
        return self.response


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
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "0")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "0")
        self.assertEqual(manager.requested_session_id, "session-a")
        self.assertEqual(
            json.loads(response.body),
            {
                "exists": True,
                "tip": 128,
                "lineage_digest": "sha256:v1:digest-128",
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
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "0")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "0")
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
            lineage_digest=None,
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
        self.assertIsNone(decoded_output.lineage_digest)
        self.assertIs(type(decoded_output.floor), int)
        self.assertIs(type(decoded_output.protected), int)
        self.assertIs(type(decoded_output.inflight), bool)
        self.assertIs(type(decoded_output.held_tokens), int)
        self.assertIsNone(decoded_output.last_rid)
        self.assertIs(type(decoded_error), GetSessionInfoReqErrorOutput)
        self.assertEqual(decoded_error, error)
        self.assertIs(type(decoded_error.correlation_id), str)
        self.assertIs(type(decoded_error.message), str)


class StreamingSessionDemotionTransportTest(unittest.IsolatedAsyncioTestCase):
    """Typed correlated transport for the administrative demotion operation."""

    async def test_http_success_exposes_lineage_and_forwards_epoch(self) -> None:
        output = DemoteSessionReqOutput(
            correlation_id="internal-correlation",
            session_id="session-a",
            success=True,
            tip=129,
            lineage_digest="sha256:v1:digest-129",
            lineage_generation=3,
            host_backed_tokens=129,
        )
        manager = _DemotionTokenizerManager(output)
        manager.session_fencing_register.install(7, 19)
        request = types.SimpleNamespace(headers={FENCING_EPOCH_HEADER: "7"})
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.demote_session(
                DemoteSessionReqInput(session_id="session-a"),
                request,
            )
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "7")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "19")
        self.assertIsNotNone(manager.requested)
        self.assertEqual(manager.requested.epoch, 7)
        self.assertEqual(
            json.loads(response.body),
            {
                "session_id": "session-a",
                "success": True,
                "tip": 129,
                "lineage_digest": "sha256:v1:digest-129",
                "lineage_generation": 3,
                "host_backed_tokens": 129,
                "error_type": None,
                "message": None,
            },
        )

    async def test_http_rejection_is_a_typed_conflict(self) -> None:
        output = DemoteSessionReqOutput(
            correlation_id="internal-correlation",
            session_id="session-a",
            success=False,
            tip=129,
            lineage_digest="sha256:v1:digest-129",
            lineage_generation=3,
            error_type="StreamingSessionDemotionError",
            message="host tier full",
        )
        manager = _DemotionTokenizerManager(output)
        request = types.SimpleNamespace(headers={})
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.demote_session(
                DemoteSessionReqInput(session_id="session-a"),
                request,
            )
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            json.loads(response.body),
            {
                "session_id": "session-a",
                "success": False,
                "tip": 129,
                "lineage_digest": "sha256:v1:digest-129",
                "lineage_generation": 3,
                "host_backed_tokens": 0,
                "error_type": "StreamingSessionDemotionError",
                "message": "host tier full",
            },
        )

    async def test_http_invalid_epoch_is_rejected_before_dispatch(self) -> None:
        output = DemoteSessionReqOutput(
            correlation_id="internal-correlation",
            session_id="session-a",
            success=True,
            tip=129,
            lineage_digest="sha256:v1:digest-129",
            lineage_generation=3,
        )
        manager = _DemotionTokenizerManager(output)
        request = types.SimpleNamespace(headers={FENCING_EPOCH_HEADER: "invalid"})
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.demote_session(
                DemoteSessionReqInput(session_id="session-a"),
                request,
            )
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(manager.requested)

    async def test_waiter_completes_with_exact_scheduler_result(self) -> None:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.demote_session_futures = {}
        manager.session_fencing_register = SessionFencingRegister()
        manager.session_fencing_dispatch_lock = asyncio.Lock()
        manager.auto_create_handle_loop = lambda: None
        dispatched: list[DemoteSessionReqInput] = []

        async def dispatch(request: DemoteSessionReqInput) -> None:
            dispatched.append(request)

        manager._async_dispatch_to_scheduler = dispatch
        task = asyncio.create_task(
            manager.demote_session(DemoteSessionReqInput(session_id="session-a"))
        )
        await asyncio.sleep(0)
        self.assertEqual(len(dispatched), 1)
        correlation_id = dispatched[0].correlation_id
        self.assertIsNotNone(correlation_id)

        manager._handle_demote_session_req_output(
            DemoteSessionReqOutput(
                correlation_id=correlation_id,
                session_id="session-a",
                success=True,
                tip=128,
                lineage_digest="sha256:v1:digest-128",
                lineage_generation=3,
                host_backed_tokens=128,
            )
        )

        result = await task
        self.assertTrue(result.success)
        self.assertEqual(result.tip, 128)
        self.assertEqual(result.lineage_digest, "sha256:v1:digest-128")
        self.assertEqual(result.lineage_generation, 3)
        self.assertEqual(result.host_backed_tokens, 128)
        self.assertEqual(manager.demote_session_futures, {})

    async def test_ipc_round_trip_preserves_typed_error_fields(self) -> None:
        request = DemoteSessionReqInput(
            session_id="session-a",
            epoch=7,
            correlation_id="correlation-a",
        )
        output = DemoteSessionReqOutput(
            correlation_id="correlation-a",
            session_id="session-a",
            success=False,
            tip=128,
            lineage_digest="sha256:v1:digest-128",
            lineage_generation=3,
            error_type="StreamingSessionDemotionError",
            message="host tier full",
            request_epoch=7,
            registered_epoch=7,
            cluster_incarnation=19,
        )

        decoded_request = msgpack_decode(msgpack_encode(request))
        decoded_output = msgpack_decode(msgpack_encode(output))

        self.assertIs(type(decoded_request), DemoteSessionReqInput)
        self.assertEqual(decoded_request, request)
        self.assertIs(type(decoded_output), DemoteSessionReqOutput)
        self.assertEqual(decoded_output, output)
        self.assertIs(type(decoded_output.success), bool)
        self.assertIs(type(decoded_output.tip), int)
        self.assertIs(type(decoded_output.lineage_digest), str)
        self.assertIs(type(decoded_output.error_type), str)


class StreamingSessionInventoryTransportTest(unittest.IsolatedAsyncioTestCase):
    """Inventory endpoint and correlated IPC behavior."""

    async def test_http_body_contains_engine_and_component_residency(self) -> None:
        manager = _InventoryTokenizerManager()
        prior_state = http_server.get_global_state()
        http_server.set_global_state(types.SimpleNamespace(tokenizer_manager=manager))
        try:
            response = await http_server.list_sessions()
        finally:
            http_server.set_global_state(prior_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers[FENCING_EPOCH_HEADER], "0")
        self.assertEqual(response.headers[CLUSTER_INCARNATION_HEADER], "0")
        self.assertEqual(
            json.loads(response.body),
            {
                "engine_incarnation_id": "engine-incarnation-a",
                "sessions": [
                    {
                        "session_id": "session-a",
                        "lineage_generation": 3,
                        "tip": 128,
                        "lineage_digest": "sha256:v1:digest-128",
                        "floor": 64,
                        "kv_residency": {
                            "full": {
                                "device_pages": 8,
                                "host_backed_pages": 4,
                            },
                            "swa": {
                                "device_pages": 2,
                                "host_backed_pages": 1,
                            },
                        },
                    }
                ],
            },
        )

    async def test_concurrent_reads_complete_their_exact_waiters(self) -> None:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.list_sessions_futures = {}
        manager.auto_create_handle_loop = lambda: None
        dispatched: list[ListSessionsReqInput] = []
        manager._dispatch_to_scheduler = dispatched.append

        tasks = [asyncio.create_task(manager.list_sessions()) for _ in range(16)]
        await asyncio.sleep(0)

        self.assertEqual(len(dispatched), len(tasks))
        self.assertEqual(
            len({request.correlation_id for request in dispatched}),
            len(tasks),
        )
        for request in reversed(dispatched):
            manager._handle_list_sessions_req_output(
                _list_sessions_output(request.correlation_id)
            )

        results = await asyncio.gather(*tasks)
        self.assertTrue(
            all(result.sessions[0].session_id == "session-a" for result in results)
        )
        self.assertEqual(manager.list_sessions_futures, {})

    async def test_inventory_ipc_round_trips_nested_value_domains(self) -> None:
        request = ListSessionsReqInput(correlation_id="correlation-a")
        output = _list_sessions_output("correlation-a")

        decoded_request = msgpack_decode(msgpack_encode(request))
        decoded_output = msgpack_decode(msgpack_encode(output))

        self.assertIs(type(decoded_request), ListSessionsReqInput)
        self.assertEqual(decoded_request, request)
        self.assertIs(type(decoded_output), ListSessionsReqOutput)
        self.assertEqual(decoded_output, output)
        [session] = decoded_output.sessions
        self.assertIs(type(session), SessionInventoryOutput)
        self.assertIs(type(session.full), SessionKVResidencyOutput)
        self.assertIs(type(session.swa), SessionKVResidencyOutput)


if __name__ == "__main__":
    unittest.main(verbosity=2)
