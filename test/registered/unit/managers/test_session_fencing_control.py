import asyncio
import types
import unittest

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    CloseSessionReqOutput,
    InstallSessionFencingReqInput,
    InstallSessionFencingReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.session.errors import StreamingSessionStaleEpochError
from sglang.srt.session.fencing import SessionFencingRegister
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _manager() -> TokenizerManager:
    """Build the session control subset of a tokenizer manager.

    :returns: Manager with deterministic local dispatch surfaces.
    """
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.auto_create_handle_loop = lambda: None
    manager.server_args = types.SimpleNamespace(enable_streaming_session=True)
    manager.session_futures = {}
    manager.close_session_futures = {}
    manager.session_fencing_register = SessionFencingRegister()
    manager.session_fencing_dispatch_lock = asyncio.Lock()
    return manager


class SessionFencingControlTest(unittest.IsolatedAsyncioTestCase):
    """Frontend/scheduler fencing coordination and acknowledgements."""

    async def test_open_waits_for_scheduler_confirmation(self) -> None:
        """Return an opened identity only after its scheduler response."""
        manager = _manager()
        dispatched: list[OpenSessionReqInput] = []
        manager._dispatch_to_scheduler = dispatched.append
        request = OpenSessionReqInput(
            capacity_of_str_len=0,
            session_id="session",
            streaming=True,
            epoch=0,
        )

        task = asyncio.create_task(manager.open_session(request))
        await asyncio.sleep(0)
        self.assertEqual(dispatched, [request])
        self.assertFalse(task.done())
        manager._handle_open_session_req_output(
            OpenSessionReqOutput(session_id="session", success=True)
        )

        self.assertEqual(await task, "session")
        self.assertEqual(manager.session_futures, {})

    async def test_close_waits_for_correlated_scheduler_confirmation(self) -> None:
        """Do not acknowledge close until the owning scheduler has handled it."""
        manager = _manager()
        dispatched: list[CloseSessionReqInput] = []

        async def dispatch(request: CloseSessionReqInput) -> None:
            dispatched.append(request)

        manager._async_dispatch_to_scheduler = dispatch
        task = asyncio.create_task(
            manager.close_session(CloseSessionReqInput(session_id="session", epoch=0))
        )
        await asyncio.sleep(0)
        self.assertEqual(len(dispatched), 1)
        correlation_id = dispatched[0].correlation_id
        assert correlation_id is not None
        self.assertFalse(task.done())
        manager._handle_close_session_req_output(
            CloseSessionReqOutput(
                correlation_id=correlation_id,
                success=True,
            )
        )

        await task
        self.assertEqual(manager.close_session_futures, {})

    async def test_stale_lifecycle_requests_never_dispatch(self) -> None:
        """Reject open and close against the mirrored register under one lock."""
        manager = _manager()
        manager.session_fencing_register.install(5, 9)
        dispatched: list[object] = []
        manager._dispatch_to_scheduler = dispatched.append

        async def dispatch(request: object) -> None:
            dispatched.append(request)

        manager._async_dispatch_to_scheduler = dispatch

        with self.assertRaises(StreamingSessionStaleEpochError):
            await manager.open_session(
                OpenSessionReqInput(
                    capacity_of_str_len=0,
                    session_id="session",
                    streaming=True,
                    epoch=4,
                )
            )
        with self.assertRaises(StreamingSessionStaleEpochError):
            await manager.close_session(
                CloseSessionReqInput(session_id="session", epoch=4)
            )

        self.assertEqual(dispatched, [])

    async def test_install_confirms_all_schedulers_before_mirroring(self) -> None:
        """Mirror only a register value confirmed by every scheduler group."""
        manager = _manager()

        async def install(
            request: InstallSessionFencingReqInput,
        ) -> list[InstallSessionFencingReqOutput]:
            return [
                InstallSessionFencingReqOutput(
                    epoch=request.epoch,
                    cluster_incarnation=request.cluster_incarnation,
                ),
                InstallSessionFencingReqOutput(
                    epoch=request.epoch,
                    cluster_incarnation=request.cluster_incarnation,
                ),
            ]

        manager.install_session_fencing_communicator = install

        state = await manager.install_session_fencing_register(
            InstallSessionFencingReqInput(epoch=7, cluster_incarnation=11)
        )

        self.assertEqual(state.epoch, 7)
        self.assertEqual(state.cluster_incarnation, 11)
        self.assertEqual(manager.session_fencing_register.state, state)

    async def test_divergent_scheduler_installation_is_not_mirrored(self) -> None:
        """Fail closed if scheduler groups report different register values."""
        manager = _manager()

        async def install(
            request: InstallSessionFencingReqInput,
        ) -> list[InstallSessionFencingReqOutput]:
            del request
            return [
                InstallSessionFencingReqOutput(epoch=7, cluster_incarnation=11),
                InstallSessionFencingReqOutput(epoch=8, cluster_incarnation=11),
            ]

        manager.install_session_fencing_communicator = install

        with self.assertRaisesRegex(RuntimeError, "divergent"):
            await manager.install_session_fencing_register(
                InstallSessionFencingReqInput(epoch=7, cluster_incarnation=11)
            )

        self.assertEqual(manager.session_fencing_register.state.epoch, 0)


if __name__ == "__main__":
    unittest.main()
