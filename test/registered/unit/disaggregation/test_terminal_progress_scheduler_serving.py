import ast
import concurrent.futures
import dataclasses
import errno
import inspect
import os
import select
import signal
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sglang.srt.disaggregation.terminal_progress.scheduler_inbox as scheduler_inbox_module
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.nixl.conn import NixlTerminalRuntimeInstallation
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeServingComposition,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    SchedulerInboxFatalCause,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalProcessIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxFatalError,
    SchedulerReceiptInboxInventory,
    SchedulerReceiptPublishResult,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalDecodeSchedulerConsumer,
    TerminalSchedulerActionPublicationDisposition,
    TerminalSchedulerActionPublicationError,
    TerminalSchedulerServing,
    TerminalSchedulerServingRole,
    TerminalSourceSchedulerConsumer,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceServing,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.distributed.scheduler_output_identity import SchedulerOutputIdentity
from sglang.srt.managers.scheduler import (
    Scheduler,
    build_scheduler_parallel_state,
    run_scheduler_process,
)
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scheduler_parallel_state(tp_rank: int, tp_size: int) -> ParallelState:
    """Build scheduler ranks through the production launch assembly path.

    :param tp_rank: Tensor-parallel process rank.
    :param tp_size: Tensor-parallel width.
    :returns: Exact process-local scheduler rank state.
    """

    return build_scheduler_parallel_state(
        ServerArgs(model_path="dummy", tp_size=tp_size),
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
    )


def _identity(marker: int, role: TerminalOwnerRole) -> TerminalProcessIdentity:
    """Build one deterministic process identity.

    :param marker: Byte marker for the process generation.
    :param role: Source or decode owner role.
    :returns: Exact process identity.
    """

    return TerminalProcessIdentity(
        process_generation=bytes([marker]) * 16,
        role=role,
        tp_rank=0,
        tp_size=1,
    )


def _binding(
    room_id: int,
    marker: int,
    role: TerminalOwnerRole,
) -> TerminalRequestBinding:
    """Build one deterministic request binding.

    :param room_id: Stable packed room identity.
    :param marker: Byte marker for generation and allocation identities.
    :param role: Scheduler role owning the binding.
    :returns: Exact request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=bytes([marker]) * 16,
        ),
        owner=_identity(marker, role),
        rank_manifest_digest=bytes([marker]) * 32,
        allocation_digest=bytes([marker + 1]) * 32,
    )


def _action(
    binding: TerminalRequestBinding,
    action_id: int,
) -> NativeTerminalOwnerAction:
    """Build one owner-minted scheduler action.

    :param binding: Exact source or decode request binding.
    :param action_id: Gap-free native action identity.
    :returns: Role-appropriate reclaim or adoption action.
    """

    native_binding = NativeTerminalRequestBinding.from_binding(binding)
    action_kind = NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED
    receipt_kind = NativeTerminalReceiptKind.RECLAIM_AUTHORIZED
    if binding.owner.role is TerminalOwnerRole.DECODE:
        action_kind = NativeTerminalOwnerActionKind.ADOPTION_READY
        receipt_kind = NativeTerminalReceiptKind.ADOPTION_READY
    receipt = NativeTerminalReceipt(
        binding=native_binding,
        issuer=NativeTerminalProcessIdentity.from_identity(binding.owner),
        kind=receipt_kind,
        outcome=NativeTerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=action_id,
        nonce=action_id.to_bytes(16, "big"),
    )
    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=action_kind,
        binding=native_binding,
        commit_timestamp_ns=action_id,
        receipt=receipt,
    )


class _SourceConsumer(TerminalSourceSchedulerConsumer):
    """Record exact source action and fatal delivery in scheduler context."""

    actions: list[NativeTerminalOwnerAction]
    fatal_inventories: list[SchedulerReceiptInboxInventory]

    def __init__(self) -> None:
        """Create one empty scheduler-side observation sink."""

        self.actions = []
        self.fatal_inventories = []

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Record one exact reclaim action.

        :param action: Exact owner action.
        """

        self.actions.append(action)

    def process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Record one scheduler-affine fatal transition.

        :param inventory: Complete retained inbox evidence.
        """

        self.fatal_inventories.append(inventory)


class _DecodeConsumer(TerminalDecodeSchedulerConsumer):
    """Record exact decode action and fatal delivery in scheduler context."""

    actions: list[NativeTerminalOwnerAction]
    fatal_inventories: list[SchedulerReceiptInboxInventory]

    def __init__(self) -> None:
        """Create one empty scheduler-side observation sink."""

        self.actions = []
        self.fatal_inventories = []

    def consume_adoption_ready(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Record one exact adoption action.

        :param action: Exact owner action.
        """

        self.actions.append(action)

    def process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Record one scheduler-affine fatal transition.

        :param inventory: Complete retained inbox evidence.
        """

        self.fatal_inventories.append(inventory)


class _FailingSourceConsumer(_SourceConsumer):
    """Raise after recording fail-closed scheduler delivery."""

    def process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Record fatal delivery and simulate teardown callback failure.

        :param inventory: Complete retained inbox evidence.
        """

        super().process_fatal(inventory)
        raise RuntimeError("synthetic fail-closed consumer failure")


class _RuntimeFence:
    """Record one deterministic native output-projection fence."""

    quiescent: bool
    fence_error: RuntimeError | None
    calls: list[tuple[str, float | None]]

    def __init__(
        self,
        *,
        quiescent: bool,
        fence_error: RuntimeError | None = None,
    ) -> None:
        """Create one controlled runtime-fence result.

        :param quiescent: Successful fence result when no exception is raised.
        :param fence_error: Optional synthetic projection-fence exception.
        """

        self.quiescent = quiescent
        self.fence_error = fence_error
        self.calls = []

    def wait_for_output_projection_quiescence(
        self,
        timeout_seconds: float,
    ) -> bool:
        """Return or raise the configured fence outcome.

        :param timeout_seconds: Exact hash-bound timeout supplied by serving.
        :returns: Configured quiescence result.
        """

        self.calls.append(("fence", timeout_seconds))
        if self.fence_error is not None:
            raise self.fence_error
        return self.quiescent

    def begin_abort(self) -> None:
        """Record fail-closed entry without closing scheduler consumers."""

        self.calls.append(("abort", None))


class _ManagerOwnedClose:
    """Close one scheduler serving object through the manager boundary."""

    serving: TerminalSchedulerServing
    order: list[str]
    sleeper: IdleSleeper | None
    process_fatal_values: list[bool]

    def __init__(
        self,
        serving: TerminalSchedulerServing,
        order: list[str],
        sleeper: IdleSleeper | None = None,
    ) -> None:
        """Create one deterministic terminal manager close.

        :param serving: Scheduler serving object owned by the manager runtime.
        :param order: Shared teardown-order receipt.
        :param sleeper: Optional scheduler poll owner used to verify detachment.
        """

        self.serving = serving
        self.order = order
        self.sleeper = sleeper
        self.process_fatal_values = []

    def close_terminal_runtime(self, *, process_fatal: bool) -> None:
        """Record and perform the sole serving close.

        :param process_fatal: Whether close must retain ambiguous authority.
        """

        if self.sleeper is not None:
            assert self.serving.fileno() not in self.sleeper._file_descriptors
        self.order.append("manager")
        self.process_fatal_values.append(process_fatal)
        if process_fatal:
            self.serving.close_fail_closed()
            return
        self.serving.close()


class _MetricsTeardown:
    """Record scheduler metrics teardown ordering."""

    order: list[str]

    def __init__(self, order: list[str]) -> None:
        """Retain the shared order receipt.

        :param order: Shared teardown-order receipt.
        """

        self.order = order

    def _shutdown_fpm(self) -> None:
        """Record metrics teardown."""

        self.order.append("metrics")


class _FailingManagerClose:
    """Expose one bounded manager close failure."""

    order: list[str]
    process_fatal_values: list[bool]

    def __init__(self, order: list[str]) -> None:
        """Retain the shared order receipt.

        :param order: Shared teardown-order receipt.
        """

        self.order = order
        self.process_fatal_values = []

    def close_terminal_runtime(self, *, process_fatal: bool) -> None:
        """Raise after recording the requested disposition.

        :param process_fatal: Requested manager close disposition.
        """

        self.order.append("manager")
        self.process_fatal_values.append(process_fatal)
        raise RuntimeError("synthetic terminal manager close failure")


class _SchedulerProcessHarness:
    """Expose controlled event-loop and release outcomes to the process entrypoint."""

    behavior: str
    gracefully_exit: bool
    release_values: list[bool]

    def __init__(self, behavior: str) -> None:
        """Create one scheduler process outcome.

        :param behavior: ``graceful``, ``unexpected``, or ``exception``.
        """

        self.behavior = behavior
        self.gracefully_exit = behavior == "graceful"
        self.release_values = []

    def get_init_info(self) -> dict[str, str]:
        """Return the minimal parent handshake.

        :returns: Stable ready receipt.
        """

        return {"status": "ready"}

    def run_event_loop(self) -> None:
        """Return or raise according to the configured process outcome."""

        if self.behavior == "exception":
            raise RuntimeError("synthetic scheduler event-loop failure")

    def release(self, *, process_fatal: bool) -> None:
        """Record the teardown disposition.

        :param process_fatal: Process failure disposition selected by the entrypoint.
        """

        self.release_values.append(process_fatal)


class _SchedulerParentProcess:
    """Record scheduler failure notifications."""

    signals: list[int]

    def __init__(self) -> None:
        """Create one empty signal receipt."""

        self.signals = []

    def send_signal(self, signal_number: int) -> None:
        """Record a parent-process signal.

        :param signal_number: Operating-system signal number.
        """

        self.signals.append(signal_number)


class _SchedulerPipeWriter:
    """Record scheduler initialization handshakes."""

    payloads: list[dict[str, str]]

    def __init__(self) -> None:
        """Create one empty handshake receipt."""

        self.payloads = []

    def send(self, payload: dict[str, str]) -> None:
        """Record a scheduler handshake.

        :param payload: Initialization payload sent to the parent.
        """

        self.payloads.append(payload)


def _source_serving(
    consumer: _SourceConsumer,
    capacity: int = 2,
) -> TerminalSchedulerServing:
    """Build one source serving adapter.

    :param consumer: Exact scheduler-affine source consumer.
    :param capacity: Maximum live request generations.
    :returns: Qualified source adapter.
    """

    return TerminalSchedulerServing(
        role=TerminalSchedulerServingRole.SOURCE,
        physical_capacity=capacity,
        source_consumer=consumer,
    )


def _decode_serving(
    consumer: _DecodeConsumer,
    capacity: int = 2,
) -> TerminalSchedulerServing:
    """Build one decode serving adapter.

    :param consumer: Exact scheduler-affine decode consumer.
    :param capacity: Maximum live request generations.
    :returns: Qualified decode adapter.
    """

    return TerminalSchedulerServing(
        role=TerminalSchedulerServingRole.DECODE,
        physical_capacity=capacity,
        decode_consumer=consumer,
    )


def test_loop_entry_consumes_exact_source_action_and_retires_generation() -> None:
    """Source reclaim crosses the native-to-scheduler boundary exactly once."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer)
    binding = _binding(1, 1, TerminalOwnerRole.SOURCE)
    action = _action(binding, 11)
    serving.register_request(binding)

    assert serving.publish_action(action) is SchedulerReceiptPublishResult.QUEUED
    assert serving.drain_at_loop_entry() == (action,)
    assert consumer.actions == [action]
    inventory = serving.inventory()
    assert inventory.inbox.live_count == 0
    assert inventory.inbox.pending_count == 0
    assert inventory.retained_action_ids == ()
    serving.close()


def test_native_runtime_binding_registers_without_reconstructing_identity() -> None:
    """The runtime's exact native binding is the scheduler admission source."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer)
    binding = _binding(3, 3, TerminalOwnerRole.DECODE)

    serving.register_native_request(NativeTerminalRequestBinding.from_binding(binding))

    assert serving.inventory().inbox.live_bindings == (binding,)
    serving.cancel_unpublished_request(binding)
    serving.close()


def test_decode_launch_handoff_consumes_while_forward_barrier_remains_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An action racing host submission cannot inherit forward completion."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer)
    binding = _binding(2, 2, TerminalOwnerRole.DECODE)
    action = _action(binding, 12)
    serving.register_request(binding)
    begin_publication = threading.Event()
    publication_announced = threading.Event()
    forward_completion = threading.Event()
    original_begin = serving._inbox._begin_publication_intent

    def observed_begin() -> int:
        """Expose the publication linearization point."""

        intent = original_begin()
        publication_announced.set()
        return intent

    monkeypatch.setattr(serving._inbox, "_begin_publication_intent", observed_begin)

    def publisher() -> SchedulerReceiptPublishResult:
        """Publish after the synthetic host submission begins."""

        assert begin_publication.wait(timeout=5)
        return serving.publish_action(action)

    def submit() -> str:
        """Return after host submission without releasing the forward barrier."""

        begin_publication.set()
        assert publication_announced.wait(timeout=5)
        assert not forward_completion.is_set()
        return "submitted"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(publisher)
        assert serving.launch_handoff(submit) == "submitted"
        assert publication.result(timeout=5) is SchedulerReceiptPublishResult.QUEUED

    assert consumer.actions == [action]
    assert not forward_completion.is_set()
    serving.close()


def test_source_launch_registers_before_racing_publication_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding may register in its inbox before a racing action is consumed."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer)
    binding = _binding(20, 20, TerminalOwnerRole.SOURCE)
    action = _action(binding, 20)
    begin_publication = threading.Event()
    publication_announced = threading.Event()
    original_begin = serving._inbox._begin_publication_intent
    ordering: list[str] = []

    def observed_begin() -> int:
        """Expose the publication linearization point."""

        intent = original_begin()
        publication_announced.set()
        return intent

    monkeypatch.setattr(serving._inbox, "_begin_publication_intent", observed_begin)

    def consume(consumed: NativeTerminalOwnerAction) -> None:
        """Record scheduler consumption after immutable binding."""

        assert consumed == action
        ordering.append("consume")
        consumer.actions.append(consumed)

    monkeypatch.setattr(consumer, "consume_reclaim_authorized", consume)

    def publisher() -> SchedulerReceiptPublishResult:
        """Publish as soon as binding makes the request owner-visible."""

        assert begin_publication.wait(timeout=5)
        return serving.publish_action(action)

    def submit() -> str:
        """Return one synthetic model-worker result."""

        ordering.append("submit")
        return "forward-result"

    def bind(result: str) -> str:
        """Register the exact generation before owner publication."""

        assert result == "forward-result"
        serving.register_request(binding)
        ordering.append("bind")
        begin_publication.set()
        assert publication_announced.wait(timeout=5)
        return "bound-submission"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(publisher)
        assert serving.launch_and_bind_handoff(submit, bind) == "bound-submission"
        assert publication.result(timeout=5) is SchedulerReceiptPublishResult.QUEUED

    ordering.append("return")
    assert ordering == ["submit", "bind", "consume", "return"]
    assert consumer.actions == [action]
    serving.close()


def test_inbox_population_never_exceeds_exact_inflight_generations() -> None:
    """Receipt and retained-action populations stay bounded by live requests."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=8)
    bindings = tuple(
        _binding(room_id, room_id, TerminalOwnerRole.SOURCE)
        for room_id in range(10, 18)
    )
    for binding in bindings:
        serving.register_request(binding)
    for action_id, binding in enumerate(bindings, start=20):
        serving.publish_action(_action(binding, action_id))
        inventory = serving.inventory()
        assert inventory.inbox.pending_count <= inventory.inbox.live_count
        assert len(inventory.retained_action_ids) <= inventory.inbox.live_count

    serving.drain_at_loop_entry()
    assert len(consumer.actions) == len(bindings)
    serving.close()


def test_unregistered_action_enters_shared_fatal_path_and_wakes_scheduler() -> None:
    """Owner output cannot target a generation absent from scheduler admission."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    action = _action(_binding(30, 30, TerminalOwnerRole.SOURCE), 30)

    with pytest.raises(TerminalSchedulerActionPublicationError) as raised:
        serving.publish_action(action)

    assert raised.value.action is action
    assert (
        raised.value.disposition
        is TerminalSchedulerActionPublicationDisposition.CALLER_RETAINS
    )
    assert not raised.value.scheduler_retains_action
    readable, _, _ = select.select([serving.fileno()], [], [], 0)
    assert readable == [serving.fileno()]
    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.drain_at_loop_entry()
    assert len(consumer.fatal_inventories) == 1
    assert serving.inventory().retained_action_ids == ()
    serving.close_fail_closed()


def test_wake_failure_surfaces_exact_scheduler_retained_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-insertion wake failure cannot return native action ownership."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    binding = _binding(32, 32, TerminalOwnerRole.SOURCE)
    action = _action(binding, 33)
    serving.register_request(binding)

    def fail_wake(_file_descriptor: int, _payload: bytes) -> int:
        """Fail the fd write after inbox insertion linearizes."""

        raise OSError(errno.EBADF, "synthetic wake failure")

    monkeypatch.setattr(scheduler_inbox_module.os, "write", fail_wake)

    with pytest.raises(TerminalSchedulerActionPublicationError) as raised:
        serving.publish_action(action)

    failure = raised.value
    assert failure.action is action
    assert (
        failure.disposition
        is TerminalSchedulerActionPublicationDisposition.SCHEDULER_RETAINS
    )
    assert failure.scheduler_retains_action
    assert failure.serving_inventory.inbox.pending_request_keys == (
        binding.request_key,
    )
    assert failure.serving_inventory.retained_action_ids == (action.action_id,)
    closure = serving.close_fail_closed()
    assert closure.retained_actions == (action,)


def test_conflicting_action_retains_only_canonical_native_authority() -> None:
    """A second action for one generation cannot inflate retained inventory."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    binding = _binding(31, 31, TerminalOwnerRole.SOURCE)
    canonical = _action(binding, 31)
    conflict = _action(binding, 32)
    serving.register_request(binding)
    serving.publish_action(canonical)

    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.publish_action(conflict)

    inventory = serving.inventory()
    assert inventory.inbox.pending_request_keys == (binding.request_key,)
    assert inventory.retained_action_ids == (canonical.action_id,)
    serving.close_fail_closed()


def test_owner_death_is_delivered_once_on_scheduler_thread() -> None:
    """Native owner death actively wakes one exact fail-closed transition."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer, capacity=1)
    serving.mark_owner_dead()

    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.drain_at_loop_entry()
    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.drain_at_loop_entry()

    assert len(consumer.fatal_inventories) == 1
    serving.close_fail_closed()


def test_runtime_teardown_fence_succeeds_before_consumer_closure() -> None:
    """A quiescent projection leaves the serving consumer available to drain."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer, capacity=1)
    runtime = _RuntimeFence(quiescent=True)

    serving.fence_runtime_teardown(runtime, 60.0)

    assert runtime.calls == [("fence", 60.0)]
    assert not serving.inventory().inbox.closed
    serving.close()


@pytest.mark.parametrize("raises", (False, True))
def test_runtime_teardown_fence_failure_begins_abort_without_closing_consumer(
    raises: bool,
) -> None:
    """Timeout and exception both enter abort with the wake fd still alive."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    fence_error = RuntimeError("synthetic projection failure") if raises else None
    runtime = _RuntimeFence(quiescent=False, fence_error=fence_error)
    file_descriptor = serving.fileno()

    with pytest.raises(RuntimeError):
        serving.fence_runtime_teardown(runtime, 60.0)

    assert runtime.calls == [("fence", 60.0), ("abort", None)]
    assert not serving.inventory().inbox.closed
    os.fstat(file_descriptor)
    serving.close()


def test_fail_closed_closure_transfers_exact_authority_once() -> None:
    """Fatal closure atomically transfers retained actions and closes its fd."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=2)
    bindings = (
        _binding(40, 40, TerminalOwnerRole.SOURCE),
        _binding(41, 41, TerminalOwnerRole.SOURCE),
    )
    later_action = _action(bindings[0], 42)
    earlier_action = _action(bindings[1], 40)
    for binding in bindings:
        serving.register_request(binding)
    serving.publish_action(later_action)
    serving.publish_action(earlier_action)
    file_descriptor = serving.fileno()
    serving.mark_owner_dead()

    closure = serving.close_fail_closed()
    inventory = closure.inventory

    assert inventory.inbox.closed
    assert set(inventory.inbox.live_bindings) == set(bindings)
    assert set(inventory.inbox.pending_request_keys) == {
        binding.request_key for binding in bindings
    }
    assert closure.retained_actions == (earlier_action, later_action)
    assert inventory.retained_action_ids == tuple(
        action.action_id for action in closure.retained_actions
    )
    assert inventory.fatal_delivered
    assert len(consumer.fatal_inventories) == 1
    assert serving.inventory().retained_action_ids == ()
    with pytest.raises(
        RuntimeError,
        match="scheduler fail-closed closure was already taken",
    ):
        serving.close_fail_closed()
    with pytest.raises(OSError) as raised:
        os.fstat(file_descriptor)
    assert raised.value.errno == errno.EBADF


def test_fail_closed_descriptor_closure_survives_consumer_failure() -> None:
    """A broken fatal callback cannot leak the process-lifetime wake channel."""

    consumer = _FailingSourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    file_descriptor = serving.fileno()

    with pytest.raises(
        RuntimeError,
        match="synthetic fail-closed consumer failure",
    ):
        serving.close_fail_closed()

    assert serving.inventory().inbox.closed
    with pytest.raises(OSError) as raised:
        os.fstat(file_descriptor)
    assert raised.value.errno == errno.EBADF


def test_manager_owned_fatal_teardown_detaches_scheduler_poll_target() -> None:
    """Scheduler detaches its poll target before manager-owned fatal closure."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer, capacity=1)
    sleeper = IdleSleeper(sockets=[])
    file_descriptor = serving.fileno()
    sleeper.register_file_descriptor(file_descriptor)
    order: list[str] = []
    manager = _ManagerOwnedClose(serving, order, sleeper)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_source_serving = None
    scheduler.terminal_scheduler_serving = serving
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = sleeper

    def resolve_manager() -> _ManagerOwnedClose:
        """Return the manager boundary used by this scheduler."""

        return manager

    scheduler.terminal_nixl_manager = resolve_manager
    Scheduler.close_terminal_runtime(scheduler, process_fatal=True)

    assert order == ["manager"]
    assert manager.process_fatal_values == [True]
    assert scheduler.terminal_scheduler_serving is None
    assert file_descriptor not in sleeper._file_descriptors
    assert len(consumer.fatal_inventories) == 1
    with pytest.raises(OSError) as raised:
        os.fstat(file_descriptor)
    assert raised.value.errno == errno.EBADF


def test_scheduler_release_orders_manager_metrics_and_host_resources() -> None:
    """Graceful host destruction follows the manager-owned serving close."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    order: list[str] = []
    manager = _ManagerOwnedClose(serving, order)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_source_serving = None
    scheduler.terminal_scheduler_serving = serving
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = None
    scheduler.metrics_reporter = _MetricsTeardown(order)

    def resolve_manager() -> _ManagerOwnedClose:
        """Return the manager boundary used by this scheduler."""

        return manager

    def release_host_resources() -> None:
        """Record host-resource destruction."""

        order.append("host")

    scheduler.terminal_nixl_manager = resolve_manager
    scheduler.release_host_resources = release_host_resources
    Scheduler.release(scheduler, process_fatal=False)

    assert order == ["manager", "metrics", "host"]
    assert manager.process_fatal_values == [False]
    assert scheduler.terminal_scheduler_serving is None
    assert serving.inventory().inbox.closed


def test_fatal_scheduler_release_retains_host_resources() -> None:
    """Fatal teardown closes through the manager without touching host pools."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    binding = _binding(41, 41, TerminalOwnerRole.SOURCE)
    serving.register_request(binding)
    order: list[str] = []
    manager = _ManagerOwnedClose(serving, order)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_source_serving = None
    scheduler.terminal_scheduler_serving = serving
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = None
    scheduler.metrics_reporter = _MetricsTeardown(order)

    def resolve_manager() -> _ManagerOwnedClose:
        """Return the manager boundary used by this scheduler."""

        return manager

    def release_host_resources() -> None:
        """Reject host-resource destruction on the fatal path."""

        raise AssertionError("fatal scheduler release destroyed host resources")

    scheduler.terminal_nixl_manager = resolve_manager
    scheduler.release_host_resources = release_host_resources
    Scheduler.release(scheduler, process_fatal=True)

    assert order == ["manager", "metrics"]
    assert manager.process_fatal_values == [True]
    assert len(consumer.fatal_inventories) == 1
    assert consumer.fatal_inventories[0].live_bindings == (binding,)


def test_terminal_close_failure_still_stops_metrics_and_skips_host_release() -> None:
    """A bounded manager failure cannot trigger unsafe host destruction."""

    order: list[str] = []
    manager = _FailingManagerClose(order)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_source_serving = None
    scheduler.terminal_scheduler_serving = None
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = None
    scheduler.metrics_reporter = _MetricsTeardown(order)

    def resolve_manager() -> _FailingManagerClose:
        """Return the failing manager boundary."""

        return manager

    def release_host_resources() -> None:
        """Reject host destruction after terminal close failure."""

        raise AssertionError("failed terminal teardown destroyed host resources")

    scheduler.terminal_nixl_manager = resolve_manager
    scheduler.release_host_resources = release_host_resources

    with pytest.raises(RuntimeError, match="scheduler service teardown failed"):
        Scheduler.release(scheduler, process_fatal=False)

    assert order == ["manager", "metrics"]
    assert scheduler.terminal_scheduler_serving is None


@pytest.mark.parametrize(
    ("skip_tokenizer_init", "expected_endpoint"),
    (
        (True, "ipc://tokenizer"),
        (False, "ipc://detokenizer"),
    ),
)
def test_scheduler_persists_role_correct_gateway_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    skip_tokenizer_init: bool,
    expected_endpoint: str,
) -> None:
    """IPC initialization retains the exact gateway output endpoint."""

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.skip_tokenizer_init = skip_tokenizer_init
    scheduler.server_args = SimpleNamespace(
        skip_tokenizer_init=skip_tokenizer_init,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
    )
    scheduler.ps = _scheduler_parallel_state(tp_rank=1, tp_size=2)
    scheduler.output_identity = SchedulerOutputIdentity.from_parallel_state(
        scheduler.ps
    )
    port_args = SimpleNamespace(
        tokenizer_ipc_name="ipc://tokenizer",
        detokenizer_ipc_name="ipc://detokenizer",
    )
    sentinel_channels = object()

    def create_channels(**kwargs: object) -> object:
        """Return inert noncanonical-rank IPC channels."""

        return sentinel_channels

    monkeypatch.setattr(
        scheduler_module.SchedulerIpcChannels,
        "create",
        staticmethod(create_channels),
    )

    Scheduler.init_ipc_channels(scheduler, port_args)

    assert scheduler.ipc_channels is sentinel_channels
    assert scheduler.terminal_gateway_endpoint == expected_endpoint


@pytest.mark.parametrize(
    ("mode", "role", "tp_rank", "expected_gateway"),
    (
        (
            DisaggregationMode.PREFILL,
            TerminalOwnerRole.SOURCE,
            0,
            "ipc://gateway",
        ),
        (DisaggregationMode.PREFILL, TerminalOwnerRole.SOURCE, 1, None),
        (DisaggregationMode.DECODE, TerminalOwnerRole.DECODE, 0, None),
    ),
)
def test_scheduler_installs_role_runtime_before_activation(
    mode: DisaggregationMode,
    role: TerminalOwnerRole,
    tp_rank: int,
    expected_gateway: str | None,
) -> None:
    """Each terminal rank installs its exact role boundary before activation."""

    order: list[str] = []
    manager = object.__new__(scheduler_module.NixlKVManager)
    manager._terminal_startup_binding = SimpleNamespace(
        advertisement=SimpleNamespace(
            role=role,
            tensor_parallel_rank=tp_rank,
            tensor_parallel_size=2,
        )
    )
    manager.kv_args = SimpleNamespace(terminal_request_capacity=37)
    manager.install_terminal_runtime = MagicMock(
        side_effect=lambda installation: order.append("install")
    )
    manager.activate_terminal_startup = MagicMock(
        side_effect=lambda: order.append("activate")
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.terminal_gateway_endpoint = "ipc://gateway"
    parallel_state = _scheduler_parallel_state(tp_rank=tp_rank, tp_size=2)
    scheduler.output_identity = SchedulerOutputIdentity.from_parallel_state(
        parallel_state
    )

    Scheduler._activate_terminal_kv_manager(scheduler, manager)

    assert order == ["install", "activate"]
    manager.install_terminal_runtime.assert_called_once()
    installation = manager.install_terminal_runtime.call_args.args[0]
    assert type(installation) is NixlTerminalRuntimeInstallation
    assert installation.terminal_request_capacity == 37
    assert installation.gateway_endpoint == expected_gateway
    if mode is DisaggregationMode.PREFILL:
        assert installation.bind_source_serving is not None
        assert installation.bind_source_serving.__self__ is scheduler
        assert installation.bind_decode_serving is None
    else:
        assert installation.bind_source_serving is None
        assert installation.bind_decode_serving is not None
        assert installation.bind_decode_serving.__self__ is scheduler
    assert installation.scheduler_process_fatal_handler.__self__ is scheduler
    assert installation.owner_dead_handler.__self__ is scheduler


def test_scheduler_leaves_nonterminal_manager_on_noop_activation_boundary() -> None:
    """A NIXL manager outside the cohort is activated without installation."""

    manager = object.__new__(scheduler_module.NixlKVManager)
    manager._terminal_startup_binding = None
    manager.install_terminal_runtime = MagicMock()
    manager.activate_terminal_startup = MagicMock()
    scheduler = Scheduler.__new__(Scheduler)

    Scheduler._activate_terminal_kv_manager(scheduler, manager)

    manager.install_terminal_runtime.assert_not_called()
    manager.activate_terminal_startup.assert_called_once_with()


def test_scheduler_source_binder_selects_serving_scheduler_boundary() -> None:
    """The source installation callback binds only its scheduler adapter."""

    scheduler_serving = object.__new__(TerminalSchedulerServing)
    scheduler_serving._role = TerminalSchedulerServingRole.SOURCE
    serving = object.__new__(PackedTerminalSourceServing)
    serving._scheduler_serving = scheduler_serving
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.terminal_source_serving = None
    scheduler.terminal_scheduler_serving = None
    scheduler.idle_sleeper = None

    Scheduler.bind_terminal_source_serving(scheduler, serving)

    assert scheduler.terminal_source_serving is serving
    assert scheduler.terminal_scheduler_serving is scheduler_serving


def test_production_source_owner_death_callback_enters_composed_abort() -> None:
    """The installed source callback makes runtime abort precede scheduler death."""

    source_serving = MagicMock(spec=PackedTerminalSourceServing)
    scheduler_serving = MagicMock(spec=TerminalSchedulerServing)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_source_serving = source_serving
    scheduler.terminal_scheduler_serving = scheduler_serving
    installation = NixlTerminalRuntimeInstallation(
        terminal_request_capacity=1,
        gateway_endpoint=None,
        bind_source_serving=None,
        bind_decode_serving=None,
        scheduler_process_fatal_handler=lambda inventory: None,
        owner_dead_handler=scheduler.mark_terminal_owner_dead,
    )

    installation.owner_dead_handler()

    source_serving.begin_fail_closed_abort.assert_called_once_with()
    scheduler_serving.mark_owner_dead.assert_not_called()


def test_scheduler_decode_binder_composes_queue_and_scheduler_atomically() -> None:
    """One callback exposes the same decode serving to both consumers."""

    scheduler_serving = object.__new__(TerminalSchedulerServing)
    scheduler_serving._role = TerminalSchedulerServingRole.DECODE
    composition = object.__new__(PackedTerminalDecodeServingComposition)
    composition._scheduler_serving = scheduler_serving
    serving = object.__new__(PackedTerminalDecodeServing)
    serving._decode_composition = composition
    queue = object.__new__(DecodePreallocQueue)
    queue._terminal_decode_serving = None
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = queue
    scheduler.terminal_scheduler_serving = None
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = None

    Scheduler.bind_terminal_decode_serving(scheduler, serving)

    assert queue.terminal_decode_serving is serving
    assert scheduler.terminal_decode_serving_composition is composition
    assert scheduler.terminal_scheduler_serving is scheduler_serving


def test_scheduler_retains_exact_process_fatal_inventory() -> None:
    """The scheduler preserves one immutable fatal receipt-inbox record."""

    inventory = SchedulerReceiptInboxInventory(
        physical_capacity=1,
        live_bindings=(),
        pending_request_keys=(),
        consuming_request_keys=(),
        outstanding_publications=0,
        active_delivery_intents=(),
        wake_armed=True,
        fatal_cause=SchedulerInboxFatalCause.OWNER_DEATH,
        closed=False,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_scheduler_process_fatal_inventory = None

    Scheduler.retain_terminal_scheduler_process_fatal(scheduler, inventory)
    Scheduler.retain_terminal_scheduler_process_fatal(scheduler, inventory)

    assert scheduler.terminal_scheduler_process_fatal_inventory == inventory
    changed = dataclasses.replace(inventory, wake_armed=False)
    with pytest.raises(RuntimeError, match="evidence changed"):
        Scheduler.retain_terminal_scheduler_process_fatal(scheduler, changed)
    nonfatal = dataclasses.replace(inventory, fatal_cause=None)
    with pytest.raises(ValueError, match="requires a fatal cause"):
        Scheduler.retain_terminal_scheduler_process_fatal(scheduler, nonfatal)


@pytest.mark.parametrize(
    "mode",
    (
        DisaggregationMode.PREFILL,
        DisaggregationMode.DECODE,
    ),
)
def test_scheduler_resolves_typed_terminal_nixl_manager(
    mode: DisaggregationMode,
) -> None:
    """Both PD roles resolve the exact startup-bound NIXL manager."""

    manager = object.__new__(scheduler_module.NixlKVManager)
    manager._terminal_startup_binding = object()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.transfer_backend = scheduler_module.TransferBackend.NIXL
    scheduler.disagg_prefill_bootstrap_queue = None
    scheduler.disagg_decode_prealloc_queue = None
    queue = SimpleNamespace(kv_manager=manager)
    if mode is DisaggregationMode.PREFILL:
        scheduler.disagg_prefill_bootstrap_queue = queue
    else:
        scheduler.disagg_decode_prealloc_queue = queue

    assert Scheduler.terminal_nixl_manager(scheduler) is manager
    manager._terminal_startup_binding = None
    assert Scheduler.terminal_nixl_manager(scheduler) is None


@pytest.mark.parametrize(
    ("behavior", "expected_process_fatal", "expected_signal_count"),
    (
        ("graceful", False, 0),
        ("unexpected", True, 1),
        ("exception", True, 1),
    ),
)
def test_scheduler_process_selects_graceful_or_fail_closed_teardown(
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    expected_process_fatal: bool,
    expected_signal_count: int,
) -> None:
    """Every process exit maps to one explicit terminal teardown disposition."""

    scheduler = _SchedulerProcessHarness(behavior)
    parent = _SchedulerParentProcess()
    pipe_writer = _SchedulerPipeWriter()
    server_args = SimpleNamespace(enable_trace=False)

    monkeypatch.setattr(scheduler_module, "load_plugins", lambda: None)
    monkeypatch.setattr(
        scheduler_module,
        "configure_scheduler_process",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_module.psutil,
        "Process",
        lambda: SimpleNamespace(parent=lambda: parent),
    )
    monkeypatch.setattr(scheduler_module, "Scheduler", lambda *args: scheduler)
    monkeypatch.setattr(
        scheduler_module.envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION,
        "get",
        lambda: False,
    )

    run_scheduler_process(
        server_args=server_args,
        port_args=object(),
        gpu_id=0,
        tp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        dp_rank=None,
        pipe_writer=pipe_writer,
    )

    assert pipe_writer.payloads == [{"status": "ready"}]
    assert scheduler.release_values == [expected_process_fatal]
    assert parent.signals == [signal.SIGQUIT] * expected_signal_count


def test_idle_sleeper_registers_raw_terminal_wake_descriptor() -> None:
    """The existing ZMQ poller wakes for the terminal inbox eventfd path."""

    sleeper = IdleSleeper(sockets=[])
    read_fd, write_fd = os.pipe()
    try:
        sleeper.register_file_descriptor(read_fd)
        os.write(write_fd, b"x")
        events = dict(sleeper.poller.poll(0))
        assert events[read_fd] & select.POLLIN
        sleeper.unregister_file_descriptor(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_terminal_cohort_requires_fd_driven_idle_sleep() -> None:
    """Terminal publication cannot share a process with idle busy polling."""

    tokenizer_read_fd, tokenizer_write_fd = os.pipe()
    rpc_read_fd, rpc_write_fd = os.pipe()
    try:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
        scheduler.server_args = SimpleNamespace(
            sleep_on_idle=False,
            pd_terminal_local_membership=object(),
        )
        scheduler.ipc_channels = SimpleNamespace(
            recv_from_tokenizer=tokenizer_read_fd,
            recv_from_rpc=rpc_read_fd,
        )

        Scheduler.init_idle_sleeper(scheduler)

        assert isinstance(scheduler.idle_sleeper, IdleSleeper)
    finally:
        os.close(tokenizer_read_fd)
        os.close(tokenizer_write_fd)
        os.close(rpc_read_fd)
        os.close(rpc_write_fd)


def test_nonterminal_cohort_retains_disabled_idle_sleep() -> None:
    """The terminal wake contract does not change ordinary scheduler polling."""

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.ps = SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
    scheduler.server_args = SimpleNamespace(
        sleep_on_idle=False,
        pd_terminal_local_membership=None,
    )

    Scheduler.init_idle_sleeper(scheduler)

    assert scheduler.idle_sleeper is None


@pytest.mark.parametrize(
    "loop",
    (
        SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill,
        SchedulerDisaggregationPrefillMixin.event_loop_overlap_disagg_prefill,
        SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode,
        SchedulerDisaggregationDecodeMixin.event_loop_overlap_disagg_decode,
    ),
)
def test_real_pd_loops_drain_before_receive(
    loop: object,
) -> None:
    """Every production PD loop drains authority before selecting work."""

    source = inspect.getsource(loop)
    drain_offset = source.index("self.drain_terminal_scheduler_receipts()")
    receive_offset = source.index("self.request_receiver.recv_requests()")
    assert drain_offset < receive_offset


def test_run_batch_handoffs_only_at_model_worker_submission_sites() -> None:
    """Production handoff cannot enclose scheduler-side result processing."""

    source = inspect.getsource(Scheduler.run_batch)
    assert source.count("self.submit_forward_with_terminal_handoff(") == 6
    assert "terminal_bind: Callable[[GenerationBatchResult]" in source
    assert source.count("terminal_bind,") == 4
    assert "launch_batch_with_terminal_handoff" not in source
    assert "lambda: self.model_worker.forward_batch_generation(" in source
    assert "lambda: self.tp_worker.forward_batch_split_prefill(batch)" in source
    assert "lambda: self.tp_worker.forward_batch_embedding(batch)" in source


def test_scheduler_process_does_not_close_consumer_before_runtime_drain() -> None:
    """The generic process finally block cannot preempt native abort routing."""

    source = inspect.getsource(run_scheduler_process)
    assert "close_terminal_scheduler_serving" not in source
    assert "scheduler.release(process_fatal=process_fatal)" in source


def test_scheduler_serving_path_contains_no_sleep_or_collective() -> None:
    """Scheduler handoff adds neither cadence polling nor rank synchronization."""

    from sglang.srt.disaggregation.terminal_progress import (
        scheduler_serving as scheduler_serving_module,
    )

    syntax = ast.parse(inspect.getsource(scheduler_serving_module))
    prohibited = tuple(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"sleep", "all_reduce", "barrier"}
    )
    assert prohibited == ()
