import ast
import concurrent.futures
import errno
import inspect
import os
import select
import threading
from collections.abc import Callable

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
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
    TerminalSchedulerServing,
    TerminalSchedulerServingRole,
    TerminalSourceSchedulerConsumer,
)
from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def test_source_launch_binds_submission_before_gate_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal submission binder remains inside the serialized handoff."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer)
    gate_held = False
    original_handoff = serving._inbox.launch_handoff

    def observed_handoff(
        submit: Callable[[], object],
        consume: Callable[[object], None],
    ) -> object:
        """Expose the full launch-gate callback lifetime."""

        nonlocal gate_held
        gate_held = True
        try:
            return original_handoff(submit, consume)
        finally:
            gate_held = False

    monkeypatch.setattr(serving._inbox, "launch_handoff", observed_handoff)

    def submit() -> str:
        """Return one synthetic model-worker result while the gate is held."""

        assert gate_held
        return "forward-result"

    def bind(result: str) -> str:
        """Freeze one synthetic terminal submission before gate release."""

        assert gate_held
        assert result == "forward-result"
        return "bound-submission"

    assert serving.launch_and_bind_handoff(submit, bind) == "bound-submission"
    assert not gate_held
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

    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.publish_action(action)

    readable, _, _ = select.select([serving.fileno()], [], [], 0)
    assert readable == [serving.fileno()]
    with pytest.raises(SchedulerReceiptInboxFatalError):
        serving.drain_at_loop_entry()
    assert len(consumer.fatal_inventories) == 1
    assert serving.inventory().retained_action_ids == ()
    serving.close_fail_closed()


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


def test_fail_closed_teardown_closes_fd_and_retains_ambiguous_authority() -> None:
    """Fatal closure preserves live evidence without leaving a wake fd open."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    binding = _binding(40, 40, TerminalOwnerRole.SOURCE)
    action = _action(binding, 40)
    serving.register_request(binding)
    serving.publish_action(action)
    file_descriptor = serving.fileno()
    serving.mark_owner_dead()

    inventory = serving.close_fail_closed()

    assert inventory.inbox.closed
    assert inventory.inbox.live_bindings == (binding,)
    assert inventory.inbox.pending_request_keys == (binding.request_key,)
    assert inventory.retained_action_ids == (action.action_id,)
    assert inventory.fatal_delivered
    assert len(consumer.fatal_inventories) == 1
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


def test_scheduler_process_fatal_teardown_unregisters_and_closes_inbox() -> None:
    """Scheduler teardown removes the poll target before fail-closed closure."""

    consumer = _DecodeConsumer()
    serving = _decode_serving(consumer, capacity=1)
    sleeper = IdleSleeper(sockets=[])
    file_descriptor = serving.fileno()
    sleeper.register_file_descriptor(file_descriptor)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_scheduler_serving = serving
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = sleeper

    Scheduler.close_terminal_scheduler_serving(scheduler, process_fatal=True)

    assert scheduler.terminal_scheduler_serving is None
    assert file_descriptor not in sleeper._file_descriptors
    assert len(consumer.fatal_inventories) == 1
    with pytest.raises(OSError) as raised:
        os.fstat(file_descriptor)
    assert raised.value.errno == errno.EBADF


def test_graceful_scheduler_teardown_quarantines_live_generation() -> None:
    """A graceful process exit cannot retire ambiguous in-flight ownership."""

    consumer = _SourceConsumer()
    serving = _source_serving(consumer, capacity=1)
    binding = _binding(41, 41, TerminalOwnerRole.SOURCE)
    serving.register_request(binding)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.terminal_scheduler_serving = serving
    scheduler.terminal_decode_serving_composition = None
    scheduler.idle_sleeper = None

    Scheduler.close_terminal_scheduler_serving(scheduler, process_fatal=False)

    assert scheduler.terminal_scheduler_serving is None
    assert len(consumer.fatal_inventories) == 1
    inventory = consumer.fatal_inventories[0]
    assert inventory.live_bindings == (binding,)
    assert inventory.closed is False
    assert serving.inventory().inbox.closed


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
    assert "launch_batch_with_terminal_handoff" not in source
    assert "lambda: self.model_worker.forward_batch_generation(" in source
    assert "lambda: self.tp_worker.forward_batch_split_prefill(batch)" in source
    assert "lambda: self.tp_worker.forward_batch_embedding(batch)" in source


def test_scheduler_process_does_not_close_consumer_before_runtime_drain() -> None:
    """The generic process finally block cannot preempt native abort routing."""

    source = inspect.getsource(run_scheduler_process)
    assert "close_terminal_scheduler_serving" not in source


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
