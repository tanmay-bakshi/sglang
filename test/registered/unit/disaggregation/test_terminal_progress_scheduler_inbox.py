import ast
import concurrent.futures
import errno
import inspect
import os
import random
import select
import threading

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress import (
    scheduler_inbox as scheduler_inbox_module,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    SchedulerInboxError,
    SchedulerInboxFatalCause,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerDeliveryIntent,
    SchedulerReceiptInboxFatalError,
    SchedulerReceiptPublishResult,
    TerminalReceiptInbox,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _identity(marker: int, role: TerminalOwnerRole) -> TerminalProcessIdentity:
    """Build one deterministic process identity.

    :param marker: Byte marker for the process generation.
    :param role: Process role carried by the identity.
    :returns: Exact process identity.
    """

    return TerminalProcessIdentity(
        process_generation=bytes([marker]) * 16,
        role=role,
        tp_rank=0,
        tp_size=1,
    )


def _binding(room_id: int, marker: int) -> TerminalRequestBinding:
    """Build one deterministic live request binding.

    :param room_id: Stable packed room identity.
    :param marker: Byte marker for generation and allocation identities.
    :returns: Exact request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=bytes([marker]) * 16,
        ),
        owner=_identity(marker=marker, role=TerminalOwnerRole.DECODE),
        rank_manifest_digest=bytes([marker]) * 32,
        allocation_digest=bytes([marker + 1]) * 32,
    )


def _receipt(
    binding: TerminalRequestBinding,
    marker: int,
) -> TerminalWireReceipt:
    """Build one deterministic decode-adoption receipt.

    :param binding: Exact live request binding.
    :param marker: Byte marker for issuer, timestamp, and nonce.
    :returns: Canonical fixed-width wire receipt.
    """

    return TerminalWireReceipt(
        binding=binding,
        issuer=_identity(marker=marker, role=TerminalOwnerRole.SOURCE),
        kind=TerminalReceiptKind.ADOPTION_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=marker,
        receipt_nonce=bytes([marker]) * 16,
    )


def _assert_readable(fd: int, expected: bool) -> None:
    """Assert the immediate readability state of one wake descriptor.

    :param fd: Descriptor to inspect without blocking.
    :param expected: Required readability state.
    """

    readable, _, _ = select.select([fd], [], [], 0)
    assert (len(readable) == 1) is expected


def _assert_conserved(inbox: TerminalReceiptInbox) -> None:
    """Assert the complete mutable inbox conservation relation.

    :param inbox: Runtime whose immutable inventory is inspected.
    """

    inventory = inbox.inventory()
    active_keys = (
        *inventory.pending_request_keys,
        *inventory.consuming_request_keys,
    )
    assert inventory.pending_count <= inventory.live_count
    assert inventory.live_count <= inventory.physical_capacity
    assert len(active_keys) == len(set(active_keys))
    assert set(active_keys).issubset(
        {binding.request_key for binding in inventory.live_bindings}
    )


class _SchedulerLaunchGate:
    """Record exact native scheduler launch-gate ownership."""

    _tokens: list[int]
    _ordering: list[str]
    _begin_entered: threading.Event | None
    _release_begin: threading.Event | None
    calls: list[tuple[str, int]]

    def __init__(
        self,
        *,
        tokens: tuple[int, ...],
        ordering: list[str],
        begin_entered: threading.Event | None = None,
        release_begin: threading.Event | None = None,
    ) -> None:
        """Create one deterministic launch gate.

        :param tokens: Exact tokens returned by successive acquisitions.
        :param ordering: Shared lifecycle ordering receipt.
        :param begin_entered: Optional first-acquisition observation event.
        :param release_begin: Optional first-acquisition release event.
        """

        self._tokens = list(tokens)
        self._ordering = ordering
        self._begin_entered = begin_entered
        self._release_begin = release_begin
        self.calls = []

    def begin_scheduler_launch_handoff(self) -> int:
        """Mint the next exact token after an optional controlled wait.

        :returns: Next configured launch token.
        """

        if len(self._tokens) == 0:
            raise AssertionError("unexpected scheduler launch-gate acquisition")
        token = self._tokens.pop(0)
        self.calls.append(("begin", token))
        self._ordering.append(f"begin:{token}")
        if self._begin_entered is not None:
            self._begin_entered.set()
            if self._release_begin is None:
                raise AssertionError("controlled begin requires a release event")
            if not self._release_begin.wait(timeout=5):
                raise AssertionError("scheduler launch-gate release timed out")
            self._begin_entered = None
            self._release_begin = None
        return token

    def end_scheduler_launch_handoff(self, token: int) -> None:
        """Record release of the exact token supplied by the inbox.

        :param token: Exact token returned by the matching acquisition.
        """

        self.calls.append(("end", token))
        self._ordering.append(f"end:{token}")


def test_publication_queues_before_fd_wake_and_loop_entry_drains_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authoritative FIFO changes before its coalesced wake is emitted."""

    inbox = TerminalReceiptInbox(physical_capacity=2)
    first_binding = _binding(room_id=1, marker=1)
    second_binding = _binding(room_id=2, marker=2)
    first_receipt = _receipt(first_binding, marker=11)
    second_receipt = _receipt(second_binding, marker=12)
    inbox.register_live(first_binding)
    inbox.register_live(second_binding)
    write_fd = inbox._write_fd
    real_write = os.write
    queue_sizes_at_wake: list[int] = []

    def observe_write(fd: int, payload: bytes) -> int:
        """Record authoritative queue size at the exact wake syscall."""

        if fd == write_fd:
            queue_sizes_at_wake.append(len(inbox._pending))
            assert len(inbox._active_encoded) == len(inbox._pending)
        return real_write(fd, payload)

    monkeypatch.setattr(os, "write", observe_write)
    _assert_readable(inbox.fileno(), expected=False)
    assert inbox.publish(first_receipt) is SchedulerReceiptPublishResult.QUEUED
    assert inbox.publish(second_receipt) is SchedulerReceiptPublishResult.QUEUED
    assert queue_sizes_at_wake == [1]
    assert inbox.inventory().pending_request_keys == (
        first_binding.request_key,
        second_binding.request_key,
    )
    _assert_readable(inbox.fileno(), expected=True)

    observed: list[TerminalWireReceipt] = []
    drained = inbox.drain_at_loop_entry(observed.append)

    assert drained == (first_receipt, second_receipt)
    assert observed == [first_receipt, second_receipt]
    assert inbox.inventory().live_count == 0
    _assert_readable(inbox.fileno(), expected=False)
    inbox.close()


def test_concurrent_byte_identical_publications_coalesce_once() -> None:
    """Concurrent retransmissions retain exactly one active receipt."""

    publication_count = 16
    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=10, marker=3)
    receipt = _receipt(binding, marker=13)
    inbox.register_live(binding)
    start = threading.Barrier(publication_count + 1)

    def publish() -> SchedulerReceiptPublishResult:
        """Publish after every worker is ready."""

        start.wait()
        return inbox.publish(TerminalWireReceipt.decode(receipt.encode()))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=publication_count
    ) as executor:
        futures = tuple(executor.submit(publish) for _ in range(publication_count))
        start.wait()
        results = tuple(future.result(timeout=5) for future in futures)

    assert results.count(SchedulerReceiptPublishResult.QUEUED) == 1
    assert results.count(SchedulerReceiptPublishResult.COALESCED) == 15
    assert inbox.inventory().pending_request_keys == (binding.request_key,)
    inbox.drain_at_loop_entry(lambda value: None)
    inbox.close()


def test_conflicting_duplicate_fails_before_second_enqueue() -> None:
    """A non-identical receipt enters sticky fatal state without mutation."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=20, marker=4)
    receipt = _receipt(binding, marker=14)
    conflict = _receipt(binding, marker=15)
    inbox.register_live(binding)
    inbox.publish(receipt)

    with pytest.raises(SchedulerReceiptInboxFatalError) as raised:
        inbox.publish(conflict)

    inventory = inbox.inventory()
    assert raised.value.cause is SchedulerInboxFatalCause.CONFLICTING_DUPLICATE
    assert inventory.fatal_cause is SchedulerInboxFatalCause.CONFLICTING_DUPLICATE
    assert inventory.pending_request_keys == (binding.request_key,)
    with pytest.raises(SchedulerReceiptInboxFatalError) as sticky:
        inbox.drain_at_loop_entry(lambda value: None)
    assert sticky.value.cause is SchedulerInboxFatalCause.CONFLICTING_DUPLICATE


def test_capacity_overflow_and_owner_death_are_sticky_active_failures() -> None:
    """Both fatal lifecycle entries wake the scheduler and retain first cause."""

    overflow = TerminalReceiptInbox(physical_capacity=1)
    first_binding = _binding(room_id=30, marker=5)
    second_binding = _binding(room_id=31, marker=6)
    overflow.register_live(first_binding)
    with pytest.raises(SchedulerReceiptInboxFatalError) as raised:
        overflow.register_live(second_binding)
    assert raised.value.cause is SchedulerInboxFatalCause.PHYSICAL_CAPACITY
    _assert_readable(overflow.fileno(), expected=True)
    with pytest.raises(SchedulerReceiptInboxFatalError) as sticky:
        overflow.unregister_live(first_binding)
    assert sticky.value.cause is SchedulerInboxFatalCause.PHYSICAL_CAPACITY

    owner_dead = TerminalReceiptInbox(physical_capacity=1)
    fatal_inventory = owner_dead.mark_owner_dead()
    assert fatal_inventory.fatal_cause is SchedulerInboxFatalCause.OWNER_DEATH
    _assert_readable(owner_dead.fileno(), expected=True)
    with pytest.raises(SchedulerReceiptInboxFatalError) as dead:
        owner_dead.register_live(first_binding)
    assert dead.value.cause is SchedulerInboxFatalCause.OWNER_DEATH
    owner_dead.close()


def test_wake_channel_failure_is_process_fatal_after_queue_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed active wake retains queued evidence and rejects publication."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=35, marker=6)
    receipt = _receipt(binding, marker=16)
    inbox.register_live(binding)
    write_fd = inbox._write_fd
    real_write = os.write

    def fail_wake(fd: int, payload: bytes) -> int:
        """Fail only the scheduler wake descriptor."""

        if fd == write_fd:
            raise BrokenPipeError(errno.EPIPE, "synthetic closed wake channel")
        return real_write(fd, payload)

    monkeypatch.setattr(os, "write", fail_wake)
    with pytest.raises(SchedulerReceiptInboxFatalError) as raised:
        inbox.publish(receipt)

    assert raised.value.cause is SchedulerInboxFatalCause.OWNER_DEATH
    assert raised.value.inventory.pending_request_keys == (binding.request_key,)
    assert inbox.inventory().fatal_cause is SchedulerInboxFatalCause.OWNER_DEATH


def test_pending_receipt_is_consumed_before_host_submission() -> None:
    """A receipt already queued at handoff wins the launch race."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=40, marker=7)
    receipt = _receipt(binding, marker=17)
    inbox.register_live(binding)
    inbox.publish(receipt)
    ordering: list[str] = []

    def submit() -> str:
        """Record narrow host submission."""

        ordering.append("submit")
        return "submitted"

    def consume(value: TerminalWireReceipt) -> None:
        """Record scheduler-affine receipt consumption."""

        assert value == receipt
        ordering.append("consume")

    result = inbox.launch_handoff(submit=submit, consume=consume)

    assert result == "submitted"
    assert ordering == ["consume", "submit"]
    inbox.close()


def test_native_launch_gate_begins_after_drain_and_spans_submit_and_bind() -> None:
    """The exact native token owns submission and immutable result binding."""

    ordering: list[str] = []
    token = (1 << 63) + 17
    launch_gate = _SchedulerLaunchGate(tokens=(token,), ordering=ordering)
    inbox = TerminalReceiptInbox(physical_capacity=1, launch_gate=launch_gate)
    binding = _binding(room_id=41, marker=27)
    receipt = _receipt(binding, marker=37)
    inbox.register_live(binding)
    inbox.publish(receipt)

    def consume(value: TerminalWireReceipt) -> None:
        """Record the already-ready receipt before native gate acquisition."""

        assert value == receipt
        ordering.append("consume")

    def submit() -> str:
        """Require native ownership around exact host submission."""

        assert launch_gate.calls == [("begin", token)]
        ordering.append("submit")
        return "submitted"

    def bind(result: str) -> str:
        """Require the same native ownership around immutable binding."""

        assert result == "submitted"
        assert launch_gate.calls == [("begin", token)]
        ordering.append("bind")
        return "bound"

    assert (
        inbox.launch_and_bind_handoff(submit=submit, bind=bind, consume=consume)
        == "bound"
    )
    assert launch_gate.calls == [("begin", token), ("end", token)]
    assert ordering == [
        "consume",
        f"begin:{token}",
        "submit",
        "bind",
        f"end:{token}",
    ]
    inbox.close()


@pytest.mark.parametrize("failure_site", ("submit", "bind"))
def test_native_launch_gate_releases_exact_token_after_callback_failure(
    failure_site: str,
) -> None:
    """Submission and binding failures cannot strand native gate ownership."""

    ordering: list[str] = []
    token = (1 << 62) + 29
    launch_gate = _SchedulerLaunchGate(tokens=(token,), ordering=ordering)
    inbox = TerminalReceiptInbox(physical_capacity=1, launch_gate=launch_gate)
    failure = RuntimeError(f"synthetic {failure_site} failure")

    def submit() -> str:
        """Return or raise at the configured lifecycle site."""

        ordering.append("submit")
        if failure_site == "submit":
            raise failure
        return "submitted"

    def bind(result: str) -> str:
        """Raise only after observing the exact submission result."""

        assert result == "submitted"
        ordering.append("bind")
        raise failure

    with pytest.raises(RuntimeError) as raised:
        inbox.launch_and_bind_handoff(
            submit=submit,
            bind=bind,
            consume=lambda value: None,
        )

    assert raised.value is failure
    assert launch_gate.calls == [("begin", token), ("end", token)]
    expected = [f"begin:{token}", "submit"]
    if failure_site == "bind":
        expected.append("bind")
    expected.append(f"end:{token}")
    assert ordering == expected
    inbox.close()


def test_publication_queued_during_native_acquisition_is_revalidated_before_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GIL-releasing acquisition cannot let newly ready work miss the launch."""

    ordering: list[str] = []
    first_token = 1009
    second_token = 1013
    begin_entered = threading.Event()
    release_begin = threading.Event()
    publication_queued = threading.Event()
    launch_gate = _SchedulerLaunchGate(
        tokens=(first_token, second_token),
        ordering=ordering,
        begin_entered=begin_entered,
        release_begin=release_begin,
    )
    inbox = TerminalReceiptInbox(physical_capacity=1, launch_gate=launch_gate)
    binding = _binding(room_id=42, marker=28)
    receipt = _receipt(binding, marker=38)
    inbox.register_live(binding)
    real_signal = inbox._signal_locked

    def observe_signal() -> None:
        """Expose the moment publication becomes authoritative."""

        real_signal()
        publication_queued.set()

    monkeypatch.setattr(inbox, "_signal_locked", observe_signal)

    def consume(value: TerminalWireReceipt) -> None:
        """Consume racing work before the eventual host submission."""

        assert value == receipt
        ordering.append("consume")

    def submit() -> str:
        """Record the only authorized host submission."""

        ordering.append("submit")
        return "submitted"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        handoff = executor.submit(inbox.launch_handoff, submit, consume)
        assert begin_entered.wait(timeout=5)
        publication = executor.submit(inbox.publish, receipt)
        queued_before_release = publication_queued.wait(timeout=5)
        release_begin.set()
        assert queued_before_release
        assert publication.result(timeout=5) is SchedulerReceiptPublishResult.QUEUED
        assert handoff.result(timeout=5) == "submitted"

    assert launch_gate.calls == [
        ("begin", first_token),
        ("end", first_token),
        ("begin", second_token),
        ("end", second_token),
    ]
    assert ordering == [
        f"begin:{first_token}",
        f"end:{first_token}",
        "consume",
        f"begin:{second_token}",
        "submit",
        f"end:{second_token}",
    ]
    inbox.close()


def test_launch_winner_drains_announced_publication_without_forward_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch winner immediately drains publications announced during submit."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=50, marker=8)
    receipt = _receipt(binding, marker=18)
    inbox.register_live(binding)
    begin_publication = threading.Event()
    publication_announced = threading.Event()
    forward_completion = threading.Event()
    original_begin = inbox._begin_publication_intent
    ordering: list[str] = []

    def observed_begin() -> int:
        """Expose the publication linearization point to the race test."""

        intent = original_begin()
        publication_announced.set()
        return intent

    monkeypatch.setattr(inbox, "_begin_publication_intent", observed_begin)

    def publisher() -> SchedulerReceiptPublishResult:
        """Announce one receipt while the launch gate is held."""

        assert begin_publication.wait(timeout=5)
        return inbox.publish(receipt)

    def submit() -> str:
        """Host-submit without releasing the synthetic forward barrier."""

        ordering.append("submit")
        begin_publication.set()
        assert publication_announced.wait(timeout=5)
        assert not forward_completion.is_set()
        return "submitted"

    def consume(value: TerminalWireReceipt) -> None:
        """Consume before any unrelated synthetic forward completion."""

        assert value == receipt
        assert not forward_completion.is_set()
        ordering.append("consume")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(publisher)
        result = inbox.launch_handoff(submit=submit, consume=consume)
        publication_result = publication.result(timeout=5)

    assert result == "submitted"
    assert publication_result is SchedulerReceiptPublishResult.QUEUED
    assert ordering == ["submit", "consume"]
    assert not forward_completion.is_set()
    inbox.close()


def test_announced_publication_wins_before_host_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An announced producer cannot be overtaken by scheduler mutex fairness."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=60, marker=9)
    receipt = _receipt(binding, marker=19)
    inbox.register_live(binding)
    publication_announced = threading.Event()
    original_begin = inbox._begin_publication_intent
    ordering: list[str] = []

    def observed_begin() -> int:
        """Expose publication intent before the producer reaches state."""

        intent = original_begin()
        publication_announced.set()
        return intent

    monkeypatch.setattr(inbox, "_begin_publication_intent", observed_begin)

    def submit() -> str:
        """Record the host launch after receipt consumption."""

        ordering.append("submit")
        return "submitted"

    def consume(value: TerminalWireReceipt) -> None:
        """Record the scheduler receipt winner."""

        assert value == receipt
        ordering.append("consume")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        inbox._state_lock.acquire()
        try:
            publication = executor.submit(inbox.publish, receipt)
            assert publication_announced.wait(timeout=5)
            handoff = executor.submit(inbox.launch_handoff, submit, consume)
        finally:
            inbox._state_lock.release()
        assert publication.result(timeout=5) is SchedulerReceiptPublishResult.QUEUED
        assert handoff.result(timeout=5) == "submitted"

    assert ordering == ["consume", "submit"]
    inbox.close()


def test_external_delivery_intent_blocks_launch_until_exact_completion() -> None:
    """A causal delivery owns the next host-launch boundary until completion."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=61, marker=21)
    receipt = _receipt(binding, marker=31)
    inbox.register_live(binding)
    intent = inbox.begin_delivery_intent(binding)
    launch_entered = threading.Event()
    submitted = threading.Event()

    def launch() -> str:
        """Enter the launch gate and return its exact result."""

        launch_entered.set()
        return inbox.launch_handoff(
            submit=lambda: submitted.set() or "submitted",
            consume=lambda value: None,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(launch)
        assert launch_entered.wait(timeout=5)
        assert not submitted.wait(timeout=0.05)
        inventory = inbox.inventory()
        assert inventory.outstanding_publications == 0
        assert inventory.active_delivery_intents == (intent,)
        inbox.complete_delivery_intent(intent)
        assert future.result(timeout=5) == "submitted"

    assert submitted.is_set()
    inbox.publish(receipt)
    inbox.drain_at_loop_entry(lambda value: None)
    inbox.close()


def test_delivery_intent_uses_independent_lock_and_rejects_forgery() -> None:
    """The exact token remains safe while scheduler-owned locks are unavailable."""

    inbox = TerminalReceiptInbox(physical_capacity=2)
    binding = _binding(room_id=62, marker=22)
    conflicting = _binding(room_id=63, marker=23)
    inbox.register_live(binding)
    inbox.register_live(conflicting)

    inbox._consumer_lock.acquire()
    inbox._state_lock.acquire()
    try:
        intent = inbox.begin_delivery_intent(binding)
    finally:
        inbox._state_lock.release()
        inbox._consumer_lock.release()

    forged = SchedulerDeliveryIntent(identity=intent.identity, binding=conflicting)
    with pytest.raises(SchedulerInboxError, match="forged"):
        inbox.complete_delivery_intent(forged)
    assert inbox.inventory().active_delivery_intents == (intent,)
    inbox.complete_delivery_intent(intent)
    with pytest.raises(SchedulerInboxError, match="already completed"):
        inbox.complete_delivery_intent(intent)

    inbox.mark_owner_dead()
    inbox.close_fail_closed()


def test_delivery_intent_may_outlive_reclaim_consumption() -> None:
    """Post-reclaim delivery remains a valid launch exclusion and inventory."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=65, marker=25)
    receipt = _receipt(binding, marker=35)
    inbox.register_live(binding)
    intent = inbox.begin_delivery_intent(binding)
    inbox.publish(receipt)

    assert inbox.drain_at_loop_entry(lambda value: None) == (receipt,)
    inventory = inbox.inventory()
    assert inventory.live_bindings == ()
    assert inventory.active_delivery_intents == (intent,)

    fatal_inventory = inbox.mark_owner_dead()
    assert fatal_inventory.fatal_cause is SchedulerInboxFatalCause.OWNER_DEATH
    assert fatal_inventory.active_delivery_intents == (intent,)
    inbox.complete_delivery_intent(intent)
    inbox.close_fail_closed()


def test_owner_death_releases_blocked_launch_into_fatal_without_submission() -> None:
    """A fatal owner wakes an excluded launch into the sticky fatal path."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=64, marker=24)
    inbox.register_live(binding)
    intent = inbox.begin_delivery_intent(binding)
    launch_entered = threading.Event()
    submitted = threading.Event()

    def launch() -> None:
        """Enter the excluded launch boundary."""

        launch_entered.set()
        inbox.launch_handoff(
            submit=lambda: submitted.set(),
            consume=lambda value: None,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(launch)
        assert launch_entered.wait(timeout=5)
        assert not submitted.wait(timeout=0.05)
        inbox.mark_owner_dead()
        inbox.complete_delivery_intent(intent)
        with pytest.raises(SchedulerReceiptInboxFatalError):
            future.result(timeout=5)

    assert not submitted.is_set()
    inbox.close_fail_closed()


def test_consumer_failure_retains_exact_evidence_and_enters_fatal_state() -> None:
    """A failed scheduler consumer cannot silently retire its live identity."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=70, marker=10)
    receipt = _receipt(binding, marker=20)
    inbox.register_live(binding)
    inbox.publish(receipt)

    def fail_consumer(value: TerminalWireReceipt) -> None:
        """Raise after receipt ownership transfers into consumption."""

        assert value == receipt
        raise ValueError("synthetic scheduler consumer failure")

    with pytest.raises(ValueError, match="synthetic scheduler"):
        inbox.drain_at_loop_entry(fail_consumer)

    inventory = inbox.inventory()
    assert inventory.fatal_cause is SchedulerInboxFatalCause.OWNER_DEATH
    assert inventory.live_bindings == (binding,)
    assert inventory.pending_request_keys == ()
    assert inventory.consuming_request_keys == (binding.request_key,)
    _assert_conserved(inbox)


def test_seeded_closed_loop_sequence_preserves_live_bound_and_one_shot_state() -> None:
    """Generated registration, publication, drain, and cancellation conserve state."""

    rng = random.Random(0xB300)
    capacity = 8
    inbox = TerminalReceiptInbox(physical_capacity=capacity)
    live: dict[PackedRequestKey, TerminalRequestBinding] = {}
    pending: dict[PackedRequestKey, TerminalWireReceipt] = {}
    next_room_id = 100

    for _ in range(1_000):
        publishable = tuple(key for key in live if key not in pending)
        cancellable = publishable
        operation = rng.randrange(5)
        if operation == 0 and len(live) < capacity:
            marker = (next_room_id % 100) + 30
            binding = _binding(room_id=next_room_id, marker=marker)
            next_room_id += 1
            inbox.register_live(binding)
            live[binding.request_key] = binding
        elif operation == 1 and len(publishable) > 0:
            request_key = rng.choice(publishable)
            receipt = _receipt(live[request_key], marker=21)
            assert inbox.publish(receipt) is SchedulerReceiptPublishResult.QUEUED
            pending[request_key] = receipt
        elif operation == 2 and len(pending) > 0:
            receipt = rng.choice(tuple(pending.values()))
            assert inbox.publish(receipt) is SchedulerReceiptPublishResult.COALESCED
        elif operation == 3 and len(pending) > 0:
            expected = tuple(pending.values())
            observed: list[TerminalWireReceipt] = []
            assert inbox.drain_at_loop_entry(observed.append) == expected
            assert tuple(observed) == expected
            for receipt in expected:
                del live[receipt.binding.request_key]
            pending.clear()
        elif len(cancellable) > 0:
            request_key = rng.choice(cancellable)
            inbox.unregister_live(live.pop(request_key))
        _assert_conserved(inbox)
        assert inbox.inventory().pending_request_keys == tuple(pending)

    if len(pending) > 0:
        drained = inbox.drain_at_loop_entry(lambda value: None)
        for receipt in drained:
            del live[receipt.binding.request_key]
    for binding in tuple(live.values()):
        inbox.unregister_live(binding)
    _assert_conserved(inbox)
    inbox.close()


def test_clean_close_rejects_retained_inventory_and_closes_descriptors() -> None:
    """Descriptor closure requires an empty live and publication inventory."""

    inbox = TerminalReceiptInbox(physical_capacity=1)
    binding = _binding(room_id=80, marker=11)
    inbox.register_live(binding)
    with pytest.raises(SchedulerInboxError, match="retained inventory"):
        inbox.close()
    inbox.unregister_live(binding)
    inbox.close()

    assert inbox.inventory().closed
    with pytest.raises(SchedulerInboxError, match="closed"):
        inbox.fileno()


def test_runtime_contains_no_sleep_polling() -> None:
    """The active scheduler path contains no sleep-based polling call."""

    syntax = ast.parse(inspect.getsource(scheduler_inbox_module))
    sleep_calls = tuple(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "sleep"
    )
    assert sleep_calls == ()
