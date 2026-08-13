import dataclasses

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    BoundedSchedulerInbox,
    SchedulerInboxDisposition,
    SchedulerInboxError,
    SchedulerInboxFatalCause,
    SchedulerInboxOverflow,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _binding(room_id: int, marker: int) -> TerminalRequestBinding:
    """Build one deterministic decode-side request binding.

    :param room_id: Stable packed room identity.
    :param marker: Byte marker distinguishing generations and allocations.
    :returns: Exact terminal request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=bytes([marker]) * 16,
        ),
        owner=TerminalProcessIdentity(
            process_generation=b"p" * 16,
            role=TerminalOwnerRole.DECODE,
            tp_rank=0,
            tp_size=1,
        ),
        rank_manifest_digest=bytes([marker]) * 32,
        allocation_digest=bytes([marker + 1]) * 32,
    )


def _receipt(
    issuer: TerminalReceiptIssuer,
    binding: TerminalRequestBinding,
    timestamp_ns: int,
) -> TerminalReceipt:
    """Issue one successful scheduler adoption receipt.

    :param issuer: Trusted process-local receipt issuer.
    :param binding: Exact live request binding.
    :param timestamp_ns: Deterministic terminal timestamp.
    :returns: Immutable issued receipt.
    """

    return issuer.issue(
        binding=binding,
        kind=TerminalReceiptKind.ADOPTION_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=timestamp_ns,
    )


def test_exact_duplicate_coalesces_and_fifo_consumption_preserves_identity() -> None:
    """One issued receipt occupies exactly one slot despite retransmission."""

    issuer = TerminalReceiptIssuer()
    first_binding = _binding(room_id=1, marker=1)
    second_binding = _binding(room_id=2, marker=2)
    first_receipt = _receipt(issuer, first_binding, timestamp_ns=10)
    second_receipt = _receipt(issuer, second_binding, timestamp_ns=20)
    inbox = BoundedSchedulerInbox(physical_capacity=2)
    inbox = inbox.register_live(first_binding).register_live(second_binding)

    first_enqueue = inbox.enqueue(first_receipt)
    duplicate_enqueue = first_enqueue.enqueue(first_receipt)
    complete = duplicate_enqueue.enqueue(second_receipt)

    assert duplicate_enqueue is first_enqueue
    assert complete.pending_count == 2
    assert complete.pending_count <= complete.live_count
    after_first, consumed_first = complete.consume_next()
    after_second, consumed_second = after_first.consume_next()
    assert consumed_first is first_receipt
    assert consumed_second is second_receipt
    assert after_second.pending_count == 0
    assert after_second.unregister_live(first_binding).live_count == 1


def test_conflicting_duplicate_and_physical_overflow_share_fatal_disposition() -> None:
    """Logical and physical queue overflow both require process termination."""

    issuer = TerminalReceiptIssuer()
    first_binding = _binding(room_id=10, marker=3)
    second_binding = _binding(room_id=11, marker=4)
    first_receipt = _receipt(issuer, first_binding, timestamp_ns=10)
    conflict = _receipt(issuer, first_binding, timestamp_ns=10)

    logical = (
        BoundedSchedulerInbox(physical_capacity=2)
        .register_live(first_binding)
        .enqueue(first_receipt)
        .enqueue(conflict)
    )
    assert logical.disposition is SchedulerInboxDisposition.PROCESS_FATAL
    assert logical.fatal_cause is SchedulerInboxFatalCause.CONFLICTING_DUPLICATE
    assert logical.pending_count == 1

    physical_control = (
        BoundedSchedulerInbox(physical_capacity=1)
        .register_live(first_binding)
        .register_live(second_binding)
        .enqueue(first_receipt)
    )
    physical = physical_control.enqueue(
        _receipt(issuer, second_binding, timestamp_ns=20)
    )
    assert physical.disposition is SchedulerInboxDisposition.PROCESS_FATAL
    assert physical.fatal_cause is SchedulerInboxFatalCause.PHYSICAL_CAPACITY
    assert physical.pending == physical_control.pending
    with pytest.raises(SchedulerInboxError):
        physical.consume_next()


def test_stale_and_wrong_binding_fail_before_queue_mutation() -> None:
    """A receipt cannot be redirected into another live allocation."""

    issuer = TerminalReceiptIssuer()
    binding = _binding(room_id=20, marker=5)
    stale_binding = dataclasses.replace(
        binding,
        allocation_digest=b"z" * 32,
    )
    inbox = BoundedSchedulerInbox(physical_capacity=1).register_live(binding)

    with pytest.raises(SchedulerInboxError):
        inbox.enqueue(_receipt(issuer, stale_binding, timestamp_ns=10))
    with pytest.raises(SchedulerInboxError):
        inbox.enqueue(
            _receipt(
                issuer,
                _binding(room_id=21, marker=6),
                timestamp_ns=20,
            )
        )
    assert inbox.pending_count == 0
    assert inbox.disposition is SchedulerInboxDisposition.HEALTHY


def test_pending_never_exceeds_live_count_under_closed_loop_replacement() -> None:
    """Generated enqueue-consume-retire cycles conserve the logical bound."""

    issuer = TerminalReceiptIssuer()
    inbox = BoundedSchedulerInbox(physical_capacity=4)
    next_room_id = 100
    for wave in range(25):
        bindings = tuple(
            _binding(room_id=next_room_id + offset, marker=offset + 1)
            for offset in range(4)
        )
        next_room_id += 4
        for binding in bindings:
            inbox = inbox.register_live(binding)
        for offset, binding in enumerate(reversed(bindings)):
            inbox = inbox.enqueue(
                _receipt(
                    issuer,
                    binding,
                    timestamp_ns=wave * 10 + offset,
                )
            )
            assert inbox.pending_count <= inbox.live_count
        for _ in bindings:
            inbox, consumed = inbox.consume_next()
            inbox = inbox.unregister_live(consumed.binding)
            assert inbox.pending_count <= inbox.live_count

    assert inbox.live_count == 0
    assert inbox.pending_count == 0


def test_owner_death_is_process_fatal_and_preserves_pending_evidence() -> None:
    """Inbox owner death cannot strand a healthy scheduler process."""

    issuer = TerminalReceiptIssuer()
    binding = _binding(room_id=30, marker=7)
    pending = (
        BoundedSchedulerInbox(physical_capacity=1)
        .register_live(binding)
        .enqueue(_receipt(issuer, binding, timestamp_ns=10))
    )

    fatal = pending.mark_owner_dead()
    assert fatal.disposition is SchedulerInboxDisposition.PROCESS_FATAL
    assert fatal.fatal_cause is SchedulerInboxFatalCause.OWNER_DEATH
    assert fatal.pending == pending.pending
    with pytest.raises(dataclasses.FrozenInstanceError):
        fatal.pending = ()  # type: ignore[misc]


def test_unregister_rejects_pending_and_conflicting_live_binding() -> None:
    """A live allocation cannot disappear while scheduler authority is queued."""

    issuer = TerminalReceiptIssuer()
    binding = _binding(room_id=40, marker=8)
    pending = (
        BoundedSchedulerInbox(physical_capacity=1)
        .register_live(binding)
        .enqueue(_receipt(issuer, binding, timestamp_ns=10))
    )
    with pytest.raises(SchedulerInboxError, match="pending"):
        pending.unregister_live(binding)

    conflicting = dataclasses.replace(binding, allocation_digest=b"q" * 32)
    clean = BoundedSchedulerInbox(physical_capacity=1).register_live(binding)
    with pytest.raises(SchedulerInboxError, match="another binding"):
        clean.register_live(conflicting)


def test_direct_construction_rejects_logical_and_physical_overflow() -> None:
    """Callers cannot fabricate an inbox that violates either queue bound."""

    issuer = TerminalReceiptIssuer()
    first_binding = _binding(room_id=50, marker=9)
    second_binding = _binding(room_id=51, marker=10)
    first_receipt = _receipt(issuer, first_binding, timestamp_ns=10)
    second_receipt = _receipt(issuer, second_binding, timestamp_ns=20)
    complete = (
        BoundedSchedulerInbox(physical_capacity=2)
        .register_live(first_binding)
        .register_live(second_binding)
        .enqueue(first_receipt)
        .enqueue(second_receipt)
    )

    with pytest.raises(ValueError, match="live request"):
        dataclasses.replace(complete, live_bindings=(first_binding,))
    with pytest.raises(ValueError, match="physical capacity"):
        dataclasses.replace(complete, physical_capacity=1)


def test_observed_logical_and_physical_overflow_share_fatal_path() -> None:
    """External bounded storage reports both overflow classes fail-closed."""

    binding = _binding(room_id=60, marker=11)
    inbox = BoundedSchedulerInbox(physical_capacity=2).register_live(binding)

    logical = inbox.observe_overflow(
        SchedulerInboxOverflow(
            pending_count=2,
            live_inflight_count=1,
            physical_capacity=2,
        )
    )
    assert logical.disposition is SchedulerInboxDisposition.PROCESS_FATAL
    assert logical.fatal_cause is SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT

    physical = inbox.observe_overflow(
        SchedulerInboxOverflow(
            pending_count=3,
            live_inflight_count=1,
            physical_capacity=2,
        )
    )
    assert physical.disposition is SchedulerInboxDisposition.PROCESS_FATAL
    assert physical.fatal_cause is SchedulerInboxFatalCause.PHYSICAL_CAPACITY
