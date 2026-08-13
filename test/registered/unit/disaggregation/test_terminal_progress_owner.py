import dataclasses
import os
import select
import threading

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.clock import ManualTerminalOwnerClock
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.event_source import (
    TerminalOwnerQueueEventSource,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
    SourceLifecycleEvent,
    SourceLifecycleEventKind,
)
from sglang.srt.disaggregation.terminal_progress.owner import (
    PackedTerminalProgressOwner,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    AcknowledgeTerminalReceipt,
    ApplyDecodeLifecycleEvent,
    ApplySourceLifecycleEvent,
    InjectTerminalOwnerFailure,
    RegisterDecodeLifecycle,
    RegisterSourceLifecycle,
    ScheduleTerminalDeadline,
    TerminalOwnerDisposition,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerEventSourceRegistration,
    TerminalOwnerFatalCause,
    TerminalOwnerOverflowError,
    TerminalOwnerPulse,
    TerminalOwnerReceiptEmission,
    TerminalOwnerTimingAnchor,
    TerminalOwnerTimingField,
    TerminalOwnerTimingSample,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
PUBLICATION_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")


def _binding(
    role: TerminalOwnerRole,
    *,
    room_id: int = 17,
) -> TerminalRequestBinding:
    """Build one exact owner-local request binding.

    :param role: Source or decode owner role.
    :param room_id: Stable room identifier.
    :returns: Immutable request and allocation binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=room_id.to_bytes(16, "big"),
        ),
        owner=TerminalProcessIdentity(
            process_generation=PROCESS_GENERATION,
            role=role,
            tp_rank=0,
            tp_size=1,
        ),
        rank_manifest_digest=bytes([room_id]) * 32,
        allocation_digest=bytes([room_id + 1]) * 32,
    )


def _publication(binding: TerminalRequestBinding) -> TerminalPublicationIdentity:
    """Build the canonical publication identity for one source binding.

    :param binding: Exact source request binding.
    :returns: Matching immutable publication generation.
    """

    return TerminalPublicationIdentity(
        request_key=binding.request_key,
        publisher_process_generation=PROCESS_GENERATION,
        publication_generation=PUBLICATION_GENERATION,
    )


def _running_owner(
    *,
    clock: ManualTerminalOwnerClock | None = None,
    event_sources: tuple[TerminalOwnerEventSourceRegistration, ...] = (),
) -> PackedTerminalProgressOwner:
    """Start one owner and wait for its process-lifetime reactor.

    :param clock: Optional deterministic owner clock.
    :param event_sources: Additional native-neutral fd sources.
    :returns: Running terminal progress owner.
    """

    owner = PackedTerminalProgressOwner(
        submission_capacity=128,
        output_capacity=128,
        event_sources=event_sources,
        clock=clock,
    )
    owner.start()
    owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.RUNNING,
        timeout_seconds=2.0,
    )
    return owner


def _fail_and_join(owner: PackedTerminalProgressOwner) -> None:
    """Enter explicit process-fatal disposition and join the owner.

    :param owner: Running terminal progress owner.
    """

    snapshot = owner.snapshot()
    if snapshot.disposition is TerminalOwnerDisposition.RUNNING:
        owner.submit(
            InjectTerminalOwnerFailure(
                cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
                reason="test-owned dependency retired",
            )
        )
    owner.wait_for_snapshot(
        lambda current: (
            current.disposition
            in (
                TerminalOwnerDisposition.PROCESS_FATAL,
                TerminalOwnerDisposition.STOPPED,
            )
        ),
        timeout_seconds=2.0,
    )
    assert owner.join(timeout_seconds=2.0)


def test_fd_source_coalesces_wakes_and_preserves_fifo_order() -> None:
    """Queue insertion is authoritative while one fd wake covers many events."""

    source = TerminalOwnerQueueEventSource(name="coalesced", capacity=4)
    try:
        envelopes = tuple(
            source.publish(TerminalOwnerPulse(), enqueued_ns=index)
            for index in range(4)
        )
        readable, _, _ = select.select((source.fileno(),), (), (), 0.0)
        assert readable == [source.fileno()]
        assert source.pending_count == 4

        drained = source.drain()
        assert drained == envelopes
        assert tuple(item.producer_sequence for item in drained) == (0, 1, 2, 3)
        assert source.pending_count == 0
        readable, _, _ = select.select((source.fileno(),), (), (), 0.0)
        assert readable == []
    finally:
        source.close()


class _GapEventSource(TerminalOwnerEventSource):
    """One-shot source which deliberately violates its sequence contract."""

    _read_fd: int
    _write_fd: int
    _drained: bool

    def __init__(self) -> None:
        """Create and signal one malformed event source."""

        self._read_fd, self._write_fd = os.pipe()
        self._drained = False
        os.write(self._write_fd, b"\x01")

    @property
    def name(self) -> str:
        """Return a stable malformed-source identity.

        :returns: Test source name.
        """

        return "gap-source"

    @property
    def pending_count(self) -> int:
        """Return whether the malformed envelope remains pending.

        :returns: Zero or one.
        """

        return int(not self._drained)

    def fileno(self) -> int:
        """Return the signalled pipe descriptor.

        :returns: Read-side descriptor.
        """

        return self._read_fd

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Return producer sequence one before sequence zero.

        :returns: One intentionally malformed envelope.
        """

        os.read(self._read_fd, 1)
        self._drained = True
        return (
            TerminalOwnerEventEnvelope(
                producer_sequence=1,
                enqueued_ns=0,
                command=TerminalOwnerPulse(),
            ),
        )

    def close(self) -> None:
        """Close both pipe descriptors."""

        os.close(self._read_fd)
        os.close(self._write_fd)


def test_event_source_sequence_gap_is_process_fatal() -> None:
    """The owner never sorts, skips, or silently accepts producer disorder."""

    source = _GapEventSource()
    owner = PackedTerminalProgressOwner(
        submission_capacity=16,
        output_capacity=16,
        event_sources=(
            TerminalOwnerEventSourceRegistration(
                source=source,
                close_on_shutdown=True,
            ),
        ),
    )
    owner.start()
    fatal = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.PROCESS_FATAL,
        timeout_seconds=2.0,
    )
    assert fatal.fatal_cause is TerminalOwnerFatalCause.EVENT_SOURCE_ORDER
    assert owner.join(timeout_seconds=2.0)


def test_bounded_event_source_overflow_is_process_fatal() -> None:
    """The owner converts a sticky producer overflow into lifecycle failure."""

    source = TerminalOwnerQueueEventSource(name="bounded-native", capacity=1)
    first_binding = _binding(TerminalOwnerRole.DECODE, room_id=20)
    rejected_binding = _binding(TerminalOwnerRole.DECODE, room_id=21)
    source.publish(
        RegisterDecodeLifecycle(
            binding=first_binding,
            trusted_authorities=frozenset(),
        ),
        enqueued_ns=0,
    )
    with pytest.raises(TerminalOwnerOverflowError):
        source.publish(
            RegisterDecodeLifecycle(
                binding=rejected_binding,
                trusted_authorities=frozenset(),
            ),
            enqueued_ns=1,
        )
    owner = PackedTerminalProgressOwner(
        submission_capacity=16,
        output_capacity=16,
        event_sources=(
            TerminalOwnerEventSourceRegistration(
                source=source,
                close_on_shutdown=True,
            ),
        ),
    )
    owner.start()
    fatal = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.PROCESS_FATAL,
        timeout_seconds=2.0,
    )
    assert fatal.fatal_cause is TerminalOwnerFatalCause.SUBMISSION_QUEUE_OVERFLOW
    assert {entry.binding for entry in fatal.quarantined} == {
        first_binding,
        rejected_binding,
    }
    assert owner.join(timeout_seconds=2.0)


def test_deadline_expiry_quarantines_the_exact_request() -> None:
    """A frozen request deadline fails closed from one deterministic clock step."""

    clock = ManualTerminalOwnerClock(initial_ns=0)
    owner = _running_owner(clock=clock)
    binding = _binding(TerminalOwnerRole.DECODE)
    owner.submit(
        RegisterDecodeLifecycle(binding=binding, trusted_authorities=frozenset())
    )
    owner.submit(
        ApplyDecodeLifecycleEvent(
            binding=binding,
            event=DecodeLifecycleEvent(
                kind=DecodeLifecycleEventKind.ALLOCATION_PUBLISHED
            ),
        )
    )
    owner.submit(
        ScheduleTerminalDeadline(
            binding=binding,
            kind=TerminalDeadlineKind.OWNER_DECODE_SCATTER,
            started_ns=0,
        )
    )
    owner.wait_for_snapshot(
        lambda snapshot: snapshot.owner_transition_count >= 3,
        timeout_seconds=2.0,
    )

    clock.advance_ns(
        terminal_deadline_spec(TerminalDeadlineKind.OWNER_DECODE_SCATTER).duration_ns
    )
    owner.notify_clock_advanced()
    expired = owner.wait_for_snapshot(
        lambda snapshot: len(snapshot.quarantined) == 1,
        timeout_seconds=2.0,
    )
    assert expired.disposition is TerminalOwnerDisposition.RUNNING
    assert expired.quarantined[0].binding == binding
    assert binding not in expired.decode_active
    owner.begin_shutdown(started_ns=clock.now_ns())
    owner.retire_shutdown_producers()
    stopped = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.STOPPED,
        timeout_seconds=2.0,
    )
    assert stopped.quarantined[0].binding == binding
    assert owner.join(timeout_seconds=2.0)


def test_explicit_owner_dependency_death_quarantines_active_inventory() -> None:
    """A required thread death closes admission and preserves every resource."""

    owner = _running_owner()
    binding = _binding(TerminalOwnerRole.SOURCE)
    owner.submit(
        RegisterSourceLifecycle(
            binding=binding,
            publication_identity=_publication(binding),
            trusted_authorities=frozenset(),
        )
    )
    owner.wait_for_snapshot(
        lambda snapshot: binding in snapshot.source_active,
        timeout_seconds=2.0,
    )
    owner.submit(
        InjectTerminalOwnerFailure(
            cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
            reason="publisher thread exited",
        )
    )
    fatal = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.PROCESS_FATAL,
        timeout_seconds=2.0,
    )
    assert not fatal.admission_open
    assert fatal.fatal_cause is TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH
    assert len(fatal.quarantined) == 1
    assert fatal.quarantined[0].binding == binding
    assert owner.join(timeout_seconds=2.0)


def test_shutdown_drains_terminal_quarantine_inventory() -> None:
    """Shutdown stops only after producers join and ambiguity is inventoried."""

    owner = _running_owner()
    binding = _binding(TerminalOwnerRole.DECODE)
    owner.submit(
        RegisterDecodeLifecycle(binding=binding, trusted_authorities=frozenset())
    )
    owner.submit(
        ApplyDecodeLifecycleEvent(
            binding=binding,
            event=DecodeLifecycleEvent(
                kind=DecodeLifecycleEventKind.CANCEL_UNPUBLISHED,
                reason="request cancelled before publication",
            ),
        )
    )
    owner.wait_for_snapshot(
        lambda snapshot: binding in snapshot.safely_retired,
        timeout_seconds=2.0,
    )
    owner.begin_shutdown()
    owner.retire_shutdown_producers()
    stopped = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.STOPPED,
        timeout_seconds=2.0,
    )
    assert not stopped.admission_open
    assert stopped.safely_retired == (binding,)
    assert stopped.quarantined == ()
    assert stopped.pending_receipts == ()
    assert owner.join(timeout_seconds=2.0)


def test_handoff_advances_while_unrelated_forward_barrier_stays_held() -> None:
    """Receipt admission never waits for an unrelated forward completion."""

    clock = ManualTerminalOwnerClock(initial_ns=100)
    owner = _running_owner(clock=clock)
    binding = _binding(TerminalOwnerRole.SOURCE)
    owner.submit(
        RegisterSourceLifecycle(
            binding=binding,
            publication_identity=_publication(binding),
            trusted_authorities=frozenset(),
        )
    )
    owner.wait_for_snapshot(
        lambda snapshot: snapshot.owner_transition_count >= 1,
        timeout_seconds=2.0,
    )

    unrelated_forward = threading.Event()
    submitter_done = threading.Event()

    def submit_before_forward_release() -> None:
        owner.submit(
            ApplySourceLifecycleEvent(
                binding=binding,
                event=SourceLifecycleEvent(
                    kind=SourceLifecycleEventKind.SUBMISSION_ACCEPTED
                ),
                timing_anchor=TerminalOwnerTimingAnchor(
                    field=TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF,
                    sample_key="source-rank-0",
                    started_ns=100,
                ),
            ),
            enqueued_ns=100,
        )
        submitter_done.set()
        unrelated_forward.wait(timeout=2.0)

    submitter = threading.Thread(target=submit_before_forward_release)
    submitter.start()
    assert submitter_done.wait(timeout=2.0)
    owner.wait_for_output_count(minimum_count=1, timeout_seconds=2.0)
    assert not unrelated_forward.is_set()
    output = owner.drain_outputs()
    assert len(output) == 1
    assert type(output[0]) is TerminalOwnerTimingSample
    assert output[0].duration_ns == 0
    unrelated_forward.set()
    submitter.join(timeout=2.0)
    assert not submitter.is_alive()
    _fail_and_join(owner)


def test_owner_mints_immutable_adoption_authority_and_tracks_ack() -> None:
    """Decode adoption authority remains pending until exact consumer ACK."""

    clock = ManualTerminalOwnerClock(initial_ns=10)
    owner = _running_owner(clock=clock)
    binding = _binding(TerminalOwnerRole.DECODE)
    owner.submit(
        RegisterDecodeLifecycle(binding=binding, trusted_authorities=frozenset())
    )
    path = (
        DecodeLifecycleEventKind.ALLOCATION_PUBLISHED,
        DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
        DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
        DecodeLifecycleEventKind.SCATTER_STARTED,
        DecodeLifecycleEventKind.SCATTER_TERMINAL,
        DecodeLifecycleEventKind.TEARDOWN_SENT,
        DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
        DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
    )
    for event_kind in path:
        owner.submit(
            ApplyDecodeLifecycleEvent(
                binding=binding,
                event=DecodeLifecycleEvent(kind=event_kind),
            )
        )
    readable, _, _ = select.select((owner.output_fileno(),), (), (), 2.0)
    assert readable == [owner.output_fileno()]
    owner.wait_for_output_count(minimum_count=1, timeout_seconds=2.0)
    outputs = owner.drain_outputs()
    readable, _, _ = select.select((owner.output_fileno(),), (), (), 0.0)
    assert readable == []
    assert len(outputs) == 1
    emission = outputs[0]
    assert type(emission) is TerminalOwnerReceiptEmission
    assert emission.receipt.kind is TerminalReceiptKind.ADOPTION_READY
    with pytest.raises(dataclasses.FrozenInstanceError):
        emission.emitted_ns = 11  # type: ignore[misc]
    pending = owner.snapshot()
    assert pending.pending_receipts == (emission.receipt,)

    owner.submit(AcknowledgeTerminalReceipt(receipt=emission.receipt))
    owner.wait_for_snapshot(
        lambda snapshot: len(snapshot.pending_receipts) == 0,
        timeout_seconds=2.0,
    )
    _fail_and_join(owner)
