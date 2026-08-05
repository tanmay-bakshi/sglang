import concurrent.futures
import logging
import sys
import threading

import pytest

from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliaryAllocationError,
    PackedAuxiliaryAllocationLease,
    PackedAuxiliaryAllocationLeaseAuthority,
    PackedAuxiliaryAllocationState,
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

METADATA_GENERATION_BYTES = 16
DESCRIPTOR_DIGEST = b"d" * 32
EVIDENCE_DIGEST = b"e" * 32


class RecordingSlotAllocation:
    """CPU-only metadata-row allocator with deterministic fault injection."""

    allocate_failure: Exception | None
    block_release: bool
    calls: list[str]
    next_slot_index: int
    owners: list[tuple[str, object]]
    quarantine_failure: Exception | None
    quarantined: list[object]
    release_continue: threading.Event
    release_entered: threading.Event
    release_failure: Exception | None
    released: list[object]
    snapshot_failure: Exception | None
    snapshots: dict[object, PackedAuxiliarySlotReservationSnapshot]

    def __init__(self) -> None:
        """Initialize an empty allocator history with fault injection disabled."""

        self.allocate_failure = None
        self.block_release = False
        self.calls = []
        self.next_slot_index = 0
        self.owners = []
        self.quarantine_failure = None
        self.quarantined = []
        self.release_continue = threading.Event()
        self.release_entered = threading.Event()
        self.release_failure = None
        self.released = []
        self.snapshot_failure = None
        self.snapshots = {}

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Mint one opaque row reservation.

        :param owner: Exact authority reservation owner.
        :returns: Opaque allocator-minted reservation.
        """

        self.calls.append("allocate")
        self.owners.append(("allocate", owner))
        if self.allocate_failure is not None:
            raise self.allocate_failure
        slot_index = self.next_slot_index
        self.next_slot_index += 1
        reservation = object()
        base_address = 0x100000 + (slot_index * 0x1000)
        self.snapshots[reservation] = PackedAuxiliarySlotReservationSnapshot(
            metadata_buffer_index=slot_index,
            metadata_slot_generation=(slot_index + 1).to_bytes(
                METADATA_GENERATION_BYTES,
                "big",
            ),
            destination_segments=(
                PackedAuxiliaryDestinationSegment(
                    address=base_address,
                    item_length=64,
                ),
                PackedAuxiliaryDestinationSegment(
                    address=base_address + 0x100,
                    item_length=32,
                ),
            ),
        )
        return reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Resolve allocator-authored row identity and geometry.

        :param reservation: Exact allocator-minted reservation.
        :returns: Immutable row snapshot.
        """

        self.calls.append("snapshot")
        if self.snapshot_failure is not None:
            raise self.snapshot_failure
        return self.snapshots[reservation]

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Release one row or inject an ambiguous callback failure.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact authority reservation owner.
        """

        self.calls.append("release")
        self.owners.append(("release", owner))
        if self.block_release:
            self.release_entered.set()
            if not self.release_continue.wait(timeout=5):
                raise TimeoutError("blocked release test timed out")
        if self.release_failure is not None:
            raise self.release_failure
        self.released.append(reservation)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Quarantine one row or inject callback failure.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact authority reservation owner.
        """

        self.calls.append("quarantine")
        self.owners.append(("quarantine", owner))
        if self.quarantine_failure is not None:
            raise self.quarantine_failure
        self.quarantined.append(reservation)


def _new_authority() -> tuple[
    PackedAuxiliaryAllocationLeaseAuthority,
    object,
    object,
]:
    """Construct one authority and both exact transition principals.

    :returns: Authority, lifecycle owner, and scheduler consumer owner.
    """

    lifecycle_authority = object()
    consumer_authority = object()
    return (
        PackedAuxiliaryAllocationLeaseAuthority(
            lifecycle_authority,
            consumer_authority,
        ),
        lifecycle_authority,
        consumer_authority,
    )


def _drive_to_request_commit(
    authority: PackedAuxiliaryAllocationLeaseAuthority,
    lease: PackedAuxiliaryAllocationLease,
    lifecycle_authority: object,
) -> None:
    """Drive one prepared lease through exact source teardown.

    :param authority: Exact metadata-row authority.
    :param lease: Prepared row lease.
    :param lifecycle_authority: Exact transport lifecycle principal.
    """

    metadata_generation = authority.snapshot(lease).metadata_slot_generation
    authority.record_publication(lease, lifecycle_authority)
    authority.record_submission(lease, lifecycle_authority)
    authority.record_teardown_completion(
        lease,
        lifecycle_authority,
        metadata_slot_generation=metadata_generation,
        native_dram_handle_generation=17,
        descriptor_digest=DESCRIPTOR_DIGEST,
        evidence_digest=EVIDENCE_DIGEST,
    )
    authority.commit_to_request_after_teardown(lease, lifecycle_authority)


def test_auxiliary_allocation_follows_exact_consumption_lifecycle() -> None:
    """Retain one row until exact scheduler consumption and then retire it."""

    allocation = RecordingSlotAllocation()
    authority, lifecycle_authority, consumer_authority = _new_authority()

    lease = authority.acquire(allocation)
    prepared = authority.snapshot(lease)
    assert prepared.state is PackedAuxiliaryAllocationState.PREPARED
    assert prepared.metadata_buffer_index == 0
    assert prepared.native_dram_handle_generation is None

    authority.record_publication(lease, lifecycle_authority)
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.PUBLISHED
    authority.record_submission(lease, lifecycle_authority)
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.SUBMITTED
    authority.record_teardown_completion(
        lease,
        lifecycle_authority,
        metadata_slot_generation=prepared.metadata_slot_generation,
        native_dram_handle_generation=17,
        descriptor_digest=DESCRIPTOR_DIGEST,
        evidence_digest=EVIDENCE_DIGEST,
    )
    teardown = authority.snapshot(lease)
    assert teardown.state is PackedAuxiliaryAllocationState.TEARDOWN_COMPLETED
    assert teardown.native_dram_handle_generation == 17
    assert teardown.descriptor_digest == DESCRIPTOR_DIGEST
    assert teardown.evidence_digest == EVIDENCE_DIGEST

    authority.commit_to_request_after_teardown(lease, lifecycle_authority)
    assert (
        authority.snapshot(lease).state
        is PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST
    )
    assert len(allocation.released) == 0
    authority.release_after_consumption(lease, consumer_authority)
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.RELEASED
    assert len(allocation.released) == 1
    assert allocation.calls == ["allocate", "snapshot", "release"]
    assert allocation.owners[0][1] is allocation.owners[1][1]

    authority.retire_terminal(lease)
    with pytest.raises(PackedAuxiliaryAllocationError, match="not registered"):
        authority.snapshot(lease)


def test_unpublished_cancellation_releases_and_retires_the_row() -> None:
    """Cancel only a prepared row and forget it after safe allocator release."""

    allocation = RecordingSlotAllocation()
    authority, _, _ = _new_authority()
    lease = authority.acquire(allocation)

    authority.cancel_unpublished(lease)
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.CANCELLED
    assert len(allocation.released) == 1
    authority.retire_terminal(lease)


def test_lifecycle_requires_exact_order_and_transition_principals() -> None:
    """Reject stale ordering, row generation, and merely equivalent owners."""

    allocation = RecordingSlotAllocation()
    authority, lifecycle_authority, consumer_authority = _new_authority()
    lease = authority.acquire(allocation)
    prepared = authority.snapshot(lease)

    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="requires the lifecycle authority",
    ):
        authority.record_publication(lease, object())
    with pytest.raises(PackedAuxiliaryAllocationError, match="expected published"):
        authority.record_submission(lease, lifecycle_authority)
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.PREPARED

    authority.record_publication(lease, lifecycle_authority)
    authority.record_submission(lease, lifecycle_authority)
    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="generation differs from live row",
    ):
        authority.record_teardown_completion(
            lease,
            lifecycle_authority,
            metadata_slot_generation=b"x" * METADATA_GENERATION_BYTES,
            native_dram_handle_generation=17,
            descriptor_digest=DESCRIPTOR_DIGEST,
            evidence_digest=EVIDENCE_DIGEST,
        )
    assert authority.snapshot(lease).state is PackedAuxiliaryAllocationState.SUBMITTED

    authority.record_teardown_completion(
        lease,
        lifecycle_authority,
        metadata_slot_generation=prepared.metadata_slot_generation,
        native_dram_handle_generation=17,
        descriptor_digest=DESCRIPTOR_DIGEST,
        evidence_digest=EVIDENCE_DIGEST,
    )
    authority.commit_to_request_after_teardown(lease, lifecycle_authority)
    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="requires the consumer authority",
    ):
        authority.release_after_consumption(lease, object())
    assert (
        authority.snapshot(lease).state
        is PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST
    )
    assert len(allocation.released) == 0
    authority.release_after_consumption(lease, consumer_authority)


def test_snapshot_fault_releases_provisional_authority_and_logs_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanly release a provisional row after a non-runtime snapshot fault."""

    allocation = RecordingSlotAllocation()
    allocation.snapshot_failure = KeyError("snapshot callback fault")
    authority, _, _ = _new_authority()
    caplog.set_level(
        logging.ERROR,
        logger="sglang.srt.disaggregation.common.packed_auxiliary_allocation",
    )

    with pytest.raises(PackedAuxiliaryAllocationError, match="acquisition failed"):
        authority.acquire(allocation)

    assert allocation.calls == ["allocate", "snapshot", "release"]
    assert len(allocation.released) == 1
    assert authority.provisional_quarantine_count() == 0
    assert "Traceback (most recent call last)" in caplog.text
    assert "KeyError: 'snapshot callback fault'" in caplog.text


def test_failed_acquisition_cleanup_retains_process_lifetime_quarantine(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retain authority when snapshot, cleanup, and quarantine callbacks fault."""

    allocation = RecordingSlotAllocation()
    allocation.snapshot_failure = LookupError("snapshot callback fault")
    allocation.release_failure = OSError("cleanup release callback fault")
    allocation.quarantine_failure = KeyError("cleanup quarantine callback fault")
    authority, _, _ = _new_authority()
    caplog.set_level(
        logging.ERROR,
        logger="sglang.srt.disaggregation.common.packed_auxiliary_allocation",
    )

    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="reservation was quarantined",
    ):
        authority.acquire(allocation)

    assert allocation.calls == [
        "allocate",
        "snapshot",
        "release",
        "quarantine",
    ]
    assert authority.provisional_quarantine_count() == 1
    assert "LookupError: snapshot callback fault" in caplog.text
    assert "OSError: cleanup release callback fault" in caplog.text
    assert "KeyError: 'cleanup quarantine callback fault'" in caplog.text


def test_quarantine_is_monotonic_when_allocator_callback_faults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Publish quarantine before callback failure and preserve its first reason."""

    allocation = RecordingSlotAllocation()
    authority, _, _ = _new_authority()
    lease = authority.acquire(allocation)
    allocation.quarantine_failure = LookupError("quarantine callback fault")
    caplog.set_level(
        logging.ERROR,
        logger="sglang.srt.disaggregation.common.packed_auxiliary_allocation",
    )

    with pytest.raises(LookupError, match="quarantine callback fault"):
        authority.quarantine(lease, "first ambiguity")

    quarantined = authority.snapshot(lease)
    assert quarantined.state is PackedAuxiliaryAllocationState.QUARANTINED
    assert quarantined.failure_reason == "first ambiguity"
    allocation.quarantine_failure = None
    authority.quarantine(lease, "later ambiguity")
    assert authority.snapshot(lease).failure_reason == "first ambiguity"
    assert allocation.calls.count("quarantine") == 1
    assert "Traceback (most recent call last)" in caplog.text
    assert "LookupError: quarantine callback fault" in caplog.text


def test_cancel_release_fault_becomes_permanent_quarantine() -> None:
    """Treat a non-runtime cancellation release fault as ambiguous ownership."""

    allocation = RecordingSlotAllocation()
    authority, _, _ = _new_authority()
    lease = authority.acquire(allocation)
    allocation.release_failure = OSError("cancel release callback fault")

    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="metadata cancellation release failed",
    ):
        authority.cancel_unpublished(lease)

    quarantined = authority.snapshot(lease)
    assert quarantined.state is PackedAuxiliaryAllocationState.QUARANTINED
    assert quarantined.failure_reason == "metadata cancellation release failed"
    assert len(allocation.quarantined) == 1
    with pytest.raises(PackedAuxiliaryAllocationError, match="cannot release"):
        authority.cancel_unpublished(lease)
    authority.quarantine(lease, "later ambiguity")
    assert authority.snapshot(lease).failure_reason == quarantined.failure_reason
    assert allocation.calls.count("quarantine") == 1


def test_consumption_release_fault_becomes_permanent_quarantine() -> None:
    """Retain a request-consumed row after an ambiguous allocator release."""

    allocation = RecordingSlotAllocation()
    authority, lifecycle_authority, consumer_authority = _new_authority()
    lease = authority.acquire(allocation)
    _drive_to_request_commit(authority, lease, lifecycle_authority)
    allocation.release_failure = KeyError("consumption release callback fault")

    with pytest.raises(
        PackedAuxiliaryAllocationError,
        match="metadata consumption release failed",
    ):
        authority.release_after_consumption(lease, consumer_authority)

    quarantined = authority.snapshot(lease)
    assert quarantined.state is PackedAuxiliaryAllocationState.QUARANTINED
    assert quarantined.failure_reason == "metadata consumption release failed"
    assert len(allocation.quarantined) == 1
    with pytest.raises(PackedAuxiliaryAllocationError, match="cannot retire"):
        authority.retire_terminal(lease)


def test_quarantine_cannot_interleave_with_ambiguous_release() -> None:
    """Serialize quarantine behind release and invoke allocator quarantine once."""

    allocation = RecordingSlotAllocation()
    authority, _, _ = _new_authority()
    lease = authority.acquire(allocation)
    allocation.block_release = True
    allocation.release_failure = OSError("blocked release callback fault")
    quarantine_started = threading.Event()

    def quarantine_concurrently() -> None:
        """Attempt quarantine while the allocator release callback is blocked."""

        quarantine_started.set()
        authority.quarantine(lease, "concurrent ambiguity")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        release_future = executor.submit(authority.cancel_unpublished, lease)
        assert allocation.release_entered.wait(timeout=2)
        quarantine_future = executor.submit(quarantine_concurrently)
        assert quarantine_started.wait(timeout=2)
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                quarantine_future.result(timeout=0.1)
            assert allocation.calls == ["allocate", "snapshot", "release"]
        finally:
            allocation.release_continue.set()
        with pytest.raises(PackedAuxiliaryAllocationError):
            release_future.result(timeout=2)
        quarantine_future.result(timeout=2)

    assert allocation.calls == [
        "allocate",
        "snapshot",
        "release",
        "quarantine",
    ]
    snapshot = authority.snapshot(lease)
    assert snapshot.state is PackedAuxiliaryAllocationState.QUARANTINED
    assert snapshot.failure_reason == "metadata cancellation release failed"


def test_allocator_callbacks_are_serialized_during_acquisition() -> None:
    """Do not enter allocation while another allocator callback owns the lock."""

    allocation = RecordingSlotAllocation()
    authority, _, _ = _new_authority()
    first_lease = authority.acquire(allocation)
    allocation.block_release = True
    acquisition_started = threading.Event()

    def acquire_concurrently() -> PackedAuxiliaryAllocationLease:
        """Attempt a second allocation while release owns callback authority."""

        acquisition_started.set()
        return authority.acquire(allocation)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        release_future = executor.submit(authority.cancel_unpublished, first_lease)
        assert allocation.release_entered.wait(timeout=2)
        acquisition_future = executor.submit(acquire_concurrently)
        assert acquisition_started.wait(timeout=2)
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                acquisition_future.result(timeout=0.1)
            assert allocation.calls == ["allocate", "snapshot", "release"]
        finally:
            allocation.release_continue.set()
        release_future.result(timeout=2)
        second_lease = acquisition_future.result(timeout=2)

    assert allocation.calls == [
        "allocate",
        "snapshot",
        "release",
        "allocate",
        "snapshot",
    ]
    authority.cancel_unpublished(second_lease)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
