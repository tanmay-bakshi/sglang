import dataclasses
import enum
import logging
import threading
import traceback
from collections.abc import Callable
from itertools import pairwise
from typing import Never, Protocol, TypeVar

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_DIGEST_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
)

logger = logging.getLogger(__name__)

_CallbackResult = TypeVar("_CallbackResult")


def _invoke_allocator_callback(
    callback: Callable[[], _CallbackResult],
    context: str,
) -> _CallbackResult:
    """Invoke one allocator callback with complete exception diagnostics.

    :param callback: Exact callback closure.
    :param context: Stable operation context.
    :returns: Callback result.
    """

    try:
        return callback()
    except Exception:
        logger.error("%s:\n%s", context, traceback.format_exc())
        raise


class PackedAuxiliaryAllocationError(RuntimeError):
    """Metadata-slot lease ownership or lifecycle invariant violation."""


class PackedAuxiliaryAllocationState(enum.StrEnum):
    """Process-local ownership state of one exact destination metadata row."""

    ACQUIRING = "acquiring"
    PREPARED = "prepared"
    PUBLISHED = "published"
    SUBMITTED = "submitted"
    TEARDOWN_COMPLETED = "teardown_completed"
    COMMITTED_TO_REQUEST = "committed_to_request"
    RELEASING = "releasing"
    RELEASED = "released"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


@dataclasses.dataclass(frozen=True)
class PackedAuxiliarySlotReservationSnapshot:
    """Allocator-authored identity and geometry of one reserved metadata row.

    :ivar metadata_buffer_index: Exact allocator row index.
    :ivar metadata_slot_generation: Row reuse generation preventing ABA replay.
    :ivar destination_segments: Ordered exact addresses and item lengths.
    """

    metadata_buffer_index: int
    metadata_slot_generation: bytes
    destination_segments: tuple[PackedAuxiliaryDestinationSegment, ...]

    def __post_init__(self) -> None:
        """Own and validate one immutable reserved-row snapshot."""

        if (
            type(self.metadata_buffer_index) is not int
            or self.metadata_buffer_index < 0
            or self.metadata_buffer_index >= (1 << 64)
        ):
            raise ValueError("metadata_buffer_index must be a uint64")
        if type(self.metadata_slot_generation) is not bytes:
            raise TypeError("metadata_slot_generation must be bytes")
        if len(self.metadata_slot_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "metadata_slot_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes"
            )
        segments = tuple(self.destination_segments)
        object.__setattr__(self, "destination_segments", segments)
        if len(segments) == 0:
            raise ValueError("metadata slot must contain destination segments")
        if any(
            type(segment) is not PackedAuxiliaryDestinationSegment
            for segment in segments
        ):
            raise TypeError(
                "metadata slot segments must be PackedAuxiliaryDestinationSegment"
            )
        if len(set(segments)) != len(segments):
            raise ValueError("metadata slot contains duplicate destination segments")
        segments_by_address = tuple(
            sorted(segments, key=lambda segment: segment.address)
        )
        for previous, current in pairwise(segments_by_address):
            if current.address < previous.address + previous.item_length:
                raise ValueError("metadata slot destination segments overlap")


class PackedAuxiliarySlotAllocation(Protocol):
    """Generation-bearing metadata row allocator consumed by the authority.

    Callbacks are serialized under the authority lock. They must not call back
    into :class:`PackedAuxiliaryAllocationLeaseAuthority` or wait for network
    or device work. Allocation failures must leave allocator ownership
    unchanged because no reservation has been returned. A release failure is
    ambiguous and permanently quarantines the exact reservation.
    """

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Allocate one metadata row directly to the supplied owner.

        Failure must be transactional and retain allocator ownership.

        :param owner: Process-local reservation owner.
        :returns: Opaque allocator-minted reservation.
        """

        ...

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Resolve exact identity and registered geometry for a reservation.

        :param reservation: Exact allocator-minted reservation.
        :returns: Immutable live row snapshot.
        """

        ...

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return one terminal metadata row to allocator ownership.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact process-local reservation owner.
        """

        ...

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Retain one ambiguous metadata row against reuse.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact process-local reservation owner.
        """

        ...


class PackedAuxiliaryAllocationLease:
    """Opaque process-local authority over one reserved metadata row."""

    __slots__ = ("_authority_nonce", "_token")

    _authority_nonce: object
    _token: object

    def __init__(
        self,
        authority_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one authority-owned lease.

        :param authority_nonce: Exact issuing authority identity.
        :param token: Authority-private record identity.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _LEASE_CONSTRUCTION_SEAL:
            raise TypeError("auxiliary allocation leases are authority owned")
        self._authority_nonce = authority_nonce
        self._token = token


@dataclasses.dataclass(frozen=True)
class PackedAuxiliaryAllocationLeaseSnapshot:
    """Immutable ownership receipt for one reserved metadata row.

    :ivar metadata_buffer_index: Exact allocator row index.
    :ivar metadata_slot_generation: Row reuse generation preventing replay.
    :ivar destination_segments: Ordered exact row destinations.
    :ivar state: Current process-local lease state.
    :ivar native_dram_handle_generation: Retired source handle after teardown.
    :ivar descriptor_digest: Exact terminal descriptor digest after teardown.
    :ivar evidence_digest: Exact terminal evidence digest after teardown.
    :ivar failure_reason: First quarantine reason, if any.
    """

    metadata_buffer_index: int
    metadata_slot_generation: bytes
    destination_segments: tuple[PackedAuxiliaryDestinationSegment, ...]
    state: PackedAuxiliaryAllocationState
    native_dram_handle_generation: int | None
    descriptor_digest: bytes | None
    evidence_digest: bytes | None
    failure_reason: str | None


@dataclasses.dataclass
class _PackedAuxiliaryAllocationRecord:
    """Mutable state owned by one auxiliary allocation authority."""

    lease: PackedAuxiliaryAllocationLease
    allocation: PackedAuxiliarySlotAllocation
    reservation: object
    slot_snapshot: PackedAuxiliarySlotReservationSnapshot | None = None
    state: PackedAuxiliaryAllocationState = PackedAuxiliaryAllocationState.ACQUIRING
    native_dram_handle_generation: int | None = None
    descriptor_digest: bytes | None = None
    evidence_digest: bytes | None = None
    failure_reason: str | None = None


_LEASE_CONSTRUCTION_SEAL = object()


def _validate_digest(value: bytes, label: str) -> None:
    """Validate one exact SHA-256 digest.

    :param value: Candidate digest.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_REQUEST_DIGEST_BYTES:
        raise ValueError(f"{label} must contain {PACKED_REQUEST_DIGEST_BYTES} bytes")


class PackedAuxiliaryAllocationLeaseAuthority:
    """Own exact destination metadata rows through scheduler consumption."""

    _authority_nonce: object
    _consumer_authority: object
    _lifecycle_authority: object
    _lock: threading.RLock
    _reservation_owner: object
    _records: dict[object, _PackedAuxiliaryAllocationRecord]

    def __init__(
        self,
        lifecycle_authority: object,
        consumer_authority: object,
    ) -> None:
        """Initialize one process-local metadata ownership authority.

        :param lifecycle_authority: Exact trusted transport lifecycle owner.
        :param consumer_authority: Exact scheduler metadata-copy owner.
        """

        if lifecycle_authority is None:
            raise ValueError("lifecycle_authority must not be None")
        if consumer_authority is None:
            raise ValueError("consumer_authority must not be None")
        self._authority_nonce = object()
        self._consumer_authority = consumer_authority
        self._lifecycle_authority = lifecycle_authority
        self._lock = threading.RLock()
        self._reservation_owner = object()
        self._records = {}

    def acquire(
        self,
        allocation: PackedAuxiliarySlotAllocation,
    ) -> PackedAuxiliaryAllocationLease:
        """Allocate and retain one allocator-authored metadata row.

        :param allocation: Generation-bearing metadata row allocator.
        :returns: Opaque live metadata allocation lease.
        """

        if allocation is None:
            raise ValueError("allocation must not be None")
        with self._lock:
            reservation = _invoke_allocator_callback(
                lambda: allocation.allocate_packed_auxiliary_slot(
                    self._reservation_owner
                ),
                "Packed auxiliary metadata allocation failed",
            )
            if reservation is None:
                raise PackedAuxiliaryAllocationError(
                    "metadata allocator returned no reservation"
                )
            token = object()
            lease = PackedAuxiliaryAllocationLease(
                self._authority_nonce,
                token,
                _LEASE_CONSTRUCTION_SEAL,
            )
            record = _PackedAuxiliaryAllocationRecord(
                lease=lease,
                allocation=allocation,
                reservation=reservation,
            )
            self._records[token] = record
            record = self._validate_locked(lease)
            if record.state is not PackedAuxiliaryAllocationState.ACQUIRING:
                raise PackedAuxiliaryAllocationError(
                    "metadata lease changed during acquisition"
                )
            try:
                slot_snapshot = _invoke_allocator_callback(
                    lambda: allocation.packed_auxiliary_slot_reservation_snapshot(
                        reservation
                    ),
                    "Packed auxiliary metadata snapshot failed",
                )
                if type(slot_snapshot) is not PackedAuxiliarySlotReservationSnapshot:
                    raise TypeError(
                        "metadata allocator snapshot must be "
                        "PackedAuxiliarySlotReservationSnapshot"
                    )
            except Exception as error:
                logger.error(
                    "Packed auxiliary metadata acquisition validation failed:\n%s",
                    traceback.format_exc(),
                )
                self._cleanup_failed_acquisition(lease, error)
            record.slot_snapshot = slot_snapshot
            record.state = PackedAuxiliaryAllocationState.PREPARED
        return lease

    def record_publication(
        self,
        lease: PackedAuxiliaryAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Cross the irreversible metadata publication boundary.

        :param lease: Exact prepared metadata lease.
        :param lifecycle_authority: Exact trusted lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                PackedAuxiliaryAllocationState.PREPARED,
                PackedAuxiliaryAllocationState.PUBLISHED,
            )

    def record_submission(
        self,
        lease: PackedAuxiliaryAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Record proof that native DRAM submission became externally visible.

        :param lease: Exact published metadata lease.
        :param lifecycle_authority: Exact trusted lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                PackedAuxiliaryAllocationState.PUBLISHED,
                PackedAuxiliaryAllocationState.SUBMITTED,
            )

    def record_teardown_completion(
        self,
        lease: PackedAuxiliaryAllocationLease,
        lifecycle_authority: object,
        *,
        metadata_slot_generation: bytes,
        native_dram_handle_generation: int,
        descriptor_digest: bytes,
        evidence_digest: bytes,
    ) -> None:
        """Bind exact source-handle retirement to the live destination row.

        This transition does not release the destination row. The scheduler
        must first copy its contents into the request under consumer authority.

        :param lease: Exact submitted metadata lease.
        :param lifecycle_authority: Exact trusted lifecycle owner.
        :param metadata_slot_generation: Authenticated row reuse generation.
        :param native_dram_handle_generation: Exact retired source handle.
        :param descriptor_digest: Exact terminal descriptor digest.
        :param evidence_digest: Exact terminal runtime evidence digest.
        """

        if type(metadata_slot_generation) is not bytes:
            raise TypeError("metadata_slot_generation must be bytes")
        if (
            type(native_dram_handle_generation) is not int
            or native_dram_handle_generation <= 0
            or native_dram_handle_generation >= (1 << 64)
        ):
            raise ValueError("native_dram_handle_generation must be a positive uint64")
        _validate_digest(descriptor_digest, "descriptor_digest")
        _validate_digest(evidence_digest, "evidence_digest")
        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            slot_snapshot = self._require_slot_snapshot(record)
            if metadata_slot_generation != slot_snapshot.metadata_slot_generation:
                raise PackedAuxiliaryAllocationError(
                    "teardown metadata generation differs from live row"
                )
            self._transition_locked(
                record,
                PackedAuxiliaryAllocationState.SUBMITTED,
                PackedAuxiliaryAllocationState.TEARDOWN_COMPLETED,
            )
            record.native_dram_handle_generation = native_dram_handle_generation
            record.descriptor_digest = descriptor_digest
            record.evidence_digest = evidence_digest

    def commit_to_request_after_teardown(
        self,
        lease: PackedAuxiliaryAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Hand the retained destination row to scheduler request consumption.

        :param lease: Exact teardown-complete metadata lease.
        :param lifecycle_authority: Exact trusted lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                PackedAuxiliaryAllocationState.TEARDOWN_COMPLETED,
                PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST,
            )

    def cancel_unpublished(
        self,
        lease: PackedAuxiliaryAllocationLease,
    ) -> None:
        """Return one never-published row to its allocator.

        :param lease: Exact prepared metadata lease.
        """

        self._release_reservation(
            lease,
            expected_state=PackedAuxiliaryAllocationState.PREPARED,
            terminal_state=PackedAuxiliaryAllocationState.CANCELLED,
            failure_reason="metadata cancellation release failed",
        )

    def release_after_consumption(
        self,
        lease: PackedAuxiliaryAllocationLease,
        consumer_authority: object,
    ) -> None:
        """Release one row only after exact scheduler metadata consumption.

        :param lease: Exact request-committed metadata lease.
        :param consumer_authority: Exact scheduler metadata-copy owner.
        """

        self._require_consumer_authority(consumer_authority)
        self._release_reservation(
            lease,
            expected_state=PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST,
            terminal_state=PackedAuxiliaryAllocationState.RELEASED,
            failure_reason="metadata consumption release failed",
        )

    def quarantine(
        self,
        lease: PackedAuxiliaryAllocationLease,
        reason: str,
    ) -> None:
        """Retain one ambiguous metadata row against process-lifetime reuse.

        The quarantined state and first reason are published before the fallible
        allocator callback. Callback failure never restores a releasable state.

        :param lease: Exact live metadata lease.
        :param reason: First stable ambiguity reason.
        """

        if len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        with self._lock:
            record = self._validate_locked(lease)
            if record.state is PackedAuxiliaryAllocationState.QUARANTINED:
                return
            if record.state in (
                PackedAuxiliaryAllocationState.RELEASED,
                PackedAuxiliaryAllocationState.CANCELLED,
            ):
                raise PackedAuxiliaryAllocationError(
                    f"quarantine is invalid in state {record.state.value}"
                )
            record.state = PackedAuxiliaryAllocationState.QUARANTINED
            if record.failure_reason is None:
                record.failure_reason = reason
            allocation = record.allocation
            reservation = record.reservation
            _invoke_allocator_callback(
                lambda: allocation.quarantine_packed_auxiliary_slot(
                    reservation,
                    self._reservation_owner,
                ),
                "Packed auxiliary metadata quarantine callback failed",
            )

    def retire_terminal(
        self,
        lease: PackedAuxiliaryAllocationLease,
    ) -> None:
        """Forget one safely released process-local metadata lease.

        :param lease: Exact released or cancelled metadata lease.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.state not in (
                PackedAuxiliaryAllocationState.RELEASED,
                PackedAuxiliaryAllocationState.CANCELLED,
            ):
                raise PackedAuxiliaryAllocationError(
                    f"metadata lease cannot retire in state {record.state.value}"
                )
            del self._records[lease._token]

    def snapshot(
        self,
        lease: PackedAuxiliaryAllocationLease,
    ) -> PackedAuxiliaryAllocationLeaseSnapshot:
        """Return an immutable snapshot of exact retained row ownership.

        :param lease: Exact authority-owned metadata lease.
        :returns: Immutable lease snapshot.
        """

        with self._lock:
            record = self._validate_locked(lease)
            slot = self._require_slot_snapshot(record)
            return PackedAuxiliaryAllocationLeaseSnapshot(
                metadata_buffer_index=slot.metadata_buffer_index,
                metadata_slot_generation=slot.metadata_slot_generation,
                destination_segments=slot.destination_segments,
                state=record.state,
                native_dram_handle_generation=record.native_dram_handle_generation,
                descriptor_digest=record.descriptor_digest,
                evidence_digest=record.evidence_digest,
                failure_reason=record.failure_reason,
            )

    def provisional_quarantine_count(self) -> int:
        """Return the number of ambiguous acquisitions retained without a lease.

        :returns: Process-lifetime provisional quarantine count.
        """

        with self._lock:
            return sum(
                record.slot_snapshot is None
                and record.state is PackedAuxiliaryAllocationState.QUARANTINED
                for record in self._records.values()
            )

    def _cleanup_failed_acquisition(
        self,
        lease: PackedAuxiliaryAllocationLease,
        acquisition_error: BaseException,
    ) -> Never:
        """Release a failed provisional acquisition or retain its exact authority.

        :param lease: Exact provisional metadata lease.
        :param acquisition_error: Original snapshot or validation failure.
        :raises PackedAuxiliaryAllocationError: Always, with retained ambiguity.
        """

        try:
            self._release_reservation(
                lease,
                expected_state=PackedAuxiliaryAllocationState.ACQUIRING,
                terminal_state=PackedAuxiliaryAllocationState.CANCELLED,
                failure_reason="metadata acquisition cleanup release failed",
            )
        except (RuntimeError, TypeError, ValueError) as cleanup_error:
            raise PackedAuxiliaryAllocationError(
                "metadata allocation acquisition failed and its reservation "
                "was quarantined"
            ) from cleanup_error
        with self._lock:
            self._records.pop(lease._token)
        raise PackedAuxiliaryAllocationError(
            "metadata allocation acquisition failed"
        ) from acquisition_error

    def _release_reservation(
        self,
        lease: PackedAuxiliaryAllocationLease,
        *,
        expected_state: PackedAuxiliaryAllocationState,
        terminal_state: PackedAuxiliaryAllocationState,
        failure_reason: str,
    ) -> None:
        """Release one reservation while serializing allocator ownership.

        :param lease: Exact authority-owned metadata lease.
        :param expected_state: Required state before release.
        :param terminal_state: State recorded after successful release.
        :param failure_reason: Stable quarantine reason after callback failure.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.state is not expected_state:
                raise PackedAuxiliaryAllocationError(
                    "metadata lease cannot release from state "
                    f"{record.state.value}; expected {expected_state.value}"
                )
            record.state = PackedAuxiliaryAllocationState.RELEASING
            allocation = record.allocation
            reservation = record.reservation
            try:
                _invoke_allocator_callback(
                    lambda: allocation.release_packed_auxiliary_slot(
                        reservation,
                        self._reservation_owner,
                    ),
                    "Packed auxiliary metadata release callback failed",
                )
            except Exception as error:
                record.state = PackedAuxiliaryAllocationState.QUARANTINED
                if record.failure_reason is None:
                    record.failure_reason = failure_reason
                try:
                    _invoke_allocator_callback(
                        lambda: allocation.quarantine_packed_auxiliary_slot(
                            reservation,
                            self._reservation_owner,
                        ),
                        "Packed auxiliary metadata release quarantine failed",
                    )
                except Exception:
                    pass
                raise PackedAuxiliaryAllocationError(failure_reason) from error
            if record.state is not PackedAuxiliaryAllocationState.RELEASING:
                raise PackedAuxiliaryAllocationError(
                    "metadata lease changed during release"
                )
            record.state = terminal_state

    def _validate_locked(
        self,
        lease: PackedAuxiliaryAllocationLease,
    ) -> _PackedAuxiliaryAllocationRecord:
        """Resolve one exact live authority-owned lease.

        :param lease: Candidate metadata lease.
        :returns: Exact private record.
        """

        if type(lease) is not PackedAuxiliaryAllocationLease:
            raise TypeError("lease must be PackedAuxiliaryAllocationLease")
        if lease._authority_nonce is not self._authority_nonce:
            raise PackedAuxiliaryAllocationError(
                "metadata lease belongs to another authority"
            )
        record = self._records.get(lease._token)
        if record is None or record.lease is not lease:
            raise PackedAuxiliaryAllocationError("metadata lease is not registered")
        return record

    @staticmethod
    def _require_slot_snapshot(
        record: _PackedAuxiliaryAllocationRecord,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Return exact allocator geometry after acquisition completed.

        :param record: Exact private metadata record.
        :returns: Immutable allocator-authored row snapshot.
        """

        snapshot = record.slot_snapshot
        if snapshot is None:
            raise PackedAuxiliaryAllocationError(
                "metadata reservation has no valid snapshot"
            )
        return snapshot

    def _require_lifecycle_authority(self, lifecycle_authority: object) -> None:
        """Require the exact configured transport lifecycle owner.

        :param lifecycle_authority: Candidate transition authority.
        """

        if lifecycle_authority is not self._lifecycle_authority:
            raise PackedAuxiliaryAllocationError(
                "metadata transition requires the lifecycle authority"
            )

    def _require_consumer_authority(self, consumer_authority: object) -> None:
        """Require the exact configured scheduler metadata-copy owner.

        :param consumer_authority: Candidate destination consumer.
        """

        if consumer_authority is not self._consumer_authority:
            raise PackedAuxiliaryAllocationError(
                "metadata release requires the consumer authority"
            )

    @staticmethod
    def _transition_locked(
        record: _PackedAuxiliaryAllocationRecord,
        expected: PackedAuxiliaryAllocationState,
        target: PackedAuxiliaryAllocationState,
    ) -> None:
        """Perform one exact metadata lease state transition.

        :param record: Exact private metadata record.
        :param expected: Required current state.
        :param target: New state.
        """

        if record.state is not expected:
            raise PackedAuxiliaryAllocationError(
                f"metadata lease is in state {record.state.value}; "
                f"expected {expected.value}"
            )
        record.state = target
