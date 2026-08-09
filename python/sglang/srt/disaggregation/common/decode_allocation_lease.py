import dataclasses
import enum
import hashlib
import secrets
import threading
from typing import Protocol

import torch

from sglang.srt.disaggregation.common.staging_layout import (
    StagingWriterId,
    source_tp_ranks_for_destination,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)
from sglang.srt.mem_cache.allocation_pin import (
    AllocationPin,
    AllocationPinSnapshot,
    PinnableAllocation,
    RequestSlotPinSnapshot,
)

DECODE_ALLOCATION_LEASE_ID_BYTES = 16
DECODE_ALLOCATION_DIGEST_BYTES = 32

_LEASE_CONSTRUCTION_SEAL = object()
_ABORT_CONSTRUCTION_SEAL = object()


class DecodeAllocationLeaseError(RuntimeError):
    """Decode allocation migration-lease invariant violation."""


class DecodeAllocationLifecycleUnavailable(DecodeAllocationLeaseError):
    """No trusted transport lifecycle is bound to the allocation authority."""


class DecodeAllocationQuarantinedError(DecodeAllocationLeaseError):
    """A migration ambiguity permanently retains the local allocation."""


class DecodeAllocationComponent(enum.StrEnum):
    """Canonical local allocation components in ordered collective phases."""

    FULL = "full"
    SWA = "swa"
    MAMBA = "mamba"


DECODE_ALLOCATION_COMPONENT_ORDER = (
    DecodeAllocationComponent.FULL,
    DecodeAllocationComponent.SWA,
    DecodeAllocationComponent.MAMBA,
)


class DecodeAllocationLeaseState(enum.StrEnum):
    """Process-local migration ownership state."""

    PREPARED = "prepared"
    PUBLISHED = "published"
    SUBMITTED = "submitted"
    WRITERS_COMPLETED = "writers_completed"
    SCATTER_COMPLETED = "scatter_completed"
    TEARDOWN_COMPLETED = "teardown_completed"
    COMMITTED_TO_REQUEST = "committed_to_request"
    ROLLED_BACK_TO_REQUEST = "rolled_back_to_request"
    ABORT_AUTHORIZED = "abort_authorized"
    QUARANTINED = "quarantined"


def _validate_writer_id(writer_id: StagingWriterId) -> None:
    """Validate one canonical source writer identity.

    :param writer_id: Candidate source writer.
    """

    if type(writer_id) is not StagingWriterId:
        raise TypeError("writer must be StagingWriterId")
    fields = (
        writer_id.transfer_source_rank,
        writer_id.source_attn_tp_rank,
        writer_id.source_pp_rank,
        writer_id.source_cp_rank,
    )
    if any(type(value) is not int or value < 0 for value in fields):
        raise ValueError("writer ranks must be non-negative integers")


@dataclasses.dataclass(frozen=True)
class DecodeWriterManifest:
    """Exact ordered source writer membership for one destination rank.

    :ivar source_tp_size: Source attention tensor-parallel width.
    :ivar destination_tp_size: Destination attention tensor-parallel width.
    :ivar destination_tp_rank: Destination attention tensor-parallel rank.
    :ivar writers: Canonically ordered exact source writer identities.
    """

    source_tp_size: int
    destination_tp_size: int
    destination_tp_rank: int
    writers: tuple[StagingWriterId, ...]

    def __post_init__(self) -> None:
        """Own and validate exact destination-local writer membership."""

        if (
            type(self.source_tp_size) is not int
            or self.source_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES
        ):
            raise ValueError(
                f"source_tp_size must be one of {SUPPORTED_PACKED_SOURCE_TP_SIZES}"
            )
        if self.source_tp_size < self.destination_tp_size:
            raise ValueError(
                "decode allocation manifests do not support destination TP "
                "wider than source TP"
            )
        expected_source_tp_ranks = source_tp_ranks_for_destination(
            self.source_tp_size,
            self.destination_tp_size,
            self.destination_tp_rank,
        )
        writers = tuple(self.writers)
        object.__setattr__(self, "writers", writers)
        writers_per_destination = len(expected_source_tp_ranks)
        if len(writers) != writers_per_destination:
            raise ValueError(
                "writer count must equal source_tp_size / destination_tp_size: "
                f"{len(writers)} != {writers_per_destination}"
            )
        for writer in writers:
            _validate_writer_id(writer)
        if len(set(writers)) != len(writers):
            raise ValueError("writer manifest contains duplicate identities")
        if writers != tuple(sorted(writers)):
            raise ValueError("writer manifest must use canonical ordering")
        source_tp_ranks = tuple(writer.source_attn_tp_rank for writer in writers)
        if source_tp_ranks != expected_source_tp_ranks:
            raise ValueError(
                "writer manifest does not match destination topology: "
                f"expected source ranks {expected_source_tp_ranks}, "
                f"got {source_tp_ranks}"
            )

    @classmethod
    def for_tensor_parallel(
        cls,
        source_tp_size: int,
        destination_tp_size: int = 1,
        destination_tp_rank: int = 0,
    ) -> "DecodeWriterManifest":
        """Build the canonical destination-local TP writer manifest.

        :param source_tp_size: Supported packed source TP width.
        :param destination_tp_size: Destination attention TP width.
        :param destination_tp_rank: Destination attention TP rank.
        :returns: Exact ordered destination-local writer manifest.
        """

        if (
            type(source_tp_size) is not int
            or source_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES
        ):
            raise ValueError(
                f"source_tp_size must be one of {SUPPORTED_PACKED_SOURCE_TP_SIZES}"
            )
        if source_tp_size < destination_tp_size:
            raise ValueError(
                "decode allocation manifests do not support destination TP "
                "wider than source TP"
            )
        source_tp_ranks = source_tp_ranks_for_destination(
            source_tp_size,
            destination_tp_size,
            destination_tp_rank,
        )
        return cls(
            source_tp_size=source_tp_size,
            destination_tp_size=destination_tp_size,
            destination_tp_rank=destination_tp_rank,
            writers=tuple(
                StagingWriterId(
                    transfer_source_rank=rank,
                    source_attn_tp_rank=rank,
                    source_pp_rank=0,
                    source_cp_rank=0,
                )
                for rank in source_tp_ranks
            ),
        )

    @property
    def digest(self) -> bytes:
        """Return a stable digest over exact ordered writer membership.

        :returns: SHA-256 manifest digest.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.decode-allocation.writer-manifest.v1")
        for value in (
            self.source_tp_size,
            self.destination_tp_size,
            self.destination_tp_rank,
        ):
            digest.update(value.to_bytes(4, "big"))
        for writer in self.writers:
            for value in (
                writer.transfer_source_rank,
                writer.source_attn_tp_rank,
                writer.source_pp_rank,
                writer.source_cp_rank,
            ):
                digest.update(value.to_bytes(4, "big"))
        return digest.digest()


@dataclasses.dataclass(frozen=True)
class DecodeAllocationComponentClaim:
    """Engine-derived local allocation range to pin transactionally.

    Zero-work components must set ``logical_length`` to zero and omit both the
    allocator and indices. Nonzero components must provide exact local
    allocator-visible token or slot IDs.

    :ivar component: Canonical FULL, SWA, or Mamba phase.
    :ivar logical_start: Request-local logical start position.
    :ivar logical_length: Exact logical work length.
    :ivar allocator: Process-local allocator owning the supplied IDs.
    :ivar indices: Exact allocator-visible token or slot IDs.
    """

    component: DecodeAllocationComponent
    logical_start: int
    logical_length: int
    allocator: PinnableAllocation | None
    indices: torch.Tensor | None

    def __post_init__(self) -> None:
        """Validate one ephemeral component claim."""

        if type(self.component) is not DecodeAllocationComponent:
            raise TypeError("component must be DecodeAllocationComponent")
        if type(self.logical_start) is not int or self.logical_start < 0:
            raise ValueError("logical_start must be a non-negative integer")
        if type(self.logical_length) is not int or self.logical_length < 0:
            raise ValueError("logical_length must be a non-negative integer")
        if self.logical_length == 0:
            if self.allocator is not None or self.indices is not None:
                raise ValueError("zero-work component must omit allocator and indices")
            return
        if self.allocator is None or self.indices is None:
            raise ValueError("nonzero component must provide allocator and indices")
        if not isinstance(self.indices, torch.Tensor):
            raise TypeError("component indices must be a torch.Tensor")
        if self.indices.ndim != 1:
            raise ValueError("component indices must be one-dimensional")
        if self.indices.numel() != self.logical_length:
            raise ValueError(
                "component logical length differs from index count: "
                f"{self.logical_length} != {self.indices.numel()}"
            )


@dataclasses.dataclass(frozen=True)
class DecodeAllocationComponentReceipt:
    """Immutable allocator-derived mapping for one local component.

    :ivar component: Canonical component phase.
    :ivar logical_start: Request-local logical start position.
    :ivar logical_length: Exact logical work length.
    :ivar page_size: Tokens represented by one allocator page.
    :ivar virtual_pages: Allocator-visible pages in request-logical order.
    :ivar physical_pages: Corresponding immutable physical pages in the same
        request-logical order.
    """

    component: DecodeAllocationComponent
    logical_start: int
    logical_length: int
    page_size: int
    virtual_pages: tuple[int, ...]
    physical_pages: tuple[int, ...]

    @property
    def zero_work(self) -> bool:
        """Return whether the component requires ordered no-work participation.

        :returns: Whether this component owns no local allocation.
        """

        return self.logical_length == 0


@dataclasses.dataclass(frozen=True)
class DecodeWriterParticipation:
    """One writer's ordered participation in one component phase.

    :ivar writer_id: Exact manifest writer.
    :ivar generation_order: Fixed component phase ordinal.
    :ivar component: Canonical component phase.
    :ivar zero_work: Whether the rank must acknowledge without transfer work.
    """

    writer_id: StagingWriterId
    generation_order: int
    component: DecodeAllocationComponent
    zero_work: bool


@dataclasses.dataclass(frozen=True)
class DecodeAllocationLeaseSnapshot:
    """Immutable process-local allocation receipt.

    :ivar lease_id: Random process-local migration lease identity.
    :ivar request_slot: Exact decode request-pool slot.
    :ivar request_generation: Allocator-derived slot reuse generation.
    :ivar writer_manifest: Exact expected packed source writers.
    :ivar components: FULL, SWA, and Mamba mappings in fixed order.
    :ivar writer_participation: Every writer's fixed ordered component phases.
    :ivar allocation_digest: Stable generation and mapping digest.
    :ivar state: Current migration ownership state.
    :ivar failure_reason: First quarantine reason, if any.
    """

    lease_id: bytes
    request_slot: int
    request_generation: int
    writer_manifest: DecodeWriterManifest
    components: tuple[DecodeAllocationComponentReceipt, ...]
    writer_participation: tuple[DecodeWriterParticipation, ...]
    allocation_digest: bytes
    state: DecodeAllocationLeaseState
    failure_reason: str | None


class DecodeAllocationLease:
    """Opaque process-local handle for one migration-pinned allocation."""

    __slots__ = ("_authority_nonce", "_token")

    _authority_nonce: object
    _token: object

    def __init__(
        self,
        authority_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one authority-owned allocation lease.

        :param authority_nonce: Exact issuing authority identity.
        :param token: Authority-private record key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _LEASE_CONSTRUCTION_SEAL:
            raise TypeError("decode allocation leases are authority owned")
        self._authority_nonce = authority_nonce
        self._token = token


class DecodeAllocationAbortPermit:
    """Opaque proof that migration is quiescent before canonical abort cleanup."""

    __slots__ = ("_authority_nonce", "_lease_token")

    _authority_nonce: object
    _lease_token: object

    def __init__(
        self,
        authority_nonce: object,
        lease_token: object,
        construction_seal: object,
    ) -> None:
        """Construct one authority-owned abort permit.

        :param authority_nonce: Exact issuing authority identity.
        :param lease_token: Exact allocation lease record key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _ABORT_CONSTRUCTION_SEAL:
            raise TypeError("decode allocation abort permits are authority owned")
        self._authority_nonce = authority_nonce
        self._lease_token = lease_token


class RequestSlotAllocation(Protocol):
    """Typed request-pool surface consumed by the allocation authority."""

    def acquire_request_slot_pin(
        self,
        slot: int,
        expected_generation: int,
        owner: object,
    ) -> AllocationPin:
        """Pin one exact request slot and generation.

        :param slot: Exact request-pool slot.
        :param expected_generation: Engine-observed reuse generation.
        :param owner: Exact pin authority.
        :returns: Opaque request-slot pin.
        """

        ...

    def request_slot_pin_snapshot(
        self,
        pin: AllocationPin,
    ) -> RequestSlotPinSnapshot:
        """Resolve the pinned request slot identity.

        :param pin: Exact request-slot pin.
        :returns: Immutable slot and generation.
        """

        ...

    def release_request_slot_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Release one request-slot migration pin.

        :param pin: Exact request-slot pin.
        :param owner: Exact pin authority.
        """

        ...

    def quarantine_request_slot_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Permanently retain one ambiguous request slot.

        :param pin: Exact request-slot pin.
        :param owner: Exact pin authority.
        """

        ...


@dataclasses.dataclass(frozen=True)
class _HeldAllocationPin:
    """One process-local allocator and its exact migration pin."""

    allocator: PinnableAllocation
    pin: AllocationPin


@dataclasses.dataclass
class _DecodeAllocationRecord:
    """Private mutable state for one decode allocation lease."""

    lease: DecodeAllocationLease
    lease_id: bytes
    request_pool: RequestSlotAllocation
    request_pin: AllocationPin
    request_snapshot: RequestSlotPinSnapshot
    writer_manifest: DecodeWriterManifest
    component_receipts: tuple[DecodeAllocationComponentReceipt, ...]
    component_pins: tuple[_HeldAllocationPin, ...]
    writer_participation: tuple[DecodeWriterParticipation, ...]
    allocation_digest: bytes
    state: DecodeAllocationLeaseState = DecodeAllocationLeaseState.PREPARED
    failure_reason: str | None = None
    abort_permit: DecodeAllocationAbortPermit | None = None
    abort_permit_consumed: bool = False


class DecodeAllocationLeaseAuthority:
    """Process-local authority for transactional migration allocation pins."""

    _authority_nonce: object
    _lifecycle_authority: object | None
    _lock: threading.Lock
    _pin_owner: object
    _records: dict[object, _DecodeAllocationRecord]

    def __init__(self, lifecycle_authority: object | None = None) -> None:
        """Initialize an empty process-local lease authority.

        ``lifecycle_authority`` is the exact trusted transport coordinator
        permitted to advance post-submission states. Omitting it hard-gates
        transport promotion while retaining pre-submission rollback tests.

        :param lifecycle_authority: Exact trusted transport lifecycle owner.
        """

        self._authority_nonce = object()
        self._lifecycle_authority = lifecycle_authority
        self._lock = threading.Lock()
        self._pin_owner = object()
        self._records = {}

    def acquire(
        self,
        *,
        request_pool: RequestSlotAllocation,
        request_slot: int,
        expected_request_generation: int,
        writer_manifest: DecodeWriterManifest,
        component_claims: tuple[DecodeAllocationComponentClaim, ...],
    ) -> DecodeAllocationLease:
        """Pin one complete local decode allocation transactionally.

        A receipt is published only after request-slot, FULL, SWA, and Mamba
        phases have all pinned successfully. Zero-work phases still appear in
        the immutable receipt and every writer's ordered participation list.

        :param request_pool: Process-local owner of the decode request slot.
        :param request_slot: Exact request-pool slot.
        :param expected_request_generation: Engine-observed slot generation.
        :param writer_manifest: Exact expected packed source writers.
        :param component_claims: FULL, SWA, and Mamba claims in fixed order.
        :returns: Opaque process-local migration lease.
        """

        if type(writer_manifest) is not DecodeWriterManifest:
            raise TypeError("writer_manifest must be DecodeWriterManifest")
        claims = tuple(component_claims)
        components = tuple(claim.component for claim in claims)
        if components != DECODE_ALLOCATION_COMPONENT_ORDER:
            raise ValueError(
                "component claims must contain FULL, SWA, and Mamba exactly once "
                "in canonical order"
            )

        request_pin: AllocationPin | None = None
        held_pins: list[_HeldAllocationPin] = []
        completed = False
        try:
            request_pin = request_pool.acquire_request_slot_pin(
                request_slot,
                expected_request_generation,
                self._pin_owner,
            )
            request_snapshot = request_pool.request_slot_pin_snapshot(request_pin)
            if request_snapshot.generation != expected_request_generation:
                raise DecodeAllocationLeaseError(
                    "request generation changed during pin acquisition"
                )

            component_receipts: list[DecodeAllocationComponentReceipt] = []
            for claim in claims:
                if claim.logical_length == 0:
                    component_receipts.append(
                        DecodeAllocationComponentReceipt(
                            component=claim.component,
                            logical_start=claim.logical_start,
                            logical_length=0,
                            page_size=1,
                            virtual_pages=(),
                            physical_pages=(),
                        )
                    )
                    continue
                allocator = claim.allocator
                indices = claim.indices
                if allocator is None or indices is None:
                    raise RuntimeError("validated nonzero claim lost its allocator")
                pin = allocator.acquire_allocation_pin(indices, self._pin_owner)
                held_pins.append(_HeldAllocationPin(allocator=allocator, pin=pin))
                pin_snapshot = allocator.allocation_pin_snapshot(pin)
                component_receipts.append(self._component_receipt(claim, pin_snapshot))

            owned_receipts = tuple(component_receipts)
            writer_participation = tuple(
                DecodeWriterParticipation(
                    writer_id=writer,
                    generation_order=phase,
                    component=receipt.component,
                    zero_work=receipt.zero_work,
                )
                for phase, receipt in enumerate(owned_receipts)
                for writer in writer_manifest.writers
            )
            lease_id = secrets.token_bytes(DECODE_ALLOCATION_LEASE_ID_BYTES)
            allocation_digest = self._allocation_digest(
                lease_id=lease_id,
                request_snapshot=request_snapshot,
                writer_manifest=writer_manifest,
                component_receipts=owned_receipts,
            )
            token = object()
            lease = DecodeAllocationLease(
                self._authority_nonce,
                token,
                _LEASE_CONSTRUCTION_SEAL,
            )
            record = _DecodeAllocationRecord(
                lease=lease,
                lease_id=lease_id,
                request_pool=request_pool,
                request_pin=request_pin,
                request_snapshot=request_snapshot,
                writer_manifest=writer_manifest,
                component_receipts=owned_receipts,
                component_pins=tuple(held_pins),
                writer_participation=writer_participation,
                allocation_digest=allocation_digest,
            )
            with self._lock:
                self._records[token] = record
            completed = True
            return lease
        finally:
            if not completed:
                for held_pin in reversed(held_pins):
                    held_pin.allocator.release_allocation_pin(
                        held_pin.pin,
                        self._pin_owner,
                    )
                if request_pin is not None:
                    request_pool.release_request_slot_pin(
                        request_pin,
                        self._pin_owner,
                    )

    def rollback_to_request(self, lease: DecodeAllocationLease) -> None:
        """Remove pre-submission migration pins without freeing request storage.

        :param lease: Exact prepared allocation lease.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.state is not DecodeAllocationLeaseState.PREPARED:
                raise DecodeAllocationLeaseError(
                    f"rollback is invalid in state {record.state.value}"
                )
            self._release_pins_locked(record)
            record.state = DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST

    def record_publication(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Make peer-visible transfer authorization irreversible locally.

        Once metadata carrying this allocation generation is visible, a source
        writer may act on it asynchronously. The allocation can no longer use
        the pre-publication rollback path, even if no exact native submission
        has been observed yet.

        :param lease: Exact prepared allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                DecodeAllocationLeaseState.PREPARED,
                DecodeAllocationLeaseState.PUBLISHED,
            )

    def record_submission(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Bind trusted exact-handle submission to the local allocation.

        :param lease: Exact published allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                DecodeAllocationLeaseState.PUBLISHED,
                DecodeAllocationLeaseState.SUBMITTED,
            )

    def record_writer_completion(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Bind trusted all-writer terminal completion to the allocation.

        :param lease: Exact submitted allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                DecodeAllocationLeaseState.SUBMITTED,
                DecodeAllocationLeaseState.WRITERS_COMPLETED,
            )

    def record_scatter_completion(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Bind trusted terminal destination scatter to the allocation.

        :param lease: Exact writer-complete allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            self._transition_locked(
                record,
                DecodeAllocationLeaseState.WRITERS_COMPLETED,
                DecodeAllocationLeaseState.SCATTER_COMPLETED,
            )

    def record_teardown_completion(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
        *,
        request_generation: int,
        writer_manifest_digest: bytes,
        allocation_digest: bytes,
    ) -> None:
        """Bind authenticated all-writer teardown to this exact allocation.

        The trusted lifecycle must present the same request generation, writer
        manifest, and physical allocation digest that were fixed before
        submission. A teardown acknowledgement for another route or reused
        request slot cannot release these local pins.

        :param lease: Exact scatter-complete allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        :param request_generation: Authenticated request-slot generation.
        :param writer_manifest_digest: Authenticated exact writer membership.
        :param allocation_digest: Authenticated local allocation identity.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            if type(request_generation) is not int:
                raise TypeError("request_generation must be an integer")
            if request_generation != record.request_snapshot.generation:
                raise DecodeAllocationLeaseError(
                    "teardown request generation differs from allocation"
                )
            if bytes(writer_manifest_digest) != record.writer_manifest.digest:
                raise DecodeAllocationLeaseError(
                    "teardown writer manifest differs from allocation"
                )
            if bytes(allocation_digest) != record.allocation_digest:
                raise DecodeAllocationLeaseError(
                    "teardown allocation digest differs from allocation"
                )
            self._transition_locked(
                record,
                DecodeAllocationLeaseState.SCATTER_COMPLETED,
                DecodeAllocationLeaseState.TEARDOWN_COMPLETED,
            )

    def commit_to_request_after_teardown(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Return migration ownership to the still-live decode request.

        This removes migration pins only. It never frees allocator pages or the
        request slot; normal decode ownership continues through the same request
        generation.

        :param lease: Exact teardown-complete allocation lease.
        :param lifecycle_authority: Exact all-writer teardown authority.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            if record.state is not DecodeAllocationLeaseState.TEARDOWN_COMPLETED:
                raise DecodeAllocationLeaseError(
                    "request commit requires authenticated all-writer teardown"
                )
            self._release_pins_locked(record)
            record.state = DecodeAllocationLeaseState.COMMITTED_TO_REQUEST

    def commit_legacy_to_request_after_consumption(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> None:
        """Return a consumed legacy staging allocation to its live request.

        Legacy staging has no prepared-transaction teardown exchange. Its
        collective successful poll is authoritative only after every source
        writer completed, every destination scatter event completed, and the
        metadata gate converged. This transition is therefore restricted to a
        published lease presented by the exact bound transport authority.
        Prepared-transaction allocations must use authenticated teardown.

        This removes migration pins only. It never frees allocator pages or the
        request slot.

        :param lease: Exact consumed legacy allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            if record.state is not DecodeAllocationLeaseState.PUBLISHED:
                raise DecodeAllocationLeaseError(
                    "legacy request commit requires a consumed published transfer"
                )
            self._release_pins_locked(record)
            record.state = DecodeAllocationLeaseState.COMMITTED_TO_REQUEST

    def authorize_legacy_abort_after_terminal_failure(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> DecodeAllocationAbortPermit:
        """Permit cleanup after a terminal legacy transfer failure.

        Legacy staging does not exchange packed teardown receipts. A collective
        failed poll may authorize cleanup only after the bound transport owner
        has established that the receiver is terminal and all local staging
        work is quiescent. The caller must consume the returned permit before
        freeing request or KV storage.

        :param lease: Exact failed legacy allocation lease.
        :param lifecycle_authority: Exact configured transport lifecycle owner.
        :returns: One-shot abort-cleanup permit.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            if record.state is not DecodeAllocationLeaseState.PUBLISHED:
                raise DecodeAllocationLeaseError(
                    "legacy terminal failure abort requires a published transfer"
                )
            return self._authorize_abort_locked(record)

    def authorize_pre_submission_abort(
        self,
        lease: DecodeAllocationLease,
    ) -> DecodeAllocationAbortPermit:
        """Permit canonical abort cleanup before metadata or native submission.

        :param lease: Exact prepared allocation lease.
        :returns: One-shot abort-cleanup permit.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.state is not DecodeAllocationLeaseState.PREPARED:
                raise DecodeAllocationLeaseError(
                    f"pre-submission abort is invalid in state {record.state.value}"
                )
            return self._authorize_abort_locked(record)

    def authorize_abort_after_teardown(
        self,
        lease: DecodeAllocationLease,
        lifecycle_authority: object,
    ) -> DecodeAllocationAbortPermit:
        """Permit canonical engine cleanup after exact migration quiescence.

        The authority does not free anything itself. The returned opaque permit
        is the seam the decode engine must consume before its normal abort path
        releases request and KV ownership.

        :param lease: Exact migration lease.
        :param lifecycle_authority: Exact terminal transport authority.
        :returns: Opaque abort-cleanup permit.
        """

        with self._lock:
            record = self._validate_locked(lease)
            self._require_lifecycle_authority(lifecycle_authority)
            if record.state is not DecodeAllocationLeaseState.TEARDOWN_COMPLETED:
                raise DecodeAllocationLeaseError(
                    "post-submission abort requires authenticated all-writer teardown"
                )
            return self._authorize_abort_locked(record)

    def consume_abort_permit(
        self,
        lease: DecodeAllocationLease,
        permit: DecodeAllocationAbortPermit,
    ) -> None:
        """Consume one exact abort permit before canonical engine cleanup.

        :param lease: Exact allocation lease.
        :param permit: Exact authority-owned abort permit.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if type(permit) is not DecodeAllocationAbortPermit:
                raise TypeError("permit must be DecodeAllocationAbortPermit")
            if (
                permit._authority_nonce is not self._authority_nonce
                or permit._lease_token is not lease._token
            ):
                raise DecodeAllocationLeaseError(
                    "abort permit belongs to another allocation"
                )
            if record.state is not DecodeAllocationLeaseState.ABORT_AUTHORIZED:
                raise DecodeAllocationLeaseError(
                    f"abort permit is invalid in state {record.state.value}"
                )
            if record.abort_permit is not permit:
                raise DecodeAllocationLeaseError(
                    "abort permit is not the live permit for this allocation"
                )
            if record.abort_permit_consumed:
                raise DecodeAllocationLeaseError("abort permit was already consumed")
            record.abort_permit_consumed = True

    def retire_terminal(self, lease: DecodeAllocationLease) -> None:
        """Forget terminal metadata while preserving stale-handle rejection.

        Quarantined records remain process-lifetime owners and cannot retire.
        Abort records retire only after their one-shot cleanup permit is
        consumed.

        :param lease: Exact terminal allocation lease.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.state not in (
                DecodeAllocationLeaseState.COMMITTED_TO_REQUEST,
                DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST,
                DecodeAllocationLeaseState.ABORT_AUTHORIZED,
            ):
                raise DecodeAllocationLeaseError(
                    f"allocation cannot retire in state {record.state.value}"
                )
            if (
                record.state is DecodeAllocationLeaseState.ABORT_AUTHORIZED
                and not record.abort_permit_consumed
            ):
                raise DecodeAllocationLeaseError(
                    "allocation cannot retire before abort permit consumption"
                )
            del self._records[lease._token]

    def quarantine(
        self,
        lease: DecodeAllocationLease,
        reason: str,
    ) -> None:
        """Permanently retain every local allocation owner after ambiguity.

        Quarantine is intentionally callable without release authority because
        retaining too much is safe; releasing too early is not.

        :param lease: Exact allocation lease.
        :param reason: First stable ambiguity reason.
        """

        if len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        with self._lock:
            record = self._validate_locked(lease)
            if record.state in (
                DecodeAllocationLeaseState.COMMITTED_TO_REQUEST,
                DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST,
                DecodeAllocationLeaseState.ABORT_AUTHORIZED,
            ):
                raise DecodeAllocationLeaseError(
                    f"quarantine is invalid in state {record.state.value}"
                )
            if record.state is DecodeAllocationLeaseState.QUARANTINED:
                return
            for held_pin in record.component_pins:
                held_pin.allocator.quarantine_allocation_pin(
                    held_pin.pin,
                    self._pin_owner,
                )
            record.request_pool.quarantine_request_slot_pin(
                record.request_pin,
                self._pin_owner,
            )
            record.failure_reason = reason
            record.state = DecodeAllocationLeaseState.QUARANTINED

    def snapshot(
        self,
        lease: DecodeAllocationLease,
    ) -> DecodeAllocationLeaseSnapshot:
        """Return an immutable diagnostic allocation receipt.

        :param lease: Exact authority-owned allocation lease.
        :returns: Immutable generation and physical-mapping receipt.
        """

        with self._lock:
            record = self._validate_locked(lease)
            return DecodeAllocationLeaseSnapshot(
                lease_id=record.lease_id,
                request_slot=record.request_snapshot.slot,
                request_generation=record.request_snapshot.generation,
                writer_manifest=record.writer_manifest,
                components=record.component_receipts,
                writer_participation=record.writer_participation,
                allocation_digest=record.allocation_digest,
                state=record.state,
                failure_reason=record.failure_reason,
            )

    def _release_pins_locked(self, record: _DecodeAllocationRecord) -> None:
        """Remove migration pins while retaining normal request allocation.

        :param record: Exact authority-owned allocation record.
        """

        record.request_pool.request_slot_pin_snapshot(record.request_pin)
        for held_pin in record.component_pins:
            held_pin.allocator.allocation_pin_snapshot(held_pin.pin)
        for held_pin in reversed(record.component_pins):
            held_pin.allocator.release_allocation_pin(
                held_pin.pin,
                self._pin_owner,
            )
        record.request_pool.release_request_slot_pin(
            record.request_pin,
            self._pin_owner,
        )

    def _authorize_abort_locked(
        self,
        record: _DecodeAllocationRecord,
    ) -> DecodeAllocationAbortPermit:
        """Remove migration pins and issue one exact cleanup permit.

        :param record: Exact authority-owned allocation record.
        :returns: One-shot abort-cleanup permit.
        """

        self._release_pins_locked(record)
        permit = DecodeAllocationAbortPermit(
            self._authority_nonce,
            record.lease._token,
            _ABORT_CONSTRUCTION_SEAL,
        )
        record.abort_permit = permit
        record.abort_permit_consumed = False
        record.state = DecodeAllocationLeaseState.ABORT_AUTHORIZED
        return permit

    def _validate_locked(
        self,
        lease: DecodeAllocationLease,
    ) -> _DecodeAllocationRecord:
        """Resolve one exact lease while the authority lock is held.

        :param lease: Candidate authority-owned lease.
        :returns: Private allocation record.
        """

        if type(lease) is not DecodeAllocationLease:
            raise TypeError("lease must be DecodeAllocationLease")
        if lease._authority_nonce is not self._authority_nonce:
            raise DecodeAllocationLeaseError(
                "allocation lease belongs to another authority"
            )
        record = self._records.get(lease._token)
        if record is None or record.lease is not lease:
            raise DecodeAllocationLeaseError("allocation lease is not registered")
        return record

    def _require_lifecycle_authority(self, authority: object) -> None:
        """Require the exact configured post-submission lifecycle owner.

        :param authority: Candidate lifecycle authority.
        """

        if self._lifecycle_authority is None:
            raise DecodeAllocationLifecycleUnavailable(
                "trusted transport lifecycle is not bound"
            )
        if authority is not self._lifecycle_authority:
            raise DecodeAllocationLeaseError(
                "post-submission transition requires exact lifecycle authority"
            )

    @staticmethod
    def _transition_locked(
        record: _DecodeAllocationRecord,
        expected: DecodeAllocationLeaseState,
        target: DecodeAllocationLeaseState,
    ) -> None:
        """Advance one exact lifecycle edge.

        :param record: Exact authority-owned allocation record.
        :param expected: Required current state.
        :param target: Next state.
        """

        if record.state is not expected:
            raise DecodeAllocationLeaseError(
                f"transition to {target.value} is invalid in state "
                f"{record.state.value}"
            )
        record.state = target

    @staticmethod
    def _component_receipt(
        claim: DecodeAllocationComponentClaim,
        pin_snapshot: AllocationPinSnapshot,
    ) -> DecodeAllocationComponentReceipt:
        """Build one immutable allocator-derived component receipt.

        :param claim: Validated engine component claim.
        :param pin_snapshot: Exact allocator-owned virtual/physical mapping.
        :returns: Immutable component receipt.
        """

        if len(pin_snapshot.virtual_pages) != len(pin_snapshot.physical_pages):
            raise DecodeAllocationLeaseError(
                "allocator pin virtual and physical page counts differ"
            )
        indices = claim.indices
        if indices is None:
            raise RuntimeError("nonzero component claim lost its indices")
        page_size = pin_snapshot.page_size
        ordered_virtual_pages = tuple(
            dict.fromkeys(
                int(index) // page_size
                for index in indices.detach().to(dtype=torch.int64).cpu().tolist()
            )
        )
        if set(ordered_virtual_pages) != set(pin_snapshot.virtual_pages):
            raise DecodeAllocationLeaseError(
                "component claim pages differ from the allocator pin snapshot"
            )
        physical_by_virtual = dict(
            zip(
                pin_snapshot.virtual_pages,
                pin_snapshot.physical_pages,
                strict=True,
            )
        )
        ordered_physical_pages = tuple(
            physical_by_virtual[page] for page in ordered_virtual_pages
        )
        return DecodeAllocationComponentReceipt(
            component=claim.component,
            logical_start=claim.logical_start,
            logical_length=claim.logical_length,
            page_size=page_size,
            virtual_pages=ordered_virtual_pages,
            physical_pages=ordered_physical_pages,
        )

    @staticmethod
    def _allocation_digest(
        *,
        lease_id: bytes,
        request_snapshot: RequestSlotPinSnapshot,
        writer_manifest: DecodeWriterManifest,
        component_receipts: tuple[DecodeAllocationComponentReceipt, ...],
    ) -> bytes:
        """Digest exact local generation, manifest, and physical mappings.

        :param lease_id: Random process-local lease identity.
        :param request_snapshot: Exact request slot and generation.
        :param writer_manifest: Exact ordered source writer membership.
        :param component_receipts: Exact local component mappings.
        :returns: SHA-256 allocation digest.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.decode-allocation.lease.v1")
        digest.update(lease_id)
        digest.update(request_snapshot.slot.to_bytes(8, "big"))
        digest.update(request_snapshot.generation.to_bytes(8, "big"))
        digest.update(writer_manifest.digest)
        for receipt in component_receipts:
            digest.update(receipt.component.value.encode("ascii"))
            digest.update(receipt.logical_start.to_bytes(8, "big"))
            digest.update(receipt.logical_length.to_bytes(8, "big"))
            digest.update(receipt.page_size.to_bytes(8, "big"))
            digest.update(len(receipt.virtual_pages).to_bytes(8, "big"))
            for virtual_page, physical_page in zip(
                receipt.virtual_pages,
                receipt.physical_pages,
                strict=True,
            ):
                digest.update(virtual_page.to_bytes(8, "big"))
                digest.update(physical_page.to_bytes(8, "big"))
        allocation_digest = digest.digest()
        if len(allocation_digest) != DECODE_ALLOCATION_DIGEST_BYTES:
            raise RuntimeError("allocation digest has an unexpected length")
        return allocation_digest
