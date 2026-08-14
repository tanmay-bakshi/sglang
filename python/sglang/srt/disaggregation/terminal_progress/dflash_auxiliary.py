import collections
import dataclasses
import enum
import logging
import secrets
import threading
import traceback
from collections.abc import Callable
from typing import Protocol

import torch

from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_DFLASH_BOUNDARY_BYTES,
    PACKED_REQUEST_DIGEST_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryPlan,
    PackedDFlashBoundaryMetadata,
    PackedDFlashBoundaryOutcome,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
)
from sglang.srt.disaggregation.terminal_progress.output_projection import (
    TerminalGatewayResultSlot,
)

logger = logging.getLogger(__name__)

DFLASH_BOUNDARY_MEMORY_KIND = "VRAM"
DFLASH_BOUNDARY_ROW_SHAPE: tuple[int, ...] = (1,)
DFLASH_BOUNDARY_ROW_DTYPE = torch.int64
DFLASH_BOUNDARY_ROW_BYTES = PACKED_DFLASH_BOUNDARY_BYTES
_GENERATION_PREFIX_BYTES = PACKED_REQUEST_GENERATION_BYTES // 2
_GENERATION_COUNTER_BYTES = PACKED_REQUEST_GENERATION_BYTES - _GENERATION_PREFIX_BYTES
_MAX_GENERATION_COUNTER = (1 << (_GENERATION_COUNTER_BYTES * 8)) - 1


class DFlashBoundaryMemoryAgent(Protocol):
    """NIXL memory-registration surface required by boundary row pools."""

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Register one process-lifetime device allocation.

        :param addresses: Exact NIXL registration descriptors.
        :param memory_kind: Required NIXL memory kind.
        :returns: Opaque process-lifetime registration handle.
        """

        ...


class DFlashBoundaryDescriptorAgent(Protocol):
    """NIXL descriptor surface used for one boundary-token transfer."""

    def get_xfer_descs(
        self,
        addresses: list[tuple[int, int, int]],
        memory_kind: str,
    ) -> object:
        """Project exact registered row addresses to native descriptors.

        :param addresses: Row address, length, and CUDA-device tuples.
        :param memory_kind: Required NIXL memory kind.
        :returns: Opaque native descriptor collection.
        """

        ...


class DFlashBoundaryNativeTransferAgent(
    DFlashBoundaryDescriptorAgent,
    Protocol,
):
    """NIXL source-transfer surface consumed by the boundary owner."""

    name: str

    def initialize_xfer(
        self,
        operation: str,
        source_descriptors: object,
        destination_descriptors: object,
        remote_handle: object,
        notification: bytes,
    ) -> object:
        """Initialize one exact device-to-device boundary write.

        :param operation: Required NIXL operation name.
        :param source_descriptors: Exact local VRAM descriptors.
        :param destination_descriptors: Exact remote VRAM descriptors.
        :param remote_handle: Exact enrolled decoder agent handle.
        :param notification: Required empty notification payload.
        :returns: Opaque initialized native transfer handle.
        """

        ...


class DFlashBoundaryNativeEnum(Protocol):
    """Native enum value required by completion attestation."""

    name: str


class DFlashBoundaryNativeSegment(Protocol):
    """One native completion segment required by exact attestation."""

    index: int
    length: int
    localAddress: int
    posted: bool
    remoteAddress: int


class DFlashBoundaryNativeReceipt(Protocol):
    """Native take-once success receipt required by boundary settlement."""

    backend: str
    completionClaimed: bool
    descriptorDigest: str
    error: str
    evidenceDigest: str
    generation: int
    handleIdentity: int
    localAgent: str
    localMemoryType: DFlashBoundaryNativeEnum
    operation: DFlashBoundaryNativeEnum
    remoteAgent: str
    remoteMemoryType: DFlashBoundaryNativeEnum
    segments: tuple[DFlashBoundaryNativeSegment, ...]
    state: DFlashBoundaryNativeEnum
    status: DFlashBoundaryNativeEnum
    submissionSealed: bool


class DFlashBoundaryDirectTransfer(Protocol):
    """Exact transfer generation armed through the native terminal owner."""

    binding_digest: bytes
    generation: int
    handle_identity: int


class DFlashBoundaryDirectOwner(Protocol):
    """Direct native-owner adapter required by the boundary transport."""

    def arm_transfer(
        self,
        handle: object,
        binding_digest: bytes,
    ) -> DFlashBoundaryDirectTransfer:
        """Arm terminal delivery before a transfer can be posted.

        :param handle: Initialized exact native transfer handle.
        :param binding_digest: Exact source lifecycle binding.
        :returns: Retained native transfer-generation authority.
        """

        ...

    def post_transfer(
        self,
        transfer: DFlashBoundaryDirectTransfer,
        post: Callable[[object], object],
    ) -> object:
        """Post through the terminal-before-return race boundary.

        :param transfer: Exact armed transfer generation.
        :param post: One-shot native post operation.
        :returns: Native post result.
        """

        ...

    def settle_success(
        self,
        transfer: DFlashBoundaryDirectTransfer,
        action: NativeTerminalOwnerAction,
    ) -> DFlashBoundaryNativeReceipt:
        """Consume exact success authority under an owner action.

        :param transfer: Exact terminal transfer generation.
        :param action: Matching source outcome action.
        :returns: Take-once native completion receipt.
        """

        ...

    def settle_failure(
        self,
        transfer: DFlashBoundaryDirectTransfer,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Settle terminal failure without manufacturing success.

        :param transfer: Exact failed transfer generation.
        :param action: Matching quarantine or process-fatal action.
        """

        ...

    def release_transfer(self, transfer: DFlashBoundaryDirectTransfer) -> None:
        """Release one settled native handle under teardown authority.

        :param transfer: Exact settled transfer generation.
        """

        ...


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryRegistration:
    """Process-lifetime device allocation backing stable boundary rows.

    :ivar base_address: Exact first byte of the CUDA allocation.
    :ivar allocation_length: Complete registered allocation byte length.
    :ivar device_index: Exact CUDA device owning the allocation.
    :ivar row_capacity: Number of fixed eight-byte rows.
    """

    base_address: int
    allocation_length: int
    device_index: int
    row_capacity: int

    def __post_init__(self) -> None:
        """Validate exact address, device, and row geometry."""

        if type(self.base_address) is not int or self.base_address <= 0:
            raise ValueError("DFlash boundary base address must be positive")
        if type(self.row_capacity) is not int or self.row_capacity <= 0:
            raise ValueError("DFlash boundary row capacity must be positive")
        expected_length = self.row_capacity * DFLASH_BOUNDARY_ROW_BYTES
        if (
            type(self.allocation_length) is not int
            or self.allocation_length != expected_length
        ):
            raise ValueError(
                "DFlash boundary allocation length differs from row geometry"
            )
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("DFlash boundary CUDA device must be non-negative")

    def segment_for_row(self, row_index: int) -> PackedAuxiliaryDestinationSegment:
        """Project one exact row into packed destination geometry.

        :param row_index: Stable row index.
        :returns: Exact registered destination segment.
        """

        self._validate_row_index(row_index)
        return PackedAuxiliaryDestinationSegment(
            address=self.base_address + row_index * DFLASH_BOUNDARY_ROW_BYTES,
            item_length=DFLASH_BOUNDARY_ROW_BYTES,
        )

    def transfer_request_for_row(self, row_index: int) -> tuple[int, int, int]:
        """Project one exact row into a native VRAM request.

        :param row_index: Stable row index.
        :returns: Address, byte length, and CUDA-device tuple.
        """

        segment = self.segment_for_row(row_index)
        return (segment.address, segment.item_length, self.device_index)

    def _validate_row_index(self, row_index: int) -> None:
        if (
            type(row_index) is not int
            or row_index < 0
            or row_index >= self.row_capacity
        ):
            raise ValueError("DFlash boundary row index is out of range")


class DFlashBoundaryNixlRegistration:
    """Strong process-lifetime ownership of one NIXL VRAM registration."""

    _receipt: DFlashBoundaryRegistration
    _native_handle: object

    def __init__(
        self,
        agent: DFlashBoundaryMemoryAgent,
        receipt: DFlashBoundaryRegistration,
    ) -> None:
        """Register one stable device allocation exactly once.

        :param agent: Exact NIXL memory-registration authority.
        :param receipt: Stable allocation geometry.
        """

        if type(receipt) is not DFlashBoundaryRegistration:
            raise TypeError("receipt must be DFlashBoundaryRegistration")
        native_handle = agent.register_memory(
            [
                (
                    receipt.base_address,
                    receipt.allocation_length,
                    receipt.device_index,
                    "",
                )
            ],
            DFLASH_BOUNDARY_MEMORY_KIND,
        )
        if native_handle is None or (
            isinstance(native_handle, list | tuple) and len(native_handle) == 0
        ):
            raise RuntimeError("NIXL returned no DFlash boundary registration")
        self._receipt = receipt
        self._native_handle = native_handle

    @property
    def receipt(self) -> DFlashBoundaryRegistration:
        """Return immutable registered allocation geometry.

        :returns: Exact process-lifetime registration receipt.
        """

        return self._receipt


@dataclasses.dataclass(frozen=True, slots=True)
class _DFlashBoundaryReservation:
    """Opaque allocator identity for one exact row generation."""

    token: object


@dataclasses.dataclass(slots=True)
class _DFlashBoundaryReservationRecord:
    """Mutable allocator authority for one exact row generation."""

    reservation: _DFlashBoundaryReservation
    owner: object
    row_index: int
    generation: bytes


class DFlashBoundaryRowAllocator:
    """Generation-bound allocator over one registered boundary row pool."""

    _registration: DFlashBoundaryNixlRegistration
    _generation_prefix: bytes
    _generation_counter: int
    _free_rows: collections.deque[int]
    _records: dict[object, _DFlashBoundaryReservationRecord]
    _quarantined_rows: set[int]
    _lock: threading.Lock

    def __init__(self, registration: DFlashBoundaryNixlRegistration) -> None:
        """Construct one clean allocator over registered VRAM.

        :param registration: Process-lifetime boundary registration.
        """

        if type(registration) is not DFlashBoundaryNixlRegistration:
            raise TypeError("registration must be DFlashBoundaryNixlRegistration")
        self._registration = registration
        self._generation_prefix = secrets.token_bytes(_GENERATION_PREFIX_BYTES)
        self._generation_counter = 0
        self._free_rows = collections.deque(range(registration.receipt.row_capacity))
        self._records = {}
        self._quarantined_rows = set()
        self._lock = threading.Lock()

    @property
    def registration(self) -> DFlashBoundaryRegistration:
        """Return exact registered allocation geometry.

        :returns: Process-lifetime registration receipt.
        """

        return self._registration.receipt

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Lease one free boundary row to an exact owner.

        :param owner: Process-local reservation authority.
        :returns: Opaque exact-generation reservation.
        """

        if owner is None:
            raise ValueError("DFlash boundary reservation owner must not be None")
        with self._lock:
            if len(self._free_rows) == 0:
                raise RuntimeError("DFlash boundary row pool is exhausted")
            row_index = self._free_rows.popleft()
            generation = self._next_generation_locked()
            token = object()
            reservation = _DFlashBoundaryReservation(token)
            self._records[token] = _DFlashBoundaryReservationRecord(
                reservation=reservation,
                owner=owner,
                row_index=row_index,
                generation=generation,
            )
            return reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Resolve exact live row identity and registered geometry.

        :param reservation: Exact allocator-minted reservation.
        :returns: Immutable live row snapshot.
        """

        with self._lock:
            record = self._require_record_locked(reservation)
            return self._snapshot_locked(record)

    def require_snapshot(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> int:
        """Prove one immutable snapshot still names its exact live row.

        :param snapshot: Candidate allocator-authored row snapshot.
        :returns: Exact live physical row index.
        """

        if type(snapshot) is not PackedAuxiliarySlotReservationSnapshot:
            raise TypeError("snapshot must be PackedAuxiliarySlotReservationSnapshot")
        with self._lock:
            matches = tuple(
                record
                for record in self._records.values()
                if record.row_index == snapshot.metadata_buffer_index
            )
            if len(matches) != 1:
                raise RuntimeError("DFlash boundary snapshot is stale")
            record = matches[0]
            if record.generation != snapshot.metadata_slot_generation:
                raise RuntimeError("DFlash boundary snapshot generation is stale")
            if self._snapshot_locked(record) != snapshot:
                raise RuntimeError("DFlash boundary snapshot geometry was altered")
            return record.row_index

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return one terminal row to allocator ownership.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact reservation authority.
        """

        with self._lock:
            record = self._require_record_locked(reservation)
            self._require_owner_locked(record, owner)
            del self._records[record.reservation.token]
            self._free_rows.append(record.row_index)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Permanently remove one ambiguous row from reuse.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact reservation authority.
        """

        with self._lock:
            record = self._require_record_locked(reservation)
            self._require_owner_locked(record, owner)
            del self._records[record.reservation.token]
            self._quarantined_rows.add(record.row_index)

    def registered_row(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> "DFlashBoundaryRegisteredRow":
        """Return a live registered-row proof for one exact generation.

        :param snapshot: Exact allocator-authored live snapshot.
        :returns: Strong registered-row proof.
        """

        self.require_snapshot(snapshot)
        return DFlashBoundaryRegisteredRow(
            registration=self.registration,
            reservation=snapshot,
        )

    def inventory(self) -> tuple[int, int, int]:
        """Return free, active, and quarantined row counts.

        :returns: Conservation-complete row population.
        """

        with self._lock:
            inventory = (
                len(self._free_rows),
                len(self._records),
                len(self._quarantined_rows),
            )
            if sum(inventory) != self.registration.row_capacity:
                raise RuntimeError("DFlash boundary row inventory does not conserve")
            return inventory

    def _next_generation_locked(self) -> bytes:
        if self._generation_counter >= _MAX_GENERATION_COUNTER:
            raise RuntimeError("DFlash boundary row generation space is exhausted")
        self._generation_counter += 1
        return self._generation_prefix + self._generation_counter.to_bytes(
            _GENERATION_COUNTER_BYTES,
            "big",
        )

    def _require_record_locked(
        self,
        reservation: object,
    ) -> _DFlashBoundaryReservationRecord:
        if type(reservation) is not _DFlashBoundaryReservation:
            raise TypeError("reservation must be a DFlash boundary reservation")
        record = self._records.get(reservation.token)
        if record is None or record.reservation is not reservation:
            raise RuntimeError("DFlash boundary reservation is stale")
        return record

    @staticmethod
    def _require_owner_locked(
        record: _DFlashBoundaryReservationRecord,
        owner: object,
    ) -> None:
        if record.owner is not owner:
            raise RuntimeError("DFlash boundary reservation owner is stale")

    def _snapshot_locked(
        self,
        record: _DFlashBoundaryReservationRecord,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        return PackedAuxiliarySlotReservationSnapshot(
            metadata_buffer_index=record.row_index,
            metadata_slot_generation=record.generation,
            destination_segments=(self.registration.segment_for_row(record.row_index),),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryRegisteredRow:
    """Generation-bound proof of one live registered device row.

    :ivar registration: Exact process-lifetime allocation geometry.
    :ivar reservation: Exact live row generation and address.
    """

    registration: DFlashBoundaryRegistration
    reservation: PackedAuxiliarySlotReservationSnapshot

    def __post_init__(self) -> None:
        """Validate exact row geometry against its registration."""

        if type(self.registration) is not DFlashBoundaryRegistration:
            raise TypeError("registration must be DFlashBoundaryRegistration")
        if type(self.reservation) is not PackedAuxiliarySlotReservationSnapshot:
            raise TypeError(
                "reservation must be PackedAuxiliarySlotReservationSnapshot"
            )
        expected = (
            self.registration.segment_for_row(self.reservation.metadata_buffer_index),
        )
        if self.reservation.destination_segments != expected:
            raise ValueError("DFlash boundary registered-row geometry was altered")

    @property
    def transfer_request(self) -> tuple[int, int, int]:
        """Return the exact native VRAM request for this row.

        :returns: Address, byte length, and CUDA-device tuple.
        """

        return self.registration.transfer_request_for_row(
            self.reservation.metadata_buffer_index
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryRemoteRow:
    """Generation-bound decoder row capability carried by a packed plan.

    :ivar metadata_buffer_index: Exact decoder row index.
    :ivar metadata_slot_generation: Exact decoder row generation.
    :ivar segment: Decoder-authored registered row address.
    :ivar device_index: Destination CUDA device selected by the live route.
    """

    metadata_buffer_index: int
    metadata_slot_generation: bytes
    segment: PackedAuxiliaryDestinationSegment
    device_index: int

    def __post_init__(self) -> None:
        """Validate exact generation and eight-byte row geometry."""

        if (
            type(self.metadata_buffer_index) is not int
            or self.metadata_buffer_index < 0
        ):
            raise ValueError("DFlash boundary remote row index must be non-negative")
        if type(self.metadata_slot_generation) is not bytes:
            raise TypeError("DFlash boundary remote generation must be bytes")
        if len(self.metadata_slot_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "DFlash boundary remote generation has an invalid byte length"
            )
        if type(self.segment) is not PackedAuxiliaryDestinationSegment:
            raise TypeError("segment must be PackedAuxiliaryDestinationSegment")
        if self.segment.item_length != DFLASH_BOUNDARY_ROW_BYTES:
            raise ValueError("DFlash boundary remote row is not eight bytes")
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("DFlash boundary remote device must be non-negative")

    @classmethod
    def from_plan(
        cls,
        plan: PackedAuxiliaryPlan,
        device_index: int,
    ) -> "DFlashBoundaryRemoteRow":
        """Bind one decoder-authored plan to a typed remote VRAM row.

        :param plan: Exact live decoder boundary-row plan.
        :param device_index: Destination CUDA device selected by the route.
        :returns: Typed generation-bound remote row capability.
        """

        if type(plan) is not PackedAuxiliaryPlan:
            raise TypeError("plan must be PackedAuxiliaryPlan")
        if len(plan.destination_segments) != 1:
            raise ValueError("DFlash boundary plan must contain exactly one segment")
        return cls(
            metadata_buffer_index=plan.metadata_buffer_index,
            metadata_slot_generation=plan.metadata_slot_generation,
            segment=plan.destination_segments[0],
            device_index=device_index,
        )

    @property
    def transfer_request(self) -> tuple[int, int, int]:
        """Return the exact remote VRAM descriptor request.

        :returns: Address, byte length, and CUDA-device tuple.
        """

        return (self.segment.address, self.segment.item_length, self.device_index)


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryTransportAccounting:
    """Exact memory-kind accounting for one boundary transfer.

    :ivar source_vram_bytes: Exact registered source bytes read.
    :ivar destination_vram_bytes: Exact registered destination bytes written.
    :ivar vram_transport_bytes: Aggregate device-to-device transport bytes.
    :ivar dram_transport_bytes: Aggregate host-memory transport bytes.
    """

    source_vram_bytes: int
    destination_vram_bytes: int
    vram_transport_bytes: int
    dram_transport_bytes: int

    def __post_init__(self) -> None:
        """Validate exact byte conservation and zero DRAM transport."""

        if (
            self.source_vram_bytes != DFLASH_BOUNDARY_ROW_BYTES
            or self.destination_vram_bytes != DFLASH_BOUNDARY_ROW_BYTES
            or self.vram_transport_bytes != DFLASH_BOUNDARY_ROW_BYTES
        ):
            raise ValueError("DFlash boundary VRAM accounting does not conserve")
        if type(self.dram_transport_bytes) is not int:
            raise TypeError("dram_transport_bytes must be an integer")
        if self.dram_transport_bytes != 0:
            raise ValueError("DFlash boundary transport must contain zero DRAM bytes")


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryNixlDescriptors:
    """Opaque native descriptors bound to exact all-VRAM row requests.

    :ivar source_request: Exact source row address and device.
    :ivar destination_request: Exact destination row address and device.
    :ivar source_descriptors: Native source VRAM descriptor collection.
    :ivar destination_descriptors: Native destination VRAM descriptors.
    :ivar accounting: Exact zero-DRAM transport accounting.
    """

    source_request: tuple[int, int, int]
    destination_request: tuple[int, int, int]
    source_descriptors: object
    destination_descriptors: object
    accounting: DFlashBoundaryTransportAccounting

    def __post_init__(self) -> None:
        """Validate complete descriptor ownership and byte geometry."""

        for label, request in (
            ("source", self.source_request),
            ("destination", self.destination_request),
        ):
            if (
                type(request) is not tuple
                or len(request) != 3
                or request[1] != DFLASH_BOUNDARY_ROW_BYTES
            ):
                raise ValueError(f"DFlash boundary {label} request is invalid")
        if self.source_descriptors is None or self.destination_descriptors is None:
            raise ValueError("DFlash boundary native descriptors must be owned")
        if type(self.accounting) is not DFlashBoundaryTransportAccounting:
            raise TypeError("accounting must be DFlashBoundaryTransportAccounting")


def build_dflash_boundary_nixl_descriptors(
    agent: DFlashBoundaryDescriptorAgent,
    source: DFlashBoundaryRegisteredRow,
    destination: DFlashBoundaryRegisteredRow | DFlashBoundaryRemoteRow,
) -> DFlashBoundaryNixlDescriptors:
    """Build exact source and destination VRAM descriptors.

    :param agent: Source-process NIXL descriptor authority.
    :param source: Exact generation-bound source row.
    :param destination: Exact generation-bound destination row.
    :returns: Opaque native descriptors plus zero-DRAM accounting.
    """

    if type(source) is not DFlashBoundaryRegisteredRow:
        raise TypeError("source must be DFlashBoundaryRegisteredRow")
    if type(destination) not in (
        DFlashBoundaryRegisteredRow,
        DFlashBoundaryRemoteRow,
    ):
        raise TypeError("destination must be a registered or remote boundary row")
    destination_request = destination.transfer_request
    source_request = source.transfer_request
    source_descriptors = agent.get_xfer_descs(
        [source_request],
        DFLASH_BOUNDARY_MEMORY_KIND,
    )
    destination_descriptors = agent.get_xfer_descs(
        [destination_request],
        DFLASH_BOUNDARY_MEMORY_KIND,
    )
    descriptor_sets = (source_descriptors, destination_descriptors)
    if any(
        value is None or (isinstance(value, list | tuple) and len(value) == 0)
        for value in descriptor_sets
    ):
        raise RuntimeError("NIXL returned no DFlash boundary transfer descriptors")
    return DFlashBoundaryNixlDescriptors(
        source_request=source_request,
        destination_request=destination_request,
        source_descriptors=source_descriptors,
        destination_descriptors=destination_descriptors,
        accounting=DFlashBoundaryTransportAccounting(
            source_vram_bytes=DFLASH_BOUNDARY_ROW_BYTES,
            destination_vram_bytes=DFLASH_BOUNDARY_ROW_BYTES,
            vram_transport_bytes=DFLASH_BOUNDARY_ROW_BYTES,
            dram_transport_bytes=0,
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundaryAdoptedValue:
    """Request-owned device token detached from a reusable destination row.

    :ivar boundary_token_id: One-element CUDA int64 tensor.
    :ivar completion_event: Event recorded after the device-to-device copy.
    """

    boundary_token_id: torch.Tensor
    completion_event: torch.cuda.Event

    def __post_init__(self) -> None:
        """Validate exact device-resident adoption shape."""

        if type(self.boundary_token_id) is not torch.Tensor:
            raise TypeError("boundary_token_id must be a torch.Tensor")
        if self.boundary_token_id.device.type != "cuda":
            raise ValueError("adopted DFlash boundary token must remain on CUDA")
        if self.boundary_token_id.dtype is not torch.int64:
            raise ValueError("adopted DFlash boundary token must use int64")
        if tuple(self.boundary_token_id.shape) != DFLASH_BOUNDARY_ROW_SHAPE:
            raise ValueError("adopted DFlash boundary token must contain one value")
        if not isinstance(self.completion_event, torch.cuda.Event):
            raise TypeError("completion_event must be a CUDA event")


class DFlashBoundaryDeviceRowPool:
    """Stable process-lifetime CUDA rows registered directly with NIXL."""

    _boundary_token_ids: torch.Tensor
    _registration: DFlashBoundaryNixlRegistration
    _row_allocator: DFlashBoundaryRowAllocator

    def __init__(
        self,
        agent: DFlashBoundaryMemoryAgent,
        *,
        row_capacity: int,
        device: torch.device,
    ) -> None:
        """Allocate, register, and own one boundary row pool.

        :param agent: Exact process-local NIXL agent.
        :param row_capacity: Maximum in-flight boundary generations.
        :param device: Indexed CUDA device owning the rows.
        """

        if type(row_capacity) is not int or row_capacity <= 0:
            raise ValueError("row_capacity must be a positive integer")
        if type(device) is not torch.device or device.type != "cuda":
            raise ValueError("device must be an indexed CUDA device")
        if device.index is None or device.index < 0:
            raise ValueError("device must be an indexed CUDA device")
        boundary_token_ids = torch.empty(
            (row_capacity, *DFLASH_BOUNDARY_ROW_SHAPE),
            dtype=DFLASH_BOUNDARY_ROW_DTYPE,
            device=device,
        )
        receipt = DFlashBoundaryRegistration(
            base_address=boundary_token_ids.data_ptr(),
            allocation_length=boundary_token_ids.nbytes,
            device_index=device.index,
            row_capacity=row_capacity,
        )
        registration = DFlashBoundaryNixlRegistration(agent, receipt)
        self._boundary_token_ids = boundary_token_ids
        self._registration = registration
        self._row_allocator = DFlashBoundaryRowAllocator(registration)

    @property
    def registration(self) -> DFlashBoundaryRegistration:
        """Return process-lifetime registered allocation geometry.

        :returns: Exact all-VRAM registration receipt.
        """

        return self._registration.receipt

    @property
    def row_capacity(self) -> int:
        """Return the exact physical request capacity.

        :returns: Number of independently leased boundary rows.
        """

        return self._registration.receipt.row_capacity

    def lease_row(self, lifecycle_authority: object) -> "DFlashBoundaryRowLease":
        """Lease one row to an exact process-level lifecycle authority.

        :param lifecycle_authority: Sole authority allowed to mutate the lease.
        :returns: Strong generation-bound row owner.
        """

        return DFlashBoundaryRowLease(self, lifecycle_authority)

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Lease one free row directly to the allocation authority.

        :param owner: Exact process-local reservation owner.
        :returns: Opaque exact-generation row reservation.
        """

        return self._row_allocator.allocate_packed_auxiliary_slot(owner)

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Return exact generation and registered address geometry.

        :param reservation: Exact allocator-minted reservation.
        :returns: Immutable live row snapshot.
        """

        return self._row_allocator.packed_auxiliary_slot_reservation_snapshot(
            reservation
        )

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return one terminal row to allocator ownership.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact reservation authority.
        """

        self._row_allocator.release_packed_auxiliary_slot(reservation, owner)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Permanently remove one ambiguous row from reuse.

        :param reservation: Exact allocator-minted reservation.
        :param owner: Exact reservation authority.
        """

        self._row_allocator.quarantine_packed_auxiliary_slot(reservation, owner)

    def registered_row(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> DFlashBoundaryRegisteredRow:
        """Prove one exact generation remains live in registered VRAM.

        :param snapshot: Exact allocator-authored live snapshot.
        :returns: Strong registered-row proof.
        """

        return self._row_allocator.registered_row(snapshot)

    def enqueue_source_projection(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
        boundary_token_id: torch.Tensor,
        gateway_result_slot: TerminalGatewayResultSlot,
        *,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        """Copy one token to both allowed destinations, then record completion.

        The same source tensor feeds the registered device row and the
        gateway-facing pinned result slot. The exact producer event is recorded
        only after both copies, so native readiness binds both views without a
        second completion channel.

        :param snapshot: Exact live source-row generation.
        :param boundary_token_id: One-element CUDA int64 target token.
        :param gateway_result_slot: Allowed gateway-facing pinned-host slot.
        :param stream: Model-producing CUDA stream.
        :param producer_event: Exact event covering both copies.
        """

        row_index = self._row_allocator.require_snapshot(snapshot)
        self._validate_boundary_token(boundary_token_id)
        if not isinstance(gateway_result_slot, TerminalGatewayResultSlot):
            raise TypeError(
                "gateway_result_slot must inherit TerminalGatewayResultSlot"
            )
        self._validate_stream_and_event(stream, producer_event)
        destination = self._boundary_token_ids[row_index]
        with torch.cuda.stream(stream):
            destination.copy_(boundary_token_id.reshape(1), non_blocking=True)
            gateway_result_slot.enqueue_copy(boundary_token_id)
            producer_event.record(stream)

    def enqueue_destination_adoption(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
        *,
        stream: torch.cuda.Stream,
    ) -> DFlashBoundaryAdoptedValue:
        """Detach one received token without reading it on the host.

        The reusable row remains leased until the returned completion event is
        consumed by terminal lifecycle authority.

        :param snapshot: Exact live destination-row generation.
        :param stream: Scheduler-authorized CUDA adoption stream.
        :returns: Request-owned CUDA token and exact copy-completion event.
        """

        row_index = self._row_allocator.require_snapshot(snapshot)
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be torch.cuda.Stream")
        if stream.device_index != self.registration.device_index:
            raise ValueError("destination stream belongs to another CUDA device")
        with torch.cuda.stream(stream):
            boundary_token_id = self._boundary_token_ids[row_index].clone()
            completion_event = torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=False,
            )
            completion_event.record(stream)
        return DFlashBoundaryAdoptedValue(
            boundary_token_id=boundary_token_id,
            completion_event=completion_event,
        )

    def inventory(self) -> tuple[int, int, int]:
        """Return free, active, and quarantined row counts.

        :returns: Conservation-complete row population.
        """

        return self._row_allocator.inventory()

    def _validate_boundary_token(self, value: torch.Tensor) -> None:
        if type(value) is not torch.Tensor:
            raise TypeError("boundary_token_id must be a torch.Tensor")
        if value.device.type != "cuda":
            raise ValueError("boundary_token_id must reside in CUDA memory")
        if value.device.index != self.registration.device_index:
            raise ValueError("boundary_token_id belongs to another CUDA device")
        if value.dtype is not DFLASH_BOUNDARY_ROW_DTYPE:
            raise ValueError("boundary_token_id must use int64")
        if value.numel() != 1 or not value.is_contiguous():
            raise ValueError("boundary_token_id must be one contiguous value")

    def _validate_stream_and_event(
        self,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be torch.cuda.Stream")
        if not isinstance(producer_event, torch.cuda.Event):
            raise TypeError("producer_event must be torch.cuda.Event")
        if stream.device_index != self.registration.device_index:
            raise ValueError("producer stream belongs to another CUDA device")


class DFlashBoundaryRowLeaseState(enum.StrEnum):
    """Exact process-local lifetime state of one device row lease."""

    ACTIVE = "active"
    RELEASED = "released"
    QUARANTINED = "quarantined"


class DFlashBoundaryRowLease:
    """Strong generation-bound ownership of one source or destination row."""

    _pool: DFlashBoundaryDeviceRowPool
    _lifecycle_authority: object
    _reservation_owner: object
    _reservation: object
    _snapshot: PackedAuxiliarySlotReservationSnapshot
    _state: DFlashBoundaryRowLeaseState
    _lock: threading.Lock

    def __init__(
        self,
        pool: DFlashBoundaryDeviceRowPool,
        lifecycle_authority: object,
    ) -> None:
        """Acquire one row under an exact lifecycle authority.

        :param pool: Process-lifetime registered device row pool.
        :param lifecycle_authority: Sole authority allowed to retire the row.
        """

        if type(pool) is not DFlashBoundaryDeviceRowPool:
            raise TypeError("pool must be DFlashBoundaryDeviceRowPool")
        if lifecycle_authority is None:
            raise ValueError("lifecycle_authority must not be None")
        self._pool = pool
        self._lifecycle_authority = lifecycle_authority
        self._reservation_owner = object()
        self._reservation = pool.allocate_packed_auxiliary_slot(self._reservation_owner)
        self._snapshot = pool.packed_auxiliary_slot_reservation_snapshot(
            self._reservation
        )
        self._state = DFlashBoundaryRowLeaseState.ACTIVE
        self._lock = threading.Lock()

    @property
    def snapshot(self) -> PackedAuxiliarySlotReservationSnapshot:
        """Return the immutable exact-generation row snapshot.

        :returns: Allocator-authored row identity and address geometry.
        """

        with self._lock:
            self._require_active_locked()
            return self._snapshot

    @property
    def state(self) -> DFlashBoundaryRowLeaseState:
        """Return the current exact row lifetime state.

        :returns: Active, released, or quarantined state.
        """

        with self._lock:
            return self._state

    @property
    def registration(self) -> DFlashBoundaryRegistration:
        """Return the process-lifetime registered row geometry.

        :returns: Exact all-VRAM registration receipt.
        """

        return self._pool.registration

    def registered_row(self, authority: object) -> DFlashBoundaryRegisteredRow:
        """Prove this exact generation remains live in registered VRAM.

        :param authority: Exact lifecycle authority captured at lease creation.
        :returns: Exact live registered-row proof.
        """

        with self._lock:
            self._require_authority_locked(authority)
            self._require_active_locked()
            return self._pool.registered_row(self._snapshot)

    def enqueue_source_projection(
        self,
        authority: object,
        boundary_token_id: torch.Tensor,
        gateway_result_slot: TerminalGatewayResultSlot,
        *,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        """Stage the boundary token under exact lifecycle ownership.

        :param authority: Exact lifecycle authority captured at lease creation.
        :param boundary_token_id: One-element device target token.
        :param gateway_result_slot: Allowed gateway-facing result slot.
        :param stream: Model-producing CUDA stream.
        :param producer_event: Event recorded after both exact copies.
        """

        with self._lock:
            self._require_authority_locked(authority)
            self._require_active_locked()
            self._pool.enqueue_source_projection(
                self._snapshot,
                boundary_token_id,
                gateway_result_slot,
                stream=stream,
                producer_event=producer_event,
            )

    def enqueue_destination_adoption(
        self,
        authority: object,
        *,
        stream: torch.cuda.Stream,
    ) -> DFlashBoundaryAdoptedValue:
        """Detach the destination token under exact lifecycle ownership.

        :param authority: Exact lifecycle authority captured at lease creation.
        :param stream: Scheduler-authorized CUDA adoption stream.
        :returns: Request-owned token and exact copy-completion event.
        """

        with self._lock:
            self._require_authority_locked(authority)
            self._require_active_locked()
            return self._pool.enqueue_destination_adoption(
                self._snapshot,
                stream=stream,
            )

    def release(self, authority: object) -> None:
        """Return this row only under exact terminal lifecycle authority.

        :param authority: Exact lifecycle authority captured at lease creation.
        """

        with self._lock:
            self._require_authority_locked(authority)
            self._require_active_locked()
            self._pool.release_packed_auxiliary_slot(
                self._reservation,
                self._reservation_owner,
            )
            self._state = DFlashBoundaryRowLeaseState.RELEASED

    def quarantine(self, authority: object) -> None:
        """Permanently retain this row after ambiguous terminality.

        :param authority: Exact lifecycle authority captured at lease creation.
        """

        with self._lock:
            self._require_authority_locked(authority)
            self._require_active_locked()
            self._pool.quarantine_packed_auxiliary_slot(
                self._reservation,
                self._reservation_owner,
            )
            self._state = DFlashBoundaryRowLeaseState.QUARANTINED

    def _require_authority_locked(self, authority: object) -> None:
        if authority is not self._lifecycle_authority:
            raise RuntimeError("DFlash boundary row lifecycle authority is stale")

    def _require_active_locked(self) -> None:
        if self._state is not DFlashBoundaryRowLeaseState.ACTIVE:
            raise RuntimeError("DFlash boundary row lease is already terminal")


class DFlashBoundarySourceTransferState(enum.StrEnum):
    """Exact source boundary transfer state owned outside the actor."""

    POSTED = "posted"
    SETTLED = "settled"
    RELEASED = "released"
    QUARANTINED = "quarantined"


class DFlashBoundarySourceTransfer:
    """Opaque source transfer identity retained by actor callbacks."""

    __slots__ = ("_owner_nonce", "_token")

    _owner_nonce: object
    _token: object

    def __init__(
        self,
        owner_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one owner-private transfer identity.

        :param owner_nonce: Exact issuing owner identity.
        :param token: Owner-private registry key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _SOURCE_TRANSFER_CONSTRUCTION_SEAL:
            raise TypeError("DFlash boundary transfers are owner constructed")
        self._owner_nonce = owner_nonce
        self._token = token


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashBoundarySourceTransportInventory:
    """Conservation-complete source boundary transport inventory.

    :ivar active_count: All retained source transfer identities.
    :ivar posted_count: Transfers awaiting native outcome authority.
    :ivar settled_count: Successful transfers retained through ACK.
    :ivar released_count: Successfully retired transfers since construction.
    :ivar quarantined_count: Ambiguous transfers retained for process lifetime.
    :ivar unowned_native_handle_count: Handles retained after arming failure.
    """

    active_count: int
    posted_count: int
    settled_count: int
    released_count: int
    quarantined_count: int
    unowned_native_handle_count: int

    def __post_init__(self) -> None:
        """Validate exact non-negative inventory counts."""

        values = (
            self.active_count,
            self.posted_count,
            self.settled_count,
            self.released_count,
            self.quarantined_count,
            self.unowned_native_handle_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("DFlash boundary transport counts must be non-negative")
        if self.posted_count + self.settled_count + self.quarantined_count != (
            self.active_count
        ):
            raise ValueError("DFlash boundary transport inventory does not conserve")


@dataclasses.dataclass(slots=True)
class _DFlashBoundarySourceTransferRecord:
    """Mutable source transfer authority retained through terminal ACK."""

    public: DFlashBoundarySourceTransfer
    plan: PackedAuxiliaryPlan
    source_lease: DFlashBoundaryRowLease
    remote_row: DFlashBoundaryRemoteRow
    descriptors: DFlashBoundaryNixlDescriptors
    remote_agent_name: str
    native_transfer: DFlashBoundaryDirectTransfer
    state: DFlashBoundarySourceTransferState
    outcome: PackedDFlashBoundaryOutcome | None = None


_SOURCE_TRANSFER_CONSTRUCTION_SEAL = object()


class DFlashBoundarySourceTransportOwner:
    """Own boundary source rows and native transfers behind actor callbacks.

    The owner never polls. Static peer enrollment makes the supplied post
    operation one-shot: a non-ready result is a failure, not a retry loop.
    """

    _pool: DFlashBoundaryDeviceRowPool
    _agent: DFlashBoundaryNativeTransferAgent
    _direct_owner: DFlashBoundaryDirectOwner
    _writer_id: StagingWriterId
    _post: Callable[[object], object]
    _lifecycle_authority: object
    _owner_nonce: object
    _records: dict[object, _DFlashBoundarySourceTransferRecord]
    _unowned_native_handles: list[object]
    _released_count: int
    _lock: threading.Lock

    def __init__(
        self,
        *,
        pool: DFlashBoundaryDeviceRowPool,
        agent: DFlashBoundaryNativeTransferAgent,
        direct_owner: DFlashBoundaryDirectOwner,
        writer_id: StagingWriterId,
        post: Callable[[object], object],
    ) -> None:
        """Bind one canonical writer to stable VRAM and a native owner.

        :param pool: Process-lifetime source boundary row pool.
        :param agent: Exact NIXL source agent.
        :param direct_owner: Direct native terminal-owner adapter.
        :param writer_id: Canonical source writer allowed to send the token.
        :param post: One-shot native post operation with no retry or polling.
        """

        if type(pool) is not DFlashBoundaryDeviceRowPool:
            raise TypeError("pool must be DFlashBoundaryDeviceRowPool")
        if type(writer_id) is not StagingWriterId:
            raise TypeError("writer_id must be StagingWriterId")
        if writer_id.source_pp_rank != 0 or writer_id.source_cp_rank != 0:
            raise ValueError("DFlash boundary writer requires PP0 and CP0")
        if not callable(post):
            raise TypeError("post must be callable")
        self._pool = pool
        self._agent = agent
        self._direct_owner = direct_owner
        self._writer_id = writer_id
        self._post = post
        self._lifecycle_authority = object()
        self._owner_nonce = object()
        self._records = {}
        self._unowned_native_handles = []
        self._released_count = 0
        self._lock = threading.Lock()

    @property
    def writer_id(self) -> StagingWriterId:
        """Return the sole canonical writer owned by this transport.

        :returns: Exact packed source writer identity.
        """

        return self._writer_id

    def lease_source_row(self) -> DFlashBoundaryRowLease:
        """Lease one stable row before the request's model submission.

        :returns: Exact generation-bound source row lease.
        """

        return self._pool.lease_row(self._lifecycle_authority)

    def enqueue_source_projection(
        self,
        lease: DFlashBoundaryRowLease,
        boundary_token_id: torch.Tensor,
        gateway_result_slot: TerminalGatewayResultSlot,
        *,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        """Stage the exact token into VRAM and the gateway result slot.

        :param lease: Exact source row leased by this owner.
        :param boundary_token_id: One-element CUDA int64 target token.
        :param gateway_result_slot: Allowed gateway-facing pinned-host slot.
        :param stream: Model-producing CUDA stream.
        :param producer_event: Event recorded after both exact copies.
        """

        if type(lease) is not DFlashBoundaryRowLease:
            raise TypeError("lease must be DFlashBoundaryRowLease")
        lease.enqueue_source_projection(
            self._lifecycle_authority,
            boundary_token_id,
            gateway_result_slot,
            stream=stream,
            producer_event=producer_event,
        )

    def post(
        self,
        *,
        plan: PackedAuxiliaryPlan,
        source_lease: DFlashBoundaryRowLease,
        destination_device_index: int,
        remote_handle: object,
        remote_agent_name: str,
        binding_digest: bytes,
    ) -> DFlashBoundarySourceTransfer:
        """Initialize, arm, and post one exact all-VRAM boundary write.

        :param plan: Exact decoder-authored destination plan.
        :param source_lease: Stable row covered by the producer event.
        :param destination_device_index: CUDA device selected by the live route.
        :param remote_handle: Exact enrolled decoder NIXL handle.
        :param remote_agent_name: Exact enrolled decoder agent identity.
        :param binding_digest: Exact source lifecycle binding digest.
        :returns: Opaque retained transfer consumed by actor callbacks.
        """

        if type(plan) is not PackedAuxiliaryPlan:
            raise TypeError("plan must be PackedAuxiliaryPlan")
        if plan.canonical_writer_id != self._writer_id:
            raise ValueError("DFlash boundary plan names another canonical writer")
        if type(source_lease) is not DFlashBoundaryRowLease:
            raise TypeError("source_lease must be DFlashBoundaryRowLease")
        if remote_handle is None:
            raise ValueError("remote_handle must not be None")
        if type(remote_agent_name) is not str or len(remote_agent_name) == 0:
            raise ValueError("remote_agent_name must be a non-empty string")
        _validate_digest(binding_digest, "binding_digest")
        source_row = source_lease.registered_row(self._lifecycle_authority)
        remote_row = DFlashBoundaryRemoteRow.from_plan(
            plan,
            destination_device_index,
        )
        descriptors = build_dflash_boundary_nixl_descriptors(
            self._agent,
            source_row,
            remote_row,
        )
        native_handle: object | None = None
        record: _DFlashBoundarySourceTransferRecord | None = None
        try:
            native_handle = self._agent.initialize_xfer(
                "WRITE",
                descriptors.source_descriptors,
                descriptors.destination_descriptors,
                remote_handle,
                b"",
            )
            if native_handle is None:
                raise RuntimeError("NIXL returned no DFlash boundary transfer handle")
            native_transfer = self._direct_owner.arm_transfer(
                native_handle,
                binding_digest,
            )
            token = object()
            public = DFlashBoundarySourceTransfer(
                self._owner_nonce,
                token,
                _SOURCE_TRANSFER_CONSTRUCTION_SEAL,
            )
            record = _DFlashBoundarySourceTransferRecord(
                public=public,
                plan=plan,
                source_lease=source_lease,
                remote_row=remote_row,
                descriptors=descriptors,
                remote_agent_name=remote_agent_name,
                native_transfer=native_transfer,
                state=DFlashBoundarySourceTransferState.POSTED,
            )
            with self._lock:
                self._records[token] = record
            self._direct_owner.post_transfer(native_transfer, self._post)
            return public
        except Exception:
            logger.error(
                "DFlash boundary source post failed:\n%s",
                traceback.format_exc(),
            )
            if record is None and native_handle is not None:
                with self._lock:
                    self._unowned_native_handles.append(native_handle)
            self._quarantine_lease_after_failure(source_lease)
            if record is not None:
                with self._lock:
                    record.state = DFlashBoundarySourceTransferState.QUARANTINED
            raise

    def settle(
        self,
        transfer: DFlashBoundarySourceTransfer,
        action: NativeTerminalOwnerAction,
        metadata: PackedDFlashBoundaryMetadata,
    ) -> PackedDFlashBoundaryOutcome:
        """Settle exact success without releasing the row or native handle.

        :param transfer: Opaque transfer returned by :meth:`post`.
        :param action: Exact source-outcome owner action.
        :param metadata: Frozen boundary token and request counters.
        :returns: Authenticated all-VRAM boundary outcome.
        """

        if type(metadata) is not PackedDFlashBoundaryMetadata:
            raise TypeError("metadata must be PackedDFlashBoundaryMetadata")
        record = self._require_record(transfer)
        with self._lock:
            if record.state is not DFlashBoundarySourceTransferState.POSTED:
                raise RuntimeError("DFlash boundary transfer cannot settle twice")
        try:
            receipt = self._direct_owner.settle_success(
                record.native_transfer,
                action,
            )
            self._validate_receipt(record, receipt)
            outcome = PackedDFlashBoundaryOutcome.create(
                plan=record.plan,
                writer_id=self._writer_id,
                native_handle_generation=receipt.generation,
                descriptor_digest=_decode_native_digest(
                    receipt.descriptorDigest,
                    "descriptor",
                ),
                evidence_digest=_decode_native_digest(
                    receipt.evidenceDigest,
                    "evidence",
                ),
                metadata=metadata,
            )
        except Exception:
            logger.error(
                "DFlash boundary source settlement failed:\n%s",
                traceback.format_exc(),
            )
            self._quarantine_lease_after_failure(record.source_lease)
            with self._lock:
                record.state = DFlashBoundarySourceTransferState.QUARANTINED
            raise
        with self._lock:
            if record.state is not DFlashBoundarySourceTransferState.POSTED:
                raise RuntimeError("DFlash boundary settlement raced lifecycle state")
            record.outcome = outcome
            record.state = DFlashBoundarySourceTransferState.SETTLED
        return outcome

    def release(
        self,
        transfer: DFlashBoundarySourceTransfer,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release native handle and source row under exact ACK authority.

        :param transfer: Exact successfully settled source transfer.
        :param action: Matching one-shot source ACK action.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
            raise ValueError("DFlash boundary release requires SOURCE_ACK_READY")
        record = self._require_record(transfer)
        with self._lock:
            if record.state is not DFlashBoundarySourceTransferState.SETTLED:
                raise RuntimeError("DFlash boundary release requires settlement")
        self._direct_owner.release_transfer(record.native_transfer, action)
        record.source_lease.release(self._lifecycle_authority)
        with self._lock:
            current = self._records.get(transfer._token)
            if current is not record:
                raise RuntimeError("DFlash boundary transfer registry changed")
            record.state = DFlashBoundarySourceTransferState.RELEASED
            del self._records[transfer._token]
            self._released_count += 1

    def settle_failure(
        self,
        transfer: DFlashBoundarySourceTransfer,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Settle failure and permanently quarantine its source row.

        :param transfer: Exact ambiguous or failed source transfer.
        :param action: Matching quarantine or process-fatal owner action.
        """

        record = self._require_record(transfer)
        with self._lock:
            if record.state is not DFlashBoundarySourceTransferState.POSTED:
                raise RuntimeError("DFlash boundary failure cannot settle twice")
        self._direct_owner.settle_failure(record.native_transfer, action)
        self._quarantine_lease_after_failure(record.source_lease)
        with self._lock:
            record.state = DFlashBoundarySourceTransferState.QUARANTINED

    def inventory(self) -> DFlashBoundarySourceTransportInventory:
        """Return complete retained source transfer authority.

        :returns: Conservation-complete transfer and orphan-handle counts.
        """

        with self._lock:
            states = tuple(record.state for record in self._records.values())
            return DFlashBoundarySourceTransportInventory(
                active_count=len(states),
                posted_count=states.count(DFlashBoundarySourceTransferState.POSTED),
                settled_count=states.count(DFlashBoundarySourceTransferState.SETTLED),
                released_count=self._released_count,
                quarantined_count=states.count(
                    DFlashBoundarySourceTransferState.QUARANTINED
                ),
                unowned_native_handle_count=len(self._unowned_native_handles),
            )

    def _require_record(
        self,
        transfer: DFlashBoundarySourceTransfer,
    ) -> _DFlashBoundarySourceTransferRecord:
        if type(transfer) is not DFlashBoundarySourceTransfer:
            raise TypeError("transfer must be DFlashBoundarySourceTransfer")
        if transfer._owner_nonce is not self._owner_nonce:
            raise RuntimeError("DFlash boundary transfer belongs to another owner")
        with self._lock:
            record = self._records.get(transfer._token)
            if record is None or record.public is not transfer:
                raise RuntimeError("DFlash boundary transfer is not active")
            return record

    def _quarantine_lease_after_failure(
        self,
        lease: DFlashBoundaryRowLease,
    ) -> None:
        if lease.state is DFlashBoundaryRowLeaseState.ACTIVE:
            lease.quarantine(self._lifecycle_authority)

    def _validate_receipt(
        self,
        record: _DFlashBoundarySourceTransferRecord,
        receipt: DFlashBoundaryNativeReceipt,
    ) -> None:
        transfer = record.native_transfer
        if (
            receipt.handleIdentity != transfer.handle_identity
            or receipt.generation != transfer.generation
        ):
            raise ValueError("DFlash boundary receipt changed transfer generation")
        if not receipt.submissionSealed or not receipt.completionClaimed:
            raise ValueError("DFlash boundary receipt lacks take-once authority")
        _require_native_enum(receipt.state, "NIXL_XFER_ATTESTATION_REMOTE_FLUSHED")
        _require_native_enum(receipt.status, "NIXL_SUCCESS")
        _require_native_enum(receipt.operation, "NIXL_WRITE")
        _require_native_enum(receipt.localMemoryType, "VRAM_SEG")
        _require_native_enum(receipt.remoteMemoryType, "VRAM_SEG")
        if receipt.backend != "UCX":
            raise ValueError("DFlash boundary receipt backend is not UCX")
        if receipt.localAgent != self._agent.name:
            raise ValueError("DFlash boundary receipt belongs to another source")
        if receipt.remoteAgent != record.remote_agent_name:
            raise ValueError("DFlash boundary receipt belongs to another decoder")
        if type(receipt.error) is not str or len(receipt.error) != 0:
            raise ValueError("successful DFlash boundary receipt contains an error")
        segments = receipt.segments
        if type(segments) is not tuple or len(segments) != 1:
            raise ValueError("DFlash boundary receipt must contain one segment")
        segment = segments[0]
        source = record.descriptors.source_request
        destination = record.descriptors.destination_request
        if (
            segment.index != 0
            or segment.localAddress != source[0]
            or segment.remoteAddress != destination[0]
            or segment.length != DFLASH_BOUNDARY_ROW_BYTES
            or source[1] != DFLASH_BOUNDARY_ROW_BYTES
            or destination[1] != DFLASH_BOUNDARY_ROW_BYTES
            or not segment.posted
        ):
            raise ValueError("DFlash boundary receipt descriptors differ")


def _validate_digest(value: bytes, label: str) -> None:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_REQUEST_DIGEST_BYTES:
        raise ValueError(f"{label} must contain {PACKED_REQUEST_DIGEST_BYTES} bytes")


def _require_native_enum(value: DFlashBoundaryNativeEnum, expected: str) -> None:
    if value.name != expected:
        raise ValueError(f"DFlash boundary receipt value is not {expected}")


def _decode_native_digest(value: str, label: str) -> bytes:
    if type(value) is not str or len(value) != PACKED_REQUEST_DIGEST_BYTES * 2:
        raise ValueError(f"native DFlash boundary {label} digest is not SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"native DFlash boundary {label} digest is not hexadecimal"
        ) from error
    if len(decoded) != PACKED_REQUEST_DIGEST_BYTES:
        raise ValueError(f"native DFlash boundary {label} digest is not SHA-256")
    return decoded
