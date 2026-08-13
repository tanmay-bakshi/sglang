import collections
import dataclasses
import enum
import secrets
import threading
from contextlib import nullcontext
from typing import Protocol

import torch
from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
)


class DFlashAuxiliaryField(enum.StrEnum):
    """Device-resident fields transferred from prefill to decode."""

    TOPK_PROBABILITIES = "output_topk_p"
    TOPK_INDICES = "output_topk_index"
    HIDDEN_STATES = "output_hidden_states"


DFLASH_AUXILIARY_FIELDS: tuple[DFlashAuxiliaryField, ...] = (
    DFlashAuxiliaryField.TOPK_PROBABILITIES,
    DFlashAuxiliaryField.TOPK_INDICES,
    DFlashAuxiliaryField.HIDDEN_STATES,
)
DFLASH_AUXILIARY_MEMORY_KIND = "VRAM"


class DFlashAuxiliaryMemoryAgent(Protocol):
    """NIXL memory-registration surface required by auxiliary row pools."""

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Register canonical auxiliary allocations.

        :param addresses: Ordered NIXL address descriptors.
        :param memory_kind: Required NIXL memory kind.
        :returns: Opaque process-lifetime registration handle.
        """

        ...


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashAuxiliaryRegistration:
    """Exact process-lifetime NIXL registration receipt for device rows.

    :ivar fields: Canonical field order shared by source and destination.
    :ivar base_addresses: Base device address for every field allocation.
    :ivar allocation_lengths: Registered byte length for every allocation.
    :ivar row_lengths: Exact byte length of one row for every field.
    :ivar memory_kinds: NIXL memory kind for every field.
    :ivar device_index: CUDA device owning every allocation.
    :ivar row_capacity: Number of independently leased rows.
    """

    fields: tuple[DFlashAuxiliaryField, ...]
    base_addresses: tuple[int, ...]
    allocation_lengths: tuple[int, ...]
    row_lengths: tuple[int, ...]
    memory_kinds: tuple[str, ...]
    device_index: int
    row_capacity: int

    def __post_init__(self) -> None:
        """Validate one exact, canonical, all-VRAM registration receipt."""

        field_count = len(DFLASH_AUXILIARY_FIELDS)
        values = (
            self.fields,
            self.base_addresses,
            self.allocation_lengths,
            self.row_lengths,
            self.memory_kinds,
        )
        if any(type(value) is not tuple for value in values):
            raise TypeError("DFlash auxiliary registration fields must be tuples")
        if any(len(value) != field_count for value in values):
            raise ValueError("DFlash auxiliary registration is incomplete")
        if self.fields != DFLASH_AUXILIARY_FIELDS:
            raise ValueError("DFlash auxiliary registration order is not canonical")
        if any(type(value) is not int or value <= 0 for value in self.base_addresses):
            raise ValueError("DFlash auxiliary base addresses must be positive")
        if any(
            type(value) is not int or value <= 0 for value in self.allocation_lengths
        ):
            raise ValueError("DFlash auxiliary allocation lengths must be positive")
        if any(type(value) is not int or value <= 0 for value in self.row_lengths):
            raise ValueError("DFlash auxiliary row lengths must be positive")
        if any(kind != DFLASH_AUXILIARY_MEMORY_KIND for kind in self.memory_kinds):
            raise ValueError("DFlash auxiliary registration must be entirely VRAM")
        if type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("DFlash auxiliary device index must be non-negative")
        if type(self.row_capacity) is not int or self.row_capacity <= 0:
            raise ValueError("DFlash auxiliary row capacity must be positive")
        for allocation_length, row_length in zip(
            self.allocation_lengths,
            self.row_lengths,
            strict=True,
        ):
            if allocation_length != row_length * self.row_capacity:
                raise ValueError(
                    "DFlash auxiliary allocation length differs from row geometry"
                )

    def memory_descriptors(self) -> list[tuple[int, int, int, str]]:
        """Project the exact canonical NIXL registration descriptors.

        :returns: One ordered device descriptor for every auxiliary field.
        """

        return [
            (base_address, allocation_length, self.device_index, "")
            for base_address, allocation_length in zip(
                self.base_addresses,
                self.allocation_lengths,
                strict=True,
            )
        ]

    def segments_for_row(
        self,
        row_index: int,
    ) -> tuple[PackedAuxiliaryDestinationSegment, ...]:
        """Project one row into exact NIXL segment addresses.

        :param row_index: Reserved row index.
        :returns: Canonically ordered device segments.
        """

        if type(row_index) is not int or not 0 <= row_index < self.row_capacity:
            raise ValueError("DFlash auxiliary row index is outside the pool")
        return tuple(
            PackedAuxiliaryDestinationSegment(
                address=base_address + row_index * row_length,
                item_length=row_length,
            )
            for base_address, row_length in zip(
                self.base_addresses,
                self.row_lengths,
                strict=True,
            )
        )


class DFlashAuxiliaryNixlRegistration:
    """Strong owner of one canonical process-lifetime NIXL registration."""

    _agent: DFlashAuxiliaryMemoryAgent
    _handle: object
    _receipt: DFlashAuxiliaryRegistration

    def __init__(
        self,
        agent: DFlashAuxiliaryMemoryAgent,
        receipt: DFlashAuxiliaryRegistration,
    ) -> None:
        """Register all canonical allocations as NIXL VRAM exactly once.

        The registration is intentionally not exposed through a request-time
        close path. Its handle and backing tensors remain strongly owned by the
        process-level row pool for the serving process lifetime.

        :param agent: Exact NIXL memory-registration owner.
        :param receipt: Canonical stable-allocation geometry.
        """

        if type(receipt) is not DFlashAuxiliaryRegistration:
            raise TypeError("receipt must be DFlashAuxiliaryRegistration")
        handle = agent.register_memory(
            receipt.memory_descriptors(),
            DFLASH_AUXILIARY_MEMORY_KIND,
        )
        if handle is None or (isinstance(handle, list) and len(handle) == 0):
            raise RuntimeError("NIXL returned no DFlash auxiliary registration")
        self._agent = agent
        self._handle = handle
        self._receipt = receipt

    @property
    def receipt(self) -> DFlashAuxiliaryRegistration:
        """Return immutable registered allocation geometry.

        :returns: Canonical all-VRAM registration receipt.
        """

        return self._receipt


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashAuxiliaryRegisteredRow:
    """Independent proof of one exact live row in registered NIXL VRAM.

    :ivar registration: Process-local NIXL registration receipt.
    :ivar reservation: Exact generation-bearing live-row snapshot.
    """

    registration: DFlashAuxiliaryRegistration
    reservation: PackedAuxiliarySlotReservationSnapshot

    def __post_init__(self) -> None:
        """Validate canonical memory kind, generation, and row geometry."""

        if type(self.registration) is not DFlashAuxiliaryRegistration:
            raise TypeError("registration must be DFlashAuxiliaryRegistration")
        if type(self.reservation) is not PackedAuxiliarySlotReservationSnapshot:
            raise TypeError(
                "reservation must be PackedAuxiliarySlotReservationSnapshot"
            )
        expected_segments = self.registration.segments_for_row(
            self.reservation.metadata_buffer_index
        )
        if self.reservation.destination_segments != expected_segments:
            raise ValueError("DFlash registered-row geometry was altered")

    @property
    def field_lengths(self) -> tuple[int, ...]:
        """Return canonically ordered bytes for one transfer row.

        :returns: Exact row byte length for every canonical field.
        """

        return tuple(
            segment.item_length for segment in self.reservation.destination_segments
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashAuxiliaryTransportAccounting:
    """Memory-kind accounting for one canonical auxiliary transfer.

    :ivar field_bytes: Exact bytes transferred for every canonical field.
    :ivar source_vram_bytes: Exact bytes read from registered source VRAM.
    :ivar destination_vram_bytes: Exact bytes written to destination VRAM.
    :ivar vram_transport_bytes: Aggregate device-to-device transport bytes.
    :ivar dram_transport_bytes: Aggregate host-memory transport bytes.
    """

    field_bytes: tuple[tuple[DFlashAuxiliaryField, int], ...]
    source_vram_bytes: int
    destination_vram_bytes: int
    vram_transport_bytes: int
    dram_transport_bytes: int

    def __post_init__(self) -> None:
        """Validate exact conservation and the zero-DRAM invariant."""

        if type(self.field_bytes) is not tuple:
            raise TypeError("field_bytes must be a tuple")
        if tuple(field for field, _ in self.field_bytes) != DFLASH_AUXILIARY_FIELDS:
            raise ValueError("DFlash auxiliary accounting order is not canonical")
        if any(
            type(length) is not int or length <= 0 for _, length in self.field_bytes
        ):
            raise ValueError("DFlash auxiliary field lengths must be positive")
        aggregate = sum(length for _, length in self.field_bytes)
        byte_totals = (
            self.source_vram_bytes,
            self.destination_vram_bytes,
            self.vram_transport_bytes,
        )
        if any(type(value) is not int or value != aggregate for value in byte_totals):
            raise ValueError("DFlash auxiliary VRAM accounting does not conserve bytes")
        if type(self.dram_transport_bytes) is not int:
            raise TypeError("dram_transport_bytes must be an integer")
        if self.dram_transport_bytes != 0:
            raise ValueError("DFlash auxiliary transport must contain zero DRAM bytes")

    @classmethod
    def between(
        cls,
        source: DFlashAuxiliaryRegisteredRow,
        destination: DFlashAuxiliaryRegisteredRow,
    ) -> "DFlashAuxiliaryTransportAccounting":
        """Account across independently registered source and destination rows.

        :param source: Source-process live-row registration proof.
        :param destination: Destination-process live-row registration proof.
        :returns: Exact conserved device-to-device byte accounting.
        """

        if type(source) is not DFlashAuxiliaryRegisteredRow:
            raise TypeError("source must be DFlashAuxiliaryRegisteredRow")
        if type(destination) is not DFlashAuxiliaryRegisteredRow:
            raise TypeError("destination must be DFlashAuxiliaryRegisteredRow")
        if source.field_lengths != destination.field_lengths:
            raise ValueError("DFlash source and destination row geometry differs")
        aggregate = sum(source.field_lengths)
        return cls(
            field_bytes=tuple(
                zip(DFLASH_AUXILIARY_FIELDS, source.field_lengths, strict=True)
            ),
            source_vram_bytes=aggregate,
            destination_vram_bytes=aggregate,
            vram_transport_bytes=aggregate,
            dram_transport_bytes=0,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DFlashAuxiliaryValues:
    """Request-owned destination copies detached from a reusable transport row.

    :ivar topk_probabilities: Copied speculative probabilities.
    :ivar topk_indices: Copied speculative token indices.
    :ivar hidden_states: Copied target hidden-state projection.
    :ivar completion_event: Event proving all three copies are complete.
    """

    topk_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    hidden_states: torch.Tensor
    completion_event: torch.cuda.Event


@dataclasses.dataclass(frozen=True, slots=True)
class _DFlashAuxiliaryReservation:
    """Pool-private generation binding for one live device row."""

    row_index: int
    generation: bytes


_GENERATION_LOCK = threading.Lock()
_ISSUED_GENERATIONS: set[bytes] = set()


def _next_process_generation() -> bytes:
    """Mint and retain one process-unique row generation.

    :returns: Fresh fixed-width generation bytes.
    """

    with _GENERATION_LOCK:
        while True:
            generation = secrets.token_bytes(PACKED_REQUEST_GENERATION_BYTES)
            if generation in _ISSUED_GENERATIONS:
                continue
            _ISSUED_GENERATIONS.add(generation)
            return generation


class DFlashAuxiliaryRowAllocator:
    """Generation-bound row lifecycle over registered immutable geometry."""

    _registration: DFlashAuxiliaryNixlRegistration
    _free_rows: collections.deque[int]
    _active: dict[int, tuple[_DFlashAuxiliaryReservation, object]]
    _quarantined: dict[int, _DFlashAuxiliaryReservation]
    _lock: threading.Lock

    def __init__(self, registration: DFlashAuxiliaryNixlRegistration) -> None:
        """Initialize one row allocator over proven NIXL VRAM.

        :param registration: Exact process-lifetime NIXL registration.
        """

        if type(registration) is not DFlashAuxiliaryNixlRegistration:
            raise TypeError("registration must be DFlashAuxiliaryNixlRegistration")
        self._registration = registration
        self._free_rows = collections.deque(range(registration.receipt.row_capacity))
        self._active = {}
        self._quarantined = {}
        self._lock = threading.Lock()

    @property
    def registration(self) -> DFlashAuxiliaryRegistration:
        """Return process-lifetime row geometry.

        :returns: Immutable all-VRAM registration receipt.
        """

        return self._registration.receipt

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Lease one free row directly to the allocation authority.

        :param owner: Exact allocation-authority reservation owner.
        :returns: Opaque generation-bound reservation.
        """

        if owner is None:
            raise ValueError("auxiliary row owner must not be None")
        with self._lock:
            if len(self._free_rows) == 0:
                raise RuntimeError("DFlash auxiliary VRAM row pool is exhausted")
            row_index = self._free_rows.popleft()
            reservation = _DFlashAuxiliaryReservation(
                row_index=row_index,
                generation=_next_process_generation(),
            )
            self._active[row_index] = (reservation, owner)
            return reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Resolve exact live row identity and device geometry.

        :param reservation: Opaque allocator reservation.
        :returns: Immutable row snapshot consumed by packed transport.
        """

        live = self._require_live_reservation(reservation)
        return PackedAuxiliarySlotReservationSnapshot(
            metadata_buffer_index=live.row_index,
            metadata_slot_generation=live.generation,
            destination_segments=self.registration.segments_for_row(live.row_index),
        )

    def registered_row(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> DFlashAuxiliaryRegisteredRow:
        """Prove one independently registered, exact live transport row.

        :param snapshot: Candidate live-row snapshot.
        :returns: Canonical NIXL VRAM row proof.
        """

        self.require_snapshot(snapshot)
        return DFlashAuxiliaryRegisteredRow(
            registration=self.registration,
            reservation=snapshot,
        )

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return one truly consumed row to the free population.

        :param reservation: Exact live row reservation.
        :param owner: Exact allocation-authority reservation owner.
        """

        with self._lock:
            live = self._require_live_reservation_locked(reservation)
            current = self._active[live.row_index]
            if current[1] is not owner:
                raise RuntimeError("DFlash auxiliary row owner is stale")
            del self._active[live.row_index]
            self._free_rows.append(live.row_index)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Retain one ambiguous row against process-lifetime reuse.

        :param reservation: Exact live row reservation.
        :param owner: Exact allocation-authority reservation owner.
        """

        with self._lock:
            live = self._require_live_reservation_locked(reservation)
            current = self._active[live.row_index]
            if current[1] is not owner:
                raise RuntimeError("DFlash auxiliary row owner is stale")
            del self._active[live.row_index]
            self._quarantined[live.row_index] = live

    def require_snapshot(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> int:
        """Validate a snapshot against the exact live generation.

        :param snapshot: Candidate live-row snapshot.
        :returns: Exact validated row index.
        """

        if type(snapshot) is not PackedAuxiliarySlotReservationSnapshot:
            raise TypeError("snapshot must be PackedAuxiliarySlotReservationSnapshot")
        with self._lock:
            current = self._active.get(snapshot.metadata_buffer_index)
            if current is None:
                raise RuntimeError("DFlash auxiliary row snapshot is stale")
            reservation = current[0]
            if reservation.generation != snapshot.metadata_slot_generation:
                raise RuntimeError("DFlash auxiliary row generation is stale")
            expected_segments = self.registration.segments_for_row(
                reservation.row_index
            )
            if snapshot.destination_segments != expected_segments:
                raise RuntimeError("DFlash auxiliary row geometry was altered")
            return reservation.row_index

    def inventory(self) -> tuple[int, int, int]:
        """Return free, active, and quarantined row counts.

        :returns: Conservation-complete row population.
        """

        with self._lock:
            return (
                len(self._free_rows),
                len(self._active),
                len(self._quarantined),
            )

    def _require_live_reservation(
        self,
        reservation: object,
    ) -> _DFlashAuxiliaryReservation:
        with self._lock:
            return self._require_live_reservation_locked(reservation)

    def _require_live_reservation_locked(
        self,
        reservation: object,
    ) -> _DFlashAuxiliaryReservation:
        if type(reservation) is not _DFlashAuxiliaryReservation:
            raise TypeError("DFlash auxiliary reservation has an invalid type")
        current = self._active.get(reservation.row_index)
        if current is None or current[0] is not reservation:
            raise RuntimeError("DFlash auxiliary reservation is stale")
        return reservation


class DFlashAuxiliaryDeviceRowPool:
    """Generation-bound source or destination rows backed only by CUDA VRAM."""

    _topk_probabilities: torch.Tensor
    _topk_indices: torch.Tensor
    _hidden_states: torch.Tensor
    _registration: DFlashAuxiliaryNixlRegistration
    _row_allocator: DFlashAuxiliaryRowAllocator

    def __init__(
        self,
        agent: DFlashAuxiliaryMemoryAgent,
        row_capacity: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        device: torch.device,
        *,
        topk_capacity: int = 16,
        custom_mem_pool: torch.cuda.MemPool | None = None,
    ) -> None:
        """Allocate and register stable process-lifetime device rows.

        :param agent: Exact NIXL registration owner.
        :param row_capacity: Maximum concurrently leased rows.
        :param hidden_size: Width of one hidden-state row.
        :param hidden_states_dtype: Hidden-state transfer dtype.
        :param device: Exact CUDA device owning the pool.
        :param topk_capacity: Maximum speculative top-k width.
        :param custom_mem_pool: Optional process-owned CUDA memory pool.
        """

        if type(row_capacity) is not int or row_capacity <= 0:
            raise ValueError("row_capacity must be positive")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if type(topk_capacity) is not int or topk_capacity <= 0:
            raise ValueError("topk_capacity must be positive")
        if type(device) is not torch.device:
            raise TypeError("device must be torch.device")
        if device.type != "cuda" or device.index is None:
            raise ValueError("DFlash auxiliary rows require an indexed CUDA device")
        if not isinstance(hidden_states_dtype, torch.dtype):
            raise TypeError("hidden_states_dtype must be torch.dtype")

        allocation_context = (
            torch.cuda.use_mem_pool(custom_mem_pool)
            if custom_mem_pool is not None
            else nullcontext()
        )
        with allocation_context:
            self._topk_probabilities = torch.empty(
                (row_capacity, topk_capacity),
                dtype=torch.float32,
                device=device,
            )
            self._topk_indices = torch.empty(
                (row_capacity, topk_capacity),
                dtype=torch.int64,
                device=device,
            )
            self._hidden_states = torch.empty(
                (row_capacity, hidden_size),
                dtype=hidden_states_dtype,
                device=device,
            )
        tensors = (
            self._topk_probabilities,
            self._topk_indices,
            self._hidden_states,
        )
        receipt = DFlashAuxiliaryRegistration(
            fields=DFLASH_AUXILIARY_FIELDS,
            base_addresses=tuple(tensor.data_ptr() for tensor in tensors),
            allocation_lengths=tuple(tensor.nbytes for tensor in tensors),
            row_lengths=tuple(tensor[0].nbytes for tensor in tensors),
            memory_kinds=(DFLASH_AUXILIARY_MEMORY_KIND,) * len(tensors),
            device_index=device.index,
            row_capacity=row_capacity,
        )
        self._registration = DFlashAuxiliaryNixlRegistration(agent, receipt)
        self._row_allocator = DFlashAuxiliaryRowAllocator(self._registration)

    @property
    def registration(self) -> DFlashAuxiliaryRegistration:
        """Return process-lifetime NIXL registration geometry.

        :returns: Immutable all-VRAM registration receipt.
        """

        return self._registration.receipt

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Lease one free row directly to the allocation authority.

        :param owner: Exact allocation-authority reservation owner.
        :returns: Opaque generation-bound reservation.
        """

        return self._row_allocator.allocate_packed_auxiliary_slot(owner)

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Resolve exact live row identity and device geometry.

        :param reservation: Opaque pool reservation.
        :returns: Immutable row snapshot consumed by packed transport.
        """

        return self._row_allocator.packed_auxiliary_slot_reservation_snapshot(
            reservation
        )

    def registered_row(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
    ) -> DFlashAuxiliaryRegisteredRow:
        """Prove one exact live row in process-registered NIXL VRAM.

        :param snapshot: Candidate live-row snapshot.
        :returns: Canonical registered-row proof.
        """

        return self._row_allocator.registered_row(snapshot)

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return one truly consumed row to the free population.

        :param reservation: Exact live row reservation.
        :param owner: Exact allocation-authority reservation owner.
        """

        self._row_allocator.release_packed_auxiliary_slot(reservation, owner)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Retain one ambiguous row against process-lifetime reuse.

        :param reservation: Exact live row reservation.
        :param owner: Exact allocation-authority reservation owner.
        """

        self._row_allocator.quarantine_packed_auxiliary_slot(reservation, owner)

    def enqueue_source_projection(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
        *,
        topk_probabilities: torch.Tensor,
        topk_indices: torch.Tensor,
        hidden_states: torch.Tensor,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        """Copy one source result into stable rows and record its producer event.

        The supplied stream must already depend on the model forward that
        produced the three inputs. Recording the event here makes the exact row
        writes, rather than mutable scheduler state, the transport boundary.

        :param snapshot: Exact live source-row generation.
        :param topk_probabilities: Request-local CUDA top-k probabilities.
        :param topk_indices: Request-local CUDA top-k indices.
        :param hidden_states: Request-local CUDA hidden-state projection.
        :param stream: Producing CUDA stream after model submission.
        :param producer_event: Event recorded after every row copy.
        """

        row_index = self._row_allocator.require_snapshot(snapshot)
        self._validate_source_tensor(
            topk_probabilities,
            self._topk_probabilities[row_index],
            DFlashAuxiliaryField.TOPK_PROBABILITIES,
        )
        self._validate_source_tensor(
            topk_indices,
            self._topk_indices[row_index],
            DFlashAuxiliaryField.TOPK_INDICES,
        )
        self._validate_source_tensor(
            hidden_states,
            self._hidden_states[row_index],
            DFlashAuxiliaryField.HIDDEN_STATES,
        )
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be torch.cuda.Stream")
        if not isinstance(producer_event, torch.cuda.Event):
            raise TypeError("producer_event must be torch.cuda.Event")
        if stream.device_index != self.registration.device_index:
            raise ValueError("producer stream belongs to another CUDA device")

        with torch.cuda.stream(stream):
            self._copy_row(self._topk_probabilities[row_index], topk_probabilities)
            self._copy_row(self._topk_indices[row_index], topk_indices)
            self._copy_row(self._hidden_states[row_index], hidden_states)
            producer_event.record(stream)

    def enqueue_destination_copy(
        self,
        snapshot: PackedAuxiliarySlotReservationSnapshot,
        *,
        stream: torch.cuda.Stream,
    ) -> DFlashAuxiliaryValues:
        """Detach one received row and return an event proving safe reuse.

        The allocation authority must not release ``snapshot`` until the
        returned event has been delivered through the native completion owner.

        :param snapshot: Exact live destination-row generation.
        :param stream: Scheduler-authorized CUDA stream for request adoption.
        :returns: Request-owned tensors and the true-copy completion event.
        """

        row_index = self._row_allocator.require_snapshot(snapshot)
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be torch.cuda.Stream")
        if stream.device_index != self.registration.device_index:
            raise ValueError("destination stream belongs to another CUDA device")
        with torch.cuda.stream(stream):
            topk_probabilities = self._topk_probabilities[row_index].clone()
            topk_indices = self._topk_indices[row_index].clone()
            hidden_states = self._hidden_states[row_index].clone()
            completion_event = torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=False,
            )
            completion_event.record(stream)
        return DFlashAuxiliaryValues(
            topk_probabilities=topk_probabilities,
            topk_indices=topk_indices,
            hidden_states=hidden_states,
            completion_event=completion_event,
        )

    def inventory(self) -> tuple[int, int, int]:
        """Return free, active, and quarantined row counts.

        :returns: Conservation-complete row population.
        """

        return self._row_allocator.inventory()

    def _validate_source_tensor(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        field: DFlashAuxiliaryField,
    ) -> None:
        if type(source) is not torch.Tensor:
            raise TypeError(f"{field.value} must be a torch.Tensor")
        if source.device.type != "cuda":
            raise ValueError(f"{field.value} must reside in CUDA memory")
        if source.device.index != self.registration.device_index:
            raise ValueError(f"{field.value} belongs to another CUDA device")
        if source.dtype is not destination.dtype:
            raise ValueError(f"{field.value} dtype differs from its stable row")
        if source.numel() > destination.numel():
            raise ValueError(f"{field.value} exceeds its stable row capacity")

    @staticmethod
    def _copy_row(destination: torch.Tensor, source: torch.Tensor) -> None:
        destination.zero_()
        destination.reshape(-1)[: source.numel()].copy_(
            source.reshape(-1),
            non_blocking=True,
        )
