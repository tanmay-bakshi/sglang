import dataclasses

import pytest
import torch
from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
)
from sglang.srt.disaggregation.terminal_progress import dflash_auxiliary
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFLASH_AUXILIARY_FIELDS,
    DFLASH_AUXILIARY_MEMORY_KIND,
    DFlashAuxiliaryField,
    DFlashAuxiliaryNixlRegistration,
    DFlashAuxiliaryRegisteredRow,
    DFlashAuxiliaryRegistration,
    DFlashAuxiliaryRowAllocator,
    DFlashAuxiliaryTransportAccounting,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingMemoryAgent:
    """CPU-only exact NIXL registration recorder."""

    calls: list[tuple[list[tuple[int, int, int, str]], str]]
    handle: object

    def __init__(self, handle: object | None = None) -> None:
        """Initialize a deterministic registration agent.

        :param handle: Registration handle returned to the caller.
        """

        self.calls = []
        self.handle = object() if handle is None else handle

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Record one registration and return the configured handle.

        :param addresses: Ordered address descriptors.
        :param memory_kind: Requested NIXL memory kind.
        :returns: Configured opaque handle.
        """

        self.calls.append((addresses, memory_kind))
        return self.handle


def _registration(
    *,
    address_offset: int = 0,
    row_lengths: tuple[int, int, int] = (64, 128, 512),
    memory_kinds: tuple[str, str, str] = ("VRAM", "VRAM", "VRAM"),
) -> DFlashAuxiliaryRegistration:
    """Build deterministic CPU-only NIXL geometry.

    :param address_offset: Offset separating independent process addresses.
    :param row_lengths: Exact canonical field row lengths.
    :param memory_kinds: Exact canonical field memory kinds.
    :returns: Valid three-field auxiliary registration receipt.
    """

    row_capacity = 8
    return DFlashAuxiliaryRegistration(
        fields=DFLASH_AUXILIARY_FIELDS,
        base_addresses=(
            0x1000 + address_offset,
            0x2000 + address_offset,
            0x4000 + address_offset,
        ),
        allocation_lengths=tuple(
            row_length * row_capacity for row_length in row_lengths
        ),
        row_lengths=row_lengths,
        memory_kinds=memory_kinds,
        device_index=3,
        row_capacity=row_capacity,
    )


def _allocator(
    *,
    address_offset: int = 0,
    row_lengths: tuple[int, int, int] = (64, 128, 512),
) -> tuple[
    _RecordingMemoryAgent,
    DFlashAuxiliaryRowAllocator,
]:
    """Build one independently NIXL-registered CPU test allocator.

    :param address_offset: Offset separating independent process addresses.
    :param row_lengths: Exact canonical field row lengths.
    :returns: Recording agent and generation-bound allocator.
    """

    agent = _RecordingMemoryAgent()
    registration = DFlashAuxiliaryNixlRegistration(
        agent,
        _registration(address_offset=address_offset, row_lengths=row_lengths),
    )
    return agent, DFlashAuxiliaryRowAllocator(registration)


def _registered_row(
    allocator: DFlashAuxiliaryRowAllocator,
    owner: object,
) -> tuple[object, DFlashAuxiliaryRegisteredRow]:
    """Lease and prove one exact live test row.

    :param allocator: Owning registered row allocator.
    :param owner: Exact reservation owner.
    :returns: Opaque reservation and registered-row proof.
    """

    reservation = allocator.allocate_packed_auxiliary_slot(owner)
    snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
    return reservation, allocator.registered_row(snapshot)


def test_canonical_fields_are_exact_and_ordered() -> None:
    """The wire-facing DFlash names cannot drift or reorder."""

    assert tuple(field.value for field in DFLASH_AUXILIARY_FIELDS) == (
        "output_topk_p",
        "output_topk_index",
        "output_hidden_states",
    )


def test_registration_projects_exact_canonical_vram_segments() -> None:
    """One row projection preserves field order and address arithmetic."""

    registration = _registration()

    segments = registration.segments_for_row(3)

    assert tuple(segment.address for segment in segments) == (
        0x1000 + 3 * 64,
        0x2000 + 3 * 128,
        0x4000 + 3 * 512,
    )
    assert tuple(segment.item_length for segment in segments) == (64, 128, 512)
    assert registration.memory_kinds == ("VRAM", "VRAM", "VRAM")


def test_nixl_registration_uses_one_exact_all_vram_descriptor_set() -> None:
    """The typed boundary registers canonical allocations directly as VRAM."""

    receipt = _registration()
    agent = _RecordingMemoryAgent()

    registration = DFlashAuxiliaryNixlRegistration(agent, receipt)

    assert registration.receipt is receipt
    assert agent.calls == [
        (
            [
                (0x1000, 512, 3, ""),
                (0x2000, 1024, 3, ""),
                (0x4000, 4096, 3, ""),
            ],
            DFLASH_AUXILIARY_MEMORY_KIND,
        )
    ]


@pytest.mark.parametrize("handle", [None, []], ids=("none", "empty"))
def test_nixl_registration_requires_an_owned_handle(handle: object) -> None:
    """An absent native registration cannot back row allocation."""

    agent = _RecordingMemoryAgent()
    agent.handle = handle

    with pytest.raises(RuntimeError, match="returned no DFlash"):
        DFlashAuxiliaryNixlRegistration(agent, _registration())


def test_registration_rejects_host_auxiliary_memory() -> None:
    """A DRAM descriptor cannot enter the functional DFlash payload."""

    with pytest.raises(ValueError, match="entirely VRAM"):
        _registration(memory_kinds=("VRAM", "DRAM", "VRAM"))


def test_registration_rejects_noncanonical_fields_and_row_geometry() -> None:
    """Field order and allocation-to-row geometry are exact contracts."""

    receipt = _registration()
    with pytest.raises(ValueError, match="order is not canonical"):
        dataclasses.replace(receipt, fields=tuple(reversed(receipt.fields)))
    with pytest.raises(ValueError, match="differs from row geometry"):
        dataclasses.replace(
            receipt,
            allocation_lengths=(513, 1024, 4096),
        )


def test_row_allocator_exhaustion_and_release_conserve_population() -> None:
    """Every physical row remains free, active, or quarantined exactly once."""

    _, allocator = _allocator()
    owner = object()
    reservations = tuple(
        allocator.allocate_packed_auxiliary_slot(owner) for _ in range(8)
    )

    assert allocator.inventory() == (0, 8, 0)
    with pytest.raises(RuntimeError, match="pool is exhausted"):
        allocator.allocate_packed_auxiliary_slot(owner)

    allocator.release_packed_auxiliary_slot(reservations[0], owner)

    assert allocator.inventory() == (1, 7, 0)


def test_row_allocator_changes_generation_before_every_reuse() -> None:
    """Repeated reuse of one physical row never creates an ABA replay."""

    _, allocator = _allocator()
    owner = object()
    seen_generations: set[bytes] = set()
    snapshots_by_row: dict[int, PackedAuxiliarySlotReservationSnapshot] = {}

    for _ in range(128):
        reservation = allocator.allocate_packed_auxiliary_slot(owner)
        snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
        assert len(snapshot.metadata_slot_generation) == PACKED_REQUEST_GENERATION_BYTES
        assert snapshot.metadata_slot_generation not in seen_generations
        previous = snapshots_by_row.get(snapshot.metadata_buffer_index)
        if previous is not None:
            assert (
                snapshot.metadata_slot_generation != previous.metadata_slot_generation
            )
        seen_generations.add(snapshot.metadata_slot_generation)
        snapshots_by_row[snapshot.metadata_buffer_index] = snapshot
        allocator.release_packed_auxiliary_slot(reservation, owner)


def test_stale_reservation_owner_generation_and_geometry_fail_closed() -> None:
    """Every reusable-row identity dimension is independently authoritative."""

    _, allocator = _allocator()
    owner = object()
    reservation = allocator.allocate_packed_auxiliary_slot(owner)
    snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)

    with pytest.raises(RuntimeError, match="owner is stale"):
        allocator.release_packed_auxiliary_slot(reservation, object())
    with pytest.raises(RuntimeError, match="generation is stale"):
        allocator.require_snapshot(
            dataclasses.replace(
                snapshot,
                metadata_slot_generation=b"x" * PACKED_REQUEST_GENERATION_BYTES,
            )
        )
    altered_segments = list(snapshot.destination_segments)
    altered_segments[0] = PackedAuxiliaryDestinationSegment(
        address=altered_segments[0].address,
        item_length=altered_segments[0].item_length + 1,
    )
    with pytest.raises(RuntimeError, match="geometry was altered"):
        allocator.require_snapshot(
            dataclasses.replace(
                snapshot,
                destination_segments=tuple(altered_segments),
            )
        )

    allocator.release_packed_auxiliary_slot(reservation, owner)

    with pytest.raises(RuntimeError, match="reservation is stale"):
        allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
    with pytest.raises(RuntimeError, match="snapshot is stale"):
        allocator.require_snapshot(snapshot)
    assert allocator.inventory() == (8, 0, 0)


def test_row_quarantine_removes_capacity_permanently() -> None:
    """An ambiguous device row never returns to the free population."""

    _, allocator = _allocator()
    owner = object()
    reservation = allocator.allocate_packed_auxiliary_slot(owner)
    snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)

    allocator.quarantine_packed_auxiliary_slot(reservation, owner)

    assert allocator.inventory() == (7, 0, 1)
    remaining = tuple(allocator.allocate_packed_auxiliary_slot(owner) for _ in range(7))
    remaining_rows = {
        allocator.packed_auxiliary_slot_reservation_snapshot(item).metadata_buffer_index
        for item in remaining
    }
    assert snapshot.metadata_buffer_index not in remaining_rows
    with pytest.raises(RuntimeError, match="pool is exhausted"):
        allocator.allocate_packed_auxiliary_slot(owner)
    with pytest.raises(RuntimeError, match="reservation is stale"):
        allocator.packed_auxiliary_slot_reservation_snapshot(reservation)


def test_cross_pool_accounting_accepts_independent_matching_vram_rows() -> None:
    """Source and destination prove VRAM separately before byte conservation."""

    source_agent, source_allocator = _allocator(address_offset=0x10000)
    destination_agent, destination_allocator = _allocator(address_offset=0x20000)
    source_owner = object()
    destination_owner = object()
    _, source = _registered_row(source_allocator, source_owner)
    _, destination = _registered_row(destination_allocator, destination_owner)

    accounting = DFlashAuxiliaryTransportAccounting.between(source, destination)

    assert source.registration.memory_kinds == ("VRAM", "VRAM", "VRAM")
    assert destination.registration.memory_kinds == ("VRAM", "VRAM", "VRAM")
    assert source_agent.calls[0][1] == DFLASH_AUXILIARY_MEMORY_KIND
    assert destination_agent.calls[0][1] == DFLASH_AUXILIARY_MEMORY_KIND
    assert accounting.field_bytes == (
        (DFlashAuxiliaryField.TOPK_PROBABILITIES, 64),
        (DFlashAuxiliaryField.TOPK_INDICES, 128),
        (DFlashAuxiliaryField.HIDDEN_STATES, 512),
    )
    assert accounting.source_vram_bytes == 704
    assert accounting.destination_vram_bytes == 704
    assert accounting.vram_transport_bytes == 704
    assert accounting.dram_transport_bytes == 0


def test_cross_pool_accounting_rejects_mismatched_row_geometry() -> None:
    """Equal field order cannot conceal a source/destination byte mismatch."""

    _, source_allocator = _allocator(row_lengths=(64, 128, 512))
    _, destination_allocator = _allocator(
        address_offset=0x10000,
        row_lengths=(64, 136, 512),
    )
    _, source = _registered_row(source_allocator, object())
    _, destination = _registered_row(destination_allocator, object())

    with pytest.raises(ValueError, match="row geometry differs"):
        DFlashAuxiliaryTransportAccounting.between(source, destination)


def test_registered_row_rejects_altered_snapshot_geometry() -> None:
    """A row proof cannot be reconstructed from unrelated registered geometry."""

    _, allocator = _allocator()
    _, row = _registered_row(allocator, object())
    other_registration = _registration(address_offset=0x10000)

    with pytest.raises(ValueError, match="geometry was altered"):
        DFlashAuxiliaryRegisteredRow(
            registration=other_registration,
            reservation=row.reservation,
        )


def test_transport_accounting_requires_zero_dram_and_exact_conservation() -> None:
    """Accounting refuses host fallback or inconsistent aggregate bytes."""

    field_bytes = (
        (DFlashAuxiliaryField.TOPK_PROBABILITIES, 64),
        (DFlashAuxiliaryField.TOPK_INDICES, 128),
        (DFlashAuxiliaryField.HIDDEN_STATES, 512),
    )
    valid = {
        "field_bytes": field_bytes,
        "source_vram_bytes": 704,
        "destination_vram_bytes": 704,
        "vram_transport_bytes": 704,
        "dram_transport_bytes": 0,
    }

    with pytest.raises(ValueError, match="zero DRAM bytes"):
        DFlashAuxiliaryTransportAccounting(**{**valid, "dram_transport_bytes": 1})
    with pytest.raises(ValueError, match="does not conserve bytes"):
        DFlashAuxiliaryTransportAccounting(**{**valid, "destination_vram_bytes": 703})


def test_device_pool_requires_real_indexed_cuda_device() -> None:
    """Construction cannot silently allocate CPU rows in a CUDA-less process."""

    with pytest.raises(ValueError, match="indexed CUDA device"):
        dflash_auxiliary.DFlashAuxiliaryDeviceRowPool(
            _RecordingMemoryAgent(),
            row_capacity=1,
            hidden_size=64,
            hidden_states_dtype=torch.float16,
            device=torch.device("cpu"),
        )
