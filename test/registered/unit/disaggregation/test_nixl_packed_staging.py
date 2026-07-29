import dataclasses

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedLayoutSpec,
    PackedPrepare,
    PackedReady,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingEndpointBufferBinding,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedComponentPages,
    PackedDestinationRegistration,
    PackedIntervalLeaseAllocator,
    PackedReadyCoordinator,
    PackedReadyError,
    active_destination_page_arrays,
    build_component_buffer_registry,
    build_decode_spec,
    build_prefill_chunk,
    writer_layout_for,
)

SWA_COMPONENT = StagingComponentId(state_index=0, state_type=StateType.SWA)
KEY = PackedChunkKey(room_id=29, chunk_id=2)
WRITERS = tuple(
    StagingWriterId(
        transfer_source_rank=rank,
        source_attn_tp_rank=rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )
    for rank in range(2)
)


def _kv_args(*, source: bool) -> KVArgs:
    """Build CPU-only source or destination registration metadata.

    :param source: Whether to build TP2 source rather than TP1 destination data.
    :returns: Fake registered KV metadata.
    """

    kv_args = KVArgs()
    main_item_len = 32 if source else 64
    state_item_len = 16 if source else 32
    kv_args.kv_data_ptrs = [0x100000, 0x110000]
    kv_args.kv_data_lens = [main_item_len * 32, main_item_len * 32]
    kv_args.kv_item_lens = [main_item_len, main_item_len]
    kv_args.kv_layer_ids = [5, 5]
    kv_args.state_types = [StateType.SWA]
    kv_args.state_data_ptrs = [[0x200000, 0x210000]]
    kv_args.state_data_lens = [[state_item_len * 32, state_item_len * 32]]
    kv_args.state_item_lens = [[state_item_len, state_item_len]]
    kv_args.state_layer_ids = [[1, 1]]
    kv_args.page_size = 4
    return kv_args


def _destination_registration() -> PackedDestinationRegistration:
    """Build the decode geometry advertised to source writers.

    :returns: TP1 destination registration.
    """

    destination = _kv_args(source=False)
    return PackedDestinationRegistration(
        main_item_lens=tuple(destination.kv_item_lens),
        main_layer_ids=tuple(destination.kv_layer_ids),
        state_item_lens=tuple(
            tuple(item_lens) for item_lens in destination.state_item_lens
        ),
        state_layer_ids=tuple(
            tuple(layer_ids) for layer_ids in destination.state_layer_ids
        ),
        page_size=destination.page_size,
    )


def _component_pages(
    component_id: StagingComponentId,
) -> PackedComponentPages:
    """Build two source and destination pages for one component.

    :param component_id: Main KV or SWA component identity.
    :returns: Immutable component page projections.
    """

    return PackedComponentPages(
        component_id=component_id,
        source_pages=np.asarray((3, 4), dtype=np.int32),
        destination_pages=np.asarray((7, 8), dtype=np.int32),
        destination_index_offset=0,
    )


def _prefill_chunk(
    component_ids: tuple[StagingComponentId, ...],
    *,
    is_last: bool,
) -> tuple[PackedLayoutSpec, StagingEndpointBufferBinding]:
    """Build one source-authored packed chunk.

    :param component_ids: Active component identities.
    :param is_last: Whether the chunk completes the room.
    :returns: Canonical spec and source binding.
    """

    return build_prefill_chunk(
        key=KEY,
        is_last=is_last,
        kv_args=_kv_args(source=True),
        destination_registration=_destination_registration(),
        components=tuple(
            _component_pages(component_id) for component_id in component_ids
        ),
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
        writers=WRITERS,
    )


def _decode_spec(
    spans: tuple[StagingComponentSpan, ...],
    *,
    is_last: bool,
    required_final_components: frozenset[StagingComponentId],
) -> PackedLayoutSpec:
    """Build trusted decode-local canonical layout input.

    :param spans: Room-derived active component spans.
    :param is_last: Room-derived final marker.
    :param required_final_components: Required non-empty final state.
    :returns: Decode-local packed spec.
    """

    return build_decode_spec(
        chunk_id=KEY.chunk_id,
        is_last=is_last,
        spans=spans,
        kv_args=_kv_args(source=False),
        expected_writers=WRITERS,
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
        required_final_components=required_final_components,
    )


@pytest.mark.parametrize(
    ("component_ids", "is_last"),
    (
        ((MAIN_KV_COMPONENT,), False),
        ((MAIN_KV_COMPONENT, SWA_COMPONENT), True),
        ((SWA_COMPONENT,), True),
    ),
)
def test_decode_rebuilds_supported_source_layouts(
    component_ids: tuple[StagingComponentId, ...],
    is_last: bool,
) -> None:
    """Trusted decode metadata reproduces every supported source shape."""

    source_spec, _ = _prefill_chunk(component_ids, is_last=is_last)
    required = (
        frozenset({SWA_COMPONENT})
        if is_last and SWA_COMPONENT in component_ids
        else frozenset()
    )

    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=is_last,
        required_final_components=required,
    )

    assert decode_spec.build() == source_spec.build()


def test_decode_rejects_final_chunk_omitting_required_swa() -> None:
    """A source cannot mark main-only data final while required SWA is absent."""

    source_spec, _ = _prefill_chunk((MAIN_KV_COMPONENT,), is_last=True)

    with pytest.raises(ValueError, match="omits required components"):
        _decode_spec(
            source_spec.spans,
            is_last=True,
            required_final_components=frozenset({SWA_COMPONENT}),
        )


def test_decode_geometry_is_derived_without_trusting_source_metadata() -> None:
    """Source-advertised byte geometry cannot become decode canonical truth."""

    source_spec, _ = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    forged_source_geometry = dataclasses.replace(
        source_spec.source_components[0],
        item_lens=(64, 64),
    )
    forged = dataclasses.replace(
        source_spec,
        source_components=(
            forged_source_geometry,
            source_spec.source_components[1],
        ),
    )

    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=True,
        required_final_components=frozenset({SWA_COMPONENT}),
    )

    assert forged.source_components != decode_spec.source_components


def test_destination_binding_rejects_page_capacity_overflow() -> None:
    """Decode registration rejects a destination page outside GPU allocation."""

    source_spec, _ = _prefill_chunk((MAIN_KV_COMPONENT,), is_last=False)
    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=False,
        required_final_components=frozenset(),
    )
    destination = _kv_args(source=False)
    page_arrays = active_destination_page_arrays(
        destination,
        np.asarray((31, 32), dtype=np.int32),
        None,
    )
    registry = build_component_buffer_registry(destination, page_arrays)
    protocol = PackedDecodeProtocol(
        PackedIntervalLeaseAllocator(
            base_address=0x800000,
            total_size=1 << 20,
        )
    )

    with pytest.raises(ValueError, match="exceeds registered page capacity"):
        protocol.register_chunk(KEY, decode_spec, registry)


def test_component_pages_require_exact_int32_arrays() -> None:
    """Page snapshots reject implicit narrowing and wraparound."""

    with pytest.raises(TypeError, match="dtype int32"):
        PackedComponentPages(
            component_id=MAIN_KV_COMPONENT,
            source_pages=np.asarray((3, 4), dtype=np.int64),
            destination_pages=np.asarray((7, 8), dtype=np.int32),
            destination_index_offset=0,
        )


def test_interval_allocator_coalesces_released_regions() -> None:
    """Released adjacent intervals become one reusable contiguous lease."""

    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1024,
    )
    first = allocator.allocate(1)
    second = allocator.allocate(257)

    assert first.base_address == 0x800000
    assert first.length_bytes == 256
    assert second.base_address == 0x800100
    assert second.length_bytes == 512

    allocator.release(second)
    allocator.release(first)

    whole = allocator.allocate(1024)
    assert whole.base_address == 0x800000
    assert whole.length_bytes == 1024


def test_interval_allocator_does_not_reuse_quarantined_region() -> None:
    """Quarantine retains ownership until the protocol explicitly releases it."""

    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1024,
    )
    quarantined = allocator.allocate(256)
    remainder = allocator.allocate(768)
    allocator.quarantine(quarantined, "DMA may still target the lease")
    allocator.release(remainder)

    with pytest.raises(MemoryError):
        allocator.allocate(1024)

    allocator.release(quarantined)
    assert allocator.allocate(1024).base_address == 0x800000


def test_interval_allocator_requires_aligned_registered_base() -> None:
    """Relative alignment cannot repair a misaligned registered base pointer."""

    with pytest.raises(ValueError, match="base_address must satisfy"):
        PackedIntervalLeaseAllocator(
            base_address=0x800001,
            total_size=1024,
        )


def _registered_ready() -> tuple[
    PackedReadyCoordinator,
    PackedPrepare,
    PackedReady,
]:
    """Register one source chunk and build its valid READY.

    :returns: Coordinator, PREPARE, and canonical READY.
    """

    spec, source_binding = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    coordinator = PackedReadyCoordinator()
    prepare = coordinator.register_chunk(
        key=KEY,
        decode_peer_name="decode-agent",
        destination_gpu_id=6,
        writer_id=WRITERS[0],
        spec=spec,
        source_binding=source_binding,
    )
    writer_layout = writer_layout_for(spec.build(), WRITERS[0])
    ready = PackedReady(
        key=KEY,
        writer_id=WRITERS[0],
        digest=spec.build().digest,
        lease_id=71,
        lease_base_address=0xA00000,
        projection_offset=writer_layout.lease_offset,
        projection_length=writer_layout.length_bytes,
    )
    return coordinator, prepare, ready


def test_ready_coordinator_returns_only_canonical_transfer_shape() -> None:
    """Validated READY produces local gather shape and matching COMMIT metadata."""

    coordinator, prepare, ready = _registered_ready()

    transfer = coordinator.handle_ready(ready, "decode-agent")

    assert isinstance(prepare, PackedPrepare)
    assert transfer.destination_address == (
        ready.lease_base_address + ready.projection_offset
    )
    assert (
        transfer.length_bytes
        == writer_layout_for(
            transfer.layout,
            transfer.writer_id,
        ).length_bytes
    )
    assert transfer.decode_peer_name == "decode-agent"
    assert transfer.destination_gpu_id == 6
    assert transfer.commit.key == KEY
    assert transfer.commit.writer_id == WRITERS[0]
    assert transfer.commit.digest == transfer.layout.digest
    assert transfer.commit.lease_id == ready.lease_id


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("writer_id", WRITERS[1], "writer identity"),
        ("digest", b"\xff" * 32, "digest"),
        ("projection_offset", 256, "projection offset"),
        ("projection_length", 1, "projection length"),
        (
            "lease_base_address",
            (1 << 64) - 1,
            "exceeds the uint64 address space",
        ),
    ),
)
def test_ready_coordinator_rejects_forged_fields_without_consuming_chunk(
    field: str,
    value: object,
    match: str,
) -> None:
    """A rejected READY cannot alter later canonical DMA work."""

    coordinator, _, ready = _registered_ready()
    forged = dataclasses.replace(ready, **{field: value})

    with pytest.raises(PackedReadyError, match=match):
        coordinator.handle_ready(forged, "decode-agent")

    assert coordinator.handle_ready(ready, "decode-agent").commit.lease_id == 71


def test_ready_coordinator_authenticates_decode_route() -> None:
    """READY from another decode peer cannot select a destination address."""

    coordinator, _, ready = _registered_ready()

    with pytest.raises(PackedReadyError, match="route is not pending"):
        coordinator.handle_ready(ready, "other-decode-agent")

    assert coordinator.handle_ready(ready, "decode-agent").length_bytes > 0


def test_ready_coordinator_tracks_same_chunk_per_decode_route() -> None:
    """One source rank can independently serve multiple destination TP ranks."""

    spec, source_binding = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    coordinator = PackedReadyCoordinator()
    for peer_name in ("decode-agent-0", "decode-agent-1"):
        coordinator.register_chunk(
            key=KEY,
            decode_peer_name=peer_name,
            destination_gpu_id=6,
            writer_id=WRITERS[0],
            spec=spec,
            source_binding=source_binding,
        )
    writer_layout = writer_layout_for(spec.build(), WRITERS[0])
    ready = PackedReady(
        key=KEY,
        writer_id=WRITERS[0],
        digest=spec.build().digest,
        lease_id=71,
        lease_base_address=0xA00000,
        projection_offset=writer_layout.lease_offset,
        projection_length=writer_layout.length_bytes,
    )

    first = coordinator.handle_ready(ready, "decode-agent-0")
    second = coordinator.handle_ready(
        dataclasses.replace(
            ready,
            lease_id=72,
            lease_base_address=0xB00000,
        ),
        "decode-agent-1",
    )

    assert first.decode_peer_name == "decode-agent-0"
    assert first.commit.lease_id == 71
    assert second.decode_peer_name == "decode-agent-1"
    assert second.commit.lease_id == 72


def test_ready_coordinator_never_hands_out_duplicate_dma_work() -> None:
    """An accepted READY is consumed before any duplicate can post another DMA."""

    coordinator, _, ready = _registered_ready()
    coordinator.handle_ready(ready, "decode-agent")

    with pytest.raises(PackedReadyError, match="not pending"):
        coordinator.handle_ready(ready, "decode-agent")
