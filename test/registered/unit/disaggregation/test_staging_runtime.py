import dataclasses
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.staging_layout import (
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
    build_staging_chunk_layout,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBuffer,
    StagingComponentBufferRegistry,
    StagingEndpoint,
    StagingEndpointBufferBinding,
    bind_staging_endpoint_buffers,
)
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAIN_KV = StagingComponentId(state_index=None, state_type=None)
SWA = StagingComponentId(state_index=0, state_type=StateType.SWA)
DSA_AT_SWA_INDEX = StagingComponentId(state_index=0, state_type=StateType.DSA)
WRITER = StagingWriterId(
    transfer_source_rank=0,
    source_attn_tp_rank=0,
    source_pp_rank=0,
    source_cp_rank=0,
)


class _SWAStateBufferPool:
    """CPU-only SWA sub-pool exposing K-then-V buffer metadata."""

    def get_contiguous_buf_infos(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return four fake registered tensor entries.

        :returns: Pointers, allocation lengths, and page item lengths.
        """

        return [11, 12, 13, 14], [1024, 1024, 1024, 1024], [64, 64, 64, 64]


def _geometry(
    component_id: StagingComponentId,
    *,
    page_size: int = 4,
    item_lens: tuple[int, ...] = (64, 64, 64, 64),
    layer_ids: tuple[int, ...] = (2, 5, 2, 5),
) -> StagingComponentGeometry:
    """Create one test component geometry.

    :param component_id: Main-KV or state identity.
    :param page_size: Tokens per page.
    :param item_lens: Bytes per registered tensor page.
    :param layer_ids: Global layer identifiers in registration order.
    :returns: Immutable test geometry.
    """

    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=item_lens,
        layer_ids=layer_ids,
        page_size=page_size,
    )


def _buffer(
    geometry: StagingComponentGeometry,
    page_array: tuple[int, ...],
    *,
    pointer_base: int = 0x100000,
    page_capacity: int = 256,
) -> StagingComponentBuffer:
    """Create one CPU-only component buffer registration.

    :param geometry: Registration geometry.
    :param page_array: Request-local page indices.
    :param pointer_base: First fake tensor pointer.
    :param page_capacity: Complete pages allocated in every fake tensor.
    :returns: Valid component buffer.
    """

    tensor_ptrs = tuple(
        pointer_base + entry_index * 0x1000
        for entry_index in range(len(geometry.item_lens))
    )
    return StagingComponentBuffer(
        component_id=geometry.component_id,
        tensor_ptrs=tensor_ptrs,
        data_lens=tuple(item_len * page_capacity for item_len in geometry.item_lens),
        item_lens=geometry.item_lens,
        layer_ids=geometry.layer_ids,
        page_size=geometry.page_size,
        page_array=np.asarray(page_array, dtype=np.int32),
    )


def _layout(
    spans: tuple[StagingComponentSpan, ...],
    source_components: tuple[StagingComponentGeometry, ...],
    destination_components: tuple[StagingComponentGeometry, ...] | None = None,
) -> StagingChunkLayout:
    """Build a TP1 packed layout for registry tests.

    :param spans: Active component spans.
    :param source_components: Source geometries.
    :param destination_components: Destination geometries, or source geometries.
    :returns: Immutable packed layout.
    """

    destination = (
        source_components if destination_components is None else destination_components
    )
    return build_staging_chunk_layout(
        chunk_id=0,
        is_last=True,
        spans=spans,
        source_components=source_components,
        destination_components=destination,
        source_tp_size=1,
        destination_tp_size=1,
        destination_tp_rank=0,
        writers=(WRITER,),
    )


def _span(
    component_id: StagingComponentId,
    *,
    source_offset: int = 0,
    destination_offset: int = 0,
    logical_tokens: int = 5,
    physical_tokens: int = 8,
) -> StagingComponentSpan:
    """Create one page-rounded component span.

    :param component_id: Active component identity.
    :param source_offset: Source component-local page offset.
    :param destination_offset: Destination component-local page offset.
    :param logical_tokens: Attention-visible tokens.
    :param physical_tokens: Page-rounded transport tokens.
    :returns: Component span.
    """

    return StagingComponentSpan(
        component_id=component_id,
        source_index_offset=source_offset,
        destination_index_offset=destination_offset,
        logical_token_count=logical_tokens,
        physical_token_count=physical_tokens,
    )


def _bind(
    layout: StagingChunkLayout,
    endpoint: StagingEndpoint,
    *components: StagingComponentBuffer,
) -> StagingEndpointBufferBinding:
    """Bind test registrations to one endpoint.

    :param layout: Immutable staging layout.
    :param endpoint: Source or destination endpoint.
    :param components: Endpoint-local registrations.
    :returns: Bound active components.
    """

    registry = StagingComponentBufferRegistry(tuple(components))
    return bind_staging_endpoint_buffers(layout, endpoint, registry)


def test_main_only_component_binding() -> None:
    """Main KV binds against geometry retained by the immutable layout."""

    geometry = _geometry(MAIN_KV)
    layout = _layout((_span(MAIN_KV),), (geometry,))
    component = _buffer(geometry, (11, 12))

    binding = _bind(layout, StagingEndpoint.SOURCE, component)

    assert layout.source_components == (geometry,)
    assert layout.destination_components == (geometry,)
    assert binding.components[0].component is component
    assert binding.components[0].page_count == 2
    np.testing.assert_array_equal(binding.components[0].page_array, [11, 12])


def test_layout_owns_immutable_geometry_metadata() -> None:
    """Caller mutations cannot desynchronize a layout from its digest."""

    source_item_lens = [64, 64, 64, 64]
    source_layer_ids = [2, 5, 2, 5]
    destination_item_lens = [64, 64, 64, 64]
    destination_layer_ids = [2, 5, 2, 5]
    source = StagingComponentGeometry(
        component_id=MAIN_KV,
        item_lens=source_item_lens,
        layer_ids=source_layer_ids,
        page_size=4,
    )
    destination = StagingComponentGeometry(
        component_id=MAIN_KV,
        item_lens=destination_item_lens,
        layer_ids=destination_layer_ids,
        page_size=4,
    )
    layout = _layout(
        (_span(MAIN_KV),),
        (source,),
        (destination,),
    )
    digest = layout.digest

    source_item_lens[0] = 128
    source_layer_ids[0] = 59
    destination_item_lens[0] = 128
    destination_layer_ids[0] = 59

    assert layout.digest == digest
    assert layout.source_components[0].item_lens == (64, 64, 64, 64)
    assert layout.source_components[0].layer_ids == (2, 5, 2, 5)
    assert layout.destination_components[0].item_lens == (64, 64, 64, 64)
    assert layout.destination_components[0].layer_ids == (2, 5, 2, 5)
    assert layout.writers[0].copy_groups[0].source_token_bytes == 16
    assert layout.writers[0].copy_groups[0].destination_token_bytes == 16


def test_component_registry_owns_registration_metadata() -> None:
    """Registries retain immutable registration sequences and component order."""

    tensor_ptrs = [0x100000, 0x101000]
    data_lens = [1024, 1024]
    item_lens = [64, 64]
    layer_ids = [5, 5]
    component = StagingComponentBuffer(
        component_id=MAIN_KV,
        tensor_ptrs=tensor_ptrs,
        data_lens=data_lens,
        item_lens=item_lens,
        layer_ids=layer_ids,
        page_size=4,
        page_array=np.asarray((1, 2), dtype=np.int32),
    )
    components = [component]
    registry = StagingComponentBufferRegistry(components)

    tensor_ptrs[0] = 0
    data_lens[0] = 0
    item_lens[0] = 128
    layer_ids[0] = 59
    components.clear()

    assert component.tensor_ptrs == (0x100000, 0x101000)
    assert component.data_lens == (1024, 1024)
    assert component.item_lens == (64, 64)
    assert component.layer_ids == (5, 5)
    assert registry.components == (component,)


def test_active_page_binding_is_an_immutable_snapshot() -> None:
    """Allocator mutations after binding cannot change asynchronous DMA indices."""

    geometry = _geometry(MAIN_KV)
    layout = _layout((_span(MAIN_KV),), (geometry,))
    component = _buffer(geometry, (11, 12))

    binding = _bind(layout, StagingEndpoint.SOURCE, component)
    active_page_array = binding.require(MAIN_KV).page_array
    component.page_array[:] = (21, 22)

    np.testing.assert_array_equal(active_page_array, [11, 12])
    assert not active_page_array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        active_page_array[0] = 31


def test_swa_only_component_binding() -> None:
    """An auxiliary SWA component binds without a synthetic main component."""

    geometry = _geometry(
        SWA,
        item_lens=(32, 32),
        layer_ids=(1, 1),
    )
    layout = _layout((_span(SWA),), (geometry,))
    component = _buffer(geometry, (21, 22))

    binding = _bind(layout, StagingEndpoint.DESTINATION, component)

    assert tuple(active.component.component_id for active in binding.components) == (
        SWA,
    )
    np.testing.assert_array_equal(binding.require(SWA).page_array, [21, 22])


def test_combined_registry_preserves_registration_order() -> None:
    """The registry retains input order while binding follows layout order."""

    main_geometry = _geometry(MAIN_KV)
    swa_geometry = _geometry(
        SWA,
        item_lens=(32, 32),
        layer_ids=(1, 1),
    )
    layout = _layout(
        (_span(SWA), _span(MAIN_KV)),
        (swa_geometry, main_geometry),
    )
    swa_component = _buffer(swa_geometry, (31, 32), pointer_base=0x200000)
    main_component = _buffer(main_geometry, (41, 42), pointer_base=0x300000)
    registry = StagingComponentBufferRegistry((swa_component, main_component))

    binding = bind_staging_endpoint_buffers(
        layout,
        StagingEndpoint.SOURCE,
        registry,
    )

    assert tuple(component.component_id for component in registry.components) == (
        SWA,
        MAIN_KV,
    )
    assert tuple(active.component.component_id for active in binding.components) == (
        MAIN_KV,
        SWA,
    )
    assert main_component.layer_ids == (2, 5, 2, 5)
    assert main_component.tensor_ptrs == (
        0x300000,
        0x301000,
        0x302000,
        0x303000,
    )


def test_nonzero_offsets_are_component_local() -> None:
    """Each active component applies its own source and destination offset."""

    main_geometry = _geometry(MAIN_KV)
    swa_geometry = _geometry(
        SWA,
        item_lens=(32, 32),
        layer_ids=(1, 1),
    )
    layout = _layout(
        (
            _span(MAIN_KV, source_offset=1, destination_offset=2),
            _span(SWA, source_offset=2, destination_offset=1),
        ),
        (main_geometry, swa_geometry),
    )
    source = _bind(
        layout,
        StagingEndpoint.SOURCE,
        _buffer(main_geometry, (10, 11, 12, 13)),
        _buffer(swa_geometry, (20, 21, 22, 23), pointer_base=0x200000),
    )
    destination = _bind(
        layout,
        StagingEndpoint.DESTINATION,
        _buffer(main_geometry, (30, 31, 32, 33)),
        _buffer(swa_geometry, (40, 41, 42, 43), pointer_base=0x300000),
    )

    np.testing.assert_array_equal(source.require(MAIN_KV).page_array, [11, 12])
    np.testing.assert_array_equal(source.require(SWA).page_array, [22, 23])
    np.testing.assert_array_equal(
        destination.require(MAIN_KV).page_array,
        [32, 33],
    )
    np.testing.assert_array_equal(destination.require(SWA).page_array, [41, 42])


def test_missing_active_component_is_rejected() -> None:
    """Exact lookup rejects an active component absent from the registry."""

    main_geometry = _geometry(MAIN_KV)
    swa_geometry = _geometry(
        SWA,
        item_lens=(32, 32),
        layer_ids=(1, 1),
    )
    layout = _layout(
        (_span(MAIN_KV), _span(SWA)),
        (main_geometry, swa_geometry),
    )
    registry = StagingComponentBufferRegistry((_buffer(main_geometry, (1, 2)),))

    with pytest.raises(ValueError, match="active component is not registered"):
        bind_staging_endpoint_buffers(layout, StagingEndpoint.SOURCE, registry)


def test_duplicate_component_and_state_index_are_rejected() -> None:
    """The registry rejects both exact duplicates and state-index aliases."""

    swa_geometry = _geometry(
        SWA,
        item_lens=(32, 32),
        layer_ids=(1, 1),
    )
    swa_component = _buffer(swa_geometry, (1, 2))
    with pytest.raises(ValueError, match="registration is duplicated"):
        StagingComponentBufferRegistry((swa_component, swa_component))

    dsa_geometry = dataclasses.replace(
        swa_geometry,
        component_id=DSA_AT_SWA_INDEX,
    )
    dsa_component = _buffer(dsa_geometry, (3, 4), pointer_base=0x200000)
    with pytest.raises(ValueError, match="state_index 0"):
        StagingComponentBufferRegistry((swa_component, dsa_component))


def test_registration_requires_global_layer_ids() -> None:
    """Packed registration never invents positional layer identities."""

    geometry = _geometry(MAIN_KV)
    with pytest.raises(ValueError, match="global layer IDs are required"):
        StagingComponentBuffer(
            component_id=MAIN_KV,
            tensor_ptrs=(0x100000, 0x101000),
            data_lens=(1024, 1024),
            item_lens=(64, 64),
            layer_ids=(),
            page_size=4,
            page_array=np.asarray((1, 2), dtype=np.int32),
        )

    with pytest.raises(ValueError, match="pointer/data-length/item-length"):
        dataclasses.replace(
            _buffer(geometry, (1, 2)),
            tensor_ptrs=(0x100000,),
        )
    with pytest.raises(ValueError, match="is not divisible by page item length"):
        dataclasses.replace(
            _buffer(geometry, (1, 2)),
            data_lens=(1025, 1024, 1024, 1024),
        )


def test_registration_geometry_must_match_immutable_layout() -> None:
    """Endpoint registration cannot replace layout-bound page geometry."""

    layout_geometry = _geometry(MAIN_KV)
    layout = _layout((_span(MAIN_KV),), (layout_geometry,))
    registered_geometry = dataclasses.replace(layout_geometry, page_size=2)
    component = _buffer(registered_geometry, (1, 2, 3, 4))

    with pytest.raises(ValueError, match="differs from immutable layout"):
        _bind(layout, StagingEndpoint.SOURCE, component)


def test_endpoint_page_bounds_are_independent() -> None:
    """Source and destination arrays are bounded without cross-endpoint state."""

    geometry = _geometry(MAIN_KV)
    layout = _layout(
        (_span(MAIN_KV, source_offset=1, destination_offset=0),),
        (geometry,),
    )
    short_source = _buffer(geometry, (1, 2))
    valid_destination = _buffer(geometry, (3, 4), pointer_base=0x200000)

    with pytest.raises(ValueError, match="source page-array bounds overflow"):
        _bind(layout, StagingEndpoint.SOURCE, short_source)
    destination = _bind(
        layout,
        StagingEndpoint.DESTINATION,
        valid_destination,
    )
    np.testing.assert_array_equal(destination.require(MAIN_KV).page_array, [3, 4])

    valid_source = _buffer(geometry, (5, 6, 7), pointer_base=0x300000)
    short_destination = _buffer(geometry, (8,), pointer_base=0x400000)
    source = _bind(layout, StagingEndpoint.SOURCE, valid_source)
    np.testing.assert_array_equal(source.require(MAIN_KV).page_array, [6, 7])
    with pytest.raises(ValueError, match="destination page-array bounds overflow"):
        _bind(layout, StagingEndpoint.DESTINATION, short_destination)


def test_active_page_indices_must_fit_every_registered_tensor() -> None:
    """Negative and out-of-capacity active physical pages are rejected."""

    geometry = _geometry(MAIN_KV)
    layout = _layout((_span(MAIN_KV),), (geometry,))

    negative_page = _buffer(geometry, (-1, 0))
    with pytest.raises(ValueError, match="contains a negative index"):
        _bind(layout, StagingEndpoint.SOURCE, negative_page)

    upper_bound_page = _buffer(
        geometry,
        (3, 4),
        page_capacity=4,
        pointer_base=0x200000,
    )
    with pytest.raises(ValueError, match="exceeds registered page capacity 4"):
        _bind(layout, StagingEndpoint.DESTINATION, upper_bound_page)

    valid_boundary = _buffer(
        geometry,
        (2, 3),
        page_capacity=4,
        pointer_base=0x300000,
    )
    binding = _bind(layout, StagingEndpoint.SOURCE, valid_boundary)
    np.testing.assert_array_equal(binding.require(MAIN_KV).page_array, [2, 3])


def test_swa_pool_exposes_global_k_then_v_layer_ids() -> None:
    """SWA pool metadata follows the buffer registration order exactly."""

    pool = object.__new__(SWAKVPool)
    pool._full_attention_layer_ids = (1, 3, 5)
    pool._swa_attention_layer_ids = (0, 2, 4)

    assert pool.get_kv_layer_ids() == [1, 3, 5, 1, 3, 5]
    assert pool.get_state_layer_ids() == [0, 2, 4, 0, 2, 4]


def test_unified_swa_pool_owns_global_layer_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unified SWA registration works without invoking the static-pool initializer."""

    full_attention_layer_ids = [1, 3, 5]
    swa_attention_layer_ids = [0, 2]
    spec = SimpleNamespace(
        store_dtype="bf16",
        head_num=4,
        head_dim=8,
    )
    unified_buffer = SimpleNamespace(
        device="cpu",
        max_slots=lambda component_name: 257,
        mha_spec=lambda component_name: spec,
    )
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_memory_pool.UnifiedMHATokenToKVPool",
        lambda **kwargs: _SWAStateBufferPool(),
    )
    pool = UnifiedSWAKVPool(
        unified_buffer=unified_buffer,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
    )

    full_attention_layer_ids[0] = 59
    swa_attention_layer_ids[0] = 58

    kv_args = KVArgs()
    setup_state_kv_args(kv_args, pool)

    assert pool.get_kv_layer_ids() == [1, 3, 5, 1, 3, 5]
    assert pool.get_state_layer_ids() == [0, 2, 0, 2]
    assert kv_args.state_types == [StateType.SWA]
    assert kv_args.state_layer_ids == [[0, 2, 0, 2]]


def test_swa_state_registration_propagates_global_layer_ids() -> None:
    """State KV arguments retain explicit SWA layer identities."""

    pool = object.__new__(SWAKVPool)
    pool._swa_attention_layer_ids = (0, 2)
    pool.swa_kv_pool = _SWAStateBufferPool()
    kv_args = KVArgs()

    setup_state_kv_args(kv_args, pool)

    assert kv_args.state_types == [StateType.SWA]
    assert kv_args.state_layer_ids == [[0, 2, 0, 2]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
