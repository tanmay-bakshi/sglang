from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_hybrid_swa_stack,
)
from sglang.srt.mem_cache.kv_cache_builder import (
    _register_unified_hicache_draft,
    build_kv_cache,
    maybe_register_hicache_draft,
)
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

GEMMA_FULL_TOKEN_CAPACITY = 12_544
GEMMA_SWA_TOKEN_CAPACITY = 152_064
GEMMA_TARGET_FULL_LAYERS = 10
GEMMA_TARGET_SWA_LAYERS = 50
DFLASH_DRAFT_LAYERS = 5


def _pool(
    *,
    size: int = GEMMA_FULL_TOKEN_CAPACITY,
    page_size: int = 16,
    layer_num: int = 60,
):
    """Build a minimal host or device pool test double.

    :param size: Number of physical token slots.
    :param page_size: Allocation page size.
    :param layer_num: Number of cache layers.
    :returns: Pool-shaped test double.
    """

    return SimpleNamespace(
        size=size,
        page_size=page_size,
        layer_num=layer_num,
        start_layer=0,
        layout="page_first",
        device="cpu",
        can_use_write_back_jit=True,
        destroy=MagicMock(),
        load_to_device_per_layer=MagicMock(),
        backup_from_device_all_layer=MagicMock(),
    )


def _host_pool_group() -> tuple[HostPoolGroup, object, object]:
    """Build one target anchor group.

    :returns: Host group, target host pool, and target device pool.
    """

    target_host = _pool()
    target_device = _pool()
    group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=target_host,
                device_pool=target_device,
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            )
        ]
    )
    return group, target_host, target_device


def test_draft_sidecar_uses_unified_lifecycle_and_all_pages_policy() -> None:
    """Draft KV shares target indices, storage validity, and cleanup."""

    group, target_host, _ = _host_pool_group()
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = group
    controller.layer_num = 60
    controller.enable_storage = True
    controller.storage_backend = MagicMock()

    cache = object.__new__(UnifiedRadixCache)
    cache.cache_controller = controller
    cache.sidecar_pool_specs = []

    draft_host = _pool(layer_num=5)
    draft_device = _pool(layer_num=5)
    _register_unified_hicache_draft(
        tree_cache=cache,
        draft_device_pool=draft_device,
        draft_host_pool=draft_host,
    )

    entry = group.entry_map[PoolName.DRAFT]
    assert entry.host_pool is draft_host
    assert entry.device_pool is draft_device
    assert entry.layer_mapper(0) == 0
    assert entry.layer_mapper(4) == 4
    assert entry.layer_mapper(5) is None
    assert len(cache.sidecar_pool_specs) == 1
    sidecar = cache.sidecar_pool_specs[0]
    assert sidecar.pool_name == PoolName.DRAFT
    assert sidecar.indices_from_pool == PoolName.KV
    assert sidecar.hit_policy == PoolHitPolicy.ALL_PAGES
    controller.storage_backend.register_mem_host_pool_v2.assert_called_once_with(
        draft_host,
        PoolName.DRAFT,
    )

    host_indices = torch.tensor([16, 17], dtype=torch.int64)
    device_indices = torch.tensor([32, 33], dtype=torch.int64)
    transfer = PoolTransfer(
        name=PoolName.DRAFT,
        indices_from_pool=PoolName.KV,
    )
    resolved = controller._resolve_pool_transfers_allocation(
        [transfer],
        alloc_host=False,
        kv_host_indices=host_indices,
        kv_device_indices=device_indices,
    )
    assert resolved == [transfer]
    assert transfer.host_indices is host_indices
    assert transfer.device_indices is device_indices

    group.load_to_device_per_layer(
        device_pool=None,
        host_indices=host_indices,
        device_indices=device_indices,
        layer_id=0,
        io_backend="direct",
        pool_transfers=resolved,
    )
    target_host.load_to_device_per_layer.assert_called_once()
    draft_host.load_to_device_per_layer.assert_called_once_with(
        draft_device,
        host_indices,
        device_indices,
        0,
        "direct",
    )

    group.destroy()
    target_host.destroy.assert_called_once_with()
    draft_host.destroy.assert_called_once_with()


def test_index_aligned_sidecar_rejects_incomplete_device_geometry() -> None:
    """A draft pool must be able to address every target full-KV slot."""

    group, _, _ = _host_pool_group()
    with pytest.raises(ValueError, match="cannot cover every anchor index"):
        group.register_index_aligned_pool(
            PoolEntry(
                name=PoolName.DRAFT,
                host_pool=_pool(),
                device_pool=_pool(size=64, layer_num=5),
                layer_mapper=lambda layer_id: layer_id,
            )
        )


def test_gemma_dflash_fixed_hicache_budget_preserves_full_index_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma FULL and DFlash share capacity under one fixed host budget.

    :param monkeypatch: Pytest patch fixture.
    """

    import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

    def device_pool(
        *,
        size: int,
        layer_num: int,
        k_bytes_per_token: int,
        v_bytes_per_token: int,
    ):
        """Build a device pool with deterministic budget weight.

        :param size: Number of physical token slots.
        :param layer_num: Number of cache layers.
        :param k_bytes_per_token: Key-cache bytes per physical slot.
        :param v_bytes_per_token: Value-cache bytes per physical slot.
        :returns: Device-pool test double.
        """

        pool = _pool(size=size, layer_num=layer_num)
        pool.get_kv_size_bytes = lambda: (
            size * k_bytes_per_token,
            size * v_bytes_per_token,
        )
        return pool

    full_pool = device_pool(
        size=GEMMA_FULL_TOKEN_CAPACITY,
        layer_num=GEMMA_TARGET_FULL_LAYERS,
        k_bytes_per_token=GEMMA_TARGET_FULL_LAYERS * 4 * 512,
        v_bytes_per_token=GEMMA_TARGET_FULL_LAYERS * 4 * 512,
    )
    swa_pool = device_pool(
        size=GEMMA_SWA_TOKEN_CAPACITY,
        layer_num=GEMMA_TARGET_SWA_LAYERS,
        k_bytes_per_token=GEMMA_TARGET_SWA_LAYERS * 16 * 256,
        v_bytes_per_token=GEMMA_TARGET_SWA_LAYERS * 16 * 256,
    )
    draft_pool = device_pool(
        size=GEMMA_FULL_TOKEN_CAPACITY,
        layer_num=DFLASH_DRAFT_LAYERS,
        k_bytes_per_token=DFLASH_DRAFT_LAYERS * 8 * 128 * 2,
        v_bytes_per_token=DFLASH_DRAFT_LAYERS * 8 * 128 * 2,
    )
    host_sizes: list[float | None] = []

    def build_host_pool(**kwargs):
        """Record each fixed-size share and mirror production page rounding.

        :param kwargs: Host-pool construction arguments.
        :returns: Host-pool test double.
        """

        device_pool = kwargs["kv_pool"]
        host_size = kwargs["host_size"]
        assert host_size is not None
        host_sizes.append(host_size)
        k_bytes, v_bytes = device_pool.get_kv_size_bytes()
        bytes_per_token = (k_bytes + v_bytes) // device_pool.size
        unaligned_size = int(host_size * 1e9 // bytes_per_token)
        page_size = kwargs["page_size"]
        host_capacity = (unaligned_size // page_size + 1) * page_size
        return _pool(
            size=host_capacity,
            page_size=page_size,
            layer_num=device_pool.layer_num,
        )

    controller = object()
    monkeypatch.setattr(assembler, "build_kv_host_pool", build_host_pool)
    monkeypatch.setattr(
        assembler,
        "HybridCacheController",
        MagicMock(return_value=controller),
    )
    allocator = SimpleNamespace(
        swa_attn_allocator=SimpleNamespace(alloc=MagicMock(), free=MagicMock())
    )
    params = SimpleNamespace(
        page_size=16,
        token_to_kv_pool_allocator=allocator,
        tp_cache_group=object(),
        attn_cp_cache_group=object(),
        attn_tp_cache_group=object(),
        pp_cache_group=object(),
    )
    server_args = SimpleNamespace(
        hicache_size=64,
        hicache_write_policy="write_through_selective",
        hicache_io_backend="kernel",
    )

    group, built_controller = build_hybrid_swa_stack(
        params=params,
        server_args=server_args,
        full_kv_pool=full_pool,
        swa_kv_pool=swa_pool,
        draft_kv_pool=draft_pool,
        full_layer_mapping={layer_id: layer_id for layer_id in range(10)},
        swa_layer_mapping={layer_id: layer_id - 10 for layer_id in range(10, 60)},
        load_cache_event=object(),
        storage_backend=None,
        use_mla=False,
    )

    device_pool_bytes = [
        sum(full_pool.get_kv_size_bytes()),
        sum(swa_pool.get_kv_size_bytes()),
        sum(draft_pool.get_kv_size_bytes()),
    ]
    total_device_pool_bytes = sum(device_pool_bytes)
    expected_host_sizes = [
        64 * pool_bytes / total_device_pool_bytes for pool_bytes in device_pool_bytes
    ]
    assert host_sizes == pytest.approx(expected_host_sizes)
    assert sum(host_sizes) == pytest.approx(64)
    assert full_pool.size == draft_pool.size == GEMMA_FULL_TOKEN_CAPACITY
    assert swa_pool.size == GEMMA_SWA_TOKEN_CAPACITY
    assert group.size == group.entry_map[PoolName.DRAFT].host_pool.size
    assert group.entry_map[PoolName.DRAFT].device_pool is draft_pool
    assert built_controller is controller


def test_dflash_hicache_rejects_an_unrepresentable_draft_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DFlash cannot continue when its draft KV would be omitted from HiCache.

    :param monkeypatch: Pytest patch fixture.
    """

    unsupported_pool = SimpleNamespace(size=128)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_builder.get_draft_kv_pool",
        lambda **_: unsupported_pool,
    )
    tree_cache = SimpleNamespace(
        cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(size=128))
    )
    server_args = SimpleNamespace(
        hicache_size=0,
        hicache_mem_layout="page_first",
        hicache_storage_backend=None,
    )

    with pytest.raises(RuntimeError, match="is not supported by HiCache"):
        maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_worker=SimpleNamespace(),
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            server_args=server_args,
            enable_hierarchical_cache=True,
            page_size=16,
        )


def test_gemma_style_swa_decode_radix_cache_reaches_unified_build_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component-aware SWA decode caching is accepted during cache construction.

    :param monkeypatch: Pytest patch fixture.
    """

    import sglang.srt.mem_cache.kv_cache_builder as builder

    monkeypatch.setattr(builder, "get_resolved_model_impl", lambda _: object())
    monkeypatch.setattr(builder, "linear_attn_model_spec", lambda _: None)
    monkeypatch.setattr(builder, "hybrid_gdn_config", lambda _: None)
    monkeypatch.setattr(builder, "mamba2_config", lambda _: None)
    monkeypatch.setattr(builder, "kimi_linear_config", lambda _: None)
    monkeypatch.setattr(builder, "hybrid_lightning_config", lambda _: None)
    monkeypatch.setattr(builder, "is_deepseek_dsa", lambda _: False)
    monkeypatch.setattr(
        builder,
        "get_parallel",
        lambda: SimpleNamespace(dcp_enabled=False),
    )
    cache = object()
    create_tree_cache = MagicMock(return_value=cache)
    monkeypatch.setattr(builder, "create_tree_cache", create_tree_cache)
    monkeypatch.setattr(builder, "init_mm_embedding_cache", MagicMock())

    req_to_token_pool = object()
    allocator = SimpleNamespace(page_size=16)
    model_runner_config = object()
    tp_worker = SimpleNamespace(
        is_hybrid_swa=True,
        sliding_window_size=2048,
        model_runner=SimpleNamespace(model_config=model_runner_config),
        get_tokens_per_layer_info=lambda: (8192, 2048),
        get_memory_pool=lambda: (req_to_token_pool, allocator),
    )
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disaggregation_decode_enable_radix_cache=True,
        disaggregation_mode="decode",
        chunked_prefill_size=8192,
        enable_dp_attention=False,
        radix_eviction_policy="lru",
        enable_session_radix_cache=False,
        enable_mamba_extra_buffer=lambda: False,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )
    model_config = SimpleNamespace(is_multimodal=False, hf_config=object())
    ps = SimpleNamespace(pp_rank=0, pp_size=1, tp_size=1, tp_rank=0)
    pp_group = SimpleNamespace(cpu_group=object())

    result = build_kv_cache(
        server_args=server_args,
        model_config=model_config,
        tp_worker=tp_worker,
        page_size=16,
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        attn_tp_cpu_group=object(),
        tp_cpu_group=object(),
        attn_cp_cpu_group=object(),
        enable_metrics=False,
        enable_kv_cache_events=False,
        ps=ps,
        tp_group=object(),
        pp_group=pp_group,
        enable_hierarchical_cache=True,
        draft_token_to_kv_pool=None,
    )

    assert result.tree_cache is cache
    context = create_tree_cache.call_args.args[0]
    assert context.is_hybrid_swa is True
    assert context.enable_hierarchical_cache is True
    assert context.server_args.disaggregation_decode_enable_radix_cache is True
