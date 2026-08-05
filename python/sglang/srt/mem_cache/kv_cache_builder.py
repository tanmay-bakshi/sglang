import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)
from sglang.srt.configs.model_config import ModelImpl, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheBuildResult:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: int | None
    full_tokens_per_layer: int | None
    swa_tokens_per_layer: int | None
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    disable_radix_cache: bool
    tree_cache: object


if TYPE_CHECKING:

    from torch.distributed import ProcessGroup

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def get_draft_kv_pool(
    *,
    draft_worker: "BaseTpWorker | None",
    spec_algorithm: "SpeculativeAlgorithm",
    server_args: "ServerArgs",
) -> KVCache | None:
    """Return the current speculative worker's draft KV pool.

    :param draft_worker: Speculative worker that owns the draft runner.
    :param spec_algorithm: Active speculative algorithm.
    :param server_args: Resolved server configuration.
    :returns: Draft token-to-KV pool, or ``None`` when none exists.
    """
    if draft_worker is None or spec_algorithm.is_ngram():
        return None

    # V2 workers nest the draft runner under `.draft_worker`.
    if server_args.enable_multi_layer_eagle:
        draft_runner = draft_worker.draft_worker.draft_runner_list[0]
    else:
        draft_runner = draft_worker.draft_worker.draft_runner
    return draft_runner.token_to_kv_pool


def _make_draft_layer_mapper(
    draft_host_pool: HostKVCache,
) -> Callable[[int], int | None]:
    """Map target transfer steps onto draft-local layer indices.

    :param draft_host_pool: Draft host pool registered as a KV sidecar.
    :returns: Layer mapper for a :class:`PoolEntry`.
    """

    start_layer = int(draft_host_pool.start_layer)
    end_layer = start_layer + int(draft_host_pool.layer_num)

    def map_layer(layer_id: int) -> int | None:
        if layer_id < start_layer or layer_id >= end_layer:
            return None
        return layer_id - start_layer

    return map_layer


def _register_unified_hicache_draft(
    *,
    tree_cache: "BasePrefixCache",
    draft_device_pool: KVCache,
    draft_host_pool: HostKVCache,
) -> None:
    """Register draft KV in the Unified HiCache component lifecycle.

    :param tree_cache: Unified target radix cache.
    :param draft_device_pool: Draft KV storage sharing target full-KV indices.
    :param draft_host_pool: Host mirror with the same physical slot geometry.
    :raises TypeError: If the target cache does not use the Unified controller.
    :raises ValueError: If the target transfer cannot cover every draft layer.
    """

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if not isinstance(tree_cache, UnifiedRadixCache):
        raise TypeError("Hybrid HiCache draft registration requires UnifiedRadixCache")
    controller = tree_cache.cache_controller
    if not isinstance(controller, HybridCacheController):
        raise TypeError("Unified draft registration requires HybridCacheController")
    if draft_host_pool.layer_num > controller.layer_num:
        raise ValueError(
            "Draft HiCache has more layers than the target transfer lifecycle: "
            f"draft_layers={draft_host_pool.layer_num}, "
            f"target_transfer_layers={controller.layer_num}"
        )

    controller.register_index_aligned_pool(
        PoolEntry(
            name=PoolName.DRAFT,
            host_pool=draft_host_pool,
            device_pool=draft_device_pool,
            layer_mapper=_make_draft_layer_mapper(draft_host_pool),
        )
    )
    tree_cache.register_sidecar_pool(
        SidecarPoolSpec(
            pool_name=PoolName.DRAFT,
            indices_from_pool=PoolName.KV,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
    )


def maybe_register_hicache_draft(
    *,
    tree_cache: "BasePrefixCache",
    draft_worker: "BaseTpWorker | None",
    spec_algorithm: "SpeculativeAlgorithm",
    server_args: "ServerArgs",
    enable_hierarchical_cache: bool,
    page_size: int,
) -> None:
    """Register draft KV pool with HiCacheController for piggyback L2/L3 ops."""
    if not enable_hierarchical_cache:
        return

    draft_kv_pool = get_draft_kv_pool(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        server_args=server_args,
    )
    if draft_kv_pool is None:
        return

    controller = tree_cache.cache_controller
    if (
        isinstance(controller, HybridCacheController)
        and isinstance(controller.mem_pool_host, HostPoolGroup)
        and PoolName.DRAFT in controller.mem_pool_host.entry_map
    ):
        return

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool

    # Create host pool for draft with the same slot count as the target host pool,
    # so that host indices stay 1-to-1 between target and draft KV caches.
    primary = controller.mem_pool_host
    if server_args.hicache_size > 0:
        raise RuntimeError(
            "Fixed-size HiCache requires draft KV registration during "
            "the initial host-pool budget split"
        )
    kw = dict(
        # HostKVCache rounds ``floor(size * ratio)`` up to the next page. Pick
        # the last token in the anchor capacity so the rounded result is exact.
        host_to_device_ratio=(primary.size - 1) / pool.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    if isinstance(pool, MHATokenToKVPool):
        draft_host_pool = get_mha_host_pool_cls(pool)(pool, **kw)
    elif isinstance(pool, MLATokenToKVPool):
        draft_host_pool = MLATokenToKVPoolHost(pool, **kw)
    else:
        message = f"Draft pool type {type(pool).__name__} is not supported by HiCache"
        if spec_algorithm.is_dflash():
            raise RuntimeError(message)
        logger.warning("%s; skipping draft KV offload.", message)
        return

    if isinstance(controller, HybridCacheController):
        _register_unified_hicache_draft(
            tree_cache=tree_cache,
            draft_device_pool=pool,
            draft_host_pool=draft_host_pool,
        )
        return

    controller.set_draft_kv_pool(pool, draft_host_pool)


def build_kv_cache(
    *,
    server_args: "ServerArgs",
    model_config: "ModelConfig",
    tp_worker: "BaseTpWorker",
    page_size: int,
    spec_algorithm: "SpeculativeAlgorithm",
    attn_tp_cpu_group: "ProcessGroup",
    tp_cpu_group: "ProcessGroup",
    attn_cp_cpu_group: "ProcessGroup",
    enable_metrics: bool,
    enable_kv_cache_events: bool,
    ps: "ParallelState",
    tp_group: "GroupCoordinator",
    pp_group: "GroupCoordinator",
    enable_hierarchical_cache: bool,
    draft_token_to_kv_pool: KVCache | None,
) -> KVCacheBuildResult:
    sliding_window_size: int | None = None
    full_tokens_per_layer: int | None = None
    swa_tokens_per_layer: int | None = None
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )

    # Hybrid memory pool
    is_hybrid_swa = tp_worker.is_hybrid_swa
    _spec = linear_attn_model_spec(tp_worker.model_runner.model_config)
    _registry_needs_mamba = _spec.uses_mamba_radix_cache if _spec is not None else False
    is_hybrid_ssm = (
        hybrid_gdn_config(tp_worker.model_runner.model_config) is not None
        or mamba2_config(tp_worker.model_runner.model_config) is not None
        or _registry_needs_mamba
        or kimi_linear_config(tp_worker.model_runner.model_config) is not None
        or hybrid_lightning_config(tp_worker.model_runner.model_config) is not None
    )
    is_dsa = is_deepseek_dsa(model_config.hf_config)

    sliding_window_size = None
    if is_hybrid_swa:
        sliding_window_size = tp_worker.sliding_window_size
        full_tokens_per_layer, swa_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()
        )

    req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()
    hicache_draft_kv_pool = draft_token_to_kv_pool
    if isinstance(hicache_draft_kv_pool, HybridLinearKVPool):
        hicache_draft_kv_pool = hicache_draft_kv_pool.full_kv_pool

    disable_radix_cache = server_args.disable_radix_cache or (
        model_config.is_multimodal and uses_transformers_backend
    )
    if disable_radix_cache and not server_args.disable_radix_cache:
        logger.warning(
            "Radix cache is disabled for multimodal models with the "
            "Transformers backend to avoid multimodal prefix-cache mismatches."
        )

    if (
        server_args.disaggregation_decode_enable_radix_cache
        and server_args.disaggregation_mode == "decode"
        and is_hybrid_ssm
    ):
        raise ValueError(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with Mamba/SSM models"
        )

    effective_chunked_prefill_size = server_args.chunked_prefill_size
    if model_config.is_multimodal and uses_transformers_backend:
        effective_chunked_prefill_size = None

    params = CacheInitParams(
        disable=disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        draft_token_to_kv_pool=hicache_draft_kv_pool,
        # When dcp enabled, kv_pool_allocator.page_size is page_size * dcp_size.
        # TreeCache.page_size should keep the same as allocator.page_size to
        # avoid kv page eviction conflicts.
        page_size=(
            page_size
            if not get_parallel().dcp_enabled
            else token_to_kv_pool_allocator.page_size
        ),
        is_eagle=spec_algorithm.is_eagle(),
        tp_cache_group=(
            attn_tp_cpu_group if server_args.enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        pp_cache_group=pp_group.cpu_group,
        eviction_policy=server_args.radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_session_radix_cache=server_args.enable_session_radix_cache,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
    )

    tree_cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            full_tokens_per_layer=full_tokens_per_layer,
            is_hybrid_ssm=is_hybrid_ssm,
            is_dsa=is_dsa,
            enable_hierarchical_cache=enable_hierarchical_cache,
            disable_radix_cache=disable_radix_cache,
            effective_chunked_prefill_size=effective_chunked_prefill_size,
            tp_worker=tp_worker,
            model_config=model_config,
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            tp_group=tp_group,
        )
    )

    embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    return KVCacheBuildResult(
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        sliding_window_size=sliding_window_size,
        full_tokens_per_layer=full_tokens_per_layer,
        swa_tokens_per_layer=swa_tokens_per_layer,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        disable_radix_cache=disable_radix_cache,
        tree_cache=tree_cache,
    )
