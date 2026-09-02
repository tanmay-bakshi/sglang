from __future__ import annotations

import atexit
import hashlib
import logging
import threading
import time
from array import array
from dataclasses import dataclass, field, replace
from itertools import pairwise
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence, TypeVar

import torch

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.lifecycle_pause_point import injection_enabled, pause_point
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    HostLockOwner,
    HostLockOwnerKind,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    KVComponentResidency,
    LoadBackResult,
    MatchPrefixParams,
    MatchResult,
    StreamingSessionCacheSnapshot,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    validate_buffer_only_stack,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.common import RetractionBackup
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeComponentDeviceSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    ReplaceWriteThroughOnNodeSplit,
)

# UnifiedTreeNode / UnifiedLRUList live on the tree core; re-exported here
# because other modules and tests import them from this module.
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    FullComponent,
    MambaComponent,
    PrepareLoadBackResult,
    SWAComponent,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    UnifiedSessionRefTracker,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.tree_core_registry import create_tree_core
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: F401
    NodeId,
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.observability.metrics_collector import (
    StorageMetrics,
    StorageMetricsCollector,
)
from sglang.srt.runtime_context import get_memory, get_observability
from sglang.srt.session.streaming_session import StreamingSession
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.session.session_controller import StreamingSessionInventory
    from sglang.srt.managers.cache_controller import HiCacheAck
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.mem_cache.pool_host import PoolEntry
    from sglang.srt.server_args import ServerArgs

from sglang.srt.utils.rank_consensus_checker import rank_consensus

T = TypeVar("T")

# Metric label per component, matching the host pool names used by
# hicache_backup_tokens_total and the host occupancy gauges.
_COMPONENT_POOL_LABEL = {
    ComponentType.FULL: PoolName.KV.value,
    ComponentType.SWA: PoolName.SWA.value,
    ComponentType.MAMBA: PoolName.MAMBA.value,
}

_COMPONENT_HOST_POOL = {
    ComponentType.SWA: PoolName.SWA,
    ComponentType.MAMBA: PoolName.MAMBA,
}


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}


logger = logging.getLogger(__name__)


def _queued_transfer_token_count(transfer: PoolTransfer) -> int:
    """Return the physical token count of an allocated H-to-D transfer.

    :param transfer: Transfer accepted by the hybrid cache controller.
    :returns: Number of physical source and destination indices.
    """

    if transfer.host_indices is None or transfer.device_indices is None:
        raise AssertionError(f"{transfer.name} load-back transfer is not allocated")
    host_tokens = int(transfer.host_indices.numel())
    device_tokens = int(transfer.device_indices.numel())
    if host_tokens != device_tokens:
        raise AssertionError(
            f"{transfer.name} load-back size mismatch: {host_tokens=} {device_tokens=}"
        )
    return device_tokens


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node_id: NodeId
    lock_params: Optional[DecLockRefParams]
    publish_node_ids: list[NodeId]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node_id: NodeId
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


@dataclass(frozen=True)
class _QueuedLoadBackResult:
    """Describes the component payload accepted by the load controller.

    :ivar new_full_device_indices: Newly allocated full-KV device indices.
    :ivar full_tokens: Physical full-KV tokens queued for copying.
    :ivar swa_tokens: Physical SWA tokens queued for copying.
    """

    new_full_device_indices: torch.Tensor
    full_tokens: int
    swa_tokens: int


@dataclass(frozen=True)
class _PendingStreamingSessionDemotion:
    """Privately staged host copy awaiting the tensor-parallel commit vote.

    :ivar key: Exact radix identity for the staged prefix.
    :ivar aligned_len: Complete-page length of the exact staged prefix.
    :ivar device_indices: Device slots copied into the private host stage.
    :ivar backup: Privately owned target, SWA, and draft host slots.
    :ivar tree_prefix_node: Existing ordinary radix frontier anchoring the suffix.
    :ivar tree_prefix_len: Prefix already owned by the ordinary radix tree.
    :ivar swa_window_start: First staged token retaining SWA device state.
    :ivar priority: Eviction priority inherited from the session request.
    """

    key: RadixKey
    aligned_len: int
    device_indices: torch.Tensor
    backup: RetractionBackup
    tree_prefix_node: NodeId
    tree_prefix_len: int
    swa_window_start: int
    priority: int


@dataclass
class _HostStageSlice:
    """One consumed range from a staged host allocation."""

    indices: torch.Tensor
    tree_owned: bool = False


@dataclass
class _HostStageAllocation:
    """Resumable ownership cursor for one independently allocated host pool."""

    pool: PoolName
    indices: torch.Tensor
    cursor: int = 0
    consumed: list[_HostStageSlice] = field(default_factory=list)
    released: bool = False

    def take(self, length: int) -> _HostStageSlice:
        """Consume the next exact range while retaining stage ownership."""
        if length <= 0 or self.cursor + length > len(self.indices):
            raise AssertionError(
                f"Invalid {self.pool} stage consumption: {self.cursor=} "
                f"{length=} capacity={len(self.indices)}"
            )
        stage_slice = _HostStageSlice(
            indices=self.indices[self.cursor : self.cursor + length]
        )
        self.cursor += length
        self.consumed.append(stage_slice)
        return stage_slice

    def stage_owned_indices(self) -> torch.Tensor:
        """Return every slot still owned by the stage in allocation order."""
        parts = [item.indices for item in self.consumed if not item.tree_owned]
        if self.cursor < len(self.indices):
            parts.append(self.indices[self.cursor :])
        if len(parts) == 0:
            return self.indices[:0]
        return torch.cat(parts)


@dataclass
class _HostStageLedger:
    """Exact ownership ledger while a demotion stage is attached to the tree."""

    allocations: dict[PoolName, _HostStageAllocation]

    @classmethod
    def from_backup(cls, backup: RetractionBackup) -> _HostStageLedger:
        """Build one ledger from independently allocated backup pools."""
        if backup.host_indices is None:
            raise AssertionError("Host-pool demotion requires Full host indices.")
        allocations = {
            PoolName.KV: _HostStageAllocation(PoolName.KV, backup.host_indices)
        }
        for transfer in backup.pool_transfers or ():
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            if transfer.name in allocations:
                raise AssertionError(
                    f"Demotion stage has multiple {transfer.name} allocations."
                )
            allocations[transfer.name] = _HostStageAllocation(
                transfer.name, transfer.host_indices
            )
        return cls(allocations=allocations)

    def take(self, pool: PoolName, length: int) -> _HostStageSlice:
        """Consume one range from an independently allocated pool."""
        allocation = self.allocations.get(pool)
        if allocation is None:
            raise AssertionError(f"Demotion stage has no {pool} allocation.")
        return allocation.take(length)

    def assert_fully_consumed(self) -> None:
        """Require every staged allocation to have an explicit disposition."""
        incomplete = {
            pool: len(allocation.indices) - allocation.cursor
            for pool, allocation in self.allocations.items()
            if allocation.cursor != len(allocation.indices)
        }
        if len(incomplete) > 0:
            raise AssertionError(f"Demotion stage has unplanned slots: {incomplete}.")

    def release_stage_owned(self, host_pool_group) -> None:
        """Release every slot not transferred to tree ownership exactly once."""
        for allocation in self.allocations.values():
            if allocation.released:
                continue
            indices = allocation.stage_owned_indices()
            if len(indices) > 0:
                host_pool_group.free(indices, pool=allocation.pool)
            allocation.released = True


@dataclass(frozen=True)
class _HostPathAttachment:
    """Tree component value whose ownership came from one staged slice."""

    node_id: NodeId
    component_type: ComponentType
    stage_slice: _HostStageSlice


@dataclass
class _StreamingSessionHostPathTransaction:
    """Rollback ledger for staged host-path publication."""

    ledger: _HostStageLedger
    attachments: list[_HostPathAttachment] = field(default_factory=list)


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node_id: NodeId
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams | None
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.disable = params.disable

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.enable_session_radix_cache = params.enable_session_radix_cache
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        # Whether SWA is enabled.
        self.is_swa_enabled = ComponentType.SWA in params.tree_components
        # Whether Mamba is enabled.
        self.is_mamba_enabled = ComponentType.MAMBA in params.tree_components
        # Whether the mamba extra (ping-pong) buffer is enabled.
        self.enable_mamba_extra_buffer = (
            params.enable_mamba_extra_buffer if self.is_mamba_enabled else False
        )
        # SWA window size (None when SWA is not enabled).
        self._sliding_window_size = (
            params.sliding_window_size if self.is_swa_enabled else None
        )
        # The TreeCore owns the tree member-var state (structure, LRUs, sizes,
        # evictable leaves) and drives the components' tree-level hooks.
        self.tree_core = create_tree_core(
            name=envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(),
            params=params,
            components=self.components,
        )
        # Components execute boundary actions through the tree core.
        for component in self.components.values():
            component.tree_core = self.tree_core

        # Session ref tracking (--enable-session-radix-cache).
        self.session_refs = UnifiedSessionRefTracker(
            components=self._components_tuple,
            tree_core=self.tree_core,
            enable_session_radix_cache=self.enable_session_radix_cache,
        )

        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)
        self._pending_streaming_session_demotions: dict[
            str, _PendingStreamingSessionDemotion
        ] = {}

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller: Optional[HybridCacheController] = None
        self.host_pool_group = None  # set by attach_hybrid_pool_to_unified_cache
        # Owns the storage backend lifecycle; built by init_hicache.
        self._storage_attachment: Optional[StorageAttachment] = None
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False
        # Buffer-only host memory mode (host RAM as transient GPU↔storage
        # staging, not an L2 tier); resolved in init_hicache, which also
        # constructs the pipeline collaborator (None = cache mode).
        self.host_memory_mode = "cache"
        self.buffer_pipeline: Optional[BufferModePipeline] = None
        # Write-side dedupe: beliefs about what storage already holds, so
        # re-inserts of hot prefixes skip the redundant backup.
        self.storage_existence_cache = StorageExistenceCache()
        # Cumulative prefetch-outcome counters, exported through the
        # log_storage_metrics flow.
        self._prefetch_outcome_stats: dict[str, float] = {
            "attempts": 0,
            "issued": 0,
            "declined_too_short": 0,
            "declined_rate_limited": 0,
            "declined_anchor_lost": 0,
            "declined_device_covered": 0,
            "revoked_insufficient": 0,
            "revoked_full_miss": 0,
            "l3_demand_requests": 0,
            "l3_miss_tokens": 0,
            "l1l2_miss_tokens": 0,
            "l3_demand_total_tokens": 0,
            "l3_sum_rate_all": 0.0,
            "l3_sum_rate_main_weighted": 0.0,
        }

        self.reset()
        logger.info(
            f"Init Unified Radix Cache. Components: {self.tree_components}. "
            f"Tree Core: {type(self.tree_core).__name__}"
        )

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.
        """
        for work in self.work_list:
            work.wait()
        self.work_list.clear()

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data,
                group_src=self.pp_rank - 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def reset(self) -> None:
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        self.tree_core.reset()
        self.session_refs.reset()

        # Reset Controller.
        self.session.slots.clear()
        self.session.demoted.clear()
        self._pending_streaming_session_demotions.clear()
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        # Rids whose storage prefetch resolved without a usable result;
        # popped by the scheduler to pace availability-check retries.
        self._storage_prefetch_missed_rids: set[str] = set()
        self.ongoing_backup: dict[int, tuple[NodeId, DecLockRefParams]] = {}
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.reset()

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self.tree_core.kv_events.record_all_cleared()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        self.host_memory_mode = get_memory().hicache_host_memory_mode
        if self.host_memory_mode == "buffer_only":
            # FULL and FULL+SWA only: Mamba has no state-handoff channel on
            # the admission-time load-back read path and is not layer-gated.
            # Lifting the fence also needs the admission charge: a staged
            # state slot is request-pinned at consumption and must ride
            # req.mamba_host_hit_length the way the SWA window does.
            supported = {ComponentType.FULL, ComponentType.SWA}
            if not set(self.tree_components) <= supported:
                raise ValueError(
                    "--hicache-host-memory-mode buffer_only supports only "
                    "FULL/SWA unified trees; got components "
                    f"{sorted(ct.name for ct in self.tree_components)}."
                )
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = get_observability().extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = get_memory().hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                get_memory().hicache_storage_backend_extra_config
            )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )
        # Tag HiCache enablement on the TreeCore.
        if self.cache_controller is not None:
            self.tree_core.set_hicache_enabled()
            if self.supports_swa():
                swa = self.components[ComponentType.SWA]
                self.tree_core.has_swa_host_pool = swa._swa_kv_pool_host is not None

        if self.host_memory_mode == "buffer_only":
            swa = self.components.get(ComponentType.SWA)
            validate_buffer_only_stack(
                sidecar_pool_specs=self.sidecar_pool_specs, swa_component=swa
            )
            self.buffer_pipeline = BufferModePipeline(
                cache=self,
                max_context_len=server_args.context_length or 0,
                swa_window_pages=(
                    swa.full_window_pages
                    if swa is not None and self.tree_core.has_swa_host_pool
                    else 0
                ),
                # Leak backstop only: live queued tokens are intrinsically
                # bounded by the FULL device pool (one intent per node, stale
                # intents swept per tick), so a cap that binds on live
                # content would drop-newest and punch storage holes.
                write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
            )
            self.cache_controller.host_write_staged_tokens_fn = (
                lambda: self.buffer_pipeline.write_staged_tokens_
            )

        # L2 backup policy is independent of whether an L3 backend is attached.
        self._apply_hicache_write_policy(get_memory().hicache_write_policy)
        # Pre-seed the dropped-tokens series at 0 per pool
        if self.metrics_collector is not None and self.cache_controller is not None:
            for ct in self.tree_components:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=0,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = get_memory().hicache_storage_prefetch_policy

        # Runtime attach/detach of the L3 backend (startup, admin API, atexit).
        self._storage_attachment = StorageAttachment(self)
        atexit.register(self.shutdown)

        if storage_backend is not None:
            self._storage_attachment.apply_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )

    def _apply_hicache_write_policy(self, write_policy: str) -> None:
        """Apply the host-backup policy to the controller and unified tree.

        :param write_policy: Validated HiCache write policy.
        """

        if self.cache_controller is None:
            raise RuntimeError("HiCache write policy requires a cache controller")
        self.cache_controller.write_policy = write_policy
        self.write_through_threshold = 1 if write_policy == "write_through" else 2
        self.is_write_back = write_policy == "write_back"
        logger.info("Set hicache_write_policy to %s", write_policy)

    def register_sidecar_pool(
        self, spec: SidecarPoolSpec, entry: Optional[PoolEntry] = None
    ) -> None:
        if entry is not None:
            if self.cache_controller is None:
                raise RuntimeError("HiCache controller is not attached.")
            self.cache_controller.register_host_pool_entry(entry)
        self.sidecar_pool_specs.append(spec)

    def release_host_resources(self) -> None:
        if self.host_pool_group is not None:
            self.host_pool_group.destroy()

    @rank_consensus(
        same_params=["params"],
        same_results=["result.full_kv_hit_length", "result.swa_host_hit_length"],
    )
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        if self.disable:
            return self.tree_core.empty_match_result
        result = self.tree_core.match_prefix(params)
        # Apply the walk's actions (e.g. a pending write-through relocation on
        # a split) before the finalizers, which can evict or raise.
        self._apply_cache_actions(result.cache_actions)
        for component in self._components_tuple:
            result = component.finalize_match_result_in_cache(params, result)
        # Finalizers must not emit actions; the walk's were applied above.
        assert not result.cache_actions
        self.session.restore_demoted_request_state(
            params.req, result.full_kv_hit_length
        )
        return result

    def is_chunk_cache(self) -> bool:
        return self.disable

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        # Fail fast on re-entrancy without touching the in-flight walk.
        assert not self.tree_core.has_ongoing_insert(), "re-entrant insert"
        # Pump the resumable insert, applying each step's actions at its barrier.
        try:
            step = self.tree_core.begin_insert(params)
            while True:
                self._apply_cache_actions(step.actions)
                if step.result is not None:
                    # Walk actions flow through the steps; the result is action-free.
                    assert not step.result.cache_actions
                    return step.result
                step = self.tree_core.resume_insert()
        finally:
            # Drain still-pending actions so frees reach the allocator on abort.
            self._apply_cache_actions(self.tree_core.end_insert())

    def evict(self, params: EvictParams) -> EvictResult:
        return self._evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict until the requested component allocations become feasible.

        ``params`` contains allocator shortfalls, not absolute eviction quotas.
        A component eviction can cascade to its peers; with a shared memory pool,
        those collateral frees can satisfy the original allocation before the
        triggering component's requested count is reached.
        """
        if self.disable:
            return EvictResult()

        request_by_type = self._evict_request_by_type(params)
        available_size_targets = {
            ct: self._component_available_size(ct) + request_cnt
            for ct, request_cnt in request_by_type.items()
            if request_cnt > 0
        }
        return self._evict(params, available_size_targets)

    @staticmethod
    def _evict_request_by_type(params: EvictParams) -> dict[ComponentType, int]:
        return {
            ComponentType.FULL: params.num_tokens,
            ComponentType.SWA: params.swa_num_tokens,
            ComponentType.MAMBA: params.mamba_num,
            ComponentType.C128: 0,
        }

    def _component_available_size(self, component_type: ComponentType) -> int:
        """Return capacity usable by the component's next allocation.

        Shared allocators expose schedulable capacity, which includes peer holes
        that an urgent allocator flush can reclaim without further eviction.
        """
        if component_type == ComponentType.FULL:
            if self.supports_swa():
                return self.token_to_kv_pool_allocator.full_available_size()
            return self.token_to_kv_pool_allocator.available_size()
        if component_type == ComponentType.SWA:
            return self.token_to_kv_pool_allocator.swa_available_size()
        if component_type == ComponentType.MAMBA:
            return self.req_to_token_pool.mamba_allocator.schedulable_available_size()
        raise ValueError(f"Unsupported cache component: {component_type}")

    def _evict(
        self,
        params: EvictParams,
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        request_by_type = self._evict_request_by_type(params)
        self._evict_components(
            request_by_type,
            tracker,
            available_size_targets=available_size_targets,
        )

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        # Report full-layer tokens only
        self.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def _free_values(
        self,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free a tree-side step's returned device and host values right away."""
        # Both drains must run even if one raises.
        try:
            self._drain_device_frees(device_frees)
        finally:
            self._drain_host_frees(host_frees)

    def _accumulate_tracker(
        self,
        tracker: dict[ComponentType, int],
        delta: dict[ComponentType, int],
    ) -> None:
        """Fold a step result's evicted delta into the running totals."""
        for ct, n in delta.items():
            tracker[ct] += n

    def _evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> tuple[Optional[NodeId], bool]:
        """Advance the eviction walk one node, consuming its step result."""
        result = self.tree_core.evict_device_next_node(component_type, tracker)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.node_id, result.made_progress

    def _evict_device_leaf(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> Optional[BackupKV]:
        """Evict one device leaf, consuming its step result; returns the
        deferred write-back BackupKV when one must run before the demote."""
        result = self.tree_core.evict_device_leaf(node_id, self.is_write_back)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.backup_kv

    def _demote(self, node_id: NodeId, tracker: dict[ComponentType, int]) -> None:
        """Demote a backed-up node, consuming its step result."""
        result = self.tree_core.demote(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)

    def _drop_subtree_no_host(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> bool:
        """Run the write-back drop fallback, consuming its step result."""
        result = self.tree_core.drop_subtree_no_host(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.is_dropped

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> None:
        # Buffer mode: eviction always wins over queued backup intents — a
        # destroyed victim's intent is stale-swept and the content rewrites
        # after its recompute.

        def target_reached(component_type: ComponentType) -> bool:
            if available_size_targets is None:
                return False
            target = available_size_targets.get(component_type)
            # Do not compact on every eviction step. Shared allocators include
            # drainable peer holes here and flush the peer once in alloc().
            return (
                target is not None
                and self._component_available_size(component_type) >= target
            )

        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # A preceding component may have cascade-evicted this component or,
            # on a shared pool, released enough bytes to satisfy its allocation.
            if tracker[ct] >= request_cnt or target_reached(ct):
                continue
            self.tree_core.evict_device_start(ct, request_cnt)
            try:
                while not target_reached(ct):
                    node_id, made_progress = self._evict_device_next_node(ct, tracker)
                    if node_id is None:
                        if made_progress:
                            # Internal tombstone frees are now allocator-visible;
                            # recheck the allocation target before walking again.
                            continue
                        break
                    backup_kv = self._evict_device_leaf(node_id, tracker)
                    if backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        written = self._execute_and_commit_kv_backup(
                            backup_kv, write_back=True
                        )
                        freed_before_drop = dict(tracker)
                        if written > 0:
                            self.writing_check(write_back=True)
                            self._demote(node_id, tracker)
                        elif self._drop_subtree_no_host(node_id, tracker):
                            self._record_dropped_tokens(tracker, freed_before_drop)
                            logger.warning(
                                "write_back: KV subtree dropped without backup "
                                "due to host memory pressure, root node %d",
                                node_id,
                            )
                        else:
                            logger.warning(
                                "write_back: backup failed under host memory "
                                "pressure but subtree drop declined (node "
                                "locked); root node %d stays device-resident "
                                "until host space frees",
                                node_id,
                            )
            finally:
                self.tree_core.evict_device_end(ct)

    def _record_dropped_tokens(
        self,
        tracker: dict[ComponentType, int],
        freed_before_drop: dict[ComponentType, int],
    ) -> None:
        """Record per-pool tokens dropped without backup under host pressure."""
        if self.metrics_collector is None:
            return
        for ct, freed in tracker.items():
            dropped = freed - freed_before_drop[ct]
            if dropped > 0:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=dropped,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node_id)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id, skip_lock_components)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node_id, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_lock_ref(node_id, params, skip_swa)

    def _dec_req_lock(self, req: Req, *, skip_swa: bool = False) -> None:
        """Release the tree lock a request holds on its last_node, honoring the
        components it skipped locking so it never drops a lock it never took."""
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
            skip_swa=skip_swa,
        )

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int] = None,
        skip_lock_node_ids: Optional[dict] = None,
    ) -> None:
        if self.disable:
            return
        result = self.tree_core.dec_swa_lock_only(
            node_id, swa_uuid_for_lock, skip_lock_node_ids
        )
        self._free_values(result.device_frees, result.host_frees)

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_host_lock_ref(node_id)

    def validate_host_lock_ref(
        self,
        node_id: NodeId,
        params: DecLockRefParams,
    ) -> None:
        if self.disable:
            return
        self.tree_core.validate_host_lock_ref(node_id, params)

    def dec_host_lock_ref(
        self, node_id: NodeId, params: DecLockRefParams
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_host_lock_ref(node_id, params)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free_segment(kv_indices, start_pos=0)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.cache_protected_len,
                priority=getattr(req, "priority", 0) or 0,
            )

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed; the tail free is deferred and batched with
            # the unaligned tail below so a shared boundary page is emitted once.
            kv_indices_full = kv_indices
            tail_free_start = None
            if effective_cache_len < len(token_ids):
                tail_free_start = max(effective_cache_len, req.cache_protected_len)
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids,
                req.extra_key,
                is_bigram=self.tree_core.is_eagle,
                cache_salt=req.cache_salt,
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail (+ deferred truncation tail)
            segments = [(kv_indices[page_aligned_len:], page_aligned_len)]
            if tail_free_start is not None:
                segments.append((kv_indices_full[tail_free_start:], tail_free_start))
            self.token_to_kv_pool_allocator.free_segments(segments)
        else:
            self.token_to_kv_pool_allocator.free_segment(
                kv_indices[req.cache_protected_len :],
                start_pos=req.cache_protected_len,
            )

        self._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)

        if is_insert and result is not None and result.last_device_node is not None:
            req.last_node = result.last_device_node

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

        if self.enable_session_radix_cache and result is not None:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            if req.finished_reason is not None and not isinstance(
                req.finished_reason, FINISH_ABORT
            ):
                self.session_refs.register_session_ref(req)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        is_streaming = req.session is not None and req.session.streaming
        # A first-turn streaming insert runs before speculative decoding has
        # finished materializing its draft KV. The session lock keeps the
        # device prefix alive; explicit demotion publishes one coherent
        # target/draft snapshot after the request is idle.
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
            trigger_backup=not is_streaming,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )

        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            # The frontier lands a page below page_floor(pre_len + 1), which has to
            # be where the insert stops, or the leaf it creates keeps less than a
            # sliding window of live SWA and the match after the insert rejects it.
            # The insert stops at page_floor(len(radix_key)), and a bigram key is
            # one shorter than the tokens it spans, so measure the key.
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = radix_key.page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix. SWA insertion retains one extra window before the
        # page-aligned boundary, so the normal match remains safe to repoint.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key, req=req))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self._dec_req_lock(req)
        # Opt-in: leave the matched-prefix mamba evictable during decode (it is
        # already COW'd to the request's own slot, never read from this node again).
        # Safe only because any future COW source is the COWing request's own
        # admission-locked last_node (recorded only if still present, locked before
        # the next alloc) -- not this evictable node. A scheduler that matched a
        # whole batch before locking would break that. Off = original full lock.
        skip_lock_components = (
            (ComponentType.MAMBA,)
            if envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
            else ()
        )
        lock_result = self.inc_lock_ref(
            new_last_node, skip_lock_components=skip_lock_components
        )

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        # carry the skip set so this node's dec releases only what we locked
        req.skip_lock_node_ids = lock_result.skip_lock_node_ids
        # The rematch acquired a new SWA prefix lock.
        req.swa_prefix_lock_released = False

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _apply_cache_actions(
        self, actions: list[CacheAction | ComponentAction]
    ) -> None:
        # Apply and consume one at a time: a spent list cannot be double-applied.
        actions.reverse()
        try:
            while actions:
                self._apply_cache_action(actions.pop())
        finally:
            actions.reverse()

    def _apply_cache_action(self, action: CacheAction | ComponentAction) -> None:
        # Component actions route to their component class; the rest are
        # cache-owned and handled here by type.
        if isinstance(action, ComponentAction):
            self.components[action.component_type].apply_component_action(action)
        elif isinstance(action, ReplaceWriteThroughOnNodeSplit):
            self._replace_pending_write_through_node(
                action.ack_id,
                action.old_node_id,
                [action.new_node_id, action.new_child_node_id],
            )
        elif isinstance(action, FreeDeviceKV):
            # tree values are page-aligned copies of a kv row: page-exact segments
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_segment(indices, start_pos=0)
        elif isinstance(action, FreeDeviceKVFullOnly):
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_full(indices)
        elif isinstance(action, BackupKV):
            self._execute_and_commit_kv_backup(action)
        else:
            raise AssertionError(f"unhandled CacheAction: {type(action).__name__}")

    def _drain_device_frees(
        self, device_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component device slots, consuming each entry as it frees.
        for ct in list(device_frees):
            self._apply_cache_action(
                FreeComponentDeviceSlot(device_frees.pop(ct), component_type=ct)
            )

    def _drain_host_frees(
        self, host_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component host-pool slots, consuming each entry as it frees.
        for ct in list(host_frees):
            self.components[ct].free_host_values(host_frees.pop(ct))

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        if self.host_memory_mode == "buffer_only":
            # The tree never holds host values in buffer mode, and staging
            # is operation-owned (freed at each ack): nothing is evictable.
            return 0
        result = self.tree_core.drive_host_eviction(component_type, num_tokens)
        self._free_values(result.device_frees, result.host_frees)
        return result.tracker.get(component_type, 0)

    # ---- Decode retraction ----

    def supports_retraction_backup(self) -> bool:
        if self.cache_controller is None or self.host_pool_group is None:
            return False
        if self.supports_mamba():
            return False

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, SWAKVPool):
            return (
                self.supports_swa()
                and {
                    PoolName.KV,
                    PoolName.SWA,
                }
                <= self.host_pool_group.entry_map.keys()
            )
        return isinstance(kv_cache, MHATokenToKVPool) and (
            PoolName.KV in self.host_pool_group.entry_map
        )

    def validate_retraction_host_capacity(self) -> None:
        if not self.supports_retraction_backup():
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "an MHA or hybrid-SWA HiCache host stack."
            )

        for spec in self.sidecar_pool_specs:
            source_size = self.host_pool_group.entry_map[
                spec.indices_from_pool
            ].host_pool.logical_size
            sidecar_size = self.host_pool_group.entry_map[
                spec.pool_name
            ].host_pool.logical_size
            if sidecar_size < source_size:
                raise ValueError(
                    "Retraction sidecar host pool is smaller than its index source: "
                    f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                    f"source={spec.indices_from_pool}, source_slots={source_size}."
                )

    @staticmethod
    def _pad_retraction_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
        aligned_len = ceil_align(len(indices), page_size)
        if aligned_len == len(indices):
            return indices
        tail = indices[-1] + torch.arange(
            1,
            aligned_len - len(indices) + 1,
            dtype=torch.int64,
            device=indices.device,
        )
        return torch.cat([indices, tail])

    def _device_transfers_from_indices(
        self,
        full_indices: torch.Tensor,
        num_tokens: int,
    ) -> list[PoolTransfer]:
        """Build auxiliary transfer descriptors for one full-KV index span.

        :param full_indices: Full-attention device slots in lineage order.
        :param num_tokens: Logical token count represented by the slots.
        :returns: SWA and draft sidecar transfer descriptors.
        """
        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if self.supports_swa():
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            assert self.sliding_window_size is not None
            window_start = max(0, num_tokens - self.sliding_window_size)
            window_start = window_start // self.page_size * self.page_size
            window_indices = full_indices[window_start:num_tokens]
            swa_indices = kv_cache.translate_loc_from_full_to_swa(window_indices)
            assert bool(
                (swa_indices > 0).all()
            ), "unmapped SWA window positions in host-backup span"
            component_transfers[ComponentType.SWA] = [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self._pad_retraction_indices(
                        swa_indices, self.page_size
                    ),
                )
            ]

        kv_transfer = PoolTransfer(name=PoolName.KV, device_indices=full_indices)
        extra_transfers = [
            transfer
            for transfers in component_transfers.values()
            for transfer in transfers
        ]
        extra_transfers.extend(
            self._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_HOST,
                kv_transfer,
                component_transfers,
            )
        )
        return extra_transfers

    def _retraction_device_transfers(
        self, req: Req
    ) -> tuple[torch.Tensor, list[PoolTransfer]]:
        num_tokens = req.seqlen - 1
        full_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :num_tokens
        ].to(torch.int64)
        full_indices = self._pad_retraction_indices(full_indices, self.page_size)
        return full_indices, self._device_transfers_from_indices(
            full_indices,
            num_tokens,
        )

    def _reclaim_retraction_host(self, num_tokens: int) -> int:
        if self.disable:
            return 0
        return self.evict_host(num_tokens)

    def _stage_retraction_backup(
        self,
        device_indices: torch.Tensor,
        extra_transfers: list[PoolTransfer],
    ) -> RetractionBackup | None:
        """Copy one complete transfer set into privately owned host slots.

        :param device_indices: Full-attention device slots to copy.
        :param extra_transfers: Auxiliary pool transfers sharing the transaction.
        :returns: Private host stage, or ``None`` when capacity cannot be reserved.
        """
        host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            self._reclaim_retraction_host(len(device_indices))
            host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            return None

        resolved = self.host_pool_group.resolve_host_transfers(
            extra_transfers or None,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
        )
        if resolved is None and extra_transfers:
            self.host_pool_group.free(host_indices)
            return None

        backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=[replace(x, device_indices=None) for x in resolved or []]
            or None,
        )
        operation = CacheOperation(
            host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        try:
            write_host, write_device, write_pools = (
                self.cache_controller._move_write_operation(operation)
            )
            completion = self.cache_controller.l2_transfer_engine.submit_device_to_host(
                self.cache_controller._l2_transfers(
                    write_host, write_device, write_pools
                )
            )
            if injection_enabled():
                pause_point(
                    "demote_d2h_inflight",
                    self._pause_point_rank(),
                    {
                        "device_tokens": int(len(device_indices)),
                        "host_tokens": int(len(host_indices)),
                        "extra_pools": [str(transfer.name) for transfer in resolved or []],
                        "finish_event_done": bool(completion.finish_event.query()),
                    },
                )
            completion.finish_event.synchronize()
        except Exception:
            self.retraction_discard(backup)
            raise
        return backup

    def retraction_backup(self, req: Req) -> Optional[RetractionBackup]:
        """Back up device KV to the host pool; None when it cannot fit after reclaim."""
        assert req.seqlen > 1

        device_indices, extra_transfers = self._retraction_device_transfers(req)
        return self._stage_retraction_backup(device_indices, extra_transfers)

    def retraction_restore(self, req: Req, backup: RetractionBackup) -> None:
        device_indices, current_transfers = self._retraction_device_transfers(req)
        assert len(backup.host_indices) == len(device_indices), (
            f"Host backup has {len(backup.host_indices)} slots, but restore has "
            f"{len(device_indices)}"
        )

        current_by_name = {transfer.name: transfer for transfer in current_transfers}
        saved_by_name = {
            transfer.name: transfer for transfer in backup.pool_transfers or []
        }
        assert current_by_name.keys() == saved_by_name.keys(), (
            f"Host backup pools {set(saved_by_name)} do not match restore pools "
            f"{set(current_by_name)}"
        )
        restored_transfers = [
            replace(
                saved,
                device_indices=current_by_name[name].device_indices,
            )
            for name, saved in saved_by_name.items()
        ]
        resolved = self.cache_controller._resolve_device_transfers(
            restored_transfers or None,
            kv_device_indices=device_indices,
            kv_host_indices=backup.host_indices,
        )
        assert resolved is not None or not restored_transfers

        operation = CacheOperation(
            backup.host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        load_host, load_device, load_pools = self.cache_controller._move_op_indices(
            operation
        )
        completion = self.cache_controller.l2_transfer_engine.submit_host_to_device(
            self.cache_controller._l2_load_transfers(
                load_host, load_device, load_pools
            ),
            layer_num=self.cache_controller.layer_num,
        )
        completion.finish_event.synchronize()
        self.retraction_discard(backup)

    def retraction_discard(self, backup: RetractionBackup) -> None:
        self.host_pool_group.free(backup.host_indices)
        self.host_pool_group.release_transfers(backup.pool_transfers)

    # ---- HiCache: Backup / LoadBack ----

    def _execute_and_commit_kv_backup(
        self, action: BackupKV, write_back: bool = False
    ) -> int:
        """Run a backup action top-down, stopping at the first failed backup."""
        if self.buffer_pipeline is not None:
            # Buffer mode bypasses the host-backup contiguity below: nothing
            # is ever host-backuped here. Contiguity comes from end-to-end
            # FIFO ordering instead (BackupKV chains are parent-before-child
            # and every pipeline stage drains in order).
            for node_id in action.node_ids:
                self.buffer_pipeline.enqueue_backup_intent(
                    self.tree_core.node_by_id(node_id)
                )
            return 0
        written = 0
        for node_id in action.node_ids:
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
            # Overlapping chain actions may revisit nodes with Full KV already
            # backed up. Skip only when no transfer remains.
            if device_value.numel() == 0 and not comp_xfers:
                continue
            sidecar_xfers = self._build_backup_sidecar(device_value, comp_xfers)
            host_indices = self._execute_kv_backup(
                node_id, device_value, comp_xfers, sidecar_xfers
            )
            if host_indices is None:
                return 0
            self.tree_core.commit_backup(node_id, host_indices, comp_xfers)
            lock_params = None
            if not write_back:
                lock_params = self.inc_lock_ref(node_id).to_dec_params()
            self._track_write_through_node(node_id, lock_params)
            written = len(host_indices)
        return written

    def _build_backup_sidecar(self, device_value, comp_xfers):
        """Gather sidecar transfer spec."""
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)
        return self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

    def _execute_kv_backup(self, node_id, device_value, comp_xfers, sidecar_xfers):
        """Execute Backup action."""
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            if self.evict_host(needed) < needed:
                return None
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        return self.cache_controller.write(
            device_value, node_id=node_id, extra_pools=aux_xfers or None
        )

    def _track_write_through_node(
        self,
        node_id: NodeId,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        self.tree_core.mark_write_through_pending(node_id)
        self.ongoing_write_through[node_id] = _OngoingWriteThrough(
            node_id, lock_params, [node_id]
        )

    def _replace_pending_write_through_node(
        self, ack_id: int, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node_id, lock_params, publish_node_ids = pending
        updated_node_ids = []
        replaced = False
        for node_id in publish_node_ids:
            if node_id == old_node_id:
                updated_node_ids.extend(new_node_ids)
                replaced = True
            else:
                updated_node_ids.append(node_id)

        if not replaced:
            return

        self.ongoing_write_through[ack_id] = _OngoingWriteThrough(
            lock_node_id,
            lock_params,
            updated_node_ids,
        )

    def _finish_write_through_ack(self, ack_id: int) -> None:
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.finish_backup_ack(ack_id)
            return

        lock_node_id, lock_params, publish_node_ids = self.ongoing_write_through.pop(
            ack_id
        )
        self.tree_core.finish_write_through(publish_node_ids, ack_id)
        if lock_params is not None:
            self.dec_lock_ref(lock_node_id, lock_params)
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node_id in publish_node_ids:
                self.write_backup_storage(node_id)

    def load_back(
        self,
        node_id: NodeId,
        mem_quota: int | None = None,
        req: Req | None = None,
    ) -> _QueuedLoadBackResult | None:
        """Prepare an evicted component payload for host-to-device loading.

        :param node_id: Deepest radix node covered by the restoration.
        :param mem_quota: Maximum additional full-KV allocation, if constrained.
        :param req: Request receiving per-request component state.
        :returns: Metadata for the queued component payload, or ``None`` on failure.
        """
        if self.cache_controller is None:
            return None

        host_anchor_params = self.inc_host_lock_ref(node_id).to_dec_params()
        ancestor_lock_params: DecLockRefParams | None = None
        preps: dict[ComponentType, PrepareLoadBackResult] = {}
        load_result: _QueuedLoadBackResult | None = None
        try:
            # Lock the path before building transfers (the aux build can evict).
            result = self.inc_lock_ref(node_id)
            ancestor_lock_params = result.to_dec_params()
            for component in self._components_tuple:
                preps[component.component_type] = component.prepare_load_back(
                    node_id,
                    req=req,
                )
            load_result = self._load_back_transfers(
                node_id=node_id,
                mem_quota=mem_quota,
                req=req,
                result=result,
                host_anchor_params=host_anchor_params,
            )
            return load_result
        finally:
            try:
                if ancestor_lock_params is not None:
                    self.dec_lock_ref(node_id, ancestor_lock_params)
            finally:
                try:
                    if load_result is None:
                        self.dec_host_lock_ref(node_id, host_anchor_params)
                finally:
                    success = load_result is not None
                    for component in self._components_tuple:
                        prep = preps.get(component.component_type)
                        if prep is not None:
                            component.finalize_load_back(req, prep, success)

    def _load_back_transfers(
        self,
        *,
        node_id: NodeId,
        mem_quota: int | None,
        req: Req | None,
        result: IncLockRefResult,
        host_anchor_params: DecLockRefParams,
    ) -> _QueuedLoadBackResult | None:
        """Allocate and commit controller-owned component transfers.

        :param node_id: Deepest radix node covered by the restoration.
        :param mem_quota: Maximum additional full-KV allocation, if constrained.
        :param req: Request receiving per-request component state.
        :param result: Device-tree lock acquisition result.
        :param host_anchor_params: Host-tree lease release parameters.
        :returns: Metadata for the queued component payload, or ``None`` on failure.
        """

        # Build the KV + per-component aux transfers.
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id, req=req)
        kv_tokens = len(kv_xfer.host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )
        aux_xfers = [xfer for xfers in comp_xfers.values() for xfer in xfers]
        aux_xfers.extend(sidecar_xfers)

        has_component_payload = kv_tokens > 0 or any(
            xfer.host_indices is not None and xfer.host_indices.numel() > 0
            for xfer in aux_xfers
        )
        if not has_component_payload:
            return None

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding. max(1, ...): an
        # entirely empty spec (e.g. foreign-pin rejection) must never report
        # success, even at load_back_threshold <= 0.
        if (kv_tokens < max(1, self.load_back_threshold) and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            return None

        avail = self._component_available_size(ComponentType.FULL)
        if avail < kv_tokens:
            needed = kv_tokens - avail
            self.evict_for_alloc(EvictParams(num_tokens=needed))
            if self._component_available_size(ComponentType.FULL) < kv_tokens:
                return None

        # Load H→D
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=node_id,
            extra_pools=aux_xfers or None,
        )

        if device_indices is None:
            return None

        kv_xfer.device_indices = device_indices
        full_tokens = _queued_transfer_token_count(kv_xfer)
        logical_full_tokens = sum(
            self.tree_core.component_logical_length(
                self.tree_core.node_by_id(loaded_node_id),
                ComponentType.FULL,
                host=True,
            )
            for loaded_node_id in kv_xfer.nodes_to_load or ()
        )
        swa_tokens = sum(
            _queued_transfer_token_count(xfer)
            for xfer in comp_xfers.get(ComponentType.SWA, ())
        )

        # Commit the loaded KV back onto the node + apply its emitted actions.
        self._apply_cache_actions(
            self.tree_core.commit_load_back(
                node_id, device_indices, kv_xfer, comp_xfers
            )
        )

        queued_result = _QueuedLoadBackResult(
            new_full_device_indices=device_indices[:logical_full_tokens],
            full_tokens=full_tokens,
            swa_tokens=swa_tokens,
        )
        self.ongoing_load_back[node_id] = _OngoingLoadBack(
            node_id,
            self.inc_lock_ref(node_id).to_dec_params(),
            host_anchor_params,
        )
        return queued_result

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            defer_kv_sidecar = (
                phase == CacheTransferPhase.PREFETCH
                and spec.indices_from_pool == PoolName.KV
            )
            if (indices is None or len(indices) == 0) and not defer_kv_sidecar:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def write_backup_storage(self, node_id: NodeId) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return
        spec = self.tree_core.build_storage_backup_spec(
            node_id, self.hicache_storage_pass_prefix_keys
        )
        if spec is None:
            return

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=spec.host_value,
            keys=spec.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, spec.comp_xfers
        )
        aux_xfers = [x for xfers in spec.comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        lock_params = self.inc_host_lock_ref(node_id).to_dec_params()
        registered = False
        try:
            operation_id = self.cache_controller.write_storage(
                spec.host_value,
                spec.token_ids,
                spec.hash_value,
                spec.prefix_keys,
                extra_pools=aux_xfers or None,
            )
            self.ongoing_backup[operation_id] = (node_id, lock_params)
            registered = True
        finally:
            if not registered:
                self.dec_host_lock_ref(node_id, lock_params)

    def is_backuped(self, node_id: NodeId) -> bool:
        return self.tree_core.is_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self.tree_core.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self.tree_core.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self.tree_core.get_prefix_hash_values(node_id)

    def query_storage_hit_length(
        self,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> int:
        """Synchronously probe L3 storage for the reusable prefix length."""
        if (
            not self.enable_storage
            or self.cache_controller is None
            or self.cache_controller.prefetch_rate_limited()
        ):
            return 0

        extra_key, cache_salt = self.tree_core.prefetch_anchor_info(last_host_node_id)
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            PrefetchOperation,
        )

        operation = PrefetchOperation(
            "__storage_hit_query__",
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        _, storage_hit_count = self.cache_controller._storage_hit_query(operation)
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count -= storage_hit_count % self.page_size
        return storage_hit_count

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return

        buffer_mode = self.host_memory_mode == "buffer_only"
        # Key the span by the request's namespace, not the anchor's (a root
        # anchor has none): a span published under the wrong namespace gets
        # re-owned by the request's own insert (double free).
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        assert (anchor_extra_key is None or anchor_extra_key == extra_key) and (
            anchor_cache_salt is None or anchor_cache_salt == cache_salt
        ), (
            f"prefetch anchor namespace {(anchor_extra_key, anchor_cache_salt)} "
            f"!= request namespace {(extra_key, cache_salt)}"
        )
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        stats = self._prefetch_outcome_stats
        if prefetch_length > 0:
            stats["attempts"] += 1
        if prefetch_length < self.prefetch_threshold:
            if prefetch_length > 0:
                stats["declined_too_short"] += 1
            # A too-short/fully-matched suffix can become a full recompute if
            # the device match evicts while queued; arm the paced retry.
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if not buffer_mode and self.cache_controller.prefetch_rate_limited():
            stats["declined_rate_limited"] += 1
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if req_id in self.ongoing_prefetch or (
            buffer_mode and self.buffer_pipeline.has_staged(req_id)
        ):
            # A fetch (or an unconsumed hold) already exists for this rid;
            # overwriting would leak its staging slots.
            return

        # Buffer mode holds no tree state during the fetch: buffers are
        # operation-owned, so the anchor needs no pin.
        anchor_lock_params = (
            None
            if buffer_mode
            else self.inc_host_lock_ref(last_host_node_id).to_dec_params()
        )
        prefetch_registered = False
        prepared_component_buffers: list[PoolTransfer] = []
        try:
            comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
            alloc_failed = False
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                # Pre-allocate the component's prefetch host buffer so the build stays pure.
                prep = self.components[ct].prepare_prefetch(
                    last_host_node_id, prefetch_tokens=len(prefetch_key)
                )
                if prep.alloc_failed:
                    alloc_failed = True
                    break
                if prep.host_indices is None:
                    continue
                pool_name = _COMPONENT_HOST_POOL.get(ct)
                if pool_name is None:
                    raise AssertionError(
                        f"Component {ct} allocated an unsupported prefetch buffer."
                    )
                prepared_component_buffers.append(
                    PoolTransfer(name=pool_name, host_indices=prep.host_indices)
                )
                transfers = self.tree_core.build_hicache_transfers(
                    ct,
                    last_host_node_id,
                    CacheTransferPhase.PREFETCH,
                    token_ids=prefetch_key.token_ids,
                    prefetch_tokens=len(prefetch_key),
                    last_hash=last_hash,
                    host_indices=prep.host_indices,
                )
                if not transfers:
                    raise AssertionError(
                        f"Component {ct} allocated a prefetch buffer without a transfer."
                    )
                comp_xfers[ct] = transfers
            kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=None)
            sidecar_xfers = self._build_sidecar_transfers(
                CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
            )
            if alloc_failed:
                # The whole storage fetch is forfeited over one aux staging
                # alloc (e.g. a single SWA window) — count it, or write-burst
                # starvation of the aux pool reads as generic hit-rate loss.
                if (
                    self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_prefetch_aux_alloc_failed_tokens(
                        len(prefetch_key)
                    )
                # Forfeited over transient staging pressure; retryable.
                self._storage_prefetch_missed_rids.add(req_id)
                return

            aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
            aux_xfers.extend(sidecar_xfers)
            operation = self.cache_controller.prefetch(
                req_id,
                prefetch_key,
                last_hash,
                prefix_keys,
                extra_pools=aux_xfers or None,
            )
            self.ongoing_prefetch[req_id] = _OngoingPrefetch(
                last_host_node_id,
                prefetch_key,
                None,
                operation,
                anchor_lock_params,
                comp_xfers,
            )
            prefetch_registered = True
            stats["issued"] += 1
            # Snapshots for the L3 miss accounting at the query outcome (the
            # hit/revoke drains): requested span and total prompt length.
            operation.stats_requested_tokens = prefetch_length
            operation.stats_total_tokens = prefetch_length + len(
                matched_prefix_tokens or []
            )
        finally:
            if not prefetch_registered:
                try:
                    if len(prepared_component_buffers) > 0:
                        self.cache_controller.append_host_mem_release(
                            extra_pools=prepared_component_buffers,
                        )
                finally:
                    if anchor_lock_params is not None:
                        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        if buffer_mode:
            self.buffer_pipeline.set_prefix_ctx(
                req_id,
                matched_prefix_tokens,
                extra_key=extra_key,
                cache_salt=cache_salt,
            )
            # Pin the just-matched anchor now: deferred to IO commit it is
            # often already deleted under churn. The IO-commit call remains
            # as the second chance that decides the fetch's fate.
            self.buffer_pipeline.try_lock_anchor(req_id)
        else:
            # Cache mode reserves the requested span up front; buffer mode
            # grants occupancy later at hit-alloc time, sized to the hit.
            self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    @rank_consensus(same_results=True)
    def _can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True
        if self.prefetch_stop_policy == "wait_complete":
            return False
        elif self.prefetch_stop_policy == "timeout":
            # Wall-clock time may differ among ranks, all-reduce is needed to ensure
            # all ranks reach the same final result. Otherwise PP/TP ranks will diverge.
            #
            # For TP, if any rank reaches the timeout, the final result is timeout.
            #
            # For PP, PP0 makes the decision and other ranks follow PP0's decision.
            should_terminate = False
            if self.pp_rank == 0:
                should_terminate = self._prefetch_timeout_check_linear_func(operation)
            should_terminate_tensor = torch.tensor(
                int(should_terminate), dtype=torch.int, device="cpu"
            )
            self._all_reduce(should_terminate_tensor, torch.distributed.ReduceOp.MAX)
            return should_terminate_tensor.item() == 1
        else:
            return True

    @rank_consensus(same_params=True, same_results=True)
    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]

        # Determine whether or not we should terminate this prefetch request.
        should_terminate = operation.is_terminated() or self._can_terminate_prefetch(
            operation
        )

        if not should_terminate:
            return False

        self.cache_controller.terminate_prefetch(operation)
        if operation.host_indices is None:
            self._storage_prefetch_missed_rids.add(req_id)
            self.revoke_pending_prefetch(req_id)
        else:
            self._handle_prefetch_result(operation)
        return True

    def _handle_prefetch_result(self, operation: PrefetchOperation) -> None:
        # This function **owns**:
        # - host_indices[0 : completed_tokens]
        # - sidecar pool hits if operation.pool_transfers_done is true
        #
        # That is, when this function returns the host memory referenced must be inserted
        # into the radix tree or released to pool.

        req_id = operation.request_id
        completed_tokens = operation.completed_tokens
        hash_value = operation.hash_value

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            _,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]

        # All PP/TP ranks will get the same `min_completed_tokens`, because `completed_tokens`
        # and `pool_hits` in their operations are same.  No need to sync cross-rank here.
        if not self._check_hybrid_prefetch_result(
            req_id,
            operation,
            completed_tokens,
            hash_value,
            host_indices,
            last_host_node_id,
            anchor_lock_params,
            prefetch_key,
        ):
            # Hybrid all-or-nothing check failed; result already discarded.
            return

        if self.buffer_pipeline is not None:
            # No graft: release the rank-local tail beyond the synced usable
            # length, then park the bounce for admission-time consumption.
            return self.buffer_pipeline.stage_completed_prefetch(
                req_id, completed_tokens, hash_value
            )

        fetched_key = prefetch_key[:completed_tokens]
        insert_result = self.tree_core.insert_host(
            last_host_node_id,
            fetched_key,
            host_indices[:completed_tokens],
            hash_value[: completed_tokens // self.page_size],
        )

        # Apply the host-insert walk's actions before the transfer commit.
        self._apply_cache_actions(insert_result.cache_actions)

        if insert_result.host_insert_dropped:
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            loaded_from_storage = 0
        else:
            commit_actions: list[CacheAction | ComponentAction] = []
            self.tree_core.commit_hicache_transfers(
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                comp_xfers,
                cache_actions=commit_actions,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )
            self._apply_cache_actions(commit_actions)
            # The commit emits via commit_actions only; the walk's were applied above.
            assert not insert_result.cache_actions

            self.cache_controller.mem_pool_host.free(
                host_indices[: insert_result.prefix_len]
            )
            loaded_from_storage = completed_tokens - insert_result.prefix_len

        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d",
            "dropped" if insert_result.host_insert_dropped else "success",
            req_id,
            completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            self.cache_controller.prefetch_tokens_occupied,
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return

    def _check_hybrid_prefetch_result(
        self,
        req_id: str,
        operation: PrefetchOperation,
        completed_tokens: int,
        hash_value: list[str],
        host_indices: torch.Tensor,
        last_host_node_id: NodeId,
        anchor_lock_params: DecLockRefParams,
        prefetch_key: RadixKey,
    ) -> bool:
        """Decide the length of usable prefix.

        Two strategies depending on the hybrid layout:

        * DSA-style (Full attention + KV-derived ALL_PAGES sidecar such as the
          DSA / MiniMax indexer): *clamp* to the minimum fetched prefix shared by
          the Full KV pool and every sidecar. A partial prefix is still usable
          because the sidecar is page-aligned with KV and required for every page.
        * Everything else (SWA / Mamba components, mixed DeepSeekV4 stacks):
          *all-or-nothing*. Their pools only cover a window / tail and cannot be
          truncated page by page, so any shortfall discards the whole prefetch.

        Returns true if prefetch success, or false when an all-or-nothing prefetch
        was discarded (the caller should then treat the prefetch as finished).
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        #
        # Skip KV-derived pools, which do not report hits in operation.pool_storage_result.
        # Their hit lengths are stored in completed_tokens.
        pool_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if transfer.indices_from_pool != PoolName.KV
        ]
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        completed_tokens = operation.completed_tokens
        # Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
        # must cover the same fetched prefix. If any pool falls short the whole
        # prefetch result is unusable, so discard it and release everything.
        expected_tokens = len(hash_value) * self.page_size
        all_succeeded = completed_tokens == expected_tokens and all(
            transfer.keys is not None and count == len(transfer.keys)
            for transfer, count in zip(pool_transfers, pool_hit_pages)
        )
        if pool_transfers and not all_succeeded:
            # Drop the KV beliefs from the first page any pool failed to serve;
            # the next insert then re-writes that span through one FULL check,
            # restoring the missing aux pages.
            keep_pages = completed_tokens // self.page_size
            for transfer, count in zip(pool_transfers, pool_hit_pages):
                if transfer.keys is None:
                    keep_pages = 0
                elif count < len(transfer.keys):
                    # Aux transfers key the chain's trailing pages.
                    keep_pages = min(
                        keep_pages, max(0, len(hash_value) - len(transfer.keys))
                    )
            self.storage_existence_cache.invalidate_beyond(
                PoolName.KV, hash_value, keep_pages=keep_pages
            )
            # The controller's prefetch IO thread already releases the untransferred
            # tail (host_indices[completed_tokens:])
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=pool_transfers if operation.pool_transfers_done else None,
            )
            if anchor_lock_params is not None:
                self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            if self.buffer_pipeline is not None:
                self.buffer_pipeline.pop_prefix_ctx(req_id)
                self.buffer_pipeline.release_anchor_lock(req_id)
            del self.ongoing_prefetch[req_id]
            self.cache_controller.prefetch_tokens_occupied -= (
                self._prefetch_occupied_span(prefetch_key, host_indices)
            )
            self.prefetch_loaded_tokens_by_reqid[req_id] = 0
            logger.warning(
                "HiCache hybrid prefetch discarded req=%s completed=%d requested=%d "
                "kv_beliefs_kept_pages=%d",
                req_id,
                completed_tokens,
                expected_tokens,
                keep_pages,
            )
            return False
        return True

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        # The request is being scheduled; a still-unserved miss marker is moot.
        self._storage_prefetch_missed_rids.discard(req_id)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def pop_storage_prefetch_miss(self, req_id: str) -> bool:
        """True once per resolved storage-prefetch miss for a live request;
        the scheduler uses it to arm the paced availability-check retry."""
        if req_id in self._storage_prefetch_missed_rids:
            self._storage_prefetch_missed_rids.discard(req_id)
            return True
        return False

    def plan_staged_splice(
        self, req_id: str, device_prefix_len: int
    ) -> tuple[int, int]:
        """(kv, swa) host-hit tokens a staged buffer-mode prefetch will splice
        given the request's live device prefix; frees unusable holds."""
        if self.buffer_pipeline is None:
            return 0, 0
        return self.buffer_pipeline.plan_staged_splice(req_id, device_prefix_len)

    def staged_prefetch_swa_tokens(self, req_id: str) -> int:
        """SWA device tokens consuming a staged buffer-mode prefetch will
        allocate; surfaced as the request's swa_host_hit_length."""
        if self.buffer_pipeline is None:
            return 0
        return self.buffer_pipeline.staged_prefetch_swa_tokens(req_id)

    @rank_consensus(same_params=True)
    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._storage_prefetch_missed_rids.discard(rid)
        if (
            self.buffer_pipeline is not None
            and self.buffer_pipeline.release_staged_hold(rid)
        ):
            return
        if rid not in self.ongoing_prefetch:
            return

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            self.cache_controller.terminate_prefetch(operation)
            self.revoke_pending_prefetch(rid)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(rid)
            self.buffer_pipeline.release_anchor_lock(rid)
        pool_transfers = [x for xfers in comp_xfers.values() for x in xfers]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=pool_transfers if operation.pool_transfers_done else None,
        )
        # Buffer mode granted occupancy at hit-alloc, sized to the bounce;
        # cache mode reserved the requested span at enqueue.
        self.cache_controller.prefetch_tokens_occupied -= self._prefetch_occupied_span(
            prefetch_key, host_indices
        )

    def _invalidate_absent_from_hit_query(self, operation) -> None:
        """Drop KV beliefs beyond the folded usable cut (rank-synced): the
        next insert then re-writes the node (all pools), healing stale
        positives and aux holes at the cut through one FULL check."""
        if self.host_memory_mode != "buffer_only":
            return
        chain = operation.all_hash_values
        if chain is None:
            return
        self.storage_existence_cache.invalidate_beyond(
            PoolName.KV, chain, keep_pages=operation.storage_hit_count // self.page_size
        )

    def _account_prefetch_outcome(self, operation, revoked: bool) -> None:
        """Feed the cumulative prefetch-outcome counters at the (rank-synced)
        query outcome: T = prompt tokens, L = requested, m = L3-miss."""
        requested = operation.stats_requested_tokens
        if requested <= 0:
            return
        stats = self._prefetch_outcome_stats
        hit = max(0, min(operation.storage_hit_count, requested))
        if revoked:
            if hit > 0:
                stats["revoked_insufficient"] += 1
            else:
                stats["revoked_full_miss"] += 1
        miss = requested - hit
        total = max(operation.stats_total_tokens, requested, 1)
        stats["l3_demand_requests"] += 1
        stats["l1l2_miss_tokens"] += requested
        stats["l3_miss_tokens"] += miss
        stats["l3_demand_total_tokens"] += total
        stats["l3_sum_rate_all"] += miss / total
        stats["l3_sum_rate_main_weighted"] += (miss / requested) * total

    def prefetch_outcome_stats_snapshot(self) -> dict:
        """Cumulative counters + instantaneous occupancy, in the schema
        log_prefetch_stats consumers expect."""
        cc = self.cache_controller
        cap = max(cc.prefetch_capacity_limit, 1)
        return {
            **self._prefetch_outcome_stats,
            "occupancy_ratio": cc.prefetch_tokens_occupied / cap,
        }

    def _prefetch_occupied_span(self, prefetch_key, host_indices) -> int:
        """Occupancy units held by a prefetch: cache mode reserves the
        requested span at enqueue; buffer mode grants at hit-alloc, sized
        to the allocation (0 while still querying / parked)."""
        if self.host_memory_mode == "buffer_only":
            return len(host_indices) if host_indices is not None else 0
        return len(prefetch_key)

    def revoke_pending_prefetch(self, req_id: str) -> None:
        info = self.ongoing_prefetch.get(req_id)
        if info is None:
            return
        (
            last_host_node_id,
            prefetch_key,
            _host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self._invalidate_absent_from_hit_query(operation)
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(req_id)
            self.buffer_pipeline.release_anchor_lock(req_id)
        cc = self.cache_controller
        cc.append_host_mem_release(
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
        )
        # Every revoke path runs before the bounce alloc, so buffer mode
        # holds no occupancy here; post-alloc aborts go through
        # release_aborted_request instead.
        assert _host_indices is None or self.host_memory_mode != "buffer_only"
        cc.prefetch_tokens_occupied = max(
            0,
            cc.prefetch_tokens_occupied
            - self._prefetch_occupied_span(prefetch_key, _host_indices),
        )

    def _drain_storage_control_queues_impl(
        self,
        n_storage_hit: Optional[int],
        n_ack_prefetch: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], n: Optional[int]) -> Iterator[T]:
            """If n is None, consume all items from the queue.
            Otherwise, consume n items from the queue.  Blocking if there are no enough n items.

            In TP, each rank consumes the a minimal number of items of all ranks.
            In PP, each rank consumes the exact number of items of PP0.  Refer to _pp_sync for more details.

            This prevents TP/PP divergence.
            """
            if n is None:
                while not q.empty():
                    item = q.get()
                    yield item
            else:
                for _ in range(n):
                    # Block when there are not enough elements.
                    # All TP/PP ranks must consume the same number of elements.
                    item = q.get()
                    yield item

        def _peek_queue(q: Queue[T]) -> T:
            """Return the FIFO head without consuming its retry trigger."""
            with q.mutex:
                if len(q.queue) == 0:
                    raise RuntimeError("A synchronized storage queue became empty.")
                return q.queue[0]

        def _commit_queue_head(q: Queue[T], expected: T) -> None:
            """Consume a previously inspected FIFO head after its state commits."""
            committed = q.get()
            if committed is not expected:
                raise RuntimeError("A storage queue changed while committing its head.")

        buffer_mode = self.host_memory_mode == "buffer_only"

        def _try_alloc_storage_hit(operation) -> bool:
            """Allocate the hit-sized bounce and launch the transfer.
            Returns False when staging pressure defers the allocation
            (buffer mode parks and retries; cache mode revokes)."""
            req_id = operation.request_id
            info = self.ongoing_prefetch.get(req_id)
            if info is None:
                return True  # aborted/cleaned; nothing to retry
            if operation.is_terminated():
                self.revoke_pending_prefetch(req_id)
                return True

            if buffer_mode and cc.prefetch_rate_limited():
                # Pool is load-saturated: hold the KNOWN hit until staged
                # prefetches ahead of us are consumed. The op stays in
                # ongoing_prefetch, so wait_complete keeps gating admission.
                return False
            if buffer_mode:
                # IO commit: pin before the bounce alloc so a cancel is a
                # plain revoke and a parked op keeps its pin; a fetch whose
                # splice base is gone is not worth its storage read.
                if self.buffer_pipeline.try_lock_anchor(req_id) == "anchor_lost":
                    self._prefetch_outcome_stats["declined_anchor_lost"] += 1
                    # Span still L3-resident: arm the paced retry to re-fetch
                    # from the shorter post-loss match.
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    return True
                if self.buffer_pipeline.staged_span_covered(
                    req_id, operation.storage_hit_count
                ):
                    # Live tree already covers the span: nothing left to
                    # splice, so skip the storage read.
                    self._prefetch_outcome_stats["declined_device_covered"] += 1
                    self.revoke_pending_prefetch(req_id)
                    return True
            alloc_len = operation.storage_hit_count
            host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                self.evict_host(alloc_len)
                host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None and not buffer_mode:
                # Memory-pressure fallback: a shorter page-aligned prefix.
                # (Cache mode only — buffer mode parks for the full hit.)
                available_size = cc.mem_pool_host.available_size()
                alloc_len = min(
                    operation.storage_hit_count,
                    available_size - (available_size % self.page_size),
                )
                if alloc_len >= self.prefetch_threshold:
                    host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                if buffer_mode:
                    return False
                self.revoke_pending_prefetch(req_id)
                return True

            operation.storage_hit_count = alloc_len
            operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
            operation.host_indices = host_indices
            self.ongoing_prefetch[req_id] = info._replace(host_indices=host_indices)
            if buffer_mode:
                cc.prefetch_tokens_occupied += alloc_len
            cc.prefetch_buffer.put(operation)
            return True

        def _drain_and_alloc_storage_hit():
            # Parked hits first (FIFO fairness with retries; buffer only).
            if buffer_mode:
                parked = self.buffer_pipeline.pending_hit_allocs
                while parked:
                    if not _try_alloc_storage_hit(parked[0]):
                        break
                    parked.popleft()
            for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
                req_id = operation.request_id
                info = self.ongoing_prefetch.get(req_id)
                if info is None:
                    # Request already aborted/cleaned up; still flush the
                    # query's absent-hash feedback.
                    self._invalidate_absent_from_hit_query(operation)
                    continue
                if operation.is_terminated():
                    # Controller-side miss termination (retryable) or an abort
                    # race (abort cleanup discards the marker).
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                if operation.storage_hit_count < self.prefetch_threshold:
                    # Below-threshold hit: classify + feed the L3 miss
                    # accounting, then revoke (not enough benefit).
                    self._account_prefetch_outcome(operation, revoked=True)
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                self._invalidate_absent_from_hit_query(operation)
                self._account_prefetch_outcome(operation, revoked=False)
                if not _try_alloc_storage_hit(operation):
                    # Counted once at first parking, not per retry tick.
                    self._prefetch_outcome_stats["declined_rate_limited"] += 1
                    self.buffer_pipeline.pending_hit_allocs.append(operation)

        def _drain_ack_prefetch():
            for ack in _drain_queue(cc.ack_prefetch_queue, n_ack_prefetch):
                operation = ack.operation
                if ack.completed_tokens is not None:
                    if operation.request_id in self.ongoing_prefetch:
                        assert operation.completed_tokens <= ack.completed_tokens
                        operation.completed_tokens = ack.completed_tokens
                if ack.pool_hits is not None:
                    if operation.request_id in self.ongoing_prefetch:
                        operation.pool_storage_result.update_extra_pool_hit_pages(
                            ack.pool_hits
                        )
                        operation.pool_transfers_done = True
                if ack.completed_req:
                    if operation.request_id in self.ongoing_prefetch:
                        # check_prefetch_progress() is not called for this rid yet.
                        # Let us insert the prefetch result into the radix tree.
                        self._handle_prefetch_result(operation)
                    cc.append_host_mem_release(
                        operation.host_indices[operation.completed_tokens :],
                        (
                            operation.pool_transfers
                            if not operation.pool_transfers_done
                            else None
                        ),
                    )

        def _drain_backup():
            drained = 0
            while n_backup is None or drained < n_backup:
                if n_backup is None and cc.ack_backup_queue.empty():
                    break
                operation = _peek_queue(cc.ack_backup_queue)
                entry = None
                if not buffer_mode:
                    entry = self.ongoing_backup.get(operation.id)
                    if entry is not None:
                        node_id, lock_params = entry
                        self.validate_host_lock_ref(node_id, lock_params)

                if buffer_mode:
                    # Storage write acked: free the staging.
                    self.buffer_pipeline.finish_storage_write_ack(operation.id)
                elif entry is not None:
                    node_id, lock_params = entry
                    self.dec_host_lock_ref(node_id, lock_params)
                    del self.ongoing_backup[operation.id]

                _commit_queue_head(cc.ack_backup_queue, operation)
                drained += 1
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    cc.mem_pool_host.free(
                        torch.cat(host_indices_list, dim=0), pool=pool_name
                    )
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_and_alloc_storage_hit()
        _drain_ack_prefetch()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_hit_queue.qsize(),
            cc.ack_prefetch_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        self._all_reduce(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_storage_hit, n_ack_prefetch, n_backup, n_release = qsize_list[:4]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[4:])
        }
        self._drain_storage_control_queues_impl(
            n_storage_hit=n_storage_hit,
            n_ack_prefetch=n_ack_prefetch,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def drain_storage_control_queues_local(self) -> None:
        """Drain the storage control queues without cross-rank synchronization.

        For the detach / shutdown path, where best-effort cleanup matters more than
        keeping the drained counts identical across ranks. The prefetch-hit queue is
        deliberately skipped: servicing it would allocate host pages for a prefetch
        that can no longer complete.
        """
        cc = self.cache_controller
        # The storage queues are created by the controller when the storage threads
        # start, so they are still None when a backend was never attached.
        if cc is None or cc.prefetch_hit_queue is None:
            return
        self._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=None,
            extra_release_counts={
                name: None for name in cc.extra_host_mem_release_queues
            },
            log_metrics=False,
        )

    # ---- HiCache: Storage backend lifecycle (delegated) ----

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return (
                False,
                "HiCache is not initialized; launch with "
                "--enable-hierarchical-cache to attach a storage backend.",
            )
        return self._storage_attachment.attach(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy=hicache_write_policy,
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return False, "HiCache storage backend is not initialized."
        return self._storage_attachment.detach()

    def shutdown(self) -> None:
        """Best-effort auto-detach of the storage backend on process shutdown."""
        if self._storage_attachment is not None:
            self._storage_attachment.shutdown()

    def clear_storage_backend(self) -> bool:
        if self._storage_attachment is None:
            return False
        ok = self._storage_attachment.clear()
        if ok:
            # L3 is empty now: every storage-presence belief is stale, and a
            # retained positive would skip that page's backup forever.
            self.storage_existence_cache.clear()
        return ok

    # ---- HiCache: Async Event Management ----

    def _count_ready_acks(self, ack_queue) -> int:
        ready_count = 0
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready_count += 1
        return ready_count

    def _sync_hicache_ready_counts(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[PoolName, ...]]:
        cc = self.cache_controller
        if cc is None:
            write_acks = 0
            load_acks = 0
            storage_queue_sizes = ()
            extra_pool_names = ()
        else:
            write_acks = self._count_ready_acks(cc.ack_write_queue)
            load_acks = self._count_ready_acks(cc.ack_load_queue)
            extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
            extra_pool_names = (
                tuple(extra_release_queues) if self.enable_storage else ()
            )
            storage_queue_sizes = (
                (
                    cc.prefetch_hit_queue.qsize(),
                    cc.ack_prefetch_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                    *(extra_release_queues[name].qsize() for name in extra_pool_names),
                )
                if self.enable_storage
                else ()
            )

        # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
        # equal iff reclaim victim order matched on every rank.
        digest = self.tree_core.write_back_duplicate_reclaim_digest
        ready_counts = torch.tensor(
            [
                write_acks,
                load_acks,
                *storage_queue_sizes,
                digest,
                -digest,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._all_reduce(ready_counts, torch.distributed.ReduceOp.MIN)

        count_values = list(map(int, ready_counts.tolist()))
        assert (
            count_values[-2] == -count_values[-1]
        ), "write_back duplicate-reclaim victims diverged across TP ranks"
        return (
            count_values[0],
            count_values[1],
            tuple(count_values[2:-2]),
            extra_pool_names,
        )

    def writing_check(
        self, write_back: bool = False, finish_count: Optional[int] = None
    ) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for ack in cc.ack_write_queue:
                    ack.finish_event.synchronize()
                    for ack_id in ack.node_ids:
                        if ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                    self._log_write_ack_metrics(ack)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_write_through can
            # diverge across ranks (e.g. write_backup returning 0 on a subset).
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_write_queue)
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        # Process completed acks
        while finish_count > 0:
            ack = cc.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id)
            self._log_write_ack_metrics(ack)
            finish_count -= 1

    def _log_write_ack_metrics(self, ack: HiCacheAck) -> None:
        """Record D->H backup volume and duration for a completed write ack."""
        if self.metrics_collector is None:
            return
        for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
            if num_tokens > 0:
                self.metrics_collector.increment_backup_num_tokens(
                    num_tokens=num_tokens, pool=pool
                )
        if ack.num_bytes > 0:
            self.metrics_collector.increment_backup_num_bytes(ack.num_bytes)
        if ack.timing_enabled:
            duration_ms = ack.start_event.elapsed_time(ack.finish_event)
            self.metrics_collector.observe_backup_duration(duration_ms / 1000.0)

    def loading_check(self, finish_count: Optional[int] = None) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_load_back can
            # diverge across ranks.
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_load_queue)
            # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
            # equal iff reclaim victim order matched on every rank.
            digest = self.tree_core.write_back_duplicate_reclaim_digest
            sync_tensor = torch.tensor(
                [finish_count, digest, -digest], dtype=torch.int64, device="cpu"
            )
            if injection_enabled() and len(self.ongoing_load_back) > 0:
                pause_point(
                    "reload_loading_collective_peer_wait",
                    self._pause_point_rank(),
                    {
                        "local_ready_count": int(finish_count),
                        "reclaim_digest": int(digest),
                        "ongoing_load_ids": sorted(self.ongoing_load_back),
                    },
                )
            self._all_reduce(sync_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = int(sync_tensor[0].item())
            assert (
                sync_tensor[1].item() == -sync_tensor[2].item()
            ), "write_back duplicate-reclaim victims diverged across TP ranks"

        while finish_count > 0:
            ack = cc.ack_load_queue[0]
            ack.finish_event.synchronize()
            if injection_enabled():
                pause_point(
                    "reload_after_transfer_before_finish",
                    self._pause_point_rank(),
                    {
                        "ack_ids": [int(ack_id) for ack_id in ack.node_ids],
                        "ongoing_load_ids": sorted(self.ongoing_load_back),
                        "finish_event_done": True,
                    },
                )

            seen_ack_ids: set[int] = set()
            for ack_id in ack.node_ids:
                if ack_id in seen_ack_ids:
                    raise RuntimeError(f"Load-back ACK repeats operation {ack_id}.")
                seen_ack_ids.add(ack_id)
                if (
                    self.buffer_pipeline is not None
                    and self.buffer_pipeline.owns_load_back_ack(ack_id)
                ):
                    continue
                load_back = self.ongoing_load_back.get(ack_id)
                if load_back is None:
                    raise RuntimeError(
                        f"Load-back ACK names unknown operation {ack_id}."
                    )
                self.validate_host_lock_ref(
                    load_back.node_id,
                    load_back.host_lock_params,
                )

            for ack_id in ack.node_ids:
                if (
                    self.buffer_pipeline is not None
                    and self.buffer_pipeline.try_finish_load_back(ack_id)
                ):
                    continue
                node, lock_params, host_lock_params = self.ongoing_load_back[ack_id]
                # Finalize the idempotent tree marks before consuming either
                # ownership receipt. A later release failure can still retry
                # from the queued ACK without rebuilding published state.
                self.tree_core.finish_load_back(node)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)
                del self.ongoing_load_back[ack_id]

            committed_ack = cc.ack_load_queue.pop(0)
            if committed_ack is not ack:
                raise RuntimeError("Load-back ACK queue changed during commit.")

            if self.metrics_collector is not None:
                for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
                    if num_tokens > 0:
                        self.metrics_collector.increment_load_back_num_tokens(
                            num_tokens=num_tokens, pool=pool
                        )
                if ack.num_bytes > 0:
                    self.metrics_collector.increment_load_back_num_bytes(ack.num_bytes)
                if ack.timing_enabled:
                    duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                    self.metrics_collector.observe_load_back_duration(
                        duration_ms / 1000.0
                    )
            finish_count -= 1

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return whether a unified component load-back event is complete.

        :param consumer_index: Layer-done counter slot returned by
            ``ready_to_load_host_cache``.
        :returns: Whether all rank-local component copies have completed.
        """

        if consumer_index < 0:
            return True
        if self.cache_controller is None:
            raise RuntimeError("HiCache load-back has no cache controller")

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        local_done = int(finish_event.query())
        done = torch.tensor(local_done, dtype=torch.int, device="cpu")
        self._all_reduce(done, torch.distributed.ReduceOp.MIN)
        if done.item() == 0:
            return False

        self.loading_check()
        return True

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> LoadBackResult:
        """Prepare rank-local cache components for host-to-device loading.

        :param params: Prefix match, request, and optional allocation quota.
        :returns: Explicit restoration payload and queue state.
        """
        if self.buffer_pipeline is not None:
            req = params.req
            assert req is not None
            swa_tokens = req.swa_host_hit_length
            device_indices, restored_node = self.buffer_pipeline.init_load_back(params)
            queued = len(device_indices) > 0
            return LoadBackResult(
                new_full_device_indices=device_indices,
                restored_node=restored_node,
                queued_any_component=queued,
                full_tokens=len(device_indices),
                swa_tokens=swa_tokens if queued else 0,
            )
        best_match_node_id = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node_id = req.last_node

        if (
            self.tree_core.is_full_device_evicted(best_match_node_id)
            or params.host_hit_length > 0
            or req.swa_host_hit_length > 0
            or req.mamba_host_hit_length > 0
        ):
            load_result = self.load_back(best_match_node_id, mem_quota, req=req)
            if load_result is not None:
                logger.debug(
                    "init_load_back queued full=%d swa=%d tokens for node %d",
                    load_result.full_tokens,
                    load_result.swa_tokens,
                    best_match_node_id,
                )
                return LoadBackResult(
                    new_full_device_indices=load_result.new_full_device_indices,
                    restored_node=best_match_node_id,
                    queued_any_component=True,
                    full_tokens=load_result.full_tokens,
                    swa_tokens=load_result.swa_tokens,
                    cache_protected_len=(
                        len(params.req.prefix_indices)
                        + len(load_result.new_full_device_indices)
                    ),
                )

        return LoadBackResult(
            new_full_device_indices=self.tree_core.empty_match_result.device_indices,
            restored_node=last_best_match_device_node_id,
            queued_any_component=False,
            full_tokens=0,
            swa_tokens=0,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()

        if self.pp_size != 1:
            finish_counts = torch.zeros(2, dtype=torch.int, device="cpu")
            if self.pp_rank == 0 and self.cache_controller is not None:
                finish_counts[0] = self._count_ready_acks(
                    self.cache_controller.ack_write_queue
                )
                finish_counts[1] = self._count_ready_acks(
                    self.cache_controller.ack_load_queue
                )
            self._all_reduce(finish_counts, torch.distributed.ReduceOp.MIN)
            write_finish_count, load_finish_count = map(int, finish_counts.tolist())
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)
            if self.enable_storage:
                self.drain_storage_control_queues()
        else:
            (
                write_finish_count,
                load_finish_count,
                storage_queue_sizes,
                extra_pool_names,
            ) = self._sync_hicache_ready_counts()
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)

            if self.enable_storage and storage_queue_sizes:
                n_storage_hit, n_ack_prefetch, n_backup, n_release = (
                    storage_queue_sizes[:4]
                )
                extra_release_counts = {
                    pool_name: count
                    for pool_name, count in zip(
                        extra_pool_names,
                        storage_queue_sizes[4:],
                    )
                }
                self._drain_storage_control_queues_impl(
                    n_storage_hit=n_storage_hit,
                    n_ack_prefetch=n_ack_prefetch,
                    n_backup=n_backup,
                    n_release=n_release,
                    extra_release_counts=extra_release_counts,
                    log_metrics=True,
                )
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.flush_pending_writes()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            storage_metrics = self.cache_controller.storage_backend.get_stats()
            if storage_metrics is None:
                # Backends without native stats (e.g. file) still carry the
                # controller-side prefetch outcome counters.
                storage_metrics = StorageMetrics()
            storage_metrics.prefetch_stats = self.prefetch_outcome_stats_snapshot()
            self.storage_metrics_collector.log_storage_metrics(storage_metrics)

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is None:
            return 0
        consumer_index = self.cache_controller.start_loading()
        if (
            injection_enabled()
            and consumer_index >= 0
            and not self.cache_controller.layer_done_counter.events[
                consumer_index
            ].finish_event.query()
        ):
            pause_point(
                "reload_h2d_inflight",
                self._pause_point_rank(),
                {
                    "consumer_index": int(consumer_index),
                    "ongoing_load_ids": sorted(self.ongoing_load_back),
                    "finish_event_done": False,
                },
            )
        return consumer_index

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        return self._sliding_window_size

    def swa_reprefill_tail_tokens(self) -> int:
        """
        Only unified_kv + HiCache needs this: SWA lives in a per-request ring
        (state_slot/pos), not content-stable and never offloaded to host, so a
        reused prefix's trailing sliding window would read another request's
        stale ring slots. Re-prefilling that window rewrites this request's ring
        (what plain radix reuse does via its SWA match gate). 0 for every other
        layout.
        """
        swa = self.components.get(ComponentType.SWA)
        unified_compress_only_hicache = (
            self.cache_controller is not None
            and swa is not None
            and not self.tree_core.has_swa_host_pool
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def swa_retain_floor(self, req) -> int | None:
        if not self.is_mamba_enabled or self._sliding_window_size is None:
            return None
        checkpoint = req.mamba_last_track_seqlen
        if checkpoint is None:
            return None
        return checkpoint - self._sliding_window_size

    def supports_swa(self) -> bool:
        return self.is_swa_enabled

    def supports_mamba(self) -> bool:
        return self.is_mamba_enabled

    # ---- Session radix cache API (delegates to composed UnifiedSessionRefTracker) ----

    def open_radix_session(self, session_id: str) -> Optional[int]:
        return self.session_refs.open_radix_session(session_id)

    def ensure_session_generation(self, session_id: str) -> int:
        return self.session_refs.ensure_session_generation(session_id)

    def release_radix_session(self, session_id: str) -> int:
        return self.session_refs.release_radix_session(session_id)

    def clear_radix_session_refs(self, session_id: str) -> int:
        """Release tagged cache coverage while keeping the generation open.

        :param session_id: Session identifier whose coverage must be released.
        :returns: Number of component frontier tags removed.
        """
        return self.session_refs.clear_session_refs(session_id)

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def supports_streaming_session_demotion(self) -> bool:
        """Return whether this cache can publish durable host-resident sessions.

        :returns: Whether transactional host demotion is available.
        """
        return (
            not self.disable
            and self.enable_session_radix_cache
            and self.host_memory_mode == "cache"
            and self.buffer_pipeline is None
            and self.supports_retraction_backup()
        )

    def is_streaming_session_demoted(self, session_id: str) -> bool:
        """Return whether one streaming session owns a host frontier.

        :param session_id: Session identifier to inspect.
        :returns: Whether the session is host-resident.
        """
        return self.session.is_demoted(session_id)

    def streaming_session_demoted_namespace(
        self, session_id: str
    ) -> tuple[str | None, str | None] | None:
        """Return the cache namespace a demoted session was seeded under.

        :param session_id: Session identifier to inspect.
        :returns: The seeded ``(extra_key, cache_salt)``, or ``None`` when the
            session is not host-resident.
        """
        return self.session.demoted_namespace(session_id)

    def prepare_streaming_session_demotion(
        self,
        session_id: str,
        token_ids: Sequence[int],
        extra_key: str | None,
        cache_salt: str | None,
        priority: int,
    ) -> int | None:
        """Stage a session's exact KV privately in the host pools.

        The slot's existing tree prefix retains ordinary radix identity. Every
        detached suffix page is published on a session-private path, and a
        trailing partial page owns one physical host page with its exact length.

        :param session_id: Session identifier to stage.
        :param token_ids: Complete committed token lineage.
        :param extra_key: Radix cache classification key.
        :param cache_salt: Radix cache namespace salt.
        :param priority: Eviction priority inherited from the session.
        :returns: Exact staged token count, or ``None`` on rejection.
        """
        if not self.supports_streaming_session_demotion():
            return None
        if session_id in self._pending_streaming_session_demotions:
            raise RuntimeError(f"Session {session_id} already has a staged demotion.")
        if self.session.is_demoted(session_id):
            raise RuntimeError(f"Session {session_id} is already host-resident.")

        slot = self.session.slots.get(session_id)
        if slot is None or not slot.is_holding_kv:
            return None

        key = RadixKey(
            array("q", token_ids),
            extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        )
        key = key[: min(len(key), slot.kv_committed_len)]
        exact_len = len(key)
        aligned_len = exact_len // self.page_size * self.page_size
        physical_len = ceil_align(exact_len, self.page_size)
        if exact_len == 0 or slot.cache_protected_len > exact_len:
            return None
        tree_prefix_len = slot.tree_protected_len
        if (
            tree_prefix_len < 0
            or tree_prefix_len % self.page_size != 0
            or tree_prefix_len > aligned_len
            or tree_prefix_len > slot.cache_protected_len
        ):
            raise AssertionError(
                "Streaming-session tree frontier is inconsistent with demotion: "
                f"{tree_prefix_len=} {slot.cache_protected_len=} {aligned_len=}"
            )
        tree_prefix_node = (
            self.tree_core.root_node.id if slot.last_node is None else slot.last_node
        )
        tree_nodes = self._streaming_session_path(
            tree_prefix_node,
            tree_prefix_len,
        )
        offset = 0
        for node in tree_nodes:
            node_len = len(node.key)
            if node.key.match(key[offset : offset + node_len]) != node_len:
                raise AssertionError(
                    f"Session {session_id} lineage does not match its tree frontier."
                )
            offset += node_len

        logical_device_indices = self.req_to_token_pool.req_to_token[
            slot.req_pool_idx, :exact_len
        ].to(dtype=torch.int64, copy=True)
        device_indices = self._pad_retraction_indices(
            logical_device_indices, self.page_size
        )
        assert len(device_indices) == physical_len
        extra_transfers = self._device_transfers_from_indices(
            device_indices,
            exact_len,
        )
        swa_window_start = 0
        if self.supports_swa():
            assert self.sliding_window_size is not None
            swa_window_start = max(0, exact_len - self.sliding_window_size)
            swa_window_start = swa_window_start // self.page_size * self.page_size

        backup = self._stage_retraction_backup(device_indices, extra_transfers)
        if backup is None:
            return None
        self._pending_streaming_session_demotions[session_id] = (
            _PendingStreamingSessionDemotion(
                key=key,
                aligned_len=aligned_len,
                device_indices=device_indices,
                backup=backup,
                tree_prefix_node=tree_prefix_node,
                tree_prefix_len=tree_prefix_len,
                swa_window_start=swa_window_start,
                priority=priority,
            )
        )
        return exact_len

    def discard_streaming_session_demotion(self, session_id: str) -> None:
        """Discard a private stage after any tensor-parallel rank votes no.

        :param session_id: Session identifier whose stage must be discarded.
        """
        pending = self._pending_streaming_session_demotions.pop(session_id, None)
        if pending is not None:
            self.retraction_discard(pending.backup)

    def commit_streaming_session_demotion(self, session_id: str) -> int:
        """Publish a unanimously staged session and retire its device slot.

        :param session_id: Session identifier whose stage must be committed.
        :returns: Exact host-backed token count.
        """
        pending = self._pending_streaming_session_demotions[session_id]
        last_node = pending.tree_prefix_node
        host_lock_params: DecLockRefParams | None = None
        host_path_transaction: _StreamingSessionHostPathTransaction | None = None
        committed = False
        try:
            if pending.swa_window_start < pending.tree_prefix_len:
                self._split_streaming_session_path_at(
                    pending.tree_prefix_node,
                    pending.tree_prefix_len,
                    pending.swa_window_start,
                )
            private_boundaries = [pending.tree_prefix_len]
            if pending.tree_prefix_len < pending.swa_window_start < pending.aligned_len:
                private_boundaries.append(pending.swa_window_start)
            private_boundaries.append(pending.aligned_len)
            for start, end in pairwise(private_boundaries):
                if start == end:
                    continue
                last_node = self.tree_core.add_streaming_session_private_node(
                    last_node,
                    session_id,
                    pending.key[start:end],
                    pending.priority,
                    is_fringe=False,
                )
            if pending.aligned_len < len(pending.key):
                fringe_key = pending.key[pending.aligned_len :]
                last_node = self.tree_core.add_streaming_session_private_node(
                    last_node,
                    session_id,
                    fringe_key,
                    pending.priority,
                    is_fringe=True,
                )

            host_path_transaction = _StreamingSessionHostPathTransaction(
                ledger=_HostStageLedger.from_backup(pending.backup)
            )
            self._commit_streaming_session_host_path(
                last_node,
                pending.backup,
                host_path_transaction,
                logical_len=len(pending.key),
                swa_window_start=pending.swa_window_start,
            )
            host_lock_params = self.inc_host_lock_ref(last_node).to_dec_params()
            self.session_refs.register_streaming_session_frontier(
                session_id,
                last_node,
            )
            self.session.transition_to_demoted(
                session_id,
                last_node,
                len(pending.key),
                pending.tree_prefix_len,
                pending.swa_window_start,
                host_lock_params,
                extra_key=pending.key.extra_key,
                cache_salt=pending.key.cache_salt,
            )
            self._demote_streaming_session_tree_path(pending.tree_prefix_node)
            host_path_transaction.ledger.release_stage_owned(self.host_pool_group)
            committed = True
        finally:
            published = committed or self.session.is_demoted(session_id)
            if published:
                self._pending_streaming_session_demotions.pop(session_id, None)
            elif self.session.demotion_retirement_started(session_id):
                # Source retirement is the final local phase. A failure inside it
                # is indeterminate: rollback could resurrect pages an allocator
                # already released, so the stage stays put and the exception
                # ends this scheduler through the post-commit fence.
                pass
            else:
                self.clear_radix_session_refs(session_id)
                if host_lock_params is not None:
                    self.dec_host_lock_ref(last_node, host_lock_params)
                if host_path_transaction is not None:
                    self._rollback_streaming_session_host_path(
                        host_path_transaction
                    )
                else:
                    self.retraction_discard(pending.backup)
                if self.tree_core.is_session_private(last_node):
                    self.retire_streaming_session_private_path(session_id, last_node)
                self._pending_streaming_session_demotions.pop(session_id, None)

        return len(pending.key)

    def _streaming_session_path(
        self,
        last_node: NodeId,
        expected_len: int,
    ) -> list[UnifiedTreeNode]:
        """Return the exact root-to-frontier node path for a staged prefix.

        :param last_node: Radix frontier anchoring the staged prefix.
        :param expected_len: Required logical length of the complete path.
        :returns: Non-root nodes in root-to-frontier order.
        :raises AssertionError: If the frontier does not span the staged prefix.
        """
        nodes: list[UnifiedTreeNode] = []
        node = self.tree_core.node_by_id(last_node)
        while node is not self.tree_core.root_node:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        path_len = sum(len(node.key) for node in nodes)
        if path_len != expected_len:
            raise AssertionError(
                "Session demotion frontier length mismatch: "
                f"{path_len=} {expected_len=}"
            )
        return nodes

    def _split_streaming_session_path_at(
        self,
        last_node: NodeId,
        path_len: int,
        boundary: int,
    ) -> None:
        """Ensure one root-relative boundary falls between ordinary nodes.

        A split is a representation-only change and deliberately survives a
        later transaction rollback. Performing it before host ownership moves
        keeps every staged slice bound to one stable node.

        :param last_node: Ordinary radix frontier to inspect.
        :param path_len: Exact logical length represented by the frontier.
        :param boundary: Root-relative token boundary to materialize.
        """
        if boundary <= 0 or boundary >= path_len:
            return
        offset = 0
        for node in self._streaming_session_path(last_node, path_len):
            node_end = offset + len(node.key)
            if boundary == offset or boundary == node_end:
                return
            if offset < boundary < node_end:
                _, action = self.tree_core._split_node(
                    node.key,
                    node,
                    boundary - offset,
                )
                if action is not None:
                    self._apply_cache_action(action)
                return
            offset = node_end
        raise AssertionError(
            f"Session path boundary {boundary} lies outside a {path_len}-token path."
        )

    def _commit_streaming_session_host_path(
        self,
        last_node: NodeId,
        backup: RetractionBackup,
        transaction: _StreamingSessionHostPathTransaction,
        *,
        logical_len: int | None = None,
        swa_window_start: int = 0,
    ) -> _StreamingSessionHostPathTransaction:
        """Attach staged full and SWA slots to the integrated radix path.

        :param last_node: Radix frontier anchoring the staged prefix.
        :param backup: Private host stage to publish.
        :param transaction: Caller-owned stage-consumption and rollback ledger.
        :param logical_len: Exact token count when a physical fringe is present.
        :param swa_window_start: First logical token represented by staged SWA.
        :returns: Exact ownership ledger for commit or rollback.
        """
        if backup.host_indices is None:
            raise AssertionError("Session demotion has no staged Full host indices.")
        expected_len = len(backup.host_indices) if logical_len is None else logical_len
        nodes = self._streaming_session_path(last_node, expected_len)
        for node in nodes:
            logical_node_len = len(node.key)
            physical_node_len = (
                logical_node_len
                if node.private_physical_length is None
                else node.private_physical_length
            )
            stage_slice = transaction.ledger.take(
                PoolName.KV, physical_node_len
            )
            full_data = node.component_data[ComponentType.FULL]
            if full_data.host_value is None:
                self.tree_core.commit_backup(node.id, stage_slice.indices, {})
                attached = node.component_data[ComponentType.FULL].host_value
                if attached is None or not torch.equal(attached, stage_slice.indices):
                    raise AssertionError(
                        f"Full host publication on node {node.id} lost its stage."
                    )
                stage_slice.tree_owned = True
                transaction.attachments.append(
                    _HostPathAttachment(
                        node_id=node.id,
                        component_type=ComponentType.FULL,
                        stage_slice=stage_slice,
                    )
                )
                self.tree_core.finish_synchronous_backup(node.id)

        independent = [
            transfer
            for transfer in backup.pool_transfers or []
            if transfer.indices_from_pool is None
        ]
        unsupported = [
            transfer.name for transfer in independent if transfer.name != PoolName.SWA
        ]
        if len(unsupported) > 0:
            raise AssertionError(
                f"Unsupported independently allocated demotion pools: {unsupported}"
            )
        swa_transfers = [
            transfer for transfer in independent if transfer.name == PoolName.SWA
        ]
        if len(swa_transfers) == 0:
            transaction.ledger.assert_fully_consumed()
            return transaction
        if len(swa_transfers) != 1 or swa_transfers[0].host_indices is None:
            raise AssertionError("Session demotion requires one resolved SWA transfer.")

        selected_nodes: list[UnifiedTreeNode] = []
        offset = 0
        for node in nodes:
            node_end = offset + len(node.key)
            if node_end > swa_window_start:
                if offset < swa_window_start:
                    raise AssertionError(
                        "SWA demotion boundary cuts through an unsplit tree node."
                    )
                selected_nodes.append(node)
            offset = node_end

        for node in selected_nodes:
            logical_node_len = len(node.key)
            physical_node_len = (
                logical_node_len
                if node.private_physical_length is None
                else node.private_physical_length
            )
            stage_slice = transaction.ledger.take(
                PoolName.SWA, physical_node_len
            )
            swa_data = node.component_data[ComponentType.SWA]
            if swa_data.host_value is None:
                self.tree_core.commit_backup(
                    node.id,
                    backup.host_indices[:0],
                    {
                        ComponentType.SWA: [
                            replace(
                                swa_transfers[0],
                                host_indices=stage_slice.indices,
                            )
                        ]
                    },
                )
                attached = node.component_data[ComponentType.SWA].host_value
                if attached is None or not torch.equal(attached, stage_slice.indices):
                    raise AssertionError(
                        f"SWA host publication on node {node.id} lost its stage."
                    )
                stage_slice.tree_owned = True
                transaction.attachments.append(
                    _HostPathAttachment(
                        node_id=node.id,
                        component_type=ComponentType.SWA,
                        stage_slice=stage_slice,
                    )
                )

        transaction.ledger.assert_fully_consumed()
        return transaction

    def _rollback_streaming_session_host_path(
        self,
        transaction: _StreamingSessionHostPathTransaction,
    ) -> None:
        """Detach every staged attachment and release the remaining stage."""
        for attachment in reversed(transaction.attachments):
            node = self.tree_core.node_by_id(attachment.node_id)
            component_data = node.component_data[attachment.component_type]
            attached = component_data.host_value
            if attached is None or not torch.equal(
                attached, attachment.stage_slice.indices
            ):
                raise AssertionError(
                    f"Rollback lost {attachment.component_type} host ownership on "
                    f"node {attachment.node_id}."
                )
            component_data.host_value = None
            attachment.stage_slice.tree_owned = False
            if attachment.component_type == ComponentType.FULL:
                self.tree_core._update_duplicate_tracking(node)
                self.tree_core._update_evictable_leaf_sets(node)
                if node.parent is not None:
                    self.tree_core._update_evictable_leaf_sets(node.parent)
            else:
                self.tree_core._reconcile_auxiliary_host_lru(
                    node, attachment.component_type
                )
        transaction.ledger.release_stage_owned(self.host_pool_group)

    def _demote_streaming_session_tree_path(self, last_node: NodeId) -> None:
        """Demote the unlocked ordinary prefix while preserving shared branches.

        :param last_node: Ordinary host-backed frontier to demote toward the root.
        """
        node = self.tree_core.node_by_id(last_node)
        while node is not self.tree_core.root_node:
            parent = node.parent
            if node.evicted:
                node = parent
                continue
            if not self.tree_core._is_device_leaf(node):
                break
            if not node.backuped:
                raise AssertionError(
                    f"Session demotion node {node.id} has no published host copy."
                )
            result = self.tree_core.demote(node.id)
            self._free_values(result.device_frees, result.host_frees)
            node = parent

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def streaming_session_private_parent(self, node: NodeId) -> NodeId | None:
        """Return the ordinary radix parent of a session-private suffix.

        :param node: Exact host frontier, or an ordinary aligned frontier.
        :returns: The ordinary parent, or ``None`` without a private suffix.
        """
        if not self.tree_core.is_session_private(node):
            return None
        return self.tree_core.streaming_session_private_parent(node)

    def retire_streaming_session_private_path(
        self, session_id: str, node: NodeId
    ) -> None:
        """Detach a private exact suffix after session ownership ends.

        :param session_id: Session identifier that owns the private suffix.
        :param node: Exact host frontier, or an ordinary radix frontier.
        """
        result = self.tree_core.retire_streaming_session_private_path(session_id, node)
        self._free_values(result.device_frees, result.host_frees)

    def validate_streaming_session_private_path_detach(
        self,
        session_id: str,
        node: NodeId,
        owner_params: DecLockRefParams,
        *,
        allow_device_locks: bool,
    ) -> None:
        """Validate private-path detachment before releasing its owner.

        :param session_id: Session that owns the private suffix.
        :param node: Exact private frontier.
        :param owner_params: Demoted session's host-lock receipt.
        :param allow_device_locks: Whether request-owned locks remain temporarily.
        """
        self.tree_core.validate_streaming_session_private_path_detach(
            session_id,
            node,
            owner_params,
            allow_device_locks=allow_device_locks,
        )

    def adopt_streaming_session_private_path(
        self, session_id: str, node: NodeId
    ) -> int:
        """Transfer a restored private suffix to detached-slot ownership.

        :param session_id: Session identifier that owns the private suffix.
        :param node: Exact restored frontier.
        :returns: Logical private-suffix length, or zero without one.
        """
        if not self.tree_core.is_session_private(node):
            return 0
        private_len = self.tree_core.streaming_session_private_length(node)
        result = self.tree_core.adopt_streaming_session_private_path(session_id, node)
        # The detached request slot inherits every restored device allocation.
        # Only the redundant host copies leave ownership during adoption.
        result.device_frees.clear()
        self._free_values(result.device_frees, result.host_frees)
        return private_len

    def truncate_session(self, session_id: str, target: int) -> None:
        self.session.truncate_session(session_id, target)

    def commit_session(self, session_id: str, floor: int) -> None:
        self.session.commit_session(session_id, floor)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def _pause_point_rank(self) -> int:
        """Return this cache's tensor-parallel rank for lifecycle pause points.

        :returns: The rank within the TP cache group, or 0 without one.
        """
        if self.tp_group is None or not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_rank(group=self.tp_group)

    def pending_streaming_session_demotion_ids(self) -> list[str]:
        """Return the sessions with a staged but unpublished demotion.

        :returns: Session identifiers in sorted order.
        """
        return sorted(self._pending_streaming_session_demotions)

    @staticmethod
    def _free_set_digest(indices: torch.Tensor) -> str:
        """Digest one allocator's free set independently of its order.

        :param indices: Free slot or page identifiers.
        :returns: Hex digest of the sorted identifiers.
        """
        ordered, _ = torch.sort(indices.detach().to("cpu", torch.int64))
        return hashlib.sha256(ordered.numpy().tobytes()).hexdigest()

    def _device_pool_evidence(
        self,
        *,
        allocator: BaseTokenToKVPoolAllocator,
        total_tokens: int,
        evictable_tokens: int,
        protected_tokens: int,
        session_held_tokens: int,
    ) -> dict[str, Any]:
        """Describe one device pool's complete ownership equation in pages.

        :param allocator: Concrete paged allocator owning the pool.
        :param total_tokens: Pool capacity in tokens.
        :param evictable_tokens: Tree-owned evictable tokens.
        :param protected_tokens: Tree-owned protected tokens.
        :param session_held_tokens: Tokens held by detached session slots.
        :returns: Page-denominated pool record.
        """
        if allocator.free_pages is None or allocator.release_pages is None:
            raise AssertionError("Idle evidence requires a paged device allocator.")
        page = self.page_size
        free = torch.cat((allocator.free_pages, allocator.release_pages))
        total = total_tokens // page
        available = allocator.available_size() // page
        return {
            "unit": "pages",
            "page_size_tokens": page,
            "total_size": total,
            "available_size": available,
            "held_size": total - available,
            "evictable_size": evictable_tokens // page,
            "protected_size": protected_tokens // page,
            "session_held_size": session_held_tokens // page,
            "transient_held_size": 0,
            "free_set_count": int(free.numel()),
            "free_set_digest": self._free_set_digest(free),
        }

    def _host_pool_evidence(
        self,
        pool_name: PoolName,
        component_type: ComponentType,
        session_held_pages: int,
    ) -> dict[str, Any]:
        """Describe one host pool's complete ownership equation in pages.

        :param pool_name: Host pool to describe.
        :param component_type: Tree component the pool backs.
        :param session_held_pages: Physical pages locked by demoted sessions.
        :returns: Page-denominated pool record.
        """
        host_pool = self.host_pool_group.get_entry(pool_name).host_pool
        page = host_pool.page_size
        free = torch.cat((host_pool.free_slots, *host_pool.release_slots))
        # Host free slots are token indices; the evidence is page-denominated,
        # so count and digest the free pages by their first slot.
        free_pages = free[free % page == 0] // page
        if int(free_pages.numel()) * page != int(free.numel()):
            raise AssertionError("Host free slots are not page-granular.")
        total = host_pool.logical_size // page
        available = host_pool.available_size() // page
        held = total - available
        evictable = 0
        for node in self.tree_core.host_lru_lists[component_type].cache.values():
            host_value = node.component_data[component_type].host_value
            if host_value is not None:
                evictable += ceil_align(len(host_value), page) // page
        return {
            "unit": "pages",
            "page_size_tokens": page,
            "total_size": total,
            "available_size": available,
            "held_size": held,
            "evictable_size": evictable,
            "protected_size": held - session_held_pages - evictable,
            "session_held_size": session_held_pages,
            "transient_held_size": 0,
            "free_set_count": int(free_pages.numel()),
            "free_set_digest": self._free_set_digest(free_pages),
        }

    def _unique_host_locked_pages(self) -> dict[str, int]:
        """Count the physical host pages locked by demoted sessions, once each.

        :returns: Page counts for the Full and SWA components.
        """
        page = self.page_size
        locked: dict[ComponentType, set[int]] = {
            ComponentType.FULL: set(),
            ComponentType.SWA: set(),
        }
        for state in self.session.demoted.values():
            node = self.tree_core.node_by_id(state.last_node)
            while node is not self.tree_core.root_node:
                for component_type, pages in locked.items():
                    if component_type not in self.components:
                        continue
                    component_data = node.component_data[component_type]
                    if component_data.host_lock_ref > 0 and component_data.host_value is not None:
                        pages.update(
                            int(index) // page
                            for index in component_data.host_value.tolist()
                        )
                node = node.parent
        return {
            "full": len(locked[ComponentType.FULL]),
            "swa": len(locked[ComponentType.SWA]),
        }

    def _session_ownership_records(
        self, inventory: "Sequence[StreamingSessionInventory]"
    ) -> list[dict[str, Any]]:
        """Describe each open session's exact stable ownership on this rank.

        Sessions without committed KV are omitted: they own nothing to verify.

        :param inventory: The session controller's current inventory.
        :returns: One record per session holding device or host KV.
        """
        page = self.page_size
        pending = set(self._pending_streaming_session_demotions)
        records: list[dict[str, Any]] = []
        for entry in inventory:
            if entry.tip <= 0:
                continue
            session_id = entry.session_id
            slot = self.session.slots.get(session_id)
            demoted = self.session.demoted.get(session_id)
            exclusive = {"full": 0, "swa": 0}
            host_locked = {"full": 0, "swa": 0}
            ongoing: list[str] = []
            if demoted is not None:
                state = "host"
                slot_id = None
                full, swa = self.streaming_session_protected_residency(demoted.last_node)
                host_locked = {"full": full.host_backed_pages, "swa": swa.host_backed_pages}
                ongoing = [
                    str(ack_id)
                    for ack_id, load_back in sorted(self.ongoing_load_back.items())
                    if load_back.node_id == demoted.last_node
                ]
            elif slot is not None and slot.is_holding_kv:
                state = "device"
                slot_id = slot.req_pool_idx
                allocated = ceil_align(slot.kv.kv_allocated_len, page)
                exclusive["full"] = max(0, allocated - slot.tree_protected_len) // page
                if self.supports_swa():
                    swa_start = max(slot.tree_protected_len, slot.kv.swa_evicted_seqlen)
                    exclusive["swa"] = max(0, allocated - swa_start) // page
            else:
                continue
            generation = self.session_refs.radix_generation(session_id)
            records.append(
                {
                    "session_id": session_id,
                    "tip": entry.tip,
                    "floor": entry.floor,
                    "lineage_digest": entry.lineage_digest,
                    "lineage_generation": entry.lineage_generation,
                    "slot_id": slot_id,
                    "radix_tag": f"{session_id}:g{generation}",
                    "state": state,
                    "kv_residency": {
                        "full": {
                            "device_pages": entry.full.device_pages,
                            "host_backed_pages": entry.full.host_backed_pages,
                        },
                        "swa": {
                            "device_pages": entry.swa.device_pages,
                            "host_backed_pages": entry.swa.host_backed_pages,
                        },
                    },
                    "exclusive_device_pages": exclusive,
                    "host_locked_pages": host_locked,
                    "pending_demotion_ids": [session_id] if session_id in pending else [],
                    "ongoing_load_back_ids": ongoing,
                }
            )
        return records

    def idle_ownership_evidence(
        self,
        *,
        inventory: "Sequence[StreamingSessionInventory]",
        full_total_tokens: int,
        swa_total_tokens: int,
        session_held_full_tokens: int,
        session_held_swa_tokens: int,
        session_held_req_count: int,
    ) -> dict[str, Any]:
        """Assemble this rank's complete idle ownership evidence.

        :param inventory: The session controller's current inventory.
        :param full_total_tokens: Full-attention device pool capacity in tokens.
        :param swa_total_tokens: Sliding-window device pool capacity in tokens.
        :param session_held_full_tokens: Full tokens held by detached slots.
        :param session_held_swa_tokens: SWA tokens held by detached slots.
        :param session_held_req_count: Request rows held by detached slots.
        :returns: Pool, tree, and session ownership evidence.
        """
        allocator = self.token_to_kv_pool_allocator
        if not isinstance(allocator, SWATokenToKVPoolAllocator):
            raise AssertionError("Idle evidence requires the hybrid SWA allocator.")
        unique_host_locked = self._unique_host_locked_pages()
        request_total = self.req_to_token_pool.size
        request_available = self.req_to_token_pool.available_size()
        request_free = torch.tensor(
            sorted(self.req_to_token_pool.free_slots), dtype=torch.int64
        )
        return {
            "device_pools": {
                "full": self._device_pool_evidence(
                    allocator=allocator.full_attn_allocator,
                    total_tokens=full_total_tokens,
                    evictable_tokens=self.tree_core.full_evictable_size(),
                    protected_tokens=self.tree_core.full_protected_size(),
                    session_held_tokens=session_held_full_tokens,
                ),
                "swa": self._device_pool_evidence(
                    allocator=allocator.swa_attn_allocator,
                    total_tokens=swa_total_tokens,
                    evictable_tokens=self.tree_core.swa_evictable_size(),
                    protected_tokens=self.tree_core.swa_protected_size(),
                    session_held_tokens=session_held_swa_tokens,
                ),
            },
            "host_pools": {
                "full": self._host_pool_evidence(
                    PoolName.KV, ComponentType.FULL, unique_host_locked["full"]
                ),
                "swa": self._host_pool_evidence(
                    PoolName.SWA, ComponentType.SWA, unique_host_locked["swa"]
                ),
            },
            "request_pool": {
                "unit": "slots",
                "total_size": request_total,
                "available_size": request_available,
                "held_size": request_total - request_available,
                "session_held_size": session_held_req_count,
                "transient_held_size": request_total
                - request_available
                - session_held_req_count,
                "free_set_count": int(request_free.numel()),
                "free_set_digest": self._free_set_digest(request_free),
            },
            "unique_host_locked_pages": unique_host_locked,
            "sessions": self._session_ownership_records(inventory),
            "pending_streaming_session_demotions": self.pending_streaming_session_demotion_ids(),
            "ongoing_load_back": [
                {"ack_id": int(ack_id), "node_id": int(load_back.node_id)}
                for ack_id, load_back in sorted(self.ongoing_load_back.items())
            ],
        }

    def streaming_session_cache_snapshot(
        self, session_id: str
    ) -> StreamingSessionCacheSnapshot:
        """Return durable cache ownership for one streaming session.

        :param session_id: Session identifier to inspect.
        :returns: The composed streaming-session cache snapshot.
        """
        return self.session.streaming_session_cache_snapshot(session_id)

    def streaming_session_protected_residency(
        self, node: NodeId | None
    ) -> tuple[KVComponentResidency, KVComponentResidency]:
        """Count device and host pages on a session's locked radix path.

        :param node: Deepest locked tree node, or ``None`` for an empty path.
        :returns: Full and SWA physical page residency.
        """
        if node is None:
            return KVComponentResidency(), KVComponentResidency()

        def component_residency(component_type: ComponentType) -> KVComponentResidency:
            if component_type not in self.components:
                return KVComponentResidency()

            device_pages = 0
            host_backed_pages = 0
            current = self.tree_core.node_by_id(node)
            while current is not self.tree_core.root_node:
                component_data = current.component_data[component_type]
                if component_data.value is not None:
                    device_pages += (
                        ceil_align(len(component_data.value), self.page_size)
                        // self.page_size
                    )
                if component_data.host_value is not None:
                    host_backed_pages += (
                        ceil_align(len(component_data.host_value), self.page_size)
                        // self.page_size
                    )
                current = current.parent
            return KVComponentResidency(
                device_pages=device_pages,
                host_backed_pages=host_backed_pages,
            )

        return (
            component_residency(ComponentType.FULL),
            component_residency(ComponentType.SWA),
        )

    def evictable_size(self) -> int:
        return self.tree_core.evictable_size()

    def protected_size(self) -> int:
        return self.tree_core.protected_size()

    def full_evictable_size(self) -> int:
        return self.tree_core.full_evictable_size()

    def full_protected_size(self) -> int:
        return self.tree_core.full_protected_size()

    def swa_evictable_size(self) -> int:
        return self.tree_core.swa_evictable_size()

    def mamba_evictable_size(self) -> int:
        return self.tree_core.mamba_evictable_size()

    def swa_protected_size(self) -> int:
        return self.tree_core.swa_protected_size()

    def mamba_protected_size(self) -> int:
        return self.tree_core.mamba_protected_size()

    def total_size(self) -> tuple[int, int]:
        return self.tree_core.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_values_flatten()

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_mamba_values_flatten()

    def available_and_evictable_str(self) -> str:
        # TODO(zhangmj): need more detailed log info for session reference.
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.tree_core.component_evictable_size(BASE_COMPONENT_TYPE)
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.tree_core.component_evictable_size(ct)} "
                f"(available_size={available_size} + component_evictable_size_={self.tree_core.component_evictable_size(ct)})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Pass ongoing ops as lightweight (id, node_id) pairs so the tree core
        # can resolve + validate them without reaching into Controller state.
        if self.buffer_pipeline is not None:
            ongoing_write_through = [
                (nid, entry.intent.node_id)
                for nid, entry in self.buffer_pipeline.ongoing_write_through.items()
            ]
        else:
            ongoing_write_through = [
                (nid, wt.node_id) for nid, wt in self.ongoing_write_through.items()
            ]
        ongoing_load_back = [
            (nid, lb.node_id) for nid, lb in self.ongoing_load_back.items()
        ]
        host_lock_owners = [
            HostLockOwner(
                kind=HostLockOwnerKind.LOAD_BACK,
                owner_id=operation_id,
                anchor_node_id=load_back.node_id,
                lock_params=load_back.host_lock_params,
            )
            for operation_id, load_back in self.ongoing_load_back.items()
        ]
        host_lock_owners.extend(
            HostLockOwner(
                kind=HostLockOwnerKind.STORAGE_BACKUP,
                owner_id=operation_id,
                anchor_node_id=node_id,
                lock_params=lock_params,
            )
            for operation_id, (node_id, lock_params) in self.ongoing_backup.items()
        )
        host_lock_owners.extend(
            HostLockOwner(
                kind=HostLockOwnerKind.PREFETCH,
                owner_id=request_id,
                anchor_node_id=prefetch.anchor_node_id,
                lock_params=prefetch.anchor_lock_params,
            )
            for request_id, prefetch in self.ongoing_prefetch.items()
            if prefetch.anchor_lock_params is not None
        )
        host_lock_owners.extend(
            HostLockOwner(
                kind=HostLockOwnerKind.DEMOTED_SESSION,
                owner_id=session_id,
                anchor_node_id=state.last_node,
                lock_params=state.host_lock_params,
            )
            for session_id, state in self.session.demoted.items()
        )
        self.tree_core.sanity_check(
            ongoing_write_through,
            ongoing_load_back,
            host_lock_owners,
            set(self.session.demoted),
        )

    def pretty_print(self) -> None:
        self.tree_core.pretty_print()

    # ---- TreeCore state delegation ----
    # The facade re-exposes tree-owned config (page_size, enable_storage, ...) so its
    # own coordination methods and external callers read them off the cache.

    # ``page_size`` keeps a setter: StreamingSession forwards assignment onto its
    # inner cache (the PrefixCacheTrait surface).
    @property
    def page_size(self):
        return self.tree_core.page_size

    @page_size.setter
    def page_size(self, value) -> None:
        self.tree_core.page_size = value

    @property
    def enable_storage(self):
        return self.tree_core.enable_storage

    @enable_storage.setter
    def enable_storage(self, value) -> None:
        self.tree_core.enable_storage = value

    @property
    def write_through_threshold(self):
        return self.tree_core.write_through_threshold

    @write_through_threshold.setter
    def write_through_threshold(self, value) -> None:
        self.tree_core.write_through_threshold = value

    @property
    def is_write_back(self):
        return self.tree_core.is_write_back

    @is_write_back.setter
    def is_write_back(self, value) -> None:
        self.tree_core.is_write_back = value

    @property
    def device(self):
        return self.tree_core.device

    @property
    def root_node(self):
        return self.tree_core.root_node

    def take_events(self):
        # Drain the KV event queue from the TreeCore.
        return self.tree_core.take_events()

    def resolve_node_handle(self, node_handle):
        """Look up the node object from its NodeId.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        if isinstance(node_handle, int):
            return self.tree_core.node_by_id(node_handle)
        # Internal callers (and the session sentinel / None) pass a non-int through.
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The root's NodeId -- URC match results carry NodeIds."""
        return self.tree_core.root_node_handle(extra_key)
