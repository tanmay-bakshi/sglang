from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_observability

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        ComponentType,
    )


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # DSV4 NPU C128 sidecar pages, one page id per physical C128 page group.
    c128_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0
    trigger_backup: bool = True


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    inserted_host_node: Any = None
    host_insert_dropped: bool = False
    # Controller-applied actions from the non-stepped channels (e.g. insert_host); the stepped insert emits via InsertStepResult.actions.
    cache_actions: list[CacheAction | ComponentAction] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class KVComponentResidency:
    """Physical page residency for one KV component.

    :ivar device_pages: Pages currently allocated in the device KV pool.
    :ivar host_backed_pages: Pages with a copy in the host KV pool.
    """

    device_pages: int = 0
    host_backed_pages: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class StreamingSessionCacheSnapshot:
    """Read-only per-session KV ownership summary.

    :ivar protected: Tokens protected through shared radix-tree ownership.
    :ivar held_tokens: Tokens held exclusively by the detached session slot.
    :ivar full: Residency of full-attention KV pages.
    :ivar swa: Residency of sliding-window-attention KV pages.
    """

    protected: int = 0
    held_tokens: int = 0
    full: KVComponentResidency = dataclasses.field(default_factory=KVComponentResidency)
    swa: KVComponentResidency = dataclasses.field(default_factory=KVComponentResidency)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionReloadPlan:
    """Allocation-free plan for restoring one host-resident session.

    :ivar session_id: Session whose private host snapshot is authoritative.
    :ivar source_node: Exact session-private host frontier.
    :ivar tree_node: Deepest ordinary radix frontier retained by the session.
    :ivar reuse_len: Exact logical prefix available to the request.
    :ivar tree_protected_len: Logical prefix represented by ordinary tree nodes.
    :ivar swa_evicted_seqlen: First token retained in the SWA window.
    """

    session_id: str
    source_node: Any
    tree_node: Any
    reuse_len: int
    tree_protected_len: int
    swa_evicted_seqlen: int

    def __post_init__(self) -> None:
        if self.reuse_len <= 0:
            raise ValueError("A session reload plan must expose a non-empty prefix.")
        if not 0 <= self.tree_protected_len <= self.reuse_len:
            raise ValueError(
                "Session reload tree ownership must lie within the reusable prefix."
            )
        if not 0 <= self.swa_evicted_seqlen <= self.reuse_len:
            raise ValueError(
                "Session reload SWA watermark must lie within the reusable prefix."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class HostLockRange:
    """Root-relative tree interval protected by one host lock.

    :ivar start: Inclusive logical token offset.
    :ivar end: Exclusive logical token offset.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid host-lock range [{self.start}, {self.end}).")


@dataclasses.dataclass(frozen=True, slots=True)
class HostLockFootprint:
    """Split-stable component ownership recorded by one host-lock receipt.

    :ivar ranges: Root-relative tree intervals occupied at acquisition.
    :ivar requires_host_value: Whether host residency existed at acquisition.
    """

    ranges: tuple[HostLockRange, ...]
    requires_host_value: bool


@dataclasses.dataclass
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""

    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None
    host_lock_id: int | None = None
    host_lock_footprints: dict[ComponentType, HostLockFootprint] = dataclasses.field(
        default_factory=dict
    )
    # Component nodes that were tombstones at acquire time. Replaying this set
    # at release prevents a short-lived lock from consuming a later load-back or
    # request lock after that tombstone becomes a valid device value.
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            host_lock_id=self.host_lock_id,
            host_lock_footprints={
                component_type: HostLockFootprint(
                    ranges=tuple(
                        sorted(
                            footprint.ranges,
                            key=lambda range_: range_.start,
                        )
                    ),
                    requires_host_value=footprint.requires_host_value,
                )
                for component_type, footprint in self.host_lock_footprints.items()
            },
            skip_lock_node_ids={
                component_type: set(node_ids)
                for component_type, node_ids in self.skip_lock_node_ids.items()
            },
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""

    swa_uuid_for_lock: Optional[int] = None
    host_lock_id: int | None = None
    host_lock_footprints: dict[ComponentType, HostLockFootprint] = dataclasses.field(
        default_factory=dict
    )
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )


class HostLockOwnerKind(StrEnum):
    """Lifecycle state responsible for one host-lock acquisition."""

    LOAD_BACK = "load_back"
    STORAGE_BACKUP = "storage_backup"
    PREFETCH = "prefetch"
    DEMOTED_SESSION = "demoted_session"


@dataclasses.dataclass(frozen=True, slots=True)
class HostLockOwner:
    """Live operation or session that owns one host-lock acquisition.

    :ivar kind: Lifecycle state retaining the lock.
    :ivar owner_id: Identity within that lifecycle state.
    :ivar anchor_node_id: Tree anchor passed to ``inc_host_lock_ref``.
    :ivar lock_params: Acquisition receipt used for release and validation.
    """

    kind: HostLockOwnerKind
    owner_id: str | int
    anchor_node_id: int
    lock_params: DecLockRefParams


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


@dataclasses.dataclass(frozen=True)
class LoadBackResult:
    """Result of preparing a host-to-device cache restoration.

    :ivar new_full_device_indices: Newly allocated full-KV device indices.
    :ivar restored_node: Deepest cache node restored to device residency.
    :ivar queued_any_component: Whether an asynchronous component transfer was queued.
    :ivar full_tokens: Number of full-KV tokens restored by this operation.
    :ivar swa_tokens: Number of sliding-window tokens restored by this operation.
    :ivar cache_protected_len: Exact restored prefix still owned by the radix tree.
    """

    new_full_device_indices: torch.Tensor
    restored_node: Any
    queued_any_component: bool
    full_tokens: int
    swa_tokens: int
    cache_protected_len: int | None = None


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
        full_kv_hit_length: Longest Full-KV prefix available on either device or
                            host, independent of other components.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None
    full_kv_hit_length: int = 0
    session_reload_plan: SessionReloadPlan | None = None
    # Actions the Controller applies: CacheActions itself, ComponentActions routed to the owning component.
    cache_actions: Sequence[CacheAction | ComponentAction] = ()


def zero_match_result(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node_handle(extra_key=extra_key)
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        full_kv_hit_length=0,
        session_reload_plan=None,
    )


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )
    cache_controller: Optional[HiCacheController] = None
    # Set by caches that publish KV placement events; None means they don't.
    kv_events: Optional[KVCacheEventRecorder] = None

    def init_metrics_collector(self):
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        labels = {"cache_type": self.__class__.__name__}
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        labels.update(
            {
                "pp_rank": parallel.pp_rank,
                "tp_rank": parallel.tp_rank,
            }
        )
        radix_cache_cls = resolve_collector_class(
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    def release_host_resources(self) -> None:
        """Release pinned host buffers in userspace on graceful shutdown.

        Kernel-side unpinning during process reclaim can stall teardown for
        tens of seconds (see HostKVCache.destroy). Idempotent.
        """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    def supports_fast_match_prefix(self) -> bool:
        return False

    def resolve_node_handle(self, node_handle: Any) -> Any:
        """Map a node handle to its node -- e.g. UnifiedRadixCache looks up the
        node object from its NodeId. Temporary API for the Unified Radix Cache
        split migration.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> Any:
        """The root handle as match results carry it -- the raw node by default,
        the root's NodeId for UnifiedRadixCache. extra_key scopes the root for
        implementations that shard trees per cache namespace."""
        return self.root_node

    def is_backuped(self, node: Any) -> bool:
        """Whether the node's Full KV is present on host."""
        return node.backuped

    def is_root(self, node: Any) -> bool:
        """Whether the node is a tree root."""
        return node is self.root_node

    def get_last_hash_value(self, node: Any) -> Optional[str]:
        """The node's last page hash, or None when it was never hashed."""
        return node.get_last_hash_value()

    def get_prefix_hash_values(self, node: Any) -> list[str]:
        """The hash chain of the node's ancestors, in root-to-parent order."""
        return node.get_prefix_hash_values(node.parent)

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict cache entries to cover allocator shortfalls.

        The default implementation preserves the component-count semantics of
        :meth:`evict`. Multi-component caches backed by shared memory can
        override this entry point to stop once collateral frees make the
        requested allocation feasible.
        """
        return self.evict(params)

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> LoadBackResult:
        """Prepare KV-cache restoration from host to device.

        :param params: Cache-specific restoration parameters.
        :returns: The prepared restoration and its transfer state.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return [] if self.kv_events is None else self.kv_events.take()

    def supports_swa(self) -> bool:
        return False

    def swa_retain_floor(self, req) -> int | None:
        # A match lands on a state checkpoint rather than on the tail, so a cache
        # that pairs SWA with mamba/conv checkpoints has to keep the window behind
        # the last checkpoint. Those caches override this. Everyone else has
        # nothing deeper than the tail to protect.
        return None

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        return False

    def supports_streaming_session(self) -> bool:
        return False

    def supports_streaming_session_demotion(self) -> bool:
        """Return whether streaming sessions can become host-resident.

        :returns: Whether transactional host demotion is available.
        """
        return False

    def is_streaming_session_demoted(self, session_id: str) -> bool:
        """Return whether one streaming session is host-resident.

        :param session_id: Session identifier to inspect.
        :returns: Whether the session has a durable host frontier.
        """
        return False

    def prepare_streaming_session_demotion(
        self,
        session_id: str,
        token_ids: Sequence[int],
        extra_key: str | None,
        cache_salt: str | None,
        priority: int,
    ) -> int | None:
        """Privately stage one session's host copy before a distributed vote.

        :param session_id: Session identifier to stage.
        :param token_ids: Complete committed token lineage.
        :param extra_key: Radix cache classification key.
        :param cache_salt: Radix cache namespace salt.
        :param priority: Eviction priority inherited from the session.
        :returns: Exact staged token count, or ``None`` on rejection.
        """
        return None

    def discard_streaming_session_demotion(self, session_id: str) -> None:
        """Discard one private host stage after a failed distributed vote.

        :param session_id: Session identifier whose stage must be discarded.
        """
        return None

    def commit_streaming_session_demotion(self, session_id: str) -> int:
        """Publish one unanimously prepared host stage.

        :param session_id: Session identifier whose stage must be committed.
        :returns: Exact host-backed token count.
        """
        raise NotImplementedError()

    def clear_radix_session_refs(self, session_id: str) -> int:
        """Release cache coverage without closing the session generation.

        :param session_id: Session identifier whose coverage must be released.
        :returns: Number of component frontier tags removed.
        """
        return 0

    def retire_streaming_session_private_path(self, session_id: str, node: Any) -> None:
        """Retire one session-private exact suffix path.

        :param session_id: Session identifier that owns the private path.
        :param node: Exact host frontier, or an ordinary radix frontier.
        """
        return None

    def streaming_session_private_parent(self, node: Any) -> Any | None:
        """Return the ordinary radix parent of a private session path.

        :param node: Exact host frontier, or an ordinary aligned frontier.
        :returns: The ordinary parent, or ``None`` without a private suffix.
        """
        return None

    def adopt_streaming_session_private_path(self, session_id: str, node: Any) -> int:
        """Transfer a restored private suffix to a detached session slot.

        :param session_id: Session identifier that owns the private path.
        :param node: Exact restored frontier.
        :returns: Logical private-suffix length, or zero without one.
        """
        return 0

    def release_session(self, session_id: str) -> None:
        pass

    def truncate_session(self, session_id: str, target: int) -> None:
        pass

    def commit_session(self, session_id: str, floor: int) -> None:
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def streaming_session_cache_snapshot(
        self, session_id: str
    ) -> StreamingSessionCacheSnapshot:
        """Return the durable KV ownership for one streaming session.

        :param session_id: Session identifier to inspect.
        :returns: An empty snapshot for caches without streaming-session state.
        """
        return StreamingSessionCacheSnapshot()

    def streaming_session_protected_residency(
        self, node: Any
    ) -> tuple[KVComponentResidency, KVComponentResidency]:
        """Return component residency on a streaming session's locked tree path.

        :param node: Cache-specific tree node held by the session slot.
        :returns: Full and SWA residency, or zeros for caches without tiering.
        """
        return KVComponentResidency(), KVComponentResidency()

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
