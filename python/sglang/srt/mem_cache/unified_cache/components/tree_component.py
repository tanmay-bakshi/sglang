from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    HostLockFootprint,
    HostLockRange,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolTransferResult
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: F401
    BASE_COMPONENT_TYPE,
    ComponentType,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


_NUM_COMPONENT_TYPES = len(ComponentType)

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0
    host_lock_ids: set[int] | None = None
    session_ref: int = 0
    session_ids: Optional[set[str]] = None


class EvictLayer(IntFlag):
    """Which storage layer(s) to evict.  Combinable via bitwise OR."""

    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


@dataclasses.dataclass(frozen=True)
class PrepareLoadBackResult:
    """Outcome of prepare_load_back; default = nothing to prepare."""

    # Freshly allocated device mamba slot, recovered on failure.
    allocated_mamba_slot: Optional[torch.Tensor] = None


@dataclasses.dataclass(frozen=True)
class PreparePrefetchResult:
    """Outcome of prepare_prefetch; default = nothing to prepare."""

    # Host pool exhausted; the caller aborts the prefetch.
    alloc_failed: bool = False
    # The component's pre-allocated host buffer (None = skip the build).
    host_indices: Optional[torch.Tensor] = None


class CacheTransferPhase(str, Enum):

    BACKUP_HOST = "backup_host"  # D→H
    LOAD_BACK = "load_back"  # H→D
    BACKUP_STORAGE = "backup_storage"  # H→Storage
    PREFETCH = "prefetch"  # Storage→H


class LRURefreshPhase(str, Enum):

    WALKDOWN = "walkdown"  # touching a node while walking through the tree
    MATCH_END = "match_end"  # end of a successful prefix match
    INSERT_END = "insert_end"  # after a new/updated leaf is committed


def get_and_increase_time_counter() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        self.cache = cache
        # Populated when the component passed to TreeCore constructor.
        self.tree_core: Optional[UnifiedTreeCore] = None
        self.is_evict_device_ongoing = False
        # Per-session frontier nodes (the deepest registered node per cached
        # path), not physical tree leaves: a frontier node may have children.
        self._session_leaves: dict[str, set[UnifiedTreeNode]] = defaultdict(set)

    # Subclasses MUST set this as a class attribute (not @property)
    component_type: ComponentType

    def reset_session_state(self) -> None:
        self._session_leaves = defaultdict(set)

    def session_leaves(self) -> dict[str, set[UnifiedTreeNode]]:
        """Return each session's registered frontier nodes, for evidence only.

        :returns: Session id to frontier nodes; callers must not mutate it.
        """
        return self._session_leaves

    def session_ref(self, node: UnifiedTreeNode) -> int:
        return node.component_data[self.component_type].session_ref

    def _can_evict_leaf_atomically(self, node: UnifiedTreeNode) -> bool:
        # Don't allow a non-session-ref component to cascade-evict
        # a high-priority session-ref component.
        if self.session_ref(node) > 0:
            return True
        priority = self.eviction_priority(is_leaf=False)
        return not any(
            comp.eviction_priority(is_leaf=False) >= priority
            and comp.session_ref(node) > 0
            for comp in self.tree_core.components
        )

    def _refresh_session_partition(self, node: UnifiedTreeNode) -> None:
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        if lru.in_list(node):
            lru.reset_node_mru(node)
        host_lru = self.tree_core.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.reset_node_mru(node)

    def _find_reusable_session_leaf(
        self, node: Optional[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        root_node = self.tree_core.root_node
        if node is None or node is root_node:
            return None

        validator = self.create_match_validator(match_device_only=False)
        cur = node
        while cur is not None and cur is not root_node:
            if validator(cur):
                return cur
            cur = cur.parent
        return None

    def resolve_session_leaf(
        self, req: Req, last_node: Optional[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        return self._find_reusable_session_leaf(last_node)

    def resolve_session_frontier(
        self, node: Optional[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        """Resolve reusable component state at an explicit radix frontier.

        :param node: Full-KV radix frontier.
        :returns: Nearest reusable frontier for this component.
        """
        return self._find_reusable_session_leaf(node)

    def _mark_session_leaf(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        self._session_leaves[session_id].add(leaf)
        cd = leaf.component_data[self.component_type]
        if cd.session_ids is None:
            cd.session_ids = set()
        cd.session_ids.add(session_id)

    def _unmark_session_leaf(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        leaves = self._session_leaves.get(session_id)
        assert leaves is not None and leaf in leaves
        leaves.remove(leaf)
        cd = leaf.component_data[self.component_type]
        assert cd.session_ids is not None and session_id in cd.session_ids
        cd.session_ids.remove(session_id)
        if not cd.session_ids:
            cd.session_ids = None
        if not leaves:
            self._session_leaves.pop(session_id, None)

    def _nearest_session_ancestor(
        self, new_leaf: UnifiedTreeNode, current_leaves: set[UnifiedTreeNode]
    ) -> Optional[UnifiedTreeNode]:
        cur = new_leaf.parent
        while cur is not None and cur is not self.tree_core.root_node:
            if cur in current_leaves:
                return cur
            cur = cur.parent
        return None

    @abstractmethod
    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        """Remove one session reference from the coverage anchored at `leaf`."""
        ...

    @abstractmethod
    def _session_coverage_nodes(
        self,
        leaf: UnifiedTreeNode,
    ) -> tuple[UnifiedTreeNode, ...]:
        """Return the nodes whose counters one indexed leaf contributes to."""
        ...

    @abstractmethod
    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        """Move the session's coverage from `old_ancestor` forward to `leaf`."""
        ...

    @abstractmethod
    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        """Move the session's coverage backward from `leaf` to `fallback`."""
        ...

    def register_session_leaf(
        self, session_id: str, leaf: Optional[UnifiedTreeNode]
    ) -> None:
        if (
            not self.tree_core.enable_session_radix_cache
            or leaf is None
            or leaf is self.tree_core.root_node
        ):
            return

        current_leaves = self._session_leaves[session_id]
        if leaf in current_leaves:
            return

        old_ancestor = self._nearest_session_ancestor(leaf, current_leaves)
        self._advance_session_coverage(session_id, leaf, old_ancestor)
        self._mark_session_leaf(session_id, leaf)
        if old_ancestor is not None:
            self._unmark_session_leaf(session_id, old_ancestor)

    def release_session(self, session_id: str) -> int:
        leaves = tuple(self._session_leaves.get(session_id, ()))
        for leaf in leaves:
            self._dec_session_coverage(session_id, leaf)
            self._unmark_session_leaf(session_id, leaf)
        return len(leaves)

    def discard_deleted_session_leaf(self, node: UnifiedTreeNode) -> None:
        cd = node.component_data[self.component_type]
        for session_id in tuple(cd.session_ids or ()):
            leaves = self._session_leaves.get(session_id)
            assert leaves is not None and node in leaves
            fallback = self._find_reusable_session_leaf(node.parent)
            if fallback is not None and fallback in self._session_leaves.get(
                session_id, ()
            ):
                fallback = None
            self._recede_session_coverage(session_id, node, fallback)
            self._unmark_session_leaf(session_id, node)
            if fallback is not None:
                self._mark_session_leaf(session_id, fallback)

    def validate_session_state(
        self,
        reachable_nodes: set[UnifiedTreeNode],
        report_error: Callable[[str], None],
    ) -> None:
        reachable_nodes = set(reachable_nodes)
        ct = self.component_type
        expected_refs: dict[UnifiedTreeNode, int] = defaultdict(int)

        for session_id, leaves in self._session_leaves.items():
            if not leaves:
                report_error(f"{ct} session {session_id!r} has an empty leaf index")
            for leaf in leaves:
                if leaf not in reachable_nodes:
                    report_error(
                        f"{ct} session {session_id!r} indexes unreachable leaf {leaf.id}"
                    )
                if session_id not in (leaf.component_data[ct].session_ids or ()):
                    report_error(
                        f"{ct} session {session_id!r} leaf {leaf.id} is missing its marker"
                    )
                if leaf not in reachable_nodes:
                    continue
                try:
                    coverage_nodes = self._session_coverage_nodes(leaf)
                except (AssertionError, RuntimeError) as error:
                    report_error(
                        f"{ct} session {session_id!r} leaf {leaf.id} has invalid "
                        f"coverage: {error}"
                    )
                    continue
                for covered_node in coverage_nodes:
                    if covered_node not in reachable_nodes:
                        report_error(
                            f"{ct} session {session_id!r} coverage reaches "
                            f"unreachable node {covered_node.id}"
                        )
                        continue
                    expected_refs[covered_node] += 1

        for node in reachable_nodes:
            cd = node.component_data[ct]
            expected_ref = expected_refs.get(node, 0)
            if cd.session_ref != expected_ref:
                report_error(
                    f"node {node.id} {ct} session_ref={cd.session_ref} "
                    f"expected={expected_ref}"
                )
            if cd.session_ids is not None and not cd.session_ids:
                report_error(f"node {node.id} {ct} has an empty session_ids marker")
            for session_id in cd.session_ids or ():
                if node not in self._session_leaves.get(session_id, ()):
                    report_error(
                        f"node {node.id} {ct} session {session_id!r} marker is not indexed"
                    )

    def node_has_component_data(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> bool:
        cd = node.component_data[self.component_type]
        if target is EvictLayer.DEVICE:
            return cd.value is not None
        return cd.host_value is not None

    def value_len(self, node: UnifiedTreeNode) -> int:
        value = node.component_data[self.component_type].value
        return len(value) if value is not None else 0

    def has_host_value_only(self, node: UnifiedTreeNode) -> bool:
        """Whether this component's data is evicted from device but host-backed."""
        cd = node.component_data[self.component_type]
        return cd.value is None and cd.host_value is not None

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:
        """Whether this component has new device data missing from Host."""
        return False

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        ct = self.component_type
        match phase:
            case LRURefreshPhase.WALKDOWN:
                if node.component_data[ct].value is None:
                    return
                self.tree_core.lru_lists[ct].reset_node_mru(node)
            case LRURefreshPhase.MATCH_END:
                self.tree_core.lru_lists[ct].reset_node_and_parents_mru(
                    node, root_node, self.node_has_component_data
                )
            case LRURefreshPhase.INSERT_END:
                # WALKDOWN already refreshed every node on the insert path
                # (including the new leaf), so there is nothing more to do.
                return
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    @abstractmethod
    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        """Return a per-match stateful predicate that decides whether a node
        is a valid match boundary for this component.
        Called once per match_prefix; the returned closure may carry state.
        When match_device_only is true, host-backed nodes must not be accepted
        as valid match boundaries.
        - Full: returns True if the node has full component data.
        - SWA: tracks accumulated length since last gap; returns True only
          when the contiguous window reaches swa_sliding_window_size.
        - Mamba: returns True iff the node has mamba component data."""
        ...

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """Tree-side post-processing inside the match walk (no cache access)."""
        return result

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        """Cache-level finalize after the match walk, dispatched by
        `UnifiedRadixCache.match_prefix`; receives the NodeId-based result.
        - Mamba: performs the copy-on-write into a per-request slot."""
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> int:
        """Called per-node when an insert's key overlaps an existing node.
        Returns the index within value_slice from which this component
        consumed (took ownership of) the underlying KV pool slots.
        Returns prefix_len if nothing was consumed (default).
        _insert_helper uses this to free only the non-consumed duplicate
        portion: value_slice[dup_start:consumed_from]."""
        return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Called after _unevict_node_on_insert restores the base (Full) value
        on an evicted node. Aux components (e.g. SWA) override this to rebuild
        their own data from the freshly assigned base value when their entry
        is still tombstoned. Default no-op."""
        return None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Finalize component data on the target (leaf) node after the insert
        walk completes. Called once per insert.
        - Full: no-op (full data is handled by _add_new_node).
        - SWA: for new leaves, checks whether the node straddles the SWA
          eviction boundary (swa_evicted_seqlen). If so, splits the node
          via _split_node — the parent becomes a tombstone (no SWA) and the
          child (the deeper portion) receives SWA data. If the entire node
          is within the window, sets SWA directly. If entirely outside,
          leaves SWA as None (tombstone).
        - Mamba: sets the mamba component value from params, inserts into
          mamba LRU list, and increments evictable size. If the node already
          has mamba data, resets its LRU position instead."""
        pass

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        """Redistribute component data between new_parent and child when a
        node is split. new_parent is the newly created prefix node.
        - Full: copies child's lock_ref to new_parent.
        - SWA: slices (or clones) the swa value for new_parent, copies
          lock_ref and component_uuid metadata, then syncs child's swa
          value with its (now-trimmed) full_value.
        - Mamba: sets new_parent's mamba value to None and lock_ref to 0
          (mamba data stays on the original leaf, not on prefix nodes)."""
        ...

    @abstractmethod
    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """Free this component's KV resources on a node being evicted.

        *target* controls which layer(s) to evict:
          - DEVICE: free device memory and tombstone (value = None).
                    Host data is untouched.
          - HOST:   free host memory (host_value = None).
                    Device data is untouched.
          - ALL:    free both device and host memory.
                    No tombstone — caller will delete the node.

        Freed indices are collected into *device_frees*/*host_frees* for the
        Controller to drain, never freed inline.

        Returns (device_freed, host_freed) token counts."""
        ...

    def eviction_priority(self, is_leaf: bool) -> int:
        """Eviction priority on this node type. Higher = evicted later.
        When a component is evicted, all other components with equal or
        lower priority on the same node are also cascade-evicted.

        Leaf: all components equal (0) — evicting any cascades to all,
        because the node will be deleted.

        Internal: full=2 > swa=1 > mamba=0.
        Why swa > mamba: SWA data on internal nodes is *path data* —
        the sliding window needs continuous SWA coverage along the path
        from root to the match boundary. E.g. A->B->C->D->E where C
        and E both have mamba and the window covers C->E: if C's mamba
        is evicted, C's SWA must stay so E remains reachable.
        Mamba data, by contrast, is only meaningful at the match
        boundary node; on internal nodes it
        contributes nothing to the path. So SWA is more valuable to
        keep and should be evicted later.

        Cascade consequences:
        - Mamba evict internal: no cascade.
        - SWA evict internal: cascades to Mamba. SWA gone -> SWA
          validator fails -> mamba data is useless (match requires all
          validators to pass).
        - Full evict internal: cascades to SWA + Mamba."""
        return 0

    def evict_device_start(self, request_cnt: int) -> None:
        """Begin this component's device-eviction walk (build its cursor/heap)."""
        assert (
            not self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction already in progress"
        self._evict_device_start(request_cnt)
        self.is_evict_device_ongoing = True

    def evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """Advance one eviction step and return a device leaf, if selected.

        Implementations must return after one allocator-relevant internal
        mutation so the caller can drain pending frees before continuing.
        """
        assert (
            self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction not started"
        return self._evict_device_next_node(tracker, device_frees, host_frees)

    def evict_device_end(self) -> None:
        """Clear this component's device-eviction walk state."""
        assert (
            self.is_evict_device_ongoing
        ), f"{self.component_type} device eviction not started"
        self._evict_device_end()
        self.is_evict_device_ongoing = False

    @abstractmethod
    def _evict_device_start(self, request_cnt: int) -> None:
        """Build this component's eviction cursor/heap."""
        ...

    @abstractmethod
    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """Advance the walk by at most one allocator-relevant mutation."""
        ...

    @abstractmethod
    def _evict_device_end(self) -> None:
        """Clear this component's eviction cursor/heap."""
        ...

    @abstractmethod
    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        """Increment component lock refs, protecting nodes from
        eviction. Updates evictable → protected size on first lock.
        - Full: path-lock — walks from node up to root, incrementing
          lock_ref on every ancestor.
        - SWA: path-lock — walks upward collecting swa values until the
          sliding window is filled; records a component_uuid at the
          boundary for release_component_lock to know where to stop.
        - Mamba: single-node lock — only increments lock_ref on the
          node itself (mamba state is per-leaf, not per-path).

        When ``lock_host`` is True, the lock applies to host-side state:
        - Full: single-node host lock.
        - SWA: host window-lock with a dedicated host UUID boundary.
        - Mamba: single-node host lock with host LRU detach."""
        ...

    @abstractmethod
    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        """Decrement component lock refs, un-protecting nodes.
        Updates protected → evictable size when lock_ref drops to 0.
        - Full: path-unlock — walks from node up to root, decrementing
          lock_ref on every ancestor.
        - SWA: path-unlock — walks upward, stopping at the node whose
          component_uuid matches the one recorded during acquire.
        - Mamba: single-node unlock — only decrements lock_ref on the
          node itself.

        When ``lock_host`` is True, the inverse host-side semantics apply."""
        ...

    def host_lock_nodes(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
    ) -> tuple[UnifiedTreeNode, ...]:
        """Resolve a receipt's split-stable footprint from tree geometry."""
        footprint = params.host_lock_footprints.get(self.component_type)
        if footprint is None:
            return ()
        return self._resolve_host_lock_ranges(
            node,
            footprint.ranges,
            require_host_value=footprint.requires_host_value,
        )

    def _host_lock_path(
        self,
        node: UnifiedTreeNode,
    ) -> tuple[tuple[UnifiedTreeNode, HostLockRange], ...]:
        """Return root-relative ranges for the root-to-anchor path."""
        path: list[UnifiedTreeNode] = []
        current = node
        while current is not self.tree_core.root_node:
            if current.parent is None or current.key is None:
                raise AssertionError(
                    f"Host-lock anchor path is detached at node {current.id}."
                )
            path.append(current)
            current = current.parent
        path.reverse()

        offset = 0
        result: list[tuple[UnifiedTreeNode, HostLockRange]] = []
        for path_node in path:
            node_end = offset + len(path_node.key)
            result.append((path_node, HostLockRange(offset, node_end)))
            offset = node_end
        return tuple(result)

    def _host_lock_range(self, node: UnifiedTreeNode) -> HostLockRange:
        """Return one node's stable root-relative acquisition range."""
        for path_node, path_range in self._host_lock_path(node):
            if path_node is node:
                return path_range
        raise AssertionError(f"Host-lock node {node.id} is absent from its own path.")

    def _resolve_host_lock_ranges(
        self,
        node: UnifiedTreeNode,
        ranges: tuple[HostLockRange, ...],
        *,
        require_host_value: bool = True,
    ) -> tuple[UnifiedTreeNode, ...]:
        """Map acquisition ranges onto the current, possibly split path."""
        if len(ranges) == 0:
            raise AssertionError(
                f"{self.component_type} host-lock receipt has an empty footprint."
            )
        ordered_ranges = tuple(sorted(ranges, key=lambda range_: range_.start))
        for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
            if current.start < previous.end:
                raise AssertionError(
                    f"{self.component_type} host-lock ranges overlap: "
                    f"{previous} and {current}."
                )

        path = self._host_lock_path(node)
        owned_nodes: list[UnifiedTreeNode] = []
        for receipt_range in ordered_ranges:
            cursor = receipt_range.start
            for path_node, path_range in path:
                if path_range.end <= receipt_range.start:
                    continue
                if path_range.start >= receipt_range.end:
                    break
                if path_range.start != cursor or path_range.end > receipt_range.end:
                    raise AssertionError(
                        f"{self.component_type} host-lock range {receipt_range} "
                        f"cuts across current node {path_node.id} range {path_range}."
                    )
                component_data = path_node.component_data[self.component_type]
                if require_host_value and component_data.host_value is None:
                    raise AssertionError(
                        f"{self.component_type} host-lock range {receipt_range} "
                        f"reaches host-empty node {path_node.id}."
                    )
                if component_data.host_value is not None:
                    logical_length = self.tree_core.component_logical_length(
                        path_node,
                        self.component_type,
                        host=True,
                    )
                    if logical_length != len(path_node.key):
                        raise AssertionError(
                            f"{self.component_type} host value on node "
                            f"{path_node.id} spans {logical_length} tokens, "
                            f"tree range spans {len(path_node.key)}."
                        )
                owned_nodes.append(path_node)
                cursor = path_range.end
            if cursor != receipt_range.end:
                raise AssertionError(
                    f"{self.component_type} host-lock range {receipt_range} "
                    f"is covered only through token {cursor}."
                )
        if len(set(owned_nodes)) != len(owned_nodes):
            raise AssertionError(
                f"{self.component_type} host-lock receipt resolves a node twice."
            )
        owned_nodes.reverse()
        return tuple(owned_nodes)

    def validate_host_lock_nodes(
        self,
        nodes: tuple[UnifiedTreeNode, ...],
        params: DecLockRefParams,
    ) -> None:
        """Validate node counters and IDs without mutating the footprint."""
        lock_id = params.host_lock_id
        if lock_id is None:
            raise AssertionError("Host-lock receipt has no ownership identifier.")
        for owned_node in nodes:
            component_data = owned_node.component_data[self.component_type]
            lock_ids = component_data.host_lock_ids
            if lock_ids is None or lock_id not in lock_ids:
                raise AssertionError(
                    f"Node {owned_node.id} {self.component_type} does not carry "
                    f"host-lock owner {lock_id}."
                )
            if component_data.host_lock_ref != len(lock_ids):
                raise AssertionError(
                    f"Node {owned_node.id} {self.component_type} host-lock "
                    f"counter disagrees with its owner ledger."
                )

    def release_prepared_host_lock(
        self,
        nodes: tuple[UnifiedTreeNode, ...],
        params: DecLockRefParams,
    ) -> None:
        """Release a footprint after every component has passed preflight."""
        for owned_node in nodes:
            self._release_host_lock_ownership(owned_node, params)
        for owned_node in nodes:
            if self.component_type == BASE_COMPONENT_TYPE:
                self.tree_core._update_evictable_leaf_sets(owned_node)
            else:
                self.tree_core._reconcile_auxiliary_host_lru(
                    owned_node,
                    self.component_type,
                )

    def _release_host_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
    ) -> None:
        """Validate and release one component outside a core-wide transaction."""
        nodes = self.host_lock_nodes(node, params)
        self.validate_host_lock_nodes(nodes, params)
        self.release_prepared_host_lock(nodes, params)

    def _acquire_host_lock_ownership(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        *,
        requires_host_value: bool = True,
    ) -> None:
        """Register one receipt as an exact owner of this component state."""
        lock_id = result.host_lock_id
        if lock_id is None:
            raise AssertionError("host-lock acquisition has no ownership identifier")
        component_data = node.component_data[self.component_type]
        lock_ids = component_data.host_lock_ids
        if lock_ids is not None and lock_id in lock_ids:
            raise AssertionError(
                f"node {node.id} already carries host-lock owner {lock_id}"
            )
        host_lock_range = self._host_lock_range(node)
        footprint = result.host_lock_footprints.get(self.component_type)
        if (
            footprint is not None
            and footprint.requires_host_value != requires_host_value
        ):
            raise AssertionError(
                f"{self.component_type} host-lock acquisition mixed host-residency "
                "requirements."
            )
        result.host_lock_footprints[self.component_type] = HostLockFootprint(
            ranges=(
                *(footprint.ranges if footprint is not None else ()),
                host_lock_range,
            ),
            requires_host_value=requires_host_value,
        )
        if lock_ids is None:
            lock_ids = set()
            component_data.host_lock_ids = lock_ids
        lock_ids.add(lock_id)
        component_data.host_lock_ref += 1
        assert component_data.host_lock_ref == len(lock_ids)

    def _release_host_lock_ownership(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams,
    ) -> None:
        """Release this component state from its exact receipt owner."""
        lock_id = params.host_lock_id
        if lock_id is None:
            raise AssertionError("host-lock release has no ownership identifier")
        component_data = node.component_data[self.component_type]
        lock_ids = component_data.host_lock_ids
        if lock_ids is None or lock_id not in lock_ids:
            raise AssertionError(
                f"node {node.id} does not carry host-lock owner {lock_id}"
            )
        assert component_data.host_lock_ref == len(lock_ids)
        lock_ids.remove(lock_id)
        component_data.host_lock_ref -= 1
        if len(lock_ids) == 0:
            component_data.host_lock_ids = None
        assert component_data.host_lock_ref == len(component_data.host_lock_ids or ())

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        """Prepare component-specific data before insert, fill component
        fields in insert_params, return effective cache_len.
        Return None for no truncation opinion (use full length);
        return int >= 0 for effective cache length.
        - Full: no-op, returns None.
        - SWA: sets insert_params.swa_evicted_seqlen on finished; returns None.
        - Mamba: prepares mamba_value (finished from ping-pong buffer,
          unfinished fork from req); returns mamba_last_track_seqlen."""
        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        """Post-cache cleanup for component-specific resources.

        ``is_finished`` — whether the request has finished generation.
        True means the request is complete and its resources can be released;
        ``insert_result`` is None when insert was skipped (cache disabled
        or effective_cache_len <= 0); treat as "no insert happened".
        ``insert_params`` is None only on the disabled path; on early-return
        paths it is still provided so components can free their resources."""
        pass

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        pass

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        """Free evicted host-tier values back to this component's host pool."""
        raise NotImplementedError(f"{self.component_type} must free its host values")

    # ---- HiCache Hooks ----

    def prepare_load_back(
        self,
        node_id: NodeId,
        *,
        req: Optional[Req] = None,
    ) -> PrepareLoadBackResult:
        """Cache-level pre-allocation before a load-back builds its transfers."""
        return PrepareLoadBackResult()

    def finalize_load_back(
        self, req: Optional[Req], prep: PrepareLoadBackResult, success: bool
    ) -> None:
        """Release state populated by prepare_load_back when the load-back did
        not go through."""
        pass

    def prepare_prefetch(
        self,
        node_id: NodeId,
        *,
        prefetch_tokens: int = 0,
    ) -> PreparePrefetchResult:
        """Cache-level host pre-allocation before a prefetch builds its transfers."""
        return PreparePrefetchResult()

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Build transfer descriptors for this component in the given phase.
        Returns None if the component has nothing to transfer."""
        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Post-transfer bookkeeping: store host indices, update LRU, etc."""
        pass

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict from this component's host-side resources, collecting freed
        values into *device_frees*/*host_frees* for the Controller to drain.
        Called by HostPoolGroup when the host pool is full.
        Default no-op for components without host storage."""
        pass

    def apply_component_action(self, action: ComponentAction) -> None:
        """Apply a component-routed cache action; dispatched by the cache."""
        raise NotImplementedError(
            f"{self.component_type} cannot apply {type(action).__name__}"
        )
