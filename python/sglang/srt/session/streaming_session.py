from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    KVComponentResidency,
    LoadBackResult,
    MatchPrefixParams,
    MatchResult,
    StreamingSessionCacheSnapshot,
)
from sglang.srt.mem_cache.common import (
    free_mapped_swa_slots,
    streaming_session_swa_eviction_plan,
)
from sglang.srt.utils.common import ceil_align, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


class _VirtualNode:
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the cache can distinguish
    streaming-session locks (no-op) from real radix-tree locks (forwarded).
    """

    pass


@dataclass
class SessionSlot:
    """Holds KV state between streaming session turns."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state
    req_pool_idx: Optional[int] = None
    kv_committed_len: int = 0
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)
    streaming_session_floor: int = 0

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    cache_protected_len: int = 0
    tree_protected_len: int = -1
    swa_uuid_for_lock: Optional[str] = None
    # components the first req skipped locking on last_node, so release dec
    # releases only what it took (may share the node with another req).
    skip_lock_node_ids: dict = field(default_factory=dict)

    # Mamba states
    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    # DSV4 compressed-state watermarks follow the same detached ownership as
    # the request row and must advance with an off-request SWA reconciliation.
    c4_state_alloc_offset: int = 0
    c128_state_alloc_offset: int = 0

    def __post_init__(self) -> None:
        if self.tree_protected_len < 0:
            self.tree_protected_len = self.cache_protected_len

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.req_pool_idx is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv = copy.copy(req.kv)
        assert req.streaming_session_floor is not None
        self.streaming_session_floor = req.streaming_session_floor

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.tree_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock
            self.skip_lock_node_ids = req.skip_lock_node_ids

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_idx = req.mamba_last_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen
        self.c4_state_alloc_offset = getattr(req, "c4_state_alloc_offset", 0)
        self.c128_state_alloc_offset = getattr(req, "c128_state_alloc_offset", 0)

        # Ownership has transferred to the slot. Null *all* of the req's
        # references so any later alloc()/free path that inspects the req
        # (e.g. the alloc-skip check on `req.mamba_ping_pong_track_buffer
        # is None`, or the retract cleanup) sees no dangling pointers
        # into slot-owned tensors. Without this the alloc path can decide
        # the req still has a ping-pong buffer and skip alloc, causing
        # the slot's tensor to be reused by a new req and leaked when
        # the slot is later freed.
        req.req_pool_idx = None
        req.kv = ReqKvInfo()
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv = copy.copy(self.kv)
        req.swa_uuid_for_lock = self.swa_uuid_for_lock
        req.streaming_session_floor = self.streaming_session_floor
        req.streaming_session_tree_protected_len = self.tree_protected_len
        req.skip_lock_node_ids = self.skip_lock_node_ids

        req.mamba_pool_idx = self.mamba_pool_idx
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer
        req.mamba_next_track_idx = self.mamba_next_track_idx
        req.mamba_last_track_idx = self.mamba_last_track_idx
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen
        req.mamba_branching_seqlen = self.mamba_branching_seqlen
        req.c4_state_alloc_offset = self.c4_state_alloc_offset
        req.c128_state_alloc_offset = self.c128_state_alloc_offset

        # NOTE: req_pool_idx and mamba_pool_idx are intentionally NOT cleared
        # from the slot. During chunked prefill, a request may be rejected by
        # the scheduler (e.g. budget exhausted) and retried in the next cycle.
        # Each retry calls match_prefix -> restore_to_req again, so the slot
        # must remain intact for idempotent restoration.


@dataclass(frozen=True, slots=True)
class DemotedSessionState:
    """Host-resident radix frontier retained for an open streaming session.

    :ivar last_node: Radix node anchoring the reusable host prefix.
    :ivar cache_protected_len: Exact number of host-backed tokens.
    :ivar swa_evicted_seqlen: First restored token with live SWA device state.
    :ivar host_lock_params: Component lock coordinates required for release.
    """

    last_node: Any
    cache_protected_len: int
    swa_evicted_seqlen: int
    host_lock_params: DecLockRefParams


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class StreamingSession(BasePrefixCache):
    """Adds streaming-session KV save/restore on top of any BasePrefixCache.

    Works both as an external wrapper (``StreamingSession(RadixCache(...))``)
    and in embedded composition (``StreamingSession(inner=self)``). For the
    embedded case, the composing cache must pre-check dispatch conditions
    (``_is_streaming`` / ``find_active_slot`` / ``has_slot``) so the internal
    fall-through to ``self.inner.xxx`` never fires -- otherwise it recurses.
    """

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}
        self.demoted: dict[str, DemotedSessionState] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    @property
    def metrics_collector(self):
        return self.inner.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self.inner.metrics_collector = value

    # -- Condition helpers (used by embedded-mode callers for pre-dispatch) --

    def has_slot(self, session_id: str) -> bool:
        return session_id in self.slots

    def any_holding_kv(self) -> bool:
        return any(s.is_holding_kv for s in self.slots.values())

    def is_demoted(self, session_id: str) -> bool:
        """Return whether one session currently owns a host frontier.

        :param session_id: Session identifier to inspect.
        :returns: Whether the session is host-resident.
        """
        return session_id in self.demoted

    def restore_demoted_request_state(self, req: Req | None, matched_len: int) -> None:
        """Restore physical ownership cursors for one host-resident match.

        :param req: Request receiving a demoted session prefix.
        :param matched_len: Exact full-KV prefix accepted by the cache.
        """
        if req is None or not _is_streaming(req):
            return
        state = self.demoted.get(req.session.session_id)
        if state is None:
            return
        if matched_len != state.cache_protected_len:
            raise AssertionError(
                "A demoted streaming session did not restore its exact frontier: "
                f"{matched_len=} expected={state.cache_protected_len}"
            )
        current_watermark = req.kv.swa_evicted_seqlen
        if current_watermark not in (0, state.swa_evicted_seqlen):
            raise AssertionError(
                "A demoted streaming request already has an unrelated SWA "
                f"watermark: {current_watermark=}"
            )
        req.kv.swa_evicted_seqlen = state.swa_evicted_seqlen
        req.streaming_session_tree_protected_len = state.cache_protected_len

    # -- Try-handle entries for composition (see class docstring) --

    def try_inc_lock_ref(self, node: Any) -> Optional[IncLockRefResult]:
        """No-op lock if ``node`` is a session-internal sentinel; returns
        None to tell the caller to run its raw tree lock path."""
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return None

    def try_dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> Optional[DecLockRefResult]:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return None

    def find_active_slot(self, req: Req) -> Optional[SessionSlot]:
        """Returns an active slot for this req, or None.

        Side effect: if req is pre-aborted (to_finish set, e.g. input too
        long), detach it from the session so cache_finished_req treats it
        as a normal req. The slot stays intact for the next request.
        """
        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        if slot is None or not slot.is_holding_kv:
            return None
        if req.to_finish is not None:
            if req.streaming_session_owns_inflight:
                req.session.abort_req(req)
            req.session = None
            return None
        return slot

    # -- BasePrefixCache abstract methods --

    def reset(self):
        self.slots.clear()
        self.demoted.clear()
        self.inner.reset()

    # -- Streaming entries: contract with embedded composers (e.g.
    # UnifiedRadixCache) is a uniform "try_handle_*" pattern. Each method
    # executes the streaming body if applicable and signals whether the
    # caller still needs to run its raw path.

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        """Returns a MatchResult iff the request hits an active session slot;
        otherwise None (caller falls back to its raw match)."""
        slot = self.find_active_slot(params.req)
        if slot is None:
            return None

        req = params.req

        # [NPU] When aligned context < page_size, release the slot's KV and
        # fall back to radix cache (full prefill). Once context >= page_size,
        # streaming session kicks in with page-aligned KV reuse.
        if is_npu() and self.page_size > 1:
            expected_prefix_len = min(slot.kv_committed_len, len(params.key))
            aligned_prefix_len = (
                expected_prefix_len // self.page_size
            ) * self.page_size
            if aligned_prefix_len < slot.cache_protected_len or aligned_prefix_len == 0:
                # Release KV to avoid leak and fallback to full prefill.
                # req remains unassigned, so alloc_for_extend treats it as new.
                self.release_session(req.session.session_id)
                return None

        slot.restore_to_req(req)

        # token_ids = get_fill_ids()[:input_len-1] (1-token logit reserve
        # already applied). min handles retract retry where committed_len
        # can exceed len(token_ids) by 1.
        prefix_len = min(
            req.kv_committed_len,
            req.kv.kv_allocated_len,
            len(params.key),
        )

        assert (
            0
            <= slot.cache_protected_len
            <= slot.kv_committed_len
            <= slot.kv.kv_allocated_len
        ), (
            "streaming session slot cursors are inconsistent: "
            f"{slot.cache_protected_len=}, {slot.kv_committed_len=}, "
            f"kv_allocated_len={slot.kv.kv_allocated_len}"
        )
        assert prefix_len >= slot.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"cache_protected_len={slot.cache_protected_len}"
        )

        # Floor-align prefix_len to page boundary (NPU workaround).
        if is_npu() and self.page_size > 1:
            prefix_len = (prefix_len // self.page_size) * self.page_size
            req.kv_committed_len = min(req.kv_committed_len, prefix_len)
            slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)

        # Free orphaned tail: alloc_for_extend will overwrite
        # req_to_token[prefix_len:] with new indices. The range
        # [prefix_len, kv_allocated_len) has stale indices from the
        # previous turn's decode (e.g. alloc-commit gap on retract,
        # or speculative draft tokens).
        self._free_tail(slot, req, prefix_len)

        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def try_cache_finished_req(
        self, req: Req, is_insert: bool = True, **kwargs
    ) -> bool:
        """Handles a streaming-session finish (save slot / abort rollback).
        Returns True if handled; False means caller runs its raw path."""
        if not _is_streaming(req):
            return False

        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None

        # Mid-processing abort only. Pre-aborted reqs have session=None
        # (set in find_active_slot) and never reach here. Preserve any fully
        # materialized admitted boundary, including a first request or a
        # truncate that retired the previous slot.
        if isinstance(req.finished_reason, FINISH_ABORT):
            if slot is None:
                rollback_len = len(req.origin_input_ids)
                can_preserve_preburst = (
                    req.streaming_session_preburst_mutation
                    and req.kv_committed_len >= rollback_len
                    and req.kv.kv_allocated_len >= rollback_len
                )
                if can_preserve_preburst:
                    slot = SessionSlot(
                        req_pool_idx=req.req_pool_idx,
                        kv_committed_len=req.kv_committed_len,
                        kv=copy.copy(req.kv),
                    )
                    self.slots[session_id] = slot
                    self._free_tail(slot, req, rollback_len)
                    slot.save_from_req(req, is_first=True)
                    slot.kv_committed_len = rollback_len
                    self._release_demoted_state(session_id)
                    req.session.abort_req(req)
                    req.time_stats.increment_streaming_session_abort_with_slot_preserved()
                    return True

                # No complete pre-burst KV boundary exists. Create an
                # ephemeral slot so release_session handles cleanup.
                # Include last_node/cache_protected_len from the req so
                # release_session calls dec_lock_ref on the tree lock.
                # Also carry the mamba refs over so _free_slot_mamba can
                # return the (possibly extra_buffer ping-pong) slots to
                # the mamba pool; otherwise the abort orphans them.
                slot = SessionSlot(
                    req_pool_idx=req.req_pool_idx,
                    kv=copy.copy(req.kv),
                    last_node=req.last_node,
                    cache_protected_len=req.cache_protected_len,
                    swa_uuid_for_lock=req.swa_uuid_for_lock,
                    skip_lock_node_ids=req.skip_lock_node_ids,
                    mamba_pool_idx=req.mamba_pool_idx,
                    mamba_ping_pong_track_buffer=req.mamba_ping_pong_track_buffer,
                )
                self.slots[session_id] = slot
                # Slot now owns the mamba state — drop the req's refs so
                # the abort fall-through doesn't double-free.
                req.mamba_pool_idx = None
                req.mamba_ping_pong_track_buffer = None
                slot.kv.kv_allocated_len = max(
                    slot.kv.kv_allocated_len, req.kv.kv_allocated_len
                )
                self.release_session(session_id)
                req.req_pool_idx = None
                req.kv = ReqKvInfo()
                req.session.abort_req(req)
                return True

            rollback_len = slot.kv_committed_len
            durable_mutation = req.streaming_session_preburst_mutation
            if durable_mutation:
                rollback_len = len(req.origin_input_ids)

            if durable_mutation:
                if (
                    req.kv_committed_len < rollback_len
                    or req.kv.kv_allocated_len < rollback_len
                ):
                    slot.kv.kv_allocated_len = max(
                        slot.kv.kv_allocated_len, req.kv.kv_allocated_len
                    )
                    self.release_session(session_id)
                    req.req_pool_idx = None
                    req.kv = ReqKvInfo()
                    req.mamba_pool_idx = None
                    req.mamba_ping_pong_track_buffer = None
                    req.session.abort_req(req)
                    return True

            slot.kv.kv_allocated_len = max(
                slot.kv.kv_allocated_len, req.kv.kv_allocated_len
            )
            self._free_tail(slot, req, rollback_len)
            slot.save_from_req(req, is_first=False)
            slot.kv_committed_len = rollback_len
            req.session.abort_req(req)
            req.time_stats.increment_streaming_session_abort_with_slot_preserved()
            return True

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        req.session.finish_req(req)
        slot.save_from_req(req, is_first=is_first)
        # Inherit the authoritative finished length on the slot, not the lagging
        # req clock (under overlap + honest committed the clock lags the in-flight
        # verify by ~1, which would short-change inheritance). Clamp to allocated
        # to keep committed <= allocated for prepare_for_decode.
        slot.kv_committed_len = min(target, slot.kv.kv_allocated_len)
        if not self._release_demoted_state(session_id):
            self._reconcile_slot_swa(slot)

        return True

    def try_cache_unfinished_req(
        self, req: Req, chunked: bool = False, **kwargs
    ) -> bool:
        """Handles a streaming-session mid-flight cache op:
          - chunked prefill: snapshot current KV as prefix, skip radix
          - subsequent turn: skip radix (slot already holds KV)
        Returns False for first-turn non-chunked (caller must run raw radix
        insert to set up the initial tree lock)."""
        if not _is_streaming(req):
            return False
        if chunked:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return True
        # A reloaded demoted session has no slot until it finishes. Keep its
        # private suffix out of ordinary radix publication until adoption.
        session_id = req.session.session_id
        if session_id in self.slots or session_id in self.demoted:
            return True
        return False

    # -- BasePrefixCache abstract methods: thin adapters over try_handle_* --

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.try_match_prefix(params)
        if result is not None:
            return result
        return self.inner.match_prefix(params)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if self.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return
        self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

    def cache_unfinished_req(self, req: Req, **kwargs):
        if self.try_cache_unfinished_req(req, **kwargs):
            return
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        return self.inner.evict_for_alloc(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.try_inc_lock_ref(node)
        if result is not None:
            return result
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        result = self.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        return self.inner.dec_lock_ref(node, params)

    # -- Session lifecycle --

    def release_session(self, session_id: str) -> None:
        slot = self.slots.get(session_id)
        self._release_demoted_state(session_id, retire_private_path=slot is None)
        slot = self.slots.pop(session_id, None)
        if slot is not None:
            self._release_slot_resources(
                session_id,
                slot,
                free_start=slot.tree_protected_len,
            )

    def transition_to_demoted(
        self,
        session_id: str,
        last_node: Any,
        cache_protected_len: int,
        tree_prefix_len: int,
        swa_evicted_seqlen: int,
        host_lock_params: DecLockRefParams,
    ) -> None:
        """Replace a detached device slot with its host-resident tree frontier.

        :param session_id: Session identifier to transition.
        :param last_node: Published radix frontier.
        :param cache_protected_len: Exact host-backed token count.
        :param tree_prefix_len: Page-aligned prefix already owned by the ordinary
            radix tree.
        :param swa_evicted_seqlen: First token retained in the restored SWA
            window.
        :param host_lock_params: Component lock coordinates required for release.
        """
        slot = self.slots.pop(session_id)
        assert session_id not in self.demoted
        assert tree_prefix_len % self.page_size == 0
        assert tree_prefix_len <= cache_protected_len
        assert cache_protected_len <= slot.kv_committed_len
        assert swa_evicted_seqlen % self.page_size == 0
        assert swa_evicted_seqlen <= cache_protected_len
        self.demoted[session_id] = DemotedSessionState(
            last_node=last_node,
            cache_protected_len=cache_protected_len,
            swa_evicted_seqlen=swa_evicted_seqlen,
            host_lock_params=host_lock_params,
        )
        self._release_slot_resources(
            session_id,
            slot,
            free_start=tree_prefix_len,
        )

    def _release_slot_resources(
        self,
        session_id: str,
        slot: SessionSlot,
        *,
        free_start: int,
    ) -> None:
        """Release one detached request row and the device KV it still owns.

        :param session_id: Session identifier used for lifecycle logging.
        :param slot: Detached slot whose ownership must be retired.
        :param free_start: First device slot not transferred to radix ownership.
        """
        lock_node = slot.last_node
        tokens_freed = (
            max(0, slot.kv.kv_allocated_len - free_start) if slot.is_holding_kv else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        if lock_node is not None:
            self.inner.dec_lock_ref(
                lock_node,
                DecLockRefParams(
                    swa_uuid_for_lock=slot.swa_uuid_for_lock,
                    skip_lock_node_ids=slot.skip_lock_node_ids,
                ),
            )
            self.inner.retire_streaming_session_private_path(session_id, lock_node)

        if slot.is_holding_kv:
            start = free_start
            end = slot.kv.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.release_detached_request_slot(slot.req_pool_idx)

        self._free_slot_mamba(slot)

    def _release_demoted_state(
        self, session_id: str, *, retire_private_path: bool = False
    ) -> bool:
        """Release one session's host lock and radix coverage.

        :param session_id: Session identifier whose host ownership must end.
        :param retire_private_path: Whether no device slot inherits the suffix.
        :returns: Whether a demoted state was released.
        """
        state = self.demoted.pop(session_id, None)
        if state is None:
            return False
        self.inner.dec_host_lock_ref(state.last_node, state.host_lock_params)
        self.inner.clear_radix_session_refs(session_id)
        if retire_private_path:
            self.inner.retire_streaming_session_private_path(
                session_id, state.last_node
            )
            return True
        slot = self.slots.get(session_id)
        private_parent = self.inner.streaming_session_private_parent(state.last_node)
        replacement_lock: IncLockRefResult | None = None
        if slot is not None and private_parent is not None:
            replacement_lock = self.inner.inc_lock_ref(private_parent)
            self.inner.dec_lock_ref(
                state.last_node,
                DecLockRefParams(
                    swa_uuid_for_lock=slot.swa_uuid_for_lock,
                    skip_lock_node_ids=slot.skip_lock_node_ids,
                ),
            )
        private_len = self.inner.adopt_streaming_session_private_path(
            session_id, state.last_node
        )
        if slot is not None and private_len > 0:
            if private_parent is None or replacement_lock is None:
                raise AssertionError(
                    "A restored private suffix has no ordinary lock frontier."
                )
            slot.last_node = private_parent
            slot.swa_uuid_for_lock = replacement_lock.swa_uuid_for_lock
            slot.skip_lock_node_ids = replacement_lock.skip_lock_node_ids
            slot.tree_protected_len = state.cache_protected_len - private_len
        if slot is not None:
            self._reconcile_adopted_swa(slot, state)
            self._reconcile_slot_swa(slot)
        return True

    def _reconcile_adopted_swa(
        self,
        slot: SessionSlot,
        state: DemotedSessionState,
    ) -> None:
        """Release restored SWA pages skipped while the private path was locked.

        :param slot: Device slot inheriting the private suffix.
        :param state: Host-resident ownership state being adopted.
        """
        if not slot.is_holding_kv or not self.supports_swa():
            return
        restored_watermark = state.swa_evicted_seqlen
        logical_watermark = slot.kv.swa_evicted_seqlen
        if logical_watermark < restored_watermark:
            raise AssertionError(
                "Restored SWA watermark regressed during private-path adoption: "
                f"{logical_watermark=} {restored_watermark=}"
            )
        private_end = ceil_align(state.cache_protected_len, self.page_size)
        tree_end = ceil_align(slot.tree_protected_len, self.page_size)
        if tree_end > private_end:
            raise AssertionError(
                "Post-adoption tree ownership exceeds the restored private prefix: "
                f"{tree_end=} {private_end=}"
            )
        free_start = max(restored_watermark, tree_end)
        free_end = min(logical_watermark, private_end)
        free_mapped_swa_slots(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            slot.req_pool_idx,
            free_start,
            free_end,
        )

    def truncate_session(self, session_id: str, target: int) -> None:
        """Trim a session slot to an already-validated logical token index."""
        slot = self.slots.get(session_id)
        if slot is None or not slot.is_holding_kv:
            assert target == 0
            return

        assert target >= 0
        if target >= slot.kv_committed_len:
            return

        old_protected_len = slot.cache_protected_len
        if self.supports_swa():
            retention = max(self.sliding_window_size, self.page_size)
            latest_safe_watermark = max(0, target - retention)
            if self.page_size > 1:
                latest_safe_watermark = (
                    latest_safe_watermark // self.page_size
                ) * self.page_size
            assert slot.kv.swa_evicted_seqlen <= latest_safe_watermark, (
                "streaming-session SWA pin invariant violated: required rollback "
                f"KV was already evicted ({slot.kv.swa_evicted_seqlen=} > "
                f"{latest_safe_watermark=})"
            )
        retained_len = target
        if target <= old_protected_len:
            # Prefix matching reserves the final logical token for the next
            # forward. Retain only complete tree-owned pages strictly before
            # that token so alloc_for_extend never overwrites shared KV.
            retained_len = max(0, target - 1)
            if self.page_size > 1:
                retained_len = (retained_len // self.page_size) * self.page_size

        # With no complete page to inherit, the cache slot has no reusable
        # ownership. Retire it through the ordinary lifecycle so its request
        # row, session-owned tail, auxiliary state, and tree lock are each
        # released exactly once. The logical Session remains open and this
        # request will rebuild its truncated context through the raw cache path.
        if retained_len == 0:
            self.release_session(session_id)
            return

        free_start = max(target, old_protected_len)
        self._free_kv_aligned(slot.req_pool_idx, free_start, slot.kv.kv_allocated_len)
        slot.cache_protected_len = min(old_protected_len, retained_len)

        slot.kv.kv_allocated_len = min(slot.kv.kv_allocated_len, retained_len)
        slot.kv_committed_len = retained_len
        slot.kv.swa_evicted_seqlen = min(slot.kv.swa_evicted_seqlen, retained_len)

    def commit_session(self, session_id: str, floor: int) -> None:
        """Advance a detached slot's rollback floor and release obsolete SWA."""
        slot = self.slots.get(session_id)
        if slot is None:
            return

        assert floor >= slot.streaming_session_floor
        slot.streaming_session_floor = floor
        self._reconcile_slot_swa(slot)

    def _reconcile_slot_swa(self, slot: SessionSlot) -> None:
        """Apply the active-request SWA frontier to a detached session slot."""
        if not slot.is_holding_kv or not self.supports_swa():
            return

        new_watermark, physical_free_start = streaming_session_swa_eviction_plan(
            slot.kv.swa_evicted_seqlen,
            slot.kv_committed_len,
            tree_protected_len=slot.tree_protected_len,
            sliding_window_size=self.sliding_window_size,
            page_size=self.page_size,
            streaming_session_floor=slot.streaming_session_floor,
        )
        if new_watermark <= slot.kv.swa_evicted_seqlen:
            return

        if new_watermark > physical_free_start:
            free_mapped_swa_slots(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                slot.req_pool_idx,
                physical_free_start,
                new_watermark,
            )
        slot.kv.swa_evicted_seqlen = new_watermark

    def release_radix_session(self, session_id: str) -> None:
        self.inner.release_radix_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total KV tokens held by session slots, not tracked by the tree.

        Excludes slots whose KV is currently owned by an owning request --
        those tokens are counted via uncached_size in the busy mem check.
        A slot's pool_idx being in active_pool_idxs indicates a req owns it.
        """
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
                total += max(0, allocated - slot.tree_protected_len)
        return total

    def streaming_session_cache_snapshot(
        self, session_id: str
    ) -> StreamingSessionCacheSnapshot:
        """Return durable cache ownership for one streaming session.

        :param session_id: Session identifier to inspect.
        :returns: The session's tree-protected and exclusively held token counts.
        """
        slot = self.slots.get(session_id)
        if slot is None:
            demoted = self.demoted.get(session_id)
            if demoted is None:
                return StreamingSessionCacheSnapshot()
            full, swa = self.inner.streaming_session_protected_residency(
                demoted.last_node
            )
            return StreamingSessionCacheSnapshot(
                protected=demoted.cache_protected_len,
                full=full,
                swa=swa,
            )
        full, swa = self.inner.streaming_session_protected_residency(slot.last_node)
        if not slot.is_holding_kv:
            return StreamingSessionCacheSnapshot(
                protected=slot.cache_protected_len,
                full=full,
                swa=swa,
            )

        allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
        full_held = max(0, allocated - slot.tree_protected_len)
        full_device_pages = full_held // self.page_size
        swa_device_pages = 0
        if self.supports_swa():
            swa_start = max(
                slot.tree_protected_len,
                slot.kv.swa_evicted_seqlen,
            )
            swa_held = max(0, allocated - swa_start)
            swa_device_pages = swa_held // self.page_size
        return StreamingSessionCacheSnapshot(
            protected=slot.cache_protected_len,
            held_tokens=full_held,
            full=KVComponentResidency(
                device_pages=full.device_pages + full_device_pages,
                host_backed_pages=full.host_backed_pages,
            ),
            swa=KVComponentResidency(
                device_pages=swa.device_pages + swa_device_pages,
                host_backed_pages=swa.host_backed_pages,
            ),
        )

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """An alias to align the naming style of SWA"""
        return self.session_held_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
                total += max(
                    0,
                    allocated
                    - max(slot.tree_protected_len, slot.kv.swa_evicted_seqlen),
                )
        return total

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        """Number of req pool slots held by session slots."""

        def _owned(s):
            in_batch = (
                active_pool_idxs is not None and s.req_pool_idx in active_pool_idxs
            )
            return s.is_holding_kv and not in_batch

        return sum(_owned(s) for s in self.slots.values())

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total mamba_pool entries held by session slots (mamba_pool_idx +
        mamba_ping_pong_track_buffer). Excludes slots whose owning req is
        currently in the batch -- those slots are counted via the normal
        alloc/free paths (same convention as the sibling ``session_held_*``
        accessors).
        """
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if in_batch:
                continue
            if slot.mamba_pool_idx is not None:
                total += slot.mamba_pool_idx.numel()
            if slot.mamba_ping_pong_track_buffer is not None:
                total += slot.mamba_ping_pong_track_buffer.numel()
        return total

    def _free_slot_mamba(self, slot: SessionSlot) -> None:
        """Return a session slot's mamba pool state to the allocator."""
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            return
        if slot.mamba_pool_idx is not None:
            mamba_allocator.free(slot.mamba_pool_idx.unsqueeze(0))
            slot.mamba_pool_idx = None
        if slot.mamba_ping_pong_track_buffer is not None:
            mamba_allocator.free(slot.mamba_ping_pong_track_buffer)
            slot.mamba_ping_pong_track_buffer = None

    # -- Internal helpers (streaming body bits) --

    def _free_tail(self, slot: SessionSlot, req: Req, prefix_len: int) -> None:
        """match_prefix path: free orphaned KV in [prefix_len, kv_allocated_len)
        before alloc_for_extend overwrites it. The gap appears when spec
        decoding pushes allocated above committed, or when retract retry's
        logit-reserve pulls prefix_len below committed.
        """
        self._free_kv_aligned(slot.req_pool_idx, prefix_len, slot.kv.kv_allocated_len)
        slot.kv.kv_allocated_len = prefix_len
        slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)
        slot.kv.swa_evicted_seqlen = min(slot.kv.swa_evicted_seqlen, prefix_len)
        req.kv.kv_allocated_len = prefix_len
        req.kv_committed_len = min(req.kv_committed_len, prefix_len)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        """Trim slot KV to finished_len boundary. Spec v2 may overshoot
        max_new_tokens (verify round commits M+1 at a time); next turn's
        input is output_ids[:finished_len], so positions past that must
        be released to avoid token/KV mismatch.
        """
        target = len(req.origin_input_ids) + finished_len
        self._free_kv_aligned(req.req_pool_idx, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv_committed_len = min(req.kv_committed_len, target)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, pool_idx: int, target: int, end: int) -> None:
        """Free req_to_token[pool_idx, ceil_align(target):end). Page-aligned
        because PagedTokenToKVPoolAllocator.free returns whole pages
        (free_index // page_size), so partial-page free would corrupt pages
        still holding committed tokens. The range [target, ceil_align(target))
        stays attached until release_session frees the whole page.
        """
        if end <= target:
            return
        start = target
        if self.page_size > 1:
            start = ceil_align(start, self.page_size)
        if start < end:
            tail = self.req_to_token_pool.req_to_token[pool_idx, start:end]
            self.token_to_kv_pool_allocator.free(tail)

    # -- Pass-through methods --

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def protected_size(self):
        return self.inner.protected_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        return self.inner.pretty_print()

    def init_load_back(self, params: InitLoadBackParams) -> LoadBackResult:
        return self.inner.init_load_back(params)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def supports_streaming_session(self) -> bool:
        return True

    def supports_streaming_session_demotion(self) -> bool:
        """Return whether the composed cache supports host demotion.

        :returns: Whether transactional host demotion is available.
        """
        return self.inner.supports_streaming_session_demotion()

    def is_streaming_session_demoted(self, session_id: str) -> bool:
        """Return whether one session owns a host frontier.

        :param session_id: Session identifier to inspect.
        :returns: Whether the session is host-resident.
        """
        return self.is_demoted(session_id)

    def prepare_streaming_session_demotion(
        self,
        session_id: str,
        token_ids: Sequence[int],
        extra_key: str | None,
        cache_salt: str | None,
        priority: int,
    ) -> int | None:
        """Delegate private host staging to the composed cache.

        :param session_id: Session identifier to stage.
        :param token_ids: Complete committed token lineage.
        :param extra_key: Radix cache classification key.
        :param cache_salt: Radix cache namespace salt.
        :param priority: Eviction priority inherited from the session.
        :returns: Exact staged token count, or ``None`` on rejection.
        """
        return self.inner.prepare_streaming_session_demotion(
            session_id,
            token_ids,
            extra_key,
            cache_salt,
            priority,
        )

    def discard_streaming_session_demotion(self, session_id: str) -> None:
        """Delegate private-stage discard to the composed cache.

        :param session_id: Session identifier whose stage must be discarded.
        """
        self.inner.discard_streaming_session_demotion(session_id)

    def commit_streaming_session_demotion(self, session_id: str) -> int:
        """Delegate unanimous host-stage publication to the composed cache.

        :param session_id: Session identifier whose stage must be committed.
        :returns: Exact host-backed token count.
        """
        return self.inner.commit_streaming_session_demotion(session_id)

    def clear_radix_session_refs(self, session_id: str) -> int:
        """Release tagged cache coverage without closing the generation.

        :param session_id: Session identifier whose coverage must be released.
        :returns: Number of component frontier tags removed.
        """
        return self.inner.clear_radix_session_refs(session_id)

    def retire_streaming_session_private_path(self, session_id: str, node: Any) -> None:
        """Delegate retirement of one private exact suffix.

        :param session_id: Session identifier that owns the private suffix.
        :param node: Exact host frontier, or an ordinary radix frontier.
        """
        self.inner.retire_streaming_session_private_path(session_id, node)

    def streaming_session_private_parent(self, node: Any) -> Any | None:
        """Delegate resolution of a private suffix's ordinary parent.

        :param node: Exact host frontier, or an ordinary aligned frontier.
        :returns: The ordinary parent, or ``None`` without a private suffix.
        """
        return self.inner.streaming_session_private_parent(node)

    def adopt_streaming_session_private_path(self, session_id: str, node: Any) -> int:
        """Delegate restored private-suffix ownership to the device slot.

        :param session_id: Session identifier that owns the private suffix.
        :param node: Exact restored frontier.
        :returns: Logical private-suffix length, or zero without one.
        """
        return self.inner.adopt_streaming_session_private_path(session_id, node)

    def is_chunk_cache(self):
        return self.inner.is_chunk_cache()

    def is_tree_cache(self):
        return self.inner.is_tree_cache()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        self.inner.sanity_check()

    # Forward attribute access for cache-specific methods (e.g.
    # sliding_window_size, all_values_flatten, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)
