"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeRestoreBudget:
    """Physical device-token demand held by pending local restores.

    :ivar full_tokens: Page-rounded full-attention device tokens.
    :ivar swa_tokens: Page-rounded sliding-window device tokens.
    """

    full_tokens: int = 0
    swa_tokens: int = 0


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    page_size: int = 1
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def full_restore_token_count(self) -> int:
        """Return page-rounded full-attention device allocation demand.

        :returns: Physical full-attention tokens needed by load-back.
        """

        token_count = self.decode_prefix_len - self.l1_prefix_len
        return (token_count + self.page_size - 1) // self.page_size * self.page_size

    @property
    def swa_restore_token_count(self) -> int:
        """Return page-rounded sliding-window device allocation demand.

        :returns: Physical SWA tokens needed by load-back.
        """

        return (
            (self.swa_host_hit_length + self.page_size - 1)
            // self.page_size
            * self.page_size
        )

    @property
    def needs_local_restore(self) -> bool:
        """Return whether any rank-local cache component needs load-back.

        :returns: Whether local restore must gate transport completion.
        """

        return (
            self.full_restore_token_count > 0
            or self.swa_restore_token_count > 0
            or self.mamba_host_hit_length > 0
        )


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side HiCache
        storage is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        if self.scheduler.enable_decode_hicache and self.tree_cache.enable_storage:
            last_host_node = self.tree_cache.resolve_node_handle(result.last_host_node)
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            swa_host_hit_length=result.swa_host_hit_length,
            mamba_host_hit_length=result.mamba_host_hit_length,
            page_size=self.token_to_kv_pool_allocator.page_size,
            last_host_node=(
                result.last_host_node if l3_storage_hit_length > 0 else None
            ),
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch before exact restore/delta admission.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = self.tree_cache.resolve_node_handle(prefix_match.last_host_node)
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid, prefix_match.last_host_node, suffix, last_hash, prefix_keys
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_budgets(self) -> DecodeRestoreBudget:
        """Return physical full and SWA demand not reflected by allocators yet.

        A restore disappears from this reservation once it owns a restored
        node, because its component allocations are then already reflected by
        the physical allocator availability.

        :returns: Pending rank-local restore demand by physical component.
        """

        if not self.scheduler.enable_decode_hicache:
            return DecodeRestoreBudget()

        pending_matches: list[DecodePrefixMatch] = []
        for decode_req in self.transfer_queue.queue:
            prefix_match = decode_req.prefix_match
            if prefix_match is None:
                continue
            if decode_req.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            if decode_req.hicache_restored_node is not None:
                continue
            pending_matches.append(prefix_match)
        pending_matches.extend(self._unpublished_preallocated_prefix_matches())

        return DecodeRestoreBudget(
            full_tokens=sum(
                prefix_match.full_restore_token_count
                for prefix_match in pending_matches
            ),
            swa_tokens=sum(
                prefix_match.swa_restore_token_count for prefix_match in pending_matches
            ),
        )

    def _abort_preallocated_hicache_prefetch(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Release storage-prefetch ownership before reservation rollback.

        :param decode_req: Unpublished prepared request being cancelled.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.prefetch_registered:
            return
        prefix_match.prefetch_registered = False
        self.tree_cache.release_aborted_request(decode_req.req.rid)


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is not None and prefix_match.prefetch_registered:
            prefix_match.prefetch_registered = False
            self.tree_cache.release_aborted_request(decode_req.req.rid)

        restored_node = decode_req.hicache_restored_node
        decode_req.prefix_match = None
        decode_req.hicache_restored_node = None
        decode_req.hicache_restored_kv_indices = None
        decode_req.hicache_load_consumer_index = -1
        if restored_node is not None:
            self.tree_cache.dec_lock_ref(restored_node)

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Prepare one local restore and report whether async work was queued.

        A successful result acquires one request-owned lock on the restored
        node. Synchronous restoration and a concurrent restoration by another
        request become ready immediately. Incomplete or contradictory results
        fail closed.

        :param dr: Decode request whose admitted local prefix must be restored.
        :returns: Whether the cache queued an asynchronous component transfer.
        """
        pm = dr.prefix_match
        if pm is None:
            raise RuntimeError("HiCache restore request has no prefix match")

        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)
            pm.prefetch_registered = False

        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        try:
            load_back_result = self.tree_cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=rematch.best_match_node,
                    host_hit_length=rematch.host_hit_length,
                    req=dr.req,
                )
            )
        finally:
            # ``match_prefix_for_req`` temporarily publishes the rematch on the
            # request so hybrid components can prepare their transfer. Until
            # commit, the request still owns the original matched-node lock.
            dr.req.prefix_indices = pm.prefix_indices
            dr.req.last_node = pm.last_device_node

        original_full_tokens = pm.l1_prefix_len
        promised_full_tokens = pm.decode_prefix_len
        if len(rematch.device_indices) < original_full_tokens:
            logger.error(
                "HiCache rematch lost locked full-KV coverage for rid=%s: "
                "rematched=%d, locked=%d",
                dr.req.rid,
                len(rematch.device_indices),
                original_full_tokens,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        concurrently_restored_indices = rematch.device_indices[
            original_full_tokens:promised_full_tokens
        ]
        full_tokens_needed = promised_full_tokens - original_full_tokens
        new_full_indices = load_back_result.new_full_device_indices
        restored_full_indices = torch.cat(
            [concurrently_restored_indices, new_full_indices]
        )[:full_tokens_needed]

        swa_tokens_needed = (
            (rematch.swa_host_hit_length + pm.page_size - 1)
            // pm.page_size
            * pm.page_size
        )
        full_restore_complete = len(restored_full_indices) == full_tokens_needed
        swa_restore_complete = (
            swa_tokens_needed == 0 or load_back_result.swa_tokens >= swa_tokens_needed
        )
        mamba_restore_complete = (
            rematch.mamba_host_hit_length == 0 or load_back_result.queued_any_component
        )
        if not (
            full_restore_complete and swa_restore_complete and mamba_restore_complete
        ):
            logger.warning(
                "HiCache load_back did not restore every admitted component for "
                "rid=%s: full=%d/%d, swa=%d/%d, mamba_host=%d, queued=%s",
                dr.req.rid,
                len(restored_full_indices),
                full_tokens_needed,
                load_back_result.swa_tokens,
                swa_tokens_needed,
                rematch.mamba_host_hit_length,
                load_back_result.queued_any_component,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        restored_node = load_back_result.restored_node
        if restored_node is None:
            logger.error(
                "HiCache load_back returned no restored node for rid=%s", dr.req.rid
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = restored_full_indices
        dr.hicache_restored_node = restored_node
        dr.hicache_load_consumer_index = -1
        self.tree_cache.inc_lock_ref(restored_node)

        if load_back_result.queued_any_component:
            return True

        dr.hicache_restore_status = HiCacheRestoreResult.READY
        return False

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and dr.hicache_load_consumer_index >= 0
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                dr.hicache_load_consumer_index = -1

        to_prepare = [
            dr
            for dr in active
            if dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        ]
        if len(to_prepare) == 0:
            return

        counter = self.tree_cache.cache_controller.layer_done_counter
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [dr for dr in to_prepare if self._try_hicache_queue_load_back(dr)]
        if len(queued) == 0:
            return

        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return
        if decode_req.hicache_restore_status != HiCacheRestoreResult.READY:
            raise RuntimeError("HiCache restore commit preceded local readiness")

        restored_node = decode_req.hicache_restored_node
        restored_indices = decode_req.hicache_restored_kv_indices
        if restored_node is None or restored_indices is None:
            raise RuntimeError("HiCache restore commit has no restored ownership")
        if prefix_match.prefetch_registered:
            raise RuntimeError("HiCache restore commit retained a storage prefetch")

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            restored_indices,
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, restored_indices]
        )
        decode_req.req.last_node = restored_node

        decode_req.prefix_match = None
        decode_req.hicache_restored_node = None
        decode_req.hicache_restored_kv_indices = None
        decode_req.hicache_load_consumer_index = -1
        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)
