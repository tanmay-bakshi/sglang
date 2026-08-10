"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InitLoadBackParams,
    LoadBackTicket,
    LoadBackTicketState,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeRestoreBudget:
    """Physical device-token demand held by pending local restores.

    :ivar full_tokens: Page-rounded full-attention device tokens.
    :ivar swa_tokens: Page-rounded sliding-window device tokens.
    :ivar mamba_slots: Recurrent-state device slots.
    """

    full_tokens: int = 0
    swa_tokens: int = 0
    mamba_slots: int = 0


@dataclass
class DecodePrefixMatch:
    """Own one decoder prefix match until commit or rollback.

    :ivar prefix_indices: Full-KV indices already resident on the decoder.
    :ivar l2_host_hit_length: Full-KV tokens resident in decode-local host memory.
    :ivar l3_storage_hit_length: Full-KV tokens resident in external storage.
    :ivar last_device_node: Deepest device-resident cache node.
    :ivar last_host_node: Deepest host-resident cache node.
    :ivar swa_host_hit_length: SWA tokens resident in host memory.
    :ivar mamba_host_hit_length: Recurrent-state host hit length.
    :ivar page_size: Physical cache allocation page size.
    :ivar last_device_lock_params: Exact release receipt for the device node.
    :ivar prefetch_registered: Whether external storage owns a prefetch lease.
    """

    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    page_size: int = 1
    last_device_lock_params: DecLockRefParams | None = None
    prefetch_registered: bool = False

    @property
    def restore_budget(self) -> DecodeRestoreBudget:
        """Return component-local physical demand for this restore.

        A host-only recurrent state needs one cache-tree slot and one
        request-owned copy-on-write slot.

        :returns: Full, SWA, and recurrent-state device demand.
        """

        return DecodeRestoreBudget(
            full_tokens=self.full_restore_token_count,
            swa_tokens=self.swa_restore_token_count,
            mamba_slots=2 * int(self.mamba_host_hit_length > 0),
        )

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
        self, req: Req, prefix_match: DecodePrefixMatch | None
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
            if decode_req.hicache_load_back_ticket is not None:
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
            mamba_slots=sum(
                prefix_match.restore_budget.mamba_slots
                for prefix_match in pending_matches
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
    """Drive decode-side local restore tickets through commit or abort."""

    def _release_hicache_prefetch(self, decode_req: DecodeRequest) -> None:
        """Release a registered storage prefetch exactly once.

        :param decode_req: Request owning the registration.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.prefetch_registered:
            return
        self.tree_cache.release_aborted_request(decode_req.req.rid)
        prefix_match.prefetch_registered = False

    def _abort_hicache_local_restore(
        self,
        decode_req: DecodeRequest,
        *,
        generation_drained: bool = False,
    ) -> None:
        """Abort or reap one ticket and release its request lock once.

        :param decode_req: Request whose local restore cannot commit.
        :param generation_drained: Whether the shared generation was synchronized.
        """

        self._release_hicache_prefetch(decode_req)
        ticket = decode_req.hicache_load_back_ticket
        if ticket is None or ticket.state in (
            LoadBackTicketState.COMMITTED,
            LoadBackTicketState.ABORTED,
        ):
            return

        try:
            if (
                ticket.state
                in (
                    LoadBackTicketState.PREPARED,
                    LoadBackTicketState.PUBLISHED,
                )
                and ticket.publication is not None
            ):
                self.tree_cache.abort_load_back_publication(ticket)
            elif (
                ticket.state == LoadBackTicketState.PREPARED
                and ticket.queued_any_component
            ):
                raise RuntimeError(
                    "Queued HiCache ticket has no reversible publication"
                )

            if ticket.state == LoadBackTicketState.STARTED and not generation_drained:
                finish_event = (
                    self.tree_cache.cache_controller.layer_done_counter.events[
                        ticket.consumer_index
                    ].finish_event
                )
                finish_event.synchronize()
                if not self.tree_cache.is_load_back_event_done(ticket.consumer_index):
                    raise RuntimeError(
                        "Synchronized HiCache abort generation remained pending"
                    )
        finally:
            if ticket.owns_restored_lock:
                self.tree_cache.dec_lock_ref(
                    ticket.restored_node,
                    ticket.restored_lock_params,
                )
                ticket.owns_restored_lock = False
            ticket.state = LoadBackTicketState.ABORTED

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        """Run the single local-restore abort path.

        :param decode_req: Request being removed from the transfer queue.
        """

        self._abort_hicache_local_restore(decode_req)

    @staticmethod
    def _restore_req_match_state(req: Req, state: tuple[Any, ...]) -> None:
        """Restore request fields mutated by a transient rematch.

        :param req: Request to restore.
        :param state: Captured prefix-match fields.
        """

        (
            req.prefix_indices,
            req.last_node,
            req.last_node_lock_params,
            req.last_host_node,
            req.best_match_node,
            req.host_hit_length,
            req.swa_host_hit_length,
            req.mamba_host_hit_length,
            req.num_matched_prefix_tokens,
            req.mamba_branching_seqlen,
            req.cache_protected_len,
            req.swa_uuid_for_lock,
        ) = state

    def _fail_hicache_restore(
        self,
        decode_req: DecodeRequest,
        message: str,
        ticket: LoadBackTicket | None = None,
    ) -> bool:
        """Transition one request to failed and retain its cleanup ticket.

        :param decode_req: Request that failed.
        :param message: Diagnostic reason.
        :param ticket: Ticket created before the failure, if any.
        :returns: False for queueing call sites.
        """

        logger.warning(
            "HiCache load-back failed for rid=%s: %s",
            decode_req.req.rid,
            message,
        )
        if ticket is not None:
            decode_req.hicache_load_back_ticket = ticket
        decode_req.hicache_restore_status = HiCacheRestoreResult.FAILED
        return False

    def _try_hicache_queue_load_back(self, decode_req: DecodeRequest) -> bool:
        """Create one validated local load-back ticket.

        :param decode_req: Request awaiting local restore.
        :returns: Whether a prepared ticket now exists.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None:
            raise RuntimeError("HiCache restore request has no prefix match")

        if prefix_match.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(decode_req.req.rid):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(decode_req.req.rid)
            prefix_match.prefetch_registered = False

        req = decode_req.req
        original_match_state = (
            req.prefix_indices,
            req.last_node,
            req.last_node_lock_params,
            req.last_host_node,
            req.best_match_node,
            req.host_hit_length,
            req.swa_host_hit_length,
            req.mamba_host_hit_length,
            req.num_matched_prefix_tokens,
            req.mamba_branching_seqlen,
            req.cache_protected_len,
            req.swa_uuid_for_lock,
        )
        try:
            rematch = match_prefix_for_req(
                self.tree_cache,
                req,
                req.origin_input_ids[: prefix_match.decode_prefix_len],
                cow_mamba=False,
                include_req=True,
            )
            rematched_full_coverage = (
                len(rematch.device_indices) + rematch.host_hit_length
            )
            if rematched_full_coverage < prefix_match.decode_prefix_len:
                return self._fail_hicache_restore(
                    decode_req,
                    "rematch coverage "
                    f"{rematched_full_coverage} is shorter than promised "
                    f"full prefix {prefix_match.decode_prefix_len}",
                )

            ticket = self.tree_cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=rematch.best_match_node,
                    host_hit_length=rematch.host_hit_length,
                    req=req,
                    defer_publication=True,
                )
            )
            decode_req.hicache_load_back_ticket = ticket
            if (
                rematch.host_hit_length > ticket.full_tokens
                or rematch.swa_host_hit_length > ticket.swa_tokens
                or rematch.mamba_host_hit_length > ticket.mamba_slots
            ):
                return self._fail_hicache_restore(
                    decode_req,
                    "ticket does not cover rematched full/SWA/Mamba host state",
                    ticket,
                )

            restore_budget = prefix_match.restore_budget
            if (
                ticket.full_tokens > restore_budget.full_tokens
                or ticket.swa_tokens > restore_budget.swa_tokens
                or ticket.mamba_device_slots > restore_budget.mamba_slots
            ):
                return self._fail_hicache_restore(
                    decode_req,
                    "ticket exceeds the admitted full/SWA/Mamba restore budget",
                    ticket,
                )

            restore_length = prefix_match.decode_prefix_len - prefix_match.l1_prefix_len
            rematched_suffix = rematch.device_indices[
                prefix_match.l1_prefix_len : prefix_match.decode_prefix_len
            ]
            restored_indices = torch.cat(
                [rematched_suffix, ticket.new_full_device_indices]
            )[:restore_length]
            if len(restored_indices) != restore_length:
                return self._fail_hicache_restore(
                    decode_req,
                    f"restored full suffix has {len(restored_indices)} "
                    f"indices, expected {restore_length}",
                    ticket,
                )

            ticket.restored_full_device_indices = restored_indices
            if ticket.queued_any_component and ticket.publication is None:
                raise RuntimeError(
                    "Decode HiCache load-back was published before TP agreement"
                )
            if ticket.publication is None:
                lock_result = self.tree_cache.inc_lock_ref(ticket.restored_node)
                ticket.restored_lock_params = lock_result.to_dec_params()
                ticket.owns_restored_lock = True
            return True
        finally:
            self._restore_req_match_state(req, original_match_state)

    def _hicache_preparation_inputs_ready(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> bool:
        """Agree that every rank can begin the same preparation round.

        :param decode_reqs: Requests selected for one merged preparation round.
        :returns: Whether every rank has completed required storage prefetches.
        """

        local_ready = all(
            decode_req.prefix_match is not None
            and (
                decode_req.prefix_match.l3_storage_hit_length == 0
                or self.tree_cache.check_prefetch_progress(decode_req.req.rid)
            )
            for decode_req in decode_reqs
        )
        globally_ready = torch.tensor(
            int(local_ready),
            dtype=torch.int32,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                globally_ready,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )
        return bool(globally_ready.item())

    def _agree_hicache_preparation(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> tuple[bool, bool]:
        """Agree on preparation success and queued work across decode TP ranks.

        :param decode_reqs: Requests attempted in one merged preparation round.
        :returns: Global preparation success and whether any rank queued a copy.
        """

        local_prepared_success = all(
            decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
            and decode_req.hicache_load_back_ticket is not None
            for decode_req in decode_reqs
        )
        local_any_queued = any(
            decode_req.hicache_load_back_ticket is not None
            and decode_req.hicache_load_back_ticket.queued_any_component
            for decode_req in decode_reqs
        )
        agreement = torch.tensor(
            [int(local_prepared_success), -int(local_any_queued)],
            dtype=torch.int32,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                agreement,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )
        return bool(agreement[0].item()), bool(-agreement[1].item())

    def _agree_hicache_generation_index(self) -> int | None:
        """Select one controller slot while publications remain reversible.

        :returns: Shared next consumer index, or ``None`` on rank divergence.
        """

        counter = self.tree_cache.cache_controller.layer_done_counter
        expected_consumer_index = (counter.producer_index + 1) % counter.num_counters
        if self.tp_size > 1:
            agreement = torch.tensor(
                [expected_consumer_index, -expected_consumer_index],
                dtype=torch.int32,
                device="cpu",
            )
            torch.distributed.all_reduce(
                agreement,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )
            if agreement[0].item() != -agreement[1].item():
                return None
        return expected_consumer_index

    def _start_hicache_generation(
        self,
        decode_reqs: list[DecodeRequest],
        expected_consumer_index: int,
    ) -> int:
        """Start the controller generation already agreed by every TP rank.

        :param decode_reqs: Requests owning any rank-local prepared tickets.
        :param expected_consumer_index: Prospectively agreed counter slot.
        :returns: Shared layer-done counter index.
        """

        consumer_index = self.tree_cache.ready_to_load_host_cache(force_empty=True)
        for decode_req in decode_reqs:
            ticket = decode_req.hicache_load_back_ticket
            if ticket is None:
                continue
            ticket.consumer_index = consumer_index
            ticket.state = LoadBackTicketState.STARTED
        if consumer_index != expected_consumer_index:
            raise RuntimeError(
                "HiCache controller started an unexpected load generation: "
                f"expected={expected_consumer_index}, actual={consumer_index}"
            )
        return consumer_index

    def _abort_failed_hicache_preparation(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> None:
        """Cancel every reversible allocation in a failed preparation round.

        :param decode_reqs: Requests attempted in the failed preparation round.
        """

        for decode_req in decode_reqs:
            decode_req.hicache_restore_status = HiCacheRestoreResult.FAILED

        cleanup_errors: list[str] = []
        for decode_req in decode_reqs:
            try:
                self._abort_hicache_local_restore(decode_req)
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        if len(cleanup_errors) > 0:
            raise RuntimeError(
                "HiCache preparation rollback failed:\n" + "\n".join(cleanup_errors)
            )

    def _publish_hicache_local_restores(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> bool:
        """Publish every local allocation while retaining rollback ownership.

        :param decode_reqs: Requests prepared in one tensor-parallel round.
        :returns: Whether every local publication succeeded.
        """

        local_success = True
        for decode_req in decode_reqs:
            ticket = decode_req.hicache_load_back_ticket
            if ticket is None or ticket.publication is None:
                continue
            try:
                self.tree_cache.publish_load_back(ticket)
            except Exception:
                local_success = False
                self._fail_hicache_restore(
                    decode_req,
                    f"local publication error:\n{traceback.format_exc()}",
                    ticket,
                )
        return local_success

    def _agree_hicache_publication(
        self,
        decode_reqs: list[DecodeRequest],
        *,
        local_success: bool,
    ) -> bool:
        """Agree that every rank published the same prepared request set.

        :param decode_reqs: Requests prepared in one tensor-parallel round.
        :param local_success: Whether local publication completed without error.
        :returns: Whether publication succeeded on every rank.
        """

        local_published = local_success and all(
            decode_req.hicache_load_back_ticket is not None
            and (
                decode_req.hicache_load_back_ticket.publication is None
                or decode_req.hicache_load_back_ticket.state
                == LoadBackTicketState.PUBLISHED
            )
            for decode_req in decode_reqs
        )
        agreement = torch.tensor(
            int(local_published),
            dtype=torch.int32,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                agreement,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )
        return bool(agreement.item())

    def _commit_hicache_publications(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> None:
        """Finalize globally accepted publications before starting copies.

        :param decode_reqs: Requests published in one tensor-parallel round.
        """

        for decode_req in decode_reqs:
            ticket = decode_req.hicache_load_back_ticket
            if ticket is None or ticket.publication is None:
                continue
            self.tree_cache.commit_load_back_publication(ticket)

    def _poll_hicache_started_generation(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> bool:
        """Advance one started generation only after every TP rank completes it.

        :param decode_reqs: Pending requests that may share a copy generation.
        :returns: Whether a started generation still blocks new preparation.
        """

        started_reqs = [
            decode_req
            for decode_req in decode_reqs
            if decode_req.hicache_load_back_ticket is not None
            and decode_req.hicache_load_back_ticket.state == LoadBackTicketState.STARTED
        ]
        if len(started_reqs) == 0:
            return False

        consumer_indices = {
            decode_req.hicache_load_back_ticket.consumer_index
            for decode_req in started_reqs
        }
        if len(consumer_indices) != 1:
            raise RuntimeError(
                "Pending HiCache requests span multiple load generations"
            )
        consumer_index = next(iter(consumer_indices))
        globally_ready = torch.tensor(
            int(self.tree_cache.is_load_back_event_done(consumer_index)),
            dtype=torch.int32,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                globally_ready,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )
        if not bool(globally_ready.item()):
            return True

        for decode_req in started_reqs:
            decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        return False

    def _process_hicache_local_restores(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> None:
        """Advance identical TP tickets through preparation and readiness.

        :param decode_reqs: Transfer-queue requests in stable rank order.
        """

        active: list[DecodeRequest] = []
        for decode_req in decode_reqs:
            if decode_req.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            prefix_match = decode_req.prefix_match
            if prefix_match is None or not prefix_match.needs_local_restore:
                decode_req.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(decode_req)

        if self._poll_hicache_started_generation(active):
            return

        unprepared = [
            decode_req
            for decode_req in active
            if decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
            and decode_req.hicache_load_back_ticket is None
        ]
        if len(unprepared) == 0:
            return

        counter = self.tree_cache.cache_controller.layer_done_counter
        next_consumer_index = (counter.producer_index + 1) % counter.num_counters
        if not self.tree_cache.is_load_back_event_done(next_consumer_index):
            return
        if not self._hicache_preparation_inputs_ready(unprepared):
            return

        for decode_req in unprepared:
            try:
                self._try_hicache_queue_load_back(decode_req)
            except Exception:
                self._fail_hicache_restore(
                    decode_req,
                    f"unexpected local preparation error:\n{traceback.format_exc()}",
                )

        all_prepared, any_queued = self._agree_hicache_preparation(unprepared)
        if not all_prepared:
            self._abort_failed_hicache_preparation(unprepared)
            return

        if not any_queued:
            for decode_req in unprepared:
                decode_req.hicache_restore_status = HiCacheRestoreResult.READY
            return

        local_published = self._publish_hicache_local_restores(unprepared)
        if not self._agree_hicache_publication(
            unprepared,
            local_success=local_published,
        ):
            self._abort_failed_hicache_preparation(unprepared)
            return

        expected_consumer_index = self._agree_hicache_generation_index()
        if expected_consumer_index is None:
            self._abort_failed_hicache_preparation(unprepared)
            return

        self._commit_hicache_publications(unprepared)
        self._start_hicache_generation(unprepared, expected_consumer_index)

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        """Commit one ready ticket and transfer its tree lock to the request.

        :param decode_req: Request entering the decode batch.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        ticket = decode_req.hicache_load_back_ticket
        if (
            ticket is None
            or decode_req.hicache_restore_status != HiCacheRestoreResult.READY
            or ticket.state
            not in (
                LoadBackTicketState.PREPARED,
                LoadBackTicketState.STARTED,
            )
            or not ticket.owns_restored_lock
            or ticket.restored_full_device_indices is None
            or ticket.restored_lock_params is None
        ):
            raise RuntimeError("Cannot commit an incomplete HiCache load-back ticket")
        if prefix_match.prefetch_registered:
            raise RuntimeError("HiCache restore commit retained a storage prefetch")
        if prefix_match.last_device_lock_params is None:
            raise RuntimeError("HiCache original match has no exact lock ownership")

        restored_indices = ticket.restored_full_device_indices
        committed_prefix_indices = torch.cat(
            [prefix_match.prefix_indices, restored_indices]
        )
        if len(restored_indices) > 0:
            self.tree_cache.req_to_token_pool.write(
                (
                    decode_req.req.req_pool_idx,
                    slice(
                        prefix_match.l1_prefix_len,
                        prefix_match.decode_prefix_len,
                    ),
                ),
                restored_indices,
            )
        self.tree_cache.dec_lock_ref(
            prefix_match.last_device_node,
            prefix_match.last_device_lock_params,
        )
        prefix_match.last_device_lock_params = None
        decode_req.req.prefix_indices = committed_prefix_indices
        decode_req.req.cache_protected_len = len(decode_req.req.prefix_indices)
        decode_req.req.last_node = ticket.restored_node
        decode_req.req.last_node_lock_params = ticket.restored_lock_params
        decode_req.req.swa_uuid_for_lock = ticket.restored_lock_params.swa_uuid_for_lock
        decode_req.req.swa_prefix_lock_released = False
        ticket.owns_restored_lock = False
        ticket.state = LoadBackTicketState.COMMITTED
        decode_req.prefix_match = None
        decode_req.hicache_load_back_ticket = None
