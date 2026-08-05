"""HiCache integration mixins for the decode side of PD disaggregation."""

import logging
import traceback
from dataclasses import dataclass, field
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
    StoragePrefixCoverage,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeRestoreBudget:
    """Independent device capacity reserved for one or more local restores.

    :ivar full_tokens: Full-attention device slots.
    :ivar swa_tokens: Sliding-window device slots.
    :ivar mamba_slots: Cached recurrent-state device slots.
    """

    full_tokens: int = 0
    swa_tokens: int = 0
    mamba_slots: int = 0

    def __add__(self, other: "DecodeRestoreBudget") -> "DecodeRestoreBudget":
        """Combine two component-local reservations.

        :param other: Reservation to add.
        :returns: Component-wise sum.
        """

        return DecodeRestoreBudget(
            full_tokens=self.full_tokens + other.full_tokens,
            swa_tokens=self.swa_tokens + other.swa_tokens,
            mamba_slots=self.mamba_slots + other.mamba_slots,
        )

    @property
    def shared_tokens(self) -> int:
        """Return the reservation for allocators sharing full and SWA indices.

        :returns: Shared slots needed by the larger component restore.
        """

        return max(self.full_tokens, self.swa_tokens)


@dataclass
class DecodePrefixMatch:
    """Decode-side cache coverage and ownership captured at admission."""

    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    last_device_node: Any
    storage_coverage: StoragePrefixCoverage = field(
        default_factory=StoragePrefixCoverage
    )
    last_host_node: Any = None
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    page_size: int = 1
    last_device_lock_params: DecLockRefParams | None = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        """Return the full-attention device prefix length.

        :returns: Number of device-resident prefix tokens.
        """

        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        """Return the full prefix promised to the prefill worker.

        :returns: Device, host, and storage full-attention coverage.
        """

        return (
            self.l1_prefix_len
            + self.l2_host_hit_length
            + self.storage_coverage.prefix_tokens
        )

    def _page_round(self, token_count: int) -> int:
        """Round a component allocation to complete device pages.

        :param token_count: Logical token count.
        :returns: Physical device-slot count.
        """

        return ((token_count + self.page_size - 1) // self.page_size) * self.page_size

    @property
    def full_restore_token_count(self) -> int:
        """Return the page-rounded full-attention restore size.

        :returns: Full-attention device slots reserved for load-back.
        """

        return self._page_round(
            self.l2_host_hit_length + self.storage_coverage.full_tokens
        )

    @property
    def swa_restore_token_count(self) -> int:
        """Return the page-rounded sliding-window restore size.

        :returns: SWA device slots reserved for load-back.
        """

        return self._page_round(
            max(self.swa_host_hit_length, self.storage_coverage.swa_tokens)
        )

    @property
    def mamba_restore_slot_count(self) -> int:
        """Return cached recurrent-state slots reserved for load-back.

        :returns: Recurrent-state device slots reserved for load-back.
        """

        return max(
            self.mamba_host_hit_length,
            self.storage_coverage.mamba_slots,
        )

    @property
    def restore_budget(self) -> DecodeRestoreBudget:
        """Return the component-aware device reservation.

        :returns: Full-attention and SWA device-slot requirements.
        """

        return DecodeRestoreBudget(
            full_tokens=self.full_restore_token_count,
            swa_tokens=self.swa_restore_token_count,
            mamba_slots=self.mamba_restore_slot_count,
        )

    @property
    def needs_local_restore(self) -> bool:
        """Return whether any rank-local component needs load-back.

        :returns: Whether full-attention, SWA, or Mamba state is host-resident.
        """

        return (
            self.full_restore_token_count > 0
            or self.swa_restore_token_count > 0
            or self.mamba_restore_slot_count > 0
        )


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for decode preallocation and component reservations."""

    def _build_decode_prefix_match(self, req: "Req", result: Any) -> DecodePrefixMatch:
        """Convert a cache match into decode-side component coverage.

        :param req: Request being admitted.
        :param result: Prefix-cache match result.
        :returns: Decode-side match and restore reservation.
        """

        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        storage_coverage = StoragePrefixCoverage()
        last_host_node = None
        if self.scheduler.enable_decode_hicache:
            resolved_host_node = self.tree_cache.resolve_node_handle(
                result.last_host_node
            )
            if (
                resolved_host_node.backuped
                or resolved_host_node is self.tree_cache.root_node
            ):
                matched_len = l1_prefix_len + l2_host_hit_length
                reprefill_tail = self.tree_cache.swa_reprefill_tail_tokens()
                storage_limit = max(
                    matched_len,
                    len(req.origin_input_ids) - reprefill_tail,
                )
                suffix_tokens = req.origin_input_ids[matched_len:storage_limit]
                last_hash = resolved_host_node.get_last_hash_value()
                prefix_keys = (
                    resolved_host_node.get_prefix_hash_values(
                        resolved_host_node.parent
                    )
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                storage_coverage = self.tree_cache.query_storage_prefix_coverage(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                    matched_prefix_tokens=matched_len,
                )
                if storage_coverage.prefix_tokens > 0:
                    last_host_node = result.last_host_node

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            last_device_node=result.last_device_node,
            storage_coverage=storage_coverage,
            swa_host_hit_length=result.swa_host_hit_length,
            mamba_host_hit_length=result.mamba_host_hit_length,
            page_size=self.token_to_kv_pool_allocator.page_size,
            last_host_node=last_host_node,
        )

    def _synchronize_storage_coverage(
        self,
        prefix_match: DecodePrefixMatch,
    ) -> None:
        """Make the prefetch descriptor identical on every decode TP rank.

        :param prefix_match: Candidate rank-local cache coverage.
        """

        coverage = prefix_match.storage_coverage
        packed = torch.tensor(
            [
                coverage.prefix_tokens,
                coverage.full_tokens,
                coverage.swa_tokens,
                coverage.mamba_slots,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                packed,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )

        prefix_tokens, full_tokens, swa_tokens, mamba_slots = map(
            int, packed.tolist()
        )
        if prefix_tokens == 0:
            prefix_match.storage_coverage = StoragePrefixCoverage()
            prefix_match.last_host_node = None
            return

        prefix_match.storage_coverage = StoragePrefixCoverage(
            prefix_tokens=prefix_tokens,
            full_tokens=full_tokens,
            swa_tokens=swa_tokens,
            mamba_slots=mamba_slots,
        )

    def _start_hicache_prefetch(
        self,
        req: "Req",
        prefix_match: DecodePrefixMatch | None,
    ) -> None:
        """Speculatively issue an L3 storage prefetch before admission.

        :param req: Request whose storage suffix should be prefetched.
        :param prefix_match: Candidate cache coverage.
        """

        if prefix_match is None:
            return

        self._synchronize_storage_coverage(prefix_match)
        storage_prefix_tokens = prefix_match.storage_coverage.prefix_tokens
        if storage_prefix_tokens == 0 or prefix_match.last_host_node is None:
            return

        try:
            node = self.tree_cache.resolve_node_handle(prefix_match.last_host_node)
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + storage_prefix_tokens
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid,
                prefix_match.last_host_node,
                suffix,
                last_hash,
                prefix_keys,
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
            if not prefix_match.prefetch_registered:
                prefix_match.storage_coverage = StoragePrefixCoverage()
        except Exception:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s; falling back to "
                "L2-only load-back:\n%s",
                req.rid,
                traceback.format_exc(),
            )
            prefix_match.storage_coverage = StoragePrefixCoverage()
            prefix_match.prefetch_registered = False

    def _cancel_hicache_prefetch(
        self,
        req: "Req",
        prefix_match: DecodePrefixMatch,
    ) -> None:
        """Cancel speculative storage work on every TP rank.

        :param req: Request whose prefetch must be discarded.
        :param prefix_match: Candidate cache coverage.
        """

        self.tree_cache.release_aborted_request(req.rid)
        prefix_match.prefetch_registered = False
        prefix_match.last_host_node = None
        prefix_match.storage_coverage = StoragePrefixCoverage()

    def _hicache_pending_restore_budget(self) -> DecodeRestoreBudget:
        """Return reservations not yet consumed by controller allocations.

        :returns: Aggregate full-attention and SWA device reservations.
        """

        if not self.scheduler.enable_decode_hicache:
            return DecodeRestoreBudget()

        budget = DecodeRestoreBudget()
        for decode_req in self.transfer_queue.queue:
            prefix_match = decode_req.prefix_match
            if (
                prefix_match is None
                or decode_req.hicache_restore_status != HiCacheRestoreResult.PENDING
                or decode_req.hicache_load_back_ticket is not None
            ):
                continue
            budget += prefix_match.restore_budget
        return budget

    def _agree_hicache_admission(
        self,
        req: "Req",
        prefix_match: DecodePrefixMatch,
        *,
        uses_separate_swa_allocator: bool,
        full_required_tokens: int,
        full_allocatable_tokens: int,
        swa_required_tokens: int,
        swa_allocatable_tokens: int,
        mamba_required_slots: int,
        mamba_allocatable_slots: int,
    ) -> tuple[bool, DecodeRestoreBudget]:
        """Agree on storage coverage and component capacity before allocation.

        A speculative prefetch can fail or be declined on only one TP rank. The
        first reduction makes the effective L3 prefix rank-symmetric while also
        carrying the normal admission vote. A second admission reduction is
        needed only for the exceptional case where ranks started with different
        L3 coverage.

        :param req: Request being admitted.
        :param prefix_match: Candidate cache coverage after local prefetch start.
        :param uses_separate_swa_allocator: Whether full and SWA restore slots
            come from independent allocators.
        :param full_required_tokens: Ordinary full-attention request requirement.
        :param full_allocatable_tokens: Full-attention capacity after earlier
            pending reservations.
        :param swa_required_tokens: Ordinary sliding-window request requirement.
        :param swa_allocatable_tokens: Sliding-window capacity after earlier
            pending reservations.
        :param mamba_required_slots: Ordinary recurrent-state request slots.
        :param mamba_allocatable_slots: Recurrent-state capacity after earlier
            pending reservations.
        :returns: Rank-consensus admission and the final restore reservation.
        """

        def admission_budget() -> DecodeRestoreBudget:
            budget = prefix_match.restore_budget
            if uses_separate_swa_allocator:
                return budget
            return DecodeRestoreBudget(
                full_tokens=budget.shared_tokens,
                mamba_slots=budget.mamba_slots,
            )

        def can_admit(budget: DecodeRestoreBudget) -> bool:
            return (
                full_required_tokens + budget.full_tokens <= full_allocatable_tokens
                and swa_required_tokens + budget.swa_tokens <= swa_allocatable_tokens
                and mamba_required_slots + budget.mamba_slots
                <= mamba_allocatable_slots
            )

        restore_budget = admission_budget()
        coverage = prefix_match.storage_coverage
        prefetch_attempted = prefix_match.last_host_node is not None
        agreement = torch.tensor(
            [
                int(can_admit(restore_budget)),
                coverage.prefix_tokens,
                -coverage.prefix_tokens,
                coverage.full_tokens,
                -coverage.full_tokens,
                coverage.swa_tokens,
                -coverage.swa_tokens,
                coverage.mamba_slots,
                -coverage.mamba_slots,
                -int(prefetch_attempted),
            ],
            dtype=torch.int64,
            device="cpu",
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                agreement,
                op=torch.distributed.ReduceOp.MIN,
                group=self.gloo_group,
            )

        admitted = bool(agreement[0].item())
        minimum_coverage = tuple(
            int(agreement[index].item()) for index in (1, 3, 5, 7)
        )
        maximum_coverage = tuple(
            -int(agreement[index].item()) for index in (2, 4, 6, 8)
        )
        any_prefetch_attempted = bool(-agreement[9].item())
        if minimum_coverage[0] == 0:
            prefix_match.storage_coverage = StoragePrefixCoverage()
        else:
            prefix_match.storage_coverage = StoragePrefixCoverage(
                prefix_tokens=minimum_coverage[0],
                full_tokens=minimum_coverage[1],
                swa_tokens=minimum_coverage[2],
                mamba_slots=minimum_coverage[3],
            )

        if minimum_coverage[0] == 0 and any_prefetch_attempted:
            self._cancel_hicache_prefetch(req, prefix_match)

        if minimum_coverage != maximum_coverage:
            restore_budget = admission_budget()
            admitted_tensor = torch.tensor(
                int(can_admit(restore_budget)),
                dtype=torch.int32,
                device="cpu",
            )
            if self.tp_size > 1:
                torch.distributed.all_reduce(
                    admitted_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.gloo_group,
                )
            admitted = bool(admitted_tensor.item())

        return admitted, restore_budget


class HiCacheRestoreGatedKVReceiver:
    """Gate transport success until rank-local HiCache restore is ready."""

    def __init__(self, decode_req: "DecodeRequest") -> None:
        """Initialize the receiver gate.

        :param decode_req: Decode request sharing the underlying receiver.
        """

        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        """Poll transport while enforcing local-restore readiness.

        :returns: Effective transport state.
        """

        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """Drive decode-side local restore tickets through commit or abort."""

    def _release_hicache_prefetch(self, decode_req: "DecodeRequest") -> None:
        """Release a registered storage prefetch exactly once.

        :param decode_req: Request owning the registration.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.prefetch_registered:
            return
        self.tree_cache.release_aborted_request(decode_req.req.rid)
        prefix_match.prefetch_registered = False

    def _abort_hicache_local_restore(self, decode_req: "DecodeRequest") -> None:
        """Abort or reap one ticket and release its request lock once.

        :param decode_req: Request whose local restore cannot commit.
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
                ticket.state == LoadBackTicketState.PREPARED
                and ticket.queued_any_component
            ):
                consumer_index = self.tree_cache.ready_to_load_host_cache()
                if consumer_index < 0:
                    raise RuntimeError(
                        "Prepared HiCache ticket lost its controller load"
                    )
                ticket.consumer_index = consumer_index
                ticket.state = LoadBackTicketState.STARTED

            if ticket.state == LoadBackTicketState.STARTED:
                finish_event = (
                    self.tree_cache.cache_controller.layer_done_counter.events[
                        ticket.consumer_index
                    ].finish_event
                )
                finish_event.synchronize()
                if not self.tree_cache.is_load_back_event_done(ticket.consumer_index):
                    raise RuntimeError(
                        "Synchronized HiCache load did not become TP-ready"
                    )
        finally:
            if ticket.owns_restored_lock:
                self.tree_cache.dec_lock_ref(
                    ticket.restored_node,
                    ticket.restored_lock_params,
                )
                ticket.owns_restored_lock = False
            ticket.state = LoadBackTicketState.ABORTED

    def _clean_hicache_prefetch_resources(self, decode_req: "DecodeRequest") -> None:
        """Run the single local-restore abort path.

        :param decode_req: Request being removed from the transfer queue.
        """

        self._abort_hicache_local_restore(decode_req)

    @staticmethod
    def _restore_req_match_state(req: "Req", state: tuple[Any, ...]) -> None:
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
        decode_req: "DecodeRequest",
        message: str,
        ticket: LoadBackTicket | None = None,
    ) -> bool:
        """Transition one request to failed and retain its ticket for cleanup.

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

    def _try_hicache_queue_load_back(self, decode_req: "DecodeRequest") -> bool:
        """Create one validated local load-back ticket.

        :param decode_req: Request awaiting local restore.
        :returns: Whether a prepared ticket now exists.
        """

        prefix_match = decode_req.prefix_match
        assert prefix_match is not None

        if prefix_match.storage_coverage.prefix_tokens > 0:
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
                )
            )
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
            decode_req.hicache_load_back_ticket = ticket
            lock_result = self.tree_cache.inc_lock_ref(ticket.restored_node)
            ticket.restored_lock_params = lock_result.to_dec_params()
            ticket.owns_restored_lock = True
            return True
        finally:
            self._restore_req_match_state(req, original_match_state)

    def _process_hicache_local_restores(
        self, decode_reqs: list["DecodeRequest"]
    ) -> None:
        """Advance identical TP tickets through queue, start, and readiness.

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

        readiness_by_consumer: dict[int, bool] = {}
        for decode_req in active:
            ticket = decode_req.hicache_load_back_ticket
            if ticket is not None and ticket.state == LoadBackTicketState.STARTED:
                if ticket.consumer_index not in readiness_by_consumer:
                    readiness_by_consumer[ticket.consumer_index] = (
                        self.tree_cache.is_load_back_event_done(ticket.consumer_index)
                    )
                if not readiness_by_consumer[ticket.consumer_index]:
                    continue
                decode_req.hicache_restore_status = HiCacheRestoreResult.READY

        if any(
            decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
            and decode_req.hicache_load_back_ticket is not None
            and decode_req.hicache_load_back_ticket.state == LoadBackTicketState.STARTED
            for decode_req in active
        ):
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

        for decode_req in unprepared:
            self._try_hicache_queue_load_back(decode_req)

        prepared = [
            decode_req
            for decode_req in unprepared
            if decode_req.hicache_load_back_ticket is not None
        ]
        if len(prepared) == 0:
            return

        queued = [
            decode_req
            for decode_req in prepared
            if decode_req.hicache_load_back_ticket.queued_any_component
        ]
        if len(queued) == 0:
            for decode_req in prepared:
                if decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING:
                    decode_req.hicache_restore_status = HiCacheRestoreResult.READY
            return

        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            raise RuntimeError(
                "Prepared HiCache tickets lost their merged controller load"
            )
        # A no-copy ticket prepared after an earlier ticket in this round can
        # consume the earlier ticket's newly published device indices. It shares
        # that merged event even though it did not enqueue another component copy.
        for decode_req in prepared:
            ticket = decode_req.hicache_load_back_ticket
            ticket.consumer_index = consumer_index
            ticket.state = LoadBackTicketState.STARTED

    def _commit_hicache_local_restore_to_req(self, decode_req: "DecodeRequest") -> None:
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

        self.tree_cache.dec_lock_ref(
            prefix_match.last_device_node,
            prefix_match.last_device_lock_params,
        )
        restored_indices = ticket.restored_full_device_indices
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
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, restored_indices]
        )
        decode_req.req.last_node = ticket.restored_node
        decode_req.req.last_node_lock_params = ticket.restored_lock_params
        decode_req.req.swa_uuid_for_lock = ticket.restored_lock_params.swa_uuid_for_lock
        decode_req.req.swa_prefix_lock_released = False
        ticket.owns_restored_lock = False
        ticket.state = LoadBackTicketState.COMMITTED
