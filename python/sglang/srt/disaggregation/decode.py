"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import (
    StateType,
    TerminalPrefillAuthorityMismatch,
    TerminalPrefillAuthorityUnavailable,
    TerminalPrefillRequestAuthority,
)
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationComponentClaim,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLeaseError,
    DecodeAllocationLeaseState,
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    DecodeRestoreBudget,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAllocation,
    DecodeReservationAttempt,
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
    derive_decode_reservation_bootstrap_rooms,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedDFlashBoundaryDecodeAdoption,
    PackedRequestPublication,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFlashBoundaryAdoptedValue,
    DFlashBoundaryDeviceRowPool,
)
from sglang.srt.disaggregation.terminal_progress.request_registration import (
    register_packed_terminal_decode_request,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    SingletonPollProgressPolicy,
    TransferBackend,
    _is_fake_transfer,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    resolve_kv_layer_ids,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocation_pin import RequestSlotPinOwner
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    page_align_floor,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_num_new_pages, is_npu
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.staging_handler import (
        DecodeStagingHandler,
    )
    from sglang.srt.managers.scheduler import Scheduler

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()

_PREPARED_COHORT_CONSTRUCTION_SEAL = object()
_FAKE_TRANSFER_BOUNDARY_TOKEN_ID = 0


def _create_singleton_poll_progress_policy(
    scheduler: Scheduler,
    stream_name: str,
) -> SingletonPollProgressPolicy:
    """Create the explicit TP1 progress policy for one decode queue.

    :param scheduler: Decode scheduler owning the queue.
    :param stream_name: Stable polling-stream label for diagnostics.
    :returns: Configured queue-local progress policy.
    """

    return SingletonPollProgressPolicy(
        stream_name=stream_name,
        mode=scheduler.server_args.disaggregation_decode_tp1_poll_progress_mode,
        yield_cadence=(
            scheduler.server_args.disaggregation_decode_tp1_poll_yield_cadence
        ),
        yield_sleep_us=(
            scheduler.server_args.disaggregation_decode_tp1_poll_yield_sleep_us
        ),
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool(RequestSlotPinOwner):
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))
        # Slot-reuse generation counter; mirrors ReqToTokenPool. Required even
        # here: HybridMambaDecodeReqToTokenPool borrows this __init__ while
        # inheriting ReqToTokenPool.alloc, which bumps it.
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
        self._initialize_request_slot_pins(type(self).__name__)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                self.req_generation[r.req_pool_idx] += 1
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.release_detached_request_slot(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self._assert_request_slots_resettable()
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
        speculative_eagle_topk: Optional[int] = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )

    def clear(self):
        self.assert_request_slots_resettable("clear hybrid decode request pool")
        self.mamba_allocator.assert_allocation_resettable(
            "clear hybrid decode request pool"
        )
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_allocator.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    is_rebootstrap: bool = False
    allocation_lease: DecodeAllocationLease | None = None
    packed_transaction: PackedDecodeRequestTransaction | None = None

    # HiCache Status
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    hicache_restored_node: Any = None
    hicache_load_consumer_index: int = -1
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority

    @property
    def terminal_binding_digest(self) -> bytes | None:
        """Return the exact terminal completion authority, when installed.

        :returns: Terminal binding digest, otherwise ``None``.
        """

        transaction = self.packed_transaction
        if transaction is None:
            return None
        return transaction.terminal_binding_digest

    def record_terminal_cancellation(self) -> bool:
        """Record client intent without revoking published terminal ownership.

        The owner still adopts and releases every transport resource. Once the
        request becomes scheduler-visible, the ordinary one-forward abort path
        consumes ``to_finish`` and performs normal request cleanup.

        :returns: Whether terminal ownership retained the cancellation.
        """

        if self.terminal_binding_digest is None:
            return False
        if self.req.to_finish is None:
            self.req.to_finish = FINISH_ABORT()
        return True


class DecodePreparedAllocationCohort:
    """Opaque queue-owned handle for one exact prepared request cohort."""

    __slots__ = ("_queue_nonce", "_token")

    _queue_nonce: object
    _token: object

    def __init__(
        self,
        queue_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one queue-owned cohort handle.

        :param queue_nonce: Exact issuing queue identity.
        :param token: Queue-private cohort record key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _PREPARED_COHORT_CONSTRUCTION_SEAL:
            raise TypeError("prepared allocation cohorts are queue owned")
        self._queue_nonce = queue_nonce
        self._token = token


class _DecodePreparedCohortState(enum.StrEnum):
    """Queue-local ownership state for a decoder reservation cohort."""

    PREPARED = "prepared"
    PROMOTED = "promoted"
    ATTACHED = "attached"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ABORTED = "aborted"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class _DecodePreparedReceiverSeal:
    """Immutable proof that PREPARE initialized one exact receiver.

    :ivar decode_req: Exact request and receiver owner.
    :ivar receiver: Exact initialized receiver identity.
    :ivar authority: Generation-bound source authority consumed by initialization.
    :ivar source_tp_size: Receiver-observed source writer width.
    :ivar prefill_dp_rank: Receiver-observed source DP rank.
    """

    decode_req: DecodeRequest
    receiver: CommonKVReceiver
    authority: TerminalPrefillRequestAuthority
    source_tp_size: int
    prefill_dp_rank: int


@dataclass
class _DecodePreparedCohortRecord:
    """Mutable queue-owned state for one keyed decoder reservation.

    :ivar handle: Opaque handle returned to the reservation authority.
    :ivar grant_id: Exact tokenizer-issued grant identity.
    :ivar reservation_attempt_id: Exact reserve-attempt identity.
    :ivar source_tp_size: Fixed supported packed source writer width.
    :ivar decode_reqs: Ordered exact reserved decode requests.
    :ivar packed_transactions: Ordered request-scoped packed transfer owners.
    :ivar allocations: Ordered immutable allocation receipts.
    :ivar prepared_receiver_seals: PREPARE-retained initialized receiver proofs.
    :ivar packed_publications: Irreversible transfer publications after promotion.
    :ivar state: Current queue-local lifecycle state.
    :ivar metadata_published: Whether the cohort crossed metadata publication.
    :ivar quarantine_reason: First stable quarantine reason, if any.
    """

    handle: DecodePreparedAllocationCohort
    grant_id: uuid.UUID
    reservation_attempt_id: uuid.UUID
    source_tp_size: int
    decode_reqs: tuple[DecodeRequest, ...]
    packed_transactions: tuple[PackedDecodeRequestTransaction, ...]
    allocations: tuple[DecodeReservationAllocation, ...]
    prepared_receiver_seals: tuple[_DecodePreparedReceiverSeal, ...] | None = None
    packed_publications: tuple[PackedRequestPublication, ...] | None = None
    state: _DecodePreparedCohortState = _DecodePreparedCohortState.PREPARED
    metadata_published: bool = False
    quarantine_reason: str | None = None


@dataclass(frozen=True)
class _DecodePartialPreparationQuarantine:
    """Process-lifetime owner for a cohort whose rollback became ambiguous.

    :ivar grant_id: Exact grant whose preparation failed.
    :ivar request_ids: Complete request identity claim retained against reuse.
    :ivar decode_reqs: Partially prepared request owners.
    :ivar packed_transactions: Successfully constructed packed transaction prefix.
    """

    grant_id: uuid.UUID
    request_ids: tuple[str, ...]
    decode_reqs: tuple[DecodeRequest, ...]
    packed_transactions: tuple[PackedDecodeRequestTransaction, ...]


@dataclass(frozen=True)
class _DecodeMetadataSubmission:
    """Metadata submission held until the whole allocation cohort is prepared."""

    decode_req: DecodeRequest
    page_indices: np.ndarray
    state_indices: list[Any] | None
    decode_prefix_len: int


@dataclass
class _DecodeAllocationPreparation:
    """Pre-publication transaction state for one scheduler cohort.

    :ivar prepared_decode_reqs: Children whose migration leases are prepared.
    :ivar publication_started: Whether staging or metadata may be externally visible.
    """

    prepared_decode_reqs: list[DecodeRequest] = field(default_factory=list)
    publication_started: bool = False

    def record_prepared(self, decode_req: DecodeRequest) -> None:
        """Record one child immediately after its lease is acquired.

        :param decode_req: Child whose live allocation may now be pinned.
        """

        if decode_req.allocation_lease is None:
            return
        self.prepared_decode_reqs.append(decode_req)


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    _prepared_cohort_lock: threading.RLock
    _prepared_cohort_nonce: object
    _prepared_cohorts: dict[object, _DecodePreparedCohortRecord]
    _prepared_grant_ids: dict[uuid.UUID, object]
    _prepared_request_ids: dict[str, object]
    _partial_preparation_quarantines: list[_DecodePartialPreparationQuarantine]
    _preparing_grant_ids: set[uuid.UUID]
    _preparing_request_ids: set[str]
    _seen_bootstrap_rooms: set[int]
    _terminal_decode_serving: PackedTerminalDecodeServing | None
    _terminal_dflash_boundary_pool: DFlashBoundaryDeviceRowPool | None
    allocation_lease_authority: DecodeAllocationLeaseAuthority
    allocation_lifecycle_authority: CommonKVManager
    tp1_poll_progress_policy: SingletonPollProgressPolicy

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator | None,
        metadata_buffers: MetadataBuffers | None,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.gloo_group = gloo_group
        self.tp1_poll_progress_policy = _create_singleton_poll_progress_policy(
            scheduler,
            "preallocation",
        )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        self._prepared_cohort_nonce = object()
        self._prepared_cohort_lock = threading.RLock()
        self._prepared_cohorts: dict[object, _DecodePreparedCohortRecord] = {}
        self._prepared_grant_ids: dict[uuid.UUID, object] = {}
        self._prepared_request_ids: dict[str, object] = {}
        self._partial_preparation_quarantines = []
        self._preparing_grant_ids: set[uuid.UUID] = set()
        self._preparing_request_ids: set[str] = set()
        self._seen_bootstrap_rooms = set()
        self._terminal_decode_serving = None
        self._terminal_dflash_boundary_pool = None
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        # Retracted requests staged for rebootstrap while generation is paused.
        # Enqueued into ``self.queue`` only on ``continue_generation`` so the
        # prefix KV is recomputed under the post-retract (updated) weights.
        # NOTE: requests held here are not reachable by ``/abort_request``; to
        # support aborting them we would need an additional fix in the
        # scheduler. In practice this shouldn't arise in the RL scenario.
        self.held_rebootstrap_reqs: List[Req] = []
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        self.allocation_lifecycle_authority = self.kv_manager
        self.allocation_lease_authority = DecodeAllocationLeaseAuthority(
            self.allocation_lifecycle_authority
        )
        self.transfer_queue.allocation_lease_authority = self.allocation_lease_authority
        self.transfer_queue.allocation_lifecycle_authority = (
            self.allocation_lifecycle_authority
        )
        terminal_dflash_boundary_pool = self.kv_manager.terminal_dflash_boundary_pool()
        if terminal_dflash_boundary_pool is not None:
            if not self.scheduler.spec_algorithm.is_dflash():
                raise RuntimeError(
                    "registered terminal DFlash rows require the DFlash algorithm"
                )
            self.transfer_queue.install_terminal_dflash_boundary_pool(
                terminal_dflash_boundary_pool
            )
            self._terminal_dflash_boundary_pool = terminal_dflash_boundary_pool
        self.kv_manager.attach_packed_decode_scheduler(
            self.req_to_metadata_buffer_idx_allocator,
            self.transfer_queue,
        )
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def bind_terminal_decode_serving(
        self,
        serving: PackedTerminalDecodeServing,
    ) -> None:
        """Bind the process-lifetime terminal owner before request promotion.

        :param serving: Exact full decode serving composition.
        """

        if type(serving) is not PackedTerminalDecodeServing:
            raise TypeError("serving must be PackedTerminalDecodeServing")
        if self._terminal_decode_serving is not None:
            raise RuntimeError("terminal decode serving is already bound")
        self._terminal_decode_serving = serving

    @property
    def terminal_decode_serving(self) -> PackedTerminalDecodeServing | None:
        """Return the manager-owned serving composition bound at startup.

        :returns: Full decode serving composition, or ``None`` before binding.
        """

        return self._terminal_decode_serving

    def _require_legacy_metadata_allocator(self) -> ReqToMetadataIdxAllocator:
        """Return legacy metadata allocation authority outside terminal DFlash.

        :returns: The process-local legacy row allocator.
        :raises DecodeAllocationLeaseError: If a legacy path enters terminal DFlash.
        """

        allocator = self.req_to_metadata_buffer_idx_allocator
        if allocator is None:
            raise DecodeAllocationLeaseError(
                "terminal DFlash cannot allocate a legacy metadata row"
            )
        return allocator

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = self._pre_alloc_fill_len(req)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        swa_reserved = self.num_reserved_decode_tokens
        if self.scheduler.server_args.disable_radix_cache:
            swa_reserved = 0
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + swa_reserved,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_parallel().attn_tp_size
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            transfer_kv_pool.get_contiguous_buf_infos()
        )
        kv_layer_ids = resolve_kv_layer_ids(
            transfer_kv_pool,
            len(kv_data_ptrs),
        )
        kv_data_mem_kinds = (
            ["DRAM"] * len(kv_data_ptrs)
            if self.scheduler.enable_hisparse
            else ["VRAM"] * len(kv_data_ptrs)
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
            kv_data_mem_kinds += ["VRAM"] * len(device_kv_data_ptrs[c4_layer_num:])
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            draft_kv_layer_ids = resolve_kv_layer_ids(
                self.draft_token_to_kv_pool,
                len(draft_kv_data_ptrs),
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)
            kv_layer_ids = (
                [*kv_layer_ids, *draft_kv_layer_ids]
                if len(kv_layer_ids) > 0 and len(draft_kv_layer_ids) > 0
                else []
            )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = kv_layer_ids
        if self.transfer_backend == TransferBackend.NIXL:
            kv_args.kv_data_mem_kinds = kv_data_mem_kinds
        kv_args.page_size = self.token_to_kv_pool.page_size
        kv_args.terminal_request_capacity = self.req_to_token_pool.size

        if self.metadata_buffers is None:
            kv_args.aux_data_ptrs = []
            kv_args.aux_data_lens = []
            kv_args.aux_item_lens = []
        else:
            (
                kv_args.aux_data_ptrs,
                kv_args.aux_data_lens,
                kv_args.aux_item_lens,
            ) = self.metadata_buffers.get_buf_infos()

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(
        self, req: Req, is_retracted: bool = False, is_rebootstrap: bool = False
    ) -> None:
        """Add a request to the pending queue.

        ``is_rebootstrap`` marks a PD true-retraction request whose prefix KV
        must be recomputed by the original prefill worker under the current
        weights (rather than resumed from stale CPU KV). It otherwise follows the
        same bootstrap-handshake path as a fresh request; the ``/generate``
        dispatch happens later, after preallocation and ``send_metadata`` (see
        ``pop_preallocated``).
        """
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(
                req, is_rebootstrap=is_rebootstrap
            )

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def prepare_preallocated(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        requests: tuple[Req, ...],
    ) -> tuple[
        tuple[DecodeReservationAllocation, ...],
        DecodePreparedAllocationCohort,
    ]:
        """Reserve one exact decoder cohort without publishing runnable work.

        :param grant_id: Tokenizer-issued non-secret grant identity.
        :param attempt: Exact authenticated reserve attempt.
        :param requests: Ordered canonical scheduler requests.
        :returns: Immutable child receipts and an opaque retained cohort.
        """

        owned_requests = tuple(requests)
        request_ids = self._validate_preallocated_request_cohort(
            grant_id,
            attempt,
            owned_requests,
        )
        self._validate_preallocated_structural_capacity(owned_requests)
        rooms = derive_decode_reservation_bootstrap_rooms(
            grant_id,
            attempt.child_request_ids,
        )

        terminal_prefill_authorities: (
            tuple[TerminalPrefillRequestAuthority, ...] | None
        ) = None
        if self._terminal_decode_serving is not None:
            bootstrap_addr = NetworkAddress(
                attempt.prefill_bootstrap_endpoint.host,
                attempt.prefill_bootstrap_endpoint.port,
            ).to_host_port_str()
            resolved_authorities: list[TerminalPrefillRequestAuthority] = []
            for req in owned_requests:
                try:
                    authority = (
                        self.kv_manager.resolve_terminal_prefill_request_authority(
                            bootstrap_addr=bootstrap_addr,
                            prefill_process_url=attempt.prefill_process.url,
                            prefill_process_instance_id=(
                                attempt.prefill_process.instance_id
                            ),
                            prefill_dp_rank=req.disagg_prefill_dp_rank,
                            source_tp_size=attempt.source_tp_size,
                        )
                    )
                except TerminalPrefillAuthorityUnavailable as error:
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                    raise DecodeReservationAdmissionRefused(
                        "terminal_prefill_authority_unavailable",
                        DecodeReservationRefusalDisposition.RETRY_SAME_DECODER,
                        str(error),
                    ) from error
                except TerminalPrefillAuthorityMismatch as error:
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                    raise DecodeReservationAdmissionRefused(
                        "terminal_prefill_authority_mismatch",
                        DecodeReservationRefusalDisposition.TERMINAL,
                        str(error),
                    ) from error
                resolved_authorities.append(authority)
            terminal_prefill_authorities = tuple(resolved_authorities)

        with self._prepared_cohort_lock:
            self._claim_preallocated_request_ids_locked(
                grant_id,
                request_ids,
                rooms,
            )

        prepared_decode_reqs: list[DecodeRequest] = []
        prepared_receiver_seals: list[_DecodePreparedReceiverSeal] = []
        prepared_packed_transactions: list[PackedDecodeRequestTransaction] = []
        allocations: list[DecodeReservationAllocation] = []
        try:
            for req, room in zip(owned_requests, rooms, strict=True):
                req.bootstrap_host = attempt.prefill_bootstrap_endpoint.host
                req.bootstrap_port = attempt.prefill_bootstrap_endpoint.port
                req.bootstrap_room = room

            prefix_matches: list[DecodePrefixMatch | None] = []
            for request_index, req in enumerate(owned_requests):
                decode_req = self._create_receiver(req)
                prepared_decode_reqs.append(decode_req)
                if terminal_prefill_authorities is not None:
                    authority = terminal_prefill_authorities[request_index]
                    try:
                        decode_req.kv_receiver.init_from_terminal_authority(authority)
                    except TerminalPrefillAuthorityMismatch as error:
                        if self.scheduler.metrics_reporter.enable_metrics:
                            self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                        raise DecodeReservationAdmissionRefused(
                            "terminal_prefill_authority_mismatch",
                            DecodeReservationRefusalDisposition.TERMINAL,
                            str(error),
                        ) from error
                    if decode_req.kv_receiver.conclude_state is KVPoll.Failed:
                        if self.scheduler.metrics_reporter.enable_metrics:
                            self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                        raise DecodeAllocationLeaseError(
                            "terminal decode receiver initialization failed"
                        )
                    actual_source_tp_size = (
                        decode_req.kv_receiver.prefill_info.attn_tp_size
                    )
                    if actual_source_tp_size != attempt.source_tp_size:
                        raise DecodeAllocationLeaseError(
                            "bootstrap source TP width differs from reservation"
                        )
                    prepared_receiver_seals.append(
                        _DecodePreparedReceiverSeal(
                            decode_req=decode_req,
                            receiver=decode_req.kv_receiver,
                            authority=authority,
                            source_tp_size=actual_source_tp_size,
                            prefill_dp_rank=decode_req.kv_receiver.prefill_dp_rank,
                        )
                    )
                    req.time_stats.set_bootstrap_done_time()
                prefix_match = self._match_preallocated_prefix_and_lock(req)
                decode_req.prefix_match = prefix_match
                prefix_matches.append(prefix_match)

            if self.scheduler.enable_decode_hicache:
                for decode_req in prepared_decode_reqs:
                    self._start_hicache_prefetch(
                        decode_req.req,
                        decode_req.prefix_match,
                    )

            self._validate_preallocated_capacity(
                owned_requests,
                tuple(prefix_matches),
            )

            for child_request_id, req, room, decode_req, prefix_match in zip(
                attempt.child_request_ids,
                owned_requests,
                rooms,
                prepared_decode_reqs,
                prefix_matches,
                strict=True,
            ):
                metadata_index: int | None = None
                if self._terminal_dflash_boundary_pool is None:
                    metadata_index = self._require_legacy_metadata_allocator().alloc()
                    if metadata_index is None:
                        raise DecodeReservationAdmissionRefused(
                            "decode_metadata_capacity",
                            DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
                        )
                    decode_req.metadata_buffer_index = metadata_index
                migration_end = self._rebootstrap_prefill_len(req)
                prefix_len = 0 if prefix_match is None else prefix_match.l1_prefix_len
                total_prefix_len = (
                    0 if prefix_match is None else prefix_match.decode_prefix_len
                )
                req.cache_protected_len = total_prefix_len
                self._pre_alloc(
                    req,
                    prefix_indices=(
                        None if prefix_match is None else prefix_match.prefix_indices
                    ),
                    prefix_len=prefix_len,
                    total_prefix_len=total_prefix_len,
                    decode_req=decode_req,
                    migration_end=migration_end,
                    source_tp_size=attempt.source_tp_size,
                )
                lease = decode_req.allocation_lease
                if lease is None:
                    raise DecodeAllocationLeaseError(
                        "reserved asymmetric request acquired no migration lease"
                    )
                snapshot = self.allocation_lease_authority.snapshot(lease)
                packed_transaction = (
                    self.kv_manager.prepare_packed_decode_request_transaction(
                        room_id=room,
                        request_owner=decode_req,
                        metadata_buffer_index=metadata_index,
                        allocation_lease=lease,
                        allocation_authority=self.allocation_lease_authority,
                        lifecycle_authority=self.allocation_lifecycle_authority,
                        source_tp_size=attempt.source_tp_size,
                    )
                )
                if packed_transaction is None:
                    raise RuntimeError(
                        "packed decode runtime accepted admission but returned no "
                        "request transaction"
                    )
                prepared_packed_transactions.append(packed_transaction)
                if self._terminal_dflash_boundary_pool is not None:
                    boundary_row_index = (
                        packed_transaction.prepared_publication().auxiliary_plan.metadata_buffer_index
                    )
                    decode_req.metadata_buffer_index = boundary_row_index
                slot_generation = uuid.UUID(bytes=snapshot.lease_id)
                allocations.append(
                    DecodeReservationAllocation(
                        child_request_id=child_request_id,
                        decoder_slot_generation=slot_generation,
                        bootstrap_room=room,
                        request_slot=snapshot.request_slot,
                        request_generation=snapshot.request_generation,
                        writer_manifest_digest=snapshot.writer_manifest.digest,
                        allocation_digest=snapshot.allocation_digest,
                        reserved_kv_tokens=(
                            self._required_alloc_tokens(
                                fill_len=self._pre_alloc_fill_len(req),
                                prefix_len=total_prefix_len,
                            )
                            + (
                                0
                                if prefix_match is None
                                else prefix_match.full_restore_token_count
                            )
                            + self.num_reserved_decode_tokens
                        ),
                        remaining_decode_tokens=min(
                            req.sampling_params.max_new_tokens,
                            CLIP_MAX_NEW_TOKEN,
                        ),
                    )
                )
        except Exception:  # noqa: BLE001
            preparation_traceback = traceback.format_exc()
            try:
                self._rollback_preallocated_decode_reqs(
                    prepared_decode_reqs,
                    prepared_packed_transactions,
                )
            except Exception as rollback_error:  # noqa: BLE001
                rollback_traceback = traceback.format_exc()
                with self._prepared_cohort_lock:
                    self._partial_preparation_quarantines.append(
                        _DecodePartialPreparationQuarantine(
                            grant_id=grant_id,
                            request_ids=request_ids,
                            decode_reqs=tuple(prepared_decode_reqs),
                            packed_transactions=tuple(prepared_packed_transactions),
                        )
                    )
                logger.critical(
                    "Reserved decode cohort preparation and rollback both failed. "
                    "Preparation traceback:\n%s\nRollback traceback:\n%s",
                    preparation_traceback,
                    rollback_traceback,
                )
                raise DecodeAllocationLeaseError(
                    "reserved decode cohort rollback failed; partial ownership "
                    "was quarantined"
                ) from rollback_error
            with self._prepared_cohort_lock:
                self._release_preallocated_claim_locked(
                    grant_id,
                    request_ids,
                )
            raise

        token = object()
        handle = DecodePreparedAllocationCohort(
            self._prepared_cohort_nonce,
            token,
            _PREPARED_COHORT_CONSTRUCTION_SEAL,
        )
        record = _DecodePreparedCohortRecord(
            handle=handle,
            grant_id=grant_id,
            reservation_attempt_id=attempt.reservation_attempt_id,
            source_tp_size=attempt.source_tp_size,
            decode_reqs=tuple(prepared_decode_reqs),
            packed_transactions=tuple(prepared_packed_transactions),
            allocations=tuple(allocations),
            prepared_receiver_seals=(
                tuple(prepared_receiver_seals)
                if terminal_prefill_authorities is not None
                else None
            ),
        )
        with self._prepared_cohort_lock:
            if grant_id not in self._preparing_grant_ids:
                raise RuntimeError("prepared cohort lost its grant claim")
            if any(
                request_id not in self._preparing_request_ids
                for request_id in request_ids
            ):
                raise RuntimeError("prepared cohort lost a request claim")
            self._preparing_grant_ids.remove(grant_id)
            for request_id in request_ids:
                self._preparing_request_ids.remove(request_id)
                self._prepared_request_ids[request_id] = token
            self._prepared_grant_ids[grant_id] = token
            self._prepared_cohorts[token] = record
        return record.allocations, handle

    def has_live_preallocated_cohorts(self) -> bool:
        """Return whether keyed reservation ownership blocks destructive idleness.

        Quarantined cohorts remain live because their request, metadata, and KV
        resources intentionally stay process-owned.

        :returns: Whether any complete cohort or ambiguous partial preparation
            remains process-owned.
        """

        with self._prepared_cohort_lock:
            return (
                len(self._prepared_cohorts) > 0
                or len(self._partial_preparation_quarantines) > 0
            )

    def promote_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> None:
        """Publish one prepared cohort's transport ownership.

        Terminal transport publication includes destination metadata because
        prefill cannot run until that metadata exists. Legacy cohorts retain
        their scheduler attachment boundary at the inference request.

        :param cohort: Exact queue-owned prepared cohort.
        """

        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            if record.state is not _DecodePreparedCohortState.PREPARED:
                raise DecodeAllocationLeaseError(
                    f"cohort promotion is invalid in state {record.state.value}"
                )
            publications: list[PackedRequestPublication] = []
            try:
                terminal_startup = self.kv_manager.terminal_startup_binding
                terminal_serving = self._terminal_decode_serving
                if terminal_startup is not None and terminal_serving is None:
                    raise RuntimeError(
                        "terminal decode promotion requires the full serving owner"
                    )
                if terminal_startup is None:
                    terminal_serving = None
                for decode_req, transaction in zip(
                    record.decode_reqs,
                    record.packed_transactions,
                    strict=True,
                ):
                    if terminal_serving is not None:
                        authority = (
                            self.kv_manager.build_terminal_decode_request_authority(
                                transaction=transaction,
                                adopt_request=partial(
                                    self._adopt_terminal_decode_request,
                                    record,
                                    decode_req,
                                    transaction,
                                ),
                                finalize_request=partial(
                                    self._finalize_terminal_decode_request,
                                    record,
                                    decode_req,
                                    transaction,
                                ),
                                cancel_request=partial(
                                    self._cancel_terminal_decode_request,
                                    record,
                                    decode_req,
                                    transaction,
                                ),
                                quarantine_request=partial(
                                    self._quarantine_terminal_decode_request,
                                    record,
                                    decode_req,
                                    transaction,
                                ),
                            )
                        )
                        register_packed_terminal_decode_request(
                            terminal_serving,
                            authority,
                        )
                    publications.append(transaction.publish())
            except Exception:  # noqa: BLE001
                promotion_traceback = traceback.format_exc()
                self._quarantine_preallocated_record_locked(
                    record,
                    "cohort promotion failed after publication may have begun",
                )
                logger.error(
                    "Reserved decode cohort promotion failed and was quarantined:\n%s",
                    promotion_traceback,
                )
                raise
            record.packed_publications = tuple(publications)
            record.state = _DecodePreparedCohortState.PROMOTED
            if terminal_serving is not None:
                try:
                    self._attach_promoted_preallocated_record(record)
                except Exception:  # noqa: BLE001
                    if record.state is not _DecodePreparedCohortState.QUARANTINED:
                        self._quarantine_preallocated_record_locked(
                            record,
                            "terminal promotion failed after packed publication",
                        )
                    raise

    def _adopt_terminal_decode_request(
        self,
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        owner: object,
    ) -> TerminalDFlashDecodeAdoption:
        """Copy terminal metadata while its actor-owned row remains pinned.

        :param record: Exact prepared cohort retaining the request.
        :param decode_req: Exact mutable decode request.
        :param transaction: Exact terminal-owned packed transaction.
        :param owner: Candidate owner returned by allocation adoption.
        :returns: Exact D2D completion authority for owner-side row release.
        """

        self._require_terminal_callback_owner(
            record,
            decode_req,
            transaction,
            owner,
        )
        with self._prepared_cohort_lock:
            if record.state is not _DecodePreparedCohortState.ATTACHED:
                raise DecodeAllocationLeaseError(
                    "terminal adoption requires an attached prepared cohort"
                )
            if not record.metadata_published:
                raise DecodeAllocationLeaseError(
                    "terminal adoption preceded packed metadata publication"
                )
            return self.transfer_queue.adopt_terminal_request(
                decode_req,
                transaction,
            )

    def _finalize_terminal_decode_request(
        self,
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        owner: object,
    ) -> None:
        """Publish one adopted request only after actor metadata release.

        :param record: Exact prepared cohort retaining the request.
        :param decode_req: Exact mutable decode request.
        :param transaction: Exact terminal-owned packed transaction.
        :param owner: Candidate owner returned by allocation adoption.
        """

        self._require_terminal_callback_owner(
            record,
            decode_req,
            transaction,
            owner,
        )
        with self._prepared_cohort_lock:
            if record.state is not _DecodePreparedCohortState.ATTACHED:
                raise DecodeAllocationLeaseError(
                    "terminal finalization requires an attached prepared cohort"
                )
            self.transfer_queue.finalize_terminal_request(decode_req, transaction)

    def _cancel_terminal_decode_request(
        self,
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        owner: object,
    ) -> None:
        """Release scheduler resources after actor-authorized cancellation.

        :param record: Exact prepared cohort retaining the request.
        :param decode_req: Exact mutable decode request.
        :param transaction: Exact safely cancelled packed transaction.
        :param owner: Candidate owner returned by cancellation.
        """

        self._require_terminal_callback_owner(
            record,
            decode_req,
            transaction,
            owner,
        )
        with self._prepared_cohort_lock:
            if record.state is not _DecodePreparedCohortState.PREPARED:
                raise DecodeAllocationLeaseError(
                    "terminal cancellation requires an unpublished prepared cohort"
                )
            self._release_cancelled_preallocated_decode_req(decode_req)

    def _quarantine_terminal_decode_request(
        self,
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        owner: object,
        reason: str,
    ) -> None:
        """Remove active queue visibility while retaining ambiguous resources.

        :param record: Exact prepared cohort retaining the request.
        :param decode_req: Exact mutable decode request.
        :param transaction: Exact quarantined packed transaction.
        :param owner: Candidate retained scheduler owner.
        :param reason: Stable first ambiguity evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        self._require_terminal_callback_owner(
            record,
            decode_req,
            transaction,
            owner,
        )
        with self._prepared_cohort_lock:
            removed_count = self._quarantine_preallocated_record_locked(record, reason)
        if self.scheduler.metrics_reporter.enable_metrics:
            for _ in range(removed_count):
                self.scheduler.metrics_collector.increment_transfer_failed_reqs()

    def _require_terminal_callback_owner(
        self,
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        owner: object,
    ) -> None:
        """Require one callback to retain the exact prepared child graph.

        :param record: Candidate live cohort record.
        :param decode_req: Candidate cohort child.
        :param transaction: Candidate child transaction.
        :param owner: Candidate callback owner.
        """

        if owner is not decode_req:
            raise DecodeAllocationLeaseError(
                "terminal decode callback returned another request owner"
            )
        with self._prepared_cohort_lock:
            if self._prepared_cohorts.get(record.handle._token) is not record:
                raise DecodeAllocationLeaseError(
                    "terminal decode callback lost its prepared cohort"
                )
            matching_children = tuple(
                (owned_req, owned_transaction)
                for owned_req, owned_transaction in zip(
                    record.decode_reqs,
                    record.packed_transactions,
                    strict=True,
                )
                if owned_req is decode_req and owned_transaction is transaction
            )
            if len(matching_children) != 1:
                raise DecodeAllocationLeaseError(
                    "terminal decode callback differs from its prepared child"
                )
            if transaction.request_owner is not decode_req:
                raise DecodeAllocationLeaseError(
                    "terminal transaction retained another decode request"
                )

    def attach_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> None:
        """Attach exact inference ownership to one promoted cohort.

        Terminal cohorts already belong to their process-lifetime progress
        owner at promotion. Their inference attachment therefore validates the
        retained state without republishing transport metadata. Legacy cohorts
        enter scheduler-owned progress here.

        :param cohort: Exact queue-owned promoted cohort.
        """

        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            if self._terminal_decode_serving is not None:
                self._validate_terminal_inference_attachment_locked(record)
                return
            self._attach_promoted_preallocated_record(record)

    def _validate_terminal_inference_attachment_locked(
        self,
        record: _DecodePreparedCohortRecord,
    ) -> None:
        """Validate retained terminal transport ownership without mutation.

        :param record: Exact cohort receiving its inference response owner.
        """

        if record.state is not _DecodePreparedCohortState.ATTACHED:
            raise DecodeAllocationLeaseError(
                "terminal inference attachment requires transport publication, "
                f"not state {record.state.value}"
            )
        publications = record.packed_publications
        if publications is None or len(publications) != len(record.decode_reqs):
            raise DecodeAllocationLeaseError(
                "terminal inference attachment lost packed publication evidence"
            )
        receiver_seals = record.prepared_receiver_seals
        if receiver_seals is None or len(receiver_seals) != len(record.decode_reqs):
            raise DecodeAllocationLeaseError(
                "terminal inference attachment lost prepared receiver evidence"
            )
        if not record.metadata_published:
            raise DecodeAllocationLeaseError(
                "terminal inference attachment preceded metadata publication"
            )

        for decode_req, transaction, publication, allocation, receiver_seal in zip(
            record.decode_reqs,
            record.packed_transactions,
            publications,
            record.allocations,
            receiver_seals,
            strict=True,
        ):
            if transaction.request_owner is not decode_req:
                raise DecodeAllocationLeaseError(
                    "terminal inference attachment retained another request owner"
                )
            if (
                publication.key.room_id != allocation.bootstrap_room
                or publication.key.request_generation
                != allocation.decoder_slot_generation.bytes
                or publication.request_slot_generation
                != allocation.request_generation
                or publication.writer_manifest_digest
                != allocation.writer_manifest_digest
                or publication.allocation_digest != allocation.allocation_digest
                or publication.terminal_source_plan is None
            ):
                raise DecodeAllocationLeaseError(
                    "terminal inference attachment publication differs from its "
                    "prepared allocation"
                )
            if (
                receiver_seal.decode_req is not decode_req
                or receiver_seal.source_tp_size != record.source_tp_size
            ):
                raise DecodeAllocationLeaseError(
                    "terminal inference attachment receiver evidence changed"
                )
            self.transfer_queue.validate_terminal_inference_attachment(
                decode_req,
                transaction,
                receiver_seal.receiver,
            )

    def _attach_promoted_preallocated_record(
        self,
        record: _DecodePreparedCohortRecord,
    ) -> None:
        """Attach one promoted record to its role-specific progress owner.

        :param record: Exact promoted record retained by this queue.
        """

        if record.state is not _DecodePreparedCohortState.PROMOTED:
            raise DecodeAllocationLeaseError(
                f"cohort attachment is invalid in state {record.state.value}"
            )
        if any(
            any(entry is decode_req for entry in self.queue)
            or any(entry is decode_req for entry in self.pending_reqs)
            for decode_req in record.decode_reqs
        ):
            self._quarantine_preallocated_record_locked(
                record,
                "cohort attachment found duplicate queue ownership",
            )
            raise DecodeAllocationLeaseError(
                "prepared cohort is already present in a decode queue"
            )

        for decode_req, transaction in zip(
            record.decode_reqs,
            record.packed_transactions,
            strict=True,
        ):
            if decode_req.packed_transaction is not None:
                self._quarantine_preallocated_record_locked(
                    record,
                    "cohort attachment found existing packed ownership",
                )
                raise DecodeAllocationLeaseError(
                    "prepared request already owns a packed transaction"
                )
            if transaction.request_owner is not decode_req:
                self._quarantine_preallocated_record_locked(
                    record,
                    "cohort attachment found another packed request owner",
                )
                raise DecodeAllocationLeaseError(
                    "packed transaction retained another decode request"
                )
            decode_req.packed_transaction = transaction

        try:
            if self._terminal_decode_serving is None:
                self._attach_legacy_preallocated_record(record)
            else:
                self._attach_terminal_preallocated_record(record)
        except Exception:  # noqa: BLE001
            attachment_traceback = traceback.format_exc()
            self.queue = [
                entry
                for entry in self.queue
                if all(entry is not owned for owned in record.decode_reqs)
            ]
            self.pending_reqs = [
                entry
                for entry in self.pending_reqs
                if all(entry is not owned for owned in record.decode_reqs)
            ]
            self._quarantine_preallocated_record_locked(
                record,
                "cohort attachment failed after progress ownership may have begun",
            )
            logger.error(
                "Reserved decode cohort attachment failed and was quarantined:\n%s",
                attachment_traceback,
            )
            raise

    def _attach_legacy_preallocated_record(
        self,
        record: _DecodePreparedCohortRecord,
    ) -> None:
        """Attach one non-terminal cohort to legacy scheduler polling.

        :param record: Exact promoted cohort retaining every child.
        """

        if any(
            transaction.terminal_binding_digest is not None
            for transaction in record.packed_transactions
        ):
            raise DecodeAllocationLeaseError(
                "terminal transaction has no terminal decode serving owner"
            )
        self.queue.extend(record.decode_reqs)
        for decode_req in record.decode_reqs:
            if _is_fake_transfer(
                decode_req.req,
                self.scheduler.server_args,
            ):
                decode_req.kv_receiver.init(0)
                continue
            prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
            if prefill_dp_rank is None:
                self.pending_reqs.append(decode_req)
                continue
            decode_req.kv_receiver.init(prefill_dp_rank)
        record.state = _DecodePreparedCohortState.ATTACHED

    def _attach_terminal_preallocated_record(
        self,
        record: _DecodePreparedCohortRecord,
    ) -> None:
        """Publish one owner-driven cohort without legacy progress queues.

        Receiver bootstrap and destination-metadata construction remain
        scheduler-affine. Once promotion installed terminal authority, however,
        no child may enter the legacy preallocation handshake or transfer poller.
        The scheduler thread cannot drain owner receipts while this method runs,
        so the complete registry and metadata cohort becomes visible atomically
        before the first adoption callback can execute.

        :param record: Exact promoted terminal cohort retaining every child.
        """

        publications = record.packed_publications
        if publications is None or len(publications) != len(record.decode_reqs):
            raise DecodeAllocationLeaseError(
                "terminal cohort publication count differs from its request count"
            )
        if any(
            transaction.terminal_binding_digest is None
            for transaction in record.packed_transactions
        ):
            raise DecodeAllocationLeaseError(
                "terminal decode serving received a non-terminal transaction"
            )
        receiver_seals = record.prepared_receiver_seals
        if receiver_seals is None or len(receiver_seals) != len(record.decode_reqs):
            raise DecodeAllocationLeaseError(
                "terminal decode cohort has no prepared receiver seal"
            )

        submissions: list[_DecodeMetadataSubmission] = []
        for decode_req, receiver_seal in zip(
            record.decode_reqs,
            receiver_seals,
            strict=True,
        ):
            if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
                raise DecodeAllocationLeaseError(
                    "terminal decode ownership requires the NIXL transport"
                )
            if (
                receiver_seal.decode_req is not decode_req
                or receiver_seal.receiver is not decode_req.kv_receiver
                or receiver_seal.source_tp_size != record.source_tp_size
                or receiver_seal.prefill_dp_rank
                != decode_req.kv_receiver.prefill_dp_rank
                or decode_req.kv_receiver.conclude_state is KVPoll.Failed
            ):
                raise DecodeAllocationLeaseError(
                    "terminal decode receiver differs from its prepared seal"
                )
            actual_source_tp_size = decode_req.kv_receiver.prefill_info.attn_tp_size
            if actual_source_tp_size != record.source_tp_size:
                raise DecodeAllocationLeaseError(
                    "bootstrap source TP width differs from reservation"
                )
            origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
            prefix_match = decode_req.prefix_match
            prefix_len = 0 if prefix_match is None else prefix_match.l1_prefix_len
            total_prefix_len = (
                0 if prefix_match is None else prefix_match.decode_prefix_len
            )
            decode_req.req.cache_protected_len = total_prefix_len
            submissions.append(
                self._build_decode_metadata_submission(
                    decode_req,
                    origin_input_len=origin_input_len,
                    prefix_len=prefix_len,
                    total_prefix_len=total_prefix_len,
                    dst_kv_indices=None,
                    allocate_metadata_index=False,
                )
            )

        self.transfer_queue.register_terminal_requests(record.decode_reqs)
        record.state = _DecodePreparedCohortState.ATTACHED
        for submission, transaction, publication in zip(
            submissions,
            record.packed_transactions,
            publications,
            strict=True,
        ):
            decode_req = submission.decode_req
            self.kv_manager.send_packed_decode_request_metadata(
                transaction=transaction,
                publication=publication,
                receiver=decode_req.kv_receiver,
                page_indices=submission.page_indices,
                metadata_buffer_index=decode_req.metadata_buffer_index,
                state_indices=submission.state_indices,
                decode_prefix_len=submission.decode_prefix_len,
            )
            if decode_req.is_rebootstrap:
                self.kv_manager.submit_prefill_recompute(
                    decode_req.kv_receiver,
                    decode_req.req.build_rebootstrap_payload(),
                )
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()
        record.metadata_published = True

    def cancel_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> DecodeReservationState:
        """Release one exact cohort before promotion.

        :param cohort: Exact queue-owned prepared cohort.
        :returns: Authoritative cancelled state.
        """

        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            if record.state is not _DecodePreparedCohortState.PREPARED:
                raise DecodeAllocationLeaseError(
                    f"cohort cancellation is invalid in state {record.state.value}"
                )
            self._rollback_preallocated_decode_reqs(
                list(record.decode_reqs),
                list(record.packed_transactions),
            )
            record.state = _DecodePreparedCohortState.CANCELLED
            self._retire_preallocated_record_locked(record)
        return DecodeReservationState.CANCELLED

    def complete_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> DecodeReservationState:
        """Reconcile normal completion after transport relinquishes every lease.

        :param cohort: Exact queue-owned attached cohort.
        :returns: Completed or conservatively quarantined state.
        """

        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            if record.state is not _DecodePreparedCohortState.ATTACHED:
                raise DecodeAllocationLeaseError(
                    f"cohort completion is invalid in state {record.state.value}"
                )
            if any(
                decode_req.allocation_lease is not None
                for decode_req in record.decode_reqs
            ):
                self._quarantine_preallocated_record_locked(
                    record,
                    "normal completion preceded packed allocation teardown",
                )
                return DecodeReservationState.QUARANTINED
            record.state = _DecodePreparedCohortState.COMPLETED
            self._retire_preallocated_record_locked(record)
            return DecodeReservationState.COMPLETED

    def abort_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Reconcile a promoted abort without releasing live migration owners.

        :param cohort: Exact queue-owned promoted or attached cohort.
        :param reason_code: Stable machine-readable failure reason.
        :param diagnostic: Optional bounded diagnostic.
        :returns: Aborted after prior teardown, otherwise quarantined.
        """

        del diagnostic
        if len(reason_code) == 0:
            raise ValueError("abort reason_code must not be empty")
        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            if record.state not in (
                _DecodePreparedCohortState.PROMOTED,
                _DecodePreparedCohortState.ATTACHED,
            ):
                raise DecodeAllocationLeaseError(
                    f"cohort abort is invalid in state {record.state.value}"
                )
            for decode_req in record.decode_reqs:
                if decode_req.allocation_lease is not None:
                    continue
                # Relinquishing the migration lease makes Req scheduler-owned.
                # Keep waiting/running cleanup and tokenizer notification on the
                # scheduler's singular abort path.
                self.scheduler.abort_request(AbortReq(rid=decode_req.req.rid))
            if any(
                decode_req.allocation_lease is not None
                for decode_req in record.decode_reqs
            ):
                self._quarantine_preallocated_record_locked(record, reason_code)
                return DecodeReservationState.QUARANTINED
            record.state = _DecodePreparedCohortState.ABORTED
            self._retire_preallocated_record_locked(record)
            return DecodeReservationState.ABORTED

    def quarantine_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Retain every ambiguous cohort owner for the process lifetime.

        :param cohort: Exact queue-owned live cohort.
        :param reason_code: Stable machine-readable failure reason.
        :param diagnostic: Optional bounded diagnostic.
        :returns: Authoritative quarantined state.
        """

        del diagnostic
        if len(reason_code) == 0:
            raise ValueError("quarantine reason_code must not be empty")
        with self._prepared_cohort_lock:
            record = self._require_preallocated_cohort_locked(cohort)
            self._quarantine_preallocated_record_locked(record, reason_code)
        return DecodeReservationState.QUARANTINED

    def _validate_preallocated_request_cohort(
        self,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        requests: tuple[Req, ...],
    ) -> tuple[str, ...]:
        """Validate one canonical request cohort before allocator mutation.

        :param grant_id: Candidate grant identity.
        :param attempt: Candidate authenticated reserve attempt.
        :param requests: Candidate ordered scheduler requests.
        :returns: Exact ordered string request identities.
        """

        if type(grant_id) is not uuid.UUID or grant_id.int == 0:
            raise ValueError("grant_id must be a non-nil UUID")
        if type(attempt) is not DecodeReservationAttempt:
            raise TypeError("attempt must be DecodeReservationAttempt")
        if self.kv_manager.attn_tp_size not in (1, 2):
            raise DecodeReservationAdmissionRefused(
                "decode_tp_not_supported",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
                "reserved decode requires a TP1 or TP2 attention destination",
            )
        if attempt.source_tp_size % self.kv_manager.attn_tp_size != 0:
            raise DecodeReservationAdmissionRefused(
                "decode_tp_not_divisible",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        if not self.enable_staging or self.transfer_backend is not TransferBackend.NIXL:
            raise DecodeReservationAdmissionRefused(
                "packed_staging_unavailable",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        if not self.kv_manager.supports_packed_decode_request_transactions():
            raise DecodeReservationAdmissionRefused(
                "packed_runtime_unavailable",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        if (
            self._terminal_decode_serving is not None
            and self.scheduler.enable_decode_hicache
        ):
            raise DecodeReservationAdmissionRefused(
                "terminal_decode_hicache_not_supported",
                DecodeReservationRefusalDisposition.TERMINAL,
                "terminal ownership has no HiCache restore consumer",
            )
        if len(requests) == 0 or len(requests) != len(attempt.child_request_ids):
            raise ValueError("request count differs from reserve child count")
        if any(not isinstance(req, Req) for req in requests):
            raise TypeError("requests must contain canonical Req objects")

        request_ids = tuple(req.rid for req in requests)
        expected_ids = tuple(str(value) for value in attempt.child_request_ids)
        if request_ids != expected_ids:
            raise ValueError("request identities differ from reserve child order")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("prepared request identities must be unique")
        for req in requests:
            if req.req_pool_idx is not None or req.kv is not None:
                raise ValueError("prepared request already owns decoder allocation")
            if req.bootstrap_room is not None:
                raise ValueError("decoder reservation requires an unassigned room")
            if req.bootstrap_host not in (
                None,
                attempt.prefill_bootstrap_endpoint.host,
            ):
                raise ValueError("request bootstrap host differs from reserve attempt")
            if req.bootstrap_port not in (
                None,
                attempt.prefill_bootstrap_endpoint.port,
            ):
                raise ValueError("request bootstrap port differs from reserve attempt")
        return request_ids

    def _validate_preallocated_structural_capacity(
        self,
        requests: tuple[Req, ...],
    ) -> None:
        """Refuse request-slot and metadata exhaustion before prefix matching.

        :param requests: Exact validated canonical request cohort.
        """

        request_count = len(requests)
        if self.req_to_token_pool.available_size() < request_count:
            raise DecodeReservationAdmissionRefused(
                "decode_request_slot_capacity",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        boundary_pool = self._terminal_dflash_boundary_pool
        if boundary_pool is None:
            metadata_allocator = self._require_legacy_metadata_allocator()
            if metadata_allocator.available_size() < request_count:
                raise DecodeReservationAdmissionRefused(
                    "decode_metadata_capacity",
                    DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
                )
        elif boundary_pool.inventory()[0] < request_count:
            raise DecodeReservationAdmissionRefused(
                "decode_dflash_boundary_capacity",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        for req in requests:
            input_len = self._rebootstrap_prefill_len(req)
            if input_len > self.max_total_num_tokens:
                raise DecodeReservationAdmissionRefused(
                    "request_exceeds_decode_capacity",
                    DecodeReservationRefusalDisposition.TERMINAL,
                )
            if self._uses_swa_tail_prealloc():
                _, swa_required = self._prealloc_required_tokens(req)
                if swa_required > self.token_to_kv_pool_allocator.size_swa:
                    raise DecodeReservationAdmissionRefused(
                        "request_exceeds_decode_swa_capacity",
                        DecodeReservationRefusalDisposition.TERMINAL,
                    )

    def _validate_preallocated_capacity(
        self,
        requests: tuple[Req, ...],
        prefix_matches: tuple[DecodePrefixMatch | None, ...] | None = None,
    ) -> None:
        """Refuse a matched cohort whose transfer and restore demand cannot fit.

        :param requests: Exact validated canonical request cohort.
        :param prefix_matches: Reservation-owned decoder prefix matches. ``None``
            preserves conservative full-transfer admission for direct callers.
        """

        self._validate_preallocated_structural_capacity(requests)
        request_count = len(requests)
        if prefix_matches is None:
            prefix_matches = tuple(None for _ in requests)
        if len(prefix_matches) != len(requests):
            raise ValueError("prefix match count differs from request count")

        pending_reserved = self._unpublished_preallocated_child_count()
        transfer_required = sum(
            self._required_alloc_tokens(
                fill_len=self._pre_alloc_fill_len(req),
                prefix_len=(
                    0 if prefix_match is None else prefix_match.decode_prefix_len
                ),
            )
            for req, prefix_match in zip(requests, prefix_matches, strict=True)
        )
        pending_restore_budget = self._hicache_pending_restore_budgets()
        restore_budget = DecodeRestoreBudget(
            full_tokens=(
                pending_restore_budget.full_tokens
                + sum(
                    prefix_match.full_restore_token_count
                    for prefix_match in prefix_matches
                    if prefix_match is not None
                )
            ),
            swa_tokens=(
                pending_restore_budget.swa_tokens
                + sum(
                    prefix_match.swa_restore_token_count
                    for prefix_match in prefix_matches
                    if prefix_match is not None
                )
            ),
        )
        full_allocatable = self._allocatable_token_budgets(
            count_retracted=True,
            extra_reserved_reqs=pending_reserved + request_count,
            hicache_restore_budget=restore_budget,
        )
        if transfer_required > full_allocatable:
            raise DecodeReservationAdmissionRefused(
                "decode_kv_capacity",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )

        if not self._uses_swa_tail_prealloc():
            return
        page_size = self.token_to_kv_pool_allocator.page_size
        swa_required = sum(
            (
                (self._prealloc_kv_lens(req)[1] + page_size - 1)
                // page_size
                * page_size
                if prefix_match is None or prefix_match.decode_prefix_len == 0
                else self._required_alloc_tokens(
                    fill_len=self._pre_alloc_fill_len(req),
                    prefix_len=prefix_match.decode_prefix_len,
                )
            )
            for req, prefix_match in zip(requests, prefix_matches, strict=True)
        )
        active_count = self._active_req_count(
            extra_reserved_reqs=pending_reserved + request_count
        )
        swa_reserve = self.num_reserved_decode_tokens * active_count
        if self.scheduler.server_args.disable_radix_cache:
            swa_reserve = 0
        swa_available = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - restore_budget.swa_tokens
        )
        if swa_required + swa_reserve > swa_available:
            raise DecodeReservationAdmissionRefused(
                "decode_swa_kv_capacity",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )

    def _claim_preallocated_request_ids_locked(
        self,
        grant_id: uuid.UUID,
        request_ids: tuple[str, ...],
        bootstrap_rooms: tuple[int, ...],
    ) -> None:
        """Claim process-lifetime-unique control and transport identities.

        :param grant_id: Exact grant identity.
        :param request_ids: Exact ordered child identities.
        :param bootstrap_rooms: Exact process-global child rooms.
        """

        if (
            grant_id in self._prepared_grant_ids
            or grant_id in self._preparing_grant_ids
        ):
            raise DecodeAllocationLeaseError("grant already owns a prepared cohort")
        if any(
            request_id in self._prepared_request_ids
            or request_id in self._preparing_request_ids
            for request_id in request_ids
        ):
            raise DecodeAllocationLeaseError(
                "request already belongs to a prepared cohort"
            )
        if len(set(bootstrap_rooms)) != len(bootstrap_rooms):
            raise DecodeAllocationLeaseError(
                "prepared cohort contains duplicate bootstrap rooms"
            )
        if any(room in self._seen_bootstrap_rooms for room in bootstrap_rooms):
            raise DecodeReservationAdmissionRefused(
                "decode_bootstrap_room_collision",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        self._preparing_grant_ids.add(grant_id)
        self._preparing_request_ids.update(request_ids)
        self._seen_bootstrap_rooms.update(bootstrap_rooms)

    def _release_preallocated_claim_locked(
        self,
        grant_id: uuid.UUID,
        request_ids: tuple[str, ...],
    ) -> None:
        """Release temporary preparation claims after allocator rollback.

        Bootstrap-room history deliberately survives rollback and terminalization.
        Delayed transport traffic must never alias a later grant in this process.

        :param grant_id: Exact claimed grant.
        :param request_ids: Exact claimed request identities.
        """

        self._preparing_grant_ids.discard(grant_id)
        for request_id in request_ids:
            self._preparing_request_ids.discard(request_id)

    def _require_preallocated_cohort_locked(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> _DecodePreparedCohortRecord:
        """Resolve an exact live handle owned by this queue.

        :param cohort: Candidate opaque cohort.
        :returns: Exact queue-owned cohort record.
        """

        if type(cohort) is not DecodePreparedAllocationCohort:
            raise TypeError("cohort must be DecodePreparedAllocationCohort")
        if cohort._queue_nonce is not self._prepared_cohort_nonce:
            raise DecodeAllocationLeaseError("cohort belongs to another decode queue")
        record = self._prepared_cohorts.get(cohort._token)
        if record is None or record.handle is not cohort:
            raise DecodeAllocationLeaseError("prepared cohort is not registered")
        return record

    def _rollback_preallocated_decode_reqs(
        self,
        decode_reqs: list[DecodeRequest],
        packed_transactions: list[PackedDecodeRequestTransaction],
    ) -> None:
        """Release every child allocation, metadata slot, and receiver.

        :param decode_reqs: Exact unpublished children to release.
        :param packed_transactions: Prepared transaction prefix owned by the
            children in the same order.
        """

        if len(packed_transactions) > len(decode_reqs):
            raise RuntimeError("packed transaction count exceeds prepared children")
        cleanup_failures: list[str] = []
        for child_index in range(len(decode_reqs) - 1, -1, -1):
            decode_req = decode_reqs[child_index]
            packed_transaction = (
                packed_transactions[child_index]
                if child_index < len(packed_transactions)
                else None
            )
            try:
                self._rollback_preallocated_decode_req(
                    decode_req,
                    packed_transaction,
                )
            except Exception:  # noqa: BLE001
                cleanup_failures.append(traceback.format_exc())
        if len(cleanup_failures) > 0:
            raise RuntimeError(
                "prepared cohort rollback failed:\n" + "\n".join(cleanup_failures)
            )

    def _rollback_preallocated_decode_req(
        self,
        decode_req: DecodeRequest,
        packed_transaction: PackedDecodeRequestTransaction | None,
    ) -> None:
        """Release one unpublished reserved child exactly.

        :param decode_req: Exact child to release.
        :param packed_transaction: Transaction owning the lease publication
            boundary, when construction completed.
        """

        lease = decode_req.allocation_lease
        if packed_transaction is not None:
            if lease is None:
                raise DecodeAllocationLeaseError(
                    "packed transaction child lost its allocation lease"
                )
            if packed_transaction.request_owner is not decode_req:
                raise DecodeAllocationLeaseError(
                    "packed transaction belongs to another decode request"
                )
            retired_owner = (
                self.kv_manager.cancel_unpublished_packed_decode_request_transaction(
                    packed_transaction
                )
            )
            if retired_owner is not decode_req:
                raise DecodeAllocationLeaseError(
                    "packed cancellation returned another decode request"
                )
            decode_req.allocation_lease = None
            # The packed auxiliary authority returned this row as part of the
            # same cancellation. Keeping the stale index would double-release
            # a generation which may already have been reissued.
            decode_req.metadata_buffer_index = -1
            lease = None
        elif lease is not None:
            snapshot = self.allocation_lease_authority.snapshot(lease)
            if snapshot.state is not DecodeAllocationLeaseState.PREPARED:
                raise DecodeAllocationLeaseError(
                    "reserved child rollback requires a prepared allocation lease"
                )
            self.allocation_lease_authority.rollback_to_request(lease)

        self._release_preallocated_scheduler_resources(decode_req)

        if lease is not None:
            self.allocation_lease_authority.retire_terminal(lease)
            decode_req.allocation_lease = None

    def _release_cancelled_preallocated_decode_req(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Release scheduler resources after packed authorities retired.

        :param decode_req: Exact request returned by terminal cancellation.
        """

        if decode_req.allocation_lease is None:
            raise DecodeAllocationLeaseError(
                "terminal cancellation lost its prepared allocation lease"
            )
        if decode_req.packed_transaction is not None:
            raise DecodeAllocationLeaseError(
                "unpublished terminal cancellation found attached ownership"
            )
        decode_req.allocation_lease = None
        # Native cancellation already released the adopted auxiliary row.
        decode_req.metadata_buffer_index = -1
        self._release_preallocated_scheduler_resources(decode_req)

    def _release_preallocated_scheduler_resources(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Release request, prefix, row, and receiver ownership still present.

        :param decode_req: Exact unpublished child being reconciled.
        """

        self._abort_preallocated_hicache_prefetch(decode_req)
        try:
            if decode_req.req.req_pool_idx is not None:
                if decode_req.req.kv is None:
                    self._release_unallocated_preallocated_prefix_lock(decode_req)
                    if (
                        isinstance(self.req_to_token_pool, HybridReqToTokenPool)
                        and decode_req.req.mamba_pool_idx is not None
                    ):
                        self.req_to_token_pool.free_mamba_cache(decode_req.req)
                    self.req_to_token_pool.free(decode_req.req)
                else:
                    release_kv_cache(
                        decode_req.req,
                        self.tree_cache,
                        is_insert=False,
                    )
            else:
                self._release_unallocated_preallocated_prefix_lock(decode_req)
        finally:
            decode_req.prefix_match = None
            try:
                if decode_req.metadata_buffer_index != -1:
                    self._require_legacy_metadata_allocator().free(
                        decode_req.metadata_buffer_index
                    )
                    decode_req.metadata_buffer_index = -1
            finally:
                if decode_req.kv_receiver is not None:
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None

    def _release_unallocated_preallocated_prefix_lock(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Release a matched-node lock when no request KV owner can do so.

        :param decode_req: Prepared request without a complete KV allocation.
        """

        prefix_match = decode_req.prefix_match
        if prefix_match is None:
            return
        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)
        decode_req.req.last_node = None

    def _quarantine_preallocated_record_locked(
        self,
        record: _DecodePreparedCohortRecord,
        reason: str,
    ) -> int:
        """Retain all live child owners and remove runnable queue visibility.

        :param record: Exact live cohort record.
        :param reason: First stable quarantine reason.
        :returns: Number of terminal callback registrations removed.
        """

        removed_count = 0
        if record.state in (
            _DecodePreparedCohortState.ATTACHED,
            _DecodePreparedCohortState.QUARANTINED,
        ) and self._terminal_decode_serving is not None:
            cleanup_failures: list[str] = []
            for decode_req, transaction in zip(
                record.decode_reqs,
                record.packed_transactions,
                strict=True,
            ):
                try:
                    removed = self.transfer_queue.quarantine_terminal_request(
                        decode_req,
                        transaction,
                    )
                    removed_count += int(removed)
                except Exception:  # noqa: BLE001
                    cleanup_failures.append(traceback.format_exc())
            if len(cleanup_failures) > 0:
                logger.critical(
                    "Terminal registry cleanup failed while quarantining a "
                    "prepared cohort:\n%s",
                    "\n".join(cleanup_failures),
                )
        if record.state is _DecodePreparedCohortState.QUARANTINED:
            return removed_count
        if record.state in (
            _DecodePreparedCohortState.CANCELLED,
            _DecodePreparedCohortState.COMPLETED,
            _DecodePreparedCohortState.ABORTED,
        ):
            raise DecodeAllocationLeaseError(
                f"cohort quarantine is invalid in state {record.state.value}"
            )
        record.state = _DecodePreparedCohortState.QUARANTINED
        record.quarantine_reason = reason
        self.queue = [
            entry
            for entry in self.queue
            if all(entry is not owned for owned in record.decode_reqs)
        ]
        self.pending_reqs = [
            entry
            for entry in self.pending_reqs
            if all(entry is not owned for owned in record.decode_reqs)
        ]
        for decode_req, packed_transaction in zip(
            record.decode_reqs,
            record.packed_transactions,
            strict=True,
        ):
            lease = decode_req.allocation_lease
            if lease is not None:
                snapshot = self.allocation_lease_authority.snapshot(lease)
                releasable_state = snapshot.state in (
                    DecodeAllocationLeaseState.COMMITTED_TO_REQUEST,
                    DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST,
                    DecodeAllocationLeaseState.ABORT_AUTHORIZED,
                )
                if (
                    snapshot.state is not DecodeAllocationLeaseState.QUARANTINED
                    and not releasable_state
                ):
                    packed_transaction.quarantine(reason)
            if decode_req.kv_receiver is not None:
                decode_req.kv_receiver.abort()
        return removed_count

    def _retire_preallocated_record_locked(
        self,
        record: _DecodePreparedCohortRecord,
    ) -> None:
        """Forget one safely terminal non-quarantined cohort.

        :param record: Exact terminal cohort record.
        """

        token = record.handle._token
        if self._prepared_cohorts.get(token) is not record:
            raise DecodeAllocationLeaseError("prepared cohort record is not live")
        del self._prepared_cohorts[token]
        del self._prepared_grant_ids[record.grant_id]
        for decode_req in record.decode_reqs:
            owner = self._prepared_request_ids.get(decode_req.req.rid)
            if owner is not token:
                raise DecodeAllocationLeaseError(
                    "prepared request ownership changed before retirement"
                )
            del self._prepared_request_ids[decode_req.req.rid]
            bootstrap_room = decode_req.req.bootstrap_room
            if bootstrap_room is None:
                raise DecodeAllocationLeaseError(
                    "prepared request lost its process-global bootstrap room"
                )
            if bootstrap_room not in self._seen_bootstrap_rooms:
                raise DecodeAllocationLeaseError(
                    "prepared request lost its process-global room history"
                )

    def _unpublished_preallocated_child_count(self) -> int:
        """Return live reserved children not yet represented by transfer queues.

        :returns: Conservative pending child count for decode reserve budgeting.
        """

        lock = getattr(self, "_prepared_cohort_lock", None)
        if lock is None:
            return 0
        with lock:
            return sum(
                len(record.decode_reqs)
                for record in self._prepared_cohorts.values()
                if record.state
                in (
                    _DecodePreparedCohortState.PREPARED,
                    _DecodePreparedCohortState.PROMOTED,
                    _DecodePreparedCohortState.ATTACHED,
                )
                and not record.metadata_published
            )

    def _unpublished_preallocated_prefix_matches(
        self,
    ) -> tuple[DecodePrefixMatch, ...]:
        """Return restore intents retained outside the transfer queue.

        :returns: Prefix matches whose promised restores still need device space.
        """

        lock = getattr(self, "_prepared_cohort_lock", None)
        if lock is None:
            return ()
        with lock:
            return tuple(
                decode_req.prefix_match
                for record in self._prepared_cohorts.values()
                if record.state
                in (
                    _DecodePreparedCohortState.PREPARED,
                    _DecodePreparedCohortState.PROMOTED,
                    _DecodePreparedCohortState.ATTACHED,
                )
                and not record.metadata_published
                for decode_req in record.decode_reqs
                if decode_req.prefix_match is not None
                and decode_req.hicache_restore_status is HiCacheRestoreResult.PENDING
                and decode_req.hicache_restored_node is None
            )

    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:
        """
        Match a request against the decode-side radix cache, lock the matched
        node to prevent eviction, and return the matched prefix information.
        """
        result = match_prefix_for_req(
            self.tree_cache,
            req,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            include_req=True,
        )
        # Always lock to match aggregated scheduling behavior
        self.tree_cache.inc_lock_ref(result.last_device_node)
        try:
            return self._build_decode_prefix_match(req, result)
        except Exception:  # noqa: BLE001
            logger.error(
                "Decoder prefix-match construction failed for %s:\n%s",
                req.rid,
                traceback.format_exc(),
            )
            self.tree_cache.dec_lock_ref(result.last_device_node)
            raise

    def _match_preallocated_prefix_and_lock(
        self,
        req: Req,
    ) -> DecodePrefixMatch | None:
        """Acquire the decoder prefix ownership retained by a reservation.

        :param req: Exact request entering prepared admission.
        :returns: Locked decoder match, or ``None`` when radix reuse is disabled.
        """

        if (
            not self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            or getattr(req, "pd_rebootstrap_in_progress", False)
        ):
            return None
        return self._match_prefix_and_lock(req)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver(
        self,
        req: Req,
        *,
        is_rebootstrap: bool = False,
    ) -> DecodeRequest:
        """Create one decode receiver without publishing queue ownership.

        :param req: Exact canonical scheduler request.
        :param is_rebootstrap: Whether prefill recomputes a retracted request.
        :returns: Unenqueued decode request and receiver.
        """

        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        return DecodeRequest(
            req=req, kv_receiver=kv_receiver, is_rebootstrap=is_rebootstrap
        )

    def _create_receiver_and_enqueue(
        self, req: Req, is_rebootstrap: bool = False
    ) -> DecodeRequest:
        decode_req = self._create_receiver(
            req,
            is_rebootstrap=is_rebootstrap,
        )
        self.queue.append(decode_req)
        return decode_req

    def hold_rebootstrap(self, req: Req) -> None:
        """Stage a retracted request for rebootstrap without enqueuing it yet.

        Retraction is always paired with a weight update
        (``pause_generation(mode="retract")`` -> ``update_weights`` ->
        ``continue_generation``). Enqueuing the rebootstrap into ``self.queue``
        here would leave the preallocation queue non-empty, which makes the
        scheduler non-idle so ``update_weights``' post-update cache flush
        asserts and crashes the decode worker. Instead we hold the request and
        enqueue it from ``enqueue_held_rebootstrap`` on resume, so its prefix KV
        is recomputed by the prefill worker under the updated weights.
        """
        self.held_rebootstrap_reqs.append(req)

    def enqueue_held_rebootstrap(self) -> None:
        """Enqueue all staged rebootstrap requests when generation resumes."""
        held = self.held_rebootstrap_reqs
        self.held_rebootstrap_reqs = []
        for req in held:
            self.add(req, is_rebootstrap=True)

    @staticmethod
    def _rebootstrap_prefill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids)

    @staticmethod
    def _pre_alloc_fill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            # pause_generation(retract) already popped the boundary token out of
            # output_ids (it is replayed via the decode-side override at commit
            # time), so output_ids here is prompt + emitted-tokens-minus-boundary,
            # i.e. the original seqlen - 1. The prefill recomputes KV for *all* of
            # these tokens, leaving no just-sampled "pending" token in the list, so
            # we allocate exactly len(origin)+len(output_ids) with no -1 (unlike
            # normal decode, where the last token's KV has not been written yet).
            # This is the same token count as offloading-based retraction, where
            # offload_kv_cache saves seqlen-1 tokens; the boundary token's KV is
            # (re)computed on the decode side once generation resumes.
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        input_len = self._rebootstrap_prefill_len(req)
        if input_len > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {input_len} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        if self._uses_swa_tail_prealloc():
            _, swa_required = self._prealloc_required_tokens(req)
            swa_capacity = self.token_to_kv_pool_allocator.size_swa
            if swa_required > swa_capacity:
                message = (
                    f"Request {req.rid} requires too many SWA KV tokens for "
                    f"decode preallocation: {swa_required} > {swa_capacity}"
                )
                logger.error(message)
                prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self.queue.clear()
        self.retracted_queue.clear()
        self.tp1_poll_progress_policy.mark_idle()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        if uses_swa_tail_prealloc:
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if full_required > full_allocatable_tokens:
                break
            if uses_swa_tail_prealloc and swa_required > swa_allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens -= full_required
            if uses_swa_tail_prealloc:
                swa_allocatable_tokens -= swa_required

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            self.tp1_poll_progress_policy.mark_idle()
            return

        # Still poll if any receiver was aborted, otherwise it stays stuck.
        if all(decode_req.waiting_for_input for decode_req in self.queue) and not any(
            decode_req.kv_receiver.conclude_state == KVPoll.Failed
            for decode_req in self.queue
        ):
            self.tp1_poll_progress_policy.mark_idle()
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue],
            self.gloo_group,
            singleton_progress_policy=self.tp1_poll_progress_policy,
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    # kv_receiver may be None from a prior self.queue cleanup
                    if decode_req.kv_receiver is not None:
                        decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Prepare and publish one preallocation cohort transactionally.

        :param rids_to_check: Optional request IDs eligible for this pass.
        :returns: Preallocated and failed decode requests.
        """

        preparation = _DecodeAllocationPreparation()
        try:
            return self._prepare_and_publish_preallocated(
                rids_to_check,
                preparation,
            )
        except Exception:
            preparation_traceback = traceback.format_exc()
            prepared_count = len(preparation.prepared_decode_reqs)
            if preparation.publication_started:
                logger.critical(
                    "Decode preallocation failed after external publication began; "
                    "retaining %d prepared allocation leases:\n%s",
                    prepared_count,
                    preparation_traceback,
                )
                raise

            logger.error(
                "Decode preallocation failed before metadata publication; "
                "rolling back %d prepared allocation leases:\n%s",
                prepared_count,
                preparation_traceback,
            )
            try:
                self._rollback_decode_allocation_leases(
                    preparation.prepared_decode_reqs
                )
            except Exception:
                logger.critical(
                    "Decode allocation rollback failed. Preparation traceback:\n"
                    "%s\nRollback traceback:\n%s",
                    preparation_traceback,
                    traceback.format_exc(),
                )
                raise
            raise

    def _prepare_and_publish_preallocated(
        self,
        rids_to_check: Optional[List[str]],
        preparation: _DecodeAllocationPreparation,
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Execute one cohort through the first metadata publication.

        :param rids_to_check: Optional request IDs eligible for this pass.
        :param preparation: Transaction state shared with the exception boundary.
        :returns: Preallocated and failed decode requests.
        """

        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        metadata_submissions: list[_DecodeMetadataSubmission] = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        pending_restore_budget = self._hicache_pending_restore_budgets()
        swa_allocatable_tokens = 0
        if uses_swa_tail_prealloc:
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                    hicache_restore_budget=pending_restore_budget,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                hicache_restore_budget=pending_restore_budget,
            )
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                if not getattr(decode_req.req, "finished_output", False):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # DecodeRequest is shared between self.queue and self.pending_reqs;
        # drop failed reqs from both
        if failed_reqs:
            failed_ids = {id(r) for r in failed_reqs}
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.live_requests()),
            )

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            prepared_record = self._attached_preallocated_record(decode_req)
            if prepared_record is not None:
                preparation.publication_started = True
                try:
                    actual_source_tp_size = (
                        decode_req.kv_receiver.prefill_info.attn_tp_size
                    )
                    if actual_source_tp_size != prepared_record.source_tp_size:
                        raise DecodeAllocationLeaseError(
                            "handshake source TP width differs from reservation"
                        )
                    lease = decode_req.allocation_lease
                    if lease is None:
                        raise DecodeAllocationLeaseError(
                            "attached reserved child lost its allocation lease"
                        )
                    lease_snapshot = self.allocation_lease_authority.snapshot(lease)
                    if lease_snapshot.state is not DecodeAllocationLeaseState.PUBLISHED:
                        raise DecodeAllocationLeaseError(
                            "attached reserved allocation was not promoted"
                        )
                    origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
                    prefix_match = decode_req.prefix_match
                    prefix_len = (
                        0 if prefix_match is None else prefix_match.l1_prefix_len
                    )
                    total_prefix_len = (
                        0 if prefix_match is None else prefix_match.decode_prefix_len
                    )
                    decode_req.req.cache_protected_len = total_prefix_len
                    metadata_submissions.append(
                        self._build_decode_metadata_submission(
                            decode_req,
                            origin_input_len=origin_input_len,
                            prefix_len=prefix_len,
                            total_prefix_len=total_prefix_len,
                            dst_kv_indices=None,
                            allocate_metadata_index=False,
                        )
                    )
                except Exception:  # noqa: BLE001
                    metadata_traceback = traceback.format_exc()
                    with self._prepared_cohort_lock:
                        self._quarantine_preallocated_record_locked(
                            prepared_record,
                            "reserved metadata preparation failed",
                        )
                    logger.error(
                        "Reserved metadata preparation failed and was quarantined:\n%s",
                        metadata_traceback,
                    )
                    raise
                preallocated_reqs.append(decode_req)
                indices_to_remove.add(i)
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            scheduler_local_fake = _is_fake_transfer(
                decode_req.req,
                self.scheduler.server_args,
            )
            if (
                not scheduler_local_fake
                and self._require_legacy_metadata_allocator().available_size() <= 0
            ):
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
            prefix_match: Optional[DecodePrefixMatch] = None
            matched_restore_budget = DecodeRestoreBudget()
            admission_restore_budget = pending_restore_budget
            use_decode_radix_cache = (
                self.scheduler.server_args.disaggregation_decode_enable_radix_cache
                and not decode_req.is_rebootstrap
                and not scheduler_local_fake
            )
            if use_decode_radix_cache:
                # Match prefix against decode's radix cache.
                prefix_match = self._match_prefix_and_lock(decode_req.req)
                prefix_indices = prefix_match.prefix_indices
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                prefix_len = prefix_match.l1_prefix_len
                total_prefix_len = prefix_match.decode_prefix_len
                matched_restore_budget = DecodeRestoreBudget(
                    full_tokens=prefix_match.full_restore_token_count,
                    swa_tokens=prefix_match.swa_restore_token_count,
                )

                fill_len = self._pre_alloc_fill_len(decode_req.req)
                joint_swa_prealloc = (
                    isinstance(
                        self.token_to_kv_pool_allocator,
                        SWATokenToKVPoolAllocator,
                    )
                    and not uses_swa_tail_prealloc
                )
                if joint_swa_prealloc:
                    admission_restore_budget = DecodeRestoreBudget(
                        full_tokens=(
                            pending_restore_budget.full_tokens
                            + matched_restore_budget.full_tokens
                        ),
                        swa_tokens=(
                            pending_restore_budget.swa_tokens
                            + matched_restore_budget.swa_tokens
                        ),
                    )
                    capacity_prefix_len = total_prefix_len
                else:
                    capacity_prefix_len = prefix_len
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len,
                    prefix_len=capacity_prefix_len,
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                    hicache_restore_budget=admission_restore_budget,
                )
            else:
                prefix_indices = None
                prefix_len = 0
                total_prefix_len = 0
                capacity_prefix_len = 0
                required_alloc_tokens = self._pre_alloc_fill_len(decode_req.req)

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    - capacity_prefix_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > full_allocatable_tokens
            ):
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break
            if required_tokens_for_request > full_allocatable_tokens:
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                break

            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                if (
                    max(
                        swa_required,
                        swa_len + max_new_tokens - retractable_swa_tokens,
                    )
                    > swa_allocatable_tokens
                ):
                    if prefix_len > 0:
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    break

            if total_prefix_len != 0 and hasattr(
                self.token_to_kv_pool_allocator, "c4_attn_allocator"
            ):
                if prefix_len > 0:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                raise RuntimeError(
                    "DSV4 NPU PD disaggregation does not support decode-side "
                    "prefix cache yet; disable disaggregation decode radix/HiCache "
                    "for PD + chunked prefill."
                )

            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
                decode_req=decode_req,
                migration_end=origin_input_len,
            )
            preparation.record_prepared(decode_req)
            decode_req.prefix_match = prefix_match
            if self.scheduler.enable_decode_hicache:
                self._start_hicache_prefetch(decode_req.req, prefix_match)
            hisparse_req_budget -= 1
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            if prefix_match is not None:
                pending_restore_budget = DecodeRestoreBudget(
                    full_tokens=(
                        pending_restore_budget.full_tokens
                        + matched_restore_budget.full_tokens
                    ),
                    swa_tokens=(
                        pending_restore_budget.swa_tokens
                        + matched_restore_budget.swa_tokens
                    ),
                )
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=len(preallocated_reqs) + 1,
                hicache_restore_budget=pending_restore_budget,
            )
            if uses_swa_tail_prealloc:
                # SWA budget uses simple decrement (no radix cache eviction in
                # the SWA pool, so page-rounding drift is negligible).
                swa_allocatable_tokens -= swa_required
            decode_req.req.cache_protected_len = total_prefix_len

            if scheduler_local_fake:
                preallocated_reqs.append(decode_req)
                indices_to_remove.add(i)
                continue

            metadata_submissions.append(
                self._build_decode_metadata_submission(
                    decode_req,
                    origin_input_len=origin_input_len,
                    prefix_len=prefix_len,
                    total_prefix_len=total_prefix_len,
                    dst_kv_indices=dst_kv_indices,
                    allocate_metadata_index=True,
                )
            )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)

        for submission in metadata_submissions:
            decode_req = submission.decode_req
            prepared_record = self._attached_preallocated_record(decode_req)
            try:
                packed_child: (
                    tuple[PackedDecodeRequestTransaction, PackedRequestPublication]
                    | None
                ) = None
                if prepared_record is None:
                    self._record_legacy_decode_allocation_publication(decode_req)
                else:
                    packed_child = self._packed_preallocated_child(
                        prepared_record,
                        decode_req,
                    )
                preparation.publication_started = True
                if (
                    packed_child is None
                    and self.transfer_queue.enable_staging
                    and decode_req.kv_receiver.require_staging
                ):
                    self.transfer_queue.staging_handler.register_decode_req(
                        decode_req.req.bootstrap_room, decode_req
                    )
                if packed_child is None:
                    decode_req.kv_receiver.send_metadata(
                        submission.page_indices,
                        decode_req.metadata_buffer_index,
                        submission.state_indices,
                        decode_prefix_len=submission.decode_prefix_len,
                    )
                else:
                    self.kv_manager.send_packed_decode_request_metadata(
                        transaction=packed_child[0],
                        publication=packed_child[1],
                        receiver=decode_req.kv_receiver,
                        page_indices=submission.page_indices,
                        metadata_buffer_index=decode_req.metadata_buffer_index,
                        state_indices=submission.state_indices,
                        decode_prefix_len=submission.decode_prefix_len,
                    )
                if decode_req.is_rebootstrap:
                    self.kv_manager.submit_prefill_recompute(
                        decode_req.kv_receiver,
                        decode_req.req.build_rebootstrap_payload(),
                    )
                decode_req.req.time_stats.set_decode_transfer_queue_entry_time()
            except Exception:  # noqa: BLE001
                publication_traceback = traceback.format_exc()
                if prepared_record is not None:
                    with self._prepared_cohort_lock:
                        self._quarantine_preallocated_record_locked(
                            prepared_record,
                            "reserved metadata publication failed",
                        )
                    logger.error(
                        "Reserved metadata publication failed and was quarantined:\n%s",
                        publication_traceback,
                    )
                raise
            if prepared_record is not None:
                with self._prepared_cohort_lock:
                    prepared_record.metadata_published = True

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    def _build_decode_metadata_submission(
        self,
        decode_req: DecodeRequest,
        *,
        origin_input_len: int,
        prefix_len: int,
        total_prefix_len: int,
        dst_kv_indices: torch.Tensor | None,
        allocate_metadata_index: bool,
    ) -> _DecodeMetadataSubmission:
        """Build one destination metadata payload from an exact allocation.

        :param decode_req: Exact allocated decode request.
        :param origin_input_len: Prompt length transferred by prefill.
        :param prefix_len: L1-resident prefix length.
        :param total_prefix_len: Full decoder-reused prefix length.
        :param dst_kv_indices: Direct-to-host destination rows, when applicable.
        :param allocate_metadata_index: Whether this path must reserve a slot.
        :returns: Complete deferred metadata submission.
        """

        page_size = self.token_to_kv_pool_allocator.page_size
        kv_transfer_page_size = page_size
        if self.scheduler.enable_hisparse:
            if dst_kv_indices is None:
                raise RuntimeError("HiSparse metadata requires destination indices")
            kv_transfer_page_size = getattr(
                self.token_to_kv_pool_allocator,
                "hisparse_page_size",
                page_size,
            )
            kv_indices = dst_kv_indices[: origin_input_len - prefix_len]
        else:
            kv_indices = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx
            ][total_prefix_len:origin_input_len]

        seq_len = origin_input_len

        def _mamba_payload() -> list[np.ndarray]:
            return [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    decode_req.req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]

        def _swa_payload() -> np.ndarray:
            window_size = self.scheduler.sliding_window_size
            window_start = max(0, seq_len - window_size)
            window_start = page_align_floor(window_start, page_size)
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx, window_start:seq_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            return kv_to_page_indices(window_kv_indices_swa, page_size)

        def _dsa_payload() -> np.ndarray:
            kv_indices_full = self.req_to_token_pool.req_to_token[
                decode_req.req.req_pool_idx, :seq_len
            ]
            device_page_size = self.token_to_kv_pool.page_size
            return kv_to_page_indices(kv_indices_full, device_page_size)

        def _swa_ring_payload() -> np.ndarray:
            ring_stride = self.token_to_kv_pool.unified_swa_ring_size
            window_size = self.token_to_kv_pool.unified_swa_window
            window_start = max(0, seq_len - window_size)
            positions = np.arange(window_start, seq_len, dtype=np.int64)
            state_slot = int(decode_req.req.req_pool_idx)
            ring_rows = state_slot * ring_stride + (positions % ring_stride)
            return ring_rows.astype(np.int32)

        def _c128_state_payload() -> np.ndarray:
            online = is_dsv4_c128_online_enabled()
            ring_size = 1 if online else self.token_to_kv_pool.get_ring_size(128)
            return get_dsv4_c128_state_indices(
                int(decode_req.req.req_pool_idx),
                seq_len,
                online=online,
                ring_size=ring_size,
            )

        state_types = self.kv_manager.kv_args.state_types
        if StateType.C128_STATE in state_types:
            clear_c128_state = getattr(
                self.token_to_kv_pool,
                "clear_c128_req_state",
                None,
            )
            if clear_c128_state is not None:
                clear_c128_state(int(decode_req.req.req_pool_idx))
        payloads = {
            StateType.MAMBA: _mamba_payload,
            StateType.SWA: _swa_payload,
            StateType.DSA: _dsa_payload,
            StateType.MINIMAX_INDEX_K: _dsa_payload,
            StateType.SWA_RING: _swa_ring_payload,
            StateType.C128_STATE: _c128_state_payload,
        }
        if hasattr(self.req_to_token_pool, "req_to_token_c4"):
            if total_prefix_len != 0:
                raise RuntimeError(
                    "DSV4 NPU PD disaggregation does not support decode-side "
                    "prefix cache yet; disable disaggregation decode radix/HiCache "
                    "for PD + chunked prefill."
                )
        if _is_npu and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                dsv4_state_payloads,
            )

            payloads.update(
                dsv4_state_payloads(
                    self.req_to_token_pool,
                    decode_req.req.req_pool_idx,
                    seq_len,
                    self.token_to_kv_pool_allocator.page_size,
                    self.scheduler.sliding_window_size,
                    prefix_len=total_prefix_len,
                )
            )
        state_indices: list[Any] | None = [
            payloads[st]() if st in payloads else None for st in state_types
        ]

        if allocate_metadata_index:
            decode_req.metadata_buffer_index = (
                self._require_legacy_metadata_allocator().alloc()
            )
            if decode_req.metadata_buffer_index is None:
                raise RuntimeError("decode metadata allocator returned no slot")
        elif decode_req.metadata_buffer_index < 0:
            raise DecodeAllocationLeaseError(
                "reserved decode request lost its metadata slot"
            )
        page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size).astype(
            np.int32
        )
        return _DecodeMetadataSubmission(
            decode_req=decode_req,
            page_indices=page_indices,
            state_indices=state_indices,
            decode_prefix_len=total_prefix_len,
        )

    def _attached_preallocated_record(
        self,
        decode_req: DecodeRequest,
    ) -> _DecodePreparedCohortRecord | None:
        """Resolve attached reserved ownership for one exact queue child.

        :param decode_req: Candidate decode queue entry.
        :returns: Exact attached cohort, otherwise ``None``.
        """

        lock = getattr(self, "_prepared_cohort_lock", None)
        if lock is None:
            return None
        with lock:
            token = self._prepared_request_ids.get(decode_req.req.rid)
            if token is None:
                return None
            record = self._prepared_cohorts.get(token)
            if record is None:
                raise DecodeAllocationLeaseError(
                    "prepared request points to a missing cohort"
                )
            if all(owned is not decode_req for owned in record.decode_reqs):
                raise DecodeAllocationLeaseError(
                    "prepared request identity differs from its cohort child"
                )
            if record.state is not _DecodePreparedCohortState.ATTACHED:
                raise DecodeAllocationLeaseError(
                    f"prepared request entered the queue in state {record.state.value}"
                )
            return record

    @staticmethod
    def _packed_preallocated_child(
        record: _DecodePreparedCohortRecord,
        decode_req: DecodeRequest,
    ) -> tuple[PackedDecodeRequestTransaction, PackedRequestPublication]:
        """Resolve exact packed transfer ownership for an attached child.

        :param record: Attached prepared cohort.
        :param decode_req: Exact child entering metadata transfer.
        :returns: Matching request transaction and irreversible publication.
        """

        publications = record.packed_publications
        if publications is None:
            raise DecodeAllocationLeaseError(
                "attached prepared cohort has no packed publications"
            )
        if len(publications) != len(record.decode_reqs):
            raise DecodeAllocationLeaseError(
                "prepared cohort publication count differs from its children"
            )
        for owned_req, transaction, publication in zip(
            record.decode_reqs,
            record.packed_transactions,
            publications,
            strict=True,
        ):
            if owned_req is not decode_req:
                continue
            if transaction.request_owner is not decode_req:
                raise DecodeAllocationLeaseError(
                    "packed transaction retained another decode request"
                )
            return transaction, publication
        raise DecodeAllocationLeaseError(
            "prepared cohort does not own the metadata transfer child"
        )

    def _record_legacy_decode_allocation_publication(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Publish an allocation owned by the legacy transfer path.

        Prepared-grant allocations never enter this method. Their packed
        transaction crossed the publication boundary during promotion.

        :param decode_req: Legacy decode request about to publish metadata.
        """

        lease = decode_req.allocation_lease
        if lease is None:
            return
        snapshot = self.allocation_lease_authority.snapshot(lease)
        if snapshot.state is DecodeAllocationLeaseState.PUBLISHED:
            return
        if snapshot.state is not DecodeAllocationLeaseState.PREPARED:
            raise DecodeAllocationLeaseError(
                f"metadata publication is invalid in state {snapshot.state.value}"
            )
        self.allocation_lease_authority.record_publication(
            lease,
            self.allocation_lifecycle_authority,
        )

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            decode_req.req.extend_range.end
            for decode_req in self.transfer_queue.live_requests()
        )

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + len(self.transfer_queue.live_requests())
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        hicache_restore_budget: DecodeRestoreBudget | None = None,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
            hicache_restore_budget=hicache_restore_budget,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
            hicache_restore_budget=hicache_restore_budget,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_restore_budget: DecodeRestoreBudget | None = None,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )
        restore_budget = (
            hicache_restore_budget
            if hicache_restore_budget is not None
            else DecodeRestoreBudget()
        )

        if self.scheduler.enable_hisparse:
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            if self._uses_swa_tail_prealloc() and hasattr(
                logical_allocator, "full_available_size"
            ):
                available_size = logical_allocator.full_available_size()
            else:
                # HiSparse pre-alloc only allocates logical indices, so the
                # logical pool is the binding constraint for admission control.
                available_size = logical_allocator.available_size()
            available_size -= restore_budget.full_tokens
        elif isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            full_available_size = (
                self.token_to_kv_pool_allocator.full_available_size()
                - restore_budget.full_tokens
            )
            if self._uses_swa_tail_prealloc():
                available_size = full_available_size
            else:
                swa_available_size = (
                    self.token_to_kv_pool_allocator.swa_available_size()
                    - restore_budget.swa_tokens
                )
                available_size = min(full_available_size, swa_available_size)
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        else:
            available_size = (
                self.token_to_kv_pool_allocator.available_size()
                - restore_budget.full_tokens
            )
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
        hicache_restore_budget: DecodeRestoreBudget | None = None,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count()
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        swa_total = self.token_to_kv_pool_allocator.size_swa
        restore_budget = (
            hicache_restore_budget
            if hicache_restore_budget is not None
            else DecodeRestoreBudget()
        )
        swa_available = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - restore_budget.swa_tokens
        )
        swa_used = swa_total - swa_available
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = swa_available - max(
            swa_reserved_tokens, need_swa_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _migration_component_claims(
        self,
        decode_req: DecodeRequest,
        *,
        migration_start: int,
        migration_end: int,
    ) -> tuple[DecodeAllocationComponentClaim, ...]:
        """Build exact FULL, SWA, and Mamba claims from live request mappings.

        :param decode_req: Request whose allocation is final.
        :param migration_start: First prompt position written by prefill.
        :param migration_end: End of the transferred prompt.
        :returns: Canonically ordered component claims.
        """

        req = decode_req.req
        request_slot = req.req_pool_idx
        if request_slot is None:
            raise DecodeAllocationLeaseError(
                "decode request has no request-pool allocation"
            )
        if migration_start < 0 or migration_end < migration_start:
            raise DecodeAllocationLeaseError("invalid migration-owned logical range")

        full_indices = self.req_to_token_pool.req_to_token[
            request_slot, migration_start:migration_end
        ]
        if full_indices.numel() > 0 and bool((full_indices <= 0).any().item()):
            raise DecodeAllocationLeaseError(
                "migration-owned FULL mapping is incomplete"
            )

        allocator = self.token_to_kv_pool_allocator
        if isinstance(allocator, SWATokenToKVPoolAllocator):
            full_allocator = allocator.full_attn_allocator
            window_size = self.scheduler.sliding_window_size
            if window_size is None or window_size <= 0:
                swa_start = migration_start
            else:
                swa_start = max(migration_start, migration_end - window_size)
            swa_full_indices = self.req_to_token_pool.req_to_token[
                request_slot, swa_start:migration_end
            ]
            if isinstance(allocator, UnifiedSWATokenToKVPoolAllocator):
                swa_indices = swa_full_indices
            else:
                swa_indices = allocator.translate_loc_from_full_to_swa(swa_full_indices)
            if swa_indices.numel() > 0 and bool((swa_indices <= 0).any().item()):
                raise DecodeAllocationLeaseError(
                    "migration-owned SWA mapping is incomplete"
                )
            swa_claim = DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.SWA,
                logical_start=swa_start,
                logical_length=int(swa_indices.numel()),
                allocator=(
                    allocator.swa_attn_allocator if swa_indices.numel() > 0 else None
                ),
                indices=swa_indices if swa_indices.numel() > 0 else None,
            )
        else:
            full_allocator = allocator
            swa_claim = DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.SWA,
                logical_start=migration_start,
                logical_length=0,
                allocator=None,
                indices=None,
            )

        model_type = self.scheduler.model_config.hf_config.model_type
        if model_type in ("gemma4", "gemma4_unified"):
            mamba_claim = DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.MAMBA,
                logical_start=0,
                logical_length=0,
                allocator=None,
                indices=None,
            )
        elif isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            if req.mamba_pool_idx is None:
                raise DecodeAllocationLeaseError(
                    "hybrid decode request has no Mamba allocation"
                )
            mamba_indices = req.mamba_pool_idx.reshape(1)
            mamba_claim = DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.MAMBA,
                logical_start=0,
                logical_length=1,
                allocator=self.req_to_token_pool.mamba_allocator,
                indices=mamba_indices,
            )
        else:
            mamba_claim = DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.MAMBA,
                logical_start=0,
                logical_length=0,
                allocator=None,
                indices=None,
            )

        full_claim = DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.FULL,
            logical_start=migration_start,
            logical_length=int(full_indices.numel()),
            allocator=full_allocator if full_indices.numel() > 0 else None,
            indices=full_indices if full_indices.numel() > 0 else None,
        )
        return full_claim, swa_claim, mamba_claim

    def _acquire_decode_allocation_lease(
        self,
        decode_req: DecodeRequest,
        *,
        migration_start: int,
        migration_end: int,
        source_tp_size: int | None = None,
    ) -> None:
        """Acquire the process-local lease for one staged TP transfer.

        :param decode_req: Exact live decode request.
        :param migration_start: First prompt position written by prefill.
        :param migration_end: End of the transferred prompt.
        :param source_tp_size: Decoder-authorized source width before handshake.
        """

        if decode_req.allocation_lease is not None:
            raise DecodeAllocationLeaseError(
                "decode request already owns an allocation lease"
            )

        if source_tp_size is None:
            if not decode_req.kv_receiver.require_staging:
                return
            source_tp_size = decode_req.kv_receiver.prefill_info.attn_tp_size
        if source_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES:
            raise DecodeAllocationLeaseError(
                "decode allocation leases require a supported packed source TP"
            )
        destination_tp_size = self.kv_manager.attn_tp_size
        destination_tp_rank = self.kv_manager.attn_tp_rank
        if (
            source_tp_size < destination_tp_size
            or source_tp_size % destination_tp_size != 0
        ):
            raise DecodeAllocationLeaseError(
                "decode allocation leases require source attention TP width "
                "divisible by destination attention TP width"
            )
        request_slot = decode_req.req.req_pool_idx
        if request_slot is None:
            raise DecodeAllocationLeaseError(
                "decode request has no request-pool allocation"
            )
        request_generation = int(
            self.req_to_token_pool.req_generation[request_slot].item()
        )
        decode_req.allocation_lease = self.allocation_lease_authority.acquire(
            request_pool=self.req_to_token_pool,
            request_slot=request_slot,
            expected_request_generation=request_generation,
            writer_manifest=DecodeWriterManifest.for_tensor_parallel(
                source_tp_size,
                destination_tp_size,
                destination_tp_rank,
            ),
            component_claims=self._migration_component_claims(
                decode_req,
                migration_start=migration_start,
                migration_end=migration_end,
            ),
        )

    def _rollback_decode_allocation_leases(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> None:
        """Rollback every still-prepared child lease in reverse order.

        :param decode_reqs: Newly prepared cohort.
        """

        for decode_req in reversed(decode_reqs):
            lease = decode_req.allocation_lease
            if lease is None:
                continue
            self.allocation_lease_authority.rollback_to_request(lease)
            self.allocation_lease_authority.retire_terminal(lease)
            decode_req.allocation_lease = None

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
        *,
        decode_req: DecodeRequest | None = None,
        migration_end: int | None = None,
        source_tp_size: int | None = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.

        :param req: Exact canonical scheduler request.
        :param prefix_indices: Existing L1 prefix destinations.
        :param prefix_len: Existing L1 prefix length.
        :param total_prefix_len: Full promised decode prefix length.
        :param decode_req: Optional migration-owning decode request.
        :param migration_end: Exact transferred prompt end.
        :param source_tp_size: Decoder-authorized source width before handshake.
        :returns: Newly allocated transfer destination indices.
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        fill_len = self._pre_alloc_fill_len(req)
        req.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if (
            self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            and self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens
        ):
            num_to_evict = (
                required_alloc_tokens - self.token_to_kv_pool_allocator.available_size()
            )
            result = self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))
            if self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self.token_to_kv_pool_allocator.available_size()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self.tree_cache.evictable_size()}, "
                    f"protected_size={self.tree_cache.protected_size()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        allocator = self.token_to_kv_pool_allocator
        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            kv_loc = alloc_for_decode_prealloc_hisparse(
                allocator,
                req=req,
                fill_len=fill_len,
                uses_swa_tail=self._uses_swa_tail_prealloc(),
                swa_tail_len=self._swa_tail_len(fill_len),
            )
            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        else:
            uses_swa_tail = self._uses_swa_tail_prealloc() and total_prefix_len == 0
            swa_tail_len = self._swa_tail_len(fill_len)
            kv_loc = alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=total_prefix_len,
                prefix_indices=prefix_indices,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
                req_to_token_pool=self.req_to_token_pool,
            )
        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self.tree_cache.evictable_size()}, "
            f"protected={self.tree_cache.protected_size()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"page_size={self.token_to_kv_pool_allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so cache_unfinished_req only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        # Set prefix_indices so downstream consumers (init_next_round_input,
        # prepare_for_extend) see the correct prefix length. In the agg path
        # this is done inside init_next_round_input, but decode-disagg needs
        # allocation info before batch assembly so we set it here.
        req.prefix_indices = (
            prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
        )
        req.set_extend_range(total_prefix_len, req.kv_committed_len)

        if decode_req is not None:
            if migration_end is None:
                raise DecodeAllocationLeaseError(
                    "migration_end is required for decode allocation issuance"
                )
            self._acquire_decode_allocation_lease(
                decode_req,
                migration_start=total_prefix_len,
                migration_end=migration_end,
                source_tp_size=source_tp_size,
            )

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


def alloc_for_decode_prealloc_hisparse(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> torch.Tensor:
    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len
    device = allocator.device
    prefix_lens = torch.tensor([0], dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([fill_len], dtype=torch.int64, device=device)
    seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
    if uses_swa_tail:
        kv_loc = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
            swa_tail_len=swa_tail_len,
        )
        req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
    else:
        kv_loc = allocator.alloc_logical_only(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
        )
    return kv_loc


def alloc_for_decode_prealloc(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
) -> torch.Tensor:
    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len
    if allocator.page_size == 1:
        kv_loc = allocator.alloc(delta_len)
    else:
        device = allocator.device
        last_loc = (
            prefix_indices[-1:].to(dtype=torch.int64, device=device)
            if prefix_len > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )
        extra_kwargs = {}
        dsv4_unwrap_prealloc = None
        if hasattr(allocator, "c4_attn_allocator"):
            assert req_to_token_pool is not None
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                dsv4_prealloc_kwargs,
                dsv4_unwrap_prealloc,
            )

            extra_kwargs = dsv4_prealloc_kwargs(
                allocator,
                req,
                fill_len,
                req_to_token_pool,
                device=device,
            )
        if uses_swa_tail:
            # Tail-only SWA allocation: only valid when prefix_len == 0.
            # When prefix_len > 0 (radix cache hit), we fall back to
            # alloc_extend which allocates SWA at full page count; the
            # SWA budget in that case may slightly under-estimate.
            kv_loc = allocator.alloc_extend_swa_tail(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=fill_len,
                swa_tail_len=swa_tail_len,
                **extra_kwargs,
            )
            req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
        else:
            kv_loc = allocator.alloc_extend(
                prefix_lens=torch.tensor(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
                **extra_kwargs,
            )
        if dsv4_unwrap_prealloc is not None:
            kv_loc = dsv4_unwrap_prealloc(
                kv_loc, req_to_token_pool, req, total_prefix_len, fill_len
            )
    return kv_loc


@dataclass(frozen=True)
class _PackedDecodeTransactionPoller:
    """Expose one packed transaction through the collective poller surface."""

    manager: CommonKVManager
    transaction: PackedDecodeRequestTransaction

    def poll(self) -> KVPoll:
        """Advance the exact packed actor transaction.

        :returns: Current packed transfer state.
        """

        return self.manager.poll_packed_decode_request_transaction(self.transaction)


@dataclass(frozen=True)
class _LegacyStagingTransactionPoller:
    """Preserve legacy staging scatter gating in a mixed packed queue."""

    decode_req: DecodeRequest
    metadata_buffers: MetadataBuffers
    server_args: object
    staging_handler: DecodeStagingHandler

    def poll(self) -> KVPoll:
        """Advance legacy staging scatter and poll its receiver.

        :returns: Receiver state gated by local scatter completion.
        """

        receiver = self.decode_req.kv_receiver
        requires_staging = receiver.require_staging
        if requires_staging and not self.staging_handler.is_done(self.decode_req):
            self.staging_handler.advance_scatter(self.decode_req)
        poll = receiver.poll()
        if (
            poll == KVPoll.Success
            and requires_staging
            and not self.staging_handler.is_done(self.decode_req)
        ):
            return KVPoll.Transferring
        if poll != KVPoll.Success or _is_fake_transfer(
            self.decode_req.req,
            self.server_args,
        ):
            return poll
        actual_room = self.metadata_buffers.bootstrap_room[
            self.decode_req.metadata_buffer_index,
            0,
        ].item()
        if actual_room == 0:
            return KVPoll.Transferring
        return poll


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    tp1_poll_progress_policy: SingletonPollProgressPolicy
    _terminal_requests: dict[bytes, DecodeRequest]

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator | None,
        tp_rank: int,
        metadata_buffers: MetadataBuffers | None,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self._terminal_requests = {}
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.tp1_poll_progress_policy = _create_singleton_poll_progress_policy(
            scheduler,
            "transfer",
        )
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None
        self.allocation_lease_authority: DecodeAllocationLeaseAuthority | None = None
        self.allocation_lifecycle_authority: CommonKVManager | None = None
        self.terminal_dflash_boundary_pool: DFlashBoundaryDeviceRowPool | None = None

    def _require_legacy_metadata_resources(
        self,
    ) -> tuple[ReqToMetadataIdxAllocator, MetadataBuffers]:
        """Return the paired legacy row resources outside terminal DFlash.

        :returns: Legacy metadata allocator and buffers.
        :raises DecodeAllocationLeaseError: If a legacy path enters terminal DFlash.
        """

        allocator = self.req_to_metadata_buffer_idx_allocator
        buffers = self.metadata_buffers
        if allocator is None or buffers is None:
            raise DecodeAllocationLeaseError(
                "terminal DFlash cannot enter a legacy metadata path"
            )
        return allocator, buffers

    def install_terminal_dflash_boundary_pool(
        self,
        pool: DFlashBoundaryDeviceRowPool,
    ) -> None:
        """Install the process-lifetime destination boundary-row owner once.

        :param pool: Registered decoder VRAM row pool used by terminal requests.
        """

        if type(pool) is not DFlashBoundaryDeviceRowPool:
            raise TypeError("pool must be DFlashBoundaryDeviceRowPool")
        if self.terminal_dflash_boundary_pool is not None:
            if self.terminal_dflash_boundary_pool is pool:
                return
            raise RuntimeError("terminal DFlash boundary pool ownership changed")
        self.terminal_dflash_boundary_pool = pool

    def _partition_scheduler_local_fake_requests(
        self,
        decode_reqs: list[DecodeRequest],
    ) -> tuple[list[DecodeRequest], list[DecodeRequest]]:
        """Validate and partition one scheduler-owned transfer cohort.

        Validation is complete before the queue or any request is mutated. A
        fake warmup owns local preallocated KV only, whereas a real legacy
        request requires the paired auxiliary metadata resources.

        :param decode_reqs: Candidate requests entering transfer completion.
        :returns: Scheduler-local fake requests and real legacy requests.
        """

        scheduler_local: list[DecodeRequest] = []
        legacy: list[DecodeRequest] = []
        for decode_req in decode_reqs:
            if decode_req.terminal_binding_digest is not None:
                raise DecodeAllocationLeaseError(
                    "terminal requests require the terminal ownership handoff"
                )
            if not _is_fake_transfer(
                decode_req.req,
                self.scheduler.server_args,
            ):
                legacy.append(decode_req)
                continue
            if decode_req.metadata_buffer_index != -1:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer retained a metadata row"
                )
            if decode_req.allocation_lease is not None:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer retained a migration lease"
                )
            if decode_req.packed_transaction is not None:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer retained a packed transaction"
                )
            if decode_req.prefix_match is not None:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer entered decode cache reuse"
                )
            if decode_req.kv_receiver is None:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer lost its local receiver"
                )
            if len(decode_req.req.output_ids) != 0:
                raise DecodeAllocationLeaseError(
                    "scheduler-local fake transfer already owns output tokens"
                )
            scheduler_local.append(decode_req)

        if len(legacy) > 0:
            self._require_legacy_metadata_resources()
        return scheduler_local, legacy

    def _commit_scheduler_local_fake_request(
        self,
        decode_req: DecodeRequest,
    ) -> Req:
        """Make one fake warmup runnable without a handoff protocol.

        Token zero is the fake transport's local boundary token. It exists only
        to exercise the decode prebuilt path and is never publication state.

        :param decode_req: Fully validated scheduler-local warmup request.
        :returns: Canonical request ready for the decode waiting queue.
        """

        self._commit_hicache_local_restore_to_req(decode_req)
        req = decode_req.req
        req.output_ids.append(_FAKE_TRANSFER_BOUNDARY_TOKEN_ID)
        req.cached_tokens = 0
        req.already_computed = 0
        req.cached_tokens_device = 0
        req.cached_tokens_host = 0
        req.cached_tokens_storage = 0
        req.mm_image_tokens = 0
        req.mm_audio_tokens = 0
        req.mm_video_tokens = 0
        req.time_stats.set_decode_transfer_queue_entry_time()
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        req.time_stats.set_wait_queue_entry_time()
        return req

    def add(self, decode_req: DecodeRequest) -> Req | None:
        """Accept one legacy transfer or complete one local warmup.

        :param decode_req: Exact preallocated decode request.
        :returns: Runnable local warmup request, otherwise ``None``.
        """

        completed = self.extend([decode_req])
        return completed[0] if len(completed) == 1 else None

    def extend(self, decode_reqs: list[DecodeRequest]) -> list[Req]:
        """Accept a mixed scheduler-local and legacy transfer cohort.

        :param decode_reqs: Exact preallocated decode requests.
        :returns: Scheduler-local warmups completed without transfer polling.
        """

        scheduler_local, legacy = self._partition_scheduler_local_fake_requests(
            decode_reqs
        )
        completed = [
            self._commit_scheduler_local_fake_request(decode_req)
            for decode_req in scheduler_local
        ]
        self.queue.extend(legacy)
        return completed

    def register_terminal_requests(
        self,
        decode_reqs: tuple[DecodeRequest, ...],
    ) -> None:
        """Register one complete owner-driven cohort before metadata publication.

        The legacy queue is deliberately not an input or output of this method.
        Validation completes before any request becomes visible to owner
        callbacks, so duplicate identities cannot partially attach a cohort.

        :param decode_reqs: Exact terminal children entering owner progress.
        """

        terminal: list[tuple[bytes, DecodeRequest]] = []
        new_terminal_digests: set[bytes] = set()
        for decode_req in decode_reqs:
            digest = decode_req.terminal_binding_digest
            if digest is None:
                raise DecodeAllocationLeaseError(
                    "terminal registry received a legacy request"
                )
            if digest in new_terminal_digests or digest in self._terminal_requests:
                raise DecodeAllocationLeaseError(
                    "terminal transfer binding identity was reused"
                )
            if any(entry is decode_req for entry in self.queue) or any(
                entry is decode_req for entry in self._terminal_requests.values()
            ):
                raise DecodeAllocationLeaseError(
                    "preallocated request already has transfer ownership"
                )
            new_terminal_digests.add(digest)
            terminal.append((digest, decode_req))

        self._terminal_requests.update(terminal)

    def validate_terminal_inference_attachment(
        self,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
        prepared_receiver: CommonKVReceiver,
    ) -> None:
        """Validate active or legitimately finalized terminal ownership.

        :param decode_req: Exact prepared request receiving response ownership.
        :param transaction: Immutable request-scoped transport owner.
        :param prepared_receiver: Receiver identity retained from PREPARE.
        """

        digest = transaction.terminal_binding_digest
        if digest is None:
            raise DecodeAllocationLeaseError(
                "terminal inference attachment lacks binding authority"
            )
        owned = self._terminal_requests.get(digest)
        active_transaction = decode_req.packed_transaction
        if active_transaction is transaction:
            if owned is not decode_req:
                raise DecodeAllocationLeaseError(
                    "terminal inference attachment lost singular registry ownership"
                )
            if decode_req.kv_receiver is not prepared_receiver:
                raise DecodeAllocationLeaseError(
                    "terminal inference attachment changed its prepared receiver"
                )
            return
        if active_transaction is not None:
            raise DecodeAllocationLeaseError(
                "terminal inference attachment retained another transaction"
            )
        if owned is not None:
            raise DecodeAllocationLeaseError(
                "terminal inference attachment registry outlived request ownership"
            )
        if (
            transaction.state is not PackedRequestTransactionState.COMMITTED
            or decode_req.allocation_lease is not None
            or decode_req.kv_receiver is not None
            or decode_req.metadata_buffer_index != -1
        ):
            raise DecodeAllocationLeaseError(
                "terminal inference attachment has incomplete finalized ownership"
            )

    def live_requests(self) -> tuple[DecodeRequest, ...]:
        """Return every legacy-polled and terminal-owned transfer request.

        :returns: Exact scheduler-visible transfer ownership population.
        """

        terminal = tuple(
            self._terminal_requests[digest]
            for digest in sorted(self._terminal_requests)
        )
        return (*self.queue, *terminal)

    def _commit_consumed_allocation(self, decode_req: DecodeRequest) -> None:
        """Release a legacy migration lease after metadata consumption.

        :param decode_req: Exact request whose transferred metadata was consumed.
        """

        lease = decode_req.allocation_lease
        if lease is None or decode_req.packed_transaction is not None:
            return
        authority = self.allocation_lease_authority
        lifecycle = self.allocation_lifecycle_authority
        if authority is None or lifecycle is None:
            raise DecodeAllocationLeaseError(
                "decode transfer allocation ownership is unavailable"
            )
        authority.commit_legacy_to_request_after_consumption(lease, lifecycle)
        authority.retire_terminal(lease)
        decode_req.allocation_lease = None

    def _complete_packed_metadata_consumption(
        self,
        decode_req: DecodeRequest,
    ) -> None:
        """Release packed metadata only after its contents are copied.

        :param decode_req: Exact request whose metadata was consumed.
        """

        transaction = decode_req.packed_transaction
        if transaction is None:
            return
        manager = self.allocation_lifecycle_authority
        if manager is None:
            raise DecodeAllocationLeaseError(
                "packed metadata consumption authority is unavailable"
            )
        metadata_index = decode_req.metadata_buffer_index
        if metadata_index < 0:
            raise DecodeAllocationLeaseError(
                "packed request lost its metadata row before consumption"
            )
        _, metadata_buffers = self._require_legacy_metadata_resources()
        metadata_buffers.bootstrap_room[metadata_index] = 0
        manager.complete_packed_decode_request_metadata_consumption(transaction)
        decode_req.metadata_buffer_index = -1
        decode_req.allocation_lease = None
        decode_req.packed_transaction = None

    def _quarantine_packed_transaction(
        self,
        decode_req: DecodeRequest,
        reason: str,
    ) -> bool:
        """Quarantine packed ownership without permitting legacy cleanup.

        :param decode_req: Exact request retaining packed ownership.
        :param reason: Stable failure reason.
        :returns: Whether the request retained a packed transaction.
        """

        transaction = decode_req.packed_transaction
        if transaction is None:
            return False
        manager = self.allocation_lifecycle_authority
        if manager is None:
            logger.error(
                "Packed transaction lost its quarantine authority for %s",
                decode_req.req.rid,
            )
            return True
        try:
            manager.quarantine_packed_decode_request_transaction(
                transaction,
                reason,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "Packed transaction quarantine failed for %s:\n%s",
                decode_req.req.rid,
                traceback.format_exc(),
            )
        return True

    def adopt_terminal_request(
        self,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
    ) -> TerminalDFlashDecodeAdoption:
        """Adopt authenticated DFlash state without releasing its VRAM row.

        :param decode_req: Exact terminal-registry request.
        :param transaction: Exact terminal-owned packed transaction.
        :returns: Device-copy completion authority retained by the owner.
        """

        self._require_terminal_transfer_request(decode_req, transaction)
        if not self.spec_algorithm.is_dflash():
            raise DecodeAllocationLeaseError(
                "terminal boundary adoption requires the DFlash algorithm"
            )
        if decode_req.allocation_lease is None:
            raise DecodeAllocationLeaseError(
                "terminal adoption lost its allocation lease field"
            )
        if decode_req.req.pd_dflash_boundary_token_id is not None:
            raise DecodeAllocationLeaseError(
                "terminal request already owns a DFlash boundary token"
            )
        pool = self.terminal_dflash_boundary_pool
        if pool is None:
            raise DecodeAllocationLeaseError(
                "terminal DFlash boundary pool is not installed"
            )
        transaction_adoption = (
            transaction.begin_dflash_boundary_adoption_on_scheduler_thread()
        )
        device_value = pool.enqueue_destination_adoption(
            transaction_adoption.slot,
            stream=self.scheduler.schedule_stream,
        )
        self._commit_terminal_dflash_metadata_to_req(
            decode_req,
            transaction_adoption,
            device_value,
        )
        return TerminalDFlashDecodeAdoption(
            transaction_adoption=transaction_adoption,
            device_value=device_value,
        )

    def finalize_terminal_request(
        self,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Make one request runnable after actor-authorized metadata release.

        :param decode_req: Exact adopted transfer-queue request.
        :param transaction: Exact terminal-owned packed transaction.
        """

        self._require_terminal_transfer_request(decode_req, transaction)
        if decode_req.allocation_lease is None:
            raise DecodeAllocationLeaseError(
                "terminal finalization lost its adopted allocation identity"
            )
        if (
            decode_req.req.pd_dflash_boundary_token_id is None
            or decode_req.req.pd_dflash_boundary_completion_event is None
        ):
            raise DecodeAllocationLeaseError(
                "terminal finalization lost request-owned DFlash boundary state"
            )
        if any(entry is decode_req.req for entry in self.scheduler.waiting_queue):
            raise DecodeAllocationLeaseError(
                "terminal request is already visible in the waiting queue"
            )
        receiver = decode_req.kv_receiver
        if receiver is None:
            raise DecodeAllocationLeaseError(
                "terminal finalization lost its transfer receiver"
            )

        # The terminal owner releases the exact VRAM row only after the D2D
        # completion event. The cloned token remains request-owned until DFlash
        # consumes it while constructing the first decode batch.
        binding_digest = transaction.terminal_binding_digest
        if binding_digest is None:
            raise DecodeAllocationLeaseError(
                "terminal finalization lost binding authority"
            )
        decode_req.metadata_buffer_index = -1
        decode_req.allocation_lease = None
        decode_req.packed_transaction = None
        receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        owned = self._terminal_requests.pop(binding_digest, None)
        if owned is not decode_req:
            raise DecodeAllocationLeaseError(
                "terminal finalization removed another registry owner"
            )
        self.scheduler.waiting_queue.append(decode_req.req)

    def quarantine_terminal_request(
        self,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
    ) -> bool:
        """Remove an ambiguous terminal request from active owner callbacks.

        The prepared cohort remains its process-lifetime retention owner.

        :param decode_req: Exact ambiguous decode request.
        :param transaction: Exact quarantined packed transaction.
        :returns: Whether this call removed the live terminal owner.
        """

        digest = transaction.terminal_binding_digest
        if digest is None:
            raise DecodeAllocationLeaseError(
                "terminal quarantine lacks terminal binding authority"
            )
        owned = self._terminal_requests.get(digest)
        if owned is None:
            return False
        if owned is not decode_req or decode_req.packed_transaction is not transaction:
            raise DecodeAllocationLeaseError(
                "terminal registry retains another packed transaction"
            )
        del self._terminal_requests[digest]
        return True

    def _require_terminal_transfer_request(
        self,
        decode_req: DecodeRequest,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Require exact singular terminal-registry ownership.

        :param decode_req: Candidate terminal request.
        :param transaction: Candidate request transaction.
        """

        if decode_req.packed_transaction is not transaction:
            raise DecodeAllocationLeaseError(
                "terminal transfer callback carries another packed transaction"
            )
        if transaction.request_owner is not decode_req:
            raise DecodeAllocationLeaseError(
                "terminal transfer transaction retains another request owner"
            )
        digest = transaction.terminal_binding_digest
        if digest is None:
            raise DecodeAllocationLeaseError(
                "terminal callback lacks terminal binding authority"
            )
        if self._terminal_requests.get(digest) is not decode_req:
            raise DecodeAllocationLeaseError(
                "terminal callback lacks singular registry ownership"
            )

    def _commit_terminal_dflash_metadata_to_req(
        self,
        decode_req: DecodeRequest,
        adoption: PackedDFlashBoundaryDecodeAdoption,
        device_value: DFlashBoundaryAdoptedValue,
    ) -> None:
        """Commit authenticated scalar state and retain the device token.

        :param decode_req: Exact terminal request becoming scheduler-owned.
        :param adoption: Actor-authenticated row and scalar state.
        :param device_value: Request-owned D2D result and row-release event.
        """

        metadata = adoption.metadata
        self._commit_hicache_local_restore_to_req(decode_req)

        replayed_boundary = (
            decode_req.is_rebootstrap
            and decode_req.req.pd_rebootstrap_forced_output_id is not None
        )
        request_completion_event = device_value.completion_event
        if replayed_boundary:
            committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id
            if committed_output_id is None:
                raise DecodeAllocationLeaseError(
                    "rebootstrap boundary disappeared during terminal adoption"
                )
            stream = self.scheduler.schedule_stream
            stream.wait_event(device_value.completion_event)
            with torch.cuda.stream(stream):
                device_value.boundary_token_id.fill_(committed_output_id)
                request_completion_event = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                request_completion_event.record(stream)
            decode_req.req.pd_rebootstrap_forced_output_id = None
        else:
            committed_output_id = metadata.boundary_token_id

        req = decode_req.req
        req.output_ids.append(committed_output_id)
        req.cached_tokens = metadata.cached_tokens
        req.already_computed = metadata.cached_tokens
        req.cached_tokens_device = metadata.cached_tokens_device
        req.cached_tokens_host = metadata.cached_tokens_host
        req.cached_tokens_storage = metadata.cached_tokens_storage
        req.mm_image_tokens = metadata.image_tokens
        req.mm_audio_tokens = metadata.audio_tokens
        req.mm_video_tokens = metadata.video_tokens
        req.pd_dflash_boundary_token_id = device_value.boundary_token_id
        req.pd_dflash_boundary_completion_event = request_completion_event

    def _reconcile_failed_allocation(
        self,
        decode_req: DecodeRequest,
        *,
        terminal_receiver_failure: bool,
    ) -> bool:
        """Resolve legacy migration ownership before failed-request cleanup.

        A failed collective is not enough by itself. The receiver must also
        expose its terminal failure, and the staging handler must drain every
        local scatter before migration pins can be returned. Any ambiguity
        quarantines the allocation and prevents request or metadata reuse.

        :param decode_req: Exact failed transfer request.
        :param terminal_receiver_failure: Whether the receiver exposed its
            terminal failure after the collective failed poll.
        :returns: Whether normal request and KV cleanup is authorized.
        """

        lease = decode_req.allocation_lease
        if lease is None:
            return True
        if decode_req.packed_transaction is not None:
            return False

        authority = self.allocation_lease_authority
        lifecycle = self.allocation_lifecycle_authority
        if authority is None or lifecycle is None:
            raise DecodeAllocationLeaseError(
                "decode transfer allocation ownership is unavailable"
            )
        if not terminal_receiver_failure:
            authority.quarantine(
                lease,
                "legacy transfer failure was not terminal at the receiver",
            )
            return False
        if not self.enable_staging or self.staging_handler is None:
            authority.quarantine(
                lease,
                "legacy transfer failure has no staging lifecycle owner",
            )
            return False

        bootstrap_room = decode_req.req.bootstrap_room
        if not self.staging_handler.is_staging_room(bootstrap_room):
            authority.quarantine(
                lease,
                "legacy transfer failure lost its staging room",
            )
            return False
        try:
            self.staging_handler.unregister_decode_req(bootstrap_room)
        except Exception:
            logger.critical(
                "Failed to quiesce legacy staging room %s; quarantining its "
                "decode allocation:\n%s",
                bootstrap_room,
                traceback.format_exc(),
            )
            authority.quarantine(
                lease,
                "legacy staging quiescence failed after terminal transfer failure",
            )
            return False

        try:
            permit = authority.authorize_legacy_abort_after_terminal_failure(
                lease,
                lifecycle,
            )
        except DecodeAllocationLeaseError:
            authority.quarantine(
                lease,
                "legacy terminal failure could not authorize request cleanup",
            )
            return False
        authority.consume_abort_permit(lease, permit)
        authority.retire_terminal(lease)
        decode_req.allocation_lease = None
        return True

    def _copy_transfer_metadata_to_req(self, decode_req: DecodeRequest) -> bool:
        """Copy one validated auxiliary row into its exact request.

        :param decode_req: Exact transfer request retaining the metadata row.
        :returns: Whether the row was valid and completely copied.
        """

        idx = decode_req.metadata_buffer_index
        _, metadata_buffers = self._require_legacy_metadata_resources()
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_token_sampling_mask_len,
            output_token_sampling_mask_idx,
            output_token_sampling_logprobs,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_dsa_topk_indices,
            output_bootstrap_room,
        ) = metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            retained = self._quarantine_packed_transaction(
                decode_req,
                "packed metadata was absent after transfer commit",
            )
            if not retained:
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
            return False
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            retained = self._quarantine_packed_transaction(
                decode_req,
                "packed metadata bootstrap room differed from its request",
            )
            if not retained:
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
            return False

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        # PD true-retraction rebootstrap: the prefill recomputed the prefix KV
        # under the current weights and sampled a fresh handoff token, but when
        # there is a remembered boundary token we are *replaying* an
        # already-emitted token. Override the handoff with it, and skip
        # re-committing a logprob for it -- it keeps its original behavior
        # logprob from before the retract (we never re-score generated tokens
        # under the new policy). A rebootstrap with no boundary token (retracted
        # before emitting any output) falls through to the normal path so its
        # first token and logprob are committed as usual.
        replayed_boundary = (
            decode_req.is_rebootstrap
            and decode_req.req.pd_rebootstrap_forced_output_id is not None
        )
        if replayed_boundary:
            committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id
            decode_req.req.pd_rebootstrap_forced_output_id = None
        else:
            committed_output_id = output_id[0].item()
        decode_req.req.output_ids.append(committed_output_id)
        decode_req.req.cached_tokens = cached_tokens[0].item()
        # The prefill node already reported its prefix-cache hit in
        # cached_tokens[0]. Seed already_computed with it so that
        # prepare_for_prebuilt's `cached_tokens += pre_len - already_computed`
        # only adds decode-side reuse *beyond* what prefill counted, instead of
        # double-counting the shared prompt prefix (which would make
        # cached_tokens exceed prompt_tokens when decode radix cache is on).
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        # Multimodal prompt token counts packed into cached_tokens slots 4-6
        # by the prefill node (see MetadataBuffers.set_buf).
        decode_req.req.mm_image_tokens = cached_tokens[4].item()
        decode_req.req.mm_audio_tokens = cached_tokens[5].item()
        decode_req.req.mm_video_tokens = cached_tokens[6].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states
            if (
                output_dsa_topk_indices is not None
                and torch.all(output_dsa_topk_indices < 0).item()
            ):
                output_dsa_topk_indices = None
            decode_req.req.output_dsa_topk_indices = output_dsa_topk_indices

        if decode_req.req.return_logprob and not replayed_boundary:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
        if decode_req.req.return_sampling_mask:
            assert (
                output_token_sampling_mask_idx is not None
            ), "sampling mask buffer disabled on decode side"
            sampling_mask_len = int(output_token_sampling_mask_len[0].item())
            if sampling_mask_len < 0:
                decode_req.req.output_token_sampling_mask.append(None)
                decode_req.req.output_token_sampling_logprobs.append(None)
            else:
                decode_req.req.output_token_sampling_mask.append(
                    output_token_sampling_mask_idx[:sampling_mask_len].cpu().tolist()
                )
                decode_req.req.output_token_sampling_logprobs.append(
                    float(output_token_sampling_logprobs[0].item())
                )

        return True

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> None:
        """Complete the legacy polled transfer path after metadata copy.

        :param decode_req: Exact transfer request becoming scheduler-owned.
        """

        if not self._copy_transfer_metadata_to_req(decode_req):
            return

        try:
            self._complete_packed_metadata_consumption(decode_req)
        except Exception:  # noqa: BLE001
            reason = "packed metadata consumption completion failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._quarantine_packed_transaction(decode_req, reason)
            prepare_abort(
                decode_req.req,
                reason,
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._commit_consumed_allocation(decode_req)

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _poll_with_metadata_gate(self) -> List[int]:
        _, metadata_buffers = self._require_legacy_metadata_resources()
        pollers = (
            [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
            if self.scheduler.enable_decode_hicache
            else [dr.kv_receiver for dr in self.queue]
        )
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=metadata_buffers,
            server_args=self.scheduler.server_args,
            singleton_progress_policy=self.tp1_poll_progress_policy,
        )

    def _poll_with_staging(self) -> list:
        _, metadata_buffers = self._require_legacy_metadata_resources()
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=metadata_buffers,
            server_args=self.scheduler.server_args,
            singleton_progress_policy=self.tp1_poll_progress_policy,
        )

    def _poll_with_packed_transactions(self) -> list:
        """Poll packed actors and any colocated legacy staging requests.

        :returns: Collectively reduced request transfer states.
        """

        manager = self.allocation_lifecycle_authority
        if manager is None:
            raise DecodeAllocationLeaseError(
                "packed decode progress authority is unavailable"
            )
        _, metadata_buffers = self._require_legacy_metadata_resources()
        pollers: list[
            _PackedDecodeTransactionPoller | _LegacyStagingTransactionPoller
        ] = []
        for decode_req in self.queue:
            transaction = decode_req.packed_transaction
            if transaction is None:
                pollers.append(
                    _LegacyStagingTransactionPoller(
                        decode_req,
                        metadata_buffers,
                        self.scheduler.server_args,
                        self.staging_handler,
                    )
                )
                continue
            pollers.append(_PackedDecodeTransactionPoller(manager, transaction))
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            singleton_progress_policy=self.tp1_poll_progress_policy,
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            self.tp1_poll_progress_policy.mark_idle()
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if any(decode_req.packed_transaction is not None for decode_req in self.queue):
            polls = self._poll_with_packed_transactions()
        elif self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        quarantined_indices: set[int] = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            hicache_restore_status = decode_req.hicache_restore_status
            if (
                poll == KVPoll.Failed
                or hicache_restore_status == HiCacheRestoreResult.FAILED
            ):
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                terminal_receiver_failure: bool = False
                if poll == KVPoll.Failed:
                    if decode_req.packed_transaction is not None:
                        error_message += " with terminal packed actor failure"
                    else:
                        try:
                            decode_req.kv_receiver.failure_exception()
                        except Exception as e:
                            terminal_receiver_failure = True
                            error_message += f" with exception {e}"
                            is_propagated = getattr(
                                e,
                                "is_from_another_rank",
                                False,
                            )
                            logger.debug(
                                "Terminal decode transfer failure traceback:\n%s",
                                traceback.format_exc(),
                            )
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                self._quarantine_packed_transaction(
                    decode_req,
                    "packed decode transfer failed before metadata consumption",
                )
                cleanup_authorized: bool = self._reconcile_failed_allocation(
                    decode_req,
                    terminal_receiver_failure=terminal_receiver_failure,
                )
                if cleanup_authorized:
                    release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                    decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None
                else:
                    quarantined_indices.add(i)
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                packed_retained = decode_req.packed_transaction is not None
                if packed_retained:
                    quarantined_indices.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    if self.scheduler.enable_hisparse:
                        self.scheduler.hisparse_coordinator.request_finished(
                            decode_req.req
                        )
                    self._clean_hicache_prefetch_resources(decode_req)
                    if not packed_retained:
                        release_kv_cache(
                            decode_req.req,
                            self.tree_cache,
                            is_insert=False,
                        )
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                else:
                    if packed_retained:
                        raise DecodeAllocationLeaseError(
                            "packed metadata failure did not abort its request"
                        )
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if i in quarantined_indices:
                continue
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            if idx == -1:
                continue
            metadata_allocator, metadata_buffers = (
                self._require_legacy_metadata_resources()
            )
            # Reset so the next owner sees actual_room == 0 ("not yet written")
            # instead of the stale value, avoiding a false-positive mismatch.
            metadata_buffers.bootstrap_room[idx] = 0
            metadata_allocator.free(idx)
            self.queue[i].metadata_buffer_index = -1

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        if len(self.queue) == 0:
            self.tp1_poll_progress_policy.mark_idle()

        return transferred_reqs

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory."""
        if len(self._terminal_requests) > 0:
            raise DecodeAllocationLeaseError(
                "cannot release memory with terminal requests in flight"
            )
        self.queue.clear()
        self.tp1_poll_progress_policy.mark_idle()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted."""
        pass


class SchedulerDisaggregationDecodeMixin:
    def get_decode_poll_progress_stats(
        self: Scheduler,
    ) -> dict[str, dict[str, int | str]]:
        """Return live TP1 polling counters for existing diagnostics.

        :returns: Queue names mapped to polling counter snapshots.
        """

        preallocation_policy = (
            self.disagg_decode_prealloc_queue.tp1_poll_progress_policy
        )
        transfer_policy = self.disagg_decode_transfer_queue.tp1_poll_progress_policy
        return {
            "preallocation": preallocation_policy.diagnostic_state(),
            "transfer": transfer_policy.diagnostic_state(),
        }

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            self.drain_terminal_scheduler_receipts()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            self.drain_terminal_scheduler_receipts()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        running_batch.hisparse_coordinator = self.hisparse_coordinator
                else:
                    running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if running_batch.is_empty():
            ret = None
        else:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_new_prebuilt_batch(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, running_batch)

        curr_batch_size = running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode-radix path: new requests already matched in
                # `pop_preallocated`. Retracted requests reset `last_node`,
                # so re-match only when that state is missing.
                if self.server_args.disaggregation_decode_enable_radix_cache:
                    tree_cache = self.tree_cache if req.last_node is None else None
                else:
                    tree_cache = self.tree_cache
                req.init_next_round_input(tree_cache)
                # Truncate fill_len to kv_committed_len so cache_unfinished_req
                # only sees committed KV (full array includes one uncommitted
                # token because init_next_round_input rebuilt it as full).
                if req.kv_committed_len is not None:
                    req.set_extend_range(len(req.prefix_indices), req.kv_committed_len)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()

        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            scheduler_local_reqs = self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = [
                *scheduler_local_reqs,
                *self.disagg_decode_transfer_queue.pop_transferred(),
            ]
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(transferred_reqs)
