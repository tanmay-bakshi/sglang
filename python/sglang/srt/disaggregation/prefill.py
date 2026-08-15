"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import time
import traceback
import uuid
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedDFlashBoundaryCounters,
)
from sglang.srt.disaggregation.nixl.packed_runtime import PackedPrefillLaunchPlan
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFlashBoundaryPrefillSource,
)
from sglang.srt.disaggregation.terminal_progress.output_projection import (
    FrozenPrefillGatewayOutputShell,
    PinnedTerminalGatewayResultSlot,
    PrefillTerminalGatewayOutputProjection,
    freeze_prefill_gateway_output_shell,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalRequestBinding
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
    resolve_kv_layer_ids,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils import is_npu
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

_is_npu = is_npu()
DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS = 64
DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS = 0.0005
DISAGG_PREFILL_TRANSFER_PROGRESS_TIME_BUDGET_SECONDS = 0.032


@dataclasses.dataclass(frozen=True, slots=True)
class _TerminalPrefillLaunch:
    """One request's complete immutable pre-model terminal ownership.

    :ivar req: Scheduler-owned request retained until native reclaim.
    :ivar result_index: Token row within the submitted generation batch.
    :ivar identity: Exact rank-local terminal lifecycle identity.
    :ivar transport: Complete pre-launch packed transfer geometry.
    :ivar dflash_source: Canonical DFlash row, absent off source rank zero.
    :ivar output_shell: Canonical immutable gateway response shell.
    :ivar result_slot: Canonical pinned token result row.
    :ivar producer_event_generation: Exact generation binding the CUDA event.
    """

    req: Req
    result_index: int
    identity: PackedTerminalSourceIdentityPlan
    transport: PackedPrefillLaunchPlan
    dflash_source: DFlashBoundaryPrefillSource | None
    output_shell: FrozenPrefillGatewayOutputShell | None
    result_slot: PinnedTerminalGatewayResultSlot | None
    producer_event_generation: bytes

    def __post_init__(self) -> None:
        """Validate rank-local ownership before model submission."""

        if not isinstance(self.req, Req):
            raise TypeError("req must be a Req")
        if type(self.result_index) is not int or self.result_index < 0:
            raise ValueError("result_index must be a non-negative integer")
        if type(self.identity) is not PackedTerminalSourceIdentityPlan:
            raise TypeError("identity must be PackedTerminalSourceIdentityPlan")
        if type(self.transport) is not PackedPrefillLaunchPlan:
            raise TypeError("transport must be PackedPrefillLaunchPlan")
        canonical = self.identity.local_binding.owner.tp_rank == 0
        canonical_values = (
            self.dflash_source,
            self.output_shell,
            self.result_slot,
        )
        if canonical and any(value is None for value in canonical_values):
            raise ValueError("canonical terminal launch is incomplete")
        if not canonical and any(value is not None for value in canonical_values):
            raise ValueError("noncanonical terminal launch owns canonical state")
        if (
            type(self.producer_event_generation) is not bytes
            or len(self.producer_event_generation) != 16
        ):
            raise ValueError("producer_event_generation must contain 16 bytes")


class _TerminalPrefillBatchLeaseLedger:
    """Track pre-model, CUDA-touched, and manager-owned DFlash rows."""

    _manager: CommonKVManager
    _unbound: dict[int, DFlashBoundaryPrefillSource]
    _cuda_touched: tuple[
        DFlashBoundaryPrefillSource,
        PackedTerminalSourceSubmission,
    ] | None

    def __init__(self, manager: CommonKVManager) -> None:
        """Create an empty batch-scope ownership ledger.

        :param manager: Sole source-row lifetime owner.
        """

        self._manager = manager
        self._unbound = {}
        self._cuda_touched = None

    def retain(self, source: DFlashBoundaryPrefillSource) -> None:
        """Retain one lease before any CUDA producer work touches it.

        :param source: Exact active source lease.
        """

        key = id(source)
        if key in self._unbound:
            raise RuntimeError("terminal DFlash source was retained twice")
        self._unbound[key] = source

    def begin_cuda(
        self,
        source: DFlashBoundaryPrefillSource,
        submission: PackedTerminalSourceSubmission,
    ) -> None:
        """Classify one retained lease as unsafe for synchronous release.

        :param source: Exact lease about to receive producer work.
        :param submission: Complete pinned transport and result-slot ownership.
        """

        if self._cuda_touched is not None:
            raise RuntimeError("another terminal DFlash source is CUDA-touched")
        if self._unbound.get(id(source)) is not source:
            raise RuntimeError("CUDA work targets an unretained DFlash source")
        self._cuda_touched = (source, submission)

    def hand_to_manager(self, source: DFlashBoundaryPrefillSource) -> None:
        """Transfer one CUDA-touched lease into manager lifecycle handling.

        :param source: Exact lease entering the manager bind boundary.
        """

        touched = self._cuda_touched
        if touched is None or touched[0] is not source:
            raise RuntimeError("manager handoff targets another DFlash source")
        owned = self._unbound.pop(id(source), None)
        if owned is not source:
            raise RuntimeError("terminal DFlash ownership changed before handoff")
        self._cuda_touched = None

    def settle_after_failure(self) -> tuple[str, ...]:
        """Quarantine CUDA work and cancel every untouched lease once.

        :returns: Complete cleanup tracebacks without masking the root failure.
        """

        failures: list[str] = []
        touched = self._cuda_touched
        if touched is not None:
            source, submission = touched
            self._unbound.pop(id(source), None)
            self._cuda_touched = None
            try:
                self._manager.quarantine_unpublished_terminal_source_submission(
                    submission
                )
            except Exception:  # noqa: BLE001
                failures.append(traceback.format_exc())
        sources = tuple(self._unbound.values())
        self._unbound.clear()
        for source in sources:
            try:
                self._manager.cancel_unpublished_terminal_dflash_source(source)
            except Exception:  # noqa: BLE001
                failures.append(traceback.format_exc())
        return tuple(failures)


class _TerminalPrefillPrelaunchBatchOwner:
    """Own preleased rows until the model worker invokes terminal binding."""

    _manager: CommonKVManager
    _launches: tuple[_TerminalPrefillLaunch, ...]
    _claimed: bool

    def __init__(
        self,
        manager: CommonKVManager,
        launches: tuple[_TerminalPrefillLaunch, ...],
    ) -> None:
        """Retain one immutable prelaunch batch.

        :param manager: Sole source-row lifetime owner.
        :param launches: Complete plans frozen before model submission.
        """

        if type(launches) is not tuple or any(
            type(launch) is not _TerminalPrefillLaunch for launch in launches
        ):
            raise TypeError("launches must contain terminal prefill plans")
        self._manager = manager
        self._launches = launches
        self._claimed = False

    def claim_for_bind(self) -> tuple[_TerminalPrefillLaunch, ...]:
        """Transfer the complete batch to post-submit lifecycle binding.

        :returns: Exact immutable launch plans.
        """

        if self._claimed:
            raise RuntimeError("terminal prelaunch batch was claimed twice")
        self._claimed = True
        return self._launches

    def cancel_if_unclaimed(self) -> tuple[str, ...]:
        """Release only rows never exposed to model-result CUDA work.

        :returns: Complete cleanup tracebacks without masking submit failure.
        """

        if self._claimed:
            return ()
        self._claimed = True
        failures: list[str] = []
        for launch in self._launches:
            source = launch.dflash_source
            if source is None:
                continue
            try:
                self._manager.cancel_unpublished_terminal_dflash_source(source)
            except Exception:  # noqa: BLE001
                failures.append(traceback.format_exc())
        return tuple(failures)


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator | None
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        if allocator is None:
            raise RuntimeError("request retained a metadata row without an allocator")
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator | None,
        metadata_buffers: MetadataBuffers | None,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        self.transfer_backend = transfer_backend
        terminal_membership = self.scheduler.server_args.pd_terminal_local_membership
        self.terminal_source = terminal_membership is not None
        if (
            self.terminal_source
            and self.scheduler.server_args.optimistic_prefill_attempts != 0
        ):
            raise ValueError(
                "terminal source requires optimistic prefill attempts disabled"
            )
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            server_args = self.scheduler.server_args
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            cps = server_args.chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {server_args.chunked_prefill_size}."
                )
            if self.pp_size > 1:
                # Staging writer accounting has no pp dimension.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support pp_size > 1."
                )
            if server_args.enable_prefill_context_parallel:
                # CP rewrites index_slice per rank, breaking the chunk grid.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support "
                    "prefill context parallelism."
                )
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.terminal_request_capacity = self.scheduler.max_running_requests * 2
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else self.token_to_kv_pool.start_layer
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_layer_ids = resolve_kv_layer_ids(
            self.token_to_kv_pool,
            len(kv_data_ptrs),
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        if self.draft_token_to_kv_pool is not None and transfer_draft_cache:
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
            kv_layer_ids = (
                [*kv_layer_ids, *draft_kv_layer_ids]
                if len(kv_layer_ids) > 0 and len(draft_kv_layer_ids) > 0
                else []
            )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = kv_layer_ids
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

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
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if self.terminal_source:
            raise RuntimeError("terminal source cannot allocate legacy metadata rows")
        allocator = self.req_to_metadata_buffer_idx_allocator
        if allocator is None:
            raise RuntimeError("legacy prefill metadata allocator is unavailable")
        if req.metadata_buffer_index >= 0:
            return True

        if allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.terminal_source and not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send, self.token_to_kv_pool.page_size
        )
        metadata_index = None if self.terminal_source else req.metadata_buffer_index
        req.disagg_kv_sender.init(num_pages, metadata_index)
        req.pending_bootstrap = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if (
                rids_to_check is not None
                and req.rid not in rids_to_check
                and poll != KVPoll.Failed
            ):
                # In PP mode, successful bootstrap still requires cross-rank
                # consensus. Local failures are terminal and must be drained
                # even if an earlier PP rank has already removed the request.
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if (
                    req.prefill_attempt_count
                    < self.scheduler.server_args.optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def disagg_prefill_live_transfer_requests(
        self: Scheduler,
    ) -> tuple[Req, ...]:
        """Return legacy and owner-managed prefill requests for accounting.

        :returns: Complete live prefill transfer population without exposing
            terminal requests to legacy polling.
        """

        return (
            *self.disagg_prefill_inflight_queue,
            *self.disagg_prefill_terminal_requests.values(),
        )

    def abort_terminal_prefill_requests(
        self: Scheduler,
        rid: str,
        abort_all: bool,
    ) -> int:
        """Record client abort while owner-managed publication completes.

        This registry is populated inside :meth:`bind_terminal_prefill_launches`
        immediately before PREPARE publication. Both this method and that bind
        run on the scheduler thread, and a request becomes visible here only
        after native rollback has become unsafe. The source owner records client
        intent while the immutable lifecycle completes its normal reclaim and
        publication joins; the disconnected gateway client independently drops
        the eventual response.

        :param rid: Request identifier or prefix selected by the abort.
        :param abort_all: Whether every owner-managed request is selected.
        :returns: Number of matching owner-managed requests.
        """

        if type(rid) is not str:
            raise TypeError("rid must be a string")
        if type(abort_all) is not bool:
            raise TypeError("abort_all must be bool")
        matches = tuple(
            req
            for req in self.disagg_prefill_terminal_requests.values()
            if abort_all or req.rid.startswith(rid)
        )
        if len(matches) == 0:
            return 0
        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        bindings = self.disagg_prefill_terminal_bindings
        for req in matches:
            binding = bindings.get(req.rid)
            if binding is None:
                raise RuntimeError("terminal request lost its source binding")
            manager.cancel_terminal_source_request(
                binding,
                "client cancelled an owner-managed source request",
            )
        return len(matches)

    def bind_disagg_prefill_producer_event(
        self: Scheduler,
        batch: ScheduleBatch,
        event: torch.cuda.Event,
    ) -> None:
        """Bind exact packed-transfer completion to its producing batch.

        :param batch: Batch whose forward produced the source KV writes.
        :param event: Event recorded immediately after that exact forward.
        """

        protocol = self.disagg_prefill_bootstrap_queue.kv_manager.kv_transfer_protocol()
        if protocol != "packed-v4":
            return
        if not isinstance(event, torch.cuda.Event):
            raise TypeError("packed prefill producer dependency must be a CUDA event")
        batch.disagg_kv_producer_event = event

    def record_disagg_prefill_producer_event(
        self: Scheduler,
        batch: ScheduleBatch,
        stream: torch.cuda.Stream,
    ) -> None:
        """Record exact packed-transfer completion after one prefill forward.

        :param batch: Batch whose forward produced the source KV writes.
        :param stream: CUDA stream on which that forward was enqueued.
        """

        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        if manager.uses_terminal_source_publication():
            return
        protocol = manager.kv_transfer_protocol()
        if protocol != "packed-v4":
            return
        event = torch.cuda.Event()
        event.record(stream)
        self.bind_disagg_prefill_producer_event(batch, event)

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        if kv_mgr.uses_terminal_source_publication():
            return
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        if self.disagg_prefill_bootstrap_queue.terminal_source:
            return
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        if not candidates:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    def build_terminal_prefill_launches(
        self: Scheduler,
        batch: ScheduleBatch,
    ) -> tuple[_TerminalPrefillLaunch, ...]:
        """Freeze every final terminal request before the model submission.

        Intermediate chunks intentionally produce no transport plan. Their
        pages remain owned by the request, so the final chunk projects the
        complete remaining migration range in one immutable submission.

        :param batch: Exact generation batch about to enter the model worker.
        :returns: Rank-local final-request launch plans in result-row order.
        """

        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        if not manager.uses_terminal_source_publication():
            return ()
        if not self.spec_algorithm.is_dflash():
            raise RuntimeError("terminal source requires the DFlash schema")

        launches: list[_TerminalPrefillLaunch] = []
        leased_sources: list[DFlashBoundaryPrefillSource] = []
        try:
            for result_index, req in enumerate(batch.reqs):
                if _is_fake_transfer(req, self.server_args):
                    continue
                if req.inflight_middle_chunks > 0:
                    continue
                if req.pending_bootstrap:
                    raise RuntimeError(
                        "terminal source admission preceded bootstrap readiness"
                    )
                main_pages, state_indices = (
                    self.freeze_disagg_prefill_final_geometry(req)
                )
                event_generation = uuid.uuid4().bytes
                dflash_source: DFlashBoundaryPrefillSource | None = None
                output_shell: FrozenPrefillGatewayOutputShell | None = None
                result_slot: PinnedTerminalGatewayResultSlot | None = None
                if manager.terminal_source_is_canonical():
                    cached_details = self.output_streamer.get_cached_tokens_details(req)
                    output_shell = freeze_prefill_gateway_output_shell(
                        req,
                        cached_tokens_details=cached_details,
                        dp_rank=self.ps.dp_rank,
                        speculative=self.spec_algorithm.is_some(),
                    )
                    result_slot = PinnedTerminalGatewayResultSlot(uuid.uuid4().bytes)
                    dflash_source = manager.lease_terminal_dflash_source(
                        PackedDFlashBoundaryCounters(
                            cached_tokens=req.cached_tokens,
                            cached_tokens_device=req.cached_tokens_device,
                            cached_tokens_host=req.cached_tokens_host,
                            cached_tokens_storage=req.cached_tokens_storage,
                            image_tokens=output_shell.image_tokens,
                            audio_tokens=output_shell.audio_tokens,
                            video_tokens=output_shell.video_tokens,
                        )
                    )
                    leased_sources.append(dflash_source)
                identity, transport = manager.build_terminal_source_launch_plan(
                    room=req.bootstrap_room,
                    source_main_pages=main_pages,
                    state_indices=state_indices,
                    dflash_source=dflash_source,
                )
                launches.append(
                    _TerminalPrefillLaunch(
                        req=req,
                        result_index=result_index,
                        identity=identity,
                        transport=transport,
                        dflash_source=dflash_source,
                        output_shell=output_shell,
                        result_slot=result_slot,
                        producer_event_generation=event_generation,
                    )
                )
        except Exception as error:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            cleanup_failures: list[str] = []
            for source in leased_sources:
                try:
                    manager.cancel_unpublished_terminal_dflash_source(source)
                except Exception:  # noqa: BLE001
                    cleanup_failures.append(traceback.format_exc())
            if len(cleanup_failures) > 0:
                error.add_note(
                    "terminal source launch cleanup failed:\n"
                    + "\n".join(cleanup_failures)
                )
            logger.error(
                "Terminal source launch construction failed:\n%s",
                formatted_traceback,
            )
            raise
        return tuple(launches)

    def bind_terminal_prefill_launches(
        self: Scheduler,
        prelaunch_owner: _TerminalPrefillPrelaunchBatchOwner,
        result: GenerationBatchResult,
    ) -> GenerationBatchResult:
        """Bind exact producer events and owner lifecycles after submission.

        :param prelaunch_owner: Batch owner retained across model submission.
        :param result: Device-resident generation result from that submission.
        :returns: The unchanged generation result for normal result handling.
        """

        if type(prelaunch_owner) is not _TerminalPrefillPrelaunchBatchOwner:
            raise TypeError("prelaunch_owner must own terminal prefill plans")
        launches = prelaunch_owner.claim_for_bind()
        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        lease_ledger = _TerminalPrefillBatchLeaseLedger(manager)
        for launch in launches:
            if launch.dflash_source is not None:
                lease_ledger.retain(launch.dflash_source)
        try:
            if not manager.uses_terminal_source_publication():
                raise RuntimeError(
                    "terminal source deactivated during model submission"
                )
            if type(result.next_token_ids) is not torch.Tensor:
                raise TypeError("terminal source requires device-resident token ids")
            stream = torch.cuda.current_stream(device=self.device)
            for launch in launches:
                req = launch.req
                if req.rid in self.disagg_prefill_terminal_requests:
                    raise RuntimeError("terminal prefill request identity was reused")
                producer_event = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                projection: PrefillTerminalGatewayOutputProjection | None = None
                if launch.dflash_source is not None:
                    output_shell = launch.output_shell
                    result_slot = launch.result_slot
                    if output_shell is None or result_slot is None:
                        raise RuntimeError(
                            "canonical terminal output state disappeared"
                        )
                    projection = PrefillTerminalGatewayOutputProjection(
                        shell=output_shell,
                        result_slot=result_slot,
                        producer_event_generation=launch.producer_event_generation,
                    )
                transport_submission = launch.transport.bind_producer_event(
                    producer_event
                )
                submission = PackedTerminalSourceSubmission(
                    identity=launch.identity,
                    output_projection=projection,
                    producer_event_generation=launch.producer_event_generation,
                    transport_submission=transport_submission,
                )
                if launch.dflash_source is not None:
                    lease_ledger.begin_cuda(launch.dflash_source, submission)
                    manager.enqueue_terminal_dflash_source_projection(
                        launch.dflash_source,
                        result.next_token_ids[
                            launch.result_index : launch.result_index + 1
                        ],
                        result_slot,
                        stream=stream,
                        producer_event=producer_event,
                    )
                else:
                    producer_event.record(stream)

                def release_resources(
                    retired: PackedTerminalSourceSubmission,
                    *,
                    expected: PackedTerminalSourceSubmission = submission,
                    retained_req: Req = req,
                ) -> None:
                    """Release state only under native reclaim authority."""

                    if retired is not expected:
                        raise RuntimeError(
                            "terminal source reclaimed another submission"
                        )
                    current = self.disagg_prefill_terminal_requests.get(
                        retained_req.rid
                    )
                    if current is not retained_req:
                        raise RuntimeError("terminal source request registry changed")
                    current_binding = self.disagg_prefill_terminal_bindings.get(
                        retained_req.rid
                    )
                    if current_binding != expected.identity.local_binding:
                        raise RuntimeError("terminal source binding registry changed")
                    retained_req.disagg_kv_sender.clear()
                    release_kv_cache(retained_req, self.tree_cache)
                    retained_req.time_stats.set_prefill_kv_transfer_finish_time()
                    retained_req.time_stats.set_completion_time()
                    del self.disagg_prefill_terminal_requests[retained_req.rid]
                    del self.disagg_prefill_terminal_bindings[retained_req.rid]

                if launch.dflash_source is not None:
                    lease_ledger.hand_to_manager(launch.dflash_source)

                def commit_scheduler_retention(
                    retained: PackedTerminalSourceSubmission,
                    *,
                    expected: PackedTerminalSourceSubmission = submission,
                    retained_req: Req = req,
                ) -> None:
                    """Commit scheduler reachability before PREPARE publication.

                    :param retained: Exact manager-bound source submission.
                    """

                    if retained is not expected:
                        raise RuntimeError(
                            "terminal source retained another submission"
                        )
                    if retained_req.rid in self.disagg_prefill_terminal_requests:
                        raise RuntimeError(
                            "terminal prefill request identity was reused"
                        )
                    if retained_req.rid in self.disagg_prefill_terminal_bindings:
                        raise RuntimeError(
                            "terminal prefill binding identity was reused"
                        )
                    maybe_cache_unfinished_req(retained_req, self.tree_cache)
                    self.disagg_prefill_terminal_requests[retained_req.rid] = (
                        retained_req
                    )
                    try:
                        self.disagg_prefill_terminal_bindings[retained_req.rid] = (
                            retained.identity.local_binding
                        )
                    except MemoryError:
                        del self.disagg_prefill_terminal_requests[retained_req.rid]
                        raise

                manager.bind_terminal_source_submission(
                    submission,
                    release_resources,
                    commit_scheduler_retention,
                )
        except Exception as error:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            cleanup_failures = list(lease_ledger.settle_after_failure())
            try:
                manager.fail_terminal_source_process(
                    "terminal prefill batch bind failed after model submission",
                    formatted_traceback,
                )
            except Exception:  # noqa: BLE001
                cleanup_failures.append(traceback.format_exc())
            if len(cleanup_failures) > 0:
                error.add_note(
                    "terminal source batch rollback failed:\n"
                    + "\n".join(cleanup_failures)
                )
            logger.error(
                "Terminal source batch binding failed:\n%s",
                formatted_traceback,
            )
            raise
        return result

    def run_terminal_prefill_batch(
        self: Scheduler,
        batch: ScheduleBatch,
    ) -> GenerationBatchResult:
        """Submit one terminal batch with exception-total prelaunch ownership.

        :param batch: Exact scheduler batch entering the model worker.
        :returns: Model result after immediate terminal lifecycle binding.
        """

        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        launches = self.build_terminal_prefill_launches(batch)
        prelaunch_owner = _TerminalPrefillPrelaunchBatchOwner(manager, launches)
        terminal_bind = functools.partial(
            self.bind_terminal_prefill_launches,
            prelaunch_owner,
        )
        try:
            return self.run_batch(batch, terminal_bind=terminal_bind)
        except Exception as error:  # noqa: BLE001
            cleanup_failures = prelaunch_owner.cancel_if_unclaimed()
            if len(cleanup_failures) > 0:
                error.add_note(
                    "terminal prelaunch cleanup failed:\n"
                    + "\n".join(cleanup_failures)
                )
            raise

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            self.drain_terminal_scheduler_receipts()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                if self.disagg_prefill_bootstrap_queue.terminal_source:
                    result = self.run_terminal_prefill_batch(batch)
                else:
                    result = self.run_batch(batch)
                self.record_disagg_prefill_producer_event(
                    batch,
                    torch.cuda.current_stream(device=self.device),
                )
                self.process_batch_result(batch, result)
            else:
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            self.drain_terminal_scheduler_receipts()
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                if self.disagg_prefill_bootstrap_queue.terminal_source:
                    batch_result = self.run_terminal_prefill_batch(batch)
                else:
                    batch_result = self.run_batch(batch)
                self.record_disagg_prefill_producer_event(
                    batch,
                    self.forward_stream,
                )
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            forward_completion = None
            if batch and len(self.disagg_prefill_inflight_queue) > 0:
                forward_completion = torch.cuda.Event()
                forward_completion.record(self.forward_stream)
            self.progress_disagg_prefill_transfers_during_forward(forward_completion)

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def progress_disagg_prefill_transfers_during_forward(
        self: Scheduler,
        forward_completion: torch.cuda.Event | None,
    ) -> None:
        """Observe completed transfers while the next prefill uses the GPU.

        Overlap scheduling submits the current forward before it resolves the
        previous batch and starts that batch's KV transfer. A single immediate
        poll commonly observes the transfer in progress, then the scheduler
        blocks on the current forward before checking again. Polling during
        that already-enqueued forward lets transfer completion become visible
        without serializing either operation. A fixed cadence plus count and
        wall-time budgets bound CPU and Gloo pressure even if the forward is
        anomalously long.

        :param forward_completion: Event recorded after the current forward,
            or ``None`` when no forward is available to hide progress work.
        """

        progress_deadline = (
            time.monotonic() + DISAGG_PREFILL_TRANSFER_PROGRESS_TIME_BUDGET_SECONDS
        )
        self.process_disagg_prefill_inflight_queue()
        if forward_completion is None:
            return

        for _ in range(DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS - 1):
            if not self.disagg_prefill_progress_may_continue_on_all_ranks(
                forward_completion,
                progress_deadline,
            ):
                return
            time.sleep(DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS)
            self.process_disagg_prefill_inflight_queue()

        logger.debug(
            "Prefill transfer progress exhausted its %d-poll budget with "
            "%d requests still in flight",
            DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS,
            len(self.disagg_prefill_inflight_queue),
        )

    def disagg_prefill_progress_may_continue_on_all_ranks(
        self: Scheduler,
        forward_completion: torch.cuda.Event,
        progress_deadline: float,
    ) -> bool:
        """Agree whether every rank may execute one more transfer poll.

        Every polling iteration contains TP/CP collectives. A rank-local event
        or queue decision could let one participant leave while another enters
        the next collective. A local time deadline is safe only as an input to
        this consensus. The loop continues only while every rank has work,
        time, and an incomplete forward.

        :param forward_completion: Event recorded after the current forward.
        :param progress_deadline: Monotonic deadline for opportunistic polling.
        :returns: Whether all TP/CP participants may enter another poll.
        """

        locally_pending = (
            len(self.disagg_prefill_inflight_queue) > 0
            and time.monotonic() < progress_deadline
            and not forward_completion.query()
        )
        pending = torch.tensor(
            [int(locally_pending)],
            dtype=torch.uint8,
            device="cpu",
        )
        dist.all_reduce(
            pending,
            op=dist.ReduceOp.MIN,
            group=self.attn_tp_cpu_group,
        )
        dist.all_reduce(
            pending,
            op=dist.ReduceOp.MIN,
            group=self.attn_cp_cpu_group,
        )
        return int(pending.item()) == 1

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        if manager.uses_terminal_source_publication():
            self.process_batch_result_terminal_disagg_prefill(batch, result)
            return
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        producer_event = batch.disagg_kv_producer_event
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = (
                        draft_input.hidden_states[i].cpu().clone()
                    )
                    dsa_topk_indices = batch.spec_info.dsa_topk_indices
                    if dsa_topk_indices is not None:
                        req.output_dsa_topk_indices = dsa_topk_indices[i].cpu().clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                if not req.pending_bootstrap:
                    self.send_kv_chunk(
                        req,
                        last_chunk=True,
                        producer_event=producer_event,
                    )
                elif producer_event is not None:
                    self.disagg_prefill_deferred_producer_events[req.rid] = (
                        producer_event
                    )
                req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if self.enable_overlap and not req.pending_bootstrap:
                    assert (
                        req.metadata_buffer_index >= 0
                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(
                        req,
                        last_chunk=False,
                        end_idx=req.tmp_end_idx,
                        producer_event=producer_event,
                    )
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_batch_result_terminal_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """Resolve bookkeeping without touching owner-managed terminal state.

        Final requests were already bound immediately after model submission.
        This method therefore cannot mutate output state, enqueue transfers,
        publish through the legacy streamer, or enter the polling queue.

        :param batch: Exact prefill batch whose immutable plans were bound.
        :param result: Model result retained only for metrics and cleanup.
        """

        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None
        scheduler_local = self.process_scheduler_local_fake_prefill_results(
            batch,
            result,
        )
        for index, req in enumerate(batch.reqs):
            if scheduler_local[index]:
                continue
            if req.inflight_middle_chunks <= 0:
                if req.rid not in self.disagg_prefill_terminal_requests:
                    raise RuntimeError("final terminal request was not launch-bound")
                req.time_stats.set_prefill_finished_time()
                req.time_stats.set_prefill_transfer_queue_entry_time()
                continue
            req.inflight_middle_chunks -= 1
            req.time_stats.set_last_chunked_prefill_finish_time()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_scheduler_local_fake_prefill_results(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> tuple[bool, ...]:
        """Complete fake prefill warmups without publication state.

        Fake requests exercise model execution, then return their sampled token
        directly. They do not own a terminal identity, DFlash boundary row,
        transfer route, or legacy metadata row.

        :param batch: Exact prefill batch containing optional local warmups.
        :param result: Model result providing sampled warmup tokens.
        :returns: Per-row mask identifying scheduler-local fake requests.
        """

        scheduler_local = tuple(
            _is_fake_transfer(req, self.server_args) for req in batch.reqs
        )
        final_indices = [
            index
            for index, req in enumerate(batch.reqs)
            if scheduler_local[index] and req.inflight_middle_chunks <= 0
        ]
        if len(final_indices) == 0:
            for index, req in enumerate(batch.reqs):
                if not scheduler_local[index]:
                    continue
                req.inflight_middle_chunks -= 1
                req.time_stats.set_last_chunked_prefill_finish_time()
            return scheduler_local
        if type(result.next_token_ids) is not torch.Tensor:
            raise TypeError("fake prefill warmup requires device-resident token ids")
        for index in final_indices:
            req = batch.reqs[index]
            if req.pending_bootstrap:
                raise RuntimeError(
                    "scheduler-local fake prefill completed before local bootstrap"
                )
            if req.metadata_buffer_index != -1:
                raise RuntimeError(
                    "scheduler-local fake prefill retained a metadata row"
                )

        next_token_ids = result.next_token_ids[final_indices].tolist()
        completed: list[Req] = []
        token_by_index = dict(zip(final_indices, next_token_ids, strict=True))
        for index, req in enumerate(batch.reqs):
            if not scheduler_local[index]:
                continue
            if req.inflight_middle_chunks > 0:
                req.inflight_middle_chunks -= 1
                req.time_stats.set_last_chunked_prefill_finish_time()
                continue

            next_token_id = int(token_by_index[index])
            req.time_stats.set_prefill_finished_time()
            req.output_ids.append(next_token_id)
            maybe_cache_unfinished_req(req, self.tree_cache)
            release_kv_cache(req, self.tree_cache)
            if not isinstance(req.finished_reason, FINISH_ABORT):
                req.finished_reason = FINISH_LENGTH(length=0)
            req.disagg_kv_sender.clear()
            req.time_stats.set_prefill_transfer_queue_entry_time()
            req.time_stats.set_prefill_kv_transfer_finish_time()
            req.time_stats.set_completion_time()
            completed.append(req)

            if req.grammar is not None:
                try:
                    req.grammar.accept_token(next_token_id)
                except ValueError as error:
                    prepare_abort(
                        req,
                        f"Grammar accept_token failed for req {req.rid} with "
                        f"token {next_token_id}: {error}",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                req.grammar.finished = req.finished()

        if len(completed) > 0:
            self.output_streamer.stream_output(
                completed,
                any(req.return_logprob for req in completed),
                None,
            )
        return scheduler_local

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                self.handle_inflight_transfer_failure(req)
                done_reqs.append(req)
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler, req: Req
    ) -> Optional[Exception]:
        """Conclude an inflight request whose KV transfer failed."""
        self.disagg_prefill_deferred_producer_events.pop(req.rid, None)
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        # Mute error message for propagated exceptions to avoid duplicate logging
        if getattr(exc, "is_from_another_rank", False):
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        release_kv_cache(req, self.tree_cache)  # unlock the tree
        if not isinstance(req.finished_reason, FINISH_ABORT):
            prepare_abort(
                req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_transfer_failed_reqs()
        return exc

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        self.disagg_prefill_deferred_producer_events.pop(req.rid, None)
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if (
            req.req_pool_idx is not None
            or req.kv is not None
            or req.mamba_pool_idx is not None
        ):
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        last_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
    ) -> None:
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.disagg_prefill_bootstrap_queue.terminal_source:
                pass
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                producer_event = (
                    last_batch.disagg_kv_producer_event
                    if last_batch is not None
                    else None
                )
                self.send_kv_chunk(req, producer_event=producer_event)

            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        if last_batch and last_batch.forward_mode.is_extend():
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        if self.disagg_prefill_bootstrap_queue.terminal_source:
            return
        # Only bootstrap-finalized requests; staging excluded.
        if (
            not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
            or self.enable_staging
            or req.pending_bootstrap
        ):
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = len(req.prefix_indices) - req.host_hit_length
        if cached_end <= req.start_send_idx:
            return
        assert cached_end % self.token_to_kv_pool_allocator.page_size == 0
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        producer_event = None
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            producer_event = ev
        self.send_kv_chunk(
            req,
            last_chunk=False,
            end_idx=cached_end,
            producer_event=producer_event,
        )

    def freeze_disagg_prefill_final_geometry(
        self: Scheduler,
        req: Req,
    ) -> tuple[np.ndarray, Optional[List]]:
        """Freeze all remaining main and state pages for final migration.

        This helper has no transport or metadata-buffer side effects. Terminal
        source publication calls it before model submission, while the legacy
        sender calls it immediately before enqueueing its final transfer.

        :param req: Exact final prefill request retaining source cache pages.
        :returns: Complete remaining main pages and final state projections.
        """

        if req.extend_range is None:
            raise RuntimeError("final prefill request has no extend range")
        page_size = self.token_to_kv_pool_allocator.page_size
        transfer_input_len = len(req.origin_input_ids)
        end_idx = min(req.extend_range.end, transfer_input_len)
        if end_idx < req.start_send_idx:
            raise RuntimeError("final prefill migration range is reversed")

        seq_len = end_idx
        c128_seq_len = transfer_input_len

        def mamba_payload() -> list[np.ndarray]:
            return [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]

        def swa_payload() -> np.ndarray:
            window_start = max(0, seq_len - self.sliding_window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:seq_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            return kv_to_page_indices(window_kv_indices_swa, page_size)

        def dsa_payload() -> np.ndarray:
            kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ]
            return kv_to_page_indices(kv_indices_full, page_size)

        def swa_ring_payload() -> np.ndarray:
            pool = self.token_to_kv_pool_allocator.get_kvcache()
            ring_stride = pool.unified_swa_ring_size
            window_start = max(0, seq_len - pool.unified_swa_window)
            positions = np.arange(window_start, seq_len, dtype=np.int64)
            state_slot = int(req.req_pool_idx)
            ring_rows = state_slot * ring_stride + (positions % ring_stride)
            return ring_rows.astype(np.int32)

        def c128_state_payload() -> np.ndarray:
            online = is_dsv4_c128_online_enabled()
            ring_size = (
                1
                if online
                else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(128)
            )
            return get_dsv4_c128_state_indices(
                int(req.req_pool_idx),
                c128_seq_len,
                online=online,
                ring_size=ring_size,
            )

        state_types = self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
        payloads = {
            StateType.MAMBA: mamba_payload,
            StateType.SWA: swa_payload,
            StateType.DSA: dsa_payload,
            StateType.MINIMAX_INDEX_K: dsa_payload,
            StateType.SWA_RING: swa_ring_payload,
            StateType.C128_STATE: c128_state_payload,
        }
        if _is_npu and isinstance(
            self.token_to_kv_pool_allocator.get_kvcache(),
            DeepSeekV4TokenToKVPool,
        ):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                dsv4_state_payloads,
            )

            payloads.update(
                dsv4_state_payloads(
                    self.req_to_token_pool,
                    req.req_pool_idx,
                    seq_len,
                    page_size,
                    self.sliding_window_size,
                    prefix_len=0,
                )
            )
        state_indices = [
            payloads[state_type]() if state_type in payloads else None
            for state_type in state_types
        ]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.start_send_idx:end_idx
        ]
        return kv_to_page_indices(kv_indices, page_size), state_indices

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
        producer_event: torch.cuda.Event | None = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        manager = self.disagg_prefill_bootstrap_queue.kv_manager
        if manager.uses_terminal_source_publication():
            raise RuntimeError(
                "terminal source request cannot enter the legacy transfer queue"
            )
        if last_chunk:
            deferred_event = self.disagg_prefill_deferred_producer_events.pop(
                req.rid,
                None,
            )
            if producer_event is None:
                producer_event = deferred_event

        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        page_indices: np.ndarray
        state_indices: Optional[List] = None
        if last_chunk:
            metadata_buffers = self.disagg_metadata_buffers
            if metadata_buffers is None:
                raise RuntimeError("legacy transfer has no metadata buffers")
            metadata_buffers.set_buf(req)
            page_indices, state_indices = self.freeze_disagg_prefill_final_geometry(
                req
            )
        else:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_idx:end_idx
            ]
            page_indices = kv_to_page_indices(kv_indices, page_size)
        if not req.disagg_kv_sender.should_send_kv_chunk(len(page_indices), last_chunk):
            return
        req.disagg_kv_sender.send(
            page_indices,
            state_indices,
            producer_event=producer_event,
        )
        req.start_send_idx = end_idx

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        self.disagg_prefill_deferred_producer_events.pop(req.rid, None)
        max_attempts = self.server_args.optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
