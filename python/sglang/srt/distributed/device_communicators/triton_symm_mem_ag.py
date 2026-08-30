# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory ``multimem.st`` all-gather along the hidden (last) dim.

Each rank stores its ``[T, H/TP]`` shard into a multicast buffer in one NVLink
pass instead of an NCCL ring; ``create_state`` rendezvous once so launches are
CUDA-graph capturable.
"""

import logging
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Each thread moves _NUMEL_PER_THREAD bf16 via one 128-bit multimem op; the
# grid-strided block count is tunable in [_MIN_BLOCKS, _MAX_BLOCKS].
_BLOCK_THREADS = 1024
_NUMEL_PER_THREAD = 8
_MIN_BLOCKS = 4
_MAX_BLOCKS = 32
_INPUT_DIMENSIONS = 2

_ConsensusValue = TypeVar("_ConsensusValue")


# ------------------------------------------------------------------------------
# Low-level PTX helpers
# ------------------------------------------------------------------------------


@triton.jit
def _multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _local_ld_128(in_ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[in_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _send_signal_release(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal_acquire(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _blockwise_barrier(
    signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    block_id = (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        if sem == "relaxed":
            _send_signal(send_addrs)
            _wait_signal(wait_addrs)
        else:
            _send_signal_release(send_addrs)
            _wait_signal_acquire(wait_addrs)


@triton.jit
def _all_gather_kernel_inner(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    total_tokens,
    hidden_offset,
    LOCAL_HIDDEN: tl.constexpr,
    TOTAL_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SKIP_ENTRY_SYNC: tl.constexpr,
) -> None:
    if SKIP_ENTRY_SYNC == 0:
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="relaxed")
        _sync_threads()

    chunks_per_row: tl.constexpr = LOCAL_HIDDEN // NUMEL_PER_THREAD
    total_hidden_chunks: tl.constexpr = TOTAL_HIDDEN // NUMEL_PER_THREAD
    hidden_offset_chunks = hidden_offset // NUMEL_PER_THREAD
    total_chunks = total_tokens * chunks_per_row

    pid = tl.program_id(axis=0)
    tid = _get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < total_chunks:
        chunk = block_start + tid
        mask = chunk < total_chunks
        row = chunk // chunks_per_row
        col_chunk = chunk % chunks_per_row

        in_ptr = input_ptr.to(tl.pointer_type(tl.uint64)) + chunk * 2
        out_chunk = row * total_hidden_chunks + hidden_offset_chunks + col_chunk
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + out_chunk * 2
        )
        x, y, z, w = _local_ld_128(in_ptr, mask)
        _multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="acq_rel")


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


@dataclass
class MultimemAllGatherState:
    """Resources for one tensor-parallel multimem all-gather.

    :ivar group: Device process group used by the symmetric-memory rendezvous.
    :ivar rank_in_group: Local rank within the tensor-parallel group.
    :ivar world_size: Tensor-parallel group size.
    :ivar device: CUDA device owning the symmetric buffer.
    :ivar max_token_num: Maximum token rows held by the symmetric buffer.
    :ivar hidden_dim: Gathered hidden width.
    :ivar comm_buff: Rank-local view of the symmetric communication buffer.
    :ivar symm_mem_hdl: Stable symmetric-memory rendezvous handle.
    """

    group: dist.ProcessGroup
    rank_in_group: int
    world_size: int
    device: torch.device
    max_token_num: int
    hidden_dim: int
    comm_buff: torch.Tensor
    # Rendezvous handle; stable for the buffer's lifetime, resolved once.
    symm_mem_hdl: Any


def create_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device | None = None,
) -> MultimemAllGatherState:
    """Allocate and rendezvous the symmetric-memory buffer. Collective: call
    once outside CUDA-graph capture with identical args on every rank."""
    assert type(group) is dist.ProcessGroup, f"Expected ProcessGroup, got {type(group)}"
    assert hidden_size % _NUMEL_PER_THREAD == 0, (
        f"hidden_size={hidden_size} must be a multiple of {_NUMEL_PER_THREAD} "
        f"bf16 for 16-byte multimem.st row alignment"
    )
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

    # Pad holds _MAX_BLOCKS * world_size uint32 slots; max() never shrinks it.
    pad_bytes = _MAX_BLOCKS * group.size() * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
    with torch.inference_mode(False), torch.no_grad():
        comm_buff = symm_mem.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
    hdl = symm_mem.rendezvous(comm_buff, group=group)
    assert hdl.rank == rank_in_group, (
        f"symm_mem handle rank {hdl.rank} != rank_in_group {rank_in_group}; the "
        f"hidden-shard offset would be wrong"
    )
    return MultimemAllGatherState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=comm_buff,
        symm_mem_hdl=hdl,
    )


def _launch_config(local_numel: int):
    assert local_numel % _NUMEL_PER_THREAD == 0
    return _MIN_BLOCKS, _BLOCK_THREADS, _BLOCK_THREADS // 32, _NUMEL_PER_THREAD


def all_gather_inner(
    state: MultimemAllGatherState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int,
    skip_entry_sync: bool = False,
    safe: bool = True,
) -> torch.Tensor:
    """Gather ``[T, H/TP]`` shards into ``[T, H]`` along the hidden dim.

    ``tp_hidden_dim`` is the gathered width ``H``. Returns a clone when ``safe``,
    else a view into the symmetric buffer (valid until the next collective)."""
    world_size = state.world_size
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.data_ptr() % 16 == 0, (
        f"hidden_states.data_ptr()={hex(hidden_states.data_ptr())} must be "
        f"16-byte aligned for 128-bit multimem.st"
    )
    assert (
        tp_hidden_dim % world_size == 0
    ), f"tp_hidden_dim={tp_hidden_dim} must be divisible by world_size={world_size}"
    local_hidden = tp_hidden_dim // world_size
    assert local_hidden % _NUMEL_PER_THREAD == 0, (
        f"per-rank hidden shard ({local_hidden}) must be a multiple of "
        f"{_NUMEL_PER_THREAD} bf16"
    )
    assert tp_hidden_dim <= state.hidden_dim, (
        f"comm buffer too narrow: tp_hidden_dim={tp_hidden_dim} > "
        f"state.hidden_dim={state.hidden_dim}"
    )
    total_tokens, in_hidden = hidden_states.shape
    assert (
        in_hidden == local_hidden
    ), f"input hidden ({in_hidden}) != this rank's shard ({local_hidden})"
    assert (
        total_tokens <= state.max_token_num
    ), f"total_tokens={total_tokens} exceeds max_token_num={state.max_token_num}"

    hidden_offset = local_hidden * state.rank_in_group
    symm_mem_hdl = state.symm_mem_hdl
    num_blocks, block_size, num_warps, numel_per_thread = _launch_config(
        total_tokens * local_hidden
    )
    grid = (num_blocks, 1, 1)
    _all_gather_kernel_inner[grid](
        input_ptr=hidden_states,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        total_tokens=total_tokens,
        hidden_offset=hidden_offset,
        LOCAL_HIDDEN=local_hidden,
        TOTAL_HIDDEN=state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        SKIP_ENTRY_SYNC=1 if skip_entry_sync else 0,
        num_warps=num_warps,
    )
    output = state.comm_buff[:total_tokens, :tp_hidden_dim]
    return output.clone() if safe else output


# ------------------------------------------------------------------------------
# Guarded wrapper
# ------------------------------------------------------------------------------


def recommended_max_tokens(include_prefill: bool, floor: int = 0) -> int:
    """Return the largest token batch retained by the multimem buffer.

    Larger replicated batches use NCCL without changing the committed
    multimem mode.

    :param include_prefill: Include configured prefill limits.
    :param floor: Minimum returned capacity.
    :returns: Recommended symmetric-buffer token capacity.
    """
    try:
        from sglang.srt.runtime_context import get_schedule, get_spec

        def g(value) -> int:
            return value if isinstance(value, int) and value > 0 else 0

        schedule, spec = get_schedule(), get_spec()
        tokens = g(schedule.max_running_requests) * max(
            g(spec.speculative_num_draft_tokens), g(spec.speculative_eagle_topk), 1
        )
        if include_prefill:
            tokens = max(
                tokens, g(schedule.chunked_prefill_size), g(schedule.max_prefill_tokens)
            )
        return max(tokens, floor)
    except ValueError:
        return floor


@dataclass(frozen=True)
class _InitializationRecord:
    """Replicated contract used to select one permanent dispatch mode.

    :ivar caller_name: Stable call-site identity.
    :ivar process_enabled: Whether the dedicated process-wide multimem policy is
        enabled.
    :ivar caller_enabled: Whether this call site can use tensor-parallel multimem.
    :ivar max_tokens: Symmetric-buffer token capacity.
    :ivar skip_entry_sync: Whether the reusable-buffer entry barrier is omitted.
    :ivar world_size: Tensor-parallel group size.
    :ivar nnodes: Deployment node count.
    :ivar input_shape: First-call local tensor shape.
    :ivar input_dtype: First-call tensor dtype.
    :ivar device_type: First-call tensor device type.
    :ivar device_capability: CUDA compute capability, when applicable.
    :ivar stream_capturing: Whether initialization was attempted during capture.
    """

    caller_name: str
    process_enabled: bool
    caller_enabled: bool
    max_tokens: int
    skip_entry_sync: bool
    world_size: int
    nnodes: int
    input_shape: tuple[int, ...]
    input_dtype: str
    device_type: str
    device_capability: tuple[int, int] | None
    stream_capturing: bool


def _all_gather_consensus_values(
    value: _ConsensusValue,
    group: dist.ProcessGroup,
    world_size: int,
) -> list[_ConsensusValue]:
    """Gather one initialization value from every tensor-parallel rank.

    :param value: Rank-local value.
    :param group: CPU process group for control-plane consensus.
    :param world_size: Expected group size.
    :returns: Values in process-group rank order.
    """
    gathered: list[_ConsensusValue | None] = [None] * world_size
    dist.all_gather_object(gathered, value, group=group)
    if any(item is None for item in gathered):
        raise RuntimeError("multimem initialization consensus returned no value")
    return cast(list[_ConsensusValue], gathered)


def _require_identical_records(
    records: list[_InitializationRecord],
) -> _InitializationRecord:
    """Validate that every rank entered with the same static contract.

    :param records: Initialization records in rank order.
    :returns: The common initialization record.
    :raises RuntimeError: If the records are empty or differ across ranks.
    """
    if len(records) == 0:
        raise RuntimeError("multimem initialization consensus returned no records")
    expected = records[0]
    if any(record != expected for record in records[1:]):
        raise RuntimeError(
            "multimem all-gather initialization contract differs across TP ranks: "
            f"{records}"
        )
    return expected


def _static_ineligibility_reason(record: _InitializationRecord) -> str | None:
    """Return why a replicated initialization contract cannot use multimem.

    :param record: Replicated first-call contract.
    :returns: A reason for permanent NCCL dispatch, or ``None`` when eligible.
    """
    if not record.caller_enabled:
        return "the caller does not use the tensor-parallel device group"
    if record.world_size <= 1:
        return "tensor parallelism has one rank"
    if record.nnodes != 1:
        return "the deployment spans multiple nodes"
    if record.device_type != "cuda":
        return f"device type {record.device_type!r} is not CUDA"
    if record.input_dtype != str(torch.bfloat16):
        return f"dtype {record.input_dtype} is not bfloat16"
    if len(record.input_shape) != _INPUT_DIMENSIONS:
        return f"input rank {len(record.input_shape)} is not two"
    if record.input_shape[-1] % _NUMEL_PER_THREAD != 0:
        return (
            f"local hidden width {record.input_shape[-1]} is not divisible by "
            f"{_NUMEL_PER_THREAD}"
        )
    return None


class MultimemAllGatherer:
    """Tensor-parallel all-gather with one-time replicated multimem selection.

    The dedicated process flag is part of the first-call consensus and controls
    whether symmetric-memory initialization is attempted. The first call
    reaches TP consensus on the static input contract, then
    permanently selects NCCL or builds one symmetric buffer. A rendezvous
    failure after that consensus is fatal; it never becomes a rank-local NCCL
    fallback. Committed multimem calls require the same dtype, device, and local
    hidden width on every rank. Token count is a TP-replicated dynamic value, so
    batches larger than the buffer safely use NCCL without another control
    collective.

    ``skip_entry_sync=True`` is valid only when another cross-rank
    synchronization separates every pair of calls.
    """

    def __init__(
        self,
        max_tokens: int,
        *,
        name: str,
        enabled: bool = True,
        skip_entry_sync: bool = False,
    ) -> None:
        """Initialize an uncommitted all-gather policy.

        :param max_tokens: Maximum token rows held by the symmetric buffer.
        :param name: Stable identity shared by this call site on every TP rank.
        :param enabled: Whether this caller can use the TP device group.
        :param skip_entry_sync: Omit the reusable-buffer entry barrier.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if len(name) == 0:
            raise ValueError("name must not be empty")
        self._max_tokens = max_tokens
        self._name = name
        self._caller_enabled = enabled
        self._skip_entry_sync = skip_entry_sync
        self._initialized = False
        self._state: MultimemAllGatherState | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Gather one rank-local hidden shard.

        :param x: Rank-local tensor with replicated leading dimensions.
        :returns: Tensor gathered along the final dimension.
        """
        if not self._initialized:
            self._state = self._initialize(x)
            self._initialized = True

        state = self._state
        if state is None:
            return self._nccl_all_gather(x)

        self._validate_committed_input(state, x)
        # Token rows are replicated within a TP collective, so every rank makes
        # the same capacity decision without another control-plane collective.
        if x.shape[0] > state.max_token_num:
            return self._nccl_all_gather(x)

        prepared_x = self._prepare_input(x)
        return all_gather_inner(
            state,
            prepared_x,
            tp_hidden_dim=state.hidden_dim,
            skip_entry_sync=self._skip_entry_sync,
            safe=False,
        )

    @staticmethod
    def _nccl_all_gather(x: torch.Tensor) -> torch.Tensor:
        """Use the ordinary tensor-parallel collective.

        :param x: Rank-local tensor.
        :returns: Tensor gathered along the final dimension.
        """
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(x, dim=-1)

    def _initialize(self, x: torch.Tensor) -> MultimemAllGatherState | None:
        """Select a permanent dispatch mode and optionally build multimem state.

        :param x: First rank-local input.
        :returns: Committed multimem state, or ``None`` for permanent NCCL.
        """
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return None

        device_capability = (
            torch.cuda.get_device_capability(x.device) if x.is_cuda else None
        )
        stream_capturing = (
            torch.cuda.is_current_stream_capturing() if x.is_cuda else False
        )
        local_record = _InitializationRecord(
            caller_name=self._name,
            process_enabled=server_args.enable_multimem_all_gather,
            caller_enabled=self._caller_enabled,
            max_tokens=self._max_tokens,
            skip_entry_sync=self._skip_entry_sync,
            world_size=tp_group.world_size,
            nnodes=server_args.nnodes,
            input_shape=tuple(x.shape),
            input_dtype=str(x.dtype),
            device_type=x.device.type,
            device_capability=device_capability,
            stream_capturing=stream_capturing,
        )
        records = _all_gather_consensus_values(
            local_record,
            tp_group.cpu_group,
            tp_group.world_size,
        )
        record = _require_identical_records(records)
        if not record.process_enabled:
            return None
        if record.stream_capturing:
            raise RuntimeError(
                f"{self._name} multimem initialization must run before CUDA "
                "graph capture"
            )

        ineligibility_reason = _static_ineligibility_reason(record)
        if ineligibility_reason is not None:
            if tp_group.rank_in_group == 0:
                logger.warning(
                    "%s multimem all-gather disabled: %s",
                    self._name,
                    ineligibility_reason,
                )
            return None

        state = create_state(
            group=tp_group.device_group,
            rank_in_group=tp_group.rank_in_group,
            max_tokens=self._max_tokens,
            hidden_size=x.shape[-1] * tp_group.world_size,
            device=x.device,
        )
        multicast_available = state.symm_mem_hdl.multicast_ptr != 0
        multicast_availability = _all_gather_consensus_values(
            multicast_available,
            tp_group.cpu_group,
            tp_group.world_size,
        )
        if all(multicast_availability):
            return state
        if any(multicast_availability):
            raise RuntimeError(
                f"{self._name} multicast availability differs across TP ranks: "
                f"{multicast_availability}"
            )
        if tp_group.rank_in_group == 0:
            logger.warning(
                "%s multimem all-gather disabled: CUDA multicast is unavailable "
                "for TP%d on compute capability %s",
                self._name,
                tp_group.world_size,
                record.device_capability,
            )
        return None

    @staticmethod
    def _validate_committed_input(
        state: MultimemAllGatherState,
        x: torch.Tensor,
    ) -> None:
        """Validate the static contract after multimem has been committed.

        :param state: Committed multimem state.
        :param x: Current rank-local input.
        :raises RuntimeError: If a static input property changed.
        """
        if x.dtype != torch.bfloat16:
            raise RuntimeError(
                f"committed multimem dtype changed from bfloat16 to {x.dtype}"
            )
        if x.device != state.device:
            raise RuntimeError(
                f"committed multimem device changed from {state.device} to {x.device}"
            )
        if x.dim() != _INPUT_DIMENSIONS:
            raise RuntimeError(
                f"committed multimem input rank changed from 2 to {x.dim()}"
            )
        expected_local_hidden = state.hidden_dim // state.world_size
        if x.shape[-1] != expected_local_hidden:
            raise RuntimeError(
                "committed multimem local hidden width changed from "
                f"{expected_local_hidden} to {x.shape[-1]}"
            )

    @staticmethod
    def _prepare_input(x: torch.Tensor) -> torch.Tensor:
        """Repair rank-local layout without changing collective dispatch.

        :param x: Valid rank-local multimem input.
        :returns: Contiguous, 16-byte-aligned input.
        :raises RuntimeError: If a fresh contiguous allocation is still unaligned.
        """
        prepared_x = x if x.is_contiguous() else x.contiguous()
        if prepared_x.data_ptr() % 16 != 0:
            prepared_x = prepared_x.clone()
        if prepared_x.data_ptr() % 16 != 0:
            raise RuntimeError("fresh multimem input allocation is not 16-byte aligned")
        return prepared_x
