import ctypes
import dataclasses
import struct
import threading
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
)


@dataclasses.dataclass
class TransferKVChunk:
    """Work unit for KV cache transfer from prefill to decode."""

    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last_chunk: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List]
    producer_event: Optional[torch.cuda.Event] = None
    chunk_id: Optional[int] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )


@dataclasses.dataclass(frozen=True)
class TensorParallelShard:
    """Contiguous byte range transferred between two tensor-parallel ranks.

    :ivar source_offset_bytes: Offset within one source token.
    :ivar destination_offset_bytes: Offset within one destination token.
    :ivar length_bytes: Number of bytes transferred for one token.
    """

    source_offset_bytes: int
    destination_offset_bytes: int
    length_bytes: int


def compute_tensor_parallel_shard(
    source_token_bytes: int,
    destination_token_bytes: int,
    source_parallel_size: int,
    destination_parallel_size: int,
    source_rank: int,
    destination_rank: int,
) -> TensorParallelShard:
    """Compute a contiguous non-replicated tensor-parallel token slice.

    :param source_token_bytes: Bytes occupied by one token on the source rank.
    :param destination_token_bytes: Bytes occupied by one token on the destination
        rank.
    :param source_parallel_size: Source attention tensor-parallel width.
    :param destination_parallel_size: Destination attention tensor-parallel width.
    :param source_rank: Source attention tensor-parallel rank.
    :param destination_rank: Destination attention tensor-parallel rank.
    :returns: The source offset, destination offset, and transfer length.
    :raises ValueError: If the layouts are not compatible contiguous partitions or
        the ranks are not connected by the expected tensor-parallel mapping.
    """

    positive_values = {
        "source_token_bytes": source_token_bytes,
        "destination_token_bytes": destination_token_bytes,
        "source_parallel_size": source_parallel_size,
        "destination_parallel_size": destination_parallel_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    if source_rank < 0 or source_rank >= source_parallel_size:
        raise ValueError(
            f"source_rank must be in [0, {source_parallel_size}), got {source_rank}"
        )
    if destination_rank < 0 or destination_rank >= destination_parallel_size:
        raise ValueError(
            "destination_rank must be in "
            f"[0, {destination_parallel_size}), got {destination_rank}"
        )

    source_total_bytes = source_token_bytes * source_parallel_size
    destination_total_bytes = destination_token_bytes * destination_parallel_size
    if source_total_bytes != destination_total_bytes:
        raise ValueError(
            "Tensor-parallel state must be a non-replicated contiguous partition: "
            f"source has {source_total_bytes} aggregate bytes per token, destination "
            f"has {destination_total_bytes}"
        )

    if source_parallel_size == destination_parallel_size:
        if source_rank != destination_rank:
            raise ValueError(
                "Equal-width tensor-parallel ranks must match, got "
                f"source rank {source_rank} and destination rank {destination_rank}"
            )
        return TensorParallelShard(0, 0, source_token_bytes)

    if source_parallel_size > destination_parallel_size:
        if source_parallel_size % destination_parallel_size != 0:
            raise ValueError(
                "Source tensor-parallel width must be divisible by destination "
                f"width, got {source_parallel_size} and {destination_parallel_size}"
            )
        sources_per_destination = source_parallel_size // destination_parallel_size
        expected_destination_rank = source_rank // sources_per_destination
        if destination_rank != expected_destination_rank:
            raise ValueError(
                f"Source rank {source_rank} maps to destination rank "
                f"{expected_destination_rank}, not {destination_rank}"
            )
        destination_offset_bytes = (
            source_rank % sources_per_destination
        ) * source_token_bytes
        return TensorParallelShard(
            source_offset_bytes=0,
            destination_offset_bytes=destination_offset_bytes,
            length_bytes=source_token_bytes,
        )

    if destination_parallel_size % source_parallel_size != 0:
        raise ValueError(
            "Destination tensor-parallel width must be divisible by source width, "
            f"got {destination_parallel_size} and {source_parallel_size}"
        )
    destinations_per_source = destination_parallel_size // source_parallel_size
    expected_source_rank = destination_rank // destinations_per_source
    if source_rank != expected_source_rank:
        raise ValueError(
            f"Destination rank {destination_rank} maps to source rank "
            f"{expected_source_rank}, not {source_rank}"
        )
    source_offset_bytes = (
        destination_rank % destinations_per_source
    ) * destination_token_bytes
    return TensorParallelShard(
        source_offset_bytes=source_offset_bytes,
        destination_offset_bytes=0,
        length_bytes=destination_token_bytes,
    )


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n+1}I", n, *(len(b) for b in buffers))
    return header + b"".join(buffers)


def unpack_list_of_buffers(buf: bytes) -> List[bytes]:
    if buf == b"":
        return []
    (n,) = struct.unpack("<I", buf[:4])
    lens = struct.unpack(f"<{n}I", buf[4 : 4 + 4 * n])
    out = []
    offset = 4 + 4 * n
    for length in lens:
        out.append(buf[offset : offset + length])
        offset += length
    return out


def pack_int_lists(lists, fmt: str) -> bytes:
    return pack_list_of_buffers([struct.pack(f"<{len(a)}{fmt}", *a) for a in lists])


def unpack_int_lists(buf: bytes, fmt: str) -> List[List[int]]:
    width = struct.calcsize(fmt)
    return [
        list(struct.unpack(f"<{len(b)//width}{fmt}", b))
        for b in unpack_list_of_buffers(buf)
    ]


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers."""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes."""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer."""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    # src/dst indices are transferred pairwise, so an empty side means there is
    # nothing to transfer. Guarding both sides (not just src) avoids a cryptic
    # NumPy broadcast error from np.diff() below when only one side is empty, e.g.
    # a non-empty prefill DSA/SWA state list paired with an empty decode registration.
    if src_indices.size == 0 or dst_indices.size == 0:
        return [], []

    if src_indices.size != dst_indices.size:
        raise ValueError(
            "group_concurrent_contiguous requires equal-length src/dst index arrays, "
            f"got {src_indices.size} and {dst_indices.size}"
        )

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
