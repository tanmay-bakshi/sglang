from __future__ import annotations

import asyncio
import base64
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import struct
import threading
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import torch
import zmq
from aiohttp import web

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.decode_allocation_lease import (
        DecodeAllocationLease,
        DecodeAllocationLeaseAuthority,
    )
    from sglang.srt.disaggregation.common.staging_handler import StagingTransferInfo
    from sglang.srt.disaggregation.nixl.packed_staging_request import (
        PackedDecodeRequestTransaction,
        PackedRequestPublication,
    )

    from nixl._api import nixl_remote_agent_handle

from sglang.srt.disaggregation.base.conn import (
    KVArgs,
    KVPoll,
    StateType,
    TerminalPrefillAuthorityMismatch,
    TerminalPrefillAuthorityUnavailable,
    TerminalPrefillRequestAuthority,
)
from sglang.srt.disaggregation.common.asymmetric_kv_geometry import (
    require_uniform_asymmetric_kv_entry_geometry,
)
from sglang.srt.disaggregation.common.conn import (
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    BootstrapPostRoute,
    BootstrapRouteHandler,
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
    PrefillServerInfo,
    decode_nixl_agent_metadata,
    validate_nixl_agent_metadata,
    validate_nixl_agent_name,
    validate_serialized_rank,
)
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryPlan,
    PackedDFlashBoundaryCounters,
    PackedReady,
    PackedRequestTeardown,
    PackedTerminalReceipt,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    PackedWireMessage,
    decode_packed_message,
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_handler import (
    STAGING_WATERMARK_WAIT_S,
    StagingRegisterInfo,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    compute_tensor_parallel_shard,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.nixl.decode import PackedNixlDecodeController
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_CONTROL_TAG,
    PACKED_KV_TRANSFER_PROTOCOL,
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedControlSender,
    PackedDecodeControlSender,
    PackedLegacyAuxiliarySource,
    PackedNoncanonicalAuxiliarySource,
    PackedPrefillAuxiliarySource,
    PackedPrefillLaunchPlan,
    PackedPrefillRuntime,
    PackedPrefillSubmission,
    PackedRegistrationAdvertisement,
    PackedTerminalDFlashAuxiliarySource,
    build_same_host_visibility_policy,
    decode_packed_control_frames,
    encode_packed_control_frames,
    load_exact_nixl_runtime_artifacts,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedComponentPages,
    PackedDestinationRegistration,
    PackedPeerIdentity,
)
from sglang.srt.disaggregation.nixl.source_publication_control import (
    TERMINAL_SOURCE_PUBLICATION_RECEIPT_TAG,
    TerminalSourcePublicationControl,
    TerminalSourcePublicationDelivery,
    TerminalSourcePublicationRouteRoster,
)
from sglang.srt.disaggregation.nixl.startup_decode_routes import (
    TerminalDecodeControlRouteTable,
    build_terminal_decode_control_route_table,
    decode_terminal_decode_control_route_table,
    encode_terminal_decode_control_route_table,
)
from sglang.srt.disaggregation.nixl.startup_enrollment_ack import (
    TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
    build_terminal_startup_enrollment_ack,
    decode_terminal_startup_enrollment_ack,
    encode_terminal_startup_enrollment_ack,
)
from sglang.srt.disaggregation.nixl.startup_source_roster import (
    TERMINAL_NIXL_SOURCE_ROSTER_ROUTE,
    TerminalNixlSourceRoster,
    TerminalNixlSourceRoute,
    encode_terminal_nixl_source_roster,
    fetch_terminal_nixl_source_roster,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)
from sglang.srt.disaggregation.terminal_progress.clock import SystemTerminalOwnerClock
from sglang.srt.disaggregation.terminal_progress.cohort_expectation import (
    build_terminal_startup_cohort_expectation,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
    PackedTerminalDecodeWireDelivery,
    PackedTerminalDecodeWork,
)
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentCohort,
    TerminalDeploymentLocalService,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFlashBoundaryDeviceRowPool,
    DFlashBoundaryPrefillSource,
    DFlashBoundarySourceTransfer,
    DFlashBoundarySourceTransportOwner,
)
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlTerminalOwner,
    grouped_nixl_source_members,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
)
from sglang.srt.disaggregation.terminal_progress.output_projection import (
    PrefillTerminalGatewayOutputProjection,
    PrefillTerminalGatewayPayloadEncoder,
    TerminalGatewayResultSlot,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    PackedTerminalOutputPublisher,
    TerminalGatewayPublicationResult,
    ZmqTerminalGatewaySinkFactory,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.request_registration import (
    PackedTerminalDecodeRequestAuthority,
    PackedTerminalRequestRegistrationError,
    build_packed_terminal_decode_request_authority,
    project_packed_terminal_source_authority,
    require_source_plan_request_key,
)
from sglang.srt.disaggregation.terminal_progress.runtime_enrollment import (
    TerminalRankRuntimeConfig,
    TerminalRankRuntimeEnrollment,
    TerminalRankRuntimeEnrollmentFactory,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.serving_reactor import (
    PackedTerminalProcessReactor,
    PackedTerminalProcessReactorFailure,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
    PackedTerminalSourcePlan,
    decode_packed_terminal_source_plan,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceResourceInventory,
    PackedTerminalSourceServing,
    PackedTerminalSourceWork,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceCancellationDisposition,
    PackedTerminalSourceMetric,
    PackedTerminalSourceMetricsSink,
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
    join_terminal_startup_rank,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortError,
    TerminalStartupCohortRegistry,
    TerminalStartupRankAdvertisement,
    decode_terminal_startup_rank_advertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_http import (
    TERMINAL_STARTUP_ROUTE,
    handle_terminal_startup_join,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceipt,
    TerminalWireReceiptImportNamespace,
    TerminalWireReceiptIssuer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    build_transfer_entry_pairs,
    compute_mamba_state_slice_byte_blocks,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

try:
    from nixl._bindings import (
        nixlBackendError,
        nixlCancelledError,
        nixlRemoteDisconnectError,
    )

    _NIXL_TRANSPORT_ERRORS = (
        nixlRemoteDisconnectError,
        nixlBackendError,
        nixlCancelledError,
    )
except ImportError:
    _NIXL_TRANSPORT_ERRORS = (RuntimeError,)

logger = logging.getLogger(__name__)

GUARD = "NixlMsgGuard".encode("ascii")
KV_MEM_KINDS = {"VRAM", "DRAM"}
NIXL_CAPABILITY_READY_TIMEOUT_SECONDS = 5.0
NIXL_CAPABILITY_RETRY_INTERVAL_SECONDS = 0.001
NIXL_RMA_SEGMENT_BYTES = 32 * 1024 * 1024
NIXL_RMA_MAX_DESCRIPTORS = 16 * 1024
NIXL_RMA_MAX_COHORT_DESCRIPTORS = 32 * 1024
NIXL_DIRECT_KV_MAX_COHORT_DESCRIPTORS = 16 * 1024
NIXL_ATTESTATION_SEGMENT_SAMPLE_COUNT = 4
_NIXL_DESCRIPTOR_VALUE_MAX = int(np.iinfo(np.int64).max)
_PACKED_REGISTRATION_FRAME_COUNT = 30


def _build_contiguous_rma_requests(
    src_base_ptr: int,
    dst_base_ptr: int,
    total_bytes: int,
    src_gpu_id: int,
    dst_gpu_id: int,
    *,
    max_segment_bytes: int = NIXL_RMA_SEGMENT_BYTES,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Split one contiguous copy into bounded, aligned NIXL requests.

    :param src_base_ptr: Source address for the first byte.
    :param dst_base_ptr: Destination address for the first byte.
    :param total_bytes: Total number of bytes to cover.
    :param src_gpu_id: Source descriptor GPU identifier.
    :param dst_gpu_id: Destination descriptor GPU identifier.
    :param max_segment_bytes: Maximum bytes in one RMA descriptor.
    :returns: Matching source and destination request arrays.
    :raises ValueError: If a size or identifier is invalid.
    :raises OverflowError: If a descriptor value or covered range exceeds int64.
    """

    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")
    if max_segment_bytes <= 0:
        raise ValueError(f"max_segment_bytes must be positive, got {max_segment_bytes}")

    scalar_values = (
        ("src_base_ptr", src_base_ptr),
        ("dst_base_ptr", dst_base_ptr),
        ("src_gpu_id", src_gpu_id),
        ("dst_gpu_id", dst_gpu_id),
        ("total_bytes", total_bytes),
        ("max_segment_bytes", max_segment_bytes),
    )
    for name, value in scalar_values:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        if value > _NIXL_DESCRIPTOR_VALUE_MAX:
            raise OverflowError(f"{name} exceeds the NIXL int64 descriptor range")

    last_byte_offset = total_bytes - 1
    for name, base_ptr in (
        ("source", src_base_ptr),
        ("destination", dst_base_ptr),
    ):
        if base_ptr > _NIXL_DESCRIPTOR_VALUE_MAX - last_byte_offset:
            raise OverflowError(f"{name} RMA range exceeds the int64 address space")

    segment_count = ((total_bytes - 1) // max_segment_bytes) + 1
    src_reqs = np.empty((segment_count, 3), dtype=np.int64)
    dst_reqs = np.empty((segment_count, 3), dtype=np.int64)
    offset = 0
    remaining_bytes = total_bytes
    for segment_index in range(segment_count):
        segment_length = min(remaining_bytes, max_segment_bytes)
        src_reqs[segment_index] = (
            src_base_ptr + offset,
            segment_length,
            src_gpu_id,
        )
        dst_reqs[segment_index] = (
            dst_base_ptr + offset,
            segment_length,
            dst_gpu_id,
        )
        offset += segment_length
        remaining_bytes -= segment_length

    return src_reqs, dst_reqs


def _bounded_request_slices(
    lengths: npt.NDArray[np.uint64],
    *,
    max_bytes: int = NIXL_RMA_SEGMENT_BYTES,
    max_descriptors: int = NIXL_RMA_MAX_DESCRIPTORS,
) -> tuple[slice, ...]:
    """Partition descriptors into independently postable request slices.

    :param lengths: One positive byte length per descriptor.
    :param max_bytes: Maximum aggregate payload in one transfer handle.
    :param max_descriptors: Maximum descriptors in one transfer handle.
    :returns: Contiguous slices covering every descriptor exactly once.
    :raises ValueError: If a bound or descriptor length is invalid.
    """

    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if max_descriptors <= 0:
        raise ValueError(f"max_descriptors must be positive, got {max_descriptors}")

    flat_lengths = np.asarray(lengths, dtype=np.uint64).reshape(-1)
    if flat_lengths.size == 0:
        return ()

    slices: list[slice] = []
    start = 0
    accumulated_bytes = 0
    for index, raw_length in enumerate(flat_lengths):
        length = int(raw_length)
        if length <= 0:
            raise ValueError(f"descriptor length must be positive, got {length}")
        if length > max_bytes:
            raise ValueError(
                f"descriptor length {length} exceeds transfer bound {max_bytes}"
            )

        descriptor_count = index - start
        exceeds_count = descriptor_count >= max_descriptors
        exceeds_bytes = accumulated_bytes + length > max_bytes
        if descriptor_count > 0 and (exceeds_count or exceeds_bytes):
            slices.append(slice(start, index))
            start = index
            accumulated_bytes = 0
        accumulated_bytes += length

    slices.append(slice(start, flat_lengths.size))
    return tuple(slices)


def _source_cohort_descriptor_limit(source_tp_size: int) -> int:
    """Resolve the per-writer descriptor bound for a source TP cohort.

    :param source_tp_size: Number of attention TP writers in the source cohort.
    :returns: Maximum descriptors one writer may post in a transfer handle.
    :raises ValueError: If the source TP size cannot fit the cohort budget.
    """

    if source_tp_size <= 0:
        raise ValueError(
            f"Source attention TP size must be positive, got {source_tp_size}"
        )
    cohort_descriptor_limit = NIXL_RMA_MAX_COHORT_DESCRIPTORS // source_tp_size
    if cohort_descriptor_limit <= 0:
        raise ValueError(
            "Source attention TP size exceeds the NIXL descriptor cohort budget"
        )
    return min(NIXL_RMA_MAX_DESCRIPTORS, cohort_descriptor_limit)


def _direct_kv_source_cohort_descriptor_limit(source_tp_size: int) -> int:
    """Resolve the per-writer descriptor bound for direct asymmetric KV.

    :param source_tp_size: Number of attention TP writers in the source cohort.
    :returns: Maximum descriptors one writer may post in a direct KV handle.
    :raises ValueError: If the source TP size cannot fit the direct KV budget.
    """

    if source_tp_size <= 0:
        raise ValueError(
            f"Source attention TP size must be positive, got {source_tp_size}"
        )
    cohort_descriptor_limit = NIXL_DIRECT_KV_MAX_COHORT_DESCRIPTORS // source_tp_size
    if cohort_descriptor_limit <= 0:
        raise ValueError(
            "Source attention TP size exceeds the direct KV descriptor cohort budget"
        )
    return min(NIXL_RMA_MAX_DESCRIPTORS, cohort_descriptor_limit)


def _normalize_kv_mem_kinds(kinds: Optional[List[str]], expected_len: int) -> List[str]:
    if kinds is None:
        return ["VRAM"] * expected_len
    kinds = [str(kind) for kind in kinds]
    if len(kinds) != expected_len:
        raise ValueError(
            f"kv_data_mem_kinds length mismatch: got {len(kinds)}, "
            f"expected {expected_len}"
        )
    invalid = sorted(set(kinds) - KV_MEM_KINDS)
    if invalid:
        raise ValueError(f"Unsupported NIXL KV memory kind(s): {invalid}")
    return kinds


def _pack_kv_mem_kinds(kinds: List[str]) -> bytes:
    return ",".join(kinds).encode("ascii")


def _unpack_kv_mem_kinds(buf: bytes, expected_len: int) -> List[str]:
    if not buf:
        return ["VRAM"] * expected_len
    return _normalize_kv_mem_kinds(buf.decode("ascii").split(","), expected_len)


def _nixl_device_id(mem_kind: str, gpu_id: int) -> int:
    return gpu_id if mem_kind == "VRAM" else 0


def _homogeneous_kv_mem_kind(kinds: List[str], context: str) -> str:
    unique = set(kinds)
    if len(unique) != 1:
        raise NotImplementedError(
            f"NIXL {context} mixed KV memory kinds are not implemented safely yet: "
            f"{sorted(unique)}"
        )
    return next(iter(unique))


@dataclasses.dataclass(frozen=True)
class _KVXferMemSegment:
    start: int
    end: int
    src_mem_kind: str
    dst_mem_kind: str


def _kv_xfer_mem_segments(
    src_kinds: List[str], dst_kinds: List[str]
) -> List[_KVXferMemSegment]:
    if len(src_kinds) != len(dst_kinds):
        raise ValueError(
            f"KV source/destination memory kind length mismatch: "
            f"src={len(src_kinds)}, dst={len(dst_kinds)}"
        )
    if not src_kinds:
        return []

    segments = []
    start = 0
    cur = (src_kinds[0], dst_kinds[0])
    for i, pair in enumerate(zip(src_kinds, dst_kinds)):
        if pair == cur:
            continue
        segments.append(_KVXferMemSegment(start, i, cur[0], cur[1]))
        start = i
        cur = pair
    segments.append(_KVXferMemSegment(start, len(src_kinds), cur[0], cur[1]))
    return segments


@dataclasses.dataclass
class _KVXferPreparedSegment:
    start: int
    end: int
    src_handle: Any
    dst_handle: Any
    dst_num_slots: int


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[List[int]]
    decode_prefix_len: Optional[int] = None  # for decode radix cache
    process_generation: str = ""
    packed_plan: PackedAuxiliaryPlan | None = None
    # The staging slot keeps its historical positional index. Terminal source
    # authority is appended after it so older direct constructors stay exact.
    staging: Optional[StagingTransferInfo] = None
    terminal_source_plan_payload: bytes | None = None

    def is_dummy(self):
        # A transfer is "dummy" only for CP non-authoritative ranks.
        # When dst_kv_indices is empty due to a decode-side radix cache
        # full hit (decode_prefix_len > 0), the transfer is NOT dummy --
        # aux/state data still needs to be sent.
        if self.dst_kv_indices.size == 0 and self.decode_prefix_len:
            return False
        return self.dst_kv_indices.size == 0

    def decode_terminal_source_plan(self) -> PackedTerminalSourcePlan | None:
        """Decode the exact terminal authority carried by request metadata.

        :returns: Validated source plan, otherwise ``None`` for legacy packed
            metadata.
        """

        payload = self.terminal_source_plan_payload
        if payload is None:
            return None
        return decode_packed_terminal_source_plan(payload)

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_indices = (
            unpack_int_lists(msg[7], "i") if len(msg) > 7 and msg[7] != b"" else []
        )

        packed_plan = (
            _decode_packed_auxiliary_plan(msg[10])
            if len(msg) > 10 and msg[10] != b""
            else None
        )
        terminal_source_plan_payload = (
            bytes(msg[11]) if len(msg) > 11 and msg[11] != b"" else None
        )
        if terminal_source_plan_payload is not None:
            if packed_plan is None:
                raise ValueError(
                    "terminal source authority requires packed request metadata"
                )
            terminal_source_plan = decode_packed_terminal_source_plan(
                terminal_source_plan_payload
            )
            require_source_plan_request_key(terminal_source_plan, packed_plan.key)

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=validate_nixl_agent_name(msg[3].decode("ascii")),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),  # hacky just add it into the message that will be sent
            process_generation=(
                msg[9].decode("ascii") if len(msg) > 9 and msg[9] != b"" else ""
            ),
            packed_plan=packed_plan,
            terminal_source_plan_payload=terminal_source_plan_payload,
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_kv_mem_kinds: list[str]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_kv_item_lens: list[int]
    dst_kv_layer_ids: list[int] = dataclasses.field(default_factory=list)
    dst_num_slots: Optional[int] = None
    dst_state_item_lens: List[List[int]] = dataclasses.field(default_factory=list)
    dst_state_dim_per_tensor: List[List[int]] = dataclasses.field(default_factory=list)
    dst_state_layer_ids: List[List[int]] = dataclasses.field(default_factory=list)
    dst_homogeneous_mem_kind: Optional[str] = None
    kv_xfer_segments: Optional[List[_KVXferPreparedSegment]] = None
    process_generation: str = ""
    registration_digest: str = ""
    packed_transfer_protocol: str | None = None
    prepared_grant_protocol: str | None = None
    packed_advertisement: PackedRegistrationAdvertisement | None = None
    remote_handle: nixl_remote_agent_handle | None = None
    # Keep last: optional, parsed from a variable-length tail of the ZMQ
    # frame in from_zmq() below, so positional construction stays stable.
    staging: Optional[StagingRegisterInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        (
            packed_transfer_protocol,
            prepared_grant_protocol,
            packed_advertisement,
        ) = _parse_packed_registration(msg)
        dst_kv_ptrs = list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5]))
        dst_kv_mem_kinds = (
            _unpack_kv_mem_kinds(msg[17], len(dst_kv_ptrs))
            if len(msg) > 17
            else ["VRAM"] * len(dst_kv_ptrs)
        )
        dst_kv_item_len = int(msg[11].decode("ascii"))
        dst_kv_item_lens = (
            list(struct.unpack(f"{len(msg[18]) // 8}Q", msg[18]))
            if len(msg) > 18 and msg[18] != b""
            else [dst_kv_item_len] * len(dst_kv_ptrs)
        )
        if len(dst_kv_item_lens) != len(dst_kv_ptrs):
            raise ValueError(
                "dst_kv_item_lens length mismatch: "
                f"got {len(dst_kv_item_lens)}, expected {len(dst_kv_ptrs)}"
            )
        dst_state_data_ptrs = (
            unpack_int_lists(msg[7], "Q") if len(msg) > 7 and msg[7] != b"" else []
        )
        dst_state_item_lens = (
            unpack_int_lists(msg[12], "I") if len(msg) > 12 and len(msg[12]) > 0 else []
        )
        dst_state_dim_per_tensor = (
            unpack_int_lists(msg[13], "I") if len(msg) > 13 and len(msg[13]) > 0 else []
        )
        dst_num_slots = (
            int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else None
        )
        dst_state_layer_ids = (
            unpack_int_lists(msg[19], "I") if len(msg) > 19 and len(msg[19]) > 0 else []
        )
        dst_kv_layer_ids = (
            list(struct.unpack(f"{len(msg[20]) // 4}I", msg[20]))
            if len(msg) > 20 and msg[20] != b""
            else []
        )
        agent_name = validate_nixl_agent_name(msg[3].decode("ascii"))
        agent_metadata = validate_nixl_agent_metadata(msg[4])

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=agent_name,
            agent_metadata=agent_metadata,
            dst_kv_ptrs=dst_kv_ptrs,
            dst_kv_mem_kinds=dst_kv_mem_kinds,
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=dst_kv_item_len,
            dst_kv_item_lens=dst_kv_item_lens,
            dst_kv_layer_ids=dst_kv_layer_ids,
            dst_num_slots=dst_num_slots,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            dst_state_layer_ids=dst_state_layer_ids,
            staging=StagingRegisterInfo.from_zmq_fields(msg, 14),
            process_generation=(
                msg[21].decode("ascii") if len(msg) > 21 and msg[21] != b"" else ""
            ),
            registration_digest=_multipart_digest(msg),
            packed_transfer_protocol=packed_transfer_protocol,
            prepared_grant_protocol=prepared_grant_protocol,
            packed_advertisement=packed_advertisement,
        )


def _multipart_digest(parts: List[bytes]) -> str:
    """Digest one multipart registration without ambiguous concatenation.

    :param parts: Ordered multipart frames.
    :returns: SHA-256 digest of the length-delimited frames.
    """

    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _decode_packed_auxiliary_plan(payload: bytes) -> PackedAuxiliaryPlan:
    """Decode one closed decoder-authored auxiliary plan.

    :param payload: Exact packed-wire payload frame.
    :returns: Validated auxiliary plan.
    :raises ValueError: If the frame contains another packed message type.
    """

    message = decode_packed_message(payload)
    if type(message) is not PackedAuxiliaryPlan:
        raise ValueError("packed transfer metadata does not contain an auxiliary plan")
    return message


def _parse_positive_ascii_integer(frame: bytes, field_name: str) -> int:
    """Parse one positive decimal registration field.

    :param frame: Exact ASCII frame.
    :param field_name: Stable diagnostic field name.
    :returns: Positive integer value.
    :raises ValueError: If the frame is malformed or non-positive.
    """

    try:
        value = int(frame.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid packed registration {field_name}") from error
    if value <= 0:
        raise ValueError(f"packed registration {field_name} must be positive")
    return value


def _parse_packed_registration(
    frames: List[bytes],
) -> tuple[
    str | None,
    str | None,
    PackedRegistrationAdvertisement | None,
]:
    """Parse the optional persistent packed decoder advertisement.

    :param frames: Registration frames after the NIXL guard.
    :returns: Transfer protocol, grant protocol, and advertisement when present.
    :raises ValueError: If an advertised packed registration is incomplete.
    """

    if len(frames) <= 22:
        return None, None, None
    if len(frames) != _PACKED_REGISTRATION_FRAME_COUNT:
        raise ValueError(
            "packed decoder registration has an invalid frame count: " f"{len(frames)}"
        )
    try:
        transfer_protocol = frames[22].decode("ascii")
        grant_protocol = frames[23].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("packed decoder protocols must be ASCII") from error
    if transfer_protocol != PACKED_KV_TRANSFER_PROTOCOL:
        raise ValueError("unsupported packed decoder transfer protocol")
    if grant_protocol != PACKED_PREPARED_GRANT_PROTOCOL:
        raise ValueError("unsupported packed decoder prepared-grant protocol")
    arena_generation = bytes(frames[26])
    visibility_policy_digest = bytes(frames[27])
    runtime_cohort_digest = bytes(frames[28])
    if len(arena_generation) != 16:
        raise ValueError("packed decoder arena generation must contain 16 bytes")
    if len(visibility_policy_digest) != 32:
        raise ValueError("packed decoder visibility digest must contain 32 bytes")
    if len(runtime_cohort_digest) != 32:
        raise ValueError("packed decoder runtime digest must contain 32 bytes")
    advertisement = PackedRegistrationAdvertisement(
        base_address=_parse_positive_ascii_integer(frames[24], "base address"),
        total_size=_parse_positive_ascii_integer(frames[25], "total size"),
        arena_generation=arena_generation,
        visibility_policy_digest=visibility_policy_digest,
        runtime_cohort_digest=runtime_cohort_digest,
        page_size=_parse_positive_ascii_integer(frames[29], "page size"),
    )
    return transfer_protocol, grant_protocol, advertisement


@dataclasses.dataclass(frozen=True)
class _NixlPrefillPeer:
    """One generation-bound native prefill writer loaded by a decoder."""

    bootstrap_addr: str
    attn_dp_rank: int
    attn_cp_rank: int
    attn_tp_rank: int
    pp_rank: int
    transfer_source_rank: int
    agent_name: str
    metadata_sha256: str
    process_generation: str
    control_endpoint: NetworkAddress
    handle: nixl_remote_agent_handle


@dataclasses.dataclass(frozen=True, slots=True)
class _NixlTerminalPrefillTopology:
    """Request-local projection of the frozen terminal source topology."""

    source_tp_size: int
    target_tp_rank: int
    target_tp_ranks: tuple[int, ...]
    required_dst_info_num: int
    required_prefill_response_num: int


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalPrefillRequestAuthority(TerminalPrefillRequestAuthority):
    """Immutable generation-bound source authority retained through attach.

    :ivar bootstrap_addr: Exact enrolled source bootstrap address.
    :ivar startup_binding: Exact local terminal startup generation.
    :ivar prefill_dp_rank: Sole terminal source data-parallel rank.
    :ivar topology: Per-decoder-rank source projection.
    :ivar peers: Canonically ordered selected source writers.
    """

    bootstrap_addr: str
    startup_binding: TerminalStartupRankBinding
    prefill_dp_rank: int
    topology: _NixlTerminalPrefillTopology
    peers: tuple[_NixlPrefillPeer, ...]

    def __post_init__(self) -> None:
        """Validate one closed authority value."""

        if type(self.bootstrap_addr) is not str or len(self.bootstrap_addr) == 0:
            raise ValueError("bootstrap_addr must be nonempty")
        if type(self.startup_binding) is not TerminalStartupRankBinding:
            raise TypeError("startup_binding must be TerminalStartupRankBinding")
        if self.prefill_dp_rank != 0:
            raise ValueError("terminal prefill authority requires DP rank zero")
        if type(self.topology) is not _NixlTerminalPrefillTopology:
            raise TypeError("topology must be _NixlTerminalPrefillTopology")
        if type(self.peers) is not tuple or len(self.peers) == 0:
            raise ValueError("terminal prefill authority requires source peers")
        if any(type(peer) is not _NixlPrefillPeer for peer in self.peers):
            raise TypeError("terminal prefill authority contains an invalid peer")
        peer_ranks = tuple(peer.attn_tp_rank for peer in self.peers)
        if peer_ranks != self.topology.target_tp_ranks:
            raise ValueError("terminal prefill peers differ from topology projection")


@dataclasses.dataclass(slots=True)
class _TerminalStartupPeerEnrollment:
    """Manager-owned cross-role native peer roster for one startup epoch.

    :ivar binding: Exact immutable startup rank binding.
    :ivar expected_remote_ranks: Canonical cross-role matrix population.
    :ivar prefill_peers: Decode-owned source peers keyed by static rank.
    :ivar decoder_peers: Source-owned decoder peers keyed by static rank.
    :ivar frozen: Whether the complete expected roster has been retained.
    :ivar lock: Serializes roster validation, publication, and freeze.
    :ivar frozen_event: Event-driven completion observed by startup composition.
    """

    binding: TerminalStartupRankBinding
    expected_remote_ranks: tuple[TerminalStartupRankAdvertisement, ...]
    prefill_peers: dict[tuple[str, int], _NixlPrefillPeer] = dataclasses.field(
        default_factory=dict
    )
    decoder_peers: dict[tuple[str, int], KVArgsRegisterInfo] = dataclasses.field(
        default_factory=dict
    )
    frozen: bool = False
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    frozen_event: threading.Event = dataclasses.field(default_factory=threading.Event)

    @property
    def expected_keys(self) -> frozenset[tuple[str, int]]:
        """Return the exact cross-role static rank keys.

        :returns: Immutable expected service-rank population.
        """

        return frozenset(rank.key for rank in self.expected_remote_ranks)

    @property
    def enrolled_keys(self) -> frozenset[tuple[str, int]]:
        """Return the retained cross-role native rank keys.

        :returns: Immutable retained service-rank population.
        """

        local_role = self.binding.advertisement.role
        if local_role is TerminalOwnerRole.SOURCE:
            return frozenset(self.decoder_peers)
        return frozenset(self.prefill_peers)

    def freeze_if_complete(self) -> None:
        """Freeze the roster exactly when every expected peer is retained.

        :raises RuntimeError: If an impossible extra peer entered the roster.
        """

        enrolled_keys = self.enrolled_keys
        expected_keys = self.expected_keys
        if not enrolled_keys.issubset(expected_keys):
            raise RuntimeError("terminal startup roster contains an unexpected peer")
        if enrolled_keys != expected_keys:
            return
        self.frozen = True
        self.frozen_event.set()


@dataclasses.dataclass(frozen=True, slots=True)
class _TerminalDecoderEnrollment:
    """One source-retained decoder registration awaiting acknowledgement.

    :ivar rank: Exact decoder row from the sealed startup matrix.
    :ivar registration: Parsed process-lifetime decoder registration.
    :ivar frames: Complete guarded multipart message retained by the source.
    """

    rank: TerminalStartupRankAdvertisement
    registration: KVArgsRegisterInfo
    frames: tuple[bytes, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalRuntimeInstallation:
    """Scheduler-owned dependencies required before terminal activation.

    :ivar terminal_request_capacity: Exact request-generation capacity governing
        every derived process queue.
    :ivar gateway_endpoint: Canonical source-rank gateway PUSH endpoint.
    :ivar bind_source_serving: Source-only scheduler serving installation.
    :ivar bind_decode_serving: Decode-only scheduler and preallocation-queue
        serving installation.
    :ivar scheduler_process_fatal_handler: Scheduler-thread fatal inventory
        consumer.
    :ivar owner_dead_handler: Thread-safe wake that makes scheduler admission
        fail closed after a publisher, reactor, or control-owner death.
    """

    terminal_request_capacity: int
    gateway_endpoint: str | None
    bind_source_serving: Callable[[PackedTerminalSourceServing], None] | None
    bind_decode_serving: Callable[[PackedTerminalDecodeServing], None] | None
    scheduler_process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None]
    owner_dead_handler: Callable[[], None]

    def __post_init__(self) -> None:
        """Validate the role-neutral installation boundary."""

        if (
            type(self.terminal_request_capacity) is not int
            or self.terminal_request_capacity <= 0
        ):
            raise ValueError("terminal_request_capacity must be a positive integer")
        if self.gateway_endpoint is not None and (
            type(self.gateway_endpoint) is not str or len(self.gateway_endpoint) == 0
        ):
            raise ValueError("gateway_endpoint must be None or a non-empty string")
        callbacks = (self.bind_source_serving, self.bind_decode_serving)
        if any(
            callback is not None and not callable(callback) for callback in callbacks
        ):
            raise TypeError("optional terminal installation callbacks must be callable")
        if not callable(self.scheduler_process_fatal_handler):
            raise TypeError("scheduler_process_fatal_handler must be callable")
        if not callable(self.owner_dead_handler):
            raise TypeError("owner_dead_handler must be callable")


class _NixlTerminalSourceMetrics(PackedTerminalSourceMetricsSink):
    """Project non-gating source metrics into the existing process logger."""

    def emit(self, metric: PackedTerminalSourceMetric) -> None:
        """Emit one source lifecycle point without affecting admission.

        :param metric: Exactly-once source lifecycle timing point.
        """

        if type(metric) is not PackedTerminalSourceMetric:
            raise TypeError("metric must be PackedTerminalSourceMetric")
        logger.debug(
            "terminal source event=%s binding=%s timestamp_ns=%d",
            metric.event_kind.name,
            metric.binding_digest.hex(),
            metric.timestamp_ns,
        )


def expand_page_indices_for_slice(
    page_indices: npt.NDArray[np.int32],
    num_ptr_pairs: int,
    num_slots: int,
    page_size: int,
    num_groups: int = 1,
    head_group_idx: int = 0,
) -> npt.NDArray[np.int32]:
    """Map page slot indices to flat dlist indices for the slice prepped path.

    Dlist layout: num_ptr_pairs blocks of (num_slots * page_size * num_groups),
    with [slot, token, group] interleaving. head_group_idx selects one group (0 for dst).
    """
    token_offsets = np.arange(page_size, dtype=np.int32)
    pair_stride = num_slots * page_size * num_groups
    within_pair = (
        page_indices[:, None] * (page_size * num_groups)
        + token_offsets[None, :] * num_groups
        + head_group_idx
    ).ravel()
    pair_offsets = np.arange(num_ptr_pairs, dtype=np.int64) * pair_stride
    return (pair_offsets[:, None] + within_pair[None, :]).ravel().astype(np.int32)


def repeat_indices_over_layers(
    indices: npt.NDArray[np.int32], num_layers: int, layer_length: int
) -> npt.NDArray[np.int32]:
    """Map per-slot token indices to flat indices in a pre-built descriptor list.

    Each of ``num_layers`` blocks has ``layer_length`` slots; block i is offset by
    ``i * layer_length``. Works uniformly for both MLA (one ptr/layer) and MHA
    (K+V ptrs, 2×N entries).
    """
    offsets = np.arange(num_layers, dtype=np.int32) * layer_length
    return (offsets[:, None] + indices[None, :]).ravel().astype(np.int32)


@dataclasses.dataclass
class _StagingPartReceipt:
    """Receiver-owned state for one independently posted staging chunk."""

    is_last_chunk: bool
    chunk_idx: int
    page_start: int
    num_pages: int
    agent_name: str
    num_parts: int
    received_parts: set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per source writer: {source_rank: set of chunk_ids}
    received_kvs_per_source: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per source writer once its last chunk arrives.
    expected_kvs_per_source: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of source writers expected to send data.
    num_source_writers_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # State transfers are independently completed by source writer and component.
    received_state_components: set[tuple[int, int]] = dataclasses.field(
        default_factory=set
    )
    # Component positions in KVArgs.state_types which have a non-empty payload.
    expected_state_indices: set[int] = dataclasses.field(default_factory=set)
    # KV part notifications for mixed-memory transfers. Keyed by
    # (source_rank, chunk_id); normal homogeneous transfers bypass this.
    received_kv_parts_per_source: Optional[Dict[Tuple[int, int], Set[int]]] = None
    expected_kv_parts_per_source: Optional[Dict[Tuple[int, int], int]] = None
    staging_parts_per_source: dict[tuple[int, int], _StagingPartReceipt] = (
        dataclasses.field(default_factory=dict)
    )
    completed_staging_chunks: set[tuple[int, int]] = dataclasses.field(
        default_factory=set
    )
    expected_source_ranks: Dict[nixl_remote_agent_handle, int] = dataclasses.field(
        default_factory=dict
    )
    expected_source_generations: Dict[nixl_remote_agent_handle, str] = (
        dataclasses.field(default_factory=dict)
    )
    canonical_aux_source: nixl_remote_agent_handle | None = None

    @property
    def expected_state_components(self) -> set[tuple[int, int]]:
        """Return the state transfers required from the discovered KV writers.

        :returns: Source-writer and state-component pairs required for completion.
        """

        return {
            (source_rank, state_index)
            for source_rank in self.expected_kvs_per_source
            for state_index in self.expected_state_indices
        }

    def is_done(self) -> bool:
        if self.num_source_writers_expected is None or not self.received_aux:
            return False
        if len(self.expected_kvs_per_source) < self.num_source_writers_expected:
            return False
        for source_rank, expected in self.expected_kvs_per_source.items():
            if len(self.received_kvs_per_source[source_rank]) != expected:
                return False
        if not self.expected_state_components.issubset(self.received_state_components):
            return False
        return True


class NixlKVManager(CommonKVManager):
    _terminal_startup_binding: TerminalStartupRankBinding | None = None
    _terminal_startup_peer_enrollment: _TerminalStartupPeerEnrollment | None = None
    _terminal_source_publication_control: TerminalSourcePublicationControl | None = None
    _terminal_dflash_boundary_pool: DFlashBoundaryDeviceRowPool | None = None
    _terminal_decode_control_routes: TerminalDecodeControlRouteTable | None = None

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self._terminal_startup_binding = None
        self._terminal_startup_peer_enrollment = None
        self._terminal_source_publication_control = None
        self._terminal_dflash_boundary_pool = None
        self._terminal_decode_control_routes = None
        self._terminal_runtime_activated = threading.Event()
        self._terminal_activation_lock = threading.Lock()
        self._terminal_activation_started = False
        self._terminal_bootstrap_thread: threading.Thread | None = None
        self._runtime_workers_started = False
        self._terminal_runtime_installation: NixlTerminalRuntimeInstallation | None = (
            None
        )
        self._terminal_runtime_enrollment: TerminalRankRuntimeEnrollment | None = None
        self._terminal_dflash_source_pool: DFlashBoundaryDeviceRowPool | None = None
        self._terminal_dflash_source_owner: (
            DFlashBoundarySourceTransportOwner | None
        ) = None
        self._terminal_grouped_nixl_owner: GroupedNixlTerminalOwner | None = None
        self._terminal_source_serving: PackedTerminalSourceServing | None = None
        self._terminal_decode_serving: PackedTerminalDecodeServing | None = None
        self._terminal_process_reactor: PackedTerminalProcessReactor | None = None
        self._terminal_output_publisher: PackedTerminalOutputPublisher | None = None
        self._terminal_source_receipt_importers: dict[
            TerminalProcessIdentity, TerminalWireReceiptImportNamespace
        ] = {}
        self._terminal_unpublished_source_quarantine: dict[
            bytes, PackedTerminalSourceSubmission
        ] = {}
        self._terminal_unpublished_source_quarantine_lock = threading.Lock()
        self._terminal_control_thread: threading.Thread | None = None
        self._terminal_control_read_fd: int | None = None
        self._terminal_control_write_fd: int | None = None
        self._terminal_control_stop_requested = False
        self._terminal_control_ready = threading.Event()
        self._terminal_control_lock = threading.Lock()
        self._terminal_runtime_close_started = False
        self._terminal_runtime_closed = False
        self._terminal_runtime_close_lock = threading.Lock()
        self._terminal_process_fatal_reason: str | None = None
        self._terminal_process_fatal_traceback: str | None = None
        self._terminal_process_fatal_lock = threading.Lock()
        super().__init__(
            args,
            disaggregation_mode,
            server_args,
            is_mla_backend,
            defer_prefill_bootstrap_registration=True,
        )
        self.transfer_source_rank = (
            self.kv_args.pp_rank * self.server_args.tp_size + self.kv_args.engine_rank
        )
        self.kv_args.kv_data_mem_kinds = _normalize_kv_mem_kinds(
            getattr(self.kv_args, "kv_data_mem_kinds", None),
            len(self.kv_args.kv_data_ptrs),
        )
        self.src_mem_kind = (
            _homogeneous_kv_mem_kind(self.kv_args.kv_data_mem_kinds, "source")
            if disaggregation_mode == DisaggregationMode.PREFILL
            and self.kv_args.kv_data_mem_kinds
            else None
        )
        try:
            from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0
        backend_params = json.loads(
            envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()
        )
        if not isinstance(backend_params, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in backend_params.items()
        ):
            raise ValueError(
                "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
                "with string keys and string values"
            )
        # self.transfer_worker and self._start_bootstrap_thread runs concurrently
        # so we cannot use sync_mode=None which is thread-unsafe.
        agent_config = nixl_agent_config(
            backends=[],
            num_threads=num_threads,
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
        )
        self.process_generation = str(uuid.uuid4())
        self.agent = nixl_agent(self.process_generation, agent_config)
        if num_threads > 0:
            # TODO: Remove this once NIXL passes thread parameters from
            # nixl_agent_config to explicitly-created backends.
            if backend == "UCX" or backend == "OBJ":
                backend_params.setdefault("num_threads", str(num_threads))
            elif backend == "GDS_MT":
                backend_params.setdefault("thread_count", str(num_threads))
            elif backend == "UCCL":
                backend_params.setdefault("num_cpus", str(num_threads))
        self.agent.create_backend(backend, backend_params)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        self.register_buffer_to_engine()
        if self._requires_terminal_dflash_boundary_pool():
            row_capacity = self.kv_args.terminal_request_capacity
            if type(row_capacity) is not int or row_capacity <= 0:
                raise ValueError(
                    "terminal DFlash boundary row capacity must be positive"
                )
            self._terminal_dflash_boundary_pool = DFlashBoundaryDeviceRowPool(
                self.agent,
                row_capacity=row_capacity,
                device=torch.device("cuda", self.kv_args.gpu_id),
            )
        self._prefill_peers: dict[tuple[str, int, int, int, int], _NixlPrefillPeer] = {}
        self._prefill_peer_keys_by_addr: Dict[
            str, set[tuple[str, int, int, int, int]]
        ] = defaultdict(set)
        self._prefill_peers_by_agent_name: dict[str, _NixlPrefillPeer] = {}
        self._prefill_peers_by_handle: dict[
            nixl_remote_agent_handle, _NixlPrefillPeer
        ] = {}
        self._prefill_peer_lock = threading.RLock()
        self._quarantined_remote_handles: set[nixl_remote_agent_handle] = set()
        self._decode_staging_registration: object | None = None
        self._packed_control_send_lock = threading.Lock()
        self._packed_decode_controller: PackedNixlDecodeController | None = None
        self._packed_prefill_runtime: PackedPrefillRuntime | None = None

        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.kv_buffer_tensors = None
        self.prep_handles: Dict[str, Any] = {}
        self.prep_handle_slice_src: Optional[Tuple[Any, int, int, int]] = (
            None  # (handle, num_groups, num_ptr_pairs, num_slots)
        )
        self.prep_handles_slice_dst: Dict[str, Tuple[Any, int, int]] = {}
        # peer_name -> (handle, num_slots, head_group_idx)
        self.prep_handles_segment_src: Dict[Tuple[int, int, str], Any] = {}
        self._num_slots_src: int = 0

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if self.kv_args.kv_item_lens:
                self._num_slots_src = (
                    self.kv_args.kv_data_lens[0] // self.kv_args.kv_item_lens[0]
                )
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            self._direct_kv_transfer_lock = threading.Lock()
            self.exceptions: Dict[int, Exception] = {}
            # Mirror mooncake: one staging buffer per worker queue, all
            # built before workers spawn so each worker owns a private
            # buffer (no cross-worker contention on the staging ring).
            if self.enable_staging:
                self._init_staging_prefill_ctx()
                self._init_staging_buffers(len(self.transfer_queues))
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            if self.enable_staging:
                self._init_staging_decode_ctx()
                self._staging_handler = None
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

        if (
            self.enable_staging
            and self.disaggregation_mode == DisaggregationMode.DECODE
        ):
            staging_registration = self._decode_staging_registration
            staging_allocator = self._staging_ctx.allocator
            if staging_registration is None or staging_allocator is None:
                raise RuntimeError("packed decode staging ownership is unavailable")
            if (
                self.attn_tp_size in (1, 2)
                and 0 <= self.attn_tp_rank < self.attn_tp_size
                and self.attn_cp_size == 1
                and self.pp_size == 1
            ):
                self._packed_decode_controller = PackedNixlDecodeController(
                    self,
                    staging_allocator.buffer.buffer,
                    staging_registration,
                    self._terminal_dflash_boundary_pool,
                )
        elif (
            self.enable_staging
            and self.disaggregation_mode == DisaggregationMode.PREFILL
            and self.attn_tp_size in SUPPORTED_PACKED_SOURCE_TP_SIZES
            and self.attn_cp_size == 1
            and self.pp_size == 1
        ):
            runtime_artifacts = load_exact_nixl_runtime_artifacts()
            self._packed_prefill_runtime = PackedPrefillRuntime(
                self,
                runtime_artifacts,
                build_same_host_visibility_policy(runtime_artifacts),
            )

        self._initialize_terminal_dflash_source_pool()

        # NIXL metadata is a snapshot of every registered memory section. No
        # transport identity may leave this manager until all long-lived
        # payload and staging regions are represented in that snapshot.
        self.agent_metadata = bytes(self.agent.get_agent_metadata())
        terminal_binding = self._join_terminal_startup_cohort()
        if terminal_binding is not None:
            self.install_terminal_startup_binding(terminal_binding)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if terminal_binding is not None:
                self.register_to_bootstrap(max_attempts=1)
            else:
                self.register_to_bootstrap()
                self._start_prefill_runtime_workers()
        elif terminal_binding is None:
            self._start_decode_runtime_workers()

    def _requires_terminal_dflash_boundary_pool(self) -> bool:
        """Return whether this manager must advertise registered boundary rows.

        :returns: Whether the local role is a terminal DFlash decoder.
        """

        return (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.server_args.pd_terminal_deployment_cohort is not None
            and self.server_args.speculative_algorithm == "DFLASH"
        )

    def terminal_dflash_boundary_pool(
        self,
    ) -> DFlashBoundaryDeviceRowPool | None:
        """Return the process-lifetime destination row owner.

        :returns: Registered DFlash rows, or ``None`` outside terminal DFlash.
        """

        return self._terminal_dflash_boundary_pool

    def _initialize_terminal_dflash_source_pool(self) -> None:
        """Register canonical source boundary rows before metadata freezes."""

        if self.server_args.pd_terminal_deployment_cohort is None:
            return
        if self.disaggregation_mode is not DisaggregationMode.PREFILL:
            return
        if not self._is_canonical_aux_writer():
            return
        if self.server_args.speculative_algorithm != "DFLASH":
            raise ValueError(
                "terminal DFlash boundary transport requires DFLASH speculation"
            )
        capacity = self.kv_args.terminal_request_capacity
        if type(capacity) is not int or capacity <= 0:
            raise ValueError(
                "terminal source requires a positive scheduler request capacity"
            )
        if self._terminal_dflash_source_pool is not None:
            raise RuntimeError("terminal DFlash source pool is already initialized")
        self._terminal_dflash_source_pool = DFlashBoundaryDeviceRowPool(
            self.agent,
            row_capacity=capacity,
            device=torch.device("cuda", self.kv_args.gpu_id),
        )

    def _start_prefill_runtime_workers(self) -> None:
        """Start source transfer workers after peer authority is immutable."""

        if self._runtime_workers_started:
            raise RuntimeError("NIXL runtime workers are already started")
        if (
            self.terminal_startup_binding is not None
            and not self._terminal_runtime_activated.is_set()
        ):
            raise RuntimeError("terminal startup is not committed")
        for worker_index, queue in enumerate(self.transfer_queues):
            staging_buffer = (
                self._staging_ctx.buffers[worker_index]
                if self.enable_staging and self._staging_ctx.buffers
                else None
            )
            threading.Thread(
                target=self.transfer_worker,
                args=(queue, staging_buffer),
                daemon=True,
            ).start()
        if self._terminal_bootstrap_thread is None:
            self._start_bootstrap_thread()
        self._runtime_workers_started = True

    def _start_decode_runtime_workers(self) -> None:
        """Start decode staging and health workers after startup commit."""

        if self._runtime_workers_started:
            raise RuntimeError("NIXL runtime workers are already started")
        if (
            self.terminal_startup_binding is not None
            and not self._terminal_runtime_activated.is_set()
        ):
            raise RuntimeError("terminal startup is not committed")
        if self.enable_staging:
            self._start_decode_staging_thread()
        self._start_heartbeat_checker_thread()
        self._runtime_workers_started = True

    @property
    def terminal_startup_binding(self) -> TerminalStartupRankBinding | None:
        """Return this rank's immutable startup epoch and producer authority.

        :returns: Complete rank binding, or ``None`` outside terminal deployments.
        """

        return self._terminal_startup_binding

    @property
    def terminal_source_publication_control(
        self,
    ) -> TerminalSourcePublicationControl | None:
        """Return the direct source-rank publication route owner.

        :returns: Startup-enrolled source control, or ``None`` outside a
            terminal source deployment.
        """

        return self._terminal_source_publication_control

    @property
    def terminal_peer_enrollment_frozen(self) -> bool:
        """Return whether every matrix-authorized remote peer is retained.

        :returns: ``True`` only after the exact cross-role roster freezes.
        """

        enrollment = self._terminal_startup_peer_enrollment
        return enrollment is not None and enrollment.frozen

    def resolve_terminal_prefill_request_authority(
        self,
        *,
        bootstrap_addr: str,
        prefill_process_url: str,
        prefill_process_instance_id: uuid.UUID,
        prefill_dp_rank: int | None,
        source_tp_size: int,
    ) -> NixlTerminalPrefillRequestAuthority:
        """Project one request's source writers from the frozen startup roster.

        This resolver performs no request mutation, receiver construction, cache
        lookup, or network operation. The startup enrollment is the sole source
        of generation and topology authority.

        :param bootstrap_addr: Exact source bootstrap service address.
        :param prefill_process_url: Reservation-authenticated source service URL.
        :param prefill_process_instance_id: Reservation-authenticated source
            launch instance.
        :param prefill_dp_rank: Explicit source DP rank, otherwise ``None``.
        :param source_tp_size: Reservation-authenticated source TP width.
        :returns: Immutable authority for receiver attachment.
        :raises TerminalPrefillAuthorityUnavailable: If startup enrollment has
            not frozen yet.
        :raises TerminalPrefillAuthorityMismatch: If the reservation differs
            from the frozen deployment generation or topology.
        """

        if type(bootstrap_addr) is not str or len(bootstrap_addr) == 0:
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill bootstrap address is invalid"
            )
        if type(prefill_process_url) is not str or len(prefill_process_url) == 0:
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill process URL is invalid"
            )
        if (
            type(prefill_process_instance_id) is not uuid.UUID
            or prefill_process_instance_id.int == 0
        ):
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill process instance is invalid"
            )
        if prefill_dp_rank not in (None, 0):
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill authority requires DP rank zero"
            )
        if type(source_tp_size) is not int or source_tp_size <= 0:
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill source TP width is invalid"
            )
        binding = self.terminal_startup_binding
        enrollment = self._terminal_startup_peer_enrollment
        if binding is None or enrollment is None or enrollment.binding != binding:
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill startup authority is not configured"
            )
        if binding.advertisement.role is not TerminalOwnerRole.DECODE:
            raise TerminalPrefillAuthorityMismatch(
                "terminal prefill authority requires a decoder manager"
            )

        with enrollment.lock:
            if not enrollment.frozen:
                raise TerminalPrefillAuthorityUnavailable(
                    "terminal prefill source enrollment is not frozen"
                )
            source_ranks = tuple(
                rank
                for rank in enrollment.expected_remote_ranks
                if rank.role is TerminalOwnerRole.SOURCE
            )
            if len(source_ranks) == 0:
                raise TerminalPrefillAuthorityMismatch(
                    "terminal startup matrix has no source ranks"
                )
            source_service_ids = {rank.service_id for rank in source_ranks}
            source_tp_sizes = {rank.tensor_parallel_size for rank in source_ranks}
            if len(source_service_ids) != 1 or source_tp_sizes != {source_tp_size}:
                raise TerminalPrefillAuthorityMismatch(
                    "reservation source TP width differs from startup authority"
                )
            if len(source_ranks) != source_tp_size:
                raise TerminalPrefillAuthorityMismatch(
                    "terminal source rank population differs from its TP width"
                )
            if any(
                rank.service_origin != prefill_process_url
                or rank.launch_instance_id != prefill_process_instance_id.bytes
                for rank in source_ranks
            ):
                raise TerminalPrefillAuthorityMismatch(
                    "reservation source process differs from startup authority"
                )
            peers = tuple(
                enrollment.prefill_peers.get(rank.key) for rank in source_ranks
            )

        if any(peer is None for peer in peers):
            raise TerminalPrefillAuthorityMismatch(
                "frozen terminal source enrollment is incomplete"
            )
        typed_peers = tuple(peer for peer in peers if peer is not None)
        for rank, peer in zip(source_ranks, typed_peers, strict=True):
            expected_generation = str(uuid.UUID(bytes=rank.process_generation))
            if (
                peer.bootstrap_addr != bootstrap_addr
                or peer.attn_dp_rank != 0
                or peer.attn_cp_rank != 0
                or peer.attn_tp_rank != rank.tensor_parallel_rank
                or peer.pp_rank != 0
                or peer.transfer_source_rank != rank.tensor_parallel_rank
                or peer.agent_name != rank.nixl_agent_name
                or peer.metadata_sha256 != rank.nixl_agent_metadata_sha256.hex()
                or peer.process_generation != expected_generation
                or peer.handle in self._quarantined_remote_handles
            ):
                raise TerminalPrefillAuthorityMismatch(
                    "terminal prefill peer differs from frozen startup authority"
                )

        decode_tp_size = self.attn_tp_size
        if (
            source_tp_size % decode_tp_size != 0
            and decode_tp_size % source_tp_size != 0
        ):
            raise TerminalPrefillAuthorityMismatch(
                "source and decode TP widths are not evenly divisible"
            )
        decode_rank = self.kv_args.engine_rank % decode_tp_size
        if decode_tp_size == source_tp_size:
            target_tp_rank = decode_rank
            target_tp_ranks = (target_tp_rank,)
            required_dst_info_num = 1
            required_prefill_response_num = 1
        elif decode_tp_size > source_tp_size:
            decode_ranks_per_source = decode_tp_size // source_tp_size
            target_tp_rank = decode_rank // decode_ranks_per_source
            target_tp_ranks = (target_tp_rank,)
            required_dst_info_num = decode_ranks_per_source
            required_prefill_response_num = 1
        else:
            source_ranks_per_decode = source_tp_size // decode_tp_size
            first_source_rank = decode_rank * source_ranks_per_decode
            target_tp_ranks = tuple(
                range(first_source_rank, first_source_rank + source_ranks_per_decode)
            )
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            required_prefill_response_num = source_ranks_per_decode

        peer_by_tp_rank = {peer.attn_tp_rank: peer for peer in typed_peers}
        if any(rank not in peer_by_tp_rank for rank in target_tp_ranks):
            raise TerminalPrefillAuthorityMismatch(
                "terminal source roster cannot satisfy the decode TP projection"
            )
        selected_peers = tuple(peer_by_tp_rank[rank] for rank in target_tp_ranks)
        topology = _NixlTerminalPrefillTopology(
            source_tp_size=source_tp_size,
            target_tp_rank=target_tp_rank,
            target_tp_ranks=target_tp_ranks,
            required_dst_info_num=required_dst_info_num,
            required_prefill_response_num=required_prefill_response_num,
        )
        return NixlTerminalPrefillRequestAuthority(
            bootstrap_addr=bootstrap_addr,
            startup_binding=binding,
            prefill_dp_rank=0,
            topology=topology,
            peers=selected_peers,
        )

    def install_terminal_runtime(
        self,
        installation: NixlTerminalRuntimeInstallation,
    ) -> None:
        """Install scheduler and route dependencies before startup activation.

        :param installation: Exact role-specific runtime dependencies.
        :raises RuntimeError: If installation repeats or activation has begun.
        """

        if type(installation) is not NixlTerminalRuntimeInstallation:
            raise TypeError("installation must be NixlTerminalRuntimeInstallation")
        binding = self.terminal_startup_binding
        if binding is None:
            raise RuntimeError("terminal startup binding is not configured")
        if self._terminal_activation_started:
            raise RuntimeError("terminal runtime installation is too late")
        if self._terminal_runtime_installation is not None:
            raise RuntimeError("terminal runtime is already installed")

        local = binding.advertisement
        if local.role is TerminalOwnerRole.SOURCE:
            if installation.bind_source_serving is None:
                raise ValueError(
                    "terminal source installation requires a scheduler binding"
                )
            if installation.bind_decode_serving is not None:
                raise ValueError(
                    "terminal source installation cannot bind decode serving"
                )
            if local.tensor_parallel_rank == 0:
                if installation.gateway_endpoint is None:
                    raise ValueError(
                        "canonical terminal source requires a gateway endpoint"
                    )
            elif installation.gateway_endpoint is not None:
                raise ValueError(
                    "noncanonical terminal source cannot own a gateway endpoint"
                )
        else:
            if installation.gateway_endpoint is not None:
                raise ValueError(
                    "terminal decode installation cannot own a gateway endpoint"
                )
            if installation.bind_decode_serving is None:
                raise ValueError(
                    "terminal decode installation requires a scheduler binding"
                )
            if installation.bind_source_serving is not None:
                raise ValueError(
                    "terminal decode installation cannot bind source serving"
                )
        self._terminal_runtime_installation = installation

    @property
    def terminal_runtime_enrollment(self) -> TerminalRankRuntimeEnrollment:
        """Return the sole active terminal runtime enrollment.

        :returns: Process-lifetime runtime and native producer ownership.
        """

        enrollment = self._terminal_runtime_enrollment
        if enrollment is None:
            raise RuntimeError("terminal runtime enrollment is unavailable")
        return enrollment

    @property
    def terminal_source_serving(self) -> PackedTerminalSourceServing:
        """Return the active source serving composition.

        :returns: Sole source serving owner for this process.
        """

        binding = self.terminal_startup_binding
        if (
            binding is None
            or binding.advertisement.role is not TerminalOwnerRole.SOURCE
        ):
            raise RuntimeError("terminal source serving requires a source manager")
        serving = self._terminal_source_serving
        if serving is None:
            raise RuntimeError("terminal source serving is unavailable")
        return serving

    @property
    def terminal_decode_serving(self) -> PackedTerminalDecodeServing:
        """Return the active decode serving composition.

        :returns: Sole decode serving owner for this process.
        """

        binding = self.terminal_startup_binding
        if (
            binding is None
            or binding.advertisement.role is not TerminalOwnerRole.DECODE
        ):
            raise RuntimeError("terminal decode serving requires a decode manager")
        serving = self._terminal_decode_serving
        if serving is None:
            raise RuntimeError("terminal decode serving is unavailable")
        return serving

    @property
    def terminal_process_reactor(self) -> PackedTerminalProcessReactor:
        """Return the active off-forward process reactor.

        :returns: Sole role-specific terminal reactor.
        """

        reactor = self._terminal_process_reactor
        if reactor is None:
            raise RuntimeError("terminal process reactor is unavailable")
        return reactor

    def bind_terminal_source_submission(
        self,
        submission: PackedTerminalSourceSubmission,
        release_resources: Callable[[PackedTerminalSourceSubmission], None],
        commit_scheduler_retention: Callable[[PackedTerminalSourceSubmission], None],
    ) -> None:
        """Bind every source owner and scheduler retention before PREPARE.

        :param submission: Exact immutable post-model-return handoff.
        :param release_resources: Scheduler-affine one-shot resource release.
        :param commit_scheduler_retention: Scheduler-affine request retention
            committed immediately before PREPARE can become observable.
        """

        if type(submission) is not PackedTerminalSourceSubmission:
            raise TypeError("submission must be PackedTerminalSourceSubmission")
        if not callable(release_resources):
            raise TypeError("release_resources must be callable")
        if not callable(commit_scheduler_retention):
            raise TypeError("commit_scheduler_retention must be callable")
        serving: PackedTerminalSourceServing | None = None
        serving_bind_started = False
        lifecycle_committed = False
        try:
            self.terminal_process_reactor.require_admission_open()
            serving = self.terminal_source_serving
            runtime = self._packed_prefill_runtime
            if runtime is None:
                raise RuntimeError("packed source actor is unavailable")
            transport = submission.transport_submission
            if type(transport) is not PackedPrefillSubmission:
                raise TypeError(
                    "source transport_submission must be PackedPrefillSubmission"
                )
            identity = submission.identity
            if (
                identity.local_binding.owner.tp_rank == 0
                and type(submission.output_projection)
                is not PrefillTerminalGatewayOutputProjection
            ):
                raise TypeError(
                    "canonical source requires a pinned prefill output projection"
                )
            importer = self._terminal_source_receipt_importers.get(
                identity.request_ready_issuer
            )
            if importer is None:
                raise RuntimeError(
                    "source request-ready issuer is absent from sealed startup"
                )
            publication_control = self._terminal_source_publication_control
            if publication_control is None:
                raise RuntimeError("source publication control is unavailable")
            runtime.bind_terminal_owner(transport, identity)
            importer.register_binding(identity.local_binding)
            serving_bind_started = True
            serving.bind_submission(submission, release_resources)
            lifecycle_committed = True
            publication_control.register_binding(identity.local_binding)
            commit_scheduler_retention(submission)
            runtime.publish_terminal_owner_prepare(transport)
            serving.attach_producer_completion(submission)
        except Exception as error:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            cleanup_failures: list[str] = []
            if serving_bind_started and not lifecycle_committed and serving is not None:
                try:
                    lifecycle_committed = (
                        submission.identity.local_binding.digest
                        in serving.inventory().wiring.active_binding_digests
                    )
                except Exception:  # noqa: BLE001
                    cleanup_failures.append(traceback.format_exc())
            if not lifecycle_committed:
                try:
                    self.quarantine_unpublished_terminal_source_submission(submission)
                except Exception:  # noqa: BLE001
                    cleanup_failures.append(traceback.format_exc())
            try:
                self.fail_terminal_source_process(
                    "terminal source bind failed after producer submission",
                    formatted_traceback,
                )
            except Exception:  # noqa: BLE001
                cleanup_failures.append(traceback.format_exc())
            if len(cleanup_failures) > 0:
                error.add_note(
                    "terminal source bind quarantine failed:\n"
                    + "\n".join(cleanup_failures)
                )
            raise

    def quarantine_unpublished_terminal_source_submission(
        self,
        submission: PackedTerminalSourceSubmission,
    ) -> None:
        """Retain CUDA-touched transport and result state fail closed.

        :param submission: Exact transport, device row, and pinned result slot.
        """

        if type(submission) is not PackedTerminalSourceSubmission:
            raise TypeError("submission must be PackedTerminalSourceSubmission")
        transport = submission.transport_submission
        if type(transport) is not PackedPrefillSubmission:
            raise TypeError("source transport_submission has another schema")
        digest = submission.identity.local_binding.digest
        with self._terminal_unpublished_source_quarantine_lock:
            existing = self._terminal_unpublished_source_quarantine.get(digest)
            if existing is not None:
                if existing is submission:
                    return
                raise RuntimeError("source quarantine identity was reused")
            self._terminal_unpublished_source_quarantine[digest] = submission
        auxiliary = transport.auxiliary_source
        if type(auxiliary) is not PackedTerminalDFlashAuxiliarySource:
            return
        owner = self._terminal_dflash_source_owner
        if owner is None:
            raise RuntimeError("terminal DFlash source owner is unavailable")
        owner.quarantine_unpublished_source_row(auxiliary.prefill_source.lease)

    def install_terminal_startup_binding(
        self,
        binding: TerminalStartupRankBinding,
    ) -> None:
        """Install one sealed matrix as this manager's native peer authority.

        The startup join owns construction of ``binding``. This method proves
        its local row against the already initialized manager before any full
        metadata route can create a remote native handle.

        :param binding: Complete generation-authenticated startup rank binding.
        :raises RuntimeError: If a binding is replaced or local identity drifts.
        """

        if type(binding) is not TerminalStartupRankBinding:
            raise TypeError("binding must be TerminalStartupRankBinding")
        if self._terminal_startup_peer_enrollment is not None:
            raise RuntimeError("terminal startup binding is already installed")
        current_binding = self._terminal_startup_binding
        if current_binding is not None and current_binding != binding:
            raise RuntimeError("terminal startup binding cannot be replaced")

        local_rank = binding.advertisement
        expected_role = (
            TerminalOwnerRole.SOURCE
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else TerminalOwnerRole.DECODE
        )
        if local_rank.role is not expected_role:
            raise RuntimeError("terminal startup binding has another manager role")
        try:
            process_generation = uuid.UUID(self.process_generation)
        except (AttributeError, ValueError) as error:
            raise RuntimeError("manager process generation is invalid") from error
        if str(process_generation) != self.process_generation:
            raise RuntimeError("manager process generation is not canonical")
        if local_rank.process_generation != process_generation.bytes:
            raise RuntimeError("terminal startup binding has another generation")
        if local_rank.nixl_agent_name != self.agent.name:
            raise RuntimeError("terminal startup binding has another NIXL agent")
        metadata_digest = hashlib.sha256(self.agent_metadata).digest()
        if local_rank.nixl_agent_metadata_sha256 != metadata_digest:
            raise RuntimeError("terminal startup binding has another NIXL metadata")
        if (
            local_rank.tensor_parallel_rank != self.attn_tp_rank
            or local_rank.tensor_parallel_size != self.attn_tp_size
        ):
            raise RuntimeError("terminal startup binding has another TP identity")

        remote_role = (
            TerminalOwnerRole.DECODE
            if expected_role is TerminalOwnerRole.SOURCE
            else TerminalOwnerRole.SOURCE
        )
        expected_remote_ranks = tuple(
            rank for rank in binding.matrix.ranks if rank.role is remote_role
        )
        if len(expected_remote_ranks) == 0:
            raise RuntimeError("terminal startup matrix has no cross-role peers")
        self._terminal_startup_binding = binding
        self._terminal_startup_peer_enrollment = _TerminalStartupPeerEnrollment(
            binding=binding,
            expected_remote_ranks=expected_remote_ranks,
        )

    def wait_for_terminal_peer_enrollment(self, timeout_seconds: float) -> None:
        """Wait event-first for the exact cross-role native roster to freeze.

        :param timeout_seconds: Positive finite enrollment deadline.
        :raises RuntimeError: If enrollment is absent or remains incomplete.
        """

        if (
            type(timeout_seconds) is not float
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be a positive finite float")
        enrollment = self._require_terminal_startup_peer_enrollment()
        if enrollment.frozen_event.wait(timeout_seconds):
            return
        with enrollment.lock:
            enrolled_count = len(enrollment.enrolled_keys)
            expected_count = len(enrollment.expected_keys)
        raise RuntimeError(
            "terminal native peer enrollment did not freeze: "
            f"{enrolled_count}/{expected_count} peers retained"
        )

    def _require_terminal_startup_peer_enrollment(
        self,
    ) -> _TerminalStartupPeerEnrollment:
        """Return the installed peer authority for a terminal deployment.

        :returns: Exact mutable-until-frozen manager-owned enrollment.
        :raises RuntimeError: If composition bypassed binding installation.
        """

        binding = self.terminal_startup_binding
        if binding is None:
            raise RuntimeError("terminal startup binding is not configured")
        enrollment = self._terminal_startup_peer_enrollment
        if enrollment is None or enrollment.binding != binding:
            raise RuntimeError(
                "terminal startup binding has no manager-owned peer enrollment"
            )
        return enrollment

    def _terminal_remote_rank(
        self,
        *,
        agent_name: str,
        agent_metadata: bytes,
        process_generation: str,
        role: TerminalOwnerRole,
        tensor_parallel_rank: int,
        tensor_parallel_size: int,
    ) -> TerminalStartupRankAdvertisement:
        """Authenticate full native metadata against one sealed matrix row.

        :param agent_name: Native metadata-selected remote agent name.
        :param agent_metadata: Complete remote NIXL agent metadata.
        :param process_generation: Canonical remote process generation.
        :param role: Required cross-role owner role.
        :param tensor_parallel_rank: Remote rank within its service.
        :param tensor_parallel_size: Exact remote service TP width.
        :returns: Sole matching static-member-bound matrix row.
        :raises RuntimeError: If any full-metadata identity field differs.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        try:
            canonical_name = validate_nixl_agent_name(agent_name)
            canonical_metadata = validate_nixl_agent_metadata(agent_metadata)
        except ValueError as error:
            raise RuntimeError("terminal remote NIXL identity is invalid") from error
        canonical_generation = self._canonical_process_generation(process_generation)
        generation_bytes = uuid.UUID(canonical_generation).bytes
        metadata_digest = hashlib.sha256(canonical_metadata).digest()
        matches = tuple(
            rank
            for rank in enrollment.expected_remote_ranks
            if rank.nixl_agent_name == canonical_name
        )
        if len(matches) != 1:
            raise RuntimeError("terminal remote agent is absent from sealed matrix")
        rank = matches[0]
        if (
            rank.role is not role
            or rank.process_generation != generation_bytes
            or rank.nixl_agent_metadata_sha256 != metadata_digest
            or rank.tensor_parallel_rank != tensor_parallel_rank
            or rank.tensor_parallel_size != tensor_parallel_size
        ):
            raise RuntimeError(
                "terminal full-metadata peer differs from sealed matrix identity"
            )
        return rank

    def _join_terminal_startup_cohort(self) -> TerminalStartupRankBinding | None:
        """Freeze complete native membership before any manager thread starts.

        :returns: Complete immutable rank binding, or ``None`` when unconfigured.
        :raises ValueError: If terminal configuration or local topology is partial.
        """

        cohort = self.server_args.pd_terminal_deployment_cohort
        local_membership = self.server_args.pd_terminal_local_membership
        timeout_seconds = self.server_args.pd_terminal_startup_timeout_seconds
        configured_values = (cohort, local_membership, timeout_seconds)
        if all(value is None for value in configured_values):
            return None
        if any(value is None for value in configured_values):
            raise ValueError(
                "terminal startup cohort, local membership, and timeout must be "
                "configured together"
            )
        if type(cohort) is not TerminalDeploymentCohort:
            raise TypeError("pd_terminal_deployment_cohort has an invalid type")
        if type(local_membership) is not TerminalDeploymentLocalService:
            raise TypeError("pd_terminal_local_membership has an invalid type")
        expected_role = (
            TerminalDeploymentRole.PREFILL
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else TerminalDeploymentRole.DECODE
        )
        if local_membership.role is not expected_role:
            raise ValueError("terminal local membership role differs from NIXL manager")
        if self.pp_size != 1 or self.attn_cp_size != 1:
            raise ValueError("terminal startup binding requires PP1 and attention CP1")
        if local_membership.tensor_parallel_size != self.attn_tp_size:
            raise ValueError(
                "terminal local membership TP width differs from NIXL manager"
            )
        if type(timeout_seconds) is not float:
            raise TypeError("pd_terminal_startup_timeout_seconds must be a float")

        expectation = build_terminal_startup_cohort_expectation(
            cohort,
            local_membership,
        )
        return join_terminal_startup_rank(
            cohort,
            local_membership,
            expectation,
            tensor_parallel_rank=self.attn_tp_rank,
            process_generation=self.process_generation,
            nixl_agent_name=self.agent.name,
            nixl_agent_metadata=self.agent_metadata,
            timeout_seconds=timeout_seconds,
        )

    def _terminal_bootstrap_address(self) -> NetworkAddress:
        """Return the static source-owned startup control address.

        :returns: Exact cohort bootstrap address.
        :raises RuntimeError: If terminal startup configuration is incomplete.
        """

        cohort = self.server_args.pd_terminal_deployment_cohort
        if type(cohort) is not TerminalDeploymentCohort:
            raise RuntimeError("terminal deployment cohort is unavailable")
        endpoint = cohort.prefill.bootstrap_endpoint
        return NetworkAddress(endpoint.host, endpoint.port)

    def _terminal_startup_timeout_seconds(self) -> float:
        """Return the hash-bound terminal startup deadline.

        :returns: Positive finite timeout in seconds.
        :raises RuntimeError: If the deadline is absent or malformed.
        """

        timeout_seconds = self.server_args.pd_terminal_startup_timeout_seconds
        if (
            type(timeout_seconds) is not float
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise RuntimeError("terminal startup timeout is unavailable")
        return timeout_seconds

    def _enroll_terminal_source_routes(
        self,
        deadline: float,
    ) -> TerminalNixlSourceRoster:
        """Fetch, authenticate, and retain the complete source peer roster.

        :param deadline: Absolute monotonic startup deadline.
        :returns: Canonical source routes used for one-time registration.
        :raises RuntimeError: If this rank is not a decoder or enrollment fails.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        binding = enrollment.binding
        if binding.advertisement.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("only a terminal decoder enrolls source routes")
        address = self._terminal_bootstrap_address()
        endpoint = address.to_url() + TERMINAL_NIXL_SOURCE_ROSTER_ROUTE
        roster = fetch_terminal_nixl_source_roster(
            endpoint,
            binding.advertisement,
            binding.matrix,
            NIXL_BOOTSTRAP_PEER_PROTOCOL,
            self._remaining_terminal_startup_seconds(deadline),
        )
        self.enroll_terminal_prefill_routes(
            address.to_host_port_str(),
            tuple(route.bootstrap_info() for route in roster.routes),
        )
        return roster

    def _enroll_terminal_source_publication_routes(
        self,
        deadline: float,
    ) -> TerminalSourcePublicationControl:
        """Fetch and freeze direct same-service source control listeners.

        :param deadline: Absolute monotonic startup deadline.
        :returns: Process-local publication route owner.
        :raises RuntimeError: If this rank is not a source or any actual
            listener differs from the sealed startup authority.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        binding = enrollment.binding
        local = binding.advertisement
        if local.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("only a terminal source enrolls source control routes")
        address = self._terminal_bootstrap_address()
        endpoint = address.to_url() + TERMINAL_NIXL_SOURCE_ROSTER_ROUTE
        roster = fetch_terminal_nixl_source_roster(
            endpoint,
            local,
            binding.matrix,
            NIXL_BOOTSTRAP_PEER_PROTOCOL,
            self._remaining_terminal_startup_seconds(deadline),
        )
        route_roster = TerminalSourcePublicationRouteRoster.from_startup_roster(
            binding,
            roster,
            NetworkAddress(self.local_ip, self.rank_port),
        )
        return TerminalSourcePublicationControl(
            route_roster,
            local.terminal_identity,
            self._send_terminal_source_publication_frames,
        )

    def _send_terminal_source_publication_frames(
        self,
        endpoint: NetworkAddress,
        frames: tuple[bytes, ...],
    ) -> None:
        """Send one publisher outcome to an enrolled source listener.

        :param endpoint: Exact manager listener from the frozen source roster.
        :param frames: Closed source-publication control message.
        """

        if type(endpoint) is not NetworkAddress:
            raise TypeError("endpoint must be NetworkAddress")
        if type(frames) is not tuple or any(
            type(frame) is not bytes for frame in frames
        ):
            raise TypeError("frames must be a tuple of bytes")
        control = self._terminal_source_publication_control
        if control is None:
            raise RuntimeError("source publication control is not enrolled")
        if endpoint not in tuple(route.endpoint for route in control.roster.routes):
            raise RuntimeError("source publication endpoint is not enrolled")
        with self._packed_control_send_lock:
            socket = self._connect(endpoint.to_tcp(), is_ipv6=endpoint.is_ipv6)
            socket.send_multipart(frames)

    @staticmethod
    def _remaining_terminal_startup_seconds(deadline: float) -> float:
        """Return positive time remaining before one startup deadline.

        :param deadline: Absolute monotonic deadline.
        :returns: Positive remaining seconds.
        :raises RuntimeError: If the deadline is exhausted or malformed.
        """

        if type(deadline) is not float or not math.isfinite(deadline):
            raise ValueError("terminal startup deadline must be finite")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeError("terminal startup activation timed out")
        return remaining

    def _receive_terminal_startup_frames(
        self,
        deadline: float,
    ) -> tuple[bytes, ...]:
        """Receive one startup multipart message through an fd-backed wait.

        This method is the sole owner of ``server_socket`` until activation
        commits. Ownership transfers once to the role-specific runtime worker.

        :param deadline: Absolute monotonic startup deadline.
        :returns: Exact multipart message.
        :raises RuntimeError: If no message arrives before the deadline.
        """

        remaining = self._remaining_terminal_startup_seconds(deadline)
        timeout_ms = max(1, math.ceil(remaining * 1000.0))
        events = self.server_socket.poll(timeout=timeout_ms, flags=zmq.POLLIN)
        if events & zmq.POLLIN == 0:
            raise RuntimeError("terminal startup activation timed out")
        frames = tuple(self.server_socket.recv_multipart())
        if len(frames) == 0:
            raise RuntimeError("terminal startup received an empty message")
        return frames

    def _receive_terminal_decoder_enrollments(
        self,
        deadline: float,
    ) -> tuple[_TerminalDecoderEnrollment, ...]:
        """Retain the exact decoder roster before source workers start.

        :param deadline: Absolute monotonic startup deadline.
        :returns: Canonically matrix-ordered decoder enrollments.
        :raises RuntimeError: If traffic or membership differs from the matrix.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        binding = enrollment.binding
        if binding.advertisement.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("only a terminal source receives decoder enrollment")

        received: dict[tuple[str, int], _TerminalDecoderEnrollment] = {}
        while len(received) < len(enrollment.expected_remote_ranks):
            frames = self._receive_terminal_startup_frames(deadline)
            if (
                len(frames) != _PACKED_REGISTRATION_FRAME_COUNT + 1
                or frames[0] != GUARD
                or frames[1] != b"None"
            ):
                raise RuntimeError(
                    "source received runtime traffic before terminal startup commit"
                )
            registration = KVArgsRegisterInfo.from_zmq(list(frames[1:]))
            rank = self.enroll_terminal_decoder_peer(registration)
            if rank.key in received:
                raise RuntimeError("terminal decoder enrollment was duplicated")
            received[rank.key] = _TerminalDecoderEnrollment(
                rank=rank,
                registration=registration,
                frames=frames,
            )

        if not self.terminal_peer_enrollment_frozen:
            raise RuntimeError("terminal decoder roster did not freeze")
        return tuple(received[rank.key] for rank in enrollment.expected_remote_ranks)

    def _send_terminal_startup_ack(
        self,
        enrollment: _TerminalDecoderEnrollment,
        route_table: TerminalDecodeControlRouteTable,
    ) -> None:
        """Acknowledge one decoder after the complete source roster freezes.

        :param enrollment: Exact retained decoder registration and matrix row.
        :param route_table: Complete listener table for the target service.
        :raises RuntimeError: If the decoder acknowledgement route is invalid.
        """

        binding = self._require_terminal_startup_peer_enrollment().binding
        acknowledgement = build_terminal_startup_enrollment_ack(
            binding.matrix,
            binding.advertisement,
            enrollment.rank,
            enrollment.frames,
            route_table.digest,
        )
        endpoint = enrollment.registration.endpoint
        port = enrollment.registration.dst_port
        if type(endpoint) is not str or len(endpoint) == 0:
            raise RuntimeError("terminal decoder acknowledgement address is absent")
        if type(port) is not int or not 1 <= port <= 65535:
            raise RuntimeError("terminal decoder acknowledgement port is invalid")
        address = NetworkAddress(endpoint, port)
        payload = encode_terminal_startup_enrollment_ack(acknowledgement)
        route_payload = encode_terminal_decode_control_route_table(route_table)
        with self._packed_control_send_lock:
            socket = self._connect(address.to_tcp(), is_ipv6=address.is_ipv6)
            socket.send_multipart(
                (TERMINAL_STARTUP_ENROLLMENT_ACK_TAG, payload, route_payload)
            )

    @staticmethod
    def _build_terminal_decode_route_tables(
        binding: TerminalStartupRankBinding,
        enrollments: tuple[_TerminalDecoderEnrollment, ...],
    ) -> dict[str, TerminalDecodeControlRouteTable]:
        """Freeze all source-retained decoder listeners by TP service.

        :param binding: Exact source startup authority.
        :param enrollments: Complete matrix-ordered decoder registrations.
        :returns: One immutable route table per decoder service.
        """

        if type(binding) is not TerminalStartupRankBinding:
            raise TypeError("binding must be TerminalStartupRankBinding")
        if binding.advertisement.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("only a source can freeze decoder route tables")
        if type(enrollments) is not tuple:
            raise TypeError("enrollments must be a tuple")
        by_key = {enrollment.rank.key: enrollment for enrollment in enrollments}
        if len(by_key) != len(enrollments):
            raise RuntimeError("decoder route enrollment identity was duplicated")

        service_ids = tuple(
            dict.fromkeys(
                rank.service_id
                for rank in binding.matrix.ranks
                if rank.role is TerminalOwnerRole.DECODE
            )
        )
        tables: dict[str, TerminalDecodeControlRouteTable] = {}
        for service_id in service_ids:
            ranks = tuple(
                rank
                for rank in binding.matrix.ranks
                if rank.role is TerminalOwnerRole.DECODE
                and rank.service_id == service_id
            )
            registrations: list[
                tuple[
                    TerminalStartupRankAdvertisement,
                    NetworkAddress,
                    tuple[bytes, ...],
                ]
            ] = []
            for rank in ranks:
                enrollment = by_key.get(rank.key)
                if enrollment is None:
                    raise RuntimeError("decoder route table is missing a matrix rank")
                registration = enrollment.registration
                registrations.append(
                    (
                        rank,
                        NetworkAddress(registration.endpoint, registration.dst_port),
                        enrollment.frames,
                    )
                )
            tables[service_id] = build_terminal_decode_control_route_table(
                binding.matrix,
                service_id,
                tuple(registrations),
            )
        if len(tables) == 0:
            raise RuntimeError("terminal deployment has no decoder route table")
        return tables

    def _wait_for_terminal_source_acks(
        self,
        registration_frames: tuple[bytes, ...],
        deadline: float,
    ) -> TerminalDecodeControlRouteTable:
        """Require one exact enrollment acknowledgement from every source rank.

        :param registration_frames: Complete registration sent to every source.
        :param deadline: Absolute monotonic startup deadline.
        :returns: Source-agreed immutable same-service decoder listener table.
        :raises RuntimeError: If any acknowledgement is absent or conflicts.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        binding = enrollment.binding
        local_rank = binding.advertisement
        if local_rank.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("only a terminal decoder receives source ACKs")

        received: set[tuple[str, int]] = set()
        expected = enrollment.expected_keys
        route_table: TerminalDecodeControlRouteTable | None = None
        while received != expected:
            frames = self._receive_terminal_startup_frames(deadline)
            if len(frames) != 3 or frames[0] != TERMINAL_STARTUP_ENROLLMENT_ACK_TAG:
                raise RuntimeError(
                    "decoder received runtime traffic before terminal startup commit"
                )
            acknowledgement = decode_terminal_startup_enrollment_ack(frames[1])
            received_route_table = decode_terminal_decode_control_route_table(
                binding.matrix,
                frames[2],
            )
            acknowledgement.require_matrix(binding.matrix)
            acknowledgement.require_decoder_registration(registration_frames)
            if acknowledgement.decoder_control_route_table_sha256 != (
                received_route_table.digest
            ):
                raise RuntimeError(
                    "terminal enrollment acknowledgement binds another route table"
                )
            if (
                acknowledgement.target_decoder_service_id != local_rank.service_id
                or acknowledgement.target_decoder_tensor_parallel_rank
                != local_rank.tensor_parallel_rank
                or acknowledgement.target_decoder_process_generation
                != local_rank.process_generation
            ):
                raise RuntimeError(
                    "terminal enrollment acknowledgement targets another decoder"
                )
            received_route_table.require_local_registration(
                binding,
                NetworkAddress(self.local_ip, self.rank_port),
                registration_frames,
            )
            source_key = (
                acknowledgement.source_service_id,
                acknowledgement.source_tensor_parallel_rank,
            )
            if source_key not in expected:
                raise RuntimeError(
                    "terminal enrollment acknowledgement has an unknown source"
                )
            if source_key in received:
                raise RuntimeError(
                    "terminal enrollment acknowledgement was received twice"
                )
            received.add(source_key)
            if route_table is None:
                route_table = received_route_table
            elif received_route_table != route_table:
                raise RuntimeError("source ranks disagree on decoder control routes")
        if route_table is None:
            raise RuntimeError("decoder received no control route table")
        return route_table

    def _activate_terminal_source(self, deadline: float) -> None:
        """Commit source peer authority and hand off to runtime workers.

        :param deadline: Absolute monotonic startup deadline.
        """

        if self._terminal_source_publication_control is not None:
            raise RuntimeError("source publication control is already enrolled")
        publication_control = self._enroll_terminal_source_publication_routes(deadline)
        self._terminal_source_publication_control = publication_control
        enrollments = self._receive_terminal_decoder_enrollments(deadline)
        route_tables = self._build_terminal_decode_route_tables(
            self._require_terminal_startup_peer_enrollment().binding,
            enrollments,
        )
        for enrollment in enrollments:
            self._send_terminal_startup_ack(
                enrollment,
                route_tables[enrollment.rank.service_id],
            )

    def _activate_terminal_decoder(self, deadline: float) -> None:
        """Commit decoder peer authority and hand off to runtime workers.

        :param deadline: Absolute monotonic startup deadline.
        """

        roster = self._enroll_terminal_source_routes(deadline)
        if not self.terminal_peer_enrollment_frozen:
            raise RuntimeError("terminal source roster did not freeze")
        registration_frames = self._decode_registration_frames()
        for route in roster.routes:
            self._send_terminal_decoder_registration(
                route.bootstrap_info(),
                registration_frames,
            )
        self._terminal_decode_control_routes = self._wait_for_terminal_source_acks(
            registration_frames,
            deadline,
        )

    @staticmethod
    def _terminal_rank_runtime_config(
        physical_capacity: int,
    ) -> TerminalRankRuntimeConfig:
        """Derive every native queue from one physical request authority.

        Each live request can emit every event at most once. Routed action
        queues use their exact applicable action populations, while scheduler,
        coordinator, decode-work, and publication queues admit at most one
        simultaneous action per live generation.

        :param physical_capacity: Exact metadata-row request capacity.
        :returns: Complete role-neutral runtime queue configuration.
        """

        if type(physical_capacity) is not int or physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        event_population = len(NativeTerminalOwnerEventKind)
        action_population = len(NativeTerminalOwnerActionKind)
        return TerminalRankRuntimeConfig(
            input_capacity=physical_capacity * event_population,
            output_capacity=physical_capacity * action_population,
            maximum_live_lifecycles=physical_capacity,
            scheduler_capacity=physical_capacity,
            coordinator_capacity=physical_capacity,
            lifecycle_capacity=physical_capacity * 2,
            source_work_capacity=physical_capacity * 3,
            decode_work_capacity=physical_capacity,
            publisher_capacity=physical_capacity,
            observation_capacity=physical_capacity * event_population,
            native_producer_retirement_timeout_seconds=terminal_deadline_spec(
                TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
            ).seconds,
        )

    def _source_request_ready_importers(
        self,
        binding: TerminalStartupRankBinding,
    ) -> dict[TerminalProcessIdentity, TerminalWireReceiptImportNamespace]:
        """Build one exact importer for every eligible decode coordinator.

        :param binding: Local source startup authority.
        :returns: Decode-rank-zero importer map keyed by authenticated identity.
        """

        importers: dict[TerminalProcessIdentity, TerminalWireReceiptImportNamespace] = (
            {}
        )
        for rank in binding.matrix.ranks:
            if rank.role is not TerminalOwnerRole.DECODE:
                continue
            if rank.tensor_parallel_rank != 0:
                continue
            identity = rank.terminal_identity
            importers[identity] = TerminalWireReceiptImportNamespace(identity)
        if len(importers) == 0:
            raise RuntimeError("terminal source has no decode coordinator importer")
        return importers

    def _compose_terminal_dflash_source_owner(
        self,
        actor: PackedPrefillRuntime,
        grouped_nixl: GroupedNixlTerminalOwner,
    ) -> DFlashBoundarySourceTransportOwner | None:
        """Bind canonical DFlash boundary transport to enrolled native ownership.

        :param actor: Exact rank-local packed source actor.
        :param grouped_nixl: Sole request-level native completion owner.
        :returns: Canonical DFlash owner, otherwise ``None`` off the writer rank.
        """

        canonical = self._is_canonical_aux_writer()
        pool = self._terminal_dflash_source_pool
        if not canonical:
            if pool is not None:
                raise RuntimeError("noncanonical source retained a DFlash row pool")
            return None
        if pool is None:
            raise RuntimeError("canonical source has no registered DFlash row pool")
        if self._terminal_dflash_source_owner is not None:
            raise RuntimeError("terminal DFlash source owner is already composed")

        owner = DFlashBoundarySourceTransportOwner(
            pool=pool,
            agent=self.agent,
            direct_owner=grouped_nixl.dflash_endpoint,
            writer_id=actor.writer_id,
            post=lambda handle: self._post_terminal_transfer_once(
                handle,
                "terminal DFlash boundary transfer",
            ),
        )
        self._terminal_dflash_source_owner = owner
        return owner

    def _compose_terminal_source_work(
        self,
        actor: PackedPrefillRuntime,
        grouped_nixl: GroupedNixlTerminalOwner,
        dflash_owner: DFlashBoundarySourceTransportOwner | None,
    ) -> PackedTerminalSourceWork:
        """Compose owner-earned source work over typed transport ownership.

        :param actor: Sole packed source request registry.
        :param grouped_nixl: Sole main and DFlash completion group owner.
        :param dflash_owner: Canonical device-row transport owner, if local.
        :returns: Complete forward-independent source work callbacks.
        """

        canonical = self._is_canonical_aux_writer()
        if canonical != (dflash_owner is not None):
            raise RuntimeError("DFlash owner presence differs from canonical rank")

        def transport_submission(
            submission: PackedTerminalSourceSubmission,
        ) -> PackedPrefillSubmission:
            transport = submission.transport_submission
            if type(transport) is not PackedPrefillSubmission:
                raise TypeError("terminal source transport has another schema")
            return transport

        def dflash_source(
            transport: PackedPrefillSubmission,
        ) -> DFlashBoundaryPrefillSource:
            auxiliary = transport.auxiliary_source
            if type(auxiliary) is not PackedTerminalDFlashAuxiliarySource:
                raise TypeError("canonical terminal source lacks DFlash ownership")
            return auxiliary.prefill_source

        def post_gather(
            submission: PackedTerminalSourceSubmission,
            action: NativeTerminalOwnerAction,
        ) -> None:
            transport = transport_submission(submission)
            expected_members = grouped_nixl_source_members(canonical)
            grouped_nixl.begin_group(action.binding.digest, expected_members)
            post_auxiliary: Callable[[PackedPrefillSubmission], object] | None = None
            if dflash_owner is not None:
                source = dflash_source(transport)

                def post_auxiliary(
                    exact_transport: PackedPrefillSubmission,
                ) -> object:
                    if exact_transport is not transport:
                        raise RuntimeError("source actor changed transport identity")
                    return dflash_owner.post(
                        plan=transport.plan,
                        source_lease=source.lease,
                        destination_device_index=(
                            transport.destination.route.destination_gpu_id
                        ),
                        remote_handle=transport.control.remote_handle,
                        remote_agent_name=transport.control.peer.agent_name,
                        binding_digest=action.binding.digest,
                    )

            try:
                actor.begin_terminal_owner_transfer(action, post_auxiliary)
                grouped_nixl.seal_group(action.binding.digest)
            except Exception:
                logger.error(
                    "Terminal grouped source post failed:\n%s",
                    traceback.format_exc(),
                )
                grouped_nixl.quarantine_group(
                    action.binding.digest,
                    "terminal source gather or transfer post became ambiguous",
                )
                raise

        def send_outcomes(
            submission: PackedTerminalSourceSubmission,
            action: NativeTerminalOwnerAction,
        ) -> None:
            transport = transport_submission(submission)
            settle_auxiliary: (
                Callable[[object, NativeTerminalOwnerAction], object] | None
            ) = None
            if dflash_owner is not None:
                source = dflash_source(transport)
                projection = submission.output_projection
                if type(projection) is not PrefillTerminalGatewayOutputProjection:
                    raise TypeError("terminal source output has another schema")

                def settle_auxiliary(
                    transfer: object,
                    exact_action: NativeTerminalOwnerAction,
                ) -> object:
                    if type(transfer) is not DFlashBoundarySourceTransfer:
                        raise TypeError("source actor retained another aux transfer")
                    metadata = source.metadata_from_result_slot(projection.result_slot)
                    return dflash_owner.settle(
                        transfer,
                        exact_action,
                        metadata,
                    )

            actor.send_terminal_owner_outcomes(
                action,
                lambda lane, exact_action: lane.settle_terminal_completion(
                    exact_action
                ),
                settle_auxiliary,
            )

        def send_ack(
            submission: PackedTerminalSourceSubmission,
            action: NativeTerminalOwnerAction,
        ) -> None:
            transport = transport_submission(submission)
            release_auxiliary: (
                Callable[[object, NativeTerminalOwnerAction], None] | None
            ) = None
            if dflash_owner is not None:
                dflash_source(transport)

                def release_auxiliary(
                    transfer: object,
                    exact_action: NativeTerminalOwnerAction,
                ) -> None:
                    if type(transfer) is not DFlashBoundarySourceTransfer:
                        raise TypeError("source actor retained another aux transfer")
                    if exact_action is not action:
                        raise RuntimeError("source actor changed ACK authority")
                    dflash_owner.release(transfer, exact_action)

            actor.settle_terminal_owner_teardown(
                action,
                lambda lane, exact_action: lane.release_terminal_transfer(exact_action),
                release_auxiliary,
            )

        def quarantine(action: NativeTerminalOwnerAction) -> None:
            actor.quarantine_terminal_owner_request(
                action,
                "native source lifecycle quarantined terminal transport",
                lambda lane, exact_action: lane.settle_terminal_failure(exact_action),
                (
                    None
                    if dflash_owner is None
                    else lambda transfer, exact_action: dflash_owner.settle_failure(
                        transfer,
                        exact_action,
                    )
                ),
            )

        return PackedTerminalSourceWork(
            post_gather=post_gather,
            send_outcomes=send_outcomes,
            send_ack=send_ack,
            quarantine=quarantine,
            observe_output=self._observe_terminal_output,
        )

    @staticmethod
    def _decode_coordinator_importers(
        binding: TerminalStartupRankBinding,
    ) -> tuple[TerminalWireReceiptImportNamespace, ...]:
        """Build canonical same-service decode importers for rank zero.

        :param binding: Local decode startup authority.
        :returns: Complete TP-rank importer roster, or empty off rank zero.
        """

        local = binding.advertisement
        if local.tensor_parallel_rank != 0:
            return ()
        ranks = tuple(
            rank
            for rank in binding.matrix.ranks
            if rank.role is TerminalOwnerRole.DECODE
            and rank.service_id == local.service_id
        )
        if tuple(rank.tensor_parallel_rank for rank in ranks) != tuple(
            range(local.tensor_parallel_size)
        ):
            raise RuntimeError("decode coordinator roster is not a complete TP group")
        return tuple(
            TerminalWireReceiptImportNamespace(rank.terminal_identity) for rank in ranks
        )

    def _compose_terminal_source(
        self,
        binding: TerminalStartupRankBinding,
        enrollment: TerminalRankRuntimeEnrollment,
        installation: NixlTerminalRuntimeInstallation,
    ) -> tuple[PackedTerminalSourceServing, PackedTerminalOutputPublisher | None]:
        """Construct dormant source serving around the enrolled runtime.

        :param binding: Exact local startup authority.
        :param enrollment: Sole rank runtime and native NIXL producer.
        :param installation: Scheduler and route-owned dependencies.
        :returns: Source serving and optional canonical publisher.
        """

        actor = self._packed_prefill_runtime
        if actor is None:
            raise RuntimeError("terminal source actor is unavailable")
        cuda_source = enrollment.cuda_source
        if cuda_source is None:
            raise RuntimeError("terminal source native work boundary is unavailable")
        publication_control = self._terminal_source_publication_control
        if publication_control is None:
            raise RuntimeError("source publication control is unavailable")
        if self._terminal_grouped_nixl_owner is not None:
            raise RuntimeError("terminal grouped NIXL owner is already composed")
        grouped_nixl = GroupedNixlTerminalOwner(
            self.agent,
            channel_capacity=installation.terminal_request_capacity * 2,
        )
        self._terminal_grouped_nixl_owner = grouped_nixl
        actor.bind_direct_terminal_owner(grouped_nixl.main_endpoint)
        dflash_owner = self._compose_terminal_dflash_source_owner(
            actor,
            grouped_nixl,
        )
        source_work = self._compose_terminal_source_work(
            actor,
            grouped_nixl,
            dflash_owner,
        )
        importers = self._source_request_ready_importers(binding)
        self._terminal_source_receipt_importers = importers
        local_identity = binding.advertisement.terminal_identity
        publication_control.roster.route_for(local_identity)
        terminal_clock_ns = SystemTerminalOwnerClock().now_ns
        publisher: PackedTerminalOutputPublisher | None = None
        if local_identity.tp_rank == 0:
            endpoint = installation.gateway_endpoint
            if endpoint is None:
                raise RuntimeError("canonical source gateway endpoint disappeared")

            def result_listener(result: TerminalGatewayPublicationResult) -> None:
                """Fan publisher authority through the sole source route owner.

                :param result: Immutable gateway publication outcome.
                """

                publication_control.publish_result(result)

            publisher = PackedTerminalOutputPublisher(
                capacity=installation.terminal_request_capacity,
                sink_factory=ZmqTerminalGatewaySinkFactory(endpoint),
                payload_encoder=PrefillTerminalGatewayPayloadEncoder(),
                wire_issuer=TerminalWireReceiptIssuer(local_identity),
                request_ready_authorities=frozenset(
                    importer.authority for importer in importers.values()
                ),
                result_listener=result_listener,
                fatal_listener=self._terminal_publisher_failed,
                clock_ns=terminal_clock_ns,
            )

        def retire_submission(
            submission: PackedTerminalSourceSubmission,
            action: NativeTerminalOwnerAction,
        ) -> None:
            """Retire route replay state after native lifecycle retirement.

            :param submission: Successfully retired source submission.
            :param action: Exact joined native retirement authority.
            """

            transport = submission.transport_submission
            if type(transport) is not PackedPrefillSubmission:
                raise TypeError("retired source transport has another schema")
            importer = self._terminal_source_receipt_importers.get(
                submission.identity.request_ready_issuer
            )
            if importer is None:
                raise RuntimeError("retired source importer disappeared")
            actor.require_terminal_owner_retirement(action, transport)
            importer.require_active_binding(submission.identity.local_binding)
            publication_control.require_retirable_binding(
                submission.identity.local_binding
            )
            importer.retire_binding(submission.identity.local_binding)
            publication_control.retire_binding(submission.identity.local_binding)
            retired_transport = actor.retire_terminal_owner_request(action)
            if retired_transport is not transport:
                raise RuntimeError("actor retired another source transport")
            self._retire_successful_source_room(transport.plan.key.room_id)

        def resource_inventory() -> PackedTerminalSourceResourceInventory:
            """Project every external source owner into one health receipt.

            :returns: Actor, DFlash, and pre-lifecycle quarantine inventory.
            """

            actor_inventory = actor.terminal_owner_inventory()
            request_ready_imports = tuple(
                sorted(
                    {
                        digest
                        for importer in self._terminal_source_receipt_importers.values()
                        for digest in importer.active_binding_digests
                    }
                )
            )
            publication_inventory = publication_control.inventory()
            source_transfer_rooms = tuple(sorted(self.transfer_infos))
            source_prefix_rooms = tuple(sorted(self.req_to_decode_prefix_len))
            source_prefetched_rooms: tuple[int, ...] = ()
            source_prefetch_requested_rooms: tuple[int, ...] = ()
            if self.enable_staging and self._staging_ctx is not None:
                source_prefetched_rooms = tuple(
                    sorted(self._staging_ctx.prefetched_rooms)
                )
                source_prefetch_requested_rooms = tuple(
                    sorted({key[0] for key in self._staging_ctx.prefetch_requested})
                )
            if dflash_owner is None:
                dflash_counts = (0, 0, 0, 0, 0, 0)
                row_counts = (0, 0, 0)
            else:
                dflash_inventory = dflash_owner.inventory()
                dflash_counts = (
                    dflash_inventory.active_count,
                    dflash_inventory.posted_count,
                    dflash_inventory.settled_count,
                    dflash_inventory.released_count,
                    dflash_inventory.quarantined_count,
                    dflash_inventory.unowned_native_handle_count,
                )
                row_counts = dflash_owner.source_row_inventory()
            with self._terminal_unpublished_source_quarantine_lock:
                unpublished = tuple(
                    sorted(self._terminal_unpublished_source_quarantine)
                )
                unpublished_result_slots = tuple(
                    sorted(
                        digest
                        for digest, submission in (
                            self._terminal_unpublished_source_quarantine.items()
                        )
                        if submission.output_projection is not None
                    )
                )
            return PackedTerminalSourceResourceInventory(
                actor_active_binding_digests=actor_inventory.active_bindings,
                actor_quarantined_binding_digests=(
                    actor_inventory.quarantined_bindings
                ),
                actor_waiting_for_ready_binding_digests=(
                    actor_inventory.waiting_for_ready_bindings
                ),
                actor_main_handle_binding_digests=(
                    actor_inventory.main_handle_bindings
                ),
                actor_auxiliary_handle_binding_digests=(
                    actor_inventory.auxiliary_handle_bindings
                ),
                actor_lane_binding_digests=actor_inventory.lane_bindings,
                request_ready_import_binding_digests=request_ready_imports,
                publication_control_active_binding_digests=(
                    publication_inventory.active_binding_digests
                ),
                publication_control_terminal_binding_digests=(
                    publication_inventory.terminal_binding_digests
                ),
                source_transfer_info_room_ids=source_transfer_rooms,
                source_prefix_length_room_ids=source_prefix_rooms,
                source_prefetched_room_ids=source_prefetched_rooms,
                source_prefetch_requested_room_ids=(source_prefetch_requested_rooms),
                dflash_active_transfer_count=dflash_counts[0],
                dflash_posted_transfer_count=dflash_counts[1],
                dflash_settled_transfer_count=dflash_counts[2],
                dflash_released_transfer_count=dflash_counts[3],
                dflash_quarantined_transfer_count=dflash_counts[4],
                dflash_unowned_native_handle_count=dflash_counts[5],
                dflash_free_row_count=row_counts[0],
                dflash_active_row_count=row_counts[1],
                dflash_quarantined_row_count=row_counts[2],
                unpublished_quarantined_binding_digests=unpublished,
                unpublished_quarantined_result_slot_binding_digests=(
                    unpublished_result_slots
                ),
            )

        serving = PackedTerminalSourceServing(
            runtime=enrollment.runtime,
            cuda_completion=cuda_source,
            local_identity=local_identity,
            publisher=publisher,
            metrics_sink=_NixlTerminalSourceMetrics(),
            clock_ns=terminal_clock_ns,
            physical_capacity=installation.terminal_request_capacity,
            process_fatal_handler=(installation.scheduler_process_fatal_handler),
            grouped_nixl=grouped_nixl,
            work=source_work,
            retire_native_producers=enrollment.retire_native_producers,
            resource_inventory=resource_inventory,
            retire_submission=retire_submission,
        )

        def deliver_publication(
            delivery: TerminalSourcePublicationDelivery,
        ) -> None:
            """Deliver startup-authenticated publisher authority to serving.

            :param delivery: Imported or locally issued publication authority.
            """

            serving.publication_receipt(
                wire_receipt=delivery.wire_receipt,
                local_receipt=delivery.local_receipt,
                authenticated_issuer=delivery.authenticated_issuer,
            )

        def publication_route_failed(reason: str) -> None:
            """Enter process-fatal ownership after route failure.

            :param reason: Sticky route or authentication failure evidence.
            """

            self._record_terminal_component_failure(reason, None)

        publication_control.bind_listener(
            deliver_publication,
            publication_route_failed,
        )
        return serving, publisher

    def _compose_terminal_decode(
        self,
        binding: TerminalStartupRankBinding,
        enrollment: TerminalRankRuntimeEnrollment,
        installation: NixlTerminalRuntimeInstallation,
    ) -> PackedTerminalDecodeServing:
        """Construct dormant decode serving around the enrolled runtime.

        :param binding: Exact local startup authority.
        :param enrollment: Sole rank runtime and native producers.
        :param installation: Scheduler and route-owned dependencies.
        :returns: Full decode serving composition.
        """

        controller = self._require_packed_decode_controller()
        cuda_scatter = enrollment.cuda_scatter
        if cuda_scatter is None:
            raise RuntimeError("terminal decode native work boundary is unavailable")
        if self._terminal_decode_control_routes is None:
            raise RuntimeError("terminal decode control routes are unavailable")
        local_identity = binding.advertisement.terminal_identity
        coordinator_issuer = (
            TerminalWireReceiptIssuer(local_identity)
            if local_identity.tp_rank == 0
            else None
        )
        terminal_clock_ns = SystemTerminalOwnerClock().now_ns
        return PackedTerminalDecodeServing(
            actor=controller.terminal_runtime,
            runtime=enrollment.runtime,
            cuda_completion=cuda_scatter,
            local_identity=local_identity,
            coordinator_issuer=coordinator_issuer,
            coordinator_importers=self._decode_coordinator_importers(binding),
            clock_ns=terminal_clock_ns,
            physical_capacity=installation.terminal_request_capacity,
            process_fatal_handler=(installation.scheduler_process_fatal_handler),
            work=PackedTerminalDecodeWork(
                send_delivery=self._send_terminal_decode_delivery,
                observe_output=self._observe_terminal_output,
            ),
            retire_native_producers=enrollment.retire_native_producers,
        )

    def _compose_terminal_runtime(self) -> None:
        """Construct, bind, and start the complete role-specific runtime."""

        binding = self._require_terminal_startup_peer_enrollment().binding
        installation = self._terminal_runtime_installation
        if installation is None:
            raise RuntimeError("terminal runtime dependencies are not installed")
        config = self._terminal_rank_runtime_config(
            installation.terminal_request_capacity
        )
        enrollment = TerminalRankRuntimeEnrollmentFactory(binding, config).create()
        local_role = binding.advertisement.role
        source_serving: PackedTerminalSourceServing | None = None
        decode_serving: PackedTerminalDecodeServing | None = None
        publisher: PackedTerminalOutputPublisher | None = None
        if local_role is TerminalOwnerRole.SOURCE:
            source_serving, publisher = self._compose_terminal_source(
                binding,
                enrollment,
                installation,
            )
            reactor = PackedTerminalProcessReactor.for_source(
                source_serving,
                self._terminal_reactor_failed,
            )
        else:
            decode_serving = self._compose_terminal_decode(
                binding,
                enrollment,
                installation,
            )
            reactor = PackedTerminalProcessReactor.for_decode(
                decode_serving,
                SystemTerminalOwnerClock().now_ns,
                self._terminal_reactor_failed,
            )

        self._terminal_runtime_enrollment = enrollment
        self._terminal_source_serving = source_serving
        self._terminal_decode_serving = decode_serving
        self._terminal_process_reactor = reactor
        self._terminal_output_publisher = publisher
        if source_serving is not None:
            bind_source = installation.bind_source_serving
            if bind_source is None:
                raise RuntimeError("source serving binder disappeared")
            bind_source(source_serving)
        else:
            bind_decode = installation.bind_decode_serving
            if bind_decode is None or decode_serving is None:
                raise RuntimeError("decode serving binder disappeared")
            bind_decode(decode_serving)

        if publisher is not None:
            publisher.start()
        if source_serving is not None:
            source_serving.start()
        elif decode_serving is not None:
            decode_serving.start()
        else:
            raise RuntimeError("terminal serving composition disappeared")
        reactor.start(self._terminal_startup_timeout_seconds())
        self._start_terminal_control_receiver()

    def _observe_terminal_output(self, output: object) -> None:
        """Project non-gating native output into the process logger.

        :param output: Immutable native runtime observation.
        """

        logger.debug("terminal runtime observation: %r", output)

    def _terminal_publisher_failed(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Record publisher death and wake scheduler fail-closed handling.

        :param reason: Stable publisher failure boundary.
        :param formatted_traceback: Complete originating traceback, if any.
        """

        self._record_terminal_component_failure(reason, formatted_traceback)

    def _terminal_reactor_failed(
        self,
        failure: PackedTerminalProcessReactorFailure,
    ) -> None:
        """Record reactor death and wake scheduler fail-closed handling.

        :param failure: Complete process-reactor failure evidence.
        """

        if type(failure) is not PackedTerminalProcessReactorFailure:
            raise TypeError("failure must be PackedTerminalProcessReactorFailure")
        self._record_terminal_component_failure(
            str(failure),
            failure.formatted_traceback,
        )

    def _record_terminal_component_failure(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Retain the first component death and wake scheduler admission once.

        :param reason: Stable process-fatal component boundary.
        :param formatted_traceback: Complete originating traceback, if any.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if formatted_traceback is not None and type(formatted_traceback) is not str:
            raise TypeError("formatted_traceback must be str or None")
        with self._terminal_process_fatal_lock:
            if self._terminal_process_fatal_reason is not None:
                return
            self._terminal_process_fatal_reason = reason
            self._terminal_process_fatal_traceback = formatted_traceback
        if formatted_traceback is None:
            logger.error("Terminal component failed: %s", reason)
        else:
            logger.error(
                "Terminal component failed: %s\n%s",
                reason,
                formatted_traceback,
            )
        reactor = self._terminal_process_reactor
        if reactor is not None:
            try:
                inventory = reactor.inventory()
                if inventory.started and inventory.admission_open:
                    reactor.stop_admission()
            except Exception:  # noqa: BLE001
                logger.error(
                    "Terminal reactor admission stop failed after component death:\n%s",
                    traceback.format_exc(),
                )
        installation = self._terminal_runtime_installation
        if installation is None:
            logger.error(
                "Terminal component failed before runtime installation: %s", reason
            )
            return
        try:
            installation.owner_dead_handler()
        except Exception:  # noqa: BLE001
            logger.error(
                "Terminal owner-death wake failed:\n%s",
                traceback.format_exc(),
            )

    def _terminal_rank_for_identity(
        self,
        identity: TerminalProcessIdentity,
    ) -> TerminalStartupRankAdvertisement:
        """Resolve one route-authenticated identity through the sealed matrix.

        :param identity: Exact process identity proved by its transport route.
        :returns: Sole matching startup rank.
        """

        if type(identity) is not TerminalProcessIdentity:
            raise TypeError("identity must be TerminalProcessIdentity")
        binding = self._require_terminal_startup_peer_enrollment().binding
        matches = tuple(
            rank for rank in binding.matrix.ranks if rank.terminal_identity == identity
        )
        if len(matches) != 1:
            raise RuntimeError("terminal identity is absent from the sealed matrix")
        return matches[0]

    def _send_terminal_decode_delivery(
        self,
        delivery: PackedTerminalDecodeWireDelivery,
    ) -> None:
        """Send one request terminal receipt directly to its exact owner.

        Same-service decoder listeners come from the source-agreed startup
        route table. Source listeners come from the already frozen native peer
        roster. Both paths use the manager's sole serialized control sender and
        introduce no request-path collective or polling cadence.

        :param delivery: Immutable coordinator fan-in or owner fan-out edge.
        """

        if type(delivery) is not PackedTerminalDecodeWireDelivery:
            raise TypeError("delivery must be PackedTerminalDecodeWireDelivery")
        enrollment = self._require_terminal_startup_peer_enrollment()
        local = enrollment.binding.advertisement
        if local.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("only a decoder can send decode terminal delivery")
        recipient_rank = self._terminal_rank_for_identity(delivery.recipient)
        if delivery.recipient == local.terminal_identity:
            raise RuntimeError("local decode delivery must not enter the transport")

        if recipient_rank.role is TerminalOwnerRole.DECODE:
            routes = self._terminal_decode_control_routes
            if routes is None:
                raise RuntimeError("terminal decode control routes are unavailable")
            endpoint = routes.route_for(delivery.recipient).endpoint
        else:
            peer = enrollment.prefill_peers.get(recipient_rank.key)
            if peer is None:
                raise RuntimeError("terminal source recipient is not enrolled")
            with self._prefill_peer_lock:
                if peer.handle in self._quarantined_remote_handles:
                    raise RuntimeError("terminal source recipient is quarantined")
            endpoint = peer.control_endpoint

        receipt = delivery.receipt
        message = PackedTerminalReceipt(
            key=receipt.binding.request_key,
            receipt_payload=receipt.encode(),
        )
        frames = tuple(
            encode_packed_control_frames(
                self.agent.name,
                self.process_generation,
                message,
            )
        )
        with self._packed_control_send_lock:
            socket = self._connect(endpoint.to_tcp(), is_ipv6=endpoint.is_ipv6)
            socket.send_multipart(frames)

    def receive_terminal_decode_receipt(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Deliver one same-service decode receipt through terminal serving.

        This is the narrow receive seam used by matrix-bound same-role routes.
        Cross-role packed controls join the same path after route authentication.

        :param wire_receipt: Canonical terminal wire authority.
        :param authenticated_issuer: Exact identity proved by the route.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        issuer_rank = self._terminal_rank_for_identity(authenticated_issuer)
        local = self._require_terminal_startup_peer_enrollment().binding.advertisement
        if (
            issuer_rank.role is not TerminalOwnerRole.DECODE
            or local.role is not TerminalOwnerRole.DECODE
            or issuer_rank.service_id != local.service_id
        ):
            raise RuntimeError("decode receipt route crosses serving identities")
        if wire_receipt.issuer != authenticated_issuer:
            raise RuntimeError("decode receipt asserts another route issuer")
        serving = self.terminal_decode_serving
        if wire_receipt.kind is TerminalReceiptKind.LOCAL_DECODE_READY or (
            wire_receipt.kind is TerminalReceiptKind.FAILURE
            and wire_receipt.issuer == wire_receipt.binding.owner
        ):
            if local.tensor_parallel_rank != 0:
                raise RuntimeError(
                    "decode coordinator receipt targets a noncanonical rank"
                )
            if wire_receipt.binding.owner != authenticated_issuer:
                raise RuntimeError(
                    "decode coordinator receipt asserts another rank binding"
                )
            serving.coordinator_receipt_received(
                wire_receipt,
                authenticated_issuer,
            )
            self.terminal_process_reactor.notify_coordinator_deadline_changed()
            return
        if wire_receipt.kind in (
            TerminalReceiptKind.REQUEST_READY,
            TerminalReceiptKind.FAILURE,
        ):
            if issuer_rank.tensor_parallel_rank != 0:
                raise RuntimeError("decode owner receipt did not come from rank zero")
            if wire_receipt.binding.owner != local.terminal_identity:
                raise RuntimeError("decode owner receipt targets another process")
            serving.request_terminal_received(
                wire_receipt,
                authenticated_issuer,
            )
            return
        raise RuntimeError("decode terminal route carried another receipt kind")

    def receive_terminal_source_receipt(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Import one decode coordinator receipt into source serving.

        :param wire_receipt: Request-ready or failure authority.
        :param authenticated_issuer: Exact decode coordinator route identity.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        issuer_rank = self._terminal_rank_for_identity(authenticated_issuer)
        local = self._require_terminal_startup_peer_enrollment().binding.advertisement
        if (
            issuer_rank.role is not TerminalOwnerRole.DECODE
            or issuer_rank.tensor_parallel_rank != 0
            or local.role is not TerminalOwnerRole.SOURCE
        ):
            raise RuntimeError("source receipt route has another owner role")
        if wire_receipt.issuer != authenticated_issuer:
            raise RuntimeError("source receipt asserts another route issuer")
        if wire_receipt.binding.owner != local.terminal_identity:
            raise RuntimeError("source receipt targets another process")
        is_ready = (
            wire_receipt.kind is TerminalReceiptKind.REQUEST_READY
            and wire_receipt.outcome is TerminalReceiptOutcome.SUCCESS
        )
        is_failure = (
            wire_receipt.kind is TerminalReceiptKind.FAILURE
            and wire_receipt.outcome is TerminalReceiptOutcome.FAILURE
        )
        if not is_ready and not is_failure:
            raise RuntimeError("source route carried another receipt kind")
        importer = self._terminal_source_receipt_importers.get(authenticated_issuer)
        if importer is None:
            raise RuntimeError("source receipt issuer has no import namespace")
        local_receipt = importer.import_receipt(
            wire_receipt,
            authenticated_issuer,
        )
        serving = self.terminal_source_serving
        if is_ready:
            serving.request_ready(
                binding_digest=wire_receipt.binding.digest,
                wire_receipt=wire_receipt,
                local_receipt=local_receipt,
                authenticated_issuer=authenticated_issuer,
            )
        else:
            serving.request_failed(
                binding_digest=wire_receipt.binding.digest,
                wire_receipt=wire_receipt,
                local_receipt=local_receipt,
                authenticated_issuer=authenticated_issuer,
                reason="request-global coordination failed",
            )
        importer.retire_binding(wire_receipt.binding)

    def _authenticated_terminal_control_rank(
        self,
        agent_name: str,
        process_generation: str,
        role: TerminalOwnerRole,
    ) -> TerminalStartupRankAdvertisement:
        """Authenticate claimed packed-control identity through enrolled peers.

        :param agent_name: NIXL agent encoded by the packed control route.
        :param process_generation: Canonical process generation string.
        :param role: Required remote owner role.
        :returns: Exact enrolled startup rank.
        """

        generation = self._canonical_process_generation(process_generation)
        enrollment = self._require_terminal_startup_peer_enrollment()
        matches = tuple(
            rank
            for rank in enrollment.expected_remote_ranks
            if rank.role is role
            and rank.nixl_agent_name == agent_name
            and rank.process_generation == uuid.UUID(generation).bytes
        )
        if len(matches) != 1:
            raise RuntimeError("packed control identity is not an enrolled peer")
        rank = matches[0]
        if role is TerminalOwnerRole.DECODE:
            registration = enrollment.decoder_peers.get(rank.key)
            if (
                registration is None
                or registration.remote_handle is None
                or registration.process_generation != generation
                or registration.agent_name != agent_name
            ):
                raise RuntimeError("packed control decoder peer is not live")
            with self._prefill_peer_lock:
                if registration.remote_handle in self._quarantined_remote_handles:
                    raise RuntimeError("packed control decoder peer is quarantined")
        else:
            peer = enrollment.prefill_peers.get(rank.key)
            if (
                peer is None
                or peer.process_generation != generation
                or peer.agent_name != agent_name
            ):
                raise RuntimeError("packed control source peer is not live")
            with self._prefill_peer_lock:
                if peer.handle in self._quarantined_remote_handles:
                    raise RuntimeError("packed control source peer is quarantined")
        return rank

    def _dispatch_terminal_source_control(self, frames: list[bytes]) -> None:
        """Authenticate and dispatch one decoder-to-source terminal control.

        :param frames: Exact packed multipart control frames.
        """

        agent_name, generation, message = decode_packed_control_frames(frames)
        rank = self._authenticated_terminal_control_rank(
            agent_name,
            generation,
            TerminalOwnerRole.DECODE,
        )
        enrollment = self._require_terminal_startup_peer_enrollment()
        registration = enrollment.decoder_peers[rank.key]
        peer = PackedPeerIdentity(
            agent_name=registration.agent_name,
            agent_generation=rank.process_generation,
        )
        actor = self._packed_prefill_runtime
        if actor is None:
            raise RuntimeError("packed source actor is unavailable")
        if type(message) is PackedReady:
            actor.deliver_terminal_owner_ready(peer, message)
            return
        if type(message) is PackedRequestTeardown:
            binding = actor.deliver_terminal_owner_teardown(peer, message)
            if binding is not None:
                self.terminal_source_serving.wiring.teardown_received(
                    binding.digest,
                    rank.terminal_identity,
                )
            return
        if type(message) is PackedTerminalReceipt:
            wire_receipt = TerminalWireReceipt.decode(message.receipt_payload)
            if wire_receipt.binding.request_key != message.key:
                raise RuntimeError("source terminal receipt key differs from framing")
            self.receive_terminal_source_receipt(
                wire_receipt,
                rank.terminal_identity,
            )
            return
        raise RuntimeError("source terminal receiver rejected another control type")

    def _dispatch_terminal_decode_control(self, frames: list[bytes]) -> None:
        """Authenticate and dispatch source or same-service decode control.

        :param frames: Exact packed multipart control frames.
        """

        agent_name, generation, message = decode_packed_control_frames(frames)
        routes = self._terminal_decode_control_routes
        if type(message) is PackedTerminalReceipt and routes is not None:
            generation_bytes = uuid.UUID(
                self._canonical_process_generation(generation)
            ).bytes
            decoder_matches = tuple(
                route
                for route in routes.routes
                if route.startup_rank.nixl_agent_name == agent_name
                and route.startup_rank.process_generation == generation_bytes
            )
            if len(decoder_matches) == 1:
                wire_receipt = TerminalWireReceipt.decode(message.receipt_payload)
                if wire_receipt.binding.request_key != message.key:
                    raise RuntimeError(
                        "decode terminal receipt key differs from framing"
                    )
                self.receive_terminal_decode_receipt(
                    wire_receipt,
                    decoder_matches[0].identity,
                )
                return
            if len(decoder_matches) > 1:
                raise RuntimeError("decode control route identity is ambiguous")

        rank = self._authenticated_terminal_control_rank(
            agent_name,
            generation,
            TerminalOwnerRole.SOURCE,
        )
        enrollment = self._require_terminal_startup_peer_enrollment()
        peer = enrollment.prefill_peers[rank.key]
        writer_id = StagingWriterId(
            transfer_source_rank=peer.transfer_source_rank,
            source_attn_tp_rank=peer.attn_tp_rank,
            source_pp_rank=peer.pp_rank,
            source_cp_rank=peer.attn_cp_rank,
        )
        if type(message) is PackedTerminalReceipt:
            wire_receipt = TerminalWireReceipt.decode(message.receipt_payload)
            if wire_receipt.binding.request_key != message.key:
                raise RuntimeError("decode terminal receipt key differs from framing")
            self.receive_terminal_decode_receipt(
                wire_receipt,
                rank.terminal_identity,
            )
            return
        self.terminal_decode_serving.control_received(writer_id, message)

    def _handle_terminal_source_metadata(self, frames: list[bytes]) -> None:
        """Retain exact terminal request metadata without a transfer worker.

        :param frames: Existing guarded request metadata frames.
        """

        if len(frames) < 2 or frames[0] != GUARD:
            raise RuntimeError("terminal source metadata framing is invalid")
        payload = frames[1:]
        if payload[0] == b"None":
            raise RuntimeError("terminal source peer registration is already frozen")
        room = int(payload[0].decode("ascii"))
        transfer_info = TransferInfo.from_zmq(payload)
        if transfer_info.room != room:
            raise RuntimeError("terminal source metadata room changed in transit")
        if transfer_info.packed_plan is None:
            raise RuntimeError("terminal source metadata omitted its packed plan")
        self.terminal_source_identity_plan(transfer_info)
        enrollment = self._require_terminal_startup_peer_enrollment()
        rank = self._authenticated_terminal_control_rank(
            transfer_info.agent_name,
            transfer_info.process_generation,
            TerminalOwnerRole.DECODE,
        )
        registration = enrollment.decoder_peers[rank.key]
        if (
            transfer_info.endpoint != registration.endpoint
            or transfer_info.dst_port != registration.dst_port
        ):
            raise RuntimeError("terminal source metadata changed its control route")
        room_infos = self.transfer_infos.setdefault(room, {})
        if transfer_info.agent_name in room_infos:
            raise RuntimeError("terminal source metadata was delivered twice")
        room_infos[transfer_info.agent_name] = transfer_info
        if len(room_infos) > transfer_info.required_dst_info_num:
            raise RuntimeError("terminal source metadata exceeds its destination count")
        if len(room_infos) != transfer_info.required_dst_info_num:
            return
        self.resolve_kv_replica_factor(room_infos)
        self.req_to_decode_prefix_len[room] = transfer_info.decode_prefix_len or 0
        self.update_status(room, KVPoll.WaitingForInput)

    def _start_terminal_control_receiver(self) -> None:
        """Start the sole blocking terminal runtime socket owner.

        The receiver blocks on the transport socket and a dedicated pipe. The
        pipe is an explicit teardown wake, not a completion cadence, so normal
        shutdown never depends on closing a ZeroMQ socket from another thread.
        """

        with self._terminal_control_lock:
            if self._terminal_control_thread is not None:
                raise RuntimeError("terminal control receiver is already started")
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            os.set_blocking(write_fd, False)
            self._terminal_control_read_fd = read_fd
            self._terminal_control_write_fd = write_fd
            self._terminal_control_stop_requested = False
            self._terminal_control_ready.clear()
        binding = self._require_terminal_startup_peer_enrollment().binding

        def receive() -> None:
            """Own the manager PULL socket for the process lifetime."""

            poller = zmq.Poller()
            try:
                poller.register(self.server_socket, zmq.POLLIN)
                poller.register(read_fd, zmq.POLLIN)
                self._terminal_control_ready.set()
                while True:
                    events = dict(poller.poll())
                    if read_fd in events:
                        os.read(read_fd, 4096)
                        with self._terminal_control_lock:
                            stopping = self._terminal_control_stop_requested
                        if not stopping:
                            raise RuntimeError(
                                "terminal control receiver woke without a stop request"
                            )
                        return
                    if self.server_socket not in events:
                        continue
                    frames = list(self.server_socket.recv_multipart())
                    if len(frames) == 0:
                        logger.warning("Rejected empty non-terminal runtime traffic")
                        continue
                    if frames[0] == TERMINAL_SOURCE_PUBLICATION_RECEIPT_TAG:
                        if binding.advertisement.role is not TerminalOwnerRole.SOURCE:
                            raise RuntimeError(
                                "decode control route received source publication"
                            )
                        self._handle_terminal_source_publication_receipt(tuple(frames))
                        continue
                    if frames[0] == PACKED_CONTROL_TAG:
                        if binding.advertisement.role is TerminalOwnerRole.SOURCE:
                            self._dispatch_terminal_source_control(frames)
                        else:
                            self._dispatch_terminal_decode_control(frames)
                        continue
                    if (
                        binding.advertisement.role is TerminalOwnerRole.SOURCE
                        and frames[0] == GUARD
                    ):
                        if len(frames) >= 2 and frames[1] == b"None":
                            logger.warning(
                                "Rejected late terminal peer-registration traffic"
                            )
                            continue
                        self._handle_terminal_source_metadata(frames)
                        continue
                    logger.warning("Rejected structurally unrelated runtime traffic")
            except BaseException:  # noqa: BLE001
                self._record_terminal_component_failure(
                    "terminal control receiver died",
                    traceback.format_exc(),
                )
            finally:
                self._terminal_control_ready.set()
                try:
                    poller.unregister(self.server_socket)
                except KeyError:
                    pass
                try:
                    poller.unregister(read_fd)
                except KeyError:
                    pass

        thread = threading.Thread(
            target=receive,
            name="nixl-terminal-control",
            daemon=True,
        )
        with self._terminal_control_lock:
            self._terminal_control_thread = thread
        thread.start()
        timeout_seconds = self._terminal_startup_timeout_seconds()
        if not self._terminal_control_ready.wait(timeout_seconds):
            self._record_terminal_component_failure(
                "terminal control receiver did not start within its bound",
                None,
            )
            raise TimeoutError("terminal control receiver startup timed out")
        if not thread.is_alive():
            raise RuntimeError("terminal control receiver failed during startup")

    def stop_terminal_control_receiver(self, timeout_seconds: float) -> None:
        """Wake and join the terminal socket owner within one explicit bound.

        :param timeout_seconds: Positive receiver join bound.
        :raises TimeoutError: If the receiver does not stop within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        with self._terminal_control_lock:
            thread = self._terminal_control_thread
            write_fd = self._terminal_control_write_fd
            if thread is None:
                return
            if threading.current_thread() is thread:
                raise RuntimeError("terminal control receiver cannot join itself")
            should_wake = not self._terminal_control_stop_requested
            self._terminal_control_stop_requested = True
        if should_wake and write_fd is not None and thread.is_alive():
            try:
                os.write(write_fd, b"\x01")
            except BlockingIOError:
                pass
            except OSError:
                self._record_terminal_component_failure(
                    "terminal control receiver stop wake failed",
                    traceback.format_exc(),
                )
                raise
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            self._record_terminal_component_failure(
                "terminal control receiver did not stop within its bound",
                None,
            )
            raise TimeoutError("terminal control receiver stop timed out")
        self._close_terminal_control_fds()

    def _close_terminal_control_fds(self) -> None:
        """Close teardown-only receiver descriptors exactly once."""

        with self._terminal_control_lock:
            descriptors = (
                self._terminal_control_read_fd,
                self._terminal_control_write_fd,
            )
            self._terminal_control_read_fd = None
            self._terminal_control_write_fd = None
        close_traceback: str | None = None
        for descriptor in descriptors:
            if descriptor is None:
                continue
            try:
                os.close(descriptor)
            except OSError:
                close_traceback = traceback.format_exc()
        if close_traceback is not None:
            self._record_terminal_component_failure(
                "terminal control receiver descriptor close failed",
                close_traceback,
            )
            raise OSError("terminal control receiver descriptor close failed")

    def close_terminal_runtime(self, *, process_fatal: bool) -> None:
        """Close every terminal owner in dependency-reverse order.

        Admission closes before the control receiver, then native producers
        retire while the reactor can still drain their final actions. The
        reactor closes before serving descriptors, and the canonical gateway
        publisher drains last because serving may still earn publication work
        during its terminal drain.

        :param process_fatal: Whether retained authority must be quarantined.
        """

        if type(process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")
        with self._terminal_runtime_close_lock:
            if self._terminal_runtime_closed:
                return
            if self._terminal_runtime_close_started:
                raise RuntimeError("terminal runtime close cannot retry")
            self._terminal_runtime_close_started = True

        reactor = self._terminal_process_reactor
        serving: PackedTerminalSourceServing | PackedTerminalDecodeServing | None = (
            self._terminal_source_serving
        )
        if serving is None:
            serving = self._terminal_decode_serving
        publisher = self._terminal_output_publisher
        if reactor is None or serving is None:
            raise RuntimeError("terminal runtime composition is incomplete")
        timeout_seconds = terminal_deadline_spec(
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
        ).seconds
        first_error: Exception | None = None

        def retain_failure(
            reason: str,
            error: Exception,
            formatted_traceback: str,
        ) -> None:
            """Retain the first teardown error without skipping later owners."""

            nonlocal first_error
            if first_error is None:
                first_error = error
            self._record_terminal_component_failure(reason, formatted_traceback)

        try:
            reactor.stop_admission()
        except Exception as error:  # noqa: BLE001
            retain_failure(
                "terminal reactor admission stop failed",
                error,
                traceback.format_exc(),
            )
        try:
            self.stop_terminal_control_receiver(timeout_seconds)
        except Exception as error:  # noqa: BLE001
            retain_failure(
                "terminal control receiver close failed",
                error,
                traceback.format_exc(),
            )
        try:
            serving.stop_admission_and_retire_producers()
        except Exception as error:  # noqa: BLE001
            retain_failure(
                "terminal native producer retirement failed",
                error,
                traceback.format_exc(),
            )
        try:
            reactor.close(timeout_seconds)
        except Exception as error:  # noqa: BLE001
            retain_failure(
                "terminal process reactor close failed",
                error,
                traceback.format_exc(),
            )

        must_abort = (
            process_fatal
            or first_error is not None
            or self._terminal_process_fatal_reason is not None
        )
        try:
            if must_abort:
                serving.abort_and_close()
            else:
                serving.close_clean(timeout_seconds)
        except Exception as error:  # noqa: BLE001
            retain_failure(
                "terminal serving close failed",
                error,
                traceback.format_exc(),
            )
        if self._terminal_decode_serving is not None:
            try:
                self._require_terminal_decode_dflash_teardown(
                    process_fatal=must_abort or first_error is not None,
                )
            except Exception as error:  # noqa: BLE001
                retain_failure(
                    "terminal decode DFlash row teardown failed",
                    error,
                    traceback.format_exc(),
                )
        if publisher is not None:
            try:
                if not publisher.stop_admission_and_join():
                    raise RuntimeError("terminal gateway publisher close timed out")
            except Exception as error:  # noqa: BLE001
                retain_failure(
                    "terminal gateway publisher close failed",
                    error,
                    traceback.format_exc(),
                )
        publication_control = self._terminal_source_publication_control
        if publication_control is not None and not must_abort:
            try:
                publication_control.close_clean()
            except Exception as error:  # noqa: BLE001
                retain_failure(
                    "terminal source publication control close failed",
                    error,
                    traceback.format_exc(),
                )

        if first_error is not None:
            raise RuntimeError("terminal runtime teardown failed") from first_error
        with self._terminal_runtime_close_lock:
            self._terminal_runtime_closed = True

    def _terminal_decode_dflash_row_counts(self) -> tuple[int, int]:
        """Return active and quarantined decoder boundary-row populations.

        :returns: Active and quarantined registered DFlash row counts.
        """

        pool = self._terminal_dflash_boundary_pool
        if pool is None:
            return 0, 0
        free_count, active_count, quarantined_count = pool.inventory()
        counts = (free_count, active_count, quarantined_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise RuntimeError("terminal DFlash row inventory is malformed")
        if sum(counts) != pool.row_capacity:
            raise RuntimeError("terminal DFlash row inventory violates conservation")
        return active_count, quarantined_count

    def _require_terminal_decode_dflash_teardown(
        self,
        *,
        process_fatal: bool,
    ) -> None:
        """Require every decode boundary row to have a terminal disposition.

        Clean close permits no retained rows. Fail-closed teardown may retain
        quarantined rows process-lifetime, but an active row means ownership
        never reached either release or quarantine and remains ambiguous.

        :param process_fatal: Whether quarantined retention is authoritative.
        """

        if type(process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")
        active_count, quarantined_count = self._terminal_decode_dflash_row_counts()
        if active_count != 0:
            raise RuntimeError("terminal decode close retains active DFlash rows")
        if not process_fatal and quarantined_count != 0:
            raise RuntimeError("clean terminal decode close retains DFlash quarantine")

    def activate_terminal_startup(self) -> None:
        """Commit immutable cross-role peers before exposing service readiness.

        :raises RuntimeError: If activation repeats or startup fails closed.
        """

        binding = self.terminal_startup_binding
        if binding is None:
            return
        with self._terminal_activation_lock:
            if self._terminal_activation_started:
                raise RuntimeError("terminal startup activation cannot repeat")
            self._terminal_activation_started = True

        deadline = time.monotonic() + self._terminal_startup_timeout_seconds()
        if binding.advertisement.role is TerminalOwnerRole.SOURCE:
            self._activate_terminal_source(deadline)
        else:
            self._activate_terminal_decoder(deadline)
        if not self.terminal_peer_enrollment_frozen:
            raise RuntimeError("terminal native peer roster is not frozen")
        try:
            if (
                binding.advertisement.role is TerminalOwnerRole.SOURCE
                and self._terminal_source_publication_control is None
            ):
                raise RuntimeError("terminal source publication routes are not frozen")
            self._compose_terminal_runtime()
        except Exception:  # noqa: BLE001
            self._record_terminal_component_failure(
                "terminal runtime composition failed",
                traceback.format_exc(),
            )
            raise
        self._terminal_runtime_activated.set()

    def _bootstrap_transport_registration(self) -> dict[str, object]:
        """Publish this prefill rank's exact native NIXL identity.

        :returns: Generation-bound bootstrap registration fields.
        """

        return {
            "transport_protocol": NIXL_BOOTSTRAP_PEER_PROTOCOL,
            "nixl_agent_name": self.agent.name,
            "nixl_agent_metadata": base64.b64encode(self.agent_metadata).decode(
                "ascii"
            ),
            "nixl_agent_metadata_sha256": hashlib.sha256(
                self.agent_metadata
            ).hexdigest(),
            "process_generation": self.process_generation,
            "transfer_source_rank": self.transfer_source_rank,
        }

    def kv_transfer_protocol(self) -> str | None:
        """Return the live packed transfer protocol for this process role.

        :returns: The closed packed protocol only after the role-specific
            runtime actor is initialized and ready.
        """

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if self._packed_prefill_runtime is None:
                return None
            return PACKED_KV_TRANSFER_PROTOCOL

        controller = self._packed_decode_controller
        if controller is None or not controller.ready:
            return None
        return PACKED_KV_TRANSFER_PROTOCOL

    def uses_terminal_source_publication(self) -> bool:
        """Return whether the activated manager owns terminal source progress.

        :returns: Whether this process is an activated terminal source rank.
        """

        binding = self.terminal_startup_binding
        return (
            self.disaggregation_mode is DisaggregationMode.PREFILL
            and binding is not None
            and binding.advertisement.role is TerminalOwnerRole.SOURCE
            and self._terminal_runtime_activated.is_set()
        )

    def terminal_source_is_canonical(self) -> bool:
        """Return whether this source rank owns gateway and DFlash publication.

        :returns: Whether the active terminal identity is source TP rank zero.
        """

        if not self.uses_terminal_source_publication():
            return False
        binding = self.terminal_startup_binding
        if binding is None:
            raise RuntimeError("terminal source binding disappeared")
        return binding.advertisement.tensor_parallel_rank == 0

    def lease_terminal_dflash_source(
        self,
        counters: PackedDFlashBoundaryCounters,
    ) -> DFlashBoundaryPrefillSource:
        """Lease canonical device storage for one terminal DFlash boundary.

        :param counters: Immutable scheduler-owned DFlash scalar counters.
        :returns: Active source row and its authenticated counters.
        """

        if type(counters) is not PackedDFlashBoundaryCounters:
            raise TypeError("counters must be PackedDFlashBoundaryCounters")
        if not self.terminal_source_is_canonical():
            raise RuntimeError("only canonical terminal source may lease DFlash")
        owner = self._terminal_dflash_source_owner
        if owner is None:
            raise RuntimeError("terminal DFlash source owner is unavailable")
        return DFlashBoundaryPrefillSource(
            lease=owner.lease_source_row(),
            counters=counters,
        )

    def cancel_unpublished_terminal_dflash_source(
        self,
        source: DFlashBoundaryPrefillSource,
    ) -> None:
        """Release one canonical row before native lifecycle publication.

        :param source: Exact unpublished DFlash source ownership.
        """

        if type(source) is not DFlashBoundaryPrefillSource:
            raise TypeError("source must be DFlashBoundaryPrefillSource")
        owner = self._terminal_dflash_source_owner
        if owner is None:
            raise RuntimeError("terminal DFlash source owner is unavailable")
        owner.cancel_unpublished_source_row(source.lease)

    def fail_terminal_source_process(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Stop source admission and enter native fail-closed drain.

        :param reason: Stable process-fatal failure boundary.
        :param formatted_traceback: Complete originating traceback, if any.
        """

        if self.disaggregation_mode is not DisaggregationMode.PREFILL:
            raise RuntimeError("only a terminal source process may fail this way")
        self._record_terminal_component_failure(reason, formatted_traceback)
        self.terminal_source_serving.begin_fail_closed_abort()

    def cancel_terminal_source_request(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> PackedTerminalSourceCancellationDisposition:
        """Record client intent while the published source lifecycle completes.

        :param binding: Exact scheduler-retained source generation.
        :param reason: Stable client-cancellation reason.
        :returns: Completion-required or too-late-for-rollback disposition.
        """

        try:
            return self.terminal_source_serving.cancel_submission(binding, reason)
        except Exception:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            self.fail_terminal_source_process(
                "terminal source cancellation recording failed",
                formatted_traceback,
            )
            raise

    def packed_terminal_health(self) -> dict[str, object] | None:
        """Return a JSON-native role-neutral terminal ownership projection.

        :returns: In-campaign health evidence, or ``None`` before activation.
        """

        source_serving = self._terminal_source_serving
        if source_serving is not None:
            inventory = source_serving.inventory()
            active = tuple(
                sorted(
                    set(inventory.wiring.active_binding_digests)
                    | set(inventory.scheduler_consumer.active_binding_digests)
                    | set(inventory.resources.actor_active_binding_digests)
                    | set(inventory.resources.request_ready_import_binding_digests)
                    | set(
                        inventory.resources.publication_control_active_binding_digests
                    )
                    | set(inventory.resources.unpublished_quarantined_binding_digests)
                    | set(inventory.runtime.quarantined_binding_digests)
                )
            )
            quarantined = tuple(
                sorted(
                    set(inventory.wiring.quarantined_binding_digests)
                    | set(inventory.resources.actor_quarantined_binding_digests)
                    | set(inventory.runtime.quarantined_binding_digests)
                    | set(inventory.scheduler_consumer.quarantined_binding_digests)
                    | set(inventory.resources.unpublished_quarantined_binding_digests)
                )
            )
            native = inventory.grouped_nixl.native
            native_problem: str | None = None
            if int(native.fatal) != 0 or native.eventfd_error != 0:
                native_problem = (
                    "grouped NIXL channel fatal="
                    f"{int(native.fatal)} eventfd_error={native.eventfd_error}"
                )
            fatal_reason = inventory.runtime.fatal_reason
            if fatal_reason is None:
                fatal_reason = native_problem
            elif native_problem is not None:
                fatal_reason = f"{fatal_reason}; {native_problem}"
            return {
                "role": "source",
                "active_count": len(active),
                "active_binding_digests": [digest.hex() for digest in active],
                "quarantine_count": len(quarantined),
                "quarantined_binding_digests": [digest.hex() for digest in quarantined],
                "retained_resource_count": inventory.retained_resource_count,
                "pending_owner_action_count": (
                    inventory.runtime.owner.pending_action_count
                ),
                "pending_runtime_action_count": (
                    inventory.runtime.consumer_pending_count
                ),
                "pending_scheduler_action_count": (
                    inventory.runtime.scheduler_pending_count
                ),
                "fatal_reason": fatal_reason,
                "owner_dead": inventory.owner_dead_marked,
                "output_reactor_alive": inventory.runtime.output_reactor_alive,
                "grouped_nixl_quarantined_transfer_count": (
                    inventory.grouped_nixl.quarantined_transfer_count
                ),
                "grouped_nixl_unowned_handle_count": (
                    inventory.grouped_nixl.unowned_handle_count
                ),
                "completion_required_count": len(
                    inventory.wiring.completion_required_binding_digests
                ),
                "active_result_slot_count": len(
                    inventory.wiring.active_result_slot_binding_digests
                ),
                "quarantined_result_slot_count": (
                    len(inventory.wiring.quarantined_result_slot_binding_digests)
                    + len(
                        inventory.resources.unpublished_quarantined_result_slot_binding_digests
                    )
                ),
                "dflash_active_transfer_count": (
                    inventory.resources.dflash_active_transfer_count
                ),
                "dflash_quarantined_transfer_count": (
                    inventory.resources.dflash_quarantined_transfer_count
                ),
                "dflash_active_row_count": (
                    inventory.resources.dflash_active_row_count
                ),
                "dflash_quarantined_row_count": (
                    inventory.resources.dflash_quarantined_row_count
                ),
                "dflash_unowned_native_handle_count": (
                    inventory.resources.dflash_unowned_native_handle_count
                ),
                "unpublished_quarantine_count": len(
                    inventory.resources.unpublished_quarantined_binding_digests
                ),
            }

        decode_serving = self._terminal_decode_serving
        if decode_serving is None:
            return None
        inventory = decode_serving.inventory()
        dflash_active_row_count, dflash_quarantined_row_count = (
            self._terminal_decode_dflash_row_counts()
        )
        active = tuple(
            sorted(
                set(inventory.active_binding_digests)
                | set(inventory.actor.active_bindings)
                | set(inventory.scheduler_consumer.active_binding_digests)
                | set(inventory.runtime.quarantined_binding_digests)
            )
        )
        quarantined = tuple(
            sorted(
                set(inventory.actor.quarantined_bindings)
                | set(inventory.runtime.quarantined_binding_digests)
                | set(inventory.scheduler_consumer.quarantined_binding_digests)
            )
        )
        return {
            "role": "decode",
            "active_count": len(active),
            "active_binding_digests": [digest.hex() for digest in active],
            "quarantine_count": len(quarantined),
            "quarantined_binding_digests": [digest.hex() for digest in quarantined],
            "retained_resource_count": (
                inventory.retained_resource_count
                + dflash_active_row_count
                + dflash_quarantined_row_count
            ),
            "pending_owner_action_count": inventory.runtime.owner.pending_action_count,
            "pending_runtime_action_count": inventory.runtime.consumer_pending_count,
            "pending_scheduler_action_count": (
                inventory.runtime.scheduler_pending_count
            ),
            "fatal_reason": inventory.runtime.fatal_reason,
            "owner_dead": inventory.owner_dead_marked,
            "output_reactor_alive": inventory.runtime.output_reactor_alive,
            "grouped_nixl_quarantined_transfer_count": 0,
            "grouped_nixl_unowned_handle_count": 0,
            "completion_required_count": 0,
            "active_result_slot_count": 0,
            "quarantined_result_slot_count": 0,
            # The boundary row rides the request's existing packed scatter. It
            # introduces no independent decode transfer or native-handle owner.
            "dflash_active_transfer_count": 0,
            "dflash_quarantined_transfer_count": 0,
            "dflash_active_row_count": dflash_active_row_count,
            "dflash_quarantined_row_count": dflash_quarantined_row_count,
            "dflash_unowned_native_handle_count": 0,
            "unpublished_quarantine_count": 0,
        }

    def enqueue_terminal_dflash_source_projection(
        self,
        source: DFlashBoundaryPrefillSource,
        boundary_token_id: torch.Tensor,
        gateway_result_slot: TerminalGatewayResultSlot,
        *,
        stream: torch.cuda.Stream,
        producer_event: torch.cuda.Event,
    ) -> None:
        """Stage the canonical boundary token and gateway result atomically.

        :param source: Active canonical source row.
        :param boundary_token_id: One device-resident sampled token.
        :param gateway_result_slot: Stable pinned gateway result row.
        :param stream: Exact model-producing CUDA stream.
        :param producer_event: Event recorded after both copies.
        """

        if type(source) is not DFlashBoundaryPrefillSource:
            raise TypeError("source must be DFlashBoundaryPrefillSource")
        owner = self._terminal_dflash_source_owner
        if owner is None:
            raise RuntimeError("terminal DFlash source owner is unavailable")
        owner.enqueue_source_projection(
            source.lease,
            boundary_token_id,
            gateway_result_slot,
            stream=stream,
            producer_event=producer_event,
        )

    def supports_packed_decode_request_transactions(self) -> bool:
        """Return whether every decode-side packed request actor is live.

        :returns: ``False`` until peer authentication, auxiliary metadata, and
            request teardown share one production lifecycle.
        """

        controller = self._packed_decode_controller
        return controller is not None and controller.ready

    def prepared_grant_protocol(self) -> str | None:
        """Return the live prefill prepared-grant protocol.

        :returns: ``None`` until the packed source actor is initialized.
        """

        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return None
        if self._packed_prefill_runtime is None:
            return None
        return PACKED_PREPARED_GRANT_PROTOCOL

    def attach_packed_decode_scheduler(
        self,
        metadata_allocator: object,
        consumer_authority: object,
    ) -> None:
        """Attach scheduler-owned metadata resources to the decode actor.

        :param metadata_allocator: Existing decode metadata-row allocator.
        :param consumer_authority: Scheduler queue consuming metadata contents.
        """

        controller = self._packed_decode_controller
        if controller is None:
            return
        controller.attach_scheduler(metadata_allocator, consumer_authority)

    def prepare_packed_decode_request_transaction(
        self,
        *,
        room_id: int,
        request_owner: object,
        metadata_buffer_index: int | None,
        allocation_lease: DecodeAllocationLease,
        allocation_authority: DecodeAllocationLeaseAuthority,
        lifecycle_authority: object,
        source_tp_size: int,
    ) -> PackedDecodeRequestTransaction | None:
        """Construct one production decode transaction when actors are live.

        :param room_id: Decoder-minted non-recycled bootstrap room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Legacy reserved auxiliary slot, or
            ``None`` for terminal DFlash registered VRAM.
        :param allocation_lease: Prepared decode allocation lease.
        :param allocation_authority: Exact allocation lease authority.
        :param lifecycle_authority: Trusted transport lifecycle authority.
        :param source_tp_size: Source attention tensor-parallel width.
        :returns: ``None`` while the production packed runtime is unavailable.
        """

        controller = self._require_packed_decode_controller()
        return controller.prepare_transaction(
            room_id=room_id,
            request_owner=request_owner,
            metadata_buffer_index=metadata_buffer_index,
            allocation_lease=allocation_lease,
            allocation_authority=allocation_authority,
            lifecycle_authority=lifecycle_authority,
            source_tp_size=source_tp_size,
        )

    def build_terminal_decode_request_authority(
        self,
        *,
        transaction: PackedDecodeRequestTransaction,
        adopt_request: Callable[[object], TerminalDFlashDecodeAdoption],
        finalize_request: Callable[[object], None],
        cancel_request: Callable[[object], None],
        quarantine_request: Callable[[object, str], None],
        destination_bindings: tuple[TerminalRequestBinding, ...] | None = None,
        publication_generation: bytes | None = None,
    ) -> PackedTerminalDecodeRequestAuthority:
        """Derive one terminal request graph from sealed manager authority.

        This operation is non-publishing. The caller must pass the result to
        :func:`register_packed_terminal_decode_request`, which binds scheduler,
        actor, and native lifecycle ownership before ``transaction.publish``.

        :param transaction: Exact prepared packed transaction.
        :param adopt_request: Scheduler request-adoption callback.
        :param finalize_request: Scheduler post-adoption finalization callback.
        :param cancel_request: Safe unpublished cancellation callback.
        :param quarantine_request: Ambiguous scheduler retention callback.
        :param destination_bindings: Cross-rank decode bindings for TP greater
            than one.
        :param publication_generation: Optional exact publication generation.
        :returns: Complete immutable prepublication authority.
        """

        binding = self.terminal_startup_binding
        if binding is None:
            raise RuntimeError("terminal startup binding is not configured")
        if binding.advertisement.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("terminal decode authority requires a decode manager")
        return build_packed_terminal_decode_request_authority(
            startup_binding=binding,
            transaction=transaction,
            adopt_request=adopt_request,
            finalize_request=finalize_request,
            cancel_request=cancel_request,
            quarantine_request=quarantine_request,
            destination_bindings=destination_bindings,
            publication_generation=publication_generation,
        )

    def send_packed_decode_request_metadata(
        self,
        *,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        receiver: CommonKVReceiver,
        page_indices: npt.NDArray[np.int32],
        metadata_buffer_index: int,
        state_indices: list[object] | None,
        decode_prefix_len: int,
    ) -> None:
        """Enter metadata into the production packed decode actor.

        :param transaction: Exact retained request transaction.
        :param publication: Matching irreversible publication.
        :param receiver: Exact retained decode receiver.
        :param page_indices: Complete destination main-KV page array.
        :param metadata_buffer_index: Reserved auxiliary metadata slot.
        :param state_indices: Complete destination state page arrays.
        :param decode_prefix_len: Decoder-reused prefix length.
        :raises RuntimeError: Until the production packed runtime is live.
        """

        controller = self._require_packed_decode_controller()
        if type(receiver) is not NixlKVReceiver:
            raise TypeError("packed NIXL metadata requires NixlKVReceiver")
        if publication.auxiliary_plan.metadata_buffer_index != metadata_buffer_index:
            raise RuntimeError("packed metadata index differs from publication")
        terminal_payload = publication.terminal_source_plan
        if self.terminal_startup_binding is not None and terminal_payload is None:
            raise RuntimeError(
                "terminal packed metadata has no encoded source authority"
            )
        if self.terminal_startup_binding is None and terminal_payload is not None:
            raise RuntimeError(
                "non-terminal packed metadata carries terminal source authority"
            )
        routes = receiver.build_packed_control_routes(controller)
        if terminal_payload is None:
            controller.bind_publication(transaction, publication, routes)
        else:
            # Terminal serving is the sole lifecycle publisher. Entering the
            # actor directly would expose writer control while the native owner
            # still considers the allocation unpublished.
            self.terminal_decode_serving.allocation_published(
                transaction,
                publication,
                routes,
            )
        receiver.send_metadata(
            page_indices,
            metadata_buffer_index,
            state_indices,
            decode_prefix_len=decode_prefix_len,
            packed_plan=publication.auxiliary_plan,
            terminal_source_plan_payload=terminal_payload,
        )

    def poll_packed_decode_request_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> KVPoll:
        """Advance one actor-owned request on the scheduler thread.

        :param transaction: Exact packed request transaction.
        :returns: Current packed transfer state.
        """

        return self._require_packed_decode_controller().poll(transaction)

    def cancel_unpublished_packed_decode_request_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        """Cancel one unpublished actor-owned request.

        :param transaction: Exact packed request transaction.
        :returns: Exact retained decode request owner.
        """

        return self._require_packed_decode_controller().cancel_unpublished(transaction)

    def complete_packed_decode_request_metadata_consumption(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Release consumed metadata and retire packed actor state.

        :param transaction: Exact committed packed request.
        """

        self._require_packed_decode_controller().complete_metadata_consumption(
            transaction
        )

    def quarantine_packed_decode_request_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Quarantine every resource retained by one packed request.

        :param transaction: Exact packed request transaction.
        :param reason: Stable failure reason.
        """

        self._require_packed_decode_controller().quarantine(transaction, reason)

    def _require_packed_decode_controller(self) -> PackedNixlDecodeController:
        """Return the ready packed decode controller.

        :returns: Exact process-lifetime decode controller.
        :raises RuntimeError: If the actor is absent or not scheduler attached.
        """

        controller = self._packed_decode_controller
        if controller is None or not controller.ready:
            raise RuntimeError("packed NIXL decode request runtime is unavailable")
        return controller

    def _packed_decode_registration_frames(self) -> tuple[bytes, ...]:
        """Serialize the ready controller's persistent registration.

        :returns: Empty legacy tail or the exact packed registration tail.
        """

        controller = self._packed_decode_controller
        if controller is None or not controller.ready:
            return ()
        protocol = controller.prepared_grant_protocol
        if protocol is None:
            raise RuntimeError("ready packed decode controller has no grant protocol")
        advertisement = controller.advertisement
        return (
            PACKED_KV_TRANSFER_PROTOCOL.encode("ascii"),
            protocol.encode("ascii"),
            str(advertisement.base_address).encode("ascii"),
            str(advertisement.total_size).encode("ascii"),
            advertisement.arena_generation,
            advertisement.visibility_policy_digest,
            advertisement.runtime_cohort_digest,
            str(advertisement.page_size).encode("ascii"),
        )

    def _decode_registration_frames(self) -> tuple[bytes, ...]:
        """Serialize this decoder's complete process-lifetime registration.

        :returns: Exact guarded multipart frames accepted by source managers.
        :raises RuntimeError: If a terminal decoder is not fully initialized.
        """

        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_args.kv_data_ptrs
        )
        packed_kv_data_mem_kinds = _pack_kv_mem_kinds(self.kv_args.kv_data_mem_kinds)
        packed_kv_item_lens = b"".join(
            struct.pack("Q", item_len) for item_len in self.kv_args.kv_item_lens
        )
        packed_kv_layer_ids = b"".join(
            struct.pack("I", layer_id) for layer_id in self.kv_args.kv_layer_ids
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_args.aux_data_ptrs
        )
        packed_state_data_ptrs = pack_int_lists(self.kv_args.state_data_ptrs or [], "Q")
        packed_state_item_lens = pack_int_lists(self.kv_args.state_item_lens or [], "I")
        packed_state_dim_per_tensor = pack_int_lists(
            self.kv_args.state_dim_per_tensor or [], "I"
        )
        packed_state_layer_ids = pack_int_lists(self.kv_args.state_layer_ids, "I")
        if self.enable_staging and self._staging_ctx.allocator is not None:
            allocator = self._staging_ctx.allocator
            packed_staging_base_ptr = struct.pack("Q", allocator.get_base_ptr())
            staging_total_size = str(allocator.get_total_size()).encode("ascii")
        else:
            packed_staging_base_ptr = b""
            staging_total_size = b""
        if len(self.kv_args.kv_item_lens) > 0:
            dst_kv_item_len = self.kv_args.kv_item_lens[0]
            dst_num_slots = self.kv_args.kv_data_lens[0] // dst_kv_item_len
        else:
            dst_kv_item_len = 0
            dst_num_slots = 0

        packed_registration_frames = self._packed_decode_registration_frames()
        if (
            self.terminal_startup_binding is not None
            and len(packed_registration_frames) == 0
        ):
            raise RuntimeError(
                "terminal decoder enrollment requires a ready packed controller"
            )
        return (
            GUARD,
            b"None",
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
            self.agent.name.encode("ascii"),
            self.agent_metadata,
            packed_kv_data_ptrs,
            packed_aux_data_ptrs,
            packed_state_data_ptrs,
            str(self.kv_args.gpu_id).encode("ascii"),
            str(self.attn_tp_size).encode("ascii"),
            str(self.kv_args.engine_rank).encode("ascii"),
            str(dst_kv_item_len).encode("ascii"),
            packed_state_item_lens,
            packed_state_dim_per_tensor,
            packed_staging_base_ptr,
            staging_total_size,
            str(dst_num_slots).encode("ascii"),
            packed_kv_data_mem_kinds,
            packed_kv_item_lens,
            packed_state_layer_ids,
            packed_kv_layer_ids,
            self.process_generation.encode("ascii"),
            *packed_registration_frames,
        )

    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
        """Validate NIXL state-transfer topology before resolving peer ranks.

        :param info: Parallel topology advertised by the prefill server.
        :raises RuntimeError: If asymmetric non-MLA SWA uses unsupported PP or CP.
        """

        state_types = self.kv_args.state_types
        uses_asymmetric_swa = (
            StateType.SWA in state_types
            and not self.is_mla_backend
            and not self.is_hybrid_mla_backend
            and self.attn_tp_size != info.attn_tp_size
        )
        unsupported_pipeline_or_context_parallelism = (
            self.pp_size != 1
            or info.pp_size != 1
            or self.attn_cp_size != 1
            or info.attn_cp_size != 1
        )
        if uses_asymmetric_swa and unsupported_pipeline_or_context_parallelism:
            raise RuntimeError(
                "NIXL asymmetric-TP SWA transfer requires PP=1 and CP=1 on "
                "both prefill and decode servers"
            )

        super()._resolve_rank_mapping(info)

    def _init_staging_prefill_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingContext,
        )

        self._staging_ctx = PrefillStagingContext()

    def _init_staging_decode_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingContext,
        )

        self._staging_ctx = DecodeStagingContext()
        self._init_staging_allocator()

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        gpu_id = self.kv_args.gpu_id
        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self._register_staging_memory(ptr, size, gpu_id),
            self.kv_args,
            count,
            self.server_args.chunked_prefill_size,
        )

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        gpu_id = self.kv_args.gpu_id
        registrations: list[object] = []

        def register_staging_memory(ptr: int, size: int) -> object:
            registration = self._register_staging_memory(ptr, size, gpu_id)
            registrations.append(registration)
            return registration

        self._staging_ctx.allocator = init_staging_allocator(
            register_staging_memory,
            self.kv_args,
        )
        if len(registrations) != 1:
            raise RuntimeError(
                "decode staging allocator must own exactly one NIXL registration"
            )
        self._decode_staging_registration = registrations[0]

    def _register_staging_memory(
        self,
        ptr: int,
        size: int,
        gpu_id: int,
    ) -> object:
        """Register and return one staging-memory descriptor owner.

        :param ptr: Base address of the staging allocation.
        :param size: Registered byte capacity.
        :param gpu_id: NIXL CUDA device identifier.
        :returns: Exact registration retained by the legacy staging owner.
        """

        addrs = [(ptr, size, gpu_id, "")]
        descs = self.agent.register_memory(addrs, "VRAM")
        if not descs:
            raise RuntimeError(
                f"NIXL memory registration failed for staging buffer "
                f"(ptr=0x{ptr:x}, size={size})"
            )
        return descs

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        # NOTE: matches mooncake behavior -- staging buffers are now
        # created in __init__ (per-worker), independent of the kv
        # tensors. This setter only stashes the tensor metadata used by
        # send_kvcache_staged().
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def _is_watermark_ready(
        self, agent_name: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, agent_name, alloc_round, alloc_end)

    def _start_decode_staging_thread(self):
        """Start the decode-side staging and packed-control receiver."""

        def decode_staging_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue
                if msg[0] == PACKED_CONTROL_TAG:
                    self._handle_packed_decode_control(msg)
                    continue
                logger.warning(
                    "decode_staging_thread: unexpected message tag %s",
                    msg[0][:20],
                )

        threading.Thread(target=decode_staging_thread, daemon=True).start()

    def _handle_packed_decode_control(self, frames: list[bytes]) -> None:
        """Authenticate and dispatch one prefill-to-decode control message.

        :param frames: Exact PACKED_V4 multipart frames.
        """

        try:
            agent_name, process_generation, _ = decode_packed_control_frames(frames)
            with self._prefill_peer_lock:
                peer = self._prefill_peers_by_agent_name.get(agent_name)
                if peer is None:
                    raise RuntimeError(
                        "packed control references an unknown prefill peer"
                    )
                if peer.process_generation != process_generation:
                    raise RuntimeError(
                        "packed control references a stale prefill generation"
                    )
                if peer.handle in self._quarantined_remote_handles:
                    raise RuntimeError(
                        "packed control references a quarantined prefill peer"
                    )
            authenticated_peer = PackedPeerIdentity(
                agent_name=peer.agent_name,
                agent_generation=uuid.UUID(peer.process_generation).bytes,
            )
            authenticated_writer = StagingWriterId(
                transfer_source_rank=peer.transfer_source_rank,
                source_attn_tp_rank=peer.attn_tp_rank,
                source_pp_rank=peer.pp_rank,
                source_cp_rank=peer.attn_cp_rank,
            )
            self._require_packed_decode_controller().handle_control_frames(
                frames,
                authenticated_peer,
                authenticated_writer,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "Rejected packed prefill control message:\n%s",
                traceback.format_exc(),
            )

    def _handle_staging_req(self, msg):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _prefetch_staging_reqs(self, room: int):
        """Send STAGING_REQ for all chunks before the prefill forward starts.

        Idempotent per room: the first call for a given room does the full
        fan-out (one STAGING_REQ per chunk per peer); subsequent calls return
        immediately. This lets the caller invoke this on every chunk without
        depending on a chunk_id == 0 sentinel.
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return
        if room in self._staging_ctx.prefetched_rooms:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy()
            and tinfo.agent_name in self.decode_kv_args_table
            and self.decode_kv_args_table[tinfo.agent_name].decode_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            # Mark anyway so we don't re-evaluate the predicate every chunk.
            self._staging_ctx.prefetched_rooms.add(room)
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            self.server_args.chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )
        self._staging_ctx.prefetched_rooms.add(room)

    def check_status(self, bootstrap_room: int):
        return self.request_status.get(bootstrap_room, KVPoll.WaitingForInput)

    def update_status(self, bootstrap_room: int, status: KVPoll):
        # Keep Failed sticky until the sender clears the room.
        if self.request_status.get(bootstrap_room) == KVPoll.Failed:
            return
        super().update_status(bootstrap_room, status)

    def _prep_equal_tp_dlist(
        self,
        peer_handle: nixl_remote_agent_handle | None,
        kv_ptrs: list[int],
        kv_item_lens: list[int],
        kv_data_lens: list[int],
        gpu_id: int,
        num_slots: Optional[int] = None,
        mem_kind: str = "VRAM",
        kv_xfer_lens: Optional[list[int]] = None,
    ):
        if kv_xfer_lens is None:
            kv_xfer_lens = kv_item_lens
        if not (
            len(kv_ptrs) == len(kv_item_lens) == len(kv_data_lens) == len(kv_xfer_lens)
        ):
            raise ValueError(
                "NIXL prepared dlist geometry length mismatch: "
                f"ptrs={len(kv_ptrs)}, item_lens={len(kv_item_lens)}, "
                f"data_lens={len(kv_data_lens)}, xfer_lens={len(kv_xfer_lens)}"
            )
        device_id = _nixl_device_id(mem_kind, gpu_id)
        arrays = []
        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set).
        # Convert once at entry; all downstream arithmetic stays in uint64.
        kv_ptrs_u64 = np.array(kv_ptrs, dtype=np.uint64)
        for base_ptr, item_len, data_len, xfer_len in zip(
            kv_ptrs_u64, kv_item_lens, kv_data_lens, kv_xfer_lens
        ):
            if xfer_len > item_len:
                raise ValueError(
                    "NIXL prepared dlist transfer length exceeds item stride: "
                    f"xfer_len={xfer_len}, item_len={item_len}, mem_kind={mem_kind}"
                )
            n = num_slots if num_slots is not None else (data_len // item_len)
            addrs = np.arange(n, dtype=np.uint64) * np.uint64(item_len) + base_ptr
            arrays.append(
                np.column_stack(
                    [
                        addrs,
                        np.full(n, xfer_len, dtype=np.uint64),
                        np.full(n, device_id, dtype=np.uint64),
                    ]
                )
            )

        peer = "" if peer_handle is None else peer_handle
        prep_handle = self.agent.prep_xfer_dlist(peer, np.vstack(arrays), mem_kind)
        assert prep_handle is not None, "prep_xfer_dlist returned None"
        return prep_handle

    def _init_equal_tp_prep_handle(
        self,
        peer_name: str,
        peer_handle: nixl_remote_agent_handle | None,
        kv_ptrs: list[int],
        gpu_id: int,
        num_slots: Optional[int] = None,
        mem_kind: str = "VRAM",
        kv_item_lens: Optional[list[int]] = None,
        kv_data_lens: Optional[list[int]] = None,
        kv_xfer_lens: Optional[list[int]] = None,
    ):
        """Pre-build NIXL dlist: all KV slots × all layers.

        peer_name="" = src side; agent name = dst side. num_slots overrides the local
        slot count — pass decode's count for the dst dlist (may differ from prefill).
        Uses prefill's kv_item_lens as stride; requires equal per-slot byte size (equal-TP or MLA).
        Source dlists use prefill geometry; destination dlists must use decode
        stride geometry but source transfer lengths, because HiSparse can transfer
        directly into a host pool whose slot stride differs from prefill.
        """
        if kv_item_lens is None:
            kv_item_lens = self.kv_args.kv_item_lens
        if kv_data_lens is None:
            kv_data_lens = self.kv_args.kv_data_lens
        self.prep_handles[peer_name] = self._prep_equal_tp_dlist(
            peer_handle,
            kv_ptrs,
            kv_item_lens,
            kv_data_lens,
            gpu_id,
            num_slots=num_slots,
            mem_kind=mem_kind,
            kv_xfer_lens=kv_xfer_lens,
        )

    def _init_hetero_tp_prep_handle(
        self,
        peer_name: str,
        decode_kv_args: KVArgsRegisterInfo,
        src_mem_kind: str = "VRAM",
        dst_mem_kind: str = "VRAM",
    ):
        """Pre-build NIXL dlists for TP-heterogeneous slice transfers.

        Src dlist shared across decode peers (same TP size). prefill_tp < decode_tp:
        interleave num_groups per token, peers select via head_group_idx.
        prefill_tp > decode_tp: num_groups=1. Dst dlist is per-peer.
        """
        decode_tp_size = decode_kv_args.decode_tp_size
        dst_kv_item_len = decode_kv_args.dst_kv_item_len
        prefill_tp_size = self.attn_tp_size

        page_size = self.kv_args.page_size

        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
        bytes_per_head_slice = dst_kv_item_len // page_size // dst_heads_per_rank

        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks feed one decode rank: each prefill rank sends
            # all its src heads to a specific head-range in the decode rank.
            src_replication = max(1, prefill_tp_size // total_kv_heads)
            local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
            num_groups = 1
            num_heads_to_send = src_heads_per_rank
            head_group_idx = 0
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
            dst_head_offset = dst_head_start * bytes_per_head_slice
        else:
            # One prefill rank feeds multiple decode ranks: interleave num_groups
            # head-groups in the src dlist so each decode rank picks its slice.
            dst_tp_rank_in_group = decode_kv_args.decode_tp_rank % decode_tp_size
            num_groups = decode_tp_size // prefill_tp_size
            num_heads_to_send = dst_heads_per_rank
            src_head_start = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            head_group_idx = src_head_start // dst_heads_per_rank
            dst_head_offset = 0

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice
        bytes_per_token_src = src_kv_item_len // page_size
        bytes_per_token_dst = dst_kv_item_len // page_size

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_pp = (
            self.get_mha_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs, decode_kv_args.dst_kv_ptrs
            )
        )
        src_ptrs = list(src_k_ptrs[:layers_pp]) + list(src_v_ptrs[:layers_pp])
        dst_ptrs = list(dst_k_ptrs[:layers_pp]) + list(dst_v_ptrs[:layers_pp])
        num_ptr_pairs = len(src_ptrs)

        num_slots = self.kv_args.kv_data_lens[0] // src_kv_item_len
        slots = np.arange(num_slots, dtype=np.uint64)
        tokens = np.arange(page_size, dtype=np.uint64)  # reused in dst dlist below
        groups = np.arange(num_groups, dtype=np.uint64)

        # Src dlist built once and shared.
        if self.prep_handle_slice_src is None:
            src_ptrs_arr = np.array(src_ptrs, dtype=np.uint64)
            addrs = (
                src_ptrs_arr[:, None, None, None]
                + slots[None, :, None, None] * np.uint64(src_kv_item_len)
                + tokens[None, None, :, None] * np.uint64(bytes_per_token_src)
                + groups[None, None, None, :] * np.uint64(bytes_per_token_to_send)
            ).ravel()
            src_array = np.column_stack(
                [
                    addrs,
                    np.full(len(addrs), bytes_per_token_to_send, dtype=np.uint64),
                    np.full(
                        len(addrs),
                        _nixl_device_id(src_mem_kind, self.kv_args.gpu_id),
                        dtype=np.uint64,
                    ),
                ]
            )
            src_handle = self.agent.prep_xfer_dlist("", src_array, src_mem_kind)
            assert (
                src_handle is not None
            ), f"prep_xfer_dlist returned None for slice src (decode_tp_size={decode_tp_size})"
            self.prep_handle_slice_src = (
                src_handle,
                num_groups,
                num_ptr_pairs,
                num_slots,
            )

        # Dst dlist per-peer; use decode's slot count (may exceed prefill's).
        num_slots_dst = (
            decode_kv_args.dst_num_slots
            if decode_kv_args.dst_num_slots is not None
            else num_slots
        )
        dst_slots = np.arange(num_slots_dst, dtype=np.uint64)
        # (ptr, slot, token) → ravel.
        dst_ptrs_arr = np.array(dst_ptrs, dtype=np.uint64)
        addrs = (
            dst_ptrs_arr[:, None, None]
            + dst_slots[None, :, None] * np.uint64(dst_kv_item_len)
            + tokens[None, None, :] * np.uint64(bytes_per_token_dst)
            + np.uint64(dst_head_offset)
        ).ravel()
        dst_array = np.column_stack(
            [
                addrs,
                np.full(len(addrs), bytes_per_token_to_send, dtype=np.uint64),
                np.full(
                    len(addrs),
                    _nixl_device_id(dst_mem_kind, decode_kv_args.gpu_id),
                    dtype=np.uint64,
                ),
            ]
        )
        if decode_kv_args.remote_handle is None:
            raise RuntimeError("Decoder NIXL peer is unavailable for slice preparation")
        dst_handle = self.agent.prep_xfer_dlist(
            decode_kv_args.remote_handle, dst_array, dst_mem_kind
        )
        assert (
            dst_handle is not None
        ), f"prep_xfer_dlist returned None for slice dst for peer '{peer_name}'"
        self.prep_handles_slice_dst[peer_name] = (
            dst_handle,
            num_slots_dst,
            head_group_idx,
        )

    def _init_mixed_equal_tp_prep_handles(
        self,
        peer_info: KVArgsRegisterInfo,
        mem_segments: List[_KVXferMemSegment],
    ):
        prepared_segments = []
        for seg in mem_segments:
            src_key = (seg.start, seg.end, seg.src_mem_kind)
            src_handle = self.prep_handles_segment_src.get(src_key)
            if src_handle is None:
                src_handle = self._prep_equal_tp_dlist(
                    None,
                    self.kv_args.kv_data_ptrs[seg.start : seg.end],
                    self.kv_args.kv_item_lens[seg.start : seg.end],
                    self.kv_args.kv_data_lens[seg.start : seg.end],
                    self.kv_args.gpu_id,
                    mem_kind=seg.src_mem_kind,
                )
                self.prep_handles_segment_src[src_key] = src_handle

            dst_num_slots = (
                peer_info.dst_num_slots
                if peer_info.dst_num_slots is not None
                else self._num_slots_src
            )
            dst_kv_item_lens = peer_info.dst_kv_item_lens[seg.start : seg.end]
            dst_kv_data_lens = [
                item_len * dst_num_slots for item_len in dst_kv_item_lens
            ]
            dst_handle = self._prep_equal_tp_dlist(
                peer_info.remote_handle,
                peer_info.dst_kv_ptrs[seg.start : seg.end],
                dst_kv_item_lens,
                dst_kv_data_lens,
                peer_info.gpu_id,
                num_slots=peer_info.dst_num_slots,
                mem_kind=seg.dst_mem_kind,
                kv_xfer_lens=self.kv_args.kv_item_lens[seg.start : seg.end],
            )
            prepared_segments.append(
                _KVXferPreparedSegment(
                    start=seg.start,
                    end=seg.end,
                    src_handle=src_handle,
                    dst_handle=dst_handle,
                    dst_num_slots=dst_num_slots,
                )
            )
        peer_info.kv_xfer_segments = prepared_segments

    def _prepare_payload_xfer(self, peer_info: KVArgsRegisterInfo):
        if peer_info.remote_handle is None:
            raise RuntimeError("Decoder NIXL peer is unavailable for transfer setup")
        # If prefill does not run speculative decoding (the usual case),
        # decode with speculative decoding will have more kv items.
        # Prefill having more kv items is impossible.
        n_src = len(self.kv_args.kv_item_lens)
        n_dst = len(peer_info.dst_kv_item_lens)
        if n_dst < n_src:
            raise ValueError(
                "NIXL PD transfer: decode registered fewer KV regions "
                f"({n_dst}) than prefill ({n_src}); unexpected geometry"
            )
        if n_src == 0:
            return
        assert self.src_mem_kind is not None
        src_mem_kind = self.src_mem_kind
        decode_only_spec_dec = n_dst > n_src
        if (
            self.is_mla_backend
            or self.is_hybrid_mla_backend
            or peer_info.decode_tp_size == self.attn_tp_size
        ):
            dst_mem_kind = None
            try:
                dst_mem_kind = _homogeneous_kv_mem_kind(
                    peer_info.dst_kv_mem_kinds, "destination"
                )
            except NotImplementedError:
                if decode_only_spec_dec:
                    raise NotImplementedError(
                        "NIXL PD transfer does not support HiSparse combined with "
                        "decode-only speculative decoding."
                    )
                mem_segments = _kv_xfer_mem_segments(
                    self.kv_args.kv_data_mem_kinds, peer_info.dst_kv_mem_kinds
                )
                if not mem_segments:
                    raise ValueError("NIXL KV transfer has no KV memory segments")
                self._init_mixed_equal_tp_prep_handles(peer_info, mem_segments)
                return

            if decode_only_spec_dec and dst_mem_kind != "VRAM":
                raise NotImplementedError(
                    "NIXL PD transfer does not support HiSparse combined with "
                    "decode-only speculative decoding."
                )

            peer_info.dst_homogeneous_mem_kind = dst_mem_kind
            # Build the shared src dlist on the first equal-TP/MLA peer; later
            # peers reuse it. Skipped entirely on heterogeneous-TP-only setups.
            if "" not in self.prep_handles:
                self._init_equal_tp_prep_handle(
                    "",
                    None,
                    self.kv_args.kv_data_ptrs,
                    self.kv_args.gpu_id,
                    mem_kind=src_mem_kind,
                )
            dst_num_slots = (
                peer_info.dst_num_slots
                if peer_info.dst_num_slots is not None
                else self._num_slots_src
            )

            pairs = build_transfer_entry_pairs(
                self.kv_args.kv_layer_ids,
                peer_info.dst_kv_layer_ids,
                n_src,
                n_dst,
                allow_positional_fallback=self.pp_size == 1,
            )
            dst_indices = [j for _, j in pairs]
            dst_kv_ptrs = [peer_info.dst_kv_ptrs[j] for j in dst_indices]
            dst_kv_item_lens = [peer_info.dst_kv_item_lens[j] for j in dst_indices]
            dst_kv_data_lens = [
                item_len * dst_num_slots for item_len in dst_kv_item_lens
            ]
            self._init_equal_tp_prep_handle(
                peer_info.agent_name,
                peer_info.remote_handle,
                dst_kv_ptrs,
                peer_info.gpu_id,
                num_slots=peer_info.dst_num_slots,
                mem_kind=dst_mem_kind,
                kv_item_lens=dst_kv_item_lens,
                kv_data_lens=dst_kv_data_lens,
                kv_xfer_lens=self.kv_args.kv_item_lens,
            )
        else:
            dst_mem_kind = _homogeneous_kv_mem_kind(
                peer_info.dst_kv_mem_kinds, "destination"
            )
            peer_info.dst_homogeneous_mem_kind = dst_mem_kind
            if dst_mem_kind != "VRAM":
                raise NotImplementedError(
                    "NIXL heterogeneous-TP direct-to-host KV transfer is not "
                    "implemented safely yet"
                )
            self._init_hetero_tp_prep_handle(
                peer_info.agent_name,
                peer_info,
                src_mem_kind=src_mem_kind,
                dst_mem_kind=dst_mem_kind,
            )

    def _notify_decode_transfer_failure(self, room: int) -> None:
        """Best-effort notify every decode peer that a room has failed.

        :param room: Bootstrap room whose transfer failed.
        """

        notification = f"{room}_failure_{self.transfer_source_rank}".encode("ascii")
        transfer_infos = self.transfer_infos.get(room, {})
        for transfer_info in transfer_infos.values():
            if transfer_info.is_dummy():
                continue
            try:
                self.agent.send_notif(
                    self._remote_decode_peer_handle(transfer_info.agent_name),
                    notification,
                )
            except Exception:
                logger.error(
                    "Failed to notify decode peer %s that room %s failed:\n%s",
                    transfer_info.agent_name,
                    room,
                    traceback.format_exc(),
                )

    def _is_canonical_aux_writer(self) -> bool:
        """Return whether this source owns the request auxiliary payload.

        :returns: Whether this rank is the canonical auxiliary writer.
        """

        return self.attn_tp_rank == 0 and self.attn_cp_rank == 0 and self.pp_rank == 0

    def _packed_source_route(
        self,
        room: int,
    ) -> tuple[TransferInfo, KVArgsRegisterInfo] | None:
        """Resolve one room's exact packed destination rank.

        :param room: Decoder-minted bootstrap room.
        :returns: Transfer metadata and registered decoder, or ``None`` for a
            legacy room.
        :raises RuntimeError: If packed and legacy metadata are mixed.
        """

        transfer_infos = tuple(self.transfer_infos[room].values())
        packed_infos = tuple(
            transfer_info
            for transfer_info in transfer_infos
            if transfer_info.packed_plan is not None
        )
        if len(packed_infos) == 0:
            return None
        if len(packed_infos) != len(transfer_infos):
            raise RuntimeError("room mixes packed and legacy destination metadata")
        if len(packed_infos) != 1:
            raise RuntimeError(
                "packed transfer requires one destination rank per source process"
            )
        transfer_info = packed_infos[0]
        registration = self.decode_kv_args_table.get(transfer_info.agent_name)
        if registration is None or registration.remote_handle is None:
            raise RuntimeError("packed transfer references an unloaded decoder")
        self._packed_destination_manifest(registration)
        if (
            registration.packed_transfer_protocol != PACKED_KV_TRANSFER_PROTOCOL
            or registration.prepared_grant_protocol != PACKED_PREPARED_GRANT_PROTOCOL
            or registration.packed_advertisement is None
        ):
            raise RuntimeError("decoder registration has no live packed capability")
        return transfer_info, registration

    def terminal_source_identity_plan(
        self,
        transfer_info: TransferInfo,
    ) -> PackedTerminalSourceIdentityPlan | None:
        """Validate and project decoder-authored terminal source authority.

        Terminal metadata is accepted only when its exact writer identities
        and request-ready issuer are members of this manager's sealed startup
        matrix. The returned projection selects the sole local writer without
        request-time discovery or a second control channel.

        :param transfer_info: Existing packed request metadata envelope.
        :returns: Rank-local terminal identity plan, otherwise ``None`` outside
            a terminal deployment.
        """

        if type(transfer_info) is not TransferInfo:
            raise TypeError("transfer_info must be TransferInfo")
        startup_binding = self.terminal_startup_binding
        source_plan = transfer_info.decode_terminal_source_plan()
        if startup_binding is None:
            if source_plan is not None:
                raise PackedTerminalRequestRegistrationError(
                    "non-terminal manager received terminal source authority"
                )
            return None
        if startup_binding.advertisement.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("terminal source authority requires a source manager")
        if source_plan is None:
            raise PackedTerminalRequestRegistrationError(
                "terminal packed metadata omitted source authority"
            )
        packed_plan = transfer_info.packed_plan
        if packed_plan is None:
            raise PackedTerminalRequestRegistrationError(
                "terminal source authority has no packed auxiliary plan"
            )
        require_source_plan_request_key(source_plan, packed_plan.key)
        if source_plan.writers[0].writer_id != packed_plan.canonical_writer_id:
            raise PackedTerminalRequestRegistrationError(
                "terminal publisher writer differs from packed auxiliary authority"
            )
        runtime = self._packed_prefill_runtime
        if runtime is None:
            raise RuntimeError("terminal source authority has no packed source actor")
        return project_packed_terminal_source_authority(
            startup_binding=startup_binding,
            source_plan=source_plan,
            local_writer_id=runtime.writer_id,
            destination_process_generation=(packed_plan.destination_process_generation),
        )

    def _packed_destination_manifest(
        self,
        registration: KVArgsRegisterInfo,
    ) -> DecodeWriterManifest:
        """Validate one packed destination against this source-rank route.

        :param registration: Generation-bound decoder registration.
        :returns: Exact destination-rank-local writer manifest.
        """

        runtime = self._packed_prefill_runtime
        if runtime is None:
            raise RuntimeError("packed NIXL prefill runtime is unavailable")
        try:
            manifest = DecodeWriterManifest.for_tensor_parallel(
                self.attn_tp_size,
                registration.decode_tp_size,
                registration.decode_tp_rank,
            )
        except ValueError as error:
            raise RuntimeError(
                "packed decoder registration topology is invalid"
            ) from error
        if runtime.writer_id not in manifest.writers:
            raise RuntimeError(
                "packed decoder rank is not connected to this source process"
            )
        return manifest

    def _packed_source_components(
        self,
        source_main_pages: npt.NDArray[np.int32],
        transfer_info: TransferInfo,
        state_indices: Optional[List],
    ) -> tuple[PackedComponentPages, ...]:
        """Build source/destination page projections for one packed request.

        :param source_main_pages: Complete migration-owned source KV pages.
        :param transfer_info: Exact decoder-authored destination metadata.
        :param state_indices: Complete final source state-window page arrays.
        :returns: Main-KV and active SWA components.
        """

        components = [
            PackedComponentPages(
                component_id=MAIN_KV_COMPONENT,
                source_pages=source_main_pages,
                destination_pages=np.asarray(
                    transfer_info.dst_kv_indices,
                    dtype=np.int32,
                ),
                destination_index_offset=0,
            )
        ]
        if state_indices is None:
            return tuple(components)
        if len(state_indices) != len(self.kv_args.state_types):
            raise RuntimeError(
                "packed source state-index count differs from registration"
            )
        if len(transfer_info.dst_state_indices) != len(self.kv_args.state_types):
            raise RuntimeError(
                "packed destination state-index count differs from registration"
            )
        for state_index, state_type in enumerate(self.kv_args.state_types):
            if state_type is not StateType.SWA:
                continue
            source_state_pages = state_indices[state_index]
            destination_state_pages = transfer_info.dst_state_indices[state_index]
            if source_state_pages is None:
                source_state_pages = []
            if destination_state_pages is None:
                destination_state_pages = []
            source_pages = np.asarray(source_state_pages, dtype=np.int32)
            destination_pages = np.asarray(
                destination_state_pages,
                dtype=np.int32,
            )
            if len(source_pages) == 0 and len(destination_pages) == 0:
                continue
            if len(source_pages) != len(destination_pages):
                raise RuntimeError(
                    "packed source/destination SWA window page counts differ: "
                    f"{len(source_pages)} and {len(destination_pages)}"
                )
            # Generic metadata describes the whole live SWA window. Packed
            # writers own only the tail overlapping the main migration range;
            # the preceding pages remain decode-local cache ownership.
            migration_page_count = min(len(source_main_pages), len(source_pages))
            source_pages = source_pages[-migration_page_count:]
            destination_pages = destination_pages[-migration_page_count:]
            components.append(
                PackedComponentPages(
                    component_id=StagingComponentId(
                        state_index=state_index,
                        state_type=state_type,
                    ),
                    source_pages=source_pages,
                    destination_pages=destination_pages,
                    destination_index_offset=0,
                )
            )
        return tuple(components)

    def _build_packed_source_launch_plan(
        self,
        *,
        transfer_info: TransferInfo,
        registration: KVArgsRegisterInfo,
        source_main_pages: npt.NDArray[np.int32],
        auxiliary_source: PackedPrefillAuxiliarySource,
        state_indices: Optional[List],
    ) -> PackedPrefillLaunchPlan:
        """Freeze one complete packed source request before model submission.

        :param transfer_info: Exact request metadata from the decoder.
        :param registration: Generation-bound decoder registration.
        :param source_main_pages: Complete source KV page projection.
        :param auxiliary_source: Explicit rank-local auxiliary ownership.
        :param state_indices: Complete final source state page arrays.
        :returns: Immutable route, pages, control, and auxiliary ownership.
        """

        runtime = self._packed_prefill_runtime
        plan = transfer_info.packed_plan
        advertisement = registration.packed_advertisement
        remote_handle = registration.remote_handle
        if runtime is None or plan is None or advertisement is None:
            raise RuntimeError("packed source runtime ownership is incomplete")
        if remote_handle is None:
            raise RuntimeError("packed source decoder handle is unavailable")
        decode_peer = PackedPeerIdentity(
            agent_name=registration.agent_name,
            agent_generation=uuid.UUID(registration.process_generation).bytes,
        )

        def send_message(message: PackedWireMessage) -> None:
            frames = encode_packed_control_frames(
                self.agent.name,
                self.process_generation,
                message,
            )
            self._send_packed_control_frames(
                registration.endpoint,
                registration.dst_port,
                frames,
            )

        control = PackedDecodeControlSender(
            peer=decode_peer,
            remote_handle=remote_handle,
            send_message=send_message,
        )
        destination = runtime.build_destination_capability(
            advertisement=advertisement,
            decode_peer=decode_peer,
            destination_gpu_id=registration.gpu_id,
            destination_tp_size=registration.decode_tp_size,
            destination_tp_rank=registration.decode_tp_rank,
            request_generation=plan.key.request_generation,
        )
        destination_registration = PackedDestinationRegistration(
            main_item_lens=tuple(registration.dst_kv_item_lens),
            main_layer_ids=tuple(registration.dst_kv_layer_ids),
            state_item_lens=tuple(
                tuple(item_lens) for item_lens in registration.dst_state_item_lens
            ),
            state_layer_ids=tuple(
                tuple(layer_ids) for layer_ids in registration.dst_state_layer_ids
            ),
            page_size=advertisement.page_size,
        )
        return PackedPrefillLaunchPlan(
            plan=plan,
            destination=destination,
            destination_registration=destination_registration,
            control=control,
            components=self._packed_source_components(
                source_main_pages,
                transfer_info,
                state_indices,
            ),
            auxiliary_source=auxiliary_source,
        )

    def build_terminal_source_launch_plan(
        self,
        *,
        room: int,
        source_main_pages: npt.NDArray[np.int32],
        state_indices: Optional[List],
        dflash_source: DFlashBoundaryPrefillSource | None,
    ) -> tuple[PackedTerminalSourceIdentityPlan, PackedPrefillLaunchPlan]:
        """Freeze terminal identity and transport ownership before launch.

        :param room: Exact decoder-minted bootstrap room.
        :param source_main_pages: Complete migration-owned source KV pages.
        :param state_indices: Complete final source state-window pages.
        :param dflash_source: Canonical-rank device row and frozen counters.
        :returns: Exact terminal identity and immutable pre-launch transport.
        """

        if type(room) is not int or room < 0:
            raise ValueError("terminal source room must be non-negative")
        route = self._packed_source_route(room)
        if route is None:
            raise RuntimeError("terminal source requires a packed destination route")
        transfer_info, registration = route
        identity = self.terminal_source_identity_plan(transfer_info)
        if identity is None:
            raise RuntimeError("terminal source identity is unavailable")
        auxiliary_source: PackedPrefillAuxiliarySource
        if self._is_canonical_aux_writer():
            if type(dflash_source) is not DFlashBoundaryPrefillSource:
                raise TypeError(
                    "canonical terminal source requires a DFlash device row"
                )
            auxiliary_source = PackedTerminalDFlashAuxiliarySource(dflash_source)
        else:
            if dflash_source is not None:
                raise ValueError(
                    "noncanonical terminal source cannot own DFlash transport"
                )
            auxiliary_source = PackedNoncanonicalAuxiliarySource()
        return identity, self._build_packed_source_launch_plan(
            transfer_info=transfer_info,
            registration=registration,
            source_main_pages=source_main_pages,
            auxiliary_source=auxiliary_source,
            state_indices=state_indices,
        )

    def _execute_packed_source_request(
        self,
        *,
        transfer_info: TransferInfo,
        registration: KVArgsRegisterInfo,
        source_main_pages: npt.NDArray[np.int32],
        auxiliary_source_index: int,
        state_indices: Optional[List],
        producer_event: torch.cuda.Event,
    ) -> None:
        """Execute one legacy packed request through the blocking actor.

        :param transfer_info: Exact request metadata from the decoder.
        :param registration: Generation-bound decoder registration.
        :param source_main_pages: Complete source KV page projection.
        :param auxiliary_source_index: Source auxiliary metadata row.
        :param state_indices: Complete final source state page arrays.
        :param producer_event: Event recorded after exact source cache writes.
        """

        runtime = self._packed_prefill_runtime
        if runtime is None:
            raise RuntimeError("packed source runtime ownership is incomplete")
        auxiliary_source: PackedPrefillAuxiliarySource
        if self._is_canonical_aux_writer():
            auxiliary_source = PackedLegacyAuxiliarySource(
                row_index=auxiliary_source_index
            )
        else:
            auxiliary_source = PackedNoncanonicalAuxiliarySource()
        launch_plan = self._build_packed_source_launch_plan(
            transfer_info=transfer_info,
            registration=registration,
            source_main_pages=source_main_pages,
            auxiliary_source=auxiliary_source,
            state_indices=state_indices,
        )
        runtime.execute(launch_plan.bind_producer_event(producer_event))

    def _retire_successful_source_room(self, room: int) -> None:
        """Drop source-side request metadata after terminal submission.

        :param room: Exact successful bootstrap room.
        """

        self.transfer_infos.pop(room, None)
        self.req_to_decode_prefix_len.pop(room, None)
        if not self.enable_staging or self._staging_ctx is None:
            return
        self._staging_ctx.prefetched_rooms.discard(room)
        for key in list(self._staging_ctx.prefetch_requested):
            if key[0] == room:
                self._staging_ctx.prefetch_requested.discard(key)

    def _wait_and_release_transfer_handles(
        self,
        handles: list[Any],
        context: str,
    ) -> None:
        """Wait for a transfer phase and release every completed handle.

        :param handles: Posted NIXL handles in the current transfer phase.
        :param context: Transfer phase used in failure diagnostics.
        :raises RuntimeError: If any handle reaches the NIXL error state.
        """

        while len(handles) > 0:
            all_done = True
            for handle in handles:
                state = self.agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError(f"NIXL transfer encountered ERR {context}")
                if state != "DONE":
                    all_done = False
            if all_done:
                break
            time.sleep(0)

        for handle in handles:
            self.agent.release_xfer_handle(handle)
        handles.clear()

    def _send_kvcache_slice_and_wait(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
        room: int,
    ) -> None:
        """Post and complete one direct asymmetric KV transfer.

        The caller retains the rank transaction lock so state and auxiliary
        transfers cannot overlap another worker's direct KV submission.

        :param peer_name: Registered decoder NIXL agent name.
        :param prefill_kv_indices: Source KV page indices.
        :param dst_kv_indices: Destination KV page indices.
        :param notif: Notification delivered by the final transfer part.
        :param room: Bootstrap room used in transfer diagnostics.
        """

        handle = self.send_kvcache_slice(
            peer_name,
            prefill_kv_indices,
            dst_kv_indices,
            notif,
        )
        if handle is None:
            return
        handles = [handle]
        self._wait_and_release_transfer_handles(
            handles,
            f"room={room} phase=direct-asymmetric-kv",
        )

    def transfer_worker(self, queue: FastQueue, staging_buffer=None):
        # Per-worker staging strategy: lazy-created on first chunk so we
        # see kv_buffer_tensors (set by ModelRunner after engine init).
        # Never cache on self -- multiple workers would race the ring.
        staging_strategy = None
        packed_source_page_chunks: dict[int, list[npt.NDArray[np.int32]]] = {}

        while True:
            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            handles: List[Any] = []
            direct_transaction_locked: bool = False
            try:
                if self.check_status(room) == KVPoll.Failed:
                    continue

                assert room in self.transfer_infos

                # Lazily build a per-worker staging strategy bound to this
                # worker's private staging buffer (matches mooncake).
                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)

                self.update_status(room, KVPoll.Transferring)

                reqs_to_be_processed = list(self.transfer_infos[room].values())
                packed_route = self._packed_source_route(room)
                if packed_route is not None:
                    page_chunks = packed_source_page_chunks.setdefault(room, [])
                    page_chunks.append(
                        np.array(
                            kv_chunk.prefill_kv_indices,
                            dtype=np.int32,
                            order="C",
                            copy=True,
                        )
                    )
                    if not kv_chunk.is_last_chunk:
                        continue
                    if kv_chunk.prefill_aux_index is None:
                        raise RuntimeError("packed source request has no auxiliary row")
                    producer_event = kv_chunk.producer_event
                    if producer_event is None:
                        raise RuntimeError(
                            "packed source request has no producer event"
                        )
                    source_main_pages = np.concatenate(page_chunks)
                    self._execute_packed_source_request(
                        transfer_info=packed_route[0],
                        registration=packed_route[1],
                        source_main_pages=source_main_pages,
                        auxiliary_source_index=kv_chunk.prefill_aux_index,
                        state_indices=kv_chunk.state_indices,
                        producer_event=producer_event,
                    )
                    packed_source_page_chunks.pop(room, None)
                    self.update_status(room, KVPoll.Success)
                    self._retire_successful_source_room(room)
                    continue

                # Set when staging allocation/watermark is not yet ready and
                # the chunk has been re-enqueued. We then break out of the
                # per-req loop and `continue` the worker main loop without
                # touching room status -- the next pop will retry.
                staging_deferred = False

                for req in reqs_to_be_processed:
                    assert room == req.room
                    if req.is_dummy():
                        continue

                    assert req.agent_name in self.decode_kv_args_table
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    decode_tp_size = dst_info.decode_tp_size

                    # Skip KV RDMA transfer when there are no pages to send
                    # (e.g., decode-side radix cache matched the entire prefix).
                    # Aux data is still sent below when is_last_chunk=True.
                    if (
                        len(kv_chunk.prefill_kv_indices) > 0
                        and self.kv_args.kv_data_ptrs
                    ):
                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        src_prefill_kv_indices = kv_chunk.prefill_kv_indices

                        notif = (
                            f"{req.room}_kv_{kv_chunk.chunk_id}"
                            f"_{int(kv_chunk.is_last_chunk)}_{self.transfer_source_rank}"
                        )

                        # Decide which kv send path to use:
                        #   1. Staging (heterogeneous TP, both sides have
                        #      registered staging, watermark/alloc ready)
                        #   2. send_kvcache (MLA or homogeneous TP)
                        #   3. send_kvcache_slice (heterogeneous TP fallback,
                        #      or staging hard-failed for this chunk)
                        use_staging = (
                            self.enable_staging
                            and staging_strategy is not None
                            and not self.is_mla_backend
                            and not self.is_hybrid_mla_backend
                            and decode_tp_size != self.attn_tp_size
                            and dst_info.staging is not None
                        )

                        kv_xfer_handle = None
                        staging_completed = False
                        if use_staging:
                            staging_completed, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                src_prefill_kv_indices,
                                req,
                                dst_info,
                                queue,
                            )
                            if deferred:
                                # Chunk re-enqueued; stop processing remaining
                                # reqs for this chunk and let the worker loop
                                # pick it up again on the next pop.
                                staging_deferred = True
                                break

                        if not staging_completed:
                            if (
                                self.is_mla_backend
                                or self.is_hybrid_mla_backend
                                or decode_tp_size == self.attn_tp_size
                            ):
                                if dst_info.kv_xfer_segments is None:
                                    if dst_info.dst_homogeneous_mem_kind is None:
                                        raise RuntimeError(
                                            "Missing NIXL destination KV memory kind"
                                        )
                                    kv_xfer_handle = self.send_kvcache(
                                        req.agent_name,
                                        src_prefill_kv_indices,
                                        dst_info.dst_kv_ptrs,
                                        chunked_dst_kv_indice,
                                        dst_info.gpu_id,
                                        notif,
                                        dst_mem_kind=(
                                            dst_info.dst_homogeneous_mem_kind
                                        ),
                                    )
                                else:
                                    handles.extend(
                                        self.send_kvcache_mixed(
                                            req.agent_name,
                                            src_prefill_kv_indices,
                                            chunked_dst_kv_indice,
                                            notif,
                                        )
                                    )
                            else:
                                if not direct_transaction_locked:
                                    self._direct_kv_transfer_lock.acquire()
                                    direct_transaction_locked = True
                                self._send_kvcache_slice_and_wait(
                                    req.agent_name,
                                    src_prefill_kv_indices,
                                    chunked_dst_kv_indice,
                                    notif,
                                    room,
                                )

                        if kv_xfer_handle is not None:
                            handles.append(kv_xfer_handle)

                    if kv_chunk.is_last_chunk:
                        dst_info = self.decode_kv_args_table[req.agent_name]
                        if kv_chunk.state_indices:
                            state_xfer_handles = self.maybe_send_extra(
                                req.agent_name,
                                kv_chunk.state_indices,
                                dst_info.dst_state_data_ptrs,
                                req.dst_state_indices,
                                dst_info.gpu_id,
                                f"{req.room}_state_{self.transfer_source_rank}",
                                decode_tp_size,
                                decode_tp_rank=dst_info.decode_tp_rank,
                                dst_state_item_lens=dst_info.dst_state_item_lens,
                                dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                                dst_state_layer_ids=dst_info.dst_state_layer_ids,
                            )
                            handles.extend(
                                h for h in state_xfer_handles if h is not None
                            )

                        if kv_chunk.prefill_aux_index is None:
                            raise RuntimeError("Missing aux index for last chunk")
                        sent_no_kv = (
                            len(kv_chunk.prefill_kv_indices) == 0
                            or not self.kv_args.kv_data_ptrs
                        )
                        if self._is_canonical_aux_writer():
                            aux_notif = (
                                f"{req.room}_aux_nokv_{self.transfer_source_rank}"
                                if sent_no_kv
                                else f"{req.room}_aux"
                            )
                            aux_xfer_handle = self.send_aux(
                                req.agent_name,
                                kv_chunk.prefill_aux_index,
                                dst_info.dst_aux_ptrs,
                                req.dst_aux_index,
                                aux_notif,
                            )
                            handles.append(aux_xfer_handle)
                        elif sent_no_kv:
                            self.agent.send_notif(
                                self._remote_decode_peer_handle(req.agent_name),
                                f"{req.room}_aux_nokv_{self.transfer_source_rank}".encode(
                                    "ascii"
                                ),
                            )

                if staging_deferred:
                    # Chunk has been re-enqueued; do not advance status.
                    continue

                self._wait_and_release_transfer_handles(
                    handles,
                    f"room={room} phase=final",
                )
                if direct_transaction_locked:
                    self._direct_kv_transfer_lock.release()
                    direct_transaction_locked = False

                if kv_chunk.is_last_chunk:
                    self.update_status(room, KVPoll.Success)
                    self._retire_successful_source_room(room)
                else:
                    self.update_status(room, KVPoll.Transferring)
            except Exception as e:
                # Catch all exceptions to prevent silently killing this
                # worker thread, but still propagate via failure_exception().
                if isinstance(e, _NIXL_TRANSPORT_ERRORS):
                    logger.warning(f"NIXL transport error for room {room}: {e}")
                else:
                    logger.exception(
                        f"Unexpected transfer worker error for room {room}"
                    )
                self._notify_decode_transfer_failure(room)
                packed_source_page_chunks.pop(room, None)
                self.exceptions[room] = e
                self.record_failure(room, str(e))
                self.update_status(room, KVPoll.Failed)
            finally:
                if direct_transaction_locked:
                    self._direct_kv_transfer_lock.release()

    def register_buffer_to_engine(self):
        self.kv_descs = []
        kv_addrs_by_mem_kind = {"VRAM": [], "DRAM": []}
        for kv_data_ptr, kv_data_len, kv_mem_kind in zip(
            self.kv_args.kv_data_ptrs,
            self.kv_args.kv_data_lens,
            self.kv_args.kv_data_mem_kinds,
        ):
            kv_addrs_by_mem_kind[kv_mem_kind].append(
                (
                    kv_data_ptr,
                    kv_data_len,
                    _nixl_device_id(kv_mem_kind, self.kv_args.gpu_id),
                    "",
                )
            )
        for mem_kind in ("VRAM", "DRAM"):
            kv_addrs = kv_addrs_by_mem_kind[mem_kind]
            if not kv_addrs:
                continue
            kv_descs = self.agent.register_memory(kv_addrs, mem_kind)
            logger.debug(
                f"Register kv tensors, kind={mem_kind}, len(kv_addr)= {len(kv_addrs)}"
            )
            if not kv_descs:
                raise Exception(
                    f"NIXL memory registration failed for {mem_kind} kv tensors"
                )
            self.kv_descs.append(kv_descs)
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = None
        if len(aux_addrs) > 0:
            self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
            logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
            if not self.aux_descs:
                raise Exception("NIXL memory registration failed for aux tensors")

        state_addrs = []
        for comp_ptrs, comp_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            for state_data_ptr, state_data_len in zip(comp_ptrs, comp_lens):
                if state_data_ptr == 0 or state_data_len == 0:
                    continue
                state_addrs.append(
                    (state_data_ptr, state_data_len, self.kv_args.gpu_id, "")
                )
        if state_addrs:
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

    @staticmethod
    def _canonical_process_generation(value: object) -> str:
        """Validate one serialized process generation.

        :param value: Candidate generation value.
        :returns: Canonical UUID string.
        :raises RuntimeError: If the value is absent or noncanonical.
        """

        if type(value) is not str or len(value) == 0:
            raise RuntimeError("Missing NIXL peer process generation")
        try:
            generation = uuid.UUID(value)
        except ValueError as error:
            raise RuntimeError("Invalid NIXL peer process generation") from error
        if str(generation) != value:
            raise RuntimeError("NIXL peer process generation is not canonical")
        return value

    def _terminal_prefill_rank_for_route(
        self,
        bootstrap_info: dict[str, object],
    ) -> TerminalStartupRankAdvertisement:
        """Authenticate one complete source route against the startup matrix.

        :param bootstrap_info: Full source rank route from the bootstrap service.
        :returns: Exact matrix row proved by the route.
        :raises RuntimeError: If topology or native identity differs.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        if enrollment.binding.advertisement.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("only a terminal decoder enrolls prefill routes")
        if bootstrap_info.get("transport_protocol") != NIXL_BOOTSTRAP_PEER_PROTOCOL:
            raise RuntimeError("terminal prefill route has no native NIXL identity")
        if bootstrap_info.get("is_dummy", False) is not False:
            raise RuntimeError(
                "terminal startup enrollment does not admit dummy routes"
            )

        normalized_ranks: dict[str, int] = {}
        for field_name in (
            "attn_dp_rank",
            "attn_cp_rank",
            "attn_tp_rank",
            "pp_rank",
            "transfer_source_rank",
        ):
            try:
                normalized_ranks[field_name] = validate_serialized_rank(
                    bootstrap_info.get(field_name), field_name
                )
            except ValueError as error:
                raise RuntimeError(
                    f"terminal prefill route has invalid {field_name}"
                ) from error
        if (
            normalized_ranks["attn_dp_rank"] != 0
            or normalized_ranks["attn_cp_rank"] != 0
            or normalized_ranks["pp_rank"] != 0
            or normalized_ranks["transfer_source_rank"]
            != normalized_ranks["attn_tp_rank"]
        ):
            raise RuntimeError("terminal prefill route is not exact DP1/CP1/PP1 TP")

        try:
            agent_name = validate_nixl_agent_name(bootstrap_info.get("nixl_agent_name"))
            agent_metadata = decode_nixl_agent_metadata(
                bootstrap_info.get("nixl_agent_metadata")
            )
        except ValueError as error:
            raise RuntimeError("terminal prefill route identity is invalid") from error
        advertised_digest = bootstrap_info.get("nixl_agent_metadata_sha256")
        actual_digest = hashlib.sha256(agent_metadata).hexdigest()
        if advertised_digest != actual_digest:
            raise RuntimeError("terminal prefill route metadata digest mismatch")

        matches = tuple(
            rank
            for rank in enrollment.expected_remote_ranks
            if rank.nixl_agent_name == agent_name
        )
        if len(matches) != 1:
            raise RuntimeError("terminal prefill route is absent from sealed matrix")
        expected_rank = matches[0]
        return self._terminal_remote_rank(
            agent_name=agent_name,
            agent_metadata=agent_metadata,
            process_generation=self._canonical_process_generation(
                bootstrap_info.get("process_generation")
            ),
            role=TerminalOwnerRole.SOURCE,
            tensor_parallel_rank=normalized_ranks["attn_tp_rank"],
            tensor_parallel_size=expected_rank.tensor_parallel_size,
        )

    def enroll_terminal_prefill_routes(
        self,
        bootstrap_addr: str,
        bootstrap_infos: tuple[dict[str, object], ...],
    ) -> tuple[_NixlPrefillPeer, ...]:
        """Authenticate, retain, and freeze every source peer.

        All routes are authenticated before the first native handle is created.
        Startup activation publishes the decoder registration only after this
        complete local roster is immutable.

        :param bootstrap_addr: Exact source-owned bootstrap service address.
        :param bootstrap_infos: Canonical full-metadata source rank routes.
        :returns: Canonically ordered retained source peers.
        :raises RuntimeError: If membership is incomplete, stale, or already frozen.
        """

        if type(bootstrap_addr) is not str or len(bootstrap_addr) == 0:
            raise ValueError("bootstrap_addr must be nonempty")
        if type(bootstrap_infos) is not tuple or len(bootstrap_infos) == 0:
            raise ValueError("bootstrap_infos must be a nonempty tuple")
        if any(type(info) is not dict for info in bootstrap_infos):
            raise TypeError("bootstrap_infos must contain dictionaries")
        enrollment = self._require_terminal_startup_peer_enrollment()
        if enrollment.binding.advertisement.role is not TerminalOwnerRole.DECODE:
            raise RuntimeError("only a terminal decoder enrolls prefill routes")
        with enrollment.lock:
            if enrollment.frozen:
                raise RuntimeError("terminal native peer roster is frozen")
            if len(enrollment.prefill_peers) > 0:
                raise RuntimeError("terminal prefill routes were already enrolled")

        authenticated = tuple(
            (self._terminal_prefill_rank_for_route(info), info)
            for info in bootstrap_infos
        )
        authenticated_keys = tuple(rank.key for rank, _ in authenticated)
        expected_keys = tuple(rank.key for rank in enrollment.expected_remote_ranks)
        if authenticated_keys != expected_keys:
            raise RuntimeError(
                "terminal prefill routes differ from the canonical matrix roster"
            )

        peers = tuple(
            self._load_prefill_peer(bootstrap_addr, info) for _, info in authenticated
        )
        with enrollment.lock:
            if enrollment.frozen or len(enrollment.prefill_peers) > 0:
                raise RuntimeError("terminal prefill enrollment changed concurrently")
            enrollment.prefill_peers.update(
                (rank.key, peer)
                for (rank, _), peer in zip(authenticated, peers, strict=True)
            )
            enrollment.freeze_if_complete()
            if not enrollment.frozen:
                raise RuntimeError("complete terminal prefill roster did not freeze")
        return peers

    def _send_terminal_decoder_registration(
        self,
        bootstrap_info: dict[str, object],
        frames: tuple[bytes, ...],
    ) -> None:
        """Send one immutable decoder registration to one source rank.

        :param bootstrap_info: Already authenticated source rank route.
        :param frames: Complete guarded registration frames.
        """

        rank_ip = bootstrap_info.get("rank_ip")
        rank_port = bootstrap_info.get("rank_port")
        if type(rank_ip) is not str or len(rank_ip) == 0:
            raise RuntimeError("terminal prefill route has no rank address")
        if type(rank_port) is not int or not 1 <= rank_port <= 65535:
            raise RuntimeError("terminal prefill route has an invalid rank port")
        address = NetworkAddress(rank_ip, rank_port)
        with self._packed_control_send_lock:
            socket = self._connect(address.to_tcp(), is_ipv6=address.is_ipv6)
            socket.send_multipart(frames)

    def _resolve_terminal_prefill_peer(
        self,
        bootstrap_addr: str,
        bootstrap_info: dict[str, object],
    ) -> _NixlPrefillPeer:
        """Resolve one request route without creating any native authority.

        :param bootstrap_addr: Exact enrolled bootstrap service address.
        :param bootstrap_info: Full source route selected for this request.
        :returns: Previously retained immutable source peer.
        :raises RuntimeError: If enrollment is open or route identity drifts.
        """

        enrollment = self._require_terminal_startup_peer_enrollment()
        rank = self._terminal_prefill_rank_for_route(bootstrap_info)
        with enrollment.lock:
            if not enrollment.frozen:
                raise RuntimeError("terminal native peer roster is not frozen")
            peer = enrollment.prefill_peers.get(rank.key)
            if peer is None:
                raise RuntimeError("terminal prefill peer was not pre-enrolled")
            if peer.bootstrap_addr != bootstrap_addr:
                raise RuntimeError("terminal prefill route changed bootstrap authority")
            rank_ip = bootstrap_info.get("rank_ip")
            rank_port = bootstrap_info.get("rank_port")
            if (
                type(rank_ip) is not str
                or type(rank_port) is not int
                or peer.control_endpoint != NetworkAddress(rank_ip, rank_port)
            ):
                raise RuntimeError("terminal prefill control listener changed")
            if (
                peer.agent_name != rank.nixl_agent_name
                or peer.process_generation
                != str(uuid.UUID(bytes=rank.process_generation))
                or peer.metadata_sha256 != rank.nixl_agent_metadata_sha256.hex()
            ):
                raise RuntimeError("terminal prefill peer differs from frozen roster")
            if peer.handle in self._quarantined_remote_handles:
                raise RuntimeError("terminal prefill peer is quarantined")
            return peer

    def _load_prefill_peer(
        self, bootstrap_addr: str, bootstrap_info: dict[str, object]
    ) -> _NixlPrefillPeer:
        """Load and connect one exact prefill source writer.

        :param bootstrap_addr: Bootstrap service which owns the route.
        :param bootstrap_info: Generation-bound rank route response.
        :returns: Retained native peer record.
        :raises RuntimeError: If metadata is missing, stale, or conflicting.
        """

        if bootstrap_info.get("transport_protocol") != NIXL_BOOTSTRAP_PEER_PROTOCOL:
            raise RuntimeError("Prefill route does not advertise native NIXL identity")

        int_fields = (
            "attn_dp_rank",
            "attn_cp_rank",
            "attn_tp_rank",
            "pp_rank",
            "transfer_source_rank",
        )
        normalized_ints: dict[str, int] = {}
        for field_name in int_fields:
            try:
                value = validate_serialized_rank(
                    bootstrap_info.get(field_name), field_name
                )
            except ValueError as error:
                raise RuntimeError(f"Invalid prefill route {field_name}") from error
            normalized_ints[field_name] = value

        try:
            agent_name = validate_nixl_agent_name(bootstrap_info.get("nixl_agent_name"))
        except ValueError as error:
            raise RuntimeError("Invalid prefill NIXL agent name") from error
        encoded_metadata = bootstrap_info.get("nixl_agent_metadata")
        metadata_sha256 = bootstrap_info.get("nixl_agent_metadata_sha256")
        if type(metadata_sha256) is not str or len(metadata_sha256) == 0:
            raise RuntimeError("Missing prefill NIXL agent metadata digest")
        process_generation = self._canonical_process_generation(
            bootstrap_info.get("process_generation")
        )
        rank_ip = bootstrap_info.get("rank_ip")
        rank_port = bootstrap_info.get("rank_port")
        if type(rank_ip) is not str or len(rank_ip) == 0:
            raise RuntimeError("Missing prefill control listener host")
        if type(rank_port) is not int or not 1 <= rank_port <= 65535:
            raise RuntimeError("Invalid prefill control listener port")
        control_endpoint = NetworkAddress(rank_ip, rank_port)

        try:
            metadata = decode_nixl_agent_metadata(encoded_metadata)
        except ValueError as error:
            raise RuntimeError("Invalid prefill NIXL agent metadata") from error
        actual_digest = hashlib.sha256(metadata).hexdigest()
        if actual_digest != metadata_sha256:
            raise RuntimeError("Prefill NIXL agent metadata digest mismatch")

        return self._retain_prefill_peer(
            bootstrap_addr=bootstrap_addr,
            attn_dp_rank=normalized_ints["attn_dp_rank"],
            attn_cp_rank=normalized_ints["attn_cp_rank"],
            attn_tp_rank=normalized_ints["attn_tp_rank"],
            pp_rank=normalized_ints["pp_rank"],
            transfer_source_rank=normalized_ints["transfer_source_rank"],
            agent_name=agent_name,
            metadata=metadata,
            metadata_sha256=actual_digest,
            process_generation=process_generation,
            control_endpoint=control_endpoint,
        )

    def _retain_prefill_peer(
        self,
        *,
        bootstrap_addr: str,
        attn_dp_rank: int,
        attn_cp_rank: int,
        attn_tp_rank: int,
        pp_rank: int,
        transfer_source_rank: int,
        agent_name: str,
        metadata: bytes,
        metadata_sha256: str,
        process_generation: str,
        control_endpoint: NetworkAddress,
    ) -> _NixlPrefillPeer:
        """Atomically resolve or create one native prefill peer record.

        :param bootstrap_addr: Bootstrap service which owns the route.
        :param attn_dp_rank: Source attention data-parallel rank.
        :param attn_cp_rank: Source attention context-parallel rank.
        :param attn_tp_rank: Source attention tensor-parallel rank.
        :param pp_rank: Source pipeline-parallel rank.
        :param transfer_source_rank: Rank encoded in transfer notifications.
        :param agent_name: Name encoded by the native metadata.
        :param metadata: Exact native agent metadata.
        :param metadata_sha256: Validated metadata digest.
        :param process_generation: Source process generation.
        :param control_endpoint: Actual manager-owned source listener.
        :returns: Exact retained native peer record.
        :raises RuntimeError: If an existing route conflicts or native setup fails.
        """

        route_key = (
            bootstrap_addr,
            attn_dp_rank,
            attn_cp_rank,
            attn_tp_rank,
            pp_rank,
        )
        if type(control_endpoint) is not NetworkAddress:
            raise TypeError("control_endpoint must be NetworkAddress")
        with self._prefill_peer_lock:
            existing = self._prefill_peers.get(route_key)
            if existing is not None:
                unchanged = (
                    existing.transfer_source_rank == transfer_source_rank
                    and existing.agent_name == agent_name
                    and existing.metadata_sha256 == metadata_sha256
                    and existing.process_generation == process_generation
                    and existing.control_endpoint == control_endpoint
                )
                if not unchanged:
                    raise RuntimeError(
                        "Conflicting or stale prefill NIXL route generation"
                    )
                if existing.handle in self._quarantined_remote_handles:
                    raise RuntimeError("Prefill NIXL route generation is quarantined")
                self.agent.make_connection(existing.handle)
                return existing

            same_name = self._prefill_peers_by_agent_name.get(agent_name)
            if same_name is not None:
                raise RuntimeError("Prefill NIXL agent name is reused by another route")

            handle = self.agent.add_remote_agent(metadata)
            tracked_peer = self._prefill_peers_by_handle.get(handle)
            if handle in self._quarantined_remote_handles:
                raise RuntimeError("Prefill NIXL handle is quarantined")
            if handle.name != agent_name:
                if tracked_peer is None:
                    self._discard_untracked_remote_handle(
                        handle, "prefill metadata name mismatch"
                    )
                raise RuntimeError(
                    "Prefill NIXL metadata resolved to a different agent"
                )
            if tracked_peer is not None:
                raise RuntimeError("Prefill NIXL handle is reused by another route")

            try:
                self.agent.make_connection(handle)
            except Exception:
                self._discard_untracked_remote_handle(
                    handle, "prefill proactive connection failure"
                )
                raise

            peer = _NixlPrefillPeer(
                bootstrap_addr=bootstrap_addr,
                attn_dp_rank=attn_dp_rank,
                attn_cp_rank=attn_cp_rank,
                attn_tp_rank=attn_tp_rank,
                pp_rank=pp_rank,
                transfer_source_rank=transfer_source_rank,
                agent_name=agent_name,
                metadata_sha256=metadata_sha256,
                process_generation=process_generation,
                control_endpoint=control_endpoint,
                handle=handle,
            )
            self._prefill_peers[route_key] = peer
            self._prefill_peer_keys_by_addr[bootstrap_addr].add(route_key)
            self._prefill_peers_by_agent_name[agent_name] = peer
            self._prefill_peers_by_handle[handle] = peer
            return peer

    def _discard_untracked_remote_handle(
        self, handle: nixl_remote_agent_handle, context: str
    ) -> None:
        """Remove or quarantine one exact half-loaded native handle.

        The caller holds ``_prefill_peer_lock`` whenever decode-side peer maps
        may be concurrently mutated.

        :param handle: Exact native handle which failed before publication.
        :param context: Failure context for diagnostics.
        """

        if handle in self._prefill_peers_by_handle:
            return
        if self.disaggregation_mode == DisaggregationMode.PREFILL and any(
            peer_info.remote_handle is handle
            for peer_info in self.decode_kv_args_table.values()
        ):
            return
        try:
            self.agent.remove_remote_agent(handle)
        except Exception:
            self._quarantined_remote_handles.add(handle)
            logger.error(
                "Quarantined half-loaded NIXL handle after %s:\n%s",
                context,
                traceback.format_exc(),
            )

    def _remove_prefill_peers(self, bootstrap_addr: str) -> None:
        """Remove every native prefill peer owned by one failed bootstrap.

        :param bootstrap_addr: Failed bootstrap service address.
        """

        with self._prefill_peer_lock:
            route_keys = set(self._prefill_peer_keys_by_addr.get(bootstrap_addr, set()))
            for route_key in route_keys:
                peer = self._prefill_peers.get(route_key)
                if peer is None:
                    continue
                try:
                    self.agent.remove_remote_agent(peer.handle)
                except Exception:
                    self._quarantined_remote_handles.add(peer.handle)
                    logger.error(
                        "Quarantined prefill NIXL peer %s after bootstrap loss:\n%s",
                        peer.agent_name,
                        traceback.format_exc(),
                    )
                    continue
                self._prefill_peers.pop(route_key, None)
                self._prefill_peers_by_agent_name.pop(peer.agent_name, None)
                self._prefill_peers_by_handle.pop(peer.handle, None)
                self._quarantined_remote_handles.discard(peer.handle)
            remaining_keys = self._prefill_peer_keys_by_addr.get(bootstrap_addr, set())
            remaining_keys.difference_update(
                route_key
                for route_key in route_keys
                if route_key not in self._prefill_peers
            )
            if len(remaining_keys) == 0:
                self._prefill_peer_keys_by_addr.pop(bootstrap_addr, None)

    def _handle_node_failure(self, failed_bootstrap_addr: str):
        super()._handle_node_failure(failed_bootstrap_addr)
        self._remove_prefill_peers(failed_bootstrap_addr)

    def _remote_decode_peer_handle(self, agent_name: str) -> nixl_remote_agent_handle:
        """Resolve an already authenticated decoder peer to its native handle.

        :param agent_name: Stable lookup key from the registered decoder route.
        :returns: Exact retained native peer handle.
        :raises RuntimeError: If the route has no live native authority.
        """

        peer_info = self.decode_kv_args_table.get(agent_name)
        if peer_info is None or peer_info.remote_handle is None:
            raise RuntimeError(f"Decoder NIXL peer is not loaded: {agent_name}")
        return peer_info.remote_handle

    def _record_and_release_failed_transfer(
        self,
        handle: Any,
        context: str,
        error: BaseException,
    ) -> None:
        """Capture live transport evidence before releasing a failed handle.

        :param handle: Failed NIXL transfer handle.
        :param context: Operation label used in failure diagnostics.
        :param error: Transport exception raised while posting the handle.
        """

        try:
            snapshot = self.agent.query_xfer_attestation(handle)
            native_segments = tuple(snapshot.segments)
            sample_indices = set(
                range(
                    min(
                        len(native_segments),
                        NIXL_ATTESTATION_SEGMENT_SAMPLE_COUNT,
                    )
                )
            )
            sample_indices.update(
                range(
                    max(
                        0,
                        len(native_segments) - NIXL_ATTESTATION_SEGMENT_SAMPLE_COUNT,
                    ),
                    len(native_segments),
                )
            )
            unposted_samples = 0
            for segment_index, segment in enumerate(native_segments):
                if segment.posted:
                    continue
                sample_indices.add(segment_index)
                unposted_samples += 1
                if unposted_samples >= NIXL_ATTESTATION_SEGMENT_SAMPLE_COUNT:
                    break

            sampled_segments = [
                {
                    "index": segment.index,
                    "local_address": f"0x{segment.localAddress:x}",
                    "remote_address": f"0x{segment.remoteAddress:x}",
                    "local_device_id": segment.localDeviceId,
                    "remote_device_id": segment.remoteDeviceId,
                    "length": segment.length,
                    "posted": segment.posted,
                    "endpoint_identity": segment.endpointIdentity,
                    "request_info": segment.requestInfo,
                    "selected_transports": [
                        {
                            "transport": transport.transport,
                            "device": transport.device,
                        }
                        for transport in segment.selectedTransports
                    ],
                }
                for segment_index, segment in enumerate(native_segments)
                if segment_index in sample_indices
            ]
            endpoints = []
            for endpoint in snapshot.endpoints:
                segment_indices = tuple(endpoint.segmentIndices)
                endpoints.append(
                    {
                        "worker_id": endpoint.workerId,
                        "worker_identity": endpoint.workerIdentity,
                        "endpoint_identity": endpoint.endpointIdentity,
                        "segment_count": len(segment_indices),
                        "first_segment_index": (
                            segment_indices[0] if len(segment_indices) > 0 else None
                        ),
                        "last_segment_index": (
                            segment_indices[-1] if len(segment_indices) > 0 else None
                        ),
                        "flush_posted": endpoint.flushPosted,
                        "remote_flushed": endpoint.remoteFlushed,
                        "transports": [
                            {
                                "transport": transport.transport,
                                "device": transport.device,
                            }
                            for transport in endpoint.transports
                        ],
                    }
                )
            evidence = {
                "state": str(snapshot.state),
                "status": str(snapshot.status),
                "error": snapshot.error,
                "backend": snapshot.backend,
                "submission_sealed": snapshot.submissionSealed,
                "completion_claimed": snapshot.completionClaimed,
                "segment_count": len(native_segments),
                "posted_segment_count": sum(
                    int(segment.posted) for segment in native_segments
                ),
                "total_bytes": sum(segment.length for segment in native_segments),
                "sampled_segments": sampled_segments,
                "endpoints": endpoints,
            }
            logger.error(
                "%s failed while posting (%s: %s); NIXL attestation=%s",
                context,
                type(error).__name__,
                error,
                json.dumps(evidence, sort_keys=True),
            )
        except Exception:
            logger.error(
                "%s failed while posting (%s: %s), and its NIXL attestation "
                "could not be queried:\n%s",
                context,
                type(error).__name__,
                error,
                traceback.format_exc(),
            )

        try:
            self.agent.release_xfer_handle(handle)
        except Exception:
            logger.error(
                "%s failed NIXL handle could not be released:\n%s",
                context,
                traceback.format_exc(),
            )

    def _post_transfer_when_ready(self, handle: Any, context: str) -> Any:
        """Post one transfer after its exact remote capability converges.

        A ``NOT_READY`` result is guaranteed to precede submission and data
        movement. Retrying the same handle therefore preserves the prepared
        descriptors and transfer-attached notification while the native
        OFFER/ACK protocol converges.

        :param handle: Prepared NIXL transfer handle.
        :param context: Operation label used in failure diagnostics.
        :returns: The posted transfer handle.
        :raises RuntimeError: If capability admission times out or posting fails.
        """

        deadline = time.monotonic() + NIXL_CAPABILITY_READY_TIMEOUT_SECONDS
        try:
            state = self.agent.transfer(handle)
            while state == "NOT_READY":
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{context} capability admission timed out after "
                        f"{NIXL_CAPABILITY_READY_TIMEOUT_SECONDS:.3f}s"
                    )
                time.sleep(NIXL_CAPABILITY_RETRY_INTERVAL_SECONDS)
                state = self.agent.transfer(handle)
        except _NIXL_TRANSPORT_ERRORS as error:
            self._record_and_release_failed_transfer(handle, context, error)
            raise

        if state == "ERR":
            error = RuntimeError(f"{context} failed to post")
            self._record_and_release_failed_transfer(handle, context, error)
            raise error
        if state not in ("DONE", "PROC"):
            raise RuntimeError(f"{context} returned unexpected state {state!r}")
        return handle

    def _post_terminal_transfer_once(self, handle: Any, context: str) -> Any:
        """Post one statically enrolled terminal transfer without polling.

        Terminal startup freezes both endpoint authority and registered memory
        before request admission. A ``NOT_READY`` result therefore contradicts
        the serving cohort and must fail closed under the native owner instead
        of borrowing progress from a retry loop.

        :param handle: Exact native-owner-armed NIXL transfer handle.
        :param context: Operation label used in failure diagnostics.
        :returns: The posted transfer handle.
        """

        try:
            state = self.agent.transfer(handle)
        except _NIXL_TRANSPORT_ERRORS:
            logger.error(
                "%s terminal post raised before terminality:\n%s",
                context,
                traceback.format_exc(),
            )
            raise
        if state not in ("DONE", "PROC"):
            raise RuntimeError(f"{context} terminal one-shot post returned {state!r}")
        return handle

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo) -> None:
        """Retain one decoder peer under legacy or sealed startup authority.

        :param decode_kv_args: Complete decoder registration.
        """

        if self.terminal_startup_binding is None:
            self._retain_decoder_peer(decode_kv_args)
            return
        self.enroll_terminal_decoder_peer(decode_kv_args)

    def enroll_terminal_decoder_peer(
        self,
        decode_kv_args: KVArgsRegisterInfo,
    ) -> TerminalStartupRankAdvertisement:
        """Authenticate, retain, and roster one startup decoder rank.

        :param decode_kv_args: Complete full-metadata decoder registration.
        :returns: Exact decoder row authenticated by the registration.
        :raises RuntimeError: If the rank is stale, duplicate, or post-freeze.
        """

        if type(decode_kv_args) is not KVArgsRegisterInfo:
            raise TypeError("decode_kv_args must be KVArgsRegisterInfo")
        enrollment = self._require_terminal_startup_peer_enrollment()
        if enrollment.binding.advertisement.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("only a terminal source enrolls decoder peers")
        rank = self._terminal_remote_rank(
            agent_name=decode_kv_args.agent_name,
            agent_metadata=decode_kv_args.agent_metadata,
            process_generation=decode_kv_args.process_generation,
            role=TerminalOwnerRole.DECODE,
            tensor_parallel_rank=decode_kv_args.decode_tp_rank,
            tensor_parallel_size=decode_kv_args.decode_tp_size,
        )
        with enrollment.lock:
            if enrollment.frozen:
                raise RuntimeError("terminal native peer roster is frozen")
            if rank.key in enrollment.decoder_peers:
                raise RuntimeError("terminal decoder rank was enrolled more than once")

        self._retain_decoder_peer(decode_kv_args)
        retained = self.decode_kv_args_table.get(decode_kv_args.agent_name)
        if retained is None or retained.remote_handle is None:
            raise RuntimeError("terminal decoder native peer was not retained")
        with enrollment.lock:
            if enrollment.frozen or rank.key in enrollment.decoder_peers:
                raise RuntimeError("terminal decoder enrollment changed concurrently")
            enrollment.decoder_peers[rank.key] = retained
            enrollment.freeze_if_complete()
        return rank

    def _retain_decoder_peer(self, decode_kv_args: KVArgsRegisterInfo) -> None:
        """Create or resolve one validated decoder native handle.

        :param decode_kv_args: Complete decoder registration.
        """

        try:
            agent_name = validate_nixl_agent_name(decode_kv_args.agent_name)
            agent_metadata = validate_nixl_agent_metadata(decode_kv_args.agent_metadata)
        except ValueError as error:
            raise RuntimeError("Invalid decoder NIXL peer identity") from error
        process_generation = self._canonical_process_generation(
            decode_kv_args.process_generation
        )
        uses_packed_runtime = decode_kv_args.packed_advertisement is not None
        if uses_packed_runtime:
            if self._packed_prefill_runtime is None:
                raise RuntimeError(
                    "decoder advertised packed transfer to a legacy prefill"
                )
            self._packed_destination_manifest(decode_kv_args)

        if (
            not uses_packed_runtime
            and not self.is_mla_backend
            and not self.is_hybrid_mla_backend
            and decode_kv_args.decode_tp_size != self.attn_tp_size
        ):
            pairs = build_transfer_entry_pairs(
                self.kv_args.kv_layer_ids,
                decode_kv_args.dst_kv_layer_ids,
                len(self.kv_args.kv_item_lens),
                len(decode_kv_args.dst_kv_item_lens),
                allow_positional_fallback=self.pp_size == 1,
            )
            require_uniform_asymmetric_kv_entry_geometry(
                source_item_lens=tuple(
                    self.kv_args.kv_item_lens[source_index] for source_index, _ in pairs
                ),
                destination_item_lens=tuple(
                    decode_kv_args.dst_kv_item_lens[destination_index]
                    for _, destination_index in pairs
                ),
                source_tp_size=self.attn_tp_size,
                destination_tp_size=decode_kv_args.decode_tp_size,
            )

        handle = self.agent.add_remote_agent(agent_metadata)
        with self._prefill_peer_lock:
            if handle in self._quarantined_remote_handles:
                raise RuntimeError("Decoder NIXL handle is quarantined")
        if handle.name != agent_name:
            with self._prefill_peer_lock:
                self._discard_untracked_remote_handle(
                    handle, "decoder metadata name mismatch"
                )
            raise RuntimeError("Decoder NIXL metadata resolved to a different agent")

        existing = self.decode_kv_args_table.get(agent_name)
        if existing is not None:
            unchanged = (
                existing.process_generation == process_generation
                and existing.agent_metadata == agent_metadata
                and existing.registration_digest == decode_kv_args.registration_digest
                and existing.remote_handle is handle
            )
            if not unchanged:
                raise RuntimeError("Conflicting or stale decoder NIXL registration")
            self.agent.make_connection(handle)
            return

        try:
            self.agent.make_connection(handle)
        except Exception:
            with self._prefill_peer_lock:
                self._discard_untracked_remote_handle(
                    handle, "decoder proactive connection failure"
                )
            raise
        decode_kv_args.process_generation = process_generation
        decode_kv_args.remote_handle = handle
        try:
            if (
                self.disaggregation_mode == DisaggregationMode.PREFILL
                and not uses_packed_runtime
            ):
                self._prepare_payload_xfer(decode_kv_args)
        except Exception:
            self.prep_handles.pop(agent_name, None)
            self.prep_handles_slice_dst.pop(agent_name, None)
            decode_kv_args.kv_xfer_segments = None
            decode_kv_args.remote_handle = None
            with self._prefill_peer_lock:
                self._discard_untracked_remote_handle(
                    handle, "decoder transfer preparation failure"
                )
            raise
        self.decode_kv_args_table[agent_name] = decode_kv_args

    def _send_kvcache_generic(
        self,
        peer_name: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        state_type: Optional[StateType] = None,
        src_mem_kind: str = "VRAM",
        dst_mem_kind: str = "VRAM",
        force_flat: bool = False,
    ):
        """Generic KV cache transfer supporting both MHA and MLA architectures.
        Used by both send_kvcache and maybe_send_extra.

        ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
        even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
        index) whose per-layer list must not be half-split into K/V."""
        # Prepped path (KV only; state transfers use the non-prepped path below).
        if (
            src_data_ptrs is self.kv_args.kv_data_ptrs
            and "" in self.prep_handles
            and peer_name in self.prep_handles
        ):
            src_prep = self.prep_handles[""]
            dst_prep = self.prep_handles[peer_name]
            info = self.decode_kv_args_table[peer_name]
            num_slots_dst = (
                info.dst_num_slots
                if info.dst_num_slots is not None
                else self._num_slots_src
            )
            num_layers = len(item_lens)
            src_indices = repeat_indices_over_layers(
                prefill_data_indices, num_layers, self._num_slots_src
            )
            dst_indices = repeat_indices_over_layers(
                dst_data_indices, num_layers, num_slots_dst
            )
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE",
                src_prep,
                src_indices,
                dst_prep,
                dst_indices,
                notif.encode("ascii"),
            )
            if not xfer_handle:
                raise Exception("KVSender failed to create prepped transfer")
            return self._post_transfer_when_ready(
                xfer_handle, "KVSender prepped transfer"
            )

        # Non-prepped path: used for state transfers (SWA/NSA) via maybe_send_extra.
        # Convert pointer lists to np.uint64 arrays up front.
        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set, e.g.
        # 0xffff81ab54e01000). Casting here prevents overflow when these values
        # are later used in numpy arithmetic.
        src_data_ptrs = np.array(src_data_ptrs, dtype=np.uint64)
        dst_data_ptrs = np.array(dst_data_ptrs, dtype=np.uint64)
        item_lens = np.array(item_lens, dtype=np.uint64)

        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        if self.is_mla_backend or force_flat:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs, state_type)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )

            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]

        src_addrs = []
        src_lens = []
        dst_addrs = []
        dst_lens = []

        # Precompute block starts/lengths to reduce Python-level loops.
        prefill_starts = np.fromiter(
            (block[0] for block in prefill_kv_blocks), dtype=np.uint64
        )
        dst_starts = np.fromiter((block[0] for block in dst_kv_blocks), dtype=np.uint64)
        block_lens = np.fromiter(
            (len(block) for block in prefill_kv_blocks), dtype=np.uint64
        )

        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)

        def make_req_array(addr_chunks, len_chunks, gpu):
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.uint64)
            flat_addrs = np.concatenate(addr_chunks).astype(np.uint64, copy=False)
            flat_lens = np.concatenate(len_chunks).astype(np.uint64, copy=False)
            return np.column_stack(
                (
                    flat_addrs,
                    flat_lens,
                    np.full_like(flat_addrs, gpu, dtype=np.uint64),
                )
            )

        src_reqs = make_req_array(
            src_addrs, src_lens, _nixl_device_id(src_mem_kind, self.kv_args.gpu_id)
        )
        dst_reqs = make_req_array(
            dst_addrs, dst_lens, _nixl_device_id(dst_mem_kind, dst_gpu_id)
        )

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, src_mem_kind)
        dst_descs = self.agent.get_xfer_descs(dst_reqs, dst_mem_kind)
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            self._remote_decode_peer_handle(peer_name),
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        return self._post_transfer_when_ready(xfer_handle, "KVSender transfer")

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        dst_mem_kind: str = "VRAM",
    ):
        assert self.src_mem_kind is not None
        return self._send_kvcache_generic(
            peer_name=peer_name,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            dst_gpu_id=dst_gpu_id,
            notif=notif,
            src_mem_kind=self.src_mem_kind,
            dst_mem_kind=dst_mem_kind,
        )

    def send_kvcache_mixed(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ):
        info = self.decode_kv_args_table[peer_name]
        segments = info.kv_xfer_segments
        assert segments is not None
        if not segments:
            raise RuntimeError(f"Missing NIXL mixed KV transfer plan for {peer_name}")

        num_parts = len(segments)
        handles = []
        for part_idx, seg in enumerate(segments):
            num_layers = seg.end - seg.start
            src_indices = repeat_indices_over_layers(
                prefill_kv_indices, num_layers, self._num_slots_src
            )
            dst_indices = repeat_indices_over_layers(
                dst_kv_indices, num_layers, seg.dst_num_slots
            )
            part_notif = f"{notif}_part_{part_idx}_{num_parts}"
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE",
                seg.src_handle,
                src_indices,
                seg.dst_handle,
                dst_indices,
                part_notif.encode("ascii"),
            )
            if not xfer_handle:
                raise Exception("KVSender failed to create mixed prepped transfer")
            handles.append(
                self._post_transfer_when_ready(
                    xfer_handle, "KVSender mixed prepped transfer"
                )
            )
        return handles

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ) -> Any | None:
        # Prepped path: src dlist is shared per decode_tp_size; dst is per peer.
        assert self.prep_handle_slice_src is not None
        assert peer_name in self.prep_handles_slice_dst
        src_handle, num_groups, num_ptr_pairs, num_slots_src = (
            self.prep_handle_slice_src
        )
        dst_handle, num_slots_dst, head_group_idx = self.prep_handles_slice_dst[
            peer_name
        ]
        page_size = self.kv_args.page_size
        src_indices = expand_page_indices_for_slice(
            np.asarray(prefill_kv_indices, dtype=np.int32),
            num_ptr_pairs,
            num_slots_src,
            page_size,
            num_groups=num_groups,
            head_group_idx=head_group_idx,
        )
        dst_indices = expand_page_indices_for_slice(
            np.asarray(dst_kv_indices, dtype=np.int32),
            num_ptr_pairs,
            num_slots_dst,
            page_size,
        )
        if src_indices.size != dst_indices.size:
            raise ValueError(
                "Prepped slice transfer index count mismatch: "
                f"source={src_indices.size}, destination={dst_indices.size}"
            )
        if src_indices.size == 0:
            return None

        descriptor_limit = _direct_kv_source_cohort_descriptor_limit(self.attn_tp_size)
        num_parts = (src_indices.size + descriptor_limit - 1) // descriptor_limit
        for part_index in range(num_parts):
            start = part_index * descriptor_limit
            request_slice = slice(
                start,
                min(start + descriptor_limit, src_indices.size),
            )
            is_last_part = part_index + 1 == num_parts
            part_notification = notif.encode("ascii") if is_last_part else b""
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE",
                src_handle,
                src_indices[request_slice],
                dst_handle,
                dst_indices[request_slice],
                part_notification,
            )
            if xfer_handle is None:
                raise RuntimeError(
                    "KVSender failed to create prepped slice transfer "
                    f"part {part_index + 1}/{num_parts}"
                )
            context = (
                "KVSender prepped slice transfer part " f"{part_index + 1}/{num_parts}"
            )
            posted_handle = self._post_transfer_when_ready(xfer_handle, context)
            if is_last_part:
                return posted_handle
            intermediate_handles = [posted_handle]
            self._wait_and_release_transfer_handles(
                intermediate_handles,
                context,
            )

        raise RuntimeError("KVSender prepped slice transfer produced no request parts")

    def send_kvcache_staged(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_gpu_id: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        notif: str,
        staging_buffer=None,
    ) -> bool:
        """Transfer KV cache through independently progressed staging writes.

        :param peer_name: Registered decoder NIXL agent name.
        :param prefill_kv_indices: Source KV page indices.
        :param dst_staging_ptr: Destination staging address for this writer.
        :param dst_staging_size: Remaining destination staging capacity.
        :param dst_gpu_id: Destination NIXL GPU identifier.
        :param dst_tp_rank: Destination attention TP rank.
        :param dst_attn_tp_size: Destination attention TP width.
        :param dst_kv_item_len: Destination KV bytes per page.
        :param notif: Base staging notification ending in ``peer_name``.
        :param staging_buffer: Source staging allocation.
        :returns: Whether every bounded write completed and released its handle.
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            gather_all_layers_to_staging,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return False

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return False
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes, "
                f"have {dst_staging_size}, falling back"
            )
            return False

        # gather_all_layers_to_staging() runs the gather kernel on its own
        # dedicated stream and synchronizes that stream before returning, so
        # the staging buffer is fully populated and visible to the NIC by the
        # time we post the RDMA WRITE below. No extra sync needed (matches
        # mooncake's send_kvcache_staged behavior).
        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        dst_write_ptr = dst_staging_ptr + rank_offset
        src_reqs, dst_reqs = _build_contiguous_rma_requests(
            staging_buffer.get_ptr(),
            dst_write_ptr,
            per_rank_bytes,
            self.kv_args.gpu_id,
            dst_gpu_id,
        )

        notification_suffix = f"_{peer_name}"
        if not notif.endswith(notification_suffix):
            raise ValueError("staging notification is not bound to the decoder peer")
        notification_prefix = notif[: -len(notification_suffix)]
        remote_handle = self._remote_decode_peer_handle(peer_name)
        num_parts = len(src_reqs)
        for part_idx, (src_req, dst_req) in enumerate(
            zip(src_reqs, dst_reqs, strict=True)
        ):
            src_descs = self.agent.get_xfer_descs(src_req.reshape(1, 3), "VRAM")
            dst_descs = self.agent.get_xfer_descs(dst_req.reshape(1, 3), "VRAM")
            part_notification = (
                f"{notification_prefix}_part_{part_idx}_{num_parts}_{peer_name}"
            )
            xfer_handle = self.agent.initialize_xfer(
                "WRITE",
                src_descs,
                dst_descs,
                remote_handle,
                part_notification.encode("ascii"),
            )
            if not xfer_handle:
                raise RuntimeError(
                    "[Staging] Failed to create a bounded NIXL transfer "
                    f"(part={part_idx}, parts={num_parts}, "
                    f"src=0x{int(src_req[0]):x}, dst=0x{int(dst_req[0]):x}, "
                    f"size={int(src_req[1])})"
                )
            posted_handle = self._post_transfer_when_ready(
                xfer_handle,
                f"[Staging] NIXL transfer part {part_idx + 1}/{num_parts}",
            )
            while True:
                state = self.agent.check_xfer_state(posted_handle)
                if state == "DONE":
                    break
                if state == "ERR":
                    raise RuntimeError(
                        "[Staging] NIXL transfer part encountered ERR "
                        f"(part={part_idx}, parts={num_parts})"
                    )
                time.sleep(0)
            self.agent.release_xfer_handle(posted_handle)

        return True

    def _try_create_staging_strategy(self, staging_buffer):
        """Create a per-worker PrefillStagingStrategy bound to ``staging_buffer``.

        Returns ``None`` if staging is disabled or kv tensors not yet set.
        Caller is expected to keep the returned strategy as a worker-local
        variable; never cache on ``self`` (multiple workers would race on
        the underlying staging ring buffer).
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk: TransferKVChunk,
        src_prefill_kv_indices: npt.NDArray[np.int32],
        req: TransferInfo,
        dst_info: KVArgsRegisterInfo,
        queue: FastQueue,
    ):
        """Attempt staging transfer for one chunk.

        Mirrors mooncake._do_staging_transfer semantics:
          - staging not ready (watermark/alloc pending) -> ``queue.put(kv_chunk)``
            re-enqueue the chunk and return ``(False, True)``. Caller should
            ``break`` out of the per-req loop and ``continue`` the worker
            main loop without updating room status -- the chunk will be
            retried on the next pop.
          - oversized chunk (will never fit) -> raise RuntimeError.
          - staging successfully posted -> return ``(True, False)``. Every
            bounded handle is complete and released before the next is posted.
          - send_kvcache_staged returned false (chunk cannot fit; decode buffer
            too small, kv_buffer_tensors missing, etc.) -> raise RuntimeError
            instead of falling back to the slice path.
        """
        page_start = kv_chunk.index_slice.start
        num_pages = len(kv_chunk.prefill_kv_indices)

        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req, page_start, num_pages, session_id=req.agent_name
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import (
                StagingAllocator,
            )

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                raise RuntimeError(
                    f"[Staging] Chunk staging allocation permanently failed: "
                    f"chunk exceeds ring buffer total size "
                    f"(room={kv_chunk.room}). Increase "
                    f"SGLANG_DISAGG_STAGING_POOL_SIZE_MB."
                )
            # Not ready yet: wait (bounded) for a watermark advance, then
            # re-enqueue to retry. A plain block-until-ready would head-of-line
            # block other rooms on this single worker thread.
            with self._staging_ctx.watermark_cv:
                self._staging_ctx.watermark_cv.wait(STAGING_WATERMARK_WAIT_S)
            queue.put(kv_chunk)
            return (False, True)

        notif_tag = (
            f"{req.room}_stg_{kv_chunk.chunk_id}_{int(kv_chunk.is_last_chunk)}"
            f"_{self.transfer_source_rank}_{chunk_idx}"
            f"_{page_start}_{num_pages}_{req.agent_name}"
        )
        completed = self.send_kvcache_staged(
            req.agent_name,
            src_prefill_kv_indices,
            dst_info.staging.base_ptr + c_offset,
            dst_info.staging.total_size - c_offset,
            dst_info.gpu_id,
            dst_info.decode_tp_rank,
            dst_info.decode_tp_size,
            dst_info.dst_kv_item_len,
            notif_tag,
            staging_buffer=staging_strategy.staging_buffer,
        )
        if not completed:
            # A silent slice fallback would leak this chunk's decode-side
            # allocation and pin the ring watermark; with grid-aligned sends
            # not fitting can only mean misconfiguration.
            raise RuntimeError(
                f"[Staging] Staged transfer cannot fit chunk "
                f"(room={kv_chunk.room}, chunk_idx={chunk_idx}, "
                f"pages={num_pages}). Increase "
                f"SGLANG_DISAGG_STAGING_POOL_SIZE_MB or reduce "
                f"chunked_prefill_size."
            )
        return (True, False)

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            self._remote_decode_peer_handle(peer_name),
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        return self._post_transfer_when_ready(
            xfer_handle, "KVSender auxiliary transfer"
        )

    def _send_tp_sharded_state(
        self,
        peer_name: str,
        source_indices: list[int],
        source_data_ptrs: list[int],
        source_item_lens: list[int],
        destination_indices: list[int],
        destination_data_ptrs: list[int],
        destination_item_lens: list[int],
        destination_gpu_id: int,
        destination_tp_size: int,
        destination_engine_rank: int,
        notification: str,
        source_layer_ids: list[int] | None = None,
        destination_layer_ids: list[int] | None = None,
    ) -> Any | None:
        """Transfer a page-indexed state whose token payload is TP-sharded.

        :param peer_name: NIXL peer receiving the state.
        :param source_indices: Source page indices.
        :param source_data_ptrs: Source state-buffer base addresses.
        :param source_item_lens: Source bytes per page for each state buffer.
        :param destination_indices: Destination page indices.
        :param destination_data_ptrs: Destination state-buffer base addresses.
        :param destination_item_lens: Destination bytes per page for each buffer.
        :param destination_gpu_id: Destination GPU identifier used by NIXL.
        :param destination_tp_size: Destination attention TP width.
        :param destination_engine_rank: Destination global engine rank.
        :param notification: Transfer completion notification.
        :param source_layer_ids: Optional source layer identifiers.
        :param destination_layer_ids: Optional destination layer identifiers.
        :returns: The NIXL transfer handle, or ``None`` when no buffers are paired.
        :raises ValueError: If page indices or TP-sharded buffer geometries differ.
        """

        source_indices_signed = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        destination_indices_signed = np.asarray(
            destination_indices, dtype=np.int64
        ).reshape(-1)
        if np.any(source_indices_signed < 0):
            raise ValueError(
                "Source TP-sharded state page indices must be non-negative"
            )
        if np.any(destination_indices_signed < 0):
            raise ValueError(
                "Destination TP-sharded state page indices must be non-negative"
            )

        source_indices_array = source_indices_signed.astype(np.uint64, copy=False)
        destination_indices_array = destination_indices_signed.astype(
            np.uint64, copy=False
        )
        if source_indices_array.size != destination_indices_array.size:
            raise ValueError(
                "TP-sharded state index count mismatch: "
                f"source={source_indices_array.size}, "
                f"destination={destination_indices_array.size}"
            )
        if source_indices_array.size == 0:
            return None

        if len(source_data_ptrs) != len(source_item_lens):
            raise ValueError(
                "Source state pointer/item-length count mismatch: "
                f"{len(source_data_ptrs)} pointers and "
                f"{len(source_item_lens)} item lengths"
            )
        if len(destination_data_ptrs) != len(destination_item_lens):
            raise ValueError(
                "Destination state pointer/item-length count mismatch: "
                f"{len(destination_data_ptrs)} pointers and "
                f"{len(destination_item_lens)} item lengths"
            )

        pairs = build_transfer_entry_pairs(
            source_layer_ids if source_layer_ids is not None else [],
            destination_layer_ids if destination_layer_ids is not None else [],
            len(source_data_ptrs),
            len(destination_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        if len(pairs) == 0:
            return None

        page_size = self.kv_args.page_size
        if page_size <= 0:
            raise ValueError(f"KV page size must be positive, got {page_size}")

        source_tp_size = self.attn_tp_size
        if source_tp_size <= 0:
            raise ValueError(
                f"Source attention TP size must be positive, got {source_tp_size}"
            )
        if destination_tp_size <= 0:
            raise ValueError(
                "Destination attention TP size must be positive, got "
                f"{destination_tp_size}"
            )
        if self.kv_args.engine_rank < 0:
            raise ValueError(
                f"Source engine rank must be non-negative, got {self.kv_args.engine_rank}"
            )
        if destination_engine_rank < 0:
            raise ValueError(
                "Destination engine rank must be non-negative, got "
                f"{destination_engine_rank}"
            )

        source_tp_rank = self.kv_args.engine_rank % source_tp_size
        destination_tp_rank = destination_engine_rank % destination_tp_size
        token_offsets = np.arange(page_size, dtype=np.uint64)
        source_address_chunks: list[npt.NDArray[np.uint64]] = []
        destination_address_chunks: list[npt.NDArray[np.uint64]] = []
        length_chunks: list[npt.NDArray[np.uint64]] = []

        for source_index, destination_index in pairs:
            source_item_bytes = source_item_lens[source_index]
            destination_item_bytes = destination_item_lens[destination_index]
            if source_item_bytes <= 0 or destination_item_bytes <= 0:
                raise ValueError(
                    "TP-sharded state item lengths must be positive, got "
                    f"{source_item_bytes} and {destination_item_bytes}"
                )
            if source_item_bytes % page_size != 0:
                raise ValueError(
                    f"Source item length {source_item_bytes} is not divisible by "
                    f"page size {page_size}"
                )
            if destination_item_bytes % page_size != 0:
                raise ValueError(
                    f"Destination item length {destination_item_bytes} is not "
                    f"divisible by page size {page_size}"
                )

            source_token_bytes = source_item_bytes // page_size
            destination_token_bytes = destination_item_bytes // page_size
            shard = compute_tensor_parallel_shard(
                source_token_bytes=source_token_bytes,
                destination_token_bytes=destination_token_bytes,
                source_parallel_size=source_tp_size,
                destination_parallel_size=destination_tp_size,
                source_rank=source_tp_rank,
                destination_rank=destination_tp_rank,
            )

            source_page_bases = np.uint64(
                source_data_ptrs[source_index]
            ) + source_indices_array[:, None] * np.uint64(source_item_bytes)
            destination_page_bases = np.uint64(
                destination_data_ptrs[destination_index]
            ) + destination_indices_array[:, None] * np.uint64(destination_item_bytes)
            source_addresses = (
                source_page_bases
                + token_offsets[None, :] * np.uint64(source_token_bytes)
                + np.uint64(shard.source_offset_bytes)
            ).reshape(-1)
            destination_addresses = (
                destination_page_bases
                + token_offsets[None, :] * np.uint64(destination_token_bytes)
                + np.uint64(shard.destination_offset_bytes)
            ).reshape(-1)
            source_address_chunks.append(source_addresses)
            destination_address_chunks.append(destination_addresses)
            length_chunks.append(
                np.full(
                    source_addresses.size,
                    shard.length_bytes,
                    dtype=np.uint64,
                )
            )

        source_addresses = np.concatenate(source_address_chunks)
        destination_addresses = np.concatenate(destination_address_chunks)
        lengths = np.concatenate(length_chunks)
        source_requests = np.column_stack(
            (
                source_addresses,
                lengths,
                np.full(
                    source_addresses.size,
                    self.kv_args.gpu_id,
                    dtype=np.uint64,
                ),
            )
        )
        destination_requests = np.column_stack(
            (
                destination_addresses,
                lengths,
                np.full(
                    destination_addresses.size,
                    destination_gpu_id,
                    dtype=np.uint64,
                ),
            )
        )

        request_slices = _bounded_request_slices(
            lengths,
            max_descriptors=_source_cohort_descriptor_limit(source_tp_size),
        )
        remote_handle = self._remote_decode_peer_handle(peer_name)
        for part_index, request_slice in enumerate(request_slices):
            source_descriptors = self.agent.get_xfer_descs(
                source_requests[request_slice], "VRAM"
            )
            destination_descriptors = self.agent.get_xfer_descs(
                destination_requests[request_slice], "VRAM"
            )
            is_last_part = part_index + 1 == len(request_slices)
            part_notification = notification.encode("ascii") if is_last_part else b""
            transfer_handle = self.agent.initialize_xfer(
                "WRITE",
                source_descriptors,
                destination_descriptors,
                remote_handle,
                part_notification,
            )
            if transfer_handle is None:
                raise RuntimeError(
                    "Failed to create TP-sharded state transfer "
                    f"part {part_index + 1}/{len(request_slices)}"
                )
            context = (
                "TP-sharded state transfer part "
                f"{part_index + 1}/{len(request_slices)}"
            )
            posted_handle = self._post_transfer_when_ready(transfer_handle, context)
            if is_last_part:
                return posted_handle

            while True:
                try:
                    state = self.agent.check_xfer_state(posted_handle)
                except _NIXL_TRANSPORT_ERRORS as error:
                    self._record_and_release_failed_transfer(
                        posted_handle, context, error
                    )
                    raise
                if state == "DONE":
                    break
                if state == "ERR":
                    error = RuntimeError(f"{context} encountered ERR")
                    self._record_and_release_failed_transfer(
                        posted_handle, context, error
                    )
                    raise error
                time.sleep(0)
            self.agent.release_xfer_handle(posted_handle)

        raise RuntimeError("TP-sharded state transfer produced no request parts")

    def _send_mamba_state(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        src_layer_ids: list[int] = None,
        dst_layer_ids: list[int] = None,
    ):
        """Transfer Mamba states via RDMA."""
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        assert len(dst_state_indices) == len(
            prefill_state_indices
        ), "State indices count mismatch between Prefill and Decode"

        src_addrs = []
        dst_addrs = []

        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            length = src_state_item_lens[i]
            if length == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_addr = src_state_data_ptrs[i] + length * int(prefill_state_indices[0])
            dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
            src_addrs.append((src_addr, length, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, length, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            self._remote_decode_peer_handle(peer_name),
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state transfer")
        return self._post_transfer_when_ready(xfer_handle, "Mamba state transfer")

    def _send_mamba_state_slice(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int,
        src_state_conv_shard_groups: list = None,
        src_state_slice_outer_counts: list[int] = None,
        src_layer_ids: list[int] = None,
        dst_layer_ids: list[int] = None,
    ):
        """Transfer Mamba states with TP slice support via RDMA.

        When prefill and decode have different attn_tp_size, we slice the
        TP-sharded dimension (3rd dim) of conv_state and temporal_state
        accordingly, mirroring Mooncake's _send_mamba_state_slice. GDN
        conv_state is [query | key | value] with each sub-block head-sharded
        independently, so on the scatter path it is sliced per sub-block via
        ``src_state_conv_shard_groups`` (see compute_mamba_state_slice_blocks).
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes. "
            f"Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={decode_tp_size}."
        )
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
                src_layer_ids=src_layer_ids,
                dst_layer_ids=dst_layer_ids,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_addrs = []
        dst_addrs = []

        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[j]
            if src_item_len == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[j]

            conv_shard_groups = (
                src_state_conv_shard_groups[i]
                if src_state_conv_shard_groups and i < len(src_state_conv_shard_groups)
                else None
            )
            outer_count = (
                src_state_slice_outer_counts[i]
                if src_state_slice_outer_counts
                and i < len(src_state_slice_outer_counts)
                else 1
            )
            for (
                src_offset,
                dst_offset,
                bytes_to_send,
            ) in compute_mamba_state_slice_byte_blocks(
                src_item_len=src_item_len,
                dst_item_len=dst_item_len,
                src_dim=src_dim,
                dst_dim=dst_dim,
                outer_count=outer_count,
                src_attn_tp_size=self.attn_tp_size,
                dst_attn_tp_size=decode_tp_size,
                dst_tp_rank_in_group=dst_tp_rank_in_group,
                local_tp_rank_in_group=local_tp_rank_in_group,
                conv_shard_groups=conv_shard_groups,
            ):
                src_addr = (
                    src_state_data_ptrs[i]
                    + src_item_len * int(prefill_state_indices[0])
                    + src_offset
                )
                dst_addr = (
                    dst_state_ptr
                    + dst_item_len * int(dst_state_indices[0])
                    + dst_offset
                )
                src_addrs.append((src_addr, bytes_to_send, self.kv_args.gpu_id))
                dst_addrs.append((dst_addr, bytes_to_send, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            self._remote_decode_peer_handle(peer_name),
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state slice transfer")
        return self._post_transfer_when_ready(xfer_handle, "Mamba state slice transfer")

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[List[int]],
        dst_state_data_ptrs: List[List[int]],
        dst_state_indices: List[List[int]],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int = 0,
        dst_state_item_lens: List[List[int]] | None = None,
        dst_state_dim_per_tensor: List[List[int]] | None = None,
        dst_state_layer_ids: List[List[int]] | None = None,
    ):
        """Send state per hybrid component, dispatching by state_type[i]."""
        state_types = getattr(self.kv_args, "state_types", []) or []
        src_state_data_ptrs = self.kv_args.state_data_ptrs or []
        src_state_item_lens = self.kv_args.state_item_lens or []
        src_state_dim_per_tensor = (
            getattr(self.kv_args, "state_dim_per_tensor", []) or []
        )
        src_state_conv_shard_groups = (
            getattr(self.kv_args, "state_conv_shard_groups", []) or []
        )
        src_state_slice_outer_counts = (
            getattr(self.kv_args, "state_slice_outer_counts", []) or []
        )
        src_state_layer_ids = self.kv_args.state_layer_ids
        dst_state_item_lens = dst_state_item_lens or []
        dst_state_dim_per_tensor = dst_state_dim_per_tensor or []
        dst_state_layer_ids = dst_state_layer_ids or []

        handles = []
        for i, st in enumerate(state_types):
            src_indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if src_indices is None or len(src_indices) == 0:
                continue
            src_ptrs = src_state_data_ptrs[i] if i < len(src_state_data_ptrs) else []
            src_lens = src_state_item_lens[i] if i < len(src_state_item_lens) else []
            src_dims = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else []
            )
            src_conv = (
                src_state_conv_shard_groups[i]
                if i < len(src_state_conv_shard_groups)
                else []
            )
            src_outer_counts = (
                src_state_slice_outer_counts[i]
                if i < len(src_state_slice_outer_counts)
                else []
            )
            src_lids = src_state_layer_ids[i] if i < len(src_state_layer_ids) else []
            dst_ptrs = dst_state_data_ptrs[i] if i < len(dst_state_data_ptrs) else []
            dst_indices = dst_state_indices[i] if i < len(dst_state_indices) else []
            dst_lens = dst_state_item_lens[i] if i < len(dst_state_item_lens) else []
            dst_dims = (
                dst_state_dim_per_tensor[i] if i < len(dst_state_dim_per_tensor) else []
            )
            dst_lids = dst_state_layer_ids[i] if i < len(dst_state_layer_ids) else []
            comp_notif = f"{notif}_{i}"

            if st == StateType.MAMBA:
                if self.attn_tp_size != decode_tp_size:
                    h = self._send_mamba_state_slice(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        src_dims,
                        dst_ptrs,
                        dst_indices,
                        dst_lens,
                        dst_dims,
                        dst_gpu_id,
                        comp_notif,
                        decode_tp_size,
                        decode_tp_rank,
                        src_conv,
                        src_outer_counts,
                        src_layer_ids=src_lids,
                        dst_layer_ids=dst_lids,
                    )
                else:
                    h = self._send_mamba_state(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        dst_ptrs,
                        dst_indices,
                        dst_gpu_id,
                        comp_notif,
                        src_layer_ids=src_lids,
                        dst_layer_ids=dst_lids,
                    )
            elif st in (
                StateType.SWA,
                StateType.DSA,
                StateType.SWA_RING,
                StateType.C128_STATE,
            ):
                if (
                    st == StateType.SWA
                    and not self.is_mla_backend
                    and not self.is_hybrid_mla_backend
                    and self.attn_tp_size != decode_tp_size
                ):
                    h = self._send_tp_sharded_state(
                        peer_name=peer_name,
                        source_indices=src_indices,
                        source_data_ptrs=src_ptrs,
                        source_item_lens=src_lens,
                        destination_indices=dst_indices,
                        destination_data_ptrs=dst_ptrs,
                        destination_item_lens=dst_lens,
                        destination_gpu_id=dst_gpu_id,
                        destination_tp_size=decode_tp_size,
                        destination_engine_rank=decode_tp_rank,
                        notification=comp_notif,
                        source_layer_ids=src_lids,
                        destination_layer_ids=dst_lids,
                    )
                    if h is not None:
                        handles.append(h)
                    continue
                if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                if (
                    st == StateType.C128_STATE
                    and len(src_indices) == 0
                    and len(dst_indices) == 0
                ):
                    continue
                if len(src_indices) != len(dst_indices):
                    raise RuntimeError(
                        f"State index length mismatch at component {i}: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices)}"
                    )
                h = self._send_kvcache_generic(
                    peer_name=peer_name,
                    src_data_ptrs=src_ptrs,
                    dst_data_ptrs=dst_ptrs,
                    item_lens=src_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices, dtype=np.int32),
                    dst_gpu_id=dst_gpu_id,
                    notif=comp_notif,
                    state_type=st,
                )
            elif st == StateType.MINIMAX_INDEX_K:
                # Equal-TP / PP=1 only. Sub-pools are compacted sparse-layer
                # lists, so PP>1 mis-slices and heterogeneous TP is unsupported.
                if self.pp_size is not None and self.pp_size > 1:
                    raise RuntimeError(
                        "PD disagg: PP>1 not supported for MiniMax sparse index yet."
                    )
                if self.attn_tp_size != decode_tp_size:
                    raise RuntimeError(
                        "PD disagg: heterogeneous TP not supported for MiniMax "
                        "sparse index yet."
                    )
                if len(src_indices) != len(dst_indices):
                    raise RuntimeError(
                        f"State index length mismatch at component {i}: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices)}"
                    )
                h = self._send_kvcache_generic(
                    peer_name=peer_name,
                    src_data_ptrs=src_ptrs,
                    dst_data_ptrs=dst_ptrs,
                    item_lens=src_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices, dtype=np.int32),
                    dst_gpu_id=dst_gpu_id,
                    notif=comp_notif,
                    force_flat=True,
                )
            else:
                raise RuntimeError(
                    f"PD Disaggregation via NIXL does NOT support {st} hybrid models yet."
                )
            if h is not None:
                handles.append(h)
        return handles

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        producer_event: torch.cuda.Event | None = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        packed_route = self._packed_source_route(bootstrap_room)
        if packed_route is not None and is_last_chunk and producer_event is None:
            raise RuntimeError("packed source final chunk has no producer event")
        if packed_route is None and self.enable_staging:
            self._prefetch_staging_reqs(bootstrap_room)

        # Transfer is async: just enqueue the chunk; the per-queue worker
        # (transfer_worker) does the actual gather + RDMA. Routing by
        # ``room % N`` keeps every chunk of a given room on the same
        # worker -- and therefore on the same private staging buffer --
        # which is required for the staging ring's offset/watermark
        # state machine to advance correctly.
        shard_idx = bootstrap_room % len(self.transfer_queues)
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                chunk_id=chunk_id,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                producer_event=producer_event,
            )
        )
        return None

    def update_transfer_status(self):
        notif_map = self.agent.get_new_notifs()
        for peer_handle, messages in notif_map.items():
            for msg in messages:
                try:
                    components = msg.decode("ascii").split("_", 11)
                    room = int(components[0])
                except UnicodeDecodeError:
                    try:
                        room = int(msg.split(b"_", 1)[0].decode("ascii"))
                    except (UnicodeDecodeError, ValueError):
                        self._fail_rooms_bound_to_notification_peer(
                            peer_handle,
                            "Rejected NIXL notification: room is not parseable",
                        )
                        continue
                    if not self._notification_peer_is_bound_to_room(peer_handle, room):
                        logger.warning(
                            "Ignoring non-ASCII NIXL notification for unbound "
                            "room %d from %r",
                            room,
                            peer_handle,
                        )
                        continue
                    try:
                        self._validate_notification_peer(peer_handle, room)
                    except RuntimeError as error:
                        reason = f"Rejected NIXL notification: {error}"
                    else:
                        reason = "Rejected NIXL notification: payload is not ASCII"
                    self._fail_notification_room(room, reason)
                    continue
                except ValueError:
                    self._fail_rooms_bound_to_notification_peer(
                        peer_handle,
                        "Rejected NIXL notification: room is not parseable",
                    )
                    continue
                if not self._notification_peer_is_bound_to_room(peer_handle, room):
                    logger.warning(
                        "Ignoring NIXL notification for unbound room %d from %r",
                        room,
                        peer_handle,
                    )
                    continue
                try:
                    self._process_transfer_notification(
                        peer_handle, msg, components, room
                    )
                except (IndexError, RuntimeError, ValueError) as error:
                    self._fail_notification_room(
                        room,
                        f"Rejected NIXL notification: {error}",
                    )

    def _notification_peer_is_bound_to_room(
        self, peer_handle: nixl_remote_agent_handle, room: int
    ) -> bool:
        """Return whether one room explicitly expects the native peer handle.

        :param peer_handle: Exact native notification source.
        :param room: Decoder-minted bootstrap room.
        :returns: Whether the peer is a writer for the room.
        """

        status = self.transfer_statuses.get(room)
        return status is not None and peer_handle in status.expected_source_ranks

    def _validate_notification_peer(
        self, peer_handle: nixl_remote_agent_handle, room: int
    ) -> tuple[TransferStatus, int]:
        """Resolve one room-bound native peer and enforce its generation.

        :param peer_handle: Exact native notification source.
        :param room: Decoder-minted bootstrap room.
        :returns: Transfer status and the peer's bound source rank.
        :raises RuntimeError: If ownership is absent, stale, or quarantined.
        """

        status = self.transfer_statuses.get(room)
        if status is None or peer_handle not in status.expected_source_ranks:
            raise RuntimeError("unexpected native peer handle")
        expected_source_rank = status.expected_source_ranks[peer_handle]
        expected_generation = status.expected_source_generations.get(peer_handle)
        if expected_generation is None:
            raise RuntimeError("native peer generation is not bound to the room")
        with self._prefill_peer_lock:
            current_peer = self._prefill_peers_by_handle.get(peer_handle)
            is_quarantined = peer_handle in self._quarantined_remote_handles
        if (
            current_peer is None
            or current_peer.process_generation != expected_generation
            or is_quarantined
        ):
            raise RuntimeError("stale native peer generation")
        return status, expected_source_rank

    def _fail_notification_room(self, room: int, reason: str) -> None:
        """Fail one active room without resurrecting completed request state.

        :param room: Decoder-minted bootstrap room.
        :param reason: Terminal failure reason.
        """

        request_status = self.request_status.get(room)
        if request_status in (None, KVPoll.Failed, KVPoll.Success):
            return
        self.record_failure(room, reason)
        self.update_status(room, KVPoll.Failed)

    def _fail_rooms_bound_to_notification_peer(
        self,
        peer_handle: nixl_remote_agent_handle,
        reason: str,
    ) -> None:
        """Fail every active room that could own an unscoped peer message.

        :param peer_handle: Exact native notification source.
        :param reason: Failure reason for a current generation.
        """

        for room, status in list(self.transfer_statuses.items()):
            if peer_handle not in status.expected_source_ranks:
                continue
            try:
                self._validate_notification_peer(peer_handle, room)
            except RuntimeError as error:
                room_reason = f"Rejected NIXL notification: {error}"
            else:
                room_reason = reason
            self._fail_notification_room(room, room_reason)

    def _process_transfer_notification(
        self,
        peer_handle: nixl_remote_agent_handle,
        msg: bytes,
        components: List[str],
        room: int,
    ) -> None:
        """Validate and apply one native-handle-attributed notification.

        :param peer_handle: Exact native source handle resolved from ``reply_ep``.
        :param msg: Raw notification payload.
        :param components: Underscore-delimited notification fields.
        :param room: Decoder-minted bootstrap room.
        :raises RuntimeError: If native or serialized source ownership conflicts.
        :raises ValueError: If numeric fields are malformed.
        :raises IndexError: If required fields are absent.
        """

        status, expected_source_rank = self._validate_notification_peer(
            peer_handle, room
        )

        tag = components[1]
        if tag == "kv":
            if len(components) not in (5, 8):
                raise RuntimeError(f"malformed KV notification: {msg!r}")
            chunk_id = int(components[2])
            is_last_value = int(components[3])
            source_rank = int(components[4])
            self._validate_notification_source_rank(expected_source_rank, source_rank)
            if is_last_value not in (0, 1):
                raise RuntimeError("KV notification has invalid last-chunk marker")
            is_last_chunk = bool(is_last_value)
            if len(components) == 8:
                if components[5] != "part":
                    raise RuntimeError(f"malformed KV-part notification: {msg!r}")
                self._track_kv_part_arrival(
                    room,
                    chunk_id,
                    is_last_chunk,
                    source_rank,
                    int(components[6]),
                    int(components[7]),
                )
                return
            self._track_kv_arrival(room, chunk_id, is_last_chunk, source_rank)
            return

        if tag == "stg":
            if len(components) != 12 or components[8] != "part":
                raise RuntimeError(f"malformed staging notification: {msg!r}")
            source_rank = int(components[4])
            self._validate_notification_source_rank(expected_source_rank, source_rank)
            self._handle_stg_notification(components, room)
            return

        if tag == "aux":
            is_no_kv_marker = len(components) == 4 and components[2] == "nokv"
            if is_no_kv_marker:
                source_rank = int(components[3])
                self._validate_notification_source_rank(
                    expected_source_rank, source_rank
                )
                self._handle_aux_notification(
                    room,
                    components,
                    owns_aux=peer_handle == status.canonical_aux_source,
                )
                return
            if len(components) != 2:
                raise RuntimeError(f"malformed auxiliary notification: {msg!r}")
            if peer_handle != status.canonical_aux_source:
                raise RuntimeError(
                    "auxiliary completion came from a noncanonical writer"
                )
            self._handle_aux_notification(room, components, owns_aux=True)
            return

        if tag == "state":
            if len(components) != 4:
                raise RuntimeError(f"malformed state notification: {msg!r}")
            source_rank = int(components[2])
            self._validate_notification_source_rank(expected_source_rank, source_rank)
            state_index = int(components[3])
            status.received_state_components.add((source_rank, state_index))
            return

        if tag == "failure":
            if len(components) != 3:
                raise RuntimeError(f"malformed failure notification: {msg!r}")
            source_rank = int(components[2])
            self._validate_notification_source_rank(expected_source_rank, source_rank)
            self.record_failure(
                room,
                f"Prefill source rank {source_rank} reported transfer failure",
            )
            self.update_status(room, KVPoll.Failed)
            return

        raise RuntimeError(f"unknown notification tag: {tag!r}")

    @staticmethod
    def _validate_notification_source_rank(
        expected_source_rank: int, claimed_source_rank: int
    ) -> None:
        """Validate a payload rank against its native handle binding.

        :param expected_source_rank: Rank bound to the native peer handle.
        :param claimed_source_rank: Rank serialized in the notification payload.
        :raises RuntimeError: If the serialized rank spoofs another writer.
        """

        if claimed_source_rank != expected_source_rank:
            raise RuntimeError(
                "source-rank mismatch: "
                f"native peer owns {expected_source_rank}, payload claimed "
                f"{claimed_source_rank}"
            )

    def _handle_stg_notification(self, components: list[str], room: int) -> None:
        """Handle one independently posted staging-RDMA completion.

        :param components: Parsed staging notification fields.
        :param room: Decoder-minted bootstrap room.
        """
        chunk_id = int(components[2])
        is_last_value = int(components[3])
        if is_last_value not in (0, 1):
            raise RuntimeError("staging notification has invalid last-chunk marker")
        is_last_chunk = bool(is_last_value)
        source_rank = int(components[4])
        chunk_idx = int(components[5])
        page_start = int(components[6])
        num_pages = int(components[7])
        part_idx = int(components[9])
        num_parts = int(components[10])
        agent_name = components[11]
        if agent_name != self.agent.name:
            raise RuntimeError("staging notification named a different decoder agent")
        if not self._track_staging_part_arrival(
            room=room,
            chunk_id=chunk_id,
            is_last_chunk=is_last_chunk,
            source_rank=source_rank,
            chunk_idx=chunk_idx,
            page_start=page_start,
            num_pages=num_pages,
            agent_name=agent_name,
            part_idx=part_idx,
            num_parts=num_parts,
        ):
            return
        self._track_kv_arrival(room, chunk_id, is_last_chunk, source_rank)
        self._handle_staging_chunk_arrived(
            room, chunk_idx, page_start, num_pages, agent_name
        )

    def _track_staging_part_arrival(
        self,
        *,
        room: int,
        chunk_id: int,
        is_last_chunk: bool,
        source_rank: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        agent_name: str,
        part_idx: int,
        num_parts: int,
    ) -> bool:
        """Retain exact staging-part ownership until one chunk is complete.

        :param room: Decoder-minted bootstrap room.
        :param chunk_id: Source chunk sequence number.
        :param is_last_chunk: Whether this is the source's final chunk.
        :param source_rank: Authenticated source writer rank.
        :param chunk_idx: Decode staging-ring allocation index.
        :param page_start: Destination request-page offset.
        :param num_pages: Number of destination pages covered.
        :param agent_name: Bound decoder NIXL agent name.
        :param part_idx: Zero-based bounded-transfer index.
        :param num_parts: Exact number of bounded transfers for the chunk.
        :returns: Whether every part has arrived exactly once.
        :raises RuntimeError: If part identity, geometry, or cardinality conflicts.
        """

        if num_parts <= 0:
            raise RuntimeError("staging part count must be positive")
        if part_idx < 0 or part_idx >= num_parts:
            raise RuntimeError(
                f"staging part index out of range: part={part_idx}, "
                f"num_parts={num_parts}"
            )
        if chunk_idx < 0 or page_start < 0 or num_pages <= 0:
            raise RuntimeError("staging part has invalid chunk geometry")

        key = (source_rank, chunk_id)
        status = self.transfer_statuses[room]
        if key in status.completed_staging_chunks:
            raise RuntimeError("duplicate staging part for a completed chunk")
        receipt = status.staging_parts_per_source.get(key)
        if receipt is None:
            receipt = _StagingPartReceipt(
                is_last_chunk=is_last_chunk,
                chunk_idx=chunk_idx,
                page_start=page_start,
                num_pages=num_pages,
                agent_name=agent_name,
                num_parts=num_parts,
            )
            status.staging_parts_per_source[key] = receipt
        elif (
            receipt.is_last_chunk != is_last_chunk
            or receipt.chunk_idx != chunk_idx
            or receipt.page_start != page_start
            or receipt.num_pages != num_pages
            or receipt.agent_name != agent_name
            or receipt.num_parts != num_parts
        ):
            raise RuntimeError("staging parts disagree on immutable chunk metadata")
        if part_idx in receipt.received_parts:
            raise RuntimeError("duplicate staging part")
        receipt.received_parts.add(part_idx)
        if len(receipt.received_parts) != receipt.num_parts:
            return False

        status.staging_parts_per_source.pop(key)
        status.completed_staging_chunks.add(key)
        return True

    def _handle_aux_notification(
        self, room: int, components: List[str], *, owns_aux: bool
    ) -> None:
        """Handle an aux notification and trigger last scatter if staging is complete.

        Notification tag layouts:
          aux:         {room}_aux                              -> 2 fields
          aux (nokv):  {room}_aux_nokv_{source_rank}            -> 4 fields
                       (decode-side radix cache hit; this writer sent
                       no KV pages, so its expected chunk count is zero)

        :param room: Decoder-minted bootstrap room.
        :param components: Parsed notification fields.
        :param owns_aux: Whether the exact native source is the canonical aux writer.
        """
        if owns_aux:
            self.transfer_statuses[room].received_aux = True
        # main's "nokv" marker (decode-side radix cache hit, see #19746).
        if len(components) > 3 and components[2] == "nokv":
            source_rank = int(components[3])
            self.transfer_statuses[room].expected_kvs_per_source[source_rank] = 0
        if self.transfer_statuses[room].num_source_writers_expected is None:
            self.transfer_statuses[room].num_source_writers_expected = (
                self.required_prefill_response_num_table.get(room, 1)
            )
        if (
            self.enable_staging
            and self._staging_handler is not None
            and self._staging_handler.is_staging_room(room)
        ):
            self._maybe_submit_last_scatter(room)

    def _track_kv_arrival(
        self, room: int, chunk_id: int, is_last_chunk: bool, source_rank: int
    ):
        """Update transfer status tracking for a kv chunk arrival."""
        self.transfer_statuses[room].received_kvs_per_source[source_rank].add(chunk_id)
        if is_last_chunk:
            self.transfer_statuses[room].expected_kvs_per_source[source_rank] = (
                chunk_id + 1
            )
            if self.transfer_statuses[room].num_source_writers_expected is None:
                self.transfer_statuses[room].num_source_writers_expected = (
                    self.required_prefill_response_num_table.get(room, 1)
                )
            if (
                self.enable_staging
                and self._staging_handler is not None
                and self._staging_handler.is_staging_room(room)
            ):
                self._maybe_submit_last_scatter(room)

    def _track_kv_part_arrival(
        self,
        room: int,
        chunk_id: int,
        is_last_chunk: bool,
        source_rank: int,
        part_idx: int,
        num_parts: int,
    ):
        """Track one segment of a mixed-memory KV transfer."""
        if num_parts <= 1:
            self._track_kv_arrival(room, chunk_id, is_last_chunk, source_rank)
            return
        if part_idx < 0 or part_idx >= num_parts:
            raise RuntimeError(
                f"NIXL KV part index out of range for room={room}, "
                f"chunk={chunk_id}, source_rank={source_rank}: part={part_idx}, "
                f"num_parts={num_parts}"
            )

        key = (source_rank, chunk_id)
        status = self.transfer_statuses[room]
        if status.received_kv_parts_per_source is None:
            status.received_kv_parts_per_source = defaultdict(set)
        if status.expected_kv_parts_per_source is None:
            status.expected_kv_parts_per_source = {}
        expected = status.expected_kv_parts_per_source.setdefault(key, num_parts)
        if expected != num_parts:
            raise RuntimeError(
                f"NIXL KV part count mismatch for room={room}, chunk={chunk_id}, "
                f"source_rank={source_rank}: got {num_parts}, expected {expected}"
            )
        parts = status.received_kv_parts_per_source[key]
        parts.add(part_idx)
        if len(parts) == num_parts:
            status.received_kv_parts_per_source.pop(key, None)
            status.expected_kv_parts_per_source.pop(key, None)
            self._track_kv_arrival(room, chunk_id, is_last_chunk, source_rank)

    def _handle_staging_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        agent_name: str,
    ):
        """Process a staging chunk arrival via RDMA notification."""
        handler = self._staging_handler
        if handler is None:
            return
        handler.handle_chunk_arrived(
            room,
            chunk_idx,
            page_start,
            num_pages,
            agent_name,
            self._chunk_writer_counts,
        )

    def _maybe_submit_last_scatter(self, room: int):
        """Check if all kv+aux transfers are done and submit last scatter if so."""
        status = self.transfer_statuses.get(room)
        if status is None:
            return
        if not status.received_aux:
            return
        if status.num_source_writers_expected is None:
            return
        if len(status.expected_kvs_per_source) < status.num_source_writers_expected:
            return
        for source_rank, expected in status.expected_kvs_per_source.items():
            if len(status.received_kvs_per_source[source_rank]) != expected:
                return
        handler = self._staging_handler
        if handler is not None and handler.is_staging_room(room):
            handler.submit_last_scatter_async(room)
            self._chunk_writer_counts.pop(room, None)

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _handle_abort_notification(self, msg: List[bytes]) -> bool:
        if not msg or msg[0] != b"ABORT":
            return False

        try:
            room_to_be_aborted = int(msg[1].decode("ascii"))
        except Exception as e:
            logger.debug(f"Ignoring malformed abort notification: {e}")
            return True

        if (
            room_to_be_aborted in self.request_status
            and self.check_status(room_to_be_aborted) != KVPoll.Success
        ):
            self.record_failure(
                room_to_be_aborted,
                "Aborted by decode-side abort notification.",
            )
            self.update_status(room_to_be_aborted, KVPoll.Failed)
            logger.debug(
                f"Received abort notification for room {room_to_be_aborted}, "
                f"marked as Failed"
            )
        else:
            logger.debug(
                f"Received abort notification for room {room_to_be_aborted}, "
                f"ignoring (already completed or unknown)"
            )

        # TODO: Define real ACK/deferred-release semantics if decode-side buffer
        # release needs to wait for prefill-side NIXL transfer quiescence.

        return True

    def _handle_packed_prefill_control(self, frames: list[bytes]) -> None:
        """Authenticate and dispatch one decode-to-prefill control message.

        :param frames: Exact PACKED_V4 multipart frames.
        """

        try:
            agent_name, process_generation, message = decode_packed_control_frames(
                frames
            )
            registration = self.decode_kv_args_table.get(agent_name)
            if registration is None or registration.remote_handle is None:
                raise RuntimeError("packed control references an unknown decoder peer")
            if registration.process_generation != process_generation:
                raise RuntimeError(
                    "packed control references a stale decoder generation"
                )
            if (
                registration.packed_transfer_protocol != PACKED_KV_TRANSFER_PROTOCOL
                or registration.prepared_grant_protocol
                != PACKED_PREPARED_GRANT_PROTOCOL
                or registration.packed_advertisement is None
            ):
                raise RuntimeError("decoder peer did not advertise the packed runtime")
            with self._prefill_peer_lock:
                if registration.remote_handle in self._quarantined_remote_handles:
                    raise RuntimeError(
                        "packed control references a quarantined decoder"
                    )
            runtime = self._packed_prefill_runtime
            if runtime is None:
                raise RuntimeError("packed NIXL prefill runtime is unavailable")
            runtime.handle_control(
                PackedPeerIdentity(
                    agent_name=registration.agent_name,
                    agent_generation=uuid.UUID(registration.process_generation).bytes,
                ),
                message,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "Rejected packed decoder control message:\n%s",
                traceback.format_exc(),
            )

    def _send_packed_control_frames(
        self,
        endpoint: str,
        port: int,
        frames: list[bytes],
    ) -> None:
        """Send one serialized packed control message to a decoder process.

        :param endpoint: Decoder host or address.
        :param port: Decoder PULL socket port.
        :param frames: Closed packed multipart message.
        """

        address = NetworkAddress(endpoint, port)
        with self._packed_control_send_lock:
            socket = self._connect(address.to_tcp(), is_ipv6=address.is_ipv6)
            socket.send_multipart(frames)

    def _start_bootstrap_thread(self):
        if self._terminal_bootstrap_thread is not None:
            raise RuntimeError("NIXL bootstrap listener is already started")

        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )

                if (
                    self.terminal_startup_binding is not None
                    and not self._terminal_runtime_activated.is_set()
                    and not (
                        len(waiting_req_bytes) >= 2
                        and waiting_req_bytes[0] == GUARD
                        and waiting_req_bytes[1] == b"None"
                    )
                ):
                    logger.error(
                        "Rejected NIXL runtime traffic before terminal peer commit"
                    )
                    continue

                # Staging: decode reports consumption watermark back to prefill
                if waiting_req_bytes[0] == b"WATERMARK":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_watermark_msg,
                        )

                        handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                    continue

                # Staging: decode replies with allocated staging offset
                if waiting_req_bytes[0] == b"STAGING_RSP":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_staging_rsp,
                        )

                        handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                    continue

                if waiting_req_bytes[0] == PACKED_CONTROL_TAG:
                    self._handle_packed_prefill_control(waiting_req_bytes)
                    continue

                if self._handle_abort_notification(waiting_req_bytes):
                    continue

                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    try:
                        registration = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                        self._add_remote_peer(registration)
                    except Exception:
                        logger.error(
                            "Rejected decoder NIXL peer registration:\n%s",
                            traceback.format_exc(),
                        )
                        continue
                    logger.debug(
                        "Registered KVArgs from %s successfully",
                        registration.agent_name,
                    )
                    continue
                room = int(room)
                try:
                    transfer_info = TransferInfo.from_zmq(waiting_req_bytes)
                    agent_name = transfer_info.agent_name
                    peer_info = self.decode_kv_args_table.get(agent_name)
                    if peer_info is None or peer_info.remote_handle is None:
                        raise RuntimeError(
                            "Transfer references an unloaded decoder peer"
                        )
                    if transfer_info.process_generation != peer_info.process_generation:
                        raise RuntimeError(
                            "Transfer references a stale decoder process generation"
                        )
                    if (
                        transfer_info.endpoint != peer_info.endpoint
                        or transfer_info.dst_port != peer_info.dst_port
                    ):
                        raise RuntimeError(
                            "Transfer route conflicts with registered decoder peer"
                        )
                    packed_plan = transfer_info.packed_plan
                    if packed_plan is not None:
                        advertisement = peer_info.packed_advertisement
                        if advertisement is None:
                            raise RuntimeError(
                                "Packed transfer references a legacy decoder registration"
                            )
                        if packed_plan.key.room_id != room:
                            raise RuntimeError("Packed transfer plan room is stale")
                        if (
                            packed_plan.destination_process_generation
                            != uuid.UUID(peer_info.process_generation).bytes
                        ):
                            raise RuntimeError(
                                "Packed transfer targets another decoder generation"
                            )
                        if (
                            packed_plan.native_route_digest
                            != advertisement.visibility_policy_digest
                        ):
                            raise RuntimeError(
                                "Packed transfer visibility policy is stale"
                            )
                        if (
                            packed_plan.runtime_cohort_digest
                            != advertisement.runtime_cohort_digest
                        ):
                            raise RuntimeError(
                                "Packed transfer runtime cohort is stale"
                            )
                    self.terminal_source_identity_plan(transfer_info)
                except Exception:
                    reason = (
                        "Rejected generation-bound decoder transfer metadata:\n"
                        f"{traceback.format_exc()}"
                    )
                    self.record_failure(room, reason)
                    self.update_status(room, KVPoll.Failed)
                    continue
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = transfer_info
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.resolve_kv_replica_factor(self.transfer_infos[room])
                    self.req_to_decode_prefix_len[room] = next(
                        (
                            info.decode_prefix_len
                            for info in self.transfer_infos[room].values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        thread = threading.Thread(
            target=bootstrap_thread,
            name="nixl-prefill-control",
            daemon=True,
        )
        self._terminal_bootstrap_thread = thread
        thread.start()

    def _handle_terminal_source_publication_receipt(
        self,
        frames: tuple[bytes, ...],
    ) -> None:
        """Deliver one direct canonical publication outcome to source serving.

        :param frames: Exact startup-matrix-bound multipart message.
        """

        control = self._terminal_source_publication_control
        if control is None:
            raise RuntimeError("source publication control is unavailable")
        control.receive_frames(frames)


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.has_sent = False
        self.chunk_id = 0
        self._send_failed = False
        self._send_error: Optional[Exception] = None
        self._transfer_start_time: Optional[float] = None

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        producer_event: torch.cuda.Event | None = None,
    ):
        if self._send_failed:
            return

        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            self.chunk_id,
            self.aux_index,
            state_indices,
            producer_event,
        )
        self._record_transfer_indices(kv_indices, state_indices)
        self.chunk_id += 1
        if is_last_chunk:
            self.has_sent = True

    def poll(self) -> KVPoll:
        if self._send_failed:
            return KVPoll.Failed  # type: ignore
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if (
            status == KVPoll.Success
            and self._transfer_start_time is not None
            and self._transfer_metric.transfer_latency_s is None
        ):
            self._transfer_metric.transfer_latency_s = (
                time.perf_counter() - self._transfer_start_time
            )
        return status

    def clear(self) -> None:
        super().clear()
        if self.kv_mgr.enable_staging and self.kv_mgr._staging_ctx is not None:
            self.kv_mgr._staging_ctx.prefetched_rooms.discard(self.bootstrap_room)
            self.kv_mgr._staging_ctx.prefetch_requested = {
                key
                for key in self.kv_mgr._staging_ctx.prefetch_requested
                if key[0] != self.bootstrap_room
            }

    def failure_exception(self):
        exc = self.kv_mgr.exceptions.pop(self.bootstrap_room, None)
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)

        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self._send_failed = True

        self.clear()

        if self._send_error is not None:
            raise self._send_error
        if exc is not None:
            raise exc
        if failure_reason is not None:
            raise KVTransferError(self.bootstrap_room, failure_reason)
        raise KVTransferError(
            self.bootstrap_room, "NIXL KVSender Exception", is_from_another_rank=True
        )


class NixlKVReceiver(CommonKVReceiver):
    _terminal_authority_initialized: bool

    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time = None
        self.prefill_peers: list[_NixlPrefillPeer] = []
        self._terminal_authority_initialized = False

    def init_from_terminal_authority(
        self,
        authority: TerminalPrefillRequestAuthority,
    ) -> None:
        """Initialize from PREPARE-retained source authority only.

        :param authority: Exact immutable NIXL source projection.
        :raises TerminalPrefillAuthorityMismatch: If manager, request, or source
            generation differs from preparation.
        """

        if self._terminal_authority_initialized or len(self.prefill_peers) > 0:
            raise TerminalPrefillAuthorityMismatch(
                "terminal receiver authority is already initialized"
            )
        self._validate_terminal_authority_identity(authority)

        assert type(authority) is NixlTerminalPrefillRequestAuthority
        self._terminal_authority_initialized = True
        topology = authority.topology
        self.prefill_dp_rank = authority.prefill_dp_rank
        self.prefill_info = PrefillServerInfo(
            attn_tp_size=topology.source_tp_size,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=self.kv_mgr.kv_args.page_size,
            kv_cache_dtype=None,
            follow_bootstrap_room=True,
            target_tp_rank=topology.target_tp_rank,
            target_tp_ranks=list(topology.target_tp_ranks),
            target_cp_ranks=[0],
            target_pp_ranks=[0],
            required_dst_info_num=topology.required_dst_info_num,
            required_prefill_response_num=topology.required_prefill_response_num,
        )
        self.target_tp_rank = topology.target_tp_rank
        self.target_tp_ranks = list(topology.target_tp_ranks)
        self.target_cp_ranks = [0]
        self.target_pp_ranks = [0]
        self.required_dst_info_num = topology.required_dst_info_num
        self.required_prefill_response_num = topology.required_prefill_response_num
        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )
        self.require_staging = (
            self.kv_mgr.enable_staging
            and topology.source_tp_size != self.kv_mgr.attn_tp_size
        )
        self.prefill_peers = list(authority.peers)
        self.bootstrap_infos = [
            {
                "rank_ip": peer.control_endpoint.host,
                "rank_port": peer.control_endpoint.port,
                "attn_dp_rank": peer.attn_dp_rank,
                "attn_cp_rank": peer.attn_cp_rank,
                "attn_tp_rank": peer.attn_tp_rank,
                "pp_rank": peer.pp_rank,
                "transfer_source_rank": peer.transfer_source_rank,
                "process_generation": peer.process_generation,
                "nixl_agent_name": peer.agent_name,
                "nixl_agent_metadata_sha256": peer.metadata_sha256,
                "transport_protocol": NIXL_BOOTSTRAP_PEER_PROTOCOL,
                "is_dummy": False,
            }
            for peer in authority.peers
        ]
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _validate_terminal_authority_identity(
        self,
        authority: TerminalPrefillRequestAuthority,
    ) -> None:
        """Validate immutable source identity against current process authority.

        :param authority: Candidate generation-bound NIXL authority.
        :raises TerminalPrefillAuthorityMismatch: If the authority is stale,
            malformed, or quarantined.
        """

        if type(authority) is not NixlTerminalPrefillRequestAuthority:
            raise TerminalPrefillAuthorityMismatch(
                "receiver received another terminal prefill authority type"
            )
        if self.kv_mgr.terminal_startup_binding != authority.startup_binding:
            raise TerminalPrefillAuthorityMismatch(
                "terminal startup generation changed after preparation"
            )
        if self.bootstrap_addr != authority.bootstrap_addr:
            raise TerminalPrefillAuthorityMismatch(
                "receiver bootstrap address differs from preparation"
            )
        if any(
            peer.handle in self.kv_mgr._quarantined_remote_handles
            for peer in authority.peers
        ):
            raise TerminalPrefillAuthorityMismatch(
                "prepared terminal prefill peer was quarantined"
            )

    def _load_bootstrap_peers(self) -> None:
        """Load every selected prefill writer before this receiver is ready.

        :raises RuntimeError: If any route lacks exact native peer authority.
        """

        peers: list[_NixlPrefillPeer] = []
        handles: set[nixl_remote_agent_handle] = set()
        source_ranks: set[int] = set()
        with self.kv_mgr._prefill_peer_lock:
            for bootstrap_info in self.bootstrap_infos:
                if self.kv_mgr.terminal_startup_binding is None:
                    peer = self.kv_mgr._load_prefill_peer(
                        self.bootstrap_addr, bootstrap_info
                    )
                else:
                    peer = self.kv_mgr._resolve_terminal_prefill_peer(
                        self.bootstrap_addr,
                        bootstrap_info,
                    )
                if peer.handle in handles:
                    raise RuntimeError("Duplicate native prefill peer in writer cohort")
                if peer.transfer_source_rank in source_ranks:
                    raise RuntimeError("Duplicate prefill source rank in writer cohort")
                peers.append(peer)
                handles.add(peer.handle)
                source_ranks.add(peer.transfer_source_rank)
        if len(peers) == 0:
            raise RuntimeError("NIXL prefill writer cohort is empty")
        self.prefill_peers = peers

    def _setup_bootstrap_infos(self):
        super()._setup_bootstrap_infos()
        if self.conclude_state == KVPoll.Failed or self.bootstrap_infos is None:
            return
        try:
            self._load_bootstrap_peers()
        except Exception:
            reason = (
                "Failed to load the generation-bound NIXL prefill writer cohort:\n"
                f"{traceback.format_exc()}"
            )
            self.kv_mgr.record_failure(self.bootstrap_room, reason)
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)

    def build_packed_control_routes(
        self,
        controller: PackedNixlDecodeController,
    ) -> tuple[PackedControlSender, ...]:
        """Build the complete generation-bound source-writer route cohort.

        :param controller: Exact process-lifetime decode controller.
        :returns: Canonically ordered packed control senders.
        :raises RuntimeError: If bootstrap and native peer state diverged.
        """

        if self.bootstrap_infos is None:
            raise RuntimeError("packed control routes require bootstrap metadata")
        if len(self.prefill_peers) != len(self.bootstrap_infos):
            raise RuntimeError("packed control writer cohort changed after bootstrap")
        routes = []
        for bootstrap_info, peer in zip(
            self.bootstrap_infos,
            self.prefill_peers,
            strict=True,
        ):
            if bootstrap_info["is_dummy"]:
                continue
            socket, socket_lock = self._connect_to_bootstrap_server(bootstrap_info)

            def send_frames(
                frames: list[bytes],
                *,
                owned_socket: zmq.Socket = socket,
                owned_lock: threading.Lock = socket_lock,
            ) -> None:
                with owned_lock:
                    owned_socket.send_multipart(frames)

            routes.append(
                controller.build_control_sender(
                    StagingWriterId(
                        transfer_source_rank=peer.transfer_source_rank,
                        source_attn_tp_rank=peer.attn_tp_rank,
                        source_pp_rank=peer.pp_rank,
                        source_cp_rank=peer.attn_cp_rank,
                    ),
                    send_frames,
                )
            )
        if len(routes) == 0:
            raise RuntimeError("packed control writer cohort is empty")
        return tuple(routes)

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        *,
        packed_plan: PackedAuxiliaryPlan | None = None,
        terminal_source_plan_payload: bytes | None = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        if terminal_source_plan_payload is not None:
            if type(terminal_source_plan_payload) is not bytes:
                raise TypeError("terminal_source_plan_payload must be bytes")
            if packed_plan is None:
                raise ValueError(
                    "terminal source authority requires packed request metadata"
                )
            terminal_source_plan = decode_packed_terminal_source_plan(
                terminal_source_plan_payload
            )
            require_source_plan_request_key(terminal_source_plan, packed_plan.key)

        expected_state_indices: set[int] = set()
        if state_indices is not None:
            expected_state_indices = {
                state_index
                for state_index, component_indices in enumerate(state_indices)
                if component_indices is not None and len(component_indices) > 0
            }
        self.kv_mgr.transfer_statuses[self.bootstrap_room].expected_state_indices = (
            expected_state_indices
        )
        if len(self.prefill_peers) != len(self.bootstrap_infos):
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "NIXL request source-writer cohort changed after bootstrap",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        active_peers = [
            peer
            for bootstrap_info, peer in zip(
                self.bootstrap_infos, self.prefill_peers, strict=True
            )
            if not bootstrap_info["is_dummy"]
        ]
        if len(active_peers) == 0:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "NIXL request has no active prefill source writers",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        transfer_status = self.kv_mgr.transfer_statuses[self.bootstrap_room]
        transfer_status.expected_source_ranks = {
            peer.handle: peer.transfer_source_rank for peer in active_peers
        }
        transfer_status.expected_source_generations = {
            peer.handle: peer.process_generation for peer in active_peers
        }
        if len(transfer_status.expected_source_ranks) != len(active_peers):
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "NIXL request source-writer cohort contains duplicate handles",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        if packed_plan is None:
            canonical_aux_sources = [
                peer
                for peer in active_peers
                if peer.attn_tp_rank == 0
                and peer.attn_cp_rank == 0
                and peer.pp_rank == 0
            ]
        else:
            canonical_aux_sources = [
                peer
                for peer in active_peers
                if StagingWriterId(
                    transfer_source_rank=peer.transfer_source_rank,
                    source_attn_tp_rank=peer.attn_tp_rank,
                    source_pp_rank=peer.pp_rank,
                    source_cp_rank=peer.attn_cp_rank,
                )
                == packed_plan.canonical_writer_id
            ]
        if len(canonical_aux_sources) != 1:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "NIXL request does not have exactly one canonical auxiliary writer",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        transfer_status.canonical_aux_source = canonical_aux_sources[0].handle

        # Register staging room bootstrap info for staging handler
        self.chunk_staging_infos = []
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            packed_state_indices = (
                pack_int_lists(
                    [(idx if idx is not None else []) for idx in state_indices], "i"
                )
                if not is_dummy and state_indices is not None
                else b""
            )
            metadata_frames = [
                GUARD,
                str(self.bootstrap_room).encode("ascii"),
                self.kv_mgr.local_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.kv_mgr.agent.name.encode("ascii"),
                kv_indices.tobytes() if not is_dummy else b"",
                str(aux_index).encode("ascii"),
                str(self.required_dst_info_num).encode("ascii"),
                packed_state_indices,
                str(decode_prefix_len or 0).encode("ascii"),
                self.kv_mgr.process_generation.encode("ascii"),
            ]
            if packed_plan is not None:
                if is_dummy:
                    raise RuntimeError("packed transfer does not support dummy writers")
                metadata_frames.append(encode_packed_message(packed_plan))
                if terminal_source_plan_payload is not None:
                    metadata_frames.append(terminal_source_plan_payload)
            try:
                with lock:
                    sock.send_multipart(metadata_frames)
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return status

        # Drain notifications before enforcing the waiting deadline. The decode
        # agent has no NIXL progress thread (num_threads=0), so incoming
        # completion notifications are only ingested here via
        # update_transfer_status(); a completion queued by NIXL at/after the
        # deadline would otherwise lose to the timeout purely by poll ordering.
        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_status(self.bootstrap_room) == KVPoll.Failed:
            self.conclude_state = KVPoll.Failed
            return self.conclude_state

        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore

        timeout_result = self._check_waiting_timeout()
        if timeout_result is not None:
            return timeout_result

        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self) -> bool:
        if self.kv_mgr.terminal_startup_binding is not None:
            try:
                self._load_bootstrap_peers()
            except Exception:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    "Failed to resolve the frozen NIXL prefill writer cohort:\n"
                    f"{traceback.format_exc()}",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
            return True

        try:
            self._load_bootstrap_peers()
        except Exception:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "Failed to load the generation-bound NIXL prefill writer cohort:\n"
                f"{traceback.format_exc()}",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return False

        registration_frames = self.kv_mgr._decode_registration_frames()
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            try:
                with lock:
                    sock.send_multipart(registration_frames)
            except zmq.ZMQError:
                rank_ip = bootstrap_info.get("rank_ip")
                rank_port = bootstrap_info.get("rank_port")
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"_register_kv_args to prefill {rank_ip}:{rank_port} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
        return True

    def clear(self) -> None:
        """Release NIXL receiver bookkeeping for this room."""

        bootstrap_room = self.bootstrap_room
        super().clear()
        if bootstrap_room is None:
            return
        self.kv_mgr.transfer_statuses.pop(bootstrap_room, None)
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(bootstrap_room)

    def failure_exception(self):
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "NIXL KVReceiver Exception"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    """NIXL bootstrap listener with optional terminal cohort admission."""

    _terminal_startup_registry: TerminalStartupCohortRegistry | None

    def __init__(
        self,
        host: str,
        port: int,
        terminal_startup_registry: TerminalStartupCohortRegistry | None = None,
    ):
        """Construct the NIXL bootstrap listener.

        :param host: Listener host.
        :param port: Listener port.
        :param terminal_startup_registry: Exact immutable startup registry.
        """

        if (
            terminal_startup_registry is not None
            and type(terminal_startup_registry) is not TerminalStartupCohortRegistry
        ):
            raise TypeError(
                "terminal_startup_registry must be TerminalStartupCohortRegistry"
            )
        self._terminal_startup_registry = terminal_startup_registry
        additional_post_routes: tuple[BootstrapPostRoute, ...] = ()
        if terminal_startup_registry is not None:
            terminal_startup_handler: BootstrapRouteHandler = functools.partial(
                handle_terminal_startup_join,
                terminal_startup_registry,
            )
            additional_post_routes = (
                BootstrapPostRoute(
                    path=TERMINAL_STARTUP_ROUTE,
                    handler=terminal_startup_handler,
                ),
                BootstrapPostRoute(
                    path=TERMINAL_NIXL_SOURCE_ROSTER_ROUTE,
                    handler=self._handle_terminal_source_roster,
                ),
            )
        super().__init__(
            host=host,
            port=port,
            additional_post_routes=additional_post_routes,
        )

    async def _handle_terminal_source_roster(
        self,
        request: web.Request,
    ) -> web.Response:
        """Return every matrix-authenticated source route to one startup rank.

        Route-table readiness waits off the HTTP event loop. The response is
        emitted once from the frozen source topology and cannot select a
        request-era subset.

        :param request: Exact decoder rank advertisement.
        :returns: Canonical complete source roster or bounded failure evidence.
        """

        registry = self._terminal_startup_registry
        if registry is None:
            return web.Response(text="terminal startup is not configured", status=404)
        try:
            requester = decode_terminal_startup_rank_advertisement(await request.read())
            matrix = registry.sealed_matrix_for(requester)
            await asyncio.to_thread(
                self.wait_until_ready,
                registry.timeout_seconds,
            )
            source_ranks = tuple(
                rank for rank in matrix.ranks if rank.role is TerminalOwnerRole.SOURCE
            )
            async with self.lock:
                routes = tuple(
                    self._terminal_source_route(rank) for rank in source_ranks
                )
            roster = TerminalNixlSourceRoster(
                matrix_sha256=matrix.digest,
                requester_service_id=requester.service_id,
                requester_tensor_parallel_rank=requester.tensor_parallel_rank,
                requester_process_generation=requester.process_generation,
                routes=routes,
            )
            roster.require_matrix(
                matrix,
                requester,
                NIXL_BOOTSTRAP_PEER_PROTOCOL,
            )
        except TerminalStartupCohortError as error:
            return web.Response(text=str(error), status=409)
        except Exception:
            reason = (
                "terminal source-route roster failed after cohort sealing:\n"
                f"{traceback.format_exc()}"
            )
            registry.fail(reason)
            logger.error(reason)
            return web.Response(
                text="terminal source-route roster failed",
                status=500,
            )
        return web.Response(
            body=encode_terminal_nixl_source_roster(roster),
            content_type="application/json",
            status=200,
        )

    def _terminal_source_route(
        self,
        rank: TerminalStartupRankAdvertisement,
    ) -> TerminalNixlSourceRoute:
        """Resolve one exact matrix row from the frozen transfer route table.

        :param rank: Source rank from the sealed startup matrix.
        :returns: Complete native source route.
        :raises RuntimeError: If the route table differs from startup identity.
        """

        if rank.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("terminal source route requires a source rank")
        try:
            route = self.prefill_port_table[0][0][rank.tensor_parallel_rank][0]
        except KeyError as error:
            raise RuntimeError("terminal source route table is incomplete") from error
        try:
            metadata = decode_nixl_agent_metadata(route.nixl_agent_metadata)
            agent_name = validate_nixl_agent_name(route.nixl_agent_name)
        except ValueError as error:
            raise RuntimeError("terminal source route identity is invalid") from error
        if route.nixl_agent_metadata_sha256 != hashlib.sha256(metadata).hexdigest():
            raise RuntimeError("terminal source route metadata digest mismatch")
        process_generation = self._canonical_route_generation(route.process_generation)
        return TerminalNixlSourceRoute(
            service_id=rank.service_id,
            tensor_parallel_rank=rank.tensor_parallel_rank,
            tensor_parallel_size=rank.tensor_parallel_size,
            process_generation=uuid.UUID(process_generation).bytes,
            nixl_agent_name=agent_name,
            nixl_agent_metadata=metadata,
            rank_ip=route.rank_ip,
            rank_port=route.rank_port,
            attn_dp_rank=route.attn_dp_rank,
            attn_cp_rank=route.attn_cp_rank,
            attn_tp_rank=route.attn_tp_rank,
            pp_rank=route.pp_rank,
            transfer_source_rank=(
                -1 if route.transfer_source_rank is None else route.transfer_source_rank
            ),
            transport_protocol=(
                "" if route.transport_protocol is None else route.transport_protocol
            ),
        )

    @staticmethod
    def _canonical_route_generation(value: object) -> str:
        """Validate one canonical route process generation.

        :param value: Candidate route generation.
        :returns: Canonical UUID string.
        :raises RuntimeError: If the generation is malformed.
        """

        if type(value) is not str:
            raise RuntimeError("terminal source route generation is absent")
        try:
            generation = uuid.UUID(value)
        except ValueError as error:
            raise RuntimeError("terminal source route generation is invalid") from error
        if generation.int == 0 or str(generation) != value:
            raise RuntimeError("terminal source route generation is not canonical")
        return value
