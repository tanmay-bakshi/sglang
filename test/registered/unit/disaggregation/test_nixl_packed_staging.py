import concurrent.futures
import contextlib
import dataclasses
import enum
import gc
import inspect
import sys
import textwrap
import threading
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedLayoutSpec,
    PackedPrepare,
    PackedProtocolError,
    PackedProtocolState,
    PackedReady,
    PackedTopology,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityAction,
    PackedWriterVisibilityEvidence,
    _PackedWriterOutcomeTicket,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingEndpointBufferBinding,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedComponentPages,
    PackedCopyExecutor,
    PackedCudaDeviceAttribute,
    PackedDestinationCapability,
    PackedDestinationOutcomeCoordinator,
    PackedDestinationRegistration,
    PackedDestinationRouteBinding,
    PackedDestinationVisibilityAction,
    PackedDestinationVisibilityActionExecutor,
    PackedDestinationVisibilityError,
    PackedDestinationVisibilityPolicy,
    PackedDestinationVisibilityProof,
    PackedDirectCudaEventAuthority,
    PackedGpuDirectFlushOptions,
    PackedGpuDirectFlushScope,
    PackedGpuDirectFlushTarget,
    PackedGpuDirectWritesOrdering,
    PackedIntervalLeaseAllocator,
    PackedNixlRuntimeArtifactCohort,
    PackedNixlRuntimeArtifactIdentity,
    PackedNixlRuntimeRoot,
    PackedPeerIdentity,
    PackedReadyCoordinator,
    PackedReadyError,
    PackedRegistrationQuarantine,
    PackedSourceTransfer,
    PackedStagingArena,
    PackedTransferLane,
    PackedTransferLaneState,
    PackedTransportPath,
    active_destination_page_arrays,
    build_component_buffer_registry,
    build_decode_spec,
    build_nixl_ucx_lane_identifier,
    build_prefill_chunk,
    derive_destination_visibility_proof,
    derive_nixl_ucx_runtime_artifact_components,
    writer_layout_for,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.nixl_owner_boundary import (
    NixlTerminalOwnerBoundary,
)
from sglang.srt.disaggregation.utils import resolve_kv_layer_ids
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SWA_COMPONENT = StagingComponentId(state_index=0, state_type=StateType.SWA)
REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
AGENT_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
ARENA_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
KEY = PackedChunkKey(
    room_id=29,
    chunk_id=2,
    request_generation=REQUEST_GENERATION,
)


def test_packed_copy_executor_prioritizes_completion_progress() -> None:
    """Gather and scatter remain schedulable across model forwards."""

    source_stream = object()
    scatter_stream = object()
    with (
        patch(
            "sglang.srt.disaggregation.nixl.packed_staging.torch.device",
            return_value="cuda-device",
        ),
        patch(
            "sglang.srt.disaggregation.nixl.packed_staging.torch.cuda.set_device"
        ) as set_device,
        patch(
            "sglang.srt.disaggregation.nixl.packed_staging.torch.cuda.Stream",
            side_effect=(source_stream, scatter_stream),
        ) as stream,
    ):
        executor = PackedCopyExecutor(gpu_id=3)

    assert executor._source_stream is source_stream
    assert executor._scatter_stream is scatter_stream
    set_device.assert_called_once_with(3)
    assert stream.call_args_list == [
        call(device="cuda-device", priority=-5),
        call(device="cuda-device", priority=-5),
    ]


def test_packed_gather_waits_on_exact_producer_event() -> None:
    """Gather has no mutable-stream fallback after event ownership is sealed."""

    producer_event = object()
    source_stream = Mock()
    source_lane = MagicMock()
    transfer = SimpleNamespace(
        layout=object(),
        writer_id=object(),
        length_bytes=4096,
    )
    writer_layout = SimpleNamespace(
        length_bytes=4096,
        lease_offset=0,
        copy_groups=(),
    )
    executor = object.__new__(PackedCopyExecutor)
    executor._source_stream = source_stream

    with (
        patch(
            "sglang.srt.disaggregation.nixl.packed_staging.writer_layout_for",
            return_value=writer_layout,
        ),
        patch(
            "sglang.srt.disaggregation.nixl.packed_staging.torch.cuda.stream",
            return_value=contextlib.nullcontext(),
        ),
    ):
        length_bytes = executor.gather(
            transfer=transfer,
            source_lane=source_lane,
            producer_event=producer_event,
        )

    assert length_bytes == 4096
    source_stream.wait_event.assert_called_once_with(producer_event)
    source_stream.wait_stream.assert_not_called()


WRITERS = tuple(
    StagingWriterId(
        transfer_source_rank=rank,
        source_attn_tp_rank=rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )
    for rank in range(2)
)


class RecordingMemoryAgent:
    """CPU-only NIXL registration agent with cleanup fault injection."""

    completion_receipts: dict[object, object]
    deregistrations: list[object]
    fail_deregistration: bool
    name: str
    registrations: list[tuple[tuple[int, int, int, str], ...]]

    def __init__(self) -> None:
        """Initialize successful registration and cleanup."""

        self.completion_receipts = {}
        self.deregistrations = []
        self.fail_deregistration = False
        self.name = "prefill-agent"
        self.registrations = []

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Record one fake registration.

        :param addresses: Registered address descriptors.
        :param memory_kind: Expected VRAM memory kind.
        :returns: Unique opaque registration handle.
        """

        if memory_kind != "VRAM":
            raise ValueError(f"unexpected memory kind {memory_kind}")
        snapshot = tuple(addresses)
        self.registrations.append(snapshot)
        return ("registration", len(self.registrations))

    def deregister_memory(self, registration: object) -> None:
        """Record or fail one fake deregistration.

        :param registration: Opaque fake handle.
        """

        if self.fail_deregistration:
            raise RuntimeError("injected deregistration failure")
        self.deregistrations.append(registration)

    def take_xfer_completion_receipt(self, handle: object) -> object | None:
        """Take one exact fake native completion receipt.

        :param handle: Exact fake transfer handle.
        :returns: One-shot receipt, or ``None`` while pending.
        """

        return self.completion_receipts.pop(handle, None)


class RecordingTerminalOwner(NixlTerminalOwnerBoundary):
    """Deterministic direct-owner boundary for packed lane integration tests."""

    calls: list[str]
    receipt: "NativeCompletionReceipt | None"
    transfer: object | None

    def __init__(self) -> None:
        """Initialize an owner without armed transfer authority."""

        self.calls = []
        self.receipt = None
        self.transfer = None

    def arm_transfer(self, handle: object, binding_digest: bytes) -> object:
        """Record exact arm ordering and return one opaque authority.

        :param handle: Exact initialized transfer handle.
        :param binding_digest: Exact terminal lifecycle identity.
        :returns: Opaque exact-generation authority.
        """

        if self.transfer is not None or len(binding_digest) != 32:
            raise RuntimeError("invalid direct terminal arm")
        transfer = (handle, binding_digest)
        self.transfer = transfer
        self.calls.append("arm")
        return transfer

    def post_transfer(
        self,
        transfer: object,
        post: Callable[[object], object],
    ) -> object:
        """Post only the exact authority returned by :meth:`arm_transfer`.

        :param transfer: Exact armed transfer authority.
        :param post: External NIXL post operation.
        :returns: Existing post result.
        """

        if transfer is not self.transfer:
            raise RuntimeError("terminal post changed transfer authority")
        self.calls.append("post")
        return post(transfer[0])

    def settle_success(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> object:
        """Return one planned receipt under the matching owner action.

        :param transfer: Exact posted transfer authority.
        :param action: Matching successful owner action.
        :returns: Planned native completion receipt.
        """

        if transfer is not self.transfer or action.binding.digest != transfer[1]:
            raise RuntimeError("terminal settlement changed authority")
        if action.kind is not NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
            raise RuntimeError("terminal success received another action")
        if self.receipt is None:
            raise RuntimeError("terminal success has no receipt")
        self.calls.append("settle_success")
        return self.receipt

    def settle_failure(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Record one authoritative failure settlement.

        :param transfer: Exact posted transfer authority.
        :param action: Matching failed owner action.
        """

        if transfer is not self.transfer or action.binding.digest != transfer[1]:
            raise RuntimeError("terminal failure changed authority")
        self.calls.append("settle_failure")

    def cancel_transfer(self, transfer: object) -> None:
        """Record cancellation without discarding retained authority.

        :param transfer: Exact posted transfer authority.
        """

        if transfer is not self.transfer:
            raise RuntimeError("terminal cancellation changed authority")
        self.calls.append("cancel")

    def release_transfer(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release one settled transfer exactly once.

        :param transfer: Exact settled transfer authority.
        :param action: Matching source ACK action.
        """

        if transfer is not self.transfer:
            raise RuntimeError("terminal release changed authority")
        if action.kind is not NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
            raise RuntimeError("terminal release received another action")
        if action.binding.digest != transfer[1]:
            raise RuntimeError("terminal release changed request binding")
        self.calls.append("release")
        self.transfer = None


class RetainedTransferOwner:
    """Weak-referenceable fake NIXL transfer or endpoint owner."""


class NativeEnum(enum.Enum):
    """Fake named pybind enum values used by native receipt tests."""

    NIXL_SUCCESS = enum.auto()
    NIXL_WRITE = enum.auto()
    NIXL_XFER_ATTESTATION_REMOTE_FLUSHED = enum.auto()
    VRAM_SEG = enum.auto()


@dataclasses.dataclass(frozen=True)
class NativeTransport:
    """Fake immutable native UCX transport context."""

    transport: str
    device: str


@dataclasses.dataclass(frozen=True)
class NativeSegment:
    """Fake immutable native transfer segment."""

    index: int
    localAddress: int
    remoteAddress: int
    localDeviceId: int
    remoteDeviceId: int
    length: int
    workerId: int
    workerIdentity: int
    endpointIdentity: int
    requestInfo: str
    selectedTransports: tuple[NativeTransport, ...]
    posted: bool


@dataclasses.dataclass(frozen=True)
class NativeEndpoint:
    """Fake immutable native endpoint flush evidence."""

    workerId: int
    workerIdentity: int
    endpointIdentity: int
    segmentIndices: tuple[int, ...]
    transports: tuple[NativeTransport, ...]
    flushPosted: bool
    remoteFlushed: bool


@dataclasses.dataclass(frozen=True)
class NativeRuntimeArtifact:
    """Fake immutable loaded-runtime identity."""

    component: str
    path: str
    buildId: str
    version: str


@dataclasses.dataclass(frozen=True)
class NativeCompletionReceipt:
    """Fake immutable one-shot native completion receipt."""

    handleIdentity: int
    generation: int
    state: NativeEnum
    status: NativeEnum
    submissionSealed: bool
    completionClaimed: bool
    backend: str
    localAgent: str
    remoteAgent: str
    operation: NativeEnum
    localMemoryType: NativeEnum
    remoteMemoryType: NativeEnum
    segments: tuple[NativeSegment, ...]
    endpoints: tuple[NativeEndpoint, ...]
    runtimeArtifacts: tuple[NativeRuntimeArtifact, ...]
    descriptorDigest: str
    evidenceDigest: str
    error: str


class RecordingDirectCudaEventAuthority(PackedDirectCudaEventAuthority):
    """Fake one-shot direct CUDA-event completion owner."""

    receipts: dict[object, object]

    def __init__(self) -> None:
        """Initialize without recorded events."""

        self.receipts = {}

    def take_recorded_event(self, transport_handle: object) -> object | None:
        """Take one exact fake event authority.

        :param transport_handle: Exact direct CUDA transfer owner.
        :returns: One-shot authority, or ``None`` while pending.
        """

        return self.receipts.pop(transport_handle, None)


class RecordingVisibilityExecutor(PackedDestinationVisibilityActionExecutor):
    """CPU-only destination action executor with fault injection."""

    actions: list[PackedDestinationVisibilityAction]
    fail_action: PackedDestinationVisibilityAction | None
    imported_cuda_event_writers: frozenset[StagingWriterId]

    def __init__(
        self,
        imported_cuda_event_writers: frozenset[StagingWriterId] = frozenset(),
    ) -> None:
        """Initialize destination visibility action state.

        :param imported_cuda_event_writers: Writers with a real imported event.
        """

        self.actions = []
        self.fail_action = None
        self.imported_cuda_event_writers = imported_cuda_event_writers

    def establish_cuda_stream_dependency(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
    ) -> None:
        """Record one destination stream dependency.

        :param writer_id: Authenticated writer used for event lookup.
        :param policy: Decode-local pinned policy.
        :param evidence: Authenticated writer evidence.
        """

        policy.validate_evidence(evidence)
        if writer_id not in self.imported_cuda_event_writers:
            raise RuntimeError("no imported CUDA event for authenticated writer")
        self._record(PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY)

    def flush_gpudirect_writes(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
        target: PackedGpuDirectFlushTarget,
        scope: PackedGpuDirectFlushScope,
    ) -> None:
        """Record one destination-owned GPUDirect host flush.

        :param writer_id: Authenticated writer owning the selected lane.
        :param policy: Decode-local pinned policy.
        :param evidence: Authenticated writer evidence.
        :param target: Required CUDA flush target.
        :param scope: Required CUDA flush scope.
        """

        if writer_id not in WRITERS:
            raise ValueError("unexpected GPUDirect writer")
        policy.validate_evidence(evidence)
        if target is not PackedGpuDirectFlushTarget.CURRENT_CONTEXT:
            raise ValueError("unexpected GPUDirect flush target")
        if scope is not PackedGpuDirectFlushScope.OWNER:
            raise ValueError("unexpected GPUDirect flush scope")
        self._record(PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH)

    def _record(self, action: PackedDestinationVisibilityAction) -> None:
        """Record or fail one destination action.

        :param action: Selected destination action.
        """

        if self.fail_action is action:
            raise RuntimeError("injected destination visibility failure")
        self.actions.append(action)


class BlockingVisibilityExecutor(RecordingVisibilityExecutor):
    """Destination executor that exposes deterministic concurrent-action gates."""

    action_count: int
    actions_entered: threading.Event
    release_actions: threading.Event
    _action_lock: threading.Lock
    _required_action_count: int

    def __init__(self, required_action_count: int = 1) -> None:
        """Initialize blocked host flushes.

        :param required_action_count: Concurrent entries required to signal.
        """

        if required_action_count <= 0:
            raise ValueError("required_action_count must be positive")
        super().__init__()
        self.action_count = 0
        self.actions_entered = threading.Event()
        self.release_actions = threading.Event()
        self._action_lock = threading.Lock()
        self._required_action_count = required_action_count

    def flush_gpudirect_writes(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
        target: PackedGpuDirectFlushTarget,
        scope: PackedGpuDirectFlushScope,
    ) -> None:
        """Block one validated host flush until the test releases it.

        :param writer_id: Authenticated writer owning the selected lane.
        :param policy: Decode-local pinned policy.
        :param evidence: Authenticated writer evidence.
        :param target: Required CUDA flush target.
        :param scope: Required CUDA flush scope.
        """

        with self._action_lock:
            self.action_count += 1
            if self.action_count >= self._required_action_count:
                self.actions_entered.set()
        if not self.release_actions.wait(timeout=10):
            raise RuntimeError("timed out waiting to release destination visibility")
        super().flush_gpudirect_writes(
            writer_id,
            policy,
            evidence,
            target,
            scope,
        )


class BlockingPreflightProtocol(PackedDecodeProtocol):
    """Decode protocol with a deterministic successful-preflight test gate."""

    preflight_entered: threading.Event
    release_preflight: threading.Event

    def __init__(self, allocator: PackedIntervalLeaseAllocator) -> None:
        """Initialize a blocked preflight protocol.

        :param allocator: Decode staging lease allocator.
        """

        super().__init__(allocator)
        self.preflight_entered = threading.Event()
        self.release_preflight = threading.Event()

    def preflight_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Pause after protocol validation and before returning to the coordinator.

        :param message: Terminal writer outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this outcome still requires admission.
        """

        admission_required = super().preflight_writer_outcome(
            message,
            authenticated_writer_id,
        )
        self.preflight_entered.set()
        if not self.release_preflight.wait(timeout=10):
            raise RuntimeError("timed out waiting to release protocol preflight")
        return admission_required


class BlockingCommitProtocol(PackedDecodeProtocol):
    """Decode protocol with a deterministic ticketed-commit test gate."""

    block_ticketed_commit: bool
    commit_entered: threading.Event
    release_commit: threading.Event

    def __init__(self, allocator: PackedIntervalLeaseAllocator) -> None:
        """Initialize an unblocked protocol.

        :param allocator: Decode staging lease allocator.
        """

        super().__init__(allocator)
        self.block_ticketed_commit = False
        self.commit_entered = threading.Event()
        self.release_commit = threading.Event()

    def handle_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
        ticket: _PackedWriterOutcomeTicket | None = None,
    ) -> bool:
        """Optionally pause after proof publication and before protocol commit.

        :param message: Terminal writer outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :param ticket: Coordinator-issued DONE admission.
        :returns: Whether this outcome newly made scatter eligible.
        """

        if self.block_ticketed_commit and ticket is not None:
            self.commit_entered.set()
            if not self.release_commit.wait(timeout=10):
                raise RuntimeError("timed out waiting to release protocol commit")
        return super().handle_writer_outcome(
            message,
            authenticated_writer_id,
            ticket,
        )


def _peer(name: str = "decode-agent") -> PackedPeerIdentity:
    """Build one authenticated decode process identity.

    :param name: NIXL agent name.
    :returns: Process-generation-bound peer identity.
    """

    return PackedPeerIdentity(
        agent_name=name,
        agent_generation=AGENT_GENERATION,
    )


def _topology() -> PackedTopology:
    """Return the canonical TP2-to-TP1 topology.

    :returns: Test transfer topology.
    """

    return PackedTopology(
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
    )


def _policy(
    *,
    transport_path: PackedTransportPath = PackedTransportPath.NIC_RDMA,
    lane_identifier: str = build_nixl_ucx_lane_identifier((("rc_mlx5", "mlx5_0:1"),)),
    completion_mechanism: PackedWriterCompletionMechanism | None = None,
    writes_ordering: PackedGpuDirectWritesOrdering = (
        PackedGpuDirectWritesOrdering.OWNER
    ),
    flush_options: PackedGpuDirectFlushOptions = PackedGpuDirectFlushOptions.HOST,
    native_data_transport: str | None = None,
    native_data_device: str | None = None,
) -> PackedDestinationVisibilityPolicy:
    """Build one decode-selected route and CUDA visibility policy.

    :param transport_path: CUDA IPC or NIC transport path.
    :param lane_identifier: Pinned route identity.
    :param completion_mechanism: Exact source completion primitive.
    :param writes_ordering: Queried CUDA writes-ordering attribute.
    :param flush_options: Queried CUDA flush-options attribute.
    :param native_data_transport: Exact selected UCX transport override.
    :param native_data_device: Exact selected UCX device override.
    :returns: Immutable destination policy.
    """

    selected_mechanism = completion_mechanism
    if selected_mechanism is None:
        selected_mechanism = (
            PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            if transport_path is PackedTransportPath.CUDA_IPC
            else PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        )
    native_completion = (
        selected_mechanism
        is PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
    )
    selected_transport = native_data_transport
    selected_device = native_data_device
    if native_completion and selected_transport is None:
        selected_transport = (
            "cuda_ipc" if transport_path is PackedTransportPath.CUDA_IPC else "rc_mlx5"
        )
    if native_completion and selected_device is None:
        selected_device = (
            "cuda0" if transport_path is PackedTransportPath.CUDA_IPC else "mlx5_0:1"
        )
    return PackedDestinationVisibilityPolicy(
        transport_path=transport_path,
        lane_identifier=lane_identifier,
        completion_mechanism=selected_mechanism,
        writes_ordering=writes_ordering,
        flush_options=flush_options,
        native_data_transport=selected_transport,
        native_data_device=selected_device,
        native_runtime_artifact_digest=(
            _runtime_artifacts().digest if native_completion else None
        ),
    )


def _runtime_artifacts() -> PackedNixlRuntimeArtifactCohort:
    """Build the complete fake SHA runtime artifact catalog.

    :returns: Immutable fake native runtime cohort.
    """

    specifications: dict[str, tuple[str, str, str]] = {
        "libnixl": ("nixl", "lib/libnixl.so", "1.3.2"),
        "libucp": ("ucx", "lib/libucp.so", "1.21.0"),
        "libuct_cma": ("ucx", "lib/ucx/libuct_cma.so", "1.21.0"),
        "libuct_cuda": ("ucx", "lib/ucx/libuct_cuda.so", "1.21.0"),
        "libuct_cuda_gdrcopy": (
            "ucx",
            "lib/ucx/libuct_cuda_gdrcopy.so",
            "1.21.0",
        ),
        "libuct_ib": ("ucx", "lib/ucx/libuct_ib.so", "1.21.0"),
        "libuct_ib_efa": ("ucx", "lib/ucx/libuct_ib_efa.so", "1.21.0"),
        "libuct_ib_mlx5": ("ucx", "lib/ucx/libuct_ib_mlx5.so", "1.21.0"),
        "libuct_ib_mlx5_gda": (
            "ucx",
            "lib/ucx/libuct_ib_mlx5_gda.so",
            "1.21.0",
        ),
        "libuct_knem": ("ucx", "lib/ucx/libuct_knem.so", "1.21.0"),
        "libuct_xpmem": ("ucx", "lib/ucx/libuct_xpmem.so", "1.21.0"),
        "ucx-plugin": (
            "nixl",
            "lib/plugins/libplugin_UCX.so",
            "1.3.2",
        ),
    }
    return PackedNixlRuntimeArtifactCohort(
        roots=(
            PackedNixlRuntimeRoot(root_id="nixl", path="/opt/nixl"),
            PackedNixlRuntimeRoot(root_id="ucx", path="/opt/ucx"),
        ),
        artifacts=tuple(
            PackedNixlRuntimeArtifactIdentity(
                component=component,
                root_id=root_id,
                relative_path=relative_path,
                build_id=f"{index:02x}" * 20,
                version=version,
            )
            for index, (component, (root_id, relative_path, version)) in enumerate(
                sorted(specifications.items()),
                start=1,
            )
        ),
    )


def _writer_policy_digests() -> dict[StagingWriterId, bytes]:
    """Build exact decode-selected route policies for every writer.

    :returns: Canonical writer-to-policy digest mapping.
    """

    return {
        writer_id: _policy(
            lane_identifier=(f"mlx5_{writer_id.transfer_source_rank}/1:ucx-rc")
        ).digest
        for writer_id in WRITERS
    }


def _visibility_evidence(
    policy: PackedDestinationVisibilityPolicy,
) -> PackedWriterVisibilityEvidence:
    """Build writer evidence bound to one exact destination policy.

    :param policy: Decode-selected path policy.
    :returns: Validated writer evidence.
    """

    native_completion = (
        policy.completion_mechanism
        is PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
    )
    return PackedWriterVisibilityEvidence(
        policy_digest=policy.digest,
        transport_path=policy.transport_path,
        lane_identifier=policy.lane_identifier,
        completion_mechanism=policy.completion_mechanism,
        writer_action=policy.expected_writer_action,
        native_handle_generation=1 if native_completion else None,
        native_descriptor_digest=(
            bytes.fromhex("11" * 32) if native_completion else None
        ),
        native_evidence_digest=bytes.fromhex("22" * 32) if native_completion else None,
    )


def _native_completion_receipt(
    *,
    agent: RecordingMemoryAgent,
    transfer: PackedSourceTransfer,
    source_address: int,
    source_gpu_id: int = 3,
    transport_context: tuple[tuple[str, str], ...] = (("rc_mlx5", "mlx5_0:1"),),
    request_memory_identity: str = "cuda memory",
) -> NativeCompletionReceipt:
    """Build one valid native receipt for an exact packed transfer.

    :param agent: Exact fake native owner.
    :param transfer: Canonical packed source transfer.
    :param source_address: Registered source lane address.
    :param source_gpu_id: Source CUDA device identifier.
    :param transport_context: Native endpoint transport/device context.
    :param request_memory_identity: Diagnostic UCX source-memory description.
    :returns: Immutable native completion receipt.
    """

    worker_id = 0
    worker_identity = 0xB001
    endpoint_identity = 0xE001
    selected_transport, selected_device = transport_context[0]
    segment = NativeSegment(
        index=0,
        localAddress=source_address,
        remoteAddress=transfer.destination_address,
        localDeviceId=source_gpu_id,
        remoteDeviceId=transfer.destination.route.destination_gpu_id,
        length=transfer.length_bytes,
        workerId=worker_id,
        workerIdentity=worker_identity,
        endpointIdentity=endpoint_identity,
        requestInfo=(
            f"{{proto_send}} put from {request_memory_identity} "
            f"length {transfer.length_bytes} zero-copy "
            f"{selected_transport}/{selected_device}"
        ),
        selectedTransports=(
            NativeTransport(
                transport=selected_transport,
                device=selected_device,
            ),
        ),
        posted=True,
    )
    endpoint = NativeEndpoint(
        workerId=worker_id,
        workerIdentity=worker_identity,
        endpointIdentity=endpoint_identity,
        segmentIndices=(0,),
        transports=tuple(
            NativeTransport(transport=transport, device=device)
            for transport, device in transport_context
        ),
        flushPosted=True,
        remoteFlushed=True,
    )
    runtime_cohort = _runtime_artifacts()
    runtime_identities = {
        artifact.component: artifact for artifact in runtime_cohort.artifacts
    }
    runtime_components = derive_nixl_ucx_runtime_artifact_components(
        tuple(transport for transport, _ in transport_context)
    )
    return NativeCompletionReceipt(
        handleIdentity=0xA001,
        generation=7,
        state=NativeEnum.NIXL_XFER_ATTESTATION_REMOTE_FLUSHED,
        status=NativeEnum.NIXL_SUCCESS,
        submissionSealed=True,
        completionClaimed=True,
        backend="UCX",
        localAgent=agent.name,
        remoteAgent=transfer.destination.route.peer.agent_name,
        operation=NativeEnum.NIXL_WRITE,
        localMemoryType=NativeEnum.VRAM_SEG,
        remoteMemoryType=NativeEnum.VRAM_SEG,
        segments=(segment,),
        endpoints=(endpoint,),
        runtimeArtifacts=tuple(
            NativeRuntimeArtifact(
                component=component,
                path=runtime_cohort.resolve_path(runtime_identities[component]),
                buildId=runtime_identities[component].build_id,
                version=runtime_identities[component].version,
            )
            for component in runtime_components
        ),
        descriptorDigest="11" * 32,
        evidenceDigest="22" * 32,
        error="",
    )


def _terminal_owner_action(
    binding_digest: bytes,
    kind: NativeTerminalOwnerActionKind = (
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY
    ),
) -> NativeTerminalOwnerAction:
    """Build one exact source owner action for packed lane tests.

    :param binding_digest: Exact lifecycle identity.
    :param kind: Closed owner action kind.
    :returns: Valid native owner action.
    """

    owner = NativeTerminalProcessIdentity(
        process_generation=b"p" * 16,
        role=NativeTerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
        digest=b"o" * 32,
    )
    binding = NativeTerminalRequestBinding(
        room_id=KEY.room_id,
        request_generation=KEY.request_generation,
        owner=owner,
        rank_manifest_digest=b"m" * 32,
        allocation_digest=b"a" * 32,
        digest=binding_digest,
    )
    return NativeTerminalOwnerAction(
        action_id=17,
        kind=kind,
        binding=binding,
        commit_timestamp_ns=31,
        receipt=None,
    )


def _capability(
    *,
    peer: PackedPeerIdentity | None = None,
    base_address: int = 0xA00000,
    total_size: int = 1 << 20,
    request_generation: bytes = REQUEST_GENERATION,
    visibility_policy: PackedDestinationVisibilityPolicy | None = None,
) -> PackedDestinationCapability:
    """Build one bootstrap-derived destination capability.

    :param peer: Destination process override.
    :param base_address: Registered arena base.
    :param total_size: Registered arena capacity.
    :param request_generation: Authorized request generation.
    :param visibility_policy: Decode-selected policy override.
    :returns: Immutable destination capability.
    """

    selected_policy = _policy() if visibility_policy is None else visibility_policy
    route = PackedDestinationRouteBinding(
        peer=_peer() if peer is None else peer,
        arena_generation=ARENA_GENERATION,
        destination_gpu_id=6,
        topology=_topology(),
        visibility_policy_digest=selected_policy.digest,
        base_address=base_address,
        total_size=total_size,
        alignment_bytes=256,
    )
    return PackedDestinationCapability(
        route=route,
        request_generation=request_generation,
    )


def _aligned_cpu_byte_tensor(size: int, alignment: int = 256) -> torch.Tensor:
    """Allocate a CPU byte view with deterministic GPU-like alignment.

    :param size: Requested byte capacity.
    :param alignment: Required data-pointer alignment.
    :returns: Contiguous aligned byte tensor.
    """

    storage = torch.empty(size + alignment - 1, dtype=torch.uint8)
    offset = (-storage.data_ptr()) % alignment
    tensor = storage[offset : offset + size]
    if tensor.data_ptr() % alignment != 0:
        raise RuntimeError("failed to construct an aligned CPU test tensor")
    return tensor


def _kv_args(*, source: bool) -> KVArgs:
    """Build CPU-only source or destination registration metadata.

    :param source: Whether to build TP2 source rather than TP1 destination data.
    :returns: Fake registered KV metadata.
    """

    kv_args = KVArgs()
    main_item_len = 32 if source else 64
    state_item_len = 16 if source else 32
    kv_args.kv_data_ptrs = [0x100000, 0x110000]
    kv_args.kv_data_lens = [main_item_len * 32, main_item_len * 32]
    kv_args.kv_item_lens = [main_item_len, main_item_len]
    kv_args.kv_layer_ids = [5, 5]
    kv_args.state_types = [StateType.SWA]
    kv_args.state_data_ptrs = [[0x200000, 0x210000]]
    kv_args.state_data_lens = [[state_item_len * 32, state_item_len * 32]]
    kv_args.state_item_lens = [[state_item_len, state_item_len]]
    kv_args.state_layer_ids = [[1, 1]]
    kv_args.page_size = 4
    return kv_args


def _destination_registration() -> PackedDestinationRegistration:
    """Build the decode geometry advertised to source writers.

    :returns: TP1 destination registration.
    """

    destination = _kv_args(source=False)
    return PackedDestinationRegistration(
        main_item_lens=tuple(destination.kv_item_lens),
        main_layer_ids=tuple(destination.kv_layer_ids),
        state_item_lens=tuple(
            tuple(item_lens) for item_lens in destination.state_item_lens
        ),
        state_layer_ids=tuple(
            tuple(layer_ids) for layer_ids in destination.state_layer_ids
        ),
        page_size=destination.page_size,
    )


def _component_pages(
    component_id: StagingComponentId,
) -> PackedComponentPages:
    """Build two source and destination pages for one component.

    :param component_id: Main KV or SWA component identity.
    :returns: Immutable component page projections.
    """

    return PackedComponentPages(
        component_id=component_id,
        source_pages=np.asarray((3, 4), dtype=np.int32),
        destination_pages=np.asarray((7, 8), dtype=np.int32),
        destination_index_offset=0,
    )


def _prefill_chunk(
    component_ids: tuple[StagingComponentId, ...],
    *,
    is_last: bool,
) -> tuple[PackedLayoutSpec, StagingEndpointBufferBinding]:
    """Build one source-authored packed chunk.

    :param component_ids: Active component identities.
    :param is_last: Whether the chunk completes the room.
    :returns: Canonical spec and source binding.
    """

    return build_prefill_chunk(
        key=KEY,
        is_last=is_last,
        kv_args=_kv_args(source=True),
        destination_registration=_destination_registration(),
        components=tuple(
            _component_pages(component_id) for component_id in component_ids
        ),
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
        writers=WRITERS,
    )


def _decode_spec(
    spans: tuple[StagingComponentSpan, ...],
    *,
    is_last: bool,
) -> PackedLayoutSpec:
    """Build trusted decode-local canonical layout input.

    :param spans: Room-derived active component spans.
    :param is_last: Room-derived final marker.
    :returns: Decode-local packed spec.
    """

    return build_decode_spec(
        chunk_id=KEY.chunk_id,
        is_last=is_last,
        spans=spans,
        kv_args=_kv_args(source=False),
        expected_writers=WRITERS,
        source_tp_size=2,
        destination_tp_size=1,
        destination_tp_rank=0,
    )


def _destination_outcome_fixture(
    action_executor: PackedDestinationVisibilityActionExecutor,
    *,
    policies: (
        dict[
            StagingWriterId,
            PackedDestinationVisibilityPolicy,
        ]
        | None
    ) = None,
    coordinator_policies: (
        dict[
            StagingWriterId,
            PackedDestinationVisibilityPolicy,
        ]
        | None
    ) = None,
    protocol_type: type[PackedDecodeProtocol] = PackedDecodeProtocol,
) -> tuple[
    PackedDecodeProtocol,
    PackedDestinationOutcomeCoordinator,
    PackedIntervalLeaseAllocator,
    dict[StagingWriterId, PackedReady],
    dict[StagingWriterId, PackedDestinationVisibilityPolicy],
]:
    """Build one READY decode chunk with its destination outcome coordinator.

    :param action_executor: Destination CUDA action implementation.
    :param policies: Writer policy overrides.
    :param coordinator_policies: Coordinator-only registration override.
    :param protocol_type: Decode protocol implementation to instantiate.
    :returns: Protocol, coordinator, allocator, READY messages, and policies.
    """

    source_spec, _ = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    decode_spec = _decode_spec(source_spec.spans, is_last=True)
    destination = _kv_args(source=False)
    registry = build_component_buffer_registry(
        destination,
        active_destination_page_arrays(
            destination,
            np.asarray((7, 8), dtype=np.int32),
            [np.asarray((9, 10), dtype=np.int32)],
        ),
    )
    selected_policies = (
        {
            writer_id: _policy(
                lane_identifier=(
                    f"mlx5_{writer_id.transfer_source_rank}/1:ucx-unordered"
                ),
                writes_ordering=PackedGpuDirectWritesOrdering.NONE,
            )
            for writer_id in WRITERS
        }
        if policies is None
        else dict(policies)
    )
    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1 << 20,
    )
    protocol = protocol_type(allocator)
    outcomes = PackedDestinationOutcomeCoordinator(protocol, action_executor)
    protocol.register_chunk(
        KEY,
        decode_spec,
        registry,
        {writer_id: policy.digest for writer_id, policy in selected_policies.items()},
    )
    outcomes.register_chunk(
        KEY,
        (selected_policies if coordinator_policies is None else coordinator_policies),
    )
    ready_messages: tuple[PackedReady, ...] = ()
    for writer_id in WRITERS:
        ready_messages = protocol.handle_prepare(
            PackedPrepare(
                key=KEY,
                writer_id=writer_id,
                spec=source_spec,
                digest=source_spec.build().digest,
            ),
            writer_id,
        )
    return (
        protocol,
        outcomes,
        allocator,
        {message.writer_id: message for message in ready_messages},
        selected_policies,
    )


def _done_outcome(
    ready: PackedReady,
    policy: PackedDestinationVisibilityPolicy,
) -> PackedWriterOutcome:
    """Build exact terminal DONE for one destination policy.

    :param ready: Decode-issued writer projection.
    :param policy: Decode-selected route policy.
    :returns: Successful writer outcome with source evidence.
    """

    return PackedWriterOutcome(
        key=ready.key,
        writer_id=ready.writer_id,
        digest=ready.digest,
        lease_id=ready.lease_id,
        status=PackedWriterOutcomeStatus.DONE,
        visibility=_visibility_evidence(policy),
    )


@pytest.mark.parametrize(
    ("component_ids", "is_last"),
    (
        ((MAIN_KV_COMPONENT,), False),
        ((MAIN_KV_COMPONENT, SWA_COMPONENT), True),
    ),
)
def test_decode_rebuilds_supported_source_layouts(
    component_ids: tuple[StagingComponentId, ...],
    is_last: bool,
) -> None:
    """Trusted decode metadata reproduces every supported source shape."""

    source_spec, _ = _prefill_chunk(component_ids, is_last=is_last)
    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=is_last,
    )

    assert decode_spec.build() == source_spec.build()


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_dflash_target_and_draft_geometry_survives_asymmetric_tp(
    source_tp_size: int,
) -> None:
    """Supported packed writers preserve heterogeneous target and draft KV."""

    draft_pool = object.__new__(MHATokenToKVPool)
    draft_pool.start_layer = 0
    draft_pool.layer_num = 5
    draft_layer_ids = resolve_kv_layer_ids(draft_pool, registered_entry_count=10)
    target_layer_ids = [
        5,
        11,
        17,
        23,
        29,
        35,
        41,
        47,
        53,
        59,
    ] * 2
    layer_ids = [*target_layer_ids, *draft_layer_ids]
    page_size = 16
    target_destination_item_len = page_size * 16 * 256 * 2
    draft_destination_item_len = page_size * 8 * 128 * 2
    destination_item_lens = [target_destination_item_len] * 20 + [
        draft_destination_item_len
    ] * 10
    source_item_lens = [
        item_len // source_tp_size for item_len in destination_item_lens
    ]

    def make_kv_args(item_lens: list[int]) -> KVArgs:
        """Build one CPU-only main-KV registration.

        :param item_lens: Per-entry bytes for one physical page.
        :returns: Registration metadata aligned with the DFlash layer IDs.
        """

        kv_args = KVArgs()
        kv_args.kv_data_ptrs = [
            0x100000 + entry_index * 0x10000 for entry_index in range(len(item_lens))
        ]
        kv_args.kv_data_lens = [item_len * 64 for item_len in item_lens]
        kv_args.kv_item_lens = item_lens
        kv_args.kv_layer_ids = layer_ids
        kv_args.state_types = []
        kv_args.state_data_ptrs = []
        kv_args.state_data_lens = []
        kv_args.state_item_lens = []
        kv_args.state_layer_ids = []
        kv_args.page_size = page_size
        return kv_args

    destination = make_kv_args(destination_item_lens)
    destination_registration = PackedDestinationRegistration(
        main_item_lens=tuple(destination.kv_item_lens),
        main_layer_ids=tuple(destination.kv_layer_ids),
        state_item_lens=(),
        state_layer_ids=(),
        page_size=page_size,
    )
    writers = tuple(
        StagingWriterId(
            transfer_source_rank=rank,
            source_attn_tp_rank=rank,
            source_pp_rank=0,
            source_cp_rank=0,
        )
        for rank in range(source_tp_size)
    )
    source_spec, _ = build_prefill_chunk(
        key=KEY,
        is_last=True,
        kv_args=make_kv_args(source_item_lens),
        destination_registration=destination_registration,
        components=(_component_pages(MAIN_KV_COMPONENT),),
        source_tp_size=source_tp_size,
        destination_tp_size=1,
        destination_tp_rank=0,
        writers=writers,
    )
    decode_spec = build_decode_spec(
        chunk_id=KEY.chunk_id,
        is_last=True,
        spans=source_spec.spans,
        kv_args=destination,
        expected_writers=writers,
        source_tp_size=source_tp_size,
        destination_tp_size=1,
        destination_tp_rank=0,
    )

    assert draft_layer_ids == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert source_spec.source_components[0].item_lens == tuple(source_item_lens)
    assert source_spec.build() == decode_spec.build()


def test_source_builder_rejects_final_chunk_omitting_registered_swa() -> None:
    """Gemma-like source registration makes final SWA non-optional."""

    with pytest.raises(ValueError, match="omits required components"):
        _prefill_chunk((MAIN_KV_COMPONENT,), is_last=True)


def test_source_builder_rejects_final_chunk_omitting_main_kv() -> None:
    """The last chunk carries its Full-KV slice alongside required SWA."""

    with pytest.raises(ValueError, match="main-KV slice"):
        _prefill_chunk((SWA_COMPONENT,), is_last=True)


def test_decode_rejects_final_chunk_omitting_required_swa() -> None:
    """Decode-local room truth independently rejects missing final SWA."""

    source_spec, _ = _prefill_chunk((MAIN_KV_COMPONENT,), is_last=False)

    with pytest.raises(ValueError, match="omits required components"):
        _decode_spec(
            source_spec.spans,
            is_last=True,
        )


def test_decode_geometry_is_derived_without_trusting_source_metadata() -> None:
    """Source-advertised byte geometry cannot become decode canonical truth."""

    source_spec, _ = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    forged_source_geometry = dataclasses.replace(
        source_spec.source_components[0],
        item_lens=(64, 64),
    )
    forged = dataclasses.replace(
        source_spec,
        source_components=(
            forged_source_geometry,
            source_spec.source_components[1],
        ),
    )

    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=True,
    )

    assert forged.source_components != decode_spec.source_components


def test_destination_binding_rejects_page_capacity_overflow() -> None:
    """Decode registration rejects a destination page outside GPU allocation."""

    source_spec, _ = _prefill_chunk((MAIN_KV_COMPONENT,), is_last=False)
    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=False,
    )
    destination = _kv_args(source=False)
    page_arrays = active_destination_page_arrays(
        destination,
        np.asarray((31, 32), dtype=np.int32),
        None,
    )
    registry = build_component_buffer_registry(destination, page_arrays)
    protocol = PackedDecodeProtocol(
        PackedIntervalLeaseAllocator(
            base_address=0x800000,
            total_size=1 << 20,
        )
    )

    with pytest.raises(ValueError, match="exceeds registered page capacity"):
        protocol.register_chunk(
            KEY,
            decode_spec,
            registry,
            _writer_policy_digests(),
        )


def test_component_pages_require_exact_int32_arrays() -> None:
    """Page snapshots reject implicit narrowing and wraparound."""

    with pytest.raises(TypeError, match="dtype int32"):
        PackedComponentPages(
            component_id=MAIN_KV_COMPONENT,
            source_pages=np.asarray((3, 4), dtype=np.int64),
            destination_pages=np.asarray((7, 8), dtype=np.int32),
            destination_index_offset=0,
        )


def test_interval_allocator_coalesces_released_regions() -> None:
    """Released adjacent intervals become one reusable contiguous lease."""

    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1024,
    )
    first = allocator.allocate(1)
    second = allocator.allocate(257)

    assert first.base_address == 0x800000
    assert first.length_bytes == 256
    assert second.base_address == 0x800100
    assert second.length_bytes == 512

    allocator.release(second)
    allocator.release(first)

    whole = allocator.allocate(1024)
    assert whole.base_address == 0x800000
    assert whole.length_bytes == 1024


def test_interval_allocator_does_not_reuse_quarantined_region() -> None:
    """Quarantine retains ownership until the protocol explicitly releases it."""

    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1024,
    )
    quarantined = allocator.allocate(256)
    remainder = allocator.allocate(768)
    allocator.quarantine(quarantined, "DMA may still target the lease")
    allocator.release(remainder)

    with pytest.raises(MemoryError):
        allocator.allocate(1024)

    allocator.release(quarantined)
    assert allocator.allocate(1024).base_address == 0x800000


def test_interval_allocator_requires_aligned_registered_base() -> None:
    """Relative alignment cannot repair a misaligned registered base pointer."""

    with pytest.raises(ValueError, match="base_address must satisfy"):
        PackedIntervalLeaseAllocator(
            base_address=0x800001,
            total_size=1024,
        )


def _registered_ready(
    visibility_policy: PackedDestinationVisibilityPolicy | None = None,
) -> tuple[
    PackedReadyCoordinator,
    PackedPrepare,
    PackedReady,
    PackedPeerIdentity,
]:
    """Register one source chunk and build its valid READY.

    :param visibility_policy: Optional route-policy override.
    :returns: Coordinator, PREPARE, canonical READY, and authenticated peer.
    """

    spec, source_binding = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    coordinator = PackedReadyCoordinator()
    peer = _peer()
    capability = _capability(
        peer=peer,
        visibility_policy=visibility_policy,
    )
    prepare = coordinator.register_chunk(
        key=KEY,
        destination=capability,
        writer_id=WRITERS[0],
        spec=spec,
        source_binding=source_binding,
    )
    writer_layout = writer_layout_for(spec.build(), WRITERS[0])
    ready = PackedReady(
        key=KEY,
        writer_id=WRITERS[0],
        digest=spec.build().digest,
        visibility_policy_digest=capability.route.visibility_policy_digest,
        lease_id=71,
        lease_base_address=0xA00000,
        projection_offset=writer_layout.lease_offset,
        projection_length=writer_layout.length_bytes,
    )
    return coordinator, prepare, ready, peer


def _direct_terminal_lane() -> tuple[
    PackedTransferLane,
    PackedSourceTransfer,
    RecordingMemoryAgent,
    RecordingTerminalOwner,
    object,
    bytes,
]:
    """Build one armed direct-owner lane with a valid planned receipt.

    :returns: Lane, transfer, agent, owner, handle, and binding digest.
    """

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    owner = RecordingTerminalOwner()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        direct_terminal_owner=owner,
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=PackedRegistrationQuarantine(),
    )
    binding_digest = b"b" * 32
    handle = RetainedTransferOwner()
    lane.reserve(transfer, binding_digest=binding_digest)
    lane.arm_submission(handle)
    owner.receipt = _native_completion_receipt(
        agent=agent,
        transfer=transfer,
        source_address=lane.data_ptr,
    )
    return lane, transfer, agent, owner, handle, binding_digest


def test_ready_coordinator_returns_only_canonical_transfer_shape() -> None:
    """Validated READY produces only locally derived canonical transfer work."""

    coordinator, prepare, ready, peer = _registered_ready()

    transfer = coordinator.handle_ready(ready, peer)

    assert isinstance(prepare, PackedPrepare)
    assert transfer.destination_address == (
        ready.lease_base_address + ready.projection_offset
    )
    assert (
        transfer.length_bytes
        == writer_layout_for(
            transfer.layout,
            transfer.writer_id,
        ).length_bytes
    )
    assert transfer.destination.route.peer == peer
    assert transfer.destination.route.destination_gpu_id == 6
    assert transfer.key == KEY
    assert transfer.writer_id == WRITERS[0]
    assert transfer.layout.digest == prepare.digest
    assert transfer.lease_id == ready.lease_id


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("writer_id", WRITERS[1], "writer identity"),
        ("digest", b"\xff" * 32, "digest"),
        (
            "visibility_policy_digest",
            b"\xee" * 32,
            "visibility policy",
        ),
        ("projection_offset", 256, "projection offset"),
        ("projection_length", 1, "projection length"),
        (
            "lease_base_address",
            (1 << 64) - 1,
            "exceeds the uint64 address space",
        ),
    ),
)
def test_ready_coordinator_rejects_forged_fields_without_consuming_chunk(
    field: str,
    value: object,
    match: str,
) -> None:
    """A rejected READY cannot alter later canonical DMA work."""

    coordinator, _, ready, peer = _registered_ready()
    forged = dataclasses.replace(ready, **{field: value})

    with pytest.raises(PackedReadyError, match=match):
        coordinator.handle_ready(forged, peer)

    assert coordinator.handle_ready(ready, peer).lease_id == 71


@pytest.mark.parametrize(
    ("lease_base_address", "match"),
    (
        (0x9FFF00, "exceeds the registered destination capability"),
        (0xA00001, "not aligned"),
        (0xAFFF00, "exceeds the registered destination capability"),
    ),
)
def test_ready_coordinator_bounds_remote_writes_to_capability(
    lease_base_address: int,
    match: str,
) -> None:
    """READY cannot select bytes outside the registered staging arena."""

    coordinator, _, ready, peer = _registered_ready()

    with pytest.raises(PackedReadyError, match=match):
        coordinator.handle_ready(
            dataclasses.replace(ready, lease_base_address=lease_base_address),
            peer,
        )

    assert coordinator.handle_ready(ready, peer).lease_id == ready.lease_id


def test_ready_replay_cannot_cross_request_generation() -> None:
    """A stale room/chunk READY cannot authorize a later request generation."""

    coordinator, _, stale_ready, peer = _registered_ready()
    coordinator.retire_pending(KEY, peer)
    next_generation = bytes.fromhex("fedcba98765432100123456789abcdef")
    next_key = dataclasses.replace(KEY, request_generation=next_generation)
    spec, source_binding = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    coordinator.register_chunk(
        key=next_key,
        destination=_capability(
            peer=peer,
            request_generation=next_generation,
        ),
        writer_id=WRITERS[0],
        spec=spec,
        source_binding=source_binding,
    )

    with pytest.raises(PackedReadyError, match="not pending"):
        coordinator.handle_ready(stale_ready, peer)


def test_ready_coordinator_authenticates_decode_route() -> None:
    """READY from another decode peer cannot select a destination address."""

    coordinator, _, ready, peer = _registered_ready()

    with pytest.raises(PackedReadyError, match="route is not pending"):
        coordinator.handle_ready(ready, _peer("other-decode-agent"))

    assert coordinator.handle_ready(ready, peer).length_bytes > 0


def test_ready_coordinator_tracks_same_chunk_per_decode_route() -> None:
    """One source rank can independently serve multiple destination TP ranks."""

    spec, source_binding = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    coordinator = PackedReadyCoordinator()
    peers = (_peer("decode-agent-0"), _peer("decode-agent-1"))
    capabilities = (
        _capability(peer=peers[0], base_address=0xA00000),
        _capability(peer=peers[1], base_address=0xB00000),
    )
    for capability in capabilities:
        coordinator.register_chunk(
            key=KEY,
            destination=capability,
            writer_id=WRITERS[0],
            spec=spec,
            source_binding=source_binding,
        )
    writer_layout = writer_layout_for(spec.build(), WRITERS[0])
    ready = PackedReady(
        key=KEY,
        writer_id=WRITERS[0],
        digest=spec.build().digest,
        visibility_policy_digest=capabilities[0].route.visibility_policy_digest,
        lease_id=71,
        lease_base_address=0xA00000,
        projection_offset=writer_layout.lease_offset,
        projection_length=writer_layout.length_bytes,
    )

    first = coordinator.handle_ready(ready, peers[0])
    second = coordinator.handle_ready(
        dataclasses.replace(
            ready,
            lease_id=72,
            lease_base_address=0xB00000,
        ),
        peers[1],
    )

    assert first.destination.route.peer == peers[0]
    assert first.lease_id == 71
    assert second.destination.route.peer == peers[1]
    assert second.lease_id == 72


def test_ready_coordinator_never_hands_out_duplicate_dma_work() -> None:
    """An accepted READY is consumed before any duplicate can post another DMA."""

    coordinator, _, ready, peer = _registered_ready()
    coordinator.handle_ready(ready, peer)

    with pytest.raises(PackedReadyError, match="not pending"):
        coordinator.handle_ready(ready, peer)


@pytest.mark.parametrize(
    ("transports", "route_components"),
    [
        (("posix", "self", "sysv", "tcp"), ()),
        (("cuda_copy", "cuda_ipc"), ("libuct_cuda",)),
        (
            ("gdr_copy",),
            ("libuct_cuda", "libuct_cuda_gdrcopy"),
        ),
        (
            ("dc_mlx5", "gga_mlx5", "rc_mlx5", "ud_mlx5"),
            ("libuct_ib", "libuct_ib_mlx5"),
        ),
        (
            ("rc_gda",),
            ("libuct_ib", "libuct_ib_mlx5", "libuct_ib_mlx5_gda"),
        ),
        (("rc_verbs", "ud_verbs"), ("libuct_ib",)),
        (("srd",), ("libuct_ib", "libuct_ib_efa")),
        (
            ("cma", "knem", "xpmem"),
            ("libuct_cma", "libuct_knem", "libuct_xpmem"),
        ),
    ],
)
def test_native_runtime_artifacts_follow_exact_transport_rules(
    transports: tuple[str, ...],
    route_components: tuple[str, ...],
) -> None:
    """The Python acceptance rule must remain identical to native NIXL."""

    expected = tuple(sorted(("libnixl", "libucp", "ucx-plugin", *route_components)))

    assert derive_nixl_ucx_runtime_artifact_components(transports) == expected


def test_native_runtime_artifacts_reject_unknown_transport_names() -> None:
    """A plausible module suffix must not widen the exact native rule table."""

    with pytest.raises(ValueError, match="no runtime artifact rule"):
        derive_nixl_ucx_runtime_artifact_components(("foo_mlx5",))


def test_runtime_artifact_catalog_rejects_duplicate_components() -> None:
    """A manifest catalog cannot contain two identities for one component."""

    cohort = _runtime_artifacts()
    duplicate = cohort.artifacts[0]
    artifacts = tuple(
        sorted(
            (*cohort.artifacts, duplicate),
            key=lambda artifact: artifact.component,
        )
    )

    with pytest.raises(ValueError, match="components are not unique"):
        dataclasses.replace(cohort, artifacts=artifacts)


def test_transfer_lane_emits_exact_no_submit_error_and_closes_idempotently() -> None:
    """A pre-submit abort proves only its exact lease and remains reusable."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=PackedRegistrationQuarantine(),
    )

    lane.reserve(transfer)
    outcome = lane.abort_before_submit("descriptor construction failed")

    assert outcome.key == transfer.key
    assert outcome.writer_id == transfer.writer_id
    assert outcome.digest == transfer.layout.digest
    assert outcome.lease_id == transfer.lease_id
    assert outcome.status is PackedWriterOutcomeStatus.ERROR
    assert outcome.reason == "descriptor construction failed"
    assert lane.state is PackedTransferLaneState.IDLE

    lane.close()
    lane.close()
    assert len(agent.deregistrations) == 1
    assert lane.state is PackedTransferLaneState.CLOSED


@pytest.mark.parametrize(
    "request_memory_identity",
    ("cuda memory", "cuda/cuda0"),
)
def test_transfer_lane_takes_exact_native_receipt_before_done(
    request_memory_identity: str,
) -> None:
    """A lane derives DONE only from its exact native one-shot receipt."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
    )

    lane.reserve(transfer)
    with pytest.raises(RuntimeError, match="completion requires"):
        lane.take_transport_completion()
    transport_handle = RetainedTransferOwner()
    lane.arm_submission(transport_handle)
    assert lane.take_transport_completion() is None
    agent.completion_receipts[transport_handle] = _native_completion_receipt(
        agent=agent,
        transfer=transfer,
        source_address=lane.data_ptr,
        request_memory_identity=request_memory_identity,
    )
    outcome = lane.take_transport_completion()

    assert outcome is not None
    assert outcome.status is PackedWriterOutcomeStatus.DONE
    assert outcome.visibility is not None
    assert (
        outcome.visibility.writer_action
        is PackedWriterVisibilityAction.TRANSPORT_HANDLE_TERMINAL
    )
    assert outcome.visibility.native_handle_generation == 7
    assert outcome.visibility.native_descriptor_digest == bytes.fromhex("11" * 32)
    assert outcome.visibility.native_evidence_digest == bytes.fromhex("22" * 32)
    assert outcome.reason is None
    assert lane.state is PackedTransferLaneState.IDLE
    with pytest.raises(RuntimeError, match="completion requires"):
        lane.take_transport_completion()
    lane.close()


def test_terminal_lane_settles_callback_before_post_returns() -> None:
    """A fast owner callback cannot be overwritten by the post return path."""

    lane, _, _, owner, handle, digest = _direct_terminal_lane()
    outcomes: list[PackedWriterOutcome] = []

    def post(exact_handle: object) -> object:
        if exact_handle is not handle:
            raise AssertionError("post changed the exact transfer handle")
        terminal = threading.Thread(
            target=lambda: outcomes.append(
                lane.settle_terminal_completion(_terminal_owner_action(digest))
            )
        )
        terminal.start()
        terminal.join(timeout=2)
        assert not terminal.is_alive()
        return exact_handle

    lane.post_submission(post)

    assert len(outcomes) == 1
    assert outcomes[0].status is PackedWriterOutcomeStatus.DONE
    assert owner.calls == ["arm", "post", "settle_success"]
    assert lane.state is PackedTransferLaneState.IN_FLIGHT
    assert lane._active_transfer is not None
    assert lane._transport_handle is handle
    assert lane._terminal_transfer is not None
    assert lane._terminal_binding_digest == digest

    lane.release_terminal_transfer(
        _terminal_owner_action(
            digest,
            NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
    )

    assert owner.calls == ["arm", "post", "settle_success", "release"]
    assert lane.state is PackedTransferLaneState.IDLE


def test_terminal_lane_async_completion_is_one_shot() -> None:
    """Asynchronous action settlement validates and releases exactly once."""

    lane, _, _, owner, _, digest = _direct_terminal_lane()
    lane.post_submission(lambda exact_handle: exact_handle)

    outcome = lane.settle_terminal_completion(_terminal_owner_action(digest))

    assert outcome.status is PackedWriterOutcomeStatus.DONE
    assert owner.calls == ["arm", "post", "settle_success"]
    with pytest.raises(RuntimeError, match="already settled"):
        lane.settle_terminal_completion(_terminal_owner_action(digest))
    assert owner.calls.count("settle_success") == 1
    assert owner.calls.count("release") == 0

    lane.release_terminal_transfer(
        _terminal_owner_action(
            digest,
            NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
    )

    assert owner.calls.count("release") == 1
    with pytest.raises(RuntimeError, match="settlement requires"):
        lane.release_terminal_transfer(
            _terminal_owner_action(
                digest,
                NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            )
        )
    assert owner.calls.count("release") == 1


def test_terminal_lane_post_failure_remains_poisoned_until_owner_failure() -> None:
    """An external post exception cannot manufacture no-submit authority."""

    lane, _, _, owner, _, digest = _direct_terminal_lane()

    def fail_post(_: object) -> object:
        raise RuntimeError("injected ambiguous post")

    with pytest.raises(RuntimeError, match="injected ambiguous post"):
        lane.post_submission(fail_post)

    assert lane.state is PackedTransferLaneState.POISONED
    assert owner.calls == ["arm", "post"]
    lane.settle_terminal_failure(
        _terminal_owner_action(
            digest,
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        )
    )
    assert owner.calls == ["arm", "post", "settle_failure"]
    assert lane.state is PackedTransferLaneState.POISONED
    assert lane._active_transfer is not None
    assert lane._transport_handle is not None
    assert lane._terminal_transfer is not None
    with pytest.raises(RuntimeError, match="already settled"):
        lane.settle_terminal_failure(
            _terminal_owner_action(
                digest,
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            )
        )
    assert "release" not in owner.calls


def test_terminal_lane_cancellation_retains_until_matching_failure() -> None:
    """Cancellation poisons reuse and leaves release to owner terminality."""

    lane, _, _, owner, _, digest = _direct_terminal_lane()
    lane.post_submission(lambda exact_handle: exact_handle)

    lane.cancel_terminal_submission()

    assert lane.state is PackedTransferLaneState.POISONED
    assert owner.calls == ["arm", "post", "cancel"]
    lane.settle_terminal_failure(
        _terminal_owner_action(
            digest,
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        )
    )
    assert owner.calls == ["arm", "post", "cancel", "settle_failure"]
    assert lane._active_transfer is not None
    assert lane._transport_handle is not None
    assert lane._terminal_transfer is not None
    assert "release" not in owner.calls


def test_terminal_lane_rejects_release_before_ack_authority() -> None:
    """Only a matching ACK action can release a successful DMA cohort."""

    lane, _, _, owner, handle, digest = _direct_terminal_lane()
    lane.post_submission(lambda exact_handle: exact_handle)

    with pytest.raises(RuntimeError, match="preceded successful completion"):
        lane.release_terminal_transfer(
            _terminal_owner_action(
                digest,
                NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            )
        )

    lane.settle_terminal_completion(_terminal_owner_action(digest))
    with pytest.raises(ValueError, match="requires SOURCE_ACK_READY"):
        lane.release_terminal_transfer(_terminal_owner_action(digest))
    with pytest.raises(RuntimeError, match="another binding"):
        lane.release_terminal_transfer(
            _terminal_owner_action(
                b"x" * 32,
                NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            )
        )

    assert owner.calls == ["arm", "post", "settle_success"]
    assert lane._transport_handle is handle
    lane.release_terminal_transfer(
        _terminal_owner_action(
            digest,
            NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
    )
    assert owner.calls == ["arm", "post", "settle_success", "release"]


def test_terminal_lane_invalid_receipt_quarantines_without_release() -> None:
    """Full receipt validation precedes every handle release or lane reuse."""

    lane, _, _, owner, handle, digest = _direct_terminal_lane()
    receipt = owner.receipt
    if receipt is None:
        raise AssertionError("direct terminal fixture lost its planned receipt")
    owner.receipt = dataclasses.replace(receipt, remoteAgent="another-decode")
    lane.post_submission(lambda exact_handle: exact_handle)

    with pytest.raises(ValueError, match="another remote agent"):
        lane.settle_terminal_completion(_terminal_owner_action(digest))

    assert lane.state is PackedTransferLaneState.POISONED
    assert lane._active_transfer is not None
    assert lane._transport_handle is handle
    assert lane._terminal_transfer is not None
    assert owner.calls == ["arm", "post", "settle_success"]
    assert "release" not in owner.calls


def test_terminal_lane_has_no_polling_completion_path() -> None:
    """Terminal settlement source cannot reach the polling receipt API."""

    source = textwrap.dedent(inspect.getsource(PackedTransferLane))
    terminal_start = source.index("    def settle_terminal_completion(")
    terminal_end = source.index("    def _take_visibility_evidence_locked(")
    terminal_source = source[terminal_start:terminal_end]
    assert "take_xfer_completion_receipt" not in terminal_source

    lane, _, agent, _, handle, _ = _direct_terminal_lane()
    agent.completion_receipts[handle] = object()
    with pytest.raises(RuntimeError, match="requires an owner action"):
        lane.take_transport_completion()
    assert handle in agent.completion_receipts


@pytest.mark.parametrize(
    ("corruption", "error_match"),
    [
        ("remote-agent", "another remote agent"),
        ("destination", "destination differs"),
        ("remote-flush", "did not complete remote flush"),
        ("runtime-build-id", "identity is incomplete"),
        ("runtime-duplicate", "exact canonical route tuple"),
        ("runtime-conflict", "exact canonical route tuple"),
        ("runtime-order", "exact canonical route tuple"),
        ("transport-context", "pinned native data resource"),
        ("request-fallback", "forbidden fallback transport"),
    ],
)
def test_transfer_lane_quarantines_invalid_native_receipt(
    corruption: str,
    error_match: str,
) -> None:
    """A taken receipt must bind every descriptor and visibility authority."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=quarantine,
    )
    handle = RetainedTransferOwner()
    lane.reserve(transfer)
    lane.arm_submission(handle)
    receipt = _native_completion_receipt(
        agent=agent,
        transfer=transfer,
        source_address=lane.data_ptr,
    )
    if corruption == "remote-agent":
        receipt = dataclasses.replace(receipt, remoteAgent="another-decode")
    elif corruption == "destination":
        segment = dataclasses.replace(
            receipt.segments[0],
            remoteAddress=transfer.destination_address + 256,
        )
        receipt = dataclasses.replace(receipt, segments=(segment,))
    elif corruption == "remote-flush":
        endpoint = dataclasses.replace(
            receipt.endpoints[0],
            remoteFlushed=False,
        )
        receipt = dataclasses.replace(receipt, endpoints=(endpoint,))
    elif corruption == "runtime-build-id":
        artifact = dataclasses.replace(receipt.runtimeArtifacts[0], buildId="")
        receipt = dataclasses.replace(
            receipt,
            runtimeArtifacts=(artifact, *receipt.runtimeArtifacts[1:]),
        )
    elif corruption == "runtime-duplicate":
        artifact = receipt.runtimeArtifacts[0]
        receipt = dataclasses.replace(
            receipt,
            runtimeArtifacts=(
                artifact,
                artifact,
                *receipt.runtimeArtifacts[1:],
            ),
        )
    elif corruption == "runtime-conflict":
        artifact = receipt.runtimeArtifacts[0]
        conflict = dataclasses.replace(
            artifact,
            path="/opt/nixl/lib/conflicting-libnixl.so",
            buildId="ff" * 20,
        )
        receipt = dataclasses.replace(
            receipt,
            runtimeArtifacts=(
                artifact,
                conflict,
                *receipt.runtimeArtifacts[1:],
            ),
        )
    elif corruption == "runtime-order":
        receipt = dataclasses.replace(
            receipt,
            runtimeArtifacts=(
                receipt.runtimeArtifacts[1],
                receipt.runtimeArtifacts[0],
                *receipt.runtimeArtifacts[2:],
            ),
        )
    elif corruption == "transport-context":
        endpoint = dataclasses.replace(
            receipt.endpoints[0],
            transports=(NativeTransport("tcp", "eth0"),),
        )
        receipt = dataclasses.replace(receipt, endpoints=(endpoint,))
    elif corruption == "request-fallback":
        segment = dataclasses.replace(
            receipt.segments[0],
            requestInfo=(
                "{proto_send} put from cuda memory "
                f"length {transfer.length_bytes} zero-copy "
                "rc_mlx5/mlx5_0:1 tcp/eth0"
            ),
            selectedTransports=(
                *receipt.segments[0].selectedTransports,
                NativeTransport("tcp", "eth0"),
            ),
        )
        endpoint = dataclasses.replace(
            receipt.endpoints[0],
            transports=(
                *receipt.endpoints[0].transports,
                NativeTransport("tcp", "eth0"),
            ),
        )
        receipt = dataclasses.replace(
            receipt,
            segments=(segment,),
            endpoints=(endpoint,),
        )
    else:
        raise AssertionError(f"unknown corruption case {corruption}")
    agent.completion_receipts[handle] = receipt

    with pytest.raises(ValueError, match=error_match):
        lane.take_transport_completion()

    assert lane.state is PackedTransferLaneState.POISONED
    assert quarantine.count == 1
    with pytest.raises(RuntimeError, match="validation"):
        lane.close()


def test_transfer_lane_supports_native_nixl_cuda_ipc_receipt() -> None:
    """NIXL CUDA IPC uses remote-flush receipt authority, not a fake event."""

    transport_context = (("cuda_ipc", "cuda0"),)
    policy = _policy(
        transport_path=PackedTransportPath.CUDA_IPC,
        lane_identifier=build_nixl_ucx_lane_identifier(transport_context),
        completion_mechanism=(
            PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        ),
        writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        flush_options=PackedGpuDirectFlushOptions.NONE,
    )
    coordinator, _, ready, peer = _registered_ready(policy)
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=policy,
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
    )
    handle = RetainedTransferOwner()
    lane.reserve(transfer)
    lane.arm_submission(handle)
    agent.completion_receipts[handle] = _native_completion_receipt(
        agent=agent,
        transfer=transfer,
        source_address=lane.data_ptr,
        transport_context=transport_context,
    )

    outcome = lane.take_transport_completion()

    assert outcome is not None
    assert outcome.visibility is not None
    assert (
        outcome.visibility.completion_mechanism
        is PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
    )
    assert (
        policy.required_action
        is PackedDestinationVisibilityAction.TRANSPORT_REMOTE_FLUSH
    )
    lane.close()


def test_transfer_lane_supports_direct_cuda_event_authority() -> None:
    """Direct CUDA IPC consumes its bound exported-event authority once."""

    policy = _policy(
        transport_path=PackedTransportPath.CUDA_IPC,
        lane_identifier="cuda-ipc:gpu3->gpu6",
        completion_mechanism=(
            PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
        ),
        writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        flush_options=PackedGpuDirectFlushOptions.NONE,
    )
    coordinator, _, ready, peer = _registered_ready(policy)
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    with pytest.raises(ValueError, match="exact event authority"):
        PackedTransferLane(
            agent=agent,
            destination_route=transfer.destination.route,
            visibility_policy=policy,
            gpu_id=3,
            tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        )
    authority = RecordingDirectCudaEventAuthority()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=policy,
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        direct_cuda_event_authority=authority,
    )
    handle = RetainedTransferOwner()
    lane.reserve(transfer)
    lane.arm_submission(handle)
    assert lane.take_transport_completion() is None
    authority.receipts[handle] = object()

    outcome = lane.take_transport_completion()

    assert outcome is not None
    assert outcome.visibility is not None
    assert (
        outcome.visibility.writer_action
        is PackedWriterVisibilityAction.CUDA_EVENT_RECORDED
    )
    assert outcome.visibility.native_handle_generation is None
    assert outcome.visibility.native_descriptor_digest is None
    assert outcome.visibility.native_evidence_digest is None
    assert (
        policy.required_action
        is PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY
    )
    lane.close()


def test_submitted_failure_without_native_receipt_emits_no_outcome() -> None:
    """A post-arm failure quarantines until native error authority exists."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=quarantine,
    )

    lane.reserve(transfer)
    lane.arm_submission(RetainedTransferOwner())
    lane.mark_submission_ambiguous(
        "native transfer failed without an unforgeable error receipt"
    )

    assert lane.state is PackedTransferLaneState.POISONED
    assert quarantine.count == 1
    assert agent.deregistrations == []
    with pytest.raises(RuntimeError, match="unforgeable error receipt"):
        lane.close()


def test_transfer_lane_rejects_another_destination_route() -> None:
    """A presized lane cannot be borrowed by another peer or arena generation."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    lane = PackedTransferLane(
        agent=RecordingMemoryAgent(),
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
    )
    other_route = dataclasses.replace(
        transfer.destination.route,
        peer=_peer("other-decode"),
    )
    other_destination = dataclasses.replace(
        transfer.destination,
        route=other_route,
    )

    with pytest.raises(ValueError, match="differs from its route lane"):
        lane.reserve(dataclasses.replace(transfer, destination=other_destination))
    with pytest.raises(ValueError, match="key generation"):
        lane.reserve(
            dataclasses.replace(
                transfer,
                key=dataclasses.replace(
                    transfer.key,
                    request_generation=bytes.fromhex(
                        "fedcba98765432100123456789abcdef"
                    ),
                ),
            )
        )

    assert lane.state is PackedTransferLaneState.IDLE
    lane.close()


def test_idle_transfer_lane_reuses_registration_across_request_generations() -> None:
    """A process-lifetime route lane is not allocated once per request."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
    )
    lane.reserve(transfer)
    lane.abort_before_submit("first request complete")

    next_generation = bytes.fromhex("fedcba98765432100123456789abcdef")
    next_capability = PackedDestinationCapability(
        route=transfer.destination.route,
        request_generation=next_generation,
    )
    next_transfer = dataclasses.replace(
        transfer,
        key=dataclasses.replace(KEY, request_generation=next_generation),
        destination=next_capability,
    )
    lane.reserve(next_transfer)
    lane.abort_before_submit("second request complete")

    assert len(agent.registrations) == 1
    assert lane.state is PackedTransferLaneState.IDLE
    lane.close()


def test_transfer_lane_rejects_policy_not_bound_by_capability() -> None:
    """A source lane cannot substitute another UCX lane after bootstrap."""

    agent = RecordingMemoryAgent()

    with pytest.raises(ValueError, match="differs from destination route"):
        PackedTransferLane(
            agent=agent,
            destination_route=_capability().route,
            visibility_policy=_policy(lane_identifier="mlx5_9/1:substituted"),
            expected_runtime_artifacts=_runtime_artifacts(),
            gpu_id=3,
            tensor=torch.empty(4096, dtype=torch.uint8),
        )

    assert agent.registrations == []


def test_ambiguous_submission_poison_retains_registration_without_outcome() -> None:
    """Ambiguous DMA ownership poisons and leak-safely retains the route lane."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=quarantine,
    )

    lane.reserve(transfer)
    lane.arm_submission(object())
    lane.mark_submission_ambiguous("connection lost after submission")

    assert lane.state is PackedTransferLaneState.POISONED
    assert quarantine.count == 1
    assert agent.deregistrations == []
    with pytest.raises(RuntimeError, match="cannot reserve in state poisoned"):
        lane.reserve(transfer)

    with pytest.raises(RuntimeError, match="connection lost after submission"):
        lane.close()
    lane.close()
    assert quarantine.count == 1
    assert agent.deregistrations == []


def test_poisoned_lane_retains_complete_dma_cohort_after_caller_gc() -> None:
    """Quarantine owns tensor, registration agent, endpoint, and exact handle."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    endpoint = RetainedTransferOwner()
    transport_handle = RetainedTransferOwner()
    tensor = torch.empty(transfer.length_bytes, dtype=torch.uint8)
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=tensor,
        quarantine=quarantine,
    )
    agent_ref = weakref.ref(agent)
    endpoint_ref = weakref.ref(endpoint)
    handle_ref = weakref.ref(transport_handle)
    tensor_ref = weakref.ref(tensor)

    lane.reserve(transfer)
    lane.arm_submission(transport_handle, owners=(endpoint,))

    def injected_ambiguous_post(
        armed_lane: PackedTransferLane,
        armed_agent: RecordingMemoryAgent,
        armed_handle: RetainedTransferOwner,
        armed_endpoint: RetainedTransferOwner,
    ) -> None:
        """Fail after verifying the lane owns the complete pre-post cohort."""

        assert armed_lane.state is PackedTransferLaneState.IN_FLIGHT
        assert armed_lane._agent is armed_agent
        assert armed_lane._transport_handle is armed_handle
        assert armed_lane._transport_owners == (armed_endpoint,)
        raise RuntimeError("injected ambiguous post")

    with pytest.raises(RuntimeError, match="injected ambiguous post"):
        injected_ambiguous_post(lane, agent, transport_handle, endpoint)
    lane.mark_submission_ambiguous("connection lost with live NIXL handle")
    with pytest.raises(RuntimeError, match="connection lost"):
        lane.close()

    del agent
    del endpoint
    del lane
    del tensor
    del transport_handle
    gc.collect()

    assert agent_ref() is not None
    assert endpoint_ref() is not None
    assert handle_ref() is not None
    assert tensor_ref() is not None
    assert quarantine.retains(agent_ref())
    assert quarantine.retains(endpoint_ref())
    assert quarantine.retains(handle_ref())


def test_unquiesced_gather_failure_retains_lane_but_proves_no_remote_dma() -> None:
    """Source-stream ambiguity retains memory without retaining decode lease."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=RecordingMemoryAgent(),
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(transfer.length_bytes, dtype=torch.uint8),
        quarantine=quarantine,
    )

    lane.reserve(transfer)
    outcome = lane.abort_before_submit(
        "gather failed",
        source_stream_quiesced=False,
    )

    assert outcome.status is PackedWriterOutcomeStatus.ERROR
    assert lane.state is PackedTransferLaneState.POISONED
    assert quarantine.count == 1


def test_transfer_lane_deregistration_failure_is_strongly_retained() -> None:
    """Failed NIXL cleanup cannot leave a registration pointing at freed memory."""

    coordinator, _, ready, peer = _registered_ready()
    transfer = coordinator.handle_ready(ready, peer)
    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    lane = PackedTransferLane(
        agent=agent,
        destination_route=transfer.destination.route,
        visibility_policy=_policy(),
        expected_runtime_artifacts=_runtime_artifacts(),
        gpu_id=3,
        tensor=torch.empty(4096, dtype=torch.uint8),
        quarantine=quarantine,
    )
    agent.fail_deregistration = True

    with pytest.raises(RuntimeError, match="injected deregistration"):
        lane.close()

    assert quarantine.count == 1
    assert lane.state is PackedTransferLaneState.CLOSED
    with pytest.raises(RuntimeError, match="cannot reserve in state closed"):
        lane.reserve(transfer)
    lane.close()
    assert quarantine.count == 1
    assert agent.deregistrations == []


def test_staging_arena_owns_registration_allocator_protocol_and_capability() -> None:
    """All decode-side address authority derives from one retained arena owner."""

    agent = RecordingMemoryAgent()
    tensor = _aligned_cpu_byte_tensor(4096)
    peer = _peer()
    arena = PackedStagingArena(
        agent=agent,
        tensor=tensor,
        gpu_id=6,
        peer=peer,
        arena_generation=ARENA_GENERATION,
        alignment_bytes=256,
    )

    visibility_policy = _policy()
    capability = arena.capability(
        request_generation=REQUEST_GENERATION,
        topology=_topology(),
        visibility_policy=visibility_policy,
    )

    assert capability.route.peer == peer
    assert capability.route.base_address == tensor.data_ptr()
    assert capability.route.total_size == tensor.numel()
    assert capability.route.arena_generation == ARENA_GENERATION
    assert capability.route.visibility_policy_digest == visibility_policy.digest
    assert arena.protocol is arena.protocol

    arena.close()
    arena.close()
    assert len(agent.deregistrations) == 1


def test_staging_arena_preserves_adopted_registration_ownership() -> None:
    """An adopted registration remains owned by the legacy staging pool."""

    agent = RecordingMemoryAgent()
    tensor = _aligned_cpu_byte_tensor(4096)
    registration = object()
    arena = PackedStagingArena(
        agent=agent,
        tensor=tensor,
        gpu_id=6,
        peer=_peer(),
        arena_generation=ARENA_GENERATION,
        registration=registration,
        alignment_bytes=256,
    )

    assert agent.registrations == []

    arena.close()
    arena.close()

    assert agent.deregistrations == []


def test_staging_arena_cleanup_failure_is_strongly_retained() -> None:
    """Arena deregistration failure follows the same leak-safe terminal policy."""

    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    tensor = _aligned_cpu_byte_tensor(4096)
    arena = PackedStagingArena(
        agent=agent,
        tensor=tensor,
        gpu_id=6,
        peer=_peer(),
        arena_generation=ARENA_GENERATION,
        alignment_bytes=256,
        quarantine=quarantine,
    )
    protocol = arena.protocol
    allocator = arena._allocator
    agent_ref = weakref.ref(agent)
    allocator_ref = weakref.ref(allocator)
    protocol_ref = weakref.ref(protocol)
    tensor_ref = weakref.ref(tensor)
    agent.fail_deregistration = True

    with pytest.raises(RuntimeError, match="injected deregistration"):
        arena.close()

    assert quarantine.count == 1
    arena.close()
    assert quarantine.count == 1
    assert agent.deregistrations == []
    with pytest.raises(RuntimeError, match="arena is closed"):
        _ = arena.protocol
    with pytest.raises(RuntimeError, match="arena is closed"):
        arena.capability(
            request_generation=REQUEST_GENERATION,
            topology=_topology(),
            visibility_policy=_policy(),
        )

    del agent
    del allocator
    del arena
    del protocol
    del tensor
    gc.collect()

    assert agent_ref() is not None
    assert allocator_ref() is not None
    assert protocol_ref() is not None
    assert tensor_ref() is not None
    assert quarantine.retains(agent_ref())
    assert quarantine.retains(allocator_ref())
    assert quarantine.retains(protocol_ref())


def test_staging_arena_cannot_close_while_protocol_owns_a_live_lease() -> None:
    """Unsafe close quarantines the complete arena cohort process-wide."""

    agent = RecordingMemoryAgent()
    quarantine = PackedRegistrationQuarantine()
    tensor = _aligned_cpu_byte_tensor(4096)
    arena = PackedStagingArena(
        agent=agent,
        tensor=tensor,
        gpu_id=6,
        peer=_peer(),
        arena_generation=ARENA_GENERATION,
        quarantine=quarantine,
    )
    source_spec, _ = _prefill_chunk((MAIN_KV_COMPONENT,), is_last=False)
    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=False,
    )
    destination = _kv_args(source=False)
    registry = build_component_buffer_registry(
        destination,
        active_destination_page_arrays(
            destination,
            np.asarray((7, 8), dtype=np.int32),
            None,
        ),
    )
    protocol = arena.protocol
    allocator = arena._allocator
    protocol.register_chunk(
        KEY,
        decode_spec,
        registry,
        _writer_policy_digests(),
    )
    for writer_id in WRITERS:
        protocol.handle_prepare(
            PackedPrepare(
                key=KEY,
                writer_id=writer_id,
                spec=source_spec,
                digest=source_spec.build().digest,
            ),
            writer_id,
        )

    with pytest.raises(RuntimeError, match="live leases"):
        arena.close()

    assert quarantine.count == 1
    assert agent.deregistrations == []
    arena.close()

    agent_ref = weakref.ref(agent)
    allocator_ref = weakref.ref(allocator)
    protocol_ref = weakref.ref(protocol)
    tensor_ref = weakref.ref(tensor)
    del agent
    del allocator
    del arena
    del protocol
    del tensor
    gc.collect()

    assert agent_ref() is not None
    assert allocator_ref() is not None
    assert protocol_ref() is not None
    assert tensor_ref() is not None
    assert quarantine.retains(agent_ref())
    assert quarantine.retains(allocator_ref())
    assert quarantine.retains(protocol_ref())


@pytest.mark.parametrize(
    ("policy", "expected_action", "executor_actions"),
    [
        (
            _policy(
                transport_path=PackedTransportPath.CUDA_IPC,
                lane_identifier="cuda-ipc:gpu3->gpu6",
                writes_ordering=PackedGpuDirectWritesOrdering.NONE,
                flush_options=PackedGpuDirectFlushOptions.NONE,
            ),
            PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY,
            (PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY,),
        ),
        (
            _policy(
                transport_path=PackedTransportPath.CUDA_IPC,
                lane_identifier=build_nixl_ucx_lane_identifier(
                    (("cuda_ipc", "cuda0"),)
                ),
                completion_mechanism=(
                    PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
                ),
                writes_ordering=PackedGpuDirectWritesOrdering.NONE,
                flush_options=PackedGpuDirectFlushOptions.NONE,
            ),
            PackedDestinationVisibilityAction.TRANSPORT_REMOTE_FLUSH,
            (),
        ),
        (
            _policy(
                lane_identifier="mlx5_0/1:ucx-ordered",
                writes_ordering=PackedGpuDirectWritesOrdering.OWNER,
            ),
            PackedDestinationVisibilityAction.TRANSPORT_REMOTE_FLUSH,
            (),
        ),
        (
            _policy(
                lane_identifier="mlx5_1/1:ucx-unordered",
                writes_ordering=PackedGpuDirectWritesOrdering.NONE,
            ),
            PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH,
            (PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH,),
        ),
    ],
    ids=[
        "direct-cuda-ipc",
        "nixl-cuda-ipc",
        "ordered-nic",
        "host-flushed-nic",
    ],
)
def test_visibility_derivation_completes_destination_owned_action(
    policy: PackedDestinationVisibilityPolicy,
    expected_action: PackedDestinationVisibilityAction,
    executor_actions: tuple[PackedDestinationVisibilityAction, ...],
) -> None:
    """Every route derives proof only after its destination-owned action."""

    evidence = _visibility_evidence(policy)
    executor = RecordingVisibilityExecutor(
        imported_cuda_event_writers=frozenset({WRITERS[0]})
    )
    proof = derive_destination_visibility_proof(
        writer_id=WRITERS[0],
        evidence=evidence,
        policy=policy,
        action_executor=executor,
    )

    assert PackedCudaDeviceAttribute.GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS.value == 117
    assert PackedCudaDeviceAttribute.GPU_DIRECT_RDMA_WRITES_ORDERING.value == 118
    assert PackedGpuDirectFlushTarget.CURRENT_CONTEXT.value == 0
    assert PackedGpuDirectFlushScope.OWNER.value == 100
    assert proof.policy == policy
    assert proof.evidence == evidence
    assert proof.completed_action is expected_action
    assert tuple(executor.actions) == executor_actions
    assert evidence.writer_action is policy.expected_writer_action
    assert evidence.writer_action.value != "gpudirect_host_flush"


def test_visibility_rejects_unflushable_policy_and_unpinned_evidence() -> None:
    """Unordered RDMA and writer-selected routes cannot forge local proof."""

    with pytest.raises(ValueError, match="HOST flush option"):
        PackedDestinationVisibilityPolicy(
            transport_path=PackedTransportPath.NIC_RDMA,
            lane_identifier="mlx5_0/1:ucx-rc",
            completion_mechanism=(
                PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
            ),
            writes_ordering=PackedGpuDirectWritesOrdering.NONE,
            flush_options=PackedGpuDirectFlushOptions.MEMOPS,
            native_data_transport="rc_mlx5",
            native_data_device="mlx5_0:1",
            native_runtime_artifact_digest=_runtime_artifacts().digest,
        )
    with pytest.raises(ValueError, match="terminal NIXL"):
        _policy(
            completion_mechanism=(
                PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            )
        )
    with pytest.raises(ValueError, match="unsupported CUDA attribute bits"):
        _policy(
            flush_options=PackedGpuDirectFlushOptions(1 << 8),
        )

    policy = _policy(lane_identifier="mlx5_1/1:ucx-unordered")
    evidence = dataclasses.replace(
        _visibility_evidence(policy),
        lane_identifier="mlx5_9/1:attacker-selected",
    )
    with pytest.raises(ValueError, match="lane differs"):
        derive_destination_visibility_proof(
            writer_id=WRITERS[0],
            evidence=evidence,
            policy=policy,
            action_executor=RecordingVisibilityExecutor(),
        )


def test_visibility_action_failure_cannot_construct_proof() -> None:
    """A failed destination host flush produces no consumer-visibility proof."""

    policy = _policy(
        writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        lane_identifier="mlx5_1/1:ucx-unordered",
    )
    executor = RecordingVisibilityExecutor()
    executor.fail_action = PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH

    with pytest.raises(RuntimeError, match="injected destination visibility"):
        derive_destination_visibility_proof(
            writer_id=WRITERS[0],
            evidence=_visibility_evidence(policy),
            policy=policy,
            action_executor=executor,
        )

    assert executor.actions == []


def test_failed_host_flush_cannot_return_staging_interval_to_allocator() -> None:
    """Transport-terminal DONE is not lease-terminal until visibility succeeds."""

    source_spec, _ = _prefill_chunk(
        (MAIN_KV_COMPONENT, SWA_COMPONENT),
        is_last=True,
    )
    decode_spec = _decode_spec(
        source_spec.spans,
        is_last=True,
    )
    destination = _kv_args(source=False)
    registry = build_component_buffer_registry(
        destination,
        active_destination_page_arrays(
            destination,
            np.asarray((7, 8), dtype=np.int32),
            [np.asarray((9, 10), dtype=np.int32)],
        ),
    )
    policies = {
        WRITERS[0]: _policy(
            lane_identifier="mlx5_0/1:ucx-unordered",
            writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        ),
        WRITERS[1]: _policy(lane_identifier="mlx5_1/1:ucx-ordered"),
    }
    allocator = PackedIntervalLeaseAllocator(
        base_address=0x800000,
        total_size=1 << 20,
    )
    protocol = PackedDecodeProtocol(allocator)
    protocol.register_chunk(
        KEY,
        decode_spec,
        registry,
        {writer_id: policy.digest for writer_id, policy in policies.items()},
    )
    ready_messages: tuple[PackedReady, ...] = ()
    for writer_id in WRITERS:
        ready_messages = protocol.handle_prepare(
            PackedPrepare(
                key=KEY,
                writer_id=writer_id,
                spec=source_spec,
                digest=source_spec.build().digest,
            ),
            writer_id,
        )
    executor = RecordingVisibilityExecutor()
    executor.fail_action = PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH
    outcomes = PackedDestinationOutcomeCoordinator(protocol, executor)
    outcomes.register_chunk(KEY, policies)
    ready_by_writer = {message.writer_id: message for message in ready_messages}

    failed_ready = ready_by_writer[WRITERS[0]]
    failed_outcome = PackedWriterOutcome(
        key=KEY,
        writer_id=WRITERS[0],
        digest=failed_ready.digest,
        lease_id=failed_ready.lease_id,
        status=PackedWriterOutcomeStatus.DONE,
        visibility=_visibility_evidence(policies[WRITERS[0]]),
    )
    with pytest.raises(PackedDestinationVisibilityError, match="action failed"):
        outcomes.handle_writer_outcome(failed_outcome, WRITERS[0])

    ordered_ready = ready_by_writer[WRITERS[1]]
    outcomes.handle_writer_outcome(
        PackedWriterOutcome(
            key=KEY,
            writer_id=WRITERS[1],
            digest=ordered_ready.digest,
            lease_id=ordered_ready.lease_id,
            status=PackedWriterOutcomeStatus.DONE,
            visibility=_visibility_evidence(policies[WRITERS[1]]),
        ),
        WRITERS[1],
    )

    assert allocator.live_lease_count == 1
    with pytest.raises(MemoryError, match="cannot allocate"):
        allocator.allocate(allocator.total_size)


def test_destination_actions_for_independent_writers_are_not_globally_serialized() -> (
    None
):
    """Per-writer CUDA actions can execute concurrently across the coordinator."""

    executor = BlockingVisibilityExecutor(required_action_count=2)
    protocol, outcomes, _, ready_by_writer, policies = _destination_outcome_fixture(
        executor
    )
    messages = {
        writer_id: _done_outcome(ready_by_writer[writer_id], policies[writer_id])
        for writer_id in WRITERS
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(
                outcomes.handle_writer_outcome,
                messages[writer_id],
                writer_id,
            )
            for writer_id in WRITERS
        )
        try:
            assert executor.actions_entered.wait(timeout=2)
            assert all(not future.done() for future in futures)
        finally:
            executor.release_actions.set()
        results = tuple(future.result(timeout=2) for future in futures)

    assert sum(results) == 1
    assert executor.action_count == 2
    assert protocol.snapshot(KEY).state is PackedProtocolState.SCATTER_READY


def test_destination_policy_registration_must_match_protocol_truth() -> None:
    """Coordinator policy objects cannot drift from READY policy digests."""

    policies = {
        writer_id: _policy(
            lane_identifier=f"mlx5_{writer_id.transfer_source_rank}/1:ucx-ordered"
        )
        for writer_id in WRITERS
    }
    mismatched = dict(policies)
    mismatched[WRITERS[0]] = _policy(
        lane_identifier="mlx5_9/1:mismatched",
    )

    with pytest.raises(ValueError, match="differ from protocol registration"):
        _destination_outcome_fixture(
            RecordingVisibilityExecutor(),
            policies=policies,
            coordinator_policies=mismatched,
        )


def test_duplicate_done_reuses_proof_without_replaying_destination_action() -> None:
    """An identical DONE duplicate performs no second CUDA visibility action."""

    executor = BlockingVisibilityExecutor()
    _, outcomes, _, ready_by_writer, policies = _destination_outcome_fixture(executor)
    message = _done_outcome(ready_by_writer[WRITERS[0]], policies[WRITERS[0]])
    executor.release_actions.set()

    assert not outcomes.handle_writer_outcome(message, WRITERS[0])
    assert not outcomes.handle_writer_outcome(message, WRITERS[0])
    assert executor.action_count == 1


def test_conflicting_concurrent_done_cannot_wait_on_or_inherit_an_attempt() -> None:
    """A different DONE cannot inherit another message's in-flight proof."""

    executor = BlockingVisibilityExecutor()
    protocol, outcomes, _, ready_by_writer, policies = _destination_outcome_fixture(
        executor
    )
    message = _done_outcome(ready_by_writer[WRITERS[0]], policies[WRITERS[0]])
    visibility = message.visibility
    if visibility is None:
        raise RuntimeError("DONE test message lost visibility evidence")
    conflicting = dataclasses.replace(
        message,
        visibility=dataclasses.replace(
            visibility,
            lane_identifier="mlx5_9/1:conflicting",
        ),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            outcomes.handle_writer_outcome,
            message,
            WRITERS[0],
        )
        assert executor.actions_entered.wait(timeout=2)
        conflicting_result = pool.submit(
            outcomes.handle_writer_outcome,
            conflicting,
            WRITERS[0],
        )
        try:
            with pytest.raises(PackedProtocolError, match="visibility ticket"):
                conflicting_result.result(timeout=2)
        finally:
            executor.release_actions.set()
        assert first.result(timeout=2) is False

    assert executor.action_count == 1
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED


def test_failed_visibility_wakes_waiters_and_allows_later_retry() -> None:
    """Concurrent duplicates observe failure while a later action can recover."""

    executor = BlockingVisibilityExecutor()
    executor.fail_action = PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH
    protocol, outcomes, allocator, ready_by_writer, policies = (
        _destination_outcome_fixture(executor)
    )
    message = _done_outcome(ready_by_writer[WRITERS[0]], policies[WRITERS[0]])
    second_started = threading.Event()

    def handle_second() -> bool:
        """Enter the duplicate outcome call and expose scheduler progress."""

        second_started.set()
        return outcomes.handle_writer_outcome(message, WRITERS[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            outcomes.handle_writer_outcome,
            message,
            WRITERS[0],
        )
        assert executor.actions_entered.wait(timeout=2)
        second = pool.submit(handle_second)
        assert second_started.wait(timeout=2)
        with pytest.raises(concurrent.futures.TimeoutError):
            second.result(timeout=0.1)
        executor.release_actions.set()
        with pytest.raises(PackedDestinationVisibilityError, match="action failed"):
            first.result(timeout=2)
        with pytest.raises(PackedDestinationVisibilityError, match="action failed"):
            second.result(timeout=2)

    assert allocator.live_lease_count == 1
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    executor.fail_action = None

    assert not outcomes.handle_writer_outcome(message, WRITERS[0])
    assert protocol.snapshot(KEY).writer_outcomes == (message,)
    assert allocator.live_lease_count == 1


def test_proof_precedes_scatter_ready_and_retirement_cannot_race_commit() -> None:
    """Published proof is visible while ticketed protocol commit is paused."""

    policies = {
        writer_id: _policy(
            lane_identifier=f"mlx5_{writer_id.transfer_source_rank}/1:ucx-ordered"
        )
        for writer_id in WRITERS
    }
    executor = RecordingVisibilityExecutor()
    protocol, outcomes, _, ready_by_writer, selected_policies = (
        _destination_outcome_fixture(
            executor,
            policies=policies,
            protocol_type=BlockingCommitProtocol,
        )
    )
    if not isinstance(protocol, BlockingCommitProtocol):
        raise RuntimeError("commit-gated destination fixture returned wrong protocol")
    first = _done_outcome(
        ready_by_writer[WRITERS[0]],
        selected_policies[WRITERS[0]],
    )
    second = _done_outcome(
        ready_by_writer[WRITERS[1]],
        selected_policies[WRITERS[1]],
    )
    assert not outcomes.handle_writer_outcome(first, WRITERS[0])
    protocol.block_ticketed_commit = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            outcomes.handle_writer_outcome,
            second,
            WRITERS[1],
        )
        assert protocol.commit_entered.wait(timeout=2)
        try:
            assert protocol.snapshot(KEY).state is PackedProtocolState.READY
            assert tuple(
                proof.writer_id for proof in outcomes.proofs(KEY, WRITERS)
            ) == (WRITERS)
            with pytest.raises(
                PackedDestinationVisibilityError,
                match="still in progress",
            ):
                outcomes.retire_chunk(KEY)
        finally:
            protocol.release_commit.set()
        assert result.result(timeout=2)

    assert protocol.snapshot(KEY).state is PackedProtocolState.SCATTER_READY


def test_retirement_rejects_done_reserved_during_protocol_preflight() -> None:
    """Retirement cannot cross the gap before a DONE creates its action attempt."""

    protocol, outcomes, _, ready_by_writer, policies = _destination_outcome_fixture(
        RecordingVisibilityExecutor(),
        protocol_type=BlockingPreflightProtocol,
    )
    if not isinstance(protocol, BlockingPreflightProtocol):
        raise RuntimeError("preflight-gated fixture returned wrong protocol")
    message = _done_outcome(ready_by_writer[WRITERS[0]], policies[WRITERS[0]])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            outcomes.handle_writer_outcome,
            message,
            WRITERS[0],
        )
        assert protocol.preflight_entered.wait(timeout=2)
        try:
            with pytest.raises(
                PackedDestinationVisibilityError,
                match="still in progress",
            ):
                outcomes.retire_chunk(KEY)
        finally:
            protocol.release_preflight.set()
        assert result.result(timeout=2) is False

    assert protocol.snapshot(KEY).writer_outcomes == (message,)


def test_destination_coordinator_retires_protocol_and_proof_state_together() -> None:
    """Terminal retirement cannot leave a protocol or proof cache half-live."""

    policies = {
        writer_id: _policy(
            lane_identifier=f"mlx5_{writer_id.transfer_source_rank}/1:ucx-ordered"
        )
        for writer_id in WRITERS
    }
    protocol, outcomes, _, ready_by_writer, selected_policies = (
        _destination_outcome_fixture(
            RecordingVisibilityExecutor(),
            policies=policies,
        )
    )
    with pytest.raises(PackedProtocolError, match="cannot retire"):
        outcomes.retire_chunk(KEY)
    for writer_id in WRITERS:
        outcomes.handle_writer_outcome(
            _done_outcome(
                ready_by_writer[writer_id],
                selected_policies[writer_id],
            ),
            writer_id,
        )
    protocol.begin_scatter(KEY)
    protocol.complete_scatter(KEY)

    outcomes.retire_chunk(KEY)

    with pytest.raises(PackedProtocolError, match="not registered"):
        protocol.snapshot(KEY)
    with pytest.raises(PackedDestinationVisibilityError, match="incomplete"):
        outcomes.proofs(KEY, WRITERS)


def test_cuda_ipc_visibility_fails_without_imported_writer_event() -> None:
    """A descriptive CUDA lane cannot substitute for an imported event."""

    policy = _policy(
        transport_path=PackedTransportPath.CUDA_IPC,
        lane_identifier="cuda-ipc:gpu3->gpu6",
        writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        flush_options=PackedGpuDirectFlushOptions.NONE,
    )

    with pytest.raises(RuntimeError, match="no imported CUDA event"):
        derive_destination_visibility_proof(
            writer_id=WRITERS[0],
            evidence=_visibility_evidence(policy),
            policy=policy,
            action_executor=RecordingVisibilityExecutor(),
        )
    with pytest.raises(TypeError, match="executor receipt"):
        PackedDestinationVisibilityProof(
            writer_id=WRITERS[0],
            policy=policy,
            evidence=_visibility_evidence(policy),
            completed_action=(PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY),
            _action_receipt=object(),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
