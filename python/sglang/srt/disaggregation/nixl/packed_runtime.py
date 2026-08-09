import dataclasses
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliaryAllocationLeaseAuthority,
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedPrepare,
    PackedProtocolState,
    PackedReady,
    PackedRequestKey,
    PackedRequestTeardown,
    PackedRequestTeardownAck,
    PackedTopology,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    PackedWireMessage,
    decode_packed_message,
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedComponentPages,
    PackedCopyExecutor,
    PackedDestinationCapability,
    PackedDestinationOutcomeCoordinator,
    PackedDestinationRegistration,
    PackedDestinationRouteBinding,
    PackedDestinationVisibilityActionExecutor,
    PackedDestinationVisibilityPolicy,
    PackedGatherError,
    PackedGpuDirectFlushOptions,
    PackedGpuDirectFlushScope,
    PackedGpuDirectFlushTarget,
    PackedGpuDirectWritesOrdering,
    PackedNixlCompletionReceipt,
    PackedNixlEnumValue,
    PackedNixlRuntimeArtifactCohort,
    PackedNixlRuntimeArtifactIdentity,
    PackedNixlRuntimeRoot,
    PackedPeerIdentity,
    PackedReadyCoordinator,
    PackedScatterSubmission,
    PackedSourceTransfer,
    PackedStagingArena,
    PackedTransferLane,
    PackedTransferLaneState,
    PackedTransportPath,
    PackedWriterVisibilityEvidence,
    build_component_buffer_registry,
    build_decode_spec,
    build_nixl_ucx_lane_identifier,
    build_prefill_chunk,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestChunkPlan,
    PackedRequestCommitReceipt,
    PackedRequestPublication,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)

logger = logging.getLogger(__name__)

PACKED_CONTROL_TAG = b"PACKED_V4"
PACKED_KV_TRANSFER_PROTOCOL = "packed-v4"
PACKED_PREPARED_GRANT_PROTOCOL = "control-v1"
PACKED_CONTROL_TIMEOUT_SECONDS = 60.0
PACKED_SOURCE_LANE_MINIMUM_BYTES = 32 * 1024 * 1024

_NIXL_VERSION = "1.3.2"
_UCX_VERSION = "1.21.0"
_UCX_PLUGIN_VERSION = "0.1.0"
_BUILD_ID_PATTERN = re.compile(r"Build ID:\s*([0-9a-fA-F]+)")


class PackedMetadataIndexAllocator(Protocol):
    """Metadata-index allocator surface retained by packed transactions."""

    def free(self, free_index: int) -> None:
        """Return one exact metadata index.

        :param free_index: Exact terminal metadata index.
        """


class PackedRuntimeManager(Protocol):
    """Narrow manager surface used by the packed worker actors."""

    agent: Any
    agent_metadata: bytes
    attn_cp_rank: int
    attn_tp_rank: int
    attn_tp_size: int
    kv_args: KVArgs
    pp_rank: int
    process_generation: str
    transfer_source_rank: int

    def _post_transfer_when_ready(self, handle: object, context: str) -> object:
        """Post an exact prepared native handle.

        :param handle: Prepared native handle.
        :param context: Stable diagnostic context.
        :returns: Posted native handle.
        """

    def record_failure(self, room: int, reason: str) -> None:
        """Record one room failure.

        :param room: Decoder-minted room.
        :param reason: Stable failure reason.
        """

    def update_status(self, room: int, status: KVPoll) -> None:
        """Update one room status.

        :param room: Decoder-minted room.
        :param status: New transfer status.
        """


@dataclasses.dataclass(frozen=True)
class PackedRegistrationAdvertisement:
    """One decoder process's persistent packed arena advertisement.

    :ivar base_address: Registered arena base address.
    :ivar total_size: Arena byte capacity.
    :ivar arena_generation: Process-local arena reuse generation.
    :ivar visibility_policy_digest: Decode-selected path-policy digest.
    :ivar runtime_cohort_digest: Exact native runtime acceptance digest.
    :ivar page_size: Destination cache page size.
    """

    base_address: int
    total_size: int
    arena_generation: bytes
    visibility_policy_digest: bytes
    runtime_cohort_digest: bytes
    page_size: int


@dataclasses.dataclass(frozen=True)
class PackedControlSender:
    """Authenticated route used to send worker control messages.

    :ivar writer_id: Exact source writer reached by this route.
    :ivar send_message: Serialized multipart send operation.
    """

    writer_id: StagingWriterId
    send_message: Callable[[PackedWireMessage], None]


@dataclasses.dataclass(frozen=True)
class PackedDecodeControlSender:
    """Authenticated decode route retained by one source request.

    :ivar peer: Exact decoder NIXL process reached by this route.
    :ivar remote_handle: Loaded NIXL remote-agent handle for data submission.
    :ivar send_message: Serialized multipart control send operation.
    """

    peer: PackedPeerIdentity
    remote_handle: object
    send_message: Callable[[PackedWireMessage], None]


@dataclasses.dataclass(frozen=True)
class PackedPrefillSubmission:
    """Complete source-local input for one packed request transfer.

    :ivar plan: Decoder-authored auxiliary metadata plan.
    :ivar destination: Request-scoped registered destination capability.
    :ivar destination_registration: Decode cache geometry used to rebuild the
        canonical layout.
    :ivar control: Exact decoder control and native data route.
    :ivar components: Main-KV and SWA source/destination page projections.
    :ivar auxiliary_source_index: Source metadata row copied by the canonical writer.
    :ivar producer_stream: CUDA stream containing every source cache write.
    """

    plan: PackedAuxiliaryPlan
    destination: PackedDestinationCapability
    destination_registration: PackedDestinationRegistration
    control: PackedDecodeControlSender
    components: tuple[PackedComponentPages, ...]
    auxiliary_source_index: int
    producer_stream: torch.cuda.Stream

    def __post_init__(self) -> None:
        """Own and validate one immutable source submission."""

        if type(self.plan) is not PackedAuxiliaryPlan:
            raise TypeError("packed source plan must be PackedAuxiliaryPlan")
        if type(self.destination) is not PackedDestinationCapability:
            raise TypeError(
                "packed source destination must be PackedDestinationCapability"
            )
        if type(self.destination_registration) is not PackedDestinationRegistration:
            raise TypeError(
                "packed source registration must be PackedDestinationRegistration"
            )
        if type(self.control) is not PackedDecodeControlSender:
            raise TypeError("packed source control must be PackedDecodeControlSender")
        components = tuple(self.components)
        object.__setattr__(self, "components", components)
        if len(components) == 0:
            raise ValueError("packed source request must contain cache components")
        if any(type(component) is not PackedComponentPages for component in components):
            raise TypeError("packed source components must be PackedComponentPages")
        if type(self.auxiliary_source_index) is not int:
            raise TypeError("packed auxiliary source index must be an integer")
        if self.auxiliary_source_index < 0:
            raise ValueError("packed auxiliary source index must be non-negative")
        if not isinstance(self.producer_stream, torch.cuda.Stream):
            raise TypeError("packed source producer stream must be a CUDA stream")
        if self.destination.request_generation != self.plan.key.request_generation:
            raise ValueError(
                "packed source capability generation differs from auxiliary plan"
            )
        if self.destination.route.peer != self.control.peer:
            raise ValueError("packed source control peer differs from destination")


def encode_packed_control_frames(
    agent_name: str,
    process_generation: str,
    message: PackedWireMessage,
) -> list[bytes]:
    """Build one authenticated packed worker multipart message.

    :param agent_name: Sending NIXL agent name.
    :param process_generation: Sending process generation UUID.
    :param message: Closed packed wire payload.
    :returns: Exact multipart frames.
    """

    if len(agent_name) == 0:
        raise ValueError("packed control agent name must not be empty")
    uuid.UUID(process_generation)
    return [
        PACKED_CONTROL_TAG,
        agent_name.encode("ascii"),
        process_generation.encode("ascii"),
        encode_packed_message(message),
    ]


def decode_packed_control_frames(
    frames: list[bytes],
) -> tuple[str, str, PackedWireMessage]:
    """Decode one bounded packed worker multipart message.

    :param frames: Untrusted multipart frames.
    :returns: Claimed agent, process generation, and validated payload.
    """

    if len(frames) != 4 or frames[0] != PACKED_CONTROL_TAG:
        raise ValueError("packed control multipart shape is invalid")
    agent_name = frames[1].decode("ascii")
    process_generation = frames[2].decode("ascii")
    if len(agent_name) == 0:
        raise ValueError("packed control agent name must not be empty")
    uuid.UUID(process_generation)
    return (
        agent_name,
        process_generation,
        decode_packed_message(frames[3]),
    )


def _read_elf_build_id(path: str) -> str:
    """Read one exact GNU build ID from an ELF object.

    :param path: Canonical ELF path.
    :returns: Lowercase GNU build ID.
    """

    result = subprocess.run(
        ["readelf", "-n", path],
        check=True,
        capture_output=True,
        text=True,
    )
    match = _BUILD_ID_PATTERN.search(result.stdout)
    if match is None:
        raise RuntimeError(f"ELF object has no GNU build ID: {path}")
    return match.group(1).lower()


def load_exact_nixl_runtime_artifacts() -> PackedNixlRuntimeArtifactCohort:
    """Load the exact in-process NIXL/UCX runtime acceptance cohort.

    :returns: Immutable artifact cohort with dynamically read build IDs.
    """

    nixl_root = os.path.realpath(
        os.environ.get(
            "GEMMA4_EXACT_NIXL_PREFIX",
            "/workspace/build/nixl-handle-attestation/prefix",
        )
    )
    ucx_root = os.path.realpath(
        os.environ.get(
            "GEMMA4_EXACT_NIXL_UCX_RUNTIME",
            "/workspace/build/nixl-ucx-cohorts/"
            "c3fca0b90f18936e77deeffdc393aace19ea05c5b6c12b50e8814d29fd4a2b35/"
            "install/nixl-ucx-runtime",
        )
    )
    specifications = {
        "libnixl": (
            "nixl",
            os.path.join(nixl_root, "lib/x86_64-linux-gnu/libnixl.so"),
            _NIXL_VERSION,
        ),
        "libucp": (
            "ucx",
            os.path.join(ucx_root, "lib/libucp.so.0"),
            _UCX_VERSION,
        ),
        "libuct_cuda": (
            "ucx",
            os.path.join(ucx_root, "lib/ucx/libuct_cuda.so.0"),
            _UCX_VERSION,
        ),
        "ucx-plugin": (
            "nixl",
            os.path.join(
                nixl_root,
                "lib/x86_64-linux-gnu/plugins/libplugin_UCX.so",
            ),
            _UCX_PLUGIN_VERSION,
        ),
    }
    roots = {
        "nixl": nixl_root,
        "ucx": ucx_root,
    }
    artifacts: list[PackedNixlRuntimeArtifactIdentity] = []
    for component, (root_id, candidate_path, version) in sorted(specifications.items()):
        path = os.path.realpath(candidate_path)
        root = roots[root_id]
        if os.path.commonpath((root, path)) != root:
            raise RuntimeError(f"runtime artifact escapes {root_id} root")
        artifacts.append(
            PackedNixlRuntimeArtifactIdentity(
                component=component,
                root_id=root_id,
                relative_path=os.path.relpath(path, root),
                build_id=_read_elf_build_id(path),
                version=version,
            )
        )
    return PackedNixlRuntimeArtifactCohort(
        roots=tuple(
            PackedNixlRuntimeRoot(root_id=root_id, path=path)
            for root_id, path in sorted(roots.items())
        ),
        artifacts=tuple(artifacts),
    )


def build_same_host_visibility_policy(
    runtime_artifacts: PackedNixlRuntimeArtifactCohort,
) -> PackedDestinationVisibilityPolicy:
    """Build the native-receipt-proven same-host CUDA IPC policy.

    :param runtime_artifacts: Exact loaded runtime acceptance cohort.
    :returns: Decode-selected CUDA IPC visibility policy.
    """

    return PackedDestinationVisibilityPolicy(
        transport_path=PackedTransportPath.CUDA_IPC,
        lane_identifier=build_nixl_ucx_lane_identifier(
            (("cuda_ipc", "cuda"), ("tcp", "lo"))
        ),
        completion_mechanism=(
            PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        ),
        writes_ordering=PackedGpuDirectWritesOrdering.NONE,
        flush_options=PackedGpuDirectFlushOptions.NONE,
        native_data_transport="cuda_ipc",
        native_data_device="cuda",
        native_runtime_artifact_digest=runtime_artifacts.digest,
    )


class _UnexpectedVisibilityActionExecutor(PackedDestinationVisibilityActionExecutor):
    """Fail if the proven CUDA IPC route unexpectedly requests a host action."""

    def establish_cuda_stream_dependency(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
    ) -> None:
        """Reject an unexpected direct-event route.

        :param writer_id: Unexpected authenticated writer.
        :param policy: Unexpected pinned visibility policy.
        :param evidence: Unexpected evidence.
        """

        del writer_id, policy, evidence
        raise RuntimeError("CUDA IPC native completion requested a stream dependency")

    def flush_gpudirect_writes(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
        target: PackedGpuDirectFlushTarget,
        scope: PackedGpuDirectFlushScope,
    ) -> None:
        """Reject an unexpected destination host flush.

        :param writer_id: Unexpected authenticated writer.
        :param policy: Unexpected pinned visibility policy.
        :param evidence: Unexpected evidence.
        :param target: Unexpected flush target.
        :param scope: Unexpected flush scope.
        """

        del writer_id, policy, evidence, target, scope
        raise RuntimeError("CUDA IPC native completion requested a host flush")


@dataclasses.dataclass(frozen=True)
class _AdoptedAuxiliaryReservation:
    """One generation-bearing adoption of an already allocated metadata row."""

    generation: bytes
    metadata_buffer_index: int
    segments: tuple[PackedAuxiliaryDestinationSegment, ...]


class AdoptedPackedAuxiliarySlotAllocation:
    """Adapt one scheduler-reserved metadata index to packed lease ownership."""

    def __init__(
        self,
        allocator: PackedMetadataIndexAllocator,
        metadata_buffer_index: int,
        kv_args: KVArgs,
    ) -> None:
        """Initialize one single-use adopted metadata row.

        :param allocator: Existing scheduler metadata allocator.
        :param metadata_buffer_index: Already reserved exact row.
        :param kv_args: Decode-local auxiliary registration.
        """

        if metadata_buffer_index < 0:
            raise ValueError("metadata_buffer_index must be non-negative")
        if len(kv_args.aux_data_ptrs) != len(kv_args.aux_item_lens):
            raise ValueError("auxiliary pointer and item-length counts differ")
        segments = tuple(
            PackedAuxiliaryDestinationSegment(
                address=base_address + metadata_buffer_index * item_length,
                item_length=item_length,
            )
            for base_address, item_length in zip(
                kv_args.aux_data_ptrs,
                kv_args.aux_item_lens,
                strict=True,
            )
        )
        self._allocator = allocator
        self._reservation = _AdoptedAuxiliaryReservation(
            generation=secrets.token_bytes(16),
            metadata_buffer_index=metadata_buffer_index,
            segments=segments,
        )
        self._owner: object | None = None
        self._released = False
        self._quarantined = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        """Return whether the underlying index was returned.

        :returns: Exact release state.
        """

        with self._lock:
            return self._released

    @property
    def quarantined(self) -> bool:
        """Return whether reuse is permanently withheld.

        :returns: Exact quarantine state.
        """

        with self._lock:
            return self._quarantined

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Adopt the pre-reserved row exactly once.

        :param owner: Packed authority reservation owner.
        :returns: Opaque adopted reservation.
        """

        with self._lock:
            if self._owner is not None:
                raise RuntimeError("metadata row was already adopted")
            self._owner = owner
            return self._reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Return exact generation and address geometry.

        :param reservation: Exact adopted reservation.
        :returns: Immutable authority snapshot.
        """

        with self._lock:
            self._require_reservation(reservation)
            return PackedAuxiliarySlotReservationSnapshot(
                metadata_buffer_index=self._reservation.metadata_buffer_index,
                metadata_slot_generation=self._reservation.generation,
                destination_segments=self._reservation.segments,
            )

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return the exact adopted row once.

        :param reservation: Exact adopted reservation.
        :param owner: Exact authority reservation owner.
        """

        with self._lock:
            self._require_owner(reservation, owner)
            if self._released or self._quarantined:
                raise RuntimeError("metadata row is already terminal")
            self._allocator.free(self._reservation.metadata_buffer_index)
            self._released = True

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Retain the adopted row against reuse.

        :param reservation: Exact adopted reservation.
        :param owner: Exact authority reservation owner.
        """

        with self._lock:
            self._require_owner(reservation, owner)
            if self._released:
                raise RuntimeError("released metadata row cannot be quarantined")
            self._quarantined = True

    def _require_reservation(self, reservation: object) -> None:
        if reservation is not self._reservation:
            raise RuntimeError("metadata reservation belongs to another adoption")

    def _require_owner(self, reservation: object, owner: object) -> None:
        self._require_reservation(reservation)
        if owner is not self._owner:
            raise RuntimeError("metadata reservation owner is stale")


@dataclasses.dataclass(frozen=True)
class _PackedPrefillTransferStats:
    """Terminal prefill-writer transfer timings.

    :ivar room_id: Decoder-minted bootstrap room.
    :ivar source_rank: Source attention tensor-parallel rank.
    :ivar ready_wait_duration_ms: PREPARE-to-READY wall time.
    :ivar source_gather_copy_duration_ms: Synchronous source gather wall time.
    :ivar main_transport_duration_ms: Posted NIXL transfer-to-receipt wall time.
    """

    room_id: int
    source_rank: int
    ready_wait_duration_ms: float
    source_gather_copy_duration_ms: float
    main_transport_duration_ms: float

    def __str__(self) -> str:
        """Render the stable production log contract.

        :returns: Parser-compatible structured timing record.
        """

        return (
            f"PackedTransferStats(room={self.room_id}, role=prefill, "
            f"source_rank={self.source_rank}, "
            f"ready_wait_duration={self.ready_wait_duration_ms:.3f}ms, "
            "source_gather_copy_duration="
            f"{self.source_gather_copy_duration_ms:.3f}ms, "
            f"main_transport_duration={self.main_transport_duration_ms:.3f}ms)"
        )


@dataclasses.dataclass(frozen=True)
class _PackedDecodeTransferStats:
    """Terminal decode-destination transfer timings.

    :ivar room_id: Decoder-minted bootstrap room.
    :ivar destination_rank: Destination attention tensor-parallel rank.
    :ivar upstream_wait_duration_ms: Transaction-to-first-scatter wall time.
    :ivar destination_scatter_copy_duration_ms: CUDA scatter execution time.
    :ivar finalize_duration_ms: Final-scatter-to-commit wall time.
    :ivar packed_pipeline_duration_ms: Transaction-to-commit wall time.
    """

    room_id: int
    destination_rank: int
    upstream_wait_duration_ms: float
    destination_scatter_copy_duration_ms: float
    finalize_duration_ms: float
    packed_pipeline_duration_ms: float

    def __str__(self) -> str:
        """Render the stable production log contract.

        :returns: Parser-compatible structured timing record.
        """

        return (
            f"PackedTransferStats(room={self.room_id}, role=decode, "
            f"destination_rank={self.destination_rank}, "
            f"upstream_wait_duration={self.upstream_wait_duration_ms:.3f}ms, "
            "destination_scatter_copy_duration="
            f"{self.destination_scatter_copy_duration_ms:.3f}ms, "
            f"finalize_duration={self.finalize_duration_ms:.3f}ms, "
            f"packed_pipeline_duration={self.packed_pipeline_duration_ms:.3f}ms)"
        )


@dataclasses.dataclass
class _PrefillRequestRecord:
    """Source actor state retained until authenticated decoder teardown."""

    submission: PackedPrefillSubmission
    writer_id: StagingWriterId
    chunk_key: PackedChunkKey
    condition: threading.Condition = dataclasses.field(
        default_factory=threading.Condition
    )
    source_transfer: PackedSourceTransfer | None = None
    failure_reason: str | None = None
    main_lane: PackedTransferLane | None = None
    main_handle: object | None = None
    auxiliary_handle: object | None = None
    main_outcome: PackedWriterOutcome | None = None
    auxiliary_outcome: PackedAuxiliaryOutcome | None = None
    outcomes_sent: bool = False
    main_handle_released: bool = False
    auxiliary_handle_released: bool = False
    ready_wait_duration_ms: float | None = None
    source_gather_copy_duration_ms: float | None = None
    main_transport_started_at: float | None = None
    main_transport_duration_ms: float | None = None


class PackedPrefillRuntime:
    """Persistent source actor for one-chunk packed request transfers."""

    def __init__(
        self,
        manager: PackedRuntimeManager,
        runtime_artifacts: PackedNixlRuntimeArtifactCohort,
        visibility_policy: PackedDestinationVisibilityPolicy,
    ) -> None:
        """Initialize the process-lifetime source actor.

        :param manager: Owning NIXL manager.
        :param runtime_artifacts: Exact native runtime cohort.
        :param visibility_policy: Same-host route visibility policy.
        """

        if manager.attn_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES:
            raise ValueError("packed source actor requires a supported source TP width")
        if manager.attn_cp_rank != 0 or manager.pp_rank != 0:
            raise ValueError("packed source actor requires CP1 and PP1")
        manifest = DecodeWriterManifest.for_tensor_parallel(manager.attn_tp_size)
        writer_id = StagingWriterId(
            transfer_source_rank=manager.transfer_source_rank,
            source_attn_tp_rank=manager.attn_tp_rank,
            source_pp_rank=manager.pp_rank,
            source_cp_rank=manager.attn_cp_rank,
        )
        if writer_id != manifest.writers[manager.attn_tp_rank]:
            raise ValueError(
                "packed source rank identity differs from the canonical manifest"
            )
        self._manager = manager
        self._runtime_artifacts = runtime_artifacts
        self._visibility_policy = visibility_policy
        self._manifest = manifest
        self._writer_id = writer_id
        self._ready = PackedReadyCoordinator()
        self._copy_executor: PackedCopyExecutor | None = None
        self._lanes: dict[
            PackedDestinationRouteBinding,
            list[PackedTransferLane],
        ] = {}
        self._records: dict[PackedRequestKey, _PrefillRequestRecord] = {}
        self._failed_records: list[_PrefillRequestRecord] = []
        self._lock = threading.RLock()

    @property
    def writer_id(self) -> StagingWriterId:
        """Return this process's canonical source identity.

        :returns: Exact writer identity.
        """

        return self._writer_id

    def build_destination_capability(
        self,
        *,
        advertisement: PackedRegistrationAdvertisement,
        decode_peer: PackedPeerIdentity,
        destination_gpu_id: int,
        destination_tp_size: int,
        destination_tp_rank: int,
        request_generation: bytes,
    ) -> PackedDestinationCapability:
        """Bind decoder registration metadata to one request generation.

        :param advertisement: Persistent decoder arena advertisement.
        :param decode_peer: Generation-bound decoder NIXL process.
        :param destination_gpu_id: Decoder CUDA device identifier.
        :param destination_tp_size: Decoder attention TP width.
        :param destination_tp_rank: Decoder attention TP rank.
        :param request_generation: Exact allocation-derived request generation.
        :returns: Request-scoped source validation capability.
        """

        if advertisement.visibility_policy_digest != self._visibility_policy.digest:
            raise ValueError("decoder visibility policy differs from source policy")
        if advertisement.runtime_cohort_digest != self._runtime_artifacts.digest:
            raise ValueError("decoder runtime cohort differs from source runtime")
        topology = PackedTopology(
            source_tp_size=self._manifest.source_tp_size,
            destination_tp_size=destination_tp_size,
            destination_tp_rank=destination_tp_rank,
        )
        self._destination_writer_manifest(topology)
        route = PackedDestinationRouteBinding(
            peer=decode_peer,
            arena_generation=advertisement.arena_generation,
            destination_gpu_id=destination_gpu_id,
            topology=topology,
            visibility_policy_digest=advertisement.visibility_policy_digest,
            base_address=advertisement.base_address,
            total_size=advertisement.total_size,
            alignment_bytes=topology.alignment_bytes,
        )
        return PackedDestinationCapability(
            route=route,
            request_generation=request_generation,
        )

    def _destination_writer_manifest(
        self,
        topology: PackedTopology,
    ) -> DecodeWriterManifest:
        """Resolve the exact writer cohort routed to one destination rank.

        :param topology: Request-scoped source and destination topology.
        :returns: Destination-rank-local canonical writer manifest.
        """

        manifest = DecodeWriterManifest.for_tensor_parallel(
            topology.source_tp_size,
            topology.destination_tp_size,
            topology.destination_tp_rank,
        )
        if self._writer_id not in manifest.writers:
            raise ValueError(
                "packed destination is not connected to this source writer"
            )
        return manifest

    def execute(self, submission: PackedPrefillSubmission) -> None:
        """Execute one complete source request through terminal outcomes.

        The request record remains live after this method returns. Native
        handles are released only when the decoder sends authenticated teardown.

        :param submission: Complete source-local request inputs.
        """

        record = self._prepare_submission(submission)
        try:
            record.submission.control.send_message(self._register_prepare(record))
            ready_wait_started_at = time.perf_counter()
            transfer = self._wait_for_ready(record)
            record.ready_wait_duration_ms = (
                time.perf_counter() - ready_wait_started_at
            ) * 1000.0
            if self._writer_id == submission.plan.canonical_writer_id:
                auxiliary_handle = self._post_auxiliary_transfer(record)
                with record.condition:
                    record.auxiliary_handle = auxiliary_handle
            main_handle, main_lane = self._post_main_transfer(record, transfer)
            with record.condition:
                record.main_handle = main_handle
                record.main_lane = main_lane

            main_outcome = self._wait_for_main_outcome(record)
            main_transport_started_at = record.main_transport_started_at
            if main_transport_started_at is not None:
                record.main_transport_duration_ms = (
                    time.perf_counter() - main_transport_started_at
                ) * 1000.0
            auxiliary_outcome: PackedAuxiliaryOutcome | None = None
            if self._writer_id == submission.plan.canonical_writer_id:
                auxiliary_outcome = self._wait_for_auxiliary_outcome(record)
            with record.condition:
                record.main_outcome = main_outcome
                record.auxiliary_outcome = auxiliary_outcome
                record.outcomes_sent = True
            submission.control.send_message(main_outcome)
            if auxiliary_outcome is not None:
                submission.control.send_message(auxiliary_outcome)
            self._emit_transfer_stats(record)
        except PackedGatherError as error:
            with record.condition:
                record.main_outcome = error.outcome
                record.outcomes_sent = True
            submission.control.send_message(error.outcome)
            self._fail_prefill_record(record, "packed source gather failed")
            self._retire_failed_record(record)
            raise
        except Exception:
            reason = "packed source request execution failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._fail_prefill_record(record, reason)
            self._retire_failed_record(record)
            raise

    def handle_control(
        self,
        authenticated_decode_peer: PackedPeerIdentity,
        message: PackedWireMessage,
    ) -> None:
        """Dispatch one generation-authenticated decoder control message.

        :param authenticated_decode_peer: Decoder derived from registration
            state, never from payload fields.
        :param message: Validated packed wire payload.
        """

        request_key = _request_key_for_message(message)
        with self._lock:
            record = self._records.get(request_key)
        if record is None:
            raise RuntimeError("packed decoder control references an unknown request")
        if authenticated_decode_peer != record.submission.control.peer:
            raise RuntimeError("packed decoder control came from another process")
        try:
            if type(message) is PackedReady:
                transfer = self._ready.handle_ready(
                    message,
                    authenticated_decode_peer,
                )
                with record.condition:
                    if record.source_transfer is not None:
                        raise RuntimeError("packed READY was duplicated")
                    record.source_transfer = transfer
                    record.condition.notify_all()
                return
            if type(message) is PackedRequestTeardown:
                self._handle_teardown(record, message)
                return
            raise RuntimeError(
                f"prefill received unsupported packed message {type(message).__name__}"
            )
        except Exception:
            reason = "packed prefill control dispatch failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._fail_prefill_record(record, reason)
            raise

    def _prepare_submission(
        self,
        submission: PackedPrefillSubmission,
    ) -> _PrefillRequestRecord:
        plan = submission.plan
        route = submission.destination.route
        if plan.destination_process_generation != route.peer.agent_generation:
            raise ValueError("packed auxiliary plan targets another decoder process")
        if plan.native_route_digest != self._visibility_policy.digest:
            raise ValueError("packed auxiliary plan targets another native route")
        if plan.runtime_cohort_digest != self._runtime_artifacts.digest:
            raise ValueError("packed auxiliary plan targets another runtime cohort")
        if route.visibility_policy_digest != self._visibility_policy.digest:
            raise ValueError("packed destination route has another visibility policy")
        if route.topology.source_tp_size != self._manifest.source_tp_size:
            raise ValueError("packed destination route has another source TP width")
        writer_manifest = self._destination_writer_manifest(route.topology)
        if plan.canonical_writer_id != writer_manifest.writers[0]:
            raise ValueError(
                "packed auxiliary plan names another destination cohort's "
                "canonical writer"
            )
        chunk_key = PackedChunkKey(
            room_id=plan.key.room_id,
            chunk_id=0,
            request_generation=plan.key.request_generation,
        )
        record = _PrefillRequestRecord(
            submission=submission,
            writer_id=self._writer_id,
            chunk_key=chunk_key,
        )
        with self._lock:
            if plan.key in self._records:
                raise RuntimeError("packed source request identity was reused")
            self._records[plan.key] = record
        return record

    def _register_prepare(self, record: _PrefillRequestRecord) -> PackedPrepare:
        submission = record.submission
        topology = submission.destination.route.topology
        writer_manifest = self._destination_writer_manifest(topology)
        spec, source_binding = build_prefill_chunk(
            key=record.chunk_key,
            is_last=True,
            kv_args=self._manager.kv_args,
            destination_registration=submission.destination_registration,
            components=submission.components,
            source_tp_size=self._manifest.source_tp_size,
            destination_tp_size=topology.destination_tp_size,
            destination_tp_rank=topology.destination_tp_rank,
            writers=writer_manifest.writers,
        )
        return self._ready.register_chunk(
            key=record.chunk_key,
            destination=submission.destination,
            writer_id=self._writer_id,
            spec=spec,
            source_binding=source_binding,
        )

    def _wait_for_ready(self, record: _PrefillRequestRecord) -> PackedSourceTransfer:
        deadline = time.monotonic() + PACKED_CONTROL_TIMEOUT_SECONDS
        with record.condition:
            while record.source_transfer is None and record.failure_reason is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                record.condition.wait(remaining)
            if record.failure_reason is not None:
                raise RuntimeError(record.failure_reason)
            transfer = record.source_transfer
            if transfer is None:
                raise TimeoutError("packed READY timed out")
            return transfer

    def _post_main_transfer(
        self,
        record: _PrefillRequestRecord,
        transfer: PackedSourceTransfer,
    ) -> tuple[object, PackedTransferLane]:
        lane = self._acquire_lane(transfer)
        executor = self._source_copy_executor()
        gather_started_at = time.perf_counter()
        executor.gather(
            transfer=transfer,
            source_lane=lane,
            producer_stream=record.submission.producer_stream,
        )
        record.source_gather_copy_duration_ms = (
            time.perf_counter() - gather_started_at
        ) * 1000.0
        agent = self._manager.agent
        try:
            source_requests = np.asarray(
                [[lane.data_ptr, transfer.length_bytes, self._manager.kv_args.gpu_id]],
                dtype=np.uint64,
            )
            destination_requests = np.asarray(
                [
                    [
                        transfer.destination_address,
                        transfer.length_bytes,
                        transfer.destination.route.destination_gpu_id,
                    ]
                ],
                dtype=np.uint64,
            )
            source_descriptors = agent.get_xfer_descs(source_requests, "VRAM")
            destination_descriptors = agent.get_xfer_descs(
                destination_requests,
                "VRAM",
            )
            handle = agent.initialize_xfer(
                "WRITE",
                source_descriptors,
                destination_descriptors,
                record.submission.control.remote_handle,
                b"",
            )
            if handle is None:
                raise RuntimeError("NIXL returned no packed main transfer handle")
        except Exception as error:
            logger.error(
                "Packed main transfer initialization failed:\n%s",
                traceback.format_exc(),
            )
            outcome = lane.abort_before_submit(
                "packed main transfer initialization failed"
            )
            raise PackedGatherError(outcome) from error
        lane.arm_submission(
            handle,
            owners=(record.submission.control.remote_handle,),
        )
        with record.condition:
            record.main_handle = handle
            record.main_lane = lane
        try:
            record.main_transport_started_at = time.perf_counter()
            self._manager._post_transfer_when_ready(
                handle,
                "packed main NIXL transfer",
            )
        except Exception:
            lane.mark_submission_ambiguous(
                "packed main NIXL submission became ambiguous"
            )
            raise
        return handle, lane

    def _emit_transfer_stats(self, record: _PrefillRequestRecord) -> None:
        ready_wait_duration_ms = record.ready_wait_duration_ms
        source_gather_copy_duration_ms = record.source_gather_copy_duration_ms
        main_transport_duration_ms = record.main_transport_duration_ms
        if (
            ready_wait_duration_ms is None
            or source_gather_copy_duration_ms is None
            or main_transport_duration_ms is None
        ):
            logger.error(
                "PackedTransferStatsUnavailable(room=%d, role=prefill, source_rank=%d)",
                record.chunk_key.room_id,
                record.writer_id.source_attn_tp_rank,
            )
            return
        logger.info(
            "%s",
            _PackedPrefillTransferStats(
                room_id=record.chunk_key.room_id,
                source_rank=record.writer_id.source_attn_tp_rank,
                ready_wait_duration_ms=ready_wait_duration_ms,
                source_gather_copy_duration_ms=source_gather_copy_duration_ms,
                main_transport_duration_ms=main_transport_duration_ms,
            ),
        )

    def _wait_for_main_outcome(
        self,
        record: _PrefillRequestRecord,
    ) -> PackedWriterOutcome:
        lane = record.main_lane
        if lane is None:
            raise RuntimeError("packed main lane was not retained")
        deadline = time.monotonic() + PACKED_CONTROL_TIMEOUT_SECONDS
        while True:
            outcome = lane.take_transport_completion()
            if outcome is not None:
                return outcome
            if time.monotonic() >= deadline:
                lane.mark_submission_ambiguous("packed main NIXL completion timed out")
                raise TimeoutError("packed main NIXL completion timed out")
            time.sleep(0)

    def _post_auxiliary_transfer(self, record: _PrefillRequestRecord) -> object:
        plan = record.submission.plan
        source_ptrs = tuple(self._manager.kv_args.aux_data_ptrs)
        source_item_lens = tuple(self._manager.kv_args.aux_item_lens)
        if len(source_ptrs) != len(source_item_lens):
            raise ValueError("source auxiliary pointers and item lengths differ")
        if len(source_ptrs) != len(plan.destination_segments):
            raise ValueError("source and destination auxiliary segment counts differ")
        source_requests: list[tuple[int, int, int]] = []
        destination_requests: list[tuple[int, int, int]] = []
        for source_ptr, source_length, destination in zip(
            source_ptrs,
            source_item_lens,
            plan.destination_segments,
            strict=True,
        ):
            if source_length != destination.item_length:
                raise ValueError("source and destination auxiliary lengths differ")
            source_requests.append(
                (
                    source_ptr
                    + source_length * record.submission.auxiliary_source_index,
                    source_length,
                    0,
                )
            )
            destination_requests.append(
                (destination.address, destination.item_length, 0)
            )
        agent = self._manager.agent
        source_descriptors = agent.get_xfer_descs(source_requests, "DRAM")
        destination_descriptors = agent.get_xfer_descs(
            destination_requests,
            "DRAM",
        )
        handle = agent.initialize_xfer(
            "WRITE",
            source_descriptors,
            destination_descriptors,
            record.submission.control.remote_handle,
            b"",
        )
        if handle is None:
            raise RuntimeError("NIXL returned no packed auxiliary transfer handle")
        with record.condition:
            record.auxiliary_handle = handle
        self._manager._post_transfer_when_ready(
            handle,
            "packed auxiliary NIXL transfer",
        )
        return handle

    def _wait_for_auxiliary_outcome(
        self,
        record: _PrefillRequestRecord,
    ) -> PackedAuxiliaryOutcome:
        handle = record.auxiliary_handle
        if handle is None:
            raise RuntimeError("packed auxiliary handle was not retained")
        deadline = time.monotonic() + PACKED_CONTROL_TIMEOUT_SECONDS
        receipt: PackedNixlCompletionReceipt | None = None
        while receipt is None:
            receipt = self._manager.agent.take_xfer_completion_receipt(handle)
            if receipt is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("packed auxiliary NIXL completion timed out")
            time.sleep(0)
        generation, descriptor_digest, evidence_digest = (
            self._validate_auxiliary_receipt(record, receipt)
        )
        return PackedAuxiliaryOutcome(
            plan=record.submission.plan,
            writer_id=self._writer_id,
            native_dram_handle_generation=generation,
            descriptor_digest=descriptor_digest,
            evidence_digest=evidence_digest,
        )

    def _validate_auxiliary_receipt(
        self,
        record: _PrefillRequestRecord,
        receipt: PackedNixlCompletionReceipt,
    ) -> tuple[int, bytes, bytes]:
        if type(receipt.generation) is not int or receipt.generation <= 0:
            raise ValueError("auxiliary receipt has an invalid handle generation")
        if not receipt.submissionSealed or not receipt.completionClaimed:
            raise ValueError("auxiliary receipt did not seal terminal completion")
        self._require_native_enum(
            receipt.state,
            "NIXL_XFER_ATTESTATION_REMOTE_FLUSHED",
            "state",
        )
        self._require_native_enum(receipt.status, "NIXL_SUCCESS", "status")
        self._require_native_enum(receipt.operation, "NIXL_WRITE", "operation")
        self._require_native_enum(
            receipt.localMemoryType,
            "DRAM_SEG",
            "local memory type",
        )
        self._require_native_enum(
            receipt.remoteMemoryType,
            "DRAM_SEG",
            "remote memory type",
        )
        if receipt.backend != "UCX":
            raise ValueError("auxiliary receipt backend is not UCX")
        if receipt.localAgent != self._manager.agent.name:
            raise ValueError("auxiliary receipt belongs to another source agent")
        if receipt.remoteAgent != record.submission.control.peer.agent_name:
            raise ValueError("auxiliary receipt belongs to another decoder agent")
        if type(receipt.error) is not str or len(receipt.error) != 0:
            raise ValueError("successful auxiliary receipt contains an error")
        segments = receipt.segments
        plan_segments = record.submission.plan.destination_segments
        if type(segments) is not tuple or len(segments) != len(plan_segments):
            raise ValueError("auxiliary receipt segment count differs from its plan")
        source_index = record.submission.auxiliary_source_index
        for index, (segment, destination) in enumerate(
            zip(segments, plan_segments, strict=True)
        ):
            source_length = self._manager.kv_args.aux_item_lens[index]
            expected_source = (
                self._manager.kv_args.aux_data_ptrs[index]
                + source_length * source_index
            )
            if (
                segment.index != index
                or segment.localAddress != expected_source
                or segment.remoteAddress != destination.address
                or segment.length != destination.item_length
                or not segment.posted
            ):
                raise ValueError("auxiliary receipt descriptors differ from the plan")
        return (
            receipt.generation,
            self._decode_native_digest(receipt.descriptorDigest, "descriptor"),
            self._decode_native_digest(receipt.evidenceDigest, "evidence"),
        )

    def _handle_teardown(
        self,
        record: _PrefillRequestRecord,
        request: PackedRequestTeardown,
    ) -> None:
        if request.writer_id != self._writer_id:
            raise RuntimeError("packed teardown targets another source writer")
        with record.condition:
            if not record.outcomes_sent:
                raise RuntimeError("packed teardown preceded terminal outcomes")
            main_outcome = record.main_outcome
            auxiliary_outcome = record.auxiliary_outcome
            main_handle = record.main_handle
            auxiliary_handle = record.auxiliary_handle
        if (
            main_outcome is None
            or main_outcome.status is not PackedWriterOutcomeStatus.DONE
            or main_handle is None
        ):
            raise RuntimeError("packed teardown lacks a successful main transfer")
        if not record.main_handle_released:
            self._manager.agent.release_xfer_handle(main_handle)
            record.main_handle_released = True
        if self._writer_id == record.submission.plan.canonical_writer_id:
            if auxiliary_outcome is None or auxiliary_handle is None:
                raise RuntimeError("canonical teardown lacks auxiliary ownership")
            if (
                request.auxiliary_handle_generation
                != auxiliary_outcome.native_dram_handle_generation
            ):
                raise RuntimeError("canonical teardown names another auxiliary handle")
            if not record.auxiliary_handle_released:
                self._manager.agent.release_xfer_handle(auxiliary_handle)
                record.auxiliary_handle_released = True
        elif request.auxiliary_handle_generation is not None:
            raise RuntimeError("noncanonical teardown names an auxiliary handle")
        acknowledgement = PackedRequestTeardownAck(
            key=request.key,
            writer_id=request.writer_id,
            request_slot_generation=request.request_slot_generation,
            writer_manifest_digest=request.writer_manifest_digest,
            allocation_digest=request.allocation_digest,
            teardown_generation=request.teardown_generation,
            auxiliary_handle_generation=request.auxiliary_handle_generation,
        )
        record.submission.control.send_message(acknowledgement)
        with self._lock:
            current = self._records.get(request.key)
            if current is not record:
                raise RuntimeError("packed source registry ownership changed")
            del self._records[request.key]

    def _acquire_lane(self, transfer: PackedSourceTransfer) -> PackedTransferLane:
        route = transfer.destination.route
        with self._lock:
            lanes = self._lanes.setdefault(route, [])
            for lane in lanes:
                if (
                    lane.state is PackedTransferLaneState.IDLE
                    and lane.capacity >= transfer.length_bytes
                ):
                    return lane
            capacity = max(
                PACKED_SOURCE_LANE_MINIMUM_BYTES,
                1 << (transfer.length_bytes - 1).bit_length(),
            )
            tensor = torch.empty(
                capacity,
                dtype=torch.uint8,
                device=f"cuda:{self._manager.kv_args.gpu_id}",
            )
            lane = PackedTransferLane(
                agent=self._manager.agent,
                destination_route=route,
                visibility_policy=self._visibility_policy,
                gpu_id=self._manager.kv_args.gpu_id,
                tensor=tensor,
                expected_runtime_artifacts=self._runtime_artifacts,
            )
            lanes.append(lane)
            return lane

    def _source_copy_executor(self) -> PackedCopyExecutor:
        with self._lock:
            executor = self._copy_executor
            if executor is None:
                executor = PackedCopyExecutor(gpu_id=self._manager.kv_args.gpu_id)
                self._copy_executor = executor
            return executor

    def _fail_prefill_record(
        self,
        record: _PrefillRequestRecord,
        reason: str,
    ) -> None:
        with record.condition:
            if record.failure_reason is None:
                record.failure_reason = reason
            record.condition.notify_all()
        room = record.chunk_key.room_id
        self._manager.record_failure(room, reason)
        self._manager.update_status(room, KVPoll.Failed)

    def _retire_failed_record(self, record: _PrefillRequestRecord) -> None:
        self._ready.retire_pending(
            record.chunk_key,
            record.submission.control.peer,
        )
        with self._lock:
            current = self._records.get(record.submission.plan.key)
            if current is record:
                del self._records[record.submission.plan.key]
                if (
                    record.main_handle is not None
                    or record.auxiliary_handle is not None
                ):
                    self._failed_records.append(record)

    @staticmethod
    def _decode_native_digest(value: str, label: str) -> bytes:
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"native auxiliary {label} digest is not SHA-256")
        try:
            digest = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(
                f"native auxiliary {label} digest is not hexadecimal"
            ) from error
        if len(digest) != 32:
            raise ValueError(f"native auxiliary {label} digest is not SHA-256")
        return digest

    @staticmethod
    def _require_native_enum(
        value: PackedNixlEnumValue,
        expected: str,
        label: str,
    ) -> None:
        if type(value.name) is not str or value.name != expected:
            raise ValueError(f"native auxiliary receipt {label} is not {expected}")


@dataclasses.dataclass
class _DecodeRequestRecord:
    """Decode actor state retained through scheduler metadata consumption."""

    transaction: PackedDecodeRequestTransaction
    auxiliary_allocation: AdoptedPackedAuxiliarySlotAllocation
    chunk_keys: tuple[PackedChunkKey, ...]
    routes: dict[StagingWriterId, PackedControlSender] = dataclasses.field(
        default_factory=dict
    )
    scatters: dict[PackedChunkKey, tuple[object, PackedScatterSubmission]] = (
        dataclasses.field(default_factory=dict)
    )
    commit_receipt: PackedRequestCommitReceipt | None = None
    teardown_sent: bool = False
    pipeline_started_at: float = dataclasses.field(default_factory=time.perf_counter)
    upstream_wait_duration_ms: float | None = None
    destination_scatter_copy_duration_ms: float = 0.0
    finalize_started_at: float | None = None
    stats_emitted: bool = False


class PackedDecodeRuntime:
    """Persistent decode actor for authenticated packed request transactions."""

    def __init__(
        self,
        manager: PackedRuntimeManager,
        arena: PackedStagingArena,
        runtime_artifacts: PackedNixlRuntimeArtifactCohort,
        visibility_policy: PackedDestinationVisibilityPolicy,
    ) -> None:
        """Initialize the process-lifetime decode actor.

        :param manager: Owning NIXL manager.
        :param arena: Persistent adopted staging arena.
        :param runtime_artifacts: Exact native runtime cohort.
        :param visibility_policy: Same-host route visibility policy.
        """

        if manager.attn_tp_size not in (1, 2):
            raise ValueError("packed decode actor supports only TP1 and TP2")
        if manager.attn_tp_rank < 0 or manager.attn_tp_rank >= manager.attn_tp_size:
            raise ValueError("packed decode actor has an invalid attention TP rank")
        self._manager = manager
        self._arena = arena
        self._runtime_artifacts = runtime_artifacts
        self._visibility_policy = visibility_policy
        self._outcomes = PackedDestinationOutcomeCoordinator(
            arena.protocol,
            _UnexpectedVisibilityActionExecutor(),
        )
        self._auxiliary_authority: PackedAuxiliaryAllocationLeaseAuthority | None = None
        self._metadata_allocator: PackedMetadataIndexAllocator | None = None
        self._consumer_authority: object | None = None
        self._records: dict[PackedRequestKey, _DecodeRequestRecord] = {}
        self._records_by_room: dict[int, PackedRequestKey] = {}
        self._lock = threading.RLock()

    @property
    def advertisement(self) -> PackedRegistrationAdvertisement:
        """Return persistent packed registration metadata.

        :returns: Exact arena and route advertisement.
        """

        # Process advertisement omits request topology. This reflexive route
        # extracts only arena-owned fields; the authenticated source rebuilds
        # the exact TP1/TP2/TP4 request capability before transfer.
        topology = PackedTopology(
            source_tp_size=self._manager.attn_tp_size,
            destination_tp_size=self._manager.attn_tp_size,
            destination_tp_rank=self._manager.attn_tp_rank,
        )
        capability = self._arena.capability(
            request_generation=b"\x01" * 16,
            topology=topology,
            visibility_policy=self._visibility_policy,
        )
        route = capability.route
        return PackedRegistrationAdvertisement(
            base_address=route.base_address,
            total_size=route.total_size,
            arena_generation=route.arena_generation,
            visibility_policy_digest=route.visibility_policy_digest,
            runtime_cohort_digest=self._runtime_artifacts.digest,
            page_size=self._manager.kv_args.page_size,
        )

    @property
    def ready(self) -> bool:
        """Return whether scheduler metadata ownership is attached.

        :returns: Runtime admission readiness.
        """

        with self._lock:
            return self._auxiliary_authority is not None

    def attach_scheduler(
        self,
        metadata_allocator: PackedMetadataIndexAllocator,
        consumer_authority: object,
    ) -> None:
        """Bind exact scheduler metadata ownership once.

        :param metadata_allocator: Existing row allocator.
        :param consumer_authority: Decode transfer queue consuming row contents.
        """

        with self._lock:
            if self._auxiliary_authority is not None:
                if (
                    self._metadata_allocator is metadata_allocator
                    and self._consumer_authority is consumer_authority
                ):
                    return
                raise RuntimeError("packed decode scheduler ownership changed")
            self._metadata_allocator = metadata_allocator
            self._consumer_authority = consumer_authority
            self._auxiliary_authority = PackedAuxiliaryAllocationLeaseAuthority(
                self._manager,
                consumer_authority,
            )

    def prepare_transaction(
        self,
        *,
        room_id: int,
        request_owner: object,
        metadata_buffer_index: int,
        allocation_lease: DecodeAllocationLease,
        allocation_authority: DecodeAllocationLeaseAuthority,
        lifecycle_authority: object,
        source_tp_size: int,
    ) -> PackedDecodeRequestTransaction:
        """Construct and register one decoder-authored one-chunk request.

        :param room_id: Decoder-minted room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Already reserved metadata row.
        :param allocation_lease: Exact pinned decode allocation.
        :param allocation_authority: Exact allocation authority.
        :param lifecycle_authority: Trusted transport lifecycle authority.
        :param source_tp_size: Supported packed writer width.
        :returns: Complete prepared transaction.
        """

        with self._lock:
            auxiliary_authority = self._auxiliary_authority
            metadata_allocator = self._metadata_allocator
            if auxiliary_authority is None or metadata_allocator is None:
                raise RuntimeError("packed decode scheduler ownership is unavailable")
            if room_id in self._records_by_room:
                raise RuntimeError("packed decode room is already registered")

        snapshot = allocation_authority.snapshot(allocation_lease)
        expected_manifest = DecodeWriterManifest.for_tensor_parallel(
            source_tp_size,
            self._manager.attn_tp_size,
            self._manager.attn_tp_rank,
        )
        if snapshot.writer_manifest != expected_manifest:
            raise ValueError(
                "decode allocation writer manifest differs from the runtime "
                "destination topology"
            )
        receipts = {receipt.component: receipt for receipt in snapshot.components}
        full = receipts[DecodeAllocationComponent.FULL]
        swa = receipts[DecodeAllocationComponent.SWA]
        page_arrays: dict[StagingComponentId, npt.NDArray[np.int32]] = {
            MAIN_KV_COMPONENT: np.asarray(full.physical_pages, dtype=np.int32)
        }
        spans = [
            StagingComponentSpan(
                component_id=MAIN_KV_COMPONENT,
                source_index_offset=0,
                destination_index_offset=0,
                logical_token_count=len(full.physical_pages) * full.page_size,
                physical_token_count=len(full.physical_pages) * full.page_size,
            )
        ]
        if not swa.zero_work:
            swa_state_indices = tuple(
                index
                for index, state_type in enumerate(self._manager.kv_args.state_types)
                if state_type is StateType.SWA
            )
            if len(swa_state_indices) != 1:
                raise RuntimeError("packed Gemma request requires one SWA component")
            swa_component = StagingComponentId(
                state_index=swa_state_indices[0],
                state_type=StateType.SWA,
            )
            page_arrays[swa_component] = np.asarray(
                swa.physical_pages,
                dtype=np.int32,
            )
            spans.append(
                StagingComponentSpan(
                    component_id=swa_component,
                    source_index_offset=0,
                    destination_index_offset=0,
                    logical_token_count=len(swa.physical_pages) * swa.page_size,
                    physical_token_count=len(swa.physical_pages) * swa.page_size,
                )
            )
        spec = build_decode_spec(
            chunk_id=0,
            is_last=True,
            spans=tuple(spans),
            kv_args=self._manager.kv_args,
            expected_writers=expected_manifest.writers,
            source_tp_size=source_tp_size,
            destination_tp_size=self._manager.attn_tp_size,
            destination_tp_rank=self._manager.attn_tp_rank,
        )
        key = PackedChunkKey(
            room_id=room_id,
            chunk_id=0,
            request_generation=snapshot.lease_id,
        )
        plan = PackedRequestChunkPlan(
            key=key,
            spec=spec,
            destination_registry=build_component_buffer_registry(
                self._manager.kv_args,
                page_arrays,
            ),
            visibility_policies=tuple(
                (writer, self._visibility_policy)
                for writer in snapshot.writer_manifest.writers
            ),
        )
        auxiliary_allocation = AdoptedPackedAuxiliarySlotAllocation(
            metadata_allocator,
            metadata_buffer_index,
            self._manager.kv_args,
        )
        auxiliary_lease = auxiliary_authority.acquire(auxiliary_allocation)
        transaction = PackedDecodeRequestTransaction(
            room_id=room_id,
            request_owner=request_owner,
            allocation_lease=allocation_lease,
            allocation_authority=allocation_authority,
            lifecycle_authority=lifecycle_authority,
            protocol=self._arena.protocol,
            outcome_coordinator=self._outcomes,
            chunk_plans=(plan,),
            auxiliary_allocation_lease=auxiliary_lease,
            auxiliary_allocation_authority=auxiliary_authority,
            destination_process_generation=uuid.UUID(
                self._manager.process_generation
            ).bytes,
            native_route_digest=self._visibility_policy.digest,
            runtime_cohort_digest=self._runtime_artifacts.digest,
        )
        request_key = PackedRequestKey.from_chunk_key(key)
        with self._lock:
            if request_key in self._records or room_id in self._records_by_room:
                transaction.cancel_unpublished()
                raise RuntimeError("packed decode request identity was reused")
            self._records[request_key] = _DecodeRequestRecord(
                transaction=transaction,
                auxiliary_allocation=auxiliary_allocation,
                chunk_keys=(key,),
            )
            self._records_by_room[room_id] = request_key
        return transaction

    def cancel_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        """Cancel and retire one unpublished transaction.

        :param transaction: Exact prepared transaction.
        :returns: Exact retained request owner.
        """

        record = self._record_for_transaction(transaction)
        owner = transaction.cancel_unpublished()
        self._retire_record(record)
        return owner

    def bind_publication(
        self,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        routes: tuple[PackedControlSender, ...],
    ) -> None:
        """Bind authenticated writer routes after receiver activation.

        :param transaction: Exact published transaction.
        :param publication: Matching irreversible publication.
        :param routes: Complete writer control routes.
        """

        record = self._record_for_transaction(transaction)
        if publication.key != record.transaction.snapshot().key:
            raise RuntimeError("packed publication belongs to another transaction")
        route_map = {route.writer_id: route for route in routes}
        expected_writers = publication.chunk_specs[0].writers
        if tuple(sorted(route_map)) != expected_writers:
            raise RuntimeError("packed control routes differ from writer manifest")
        with self._lock:
            if len(record.routes) > 0:
                raise RuntimeError("packed control routes were already bound")
            record.routes = route_map

    def handle_control(
        self,
        authenticated_writer_id: StagingWriterId,
        message: PackedWireMessage,
    ) -> None:
        """Dispatch one peer-authenticated source control message.

        :param authenticated_writer_id: Writer derived from native peer state.
        :param message: Validated packed payload.
        """

        request_key = _request_key_for_message(message)
        with self._lock:
            record = self._records.get(request_key)
        if record is None:
            raise RuntimeError(
                "packed control references an unknown request generation"
            )
        try:
            if type(message) is PackedPrepare:
                ready_messages = record.transaction.handle_prepare(
                    message,
                    authenticated_writer_id,
                )
                for ready in ready_messages:
                    route = record.routes.get(ready.writer_id)
                    if route is None:
                        raise RuntimeError("READY writer has no authenticated route")
                    route.send_message(ready)
                return
            if type(message) is PackedWriterOutcome:
                record.transaction.handle_writer_outcome(
                    message,
                    authenticated_writer_id,
                )
                return
            if type(message) is PackedAuxiliaryOutcome:
                record.transaction.handle_auxiliary_outcome(
                    message,
                    authenticated_writer_id,
                )
                return
            if type(message) is PackedRequestTeardownAck:
                receipt = record.transaction.handle_teardown_ack(
                    message,
                    authenticated_writer_id,
                )
                if receipt is not None:
                    with self._lock:
                        if record.commit_receipt is not None:
                            raise RuntimeError("packed commit receipt was duplicated")
                        record.commit_receipt = receipt
                return
            raise RuntimeError(
                f"decode received unsupported packed message {type(message).__name__}"
            )
        except Exception:
            reason = "packed decode control dispatch failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._fail_record(record, reason)
            raise

    def poll(self, transaction: PackedDecodeRequestTransaction) -> KVPoll:
        """Advance scheduler-owned scatter, teardown, and commit work.

        :param transaction: Exact request transaction.
        :returns: Scheduler transfer state.
        """

        record = self._record_for_transaction(transaction)
        try:
            if (
                transaction.state
                is PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
            ):
                return KVPoll.Success
            self._poll_scatters(record)
            self._begin_teardown_if_ready(record)
            receipt = record.commit_receipt
            if receipt is None:
                if transaction.state is PackedRequestTransactionState.QUARANTINED:
                    return KVPoll.Failed
                return KVPoll.Transferring
            owner = transaction.commit_on_scheduler_thread(receipt)
            if owner is not transaction.request_owner:
                raise RuntimeError("packed scheduler commit returned another request")
            record.commit_receipt = None
            self._emit_transfer_stats(record, time.perf_counter())
            return KVPoll.Success
        except Exception:  # noqa: BLE001
            reason = "packed decode scheduler progress failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._fail_record(record, reason)
            return KVPoll.Failed

    def complete_metadata_consumption(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Release the exact adopted row and retire committed actor state.

        :param transaction: Exact destination-consumption-waiting transaction.
        """

        record = self._record_for_transaction(transaction)
        consumer = self._consumer_authority
        if consumer is None:
            raise RuntimeError("packed metadata consumer authority is unavailable")
        transaction.complete_auxiliary_consumption_on_scheduler_thread(consumer)
        if not record.auxiliary_allocation.released:
            raise RuntimeError("packed metadata adapter did not release its row")
        self._retire_record(record)

    def quarantine(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Quarantine one complete request allocation.

        :param transaction: Exact request transaction.
        :param reason: Stable failure reason.
        """

        record = self._record_for_transaction(transaction)
        self._fail_record(record, reason)

    def _poll_scatters(self, record: _DecodeRequestRecord) -> None:
        transaction = record.transaction
        snapshot = transaction.snapshot()
        for chunk_state, key in zip(
            snapshot.chunk_states,
            record.chunk_keys,
            strict=True,
        ):
            active = record.scatters.get(key)
            if active is None:
                if chunk_state is not PackedProtocolState.SCATTER_READY:
                    continue
                if record.upstream_wait_duration_ms is None:
                    record.upstream_wait_duration_ms = (
                        time.perf_counter() - record.pipeline_started_at
                    ) * 1000.0
                scatter = transaction.begin_scatter(key)
                submission = self._arena.copy_executor.scatter(
                    scatter.work,
                    scatter.proofs,
                )
                record.scatters[key] = (scatter, submission)
                continue
            scatter, submission = active
            if not submission.event.query():
                continue
            scatter_completed_at = time.perf_counter()
            record.destination_scatter_copy_duration_ms += (
                submission.elapsed_milliseconds()
            )
            transaction.complete_scatter(scatter)
            del record.scatters[key]
            terminal_snapshot = transaction.snapshot()
            expected_terminal_chunks = tuple(
                chunk_key.chunk_id for chunk_key in record.chunk_keys
            )
            # Auxiliary completion can trail the final device scatter. Chunk
            # terminality is the stable boundary for final-scatter timing.
            if (
                terminal_snapshot.scatter_terminal == expected_terminal_chunks
                and record.finalize_started_at is None
            ):
                record.finalize_started_at = scatter_completed_at

    def _emit_transfer_stats(
        self,
        record: _DecodeRequestRecord,
        completed_at: float,
    ) -> None:
        if record.stats_emitted:
            return
        upstream_wait_duration_ms = record.upstream_wait_duration_ms
        finalize_started_at = record.finalize_started_at
        request_key = record.transaction.snapshot().key
        if upstream_wait_duration_ms is None or finalize_started_at is None:
            logger.error(
                "PackedTransferStatsUnavailable(room=%d, role=decode, "
                "destination_rank=%d)",
                request_key.room_id,
                self._manager.attn_tp_rank,
            )
            record.stats_emitted = True
            return
        logger.info(
            "%s",
            _PackedDecodeTransferStats(
                room_id=request_key.room_id,
                destination_rank=self._manager.attn_tp_rank,
                upstream_wait_duration_ms=upstream_wait_duration_ms,
                destination_scatter_copy_duration_ms=(
                    record.destination_scatter_copy_duration_ms
                ),
                finalize_duration_ms=(completed_at - finalize_started_at) * 1000.0,
                packed_pipeline_duration_ms=(completed_at - record.pipeline_started_at)
                * 1000.0,
            ),
        )
        record.stats_emitted = True

    def _begin_teardown_if_ready(self, record: _DecodeRequestRecord) -> None:
        if record.teardown_sent:
            return
        if (
            record.transaction.state
            is not PackedRequestTransactionState.SCATTER_COMPLETED
        ):
            return
        requests = record.transaction.begin_teardown()
        for request in requests:
            route = record.routes.get(request.writer_id)
            if route is None:
                raise RuntimeError("teardown writer has no authenticated route")
            route.send_message(request)
        record.teardown_sent = True

    def _record_for_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> _DecodeRequestRecord:
        key = transaction.snapshot().key
        with self._lock:
            record = self._records.get(key)
        if record is None or record.transaction is not transaction:
            raise RuntimeError("packed transaction is not registered by this actor")
        return record

    def _retire_record(self, record: _DecodeRequestRecord) -> None:
        key = record.transaction.snapshot().key
        with self._lock:
            current = self._records.get(key)
            if current is not record:
                raise RuntimeError("packed request registry ownership changed")
            del self._records[key]
            self._records_by_room.pop(key.room_id, None)

    def _fail_record(self, record: _DecodeRequestRecord, reason: str) -> None:
        try:
            record.transaction.quarantine(reason)
        except Exception:  # noqa: BLE001
            logger.error(
                "Packed transaction quarantine failed:\n%s",
                traceback.format_exc(),
            )
        room = record.transaction.snapshot().key.room_id
        self._manager.record_failure(room, reason)
        self._manager.update_status(room, KVPoll.Failed)


def _request_key_for_message(message: PackedWireMessage) -> PackedRequestKey:
    """Project any packed control message to its request identity.

    :param message: Validated closed wire message.
    :returns: Exact request identity.
    """

    if type(message) in (PackedPrepare, PackedReady, PackedWriterOutcome):
        return PackedRequestKey.from_chunk_key(message.key)
    if type(message) is PackedAuxiliaryPlan:
        return message.key
    if type(message) is PackedAuxiliaryOutcome:
        return message.plan.key
    if type(message) in (PackedRequestTeardown, PackedRequestTeardownAck):
        return message.key
    raise TypeError(f"unsupported packed message {type(message).__name__}")
