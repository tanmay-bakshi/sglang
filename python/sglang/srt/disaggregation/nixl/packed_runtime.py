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
from typing import Any, Protocol, runtime_checkable

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
    PackedDFlashBoundaryOutcome,
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
    writer_layout_for,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedDFlashBoundaryDecodeAdoption,
    PackedRequestChunkPlan,
    PackedRequestCommitReceipt,
    PackedRequestPublication,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFLASH_BOUNDARY_ROW_BYTES,
    DFlashBoundaryDeviceRowPool,
    DFlashBoundaryPrefillSource,
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
from sglang.srt.disaggregation.terminal_progress.nixl_owner_boundary import (
    NixlTerminalOwnerBoundary,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
    PackedTerminalSourcePlan,
    encode_packed_terminal_source_plan,
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

    def _post_terminal_transfer_once(self, handle: object, context: str) -> object:
        """Post one terminal handle after static peer enrollment.

        :param handle: Exact native-owner-armed handle.
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


@dataclasses.dataclass(frozen=True, slots=True)
class PackedLegacyAuxiliarySource:
    """Canonical legacy DRAM metadata row.

    :ivar row_index: Exact row copied from registered metadata buffers.
    """

    row_index: int

    def __post_init__(self) -> None:
        """Validate one non-negative source row."""

        if type(self.row_index) is not int or self.row_index < 0:
            raise ValueError("legacy auxiliary row_index must be non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDFlashAuxiliarySource:
    """Canonical terminal DFlash device-row ownership.

    :ivar prefill_source: Active all-VRAM row and frozen scalar counters.
    """

    prefill_source: DFlashBoundaryPrefillSource

    def __post_init__(self) -> None:
        """Validate one active DFlash source."""

        if type(self.prefill_source) is not DFlashBoundaryPrefillSource:
            raise TypeError("prefill_source must be DFlashBoundaryPrefillSource")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedNoncanonicalAuxiliarySource:
    """Explicit proof that this source rank owns no auxiliary transfer."""


PackedPrefillAuxiliarySource = (
    PackedLegacyAuxiliarySource
    | PackedTerminalDFlashAuxiliarySource
    | PackedNoncanonicalAuxiliarySource
)
PackedPrefillAuxiliaryOutcome = PackedAuxiliaryOutcome | PackedDFlashBoundaryOutcome


def _validate_prefill_launch_fields(
    *,
    plan: PackedAuxiliaryPlan,
    destination: PackedDestinationCapability,
    destination_registration: PackedDestinationRegistration,
    control: PackedDecodeControlSender,
    components: tuple[PackedComponentPages, ...],
    auxiliary_source: PackedPrefillAuxiliarySource,
) -> tuple[PackedComponentPages, ...]:
    """Validate the immutable source plan shared across launch and submission.

    :param plan: Decoder-authored auxiliary metadata plan.
    :param destination: Request-scoped destination capability.
    :param destination_registration: Decode cache geometry.
    :param control: Exact decoder route.
    :param components: Main-KV and SWA page projections.
    :param auxiliary_source: Explicit rank-local auxiliary ownership.
    :returns: Canonical immutable component tuple.
    """

    if type(plan) is not PackedAuxiliaryPlan:
        raise TypeError("packed source plan must be PackedAuxiliaryPlan")
    if type(destination) is not PackedDestinationCapability:
        raise TypeError("packed source destination must be PackedDestinationCapability")
    if type(destination_registration) is not PackedDestinationRegistration:
        raise TypeError(
            "packed source registration must be PackedDestinationRegistration"
        )
    if type(control) is not PackedDecodeControlSender:
        raise TypeError("packed source control must be PackedDecodeControlSender")
    canonical_components = tuple(components)
    if len(canonical_components) == 0:
        raise ValueError("packed source request must contain cache components")
    if any(
        type(component) is not PackedComponentPages
        for component in canonical_components
    ):
        raise TypeError("packed source components must be PackedComponentPages")
    auxiliary_types = (
        PackedLegacyAuxiliarySource,
        PackedTerminalDFlashAuxiliarySource,
        PackedNoncanonicalAuxiliarySource,
    )
    if type(auxiliary_source) not in auxiliary_types:
        raise TypeError("packed source auxiliary ownership is invalid")
    if type(auxiliary_source) is PackedTerminalDFlashAuxiliarySource:
        if len(plan.destination_segments) != 1:
            raise ValueError("terminal DFlash requires one destination row")
        if plan.destination_segments[0].item_length != DFLASH_BOUNDARY_ROW_BYTES:
            raise ValueError("terminal DFlash destination must contain eight bytes")
    if destination.request_generation != plan.key.request_generation:
        raise ValueError(
            "packed source capability generation differs from auxiliary plan"
        )
    if destination.route.peer != control.peer:
        raise ValueError("packed source control peer differs from destination")
    return canonical_components


@dataclasses.dataclass(frozen=True, slots=True)
class PackedPrefillLaunchPlan:
    """Complete source request ownership frozen before model submission.

    :ivar plan: Decoder-authored auxiliary metadata plan.
    :ivar destination: Request-scoped registered destination capability.
    :ivar destination_registration: Decode cache geometry used to rebuild the
        canonical layout.
    :ivar control: Exact decoder control and native data route.
    :ivar components: Main-KV and SWA source/destination page projections.
    :ivar auxiliary_source: Explicit legacy, DFlash, or noncanonical ownership.
    """

    plan: PackedAuxiliaryPlan
    destination: PackedDestinationCapability
    destination_registration: PackedDestinationRegistration
    control: PackedDecodeControlSender
    components: tuple[PackedComponentPages, ...]
    auxiliary_source: PackedPrefillAuxiliarySource

    def __post_init__(self) -> None:
        """Own and validate one immutable pre-launch plan."""

        canonical_components = _validate_prefill_launch_fields(
            plan=self.plan,
            destination=self.destination,
            destination_registration=self.destination_registration,
            control=self.control,
            components=self.components,
            auxiliary_source=self.auxiliary_source,
        )
        object.__setattr__(self, "components", canonical_components)

    def bind_producer_event(
        self,
        producer_event: torch.cuda.Event,
    ) -> "PackedPrefillSubmission":
        """Bind exact producer completion after model host submission.

        :param producer_event: Event recorded after every source-side copy.
        :returns: Complete immutable terminal transport submission.
        """

        return PackedPrefillSubmission(
            plan=self.plan,
            destination=self.destination,
            destination_registration=self.destination_registration,
            control=self.control,
            components=self.components,
            auxiliary_source=self.auxiliary_source,
            producer_event=producer_event,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PackedPrefillSubmission:
    """Complete source-local input bound to producer completion.

    :ivar plan: Decoder-authored auxiliary metadata plan.
    :ivar destination: Request-scoped registered destination capability.
    :ivar destination_registration: Decode cache geometry used to rebuild the
        canonical layout.
    :ivar control: Exact decoder control and native data route.
    :ivar components: Main-KV and SWA source/destination page projections.
    :ivar auxiliary_source: Explicit legacy, DFlash, or noncanonical ownership.
    :ivar producer_event: Event recorded after the exact source cache writes.
    """

    plan: PackedAuxiliaryPlan
    destination: PackedDestinationCapability
    destination_registration: PackedDestinationRegistration
    control: PackedDecodeControlSender
    components: tuple[PackedComponentPages, ...]
    auxiliary_source: PackedPrefillAuxiliarySource
    producer_event: torch.cuda.Event

    def __post_init__(self) -> None:
        """Own and validate one immutable source submission."""

        canonical_components = _validate_prefill_launch_fields(
            plan=self.plan,
            destination=self.destination,
            destination_registration=self.destination_registration,
            control=self.control,
            components=self.components,
            auxiliary_source=self.auxiliary_source,
        )
        object.__setattr__(self, "components", canonical_components)
        if not isinstance(self.producer_event, torch.cuda.Event):
            raise TypeError("packed source producer event must be a CUDA event")


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


class DFlashBoundaryPackedAuxiliarySlotAllocation:
    """Give one packed transaction exclusive ownership of a pool reservation."""

    def __init__(self, pool: DFlashBoundaryDeviceRowPool) -> None:
        """Initialize one unallocated terminal DFlash boundary row.

        :param pool: Process-lifetime registered destination row pool.
        """

        if type(pool) is not DFlashBoundaryDeviceRowPool:
            raise TypeError("pool must be DFlashBoundaryDeviceRowPool")
        self._pool = pool
        self._reservation: object | None = None
        self._owner: object | None = None
        self._released = False
        self._quarantined = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        """Return whether the exact row generation was returned.

        :returns: Exact release state.
        """

        with self._lock:
            return self._released

    @property
    def quarantined(self) -> bool:
        """Return whether the exact row was withheld from reuse.

        :returns: Exact quarantine state.
        """

        with self._lock:
            return self._quarantined

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Acquire one pool row under the packed allocation authority.

        :param owner: Exact process-local reservation authority.
        :returns: Opaque pool reservation.
        """

        with self._lock:
            if self._reservation is not None:
                raise RuntimeError("DFlash boundary row was already allocated")
            reservation = self._pool.allocate_packed_auxiliary_slot(owner)
            self._reservation = reservation
            self._owner = owner
            return reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Return the exact live pool generation and registered geometry.

        :param reservation: Exact pool reservation.
        :returns: Immutable live row snapshot.
        """

        with self._lock:
            self._require_reservation(reservation)
            return self._pool.packed_auxiliary_slot_reservation_snapshot(
                reservation
            )

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Return the exact completed row to process-lifetime ownership.

        :param reservation: Exact pool reservation.
        :param owner: Exact reservation authority.
        """

        with self._lock:
            self._require_owner(reservation, owner)
            if self._released or self._quarantined:
                raise RuntimeError("DFlash boundary row is already terminal")
            self._pool.release_packed_auxiliary_slot(reservation, owner)
            self._released = True

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Permanently withhold one ambiguous row from reuse.

        :param reservation: Exact pool reservation.
        :param owner: Exact reservation authority.
        """

        with self._lock:
            self._require_owner(reservation, owner)
            if self._released:
                raise RuntimeError("released DFlash boundary row cannot be quarantined")
            if self._quarantined:
                return
            self._pool.quarantine_packed_auxiliary_slot(reservation, owner)
            self._quarantined = True

    def _require_reservation(self, reservation: object) -> None:
        if self._reservation is None or reservation is not self._reservation:
            raise RuntimeError("DFlash boundary reservation belongs to another request")

    def _require_owner(self, reservation: object, owner: object) -> None:
        self._require_reservation(reservation)
        if owner is not self._owner:
            raise RuntimeError("DFlash boundary reservation owner is stale")


PackedDecodeAuxiliarySlotAllocation = (
    AdoptedPackedAuxiliarySlotAllocation
    | DFlashBoundaryPackedAuxiliarySlotAllocation
)


@dataclasses.dataclass(frozen=True)
class _PackedPrefillTransferStats:
    """Terminal prefill-writer transfer timings.

    :ivar room_id: Decoder-minted bootstrap room.
    :ivar source_rank: Source attention tensor-parallel rank.
    :ivar copy_group_count: Source component-entry groups carrying payload.
    :ivar payload_bytes: Logical KV bytes gathered from source tensors.
    :ivar transport_bytes: Bytes submitted through the physical transport.
    :ivar ready_wait_duration_ms: PREPARE-to-READY wall time.
    :ivar source_gather_copy_duration_ms: Synchronous source gather wall time.
    :ivar main_transport_duration_ms: Posted NIXL transfer-to-receipt wall time.
    """

    room_id: int
    source_rank: int
    copy_group_count: int
    payload_bytes: int
    transport_bytes: int
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
            f"copy_group_count={self.copy_group_count}, "
            f"payload_bytes={self.payload_bytes}, "
            f"transport_bytes={self.transport_bytes}, "
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
    auxiliary_outcome: PackedPrefillAuxiliaryOutcome | None = None
    outcomes_sent: bool = False
    main_handle_released: bool = False
    auxiliary_handle_released: bool = False
    ready_wait_duration_ms: float | None = None
    source_gather_copy_duration_ms: float | None = None
    main_transport_started_at: float | None = None
    main_transport_duration_ms: float | None = None
    terminal_identity: PackedTerminalSourceIdentityPlan | None = None
    terminal_prepare: PackedPrepare | None = None
    terminal_prepare_sent: bool = False
    terminal_prepare_started_at: float | None = None
    terminal_ready: PackedReady | None = None
    terminal_gather_started: bool = False
    terminal_gather_posted: bool = False
    terminal_teardown: PackedRequestTeardown | None = None
    terminal_ack_sent: bool = False
    terminal_quarantined: bool = False
    terminal_main_failure_settled: bool = False
    terminal_auxiliary_failure_settled: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class PackedPrefillOwnerInventory:
    """Exact packed source resources retained by terminal-owner requests.

    :ivar active_bindings: Every live terminal source lifecycle.
    :ivar quarantined_bindings: Lifecycles retaining ambiguous resources.
    :ivar waiting_for_ready_bindings: PREPARE publications without READY.
    :ivar main_handle_bindings: Lifecycles retaining a main NIXL handle.
    :ivar auxiliary_handle_bindings: Lifecycles retaining an auxiliary handle.
    :ivar lane_bindings: Lifecycles retaining an exact packed transfer lane.
    """

    active_bindings: tuple[bytes, ...]
    quarantined_bindings: tuple[bytes, ...]
    waiting_for_ready_bindings: tuple[bytes, ...]
    main_handle_bindings: tuple[bytes, ...]
    auxiliary_handle_bindings: tuple[bytes, ...]
    lane_bindings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        """Validate sorted exact-identity resource inventories."""

        collections = (
            self.active_bindings,
            self.quarantined_bindings,
            self.waiting_for_ready_bindings,
            self.main_handle_bindings,
            self.auxiliary_handle_bindings,
            self.lane_bindings,
        )
        if any(type(values) is not tuple for values in collections):
            raise TypeError("prefill owner inventory collections must be tuples")
        if any(
            type(value) is not bytes or len(value) != 32
            for values in collections
            for value in values
        ):
            raise ValueError("prefill owner inventory bindings must contain 32 bytes")
        if any(values != tuple(sorted(values)) for values in collections):
            raise ValueError("prefill owner inventory bindings must use digest order")
        active = set(self.active_bindings)
        if any(not set(values).issubset(active) for values in collections[1:]):
            raise ValueError("prefill owner resource bindings must remain active")


class PackedPrefillRuntime:
    """Persistent source actor for one-chunk packed request transfers."""

    def __init__(
        self,
        manager: PackedRuntimeManager,
        runtime_artifacts: PackedNixlRuntimeArtifactCohort,
        visibility_policy: PackedDestinationVisibilityPolicy,
        direct_terminal_owner: NixlTerminalOwnerBoundary | None = None,
    ) -> None:
        """Initialize the process-lifetime source actor.

        :param manager: Owning NIXL manager.
        :param runtime_artifacts: Exact native runtime cohort.
        :param visibility_policy: Same-host route visibility policy.
        :param direct_terminal_owner: Optional process-lifetime direct NIXL
            completion owner for terminal serving.
        """

        if manager.attn_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES:
            raise ValueError("packed source actor requires a supported source TP width")
        if manager.attn_cp_rank != 0 or manager.pp_rank != 0:
            raise ValueError("packed source actor requires CP1 and PP1")
        if direct_terminal_owner is not None and not isinstance(
            direct_terminal_owner,
            NixlTerminalOwnerBoundary,
        ):
            raise TypeError(
                "direct_terminal_owner must inherit NixlTerminalOwnerBoundary"
            )
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
        self._direct_terminal_owner = direct_terminal_owner
        self._lanes: dict[
            PackedDestinationRouteBinding,
            list[PackedTransferLane],
        ] = {}
        self._records: dict[PackedRequestKey, _PrefillRequestRecord] = {}
        self._failed_records: list[_PrefillRequestRecord] = []
        self._lock = threading.RLock()

    def bind_direct_terminal_owner(
        self,
        owner: NixlTerminalOwnerBoundary,
    ) -> None:
        """Attach the process-lifetime owner before any request or lane exists.

        :param owner: Registered direct NIXL terminal owner.
        """

        if not isinstance(owner, NixlTerminalOwnerBoundary):
            raise TypeError("owner must inherit NixlTerminalOwnerBoundary")
        with self._lock:
            if self._direct_terminal_owner is not None:
                raise RuntimeError("packed prefill terminal owner is already bound")
            if len(self._records) != 0 or len(self._lanes) != 0:
                raise RuntimeError(
                    "packed prefill terminal owner must precede all request state"
                )
            self._direct_terminal_owner = owner

    @property
    def writer_id(self) -> StagingWriterId:
        """Return this process's canonical source identity.

        :returns: Exact writer identity.
        """

        return self._writer_id

    def bind_terminal_owner(
        self,
        submission: PackedPrefillSubmission,
        identity: PackedTerminalSourceIdentityPlan,
    ) -> PackedPrepare:
        """Bind one source lifecycle and register PREPARE without waiting.

        The returned message is retained by the actor and is not sent until
        :meth:`publish_terminal_owner_prepare`. This lets process composition
        publish native lifecycle ownership before control traffic can return a
        READY message.

        :param submission: Immutable packed transport submission.
        :param identity: Decoder-authored rank-local terminal identity.
        :returns: Exact PREPARE registered by the source coordinator.
        """

        if type(submission) is not PackedPrefillSubmission:
            raise TypeError("submission must be PackedPrefillSubmission")
        if type(identity) is not PackedTerminalSourceIdentityPlan:
            raise TypeError("identity must be PackedTerminalSourceIdentityPlan")
        self._validate_terminal_identity(submission, identity)
        record = self._prepare_submission(submission)
        with self._lock:
            record.terminal_identity = identity
        try:
            prepare = self._register_prepare(record)
        except (RuntimeError, TypeError, ValueError):
            self._cancel_terminal_record_before_publish(record)
            raise
        with self._lock:
            current = self._records.get(identity.request_key)
            if current is not record:
                raise RuntimeError("packed source registry changed during owner bind")
            record.terminal_prepare = prepare
        return prepare

    def publish_terminal_owner_prepare(
        self,
        submission: PackedPrefillSubmission,
    ) -> None:
        """Send one actor-registered PREPARE without entering a wait path.

        :param submission: Exact submission previously bound to terminal owner.
        """

        record = self._terminal_record_for_submission(submission)
        with self._lock:
            prepare = record.terminal_prepare
            if prepare is None:
                raise RuntimeError("terminal PREPARE was not registered")
            if record.terminal_prepare_sent:
                raise RuntimeError("terminal PREPARE was already published")
            record.terminal_prepare_started_at = time.perf_counter()
            record.terminal_prepare_sent = True
        try:
            submission.control.send_message(prepare)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._quarantine_terminal_record(
                record,
                "terminal PREPARE publication became ambiguous",
            )
            raise

    def cancel_terminal_owner_unpublished(
        self,
        submission: PackedPrefillSubmission,
    ) -> None:
        """Cancel an actor binding before PREPARE or native work is published.

        :param submission: Exact unpublished terminal submission.
        """

        record = self._terminal_record_for_submission(submission)
        with self._lock:
            if (
                record.terminal_prepare_sent
                or record.source_transfer is not None
                or record.terminal_gather_started
                or record.main_handle is not None
                or record.auxiliary_handle is not None
            ):
                raise RuntimeError("published terminal source request cannot cancel")
        self._cancel_terminal_record_before_publish(record)

    def deliver_terminal_owner_ready(
        self,
        authenticated_decode_peer: PackedPeerIdentity,
        message: PackedReady,
    ) -> TerminalRequestBinding:
        """Deliver one authenticated READY without waking a blocking waiter.

        :param authenticated_decode_peer: Decoder proved by control routing.
        :param message: Exact READY for the actor's sole packed chunk.
        :returns: Rank-local lifecycle binding made ready for owner work.
        """

        if type(authenticated_decode_peer) is not PackedPeerIdentity:
            raise TypeError("authenticated_decode_peer must be PackedPeerIdentity")
        if type(message) is not PackedReady:
            raise TypeError("message must be PackedReady")
        record = self._terminal_record_for_key(
            PackedRequestKey.from_chunk_key(message.key)
        )
        identity = self._require_terminal_identity(record)
        self._validate_terminal_decode_peer(record, authenticated_decode_peer)
        with self._lock:
            if not record.terminal_prepare_sent:
                raise RuntimeError("terminal READY preceded PREPARE publication")
            prepare_started_at = record.terminal_prepare_started_at
            if prepare_started_at is None:
                raise RuntimeError("terminal READY lost PREPARE timing authority")
            if record.source_transfer is not None:
                if record.terminal_ready == message:
                    return identity.local_binding
                self._quarantine_terminal_record(
                    record,
                    "terminal READY conflicts with prior delivery",
                )
                raise RuntimeError("terminal READY conflicts with prior delivery")
        try:
            transfer = self._ready.handle_ready(message, authenticated_decode_peer)
        except (RuntimeError, TypeError, ValueError):
            self._quarantine_terminal_record(
                record,
                "terminal READY validation failed",
            )
            raise
        with record.condition:
            if record.source_transfer is not None:
                raise RuntimeError("terminal READY raced another delivery")
            record.ready_wait_duration_ms = (
                time.perf_counter() - prepare_started_at
            ) * 1000.0
            record.source_transfer = transfer
            record.terminal_ready = message
            record.condition.notify_all()
        return identity.local_binding

    def begin_terminal_owner_transfer(
        self,
        action: NativeTerminalOwnerAction,
        post_auxiliary: Callable[[PackedPrefillSubmission], object] | None,
    ) -> None:
        """Post gather and direct transfers only under owner authorization.

        The canonical writer must supply a device-resident auxiliary poster.
        This actor intentionally does not call the legacy DRAM auxiliary path.
        The callback must return the exact retained native handle and must not
        wait for terminality.

        :param action: Exact ``SOURCE_GATHER_READY`` owner action.
        :param post_auxiliary: Nonblocking DFlash auxiliary transfer poster on
            the canonical writer, otherwise ``None``.
        """

        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        canonical = self._writer_id == record.submission.plan.canonical_writer_id
        if canonical and not callable(post_auxiliary):
            raise TypeError("canonical terminal source requires auxiliary poster")
        if not canonical and post_auxiliary is not None:
            raise ValueError("noncanonical terminal source cannot post auxiliary data")
        with self._lock:
            if not record.terminal_prepare_sent:
                raise RuntimeError("terminal gather preceded PREPARE publication")
            transfer = record.source_transfer
            if transfer is None:
                raise RuntimeError("terminal gather preceded authenticated READY")
            if record.terminal_gather_started:
                raise RuntimeError("terminal gather was already started")
            record.terminal_gather_started = True
        try:
            if canonical:
                if post_auxiliary is None:
                    raise RuntimeError("terminal auxiliary poster disappeared")
                auxiliary_handle = post_auxiliary(record.submission)
                if auxiliary_handle is None:
                    raise RuntimeError("terminal auxiliary poster returned no handle")
                with record.condition:
                    record.auxiliary_handle = auxiliary_handle
            main_handle, main_lane = self._post_main_transfer(
                record,
                transfer,
                binding_digest=action.binding.digest,
            )
            with record.condition:
                if (
                    record.main_handle is not main_handle
                    or record.main_lane is not main_lane
                ):
                    raise RuntimeError("terminal main post lost actor ownership")
                record.terminal_gather_posted = True
        except Exception:
            logger.error(
                "Terminal source gather or transfer post failed:\n%s",
                traceback.format_exc(),
            )
            self._quarantine_terminal_record(
                record,
                "terminal source gather or transfer post failed",
            )
            raise

    def send_terminal_owner_outcomes(
        self,
        action: NativeTerminalOwnerAction,
        settle_main: Callable[
            [PackedTransferLane, NativeTerminalOwnerAction], PackedWriterOutcome
        ],
        settle_auxiliary: (
            Callable[[object, NativeTerminalOwnerAction], PackedPrefillAuxiliaryOutcome]
            | None
        ),
    ) -> tuple[PackedWriterOutcome, PackedPrefillAuxiliaryOutcome | None]:
        """Construct and send outcomes under exact native terminal authority.

        Settlement callbacks may consume take-once native completion receipts,
        but must retain the native handles and packed lane. Teardown authority,
        not transport completion, owns their release.

        :param action: Exact ``SOURCE_OUTCOME_READY`` owner action.
        :param settle_main: Main-lane terminal receipt settlement.
        :param settle_auxiliary: Canonical auxiliary receipt settlement.
        :returns: Main and optional canonical auxiliary outcomes sent on control.
        """

        if not callable(settle_main):
            raise TypeError("settle_main must be callable")
        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        )
        canonical = self._writer_id == record.submission.plan.canonical_writer_id
        if canonical and not callable(settle_auxiliary):
            raise TypeError("canonical terminal source requires auxiliary settlement")
        if not canonical and settle_auxiliary is not None:
            raise ValueError("noncanonical terminal source has no auxiliary settlement")
        with self._lock:
            if not record.terminal_gather_posted:
                raise RuntimeError("terminal outcomes preceded transfer post")
            if record.main_outcome is not None or record.outcomes_sent:
                raise RuntimeError("terminal outcomes were already constructed")
            lane = record.main_lane
            auxiliary_handle = record.auxiliary_handle
            if lane is None or record.main_handle is None:
                raise RuntimeError("terminal outcome lost main native ownership")
            if canonical and auxiliary_handle is None:
                raise RuntimeError("terminal outcome lost auxiliary ownership")
        try:
            main_outcome = settle_main(lane, action)
            self._validate_terminal_main_outcome(record, main_outcome)
            auxiliary_outcome: PackedPrefillAuxiliaryOutcome | None = None
            if canonical:
                if auxiliary_handle is None or settle_auxiliary is None:
                    raise RuntimeError("terminal auxiliary settlement disappeared")
                auxiliary_outcome = settle_auxiliary(auxiliary_handle, action)
                self._validate_terminal_auxiliary_outcome(record, auxiliary_outcome)
            with record.condition:
                record.main_outcome = main_outcome
                record.auxiliary_outcome = auxiliary_outcome
            record.submission.control.send_message(main_outcome)
            if auxiliary_outcome is not None:
                record.submission.control.send_message(auxiliary_outcome)
            with record.condition:
                record.outcomes_sent = True
            self._emit_transfer_stats(record)
            return main_outcome, auxiliary_outcome
        except Exception:
            logger.error(
                "Terminal source outcome settlement or publication failed:\n%s",
                traceback.format_exc(),
            )
            self._quarantine_terminal_record(
                record,
                "terminal source outcome settlement or publication failed",
            )
            raise

    def deliver_terminal_owner_teardown(
        self,
        authenticated_decode_peer: PackedPeerIdentity,
        request: PackedRequestTeardown,
    ) -> TerminalRequestBinding | None:
        """Retain authenticated teardown without releasing any resource.

        :param authenticated_decode_peer: Decoder proved by control routing.
        :param request: Exact writer-specific teardown request.
        :returns: Local binding for a new teardown, otherwise ``None`` for an
            exact duplicate which must not be resubmitted to native state.
        """

        if type(authenticated_decode_peer) is not PackedPeerIdentity:
            raise TypeError("authenticated_decode_peer must be PackedPeerIdentity")
        if type(request) is not PackedRequestTeardown:
            raise TypeError("request must be PackedRequestTeardown")
        record = self._terminal_record_for_key(request.key)
        identity = self._require_terminal_identity(record)
        self._validate_terminal_decode_peer(record, authenticated_decode_peer)
        try:
            self._validate_terminal_teardown(record, request)
            with self._lock:
                previous = record.terminal_teardown
                if previous is not None:
                    if previous == request:
                        return None
                    raise RuntimeError(
                        "terminal teardown conflicts with prior delivery"
                    )
                record.terminal_teardown = request
            return identity.local_binding
        except (RuntimeError, TypeError, ValueError):
            self._quarantine_terminal_record(
                record,
                "terminal source teardown validation failed",
            )
            raise

    def settle_terminal_owner_teardown(
        self,
        action: NativeTerminalOwnerAction,
        release_main: Callable[[PackedTransferLane, NativeTerminalOwnerAction], None],
        release_auxiliary: Callable[[object, NativeTerminalOwnerAction], None] | None,
    ) -> PackedRequestTeardownAck:
        """Release exact handles and ACK only under teardown owner authority.

        :param action: Exact ``SOURCE_ACK_READY`` owner action.
        :param release_main: Main lane release under exact ACK authority.
        :param release_auxiliary: Canonical auxiliary release under the same
            exact ACK authority.
        :returns: Exact acknowledgement sent to the decoder.
        """

        if not callable(release_main):
            raise TypeError("release_main must be callable")
        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        )
        canonical = self._writer_id == record.submission.plan.canonical_writer_id
        if canonical and not callable(release_auxiliary):
            raise TypeError("canonical terminal source requires auxiliary release")
        if not canonical and release_auxiliary is not None:
            raise ValueError("noncanonical terminal source has no auxiliary release")
        with self._lock:
            request = record.terminal_teardown
            if request is None:
                raise RuntimeError("terminal ACK preceded authenticated teardown")
            if record.terminal_ack_sent:
                raise RuntimeError("terminal ACK authority was replayed")
            lane = record.main_lane
            main_handle = record.main_handle
            auxiliary_handle = record.auxiliary_handle
            if lane is None or main_handle is None:
                raise RuntimeError("terminal teardown lost main native ownership")
            if canonical and auxiliary_handle is None:
                raise RuntimeError("terminal teardown lost auxiliary ownership")
        try:
            release_main(lane, action)
            with record.condition:
                record.main_handle_released = True
                record.main_handle = None
                record.main_lane = None
            if canonical:
                if auxiliary_handle is None or release_auxiliary is None:
                    raise RuntimeError("terminal auxiliary release disappeared")
                release_auxiliary(auxiliary_handle, action)
                with record.condition:
                    record.auxiliary_handle_released = True
                    record.auxiliary_handle = None
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
            with record.condition:
                record.terminal_ack_sent = True
            return acknowledgement
        except Exception:
            logger.error(
                "Terminal source teardown settlement or ACK failed:\n%s",
                traceback.format_exc(),
            )
            self._quarantine_terminal_record(
                record,
                "terminal source teardown settlement or ACK failed",
            )
            raise

    def quarantine_terminal_owner_request(
        self,
        action: NativeTerminalOwnerAction,
        reason: str,
        settle_main: Callable[[PackedTransferLane, NativeTerminalOwnerAction], None],
        settle_auxiliary: Callable[[object, NativeTerminalOwnerAction], None] | None,
    ) -> None:
        """Retain every ambiguous source resource under native quarantine.

        :param action: Exact quarantine or process-fatal owner action.
        :param reason: Stable fail-closed evidence.
        :param settle_main: Main-lane failure settlement callback.
        :param settle_auxiliary: Canonical auxiliary failure settlement callback.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in (
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        ):
            raise ValueError("source quarantine requires a failure owner action")
        if not callable(settle_main):
            raise TypeError("settle_main must be callable")
        record = self._terminal_record_for_action(action, action.kind)
        self._quarantine_terminal_record(record, reason)
        canonical = self._writer_id == record.submission.plan.canonical_writer_id
        with record.condition:
            lane = record.main_lane
            auxiliary_handle = record.auxiliary_handle
            main_settled = record.terminal_main_failure_settled
            auxiliary_settled = record.terminal_auxiliary_failure_settled
        if (
            canonical
            and auxiliary_handle is not None
            and not callable(settle_auxiliary)
        ):
            raise TypeError("canonical source failure requires auxiliary settlement")
        if not canonical and settle_auxiliary is not None:
            raise ValueError("noncanonical source has no auxiliary failure settlement")
        if lane is not None and not main_settled:
            settle_main(lane, action)
            with record.condition:
                record.terminal_main_failure_settled = True
        if auxiliary_handle is not None and not auxiliary_settled:
            if settle_auxiliary is None:
                raise RuntimeError("auxiliary failure settlement disappeared")
            settle_auxiliary(auxiliary_handle, action)
            with record.condition:
                record.terminal_auxiliary_failure_settled = True

    def retire_terminal_owner_request(
        self,
        action: NativeTerminalOwnerAction,
    ) -> PackedPrefillSubmission:
        """Remove actor identity only after native joined retirement.

        :param action: Exact ``REQUEST_RETIRED`` owner action.
        :returns: Transport submission whose actor resources are exhausted.
        """

        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        )
        with self._lock:
            if record.terminal_quarantined:
                raise RuntimeError("quarantined terminal source cannot retire")
            if not record.terminal_ack_sent:
                raise RuntimeError("terminal source retirement preceded ACK")
            if (
                record.main_handle is not None
                or record.auxiliary_handle is not None
                or record.main_lane is not None
            ):
                raise RuntimeError(
                    "terminal source retirement retains native resources"
                )
            key = record.submission.plan.key
            current = self._records.get(key)
            if current is not record:
                raise RuntimeError("packed source registry changed during retirement")
            del self._records[key]
        return record.submission

    def require_terminal_owner_retirement(
        self,
        action: NativeTerminalOwnerAction,
        expected_submission: PackedPrefillSubmission,
    ) -> None:
        """Validate actor retirement without mutating retained ownership.

        :param action: Exact joined ``REQUEST_RETIRED`` authority.
        :param expected_submission: Submission the manager intends to retire.
        """

        if type(expected_submission) is not PackedPrefillSubmission:
            raise TypeError("expected_submission must be PackedPrefillSubmission")
        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        )
        with self._lock:
            if record.submission is not expected_submission:
                raise RuntimeError("actor retirement targets another submission")
            if record.terminal_quarantined:
                raise RuntimeError("quarantined terminal source cannot retire")
            if not record.terminal_ack_sent:
                raise RuntimeError("terminal source retirement preceded ACK")
            if (
                record.main_handle is not None
                or record.auxiliary_handle is not None
                or record.main_lane is not None
            ):
                raise RuntimeError(
                    "terminal source retirement retains native resources"
                )
            if self._records.get(record.submission.plan.key) is not record:
                raise RuntimeError("packed source registry changed before retirement")

    def terminal_owner_inventory(self) -> PackedPrefillOwnerInventory:
        """Return exact actor-owned terminal source resource identities.

        :returns: Active, quarantine, READY, native-handle, and lane inventory.
        """

        with self._lock:
            records = tuple(
                record
                for record in self._records.values()
                if record.terminal_identity is not None
            )
            active = tuple(
                sorted(
                    self._require_terminal_identity(record).local_binding.digest
                    for record in records
                )
            )

            def selected(
                predicate: Callable[[_PrefillRequestRecord], bool],
            ) -> tuple[bytes, ...]:
                return tuple(
                    sorted(
                        self._require_terminal_identity(record).local_binding.digest
                        for record in records
                        if predicate(record)
                    )
                )

            return PackedPrefillOwnerInventory(
                active_bindings=active,
                quarantined_bindings=selected(
                    lambda record: record.terminal_quarantined
                ),
                waiting_for_ready_bindings=selected(
                    lambda record: (
                        record.terminal_prepare_sent and record.source_transfer is None
                    )
                ),
                main_handle_bindings=selected(
                    lambda record: record.main_handle is not None
                ),
                auxiliary_handle_bindings=selected(
                    lambda record: record.auxiliary_handle is not None
                ),
                lane_bindings=selected(lambda record: record.main_lane is not None),
            )

    def _validate_terminal_identity(
        self,
        submission: PackedPrefillSubmission,
        identity: PackedTerminalSourceIdentityPlan,
    ) -> None:
        """Validate one actor binding before it enters the sole registry.

        :param submission: Candidate packed source submission.
        :param identity: Candidate terminal identity graph.
        """

        binding = identity.local_binding
        owner = binding.owner
        if identity.request_key != submission.plan.key:
            raise RuntimeError("terminal identity belongs to another packed request")
        if owner.role is not TerminalOwnerRole.SOURCE:
            raise RuntimeError("terminal identity does not belong to source")
        if (
            owner.tp_rank != self._manager.attn_tp_rank
            or owner.tp_size != self._manager.attn_tp_size
            or owner.tp_rank != self._writer_id.source_attn_tp_rank
        ):
            raise RuntimeError("terminal identity belongs to another source rank")
        local_process_generation = uuid.UUID(self._manager.process_generation).bytes
        if owner.process_generation != local_process_generation:
            raise RuntimeError("terminal identity belongs to another source process")
        if (
            identity.request_ready_issuer.process_generation
            != submission.control.peer.agent_generation
            or submission.plan.destination_process_generation
            != submission.control.peer.agent_generation
        ):
            raise RuntimeError("terminal identity names another decoder process")

    def _terminal_record_for_submission(
        self,
        submission: PackedPrefillSubmission,
    ) -> _PrefillRequestRecord:
        """Resolve an exact terminal record without another request registry.

        :param submission: Exact actor-owned transport submission.
        :returns: Matching terminal request record.
        """

        if type(submission) is not PackedPrefillSubmission:
            raise TypeError("submission must be PackedPrefillSubmission")
        record = self._terminal_record_for_key(submission.plan.key)
        if record.submission is not submission:
            raise RuntimeError("terminal actor owns another submission instance")
        return record

    def _terminal_record_for_key(
        self,
        key: PackedRequestKey,
    ) -> _PrefillRequestRecord:
        """Resolve the sole actor record for one terminal request key.

        :param key: Exact packed request generation.
        :returns: Matching terminal request record.
        """

        if type(key) is not PackedRequestKey:
            raise TypeError("key must be PackedRequestKey")
        with self._lock:
            record = self._records.get(key)
        if record is None:
            raise RuntimeError("terminal source references an unknown request")
        self._require_terminal_identity(record)
        return record

    def _terminal_record_for_action(
        self,
        action: NativeTerminalOwnerAction,
        expected_kind: NativeTerminalOwnerActionKind,
    ) -> _PrefillRequestRecord:
        """Resolve an owner action through the actor's sole request registry.

        :param action: Exact native owner action.
        :param expected_kind: Action kind required by the actor operation.
        :returns: Matching terminal request record.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if type(expected_kind) is not NativeTerminalOwnerActionKind:
            raise TypeError("expected_kind must be NativeTerminalOwnerActionKind")
        if action.kind is not expected_kind:
            raise ValueError(f"terminal source operation requires {expected_kind.name}")
        binding = action.binding.to_binding()
        record = self._terminal_record_for_key(binding.request_key)
        identity = self._require_terminal_identity(record)
        if binding != identity.local_binding:
            raise RuntimeError("terminal owner action belongs to another binding")
        return record

    @staticmethod
    def _require_terminal_identity(
        record: _PrefillRequestRecord,
    ) -> PackedTerminalSourceIdentityPlan:
        """Return the exact terminal identity retained by one actor record.

        :param record: Candidate source actor record.
        :returns: Bound rank-local source identity.
        """

        identity = record.terminal_identity
        if identity is None:
            raise RuntimeError("packed source request has no terminal owner binding")
        return identity

    def _validate_terminal_decode_peer(
        self,
        record: _PrefillRequestRecord,
        authenticated_decode_peer: PackedPeerIdentity,
    ) -> None:
        """Join transport authentication with the frozen terminal identity.

        :param record: Exact terminal source record.
        :param authenticated_decode_peer: Peer proved by control routing.
        """

        identity = self._require_terminal_identity(record)
        if authenticated_decode_peer != record.submission.control.peer:
            raise RuntimeError("terminal control came from another decoder route")
        if (
            authenticated_decode_peer.agent_generation
            != identity.request_ready_issuer.process_generation
        ):
            raise RuntimeError("terminal control came from another decoder process")

    def _cancel_terminal_record_before_publish(
        self,
        record: _PrefillRequestRecord,
    ) -> None:
        """Remove one request whose external publication never began.

        :param record: Exact unpublished actor record.
        """

        self._ready.retire_pending(
            record.chunk_key,
            record.submission.control.peer,
        )
        with self._lock:
            key = record.submission.plan.key
            current = self._records.get(key)
            if current is not record:
                raise RuntimeError("packed source registry changed during cancellation")
            del self._records[key]

    def _validate_terminal_main_outcome(
        self,
        record: _PrefillRequestRecord,
        outcome: PackedWriterOutcome,
    ) -> None:
        """Validate main transport settlement before control publication.

        :param record: Exact terminal source record.
        :param outcome: Candidate main terminal outcome.
        """

        if type(outcome) is not PackedWriterOutcome:
            raise TypeError("main settlement must return PackedWriterOutcome")
        transfer = record.source_transfer
        if transfer is None:
            raise RuntimeError("main settlement lost authenticated READY ownership")
        if (
            outcome.key != record.chunk_key
            or outcome.writer_id != self._writer_id
            or outcome.digest != transfer.layout.digest
            or outcome.lease_id != transfer.lease_id
            or outcome.status is not PackedWriterOutcomeStatus.DONE
        ):
            raise RuntimeError("main settlement outcome differs from actor ownership")

    def _validate_terminal_auxiliary_outcome(
        self,
        record: _PrefillRequestRecord,
        outcome: PackedPrefillAuxiliaryOutcome,
    ) -> None:
        """Validate auxiliary settlement before control publication.

        :param record: Exact terminal source record.
        :param outcome: Candidate canonical auxiliary outcome.
        """

        if type(outcome) not in (
            PackedAuxiliaryOutcome,
            PackedDFlashBoundaryOutcome,
        ):
            raise TypeError("auxiliary settlement returned an unsupported outcome")
        if (
            outcome.plan != record.submission.plan
            or outcome.writer_id != self._writer_id
        ):
            raise RuntimeError(
                "auxiliary settlement outcome differs from actor ownership"
            )

    @staticmethod
    def _auxiliary_handle_generation(
        outcome: PackedPrefillAuxiliaryOutcome,
    ) -> int:
        """Return the native generation from either auxiliary schema.

        :param outcome: Exact canonical legacy or DFlash boundary outcome.
        :returns: Native transfer-handle generation named by teardown.
        """

        if type(outcome) is PackedAuxiliaryOutcome:
            return outcome.native_dram_handle_generation
        if type(outcome) is PackedDFlashBoundaryOutcome:
            return outcome.native_handle_generation
        raise TypeError("auxiliary outcome schema is unsupported")

    def _validate_terminal_teardown(
        self,
        record: _PrefillRequestRecord,
        request: PackedRequestTeardown,
    ) -> None:
        """Validate teardown against outcomes and frozen source identity.

        :param record: Exact terminal source record.
        :param request: Candidate authenticated teardown.
        """

        identity = self._require_terminal_identity(record)
        if not record.outcomes_sent:
            raise RuntimeError("terminal teardown preceded outcome publication")
        if request.key != record.submission.plan.key:
            raise RuntimeError("terminal teardown belongs to another request")
        if request.writer_id != self._writer_id:
            raise RuntimeError("terminal teardown targets another writer")
        if (
            request.request_slot_generation
            != record.submission.plan.request_slot_generation
            or request.writer_manifest_digest
            != identity.local_binding.rank_manifest_digest
            or request.allocation_digest != identity.local_binding.allocation_digest
        ):
            raise RuntimeError("terminal teardown differs from frozen allocation")
        auxiliary_outcome = record.auxiliary_outcome
        if self._writer_id == record.submission.plan.canonical_writer_id:
            if auxiliary_outcome is None:
                raise RuntimeError("canonical terminal teardown lacks aux outcome")
            if (
                request.auxiliary_handle_generation
                != self._auxiliary_handle_generation(auxiliary_outcome)
            ):
                raise RuntimeError("terminal teardown names another aux handle")
        elif request.auxiliary_handle_generation is not None:
            raise RuntimeError("noncanonical terminal teardown names an aux handle")

    def _quarantine_terminal_record(
        self,
        record: _PrefillRequestRecord,
        reason: str,
    ) -> None:
        """Mark one terminal record ambiguous without releasing any resource.

        :param record: Exact actor-owned terminal record.
        :param reason: Stable fail-closed evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        with record.condition:
            newly_quarantined = not record.terminal_quarantined
            record.terminal_quarantined = True
            if record.failure_reason is None:
                record.failure_reason = reason
            record.condition.notify_all()
        if newly_quarantined:
            room = record.chunk_key.room_id
            self._manager.record_failure(room, reason)
            self._manager.update_status(room, KVPoll.Failed)

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

        with self._lock:
            existing = self._records.get(submission.plan.key)
        if existing is not None and existing.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner source request cannot use blocking execution"
            )
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
        if record.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner source control requires explicit actor delivery"
            )
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
        if record.terminal_identity is not None:
            raise RuntimeError("terminal-owner READY cannot use a blocking wait")
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
        *,
        binding_digest: bytes | None = None,
    ) -> tuple[object, PackedTransferLane]:
        if self._direct_terminal_owner is None:
            if binding_digest is not None:
                raise ValueError(
                    "non-terminal packed runtime cannot bind a lifecycle digest"
                )
        elif type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError(
                "terminal packed runtime requires a 32-byte binding digest"
            )
        if binding_digest is not None:
            identity = self._require_terminal_identity(record)
            if binding_digest != identity.local_binding.digest:
                raise RuntimeError(
                    "packed source terminal binding differs from actor authority"
                )
        lane = self._acquire_lane(transfer)
        executor = self._source_copy_executor()
        gather_started_at = time.perf_counter()
        if binding_digest is None:
            executor.gather(
                transfer=transfer,
                source_lane=lane,
                producer_event=record.submission.producer_event,
            )
        else:
            executor.gather(
                transfer=transfer,
                source_lane=lane,
                producer_event=record.submission.producer_event,
                binding_digest=binding_digest,
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
        record.main_transport_started_at = time.perf_counter()
        if binding_digest is None:
            post = self._manager._post_transfer_when_ready
        else:
            post = self._manager._post_terminal_transfer_once
        lane.post_submission(
            lambda exact_handle: post(exact_handle, "packed main NIXL transfer")
        )
        return handle, lane

    def settle_terminal_main_transfer(
        self,
        action: NativeTerminalOwnerAction,
    ) -> PackedWriterOutcome:
        """Settle one main transfer under its matching native owner action.

        :param action: Exact ``SOURCE_OUTCOME_READY`` action.
        :returns: Validated packed writer outcome ready for authenticated send.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
            raise ValueError("terminal main completion requires SOURCE_OUTCOME_READY")
        record = self._terminal_record_for_action(
            action,
            NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        )
        with record.condition:
            lane = record.main_lane
            if lane is None:
                raise RuntimeError("terminal main completion preceded transfer post")
            if record.main_outcome is not None:
                raise RuntimeError("terminal main completion was already settled")
        outcome = lane.settle_terminal_completion(action)
        with record.condition:
            started_at = record.main_transport_started_at
            if started_at is not None:
                record.main_transport_duration_ms = (
                    time.perf_counter() - started_at
                ) * 1000.0
        return outcome

    def settle_terminal_main_failure(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Settle one failed transfer while retaining its poisoned lane.

        :param action: Matching quarantine or process-fatal owner action.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in (
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        ):
            raise ValueError("terminal main failure requires a failure owner action")
        record = self._terminal_record_for_action(action, action.kind)
        if record.main_lane is None:
            raise RuntimeError("terminal main failure references an unknown transfer")
        record.main_lane.settle_terminal_failure(action)

    def cancel_terminal_main_transfer(self, binding: TerminalRequestBinding) -> None:
        """Request cancellation without releasing ambiguous transfer authority.

        :param binding: Exact active terminal lifecycle identity.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        record = self._terminal_record_for_key(binding.request_key)
        identity = self._require_terminal_identity(record)
        if binding != identity.local_binding or record.main_lane is None:
            raise RuntimeError("terminal cancellation references an unknown transfer")
        record.main_lane.cancel_terminal_submission()

    def _emit_transfer_stats(self, record: _PrefillRequestRecord) -> None:
        ready_wait_duration_ms = record.ready_wait_duration_ms
        source_gather_copy_duration_ms = record.source_gather_copy_duration_ms
        main_transport_duration_ms = record.main_transport_duration_ms
        source_transfer = record.source_transfer
        if (
            ready_wait_duration_ms is None
            or source_gather_copy_duration_ms is None
            or main_transport_duration_ms is None
            or source_transfer is None
        ):
            logger.error(
                "PackedTransferStatsUnavailable(room=%d, role=prefill, source_rank=%d)",
                record.chunk_key.room_id,
                record.writer_id.source_attn_tp_rank,
            )
            return
        writer_layout = writer_layout_for(
            source_transfer.layout,
            record.writer_id,
        )
        logger.info(
            "%s",
            _PackedPrefillTransferStats(
                room_id=record.chunk_key.room_id,
                source_rank=record.writer_id.source_attn_tp_rank,
                copy_group_count=len(writer_layout.copy_groups),
                payload_bytes=sum(
                    group.length_bytes for group in writer_layout.copy_groups
                ),
                transport_bytes=source_transfer.length_bytes,
                ready_wait_duration_ms=ready_wait_duration_ms,
                source_gather_copy_duration_ms=source_gather_copy_duration_ms,
                main_transport_duration_ms=main_transport_duration_ms,
            ),
        )

    def _wait_for_main_outcome(
        self,
        record: _PrefillRequestRecord,
    ) -> PackedWriterOutcome:
        if record.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner main completion requires an owner action"
            )
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
        if record.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner auxiliary transport requires device-resident storage"
            )
        plan = record.submission.plan
        auxiliary_source = record.submission.auxiliary_source
        if type(auxiliary_source) is not PackedLegacyAuxiliarySource:
            raise RuntimeError("legacy auxiliary transport requires a DRAM row source")
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
                    source_ptr + source_length * auxiliary_source.row_index,
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
        if record.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner auxiliary completion requires an owner action"
            )
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
        auxiliary_source = record.submission.auxiliary_source
        if type(auxiliary_source) is not PackedLegacyAuxiliarySource:
            raise RuntimeError("legacy auxiliary receipt requires a DRAM row source")
        source_index = auxiliary_source.row_index
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
        if record.terminal_identity is not None:
            raise RuntimeError(
                "terminal-owner teardown requires explicit owner settlement"
            )
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
                != self._auxiliary_handle_generation(auxiliary_outcome)
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
                direct_terminal_owner=self._direct_terminal_owner,
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
    auxiliary_allocation: PackedDecodeAuxiliarySlotAllocation
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
    terminal_owner_bound: bool = False
    writer_aggregation_reported: bool = False
    writer_manifest_reported: bool = False
    scatter_started_by_owner: bool = False
    scatter_callback_attached: bool = False
    scatter_terminal_reported: bool = False
    ack_aggregation_reported: bool = False
    ack_manifest_reported: bool = False
    adoption_consumed_by_owner: bool = False
    metadata_consumed_by_owner: bool = False
    terminal_source_plan: PackedTerminalSourcePlan | None = None
    terminal_binding: TerminalRequestBinding | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDecodeTerminalRegistration:
    """Native-owner registration inputs derived from decoder-authored identity.

    :ivar binding: Exact local decode request generation.
    :ivar trusted_issuers: Canonical source writers followed by the destination
        request coordinator when it is not already present.
    """

    binding: TerminalRequestBinding
    trusted_issuers: tuple[TerminalProcessIdentity, ...]

    def __post_init__(self) -> None:
        """Validate one complete decode registration boundary."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if self.binding.owner.role is not TerminalOwnerRole.DECODE:
            raise ValueError("decode registration requires a decode binding")
        if type(self.trusted_issuers) is not tuple or len(self.trusted_issuers) == 0:
            raise ValueError("trusted_issuers must be a non-empty tuple")
        if any(
            type(issuer) is not TerminalProcessIdentity
            for issuer in self.trusted_issuers
        ):
            raise TypeError("trusted_issuers must contain TerminalProcessIdentity")
        digests = tuple(issuer.digest for issuer in self.trusted_issuers)
        if len(set(digests)) != len(digests):
            raise ValueError("trusted issuer identities must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDecodeOwnerSignal:
    """One authenticated control-ingress transition for the native owner.

    :ivar binding_digest: Exact local decode lifecycle identity.
    :ivar kind: Closed control-ingress native event.
    :ivar issuer: Source process authenticated by the packed control route.
    """

    binding_digest: bytes
    kind: NativeTerminalOwnerEventKind
    issuer: TerminalProcessIdentity

    def __post_init__(self) -> None:
        """Validate one route-authenticated owner transition."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        allowed = (
            NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
            NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
            NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
            NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
        )
        if self.kind not in allowed:
            raise ValueError("decode owner signal is not control ingress")
        if type(self.issuer) is not TerminalProcessIdentity:
            raise TypeError("issuer must be TerminalProcessIdentity")
        if self.issuer.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("decode control ingress requires a source issuer")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDecodeScatterBatch:
    """One request-level scatter batch submitted to the dedicated stream.

    :ivar binding_digest: Exact native lifecycle receiving callback terminality.
    :ivar chunk_keys: Exact chunk generations covered by the tail callback.
    :ivar stream_handle: Raw CUDA stream carrying every submitted scatter.
    """

    binding_digest: bytes
    chunk_keys: tuple[PackedChunkKey, ...]
    stream_handle: int

    def __post_init__(self) -> None:
        """Validate one exact callback attachment boundary."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if type(self.chunk_keys) is not tuple or len(self.chunk_keys) == 0:
            raise ValueError("chunk_keys must be a non-empty tuple")
        if any(type(key) is not PackedChunkKey for key in self.chunk_keys):
            raise TypeError("chunk_keys must contain PackedChunkKey values")
        if type(self.stream_handle) is not int or self.stream_handle < 0:
            raise ValueError("stream_handle must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDecodeOwnerInventory:
    """Exact actor state retained by terminal-owner request generations.

    :ivar active_bindings: All actor-owned terminal binding digests.
    :ivar quarantined_bindings: Bindings retaining ambiguous resources.
    :ivar in_flight_scatter_count: Submitted scatters awaiting one tail callback.
    :ivar pending_adoption_count: Commit receipts not consumed by the scheduler.
    """

    active_bindings: tuple[bytes, ...]
    quarantined_bindings: tuple[bytes, ...]
    in_flight_scatter_count: int
    pending_adoption_count: int


class PackedDecodeRuntime:
    """Persistent decode actor for authenticated packed request transactions."""

    def __init__(
        self,
        manager: PackedRuntimeManager,
        arena: PackedStagingArena,
        runtime_artifacts: PackedNixlRuntimeArtifactCohort,
        visibility_policy: PackedDestinationVisibilityPolicy,
        dflash_boundary_pool: DFlashBoundaryDeviceRowPool | None = None,
    ) -> None:
        """Initialize the process-lifetime decode actor.

        :param manager: Owning NIXL manager.
        :param arena: Persistent adopted staging arena.
        :param runtime_artifacts: Exact native runtime cohort.
        :param visibility_policy: Same-host route visibility policy.
        :param dflash_boundary_pool: Optional registered terminal DFlash rows.
        """

        if manager.attn_tp_size not in (1, 2):
            raise ValueError("packed decode actor supports only TP1 and TP2")
        if manager.attn_tp_rank < 0 or manager.attn_tp_rank >= manager.attn_tp_size:
            raise ValueError("packed decode actor has an invalid attention TP rank")
        self._manager = manager
        self._arena = arena
        self._runtime_artifacts = runtime_artifacts
        self._visibility_policy = visibility_policy
        if (
            dflash_boundary_pool is not None
            and type(dflash_boundary_pool) is not DFlashBoundaryDeviceRowPool
        ):
            raise TypeError(
                "dflash_boundary_pool must be DFlashBoundaryDeviceRowPool"
            )
        self._dflash_boundary_pool = dflash_boundary_pool
        self._outcomes = PackedDestinationOutcomeCoordinator(
            arena.protocol,
            _UnexpectedVisibilityActionExecutor(),
        )
        self._auxiliary_authority: PackedAuxiliaryAllocationLeaseAuthority | None = None
        self._metadata_allocator: PackedMetadataIndexAllocator | None = None
        self._consumer_authority: object | None = None
        self._records: dict[PackedRequestKey, _DecodeRequestRecord] = {}
        self._records_by_room: dict[int, PackedRequestKey] = {}
        self._records_by_terminal_binding: dict[bytes, PackedRequestKey] = {}
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
    def dflash_boundary_pool(self) -> DFlashBoundaryDeviceRowPool | None:
        """Return the registered terminal DFlash destination rows.

        :returns: Exact process-lifetime row pool, or ``None`` for legacy mode.
        """

        return self._dflash_boundary_pool

    @property
    def ready(self) -> bool:
        """Return whether scheduler metadata ownership is attached.

        :returns: Runtime admission readiness.
        """

        with self._lock:
            return self._auxiliary_authority is not None

    def attach_scheduler(
        self,
        metadata_allocator: PackedMetadataIndexAllocator | None,
        consumer_authority: object,
    ) -> None:
        """Bind exact scheduler metadata ownership once.

        :param metadata_allocator: Legacy row allocator, absent for terminal DFlash.
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
        metadata_buffer_index: int | None,
        allocation_lease: DecodeAllocationLease,
        allocation_authority: DecodeAllocationLeaseAuthority,
        lifecycle_authority: object,
        source_tp_size: int,
    ) -> PackedDecodeRequestTransaction:
        """Construct and register one decoder-authored one-chunk request.

        :param room_id: Decoder-minted room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Legacy pre-reserved metadata row. Terminal
            DFlash requests must pass ``None`` and acquire registered VRAM here.
        :param allocation_lease: Exact pinned decode allocation.
        :param allocation_authority: Exact allocation authority.
        :param lifecycle_authority: Trusted transport lifecycle authority.
        :param source_tp_size: Supported packed writer width.
        :returns: Complete prepared transaction.
        """

        with self._lock:
            auxiliary_authority = self._auxiliary_authority
            metadata_allocator = self._metadata_allocator
            if auxiliary_authority is None:
                raise RuntimeError("packed decode scheduler ownership is unavailable")
            if self._dflash_boundary_pool is None and metadata_allocator is None:
                raise RuntimeError("packed metadata allocator is unavailable")
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
        dflash_boundary_pool = self._dflash_boundary_pool
        if dflash_boundary_pool is None:
            if metadata_allocator is None:
                raise RuntimeError("packed metadata allocator is unavailable")
            if type(metadata_buffer_index) is not int:
                raise TypeError("legacy metadata_buffer_index must be an integer")
            auxiliary_allocation: PackedDecodeAuxiliarySlotAllocation = (
                AdoptedPackedAuxiliarySlotAllocation(
                    metadata_allocator,
                    metadata_buffer_index,
                    self._manager.kv_args,
                )
            )
        else:
            if metadata_buffer_index is not None:
                raise ValueError(
                    "terminal DFlash transaction cannot adopt a legacy metadata row"
                )
            auxiliary_allocation = DFlashBoundaryPackedAuxiliarySlotAllocation(
                dflash_boundary_pool
            )
        auxiliary_lease = auxiliary_authority.acquire(auxiliary_allocation)
        try:
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
        except (RuntimeError, TypeError, ValueError):
            auxiliary_authority.cancel_unpublished(auxiliary_lease)
            auxiliary_authority.retire_terminal(auxiliary_lease)
            raise
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

    def bind_terminal_owner(
        self,
        transaction: PackedDecodeRequestTransaction,
        binding: TerminalRequestBinding,
        source_plan: PackedTerminalSourcePlan,
    ) -> PackedDecodeTerminalRegistration:
        """Give terminal progress exclusive authority over actor continuations.

        This binding must precede allocation publication. Once bound, scatter,
        teardown, adoption, and metadata release may advance only through the
        explicit owner methods below; scheduler polling is rejected.

        :param transaction: Exact prepared request transaction.
        :param binding: Exact local decode request-generation identity.
        :param source_plan: Decoder-authored cross-rank terminal identity plan.
        :returns: Complete native lifecycle registration inputs.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(source_plan) is not PackedTerminalSourcePlan:
            raise TypeError("source_plan must be PackedTerminalSourcePlan")
        record = self._record_for_transaction(transaction)
        with self._lock:
            if record.terminal_owner_bound:
                raise RuntimeError("packed terminal owner was already bound")
            if transaction.state is not PackedRequestTransactionState.PREPARED:
                raise RuntimeError(
                    "packed terminal owner must bind before allocation publication"
                )
            if source_plan.request_key != transaction.snapshot().key:
                raise RuntimeError(
                    "terminal source plan belongs to another request generation"
                )
            if (
                binding.request_key != source_plan.request_key
                or binding.owner.role is not TerminalOwnerRole.DECODE
                or binding.owner.tp_size != source_plan.request_ready_issuer.tp_size
                or binding.rank_manifest_digest != source_plan.rank_manifest_digest
                or binding.allocation_digest != source_plan.allocation_digest
            ):
                raise RuntimeError(
                    "decode terminal binding differs from the source identity plan"
                )
            if binding.digest in self._records_by_terminal_binding:
                raise RuntimeError("decode terminal binding is already registered")
            transaction.bind_terminal_owner_authority(
                encode_packed_terminal_source_plan(source_plan),
                binding.digest,
            )
            record.terminal_owner_bound = True
            record.terminal_source_plan = source_plan
            record.terminal_binding = binding
            self._records_by_terminal_binding[binding.digest] = (
                transaction.snapshot().key
            )
            issuers = tuple(writer.process_identity for writer in source_plan.writers)
            coordinator = source_plan.request_ready_issuer
            if coordinator.digest not in tuple(issuer.digest for issuer in issuers):
                issuers = (*issuers, coordinator)
            return PackedDecodeTerminalRegistration(
                binding=binding,
                trusted_issuers=issuers,
            )

    def terminal_owner_transaction(
        self,
        binding_digest: bytes,
    ) -> PackedDecodeRequestTransaction:
        """Resolve one native action without a second request registry.

        :param binding_digest: Exact local decode binding digest.
        :returns: Actor-owned mutable request transaction.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        with self._lock:
            key = self._records_by_terminal_binding.get(binding_digest)
            if key is None:
                raise RuntimeError(
                    "decode terminal action references an unknown binding"
                )
            record = self._records.get(key)
            if (
                record is None
                or record.terminal_binding is None
                or record.terminal_binding.digest != binding_digest
            ):
                raise RuntimeError("decode terminal binding index is inconsistent")
            return record.transaction

    def terminal_owner_binding(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> TerminalRequestBinding:
        """Return the sole native lifecycle identity for one actor transaction.

        :param transaction: Exact registered packed transaction.
        :returns: Bound local decode lifecycle identity.
        """

        record = self._record_for_transaction(transaction)
        with self._lock:
            return self._require_terminal_binding_locked(record)

    def terminal_owner_request_ready_issuer(
        self,
        binding_digest: bytes,
    ) -> TerminalProcessIdentity:
        """Return the coordinator trusted by one live decode lifecycle.

        :param binding_digest: Exact local decode binding digest.
        :returns: Destination coordinator from the frozen source plan.
        """

        transaction = self.terminal_owner_transaction(binding_digest)
        record = self._record_for_transaction(transaction)
        with self._lock:
            plan = record.terminal_source_plan
            if plan is None:
                raise RuntimeError("terminal source identity plan is unavailable")
            return plan.request_ready_issuer

    def bind_publication(
        self,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        routes: tuple[PackedControlSender, ...],
    ) -> NativeTerminalOwnerEventKind | None:
        """Bind authenticated writer routes after receiver activation.

        :param transaction: Exact published transaction.
        :param publication: Matching irreversible publication.
        :param routes: Complete writer control routes.
        :returns: Local allocation-publication transition for an owner-bound request.
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
            source_plan = record.terminal_source_plan
            if record.terminal_owner_bound:
                if source_plan is None:
                    raise RuntimeError(
                        "terminal-owner publication lacks a source identity plan"
                    )
                if (
                    source_plan.rank_manifest_digest
                    != publication.writer_manifest_digest
                    or source_plan.allocation_digest != publication.allocation_digest
                    or tuple(writer.writer_id for writer in source_plan.writers)
                    != expected_writers
                ):
                    raise RuntimeError(
                        "terminal source plan differs from decode publication"
                    )
            record.routes = route_map
            if not record.terminal_owner_bound:
                return None
            return NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED

    def handle_control(
        self,
        authenticated_writer_id: StagingWriterId,
        message: PackedWireMessage,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Dispatch one peer-authenticated source control message.

        :param authenticated_writer_id: Writer derived from native peer state.
        :param message: Validated packed payload.
        :returns: Ordered control-ingress transitions earned by this message.
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
                return self._take_writer_aggregation_signal(
                    record,
                    authenticated_writer_id,
                )
            if type(message) is PackedWriterOutcome:
                record.transaction.handle_writer_outcome(
                    message,
                    authenticated_writer_id,
                )
                return self._take_writer_manifest_signal(
                    record,
                    authenticated_writer_id,
                )
            if type(message) is PackedAuxiliaryOutcome:
                record.transaction.handle_auxiliary_outcome(
                    message,
                    authenticated_writer_id,
                )
                return self._take_writer_manifest_signal(
                    record,
                    authenticated_writer_id,
                )
            if type(message) is PackedDFlashBoundaryOutcome:
                if self._dflash_boundary_pool is None:
                    raise RuntimeError(
                        "DFlash boundary outcome reached a legacy auxiliary runtime"
                    )
                record.transaction.handle_dflash_boundary_outcome(
                    message,
                    authenticated_writer_id,
                )
                return self._take_writer_manifest_signal(
                    record,
                    authenticated_writer_id,
                )
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
                return self._take_ack_signals(record, authenticated_writer_id)
            raise RuntimeError(
                f"decode received unsupported packed message {type(message).__name__}"
            )
        except Exception:
            reason = "packed decode control dispatch failed"
            logger.error("%s:\n%s", reason, traceback.format_exc())
            self._fail_record(record, reason)
            raise

    def begin_terminal_owner_scatter(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> PackedDecodeScatterBatch:
        """Launch the complete request scatter after owner authorization.

        Every chunk is submitted to one dedicated stream. The caller attaches
        one native tail callback to ``stream_handle``; that callback therefore
        proves terminality for the complete immutable ``chunk_keys`` manifest.

        :param transaction: Exact terminal-owner-bound transaction.
        :returns: Tail-callback identity for the submitted scatter batch.
        """

        record = self._record_for_transaction(transaction)
        try:
            with self._lock:
                self._require_terminal_owner_record_locked(record)
                if not record.writer_manifest_reported:
                    raise RuntimeError("decode scatter lacks writer-manifest authority")
                if record.scatter_started_by_owner or len(record.scatters) > 0:
                    raise RuntimeError("decode scatter was already submitted")
                if (
                    transaction.state
                    is not PackedRequestTransactionState.WRITERS_COMPLETED
                ):
                    raise RuntimeError(
                        "decode scatter requires complete writer aggregation"
                    )
                if record.upstream_wait_duration_ms is None:
                    record.upstream_wait_duration_ms = (
                        time.perf_counter() - record.pipeline_started_at
                    ) * 1000.0
                for key in record.chunk_keys:
                    scatter = transaction.begin_scatter(key)
                    submission = self._arena.copy_executor.scatter(
                        scatter.work,
                        scatter.proofs,
                    )
                    record.scatters[key] = (scatter, submission)
                record.scatter_started_by_owner = True
                return PackedDecodeScatterBatch(
                    binding_digest=self._require_terminal_binding_locked(record).digest,
                    chunk_keys=record.chunk_keys,
                    stream_handle=self._arena.copy_executor.scatter_stream_handle,
                )
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner scatter submission failed")
            raise

    def confirm_terminal_owner_scatter_callback(
        self,
        transaction: PackedDecodeRequestTransaction,
        batch: PackedDecodeScatterBatch,
    ) -> None:
        """Record successful direct callback attachment to the scatter tail.

        The native lifecycle receives ``DECODE_SCATTER_STARTED`` before the
        callback is attached. That enqueue order guarantees the direct native
        terminal producer cannot overtake the start transition.

        :param transaction: Exact terminal-owner-bound transaction.
        :param batch: Exact scatter batch whose callback registration returned.
        """

        if type(batch) is not PackedDecodeScatterBatch:
            raise TypeError("batch must be PackedDecodeScatterBatch")
        record = self._record_for_transaction(transaction)
        try:
            with self._lock:
                binding = self._require_terminal_binding_locked(record)
                if not record.scatter_started_by_owner:
                    raise RuntimeError("scatter callback preceded scatter submission")
                if record.scatter_callback_attached:
                    raise RuntimeError("scatter callback attachment was replayed")
                if (
                    batch.binding_digest != binding.digest
                    or batch.chunk_keys != record.chunk_keys
                    or batch.stream_handle
                    != self._arena.copy_executor.scatter_stream_handle
                ):
                    raise RuntimeError(
                        "scatter callback attachment differs from its batch"
                    )
                record.scatter_callback_attached = True
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner callback attachment failed")
            raise

    def complete_terminal_owner_scatter(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Consume native tail-callback authority without querying CUDA.

        :param transaction: Exact terminal-owner-bound transaction.
        """

        record = self._record_for_transaction(transaction)
        try:
            with self._lock:
                self._require_terminal_owner_record_locked(record)
                if not record.scatter_started_by_owner:
                    raise RuntimeError("decode scatter callback arrived before launch")
                if not record.scatter_callback_attached:
                    raise RuntimeError(
                        "decode scatter terminality lacks callback attachment"
                    )
                if record.scatter_terminal_reported:
                    raise RuntimeError("decode scatter callback was replayed")
                if tuple(record.scatters) != record.chunk_keys:
                    raise RuntimeError(
                        "decode scatter callback lacks complete live ownership"
                    )
                completed_at = time.perf_counter()
                for key in record.chunk_keys:
                    scatter, submission = record.scatters[key]
                    record.destination_scatter_copy_duration_ms += (
                        submission.elapsed_milliseconds()
                    )
                    transaction.complete_scatter(scatter)
                record.scatters.clear()
                if (
                    transaction.state
                    is not PackedRequestTransactionState.SCATTER_COMPLETED
                ):
                    raise RuntimeError(
                        "tail callback did not complete the request scatter"
                    )
                record.finalize_started_at = completed_at
                record.scatter_terminal_reported = True
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner scatter completion failed")
            raise

    def begin_terminal_owner_teardown(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Send writer-specific teardown after owner scatter terminality.

        :param transaction: Exact terminal-owner-bound transaction.
        """

        record = self._record_for_transaction(transaction)
        try:
            with self._lock:
                self._require_terminal_owner_record_locked(record)
                if not record.scatter_terminal_reported:
                    raise RuntimeError(
                        "decode teardown lacks scatter-terminal authority"
                    )
                self._begin_teardown_locked(record)
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner teardown dispatch failed")
            raise

    def consume_terminal_owner_adoption(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        """Adopt the exact allocation on the scheduler thread.

        :param transaction: Exact terminal-owner-bound transaction.
        :returns: Retained request owner associated with the allocation.
        """

        record = self._record_for_transaction(transaction)
        try:
            with self._lock:
                self._require_terminal_owner_record_locked(record)
                if not record.ack_manifest_reported:
                    raise RuntimeError("decode adoption lacks ACK-manifest authority")
                if record.adoption_consumed_by_owner:
                    raise RuntimeError("decode adoption authority was replayed")
                receipt = record.commit_receipt
                if receipt is None:
                    raise RuntimeError("decode adoption lacks a commit receipt")
                owner = transaction.commit_on_scheduler_thread(receipt)
                if owner is not transaction.request_owner:
                    raise RuntimeError(
                        "packed scheduler commit returned another request"
                    )
                record.commit_receipt = None
                record.adoption_consumed_by_owner = True
                self._emit_transfer_stats(record, time.perf_counter())
                return owner
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner allocation adoption failed")
            raise

    def complete_terminal_owner_metadata_consumption(
        self,
        transaction: PackedDecodeRequestTransaction,
        dflash_adoption: PackedDFlashBoundaryDecodeAdoption | None = None,
    ) -> None:
        """Release copied metadata while retaining lifecycle actor identity.

        The actor record remains live until native request retirement. This is
        what makes teardown inventory authoritative after scheduler adoption.

        :param transaction: Exact adopted terminal-owner transaction.
        :param dflash_adoption: Exact row-copy authority after its CUDA event
            reached terminal success.
        """

        record = self._record_for_transaction(transaction)
        consumer = self._consumer_authority
        if consumer is None:
            raise RuntimeError("packed metadata consumer authority is unavailable")
        try:
            with self._lock:
                self._require_terminal_owner_record_locked(record)
                if not record.adoption_consumed_by_owner:
                    raise RuntimeError("metadata consumption lacks allocation adoption")
                if record.metadata_consumed_by_owner:
                    raise RuntimeError("metadata consumption authority was replayed")
                transaction.complete_auxiliary_consumption_on_scheduler_thread(
                    consumer,
                    dflash_adoption=dflash_adoption,
                )
                if not record.auxiliary_allocation.released:
                    raise RuntimeError(
                        "packed metadata adapter did not release its row"
                    )
                record.metadata_consumed_by_owner = True
        except (RuntimeError, TypeError, ValueError):
            self._fail_record(record, "terminal-owner metadata consumption failed")
            raise

    def retire_terminal_owner_request(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> None:
        """Retire actor identity after native request-global completion.

        :param transaction: Exact committed terminal-owner transaction.
        """

        record = self._record_for_transaction(transaction)
        with self._lock:
            self._require_terminal_owner_record_locked(record)
            state = transaction.state
            if state is PackedRequestTransactionState.COMMITTED:
                if not record.metadata_consumed_by_owner:
                    raise RuntimeError(
                        "decode actor retirement precedes metadata consumption"
                    )
            elif state is not PackedRequestTransactionState.CANCELLED:
                raise RuntimeError(
                    "decode actor retirement requires committed or cancelled state"
                )
            self._retire_record_locked(record)

    def cancel_terminal_owner_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
    ) -> object:
        """Cancel unpublished resources while retaining native actor identity.

        Native retirement remains the sole authority for removing the actor
        binding. The unpublished rollback itself is scheduler-affine and safe
        before any transport publication.

        :param transaction: Exact prepared terminal-owner transaction.
        :returns: Retained request owner released by cancellation.
        """

        record = self._record_for_transaction(transaction)
        with self._lock:
            self._require_terminal_owner_record_locked(record)
            return transaction.cancel_unpublished()

    def terminal_owner_inventory(self) -> PackedDecodeOwnerInventory:
        """Return exact actor inventory for health and fail-closed teardown.

        :returns: Active, quarantined, scatter, and adoption populations.
        """

        with self._lock:
            records = tuple(
                record
                for record in self._records.values()
                if record.terminal_owner_bound
            )
            active = tuple(
                sorted(
                    (
                        record.terminal_binding.digest
                        for record in records
                        if record.terminal_binding is not None
                    ),
                )
            )
            quarantined = tuple(
                record.terminal_binding.digest
                for record in records
                if record.terminal_binding is not None
                and record.transaction.state
                is PackedRequestTransactionState.QUARANTINED
            )
            return PackedDecodeOwnerInventory(
                active_bindings=active,
                quarantined_bindings=tuple(sorted(quarantined)),
                in_flight_scatter_count=sum(len(record.scatters) for record in records),
                pending_adoption_count=sum(
                    record.commit_receipt is not None for record in records
                ),
            )

    def poll(self, transaction: PackedDecodeRequestTransaction) -> KVPoll:
        """Advance scheduler-owned scatter, teardown, and commit work.

        :param transaction: Exact request transaction.
        :returns: Scheduler transfer state.
        """

        record = self._record_for_transaction(transaction)
        if record.terminal_owner_bound:
            raise RuntimeError(
                "terminal-owner transaction cannot advance through polling"
            )
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
        if record.terminal_owner_bound:
            raise RuntimeError(
                "terminal-owner metadata requires explicit owner consumption"
            )
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
        with self._lock:
            if record.teardown_sent:
                return
            if (
                record.transaction.state
                is not PackedRequestTransactionState.SCATTER_COMPLETED
            ):
                return
            self._begin_teardown_locked(record)

    def _begin_teardown_locked(self, record: _DecodeRequestRecord) -> None:
        """Send one exact teardown manifest while holding the actor lock.

        :param record: Exact actor-owned request generation.
        """

        if record.teardown_sent:
            raise RuntimeError("packed decode teardown was already sent")
        if (
            record.transaction.state
            is not PackedRequestTransactionState.SCATTER_COMPLETED
        ):
            raise RuntimeError("packed decode teardown precedes scatter completion")
        requests = record.transaction.begin_teardown()
        for request in requests:
            route = record.routes.get(request.writer_id)
            if route is None:
                raise RuntimeError("teardown writer has no authenticated route")
            route.send_message(request)
        record.teardown_sent = True

    def _take_writer_aggregation_signal(
        self,
        record: _DecodeRequestRecord,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Take the first authenticated writer-aggregation boundary.

        :param record: Exact actor-owned request generation.
        :param authenticated_writer_id: Writer proved by the control route.
        :returns: One start transition, or an empty tuple after replay.
        """

        with self._lock:
            if not record.terminal_owner_bound:
                return ()
            issuer = self._source_issuer_locked(record, authenticated_writer_id)
            if record.writer_aggregation_reported:
                return ()
            record.writer_aggregation_reported = True
            return (
                self._control_signal_locked(
                    record,
                    NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
                    issuer,
                ),
            )

    def _take_writer_manifest_signal(
        self,
        record: _DecodeRequestRecord,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Take the one-shot writer-manifest boundary for a bound request.

        :param record: Exact actor-owned request generation.
        :param authenticated_writer_id: Writer proved by the control route.
        :returns: Newly earned boundary, or an empty tuple when incomplete.
        """

        with self._lock:
            if not record.terminal_owner_bound:
                return ()
            if (
                record.transaction.state
                is not PackedRequestTransactionState.WRITERS_COMPLETED
                or record.writer_manifest_reported
            ):
                return ()
            if not record.writer_aggregation_reported:
                raise RuntimeError(
                    "writer manifest completed before aggregation started"
                )
            issuer = self._source_issuer_locked(record, authenticated_writer_id)
            record.writer_manifest_reported = True
            return (
                self._control_signal_locked(
                    record,
                    NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
                    issuer,
                ),
            )

    def _take_ack_signals(
        self,
        record: _DecodeRequestRecord,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Take newly earned ACK aggregation and manifest boundaries.

        :param record: Exact actor-owned request generation.
        :param authenticated_writer_id: Writer proved by the control route.
        :returns: Ordered start and optional completion transitions.
        """

        with self._lock:
            if not record.terminal_owner_bound:
                return ()
            issuer = self._source_issuer_locked(record, authenticated_writer_id)
            signals: list[PackedDecodeOwnerSignal] = []
            if not record.ack_aggregation_reported:
                record.ack_aggregation_reported = True
                signals.append(
                    self._control_signal_locked(
                        record,
                        NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
                        issuer,
                    )
                )
            if record.commit_receipt is None or record.ack_manifest_reported:
                return tuple(signals)
            if (
                record.transaction.state
                is not PackedRequestTransactionState.COMMIT_READY
            ):
                raise RuntimeError(
                    "decode commit receipt exists before ACK-manifest completion"
                )
            record.ack_manifest_reported = True
            signals.append(
                self._control_signal_locked(
                    record,
                    NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
                    issuer,
                )
            )
            return tuple(signals)

    @staticmethod
    def _control_signal_locked(
        record: _DecodeRequestRecord,
        kind: NativeTerminalOwnerEventKind,
        issuer: TerminalProcessIdentity,
    ) -> PackedDecodeOwnerSignal:
        """Build one signal from the exact actor binding under its lock.

        :param record: Exact actor-owned request generation.
        :param kind: Control-ingress transition earned by the message.
        :param issuer: Source process authenticated by the route.
        :returns: Immutable signal for native runtime submission.
        """

        binding = PackedDecodeRuntime._require_terminal_binding_locked(record)
        return PackedDecodeOwnerSignal(
            binding_digest=binding.digest,
            kind=kind,
            issuer=issuer,
        )

    @staticmethod
    def _source_issuer_locked(
        record: _DecodeRequestRecord,
        writer_id: StagingWriterId,
    ) -> TerminalProcessIdentity:
        """Resolve the route writer to its pre-bound source identity.

        :param record: Exact actor-owned request generation.
        :param writer_id: Writer authenticated by the control transport.
        :returns: Unique source process identity from the frozen plan.
        """

        plan = record.terminal_source_plan
        if plan is None:
            raise RuntimeError("terminal source identity plan is unavailable")
        matches = tuple(
            writer.process_identity
            for writer in plan.writers
            if writer.writer_id == writer_id
        )
        if len(matches) != 1:
            raise RuntimeError(
                "authenticated writer is absent from the terminal source plan"
            )
        return matches[0]

    @staticmethod
    def _require_terminal_owner_record_locked(record: _DecodeRequestRecord) -> None:
        """Require one record under exclusive terminal-owner progression.

        :param record: Exact actor-owned request generation.
        """

        if not record.terminal_owner_bound:
            raise RuntimeError("packed request has no terminal owner binding")

    @staticmethod
    def _require_terminal_binding_locked(
        record: _DecodeRequestRecord,
    ) -> TerminalRequestBinding:
        """Return the exact bound lifecycle identity under the actor lock.

        :param record: Exact actor-owned request generation.
        :returns: Bound decode lifecycle identity.
        """

        PackedDecodeRuntime._require_terminal_owner_record_locked(record)
        binding = record.terminal_binding
        if binding is None:
            raise RuntimeError("terminal-owner record lost its binding")
        return binding

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
        with self._lock:
            self._retire_record_locked(record)

    def _retire_record_locked(self, record: _DecodeRequestRecord) -> None:
        """Remove exact actor identity while holding its registry lock.

        :param record: Terminal request whose mutable ownership is exhausted.
        """

        key = record.transaction.snapshot().key
        current = self._records.get(key)
        if current is not record:
            raise RuntimeError("packed request registry ownership changed")
        binding = record.terminal_binding
        if binding is not None:
            indexed_key = self._records_by_terminal_binding.get(binding.digest)
            if indexed_key != key:
                raise RuntimeError("packed terminal binding ownership changed")
            del self._records_by_terminal_binding[binding.digest]
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
    if type(message) in (PackedAuxiliaryOutcome, PackedDFlashBoundaryOutcome):
        return message.plan.key
    if type(message) in (PackedRequestTeardown, PackedRequestTeardownAck):
        return message.key
    raise TypeError(f"unsupported packed message {type(message).__name__}")
