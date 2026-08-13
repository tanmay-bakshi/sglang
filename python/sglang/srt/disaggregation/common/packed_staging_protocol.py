import dataclasses
import enum
import logging
import threading
import traceback
from itertools import pairwise
from typing import Never, Protocol

from sglang.srt.disaggregation.common.staging_layout import (
    DEFAULT_STAGING_ALIGNMENT_BYTES,
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentSpan,
    StagingWriterId,
    StagingWriterLayout,
    build_staging_chunk_layout,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBufferRegistry,
    StagingEndpoint,
    StagingEndpointBufferBinding,
    bind_staging_endpoint_buffers,
)
from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)

logger = logging.getLogger(__name__)

PACKED_REQUEST_GENERATION_BYTES = 16
PACKED_REQUEST_DIGEST_BYTES = 32
PACKED_TEARDOWN_GENERATION_BYTES = 16
PACKED_VISIBILITY_POLICY_DIGEST_BYTES = 32
PACKED_NATIVE_ATTESTATION_DIGEST_BYTES = 32
MAX_PACKED_AUXILIARY_DESTINATION_SEGMENTS = 4096
MAX_PACKED_WRITER_ERROR_BYTES = 4096
MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES = 512
MAX_PACKED_TERMINAL_RECEIPT_BYTES = 512


def _validate_visibility_policy_digest(value: bytes, label: str) -> None:
    """Validate one exact route-policy digest.

    :param value: Candidate SHA-256 digest.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_VISIBILITY_POLICY_DIGEST_BYTES:
        raise ValueError(
            f"{label} must contain {PACKED_VISIBILITY_POLICY_DIGEST_BYTES} "
            f"bytes, got {len(value)}"
        )


@dataclasses.dataclass(frozen=True, order=True)
class PackedChunkKey:
    """Stable identity of one request-local packed transfer chunk.

    :ivar room_id: Bootstrap room identifying the request.
    :ivar chunk_id: Request-local monotonically increasing chunk identifier.
    :ivar request_generation: Bootstrap generation preventing room-key replay.
    """

    room_id: int
    chunk_id: int
    request_generation: bytes

    def __post_init__(self) -> None:
        """Validate the packed chunk identity."""

        if type(self.request_generation) is not bytes:
            raise TypeError("request_generation must be bytes")
        if type(self.room_id) is not int or self.room_id < 0:
            raise ValueError(
                f"room_id must be a non-negative integer, got {self.room_id!r}"
            )
        if type(self.chunk_id) is not int or self.chunk_id < 0:
            raise ValueError(
                f"chunk_id must be a non-negative integer, got {self.chunk_id!r}"
            )
        if len(self.request_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "request_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes, got "
                f"{len(self.request_generation)}"
            )


@dataclasses.dataclass(frozen=True, order=True)
class PackedRequestKey:
    """Stable identity shared by every chunk in one packed request.

    :ivar room_id: Bootstrap room identifying the request.
    :ivar request_generation: Decode allocation generation preventing replay.
    """

    room_id: int
    request_generation: bytes

    def __post_init__(self) -> None:
        """Validate the packed request identity."""

        if type(self.request_generation) is not bytes:
            raise TypeError("request_generation must be bytes")
        if type(self.room_id) is not int or self.room_id < 0:
            raise ValueError(
                f"room_id must be a non-negative integer, got {self.room_id!r}"
            )
        if len(self.request_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "request_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes, got "
                f"{len(self.request_generation)}"
            )

    @classmethod
    def from_chunk_key(cls, key: PackedChunkKey) -> "PackedRequestKey":
        """Project one chunk identity to its request identity.

        :param key: Exact request-local chunk identity.
        :returns: Stable request identity.
        """

        return cls(
            room_id=key.room_id,
            request_generation=key.request_generation,
        )


def _validate_request_digest(value: bytes, label: str) -> None:
    """Validate one exact request transcript digest.

    :param value: Candidate SHA-256 digest.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_REQUEST_DIGEST_BYTES:
        raise ValueError(
            f"{label} must contain {PACKED_REQUEST_DIGEST_BYTES} bytes, "
            f"got {len(value)}"
        )


def _validate_nonnegative_uint64(value: int, label: str) -> None:
    """Validate one bounded unsigned generation or index.

    :param value: Candidate integer.
    :param label: Reader-facing field label.
    """

    if type(value) is not int or value < 0 or value >= (1 << 64):
        raise ValueError(f"{label} must be a uint64")


def _validate_positive_uint64(value: int, label: str) -> None:
    """Validate one bounded nonzero native integer.

    :param value: Candidate integer.
    :param label: Reader-facing field label.
    """

    if type(value) is not int or value <= 0 or value >= (1 << 64):
        raise ValueError(f"{label} must be a positive uint64")


@dataclasses.dataclass(frozen=True, order=True)
class PackedAuxiliaryDestinationSegment:
    """One ordered decoder-owned auxiliary metadata destination.

    :ivar address: Exact process-local destination address for one metadata row.
    :ivar item_length: Exact byte count copied into that address.
    """

    address: int
    item_length: int

    def __post_init__(self) -> None:
        """Validate one bounded native DRAM segment."""

        _validate_positive_uint64(
            self.address,
            "auxiliary destination address",
        )
        _validate_positive_uint64(
            self.item_length,
            "auxiliary destination item_length",
        )
        if self.address + self.item_length > (1 << 64):
            raise ValueError(
                "auxiliary destination segment exceeds uint64 address space"
            )


@dataclasses.dataclass(frozen=True)
class PackedAuxiliaryPlan:
    """Decoder-authored ownership contract for request auxiliary metadata.

    :ivar key: Exact allocation-derived packed request identity.
    :ivar request_slot_generation: Decode request-slot reuse generation.
    :ivar metadata_buffer_index: Exact reserved metadata row.
    :ivar metadata_slot_generation: Exact metadata-row reuse generation.
    :ivar destination_segments: Ordered row addresses and item lengths.
    :ivar canonical_writer_id: First writer in the destination-local cohort.
    :ivar destination_process_generation: Exact destination process generation.
    :ivar native_route_digest: Decode-selected native transport route digest.
    :ivar runtime_cohort_digest: Exact loaded runtime cohort digest.
    """

    key: PackedRequestKey
    request_slot_generation: int
    metadata_buffer_index: int
    metadata_slot_generation: bytes
    destination_segments: tuple[PackedAuxiliaryDestinationSegment, ...]
    canonical_writer_id: StagingWriterId
    destination_process_generation: bytes
    native_route_digest: bytes
    runtime_cohort_digest: bytes

    def __post_init__(self) -> None:
        """Own and validate one complete immutable metadata destination plan."""

        if type(self.key) is not PackedRequestKey:
            raise TypeError("auxiliary plan key must be PackedRequestKey")
        _validate_nonnegative_uint64(
            self.request_slot_generation,
            "auxiliary request_slot_generation",
        )
        _validate_nonnegative_uint64(
            self.metadata_buffer_index,
            "auxiliary metadata_buffer_index",
        )
        if type(self.metadata_slot_generation) is not bytes:
            raise TypeError("auxiliary metadata_slot_generation must be bytes")
        if len(self.metadata_slot_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "auxiliary metadata_slot_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes"
            )
        segments = tuple(self.destination_segments)
        object.__setattr__(self, "destination_segments", segments)
        if len(segments) == 0:
            raise ValueError(
                "auxiliary plan must contain at least one destination segment"
            )
        if len(segments) > MAX_PACKED_AUXILIARY_DESTINATION_SEGMENTS:
            raise ValueError(
                "auxiliary plan destination segment count exceeds "
                f"{MAX_PACKED_AUXILIARY_DESTINATION_SEGMENTS}"
            )
        if any(
            type(segment) is not PackedAuxiliaryDestinationSegment
            for segment in segments
        ):
            raise TypeError(
                "auxiliary destination segments must be "
                "PackedAuxiliaryDestinationSegment"
            )
        if len(set(segments)) != len(segments):
            raise ValueError("auxiliary plan contains duplicate destination segments")
        writer_id = self.canonical_writer_id
        segments_by_address = tuple(
            sorted(segments, key=lambda segment: segment.address)
        )
        for previous, current in pairwise(segments_by_address):
            if current.address < previous.address + previous.item_length:
                raise ValueError(
                    "auxiliary plan contains overlapping destination segments"
                )
        if type(writer_id) is not StagingWriterId:
            raise TypeError("auxiliary canonical_writer_id must be StagingWriterId")
        writer_fields = (
            writer_id.transfer_source_rank,
            writer_id.source_attn_tp_rank,
            writer_id.source_pp_rank,
            writer_id.source_cp_rank,
        )
        if any(
            type(value) is not int or value < 0 or value >= (1 << 32)
            for value in writer_fields
        ):
            raise ValueError("auxiliary canonical writer ranks must be uint32 values")
        if writer_id.source_pp_rank != 0 or writer_id.source_cp_rank != 0:
            raise ValueError("auxiliary canonical writer must use PP0 and CP0")
        if type(self.destination_process_generation) is not bytes:
            raise TypeError("auxiliary destination_process_generation must be bytes")
        if len(self.destination_process_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "auxiliary destination_process_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes"
            )
        _validate_request_digest(
            self.native_route_digest,
            "auxiliary native_route_digest",
        )
        _validate_request_digest(
            self.runtime_cohort_digest,
            "auxiliary runtime_cohort_digest",
        )


@dataclasses.dataclass(frozen=True)
class PackedAuxiliaryOutcome:
    """Authenticated terminal outcome for the exact auxiliary metadata plan.

    :ivar plan: Complete decoder-authored destination plan.
    :ivar writer_id: Claimed source writer identity.
    :ivar native_dram_handle_generation: Exact terminal native DRAM handle.
    :ivar descriptor_digest: Native digest of exact submitted DRAM descriptors.
    :ivar evidence_digest: Native digest of endpoint and runtime evidence.
    """

    plan: PackedAuxiliaryPlan
    writer_id: StagingWriterId
    native_dram_handle_generation: int
    descriptor_digest: bytes
    evidence_digest: bytes

    def __post_init__(self) -> None:
        """Validate bounded terminal auxiliary outcome evidence."""

        if type(self.plan) is not PackedAuxiliaryPlan:
            raise TypeError("auxiliary outcome plan must be PackedAuxiliaryPlan")
        if type(self.writer_id) is not StagingWriterId:
            raise TypeError("auxiliary outcome writer_id must be StagingWriterId")
        if self.writer_id != self.plan.canonical_writer_id:
            raise ValueError(
                "auxiliary outcome writer differs from the plan's canonical writer"
            )
        _validate_positive_uint64(
            self.native_dram_handle_generation,
            "auxiliary native_dram_handle_generation",
        )
        _validate_request_digest(
            self.descriptor_digest,
            "auxiliary descriptor_digest",
        )
        _validate_request_digest(
            self.evidence_digest,
            "auxiliary evidence_digest",
        )


@dataclasses.dataclass(frozen=True)
class PackedRequestTeardown:
    """Destination request for one writer to retire a complete request.

    :ivar key: Exact packed request identity.
    :ivar writer_id: Exact source writer being retired.
    :ivar request_slot_generation: Decode request-slot reuse generation.
    :ivar writer_manifest_digest: Exact writer-membership digest.
    :ivar allocation_digest: Exact decode allocation digest.
    :ivar teardown_generation: One-shot request teardown generation.
    :ivar auxiliary_handle_generation: Exact auxiliary DRAM handle retired by
        the canonical writer, otherwise None.
    """

    key: PackedRequestKey
    writer_id: StagingWriterId
    request_slot_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    teardown_generation: bytes
    auxiliary_handle_generation: int | None = None

    def __post_init__(self) -> None:
        """Validate bounded request teardown identity."""

        if type(self.key) is not PackedRequestKey:
            raise TypeError("teardown key must be PackedRequestKey")
        if type(self.writer_id) is not StagingWriterId:
            raise TypeError("teardown writer_id must be StagingWriterId")
        _validate_nonnegative_uint64(
            self.request_slot_generation,
            "teardown request_slot_generation",
        )
        _validate_request_digest(
            self.writer_manifest_digest,
            "teardown writer_manifest_digest",
        )
        _validate_request_digest(
            self.allocation_digest,
            "teardown allocation_digest",
        )
        if type(self.teardown_generation) is not bytes:
            raise TypeError("teardown_generation must be bytes")
        if len(self.teardown_generation) != PACKED_TEARDOWN_GENERATION_BYTES:
            raise ValueError(
                "teardown_generation must contain "
                f"{PACKED_TEARDOWN_GENERATION_BYTES} bytes, got "
                f"{len(self.teardown_generation)}"
            )
        if self.auxiliary_handle_generation is not None:
            _validate_positive_uint64(
                self.auxiliary_handle_generation,
                "teardown auxiliary_handle_generation",
            )


@dataclasses.dataclass(frozen=True)
class PackedRequestTeardownAck:
    """Authenticated acknowledgement of one exact request teardown.

    :ivar key: Exact packed request identity.
    :ivar writer_id: Claimed source writer identity.
    :ivar request_slot_generation: Decode request-slot reuse generation.
    :ivar writer_manifest_digest: Exact writer-membership digest.
    :ivar allocation_digest: Exact decode allocation digest.
    :ivar teardown_generation: Exact one-shot teardown generation.
    :ivar auxiliary_handle_generation: Exact auxiliary DRAM handle retired by
        the canonical writer, otherwise None.
    """

    key: PackedRequestKey
    writer_id: StagingWriterId
    request_slot_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    teardown_generation: bytes
    auxiliary_handle_generation: int | None = None

    def __post_init__(self) -> None:
        """Validate bounded request teardown acknowledgement identity."""

        PackedRequestTeardown(
            key=self.key,
            writer_id=self.writer_id,
            request_slot_generation=self.request_slot_generation,
            writer_manifest_digest=self.writer_manifest_digest,
            allocation_digest=self.allocation_digest,
            teardown_generation=self.teardown_generation,
            auxiliary_handle_generation=self.auxiliary_handle_generation,
        )


@dataclasses.dataclass(frozen=True)
class PackedTerminalReceipt:
    """Authenticated-route envelope carrying terminal owner authority.

    The packed control route authenticates the sending process independently.
    The opaque receipt payload is decoded and joined with that route identity
    by the terminal-progress import namespace, never by this framing layer.

    :ivar key: Exact packed request identity used for bounded routing.
    :ivar receipt_payload: Canonical fixed-width terminal receipt bytes.
    """

    key: PackedRequestKey
    receipt_payload: bytes

    def __post_init__(self) -> None:
        """Validate bounded terminal receipt framing."""

        if type(self.key) is not PackedRequestKey:
            raise TypeError("terminal receipt key must be PackedRequestKey")
        if type(self.receipt_payload) is not bytes:
            raise TypeError("terminal receipt payload must be bytes")
        if len(self.receipt_payload) == 0:
            raise ValueError("terminal receipt payload must not be empty")
        if len(self.receipt_payload) > MAX_PACKED_TERMINAL_RECEIPT_BYTES:
            raise ValueError(
                "terminal receipt payload exceeds "
                f"{MAX_PACKED_TERMINAL_RECEIPT_BYTES} bytes"
            )


@dataclasses.dataclass(frozen=True)
class PackedTopology:
    """Explicit tensor-parallel topology used to derive a packed layout.

    :ivar source_tp_size: Source attention tensor-parallel width.
    :ivar destination_tp_size: Destination attention tensor-parallel width.
    :ivar destination_tp_rank: Destination attention tensor-parallel rank.
    :ivar alignment_bytes: Packed projection alignment.
    """

    source_tp_size: int
    destination_tp_size: int
    destination_tp_rank: int
    alignment_bytes: int = DEFAULT_STAGING_ALIGNMENT_BYTES

    def __post_init__(self) -> None:
        """Validate topology values before layout construction."""

        if (
            type(self.source_tp_size) is not int
            or self.source_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES
        ):
            raise ValueError(
                "source_tp_size must be one of "
                f"{SUPPORTED_PACKED_SOURCE_TP_SIZES}, got {self.source_tp_size}"
            )
        if type(self.destination_tp_size) is not int or self.destination_tp_size <= 0:
            raise ValueError(
                f"destination_tp_size must be positive, got {self.destination_tp_size}"
            )
        if (
            type(self.destination_tp_rank) is not int
            or self.destination_tp_rank < 0
            or self.destination_tp_rank >= self.destination_tp_size
        ):
            raise ValueError(
                "destination_tp_rank must be in "
                f"[0, {self.destination_tp_size}), got "
                f"{self.destination_tp_rank}"
            )
        if type(self.alignment_bytes) is not int or self.alignment_bytes <= 0:
            raise ValueError(
                f"alignment_bytes must be positive, got {self.alignment_bytes}"
            )


@dataclasses.dataclass(frozen=True)
class PackedLayoutSpec:
    """Complete immutable input used to rebuild a canonical packed layout.

    A received spec is never trusted as a prebuilt layout. The decode side
    reconstructs it through :func:`build_staging_chunk_layout` and compares the
    result with its independently registered canonical layout.

    :ivar chunk_id: Request-local chunk identifier.
    :ivar is_last: Whether the chunk completes the request.
    :ivar spans: Exact component-local source and destination spans.
    :ivar source_components: Source registration geometries.
    :ivar destination_components: Destination registration geometries.
    :ivar writers: Complete authenticated writer topology.
    :ivar topology: Tensor-parallel topology and packed alignment.
    """

    chunk_id: int
    is_last: bool
    spans: tuple[StagingComponentSpan, ...]
    source_components: tuple[StagingComponentGeometry, ...]
    destination_components: tuple[StagingComponentGeometry, ...]
    writers: tuple[StagingWriterId, ...]
    topology: PackedTopology

    def __post_init__(self) -> None:
        """Own immutable copies of every layout-bearing sequence."""

        if type(self.chunk_id) is not int or self.chunk_id < 0:
            raise ValueError("packed layout chunk_id must be a non-negative integer")
        if type(self.is_last) is not bool:
            raise TypeError("packed layout is_last must be bool")
        if type(self.topology) is not PackedTopology:
            raise TypeError("packed layout topology must be PackedTopology")
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "source_components", tuple(self.source_components))
        object.__setattr__(
            self, "destination_components", tuple(self.destination_components)
        )
        object.__setattr__(self, "writers", tuple(self.writers))

    def build(self) -> StagingChunkLayout:
        """Rebuild the canonical packed layout represented by this spec.

        :returns: Canonical immutable packed layout.
        :raises ValueError: If the spec is incomplete or internally inconsistent.
        """

        layout = build_staging_chunk_layout(
            chunk_id=self.chunk_id,
            is_last=self.is_last,
            spans=self.spans,
            source_components=self.source_components,
            destination_components=self.destination_components,
            source_tp_size=self.topology.source_tp_size,
            destination_tp_size=self.topology.destination_tp_size,
            destination_tp_rank=self.topology.destination_tp_rank,
            writers=self.writers,
            alignment_bytes=self.topology.alignment_bytes,
        )
        active_component_ids = {span.component_id for span in layout.component_spans}
        source_component_ids = {
            geometry.component_id for geometry in self.source_components
        }
        destination_component_ids = {
            geometry.component_id for geometry in self.destination_components
        }
        if source_component_ids != active_component_ids:
            raise ValueError(
                "source component geometries must exactly match active spans"
            )
        if destination_component_ids != active_component_ids:
            raise ValueError(
                "destination component geometries must exactly match active spans"
            )
        if len(self.source_components) != len(active_component_ids):
            raise ValueError("source component geometries must be unique")
        if len(self.destination_components) != len(active_component_ids):
            raise ValueError("destination component geometries must be unique")
        return layout


@dataclasses.dataclass(frozen=True)
class PackedPrepare:
    """Writer declaration of one immutable packed transfer plan.

    ``writer_id`` is descriptive wire data, not an authority. The receiver must
    pass the independently authenticated transport peer to
    :meth:`PackedDecodeProtocol.handle_prepare`.

    :ivar key: Request and chunk identity.
    :ivar writer_id: Claimed writer identity.
    :ivar spec: Complete layout-construction input.
    :ivar digest: Claimed canonical layout digest.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    spec: PackedLayoutSpec
    digest: bytes

    def __post_init__(self) -> None:
        """Own and validate an immutable copy of the digest."""

        if type(self.digest) is not bytes:
            raise TypeError("PREPARE digest must be bytes")
        if len(self.digest) != 32:
            raise ValueError(
                f"PREPARE digest must contain 32 bytes, got {len(self.digest)}"
            )


@dataclasses.dataclass(frozen=True)
class PackedReady:
    """Decode lease projection granted to exactly one writer.

    :ivar key: Request and chunk identity.
    :ivar writer_id: Writer owning the projection.
    :ivar digest: Canonical layout digest.
    :ivar visibility_policy_digest: Decode-selected route-policy identity.
    :ivar lease_id: Decode allocation identity.
    :ivar lease_base_address: Base address of the one contiguous decode lease.
    :ivar projection_offset: Writer-local offset into the lease.
    :ivar projection_length: Exact writer projection length.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    digest: bytes
    visibility_policy_digest: bytes
    lease_id: int
    lease_base_address: int
    projection_offset: int
    projection_length: int

    def __post_init__(self) -> None:
        """Own and validate READY lease metadata."""

        if type(self.digest) is not bytes:
            raise TypeError("READY digest must be bytes")
        if len(self.digest) != 32:
            raise ValueError(
                f"READY digest must contain 32 bytes, got {len(self.digest)}"
            )
        _validate_visibility_policy_digest(
            self.visibility_policy_digest,
            "READY visibility_policy_digest",
        )
        if type(self.lease_id) is not int or self.lease_id < 0:
            raise ValueError(
                f"READY lease_id must be non-negative, got {self.lease_id!r}"
            )
        if type(self.lease_base_address) is not int or self.lease_base_address <= 0:
            raise ValueError(
                "READY lease_base_address must be positive, got "
                f"{self.lease_base_address!r}"
            )
        if type(self.projection_offset) is not int or self.projection_offset < 0:
            raise ValueError(
                "READY projection_offset must be non-negative, got "
                f"{self.projection_offset!r}"
            )
        if type(self.projection_length) is not int or self.projection_length <= 0:
            raise ValueError(
                "READY projection_length must be positive, got "
                f"{self.projection_length!r}"
            )


class PackedWriterOutcomeStatus(enum.StrEnum):
    """Terminal status of one writer's exact registered transfer."""

    DONE = "done"
    ERROR = "error"


class PackedTransportPath(enum.StrEnum):
    """One independently ordered data path into destination memory."""

    CUDA_IPC = "cuda_ipc"
    NIC_RDMA = "nic_rdma"


class PackedWriterVisibilityAction(enum.StrEnum):
    """Source-side action proven before a successful terminal outcome."""

    CUDA_EVENT_RECORDED = "cuda_event_recorded"
    TRANSPORT_HANDLE_TERMINAL = "transport_handle_terminal"


class PackedWriterCompletionMechanism(enum.StrEnum):
    """Concrete source primitive backing successful visibility evidence."""

    EXPORTED_CUDA_EVENT_RECORDED = "exported_cuda_event_recorded"
    NIXL_TRANSFER_HANDLE_TERMINAL = "nixl_transfer_handle_terminal"


@dataclasses.dataclass(frozen=True)
class PackedWriterVisibilityEvidence:
    """Bounded source evidence used to derive destination-local visibility.

    :ivar policy_digest: Decode-selected route-policy identity from READY.
    :ivar transport_path: Exact independently ordered transfer path.
    :ivar lane_identifier: Bootstrap-pinned CUDA IPC or UCX lane identity.
    :ivar completion_mechanism: Exact source completion primitive observed.
    :ivar writer_action: Source-side action completed before the outcome.
    :ivar native_handle_generation: NIXL handle generation authorized by a
        native completion receipt.
    :ivar native_descriptor_digest: Native digest of the exact submitted
        descriptors.
    :ivar native_evidence_digest: Native digest of the endpoint, remote-flush,
        and loaded-runtime evidence.
    """

    policy_digest: bytes
    transport_path: PackedTransportPath
    lane_identifier: str
    completion_mechanism: PackedWriterCompletionMechanism
    writer_action: PackedWriterVisibilityAction
    native_handle_generation: int | None = None
    native_descriptor_digest: bytes | None = None
    native_evidence_digest: bytes | None = None

    def __post_init__(self) -> None:
        """Validate exact, bounded visibility evidence."""

        _validate_visibility_policy_digest(
            self.policy_digest,
            "visibility policy_digest",
        )
        if type(self.transport_path) is not PackedTransportPath:
            raise TypeError("visibility transport_path must be PackedTransportPath")
        if type(self.writer_action) is not PackedWriterVisibilityAction:
            raise TypeError(
                "visibility writer_action must be PackedWriterVisibilityAction"
            )
        if type(self.completion_mechanism) is not PackedWriterCompletionMechanism:
            raise TypeError(
                "visibility completion_mechanism must be "
                "PackedWriterCompletionMechanism"
            )
        self._validate_bounded_text(
            self.lane_identifier,
            "visibility lane_identifier",
            MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES,
        )
        expected_action = (
            PackedWriterVisibilityAction.CUDA_EVENT_RECORDED
            if self.completion_mechanism
            is PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            else PackedWriterVisibilityAction.TRANSPORT_HANDLE_TERMINAL
        )
        if self.writer_action is not expected_action:
            raise ValueError(
                "visibility writer_action does not match its completion mechanism"
            )
        if (
            self.transport_path is PackedTransportPath.NIC_RDMA
            and self.completion_mechanism
            is not PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        ):
            raise ValueError("NIC RDMA visibility requires native NIXL completion")
        if (
            self.completion_mechanism
            is PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
        ):
            if (
                self.native_handle_generation is not None
                or self.native_descriptor_digest is not None
                or self.native_evidence_digest is not None
            ):
                raise ValueError(
                    "direct CUDA-event visibility must not contain native "
                    "NIXL attestation"
                )
            return
        if (
            type(self.native_handle_generation) is not int
            or self.native_handle_generation <= 0
            or self.native_handle_generation >= (1 << 64)
        ):
            raise ValueError(
                "native NIXL visibility requires a positive uint64 handle generation"
            )
        _validate_native_attestation_digest(
            self.native_descriptor_digest,
            "native descriptor digest",
        )
        _validate_native_attestation_digest(
            self.native_evidence_digest,
            "native evidence digest",
        )

    @staticmethod
    def _validate_bounded_text(value: str, label: str, maximum_bytes: int) -> None:
        """Validate one non-empty bounded UTF-8 evidence field.

        :param value: Candidate text.
        :param label: Reader-facing field label.
        :param maximum_bytes: Maximum encoded length.
        """

        if type(value) is not str or len(value) == 0:
            raise ValueError(f"{label} must be a non-empty string")
        encoded_length = len(value.encode("utf-8"))
        if encoded_length > maximum_bytes:
            raise ValueError(f"{label} exceeds {maximum_bytes} UTF-8 bytes")


def _validate_native_attestation_digest(
    value: bytes | None,
    label: str,
) -> None:
    """Validate one exact native SHA-256 attestation digest.

    :param value: Candidate native digest.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_NATIVE_ATTESTATION_DIGEST_BYTES:
        raise ValueError(
            f"{label} must contain {PACKED_NATIVE_ATTESTATION_DIGEST_BYTES} "
            f"bytes, got {len(value)}"
        )


@dataclasses.dataclass(frozen=True)
class PackedWriterOutcome:
    """Authenticated terminal outcome of one exact writer transfer.

    ``DONE`` is valid only after transport completion. ``ERROR`` is valid only
    after the source has proven that no DMA was submitted or the submitted
    transport handle reached terminal error. Ambiguous submission or connection
    loss must not produce an outcome.

    :ivar key: Request and chunk identity.
    :ivar writer_id: Claimed writer identity.
    :ivar digest: Canonical layout digest.
    :ivar lease_id: Decode allocation identity received in READY.
    :ivar status: Proven terminal transport status.
    :ivar visibility: Successful writer-side visibility evidence.
    :ivar reason: Bounded reader-facing failure reason for ``ERROR``.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    digest: bytes
    lease_id: int
    status: PackedWriterOutcomeStatus
    visibility: PackedWriterVisibilityEvidence | None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Own and validate terminal outcome metadata."""

        if type(self.digest) is not bytes:
            raise TypeError("writer outcome digest must be bytes")
        if len(self.digest) != 32:
            raise ValueError(
                f"writer outcome digest must contain 32 bytes, got {len(self.digest)}"
            )
        if type(self.lease_id) is not int or self.lease_id < 0:
            raise ValueError(
                f"writer outcome lease_id must be non-negative, got {self.lease_id!r}"
            )
        if type(self.status) is not PackedWriterOutcomeStatus:
            raise TypeError(
                "writer outcome status must be PackedWriterOutcomeStatus, got "
                f"{type(self.status)!r}"
            )
        if self.status is PackedWriterOutcomeStatus.DONE:
            if self.reason is not None:
                raise ValueError("DONE writer outcome must not contain a reason")
            if type(self.visibility) is not PackedWriterVisibilityEvidence:
                raise TypeError("DONE writer outcome must contain visibility evidence")
            return
        if self.visibility is not None:
            raise ValueError(
                "ERROR writer outcome must not contain visibility evidence"
            )
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("ERROR writer outcome must contain a non-empty reason")
        if len(self.reason.encode("utf-8")) > MAX_PACKED_WRITER_ERROR_BYTES:
            raise ValueError(
                "writer outcome reason exceeds "
                f"{MAX_PACKED_WRITER_ERROR_BYTES} UTF-8 bytes"
            )


_WRITER_OUTCOME_TICKET_ISSUER_RECEIPT = object()
_WRITER_OUTCOME_TICKET_RECEIPT = object()


@dataclasses.dataclass(frozen=True, eq=False)
class _PackedWriterOutcomeTicket:
    """Protocol-instance-bound admission for one exact successful outcome.

    Tickets are issued only by the destination visibility coordinator after its
    required CUDA action completes. They are intentionally identity-bearing and
    single-use rather than wire-serializable values.

    :ivar message: Exact successful writer outcome admitted by the coordinator.
    :ivar authenticated_writer_id: Transport-authenticated writer identity.
    """

    message: PackedWriterOutcome
    authenticated_writer_id: StagingWriterId
    _protocol_nonce: object = dataclasses.field(repr=False, compare=False)
    _ticket_identity: object = dataclasses.field(repr=False, compare=False)
    _construction_receipt: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject tickets not constructed by the claimed protocol issuer."""

        if self._construction_receipt is not _WRITER_OUTCOME_TICKET_RECEIPT:
            raise TypeError("writer outcome ticket requires a protocol issuer")
        if type(self.message) is not PackedWriterOutcome:
            raise TypeError("writer outcome ticket message must be PackedWriterOutcome")
        if self.message.status is not PackedWriterOutcomeStatus.DONE:
            raise ValueError("writer outcome tickets admit only DONE")
        if self.message.writer_id != self.authenticated_writer_id:
            raise ValueError(
                "writer outcome ticket identity differs from its authenticated writer"
            )


class _PackedWriterOutcomeTicketIssuer:
    """Single-coordinator authority to issue protocol-bound DONE tickets."""

    _protocol_nonce: object

    def __init__(self, protocol_nonce: object, construction_receipt: object) -> None:
        """Construct one protocol-owned issuer.

        :param protocol_nonce: Exact protocol-instance identity.
        :param construction_receipt: Module-private construction authority.
        """

        if construction_receipt is not _WRITER_OUTCOME_TICKET_ISSUER_RECEIPT:
            raise TypeError("writer outcome ticket issuer is protocol-owned")
        self._protocol_nonce = protocol_nonce

    def _issue(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> _PackedWriterOutcomeTicket:
        """Issue one single-use ticket bound to exact authenticated DONE.

        :param message: Successful outcome whose destination action completed.
        :param authenticated_writer_id: Writer authenticated by the transport.
        :returns: Protocol-instance-bound admission ticket.
        """

        return _PackedWriterOutcomeTicket(
            message=message,
            authenticated_writer_id=authenticated_writer_id,
            _protocol_nonce=self._protocol_nonce,
            _ticket_identity=object(),
            _construction_receipt=_WRITER_OUTCOME_TICKET_RECEIPT,
        )


@dataclasses.dataclass(frozen=True)
class PackedLease:
    """One contiguous decode-side staging allocation.

    :ivar lease_id: Allocator identity.
    :ivar base_address: Registered destination base pointer.
    :ivar length_bytes: Contiguous registered capacity.
    """

    lease_id: int
    base_address: int
    length_bytes: int

    def __post_init__(self) -> None:
        """Validate the allocator-provided lease."""

        if type(self.lease_id) is not int or self.lease_id < 0:
            raise ValueError(
                f"lease_id must be a non-negative integer, got {self.lease_id!r}"
            )
        if type(self.base_address) is not int or self.base_address <= 0:
            raise ValueError(
                f"base_address must be a positive integer, got {self.base_address!r}"
            )
        if type(self.length_bytes) is not int or self.length_bytes <= 0:
            raise ValueError(
                f"length_bytes must be a positive integer, got {self.length_bytes!r}"
            )


class PackedLeaseAllocator(Protocol):
    """Allocator contract required by the decode protocol.

    Every callback runs while the protocol serializes chunk transitions. An
    implementation must complete without waiting for device or network work,
    must not call back into :class:`PackedDecodeProtocol`, and must be
    transactional: if it raises, lease ownership must remain unchanged so the
    same operation can be retried safely.
    """

    def allocate(self, length_bytes: int) -> PackedLease:
        """Allocate one contiguous registered staging lease.

        :param length_bytes: Minimum required bytes.
        :returns: Contiguous registered lease.
        """

        ...

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Prevent a failed lease from being reused.

        :param lease: Lease that may still be targeted or read asynchronously.
        :param reason: First protocol failure reason.
        """

        ...

    def release(self, lease: PackedLease) -> None:
        """Return a terminally quiescent lease to the allocator.

        :param lease: Lease safe for immediate reuse.
        """

        ...


class PackedProtocolState(enum.StrEnum):
    """Decode-side packed chunk lifecycle."""

    COLLECTING = "collecting"
    READY = "ready"
    SCATTER_READY = "scatter_ready"
    SCATTERING = "scattering"
    RELEASED = "released"
    FAILED_QUARANTINED = "failed_quarantined"
    FAILED_RELEASED = "failed_released"


class PackedProtocolError(RuntimeError):
    """Protocol violation that has failed one packed chunk."""

    key: PackedChunkKey
    reason: str

    def __init__(self, key: PackedChunkKey, reason: str) -> None:
        """Initialize a chunk-scoped protocol failure.

        :param key: Failed request and chunk identity.
        :param reason: Stable reader-facing failure reason.
        """

        self.key = key
        self.reason = reason
        super().__init__(f"packed chunk {key} failed: {reason}")


@dataclasses.dataclass(frozen=True)
class PackedScatterWork:
    """Immutable inputs for one asynchronous destination scatter.

    :ivar key: Request and chunk identity.
    :ivar layout: Canonical packed layout.
    :ivar lease: Contiguous source staging lease.
    :ivar destination_binding: Immutable destination page-array snapshots.
    """

    key: PackedChunkKey
    layout: StagingChunkLayout
    lease: PackedLease
    destination_binding: StagingEndpointBufferBinding


@dataclasses.dataclass(frozen=True)
class PackedChunkSnapshot:
    """Read-only protocol state for diagnostics and tests.

    :ivar key: Request and chunk identity.
    :ivar state: Current lifecycle state.
    :ivar prepared_writers: Canonically ordered accepted PREPARE writers.
    :ivar writer_outcomes: Canonically ordered authenticated terminal outcomes.
    :ivar quiesced_writers: Canonically ordered explicitly quiesced writers.
    :ivar lease_id: Current or released lease identity.
    :ivar ready_issued: Whether any writer could have observed READY.
    :ivar scatter_started: Whether scatter ownership was handed to async work.
    :ivar scatter_terminal: Whether started scatter work is terminal.
    :ivar failure_reason: First protocol failure, if any.
    """

    key: PackedChunkKey
    state: PackedProtocolState
    prepared_writers: tuple[StagingWriterId, ...]
    writer_outcomes: tuple[PackedWriterOutcome, ...]
    quiesced_writers: tuple[StagingWriterId, ...]
    lease_id: int | None
    ready_issued: bool
    scatter_started: bool
    scatter_terminal: bool
    failure_reason: str | None


@dataclasses.dataclass
class _PackedChunk:
    """Mutable state owned exclusively by :class:`PackedDecodeProtocol`."""

    key: PackedChunkKey
    spec: PackedLayoutSpec
    layout: StagingChunkLayout
    destination_binding: StagingEndpointBufferBinding
    expected_writers: tuple[StagingWriterId, ...]
    writer_visibility_policy_digests: dict[StagingWriterId, bytes]
    state: PackedProtocolState = PackedProtocolState.COLLECTING
    prepares: dict[StagingWriterId, PackedPrepare] = dataclasses.field(
        default_factory=dict
    )
    writer_outcomes: dict[StagingWriterId, PackedWriterOutcome] = dataclasses.field(
        default_factory=dict
    )
    consumed_writer_outcome_tickets: set[object] = dataclasses.field(
        default_factory=set
    )
    quiesced_writers: set[StagingWriterId] = dataclasses.field(default_factory=set)
    lease: PackedLease | None = None
    ready_issued: bool = False
    quarantined: bool = False
    scatter_started: bool = False
    scatter_terminal: bool = False
    failure_reason: str | None = None


def _writer_layout(
    layout: StagingChunkLayout,
    writer_id: StagingWriterId,
) -> StagingWriterLayout:
    """Return one canonical writer projection.

    :param layout: Canonical packed layout.
    :param writer_id: Expected writer identity.
    :returns: Exact writer projection.
    :raises ValueError: If the writer is absent.
    """

    for writer_layout in layout.writers:
        if writer_layout.writer_id == writer_id:
            return writer_layout
    raise ValueError(f"writer is absent from canonical layout: {writer_id}")


class PackedDecodeProtocol:
    """Thread-safe decode-side consensus and lease-safety state machine."""

    _allocator: PackedLeaseAllocator
    _allocator_callback_active: bool
    _chunks: dict[PackedChunkKey, _PackedChunk]
    _lock: threading.RLock
    _writer_outcome_ticket_issuer: _PackedWriterOutcomeTicketIssuer
    _writer_outcome_ticket_issuer_claimed: bool
    _writer_outcome_ticket_nonce: object

    def __init__(self, allocator: PackedLeaseAllocator) -> None:
        """Initialize an empty decode protocol.

        :param allocator: Contiguous registered staging lease allocator.
        """

        self._allocator = allocator
        self._allocator_callback_active = False
        self._chunks = {}
        self._lock = threading.RLock()
        self._writer_outcome_ticket_nonce = object()
        self._writer_outcome_ticket_issuer = _PackedWriterOutcomeTicketIssuer(
            self._writer_outcome_ticket_nonce,
            _WRITER_OUTCOME_TICKET_ISSUER_RECEIPT,
        )
        self._writer_outcome_ticket_issuer_claimed = False

    def _claim_writer_outcome_ticket_issuer(
        self,
    ) -> _PackedWriterOutcomeTicketIssuer:
        """Transfer DONE-admission authority to one destination coordinator.

        :returns: Protocol-bound ticket issuer.
        :raises RuntimeError: If an issuer has already been claimed.
        """

        with self._lock:
            self._require_no_allocator_callback_locked()
            if self._writer_outcome_ticket_issuer_claimed:
                raise RuntimeError(
                    "writer outcome ticket issuer has already been claimed"
                )
            self._writer_outcome_ticket_issuer_claimed = True
            return self._writer_outcome_ticket_issuer

    def _writer_visibility_policy_digests(
        self,
        key: PackedChunkKey,
    ) -> dict[StagingWriterId, bytes]:
        """Return immutable-value policy truth for coordinator registration.

        :param key: Exact registered chunk identity.
        :returns: Copy of the canonical writer policy digests.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            return dict(chunk.writer_visibility_policy_digests)

    def register_chunk(
        self,
        key: PackedChunkKey,
        spec: PackedLayoutSpec,
        destination_registry: StagingComponentBufferRegistry,
        writer_visibility_policy_digests: dict[StagingWriterId, bytes],
    ) -> StagingChunkLayout:
        """Register trusted decode-local geometry and destination bounds.

        The canonical layout is rebuilt locally and all destination page arrays
        are copied into immutable snapshots before any PREPARE can reach
        consensus.

        :param key: Request and chunk identity.
        :param spec: Trusted layout input assembled from bootstrap metadata.
        :param destination_registry: Decode-local registered buffers and pages.
        :param writer_visibility_policy_digests: Exact policy per writer route.
        :returns: Canonical packed layout.
        :raises ValueError: If identity, geometry, topology, or bounds are invalid.
        """

        if key.chunk_id != spec.chunk_id:
            raise ValueError(
                f"chunk key/spec mismatch: {key.chunk_id} and {spec.chunk_id}"
            )
        layout = spec.build()
        destination_binding = bind_staging_endpoint_buffers(
            layout,
            StagingEndpoint.DESTINATION,
            destination_registry,
        )
        expected_writers = tuple(
            writer_layout.writer_id for writer_layout in layout.writers
        )
        if set(writer_visibility_policy_digests) != set(expected_writers):
            raise ValueError(
                "writer visibility policies must exactly cover canonical writers"
            )
        owned_policy_digests: dict[StagingWriterId, bytes] = {}
        for writer_id in expected_writers:
            digest = writer_visibility_policy_digests[writer_id]
            _validate_visibility_policy_digest(
                digest,
                f"writer {writer_id} visibility policy digest",
            )
            owned_policy_digests[writer_id] = digest
        with self._lock:
            self._require_no_allocator_callback_locked()
            if key in self._chunks:
                raise ValueError(f"packed chunk is already registered: {key}")
            self._chunks[key] = _PackedChunk(
                key=key,
                spec=spec,
                layout=layout,
                destination_binding=destination_binding,
                expected_writers=expected_writers,
                writer_visibility_policy_digests=owned_policy_digests,
            )
        return layout

    def handle_prepare(
        self,
        message: PackedPrepare,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedReady, ...]:
        """Accept one authenticated writer's immutable layout declaration.

        The claimed writer identity is checked against the transport-authenticated
        peer before it can affect consensus. Allocation occurs exactly once,
        after every expected writer has submitted the same canonical layout.

        :param message: Untrusted PREPARE payload.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: READY messages produced by final unique-writer consensus.
        :raises PackedProtocolError: If the payload conflicts with local truth.
        """

        with self._lock:
            chunk = self._require_chunk_locked(message.key)
            self._require_live_locked(chunk)
            if authenticated_writer_id != message.writer_id:
                self._reject_locked(
                    chunk,
                    "PREPARE writer identity does not match authenticated peer",
                )
            self._validate_prepare_locked(chunk, message)

            previous = chunk.prepares.get(authenticated_writer_id)
            if previous is not None:
                if previous != message:
                    self._reject_locked(
                        chunk,
                        f"conflicting duplicate PREPARE from {authenticated_writer_id}",
                    )
                return ()

            chunk.prepares[authenticated_writer_id] = message
            if set(chunk.prepares) != set(chunk.expected_writers):
                return ()
            return self._allocate_and_ready_locked(chunk)

    def handle_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
        ticket: _PackedWriterOutcomeTicket | None = None,
    ) -> bool:
        """Record one authenticated writer's proven terminal transport outcome.

        A successful outcome contributes to scatter consensus. An error fails
        the chunk and proves only that exact writer terminal; every other
        possible writer remains an owner until its own outcome or trusted local
        quiescence proof arrives. Adapter callers must establish destination
        CUDA visibility and supply its protocol-bound ticket before submitting
        ``DONE`` here. Identical duplicates are idempotent and do not consume a
        second ticket.

        :param message: Untrusted terminal outcome payload.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :param ticket: Coordinator-issued admission required for new ``DONE``.
        :returns: Whether this outcome newly made scatter eligible.
        :raises PackedProtocolError: If identity or lease metadata conflicts.
        """

        with self._lock:
            chunk, previous = self._preflight_writer_outcome_locked(
                message,
                authenticated_writer_id,
            )
            if previous is not None:
                self._release_failed_if_safe_locked(chunk)
                return False
            if message.status is PackedWriterOutcomeStatus.DONE:
                self._consume_writer_outcome_ticket_locked(
                    chunk,
                    message,
                    authenticated_writer_id,
                    ticket,
                )
            elif ticket is not None:
                self._reject_locked(
                    chunk,
                    "ERROR writer outcome must not contain a visibility ticket",
                )

            chunk.writer_outcomes[authenticated_writer_id] = message
            if chunk.state is PackedProtocolState.FAILED_QUARANTINED:
                self._release_failed_if_safe_locked(chunk)
                return False
            if message.status is PackedWriterOutcomeStatus.ERROR:
                self._fail_locked(
                    chunk,
                    f"writer {authenticated_writer_id} failed: {message.reason}",
                )
                return False
            if set(chunk.writer_outcomes) != set(chunk.expected_writers):
                return False
            if chunk.state is not PackedProtocolState.READY:
                self._reject_locked(
                    chunk,
                    "complete writer outcome set is invalid in state "
                    f"{chunk.state.value}",
                )
            chunk.state = PackedProtocolState.SCATTER_READY
            return True

    def preflight_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Validate an outcome before a coordinator performs CUDA work.

        This check does not admit or record a new outcome. The later commit
        revalidates the same invariants under the protocol lock.

        :param message: Untrusted terminal outcome payload.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether the exact outcome still requires admission.
        """

        with self._lock:
            _, previous = self._preflight_writer_outcome_locked(
                message,
                authenticated_writer_id,
            )
            return previous is None

    def begin_scatter(self, key: PackedChunkKey) -> PackedScatterWork:
        """Transfer lease ownership to one asynchronous scatter operation.

        Calling this method establishes that scatter may be in flight. Any later
        failure must be followed by :meth:`quiesce_scatter` before the lease can
        be released.

        :param key: Request and chunk identity.
        :returns: Immutable scatter inputs.
        :raises PackedProtocolError: If successful writer consensus is incomplete.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state is not PackedProtocolState.SCATTER_READY:
                raise PackedProtocolError(
                    key,
                    f"scatter cannot begin in state {chunk.state.value}",
                )
            lease = chunk.lease
            if lease is None:
                raise PackedProtocolError(key, "scatter-ready chunk has no lease")
            chunk.scatter_started = True
            chunk.scatter_terminal = False
            chunk.state = PackedProtocolState.SCATTERING
            return PackedScatterWork(
                key=key,
                layout=chunk.layout,
                lease=lease,
                destination_binding=chunk.destination_binding,
            )

    def complete_scatter(self, key: PackedChunkKey) -> None:
        """Release a lease after successful terminal scatter completion.

        :param key: Request and chunk identity.
        :raises PackedProtocolError: If no successful scatter is in flight.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state is not PackedProtocolState.SCATTERING:
                raise PackedProtocolError(
                    key,
                    f"scatter cannot complete in state {chunk.state.value}",
                )
            chunk.scatter_terminal = True
            self._release_lease_locked(chunk)
            chunk.state = PackedProtocolState.RELEASED

    def fail_chunk(self, key: PackedChunkKey, reason: str) -> None:
        """Fail a chunk while preserving every possible asynchronous owner.

        Failure before READY needs no quarantine. Failure after READY prohibits
        fallback and quarantines the lease until every possible writer DMA is
        terminal by authenticated outcome or explicitly quiesced. If scatter
        has begun, it must also become terminal.

        :param key: Request and chunk identity.
        :param reason: Reader-facing failure reason.
        """

        if len(reason) == 0:
            raise ValueError("failure reason must not be empty")
        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state in (
                PackedProtocolState.RELEASED,
                PackedProtocolState.FAILED_RELEASED,
            ):
                return
            self._fail_locked(chunk, reason)

    def fail_scatter(self, key: PackedChunkKey, reason: str) -> None:
        """Fail begun scatter work without claiming that it is quiescent.

        :param key: Request and chunk identity.
        :param reason: Reader-facing scatter failure reason.
        :raises PackedProtocolError: If scatter ownership was never handed out.
        """

        if len(reason) == 0:
            raise ValueError("failure reason must not be empty")
        with self._lock:
            chunk = self._require_chunk_locked(key)
            if not chunk.scatter_started:
                raise PackedProtocolError(key, "scatter has not begun")
            if chunk.state in (
                PackedProtocolState.RELEASED,
                PackedProtocolState.FAILED_RELEASED,
            ):
                raise PackedProtocolError(key, "scatter lease is already released")
            self._fail_locked(chunk, reason)

    def quiesce_writer(
        self,
        key: PackedChunkKey,
        writer_id: StagingWriterId,
    ) -> None:
        """Record trusted out-of-band proof that a writer cannot DMA.

        This is an internal transport-lifecycle assertion, not a peer wire
        message. It is legal only after the chunk has already failed.

        :param key: Request and chunk identity.
        :param writer_id: Canonical writer proven terminal without a wire outcome.
        :raises PackedProtocolError: If the chunk is not quarantined.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state is not PackedProtocolState.FAILED_QUARANTINED:
                raise PackedProtocolError(
                    key,
                    f"writer quiescence is invalid in state {chunk.state.value}",
                )
            if writer_id not in chunk.expected_writers:
                raise PackedProtocolError(key, f"unexpected writer {writer_id}")
            chunk.quiesced_writers.add(writer_id)
            self._release_failed_if_safe_locked(chunk)

    def quiesce_scatter(self, key: PackedChunkKey) -> None:
        """Record trusted terminal completion of failed scatter work.

        :param key: Request and chunk identity.
        :raises PackedProtocolError: If failed scatter work is not outstanding.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state is not PackedProtocolState.FAILED_QUARANTINED:
                raise PackedProtocolError(
                    key,
                    f"scatter quiescence is invalid in state {chunk.state.value}",
                )
            if not chunk.scatter_started:
                raise PackedProtocolError(key, "scatter has not begun")
            if chunk.scatter_terminal:
                return
            chunk.scatter_terminal = True
            self._release_failed_if_safe_locked(chunk)

    def snapshot(self, key: PackedChunkKey) -> PackedChunkSnapshot:
        """Return a stable diagnostic view of one registered chunk.

        :param key: Request and chunk identity.
        :returns: Immutable state snapshot.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            lease_id = chunk.lease.lease_id if chunk.lease is not None else None
            return PackedChunkSnapshot(
                key=key,
                state=chunk.state,
                prepared_writers=tuple(sorted(chunk.prepares)),
                writer_outcomes=tuple(
                    chunk.writer_outcomes[writer_id]
                    for writer_id in sorted(chunk.writer_outcomes)
                ),
                quiesced_writers=tuple(sorted(chunk.quiesced_writers)),
                lease_id=lease_id,
                ready_issued=chunk.ready_issued,
                scatter_started=chunk.scatter_started,
                scatter_terminal=chunk.scatter_terminal,
                failure_reason=chunk.failure_reason,
            )

    def retire_chunk(self, key: PackedChunkKey) -> None:
        """Forget terminal metadata and immutable destination-page snapshots.

        Retirement is forbidden while a writer DMA or scatter may still own the
        lease. Adapters must drive quarantine to a terminal state before
        retiring a failed room.

        :param key: Request and chunk identity.
        :raises PackedProtocolError: If asynchronous ownership is not terminal.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state not in (
                PackedProtocolState.RELEASED,
                PackedProtocolState.FAILED_RELEASED,
            ):
                raise PackedProtocolError(
                    key,
                    f"chunk cannot retire in state {chunk.state.value}",
                )
            del self._chunks[key]

    def cancel_unpublished_chunk(self, key: PackedChunkKey) -> None:
        """Forget one chunk that was never exposed to a source writer.

        This path exists only for request preparation rollback. Once a PREPARE
        has been accepted, external ownership may exist and normal failure plus
        quiescence must retire the chunk instead.

        :param key: Exact request and chunk identity.
        :raises PackedProtocolError: If the chunk may have an external owner.
        """

        with self._lock:
            chunk = self._require_chunk_locked(key)
            if chunk.state is not PackedProtocolState.COLLECTING:
                raise PackedProtocolError(
                    key,
                    f"unpublished cancellation is invalid in state {chunk.state.value}",
                )
            if (
                len(chunk.prepares) != 0
                or chunk.ready_issued
                or chunk.lease is not None
            ):
                raise PackedProtocolError(
                    key,
                    "unpublished cancellation found source-visible ownership",
                )
            del self._chunks[key]

    def _require_chunk_locked(self, key: PackedChunkKey) -> _PackedChunk:
        """Return one registered chunk while the protocol lock is held.

        :param key: Request and chunk identity.
        :returns: Mutable protocol-owned chunk.
        :raises PackedProtocolError: If the chunk was never registered.
        """

        self._require_no_allocator_callback_locked()
        try:
            return self._chunks[key]
        except KeyError as error:
            raise PackedProtocolError(key, "chunk is not registered") from error

    def _require_live_locked(self, chunk: _PackedChunk) -> None:
        """Reject PREPARE after a chunk enters a failure or terminal state.

        :param chunk: Protocol-owned chunk.
        :raises PackedProtocolError: If PREPARE cannot be accepted.
        """

        if chunk.state in (
            PackedProtocolState.FAILED_QUARANTINED,
            PackedProtocolState.FAILED_RELEASED,
            PackedProtocolState.RELEASED,
        ):
            raise PackedProtocolError(
                chunk.key,
                f"PREPARE received in terminal state {chunk.state.value}",
            )

    def _validate_prepare_locked(
        self,
        chunk: _PackedChunk,
        message: PackedPrepare,
    ) -> None:
        """Validate untrusted PREPARE data against decode-local truth.

        :param chunk: Protocol-owned expected chunk.
        :param message: Untrusted PREPARE payload.
        :raises PackedProtocolError: If any identity or layout field conflicts.
        """

        if message.writer_id not in chunk.expected_writers:
            self._reject_locked(
                chunk,
                f"PREPARE names unexpected writer {message.writer_id}",
            )
        if message.spec.topology != chunk.spec.topology:
            self._reject_locked(chunk, "PREPARE topology differs from decode layout")
        if message.spec.chunk_id != message.key.chunk_id:
            self._reject_locked(chunk, "PREPARE chunk identity differs from its spec")
        try:
            received_layout = message.spec.build()
        except ValueError as error:
            self._reject_locked(chunk, f"PREPARE layout is invalid: {error}")
        if message.digest != received_layout.digest:
            self._reject_locked(
                chunk,
                "PREPARE digest does not match its rebuilt layout",
            )
        if received_layout != chunk.layout:
            self._reject_locked(
                chunk,
                "PREPARE layout differs from decode-local canonical layout",
            )

    def _validate_writer_outcome_locked(
        self,
        chunk: _PackedChunk,
        message: PackedWriterOutcome,
    ) -> None:
        """Validate one terminal outcome against the issued READY lease.

        :param chunk: Protocol-owned chunk.
        :param message: Untrusted terminal outcome payload.
        :raises PackedProtocolError: If the outcome could not follow issued READY.
        """

        if message.writer_id not in chunk.expected_writers:
            self._reject_locked(
                chunk,
                f"writer outcome names unexpected writer {message.writer_id}",
            )
        lease = chunk.lease
        if lease is None or not chunk.ready_issued:
            self._reject_locked(chunk, "writer outcome arrived before READY")
        if message.digest != chunk.layout.digest:
            self._reject_locked(chunk, "writer outcome digest differs from READY")
        if message.lease_id != lease.lease_id:
            self._reject_locked(chunk, "writer outcome lease differs from READY")
        visibility = message.visibility
        if visibility is None:
            return
        expected_policy_digest = chunk.writer_visibility_policy_digests[
            message.writer_id
        ]
        if visibility.policy_digest != expected_policy_digest:
            self._reject_locked(
                chunk,
                "writer outcome visibility policy differs from READY",
            )

    def _preflight_writer_outcome_locked(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[_PackedChunk, PackedWriterOutcome | None]:
        """Validate exact writer metadata without admitting a new outcome.

        :param message: Untrusted terminal outcome payload.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Protocol chunk and an identical previously admitted outcome.
        """

        chunk = self._require_chunk_locked(message.key)
        if chunk.state in (
            PackedProtocolState.RELEASED,
            PackedProtocolState.FAILED_RELEASED,
        ):
            raise PackedProtocolError(
                chunk.key,
                f"writer outcome received after terminal state {chunk.state.value}",
            )
        if authenticated_writer_id != message.writer_id:
            self._reject_locked(
                chunk,
                "writer outcome identity does not match authenticated peer",
            )
        self._validate_writer_outcome_locked(chunk, message)
        previous = chunk.writer_outcomes.get(authenticated_writer_id)
        if previous is not None and previous != message:
            self._reject_locked(
                chunk,
                f"conflicting duplicate writer outcome from {authenticated_writer_id}",
            )
        if previous is None and chunk.state not in (
            PackedProtocolState.READY,
            PackedProtocolState.FAILED_QUARANTINED,
        ):
            self._reject_locked(
                chunk,
                f"new writer outcome is invalid in state {chunk.state.value}",
            )
        return chunk, previous

    def _consume_writer_outcome_ticket_locked(
        self,
        chunk: _PackedChunk,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
        ticket: _PackedWriterOutcomeTicket | None,
    ) -> None:
        """Consume one exact protocol-bound destination visibility ticket.

        :param chunk: Protocol-owned chunk.
        :param message: Successful outcome being admitted.
        :param authenticated_writer_id: Writer authenticated by the transport.
        :param ticket: Candidate single-use admission.
        """

        if type(ticket) is not _PackedWriterOutcomeTicket:
            self._reject_locked(
                chunk,
                "DONE writer outcome requires a destination visibility ticket",
            )
        if ticket._protocol_nonce is not self._writer_outcome_ticket_nonce:
            self._reject_locked(
                chunk,
                "writer outcome ticket belongs to another protocol",
            )
        if ticket.message != message:
            self._reject_locked(
                chunk,
                "writer outcome ticket belongs to another message",
            )
        if ticket.authenticated_writer_id != authenticated_writer_id:
            self._reject_locked(
                chunk,
                "writer outcome ticket belongs to another authenticated writer",
            )
        if ticket._ticket_identity in chunk.consumed_writer_outcome_tickets:
            self._reject_locked(
                chunk,
                "writer outcome ticket was already consumed",
            )
        chunk.consumed_writer_outcome_tickets.add(ticket._ticket_identity)

    def _allocate_and_ready_locked(
        self,
        chunk: _PackedChunk,
    ) -> tuple[PackedReady, ...]:
        """Allocate exactly one lease and construct every writer projection.

        :param chunk: Fully prepared protocol-owned chunk.
        :returns: READY messages in canonical writer order.
        :raises PackedProtocolError: If allocation fails or is undersized.
        """

        if chunk.lease is not None or chunk.ready_issued:
            raise PackedProtocolError(chunk.key, "packed lease was already allocated")
        try:
            lease = self._allocate_lease_locked(chunk.layout.total_bytes)
        except Exception as error:
            logger.error(
                "Packed staging lease allocation failed:\n%s",
                traceback.format_exc(),
            )
            chunk.failure_reason = f"lease allocation failed: {error}"
            chunk.state = PackedProtocolState.FAILED_RELEASED
            raise PackedProtocolError(chunk.key, chunk.failure_reason) from error
        chunk.lease = lease
        if lease.length_bytes < chunk.layout.total_bytes:
            chunk.failure_reason = (
                "allocator returned undersized lease: "
                f"{lease.length_bytes} < {chunk.layout.total_bytes}"
            )
            chunk.state = PackedProtocolState.FAILED_QUARANTINED
            chunk.quarantined = True
            chunk.quiesced_writers.update(chunk.expected_writers)
            try:
                self._release_failed_if_safe_locked(chunk)
            except Exception as error:
                logger.error(
                    "Undersized packed staging lease release failed:\n%s",
                    traceback.format_exc(),
                )
                raise PackedProtocolError(
                    chunk.key,
                    f"{chunk.failure_reason}; release failed: {error}",
                ) from error
            raise PackedProtocolError(chunk.key, chunk.failure_reason)

        ready_messages: dict[StagingWriterId, PackedReady] = {}
        for writer_id in chunk.expected_writers:
            projection = _writer_layout(chunk.layout, writer_id)
            ready_messages[writer_id] = PackedReady(
                key=chunk.key,
                writer_id=writer_id,
                digest=chunk.layout.digest,
                visibility_policy_digest=chunk.writer_visibility_policy_digests[
                    writer_id
                ],
                lease_id=lease.lease_id,
                lease_base_address=lease.base_address,
                projection_offset=projection.lease_offset,
                projection_length=projection.length_bytes,
            )
        chunk.ready_issued = True
        chunk.state = PackedProtocolState.READY
        return tuple(ready_messages[writer_id] for writer_id in chunk.expected_writers)

    def _reject_locked(self, chunk: _PackedChunk, reason: str) -> Never:
        """Fail a chunk and raise its protocol violation.

        :param chunk: Protocol-owned chunk.
        :param reason: Stable failure reason.
        :raises PackedProtocolError: Always.
        """

        self._fail_locked(chunk, reason)
        raise PackedProtocolError(chunk.key, reason)

    def _fail_locked(self, chunk: _PackedChunk, reason: str) -> None:
        """Enter the correct failure state without violating async ownership.

        :param chunk: Protocol-owned chunk.
        :param reason: Stable failure reason.
        """

        if chunk.failure_reason is None:
            chunk.failure_reason = reason
        lease = chunk.lease
        if lease is None:
            chunk.state = PackedProtocolState.FAILED_RELEASED
            return
        chunk.state = PackedProtocolState.FAILED_QUARANTINED
        if not chunk.quarantined:
            self._quarantine_lease_locked(lease, chunk.failure_reason)
            chunk.quarantined = True
        self._release_failed_if_safe_locked(chunk)

    def _release_failed_if_safe_locked(self, chunk: _PackedChunk) -> None:
        """Release a quarantined lease only after every async owner is terminal.

        :param chunk: Failed protocol-owned chunk.
        """

        if chunk.state is not PackedProtocolState.FAILED_QUARANTINED:
            return
        terminal_writers = set(chunk.writer_outcomes) | chunk.quiesced_writers
        if terminal_writers != set(chunk.expected_writers):
            return
        if chunk.scatter_started and not chunk.scatter_terminal:
            return
        self._release_lease_locked(chunk)
        chunk.state = PackedProtocolState.FAILED_RELEASED

    def _release_lease_locked(self, chunk: _PackedChunk) -> None:
        """Release one allocated lease exactly once.

        :param chunk: Protocol-owned chunk.
        :raises PackedProtocolError: If the chunk has no allocated lease.
        """

        lease = chunk.lease
        if lease is None:
            raise PackedProtocolError(chunk.key, "chunk has no lease to release")
        self._release_allocator_lease_locked(lease)

    def _require_no_allocator_callback_locked(self) -> None:
        """Reject allocator reentry before it can mutate protocol state.

        :raises RuntimeError: If an allocator callback reenters the protocol.
        """

        if self._allocator_callback_active:
            raise RuntimeError("packed lease allocator must not reenter the protocol")

    def _allocate_lease_locked(self, length_bytes: int) -> PackedLease:
        """Invoke the non-reentrant allocator allocation callback.

        :param length_bytes: Minimum contiguous lease capacity.
        :returns: Allocated lease.
        """

        self._allocator_callback_active = True
        try:
            return self._allocator.allocate(length_bytes)
        finally:
            self._allocator_callback_active = False

    def _quarantine_lease_locked(self, lease: PackedLease, reason: str) -> None:
        """Invoke the non-reentrant allocator quarantine callback.

        :param lease: Failed lease that may retain asynchronous owners.
        :param reason: First failure reason.
        """

        self._allocator_callback_active = True
        try:
            self._allocator.quarantine(lease, reason)
        finally:
            self._allocator_callback_active = False

    def _release_allocator_lease_locked(self, lease: PackedLease) -> None:
        """Invoke the non-reentrant allocator release callback.

        :param lease: Terminally quiescent lease.
        """

        self._allocator_callback_active = True
        try:
            self._allocator.release(lease)
        finally:
            self._allocator_callback_active = False
