import dataclasses
import enum
import logging
import threading
import traceback
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

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, order=True)
class PackedChunkKey:
    """Stable identity of one request-local packed transfer chunk.

    :ivar room_id: Bootstrap room identifying the request.
    :ivar chunk_id: Request-local monotonically increasing chunk identifier.
    """

    room_id: int
    chunk_id: int

    def __post_init__(self) -> None:
        """Validate the packed chunk identity."""

        if type(self.room_id) is not int or self.room_id < 0:
            raise ValueError(
                f"room_id must be a non-negative integer, got {self.room_id!r}"
            )
        if type(self.chunk_id) is not int or self.chunk_id < 0:
            raise ValueError(
                f"chunk_id must be a non-negative integer, got {self.chunk_id!r}"
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

        if self.source_tp_size <= 0:
            raise ValueError(
                f"source_tp_size must be positive, got {self.source_tp_size}"
            )
        if self.destination_tp_size <= 0:
            raise ValueError(
                "destination_tp_size must be positive, got "
                f"{self.destination_tp_size}"
            )
        if (
            self.destination_tp_rank < 0
            or self.destination_tp_rank >= self.destination_tp_size
        ):
            raise ValueError(
                "destination_tp_rank must be in "
                f"[0, {self.destination_tp_size}), got "
                f"{self.destination_tp_rank}"
            )
        if self.alignment_bytes <= 0:
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

        object.__setattr__(self, "digest", bytes(self.digest))
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
    :ivar lease_id: Decode allocation identity.
    :ivar lease_base_address: Base address of the one contiguous decode lease.
    :ivar projection_offset: Writer-local offset into the lease.
    :ivar projection_length: Exact writer projection length.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    digest: bytes
    lease_id: int
    lease_base_address: int
    projection_offset: int
    projection_length: int

    def __post_init__(self) -> None:
        """Own and validate READY lease metadata."""

        object.__setattr__(self, "digest", bytes(self.digest))
        if len(self.digest) != 32:
            raise ValueError(
                f"READY digest must contain 32 bytes, got {len(self.digest)}"
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


@dataclasses.dataclass(frozen=True)
class PackedCommit:
    """Writer proof that its registered contiguous transfer completed.

    A COMMIT is valid only after the transport reports terminal completion for
    that writer's one registered DMA. Enqueue or submission alone is not
    completion.

    :ivar key: Request and chunk identity.
    :ivar writer_id: Claimed writer identity.
    :ivar digest: Canonical layout digest.
    :ivar lease_id: Decode allocation identity received in READY.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    digest: bytes
    lease_id: int

    def __post_init__(self) -> None:
        """Own and validate COMMIT identity metadata."""

        object.__setattr__(self, "digest", bytes(self.digest))
        if len(self.digest) != 32:
            raise ValueError(
                f"COMMIT digest must contain 32 bytes, got {len(self.digest)}"
            )
        if type(self.lease_id) is not int or self.lease_id < 0:
            raise ValueError(
                f"COMMIT lease_id must be non-negative, got {self.lease_id!r}"
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
    """Allocator contract required by the decode protocol."""

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
    :ivar committed_writers: Canonically ordered terminal DMA writers.
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
    committed_writers: tuple[StagingWriterId, ...]
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
    state: PackedProtocolState = PackedProtocolState.COLLECTING
    prepares: dict[StagingWriterId, PackedPrepare] = dataclasses.field(
        default_factory=dict
    )
    commits: dict[StagingWriterId, PackedCommit] = dataclasses.field(
        default_factory=dict
    )
    quiesced_writers: set[StagingWriterId] = dataclasses.field(default_factory=set)
    lease: PackedLease | None = None
    ready_messages: dict[StagingWriterId, PackedReady] = dataclasses.field(
        default_factory=dict
    )
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
    _chunks: dict[PackedChunkKey, _PackedChunk]
    _lock: threading.RLock

    def __init__(self, allocator: PackedLeaseAllocator) -> None:
        """Initialize an empty decode protocol.

        :param allocator: Contiguous registered staging lease allocator.
        """

        self._allocator = allocator
        self._chunks = {}
        self._lock = threading.RLock()

    def register_chunk(
        self,
        key: PackedChunkKey,
        spec: PackedLayoutSpec,
        destination_registry: StagingComponentBufferRegistry,
    ) -> StagingChunkLayout:
        """Register trusted decode-local geometry and destination bounds.

        The canonical layout is rebuilt locally and all destination page arrays
        are copied into immutable snapshots before any PREPARE can reach
        consensus.

        :param key: Request and chunk identity.
        :param spec: Trusted layout input assembled from bootstrap metadata.
        :param destination_registry: Decode-local registered buffers and pages.
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
        with self._lock:
            if key in self._chunks:
                raise ValueError(f"packed chunk is already registered: {key}")
            self._chunks[key] = _PackedChunk(
                key=key,
                spec=spec,
                layout=layout,
                destination_binding=destination_binding,
                expected_writers=expected_writers,
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
        :returns: READY messages produced or replayed by this PREPARE.
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
                ready = chunk.ready_messages.get(authenticated_writer_id)
                if ready is None:
                    return ()
                return (ready,)

            chunk.prepares[authenticated_writer_id] = message
            if set(chunk.prepares) != set(chunk.expected_writers):
                return ()
            return self._allocate_and_ready_locked(chunk)

    def handle_commit(
        self,
        message: PackedCommit,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Record terminal completion of one writer's registered transfer.

        COMMIT must be sent and accepted only after the writer's NIXL transfer
        reports completion, never merely after enqueue. Identical duplicates are
        idempotent and cannot advance consensus.

        :param message: Untrusted COMMIT payload.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this COMMIT newly made scatter eligible.
        :raises PackedProtocolError: If identity or lease metadata conflicts.
        """

        with self._lock:
            chunk = self._require_chunk_locked(message.key)
            if chunk.state in (
                PackedProtocolState.RELEASED,
                PackedProtocolState.FAILED_RELEASED,
            ):
                raise PackedProtocolError(
                    chunk.key,
                    f"COMMIT received after terminal state {chunk.state.value}",
                )
            if authenticated_writer_id != message.writer_id:
                self._reject_locked(
                    chunk,
                    "COMMIT writer identity does not match authenticated peer",
                )
            self._validate_commit_locked(chunk, message)

            previous = chunk.commits.get(authenticated_writer_id)
            if previous is not None:
                if previous != message:
                    self._reject_locked(
                        chunk,
                        f"conflicting duplicate COMMIT from {authenticated_writer_id}",
                    )
                return False

            chunk.commits[authenticated_writer_id] = message
            if chunk.state is PackedProtocolState.FAILED_QUARANTINED:
                self._release_failed_if_safe_locked(chunk)
                return False
            if set(chunk.commits) != set(chunk.expected_writers):
                return False
            if chunk.state is not PackedProtocolState.READY:
                self._reject_locked(
                    chunk,
                    f"complete COMMIT set is invalid in state {chunk.state.value}",
                )
            chunk.state = PackedProtocolState.SCATTER_READY
            return True

    def begin_scatter(self, key: PackedChunkKey) -> PackedScatterWork:
        """Transfer lease ownership to one asynchronous scatter operation.

        Calling this method establishes that scatter may be in flight. Any later
        failure must be followed by :meth:`quiesce_scatter` before the lease can
        be released.

        :param key: Request and chunk identity.
        :returns: Immutable scatter inputs.
        :raises PackedProtocolError: If writer COMMIT consensus is incomplete.
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
        committed or explicitly quiesced. If scatter has begun, it must also
        become terminal.

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
        :param writer_id: Canonical writer proven terminal without COMMIT.
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
                committed_writers=tuple(sorted(chunk.commits)),
                quiesced_writers=tuple(sorted(chunk.quiesced_writers)),
                lease_id=lease_id,
                ready_issued=chunk.ready_issued,
                scatter_started=chunk.scatter_started,
                scatter_terminal=chunk.scatter_terminal,
                failure_reason=chunk.failure_reason,
            )

    def _require_chunk_locked(self, key: PackedChunkKey) -> _PackedChunk:
        """Return one registered chunk while the protocol lock is held.

        :param key: Request and chunk identity.
        :returns: Mutable protocol-owned chunk.
        :raises PackedProtocolError: If the chunk was never registered.
        """

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

    def _validate_commit_locked(
        self,
        chunk: _PackedChunk,
        message: PackedCommit,
    ) -> None:
        """Validate one COMMIT against the issued READY lease.

        :param chunk: Protocol-owned chunk.
        :param message: Untrusted COMMIT payload.
        :raises PackedProtocolError: If COMMIT could not follow issued READY.
        """

        if message.writer_id not in chunk.expected_writers:
            self._reject_locked(
                chunk,
                f"COMMIT names unexpected writer {message.writer_id}",
            )
        lease = chunk.lease
        if lease is None or not chunk.ready_issued:
            self._reject_locked(chunk, "COMMIT arrived before READY")
        if message.digest != chunk.layout.digest:
            self._reject_locked(chunk, "COMMIT digest differs from READY")
        if message.lease_id != lease.lease_id:
            self._reject_locked(chunk, "COMMIT lease differs from READY")

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
            lease = self._allocator.allocate(chunk.layout.total_bytes)
        except Exception as error:
            logger.error(
                "Packed staging lease allocation failed:\n%s",
                traceback.format_exc(),
            )
            chunk.failure_reason = f"lease allocation failed: {error}"
            chunk.state = PackedProtocolState.FAILED_RELEASED
            raise PackedProtocolError(chunk.key, chunk.failure_reason) from error
        if lease.length_bytes < chunk.layout.total_bytes:
            chunk.failure_reason = (
                "allocator returned undersized lease: "
                f"{lease.length_bytes} < {chunk.layout.total_bytes}"
            )
            chunk.state = PackedProtocolState.FAILED_RELEASED
            try:
                self._allocator.release(lease)
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

        chunk.lease = lease
        ready_messages: dict[StagingWriterId, PackedReady] = {}
        for writer_id in chunk.expected_writers:
            projection = _writer_layout(chunk.layout, writer_id)
            ready_messages[writer_id] = PackedReady(
                key=chunk.key,
                writer_id=writer_id,
                digest=chunk.layout.digest,
                lease_id=lease.lease_id,
                lease_base_address=lease.base_address,
                projection_offset=projection.lease_offset,
                projection_length=projection.length_bytes,
            )
        chunk.ready_messages = ready_messages
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
        if not chunk.ready_issued:
            chunk.state = PackedProtocolState.FAILED_RELEASED
            return
        lease = chunk.lease
        if lease is None:
            chunk.state = PackedProtocolState.FAILED_RELEASED
            return
        chunk.state = PackedProtocolState.FAILED_QUARANTINED
        if not chunk.quarantined:
            self._allocator.quarantine(lease, chunk.failure_reason)
            chunk.quarantined = True
        self._release_failed_if_safe_locked(chunk)

    def _release_failed_if_safe_locked(self, chunk: _PackedChunk) -> None:
        """Release a quarantined lease only after every async owner is terminal.

        :param chunk: Failed protocol-owned chunk.
        """

        if chunk.state is not PackedProtocolState.FAILED_QUARANTINED:
            return
        terminal_writers = set(chunk.commits) | chunk.quiesced_writers
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
        self._allocator.release(lease)
