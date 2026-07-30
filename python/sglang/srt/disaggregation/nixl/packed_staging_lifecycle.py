import dataclasses
import enum
import hashlib
import hmac
import logging
import secrets
import threading
import traceback
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
    PackedChunkKey,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingWriterId,
)

PACKED_AUTHENTICATOR_BYTES = 32
PACKED_CAPABILITY_ID_BYTES = 16
PACKED_GENERATION_BYTES = 16
PACKED_ROUTE_DIGEST_BYTES = 32
PACKED_RUNTIME_IDENTITY_BYTES = 32

logger = logging.getLogger(__name__)

_AUTHORITY_SEAL = object()
_OPAQUE_HANDLE_SEAL = object()
_RECEIPT_SEAL = object()


class PackedLifecycleError(RuntimeError):
    """Packed staging lifecycle invariant violation."""


class PackedNativeAttestationUnavailable(PackedLifecycleError):
    """The native runtime cannot attest the exact submitted transport path."""


class PackedAmbiguousTransportError(PackedLifecycleError):
    """Native submission or progress lost exact-handle terminality."""


class PackedProcessQuarantinedError(PackedLifecycleError):
    """The complete resource cohort is retained for the process lifetime."""


class PackedCapabilityState(enum.StrEnum):
    """Lifecycle state of one authenticated arena capability."""

    ACTIVE = "active"
    TEARING_DOWN = "tearing_down"
    REVOKED = "revoked"
    QUARANTINED = "quarantined"


class PackedPageLeaseRole(enum.StrEnum):
    """Allocator ownership role for request-local KV pages."""

    SOURCE = "source"
    DESTINATION = "destination"


class PackedNativePostState(enum.StrEnum):
    """Trusted native result from posting one exact prepared handle."""

    SUBMITTED = "submitted"
    COMPLETED = "completed"
    ERROR = "error"


class PackedNativePollState(enum.StrEnum):
    """Trusted native progress result for one exact submitted handle."""

    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class PackedScatterPollState(enum.StrEnum):
    """Trusted CUDA event progress for one exact scatter submission."""

    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class PackedNixlBackend(enum.StrEnum):
    """NIXL backend accepted by the packed B300 transport."""

    UCX = "UCX"


class PackedUcpTransport(enum.StrEnum):
    """UCP transports admitted for same-host packed GPU transfers."""

    CUDA_IPC = "cuda_ipc"
    CUDA = "cuda"


class PackedNativeCompletionContract(enum.StrEnum):
    """Native completion chain required before a writer can report DONE."""

    NIXL_UCX_ENDPOINT_FLUSH = "nixl_ucx_endpoint_flush"


class PackedNativeTransferState(enum.StrEnum):
    """Ownership state of one exact native transfer handle."""

    PREPARED = "prepared"
    POSTING = "posting"
    SUBMITTED = "submitted"
    IN_FLIGHT = "in_flight"
    COMPLETION_REPORTED = "completion_reported"
    COMPLETED = "completed"
    RELEASED = "released"
    QUARANTINED = "quarantined"


class PackedScatterState(enum.StrEnum):
    """Ownership state of one exact destination scatter event."""

    SUBMISSION_REPORTED = "submission_reported"
    IN_FLIGHT = "in_flight"
    COMPLETION_REPORTED = "completion_reported"
    COMPLETED = "completed"
    RELEASED = "released"
    QUARANTINED = "quarantined"


class PackedTeardownState(enum.StrEnum):
    """Explicit request/ack teardown-barrier state."""

    CREATED = "created"
    WAITING_FOR_ACKS = "waiting_for_acks"
    BARRIER_ISSUED = "barrier_issued"
    RELEASED = "released"
    QUARANTINED = "quarantined"


def _require_exact_bytes(value: bytes, length: int, label: str) -> None:
    """Validate one fixed-width byte identity.

    :param value: Candidate identity.
    :param length: Required byte count.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} bytes, got {len(value)}")


def _digest_fields(domain: bytes, fields: tuple[bytes, ...]) -> bytes:
    """Hash length-delimited fields under one protocol domain.

    :param domain: Protocol-specific digest domain.
    :param fields: Ordered byte fields.
    :returns: SHA-256 digest.
    """

    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for field in fields:
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)
    return digest.digest()


def _log_ambiguous_failure(context: str) -> None:
    """Log the active exception before fail-closed lifecycle handling.

    :param context: Stable operation context.
    """

    logger.error("%s\n%s", context, traceback.format_exc())


def _contains_exact(values: tuple[object, ...], target: object) -> bool:
    """Return whether a sequence contains an exact object.

    :param values: Candidate object sequence.
    :param target: Exact object to locate.
    :returns: Whether identity, rather than equality, matched.
    """

    return any(value is target for value in values)


def _component_sort_key(
    component_id: StagingComponentId,
) -> tuple[int, int, str]:
    """Return a canonical main-KV-first component key.

    :param component_id: Page component identity.
    :returns: Stable sort key.
    """

    if type(component_id) is not StagingComponentId:
        raise TypeError("page lease key must be StagingComponentId")
    if component_id.state_index is None:
        if component_id.state_type is not None:
            raise ValueError("main KV component cannot declare a state type")
        return (0, -1, "")
    if component_id.state_index < 0:
        raise ValueError("state component index must be non-negative")
    if component_id.state_type is None:
        raise ValueError("state component must declare a state type")
    return (
        1,
        component_id.state_index,
        component_id.state_type.value,
    )


def _validate_writer_id(writer_id: StagingWriterId) -> None:
    """Validate one untrusted canonical writer identity.

    :param writer_id: Candidate writer identity.
    """

    if type(writer_id) is not StagingWriterId:
        raise TypeError("writer_id must be StagingWriterId")
    for field_name, value in (
        ("transfer_source_rank", writer_id.transfer_source_rank),
        ("source_attn_tp_rank", writer_id.source_attn_tp_rank),
        ("source_pp_rank", writer_id.source_pp_rank),
        ("source_cp_rank", writer_id.source_cp_rank),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"writer {field_name} must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class PackedWriterProjection:
    """One authenticated writer and its non-overlapping destination projection.

    :ivar writer_id: Canonical source writer identity.
    :ivar peer_generation: Authenticated transport-session generation.
    :ivar destination_offset: Byte offset into the destination staging lease.
    :ivar length_bytes: Exact writer transfer length.
    """

    writer_id: StagingWriterId
    peer_generation: bytes
    destination_offset: int
    length_bytes: int

    def __post_init__(self) -> None:
        """Validate one manifest projection."""

        _validate_writer_id(self.writer_id)
        _require_exact_bytes(
            self.peer_generation,
            PACKED_GENERATION_BYTES,
            "peer_generation",
        )
        if type(self.destination_offset) is not int or self.destination_offset < 0:
            raise ValueError("destination_offset must be a non-negative integer")
        if type(self.length_bytes) is not int or self.length_bytes <= 0:
            raise ValueError("length_bytes must be a positive integer")


@dataclasses.dataclass(frozen=True)
class PackedWriterCohortManifest:
    """Request-level authority for an ordered asymmetric-TP writer cohort.

    :ivar request_generation: Request generation preventing room replay.
    :ivar source_tp_size: Source attention tensor-parallel width.
    :ivar destination_tp_size: Destination attention tensor-parallel width.
    :ivar destination_tp_rank: Destination attention tensor-parallel rank.
    :ivar total_bytes: Complete destination staging lease length.
    :ivar alignment_bytes: Required projection alignment.
    :ivar projections: Canonically ordered authenticated writer projections.
    """

    request_generation: bytes
    source_tp_size: int
    destination_tp_size: int
    destination_tp_rank: int
    total_bytes: int
    alignment_bytes: int
    projections: tuple[PackedWriterProjection, ...]

    def __post_init__(self) -> None:
        """Own and validate the complete writer topology."""

        _require_exact_bytes(
            self.request_generation,
            PACKED_REQUEST_GENERATION_BYTES,
            "request_generation",
        )
        if type(self.source_tp_size) is not int or self.source_tp_size <= 0:
            raise ValueError("source_tp_size must be a positive integer")
        if type(self.destination_tp_size) is not int or self.destination_tp_size <= 0:
            raise ValueError("destination_tp_size must be a positive integer")
        if (
            type(self.destination_tp_rank) is not int
            or self.destination_tp_rank < 0
            or self.destination_tp_rank >= self.destination_tp_size
        ):
            raise ValueError("destination_tp_rank is outside destination topology")
        if type(self.total_bytes) is not int or self.total_bytes <= 0:
            raise ValueError("total_bytes must be a positive integer")
        if type(self.alignment_bytes) is not int or self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be a positive integer")

        projections = tuple(self.projections)
        object.__setattr__(self, "projections", projections)
        if len(projections) != self.source_tp_size:
            raise ValueError(
                "writer count must equal source_tp_size: "
                f"{len(projections)} != {self.source_tp_size}"
            )
        writer_ids = tuple(projection.writer_id for projection in projections)
        if len(set(writer_ids)) != len(writer_ids):
            raise ValueError("writer cohort contains duplicate identities")
        if writer_ids != tuple(sorted(writer_ids)):
            raise ValueError("writer cohort must use canonical writer ordering")
        source_tp_ranks = tuple(
            projection.writer_id.source_attn_tp_rank for projection in projections
        )
        if source_tp_ranks != tuple(range(self.source_tp_size)):
            raise ValueError(
                "writer cohort must contain each source attention TP rank exactly once"
            )

        intervals: list[tuple[int, int, StagingWriterId]] = []
        for projection in projections:
            if projection.destination_offset % self.alignment_bytes != 0:
                raise ValueError("writer projection offset is not aligned")
            if projection.length_bytes % self.alignment_bytes != 0:
                raise ValueError("writer projection length is not aligned")
            end = projection.destination_offset + projection.length_bytes
            if end > self.total_bytes:
                raise ValueError("writer projection exceeds destination lease")
            intervals.append((projection.destination_offset, end, projection.writer_id))
        intervals.sort()
        if intervals[0][0] != 0:
            raise ValueError("writer projections do not begin at lease offset zero")
        for previous, current in pairwise(intervals):
            if previous[1] > current[0]:
                raise ValueError(
                    "writer destination projections overlap: "
                    f"{previous[2]} and {current[2]}"
                )
            if previous[1] != current[0]:
                raise ValueError("writer destination projections contain a gap")
        if intervals[-1][1] != self.total_bytes:
            raise ValueError("writer projections do not cover the destination lease")

    @property
    def digest(self) -> bytes:
        """Return the authenticated canonical manifest digest.

        :returns: SHA-256 manifest digest.
        """

        fields: list[bytes] = [
            self.request_generation,
            self.source_tp_size.to_bytes(4, "big"),
            self.destination_tp_size.to_bytes(4, "big"),
            self.destination_tp_rank.to_bytes(4, "big"),
            self.total_bytes.to_bytes(8, "big"),
            self.alignment_bytes.to_bytes(8, "big"),
        ]
        for projection in self.projections:
            writer_id = projection.writer_id
            fields.extend(
                (
                    writer_id.transfer_source_rank.to_bytes(8, "big"),
                    writer_id.source_attn_tp_rank.to_bytes(4, "big"),
                    writer_id.source_pp_rank.to_bytes(4, "big"),
                    writer_id.source_cp_rank.to_bytes(4, "big"),
                    projection.peer_generation,
                    projection.destination_offset.to_bytes(8, "big"),
                    projection.length_bytes.to_bytes(8, "big"),
                )
            )
        return _digest_fields(
            b"sglang-packed-writer-cohort-v1",
            tuple(fields),
        )

    def projection_for(self, writer_id: StagingWriterId) -> PackedWriterProjection:
        """Return the exact projection for one canonical writer.

        :param writer_id: Writer to locate.
        :returns: Authenticated writer projection.
        """

        for projection in self.projections:
            if projection.writer_id == writer_id:
                return projection
        raise PackedLifecycleError(f"writer is absent from cohort: {writer_id}")


class PackedAuthenticatedPeer:
    """Opaque transport-session authority for one canonical writer."""

    __slots__ = ("_authority_nonce", "_token")

    def __init__(
        self,
        authority_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one authority-owned peer handle."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("authenticated peers are transport-authority owned")
        self._authority_nonce = authority_nonce
        self._token = token


@dataclasses.dataclass(frozen=True)
class _AuthenticatedPeerRecord:
    """Private immutable transport-session binding."""

    peer: PackedAuthenticatedPeer
    native_endpoint: object
    writer_id: StagingWriterId
    generation: bytes


class PackedAuthenticatedPeerAuthority:
    """Exact-object registry for transport-authenticated writer sessions."""

    _authority_nonce: object
    _lock: threading.Lock
    _records: dict[object, _AuthenticatedPeerRecord]

    def __init__(self) -> None:
        """Initialize an empty transport-session registry."""

        self._authority_nonce = object()
        self._lock = threading.Lock()
        self._records = {}

    def bind_native_endpoint(
        self,
        native_endpoint: object,
        writer_id: StagingWriterId,
        generation: bytes,
    ) -> PackedAuthenticatedPeer:
        """Bind a transport-authenticated native endpoint to one writer.

        This is the narrow trust boundary implemented by the native transport
        adapter. Descriptive agent names never enter authority validation.

        :param native_endpoint: Exact connected native endpoint owner.
        :param writer_id: Canonical writer authenticated by bootstrap.
        :param generation: Process generation for that native endpoint.
        :returns: Opaque peer authority.
        """

        if native_endpoint is None or type(native_endpoint) in (str, bytes):
            raise TypeError("native_endpoint must be an opaque native owner")
        if type(writer_id) is not StagingWriterId:
            raise TypeError("writer_id must be StagingWriterId")
        _require_exact_bytes(
            generation,
            PACKED_GENERATION_BYTES,
            "peer generation",
        )
        token = object()
        peer = PackedAuthenticatedPeer(
            self._authority_nonce,
            token,
            _OPAQUE_HANDLE_SEAL,
        )
        record = _AuthenticatedPeerRecord(
            peer=peer,
            native_endpoint=native_endpoint,
            writer_id=writer_id,
            generation=generation,
        )
        with self._lock:
            self._records[token] = record
        return peer

    def validate(
        self,
        peer: PackedAuthenticatedPeer,
    ) -> _AuthenticatedPeerRecord:
        """Resolve one exact authority-owned peer.

        :param peer: Candidate opaque peer handle.
        :returns: Exact private session binding.
        """

        if type(peer) is not PackedAuthenticatedPeer:
            raise TypeError("peer must be PackedAuthenticatedPeer")
        if peer._authority_nonce is not self._authority_nonce:
            raise PackedLifecycleError("peer belongs to another authority")
        with self._lock:
            record = self._records.get(peer._token)
            if record is None or record.peer is not peer:
                raise PackedLifecycleError("peer authority is not live")
            return record


@dataclasses.dataclass(frozen=True)
class PackedArenaGrant:
    """Serializable arena grant authenticated by its destination owner.

    The grant is wire data rather than local authority. It becomes usable only
    when resolved by the issuing authority against the exact authenticated
    writer cohort.

    :ivar capability_id: Random capability identity.
    :ivar authenticator: HMAC over every grant-bearing field.
    :ivar arena_generation: Exact retained registration generation.
    :ivar request_generation: Exact request generation.
    :ivar route_digest: Decode-selected native route identity.
    :ivar writer_manifest_digest: Ordered writer-cohort identity.
    :ivar base_address: Registered destination staging base address.
    :ivar total_size: Registered destination staging capacity.
    :ivar destination_gpu_id: Destination CUDA device identifier.
    :ivar alignment_bytes: Required projection alignment.
    """

    capability_id: bytes
    authenticator: bytes
    arena_generation: bytes
    request_generation: bytes
    route_digest: bytes
    writer_manifest_digest: bytes
    base_address: int
    total_size: int
    destination_gpu_id: int
    alignment_bytes: int

    def __post_init__(self) -> None:
        """Validate bounded grant data without treating it as authority."""

        _require_exact_bytes(
            self.capability_id,
            PACKED_CAPABILITY_ID_BYTES,
            "capability_id",
        )
        _require_exact_bytes(
            self.authenticator,
            PACKED_AUTHENTICATOR_BYTES,
            "authenticator",
        )
        _require_exact_bytes(
            self.arena_generation,
            PACKED_GENERATION_BYTES,
            "arena_generation",
        )
        _require_exact_bytes(
            self.request_generation,
            PACKED_REQUEST_GENERATION_BYTES,
            "request_generation",
        )
        _require_exact_bytes(
            self.route_digest,
            PACKED_ROUTE_DIGEST_BYTES,
            "route_digest",
        )
        _require_exact_bytes(
            self.writer_manifest_digest,
            PACKED_ROUTE_DIGEST_BYTES,
            "writer_manifest_digest",
        )
        if type(self.base_address) is not int or self.base_address <= 0:
            raise ValueError("base_address must be a positive integer")
        if type(self.total_size) is not int or self.total_size <= 0:
            raise ValueError("total_size must be a positive integer")
        if type(self.destination_gpu_id) is not int or self.destination_gpu_id < 0:
            raise ValueError("destination_gpu_id must be non-negative")
        if type(self.alignment_bytes) is not int or self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be a positive integer")
        if self.base_address % self.alignment_bytes != 0:
            raise ValueError("arena grant base address is not aligned")


class PackedArenaCapability:
    """Opaque, revocable authority for one arena and request generation."""

    __slots__ = ("_authority_nonce", "_token")

    def __init__(
        self,
        authority_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one arena-authority-owned handle."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("arena capabilities are authority owned")
        self._authority_nonce = authority_nonce
        self._token = token


@dataclasses.dataclass
class _ArenaCapabilityRecord:
    """Private mutable arena capability state."""

    capability: PackedArenaCapability
    grant: PackedArenaGrant
    manifest: PackedWriterCohortManifest
    peers: tuple[PackedAuthenticatedPeer, ...]
    registration: object
    arena_owner: object
    state: PackedCapabilityState = PackedCapabilityState.ACTIVE
    resolved: bool = False


class PackedArenaCapabilityAuthority:
    """Authenticated issuance, exact resolution, revocation, and quarantine."""

    _arena_generation: bytes
    _arena_owner: object
    _authority_nonce: object
    _lock: threading.Lock
    _peer_authority: PackedAuthenticatedPeerAuthority
    _records: dict[object, _ArenaCapabilityRecord]
    _records_by_id: dict[bytes, object]
    _registration: object
    _secret: bytes

    def __init__(
        self,
        *,
        peer_authority: PackedAuthenticatedPeerAuthority,
        arena_generation: bytes,
        registration: object,
        arena_owner: object,
        secret: bytes | None = None,
    ) -> None:
        """Initialize authority for one exact retained arena registration.

        :param peer_authority: Authenticated transport-session registry.
        :param arena_generation: Exact registration generation.
        :param registration: Exact native registration owner.
        :param arena_owner: Strong arena allocation owner.
        :param secret: Optional deterministic 32-byte test secret.
        """

        _require_exact_bytes(
            arena_generation,
            PACKED_GENERATION_BYTES,
            "arena_generation",
        )
        if registration is None:
            raise ValueError("registration must not be None")
        if arena_owner is None:
            raise ValueError("arena_owner must not be None")
        selected_secret = secrets.token_bytes(PACKED_AUTHENTICATOR_BYTES)
        if secret is not None:
            _require_exact_bytes(
                secret,
                PACKED_AUTHENTICATOR_BYTES,
                "capability secret",
            )
            selected_secret = secret
        self._arena_generation = arena_generation
        self._arena_owner = arena_owner
        self._authority_nonce = object()
        self._lock = threading.Lock()
        self._peer_authority = peer_authority
        self._records = {}
        self._records_by_id = {}
        self._registration = registration
        self._secret = selected_secret

    def issue(
        self,
        *,
        peers: tuple[PackedAuthenticatedPeer, ...],
        manifest: PackedWriterCohortManifest,
        route_digest: bytes,
        base_address: int,
        total_size: int,
        destination_gpu_id: int,
        alignment_bytes: int,
    ) -> tuple[PackedArenaCapability, PackedArenaGrant]:
        """Issue one capability bound to the complete authenticated cohort.

        :param peers: Exact authenticated peers in manifest order.
        :param manifest: Request-level writer and projection truth.
        :param route_digest: Decode-selected native route digest.
        :param base_address: Registered arena base.
        :param total_size: Registered arena capacity.
        :param destination_gpu_id: Destination CUDA device.
        :param alignment_bytes: Arena allocation alignment.
        :returns: Opaque local authority and authenticated wire grant.
        """

        if type(manifest) is not PackedWriterCohortManifest:
            raise TypeError("manifest must be PackedWriterCohortManifest")
        _require_exact_bytes(
            route_digest,
            PACKED_ROUTE_DIGEST_BYTES,
            "route_digest",
        )
        if type(base_address) is not int or base_address <= 0:
            raise ValueError("base_address must be a positive integer")
        if type(total_size) is not int or total_size <= 0:
            raise ValueError("total_size must be a positive integer")
        if total_size != manifest.total_bytes:
            raise ValueError("arena size differs from writer manifest")
        if type(destination_gpu_id) is not int or destination_gpu_id < 0:
            raise ValueError("destination_gpu_id must be non-negative")
        if type(alignment_bytes) is not int or alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be a positive integer")
        if alignment_bytes != manifest.alignment_bytes:
            raise ValueError("arena alignment differs from writer manifest")
        if base_address % alignment_bytes != 0:
            raise ValueError("arena base address is not aligned")
        owned_peers = tuple(peers)
        if len(owned_peers) != len(manifest.projections):
            raise ValueError("peer count differs from writer manifest")
        for peer, projection in zip(
            owned_peers,
            manifest.projections,
            strict=True,
        ):
            record = self._peer_authority.validate(peer)
            if record.writer_id != projection.writer_id:
                raise ValueError("authenticated peer order differs from manifest")
            if record.generation != projection.peer_generation:
                raise ValueError("authenticated peer generation differs from manifest")

        capability_id = secrets.token_bytes(PACKED_CAPABILITY_ID_BYTES)
        unsigned_fields = (
            capability_id,
            self._arena_generation,
            manifest.request_generation,
            route_digest,
            manifest.digest,
            base_address.to_bytes(8, "big"),
            total_size.to_bytes(8, "big"),
            destination_gpu_id.to_bytes(4, "big"),
            alignment_bytes.to_bytes(8, "big"),
        )
        authenticator = hmac.new(
            self._secret,
            _digest_fields(b"sglang-packed-arena-grant-v1", unsigned_fields),
            hashlib.sha256,
        ).digest()
        grant = PackedArenaGrant(
            capability_id=capability_id,
            authenticator=authenticator,
            arena_generation=self._arena_generation,
            request_generation=manifest.request_generation,
            route_digest=route_digest,
            writer_manifest_digest=manifest.digest,
            base_address=base_address,
            total_size=total_size,
            destination_gpu_id=destination_gpu_id,
            alignment_bytes=alignment_bytes,
        )
        token = object()
        capability = PackedArenaCapability(
            self._authority_nonce,
            token,
            _OPAQUE_HANDLE_SEAL,
        )
        record = _ArenaCapabilityRecord(
            capability=capability,
            grant=grant,
            manifest=manifest,
            peers=owned_peers,
            registration=self._registration,
            arena_owner=self._arena_owner,
        )
        with self._lock:
            if capability_id in self._records_by_id:
                raise RuntimeError("capability identity collision")
            self._records[token] = record
            self._records_by_id[capability_id] = token
        return capability, grant

    def resolve(
        self,
        grant: PackedArenaGrant,
        peers: tuple[PackedAuthenticatedPeer, ...],
    ) -> PackedArenaCapability:
        """Authenticate wire grant data and resolve its exact live authority.

        :param grant: Candidate serializable grant.
        :param peers: Exact authenticated cohort presenting the grant.
        :returns: Original opaque capability.
        """

        if type(grant) is not PackedArenaGrant:
            raise TypeError("grant must be PackedArenaGrant")
        with self._lock:
            token = self._records_by_id.get(grant.capability_id)
            if token is None:
                raise PackedLifecycleError("arena grant is unknown")
            record = self._records[token]
            if not hmac.compare_digest(
                grant.authenticator,
                record.grant.authenticator,
            ):
                raise PackedLifecycleError("arena grant authentication failed")
            if grant != record.grant:
                raise PackedLifecycleError("arena grant fields were modified")
            if tuple(peers) != record.peers:
                raise PackedLifecycleError(
                    "arena grant was presented by another writer cohort"
                )
            if record.state is not PackedCapabilityState.ACTIVE:
                raise PackedLifecycleError(f"arena capability is {record.state.value}")
            if record.resolved:
                raise PackedLifecycleError("arena grant was already resolved")
            record.resolved = True
            return record.capability

    def validate(
        self,
        capability: PackedArenaCapability,
        peer: PackedAuthenticatedPeer | None = None,
    ) -> _ArenaCapabilityRecord:
        """Resolve one exact active local capability.

        :param capability: Candidate local capability.
        :param peer: Optional exact cohort member.
        :returns: Private capability record.
        """

        if type(capability) is not PackedArenaCapability:
            raise TypeError("capability must be PackedArenaCapability")
        if capability._authority_nonce is not self._authority_nonce:
            raise PackedLifecycleError("capability belongs to another authority")
        with self._lock:
            record = self._records.get(capability._token)
            if record is None or record.capability is not capability:
                raise PackedLifecycleError("capability is not live")
            if record.state is not PackedCapabilityState.ACTIVE:
                raise PackedLifecycleError(f"arena capability is {record.state.value}")
            if peer is not None and peer not in record.peers:
                raise PackedLifecycleError("peer is absent from arena capability")
            return record

    def _revoke_after_teardown(
        self,
        capability: PackedArenaCapability,
        finalizer: object,
    ) -> None:
        """Revoke one capability under teardown-coordinator authority."""

        if finalizer is not _AUTHORITY_SEAL:
            raise TypeError("capability revocation requires teardown authority")
        with self._lock:
            record = self._records.get(capability._token)
            if record is None or record.capability is not capability:
                raise PackedLifecycleError("capability is not live")
            if record.state is PackedCapabilityState.REVOKED:
                return
            if record.state is PackedCapabilityState.QUARANTINED:
                raise PackedProcessQuarantinedError(
                    "quarantined capability cannot be revoked"
                )
            if record.state is not PackedCapabilityState.TEARING_DOWN:
                raise PackedLifecycleError("capability was not reserved for teardown")
            record.state = PackedCapabilityState.REVOKED

    def _begin_teardown(
        self,
        capability: PackedArenaCapability,
        coordinator: object,
    ) -> None:
        """Reserve one active capability for exact teardown."""

        if coordinator is not _AUTHORITY_SEAL:
            raise TypeError("teardown reservation requires lifecycle authority")
        with self._lock:
            record = self._records.get(capability._token)
            if record is None or record.capability is not capability:
                raise PackedLifecycleError("capability is not live")
            if record.state is not PackedCapabilityState.ACTIVE:
                raise PackedLifecycleError(f"capability is {record.state.value}")
            record.state = PackedCapabilityState.TEARING_DOWN

    def _quarantine(
        self,
        capability: PackedArenaCapability,
        authority: object,
    ) -> None:
        """Make one capability permanently non-revocable."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("capability quarantine requires lifecycle authority")
        with self._lock:
            record = self._records.get(capability._token)
            if record is None or record.capability is not capability:
                raise PackedLifecycleError("capability is not live")
            record.state = PackedCapabilityState.QUARANTINED

    def state(self, capability: PackedArenaCapability) -> PackedCapabilityState:
        """Return one capability state for diagnostics.

        :param capability: Exact authority-owned capability.
        :returns: Current capability state.
        """

        if type(capability) is not PackedArenaCapability:
            raise TypeError("capability must be PackedArenaCapability")
        with self._lock:
            record = self._records.get(capability._token)
            if record is None or record.capability is not capability:
                raise PackedLifecycleError("capability is not live")
            return record.state

    def create_resource_cohort(
        self,
        resources: "PackedCohortResources",
    ) -> "PackedResourceCohort":
        """Create a cohort only after validating its authority-owned anchors.

        :param resources: Complete local resource ownership set.
        :returns: Opaque cohort bound to this capability authority.
        """

        if type(resources) is not PackedCohortResources:
            raise TypeError("resources must be PackedCohortResources")
        record = self.validate(resources.capability)
        if not _contains_exact(resources.registrations, record.registration):
            raise PackedLifecycleError("cohort omits the capability registration owner")
        if not _contains_exact(resources.arenas, record.arena_owner):
            raise PackedLifecycleError("cohort omits the capability arena owner")
        expected_endpoints = tuple(
            self._peer_authority.validate(peer).native_endpoint for peer in record.peers
        )
        for endpoint in expected_endpoints:
            if not _contains_exact(resources.endpoints, endpoint):
                raise PackedLifecycleError(
                    "cohort omits an authenticated native endpoint"
                )
        return PackedResourceCohort(
            resources,
            tuple(projection.writer_id for projection in record.manifest.projections),
            self,
            self._authority_nonce,
            _OPAQUE_HANDLE_SEAL,
        )

    def _attach_native_handle(
        self,
        *,
        cohort: "PackedResourceCohort",
        capability: PackedArenaCapability,
        writer_id: StagingWriterId,
        handle: object,
    ) -> None:
        """Attach a handle after exact capability/cohort validation."""

        self.validate(capability)
        cohort._attach_native_handle(
            capability=capability,
            writer_id=writer_id,
            handle=handle,
            authority_nonce=self._authority_nonce,
            authority=_AUTHORITY_SEAL,
        )

    def _resource_cohort_details(
        self,
        *,
        cohort: "PackedResourceCohort",
        capability: PackedArenaCapability,
    ) -> (
        "tuple[tuple[StagingWriterId, ...], "
        "tuple[tuple[PackedPageLeaseAllocator, PackedPageLease], ...]]"
    ):
        """Return authority-owned cohort members for teardown validation."""

        self.validate(capability)
        return cohort._authority_details(
            capability=capability,
            authority_nonce=self._authority_nonce,
            authority=_AUTHORITY_SEAL,
        )


class PackedPageLease:
    """Opaque allocator-owned claim on immutable request-local pages."""

    __slots__ = ("_allocator_nonce", "_token")

    def __init__(
        self,
        allocator_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one allocator-owned page lease."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("page leases are allocator owned")
        self._allocator_nonce = allocator_nonce
        self._token = token


@dataclasses.dataclass
class _PageLeaseRecord:
    """Private mutable page-lease state."""

    lease: PackedPageLease
    role: PackedPageLeaseRole
    owner: object
    pages: tuple[tuple[StagingComponentId, npt.NDArray[np.int32]], ...]
    released: bool = False
    quarantined: bool = False


class PackedPageLeaseAllocator:
    """Exact-object owner for source or destination KV page claims."""

    _allocator_nonce: object
    _lock: threading.Lock
    _records: dict[object, _PageLeaseRecord]
    _role: PackedPageLeaseRole

    def __init__(self, role: PackedPageLeaseRole) -> None:
        """Initialize one role-specific page lease owner.

        :param role: Source or destination ownership role.
        """

        if type(role) is not PackedPageLeaseRole:
            raise TypeError("role must be PackedPageLeaseRole")
        self._allocator_nonce = object()
        self._lock = threading.Lock()
        self._records = {}
        self._role = role

    def claim(
        self,
        *,
        owner: object,
        pages: dict[StagingComponentId, npt.NDArray[np.int32]],
    ) -> PackedPageLease:
        """Claim exact immutable component pages until teardown or quarantine.

        :param owner: Upstream page allocator or request owner.
        :param pages: Active component page arrays.
        :returns: Opaque allocator-owned page lease.
        """

        if owner is None:
            raise ValueError("page lease owner must not be None")
        if len(pages) == 0:
            raise ValueError("page lease must contain at least one component")
        snapshots: list[tuple[StagingComponentId, npt.NDArray[np.int32]]] = []
        for component_id, page_array in sorted(
            pages.items(),
            key=lambda item: _component_sort_key(item[0]),
        ):
            if not isinstance(page_array, np.ndarray):
                raise TypeError("page lease pages must be NumPy arrays")
            if page_array.dtype != np.dtype(np.int32) or page_array.ndim != 1:
                raise TypeError("page lease pages must be one-dimensional int32")
            if len(page_array) == 0:
                raise ValueError("page lease component must not be empty")
            copied_pages = np.array(
                page_array,
                dtype=np.int32,
                order="C",
                copy=True,
            )
            if np.any(copied_pages < 0):
                raise ValueError("page lease contains a negative page index")
            snapshot = np.frombuffer(
                copied_pages.tobytes(order="C"),
                dtype=np.int32,
            )
            snapshots.append((component_id, snapshot))
        token = object()
        lease = PackedPageLease(
            self._allocator_nonce,
            token,
            _OPAQUE_HANDLE_SEAL,
        )
        record = _PageLeaseRecord(
            lease=lease,
            role=self._role,
            owner=owner,
            pages=tuple(snapshots),
        )
        with self._lock:
            self._records[token] = record
        return lease

    def pages(
        self,
        lease: PackedPageLease,
    ) -> tuple[tuple[StagingComponentId, npt.NDArray[np.int32]], ...]:
        """Return immutable page snapshots for one exact live claim.

        :param lease: Exact allocator-owned lease.
        :returns: Canonically ordered component pages.
        """

        with self._lock:
            record = self._validate_locked(lease)
            if record.released:
                raise PackedLifecycleError("page lease is released")
            return record.pages

    def _release_after_teardown(
        self,
        lease: PackedPageLease,
        finalizer: object,
    ) -> None:
        """Release one page lease under teardown-coordinator authority."""

        if finalizer is not _AUTHORITY_SEAL:
            raise TypeError("page reuse requires teardown authority")
        with self._lock:
            record = self._validate_locked(lease)
            if record.quarantined:
                raise PackedProcessQuarantinedError(
                    "quarantined page lease cannot be released"
                )
            record.released = True

    def _quarantine(
        self,
        lease: PackedPageLease,
        authority: object,
    ) -> None:
        """Make one page lease permanently non-reusable."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("page quarantine requires lifecycle authority")
        with self._lock:
            record = self._validate_locked(lease)
            record.quarantined = True

    def is_reusable(self, lease: PackedPageLease) -> bool:
        """Return whether teardown released one exact page lease.

        :param lease: Exact allocator-owned lease.
        :returns: Whether its pages may be reused.
        """

        with self._lock:
            record = self._validate_locked(lease)
            return record.released and not record.quarantined

    @property
    def role(self) -> PackedPageLeaseRole:
        """Return this allocator's fixed ownership role.

        :returns: Source or destination role.
        """

        return self._role

    def _validate_locked(self, lease: PackedPageLease) -> _PageLeaseRecord:
        """Validate exact allocator ownership while its lock is held."""

        if type(lease) is not PackedPageLease:
            raise TypeError("lease must be PackedPageLease")
        if lease._allocator_nonce is not self._allocator_nonce:
            raise PackedLifecycleError("page lease belongs to another allocator")
        record = self._records.get(lease._token)
        if record is None or record.lease is not lease:
            raise PackedLifecycleError("page lease is not live")
        return record


@dataclasses.dataclass(frozen=True)
class PackedCohortResources:
    """Complete local ownership cohort retained on lifecycle ambiguity.

    :ivar capability: Request-scoped remote-memory authority.
    :ivar source_page_leases: Source allocator and exact page claims.
    :ivar destination_page_leases: Destination allocator and exact page claims.
    :ivar staging_leases: Exact contiguous staging allocations.
    :ivar tensors: Source and destination allocation owners.
    :ivar registrations: Native memory-registration owners.
    :ivar endpoints: Native connection and progress owners.
    :ivar arenas: Arena and allocator owners.
    """

    capability: PackedArenaCapability
    source_page_leases: tuple[
        tuple[PackedPageLeaseAllocator, PackedPageLease],
        ...,
    ]
    destination_page_leases: tuple[
        tuple[PackedPageLeaseAllocator, PackedPageLease],
        ...,
    ]
    staging_leases: tuple[object, ...]
    tensors: tuple[object, ...]
    registrations: tuple[object, ...]
    endpoints: tuple[object, ...]
    arenas: tuple[object, ...]

    def __post_init__(self) -> None:
        """Own and validate all pre-submission cohort members."""

        if type(self.capability) is not PackedArenaCapability:
            raise TypeError("cohort capability must be PackedArenaCapability")
        for field_name, expected_role in (
            ("source_page_leases", PackedPageLeaseRole.SOURCE),
            ("destination_page_leases", PackedPageLeaseRole.DESTINATION),
        ):
            bindings = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, bindings)
            if len(bindings) == 0:
                raise ValueError(f"cohort {field_name} must not be empty")
            for allocator, lease in bindings:
                if type(allocator) is not PackedPageLeaseAllocator:
                    raise TypeError(
                        f"cohort {field_name} allocator has an invalid type"
                    )
                if allocator.role is not expected_role:
                    raise ValueError(
                        f"cohort {field_name} allocator has the wrong role"
                    )
                allocator.pages(lease)
        for field_name, values in (
            ("staging_leases", tuple(self.staging_leases)),
            ("tensors", tuple(self.tensors)),
            ("registrations", tuple(self.registrations)),
            ("endpoints", tuple(self.endpoints)),
            ("arenas", tuple(self.arenas)),
        ):
            object.__setattr__(self, field_name, values)
            if len(values) == 0:
                raise ValueError(f"cohort {field_name} must not be empty")
            if any(value is None for value in values):
                raise ValueError(f"cohort {field_name} contains None")
        page_leases = tuple(
            lease
            for _, lease in (
                *self.source_page_leases,
                *self.destination_page_leases,
            )
        )
        if len({id(lease) for lease in page_leases}) != len(page_leases):
            raise ValueError("cohort page leases must be exact and unique")


class PackedResourceCohort:
    """Opaque mutable owner of a complete local transfer-resource cohort."""

    __slots__ = (
        "_authority_nonce",
        "_capability_authority",
        "_expected_writers",
        "_handles_by_writer",
        "_lock",
        "_quarantined",
        "_resources",
        "_scatter_ownership",
    )

    def __init__(
        self,
        resources: PackedCohortResources,
        expected_writers: tuple[StagingWriterId, ...],
        capability_authority: PackedArenaCapabilityAuthority,
        authority_nonce: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned resource cohort."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("resource cohorts are lifecycle owned")
        if type(resources) is not PackedCohortResources:
            raise TypeError("resources must be PackedCohortResources")
        if type(capability_authority) is not PackedArenaCapabilityAuthority:
            raise TypeError(
                "capability_authority must be PackedArenaCapabilityAuthority"
            )
        owned_expected_writers = tuple(expected_writers)
        if len(owned_expected_writers) == 0:
            raise ValueError("resource cohort must contain at least one writer")
        self._authority_nonce = authority_nonce
        self._capability_authority = capability_authority
        self._expected_writers = owned_expected_writers
        self._handles_by_writer: dict[StagingWriterId, object] = {}
        self._lock = threading.Lock()
        self._quarantined = False
        self._resources = resources
        self._scatter_ownership: list[tuple[object, tuple[object, ...]]] = []

    def _attach_native_handle(
        self,
        *,
        capability: PackedArenaCapability,
        writer_id: StagingWriterId,
        handle: object,
        authority_nonce: object,
        authority: object,
    ) -> None:
        """Attach an exact handle under its issuing arena authority."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("native handle attachment requires lifecycle authority")
        if authority_nonce is not self._authority_nonce:
            raise PackedLifecycleError("cohort belongs to another arena authority")
        if capability is not self._resources.capability:
            raise PackedLifecycleError("cohort belongs to another capability")
        if type(writer_id) is not StagingWriterId:
            raise TypeError("writer_id must be StagingWriterId")
        if writer_id not in self._expected_writers:
            raise PackedLifecycleError("writer is absent from resource cohort")
        if handle is None:
            raise ValueError("native handle must not be None")
        with self._lock:
            if self._quarantined:
                raise PackedProcessQuarantinedError(
                    "cannot attach to a quarantined cohort"
                )
            if writer_id in self._handles_by_writer:
                raise PackedLifecycleError(
                    "cohort already owns a handle for this writer"
                )
            if _contains_exact(
                tuple(self._handles_by_writer.values()),
                handle,
            ):
                raise PackedLifecycleError(
                    "native handle is already owned by another writer"
                )
            self._handles_by_writer[writer_id] = handle

    def require_ready_for_post(self) -> None:
        """Require exact prepared-handle ownership for every cohort writer."""

        with self._lock:
            if self._quarantined:
                raise PackedProcessQuarantinedError("resource cohort is quarantined")
            if set(self._handles_by_writer) != set(self._expected_writers):
                raise PackedLifecycleError(
                    "every writer handle must be registered before native post"
                )

    def _reserve_scatter_submission(
        self,
        *,
        destination_lease: object,
        resources: tuple[object, ...],
        authority: object,
    ) -> object:
        """Retain scatter resources before the trusted native launch."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("scatter reservation requires lifecycle authority")
        if destination_lease is None:
            raise ValueError("destination_lease must not be None")
        owned_resources = tuple(resources)
        if len(owned_resources) == 0:
            raise ValueError("scatter resources must not be empty")
        if any(resource is None for resource in owned_resources):
            raise ValueError("scatter resources contain None")
        if not _contains_exact(
            self._resources.staging_leases,
            destination_lease,
        ):
            raise PackedLifecycleError(
                "scatter destination lease is absent from resource cohort"
            )
        with self._lock:
            if self._quarantined:
                raise PackedProcessQuarantinedError(
                    "cannot reserve scatter on a quarantined cohort"
                )
            reservation = object()
            self._scatter_ownership.append(
                (
                    reservation,
                    (destination_lease, *owned_resources),
                )
            )
            return reservation

    def _bind_scatter_event(
        self,
        *,
        reservation: object,
        event: object,
        authority: object,
    ) -> None:
        """Bind the exact trusted event returned by scatter submission."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("scatter event binding requires lifecycle authority")
        if event is None:
            raise ValueError("scatter event must not be None")
        with self._lock:
            if self._quarantined:
                raise PackedProcessQuarantinedError(
                    "cannot bind scatter on a quarantined cohort"
                )
            matching_indices = tuple(
                index
                for index, (owner, _) in enumerate(self._scatter_ownership)
                if owner is reservation
            )
            if len(matching_indices) != 1:
                raise PackedLifecycleError("scatter reservation is not uniquely owned")
            if any(
                owner is event or _contains_exact(ownership, event)
                for owner, ownership in self._scatter_ownership
            ):
                raise PackedLifecycleError(
                    "scatter event is already owned by this cohort"
                )
            index = matching_indices[0]
            _, ownership = self._scatter_ownership[index]
            self._scatter_ownership[index] = (
                reservation,
                (event, *ownership),
            )

    def _authority_details(
        self,
        *,
        capability: PackedArenaCapability,
        authority_nonce: object,
        authority: object,
    ) -> tuple[
        tuple[StagingWriterId, ...],
        tuple[tuple[PackedPageLeaseAllocator, PackedPageLease], ...],
    ]:
        """Return exact teardown bindings under arena authority."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("cohort inspection requires lifecycle authority")
        if authority_nonce is not self._authority_nonce:
            raise PackedLifecycleError("cohort belongs to another arena authority")
        if capability is not self._resources.capability:
            raise PackedLifecycleError("cohort belongs to another capability")
        resources = self._resources
        return (
            self._expected_writers,
            (
                *resources.source_page_leases,
                *resources.destination_page_leases,
            ),
        )

    def retained_objects(self) -> tuple[object, ...]:
        """Return the complete strong-retention set.

        :returns: All cohort resources, including the native handle.
        """

        with self._lock:
            if len(self._handles_by_writer) == 0:
                raise PackedLifecycleError(
                    "resource cohort has no attached native handles"
                )
            resources = self._resources
            page_lease_objects = tuple(
                value
                for allocator, lease in (
                    *resources.source_page_leases,
                    *resources.destination_page_leases,
                )
                for value in (allocator, lease)
            )
            handles = tuple(
                self._handles_by_writer[writer_id]
                for writer_id in self._expected_writers
                if writer_id in self._handles_by_writer
            )
            scatter_objects = tuple(
                value
                for event, ownership in self._scatter_ownership
                for value in (event, *ownership)
            )
            return (
                self,
                resources.capability,
                *page_lease_objects,
                *resources.staging_leases,
                *resources.tensors,
                *resources.registrations,
                *resources.endpoints,
                *resources.arenas,
                *handles,
                *scatter_objects,
            )

    def _mark_quarantined(self, authority: object) -> None:
        """Make the cohort permanently non-releasable."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("cohort quarantine requires lifecycle authority")
        with self._lock:
            self._quarantined = True
            page_lease_bindings = (
                *self._resources.source_page_leases,
                *self._resources.destination_page_leases,
            )
            capability = self._resources.capability
        self._capability_authority._quarantine(
            capability,
            _AUTHORITY_SEAL,
        )
        for allocator, lease in page_lease_bindings:
            allocator._quarantine(lease, _AUTHORITY_SEAL)

    @property
    def quarantined(self) -> bool:
        """Return whether this cohort is permanently retained.

        :returns: Quarantine state.
        """

        with self._lock:
            return self._quarantined


@dataclasses.dataclass(frozen=True)
class _QuarantinedCohort:
    """Private process-lifetime retention record."""

    cohort: PackedResourceCohort
    reason: str
    resources: tuple[object, ...]


class PackedProcessLifetimeQuarantine:
    """Strong process-lifetime retention with deliberately no release API."""

    _lock: threading.Lock
    _records: list[_QuarantinedCohort]
    _retained_ids: set[int]

    def __init__(self) -> None:
        """Initialize an empty process-lifetime quarantine."""

        self._lock = threading.Lock()
        self._records = []
        self._retained_ids = set()

    def retain(self, cohort: PackedResourceCohort, reason: str) -> None:
        """Retain one complete cohort exactly once.

        :param cohort: Complete lifecycle cohort.
        :param reason: Stable ambiguity reason.
        """

        if type(cohort) is not PackedResourceCohort:
            raise TypeError("cohort must be PackedResourceCohort")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        cohort._mark_quarantined(_AUTHORITY_SEAL)
        resources = cohort.retained_objects()
        with self._lock:
            cohort_id = id(cohort)
            if cohort_id in self._retained_ids:
                return
            self._retained_ids.add(cohort_id)
            self._records.append(
                _QuarantinedCohort(
                    cohort=cohort,
                    reason=reason,
                    resources=resources,
                )
            )

    @property
    def count(self) -> int:
        """Return quarantined cohort count.

        :returns: Process-lifetime cohort count.
        """

        with self._lock:
            return len(self._records)

    def retains(self, resource: object) -> bool:
        """Return whether an exact object is strongly retained.

        :param resource: Object identity to locate.
        :returns: Whether a cohort retains that exact object.
        """

        with self._lock:
            return any(
                candidate is resource
                for record in self._records
                for candidate in record.resources
            )

    def reasons(self) -> tuple[str, ...]:
        """Return stable quarantine reasons.

        :returns: Reasons in retention order.
        """

        with self._lock:
            return tuple(record.reason for record in self._records)


class _OneShotReceipt:
    """Opaque issuer-owned one-shot lifecycle receipt."""

    __slots__ = ("_issuer_nonce", "_token")

    def __init__(
        self,
        issuer_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one issuer-owned receipt."""

        if construction_seal is not _RECEIPT_SEAL:
            raise TypeError("lifecycle receipts are issuer owned")
        self._issuer_nonce = issuer_nonce
        self._token = token


class PackedSubmitReceipt(_OneShotReceipt):
    """One-shot proof that one exact native handle was submitted."""


class PackedCompletionReceipt(_OneShotReceipt):
    """One-shot exact-handle endpoint-flush completion proof."""


class PackedScatterReceipt(_OneShotReceipt):
    """One-shot proof of one exact destination scatter submission."""


class PackedScatterCompletionReceipt(_OneShotReceipt):
    """One-shot terminal proof for one exact scatter event."""


class PackedWriterCohortCompletionReceipt(_OneShotReceipt):
    """One-shot proof that every authenticated writer completed."""


class PackedTeardownBarrierReceipt(_OneShotReceipt):
    """One-shot proof that every writer acknowledged exact teardown."""


@dataclasses.dataclass
class _ReceiptRecord:
    """Private issuer state for one receipt."""

    receipt: _OneShotReceipt
    binding: tuple[object, ...]
    consumed: bool = False


class _ReceiptAuthority:
    """Exact-object issuance and single-use consumption."""

    _issuer_nonce: object
    _lock: threading.Lock
    _records: dict[object, _ReceiptRecord]

    def __init__(self) -> None:
        """Initialize one independent receipt namespace."""

        self._issuer_nonce = object()
        self._lock = threading.Lock()
        self._records = {}

    def issue(
        self,
        receipt_type: type[_OneShotReceipt],
        binding: tuple[object, ...],
    ) -> _OneShotReceipt:
        """Issue one receipt bound to exact lifecycle objects."""

        if not issubclass(receipt_type, _OneShotReceipt):
            raise TypeError("receipt_type must derive from _OneShotReceipt")
        token = object()
        receipt = receipt_type(
            self._issuer_nonce,
            token,
            _RECEIPT_SEAL,
        )
        with self._lock:
            self._records[token] = _ReceiptRecord(
                receipt=receipt,
                binding=tuple(binding),
            )
        return receipt

    def consume(
        self,
        receipt: _OneShotReceipt,
        receipt_type: type[_OneShotReceipt],
        binding: tuple[object, ...],
    ) -> None:
        """Consume one exact receipt once."""

        if type(receipt) is not receipt_type:
            raise TypeError(
                f"receipt must be exact {receipt_type.__name__}, "
                f"got {type(receipt).__name__}"
            )
        if receipt._issuer_nonce is not self._issuer_nonce:
            raise PackedLifecycleError("receipt belongs to another issuer")
        with self._lock:
            record = self._records.get(receipt._token)
            if record is None or record.receipt is not receipt:
                raise PackedLifecycleError("receipt is unknown")
            candidate_binding = tuple(binding)
            if len(record.binding) != len(candidate_binding) or any(
                expected is not candidate
                for expected, candidate in zip(
                    record.binding,
                    candidate_binding,
                    strict=True,
                )
            ):
                raise PackedLifecycleError("receipt belongs to another operation")
            if record.consumed:
                raise PackedLifecycleError("receipt was already consumed")
            record.consumed = True


class PackedNixlUcxNativeDriver:
    """Narrow trusted native NIXL/UCX handle interface.

    Implementations must derive observations from the exact native handle.
    Stock NIXL 1.3.2 does not expose the two attestation methods required by
    :func:`require_nixl_ucx_native_attestation`.
    """

    def post(self, handle: object) -> PackedNativePostState:
        """Post one exact prepared native handle.

        :param handle: Exact prepared NIXL handle.
        :returns: Submitted or synchronously completed state.
        """

        raise NotImplementedError

    def poll(self, handle: object) -> PackedNativePollState:
        """Poll one exact submitted native handle.

        :param handle: Exact submitted NIXL handle.
        :returns: Pending, done, or ambiguous error state.
        """

        raise NotImplementedError

    def query_backend(self, handle: object) -> PackedNixlBackend:
        """Attest the backend selected for one exact handle.

        :param handle: Exact prepared or submitted NIXL handle.
        :returns: Native backend identity.
        """

        raise NotImplementedError

    def query_ucp_transports(
        self,
        handle: object,
    ) -> frozenset[PackedUcpTransport] | None:
        """Attest every UCP transport selected by one exact handle.

        :param handle: Exact submitted NIXL handle.
        :returns: Exact native transport set, or ``None`` when unobservable.
        """

        raise NotImplementedError

    def query_runtime_identity(self, handle: object) -> bytes:
        """Return the exact loaded NIXL/UCX runtime identity.

        :param handle: Exact prepared or submitted NIXL handle.
        :returns: Runtime artifact digest.
        """

        raise NotImplementedError

    def query_completion_contract(
        self,
        handle: object,
    ) -> PackedNativeCompletionContract | None:
        """Attest the exact native chain represented by DONE.

        :param handle: Exact completed NIXL handle.
        :returns: Endpoint-flush contract, or ``None`` when unobservable.
        """

        raise NotImplementedError

    def release_completed(self, handle: object) -> None:
        """Release an exact handle already proven complete.

        :param handle: Exact completed NIXL handle.
        """

        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class PackedExpectedNixlUcxRoute:
    """Decode-selected native route expected from exact-handle attestation.

    :ivar route_digest: Capability-bound route identity.
    :ivar runtime_identity: Exact loaded NIXL/UCX artifact digest.
    :ivar transports: Complete allowed UCP transport set.
    """

    route_digest: bytes
    runtime_identity: bytes
    transports: frozenset[PackedUcpTransport]

    def __post_init__(self) -> None:
        """Validate one exact native route contract."""

        _require_exact_bytes(
            self.route_digest,
            PACKED_ROUTE_DIGEST_BYTES,
            "route_digest",
        )
        _require_exact_bytes(
            self.runtime_identity,
            PACKED_RUNTIME_IDENTITY_BYTES,
            "runtime_identity",
        )
        transports = frozenset(self.transports)
        object.__setattr__(self, "transports", transports)
        if len(transports) == 0:
            raise ValueError("expected UCP transport set must not be empty")
        if any(type(transport) is not PackedUcpTransport for transport in transports):
            raise TypeError("expected UCP transports must be PackedUcpTransport")
        allowed = frozenset(
            (
                PackedUcpTransport.CUDA_IPC,
                PackedUcpTransport.CUDA,
            )
        )
        if not transports.issubset(allowed):
            raise ValueError("expected route contains an unsupported UCP transport")


class PackedPreparedTransfer:
    """Opaque ownership of one exact prepared native handle."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned prepared transfer."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("prepared transfers are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


class PackedInFlightTransfer:
    """Opaque ownership of one exact submitted native handle."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned in-flight transfer."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("in-flight transfers are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


class PackedCompletedTransfer:
    """Opaque exact-handle completion retained until teardown."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned completed transfer."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("completed transfers are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


@dataclasses.dataclass
class _NativeTransferRecord:
    """Private mutable state for one exact native handle."""

    prepared: PackedPreparedTransfer
    writer_id: StagingWriterId
    projection: PackedWriterProjection
    peer: PackedAuthenticatedPeer
    capability: PackedArenaCapability
    handle: object
    cohort: PackedResourceCohort
    state: PackedNativeTransferState = PackedNativeTransferState.PREPARED
    post_state: PackedNativePostState | None = None
    inflight: PackedInFlightTransfer | None = None
    completed: PackedCompletedTransfer | None = None


class PackedNixlUcxTransferLifecycle:
    """Exact-handle NIXL/UCX submission, completion, and quarantine owner."""

    _capability_authority: PackedArenaCapabilityAuthority
    _completion_receipts: _ReceiptAuthority
    _driver: PackedNixlUcxNativeDriver
    _lifecycle_nonce: object
    _lock: threading.Lock
    _quarantine: PackedProcessLifetimeQuarantine
    _records: dict[object, _NativeTransferRecord]
    _route: PackedExpectedNixlUcxRoute
    _submit_receipts: _ReceiptAuthority

    def __init__(
        self,
        *,
        driver: PackedNixlUcxNativeDriver,
        capability_authority: PackedArenaCapabilityAuthority,
        route: PackedExpectedNixlUcxRoute,
        quarantine: PackedProcessLifetimeQuarantine,
    ) -> None:
        """Bind exact native observations to one route and quarantine.

        :param driver: Trusted exact-handle native adapter.
        :param capability_authority: Arena capability owner.
        :param route: Required backend, runtime, and UCP transports.
        :param quarantine: Process-lifetime ambiguity retention.
        """

        if not isinstance(driver, PackedNixlUcxNativeDriver):
            raise TypeError("driver must derive from PackedNixlUcxNativeDriver")
        if type(capability_authority) is not PackedArenaCapabilityAuthority:
            raise TypeError(
                "capability_authority must be PackedArenaCapabilityAuthority"
            )
        if type(route) is not PackedExpectedNixlUcxRoute:
            raise TypeError("route must be PackedExpectedNixlUcxRoute")
        if type(quarantine) is not PackedProcessLifetimeQuarantine:
            raise TypeError("quarantine must be PackedProcessLifetimeQuarantine")
        self._capability_authority = capability_authority
        self._completion_receipts = _ReceiptAuthority()
        self._driver = driver
        self._lifecycle_nonce = object()
        self._lock = threading.Lock()
        self._quarantine = quarantine
        self._records = {}
        self._route = route
        self._submit_receipts = _ReceiptAuthority()

    def register_prepared(
        self,
        *,
        writer_id: StagingWriterId,
        peer: PackedAuthenticatedPeer,
        capability: PackedArenaCapability,
        handle: object,
        cohort: PackedResourceCohort,
    ) -> PackedPreparedTransfer:
        """Take exact handle ownership before any native post call.

        :param writer_id: Canonical writer owning this projection.
        :param peer: Exact authenticated writer session.
        :param capability: Exact live arena capability.
        :param handle: Prepared native NIXL handle.
        :param cohort: Complete local resource cohort.
        :returns: Opaque prepared-handle owner.
        """

        if handle is None:
            raise ValueError("native handle must not be None")
        if type(cohort) is not PackedResourceCohort:
            raise TypeError("cohort must be PackedResourceCohort")
        capability_record = self._capability_authority.validate(
            capability,
            peer,
        )
        if capability_record.grant.route_digest != self._route.route_digest:
            raise ValueError("capability route differs from native lifecycle")
        projection = capability_record.manifest.projection_for(writer_id)
        peer_index = capability_record.peers.index(peer)
        if capability_record.manifest.projections[peer_index] != projection:
            raise ValueError("authenticated peer does not own writer projection")
        self._capability_authority._attach_native_handle(
            cohort=cohort,
            capability=capability,
            writer_id=writer_id,
            handle=handle,
        )
        token = object()
        prepared = PackedPreparedTransfer(
            self._lifecycle_nonce,
            token,
            _OPAQUE_HANDLE_SEAL,
        )
        record = _NativeTransferRecord(
            prepared=prepared,
            writer_id=writer_id,
            projection=projection,
            peer=peer,
            capability=capability,
            handle=handle,
            cohort=cohort,
        )
        with self._lock:
            self._records[token] = record
        return prepared

    def post(self, prepared: PackedPreparedTransfer) -> PackedSubmitReceipt:
        """Post once and return an exact one-shot submission receipt.

        An exception, native ERROR, or missing per-handle route attestation
        yields no receipt and quarantines the entire cohort.

        :param prepared: Exact prepared-handle owner.
        :returns: One-shot submission receipt.
        """

        prepared_record: _NativeTransferRecord
        with self._lock:
            prepared_record = self._require_prepared_locked(prepared)
            if prepared_record.state is not PackedNativeTransferState.PREPARED:
                raise PackedLifecycleError(
                    f"native post is invalid in state {prepared_record.state.value}"
                )
        prepared_record.cohort.require_ready_for_post()
        with self._lock:
            record = self._require_prepared_locked(prepared)
            if record.state is not PackedNativeTransferState.PREPARED:
                raise PackedLifecycleError(
                    f"native post is invalid in state {record.state.value}"
                )
            record.state = PackedNativeTransferState.POSTING
        try:
            post_state = self._driver.post(record.handle)
            if type(post_state) is not PackedNativePostState:
                raise TypeError("native post returned an invalid state")
            if post_state is PackedNativePostState.ERROR:
                raise PackedAmbiguousTransportError(
                    "native post returned ERR without quiescence proof"
                )
            self._attest_route(record.handle, completion=False)
        except BaseException as error:
            _log_ambiguous_failure("packed native post failed")
            self._quarantine_record(
                record,
                "native post did not prove exact-handle submission and route",
            )
            if not isinstance(error, Exception):
                raise
            raise PackedAmbiguousTransportError(
                "native post is ambiguous; no writer outcome exists"
            ) from error

        with self._lock:
            if record.state is not PackedNativeTransferState.POSTING:
                if record.state is PackedNativeTransferState.QUARANTINED:
                    raise PackedAmbiguousTransportError(
                        "native post cohort was quarantined concurrently"
                    )
                raise RuntimeError("native post ownership changed")
            record.post_state = post_state
            record.state = PackedNativeTransferState.SUBMITTED
            receipt = self._submit_receipts.issue(
                PackedSubmitReceipt,
                (record.prepared, record.handle, record.cohort),
            )
        if type(receipt) is not PackedSubmitReceipt:
            raise RuntimeError("submit receipt issuer returned wrong type")
        return receipt

    def accept_submission(
        self,
        prepared: PackedPreparedTransfer,
        receipt: PackedSubmitReceipt,
    ) -> PackedInFlightTransfer:
        """Consume one submit receipt into exact in-flight handle ownership.

        :param prepared: Exact prepared owner.
        :param receipt: Exact one-shot submit receipt.
        :returns: Opaque in-flight owner.
        """

        with self._lock:
            record = self._require_prepared_locked(prepared)
            if record.state is not PackedNativeTransferState.SUBMITTED:
                raise PackedLifecycleError(
                    f"submission admission is invalid in state {record.state.value}"
                )
            self._submit_receipts.consume(
                receipt,
                PackedSubmitReceipt,
                (record.prepared, record.handle, record.cohort),
            )
            inflight = PackedInFlightTransfer(
                self._lifecycle_nonce,
                prepared._token,
                _OPAQUE_HANDLE_SEAL,
            )
            record.inflight = inflight
            record.state = PackedNativeTransferState.IN_FLIGHT
            return inflight

    def poll(
        self,
        inflight: PackedInFlightTransfer,
    ) -> PackedCompletionReceipt | None:
        """Poll exact native ownership and issue completion only after flush.

        :param inflight: Exact submitted-handle owner.
        :returns: One-shot completion receipt, or ``None`` while pending.
        """

        with self._lock:
            record = self._require_inflight_locked(inflight)
            if record.state is not PackedNativeTransferState.IN_FLIGHT:
                raise PackedLifecycleError(
                    f"native poll is invalid in state {record.state.value}"
                )
        try:
            poll_state = self._driver.poll(record.handle)
            if type(poll_state) is not PackedNativePollState:
                raise TypeError("native poll returned an invalid state")
            if poll_state is PackedNativePollState.ERROR:
                raise PackedAmbiguousTransportError(
                    "native poll returned ERR without quiescence proof"
                )
            if poll_state is PackedNativePollState.PENDING:
                return None
            self._attest_route(record.handle, completion=True)
        except BaseException as error:
            _log_ambiguous_failure("packed native progress failed")
            self._quarantine_record(
                record,
                "native progress lost exact-handle endpoint-flush terminality",
            )
            if not isinstance(error, Exception):
                raise
            raise PackedAmbiguousTransportError(
                "native progress is ambiguous; no writer outcome exists"
            ) from error

        with self._lock:
            if record.state is not PackedNativeTransferState.IN_FLIGHT:
                if record.state is PackedNativeTransferState.QUARANTINED:
                    raise PackedAmbiguousTransportError(
                        "native completion cohort was quarantined concurrently"
                    )
                raise RuntimeError("native completion ownership changed")
            record.state = PackedNativeTransferState.COMPLETION_REPORTED
            receipt = self._completion_receipts.issue(
                PackedCompletionReceipt,
                (record.inflight, record.handle, record.cohort),
            )
        if type(receipt) is not PackedCompletionReceipt:
            raise RuntimeError("completion receipt issuer returned wrong type")
        return receipt

    def accept_completion(
        self,
        inflight: PackedInFlightTransfer,
        receipt: PackedCompletionReceipt,
    ) -> PackedCompletedTransfer:
        """Consume endpoint-flush proof into completed-handle ownership.

        :param inflight: Exact in-flight owner.
        :param receipt: Exact one-shot completion receipt.
        :returns: Completed owner retained until teardown.
        """

        with self._lock:
            record = self._require_inflight_locked(inflight)
            if record.state is not PackedNativeTransferState.COMPLETION_REPORTED:
                raise PackedLifecycleError(
                    f"completion admission is invalid in state {record.state.value}"
                )
            self._completion_receipts.consume(
                receipt,
                PackedCompletionReceipt,
                (record.inflight, record.handle, record.cohort),
            )
            completed = PackedCompletedTransfer(
                self._lifecycle_nonce,
                inflight._token,
                _OPAQUE_HANDLE_SEAL,
            )
            record.completed = completed
            record.state = PackedNativeTransferState.COMPLETED
            return completed

    def abandon(
        self,
        owner: PackedPreparedTransfer | PackedInFlightTransfer,
        reason: str,
    ) -> None:
        """Quarantine connection loss or lost lifecycle ownership.

        :param owner: Exact prepared or in-flight owner.
        :param reason: Stable ambiguity reason.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("abandon reason must not be empty")
        with self._lock:
            if type(owner) is PackedPreparedTransfer:
                record = self._require_prepared_locked(owner)
            elif type(owner) is PackedInFlightTransfer:
                record = self._require_inflight_locked(owner)
            else:
                raise TypeError("owner must be prepared or in-flight transfer")
        self._quarantine_record(record, reason)

    def writer_id(
        self,
        completed: PackedCompletedTransfer,
        capability: PackedArenaCapability,
    ) -> StagingWriterId:
        """Return the writer after exact request-capability validation.

        :param completed: Exact completed transfer owner.
        :param capability: Exact request capability.
        :returns: Canonical writer identity.
        """

        with self._lock:
            record = self._require_completed_locked(completed)
            if record.state is not PackedNativeTransferState.COMPLETED:
                raise PackedLifecycleError(
                    f"writer completion is invalid in state {record.state.value}"
                )
            if record.capability is not capability:
                raise PackedLifecycleError(
                    "writer completion belongs to another capability"
                )
            return record.writer_id

    def writer_manifest(
        self,
        capability: PackedArenaCapability,
    ) -> PackedWriterCohortManifest:
        """Return the authority-owned manifest for one exact capability.

        :param capability: Exact request capability.
        :returns: Canonical manifest owned by the capability authority.
        """

        return self._capability_authority.validate(capability).manifest

    def teardown_action(
        self,
        completed: PackedCompletedTransfer,
        capability: PackedArenaCapability,
    ) -> Callable[[object], None]:
        """Build one exact completed-handle release action.

        :param completed: Exact completed transfer.
        :param capability: Exact request capability.
        :returns: Teardown-authorized release callback.
        """

        with self._lock:
            record = self._require_completed_locked(completed)
            if record.state is not PackedNativeTransferState.COMPLETED:
                raise PackedLifecycleError(
                    f"native teardown is invalid in state {record.state.value}"
                )
            if record.capability is not capability:
                raise PackedLifecycleError(
                    "native teardown belongs to another capability"
                )

        def release(finalizer: object) -> None:
            if finalizer is not _AUTHORITY_SEAL:
                raise TypeError("native handle release requires teardown authority")
            with self._lock:
                record = self._require_completed_locked(completed)
                if record.state is PackedNativeTransferState.RELEASED:
                    return
                if record.state is PackedNativeTransferState.QUARANTINED:
                    raise PackedProcessQuarantinedError(
                        "quarantined native handle cannot be released"
                    )
            self._driver.release_completed(record.handle)
            with self._lock:
                if record.state is not PackedNativeTransferState.COMPLETED:
                    raise RuntimeError("native teardown ownership changed")
                record.state = PackedNativeTransferState.RELEASED

        return release

    def state(
        self,
        owner: PackedPreparedTransfer
        | PackedInFlightTransfer
        | PackedCompletedTransfer,
    ) -> PackedNativeTransferState:
        """Return current state for one exact owner.

        :param owner: Prepared, in-flight, or completed owner.
        :returns: Native transfer state.
        """

        with self._lock:
            if type(owner) is PackedPreparedTransfer:
                record = self._require_prepared_locked(owner)
            elif type(owner) is PackedInFlightTransfer:
                record = self._require_inflight_locked(owner)
            elif type(owner) is PackedCompletedTransfer:
                record = self._require_completed_locked(owner)
            else:
                raise TypeError("unsupported native transfer owner")
            return record.state

    def _attest_route(self, handle: object, *, completion: bool) -> None:
        """Validate exact native backend, runtime, lane, and completion chain."""

        backend = self._driver.query_backend(handle)
        if type(backend) is not PackedNixlBackend:
            raise PackedNativeAttestationUnavailable(
                "native backend attestation is not typed"
            )
        if backend is not PackedNixlBackend.UCX:
            raise PackedNativeAttestationUnavailable(
                "exact handle did not select the UCX backend"
            )
        runtime_identity = self._driver.query_runtime_identity(handle)
        _require_exact_bytes(
            runtime_identity,
            PACKED_RUNTIME_IDENTITY_BYTES,
            "native runtime identity",
        )
        if runtime_identity != self._route.runtime_identity:
            raise PackedNativeAttestationUnavailable(
                "native runtime identity differs from the capability route"
            )
        transports = self._driver.query_ucp_transports(handle)
        if transports is None:
            raise PackedNativeAttestationUnavailable(
                "native runtime exposes no per-handle UCP transport proof"
            )
        exact_transports = frozenset(transports)
        if any(
            type(transport) is not PackedUcpTransport for transport in exact_transports
        ):
            raise PackedNativeAttestationUnavailable(
                "native UCP transport proof is not typed"
            )
        if exact_transports != self._route.transports:
            raise PackedNativeAttestationUnavailable(
                "native UCP transports differ from the capability route"
            )
        if not completion:
            return
        contract = self._driver.query_completion_contract(handle)
        if contract is not PackedNativeCompletionContract.NIXL_UCX_ENDPOINT_FLUSH:
            raise PackedNativeAttestationUnavailable(
                "native DONE lacks the exact endpoint-flush completion chain"
            )

    def _quarantine_record(
        self,
        record: _NativeTransferRecord,
        reason: str,
    ) -> None:
        """Quarantine a complete cohort without producing an outcome."""

        with self._lock:
            for cohort_record in self._records.values():
                if cohort_record.cohort is record.cohort:
                    cohort_record.state = PackedNativeTransferState.QUARANTINED
        self._capability_authority._quarantine(
            record.capability,
            _AUTHORITY_SEAL,
        )
        self._quarantine.retain(record.cohort, reason)

    def _require_prepared_locked(
        self,
        prepared: PackedPreparedTransfer,
    ) -> _NativeTransferRecord:
        """Resolve one exact prepared owner while locked."""

        if type(prepared) is not PackedPreparedTransfer:
            raise TypeError("prepared must be PackedPreparedTransfer")
        if prepared._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("prepared transfer belongs elsewhere")
        record = self._records.get(prepared._token)
        if record is None or record.prepared is not prepared:
            raise PackedLifecycleError("prepared transfer is not live")
        return record

    def _require_inflight_locked(
        self,
        inflight: PackedInFlightTransfer,
    ) -> _NativeTransferRecord:
        """Resolve one exact in-flight owner while locked."""

        if type(inflight) is not PackedInFlightTransfer:
            raise TypeError("inflight must be PackedInFlightTransfer")
        if inflight._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("in-flight transfer belongs elsewhere")
        record = self._records.get(inflight._token)
        if record is None or record.inflight is not inflight:
            raise PackedLifecycleError("in-flight transfer is not live")
        return record

    def _require_completed_locked(
        self,
        completed: PackedCompletedTransfer,
    ) -> _NativeTransferRecord:
        """Resolve one exact completed owner while locked."""

        if type(completed) is not PackedCompletedTransfer:
            raise TypeError("completed must be PackedCompletedTransfer")
        if completed._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("completed transfer belongs elsewhere")
        record = self._records.get(completed._token)
        if record is None or record.completed is not completed:
            raise PackedLifecycleError("completed transfer is not live")
        return record


def require_nixl_ucx_native_attestation(agent: object) -> None:
    """Fail unless the native agent exposes exact per-handle route receipts.

    NIXL 1.3.2's public Python surface exposes the selected backend but not the
    UCP transports selected for an exact request and not an opaque completion
    object bound to its endpoint-flush chain. Production packed transfer stays
    gated until a native adapter supplies both methods.

    :param agent: Candidate native NIXL agent wrapper.
    :raises PackedNativeAttestationUnavailable: If exact proof is unavailable.
    """

    if agent is None:
        raise TypeError("agent must not be None")
    methods = vars(type(agent))
    required_methods = (
        "query_xfer_backend",
        "query_xfer_ucp_transports",
        "take_xfer_completion_receipt",
    )
    missing = tuple(
        name
        for name in required_methods
        if name not in methods or not callable(methods[name])
    )
    if len(missing) > 0:
        raise PackedNativeAttestationUnavailable(
            "native NIXL/UCX exact-handle attestation is unavailable: "
            + ", ".join(missing)
        )


class PackedWriterCohortCompletion:
    """All-member completion barrier for an arbitrary writer manifest."""

    _completed_by_writer: dict[StagingWriterId, PackedCompletedTransfer]
    _capability: PackedArenaCapability
    _issued: bool
    _lifecycle: PackedNixlUcxTransferLifecycle
    _lock: threading.Lock
    _manifest: PackedWriterCohortManifest
    _receipts: _ReceiptAuthority

    def __init__(
        self,
        capability: PackedArenaCapability,
        lifecycle: PackedNixlUcxTransferLifecycle,
    ) -> None:
        """Initialize an empty all-writer completion barrier.

        :param capability: Exact request capability owning the writer cohort.
        :param lifecycle: Exact native completion owner.
        """

        if type(capability) is not PackedArenaCapability:
            raise TypeError("capability must be PackedArenaCapability")
        if type(lifecycle) is not PackedNixlUcxTransferLifecycle:
            raise TypeError("lifecycle must be PackedNixlUcxTransferLifecycle")
        manifest = lifecycle.writer_manifest(capability)
        self._completed_by_writer = {}
        self._capability = capability
        self._issued = False
        self._lifecycle = lifecycle
        self._lock = threading.Lock()
        self._manifest = manifest
        self._receipts = _ReceiptAuthority()

    def record(
        self,
        completed: PackedCompletedTransfer,
    ) -> PackedWriterCohortCompletionReceipt | None:
        """Record one exact writer and issue once when all members complete.

        :param completed: Exact native completion.
        :returns: One-shot all-member receipt, or ``None`` before consensus.
        """

        writer_id = self._lifecycle.writer_id(
            completed,
            self._capability,
        )
        self._manifest.projection_for(writer_id)
        with self._lock:
            previous = self._completed_by_writer.get(writer_id)
            if previous is not None:
                if previous is completed:
                    return None
                raise PackedLifecycleError(
                    f"writer completion conflicts for {writer_id}"
                )
            self._completed_by_writer[writer_id] = completed
            expected = tuple(
                projection.writer_id for projection in self._manifest.projections
            )
            if set(self._completed_by_writer) != set(expected):
                return None
            if self._issued:
                return None
            ordered = tuple(
                self._completed_by_writer[writer_id] for writer_id in expected
            )
            receipt = self._receipts.issue(
                PackedWriterCohortCompletionReceipt,
                (self._capability, self._manifest, *ordered),
            )
            self._issued = True
        if type(receipt) is not PackedWriterCohortCompletionReceipt:
            raise RuntimeError("writer cohort issuer returned wrong receipt")
        return receipt

    def consume(
        self,
        receipt: PackedWriterCohortCompletionReceipt,
    ) -> tuple[PackedCompletedTransfer, ...]:
        """Consume exact all-member completion once.

        :param receipt: All-member completion receipt.
        :returns: Completions in canonical writer order.
        """

        with self._lock:
            expected = tuple(
                projection.writer_id for projection in self._manifest.projections
            )
            if set(self._completed_by_writer) != set(expected):
                raise PackedLifecycleError("writer completion cohort is incomplete")
            ordered = tuple(
                self._completed_by_writer[writer_id] for writer_id in expected
            )
            self._receipts.consume(
                receipt,
                PackedWriterCohortCompletionReceipt,
                (self._capability, self._manifest, *ordered),
            )
            return ordered


class PackedScatterEventDriver:
    """Narrow trusted CUDA scatter submission and event progress interface."""

    def submit(
        self,
        destination_lease: object,
        resources: tuple[object, ...],
    ) -> object:
        """Launch scatter and return its exact recorded CUDA event.

        :param destination_lease: Exact destination staging lease.
        :param resources: Exact kernel and allocation owners.
        :returns: Recorded event ordered after every scatter write.
        """

        raise NotImplementedError

    def poll(self, event: object) -> PackedScatterPollState:
        """Poll one exact destination scatter event.

        :param event: Exact native CUDA event.
        :returns: Pending, done, or ambiguous error.
        """

        raise NotImplementedError


class PackedInFlightScatter:
    """Opaque ownership of one exact submitted scatter event."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned scatter owner."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("in-flight scatters are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


class PackedReportedScatter:
    """Opaque ownership of one exact trusted scatter submission."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned reported scatter."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("reported scatters are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


class PackedCompletedScatter:
    """Opaque terminal scatter retained until teardown."""

    __slots__ = ("_lifecycle_nonce", "_token")

    def __init__(
        self,
        lifecycle_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one lifecycle-owned completed scatter."""

        if construction_seal is not _OPAQUE_HANDLE_SEAL:
            raise TypeError("completed scatters are lifecycle owned")
        self._lifecycle_nonce = lifecycle_nonce
        self._token = token


@dataclasses.dataclass
class _ScatterRecord:
    """Private exact-event scatter state."""

    event: object
    destination_lease: object
    cohort: PackedResourceCohort
    state: PackedScatterState
    resources: tuple[object, ...]
    reported: PackedReportedScatter
    inflight: PackedInFlightScatter | None = None
    completed: PackedCompletedScatter | None = None


class PackedScatterLifecycle:
    """Exact-event scatter submission, terminality, and quarantine owner."""

    _completion_receipts: _ReceiptAuthority
    _driver: PackedScatterEventDriver
    _lifecycle_nonce: object
    _lock: threading.Lock
    _quarantine: PackedProcessLifetimeQuarantine
    _records: dict[object, _ScatterRecord]
    _submit_receipts: _ReceiptAuthority

    def __init__(
        self,
        *,
        driver: PackedScatterEventDriver,
        quarantine: PackedProcessLifetimeQuarantine,
    ) -> None:
        """Initialize exact-event scatter ownership.

        :param driver: Trusted CUDA event progress adapter.
        :param quarantine: Process-lifetime ambiguity retention.
        """

        if not isinstance(driver, PackedScatterEventDriver):
            raise TypeError("driver must derive from PackedScatterEventDriver")
        if type(quarantine) is not PackedProcessLifetimeQuarantine:
            raise TypeError("quarantine must be PackedProcessLifetimeQuarantine")
        self._completion_receipts = _ReceiptAuthority()
        self._driver = driver
        self._lifecycle_nonce = object()
        self._lock = threading.Lock()
        self._quarantine = quarantine
        self._records = {}
        self._submit_receipts = _ReceiptAuthority()

    def submit(
        self,
        *,
        destination_lease: object,
        resources: tuple[object, ...],
        cohort: PackedResourceCohort,
    ) -> tuple[PackedReportedScatter, PackedScatterReceipt]:
        """Launch through the trusted driver and report exact event ownership.

        :param destination_lease: Exact destination staging lease.
        :param resources: Temporary kernel resources retained by the event.
        :param cohort: Complete local cohort.
        :returns: Opaque reported owner and one-shot submission receipt.
        """

        if destination_lease is None:
            raise ValueError("destination_lease must not be None")
        if type(cohort) is not PackedResourceCohort:
            raise TypeError("cohort must be PackedResourceCohort")
        owned_resources = tuple(resources)
        if len(owned_resources) == 0:
            raise ValueError("scatter resources must not be empty")
        reservation = cohort._reserve_scatter_submission(
            destination_lease=destination_lease,
            resources=owned_resources,
            authority=_AUTHORITY_SEAL,
        )
        try:
            event = self._driver.submit(
                destination_lease,
                owned_resources,
            )
            if event is None:
                raise TypeError("scatter driver returned no exact event")
            cohort._bind_scatter_event(
                reservation=reservation,
                event=event,
                authority=_AUTHORITY_SEAL,
            )
        except BaseException as error:
            _log_ambiguous_failure("packed scatter submission failed")
            self._quarantine.retain(
                cohort,
                "scatter submission did not prove exact-event ownership",
            )
            if not isinstance(error, Exception):
                raise
            raise PackedAmbiguousTransportError(
                "scatter submission is ambiguous"
            ) from error
        token = object()
        reported = PackedReportedScatter(
            self._lifecycle_nonce,
            token,
            _OPAQUE_HANDLE_SEAL,
        )
        record = _ScatterRecord(
            event=event,
            destination_lease=destination_lease,
            cohort=cohort,
            state=PackedScatterState.SUBMISSION_REPORTED,
            resources=owned_resources,
            reported=reported,
        )
        with self._lock:
            self._records[token] = record
            receipt = self._submit_receipts.issue(
                PackedScatterReceipt,
                (reported, event, destination_lease, cohort),
            )
        if type(receipt) is not PackedScatterReceipt:
            raise RuntimeError("scatter issuer returned wrong receipt")
        return reported, receipt

    def accept_submission(
        self,
        reported: PackedReportedScatter,
        receipt: PackedScatterReceipt,
    ) -> PackedInFlightScatter:
        """Consume one exact scatter receipt into in-flight event ownership.

        :param reported: Exact trusted submission owner.
        :param receipt: Exact one-shot scatter receipt.
        :returns: Opaque in-flight scatter owner.
        """

        with self._lock:
            record = self._require_reported_locked(reported)
            if record.state is not PackedScatterState.SUBMISSION_REPORTED:
                raise PackedLifecycleError(
                    f"scatter admission is invalid in state {record.state.value}"
                )
            self._submit_receipts.consume(
                receipt,
                PackedScatterReceipt,
                (
                    reported,
                    record.event,
                    record.destination_lease,
                    record.cohort,
                ),
            )
            inflight = PackedInFlightScatter(
                self._lifecycle_nonce,
                reported._token,
                _OPAQUE_HANDLE_SEAL,
            )
            record.inflight = inflight
            record.state = PackedScatterState.IN_FLIGHT
            return inflight

    def poll(
        self,
        inflight: PackedInFlightScatter,
    ) -> PackedScatterCompletionReceipt | None:
        """Poll one exact scatter event and quarantine ambiguous failure.

        :param inflight: Exact in-flight event owner.
        :returns: Completion receipt, or ``None`` while pending.
        """

        with self._lock:
            record = self._require_inflight_locked(inflight)
            if record.state is not PackedScatterState.IN_FLIGHT:
                raise PackedLifecycleError(
                    f"scatter poll is invalid in state {record.state.value}"
                )
        try:
            poll_state = self._driver.poll(record.event)
            if type(poll_state) is not PackedScatterPollState:
                raise TypeError("scatter driver returned an invalid state")
            if poll_state is PackedScatterPollState.ERROR:
                raise PackedAmbiguousTransportError(
                    "scatter event returned ERROR without quiescence"
                )
            if poll_state is PackedScatterPollState.PENDING:
                return None
        except BaseException as error:
            _log_ambiguous_failure("packed scatter progress failed")
            with self._lock:
                record.state = PackedScatterState.QUARANTINED
            self._quarantine.retain(
                record.cohort,
                "scatter event terminality is ambiguous",
            )
            if not isinstance(error, Exception):
                raise
            raise PackedAmbiguousTransportError("scatter event is ambiguous") from error

        with self._lock:
            if record.state is not PackedScatterState.IN_FLIGHT:
                raise RuntimeError("scatter completion ownership changed")
            record.state = PackedScatterState.COMPLETION_REPORTED
            receipt = self._completion_receipts.issue(
                PackedScatterCompletionReceipt,
                (
                    record.inflight,
                    record.event,
                    record.destination_lease,
                    record.cohort,
                ),
            )
        if type(receipt) is not PackedScatterCompletionReceipt:
            raise RuntimeError("scatter completion issuer returned wrong receipt")
        return receipt

    def accept_completion(
        self,
        inflight: PackedInFlightScatter,
        receipt: PackedScatterCompletionReceipt,
    ) -> PackedCompletedScatter:
        """Consume one exact-event terminal receipt.

        :param inflight: Exact in-flight scatter.
        :param receipt: Exact one-shot completion receipt.
        :returns: Completed scatter retained until teardown.
        """

        with self._lock:
            record = self._require_inflight_locked(inflight)
            if record.state is not PackedScatterState.COMPLETION_REPORTED:
                raise PackedLifecycleError(
                    f"scatter completion is invalid in state {record.state.value}"
                )
            self._completion_receipts.consume(
                receipt,
                PackedScatterCompletionReceipt,
                (
                    record.inflight,
                    record.event,
                    record.destination_lease,
                    record.cohort,
                ),
            )
            completed = PackedCompletedScatter(
                self._lifecycle_nonce,
                inflight._token,
                _OPAQUE_HANDLE_SEAL,
            )
            record.completed = completed
            record.state = PackedScatterState.COMPLETED
            return completed

    def _release_after_teardown(
        self,
        *,
        completed: PackedCompletedScatter,
        cohort: PackedResourceCohort,
        authority: object,
    ) -> None:
        """Release completed scatter ownership after the ack barrier."""

        if authority is not _AUTHORITY_SEAL:
            raise TypeError("scatter release requires teardown authority")
        with self._lock:
            record = self._require_completed_locked(completed)
            if record.cohort is not cohort:
                raise PackedLifecycleError(
                    "completed scatter belongs to another resource cohort"
                )
            if record.state is PackedScatterState.RELEASED:
                return
            if record.state is PackedScatterState.QUARANTINED:
                raise PackedProcessQuarantinedError(
                    "quarantined scatter cannot be released"
                )
            if record.state is not PackedScatterState.COMPLETED:
                raise PackedLifecycleError(
                    f"scatter release is invalid in state {record.state.value}"
                )
            record.state = PackedScatterState.RELEASED

    def validate_completed(
        self,
        completed: PackedCompletedScatter,
        cohort: PackedResourceCohort,
    ) -> None:
        """Validate exact completed scatter and resource-cohort binding.

        :param completed: Exact completed scatter owner.
        :param cohort: Exact resource cohort.
        """

        with self._lock:
            record = self._require_completed_locked(completed)
            if record.cohort is not cohort:
                raise PackedLifecycleError(
                    "completed scatter belongs to another resource cohort"
                )
            if record.state is not PackedScatterState.COMPLETED:
                raise PackedLifecycleError(
                    f"scatter is invalid in state {record.state.value}"
                )

    def _require_inflight_locked(
        self,
        inflight: PackedInFlightScatter,
    ) -> _ScatterRecord:
        """Resolve one exact scatter owner while locked."""

        if type(inflight) is not PackedInFlightScatter:
            raise TypeError("inflight must be PackedInFlightScatter")
        if inflight._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("scatter belongs to another lifecycle")
        record = self._records.get(inflight._token)
        if record is None or record.inflight is not inflight:
            raise PackedLifecycleError("scatter is not live")
        return record

    def _require_reported_locked(
        self,
        reported: PackedReportedScatter,
    ) -> _ScatterRecord:
        """Resolve one exact reported scatter owner while locked."""

        if type(reported) is not PackedReportedScatter:
            raise TypeError("reported must be PackedReportedScatter")
        if reported._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("scatter belongs to another lifecycle")
        record = self._records.get(reported._token)
        if record is None or record.reported is not reported:
            raise PackedLifecycleError("reported scatter is not live")
        return record

    def _require_completed_locked(
        self,
        completed: PackedCompletedScatter,
    ) -> _ScatterRecord:
        """Resolve one exact completed scatter owner while locked."""

        if type(completed) is not PackedCompletedScatter:
            raise TypeError("completed must be PackedCompletedScatter")
        if completed._lifecycle_nonce is not self._lifecycle_nonce:
            raise PackedLifecycleError("scatter belongs to another lifecycle")
        record = self._records.get(completed._token)
        if record is None or record.completed is not completed:
            raise PackedLifecycleError("completed scatter is not live")
        return record


@dataclasses.dataclass(frozen=True)
class PackedTeardownRequest:
    """Destination request to retire one exact writer transfer.

    :ivar key: Exact request and chunk identity.
    :ivar writer_id: Exact cohort writer.
    :ivar capability_id: Exact arena capability identity.
    :ivar teardown_generation: One-shot teardown barrier generation.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    capability_id: bytes
    teardown_generation: bytes

    def __post_init__(self) -> None:
        """Validate bounded teardown request data."""

        if type(self.key) is not PackedChunkKey:
            raise TypeError("teardown key must be PackedChunkKey")
        if type(self.writer_id) is not StagingWriterId:
            raise TypeError("teardown writer_id must be StagingWriterId")
        _require_exact_bytes(
            self.capability_id,
            PACKED_CAPABILITY_ID_BYTES,
            "teardown capability_id",
        )
        _require_exact_bytes(
            self.teardown_generation,
            PACKED_GENERATION_BYTES,
            "teardown_generation",
        )


@dataclasses.dataclass(frozen=True)
class PackedTeardownAck:
    """Authenticated writer acknowledgement of exact teardown.

    :ivar key: Exact request and chunk identity.
    :ivar writer_id: Claimed writer identity.
    :ivar capability_id: Exact arena capability identity.
    :ivar teardown_generation: Exact teardown barrier generation.
    """

    key: PackedChunkKey
    writer_id: StagingWriterId
    capability_id: bytes
    teardown_generation: bytes

    def __post_init__(self) -> None:
        """Validate bounded teardown acknowledgement data."""

        if type(self.key) is not PackedChunkKey:
            raise TypeError("teardown key must be PackedChunkKey")
        if type(self.writer_id) is not StagingWriterId:
            raise TypeError("teardown writer_id must be StagingWriterId")
        _require_exact_bytes(
            self.capability_id,
            PACKED_CAPABILITY_ID_BYTES,
            "teardown capability_id",
        )
        _require_exact_bytes(
            self.teardown_generation,
            PACKED_GENERATION_BYTES,
            "teardown_generation",
        )


class PackedTeardownCoordinator:
    """All-writer ack barrier preceding handle release, revocation, and reuse."""

    _acknowledged: set[StagingWriterId]
    _barrier_receipts: _ReceiptAuthority
    _capability: PackedArenaCapability
    _capability_authority: PackedArenaCapabilityAuthority
    _cohort: PackedResourceCohort
    _completed_scatter: PackedCompletedScatter
    _key: PackedChunkKey
    _lock: threading.Lock
    _page_leases: tuple[
        tuple[PackedPageLeaseAllocator, PackedPageLease],
        ...,
    ]
    _peers: tuple[PackedAuthenticatedPeer, ...]
    _quarantine: PackedProcessLifetimeQuarantine
    _release_actions: tuple[Callable[[object], None], ...]
    _requests: tuple[PackedTeardownRequest, ...]
    _scatter_lifecycle: PackedScatterLifecycle
    _state: PackedTeardownState

    def __init__(
        self,
        *,
        key: PackedChunkKey,
        capability: PackedArenaCapability,
        capability_authority: PackedArenaCapabilityAuthority,
        page_leases: tuple[
            tuple[PackedPageLeaseAllocator, PackedPageLease],
            ...,
        ],
        completed_transfers: tuple[PackedCompletedTransfer, ...],
        native_lifecycle: PackedNixlUcxTransferLifecycle,
        completed_scatter: PackedCompletedScatter,
        scatter_lifecycle: PackedScatterLifecycle,
        cohort: PackedResourceCohort,
        quarantine: PackedProcessLifetimeQuarantine,
    ) -> None:
        """Initialize one exact request teardown barrier.

        :param key: Exact request and chunk identity.
        :param capability: Arena capability retained through teardown.
        :param capability_authority: Capability owner.
        :param page_leases: Source and destination leases retained through ack.
        :param completed_transfers: Canonically ordered exact native completions.
        :param native_lifecycle: Exact native completion owner.
        :param completed_scatter: Exact terminal destination scatter.
        :param scatter_lifecycle: Exact scatter completion owner.
        :param cohort: Complete resource cohort.
        :param quarantine: Process-lifetime failure retention.
        """

        if type(key) is not PackedChunkKey:
            raise TypeError("key must be PackedChunkKey")
        record = capability_authority.validate(capability)
        if record.manifest.request_generation != key.request_generation:
            raise ValueError("teardown key generation differs from capability")
        expected_writers, expected_page_lease_bindings = (
            capability_authority._resource_cohort_details(
                cohort=cohort,
                capability=capability,
            )
        )
        owned_page_leases = tuple(page_leases)
        if len(owned_page_leases) == 0:
            raise ValueError("teardown must own at least one page lease")
        for allocator, lease in owned_page_leases:
            if type(allocator) is not PackedPageLeaseAllocator:
                raise TypeError("page lease allocator has an invalid type")
            allocator.pages(lease)
        if len(expected_page_lease_bindings) != len(owned_page_leases) or any(
            expected_allocator is not actual_allocator
            or expected_lease is not actual_lease
            for (
                expected_allocator,
                expected_lease,
            ), (
                actual_allocator,
                actual_lease,
            ) in zip(
                expected_page_lease_bindings,
                owned_page_leases,
                strict=True,
            )
        ):
            raise PackedLifecycleError(
                "teardown page leases differ from the resource cohort"
            )
        if type(native_lifecycle) is not PackedNixlUcxTransferLifecycle:
            raise TypeError("native_lifecycle must be PackedNixlUcxTransferLifecycle")
        owned_completed_transfers = tuple(completed_transfers)
        if len(owned_completed_transfers) != len(expected_writers):
            raise ValueError("teardown must own one native completion per writer")
        for expected_writer, completed in zip(
            expected_writers,
            owned_completed_transfers,
            strict=True,
        ):
            actual_writer = native_lifecycle.writer_id(
                completed,
                capability,
            )
            if actual_writer != expected_writer:
                raise PackedLifecycleError(
                    "native completions differ from canonical writer order"
                )
        owned_release_actions = tuple(
            native_lifecycle.teardown_action(
                completed,
                capability,
            )
            for completed in owned_completed_transfers
        )
        if type(scatter_lifecycle) is not PackedScatterLifecycle:
            raise TypeError("scatter_lifecycle must be PackedScatterLifecycle")
        scatter_lifecycle.validate_completed(completed_scatter, cohort)
        self._acknowledged = set()
        self._barrier_receipts = _ReceiptAuthority()
        self._capability = capability
        self._capability_authority = capability_authority
        self._cohort = cohort
        self._completed_scatter = completed_scatter
        self._key = key
        self._lock = threading.Lock()
        self._page_leases = owned_page_leases
        self._peers = record.peers
        self._quarantine = quarantine
        self._release_actions = owned_release_actions
        self._requests = ()
        self._scatter_lifecycle = scatter_lifecycle
        self._state = PackedTeardownState.CREATED

    def begin(self) -> tuple[PackedTeardownRequest, ...]:
        """Create one exact request per authenticated writer.

        :returns: Teardown requests in canonical manifest order.
        """

        with self._lock:
            if self._state is not PackedTeardownState.CREATED:
                raise PackedLifecycleError(
                    f"teardown cannot begin in state {self._state.value}"
                )
            record = self._capability_authority.validate(self._capability)
            generation = secrets.token_bytes(PACKED_GENERATION_BYTES)
            self._requests = tuple(
                PackedTeardownRequest(
                    key=self._key,
                    writer_id=projection.writer_id,
                    capability_id=record.grant.capability_id,
                    teardown_generation=generation,
                )
                for projection in record.manifest.projections
            )
            self._state = PackedTeardownState.WAITING_FOR_ACKS
            return self._requests

    def acknowledge(
        self,
        message: PackedTeardownAck,
        peer: PackedAuthenticatedPeer,
    ) -> PackedTeardownBarrierReceipt | None:
        """Authenticate one exact ack and issue once after all writers.

        :param message: Untrusted wire acknowledgement.
        :param peer: Exact transport-authenticated sender.
        :returns: Barrier receipt after final unique acknowledgement.
        """

        if type(message) is not PackedTeardownAck:
            raise TypeError("message must be PackedTeardownAck")
        with self._lock:
            if self._state is not PackedTeardownState.WAITING_FOR_ACKS:
                raise PackedLifecycleError(
                    f"teardown ack is invalid in state {self._state.value}"
                )
            record = self._capability_authority.validate(
                self._capability,
                peer,
            )
            peer_index = record.peers.index(peer)
            expected = self._requests[peer_index]
            if message.writer_id != expected.writer_id:
                raise PackedLifecycleError(
                    "teardown writer differs from authenticated peer"
                )
            if message != PackedTeardownAck(
                key=expected.key,
                writer_id=expected.writer_id,
                capability_id=expected.capability_id,
                teardown_generation=expected.teardown_generation,
            ):
                raise PackedLifecycleError(
                    "teardown acknowledgement differs from request"
                )
            if message.writer_id in self._acknowledged:
                return None
            self._acknowledged.add(message.writer_id)
            expected_writers = {request.writer_id for request in self._requests}
            if self._acknowledged != expected_writers:
                return None
            receipt = self._barrier_receipts.issue(
                PackedTeardownBarrierReceipt,
                (self._capability, *self._requests),
            )
            self._state = PackedTeardownState.BARRIER_ISSUED
        if type(receipt) is not PackedTeardownBarrierReceipt:
            raise RuntimeError("teardown issuer returned wrong receipt")
        return receipt

    def finalize(self, receipt: PackedTeardownBarrierReceipt) -> None:
        """Consume the ack barrier, then release and revoke exact resources.

        Native completed handles are released before page leases or the arena
        capability become reusable. Any failure retains the complete cohort.

        :param receipt: Exact all-writer teardown barrier.
        """

        with self._lock:
            if self._state is not PackedTeardownState.BARRIER_ISSUED:
                raise PackedLifecycleError(
                    f"teardown finalization is invalid in state {self._state.value}"
                )
            self._barrier_receipts.consume(
                receipt,
                PackedTeardownBarrierReceipt,
                (self._capability, *self._requests),
            )
        try:
            self._capability_authority._begin_teardown(
                self._capability,
                _AUTHORITY_SEAL,
            )
            for release_action in self._release_actions:
                release_action(_AUTHORITY_SEAL)
            self._scatter_lifecycle._release_after_teardown(
                completed=self._completed_scatter,
                cohort=self._cohort,
                authority=_AUTHORITY_SEAL,
            )
            for allocator, lease in self._page_leases:
                allocator._release_after_teardown(lease, _AUTHORITY_SEAL)
            self._capability_authority._revoke_after_teardown(
                self._capability,
                _AUTHORITY_SEAL,
            )
        except BaseException as error:
            _log_ambiguous_failure("packed teardown release failed")
            with self._lock:
                self._state = PackedTeardownState.QUARANTINED
            self._capability_authority._quarantine(
                self._capability,
                _AUTHORITY_SEAL,
            )
            for allocator, lease in self._page_leases:
                allocator._quarantine(lease, _AUTHORITY_SEAL)
            self._quarantine.retain(
                self._cohort,
                "teardown release failed after all-writer acknowledgement",
            )
            if not isinstance(error, Exception):
                raise
            raise PackedProcessQuarantinedError(
                "teardown release failed; cohort is quarantined"
            ) from error
        with self._lock:
            self._state = PackedTeardownState.RELEASED

    def connection_lost(
        self,
        peer: PackedAuthenticatedPeer,
    ) -> None:
        """Quarantine the complete cohort when an ack becomes unprovable.

        :param peer: Exact authenticated peer whose connection was lost.
        """

        with self._lock:
            if self._state is not PackedTeardownState.WAITING_FOR_ACKS:
                raise PackedLifecycleError(
                    f"connection loss is invalid in state {self._state.value}"
                )
            record = self._capability_authority.validate(
                self._capability,
                peer,
            )
            if peer not in record.peers:
                raise PackedLifecycleError("connection peer is absent")
            self._state = PackedTeardownState.QUARANTINED
        self._capability_authority._quarantine(
            self._capability,
            _AUTHORITY_SEAL,
        )
        for allocator, lease in self._page_leases:
            allocator._quarantine(lease, _AUTHORITY_SEAL)
        self._quarantine.retain(
            self._cohort,
            "teardown acknowledgement became unprovable after connection loss",
        )

    @property
    def state(self) -> PackedTeardownState:
        """Return current teardown state.

        :returns: Teardown lifecycle state.
        """

        with self._lock:
            return self._state
