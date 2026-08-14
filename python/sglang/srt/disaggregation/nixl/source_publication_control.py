import dataclasses
import hashlib
import threading
from collections.abc import Callable

from sglang.srt.disaggregation.nixl.startup_source_roster import (
    TerminalNixlSourceRoster,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationResult,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import TerminalReceipt
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceipt,
    TerminalWireReceiptImportNamespace,
)
from sglang.srt.utils.network import NetworkAddress

TERMINAL_SOURCE_PUBLICATION_RECEIPT_TAG: bytes = (
    b"TERMINAL_SOURCE_PUBLICATION_RECEIPT_V1"
)
_TERMINAL_SOURCE_PUBLICATION_FRAME_COUNT: int = 4


class TerminalSourcePublicationControlError(RuntimeError):
    """Source-rank publication control or lifecycle invariant violation."""


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSourcePublicationRoute:
    """One exact source owner and its actual control listener.

    :ivar startup_rank: Generation-bound source row from the sealed matrix.
    :ivar endpoint: Manager-owned PULL listener advertised by that exact rank.
    """

    startup_rank: TerminalStartupRankAdvertisement
    endpoint: NetworkAddress

    def __post_init__(self) -> None:
        """Validate one source-only control route."""

        if type(self.startup_rank) is not TerminalStartupRankAdvertisement:
            raise TypeError("startup_rank must be TerminalStartupRankAdvertisement")
        if self.startup_rank.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("publication control routes require source ranks")
        if type(self.endpoint) is not NetworkAddress:
            raise TypeError("endpoint must be NetworkAddress")
        if type(self.endpoint.host) is not str or len(self.endpoint.host) == 0:
            raise ValueError("control endpoint host must be nonempty")
        try:
            self.endpoint.host.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("control endpoint host must be ASCII") from error
        if type(self.endpoint.port) is not int or not 1 <= self.endpoint.port <= 65535:
            raise ValueError("control endpoint port is invalid")

    @property
    def identity(self) -> TerminalProcessIdentity:
        """Return the exact process identity authenticated by this route.

        :returns: Generation-bound source process identity.
        """

        return self.startup_rank.terminal_identity


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSourcePublicationRouteRoster:
    """Complete source service control routes frozen at startup.

    :ivar startup_matrix_sha256: Exact sealed startup matrix digest.
    :ivar source_service_id: Sole source service represented by the roster.
    :ivar routes: Canonical TP-rank ordered source control routes.
    """

    startup_matrix_sha256: bytes
    source_service_id: str
    routes: tuple[TerminalSourcePublicationRoute, ...]

    def __post_init__(self) -> None:
        """Validate complete canonical membership and endpoint uniqueness."""

        if (
            type(self.startup_matrix_sha256) is not bytes
            or len(self.startup_matrix_sha256) != hashlib.sha256().digest_size
        ):
            raise ValueError("startup_matrix_sha256 must contain 32 bytes")
        if type(self.source_service_id) is not str or len(self.source_service_id) == 0:
            raise ValueError("source_service_id must be nonempty")
        if type(self.routes) is not tuple or len(self.routes) == 0:
            raise ValueError("routes must be a nonempty tuple")
        if any(
            type(route) is not TerminalSourcePublicationRoute for route in self.routes
        ):
            raise TypeError("routes must contain TerminalSourcePublicationRoute values")

        ranks = tuple(route.startup_rank for route in self.routes)
        tp_size = ranks[0].tensor_parallel_size
        if len(ranks) != tp_size:
            raise ValueError("source control roster is incomplete")
        if any(
            rank.service_id != self.source_service_id
            or rank.role is not TerminalOwnerRole.SOURCE
            or rank.tensor_parallel_size != tp_size
            for rank in ranks
        ):
            raise ValueError("source control roster spans services or TP widths")
        if tuple(rank.tensor_parallel_rank for rank in ranks) != tuple(range(tp_size)):
            raise ValueError("source control routes must use canonical TP-rank order")
        endpoints = tuple(
            (route.endpoint.host, route.endpoint.port) for route in self.routes
        )
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("source control endpoints must be unique")

    @classmethod
    def from_startup_roster(
        cls,
        binding: TerminalStartupRankBinding,
        roster: TerminalNixlSourceRoster,
        local_endpoint: NetworkAddress,
    ) -> "TerminalSourcePublicationRouteRoster":
        """Freeze actual source listeners against one sealed source rank.

        The endpoint is taken only from the manager route registration. The
        service HTTP origin never participates in control routing.

        :param binding: Exact local startup authority.
        :param roster: Complete manager-advertised source route population.
        :param local_endpoint: Actual listener owned by this local manager.
        :returns: Complete immutable source publication route roster.
        :raises TerminalSourcePublicationControlError: If any route differs
            from the startup matrix or the local listener.
        """

        if type(binding) is not TerminalStartupRankBinding:
            raise TypeError("binding must be TerminalStartupRankBinding")
        if type(roster) is not TerminalNixlSourceRoster:
            raise TypeError("roster must be TerminalNixlSourceRoster")
        if type(local_endpoint) is not NetworkAddress:
            raise TypeError("local_endpoint must be NetworkAddress")
        local = binding.advertisement
        if local.role is not TerminalOwnerRole.SOURCE:
            raise TerminalSourcePublicationControlError(
                "only a source rank can enroll source publication routes"
            )
        try:
            roster.require_matrix(
                binding.matrix,
                local,
                roster.routes[0].transport_protocol,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise TerminalSourcePublicationControlError(
                "source control roster differs from the sealed startup matrix"
            ) from error

        source_ranks = tuple(
            rank
            for rank in binding.matrix.ranks
            if rank.role is TerminalOwnerRole.SOURCE
        )
        if len(roster.routes) != len(source_ranks):
            raise TerminalSourcePublicationControlError(
                "source control roster is incomplete"
            )
        routes = tuple(
            TerminalSourcePublicationRoute(
                startup_rank=rank,
                endpoint=NetworkAddress(route.rank_ip, route.rank_port),
            )
            for rank, route in zip(source_ranks, roster.routes, strict=True)
        )
        route_roster = cls(
            startup_matrix_sha256=binding.matrix.digest,
            source_service_id=local.service_id,
            routes=routes,
        )
        local_route = route_roster.route_for(local.terminal_identity)
        if local_route.endpoint != local_endpoint:
            raise TerminalSourcePublicationControlError(
                "source control roster conflicts with the local manager listener"
            )
        return route_roster

    @property
    def canonical_identity(self) -> TerminalProcessIdentity:
        """Return the sole source rank allowed to publish gateway outcomes.

        :returns: Generation-bound source rank-zero identity.
        """

        return self.routes[0].identity

    @property
    def identities(self) -> tuple[TerminalProcessIdentity, ...]:
        """Return canonical source process membership.

        :returns: One exact identity per source TP rank.
        """

        return tuple(route.identity for route in self.routes)

    def route_for(
        self,
        identity: TerminalProcessIdentity,
    ) -> TerminalSourcePublicationRoute:
        """Resolve one exact-generation source route.

        :param identity: Source identity selected by a request binding.
        :returns: Sole matching frozen route.
        :raises TerminalSourcePublicationControlError: If identity is stale or
            absent.
        """

        if type(identity) is not TerminalProcessIdentity:
            raise TypeError("identity must be TerminalProcessIdentity")
        matches = tuple(route for route in self.routes if route.identity == identity)
        if len(matches) != 1:
            raise TerminalSourcePublicationControlError(
                "source identity is absent or stale in the control roster"
            )
        return matches[0]


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSourcePublicationDelivery:
    """Imported publication authority ready for one source owner.

    :ivar wire_receipt: Exact bytes received through the canonical route.
    :ivar local_receipt: Process-local authority minted by the importer.
    :ivar authenticated_issuer: Canonical source identity proved by enrollment.
    """

    wire_receipt: TerminalWireReceipt
    local_receipt: TerminalReceipt
    authenticated_issuer: TerminalProcessIdentity

    def __post_init__(self) -> None:
        """Validate public receipt fields and canonical issuer agreement."""

        if type(self.wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(self.local_receipt) is not TerminalReceipt:
            raise TypeError("local_receipt must be TerminalReceipt")
        if type(self.authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if self.wire_receipt.issuer != self.authenticated_issuer:
            raise ValueError("wire receipt asserts another authenticated issuer")
        shared_fields = (
            self.wire_receipt.binding == self.local_receipt.binding,
            self.wire_receipt.kind is self.local_receipt.kind,
            self.wire_receipt.outcome is self.local_receipt.outcome,
            self.wire_receipt.terminal_timestamp_ns
            == self.local_receipt.terminal_timestamp_ns,
        )
        if not all(shared_fields):
            raise ValueError("local publication authority differs from wire receipt")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSourcePublicationControlInventory:
    """Lifecycle inventory for one process-local publication route owner.

    :ivar route_count: Exact frozen source route population.
    :ivar active_binding_digests: Registered request generations.
    :ivar terminal_binding_digests: Generations with one accepted outcome.
    :ivar listener_bound: Whether local lifecycle delivery is installed.
    :ivar closed: Whether clean route teardown completed.
    :ivar fatal_reason: Sticky route or listener failure, when present.
    """

    route_count: int
    active_binding_digests: tuple[bytes, ...]
    terminal_binding_digests: tuple[bytes, ...]
    listener_bound: bool
    closed: bool
    fatal_reason: str | None


TerminalSourcePublicationFrameSender = Callable[
    [NetworkAddress, tuple[bytes, ...]],
    None,
]
TerminalSourcePublicationListener = Callable[[TerminalSourcePublicationDelivery], None]
TerminalSourcePublicationFatalHandler = Callable[[str], None]


class TerminalSourcePublicationControl:
    """Direct point-to-point source publication receipt owner.

    The canonical source rank fans publisher outcomes directly to every source
    manager listener. Noncanonical ranks import only rank-zero receipts through
    the generation-bound startup roster. This component owns no progress thread
    and introduces neither polling nor a collective.
    """

    _roster: TerminalSourcePublicationRouteRoster
    _local_identity: TerminalProcessIdentity
    _send_frames: TerminalSourcePublicationFrameSender
    _importer: TerminalWireReceiptImportNamespace | None
    _listener: TerminalSourcePublicationListener | None
    _process_fatal_handler: TerminalSourcePublicationFatalHandler | None
    _active: dict[bytes, TerminalRequestBinding]
    _terminal: dict[bytes, TerminalWireReceipt]
    _closed: bool
    _fatal_reason: str | None
    _fatal_notified: bool
    _lock: threading.Lock

    def __init__(
        self,
        roster: TerminalSourcePublicationRouteRoster,
        local_identity: TerminalProcessIdentity,
        send_frames: TerminalSourcePublicationFrameSender,
    ) -> None:
        """Construct a dormant route owner over one frozen source cohort.

        :param roster: Exact startup-enrolled source listeners.
        :param local_identity: Process identity owning this route instance.
        :param send_frames: Manager-owned point-to-point send operation.
        """

        if type(roster) is not TerminalSourcePublicationRouteRoster:
            raise TypeError("roster must be TerminalSourcePublicationRouteRoster")
        if type(local_identity) is not TerminalProcessIdentity:
            raise TypeError("local_identity must be TerminalProcessIdentity")
        roster.route_for(local_identity)
        if not callable(send_frames):
            raise TypeError("send_frames must be callable")
        self._roster = roster
        self._local_identity = local_identity
        self._send_frames = send_frames
        self._importer = (
            None
            if local_identity == roster.canonical_identity
            else TerminalWireReceiptImportNamespace(roster.canonical_identity)
        )
        self._listener = None
        self._process_fatal_handler = None
        self._active = {}
        self._terminal = {}
        self._closed = False
        self._fatal_reason = None
        self._fatal_notified = False
        self._lock = threading.Lock()

    @property
    def roster(self) -> TerminalSourcePublicationRouteRoster:
        """Return immutable route authority.

        :returns: Complete source control roster.
        """

        return self._roster

    def bind_listener(
        self,
        listener: TerminalSourcePublicationListener,
        process_fatal_handler: TerminalSourcePublicationFatalHandler,
    ) -> None:
        """Bind lifecycle delivery and its process-fatal owner together.

        :param listener: Callback consuming imported publication authority.
        :param process_fatal_handler: Callback which closes admission and
            quarantines every live source identity after route failure.
        """

        if not callable(listener):
            raise TypeError("listener must be callable")
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        with self._lock:
            self._require_open_locked()
            if self._listener is not None:
                raise TerminalSourcePublicationControlError(
                    "source publication listener is already bound"
                )
            if len(self._active) > 0:
                raise TerminalSourcePublicationControlError(
                    "source publication listener must bind before requests"
                )
            self._listener = listener
            self._process_fatal_handler = process_fatal_handler

    def register_binding(self, binding: TerminalRequestBinding) -> None:
        """Register one local request before publication receipt admission.

        :param binding: Exact local source lifecycle binding.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if binding.owner != self._local_identity:
            raise TerminalSourcePublicationControlError(
                "publication control binding belongs to another source owner"
            )
        digest = binding.digest
        fatal_reason: str | None = None
        fatal_cause: RuntimeError | TypeError | ValueError | None = None
        with self._lock:
            self._require_ready_locked()
            current = self._active.get(digest)
            if current is not None:
                if current == binding:
                    return
                fatal_reason = (
                    "publication binding digest was reused with conflicting fields"
                )
                self._enter_fatal_locked(fatal_reason)
            else:
                try:
                    importer = self._importer
                    if importer is not None:
                        importer.register_binding(binding)
                except (RuntimeError, TypeError, ValueError) as error:
                    fatal_reason = "publication importer rejected a registered binding"
                    fatal_cause = error
                    self._enter_fatal_locked(fatal_reason)
                else:
                    self._active[digest] = binding
        if fatal_reason is not None:
            self._notify_process_fatal(fatal_reason)
            raise TerminalSourcePublicationControlError(fatal_reason) from fatal_cause

    def publish_result(self, result: TerminalGatewayPublicationResult) -> None:
        """Fan one canonical publisher result directly to every source rank.

        :param result: Exact success or failure minted by the gateway publisher.
        """

        if type(result) not in (
            TerminalGatewayPublicationSuccess,
            TerminalGatewayPublicationFailure,
        ):
            raise TypeError("result must be a terminal gateway publication result")
        if self._local_identity != self._roster.canonical_identity:
            raise TerminalSourcePublicationControlError(
                "only canonical source rank zero can fan out publication results"
            )
        publication = result.publication
        expected_bindings = publication.source_bindings
        if tuple(binding.owner for binding in expected_bindings) != (
            self._roster.identities
        ):
            raise TerminalSourcePublicationControlError(
                "publication source manifest differs from the startup route roster"
            )
        if publication.canonical_binding.owner != self._local_identity:
            raise TerminalSourcePublicationControlError(
                "publication belongs to another canonical source process"
            )
        receipts = result.source_receipts
        if tuple(receipt.wire_receipt.binding for receipt in receipts) != (
            expected_bindings
        ):
            raise TerminalSourcePublicationControlError(
                "publication receipt fan-out differs from the source manifest"
            )
        local_binding = publication.canonical_binding
        with self._lock:
            self._require_ready_locked()
            active = self._active.get(local_binding.digest)
            if active is None or active != local_binding:
                raise TerminalSourcePublicationControlError(
                    "canonical publication targets an inactive local binding"
                )
            if local_binding.digest in self._terminal:
                raise TerminalSourcePublicationControlError(
                    "canonical publication outcome was delivered twice"
                )

        local_receipt: IssuedTerminalWireReceipt | None = None
        try:
            for route, receipt in zip(self._roster.routes, receipts, strict=True):
                if receipt.wire_receipt.issuer != self._local_identity:
                    raise TerminalSourcePublicationControlError(
                        "publication receipt was minted by another source process"
                    )
                if route.identity == self._local_identity:
                    local_receipt = receipt
                    continue
                self._send_frames(
                    route.endpoint,
                    encode_terminal_source_publication_receipt(
                        self._roster,
                        receipt.wire_receipt,
                    ),
                )
            if local_receipt is None:
                raise TerminalSourcePublicationControlError(
                    "publication fan-out omitted canonical local authority"
                )
            self._deliver(
                TerminalSourcePublicationDelivery(
                    wire_receipt=local_receipt.wire_receipt,
                    local_receipt=local_receipt.local_receipt,
                    authenticated_issuer=self._local_identity,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._notify_process_fatal(
                "source publication point-to-point fan-out failed"
            )
            raise TerminalSourcePublicationControlError(
                "source publication point-to-point fan-out failed"
            ) from error

    def receive_frames(self, frames: tuple[bytes, ...]) -> bool:
        """Import one canonical rank-zero outcome from the local PULL route.

        :param frames: Exact source publication control multipart message.
        :returns: Whether a new terminal receipt reached the lifecycle listener.
        """

        try:
            return self._receive_frames(frames)
        except (RuntimeError, TypeError, ValueError) as error:
            self._notify_process_fatal(
                "source publication control ingress failed: "
                f"{type(error).__name__}: {error}"
            )
            if type(error) is TerminalSourcePublicationControlError:
                raise
            raise TerminalSourcePublicationControlError(
                "source publication control ingress failed"
            ) from error

    def _receive_frames(self, frames: tuple[bytes, ...]) -> bool:
        """Authenticate and import one source-rank publication message.

        :param frames: Exact source publication control multipart message.
        :returns: Whether a new terminal receipt reached the lifecycle listener.
        """

        importer = self._importer
        if importer is None:
            raise TerminalSourcePublicationControlError(
                "canonical source rank cannot receive its own publication route"
            )
        wire_receipt = decode_terminal_source_publication_receipt(self._roster, frames)
        if wire_receipt.binding.owner != self._local_identity:
            raise TerminalSourcePublicationControlError(
                "publication receipt targets another source listener"
            )
        digest = wire_receipt.binding.digest
        with self._lock:
            self._require_ready_locked()
            active = self._active.get(digest)
            if active is None or active != wire_receipt.binding:
                raise TerminalSourcePublicationControlError(
                    "publication receipt targets an inactive source binding"
                )
            existing = self._terminal.get(digest)
            if existing is not None:
                if existing == wire_receipt:
                    return False
                self._enter_fatal_locked(
                    "source publication route received conflicting outcomes"
                )
                raise TerminalSourcePublicationControlError(self._fatal_reason)
        local_receipt = importer.import_receipt(
            wire_receipt,
            self._roster.canonical_identity,
        )
        self._deliver(
            TerminalSourcePublicationDelivery(
                wire_receipt=wire_receipt,
                local_receipt=local_receipt,
                authenticated_issuer=self._roster.canonical_identity,
            )
        )
        return True

    def retire_binding(self, binding: TerminalRequestBinding) -> None:
        """Retire replay state after the source lifecycle consumes terminality.

        :param binding: Exact local binding reaching retirement or quarantine.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        digest = binding.digest
        with self._lock:
            self._require_open_locked()
            active = self._active.get(digest)
            if active is None or active != binding:
                raise TerminalSourcePublicationControlError(
                    "source publication binding is not active"
                )
            if digest not in self._terminal:
                raise TerminalSourcePublicationControlError(
                    "source publication binding retired before terminal delivery"
                )
        importer = self._importer
        if importer is not None:
            importer.retire_binding(binding)
        fatal_reason: str | None = None
        with self._lock:
            if self._active.pop(digest, None) != binding:
                fatal_reason = "source publication binding changed during retirement"
                self._enter_fatal_locked(fatal_reason)
            else:
                self._terminal.pop(digest, None)
        if fatal_reason is not None:
            self._notify_process_fatal(fatal_reason)
            raise TerminalSourcePublicationControlError(fatal_reason)

    def inventory(self) -> TerminalSourcePublicationControlInventory:
        """Return route health and lifecycle retention.

        :returns: Immutable process-local control inventory.
        """

        with self._lock:
            return TerminalSourcePublicationControlInventory(
                route_count=len(self._roster.routes),
                active_binding_digests=tuple(sorted(self._active)),
                terminal_binding_digests=tuple(sorted(self._terminal)),
                listener_bound=self._listener is not None,
                closed=self._closed,
                fatal_reason=self._fatal_reason,
            )

    def close_clean(self) -> None:
        """Close only after every route-local lifecycle is retired."""

        with self._lock:
            self._require_open_locked()
            if self._fatal_reason is not None:
                raise TerminalSourcePublicationControlError(
                    "fatal source publication control cannot close cleanly"
                )
            if len(self._active) > 0 or len(self._terminal) > 0:
                raise TerminalSourcePublicationControlError(
                    "source publication control retains request authority"
                )
            importer = self._importer
            if importer is not None and (
                importer.active_binding_count != 0
                or importer.imported_receipt_count != 0
            ):
                raise TerminalSourcePublicationControlError(
                    "source publication importer retains replay authority"
                )
            self._closed = True

    def _deliver(self, delivery: TerminalSourcePublicationDelivery) -> None:
        """Deliver one new result and retain its exact terminal wire value.

        :param delivery: Process-local imported or locally issued authority.
        """

        digest = delivery.wire_receipt.binding.digest
        with self._lock:
            self._require_ready_locked()
            active = self._active.get(digest)
            if active is None or active != delivery.wire_receipt.binding:
                raise TerminalSourcePublicationControlError(
                    "source publication delivery targets an inactive binding"
                )
            existing = self._terminal.get(digest)
            if existing is not None:
                if existing == delivery.wire_receipt:
                    return
                self._enter_fatal_locked(
                    "source publication delivery conflicts with terminal authority"
                )
                raise TerminalSourcePublicationControlError(self._fatal_reason)
            listener = self._listener
            if listener is None:
                raise TerminalSourcePublicationControlError(
                    "source publication listener is absent"
                )
            self._terminal[digest] = delivery.wire_receipt
        listener(delivery)

    def _require_open_locked(self) -> None:
        """Require a nonclosed route owner under its transition lock."""

        if self._closed:
            raise TerminalSourcePublicationControlError(
                "source publication control is closed"
            )

    def _require_ready_locked(self) -> None:
        """Require healthy listener-bound route ownership under lock."""

        self._require_open_locked()
        if self._fatal_reason is not None:
            raise TerminalSourcePublicationControlError(self._fatal_reason)
        if self._listener is None:
            raise TerminalSourcePublicationControlError(
                "source publication listener is not bound"
            )

    def _enter_fatal_locked(self, reason: str) -> None:
        """Retain first route failure under the transition lock.

        :param reason: Stable fail-closed evidence.
        """

        if self._fatal_reason is None:
            self._fatal_reason = reason

    def _notify_process_fatal(self, reason: str) -> None:
        """Retain first failure and synchronously enter process-fatal ownership.

        :param reason: Stable fail-closed evidence.
        """

        handler: TerminalSourcePublicationFatalHandler | None = None
        fatal_reason: str
        with self._lock:
            self._enter_fatal_locked(reason)
            if self._fatal_reason is None:
                raise RuntimeError("source publication fatal reason was not retained")
            fatal_reason = self._fatal_reason
            if not self._fatal_notified:
                handler = self._process_fatal_handler
                if handler is None:
                    raise TerminalSourcePublicationControlError(
                        "source publication process-fatal handler is absent"
                    )
                self._fatal_notified = True
        if handler is not None:
            handler(fatal_reason)


def encode_terminal_source_publication_receipt(
    roster: TerminalSourcePublicationRouteRoster,
    receipt: TerminalWireReceipt,
) -> tuple[bytes, ...]:
    """Encode one canonical source publisher receipt multipart message.

    :param roster: Exact startup route authority.
    :param receipt: Rank-zero-issued source publication outcome.
    :returns: Closed generation-bound multipart message.
    """

    if type(roster) is not TerminalSourcePublicationRouteRoster:
        raise TypeError("roster must be TerminalSourcePublicationRouteRoster")
    if type(receipt) is not TerminalWireReceipt:
        raise TypeError("receipt must be TerminalWireReceipt")
    if receipt.issuer != roster.canonical_identity:
        raise TerminalSourcePublicationControlError(
            "source publication receipt has another canonical issuer"
        )
    roster.route_for(receipt.binding.owner)
    return (
        TERMINAL_SOURCE_PUBLICATION_RECEIPT_TAG,
        roster.startup_matrix_sha256,
        roster.canonical_identity.process_generation,
        receipt.encode(),
    )


def decode_terminal_source_publication_receipt(
    roster: TerminalSourcePublicationRouteRoster,
    frames: tuple[bytes, ...],
) -> TerminalWireReceipt:
    """Decode one receipt only under exact startup-route authority.

    :param roster: Exact startup route authority.
    :param frames: Candidate multipart control message.
    :returns: Validated canonical publication receipt.
    """

    if type(roster) is not TerminalSourcePublicationRouteRoster:
        raise TypeError("roster must be TerminalSourcePublicationRouteRoster")
    if type(frames) is not tuple or any(type(frame) is not bytes for frame in frames):
        raise TypeError("frames must be a tuple of bytes")
    if (
        len(frames) != _TERMINAL_SOURCE_PUBLICATION_FRAME_COUNT
        or frames[0] != TERMINAL_SOURCE_PUBLICATION_RECEIPT_TAG
    ):
        raise TerminalSourcePublicationControlError(
            "source publication control frame shape is invalid"
        )
    if frames[1] != roster.startup_matrix_sha256:
        raise TerminalSourcePublicationControlError(
            "source publication control belongs to another startup matrix"
        )
    if frames[2] != roster.canonical_identity.process_generation:
        raise TerminalSourcePublicationControlError(
            "source publication control references a stale publisher generation"
        )
    try:
        receipt = TerminalWireReceipt.decode(frames[3])
    except (RuntimeError, TypeError, ValueError) as error:
        raise TerminalSourcePublicationControlError(
            "source publication receipt payload is invalid"
        ) from error
    if receipt.issuer != roster.canonical_identity:
        raise TerminalSourcePublicationControlError(
            "source publication receipt asserts another issuer"
        )
    roster.route_for(receipt.binding.owner)
    return receipt
