import dataclasses
import enum
import hashlib
import threading

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    BoundTerminalDeadline,
    TerminalDeadlineKind,
    start_terminal_deadline,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceipt,
    TerminalWireReceiptImportNamespace,
    TerminalWireReceiptIssuer,
)


class TerminalRequestCoordinatorError(RuntimeError):
    """Request-global terminal coordination invariant violation."""


class TerminalRequestCoordinatorDisposition(enum.StrEnum):
    """Request-global coordination state."""

    COLLECTING = "collecting"
    READY = "ready"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRequestCoordinatorManifest:
    """Exact destination fan-in and terminal receipt fan-out membership.

    :ivar request_key: Stable packed request generation.
    :ivar destination_bindings: Complete destination TP-rank membership.
    :ivar recipient_bindings: Owners receiving request-global terminality.
    """

    request_key: PackedRequestKey
    destination_bindings: tuple[TerminalRequestBinding, ...]
    recipient_bindings: tuple[TerminalRequestBinding, ...]

    def __post_init__(self) -> None:
        """Validate complete, canonical fan-in and fan-out membership."""

        if type(self.request_key) is not PackedRequestKey:
            raise TypeError("request_key must be PackedRequestKey")
        if type(self.destination_bindings) is not tuple:
            raise TypeError("destination_bindings must be a tuple")
        if type(self.recipient_bindings) is not tuple:
            raise TypeError("recipient_bindings must be a tuple")
        if len(self.destination_bindings) == 0:
            raise ValueError("destination_bindings must not be empty")
        if len(self.recipient_bindings) == 0:
            raise ValueError("recipient_bindings must not be empty")

        for binding in (*self.destination_bindings, *self.recipient_bindings):
            if type(binding) is not TerminalRequestBinding:
                raise TypeError("coordinator bindings must be TerminalRequestBinding")
            if binding.request_key != self.request_key:
                raise ValueError("coordinator binding belongs to another request")

        destination_size = self.destination_bindings[0].owner.tp_size
        destination_ranks: list[int] = []
        for binding in self.destination_bindings:
            owner = binding.owner
            if owner.role is not TerminalOwnerRole.DECODE:
                raise ValueError("destination bindings require decode owners")
            if owner.tp_size != destination_size:
                raise ValueError("destination bindings disagree on TP width")
            destination_ranks.append(owner.tp_rank)
        if tuple(destination_ranks) != tuple(range(destination_size)):
            raise ValueError(
                "destination bindings must contain every TP rank in canonical order"
            )

        recipient_digests = tuple(binding.digest for binding in self.recipient_bindings)
        if len(set(recipient_digests)) != len(recipient_digests):
            raise ValueError("recipient bindings must be unique")
        recipient_set = set(self.recipient_bindings)
        if any(binding not in recipient_set for binding in self.destination_bindings):
            raise ValueError("every destination owner must receive request terminality")

    @property
    def digest(self) -> bytes:
        """Return the canonical fan-in and fan-out manifest digest.

        :returns: SHA-256 manifest digest.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.packed-terminal.request-coordinator-manifest.v1")
        digest.update(self.request_key.room_id.to_bytes(8, "big"))
        digest.update(self.request_key.request_generation)
        for label, bindings in (
            (b"destination", self.destination_bindings),
            (b"recipient", self.recipient_bindings),
        ):
            digest.update(label)
            digest.update(len(bindings).to_bytes(4, "big"))
            for binding in bindings:
                digest.update(binding.digest)
        return digest.digest()


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRequestCoordinationTiming:
    """Same-process request-global coordination interval.

    :ivar first_local_ready_received_ns: Coordinator-local timestamp of the
        first destination receipt.
    :ivar terminal_emitted_ns: Coordinator-local timestamp of terminal fan-out.
    """

    first_local_ready_received_ns: int
    terminal_emitted_ns: int

    def __post_init__(self) -> None:
        """Validate one nonnegative monotonic coordination interval."""

        values = (
            self.first_local_ready_received_ns,
            self.terminal_emitted_ns,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("coordination timestamps must be non-negative integers")
        if self.terminal_emitted_ns < self.first_local_ready_received_ns:
            raise ValueError("coordination terminal timestamp precedes its start")

    @property
    def duration_ns(self) -> int:
        """Return the same-clock coordination duration.

        :returns: Nonnegative request-global coordination duration.
        """

        return self.terminal_emitted_ns - self.first_local_ready_received_ns


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRequestCoordinatorEmission:
    """One exact recipient's request-global terminal receipt.

    :ivar recipient: Target binding for this fan-out edge.
    :ivar receipt: Joined local and authenticated-wire receipt forms.
    """

    recipient: TerminalRequestBinding
    receipt: IssuedTerminalWireReceipt

    def __post_init__(self) -> None:
        """Validate that the issued receipt targets this exact recipient."""

        if type(self.recipient) is not TerminalRequestBinding:
            raise TypeError("recipient must be TerminalRequestBinding")
        if type(self.receipt) is not IssuedTerminalWireReceipt:
            raise TypeError("receipt must be IssuedTerminalWireReceipt")
        if self.receipt.wire_receipt.binding != self.recipient:
            raise ValueError("coordinator emission targets another binding")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRequestCoordinatorResult:
    """Observable result of one coordinator input.

    :ivar disposition: Coordination state after the input.
    :ivar accepted_rank_count: Number of exact destination ranks accepted.
    :ivar newly_terminal: Whether this input created terminal fan-out.
    :ivar emissions: Newly created terminal fan-out receipts.
    :ivar timing: Coordination timing once terminal, otherwise ``None``.
    """

    disposition: TerminalRequestCoordinatorDisposition
    accepted_rank_count: int
    newly_terminal: bool
    emissions: tuple[TerminalRequestCoordinatorEmission, ...]
    timing: TerminalRequestCoordinationTiming | None

    def __post_init__(self) -> None:
        """Validate one complete coordinator result."""

        if type(self.disposition) is not TerminalRequestCoordinatorDisposition:
            raise TypeError("disposition must be TerminalRequestCoordinatorDisposition")
        if type(self.accepted_rank_count) is not int or self.accepted_rank_count < 0:
            raise ValueError("accepted_rank_count must be non-negative")
        if type(self.newly_terminal) is not bool:
            raise TypeError("newly_terminal must be bool")
        if type(self.emissions) is not tuple:
            raise TypeError("emissions must be a tuple")
        if any(
            type(emission) is not TerminalRequestCoordinatorEmission
            for emission in self.emissions
        ):
            raise TypeError("emissions must contain coordinator emissions")
        if self.newly_terminal != (len(self.emissions) > 0):
            raise ValueError("new terminality and emission population disagree")
        if self.disposition is TerminalRequestCoordinatorDisposition.COLLECTING:
            if self.timing is not None or len(self.emissions) > 0:
                raise ValueError("a collecting result cannot be terminal")
            return
        if type(self.timing) is not TerminalRequestCoordinationTiming:
            raise ValueError("a terminal result requires coordination timing")


class TerminalRequestCoordinator:
    """Point-to-point destination-rank fan-in without a hot-path collective.

    Every destination receipt is imported through the namespace for the exact
    process identity authenticated by its control route. A failure wins over an
    incomplete success manifest. Byte-identical duplicates are idempotent;
    conflicting duplicates fail the request and emit one failure fan-out.
    """

    _manifest: TerminalRequestCoordinatorManifest
    _issuer: TerminalWireReceiptIssuer
    _importers: dict[TerminalProcessIdentity, TerminalWireReceiptImportNamespace]
    _receipt_ledger: TerminalReceiptLedger
    _accepted: dict[bytes, TerminalWireReceipt]
    _disposition: TerminalRequestCoordinatorDisposition
    _deadline: BoundTerminalDeadline | None
    _terminal_emissions: tuple[TerminalRequestCoordinatorEmission, ...]
    _timing: TerminalRequestCoordinationTiming | None
    _closed: bool
    _lock: threading.Lock

    def __init__(
        self,
        manifest: TerminalRequestCoordinatorManifest,
        issuer: TerminalWireReceiptIssuer,
        importers: tuple[TerminalWireReceiptImportNamespace, ...],
    ) -> None:
        """Initialize one request-local coordinator.

        :param manifest: Exact destination and recipient membership.
        :param issuer: Canonical destination-rank issuer for terminal fan-out.
        :param importers: One authenticated import namespace per destination.
        """

        if type(manifest) is not TerminalRequestCoordinatorManifest:
            raise TypeError("manifest must be TerminalRequestCoordinatorManifest")
        if type(issuer) is not TerminalWireReceiptIssuer:
            raise TypeError("issuer must be TerminalWireReceiptIssuer")
        if type(importers) is not tuple:
            raise TypeError("importers must be a tuple")
        canonical_coordinator = manifest.destination_bindings[0].owner
        if issuer.identity != canonical_coordinator:
            raise ValueError("request coordinator must use destination rank zero")

        importers_by_identity: dict[
            TerminalProcessIdentity, TerminalWireReceiptImportNamespace
        ] = {}
        for importer in importers:
            if type(importer) is not TerminalWireReceiptImportNamespace:
                raise TypeError(
                    "importers must contain TerminalWireReceiptImportNamespace"
                )
            identity = importer.remote_issuer
            if identity in importers_by_identity:
                raise ValueError("destination importer identity was duplicated")
            importers_by_identity[identity] = importer
        expected_identities = {
            binding.owner for binding in manifest.destination_bindings
        }
        if set(importers_by_identity) != expected_identities:
            raise ValueError("importers differ from the destination manifest")

        for binding in manifest.destination_bindings:
            importers_by_identity[binding.owner].register_binding(binding)

        self._manifest = manifest
        self._issuer = issuer
        self._importers = importers_by_identity
        self._receipt_ledger = TerminalReceiptLedger(
            authorities=frozenset(
                importer.authority for importer in importers_by_identity.values()
            )
        )
        self._accepted = {}
        self._disposition = TerminalRequestCoordinatorDisposition.COLLECTING
        self._deadline = None
        self._terminal_emissions = ()
        self._timing = None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def disposition(self) -> TerminalRequestCoordinatorDisposition:
        """Return the current request-global disposition.

        :returns: Collecting, ready, or failed.
        """

        with self._lock:
            return self._disposition

    @property
    def terminal_emissions(self) -> tuple[TerminalRequestCoordinatorEmission, ...]:
        """Return the stable terminal fan-out population.

        :returns: Empty while collecting, otherwise one emission per recipient.
        """

        with self._lock:
            return self._terminal_emissions

    @property
    def deadline_expires_ns(self) -> int | None:
        """Return the armed request-global deadline, when one exists.

        The process reactor uses this value to arm its timer source after the
        first local-ready receipt. Reading the value never advances coordinator
        state and therefore cannot become a progress mechanism.

        :returns: Exact monotonic expiration timestamp, otherwise ``None``.
        """

        with self._lock:
            if self._deadline is None:
                return None
            return self._deadline.expires_ns

    def accept(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        received_ns: int,
    ) -> TerminalRequestCoordinatorResult:
        """Accept one destination-local ready or failure receipt.

        :param wire_receipt: Receipt decoded from the point-to-point route.
        :param authenticated_issuer: Exact sender identity proved by that route.
        :param received_ns: Coordinator-local monotonic receive timestamp.
        :returns: Observable state and any newly created terminal fan-out.
        :raises TerminalRequestCoordinatorError: If a protocol invariant fails.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if type(received_ns) is not int or received_ns < 0:
            raise ValueError("received_ns must be a non-negative integer")

        with self._lock:
            self._require_open()
            expected_binding = self._destination_binding(wire_receipt.binding)
            if authenticated_issuer != expected_binding.owner:
                raise TerminalRequestCoordinatorError(
                    "local-ready route authenticated another destination rank"
                )
            existing = self._accepted.get(expected_binding.digest)
            if existing is not None:
                if existing == wire_receipt:
                    return self._result(newly_terminal=False, emissions=())
                self._finish(
                    TerminalRequestCoordinatorDisposition.FAILED,
                    received_ns,
                )
                raise TerminalRequestCoordinatorError(
                    "destination rank sent conflicting terminal receipts"
                )
            if (
                self._disposition
                is not TerminalRequestCoordinatorDisposition.COLLECTING
            ):
                raise TerminalRequestCoordinatorError(
                    "new destination receipt arrived after request terminality"
                )

            importer = self._importers[authenticated_issuer]
            local_receipt = importer.import_receipt(
                wire_receipt,
                authenticated_issuer,
            )
            if wire_receipt.kind is TerminalReceiptKind.LOCAL_DECODE_READY:
                if wire_receipt.outcome is not TerminalReceiptOutcome.SUCCESS:
                    raise TerminalRequestCoordinatorError(
                        "local decode readiness requires a success outcome"
                    )
            elif wire_receipt.kind is TerminalReceiptKind.FAILURE:
                if wire_receipt.outcome is not TerminalReceiptOutcome.FAILURE:
                    raise TerminalRequestCoordinatorError(
                        "failure receipt requires a failure outcome"
                    )
            else:
                raise TerminalRequestCoordinatorError(
                    "coordinator accepts only local-ready or failure receipts"
                )
            self._receipt_ledger = self._receipt_ledger.consume(
                local_receipt,
                expected_binding,
                wire_receipt.kind,
                wire_receipt.outcome,
            )
            self._accepted[expected_binding.digest] = wire_receipt
            if self._deadline is None:
                self._deadline = start_terminal_deadline(
                    TerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
                    received_ns,
                )

            if wire_receipt.kind is TerminalReceiptKind.FAILURE:
                emissions = self._finish(
                    TerminalRequestCoordinatorDisposition.FAILED,
                    received_ns,
                )
                return self._result(newly_terminal=True, emissions=emissions)
            if len(self._accepted) == len(self._manifest.destination_bindings):
                emissions = self._finish(
                    TerminalRequestCoordinatorDisposition.READY,
                    received_ns,
                )
                return self._result(newly_terminal=True, emissions=emissions)
            return self._result(newly_terminal=False, emissions=())

    def expire(self, now_ns: int) -> TerminalRequestCoordinatorResult:
        """Fail an incomplete manifest at its frozen one-shot deadline.

        :param now_ns: Coordinator-local monotonic timestamp.
        :returns: Current state or newly emitted timeout failure.
        """

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        with self._lock:
            self._require_open()
            if (
                self._disposition
                is not TerminalRequestCoordinatorDisposition.COLLECTING
            ):
                return self._result(newly_terminal=False, emissions=())
            if self._deadline is None or not self._deadline.expired(now_ns):
                return self._result(newly_terminal=False, emissions=())
            emissions = self._finish(
                TerminalRequestCoordinatorDisposition.FAILED,
                now_ns,
            )
            return self._result(newly_terminal=True, emissions=emissions)

    def close(self) -> None:
        """Release import replay state after request-global terminality.

        :raises TerminalRequestCoordinatorError: If called while collecting.
        """

        with self._lock:
            if self._closed:
                return
            if self._disposition is TerminalRequestCoordinatorDisposition.COLLECTING:
                raise TerminalRequestCoordinatorError(
                    "cannot close an incomplete request coordinator"
                )
            for binding in self._manifest.destination_bindings:
                self._importers[binding.owner].retire_binding(binding)
            self._closed = True

    def cancel_unpublished(self) -> None:
        """Release replay state for a request never made externally visible.

        This is the sole nonterminal coordinator rollback. Once any destination
        receipt has arrived, request-global identity is externally observable
        and only terminal fan-out or fail-closed retention may release it.

        :raises TerminalRequestCoordinatorError: If coordination already began.
        """

        with self._lock:
            self._require_open()
            if (
                self._disposition
                is not TerminalRequestCoordinatorDisposition.COLLECTING
                or len(self._accepted) != 0
                or self._deadline is not None
            ):
                raise TerminalRequestCoordinatorError(
                    "cannot cancel a published request coordinator"
                )
            for binding in self._manifest.destination_bindings:
                self._importers[binding.owner].retire_binding(binding)
            self._closed = True

    def _destination_binding(
        self, candidate: TerminalRequestBinding
    ) -> TerminalRequestBinding:
        """Resolve one exact destination binding from the frozen manifest.

        :param candidate: Binding carried by an incoming receipt.
        :returns: Exact manifest-owned binding.
        :raises TerminalRequestCoordinatorError: If the binding is absent.
        """

        matches = tuple(
            binding
            for binding in self._manifest.destination_bindings
            if binding.owner == candidate.owner
        )
        if len(matches) != 1 or matches[0] != candidate:
            raise TerminalRequestCoordinatorError(
                "terminal receipt targets another destination binding"
            )
        return matches[0]

    def _finish(
        self,
        disposition: TerminalRequestCoordinatorDisposition,
        terminal_ns: int,
    ) -> tuple[TerminalRequestCoordinatorEmission, ...]:
        """Create exactly one request-global terminal fan-out.

        :param disposition: Ready or failed terminal state.
        :param terminal_ns: Coordinator-local terminal timestamp.
        :returns: One new emission per recipient.
        """

        if disposition not in (
            TerminalRequestCoordinatorDisposition.READY,
            TerminalRequestCoordinatorDisposition.FAILED,
        ):
            raise ValueError("coordinator terminal disposition is invalid")
        if self._disposition is not TerminalRequestCoordinatorDisposition.COLLECTING:
            raise TerminalRequestCoordinatorError(
                "request-global terminality was already emitted"
            )
        if self._deadline is None:
            raise TerminalRequestCoordinatorError(
                "request-global terminality requires a first receipt anchor"
            )
        kind = TerminalReceiptKind.REQUEST_READY
        outcome = TerminalReceiptOutcome.SUCCESS
        if disposition is TerminalRequestCoordinatorDisposition.FAILED:
            kind = TerminalReceiptKind.FAILURE
            outcome = TerminalReceiptOutcome.FAILURE
        emissions = tuple(
            TerminalRequestCoordinatorEmission(
                recipient=binding,
                receipt=self._issuer.issue(
                    binding=binding,
                    kind=kind,
                    outcome=outcome,
                    terminal_timestamp_ns=terminal_ns,
                ),
            )
            for binding in self._manifest.recipient_bindings
        )
        self._disposition = disposition
        self._terminal_emissions = emissions
        self._timing = TerminalRequestCoordinationTiming(
            first_local_ready_received_ns=self._deadline.started_ns,
            terminal_emitted_ns=terminal_ns,
        )
        return emissions

    def _result(
        self,
        newly_terminal: bool,
        emissions: tuple[TerminalRequestCoordinatorEmission, ...],
    ) -> TerminalRequestCoordinatorResult:
        """Build one immutable observation while holding the coordinator lock.

        :param newly_terminal: Whether the current input caused terminality.
        :param emissions: Receipts newly emitted by the current input.
        :returns: Immutable coordinator result.
        """

        return TerminalRequestCoordinatorResult(
            disposition=self._disposition,
            accepted_rank_count=len(self._accepted),
            newly_terminal=newly_terminal,
            emissions=emissions,
            timing=self._timing,
        )

    def _require_open(self) -> None:
        """Reject receipt processing after import replay state is released.

        :raises TerminalRequestCoordinatorError: If the coordinator is closed.
        """

        if self._closed:
            raise TerminalRequestCoordinatorError("request coordinator is closed")
