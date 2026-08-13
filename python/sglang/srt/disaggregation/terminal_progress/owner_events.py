import abc
import dataclasses
import enum
from typing import TypeAlias

from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEvent,
    SourceLifecycleEvent,
    TerminalResourceKind,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptAuthority,
)


class TerminalOwnerError(RuntimeError):
    """Packed terminal-progress owner invariant violation."""


class TerminalOwnerClosedError(TerminalOwnerError):
    """Submission attempted after the owner closed the relevant boundary."""


class TerminalOwnerOverflowError(TerminalOwnerError):
    """A bounded owner queue could not accept another immutable value."""

    source_name: str | None
    pending_envelopes: tuple["TerminalOwnerEventEnvelope", ...]
    rejected_envelope: "TerminalOwnerEventEnvelope | None"

    def __init__(
        self,
        message: str,
        source_name: str | None = None,
        pending_envelopes: tuple["TerminalOwnerEventEnvelope", ...] = (),
        rejected_envelope: "TerminalOwnerEventEnvelope | None" = None,
    ) -> None:
        """Retain the complete bounded-queue failure inventory.

        :param message: Reader-facing overflow evidence.
        :param source_name: Stable source identity, when known.
        :param pending_envelopes: Commands accepted before the overflow.
        :param rejected_envelope: Exact command which crossed the bound.
        """

        super().__init__(message)
        if source_name is not None and (
            type(source_name) is not str or len(source_name) == 0
        ):
            raise ValueError("source_name must be a non-empty string")
        if type(pending_envelopes) is not tuple:
            raise TypeError("pending_envelopes must be a tuple")
        self.source_name = source_name
        self.pending_envelopes = pending_envelopes
        self.rejected_envelope = rejected_envelope


class TerminalOwnerEventSourceFatalError(TerminalOwnerError):
    """Process-fatal event-source outcome retaining exact native identities."""

    source_name: str
    reason: str
    retained_identity_labels: tuple[str, ...]

    def __init__(
        self,
        source_name: str,
        reason: str,
        retained_identity_labels: tuple[str, ...],
    ) -> None:
        """Create immutable reader-facing evidence for one source fatal.

        :param source_name: Stable registered event-source identity.
        :param reason: Precise native or routing failure.
        :param retained_identity_labels: Sorted exact identities which remain
            owned or were observed while entering the fatal state.
        """

        if type(source_name) is not str or len(source_name) == 0:
            raise ValueError("source_name must be a non-empty string")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if type(retained_identity_labels) is not tuple or any(
            type(label) is not str or len(label) == 0
            for label in retained_identity_labels
        ):
            raise TypeError(
                "retained_identity_labels must be a tuple of non-empty strings"
            )
        self.source_name = source_name
        self.reason = reason
        self.retained_identity_labels = retained_identity_labels
        inventory = ",".join(retained_identity_labels)
        super().__init__(
            f"event source {source_name} is process-fatal: {reason}; "
            f"retained_identities=[{inventory}]"
        )


class TerminalOwnerDisposition(enum.StrEnum):
    """Process-lifetime reactor disposition."""

    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    PROCESS_FATAL = "process_fatal"


class TerminalOwnerFatalCause(enum.StrEnum):
    """First cause placing the owner process in fail-closed disposition."""

    SUBMISSION_QUEUE_OVERFLOW = "submission_queue_overflow"
    OUTPUT_QUEUE_OVERFLOW = "output_queue_overflow"
    EVENT_SOURCE_FAILURE = "event_source_failure"
    EVENT_SOURCE_ORDER = "event_source_order"
    OWNER_EXCEPTION = "owner_exception"
    OWNER_DEPENDENCY_DEATH = "owner_dependency_death"
    SHUTDOWN_DEADLINE = "shutdown_deadline"


class TerminalOwnerTimingField(enum.StrEnum):
    """Request-local timing fields required by the mechanism smoke receipt."""

    PRODUCER_TO_OWNER_HANDOFF = "producer_to_owner_handoff_ms"
    NATIVE_TERMINAL_DELIVERY = "native_terminal_delivery_ms"
    SCATTER_CALLBACK_DELIVERY = "scatter_callback_delivery_ms"
    ACK_AGGREGATION = "ack_aggregation_ms"
    REQUEST_GLOBAL_COORDINATION = "request_global_coordination_ms"
    SCHEDULER_INBOX_DELAY = "scheduler_inbox_delay_ms"
    METADATA_CONSUMPTION = "metadata_consumption_ms"
    GATEWAY_PUBLICATION = "gateway_publication_ms"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerTimingAnchor:
    """Start of one same-process interval completed by the owner.

    :ivar field: Frozen smoke-receipt field represented by the interval.
    :ivar sample_key: Stable cardinality key within one request and field.
    :ivar started_ns: Origin-local monotonic start timestamp.
    """

    field: TerminalOwnerTimingField
    sample_key: str
    started_ns: int

    def __post_init__(self) -> None:
        """Validate one complete timing anchor."""

        if type(self.field) is not TerminalOwnerTimingField:
            raise TypeError("field must be TerminalOwnerTimingField")
        if type(self.sample_key) is not str or len(self.sample_key) == 0:
            raise ValueError("sample_key must be a non-empty string")
        if type(self.started_ns) is not int or self.started_ns < 0:
            raise ValueError("started_ns must be a non-negative integer")


class TerminalOwnerCommand:
    """Closed base class for immutable owner commands."""


@dataclasses.dataclass(frozen=True, slots=True)
class RegisterSourceLifecycle(TerminalOwnerCommand):
    """Create one source-owner lifecycle before accepting terminal work.

    :ivar binding: Exact source-rank request binding.
    :ivar publication_identity: Canonical request publication generation.
    :ivar trusted_authorities: External issuers accepted by this lifecycle.
    """

    binding: TerminalRequestBinding
    publication_identity: TerminalPublicationIdentity
    trusted_authorities: frozenset[TerminalReceiptAuthority]

    def __post_init__(self) -> None:
        """Validate one source registration."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.publication_identity) is not TerminalPublicationIdentity:
            raise TypeError("publication_identity must be TerminalPublicationIdentity")
        if type(self.trusted_authorities) is not frozenset:
            raise TypeError("trusted_authorities must be a frozenset")
        if any(
            type(authority) is not TerminalReceiptAuthority
            for authority in self.trusted_authorities
        ):
            raise TypeError(
                "trusted_authorities must contain TerminalReceiptAuthority values"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RegisterDecodeLifecycle(TerminalOwnerCommand):
    """Create one decode-owner lifecycle before allocation publication.

    :ivar binding: Exact decode-rank request binding.
    :ivar trusted_authorities: External issuers accepted by this lifecycle.
    """

    binding: TerminalRequestBinding
    trusted_authorities: frozenset[TerminalReceiptAuthority]

    def __post_init__(self) -> None:
        """Validate one decode registration."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.trusted_authorities) is not frozenset:
            raise TypeError("trusted_authorities must be a frozenset")
        if any(
            type(authority) is not TerminalReceiptAuthority
            for authority in self.trusted_authorities
        ):
            raise TypeError(
                "trusted_authorities must contain TerminalReceiptAuthority values"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ApplySourceLifecycleEvent(TerminalOwnerCommand):
    """Apply one immutable event to an exact source lifecycle.

    :ivar binding: Exact source-rank request binding.
    :ivar event: Source reducer event.
    :ivar timing_anchor: Optional same-process interval completed by the event.
    """

    binding: TerminalRequestBinding
    event: SourceLifecycleEvent
    timing_anchor: TerminalOwnerTimingAnchor | None = None

    def __post_init__(self) -> None:
        """Validate one source event command."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.event) is not SourceLifecycleEvent:
            raise TypeError("event must be SourceLifecycleEvent")
        if (
            self.timing_anchor is not None
            and type(self.timing_anchor) is not TerminalOwnerTimingAnchor
        ):
            raise TypeError("timing_anchor must be TerminalOwnerTimingAnchor")


@dataclasses.dataclass(frozen=True, slots=True)
class ApplyDecodeLifecycleEvent(TerminalOwnerCommand):
    """Apply one immutable event to an exact decode lifecycle.

    :ivar binding: Exact decode-rank request binding.
    :ivar event: Decode reducer event.
    :ivar timing_anchor: Optional same-process interval completed by the event.
    """

    binding: TerminalRequestBinding
    event: DecodeLifecycleEvent
    timing_anchor: TerminalOwnerTimingAnchor | None = None

    def __post_init__(self) -> None:
        """Validate one decode event command."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.event) is not DecodeLifecycleEvent:
            raise TypeError("event must be DecodeLifecycleEvent")
        if (
            self.timing_anchor is not None
            and type(self.timing_anchor) is not TerminalOwnerTimingAnchor
        ):
            raise TypeError("timing_anchor must be TerminalOwnerTimingAnchor")


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleTerminalDeadline(TerminalOwnerCommand):
    """Start one hash-bound owner deadline at its exact frozen anchor.

    :ivar binding: Request generation governed by the deadline.
    :ivar kind: Frozen deadline phase.
    :ivar started_ns: Exact origin-local monotonic start timestamp.
    """

    binding: TerminalRequestBinding
    kind: TerminalDeadlineKind
    started_ns: int

    def __post_init__(self) -> None:
        """Validate one deadline start command."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.kind) is not TerminalDeadlineKind:
            raise TypeError("kind must be TerminalDeadlineKind")
        if type(self.started_ns) is not int or self.started_ns < 0:
            raise ValueError("started_ns must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class CancelTerminalDeadline(TerminalOwnerCommand):
    """Cancel one exact deadline after its phase obtained terminal proof.

    :ivar binding: Request generation governed by the deadline.
    :ivar kind: Frozen deadline phase.
    """

    binding: TerminalRequestBinding
    kind: TerminalDeadlineKind

    def __post_init__(self) -> None:
        """Validate one deadline cancellation command."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.kind) is not TerminalDeadlineKind:
            raise TypeError("kind must be TerminalDeadlineKind")


@dataclasses.dataclass(frozen=True, slots=True)
class AcknowledgeTerminalReceipt(TerminalOwnerCommand):
    """Acknowledge downstream consumption of one owner-emitted receipt.

    :ivar receipt: Exact immutable authority returned by the consumer.
    """

    receipt: TerminalReceipt

    def __post_init__(self) -> None:
        """Validate one receipt acknowledgement."""

        if type(self.receipt) is not TerminalReceipt:
            raise TypeError("receipt must be TerminalReceipt")


@dataclasses.dataclass(frozen=True, slots=True)
class BeginTerminalOwnerShutdown(TerminalOwnerCommand):
    """Close admission and begin the hash-bound owner drain.

    :ivar started_ns: Exact monotonic shutdown anchor.
    """

    started_ns: int

    def __post_init__(self) -> None:
        """Validate the shutdown start timestamp."""

        if type(self.started_ns) is not int or self.started_ns < 0:
            raise ValueError("started_ns must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class RetireTerminalOwnerShutdown(TerminalOwnerCommand):
    """Signal that external event producers are joined and drained."""


@dataclasses.dataclass(frozen=True, slots=True)
class InjectTerminalOwnerFailure(TerminalOwnerCommand):
    """Explicitly report death of a required process-lifetime component.

    :ivar cause: Exact fatal lifecycle class.
    :ivar reason: Stable reader-facing failure evidence.
    """

    cause: TerminalOwnerFatalCause
    reason: str

    def __post_init__(self) -> None:
        """Validate one explicit process-fatal command."""

        if type(self.cause) is not TerminalOwnerFatalCause:
            raise TypeError("cause must be TerminalOwnerFatalCause")
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerPulse(TerminalOwnerCommand):
    """Wake the owner to re-evaluate deadlines and drain completion."""


TerminalOwnerCommandValue: TypeAlias = (
    RegisterSourceLifecycle
    | RegisterDecodeLifecycle
    | ApplySourceLifecycleEvent
    | ApplyDecodeLifecycleEvent
    | ScheduleTerminalDeadline
    | CancelTerminalDeadline
    | AcknowledgeTerminalReceipt
    | BeginTerminalOwnerShutdown
    | RetireTerminalOwnerShutdown
    | InjectTerminalOwnerFailure
    | TerminalOwnerPulse
)

TERMINAL_OWNER_COMMAND_TYPES = (
    RegisterSourceLifecycle,
    RegisterDecodeLifecycle,
    ApplySourceLifecycleEvent,
    ApplyDecodeLifecycleEvent,
    ScheduleTerminalDeadline,
    CancelTerminalDeadline,
    AcknowledgeTerminalReceipt,
    BeginTerminalOwnerShutdown,
    RetireTerminalOwnerShutdown,
    InjectTerminalOwnerFailure,
    TerminalOwnerPulse,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerEventEnvelope:
    """One producer-ordered command delivered through an fd event source.

    :ivar producer_sequence: Gap-free sequence within this event source.
    :ivar enqueued_ns: Producer-local monotonic enqueue timestamp.
    :ivar command: Immutable owner command.
    """

    producer_sequence: int
    enqueued_ns: int
    command: TerminalOwnerCommandValue

    def __post_init__(self) -> None:
        """Validate one event-source envelope."""

        if type(self.producer_sequence) is not int or self.producer_sequence < 0:
            raise ValueError("producer_sequence must be a non-negative integer")
        if type(self.enqueued_ns) is not int or self.enqueued_ns < 0:
            raise ValueError("enqueued_ns must be a non-negative integer")
        if type(self.command) not in TERMINAL_OWNER_COMMAND_TYPES:
            raise TypeError("command must be an exact terminal owner command")


class TerminalOwnerEventSource(abc.ABC):
    """Explicit native-neutral boundary for one fd-driven event producer."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the stable source identity used in owner evidence.

        :returns: Non-empty event-source name.
        """

    @abc.abstractmethod
    def fileno(self) -> int:
        """Return the readable fd which signals queued envelopes.

        :returns: Open non-negative file descriptor.
        """

    @property
    @abc.abstractmethod
    def pending_count(self) -> int:
        """Return the exact number of envelopes awaiting owner drain.

        :returns: Non-negative queued envelope count.
        """

    @abc.abstractmethod
    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Drain every envelope published before the observed wake.

        :returns: Producer-ordered immutable envelopes.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release this source after every producer has joined."""


class TerminalOwnerDispatchObserver(abc.ABC):
    """Post-dispatch authority for a source requiring closed-loop progress."""

    @abc.abstractmethod
    def acknowledge_dispatch(
        self,
        envelope: TerminalOwnerEventEnvelope,
    ) -> None:
        """Observe one command only after its owner transition committed.

        :param envelope: Exact source envelope committed by the owner.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerEventSourceRegistration:
    """One event source registered before the reactor starts.

    :ivar source: Explicit fd-driven event-source adapter.
    :ivar close_on_shutdown: Whether the owner owns final source closure.
    :ivar dispatch_observer: Optional source authority notified only after a
        transition commits.
    """

    source: TerminalOwnerEventSource
    close_on_shutdown: bool
    dispatch_observer: TerminalOwnerDispatchObserver | None = None

    def __post_init__(self) -> None:
        """Validate one source registration."""

        if not isinstance(self.source, TerminalOwnerEventSource):
            raise TypeError("source must inherit TerminalOwnerEventSource")
        if type(self.close_on_shutdown) is not bool:
            raise TypeError("close_on_shutdown must be a bool")
        if self.dispatch_observer is not None and not isinstance(
            self.dispatch_observer,
            TerminalOwnerDispatchObserver,
        ):
            raise TypeError(
                "dispatch_observer must inherit TerminalOwnerDispatchObserver"
            )
        if type(self.source.name) is not str or len(self.source.name) == 0:
            raise ValueError("event source name must be non-empty")
        if type(self.source.fileno()) is not int or self.source.fileno() < 0:
            raise ValueError("event source must expose an open file descriptor")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerTimingSample:
    """One immutable request-correlated owner interval.

    :ivar binding: Exact request and process generation.
    :ivar field: Frozen timing field.
    :ivar sample_key: Cardinality key within the field.
    :ivar started_ns: Origin-local interval start.
    :ivar owner_started_ns: Time the owner began dispatching the command.
    :ivar completed_ns: Time the owner completed the state transition.
    :ivar owner_sequence: Gap-free sequence assigned at owner dispatch.
    """

    binding: TerminalRequestBinding
    field: TerminalOwnerTimingField
    sample_key: str
    started_ns: int
    owner_started_ns: int
    completed_ns: int
    owner_sequence: int

    def __post_init__(self) -> None:
        """Validate interval ordering and identity."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.field) is not TerminalOwnerTimingField:
            raise TypeError("field must be TerminalOwnerTimingField")
        if type(self.sample_key) is not str or len(self.sample_key) == 0:
            raise ValueError("sample_key must be a non-empty string")
        timestamps = (self.started_ns, self.owner_started_ns, self.completed_ns)
        if any(type(timestamp) is not int or timestamp < 0 for timestamp in timestamps):
            raise ValueError("timing timestamps must be non-negative integers")
        if self.owner_started_ns < self.started_ns:
            raise ValueError("owner dispatch cannot precede the timing anchor")
        if self.completed_ns < self.owner_started_ns:
            raise ValueError("timing completion cannot precede owner dispatch")
        if type(self.owner_sequence) is not int or self.owner_sequence < 0:
            raise ValueError("owner_sequence must be a non-negative integer")

    @property
    def duration_ns(self) -> int:
        """Return the complete anchored interval.

        :returns: Nanoseconds from the external anchor through owner completion.
        """

        return self.completed_ns - self.started_ns

    @property
    def owner_queue_delay_ns(self) -> int:
        """Return time between external enqueue and owner dispatch.

        :returns: Nanoseconds spent awaiting owner execution.
        """

        return self.owner_started_ns - self.started_ns


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerReceiptEmission:
    """One immutable owner-minted authority awaiting consumer acknowledgement.

    :ivar receipt: Exact one-shot authority receipt.
    :ivar emitted_ns: Owner-local monotonic emission timestamp.
    :ivar owner_sequence: Gap-free owner transition sequence.
    """

    receipt: TerminalReceipt
    emitted_ns: int
    owner_sequence: int

    def __post_init__(self) -> None:
        """Validate one owner receipt emission."""

        if type(self.receipt) is not TerminalReceipt:
            raise TypeError("receipt must be TerminalReceipt")
        if type(self.emitted_ns) is not int or self.emitted_ns < 0:
            raise ValueError("emitted_ns must be a non-negative integer")
        if type(self.owner_sequence) is not int or self.owner_sequence < 0:
            raise ValueError("owner_sequence must be a non-negative integer")


TerminalOwnerOutput: TypeAlias = (
    TerminalOwnerTimingSample | TerminalOwnerReceiptEmission
)
TERMINAL_OWNER_OUTPUT_TYPES = (TerminalOwnerTimingSample, TerminalOwnerReceiptEmission)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerQuarantineEntry:
    """Exact request-local inventory retained by fail-closed ownership.

    :ivar binding: Request and owner generation carrying ambiguity.
    :ivar resources: Exact quarantined resource kinds.
    :ivar reason: Stable reader-facing quarantine cause.
    """

    binding: TerminalRequestBinding
    resources: frozenset[TerminalResourceKind]
    reason: str

    def __post_init__(self) -> None:
        """Validate one non-empty quarantine inventory."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.resources) is not frozenset or len(self.resources) == 0:
            raise ValueError("resources must be a non-empty frozenset")
        if any(
            type(resource) is not TerminalResourceKind for resource in self.resources
        ):
            raise TypeError("resources must contain TerminalResourceKind values")
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalOwnerSnapshot:
    """Thread-safe immutable view of all process-lifetime owner inventory.

    :ivar disposition: Current reactor lifecycle disposition.
    :ivar admission_open: Whether new request registrations are accepted.
    :ivar reactor_alive: Whether the process-lifetime owner thread is running.
    :ivar source_active: Active source lifecycle bindings.
    :ivar decode_active: Active decode lifecycle bindings.
    :ivar safely_retired: Bindings fully retired by exact proof.
    :ivar quarantined: Exact retained resource inventories.
    :ivar pending_receipts: Owner-minted receipts awaiting consumption ACK.
    :ivar queued_submission_count: Commands currently awaiting owner dispatch.
    :ivar queued_output_count: Outputs awaiting consumer drain.
    :ivar owner_transition_count: Gap-free completed owner transition count.
    :ivar fatal_cause: First process-fatal cause, if any.
    :ivar fatal_reason: Reader-facing first fatal reason, if any.
    :ivar fatal_traceback: Complete traceback for an unexpected owner failure.
    """

    disposition: TerminalOwnerDisposition
    admission_open: bool
    reactor_alive: bool
    source_active: tuple[TerminalRequestBinding, ...]
    decode_active: tuple[TerminalRequestBinding, ...]
    safely_retired: tuple[TerminalRequestBinding, ...]
    quarantined: tuple[TerminalOwnerQuarantineEntry, ...]
    pending_receipts: tuple[TerminalReceipt, ...]
    queued_submission_count: int
    queued_output_count: int
    owner_transition_count: int
    fatal_cause: TerminalOwnerFatalCause | None = None
    fatal_reason: str | None = None
    fatal_traceback: str | None = None

    def __post_init__(self) -> None:
        """Validate snapshot shape and fatal-state consistency."""

        if type(self.disposition) is not TerminalOwnerDisposition:
            raise TypeError("disposition must be TerminalOwnerDisposition")
        if (
            type(self.admission_open) is not bool
            or type(self.reactor_alive) is not bool
        ):
            raise TypeError("snapshot lifecycle flags must be bool values")
        tuple_fields = (
            self.source_active,
            self.decode_active,
            self.safely_retired,
            self.quarantined,
            self.pending_receipts,
        )
        if any(type(value) is not tuple for value in tuple_fields):
            raise TypeError("snapshot inventory fields must be tuples")
        counts = (
            self.queued_submission_count,
            self.queued_output_count,
            self.owner_transition_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("snapshot counts must be non-negative integers")
        if self.disposition is TerminalOwnerDisposition.PROCESS_FATAL:
            if type(self.fatal_cause) is not TerminalOwnerFatalCause:
                raise ValueError("process-fatal snapshot requires a fatal cause")
            if type(self.fatal_reason) is not str or len(self.fatal_reason) == 0:
                raise ValueError("process-fatal snapshot requires a fatal reason")
            return
        if self.fatal_cause is not None or self.fatal_reason is not None:
            raise ValueError("non-fatal snapshot cannot carry fatal evidence")
        if self.fatal_traceback is not None:
            raise ValueError("non-fatal snapshot cannot carry a traceback")
