import dataclasses
import enum
import hashlib

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    PACKED_TERMINAL_DEADLINES,
    TerminalDeadlineKind,
    start_terminal_deadline,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DECODE_RESOURCE_KINDS,
    SOURCE_RESOURCE_KINDS,
    DecodeLifecycle,
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
    DecodeLifecyclePhase,
    SourceLifecycle,
    SourceLifecycleEvent,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
    TerminalLifecycleError,
    TerminalProcessDisposition,
    TerminalResourceKind,
    create_decode_lifecycle,
    create_source_lifecycle,
    reduce_decode_lifecycle,
    reduce_source_lifecycle,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptError,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)

_REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
_SOURCE_PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_DECODE_PROCESS_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_PUBLICATION_GENERATION = bytes.fromhex("0123456789abcdeffedcba9876543210")
_RANK_MANIFEST_DIGEST = b"r" * 32
_ALLOCATION_DIGEST = b"a" * 32
_DEFAULT_FAILURE_REASON = "oracle failure"


class OracleReceiptIssuer(enum.StrEnum):
    """Receipt issuer selected by one semantic oracle stimulus."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class OracleReceiptBinding(enum.StrEnum):
    """Receipt binding selected by one semantic oracle stimulus."""

    TARGET = "target"
    OTHER_REQUEST = "other_request"


class OracleReductionError(enum.StrEnum):
    """Stable rejection class returned by the canonical reducer oracle."""

    LIFECYCLE = "lifecycle"
    RECEIPT = "receipt"
    STATE_INVARIANT = "state_invariant"


class OracleOwnerAction(enum.StrEnum):
    """Semantic side effects earned by an accepted lifecycle commit."""

    STATE_COMMITTED = "state_committed"
    RECLAIM_AUTHORIZED = "reclaim_authorized"
    ADOPTION_READY = "adoption_ready"
    LOCAL_DECODE_READY = "local_decode_ready"
    REQUEST_RETIRED = "request_retired"
    REQUEST_QUARANTINED = "request_quarantined"
    PROCESS_FATAL = "process_fatal"
    SOURCE_GATHER_READY = "source_gather_ready"
    SOURCE_OUTCOME_READY = "source_outcome_ready"
    SOURCE_ACK_READY = "source_ack_ready"
    DECODE_SCATTER_READY = "decode_scatter_ready"
    DECODE_TEARDOWN_READY = "decode_teardown_ready"
    GATEWAY_PUBLICATION_READY = "gateway_publication_ready"


class OracleDeadlineOutcome(enum.StrEnum):
    """Fail-closed disposition earned at one deadline boundary."""

    REQUEST_QUARANTINE = "request_quarantine"
    PROCESS_FATAL = "process_fatal"


LifecycleEventKind = SourceLifecycleEventKind | DecodeLifecycleEventKind
Lifecycle = SourceLifecycle | DecodeLifecycle


@dataclasses.dataclass(frozen=True, slots=True)
class OracleReceiptSpec:
    """Semantic authority input shared by Python and native test adapters.

    :ivar key: Stable one-shot identity. Reusing a key replays the same receipt.
    :ivar kind: Authority carried by the receipt.
    :ivar outcome: Terminal outcome authenticated by the issuer.
    :ivar issuer: Whether the receipt is minted by a registered issuer.
    :ivar binding: Whether the receipt targets this lifecycle generation.
    """

    key: str
    kind: TerminalReceiptKind
    outcome: TerminalReceiptOutcome
    issuer: OracleReceiptIssuer = OracleReceiptIssuer.TRUSTED
    binding: OracleReceiptBinding = OracleReceiptBinding.TARGET

    def __post_init__(self) -> None:
        """Validate one complete semantic receipt input."""

        if type(self.key) is not str or len(self.key) == 0:
            raise ValueError("oracle receipt key must be a non-empty string")
        if type(self.kind) is not TerminalReceiptKind:
            raise TypeError("kind must be TerminalReceiptKind")
        if type(self.outcome) is not TerminalReceiptOutcome:
            raise TypeError("outcome must be TerminalReceiptOutcome")
        if type(self.issuer) is not OracleReceiptIssuer:
            raise TypeError("issuer must be OracleReceiptIssuer")
        if type(self.binding) is not OracleReceiptBinding:
            raise TypeError("binding must be OracleReceiptBinding")

    @property
    def deterministic_nonce(self) -> bytes:
        """Return a stable sixteen-byte nonce for a native test adapter.

        :returns: Domain-separated one-shot receipt identity.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.terminal-progress.native-oracle.receipt.v1")
        digest.update(self.key.encode("utf-8"))
        return digest.digest()[:16]


@dataclasses.dataclass(frozen=True, slots=True)
class OracleEventSpec:
    """One structurally valid lifecycle event independent of an implementation.

    :ivar kind: Exact source or decode event identity.
    :ivar receipt: Authority input required by receipt-bearing events.
    :ivar reason: Stable evidence required by failure events.
    """

    kind: LifecycleEventKind
    receipt: OracleReceiptSpec | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the event against the public lifecycle vocabulary."""

        if type(self.kind) not in (
            SourceLifecycleEventKind,
            DecodeLifecycleEventKind,
        ):
            raise TypeError("kind must be a source or decode lifecycle event kind")
        receipt_requirement = _receipt_requirement(self.kind)
        if receipt_requirement is None and self.receipt is not None:
            raise ValueError(f"{self.kind.value} does not accept a receipt")
        if (
            receipt_requirement is not None
            and type(self.receipt) is not OracleReceiptSpec
        ):
            raise ValueError(f"{self.kind.value} requires an oracle receipt")
        if _event_requires_reason(self.kind):
            if type(self.reason) is not str or len(self.reason) == 0:
                raise ValueError(f"{self.kind.value} requires a non-empty reason")
        elif self.reason is not None:
            raise ValueError(f"{self.kind.value} does not accept a reason")

    @property
    def role(self) -> TerminalOwnerRole:
        """Return the lifecycle role accepting this event.

        :returns: Source or decode owner role.
        """

        if type(self.kind) is SourceLifecycleEventKind:
            return TerminalOwnerRole.SOURCE
        return TerminalOwnerRole.DECODE


@dataclasses.dataclass(frozen=True, slots=True)
class OracleLifecyclePath:
    """Exact accepted event history reaching one canonical lifecycle state.

    :ivar name: Stable reader-facing path identity.
    :ivar role: Source or decode lifecycle role.
    :ivar events: Accepted events following initial registration.
    """

    name: str
    role: TerminalOwnerRole
    events: tuple[OracleEventSpec, ...]

    def __post_init__(self) -> None:
        """Validate one role-consistent accepted history."""

        if type(self.name) is not str or len(self.name) == 0:
            raise ValueError("oracle path name must be a non-empty string")
        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if any(type(event) is not OracleEventSpec for event in self.events):
            raise TypeError("events must contain OracleEventSpec values")
        if any(event.role is not self.role for event in self.events):
            raise ValueError("every path event must target the path role")


@dataclasses.dataclass(frozen=True, slots=True)
class OracleTransitionCase:
    """One differential transition case for the Python and native reducers.

    :ivar name: Stable case identity.
    :ivar path: Accepted history establishing the input state.
    :ivar event: Candidate event to adjudicate from that state.
    """

    name: str
    path: OracleLifecyclePath
    event: OracleEventSpec

    def __post_init__(self) -> None:
        """Validate one role-consistent transition case."""

        if type(self.name) is not str or len(self.name) == 0:
            raise ValueError("oracle case name must be a non-empty string")
        if type(self.path) is not OracleLifecyclePath:
            raise TypeError("path must be OracleLifecyclePath")
        if type(self.event) is not OracleEventSpec:
            raise TypeError("event must be OracleEventSpec")
        if self.event.role is not self.path.role:
            raise ValueError("candidate event must target the path role")


@dataclasses.dataclass(frozen=True, slots=True)
class OracleLifecycleProjection:
    """Implementation-neutral projection of one lifecycle state.

    :ivar role: Source or decode lifecycle role.
    :ivar phase: Canonical phase string.
    :ivar live_resources: Resources still pinned by the owner.
    :ivar retired_resources: Resources carrying exact reuse proof.
    :ivar quarantined_resources: Resources retained fail-closed.
    :ivar process_fatal: Whether process continuation is unsafe.
    """

    role: TerminalOwnerRole
    phase: str
    live_resources: frozenset[TerminalResourceKind]
    retired_resources: frozenset[TerminalResourceKind]
    quarantined_resources: frozenset[TerminalResourceKind]
    process_fatal: bool

    def __post_init__(self) -> None:
        """Validate exact resource conservation in the projection."""

        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        if type(self.phase) is not str or len(self.phase) == 0:
            raise ValueError("phase must be a non-empty string")
        partitions = (
            self.live_resources,
            self.retired_resources,
            self.quarantined_resources,
        )
        if any(type(partition) is not frozenset for partition in partitions):
            raise TypeError("resource partitions must be frozensets")
        if any(
            type(resource) is not TerminalResourceKind
            for partition in partitions
            for resource in partition
        ):
            raise TypeError("resource partitions require TerminalResourceKind values")
        if (
            len(partitions[0] & partitions[1]) > 0
            or len(partitions[0] & partitions[2]) > 0
            or len(partitions[1] & partitions[2]) > 0
        ):
            raise ValueError("oracle resource partitions overlap")
        universe = (
            SOURCE_RESOURCE_KINDS
            if self.role is TerminalOwnerRole.SOURCE
            else DECODE_RESOURCE_KINDS
        )
        if partitions[0] | partitions[1] | partitions[2] != universe:
            raise ValueError("oracle resource partitions are not conservative")
        if type(self.process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")


@dataclasses.dataclass(frozen=True, slots=True)
class OracleTransitionEvaluation:
    """Canonical verdict and post-state for one differential case.

    :ivar case: Transition case which was evaluated.
    :ivar accepted: Whether the canonical reducer committed the event.
    :ivar before: State immediately before the candidate event.
    :ivar after: State after commit, or the unchanged input after rejection.
    :ivar actions: Semantic actions earned by an accepted commit.
    :ivar emitted_receipts: Authority receipts earned by an accepted commit.
    :ivar error: Stable rejection class, absent after an accepted commit.
    :ivar error_message: Canonical rejection evidence.
    """

    case: OracleTransitionCase
    accepted: bool
    before: OracleLifecycleProjection
    after: OracleLifecycleProjection
    actions: tuple[OracleOwnerAction, ...]
    emitted_receipts: tuple[TerminalReceiptKind, ...]
    error: OracleReductionError | None
    error_message: str | None

    def __post_init__(self) -> None:
        """Validate accepted and rejected verdict shapes."""

        if type(self.case) is not OracleTransitionCase:
            raise TypeError("case must be OracleTransitionCase")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be bool")
        if type(self.before) is not OracleLifecycleProjection:
            raise TypeError("before must be OracleLifecycleProjection")
        if type(self.after) is not OracleLifecycleProjection:
            raise TypeError("after must be OracleLifecycleProjection")
        if type(self.actions) is not tuple or any(
            type(action) is not OracleOwnerAction for action in self.actions
        ):
            raise TypeError("actions must contain OracleOwnerAction values")
        if type(self.emitted_receipts) is not tuple or any(
            type(kind) is not TerminalReceiptKind for kind in self.emitted_receipts
        ):
            raise TypeError("emitted_receipts must contain receipt kinds")
        if self.accepted:
            if self.error is not None or self.error_message is not None:
                raise ValueError(
                    "an accepted transition cannot carry rejection evidence"
                )
            return
        if type(self.error) is not OracleReductionError:
            raise ValueError("a rejected transition requires an error class")
        if type(self.error_message) is not str or len(self.error_message) == 0:
            raise ValueError("a rejected transition requires error evidence")
        if self.before != self.after:
            raise ValueError("a rejected canonical reduction must not mutate state")
        if len(self.actions) > 0 or len(self.emitted_receipts) > 0:
            raise ValueError("a rejected transition cannot earn side effects")


@dataclasses.dataclass(frozen=True, slots=True)
class OraclePathEvaluation:
    """Canonical state sequence reached by one accepted path.

    :ivar path: Accepted event history.
    :ivar states: Initial registration state followed by every committed state.
    """

    path: OracleLifecyclePath
    states: tuple[OracleLifecycleProjection, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class OracleDeadlineBoundaryCase:
    """Exact before/at/after boundary for one hash-bound deadline.

    :ivar kind: Frozen deadline identity.
    :ivar started_ns: Test-controlled start timestamp.
    :ivar before_expiry_ns: Last timestamp which must not expire.
    :ivar expires_ns: First timestamp which must expire.
    :ivar after_expiry_ns: A later timestamp which remains expired.
    :ivar outcome: Fail-closed disposition at expiry.
    """

    kind: TerminalDeadlineKind
    started_ns: int
    before_expiry_ns: int
    expires_ns: int
    after_expiry_ns: int
    outcome: OracleDeadlineOutcome


_SOURCE_RECEIPT_REQUIREMENTS = {
    SourceLifecycleEventKind.REQUEST_READY_RECEIVED: (
        TerminalReceiptKind.REQUEST_READY,
        TerminalReceiptOutcome.SUCCESS,
    ),
    SourceLifecycleEventKind.RECLAIM_CONSUMED: (
        TerminalReceiptKind.RECLAIM_CONSUMED,
        TerminalReceiptOutcome.SUCCESS,
    ),
    SourceLifecycleEventKind.GATEWAY_PUBLISHED: (
        TerminalReceiptKind.GATEWAY_PUBLISHED,
        TerminalReceiptOutcome.SUCCESS,
    ),
    SourceLifecycleEventKind.PUBLICATION_FAILED: (
        TerminalReceiptKind.FAILURE,
        TerminalReceiptOutcome.FAILURE,
    ),
    SourceLifecycleEventKind.REQUEST_FAILED: (
        TerminalReceiptKind.FAILURE,
        TerminalReceiptOutcome.FAILURE,
    ),
}
_DECODE_RECEIPT_REQUIREMENTS = {
    DecodeLifecycleEventKind.ADOPTION_CONSUMED: (
        TerminalReceiptKind.ADOPTION_READY,
        TerminalReceiptOutcome.SUCCESS,
    ),
    DecodeLifecycleEventKind.REQUEST_READY_RECEIVED: (
        TerminalReceiptKind.REQUEST_READY,
        TerminalReceiptOutcome.SUCCESS,
    ),
    DecodeLifecycleEventKind.REQUEST_FAILED: (
        TerminalReceiptKind.FAILURE,
        TerminalReceiptOutcome.FAILURE,
    ),
}
_SOURCE_REASON_EVENTS = frozenset(
    (
        SourceLifecycleEventKind.PUBLICATION_FAILED,
        SourceLifecycleEventKind.REQUEST_FAILED,
        SourceLifecycleEventKind.OWNER_DIED,
        SourceLifecycleEventKind.PUBLISHER_DIED,
        SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    )
)
_DECODE_REASON_EVENTS = frozenset(
    (
        DecodeLifecycleEventKind.CANCEL_UNPUBLISHED,
        DecodeLifecycleEventKind.REQUEST_FAILED,
        DecodeLifecycleEventKind.OWNER_DIED,
        DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    )
)


def _receipt_requirement(
    kind: LifecycleEventKind,
) -> tuple[TerminalReceiptKind, TerminalReceiptOutcome] | None:
    """Return the receipt authority required by one event.

    :param kind: Source or decode lifecycle event.
    :returns: Required receipt kind and outcome, when receipt-bearing.
    """

    if type(kind) is SourceLifecycleEventKind:
        return _SOURCE_RECEIPT_REQUIREMENTS.get(kind)
    return _DECODE_RECEIPT_REQUIREMENTS.get(kind)


def _event_requires_reason(kind: LifecycleEventKind) -> bool:
    """Return whether one event carries stable failure evidence.

    :param kind: Source or decode lifecycle event.
    :returns: Whether a non-empty reason is required.
    """

    if type(kind) is SourceLifecycleEventKind:
        return kind in _SOURCE_REASON_EVENTS
    return kind in _DECODE_REASON_EVENTS


def make_oracle_event(
    kind: LifecycleEventKind,
    *,
    receipt_key: str | None = None,
    receipt_kind: TerminalReceiptKind | None = None,
    receipt_outcome: TerminalReceiptOutcome | None = None,
    receipt_issuer: OracleReceiptIssuer = OracleReceiptIssuer.TRUSTED,
    receipt_binding: OracleReceiptBinding = OracleReceiptBinding.TARGET,
    reason: str | None = None,
) -> OracleEventSpec:
    """Construct one structurally valid semantic lifecycle event.

    :param kind: Exact source or decode event.
    :param receipt_key: Stable one-shot identity for a receipt-bearing event.
    :param receipt_kind: Optional authority override for negative cases.
    :param receipt_outcome: Optional outcome override for negative cases.
    :param receipt_issuer: Trusted or untrusted issuer selection.
    :param receipt_binding: Target or conflicting request binding.
    :param reason: Optional failure reason override.
    :returns: Implementation-independent event specification.
    """

    requirement = _receipt_requirement(kind)
    receipt = None
    if requirement is not None:
        if type(receipt_key) is not str or len(receipt_key) == 0:
            raise ValueError("receipt-bearing oracle events require receipt_key")
        receipt = OracleReceiptSpec(
            key=receipt_key,
            kind=requirement[0] if receipt_kind is None else receipt_kind,
            outcome=requirement[1] if receipt_outcome is None else receipt_outcome,
            issuer=receipt_issuer,
            binding=receipt_binding,
        )
    elif (
        receipt_key is not None
        or receipt_kind is not None
        or receipt_outcome is not None
        or receipt_issuer is not OracleReceiptIssuer.TRUSTED
        or receipt_binding is not OracleReceiptBinding.TARGET
    ):
        raise ValueError("non-receipt oracle events cannot carry receipt fields")
    event_reason = reason
    if _event_requires_reason(kind) and event_reason is None:
        event_reason = _DEFAULT_FAILURE_REASON
    return OracleEventSpec(kind=kind, receipt=receipt, reason=event_reason)


def _source_forward_events() -> tuple[OracleEventSpec, ...]:
    """Return the exact source forward path before request readiness.

    :returns: Seven accepted source events.
    """

    return tuple(
        make_oracle_event(kind)
        for kind in (
            SourceLifecycleEventKind.SUBMISSION_ACCEPTED,
            SourceLifecycleEventKind.PRODUCER_COMPLETED,
            SourceLifecycleEventKind.GATHER_COMPLETED_AND_NATIVE_POSTED,
            SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL,
            SourceLifecycleEventKind.OUTCOMES_SENT,
            SourceLifecycleEventKind.TEARDOWN_RECEIVED,
            SourceLifecycleEventKind.ACK_SENT,
        )
    )


def _decode_forward_events() -> tuple[OracleEventSpec, ...]:
    """Return the exact decode forward path before scheduler adoption.

    :returns: Eight accepted decode events.
    """

    return tuple(
        make_oracle_event(kind)
        for kind in (
            DecodeLifecycleEventKind.ALLOCATION_PUBLISHED,
            DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
            DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
            DecodeLifecycleEventKind.SCATTER_STARTED,
            DecodeLifecycleEventKind.SCATTER_TERMINAL,
            DecodeLifecycleEventKind.TEARDOWN_SENT,
            DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
            DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
        )
    )


def source_oracle_paths() -> tuple[OracleLifecyclePath, ...]:
    """Return every semantically distinct reachable source state used by tests.

    :returns: Accepted histories covering forward, join, and failure states.
    """

    forward = _source_forward_events()
    paths = [
        OracleLifecyclePath(
            name=f"source-{SourceLifecyclePhase.FROZEN.value}",
            role=TerminalOwnerRole.SOURCE,
            events=(),
        )
    ]
    forward_phases = (
        SourceLifecyclePhase.WAITING_FOR_PRODUCER,
        SourceLifecyclePhase.GATHERING,
        SourceLifecyclePhase.NATIVE_IN_FLIGHT,
        SourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL,
        SourceLifecyclePhase.OUTCOMES_SENT,
        SourceLifecyclePhase.TEARDOWN_RECEIVED,
        SourceLifecyclePhase.ACK_SENT,
    )
    for index, phase in enumerate(forward_phases, start=1):
        paths.append(
            OracleLifecyclePath(
                name=f"source-{phase.value}",
                role=TerminalOwnerRole.SOURCE,
                events=forward[:index],
            )
        )
    ready_event = make_oracle_event(
        SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
        receipt_key="source-path-request-ready",
    )
    ready = (*forward, ready_event)
    reclaim_event = make_oracle_event(
        SourceLifecycleEventKind.RECLAIM_CONSUMED,
        receipt_key="source-path-reclaim",
    )
    publish_event = make_oracle_event(
        SourceLifecycleEventKind.GATEWAY_PUBLISHED,
        receipt_key="source-path-publish",
    )
    publication_failure = make_oracle_event(
        SourceLifecycleEventKind.PUBLICATION_FAILED,
        receipt_key="source-path-publication-failure",
    )
    paths.extend(
        (
            OracleLifecyclePath(
                name="source-ready",
                role=TerminalOwnerRole.SOURCE,
                events=ready,
            ),
            OracleLifecyclePath(
                name="source-ready-reclaim-consumed",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, reclaim_event),
            ),
            OracleLifecyclePath(
                name="source-ready-gateway-published",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, publish_event),
            ),
            OracleLifecyclePath(
                name="source-publication-quarantined",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, publication_failure),
            ),
            OracleLifecyclePath(
                name="source-publication-quarantined-after-reclaim",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, reclaim_event, publication_failure),
            ),
            OracleLifecyclePath(
                name="source-retired-reclaim-then-publish",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, reclaim_event, publish_event),
            ),
            OracleLifecyclePath(
                name="source-retired-publish-then-reclaim",
                role=TerminalOwnerRole.SOURCE,
                events=(*ready, publish_event, reclaim_event),
            ),
            OracleLifecyclePath(
                name="source-quarantined-request-failure",
                role=TerminalOwnerRole.SOURCE,
                events=(
                    make_oracle_event(
                        SourceLifecycleEventKind.REQUEST_FAILED,
                        receipt_key="source-path-request-failure",
                    ),
                ),
            ),
            OracleLifecyclePath(
                name="source-quarantined-owner-death",
                role=TerminalOwnerRole.SOURCE,
                events=(make_oracle_event(SourceLifecycleEventKind.OWNER_DIED),),
            ),
            OracleLifecyclePath(
                name="source-publication-quarantined-publisher-death",
                role=TerminalOwnerRole.SOURCE,
                events=(
                    *ready,
                    make_oracle_event(SourceLifecycleEventKind.PUBLISHER_DIED),
                ),
            ),
        )
    )
    return tuple(paths)


def decode_oracle_paths() -> tuple[OracleLifecyclePath, ...]:
    """Return every semantically distinct reachable decode state used by tests.

    :returns: Accepted histories covering forward, adoption, and failure states.
    """

    forward = _decode_forward_events()
    paths = [
        OracleLifecyclePath(
            name=f"decode-{DecodeLifecyclePhase.PREPARED.value}",
            role=TerminalOwnerRole.DECODE,
            events=(),
        )
    ]
    forward_phases = (
        DecodeLifecyclePhase.PUBLISHED,
        DecodeLifecyclePhase.WRITER_AGGREGATING,
        DecodeLifecyclePhase.SCATTER_READY,
        DecodeLifecyclePhase.SCATTER_IN_FLIGHT,
        DecodeLifecyclePhase.SCATTER_TERMINAL,
        DecodeLifecyclePhase.TEARDOWN_SENT,
        DecodeLifecyclePhase.ACK_AGGREGATING,
        DecodeLifecyclePhase.ADOPTION_READY,
    )
    for index, phase in enumerate(forward_phases, start=1):
        paths.append(
            OracleLifecyclePath(
                name=f"decode-{phase.value}",
                role=TerminalOwnerRole.DECODE,
                events=forward[:index],
            )
        )
    adoption = make_oracle_event(
        DecodeLifecycleEventKind.ADOPTION_CONSUMED,
        receipt_key="decode-path-adoption",
    )
    metadata = make_oracle_event(
        DecodeLifecycleEventKind.METADATA_CONSUMED,
    )
    local_ready = make_oracle_event(DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED)
    request_ready = make_oracle_event(
        DecodeLifecycleEventKind.REQUEST_READY_RECEIVED,
        receipt_key="decode-path-request-ready",
    )
    adopted = (*forward, adoption)
    metadata_consumed = (*adopted, metadata)
    locally_ready = (*metadata_consumed, local_ready)
    paths.extend(
        (
            OracleLifecyclePath(
                name="decode-adopted-by-scheduler",
                role=TerminalOwnerRole.DECODE,
                events=adopted,
            ),
            OracleLifecyclePath(
                name="decode-metadata-consumed",
                role=TerminalOwnerRole.DECODE,
                events=metadata_consumed,
            ),
            OracleLifecyclePath(
                name="decode-local-ready",
                role=TerminalOwnerRole.DECODE,
                events=locally_ready,
            ),
            OracleLifecyclePath(
                name="decode-retired-request-ready",
                role=TerminalOwnerRole.DECODE,
                events=(*locally_ready, request_ready),
            ),
            OracleLifecyclePath(
                name="decode-retired-cancel-unpublished",
                role=TerminalOwnerRole.DECODE,
                events=(
                    make_oracle_event(DecodeLifecycleEventKind.CANCEL_UNPUBLISHED),
                ),
            ),
            OracleLifecyclePath(
                name="decode-quarantined-request-failure",
                role=TerminalOwnerRole.DECODE,
                events=(
                    forward[0],
                    make_oracle_event(
                        DecodeLifecycleEventKind.REQUEST_FAILED,
                        receipt_key="decode-path-request-failure",
                    ),
                ),
            ),
            OracleLifecyclePath(
                name="decode-quarantined-owner-death",
                role=TerminalOwnerRole.DECODE,
                events=(make_oracle_event(DecodeLifecycleEventKind.OWNER_DIED),),
            ),
        )
    )
    return tuple(paths)


def exhaustive_source_transition_cases() -> tuple[OracleTransitionCase, ...]:
    """Generate every reachable-source-state by source-event pair.

    :returns: Exhaustive transition-order cases using otherwise valid receipts.
    """

    return tuple(
        OracleTransitionCase(
            name=f"{path.name}--{kind.value}",
            path=path,
            event=make_oracle_event(
                kind,
                receipt_key=(
                    f"candidate:{path.name}:{kind.value}"
                    if _receipt_requirement(kind) is not None
                    else None
                ),
            ),
        )
        for path in source_oracle_paths()
        for kind in SourceLifecycleEventKind
    )


def exhaustive_decode_transition_cases() -> tuple[OracleTransitionCase, ...]:
    """Generate every reachable-decode-state by decode-event pair.

    :returns: Exhaustive transition-order cases using otherwise valid receipts.
    """

    return tuple(
        OracleTransitionCase(
            name=f"{path.name}--{kind.value}",
            path=path,
            event=make_oracle_event(
                kind,
                receipt_key=(
                    f"candidate:{path.name}:{kind.value}"
                    if _receipt_requirement(kind) is not None
                    else None
                ),
            ),
        )
        for path in decode_oracle_paths()
        for kind in DecodeLifecycleEventKind
    )


def receipt_attack_cases() -> tuple[OracleTransitionCase, ...]:
    """Return exact trust, binding, semantics, and replay attacks.

    :returns: Receipt-bearing cases which every native reducer must reject.
    """

    source_ack = next(
        path for path in source_oracle_paths() if path.name == "source-ack_sent"
    )
    decode_adoption = next(
        path for path in decode_oracle_paths() if path.name == "decode-adoption_ready"
    )
    source_ready = next(
        path for path in source_oracle_paths() if path.name == "source-ready"
    )
    replay_key = "source-replayed-failure-authority"
    replay_path = OracleLifecyclePath(
        name="source-publication-failure-before-replay",
        role=TerminalOwnerRole.SOURCE,
        events=(
            *source_ready.events,
            make_oracle_event(
                SourceLifecycleEventKind.PUBLICATION_FAILED,
                receipt_key=replay_key,
            ),
        ),
    )
    return (
        OracleTransitionCase(
            name="source-request-ready-untrusted-issuer",
            path=source_ack,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt_key="source-untrusted",
                receipt_issuer=OracleReceiptIssuer.UNTRUSTED,
            ),
        ),
        OracleTransitionCase(
            name="source-request-ready-wrong-binding",
            path=source_ack,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt_key="source-wrong-binding",
                receipt_binding=OracleReceiptBinding.OTHER_REQUEST,
            ),
        ),
        OracleTransitionCase(
            name="source-request-ready-wrong-kind",
            path=source_ack,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt_key="source-wrong-kind",
                receipt_kind=TerminalReceiptKind.METADATA_CONSUMED,
            ),
        ),
        OracleTransitionCase(
            name="source-request-ready-wrong-outcome",
            path=source_ack,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt_key="source-wrong-outcome",
                receipt_outcome=TerminalReceiptOutcome.FAILURE,
            ),
        ),
        OracleTransitionCase(
            name="decode-adoption-untrusted-issuer",
            path=decode_adoption,
            event=make_oracle_event(
                DecodeLifecycleEventKind.ADOPTION_CONSUMED,
                receipt_key="decode-untrusted",
                receipt_issuer=OracleReceiptIssuer.UNTRUSTED,
            ),
        ),
        OracleTransitionCase(
            name="decode-adoption-wrong-binding",
            path=decode_adoption,
            event=make_oracle_event(
                DecodeLifecycleEventKind.ADOPTION_CONSUMED,
                receipt_key="decode-wrong-binding",
                receipt_binding=OracleReceiptBinding.OTHER_REQUEST,
            ),
        ),
        OracleTransitionCase(
            name="decode-adoption-wrong-kind",
            path=decode_adoption,
            event=make_oracle_event(
                DecodeLifecycleEventKind.ADOPTION_CONSUMED,
                receipt_key="decode-wrong-kind",
                receipt_kind=TerminalReceiptKind.REQUEST_READY,
            ),
        ),
        OracleTransitionCase(
            name="decode-adoption-wrong-outcome",
            path=decode_adoption,
            event=make_oracle_event(
                DecodeLifecycleEventKind.ADOPTION_CONSUMED,
                receipt_key="decode-wrong-outcome",
                receipt_outcome=TerminalReceiptOutcome.FAILURE,
            ),
        ),
        OracleTransitionCase(
            name="source-failure-receipt-replay",
            path=replay_path,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_FAILED,
                receipt_key=replay_key,
            ),
        ),
    )


class _OracleRuntime:
    """Fresh canonical reducer runtime for one path evaluation."""

    _role: TerminalOwnerRole
    _binding: TerminalRequestBinding
    _other_binding: TerminalRequestBinding
    _trusted_issuer: TerminalReceiptIssuer
    _untrusted_issuer: TerminalReceiptIssuer
    _receipts: dict[str, tuple[OracleReceiptSpec, TerminalReceipt]]
    _lifecycle: Lifecycle

    def __init__(self, role: TerminalOwnerRole) -> None:
        """Register one canonical lifecycle and its receipt namespace.

        :param role: Source or decode owner role.
        """

        self._role = role
        self._binding = _make_binding(role, room_id=71)
        self._other_binding = _make_binding(role, room_id=72)
        self._trusted_issuer = TerminalReceiptIssuer()
        self._untrusted_issuer = TerminalReceiptIssuer()
        self._receipts = {}
        ledger = TerminalReceiptLedger(
            authorities=frozenset((self._trusted_issuer.authority,))
        )
        if role is TerminalOwnerRole.SOURCE:
            self._lifecycle = create_source_lifecycle(
                binding=self._binding,
                publication_identity=TerminalPublicationIdentity(
                    request_key=self._binding.request_key,
                    publisher_process_generation=_SOURCE_PROCESS_GENERATION,
                    publication_generation=_PUBLICATION_GENERATION,
                ),
                receipt_ledger=ledger,
            )
            return
        self._lifecycle = create_decode_lifecycle(
            binding=self._binding,
            receipt_ledger=ledger,
        )

    @property
    def lifecycle(self) -> Lifecycle:
        """Return the current immutable canonical lifecycle.

        :returns: Source or decode lifecycle.
        """

        return self._lifecycle

    def apply(self, spec: OracleEventSpec) -> tuple[Lifecycle, Lifecycle]:
        """Materialize and reduce one semantic event.

        :param spec: Implementation-independent event specification.
        :returns: Previous and committed canonical lifecycle states.
        """

        if spec.role is not self._role:
            raise ValueError("oracle event targets another lifecycle role")
        previous = self._lifecycle
        receipt = self._materialize_receipt(spec.receipt)
        if self._role is TerminalOwnerRole.SOURCE:
            if type(spec.kind) is not SourceLifecycleEventKind:
                raise TypeError("source runtime requires a source event")
            if type(previous) is not SourceLifecycle:
                raise TypeError("source runtime lost its source lifecycle")
            transition = reduce_source_lifecycle(
                previous,
                SourceLifecycleEvent(
                    kind=spec.kind,
                    receipt=receipt,
                    reason=spec.reason,
                ),
            )
            self._lifecycle = transition.current
            return transition.previous, transition.current
        if type(spec.kind) is not DecodeLifecycleEventKind:
            raise TypeError("decode runtime requires a decode event")
        if type(previous) is not DecodeLifecycle:
            raise TypeError("decode runtime lost its decode lifecycle")
        transition = reduce_decode_lifecycle(
            previous,
            DecodeLifecycleEvent(
                kind=spec.kind,
                receipt=receipt,
                reason=spec.reason,
            ),
        )
        self._lifecycle = transition.current
        return transition.previous, transition.current

    def _materialize_receipt(
        self, spec: OracleReceiptSpec | None
    ) -> TerminalReceipt | None:
        """Mint or replay the exact receipt named by one stimulus.

        :param spec: Optional semantic receipt authority.
        :returns: Canonical opaque receipt, when required.
        """

        if spec is None:
            return None
        cached = self._receipts.get(spec.key)
        if cached is not None:
            cached_spec, receipt = cached
            if cached_spec != spec:
                raise ValueError("one receipt key cannot describe two authorities")
            return receipt
        issuer = (
            self._trusted_issuer
            if spec.issuer is OracleReceiptIssuer.TRUSTED
            else self._untrusted_issuer
        )
        binding = (
            self._binding
            if spec.binding is OracleReceiptBinding.TARGET
            else self._other_binding
        )
        receipt = issuer.issue(
            binding=binding,
            kind=spec.kind,
            outcome=spec.outcome,
            terminal_timestamp_ns=len(self._receipts) + 1,
        )
        self._receipts[spec.key] = (spec, receipt)
        return receipt


def _make_binding(role: TerminalOwnerRole, room_id: int) -> TerminalRequestBinding:
    """Construct one exact canonical request binding.

    :param role: Source or decode owner role.
    :param room_id: Stable packed room identity.
    :returns: Immutable request and allocation binding.
    """

    process_generation = _SOURCE_PROCESS_GENERATION
    if role is TerminalOwnerRole.DECODE:
        process_generation = _DECODE_PROCESS_GENERATION
    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=_REQUEST_GENERATION,
        ),
        owner=TerminalProcessIdentity(
            process_generation=process_generation,
            role=role,
            tp_rank=1,
            tp_size=4,
        ),
        rank_manifest_digest=_RANK_MANIFEST_DIGEST,
        allocation_digest=_ALLOCATION_DIGEST,
    )


def project_oracle_lifecycle(lifecycle: Lifecycle) -> OracleLifecycleProjection:
    """Project a canonical lifecycle into the bridge-comparison schema.

    :param lifecycle: Source or decode lifecycle state.
    :returns: Implementation-neutral phase and resource projection.
    """

    if type(lifecycle) is SourceLifecycle:
        role = TerminalOwnerRole.SOURCE
    elif type(lifecycle) is DecodeLifecycle:
        role = TerminalOwnerRole.DECODE
    else:
        raise TypeError("lifecycle must be SourceLifecycle or DecodeLifecycle")
    return OracleLifecycleProjection(
        role=role,
        phase=lifecycle.phase.value,
        live_resources=lifecycle.inventory.live,
        retired_resources=lifecycle.inventory.safely_retired,
        quarantined_resources=lifecycle.inventory.quarantined,
        process_fatal=(
            lifecycle.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
        ),
    )


def evaluate_oracle_path(path: OracleLifecyclePath) -> OraclePathEvaluation:
    """Evaluate every accepted transition in one canonical history.

    :param path: Exact path following initial registration.
    :returns: Initial and post-event state projections.
    :raises RuntimeError: If a declared path contains a rejected event.
    """

    if type(path) is not OracleLifecyclePath:
        raise TypeError("path must be OracleLifecyclePath")
    runtime = _OracleRuntime(path.role)
    states = [project_oracle_lifecycle(runtime.lifecycle)]
    for event in path.events:
        try:
            _, current = runtime.apply(event)
        except (TerminalLifecycleError, TerminalReceiptError, ValueError) as error:
            raise RuntimeError(
                f"oracle path {path.name} rejected {event.kind.value}: {error}"
            ) from error
        states.append(project_oracle_lifecycle(current))
    return OraclePathEvaluation(path=path, states=tuple(states))


def evaluate_oracle_transition(
    case: OracleTransitionCase,
) -> OracleTransitionEvaluation:
    """Adjudicate one transition against the canonical Python reducer.

    :param case: Accepted history plus one candidate event.
    :returns: Exact acceptance, state, action, receipt, and rejection oracle.
    """

    if type(case) is not OracleTransitionCase:
        raise TypeError("case must be OracleTransitionCase")
    runtime = _OracleRuntime(case.path.role)
    for event in case.path.events:
        try:
            runtime.apply(event)
        except (TerminalLifecycleError, TerminalReceiptError, ValueError) as error:
            raise RuntimeError(
                f"oracle path {case.path.name} rejected {event.kind.value}: {error}"
            ) from error
    before_lifecycle = runtime.lifecycle
    before = project_oracle_lifecycle(before_lifecycle)
    try:
        previous, current = runtime.apply(case.event)
    except TerminalReceiptError as error:
        return _rejected_evaluation(
            case, before, OracleReductionError.RECEIPT, str(error)
        )
    except TerminalLifecycleError as error:
        return _rejected_evaluation(
            case, before, OracleReductionError.LIFECYCLE, str(error)
        )
    except ValueError as error:
        return _rejected_evaluation(
            case, before, OracleReductionError.STATE_INVARIANT, str(error)
        )
    after = project_oracle_lifecycle(current)
    return OracleTransitionEvaluation(
        case=case,
        accepted=True,
        before=before,
        after=after,
        actions=_oracle_actions(previous, current),
        emitted_receipts=_oracle_emitted_receipts(previous, current),
        error=None,
        error_message=None,
    )


def _rejected_evaluation(
    case: OracleTransitionCase,
    before: OracleLifecycleProjection,
    error: OracleReductionError,
    error_message: str,
) -> OracleTransitionEvaluation:
    """Construct one side-effect-free canonical rejection.

    :param case: Rejected transition case.
    :param before: State retained after rejection.
    :param error: Stable rejection class.
    :param error_message: Canonical reducer evidence.
    :returns: Immutable rejected evaluation.
    """

    return OracleTransitionEvaluation(
        case=case,
        accepted=False,
        before=before,
        after=before,
        actions=(),
        emitted_receipts=(),
        error=error,
        error_message=error_message,
    )


def _oracle_actions(
    previous: Lifecycle, current: Lifecycle
) -> tuple[OracleOwnerAction, ...]:
    """Derive semantic owner actions earned by one canonical commit.

    :param previous: State before the event.
    :param current: State after the event.
    :returns: Ordered semantic action tuple.
    """

    actions = [OracleOwnerAction.STATE_COMMITTED]
    if type(previous) is SourceLifecycle and type(current) is SourceLifecycle:
        if current.phase is SourceLifecyclePhase.GATHERING:
            actions.append(OracleOwnerAction.SOURCE_GATHER_READY)
        if current.phase is SourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL:
            actions.append(OracleOwnerAction.SOURCE_OUTCOME_READY)
        if current.phase is SourceLifecyclePhase.TEARDOWN_RECEIVED:
            actions.append(OracleOwnerAction.SOURCE_ACK_READY)
        if (
            previous.phase is not SourceLifecyclePhase.REQUEST_READY_RECEIVED
            and current.phase is SourceLifecyclePhase.REQUEST_READY_RECEIVED
        ):
            actions.append(OracleOwnerAction.RECLAIM_AUTHORIZED)
            actions.append(OracleOwnerAction.GATEWAY_PUBLICATION_READY)
    elif type(previous) is DecodeLifecycle and type(current) is DecodeLifecycle:
        if current.phase is DecodeLifecyclePhase.SCATTER_READY:
            actions.append(OracleOwnerAction.DECODE_SCATTER_READY)
        if current.phase is DecodeLifecyclePhase.SCATTER_TERMINAL:
            actions.append(OracleOwnerAction.DECODE_TEARDOWN_READY)
        if current.phase is DecodeLifecyclePhase.ADOPTION_READY:
            actions.append(OracleOwnerAction.ADOPTION_READY)
        if current.phase is DecodeLifecyclePhase.LOCAL_DECODE_READY:
            actions.append(OracleOwnerAction.LOCAL_DECODE_READY)
    if (
        len(previous.inventory.quarantined) == 0
        and len(current.inventory.quarantined) > 0
    ):
        actions.append(OracleOwnerAction.REQUEST_QUARANTINED)
    previous_retired = (
        previous.phase is SourceLifecyclePhase.RETIRED
        if type(previous) is SourceLifecycle
        else previous.phase is DecodeLifecyclePhase.RETIRED
    )
    current_retired = (
        current.phase is SourceLifecyclePhase.RETIRED
        if type(current) is SourceLifecycle
        else current.phase is DecodeLifecyclePhase.RETIRED
    )
    if not previous_retired and current_retired:
        actions.append(OracleOwnerAction.REQUEST_RETIRED)
    if (
        previous.process_disposition is TerminalProcessDisposition.HEALTHY
        and current.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    ):
        actions.append(OracleOwnerAction.PROCESS_FATAL)
    return tuple(actions)


def _oracle_emitted_receipts(
    previous: Lifecycle, current: Lifecycle
) -> tuple[TerminalReceiptKind, ...]:
    """Derive authority receipts earned by one canonical owner commit.

    :param previous: State before the event.
    :param current: State after the event.
    :returns: Ordered receipt-kind tuple.
    """

    if type(previous) is SourceLifecycle and type(current) is SourceLifecycle:
        if (
            previous.phase is not SourceLifecyclePhase.REQUEST_READY_RECEIVED
            and current.phase is SourceLifecyclePhase.REQUEST_READY_RECEIVED
        ):
            return (TerminalReceiptKind.RECLAIM_AUTHORIZED,)
        return ()
    if type(previous) is not DecodeLifecycle or type(current) is not DecodeLifecycle:
        raise TypeError("transition crossed lifecycle roles")
    if current.phase is DecodeLifecyclePhase.ADOPTION_READY:
        return (TerminalReceiptKind.ADOPTION_READY,)
    if current.phase is DecodeLifecyclePhase.LOCAL_DECODE_READY:
        return (TerminalReceiptKind.LOCAL_DECODE_READY,)
    return ()


def deadline_boundary_cases(
    *, started_ns: int = 1_000_000_007
) -> tuple[OracleDeadlineBoundaryCase, ...]:
    """Return exact expiration boundaries for every hash-bound deadline.

    The supplied timestamp is intentionally test-controlled. Production native
    adapters must retain ownership of ``CLOCK_MONOTONIC_RAW`` and expose clock
    control only in a test build.

    :param started_ns: Synthetic one-shot start timestamp.
    :returns: One boundary case per packaged deadline.
    """

    if type(started_ns) is not int or started_ns < 0:
        raise ValueError("started_ns must be a non-negative integer")
    process_fatal_kinds = frozenset(
        (
            TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
            TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
        )
    )
    cases = []
    for spec in PACKED_TERMINAL_DEADLINES:
        bound = start_terminal_deadline(spec.kind, started_ns)
        outcome = OracleDeadlineOutcome.REQUEST_QUARANTINE
        if spec.kind in process_fatal_kinds:
            outcome = OracleDeadlineOutcome.PROCESS_FATAL
        cases.append(
            OracleDeadlineBoundaryCase(
                kind=spec.kind,
                started_ns=started_ns,
                before_expiry_ns=bound.expires_ns - 1,
                expires_ns=bound.expires_ns,
                after_expiry_ns=bound.expires_ns + 1,
                outcome=outcome,
            )
        )
    return tuple(cases)
