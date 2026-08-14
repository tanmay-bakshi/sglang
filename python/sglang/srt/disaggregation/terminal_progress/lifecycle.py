import dataclasses
import enum

from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)


class TerminalLifecycleError(RuntimeError):
    """Packed terminal-owner lifecycle invariant violation."""


class TerminalProcessDisposition(enum.StrEnum):
    """Health disposition of the process owning one lifecycle record."""

    HEALTHY = "healthy"
    PROCESS_FATAL = "process_fatal"


class TerminalResourceDisposition(enum.StrEnum):
    """Exclusive lifetime disposition for one owner-tracked resource."""

    LIVE = "live"
    SAFELY_RETIRED = "safely_retired"
    QUARANTINED = "quarantined"


class TerminalResourceKind(enum.StrEnum):
    """Resource identities conserved by the native-free owner model."""

    SOURCE_KV_PAGES = "source_kv_pages"
    SOURCE_PRODUCER_RESULTS = "source_producer_results"
    SOURCE_PACKED_LANE = "source_packed_lane"
    SOURCE_REMOTE_REGISTRATIONS = "source_remote_registrations"
    SOURCE_NIXL_HANDLES = "source_nixl_handles"
    SOURCE_RESULT_SLOT = "source_result_slot"
    SOURCE_REQUEST_IDENTITY = "source_request_identity"
    SOURCE_DFLASH_BOUNDARY_VRAM_ROWS = "source_dflash_boundary_vram_rows"
    PUBLICATION_IDENTITY = "publication_identity"
    DECODE_FULL_PAGES = "decode_full_pages"
    DECODE_SWA_PAGES = "decode_swa_pages"
    DECODE_REQUEST_SLOT = "decode_request_slot"
    DECODE_STAGING_LEASE = "decode_staging_lease"
    DECODE_DFLASH_BOUNDARY_VRAM_ROWS = "decode_dflash_boundary_vram_rows"
    DECODE_WRITER_STATE = "decode_writer_state"
    DECODE_NATIVE_IDENTITY = "decode_native_identity"


SOURCE_RECLAIMABLE_RESOURCES = frozenset(
    (
        TerminalResourceKind.SOURCE_KV_PAGES,
        TerminalResourceKind.SOURCE_PRODUCER_RESULTS,
        TerminalResourceKind.SOURCE_PACKED_LANE,
        TerminalResourceKind.SOURCE_REMOTE_REGISTRATIONS,
        TerminalResourceKind.SOURCE_NIXL_HANDLES,
        TerminalResourceKind.SOURCE_RESULT_SLOT,
        TerminalResourceKind.SOURCE_REQUEST_IDENTITY,
        TerminalResourceKind.SOURCE_DFLASH_BOUNDARY_VRAM_ROWS,
    )
)
SOURCE_RESOURCE_KINDS = SOURCE_RECLAIMABLE_RESOURCES | frozenset(
    (TerminalResourceKind.PUBLICATION_IDENTITY,)
)
DECODE_RESOURCE_KINDS = frozenset(
    (
        TerminalResourceKind.DECODE_FULL_PAGES,
        TerminalResourceKind.DECODE_SWA_PAGES,
        TerminalResourceKind.DECODE_REQUEST_SLOT,
        TerminalResourceKind.DECODE_STAGING_LEASE,
        TerminalResourceKind.DECODE_DFLASH_BOUNDARY_VRAM_ROWS,
        TerminalResourceKind.DECODE_WRITER_STATE,
        TerminalResourceKind.DECODE_NATIVE_IDENTITY,
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalResourceInventory:
    """Conserved partition of every resource owned by one request generation.

    :ivar universe: Complete resource set bound to the request generation.
    :ivar live: Resources which cannot yet be reused.
    :ivar safely_retired: Resources released by exact terminal proof.
    :ivar quarantined: Resources retained because terminality is ambiguous.
    """

    universe: frozenset[TerminalResourceKind]
    live: frozenset[TerminalResourceKind]
    safely_retired: frozenset[TerminalResourceKind]
    quarantined: frozenset[TerminalResourceKind]

    def __post_init__(self) -> None:
        """Validate exclusive and complete resource conservation."""

        values = (
            self.universe,
            self.live,
            self.safely_retired,
            self.quarantined,
        )
        if any(type(value) is not frozenset for value in values):
            raise TypeError("resource inventory fields must be frozensets")
        if any(
            type(resource) is not TerminalResourceKind
            for value in values
            for resource in value
        ):
            raise TypeError("resource inventories require TerminalResourceKind values")
        if len(self.live & self.safely_retired) > 0:
            raise ValueError("live and safely retired resources overlap")
        if len(self.live & self.quarantined) > 0:
            raise ValueError("live and quarantined resources overlap")
        if len(self.safely_retired & self.quarantined) > 0:
            raise ValueError("safely retired and quarantined resources overlap")
        classified = self.live | self.safely_retired | self.quarantined
        if classified != self.universe:
            raise ValueError("every resource must have exactly one disposition")

    @classmethod
    def all_live(
        cls, resources: frozenset[TerminalResourceKind]
    ) -> "TerminalResourceInventory":
        """Construct an inventory with every resource pinned and live.

        :param resources: Complete request-local resource set.
        :returns: Conserved all-live inventory.
        """

        if type(resources) is not frozenset or len(resources) == 0:
            raise ValueError("resources must be a non-empty frozenset")
        return cls(
            universe=resources,
            live=resources,
            safely_retired=frozenset(),
            quarantined=frozenset(),
        )

    def safely_retire(
        self, resources: frozenset[TerminalResourceKind]
    ) -> "TerminalResourceInventory":
        """Move exact live resources into the safely retired partition.

        :param resources: Resources carrying exact terminal proof.
        :returns: New conserved inventory.
        :raises TerminalLifecycleError: If any resource is not live.
        """

        self._require_live(resources)
        return dataclasses.replace(
            self,
            live=self.live - resources,
            safely_retired=self.safely_retired | resources,
        )

    def quarantine(
        self, resources: frozenset[TerminalResourceKind]
    ) -> "TerminalResourceInventory":
        """Move exact live resources into the quarantine partition.

        :param resources: Resources whose terminality is ambiguous.
        :returns: New conserved inventory.
        :raises TerminalLifecycleError: If any resource is not live.
        """

        self._require_live(resources)
        return dataclasses.replace(
            self,
            live=self.live - resources,
            quarantined=self.quarantined | resources,
        )

    def disposition(
        self, resource: TerminalResourceKind
    ) -> TerminalResourceDisposition:
        """Return the exclusive disposition of one resource.

        :param resource: Resource within this inventory's universe.
        :returns: Exclusive lifetime disposition.
        """

        if type(resource) is not TerminalResourceKind:
            raise TypeError("resource must be TerminalResourceKind")
        if resource not in self.universe:
            raise TerminalLifecycleError("resource is outside this inventory")
        if resource in self.live:
            return TerminalResourceDisposition.LIVE
        if resource in self.safely_retired:
            return TerminalResourceDisposition.SAFELY_RETIRED
        return TerminalResourceDisposition.QUARANTINED

    def _require_live(self, resources: frozenset[TerminalResourceKind]) -> None:
        """Require an exact non-empty subset of currently live resources.

        :param resources: Candidate resources to transition.
        :raises TerminalLifecycleError: If any resource is not live.
        """

        if type(resources) is not frozenset or len(resources) == 0:
            raise ValueError("resources must be a non-empty frozenset")
        if not resources <= self.live:
            raise TerminalLifecycleError("only live resources may change disposition")


class SourceLifecyclePhase(enum.StrEnum):
    """Source-owner phase independent of the terminal retirement join."""

    FROZEN = "frozen"
    WAITING_FOR_PRODUCER = "waiting_for_producer"
    GATHERING = "gathering"
    NATIVE_IN_FLIGHT = "native_in_flight"
    LOCAL_TRANSFER_TERMINAL = "local_transfer_terminal"
    OUTCOMES_SENT = "outcomes_sent"
    TEARDOWN_RECEIVED = "teardown_received"
    ACK_SENT = "ack_sent"
    REQUEST_READY_RECEIVED = "request_ready_received"
    PUBLICATION_QUARANTINED = "publication_quarantined"
    RETIRED = "retired"
    QUARANTINED = "quarantined"


class SourceLifecycleEventKind(enum.StrEnum):
    """Events accepted by the native-free source-owner reducer."""

    SUBMISSION_ACCEPTED = "submission_accepted"
    PRODUCER_COMPLETED = "producer_completed"
    GATHER_COMPLETED_AND_NATIVE_POSTED = "gather_completed_and_native_posted"
    NATIVE_TRANSFER_TERMINAL = "native_transfer_terminal"
    OUTCOMES_SENT = "outcomes_sent"
    TEARDOWN_RECEIVED = "teardown_received"
    ACK_SENT = "ack_sent"
    REQUEST_READY_RECEIVED = "request_ready_received"
    RECLAIM_CONSUMED = "reclaim_consumed"
    GATEWAY_PUBLISHED = "gateway_published"
    PUBLICATION_FAILED = "publication_failed"
    REQUEST_FAILED = "request_failed"
    OWNER_DIED = "owner_died"
    PUBLISHER_DIED = "publisher_died"
    SCHEDULER_INBOX_OVERFLOW = "scheduler_inbox_overflow"


_SOURCE_RECEIPT_EVENTS = {
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
_SOURCE_REASON_EVENTS = frozenset(
    (
        SourceLifecycleEventKind.PUBLICATION_FAILED,
        SourceLifecycleEventKind.REQUEST_FAILED,
        SourceLifecycleEventKind.OWNER_DIED,
        SourceLifecycleEventKind.PUBLISHER_DIED,
        SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class SourceLifecycleEvent:
    """One immutable input to a source-owner lifecycle reducer.

    :ivar kind: Exact transition identity.
    :ivar receipt: Trusted authority required by receipt-bearing transitions.
    :ivar reason: Stable failure reason required by failure transitions.
    """

    kind: SourceLifecycleEventKind
    receipt: TerminalReceipt | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate one structurally complete source event."""

        if type(self.kind) is not SourceLifecycleEventKind:
            raise TypeError("kind must be SourceLifecycleEventKind")
        receipt_spec = _SOURCE_RECEIPT_EVENTS.get(self.kind)
        if receipt_spec is None and self.receipt is not None:
            raise ValueError(f"{self.kind.value} does not accept a receipt")
        if receipt_spec is not None and type(self.receipt) is not TerminalReceipt:
            raise ValueError(f"{self.kind.value} requires a TerminalReceipt")
        if self.kind in _SOURCE_REASON_EVENTS:
            if type(self.reason) is not str or len(self.reason) == 0:
                raise ValueError(f"{self.kind.value} requires a non-empty reason")
        elif self.reason is not None:
            raise ValueError(f"{self.kind.value} does not accept a reason")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceLifecycle:
    """Immutable source-owner request state and conserved resource inventory.

    :ivar binding: Exact source owner and request generation.
    :ivar publication_identity: Exact canonical gateway publication generation.
    :ivar phase: Forward lifecycle phase.
    :ivar inventory: Complete resource disposition inventory.
    :ivar receipt_ledger: Trusted issuers and consumed one-shot receipts.
    :ivar reclaim_authorized: Whether exact local and global proof permits reclaim.
    :ivar reclaim_consumed: Whether the scheduler consumed reclaim authority.
    :ivar publication_authorized: Whether request-global readiness permits publish.
    :ivar gateway_published: Whether canonical publication succeeded.
    :ivar publication_identity_quarantined: Whether only publication is ambiguous.
    :ivar process_disposition: Process health after lifecycle-fatal events.
    """

    binding: TerminalRequestBinding
    publication_identity: TerminalPublicationIdentity
    phase: SourceLifecyclePhase
    inventory: TerminalResourceInventory
    receipt_ledger: TerminalReceiptLedger
    reclaim_authorized: bool = False
    reclaim_consumed: bool = False
    publication_authorized: bool = False
    gateway_published: bool = False
    publication_identity_quarantined: bool = False
    process_disposition: TerminalProcessDisposition = TerminalProcessDisposition.HEALTHY

    def __post_init__(self) -> None:
        """Validate source phase, authority joins, and resource conservation."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if self.binding.owner.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("source lifecycle requires a source owner binding")
        if type(self.publication_identity) is not TerminalPublicationIdentity:
            raise TypeError("publication_identity must be TerminalPublicationIdentity")
        if self.publication_identity.request_key != self.binding.request_key:
            raise ValueError("publication identity belongs to another request")
        if type(self.phase) is not SourceLifecyclePhase:
            raise TypeError("phase must be SourceLifecyclePhase")
        if type(self.inventory) is not TerminalResourceInventory:
            raise TypeError("inventory must be TerminalResourceInventory")
        if self.inventory.universe != SOURCE_RESOURCE_KINDS:
            raise ValueError("source lifecycle requires the complete source inventory")
        if type(self.receipt_ledger) is not TerminalReceiptLedger:
            raise TypeError("receipt_ledger must be TerminalReceiptLedger")
        flags = (
            self.reclaim_authorized,
            self.reclaim_consumed,
            self.publication_authorized,
            self.gateway_published,
            self.publication_identity_quarantined,
        )
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("source lifecycle flags must be bool values")
        if type(self.process_disposition) is not TerminalProcessDisposition:
            raise TypeError("process_disposition must be TerminalProcessDisposition")
        if self.reclaim_consumed and not self.reclaim_authorized:
            raise ValueError("reclaim consumption requires reclaim authorization")
        if self.gateway_published and not self.publication_authorized:
            raise ValueError("gateway publication requires publication authorization")
        if self.publication_identity_quarantined and not self.publication_authorized:
            raise ValueError(
                "publication quarantine requires publication authorization"
            )
        if self.gateway_published and self.publication_identity_quarantined:
            raise ValueError("publication cannot be both successful and quarantined")
        publication_disposition = self.inventory.disposition(
            TerminalResourceKind.PUBLICATION_IDENTITY
        )
        if self.gateway_published and (
            publication_disposition is not TerminalResourceDisposition.SAFELY_RETIRED
        ):
            raise ValueError("published identity must be safely retired")
        if self.publication_identity_quarantined and (
            publication_disposition is not TerminalResourceDisposition.QUARANTINED
        ):
            raise ValueError("failed publication identity must be quarantined")
        if self.reclaim_consumed and not (
            self.inventory.safely_retired >= SOURCE_RECLAIMABLE_RESOURCES
        ):
            raise ValueError("consumed reclaim authority must retire source storage")
        if self.phase is SourceLifecyclePhase.RETIRED:
            if not self.reclaim_consumed or not self.gateway_published:
                raise ValueError("RETIRED requires both terminal join branches")
            if self.inventory.safely_retired != self.inventory.universe:
                raise ValueError("RETIRED requires every resource safely retired")
        if (
            self.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
            and not self.publication_identity_quarantined
        ):
            raise ValueError("publication-quarantined phase requires its identity")
        if (
            self.phase is SourceLifecyclePhase.QUARANTINED
            and len(self.inventory.quarantined) == 0
        ):
            raise ValueError("quarantined phase requires quarantined resources")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceTransitionResult:
    """Pure source reduction result retaining both sides of the transition.

    :ivar previous: Input lifecycle record.
    :ivar current: New lifecycle record.
    """

    previous: SourceLifecycle
    current: SourceLifecycle


def create_source_lifecycle(
    binding: TerminalRequestBinding,
    publication_identity: TerminalPublicationIdentity,
    receipt_ledger: TerminalReceiptLedger,
) -> SourceLifecycle:
    """Create one all-live source owner record before submission acceptance.

    :param binding: Exact source owner and request binding.
    :param publication_identity: Exact gateway publication generation.
    :param receipt_ledger: Trusted issuer set with no consumed receipts.
    :returns: Initial immutable source lifecycle.
    """

    if len(receipt_ledger.consumed_tokens) > 0:
        raise ValueError("a new lifecycle requires an unconsumed receipt ledger")
    return SourceLifecycle(
        binding=binding,
        publication_identity=publication_identity,
        phase=SourceLifecyclePhase.FROZEN,
        inventory=TerminalResourceInventory.all_live(SOURCE_RESOURCE_KINDS),
        receipt_ledger=receipt_ledger,
    )


_SOURCE_LINEAR_TRANSITIONS = {
    (
        SourceLifecyclePhase.FROZEN,
        SourceLifecycleEventKind.SUBMISSION_ACCEPTED,
    ): SourceLifecyclePhase.WAITING_FOR_PRODUCER,
    (
        SourceLifecyclePhase.WAITING_FOR_PRODUCER,
        SourceLifecycleEventKind.PRODUCER_COMPLETED,
    ): SourceLifecyclePhase.GATHERING,
    (
        SourceLifecyclePhase.GATHERING,
        SourceLifecycleEventKind.GATHER_COMPLETED_AND_NATIVE_POSTED,
    ): SourceLifecyclePhase.NATIVE_IN_FLIGHT,
    (
        SourceLifecyclePhase.NATIVE_IN_FLIGHT,
        SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL,
    ): SourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL,
    (
        SourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL,
        SourceLifecycleEventKind.OUTCOMES_SENT,
    ): SourceLifecyclePhase.OUTCOMES_SENT,
    (
        SourceLifecyclePhase.OUTCOMES_SENT,
        SourceLifecycleEventKind.TEARDOWN_RECEIVED,
    ): SourceLifecyclePhase.TEARDOWN_RECEIVED,
    (
        SourceLifecyclePhase.TEARDOWN_RECEIVED,
        SourceLifecycleEventKind.ACK_SENT,
    ): SourceLifecyclePhase.ACK_SENT,
}


def reduce_source_lifecycle(
    lifecycle: SourceLifecycle, event: SourceLifecycleEvent
) -> SourceTransitionResult:
    """Apply one source event atomically to an immutable lifecycle record.

    :param lifecycle: Current source lifecycle.
    :param event: Exact next event.
    :returns: Pure transition result with unchanged input and new state.
    :raises TerminalLifecycleError: If the transition is illegal.
    """

    if type(lifecycle) is not SourceLifecycle:
        raise TypeError("lifecycle must be SourceLifecycle")
    if type(event) is not SourceLifecycleEvent:
        raise TypeError("event must be SourceLifecycleEvent")
    if lifecycle.phase is SourceLifecyclePhase.RETIRED:
        raise TerminalLifecycleError("a retired source lifecycle is terminal")
    if lifecycle.phase is SourceLifecyclePhase.QUARANTINED:
        raise TerminalLifecycleError("a fully quarantined source lifecycle is terminal")

    if event.kind is SourceLifecycleEventKind.OWNER_DIED:
        return _source_quarantine_all(lifecycle, process_fatal=True)
    if event.kind is SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW:
        return _source_quarantine_all(lifecycle, process_fatal=True)
    if event.kind is SourceLifecycleEventKind.PUBLISHER_DIED:
        return _source_publisher_failure(lifecycle, process_fatal=True)
    if event.kind is SourceLifecycleEventKind.REQUEST_FAILED:
        receipt_ledger = _consume_source_receipt(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=SourceLifecyclePhase.QUARANTINED,
            inventory=lifecycle.inventory.quarantine(lifecycle.inventory.live),
            receipt_ledger=receipt_ledger,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)
    if event.kind is SourceLifecycleEventKind.PUBLICATION_FAILED:
        receipt_ledger = _consume_source_receipt(lifecycle, event)
        return _source_publisher_failure(
            lifecycle,
            process_fatal=False,
            receipt_ledger=receipt_ledger,
        )

    next_phase = _SOURCE_LINEAR_TRANSITIONS.get((lifecycle.phase, event.kind))
    if next_phase is not None:
        current = dataclasses.replace(lifecycle, phase=next_phase)
        return SourceTransitionResult(previous=lifecycle, current=current)

    if event.kind is SourceLifecycleEventKind.REQUEST_READY_RECEIVED:
        if lifecycle.phase is not SourceLifecyclePhase.ACK_SENT:
            raise _illegal_source_transition(lifecycle, event)
        receipt_ledger = _consume_source_receipt(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=SourceLifecyclePhase.REQUEST_READY_RECEIVED,
            receipt_ledger=receipt_ledger,
            reclaim_authorized=True,
            publication_authorized=True,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)

    if event.kind is SourceLifecycleEventKind.RECLAIM_CONSUMED:
        allowed_phases = (
            SourceLifecyclePhase.REQUEST_READY_RECEIVED,
            SourceLifecyclePhase.PUBLICATION_QUARANTINED,
        )
        if lifecycle.phase not in allowed_phases or lifecycle.reclaim_consumed:
            raise _illegal_source_transition(lifecycle, event)
        receipt_ledger = _consume_source_receipt(lifecycle, event)
        inventory = lifecycle.inventory.safely_retire(SOURCE_RECLAIMABLE_RESOURCES)
        phase = lifecycle.phase
        if lifecycle.gateway_published:
            phase = SourceLifecyclePhase.RETIRED
        current = dataclasses.replace(
            lifecycle,
            phase=phase,
            inventory=inventory,
            receipt_ledger=receipt_ledger,
            reclaim_consumed=True,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)

    if event.kind is SourceLifecycleEventKind.GATEWAY_PUBLISHED:
        if (
            lifecycle.phase is not SourceLifecyclePhase.REQUEST_READY_RECEIVED
            or lifecycle.gateway_published
            or lifecycle.publication_identity_quarantined
        ):
            raise _illegal_source_transition(lifecycle, event)
        receipt_ledger = _consume_source_receipt(lifecycle, event)
        inventory = lifecycle.inventory.safely_retire(
            frozenset((TerminalResourceKind.PUBLICATION_IDENTITY,))
        )
        phase = lifecycle.phase
        if lifecycle.reclaim_consumed:
            phase = SourceLifecyclePhase.RETIRED
        current = dataclasses.replace(
            lifecycle,
            phase=phase,
            inventory=inventory,
            receipt_ledger=receipt_ledger,
            gateway_published=True,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)

    raise _illegal_source_transition(lifecycle, event)


def _consume_source_receipt(
    lifecycle: SourceLifecycle, event: SourceLifecycleEvent
) -> TerminalReceiptLedger:
    """Consume the exact trusted receipt required by a source event.

    :param lifecycle: Current source lifecycle.
    :param event: Receipt-bearing event.
    :returns: New immutable receipt ledger.
    """

    receipt_spec = _SOURCE_RECEIPT_EVENTS[event.kind]
    receipt = event.receipt
    if receipt is None:
        raise TerminalLifecycleError("receipt-bearing event lost its receipt")
    return lifecycle.receipt_ledger.consume(
        receipt,
        lifecycle.binding,
        receipt_spec[0],
        receipt_spec[1],
    )


def _source_quarantine_all(
    lifecycle: SourceLifecycle, process_fatal: bool
) -> SourceTransitionResult:
    """Quarantine every still-live source resource after ambiguous failure.

    :param lifecycle: Current source lifecycle.
    :param process_fatal: Whether the failure terminates the owner process.
    :returns: Pure terminal transition result.
    """

    disposition = lifecycle.process_disposition
    if process_fatal:
        disposition = TerminalProcessDisposition.PROCESS_FATAL
    if len(lifecycle.inventory.live) == 0:
        if not process_fatal:
            raise TerminalLifecycleError(
                "source failure has no live resource to quarantine"
            )
        current = dataclasses.replace(
            lifecycle,
            process_disposition=disposition,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)
    current = dataclasses.replace(
        lifecycle,
        phase=SourceLifecyclePhase.QUARANTINED,
        inventory=lifecycle.inventory.quarantine(lifecycle.inventory.live),
        process_disposition=disposition,
    )
    return SourceTransitionResult(previous=lifecycle, current=current)


def _source_publisher_failure(
    lifecycle: SourceLifecycle,
    process_fatal: bool,
    receipt_ledger: TerminalReceiptLedger | None = None,
) -> SourceTransitionResult:
    """Quarantine only publication identity once request-global proof exists.

    :param lifecycle: Current source lifecycle.
    :param process_fatal: Whether publisher death makes the process fatal.
    :param receipt_ledger: Ledger after consuming an authenticated failure.
    :returns: Pure failure transition result.
    """

    if lifecycle.gateway_published:
        if not process_fatal:
            raise TerminalLifecycleError("a published identity cannot fail publication")
        current = dataclasses.replace(
            lifecycle,
            process_disposition=TerminalProcessDisposition.PROCESS_FATAL,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)
    if lifecycle.publication_identity_quarantined:
        if not process_fatal:
            raise TerminalLifecycleError("publication identity is already quarantined")
        current = dataclasses.replace(
            lifecycle,
            process_disposition=TerminalProcessDisposition.PROCESS_FATAL,
        )
        return SourceTransitionResult(previous=lifecycle, current=current)
    if (
        lifecycle.phase is not SourceLifecyclePhase.REQUEST_READY_RECEIVED
        or not lifecycle.publication_authorized
    ):
        if process_fatal:
            return _source_quarantine_all(lifecycle, process_fatal=True)
        raise TerminalLifecycleError(
            "publication failure requires request-global publication authority"
        )
    inventory = lifecycle.inventory.quarantine(
        frozenset((TerminalResourceKind.PUBLICATION_IDENTITY,))
    )
    disposition = lifecycle.process_disposition
    if process_fatal:
        disposition = TerminalProcessDisposition.PROCESS_FATAL
    current = dataclasses.replace(
        lifecycle,
        phase=SourceLifecyclePhase.PUBLICATION_QUARANTINED,
        inventory=inventory,
        receipt_ledger=(
            receipt_ledger if receipt_ledger is not None else lifecycle.receipt_ledger
        ),
        publication_identity_quarantined=True,
        process_disposition=disposition,
    )
    return SourceTransitionResult(previous=lifecycle, current=current)


def _illegal_source_transition(
    lifecycle: SourceLifecycle, event: SourceLifecycleEvent
) -> TerminalLifecycleError:
    """Construct a stable source-transition error.

    :param lifecycle: Current source lifecycle.
    :param event: Rejected event.
    :returns: Reader-facing lifecycle error.
    """

    return TerminalLifecycleError(
        f"source event {event.kind.value} is illegal from {lifecycle.phase.value}"
    )


class DecodeLifecyclePhase(enum.StrEnum):
    """Decode-owner phase from unpublished allocation through retirement."""

    PREPARED = "prepared"
    PUBLISHED = "published"
    WRITER_AGGREGATING = "writer_aggregating"
    SCATTER_READY = "scatter_ready"
    SCATTER_IN_FLIGHT = "scatter_in_flight"
    SCATTER_TERMINAL = "scatter_terminal"
    TEARDOWN_SENT = "teardown_sent"
    ACK_AGGREGATING = "ack_aggregating"
    ADOPTION_READY = "adoption_ready"
    ADOPTED_BY_SCHEDULER = "adopted_by_scheduler"
    METADATA_CONSUMED = "metadata_consumed"
    LOCAL_DECODE_READY = "local_decode_ready"
    REQUEST_READY = "request_ready"
    RETIRED = "retired"
    QUARANTINED = "quarantined"


class DecodeLifecycleEventKind(enum.StrEnum):
    """Events accepted by the native-free decode-owner reducer."""

    ALLOCATION_PUBLISHED = "allocation_published"
    WRITER_AGGREGATION_STARTED = "writer_aggregation_started"
    WRITER_MANIFEST_COMPLETED = "writer_manifest_completed"
    SCATTER_STARTED = "scatter_started"
    SCATTER_TERMINAL = "scatter_terminal"
    TEARDOWN_SENT = "teardown_sent"
    ACK_AGGREGATION_STARTED = "ack_aggregation_started"
    ACK_MANIFEST_COMPLETED = "ack_manifest_completed"
    ADOPTION_CONSUMED = "adoption_consumed"
    METADATA_CONSUMED = "metadata_consumed"
    LOCAL_DECODE_READY_ISSUED = "local_decode_ready_issued"
    REQUEST_READY_RECEIVED = "request_ready_received"
    CANCEL_UNPUBLISHED = "cancel_unpublished"
    REQUEST_FAILED = "request_failed"
    OWNER_DIED = "owner_died"
    SCHEDULER_INBOX_OVERFLOW = "scheduler_inbox_overflow"


_DECODE_RECEIPT_EVENTS = {
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
_DECODE_REASON_EVENTS = frozenset(
    (
        DecodeLifecycleEventKind.CANCEL_UNPUBLISHED,
        DecodeLifecycleEventKind.REQUEST_FAILED,
        DecodeLifecycleEventKind.OWNER_DIED,
        DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class DecodeLifecycleEvent:
    """One immutable input to a decode-owner lifecycle reducer.

    :ivar kind: Exact transition identity.
    :ivar receipt: Trusted authority required by receipt-bearing transitions.
    :ivar reason: Stable failure reason required by failure transitions.
    """

    kind: DecodeLifecycleEventKind
    receipt: TerminalReceipt | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate one structurally complete decode event."""

        if type(self.kind) is not DecodeLifecycleEventKind:
            raise TypeError("kind must be DecodeLifecycleEventKind")
        receipt_spec = _DECODE_RECEIPT_EVENTS.get(self.kind)
        if receipt_spec is None and self.receipt is not None:
            raise ValueError(f"{self.kind.value} does not accept a receipt")
        if receipt_spec is not None and type(self.receipt) is not TerminalReceipt:
            raise ValueError(f"{self.kind.value} requires a TerminalReceipt")
        if self.kind in _DECODE_REASON_EVENTS:
            if type(self.reason) is not str or len(self.reason) == 0:
                raise ValueError(f"{self.kind.value} requires a non-empty reason")
        elif self.reason is not None:
            raise ValueError(f"{self.kind.value} does not accept a reason")


@dataclasses.dataclass(frozen=True, slots=True)
class DecodeLifecycle:
    """Immutable decode-owner request state and conserved resources.

    :ivar binding: Exact decode owner and allocation generation.
    :ivar phase: Forward lifecycle phase.
    :ivar inventory: Complete resource disposition inventory.
    :ivar receipt_ledger: Trusted issuers and consumed one-shot receipts.
    :ivar process_disposition: Process health after lifecycle-fatal events.
    """

    binding: TerminalRequestBinding
    phase: DecodeLifecyclePhase
    inventory: TerminalResourceInventory
    receipt_ledger: TerminalReceiptLedger
    process_disposition: TerminalProcessDisposition = TerminalProcessDisposition.HEALTHY

    def __post_init__(self) -> None:
        """Validate decode phase and resource conservation."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if self.binding.owner.role is not TerminalOwnerRole.DECODE:
            raise ValueError("decode lifecycle requires a decode owner binding")
        if type(self.phase) is not DecodeLifecyclePhase:
            raise TypeError("phase must be DecodeLifecyclePhase")
        if type(self.inventory) is not TerminalResourceInventory:
            raise TypeError("inventory must be TerminalResourceInventory")
        if self.inventory.universe != DECODE_RESOURCE_KINDS:
            raise ValueError("decode lifecycle requires the complete decode inventory")
        if type(self.receipt_ledger) is not TerminalReceiptLedger:
            raise TypeError("receipt_ledger must be TerminalReceiptLedger")
        if type(self.process_disposition) is not TerminalProcessDisposition:
            raise TypeError("process_disposition must be TerminalProcessDisposition")
        if (
            self.phase is DecodeLifecyclePhase.RETIRED
            and self.inventory.safely_retired != self.inventory.universe
        ):
            raise ValueError("retired decode lifecycle requires safe retirement")
        if (
            self.phase is DecodeLifecyclePhase.QUARANTINED
            and len(self.inventory.quarantined) == 0
        ):
            raise ValueError("quarantined decode lifecycle requires resources")


@dataclasses.dataclass(frozen=True, slots=True)
class DecodeTransitionResult:
    """Pure decode reduction result retaining both sides of the transition.

    :ivar previous: Input lifecycle record.
    :ivar current: New lifecycle record.
    """

    previous: DecodeLifecycle
    current: DecodeLifecycle


def create_decode_lifecycle(
    binding: TerminalRequestBinding, receipt_ledger: TerminalReceiptLedger
) -> DecodeLifecycle:
    """Create one all-live decode record before allocation publication.

    :param binding: Exact decode owner and allocation binding.
    :param receipt_ledger: Trusted issuer set with no consumed receipts.
    :returns: Initial immutable decode lifecycle.
    """

    if len(receipt_ledger.consumed_tokens) > 0:
        raise ValueError("a new lifecycle requires an unconsumed receipt ledger")
    return DecodeLifecycle(
        binding=binding,
        phase=DecodeLifecyclePhase.PREPARED,
        inventory=TerminalResourceInventory.all_live(DECODE_RESOURCE_KINDS),
        receipt_ledger=receipt_ledger,
    )


_DECODE_LINEAR_TRANSITIONS = {
    (
        DecodeLifecyclePhase.PREPARED,
        DecodeLifecycleEventKind.ALLOCATION_PUBLISHED,
    ): DecodeLifecyclePhase.PUBLISHED,
    (
        DecodeLifecyclePhase.PUBLISHED,
        DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
    ): DecodeLifecyclePhase.WRITER_AGGREGATING,
    (
        DecodeLifecyclePhase.WRITER_AGGREGATING,
        DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
    ): DecodeLifecyclePhase.SCATTER_READY,
    (
        DecodeLifecyclePhase.SCATTER_READY,
        DecodeLifecycleEventKind.SCATTER_STARTED,
    ): DecodeLifecyclePhase.SCATTER_IN_FLIGHT,
    (
        DecodeLifecyclePhase.SCATTER_IN_FLIGHT,
        DecodeLifecycleEventKind.SCATTER_TERMINAL,
    ): DecodeLifecyclePhase.SCATTER_TERMINAL,
    (
        DecodeLifecyclePhase.SCATTER_TERMINAL,
        DecodeLifecycleEventKind.TEARDOWN_SENT,
    ): DecodeLifecyclePhase.TEARDOWN_SENT,
    (
        DecodeLifecyclePhase.TEARDOWN_SENT,
        DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
    ): DecodeLifecyclePhase.ACK_AGGREGATING,
    (
        DecodeLifecyclePhase.ACK_AGGREGATING,
        DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
    ): DecodeLifecyclePhase.ADOPTION_READY,
    (
        DecodeLifecyclePhase.ADOPTED_BY_SCHEDULER,
        DecodeLifecycleEventKind.METADATA_CONSUMED,
    ): DecodeLifecyclePhase.METADATA_CONSUMED,
    (
        DecodeLifecyclePhase.METADATA_CONSUMED,
        DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED,
    ): DecodeLifecyclePhase.LOCAL_DECODE_READY,
}


def reduce_decode_lifecycle(
    lifecycle: DecodeLifecycle, event: DecodeLifecycleEvent
) -> DecodeTransitionResult:
    """Apply one decode event atomically to an immutable lifecycle record.

    :param lifecycle: Current decode lifecycle.
    :param event: Exact next event.
    :returns: Pure transition result with unchanged input and new state.
    :raises TerminalLifecycleError: If the transition is illegal.
    """

    if type(lifecycle) is not DecodeLifecycle:
        raise TypeError("lifecycle must be DecodeLifecycle")
    if type(event) is not DecodeLifecycleEvent:
        raise TypeError("event must be DecodeLifecycleEvent")
    if lifecycle.phase in (
        DecodeLifecyclePhase.RETIRED,
        DecodeLifecyclePhase.QUARANTINED,
    ):
        raise TerminalLifecycleError(
            f"a {lifecycle.phase.value} decode lifecycle is terminal"
        )

    if event.kind is DecodeLifecycleEventKind.OWNER_DIED:
        return _decode_quarantine_all(lifecycle, process_fatal=True)
    if event.kind is DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW:
        return _decode_quarantine_all(lifecycle, process_fatal=True)
    if event.kind is DecodeLifecycleEventKind.CANCEL_UNPUBLISHED:
        if lifecycle.phase is not DecodeLifecyclePhase.PREPARED:
            raise _illegal_decode_transition(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=DecodeLifecyclePhase.RETIRED,
            inventory=lifecycle.inventory.safely_retire(lifecycle.inventory.live),
        )
        return DecodeTransitionResult(previous=lifecycle, current=current)
    if event.kind is DecodeLifecycleEventKind.REQUEST_FAILED:
        if lifecycle.phase is DecodeLifecyclePhase.PREPARED:
            raise _illegal_decode_transition(lifecycle, event)
        receipt_ledger = _consume_decode_receipt(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=DecodeLifecyclePhase.QUARANTINED,
            inventory=lifecycle.inventory.quarantine(lifecycle.inventory.live),
            receipt_ledger=receipt_ledger,
        )
        return DecodeTransitionResult(previous=lifecycle, current=current)

    next_phase = _DECODE_LINEAR_TRANSITIONS.get((lifecycle.phase, event.kind))
    if next_phase is not None:
        receipt_ledger = lifecycle.receipt_ledger
        if event.kind in _DECODE_RECEIPT_EVENTS:
            receipt_ledger = _consume_decode_receipt(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=next_phase,
            receipt_ledger=receipt_ledger,
        )
        return DecodeTransitionResult(previous=lifecycle, current=current)

    if event.kind is DecodeLifecycleEventKind.ADOPTION_CONSUMED:
        if lifecycle.phase is not DecodeLifecyclePhase.ADOPTION_READY:
            raise _illegal_decode_transition(lifecycle, event)
        receipt_ledger = _consume_decode_receipt(lifecycle, event)
        current = dataclasses.replace(
            lifecycle,
            phase=DecodeLifecyclePhase.ADOPTED_BY_SCHEDULER,
            receipt_ledger=receipt_ledger,
        )
        return DecodeTransitionResult(previous=lifecycle, current=current)

    if event.kind is DecodeLifecycleEventKind.REQUEST_READY_RECEIVED:
        if lifecycle.phase is not DecodeLifecyclePhase.LOCAL_DECODE_READY:
            raise _illegal_decode_transition(lifecycle, event)
        receipt_ledger = _consume_decode_receipt(lifecycle, event)
        request_ready = dataclasses.replace(
            lifecycle,
            phase=DecodeLifecyclePhase.REQUEST_READY,
            receipt_ledger=receipt_ledger,
        )
        current = dataclasses.replace(
            request_ready,
            phase=DecodeLifecyclePhase.RETIRED,
            inventory=request_ready.inventory.safely_retire(
                request_ready.inventory.live
            ),
        )
        return DecodeTransitionResult(previous=lifecycle, current=current)

    raise _illegal_decode_transition(lifecycle, event)


def _consume_decode_receipt(
    lifecycle: DecodeLifecycle, event: DecodeLifecycleEvent
) -> TerminalReceiptLedger:
    """Consume the exact trusted receipt required by a decode event.

    :param lifecycle: Current decode lifecycle.
    :param event: Receipt-bearing event.
    :returns: New immutable receipt ledger.
    """

    receipt_spec = _DECODE_RECEIPT_EVENTS[event.kind]
    receipt = event.receipt
    if receipt is None:
        raise TerminalLifecycleError("receipt-bearing event lost its receipt")
    return lifecycle.receipt_ledger.consume(
        receipt,
        lifecycle.binding,
        receipt_spec[0],
        receipt_spec[1],
    )


def _decode_quarantine_all(
    lifecycle: DecodeLifecycle, process_fatal: bool
) -> DecodeTransitionResult:
    """Quarantine every still-live decode resource after ambiguous failure.

    :param lifecycle: Current decode lifecycle.
    :param process_fatal: Whether the failure terminates the owner process.
    :returns: Pure terminal transition result.
    """

    if len(lifecycle.inventory.live) == 0:
        raise TerminalLifecycleError(
            "decode failure has no live resource to quarantine"
        )
    disposition = lifecycle.process_disposition
    if process_fatal:
        disposition = TerminalProcessDisposition.PROCESS_FATAL
    current = dataclasses.replace(
        lifecycle,
        phase=DecodeLifecyclePhase.QUARANTINED,
        inventory=lifecycle.inventory.quarantine(lifecycle.inventory.live),
        process_disposition=disposition,
    )
    return DecodeTransitionResult(previous=lifecycle, current=current)


def _illegal_decode_transition(
    lifecycle: DecodeLifecycle, event: DecodeLifecycleEvent
) -> TerminalLifecycleError:
    """Construct a stable decode-transition error.

    :param lifecycle: Current decode lifecycle.
    :param event: Rejected event.
    :returns: Reader-facing lifecycle error.
    """

    return TerminalLifecycleError(
        f"decode event {event.kind.value} is illegal from {lifecycle.phase.value}"
    )
