import dataclasses
import enum
import threading
from collections.abc import Callable
from typing import ClassVar, Final, Protocol

from nixl._api import (
    nixl_agent,
    nixl_remote_agent_handle,
    nixl_terminal_backend_lifecycle_inventory,
    nixl_terminal_capability_state_t,
    nixl_terminal_channel_fatal_t,
    nixl_terminal_channel_inventory,
    nixl_terminal_deadline,
    nixl_terminal_destination_delivery,
    nixl_terminal_destination_phase_t,
    nixl_terminal_event,
    nixl_terminal_event_channel,
    nixl_terminal_event_kind_t,
    nixl_terminal_event_subscription,
    nixl_terminal_owner_producer,
    nixl_terminal_owner_producer_inventory,
    nixl_terminal_owner_subscription,
    nixl_terminal_source_delivery,
    nixl_terminal_subscription_info,
    nixl_xfer_completion_receipt,
    nixl_xfer_handle,
)
from nixl._bindings import NIXL_IN_PROG, NIXL_SUCCESS, nixl_status_t
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalNativeProducerBinding,
    NativeTerminalRuntime,
)

QUALIFIED_NIXL_REVISION: Final[str] = "b7e05f416c2ad50376a626a8aa25c2907f47e7c6"
QUALIFIED_NIXL_DIRECT_OWNER_REVISION: Final[str] = (
    "f04c03c7c0b6b51e67d743b5e13192524cf6b58a"
)


class NixlTerminalEventKind(enum.StrEnum):
    """Kind of one event projected from the native terminal channel."""

    TRANSFER = "transfer"
    CAPABILITY = "capability"


class NixlTerminalCapabilityState(enum.StrEnum):
    """State of one exact remote notification route."""

    READY = "ready"
    FAILED = "failed"
    RETIRED = "retired"


class NixlTerminalDestinationPhase(enum.StrEnum):
    """Active phase of one native destination delivery."""

    PENDING = "pending"
    ADMITTING = "admitting"
    COMMITTED = "committed"
    REPLAYING = "replaying"
    QUARANTINED = "quarantined"


class NixlTerminalChannelFatal(enum.IntFlag):
    """Sticky process-fatal state projected from the native channel."""

    NONE = 0
    QUEUE_OVERFLOW = 1 << 0
    EVENTFD_FAILURE = 1 << 1
    ACTIVE_SUBSCRIPTIONS_ON_CLOSE = 1 << 2
    INVALID_PUBLICATION = 1 << 3


class NixlTerminalCancelOutcome(enum.StrEnum):
    """Outcome of the one allowed explicit cancellation request."""

    RELEASED = "released"
    PENDING_TERMINAL_EVENT = "pending_terminal_event"


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalSourceDelivery:
    """Immutable projection of one active native source delivery.

    :ivar backend: Native backend name.
    :ivar delivery_identity: Backend-local delivery identity.
    :ivar source_handle_identity: Exact source transfer-handle identity.
    :ivar source_generation: Exact source transfer generation.
    :ivar local_pending: Whether local completion remains outstanding.
    :ivar receipt_pending: Whether remote admission receipt remains outstanding.
    :ivar deadline_active: Whether the native deadline remains armed.
    """

    backend: str
    delivery_identity: int
    source_handle_identity: int
    source_generation: int
    local_pending: bool
    receipt_pending: bool
    deadline_active: bool


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalDestinationDelivery:
    """Immutable projection of one active native destination delivery.

    :ivar backend: Native backend name.
    :ivar source_backend_incarnation: Exact source-backend incarnation.
    :ivar source_handle_identity: Exact source transfer-handle identity.
    :ivar source_generation: Exact source transfer generation.
    :ivar delivery_identity: Backend-local delivery identity.
    :ivar phase: Current destination lifecycle phase.
    """

    backend: str
    source_backend_incarnation: str
    source_handle_identity: int
    source_generation: int
    delivery_identity: int
    phase: NixlTerminalDestinationPhase


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalDeadline:
    """Immutable projection of one active native transfer deadline.

    :ivar backend: Native backend name.
    :ivar handle_identity: Exact transfer-handle identity.
    :ivar generation: Exact transfer generation.
    """

    backend: str
    handle_identity: int
    generation: int


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalBackendLifecycleInventory:
    """Immutable native delivery and deadline inventory.

    :ivar source_deliveries_outstanding: Active source-delivery count.
    :ivar source_local_pending: Source deliveries awaiting local completion.
    :ivar source_receipt_pending: Source deliveries awaiting remote admission.
    :ivar destination_pending: Destination deliveries awaiting admission.
    :ivar destination_admitting: Destination deliveries being admitted.
    :ivar destination_committed: Committed destination deliveries.
    :ivar destination_replaying: Destination deliveries replaying an ACK.
    :ivar destination_quarantined: Quarantined destination deliveries.
    :ivar active_native_deadlines: Active native deadline count.
    :ivar source_deliveries: Exact active source-delivery identities.
    :ivar destination_deliveries: Exact active destination-delivery identities.
    :ivar native_deadlines: Exact active deadline identities.
    """

    source_deliveries_outstanding: int
    source_local_pending: int
    source_receipt_pending: int
    destination_pending: int
    destination_admitting: int
    destination_committed: int
    destination_replaying: int
    destination_quarantined: int
    active_native_deadlines: int
    source_deliveries: tuple[NixlTerminalSourceDelivery, ...]
    destination_deliveries: tuple[NixlTerminalDestinationDelivery, ...]
    native_deadlines: tuple[NixlTerminalDeadline, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether every native delivery obligation is absent.

        :returns: Whether the lifecycle inventory is exactly empty.
        """

        return (
            self.source_deliveries_outstanding == 0
            and self.source_local_pending == 0
            and self.source_receipt_pending == 0
            and self.destination_pending == 0
            and self.destination_admitting == 0
            and self.destination_committed == 0
            and self.destination_replaying == 0
            and self.destination_quarantined == 0
            and self.active_native_deadlines == 0
            and len(self.source_deliveries) == 0
            and len(self.destination_deliveries) == 0
            and len(self.native_deadlines) == 0
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalChannelInventory:
    """Immutable channel and native-producer inventory.

    :ivar capacity: Maximum queued event count.
    :ivar queued_channel_events: Events not yet drained.
    :ivar active_channel_subscriptions: Subscriptions still producing events.
    :ivar retained_public_subscriptions: Native handles awaiting release.
    :ivar backend_producers: Backend callbacks able to publish.
    :ivar active_callback_slots: Callbacks currently inside the publication path.
    :ivar queued_owner_continuations: Backend owner continuations awaiting work.
    :ivar backend_lifecycle: Native delivery and deadline inventory.
    :ivar accepting_subscriptions: Whether the channel accepts new subscriptions.
    :ivar closed: Whether native channel closure completed.
    :ivar fatal: Sticky process-fatal channel flags.
    :ivar eventfd_error: Errno captured from an eventfd failure.
    """

    capacity: int
    queued_channel_events: int
    active_channel_subscriptions: int
    retained_public_subscriptions: int
    backend_producers: int
    active_callback_slots: int
    queued_owner_continuations: int
    backend_lifecycle: NixlTerminalBackendLifecycleInventory
    accepting_subscriptions: bool
    closed: bool
    fatal: NixlTerminalChannelFatal
    eventfd_error: int

    @property
    def is_clean_closed(self) -> bool:
        """Return whether closure restored exact zero lifecycle inventory.

        :returns: Whether every close invariant holds.
        """

        return (
            self.queued_channel_events == 0
            and self.active_channel_subscriptions == 0
            and self.retained_public_subscriptions == 0
            and self.backend_producers == 0
            and self.active_callback_slots == 0
            and self.queued_owner_continuations == 0
            and self.backend_lifecycle.is_empty
            and not self.accepting_subscriptions
            and self.closed
            and self.fatal == NixlTerminalChannelFatal.NONE
            and self.eventfd_error == 0
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalSubscriptionBinding:
    """Exact native subscription identity owned by the adapter.

    :ivar kind: Transfer or route-capability subscription kind.
    :ivar owner_cookie: Opaque SGLang owner correlation identity.
    :ivar identity: Native transfer or remote-agent handle identity.
    :ivar generation: Exact subscribed generation.
    """

    kind: NixlTerminalEventKind
    owner_cookie: int
    identity: int
    generation: int


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTransferTerminalEvent:
    """Immutable exact-generation transfer terminal event.

    :ivar owner_cookie: Opaque SGLang owner correlation identity.
    :ivar identity: Native transfer-handle identity.
    :ivar generation: Exact completed transfer generation.
    :ivar status: Native terminal transfer status.
    :ivar native_timestamp_ns: Native monotonic publication timestamp.
    """

    kind: ClassVar[NixlTerminalEventKind] = NixlTerminalEventKind.TRANSFER

    owner_cookie: int
    identity: int
    generation: int
    status: nixl_status_t
    native_timestamp_ns: int

    @property
    def binding(self) -> NixlTerminalSubscriptionBinding:
        """Return the exact subscription binding carried by this event.

        :returns: Transfer subscription binding.
        """

        return NixlTerminalSubscriptionBinding(
            kind=self.kind,
            owner_cookie=self.owner_cookie,
            identity=self.identity,
            generation=self.generation,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NixlCapabilityTerminalEvent:
    """Immutable event for one exact remote notification route.

    :ivar owner_cookie: Opaque SGLang owner correlation identity.
    :ivar identity: Native remote-agent handle identity.
    :ivar generation: Exact remote-agent generation.
    :ivar state: Current route capability state.
    :ivar capability_epoch: Monotonic route-state epoch.
    :ivar native_timestamp_ns: Native monotonic publication timestamp.
    """

    kind: ClassVar[NixlTerminalEventKind] = NixlTerminalEventKind.CAPABILITY

    owner_cookie: int
    identity: int
    generation: int
    state: NixlTerminalCapabilityState
    capability_epoch: int
    native_timestamp_ns: int

    @property
    def binding(self) -> NixlTerminalSubscriptionBinding:
        """Return the exact subscription binding carried by this event.

        :returns: Route-capability subscription binding.
        """

        return NixlTerminalSubscriptionBinding(
            kind=self.kind,
            owner_cookie=self.owner_cookie,
            identity=self.identity,
            generation=self.generation,
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether the route can publish no later state transition.

        :returns: Whether this event is ``FAILED`` or ``RETIRED``.
        """

        return self.state in (
            NixlTerminalCapabilityState.FAILED,
            NixlTerminalCapabilityState.RETIRED,
        )


NixlTerminalEvent = NixlTransferTerminalEvent | NixlCapabilityTerminalEvent


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalEventBatch:
    """One immutable native channel drain result.

    :ivar events: Events drained in native publication order.
    :ivar wake_count: Coalesced eventfd wake count consumed by the drain.
    :ivar inventory: Post-drain native lifecycle inventory.
    """

    events: tuple[NixlTerminalEvent, ...]
    wake_count: int
    inventory: NixlTerminalChannelInventory


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalSubscription:
    """Opaque adapter-owned exact-generation subscription.

    :ivar binding: Immutable native subscription binding.
    """

    binding: NixlTerminalSubscriptionBinding


class NixlTerminalProcessFatalError(RuntimeError):
    """Process-fatal terminal channel or publication failure."""

    inventory: NixlTerminalChannelInventory | None
    reason: str

    def __init__(
        self,
        reason: str,
        inventory: NixlTerminalChannelInventory | None,
    ) -> None:
        """Create one process-fatal propagation value.

        :param reason: Precise failure context.
        :param inventory: Latest immutable channel inventory, when projection
            succeeded.
        """

        self.reason = reason
        self.inventory = inventory
        if inventory is None:
            super().__init__(reason)
            return
        super().__init__(
            f"{reason}; fatal={inventory.fatal!s}; "
            f"eventfd_error={inventory.eventfd_error}"
        )


class NixlTerminalLifecycleError(RuntimeError):
    """Adapter subscription or channel lifecycle invariant violation."""


class NixlTerminalAgentBoundary(Protocol):
    """Strict high-level NIXL API surface consumed by the adapter."""

    def create_terminal_event_channel(
        self, capacity: int
    ) -> nixl_terminal_event_channel:
        """Create the agent's sole native terminal channel.

        :param capacity: Positive event capacity.
        :returns: Agent-owned channel.
        """

        ...

    def subscribe_xfer_terminal(
        self,
        channel: nixl_terminal_event_channel,
        handle: nixl_xfer_handle,
        owner_cookie: int,
    ) -> nixl_terminal_event_subscription:
        """Subscribe to one transfer's exact next generation.

        :param channel: Agent-owned terminal channel.
        :param handle: Agent-owned transfer handle.
        :param owner_cookie: Opaque SGLang owner identity.
        :returns: Exact-generation native subscription.
        """

        ...

    def subscribe_remote_notification_state(
        self,
        channel: nixl_terminal_event_channel,
        remote_agent: nixl_remote_agent_handle,
        backend: str,
        owner_cookie: int,
    ) -> nixl_terminal_event_subscription:
        """Subscribe to one exact remote notification route.

        :param channel: Agent-owned terminal channel.
        :param remote_agent: Agent-owned remote handle.
        :param backend: Instantiated native backend name.
        :param owner_cookie: Opaque SGLang owner identity.
        :returns: Exact-route native subscription.
        """

        ...


@dataclasses.dataclass(slots=True)
class _OwnedSubscription:
    """Native handle and one-shot lifecycle retained by the adapter.

    :ivar public: Opaque public subscription identity.
    :ivar native: Native subscription wrapper.
    :ivar cancel_requested: Whether the sole explicit cancellation was spent.
    """

    public: NixlTerminalSubscription
    native: nixl_terminal_event_subscription
    cancel_requested: bool = False


def _require_positive(value: int, label: str) -> int:
    """Require one positive native identity.

    :param value: Candidate integer.
    :param label: Reader-facing field name.
    :returns: Validated integer.
    :raises NixlTerminalLifecycleError: If the value is not positive.
    """

    if type(value) is not int or value <= 0:
        raise NixlTerminalLifecycleError(f"{label} must be a positive integer")
    return value


def _require_nonnegative(value: int, label: str) -> int:
    """Require one non-negative native counter.

    :param value: Candidate integer.
    :param label: Reader-facing field name.
    :returns: Validated integer.
    :raises NixlTerminalLifecycleError: If the value is negative.
    """

    if type(value) is not int or value < 0:
        raise NixlTerminalLifecycleError(f"{label} must be a non-negative integer")
    return value


def _require_backend(value: object, label: str) -> str:
    """Require one non-empty native backend name.

    :param value: Candidate native backend value.
    :param label: Reader-facing field name.
    :returns: Validated backend name.
    :raises NixlTerminalLifecycleError: If the backend name is empty.
    """

    backend = str(value)
    if len(backend) == 0:
        raise NixlTerminalLifecycleError(f"{label} must not be empty")
    return backend


def _project_event_kind(
    value: nixl_terminal_event_kind_t,
) -> NixlTerminalEventKind:
    """Project one exact native event kind.

    :param value: Native event kind.
    :returns: SGLang event kind.
    :raises NixlTerminalLifecycleError: If the native value is unknown.
    """

    if value == nixl_terminal_event_kind_t.TRANSFER:
        return NixlTerminalEventKind.TRANSFER
    if value == nixl_terminal_event_kind_t.CAPABILITY:
        return NixlTerminalEventKind.CAPABILITY
    raise NixlTerminalLifecycleError(f"unknown native terminal event kind: {value}")


def _project_capability_state(
    value: nixl_terminal_capability_state_t,
) -> NixlTerminalCapabilityState:
    """Project one exact native route state.

    :param value: Native route state.
    :returns: SGLang route state.
    :raises NixlTerminalLifecycleError: If the native value is unknown.
    """

    if value == nixl_terminal_capability_state_t.READY:
        return NixlTerminalCapabilityState.READY
    if value == nixl_terminal_capability_state_t.FAILED:
        return NixlTerminalCapabilityState.FAILED
    if value == nixl_terminal_capability_state_t.RETIRED:
        return NixlTerminalCapabilityState.RETIRED
    raise NixlTerminalLifecycleError(
        f"unknown native terminal capability state: {value}"
    )


def _project_destination_phase(
    value: nixl_terminal_destination_phase_t,
) -> NixlTerminalDestinationPhase:
    """Project one exact native destination phase.

    :param value: Native destination phase.
    :returns: SGLang destination phase.
    :raises NixlTerminalLifecycleError: If the native value is unknown.
    """

    if value == nixl_terminal_destination_phase_t.PENDING:
        return NixlTerminalDestinationPhase.PENDING
    if value == nixl_terminal_destination_phase_t.ADMITTING:
        return NixlTerminalDestinationPhase.ADMITTING
    if value == nixl_terminal_destination_phase_t.COMMITTED:
        return NixlTerminalDestinationPhase.COMMITTED
    if value == nixl_terminal_destination_phase_t.REPLAYING:
        return NixlTerminalDestinationPhase.REPLAYING
    if value == nixl_terminal_destination_phase_t.QUARANTINED:
        return NixlTerminalDestinationPhase.QUARANTINED
    raise NixlTerminalLifecycleError(
        f"unknown native terminal destination phase: {value}"
    )


def _project_fatal(
    value: nixl_terminal_channel_fatal_t,
) -> NixlTerminalChannelFatal:
    """Project the native sticky fatal bitmask without losing unknown bits.

    :param value: Native fatal bitmask.
    :returns: SGLang fatal bitmask.
    """

    return NixlTerminalChannelFatal(int(value.value))


def _project_source_delivery(
    value: nixl_terminal_source_delivery,
) -> NixlTerminalSourceDelivery:
    """Project one exact native source delivery.

    :param value: Native source-delivery snapshot.
    :returns: Immutable SGLang source delivery.
    """

    return NixlTerminalSourceDelivery(
        backend=_require_backend(value.backend, "source backend"),
        delivery_identity=_require_positive(
            int(value.deliveryIdentity), "source delivery identity"
        ),
        source_handle_identity=_require_positive(
            int(value.sourceHandleIdentity), "source handle identity"
        ),
        source_generation=_require_positive(
            int(value.sourceGeneration), "source generation"
        ),
        local_pending=bool(value.localPending),
        receipt_pending=bool(value.receiptPending),
        deadline_active=bool(value.deadlineActive),
    )


def _project_destination_delivery(
    value: nixl_terminal_destination_delivery,
) -> NixlTerminalDestinationDelivery:
    """Project one exact native destination delivery.

    :param value: Native destination-delivery snapshot.
    :returns: Immutable SGLang destination delivery.
    """

    source_backend_incarnation = str(value.sourceBackendIncarnation)
    if len(source_backend_incarnation) == 0:
        raise NixlTerminalLifecycleError("source backend incarnation must not be empty")
    return NixlTerminalDestinationDelivery(
        backend=_require_backend(value.backend, "destination backend"),
        source_backend_incarnation=source_backend_incarnation,
        source_handle_identity=_require_positive(
            int(value.sourceHandleIdentity), "destination source handle identity"
        ),
        source_generation=_require_positive(
            int(value.sourceGeneration), "destination source generation"
        ),
        delivery_identity=_require_positive(
            int(value.deliveryIdentity), "destination delivery identity"
        ),
        phase=_project_destination_phase(value.phase),
    )


def _project_deadline(value: nixl_terminal_deadline) -> NixlTerminalDeadline:
    """Project one exact native transfer deadline.

    :param value: Native deadline snapshot.
    :returns: Immutable SGLang deadline.
    """

    return NixlTerminalDeadline(
        backend=_require_backend(value.backend, "deadline backend"),
        handle_identity=_require_positive(
            int(value.handleIdentity), "deadline handle identity"
        ),
        generation=_require_positive(int(value.generation), "deadline generation"),
    )


def _project_backend_inventory(
    value: nixl_terminal_backend_lifecycle_inventory,
) -> NixlTerminalBackendLifecycleInventory:
    """Project native backend lifecycle inventory.

    :param value: Native backend inventory snapshot.
    :returns: Immutable SGLang backend inventory.
    """

    source_deliveries = tuple(
        _project_source_delivery(delivery) for delivery in value.sourceDeliveries
    )
    destination_deliveries = tuple(
        _project_destination_delivery(delivery)
        for delivery in value.destinationDeliveries
    )
    native_deadlines = tuple(
        _project_deadline(deadline) for deadline in value.nativeDeadlines
    )
    inventory = NixlTerminalBackendLifecycleInventory(
        source_deliveries_outstanding=_require_nonnegative(
            int(value.sourceDeliveriesOutstanding),
            "source deliveries outstanding",
        ),
        source_local_pending=_require_nonnegative(
            int(value.sourceLocalPending), "source local pending"
        ),
        source_receipt_pending=_require_nonnegative(
            int(value.sourceReceiptPending), "source receipt pending"
        ),
        destination_pending=_require_nonnegative(
            int(value.destinationPending), "destination pending"
        ),
        destination_admitting=_require_nonnegative(
            int(value.destinationAdmitting), "destination admitting"
        ),
        destination_committed=_require_nonnegative(
            int(value.destinationCommitted), "destination committed"
        ),
        destination_replaying=_require_nonnegative(
            int(value.destinationReplaying), "destination replaying"
        ),
        destination_quarantined=_require_nonnegative(
            int(value.destinationQuarantined), "destination quarantined"
        ),
        active_native_deadlines=_require_nonnegative(
            int(value.activeNativeDeadlines), "active native deadlines"
        ),
        source_deliveries=source_deliveries,
        destination_deliveries=destination_deliveries,
        native_deadlines=native_deadlines,
    )
    destination_count = (
        inventory.destination_pending
        + inventory.destination_admitting
        + inventory.destination_committed
        + inventory.destination_replaying
        + inventory.destination_quarantined
    )
    if inventory.source_deliveries_outstanding != len(source_deliveries):
        raise NixlTerminalLifecycleError(
            "source delivery count disagrees with exact native inventory"
        )
    if inventory.source_local_pending > inventory.source_deliveries_outstanding:
        raise NixlTerminalLifecycleError(
            "source local-pending count exceeds outstanding deliveries"
        )
    if inventory.source_receipt_pending > inventory.source_deliveries_outstanding:
        raise NixlTerminalLifecycleError(
            "source receipt-pending count exceeds outstanding deliveries"
        )
    if destination_count != len(destination_deliveries):
        raise NixlTerminalLifecycleError(
            "destination delivery counts disagree with exact native inventory"
        )
    if inventory.active_native_deadlines != len(native_deadlines):
        raise NixlTerminalLifecycleError(
            "deadline count disagrees with exact native inventory"
        )
    return inventory


def project_nixl_terminal_inventory(
    value: nixl_terminal_channel_inventory,
) -> NixlTerminalChannelInventory:
    """Project one immutable native channel inventory.

    :param value: Native channel inventory snapshot.
    :returns: Immutable SGLang channel inventory.
    """

    return NixlTerminalChannelInventory(
        capacity=_require_positive(int(value.capacity), "channel capacity"),
        queued_channel_events=_require_nonnegative(
            int(value.queuedChannelEvents), "queued channel events"
        ),
        active_channel_subscriptions=_require_nonnegative(
            int(value.activeChannelSubscriptions),
            "active channel subscriptions",
        ),
        retained_public_subscriptions=_require_nonnegative(
            int(value.retainedPublicSubscriptions),
            "retained public subscriptions",
        ),
        backend_producers=_require_nonnegative(
            int(value.backendProducers), "backend producers"
        ),
        active_callback_slots=_require_nonnegative(
            int(value.activeCallbackSlots), "active callback slots"
        ),
        queued_owner_continuations=_require_nonnegative(
            int(value.queuedOwnerContinuations), "queued owner continuations"
        ),
        backend_lifecycle=_project_backend_inventory(value.backendLifecycle),
        accepting_subscriptions=bool(value.acceptingSubscriptions),
        closed=bool(value.closed),
        fatal=_project_fatal(value.fatal),
        eventfd_error=_require_nonnegative(int(value.eventfdError), "eventfd error"),
    )


def _project_subscription(
    value: nixl_terminal_subscription_info,
) -> tuple[NixlTerminalSubscriptionBinding, bool]:
    """Project one exact native subscription snapshot.

    :param value: Native subscription snapshot.
    :returns: Exact binding and current active state.
    """

    return (
        NixlTerminalSubscriptionBinding(
            kind=_project_event_kind(value.kind),
            owner_cookie=_require_positive(
                int(value.ownerCookie), "subscription owner cookie"
            ),
            identity=_require_positive(int(value.identity), "subscription identity"),
            generation=_require_positive(
                int(value.generation), "subscription generation"
            ),
        ),
        bool(value.active),
    )


def project_nixl_terminal_event(value: nixl_terminal_event) -> NixlTerminalEvent:
    """Project one immutable exact-generation native event.

    :param value: Native terminal event snapshot.
    :returns: Immutable SGLang event.
    """

    kind = _project_event_kind(value.kind)
    owner_cookie = _require_positive(int(value.ownerCookie), "event owner cookie")
    identity = _require_positive(int(value.identity), "event identity")
    generation = _require_positive(int(value.generation), "event generation")
    native_timestamp_ns = _require_positive(
        int(value.nativeTimestampNs), "native event timestamp"
    )
    if kind is NixlTerminalEventKind.TRANSFER:
        status = value.transferStatus
        if status is None:
            raise NixlTerminalLifecycleError(
                "transfer event omitted its native terminal status"
            )
        return NixlTransferTerminalEvent(
            owner_cookie=owner_cookie,
            identity=identity,
            generation=generation,
            status=status,
            native_timestamp_ns=native_timestamp_ns,
        )

    state = value.capabilityState
    capability_epoch = value.capabilityEpoch
    if state is None or capability_epoch is None:
        raise NixlTerminalLifecycleError("capability event omitted its state or epoch")
    return NixlCapabilityTerminalEvent(
        owner_cookie=owner_cookie,
        identity=identity,
        generation=generation,
        state=_project_capability_state(state),
        capability_epoch=_require_positive(int(capability_epoch), "capability epoch"),
        native_timestamp_ns=native_timestamp_ns,
    )


class NixlTerminalEventAdapter:
    """SGLang ownership boundary for one qualified native terminal channel."""

    _agent: NixlTerminalAgentBoundary
    _channel: nixl_terminal_event_channel
    _closed: bool
    _subscriptions: dict[NixlTerminalSubscriptionBinding, _OwnedSubscription]

    def __init__(self, agent: NixlTerminalAgentBoundary, capacity: int) -> None:
        """Create and own an agent-scoped native terminal channel.

        :param agent: Qualified high-level NIXL agent boundary.
        :param capacity: Positive native event capacity.
        """

        _require_positive(capacity, "channel capacity")
        self._agent = agent
        self._channel = agent.create_terminal_event_channel(capacity)
        self._closed = False
        self._subscriptions = {}
        inventory = self.query_inventory()
        if inventory.capacity != capacity:
            raise NixlTerminalLifecycleError(
                "native terminal channel changed its requested capacity"
            )

    @classmethod
    def from_nixl_agent(
        cls, agent: nixl_agent, capacity: int
    ) -> "NixlTerminalEventAdapter":
        """Bind the qualified concrete high-level NIXL agent API.

        :param agent: Concrete NIXL agent at the qualified revision.
        :param capacity: Positive native event capacity.
        :returns: Agent-scoped terminal adapter.
        """

        return cls(agent, capacity)

    @property
    def subscription_count(self) -> int:
        """Return the exact locally retained subscription count.

        :returns: Number of owned subscription handles.
        """

        return len(self._subscriptions)

    def fileno(self) -> int:
        """Return the borrowed native eventfd for selector registration.

        :returns: Borrowed nonblocking eventfd.
        :raises NixlTerminalLifecycleError: If the channel is closed.
        """

        self._require_open()
        descriptor = int(self._channel.fileno())
        if descriptor < 0:
            raise NixlTerminalLifecycleError(
                "native terminal channel returned an invalid eventfd"
            )
        return descriptor

    def query_inventory(self) -> NixlTerminalChannelInventory:
        """Return immutable inventory or propagate a sticky process fatal.

        :returns: Current immutable channel inventory.
        :raises NixlTerminalProcessFatalError: If native health is fatal.
        """

        inventory = self._project_inventory(self._channel.query_inventory())
        self._raise_if_fatal(inventory, "native terminal channel is fatal")
        return inventory

    def subscribe_transfer(
        self,
        handle: nixl_xfer_handle,
        owner_cookie: int,
    ) -> NixlTerminalSubscription:
        """Own one transfer's exact next-generation subscription.

        :param handle: Agent-owned native transfer handle.
        :param owner_cookie: Positive opaque SGLang owner identity.
        :returns: Exact-generation adapter subscription.
        """

        self._require_open()
        owner_cookie = _require_positive(owner_cookie, "owner cookie")
        native = self._agent.subscribe_xfer_terminal(
            self._channel, handle, owner_cookie
        )
        return self._retain_subscription(
            native=native,
            expected_kind=NixlTerminalEventKind.TRANSFER,
            expected_owner_cookie=owner_cookie,
            expected_identity=None,
            expected_generation=None,
        )

    def subscribe_route_capability(
        self,
        remote_agent: nixl_remote_agent_handle,
        backend: str,
        owner_cookie: int,
    ) -> NixlTerminalSubscription:
        """Own one exact remote notification-route subscription.

        :param remote_agent: Agent-owned exact-generation remote handle.
        :param backend: Instantiated native backend name.
        :param owner_cookie: Positive opaque SGLang owner identity.
        :returns: Exact-route adapter subscription.
        """

        self._require_open()
        owner_cookie = _require_positive(owner_cookie, "owner cookie")
        if type(backend) is not str or len(backend) == 0:
            raise ValueError("backend must be a non-empty string")
        remote_identity = _require_positive(
            int(remote_agent.identity), "remote handle identity"
        )
        remote_generation = _require_positive(
            int(remote_agent.generation), "remote handle generation"
        )
        native = self._agent.subscribe_remote_notification_state(
            self._channel,
            remote_agent,
            backend,
            owner_cookie,
        )
        return self._retain_subscription(
            native=native,
            expected_kind=NixlTerminalEventKind.CAPABILITY,
            expected_owner_cookie=owner_cookie,
            expected_identity=remote_identity,
            expected_generation=remote_generation,
        )

    def cancel(
        self, subscription: NixlTerminalSubscription
    ) -> NixlTerminalCancelOutcome:
        """Spend the subscription's sole explicit cancellation request.

        Transfer cancellation may remain pending until the exact terminal event
        is drained. The adapter retains the native handle in that case and
        consumes it exactly once when that event arrives.

        :param subscription: Exact adapter-owned subscription.
        :returns: Immediate release or pending-terminal outcome.
        :raises NixlTerminalLifecycleError: If cancellation was already spent.
        """

        self._require_open()
        owned = self._require_owned(subscription)
        if owned.cancel_requested:
            raise NixlTerminalLifecycleError(
                "terminal subscription cancellation was already requested"
            )
        owned.cancel_requested = True
        status = owned.native.release()
        if status == NIXL_SUCCESS:
            del self._subscriptions[subscription.binding]
            return NixlTerminalCancelOutcome.RELEASED
        if (
            status == NIXL_IN_PROG
            and subscription.binding.kind is NixlTerminalEventKind.TRANSFER
        ):
            return NixlTerminalCancelOutcome.PENDING_TERMINAL_EVENT
        raise NixlTerminalLifecycleError(
            "native terminal cancellation returned an invalid lifecycle status: "
            f"{status}"
        )

    def drain(self) -> NixlTerminalEventBatch:
        """Drain native events, validate ownership, and consume terminal handles.

        :returns: Immutable events in native publication order.
        :raises NixlTerminalProcessFatalError: If channel health or event
            ownership is invalid.
        """

        self._require_open()
        native_batch = self._channel.drain()
        inventory = self._project_inventory(native_batch.inventory)
        self._raise_if_fatal(inventory, "native terminal drain is fatal")
        try:
            events = tuple(
                project_nixl_terminal_event(event) for event in native_batch.events
            )
        except NixlTerminalLifecycleError as error:
            raise NixlTerminalProcessFatalError(
                "native terminal drain returned an invalid event",
                inventory,
            ) from error
        for event in events:
            self._accept_event(event, inventory)
        try:
            wake_count = _require_nonnegative(
                int(native_batch.wakeCount), "terminal wake count"
            )
        except NixlTerminalLifecycleError as error:
            raise NixlTerminalProcessFatalError(
                "native terminal drain returned an invalid wake count",
                inventory,
            ) from error
        return NixlTerminalEventBatch(
            events=events,
            wake_count=wake_count,
            inventory=inventory,
        )

    def close(self) -> NixlTerminalChannelInventory:
        """Close the native channel only at exact zero lifecycle inventory.

        :returns: Immutable post-close zero inventory.
        :raises NixlTerminalLifecycleError: If local or native state remains.
        :raises NixlTerminalProcessFatalError: If native closure is fatal.
        """

        self._require_open()
        if len(self._subscriptions) != 0:
            raise NixlTerminalLifecycleError(
                "terminal channel cannot close with owned subscriptions"
            )
        before = self.query_inventory()
        if (
            before.queued_channel_events != 0
            or before.active_channel_subscriptions != 0
            or before.retained_public_subscriptions != 0
            or before.backend_producers != 0
            or before.active_callback_slots != 0
            or before.queued_owner_continuations != 0
            or not before.backend_lifecycle.is_empty
        ):
            raise NixlTerminalLifecycleError(
                "terminal channel cannot close with native lifecycle inventory"
            )
        status = self._channel.close()
        if status != NIXL_SUCCESS:
            raise NixlTerminalLifecycleError(
                f"native terminal channel close returned {status}"
            )
        after = self.query_inventory()
        if not after.is_clean_closed:
            raise NixlTerminalLifecycleError(
                "native terminal channel did not close at exact zero inventory"
            )
        self._closed = True
        return after

    def _retain_subscription(
        self,
        native: nixl_terminal_event_subscription,
        expected_kind: NixlTerminalEventKind,
        expected_owner_cookie: int,
        expected_identity: int | None,
        expected_generation: int | None,
    ) -> NixlTerminalSubscription:
        """Retain and validate one native exact-generation subscription.

        :param native: Native subscription wrapper.
        :param expected_kind: Required subscription kind.
        :param expected_owner_cookie: Required owner correlation identity.
        :param expected_identity: Required native handle identity, if known.
        :param expected_generation: Required native generation, if known.
        :returns: Opaque adapter-owned subscription.
        """

        binding, active = _project_subscription(native.query())
        if not active:
            raise NixlTerminalLifecycleError(
                "new native terminal subscription is not active"
            )
        if binding.kind is not expected_kind:
            raise NixlTerminalLifecycleError(
                "native terminal subscription kind changed during creation"
            )
        if binding.owner_cookie != expected_owner_cookie:
            raise NixlTerminalLifecycleError(
                "native terminal subscription owner cookie changed during creation"
            )
        if expected_identity is not None and binding.identity != expected_identity:
            raise NixlTerminalLifecycleError(
                "native route subscription changed remote handle identity"
            )
        if (
            expected_generation is not None
            and binding.generation != expected_generation
        ):
            raise NixlTerminalLifecycleError(
                "native route subscription changed remote handle generation"
            )
        if binding in self._subscriptions:
            raise NixlTerminalLifecycleError(
                "native terminal subscription binding is not unique"
            )
        public = NixlTerminalSubscription(binding=binding)
        self._subscriptions[binding] = _OwnedSubscription(
            public=public,
            native=native,
        )
        return public

    def _accept_event(
        self,
        event: NixlTerminalEvent,
        inventory: NixlTerminalChannelInventory,
    ) -> None:
        """Validate one event and release its terminal public handle once.

        :param event: Immutable event projection.
        :param inventory: Post-drain native inventory for fatal propagation.
        :raises NixlTerminalProcessFatalError: If the event has no exact owner.
        """

        owned = self._subscriptions.get(event.binding)
        if owned is None:
            raise NixlTerminalProcessFatalError(
                "native terminal event has no exact owned subscription",
                inventory,
            )
        is_terminal = isinstance(event, NixlTransferTerminalEvent) or (
            isinstance(event, NixlCapabilityTerminalEvent) and event.is_terminal
        )
        if not is_terminal:
            return
        status = owned.native.release()
        if status != NIXL_SUCCESS:
            raise NixlTerminalProcessFatalError(
                "terminal event did not permit take-once native release",
                inventory,
            )
        del self._subscriptions[event.binding]

    def _require_owned(
        self, subscription: NixlTerminalSubscription
    ) -> _OwnedSubscription:
        """Resolve one exact public subscription to its native owner record.

        :param subscription: Candidate public subscription.
        :returns: Exact owned subscription record.
        :raises NixlTerminalLifecycleError: If ownership differs or ended.
        """

        if type(subscription) is not NixlTerminalSubscription:
            raise TypeError("subscription must be NixlTerminalSubscription")
        owned = self._subscriptions.get(subscription.binding)
        if owned is None or owned.public is not subscription:
            raise NixlTerminalLifecycleError(
                "terminal subscription is not owned by this adapter"
            )
        return owned

    def _require_open(self) -> None:
        """Reject operations after clean channel closure.

        :raises NixlTerminalLifecycleError: If the channel is closed.
        """

        if self._closed:
            raise NixlTerminalLifecycleError("terminal event adapter is already closed")

    @staticmethod
    def _project_inventory(
        value: nixl_terminal_channel_inventory,
    ) -> NixlTerminalChannelInventory:
        """Project native inventory as a process-fatal trust boundary.

        :param value: Native inventory snapshot.
        :returns: Valid immutable inventory.
        :raises NixlTerminalProcessFatalError: If native inventory violates its
            qualified schema.
        """

        try:
            return project_nixl_terminal_inventory(value)
        except NixlTerminalLifecycleError as error:
            raise NixlTerminalProcessFatalError(
                "native terminal inventory violates its qualified schema",
                None,
            ) from error

    @staticmethod
    def _raise_if_fatal(
        inventory: NixlTerminalChannelInventory,
        reason: str,
    ) -> None:
        """Propagate every sticky native fatal as process-fatal state.

        :param inventory: Current immutable native inventory.
        :param reason: Failure context.
        :raises NixlTerminalProcessFatalError: If any fatal bit or eventfd
            error is present.
        """

        if (
            inventory.fatal != NixlTerminalChannelFatal.NONE
            or inventory.eventfd_error != 0
        ):
            raise NixlTerminalProcessFatalError(reason, inventory)


class NixlDirectTerminalTransferPhase(enum.StrEnum):
    """Adapter-owned lifecycle of one direct terminal transfer."""

    ARMED = "armed"
    POSTING = "posting"
    POSTED = "posted"
    AMBIGUOUS = "ambiguous"
    SETTLED = "settled"


class NixlDirectTerminalAdapterDisposition(enum.StrEnum):
    """Process-lifetime disposition of one direct terminal adapter."""

    OPEN = "open"
    DRAINING = "draining"
    JOINED = "joined"
    CLOSED = "closed"


@dataclasses.dataclass(frozen=True, slots=True)
class NixlDirectTerminalTransfer:
    """Opaque exact-generation transfer authority retained by the adapter.

    :ivar binding_digest: Exact immutable owner lifecycle binding.
    :ivar handle_identity: Native transfer-handle identity.
    :ivar generation: Exact armed transfer generation.
    """

    binding_digest: bytes
    handle_identity: int
    generation: int

    def __post_init__(self) -> None:
        """Validate the public exact-generation identity."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        _require_positive(self.handle_identity, "direct transfer handle identity")
        _require_positive(self.generation, "direct transfer generation")


@dataclasses.dataclass(frozen=True, slots=True)
class NixlDirectTerminalProducerInventory:
    """Immutable direct native-producer lifecycle inventory.

    :ivar registering_count: Bindings whose native registration is returning.
    :ivar submitted_count: Bindings awaiting direct terminal delivery.
    :ivar active_callback_count: Terminal callbacks still outstanding.
    :ivar active_registration_count: Backend subscription calls still returning.
    :ivar total_subscriptions: Exact transfer generations armed since creation.
    :ivar total_delivered: Terminal events admitted into the immutable owner.
    :ivar successful_terminal_event_count: Successful terminal deliveries.
    :ivar failure_terminal_event_count: Failed terminal deliveries.
    :ivar owner_submission_failure_count: Rejected native owner submissions.
    :ivar admission_open: Whether new transfer generations may be armed.
    :ivar retirement_requested: Whether ordered producer retirement began.
    :ivar joined: Whether native callbacks and retirement joined.
    :ivar closed: Whether exact-zero producer closure completed.
    :ivar fatal_code: Normalized sticky native fatal code.
    :ivar fatal_status: Native errno or NIXL status for the first fatal.
    :ivar fatal_binding: Exact binding associated with the first fatal.
    """

    registering_count: int
    submitted_count: int
    active_callback_count: int
    active_registration_count: int
    total_subscriptions: int
    total_delivered: int
    successful_terminal_event_count: int
    failure_terminal_event_count: int
    owner_submission_failure_count: int
    admission_open: bool
    retirement_requested: bool
    joined: bool
    closed: bool
    fatal_code: str
    fatal_status: int
    fatal_binding: bytes | None

    @property
    def retained_count(self) -> int:
        """Return every exact native owner binding still retained.

        :returns: Registering and submitted binding count.
        """

        return self.registering_count + self.submitted_count

    @property
    def is_healthy(self) -> bool:
        """Return whether the producer carries no sticky fatal evidence.

        :returns: Whether native producer health is clean.
        """

        return (
            self.fatal_code == "none"
            and self.fatal_status == 0
            and self.fatal_binding is None
            and self.owner_submission_failure_count == 0
        )

    @property
    def has_zero_live_authority(self) -> bool:
        """Return whether callbacks and exact bindings are absent.

        :returns: Whether all live native producer counters are zero.
        """

        return (
            self.registering_count == 0
            and self.submitted_count == 0
            and self.active_callback_count == 0
            and self.active_registration_count == 0
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NixlDirectTerminalAdapterInventory:
    """Immutable local and native direct-owner inventory.

    :ivar disposition: Adapter process-lifetime disposition.
    :ivar transfer_count: Exact locally retained transfer count.
    :ivar armed_count: Transfers subscribed but not posted.
    :ivar posting_count: Transfers currently inside the post call.
    :ivar posted_count: Transfers awaiting authoritative owner terminality.
    :ivar ambiguous_count: Transfers whose submission outcome is ambiguous.
    :ivar settled_count: Terminal transfers retaining release authority.
    :ivar orphaned_subscription_count: Native wrappers retained after a failed
        subscription-schema validation.
    :ivar producer: Exact native producer inventory.
    """

    disposition: NixlDirectTerminalAdapterDisposition
    transfer_count: int
    armed_count: int
    posting_count: int
    posted_count: int
    ambiguous_count: int
    settled_count: int
    orphaned_subscription_count: int
    producer: NixlDirectTerminalProducerInventory

    @property
    def has_zero_local_authority(self) -> bool:
        """Return whether the adapter retains no transfer authority.

        :returns: Whether all local strong-retention inventories are zero.
        """

        return self.transfer_count == 0 and self.orphaned_subscription_count == 0

    @property
    def is_clean_closed(self) -> bool:
        """Return whether ordered shutdown restored exact zero authority.

        :returns: Whether both local and native close invariants hold.
        """

        return (
            self.disposition is NixlDirectTerminalAdapterDisposition.CLOSED
            and self.has_zero_local_authority
            and self.producer.has_zero_live_authority
            and self.producer.is_healthy
            and not self.producer.admission_open
            and self.producer.retirement_requested
            and self.producer.joined
            and self.producer.closed
        )


class NixlDirectTransferHandleBoundary(Protocol):
    """Transfer-handle lifetime surface owned by the direct adapter."""

    def release(self) -> None:
        """Release one terminal or provably unposted transfer handle."""

        ...


class NixlDirectTerminalAgentBoundary(Protocol):
    """Qualified direct NIXL-to-owner API consumed by the adapter."""

    def create_terminal_owner_producer(
        self,
        producer_api: object,
        producer_context: object,
    ) -> nixl_terminal_owner_producer:
        """Bind one registered native owner producer.

        :param producer_api: Versioned producer API capsule.
        :param producer_context: Registered producer context capsule.
        :returns: Process-lifetime native producer.
        """

        ...

    def subscribe_xfer_terminal_owner(
        self,
        producer: nixl_terminal_owner_producer,
        handle: nixl_xfer_handle,
        binding_digest: bytes,
    ) -> nixl_terminal_owner_subscription:
        """Arm direct owner delivery for an exact transfer generation.

        :param producer: Capsule-bound direct owner producer.
        :param handle: Agent-owned transfer handle.
        :param binding_digest: Exact 32-byte lifecycle binding.
        :returns: Retained exact-generation subscription.
        """

        ...

    def take_xfer_completion_receipt(
        self,
        handle: nixl_xfer_handle,
    ) -> nixl_xfer_completion_receipt | None:
        """Take successful completion authority exactly once.

        :param handle: Exact terminal transfer handle.
        :returns: Take-once receipt, or ``None`` while incomplete.
        """

        ...


@dataclasses.dataclass(slots=True)
class _NixlDirectOwnedTransfer:
    """Strongly retained native authority for one exact transfer generation.

    :ivar public: Opaque caller-facing identity.
    :ivar handle: Exact agent-owned transfer handle.
    :ivar subscription: Exact native callback subscription until release.
    :ivar phase: Current adapter lifecycle phase.
    :ivar settlement_action_id: Owner action first presented for settlement.
    :ivar completion_receipt: Successful take-once authority after consumption.
    """

    public: NixlDirectTerminalTransfer
    handle: NixlDirectTransferHandleBoundary
    subscription: nixl_terminal_owner_subscription | None
    phase: NixlDirectTerminalTransferPhase
    settlement_action_id: int | None = None
    completion_receipt: nixl_xfer_completion_receipt | None = None
    completion_validated: bool = False
    cancel_requested: bool = False


def _project_direct_producer_inventory(
    value: nixl_terminal_owner_producer_inventory,
) -> NixlDirectTerminalProducerInventory:
    """Project and validate one native direct-producer inventory.

    :param value: Native high-level inventory value.
    :returns: Immutable validated adapter inventory.
    """

    fatal_binding = value.fatal_binding
    if fatal_binding is not None and (
        type(fatal_binding) is not bytes or len(fatal_binding) != 32
    ):
        raise NixlTerminalLifecycleError(
            "direct producer fatal binding must contain 32 bytes"
        )
    inventory = NixlDirectTerminalProducerInventory(
        registering_count=_require_nonnegative(
            int(value.registering_count), "direct registering count"
        ),
        submitted_count=_require_nonnegative(
            int(value.submitted_count), "direct submitted count"
        ),
        active_callback_count=_require_nonnegative(
            int(value.active_callback_count), "direct active callback count"
        ),
        active_registration_count=_require_nonnegative(
            int(value.active_registration_count),
            "direct active registration count",
        ),
        total_subscriptions=_require_nonnegative(
            int(value.total_subscriptions), "direct total subscriptions"
        ),
        total_delivered=_require_nonnegative(
            int(value.total_delivered), "direct total delivered"
        ),
        successful_terminal_event_count=_require_nonnegative(
            int(value.successful_terminal_event_count),
            "direct successful terminal count",
        ),
        failure_terminal_event_count=_require_nonnegative(
            int(value.failure_terminal_event_count),
            "direct failure terminal count",
        ),
        owner_submission_failure_count=_require_nonnegative(
            int(value.owner_submission_failure_count),
            "direct owner submission failure count",
        ),
        admission_open=bool(value.admission_open),
        retirement_requested=bool(value.retirement_requested),
        joined=bool(value.joined),
        closed=bool(value.closed),
        fatal_code=str(value.fatal_code).lower(),
        fatal_status=int(value.fatal_status),
        fatal_binding=fatal_binding,
    )
    if inventory.total_delivered > inventory.total_subscriptions:
        raise NixlTerminalLifecycleError(
            "direct producer delivered more events than it subscribed"
        )
    terminal_count = (
        inventory.successful_terminal_event_count
        + inventory.failure_terminal_event_count
    )
    if terminal_count != inventory.total_delivered:
        raise NixlTerminalLifecycleError(
            "direct producer terminal counts disagree with delivered events"
        )
    return inventory


class NixlDirectTerminalOwnerAdapter:
    """Typed ownership boundary for direct NIXL terminal publication."""

    _agent: NixlDirectTerminalAgentBoundary
    _binding: NativeTerminalNativeProducerBinding
    _producer: nixl_terminal_owner_producer
    _disposition: NixlDirectTerminalAdapterDisposition
    _transfers: dict[NixlDirectTerminalTransfer, _NixlDirectOwnedTransfer]
    _orphaned_subscriptions: list[nixl_terminal_owner_subscription]
    _lock: threading.RLock

    def __init__(
        self,
        agent: NixlDirectTerminalAgentBoundary,
        binding: NativeTerminalNativeProducerBinding,
    ) -> None:
        """Bind one qualified NIXL agent to a registered native producer.

        :param agent: Qualified high-level NIXL agent.
        :param binding: Runtime-owned API and producer-context capsules.
        """

        if type(binding) is not NativeTerminalNativeProducerBinding:
            raise TypeError("binding must be NativeTerminalNativeProducerBinding")
        self._agent = agent
        self._binding = binding
        self._producer = agent.create_terminal_owner_producer(
            binding.producer_api,
            binding.producer_context,
        )
        self._disposition = NixlDirectTerminalAdapterDisposition.OPEN
        self._transfers = {}
        self._orphaned_subscriptions = []
        self._lock = threading.RLock()
        inventory = self.query_inventory().producer
        if (
            not inventory.has_zero_live_authority
            or not inventory.admission_open
            or inventory.retirement_requested
            or inventory.joined
            or inventory.closed
        ):
            raise NixlTerminalLifecycleError(
                "new direct producer did not begin at clean open inventory"
            )

    @classmethod
    def from_runtime(
        cls,
        agent: NixlDirectTerminalAgentBoundary,
        runtime: NativeTerminalRuntime,
        producer_name: str,
    ) -> "NixlDirectTerminalOwnerAdapter":
        """Construct from one runtime-registered native producer.

        :param agent: Qualified concrete high-level NIXL agent.
        :param runtime: Running immutable owner runtime.
        :param producer_name: Stable pre-registered native producer name.
        :returns: Bound direct terminal adapter.
        """

        if type(runtime) is not NativeTerminalRuntime:
            raise TypeError("runtime must be NativeTerminalRuntime")
        return cls(agent, runtime.native_producer_binding(producer_name))

    @property
    def producer_id(self) -> int:
        """Return the exact native owner producer namespace.

        :returns: Runtime-registered producer ID.
        """

        return self._binding.producer_id

    def query_inventory(self) -> NixlDirectTerminalAdapterInventory:
        """Return exact local and native direct-owner inventory.

        :returns: Immutable adapter inventory.
        :raises NixlTerminalLifecycleError: If native inventory is fatal or
            violates its qualified schema.
        """

        with self._lock:
            producer = _project_direct_producer_inventory(self._producer.inventory())
            if not producer.is_healthy:
                raise NixlTerminalLifecycleError(
                    "direct terminal producer carries process-fatal evidence"
                )
            phases = tuple(record.phase for record in self._transfers.values())
            return NixlDirectTerminalAdapterInventory(
                disposition=self._disposition,
                transfer_count=len(phases),
                armed_count=phases.count(NixlDirectTerminalTransferPhase.ARMED),
                posting_count=phases.count(NixlDirectTerminalTransferPhase.POSTING),
                posted_count=phases.count(NixlDirectTerminalTransferPhase.POSTED),
                ambiguous_count=phases.count(NixlDirectTerminalTransferPhase.AMBIGUOUS),
                settled_count=phases.count(NixlDirectTerminalTransferPhase.SETTLED),
                orphaned_subscription_count=len(self._orphaned_subscriptions),
                producer=producer,
            )

    def arm_transfer(
        self,
        handle: nixl_xfer_handle,
        binding_digest: bytes,
    ) -> NixlDirectTerminalTransfer:
        """Arm and strongly retain owner delivery before transfer posting.

        :param handle: Initialized but unposted transfer handle.
        :param binding_digest: Exact registered source lifecycle digest.
        :returns: Opaque exact-generation transfer authority.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if handle is None:
            raise ValueError("handle must not be None")
        with self._lock:
            self._require_open_locked()
            native = self._agent.subscribe_xfer_terminal_owner(
                self._producer,
                handle,
                binding_digest,
            )
            try:
                info = native.query()
                if info.kind != nixl_terminal_event_kind_t.TRANSFER:
                    raise NixlTerminalLifecycleError(
                        "direct owner subscription is not a transfer"
                    )
                if not bool(info.active):
                    raise NixlTerminalLifecycleError(
                        "new direct owner subscription is not active"
                    )
                public = NixlDirectTerminalTransfer(
                    binding_digest=binding_digest,
                    handle_identity=_require_positive(
                        int(info.identity), "direct subscription identity"
                    ),
                    generation=_require_positive(
                        int(info.generation), "direct subscription generation"
                    ),
                )
                if public in self._transfers:
                    raise NixlTerminalLifecycleError(
                        "direct transfer generation is already armed"
                    )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                self._orphaned_subscriptions.append(native)
                raise
            self._transfers[public] = _NixlDirectOwnedTransfer(
                public=public,
                handle=handle,
                subscription=native,
                phase=NixlDirectTerminalTransferPhase.ARMED,
            )
            return public

    def post_transfer(
        self,
        transfer: NixlDirectTerminalTransfer,
        post: Callable[[object], object],
    ) -> object:
        """Post only an exact transfer whose subscription is already armed.

        Terminal delivery may race ahead of the posting call's return. The
        settlement path may therefore consume a ``POSTING`` record, while this
        method never overwrites a concurrently committed terminal state.

        :param transfer: Exact adapter-owned armed transfer.
        :param post: Existing transfer-post operation.
        :returns: Existing post operation result.
        """

        if not callable(post):
            raise TypeError("post must be callable")
        with self._lock:
            self._require_open_locked()
            record = self._require_transfer_locked(transfer)
            if record.phase is not NixlDirectTerminalTransferPhase.ARMED:
                raise NixlTerminalLifecycleError(
                    "direct transfer can be posted exactly once after arming"
                )
            record.phase = NixlDirectTerminalTransferPhase.POSTING
            handle = record.handle
        post_returned = False
        try:
            result = post(handle)
            post_returned = True
            return result
        finally:
            with self._lock:
                if record.phase is NixlDirectTerminalTransferPhase.POSTING:
                    record.phase = (
                        NixlDirectTerminalTransferPhase.POSTED
                        if post_returned
                        else NixlDirectTerminalTransferPhase.AMBIGUOUS
                    )

    def settle_success(
        self,
        transfer: NixlDirectTerminalTransfer,
        action: NativeTerminalOwnerAction,
    ) -> nixl_xfer_completion_receipt:
        """Consume success only after authoritative owner terminality.

        ``SOURCE_NATIVE_TERMINAL`` is a local producer event and deliberately
        carries no wire receipt. The exact completion receipt is taken here,
        after the owner emitted the matching ``SOURCE_OUTCOME_READY`` action.

        :param transfer: Exact adapter-owned terminal transfer.
        :param action: Matching one-shot owner outcome action.
        :returns: Native take-once completion receipt.
        """

        with self._lock:
            record = self._prepare_settlement_locked(
                transfer,
                action,
                (NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,),
            )
            receipt = record.completion_receipt
            if receipt is None:
                receipt = self._agent.take_xfer_completion_receipt(record.handle)
                if receipt is None:
                    raise NixlTerminalLifecycleError(
                        "owner terminality has no NIXL completion receipt"
                    )
                record.completion_receipt = receipt
            if not record.completion_validated:
                self._validate_completion_receipt(record.public, receipt)
                record.completion_validated = True
            self._release_terminal_subscription_locked(record)
            record.phase = NixlDirectTerminalTransferPhase.SETTLED
            return receipt

    def settle_failure(
        self,
        transfer: NixlDirectTerminalTransfer,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Settle terminal failure without manufacturing success authority.

        :param transfer: Exact adapter-owned failed transfer.
        :param action: Matching quarantine or process-fatal owner action.
        """

        with self._lock:
            record = self._prepare_settlement_locked(
                transfer,
                action,
                (
                    NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
                    NativeTerminalOwnerActionKind.PROCESS_FATAL,
                ),
            )
            if record.completion_receipt is not None:
                raise NixlTerminalLifecycleError(
                    "failed transfer retained successful completion authority"
                )
            self._release_terminal_subscription_locked(record)
            record.phase = NixlDirectTerminalTransferPhase.SETTLED

    def cancel_transfer(self, transfer: NixlDirectTerminalTransfer) -> None:
        """Request cancellation while retaining every pending authority.

        A posted transfer normally returns ``NIXL_IN_PROG`` and remains owned
        until native terminal failure reaches the immutable owner. Immediate
        release is accepted only for an ``ARMED`` transfer, which proves that
        no transfer post began.

        :param transfer: Exact adapter-owned transfer generation.
        """

        with self._lock:
            record = self._require_transfer_locked(transfer)
            if record.phase not in (
                NixlDirectTerminalTransferPhase.ARMED,
                NixlDirectTerminalTransferPhase.POSTING,
                NixlDirectTerminalTransferPhase.POSTED,
                NixlDirectTerminalTransferPhase.AMBIGUOUS,
            ):
                raise NixlTerminalLifecycleError(
                    "direct transfer cancellation is illegal in its current phase"
                )
            subscription = record.subscription
            if subscription is None:
                raise NixlTerminalLifecycleError(
                    "direct transfer cancellation lost its subscription"
                )
            if record.cancel_requested:
                raise NixlTerminalLifecycleError(
                    "direct transfer cancellation was already requested"
                )
            record.cancel_requested = True
            status = subscription.release()
            if status == NIXL_IN_PROG:
                record.phase = NixlDirectTerminalTransferPhase.AMBIGUOUS
                return
            if status == NIXL_SUCCESS:
                record.subscription = None
                record.phase = (
                    NixlDirectTerminalTransferPhase.SETTLED
                    if record.phase is NixlDirectTerminalTransferPhase.ARMED
                    else NixlDirectTerminalTransferPhase.AMBIGUOUS
                )
                return
            raise NixlTerminalLifecycleError(
                "direct transfer cancellation returned an invalid lifecycle "
                f"status: {status}"
            )

    def release_transfer(self, transfer: NixlDirectTerminalTransfer) -> None:
        """Release the exact handle after terminal settlement.

        :param transfer: Settled adapter-owned transfer.
        """

        with self._lock:
            record = self._require_transfer_locked(transfer)
            if record.phase is not NixlDirectTerminalTransferPhase.SETTLED:
                raise NixlTerminalLifecycleError(
                    "direct transfer release requires terminal settlement"
                )
            if record.subscription is not None:
                raise NixlTerminalLifecycleError(
                    "direct transfer release retained its native subscription"
                )
            record.handle.release()
            del self._transfers[transfer]

    def discard_unposted(self, transfer: NixlDirectTerminalTransfer) -> None:
        """Release a provably unposted exact transfer generation.

        :param transfer: Adapter-owned transfer still in ``ARMED`` state.
        """

        with self._lock:
            record = self._require_transfer_locked(transfer)
            if record.phase is not NixlDirectTerminalTransferPhase.ARMED:
                raise NixlTerminalLifecycleError(
                    "only a provably unposted direct transfer may be discarded"
                )
            subscription = record.subscription
            if subscription is None:
                raise NixlTerminalLifecycleError(
                    "armed direct transfer lost its subscription"
                )
            status = subscription.release()
            if status != NIXL_SUCCESS:
                raise NixlTerminalLifecycleError(
                    "unposted direct subscription did not release immediately: "
                    f"{status}"
                )
            record.subscription = None
            record.phase = NixlDirectTerminalTransferPhase.SETTLED
            record.handle.release()
            del self._transfers[transfer]

    def stop_admission(self) -> None:
        """Permanently stop new exact-generation bindings."""

        with self._lock:
            self._require_disposition_locked(NixlDirectTerminalAdapterDisposition.OPEN)
            self._producer.stop_admission()
            self._disposition = NixlDirectTerminalAdapterDisposition.DRAINING

    def join(self, timeout_seconds: float) -> bool:
        """Join native callbacks and ordered producer retirement.

        :param timeout_seconds: Positive native wait bound.
        :returns: Whether producer retirement joined within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        with self._lock:
            if self._disposition is NixlDirectTerminalAdapterDisposition.JOINED:
                return True
            self._require_disposition_locked(
                NixlDirectTerminalAdapterDisposition.DRAINING
            )
        joined = self._producer.join(timeout_seconds)
        if not joined:
            return False
        with self._lock:
            self._disposition = NixlDirectTerminalAdapterDisposition.JOINED
        return True

    def close(self) -> NixlDirectTerminalAdapterInventory:
        """Close only after joined and exact-zero local/native authority.

        :returns: Exact clean post-close inventory.
        """

        with self._lock:
            self._require_disposition_locked(
                NixlDirectTerminalAdapterDisposition.JOINED
            )
            before = self.query_inventory()
            if not before.has_zero_local_authority:
                raise NixlTerminalLifecycleError(
                    "direct adapter cannot close with local transfer authority"
                )
            producer = before.producer
            if (
                not producer.has_zero_live_authority
                or producer.admission_open
                or not producer.retirement_requested
                or not producer.joined
                or producer.closed
            ):
                raise NixlTerminalLifecycleError(
                    "direct adapter cannot close with native producer authority"
                )
            self._producer.close()
            self._disposition = NixlDirectTerminalAdapterDisposition.CLOSED
            after = self.query_inventory()
            if not after.is_clean_closed:
                raise NixlTerminalLifecycleError(
                    "direct adapter did not close at exact zero inventory"
                )
            return after

    def shutdown(self, timeout_seconds: float) -> NixlDirectTerminalAdapterInventory:
        """Perform ordered stop, join, and exact-zero close.

        :param timeout_seconds: Positive native producer join bound.
        :returns: Exact clean post-close inventory.
        """

        with self._lock:
            if self._disposition is NixlDirectTerminalAdapterDisposition.OPEN:
                self.stop_admission()
        if not self.join(timeout_seconds):
            raise NixlTerminalLifecycleError(
                "direct terminal producer did not join before its deadline"
            )
        return self.close()

    def _prepare_settlement_locked(
        self,
        transfer: NixlDirectTerminalTransfer,
        action: NativeTerminalOwnerAction,
        allowed_kinds: tuple[NativeTerminalOwnerActionKind, ...],
    ) -> _NixlDirectOwnedTransfer:
        """Validate one owner action before any native authority is consumed.

        :param transfer: Exact adapter-owned terminal transfer.
        :param action: One-shot authoritative owner action.
        :param allowed_kinds: Closed action kinds accepted by the settlement.
        :returns: Exact retained transfer record.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in allowed_kinds:
            raise NixlTerminalLifecycleError(
                "owner action does not authorize this transfer settlement"
            )
        record = self._require_transfer_locked(transfer)
        if action.binding.digest != record.public.binding_digest:
            raise NixlTerminalLifecycleError(
                "owner action belongs to another transfer binding"
            )
        if record.phase not in (
            NixlDirectTerminalTransferPhase.POSTING,
            NixlDirectTerminalTransferPhase.POSTED,
            NixlDirectTerminalTransferPhase.AMBIGUOUS,
        ):
            raise NixlTerminalLifecycleError(
                "direct transfer settlement requires a posted generation"
            )
        if record.settlement_action_id is None:
            record.settlement_action_id = action.action_id
        elif record.settlement_action_id != action.action_id:
            raise NixlTerminalLifecycleError(
                "direct transfer settlement changed owner action authority"
            )
        return record

    @staticmethod
    def _validate_completion_receipt(
        transfer: NixlDirectTerminalTransfer,
        receipt: nixl_xfer_completion_receipt,
    ) -> None:
        """Validate take-once NIXL success authority against its subscription.

        :param transfer: Exact adapter-owned transfer generation.
        :param receipt: Native take-once completion receipt.
        """

        if int(receipt.handleIdentity) != transfer.handle_identity:
            raise NixlTerminalLifecycleError(
                "completion receipt changed transfer handle identity"
            )
        if int(receipt.generation) != transfer.generation:
            raise NixlTerminalLifecycleError(
                "completion receipt changed subscribed generation"
            )
        if not bool(receipt.submissionSealed) or not bool(receipt.completionClaimed):
            raise NixlTerminalLifecycleError(
                "completion receipt lacks sealed take-once authority"
            )
        if receipt.status.name != "NIXL_SUCCESS":
            raise NixlTerminalLifecycleError(
                "completion receipt does not carry successful terminal status"
            )

    @staticmethod
    def _release_terminal_subscription_locked(
        record: _NixlDirectOwnedTransfer,
    ) -> None:
        """Consume one exact terminal subscription wrapper.

        :param record: Exact terminal transfer record.
        """

        subscription = record.subscription
        if subscription is None:
            return
        info = subscription.query()
        if bool(info.active):
            raise NixlTerminalLifecycleError(
                "authoritative owner terminality left subscription active"
            )
        if (
            int(info.identity) != record.public.handle_identity
            or int(info.generation) != record.public.generation
        ):
            raise NixlTerminalLifecycleError(
                "terminal subscription changed exact transfer generation"
            )
        status = subscription.release()
        if status != NIXL_SUCCESS:
            raise NixlTerminalLifecycleError(
                f"terminal direct subscription release returned {status}"
            )
        record.subscription = None

    def _require_transfer_locked(
        self,
        transfer: NixlDirectTerminalTransfer,
    ) -> _NixlDirectOwnedTransfer:
        """Resolve one exact public token to its strongly retained record.

        :param transfer: Candidate public transfer token.
        :returns: Exact retained transfer record.
        """

        if type(transfer) is not NixlDirectTerminalTransfer:
            raise TypeError("transfer must be NixlDirectTerminalTransfer")
        record = self._transfers.get(transfer)
        if record is None or record.public is not transfer:
            raise NixlTerminalLifecycleError(
                "direct transfer is not owned by this adapter"
            )
        return record

    def _require_open_locked(self) -> None:
        """Require open transfer-generation admission."""

        self._require_disposition_locked(NixlDirectTerminalAdapterDisposition.OPEN)

    def _require_disposition_locked(
        self,
        expected: NixlDirectTerminalAdapterDisposition,
    ) -> None:
        """Require one exact process-lifetime disposition.

        :param expected: Required current adapter disposition.
        """

        if self._disposition is not expected:
            raise NixlTerminalLifecycleError(
                "direct adapter disposition is "
                f"{self._disposition.value}, expected {expected.value}"
            )
