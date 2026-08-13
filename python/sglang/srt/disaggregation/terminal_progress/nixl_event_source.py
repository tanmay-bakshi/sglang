import dataclasses
import threading

from nixl._api import nixl_remote_agent_handle, nixl_xfer_handle
from nixl._bindings import NIXL_IN_PROG, NIXL_SUCCESS
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlCapabilityTerminalEvent,
    NixlTerminalCancelOutcome,
    NixlTerminalCapabilityState,
    NixlTerminalChannelInventory,
    NixlTerminalEventAdapter,
    NixlTerminalEventKind,
    NixlTerminalLifecycleError,
    NixlTerminalProcessFatalError,
    NixlTerminalSubscription,
    NixlTerminalSubscriptionBinding,
    NixlTransferTerminalEvent,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TERMINAL_OWNER_COMMAND_TYPES,
    TerminalOwnerClosedError,
    TerminalOwnerCommandValue,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerEventSourceFatalError,
)


def _require_command(
    command: TerminalOwnerCommandValue, label: str
) -> TerminalOwnerCommandValue:
    """Validate one exact immutable owner command.

    :param command: Candidate command.
    :param label: Reader-facing route field name.
    :returns: The exact validated command.
    """

    if type(command) not in TERMINAL_OWNER_COMMAND_TYPES:
        raise TypeError(f"{label} must be an exact terminal owner command")
    return command


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTransferOwnerRoute:
    """Commands selected by one exact-generation transfer outcome.

    :ivar success: Command delivered for ``NIXL_SUCCESS``.
    :ivar failure: Command delivered for every terminal failure status.
    """

    success: TerminalOwnerCommandValue
    failure: TerminalOwnerCommandValue

    def __post_init__(self) -> None:
        """Validate both immutable route outcomes."""

        _require_command(self.success, "success")
        _require_command(self.failure, "failure")


@dataclasses.dataclass(frozen=True, slots=True)
class NixlCapabilityOwnerRoute:
    """Commands selected by one exact remote-route capability transition.

    :ivar ready: Command delivered for each native READY epoch.
    :ivar failed: Command delivered for terminal route failure.
    :ivar retired: Command delivered for terminal route retirement.
    """

    ready: TerminalOwnerCommandValue
    failed: TerminalOwnerCommandValue
    retired: TerminalOwnerCommandValue

    def __post_init__(self) -> None:
        """Validate all immutable capability outcomes."""

        _require_command(self.ready, "ready")
        _require_command(self.failed, "failed")
        _require_command(self.retired, "retired")


NixlTerminalOwnerRoute = NixlTransferOwnerRoute | NixlCapabilityOwnerRoute


@dataclasses.dataclass(frozen=True, slots=True)
class NixlTerminalOwnerEventSourceInventory:
    """Complete native and owner-route inventory for one NIXL source.

    :ivar native: Latest qualified native-channel inventory.
    :ivar registered_routes: Exact subscriptions still carrying owner routes.
    :ivar observed_unrouted: Native event identities observed without a route.
    :ivar admission_open: Whether new subscriptions may enter the source.
    :ivar closed: Whether exact-zero native closure completed.
    :ivar fatal_reason: Sticky source-fatal reason, when present.
    """

    native: NixlTerminalChannelInventory
    registered_routes: tuple[NixlTerminalSubscriptionBinding, ...]
    observed_unrouted: tuple[NixlTerminalSubscriptionBinding, ...]
    admission_open: bool
    closed: bool
    fatal_reason: str | None

    @property
    def retained_count(self) -> int:
        """Return every exact owner or orphan identity retained as evidence.

        :returns: Combined retained route and unrouted-event count.
        """

        return len(self.registered_routes) + len(self.observed_unrouted)

    @property
    def producers_joined(self) -> bool:
        """Return whether no native producer or subscription can publish.

        :returns: Exact native and owner-route quiescence state.
        """

        return (
            len(self.registered_routes) == 0
            and self.native.queued_channel_events == 0
            and self.native.active_channel_subscriptions == 0
            and self.native.retained_public_subscriptions == 0
            and self.native.backend_producers == 0
            and self.native.active_callback_slots == 0
            and self.native.queued_owner_continuations == 0
            and self.native.backend_lifecycle.is_empty
        )


class NixlTerminalOwnerEventSourceFatalError(TerminalOwnerEventSourceFatalError):
    """Process-fatal NIXL source result with complete typed inventory."""

    inventory: NixlTerminalOwnerEventSourceInventory

    def __init__(
        self,
        source_name: str,
        reason: str,
        inventory: NixlTerminalOwnerEventSourceInventory,
    ) -> None:
        """Retain native and exact routing evidence on a source fatal.

        :param source_name: Stable owner event-source name.
        :param reason: Exact failure context.
        :param inventory: Complete typed inventory at failure observation.
        """

        if type(inventory) is not NixlTerminalOwnerEventSourceInventory:
            raise TypeError("inventory must be NixlTerminalOwnerEventSourceInventory")
        self.inventory = inventory
        labels = tuple(
            _nixl_binding_label(binding)
            for binding in sorted(
                set(inventory.registered_routes) | set(inventory.observed_unrouted),
                key=_nixl_binding_sort_key,
            )
        )
        super().__init__(source_name, reason, labels)


@dataclasses.dataclass(slots=True)
class _OwnedNixlRoute:
    """Opaque native subscription and its immutable owner command route.

    :ivar subscription: Exact adapter-owned subscription object.
    :ivar route: Outcome-to-command mapping retained until terminality.
    """

    subscription: NixlTerminalSubscription
    route: NixlTerminalOwnerRoute


def _nixl_binding_sort_key(
    binding: NixlTerminalSubscriptionBinding,
) -> tuple[str, int, int, int]:
    """Return deterministic inventory order for one native binding.

    :param binding: Exact native subscription identity.
    :returns: Stable sortable fields.
    """

    return (
        binding.kind.value,
        binding.owner_cookie,
        binding.identity,
        binding.generation,
    )


def _nixl_binding_label(binding: NixlTerminalSubscriptionBinding) -> str:
    """Render one lossless native identity for process-fatal evidence.

    :param binding: Exact native subscription identity.
    :returns: Stable identity label.
    """

    return (
        f"{binding.kind.value}:cookie={binding.owner_cookie}:"
        f"identity={binding.identity}:generation={binding.generation}"
    )


class NixlTerminalOwnerEventSource(TerminalOwnerEventSource):
    """Zero-poll adapter from the qualified NIXL channel into owner commands.

    Ownership of the supplied adapter transfers to this source. Subscription,
    route insertion, native drain, and route retirement share one lock so an
    immediately published capability snapshot cannot outrun its command route.
    """

    _name: str
    _adapter: NixlTerminalEventAdapter
    _routes: dict[NixlTerminalSubscriptionBinding, _OwnedNixlRoute]
    _observed_unrouted: set[NixlTerminalSubscriptionBinding]
    _next_sequence: int
    _admission_open: bool
    _closed: bool
    _last_native_inventory: NixlTerminalChannelInventory
    _fatal_error: NixlTerminalOwnerEventSourceFatalError | None
    _lock: threading.Lock

    def __init__(self, name: str, adapter: NixlTerminalEventAdapter) -> None:
        """Take exclusive ownership of one clean qualified NIXL adapter.

        :param name: Stable source identity used in owner evidence.
        :param adapter: Clean agent-scoped terminal event adapter.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(adapter) is not NixlTerminalEventAdapter:
            raise TypeError("adapter must be NixlTerminalEventAdapter")
        if adapter.subscription_count != 0:
            raise ValueError("NIXL owner source requires a clean adapter")
        inventory = adapter.query_inventory()
        if (
            inventory.queued_channel_events != 0
            or inventory.active_channel_subscriptions != 0
            or inventory.retained_public_subscriptions != 0
        ):
            raise ValueError("NIXL owner source requires an empty native channel")
        self._name = name
        self._adapter = adapter
        self._routes = {}
        self._observed_unrouted = set()
        self._next_sequence = 0
        self._admission_open = True
        self._closed = False
        self._last_native_inventory = inventory
        self._fatal_error = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return the stable registered source identity.

        :returns: Source name supplied at construction.
        """

        return self._name

    @property
    def pending_count(self) -> int:
        """Return the exact native event count awaiting drain.

        :returns: Native queued-channel event count.
        """

        with self._lock:
            self._refresh_inventory_locked()
            return self._last_native_inventory.queued_channel_events

    def fileno(self) -> int:
        """Return the qualified native eventfd without creating a poll loop.

        :returns: Open borrowed native eventfd.
        """

        with self._lock:
            self._require_not_closed_locked()
            return self._adapter.fileno()

    def subscribe_transfer(
        self,
        handle: nixl_xfer_handle,
        owner_cookie: int,
        route: NixlTransferOwnerRoute,
    ) -> NixlTerminalSubscription:
        """Atomically subscribe and register one exact transfer route.

        :param handle: Agent-owned native transfer handle before post.
        :param owner_cookie: Positive opaque owner correlation cookie.
        :param route: Immutable success and failure commands.
        :returns: Exact-generation subscription owned by this source.
        """

        if type(route) is not NixlTransferOwnerRoute:
            raise TypeError("route must be NixlTransferOwnerRoute")
        with self._lock:
            self._require_accepting_locked()
            try:
                subscription = self._adapter.subscribe_transfer(handle, owner_cookie)
            except NixlTerminalProcessFatalError as error:
                raise self._record_native_fatal_locked(
                    "native transfer subscription failed", error
                ) from error
            self._retain_route_locked(subscription, route)
            return subscription

    def subscribe_route_capability(
        self,
        remote_agent: nixl_remote_agent_handle,
        backend: str,
        owner_cookie: int,
        route: NixlCapabilityOwnerRoute,
    ) -> NixlTerminalSubscription:
        """Atomically subscribe and register one exact capability route.

        :param remote_agent: Exact-generation remote agent handle.
        :param backend: Instantiated native backend name.
        :param owner_cookie: Positive opaque owner correlation cookie.
        :param route: Immutable READY, FAILED, and RETIRED commands.
        :returns: Exact-route subscription owned by this source.
        """

        if type(route) is not NixlCapabilityOwnerRoute:
            raise TypeError("route must be NixlCapabilityOwnerRoute")
        with self._lock:
            self._require_accepting_locked()
            try:
                subscription = self._adapter.subscribe_route_capability(
                    remote_agent,
                    backend,
                    owner_cookie,
                )
            except NixlTerminalProcessFatalError as error:
                raise self._record_native_fatal_locked(
                    "native capability subscription failed", error
                ) from error
            self._retain_route_locked(subscription, route)
            return subscription

    def cancel(
        self, subscription: NixlTerminalSubscription
    ) -> NixlTerminalCancelOutcome:
        """Cancel an exact owned subscription without dropping pending authority.

        :param subscription: Exact object returned by this source.
        :returns: Immediate release or pending-terminal outcome.
        """

        if type(subscription) is not NixlTerminalSubscription:
            raise TypeError("subscription must be NixlTerminalSubscription")
        with self._lock:
            self._require_not_closed_locked()
            owned = self._routes.get(subscription.binding)
            if owned is None or owned.subscription is not subscription:
                raise NixlTerminalLifecycleError(
                    "terminal subscription is not owned by this event source"
                )
            try:
                outcome = self._adapter.cancel(subscription)
            except NixlTerminalProcessFatalError as error:
                raise self._record_native_fatal_locked(
                    "native subscription cancellation failed", error
                ) from error
            if outcome is NixlTerminalCancelOutcome.RELEASED:
                del self._routes[subscription.binding]
            return outcome

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Translate one complete native wake into producer-ordered commands.

        :returns: Immutable owner envelopes in native publication order.
        :raises NixlTerminalOwnerEventSourceFatalError: If native health or
            exact routing is ambiguous.
        """

        with self._lock:
            self._require_not_closed_locked()
            if self._fatal_error is not None:
                raise self._fatal_error
            try:
                batch = self._adapter.drain()
            except NixlTerminalProcessFatalError as error:
                raise self._record_native_fatal_locked(
                    "native terminal drain failed", error
                ) from error
            self._last_native_inventory = batch.inventory

            commands: list[tuple[int, TerminalOwnerCommandValue]] = []
            terminal_bindings: set[NixlTerminalSubscriptionBinding] = set()
            for event in batch.events:
                owned = self._routes.get(event.binding)
                if owned is None:
                    self._observed_unrouted.add(event.binding)
                    raise self._record_routing_fatal_locked(
                        "native terminal event has no exact owner route"
                    )
                command, is_terminal = self._route_event_locked(owned.route, event)
                commands.append((event.native_timestamp_ns, command))
                if is_terminal:
                    terminal_bindings.add(event.binding)

            envelopes = tuple(
                TerminalOwnerEventEnvelope(
                    producer_sequence=self._next_sequence + index,
                    enqueued_ns=timestamp_ns,
                    command=command,
                )
                for index, (timestamp_ns, command) in enumerate(commands)
            )
            self._next_sequence += len(envelopes)
            for binding in terminal_bindings:
                del self._routes[binding]
            return envelopes

    def stop_submissions(self) -> None:
        """Permanently reject new native subscriptions at this boundary."""

        with self._lock:
            self._admission_open = False

    def join_producers(self) -> bool:
        """Attest native quiescence once without sleep-polling.

        :returns: Whether all subscriptions, callbacks, and events are absent.
        """

        with self._lock:
            self._admission_open = False
            self._refresh_inventory_locked()
            return self._inventory_locked().producers_joined

    def inventory(self) -> NixlTerminalOwnerEventSourceInventory:
        """Return complete current native and routing inventory.

        :returns: Immutable teardown and fatal evidence.
        """

        with self._lock:
            self._refresh_inventory_locked()
            return self._inventory_locked()

    def close(self) -> None:
        """Close only after admission stopped and every producer joined."""

        with self._lock:
            if self._closed:
                return
            self._admission_open = False
            self._refresh_inventory_locked()
            if self._fatal_error is not None:
                raise self._fatal_error
            inventory = self._inventory_locked()
            if not inventory.producers_joined:
                raise NixlTerminalLifecycleError(
                    "NIXL owner source cannot close with retained producers"
                )
            try:
                self._last_native_inventory = self._adapter.close()
            except NixlTerminalProcessFatalError as error:
                raise self._record_native_fatal_locked(
                    "native terminal channel close failed", error
                ) from error
            self._closed = True

    def _retain_route_locked(
        self,
        subscription: NixlTerminalSubscription,
        route: NixlTerminalOwnerRoute,
    ) -> None:
        """Retain one route before an immediate native wake may be drained.

        :param subscription: Exact native subscription object.
        :param route: Immutable outcome-to-command route.
        """

        binding = subscription.binding
        if binding in self._routes:
            raise NixlTerminalLifecycleError(
                "native subscription route identity is not unique"
            )
        if (
            binding.kind is NixlTerminalEventKind.TRANSFER
            and type(route) is not NixlTransferOwnerRoute
        ):
            raise NixlTerminalLifecycleError(
                "transfer subscription requires a transfer command route"
            )
        if (
            binding.kind is NixlTerminalEventKind.CAPABILITY
            and type(route) is not NixlCapabilityOwnerRoute
        ):
            raise NixlTerminalLifecycleError(
                "capability subscription requires a capability command route"
            )
        self._routes[binding] = _OwnedNixlRoute(
            subscription=subscription,
            route=route,
        )

    def _route_event_locked(
        self,
        route: NixlTerminalOwnerRoute,
        event: NixlTransferTerminalEvent | NixlCapabilityTerminalEvent,
    ) -> tuple[TerminalOwnerCommandValue, bool]:
        """Select one immutable command without mutating lifecycle state.

        :param route: Exact route registered before native publication.
        :param event: Qualified native terminal or capability event.
        :returns: Selected command and whether the subscription is terminal.
        """

        if isinstance(event, NixlTransferTerminalEvent):
            if type(route) is not NixlTransferOwnerRoute:
                raise self._record_routing_fatal_locked(
                    "transfer event resolved to a capability command route"
                )
            if event.status == NIXL_SUCCESS:
                return route.success, True
            if event.status == NIXL_IN_PROG:
                raise self._record_routing_fatal_locked(
                    "native terminal transfer event carried an in-progress status"
                )
            return route.failure, True

        if type(route) is not NixlCapabilityOwnerRoute:
            raise self._record_routing_fatal_locked(
                "capability event resolved to a transfer command route"
            )
        if event.state is NixlTerminalCapabilityState.READY:
            return route.ready, False
        if event.state is NixlTerminalCapabilityState.FAILED:
            return route.failed, True
        if event.state is NixlTerminalCapabilityState.RETIRED:
            return route.retired, True
        raise self._record_routing_fatal_locked(
            f"unknown capability state {event.state!s}"
        )

    def _refresh_inventory_locked(self) -> None:
        """Refresh native inventory while retaining a first sticky fatal."""

        if self._closed:
            return
        try:
            self._last_native_inventory = self._adapter.query_inventory()
        except NixlTerminalProcessFatalError as error:
            if error.inventory is not None:
                self._last_native_inventory = error.inventory
            if self._fatal_error is None:
                self._record_native_fatal_locked(
                    "native terminal inventory is fatal", error
                )

    def _record_native_fatal_locked(
        self,
        reason: str,
        error: NixlTerminalProcessFatalError,
    ) -> NixlTerminalOwnerEventSourceFatalError:
        """Store the first native fatal with its exact current inventory.

        :param reason: Adapter operation which observed the failure.
        :param error: Qualified native fatal projection.
        :returns: Sticky source-fatal object.
        """

        if error.inventory is not None:
            self._last_native_inventory = error.inventory
        if self._fatal_error is None:
            self._admission_open = False
            self._fatal_error = NixlTerminalOwnerEventSourceFatalError(
                source_name=self._name,
                reason=f"{reason}: {error.reason}",
                inventory=self._inventory_locked(
                    fatal_reason=f"{reason}: {error.reason}"
                ),
            )
        return self._fatal_error

    def _record_routing_fatal_locked(
        self, reason: str
    ) -> NixlTerminalOwnerEventSourceFatalError:
        """Store one source-level exact-routing fatal.

        :param reason: Precise routing invariant failure.
        :returns: Sticky source-fatal object.
        """

        if self._fatal_error is None:
            self._admission_open = False
            self._fatal_error = NixlTerminalOwnerEventSourceFatalError(
                source_name=self._name,
                reason=reason,
                inventory=self._inventory_locked(fatal_reason=reason),
            )
        return self._fatal_error

    def _inventory_locked(
        self, fatal_reason: str | None = None
    ) -> NixlTerminalOwnerEventSourceInventory:
        """Project complete immutable source inventory while holding the lock.

        :param fatal_reason: Prospective first fatal reason during construction.
        :returns: Complete typed inventory.
        """

        reason = fatal_reason
        if reason is None and self._fatal_error is not None:
            reason = self._fatal_error.reason
        return NixlTerminalOwnerEventSourceInventory(
            native=self._last_native_inventory,
            registered_routes=tuple(sorted(self._routes, key=_nixl_binding_sort_key)),
            observed_unrouted=tuple(
                sorted(self._observed_unrouted, key=_nixl_binding_sort_key)
            ),
            admission_open=self._admission_open,
            closed=self._closed,
            fatal_reason=reason,
        )

    def _require_accepting_locked(self) -> None:
        """Require healthy open subscription admission."""

        self._require_not_closed_locked()
        if self._fatal_error is not None:
            raise self._fatal_error
        if not self._admission_open:
            raise TerminalOwnerClosedError(
                "NIXL terminal owner source admission is closed"
            )

    def _require_not_closed_locked(self) -> None:
        """Reject use after exact-zero native closure."""

        if self._closed:
            raise TerminalOwnerClosedError("NIXL terminal owner source is closed")
