import dataclasses
from collections.abc import Sequence

import pytest
from nixl._api import (
    nixl_terminal_capability_state_t,
    nixl_terminal_destination_phase_t,
    nixl_terminal_event_kind_t,
)
from nixl._bindings import (
    NIXL_ERR_CANCELED,
    NIXL_IN_PROG,
    NIXL_SUCCESS,
    nixl_status_t,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlCapabilityTerminalEvent,
    NixlTerminalCancelOutcome,
    NixlTerminalCapabilityState,
    NixlTerminalChannelFatal,
    NixlTerminalDestinationPhase,
    NixlTerminalEventAdapter,
    NixlTerminalEventKind,
    NixlTerminalLifecycleError,
    NixlTerminalProcessFatalError,
    NixlTransferTerminalEvent,
    project_nixl_terminal_event,
    project_nixl_terminal_inventory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeSourceDelivery:
    """Native-shaped source delivery fixture."""

    backend: str
    deliveryIdentity: int
    sourceHandleIdentity: int
    sourceGeneration: int
    localPending: bool
    receiptPending: bool
    deadlineActive: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeDestinationDelivery:
    """Native-shaped destination delivery fixture."""

    backend: str
    sourceBackendIncarnation: str
    sourceHandleIdentity: int
    sourceGeneration: int
    deliveryIdentity: int
    phase: nixl_terminal_destination_phase_t


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeDeadline:
    """Native-shaped deadline fixture."""

    backend: str
    handleIdentity: int
    generation: int


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeBackendInventory:
    """Native-shaped backend lifecycle inventory fixture."""

    sourceDeliveriesOutstanding: int = 0
    sourceLocalPending: int = 0
    sourceReceiptPending: int = 0
    destinationPending: int = 0
    destinationAdmitting: int = 0
    destinationCommitted: int = 0
    destinationReplaying: int = 0
    destinationQuarantined: int = 0
    activeNativeDeadlines: int = 0
    sourceDeliveries: tuple[_NativeSourceDelivery, ...] = ()
    destinationDeliveries: tuple[_NativeDestinationDelivery, ...] = ()
    nativeDeadlines: tuple[_NativeDeadline, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeFatalValue:
    """Integer-valued native enum fixture, including combined flags."""

    value: int


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeChannelInventory:
    """Native-shaped channel inventory fixture."""

    capacity: int
    queuedChannelEvents: int = 0
    activeChannelSubscriptions: int = 0
    retainedPublicSubscriptions: int = 0
    backendProducers: int = 0
    activeCallbackSlots: int = 0
    queuedOwnerContinuations: int = 0
    backendLifecycle: _NativeBackendInventory = _NativeBackendInventory()
    acceptingSubscriptions: bool = True
    closed: bool = False
    fatal: object = _NativeFatalValue(0)
    eventfdError: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeSubscriptionInfo:
    """Native-shaped exact subscription fixture."""

    kind: nixl_terminal_event_kind_t
    ownerCookie: int
    identity: int
    generation: int
    active: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeEvent:
    """Native-shaped immutable terminal event fixture."""

    kind: nixl_terminal_event_kind_t
    ownerCookie: int
    identity: int
    generation: int
    nativeTimestampNs: int
    transferStatus: nixl_status_t | None = None
    capabilityState: nixl_terminal_capability_state_t | None = None
    capabilityEpoch: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeBatch:
    """Native-shaped channel drain fixture."""

    events: tuple[_NativeEvent, ...]
    wakeCount: int
    inventory: _NativeChannelInventory


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteHandle:
    """Remote handle identity fixture."""

    identity: int
    generation: int


class _FakeNativeSubscription:
    """Deterministic native subscription lifecycle."""

    _info: _NativeSubscriptionInfo
    _release_statuses: list[nixl_status_t]
    release_calls: int

    def __init__(
        self,
        info: _NativeSubscriptionInfo,
        release_statuses: Sequence[nixl_status_t],
    ) -> None:
        """Create a native-shaped subscription.

        :param info: Exact subscription binding.
        :param release_statuses: Prospective one-shot release results.
        """

        self._info = info
        self._release_statuses = list(release_statuses)
        self.release_calls = 0

    def query(self) -> _NativeSubscriptionInfo:
        """Return the exact native subscription binding.

        :returns: Immutable binding fixture.
        """

        return self._info

    def release(self) -> nixl_status_t:
        """Consume the next prospective release result.

        :returns: Native release status.
        """

        if len(self._release_statuses) == 0:
            raise AssertionError("native release called more than planned")
        self.release_calls += 1
        return self._release_statuses.pop(0)


class _FakeChannel:
    """Deterministic native event channel boundary."""

    batches: list[_NativeBatch]
    close_calls: int
    close_status: nixl_status_t
    inventory: _NativeChannelInventory

    def __init__(self, capacity: int) -> None:
        """Create an empty open channel.

        :param capacity: Channel capacity.
        """

        self.inventory = _NativeChannelInventory(capacity=capacity)
        self.batches = []
        self.close_status = NIXL_SUCCESS
        self.close_calls = 0

    def fileno(self) -> int:
        """Return a stable borrowed descriptor fixture.

        :returns: Borrowed descriptor number.
        """

        return 37

    def query_inventory(self) -> _NativeChannelInventory:
        """Return current immutable native inventory.

        :returns: Current inventory fixture.
        """

        return self.inventory

    def drain(self) -> _NativeBatch:
        """Return the next planned native batch.

        :returns: Planned batch, or an empty non-waking batch.
        """

        if len(self.batches) == 0:
            return _NativeBatch(events=(), wakeCount=0, inventory=self.inventory)
        return self.batches.pop(0)

    def close(self) -> nixl_status_t:
        """Close and publish exact clean inventory on native success.

        :returns: Prospective native close status.
        """

        self.close_calls += 1
        if self.close_status != NIXL_SUCCESS:
            return self.close_status
        self.inventory = _NativeChannelInventory(
            capacity=self.inventory.capacity,
            acceptingSubscriptions=False,
            closed=True,
        )
        return NIXL_SUCCESS


class _FakeAgent:
    """Strict high-level NIXL surface consumed by the adapter."""

    channel: _FakeChannel
    subscriptions: list[_FakeNativeSubscription]

    def __init__(
        self,
        capacity: int,
        subscriptions: Sequence[_FakeNativeSubscription] = (),
    ) -> None:
        """Create one agent-scoped channel and subscription plan.

        :param capacity: Channel capacity.
        :param subscriptions: Prospective subscriptions in creation order.
        """

        self.channel = _FakeChannel(capacity)
        self.subscriptions = list(subscriptions)

    def create_terminal_event_channel(self, capacity: int) -> _FakeChannel:
        """Return the sole channel after checking its capacity.

        :param capacity: Requested channel capacity.
        :returns: Agent-scoped channel.
        """

        if capacity != self.channel.inventory.capacity:
            raise AssertionError("unexpected channel capacity")
        return self.channel

    def subscribe_xfer_terminal(
        self,
        channel: _FakeChannel,
        handle: object,
        owner_cookie: int,
    ) -> _FakeNativeSubscription:
        """Return the next planned transfer subscription.

        :param channel: Agent-scoped channel.
        :param handle: Opaque transfer handle fixture.
        :param owner_cookie: Exact owner correlation identity.
        :returns: Planned native subscription.
        """

        if channel is not self.channel or handle is None or owner_cookie <= 0:
            raise AssertionError("invalid transfer subscription arguments")
        return self._take_subscription()

    def subscribe_remote_notification_state(
        self,
        channel: _FakeChannel,
        remote_agent: _RemoteHandle,
        backend: str,
        owner_cookie: int,
    ) -> _FakeNativeSubscription:
        """Return the next planned route subscription.

        :param channel: Agent-scoped channel.
        :param remote_agent: Exact remote handle fixture.
        :param backend: Native backend name.
        :param owner_cookie: Exact owner correlation identity.
        :returns: Planned native subscription.
        """

        if (
            channel is not self.channel
            or remote_agent.identity <= 0
            or len(backend) == 0
            or owner_cookie <= 0
        ):
            raise AssertionError("invalid capability subscription arguments")
        return self._take_subscription()

    def _take_subscription(self) -> _FakeNativeSubscription:
        """Consume one planned subscription.

        :returns: Next native subscription.
        """

        if len(self.subscriptions) == 0:
            raise AssertionError("adapter requested an unplanned subscription")
        return self.subscriptions.pop(0)


def _transfer_subscription(
    *,
    owner_cookie: int = 7,
    identity: int = 11,
    generation: int = 3,
    release_statuses: Sequence[nixl_status_t] = (NIXL_SUCCESS,),
) -> _FakeNativeSubscription:
    """Create one exact transfer subscription fixture.

    :param owner_cookie: Owner correlation identity.
    :param identity: Transfer-handle identity.
    :param generation: Transfer generation.
    :param release_statuses: Prospective release outcomes.
    :returns: Native-shaped subscription.
    """

    return _FakeNativeSubscription(
        _NativeSubscriptionInfo(
            kind=nixl_terminal_event_kind_t.TRANSFER,
            ownerCookie=owner_cookie,
            identity=identity,
            generation=generation,
        ),
        release_statuses,
    )


def _capability_subscription(
    *,
    owner_cookie: int = 13,
    identity: int = 17,
    generation: int = 5,
    release_statuses: Sequence[nixl_status_t] = (NIXL_SUCCESS,),
) -> _FakeNativeSubscription:
    """Create one exact route-capability subscription fixture.

    :param owner_cookie: Owner correlation identity.
    :param identity: Remote-handle identity.
    :param generation: Remote-handle generation.
    :param release_statuses: Prospective release outcomes.
    :returns: Native-shaped subscription.
    """

    return _FakeNativeSubscription(
        _NativeSubscriptionInfo(
            kind=nixl_terminal_event_kind_t.CAPABILITY,
            ownerCookie=owner_cookie,
            identity=identity,
            generation=generation,
        ),
        release_statuses,
    )


def _transfer_event(
    *,
    owner_cookie: int = 7,
    identity: int = 11,
    generation: int = 3,
    status: nixl_status_t = NIXL_SUCCESS,
    timestamp_ns: int = 101,
) -> _NativeEvent:
    """Create one native transfer event fixture.

    :param owner_cookie: Owner correlation identity.
    :param identity: Transfer-handle identity.
    :param generation: Transfer generation.
    :param status: Terminal transfer status.
    :param timestamp_ns: Native publication timestamp.
    :returns: Native-shaped event.
    """

    return _NativeEvent(
        kind=nixl_terminal_event_kind_t.TRANSFER,
        ownerCookie=owner_cookie,
        identity=identity,
        generation=generation,
        transferStatus=status,
        nativeTimestampNs=timestamp_ns,
    )


def _capability_event(
    state: nixl_terminal_capability_state_t,
    *,
    owner_cookie: int = 13,
    identity: int = 17,
    generation: int = 5,
    epoch: int = 2,
    timestamp_ns: int = 103,
) -> _NativeEvent:
    """Create one native route-capability event fixture.

    :param state: Native route state.
    :param owner_cookie: Owner correlation identity.
    :param identity: Remote-handle identity.
    :param generation: Remote-handle generation.
    :param epoch: Capability-state epoch.
    :param timestamp_ns: Native publication timestamp.
    :returns: Native-shaped event.
    """

    return _NativeEvent(
        kind=nixl_terminal_event_kind_t.CAPABILITY,
        ownerCookie=owner_cookie,
        identity=identity,
        generation=generation,
        capabilityState=state,
        capabilityEpoch=epoch,
        nativeTimestampNs=timestamp_ns,
    )


def _batch(
    channel: _FakeChannel,
    *events: _NativeEvent,
    inventory: _NativeChannelInventory | None = None,
) -> _NativeBatch:
    """Create one native drain batch.

    :param channel: Owning channel fixture.
    :param events: Events in native publication order.
    :param inventory: Optional post-drain inventory override.
    :returns: Native-shaped batch.
    """

    projected_inventory = channel.inventory if inventory is None else inventory
    return _NativeBatch(
        events=tuple(events),
        wakeCount=1 if len(events) > 0 else 0,
        inventory=projected_inventory,
    )


def test_projects_exact_immutable_event_and_lifecycle_inventory() -> None:
    source = _NativeSourceDelivery(
        backend="UCX",
        deliveryIdentity=19,
        sourceHandleIdentity=23,
        sourceGeneration=7,
        localPending=True,
        receiptPending=False,
        deadlineActive=True,
    )
    destination = _NativeDestinationDelivery(
        backend="UCX",
        sourceBackendIncarnation="source-a:4",
        sourceHandleIdentity=23,
        sourceGeneration=7,
        deliveryIdentity=29,
        phase=nixl_terminal_destination_phase_t.QUARANTINED,
    )
    deadline = _NativeDeadline(backend="UCX", handleIdentity=23, generation=7)
    native = _NativeChannelInventory(
        capacity=64,
        activeChannelSubscriptions=2,
        retainedPublicSubscriptions=2,
        backendProducers=1,
        backendLifecycle=_NativeBackendInventory(
            sourceDeliveriesOutstanding=1,
            sourceLocalPending=1,
            destinationQuarantined=1,
            activeNativeDeadlines=1,
            sourceDeliveries=(source,),
            destinationDeliveries=(destination,),
            nativeDeadlines=(deadline,),
        ),
        fatal=_NativeFatalValue(
            NixlTerminalChannelFatal.QUEUE_OVERFLOW.value
            | NixlTerminalChannelFatal.EVENTFD_FAILURE.value
        ),
        eventfdError=5,
    )

    inventory = project_nixl_terminal_inventory(native)
    event = project_nixl_terminal_event(_transfer_event(status=NIXL_ERR_CANCELED))

    assert inventory.capacity == 64
    assert inventory.backend_lifecycle.source_deliveries[0].source_generation == 7
    assert (
        inventory.backend_lifecycle.destination_deliveries[0].phase
        is NixlTerminalDestinationPhase.QUARANTINED
    )
    assert inventory.backend_lifecycle.native_deadlines[0].handle_identity == 23
    assert inventory.fatal == (
        NixlTerminalChannelFatal.QUEUE_OVERFLOW
        | NixlTerminalChannelFatal.EVENTFD_FAILURE
    )
    assert isinstance(event, NixlTransferTerminalEvent)
    assert event.status == NIXL_ERR_CANCELED
    assert event.binding.kind is NixlTerminalEventKind.TRANSFER
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.generation = 8


def test_rejects_internally_inconsistent_native_inventory() -> None:
    native = _NativeChannelInventory(
        capacity=8,
        backendLifecycle=_NativeBackendInventory(
            sourceDeliveriesOutstanding=1,
        ),
    )

    with pytest.raises(
        NixlTerminalLifecycleError,
        match="source delivery count",
    ):
        project_nixl_terminal_inventory(native)


def test_agent_scoped_channel_exposes_fd_drains_and_closes_cleanly() -> None:
    agent = _FakeAgent(capacity=8)
    adapter = NixlTerminalEventAdapter(agent, capacity=8)

    assert adapter.fileno() == 37
    assert adapter.subscription_count == 0
    assert adapter.drain().events == ()
    closed = adapter.close()

    assert closed.is_clean_closed
    assert agent.channel.close_calls == 1
    with pytest.raises(NixlTerminalLifecycleError, match="already closed"):
        adapter.fileno()


def test_transfer_subscription_releases_only_its_exact_terminal_generation() -> None:
    native_subscription = _transfer_subscription()
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    subscription = adapter.subscribe_transfer(object(), owner_cookie=7)
    agent.channel.batches.append(_batch(agent.channel, _transfer_event()))

    result = adapter.drain()

    assert result.events == (
        NixlTransferTerminalEvent(
            owner_cookie=7,
            identity=11,
            generation=3,
            status=NIXL_SUCCESS,
            native_timestamp_ns=101,
        ),
    )
    assert subscription.binding.generation == 3
    assert native_subscription.release_calls == 1
    assert adapter.subscription_count == 0
    with pytest.raises(NixlTerminalLifecycleError, match="not owned"):
        adapter.cancel(subscription)


@pytest.mark.parametrize(
    "terminal_state",
    (
        nixl_terminal_capability_state_t.FAILED,
        nixl_terminal_capability_state_t.RETIRED,
    ),
)
def test_route_subscription_retains_ready_and_releases_terminal_state(
    terminal_state: nixl_terminal_capability_state_t,
) -> None:
    native_subscription = _capability_subscription()
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    subscription = adapter.subscribe_route_capability(
        _RemoteHandle(identity=17, generation=5),
        backend="UCX",
        owner_cookie=13,
    )
    agent.channel.batches.extend(
        (
            _batch(
                agent.channel,
                _capability_event(nixl_terminal_capability_state_t.READY),
            ),
            _batch(agent.channel, _capability_event(terminal_state, epoch=3)),
        )
    )

    ready = adapter.drain().events[0]
    assert isinstance(ready, NixlCapabilityTerminalEvent)
    assert ready.state is NixlTerminalCapabilityState.READY
    assert native_subscription.release_calls == 0
    assert adapter.subscription_count == 1

    terminal = adapter.drain().events[0]
    assert isinstance(terminal, NixlCapabilityTerminalEvent)
    assert terminal.state.value == terminal_state.name.lower()
    assert native_subscription.release_calls == 1
    assert adapter.subscription_count == 0
    with pytest.raises(NixlTerminalLifecycleError, match="not owned"):
        adapter.cancel(subscription)


def test_route_subscription_requires_exact_remote_identity_and_generation() -> None:
    native_subscription = _capability_subscription(identity=18)
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)

    with pytest.raises(NixlTerminalLifecycleError, match="identity"):
        adapter.subscribe_route_capability(
            _RemoteHandle(identity=17, generation=5),
            backend="UCX",
            owner_cookie=13,
        )


def test_pending_transfer_cancellation_is_one_shot_until_terminal_event() -> None:
    native_subscription = _transfer_subscription(
        release_statuses=(NIXL_IN_PROG, NIXL_SUCCESS)
    )
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    subscription = adapter.subscribe_transfer(object(), owner_cookie=7)

    assert (
        adapter.cancel(subscription) is NixlTerminalCancelOutcome.PENDING_TERMINAL_EVENT
    )
    assert adapter.subscription_count == 1
    with pytest.raises(NixlTerminalLifecycleError, match="already requested"):
        adapter.cancel(subscription)

    agent.channel.batches.append(
        _batch(
            agent.channel,
            _transfer_event(status=NIXL_ERR_CANCELED),
        )
    )
    adapter.drain()
    assert native_subscription.release_calls == 2
    assert adapter.subscription_count == 0


def test_immediate_cancellation_rejects_replay() -> None:
    native_subscription = _capability_subscription()
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    subscription = adapter.subscribe_route_capability(
        _RemoteHandle(identity=17, generation=5),
        backend="UCX",
        owner_cookie=13,
    )

    assert adapter.cancel(subscription) is NixlTerminalCancelOutcome.RELEASED
    with pytest.raises(NixlTerminalLifecycleError, match="not owned"):
        adapter.cancel(subscription)
    assert native_subscription.release_calls == 1


def test_unowned_or_malformed_native_event_is_process_fatal() -> None:
    agent = _FakeAgent(capacity=8)
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    agent.channel.batches.append(_batch(agent.channel, _transfer_event()))

    with pytest.raises(
        NixlTerminalProcessFatalError,
        match="no exact owned subscription",
    ) as unowned:
        adapter.drain()
    assert unowned.value.inventory is not None

    agent.channel.batches.append(_batch(agent.channel, _transfer_event(owner_cookie=0)))
    with pytest.raises(
        NixlTerminalProcessFatalError,
        match="invalid event",
    ) as malformed:
        adapter.drain()
    assert isinstance(malformed.value.__cause__, NixlTerminalLifecycleError)


@pytest.mark.parametrize(
    ("fatal", "eventfd_error"),
    (
        (NixlTerminalChannelFatal.QUEUE_OVERFLOW.value, 0),
        (NixlTerminalChannelFatal.EVENTFD_FAILURE.value, 5),
        (NixlTerminalChannelFatal.INVALID_PUBLICATION.value, 0),
    ),
)
def test_channel_fatal_and_eventfd_error_propagate_as_process_fatal(
    fatal: int,
    eventfd_error: int,
) -> None:
    agent = _FakeAgent(capacity=8)
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    agent.channel.inventory = dataclasses.replace(
        agent.channel.inventory,
        fatal=_NativeFatalValue(fatal),
        eventfdError=eventfd_error,
    )

    with pytest.raises(NixlTerminalProcessFatalError) as raised:
        adapter.query_inventory()
    assert raised.value.inventory is not None
    assert raised.value.inventory.fatal.value == fatal
    assert raised.value.inventory.eventfd_error == eventfd_error


def test_close_fails_closed_for_local_and_native_lifecycle_inventory() -> None:
    native_subscription = _capability_subscription()
    agent = _FakeAgent(capacity=8, subscriptions=(native_subscription,))
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    subscription = adapter.subscribe_route_capability(
        _RemoteHandle(identity=17, generation=5),
        backend="UCX",
        owner_cookie=13,
    )

    with pytest.raises(NixlTerminalLifecycleError, match="owned subscriptions"):
        adapter.close()
    assert agent.channel.close_calls == 0

    adapter.cancel(subscription)
    agent.channel.inventory = dataclasses.replace(
        agent.channel.inventory,
        backendProducers=1,
    )
    with pytest.raises(NixlTerminalLifecycleError, match="native lifecycle"):
        adapter.close()
    assert agent.channel.close_calls == 0

    agent.channel.inventory = _NativeChannelInventory(capacity=8)
    assert adapter.close().is_clean_closed


def test_invalid_native_inventory_is_process_fatal_at_adapter_boundary() -> None:
    agent = _FakeAgent(capacity=8)
    adapter = NixlTerminalEventAdapter(agent, capacity=8)
    agent.channel.inventory = dataclasses.replace(
        agent.channel.inventory,
        capacity=0,
    )

    with pytest.raises(
        NixlTerminalProcessFatalError,
        match="violates its qualified schema",
    ) as raised:
        adapter.query_inventory()
    assert raised.value.inventory is None
    assert isinstance(raised.value.__cause__, NixlTerminalLifecycleError)
