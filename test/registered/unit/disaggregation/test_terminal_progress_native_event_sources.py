import dataclasses
from collections.abc import Sequence

import pytest
from nixl._api import (
    nixl_terminal_capability_state_t,
    nixl_terminal_event_kind_t,
)
from nixl._bindings import (
    NIXL_ERR_CANCELED,
    NIXL_IN_PROG,
    NIXL_SUCCESS,
    nixl_status_t,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.cuda_bridge import (
    CudaCompletionBridge,
    CudaCompletionFatalCode,
    CudaCompletionIdentity,
)
from sglang.srt.disaggregation.terminal_progress.cuda_event_source import (
    CudaTerminalOwnerEventSource,
    CudaTerminalOwnerEventSourceFatalError,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlTerminalChannelFatal,
    NixlTerminalEventAdapter,
)
from sglang.srt.disaggregation.terminal_progress.nixl_event_source import (
    NixlCapabilityOwnerRoute,
    NixlTerminalOwnerEventSource,
    NixlTerminalOwnerEventSourceFatalError,
    NixlTransferOwnerRoute,
)
from sglang.srt.disaggregation.terminal_progress.owner import (
    PackedTerminalProgressOwner,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    ApplyDecodeLifecycleEvent,
    InjectTerminalOwnerFailure,
    TerminalOwnerDisposition,
    TerminalOwnerEventSourceRegistration,
    TerminalOwnerFatalCause,
    TerminalOwnerPulse,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeBackendInventory:
    """Empty native backend-lifecycle inventory fixture."""

    sourceDeliveriesOutstanding: int = 0
    sourceLocalPending: int = 0
    sourceReceiptPending: int = 0
    destinationPending: int = 0
    destinationAdmitting: int = 0
    destinationCommitted: int = 0
    destinationReplaying: int = 0
    destinationQuarantined: int = 0
    activeNativeDeadlines: int = 0
    sourceDeliveries: tuple[object, ...] = ()
    destinationDeliveries: tuple[object, ...] = ()
    nativeDeadlines: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeFatalValue:
    """Integer-valued native fatal enum fixture."""

    value: int


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeChannelInventory:
    """Native-shaped terminal channel inventory fixture."""

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
    """Native-shaped terminal channel drain fixture."""

    events: tuple[_NativeEvent, ...]
    wakeCount: int
    inventory: _NativeChannelInventory


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteHandle:
    """Remote-agent identity fixture."""

    identity: int
    generation: int


class _NativeSubscription:
    """Deterministic native subscription lifecycle fixture."""

    _info: _NativeSubscriptionInfo
    _release_statuses: list[nixl_status_t]
    release_calls: int

    def __init__(
        self,
        info: _NativeSubscriptionInfo,
        release_statuses: Sequence[nixl_status_t] = (NIXL_SUCCESS,),
    ) -> None:
        """Create one prospective native subscription.

        :param info: Exact subscription binding.
        :param release_statuses: Planned release outcomes.
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
        """Consume one planned release result.

        :returns: Next native lifecycle status.
        """

        if len(self._release_statuses) == 0:
            raise AssertionError("native release called more than planned")
        self.release_calls += 1
        return self._release_statuses.pop(0)


class _NativeChannel:
    """Deterministic native terminal channel fixture."""

    inventory: _NativeChannelInventory
    batches: list[_NativeBatch]

    def __init__(self, capacity: int) -> None:
        """Create one empty open channel.

        :param capacity: Frozen native capacity.
        """

        self.inventory = _NativeChannelInventory(capacity=capacity)
        self.batches = []

    def fileno(self) -> int:
        """Return a stable borrowed descriptor fixture.

        :returns: Non-negative descriptor identity.
        """

        return 37

    def query_inventory(self) -> _NativeChannelInventory:
        """Return current native inventory.

        :returns: Immutable inventory fixture.
        """

        return self.inventory

    def drain(self) -> _NativeBatch:
        """Consume the next planned native drain.

        :returns: Next native batch.
        """

        if len(self.batches) == 0:
            return _NativeBatch(events=(), wakeCount=0, inventory=self.inventory)
        batch = self.batches.pop(0)
        self.inventory = batch.inventory
        return batch

    def close(self) -> nixl_status_t:
        """Publish exact clean closed inventory.

        :returns: Native success.
        """

        self.inventory = _NativeChannelInventory(
            capacity=self.inventory.capacity,
            acceptingSubscriptions=False,
            closed=True,
        )
        return NIXL_SUCCESS


class _NativeAgent:
    """Strict fake agent consumed through the real typed adapter."""

    channel: _NativeChannel
    subscriptions: list[_NativeSubscription]

    def __init__(
        self,
        capacity: int,
        subscriptions: Sequence[_NativeSubscription],
    ) -> None:
        """Create one planned agent fixture.

        :param capacity: Frozen native channel capacity.
        :param subscriptions: Prospective subscription objects.
        """

        self.channel = _NativeChannel(capacity)
        self.subscriptions = list(subscriptions)

    def create_terminal_event_channel(self, capacity: int) -> _NativeChannel:
        """Return the sole planned channel.

        :param capacity: Requested channel capacity.
        :returns: Agent-owned channel fixture.
        """

        if capacity != self.channel.inventory.capacity:
            raise AssertionError("unexpected native channel capacity")
        return self.channel

    def subscribe_xfer_terminal(
        self,
        channel: _NativeChannel,
        handle: object,
        owner_cookie: int,
    ) -> _NativeSubscription:
        """Return the next planned transfer subscription.

        :param channel: Agent-owned channel.
        :param handle: Opaque transfer handle.
        :param owner_cookie: Positive owner route cookie.
        :returns: Planned subscription.
        """

        if channel is not self.channel or handle is None or owner_cookie <= 0:
            raise AssertionError("invalid transfer subscription")
        return self._take_subscription()

    def subscribe_remote_notification_state(
        self,
        channel: _NativeChannel,
        remote_agent: _RemoteHandle,
        backend: str,
        owner_cookie: int,
    ) -> _NativeSubscription:
        """Return the next planned capability subscription.

        :param channel: Agent-owned channel.
        :param remote_agent: Exact remote identity.
        :param backend: Native backend name.
        :param owner_cookie: Positive owner route cookie.
        :returns: Planned subscription.
        """

        if (
            channel is not self.channel
            or remote_agent.identity <= 0
            or len(backend) == 0
            or owner_cookie <= 0
        ):
            raise AssertionError("invalid capability subscription")
        return self._take_subscription()

    def _take_subscription(self) -> _NativeSubscription:
        """Consume one planned subscription.

        :returns: Next native subscription.
        """

        if len(self.subscriptions) == 0:
            raise AssertionError("unplanned native subscription")
        return self.subscriptions.pop(0)


def _subscription(
    kind: nixl_terminal_event_kind_t,
    cookie: int,
    identity: int,
    generation: int,
) -> _NativeSubscription:
    """Create one deterministic native subscription.

    :param kind: Transfer or capability kind.
    :param cookie: Owner correlation cookie.
    :param identity: Native handle identity.
    :param generation: Native handle generation.
    :returns: Planned native subscription.
    """

    return _NativeSubscription(
        _NativeSubscriptionInfo(
            kind=kind,
            ownerCookie=cookie,
            identity=identity,
            generation=generation,
        )
    )


def _nixl_source(
    subscriptions: Sequence[_NativeSubscription],
) -> tuple[NixlTerminalOwnerEventSource, _NativeAgent, NixlTerminalEventAdapter]:
    """Create a source around the real typed NIXL adapter.

    :param subscriptions: Prospective native subscriptions.
    :returns: Owner source, native fixture, and underlying typed adapter.
    """

    agent = _NativeAgent(capacity=16, subscriptions=subscriptions)
    adapter = NixlTerminalEventAdapter(agent, capacity=16)
    source = NixlTerminalOwnerEventSource("nixl-owner-events", adapter)
    return source, agent, adapter


def _queue_nixl_batch(
    agent: _NativeAgent,
    events: tuple[_NativeEvent, ...],
    *,
    active_subscriptions: int,
    fatal: NixlTerminalChannelFatal = NixlTerminalChannelFatal.NONE,
) -> None:
    """Queue one native batch and publish its pre-drain inventory.

    :param agent: Native channel owner.
    :param events: Native publication-order events.
    :param active_subscriptions: Post-drain active subscription count.
    :param fatal: Optional sticky native fatal bitmask.
    """

    after = _NativeChannelInventory(
        capacity=16,
        activeChannelSubscriptions=active_subscriptions,
        retainedPublicSubscriptions=active_subscriptions,
        fatal=_NativeFatalValue(int(fatal)),
    )
    agent.channel.batches.append(
        _NativeBatch(events=events, wakeCount=1, inventory=after)
    )
    agent.channel.inventory = dataclasses.replace(
        after,
        queuedChannelEvents=len(events),
    )


def _failure(reason: str) -> InjectTerminalOwnerFailure:
    """Create one distinguishable immutable command fixture.

    :param reason: Stable command identity.
    :returns: Immutable owner failure command.
    """

    return InjectTerminalOwnerFailure(
        cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
        reason=reason,
    )


def _cuda_identity(cookie: int) -> CudaCompletionIdentity:
    """Create one deterministic exact callback identity.

    :param cookie: Owner correlation cookie.
    :returns: Exact cookie and request generation.
    """

    return CudaCompletionIdentity(
        cookie=cookie,
        generation=cookie.to_bytes(16, "big"),
    )


def _decode_binding(room_id: int) -> TerminalRequestBinding:
    """Create one deterministic decode lifecycle binding.

    :param room_id: Packed request room identity.
    :returns: Exact decode request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=room_id.to_bytes(16, "big"),
        ),
        owner=TerminalProcessIdentity(
            process_generation=bytes.fromhex("00112233445566778899aabbccddeeff"),
            role=TerminalOwnerRole.DECODE,
            tp_rank=0,
            tp_size=1,
        ),
        rank_manifest_digest=bytes((room_id,)) * 32,
        allocation_digest=bytes((room_id + 1,)) * 32,
    )


def test_nixl_transfer_success_and_failure_route_exact_commands() -> None:
    """Terminal statuses select immutable commands and retire both routes."""

    success_subscription = _subscription(
        nixl_terminal_event_kind_t.TRANSFER, 10, 101, 1
    )
    failure_subscription = _subscription(
        nixl_terminal_event_kind_t.TRANSFER, 11, 102, 2
    )
    source, agent, _ = _nixl_source((success_subscription, failure_subscription))
    success_command = _failure("transfer-success")
    failure_command = _failure("transfer-failure")
    first = source.subscribe_transfer(
        object(),
        10,
        NixlTransferOwnerRoute(
            success=success_command,
            failure=_failure("unexpected-first-failure"),
        ),
    )
    second = source.subscribe_transfer(
        object(),
        11,
        NixlTransferOwnerRoute(
            success=_failure("unexpected-second-success"),
            failure=failure_command,
        ),
    )
    _queue_nixl_batch(
        agent,
        (
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.TRANSFER,
                ownerCookie=10,
                identity=101,
                generation=1,
                nativeTimestampNs=1001,
                transferStatus=NIXL_SUCCESS,
            ),
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.TRANSFER,
                ownerCookie=11,
                identity=102,
                generation=2,
                nativeTimestampNs=1002,
                transferStatus=NIXL_ERR_CANCELED,
            ),
        ),
        active_subscriptions=0,
    )

    envelopes = source.drain()
    assert tuple(envelope.command for envelope in envelopes) == (
        success_command,
        failure_command,
    )
    assert tuple(envelope.enqueued_ns for envelope in envelopes) == (1001, 1002)
    assert first.binding.owner_cookie == 10
    assert second.binding.owner_cookie == 11
    assert source.inventory().retained_count == 0
    assert success_subscription.release_calls == 1
    assert failure_subscription.release_calls == 1
    source.close()
    assert source.inventory().closed


def test_nixl_in_progress_terminal_event_is_source_fatal() -> None:
    """A nonterminal status cannot be laundered into a request failure."""

    native_subscription = _subscription(nixl_terminal_event_kind_t.TRANSFER, 12, 103, 7)
    source, agent, _ = _nixl_source((native_subscription,))
    subscription = source.subscribe_transfer(
        object(),
        12,
        NixlTransferOwnerRoute(
            success=TerminalOwnerPulse(),
            failure=_failure("terminal-failure"),
        ),
    )
    _queue_nixl_batch(
        agent,
        (
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.TRANSFER,
                ownerCookie=12,
                identity=103,
                generation=7,
                nativeTimestampNs=1003,
                transferStatus=NIXL_IN_PROG,
            ),
        ),
        active_subscriptions=0,
    )

    with pytest.raises(NixlTerminalOwnerEventSourceFatalError) as captured:
        source.drain()
    assert captured.value.inventory.registered_routes == (subscription.binding,)
    assert "in-progress status" in str(captured.value)


def test_nixl_capability_ready_failed_and_retired_preserve_route_lifetime() -> None:
    """READY retains its route while FAILED and RETIRED consume theirs."""

    failed_subscription = _subscription(
        nixl_terminal_event_kind_t.CAPABILITY, 20, 201, 3
    )
    retired_subscription = _subscription(
        nixl_terminal_event_kind_t.CAPABILITY, 21, 202, 4
    )
    source, agent, _ = _nixl_source((failed_subscription, retired_subscription))
    ready_command = _failure("capability-ready")
    failed_command = _failure("capability-failed")
    retired_command = _failure("capability-retired")
    source.subscribe_route_capability(
        _RemoteHandle(identity=201, generation=3),
        "UCX",
        20,
        NixlCapabilityOwnerRoute(
            ready=ready_command,
            failed=failed_command,
            retired=_failure("unexpected-retired"),
        ),
    )
    source.subscribe_route_capability(
        _RemoteHandle(identity=202, generation=4),
        "UCX",
        21,
        NixlCapabilityOwnerRoute(
            ready=_failure("unexpected-ready"),
            failed=_failure("unexpected-failed"),
            retired=retired_command,
        ),
    )
    _queue_nixl_batch(
        agent,
        (
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.CAPABILITY,
                ownerCookie=20,
                identity=201,
                generation=3,
                nativeTimestampNs=2001,
                capabilityState=nixl_terminal_capability_state_t.READY,
                capabilityEpoch=1,
            ),
        ),
        active_subscriptions=2,
    )
    assert source.drain()[0].command == ready_command
    assert len(source.inventory().registered_routes) == 2

    _queue_nixl_batch(
        agent,
        (
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.CAPABILITY,
                ownerCookie=20,
                identity=201,
                generation=3,
                nativeTimestampNs=2002,
                capabilityState=nixl_terminal_capability_state_t.FAILED,
                capabilityEpoch=2,
            ),
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.CAPABILITY,
                ownerCookie=21,
                identity=202,
                generation=4,
                nativeTimestampNs=2003,
                capabilityState=nixl_terminal_capability_state_t.RETIRED,
                capabilityEpoch=2,
            ),
        ),
        active_subscriptions=0,
    )
    assert tuple(envelope.command for envelope in source.drain()) == (
        failed_command,
        retired_command,
    )
    assert source.inventory().retained_count == 0
    assert failed_subscription.release_calls == 1
    assert retired_subscription.release_calls == 1
    source.close()


def test_nixl_unrouted_exact_identity_is_process_fatal_and_retained() -> None:
    """A native subscription outside the route registry cannot disappear."""

    native_subscription = _subscription(nixl_terminal_event_kind_t.TRANSFER, 30, 301, 5)
    source, agent, adapter = _nixl_source((native_subscription,))
    subscription = adapter.subscribe_transfer(object(), owner_cookie=30)
    _queue_nixl_batch(
        agent,
        (
            _NativeEvent(
                kind=nixl_terminal_event_kind_t.TRANSFER,
                ownerCookie=30,
                identity=301,
                generation=5,
                nativeTimestampNs=3001,
                transferStatus=NIXL_SUCCESS,
            ),
        ),
        active_subscriptions=0,
    )

    with pytest.raises(NixlTerminalOwnerEventSourceFatalError) as captured:
        source.drain()
    fatal = captured.value
    assert fatal.inventory.observed_unrouted == (subscription.binding,)
    assert fatal.inventory.retained_count == 1
    assert "cookie=30:identity=301:generation=5" in str(fatal)
    assert native_subscription.release_calls == 1


def test_nixl_native_fatal_keeps_every_registered_route_in_inventory() -> None:
    """Native queue failure reaches the owner with exact route evidence."""

    native_subscription = _subscription(nixl_terminal_event_kind_t.TRANSFER, 40, 401, 6)
    source, agent, _ = _nixl_source((native_subscription,))
    subscription = source.subscribe_transfer(
        object(),
        40,
        NixlTransferOwnerRoute(
            success=TerminalOwnerPulse(),
            failure=_failure("native-failure"),
        ),
    )
    _queue_nixl_batch(
        agent,
        (),
        active_subscriptions=1,
        fatal=NixlTerminalChannelFatal.QUEUE_OVERFLOW,
    )

    with pytest.raises(NixlTerminalOwnerEventSourceFatalError) as captured:
        source.drain()
    fatal = captured.value
    assert fatal.inventory.native.fatal is NixlTerminalChannelFatal.QUEUE_OVERFLOW
    assert fatal.inventory.registered_routes == (subscription.binding,)
    assert "cookie=40:identity=401:generation=6" in str(fatal)


def test_cuda_scatter_completion_routes_exact_generation_and_closes_cleanly() -> None:
    """One native callback becomes the exact immutable scatter transition."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    source = CudaTerminalOwnerEventSource("cuda-scatter-events", bridge)
    identity = _cuda_identity(50)
    command = ApplyDecodeLifecycleEvent(
        binding=_decode_binding(50),
        event=DecodeLifecycleEvent(kind=DecodeLifecycleEventKind.SCATTER_TERMINAL),
    )
    source.arm(identity, command)
    bridge.complete_synchronously_for_test(identity)

    envelopes = source.drain()
    assert len(envelopes) == 1
    assert envelopes[0].command == command
    assert source.inventory().retained_count == 0
    source.stop_submissions()
    assert source.join_producers()
    source.close()
    closed = source.inventory()
    assert closed.closed
    assert closed.native.closed
    assert closed.native.fatal_code is CudaCompletionFatalCode.NONE


def test_cuda_stale_identity_is_source_fatal_without_losing_generation() -> None:
    """A callback armed outside the registry remains exact fatal evidence."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    source = CudaTerminalOwnerEventSource("cuda-stale-events", bridge)
    identity = _cuda_identity(60)
    bridge.arm(identity)
    bridge.complete_synchronously_for_test(identity)

    with pytest.raises(CudaTerminalOwnerEventSourceFatalError) as captured:
        source.drain()
    fatal = captured.value
    assert fatal.inventory.observed_unrouted == (identity,)
    assert fatal.inventory.native.live_count == 0
    assert identity.generation.hex() in str(fatal)


def test_cuda_overflow_is_process_fatal_and_retains_all_route_identities() -> None:
    """Queue overflow does not erase drained or rejected callback identities."""

    bridge = CudaCompletionBridge(capacity=2, testing=True)
    source = CudaTerminalOwnerEventSource("cuda-overflow-events", bridge)
    identities = tuple(_cuda_identity(cookie) for cookie in (70, 71, 72))
    for identity in identities:
        source.arm(identity, TerminalOwnerPulse())
        bridge.complete_synchronously_for_test(identity)

    with pytest.raises(CudaTerminalOwnerEventSourceFatalError) as captured:
        source.drain()
    fatal = captured.value
    assert fatal.inventory.native.fatal_code is CudaCompletionFatalCode.QUEUE_OVERFLOW
    assert fatal.inventory.native.fatal_identity == identities[2]
    assert fatal.inventory.registered_routes == identities
    assert fatal.inventory.native.live_count == 1
    assert len(fatal.retained_identity_labels) == 3


def test_owner_converts_native_source_fatal_and_keeps_identity_in_reason() -> None:
    """The reactor enters process-fatal with lossless source evidence."""

    bridge = CudaCompletionBridge(capacity=2, testing=True)
    source = CudaTerminalOwnerEventSource("cuda-owner-fatal", bridge)
    owner = PackedTerminalProgressOwner(
        submission_capacity=8,
        output_capacity=8,
        event_sources=(
            TerminalOwnerEventSourceRegistration(
                source=source,
                close_on_shutdown=False,
            ),
        ),
    )
    owner.start()
    owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.RUNNING,
        timeout_seconds=2.0,
    )
    identities = tuple(_cuda_identity(cookie) for cookie in (80, 81, 82))
    for identity in identities:
        source.arm(identity, TerminalOwnerPulse())
        bridge.complete_synchronously_for_test(identity)

    fatal = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.PROCESS_FATAL,
        timeout_seconds=2.0,
    )
    assert fatal.fatal_cause is TerminalOwnerFatalCause.EVENT_SOURCE_FAILURE
    assert fatal.fatal_reason is not None
    assert identities[2].generation.hex() in fatal.fatal_reason
    assert owner.join(timeout_seconds=2.0)
