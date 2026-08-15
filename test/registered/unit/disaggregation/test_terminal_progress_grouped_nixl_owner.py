import dataclasses
import time
from collections.abc import Callable, Sequence

import pytest
from nixl._bindings import NIXL_ERR_CANCELED, NIXL_IN_PROG, NIXL_SUCCESS

from sglang.srt.disaggregation.terminal_progress import clock as clock_module
from sglang.srt.disaggregation.terminal_progress import (
    grouped_nixl_owner as grouped_module,
)
from sglang.srt.disaggregation.terminal_progress.clock import TerminalOwnerClock
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlTerminalOwner,
    GroupedNixlTransferMember,
    grouped_nixl_source_members,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlTerminalBackendLifecycleInventory,
    NixlTerminalCancelOutcome,
    NixlTerminalChannelFatal,
    NixlTerminalChannelInventory,
    NixlTerminalEventBatch,
    NixlTerminalEventKind,
    NixlTerminalLifecycleError,
    NixlTerminalProcessFatalError,
    NixlTerminalSubscription,
    NixlTerminalSubscriptionBinding,
    NixlTransferTerminalEvent,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DIGEST = b"d" * 32


@dataclasses.dataclass(frozen=True, slots=True)
class _Status:
    """Named native completion status fixture."""

    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class _Receipt:
    """Native-shaped take-once transfer completion receipt."""

    handleIdentity: int
    generation: int
    submissionSealed: bool = True
    completionClaimed: bool = True
    status: _Status = _Status("NIXL_SUCCESS")


class _Handle:
    """Exact transfer-handle and receipt lifetime fixture."""

    identity: int
    generation: int
    release_calls: int
    receipt_taken: bool

    def __init__(self, identity: int, generation: int) -> None:
        """Create one live exact-generation handle.

        :param identity: Stable native handle identity.
        :param generation: Exact posted generation.
        """

        self.identity = identity
        self.generation = generation
        self.release_calls = 0
        self.receipt_taken = False

    def release(self) -> None:
        """Record exact handle release."""

        self.release_calls += 1


class _Agent:
    """Completion-receipt surface consumed by grouped ownership."""

    def take_xfer_completion_receipt(self, handle: _Handle) -> _Receipt | None:
        """Take one exact successful receipt.

        :param handle: Exact terminal transfer handle.
        :returns: Take-once receipt, otherwise ``None`` on replay.
        """

        if handle.receipt_taken:
            return None
        handle.receipt_taken = True
        return _Receipt(
            handleIdentity=handle.identity,
            generation=handle.generation,
        )


class _SequenceClock(TerminalOwnerClock):
    """Prospective raw-clock values for deterministic post timing."""

    values: list[int]

    def __init__(self, values: Sequence[int]) -> None:
        """Retain prospective timestamp values.

        :param values: Timestamps returned in call order.
        """

        self.values = list(values)

    def now_ns(self) -> int:
        """Consume the next planned timestamp.

        :returns: Next prospective timestamp.
        """

        if len(self.values) == 0:
            raise AssertionError("post clock consumed more than planned")
        return self.values.pop(0)


class _Adapter:
    """Deterministic terminal channel preserving exact subscription routing."""

    capacity: int
    closed: bool
    subscriptions: dict[NixlTerminalSubscriptionBinding, NixlTerminalSubscription]
    bindings_by_handle: dict[_Handle, NixlTerminalSubscriptionBinding]
    batches: list[tuple[NixlTransferTerminalEvent, ...]]

    def __init__(self, agent: object, capacity: int) -> None:
        """Create one empty channel fixture.

        :param agent: Unused qualified-agent sentinel.
        :param capacity: Frozen event capacity.
        """

        del agent
        self.capacity = capacity
        self.closed = False
        self.subscriptions = {}
        self.bindings_by_handle = {}
        self.batches = []

    def fileno(self) -> int:
        """Return one stable borrowed descriptor.

        :returns: Fixture descriptor.
        """

        return 41

    def subscribe_transfer(
        self,
        handle: _Handle,
        owner_cookie: int,
    ) -> NixlTerminalSubscription:
        """Retain one exact member subscription.

        :param handle: Exact unposted transfer handle.
        :param owner_cookie: Group-owner correlation identity.
        :returns: Exact public subscription.
        """

        binding = NixlTerminalSubscriptionBinding(
            kind=NixlTerminalEventKind.TRANSFER,
            owner_cookie=owner_cookie,
            identity=handle.identity,
            generation=handle.generation,
        )
        if binding in self.subscriptions or handle in self.bindings_by_handle:
            raise NixlTerminalLifecycleError("duplicate fixture subscription")
        subscription = NixlTerminalSubscription(binding=binding)
        self.subscriptions[binding] = subscription
        self.bindings_by_handle[handle] = binding
        return subscription

    def queue(
        self,
        handle: _Handle,
        status: object,
        timestamp_ns: int,
    ) -> None:
        """Queue one exact terminal token as its own drain batch.

        :param handle: Subscribed exact-generation handle.
        :param status: Native terminal status.
        :param timestamp_ns: Native publication timestamp.
        """

        binding = self.bindings_by_handle[handle]
        self.batches.append(
            (
                NixlTransferTerminalEvent(
                    owner_cookie=binding.owner_cookie,
                    identity=binding.identity,
                    generation=binding.generation,
                    status=status,
                    native_timestamp_ns=timestamp_ns,
                ),
            )
        )

    def queue_batch(
        self,
        events: Sequence[tuple[_Handle, object, int]],
    ) -> None:
        """Queue several native terminal tokens in publication order.

        :param events: Handle, status, and native timestamp triples.
        """

        batch: list[NixlTransferTerminalEvent] = []
        for handle, status, timestamp_ns in events:
            binding = self.bindings_by_handle[handle]
            batch.append(
                NixlTransferTerminalEvent(
                    owner_cookie=binding.owner_cookie,
                    identity=binding.identity,
                    generation=binding.generation,
                    status=status,
                    native_timestamp_ns=timestamp_ns,
                )
            )
        self.batches.append(tuple(batch))

    def drain(self) -> NixlTerminalEventBatch:
        """Drain one exact publication batch.

        :returns: Immutable typed event batch.
        """

        events = () if len(self.batches) == 0 else self.batches.pop(0)
        for event in events:
            if event.binding not in self.subscriptions:
                raise NixlTerminalProcessFatalError(
                    "fixture terminal event has no exact subscription",
                    self.query_inventory(),
                )
            del self.subscriptions[event.binding]
        return NixlTerminalEventBatch(
            events=events,
            wake_count=int(len(events) > 0),
            inventory=self.query_inventory(),
        )

    def cancel(
        self,
        subscription: NixlTerminalSubscription,
    ) -> NixlTerminalCancelOutcome:
        """Immediately cancel an uncompleted fixture subscription.

        :param subscription: Exact owned subscription.
        :returns: Immediate release outcome.
        """

        current = self.subscriptions.get(subscription.binding)
        if current is not subscription:
            raise NixlTerminalLifecycleError("fixture subscription is not active")
        del self.subscriptions[subscription.binding]
        return NixlTerminalCancelOutcome.RELEASED

    def query_inventory(self) -> NixlTerminalChannelInventory:
        """Return exact native-shaped channel inventory.

        :returns: Immutable channel inventory.
        """

        return _native_inventory(
            capacity=self.capacity,
            active_subscriptions=len(self.subscriptions),
            closed=self.closed,
        )

    def close(self) -> NixlTerminalChannelInventory:
        """Close only after every fixture subscription drained.

        :returns: Exact clean-closed inventory.
        """

        if len(self.subscriptions) != 0:
            raise NixlTerminalLifecycleError(
                "fixture channel retained active subscriptions"
            )
        self.closed = True
        return self.query_inventory()


def _native_inventory(
    *,
    capacity: int,
    active_subscriptions: int,
    closed: bool,
) -> NixlTerminalChannelInventory:
    """Construct one typed empty-backend channel inventory.

    :param capacity: Frozen channel capacity.
    :param active_subscriptions: Current exact subscription population.
    :param closed: Whether clean closure completed.
    :returns: Complete native inventory.
    """

    backend = NixlTerminalBackendLifecycleInventory(
        source_deliveries_outstanding=0,
        source_local_pending=0,
        source_receipt_pending=0,
        destination_pending=0,
        destination_admitting=0,
        destination_committed=0,
        destination_replaying=0,
        destination_quarantined=0,
        active_native_deadlines=0,
        source_deliveries=(),
        destination_deliveries=(),
        native_deadlines=(),
    )
    return NixlTerminalChannelInventory(
        capacity=capacity,
        queued_channel_events=0,
        active_channel_subscriptions=active_subscriptions,
        retained_public_subscriptions=active_subscriptions,
        backend_producers=0,
        active_callback_slots=0,
        queued_owner_continuations=0,
        backend_lifecycle=backend,
        accepting_subscriptions=not closed,
        closed=closed,
        fatal=NixlTerminalChannelFatal.NONE,
        eventfd_error=0,
    )


def _request_binding(digest: bytes) -> NativeTerminalRequestBinding:
    """Create one exact source request binding.

    :param digest: Exact lifecycle digest.
    :returns: Native request binding.
    """

    owner = NativeTerminalProcessIdentity(
        process_generation=b"g" * 16,
        role=NativeTerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
        digest=b"o" * 32,
    )
    return NativeTerminalRequestBinding(
        room_id=9,
        request_generation=b"r" * 16,
        owner=owner,
        rank_manifest_digest=b"m" * 32,
        allocation_digest=b"a" * 32,
        digest=digest,
    )


def _action(
    kind: NativeTerminalOwnerActionKind,
    *,
    action_id: int,
    digest: bytes = _DIGEST,
) -> NativeTerminalOwnerAction:
    """Create one authoritative owner action.

    :param kind: Exact action kind.
    :param action_id: One-shot native action identity.
    :param digest: Exact source lifecycle digest.
    :returns: Immutable native owner action.
    """

    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=kind,
        binding=_request_binding(digest),
        commit_timestamp_ns=19,
        receipt=None,
    )


def _owner(
    monkeypatch: pytest.MonkeyPatch,
    post_timestamps: Sequence[int],
) -> tuple[GroupedNixlTerminalOwner, _Adapter, _Agent]:
    """Create one grouped owner around the deterministic adapter.

    :param monkeypatch: Pytest mutation fixture.
    :param post_timestamps: Prospective post-start clock values.
    :returns: Grouped owner, adapter, and completion-receipt agent.
    """

    adapters: list[_Adapter] = []

    def create_adapter(agent: object, capacity: int) -> _Adapter:
        adapter = _Adapter(agent, capacity)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(grouped_module, "NixlTerminalEventAdapter", create_adapter)
    agent = _Agent()
    owner = GroupedNixlTerminalOwner(
        agent,
        channel_capacity=8,
        clock=_SequenceClock(post_timestamps),
    )
    return owner, adapters[0], agent


def _post_member(
    endpoint: object,
    handle: _Handle,
    *,
    digest: bytes = _DIGEST,
    post: Callable[[object], object] | None = None,
) -> object:
    """Arm and post one endpoint member.

    :param endpoint: Main or DFlash grouped endpoint.
    :param handle: Exact transfer handle.
    :param digest: Exact request lifecycle digest.
    :param post: Optional prospective post callback.
    :returns: Opaque grouped member authority.
    """

    transfer = endpoint.arm_transfer(handle, digest)
    callback = (lambda exact_handle: exact_handle) if post is None else post
    assert endpoint.post_transfer(transfer, callback) is handle
    return transfer


def test_tp2_schema_has_two_main_transfers_and_one_canonical_boundary() -> None:
    """The authenticated DFlash token is canonical-only, never TP-replicated."""

    source_rank_members = tuple(
        grouped_nixl_source_members(rank == 0) for rank in range(2)
    )
    assert source_rank_members == (
        (
            GroupedNixlTransferMember.MAIN,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        ),
        (GroupedNixlTransferMember.MAIN,),
    )
    flattened = tuple(member for members in source_rank_members for member in members)
    assert flattened.count(GroupedNixlTransferMember.MAIN) == 2
    assert flattened.count(GroupedNixlTransferMember.DFLASH_BOUNDARY) == 1


def test_grouped_transfer_exposes_immutable_native_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream receipt validation sees the exact subscribed generation."""

    owner, _, _ = _owner(monkeypatch, (100,))
    handle = _Handle(43, 7)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))

    transfer = owner.main_endpoint.arm_transfer(handle, _DIGEST)

    assert transfer.handle_identity == 43
    assert transfer.generation == 7
    with pytest.raises(AttributeError):
        transfer.handle_identity = 44
    with pytest.raises(AttributeError):
        transfer.generation = 8


def test_default_post_clock_reads_clock_monotonic_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production post anchors share NIXL's native raw-clock domain."""

    adapters: list[_Adapter] = []

    def create_adapter(agent: object, capacity: int) -> _Adapter:
        adapter = _Adapter(agent, capacity)
        adapters.append(adapter)
        return adapter

    clock_ids: list[int] = []

    def clock_gettime_ns(clock_id: int) -> int:
        clock_ids.append(clock_id)
        return 100

    monkeypatch.setattr(grouped_module, "NixlTerminalEventAdapter", create_adapter)
    monkeypatch.setattr(clock_module.time, "clock_gettime_ns", clock_gettime_ns)
    owner = GroupedNixlTerminalOwner(_Agent(), channel_capacity=8)
    handle = _Handle(31, 1)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))
    _post_member(owner.main_endpoint, handle)
    owner.seal_group(_DIGEST)
    adapters[0].queue(handle, NIXL_SUCCESS, 101)

    result = owner.drain()[0]
    assert clock_ids == [time.CLOCK_MONOTONIC_RAW]
    assert result.member_timings[0].post_started_ns == 100


@pytest.mark.parametrize("completion_order", ((0, 1), (1, 0)))
def test_dual_member_success_waits_for_both_orders_and_releases_exactly(
    monkeypatch: pytest.MonkeyPatch,
    completion_order: tuple[int, int],
) -> None:
    """Success aggregates both handles in either terminal publication order."""

    owner, adapter, _ = _owner(monkeypatch, (100, 200))
    handles = (_Handle(41, 1), _Handle(42, 1))
    owner.begin_group(
        _DIGEST,
        (
            GroupedNixlTransferMember.MAIN,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        ),
    )
    transfers = (
        _post_member(owner.main_endpoint, handles[0]),
        _post_member(owner.dflash_endpoint, handles[1]),
    )
    owner.seal_group(_DIGEST)

    first, second = completion_order
    adapter.queue(handles[first], NIXL_SUCCESS, 300 + first)
    assert owner.drain() == ()
    adapter.queue(handles[second], NIXL_SUCCESS, 400 + second)
    results = owner.drain()

    assert len(results) == 1
    result = results[0]
    assert result.successful
    assert result.transfer_count == 2
    assert tuple(timing.member for timing in result.member_timings) == (
        GroupedNixlTransferMember.MAIN,
        GroupedNixlTransferMember.DFLASH_BOUNDARY,
    )
    assert tuple(timing.post_started_ns for timing in result.member_timings) == (
        100,
        200,
    )
    owner.acknowledge_result(result)
    outcome = _action(
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        action_id=11,
    )
    owner.main_endpoint.settle_success(transfers[0], outcome)
    owner.dflash_endpoint.settle_success(transfers[1], outcome)
    ack = _action(NativeTerminalOwnerActionKind.SOURCE_ACK_READY, action_id=12)
    owner.main_endpoint.release_transfer(transfers[0], ack)
    owner.dflash_endpoint.release_transfer(transfers[1], ack)

    assert tuple(handle.release_calls for handle in handles) == (1, 1)
    owner.stop_admission()
    inventory = owner.close_clean()
    assert inventory.closed
    assert inventory.active_group_count == 0
    assert inventory.released_transfer_count == 2
    assert inventory.native.is_clean_closed


def test_immediate_completion_during_post_remains_producer_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback queued inside post cannot publish before the group seals."""

    owner, adapter, _ = _owner(monkeypatch, (100,))
    handle = _Handle(51, 2)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))

    def post(exact_handle: object) -> object:
        if exact_handle is not handle:
            raise AssertionError("group owner changed transfer handle identity")
        adapter.queue(handle, NIXL_SUCCESS, 101)
        return exact_handle

    transfer = _post_member(owner.main_endpoint, handle, post=post)
    assert owner.drain() == ()
    owner.seal_group(_DIGEST)
    result = owner.drain()[0]
    assert result.member_timings[0].post_started_ns == 100
    assert result.member_timings[0].native_terminal_ns == 101
    owner.acknowledge_result(result)
    outcome = _action(
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        action_id=21,
    )
    owner.main_endpoint.settle_success(transfer, outcome)
    ack = _action(NativeTerminalOwnerActionKind.SOURCE_ACK_READY, action_id=22)
    owner.main_endpoint.release_transfer(transfer, ack)


def test_first_member_failure_emits_once_without_waiting_for_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first failed member owns one aggregate failure transition."""

    owner, adapter, _ = _owner(monkeypatch, (100, 200))
    main = _Handle(61, 3)
    boundary = _Handle(62, 3)
    owner.begin_group(
        _DIGEST,
        (
            GroupedNixlTransferMember.MAIN,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        ),
    )
    main_transfer = _post_member(owner.main_endpoint, main)
    boundary_transfer = _post_member(owner.dflash_endpoint, boundary)
    owner.seal_group(_DIGEST)

    adapter.queue(boundary, NIXL_ERR_CANCELED, 250)
    result = owner.drain()[0]
    assert not result.successful
    assert result.native_timestamp_ns == 250
    assert tuple(timing.member for timing in result.member_timings) == (
        GroupedNixlTransferMember.DFLASH_BOUNDARY,
    )
    owner.acknowledge_result(result)
    failure = _action(
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        action_id=31,
    )
    owner.main_endpoint.settle_failure(main_transfer, failure)
    owner.dflash_endpoint.settle_failure(boundary_transfer, failure)

    adapter.queue(main, NIXL_SUCCESS, 300)
    assert owner.drain() == ()
    inventory = owner.inventory()
    assert inventory.active_group_count == 1
    assert inventory.quarantined_transfer_count == 2


def test_post_exception_quarantine_suppresses_later_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous post failure cannot issue a second lifecycle failure."""

    owner, adapter, _ = _owner(monkeypatch, (100,))
    handle = _Handle(71, 4)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))
    transfer = owner.main_endpoint.arm_transfer(handle, _DIGEST)

    def fail_post(exact_handle: object) -> object:
        if exact_handle is not handle:
            raise AssertionError("group owner changed transfer handle identity")
        adapter.queue(handle, NIXL_ERR_CANCELED, 150)
        raise OSError("post failed after native ownership became ambiguous")

    with pytest.raises(OSError, match="post failed"):
        owner.main_endpoint.post_transfer(transfer, fail_post)
    owner.quarantine_group(_DIGEST, "source work already issued request failure")
    assert owner.drain() == ()
    assert owner.inventory().quarantined_transfer_count == 1


def test_dual_release_requires_one_exact_source_ack_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both handles release only under the same one-shot ACK authority."""

    owner, adapter, _ = _owner(monkeypatch, (100, 200))
    handles = (_Handle(81, 5), _Handle(82, 5))
    owner.begin_group(
        _DIGEST,
        (
            GroupedNixlTransferMember.MAIN,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        ),
    )
    transfers = (
        _post_member(owner.main_endpoint, handles[0]),
        _post_member(owner.dflash_endpoint, handles[1]),
    )
    owner.seal_group(_DIGEST)
    adapter.queue_batch(
        (
            (handles[0], NIXL_SUCCESS, 300),
            (handles[1], NIXL_SUCCESS, 301),
        )
    )
    result = owner.drain()[0]
    owner.acknowledge_result(result)
    outcome = _action(
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        action_id=41,
    )
    owner.main_endpoint.settle_success(transfers[0], outcome)
    owner.dflash_endpoint.settle_success(transfers[1], outcome)
    first_ack = _action(
        NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        action_id=42,
    )
    owner.main_endpoint.release_transfer(transfers[0], first_ack)

    wrong_ack = _action(
        NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
        action_id=43,
    )
    with pytest.raises(NixlTerminalLifecycleError, match="changed source ACK"):
        owner.dflash_endpoint.release_transfer(transfers[1], wrong_ack)
    owner.dflash_endpoint.release_transfer(transfers[1], first_ack)


def test_cancellation_retains_quarantined_authority_and_blocks_clean_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation never launders ambiguous ownership into clean teardown."""

    owner, _, _ = _owner(monkeypatch, (100,))
    handle = _Handle(91, 6)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))
    transfer = _post_member(owner.main_endpoint, handle)
    owner.seal_group(_DIGEST)
    owner.main_endpoint.cancel_transfer(transfer)

    inventory = owner.inventory()
    assert inventory.active_group_count == 1
    assert inventory.active_transfer_count == 1
    assert inventory.quarantined_transfer_count == 1
    assert inventory.native.active_channel_subscriptions == 0
    owner.stop_admission()
    with pytest.raises(NixlTerminalLifecycleError, match="retained authority"):
        owner.close_clean()


def test_in_progress_or_replayed_terminal_token_is_process_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid terminal semantics cannot become a request-level result."""

    owner, adapter, _ = _owner(monkeypatch, (100,))
    handle = _Handle(101, 7)
    owner.begin_group(_DIGEST, (GroupedNixlTransferMember.MAIN,))
    _post_member(owner.main_endpoint, handle)
    owner.seal_group(_DIGEST)
    adapter.queue(handle, NIXL_IN_PROG, 150)
    with pytest.raises(NixlTerminalProcessFatalError, match="in-progress"):
        owner.drain()

    adapter.queue(handle, NIXL_SUCCESS, 160)
    with pytest.raises(NixlTerminalProcessFatalError, match="no exact subscription"):
        owner.drain()
