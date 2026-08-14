import dataclasses
import threading
from collections.abc import Callable, Sequence

import pytest
from nixl._api import nixl_terminal_event_kind_t
from nixl._bindings import NIXL_IN_PROG, NIXL_SUCCESS, nixl_status_t

from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlDirectTerminalAdapterDisposition,
    NixlDirectTerminalOwnerAdapter,
    NixlTerminalLifecycleError,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalNativeProducerBinding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclasses.dataclass(frozen=True, slots=True)
class _Status:
    """Named native status fixture."""

    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class _CompletionReceipt:
    """Native-shaped take-once completion receipt fixture."""

    handleIdentity: int = 41
    generation: int = 7
    submissionSealed: bool = True
    completionClaimed: bool = True
    status: _Status = _Status("NIXL_SUCCESS")


@dataclasses.dataclass(frozen=True, slots=True)
class _SubscriptionInfo:
    """Native-shaped exact transfer subscription snapshot."""

    kind: nixl_terminal_event_kind_t = nixl_terminal_event_kind_t.TRANSFER
    identity: int = 41
    generation: int = 7
    active: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class _ProducerInventory:
    """Native-shaped direct producer inventory fixture."""

    registering_count: int = 0
    submitted_count: int = 0
    active_callback_count: int = 0
    active_registration_count: int = 0
    total_subscriptions: int = 0
    total_delivered: int = 0
    successful_terminal_event_count: int = 0
    failure_terminal_event_count: int = 0
    owner_submission_failure_count: int = 0
    admission_open: bool = True
    retirement_requested: bool = False
    joined: bool = False
    closed: bool = False
    fatal_code: str = "NONE"
    fatal_status: int = 0
    fatal_binding: bytes | None = None


class _Handle:
    """Deterministic exact transfer-handle lifetime fixture."""

    release_calls: int

    def __init__(self) -> None:
        """Construct one live handle."""

        self.release_calls = 0

    def release(self) -> None:
        """Record exact handle release."""

        self.release_calls += 1


class _Subscription:
    """Deterministic direct owner subscription fixture."""

    _info: _SubscriptionInfo
    _statuses: list[nixl_status_t]
    release_calls: int

    def __init__(
        self,
        info: _SubscriptionInfo | None = None,
        statuses: Sequence[nixl_status_t] = (NIXL_SUCCESS,),
    ) -> None:
        """Construct one exact-generation subscription.

        :param info: Initial native subscription snapshot.
        :param statuses: Prospective release results.
        """

        self._info = _SubscriptionInfo() if info is None else info
        self._statuses = list(statuses)
        self.release_calls = 0

    def query(self) -> _SubscriptionInfo:
        """Return the current exact-generation snapshot.

        :returns: Current subscription snapshot.
        """

        return self._info

    def mark_terminal(self) -> None:
        """Make this exact transfer subscription terminal."""

        self._info = dataclasses.replace(self._info, active=False)

    def release(self) -> nixl_status_t:
        """Consume the next prospective release result.

        :returns: Planned native release status.
        """

        if len(self._statuses) == 0:
            raise AssertionError("subscription released more than planned")
        self.release_calls += 1
        return self._statuses.pop(0)


class _Producer:
    """Deterministic process-lifetime direct producer fixture."""

    _inventory: _ProducerInventory
    join_results: list[bool]
    calls: list[str]

    def __init__(self, join_results: Sequence[bool] = (True,)) -> None:
        """Construct one clean open producer.

        :param join_results: Prospective native join results.
        """

        self._inventory = _ProducerInventory()
        self.join_results = list(join_results)
        self.calls = []

    def inventory(self) -> _ProducerInventory:
        """Return current native producer inventory.

        :returns: Immutable inventory fixture.
        """

        return self._inventory

    def note_subscription(self) -> None:
        """Record one newly armed transfer generation."""

        self._inventory = dataclasses.replace(
            self._inventory,
            submitted_count=self._inventory.submitted_count + 1,
            total_subscriptions=self._inventory.total_subscriptions + 1,
        )

    def note_terminal(self, *, success: bool) -> None:
        """Record one direct terminal callback delivery.

        :param success: Whether native completion succeeded.
        """

        self._inventory = dataclasses.replace(
            self._inventory,
            submitted_count=self._inventory.submitted_count - 1,
            total_delivered=self._inventory.total_delivered + 1,
            successful_terminal_event_count=(
                self._inventory.successful_terminal_event_count + int(success)
            ),
            failure_terminal_event_count=(
                self._inventory.failure_terminal_event_count + int(not success)
            ),
        )

    def note_discarded(self) -> None:
        """Remove one unposted subscription from live inventory."""

        self._inventory = dataclasses.replace(
            self._inventory,
            submitted_count=self._inventory.submitted_count - 1,
        )

    def stop_admission(self) -> None:
        """Record permanent admission closure."""

        self.calls.append("stop_admission")
        self._inventory = dataclasses.replace(
            self._inventory,
            admission_open=False,
            retirement_requested=True,
        )

    def join(self, timeout_seconds: float) -> bool:
        """Return the next prospective join result.

        :param timeout_seconds: Required positive join bound.
        :returns: Planned join result.
        """

        if timeout_seconds <= 0.0 or len(self.join_results) == 0:
            raise AssertionError("invalid or unplanned producer join")
        self.calls.append("join")
        result = self.join_results.pop(0)
        if result:
            self._inventory = dataclasses.replace(
                self._inventory,
                joined=True,
            )
        return result

    def close(self) -> None:
        """Record exact-zero native producer closure."""

        self.calls.append("close")
        self._inventory = dataclasses.replace(
            self._inventory,
            closed=True,
        )


class _Agent:
    """Strict qualified direct NIXL agent surface fixture."""

    producer: _Producer
    subscriptions: list[_Subscription]
    receipts: list[_CompletionReceipt | None]
    calls: list[str]

    def __init__(
        self,
        *,
        producer: _Producer | None = None,
        subscriptions: Sequence[_Subscription] | None = None,
        receipts: Sequence[_CompletionReceipt | None] | None = None,
    ) -> None:
        """Construct one deterministic agent.

        :param producer: Optional native producer fixture.
        :param subscriptions: Prospective subscriptions.
        :param receipts: Prospective completion receipts.
        """

        self.producer = _Producer() if producer is None else producer
        self.subscriptions = (
            [_Subscription()] if subscriptions is None else list(subscriptions)
        )
        self.receipts = [_CompletionReceipt()] if receipts is None else list(receipts)
        self.calls = []

    def create_terminal_owner_producer(
        self,
        producer_api: object,
        producer_context: object,
    ) -> _Producer:
        """Return the sole direct producer after validating capsules.

        :param producer_api: Exact test API capsule.
        :param producer_context: Exact test context capsule.
        :returns: Sole native producer fixture.
        """

        if producer_api != "api" or producer_context != "context":
            raise AssertionError("unexpected native producer capsules")
        self.calls.append("create_producer")
        return self.producer

    def subscribe_xfer_terminal_owner(
        self,
        producer: _Producer,
        handle: _Handle,
        binding_digest: bytes,
    ) -> _Subscription:
        """Arm and return the next exact subscription.

        :param producer: Sole native producer fixture.
        :param handle: Exact live transfer handle.
        :param binding_digest: Exact owner binding.
        :returns: Planned exact-generation subscription.
        """

        if (
            producer is not self.producer
            or type(handle) is not _Handle
            or len(binding_digest) != 32
            or len(self.subscriptions) == 0
        ):
            raise AssertionError("invalid direct subscription call")
        self.calls.append("subscribe")
        self.producer.note_subscription()
        return self.subscriptions.pop(0)

    def take_xfer_completion_receipt(
        self,
        handle: _Handle,
    ) -> _CompletionReceipt | None:
        """Return the next take-once completion receipt.

        :param handle: Exact live transfer handle.
        :returns: Planned receipt value.
        """

        if type(handle) is not _Handle or len(self.receipts) == 0:
            raise AssertionError("invalid or replayed receipt take")
        self.calls.append("take_receipt")
        return self.receipts.pop(0)


def _binding() -> NativeTerminalNativeProducerBinding:
    """Construct the exact runtime-owned native producer binding.

    :returns: Stable fixture binding.
    """

    return NativeTerminalNativeProducerBinding(
        producer_id=5,
        producer_api="api",
        producer_context="context",
    )


def _request_binding(digest: bytes) -> NativeTerminalRequestBinding:
    """Construct one native source request binding.

    :param digest: Exact lifecycle digest.
    :returns: Valid native request binding.
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
    digest: bytes,
    kind: NativeTerminalOwnerActionKind = (
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY
    ),
    *,
    action_id: int = 11,
) -> NativeTerminalOwnerAction:
    """Construct one authoritative owner action fixture.

    :param digest: Exact lifecycle digest.
    :param kind: Closed action kind.
    :param action_id: Exact one-shot action identity.
    :returns: Valid owner action.
    """

    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=kind,
        binding=_request_binding(digest),
        commit_timestamp_ns=19,
        receipt=None,
    )


def _post(agent: _Agent) -> Callable[[object], object]:
    """Construct a posting operation which proves arm-before-post ordering.

    :param agent: Agent whose call order is asserted.
    :returns: Posting callback.
    """

    def post(handle: object) -> object:
        if type(handle) is not _Handle or agent.calls[-1] != "subscribe":
            raise AssertionError("transfer posted before its exact subscription")
        agent.calls.append("post")
        return handle

    return post


def _terminal_success(
    agent: _Agent,
    subscription: _Subscription,
) -> None:
    """Commit native successful terminality in fixtures.

    :param agent: Agent owning the producer.
    :param subscription: Exact transfer subscription.
    """

    subscription.mark_terminal()
    agent.producer.note_terminal(success=True)


def test_arms_before_post_and_consumes_exact_completion_authority_once() -> None:
    subscription = _Subscription()
    agent = _Agent(subscriptions=(subscription,))
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    digest = b"b" * 32

    transfer = adapter.arm_transfer(handle, digest)
    assert adapter.query_inventory().armed_count == 1
    adapter.post_transfer(transfer, _post(agent))
    _terminal_success(agent, subscription)
    receipt = adapter.settle_success(transfer, _action(digest))

    assert receipt.generation == transfer.generation
    assert agent.calls == [
        "create_producer",
        "subscribe",
        "post",
        "take_receipt",
    ]
    assert subscription.release_calls == 1
    assert adapter.query_inventory().settled_count == 1
    ack = _action(digest, NativeTerminalOwnerActionKind.SOURCE_ACK_READY)
    adapter.release_transfer(transfer, ack)
    assert handle.release_calls == 1
    assert adapter.query_inventory().transfer_count == 0
    with pytest.raises(NixlTerminalLifecycleError, match="not owned"):
        adapter.release_transfer(transfer, ack)


def test_fast_terminal_callback_during_post_preserves_settled_state() -> None:
    subscription = _Subscription()
    agent = _Agent(subscriptions=(subscription,))
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    digest = b"c" * 32
    transfer = adapter.arm_transfer(handle, digest)

    def post(_: object) -> object:
        terminal = threading.Thread(
            target=lambda: (
                _terminal_success(agent, subscription),
                adapter.settle_success(transfer, _action(digest)),
            )
        )
        terminal.start()
        terminal.join(timeout=1.0)
        assert not terminal.is_alive()
        return handle

    adapter.post_transfer(transfer, post)

    inventory = adapter.query_inventory()
    assert inventory.settled_count == 1
    assert inventory.posted_count == 0
    assert subscription.release_calls == 1


def test_post_failure_is_ambiguous_and_retains_every_authority() -> None:
    subscription = _Subscription()
    agent = _Agent(subscriptions=(subscription,), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    transfer = adapter.arm_transfer(handle, b"q" * 32)

    def fail_post(_: object) -> object:
        raise RuntimeError("post failed after entering native boundary")

    with pytest.raises(RuntimeError, match="post failed"):
        adapter.post_transfer(transfer, fail_post)

    inventory = adapter.query_inventory()
    assert inventory.ambiguous_count == 1
    assert subscription.release_calls == 0
    assert handle.release_calls == 0


def test_rejects_wrong_owner_action_before_taking_completion() -> None:
    subscription = _Subscription()
    agent = _Agent(subscriptions=(subscription,))
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    digest = b"d" * 32
    transfer = adapter.arm_transfer(_Handle(), digest)
    adapter.post_transfer(transfer, _post(agent))
    _terminal_success(agent, subscription)

    with pytest.raises(NixlTerminalLifecycleError, match="another transfer"):
        adapter.settle_success(transfer, _action(b"e" * 32))
    with pytest.raises(NixlTerminalLifecycleError, match="does not authorize"):
        adapter.settle_success(
            transfer,
            _action(
                digest,
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            ),
        )
    assert "take_receipt" not in agent.calls
    assert subscription.release_calls == 0


def test_generation_mismatch_retains_handle_and_subscription_fail_closed() -> None:
    subscription = _Subscription()
    agent = _Agent(
        subscriptions=(subscription,),
        receipts=(_CompletionReceipt(generation=8),),
    )
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    digest = b"f" * 32
    transfer = adapter.arm_transfer(handle, digest)
    adapter.post_transfer(transfer, _post(agent))
    _terminal_success(agent, subscription)

    with pytest.raises(NixlTerminalLifecycleError, match="generation"):
        adapter.settle_success(transfer, _action(digest))
    inventory = adapter.query_inventory()
    assert inventory.transfer_count == 1
    assert inventory.posted_count == 1
    assert subscription.release_calls == 0
    assert handle.release_calls == 0


def test_failure_settlement_never_takes_success_receipt() -> None:
    subscription = _Subscription(statuses=(NIXL_SUCCESS,))
    agent = _Agent(subscriptions=(subscription,), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    digest = b"h" * 32
    transfer = adapter.arm_transfer(handle, digest)
    adapter.post_transfer(transfer, _post(agent))
    subscription.mark_terminal()
    agent.producer.note_terminal(success=False)

    adapter.settle_failure(
        transfer,
        _action(digest, NativeTerminalOwnerActionKind.REQUEST_QUARANTINED),
    )

    assert "take_receipt" not in agent.calls
    assert subscription.release_calls == 1
    assert handle.release_calls == 0
    assert adapter.query_inventory().settled_count == 1


def test_pending_cancellation_retains_authority_until_terminal_failure() -> None:
    subscription = _Subscription(statuses=(NIXL_IN_PROG, NIXL_SUCCESS))
    agent = _Agent(subscriptions=(subscription,), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    digest = b"i" * 32
    transfer = adapter.arm_transfer(handle, digest)
    adapter.post_transfer(transfer, _post(agent))

    adapter.cancel_transfer(transfer)
    assert adapter.query_inventory().ambiguous_count == 1
    assert handle.release_calls == 0
    with pytest.raises(NixlTerminalLifecycleError, match="already requested"):
        adapter.cancel_transfer(transfer)
    subscription.mark_terminal()
    agent.producer.note_terminal(success=False)
    adapter.settle_failure(
        transfer,
        _action(digest, NativeTerminalOwnerActionKind.REQUEST_QUARANTINED),
    )
    assert subscription.release_calls == 2


def test_unposted_discard_is_the_only_preterminal_handle_release() -> None:
    subscription = _Subscription()
    agent = _Agent(subscriptions=(subscription,), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    handle = _Handle()
    transfer = adapter.arm_transfer(handle, b"j" * 32)

    adapter.discard_unposted(transfer)
    agent.producer.note_discarded()

    assert subscription.release_calls == 1
    assert handle.release_calls == 1
    assert adapter.query_inventory().transfer_count == 0


def test_ordered_shutdown_refuses_live_authority_and_handles_join_timeout() -> None:
    live_producer = _Producer()
    subscription = _Subscription()
    live_agent = _Agent(
        producer=live_producer,
        subscriptions=(subscription,),
        receipts=(),
    )
    live_adapter = NixlDirectTerminalOwnerAdapter(live_agent, _binding())
    handle = _Handle()
    live_adapter.arm_transfer(handle, b"k" * 32)
    live_adapter.stop_admission()
    assert live_adapter.join(0.1)
    with pytest.raises(NixlTerminalLifecycleError, match="local transfer"):
        live_adapter.close()
    assert live_producer.calls == ["stop_admission", "join"]

    producer = _Producer(join_results=(False, True))
    agent = _Agent(producer=producer, subscriptions=(), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())
    adapter.stop_admission()

    assert not adapter.join(0.1)
    assert adapter.query_inventory().disposition is (
        NixlDirectTerminalAdapterDisposition.DRAINING
    )
    assert adapter.join(0.1)
    assert producer.calls == ["stop_admission", "join", "join"]
    closed = adapter.close()

    assert producer.calls == ["stop_admission", "join", "join", "close"]
    assert closed.is_clean_closed


def test_shutdown_calls_stop_join_close_in_order_at_exact_zero() -> None:
    producer = _Producer()
    agent = _Agent(producer=producer, subscriptions=(), receipts=())
    adapter = NixlDirectTerminalOwnerAdapter(agent, _binding())

    closed = adapter.shutdown(0.1)

    assert producer.calls == ["stop_admission", "join", "close"]
    assert closed.is_clean_closed
    with pytest.raises(NixlTerminalLifecycleError, match="closed"):
        adapter.arm_transfer(_Handle(), b"l" * 32)
