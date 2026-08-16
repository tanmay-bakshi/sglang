import concurrent.futures
import threading
import types

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeSchedulerConsumer,
    PackedTerminalDecodeSchedulerRegistration,
    PackedTerminalDecodeServingComposition,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    SchedulerInboxFatalCause,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalProcessIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourcePlan,
    PackedTerminalSourceWriter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _DecodeWiring:
    """Deterministic decode lifecycle boundary for scheduler tests."""

    owner: object
    events: list[str]
    bound_transactions: list[PackedDecodeRequestTransaction]
    cancelled_transactions: list[PackedDecodeRequestTransaction]
    quarantines: list[tuple[PackedDecodeRequestTransaction, str]]
    return_another_owner: bool

    def __init__(self, owner: object, *, return_another_owner: bool = False) -> None:
        """Construct one exact-owner wiring fixture.

        :param owner: Scheduler request retained by the transaction.
        :param return_another_owner: Whether adoption violates owner identity.
        """

        self.owner = owner
        self.events = []
        self.bound_transactions = []
        self.cancelled_transactions = []
        self.quarantines = []
        self.return_another_owner = return_another_owner

    def bind_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        binding: TerminalRequestBinding,
        source_plan: PackedTerminalSourcePlan,
    ) -> None:
        """Record exact native lifecycle binding.

        :param transaction: Exact retained transaction.
        :param binding: Exact local decode binding.
        :param source_plan: Complete source identity plan.
        """

        assert transaction.request_owner is self.owner
        assert binding.request_key == source_plan.request_key
        self.bound_transactions.append(transaction)

    def consume_adoption_action(
        self,
        action: NativeTerminalOwnerAction,
        adopt_request,
        finalize_request,
    ) -> object:
        """Drive the frozen scheduler and metadata ordering.

        :param action: Exact adoption authority.
        :param adopt_request: Scheduler adoption callback.
        :param finalize_request: Scheduler finalization callback.
        :returns: Exact retained owner unless fault injection is active.
        """

        del action
        owner = object() if self.return_another_owner else self.owner
        self.events.append("allocation_adopted")
        adopt_request(owner)
        self.events.append("metadata_consumed")
        finalize_request(owner)
        self.events.append("local_ready")
        return owner

    def cancel_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> object:
        """Return exact safe rollback ownership.

        :param transaction: Exact retained transaction.
        :param reason: Stable cancellation reason.
        :returns: Exact retained owner.
        """

        assert len(reason) > 0
        self.cancelled_transactions.append(transaction)
        return self.owner

    def quarantine_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Record retained ambiguous transaction ownership.

        :param transaction: Exact retained transaction.
        :param reason: Stable failure reason.
        """

        self.quarantines.append((transaction, reason))


class _LaunchGate:
    """Provide exact native launch ownership to composition tests."""

    _next_token: int
    _active_token: int | None

    def __init__(self) -> None:
        """Construct one deterministic exact-token gate."""

        self._next_token = 1
        self._active_token = None

    def begin_scheduler_launch_handoff(self) -> int:
        """Acquire one exact test token.

        :returns: Newly minted positive token.
        """

        if self._active_token is not None:
            raise RuntimeError("test launch gate is already active")
        token = self._next_token
        self._next_token += 1
        self._active_token = token
        return token

    def end_scheduler_launch_handoff(self, token: int) -> None:
        """Release the matching exact test token.

        :param token: Token returned by the preceding acquisition.
        """

        if self._active_token != token:
            raise RuntimeError("test launch token is absent or stale")
        self._active_token = None


def _identity_graph(
    marker: int = 1,
) -> tuple[TerminalRequestBinding, PackedTerminalSourcePlan]:
    """Build one canonical TP1 source-to-decode identity graph.

    :param marker: Stable fixture generation marker.
    :returns: Decode binding and matching source plan.
    """

    key = PackedRequestKey(
        room_id=marker,
        request_generation=bytes((marker,)) * 16,
    )
    source = TerminalProcessIdentity(
        process_generation=bytes((marker + 1,)) * 16,
        role=TerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
    )
    decode = TerminalProcessIdentity(
        process_generation=bytes((marker + 2,)) * 16,
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    rank_manifest_digest = bytes((marker + 3,)) * 32
    allocation_digest = bytes((marker + 4,)) * 32
    binding = TerminalRequestBinding(
        request_key=key,
        owner=decode,
        rank_manifest_digest=rank_manifest_digest,
        allocation_digest=allocation_digest,
    )
    writer_id = StagingWriterId(
        transfer_source_rank=0,
        source_attn_tp_rank=0,
        source_pp_rank=0,
        source_cp_rank=0,
    )
    source_plan = PackedTerminalSourcePlan(
        request_key=key,
        writers=(
            PackedTerminalSourceWriter(
                writer_id=writer_id,
                process_identity=source,
            ),
        ),
        rank_manifest_digest=rank_manifest_digest,
        allocation_digest=allocation_digest,
        publication_identity=TerminalPublicationIdentity(
            request_key=key,
            publisher_process_generation=source.process_generation,
            publication_generation=bytes((marker + 5,)) * 16,
        ),
        request_ready_issuer=decode,
    )
    return binding, source_plan


def _transaction(
    key: PackedRequestKey,
    owner: object,
) -> PackedDecodeRequestTransaction:
    """Build a type-exact transaction shell for scheduler-boundary tests.

    The consumer intentionally needs only immutable identity and retained owner
    access. Lower-level actor semantics have their own real-runtime suite.

    :param key: Exact packed request generation.
    :param owner: Exact mutable scheduler request owner.
    :returns: Type-exact minimal transaction shell.
    """

    transaction = object.__new__(PackedDecodeRequestTransaction)
    transaction._request_key = key
    transaction._request_owner = owner
    transaction._lock = threading.RLock()
    transaction._state = PackedRequestTransactionState.PREPARED
    transaction._protocol = object()
    transaction._chunks = ()
    transaction._teardown_acks = set()
    transaction._auxiliary_outcome = None
    transaction._auxiliary_plan = types.SimpleNamespace(
        canonical_writer_id=StagingWriterId(
            transfer_source_rank=0,
            source_attn_tp_rank=0,
            source_pp_rank=0,
            source_cp_rank=0,
        )
    )
    return transaction


def _registration(
    *,
    marker: int = 1,
    adopt_request,
    finalize_request,
    cancel_request,
    quarantine_request,
) -> PackedTerminalDecodeSchedulerRegistration:
    """Build one exact callback-bearing decode registration.

    :param marker: Stable fixture generation marker.
    :param adopt_request: Scheduler adoption callback.
    :param finalize_request: Scheduler finalization callback.
    :param cancel_request: Safe cancellation callback.
    :param quarantine_request: Fail-closed retention callback.
    :returns: Complete scheduler registration.
    """

    binding, source_plan = _identity_graph(marker)
    owner = object()
    return PackedTerminalDecodeSchedulerRegistration(
        binding=binding,
        source_plan=source_plan,
        transaction=_transaction(binding.request_key, owner),
        request_owner=owner,
        adopt_request=adopt_request,
        finalize_request=finalize_request,
        cancel_request=cancel_request,
        quarantine_request=quarantine_request,
    )


def _terminal_dflash_adoption() -> TerminalDFlashDecodeAdoption:
    """Build a type-exact adoption envelope for scheduler-boundary tests.

    :returns: Exact outer adoption authority consumed by the scheduler.
    """

    return object.__new__(TerminalDFlashDecodeAdoption)


def _action(
    binding: TerminalRequestBinding,
    action_id: int = 1,
) -> NativeTerminalOwnerAction:
    """Build one exact owner-minted decode adoption action.

    :param binding: Decode lifecycle earning adoption authority.
    :param action_id: Stable one-shot action identity.
    :returns: Valid native adoption action.
    """

    native_binding = NativeTerminalRequestBinding.from_binding(binding)
    native_issuer = NativeTerminalProcessIdentity.from_identity(binding.owner)
    receipt = NativeTerminalReceipt(
        binding=native_binding,
        issuer=native_issuer,
        kind=NativeTerminalReceiptKind.ADOPTION_READY,
        outcome=NativeTerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=action_id,
        nonce=action_id.to_bytes(16, "big"),
    )
    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=NativeTerminalOwnerActionKind.ADOPTION_READY,
        binding=native_binding,
        commit_timestamp_ns=action_id,
        receipt=receipt,
    )


def _fatal_inventory(
    bindings: tuple[TerminalRequestBinding, ...],
) -> SchedulerReceiptInboxInventory:
    """Build sticky owner-death evidence for retained decode generations.

    :param bindings: Exact retained decode generations.
    :returns: Valid scheduler-inbox fatal inventory.
    """

    return SchedulerReceiptInboxInventory(
        physical_capacity=len(bindings),
        live_bindings=bindings,
        pending_request_keys=(),
        consuming_request_keys=(),
        outstanding_publications=0,
        active_delivery_intents=(),
        wake_armed=True,
        fatal_cause=SchedulerInboxFatalCause.OWNER_DEATH,
        closed=False,
    )


def test_composition_adopts_exact_request_before_local_ready() -> None:
    """Qualified inbox consumption preserves complete scheduler ordering."""

    scheduler_events: list[tuple[str, object]] = []

    def adopt(owner: object) -> TerminalDFlashDecodeAdoption:
        """Record adoption and return its exact typed authority.

        :param owner: Exact retained scheduler request.
        :returns: Exact DFlash adoption envelope.
        """

        scheduler_events.append(("adopt", owner))
        return _terminal_dflash_adoption()

    registration = _registration(
        adopt_request=adopt,
        finalize_request=lambda owner: scheduler_events.append(("finalize", owner)),
        cancel_request=lambda owner: None,
        quarantine_request=lambda owner, reason: None,
    )
    wiring = _DecodeWiring(registration.request_owner)
    composition = PackedTerminalDecodeServingComposition(
        wiring=wiring,
        physical_capacity=2,
        process_fatal_handler=lambda inventory: None,
        launch_gate=_LaunchGate(),
    )

    composition.register(registration)
    composition.scheduler_serving.publish_action(
        _action(registration.binding),
        publication_commit=lambda: None,
    )
    consumed = composition.scheduler_serving.drain_at_loop_entry()

    assert tuple(action.action_id for action in consumed) == (1,)
    assert wiring.events == [
        "allocation_adopted",
        "metadata_consumed",
        "local_ready",
    ]
    assert scheduler_events == [
        ("adopt", registration.request_owner),
        ("finalize", registration.request_owner),
    ]
    consumer_inventory, inbox_inventory = composition.inventory()
    assert consumer_inventory.active_binding_digests == ()
    assert consumer_inventory.adopted_count == 1
    assert inbox_inventory.live_count == 0
    assert inbox_inventory.pending_count == 0


def test_unpublished_cancellation_pairs_resources_and_inbox() -> None:
    """Safe rollback cannot leave either a request owner or live inbox key."""

    cancelled: list[object] = []
    registration = _registration(
        adopt_request=lambda owner: None,
        finalize_request=lambda owner: None,
        cancel_request=cancelled.append,
        quarantine_request=lambda owner, reason: None,
    )
    wiring = _DecodeWiring(registration.request_owner)
    composition = PackedTerminalDecodeServingComposition(
        wiring=wiring,
        physical_capacity=1,
        process_fatal_handler=lambda inventory: None,
        launch_gate=_LaunchGate(),
    )
    composition.register(registration)

    composition.cancel_unpublished(registration.binding, "gateway cancelled")

    assert wiring.cancelled_transactions == [registration.transaction]
    assert cancelled == [registration.request_owner]
    consumer_inventory, inbox_inventory = composition.inventory()
    assert consumer_inventory.active_binding_digests == ()
    assert consumer_inventory.cancelled_count == 1
    assert inbox_inventory.live_count == 0


def test_wrong_adoption_owner_quarantines_without_retry() -> None:
    """An identity-changing lower layer cannot install or release a request."""

    adopted: list[object] = []
    quarantined: list[tuple[object, str]] = []
    registration = _registration(
        adopt_request=adopted.append,
        finalize_request=lambda owner: None,
        cancel_request=lambda owner: None,
        quarantine_request=lambda owner, reason: quarantined.append((owner, reason)),
    )
    wiring = _DecodeWiring(
        registration.request_owner,
        return_another_owner=True,
    )
    consumer = PackedTerminalDecodeSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=lambda inventory: None,
    )
    consumer.register_adoption(registration)
    consumer.bind_adoption(registration.binding)
    action = _action(registration.binding)

    with pytest.raises(RuntimeError, match="another request owner"):
        consumer.consume_adoption_ready(action)

    inventory = consumer.inventory()
    assert adopted == []
    assert inventory.active_binding_digests == (registration.binding.digest,)
    assert inventory.quarantined_binding_digests == (registration.binding.digest,)
    assert wiring.quarantines == [
        (registration.transaction, "decode scheduler adoption failed")
    ]
    assert quarantined == [
        (registration.request_owner, "decode scheduler adoption failed")
    ]
    with pytest.raises(RuntimeError, match="quarantined decode resources"):
        consumer.consume_adoption_ready(action)


def test_process_fatal_quarantines_all_retained_requests_once() -> None:
    """Owner death retains every transaction and scheduler request exactly once."""

    quarantined: list[tuple[object, str]] = []
    registration = _registration(
        adopt_request=lambda owner: None,
        finalize_request=lambda owner: None,
        cancel_request=lambda owner: None,
        quarantine_request=lambda owner, reason: quarantined.append((owner, reason)),
    )
    wiring = _DecodeWiring(registration.request_owner)
    fatal_inventories: list[SchedulerReceiptInboxInventory] = []
    consumer = PackedTerminalDecodeSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=fatal_inventories.append,
    )
    consumer.register_adoption(registration)
    consumer.bind_adoption(registration.binding)
    fatal = _fatal_inventory((registration.binding,))

    consumer.process_fatal(fatal)
    consumer.process_fatal(fatal)

    assert fatal_inventories == [fatal]
    assert wiring.quarantines == [
        (registration.transaction, "decode scheduler inbox is process-fatal")
    ]
    assert quarantined == [
        (registration.request_owner, "decode scheduler inbox is process-fatal")
    ]
    inventory = consumer.inventory()
    assert inventory.quarantined_binding_digests == (registration.binding.digest,)
    assert inventory.fatal_inventory == fatal


def test_scheduler_request_never_crosses_thread_affinity() -> None:
    """Native authority cannot move mutable decode state to an owner thread."""

    registration = _registration(
        adopt_request=lambda owner: None,
        finalize_request=lambda owner: None,
        cancel_request=lambda owner: None,
        quarantine_request=lambda owner, reason: None,
    )
    wiring = _DecodeWiring(registration.request_owner)
    consumer = PackedTerminalDecodeSchedulerConsumer(
        wiring=wiring,
        process_fatal_handler=lambda inventory: None,
    )
    consumer.register_adoption(registration)
    consumer.bind_adoption(registration.binding)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            consumer.consume_adoption_ready,
            _action(registration.binding),
        )
        with pytest.raises(
            RuntimeError,
            match="decode scheduler resources crossed thread affinity",
        ):
            future.result()

    assert consumer.inventory().active_binding_digests == (registration.binding.digest,)
    assert wiring.events == []
