import concurrent.futures
import dataclasses
import selectors
import sys
import threading
import time
from unittest import mock

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerInventory,
    NativeTerminalOwnerObservation,
    NativeTerminalOwnerOutput,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalPublicationIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalActionClaimError,
    NativeTerminalActionInbox,
    NativeTerminalObservationInbox,
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeClosedError,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeError,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_OWNER_GENERATION = bytes.fromhex("10" * 16)
_REMOTE_GENERATION = bytes.fromhex("20" * 16)
_WAIT_SECONDS = 5.0
_LOCAL_PRODUCER_ID = 1
_OWNER_RECEIPT_PRODUCER_ID = 2
_REMOTE_RECEIPT_PRODUCER_ID = 3
_REMOTE_CONTROL_PRODUCER_ID = 4
_NATIVE_PRODUCER_ID = 5
_TEST_CLOCK_NS = 1_000_000_000


def _process_identity(
    role: TerminalOwnerRole, generation: bytes
) -> TerminalProcessIdentity:
    """Construct one exact process identity.

    :param role: Source or decode owner role.
    :param generation: Exact process generation.
    :returns: Canonical TP1 process identity.
    """

    return TerminalProcessIdentity(
        process_generation=generation,
        role=role,
        tp_rank=0,
        tp_size=1,
    )


def _runtime(
    role: TerminalOwnerRole,
    *,
    scheduler_capacity: int = 16,
    observation_capacity: int = 64,
    output_capacity: int = 64,
    maximum_live_lifecycles: int = 16,
    enable_forward_independent_handoff: bool = False,
) -> tuple[
    NativeTerminalRuntime,
    NativeTerminalProcessIdentity,
    NativeTerminalProcessIdentity,
]:
    """Construct one runtime with a complete frozen producer directory.

    :param role: Lifecycle role owned by the runtime.
    :param scheduler_capacity: Scheduler action bound.
    :param observation_capacity: Non-authoritative observation bound.
    :param output_capacity: Native normal-action queue bound.
    :param maximum_live_lifecycles: Admission and fail-closed reserve bound.
    :param enable_forward_independent_handoff: Whether this runtime exercises
        the CPython scheduler handoff.
    :returns: Runtime, owner identity, and remote peer identity.
    """

    owner = NativeTerminalProcessIdentity.from_identity(
        _process_identity(role, _OWNER_GENERATION)
    )
    remote_role = TerminalOwnerRole.DECODE
    if role is TerminalOwnerRole.DECODE:
        remote_role = TerminalOwnerRole.SOURCE
    remote = NativeTerminalProcessIdentity.from_identity(
        _process_identity(remote_role, _REMOTE_GENERATION)
    )
    native_role = NativeTerminalOwnerRole(int(owner.role))
    specs = (
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_LOCAL_PRODUCER_ID,
                name="python-local",
                producer_class=NativeTerminalProducerClass.LOCAL,
                allowed_role=native_role,
                authenticated_issuer=None,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_OWNER_RECEIPT_PRODUCER_ID,
                name="python-owner-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=native_role,
                authenticated_issuer=owner,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_REMOTE_RECEIPT_PRODUCER_ID,
                name="python-remote-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=native_role,
                authenticated_issuer=remote,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_REMOTE_CONTROL_PRODUCER_ID,
                name="python-remote-control",
                producer_class=NativeTerminalProducerClass.CONTROL,
                allowed_role=native_role,
                authenticated_issuer=remote,
            ),
            delivery=NativeTerminalProducerDelivery.PYTHON,
        ),
        NativeTerminalRuntimeProducerSpec(
            registration=NativeTerminalProducerRegistration(
                producer_id=_NATIVE_PRODUCER_ID,
                name="native-terminal",
                producer_class=NativeTerminalProducerClass.LOCAL,
                allowed_role=native_role,
                authenticated_issuer=None,
            ),
            delivery=NativeTerminalProducerDelivery.NATIVE,
        ),
    )
    runtime = NativeTerminalRuntime(
        owner_identity=owner,
        producer_specs=specs,
        fatal_producer_id=_LOCAL_PRODUCER_ID,
        input_capacity=64,
        output_capacity=output_capacity,
        maximum_live_lifecycles=maximum_live_lifecycles,
        scheduler_capacity=scheduler_capacity,
        coordinator_capacity=16,
        lifecycle_capacity=16,
        source_gather_capacity=16,
        source_work_capacity=16,
        decode_scatter_capacity=16,
        decode_work_capacity=16,
        publisher_capacity=16,
        observation_capacity=observation_capacity,
        enable_forward_independent_handoff=enable_forward_independent_handoff,
    )
    return runtime, owner, remote


def _registration(
    owner: NativeTerminalProcessIdentity,
    remote: NativeTerminalProcessIdentity,
    room_id: int,
) -> NativeTerminalLifecycleRegistration:
    """Construct one exact role-local lifecycle registration.

    :param owner: Runtime owner identity.
    :param remote: Authenticated peer identity.
    :param room_id: Stable request room identity.
    :returns: Complete native lifecycle registration.
    """

    request_key = PackedRequestKey(
        room_id=room_id,
        request_generation=room_id.to_bytes(16, "big"),
    )
    binding = TerminalRequestBinding(
        request_key=request_key,
        owner=TerminalProcessIdentity(
            process_generation=owner.process_generation,
            role=(
                TerminalOwnerRole.SOURCE
                if owner.role is NativeTerminalOwnerRole.SOURCE
                else TerminalOwnerRole.DECODE
            ),
            tp_rank=owner.tp_rank,
            tp_size=owner.tp_size,
        ),
        rank_manifest_digest=b"r" * 32,
        allocation_digest=room_id.to_bytes(32, "big"),
    )
    publication = None
    if owner.role is NativeTerminalOwnerRole.SOURCE:
        publication = NativeTerminalPublicationIdentity.from_identity(
            TerminalPublicationIdentity(
                request_key=request_key,
                publisher_process_generation=owner.process_generation,
                publication_generation=(room_id + 10_000).to_bytes(16, "big"),
            )
        )
    return NativeTerminalLifecycleRegistration(
        binding=NativeTerminalRequestBinding.from_binding(binding),
        publication_identity=publication,
        trusted_issuers=(owner, remote),
    )


def _direct_handoff_owner(
    room_id: int,
) -> tuple[NativeTerminalOwner, NativeTerminalLifecycleRegistration]:
    """Start a deterministic native owner with handoff enabled.

    :param room_id: Stable source request identity.
    :returns: Running owner and its registered source lifecycle.
    """

    owner_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.SOURCE, _OWNER_GENERATION)
    )
    remote_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.DECODE, _REMOTE_GENERATION)
    )
    registration = _registration(owner_identity, remote_identity, room_id)
    owner = NativeTerminalOwner(
        input_capacity=16,
        output_capacity=16,
        observation_capacity=16,
        maximum_live_lifecycles=4,
        owner_identity=owner_identity,
        testing=True,
    )
    owner.register_producer(
        NativeTerminalProducerRegistration(
            producer_id=_LOCAL_PRODUCER_ID,
            name="python-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=None,
        )
    )
    owner.enable_test_clock(_TEST_CLOCK_NS)
    owner.enable_forward_independent_handoff()
    owner.register_lifecycle(registration)
    owner.start()
    return owner, registration


def _submit_direct_source_gather(
    owner: NativeTerminalOwner,
    registration: NativeTerminalLifecycleRegistration,
) -> None:
    """Earn one source-gather action through the real native reducer.

    Event enqueue time ends at the frozen owner clock so native completion
    remains causally ordered without advancing the timeout test's clock.

    :param owner: Running deterministic source owner.
    :param registration: Exact source lifecycle.
    """

    for offset, kind in enumerate(
        (
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        )
    ):
        owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=_LOCAL_PRODUCER_ID,
                binding_digest=registration.binding.digest,
                kind=kind,
                enqueued_ns=_TEST_CLOCK_NS - 1 + offset,
            )
        )


def _receipt(
    registration: NativeTerminalLifecycleRegistration,
    issuer: NativeTerminalProcessIdentity,
    kind: NativeTerminalReceiptKind,
    nonce_value: int,
    *,
    outcome: NativeTerminalReceiptOutcome = NativeTerminalReceiptOutcome.SUCCESS,
) -> NativeTerminalReceipt:
    """Mint one deterministic route-authenticated test receipt.

    :param registration: Exact target lifecycle.
    :param issuer: Route-authenticated process identity.
    :param kind: Exact one-shot authority kind.
    :param nonce_value: Unique deterministic nonce integer.
    :param outcome: Success or failure outcome.
    :returns: Immutable native receipt.
    """

    return NativeTerminalReceipt(
        binding=registration.binding,
        issuer=issuer,
        kind=kind,
        outcome=outcome,
        terminal_timestamp_ns=nonce_value,
        nonce=nonce_value.to_bytes(16, "big"),
    )


def _drain_actions(
    inbox: NativeTerminalActionInbox,
) -> tuple[NativeTerminalOwnerAction, ...]:
    """Wait for and drain one fd-signalled action population.

    :param inbox: Exact runtime consumer inbox.
    :returns: Non-empty immutable FIFO population.
    """

    expires_at = time.monotonic() + _WAIT_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(inbox.fileno(), selectors.EVENT_READ)
        while True:
            remaining = expires_at - time.monotonic()
            if remaining <= 0.0 or len(selector.select(remaining)) == 0:
                raise TimeoutError("terminal runtime action inbox did not wake")
            actions = inbox.drain()
            if len(actions) > 0:
                return actions


def _drain_observations(runtime: NativeTerminalRuntime) -> None:
    """Drain every retained non-authoritative observation.

    :param runtime: Runtime being prepared for exact clean close.
    """

    assert runtime.wait_for_output_projection_quiescence(_WAIT_SECONDS)
    while True:
        if runtime.observations.snapshot().queued_count > 0:
            runtime.observations.drain()
        if runtime.observations.snapshot().queued_count == 0:
            return


def _retire_all_producers(runtime: NativeTerminalRuntime) -> None:
    """Retire every test producer through its owning delivery boundary.

    :param runtime: Draining runtime with no further producer work.
    """

    for producer_id in (
        _LOCAL_PRODUCER_ID,
        _OWNER_RECEIPT_PRODUCER_ID,
        _REMOTE_RECEIPT_PRODUCER_ID,
        _REMOTE_CONTROL_PRODUCER_ID,
    ):
        runtime.retire_python_producer(producer_id)
    runtime._owner.retire_python_producer(_NATIVE_PRODUCER_ID)


def _finish_fail_closed(runtime: NativeTerminalRuntime) -> None:
    """Drive the explicit abort drain used by exceptional test cleanup.

    :param runtime: Runtime whose producer and consumer authority must drain.
    """

    if runtime.snapshot().disposition is NativeTerminalRuntimeDisposition.STOPPED:
        return
    runtime.begin_abort()
    if not runtime.snapshot().producers_joined:
        _retire_all_producers(runtime)
        runtime.join_producers()
    action_inboxes = (
        runtime.scheduler_actions,
        runtime.coordinator_actions,
        runtime.lifecycle_actions,
        runtime.source_gather_actions,
        runtime.source_work_actions,
        runtime.decode_scatter_actions,
        runtime.decode_work_actions,
        runtime.publisher_actions,
    )
    expires_at = time.monotonic() + _WAIT_SECONDS
    with selectors.DefaultSelector() as selector:
        for inbox in action_inboxes:
            selector.register(inbox.fileno(), selectors.EVENT_READ, inbox)
        selector.register(
            runtime.observations.fileno(),
            selectors.EVENT_READ,
            runtime.observations,
        )
        while True:
            snapshot = runtime.snapshot()
            queued_count = sum(
                inbox.snapshot().queued_count for inbox in action_inboxes
            )
            queued_count += runtime.observations.snapshot().queued_count
            if (
                snapshot.owner.pending_action_count == 0
                and snapshot.consumer_pending_count == 0
                and queued_count == 0
            ):
                break
            remaining = expires_at - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("fail-closed runtime drain expired")
            ready = selector.select(remaining)
            if len(ready) == 0:
                raise TimeoutError("fail-closed runtime drain did not wake")
            for key, _ in ready:
                inbox = key.data
                if inbox is runtime.observations:
                    inbox.drain()
                    continue
                for action in inbox.drain():
                    runtime.acknowledge_aborted_action(action)
    runtime.finish_abort_close()


def _finish_handoff_runtime(runtime: NativeTerminalRuntime) -> None:
    """Drive fail-closed cleanup from a non-main consumer context.

    A pending-call callback can only run on the main interpreter thread. The
    cleanup worker therefore remains able to consume final quarantine actions
    even when the callback is queued while the main thread waits for it.

    :param runtime: Handoff-enabled runtime requiring exact cleanup.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_finish_fail_closed, runtime)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while not future.done() and time.monotonic() < expires_at:
            pass
        future.result(timeout=_WAIT_SECONDS)


def _complete_source(
    runtime: NativeTerminalRuntime,
    registration: NativeTerminalLifecycleRegistration,
    owner: NativeTerminalProcessIdentity,
    remote: NativeTerminalProcessIdentity,
) -> None:
    """Drive one source lifecycle through every off-loop continuation.

    :param runtime: Running source runtime.
    :param registration: Exact source lifecycle.
    :param owner: Source owner identity.
    :param remote: Decode peer identity.
    """

    scheduler, publisher = _emit_source_ready_actions(
        runtime,
        registration,
        remote,
        nonce_value=1,
    )
    runtime.complete_scheduler_action(
        _OWNER_RECEIPT_PRODUCER_ID,
        scheduler,
        NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
        completion_receipt=_receipt(
            registration,
            owner,
            NativeTerminalReceiptKind.RECLAIM_CONSUMED,
            2,
        ),
    )
    runtime.complete_work_action(
        _OWNER_RECEIPT_PRODUCER_ID,
        publisher,
        NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
        receipt=_receipt(
            registration,
            owner,
            NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
            3,
        ),
    )
    lifecycle = _drain_actions(runtime.lifecycle_actions)
    assert lifecycle[0].kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED
    runtime.acknowledge_consumed_action(lifecycle[0])


def _emit_source_ready_actions(
    runtime: NativeTerminalRuntime,
    registration: NativeTerminalLifecycleRegistration,
    remote: NativeTerminalProcessIdentity,
    *,
    nonce_value: int,
) -> tuple[NativeTerminalOwnerAction, NativeTerminalOwnerAction]:
    """Advance one source lifecycle to its scheduler and publisher actions.

    :param runtime: Running source runtime.
    :param registration: Exact source lifecycle.
    :param remote: Decode peer identity.
    :param nonce_value: Unique request-ready receipt nonce.
    :returns: Claimed reclaim and gateway-publication actions.
    """

    binding_digest = registration.binding.digest
    runtime.submit(
        _LOCAL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
    )
    runtime.submit(
        _LOCAL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
    )
    gather = _drain_actions(runtime.source_gather_actions)
    assert tuple(action.kind for action in gather) == (
        NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
    )
    runtime.complete_work_action(
        _LOCAL_PRODUCER_ID,
        gather[0],
        NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
    )
    runtime.submit(
        _LOCAL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
    )
    outcome = _drain_actions(runtime.source_work_actions)
    assert tuple(action.kind for action in outcome) == (
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
    )
    runtime.complete_work_action(
        _LOCAL_PRODUCER_ID,
        outcome[0],
        NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
    )
    runtime.submit(
        _REMOTE_CONTROL_PRODUCER_ID,
        binding_digest,
        NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
    )
    acknowledgement = _drain_actions(runtime.source_work_actions)
    assert tuple(action.kind for action in acknowledgement) == (
        NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
    )
    runtime.complete_work_action(
        _LOCAL_PRODUCER_ID,
        acknowledgement[0],
        NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
    )
    runtime.submit_imported_receipt(
        _REMOTE_RECEIPT_PRODUCER_ID,
        _receipt(
            registration,
            remote,
            NativeTerminalReceiptKind.REQUEST_READY,
            nonce_value,
        ),
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
    )
    scheduler_actions = _drain_actions(runtime.scheduler_actions)
    publisher_actions = _drain_actions(runtime.publisher_actions)
    assert len(scheduler_actions) == 1
    assert len(publisher_actions) == 1
    scheduler = scheduler_actions[0]
    publisher = publisher_actions[0]
    assert scheduler.kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED
    assert publisher.kind is (NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY)
    return scheduler, publisher


def test_pending_call_returns_at_inbox_claim_before_downstream_lock() -> None:
    """The callback ends at claim while downstream authority stays pending."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 61)
    runtime.start()
    previous_switch_interval = sys.getswitchinterval()
    scheduler_hot = threading.Event()
    scheduler_owned_lock = threading.Lock()
    inbox_claimed = threading.Event()
    downstream_entered = threading.Event()
    activation_observed = threading.Event()
    try:
        runtime.register_lifecycle(registration)

        original_activate = runtime._owner.activate_forward_independent_handoff

        def activate_after_route(action: NativeTerminalOwnerAction) -> bool:
            """Prove route publication and lock release precede activation.

            :param action: Exact action being enrolled in the scheduler handoff.
            :returns: Whether the native owner scheduled a pending callback.
            """

            assert runtime.source_gather_actions.snapshot().queued_count == 1
            assert not runtime._condition._is_owned()
            activation_observed.set()
            return original_activate(action)

        def consume_gather() -> NativeTerminalOwnerAction:
            assert runtime._owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            action = _drain_actions(runtime.source_gather_actions)[0]
            inbox_claimed.set()
            downstream_entered.set()
            if not scheduler_owned_lock.acquire(timeout=_WAIT_SECONDS):
                raise TimeoutError("scheduler-owned downstream lock was not released")
            scheduler_owned_lock.release()
            runtime.acknowledge_consumed_action(action)
            return action

        def submit_gather() -> None:
            assert scheduler_hot.wait(timeout=_WAIT_SECONDS)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )

        with mock.patch.object(
            runtime._owner,
            "activate_forward_independent_handoff",
            side_effect=activate_after_route,
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                scheduler_owned_lock.acquire()
                gather_future = executor.submit(consume_gather)
                producer_future = executor.submit(submit_gather)
                sys.setswitchinterval(0.005)
                scheduler_hot.set()
                expires_at = time.monotonic() + _WAIT_SECONDS
                scheduler_iterations = 0
                while not inbox_claimed.is_set() and time.monotonic() < expires_at:
                    scheduler_iterations += 1
                assert activation_observed.is_set()
                assert inbox_claimed.is_set()
                assert downstream_entered.is_set()
                assert not gather_future.done()
                snapshot = runtime.snapshot()
                assert snapshot.owner.unclaimed_handoff_action_count == 0
                assert snapshot.owner.claimed_handoff_action_count == 1
                assert snapshot.consumer_pending_count == 1
                scheduler_owned_lock.release()
                action = gather_future.result(timeout=_WAIT_SECONDS)
                producer_future.result(timeout=_WAIT_SECONDS)
                sys.setswitchinterval(previous_switch_interval)

            assert scheduler_iterations > 0
            inventory = runtime.snapshot().owner
            assert inventory.unclaimed_handoff_action_count == 0
            assert inventory.claimed_handoff_action_count == 1
            assert inventory.handoff_callback_count == 1

            def consume_process_fatal() -> tuple[NativeTerminalOwnerAction, ...]:
                actions = _drain_actions(runtime.lifecycle_actions)
                for current in actions:
                    runtime.acknowledge_aborted_action(current)
                return actions

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                fatal_future = executor.submit(consume_process_fatal)
                with pytest.raises(
                    NativeTerminalRuntimeError,
                    match="absent, stale, or already acknowledged",
                ):
                    runtime.acknowledge_consumed_action(action)
                fatal_actions = fatal_future.result(timeout=_WAIT_SECONDS)
            assert any(
                current.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
                for current in fatal_actions
            )
            assert (
                runtime.snapshot().owner.fatal_code
                is NativeTerminalOwnerFatalCode.HANDOFF_AUTHORITY
            )
    finally:
        if scheduler_owned_lock.locked():
            scheduler_owned_lock.release()
        sys.setswitchinterval(previous_switch_interval)
        _finish_handoff_runtime(runtime)


def test_active_handoff_coalesces_new_actions_into_one_later_watermark() -> None:
    """Actions arriving during a handoff cannot extend its captured watermark."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registrations = tuple(
        _registration(owner, remote, room_id) for room_id in (62, 63, 64)
    )
    runtime.start()
    previous_switch_interval = sys.getswitchinterval()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )

        def consume_all_gathers() -> tuple[NativeTerminalOwnerAction, ...]:
            assert runtime._owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            actions = list(_drain_actions(runtime.source_gather_actions))
            assert len(actions) == 1
            for registration in registrations[1:]:
                runtime.submit(
                    _LOCAL_PRODUCER_ID,
                    registration.binding.digest,
                    NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
                )
            runtime.acknowledge_consumed_action(actions[0])
            while len(actions) < len(registrations):
                current_actions = _drain_actions(runtime.source_gather_actions)
                for action in current_actions:
                    runtime.acknowledge_consumed_action(action)
                actions.extend(current_actions)
            return tuple(actions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            consumer_future = executor.submit(consume_all_gathers)
            sys.setswitchinterval(0.005)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registrations[0].binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )
            expires_at = time.monotonic() + 0.5
            while not consumer_future.done() and time.monotonic() < expires_at:
                pass
            actions = consumer_future.result(timeout=_WAIT_SECONDS)
            sys.setswitchinterval(previous_switch_interval)

        inventory = runtime.snapshot().owner
        assert len(actions) == 3
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == 3
        assert inventory.handoff_callback_count == 2
    finally:
        sys.setswitchinterval(previous_switch_interval)
        _finish_handoff_runtime(runtime)


def test_scheduler_actions_never_enter_the_forward_independent_handoff() -> None:
    """The source reclaim action stays scheduler-affine while peers hand off."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 65)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)

        def complete_and_close() -> object:
            _complete_source(runtime, registration, owner, remote)
            snapshot = runtime.snapshot()
            runtime.stop_admission()
            _retire_all_producers(runtime)
            runtime.join_producers()
            _drain_observations(runtime)
            runtime.close_clean()
            return snapshot

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            snapshot = executor.submit(complete_and_close).result(timeout=_WAIT_SECONDS)
        assert snapshot.owner.action_count == 6
        assert snapshot.owner.claimed_handoff_action_count == 5
        assert snapshot.owner.unclaimed_handoff_action_count == 0
    finally:
        _finish_handoff_runtime(runtime)


def test_inbox_claim_may_win_before_post_route_activation() -> None:
    """A consumer which wins publication makes activation a clean no-op."""

    owner, registration = _direct_handoff_owner(68)
    try:
        _submit_direct_source_gather(owner, registration)
        with selectors.DefaultSelector() as selector:
            selector.register(owner.output_fileno(), selectors.EVENT_READ)
            assert len(selector.select(_WAIT_SECONDS)) == 1
        actions = tuple(
            action for output in owner.drain_outputs() for action in output.actions
        )
        assert len(actions) == 1
        action = actions[0]
        before_claim = owner.inventory()
        assert before_claim.unclaimed_handoff_action_count == 1
        assert before_claim.claimed_handoff_action_count == 0
        assert not before_claim.handoff_callback_scheduled

        owner.claim_forward_independent_handoff(action)
        assert not owner.activate_forward_independent_handoff(action)
        owner.acknowledge_action(action)

        after_activation = owner.inventory()
        assert after_activation.unclaimed_handoff_action_count == 0
        assert after_activation.claimed_handoff_action_count == 1
        assert after_activation.discarded_handoff_action_count == 0
        assert after_activation.handoff_callback_count == 0
        assert not after_activation.handoff_callback_scheduled
        assert not after_activation.handoff_callback_active
    finally:
        owner.abort_and_close()


def test_delivery_failure_discards_unclaimed_handoff_authority() -> None:
    """Rejected inbox delivery has distinct, conserved handoff accounting."""

    owner, registration = _direct_handoff_owner(70)
    try:
        _submit_direct_source_gather(owner, registration)
        with selectors.DefaultSelector() as selector:
            selector.register(owner.output_fileno(), selectors.EVENT_READ)
            assert len(selector.select(_WAIT_SECONDS)) == 1
        actions = tuple(
            action for output in owner.drain_outputs() for action in output.actions
        )
        assert len(actions) == 1
        owner.fail_action_delivery(actions[0], "synthetic bounded-inbox rejection")

        with selectors.DefaultSelector() as selector:
            selector.register(owner.output_fileno(), selectors.EVENT_READ)
            assert len(selector.select(_WAIT_SECONDS)) == 1
        fatal_actions = tuple(
            action for output in owner.drain_outputs() for action in output.actions
        )
        assert tuple(action.kind for action in fatal_actions) == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        for action in fatal_actions:
            owner.acknowledge_action(action)

        inventory = owner.inventory()
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == 0
        assert inventory.discarded_handoff_action_count == 1
        assert (
            inventory.fatal_code is NativeTerminalOwnerFatalCode.OUTPUT_QUEUE_OVERFLOW
        )
    finally:
        owner.abort_and_close()


def test_post_claim_abort_preserves_downstream_authority_without_replay() -> None:
    """Abort drains a claimed action without claiming native authority twice."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 69)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)

        def claim_gather() -> NativeTerminalOwnerAction:
            assert runtime._owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            return _drain_actions(runtime.source_gather_actions)[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            claim_future = executor.submit(claim_gather)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not claim_future.done() and time.monotonic() < expires_at:
                pass
            gather = claim_future.result(timeout=_WAIT_SECONDS)

        claimed = runtime.snapshot()
        assert claimed.owner.unclaimed_handoff_action_count == 0
        assert claimed.owner.claimed_handoff_action_count == 1
        assert claimed.consumer_pending_count == 1

        runtime.begin_abort("synthetic downstream failure after inbox claim")
        _retire_all_producers(runtime)
        runtime.join_producers()
        fatal_actions = _drain_actions(runtime.lifecycle_actions)
        assert tuple(action.kind for action in fatal_actions) == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        before_acknowledgement = runtime.snapshot()
        assert before_acknowledgement.owner.claimed_handoff_action_count == 1
        assert before_acknowledgement.owner.unclaimed_handoff_action_count == 0
        assert before_acknowledgement.consumer_pending_count == 2

        runtime.acknowledge_aborted_action(gather)
        assert not runtime.acknowledge_aborted_action_if_pending(gather)
        for action in fatal_actions:
            runtime.acknowledge_aborted_action(action)

        after_acknowledgement = runtime.snapshot()
        assert after_acknowledgement.owner.claimed_handoff_action_count == 1
        assert after_acknowledgement.owner.unclaimed_handoff_action_count == 0
        assert after_acknowledgement.consumer_pending_count == 0
        _drain_observations(runtime)
        runtime.finish_abort_close()
        stopped = runtime.snapshot()
        assert stopped.disposition is NativeTerminalRuntimeDisposition.STOPPED
        assert stopped.consumer_pending_count == 0
    finally:
        _finish_handoff_runtime(runtime)


def test_failed_claim_excludes_preclaimed_duplicate_from_local_authority() -> None:
    """A stale replay cannot transfer another downstream owner's authority."""

    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registrations = (
        _registration(owner, remote, 70),
        _registration(owner, remote, 71),
    )
    runtime.start()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )

        expires_at = time.monotonic() + _WAIT_SECONDS
        while runtime.source_gather_actions.snapshot().queued_count != 2:
            if time.monotonic() >= expires_at:
                raise TimeoutError("source gather population did not settle")
            time.sleep(0.001)

        already_claimed = runtime.source_gather_actions.drain(maximum_items=1)[0]
        runtime.source_gather_actions._enqueue(already_claimed)

        with pytest.raises(NativeTerminalActionClaimError) as raised:
            runtime.source_gather_actions.drain()

        error = raised.value
        assert len(error.removed_actions) == 2
        assert error.removed_actions[-1] == already_claimed
        assert error.locally_claimed_actions == (error.removed_actions[0],)
        assert already_claimed not in error.locally_claimed_actions
        snapshot = runtime.snapshot()
        assert snapshot.consumer_pending_count == 2
        with runtime._condition:
            assert runtime._inbox_claimed_action_ids == {
                action.action_id for action in error.removed_actions
            }

        runtime.acknowledge_consumed_action(already_claimed)
        runtime.acknowledge_consumed_action(error.locally_claimed_actions[0])
    finally:
        _finish_fail_closed(runtime)


def test_handoff_timeout_uses_the_hash_bound_owner_shutdown_deadline() -> None:
    """An unconsumed watermark expires into native process-fatal authority."""

    owner, registration = _direct_handoff_owner(66)
    previous_switch_interval = sys.getswitchinterval()
    output_queued = threading.Event()
    try:

        def expire_active_handoff() -> tuple[NativeTerminalOwnerAction, ...]:
            with selectors.DefaultSelector() as selector:
                selector.register(owner.output_fileno(), selectors.EVENT_READ)
                assert len(selector.select(_WAIT_SECONDS)) == 1
            initial_actions = tuple(
                action for output in owner.drain_outputs() for action in output.actions
            )
            assert len(initial_actions) == 1
            before_activation = owner.inventory()
            assert before_activation.unclaimed_handoff_action_count == 1
            assert not before_activation.handoff_callback_scheduled
            assert not before_activation.handoff_callback_active
            assert owner.activate_forward_independent_handoff(initial_actions[0])
            owner.acknowledge_action(initial_actions[0])
            output_queued.set()
            assert owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            owner.set_test_clock(_TEST_CLOCK_NS + 120_000_000_000)
            assert owner.wait_for_process_fatal(_WAIT_SECONDS)
            fatal_actions = tuple(
                action for output in owner.drain_outputs() for action in output.actions
            )
            for action in fatal_actions:
                owner.acknowledge_action(action)
            owner.claim_forward_independent_handoff(initial_actions[0])
            return (*initial_actions, *fatal_actions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            expiry_future = executor.submit(expire_active_handoff)
            sys.setswitchinterval(0.005)
            _submit_direct_source_gather(owner, registration)
            assert output_queued.wait(_WAIT_SECONDS)
            actions = expiry_future.result(timeout=_WAIT_SECONDS)
            sys.setswitchinterval(previous_switch_interval)

        inventory = owner.inventory()
        assert len(actions) == 2
        assert any(
            action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
            for action in actions
        )
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.HANDOFF_TIMEOUT
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == 1
        assert inventory.handoff_callback_count == 1
    finally:
        sys.setswitchinterval(previous_switch_interval)
        owner.abort_and_close()


def test_close_with_a_pending_handoff_fails_closed_before_release() -> None:
    """Clean close cannot discard unclaimed scheduler-handoff authority."""

    owner, registration = _direct_handoff_owner(67)
    previous_switch_interval = sys.getswitchinterval()
    output_queued = threading.Event()
    try:

        def reject_close_and_resolve_actions() -> str:
            with selectors.DefaultSelector() as selector:
                selector.register(owner.output_fileno(), selectors.EVENT_READ)
                assert len(selector.select(_WAIT_SECONDS)) == 1
            initial_actions = tuple(
                action for output in owner.drain_outputs() for action in output.actions
            )
            assert len(initial_actions) == 1
            assert owner.activate_forward_independent_handoff(initial_actions[0])
            owner.acknowledge_action(initial_actions[0])
            output_queued.set()
            assert owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            with pytest.raises(RuntimeError) as error:
                owner.close()
            owner.claim_forward_independent_handoff(initial_actions[0])
            for output in owner.drain_outputs():
                for action in output.actions:
                    owner.acknowledge_action(action)
            return str(error.value)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            close_future = executor.submit(reject_close_and_resolve_actions)
            sys.setswitchinterval(0.005)
            _submit_direct_source_gather(owner, registration)
            assert output_queued.wait(_WAIT_SECONDS)
            close_error = close_future.result(timeout=_WAIT_SECONDS)
            sys.setswitchinterval(previous_switch_interval)

        assert "retained unresolved inventory" in close_error
        inventory = owner.inventory()
        assert inventory.fatal_code is (
            NativeTerminalOwnerFatalCode.CLOSE_WITH_RETAINED_INVENTORY
        )
        assert inventory.unclaimed_handoff_action_count == 0
    finally:
        sys.setswitchinterval(previous_switch_interval)
        owner.abort_and_close()


def test_runtime_routes_source_continuations_and_drains_after_admission_close() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE, observation_capacity=1)
    registration = _registration(owner, remote, 71)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        runtime.stop_admission()
        with pytest.raises(NativeTerminalRuntimeClosedError):
            runtime.register_lifecycle(_registration(owner, remote, 72))

        _complete_source(runtime, registration, owner, remote)

        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.DRAINING
        assert snapshot.owner.admission_open is False
        assert snapshot.owner.event_admission_open
        assert snapshot.scheduler_live_count == 0
        assert snapshot.scheduler_pending_count == 0
        assert snapshot.consumer_pending_count == 0
        assert snapshot.dropped_observation_count > 0
        _retire_all_producers(runtime)
        runtime.join_producers()
        with pytest.raises(NativeTerminalRuntimeClosedError):
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_OWNER_DIED,
                reason="producer submitted after join",
            )
        runtime.close_clean()
        closed = runtime.snapshot()
        assert closed.observations.closed
        assert closed.observations.queued_count == 0
        assert closed.dropped_observation_count > 0
    finally:
        _finish_fail_closed(runtime)


def test_python_producer_retirement_resumes_after_mid_roster_failure() -> None:
    """Completed retirements remain exact while the remaining roster resumes."""

    runtime, _, _ = _runtime(TerminalOwnerRole.SOURCE)
    runtime.start()
    try:
        runtime.stop_admission()
        producer_ids = runtime.python_producer_ids
        assert runtime.unretired_python_producer_ids == producer_ids
        failed_producer_id = producer_ids[1]
        original_retire = runtime._owner.retire_python_producer
        attempted_ids: list[int] = []

        def retire_until_failure(producer_id: int) -> None:
            """Retire the first producer, then expose one synthetic boundary failure.

            :param producer_id: Exact producer selected by the runtime roster.
            """

            attempted_ids.append(producer_id)
            if producer_id == failed_producer_id:
                raise RuntimeError("synthetic producer retirement failure")
            original_retire(producer_id)

        with mock.patch.object(
            runtime._owner,
            "retire_python_producer",
            side_effect=retire_until_failure,
        ):
            with pytest.raises(
                RuntimeError,
                match="synthetic producer retirement failure",
            ):
                for producer_id in runtime.unretired_python_producer_ids:
                    runtime.retire_python_producer(producer_id)

        assert attempted_ids == [producer_ids[0], failed_producer_id]
        assert runtime.unretired_python_producer_ids == producer_ids[1:]
        with pytest.raises(
            NativeTerminalRuntimeError,
            match="producer was already retired",
        ):
            runtime.retire_python_producer(producer_ids[0])

        for producer_id in runtime.unretired_python_producer_ids:
            runtime.retire_python_producer(producer_id)
        assert runtime.unretired_python_producer_ids == ()
        runtime._owner.retire_python_producer(_NATIVE_PRODUCER_ID)
        runtime.join_producers()
        runtime.close_clean()
    finally:
        _finish_fail_closed(runtime)


def test_observation_sink_failure_isolated_per_record() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registrations = (
        _registration(owner, remote, 711),
        _registration(owner, remote, 712),
    )
    observations = tuple(
        NativeTerminalOwnerObservation(
            binding=registration.binding,
            owner_sequence=index,
            producer_id=_LOCAL_PRODUCER_ID,
            producer_sequence=index,
            producer_rank=owner.tp_rank,
            event_kind=NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            enqueued_ns=1_000 + index,
            completed_ns=2_000 + index,
            role=owner.role,
        )
        for index, registration in enumerate(registrations)
    )
    observation_type = type(runtime.observations)
    original_enqueue = observation_type._enqueue
    attempts = 0

    def fail_first(
        inbox: NativeTerminalObservationInbox,
        observation: NativeTerminalOwnerObservation,
    ) -> None:
        """Reject one record, then delegate every later record.

        :param inbox: Runtime observation inbox.
        :param observation: Candidate immutable evidence record.
        """

        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("synthetic evidence sink failure")
        original_enqueue(inbox, observation)

    runtime.start()
    try:
        with mock.patch.object(observation_type, "_enqueue", fail_first):
            for observation in observations:
                runtime._route_observation(observation)

        assert runtime.observations.drain() == (observations[1],)
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.RUNNING
        assert snapshot.dropped_observation_count == 1
        assert snapshot.owner.fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        _finish_fail_closed(runtime)


def test_native_quiescence_waits_for_complete_output_projection() -> None:
    """Native delivery stays retained through the non-authoritative projection."""

    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registration = _registration(owner, remote, 72)
    projection_started = threading.Event()
    release_projection = threading.Event()
    observation_type = type(runtime.observations)
    original_enqueue = observation_type._enqueue

    def delayed_enqueue(
        inbox: NativeTerminalObservationInbox,
        output: NativeTerminalOwnerOutput,
    ) -> None:
        if any(
            action.kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED
            for action in output.actions
        ):
            projection_started.set()
            if not release_projection.wait(_WAIT_SECONDS):
                raise TimeoutError("terminal output projection was not released")
        original_enqueue(inbox, output)

    with mock.patch.object(observation_type, "_enqueue", delayed_enqueue):
        runtime.start()
        try:
            runtime.register_lifecycle(registration)
            runtime.stop_admission()
            _complete_source(runtime, registration, owner, remote)

            assert projection_started.wait(_WAIT_SECONDS)
            assert runtime.snapshot().owner.pending_action_count == 1

            release_projection.set()
            _retire_all_producers(runtime)
            runtime.join_producers()
            _drain_observations(runtime)
            runtime.close_clean()
        finally:
            release_projection.set()
            _finish_fail_closed(runtime)


def test_decode_local_ready_claim_allows_nested_request_ready_before_ack() -> None:
    """The exact TP1 local fanout chain cannot extend its active callback."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 75)
    binding_digest = registration.binding.digest
    runtime.start()
    try:

        def drive_tp1_chain() -> NativeTerminalOwnerInventory:
            runtime.register_lifecycle(registration)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED,
            )
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
            )
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
            )
            scatter = _drain_actions(runtime.decode_scatter_actions)
            assert scatter[0].kind is NativeTerminalOwnerActionKind.DECODE_SCATTER_READY
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                scatter[0],
                NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_SCATTER_TERMINAL,
            )
            teardown = _drain_actions(runtime.decode_work_actions)
            assert (
                teardown[0].kind is NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY
            )
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                teardown[0],
                NativeTerminalOwnerEventKind.DECODE_TEARDOWN_SENT,
            )
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
            )
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
            )
            adoption = _drain_actions(runtime.scheduler_actions)
            assert adoption[0].kind is NativeTerminalOwnerActionKind.ADOPTION_READY
            runtime.complete_scheduler_action(
                _OWNER_RECEIPT_PRODUCER_ID,
                adoption[0],
                NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_METADATA_CONSUMED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.DECODE_LOCAL_READY_ISSUED,
            )
            local_ready = _drain_actions(runtime.coordinator_actions)
            assert (
                local_ready[0].kind is NativeTerminalOwnerActionKind.LOCAL_DECODE_READY
            )
            runtime.submit_imported_receipt(
                _OWNER_RECEIPT_PRODUCER_ID,
                _receipt(
                    registration,
                    owner,
                    NativeTerminalReceiptKind.REQUEST_READY,
                    50,
                ),
                NativeTerminalOwnerEventKind.DECODE_REQUEST_READY,
            )
            runtime.acknowledge_consumed_action(local_ready[0])
            lifecycle = _drain_actions(runtime.lifecycle_actions)
            assert lifecycle[0].kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED
            runtime.acknowledge_consumed_action(lifecycle[0])
            return runtime.snapshot().owner

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(drive_tp1_chain)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not future.done() and time.monotonic() < expires_at:
                pass
            owner_inventory = future.result(timeout=_WAIT_SECONDS)

        assert owner_inventory.unclaimed_handoff_action_count == 0
        assert owner_inventory.claimed_handoff_action_count == 4
        assert owner_inventory.handoff_callback_count >= 2
        snapshot = runtime.snapshot()
        assert snapshot.scheduler_live_count == 0
        assert snapshot.scheduler_pending_count == 0
        assert snapshot.consumer_pending_count == 0
        runtime.stop_admission()
        _retire_all_producers(runtime)
        runtime.join_producers()
        _drain_observations(runtime)
        runtime.close_clean()
    finally:
        _finish_handoff_runtime(runtime)


def test_runtime_producer_directory_prevents_dynamic_or_cross_domain_authority() -> (
    None
):
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    runtime.start()
    try:
        assert (
            runtime.python_producer_id(NativeTerminalProducerClass.LOCAL)
            == _LOCAL_PRODUCER_ID
        )
        assert (
            runtime.python_producer_id(NativeTerminalProducerClass.CONTROL, remote)
            == _REMOTE_CONTROL_PRODUCER_ID
        )
        assert (
            runtime.python_producer_id(NativeTerminalProducerClass.RECEIPT, owner)
            == _OWNER_RECEIPT_PRODUCER_ID
        )
        with pytest.raises(KeyError, match="not registered"):
            runtime.python_producer_id(NativeTerminalProducerClass.CONTROL, owner)
        binding = runtime.native_producer_binding("native-terminal")
        assert binding.producer_id == _NATIVE_PRODUCER_ID
        with pytest.raises(NativeTerminalRuntimeError, match="Python-owned"):
            runtime.native_producer_binding("python-local")
        with pytest.raises(NativeTerminalRuntimeError, match="native producer"):
            runtime.submit(
                binding.producer_id,
                b"b" * 32,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
        with pytest.raises(ValueError, match="control event"):
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                b"b" * 32,
                NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
            )
        with pytest.raises(ValueError, match="local lifecycle"):
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                b"b" * 32,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
        runtime.stop_admission()
        _retire_all_producers(runtime)
        runtime.join_producers()
        runtime.close_clean()
    finally:
        _finish_fail_closed(runtime)


def test_runtime_queue_overflow_enters_one_process_fatal_path() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE, scheduler_capacity=1)
    first = _registration(owner, remote, 81)
    second = _registration(owner, remote, 82)
    runtime.start()
    try:
        runtime.register_lifecycle(first)
        runtime.register_lifecycle(second)
        for registration, nonce_base in ((first, 100), (second, 200)):
            binding_digest = registration.binding.digest
            for kind in (
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            ):
                runtime.submit(_LOCAL_PRODUCER_ID, binding_digest, kind)
            gather = _drain_actions(runtime.source_gather_actions)[0]
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                gather,
                NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
            )
            outcome = _drain_actions(runtime.source_work_actions)[0]
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                outcome,
                NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
            )
            runtime.submit(
                _REMOTE_CONTROL_PRODUCER_ID,
                binding_digest,
                NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
            )
            acknowledgement = _drain_actions(runtime.source_work_actions)[0]
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                acknowledgement,
                NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
            )
            runtime.submit_imported_receipt(
                _REMOTE_RECEIPT_PRODUCER_ID,
                _receipt(
                    registration,
                    remote,
                    NativeTerminalReceiptKind.REQUEST_READY,
                    nonce_base,
                ),
                NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
            )

        with selectors.DefaultSelector() as selector:
            selector.register(runtime.lifecycle_actions.fileno(), selectors.EVENT_READ)
            assert len(selector.select(_WAIT_SECONDS)) > 0
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert snapshot.fatal_reason is not None
        assert "scheduler" in snapshot.fatal_reason
        assert snapshot.output_reactor_alive
    finally:
        _finish_fail_closed(runtime)


def test_scheduler_failure_releases_action_accounting_and_retains_quarantine() -> None:
    """A failed reclaim cannot leak pending authority or imply safe cleanup."""

    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registration = _registration(owner, remote, 89)
    binding_digest = registration.binding.digest
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        )
        gather = _drain_actions(runtime.source_gather_actions)[0]
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            gather,
            NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
        )
        outcome = _drain_actions(runtime.source_work_actions)[0]
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            outcome,
            NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
        )
        runtime.submit(
            _REMOTE_CONTROL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        )
        acknowledgement = _drain_actions(runtime.source_work_actions)[0]
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            acknowledgement,
            NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
        )
        runtime.submit_imported_receipt(
            _REMOTE_RECEIPT_PRODUCER_ID,
            _receipt(
                registration,
                remote,
                NativeTerminalReceiptKind.REQUEST_READY,
                890,
            ),
            NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
        )
        reclaim = _drain_actions(runtime.scheduler_actions)[0]
        assert reclaim.kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED
        runtime.fail_scheduler_action(reclaim, "synthetic scheduler cleanup failure")

        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert snapshot.scheduler_pending_count == 0
        assert snapshot.consumer_pending_count >= 1
        lifecycle = _drain_actions(runtime.lifecycle_actions)
        quarantine = tuple(
            action
            for action in lifecycle
            if action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED
        )
        assert len(quarantine) == 1
        runtime.acknowledge_consumed_action(quarantine[0])
        assert runtime.snapshot().quarantined_binding_digests == (binding_digest,)
    finally:
        _finish_fail_closed(runtime)


def test_abort_preserves_every_registration_and_quarantine_identity() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registrations = (
        _registration(owner, remote, 91),
        _registration(owner, remote, 92),
    )
    runtime.start()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
        runtime.begin_abort()
        _retire_all_producers(runtime)
        runtime.join_producers()

        actions = _drain_actions(runtime.lifecycle_actions)
        assert len(actions) == len(registrations)
        assert all(
            action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
            for action in actions
        )
        assert {action.binding.digest for action in actions} == {
            registration.binding.digest for registration in registrations
        }
        for action in actions:
            runtime.acknowledge_aborted_action(action)
        snapshot = runtime.snapshot()
        assert snapshot.scheduler_live_count == 0
        assert set(snapshot.quarantined_binding_digests) == {
            registration.binding.digest for registration in registrations
        }
        assert set(snapshot.owner.quarantined_binding_digests) == set(
            snapshot.quarantined_binding_digests
        )
        runtime.finish_abort_close()
        closed = runtime.snapshot()
        assert closed.observations.closed
        assert closed.observations.queued_count == 0
        assert closed.dropped_observation_count > 0
    finally:
        _finish_fail_closed(runtime)


def test_fail_closed_surrender_is_atomic_and_cannot_replay() -> None:
    """A closure mismatch removes no authority and a valid closure is take-once."""

    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registration = _registration(owner, remote, 921)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        scheduler, publisher = _emit_source_ready_actions(
            runtime,
            registration,
            remote,
            nonce_value=921,
        )
        runtime.begin_abort("synthetic downstream closure")
        expected_pending = {
            scheduler.action_id: scheduler,
            publisher.action_id: publisher,
        }
        with pytest.raises(
            NativeTerminalRuntimeError,
            match="fail-closed surrender requires joined producers",
        ):
            runtime.surrender_fail_closed_actions((scheduler, publisher))
        with runtime._condition:
            assert runtime._consumer_pending == expected_pending
            assert runtime._inbox_claimed_action_ids == set(expected_pending)
            assert runtime._scheduler_pending == {
                registration.binding.digest: scheduler
            }

        _retire_all_producers(runtime)
        runtime.join_producers()
        lifecycle_actions = _drain_actions(runtime.lifecycle_actions)
        assert all(
            action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
            for action in lifecycle_actions
        )
        for action in lifecycle_actions:
            runtime.acknowledge_aborted_action(action)
        assert runtime.wait_for_output_projection_quiescence(_WAIT_SECONDS)

        with pytest.raises(
            NativeTerminalRuntimeError,
            match="closure differs from pending consumer authority",
        ):
            runtime.surrender_fail_closed_actions((scheduler,))
        with runtime._condition:
            assert runtime._consumer_pending == expected_pending
            assert runtime._inbox_claimed_action_ids == set(expected_pending)
            assert runtime._scheduler_pending == {
                registration.binding.digest: scheduler
            }

        stale_publisher = dataclasses.replace(
            publisher,
            commit_timestamp_ns=publisher.commit_timestamp_ns + 1,
        )
        with pytest.raises(
            NativeTerminalRuntimeError,
            match="aliases pending consumer authority",
        ):
            runtime.surrender_fail_closed_actions((scheduler, stale_publisher))
        with runtime._condition:
            assert runtime._consumer_pending == expected_pending
            assert runtime._inbox_claimed_action_ids == set(expected_pending)
            assert runtime._scheduler_pending == {
                registration.binding.digest: scheduler
            }

        surrender = runtime.surrender_fail_closed_actions((scheduler, publisher))
        assert surrender.action_ids == (scheduler.action_id, publisher.action_id)
        assert surrender.binding_digests == (
            registration.binding.digest,
            registration.binding.digest,
        )
        assert runtime.snapshot().consumer_pending_count == 0
        assert runtime.snapshot().scheduler_pending_count == 0
        assert runtime.snapshot().scheduler_live_count == 0

        with pytest.raises(
            NativeTerminalRuntimeError,
            match="closure differs from pending consumer authority",
        ):
            runtime.surrender_fail_closed_actions((scheduler, publisher))
        assert runtime.snapshot().consumer_pending_count == 0

        _drain_observations(runtime)
        runtime.finish_abort_close()
    finally:
        _finish_fail_closed(runtime)


def test_close_waits_for_swapped_output_before_rejecting_consumer_inventory() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.SOURCE)
    registration = _registration(owner, remote, 93)
    route_entered = threading.Event()
    release_route = threading.Event()
    original_route = NativeTerminalRuntime._route_output

    def blocked_route(
        candidate: NativeTerminalRuntime,
        output: object,
    ) -> None:
        """Hold one swapped native batch outside its queue inventory.

        :param candidate: Runtime whose sole consumer owns the batch.
        :param output: Typed native output pending Python routing.
        """

        route_entered.set()
        if not release_route.wait(_WAIT_SECONDS):
            raise TimeoutError("test output-route barrier expired")
        original_route(candidate, output)  # type: ignore[arg-type]

    runtime.start()
    try:
        with mock.patch.object(NativeTerminalRuntime, "_route_output", blocked_route):
            runtime.register_lifecycle(registration)
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )
            assert route_entered.wait(_WAIT_SECONDS)
            runtime.stop_admission()
            _retire_all_producers(runtime)
            runtime.join_producers()

            close_errors: list[BaseException] = []

            def close_runtime() -> None:
                """Attempt clean close while one native batch is swapped."""

                try:
                    runtime.close_clean()
                except NativeTerminalRuntimeError as error:
                    close_errors.append(error)

            close_thread = threading.Thread(target=close_runtime)
            close_thread.start()
            close_thread.join(timeout=0.05)
            assert close_thread.is_alive()
            release_route.set()
            close_thread.join(timeout=_WAIT_SECONDS)
            assert not close_thread.is_alive()
            assert len(close_errors) == 1
            assert "consumer" in str(close_errors[0])
            snapshot = runtime.snapshot()
            assert snapshot.output_reactor_alive
            assert snapshot.disposition is NativeTerminalRuntimeDisposition.DRAINING
    finally:
        release_route.set()
        _finish_fail_closed(runtime)


def test_fatal_reserve_survives_saturated_normal_output_queue() -> None:
    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        output_capacity=1,
        maximum_live_lifecycles=3,
    )
    registrations = tuple(
        _registration(owner, remote, room_id) for room_id in range(94, 97)
    )
    drain_entered = threading.Event()
    release_drain = threading.Event()
    original_drain = runtime._owner.drain_outputs

    def blocked_drain() -> tuple[object, ...]:
        """Hold the sole consumer before it swaps the saturated queues.

        :returns: Native outputs after the deterministic barrier releases.
        """

        drain_entered.set()
        if not release_drain.wait(_WAIT_SECONDS):
            raise TimeoutError("test native-drain barrier expired")
        return original_drain()

    runtime._owner.drain_outputs = blocked_drain  # type: ignore[method-assign]
    runtime.start()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
        for registration in registrations[:2]:
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
            )
        assert drain_entered.wait(_WAIT_SECONDS)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while True:
            inventory = runtime.snapshot().owner
            if (
                inventory.fatal_code
                is NativeTerminalOwnerFatalCode.OUTPUT_QUEUE_OVERFLOW
            ):
                break
            if time.monotonic() >= expires_at:
                raise TimeoutError("native fatal reserve did not settle")
            time.sleep(0.001)
        assert inventory.queued_output_count == 4
        assert inventory.queued_fatal_output_count == 3
        assert inventory.pending_action_count == 4
        assert inventory.fatal_reason == "production action queue overflowed"

        release_drain.set()
        with selectors.DefaultSelector() as selector:
            selector.register(runtime.lifecycle_actions.fileno(), selectors.EVENT_READ)
            assert len(selector.select(_WAIT_SECONDS)) > 0
        actions = _drain_actions(runtime.lifecycle_actions)
        assert len(actions) == len(registrations)
        assert all(
            action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
            for action in actions
        )
        for action in actions:
            runtime.acknowledge_aborted_action(action)
        runtime.begin_abort()
        _retire_all_producers(runtime)
        runtime.join_producers()
        _finish_fail_closed(runtime)
    finally:
        release_drain.set()
        _finish_fail_closed(runtime)
