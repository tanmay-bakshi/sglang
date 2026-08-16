import concurrent.futures
import dataclasses
import selectors
import sys
import threading
import time
from collections.abc import Callable
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
    NativeTerminalHandoffCallbackTerminalState,
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
    NativeTerminalDeliveryLeaseDisposition,
    NativeTerminalObservationInbox,
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeClosedError,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeError,
    NativeTerminalRuntimeProducerSpec,
    NativeTerminalSourceDeliveryAuthority,
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


def _acquire_test_source_delivery(
    actions: tuple[NativeTerminalOwnerAction, ...],
) -> NativeTerminalDeliveryLeaseDisposition:
    """Return normal delivery authority for a fixture action population.

    :param actions: Exact action population crossing native ownership.
    :returns: Normal acquired delivery disposition.
    """

    assert type(actions) is tuple
    return NativeTerminalDeliveryLeaseDisposition.ACQUIRED


class _TestSourceDeliveryAuthority(NativeTerminalSourceDeliveryAuthority):
    """Configurable atomic source delivery authority for runtime tests."""

    _acquire: Callable[
        [tuple[NativeTerminalOwnerAction, ...]],
        NativeTerminalDeliveryLeaseDisposition,
    ]

    def __init__(
        self,
        acquire: Callable[
            [tuple[NativeTerminalOwnerAction, ...]],
            NativeTerminalDeliveryLeaseDisposition,
        ]
        | None = None,
    ) -> None:
        """Construct one authority with an optional acquisition observer.

        :param acquire: Exact acquisition implementation used by the test.
        """

        super().__init__()
        if acquire is None:
            acquire = _acquire_test_source_delivery
        self._acquire = acquire

    def acquire_for_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> NativeTerminalDeliveryLeaseDisposition:
        """Run the configured acquisition implementation.

        :param actions: Exact action population crossing native ownership.
        :returns: Configured delivery disposition.
        """

        return self._acquire(actions)

    @property
    def functional_admission_closed(self) -> bool:
        """Return whether runtime fatality closed functional admission.

        :returns: Current functional-admission disposition.
        """

        with self._lock:
            return self._functional_admission_closed


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
    bind_source_delivery_authority: bool = True,
    testing: bool = False,
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
    :param bind_source_delivery_authority: Whether an enabled source runtime
        receives its atomic fixture delivery owner before start.
    :param testing: Whether to expose deterministic native test controls.
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
        testing=testing,
    )
    if (
        role is TerminalOwnerRole.SOURCE
        and enable_forward_independent_handoff
        and bind_source_delivery_authority
    ):
        runtime.bind_source_delivery_authority(_TestSourceDeliveryAuthority())
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

    owner, registrations = _direct_handoff_owner_population((room_id,))
    return owner, registrations[0]


def _direct_handoff_owner_population(
    room_ids: tuple[int, ...],
) -> tuple[NativeTerminalOwner, tuple[NativeTerminalLifecycleRegistration, ...]]:
    """Start one deterministic native owner for an exact request population.

    :param room_ids: Non-empty unique source request identities.
    :returns: Running owner and its registered source lifecycles.
    """

    if len(room_ids) == 0 or len(set(room_ids)) != len(room_ids):
        raise ValueError("room_ids must be non-empty and unique")
    owner_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.SOURCE, _OWNER_GENERATION)
    )
    remote_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.DECODE, _REMOTE_GENERATION)
    )
    registrations = tuple(
        _registration(owner_identity, remote_identity, room_id)
        for room_id in room_ids
    )
    owner = NativeTerminalOwner(
        input_capacity=16,
        output_capacity=16,
        observation_capacity=16,
        maximum_live_lifecycles=max(4, len(registrations)),
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
    for registration in registrations:
        owner.register_lifecycle(registration)
    owner.start()
    return owner, registrations


def _direct_decode_handoff_owner(
    room_id: int,
) -> tuple[NativeTerminalOwner, NativeTerminalLifecycleRegistration]:
    """Start a deterministic decode owner with its first action producers.

    :param room_id: Stable decode request identity.
    :returns: Running owner and its registered decode lifecycle.
    """

    owner_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.DECODE, _OWNER_GENERATION)
    )
    remote_identity = NativeTerminalProcessIdentity.from_identity(
        _process_identity(TerminalOwnerRole.SOURCE, _REMOTE_GENERATION)
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
    for producer in (
        NativeTerminalProducerRegistration(
            producer_id=_LOCAL_PRODUCER_ID,
            name="python-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.DECODE,
            authenticated_issuer=None,
        ),
        NativeTerminalProducerRegistration(
            producer_id=_REMOTE_CONTROL_PRODUCER_ID,
            name="python-remote-control",
            producer_class=NativeTerminalProducerClass.CONTROL,
            allowed_role=NativeTerminalOwnerRole.DECODE,
            authenticated_issuer=remote_identity,
        ),
    ):
        owner.register_producer(producer)
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


def _drain_direct_source_gathers(
    owner: NativeTerminalOwner,
    registrations: tuple[NativeTerminalLifecycleRegistration, ...],
) -> tuple[NativeTerminalOwnerAction, ...]:
    """Earn and swap one exact source-gather population.

    :param owner: Running deterministic source owner.
    :param registrations: Source lifecycles contributing one action each.
    :returns: Exact source-gather action population in native output order.
    """

    for registration in registrations:
        _submit_direct_source_gather(owner, registration)
    expires_at = time.monotonic() + _WAIT_SECONDS
    while owner.inventory().queued_output_count != len(registrations):
        if time.monotonic() >= expires_at:
            raise TimeoutError("native source-gather population did not settle")
        time.sleep(0.001)
    actions = tuple(
        action for output in owner.drain_outputs() for action in output.actions
    )
    assert len(actions) == len(registrations)
    assert all(
        action.kind is NativeTerminalOwnerActionKind.SOURCE_GATHER_READY
        for action in actions
    )
    return actions


def _drain_direct_decode_scatter(
    owner: NativeTerminalOwner,
    registration: NativeTerminalLifecycleRegistration,
) -> NativeTerminalOwnerAction:
    """Earn and swap the first decode functional action.

    :param owner: Running deterministic decode owner.
    :param registration: Exact decode lifecycle.
    :returns: Native decode-scatter action.
    """

    events = (
        (
            _LOCAL_PRODUCER_ID,
            NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED,
        ),
        (
            _REMOTE_CONTROL_PRODUCER_ID,
            NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
        ),
        (
            _REMOTE_CONTROL_PRODUCER_ID,
            NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
        ),
    )
    for offset, (producer_id, kind) in enumerate(events):
        owner.submit(
            NativeTerminalOwnerEvent(
                producer_id=producer_id,
                binding_digest=registration.binding.digest,
                kind=kind,
                enqueued_ns=_TEST_CLOCK_NS - len(events) + offset + 1,
            )
        )
    _wait_for_owner_inventory(
        owner,
        lambda inventory: inventory.queued_output_count == 1,
    )
    actions = tuple(
        action for output in owner.drain_outputs() for action in output.actions
    )
    assert len(actions) == 1
    assert actions[0].kind is NativeTerminalOwnerActionKind.DECODE_SCATTER_READY
    return actions[0]


def _wait_for_owner_inventory(
    owner: NativeTerminalOwner,
    predicate: Callable[[NativeTerminalOwnerInventory], bool],
) -> NativeTerminalOwnerInventory:
    """Wait for one deterministic native owner phase.

    :param owner: Running native owner.
    :param predicate: Exact inventory condition ending the wait.
    :returns: First inventory satisfying the predicate.
    """

    expires_at = time.monotonic() + _WAIT_SECONDS
    while True:
        inventory = owner.inventory()
        if predicate(inventory):
            return inventory
        if time.monotonic() >= expires_at:
            raise TimeoutError("native owner phase did not settle")
        time.sleep(0.001)


def _claim_direct_source_batch(
    owner: NativeTerminalOwner,
    actions: tuple[NativeTerminalOwnerAction, ...],
) -> None:
    """Drive one exact source callback from a non-main caller.

    :param owner: Running deterministic source owner.
    :param actions: Complete eligible population from the current output drain.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            owner.claim_source_forward_independent_handoffs,
            actions,
        )
        expires_at = time.monotonic() + _WAIT_SECONDS
        while not future.done() and time.monotonic() < expires_at:
            pass
        future.result(timeout=_WAIT_SECONDS)


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


def test_source_handoff_requires_one_atomic_prestart_delivery_authority() -> None:
    """Source interrupt delivery requires one owner for leases and closure."""

    runtime, _, _ = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
        bind_source_delivery_authority=False,
    )
    with pytest.raises(
        NativeTerminalRuntimeError,
        match="lacks a delivery authority",
    ):
        runtime.start()

    authority = _TestSourceDeliveryAuthority()
    runtime.bind_source_delivery_authority(authority)
    with pytest.raises(NativeTerminalRuntimeError, match="already bound"):
        runtime.bind_source_delivery_authority(_TestSourceDeliveryAuthority())
    runtime.start()
    runtime.begin_abort("test fatal")
    assert authority.functional_admission_closed
    with pytest.raises(NativeTerminalRuntimeClosedError, match="before start"):
        runtime.bind_source_delivery_authority(_TestSourceDeliveryAuthority())
    _finish_handoff_runtime(runtime)

    decode, _, _ = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    with pytest.raises(NativeTerminalRuntimeError, match="requires a source runtime"):
        decode.bind_source_delivery_authority(_TestSourceDeliveryAuthority())
    decode.start()
    _finish_handoff_runtime(decode)


@pytest.mark.parametrize(
    ("disposition", "expected_runtime_fatal"),
    (
        (NativeTerminalDeliveryLeaseDisposition.ACQUIRED, False),
        (NativeTerminalDeliveryLeaseDisposition.FAIL_CLOSED_DRAIN, True),
    ),
)
def test_delivery_disposition_precedes_native_handoff_claim(
    monkeypatch: pytest.MonkeyPatch,
    disposition: NativeTerminalDeliveryLeaseDisposition,
    expected_runtime_fatal: bool,
) -> None:
    """Lease state is settled before one exact native batch is claimed."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
        bind_source_delivery_authority=False,
    )
    registration = _registration(owner, remote, 60)
    ordering: list[str] = []

    def acquire(
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> NativeTerminalDeliveryLeaseDisposition:
        """Record the lease decision for one exact action population.

        :param actions: Actions about to cross native ownership.
        :returns: Parameterized lease disposition.
        """

        assert len(actions) == 1
        ordering.append("acquire")
        return disposition

    original_batch_claim = runtime._owner.claim_source_forward_independent_handoffs
    original_single_claim = runtime._owner.claim_forward_independent_handoff

    def claim_batch(actions: tuple[NativeTerminalOwnerAction, ...]) -> None:
        """Record and execute the exact source-batch claim.

        :param actions: Complete action population crossing the handoff.
        """

        ordering.append("claim")
        original_batch_claim(actions)

    def claim_single(action: NativeTerminalOwnerAction) -> None:
        """Record fail-closed single-action claim without a callback.

        :param action: Exact action crossing the handoff.
        """

        ordering.append("claim")
        original_single_claim(action)

    runtime.bind_source_delivery_authority(_TestSourceDeliveryAuthority(acquire))
    monkeypatch.setattr(
        runtime._owner,
        "claim_source_forward_independent_handoffs",
        claim_batch,
    )
    monkeypatch.setattr(
        runtime._owner,
        "claim_forward_independent_handoff",
        claim_single,
    )
    runtime.start()
    try:
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
        if expected_runtime_fatal:
            assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        else:
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                runtime.source_gather_actions.snapshot().queued_count == 0
                and time.monotonic() < expires_at
            ):
                pass
        assert ordering == ["acquire", "claim"]
        snapshot = runtime.snapshot()
        assert (snapshot.fatal_reason is not None) is expected_runtime_fatal
        if expected_runtime_fatal:
            assert runtime.source_gather_actions.snapshot().queued_count == 0
            assert snapshot.source_preclaimed_count == 0
            assert snapshot.source_preclaimed_consumer_count == 0
            assert snapshot.owner.pending_action_count == 0
            return
        action = _drain_actions(runtime.source_gather_actions)[0]
        runtime.acknowledge_consumed_action(action)
    finally:
        _finish_handoff_runtime(runtime)


def test_abort_after_native_claim_rejects_functional_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native claim cannot leave functional work visible after abort wins."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 62)
    claim_returned = threading.Event()
    release_preclaim = threading.Event()
    original_record = runtime._record_source_preclaims

    def record_after_abort(
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> None:
        """Expose the exact native-claim to Python-preclaim boundary.

        :param actions: Claimed native action population.
        """

        claim_returned.set()
        if not release_preclaim.wait(_WAIT_SECONDS):
            raise TimeoutError("source preclaim race barrier expired")
        original_record(actions)

    monkeypatch.setattr(runtime, "_record_source_preclaims", record_after_abort)
    runtime.start()
    try:
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
        while not claim_returned.is_set() and time.monotonic() < expires_at:
            pass
        assert claim_returned.is_set()

        runtime.begin_abort("abort won after the native source claim")
        release_preclaim.set()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.ABORT_DRAINING
        assert runtime.source_gather_actions.snapshot().queued_count == 0
        assert snapshot.source_preclaimed_count == 0
        assert snapshot.source_preclaimed_consumer_count == 0
        assert snapshot.owner.claimed_handoff_action_count == 1
    finally:
        release_preclaim.set()
        _finish_handoff_runtime(runtime)


def test_source_batch_callback_restores_before_downstream_lock() -> None:
    """The callback restores before downstream action processing can block."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 61)
    runtime.start()
    previous_switch_interval = sys.getswitchinterval()
    scheduler_owned_lock = threading.Lock()
    inbox_claimed = threading.Event()
    downstream_entered = threading.Event()
    try:
        runtime.register_lifecycle(registration)

        def consume_gather() -> NativeTerminalOwnerAction:
            action = _drain_actions(runtime.source_gather_actions)[0]
            inbox_claimed.set()
            downstream_entered.set()
            if not scheduler_owned_lock.acquire(timeout=_WAIT_SECONDS):
                raise TimeoutError("scheduler-owned downstream lock was not released")
            scheduler_owned_lock.release()
            runtime.acknowledge_consumed_action(action)
            return action

        def submit_gather() -> None:
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            scheduler_owned_lock.acquire()
            gather_future = executor.submit(consume_gather)
            producer_future = executor.submit(submit_gather)
            sys.setswitchinterval(0.005)
            expires_at = time.monotonic() + _WAIT_SECONDS
            scheduler_iterations = 0
            while not inbox_claimed.is_set() and time.monotonic() < expires_at:
                scheduler_iterations += 1
            assert inbox_claimed.is_set()
            assert downstream_entered.is_set()
            assert not gather_future.done()
            snapshot = runtime.snapshot()
            assert snapshot.owner.unclaimed_handoff_action_count == 0
            assert snapshot.owner.claimed_handoff_action_count == 1
            assert snapshot.owner.handoff_callback_count == 1
            assert snapshot.owner.terminal_handoff_callback_state is (
                NativeTerminalHandoffCallbackTerminalState.RESTORED
            )
            assert snapshot.consumer_pending_count == 1
            scheduler_owned_lock.release()
            gather_future.result(timeout=_WAIT_SECONDS)
            producer_future.result(timeout=_WAIT_SECONDS)
            sys.setswitchinterval(previous_switch_interval)

        assert scheduler_iterations > 0
    finally:
        if scheduler_owned_lock.locked():
            scheduler_owned_lock.release()
        sys.setswitchinterval(previous_switch_interval)
        _finish_handoff_runtime(runtime)


@pytest.mark.parametrize("invalid_case", ("duplicate", "omitted", "extra"))
def test_exact_source_batch_rejection_is_atomic(invalid_case: str) -> None:
    """Malformed or ineligible exact batches transfer no native authority."""

    owner, registrations = _direct_handoff_owner_population((80, 81))
    try:
        actions = _drain_direct_source_gathers(owner, registrations)
        if invalid_case == "duplicate":
            invalid_actions = (actions[0], actions[0])
            expected_error = ValueError
            expected_fatal = NativeTerminalOwnerFatalCode.NONE
        elif invalid_case == "omitted":
            invalid_actions = (actions[0],)
            expected_error = RuntimeError
            expected_fatal = NativeTerminalOwnerFatalCode.HANDOFF_AUTHORITY
        else:
            invalid_actions = (
                actions[0],
                dataclasses.replace(
                    actions[1],
                    action_id=max(action.action_id for action in actions) + 1_000,
                ),
            )
            expected_error = RuntimeError
            expected_fatal = NativeTerminalOwnerFatalCode.HANDOFF_AUTHORITY

        with pytest.raises(expected_error):
            owner.claim_source_forward_independent_handoffs(invalid_actions)

        inventory = owner.inventory()
        assert inventory.fatal_code is expected_fatal
        assert inventory.unclaimed_handoff_action_count == len(actions)
        assert inventory.claimed_handoff_action_count == 0
        assert inventory.source_batch_handoff_count == 0
        assert inventory.source_batch_handoff_action_count == 0
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
        assert inventory.scheduled_source_batch_action_count == 0
        assert inventory.active_source_batch_action_count == 0
        assert inventory.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.NONE
        )
    finally:
        owner.abort_and_close()


@pytest.mark.parametrize("reverse_order", (False, True))
def test_exact_source_batch_claims_complete_multi_action_population(
    reverse_order: bool,
) -> None:
    """One callback claims every eligible drained action independent of order."""

    owner, registrations = _direct_handoff_owner_population((90, 91, 92))
    actions = _drain_direct_source_gathers(owner, registrations)
    claimed_actions = tuple(reversed(actions)) if reverse_order else actions
    try:
        _claim_direct_source_batch(owner, claimed_actions)

        inventory = owner.inventory()
        assert inventory.pending_action_count == len(actions)
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == len(actions)
        assert inventory.source_batch_handoff_count == 1
        assert inventory.source_batch_handoff_action_count == len(actions)
        assert inventory.handoff_callback_count == 1
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
        assert inventory.scheduled_source_batch_action_count == 0
        assert inventory.active_source_batch_action_count == 0
        assert inventory.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.RESTORED
        )

        for action in actions:
            owner.acknowledge_action(action)
        assert owner.inventory().pending_action_count == 0
    finally:
        owner.abort_and_close()


def test_exact_source_batch_pending_call_rejection_rolls_back_activation() -> None:
    """Pending-call rejection retains no activated exact-batch authority."""

    owner, registrations = _direct_handoff_owner_population((82, 83))
    try:
        actions = _drain_direct_source_gathers(owner, registrations)
        owner.reject_next_handoff_pending_call_for_testing()
        with pytest.raises(
            RuntimeError,
            match="CPython rejected the source terminal handoff pending call",
        ):
            owner.claim_source_forward_independent_handoffs(actions)

        inventory = owner.inventory()
        assert inventory.fatal_code is (
            NativeTerminalOwnerFatalCode.PENDING_CALL_QUEUE_FAILURE
        )
        assert inventory.unclaimed_handoff_action_count == len(actions)
        assert inventory.claimed_handoff_action_count == 0
        assert inventory.source_batch_handoff_count == 0
        assert inventory.source_batch_handoff_action_count == 0
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
        assert inventory.scheduled_source_batch_action_count == 0
        assert inventory.active_source_batch_action_count == 0
        assert inventory.terminal_handoff_callback_id > 0
        assert inventory.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.SCHEDULING_REJECTED
        )
    finally:
        owner.abort_and_close()


def test_decode_pending_call_rejection_is_fatal_and_discards_functional_work() -> (
    None
):
    """Generic decode activation distinguishes rejection from a won claim."""

    owner, registration = _direct_decode_handoff_owner(86)
    try:
        action = _drain_direct_decode_scatter(owner, registration)
        owner.reject_next_handoff_pending_call_for_testing()
        with pytest.raises(
            RuntimeError,
            match="terminal handoff pending-call scheduling failed",
        ):
            owner.activate_forward_independent_handoff(action)

        rejected = owner.inventory()
        assert rejected.fatal_code is (
            NativeTerminalOwnerFatalCode.PENDING_CALL_QUEUE_FAILURE
        )
        assert rejected.unclaimed_handoff_action_count == 0
        assert rejected.claimed_handoff_action_count == 0
        assert rejected.discarded_handoff_action_count == 1
        assert not rejected.handoff_callback_scheduled
        assert not rejected.handoff_callback_active
        assert not rejected.handoff_callback_restoring

        owner.fail_action_delivery(
            action,
            "discard decode work after pending-call scheduling rejection",
        )
        _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.queued_fatal_output_count == 1,
        )
        fatal_actions = tuple(
            current
            for output in owner.drain_outputs()
            for current in output.actions
        )
        assert tuple(current.kind for current in fatal_actions) == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        for fatal_action in fatal_actions:
            owner.acknowledge_action(fatal_action)

        drained = owner.inventory()
        assert drained.pending_action_count == 0
        assert drained.unclaimed_handoff_action_count == 0
        assert drained.claimed_handoff_action_count == 0
        assert drained.discarded_handoff_action_count == 1
        assert drained.fatal_code is (
            NativeTerminalOwnerFatalCode.PENDING_CALL_QUEUE_FAILURE
        )
    finally:
        owner.abort_and_close()


def test_decode_activation_rejection_precedes_inbox_publication() -> None:
    """A composed decode runtime never publishes rejected handoff work."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
        testing=True,
    )
    registration = _registration(owner, remote, 87)
    runtime._owner.enable_test_clock(_TEST_CLOCK_NS)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        runtime._owner.reject_next_handoff_pending_call_for_testing()
        binding_digest = registration.binding.digest
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.DECODE_ALLOCATION_PUBLISHED,
            enqueued_ns=_TEST_CLOCK_NS - 3,
        )
        runtime.submit(
            _REMOTE_CONTROL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
            enqueued_ns=_TEST_CLOCK_NS - 2,
        )
        runtime.submit(
            _REMOTE_CONTROL_PRODUCER_ID,
            binding_digest,
            NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
            enqueued_ns=_TEST_CLOCK_NS - 1,
        )

        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert runtime.decode_scatter_actions.snapshot().queued_count == 0
        assert snapshot.owner.unclaimed_handoff_action_count == 0
        assert snapshot.owner.claimed_handoff_action_count == 0
        assert snapshot.owner.discarded_handoff_action_count == 1
        assert snapshot.owner.handoff_callback_count == 0
        assert not snapshot.owner.handoff_callback_scheduled
        assert not snapshot.owner.handoff_callback_active
        assert not snapshot.owner.handoff_callback_restoring
        fatal_actions = _drain_actions(runtime.lifecycle_actions)
        assert tuple(action.kind for action in fatal_actions) == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        runtime.acknowledge_aborted_action(fatal_actions[0])
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        _finish_handoff_runtime(runtime)


def test_native_abort_before_decode_activation_projects_no_functional_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native fatality closes decode activation before inbox publication."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 88)
    activation_entered = threading.Event()
    release_activation = threading.Event()
    original_activation = runtime._owner.activate_forward_independent_handoff

    def pause_before_activation(action: NativeTerminalOwnerAction) -> bool:
        """Expose the native-fatal to decode-activation ordering.

        :param action: Exact decode action about to activate.
        :returns: Native activation disposition after the race release.
        """

        activation_entered.set()
        if not release_activation.wait(_WAIT_SECONDS):
            raise TimeoutError("decode activation was not released")
        return original_activation(action)

    monkeypatch.setattr(
        runtime._owner,
        "activate_forward_independent_handoff",
        pause_before_activation,
    )
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        binding_digest = registration.binding.digest
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
        assert activation_entered.wait(_WAIT_SECONDS)

        runtime._owner.begin_abort()
        native_fatal = runtime._owner.inventory()
        assert native_fatal.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        release_activation.set()

        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert snapshot.owner.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        assert runtime.decode_scatter_actions.snapshot().queued_count == 0
        assert snapshot.owner.handoff_callback_count == 0
        assert snapshot.owner.unclaimed_handoff_action_count == 0
        assert snapshot.owner.claimed_handoff_action_count == 0
        assert snapshot.owner.discarded_handoff_action_count == 1
        fatal_actions = _drain_actions(runtime.lifecycle_actions)
        assert tuple(action.kind for action in fatal_actions) == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        runtime.acknowledge_aborted_action(fatal_actions[0])
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        release_activation.set()
        _finish_handoff_runtime(runtime)


def test_decode_activation_and_projection_linearize_before_runtime_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime fatality cannot enter between activation and publication."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 89)
    activation_entered = threading.Event()
    release_activation = threading.Event()

    def hold_activation(action: NativeTerminalOwnerAction) -> bool:
        """Hold the exact activation-to-projection boundary.

        :param action: Decode action being made forward-independent.
        :returns: ``True`` after the test releases activation.
        """

        assert action.kind is NativeTerminalOwnerActionKind.DECODE_SCATTER_READY
        activation_entered.set()
        if not release_activation.wait(_WAIT_SECONDS):
            raise TimeoutError("decode activation was not released")
        return True

    monkeypatch.setattr(
        runtime._owner,
        "activate_forward_independent_handoff",
        hold_activation,
    )
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        binding_digest = registration.binding.digest
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
        assert activation_entered.wait(_WAIT_SECONDS)

        condition_acquired = runtime._condition.acquire(blocking=False)
        if condition_acquired:
            runtime._condition.release()
        assert not condition_acquired

        release_activation.set()
        with runtime._condition:
            assert runtime.decode_scatter_actions.snapshot().queued_count == 1
            assert len(runtime._consumer_pending) == 1
            runtime._enter_runtime_fatal_locked(
                "synthetic fatal after decode activation linearization"
            )

        actions = _drain_actions(runtime.decode_scatter_actions)
        assert len(actions) == 1
        runtime.acknowledge_aborted_action(actions[0])
        snapshot = runtime.snapshot()
        assert snapshot.consumer_pending_count == 0
        assert snapshot.owner.unclaimed_handoff_action_count == 0
        assert snapshot.owner.claimed_handoff_action_count == 1
    finally:
        release_activation.set()
        _finish_handoff_runtime(runtime)


def test_decode_handoff_schedules_next_generation_during_restoration() -> None:
    """Generic decode restoration does not masquerade as a source batch."""

    owner, registration = _direct_decode_handoff_owner(88)
    scatter = _drain_direct_decode_scatter(owner, registration)
    owner.set_handoff_callback_holds_for_testing(
        hold_activation=False,
        hold_restoration=True,
    )

    def drive_both_generations() -> NativeTerminalOwnerInventory:
        """Claim two decode actions across the first callback restoration.

        :returns: Quiescent native controller inventory.
        """

        assert owner.activate_forward_independent_handoff(scatter)
        _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.handoff_callback_active,
        )
        owner.claim_forward_independent_handoff(scatter)
        owner.acknowledge_action(scatter)
        _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.handoff_callback_restoring,
        )

        for kind in (
            NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
            NativeTerminalOwnerEventKind.DECODE_SCATTER_TERMINAL,
        ):
            owner.submit(
                NativeTerminalOwnerEvent(
                    producer_id=_LOCAL_PRODUCER_ID,
                    binding_digest=registration.binding.digest,
                    kind=kind,
                    enqueued_ns=_TEST_CLOCK_NS,
                )
            )
        _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.queued_output_count == 1,
        )
        teardown_actions = tuple(
            action for output in owner.drain_outputs() for action in output.actions
        )
        assert len(teardown_actions) == 1
        teardown = teardown_actions[0]
        assert teardown.kind is NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY

        assert owner.activate_forward_independent_handoff(teardown)
        overlapping = owner.inventory()
        assert overlapping.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert overlapping.handoff_callback_restoring
        assert overlapping.handoff_callback_scheduled
        owner.claim_forward_independent_handoff(teardown)
        owner.acknowledge_action(teardown)
        owner.set_handoff_callback_holds_for_testing(
            hold_activation=False,
            hold_restoration=False,
        )
        return _wait_for_owner_inventory(
            owner,
            lambda inventory: (
                inventory.handoff_callback_count == 2
                and not inventory.handoff_callback_scheduled
                and not inventory.handoff_callback_active
                and not inventory.handoff_callback_restoring
            ),
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(drive_both_generations)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not future.done() and time.monotonic() < expires_at:
                pass
            inventory = future.result(timeout=_WAIT_SECONDS)
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert inventory.pending_action_count == 0
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == 2
        assert inventory.source_batch_handoff_count == 0
        assert inventory.source_batch_handoff_action_count == 0
        assert inventory.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.RESTORED
        )
    finally:
        if not owner.inventory().closed:
            owner.set_handoff_callback_holds_for_testing(
                hold_activation=False,
                hold_restoration=False,
            )
        owner.abort_and_close()


@pytest.mark.parametrize(
    ("phase", "expected_claimed_count", "expected_callback_count"),
    (
        ("activation", 0, 0),
        ("restoration", 1, 1),
    ),
)
def test_exact_source_batch_callback_phase_timeout(
    phase: str,
    expected_claimed_count: int,
    expected_callback_count: int,
) -> None:
    """Activation and restoration deadlines fail closed without live state."""

    owner, registrations = _direct_handoff_owner_population((84,))
    actions = _drain_direct_source_gathers(owner, registrations)
    owner.set_handoff_callback_holds_for_testing(
        hold_activation=phase == "activation",
        hold_restoration=phase == "restoration",
    )

    def expire_callback_phase() -> None:
        """Advance the deterministic clock only after the target phase begins."""

        _wait_for_owner_inventory(
            owner,
            lambda inventory: (
                inventory.handoff_callback_scheduled
                if phase == "activation"
                else inventory.handoff_callback_restoring
            ),
        )
        owner.set_test_clock(_TEST_CLOCK_NS + 120_000_000_000)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            expiry_future = executor.submit(expire_callback_phase)
            claim_future = executor.submit(
                owner.claim_source_forward_independent_handoffs,
                actions,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not claim_future.done() and time.monotonic() < expires_at:
                pass
            expiry_future.result(timeout=_WAIT_SECONDS)
            with pytest.raises(RuntimeError):
                claim_future.result(timeout=_WAIT_SECONDS)

        inventory = owner.inventory()
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.HANDOFF_TIMEOUT
        assert inventory.unclaimed_handoff_action_count == (
            len(actions) - expected_claimed_count
        )
        assert inventory.claimed_handoff_action_count == expected_claimed_count
        assert inventory.handoff_callback_count == expected_callback_count
        assert inventory.source_batch_handoff_count == expected_callback_count
        assert inventory.source_batch_handoff_action_count == expected_claimed_count
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
        assert inventory.scheduled_source_batch_action_count == 0
        assert inventory.active_source_batch_action_count == 0
        assert inventory.terminal_handoff_callback_id > 0
        assert inventory.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.TIMED_OUT
        )
    finally:
        owner.abort_and_close()


@pytest.mark.parametrize(
    ("phase", "expected_claimed_count", "expected_callback_count"),
    (
        ("scheduled", 0, 0),
        ("restoring", 1, 1),
    ),
)
def test_exact_source_batch_abort_defers_close_through_callback_lifetime(
    phase: str,
    expected_claimed_count: int,
    expected_callback_count: int,
) -> None:
    """Abort keeps the owner alive through scheduled and restoring callbacks."""

    owner, registrations = _direct_handoff_owner_population((85,))
    actions = _drain_direct_source_gathers(owner, registrations)
    owner.set_handoff_callback_holds_for_testing(
        hold_activation=phase == "scheduled",
        hold_restoration=phase == "restoring",
    )

    def abort_during_callback() -> None:
        """Abort only after the requested native callback phase is visible."""

        _wait_for_owner_inventory(
            owner,
            lambda inventory: (
                inventory.handoff_callback_scheduled
                if phase == "scheduled"
                else inventory.handoff_callback_restoring
            ),
        )
        owner.abort_and_close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        abort_future = executor.submit(abort_during_callback)
        claim_future = executor.submit(
            owner.claim_source_forward_independent_handoffs,
            actions,
        )
        expires_at = time.monotonic() + _WAIT_SECONDS
        while not claim_future.done() and time.monotonic() < expires_at:
            pass
        abort_future.result(timeout=_WAIT_SECONDS)
        with pytest.raises(RuntimeError):
            claim_future.result(timeout=_WAIT_SECONDS)

    inventory = _wait_for_owner_inventory(owner, lambda current: current.closed)
    assert inventory.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
    assert inventory.pending_action_count == 0
    assert inventory.unclaimed_handoff_action_count == 0
    assert inventory.claimed_handoff_action_count == expected_claimed_count
    assert inventory.handoff_callback_count == expected_callback_count
    assert inventory.source_batch_handoff_count == expected_callback_count
    assert inventory.source_batch_handoff_action_count == expected_claimed_count
    assert not inventory.handoff_callback_scheduled
    assert not inventory.handoff_callback_active
    assert not inventory.handoff_callback_restoring
    assert inventory.scheduled_source_batch_action_count == 0
    assert inventory.active_source_batch_action_count == 0
    assert inventory.terminal_handoff_callback_id > 0
    assert inventory.terminal_handoff_callback_state is (
        NativeTerminalHandoffCallbackTerminalState.CANCELLED
    )
    assert not inventory.input_eventfd_open
    assert not inventory.output_eventfd_open
    assert not inventory.observation_eventfd_open


def test_exact_source_batch_linearizes_before_post_restored_abort() -> None:
    """A committed native claim remains successful when abort follows it."""

    owner, registrations = _direct_handoff_owner_population((87,))
    actions = _drain_direct_source_gathers(owner, registrations)
    previous_switch_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            claim_future = executor.submit(
                owner.claim_source_forward_independent_handoffs,
                actions,
            )
            restored = _wait_for_owner_inventory(
                owner,
                lambda inventory: inventory.terminal_handoff_callback_state
                is NativeTerminalHandoffCallbackTerminalState.RESTORED,
            )
            assert restored.terminal_handoff_callback_id > 0
            assert not claim_future.done()

            owner.abort_and_close()
            claim_future.result(timeout=_WAIT_SECONDS)

        closed = _wait_for_owner_inventory(owner, lambda inventory: inventory.closed)
        assert closed.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        assert closed.terminal_handoff_callback_state is (
            NativeTerminalHandoffCallbackTerminalState.RESTORED
        )
        assert closed.pending_action_count == 0
        assert closed.unclaimed_handoff_action_count == 0
        assert closed.claimed_handoff_action_count == len(actions)
        assert not closed.input_eventfd_open
        assert not closed.output_eventfd_open
        assert not closed.observation_eventfd_open
    finally:
        sys.setswitchinterval(previous_switch_interval)
        owner.abort_and_close()


def test_clean_close_returns_after_concurrent_deferred_abort() -> None:
    """Clean close becomes idempotent when deferred abort wins restoration."""

    owner, registrations = _direct_handoff_owner_population((89,))
    actions = _drain_direct_source_gathers(owner, registrations)
    begin_close = threading.Event()
    close_worker_ready = threading.Event()
    abort_worker_ready = threading.Event()
    owner.set_handoff_callback_holds_for_testing(
        hold_activation=False,
        hold_restoration=True,
    )

    def close_after_claim() -> None:
        """Enter clean close only after callback authority is consumed."""

        close_worker_ready.set()
        if not begin_close.wait(_WAIT_SECONDS):
            raise TimeoutError("clean-close race was not released")
        owner.close()

    def abort_after_close_waits() -> None:
        """Let clean close park, then win closure through deferred abort."""

        abort_worker_ready.set()
        _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.handoff_callback_restoring,
        )
        owner.acknowledge_action(actions[0])
        begin_close.set()
        _wait_for_owner_inventory(
            owner,
            lambda inventory: not inventory.admission_open,
        )
        owner.abort_and_close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            close_future = executor.submit(close_after_claim)
            abort_future = executor.submit(abort_after_close_waits)
            assert close_worker_ready.wait(_WAIT_SECONDS)
            assert abort_worker_ready.wait(_WAIT_SECONDS)
            claim_future = executor.submit(
                owner.claim_source_forward_independent_handoffs,
                actions,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not close_future.done() and time.monotonic() < expires_at:
                pass

            abort_future.result(timeout=_WAIT_SECONDS)
            close_future.result(timeout=_WAIT_SECONDS)
            with pytest.raises(RuntimeError):
                claim_future.result(timeout=_WAIT_SECONDS)

        inventory = _wait_for_owner_inventory(
            owner,
            lambda current: current.closed,
        )
        assert (
            inventory.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        )
        assert inventory.pending_action_count == 0
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == len(actions)
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
        assert not inventory.input_eventfd_open
        assert not inventory.output_eventfd_open
        assert not inventory.observation_eventfd_open
    finally:
        inventory = owner.inventory()
        if not inventory.closed:
            owner.set_handoff_callback_holds_for_testing(
                hold_activation=False,
                hold_restoration=False,
            )
        owner.abort_and_close()


def test_later_arrivals_form_a_distinct_exact_source_batch() -> None:
    """Later outputs cannot join an exact batch already restoring its callback."""

    owner, registrations = _direct_handoff_owner_population((93, 94, 95))
    first_actions = _drain_direct_source_gathers(owner, registrations[:1])
    owner.set_handoff_callback_holds_for_testing(
        hold_activation=False,
        hold_restoration=True,
    )
    try:
        def submit_later_arrivals() -> tuple[
            NativeTerminalOwnerInventory,
            NativeTerminalOwnerInventory,
        ]:
            """Queue later outputs while the first callback is restoring.

            :returns: Inventories before and after the later arrivals queue.
            """

            try:
                restoring = _wait_for_owner_inventory(
                    owner,
                    lambda inventory: inventory.handoff_callback_restoring,
                )
                assert restoring.active_source_batch_action_count == 1
                assert restoring.source_batch_handoff_count == 1
                assert restoring.source_batch_handoff_action_count == 1

                for registration in registrations[1:]:
                    _submit_direct_source_gather(owner, registration)
                later_queued = _wait_for_owner_inventory(
                    owner,
                    lambda inventory: inventory.queued_output_count == 2,
                )
                assert later_queued.handoff_callback_restoring
                assert later_queued.active_source_batch_action_count == 1
                assert later_queued.unclaimed_handoff_action_count == 2
                assert later_queued.source_batch_handoff_count == 1
                assert later_queued.source_batch_handoff_action_count == 1
                return restoring, later_queued
            finally:
                owner.set_handoff_callback_holds_for_testing(
                    hold_activation=False,
                    hold_restoration=False,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            later_arrivals = executor.submit(submit_later_arrivals)
            first_claim = executor.submit(
                owner.claim_source_forward_independent_handoffs,
                first_actions,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                (not first_claim.done() or not later_arrivals.done())
                and time.monotonic() < expires_at
            ):
                pass
            if later_arrivals.done():
                later_arrivals.result(timeout=0.0)
            if not first_claim.done() or not later_arrivals.done():
                owner.set_handoff_callback_holds_for_testing(
                    hold_activation=False,
                    hold_restoration=False,
                )
                raise TimeoutError("exact source-batch callback did not settle")

            restoring, later_queued = later_arrivals.result(timeout=_WAIT_SECONDS)
            first_claim.result(timeout=_WAIT_SECONDS)

            assert restoring.active_source_batch_action_count == 1
            assert later_queued.active_source_batch_action_count == 1
            assert later_queued.unclaimed_handoff_action_count == 2

        owner.acknowledge_action(first_actions[0])
        later_actions = tuple(
            action for output in owner.drain_outputs() for action in output.actions
        )
        assert len(later_actions) == 2
        _claim_direct_source_batch(owner, later_actions)
        for action in later_actions:
            owner.acknowledge_action(action)

        inventory = owner.inventory()
        assert inventory.pending_action_count == 0
        assert inventory.unclaimed_handoff_action_count == 0
        assert inventory.claimed_handoff_action_count == 3
        assert inventory.source_batch_handoff_count == 2
        assert inventory.source_batch_handoff_action_count == 3
        assert inventory.handoff_callback_count == 2
        assert not inventory.handoff_callback_scheduled
        assert not inventory.handoff_callback_active
        assert not inventory.handoff_callback_restoring
    finally:
        if not owner.inventory().closed:
            owner.set_handoff_callback_holds_for_testing(
                hold_activation=False,
                hold_restoration=False,
            )
        owner.abort_and_close()


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
            future = executor.submit(complete_and_close)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not future.done() and time.monotonic() < expires_at:
                pass
            snapshot = future.result(timeout=_WAIT_SECONDS)
        assert snapshot.owner.action_count == 6
        assert snapshot.owner.claimed_handoff_action_count == 5
        assert snapshot.owner.unclaimed_handoff_action_count == 0
    finally:
        _finish_handoff_runtime(runtime)


def test_direct_decode_claim_may_win_before_generic_activation() -> None:
    """A direct decode claim makes later generic activation a clean no-op."""

    owner, registration = _direct_decode_handoff_owner(68)
    try:
        action = _drain_direct_decode_scatter(owner, registration)
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


def test_source_generic_activation_fails_closed_but_direct_claim_remains() -> None:
    """Source can reconcile directly but cannot re-enter post-route activation."""

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

        with pytest.raises(RuntimeError, match="requires a decode-role owner"):
            owner.activate_forward_independent_handoff(action)

        rejected = owner.inventory()
        assert rejected.fatal_code is NativeTerminalOwnerFatalCode.HANDOFF_AUTHORITY
        assert rejected.unclaimed_handoff_action_count == 1
        assert rejected.claimed_handoff_action_count == 0
        assert not rejected.handoff_callback_scheduled
        assert not rejected.handoff_callback_active
        assert not rejected.handoff_callback_restoring

        owner.claim_forward_independent_handoff(action)
        owner.acknowledge_action(action)
        reconciled = owner.inventory()
        assert reconciled.unclaimed_handoff_action_count == 0
        assert reconciled.claimed_handoff_action_count == 1
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
    """An unconsumed decode callback expires into process-fatal authority."""

    owner, registration = _direct_decode_handoff_owner(66)
    initial_action = _drain_direct_decode_scatter(owner, registration)
    previous_switch_interval = sys.getswitchinterval()
    output_queued = threading.Event()
    try:

        def expire_active_handoff() -> tuple[NativeTerminalOwnerAction, ...]:
            before_activation = owner.inventory()
            assert before_activation.unclaimed_handoff_action_count == 1
            assert not before_activation.handoff_callback_scheduled
            assert not before_activation.handoff_callback_active
            assert owner.activate_forward_independent_handoff(initial_action)
            owner.acknowledge_action(initial_action)
            output_queued.set()
            assert owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            owner.set_test_clock(_TEST_CLOCK_NS + 120_000_000_000)
            assert owner.wait_for_process_fatal(_WAIT_SECONDS)
            fatal_actions = tuple(
                action for output in owner.drain_outputs() for action in output.actions
            )
            for action in fatal_actions:
                owner.acknowledge_action(action)
            owner.claim_forward_independent_handoff(initial_action)
            return (initial_action, *fatal_actions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            expiry_future = executor.submit(expire_active_handoff)
            sys.setswitchinterval(0.005)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not output_queued.is_set() and time.monotonic() < expires_at:
                pass
            assert output_queued.is_set()
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

    owner, registration = _direct_decode_handoff_owner(67)
    initial_action = _drain_direct_decode_scatter(owner, registration)
    previous_switch_interval = sys.getswitchinterval()
    output_queued = threading.Event()
    try:

        def reject_close_and_resolve_actions() -> str:
            assert owner.activate_forward_independent_handoff(initial_action)
            owner.acknowledge_action(initial_action)
            output_queued.set()
            assert owner.wait_for_forward_independent_handoff(_WAIT_SECONDS)
            with pytest.raises(RuntimeError) as error:
                owner.close()
            owner.claim_forward_independent_handoff(initial_action)
            for output in owner.drain_outputs():
                for action in output.actions:
                    owner.acknowledge_action(action)
            return str(error.value)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            close_future = executor.submit(reject_close_and_resolve_actions)
            sys.setswitchinterval(0.005)
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not output_queued.is_set() and time.monotonic() < expires_at:
                pass
            assert output_queued.is_set()
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
            assert all(
                runtime._consumer_pending.get(action_id) == action
                for action_id, action in expected_pending.items()
            )
            assert all(
                action_id in expected_pending
                or action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
                for action_id, action in runtime._consumer_pending.items()
            )
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


def test_mixed_source_batch_rejects_work_and_preserves_lifecycle_authority() -> (
    None
):
    """A co-drained native fatal preserves only fail-closed lifecycle work."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        output_capacity=4,
        maximum_live_lifecycles=8,
        enable_forward_independent_handoff=True,
    )
    retired, scheduler_work, gather_work, quarantined, overflow = tuple(
        _registration(owner, remote, room_id) for room_id in range(970, 975)
    )
    block_drain = threading.Event()
    drain_entered = threading.Event()
    release_drain = threading.Event()
    original_drain = runtime._owner.drain_outputs

    def conditionally_blocked_drain() -> tuple[NativeTerminalOwnerOutput, ...]:
        """Hold the exact mixed native population before its sole swap."""

        if block_drain.is_set():
            drain_entered.set()
            if not release_drain.wait(_WAIT_SECONDS):
                raise TimeoutError("mixed native output drain barrier expired")
        return original_drain()

    def advance_to_request_ready_ingress(
        registration: NativeTerminalLifecycleRegistration,
    ) -> None:
        """Advance one source lifecycle to its final imported receipt."""

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

    runtime._owner.drain_outputs = conditionally_blocked_drain  # type: ignore[method-assign]
    runtime.start()
    try:
        for registration in (
            retired,
            scheduler_work,
            gather_work,
            quarantined,
            overflow,
        ):
            runtime.register_lifecycle(registration)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ready_future = executor.submit(
                _emit_source_ready_actions,
                runtime,
                retired,
                remote,
                nonce_value=970,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not ready_future.done() and time.monotonic() < expires_at:
                pass
            reclaim, publisher = ready_future.result(timeout=_WAIT_SECONDS)
        runtime.complete_scheduler_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            reclaim,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
            completion_receipt=_receipt(
                retired,
                owner,
                NativeTerminalReceiptKind.RECLAIM_CONSUMED,
                971,
            ),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ingress_future = executor.submit(
                advance_to_request_ready_ingress,
                scheduler_work,
            )
            expires_at = time.monotonic() + _WAIT_SECONDS
            while not ingress_future.done() and time.monotonic() < expires_at:
                pass
            ingress_future.result(timeout=_WAIT_SECONDS)

        block_drain.set()
        runtime.complete_work_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            publisher,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
            receipt=_receipt(
                retired,
                owner,
                NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
                972,
            ),
        )
        assert drain_entered.wait(_WAIT_SECONDS)

        runtime.submit_imported_receipt(
            _REMOTE_RECEIPT_PRODUCER_ID,
            _receipt(
                scheduler_work,
                remote,
                NativeTerminalReceiptKind.REQUEST_READY,
                973,
            ),
            NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            gather_work.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            gather_work.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            quarantined.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
            reason="synthetic request-local quarantine before native fatality",
        )

        expires_at = time.monotonic() + _WAIT_SECONDS
        while runtime.snapshot().owner.queued_output_count != 4:
            if time.monotonic() >= expires_at:
                raise TimeoutError("mixed normal native outputs did not settle")
            time.sleep(0.001)

        runtime.submit(
            _LOCAL_PRODUCER_ID,
            overflow.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        )
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            overflow.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        )
        expires_at = time.monotonic() + _WAIT_SECONDS
        while True:
            inventory = runtime.snapshot().owner
            if (
                inventory.fatal_code
                is NativeTerminalOwnerFatalCode.OUTPUT_QUEUE_OVERFLOW
            ):
                break
            if time.monotonic() >= expires_at:
                raise TimeoutError("mixed native fatal reserve did not settle")
            time.sleep(0.001)
        assert inventory.queued_output_count == 7
        assert inventory.queued_fatal_output_count == 3

        release_drain.set()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while runtime.lifecycle_actions.snapshot().queued_count != 5:
            if time.monotonic() >= expires_at:
                raise TimeoutError("preserved lifecycle authority did not settle")
            time.sleep(0.001)

        before_claim = runtime.snapshot()
        assert before_claim.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert runtime.scheduler_actions.snapshot().queued_count == 0
        assert runtime.publisher_actions.snapshot().queued_count == 0
        assert runtime.source_gather_actions.snapshot().queued_count == 0
        assert before_claim.owner.pending_action_count == 0
        assert before_claim.source_preclaimed_count == 0
        assert before_claim.source_preclaimed_consumer_count == 1

        lifecycle_actions = _drain_actions(runtime.lifecycle_actions)
        kinds = tuple(action.kind for action in lifecycle_actions)
        assert kinds.count(NativeTerminalOwnerActionKind.REQUEST_RETIRED) == 1
        assert kinds.count(NativeTerminalOwnerActionKind.REQUEST_QUARANTINED) == 1
        assert kinds.count(NativeTerminalOwnerActionKind.PROCESS_FATAL) == 3
        for action in lifecycle_actions:
            runtime.acknowledge_aborted_action(action)

        reconciled = runtime.snapshot()
        assert reconciled.consumer_pending_count == 0
        assert reconciled.source_preclaimed_count == 0
        assert reconciled.source_preclaimed_consumer_count == 0
        assert reconciled.scheduler_live_count == 0
        expected_quarantined = {
            scheduler_work.binding.digest,
            gather_work.binding.digest,
            quarantined.binding.digest,
            overflow.binding.digest,
        }
        assert set(reconciled.quarantined_binding_digests) == expected_quarantined

        runtime.begin_abort()
        _retire_all_producers(runtime)
        runtime.join_producers()
        _drain_observations(runtime)
        runtime.finish_abort_close()
        assert runtime.snapshot().disposition is NativeTerminalRuntimeDisposition.STOPPED
    finally:
        release_drain.set()
        _finish_handoff_runtime(runtime)
