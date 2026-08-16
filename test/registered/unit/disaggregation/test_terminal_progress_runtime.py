import concurrent.futures
import dataclasses
import selectors
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
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
        role-specific forward-independent delivery.
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


def _submit_runtime_decode_scatter(
    runtime: NativeTerminalRuntime,
    registration: NativeTerminalLifecycleRegistration,
) -> None:
    """Earn one decode-scatter action through a composed runtime.

    :param runtime: Running decode runtime.
    :param registration: Registered decode lifecycle.
    """

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


def _submit_runtime_source_gather(
    runtime: NativeTerminalRuntime,
    registration: NativeTerminalLifecycleRegistration,
) -> None:
    """Earn one source-gather action through a composed runtime.

    :param runtime: Running source runtime.
    :param registration: Registered source lifecycle.
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
    """Claim one exact source output-drain batch synchronously.

    :param owner: Running deterministic source owner.
    :param actions: Complete eligible population from the current output drain.
    """

    owner.claim_source_forward_independent_handoffs(actions)


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
    """Drive exact fail-closed cleanup for a handoff-enabled runtime.

    :param runtime: Handoff-enabled runtime requiring exact cleanup.
    """

    _finish_fail_closed(runtime)


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


def test_source_batch_claim_does_not_borrow_downstream_lock() -> None:
    """A source batch transfers before downstream processing can block."""

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
            assert snapshot.owner.active_handoff_action_count == 0
            assert snapshot.owner.settled_handoff_action_count == 1
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
    finally:
        owner.abort_and_close()


@pytest.mark.parametrize("reverse_order", (False, True))
def test_exact_source_batch_claims_complete_multi_action_population(
    reverse_order: bool,
) -> None:
    """One atomic commit claims every eligible action independent of order."""

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

        for action in actions:
            owner.acknowledge_action(action)
        assert owner.inventory().pending_action_count == 0
    finally:
        owner.abort_and_close()


@pytest.mark.parametrize(
    ("failure_mode", "expected_claimed", "expected_discarded"),
    (
        ("before_claim", 0, 1),
        ("after_claim", 1, 0),
        ("enqueue", 1, 0),
    ),
)
def test_decode_publication_failure_exposes_no_consumer_work(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_claimed: int,
    expected_discarded: int,
) -> None:
    """Every failed publication rolls back Python authority before its wake."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 86)
    original_claim = runtime._owner.claim_forward_independent_handoff

    def fail_before_claim(action: NativeTerminalOwnerAction) -> None:
        """Fail without mutating native authority.

        :param action: Exact decode action entering publication.
        """

        raise RuntimeError(f"synthetic pre-claim failure for {action.action_id}")

    def fail_after_claim(action: NativeTerminalOwnerAction) -> None:
        """Fail after the real native claim commits.

        :param action: Exact decode action entering publication.
        """

        original_claim(action)
        raise RuntimeError(f"synthetic post-claim failure for {action.action_id}")

    if failure_mode == "before_claim":
        monkeypatch.setattr(
            runtime._owner,
            "claim_forward_independent_handoff",
            fail_before_claim,
        )
    elif failure_mode == "after_claim":
        monkeypatch.setattr(
            runtime._owner,
            "claim_forward_independent_handoff",
            fail_after_claim,
        )
    else:

        def fail_enqueue(action: NativeTerminalOwnerAction) -> None:
            """Reject the destination wake after native claim.

            :param action: Exact decode action entering its destination.
            """

            raise NativeTerminalRuntimeError(
                f"synthetic decode inbox failure for {action.action_id}"
            )

        monkeypatch.setattr(runtime._decode_scatter_actions, "_enqueue", fail_enqueue)

    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_decode_scatter(runtime, registration)
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        snapshot = runtime.snapshot()
        assert snapshot.fatal_reason is not None
        assert runtime.decode_scatter_actions.snapshot().queued_count == 0
        assert snapshot.decode_publication_preclaimed_count == 0
        assert snapshot.owner.unclaimed_handoff_action_count == 0
        assert snapshot.owner.claimed_handoff_action_count == expected_claimed
        assert snapshot.owner.discarded_handoff_action_count == expected_discarded
        assert all(
            action.kind is not NativeTerminalOwnerActionKind.DECODE_SCATTER_READY
            for action in runtime._consumer_pending.values()
        )
    finally:
        _finish_handoff_runtime(runtime)


def test_decode_publication_preclaim_is_atomic_for_an_awakened_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inbox wake cannot expose work before its exact preclaim is durable."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 87)
    publication_entered = threading.Event()
    release_publication = threading.Event()
    original_enqueue = runtime._decode_scatter_actions._enqueue

    def hold_after_enqueue(action: NativeTerminalOwnerAction) -> None:
        """Hold the transaction after the wake becomes readable.

        :param action: Exact decode action durably queued by the real inbox.
        """

        original_enqueue(action)
        publication_entered.set()
        if not release_publication.wait(_WAIT_SECONDS):
            raise TimeoutError("decode publication transaction was not released")

    monkeypatch.setattr(
        runtime._decode_scatter_actions,
        "_enqueue",
        hold_after_enqueue,
    )
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            consumer = executor.submit(_drain_actions, runtime.decode_scatter_actions)
            _submit_runtime_decode_scatter(runtime, registration)
            assert publication_entered.wait(_WAIT_SECONDS)
            assert not consumer.done()
            condition_acquired = runtime._condition.acquire(blocking=False)
            if condition_acquired:
                runtime._condition.release()
            assert not condition_acquired

            release_publication.set()
            actions = consumer.result(timeout=_WAIT_SECONDS)

        assert len(actions) == 1
        snapshot = runtime.snapshot()
        assert snapshot.decode_publication_preclaimed_count == 0
        assert snapshot.owner.unclaimed_handoff_action_count == 0
        assert snapshot.owner.claimed_handoff_action_count == 1
        with runtime._condition:
            assert actions[0].action_id in runtime._inbox_claimed_action_ids
        runtime.acknowledge_consumed_action(actions[0])
    finally:
        release_publication.set()
        _finish_handoff_runtime(runtime)


def test_busy_consumer_does_not_gate_later_decode_publication() -> None:
    """Later decode work reaches its inbox while earlier work remains queued."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registrations = (
        _registration(owner, remote, 88),
        _registration(owner, remote, 89),
    )
    runtime.start()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
            _submit_runtime_decode_scatter(runtime, registration)

        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.decode_scatter_actions.snapshot().queued_count != 2
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.decode_scatter_actions.snapshot().queued_count == 2
        snapshot = runtime.snapshot()
        assert snapshot.decode_publication_preclaimed_count == 2
        assert snapshot.owner.claimed_handoff_action_count == 2
        assert snapshot.owner.unclaimed_handoff_action_count == 0

        for action in _drain_actions(runtime.decode_scatter_actions):
            runtime.acknowledge_consumed_action(action)
        assert runtime.snapshot().decode_publication_preclaimed_count == 0
    finally:
        _finish_handoff_runtime(runtime)


def test_sibling_routing_failure_preserves_decode_retirement_preclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling failure preserves real native retirement authority."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 91)
    routed_retirement: list[NativeTerminalOwnerAction] = []
    original_route_output = runtime._route_output
    original_project = runtime._project_action

    def reject_sibling_projection(action: NativeTerminalOwnerAction) -> None:
        """Fail the first action before it acquires a consumer.

        :param action: Exact action entering output projection.
        """

        if action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED:
            raise NativeTerminalRuntimeError("synthetic sibling routing failure")
        original_project(action)

    def route_with_failing_sibling(output: NativeTerminalOwnerOutput) -> None:
        """Prepend one decode-valid routing failure to a native retirement.

        The retirement itself is earned by the real native reducer. Only the
        synthetic sibling exists to enter the generic post-failure lifecycle
        branch, which native decode outputs do not otherwise batch today.

        :param output: Real native decode-cancellation output.
        """

        assert len(output.actions) == 1
        retirement = output.actions[0]
        assert retirement.kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED
        routed_retirement.append(retirement)
        sibling = NativeTerminalOwnerAction(
            action_id=retirement.action_id + 1_000_000,
            kind=NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            binding=retirement.binding,
            commit_timestamp_ns=retirement.commit_timestamp_ns,
            receipt=None,
        )
        original_route_output(
            dataclasses.replace(output, actions=(sibling, retirement))
        )

    monkeypatch.setattr(runtime, "_project_action", reject_sibling_projection)
    monkeypatch.setattr(runtime, "_route_output", route_with_failing_sibling)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        runtime.submit(
            _LOCAL_PRODUCER_ID,
            registration.binding.digest,
            NativeTerminalOwnerEventKind.DECODE_CANCEL_UNPUBLISHED,
            reason="synthetic cancellation before publication",
        )
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        assert len(routed_retirement) == 1
        retirement = routed_retirement[0]
        snapshot = runtime.snapshot()
        assert snapshot.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        assert snapshot.decode_publication_preclaimed_count == 1
        native_inventory = snapshot.owner
        assert native_inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert native_inventory.pending_action_count == 0
        assert native_inventory.unclaimed_handoff_action_count == 0
        assert native_inventory.claimed_handoff_action_count == 1
        assert native_inventory.discarded_handoff_action_count == 0

        actions = _drain_actions(runtime.lifecycle_actions)
        assert actions == (retirement,)
        assert runtime.snapshot().decode_publication_preclaimed_count == 0
        runtime.acknowledge_aborted_action(retirement)
        assert runtime.snapshot().consumer_pending_count == 0
    finally:
        monkeypatch.undo()
        _finish_handoff_runtime(runtime)


def test_native_abort_racing_decode_publication_preserves_exact_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native abort before claim still leaves one conservable Python owner."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 90)
    claim_entered = threading.Event()
    release_claim = threading.Event()
    original_claim = runtime._owner.claim_forward_independent_handoff

    def hold_claim(action: NativeTerminalOwnerAction) -> None:
        """Expose native abort immediately before the publication claim.

        :param action: Exact decode action entering publication.
        """

        claim_entered.set()
        if not release_claim.wait(_WAIT_SECONDS):
            raise TimeoutError("decode claim race was not released")
        original_claim(action)

    monkeypatch.setattr(
        runtime._owner,
        "claim_forward_independent_handoff",
        hold_claim,
    )
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_decode_scatter(runtime, registration)
        assert claim_entered.wait(_WAIT_SECONDS)
        assert runtime.decode_scatter_actions.snapshot().queued_count == 0
        runtime._owner.begin_abort()
        release_claim.set()
        assert runtime.wait_for_output_projection(_WAIT_SECONDS)

        actions = _drain_actions(runtime.decode_scatter_actions)
        assert len(actions) == 1
        snapshot = runtime.snapshot()
        assert (
            snapshot.owner.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        )
        assert snapshot.owner.claimed_handoff_action_count == 1
        assert snapshot.decode_publication_preclaimed_count == 0
        runtime.begin_abort("reconcile native abort after publication")
        runtime.acknowledge_aborted_action(actions[0])
    finally:
        release_claim.set()
        _finish_handoff_runtime(runtime)


def test_runtime_abort_after_decode_publication_consumes_preclaim_once() -> None:
    """Abort after the publication transaction preserves one downstream owner."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 92)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_decode_scatter(runtime, registration)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.decode_scatter_actions.snapshot().queued_count == 0
            and time.monotonic() < expires_at
        ):
            pass
        before_abort = runtime.snapshot()
        assert before_abort.decode_publication_preclaimed_count == 1
        assert before_abort.owner.claimed_handoff_action_count == 1

        runtime.begin_abort("synthetic abort after decode publication")
        actions = _drain_actions(runtime.decode_scatter_actions)
        assert len(actions) == 1
        assert runtime.snapshot().decode_publication_preclaimed_count == 0
        runtime.acknowledge_aborted_action(actions[0])
        assert not runtime.acknowledge_aborted_action_if_pending(actions[0])
    finally:
        _finish_handoff_runtime(runtime)


def test_production_handoff_has_no_pending_call_surface() -> None:
    """Production handoff code contains no main-thread callback rendezvous."""

    terminal_progress = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "disaggregation"
        / "terminal_progress"
    )
    sources = (
        terminal_progress / "native_owner_bridge.cpp",
        terminal_progress / "native_owner.py",
        terminal_progress / "runtime.py",
    )
    forbidden = (
        "Py_AddPendingCall",
        "activate_forward_independent_handoff",
        "wait_for_forward_independent_handoff",
        "handoff_callback",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), source


def test_scheduler_launch_handoff_waits_for_typed_inbox_delivery() -> None:
    """A launch waits for inbox claim, not downstream functional completion."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 96)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_source_gather(runtime, registration)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.source_gather_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        preclaimed = runtime.snapshot().owner
        assert preclaimed.claimed_handoff_action_count == 1
        assert preclaimed.active_handoff_action_count == 1
        assert runtime.source_gather_actions.snapshot().queued_count == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            inventory = _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            assert inventory.scheduler_launch_handoff_begin_watermark is not None
            assert not launch.done()

            gather = _drain_actions(runtime.source_gather_actions)[0]
            token = launch.result(timeout=_WAIT_SECONDS)

        assert type(token) is int
        delivered = runtime.snapshot()
        assert delivered.owner.active_handoff_action_count == 0
        assert delivered.owner.settled_handoff_action_count == 1
        assert delivered.owner.pending_action_count == 0
        assert delivered.consumer_pending_count == 1
        runtime.end_scheduler_launch_handoff(token)
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            gather,
            NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        )
    finally:
        _finish_handoff_runtime(runtime)


def test_action_emitted_during_launch_token_blocks_only_the_next_begin() -> None:
    """Work racing an authorized enqueue is charged to the next launch."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 97)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        token = runtime.begin_scheduler_launch_handoff()
        _submit_runtime_source_gather(runtime, registration)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.source_gather_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.source_gather_actions.snapshot().queued_count == 1
        runtime.end_scheduler_launch_handoff(token)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            next_launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            inventory = _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            assert inventory.scheduler_launch_handoff_begin_watermark is not None
            assert not next_launch.done()

            gather = _drain_actions(runtime.source_gather_actions)[0]
            next_token = next_launch.result(timeout=_WAIT_SECONDS)

        runtime.end_scheduler_launch_handoff(next_token)
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            gather,
            NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        )
    finally:
        _finish_handoff_runtime(runtime)


def test_newer_action_does_not_extend_a_captured_launch_watermark() -> None:
    """A later action cannot starve a wait over the captured action prefix."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registrations = (
        _registration(owner, remote, 98),
        _registration(owner, remote, 99),
    )
    runtime.start()
    try:
        for registration in registrations:
            runtime.register_lifecycle(registration)
        _submit_runtime_source_gather(runtime, registrations[0])
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.source_gather_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.source_gather_actions.snapshot().queued_count == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            waiting = _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            watermark = waiting.scheduler_launch_handoff_begin_watermark
            assert watermark is not None
            assert not launch.done()

            _submit_runtime_source_gather(runtime, registrations[1])
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                runtime.source_gather_actions.snapshot().queued_count != 2
                and time.monotonic() < expires_at
            ):
                pass
            assert runtime.source_gather_actions.snapshot().queued_count == 2
            first = runtime.source_gather_actions.drain(maximum_items=1)[0]
            assert first.action_id <= watermark
            second_inventory = runtime.snapshot().owner
            assert second_inventory.active_handoff_action_count == 1
            token = launch.result(timeout=_WAIT_SECONDS)

        runtime.end_scheduler_launch_handoff(token)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            next_launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            inventory = _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            assert inventory.scheduler_launch_handoff_begin_watermark is not None
            assert not next_launch.done()
            second = _drain_actions(runtime.source_gather_actions)[0]
            assert second.action_id > watermark
            next_token = next_launch.result(timeout=_WAIT_SECONDS)
        runtime.end_scheduler_launch_handoff(next_token)
        for action in (first, second):
            runtime.complete_work_action(
                _LOCAL_PRODUCER_ID,
                action,
                NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
            )
    finally:
        _finish_handoff_runtime(runtime)


def test_scheduler_launch_handoff_rejects_a_replayed_release() -> None:
    """A released scheduler launch token is permanently stale."""

    runtime, _, _ = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    runtime.start()
    try:
        token = runtime.begin_scheduler_launch_handoff()
        runtime.end_scheduler_launch_handoff(token)
        with pytest.raises(RuntimeError, match="absent or replayed"):
            runtime.end_scheduler_launch_handoff(token)
    finally:
        _finish_handoff_runtime(runtime)


def test_scheduler_launch_handoff_rejects_nested_acquisition() -> None:
    """One process cannot hold or await two scheduler launch tokens."""

    runtime, _, _ = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    runtime.start()
    try:
        token = runtime.begin_scheduler_launch_handoff()
        with pytest.raises(RuntimeError, match="acquisition is concurrent"):
            runtime.begin_scheduler_launch_handoff()
        runtime.end_scheduler_launch_handoff(token)
    finally:
        _finish_handoff_runtime(runtime)


def test_scheduler_action_does_not_enter_the_native_launch_watermark() -> None:
    """Scheduler-affine reclaim authority cannot block its own launch seam."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 101)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        scheduler, publisher = _emit_source_ready_actions(
            runtime,
            registration,
            remote,
            nonce_value=101,
        )
        runtime.complete_work_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            publisher,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
            receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
                102,
            ),
        )

        token = runtime.begin_scheduler_launch_handoff()
        runtime.end_scheduler_launch_handoff(token)

        runtime.complete_scheduler_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            scheduler,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
            completion_receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.RECLAIM_CONSUMED,
                103,
            ),
        )
        retired = _drain_actions(runtime.lifecycle_actions)
        assert tuple(action.kind for action in retired) == (
            NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        )
        runtime.acknowledge_consumed_action(retired[0])
    finally:
        _finish_handoff_runtime(runtime)


def test_gateway_result_lifetime_does_not_extend_launch_exclusion() -> None:
    """A delivered publication action may await its result across a launch."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 102)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        scheduler, publisher = _emit_source_ready_actions(
            runtime,
            registration,
            remote,
            nonce_value=104,
        )
        delivered = runtime.snapshot()
        assert delivered.owner.active_handoff_action_count == 0
        assert delivered.owner.pending_action_count == 0
        assert delivered.consumer_pending_count == 2
        with runtime._condition:
            assert publisher.action_id in runtime._inbox_claimed_action_ids

        token = runtime.begin_scheduler_launch_handoff()
        runtime.end_scheduler_launch_handoff(token)
        awaiting_result = runtime.snapshot()
        assert awaiting_result.owner.pending_action_count == 0
        assert awaiting_result.consumer_pending_count == 2

        runtime.complete_work_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            publisher,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
            receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
                105,
            ),
        )
        runtime.complete_scheduler_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            scheduler,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
            completion_receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.RECLAIM_CONSUMED,
                106,
            ),
        )
        retired = _drain_actions(runtime.lifecycle_actions)
        assert tuple(action.kind for action in retired) == (
            NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        )
        runtime.acknowledge_consumed_action(retired[0])
    finally:
        _finish_handoff_runtime(runtime)


def test_decode_scatter_lifetime_does_not_extend_launch_exclusion() -> None:
    """Delivered scatter work retains decode authority without gating launch."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 103)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_decode_scatter(runtime, registration)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.decode_scatter_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.decode_scatter_actions.snapshot().queued_count == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            assert not launch.done()
            scatter = _drain_actions(runtime.decode_scatter_actions)[0]
            token = launch.result(timeout=_WAIT_SECONDS)

        delivered = runtime.snapshot()
        assert delivered.owner.active_handoff_action_count == 0
        assert delivered.owner.pending_action_count == 0
        assert delivered.consumer_pending_count == 1
        assert delivered.scheduler_live_count == 1
        runtime.end_scheduler_launch_handoff(token)
        runtime.complete_work_action(
            _LOCAL_PRODUCER_ID,
            scatter,
            NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
        )
    finally:
        _finish_handoff_runtime(runtime)


def test_clean_close_rejects_delivered_consumer_authority() -> None:
    """A settled launch exclusion does not make consumer work disposable."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 104)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        scheduler, publisher = _emit_source_ready_actions(
            runtime,
            registration,
            remote,
            nonce_value=107,
        )
        runtime.complete_work_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            publisher,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
            receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
                108,
            ),
        )
        runtime.complete_scheduler_action(
            _OWNER_RECEIPT_PRODUCER_ID,
            scheduler,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
            completion_receipt=_receipt(
                registration,
                owner,
                NativeTerminalReceiptKind.RECLAIM_CONSUMED,
                109,
            ),
        )
        retired = _drain_actions(runtime.lifecycle_actions)[0]
        delivered = runtime.snapshot()
        assert delivered.owner.active_handoff_action_count == 0
        assert delivered.consumer_pending_count == 1

        runtime.stop_admission()
        _retire_all_producers(runtime)
        runtime.join_producers()
        _drain_observations(runtime)
        with pytest.raises(
            NativeTerminalRuntimeError,
            match="consumer authority",
        ):
            runtime.close_clean()

        runtime.acknowledge_consumed_action(retired)
        runtime.close_clean()
        assert (
            runtime.snapshot().disposition
            is NativeTerminalRuntimeDisposition.STOPPED
        )
    finally:
        _finish_handoff_runtime(runtime)


def test_abort_settles_a_blocked_scheduler_launch_handoff() -> None:
    """Fail-closed transition wakes a waiter without authorizing a launch."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.SOURCE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 100)
    runtime.start()
    try:
        runtime.register_lifecycle(registration)
        _submit_runtime_source_gather(runtime, registration)
        expires_at = time.monotonic() + _WAIT_SECONDS
        while (
            runtime.source_gather_actions.snapshot().queued_count != 1
            and time.monotonic() < expires_at
        ):
            pass
        assert runtime.source_gather_actions.snapshot().queued_count == 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launch = executor.submit(runtime.begin_scheduler_launch_handoff)
            _wait_for_owner_inventory(
                runtime._owner,
                lambda value: value.scheduler_launch_handoff_begin_active,
            )
            assert not launch.done()
            runtime.begin_abort("synthetic abort during scheduler launch handoff")
            with pytest.raises(RuntimeError, match="interrupted"):
                launch.result(timeout=_WAIT_SECONDS)

        aborted = runtime.snapshot()
        assert aborted.owner.active_handoff_action_count == 0
        assert aborted.owner.settled_handoff_action_count == 1
        gather = _drain_actions(runtime.source_gather_actions)[0]
        claimed_after_abort = runtime.snapshot()
        assert claimed_after_abort.owner.active_handoff_action_count == 0
        assert claimed_after_abort.owner.settled_handoff_action_count == 1
        runtime.acknowledge_aborted_action(gather)
    finally:
        _finish_handoff_runtime(runtime)


def test_exact_source_batch_claim_survives_later_abort() -> None:
    """An abort cannot roll back a completed atomic source-batch transfer."""

    owner, registrations = _direct_handoff_owner_population((87,))
    actions = _drain_direct_source_gathers(owner, registrations)
    try:
        owner.claim_source_forward_independent_handoffs(actions)
        claimed = owner.inventory()
        assert claimed.claimed_handoff_action_count == len(actions)
        assert claimed.source_batch_handoff_count == 1
        assert claimed.source_batch_handoff_action_count == len(actions)

        owner.abort_and_close()
        closed = owner.inventory()
        assert closed.closed
        assert closed.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        assert closed.pending_action_count == 0
        assert closed.unclaimed_handoff_action_count == 0
        assert closed.claimed_handoff_action_count == len(actions)
        assert closed.source_batch_handoff_count == 1
        assert closed.source_batch_handoff_action_count == len(actions)
    finally:
        owner.abort_and_close()


def test_abort_before_exact_source_batch_claim_transfers_no_authority() -> None:
    """Abort discards an unclaimed source drain without a partial transfer."""

    owner, registrations = _direct_handoff_owner_population((89,))
    actions = _drain_direct_source_gathers(owner, registrations)
    owner.abort_and_close()

    with pytest.raises(RuntimeError, match="running enabled owner"):
        owner.claim_source_forward_independent_handoffs(actions)

    inventory = owner.inventory()
    assert inventory.closed
    assert inventory.pending_action_count == 0
    assert inventory.unclaimed_handoff_action_count == 0
    assert inventory.claimed_handoff_action_count == 0
    assert inventory.source_batch_handoff_count == 0
    assert inventory.source_batch_handoff_action_count == 0


def test_later_arrivals_form_a_distinct_exact_source_batch() -> None:
    """Later outputs cannot join the current swapped output-drain batch."""

    owner, registrations = _direct_handoff_owner_population((93, 94, 95))
    first_actions = _drain_direct_source_gathers(owner, registrations[:1])
    try:
        for registration in registrations[1:]:
            _submit_direct_source_gather(owner, registration)
        later_queued = _wait_for_owner_inventory(
            owner,
            lambda inventory: inventory.queued_output_count == 2,
        )
        assert later_queued.unclaimed_handoff_action_count == 3

        _claim_direct_source_batch(owner, first_actions)
        first_claimed = owner.inventory()
        assert first_claimed.unclaimed_handoff_action_count == 2
        assert first_claimed.claimed_handoff_action_count == 1
        assert first_claimed.source_batch_handoff_count == 1
        assert first_claimed.source_batch_handoff_action_count == 1

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
    finally:
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
        assert claimed.owner.active_handoff_action_count == 0
        assert claimed.owner.settled_handoff_action_count == 1
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
        assert before_acknowledgement.owner.active_handoff_action_count == 0
        assert before_acknowledgement.owner.settled_handoff_action_count == 1
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


def test_decode_publication_preclaims_all_actions_and_excludes_adoption() -> None:
    """Four decode actions preclaim at publication; adoption stays scheduler-owned."""

    runtime, owner, remote = _runtime(
        TerminalOwnerRole.DECODE,
        enable_forward_independent_handoff=True,
    )
    registration = _registration(owner, remote, 75)
    binding_digest = registration.binding.digest
    runtime.start()
    try:

        def drain_preclaimed(
            inbox: NativeTerminalActionInbox,
            expected_claim_count: int,
        ) -> tuple[NativeTerminalOwnerAction, ...]:
            """Observe then consume one publication-preclaimed action.

            :param inbox: Exact decode destination inbox.
            :param expected_claim_count: Cumulative native claim count.
            :returns: Exact claimed action population.
            """

            expires_at = time.monotonic() + _WAIT_SECONDS
            while inbox.snapshot().queued_count == 0 and time.monotonic() < expires_at:
                pass
            assert inbox.snapshot().queued_count == 1
            before_drain = runtime.snapshot()
            assert before_drain.decode_publication_preclaimed_count == 1
            assert (
                before_drain.owner.claimed_handoff_action_count == expected_claim_count
            )
            actions = _drain_actions(inbox)
            assert runtime.snapshot().decode_publication_preclaimed_count == 0
            return actions

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
            scatter = drain_preclaimed(runtime.decode_scatter_actions, 1)
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
            teardown = drain_preclaimed(runtime.decode_work_actions, 2)
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
            expires_at = time.monotonic() + _WAIT_SECONDS
            while (
                runtime.scheduler_actions.snapshot().queued_count == 0
                and time.monotonic() < expires_at
            ):
                pass
            adoption_snapshot = runtime.snapshot()
            assert adoption_snapshot.decode_publication_preclaimed_count == 0
            assert adoption_snapshot.owner.claimed_handoff_action_count == 2
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
            local_ready = drain_preclaimed(runtime.coordinator_actions, 3)
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
            lifecycle = drain_preclaimed(runtime.lifecycle_actions, 4)
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
        snapshot = runtime.snapshot()
        assert snapshot.scheduler_live_count == 0
        assert snapshot.scheduler_pending_count == 0
        assert snapshot.consumer_pending_count == 0
        assert snapshot.decode_publication_preclaimed_count == 0
        runtime.stop_admission()
        _retire_all_producers(runtime)
        runtime.join_producers()
        _drain_observations(runtime)
        runtime.close_clean()
        closed = runtime.snapshot()
        assert closed.disposition is NativeTerminalRuntimeDisposition.STOPPED
        assert closed.owner.closed
        assert closed.decode_publication_preclaimed_count == 0
        assert closed.consumer_pending_count == 0
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


def test_mixed_source_batch_rejects_work_and_preserves_lifecycle_authority() -> None:
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
        assert (
            before_claim.disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL
        )
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
        assert (
            runtime.snapshot().disposition is NativeTerminalRuntimeDisposition.STOPPED
        )
    finally:
        release_drain.set()
        _finish_handoff_runtime(runtime)
