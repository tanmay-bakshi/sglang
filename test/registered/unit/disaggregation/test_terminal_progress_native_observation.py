import selectors

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
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerObservation,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalPublicationIdentity,
    NativeTerminalRequestBinding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PRODUCER_ID = 17
_COMPLETED_NS = 50_000
_WAIT_SECONDS = 5.0


def _owner(
    observation_capacity: int,
) -> tuple[NativeTerminalOwner, NativeTerminalProcessIdentity]:
    """Build one deterministic source owner with a local producer.

    :param observation_capacity: Evidence-only native queue capacity.
    :returns: Dormant owner and exact TP-rank identity.
    """

    identity = NativeTerminalProcessIdentity.from_identity(
        TerminalProcessIdentity(
            process_generation=b"o" * 16,
            role=TerminalOwnerRole.SOURCE,
            tp_rank=1,
            tp_size=2,
        )
    )
    owner = NativeTerminalOwner(
        input_capacity=16,
        output_capacity=1,
        observation_capacity=observation_capacity,
        owner_identity=identity,
        maximum_live_lifecycles=16,
        testing=True,
    )
    owner.enable_test_clock(_COMPLETED_NS)
    owner.register_producer(
        NativeTerminalProducerRegistration(
            producer_id=_PRODUCER_ID,
            name="source-submission",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=None,
        )
    )
    return owner, identity


def _registration(
    identity: NativeTerminalProcessIdentity, room_id: int
) -> NativeTerminalLifecycleRegistration:
    """Build one exact source lifecycle registration.

    :param identity: Local native source process.
    :param room_id: Stable request room identity.
    :returns: Complete source lifecycle registration.
    """

    request_key = PackedRequestKey(
        room_id=room_id,
        request_generation=room_id.to_bytes(16, "big"),
    )
    process = TerminalProcessIdentity(
        process_generation=identity.process_generation,
        role=TerminalOwnerRole.SOURCE,
        tp_rank=identity.tp_rank,
        tp_size=identity.tp_size,
    )
    binding = NativeTerminalRequestBinding.from_binding(
        TerminalRequestBinding(
            request_key=request_key,
            owner=process,
            rank_manifest_digest=b"r" * 32,
            allocation_digest=room_id.to_bytes(32, "big"),
        )
    )
    publication = NativeTerminalPublicationIdentity.from_identity(
        TerminalPublicationIdentity(
            request_key=request_key,
            publisher_process_generation=identity.process_generation,
            publication_generation=(room_id + 1_000).to_bytes(16, "big"),
        )
    )
    return NativeTerminalLifecycleRegistration(
        binding=binding,
        publication_identity=publication,
        trusted_issuers=(identity,),
    )


def _wait_readable(file_descriptor: int) -> None:
    """Wait for one native eventfd without cadence polling.

    :param file_descriptor: Borrowed native wake descriptor.
    """

    with selectors.DefaultSelector() as selector:
        selector.register(file_descriptor, selectors.EVENT_READ)
        if len(selector.select(_WAIT_SECONDS)) == 0:
            raise TimeoutError("native observation did not become readable")


def _submit(
    owner: NativeTerminalOwner,
    registration: NativeTerminalLifecycleRegistration,
    kind: NativeTerminalOwnerEventKind,
    enqueued_ns: int,
) -> None:
    """Submit one exact deterministic local event.

    :param owner: Native source owner.
    :param registration: Target lifecycle.
    :param kind: Local source event kind.
    :param enqueued_ns: Exact producer-side timestamp.
    """

    owner.submit(
        NativeTerminalOwnerEvent(
            producer_id=_PRODUCER_ID,
            binding_digest=registration.binding.digest,
            kind=kind,
            enqueued_ns=enqueued_ns,
        )
    )


def test_actionless_submission_projects_exact_non_authoritative_interval() -> None:
    owner, identity = _owner(observation_capacity=4)
    registration = _registration(identity, 71)
    enqueued_ns = _COMPLETED_NS - 1_337
    owner.register_lifecycle(registration)
    _submit(
        owner,
        registration,
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        enqueued_ns,
    )
    owner.start()
    try:
        _wait_readable(owner.observation_fileno())
        assert owner.drain_outputs() == ()

        inventory_before = owner.inventory()
        assert inventory_before.observation_count == 1
        assert inventory_before.queued_observation_count == 1
        assert inventory_before.delivered_observation_count == 0
        assert inventory_before.dropped_observation_count == 0

        batch = owner.drain_observations()
        assert batch.dropped_count == 0
        observations = batch.observations
        assert len(observations) == 1
        observation = observations[0]
        assert type(observation) is NativeTerminalOwnerObservation
        assert observation.binding == registration.binding
        assert observation.owner_sequence == 0
        assert observation.producer_id == _PRODUCER_ID
        assert observation.producer_sequence == 0
        assert observation.producer_rank == identity.tp_rank
        assert observation.event_kind is (
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED
        )
        assert observation.enqueued_ns == enqueued_ns
        assert observation.completed_ns == _COMPLETED_NS
        assert observation.latency_ns == 1_337
        assert observation.role is NativeTerminalOwnerRole.SOURCE

        with pytest.raises(TypeError):
            owner.acknowledge_action(observation)

        inventory_after = owner.inventory()
        assert inventory_after.observation_count == 1
        assert inventory_after.queued_observation_count == 0
        assert inventory_after.delivered_observation_count == 1
        assert inventory_after.dropped_observation_count == 0
    finally:
        owner.abort_and_close()


def test_observation_overflow_cannot_consume_or_fail_action_capacity() -> None:
    owner, identity = _owner(observation_capacity=1)
    first = _registration(identity, 81)
    second = _registration(identity, 82)
    for registration in (first, second):
        owner.register_lifecycle(registration)
        _submit(
            owner,
            registration,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            _COMPLETED_NS - registration.binding.room_id,
        )
    _submit(
        owner,
        second,
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        _COMPLETED_NS - 1,
    )
    owner.start()
    try:
        _wait_readable(owner.output_fileno())
        inventory = owner.inventory()
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert inventory.observation_count == 2
        assert inventory.queued_observation_count == 1
        assert inventory.dropped_observation_count == 1
        assert inventory.queued_output_count == 1

        outputs = owner.drain_outputs()
        assert len(outputs) == 1
        assert outputs[0].event_kind is (
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED
        )
        assert tuple(action.kind for action in outputs[0].actions) == (
            NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        )
        owner.acknowledge_action(outputs[0].actions[0])

        batch = owner.drain_observations()
        assert batch.dropped_count == 0
        observations = batch.observations
        assert tuple(value.binding for value in observations) == (first.binding,)
        settled = owner.inventory()
        assert settled.pending_action_count == 0
        assert settled.delivered_observation_count == 1
        assert settled.dropped_observation_count == 1
        assert settled.observation_count == (
            settled.queued_observation_count
            + settled.delivered_observation_count
            + settled.dropped_observation_count
        )
    finally:
        owner.abort_and_close()


def test_native_observations_preserve_commit_order() -> None:
    owner, identity = _owner(observation_capacity=3)
    registrations = tuple(_registration(identity, room_id) for room_id in range(91, 94))
    for offset, registration in enumerate(registrations):
        owner.register_lifecycle(registration)
        _submit(
            owner,
            registration,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            _COMPLETED_NS - 100 + offset,
        )
    owner.start()
    try:
        _wait_readable(owner.observation_fileno())
        batch = owner.drain_observations()
        assert batch.dropped_count == 0
        observations = batch.observations
        assert tuple(value.binding for value in observations) == tuple(
            value.binding for value in registrations
        )
        assert tuple(value.owner_sequence for value in observations) == (0, 1, 2)
        assert tuple(value.producer_sequence for value in observations) == (0, 1, 2)
    finally:
        owner.abort_and_close()


def test_undrained_observation_is_dropped_without_gating_abort_close() -> None:
    owner, identity = _owner(observation_capacity=1)
    registration = _registration(identity, 101)
    owner.register_lifecycle(registration)
    _submit(
        owner,
        registration,
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        _COMPLETED_NS - 1,
    )
    owner.start()
    _wait_readable(owner.observation_fileno())

    owner.abort_and_close()

    inventory = owner.inventory()
    assert inventory.closed
    assert inventory.queued_observation_count == 0
    assert inventory.delivered_observation_count == 0
    assert inventory.dropped_observation_count == 1
    assert inventory.observation_count == 1
    assert inventory.observation_eventfd_open is False
    assert inventory.output_eventfd_open is False
    assert inventory.fatal_code is not NativeTerminalOwnerFatalCode.NONE
