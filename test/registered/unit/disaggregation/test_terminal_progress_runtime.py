import selectors

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
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
    NativeTerminalActionInbox,
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
) -> tuple[
    NativeTerminalRuntime,
    NativeTerminalProcessIdentity,
    NativeTerminalProcessIdentity,
]:
    """Construct one runtime with a complete frozen producer directory.

    :param role: Lifecycle role owned by the runtime.
    :param scheduler_capacity: Scheduler action bound.
    :param observation_capacity: Non-authoritative observation bound.
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
        output_capacity=64,
        scheduler_capacity=scheduler_capacity,
        coordinator_capacity=16,
        lifecycle_capacity=16,
        source_work_capacity=16,
        decode_work_capacity=16,
        publisher_capacity=16,
        observation_capacity=observation_capacity,
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

    with selectors.DefaultSelector() as selector:
        selector.register(inbox.fileno(), selectors.EVENT_READ)
        if len(selector.select(_WAIT_SECONDS)) == 0:
            raise TimeoutError("terminal runtime action inbox did not wake")
    actions = inbox.drain()
    if len(actions) == 0:
        raise RuntimeError("terminal runtime wake carried no action")
    return actions


def _drain_observations(runtime: NativeTerminalRuntime) -> None:
    """Drain every retained non-authoritative observation.

    :param runtime: Runtime being prepared for exact clean close.
    """

    if runtime.observations.snapshot().queued_count > 0:
        runtime.observations.drain()


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
    gather = _drain_actions(runtime.source_work_actions)
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
            1,
        ),
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
    )
    scheduler = _drain_actions(runtime.scheduler_actions)
    publisher = _drain_actions(runtime.publisher_actions)
    assert scheduler[0].kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED
    assert publisher[0].kind is (
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY
    )
    runtime.complete_scheduler_action(
        _OWNER_RECEIPT_PRODUCER_ID,
        scheduler[0],
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
        publisher[0],
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
        _drain_observations(runtime)
        runtime.join_producers()
        with pytest.raises(NativeTerminalRuntimeClosedError):
            runtime.submit(
                _LOCAL_PRODUCER_ID,
                registration.binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_OWNER_DIED,
                reason="producer submitted after join",
            )
        runtime.close_clean()
    finally:
        runtime.abort_and_close()


def test_runtime_routes_decode_work_adoption_and_request_coordination() -> None:
    runtime, owner, remote = _runtime(TerminalOwnerRole.DECODE)
    registration = _registration(owner, remote, 75)
    binding_digest = registration.binding.digest
    runtime.start()
    try:
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
        scatter = _drain_actions(runtime.decode_work_actions)
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
        assert teardown[0].kind is (NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY)
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
        snapshot = runtime.snapshot()
        assert snapshot.scheduler_live_count == 1
        assert snapshot.scheduler_pending_count == 1
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
        assert local_ready[0].kind is NativeTerminalOwnerActionKind.LOCAL_DECODE_READY
        runtime.acknowledge_consumed_action(local_ready[0])
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
        lifecycle = _drain_actions(runtime.lifecycle_actions)
        assert lifecycle[0].kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED
        runtime.acknowledge_consumed_action(lifecycle[0])
        snapshot = runtime.snapshot()
        assert snapshot.scheduler_live_count == 0
        assert snapshot.scheduler_pending_count == 0
        assert snapshot.consumer_pending_count == 0
        runtime.stop_admission()
        runtime.join_producers()
        _drain_observations(runtime)
        runtime.close_clean()
    finally:
        runtime.abort_and_close()


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
        runtime.join_producers()
        runtime.close_clean()
    finally:
        runtime.abort_and_close()


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
            gather = _drain_actions(runtime.source_work_actions)[0]
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
        assert snapshot.output_reactor_alive is False
    finally:
        runtime.abort_and_close()
