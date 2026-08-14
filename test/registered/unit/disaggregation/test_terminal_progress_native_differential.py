import concurrent.futures
import dataclasses
import errno
import sys
import time

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEventKind,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalLifecycleSnapshot,
    NativeTerminalOwner,
    NativeTerminalOwnerInventory,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NATIVE_SOURCE_RECLAIMABLE_MASK,
    NativeDecodeLifecyclePhase,
    NativeSourceLifecyclePhase,
    NativeTerminalDeadlineKind,
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalPublicationIdentity,
    NativeTerminalRequestBinding,
    NativeTerminalResource,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.terminal_progress_native_differential import (
    NativeDeadlineArmContext,
    NativeDifferentialPathError,
    NativeDifferentialResult,
    evaluate_native_deadline_boundary,
    evaluate_native_differential_case,
    evaluate_native_post_publication_quarantine_request_failure,
    evaluate_native_publisher_death_blast_radius,
    native_deadline_arm_contexts,
)
from sglang.test.terminal_progress_native_oracle import (
    OracleTransitionCase,
    decode_oracle_paths,
    exhaustive_decode_transition_cases,
    exhaustive_source_transition_cases,
    make_oracle_event,
    receipt_attack_cases,
    source_oracle_paths,
)

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="native terminal owner requires Linux eventfd and timerfd",
)

_DYNAMIC_PRODUCER_ID = 97
_DYNAMIC_WAIT_SECONDS = 2.0
_DYNAMIC_PROCESS_GENERATION = bytes.fromhex("11223344556677889900aabbccddeeff")
_DEADLINE_STARTED_NS = 5_000_000_003


def test_real_native_owner_matches_all_526_reachable_state_event_pairs() -> None:
    cases = (
        *exhaustive_source_transition_cases(),
        *exhaustive_decode_transition_cases(),
    )
    results: list[NativeDifferentialResult] = []
    excluded_process_fatal_paths: dict[str, str] = {}
    for case in cases:
        try:
            results.append(evaluate_native_differential_case(case))
        except NativeDifferentialPathError as error:
            excluded_process_fatal_paths.setdefault(case.path.name, str(error))

    assert len(results) == 480
    assert excluded_process_fatal_paths == {
        "source-quarantined-owner-death": (
            "source-quarantined-owner-death ends in process-fatal state and "
            "cannot accept a candidate event"
        ),
        "source-publication-quarantined-publisher-death": (
            "source-publication-quarantined-publisher-death ends in "
            "process-fatal state and cannot accept a candidate event"
        ),
        "decode-quarantined-owner-death": (
            "decode-quarantined-owner-death ends in process-fatal state and "
            "cannot accept a candidate event"
        ),
    }
    _assert_no_mismatches(tuple(results))


def test_real_native_owner_rejects_every_receipt_authority_attack() -> None:
    results = tuple(
        evaluate_native_differential_case(case) for case in receipt_attack_cases()
    )

    assert len(results) == 9
    _assert_no_mismatches(results)


def test_owner_minted_local_failures_match_canonical_failure_semantics() -> None:
    source_frozen = next(
        path for path in source_oracle_paths() if path.name == "source-frozen"
    )
    decode_prepared = next(
        path for path in decode_oracle_paths() if path.name == "decode-prepared"
    )
    decode_published = next(
        path for path in decode_oracle_paths() if path.name == "decode-published"
    )
    source_exhausted = next(
        path
        for path in source_oracle_paths()
        if path.name == "source-publication-quarantined-after-reclaim"
    )
    cases = (
        OracleTransitionCase(
            name="local-source-request-failed-from-frozen",
            path=source_frozen,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_FAILED,
                receipt_key="local-source-failure",
            ),
        ),
        OracleTransitionCase(
            name="local-source-request-failed-with-no-live-storage",
            path=source_exhausted,
            event=make_oracle_event(
                SourceLifecycleEventKind.REQUEST_FAILED,
                receipt_key="local-source-exhausted-failure",
            ),
        ),
        OracleTransitionCase(
            name="local-decode-request-failed-from-prepared",
            path=decode_prepared,
            event=make_oracle_event(
                DecodeLifecycleEventKind.REQUEST_FAILED,
                receipt_key="local-decode-prepared-failure",
            ),
        ),
        OracleTransitionCase(
            name="local-decode-request-failed-from-published",
            path=decode_published,
            event=make_oracle_event(
                DecodeLifecycleEventKind.REQUEST_FAILED,
                receipt_key="local-decode-published-failure",
            ),
        ),
    )
    results = tuple(
        evaluate_native_differential_case(
            case,
            owner_minted_local_failure=True,
        )
        for case in cases
    )

    _assert_no_mismatches(results)
    assert results[0].expected.accepted
    assert not results[1].expected.accepted
    assert not results[2].expected.accepted
    assert results[3].expected.accepted


def test_publisher_death_is_process_fatal_with_phase_exact_blast_radius() -> None:
    frozen = next(
        path for path in source_oracle_paths() if path.name == "source-frozen"
    )
    ready = next(path for path in source_oracle_paths() if path.name == "source-ready")
    cases = tuple(
        OracleTransitionCase(
            name=f"{path.name}-publisher-death-blast-radius",
            path=path,
            event=make_oracle_event(SourceLifecycleEventKind.PUBLISHER_DIED),
        )
        for path in (frozen, ready)
    )
    before_ready, after_ready = tuple(
        evaluate_native_differential_case(case) for case in cases
    )

    _assert_no_mismatches((before_ready, after_ready))
    assert before_ready.observed.after.phase == int(
        NativeSourceLifecyclePhase.QUARANTINED
    )
    assert before_ready.observed.after.live_resources == 0
    assert after_ready.observed.after.phase == int(
        NativeSourceLifecyclePhase.PUBLICATION_QUARANTINED
    )
    assert after_ready.observed.after.live_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
    assert after_ready.observed.after.quarantined_resources == int(
        NativeTerminalResource.PUBLICATION_IDENTITY
    )
    for result in (before_ready, after_ready):
        assert result.observed.fatal_code is (
            NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        )
        assert result.observed.actions == (NativeTerminalOwnerActionKind.PROCESS_FATAL,)
        assert result.observed.output_previous_phases == (result.observed.before.phase,)
        assert result.expected.after.process_fatal
    assert before_ready.expected.after.phase == SourceLifecyclePhase.QUARANTINED.value
    assert (
        after_ready.expected.after.phase
        == SourceLifecyclePhase.PUBLICATION_QUARANTINED.value
    )


def test_publisher_death_quarantines_only_publication_for_ready_siblings() -> None:
    blast = evaluate_native_publisher_death_blast_radius()

    assert blast.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
    publication = int(NativeTerminalResource.PUBLICATION_IDENTITY)
    for snapshot in (
        blast.target,
        blast.reclaimed_sibling,
        blast.live_sibling,
    ):
        assert snapshot.phase == int(NativeSourceLifecyclePhase.PUBLICATION_QUARANTINED)
        assert snapshot.quarantined_resources == publication
        assert snapshot.process_fatal
    assert blast.target.live_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
    assert blast.live_sibling.live_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
    assert blast.reclaimed_sibling.live_resources == 0
    assert blast.reclaimed_sibling.retired_resources == (NATIVE_SOURCE_RECLAIMABLE_MASK)


def test_request_failure_after_publication_quarantine_is_notification_idempotent() -> (
    None
):
    result = evaluate_native_post_publication_quarantine_request_failure()

    assert len(result.mismatches) == 0
    assert result.expected.accepted
    assert result.observed.before.quarantined_resources == int(
        NativeTerminalResource.PUBLICATION_IDENTITY
    )
    assert result.observed.after.live_resources == 0
    assert result.observed.after.quarantined_resources == (
        result.observed.before.live_resources
        | result.observed.before.quarantined_resources
    )
    assert result.observed.actions == (
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
    )


def test_active_qualification_abort_is_bounded_and_closes_the_reactor() -> None:
    identity = NativeTerminalProcessIdentity.from_identity(
        TerminalProcessIdentity(
            process_generation=b"q" * 16,
            role=TerminalOwnerRole.SOURCE,
            tp_rank=0,
            tp_size=1,
        )
    )
    owner = NativeTerminalOwner(
        input_capacity=64,
        output_capacity=64,
        observation_capacity=64,
        owner_identity=identity,
        testing=True,
    )
    owner.start()
    owner.start_qualification(
        machine_count=16,
        minimum_duration_seconds=60.0,
        minimum_transition_count=100_000,
    )

    started = time.monotonic()
    owner.abort_active_qualification_for_testing()

    assert time.monotonic() - started < 2.0
    inventory = owner.inventory()
    assert inventory.closed
    assert inventory.queued_input_count == 0


def test_native_deadline_contexts_cover_every_current_role_specific_arm() -> None:
    contexts = native_deadline_arm_contexts()

    assert len(contexts) == 10
    assert tuple(context.kind for context in contexts) == (
        NativeTerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER,
        NativeTerminalDeadlineKind.OWNER_NATIVE_TRANSFER,
        NativeTerminalDeadlineKind.OWNER_DECODE_SCATTER,
        NativeTerminalDeadlineKind.OWNER_TEARDOWN_ACK,
        NativeTerminalDeadlineKind.OWNER_TEARDOWN_ACK,
        NativeTerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
        NativeTerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
        NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
        NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
        NativeTerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
    )
    assert NativeTerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN not in {
        context.kind for context in contexts
    }


@pytest.mark.parametrize(
    "context",
    native_deadline_arm_contexts(),
    ids=lambda context: context.name,
)
@pytest.mark.parametrize(
    "boundary_delta_ns",
    (-1, 0, 1),
    ids=("before", "at", "after"),
)
def test_native_deadline_expiry_is_exact_and_preserves_lifetime_proof(
    context: NativeDeadlineArmContext,
    boundary_delta_ns: int,
) -> None:
    expires_ns = _DEADLINE_STARTED_NS + context.duration_ns
    step = evaluate_native_deadline_boundary(
        context,
        started_ns=_DEADLINE_STARTED_NS,
        now_ns=expires_ns + boundary_delta_ns,
    )

    assert step.before.armed_deadline_mask == context.armed_mask
    if boundary_delta_ns < 0:
        assert step.after == step.before
        assert step.actions == ()
        assert step.receipts == ()
        assert step.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert step.output_previous_phases == ()
        return

    assert step.after.armed_deadline_mask == 0
    assert step.after.live_resources == 0
    assert step.after.retired_resources == step.before.retired_resources
    assert step.after.quarantined_resources == (
        step.before.quarantined_resources | step.before.live_resources
    )
    assert step.receipts == ()
    assert step.output_previous_phases == (step.before.phase,)
    if context.kind < NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION:
        expected_phase = int(NativeSourceLifecyclePhase.QUARANTINED)
        if context.role is TerminalOwnerRole.DECODE:
            expected_phase = int(NativeDecodeLifecyclePhase.QUARANTINED)
        assert step.after.phase == expected_phase
        assert step.actions == (NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,)
        assert step.fatal_code is NativeTerminalOwnerFatalCode.NONE
        assert not step.after.process_fatal
        return

    assert step.actions == (NativeTerminalOwnerActionKind.PROCESS_FATAL,)
    assert step.fatal_code is NativeTerminalOwnerFatalCode.DEADLINE_EXPIRY
    assert step.after.process_fatal
    if context.kind is NativeTerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION:
        assert step.before.retired_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
        assert step.before.live_resources == int(
            NativeTerminalResource.PUBLICATION_IDENTITY
        )
        assert step.after.phase == int(
            NativeSourceLifecyclePhase.PUBLICATION_QUARANTINED
        )
        return
    expected_phase = int(NativeSourceLifecyclePhase.QUARANTINED)
    if context.role is TerminalOwnerRole.DECODE:
        expected_phase = int(NativeDecodeLifecyclePhase.QUARANTINED)
    assert step.after.phase == expected_phase


def test_dynamic_registration_and_first_event_share_reactor_order() -> None:
    owner, identity = _make_dynamic_source_owner(start=True)
    try:
        first = _make_dynamic_source_registration(identity, room_id=801)
        owner.register_lifecycle(first)
        _submit_dynamic_source_start(owner, first.binding)
        first_snapshot = _wait_for_source_phase(
            owner,
            first.binding.digest,
            NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
        )

        second = _make_dynamic_source_registration(identity, room_id=802)
        owner.register_lifecycle(second)
        _submit_dynamic_source_start(owner, second.binding)
        second_snapshot = _wait_for_source_phase(
            owner,
            second.binding.digest,
            NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
        )

        assert first_snapshot.binding_digest == first.binding.digest
        assert second_snapshot.binding_digest == second.binding.digest
        inventory = owner.inventory()
        assert inventory.active_source_count == 2
        assert inventory.transition_count == 2
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        owner.abort_and_close()


def test_producer_retirement_is_ordered_behind_every_accepted_event() -> None:
    owner, identity = _make_dynamic_source_owner(start=False)
    registration = _make_dynamic_source_registration(identity, room_id=807)
    try:
        owner.register_lifecycle(registration)
        _submit_dynamic_source_start(owner, registration.binding)
        owner.retire_python_producer(_DYNAMIC_PRODUCER_ID)
        retired_fence_inventory = owner.inventory()

        with pytest.raises(OSError) as duplicate_retirement:
            owner.retire_python_producer(_DYNAMIC_PRODUCER_ID)
        assert duplicate_retirement.value.errno == errno.EALREADY
        assert owner.inventory() == retired_fence_inventory

        with pytest.raises(OSError) as post_retirement_event:
            _submit_dynamic_source_start(owner, registration.binding)
        assert post_retirement_event.value.errno == errno.ESHUTDOWN
        assert owner.inventory() == retired_fence_inventory

        owner.start()
        snapshot = _wait_for_source_phase(
            owner,
            registration.binding.digest,
            NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
        )

        assert snapshot.live_resources != 0
        assert owner.wait_for_producer_retirement(
            _DYNAMIC_PRODUCER_ID,
            _DYNAMIC_WAIT_SECONDS,
        )
        assert owner.join_producers()
        inventory = owner.inventory()
        assert inventory.transition_count == 1
        assert inventory.joined_producer_count == 1
        assert not inventory.event_admission_open
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        owner.abort_and_close()


def test_concurrent_producers_receive_gap_free_queue_insertion_order() -> None:
    owner, identity = _make_dynamic_source_owner(start=False)
    registrations = tuple(
        _make_dynamic_source_registration(identity, room_id=900 + index)
        for index in range(31)
    )
    try:
        for registration in registrations:
            owner.register_lifecycle(registration)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = tuple(
                executor.submit(
                    _submit_dynamic_source_start,
                    owner,
                    registration.binding,
                )
                for registration in registrations
            )
            for future in futures:
                future.result()

        owner.retire_python_producer(_DYNAMIC_PRODUCER_ID)
        owner.start()
        for registration in registrations:
            _wait_for_source_phase(
                owner,
                registration.binding.digest,
                NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
            )

        assert owner.wait_for_producer_retirement(
            _DYNAMIC_PRODUCER_ID,
            _DYNAMIC_WAIT_SECONDS,
        )
        assert owner.join_producers()
        inventory = owner.inventory()
        assert inventory.transition_count == len(registrations)
        assert inventory.joined_producer_count == 1
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        owner.abort_and_close()


def test_unknown_lifecycle_snapshot_uses_public_key_error_contract() -> None:
    owner, _ = _make_dynamic_source_owner(start=True)
    try:
        with pytest.raises(KeyError, match="snapshot binding is unknown"):
            owner.lifecycle_snapshot_for_testing(b"\xff" * 32)
    finally:
        owner.abort_and_close()


def test_event_ordered_before_registration_fails_closed_as_unknown_binding() -> None:
    owner, identity = _make_dynamic_source_owner(start=False)
    registration = _make_dynamic_source_registration(identity, room_id=803)
    try:
        _submit_dynamic_source_start(owner, registration.binding)
        owner.register_lifecycle(registration)
        owner.start()

        inventory = _wait_for_fatal_and_empty(
            owner,
            NativeTerminalOwnerFatalCode.UNKNOWN_BINDING,
        )
        assert inventory.fatal_binding_digest == registration.binding.digest
        assert not inventory.admission_open
        assert not owner.wait_for_lifecycle_registration(
            registration.binding.digest,
            _DYNAMIC_WAIT_SECONDS,
        )
    finally:
        owner.abort_and_close()


def test_exact_dynamic_registration_duplicate_is_process_fatal() -> None:
    owner, identity = _make_dynamic_source_owner(start=True)
    registration = _make_dynamic_source_registration(identity, room_id=804)
    try:
        owner.register_lifecycle(registration)
        assert owner.wait_for_lifecycle_registration(
            registration.binding.digest,
            _DYNAMIC_WAIT_SECONDS,
        )

        owner.register_lifecycle(registration)
        inventory = _wait_for_fatal_and_empty(
            owner,
            NativeTerminalOwnerFatalCode.DUPLICATE_BINDING,
        )
        snapshot = owner.lifecycle_snapshot_for_testing(registration.binding.digest)

        assert inventory.fatal_binding_digest == registration.binding.digest
        assert snapshot.process_fatal
        assert snapshot.live_resources == 0
        assert snapshot.quarantined_resources == (
            NATIVE_SOURCE_RECLAIMABLE_MASK
            | int(NativeTerminalResource.PUBLICATION_IDENTITY)
        )
    finally:
        owner.abort_and_close()


def test_dynamic_registration_digest_collision_is_process_fatal() -> None:
    owner, identity = _make_dynamic_source_owner(start=True)
    registered = _make_dynamic_source_registration(identity, room_id=805)
    candidate = _make_dynamic_source_registration(identity, room_id=806)
    collision = dataclasses.replace(
        candidate,
        binding=dataclasses.replace(
            candidate.binding,
            digest=registered.binding.digest,
        ),
    )
    try:
        owner.register_lifecycle(registered)
        assert owner.wait_for_lifecycle_registration(
            registered.binding.digest,
            _DYNAMIC_WAIT_SECONDS,
        )

        owner.register_lifecycle(collision)
        inventory = _wait_for_fatal_and_empty(
            owner,
            NativeTerminalOwnerFatalCode.DUPLICATE_BINDING,
        )
        snapshot = owner.lifecycle_snapshot_for_testing(registered.binding.digest)

        assert inventory.fatal_binding_digest == registered.binding.digest
        assert inventory.quarantined_count == 1
        assert snapshot.process_fatal
        assert snapshot.binding_digest == registered.binding.digest
        assert snapshot.live_resources == 0
    finally:
        owner.abort_and_close()


def _make_dynamic_source_owner(
    *, start: bool
) -> tuple[NativeTerminalOwner, NativeTerminalProcessIdentity]:
    """Create one process-lifetime source owner with a local producer.

    :param start: Whether to start the reactor before returning.
    :returns: Native owner and its exact process identity.
    """

    process_identity = TerminalProcessIdentity(
        process_generation=_DYNAMIC_PROCESS_GENERATION,
        role=TerminalOwnerRole.SOURCE,
        tp_rank=0,
        tp_size=1,
    )
    identity = NativeTerminalProcessIdentity.from_identity(process_identity)
    owner = NativeTerminalOwner(
        input_capacity=64,
        output_capacity=64,
        observation_capacity=64,
        owner_identity=identity,
        testing=True,
    )
    owner.register_producer(
        NativeTerminalProducerRegistration(
            producer_id=_DYNAMIC_PRODUCER_ID,
            name="dynamic-registration-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=NativeTerminalOwnerRole.SOURCE,
            authenticated_issuer=None,
        )
    )
    if start:
        owner.start()
    return owner, identity


def _make_dynamic_source_registration(
    identity: NativeTerminalProcessIdentity,
    *,
    room_id: int,
) -> NativeTerminalLifecycleRegistration:
    """Create one source lifecycle bound to a process-lifetime owner.

    :param identity: Exact owner process identity.
    :param room_id: Stable request room identity.
    :returns: Complete dynamic source registration.
    """

    request_generation = room_id.to_bytes(16, "big")
    request_key = PackedRequestKey(
        room_id=room_id,
        request_generation=request_generation,
    )
    binding = TerminalRequestBinding(
        request_key=request_key,
        owner=TerminalProcessIdentity(
            process_generation=identity.process_generation,
            role=TerminalOwnerRole.SOURCE,
            tp_rank=identity.tp_rank,
            tp_size=identity.tp_size,
        ),
        rank_manifest_digest=b"r" * 32,
        allocation_digest=room_id.to_bytes(32, "big"),
    )
    publication = TerminalPublicationIdentity(
        request_key=request_key,
        publisher_process_generation=identity.process_generation,
        publication_generation=(room_id + 1_000).to_bytes(16, "big"),
    )
    return NativeTerminalLifecycleRegistration(
        binding=NativeTerminalRequestBinding.from_binding(binding),
        publication_identity=NativeTerminalPublicationIdentity.from_identity(
            publication
        ),
        trusted_issuers=(identity,),
    )


def _submit_dynamic_source_start(
    owner: NativeTerminalOwner,
    binding: NativeTerminalRequestBinding,
) -> None:
    """Submit the first event for one dynamically admitted source lifecycle.

    :param owner: Process-lifetime native owner.
    :param binding: Exact target lifecycle binding.
    """

    owner.submit(
        NativeTerminalOwnerEvent(
            producer_id=_DYNAMIC_PRODUCER_ID,
            binding_digest=binding.digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            enqueued_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        )
    )


def _wait_for_source_phase(
    owner: NativeTerminalOwner,
    binding_digest: bytes,
    phase: NativeSourceLifecyclePhase,
) -> NativeTerminalLifecycleSnapshot:
    """Wait for one actionless native source commit.

    :param owner: Process-lifetime native owner.
    :param binding_digest: Exact request-generation digest.
    :param phase: Expected committed source phase.
    :returns: Exact native lifecycle snapshot.
    """

    deadline = time.monotonic() + _DYNAMIC_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            snapshot = owner.lifecycle_snapshot_for_testing(binding_digest)
        except KeyError:
            time.sleep(0.001)
            continue
        if snapshot.phase == int(phase):
            return snapshot
        time.sleep(0.001)
    raise TimeoutError(f"native lifecycle did not reach {phase.name}")


def _wait_for_fatal_and_empty(
    owner: NativeTerminalOwner,
    fatal_code: NativeTerminalOwnerFatalCode,
) -> NativeTerminalOwnerInventory:
    """Wait for a sticky native fatal disposition and drained input queue.

    :param owner: Process-lifetime native owner.
    :param fatal_code: Expected sticky fatal code.
    :returns: Complete fatal native inventory.
    """

    deadline = time.monotonic() + _DYNAMIC_WAIT_SECONDS
    while time.monotonic() < deadline:
        inventory = owner.inventory()
        if inventory.fatal_code is fatal_code and inventory.queued_input_count == 0:
            return inventory
        time.sleep(0.001)
    raise TimeoutError(f"native owner did not reach {fatal_code.name}")


def _assert_no_mismatches(results: tuple[NativeDifferentialResult, ...]) -> None:
    """Render every mismatch instead of hiding behind the first state pair.

    :param results: Complete native differential evaluations.
    """

    mismatches = tuple(
        f"{result.case.name}: {message}"
        for result in results
        for message in result.mismatches
    )
    assert len(mismatches) == 0, "\n" + "\n".join(mismatches)
