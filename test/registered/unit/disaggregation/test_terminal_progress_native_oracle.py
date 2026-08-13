import itertools

from sglang.srt.disaggregation.terminal_progress.deadlines import (
    PACKED_TERMINAL_DEADLINES,
    TerminalDeadlineKind,
    start_terminal_deadline,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DECODE_RESOURCE_KINDS,
    SOURCE_RECLAIMABLE_RESOURCES,
    SOURCE_RESOURCE_KINDS,
    DecodeLifecycleEventKind,
    DecodeLifecyclePhase,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
    TerminalResourceKind,
)
from sglang.srt.disaggregation.terminal_progress.receipts import TerminalReceiptKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.terminal_progress_native_oracle import (
    OracleDeadlineOutcome,
    OracleOwnerAction,
    OracleReductionError,
    deadline_boundary_cases,
    decode_oracle_paths,
    evaluate_oracle_path,
    evaluate_oracle_transition,
    exhaustive_decode_transition_cases,
    exhaustive_source_transition_cases,
    receipt_attack_cases,
    source_oracle_paths,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_source_oracle_matrix_covers_every_reachable_state_event_pair() -> None:
    paths = source_oracle_paths()
    cases = exhaustive_source_transition_cases()

    assert len(cases) == len(paths) * len(SourceLifecycleEventKind)
    assert {(case.path.name, case.event.kind) for case in cases} == set(
        itertools.product(
            (path.name for path in paths),
            tuple(SourceLifecycleEventKind),
        )
    )

    evaluations = tuple(evaluate_oracle_transition(case) for case in cases)
    assert sum(evaluation.accepted for evaluation in evaluations) == 71
    assert sum(not evaluation.accepted for evaluation in evaluations) == 199
    assert (
        sum(
            evaluation.error is OracleReductionError.STATE_INVARIANT
            for evaluation in evaluations
        )
        == 1
    )
    assert all(
        evaluation.before.role is TerminalOwnerRole.SOURCE
        and evaluation.after.role is TerminalOwnerRole.SOURCE
        for evaluation in evaluations
    )


def test_process_fatal_events_preserve_exhausted_source_dispositions() -> None:
    fatal_kinds = frozenset(
        (
            SourceLifecycleEventKind.OWNER_DIED,
            SourceLifecycleEventKind.PUBLISHER_DIED,
            SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
        )
    )
    evaluations = tuple(
        evaluate_oracle_transition(case)
        for case in exhaustive_source_transition_cases()
        if case.path.name == "source-publication-quarantined-after-reclaim"
        and case.event.kind in fatal_kinds
    )

    assert len(evaluations) == len(fatal_kinds)
    assert {evaluation.case.event.kind for evaluation in evaluations} == fatal_kinds
    for evaluation in evaluations:
        assert evaluation.accepted
        assert not evaluation.before.process_fatal
        assert evaluation.after.process_fatal
        assert evaluation.after.phase == evaluation.before.phase
        assert evaluation.after.live_resources == evaluation.before.live_resources
        assert evaluation.after.retired_resources == evaluation.before.retired_resources
        assert (
            evaluation.after.quarantined_resources
            == evaluation.before.quarantined_resources
        )
        assert evaluation.actions == (
            OracleOwnerAction.STATE_COMMITTED,
            OracleOwnerAction.PROCESS_FATAL,
        )
        assert evaluation.emitted_receipts == ()
        assert evaluation.error is None
        assert evaluation.error_message is None


def test_decode_oracle_matrix_covers_every_reachable_state_event_pair() -> None:
    paths = decode_oracle_paths()
    cases = exhaustive_decode_transition_cases()

    assert len(cases) == len(paths) * len(DecodeLifecycleEventKind)
    assert {(case.path.name, case.event.kind) for case in cases} == set(
        itertools.product(
            (path.name for path in paths),
            tuple(DecodeLifecycleEventKind),
        )
    )

    evaluations = tuple(evaluate_oracle_transition(case) for case in cases)
    assert sum(evaluation.accepted for evaluation in evaluations) == 48
    assert sum(not evaluation.accepted for evaluation in evaluations) == 208
    assert all(
        evaluation.before.role is TerminalOwnerRole.DECODE
        and evaluation.after.role is TerminalOwnerRole.DECODE
        for evaluation in evaluations
    )


def test_source_join_orders_reach_the_same_fully_retired_state() -> None:
    paths = {path.name: evaluate_oracle_path(path) for path in source_oracle_paths()}
    reclaim_first = paths["source-retired-reclaim-then-publish"].states[-1]
    publish_first = paths["source-retired-publish-then-reclaim"].states[-1]

    assert reclaim_first == publish_first
    assert reclaim_first.phase == SourceLifecyclePhase.RETIRED.value
    assert reclaim_first.live_resources == frozenset()
    assert reclaim_first.retired_resources == SOURCE_RESOURCE_KINDS
    assert reclaim_first.quarantined_resources == frozenset()

    reclaim_intermediate = paths["source-ready-reclaim-consumed"].states[-1]
    assert (
        reclaim_intermediate.phase == SourceLifecyclePhase.REQUEST_READY_RECEIVED.value
    )
    assert reclaim_intermediate.retired_resources == SOURCE_RECLAIMABLE_RESOURCES
    assert reclaim_intermediate.live_resources == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )

    publication_intermediate = paths["source-ready-gateway-published"].states[-1]
    assert (
        publication_intermediate.phase
        == SourceLifecyclePhase.REQUEST_READY_RECEIVED.value
    )
    assert publication_intermediate.retired_resources == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert publication_intermediate.live_resources == SOURCE_RECLAIMABLE_RESOURCES


def test_decode_path_separates_receipt_authority_from_local_metadata_ack() -> None:
    path_by_name = {path.name: path for path in decode_oracle_paths()}
    adoption_case = next(
        case
        for case in exhaustive_decode_transition_cases()
        if case.path.name == "decode-adoption_ready"
        and case.event.kind is DecodeLifecycleEventKind.ADOPTION_CONSUMED
    )
    metadata_case = next(
        case
        for case in exhaustive_decode_transition_cases()
        if case.path.name == "decode-adopted-by-scheduler"
        and case.event.kind is DecodeLifecycleEventKind.METADATA_CONSUMED
    )
    ready_case = next(
        case
        for case in exhaustive_decode_transition_cases()
        if case.path.name == "decode-local-ready"
        and case.event.kind is DecodeLifecycleEventKind.REQUEST_READY_RECEIVED
    )

    adoption = evaluate_oracle_transition(adoption_case)
    metadata = evaluate_oracle_transition(metadata_case)
    ready = evaluate_oracle_transition(ready_case)

    assert adoption.accepted
    assert adoption.after.phase == DecodeLifecyclePhase.ADOPTED_BY_SCHEDULER.value
    assert metadata.accepted
    assert metadata.after.phase == DecodeLifecyclePhase.METADATA_CONSUMED.value
    assert ready.accepted
    assert ready.after.phase == DecodeLifecyclePhase.RETIRED.value
    assert ready.after.retired_resources == DECODE_RESOURCE_KINDS
    assert (
        evaluate_oracle_path(path_by_name["decode-retired-request-ready"]).states[-1]
        == ready.after
    )


def test_owner_action_and_receipt_oracle_marks_exact_authority_boundaries() -> None:
    source_ready = next(
        evaluate_oracle_transition(case)
        for case in exhaustive_source_transition_cases()
        if case.path.name == "source-ack_sent"
        and case.event.kind is SourceLifecycleEventKind.REQUEST_READY_RECEIVED
    )
    decode_adoption = next(
        evaluate_oracle_transition(case)
        for case in exhaustive_decode_transition_cases()
        if case.path.name == "decode-ack_aggregating"
        and case.event.kind is DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED
    )
    decode_local_ready = next(
        evaluate_oracle_transition(case)
        for case in exhaustive_decode_transition_cases()
        if case.path.name == "decode-metadata-consumed"
        and case.event.kind is DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED
    )

    assert source_ready.actions == (
        OracleOwnerAction.STATE_COMMITTED,
        OracleOwnerAction.RECLAIM_AUTHORIZED,
        OracleOwnerAction.GATEWAY_PUBLICATION_READY,
    )
    assert source_ready.emitted_receipts == (TerminalReceiptKind.RECLAIM_AUTHORIZED,)
    assert decode_adoption.actions == (
        OracleOwnerAction.STATE_COMMITTED,
        OracleOwnerAction.ADOPTION_READY,
    )
    assert decode_adoption.emitted_receipts == (TerminalReceiptKind.ADOPTION_READY,)
    assert decode_local_ready.actions == (
        OracleOwnerAction.STATE_COMMITTED,
        OracleOwnerAction.LOCAL_DECODE_READY,
    )
    assert decode_local_ready.emitted_receipts == (
        TerminalReceiptKind.LOCAL_DECODE_READY,
    )


def test_receipt_attacks_are_rejected_without_state_or_side_effect_changes() -> None:
    evaluations = tuple(
        evaluate_oracle_transition(case) for case in receipt_attack_cases()
    )

    assert len(evaluations) == 9
    assert all(not evaluation.accepted for evaluation in evaluations)
    assert all(
        evaluation.error is OracleReductionError.RECEIPT for evaluation in evaluations
    )
    assert all(evaluation.after == evaluation.before for evaluation in evaluations)
    assert all(len(evaluation.actions) == 0 for evaluation in evaluations)
    assert all(len(evaluation.emitted_receipts) == 0 for evaluation in evaluations)
    replay = next(
        evaluation
        for evaluation in evaluations
        if evaluation.case.name == "source-failure-receipt-replay"
    )
    assert replay.error_message is not None
    assert "already consumed" in replay.error_message


def test_fatal_events_quarantine_all_live_resources_and_mark_process_fatal() -> None:
    source = next(
        evaluate_oracle_path(path).states[-1]
        for path in source_oracle_paths()
        if path.name == "source-quarantined-owner-death"
    )
    decode = next(
        evaluate_oracle_path(path).states[-1]
        for path in decode_oracle_paths()
        if path.name == "decode-quarantined-owner-death"
    )
    publisher = next(
        evaluate_oracle_path(path).states[-1]
        for path in source_oracle_paths()
        if path.name == "source-publication-quarantined-publisher-death"
    )

    assert source.process_fatal
    assert source.quarantined_resources == SOURCE_RESOURCE_KINDS
    assert decode.process_fatal
    assert decode.quarantined_resources == DECODE_RESOURCE_KINDS
    assert publisher.process_fatal
    assert publisher.phase == SourceLifecyclePhase.PUBLICATION_QUARANTINED.value
    assert publisher.quarantined_resources == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert publisher.live_resources == SOURCE_RECLAIMABLE_RESOURCES


def test_deadline_oracle_covers_every_frozen_timer_at_the_exact_boundary() -> None:
    cases = deadline_boundary_cases(started_ns=5_000_000_003)

    assert len(cases) == len(PACKED_TERMINAL_DEADLINES)
    assert {case.kind for case in cases} == set(TerminalDeadlineKind)
    for case in cases:
        deadline = start_terminal_deadline(case.kind, case.started_ns)
        assert not deadline.expired(case.before_expiry_ns)
        assert deadline.expired(case.expires_ns)
        assert deadline.expired(case.after_expiry_ns)

    process_fatal = {
        case.kind
        for case in cases
        if case.outcome is OracleDeadlineOutcome.PROCESS_FATAL
    }
    assert process_fatal == {
        TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
        TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
        TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
    }
