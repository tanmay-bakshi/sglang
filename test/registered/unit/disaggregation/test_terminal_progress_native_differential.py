import sys

import pytest
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycleEventKind,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NATIVE_SOURCE_RECLAIMABLE_MASK,
    NativeSourceLifecyclePhase,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalResource,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.terminal_progress_native_differential import (
    NativeDifferentialPathError,
    NativeDifferentialResult,
    evaluate_native_differential_case,
    evaluate_native_post_publication_quarantine_request_failure,
    evaluate_native_publisher_death_blast_radius,
)
from sglang.test.terminal_progress_native_oracle import (
    OracleLifecyclePath,
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
    ready = next(
        path for path in source_oracle_paths() if path.name == "source-ready"
    )
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
        assert result.observed.actions == (
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        )
        assert result.observed.output_previous_phases == (
            result.observed.before.phase,
        )
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
        assert snapshot.phase == int(
            NativeSourceLifecyclePhase.PUBLICATION_QUARANTINED
        )
        assert snapshot.quarantined_resources == publication
        assert snapshot.process_fatal
    assert blast.target.live_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
    assert blast.live_sibling.live_resources == NATIVE_SOURCE_RECLAIMABLE_MASK
    assert blast.reclaimed_sibling.live_resources == 0
    assert blast.reclaimed_sibling.retired_resources == (
        NATIVE_SOURCE_RECLAIMABLE_MASK
    )


def test_request_failure_after_publication_quarantine_is_notification_idempotent(
) -> None:
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
    assert result.observed.actions == ()


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
