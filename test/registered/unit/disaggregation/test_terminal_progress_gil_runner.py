import json
from pathlib import Path
from typing import cast

import pytest
from sglang.srt.disaggregation.terminal_progress.gil_qualification_runner import (
    GILQualificationExecutionConfig,
    GILQualificationRunMode,
    NativeGILQualificationSummary,
    canonical_json_bytes,
    prepare_gil_qualification_output_root,
    require_frozen_switch_interval,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeSourceLifecyclePhase,
    NativeTerminalOwnerEventKind,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_execution_populations_and_switch_interval_are_frozen() -> None:
    """Neither a short run nor a caller can mutate authoritative semantics."""

    functional = GILQualificationExecutionConfig.for_mode(
        GILQualificationRunMode.FUNCTIONAL
    )
    authoritative = GILQualificationExecutionConfig.for_mode(
        GILQualificationRunMode.AUTHORITATIVE
    )
    assert functional.minimum_duration_seconds == 5.0
    assert functional.minimum_transition_count == 10_080
    assert authoritative.minimum_duration_seconds == 60.0
    assert authoritative.minimum_transition_count == 100_000
    require_frozen_switch_interval(0.005)

    with pytest.raises(ValueError, match="bounds are frozen"):
        GILQualificationExecutionConfig(
            mode=GILQualificationRunMode.AUTHORITATIVE,
            minimum_duration_seconds=59.0,
            minimum_transition_count=100_000,
            timeout_seconds=300.0,
        )
    with pytest.raises(RuntimeError, match="switch interval"):
        require_frozen_switch_interval(0.001)


def test_canonical_receipt_serialization_is_byte_deterministic() -> None:
    """Mapping insertion order cannot alter receipt or checksum bytes."""

    first = canonical_json_bytes({"z": [3, 2, 1], "a": {"b": True}})
    second = canonical_json_bytes({"a": {"b": True}, "z": [3, 2, 1]})
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == {"a": {"b": True}, "z": [3, 2, 1]}


def test_artifact_writer_accepts_concrete_path_subclasses(tmp_path: Path) -> None:
    """The platform's concrete ``PosixPath`` is a valid artifact root."""

    output_root = tmp_path / "evidence"
    prepare_gil_qualification_output_root(output_root)
    assert output_root.is_dir()

    (output_root / "existing").write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_gil_qualification_output_root(output_root)


def test_native_summary_validates_population_and_bounded_audit_evidence() -> None:
    """Native aggregates retain exact population and first/last audit proof."""

    summary = NativeGILQualificationSummary.from_native(_native_summary())

    assert summary.machine_count == 16
    assert summary.sample_count == 16
    assert summary.transition_count == 112
    assert summary.lifecycle_transition_count == 160
    assert summary.elapsed_seconds == 60.0
    assert len(summary.first_audit_samples) == 112
    assert len(summary.last_audit_samples) == 112
    assert summary.transition_classes[0].event_kind is (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED
    )
    assert summary.seven_hop_path.p99_ns == 7_000


def test_native_summary_rejects_population_and_audit_forgery() -> None:
    """Aggregate counts cannot disagree with the bounded native audit trail."""

    population = _native_summary()
    population["transition_count"] = 111
    with pytest.raises(ValueError, match="measured transitions"):
        NativeGILQualificationSummary.from_native(population)

    audit = _native_summary()
    first = cast(list[dict[str, object]], audit["first_audit_samples"])
    first[-1] = dict(first[0])
    with pytest.raises(ValueError, match="duplicated or incomplete"):
        NativeGILQualificationSummary.from_native(audit)


def _native_summary() -> dict[str, object]:
    """Build one minimal conservative native-summary mapping.

    :returns: Complete native summary with sixteen one-request machines.
    """

    event_kinds = (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
        NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
        NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
    )
    phases = (
        (
            NativeSourceLifecyclePhase.FROZEN,
            NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
        ),
        (
            NativeSourceLifecyclePhase.WAITING_FOR_PRODUCER,
            NativeSourceLifecyclePhase.GATHERING,
        ),
        (
            NativeSourceLifecyclePhase.GATHERING,
            NativeSourceLifecyclePhase.NATIVE_IN_FLIGHT,
        ),
        (
            NativeSourceLifecyclePhase.NATIVE_IN_FLIGHT,
            NativeSourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL,
        ),
        (
            NativeSourceLifecyclePhase.LOCAL_TRANSFER_TERMINAL,
            NativeSourceLifecyclePhase.OUTCOMES_SENT,
        ),
        (
            NativeSourceLifecyclePhase.OUTCOMES_SENT,
            NativeSourceLifecyclePhase.TEARDOWN_RECEIVED,
        ),
        (
            NativeSourceLifecyclePhase.TEARDOWN_RECEIVED,
            NativeSourceLifecyclePhase.ACK_SENT,
        ),
    )
    transition_classes = [
        {
            "hop_index": hop_index,
            "event_kind": int(event_kind),
            **_latency_statistics(count=16, latency_ns=1_000),
        }
        for hop_index, event_kind in enumerate(event_kinds)
    ]
    first_audit_samples: list[dict[str, object]] = []
    last_audit_samples: list[dict[str, object]] = []
    for machine_index in range(16):
        for hop_index, event_kind in enumerate(event_kinds):
            previous_phase, phase = phases[hop_index]
            record = {
                "machine_index": machine_index,
                "generation_index": 0,
                "hop_index": hop_index,
                "binding_digest": bytes((machine_index, 0)) + b"a" * 30,
                "event_kind": int(event_kind),
                "previous_phase": int(previous_phase),
                "phase": int(phase),
                "enqueued_ns": 1_000 + hop_index * 2_000,
                "completed_ns": 2_000 + hop_index * 2_000,
            }
            first_audit_samples.append(record)
            last_audit_samples.append(dict(record))
    return {
        "machine_count": 16,
        "measured_hop_count": 7,
        "lifecycle_hop_count": 10,
        "statistics_sample_capacity": 8_000_000,
        "minimum_duration_ns": 60_000_000_000,
        "minimum_transition_count": 100_000,
        "started_ns": 1_000_000_000,
        "ended_ns": 61_000_000_000,
        "transition_count": 112,
        "lifecycle_transition_count": 160,
        "sample_count": 16,
        "owner_sequence_start": 0,
        "owner_sequence_end": 160,
        "raw_trace_retained_count": 0,
        "transition_classes": transition_classes,
        "seven_hop_path": _latency_statistics(count=16, latency_ns=7_000),
        "completed_generations_by_machine": [1] * 16,
        "producer_sequences_by_machine": [10] * 16,
        "audit_sample_bound": 224,
        "audit_sample_count": 224,
        "first_audit_samples": first_audit_samples,
        "last_audit_samples": last_audit_samples,
    }


def _latency_statistics(*, count: int, latency_ns: int) -> dict[str, int]:
    """Build one constant native latency distribution.

    :param count: Exact population size.
    :param latency_ns: Shared distribution value.
    :returns: Complete native statistics mapping.
    """

    return {
        "count": count,
        "p50_ns": latency_ns,
        "p95_ns": latency_ns,
        "p99_ns": latency_ns,
        "maximum_ns": latency_ns,
    }
