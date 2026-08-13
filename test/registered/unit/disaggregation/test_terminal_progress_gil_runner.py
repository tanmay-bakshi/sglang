import json

import pytest
from sglang.srt.disaggregation.terminal_progress.gil_qualification_runner import (
    GILQualificationExecutionConfig,
    GILQualificationRunMode,
    canonical_json_bytes,
    require_frozen_switch_interval,
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
