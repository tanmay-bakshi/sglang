import dataclasses
import math

import pytest
from sglang.srt.disaggregation.terminal_progress.gil_qualification import (
    GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
    GIL_QUALIFICATION_MINIMUM_TRANSITIONS,
    GIL_QUALIFICATION_OWNER_HOP_COUNT,
    GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS,
    GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS,
    GILHopLatencySample,
    GILQualificationConfig,
    GILQualificationProducer,
    GILSchedulerPressure,
    GILStressCollection,
    GILStressPlan,
    evaluate_gil_qualification,
    execute_gil_stress_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _complete_samples(
    latency_ns: int,
) -> tuple[GILHopLatencySample, ...]:
    """Build the smallest population satisfying the transition volume.

    :param latency_ns: Latency assigned to every owner hop.
    :returns: Complete correlated sample population.
    """

    sample_count = math.ceil(
        GIL_QUALIFICATION_MINIMUM_TRANSITIONS / GIL_QUALIFICATION_OWNER_HOP_COUNT
    )
    return tuple(
        GILHopLatencySample(
            machine_index=index % GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
            hop_latencies_ns=(latency_ns,) * GIL_QUALIFICATION_OWNER_HOP_COUNT,
        )
        for index in range(sample_count)
    )


def test_contract_values_are_exact_and_immutable() -> None:
    """The scaffold rejects silent changes to every frozen field."""

    config = GILQualificationConfig()
    assert config.live_machine_count == 16
    assert config.closed_loop_replacement
    assert config.minimum_transition_count == 100_000
    assert config.minimum_duration_seconds == 60.0
    assert config.switch_interval_seconds == 0.005
    assert config.per_hop_p99_limit_ns == 2_000_000
    assert config.seven_hop_p99_limit_ns == 16_000_000

    with pytest.raises(ValueError):
        GILQualificationConfig(live_machine_count=15)
    with pytest.raises(ValueError):
        GILQualificationConfig(minimum_transition_count=99_999)
    with pytest.raises(ValueError):
        GILQualificationConfig(minimum_duration_seconds=59.0)
    with pytest.raises(ValueError):
        GILQualificationConfig(switch_interval_seconds=0.001)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.live_machine_count = 8  # type: ignore[misc]


def test_python_producer_can_exercise_arithmetic_but_never_qualifies() -> None:
    """A GIL-contending Python feeder cannot prove its own enqueue latency."""

    plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.PYTHON_THREAD,
    )
    result = evaluate_gil_qualification(
        plan=plan,
        samples=_complete_samples(latency_ns=1_000_000),
        elapsed_seconds=60.0,
    )

    assert plan.scheduler_pressure is GILSchedulerPressure.GIL_HOGGING_PYTHON_THREAD
    assert not plan.authoritative_producer
    assert result.population_complete
    assert result.latency_within_bounds
    assert not result.qualified

    authoritative = dataclasses.replace(
        result,
        plan=dataclasses.replace(
            plan,
            producer=GILQualificationProducer.NATIVE_OR_GIL_RELEASING,
        ),
    )
    assert authoritative.plan.authoritative_producer
    assert authoritative.qualified


def test_population_and_both_latency_bounds_are_independent() -> None:
    """Duration, machine coverage, per-hop p99, and path p99 each matter."""

    plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.NATIVE_OR_GIL_RELEASING,
    )
    incomplete = evaluate_gil_qualification(
        plan=plan,
        samples=(
            GILHopLatencySample(
                machine_index=0,
                hop_latencies_ns=(1_000_000,) * 7,
            ),
        ),
        elapsed_seconds=1.0,
    )
    assert not incomplete.population_complete
    assert incomplete.latency_within_bounds
    assert not incomplete.qualified

    per_hop_failure = evaluate_gil_qualification(
        plan=plan,
        samples=tuple(
            GILHopLatencySample(
                machine_index=index % 16,
                hop_latencies_ns=(
                    GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS + 1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
            )
            for index in range(100)
        ),
        elapsed_seconds=60.0,
    )
    assert not per_hop_failure.latency_within_bounds

    path_failure = evaluate_gil_qualification(
        plan=plan,
        samples=tuple(
            GILHopLatencySample(
                machine_index=index % 16,
                hop_latencies_ns=(2_000_000,) * 7,
            )
            for index in range(100)
        ),
        elapsed_seconds=60.0,
    )
    assert path_failure.seven_hop_p99_ns <= (GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS)
    assert path_failure.latency_within_bounds

    distributed_tail_samples: list[GILHopLatencySample] = []
    for index in range(1_000):
        hop_latencies = [1_000_000] * 7
        if index < 70:
            hop_latencies[index // 10] = 20_000_000
        distributed_tail_samples.append(
            GILHopLatencySample(
                machine_index=index % 16,
                hop_latencies_ns=tuple(hop_latencies),
            )
        )
    over_path = evaluate_gil_qualification(
        plan=plan,
        samples=tuple(distributed_tail_samples),
        elapsed_seconds=60.0,
    )
    assert max(over_path.per_hop_p99_ns) <= (GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS)
    assert over_path.seven_hop_p99_ns > (GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS)
    assert not over_path.latency_within_bounds


def test_nearest_rank_p99_and_sample_validation_are_deterministic() -> None:
    """Two slowest values place the 99th nearest-rank sample at the tail."""

    plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.PYTHON_THREAD,
    )
    samples = tuple(
        GILHopLatencySample(
            machine_index=index % 16,
            hop_latencies_ns=((10_000_000 if index >= 98 else 1_000_000),) * 7,
        )
        for index in range(100)
    )
    result = evaluate_gil_qualification(
        plan=plan,
        samples=samples,
        elapsed_seconds=1.0,
    )
    assert result.per_hop_p99_ns == (10_000_000,) * 7
    assert result.seven_hop_p99_ns == 70_000_000

    with pytest.raises(ValueError):
        GILHopLatencySample(machine_index=16, hop_latencies_ns=(1,) * 7)
    with pytest.raises(ValueError):
        GILHopLatencySample(machine_index=0, hop_latencies_ns=(1,) * 6)
    with pytest.raises(ValueError):
        evaluate_gil_qualification(plan=plan, samples=(), elapsed_seconds=1.0)


def test_executable_scaffold_preserves_collector_authority() -> None:
    """A concrete collector cannot upgrade a Python feeder to authority."""

    samples = _complete_samples(latency_ns=1_000_000)
    python_plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.PYTHON_THREAD,
    )

    def collect(plan: GILStressPlan) -> GILStressCollection:
        """Return deterministic samples while preserving the supplied plan.

        :param plan: Exact frozen stress plan.
        :returns: Complete deterministic sample population.
        """

        assert plan is python_plan
        return GILStressCollection(samples=samples, elapsed_seconds=60.0)

    result = execute_gil_stress_plan(python_plan, collect)
    assert result.population_complete
    assert result.latency_within_bounds
    assert not result.qualified


def test_executable_scaffold_accepts_native_collector_results() -> None:
    """A native producer can furnish the real population without API shims."""

    native_plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.NATIVE_OR_GIL_RELEASING,
    )
    result = execute_gil_stress_plan(
        native_plan,
        lambda plan: GILStressCollection(
            samples=_complete_samples(latency_ns=1_000_000),
            elapsed_seconds=plan.config.minimum_duration_seconds,
        ),
    )
    assert result.qualified

    with pytest.raises(TypeError):
        execute_gil_stress_plan(native_plan, lambda plan: object())  # type: ignore[arg-type,return-value]
