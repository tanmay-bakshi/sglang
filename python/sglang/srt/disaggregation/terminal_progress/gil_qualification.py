import dataclasses
import enum
import itertools
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.gil_qualification_native import (
    GILNativeHopTrace,
)

GIL_QUALIFICATION_LIVE_MACHINE_COUNT = 16
GIL_QUALIFICATION_MINIMUM_TRANSITIONS = 100_000
GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS = 60.0
GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS = 0.005
GIL_QUALIFICATION_OWNER_HOP_COUNT = 7
GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS = 2_000_000
GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS = 16_000_000


class GILQualificationProducer(enum.StrEnum):
    """Execution context producing owner work during qualification."""

    NATIVE_OR_GIL_RELEASING = "native_or_gil_releasing"
    PYTHON_THREAD = "python_thread"


class GILSchedulerPressure(enum.StrEnum):
    """Scheduler-side contention applied throughout qualification."""

    GIL_HOGGING_PYTHON_THREAD = "gil_hogging_python_thread"


@dataclasses.dataclass(frozen=True, slots=True)
class GILQualificationConfig:
    """Frozen sustained-event-storm qualification contract.

    :ivar live_machine_count: Concurrent closed-loop lifecycle machines.
    :ivar closed_loop_replacement: Whether retirement immediately admits a
        replacement lifecycle.
    :ivar minimum_transition_count: Minimum owner transitions observed.
    :ivar minimum_duration_seconds: Minimum sustained exercise duration.
    :ivar switch_interval_seconds: Process-global Python thread switch interval.
    :ivar owner_hop_count: Correlated owner transitions in one completion path.
    :ivar per_hop_p99_limit_ns: Maximum p99 for every individual owner hop.
    :ivar seven_hop_p99_limit_ns: Maximum p99 for the correlated full path.
    """

    live_machine_count: int = GIL_QUALIFICATION_LIVE_MACHINE_COUNT
    closed_loop_replacement: bool = True
    minimum_transition_count: int = GIL_QUALIFICATION_MINIMUM_TRANSITIONS
    minimum_duration_seconds: float = GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS
    switch_interval_seconds: float = GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS
    owner_hop_count: int = GIL_QUALIFICATION_OWNER_HOP_COUNT
    per_hop_p99_limit_ns: int = GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS
    seven_hop_p99_limit_ns: int = GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS

    def __post_init__(self) -> None:
        """Reject any silent mutation of the frozen qualification contract."""

        exact_values = (
            (
                self.live_machine_count,
                GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
                "live_machine_count",
            ),
            (
                self.minimum_transition_count,
                GIL_QUALIFICATION_MINIMUM_TRANSITIONS,
                "minimum_transition_count",
            ),
            (
                self.owner_hop_count,
                GIL_QUALIFICATION_OWNER_HOP_COUNT,
                "owner_hop_count",
            ),
            (
                self.per_hop_p99_limit_ns,
                GIL_QUALIFICATION_PER_HOP_P99_LIMIT_NS,
                "per_hop_p99_limit_ns",
            ),
            (
                self.seven_hop_p99_limit_ns,
                GIL_QUALIFICATION_SEVEN_HOP_P99_LIMIT_NS,
                "seven_hop_p99_limit_ns",
            ),
        )
        for observed, expected, label in exact_values:
            if type(observed) is not int or observed != expected:
                raise ValueError(f"{label} is frozen at {expected}")
        if self.closed_loop_replacement is not True:
            raise ValueError("closed_loop_replacement is frozen as enabled")

        exact_float_values = (
            (
                self.minimum_duration_seconds,
                GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS,
                "minimum_duration_seconds",
            ),
            (
                self.switch_interval_seconds,
                GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS,
                "switch_interval_seconds",
            ),
        )
        for observed, expected, label in exact_float_values:
            if type(observed) is not float or observed != expected:
                raise ValueError(f"{label} is frozen at {expected}")


@dataclasses.dataclass(frozen=True, slots=True)
class GILStressPlan:
    """Native-free description of one GIL-contention qualification run.

    A Python-thread producer may exercise state and arithmetic, but cannot
    qualify enqueue production because it shares the scheduler's contested GIL.
    Authoritative production requires native code or a GIL-releasing extension.

    :ivar config: Exact frozen qualification contract.
    :ivar producer: Work-production implementation under qualification.
    :ivar scheduler_pressure: Synthetic scheduler load held during the storm.
    """

    config: GILQualificationConfig
    producer: GILQualificationProducer
    scheduler_pressure: GILSchedulerPressure = (
        GILSchedulerPressure.GIL_HOGGING_PYTHON_THREAD
    )

    def __post_init__(self) -> None:
        """Validate one qualification plan."""

        if type(self.config) is not GILQualificationConfig:
            raise TypeError("config must be GILQualificationConfig")
        if type(self.producer) is not GILQualificationProducer:
            raise TypeError("producer must be GILQualificationProducer")
        if (
            self.scheduler_pressure
            is not GILSchedulerPressure.GIL_HOGGING_PYTHON_THREAD
        ):
            raise ValueError("scheduler_pressure must be the frozen GIL-hogging thread")

    @property
    def authoritative_producer(self) -> bool:
        """Return whether enqueue production can yield authoritative evidence.

        :returns: Whether production is native or explicitly GIL releasing.
        """

        return self.producer is GILQualificationProducer.NATIVE_OR_GIL_RELEASING


@dataclasses.dataclass(frozen=True, slots=True)
class GILHopLatencySample:
    """One correlated seven-hop owner completion path.

    :ivar machine_index: Closed-loop machine producing the sample.
    :ivar generation_index: Gap-free request generation within the machine.
    :ivar hop_latencies_ns: Ordered latency for every owner transition.
    """

    machine_index: int
    generation_index: int
    hop_latencies_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate one complete correlated latency sample."""

        if (
            type(self.machine_index) is not int
            or self.machine_index < 0
            or self.machine_index >= GIL_QUALIFICATION_LIVE_MACHINE_COUNT
        ):
            raise ValueError("machine_index must identify one frozen live machine")
        if type(self.generation_index) is not int or self.generation_index < 0:
            raise ValueError("generation_index must be a non-negative integer")
        if type(self.hop_latencies_ns) is not tuple:
            raise TypeError("hop_latencies_ns must be a tuple")
        if len(self.hop_latencies_ns) != GIL_QUALIFICATION_OWNER_HOP_COUNT:
            raise ValueError(
                "hop_latencies_ns must contain exactly "
                f"{GIL_QUALIFICATION_OWNER_HOP_COUNT} values"
            )
        for latency_ns in self.hop_latencies_ns:
            if type(latency_ns) is not int or latency_ns < 0:
                raise ValueError("hop_latencies_ns must contain non-negative integers")

    @property
    def full_path_latency_ns(self) -> int:
        """Return the correlated seven-hop path latency.

        :returns: Sum of every owner hop in this sample.
        """

        return sum(self.hop_latencies_ns)


def _p99_nearest_rank(values: tuple[int, ...]) -> int:
    """Return deterministic nearest-rank p99 for non-empty integer values.

    :param values: Non-empty immutable latency sample.
    :returns: Nearest-rank 99th percentile.
    """

    if type(values) is not tuple or len(values) == 0:
        raise ValueError("values must be a non-empty tuple")
    ordered = tuple(sorted(values))
    rank = (99 * len(ordered) + 99) // 100
    return ordered[rank - 1]


@dataclasses.dataclass(frozen=True, slots=True)
class GILQualificationResult:
    """Computed population, latency, and authority verdict.

    :ivar plan: Exact plan used to collect the samples.
    :ivar elapsed_seconds: Sustained event-storm wall duration.
    :ivar sample_count: Correlated full-path sample count.
    :ivar transition_count: Total individual owner transitions.
    :ivar observed_machine_indices: Closed-loop machines represented.
    :ivar per_hop_p99_ns: P99 latency independently for all seven hops.
    :ivar seven_hop_p99_ns: P99 of correlated full-path latency.
    """

    plan: GILStressPlan
    elapsed_seconds: float
    sample_count: int
    transition_count: int
    observed_machine_indices: frozenset[int]
    per_hop_p99_ns: tuple[int, ...]
    seven_hop_p99_ns: int

    def __post_init__(self) -> None:
        """Validate result arithmetic and immutable population evidence."""

        if type(self.plan) is not GILStressPlan:
            raise TypeError("plan must be GILStressPlan")
        if type(self.elapsed_seconds) is not float or self.elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be a positive float")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if (
            type(self.transition_count) is not int
            or self.transition_count
            != self.sample_count * self.plan.config.owner_hop_count
        ):
            raise ValueError("transition_count must equal sample_count times hops")
        if type(self.observed_machine_indices) is not frozenset:
            raise TypeError("observed_machine_indices must be a frozenset")
        if (
            len(self.observed_machine_indices) == 0
            or len(self.observed_machine_indices) > self.sample_count
        ):
            raise ValueError("observed_machine_indices must cover one or more samples")
        for machine_index in self.observed_machine_indices:
            if (
                type(machine_index) is not int
                or machine_index < 0
                or machine_index >= self.plan.config.live_machine_count
            ):
                raise ValueError("observed machine index is outside the plan")
        if type(self.per_hop_p99_ns) is not tuple:
            raise TypeError("per_hop_p99_ns must be a tuple")
        if len(self.per_hop_p99_ns) != self.plan.config.owner_hop_count:
            raise ValueError("per_hop_p99_ns must report every owner hop")
        for latency_ns in self.per_hop_p99_ns:
            if type(latency_ns) is not int or latency_ns < 0:
                raise ValueError("per_hop_p99_ns values must be non-negative")
        if type(self.seven_hop_p99_ns) is not int or self.seven_hop_p99_ns < 0:
            raise ValueError("seven_hop_p99_ns must be a non-negative integer")

    @property
    def population_complete(self) -> bool:
        """Return whether duration, volume, and machine coverage are complete.

        :returns: Whether the frozen stress population was observed.
        """

        config = self.plan.config
        return (
            self.elapsed_seconds >= config.minimum_duration_seconds
            and self.transition_count >= config.minimum_transition_count
            and len(self.observed_machine_indices) == config.live_machine_count
        )

    @property
    def latency_within_bounds(self) -> bool:
        """Return whether individual and correlated p99 limits pass.

        :returns: Whether all frozen latency inequalities hold.
        """

        return (
            max(self.per_hop_p99_ns) <= self.plan.config.per_hop_p99_limit_ns
            and self.seven_hop_p99_ns <= self.plan.config.seven_hop_p99_limit_ns
        )

    @property
    def qualified(self) -> bool:
        """Return the complete authoritative qualification verdict.

        :returns: Whether authority, population, and latency all pass.
        """

        return (
            self.plan.authoritative_producer
            and self.population_complete
            and self.latency_within_bounds
        )


def evaluate_gil_qualification(
    plan: GILStressPlan,
    samples: tuple[GILHopLatencySample, ...],
    elapsed_seconds: float,
) -> GILQualificationResult:
    """Compute frozen p99 and authority arithmetic from collected samples.

    :param plan: Exact sustained-event-storm qualification plan.
    :param samples: Correlated owner-hop latency samples.
    :param elapsed_seconds: Sustained event-storm wall duration.
    :returns: Immutable qualification result.
    """

    if type(plan) is not GILStressPlan:
        raise TypeError("plan must be GILStressPlan")
    if type(samples) is not tuple or len(samples) == 0:
        raise ValueError("samples must be a non-empty tuple")
    if type(elapsed_seconds) is not float or elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be a positive float")
    for sample in samples:
        if type(sample) is not GILHopLatencySample:
            raise TypeError("samples entries must be GILHopLatencySample")
    samples_by_machine: dict[int, list[GILHopLatencySample]] = {}
    for sample in samples:
        samples_by_machine.setdefault(sample.machine_index, []).append(sample)
    for machine_samples in samples_by_machine.values():
        observed_generations = sorted(
            sample.generation_index for sample in machine_samples
        )
        if observed_generations != list(range(len(observed_generations))):
            raise ValueError("sample generations must be unique and gap-free")

    hop_p99_ns = tuple(
        _p99_nearest_rank(
            tuple(sample.hop_latencies_ns[hop_index] for sample in samples)
        )
        for hop_index in range(plan.config.owner_hop_count)
    )
    seven_hop_p99_ns = _p99_nearest_rank(
        tuple(sample.full_path_latency_ns for sample in samples)
    )
    return GILQualificationResult(
        plan=plan,
        elapsed_seconds=elapsed_seconds,
        sample_count=len(samples),
        transition_count=len(samples) * plan.config.owner_hop_count,
        observed_machine_indices=frozenset(sample.machine_index for sample in samples),
        per_hop_p99_ns=hop_p99_ns,
        seven_hop_p99_ns=seven_hop_p99_ns,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class GILStressCollection:
    """Raw evidence returned by a concrete event-storm producer.

    :ivar samples: Correlated latency samples observed during the storm.
    :ivar native_hop_traces: Raw native enqueue and committed-dispatch evidence.
    :ivar elapsed_seconds: Measured sustained storm duration.
    """

    samples: tuple[GILHopLatencySample, ...]
    native_hop_traces: tuple[GILNativeHopTrace, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        """Validate one non-empty stress population."""

        if type(self.samples) is not tuple or len(self.samples) == 0:
            raise ValueError("samples must be a non-empty tuple")
        if any(type(sample) is not GILHopLatencySample for sample in self.samples):
            raise TypeError("samples must contain GILHopLatencySample values")
        if type(self.native_hop_traces) is not tuple:
            raise TypeError("native_hop_traces must be a tuple")
        if any(
            type(trace) is not GILNativeHopTrace for trace in self.native_hop_traces
        ):
            raise TypeError("native_hop_traces must contain GILNativeHopTrace values")
        if len(self.native_hop_traces) > 0:
            correlated = correlate_gil_native_traces(self.native_hop_traces)
            if correlated != self.samples:
                raise ValueError("native traces do not reproduce correlated samples")
        if type(self.elapsed_seconds) is not float or self.elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be a positive float")


def correlate_gil_native_traces(
    traces: tuple[GILNativeHopTrace, ...],
) -> tuple[GILHopLatencySample, ...]:
    """Validate and correlate raw native traces into seven-hop requests.

    The validation proves global source order, all 16 machine identities,
    gap-free request generations and hop indices, and the one-outstanding-event
    closed-loop invariant. No aggregate percentile can repair a malformed raw
    population.

    :param traces: Native enqueue and post-dispatch completion evidence.
    :returns: Deterministically ordered correlated request samples.
    """

    if type(traces) is not tuple or len(traces) == 0:
        raise ValueError("traces must be a non-empty tuple")
    if any(type(trace) is not GILNativeHopTrace for trace in traces):
        raise TypeError("traces must contain GILNativeHopTrace values")
    ordered = tuple(sorted(traces, key=lambda trace: trace.event.producer_sequence))
    sequences = tuple(trace.event.producer_sequence for trace in ordered)
    if sequences != tuple(range(len(ordered))):
        raise ValueError("native producer sequences must be gap-free from zero")

    grouped: dict[tuple[int, int], list[GILNativeHopTrace]] = {}
    machine_traces: dict[int, list[GILNativeHopTrace]] = {}
    for trace in ordered:
        event = trace.event
        if event.machine_index >= GIL_QUALIFICATION_LIVE_MACHINE_COUNT:
            raise ValueError("native trace machine is outside the frozen population")
        if event.hop_index >= GIL_QUALIFICATION_OWNER_HOP_COUNT:
            raise ValueError("native trace hop is outside the frozen request path")
        grouped.setdefault((event.machine_index, event.generation_index), []).append(
            trace
        )
        machine_traces.setdefault(event.machine_index, []).append(trace)

    if set(machine_traces) != set(range(GIL_QUALIFICATION_LIVE_MACHINE_COUNT)):
        raise ValueError("native traces must cover exactly all frozen machines")
    for machine_index, current_traces in machine_traces.items():
        generations = sorted({trace.event.generation_index for trace in current_traces})
        if generations != list(range(len(generations))):
            raise ValueError(
                f"machine {machine_index} generations must be gap-free from zero"
            )
        ordered_machine = sorted(
            current_traces,
            key=lambda trace: (
                trace.event.generation_index,
                trace.event.hop_index,
            ),
        )
        for previous, current in itertools.pairwise(ordered_machine):
            if current.event.enqueued_ns < previous.completed_ns:
                raise ValueError(
                    "a machine published a successor before dispatch acknowledgment"
                )

    samples: list[GILHopLatencySample] = []
    for (machine_index, generation_index), current_traces in sorted(grouped.items()):
        ordered_hops = tuple(
            sorted(current_traces, key=lambda trace: trace.event.hop_index)
        )
        if tuple(trace.event.hop_index for trace in ordered_hops) != tuple(
            range(GIL_QUALIFICATION_OWNER_HOP_COUNT)
        ):
            raise ValueError("each native generation must contain seven exact hops")
        samples.append(
            GILHopLatencySample(
                machine_index=machine_index,
                generation_index=generation_index,
                hop_latencies_ns=tuple(trace.latency_ns for trace in ordered_hops),
            )
        )
    return tuple(samples)


GILStressCollector = Callable[[GILStressPlan], GILStressCollection]


def execute_gil_stress_plan(
    plan: GILStressPlan,
    collector: GILStressCollector,
) -> GILQualificationResult:
    """Execute an injected producer and evaluate its complete population.

    This is deliberately only orchestration. A Python collector remains
    non-authoritative even if its arithmetic passes. Native or GIL-releasing
    implementations declare that production mode in the frozen plan and must
    furnish the actual correlated samples.

    :param plan: Exact frozen stress contract and producer authority.
    :param collector: Concrete sustained event-storm implementation.
    :returns: Evaluated population, latency, and authority verdict.
    """

    if type(plan) is not GILStressPlan:
        raise TypeError("plan must be GILStressPlan")
    if not callable(collector):
        raise TypeError("collector must be callable")
    collection = collector(plan)
    if type(collection) is not GILStressCollection:
        raise TypeError("collector must return GILStressCollection")
    if plan.authoritative_producer and len(collection.native_hop_traces) == 0:
        raise ValueError("authoritative production requires raw native hop traces")
    return evaluate_gil_qualification(
        plan=plan,
        samples=collection.samples,
        elapsed_seconds=collection.elapsed_seconds,
    )
