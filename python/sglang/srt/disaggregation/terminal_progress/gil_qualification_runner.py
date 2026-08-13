import argparse
import dataclasses
import enum
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

from sglang.srt.disaggregation.terminal_progress.gil_qualification import (
    GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
    GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS,
    GIL_QUALIFICATION_MINIMUM_TRANSITIONS,
    GIL_QUALIFICATION_OWNER_HOP_COUNT,
    GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS,
    GILQualificationConfig,
    GILQualificationProducer,
    GILQualificationResult,
    GILStressPlan,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeSourceLifecyclePhase,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerInventory,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalQualificationTrace,
    native_terminal_deadline_table_digest,
)

_FUNCTIONAL_DURATION_SECONDS = 5.0
_FUNCTIONAL_MINIMUM_TRANSITIONS = 10_080
_FUNCTIONAL_TIMEOUT_SECONDS = 90.0
_AUTHORITATIVE_TIMEOUT_SECONDS = 300.0
_NATIVE_QUEUE_CAPACITY = 64
_HOG_HOT_ITERATIONS = 100_000
_QUALIFICATION_LIFECYCLE_HOP_COUNT = 10
_QUALIFICATION_AUDIT_SAMPLE_COUNT = (
    GIL_QUALIFICATION_LIVE_MACHINE_COUNT * GIL_QUALIFICATION_OWNER_HOP_COUNT * 2
)
_QUALIFICATION_EVENT_KINDS = (
    NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
    NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
    NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
    NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
    NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
    NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
    NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
)
_QUALIFICATION_PHASE_TRANSITIONS = (
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


class GILQualificationRunMode(enum.StrEnum):
    """Supported executable qualification populations."""

    FUNCTIONAL = "functional"
    AUTHORITATIVE = "authoritative"


@dataclasses.dataclass(frozen=True, slots=True)
class GILQualificationExecutionConfig:
    """Concrete population bounds for one executable qualification.

    :ivar mode: Short functional or frozen authoritative population.
    :ivar minimum_duration_seconds: Native sustained-duration floor.
    :ivar minimum_transition_count: Native committed-transition floor.
    :ivar timeout_seconds: Fail-closed wall-clock execution bound.
    """

    mode: GILQualificationRunMode
    minimum_duration_seconds: float
    minimum_transition_count: int
    timeout_seconds: float

    @classmethod
    def for_mode(
        cls,
        mode: GILQualificationRunMode,
    ) -> "GILQualificationExecutionConfig":
        """Build the predeclared execution bounds for one mode.

        :param mode: Requested functional or authoritative run.
        :returns: Immutable execution configuration.
        """

        if type(mode) is not GILQualificationRunMode:
            raise TypeError("mode must be GILQualificationRunMode")
        if mode is GILQualificationRunMode.FUNCTIONAL:
            return cls(
                mode=mode,
                minimum_duration_seconds=_FUNCTIONAL_DURATION_SECONDS,
                minimum_transition_count=_FUNCTIONAL_MINIMUM_TRANSITIONS,
                timeout_seconds=_FUNCTIONAL_TIMEOUT_SECONDS,
            )
        return cls(
            mode=mode,
            minimum_duration_seconds=GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS,
            minimum_transition_count=GIL_QUALIFICATION_MINIMUM_TRANSITIONS,
            timeout_seconds=_AUTHORITATIVE_TIMEOUT_SECONDS,
        )

    def __post_init__(self) -> None:
        """Reject mutations of either predeclared population."""

        if type(self.mode) is not GILQualificationRunMode:
            raise TypeError("mode must be GILQualificationRunMode")
        if self.mode is GILQualificationRunMode.FUNCTIONAL:
            expected = (
                _FUNCTIONAL_DURATION_SECONDS,
                _FUNCTIONAL_MINIMUM_TRANSITIONS,
                _FUNCTIONAL_TIMEOUT_SECONDS,
            )
        else:
            expected = (
                GIL_QUALIFICATION_MINIMUM_DURATION_SECONDS,
                GIL_QUALIFICATION_MINIMUM_TRANSITIONS,
                _AUTHORITATIVE_TIMEOUT_SECONDS,
            )
        observed = (
            self.minimum_duration_seconds,
            self.minimum_transition_count,
            self.timeout_seconds,
        )
        if observed != expected:
            raise ValueError(f"{self.mode.value} execution bounds are frozen")


@dataclasses.dataclass(frozen=True, slots=True)
class NativeGILLatencyStatistics:
    """Native nearest-rank latency distribution for one population.

    :ivar count: Exact sample population.
    :ivar p50_ns: Nearest-rank median in nanoseconds.
    :ivar p95_ns: Nearest-rank p95 in nanoseconds.
    :ivar p99_ns: Nearest-rank p99 in nanoseconds.
    :ivar maximum_ns: Maximum observed latency in nanoseconds.
    """

    count: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    maximum_ns: int

    def __post_init__(self) -> None:
        """Validate a complete monotonic latency summary."""

        values = (
            self.count,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.maximum_ns,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("native latency statistics must be non-negative integers")
        if self.count == 0:
            raise ValueError("native latency statistics require a non-empty population")
        if not (self.p50_ns <= self.p95_ns <= self.p99_ns <= self.maximum_ns):
            raise ValueError("native latency percentiles must be monotonic")

    @classmethod
    def from_native(cls, value: Mapping[str, object]) -> "NativeGILLatencyStatistics":
        """Parse one native latency-statistics mapping.

        :param value: Raw native summary mapping.
        :returns: Validated immutable latency statistics.
        """

        return cls(
            count=int(value["count"]),
            p50_ns=int(value["p50_ns"]),
            p95_ns=int(value["p95_ns"]),
            p99_ns=int(value["p99_ns"]),
            maximum_ns=int(value["maximum_ns"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeGILTransitionStatistics:
    """Native latency distribution for one exact reducer transition.

    :ivar hop_index: Correlated hop identity within the seven-hop path.
    :ivar event_kind: Exact production event reduced at this hop.
    :ivar latency: Complete nearest-rank distribution for the hop.
    """

    hop_index: int
    event_kind: NativeTerminalOwnerEventKind
    latency: NativeGILLatencyStatistics

    def __post_init__(self) -> None:
        """Validate one exact transition-class summary."""

        if (
            type(self.hop_index) is not int
            or self.hop_index < 0
            or self.hop_index >= GIL_QUALIFICATION_OWNER_HOP_COUNT
        ):
            raise ValueError("native transition hop is outside the frozen path")
        if type(self.event_kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("event_kind must be NativeTerminalOwnerEventKind")
        if self.event_kind is not _QUALIFICATION_EVENT_KINDS[self.hop_index]:
            raise ValueError("native transition event does not match its frozen hop")
        if type(self.latency) is not NativeGILLatencyStatistics:
            raise TypeError("latency must be NativeGILLatencyStatistics")

    @classmethod
    def from_native(
        cls, value: Mapping[str, object]
    ) -> "NativeGILTransitionStatistics":
        """Parse one native transition-class mapping.

        :param value: Raw native transition summary.
        :returns: Validated immutable transition statistics.
        """

        return cls(
            hop_index=int(value["hop_index"]),
            event_kind=NativeTerminalOwnerEventKind(int(value["event_kind"])),
            latency=NativeGILLatencyStatistics.from_native(value),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeGILQualificationSummary:
    """Bounded authoritative evidence emitted by the native owner.

    :ivar machine_count: Concurrent closed-loop lifecycle machines.
    :ivar measured_hop_count: Latency-measured transitions per request.
    :ivar lifecycle_hop_count: Total reducer transitions per request.
    :ivar statistics_sample_capacity: Native bounded-statistics capacity.
    :ivar minimum_duration_ns: Requested sustained-duration floor.
    :ivar minimum_transition_count: Requested measured-transition floor.
    :ivar started_ns: Native qualification start timestamp.
    :ivar ended_ns: Native qualification completion timestamp.
    :ivar transition_count: Measured seven-hop transition population.
    :ivar lifecycle_transition_count: Complete lifecycle transition population.
    :ivar sample_count: Complete correlated request population.
    :ivar owner_sequence_start: First native owner sequence in the population.
    :ivar owner_sequence_end: Exclusive final native owner sequence.
    :ivar raw_trace_retained_count: Unbounded raw traces retained by native code.
    :ivar transition_classes: Per-hop latency distributions.
    :ivar seven_hop_path: Correlated full-path latency distribution.
    :ivar completed_generations_by_machine: Complete requests per machine.
    :ivar producer_sequences_by_machine: Native producer events per machine.
    :ivar audit_sample_bound: Maximum retained first/last audit population.
    :ivar audit_sample_count: Exact retained first/last audit population.
    :ivar first_audit_samples: First exact transition per machine and hop.
    :ivar last_audit_samples: Last exact transition per machine and hop.
    """

    machine_count: int
    measured_hop_count: int
    lifecycle_hop_count: int
    statistics_sample_capacity: int
    minimum_duration_ns: int
    minimum_transition_count: int
    started_ns: int
    ended_ns: int
    transition_count: int
    lifecycle_transition_count: int
    sample_count: int
    owner_sequence_start: int
    owner_sequence_end: int
    raw_trace_retained_count: int
    transition_classes: tuple[NativeGILTransitionStatistics, ...]
    seven_hop_path: NativeGILLatencyStatistics
    completed_generations_by_machine: tuple[int, ...]
    producer_sequences_by_machine: tuple[int, ...]
    audit_sample_bound: int
    audit_sample_count: int
    first_audit_samples: tuple[NativeTerminalQualificationTrace, ...]
    last_audit_samples: tuple[NativeTerminalQualificationTrace, ...]

    def __post_init__(self) -> None:
        """Validate population, ordering, and bounded-evidence conservation."""

        exact = (
            (self.machine_count, GIL_QUALIFICATION_LIVE_MACHINE_COUNT, "machines"),
            (
                self.measured_hop_count,
                GIL_QUALIFICATION_OWNER_HOP_COUNT,
                "measured hops",
            ),
            (
                self.lifecycle_hop_count,
                _QUALIFICATION_LIFECYCLE_HOP_COUNT,
                "lifecycle hops",
            ),
            (
                self.audit_sample_bound,
                _QUALIFICATION_AUDIT_SAMPLE_COUNT,
                "audit sample bound",
            ),
            (
                self.audit_sample_count,
                _QUALIFICATION_AUDIT_SAMPLE_COUNT,
                "audit sample count",
            ),
            (self.raw_trace_retained_count, 0, "raw retained traces"),
        )
        for observed, expected, label in exact:
            if type(observed) is not int or observed != expected:
                raise ValueError(f"native qualification {label} must equal {expected}")
        counts = (
            self.statistics_sample_capacity,
            self.minimum_duration_ns,
            self.minimum_transition_count,
            self.started_ns,
            self.ended_ns,
            self.transition_count,
            self.lifecycle_transition_count,
            self.sample_count,
            self.owner_sequence_start,
            self.owner_sequence_end,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(
                "native qualification counts must be non-negative integers"
            )
        if self.sample_count == 0 or self.ended_ns <= self.started_ns:
            raise ValueError("native qualification population must be non-empty")
        if self.minimum_duration_ns == 0 or self.minimum_transition_count == 0:
            raise ValueError("native qualification floors must be positive")
        if self.owner_sequence_start != 0:
            raise ValueError("native qualification must start from an unused owner")
        if self.statistics_sample_capacity < self.sample_count:
            raise ValueError("native qualification exceeded its statistics capacity")
        if self.transition_count != self.sample_count * self.measured_hop_count:
            raise ValueError("measured transitions do not conserve complete paths")
        if (
            self.lifecycle_transition_count
            != self.sample_count * self.lifecycle_hop_count
        ):
            raise ValueError("lifecycle transitions do not conserve requests")
        if (
            self.owner_sequence_end - self.owner_sequence_start
            != self.lifecycle_transition_count
        ):
            raise ValueError("native owner sequence does not conserve transitions")
        if len(self.transition_classes) != self.measured_hop_count:
            raise ValueError("native qualification must summarize every measured hop")
        if tuple(value.hop_index for value in self.transition_classes) != tuple(
            range(self.measured_hop_count)
        ):
            raise ValueError("native transition classes must be ordered and complete")
        if any(
            value.latency.count != self.sample_count
            for value in self.transition_classes
        ):
            raise ValueError("native transition classes have unequal populations")
        if self.seven_hop_path.count != self.sample_count:
            raise ValueError("native full-path population differs from its hops")
        self._validate_machine_populations()
        self._validate_audit_samples()

    @property
    def elapsed_seconds(self) -> float:
        """Return native sustained duration in seconds.

        :returns: Exact native interval converted to seconds.
        """

        return float(self.ended_ns - self.started_ns) / 1_000_000_000.0

    def _validate_machine_populations(self) -> None:
        """Validate closed-loop generations and producer sequences."""

        populations = (
            self.completed_generations_by_machine,
            self.producer_sequences_by_machine,
        )
        if any(len(values) != self.machine_count for values in populations):
            raise ValueError("native machine populations must cover every machine")
        if any(
            type(value) is not int or value <= 0
            for values in populations
            for value in values
        ):
            raise ValueError("native machine populations must be positive integers")
        if sum(self.completed_generations_by_machine) != self.sample_count:
            raise ValueError("per-machine generations do not conserve samples")
        expected_sequences = tuple(
            count * self.lifecycle_hop_count
            for count in self.completed_generations_by_machine
        )
        if self.producer_sequences_by_machine != expected_sequences:
            raise ValueError("native producer sequences do not conserve lifecycles")

    def _validate_audit_samples(self) -> None:
        """Validate bounded first/last evidence for every machine and hop."""

        expected_per_cohort = self.machine_count * self.measured_hop_count
        cohorts = (self.first_audit_samples, self.last_audit_samples)
        if any(len(values) != expected_per_cohort for values in cohorts):
            raise ValueError("native audit evidence must cover every machine and hop")
        for values in cohorts:
            keys = tuple((value.machine_index, value.hop_index) for value in values)
            expected = tuple(
                (machine_index, hop_index)
                for machine_index in range(self.machine_count)
                for hop_index in range(self.measured_hop_count)
            )
            if tuple(sorted(keys)) != expected:
                raise ValueError("native audit evidence is duplicated or incomplete")
            for value in values:
                if value.event_kind is not _QUALIFICATION_EVENT_KINDS[value.hop_index]:
                    raise ValueError("native audit event does not match its frozen hop")
                expected_phases = _QUALIFICATION_PHASE_TRANSITIONS[value.hop_index]
                if (value.previous_phase, value.phase) != expected_phases:
                    raise ValueError("native audit phases do not match the frozen path")
            for machine_index in range(self.machine_count):
                machine_values = tuple(
                    sorted(
                        (
                            value
                            for value in values
                            if value.machine_index == machine_index
                        ),
                        key=lambda value: value.hop_index,
                    )
                )
                if len({value.binding_digest for value in machine_values}) != 1:
                    raise ValueError("native audit path changed binding mid-request")
                for previous, current in itertools.pairwise(machine_values):
                    if current.enqueued_ns < previous.completed_ns:
                        raise ValueError(
                            "native audit successor preceded committed completion"
                        )
        if any(value.generation_index != 0 for value in self.first_audit_samples):
            raise ValueError(
                "first native audit samples must come from generation zero"
            )
        last_by_key = {
            (value.machine_index, value.hop_index): value
            for value in self.last_audit_samples
        }
        for first in self.first_audit_samples:
            last = last_by_key[(first.machine_index, first.hop_index)]
            expected_last_generation = (
                self.completed_generations_by_machine[first.machine_index] - 1
            )
            if last.generation_index != expected_last_generation:
                raise ValueError("native last audit generation differs from inventory")

    @classmethod
    def from_native(
        cls, value: Mapping[str, object]
    ) -> "NativeGILQualificationSummary":
        """Parse the complete bounded native qualification summary.

        :param value: Raw mapping returned by the native owner.
        :returns: Validated immutable qualification summary.
        """

        transition_values = _mapping_sequence(value["transition_classes"])
        first_values = _mapping_sequence(value["first_audit_samples"])
        last_values = _mapping_sequence(value["last_audit_samples"])
        path_value = value["seven_hop_path"]
        if not isinstance(path_value, Mapping):
            raise TypeError("native seven-hop path must be a mapping")
        return cls(
            machine_count=int(value["machine_count"]),
            measured_hop_count=int(value["measured_hop_count"]),
            lifecycle_hop_count=int(value["lifecycle_hop_count"]),
            statistics_sample_capacity=int(value["statistics_sample_capacity"]),
            minimum_duration_ns=int(value["minimum_duration_ns"]),
            minimum_transition_count=int(value["minimum_transition_count"]),
            started_ns=int(value["started_ns"]),
            ended_ns=int(value["ended_ns"]),
            transition_count=int(value["transition_count"]),
            lifecycle_transition_count=int(value["lifecycle_transition_count"]),
            sample_count=int(value["sample_count"]),
            owner_sequence_start=int(value["owner_sequence_start"]),
            owner_sequence_end=int(value["owner_sequence_end"]),
            raw_trace_retained_count=int(value["raw_trace_retained_count"]),
            transition_classes=tuple(
                NativeGILTransitionStatistics.from_native(current)
                for current in transition_values
            ),
            seven_hop_path=NativeGILLatencyStatistics.from_native(path_value),
            completed_generations_by_machine=_integer_tuple(
                value["completed_generations_by_machine"],
                "completed generations",
            ),
            producer_sequences_by_machine=_integer_tuple(
                value["producer_sequences_by_machine"],
                "producer sequences",
            ),
            audit_sample_bound=int(value["audit_sample_bound"]),
            audit_sample_count=int(value["audit_sample_count"]),
            first_audit_samples=tuple(
                NativeTerminalQualificationTrace.from_native(current)
                for current in first_values
            ),
            last_audit_samples=tuple(
                NativeTerminalQualificationTrace.from_native(current)
                for current in last_values
            ),
        )


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    """Return a tuple of exact mapping records.

    :param value: Candidate native sequence.
    :returns: Immutable mapping sequence.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("native summary field must be a sequence")
    records: list[Mapping[str, object]] = []
    for current in value:
        if not isinstance(current, Mapping):
            raise TypeError("native summary sequence must contain mappings")
        records.append(current)
    return tuple(records)


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    """Return one exact native integer vector.

    :param value: Candidate native sequence.
    :param label: Reader-facing field identity.
    :returns: Immutable integer tuple.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"native {label} must be a sequence")
    integers: list[int] = []
    for current in value:
        if type(current) is not int:
            raise TypeError(f"native {label} must contain integers")
        integers.append(current)
    return tuple(integers)


@dataclasses.dataclass(frozen=True, slots=True)
class GILQualificationRunReceipt:
    """Complete executable result and lifecycle attestation.

    :ivar execution: Exact functional or authoritative population.
    :ivar effective_switch_interval_seconds: Verified process switch interval.
    :ivar summary: Bounded native owner statistics and audit evidence.
    :ivar result: Frozen qualification arithmetic over the native summary.
    :ivar owner_identity: Exact native owner process identity.
    :ivar native_inventory: Complete native inventory before close.
    :ivar native_closed_inventory: Native inventory after exact closure.
    :ivar scheduler_hog_iterations: Pure-Python scheduler work executed.
    :ivar source_hashes: Hash-bound implementation files used by the run.
    :ivar git_revision: Exact serving revision under qualification.
    :ivar hostname: Environment hostname carrying the evidence.
    :ivar python_version: Python runtime identity.
    """

    execution: GILQualificationExecutionConfig
    effective_switch_interval_seconds: float
    summary: NativeGILQualificationSummary
    result: GILQualificationResult
    owner_identity: NativeTerminalProcessIdentity
    native_inventory: NativeTerminalOwnerInventory
    native_closed_inventory: NativeTerminalOwnerInventory
    scheduler_hog_iterations: int
    source_hashes: tuple[tuple[str, str], ...]
    git_revision: str
    hostname: str
    python_version: str

    def __post_init__(self) -> None:
        """Validate cross-layer conservation and clean closure."""

        if type(self.execution) is not GILQualificationExecutionConfig:
            raise TypeError("execution must be GILQualificationExecutionConfig")
        if type(self.summary) is not NativeGILQualificationSummary:
            raise TypeError("summary must be NativeGILQualificationSummary")
        if type(self.result) is not GILQualificationResult:
            raise TypeError("result must be GILQualificationResult")
        if type(self.owner_identity) is not NativeTerminalProcessIdentity:
            raise TypeError("owner_identity must be NativeTerminalProcessIdentity")
        if type(self.native_inventory) is not NativeTerminalOwnerInventory:
            raise TypeError("native_inventory must be NativeTerminalOwnerInventory")
        if type(self.native_closed_inventory) is not NativeTerminalOwnerInventory:
            raise TypeError(
                "native_closed_inventory must be NativeTerminalOwnerInventory"
            )
        if (
            type(self.effective_switch_interval_seconds) is not float
            or self.effective_switch_interval_seconds
            != GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS
        ):
            raise ValueError("effective switch interval must equal the frozen value")
        if (
            type(self.scheduler_hog_iterations) is not int
            or self.scheduler_hog_iterations <= 0
        ):
            raise ValueError("scheduler_hog_iterations must be a positive integer")
        if self.owner_identity.role is not NativeTerminalOwnerRole.SOURCE:
            raise ValueError("qualification owner must carry the source role")
        expected_duration_ns = int(
            self.execution.minimum_duration_seconds * 1_000_000_000
        )
        if self.summary.minimum_duration_ns != expected_duration_ns:
            raise ValueError("native duration floor differs from execution contract")
        if (
            self.summary.minimum_transition_count
            != self.execution.minimum_transition_count
        ):
            raise ValueError("native transition floor differs from execution contract")
        if self.result.transition_count != self.summary.transition_count:
            raise ValueError("result and native measured transitions must agree")
        if self.result.sample_count != self.summary.sample_count:
            raise ValueError("result and native sample populations must agree")
        if self.result.elapsed_seconds != self.summary.elapsed_seconds:
            raise ValueError("result and native sustained durations must agree")
        expected_hop_p99 = tuple(
            value.latency.p99_ns for value in self.summary.transition_classes
        )
        if self.result.per_hop_p99_ns != expected_hop_p99:
            raise ValueError("result and native per-hop p99 values must agree")
        if self.result.seven_hop_p99_ns != self.summary.seven_hop_path.p99_ns:
            raise ValueError("result and native full-path p99 values must agree")
        self._validate_inventory()
        if not self.native_closed_inventory.closed:
            raise ValueError("native owner must close in the final receipt")
        if type(self.source_hashes) is not tuple or len(self.source_hashes) == 0:
            raise ValueError("source_hashes must be a non-empty tuple")
        strings = (self.git_revision, self.hostname, self.python_version)
        if any(type(value) is not str or len(value) == 0 for value in strings):
            raise ValueError("receipt identity strings must be non-empty")

    def _validate_inventory(self) -> None:
        """Validate clean process-lifetime native owner closure."""

        before = self.native_inventory
        after = self.native_closed_inventory
        if before.fatal_code is not NativeTerminalOwnerFatalCode.NONE:
            raise ValueError("native qualification ended process-fatal")
        deadline_digest = native_terminal_deadline_table_digest()
        if (
            before.deadline_table_digest != deadline_digest
            or after.deadline_table_digest != deadline_digest
        ):
            raise ValueError(
                "native owner deadline table differs from the frozen table"
            )
        if before.admission_open or not before.draining or before.closed:
            raise ValueError("native pre-close inventory is outside clean drain")
        if before.joined_producer_count != before.registered_producer_count:
            raise ValueError("native qualification producers were not joined")
        zero_counts = (
            before.queued_input_count,
            before.queued_output_count,
            before.active_source_count,
            before.active_decode_count,
            before.quarantined_count,
            before.armed_deadline_count,
            before.qualification_trace_count,
        )
        if any(value != 0 for value in zero_counts):
            raise ValueError("native pre-close inventory retained unresolved work")
        if before.transition_count != self.summary.lifecycle_transition_count:
            raise ValueError("native inventory and lifecycle transitions must agree")
        if after.fatal_code is not NativeTerminalOwnerFatalCode.NONE:
            raise ValueError("native clean close cannot acquire a fatal disposition")
        conserved = (
            "registered_producer_count",
            "joined_producer_count",
            "safely_retired_count",
            "quarantined_count",
            "transition_count",
            "action_count",
            "qualification_trace_count",
        )
        if any(getattr(before, field) != getattr(after, field) for field in conserved):
            raise ValueError("native clean close mutated sealed inventory counts")

    @property
    def verdict(self) -> bool:
        """Return the mode-appropriate executable verdict.

        Functional qualification proves structure and lifecycle only. The
        authoritative mode additionally adjudicates the frozen population and
        p99 bounds.

        :returns: Whether this executable run passes its declared purpose.
        """

        if self.execution.mode is GILQualificationRunMode.FUNCTIONAL:
            return (
                self.summary.transition_count >= self.execution.minimum_transition_count
                and self.summary.elapsed_seconds
                >= self.execution.minimum_duration_seconds
            )
        return self.result.qualified


class _PurePythonSchedulerHog:
    """CPU-hot non-yielding scheduler surrogate sharing the process GIL."""

    _stop_requested: threading.Event
    _hot: threading.Event
    _thread: threading.Thread
    _iterations: int
    _accumulator: int

    def __init__(self) -> None:
        """Create the scheduler surrogate before the owner starts sampling."""

        self._stop_requested = threading.Event()
        self._hot = threading.Event()
        self._iterations = 0
        self._accumulator = 0
        self._thread = threading.Thread(
            target=self._run,
            name="packed-terminal-gil-hog",
            daemon=False,
        )

    @property
    def iterations(self) -> int:
        """Return completed pure-Python work iterations.

        :returns: Scheduler work population after join.
        """

        return self._iterations

    def start_and_wait_until_hot(self, timeout_seconds: float) -> None:
        """Start and require a nontrivial hot population before sampling.

        :param timeout_seconds: Positive setup wait bound.
        """

        self._thread.start()
        if not self._hot.wait(timeout=timeout_seconds):
            self._stop_requested.set()
            self._thread.join(timeout=timeout_seconds)
            raise TimeoutError("synthetic scheduler did not become CPU-hot")

    def stop_and_join(self, timeout_seconds: float) -> None:
        """Stop only after the native final acknowledgment and join.

        :param timeout_seconds: Positive join bound.
        """

        self._stop_requested.set()
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("synthetic scheduler did not stop")

    def _run(self) -> None:
        """Hold the GIL with non-yielding pure-Python integer work."""

        iterations = 0
        accumulator = 0x9E3779B97F4A7C15
        while not self._stop_requested.is_set():
            accumulator ^= accumulator << 7
            accumulator ^= accumulator >> 9
            accumulator ^= accumulator << 8
            accumulator &= (1 << 127) - 1
            iterations += 1
            if iterations == _HOG_HOT_ITERATIONS:
                self._hot.set()
        self._iterations = iterations
        self._accumulator = accumulator


def _implementation_hashes() -> tuple[tuple[str, str], ...]:
    """Hash the complete executable qualification implementation.

    :returns: Sorted relative paths and SHA-256 digests.
    """

    module_root = Path(__file__).resolve().parent
    paths = (
        module_root / "gil_qualification.py",
        module_root / "gil_qualification_runner.py",
        module_root / "native_owner.py",
        module_root / "native_owner_bridge.cpp",
        module_root / "native_state.py",
        module_root / "deadlines.py",
        module_root / "identity.py",
        module_root / "lifecycle.py",
        module_root / "receipts.py",
    )
    return tuple(
        sorted(
            (
                path.name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in paths
        )
    )


def require_frozen_switch_interval(effective_seconds: float) -> None:
    """Reject sampling under any process-global switch-interval drift.

    :param effective_seconds: Value read immediately before native production.
    """

    if type(effective_seconds) is not float:
        raise TypeError("effective_seconds must be a float")
    if effective_seconds != GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS:
        raise RuntimeError("Python did not retain the frozen switch interval")


def _git_revision() -> str:
    """Return the exact repository revision.

    :returns: Full Git commit hash.
    """

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[5],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _qualification_owner_identity() -> NativeTerminalProcessIdentity:
    """Create one exact process incarnation for the qualification owner.

    :returns: Native source identity recorded in the sealed receipt.
    """

    return NativeTerminalProcessIdentity.from_identity(
        TerminalProcessIdentity(
            process_generation=os.urandom(16),
            role=TerminalOwnerRole.SOURCE,
            tp_rank=0,
            tp_size=1,
        )
    )


def _result_from_summary(
    summary: NativeGILQualificationSummary,
) -> GILQualificationResult:
    """Project native aggregate evidence into the frozen verdict arithmetic.

    :param summary: Validated native owner summary.
    :returns: Frozen qualification result without Python-side resampling.
    """

    plan = GILStressPlan(
        config=GILQualificationConfig(),
        producer=GILQualificationProducer.NATIVE_OR_GIL_RELEASING,
    )
    return GILQualificationResult(
        plan=plan,
        elapsed_seconds=summary.elapsed_seconds,
        sample_count=summary.sample_count,
        transition_count=summary.transition_count,
        observed_machine_indices=frozenset(range(summary.machine_count)),
        per_hop_p99_ns=tuple(
            value.latency.p99_ns for value in summary.transition_classes
        ),
        seven_hop_p99_ns=summary.seven_hop_path.p99_ns,
    )


def run_gil_qualification(
    execution: GILQualificationExecutionConfig,
) -> GILQualificationRunReceipt:
    """Run the closed-loop storm through the production native reducer.

    :param execution: Exact predeclared functional or authoritative bounds.
    :returns: Immutable receipt with bounded native audit evidence.
    """

    if type(execution) is not GILQualificationExecutionConfig:
        raise TypeError("execution must be GILQualificationExecutionConfig")
    original_switch_interval = sys.getswitchinterval()
    owner_identity = _qualification_owner_identity()
    owner = NativeTerminalOwner(
        input_capacity=_NATIVE_QUEUE_CAPACITY,
        output_capacity=_NATIVE_QUEUE_CAPACITY,
        owner_identity=owner_identity,
    )
    hog = _PurePythonSchedulerHog()
    hog_started = False
    try:
        sys.setswitchinterval(GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS)
        effective_switch_interval = sys.getswitchinterval()
        require_frozen_switch_interval(effective_switch_interval)

        owner.start()
        hog.start_and_wait_until_hot(timeout_seconds=5.0)
        hog_started = True
        require_frozen_switch_interval(sys.getswitchinterval())

        owner.start_qualification(
            machine_count=GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
            minimum_duration_seconds=execution.minimum_duration_seconds,
            minimum_transition_count=execution.minimum_transition_count,
        )
        if not owner.qualification_join(execution.timeout_seconds):
            raise TimeoutError("native qualification population did not complete")
        hog.stop_and_join(timeout_seconds=5.0)
        hog_started = False

        summary = NativeGILQualificationSummary.from_native(
            owner.qualification_summary()
        )
        result = _result_from_summary(summary)
        owner.stop_admission()
        owner.join_producers()
        native_inventory = owner.inventory()
        owner.close()
        closed_inventory = owner.inventory()
        receipt = GILQualificationRunReceipt(
            execution=execution,
            effective_switch_interval_seconds=effective_switch_interval,
            summary=summary,
            result=result,
            owner_identity=owner_identity,
            native_inventory=native_inventory,
            native_closed_inventory=closed_inventory,
            scheduler_hog_iterations=hog.iterations,
            source_hashes=_implementation_hashes(),
            git_revision=_git_revision(),
            hostname=os.uname().nodename,
            python_version=platform.python_version(),
        )
        return receipt
    finally:
        if hog_started:
            hog.stop_and_join(timeout_seconds=5.0)
        owner.abort_and_close()
        sys.setswitchinterval(original_switch_interval)


def _inventory_dict(inventory: NativeTerminalOwnerInventory) -> dict[str, object]:
    """Return a JSON-safe native inventory mapping.

    :param inventory: Typed native lifecycle inventory.
    :returns: Canonically serializable mapping.
    """

    value = dataclasses.asdict(inventory)
    value["fatal_code"] = int(inventory.fatal_code)
    if inventory.fatal_binding_digest is not None:
        value["fatal_binding_digest"] = inventory.fatal_binding_digest.hex()
    value["deadline_table_digest"] = inventory.deadline_table_digest.hex()
    return value


def receipt_dict(receipt: GILQualificationRunReceipt) -> dict[str, object]:
    """Return the stable JSON schema for one executable receipt.

    :param receipt: Complete immutable run receipt.
    :returns: Canonically serializable mapping.
    """

    return {
        "schema": "sglang.packed-terminal-native-owner-gil-qualification.v2",
        "mode": receipt.execution.mode.value,
        "verdict": receipt.verdict,
        "execution": {
            "minimum_duration_seconds": (receipt.execution.minimum_duration_seconds),
            "minimum_transition_count": (receipt.execution.minimum_transition_count),
            "timeout_seconds": receipt.execution.timeout_seconds,
        },
        "contract": {
            "live_machine_count": receipt.result.plan.config.live_machine_count,
            "owner_hop_count": receipt.result.plan.config.owner_hop_count,
            "switch_interval_seconds": (
                receipt.result.plan.config.switch_interval_seconds
            ),
            "minimum_duration_seconds": (
                receipt.result.plan.config.minimum_duration_seconds
            ),
            "minimum_transition_count": (
                receipt.result.plan.config.minimum_transition_count
            ),
            "per_hop_p99_limit_ns": (receipt.result.plan.config.per_hop_p99_limit_ns),
            "seven_hop_p99_limit_ns": (
                receipt.result.plan.config.seven_hop_p99_limit_ns
            ),
        },
        "result": {
            "authoritative_producer": (receipt.result.plan.authoritative_producer),
            "elapsed_seconds": receipt.result.elapsed_seconds,
            "sample_count": receipt.result.sample_count,
            "transition_count": receipt.result.transition_count,
            "observed_machine_indices": sorted(receipt.result.observed_machine_indices),
            "per_hop_p99_ns": list(receipt.result.per_hop_p99_ns),
            "seven_hop_p99_ns": receipt.result.seven_hop_p99_ns,
            "population_complete": receipt.result.population_complete,
            "latency_within_bounds": receipt.result.latency_within_bounds,
            "qualified": receipt.result.qualified,
            "raw_transition_throughput_per_second": (
                receipt.result.transition_count / receipt.result.elapsed_seconds
            ),
        },
        "native_summary": {
            "machine_count": receipt.summary.machine_count,
            "measured_hop_count": receipt.summary.measured_hop_count,
            "lifecycle_hop_count": receipt.summary.lifecycle_hop_count,
            "statistics_sample_capacity": (receipt.summary.statistics_sample_capacity),
            "minimum_duration_ns": receipt.summary.minimum_duration_ns,
            "minimum_transition_count": (receipt.summary.minimum_transition_count),
            "started_ns": receipt.summary.started_ns,
            "ended_ns": receipt.summary.ended_ns,
            "transition_count": receipt.summary.transition_count,
            "lifecycle_transition_count": (receipt.summary.lifecycle_transition_count),
            "sample_count": receipt.summary.sample_count,
            "owner_sequence_start": receipt.summary.owner_sequence_start,
            "owner_sequence_end": receipt.summary.owner_sequence_end,
            "raw_trace_retained_count": receipt.summary.raw_trace_retained_count,
            "transition_classes": [
                {
                    "hop_index": value.hop_index,
                    "event_kind": int(value.event_kind),
                    **_latency_dict(value.latency),
                }
                for value in receipt.summary.transition_classes
            ],
            "seven_hop_path": _latency_dict(receipt.summary.seven_hop_path),
            "completed_generations_by_machine": list(
                receipt.summary.completed_generations_by_machine
            ),
            "producer_sequences_by_machine": list(
                receipt.summary.producer_sequences_by_machine
            ),
            "audit_sample_bound": receipt.summary.audit_sample_bound,
            "audit_sample_count": receipt.summary.audit_sample_count,
        },
        "lifecycle": {
            "effective_switch_interval_seconds": (
                receipt.effective_switch_interval_seconds
            ),
            "scheduler_hog_iterations": receipt.scheduler_hog_iterations,
            "native_inventory_before_close": _inventory_dict(receipt.native_inventory),
            "native_inventory_after_close": _inventory_dict(
                receipt.native_closed_inventory
            ),
        },
        "identity": {
            "git_revision": receipt.git_revision,
            "hostname": receipt.hostname,
            "python_version": receipt.python_version,
            "owner": {
                "process_generation": receipt.owner_identity.process_generation.hex(),
                "role": int(receipt.owner_identity.role),
                "tp_rank": receipt.owner_identity.tp_rank,
                "tp_size": receipt.owner_identity.tp_size,
                "digest": receipt.owner_identity.digest.hex(),
            },
            "source_hashes": dict(receipt.source_hashes),
        },
    }


def _latency_dict(value: NativeGILLatencyStatistics) -> dict[str, int]:
    """Return one JSON-safe native latency summary.

    :param value: Validated native latency statistics.
    :returns: Stable integer mapping.
    """

    return {
        "count": value.count,
        "p50_ns": value.p50_ns,
        "p95_ns": value.p95_ns,
        "p99_ns": value.p99_ns,
        "maximum_ns": value.maximum_ns,
    }


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    """Encode a mapping with deterministic canonical JSON formatting.

    :param value: JSON-safe mapping.
    :returns: UTF-8 canonical encoding with one trailing newline.
    """

    if type(value) is not dict:
        raise TypeError("value must be a dict")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _audit_trace_dict(
    cohort: str,
    trace: NativeTerminalQualificationTrace,
) -> dict[str, int | str]:
    """Return one canonical bounded native audit trace.

    :param cohort: First or last retained sample cohort.
    :param trace: Exact native reducer audit trace.
    :returns: JSON-safe bounded evidence mapping.
    """

    return {
        "cohort": cohort,
        "machine_index": trace.machine_index,
        "generation_index": trace.generation_index,
        "hop_index": trace.hop_index,
        "binding_digest": trace.binding_digest.hex(),
        "event_kind": int(trace.event_kind),
        "previous_phase": int(trace.previous_phase),
        "phase": int(trace.phase),
        "enqueued_ns": trace.enqueued_ns,
        "completed_ns": trace.completed_ns,
        "latency_ns": trace.latency_ns,
    }


def prepare_gil_qualification_output_root(output_root: Path) -> None:
    """Create a new or empty qualification evidence root.

    :param output_root: Concrete platform path chosen for evidence.
    """

    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("output_root must be new or empty")
    output_root.mkdir(parents=True, exist_ok=True)


def write_gil_qualification_artifacts(
    output_root: Path,
    receipt: GILQualificationRunReceipt,
) -> tuple[Path, Path, Path]:
    """Write the canonical receipt, bounded audit, and checksum closure.

    :param output_root: New or empty evidence directory.
    :param receipt: Complete immutable qualification receipt.
    :returns: Receipt, audit, and checksum paths.
    """

    prepare_gil_qualification_output_root(output_root)
    receipt_path = output_root / "gil-qualification-receipt.json"
    audit_path = output_root / "gil-qualification-audit.ndjson"
    checksums_path = output_root / "SHA256SUMS"
    receipt_path.write_bytes(canonical_json_bytes(receipt_dict(receipt)))
    with audit_path.open("wb") as audit_file:
        cohorts = (
            ("first", receipt.summary.first_audit_samples),
            ("last", receipt.summary.last_audit_samples),
        )
        for cohort, values in cohorts:
            for trace in sorted(
                values,
                key=lambda value: (value.machine_index, value.hop_index),
            ):
                audit_file.write(canonical_json_bytes(_audit_trace_dict(cohort, trace)))
    checksum_lines = []
    for path in (receipt_path, audit_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}\n")
    checksums_path.write_text("".join(checksum_lines), encoding="utf-8")
    return receipt_path, audit_path, checksums_path


def _parse_arguments() -> argparse.Namespace:
    """Parse the executable qualification command line.

    :returns: Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description="Run native GIL qualification through the terminal owner"
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in GILQualificationRunMode),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run qualification and emit its canonical evidence closure.

    :returns: Zero for the functional harness or authoritative latency pass,
        one for a valid authoritative latency rejection, and two for an
        execution failure.
    """

    arguments = _parse_arguments()
    mode = GILQualificationRunMode(arguments.mode)
    execution = GILQualificationExecutionConfig.for_mode(mode)
    try:
        receipt = run_gil_qualification(execution)
        receipt_path, _, _ = write_gil_qualification_artifacts(
            output_root=arguments.output_root,
            receipt=receipt,
        )
        sys.stdout.buffer.write(receipt_path.read_bytes())
        return 0 if receipt.verdict else 1
    except Exception:  # noqa: BLE001
        formatted_traceback = traceback.format_exc()
        failure = {
            "schema": "sglang.packed-terminal-gil-qualification.failure.v1",
            "mode": mode.value,
            "traceback": formatted_traceback,
        }
        arguments.output_root.mkdir(parents=True, exist_ok=True)
        failure_path = arguments.output_root / "gil-qualification-failure.json"
        failure_path.write_bytes(canonical_json_bytes(failure))
        sys.stderr.write(formatted_traceback)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
