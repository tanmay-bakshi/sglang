import argparse
import dataclasses
import enum
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import traceback
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
    GILStressCollection,
    GILStressPlan,
    correlate_gil_native_traces,
    evaluate_gil_qualification,
)
from sglang.srt.disaggregation.terminal_progress.gil_qualification_native import (
    GILNativeHopTrace,
    GILNativeInventory,
    NativeGILQualificationEventSource,
)
from sglang.srt.disaggregation.terminal_progress.owner import (
    PackedTerminalProgressOwner,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    InjectTerminalOwnerFailure,
    TerminalOwnerDisposition,
    TerminalOwnerEventSourceRegistration,
    TerminalOwnerFatalCause,
    TerminalOwnerSnapshot,
)

_FUNCTIONAL_DURATION_SECONDS = 5.0
_FUNCTIONAL_MINIMUM_TRANSITIONS = 10_080
_FUNCTIONAL_TIMEOUT_SECONDS = 90.0
_AUTHORITATIVE_TIMEOUT_SECONDS = 300.0
_NATIVE_QUEUE_CAPACITY = 64
_HOG_HOT_ITERATIONS = 100_000


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
class GILQualificationRunReceipt:
    """Complete executable result and lifecycle attestation.

    :ivar execution: Exact functional or authoritative population.
    :ivar effective_switch_interval_seconds: Verified process switch interval.
    :ivar result: Frozen qualification arithmetic over raw native evidence.
    :ivar native_inventory: Complete native inventory before close.
    :ivar native_closed_inventory: Native inventory after exact closure.
    :ivar owner_transition_count: Real owner transitions before shutdown.
    :ivar owner_final_transition_count: Owner transitions after clean shutdown.
    :ivar scheduler_hog_iterations: Pure-Python scheduler work executed.
    :ivar source_hashes: Hash-bound implementation files used by the run.
    :ivar git_revision: Exact serving revision under qualification.
    :ivar hostname: Environment hostname carrying the evidence.
    :ivar python_version: Python runtime identity.
    """

    execution: GILQualificationExecutionConfig
    effective_switch_interval_seconds: float
    result: GILQualificationResult
    native_inventory: GILNativeInventory
    native_closed_inventory: GILNativeInventory
    owner_transition_count: int
    owner_final_transition_count: int
    scheduler_hog_iterations: int
    source_hashes: tuple[tuple[str, str], ...]
    git_revision: str
    hostname: str
    python_version: str

    def __post_init__(self) -> None:
        """Validate cross-layer conservation and clean closure."""

        if type(self.execution) is not GILQualificationExecutionConfig:
            raise TypeError("execution must be GILQualificationExecutionConfig")
        if type(self.result) is not GILQualificationResult:
            raise TypeError("result must be GILQualificationResult")
        if type(self.native_inventory) is not GILNativeInventory:
            raise TypeError("native_inventory must be GILNativeInventory")
        if type(self.native_closed_inventory) is not GILNativeInventory:
            raise TypeError("native_closed_inventory must be GILNativeInventory")
        if (
            type(self.effective_switch_interval_seconds) is not float
            or self.effective_switch_interval_seconds
            != GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS
        ):
            raise ValueError("effective switch interval must equal the frozen value")
        counts = (
            self.owner_transition_count,
            self.owner_final_transition_count,
            self.scheduler_hog_iterations,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("receipt counts must be positive integers")
        if self.owner_transition_count != self.native_inventory.transition_count:
            raise ValueError("native and real-owner transition counts must agree")
        if self.result.transition_count != self.owner_transition_count:
            raise ValueError("result and real-owner transition counts must agree")
        if self.owner_final_transition_count != self.owner_transition_count + 2:
            raise ValueError("clean owner shutdown must commit exactly two commands")
        if self.native_inventory.pending_count != 0:
            raise ValueError("native inventory must reach zero before owner shutdown")
        if not self.native_inventory.complete:
            raise ValueError("native population must be complete")
        if not self.native_closed_inventory.closed:
            raise ValueError("native source must close in the final receipt")
        if self.native_closed_inventory.eventfd_open:
            raise ValueError("closed native receipt cannot retain an eventfd")
        if type(self.source_hashes) is not tuple or len(self.source_hashes) == 0:
            raise ValueError("source_hashes must be a non-empty tuple")
        strings = (self.git_revision, self.hostname, self.python_version)
        if any(type(value) is not str or len(value) == 0 for value in strings):
            raise ValueError("receipt identity strings must be non-empty")

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
                self.native_inventory.transition_count
                >= self.execution.minimum_transition_count
                and self.native_inventory.ended_ns - self.native_inventory.started_ns
                >= int(self.execution.minimum_duration_seconds * 1_000_000_000)
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
        module_root / "gil_qualification_native.py",
        module_root / "gil_qualification_bridge.cpp",
        module_root / "gil_qualification_runner.py",
        module_root / "owner.py",
        module_root / "owner_events.py",
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


def _stop_owner(owner: PackedTerminalProgressOwner) -> TerminalOwnerSnapshot:
    """Drain and stop the owner through its exact lifecycle protocol.

    :param owner: Running owner with zero external-source inventory.
    :returns: Final stopped snapshot.
    """

    owner.begin_shutdown()
    owner.retire_shutdown_producers()
    final_snapshot = owner.wait_for_snapshot(
        lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.STOPPED,
        timeout_seconds=5.0,
    )
    if not owner.join(timeout_seconds=5.0):
        raise TimeoutError("terminal owner did not join after clean drain")
    return final_snapshot


def run_gil_qualification(
    execution: GILQualificationExecutionConfig,
) -> tuple[GILQualificationRunReceipt, tuple[GILNativeHopTrace, ...]]:
    """Run the native closed-loop storm through the integrated owner.

    :param execution: Exact predeclared functional or authoritative bounds.
    :returns: Immutable receipt and raw attributable native traces.
    """

    if type(execution) is not GILQualificationExecutionConfig:
        raise TypeError("execution must be GILQualificationExecutionConfig")
    original_switch_interval = sys.getswitchinterval()
    source = NativeGILQualificationEventSource(
        name="packed-terminal-gil-qualification",
        machine_count=GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
        hop_count=GIL_QUALIFICATION_OWNER_HOP_COUNT,
        capacity=_NATIVE_QUEUE_CAPACITY,
    )
    owner = PackedTerminalProgressOwner(
        submission_capacity=64,
        output_capacity=64,
        event_sources=(
            TerminalOwnerEventSourceRegistration(
                source=source,
                close_on_shutdown=False,
                dispatch_observer=source,
            ),
        ),
        thread_name="packed-terminal-gil-qualification-owner",
    )
    hog = _PurePythonSchedulerHog()
    owner_started = False
    hog_started = False
    try:
        sys.setswitchinterval(GIL_QUALIFICATION_SWITCH_INTERVAL_SECONDS)
        effective_switch_interval = sys.getswitchinterval()
        require_frozen_switch_interval(effective_switch_interval)

        owner.start()
        owner_started = True
        owner.wait_for_snapshot(
            lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.RUNNING,
            timeout_seconds=5.0,
        )
        hog_started = True
        hog.start_and_wait_until_hot(timeout_seconds=5.0)
        require_frozen_switch_interval(sys.getswitchinterval())

        source.start(
            minimum_duration_seconds=execution.minimum_duration_seconds,
            minimum_transition_count=execution.minimum_transition_count,
        )
        if not source.wait_until_complete(execution.timeout_seconds):
            raise TimeoutError("native qualification population did not complete")
        hog.stop_and_join(timeout_seconds=5.0)
        hog_started = False

        native_inventory = source.inventory()
        traces = source.traces()
        samples = correlate_gil_native_traces(traces)
        qualification_snapshot = owner.snapshot()
        if qualification_snapshot.disposition is not TerminalOwnerDisposition.RUNNING:
            raise RuntimeError("owner did not remain healthy through qualification")
        if qualification_snapshot.queued_submission_count != 0:
            raise RuntimeError("owner retained queued qualification work")
        elapsed_seconds = (
            native_inventory.ended_ns - native_inventory.started_ns
        ) / 1_000_000_000
        plan = GILStressPlan(
            config=GILQualificationConfig(),
            producer=GILQualificationProducer.NATIVE_OR_GIL_RELEASING,
        )
        collection = GILStressCollection(
            samples=samples,
            native_hop_traces=traces,
            elapsed_seconds=float(elapsed_seconds),
        )
        result = evaluate_gil_qualification(
            plan=plan,
            samples=collection.samples,
            elapsed_seconds=collection.elapsed_seconds,
        )
        final_owner_snapshot = _stop_owner(owner)
        owner_started = False
        source.close()
        closed_inventory = source.inventory()
        receipt = GILQualificationRunReceipt(
            execution=execution,
            effective_switch_interval_seconds=effective_switch_interval,
            result=result,
            native_inventory=native_inventory,
            native_closed_inventory=closed_inventory,
            owner_transition_count=qualification_snapshot.owner_transition_count,
            owner_final_transition_count=final_owner_snapshot.owner_transition_count,
            scheduler_hog_iterations=hog.iterations,
            source_hashes=_implementation_hashes(),
            git_revision=_git_revision(),
            hostname=os.uname().nodename,
            python_version=platform.python_version(),
        )
        return receipt, traces
    finally:
        if hog_started:
            hog.stop_and_join(timeout_seconds=5.0)
        if owner_started:
            snapshot = owner.snapshot()
            if snapshot.reactor_alive and snapshot.disposition not in (
                TerminalOwnerDisposition.STOPPED,
                TerminalOwnerDisposition.PROCESS_FATAL,
            ):
                owner.submit(
                    InjectTerminalOwnerFailure(
                        cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
                        reason="GIL qualification aborted before complete evidence",
                    )
                )
                owner.wait_for_snapshot(
                    lambda current: (
                        current.disposition is TerminalOwnerDisposition.PROCESS_FATAL
                    ),
                    timeout_seconds=5.0,
                )
            if owner.snapshot().reactor_alive:
                owner.join(timeout_seconds=5.0)
        source.abort_and_close()
        sys.setswitchinterval(original_switch_interval)


def _inventory_dict(inventory: GILNativeInventory) -> dict[str, object]:
    """Return a JSON-safe native inventory mapping.

    :param inventory: Typed native lifecycle inventory.
    :returns: Canonically serializable mapping.
    """

    value = dataclasses.asdict(inventory)
    value["fatal_code"] = inventory.fatal_code.value
    if inventory.rejected_record is not None:
        value["rejected_record"] = dataclasses.asdict(inventory.rejected_record)
    return value


def receipt_dict(receipt: GILQualificationRunReceipt) -> dict[str, object]:
    """Return the stable JSON schema for one executable receipt.

    :param receipt: Complete immutable run receipt.
    :returns: Canonically serializable mapping.
    """

    return {
        "schema": "sglang.packed-terminal-gil-qualification.v1",
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
        "lifecycle": {
            "effective_switch_interval_seconds": (
                receipt.effective_switch_interval_seconds
            ),
            "owner_transition_count": receipt.owner_transition_count,
            "owner_final_transition_count": (receipt.owner_final_transition_count),
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
            "source_hashes": dict(receipt.source_hashes),
        },
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


def _trace_dict(trace: GILNativeHopTrace) -> dict[str, int]:
    """Return one canonical raw trace mapping.

    :param trace: Native enqueue and committed-dispatch trace.
    :returns: JSON-safe trace mapping.
    """

    event = trace.event
    return {
        "producer_sequence": event.producer_sequence,
        "machine_index": event.machine_index,
        "generation_index": event.generation_index,
        "hop_index": event.hop_index,
        "enqueued_ns": event.enqueued_ns,
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
    traces: tuple[GILNativeHopTrace, ...],
) -> tuple[Path, Path, Path]:
    """Write the canonical receipt, raw evidence, and checksum closure.

    :param output_root: New or empty evidence directory.
    :param receipt: Complete immutable qualification receipt.
    :param traces: Raw native hop evidence.
    :returns: Receipt, trace, and checksum paths.
    """

    prepare_gil_qualification_output_root(output_root)
    receipt_path = output_root / "gil-qualification-receipt.json"
    traces_path = output_root / "gil-qualification-traces.ndjson"
    checksums_path = output_root / "SHA256SUMS"
    receipt_path.write_bytes(canonical_json_bytes(receipt_dict(receipt)))
    with traces_path.open("wb") as trace_file:
        for trace in sorted(
            traces,
            key=lambda value: value.event.producer_sequence,
        ):
            trace_file.write(canonical_json_bytes(_trace_dict(trace)))
    checksum_lines = []
    for path in (receipt_path, traces_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}\n")
    checksums_path.write_text("".join(checksum_lines), encoding="utf-8")
    return receipt_path, traces_path, checksums_path


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
        receipt, traces = run_gil_qualification(execution)
        receipt_path, _, _ = write_gil_qualification_artifacts(
            output_root=arguments.output_root,
            receipt=receipt,
            traces=traces,
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
