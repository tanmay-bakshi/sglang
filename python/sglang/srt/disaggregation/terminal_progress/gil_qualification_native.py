import dataclasses
import enum
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerClosedError,
    TerminalOwnerDispatchObserver,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerOverflowError,
    TerminalOwnerPulse,
)
from torch.utils.cpp_extension import load


class GILNativeFatalCode(enum.StrEnum):
    """Sticky native qualification-channel outcomes."""

    NONE = "none"
    QUEUE_OVERFLOW = "queue_overflow"
    EVENTFD_WRITE_FAILURE = "eventfd_write_failure"
    EVENTFD_READ_FAILURE = "eventfd_read_failure"
    UNKNOWN_SEQUENCE = "unknown_sequence"
    INVALID_MACHINE_PHASE = "invalid_machine_phase"
    CONCURRENT_DRAIN = "concurrent_drain"
    START_AFTER_CLOSE = "start_after_close"


@dataclasses.dataclass(frozen=True, slots=True)
class GILNativeEventRecord:
    """One native-produced owner transition.

    :ivar producer_sequence: Gap-free sequence for the qualification source.
    :ivar machine_index: Closed-loop state-machine identity.
    :ivar generation_index: Gap-free request generation within the machine.
    :ivar hop_index: Ordered owner hop within the request generation.
    :ivar enqueued_ns: Native ``CLOCK_MONOTONIC_RAW`` publication timestamp.
    """

    producer_sequence: int
    machine_index: int
    generation_index: int
    hop_index: int
    enqueued_ns: int

    @classmethod
    def from_native(cls, value: dict[str, object]) -> "GILNativeEventRecord":
        """Construct a typed record from the native boundary.

        :param value: Native record mapping.
        :returns: Validated immutable event record.
        """

        return cls(
            producer_sequence=int(value["producer_sequence"]),
            machine_index=int(value["machine_index"]),
            generation_index=int(value["generation_index"]),
            hop_index=int(value["hop_index"]),
            enqueued_ns=int(value["enqueued_ns"]),
        )

    def __post_init__(self) -> None:
        """Validate exact non-negative native fields."""

        values = (
            self.producer_sequence,
            self.machine_index,
            self.generation_index,
            self.hop_index,
            self.enqueued_ns,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("native event fields must be non-negative integers")


@dataclasses.dataclass(frozen=True, slots=True)
class GILNativeHopTrace:
    """One native enqueue completed by a committed owner transition.

    :ivar event: Exact native event accepted by the owner.
    :ivar completed_ns: Native timestamp taken after owner dispatch committed.
    """

    event: GILNativeEventRecord
    completed_ns: int

    @classmethod
    def from_native(cls, value: dict[str, object]) -> "GILNativeHopTrace":
        """Construct a typed trace from the native boundary.

        :param value: Native trace mapping.
        :returns: Validated immutable hop trace.
        """

        return cls(
            event=GILNativeEventRecord.from_native(value),
            completed_ns=int(value["completed_ns"]),
        )

    def __post_init__(self) -> None:
        """Require completion at or after native publication."""

        if type(self.event) is not GILNativeEventRecord:
            raise TypeError("event must be GILNativeEventRecord")
        if type(self.completed_ns) is not int:
            raise TypeError("completed_ns must be an integer")
        if self.completed_ns < self.event.enqueued_ns:
            raise ValueError("owner completion cannot precede native enqueue")

    @property
    def latency_ns(self) -> int:
        """Return enqueue-to-committed-dispatch latency.

        :returns: Complete owner-hop latency in nanoseconds.
        """

        return self.completed_ns - self.event.enqueued_ns


@dataclasses.dataclass(frozen=True, slots=True)
class GILNativeInventory:
    """Complete native channel population and lifecycle inventory.

    :ivar machine_count: Configured closed-loop machine count.
    :ivar hop_count: Hops in one correlated generation.
    :ivar capacity: Physical native queue capacity.
    :ivar queued_count: Native events not yet drained into Python.
    :ivar delivered_unacknowledged_count: Drained events awaiting dispatch ACK.
    :ivar pending_count: Queued, delivered, and rejected identities retained.
    :ivar retired_count: Machines retired after completing the frozen run.
    :ivar transition_count: Owner-acknowledged native transitions.
    :ivar trace_count: Raw attributable hop traces retained.
    :ivar next_sequence: Next gap-free producer sequence.
    :ivar successful_wake_count: Successful eventfd publications.
    :ivar consumed_wake_count: Eventfd counter units consumed.
    :ivar started_ns: Native qualification start timestamp.
    :ivar ended_ns: Timestamp of the final committed owner transition.
    :ivar minimum_duration_ns: Configured sustained-duration floor.
    :ivar minimum_transition_count: Configured transition floor.
    :ivar started: Whether native production began.
    :ivar draining: Whether closed-loop replacement has stopped.
    :ivar complete: Whether all live generations completed and retired.
    :ivar closed: Whether the eventfd is closed.
    :ivar eventfd_open: Whether the source descriptor remains open.
    :ivar fatal_code: First sticky native failure.
    :ivar fatal_system_error: Captured errno for the first failure.
    :ivar rejected_record: Exact record rejected by queue overflow, if any.
    """

    machine_count: int
    hop_count: int
    capacity: int
    queued_count: int
    delivered_unacknowledged_count: int
    pending_count: int
    retired_count: int
    transition_count: int
    trace_count: int
    next_sequence: int
    successful_wake_count: int
    consumed_wake_count: int
    started_ns: int
    ended_ns: int
    minimum_duration_ns: int
    minimum_transition_count: int
    started: bool
    draining: bool
    complete: bool
    closed: bool
    eventfd_open: bool
    fatal_code: GILNativeFatalCode
    fatal_system_error: int
    rejected_record: GILNativeEventRecord | None

    @classmethod
    def from_native(cls, value: dict[str, object]) -> "GILNativeInventory":
        """Construct typed inventory from a native mapping.

        :param value: Native inventory mapping.
        :returns: Complete validated inventory.
        """

        rejected_value = value["rejected_record"]
        rejected_record: GILNativeEventRecord | None = None
        if rejected_value is not None:
            if type(rejected_value) is not dict:
                raise TypeError("rejected_record must be a mapping or None")
            rejected_record = GILNativeEventRecord.from_native(rejected_value)
        return cls(
            machine_count=int(value["machine_count"]),
            hop_count=int(value["hop_count"]),
            capacity=int(value["capacity"]),
            queued_count=int(value["queued_count"]),
            delivered_unacknowledged_count=int(value["delivered_unacknowledged_count"]),
            pending_count=int(value["pending_count"]),
            retired_count=int(value["retired_count"]),
            transition_count=int(value["transition_count"]),
            trace_count=int(value["trace_count"]),
            next_sequence=int(value["next_sequence"]),
            successful_wake_count=int(value["successful_wake_count"]),
            consumed_wake_count=int(value["consumed_wake_count"]),
            started_ns=int(value["started_ns"]),
            ended_ns=int(value["ended_ns"]),
            minimum_duration_ns=int(value["minimum_duration_ns"]),
            minimum_transition_count=int(value["minimum_transition_count"]),
            started=bool(value["started"]),
            draining=bool(value["draining"]),
            complete=bool(value["complete"]),
            closed=bool(value["closed"]),
            eventfd_open=bool(value["eventfd_open"]),
            fatal_code=GILNativeFatalCode(str(value["fatal_code"])),
            fatal_system_error=int(value["fatal_system_error"]),
            rejected_record=rejected_record,
        )

    def __post_init__(self) -> None:
        """Validate inventory conservation and failure consistency."""

        counts = (
            self.machine_count,
            self.hop_count,
            self.capacity,
            self.queued_count,
            self.delivered_unacknowledged_count,
            self.pending_count,
            self.retired_count,
            self.transition_count,
            self.trace_count,
            self.next_sequence,
            self.successful_wake_count,
            self.consumed_wake_count,
            self.started_ns,
            self.ended_ns,
            self.minimum_duration_ns,
            self.minimum_transition_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("native inventory counts must be non-negative integers")
        booleans = (
            self.started,
            self.draining,
            self.complete,
            self.closed,
            self.eventfd_open,
        )
        if any(type(value) is not bool for value in booleans):
            raise TypeError("native inventory flags must be bool values")
        retained = self.queued_count + self.delivered_unacknowledged_count
        if self.rejected_record is not None:
            retained += 1
        if self.pending_count != retained:
            raise ValueError("native pending inventory does not conserve identities")
        if self.trace_count != self.transition_count:
            raise ValueError("every acknowledged transition must retain one trace")
        if self.complete and (
            self.pending_count != 0 or self.retired_count != self.machine_count
        ):
            raise ValueError("complete native inventory must be fully retired")
        if self.closed and self.eventfd_open:
            raise ValueError("a closed native channel cannot retain an open eventfd")


@dataclasses.dataclass(frozen=True, slots=True)
class GILNativeDrain:
    """One take-once native eventfd and queue drain.

    :ivar wake_count: Coalesced eventfd counter observed.
    :ivar records: Events moved into delivered-but-unacknowledged state.
    :ivar inventory: Native lifecycle snapshot after delivery.
    """

    wake_count: int
    records: tuple[GILNativeEventRecord, ...]
    inventory: GILNativeInventory


class _NativeGILQualificationBridge(Protocol):
    """Typed pybind11 qualification-channel boundary."""

    def fileno(self) -> int:
        """Return the native eventfd.

        :returns: Pollable native descriptor.
        """

    def start(
        self,
        minimum_duration_seconds: float,
        minimum_transition_count: int,
    ) -> None:
        """Publish the initial closed-loop generation without the GIL.

        :param minimum_duration_seconds: Sustained duration before drain.
        :param minimum_transition_count: Completed transition floor.
        """

    def drain(self) -> dict[str, object]:
        """Move queued records to delivered-but-unacknowledged state.

        :returns: Records, wake count, and native inventory.
        """

    def acknowledge_dispatch(self, producer_sequence: int) -> None:
        """Commit one owner transition and produce its successor.

        :param producer_sequence: Exact delivered native sequence.
        """

    def wait_until_complete(self, timeout_seconds: float) -> bool:
        """Wait without the GIL for all live generations to retire.

        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: Whether qualification completed within the bound.
        """

    def traces(self) -> list[dict[str, object]]:
        """Return raw native trace evidence.

        :returns: Every acknowledged native hop.
        """

    def inventory(self) -> dict[str, object]:
        """Return complete native lifecycle inventory.

        :returns: Native inventory mapping.
        """

    def close(self) -> None:
        """Close a complete zero-inventory channel."""

    def abort_and_close(self) -> None:
        """Release the descriptor after failure evidence is retained."""

    def _break_eventfd_for_test(self) -> None:
        """Close the descriptor in a native test build."""


def _native_source_path() -> Path:
    """Return the packaged native qualification source.

    :returns: Absolute C++ source path.
    """

    return Path(__file__).with_name("gil_qualification_bridge.cpp")


@lru_cache(maxsize=2)
def _load_native_gil_qualification_bridge(testing: bool = False) -> ModuleType:
    """Compile and load the CPU-only native qualification bridge.

    :param testing: Whether deterministic fault hooks are required.
    :returns: Loaded pybind11 extension module.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("GIL qualification bridge requires Linux eventfd")
    source_path = _native_source_path()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    variant = "test" if testing else "runtime"
    module_name = f"sglang_gil_qualification_bridge_{source_digest}_{variant}"
    extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
    if testing:
        extra_cflags.append("-DSGLANG_GIL_QUALIFICATION_TESTING")
    build_directory_value = os.environ.get("SGLANG_GIL_BRIDGE_BUILD_DIR")
    build_directory: str | None = None
    if build_directory_value is not None:
        build_path = Path(build_directory_value).resolve() / module_name
        build_path.mkdir(parents=True, exist_ok=True)
        build_directory = str(build_path)
    return load(
        name=module_name,
        sources=[str(source_path)],
        extra_cflags=extra_cflags,
        extra_ldflags=["-pthread"],
        build_directory=build_directory,
        with_cuda=False,
        verbose=False,
    )


class NativeGILQualificationEventSource(
    TerminalOwnerEventSource,
    TerminalOwnerDispatchObserver,
):
    """Native closed-loop event source traversing the real owner reactor."""

    _name: str
    _native: _NativeGILQualificationBridge
    _testing: bool
    _delivered: dict[int, GILNativeEventRecord]
    _closed: bool

    def __init__(
        self,
        name: str,
        machine_count: int,
        hop_count: int,
        capacity: int,
        *,
        testing: bool = False,
    ) -> None:
        """Create one native qualification channel before owner startup.

        :param name: Stable owner event-source identity.
        :param machine_count: Concurrent closed-loop state machines.
        :param hop_count: Correlated transitions in one generation.
        :param capacity: Physical native queue bound.
        :param testing: Whether native fault hooks are required.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        counts = (machine_count, hop_count, capacity)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("machine_count, hop_count, and capacity must be positive")
        native_module = _load_native_gil_qualification_bridge(testing=testing)
        self._name = name
        self._native = cast(
            _NativeGILQualificationBridge,
            native_module.GILQualificationBridge(
                machine_count,
                hop_count,
                capacity,
            ),
        )
        self._testing = testing
        self._delivered = {}
        self._closed = False

    @property
    def name(self) -> str:
        """Return the stable owner source name.

        :returns: Event-source identity.
        """

        return self._name

    def fileno(self) -> int:
        """Return the open native eventfd.

        :returns: Pollable native descriptor.
        """

        if self._closed:
            raise TerminalOwnerClosedError("GIL qualification source is closed")
        return int(self._native.fileno())

    @property
    def pending_count(self) -> int:
        """Return queued plus delivered-but-unacknowledged identities.

        :returns: Exact retained native identity count.
        """

        return self.inventory().pending_count

    def start(
        self,
        minimum_duration_seconds: float,
        minimum_transition_count: int,
    ) -> None:
        """Publish the initial machine population entirely in native code.

        :param minimum_duration_seconds: Sustained duration before replacement
            closes.
        :param minimum_transition_count: Completed transition floor.
        """

        if self._closed:
            raise TerminalOwnerClosedError("GIL qualification source is closed")
        if (
            type(minimum_duration_seconds) is not float
            or minimum_duration_seconds <= 0.0
        ):
            raise ValueError("minimum_duration_seconds must be a positive float")
        if type(minimum_transition_count) is not int or minimum_transition_count <= 0:
            raise ValueError("minimum_transition_count must be a positive integer")
        self._native.start(minimum_duration_seconds, minimum_transition_count)

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Deliver every queued native record without acknowledging dispatch.

        :returns: Exact producer-ordered pulse envelopes.
        :raises TerminalOwnerOverflowError: If the native physical bound failed.
        """

        if self._closed:
            raise TerminalOwnerClosedError("GIL qualification source is closed")
        value = self._native.drain()
        raw_records = value["records"]
        raw_inventory = value["inventory"]
        if type(raw_records) is not list or type(raw_inventory) is not dict:
            raise RuntimeError("native GIL qualification drain has invalid shape")
        records = tuple(
            GILNativeEventRecord.from_native(raw_record)
            for raw_record in raw_records
            if type(raw_record) is dict
        )
        if len(records) != len(raw_records):
            raise RuntimeError("native GIL qualification record has invalid shape")
        envelopes = tuple(
            TerminalOwnerEventEnvelope(
                producer_sequence=record.producer_sequence,
                enqueued_ns=record.enqueued_ns,
                command=TerminalOwnerPulse(),
            )
            for record in records
        )
        for record in records:
            if record.producer_sequence in self._delivered:
                raise RuntimeError("native sequence was delivered more than once")
            self._delivered[record.producer_sequence] = record
        inventory = GILNativeInventory.from_native(raw_inventory)
        if inventory.fatal_code is GILNativeFatalCode.NONE:
            return envelopes
        if inventory.fatal_code is GILNativeFatalCode.QUEUE_OVERFLOW:
            rejected_envelope: TerminalOwnerEventEnvelope | None = None
            if inventory.rejected_record is not None:
                rejected_envelope = TerminalOwnerEventEnvelope(
                    producer_sequence=inventory.rejected_record.producer_sequence,
                    enqueued_ns=inventory.rejected_record.enqueued_ns,
                    command=TerminalOwnerPulse(),
                )
            raise TerminalOwnerOverflowError(
                "native GIL qualification queue overflowed",
                source_name=self._name,
                pending_envelopes=envelopes,
                rejected_envelope=rejected_envelope,
            )
        raise RuntimeError(
            f"native GIL qualification source failed: {inventory.fatal_code.value}"
        )

    def acknowledge_dispatch(self, envelope: TerminalOwnerEventEnvelope) -> None:
        """Acknowledge only after the matching owner transition commits.

        Successor publication occurs inside a GIL-releasing native call, so
        scheduler contention before the owner's next acquisition remains in
        the measured interval.

        :param envelope: Exact committed event-source envelope.
        """

        if type(envelope) is not TerminalOwnerEventEnvelope:
            raise TypeError("envelope must be TerminalOwnerEventEnvelope")
        record = self._delivered.get(envelope.producer_sequence)
        if record is None:
            raise RuntimeError("owner acknowledged an unknown native sequence")
        if envelope.enqueued_ns != record.enqueued_ns:
            raise RuntimeError("owner acknowledgment changed native enqueue time")
        if type(envelope.command) is not TerminalOwnerPulse:
            raise RuntimeError("qualification source delivered a non-pulse command")
        self._native.acknowledge_dispatch(envelope.producer_sequence)
        del self._delivered[envelope.producer_sequence]

    def wait_until_complete(self, timeout_seconds: float) -> bool:
        """Wait without the GIL for the complete native population.

        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: Whether all machines retired within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(self._native.wait_until_complete(timeout_seconds))

    def traces(self) -> tuple[GILNativeHopTrace, ...]:
        """Return every raw attributable native hop trace.

        :returns: Immutable trace population in owner completion order.
        """

        raw_traces = self._native.traces()
        if type(raw_traces) is not list:
            raise RuntimeError("native GIL qualification traces must be a list")
        traces: list[GILNativeHopTrace] = []
        for raw_trace in raw_traces:
            if type(raw_trace) is not dict:
                raise RuntimeError("native GIL qualification trace has invalid shape")
            traces.append(GILNativeHopTrace.from_native(raw_trace))
        return tuple(traces)

    def inventory(self) -> GILNativeInventory:
        """Return complete typed native lifecycle inventory.

        :returns: Current immutable inventory.
        """

        value = self._native.inventory()
        if type(value) is not dict:
            raise RuntimeError("native GIL qualification inventory must be a mapping")
        return GILNativeInventory.from_native(value)

    def close(self) -> None:
        """Close only after exact complete zero inventory."""

        if self._closed:
            return
        if len(self._delivered) != 0:
            raise RuntimeError("cannot close with delivered native records")
        self._native.close()
        self._closed = True

    def abort_and_close(self) -> None:
        """Release the descriptor after preserving failure inventory."""

        if self._closed:
            return
        self._native.abort_and_close()
        self._closed = True

    def break_eventfd_for_test(self) -> None:
        """Break the native descriptor in a test build."""

        if not self._testing:
            raise RuntimeError("eventfd fault injection requires a test build")
        self._native._break_eventfd_for_test()
