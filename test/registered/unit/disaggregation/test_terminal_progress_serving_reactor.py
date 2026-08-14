import contextlib
import inspect
import os
import selectors
import threading

import pytest
from sglang.srt.disaggregation.terminal_progress.clock import (
    ManualTerminalOwnerClock,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.serving_reactor import (
    PackedTerminalProcessReactor,
    PackedTerminalProcessReactorClosedError,
    PackedTerminalProcessReactorDisposition,
    PackedTerminalProcessReactorError,
    PackedTerminalProcessReactorFailure,
    PackedTerminalProcessReactorFailureCause,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceServing,
)

_WAIT_SECONDS = 2.0


class _RuntimeWake:
    """One nonblocking fd-backed runtime wake for reactor tests."""

    _read_fd: int
    _write_fd: int
    _closed: bool

    def __init__(self) -> None:
        """Construct one open coalesced wake."""

        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._closed = False

    @property
    def fileno(self) -> int:
        """Return the selector-side descriptor.

        :returns: Open read descriptor.
        """

        return self._read_fd

    def signal(self, count: int = 1) -> None:
        """Publish a bounded wake population.

        :param count: Number of immutable fake runtime items.
        """

        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        os.write(self._write_fd, b"x" * count)

    def drain(self) -> int:
        """Drain the complete coalesced byte population.

        :returns: Number of fake runtime items consumed.
        """

        consumed = 0
        while True:
            try:
                payload = os.read(self._read_fd, 4096)
            except BlockingIOError:
                return consumed
            if len(payload) == 0:
                raise RuntimeError("runtime wake closed before drain")
            consumed += len(payload)

    def close(self) -> None:
        """Close both fake-runtime descriptors exactly once."""

        if self._closed:
            return
        os.close(self._read_fd)
        os.close(self._write_fd)
        self._closed = True


class _SourceServing(PackedTerminalSourceServing):
    """Narrow source serving test double with real readable descriptors."""

    _wakes: tuple[_RuntimeWake, ...]
    _drain_count: int
    _drained_event: threading.Event
    _drain_error: RuntimeError | None

    def __init__(self, wake_count: int = 1) -> None:
        """Construct an inert source serving double.

        :param wake_count: Number of runtime descriptors to expose.
        """

        self._wakes = tuple(_RuntimeWake() for _ in range(wake_count))
        self._drain_count = 0
        self._drained_event = threading.Event()
        self._drain_error = None

    @property
    def runtime_filenos(self) -> tuple[int, ...]:
        """Return the stable fake runtime descriptor population.

        :returns: Stable read descriptors.
        """

        return tuple(wake.fileno for wake in self._wakes)

    @property
    def drain_count(self) -> int:
        """Return the number of all-context drains.

        :returns: Exact drain count.
        """

        return self._drain_count

    def fail_next_drain(self) -> None:
        """Inject one source serving drain failure."""

        self._drain_error = RuntimeError("injected source drain failure")

    def signal(self, wake_index: int, count: int = 1) -> None:
        """Signal one exact fake runtime descriptor.

        :param wake_index: Descriptor index to signal.
        :param count: Fake item population.
        """

        self._wakes[wake_index].signal(count)

    def drain_runtime_actions(self) -> int:
        """Drain every fake runtime context in one coalesced call.

        :returns: Total fake items consumed.
        """

        if self._drain_error is not None:
            raise self._drain_error
        consumed = sum(wake.drain() for wake in self._wakes)
        self._drain_count += 1
        self._drained_event.set()
        return consumed

    def wait_for_drain(self) -> None:
        """Wait for one bounded reactor drain."""

        if not self._drained_event.wait(_WAIT_SECONDS):
            raise TimeoutError("source serving drain did not run")

    def close_wakes(self) -> None:
        """Close fake descriptors after reactor closure."""

        for wake in self._wakes:
            wake.close()


class _DecodeServing(PackedTerminalDecodeServing):
    """Narrow decode serving test double with exact manual deadlines."""

    _wake: _RuntimeWake
    _deadlines: tuple[int, ...]
    _condition: threading.Condition
    _drained_event: threading.Event
    _expiration_event: threading.Event
    _observed_deadline: int | None
    _deadline_observed: bool
    _drain_count: int
    _expiration_timestamps: list[int]

    def __init__(self) -> None:
        """Construct one dormant decode serving double."""

        self._wake = _RuntimeWake()
        self._deadlines = ()
        self._condition = threading.Condition()
        self._drained_event = threading.Event()
        self._expiration_event = threading.Event()
        self._observed_deadline = None
        self._deadline_observed = False
        self._drain_count = 0
        self._expiration_timestamps = []

    @property
    def runtime_filenos(self) -> tuple[int, ...]:
        """Return one stable decode runtime descriptor.

        :returns: Singleton read descriptor.
        """

        return (self._wake.fileno,)

    @property
    def next_coordinator_deadline_ns(self) -> int | None:
        """Return and observe the earliest exact fake deadline.

        :returns: Earliest deadline, otherwise ``None``.
        """

        with self._condition:
            deadline = None if len(self._deadlines) == 0 else min(self._deadlines)
            self._observed_deadline = deadline
            self._deadline_observed = True
            self._condition.notify_all()
            return deadline

    @property
    def drain_count(self) -> int:
        """Return the number of all-context drains.

        :returns: Exact drain count.
        """

        return self._drain_count

    @property
    def expiration_timestamps(self) -> tuple[int, ...]:
        """Return exact timestamps supplied to deadline expiration.

        :returns: Stable expiration timestamp population.
        """

        with self._condition:
            return tuple(self._expiration_timestamps)

    def set_deadlines(self, deadlines: tuple[int, ...]) -> None:
        """Replace the fake coordinator deadline population.

        :param deadlines: Non-negative exact deadlines.
        """

        if type(deadlines) is not tuple or any(
            type(value) is not int or value < 0 for value in deadlines
        ):
            raise ValueError("deadlines must be non-negative integers")
        with self._condition:
            self._deadlines = deadlines
            self._deadline_observed = False
        self._expiration_event.clear()

    def signal_runtime(self, count: int = 1) -> None:
        """Signal fake decode runtime work.

        :param count: Fake item population.
        """

        self._wake.signal(count)

    def drain_runtime_actions(self) -> int:
        """Drain one fake decode runtime context.

        :returns: Number of fake runtime items consumed.
        """

        consumed = self._wake.drain()
        self._drain_count += 1
        self._drained_event.set()
        return consumed

    def expire_coordinators(self, now_ns: int) -> int:
        """Remove all fake deadlines due at one exact timestamp.

        :param now_ns: Manual monotonic timestamp.
        :returns: Number of terminalized fake coordinators.
        """

        with self._condition:
            due = tuple(value for value in self._deadlines if value <= now_ns)
            self._deadlines = tuple(
                value for value in self._deadlines if value > now_ns
            )
            self._expiration_timestamps.append(now_ns)
        if len(due) > 0:
            self._expiration_event.set()
        return len(due)

    def wait_for_drain(self) -> None:
        """Wait for one bounded runtime drain."""

        if not self._drained_event.wait(_WAIT_SECONDS):
            raise TimeoutError("decode serving drain did not run")

    def wait_for_deadline(self, deadline_ns: int | None) -> None:
        """Wait for the reactor to observe one exact deadline.

        :param deadline_ns: Expected armed deadline, otherwise ``None``.
        """

        with self._condition:
            observed = self._condition.wait_for(
                lambda: (
                    self._deadline_observed and self._observed_deadline == deadline_ns
                ),
                timeout=_WAIT_SECONDS,
            )
        if not observed:
            raise TimeoutError("decode deadline was not re-derived")

    def wait_for_expiration(self) -> None:
        """Wait for one bounded coordinator expiration."""

        if not self._expiration_event.wait(_WAIT_SECONDS):
            raise TimeoutError("decode deadline did not expire")

    def close_wakes(self) -> None:
        """Close fake descriptors after reactor closure."""

        self._wake.close()


class _UnexpectedExitReactor(PackedTerminalProcessReactor):
    """Reactor whose loop violates the process-lifetime exit contract."""

    def _run_loop(self, selector: selectors.BaseSelector) -> None:
        """Return without an explicit stop.

        :param selector: Fully registered selector.
        """


def _close_source(
    reactor: PackedTerminalProcessReactor,
    serving: _SourceServing,
) -> None:
    """Bound reactor and fake-source cleanup.

    :param reactor: Reactor to stop and close.
    :param serving: Fake serving descriptors to release afterward.
    """

    reactor.close(_WAIT_SECONDS)
    serving.close_wakes()


def _close_decode(
    reactor: PackedTerminalProcessReactor,
    serving: _DecodeServing,
) -> None:
    """Bound reactor and fake-decode cleanup.

    :param reactor: Reactor to stop and close.
    :param serving: Fake serving descriptors to release afterward.
    """

    reactor.close(_WAIT_SECONDS)
    serving.close_wakes()


def test_source_runtime_fds_coalesce_into_one_complete_drain() -> None:
    """One selector wave drains all readable source execution contexts."""

    serving = _SourceServing(wake_count=3)
    failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_source(serving, failures.append)
    serving.signal(0, 2)
    serving.signal(1, 3)
    serving.signal(2, 4)
    reactor.start(_WAIT_SECONDS)
    try:
        serving.wait_for_drain()
        inventory = reactor.inventory()
        assert serving.drain_count == 1
        assert inventory.runtime_ready_descriptor_count == 3
        assert inventory.runtime_drain_count == 1
        assert inventory.runtime_item_count == 9
        assert failures == []
    finally:
        _close_source(reactor, serving)


def test_decode_runtime_fd_drains_without_scheduler_affine_consumption() -> None:
    """Decode runtime wakes execute only the serving all-context drain."""

    serving = _DecodeServing()
    clock = ManualTerminalOwnerClock()
    failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_decode(
        serving,
        clock.now_ns,
        failures.append,
    )
    reactor.start(_WAIT_SECONDS)
    try:
        serving.signal_runtime(7)
        serving.wait_for_drain()
        inventory = reactor.inventory()
        assert serving.drain_count == 1
        assert inventory.runtime_drain_count == 1
        assert inventory.runtime_item_count == 7
        assert failures == []
    finally:
        _close_decode(reactor, serving)


def test_decode_deadline_rearms_and_expires_at_exact_manual_time() -> None:
    """An earlier deadline wake replaces the wait and re-arms after expiry."""

    serving = _DecodeServing()
    clock = ManualTerminalOwnerClock()
    failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_decode(
        serving,
        clock.now_ns,
        failures.append,
    )
    reactor.start(_WAIT_SECONDS)
    try:
        serving.set_deadlines((1_000,))
        reactor.notify_coordinator_deadline_changed()
        serving.wait_for_deadline(1_000)

        serving.set_deadlines((100, 200))
        reactor.notify_coordinator_deadline_changed()
        serving.wait_for_deadline(100)
        assert reactor.inventory().next_coordinator_deadline_ns == 100

        clock.advance_ns(99)
        serving.set_deadlines((100, 200))
        reactor.notify_coordinator_deadline_changed()
        serving.wait_for_deadline(100)
        assert serving.expiration_timestamps == ()

        serving.set_deadlines((100, 200))
        clock.advance_ns(1)
        reactor.notify_coordinator_deadline_changed()
        serving.wait_for_expiration()
        assert serving.expiration_timestamps == (100,)
        serving.wait_for_deadline(200)
        assert reactor.inventory().next_coordinator_deadline_ns == 200

        serving.set_deadlines((200,))
        clock.advance_ns(100)
        reactor.notify_coordinator_deadline_changed()
        serving.wait_for_expiration()
        assert serving.expiration_timestamps == (100, 200)
        serving.wait_for_deadline(None)
        inventory = reactor.inventory()
        assert inventory.deadline_expiration_count == 2
        assert inventory.expired_coordinator_count == 2
        assert failures == []
    finally:
        _close_decode(reactor, serving)


def test_explicit_stop_wakes_an_indefinitely_blocked_source_reactor() -> None:
    """The owned control fd releases a source reactor with no runtime work."""

    serving = _SourceServing()
    failures: list[PackedTerminalProcessReactorFailure] = []
    reactor = PackedTerminalProcessReactor.for_source(serving, failures.append)
    reactor.start(_WAIT_SECONDS)
    reactor.require_admission_open()
    reactor.stop_admission()
    with pytest.raises(PackedTerminalProcessReactorClosedError):
        reactor.require_admission_open()
    reactor.begin_stop()
    assert reactor.join(_WAIT_SECONDS)
    inventory = reactor.close(_WAIT_SECONDS)
    serving.close_wakes()
    assert inventory.disposition is PackedTerminalProcessReactorDisposition.STOPPED
    assert inventory.thread_finished
    assert inventory.control_wake_count == 1
    assert failures == []


def test_serving_drain_failure_is_delivered_once_as_process_fatal() -> None:
    """A serving exception closes admission and preserves exact fatal cause."""

    serving = _SourceServing()
    serving.fail_next_drain()
    failures: list[PackedTerminalProcessReactorFailure] = []
    fatal_event = threading.Event()

    def process_fatal(failure: PackedTerminalProcessReactorFailure) -> None:
        """Record one process-fatal callback.

        :param failure: Exact reactor failure.
        """

        failures.append(failure)
        fatal_event.set()

    reactor = PackedTerminalProcessReactor.for_source(serving, process_fatal)
    reactor.start(_WAIT_SECONDS)
    serving.signal(0)
    assert fatal_event.wait(_WAIT_SECONDS)
    assert reactor.join(_WAIT_SECONDS)
    try:
        inventory = reactor.inventory()
        assert inventory.disposition is (
            PackedTerminalProcessReactorDisposition.PROCESS_FATAL
        )
        assert inventory.failure is not None
        assert inventory.failure.cause is (
            PackedTerminalProcessReactorFailureCause.SERVING_DRAIN_FAILURE
        )
        assert inventory.failure.formatted_traceback is not None
        assert not inventory.admission_open
        assert len(failures) == 1
    finally:
        _close_source(reactor, serving)


def test_unexpected_reactor_exit_is_process_fatal_and_cannot_restart() -> None:
    """A return outside stop is fatal and consumes the one-shot start."""

    serving = _SourceServing()
    failures: list[PackedTerminalProcessReactorFailure] = []
    fatal_event = threading.Event()

    def process_fatal(failure: PackedTerminalProcessReactorFailure) -> None:
        """Record one unexpected-death callback.

        :param failure: Exact reactor failure.
        """

        failures.append(failure)
        fatal_event.set()

    reactor = _UnexpectedExitReactor.for_source(serving, process_fatal)
    with contextlib.suppress(PackedTerminalProcessReactorError):
        reactor.start(_WAIT_SECONDS)
    assert fatal_event.wait(_WAIT_SECONDS)
    assert reactor.join(_WAIT_SECONDS)
    inventory = reactor.close(_WAIT_SECONDS)
    serving.close_wakes()
    assert inventory.failure is not None
    assert inventory.failure.cause is (
        PackedTerminalProcessReactorFailureCause.UNEXPECTED_EXIT
    )
    assert len(failures) == 1
    with pytest.raises(PackedTerminalProcessReactorClosedError, match="restart"):
        reactor.start(_WAIT_SECONDS)


def test_reactor_source_contains_no_polling_collective_or_scheduler_drain() -> None:
    """Process progress remains fd-driven and outside scheduler affinity."""

    source = inspect.getsource(PackedTerminalProcessReactor)
    forbidden = (
        "time.sleep(",
        ".poll(",
        "all_reduce(",
        "all_gather(",
        "barrier(",
        "drain_scheduler_at_loop_entry",
    )
    assert tuple(token for token in forbidden if token in source) == ()
