import dataclasses
import enum
import os
import selectors
import threading
import time
import traceback
from collections.abc import Callable
from typing import ClassVar

from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.source_serving import (
    PackedTerminalSourceServing,
)


class PackedTerminalProcessReactorError(RuntimeError):
    """Base failure raised by the process-lifetime serving reactor."""


class PackedTerminalProcessReactorClosedError(PackedTerminalProcessReactorError):
    """Operation rejected after admission or reactor lifetime ended."""


class PackedTerminalProcessReactorRole(enum.Enum):
    """Serving role whose forward-independent work the reactor owns."""

    SOURCE = "source"
    DECODE = "decode"


class PackedTerminalProcessReactorDisposition(enum.Enum):
    """Sticky process-lifetime reactor disposition."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    PROCESS_FATAL = "process_fatal"


class PackedTerminalProcessReactorFailureCause(enum.Enum):
    """First cause which makes process continuation unsafe."""

    STARTUP_TIMEOUT = "startup_timeout"
    STARTUP_FAILURE = "startup_failure"
    SELECTOR_FAILURE = "selector_failure"
    SERVING_DRAIN_FAILURE = "serving_drain_failure"
    COORDINATOR_DEADLINE_FAILURE = "coordinator_deadline_failure"
    CONTROL_CHANNEL_FAILURE = "control_channel_failure"
    UNEXPECTED_EXIT = "unexpected_exit"
    UNEXPECTED_EXCEPTION = "unexpected_exception"
    JOIN_TIMEOUT = "join_timeout"


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalProcessReactorFailure:
    """First process-fatal reactor failure.

    :ivar cause: Stable failure classification.
    :ivar reason: Reader-facing failure boundary.
    :ivar formatted_traceback: Complete traceback when an exception was active.
    :ivar occurred_ns: Local monotonic failure timestamp.
    """

    cause: PackedTerminalProcessReactorFailureCause
    reason: str
    formatted_traceback: str | None
    occurred_ns: int

    def __post_init__(self) -> None:
        """Validate immutable process-fatal evidence."""

        if type(self.cause) is not PackedTerminalProcessReactorFailureCause:
            raise TypeError("cause must be PackedTerminalProcessReactorFailureCause")
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if self.formatted_traceback is not None and (
            type(self.formatted_traceback) is not str
            or len(self.formatted_traceback) == 0
        ):
            raise ValueError("formatted_traceback must be non-empty when present")
        if type(self.occurred_ns) is not int or self.occurred_ns < 0:
            raise ValueError("occurred_ns must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalProcessReactorInventory:
    """Complete process-reactor lifecycle and progress evidence.

    :ivar role: Source or decode serving role.
    :ivar disposition: Sticky reactor lifecycle disposition.
    :ivar thread_name: Stable process-thread identity.
    :ivar runtime_filenos: Borrowed serving descriptors registered at startup.
    :ivar started: Whether the one-shot start was consumed.
    :ivar startup_completed: Whether descriptor registration completed.
    :ivar reactor_alive: Whether the reactor thread is currently live.
    :ivar thread_finished: Whether the started thread has exited.
    :ivar admission_open: Whether composition-level request admission is open.
    :ivar stop_requested: Whether explicit or fatal stop was requested.
    :ivar closed: Whether the owned control descriptors are closed.
    :ivar control_read_open: Whether the selector-side wake descriptor is open.
    :ivar control_write_open: Whether the producer-side wake descriptor is open.
    :ivar selector_wait_count: Number of blocking selector waits completed.
    :ivar runtime_ready_descriptor_count: Runtime descriptors reported readable.
    :ivar runtime_drain_count: Coalesced serving drains executed.
    :ivar runtime_item_count: Immutable serving items consumed by those drains.
    :ivar control_wake_count: Coalesced control wakes consumed.
    :ivar deadline_notification_count: External exact-deadline rearm notices.
    :ivar deadline_evaluation_count: Fresh deadline values read from serving.
    :ivar deadline_expiration_count: Exact deadline-expiration calls executed.
    :ivar expired_coordinator_count: Coordinators terminalized by expiration.
    :ivar next_coordinator_deadline_ns: Last exact armed decode deadline.
    :ivar started_ns: Local monotonic successful-start timestamp.
    :ivar stopped_ns: Local monotonic reactor-exit timestamp.
    :ivar failure: First sticky process-fatal failure, when present.
    :ivar process_fatal_callback_delivered: Whether fatal delivery was attempted.
    :ivar process_fatal_callback_traceback: Callback failure traceback, if any.
    """

    role: PackedTerminalProcessReactorRole
    disposition: PackedTerminalProcessReactorDisposition
    thread_name: str
    runtime_filenos: tuple[int, ...]
    started: bool
    startup_completed: bool
    reactor_alive: bool
    thread_finished: bool
    admission_open: bool
    stop_requested: bool
    closed: bool
    control_read_open: bool
    control_write_open: bool
    selector_wait_count: int
    runtime_ready_descriptor_count: int
    runtime_drain_count: int
    runtime_item_count: int
    control_wake_count: int
    deadline_notification_count: int
    deadline_evaluation_count: int
    deadline_expiration_count: int
    expired_coordinator_count: int
    next_coordinator_deadline_ns: int | None
    started_ns: int | None
    stopped_ns: int | None
    failure: PackedTerminalProcessReactorFailure | None
    process_fatal_callback_delivered: bool
    process_fatal_callback_traceback: str | None

    def __post_init__(self) -> None:
        """Validate one internally consistent reactor inventory."""

        if type(self.role) is not PackedTerminalProcessReactorRole:
            raise TypeError("role must be PackedTerminalProcessReactorRole")
        if type(self.disposition) is not PackedTerminalProcessReactorDisposition:
            raise TypeError(
                "disposition must be PackedTerminalProcessReactorDisposition"
            )
        if type(self.thread_name) is not str or len(self.thread_name) == 0:
            raise ValueError("thread_name must be a non-empty string")
        if type(self.runtime_filenos) is not tuple or any(
            type(fd) is not int or fd < 0 for fd in self.runtime_filenos
        ):
            raise ValueError("runtime_filenos must contain non-negative integers")
        if len(set(self.runtime_filenos)) != len(self.runtime_filenos):
            raise ValueError("runtime_filenos must be unique")
        flags = (
            self.started,
            self.startup_completed,
            self.reactor_alive,
            self.thread_finished,
            self.admission_open,
            self.stop_requested,
            self.closed,
            self.control_read_open,
            self.control_write_open,
            self.process_fatal_callback_delivered,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("reactor inventory flags must be booleans")
        counts = (
            self.selector_wait_count,
            self.runtime_ready_descriptor_count,
            self.runtime_drain_count,
            self.runtime_item_count,
            self.control_wake_count,
            self.deadline_notification_count,
            self.deadline_evaluation_count,
            self.deadline_expiration_count,
            self.expired_coordinator_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("reactor inventory counts must be non-negative integers")
        timestamps = (
            self.next_coordinator_deadline_ns,
            self.started_ns,
            self.stopped_ns,
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in timestamps
        ):
            raise ValueError("reactor timestamps must be non-negative when present")
        if self.failure is not None and (
            type(self.failure) is not PackedTerminalProcessReactorFailure
        ):
            raise TypeError("failure must be PackedTerminalProcessReactorFailure")
        if self.process_fatal_callback_traceback is not None and (
            type(self.process_fatal_callback_traceback) is not str
            or len(self.process_fatal_callback_traceback) == 0
        ):
            raise ValueError(
                "process_fatal_callback_traceback must be non-empty when present"
            )
        if self.admission_open and (
            self.disposition is not PackedTerminalProcessReactorDisposition.RUNNING
            or not self.reactor_alive
        ):
            raise ValueError("admission can open only on a running live reactor")
        if self.disposition is PackedTerminalProcessReactorDisposition.PROCESS_FATAL:
            if self.failure is None or self.admission_open:
                raise ValueError("process-fatal inventory must retain failure evidence")
        elif self.failure is not None:
            raise ValueError("nonfatal disposition cannot retain fatal evidence")
        if self.closed and (self.control_read_open or self.control_write_open):
            raise ValueError("closed reactor inventory retains a control descriptor")


class _PackedTerminalProcessReactorLoopError(RuntimeError):
    """Typed internal transfer from the reactor loop to fatal handling."""

    cause: PackedTerminalProcessReactorFailureCause
    formatted_traceback: str

    def __init__(
        self,
        cause: PackedTerminalProcessReactorFailureCause,
        reason: str,
        formatted_traceback: str,
    ) -> None:
        """Retain one exact loop failure.

        :param cause: Stable process-fatal classification.
        :param reason: Reader-facing failure boundary.
        :param formatted_traceback: Complete originating traceback.
        """

        self.cause = cause
        self.formatted_traceback = formatted_traceback
        super().__init__(reason)


class _PackedTerminalProcessReactorSelectorKey(enum.Enum):
    """Descriptor class registered in the process selector."""

    RUNTIME = "runtime"
    CONTROL = "control"


class PackedTerminalProcessReactor:
    """Drain serving work independently from model-forward execution.

    The reactor owns borrowed runtime descriptors for exactly one source or
    decode serving composition. Runtime eventfds are coalesced into one serving
    drain per selector wake. Decode deadlines use an absolute serving timestamp
    converted into one exact selector wait, then are read again after every
    runtime drain, coordinator expiration, or explicit deadline-change wake.
    Scheduler inbox consumption remains outside this thread.
    """

    _CONTROL_WAKE: ClassVar[bytes] = b"\x01"

    _role: PackedTerminalProcessReactorRole
    _source_serving: PackedTerminalSourceServing | None
    _decode_serving: PackedTerminalDecodeServing | None
    _coordinator_clock_ns: Callable[[], int] | None
    _process_fatal_handler: Callable[[PackedTerminalProcessReactorFailure], None]
    _thread_name: str
    _runtime_filenos: tuple[int, ...]
    _control_read_fd: int
    _control_write_fd: int
    _control_read_open: bool
    _control_write_open: bool
    _condition: threading.Condition
    _thread: threading.Thread
    _disposition: PackedTerminalProcessReactorDisposition
    _started: bool
    _startup_completed: bool
    _reactor_alive: bool
    _thread_finished: bool
    _admission_open: bool
    _stop_requested: bool
    _closed: bool
    _selector_wait_count: int
    _runtime_ready_descriptor_count: int
    _runtime_drain_count: int
    _runtime_item_count: int
    _control_wake_count: int
    _deadline_notification_count: int
    _deadline_evaluation_count: int
    _deadline_expiration_count: int
    _expired_coordinator_count: int
    _next_coordinator_deadline_ns: int | None
    _started_ns: int | None
    _stopped_ns: int | None
    _failure: PackedTerminalProcessReactorFailure | None
    _process_fatal_callback_delivered: bool
    _process_fatal_callback_traceback: str | None

    def __init__(
        self,
        *,
        source_serving: PackedTerminalSourceServing | None,
        decode_serving: PackedTerminalDecodeServing | None,
        coordinator_clock_ns: Callable[[], int] | None,
        process_fatal_handler: Callable[[PackedTerminalProcessReactorFailure], None],
        thread_name: str,
    ) -> None:
        """Construct one dormant process reactor and its stop wake.

        :param source_serving: Exact source composition, otherwise ``None``.
        :param decode_serving: Exact decode composition, otherwise ``None``.
        :param coordinator_clock_ns: Decode coordinator's monotonic clock.
        :param process_fatal_handler: One-shot process-fatal delivery boundary.
        :param thread_name: Stable process-thread identity.
        """

        if (source_serving is None) == (decode_serving is None):
            raise ValueError("exactly one serving composition is required")
        if source_serving is not None and not isinstance(
            source_serving, PackedTerminalSourceServing
        ):
            raise TypeError("source_serving must be PackedTerminalSourceServing")
        if decode_serving is not None and not isinstance(
            decode_serving, PackedTerminalDecodeServing
        ):
            raise TypeError("decode_serving must be PackedTerminalDecodeServing")
        if decode_serving is None and coordinator_clock_ns is not None:
            raise ValueError("source reactors cannot carry a coordinator clock")
        if decode_serving is not None and not callable(coordinator_clock_ns):
            raise TypeError("decode reactors require a coordinator clock")
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        if type(thread_name) is not str or len(thread_name) == 0:
            raise ValueError("thread_name must be a non-empty string")

        role = (
            PackedTerminalProcessReactorRole.SOURCE
            if source_serving is not None
            else PackedTerminalProcessReactorRole.DECODE
        )
        serving = source_serving if source_serving is not None else decode_serving
        if serving is None:
            raise RuntimeError("validated serving composition disappeared")
        runtime_filenos = serving.runtime_filenos
        if type(runtime_filenos) is not tuple or len(runtime_filenos) == 0:
            raise ValueError("serving runtime_filenos must be a non-empty tuple")
        if any(type(fd) is not int or fd < 0 for fd in runtime_filenos):
            raise ValueError("serving runtime_filenos must be open descriptors")
        if len(set(runtime_filenos)) != len(runtime_filenos):
            raise ValueError("serving runtime_filenos must be unique")

        control_read_fd, control_write_fd = os.pipe()
        os.set_blocking(control_read_fd, False)
        os.set_blocking(control_write_fd, False)
        os.set_inheritable(control_read_fd, False)
        os.set_inheritable(control_write_fd, False)
        if control_read_fd in runtime_filenos or control_write_fd in runtime_filenos:
            os.close(control_read_fd)
            os.close(control_write_fd)
            raise RuntimeError("control descriptors alias a serving descriptor")

        self._role = role
        self._source_serving = source_serving
        self._decode_serving = decode_serving
        self._coordinator_clock_ns = coordinator_clock_ns
        self._process_fatal_handler = process_fatal_handler
        self._thread_name = thread_name
        self._runtime_filenos = runtime_filenos
        self._control_read_fd = control_read_fd
        self._control_write_fd = control_write_fd
        self._control_read_open = True
        self._control_write_open = True
        self._condition = threading.Condition()
        self._disposition = PackedTerminalProcessReactorDisposition.CREATED
        self._started = False
        self._startup_completed = False
        self._reactor_alive = False
        self._thread_finished = False
        self._admission_open = False
        self._stop_requested = False
        self._closed = False
        self._selector_wait_count = 0
        self._runtime_ready_descriptor_count = 0
        self._runtime_drain_count = 0
        self._runtime_item_count = 0
        self._control_wake_count = 0
        self._deadline_notification_count = 0
        self._deadline_evaluation_count = 0
        self._deadline_expiration_count = 0
        self._expired_coordinator_count = 0
        self._next_coordinator_deadline_ns = None
        self._started_ns = None
        self._stopped_ns = None
        self._failure = None
        self._process_fatal_callback_delivered = False
        self._process_fatal_callback_traceback = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    @classmethod
    def for_source(
        cls,
        serving: PackedTerminalSourceServing,
        process_fatal_handler: Callable[[PackedTerminalProcessReactorFailure], None],
        thread_name: str = "packed-terminal-source-serving-reactor",
    ) -> "PackedTerminalProcessReactor":
        """Construct one source process reactor.

        :param serving: Exact source serving composition.
        :param process_fatal_handler: One-shot process-fatal delivery boundary.
        :param thread_name: Stable process-thread identity.
        :returns: Dormant source reactor.
        """

        return cls(
            source_serving=serving,
            decode_serving=None,
            coordinator_clock_ns=None,
            process_fatal_handler=process_fatal_handler,
            thread_name=thread_name,
        )

    @classmethod
    def for_decode(
        cls,
        serving: PackedTerminalDecodeServing,
        coordinator_clock_ns: Callable[[], int],
        process_fatal_handler: Callable[[PackedTerminalProcessReactorFailure], None],
        thread_name: str = "packed-terminal-decode-serving-reactor",
    ) -> "PackedTerminalProcessReactor":
        """Construct one decode process reactor.

        :param serving: Exact decode serving composition.
        :param coordinator_clock_ns: Same monotonic clock used by coordinators.
        :param process_fatal_handler: One-shot process-fatal delivery boundary.
        :param thread_name: Stable process-thread identity.
        :returns: Dormant decode reactor.
        """

        return cls(
            source_serving=None,
            decode_serving=serving,
            coordinator_clock_ns=coordinator_clock_ns,
            process_fatal_handler=process_fatal_handler,
            thread_name=thread_name,
        )

    def start(self, timeout_seconds: float) -> None:
        """Start descriptor ownership exactly once and await registration.

        :param timeout_seconds: Positive startup handshake bound.
        """

        self._validate_positive_timeout(timeout_seconds)
        with self._condition:
            if self._started or self._closed:
                raise PackedTerminalProcessReactorClosedError(
                    "process reactor cannot restart"
                )
            self._started = True
            self._disposition = PackedTerminalProcessReactorDisposition.STARTING
            self._thread.start()
            expires_at = time.monotonic() + timeout_seconds
            while (
                not self._startup_completed
                and self._failure is None
                and not self._thread_finished
            ):
                remaining = expires_at - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            startup_completed = self._startup_completed
            failure = self._failure

        if startup_completed and failure is None:
            return
        if failure is None:
            self._record_failure(
                PackedTerminalProcessReactorFailureCause.STARTUP_TIMEOUT,
                "process reactor descriptor registration timed out",
                None,
            )
            self._signal_control_wake()
        raise PackedTerminalProcessReactorError(
            "process reactor failed before startup completed"
        )

    def require_admission_open(self) -> None:
        """Reject request admission outside the running reactor lifetime."""

        with self._condition:
            if not self._admission_open:
                raise PackedTerminalProcessReactorClosedError(
                    "process reactor request admission is closed"
                )

    def stop_admission(self) -> None:
        """Close composition-level request admission without stopping drains."""

        with self._condition:
            if not self._started:
                raise PackedTerminalProcessReactorClosedError(
                    "process reactor is not started"
                )
            self._admission_open = False
            self._condition.notify_all()

    def notify_coordinator_deadline_changed(self) -> None:
        """Wake a decode reactor after direct coordinator ingress.

        Runtime-driven coordinator changes need no call because every runtime
        drain automatically re-reads the deadline. A control receiver which
        mutates decode coordination directly must issue this wake after the
        mutation, so an earlier deadline cannot remain hidden behind an older
        selector timeout.
        """

        if self._role is not PackedTerminalProcessReactorRole.DECODE:
            raise PackedTerminalProcessReactorError(
                "source reactors have no coordinator deadline"
            )
        with self._condition:
            if self._closed or self._stop_requested:
                raise PackedTerminalProcessReactorClosedError(
                    "process reactor no longer accepts deadline changes"
                )
            self._deadline_notification_count += 1
            should_wake = self._started and not self._thread_finished
        if should_wake:
            self._signal_control_wake()

    def begin_stop(self) -> None:
        """Close admission and wake the blocking reactor into final drain."""

        with self._condition:
            if not self._started:
                raise PackedTerminalProcessReactorClosedError(
                    "process reactor is not started"
                )
            if self._closed or self._stop_requested:
                return
            self._admission_open = False
            self._stop_requested = True
            if self._failure is None and not self._thread_finished:
                self._disposition = PackedTerminalProcessReactorDisposition.STOPPING
            should_wake = not self._thread_finished
            self._condition.notify_all()
        if should_wake:
            self._signal_control_wake()

    def join(self, timeout_seconds: float) -> bool:
        """Join the process reactor within one explicit bound.

        :param timeout_seconds: Non-negative join bound.
        :returns: Whether the thread has stopped.
        """

        if type(timeout_seconds) is not float or timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must be a non-negative float")
        with self._condition:
            started = self._started
        if not started:
            return True
        if threading.current_thread() is self._thread:
            raise PackedTerminalProcessReactorError(
                "process reactor cannot join itself"
            )
        self._thread.join(timeout=timeout_seconds)
        return not self._thread.is_alive()

    def close(self, timeout_seconds: float) -> PackedTerminalProcessReactorInventory:
        """Stop, join, and close owned wake descriptors within one bound.

        :param timeout_seconds: Positive join bound.
        :returns: Final typed reactor inventory.
        :raises TimeoutError: If the live reactor does not stop within the bound.
        """

        self._validate_positive_timeout(timeout_seconds)
        with self._condition:
            if self._closed:
                return self._inventory_locked()
            started = self._started
        if started:
            self.begin_stop()
            if not self.join(timeout_seconds):
                self._record_failure(
                    PackedTerminalProcessReactorFailureCause.JOIN_TIMEOUT,
                    "process reactor did not stop within its close bound",
                    None,
                )
                raise TimeoutError("process reactor close timed out")
        else:
            with self._condition:
                self._stop_requested = True
                self._thread_finished = True
                self._disposition = PackedTerminalProcessReactorDisposition.STOPPED

        close_failure: str | None = None
        for fd, field_name in (
            (self._control_read_fd, "_control_read_open"),
            (self._control_write_fd, "_control_write_open"),
        ):
            with self._condition:
                is_open = (
                    self._control_read_open
                    if field_name == "_control_read_open"
                    else self._control_write_open
                )
            if not is_open:
                continue
            try:
                os.close(fd)
            except OSError:
                close_failure = traceback.format_exc()
                continue
            with self._condition:
                if field_name == "_control_read_open":
                    self._control_read_open = False
                else:
                    self._control_write_open = False
        if close_failure is not None:
            self._record_failure(
                PackedTerminalProcessReactorFailureCause.CONTROL_CHANNEL_FAILURE,
                "process reactor control descriptor close failed",
                close_failure,
            )
            raise PackedTerminalProcessReactorError(
                "process reactor control descriptor close failed"
            )
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            return self._inventory_locked()

    def inventory(self) -> PackedTerminalProcessReactorInventory:
        """Return complete immutable reactor evidence.

        :returns: Current process-lifetime inventory.
        """

        with self._condition:
            return self._inventory_locked()

    def _run(self) -> None:
        """Own all forward-independent serving drains until explicit stop."""

        selector: selectors.BaseSelector | None = None
        try:
            selector = selectors.DefaultSelector()
            for fd in self._runtime_filenos:
                selector.register(
                    fd,
                    selectors.EVENT_READ,
                    _PackedTerminalProcessReactorSelectorKey.RUNTIME,
                )
            selector.register(
                self._control_read_fd,
                selectors.EVENT_READ,
                _PackedTerminalProcessReactorSelectorKey.CONTROL,
            )
            with self._condition:
                self._startup_completed = True
                self._reactor_alive = True
                if self._failure is None and not self._stop_requested:
                    self._admission_open = True
                    self._disposition = PackedTerminalProcessReactorDisposition.RUNNING
                    self._started_ns = time.monotonic_ns()
                self._condition.notify_all()
            self._run_loop(selector)
            with self._condition:
                expected_exit = self._stop_requested
            if not expected_exit:
                self._record_failure(
                    PackedTerminalProcessReactorFailureCause.UNEXPECTED_EXIT,
                    "process reactor returned outside explicit stop",
                    None,
                )
        except _PackedTerminalProcessReactorLoopError as error:
            self._record_failure(error.cause, str(error), error.formatted_traceback)
        except BaseException:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            with self._condition:
                startup_completed = self._startup_completed
            cause = (
                PackedTerminalProcessReactorFailureCause.UNEXPECTED_EXCEPTION
                if startup_completed
                else PackedTerminalProcessReactorFailureCause.STARTUP_FAILURE
            )
            self._record_failure(
                cause,
                "process reactor raised unexpectedly",
                formatted_traceback,
            )
        finally:
            if selector is not None:
                try:
                    selector.close()
                except BaseException:  # noqa: BLE001
                    self._record_failure(
                        PackedTerminalProcessReactorFailureCause.SELECTOR_FAILURE,
                        "process reactor selector close failed",
                        traceback.format_exc(),
                    )
            with self._condition:
                self._reactor_alive = False
                self._admission_open = False
                self._thread_finished = True
                self._stopped_ns = time.monotonic_ns()
                if self._failure is None:
                    self._disposition = PackedTerminalProcessReactorDisposition.STOPPED
                self._condition.notify_all()

    def _run_loop(self, selector: selectors.BaseSelector) -> None:
        """Block on runtime, deadline, and explicit-control readiness.

        :param selector: Fully registered process-lifetime selector.
        """

        while True:
            with self._condition:
                if self._stop_requested:
                    return
            timeout_seconds = self._evaluate_deadlines()
            with self._condition:
                if self._stop_requested:
                    return
            try:
                ready = selector.select(timeout=timeout_seconds)
            except BaseException as error:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.SELECTOR_FAILURE,
                    "process reactor selector wait failed",
                    traceback.format_exc(),
                ) from error
            runtime_ready_count = sum(
                1
                for key, _ in ready
                if key.data is _PackedTerminalProcessReactorSelectorKey.RUNTIME
            )
            control_ready = any(
                key.data is _PackedTerminalProcessReactorSelectorKey.CONTROL
                for key, _ in ready
            )
            with self._condition:
                self._selector_wait_count += 1
                self._runtime_ready_descriptor_count += runtime_ready_count
            if runtime_ready_count > 0:
                consumed = self._drain_serving()
                with self._condition:
                    self._runtime_drain_count += 1
                    self._runtime_item_count += consumed
                    self._condition.notify_all()
            if control_ready:
                self._drain_control_wake()
                with self._condition:
                    self._control_wake_count += 1
                    self._condition.notify_all()

    def _evaluate_deadlines(self) -> float | None:
        """Expire due coordinators and derive the next exact selector wait.

        :returns: Exact relative wait to the current absolute deadline.
        """

        serving = self._decode_serving
        if serving is None:
            with self._condition:
                self._next_coordinator_deadline_ns = None
            return None
        clock_ns = self._coordinator_clock_ns
        if clock_ns is None:
            raise RuntimeError("decode reactor lost its coordinator clock")
        while True:
            try:
                deadline_ns = serving.next_coordinator_deadline_ns
                now_ns = clock_ns()
            except BaseException as error:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "decode coordinator deadline read failed",
                    traceback.format_exc(),
                ) from error
            if deadline_ns is not None and (
                type(deadline_ns) is not int or deadline_ns < 0
            ):
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "decode coordinator returned an invalid deadline",
                    "deadline value failed structural validation",
                )
            if type(now_ns) is not int or now_ns < 0:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "decode coordinator clock returned an invalid timestamp",
                    "clock value failed structural validation",
                )
            with self._condition:
                self._deadline_evaluation_count += 1
                self._next_coordinator_deadline_ns = deadline_ns
                self._condition.notify_all()
            if deadline_ns is None:
                return None
            if deadline_ns > now_ns:
                return (deadline_ns - now_ns) / 1_000_000_000.0
            try:
                expired_count = serving.expire_coordinators(now_ns)
            except BaseException as error:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "decode coordinator expiration failed",
                    traceback.format_exc(),
                ) from error
            if type(expired_count) is not int or expired_count < 0:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "decode coordinator expiration returned an invalid count",
                    "expiration count failed structural validation",
                )
            with self._condition:
                self._deadline_expiration_count += 1
                self._expired_coordinator_count += expired_count
                self._condition.notify_all()
            if expired_count == 0:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.COORDINATOR_DEADLINE_FAILURE,
                    "due decode coordinator deadline made no terminal progress",
                    "a due deadline remained armed after a zero-progress expiration",
                )

    def _drain_serving(self) -> int:
        """Execute one coalesced all-context serving drain.

        :returns: Number of immutable actions and observations consumed.
        """

        try:
            if self._source_serving is not None:
                consumed = self._source_serving.drain_runtime_actions()
            elif self._decode_serving is not None:
                consumed = self._decode_serving.drain_runtime_actions()
            else:
                raise RuntimeError("process reactor lost its serving composition")
        except BaseException as error:
            raise _PackedTerminalProcessReactorLoopError(
                PackedTerminalProcessReactorFailureCause.SERVING_DRAIN_FAILURE,
                "process reactor serving drain failed",
                traceback.format_exc(),
            ) from error
        if type(consumed) is not int or consumed < 0:
            raise _PackedTerminalProcessReactorLoopError(
                PackedTerminalProcessReactorFailureCause.SERVING_DRAIN_FAILURE,
                "process reactor serving drain returned an invalid count",
                "serving drain count failed structural validation",
            )
        return consumed

    def _signal_control_wake(self) -> None:
        """Publish one coalesced stop or deadline-change wake."""

        with self._condition:
            if not self._control_write_open:
                return
            fd = self._control_write_fd
        try:
            os.write(fd, self._CONTROL_WAKE)
        except BlockingIOError:
            return
        except OSError:
            self._record_failure(
                PackedTerminalProcessReactorFailureCause.CONTROL_CHANNEL_FAILURE,
                "process reactor control wake failed",
                traceback.format_exc(),
            )

    def _drain_control_wake(self) -> None:
        """Consume the complete coalesced control-pipe population."""

        while True:
            try:
                payload = os.read(self._control_read_fd, 4096)
            except BlockingIOError:
                return
            except OSError as error:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.CONTROL_CHANNEL_FAILURE,
                    "process reactor control wake drain failed",
                    traceback.format_exc(),
                ) from error
            if len(payload) == 0:
                raise _PackedTerminalProcessReactorLoopError(
                    PackedTerminalProcessReactorFailureCause.CONTROL_CHANNEL_FAILURE,
                    "process reactor control channel closed unexpectedly",
                    "control wake reached EOF before reactor shutdown",
                )

    def _record_failure(
        self,
        cause: PackedTerminalProcessReactorFailureCause,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Retain and deliver the first process-fatal failure exactly once.

        :param cause: Stable failure classification.
        :param reason: Reader-facing failure boundary.
        :param formatted_traceback: Complete originating traceback, if any.
        """

        failure = PackedTerminalProcessReactorFailure(
            cause=cause,
            reason=reason,
            formatted_traceback=formatted_traceback,
            occurred_ns=time.monotonic_ns(),
        )
        with self._condition:
            if self._failure is not None:
                return
            self._failure = failure
            self._admission_open = False
            self._stop_requested = True
            self._disposition = PackedTerminalProcessReactorDisposition.PROCESS_FATAL
            self._process_fatal_callback_delivered = True
            self._condition.notify_all()
        try:
            self._process_fatal_handler(failure)
        except BaseException:  # noqa: BLE001
            callback_traceback = traceback.format_exc()
            with self._condition:
                self._process_fatal_callback_traceback = callback_traceback
                self._condition.notify_all()

    def _inventory_locked(self) -> PackedTerminalProcessReactorInventory:
        """Build immutable inventory while holding the lifecycle condition.

        :returns: Current complete inventory.
        """

        return PackedTerminalProcessReactorInventory(
            role=self._role,
            disposition=self._disposition,
            thread_name=self._thread_name,
            runtime_filenos=self._runtime_filenos,
            started=self._started,
            startup_completed=self._startup_completed,
            reactor_alive=self._reactor_alive,
            thread_finished=self._thread_finished,
            admission_open=self._admission_open,
            stop_requested=self._stop_requested,
            closed=self._closed,
            control_read_open=self._control_read_open,
            control_write_open=self._control_write_open,
            selector_wait_count=self._selector_wait_count,
            runtime_ready_descriptor_count=self._runtime_ready_descriptor_count,
            runtime_drain_count=self._runtime_drain_count,
            runtime_item_count=self._runtime_item_count,
            control_wake_count=self._control_wake_count,
            deadline_notification_count=self._deadline_notification_count,
            deadline_evaluation_count=self._deadline_evaluation_count,
            deadline_expiration_count=self._deadline_expiration_count,
            expired_coordinator_count=self._expired_coordinator_count,
            next_coordinator_deadline_ns=self._next_coordinator_deadline_ns,
            started_ns=self._started_ns,
            stopped_ns=self._stopped_ns,
            failure=self._failure,
            process_fatal_callback_delivered=(self._process_fatal_callback_delivered),
            process_fatal_callback_traceback=(self._process_fatal_callback_traceback),
        )

    @staticmethod
    def _validate_positive_timeout(timeout_seconds: float) -> None:
        """Validate one explicit positive lifecycle bound.

        :param timeout_seconds: Candidate timeout value.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
