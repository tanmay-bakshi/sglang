import dataclasses
import enum
import errno
import json
import logging
import os
import selectors
import threading
import traceback
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalActionInbox,
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
)

logger = logging.getLogger(__name__)

TERMINAL_DECODE_SCATTER_TIMING_LOG_PREFIX = "TerminalDecodeScatterTiming:"


class PackedTerminalDecodeScatterWorkerDisposition(enum.StrEnum):
    """Process-lifetime state of the decode scatter execution context."""

    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    PROCESS_FATAL = "process_fatal"


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeScatterWorkerInventory:
    """Exact liveness and retained authority of one decode scatter worker.

    :ivar disposition: Current process-lifetime worker state.
    :ivar admission_closed: Whether explicit clean or abort drain has begun.
    :ivar abort_requested: Whether remaining actions are being discarded closed.
    :ivar cuda_device_bound: Whether the worker established CUDA device affinity.
    :ivar thread_alive: Whether the worker thread is currently alive.
    :ivar queued_action_count: Actions retained in the direct runtime inbox.
    :ivar active_action_id: Action currently executing, otherwise ``None``.
    :ivar active_binding_digest: Binding currently executing, otherwise ``None``.
    :ivar completed_action_count: Cumulative successful scatter actions.
    :ivar aborted_action_count: Cumulative fail-closed discarded actions.
    :ivar last_queue_residence_ns: Most recent commit-to-worker-start interval.
    :ivar maximum_queue_residence_ns: Largest observed queue-residence interval.
    :ivar last_submission_duration_ns: Most recent host submission duration.
    :ivar maximum_submission_duration_ns: Largest host submission duration.
    :ivar fatal_reason: Sticky worker failure evidence, when present.
    :ivar fatal_traceback: Complete worker traceback, when present.
    """

    disposition: PackedTerminalDecodeScatterWorkerDisposition
    admission_closed: bool
    abort_requested: bool
    cuda_device_bound: bool
    thread_alive: bool
    queued_action_count: int
    active_action_id: int | None
    active_binding_digest: bytes | None
    completed_action_count: int
    aborted_action_count: int
    last_queue_residence_ns: int | None
    maximum_queue_residence_ns: int
    last_submission_duration_ns: int | None
    maximum_submission_duration_ns: int
    fatal_reason: str | None
    fatal_traceback: str | None

    def __post_init__(self) -> None:
        """Validate one complete immutable worker snapshot."""

        if type(self.disposition) is not PackedTerminalDecodeScatterWorkerDisposition:
            raise TypeError("disposition must be a decode scatter worker disposition")
        flags = (
            self.admission_closed,
            self.abort_requested,
            self.cuda_device_bound,
            self.thread_alive,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("decode scatter worker flags must be bool")
        counts = (
            self.queued_action_count,
            self.completed_action_count,
            self.aborted_action_count,
            self.maximum_queue_residence_ns,
            self.maximum_submission_duration_ns,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("decode scatter worker counts must be non-negative")
        if self.active_action_id is not None and (
            type(self.active_action_id) is not int or self.active_action_id < 0
        ):
            raise ValueError("active_action_id must be a non-negative integer")
        if self.active_binding_digest is not None and (
            type(self.active_binding_digest) is not bytes
            or len(self.active_binding_digest) != 32
        ):
            raise ValueError("active_binding_digest must contain 32 bytes")
        if (self.active_action_id is None) != (self.active_binding_digest is None):
            raise ValueError("active decode scatter identity must be complete")
        optional_durations = (
            self.last_queue_residence_ns,
            self.last_submission_duration_ns,
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in optional_durations
        ):
            raise ValueError("decode scatter durations must be non-negative")
        if self.fatal_reason is not None and (
            type(self.fatal_reason) is not str or len(self.fatal_reason) == 0
        ):
            raise ValueError("fatal_reason must be a non-empty string")
        if self.fatal_traceback is not None and (
            type(self.fatal_traceback) is not str or len(self.fatal_traceback) == 0
        ):
            raise ValueError("fatal_traceback must be a non-empty string")
        if self.fatal_traceback is not None and self.fatal_reason is None:
            raise ValueError("fatal_traceback requires a fatal_reason")

    @property
    def retained_action_count(self) -> int:
        """Count action authority retained by the worker and its direct inbox.

        :returns: Queued plus actively executing scatter actions.
        """

        return self.queued_action_count + int(self.active_action_id is not None)


class PackedTerminalDecodeScatterWorkerError(RuntimeError):
    """Dedicated decode scatter worker lifecycle failure."""


class PackedTerminalDecodeScatterWorker:
    """Own decode scatter submission on its direct runtime FD inbox."""

    _STOP_WAKE: bytes = b"\x01"

    _runtime: NativeTerminalRuntime
    _inbox: NativeTerminalActionInbox
    _consume_action: Callable[[NativeTerminalOwnerAction], None]
    _bind_cuda_device: Callable[[], None]
    _fatal_listener: Callable[[str, str | None], None]
    _clock_ns: Callable[[], int]
    _control_read_fd: int
    _control_write_fd: int
    _condition: threading.Condition
    _disposition: PackedTerminalDecodeScatterWorkerDisposition
    _started: bool
    _startup_complete: bool
    _admission_closed: bool
    _abort_requested: bool
    _cuda_device_bound: bool
    _thread_finished: bool
    _control_closed: bool
    _active_action: NativeTerminalOwnerAction | None
    _completed_action_count: int
    _aborted_action_count: int
    _last_queue_residence_ns: int | None
    _maximum_queue_residence_ns: int
    _last_submission_duration_ns: int | None
    _maximum_submission_duration_ns: int
    _fatal_reason: str | None
    _fatal_traceback: str | None
    _thread: threading.Thread

    def __init__(
        self,
        *,
        runtime: NativeTerminalRuntime,
        consume_action: Callable[[NativeTerminalOwnerAction], None],
        bind_cuda_device: Callable[[], None],
        fatal_listener: Callable[[str, str | None], None],
        clock_ns: Callable[[], int],
        thread_name: str = "packed-terminal-decode-scatter-worker",
    ) -> None:
        """Construct one dormant process-lifetime scatter worker.

        :param runtime: Runtime owning the bounded direct scatter inbox.
        :param consume_action: Exact one-shot scatter action consumer.
        :param bind_cuda_device: Worker-thread CUDA device binding operation.
        :param fatal_listener: Process-fatal notification boundary.
        :param clock_ns: Process-local ``CLOCK_MONOTONIC_RAW`` clock.
        :param thread_name: Stable worker thread identity.
        """

        if type(runtime) is not NativeTerminalRuntime:
            raise TypeError("runtime must be NativeTerminalRuntime")
        if not callable(consume_action):
            raise TypeError("consume_action must be callable")
        if not callable(bind_cuda_device):
            raise TypeError("bind_cuda_device must be callable")
        if not callable(fatal_listener):
            raise TypeError("fatal_listener must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if type(thread_name) is not str or len(thread_name) == 0:
            raise ValueError("thread_name must be a non-empty string")
        control_read_fd, control_write_fd = os.pipe()
        os.set_blocking(control_read_fd, False)
        os.set_blocking(control_write_fd, False)
        os.set_inheritable(control_read_fd, False)
        os.set_inheritable(control_write_fd, False)
        inbox = runtime.decode_scatter_actions
        if inbox.fileno() in (control_read_fd, control_write_fd):
            os.close(control_read_fd)
            os.close(control_write_fd)
            raise RuntimeError("decode scatter control descriptors alias its inbox")

        self._runtime = runtime
        self._inbox = inbox
        self._consume_action = consume_action
        self._bind_cuda_device = bind_cuda_device
        self._fatal_listener = fatal_listener
        self._clock_ns = clock_ns
        self._control_read_fd = control_read_fd
        self._control_write_fd = control_write_fd
        self._condition = threading.Condition()
        self._disposition = PackedTerminalDecodeScatterWorkerDisposition.CREATED
        self._started = False
        self._startup_complete = False
        self._admission_closed = False
        self._abort_requested = False
        self._cuda_device_bound = False
        self._thread_finished = False
        self._control_closed = False
        self._active_action = None
        self._completed_action_count = 0
        self._aborted_action_count = 0
        self._last_queue_residence_ns = None
        self._maximum_queue_residence_ns = 0
        self._last_submission_duration_ns = None
        self._maximum_submission_duration_ns = 0
        self._fatal_reason = None
        self._fatal_traceback = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    def start(self) -> None:
        """Start exactly once and require worker-thread CUDA affinity."""

        with self._condition:
            if self._started:
                raise PackedTerminalDecodeScatterWorkerError(
                    "decode scatter worker cannot restart"
                )
            self._started = True
        try:
            self._thread.start()
        except BaseException:
            formatted_traceback = traceback.format_exc()
            with self._condition:
                self._thread_finished = True
                self._startup_complete = True
                self._condition.notify_all()
            self._enter_fatal(
                "decode scatter worker thread failed to start",
                formatted_traceback,
            )
            raise
        with self._condition:
            while not self._startup_complete:
                self._condition.wait()
            if (
                self._disposition
                is PackedTerminalDecodeScatterWorkerDisposition.RUNNING
            ):
                return
            reason = self._fatal_reason or "decode scatter worker failed during startup"
        raise PackedTerminalDecodeScatterWorkerError(reason)

    def stop_and_join(
        self,
        timeout_seconds: float,
        *,
        abort: bool,
    ) -> PackedTerminalDecodeScatterWorkerInventory:
        """Drain or discard direct-inbox actions and join within one bound.

        :param timeout_seconds: Positive worker shutdown bound.
        :param abort: Whether remaining actions must be acknowledged fail closed.
        :returns: Final worker inventory after a successful join.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        if type(abort) is not bool:
            raise TypeError("abort must be bool")
        if abort and self._runtime.disposition not in (
            NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            NativeTerminalRuntimeDisposition.ABORT_DRAINING,
        ):
            self._runtime.begin_abort("decode scatter worker abort requested")
        with self._condition:
            if not self._started:
                if not abort:
                    raise PackedTerminalDecodeScatterWorkerError(
                        "decode scatter worker was never started"
                    )
                self._admission_closed = True
                self._abort_requested = True
                self._thread_finished = True
                self._disposition = PackedTerminalDecodeScatterWorkerDisposition.STOPPED
            self._admission_closed = True
            self._abort_requested = self._abort_requested or abort
            if (
                self._disposition
                is PackedTerminalDecodeScatterWorkerDisposition.RUNNING
            ):
                self._disposition = (
                    PackedTerminalDecodeScatterWorkerDisposition.DRAINING
                )
            thread_finished = self._thread_finished
            self._condition.notify_all()
        if not thread_finished:
            self._signal_control()
            if threading.current_thread() is self._thread:
                raise PackedTerminalDecodeScatterWorkerError(
                    "decode scatter worker cannot join itself"
                )
            self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            reason = "decode scatter worker did not stop within its shutdown bound"
            self._enter_fatal(reason, None)
            raise TimeoutError(reason)
        if abort:
            self._drain_aborted_actions()
        self._close_control()
        inventory = self.inventory()
        if not abort and inventory.fatal_reason is not None:
            raise PackedTerminalDecodeScatterWorkerError(inventory.fatal_reason)
        return inventory

    def inventory(self) -> PackedTerminalDecodeScatterWorkerInventory:
        """Return complete immutable worker liveness and authority.

        :returns: Current typed decode scatter worker inventory.
        """

        queued_count = self._inbox.snapshot().queued_count
        with self._condition:
            active = self._active_action
            return PackedTerminalDecodeScatterWorkerInventory(
                disposition=self._disposition,
                admission_closed=self._admission_closed,
                abort_requested=self._abort_requested,
                cuda_device_bound=self._cuda_device_bound,
                thread_alive=self._thread.is_alive(),
                queued_action_count=queued_count,
                active_action_id=None if active is None else active.action_id,
                active_binding_digest=(
                    None if active is None else active.binding.digest
                ),
                completed_action_count=self._completed_action_count,
                aborted_action_count=self._aborted_action_count,
                last_queue_residence_ns=self._last_queue_residence_ns,
                maximum_queue_residence_ns=self._maximum_queue_residence_ns,
                last_submission_duration_ns=self._last_submission_duration_ns,
                maximum_submission_duration_ns=self._maximum_submission_duration_ns,
                fatal_reason=self._fatal_reason,
                fatal_traceback=self._fatal_traceback,
            )

    def _run(self) -> None:
        """Bind CUDA once and own every decode scatter action to terminality."""

        selector: selectors.BaseSelector | None = None
        try:
            self._bind_cuda_device()
            selector = selectors.DefaultSelector()
            selector.register(self._inbox.fileno(), selectors.EVENT_READ)
            selector.register(self._control_read_fd, selectors.EVENT_READ)
            with self._condition:
                self._cuda_device_bound = True
                self._disposition = PackedTerminalDecodeScatterWorkerDisposition.RUNNING
                self._startup_complete = True
                self._condition.notify_all()
            self._run_loop(selector)
        except BaseException:  # noqa: BLE001
            self._enter_fatal(
                "decode scatter worker raised unexpectedly",
                traceback.format_exc(),
            )
        finally:
            if selector is not None:
                try:
                    selector.close()
                except BaseException:  # noqa: BLE001
                    self._enter_fatal(
                        "decode scatter worker selector close failed",
                        traceback.format_exc(),
                    )
            with self._condition:
                unexpected_exit = (
                    self._fatal_reason is None
                    and not self._admission_closed
                    and not self._abort_requested
                )
                if not self._startup_complete:
                    self._startup_complete = True
                if self._fatal_reason is None and not unexpected_exit:
                    self._disposition = (
                        PackedTerminalDecodeScatterWorkerDisposition.STOPPED
                    )
                self._thread_finished = True
                self._condition.notify_all()
            if unexpected_exit:
                self._enter_fatal(
                    "decode scatter worker exited without a shutdown request",
                    None,
                )

    def _run_loop(self, selector: selectors.BaseSelector) -> None:
        """Block on direct scatter or explicit stop readiness without polling.

        :param selector: Worker-owned direct-inbox selector.
        """

        while True:
            self._synchronize_runtime_abort()
            if self._should_exit():
                return
            selector.select()
            self._drain_control_wake()
            self._synchronize_runtime_abort()
            self._consume_one_action()

    def _consume_one_action(self) -> None:
        """Consume at most one direct-inbox action and preserve FIFO fairness."""

        actions = self._inbox.drain(maximum_items=1)
        if len(actions) == 0:
            return
        action = actions[0]
        if action.kind is not NativeTerminalOwnerActionKind.DECODE_SCATTER_READY:
            raise RuntimeError("decode scatter inbox carried another action kind")
        with self._condition:
            if self._active_action is not None:
                raise RuntimeError("decode scatter worker already owns an action")
            self._active_action = action
            abort_requested = self._abort_requested
            self._condition.notify_all()
        started_ns: int | None = None
        try:
            started_ns = self._clock_ns()
            if type(started_ns) is not int or started_ns < action.commit_timestamp_ns:
                raise RuntimeError("decode scatter worker clock precedes native commit")
            queue_residence_ns = started_ns - action.commit_timestamp_ns
            with self._condition:
                self._last_queue_residence_ns = queue_residence_ns
                self._maximum_queue_residence_ns = max(
                    self._maximum_queue_residence_ns,
                    queue_residence_ns,
                )
            if abort_requested:
                if self._acknowledge_aborted_action_if_pending(
                    action,
                    context="worker abort",
                ):
                    with self._condition:
                        self._aborted_action_count += 1
                return
            try:
                self._consume_action(action)
            except BaseException:  # noqa: BLE001
                formatted_traceback = traceback.format_exc()
                self._enter_fatal(
                    "decode scatter action failed",
                    formatted_traceback,
                )
                if self._acknowledge_aborted_action_if_pending(
                    action,
                    context="scatter action failure",
                ):
                    with self._condition:
                        self._aborted_action_count += 1
                return
            completed_ns = self._clock_ns()
            if type(completed_ns) is not int or completed_ns < started_ns:
                raise RuntimeError("decode scatter submission clock regressed")
            submission_duration_ns = completed_ns - started_ns
            with self._condition:
                self._completed_action_count += 1
                self._last_submission_duration_ns = submission_duration_ns
                self._maximum_submission_duration_ns = max(
                    self._maximum_submission_duration_ns,
                    submission_duration_ns,
                )
            self._emit_action_timing(
                action,
                queue_residence_ns=queue_residence_ns,
                submission_duration_ns=submission_duration_ns,
            )
        except BaseException:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            self._enter_fatal(
                "decode scatter worker timing or ownership failed",
                formatted_traceback,
            )
            if self._acknowledge_aborted_action_if_pending(
                action,
                context="timing or ownership failure",
            ):
                with self._condition:
                    self._aborted_action_count += 1
        finally:
            with self._condition:
                self._active_action = None
                self._condition.notify_all()

    def _drain_aborted_actions(self) -> None:
        """Discard queued original actions after worker infrastructure failure."""

        with self._condition:
            self._abort_requested = True
            self._condition.notify_all()
        while self._inbox.snapshot().queued_count > 0:
            actions = self._inbox.drain(maximum_items=1)
            if len(actions) == 0:
                return
            action = actions[0]
            if self._acknowledge_aborted_action_if_pending(
                action,
                context="abort drain",
            ):
                with self._condition:
                    self._aborted_action_count += 1
                    self._condition.notify_all()

    def _acknowledge_aborted_action_if_pending(
        self,
        action: NativeTerminalOwnerAction,
        *,
        context: str,
    ) -> bool:
        """Reconcile fail-closed authority without replacing its first cause.

        :param action: Exact scatter authority being discarded.
        :param context: Stable secondary-failure location.
        :returns: Whether the runtime still retained the original action.
        """

        try:
            return self._runtime.acknowledge_aborted_action_if_pending(action)
        except BaseException:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            self._enter_fatal(
                "decode scatter abort acknowledgement failed",
                formatted_traceback,
            )
            logger.critical(
                "Decode scatter abort acknowledgement failed during %s:\n%s",
                context,
                formatted_traceback,
            )
            return False

    def _emit_action_timing(
        self,
        action: NativeTerminalOwnerAction,
        *,
        queue_residence_ns: int,
        submission_duration_ns: int,
    ) -> None:
        """Emit one parser-stable scatter queue and host-submission record.

        :param action: Exact request-correlated scatter authority.
        :param queue_residence_ns: Native commit to worker start interval.
        :param submission_duration_ns: Worker host-submission interval.
        """

        try:
            payload = {
                "binding_digest": action.binding.digest.hex(),
                "host_submission_ms": submission_duration_ns / 1_000_000.0,
                "queue_residence_ms": queue_residence_ns / 1_000_000.0,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            logger.info("%s%s", TERMINAL_DECODE_SCATTER_TIMING_LOG_PREFIX, encoded)
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode scatter timing evidence projection failed:\n%s",
                traceback.format_exc(),
            )

    def _synchronize_runtime_abort(self) -> None:
        """Convert a runtime-wide fatal transition into worker abort drain."""

        if self._runtime.disposition not in (
            NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            NativeTerminalRuntimeDisposition.ABORT_DRAINING,
        ):
            return
        with self._condition:
            self._abort_requested = True
            self._condition.notify_all()

    def _should_exit(self) -> bool:
        """Return whether an explicit drain reached exact zero authority.

        :returns: Whether the worker may terminate.
        """

        with self._condition:
            draining = self._admission_closed
            active = self._active_action is not None
        return draining and not active and self._inbox.snapshot().queued_count == 0

    def _enter_fatal(self, reason: str, formatted_traceback: str | None) -> None:
        """Record one sticky failure and enter process-fatal runtime drain.

        :param reason: Stable worker failure evidence.
        :param formatted_traceback: Complete originating traceback, if available.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if formatted_traceback is not None and (
            type(formatted_traceback) is not str or len(formatted_traceback) == 0
        ):
            raise ValueError("formatted_traceback must be a non-empty string")
        with self._condition:
            first_failure = self._fatal_reason is None
            if first_failure:
                self._fatal_reason = reason
                self._fatal_traceback = formatted_traceback
            self._abort_requested = True
            self._disposition = (
                PackedTerminalDecodeScatterWorkerDisposition.PROCESS_FATAL
            )
            retained_reason = self._fatal_reason
            retained_traceback = self._fatal_traceback
            self._condition.notify_all()
        if not first_failure:
            return
        runtime_traceback: str | None = None
        try:
            self._runtime.begin_abort(retained_reason)
        except BaseException:  # noqa: BLE001
            runtime_traceback = traceback.format_exc()
        try:
            self._fatal_listener(retained_reason, retained_traceback)
        except BaseException:  # noqa: BLE001
            logger.critical(
                "Decode scatter fatal listener also failed:\n%s",
                traceback.format_exc(),
            )
        if runtime_traceback is not None:
            logger.critical(
                "Decode scatter runtime abort also failed:\n%s",
                runtime_traceback,
            )
        retained_traceback_text = retained_traceback
        if retained_traceback_text is None:
            retained_traceback_text = "no traceback available"
        logger.critical(
            "Decode scatter worker failed: %s\n%s",
            retained_reason,
            retained_traceback_text,
        )

    def _signal_control(self) -> None:
        """Wake the blocking worker after explicit drain or abort."""

        with self._condition:
            if self._control_closed:
                return
            descriptor = self._control_write_fd
        try:
            os.write(descriptor, self._STOP_WAKE)
        except BlockingIOError:
            return
        except OSError as error:
            if error.errno == errno.EINTR:
                self._signal_control()
                return
            raise

    def _drain_control_wake(self) -> None:
        """Consume the complete coalesced worker-control wake."""

        while True:
            try:
                payload = os.read(self._control_read_fd, 4096)
            except BlockingIOError:
                return
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                raise
            if len(payload) == 0 or len(payload) < 4096:
                return

    def _close_control(self) -> None:
        """Close worker-owned control descriptors after the thread joins."""

        with self._condition:
            if self._control_closed:
                return
            self._control_closed = True
            descriptors = (self._control_read_fd, self._control_write_fd)
        first_error: OSError | None = None
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
