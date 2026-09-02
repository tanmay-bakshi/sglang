import os
import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import NoReturn, Protocol

import torch
import torch.distributed as dist


class AsyncCollectiveWork(Protocol):
    """Asynchronous collective handle required by the fail-stop runner."""

    def is_completed(self) -> bool:
        """Return whether the collective has completed."""
        ...

    def wait(self) -> bool:
        """Return the completed collective result."""
        ...


class AllReduceLauncher(Protocol):
    """Launch an asynchronous all-reduce operation."""

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
        group: dist.ProcessGroup | None,
        async_op: bool,
    ) -> AsyncCollectiveWork:
        """Launch the operation and return its work handle."""
        ...


class FailStopCollectiveTerminated(RuntimeError):
    """Raised when an injected process-exit callback unexpectedly returns."""


class _AttemptOutcome(Enum):
    """Terminal election state shared by the caller and watchdog."""

    PENDING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


class _CollectiveAttempt:
    """Coordinate one collective caller with its independent watchdog."""

    _condition: threading.Condition
    _outcome: _AttemptOutcome
    _failure_reason: str | None
    _watchdog_started: threading.Event

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._outcome = _AttemptOutcome.PENDING
        self._failure_reason = None
        self._watchdog_started = threading.Event()

    def mark_watchdog_started(self) -> None:
        """Publish that the watchdog is running independently of the caller."""
        self._watchdog_started.set()

    def wait_until_watchdog_started(self) -> None:
        """Wait until collective launch is protected by the watchdog."""
        self._watchdog_started.wait()

    def try_succeed(self) -> bool:
        """Elect successful completion if failure has not already won.

        :returns: Whether this call elected successful completion.
        """
        with self._condition:
            if self._outcome is not _AttemptOutcome.PENDING:
                return False
            self._outcome = _AttemptOutcome.SUCCEEDED
            self._condition.notify_all()
            return True

    def fail(self, reason: str) -> None:
        """Elect fail-stop termination if completion has not already won.

        :param reason: Concise diagnostic describing the failure.
        """
        with self._condition:
            if self._outcome is not _AttemptOutcome.PENDING:
                return
            self._outcome = _AttemptOutcome.FAILED
            self._failure_reason = reason
            self._condition.notify_all()

    def outcome(self) -> _AttemptOutcome:
        """Return the current terminal election state.

        :returns: The current outcome.
        """
        with self._condition:
            return self._outcome

    def failure_reason(self) -> str:
        """Return the elected failure diagnostic.

        :returns: The failure diagnostic.
        :raises RuntimeError: If fail-stop termination has not been elected.
        """
        with self._condition:
            if self._failure_reason is None:
                raise RuntimeError("fail-stop termination has not been elected")
            return self._failure_reason

    def wait(self, timeout_seconds: float) -> None:
        """Wait for a terminal election or the next poll interval.

        :param timeout_seconds: Maximum number of seconds to wait.
        """
        with self._condition:
            if self._outcome is _AttemptOutcome.PENDING:
                self._condition.wait(timeout_seconds)


def _torch_all_reduce(
    tensor: torch.Tensor,
    *,
    op: dist.ReduceOp,
    group: dist.ProcessGroup | None,
    async_op: bool,
) -> AsyncCollectiveWork:
    """Launch the repository's torch-distributed all-reduce.

    :param tensor: Tensor reduced in place.
    :param op: Reduction operation.
    :param group: Process group participating in the reduction.
    :param async_op: Whether the collective should be asynchronous.
    :returns: The asynchronous work handle.
    :raises RuntimeError: If the backend returns no asynchronous work handle.
    """
    work = dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
    if work is None:
        raise RuntimeError("asynchronous all-reduce returned no work handle")
    return work


class FailStopCollective:
    """Run bounded all-reduces that terminate the worker on uncertain outcome."""

    _all_reduce_launcher: AllReduceLauncher
    _clock: Callable[[], float]
    _exit_callback: Callable[[int], None]
    _poll_interval_seconds: float
    _timeout_seconds: float

    def __init__(
        self,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
        exit_callback: Callable[[int], None] = os._exit,
        all_reduce_launcher: AllReduceLauncher = _torch_all_reduce,
    ) -> None:
        """Initialize a bounded collective runner.

        :param timeout_seconds: Maximum time allowed for one collective.
        :param poll_interval_seconds: Maximum delay between completion polls.
        :param clock: Monotonic clock used to enforce the deadline.
        :param exit_callback: Process-exit callback invoked with status one.
        :param all_reduce_launcher: Asynchronous all-reduce launch function.
        :raises ValueError: If either duration is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._exit_callback = exit_callback
        self._all_reduce_launcher = all_reduce_launcher

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        operation: str,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
    ) -> bool:
        """Run one asynchronous all-reduce within the configured deadline.

        :param tensor: Tensor reduced in place.
        :param operation: Stable operation label used in failure diagnostics.
        :param op: Reduction operation.
        :param group: Process group participating in the reduction.
        :returns: The result returned by the asynchronous work handle.
        :raises FailStopCollectiveTerminated: If an injected exit callback returns.
        :raises ValueError: If the operation label is empty.
        """
        if len(operation.strip()) == 0:
            raise ValueError("operation must not be empty")

        deadline = self._clock() + self._timeout_seconds
        attempt = _CollectiveAttempt()
        watchdog = threading.Thread(
            target=self._watchdog,
            args=(attempt, deadline, operation),
            name=f"fail-stop-collective:{operation}",
            daemon=True,
        )
        watchdog.start()
        attempt.wait_until_watchdog_started()

        if attempt.outcome() is _AttemptOutcome.FAILED:
            self._raise_after_injected_exit(attempt, watchdog)

        try:
            work = self._all_reduce_launcher(
                tensor,
                op=op,
                group=group,
                async_op=True,
            )
        except RuntimeError as error:
            self._fail_from_caller(
                attempt,
                watchdog,
                f"collective launch error: {error}",
            )

        while True:
            if attempt.outcome() is _AttemptOutcome.FAILED:
                self._raise_after_injected_exit(attempt, watchdog)

            if self._clock() >= deadline:
                self._fail_from_caller(
                    attempt,
                    watchdog,
                    f"deadline exceeded after {self._timeout_seconds:.3f}s",
                )

            try:
                completed = work.is_completed()
            except RuntimeError as error:
                self._fail_from_caller(
                    attempt,
                    watchdog,
                    f"collective completion error: {error}",
                )

            if completed:
                try:
                    result = work.wait()
                except RuntimeError as error:
                    self._fail_from_caller(
                        attempt,
                        watchdog,
                        f"collective result error: {error}",
                    )

                if attempt.try_succeed():
                    watchdog.join()
                    return result
                self._raise_after_injected_exit(attempt, watchdog)

            attempt.wait(self._poll_interval_seconds)

    def terminate(self, *, operation: str, reason: str) -> NoReturn:
        """Terminate the worker after a locally indeterminate transition.

        :param operation: Stable operation label used in failure diagnostics.
        :param reason: Concise diagnostic describing the failure.
        :raises FailStopCollectiveTerminated: If an injected exit callback returns.
        :raises ValueError: If either diagnostic field is empty.
        """
        if len(operation.strip()) == 0:
            raise ValueError("operation must not be empty")
        if len(reason.strip()) == 0:
            raise ValueError("reason must not be empty")
        self._emit_and_exit(operation, reason)
        raise FailStopCollectiveTerminated(reason)

    def _watchdog(
        self,
        attempt: _CollectiveAttempt,
        deadline: float,
        operation: str,
    ) -> None:
        """Terminate the process if the caller cannot enforce its deadline.

        :param attempt: Shared terminal-election state.
        :param deadline: Absolute monotonic deadline.
        :param operation: Stable operation label used in diagnostics.
        """
        attempt.mark_watchdog_started()
        while True:
            outcome = attempt.outcome()
            if outcome is _AttemptOutcome.SUCCEEDED:
                return
            if outcome is _AttemptOutcome.FAILED:
                self._emit_and_exit(operation, attempt.failure_reason())
                return

            remaining_seconds = deadline - self._clock()
            if remaining_seconds <= 0:
                reason = f"deadline exceeded after {self._timeout_seconds:.3f}s"
                attempt.fail(reason)
                continue
            attempt.wait(min(self._poll_interval_seconds, remaining_seconds))

    def _fail_from_caller(
        self,
        attempt: _CollectiveAttempt,
        watchdog: threading.Thread,
        reason: str,
    ) -> NoReturn:
        """Publish failure for watchdog-owned termination.

        :param attempt: Shared terminal-election state.
        :param watchdog: Watchdog guarding this attempt.
        :param reason: Concise diagnostic describing the failure.
        """
        attempt.fail(reason)
        self._raise_after_injected_exit(attempt, watchdog)

    def _emit_and_exit(self, operation: str, reason: str) -> None:
        """Emit one unbuffered diagnostic and terminate the worker.

        :param operation: Stable operation label used in diagnostics.
        :param reason: Concise diagnostic describing the failure.
        """
        compact_operation = " ".join(operation.split())
        compact_reason = " ".join(reason.split())
        diagnostic = (
            f"fail-stop collective '{compact_operation}' failed: {compact_reason}\n"
        )
        try:
            os.write(2, diagnostic.encode("utf-8", errors="replace"))
        except OSError:
            pass
        self._exit_callback(1)

    @staticmethod
    def _raise_after_injected_exit(
        attempt: _CollectiveAttempt,
        watchdog: threading.Thread,
    ) -> NoReturn:
        """Prevent continuation when a test exit callback returns.

        :param attempt: Shared terminal-election state.
        :param watchdog: Watchdog guarding this attempt.
        """
        watchdog.join()
        raise FailStopCollectiveTerminated(attempt.failure_reason())
