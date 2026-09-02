import queue
import threading
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed.fail_stop_collective import (
    AsyncCollectiveWork,
    FailStopCollective,
    FailStopCollectiveTerminated,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_WAIT_SECONDS = 5.0
_WATCHDOG_THREAD_PREFIX = "fail-stop-collective:"


class _ManualClock:
    """Thread-safe monotonic clock advanced explicitly by a test."""

    _lock: threading.Lock
    _value: float

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0.0

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        """Advance the clock.

        :param seconds: Positive monotonic-clock increment.
        """
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        with self._lock:
            self._value += seconds


class _ExitRecorder:
    """Record fail-stop requests without terminating the test process."""

    _called: threading.Event
    _codes: list[int]
    _hook: Callable[[], None] | None
    _lock: threading.Lock

    def __init__(self, hook: Callable[[], None] | None = None) -> None:
        self._called = threading.Event()
        self._codes = []
        self._hook = hook
        self._lock = threading.Lock()

    def __call__(self, code: int) -> None:
        with self._lock:
            self._codes.append(code)
        self._called.set()
        if self._hook is not None:
            self._hook()

    def wait(self) -> bool:
        """Wait for a fail-stop request.

        :returns: Whether the request arrived within the test bound.
        """
        return self._called.wait(_WAIT_SECONDS)

    def codes(self) -> list[int]:
        """Return a snapshot of recorded exit codes.

        :returns: Recorded exit codes in invocation order.
        """
        with self._lock:
            return list(self._codes)


class _Work:
    """Configurable asynchronous collective work handle."""

    _completed: bool
    _completion_error: RuntimeError | None
    _result: bool
    _result_error: RuntimeError | None
    completion_polls: int
    waits: int

    def __init__(
        self,
        *,
        completed: bool,
        result: bool = True,
        completion_error: RuntimeError | None = None,
        result_error: RuntimeError | None = None,
    ) -> None:
        self._completed = completed
        self._result = result
        self._completion_error = completion_error
        self._result_error = result_error
        self.completion_polls = 0
        self.waits = 0

    def is_completed(self) -> bool:
        self.completion_polls += 1
        if self._completion_error is not None:
            raise self._completion_error
        return self._completed

    def wait(self) -> bool:
        self.waits += 1
        if self._result_error is not None:
            raise self._result_error
        return self._result


class _AdvancingPendingWork:
    """Pending work that advances the injected clock on every poll."""

    _clock: _ManualClock
    _increment_seconds: float
    waits: int

    def __init__(self, clock: _ManualClock, increment_seconds: float) -> None:
        self._clock = clock
        self._increment_seconds = increment_seconds
        self.waits = 0

    def is_completed(self) -> bool:
        self._clock.advance(self._increment_seconds)
        return False

    def wait(self) -> bool:
        self.waits += 1
        return True


class _GatedCompletionWork:
    """Completed work whose status poll can be held across watchdog expiry."""

    _poll_entered: threading.Event
    _release_poll: threading.Event
    waits: int

    def __init__(self) -> None:
        self._poll_entered = threading.Event()
        self._release_poll = threading.Event()
        self.waits = 0

    def is_completed(self) -> bool:
        self._poll_entered.set()
        if not self._release_poll.wait(_WAIT_SECONDS):
            raise RuntimeError("test did not release completion poll")
        return True

    def wait(self) -> bool:
        self.waits += 1
        return True

    def wait_until_polled(self) -> bool:
        """Wait until the caller is inside the completion poll.

        :returns: Whether the caller entered the poll within the test bound.
        """
        return self._poll_entered.wait(_WAIT_SECONDS)

    def release(self) -> None:
        """Allow the blocked completion poll to return."""
        self._release_poll.set()


class _RecordingLauncher:
    """Return one work handle while recording the launch contract."""

    _work: AsyncCollectiveWork
    calls: list[tuple[torch.Tensor, dist.ReduceOp, dist.ProcessGroup | None, bool]]

    def __init__(self, work: AsyncCollectiveWork) -> None:
        self._work = work
        self.calls = []

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
        group: dist.ProcessGroup | None,
        async_op: bool,
    ) -> AsyncCollectiveWork:
        self.calls.append((tensor, op, group, async_op))
        return self._work


class _BlockingLauncher:
    """Hold collective launch until its independent watchdog requests exit."""

    _entered: threading.Event
    _release: threading.Event
    _work: AsyncCollectiveWork

    def __init__(self, work: AsyncCollectiveWork) -> None:
        self._entered = threading.Event()
        self._release = threading.Event()
        self._work = work

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
        group: dist.ProcessGroup | None,
        async_op: bool,
    ) -> AsyncCollectiveWork:
        self._entered.set()
        if not self._release.wait(_WAIT_SECONDS):
            raise RuntimeError("test did not release collective launch")
        return self._work

    def wait_until_entered(self) -> bool:
        """Wait until the caller enters collective launch.

        :returns: Whether launch was entered within the test bound.
        """
        return self._entered.wait(_WAIT_SECONDS)

    def release(self) -> None:
        """Allow collective launch to return."""
        self._release.set()


def _watchdog_threads() -> list[threading.Thread]:
    """Return live fail-stop collective watchdog threads.

    :returns: Live watchdog threads.
    """
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(_WATCHDOG_THREAD_PREFIX)
    ]


def _run_in_thread(
    collective: FailStopCollective,
    tensor: torch.Tensor,
    outcomes: queue.Queue[bool | FailStopCollectiveTerminated],
) -> None:
    """Run one collective and capture its terminal result.

    :param collective: Collective runner under test.
    :param tensor: Tensor passed to the launcher.
    :param outcomes: Queue receiving the result or injected-exit exception.
    """
    try:
        outcomes.put(collective.all_reduce(tensor, operation="test-consensus"))
    except FailStopCollectiveTerminated as error:
        outcomes.put(error)


def test_success_returns_work_result_and_cleans_up_watchdog() -> None:
    work = _Work(completed=True, result=False)
    launcher = _RecordingLauncher(work)
    exits = _ExitRecorder()
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=_ManualClock(),
        exit_callback=exits,
        all_reduce_launcher=launcher,
    )
    tensor = torch.tensor([3], dtype=torch.int64)

    result = collective.all_reduce(
        tensor,
        operation="preparation-vote",
        op=dist.ReduceOp.MIN,
    )

    assert result is False
    assert len(launcher.calls) == 1
    launched_tensor, launched_op, launched_group, launched_async = launcher.calls[0]
    assert launched_tensor is tensor
    assert launched_op is dist.ReduceOp.MIN
    assert launched_group is None
    assert launched_async is True
    assert work.completion_polls == 1
    assert work.waits == 1
    assert exits.codes() == []
    assert _watchdog_threads() == []


def test_timeout_emits_diagnostic_and_requests_one_exit(
    capfd: pytest.CaptureFixture[str],
) -> None:
    clock = _ManualClock()
    work = _AdvancingPendingWork(clock, increment_seconds=0.6)
    exits = _ExitRecorder()
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=clock,
        exit_callback=exits,
        all_reduce_launcher=_RecordingLauncher(work),
    )

    with pytest.raises(FailStopCollectiveTerminated, match="deadline exceeded"):
        collective.all_reduce(
            torch.tensor([1], dtype=torch.int64),
            operation="load-completion-consensus",
        )

    assert exits.codes() == [1]
    assert work.waits == 0
    assert "load-completion-consensus" in capfd.readouterr().err
    assert _watchdog_threads() == []


def test_collective_error_emits_diagnostic_and_requests_one_exit(
    capfd: pytest.CaptureFixture[str],
) -> None:
    work = _Work(
        completed=True,
        result_error=RuntimeError("transport unavailable"),
    )
    exits = _ExitRecorder()
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=_ManualClock(),
        exit_callback=exits,
        all_reduce_launcher=_RecordingLauncher(work),
    )

    with pytest.raises(FailStopCollectiveTerminated, match="transport unavailable"):
        collective.all_reduce(
            torch.tensor([1], dtype=torch.int64),
            operation="post-commit-fence",
        )

    assert exits.codes() == [1]
    assert work.waits == 1
    diagnostic = capfd.readouterr().err
    assert "post-commit-fence" in diagnostic
    assert "collective result error" in diagnostic
    assert _watchdog_threads() == []


def test_explicit_termination_emits_diagnostic_and_cannot_return(
    capfd: pytest.CaptureFixture[str],
) -> None:
    exits = _ExitRecorder()
    collective = FailStopCollective(
        timeout_seconds=1.0,
        exit_callback=exits,
    )

    with pytest.raises(FailStopCollectiveTerminated, match="indeterminate owner"):
        collective.terminate(
            operation="post-commit-fence",
            reason="indeterminate owner",
        )

    assert exits.codes() == [1]
    diagnostic = capfd.readouterr().err
    assert "post-commit-fence" in diagnostic
    assert "indeterminate owner" in diagnostic


def test_watchdog_bounds_a_collective_launch_that_does_not_return() -> None:
    clock = _ManualClock()
    launcher = _BlockingLauncher(_Work(completed=True))
    exits = _ExitRecorder(hook=launcher.release)
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=clock,
        exit_callback=exits,
        all_reduce_launcher=launcher,
    )
    outcomes: queue.Queue[bool | FailStopCollectiveTerminated] = queue.Queue()
    caller = threading.Thread(
        target=_run_in_thread,
        args=(collective, torch.tensor([1], dtype=torch.int64), outcomes),
    )
    caller.start()

    assert launcher.wait_until_entered()
    clock.advance(2.0)
    assert exits.wait()
    caller.join(_WAIT_SECONDS)

    assert not caller.is_alive()
    assert isinstance(outcomes.get_nowait(), FailStopCollectiveTerminated)
    assert exits.codes() == [1]
    assert _watchdog_threads() == []


def test_watchdog_wins_completion_race_once_and_cleans_up() -> None:
    clock = _ManualClock()
    work = _GatedCompletionWork()
    allow_exit_return = threading.Event()

    def hold_exit_callback() -> None:
        allow_exit_return.wait(_WAIT_SECONDS)

    exits = _ExitRecorder(hook=hold_exit_callback)
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=clock,
        exit_callback=exits,
        all_reduce_launcher=_RecordingLauncher(work),
    )
    outcomes: queue.Queue[bool | FailStopCollectiveTerminated] = queue.Queue()
    caller = threading.Thread(
        target=_run_in_thread,
        args=(collective, torch.tensor([1], dtype=torch.int64), outcomes),
    )
    caller.start()

    assert work.wait_until_polled()
    clock.advance(2.0)
    assert exits.wait()
    work.release()
    allow_exit_return.set()
    caller.join(_WAIT_SECONDS)

    assert not caller.is_alive()
    assert isinstance(outcomes.get_nowait(), FailStopCollectiveTerminated)
    assert work.waits == 1
    assert exits.codes() == [1]
    assert _watchdog_threads() == []


def test_repeated_success_leaves_no_watchdog_threads() -> None:
    exits = _ExitRecorder()
    collective = FailStopCollective(
        timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        clock=_ManualClock(),
        exit_callback=exits,
        all_reduce_launcher=_RecordingLauncher(_Work(completed=True)),
    )

    for index in range(20):
        assert collective.all_reduce(
            torch.tensor([index], dtype=torch.int64),
            operation="cleanup-check",
        )

    assert exits.codes() == []
    assert _watchdog_threads() == []
