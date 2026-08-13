import abc
import threading
import time


class TerminalOwnerClock(abc.ABC):
    """Explicit same-process monotonic clock used by owner evidence."""

    @abc.abstractmethod
    def now_ns(self) -> int:
        """Return the current monotonic timestamp.

        :returns: Non-negative process-local monotonic nanoseconds.
        """


class SystemTerminalOwnerClock(TerminalOwnerClock):
    """Production clock using ``CLOCK_MONOTONIC_RAW`` when available."""

    def now_ns(self) -> int:
        """Return the current owner timestamp.

        :returns: Monotonic raw nanoseconds on Linux.
        """

        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class ManualTerminalOwnerClock(TerminalOwnerClock):
    """Thread-safe deterministic clock for owner qualification tests."""

    _current_ns: int
    _lock: threading.Lock

    def __init__(self, initial_ns: int = 0) -> None:
        """Create one manually advanced monotonic clock.

        :param initial_ns: Initial non-negative timestamp.
        """

        if type(initial_ns) is not int or initial_ns < 0:
            raise ValueError("initial_ns must be a non-negative integer")
        self._current_ns = initial_ns
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        """Return the current deterministic timestamp.

        :returns: Current manually controlled nanoseconds.
        """

        with self._lock:
            return self._current_ns

    def advance_ns(self, delta_ns: int) -> int:
        """Advance monotonically and return the new timestamp.

        :param delta_ns: Non-negative nanoseconds to add.
        :returns: New deterministic timestamp.
        """

        if type(delta_ns) is not int or delta_ns < 0:
            raise ValueError("delta_ns must be a non-negative integer")
        with self._lock:
            self._current_ns += delta_ns
            return self._current_ns
