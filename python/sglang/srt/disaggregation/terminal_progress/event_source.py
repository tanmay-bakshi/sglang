import collections
import errno
import os
import threading

from sglang.srt.disaggregation.terminal_progress.clock import (
    SystemTerminalOwnerClock,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerClosedError,
    TerminalOwnerCommandValue,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerOverflowError,
    TerminalOwnerPulse,
)


class TerminalOwnerQueueEventSource(TerminalOwnerEventSource):
    """Bounded in-process reference adapter with a coalesced fd wakeup.

    Queue insertion is authoritative and occurs before the wake byte is
    published. At most one wake byte is outstanding, so a full pipe can never
    make a successfully queued command invisible. Native adapters implement the
    same boundary without routing their producer hot loop through Python.
    """

    _name: str
    _capacity: int
    _read_fd: int
    _write_fd: int
    _pending: collections.deque[TerminalOwnerEventEnvelope]
    _next_sequence: int
    _wake_armed: bool
    _closed: bool
    _overflowed: bool
    _rejected_envelope: TerminalOwnerEventEnvelope | None
    _lock: threading.Lock
    _clock: SystemTerminalOwnerClock

    def __init__(self, name: str, capacity: int) -> None:
        """Create one bounded queue and its process-local wake pipe.

        :param name: Stable source identity used in owner evidence.
        :param capacity: Maximum immutable envelopes retained at once.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        self._name = name
        self._capacity = capacity
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._pending = collections.deque()
        self._next_sequence = 0
        self._wake_armed = False
        self._closed = False
        self._overflowed = False
        self._rejected_envelope = None
        self._lock = threading.Lock()
        self._clock = SystemTerminalOwnerClock()

    @property
    def name(self) -> str:
        """Return the stable event-source identity.

        :returns: Source name supplied at construction.
        """

        return self._name

    @property
    def overflowed(self) -> bool:
        """Return whether any producer crossed the bounded queue.

        :returns: Sticky overflow state.
        """

        with self._lock:
            return self._overflowed

    @property
    def pending_count(self) -> int:
        """Return the exact number of queued envelopes.

        :returns: Pending immutable envelope count.
        """

        with self._lock:
            return len(self._pending)

    def fileno(self) -> int:
        """Return the readable wake descriptor.

        :returns: Open read-side pipe descriptor.
        """

        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("event source is closed")
            return self._read_fd

    def publish(
        self,
        command: TerminalOwnerCommandValue,
        enqueued_ns: int | None = None,
    ) -> TerminalOwnerEventEnvelope:
        """Append one command and signal the fd after queue publication.

        :param command: Immutable command accepted by the terminal owner.
        :param enqueued_ns: Optional producer-local monotonic timestamp.
        :returns: Exact producer-sequenced envelope that was queued.
        :raises TerminalOwnerOverflowError: If the bounded queue is full.
        """

        timestamp_ns = self._clock.now_ns() if enqueued_ns is None else enqueued_ns
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("enqueued_ns must be a non-negative integer")
        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("event source is closed")
            if self._overflowed:
                raise TerminalOwnerOverflowError(
                    f"event source {self._name} previously overflowed",
                    source_name=self._name,
                    pending_envelopes=tuple(self._pending),
                    rejected_envelope=self._rejected_envelope,
                )
            if len(self._pending) >= self._capacity:
                rejected_envelope = TerminalOwnerEventEnvelope(
                    producer_sequence=self._next_sequence,
                    enqueued_ns=timestamp_ns,
                    command=command,
                )
                self._next_sequence += 1
                self._overflowed = True
                self._rejected_envelope = rejected_envelope
                self._signal_locked()
                raise TerminalOwnerOverflowError(
                    f"event source {self._name} exceeded capacity {self._capacity}",
                    source_name=self._name,
                    pending_envelopes=tuple(self._pending),
                    rejected_envelope=rejected_envelope,
                )
            envelope = TerminalOwnerEventEnvelope(
                producer_sequence=self._next_sequence,
                enqueued_ns=timestamp_ns,
                command=command,
            )
            self._next_sequence += 1
            self._pending.append(envelope)
            self._signal_locked()
            return envelope

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Drain the coalesced wake and every already-published envelope.

        :returns: Exact producer FIFO population observed at this wake.
        :raises TerminalOwnerOverflowError: If a producer crossed the bound.
        """

        self._drain_wake_fd()
        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("event source is closed")
            envelopes = tuple(self._pending)
            self._pending.clear()
            self._wake_armed = False
            overflowed = self._overflowed
            rejected_envelope = self._rejected_envelope
        if overflowed:
            raise TerminalOwnerOverflowError(
                f"event source {self._name} crossed its bounded capacity",
                source_name=self._name,
                pending_envelopes=envelopes,
                rejected_envelope=rejected_envelope,
            )
        return envelopes

    def close(self) -> None:
        """Close both pipe descriptors after the owner stops."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            read_fd = self._read_fd
            write_fd = self._write_fd
        os.close(read_fd)
        os.close(write_fd)

    def _signal_locked(self) -> None:
        """Publish one coalesced wake while the queue lock is held."""

        if self._wake_armed:
            return
        try:
            os.write(self._write_fd, b"\x01")
        except BlockingIOError as error:
            raise TerminalOwnerOverflowError(
                f"event source {self._name} wake pipe is unexpectedly full"
            ) from error
        self._wake_armed = True

    def _drain_wake_fd(self) -> None:
        """Consume all wake bytes without fixed-cadence polling."""

        while True:
            try:
                data = os.read(self._read_fd, 4096)
            except BlockingIOError:
                return
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                raise
            if len(data) == 0 or len(data) < 4096:
                return


class TerminalOwnerPulseEventSource(TerminalOwnerEventSource):
    """Coalesced fd wake used when no payload queue is required."""

    _name: str
    _read_fd: int
    _write_fd: int
    _armed: bool
    _closed: bool
    _next_sequence: int
    _lock: threading.Lock

    def __init__(self, name: str) -> None:
        """Create one coalesced pulse source.

        :param name: Stable source identity.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        self._name = name
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._armed = False
        self._closed = False
        self._next_sequence = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return this pulse source's stable identity.

        :returns: Source name supplied at construction.
        """

        return self._name

    @property
    def pending_count(self) -> int:
        """Return whether one coalesced pulse is pending.

        :returns: Zero or one.
        """

        with self._lock:
            return int(self._armed)

    def fileno(self) -> int:
        """Return the readable pulse descriptor.

        :returns: Open read-side pipe descriptor.
        """

        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("pulse source is closed")
            return self._read_fd

    def signal(self) -> None:
        """Publish one coalesced owner pulse."""

        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("pulse source is closed")
            if self._armed:
                return
            os.write(self._write_fd, b"\x01")
            self._armed = True

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Consume one pulse and return its gap-free sequence.

        :returns: Empty tuple or one immutable pulse envelope.
        """

        with self._lock:
            if self._closed:
                raise TerminalOwnerClosedError("pulse source is closed")
            if not self._armed:
                return ()
            try:
                os.read(self._read_fd, 1)
            except BlockingIOError:
                pass
            sequence = self._next_sequence
            self._next_sequence += 1
            self._armed = False
        return (
            TerminalOwnerEventEnvelope(
                producer_sequence=sequence,
                enqueued_ns=0,
                command=TerminalOwnerPulse(),
            ),
        )

    def close(self) -> None:
        """Close both pulse descriptors after owner exit."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            read_fd = self._read_fd
            write_fd = self._write_fd
        os.close(read_fd)
        os.close(write_fd)
