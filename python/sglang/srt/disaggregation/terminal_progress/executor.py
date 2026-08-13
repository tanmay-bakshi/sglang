import dataclasses
from collections.abc import Callable
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
EventT = TypeVar("EventT")


class DeterministicExecutionError(RuntimeError):
    """Deterministic state-machine execution invariant violation."""


class DeterministicReplayError(DeterministicExecutionError):
    """Replay diverged from its expected deterministic trace."""


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicEmission(Generic[EventT]):
    """One labeled event emitted into deterministic FIFO order.

    :ivar label: Stable reader-facing event identity.
    :ivar event: Immutable state-machine event payload.
    """

    label: str
    event: EventT

    def __post_init__(self) -> None:
        """Validate one deterministic event emission."""

        if type(self.label) is not str or len(self.label) == 0:
            raise ValueError("label must be a non-empty string")


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicTransition(Generic[StateT, EventT]):
    """Pure result of reducing one queued event.

    :ivar state: Complete immutable state after the event.
    :ivar emitted: Follow-up events appended after the existing queue.
    """

    state: StateT
    emitted: tuple[DeterministicEmission[EventT], ...] = ()

    def __post_init__(self) -> None:
        """Validate the immutable follow-up event collection."""

        if type(self.emitted) is not tuple:
            raise TypeError("emitted must be a tuple")
        for emission in self.emitted:
            if type(emission) is not DeterministicEmission:
                raise TypeError("emitted entries must be DeterministicEmission")


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicQueuedEvent(Generic[EventT]):
    """One event assigned a stable executor-local sequence.

    :ivar sequence: Monotonic FIFO sequence assigned at enqueue time.
    :ivar emission: Stable event label and immutable payload.
    """

    sequence: int
    emission: DeterministicEmission[EventT]

    def __post_init__(self) -> None:
        """Validate one sequenced event."""

        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if type(self.emission) is not DeterministicEmission:
            raise TypeError("emission must be DeterministicEmission")


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicTraceEntry(Generic[StateT, EventT]):
    """Complete replay evidence for one reduced event.

    :ivar queued_event: Exact event and FIFO sequence that ran.
    :ivar state_before: Complete state observed by the reducer.
    :ivar state_after: Complete state returned by the reducer.
    :ivar emitted_sequences: FIFO sequences assigned to follow-up events.
    """

    queued_event: DeterministicQueuedEvent[EventT]
    state_before: StateT
    state_after: StateT
    emitted_sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate one deterministic trace entry."""

        if type(self.queued_event) is not DeterministicQueuedEvent:
            raise TypeError("queued_event must be DeterministicQueuedEvent")
        if type(self.emitted_sequences) is not tuple:
            raise TypeError("emitted_sequences must be a tuple")
        for sequence in self.emitted_sequences:
            if type(sequence) is not int or sequence < 0:
                raise ValueError("emitted_sequences must contain non-negative integers")


DeterministicReducer = Callable[
    [StateT, EventT], DeterministicTransition[StateT, EventT]
]


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicExecutor(Generic[StateT, EventT]):
    """Immutable FIFO executor for pure lifecycle reducers.

    The executor deliberately models state-machine events rather than native
    notifications or runtime callbacks. A replay therefore depends only on
    the initial state, root emissions, and reducer.

    :ivar initial_state: Complete state before the first event.
    :ivar state: Complete state after all recorded trace entries.
    :ivar pending: Events awaiting reduction in FIFO order.
    :ivar trace: Stable ordered transition evidence.
    :ivar next_sequence: Sequence reserved for the next enqueue.
    """

    initial_state: StateT
    state: StateT
    pending: tuple[DeterministicQueuedEvent[EventT], ...]
    trace: tuple[DeterministicTraceEntry[StateT, EventT], ...]
    next_sequence: int

    def __post_init__(self) -> None:
        """Validate executor sequence and queue conservation."""

        if type(self.pending) is not tuple:
            raise TypeError("pending must be a tuple")
        if type(self.trace) is not tuple:
            raise TypeError("trace must be a tuple")
        if type(self.next_sequence) is not int or self.next_sequence < 0:
            raise ValueError("next_sequence must be a non-negative integer")

        pending_sequences: list[int] = []
        trace_sequences: list[int] = []
        known_sequences: set[int] = set()
        for queued_event in self.pending:
            if type(queued_event) is not DeterministicQueuedEvent:
                raise TypeError("pending entries must be DeterministicQueuedEvent")
            pending_sequences.append(queued_event.sequence)
            known_sequences.add(queued_event.sequence)
        for entry in self.trace:
            if type(entry) is not DeterministicTraceEntry:
                raise TypeError("trace entries must be DeterministicTraceEntry")
            trace_sequences.append(entry.queued_event.sequence)
            known_sequences.add(entry.queued_event.sequence)

        if len(known_sequences) != len(self.pending) + len(self.trace):
            raise ValueError("pending and trace sequences must be unique")
        if known_sequences != set(range(self.next_sequence)):
            raise ValueError("trace and queue must conserve every assigned sequence")
        if trace_sequences != sorted(trace_sequences):
            raise ValueError("trace entries must preserve FIFO sequence order")
        if pending_sequences != sorted(pending_sequences):
            raise ValueError("pending entries must preserve FIFO sequence order")
        if (
            len(trace_sequences) > 0
            and len(pending_sequences) > 0
            and trace_sequences[-1] >= pending_sequences[0]
        ):
            raise ValueError("processed sequences must precede pending sequences")

        if len(self.trace) == 0:
            if self.state != self.initial_state:
                raise ValueError("an empty trace cannot change executor state")
            return
        if self.trace[0].state_before != self.initial_state:
            raise ValueError("the first trace entry must observe initial_state")
        for previous, current in zip(self.trace, self.trace[1:], strict=False):
            if current.state_before != previous.state_after:
                raise ValueError("adjacent trace state snapshots must join exactly")
        if self.state != self.trace[-1].state_after:
            raise ValueError("executor state must equal the final trace state")

    @classmethod
    def create(
        cls,
        initial_state: StateT,
    ) -> "DeterministicExecutor[StateT, EventT]":
        """Create an empty executor for one initial state.

        :param initial_state: Complete immutable initial state.
        :returns: Empty deterministic executor.
        """

        return cls(
            initial_state=initial_state,
            state=initial_state,
            pending=(),
            trace=(),
            next_sequence=0,
        )

    def enqueue(
        self,
        emission: DeterministicEmission[EventT],
    ) -> "DeterministicExecutor[StateT, EventT]":
        """Append one event after every currently queued event.

        :param emission: Stable event label and immutable payload.
        :returns: New executor with the event queued.
        """

        if type(emission) is not DeterministicEmission:
            raise TypeError("emission must be DeterministicEmission")
        queued_event = DeterministicQueuedEvent(
            sequence=self.next_sequence,
            emission=emission,
        )
        return dataclasses.replace(
            self,
            pending=(*self.pending, queued_event),
            next_sequence=self.next_sequence + 1,
        )

    def enqueue_many(
        self,
        emissions: tuple[DeterministicEmission[EventT], ...],
    ) -> "DeterministicExecutor[StateT, EventT]":
        """Append root events in exact tuple order.

        :param emissions: Ordered immutable event collection.
        :returns: New executor with every event queued.
        """

        if type(emissions) is not tuple:
            raise TypeError("emissions must be a tuple")
        executor = self
        for emission in emissions:
            executor = executor.enqueue(emission)
        return executor

    def step(
        self,
        reducer: DeterministicReducer[StateT, EventT],
    ) -> "DeterministicExecutor[StateT, EventT]":
        """Reduce the next event and append its emitted work.

        :param reducer: Pure state and event reducer.
        :returns: New executor after exactly one transition.
        :raises DeterministicExecutionError: If no event is pending or the
            reducer returns an invalid result.
        """

        if len(self.pending) == 0:
            raise DeterministicExecutionError("cannot step an empty executor")

        queued_event = self.pending[0]
        transition = reducer(self.state, queued_event.emission.event)
        if type(transition) is not DeterministicTransition:
            raise DeterministicExecutionError(
                "reducer must return DeterministicTransition"
            )

        pending = self.pending[1:]
        next_sequence = self.next_sequence
        emitted_sequences: list[int] = []
        for emission in transition.emitted:
            pending += (
                DeterministicQueuedEvent(
                    sequence=next_sequence,
                    emission=emission,
                ),
            )
            emitted_sequences.append(next_sequence)
            next_sequence += 1

        trace_entry = DeterministicTraceEntry(
            queued_event=queued_event,
            state_before=self.state,
            state_after=transition.state,
            emitted_sequences=tuple(emitted_sequences),
        )
        return dataclasses.replace(
            self,
            state=transition.state,
            pending=pending,
            trace=(*self.trace, trace_entry),
            next_sequence=next_sequence,
        )

    def drain(
        self,
        reducer: DeterministicReducer[StateT, EventT],
        maximum_steps: int,
    ) -> "DeterministicExecutor[StateT, EventT]":
        """Reduce the queue to exhaustion within one explicit step bound.

        :param reducer: Pure state and event reducer.
        :param maximum_steps: Maximum transitions accepted for this replay.
        :returns: New executor with an empty queue.
        :raises DeterministicExecutionError: If emitted work exceeds the bound.
        """

        if type(maximum_steps) is not int or maximum_steps <= 0:
            raise ValueError("maximum_steps must be a positive integer")

        executor = self
        steps = 0
        while len(executor.pending) > 0:
            if steps >= maximum_steps:
                raise DeterministicExecutionError(
                    f"execution exceeded the {maximum_steps}-step bound"
                )
            executor = executor.step(reducer)
            steps += 1
        return executor


def replay_deterministic_trace(
    initial_state: StateT,
    root_emissions: tuple[DeterministicEmission[EventT], ...],
    reducer: DeterministicReducer[StateT, EventT],
    expected_trace: tuple[DeterministicTraceEntry[StateT, EventT], ...],
    maximum_steps: int,
) -> DeterministicExecutor[StateT, EventT]:
    """Replay roots and require byte-for-byte Python value equivalence.

    :param initial_state: Complete immutable initial state.
    :param root_emissions: Exact ordered root events.
    :param reducer: Pure state and event reducer.
    :param expected_trace: Previously recorded transition evidence.
    :param maximum_steps: Maximum transitions accepted for the replay.
    :returns: Fully drained replay executor.
    :raises DeterministicReplayError: If any replayed entry differs.
    """

    executor: DeterministicExecutor[StateT, EventT] = DeterministicExecutor.create(
        initial_state
    )
    replayed = executor.enqueue_many(root_emissions).drain(reducer, maximum_steps)
    if replayed.trace != expected_trace:
        raise DeterministicReplayError("deterministic trace replay diverged")
    return replayed
