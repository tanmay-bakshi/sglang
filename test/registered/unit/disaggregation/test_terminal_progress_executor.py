import dataclasses
import enum

import pytest
from sglang.srt.disaggregation.terminal_progress.executor import (
    DeterministicEmission,
    DeterministicExecutionError,
    DeterministicExecutor,
    DeterministicReplayError,
    DeterministicTraceEntry,
    DeterministicTransition,
    replay_deterministic_trace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Operation(enum.StrEnum):
    """Counter operation used by the pure executor tests."""

    ADD_ONE = "add_one"
    SPAWN = "spawn"
    ADD_TEN = "add_ten"
    ADD_TWENTY = "add_twenty"


@dataclasses.dataclass(frozen=True, slots=True)
class _CounterState:
    """Immutable counter state used by the pure executor tests."""

    value: int


def _reduce_counter(
    state: _CounterState,
    operation: _Operation,
) -> DeterministicTransition[_CounterState, _Operation]:
    """Reduce one counter operation and optionally emit follow-up work.

    :param state: Current immutable counter state.
    :param operation: Next deterministic operation.
    :returns: Complete transition and ordered follow-up work.
    """

    if operation is _Operation.SPAWN:
        return DeterministicTransition(
            state=state,
            emitted=(
                DeterministicEmission(label="spawn.ten", event=_Operation.ADD_TEN),
                DeterministicEmission(
                    label="spawn.twenty",
                    event=_Operation.ADD_TWENTY,
                ),
            ),
        )
    increments = {
        _Operation.ADD_ONE: 1,
        _Operation.ADD_TEN: 10,
        _Operation.ADD_TWENTY: 20,
    }
    return DeterministicTransition(
        state=_CounterState(value=state.value + increments[operation])
    )


def _root_emissions() -> tuple[DeterministicEmission[_Operation], ...]:
    """Return the fixed root event sequence.

    :returns: Spawn followed by one already-queued increment.
    """

    return (
        DeterministicEmission(label="root.spawn", event=_Operation.SPAWN),
        DeterministicEmission(label="root.one", event=_Operation.ADD_ONE),
    )


def test_fifo_execution_and_emission_order_are_stable() -> None:
    """Follow-up events append after work that was already queued."""

    executor: DeterministicExecutor[_CounterState, _Operation] = (
        DeterministicExecutor.create(_CounterState(value=0))
    )
    drained = executor.enqueue_many(_root_emissions()).drain(
        _reduce_counter,
        maximum_steps=4,
    )

    assert drained.state == _CounterState(value=31)
    assert len(drained.pending) == 0
    assert tuple(entry.queued_event.sequence for entry in drained.trace) == (0, 1, 2, 3)
    assert tuple(entry.queued_event.emission.label for entry in drained.trace) == (
        "root.spawn",
        "root.one",
        "spawn.ten",
        "spawn.twenty",
    )
    assert drained.trace[0].emitted_sequences == (2, 3)


def test_trace_replays_exactly_and_detects_divergence() -> None:
    """The same roots and reducer recreate every trace value exactly."""

    initial_state = _CounterState(value=4)
    executor: DeterministicExecutor[_CounterState, _Operation] = (
        DeterministicExecutor.create(initial_state)
    )
    original = executor.enqueue_many(_root_emissions()).drain(
        _reduce_counter,
        maximum_steps=4,
    )

    replayed = replay_deterministic_trace(
        initial_state=initial_state,
        root_emissions=_root_emissions(),
        reducer=_reduce_counter,
        expected_trace=original.trace,
        maximum_steps=4,
    )
    assert replayed == original

    wrong_entry: DeterministicTraceEntry[_CounterState, _Operation] = (
        dataclasses.replace(
            original.trace[-1],
            state_after=_CounterState(value=999),
        )
    )
    wrong_trace = (*original.trace[:-1], wrong_entry)
    with pytest.raises(DeterministicReplayError):
        replay_deterministic_trace(
            initial_state=initial_state,
            root_emissions=_root_emissions(),
            reducer=_reduce_counter,
            expected_trace=wrong_trace,
            maximum_steps=4,
        )


def test_executor_is_immutable_and_bounded() -> None:
    """Queue mutation returns new values and emitted loops cannot run forever."""

    executor: DeterministicExecutor[_CounterState, _Operation] = (
        DeterministicExecutor.create(_CounterState(value=0))
    )
    queued = executor.enqueue(
        DeterministicEmission(label="root.spawn", event=_Operation.SPAWN)
    )

    assert len(executor.pending) == 0
    assert len(queued.pending) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        queued.next_sequence = 99  # type: ignore[misc]
    with pytest.raises(DeterministicExecutionError):
        queued.drain(_reduce_counter, maximum_steps=2)
    with pytest.raises(DeterministicExecutionError):
        executor.step(_reduce_counter)


def test_executor_rejects_non_transition_reducer_results() -> None:
    """An impure or malformed callback cannot enter trace evidence."""

    executor: DeterministicExecutor[_CounterState, _Operation] = (
        DeterministicExecutor.create(_CounterState(value=0))
    )
    queued = executor.enqueue(
        DeterministicEmission(label="root.one", event=_Operation.ADD_ONE)
    )

    def invalid_reducer(state: _CounterState, operation: _Operation) -> object:
        """Return a malformed reducer value.

        :param state: Ignored state.
        :param operation: Ignored operation.
        :returns: Invalid transition sentinel.
        """

        del state, operation
        return object()

    with pytest.raises(DeterministicExecutionError):
        queued.step(invalid_reducer)  # type: ignore[arg-type]


def test_executor_rejects_forged_or_nonconserved_trace_state() -> None:
    """Trace evidence cannot skip a sequence or break a state join."""

    initial = _CounterState(value=0)
    emission = DeterministicEmission(label="root.one", event=_Operation.ADD_ONE)
    executor: DeterministicExecutor[_CounterState, _Operation] = (
        DeterministicExecutor.create(initial)
    )
    traced = executor.enqueue(emission).drain(_reduce_counter, maximum_steps=1)

    with pytest.raises(ValueError, match="conserve"):
        dataclasses.replace(traced, next_sequence=2)
    forged_entry = dataclasses.replace(
        traced.trace[0],
        state_before=_CounterState(value=9),
    )
    with pytest.raises(ValueError, match="initial_state"):
        dataclasses.replace(traced, trace=(forged_entry,))
