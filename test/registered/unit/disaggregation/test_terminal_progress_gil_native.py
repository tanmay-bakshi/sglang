import select
import sys

import pytest
from sglang.srt.disaggregation.terminal_progress.gil_qualification import (
    GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
    GIL_QUALIFICATION_OWNER_HOP_COUNT,
    correlate_gil_native_traces,
)
from sglang.srt.disaggregation.terminal_progress.gil_qualification_native import (
    GILNativeFatalCode,
    NativeGILQualificationEventSource,
)
from sglang.srt.disaggregation.terminal_progress.owner import (
    PackedTerminalProgressOwner,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerDisposition,
    TerminalOwnerEventSourceRegistration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="native qualification uses Linux eventfd",
)


def test_native_drain_retains_delivery_until_post_dispatch_ack() -> None:
    """Drain cannot produce a successor before committed-dispatch authority."""

    source = NativeGILQualificationEventSource(
        name="native-gil-lifetime",
        machine_count=2,
        hop_count=2,
        capacity=4,
        testing=True,
    )
    try:
        source.start(minimum_duration_seconds=0.000_001, minimum_transition_count=4)
        readable, _, _ = select.select((source.fileno(),), (), (), 1.0)
        assert readable == [source.fileno()]
        first = source.drain()
        assert len(first) == 2
        delivered = source.inventory()
        assert delivered.queued_count == 0
        assert delivered.delivered_unacknowledged_count == 2
        assert delivered.pending_count == 2

        source.acknowledge_dispatch(first[0])
        one_acknowledged = source.inventory()
        assert one_acknowledged.delivered_unacknowledged_count == 1
        assert one_acknowledged.queued_count == 1
        assert one_acknowledged.pending_count == 2

        source.acknowledge_dispatch(first[1])
        while not source.inventory().complete:
            readable, _, _ = select.select((source.fileno(),), (), (), 1.0)
            assert readable == [source.fileno()]
            for envelope in source.drain():
                source.acknowledge_dispatch(envelope)
        assert source.wait_until_complete(timeout_seconds=1.0)
        complete = source.inventory()
        assert complete.pending_count == 0
        assert complete.retired_count == 2
        assert complete.trace_count == complete.transition_count
        source.close()
        assert source.inventory().closed
    finally:
        source.abort_and_close()


def test_native_source_traverses_the_integrated_owner_reactor() -> None:
    """Every native hop is acknowledged only after a real owner transition."""

    source = NativeGILQualificationEventSource(
        name="native-gil-owner-integration",
        machine_count=GIL_QUALIFICATION_LIVE_MACHINE_COUNT,
        hop_count=GIL_QUALIFICATION_OWNER_HOP_COUNT,
        capacity=64,
    )
    owner = PackedTerminalProgressOwner(
        submission_capacity=16,
        output_capacity=16,
        event_sources=(
            TerminalOwnerEventSourceRegistration(
                source=source,
                close_on_shutdown=False,
                dispatch_observer=source,
            ),
        ),
    )
    owner.start()
    try:
        owner.wait_for_snapshot(
            lambda snapshot: snapshot.disposition is TerminalOwnerDisposition.RUNNING,
            timeout_seconds=2.0,
        )
        source.start(minimum_duration_seconds=0.01, minimum_transition_count=112)
        assert source.wait_until_complete(timeout_seconds=5.0)
        inventory = source.inventory()
        traces = source.traces()
        samples = correlate_gil_native_traces(traces)
        snapshot = owner.snapshot()
        assert snapshot.owner_transition_count == inventory.transition_count
        assert inventory.pending_count == 0
        assert len(samples) * GIL_QUALIFICATION_OWNER_HOP_COUNT == (
            inventory.transition_count
        )

        owner.begin_shutdown()
        owner.retire_shutdown_producers()
        stopped = owner.wait_for_snapshot(
            lambda current: current.disposition is TerminalOwnerDisposition.STOPPED,
            timeout_seconds=2.0,
        )
        assert stopped.owner_transition_count == inventory.transition_count + 2
        assert owner.join(timeout_seconds=2.0)
        source.close()
    finally:
        if owner.snapshot().reactor_alive:
            owner.join(timeout_seconds=2.0)
        source.abort_and_close()


def test_native_overflow_and_eventfd_failure_are_sticky() -> None:
    """Physical capacity and descriptor failures retain attributable evidence."""

    overflow = NativeGILQualificationEventSource(
        name="native-gil-overflow",
        machine_count=16,
        hop_count=7,
        capacity=2,
        testing=True,
    )
    try:
        with pytest.raises(RuntimeError, match="queue_overflow"):
            overflow.start(
                minimum_duration_seconds=1.0,
                minimum_transition_count=112,
            )
        inventory = overflow.inventory()
        assert inventory.fatal_code is GILNativeFatalCode.QUEUE_OVERFLOW
        assert inventory.rejected_record is not None
        assert inventory.pending_count == 3
    finally:
        overflow.abort_and_close()

    broken = NativeGILQualificationEventSource(
        name="native-gil-eventfd-failure",
        machine_count=16,
        hop_count=7,
        capacity=64,
        testing=True,
    )
    try:
        broken.break_eventfd_for_test()
        with pytest.raises(RuntimeError, match="eventfd_write_failure"):
            broken.start(
                minimum_duration_seconds=1.0,
                minimum_transition_count=112,
            )
        inventory = broken.inventory()
        assert inventory.fatal_code is GILNativeFatalCode.EVENTFD_WRITE_FAILURE
        assert not inventory.eventfd_open
    finally:
        broken.abort_and_close()
