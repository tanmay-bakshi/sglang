import gc
import select

import pytest

from sglang.srt.disaggregation.terminal_progress.cuda_bridge import (
    CudaCompletionBridge,
    CudaCompletionFatalCode,
    CudaCompletionIdentity,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _identity(cookie: int, generation_byte: int | None = None) -> CudaCompletionIdentity:
    """Build one deterministic exact-generation identity.

    :param cookie: Process-local owner cookie.
    :param generation_byte: Repeated generation byte, defaulting to the cookie.
    :returns: Valid callback identity.
    """

    byte = cookie if generation_byte is None else generation_byte
    return CudaCompletionIdentity(cookie=cookie, generation=bytes((byte,)) * 16)


def _assert_readable(fd: int) -> None:
    """Assert that one completion descriptor is currently readable.

    :param fd: Native eventfd.
    """

    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(2_000) == [(fd, select.POLLIN)]


def _close_healthy_bridge(bridge: CudaCompletionBridge) -> None:
    """Join and close one fully drained bridge.

    :param bridge: Healthy bridge with no retained identity.
    """

    bridge.stop_submissions()
    assert bridge.join_producers()
    bridge.close()


def test_deterministic_completion_is_take_once_and_closes_without_inventory() -> None:
    """One native publication delivers exactly once and retires cleanly."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(1)
    bridge.arm(identity)
    bridge.complete_synchronously_for_test(identity)

    _assert_readable(bridge.fileno())
    first = bridge.drain()
    second = bridge.drain()
    assert first.identities == (identity,)
    assert first.wake_count == 1
    assert second.identities == ()
    assert second.wake_count == 0
    assert first.inventory.live_count == 0
    assert first.inventory.active_callback_count == 0
    assert first.inventory.queued_count == 0

    _close_healthy_bridge(bridge)
    closed = bridge.inventory()
    assert closed.closed
    assert not closed.eventfd_open
    assert closed.retained_count == 0
    assert closed.fatal_code is CudaCompletionFatalCode.NONE


def test_close_with_armed_identity_fails_closed() -> None:
    """Shutdown cannot erase one armed exact identity."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(2)
    bridge.arm(identity)

    assert bridge.join_producers()
    with pytest.raises(RuntimeError, match="close_with_retained_inventory"):
        bridge.close()
    failed = bridge.inventory()
    assert failed.armed_count == 1
    assert failed.active_callback_count == 0
    assert (
        failed.fatal_code is CudaCompletionFatalCode.CLOSE_WITH_RETAINED_INVENTORY
    )


def test_wrapper_destruction_with_armed_identity_preserves_memory_safety() -> None:
    """Destroying a failed wrapper releases native state without a UAF."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(3)
    bridge.arm(identity)

    with pytest.raises(RuntimeError, match="close_with_retained_inventory"):
        bridge.close()
    del bridge
    gc.collect()


def test_eventfd_coalesces_multiple_native_producers() -> None:
    """Multiple native producer publications share one lossless eventfd wake."""

    bridge = CudaCompletionBridge(capacity=16, testing=True)
    identities = tuple(_identity(index + 10) for index in range(8))
    for identity in identities:
        bridge.arm(identity)
        bridge.complete_synchronously_for_test(identity)

    _assert_readable(bridge.fileno())
    drained = bridge.drain()
    assert set(drained.identities) == set(identities)
    assert drained.wake_count == len(identities)
    assert drained.inventory.successful_wake_count == len(identities)
    assert drained.inventory.consumed_wake_count == len(identities)
    _close_healthy_bridge(bridge)


def test_overflow_is_sticky_and_preserves_unresolved_inventory() -> None:
    """A full queue never drops a completion into apparent success."""

    bridge = CudaCompletionBridge(capacity=2, testing=True)
    identities = tuple(_identity(index + 30) for index in range(3))
    for identity in identities:
        bridge.arm(identity)
        bridge.complete_synchronously_for_test(identity)

    inventory = bridge.inventory()
    assert inventory.fatal_code is CudaCompletionFatalCode.QUEUE_OVERFLOW
    assert inventory.overflow_count == 1
    assert inventory.queued_count == 2
    assert inventory.live_count == 3

    drained = bridge.drain()
    assert drained.identities == identities[:2]
    assert drained.inventory.live_count == 1
    assert drained.inventory.queued_count == 0
    assert drained.inventory.fatal_identity == identities[2]


def test_exact_generation_mismatch_fails_before_cuda_registration() -> None:
    """An armed cookie cannot be submitted under a stale generation."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    armed = _identity(40, generation_byte=1)
    stale = _identity(40, generation_byte=2)
    bridge.arm(armed)

    with pytest.raises(RuntimeError, match="exact_generation_mismatch"):
        bridge.submit(0, stale)
    inventory = bridge.inventory()
    assert inventory.total_submissions == 0
    assert inventory.active_callback_count == 0
    assert inventory.live_count == 1
    assert inventory.fatal_code is CudaCompletionFatalCode.EXACT_GENERATION_MISMATCH
    assert inventory.fatal_identity == stale


def test_eventfd_failure_is_sticky_while_token_remains_drainable() -> None:
    """Descriptor failure poisons the process without erasing authority."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(50)
    bridge.arm(identity)
    bridge.break_eventfd_for_test()
    bridge.complete_synchronously_for_test(identity)

    inventory = bridge.inventory()
    assert inventory.fatal_code is CudaCompletionFatalCode.EVENTFD_WRITE_FAILURE
    assert inventory.eventfd_failure_count == 1
    drained = bridge.drain()
    assert drained.identities == (identity,)
    assert drained.inventory.live_count == 0
    assert drained.inventory.eventfd_failure_count == 2
