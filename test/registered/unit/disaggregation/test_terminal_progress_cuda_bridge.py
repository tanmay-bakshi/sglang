import gc
import select

import pytest
import torch

from sglang.srt.disaggregation.terminal_progress.cuda_bridge import (
    CudaCompletionBridge,
    CudaCompletionFatalCode,
    CudaCompletionIdentity,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA host callbacks require one CUDA device",
)


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


def test_immediate_callback_is_take_once_and_closes_without_inventory() -> None:
    """An empty-stream callback delivers exactly once and retires cleanly."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(1)
    stream = torch.cuda.Stream()
    bridge.arm(identity)
    bridge.submit(stream.cuda_stream, identity)
    stream.synchronize()

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


def test_deferred_callback_blocks_join_and_active_close_fails_closed() -> None:
    """Shutdown cannot outrun one callback queued behind GPU work."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
    bridge.arm(identity)
    bridge.submit(stream.cuda_stream, identity)

    assert not bridge.join_producers()
    with pytest.raises(RuntimeError, match="close_with_active_callbacks"):
        bridge.close()
    failed = bridge.inventory()
    assert failed.active_callback_count == 1
    assert failed.fatal_code is CudaCompletionFatalCode.CLOSE_WITH_ACTIVE_CALLBACKS

    stream.synchronize()
    _assert_readable(bridge.fileno())
    drained = bridge.drain()
    assert drained.identities == (identity,)
    assert bridge.join_producers()
    settled = bridge.inventory()
    assert settled.active_callback_count == 0
    assert settled.live_count == 0
    assert settled.fatal_code is CudaCompletionFatalCode.CLOSE_WITH_ACTIVE_CALLBACKS


def test_callback_keeps_native_state_alive_after_failed_close() -> None:
    """Destroying a failed wrapper cannot race a deferred callback into UAF."""

    bridge = CudaCompletionBridge(capacity=4, testing=True)
    identity = _identity(3)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
    bridge.arm(identity)
    bridge.submit(stream.cuda_stream, identity)

    with pytest.raises(RuntimeError, match="close_with_active_callbacks"):
        bridge.close()
    del bridge
    gc.collect()
    stream.synchronize()


def test_eventfd_coalesces_multiple_cuda_callback_producers() -> None:
    """Concurrent callback producers share one lossless eventfd wake."""

    bridge = CudaCompletionBridge(capacity=16, testing=True)
    streams = [torch.cuda.Stream() for _ in range(8)]
    identities = tuple(_identity(index + 10) for index in range(len(streams)))
    for stream, identity in zip(streams, identities, strict=True):
        bridge.arm(identity)
        bridge.submit(stream.cuda_stream, identity)
    for stream in streams:
        stream.synchronize()

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
