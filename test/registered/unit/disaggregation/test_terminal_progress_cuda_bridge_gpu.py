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


def _identity(cookie: int) -> CudaCompletionIdentity:
    """Build one deterministic exact-generation identity.

    :param cookie: Process-local owner cookie.
    :returns: Valid callback identity.
    """

    return CudaCompletionIdentity(cookie=cookie, generation=bytes((cookie,)) * 16)


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


def test_empty_stream_callback_is_take_once_and_retires_all_inventory() -> None:
    """An immediately eligible CUDA callback delivers one exact token."""

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
    assert first.inventory.active_callback_count == 0
    assert first.inventory.active_registration_count == 0
    assert first.inventory.queued_count == 0
    assert first.inventory.retained_count == 0

    _close_healthy_bridge(bridge)
    closed = bridge.inventory()
    assert closed.closed
    assert closed.producers_joined
    assert not closed.eventfd_open
    assert closed.retained_count == 0
    assert closed.fatal_code is CudaCompletionFatalCode.NONE


def test_deferred_callback_makes_early_close_process_fatal_without_uaf() -> None:
    """A callback behind device work retains state and prevents early close."""

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
    assert failed.live_count == 1
    assert failed.fatal_code is CudaCompletionFatalCode.CLOSE_WITH_ACTIVE_CALLBACKS

    stream.synchronize()
    _assert_readable(bridge.fileno())
    drained = bridge.drain()
    assert drained.identities == (identity,)
    assert bridge.join_producers()
    assert drained.inventory.active_callback_count == 0
    assert drained.inventory.retained_count == 0

    del bridge
    gc.collect()


def test_cuda_callbacks_from_multiple_streams_share_one_mpsc_queue() -> None:
    """Independent CUDA callback threads publish without lost or forged tokens."""

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
    assert drained.inventory.total_submissions == len(identities)
    assert drained.inventory.total_enqueued == len(identities)
    assert drained.inventory.total_drained == len(identities)
    assert drained.inventory.successful_wake_count == len(identities)
    assert drained.inventory.consumed_wake_count == len(identities)
    assert drained.inventory.active_callback_count == 0
    assert drained.inventory.active_registration_count == 0
    assert drained.inventory.retained_count == 0
    _close_healthy_bridge(bridge)
