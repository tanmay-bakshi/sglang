"""Focused tests for HiCache prepared-operation ownership."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    OperationLifecycleError,
    OperationOwnerTable,
    OperationRegistrationError,
    OperationState,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _IndexPool:
    """Deterministic CPU index allocator with exact release accounting."""

    _next_index: int
    _allocation_calls: int
    _fail_on_calls: set[int]
    free_calls: list[tuple[int, ...]]

    layout = "page_first"
    page_size = 1
    device = "cpu"
    size = 4096
    logical_size = 4096
    size_per_token = 4
    can_use_write_back_jit = False

    def __init__(self, *, start: int, fail_on_calls: set[int] | None = None) -> None:
        self._next_index = start
        self._allocation_calls = 0
        self._fail_on_calls = set() if fail_on_calls is None else fail_on_calls
        self.free_calls = []

    def alloc(self, size: int) -> torch.Tensor | None:
        """Allocate a deterministic contiguous CPU range.

        :param size: Number of indices requested.
        :returns: Allocated indices, or ``None`` for an injected failure.
        """

        call_index = self._allocation_calls
        self._allocation_calls += 1
        if call_index in self._fail_on_calls:
            return None
        indices = torch.arange(self._next_index, self._next_index + size)
        self._next_index += size
        return indices

    def free(self, indices: torch.Tensor) -> int:
        """Record one exact release.

        :param indices: Indices being released.
        :returns: Number of released indices.
        """

        self.free_calls.append(tuple(int(index) for index in indices.tolist()))
        return int(indices.numel())


class _StartEvent:
    """Minimal event used by the batch-submission test."""

    record_calls: int

    def __init__(self) -> None:
        self.record_calls = 0

    def record(self) -> None:
        """Record one producer-start marker."""

        self.record_calls += 1


def _pool_entry(
    name: PoolName,
    host_pool: _IndexPool,
    device_pool: _IndexPool,
    *,
    is_anchor: bool = False,
) -> PoolEntry:
    """Build a host-group entry backed by deterministic allocators.

    :param name: Logical pool name.
    :param host_pool: Host allocator.
    :param device_pool: Device allocator.
    :param is_anchor: Whether this is the Full-KV anchor.
    :returns: Configured pool entry.
    """

    return PoolEntry(
        name=name,
        host_pool=host_pool,
        device_pool=device_pool,
        layer_mapper=lambda layer_id: layer_id,
        is_primary_index_anchor=is_anchor,
        device_alloc_fn=device_pool.alloc,
        device_free_fn=device_pool.free,
    )


def _hybrid_controller(
    *,
    full_device: _IndexPool | None = None,
    side_devices: dict[PoolName, _IndexPool] | None = None,
) -> tuple[HybridCacheController, _IndexPool, dict[PoolName, _IndexPool]]:
    """Build a bare CPU controller containing the requested logical pools.

    :param full_device: Optional anchor device allocator.
    :param side_devices: Optional side-pool device allocators.
    :returns: Controller, anchor allocator, and all side allocators.
    """

    anchor_device = _IndexPool(start=1000) if full_device is None else full_device
    devices = {} if side_devices is None else side_devices
    anchor_host = _IndexPool(start=100)
    entries = [
        _pool_entry(
            PoolName.KV,
            anchor_host,
            anchor_device,
            is_anchor=True,
        )
    ]
    for ordinal, (name, device_pool) in enumerate(devices.items()):
        entries.append(
            _pool_entry(
                name,
                _IndexPool(start=200 + 100 * ordinal),
                device_pool,
            )
        )

    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = HostPoolGroup(entries)
    controller.mem_pool_device_allocator = anchor_device
    controller._anchor_device_allocator = anchor_device
    controller.mem_pool_device = anchor_device
    controller.device = "cpu"
    controller.enable_storage = False
    controller.storage_stop_event = threading.Event()
    controller.ack_load_queue = []
    controller.ack_write_queue = []
    controller.l2_transfer_engine = mock.Mock()
    controller.load_fence_stream = None
    controller.layer_num = 1
    controller._initialize_operation_lifecycle()
    return controller, anchor_device, devices


class TestHiCachePreparedOperations(unittest.TestCase):
    def test_preparation_has_no_worker_or_queue_visibility(self) -> None:
        controller, _, _ = _hybrid_controller()

        load = controller.prepare_host_to_device(
            torch.tensor([7, 8]), domain_owner_id=41
        )
        write = controller.prepare_device_to_host(
            torch.tensor([11, 12]), domain_owner_id=42
        )

        self.assertIsNotNone(load)
        self.assertIsNotNone(write)
        assert load is not None
        assert write is not None
        self.assertEqual(load.state, OperationState.PREPARED)
        self.assertEqual(write.state, OperationState.PREPARED)
        self.assertEqual(controller.load_queue, [])
        self.assertEqual(controller.ack_load_queue, [])
        self.assertEqual(controller.ack_write_queue, [])
        controller.l2_transfer_engine.submit_host_to_device.assert_not_called()
        controller.l2_transfer_engine.submit_device_to_host.assert_not_called()

        controller.cancel_transfer(load)
        controller.cancel_transfer(write)

    def test_absent_stale_and_mismatched_registrations_are_rejected(self) -> None:
        controller, _, _ = _hybrid_controller()
        first = controller.prepare_host_to_device(torch.tensor([1]), domain_owner_id=7)
        second = controller.prepare_host_to_device(torch.tensor([2]), domain_owner_id=7)
        assert first is not None
        assert second is not None
        owners = OperationOwnerTable[object]()
        first_registration = owners.register(first, object())
        second_registration = owners.register(second, object())

        with self.assertRaises(OperationRegistrationError):
            controller.enqueue_host_to_device(first, None)
        with self.assertRaises(OperationRegistrationError):
            controller.enqueue_host_to_device(first, second_registration)

        owners.unregister(first_registration)
        with self.assertRaises(OperationRegistrationError):
            controller.enqueue_host_to_device(first, first_registration)

    def test_same_domain_owner_gets_distinct_operation_ids(self) -> None:
        controller, _, _ = _hybrid_controller()
        first = controller.prepare_host_to_device(torch.tensor([1]), domain_owner_id=19)
        second = controller.prepare_host_to_device(
            torch.tensor([2]), domain_owner_id=19
        )
        assert first is not None
        assert second is not None

        self.assertEqual(first.domain_owner_id, second.domain_owner_id)
        self.assertNotEqual(first.operation_id, second.operation_id)
        self.assertNotEqual(first.operation_id, first.domain_owner_id)

        controller.cancel_transfer(first)
        controller.cancel_transfer(second)

    def test_queued_cancellation_releases_each_independent_pool_once(self) -> None:
        side_devices = {
            PoolName.SWA: _IndexPool(start=2000),
            PoolName.MAMBA: _IndexPool(start=3000),
            PoolName.INDEXER: _IndexPool(start=4000),
            PoolName.DRAFT_SWA: _IndexPool(start=5000),
        }
        controller, full_device, devices = _hybrid_controller(side_devices=side_devices)
        transfers = [
            PoolTransfer(name=PoolName.SWA, host_indices=torch.tensor([20, 21])),
            PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([30])),
            PoolTransfer(name=PoolName.INDEXER, host_indices=torch.tensor([40, 41])),
            PoolTransfer(
                name=PoolName.DRAFT_SWA,
                indices_from_pool=PoolName.SWA,
            ),
        ]
        operation = controller.prepare_host_to_device(
            torch.tensor([10, 11]),
            domain_owner_id=5,
            extra_pools=transfers,
        )
        assert operation is not None
        owners = OperationOwnerTable[object]()
        registration = owners.register(operation, object())
        controller.enqueue_host_to_device(operation, registration)

        controller.cancel_transfer(operation, registration)
        controller.cancel_transfer(operation, registration)

        self.assertEqual(operation.state, OperationState.CANCELLED)
        self.assertTrue(operation.allocations.release_complete)
        self.assertEqual(len(full_device.free_calls), 1)
        self.assertEqual(len(devices[PoolName.SWA].free_calls), 1)
        self.assertEqual(len(devices[PoolName.MAMBA].free_calls), 1)
        self.assertEqual(len(devices[PoolName.INDEXER].free_calls), 1)
        self.assertEqual(len(devices[PoolName.DRAFT_SWA].free_calls), 0)
        self.assertEqual(controller.load_queue, [])
        self.assertEqual(len(owners), 0)

    def test_partial_side_pool_allocation_rolls_back_every_holding(self) -> None:
        swa_device = _IndexPool(start=2000)
        mamba_device = _IndexPool(start=3000, fail_on_calls={0})
        controller, full_device, _ = _hybrid_controller(
            side_devices={
                PoolName.SWA: swa_device,
                PoolName.MAMBA: mamba_device,
            }
        )
        transfers = [
            PoolTransfer(name=PoolName.SWA, host_indices=torch.tensor([20, 21])),
            PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([30])),
        ]

        operation = controller.prepare_host_to_device(
            torch.tensor([10, 11]), extra_pools=transfers
        )

        self.assertIsNone(operation)
        self.assertEqual(len(full_device.free_calls), 1)
        self.assertEqual(len(swa_device.free_calls), 1)
        self.assertEqual(len(mamba_device.free_calls), 0)
        self.assertEqual(controller._operations, {})
        self.assertTrue(all(transfer.device_indices is None for transfer in transfers))

    def test_hybrid_resolution_never_mutates_input_descriptors(self) -> None:
        controller, _, _ = _hybrid_controller(
            side_devices={
                PoolName.SWA: _IndexPool(start=2000),
                PoolName.DRAFT_SWA: _IndexPool(start=3000),
            }
        )
        keys = ["a", "b"]
        source = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([20, 21]),
            keys=keys,
        )
        derived = PoolTransfer(
            name=PoolName.DRAFT_SWA,
            keys=["draft"],
            indices_from_pool=PoolName.SWA,
        )

        operation = controller.prepare_host_to_device(
            torch.tensor([10, 11]), extra_pools=[source, derived]
        )
        assert operation is not None
        assert operation.pool_transfers is not None

        self.assertIsNone(source.device_indices)
        self.assertIsNone(derived.host_indices)
        self.assertIsNone(derived.device_indices)
        self.assertIsNot(operation.pool_transfers[0], source)
        self.assertIsNot(operation.pool_transfers[1], derived)
        self.assertIsNot(operation.pool_transfers[0].keys, keys)
        self.assertIs(
            operation.pool_transfers[1].device_indices,
            operation.pool_transfers[0].device_indices,
        )

        controller.cancel_transfer(operation)

    def test_physical_ack_uses_operation_ids_and_persistent_progress(self) -> None:
        controller, _, _ = _hybrid_controller()
        first = controller.prepare_host_to_device(torch.tensor([1]), domain_owner_id=23)
        second = controller.prepare_host_to_device(
            torch.tensor([2]), domain_owner_id=23
        )
        assert first is not None
        assert second is not None
        owners = OperationOwnerTable[object]()
        first_registration = owners.register(first, object())
        second_registration = owners.register(second, object())
        controller.enqueue_host_to_device(first, first_registration)
        controller.enqueue_host_to_device(second, second_registration)

        start_event = _StartEvent()
        controller.layer_done_counter = SimpleNamespace(
            update_producer=mock.Mock(return_value=0),
            events=[SimpleNamespace(start_event=start_event, complete=mock.Mock())],
        )
        controller._move_op_indices = mock.Mock(
            side_effect=lambda operation: (
                operation.host_indices,
                operation.device_indices,
                operation.pool_transfers,
            )
        )
        controller._l2_load_transfers = mock.Mock(return_value=[])
        controller.l2_transfer_engine.submit_host_to_device.return_value = (
            SimpleNamespace(
                start_event=object(), finish_event=object(), timing_enabled=False
            )
        )

        self.assertEqual(controller.submit_host_to_device_batch(), 0)
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.operation_ids, (first.operation_id, second.operation_id))
        self.assertEqual(ack.commit_progress.next_operation_index, 0)
        self.assertFalse(ack.commit_progress.metrics_emitted)

        controller.acknowledge_transfer(ack, first_registration)
        self.assertEqual(ack.commit_progress.next_operation_index, 1)
        controller.acknowledge_transfer(ack, second_registration)
        self.assertEqual(ack.commit_progress.next_operation_index, 2)
        self.assertTrue(HiCacheController.mark_ack_metrics_emitted(ack))
        self.assertFalse(HiCacheController.mark_ack_metrics_emitted(ack))
        self.assertEqual(len(owners), 0)

    def test_reset_refuses_live_operations(self) -> None:
        controller, _, _ = _hybrid_controller()
        stop_event = mock.Mock(spec=threading.Event)
        controller.storage_stop_event = stop_event
        operation = controller.prepare_host_to_device(torch.tensor([1]))
        assert operation is not None

        with self.assertRaises(OperationLifecycleError):
            controller.reset()
        stop_event.set.assert_not_called()

        controller.cancel_transfer(operation)
        controller.reset()
        stop_event.set.assert_called_once_with()
        stop_event.clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
