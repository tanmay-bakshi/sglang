"""CPU-only tests for cancellable HiCache controller allocations."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _entry(
    *,
    alloc: MagicMock,
    free: MagicMock,
) -> SimpleNamespace:
    """Build one auxiliary device-allocation entry.

    :param alloc: Component allocation function.
    :param free: Component release function.
    :returns: Minimal host-pool entry fixture.
    """

    return SimpleNamespace(
        device_pool=SimpleNamespace(alloc=alloc, free=free),
        device_alloc_fn=alloc,
        device_free_fn=free,
        device_evict_fn=None,
    )


def _controller(
    *,
    full_alloc: MagicMock,
    full_free: MagicMock,
    entries: dict[PoolName, SimpleNamespace],
) -> HybridCacheController:
    """Build a controller without streams or payload storage.

    :param full_alloc: Full-attention allocation function.
    :param full_free: Full-attention release function.
    :param entries: Auxiliary pool entries by name.
    :returns: Controller fixture for allocation-only methods.
    """

    controller = object.__new__(HybridCacheController)
    controller.device = "cpu"
    controller.mem_pool_device_allocator = SimpleNamespace(
        full_attn_allocator=SimpleNamespace(alloc=full_alloc, free=full_free)
    )
    controller.mem_pool_host = SimpleNamespace(entry_map=entries)
    controller.load_queue = []
    return controller


class TestPreparedLoadOwnership(unittest.TestCase):
    """Validate allocation rollback before a generation starts."""

    def test_partial_component_allocation_failure_rolls_back_every_pool(self) -> None:
        full_indices = torch.tensor([10, 11], dtype=torch.int64)
        swa_indices = torch.tensor([20], dtype=torch.int64)
        full_alloc = MagicMock(return_value=full_indices)
        full_free = MagicMock()
        swa_alloc = MagicMock(return_value=swa_indices)
        swa_free = MagicMock()
        mamba_alloc = MagicMock(return_value=None)
        mamba_free = MagicMock()
        controller = _controller(
            full_alloc=full_alloc,
            full_free=full_free,
            entries={
                PoolName.SWA: _entry(alloc=swa_alloc, free=swa_free),
                PoolName.MAMBA: _entry(alloc=mamba_alloc, free=mamba_free),
            },
        )
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2], dtype=torch.int64),
        )

        prepared = controller.prepare_load(
            torch.tensor([3, 4], dtype=torch.int64),
            extra_pools=[swa_transfer, mamba_transfer],
        )

        self.assertIsNone(prepared)
        full_free.assert_called_once_with(full_indices)
        swa_free.assert_called_once_with(swa_indices)
        mamba_free.assert_not_called()
        self.assertIsNone(swa_transfer.device_indices)
        self.assertIsNone(mamba_transfer.device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_cancel_prepared_load_releases_real_and_derived_allocations_once(
        self,
    ) -> None:
        full_indices = torch.tensor([10, 11], dtype=torch.int64)
        swa_indices = torch.tensor([20], dtype=torch.int64)
        full_alloc = MagicMock(return_value=full_indices)
        full_free = MagicMock()
        swa_alloc = MagicMock(return_value=swa_indices)
        swa_free = MagicMock()
        controller = _controller(
            full_alloc=full_alloc,
            full_free=full_free,
            entries={PoolName.SWA: _entry(alloc=swa_alloc, free=swa_free)},
        )
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )
        derived_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.tensor([1], dtype=torch.int64),
            indices_from_pool=PoolName.SWA,
        )

        prepared = controller.prepare_load(
            torch.tensor([3, 4], dtype=torch.int64),
            extra_pools=[swa_transfer, derived_transfer],
        )
        assert prepared is not None
        controller.enqueue_prepared_load(prepared)
        controller.cancel_prepared_load(prepared)

        self.assertTrue(prepared.cancelled)
        self.assertEqual(controller.load_queue, [])
        full_free.assert_called_once_with(full_indices)
        swa_free.assert_called_once_with(swa_indices)
        self.assertIsNone(swa_transfer.device_indices)
        self.assertIsNone(derived_transfer.device_indices)
        with self.assertRaisesRegex(ValueError, "already cancelled"):
            controller.cancel_prepared_load(prepared)

    def test_allocator_exception_rolls_back_completed_component_allocations(
        self,
    ) -> None:
        full_indices = torch.tensor([10, 11], dtype=torch.int64)
        swa_indices = torch.tensor([20], dtype=torch.int64)
        full_alloc = MagicMock(return_value=full_indices)
        full_free = MagicMock()
        swa_alloc = MagicMock(return_value=swa_indices)
        swa_free = MagicMock()
        mamba_alloc = MagicMock(side_effect=RuntimeError("injected failure"))
        mamba_free = MagicMock()
        controller = _controller(
            full_alloc=full_alloc,
            full_free=full_free,
            entries={
                PoolName.SWA: _entry(alloc=swa_alloc, free=swa_free),
                PoolName.MAMBA: _entry(alloc=mamba_alloc, free=mamba_free),
            },
        )
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2], dtype=torch.int64),
        )

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            controller.prepare_load(
                torch.tensor([3, 4], dtype=torch.int64),
                extra_pools=[swa_transfer, mamba_transfer],
            )

        full_free.assert_called_once_with(full_indices)
        swa_free.assert_called_once_with(swa_indices)
        mamba_free.assert_not_called()
        self.assertIsNone(swa_transfer.device_indices)
        self.assertIsNone(mamba_transfer.device_indices)


if __name__ == "__main__":
    unittest.main()
