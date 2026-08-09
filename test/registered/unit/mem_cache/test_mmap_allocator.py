from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import mmap
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    ShmHostTensorAllocator,
)
from sglang.srt.mem_cache.storage.mmap import (
    NumaPlacement,
    alloc_mmap,
    alloc_shm,
    mmap_allocator,
    sample_numa_placement,
)


class TestMmapAllocator(unittest.TestCase):
    def test_alloc_mmap(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor = alloc_mmap(dims, dtype)
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        # Verify it has mapped memory address
        self.assertGreater(tensor.data_ptr(), 0)

    def test_alloc_shm(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertGreater(tensor.data_ptr(), 0)
        self.assertGreaterEqual(fd, 0)
        self.assertIsInstance(mm, mmap.mmap)

        # Check that we can write to the tensor
        tensor[0, 0] = 42.0
        self.assertEqual(tensor[0, 0].item(), 42.0)

        # Check that the FD is open and valid
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            self.fail("FD is not valid or closed")

        # Cleanup
        mm.close()
        os.close(fd)

    def test_shm_host_tensor_allocator(self):
        allocator = ShmHostTensorAllocator()
        dims = (2, 512)
        dtype = torch.int32

        tensor = allocator.allocate(dims, dtype, "cpu")
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertIsNotNone(allocator.fd)
        self.assertGreaterEqual(allocator.fd, 0)

        # Write data and check
        tensor[1, 1] = 99
        self.assertEqual(tensor[1, 1].item(), 99)

        # Test destructor cleans up fd
        fd = allocator.fd
        # Trigger GC / deletion
        del allocator

        # Verify fd is closed
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_alloc_shm_unlinked(self):
        dims = (4, 256)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        # On Linux, the path of an unlinked fd shows up in /proc/self/fd/
        # with a ' (deleted)' suffix.
        fd_path = f"/proc/self/fd/{fd}"
        try:
            resolved_path = os.readlink(fd_path)
            self.assertIn("sglang_host_pool_", resolved_path)
            self.assertTrue(resolved_path.endswith(" (deleted)"))
        except OSError:
            # If procfs is not available or readlink fails, fallback to direct path existence check
            self.assertFalse(os.path.exists(f"/dev/shm/sglang_host_pool_"))

        # Cleanup
        mm.close()
        os.close(fd)

    def test_alloc_shm_hugepage_warning(self):
        from sglang.srt.environ import envs

        envs.SGLANG_HUGEPAGE_SIZE.override("2MB")
        try:
            # Should succeed by falling back to plain page size mapping
            dims = (2, 2)
            tensor, fd, mm = alloc_shm(dims, torch.float32)
            self.assertEqual(tensor.shape, dims)
            mm.close()
            os.close(fd)
        finally:
            envs.SGLANG_HUGEPAGE_SIZE.override(None)

    def test_shm_host_tensor_allocator_invalid_device(self):
        allocator = ShmHostTensorAllocator()
        with self.assertRaises(AssertionError) as ctx:
            allocator.allocate((2, 2), torch.float32, device="cuda")
        self.assertIn("only supports CPU allocations", str(ctx.exception))

    def test_sample_numa_placement_reports_node_distribution(self):
        fake_libnuma = MagicMock()

        def move_pages(_pid, count, _pages, _nodes, statuses, _flags):
            for index in range(count.value):
                statuses[index] = 1 if index < count.value - 1 else 0
            return 0

        fake_libnuma.move_pages.side_effect = move_pages
        with (
            patch.object(mmap_allocator, "_libnuma", fake_libnuma),
            patch.object(mmap_allocator.os, "sysconf", return_value=4096),
        ):
            placement = sample_numa_placement(
                address=0x100000,
                n_bytes=4096 * 4,
                requested_node=1,
                max_sample_pages=4,
            )

        self.assertEqual(placement.pages_by_node, ((0, 1), (1, 3)))
        self.assertFalse(placement.is_local)

    def test_mbind_uses_a_complete_machine_word_mask(self):
        fake_libnuma = MagicMock()
        fake_libnuma.mbind.return_value = 0
        with patch.object(mmap_allocator, "_libnuma", fake_libnuma):
            bound = mmap_allocator._bind_memory_to_numa_node(
                address=0x100000,
                n_bytes=4096,
                numa_node=1,
            )

        self.assertTrue(bound)
        args = fake_libnuma.mbind.call_args.args
        self.assertEqual(args[3][0], 0b10)
        self.assertEqual(args[4].value, 64)

    def test_host_allocator_propagates_and_records_rank_local_node(self):
        buffer = torch.empty((16,), dtype=torch.float32)
        placement = NumaPlacement(
            requested_node=1,
            sampled_pages=1,
            pages_by_node=((1, 1),),
            errors_by_code=(),
            query_error=None,
        )
        with (
            envs.SGLANG_LOCAL_NUMA_NODE.override(1),
            patch(
                "sglang.srt.mem_cache.pool_host.common.alloc_mmap",
                return_value=buffer,
            ) as mock_alloc_mmap,
            patch(
                "sglang.srt.mem_cache.pool_host.common.sample_numa_placement",
                return_value=placement,
            ) as mock_sample,
        ):
            allocator = HostTensorAllocator()
            actual = allocator.allocate((16,), torch.float32, "cpu")

        self.assertIs(actual, buffer)
        self.assertEqual(allocator.placements, [placement])
        mock_alloc_mmap.assert_called_once_with(
            (16,),
            torch.float32,
            numa_node=1,
        )
        mock_sample.assert_called_once_with(
            address=buffer.data_ptr(),
            n_bytes=buffer.numel() * buffer.element_size(),
            requested_node=1,
        )

    def test_strict_host_allocator_rejects_remote_pages(self):
        buffer = torch.empty((16,), dtype=torch.float32)
        placement = NumaPlacement(
            requested_node=1,
            sampled_pages=1,
            pages_by_node=((0, 1),),
            errors_by_code=(),
            query_error=None,
        )
        with (
            envs.SGLANG_LOCAL_NUMA_NODE.override(1),
            envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.override(True),
            patch(
                "sglang.srt.mem_cache.pool_host.common.alloc_mmap",
                return_value=buffer,
            ),
            patch(
                "sglang.srt.mem_cache.pool_host.common.sample_numa_placement",
                return_value=placement,
            ),
        ):
            allocator = HostTensorAllocator()
            with self.assertRaisesRegex(RuntimeError, "requested_node=1"):
                allocator.allocate((16,), torch.float32, "cpu")


if __name__ == "__main__":
    unittest.main()
