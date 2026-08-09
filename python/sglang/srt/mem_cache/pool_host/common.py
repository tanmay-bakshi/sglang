import json
import logging
import mmap
import os
from collections import defaultdict

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import (
    NumaPlacement,
    alloc_mmap,
    alloc_shm,
    sample_numa_placement,
)

logger = logging.getLogger(__name__)


class HostTensorAllocator:
    """Allocate page-populated host tensors for hierarchical KV cache."""

    dtype: torch.dtype | None
    dims: tuple[int, ...] | None
    numa_node: int | None
    placements: list[NumaPlacement]

    def __init__(self) -> None:
        """Initialize a rank-local host allocator."""

        self.dtype = None
        self.dims = None
        self.numa_node = envs.SGLANG_LOCAL_NUMA_NODE.get()
        self.placements = []

    def allocate(
        self,
        dims: tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        """Allocate and verify one host tensor.

        :param dims: Tensor dimensions.
        :param dtype: Tensor element type.
        :param device: Host device, which must be ``cpu``.
        :returns: Allocated host tensor.
        """

        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        buffer = alloc_mmap(dims, dtype, numa_node=self.numa_node)
        self._record_numa_placement(buffer)
        return buffer

    def _record_numa_placement(self, buffer: torch.Tensor) -> None:
        """Record bounded placement evidence for one populated allocation.

        :param buffer: Newly allocated host tensor.
        :raises RuntimeError: If strict NUMA validation is enabled and sampled
            pages are not local.
        """

        if self.numa_node is None:
            return

        n_bytes = buffer.numel() * buffer.element_size()
        placement = sample_numa_placement(
            address=buffer.data_ptr(),
            n_bytes=n_bytes,
            requested_node=self.numa_node,
        )
        self.placements.append(placement)
        identity = (
            f"logical_gpu_id={envs.SGLANG_LOCAL_GPU_ID.get()}, "
            "physical_gpu_id="
            f"{envs.SGLANG_LOCAL_NVML_DEVICE_INDEX.get()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        )
        if placement.query_error is not None:
            message = (
                "Could not verify NUMA placement for "
                f"{n_bytes / 1e9:.3f} GB HiCache host allocation on requested "
                f"node {self.numa_node} ({identity}): {placement.query_error}"
            )
            logger.warning(message)
            if envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.get():
                raise RuntimeError(message)
            return

        message = (
            f"HiCache host allocation NUMA placement: bytes={n_bytes}, "
            f"requested_node={self.numa_node}, "
            f"sampled_pages={placement.sampled_pages}, "
            f"pages_by_node={dict(placement.pages_by_node)}, "
            f"errors_by_code={dict(placement.errors_by_code)}, {identity}"
        )
        if placement.is_local:
            logger.info(message)
            return

        logger.warning(message)
        if envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.get():
            raise RuntimeError(message)


class ShmHostTensorAllocator(HostTensorAllocator):
    """Allocate NUMA-local host tensors backed by shareable memory mappings."""

    fds: list[int]
    mms: list[mmap.mmap]

    def __init__(self) -> None:
        """Initialize a rank-local shared-memory allocator."""

        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(
        self,
        dims: tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        """Allocate and verify one shared-memory host tensor.

        :param dims: Tensor dimensions.
        :param dtype: Tensor element type.
        :param device: Host device, which must be ``cpu``.
        :returns: Allocated host tensor.
        """

        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        tensor, fd, mm = alloc_shm(dims, dtype, numa_node=self.numa_node)
        self.fds.append(fd)
        self.mms.append(mm)
        self._record_numa_placement(tensor)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
