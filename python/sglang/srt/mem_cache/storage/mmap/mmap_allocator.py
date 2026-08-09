import ctypes
import ctypes.util
import errno
import logging
import math
import mmap
import os
import uuid
import weakref
from dataclasses import dataclass

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
try:
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.madvise.restype = ctypes.c_int
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
except OSError:
    _libc = None

try:
    _libnuma_name = ctypes.util.find_library("numa") or "libnuma.so.1"
    _libnuma = ctypes.CDLL(_libnuma_name, use_errno=True)
    _libnuma.mbind.restype = ctypes.c_long
    _libnuma.mbind.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    _libnuma.move_pages.restype = ctypes.c_long
    _libnuma.move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
except OSError:
    _libnuma = None

# MAP_POPULATE is in Python's mmap module only since 3.11.
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x08000)
# MAP_HUGETLB and MAP_HUGE_* are Linux-specific and not in Python's mmap module.
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26  # 0x1400000
_MAP_HUGE_1GB = 30 << 26  # 0x78000000
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_MPOL_BIND = 2
_NUMA_SAMPLE_PAGES = 64


@dataclass(frozen=True)
class NumaPlacement:
    """Observed physical placement of a populated host allocation.

    :ivar requested_node: NUMA node requested for the allocation.
    :ivar sampled_pages: Number of pages queried with ``move_pages``.
    :ivar pages_by_node: Sorted ``(node, page_count)`` pairs.
    :ivar errors_by_code: Sorted ``(errno, page_count)`` query failures.
    :ivar query_error: Allocation-wide query failure, if placement could not be
        inspected.
    """

    requested_node: int
    sampled_pages: int
    pages_by_node: tuple[tuple[int, int], ...]
    errors_by_code: tuple[tuple[int, int], ...]
    query_error: str | None

    @property
    def is_local(self) -> bool:
        """Return whether every successfully queried page is on the target node.

        :returns: ``True`` when all observed pages are local and no page query
            failed.
        """

        return (
            self.query_error is None
            and len(self.errors_by_code) == 0
            and self.pages_by_node == ((self.requested_node, self.sampled_pages),)
        )


def _handle_numa_allocation_failure(reason: str) -> None:
    """Apply the configured NUMA failure policy.

    :param reason: Actionable placement failure description.
    :raises RuntimeError: If strict NUMA binding is enabled.
    """

    logger.warning(reason)
    if envs.SGLANG_CRASH_ON_NUMA_BIND_FAILURE.get():
        raise RuntimeError(reason)


def _bind_memory_to_numa_node(address: int, n_bytes: int, numa_node: int) -> bool:
    """Attach an ``MPOL_BIND`` policy to an unpopulated virtual-memory range.

    :param address: Page-aligned mapping address.
    :param n_bytes: Mapped byte length.
    :param numa_node: Physical NUMA node receiving future page faults.
    :returns: Whether the range policy was installed.
    """

    if numa_node < 0:
        raise ValueError(f"NUMA node must be non-negative, got {numa_node}.")
    if _libnuma is None:
        _handle_numa_allocation_failure(
            "Cannot bind HiCache host allocation to NUMA node "
            f"{numa_node}: libnuma is unavailable."
        )
        return False

    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    word_count = numa_node // bits_per_word + 1
    node_mask = (ctypes.c_ulong * word_count)()
    node_mask[numa_node // bits_per_word] = 1 << (numa_node % bits_per_word)
    result = _libnuma.mbind(
        ctypes.c_void_p(address),
        ctypes.c_ulong(n_bytes),
        ctypes.c_int(_MPOL_BIND),
        node_mask,
        ctypes.c_ulong(word_count * bits_per_word),
        ctypes.c_uint(0),
    )
    if result == 0:
        return True

    error_number = ctypes.get_errno()
    _handle_numa_allocation_failure(
        "Failed to bind HiCache host allocation to NUMA node "
        f"{numa_node}: [errno {error_number}] {os.strerror(error_number)}."
    )
    return False


def _populate_mapping(address: int, n_bytes: int) -> None:
    """Fault writable pages after the range NUMA policy is installed.

    :param address: Mapping address.
    :param n_bytes: Mapped byte length.
    """

    if _libc is not None:
        result = _libc.madvise(
            ctypes.c_void_p(address),
            ctypes.c_size_t(n_bytes),
            ctypes.c_int(_MADV_POPULATE_WRITE),
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        unsupported_errors = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
        if error_number not in unsupported_errors:
            raise OSError(error_number, os.strerror(error_number))

    ctypes.memset(ctypes.c_void_p(address), 0, n_bytes)


def _prepare_numa_mapping(address: int, n_bytes: int, numa_node: int) -> None:
    """Bind and populate a new host mapping in the required order.

    :param address: Mapping address.
    :param n_bytes: Mapped byte length.
    :param numa_node: Physical NUMA node for page placement.
    """

    _bind_memory_to_numa_node(address, n_bytes, numa_node)
    _populate_mapping(address, n_bytes)


def sample_numa_placement(
    address: int,
    n_bytes: int,
    requested_node: int,
    max_sample_pages: int = _NUMA_SAMPLE_PAGES,
) -> NumaPlacement:
    """Query a bounded, evenly spaced sample of an allocation's physical pages.

    :param address: Allocation address.
    :param n_bytes: Logical allocation byte length.
    :param requested_node: NUMA node used when creating the mapping.
    :param max_sample_pages: Maximum number of pages to inspect.
    :returns: NUMA placement observation.
    """

    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = max(1, math.ceil(n_bytes / page_size))
    sample_count = min(total_pages, max_sample_pages)
    if sample_count == 1:
        page_indices = (0,)
    else:
        page_indices = tuple(
            index * (total_pages - 1) // (sample_count - 1)
            for index in range(sample_count)
        )

    if _libnuma is None:
        return NumaPlacement(
            requested_node=requested_node,
            sampled_pages=sample_count,
            pages_by_node=(),
            errors_by_code=(),
            query_error="libnuma is unavailable",
        )

    pages = (ctypes.c_void_p * sample_count)(
        *(address + page_index * page_size for page_index in page_indices)
    )
    statuses = (ctypes.c_int * sample_count)()
    result = _libnuma.move_pages(
        ctypes.c_int(0),
        ctypes.c_ulong(sample_count),
        pages,
        None,
        statuses,
        ctypes.c_int(0),
    )
    if result < 0:
        error_number = ctypes.get_errno()
        return NumaPlacement(
            requested_node=requested_node,
            sampled_pages=sample_count,
            pages_by_node=(),
            errors_by_code=(),
            query_error=(
                f"move_pages failed: [errno {error_number}] "
                f"{os.strerror(error_number)}"
            ),
        )

    pages_by_node: dict[int, int] = {}
    errors_by_code: dict[int, int] = {}
    for status in statuses:
        if status >= 0:
            pages_by_node[status] = pages_by_node.get(status, 0) + 1
            continue
        error_number = -status
        errors_by_code[error_number] = errors_by_code.get(error_number, 0) + 1

    return NumaPlacement(
        requested_node=requested_node,
        sampled_pages=sample_count,
        pages_by_node=tuple(sorted(pages_by_node.items())),
        errors_by_code=tuple(sorted(errors_by_code.items())),
        query_error=None,
    )


def _alloc_hugepage(
    n_bytes: int,
    alloc_bytes: int,
    extra_flags: int,
    numa_node: int | None,
) -> ctypes.Array:
    """Call mmap via libc with hugepage flags and return an owning ctypes array.

    munmap fires automatically via weakref.finalize when the array is
    garbage-collected (i.e. when the tensor that wraps it is freed).

    :param n_bytes: Logical tensor byte length.
    :param alloc_bytes: Page-aligned mapping byte length.
    :param extra_flags: Linux huge-page mmap flags.
    :param numa_node: Optional target NUMA node.
    :returns: Owning ctypes byte array.
    """
    populate_flag = _MAP_POPULATE if numa_node is None else 0
    ptr = _libc.mmap(
        None,
        alloc_bytes,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | populate_flag | extra_flags,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if numa_node is not None:
        _prepare_numa_mapping(ptr, alloc_bytes, numa_node)
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def alloc_mmap(
    dims: tuple[int, ...],
    dtype: torch.dtype,
    numa_node: int | None = None,
) -> torch.Tensor:
    """Allocate a populated host tensor through anonymous mmap.

    Shared mappings and eager page population are both required so
    ``cudaHostRegister`` pins real, pre-faulted physical pages. The NUMA path
    installs its range policy before population; the default path uses
    ``MAP_POPULATE`` directly.

    The tensor owns the mapping; munmap fires when the tensor is freed.

    Set ``SGLANG_HUGEPAGE_SIZE=2MB`` or ``1GB`` for explicit huge pages.

    :param dims: Tensor dimensions.
    :param dtype: Tensor element type.
    :param numa_node: Optional target NUMA node. Binding precedes page faults.
    :returns: Allocated host tensor.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    if hugepage_size == "":
        page_size, extra_flags = mmap.PAGESIZE, 0
    elif hugepage_size == "2MB":
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_size == "1GB":
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB
    else:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Falling back to plain page-size mmap.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        page_size, extra_flags = mmap.PAGESIZE, 0

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            logger.error(
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                hugepage_size,
            )
        else:
            try:
                array = _alloc_hugepage(
                    n_bytes,
                    alloc_bytes,
                    extra_flags,
                    numa_node,
                )
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
            except OSError as e:
                logger.error(
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. "
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                    e,
                    hugepage_size,
                )
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    populate_flag = _MAP_POPULATE if numa_node is None else 0
    mm = mmap.mmap(
        -1,
        alloc_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | populate_flag,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    if numa_node is not None:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        _prepare_numa_mapping(address, alloc_bytes, numa_node)
    else:
        try:
            # MADV_POPULATE_WRITE guarantees pages are populated and writable,
            # throwing an error on failure (e.g. out of memory).
            mm.madvise(_MADV_POPULATE_WRITE)
        except OSError:
            # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
            pass
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)


def alloc_shm(
    dims: tuple[int, ...],
    dtype: torch.dtype,
    numa_node: int | None = None,
) -> tuple[torch.Tensor, int, mmap.mmap]:
    """Allocate a host tensor via shared memory (/dev/shm).

    :param dims: Tensor dimensions.
    :param dtype: Tensor element type.
    :param numa_node: Optional target NUMA node. Binding precedes page faults.
    :returns: Tensor, open file descriptor, and owning mmap. The caller must
        retain and close the descriptor when sharing is no longer required.
    """
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    # Note: hugepages are not directly supported with /dev/shm mmap files
    # without mounting hugetlbfs there, so we fall back to plain page size.
    if hugepage_size != "":
        logger.warning(
            "Hugepages are not supported with SHM allocator. "
            "Falling back to plain page-size mmap."
        )

    page_size = mmap.PAGESIZE
    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    # Create an anonymous shared memory file descriptor via memfd_create
    fd = None
    try:
        # MFD_CLOEXEC is standard on Linux 3.17+
        fd = os.memfd_create(
            f"sglang_host_pool_{uuid.uuid4().hex}",
            flags=getattr(os, "MFD_CLOEXEC", 1),
        )
    except (AttributeError, OSError):
        # Fallback to creating a file in /dev/shm if memfd_create is not supported
        shm_path = f"/dev/shm/sglang_host_pool_{uuid.uuid4().hex}.mmap"
        try:
            fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.unlink(shm_path)
            except OSError:
                pass
        except Exception as e:
            raise OSError(f"Failed to create shm file: {e}")

    try:
        os.ftruncate(fd, alloc_bytes)
        populate_flag = _MAP_POPULATE if numa_node is None else 0
        mm = mmap.mmap(
            fd,
            alloc_bytes,
            flags=mmap.MAP_SHARED | populate_flag,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        if numa_node is not None:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            _prepare_numa_mapping(address, alloc_bytes, numa_node)
        else:
            try:
                # MADV_POPULATE_WRITE guarantees pages are populated and writable,
                # throwing an error on failure (e.g. out of memory).
                mm.madvise(_MADV_POPULATE_WRITE)
            except OSError:
                # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
                pass
    except Exception as e:
        if fd is not None:
            os.close(fd)
        raise e

    tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return tensor, fd, mm
