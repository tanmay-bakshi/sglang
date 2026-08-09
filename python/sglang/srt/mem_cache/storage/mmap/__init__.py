# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Mmap allocator storage backend helpers for SGLang HiCache."""

from .mmap_allocator import (
    NumaPlacement,
    alloc_mmap,
    alloc_shm,
    sample_numa_placement,
)

__all__ = [
    "NumaPlacement",
    "alloc_mmap",
    "alloc_shm",
    "sample_numa_placement",
]
