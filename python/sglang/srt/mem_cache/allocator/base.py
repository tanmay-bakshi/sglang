"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from sglang.srt.mem_cache.allocation_pin import (
    AllocationPin,
    AllocationPinRegistry,
    AllocationPinSnapshot,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self._allocation_pin_registry = AllocationPinRegistry(type(self).__name__)

    def acquire_allocation_pin(
        self,
        indices: torch.Tensor,
        owner: object,
    ) -> AllocationPin:
        """Pin exact allocator-visible token indices.

        :param indices: Allocator-visible token indices.
        :param owner: Exact authority allowed to release the pin.
        :returns: Opaque allocator-owned pin.
        """

        return self._allocation_pin_registry.acquire(
            self._allocation_page_ids(indices),
            owner,
        )

    def allocation_pin_snapshot(
        self,
        pin: AllocationPin,
    ) -> AllocationPinSnapshot:
        """Resolve immutable virtual and physical page identities.

        Static allocators use identity virtual-to-physical mappings.

        :param pin: Exact allocator-owned pin.
        :returns: Immutable allocation mapping.
        """

        page_ids = self._allocation_pin_registry.page_ids(pin)
        return AllocationPinSnapshot(
            allocator_label=type(self).__name__,
            page_size=self.page_size,
            virtual_pages=page_ids,
            physical_pages=page_ids,
            quarantined=self._allocation_pin_registry.is_quarantined(pin),
        )

    def release_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Release one exact allocation pin.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact acquisition authority.
        """

        self._allocation_pin_registry.release(pin, owner)

    def quarantine_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Permanently retain one ambiguous allocation pin.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact acquisition authority.
        """

        self._allocation_pin_registry.quarantine(pin, owner)

    def _assert_allocation_indices_reusable(
        self,
        indices: torch.Tensor,
    ) -> None:
        """Reject mutation of pinned token indices.

        :param indices: Allocator-visible token indices about to be mutated.
        """

        self.assert_allocation_indices_reusable(indices)

    def assert_allocation_indices_reusable(
        self,
        indices: torch.Tensor,
    ) -> None:
        """Reject mutation of exact pinned token indices.

        Composite allocators use this preflight before mutating either child
        allocator, so a pin on the second child cannot leave the first child
        partially freed.

        :param indices: Allocator-visible token indices about to be mutated.
        """

        if indices.numel() == 0:
            return
        self._allocation_pin_registry.assert_pages_reusable(
            self._allocation_page_ids(indices)
        )

    def _assert_allocation_resettable(self, operation: str) -> None:
        """Reject allocator-wide mutation while any page is pinned.

        :param operation: Reader-facing allocator operation.
        """

        self.assert_allocation_resettable(operation)

    def assert_allocation_resettable(self, operation: str) -> None:
        """Reject allocator-wide mutation while any page is pinned.

        :param operation: Reader-facing allocator operation.
        """

        self._allocation_pin_registry.assert_resettable(operation)

    def _allocation_page_ids(self, indices: torch.Tensor) -> tuple[int, ...]:
        """Canonicalize token indices to positive page IDs.

        :param indices: Allocator-visible token indices.
        :returns: Sorted unique allocator page IDs.
        """

        if not isinstance(indices, torch.Tensor):
            raise TypeError("allocation pin indices must be a torch.Tensor")
        if indices.ndim != 1:
            raise ValueError("allocation pin indices must be one-dimensional")
        if indices.numel() == 0:
            raise ValueError("allocation pin indices must not be empty")
        pages = torch.unique(
            indices.detach().to(dtype=torch.int64) // self.page_size
        )
        page_ids = tuple(sorted(int(page_id) for page_id in pages.cpu().tolist()))
        if page_ids[0] <= 0:
            raise ValueError("allocation pin indices include reserved page zero")
        return page_ids

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self._assert_allocation_resettable("restore allocator state")
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
