"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from sglang.srt.mem_cache.allocation_pin import (
    AllocationPin,
    AllocationPinRegistry,
    AllocationPinSnapshot,
)


class MambaSlotAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self.page_size = 1
        self._allocation_pin_registry = AllocationPinRegistry(type(self).__name__)
        # Active preallocated batch for `alloc_group_begin` / `alloc_group_end`.
        # When non-None, `alloc(1)` consumes the next slot from this iterator
        # instead of calling `_do_alloc(1)` per request. Reset to None outside
        # a group window so `alloc` falls through to the per-call path.
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Identity to ``available_size`` for the
        static pool (slot-count and byte-coordinated views coincide); the shared
        ``UnifiedMambaSlotAllocator`` overrides it with the byte-coordinated view.
        Lets ``alloc_req_slots`` call it uniformly without a getattr fallback."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.assert_allocation_indices_reusable(free_index)
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        self._allocation_pin_registry.assert_resettable("clear")
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def acquire_allocation_pin(
        self,
        indices: torch.Tensor,
        owner: object,
    ) -> AllocationPin:
        """Pin exact request-scoped Mamba slots.

        :param indices: Allocator-visible Mamba slot IDs.
        :param owner: Exact authority allowed to release the pin.
        :returns: Opaque allocator-owned pin.
        """

        return self._allocation_pin_registry.acquire(
            self._allocation_slot_ids(indices),
            owner,
        )

    def allocation_pin_snapshot(
        self,
        pin: AllocationPin,
    ) -> AllocationPinSnapshot:
        """Resolve immutable static Mamba slot identities.

        :param pin: Exact allocator-owned pin.
        :returns: Immutable identity mapping.
        """

        slot_ids = self._allocation_pin_registry.page_ids(pin)
        return AllocationPinSnapshot(
            allocator_label=type(self).__name__,
            page_size=1,
            virtual_pages=slot_ids,
            physical_pages=slot_ids,
            quarantined=self._allocation_pin_registry.is_quarantined(pin),
        )

    def release_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Release one exact Mamba allocation pin.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact acquisition authority.
        """

        self._allocation_pin_registry.release(pin, owner)

    def quarantine_allocation_pin(
        self,
        pin: AllocationPin,
        owner: object,
    ) -> None:
        """Permanently retain one ambiguous Mamba allocation.

        :param pin: Exact allocator-owned pin.
        :param owner: Exact acquisition authority.
        """

        self._allocation_pin_registry.quarantine(pin, owner)

    def assert_allocation_resettable(self, operation: str) -> None:
        """Reject allocator-wide mutation while any Mamba slot is pinned.

        :param operation: Reader-facing allocator operation.
        """

        self._allocation_pin_registry.assert_resettable(operation)

    def assert_allocation_indices_reusable(
        self,
        indices: torch.Tensor,
    ) -> None:
        """Reject mutation of exact pinned Mamba slots.

        :param indices: Allocator-visible Mamba slot IDs about to be mutated.
        """

        if indices.numel() == 0:
            return
        self._allocation_pin_registry.assert_pages_reusable(
            self._allocation_slot_ids(indices)
        )

    def _allocation_slot_ids(self, indices: torch.Tensor) -> tuple[int, ...]:
        """Canonicalize exact positive Mamba slot IDs.

        :param indices: Allocator-visible slot IDs.
        :returns: Sorted unique slot IDs.
        """

        if not isinstance(indices, torch.Tensor):
            raise TypeError("Mamba allocation pin indices must be a torch.Tensor")
        if indices.ndim != 1:
            raise ValueError("Mamba allocation pin indices must be one-dimensional")
        if indices.numel() == 0:
            raise ValueError("Mamba allocation pin indices must not be empty")
        slot_ids = tuple(
            sorted(
                int(slot_id)
                for slot_id in torch.unique(
                    indices.detach().to(dtype=torch.int64)
                )
                .cpu()
                .tolist()
            )
        )
        if slot_ids[0] <= 0:
            raise ValueError("Mamba allocation pin includes reserved slot zero")
        return slot_ids
