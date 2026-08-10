"""Tests for the explicit cache load-back ticket contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    LoadBackTicket,
    LoadBackTicketState,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLoadBackTicket(unittest.TestCase):
    """Validates transfer state independently of the full-index tensor."""

    def test_default_ticket_owns_no_copy_and_is_prepared(self) -> None:
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=object(),
        )

        self.assertFalse(ticket.queued_any_component)
        self.assertEqual(ticket.state, LoadBackTicketState.PREPARED)
        self.assertFalse(ticket.owns_restored_lock)

    def test_sidecar_only_ticket_reports_controller_work(self) -> None:
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=object(),
            queued_sidecars=frozenset({PoolName.INDEXER}),
        )

        self.assertTrue(ticket.queued_any_component)

    def test_hiradix_success_reports_async_full_transfer(self) -> None:
        node = SimpleNamespace(evicted=True, id=17)
        loaded_indices = torch.tensor([3, 5, 7], dtype=torch.int64)
        cache = SimpleNamespace(load_back=MagicMock(return_value=loaded_indices))

        result = HiRadixCache.init_load_back(
            cache,
            InitLoadBackParams(
                best_match_node=node,
                host_hit_length=len(loaded_indices),
            ),
        )

        self.assertIs(result.restored_node, node)
        self.assertIs(result.new_full_device_indices, loaded_indices)
        self.assertTrue(result.queued_any_component)
        self.assertEqual(result.full_tokens, len(loaded_indices))
        self.assertEqual(result.swa_tokens, 0)

    def test_hiradix_failed_load_falls_back_without_queued_transfer(self) -> None:
        resident_parent = SimpleNamespace(evicted=False)
        evicted_node = SimpleNamespace(evicted=True, parent=resident_parent)
        empty_indices = torch.empty(0, dtype=torch.int64)
        cache = SimpleNamespace(
            load_back=MagicMock(return_value=None),
            _empty_match_result=SimpleNamespace(device_indices=empty_indices),
        )

        result = HiRadixCache.init_load_back(
            cache,
            InitLoadBackParams(
                best_match_node=evicted_node,
                host_hit_length=8,
            ),
        )

        self.assertIs(result.restored_node, resident_parent)
        self.assertIs(result.new_full_device_indices, empty_indices)
        self.assertFalse(result.queued_any_component)
        self.assertEqual(result.full_tokens, 0)
        self.assertEqual(result.swa_tokens, 0)

    def test_mamba_only_success_reports_queued_component(self) -> None:
        node = SimpleNamespace(
            evicted=False,
            id=23,
            mamba_evicted=True,
            mamba_backuped=True,
        )
        empty_full_indices = torch.empty(0, dtype=torch.int64)
        cache = SimpleNamespace(
            load_back=MagicMock(return_value=empty_full_indices),
        )

        result = HiMambaRadixCache.init_load_back(
            cache,
            InitLoadBackParams(
                best_match_node=node,
                host_hit_length=0,
                req=SimpleNamespace(rid="mamba-only"),
            ),
        )

        self.assertIs(result.restored_node, node)
        self.assertIs(result.new_full_device_indices, empty_full_indices)
        self.assertTrue(result.queued_any_component)
        self.assertEqual(result.full_tokens, 0)
        self.assertEqual(result.swa_tokens, 0)


if __name__ == "__main__":
    unittest.main()
