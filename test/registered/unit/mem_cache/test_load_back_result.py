"""Tests for the explicit cache load-back result contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    LoadBackResult,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache


class TestLoadBackResult(unittest.TestCase):
    """Validates transfer state independently of the full-index tensor."""

    def test_all_fields_are_required(self) -> None:
        with self.assertRaises(TypeError):
            LoadBackResult(  # type: ignore[call-arg]
                new_full_device_indices=torch.empty(0, dtype=torch.int64),
                restored_node=object(),
                queued_any_component=False,
            )

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

if __name__ == "__main__":
    unittest.main()
