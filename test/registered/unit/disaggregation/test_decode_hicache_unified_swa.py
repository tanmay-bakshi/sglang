import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodePrefixMatch(unittest.TestCase):
    """Tests component-aware decode-side restore intent and capacity."""

    def test_swa_only_hit_requires_page_rounded_local_restore(self) -> None:
        """Full KV on device does not make a host-resident SWA tail runnable."""

        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(4096),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=9,
            swa_host_hit_length=1023,
            page_size=64,
        )

        self.assertTrue(prefix_match.needs_local_restore)
        self.assertEqual(prefix_match.full_restore_token_count, 0)
        self.assertEqual(prefix_match.swa_restore_token_count, 1024)

    def test_full_and_swa_restore_budgets_are_independent(self) -> None:
        """Full-prefix length cannot stand in for a distinct SWA allocation."""

        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(256),
            l2_host_hit_length=769,
            l3_storage_hit_length=0,
            last_device_node=4,
            swa_host_hit_length=127,
            page_size=64,
        )

        self.assertEqual(prefix_match.full_restore_token_count, 832)
        self.assertEqual(prefix_match.swa_restore_token_count, 128)

    def test_mamba_only_hit_requires_local_restore(self) -> None:
        """Host-resident recurrent state must complete before decode runs."""

        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(128),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=4,
            mamba_host_hit_length=1,
            page_size=64,
        )

        self.assertTrue(prefix_match.needs_local_restore)
        self.assertEqual(prefix_match.full_restore_token_count, 0)
        self.assertEqual(prefix_match.swa_restore_token_count, 0)


class TestDecodePrefixMatchConstruction(unittest.TestCase):
    """Tests propagation of rank-local unified-cache match metadata."""

    def test_component_hits_and_allocator_page_size_are_preserved(self) -> None:
        """Admission must retain the component restore intent from matching."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=False)
        mixin.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
        result = SimpleNamespace(
            device_indices=torch.arange(256),
            host_hit_length=129,
            swa_host_hit_length=63,
            mamba_host_hit_length=1,
            last_device_node=7,
            last_host_node=8,
        )

        prefix_match = mixin._build_decode_prefix_match(
            SimpleNamespace(origin_input_ids=list(range(512))),
            result,
        )

        self.assertEqual(prefix_match.full_restore_token_count, 192)
        self.assertEqual(prefix_match.swa_restore_token_count, 64)
        self.assertEqual(prefix_match.mamba_host_hit_length, 1)


class TestPendingRestoreBudgets(unittest.TestCase):
    """Tests rank-local pending full and SWA admission reservations."""

    def test_pending_requests_reserve_each_component_pool(self) -> None:
        """Only pending, not-yet-allocated restores contribute to both budgets."""

        first = DecodePrefixMatch(
            prefix_indices=torch.arange(64),
            l2_host_hit_length=65,
            l3_storage_hit_length=0,
            last_device_node=1,
            swa_host_hit_length=33,
            page_size=32,
        )
        second = DecodePrefixMatch(
            prefix_indices=torch.arange(64),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=2,
            swa_host_hit_length=63,
            page_size=32,
        )
        ignored = dataclasses.replace(first, l2_host_hit_length=4096)
        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)
        mixin.transfer_queue = SimpleNamespace(
            queue=[
                SimpleNamespace(
                    prefix_match=first,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_restored_node=None,
                ),
                SimpleNamespace(
                    prefix_match=second,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_restored_node=None,
                ),
                SimpleNamespace(
                    prefix_match=ignored,
                    hicache_restore_status=HiCacheRestoreResult.READY,
                    hicache_restored_node=None,
                ),
            ]
        )

        self.assertEqual(mixin._hicache_pending_restore_budgets(), (96, 128))


class TestUnifiedSwaRestoreQueueing(unittest.TestCase):
    """Tests SWA-only restore submission through the PD local-restore state machine."""

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_empty_full_indices_still_queue_component_dma(
        self,
        match_prefix: Mock,
    ) -> None:
        """An empty full-KV result must not strand a queued SWA H2D copy."""

        prefix_indices = torch.arange(8)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=17,
            swa_host_hit_length=7,
            page_size=1,
        )
        request = SimpleNamespace(
            rid="swa-only",
            origin_input_ids=list(range(8)),
        )
        match_prefix.return_value = SimpleNamespace(
            best_match_node=31,
            host_hit_length=0,
            device_indices=prefix_indices,
        )
        tree_cache = SimpleNamespace(
            init_load_back=Mock(return_value=(torch.empty(0, dtype=torch.int64), 17)),
            inc_lock_ref=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = SimpleNamespace(
            prefix_match=prefix_match,
            req=request,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            hicache_restored_node=None,
            hicache_restored_kv_indices=None,
        )

        queued = mixin._try_hicache_queue_load_back(decode_request)

        self.assertTrue(queued)
        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.PENDING,
        )
        self.assertEqual(decode_request.hicache_restored_kv_indices.numel(), 0)
        tree_cache.inc_lock_ref.assert_called_once_with(17)


class TestUnifiedRestoreCompletion(unittest.TestCase):
    """Tests unified full/SWA completion polling and acknowledgement."""

    def test_completed_event_reaps_unified_controller_ack(self) -> None:
        """A signaled component event is acknowledged before reporting ready."""

        finish_event = Mock()
        finish_event.query.return_value = True
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                events=[SimpleNamespace(finish_event=finish_event)]
            )
        )
        cache.loading_check = Mock()

        self.assertTrue(cache.is_load_back_event_done(0))
        cache.loading_check.assert_called_once_with()

    def test_incomplete_event_does_not_acknowledge_or_report_ready(self) -> None:
        """A pending component copy remains behind the PD success gate."""

        finish_event = Mock()
        finish_event.query.return_value = False
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                events=[SimpleNamespace(finish_event=finish_event)]
            )
        )
        cache.loading_check = Mock()

        self.assertFalse(cache.is_load_back_event_done(0))
        cache.loading_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
