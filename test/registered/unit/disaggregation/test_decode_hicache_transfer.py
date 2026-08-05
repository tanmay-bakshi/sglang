"""Focused tests for decode-side HiCache restoration lifecycle semantics."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import LoadBackResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TransferHarness(DecodeHiCacheTransferMixin):
    """Minimal owner for exercising the transfer mixin directly."""

    tree_cache: Any

    def __init__(self, tree_cache: Any) -> None:
        """Initialize the harness.

        :param tree_cache: Cache fixture implementing the mixin's dependencies.
        """

        self.tree_cache = tree_cache


def _make_tree_cache(
    *,
    load_back_result: LoadBackResult,
    consumer_index: int = 0,
) -> SimpleNamespace:
    """Build a cache fixture with one free producer slot.

    :param load_back_result: Result returned by ``init_load_back``.
    :param consumer_index: Controller ticket returned when queued work starts.
    :returns: Minimal cache fixture for local-restore processing.
    """

    return SimpleNamespace(
        cache_controller=SimpleNamespace(
            layer_done_counter=SimpleNamespace(producer_index=0, num_counters=2)
        ),
        check_prefetch_progress=MagicMock(return_value=True),
        pop_prefetch_loaded_tokens=MagicMock(),
        init_load_back=MagicMock(return_value=load_back_result),
        inc_lock_ref=MagicMock(),
        dec_lock_ref=MagicMock(),
        is_load_back_event_done=MagicMock(return_value=True),
        ready_to_load_host_cache=MagicMock(return_value=consumer_index),
        release_aborted_request=MagicMock(),
        req_to_token_pool=SimpleNamespace(write=MagicMock()),
    )


def _make_prefix_match(
    *,
    full_device_tokens: int,
    full_host_tokens: int,
    swa_host_tokens: int,
) -> DecodePrefixMatch:
    """Build one component-aware prefix match.

    :param full_device_tokens: Full-KV tokens already resident on device.
    :param full_host_tokens: Full-KV tokens requiring local restoration.
    :param swa_host_tokens: SWA tokens requiring local restoration.
    :returns: Prefix-match fixture.
    """

    return DecodePrefixMatch(
        prefix_indices=torch.arange(full_device_tokens, dtype=torch.int64),
        l2_host_hit_length=full_host_tokens,
        l3_storage_hit_length=0,
        last_device_node=object(),
        swa_host_hit_length=swa_host_tokens,
        page_size=1,
    )


def _make_decode_request(prefix_match: DecodePrefixMatch, *, rid: str) -> DecodeRequest:
    """Build a decode request carrying one prefix-match lease.

    :param prefix_match: Component-aware local-restore intent.
    :param rid: Stable request identifier.
    :returns: Decode request fixture.
    """

    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(prefix_match.decode_prefix_len)),
        req_pool_idx=3,
        prefix_indices=prefix_match.prefix_indices.clone(),
        last_node=prefix_match.last_device_node,
    )
    return DecodeRequest(
        req=req,
        kv_receiver=MagicMock(),
        prefix_match=prefix_match,
    )


def _make_rematch(
    prefix_match: DecodePrefixMatch,
    *,
    device_indices: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Build the cache match observed immediately before load-back.

    :param prefix_match: Original admitted match.
    :param device_indices: Current full-KV device indices, if changed.
    :returns: Match fixture consumed by the transfer mixin.
    """

    resolved_device_indices = (
        prefix_match.prefix_indices if device_indices is None else device_indices
    )
    return SimpleNamespace(
        device_indices=resolved_device_indices,
        best_match_node=object(),
        last_device_node=prefix_match.last_device_node,
        last_host_node=prefix_match.last_host_node,
        host_hit_length=prefix_match.l2_host_hit_length,
        swa_host_hit_length=prefix_match.swa_host_hit_length,
        mamba_host_hit_length=prefix_match.mamba_host_hit_length,
    )


class TestDecodeHiCacheTransferLifecycle(unittest.TestCase):
    """Validates component-aware restoration, completion, and ownership."""

    def test_swa_only_queued_restore_with_no_full_indices_stays_pending(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=4,
            full_host_tokens=0,
            swa_host_tokens=2,
        )
        restored_node = object()
        result = LoadBackResult(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=restored_node,
            queued_any_component=True,
            full_tokens=0,
            swa_tokens=2,
        )
        tree_cache = _make_tree_cache(load_back_result=result, consumer_index=0)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="swa-only")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.PENDING,
        )
        self.assertIs(decode_req.hicache_restored_node, restored_node)
        self.assertEqual(decode_req.hicache_load_consumer_index, 0)
        tree_cache.ready_to_load_host_cache.assert_called_once_with()

    def test_merged_full_and_swa_restore_waits_for_one_completion_ticket(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=2,
        )
        restored_node = object()
        new_full_indices = torch.tensor([7, 8], dtype=torch.int64)
        result = LoadBackResult(
            new_full_device_indices=new_full_indices,
            restored_node=restored_node,
            queued_any_component=True,
            full_tokens=2,
            swa_tokens=2,
        )
        tree_cache = _make_tree_cache(load_back_result=result, consumer_index=0)
        completion = {"done": False}
        tree_cache.is_load_back_event_done.side_effect = lambda index: (
            completion["done"] if index == 0 else True
        )
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="merged")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])
            harness._process_hicache_local_restores([decode_req])
            self.assertEqual(
                decode_req.hicache_restore_status,
                HiCacheRestoreResult.PENDING,
            )

            completion["done"] = True
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        tree_cache.init_load_back.assert_called_once()
        tree_cache.ready_to_load_host_cache.assert_called_once_with()

    def test_synchronous_restoration_completes_without_controller_ticket(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        restored_node = object()
        new_full_indices = torch.tensor([11, 12], dtype=torch.int64)
        result = LoadBackResult(
            new_full_device_indices=new_full_indices,
            restored_node=restored_node,
            queued_any_component=False,
            full_tokens=2,
            swa_tokens=0,
        )
        tree_cache = _make_tree_cache(load_back_result=result)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="synchronous")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        self.assertIs(decode_req.hicache_restored_node, restored_node)
        self.assertTrue(
            torch.equal(
                decode_req.hicache_restored_kv_indices,
                new_full_indices,
            )
        )
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_concurrent_full_restore_is_preserved_without_a_second_copy(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        restored_node = object()
        result = LoadBackResult(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=restored_node,
            queued_any_component=False,
            full_tokens=0,
            swa_tokens=0,
        )
        tree_cache = _make_tree_cache(load_back_result=result)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="concurrent-full")
        concurrent_indices = torch.tensor([0, 1, 71, 72], dtype=torch.int64)

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(
                prefix_match,
                device_indices=concurrent_indices,
            ),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        self.assertTrue(
            torch.equal(
                decode_req.hicache_restored_kv_indices,
                concurrent_indices[2:],
            )
        )
        self.assertIs(decode_req.req.last_node, prefix_match.last_device_node)
        self.assertTrue(
            torch.equal(decode_req.req.prefix_indices, prefix_match.prefix_indices)
        )
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_host_resident_component_without_restore_or_queue_fails_closed(
        self,
    ) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=4,
            full_host_tokens=0,
            swa_host_tokens=2,
        )
        result = LoadBackResult(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=prefix_match.last_device_node,
            queued_any_component=False,
            full_tokens=0,
            swa_tokens=0,
        )
        tree_cache = _make_tree_cache(load_back_result=result)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="missing-swa")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertIsNone(decode_req.hicache_restored_node)
        tree_cache.inc_lock_ref.assert_not_called()
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_queued_restore_without_controller_ticket_fails(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        result = LoadBackResult(
            new_full_device_indices=torch.tensor([21, 22], dtype=torch.int64),
            restored_node=object(),
            queued_any_component=True,
            full_tokens=2,
            swa_tokens=0,
        )
        tree_cache = _make_tree_cache(
            load_back_result=result,
            consumer_index=-1,
        )
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="missing-ticket")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertEqual(decode_req.hicache_load_consumer_index, -1)

    def test_cleanup_and_commit_transfer_ownership_exactly_once(self) -> None:
        abort_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        abort_match.prefetch_registered = True
        abort_node = object()
        abort_result = LoadBackResult(
            new_full_device_indices=torch.tensor([31, 32], dtype=torch.int64),
            restored_node=abort_node,
            queued_any_component=False,
            full_tokens=2,
            swa_tokens=0,
        )
        abort_cache = _make_tree_cache(load_back_result=abort_result)
        abort_harness = _TransferHarness(abort_cache)
        abort_req = _make_decode_request(abort_match, rid="abort")
        abort_req.hicache_restored_node = abort_node

        abort_harness._clean_hicache_prefetch_resources(abort_req)
        abort_harness._clean_hicache_prefetch_resources(abort_req)

        abort_cache.release_aborted_request.assert_called_once_with("abort")
        abort_cache.dec_lock_ref.assert_called_once_with(abort_node)
        self.assertFalse(abort_match.prefetch_registered)
        self.assertIsNone(abort_req.hicache_restored_node)

        commit_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        restored_node = object()
        commit_result = LoadBackResult(
            new_full_device_indices=torch.tensor([41, 42], dtype=torch.int64),
            restored_node=restored_node,
            queued_any_component=False,
            full_tokens=2,
            swa_tokens=0,
        )
        commit_cache = _make_tree_cache(load_back_result=commit_result)
        commit_harness = _TransferHarness(commit_cache)
        commit_req = _make_decode_request(commit_match, rid="commit")
        commit_req.hicache_restored_node = restored_node
        commit_req.hicache_restored_kv_indices = commit_result.new_full_device_indices
        commit_req.hicache_restore_status = HiCacheRestoreResult.READY
        expected_prefix_indices = torch.cat(
            [
                commit_match.prefix_indices,
                commit_result.new_full_device_indices,
            ]
        )

        commit_harness._commit_hicache_local_restore_to_req(commit_req)
        commit_harness._commit_hicache_local_restore_to_req(commit_req)
        commit_harness._clean_hicache_prefetch_resources(commit_req)

        commit_cache.dec_lock_ref.assert_called_once_with(commit_match.last_device_node)
        commit_cache.req_to_token_pool.write.assert_called_once()
        self.assertTrue(
            torch.equal(
                commit_req.req.prefix_indices,
                expected_prefix_indices,
            )
        )
        self.assertIs(commit_req.req.last_node, restored_node)
        self.assertIsNone(commit_req.prefix_match)
        self.assertIsNone(commit_req.hicache_restored_node)
        self.assertIsNone(commit_req.hicache_restored_kv_indices)
        self.assertEqual(commit_req.hicache_load_consumer_index, -1)


if __name__ == "__main__":
    unittest.main()
