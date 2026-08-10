"""Focused tests for decode-side HiCache ticket ownership."""

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
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    LoadBackPublication,
    LoadBackTicket,
    LoadBackTicketState,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TransferHarness(DecodeHiCacheTransferMixin):
    """Minimal owner for exercising the transfer mixin directly."""

    tree_cache: Any
    tp_size: int
    gloo_group: Any

    def __init__(self, tree_cache: Any) -> None:
        """Initialize the harness.

        :param tree_cache: Cache fixture implementing mixin dependencies.
        """

        self.tree_cache = tree_cache
        self.tp_size = 1
        self.gloo_group = None


def _make_tree_cache(
    *,
    ticket: LoadBackTicket,
    consumer_index: int = 1,
) -> SimpleNamespace:
    """Build a cache fixture with one free producer slot.

    :param ticket: Ticket returned by ``init_load_back``.
    :param consumer_index: Generation returned when prepared work starts.
    :returns: Minimal cache fixture for local-restore processing.
    """

    finish_event = MagicMock()
    tree_cache = SimpleNamespace(
        cache_controller=SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                producer_index=0,
                num_counters=2,
                events=[
                    SimpleNamespace(finish_event=finish_event),
                    SimpleNamespace(finish_event=finish_event),
                ],
            )
        ),
        check_prefetch_progress=MagicMock(return_value=True),
        pop_prefetch_loaded_tokens=MagicMock(),
        init_load_back=MagicMock(return_value=ticket),
        inc_lock_ref=MagicMock(return_value=IncLockRefResult(swa_uuid_for_lock=27)),
        dec_lock_ref=MagicMock(),
        is_load_back_event_done=MagicMock(return_value=True),
        ready_to_load_host_cache=MagicMock(return_value=consumer_index),
        release_aborted_request=MagicMock(),
        req_to_token_pool=SimpleNamespace(write=MagicMock()),
    )

    def publish_load_back(prepared_ticket: LoadBackTicket) -> None:
        """Model reversible local tree publication.

        :param prepared_ticket: Ticket receiving its restored-node lock.
        """

        lock_result = tree_cache.inc_lock_ref(prepared_ticket.restored_node)
        prepared_ticket.restored_lock_params = lock_result.to_dec_params()
        prepared_ticket.owns_restored_lock = True
        prepared_ticket.state = LoadBackTicketState.PUBLISHED

    def commit_load_back_publication(prepared_ticket: LoadBackTicket) -> None:
        """Model irreversible publication after TP agreement.

        :param prepared_ticket: Globally accepted ticket.
        """

        prepared_ticket.publication = None

    tree_cache.publish_load_back = MagicMock(side_effect=publish_load_back)
    tree_cache.commit_load_back_publication = MagicMock(
        side_effect=commit_load_back_publication
    )
    tree_cache.abort_load_back_publication = MagicMock()
    return tree_cache


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
        last_device_lock_params=DecLockRefParams(swa_uuid_for_lock=11),
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
        last_node_lock_params=prefix_match.last_device_lock_params,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=prefix_match.l2_host_hit_length,
        swa_host_hit_length=prefix_match.swa_host_hit_length,
        mamba_host_hit_length=prefix_match.mamba_host_hit_length,
        num_matched_prefix_tokens=prefix_match.decode_prefix_len,
        mamba_branching_seqlen=None,
        cache_protected_len=prefix_match.l1_prefix_len,
        swa_uuid_for_lock=11,
        swa_prefix_lock_released=False,
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
    full_host_tokens: int | None = None,
) -> SimpleNamespace:
    """Build the cache match observed immediately before load-back.

    :param prefix_match: Original admitted match.
    :param device_indices: Current full-KV device indices, if changed.
    :param full_host_tokens: Current host-resident full-KV coverage.
    :returns: Match fixture consumed by the transfer mixin.
    """

    resolved_device_indices = (
        prefix_match.prefix_indices if device_indices is None else device_indices
    )
    resolved_host_tokens = (
        prefix_match.l2_host_hit_length
        if full_host_tokens is None
        else full_host_tokens
    )
    return SimpleNamespace(
        device_indices=resolved_device_indices,
        best_match_node=object(),
        last_device_node=prefix_match.last_device_node,
        last_host_node=prefix_match.last_host_node,
        host_hit_length=resolved_host_tokens,
        swa_host_hit_length=prefix_match.swa_host_hit_length,
        mamba_host_hit_length=prefix_match.mamba_host_hit_length,
    )


def _ticket(
    *,
    restored_node: Any,
    full_indices: torch.Tensor,
    full_tokens: int,
    swa_tokens: int,
    queued_components: frozenset[ComponentType] = frozenset(),
) -> LoadBackTicket:
    """Create one unowned load-back ticket.

    :param restored_node: Restored cache node.
    :param full_indices: Newly allocated full-KV indices.
    :param full_tokens: Full-KV transfer coverage.
    :param swa_tokens: SWA transfer coverage.
    :param queued_components: Components queued for asynchronous copying.
    :returns: Prepared ticket.
    """

    return LoadBackTicket(
        new_full_device_indices=full_indices,
        restored_node=restored_node,
        queued_components=queued_components,
        full_tokens=full_tokens,
        swa_tokens=swa_tokens,
        publication=(LoadBackPublication() if len(queued_components) > 0 else None),
    )


class TestDecodeHiCacheTicketLifecycle(unittest.TestCase):
    """Validate merged generations, exact cleanup, and lock transfer."""

    def test_swa_only_restore_stays_pending_until_its_generation_finishes(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=4,
            full_host_tokens=0,
            swa_host_tokens=2,
        )
        restored_node = object()
        ticket = _ticket(
            restored_node=restored_node,
            full_indices=torch.empty(0, dtype=torch.int64),
            full_tokens=0,
            swa_tokens=2,
            queued_components=frozenset({ComponentType.SWA}),
        )
        tree_cache = _make_tree_cache(ticket=ticket)
        completion = {"done": False}
        tree_cache.is_load_back_event_done.side_effect = lambda index: (
            completion["done"]
            if index == 1 and ticket.state == LoadBackTicketState.STARTED
            else True
        )
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
            completion["done"] = True
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(ticket.consumer_index, 1)
        tree_cache.ready_to_load_host_cache.assert_called_once_with(force_empty=True)

    def test_started_generation_waits_for_every_tp_rank(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([7, 8], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
            queued_components=frozenset({ComponentType.FULL}),
        )
        ticket.state = LoadBackTicketState.STARTED
        ticket.consumer_index = 1
        tree_cache = _make_tree_cache(ticket=ticket)
        tree_cache.is_load_back_event_done.return_value = True
        harness = _TransferHarness(tree_cache)
        harness.tp_size = 2
        harness.gloo_group = object()
        decode_req = _make_decode_request(prefix_match, rid="tp-completion")
        decode_req.hicache_load_back_ticket = ticket

        def hold_global_completion(result: torch.Tensor, **_: Any) -> None:
            result.fill_(0)

        with patch(
            "torch.distributed.all_reduce",
            side_effect=hold_global_completion,
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.PENDING,
        )
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_no_copy_ticket_shares_an_earlier_requests_merged_generation(self) -> None:
        queued_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        no_copy_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        queued_ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([7, 8], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
            queued_components=frozenset({ComponentType.FULL}),
        )
        no_copy_ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([17, 18], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
        )
        tree_cache = _make_tree_cache(ticket=queued_ticket, consumer_index=1)
        tree_cache.init_load_back.side_effect = [queued_ticket, no_copy_ticket]
        harness = _TransferHarness(tree_cache)
        queued_req = _make_decode_request(queued_match, rid="queued")
        no_copy_req = _make_decode_request(no_copy_match, rid="no-copy")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            side_effect=[
                _make_rematch(queued_match),
                _make_rematch(no_copy_match),
            ],
        ):
            harness._process_hicache_local_restores([queued_req, no_copy_req])

        self.assertEqual(queued_ticket.consumer_index, 1)
        self.assertEqual(no_copy_ticket.consumer_index, 1)
        self.assertEqual(queued_ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(no_copy_ticket.state, LoadBackTicketState.STARTED)
        tree_cache.ready_to_load_host_cache.assert_called_once_with(force_empty=True)

    def test_partial_preparation_failure_cancels_every_local_ticket(
        self,
    ) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([21, 22], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
            queued_components=frozenset({ComponentType.FULL}),
        )
        ticket.restored_lock_params = DecLockRefParams(swa_uuid_for_lock=27)
        ticket.owns_restored_lock = True
        tree_cache = _make_tree_cache(ticket=ticket)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="partial")
        decode_req.hicache_load_back_ticket = ticket
        harness._agree_hicache_preparation = MagicMock(return_value=(False, True))

        harness._abort_failed_hicache_preparation([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.ABORTED)
        self.assertFalse(ticket.owns_restored_lock)
        tree_cache.ready_to_load_host_cache.assert_not_called()
        tree_cache.abort_load_back_publication.assert_called_once_with(ticket)
        tree_cache.dec_lock_ref.assert_called_once_with(
            ticket.restored_node,
            ticket.restored_lock_params,
        )

    def test_preparation_rollback_attempts_every_ticket_after_local_error(self) -> None:
        prefix_matches = [
            _make_prefix_match(
                full_device_tokens=2,
                full_host_tokens=2,
                swa_host_tokens=0,
            )
            for _ in range(2)
        ]
        tickets = [
            _ticket(
                restored_node=object(),
                full_indices=torch.tensor([21, 22], dtype=torch.int64),
                full_tokens=2,
                swa_tokens=0,
                queued_components=frozenset({ComponentType.FULL}),
            )
            for _ in range(2)
        ]
        tree_cache = _make_tree_cache(ticket=tickets[0])
        tree_cache.abort_load_back_publication.side_effect = [
            RuntimeError("injected rollback failure"),
            None,
        ]
        harness = _TransferHarness(tree_cache)
        decode_reqs = [
            _make_decode_request(prefix_match, rid=f"rollback-{index}")
            for index, prefix_match in enumerate(prefix_matches)
        ]
        for decode_req, ticket in zip(decode_reqs, tickets, strict=True):
            decode_req.hicache_load_back_ticket = ticket

        with self.assertRaisesRegex(
            RuntimeError,
            "HiCache preparation rollback failed",
        ):
            harness._abort_failed_hicache_preparation(decode_reqs)

        self.assertEqual(tree_cache.abort_load_back_publication.call_count, 2)
        self.assertEqual(
            [ticket.state for ticket in tickets],
            [LoadBackTicketState.ABORTED, LoadBackTicketState.ABORTED],
        )

    def test_cross_rank_publication_rejection_rolls_back_before_generation(
        self,
    ) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([25, 26], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
            queued_components=frozenset({ComponentType.FULL}),
        )
        tree_cache = _make_tree_cache(ticket=ticket)
        harness = _TransferHarness(tree_cache)
        harness._agree_hicache_publication = MagicMock(return_value=False)
        decode_req = _make_decode_request(prefix_match, rid="publication-rejected")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.ABORTED)
        tree_cache.publish_load_back.assert_called_once_with(ticket)
        tree_cache.abort_load_back_publication.assert_called_once_with(ticket)
        tree_cache.commit_load_back_publication.assert_not_called()
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_cross_rank_generation_divergence_rolls_back_before_commit(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        ticket = _ticket(
            restored_node=object(),
            full_indices=torch.tensor([27, 28], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
            queued_components=frozenset({ComponentType.FULL}),
        )
        tree_cache = _make_tree_cache(ticket=ticket)
        harness = _TransferHarness(tree_cache)
        harness._agree_hicache_generation_index = MagicMock(return_value=None)
        decode_req = _make_decode_request(prefix_match, rid="generation-diverged")

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            return_value=_make_rematch(prefix_match),
        ):
            harness._process_hicache_local_restores([decode_req])

        self.assertEqual(
            decode_req.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.ABORTED)
        tree_cache.abort_load_back_publication.assert_called_once_with(ticket)
        tree_cache.commit_load_back_publication.assert_not_called()
        tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_abort_is_idempotent_and_preserves_exact_lock_parameters(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        prefix_match.prefetch_registered = True
        restored_node = object()
        ticket = _ticket(
            restored_node=restored_node,
            full_indices=torch.tensor([31, 32], dtype=torch.int64),
            full_tokens=2,
            swa_tokens=0,
        )
        ticket.restored_lock_params = DecLockRefParams(swa_uuid_for_lock=27)
        ticket.owns_restored_lock = True
        tree_cache = _make_tree_cache(ticket=ticket)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="abort")
        decode_req.hicache_load_back_ticket = ticket

        harness._clean_hicache_prefetch_resources(decode_req)
        harness._clean_hicache_prefetch_resources(decode_req)

        tree_cache.release_aborted_request.assert_called_once_with("abort")
        tree_cache.dec_lock_ref.assert_called_once_with(
            restored_node,
            ticket.restored_lock_params,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.ABORTED)
        self.assertFalse(ticket.owns_restored_lock)

    def test_commit_transfers_lock_ownership_exactly_once(self) -> None:
        prefix_match = _make_prefix_match(
            full_device_tokens=2,
            full_host_tokens=2,
            swa_host_tokens=0,
        )
        restored_node = object()
        restored_indices = torch.tensor([41, 42], dtype=torch.int64)
        ticket = _ticket(
            restored_node=restored_node,
            full_indices=restored_indices,
            full_tokens=2,
            swa_tokens=0,
        )
        ticket.restored_full_device_indices = restored_indices
        ticket.restored_lock_params = DecLockRefParams(swa_uuid_for_lock=27)
        ticket.owns_restored_lock = True
        tree_cache = _make_tree_cache(ticket=ticket)
        harness = _TransferHarness(tree_cache)
        decode_req = _make_decode_request(prefix_match, rid="commit")
        decode_req.hicache_load_back_ticket = ticket
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        original_lock_params = prefix_match.last_device_lock_params

        harness._commit_hicache_local_restore_to_req(decode_req)

        harness._commit_hicache_local_restore_to_req(decode_req)
        harness._clean_hicache_prefetch_resources(decode_req)

        tree_cache.dec_lock_ref.assert_called_once_with(
            prefix_match.last_device_node,
            original_lock_params,
        )
        tree_cache.req_to_token_pool.write.assert_called_once()
        self.assertTrue(
            torch.equal(
                decode_req.req.prefix_indices,
                torch.tensor([0, 1, 41, 42], dtype=torch.int64),
            )
        )
        self.assertIs(decode_req.req.last_node, restored_node)
        self.assertEqual(decode_req.req.cache_protected_len, 4)
        self.assertEqual(decode_req.req.swa_uuid_for_lock, 27)
        self.assertIsNone(decode_req.prefix_match)
        self.assertIsNone(decode_req.hicache_load_back_ticket)
        self.assertEqual(ticket.state, LoadBackTicketState.COMMITTED)
        self.assertFalse(ticket.owns_restored_lock)


if __name__ == "__main__":
    unittest.main()
