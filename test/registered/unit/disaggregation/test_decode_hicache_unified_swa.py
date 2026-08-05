import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    DecodeRestoreBudget,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    LoadBackTicket,
    LoadBackTicketState,
    StoragePrefixCoverage,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
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
            last_device_node=4,
            mamba_host_hit_length=1,
            page_size=64,
        )

        self.assertTrue(prefix_match.needs_local_restore)
        self.assertEqual(prefix_match.full_restore_token_count, 0)
        self.assertEqual(prefix_match.swa_restore_token_count, 0)
        self.assertEqual(prefix_match.mamba_restore_slot_count, 1)

    def test_storage_coverage_reserves_one_complete_active_window(self) -> None:
        """L2 and L3 SWA state describe one window, not two allocations."""

        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(128),
            l2_host_hit_length=64,
            last_device_node=4,
            storage_coverage=StoragePrefixCoverage(
                prefix_tokens=32,
                full_tokens=32,
                swa_tokens=96,
                mamba_slots=1,
            ),
            swa_host_hit_length=64,
            page_size=32,
        )

        self.assertEqual(prefix_match.decode_prefix_len, 224)
        self.assertEqual(prefix_match.full_restore_token_count, 96)
        self.assertEqual(prefix_match.swa_restore_token_count, 96)
        self.assertEqual(prefix_match.mamba_restore_slot_count, 1)


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

    def test_storage_candidate_excludes_the_swa_reprefill_tail(self) -> None:
        """L3 cannot reclaim the unified-KV tail reserved for recomputation."""

        anchor = SimpleNamespace(
            backuped=True,
            parent=object(),
            get_last_hash_value=Mock(return_value="last"),
            get_prefix_hash_values=Mock(return_value=["prefix"]),
        )
        tree_cache = SimpleNamespace(
            root_node=object(),
            resolve_node_handle=Mock(return_value=anchor),
            swa_reprefill_tail_tokens=Mock(return_value=8),
            hicache_storage_pass_prefix_keys=True,
            query_storage_prefix_coverage=Mock(
                return_value=StoragePrefixCoverage(
                    prefix_tokens=16,
                    full_tokens=16,
                )
            ),
        )
        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)
        mixin.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        mixin.tree_cache = tree_cache
        result = SimpleNamespace(
            device_indices=torch.arange(8),
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            last_device_node=7,
            last_host_node=8,
        )

        prefix_match = mixin._build_decode_prefix_match(
            SimpleNamespace(origin_input_ids=list(range(32))),
            result,
        )

        self.assertEqual(prefix_match.decode_prefix_len, 24)
        query = tree_cache.query_storage_prefix_coverage
        self.assertEqual(query.call_args.args[1], list(range(8, 24)))
        self.assertEqual(query.call_args.kwargs["matched_prefix_tokens"], 8)

    @staticmethod
    def _unified_storage_query_cache(
        component_boundaries: dict[PoolName, int],
    ) -> UnifiedRadixCache:
        cache = object.__new__(UnifiedRadixCache)
        cache.prefetch_threshold = 4
        cache.tree_components = (ComponentType.FULL, ComponentType.SWA)
        cache._sliding_window_size = 8
        cache.sidecar_pool_specs = []
        cache.tree_core = SimpleNamespace(
            is_eagle=False,
            has_swa_host_pool=True,
            enable_storage=True,
            page_size=4,
            prefetch_anchor_info=Mock(return_value="extra"),
        )

        def query(operation: object) -> tuple[list[str], int]:
            operation.pool_storage_result = PoolTransferResult(
                kv_hit_pages=2,
                extra_pool_hit_pages=component_boundaries,
            )
            return ["a", "b"], 8

        cache.cache_controller = SimpleNamespace(
            prefetch_rate_limited=Mock(return_value=False),
            _storage_hit_query=Mock(side_effect=query),
        )
        return cache

    def test_unified_storage_query_reports_full_and_active_swa_coverage(
        self,
    ) -> None:
        """A complete SWA tail produces one structured restore reservation."""

        cache = self._unified_storage_query_cache({PoolName.SWA: 2})

        coverage = cache.query_storage_prefix_coverage(
            9,
            list(range(8)),
            last_hash="previous",
            prefix_keys=["ancestor"],
            matched_prefix_tokens=4,
        )

        self.assertEqual(
            coverage,
            StoragePrefixCoverage(
                prefix_tokens=8,
                full_tokens=8,
                swa_tokens=8,
            ),
        )
        operation = cache.cache_controller._storage_hit_query.call_args.args[0]
        self.assertEqual(len(operation.pool_transfers), 1)
        transfer = operation.pool_transfers[0]
        self.assertEqual(transfer.name, PoolName.SWA)
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(len(transfer.keys), 2)

    def test_missing_unified_swa_boundary_removes_storage_coverage(self) -> None:
        """A Full hit without its required SWA tail is not admissible."""

        cache = self._unified_storage_query_cache({})

        coverage = cache.query_storage_prefix_coverage(
            9,
            list(range(8)),
            matched_prefix_tokens=4,
        )

        self.assertEqual(coverage, StoragePrefixCoverage())

    def test_missing_hi_mamba_boundary_removes_storage_coverage(self) -> None:
        """A Full hit without a trailing recurrent state cannot move the boundary."""

        cache = object.__new__(HiMambaRadixCache)
        cache.enable_storage = True
        cache.prefetch_threshold = 4
        cache.page_size = 4
        cache.is_eagle = False
        cache.cache_controller = SimpleNamespace(
            prefetch_rate_limited=Mock(return_value=False),
        )

        def query(operation: object) -> tuple[list[str], int]:
            operation.pool_storage_result = PoolTransferResult(
                kv_hit_pages=2,
                extra_pool_hit_pages={},
            )
            return ["a", "b"], 8

        cache.cache_controller._storage_hit_query = Mock(side_effect=query)
        anchor = SimpleNamespace(key=SimpleNamespace(extra_key=None))

        coverage = cache.query_storage_prefix_coverage(
            anchor,
            list(range(8)),
            matched_prefix_tokens=4,
        )

        self.assertEqual(coverage, StoragePrefixCoverage())


class TestSwaPrefetchFootprint(unittest.TestCase):
    """Tests storage-to-host SWA allocation at window boundaries."""

    def test_prefetch_allocates_the_bounded_candidate_suffix(self) -> None:
        """Short suffixes combine with L2 state; long suffixes cap at one window."""

        host_pool = SimpleNamespace(
            alloc=Mock(side_effect=lambda count: torch.arange(count)),
        )
        component = object.__new__(SWAComponent)
        component.cache = SimpleNamespace(
            sliding_window_size=4,
            page_size=1,
            evict_host=Mock(),
        )
        component._swa_kv_pool_host = host_pool

        for candidate_tokens, expected_tokens in (
            (0, 0),
            (3, 3),
            (4, 4),
            (5, 4),
        ):
            with self.subTest(candidate_tokens=candidate_tokens):
                host_pool.alloc.reset_mock()
                result = component.prepare_prefetch(
                    9,
                    prefetch_tokens=candidate_tokens,
                )
                if expected_tokens == 0:
                    self.assertIsNone(result.host_indices)
                    host_pool.alloc.assert_not_called()
                    continue
                self.assertEqual(len(result.host_indices), expected_tokens)
                host_pool.alloc.assert_called_once_with(expected_tokens)


class TestPendingRestoreBudgets(unittest.TestCase):
    """Tests rank-local pending full and SWA admission reservations."""

    def test_shared_pool_reserves_the_larger_component_restore(self) -> None:
        """Shared-index allocators do not double-count paired component slots."""

        budget = DecodeRestoreBudget(full_tokens=96, swa_tokens=128)

        self.assertEqual(budget.shared_tokens, 128)

    def test_pending_requests_reserve_each_component_pool(self) -> None:
        """Only pending, not-yet-allocated restores contribute to both budgets."""

        first = DecodePrefixMatch(
            prefix_indices=torch.arange(64),
            l2_host_hit_length=65,
            last_device_node=1,
            swa_host_hit_length=33,
            mamba_host_hit_length=1,
            page_size=32,
        )
        second = DecodePrefixMatch(
            prefix_indices=torch.arange(64),
            l2_host_hit_length=0,
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
                    hicache_load_back_ticket=None,
                ),
                SimpleNamespace(
                    prefix_match=second,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_load_back_ticket=None,
                ),
                SimpleNamespace(
                    prefix_match=ignored,
                    hicache_restore_status=HiCacheRestoreResult.READY,
                    hicache_load_back_ticket=None,
                ),
            ]
        )

        self.assertEqual(
            mixin._hicache_pending_restore_budget(),
            DecodeRestoreBudget(full_tokens=96, swa_tokens=128, mamba_slots=1),
        )


def _make_request(
    rid: str,
    prefix_indices: torch.Tensor,
    *,
    origin_length: int | None = None,
    last_node: int = 17,
    host_hit_length: int = 0,
    swa_host_hit_length: int = 0,
    mamba_host_hit_length: int = 0,
) -> SimpleNamespace:
    """Build the request fields mutated by a transient cache rematch."""

    if origin_length is None:
        origin_length = len(prefix_indices)
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(origin_length)),
        prefix_indices=prefix_indices,
        last_node=last_node,
        last_host_node=18,
        best_match_node=19,
        host_hit_length=host_hit_length,
        swa_host_hit_length=swa_host_hit_length,
        mamba_host_hit_length=mamba_host_hit_length,
        num_matched_prefix_tokens=len(prefix_indices) + host_hit_length,
        mamba_branching_seqlen=None,
        cache_protected_len=len(prefix_indices),
        last_node_lock_params=DecLockRefParams(swa_uuid_for_lock=101),
        swa_uuid_for_lock=101,
        swa_prefix_lock_released=False,
        req_pool_idx=3,
    )


def _make_decode_request(
    prefix_match: DecodePrefixMatch,
    request: SimpleNamespace,
    *,
    ticket: LoadBackTicket | None = None,
) -> SimpleNamespace:
    """Build one decode request owning a local-restore ticket."""

    return SimpleNamespace(
        prefix_match=prefix_match,
        req=request,
        hicache_restore_status=HiCacheRestoreResult.PENDING,
        hicache_load_back_ticket=ticket,
    )


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
            last_device_node=17,
            swa_host_hit_length=7,
            page_size=1,
        )
        request = _make_request("swa-only", prefix_indices, swa_host_hit_length=7)
        match_prefix.return_value = SimpleNamespace(
            best_match_node=31,
            host_hit_length=0,
            device_indices=prefix_indices,
            swa_host_hit_length=7,
            mamba_host_hit_length=0,
        )
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=31,
            queued_components=frozenset({ComponentType.SWA}),
            swa_tokens=7,
        )
        counter = SimpleNamespace(
            producer_index=0,
            num_counters=2,
        )
        tree_cache = SimpleNamespace(
            init_load_back=Mock(return_value=ticket),
            inc_lock_ref=Mock(return_value=IncLockRefResult(delta=0)),
            cache_controller=SimpleNamespace(layer_done_counter=counter),
            is_load_back_event_done=Mock(return_value=True),
            ready_to_load_host_cache=Mock(return_value=1),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = _make_decode_request(prefix_match, request)

        mixin._process_hicache_local_restores([decode_request])

        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.PENDING,
        )
        self.assertIs(decode_request.hicache_load_back_ticket, ticket)
        self.assertEqual(ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(ticket.new_full_device_indices.numel(), 0)
        tree_cache.ready_to_load_host_cache.assert_called_once_with()
        tree_cache.inc_lock_ref.assert_called_once_with(31)


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
        cache._all_reduce = Mock()

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
        cache._all_reduce = Mock()

        self.assertFalse(cache.is_load_back_event_done(0))
        cache.loading_check.assert_not_called()

    def test_tp2_waits_for_the_slowest_rank(self) -> None:
        """One completed rank cannot acknowledge a peer's pending copy."""

        finish_event = Mock()
        finish_event.query.return_value = True
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                events=[SimpleNamespace(finish_event=finish_event)]
            )
        )
        cache.loading_check = Mock()

        def force_remote_pending(value: torch.Tensor, _op: object) -> None:
            value.zero_()

        cache._all_reduce = Mock(side_effect=force_remote_pending)

        self.assertFalse(cache.is_load_back_event_done(0))
        cache.loading_check.assert_not_called()


class TestHierarchicalRestoreCompletion(unittest.TestCase):
    """Tests the shared TP-safe readiness contract on legacy caches."""

    def test_hi_mamba_mamba_only_ticket_does_not_claim_full_copy(self) -> None:
        """Mamba-only restoration reports only the queued recurrent state."""

        node = SimpleNamespace(
            id=31,
            evicted=False,
            mamba_evicted=True,
            mamba_backuped=True,
        )
        cache = object.__new__(HiMambaRadixCache)
        cache.device = torch.device("cpu")
        cache.root_node = SimpleNamespace()
        cache.load_back = Mock(return_value=torch.empty(0, dtype=torch.int64))

        ticket = cache.init_load_back(
            SimpleNamespace(
                best_match_node=node,
                mem_quota=None,
                req=SimpleNamespace(mamba_host_hit_length=1),
            )
        )

        self.assertEqual(
            ticket.queued_components,
            frozenset({ComponentType.MAMBA}),
        )
        self.assertEqual(ticket.full_tokens, 0)
        self.assertEqual(ticket.mamba_slots, 1)
        self.assertEqual(ticket.mamba_device_slots, 1)

    def test_hiradix_waits_for_the_slowest_rank(self) -> None:
        """HiRadix cannot reap a load while a peer remains pending."""

        finish_event = Mock()
        finish_event.query.return_value = True
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                events=[SimpleNamespace(finish_event=finish_event)]
            )
        )
        cache.loading_check = Mock()

        def force_remote_pending(value: torch.Tensor, _op: object) -> None:
            value.zero_()

        cache._all_reduce = Mock(side_effect=force_remote_pending)

        self.assertFalse(cache.is_load_back_event_done(0))
        cache.loading_check.assert_not_called()

    def test_hi_mamba_waits_for_the_slowest_rank(self) -> None:
        """HiMamba uses the same MIN consensus before acknowledgement."""

        finish_event = Mock()
        finish_event.query.return_value = True
        cache = object.__new__(HiMambaRadixCache)
        cache.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(
                events=[SimpleNamespace(finish_event=finish_event)]
            )
        )
        cache.tp_world_size = 2
        cache.tp_group = object()
        cache.loading_check = Mock()

        def force_remote_pending(
            value: torch.Tensor,
            op: object,
            group: object,
        ) -> None:
            self.assertIs(op, torch.distributed.ReduceOp.MIN)
            self.assertIs(group, cache.tp_group)
            value.zero_()

        with patch(
            "torch.distributed.all_reduce",
            side_effect=force_remote_pending,
        ):
            self.assertFalse(cache.is_load_back_event_done(0))

        cache.loading_check.assert_not_called()


class TestRestoreAdmission(unittest.TestCase):
    """Tests component-aware admission and TP consensus."""

    @staticmethod
    def _prefix_match(
        *,
        full_tokens: int = 0,
        swa_tokens: int = 0,
        storage_coverage: StoragePrefixCoverage | None = None,
        last_host_node: int | None = None,
    ) -> DecodePrefixMatch:
        return DecodePrefixMatch(
            prefix_indices=torch.empty(0, dtype=torch.int64),
            l2_host_hit_length=full_tokens,
            last_device_node=1,
            storage_coverage=(
                storage_coverage
                if storage_coverage is not None
                else StoragePrefixCoverage()
            ),
            last_host_node=last_host_node,
            swa_host_hit_length=swa_tokens,
        )

    def test_tp1_admits_independent_full_and_swa_capacity(self) -> None:
        """Each component budget is checked against its own allocator."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tp_size = 1
        mixin.gloo_group = None
        request = SimpleNamespace(rid="tp1-admit")
        prefix_match = self._prefix_match(full_tokens=32, swa_tokens=64)

        admitted, restore_budget = mixin._agree_hicache_admission(
            request,
            prefix_match,
            uses_separate_swa_allocator=True,
            full_required_tokens=64,
            full_allocatable_tokens=96,
            swa_required_tokens=64,
            swa_allocatable_tokens=128,
            mamba_required_slots=0,
            mamba_allocatable_slots=0,
        )

        self.assertTrue(admitted)
        self.assertEqual(
            restore_budget,
            DecodeRestoreBudget(full_tokens=32, swa_tokens=64),
        )

    def test_restore_and_request_share_capacity_equations(self) -> None:
        """Separately fitting reservations cannot overcommit either pool."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tp_size = 1
        mixin.gloo_group = None
        request = SimpleNamespace(rid="tp1-reject")

        full_rejected, _ = mixin._agree_hicache_admission(
            request,
            self._prefix_match(full_tokens=41),
            uses_separate_swa_allocator=True,
            full_required_tokens=60,
            full_allocatable_tokens=100,
            swa_required_tokens=0,
            swa_allocatable_tokens=0,
            mamba_required_slots=0,
            mamba_allocatable_slots=0,
        )
        swa_rejected, _ = mixin._agree_hicache_admission(
            request,
            self._prefix_match(swa_tokens=41),
            uses_separate_swa_allocator=True,
            full_required_tokens=0,
            full_allocatable_tokens=0,
            swa_required_tokens=60,
            swa_allocatable_tokens=100,
            mamba_required_slots=0,
            mamba_allocatable_slots=0,
        )

        self.assertFalse(full_rejected)
        self.assertFalse(swa_rejected)

    def test_mamba_restore_and_request_slots_share_capacity(self) -> None:
        """A tree-state restore cannot consume the request's reserved slot."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tp_size = 1
        mixin.gloo_group = None
        request = SimpleNamespace(rid="mamba-reject")
        prefix_match = self._prefix_match(
            storage_coverage=StoragePrefixCoverage(
                prefix_tokens=4,
                full_tokens=4,
                mamba_slots=1,
            ),
            last_host_node=9,
        )

        admitted, restore_budget = mixin._agree_hicache_admission(
            request,
            prefix_match,
            uses_separate_swa_allocator=True,
            full_required_tokens=0,
            full_allocatable_tokens=4,
            swa_required_tokens=0,
            swa_allocatable_tokens=0,
            mamba_required_slots=3,
            mamba_allocatable_slots=3,
        )

        self.assertFalse(admitted)
        self.assertEqual(
            restore_budget,
            DecodeRestoreBudget(full_tokens=4, mamba_slots=1),
        )

    def test_tp2_rejects_when_one_rank_cannot_admit(self) -> None:
        """A local capacity pass cannot overrule a peer's rejection."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tp_size = 2
        mixin.gloo_group = object()
        request = SimpleNamespace(rid="tp2-reject")
        prefix_match = self._prefix_match(full_tokens=32, swa_tokens=64)

        def reject_remote_rank(
            value: torch.Tensor,
            op: object,
            group: object,
        ) -> None:
            self.assertIs(group, mixin.gloo_group)
            self.assertIs(op, torch.distributed.ReduceOp.MIN)
            value[0] = 0

        with patch(
            "torch.distributed.all_reduce",
            side_effect=reject_remote_rank,
        ) as reduce_mock:
            admitted, _ = mixin._agree_hicache_admission(
                request,
                prefix_match,
                uses_separate_swa_allocator=True,
                full_required_tokens=64,
                full_allocatable_tokens=96,
                swa_required_tokens=64,
                swa_allocatable_tokens=128,
                mamba_required_slots=0,
                mamba_allocatable_slots=0,
            )
        self.assertFalse(admitted)
        reduce_mock.assert_called_once()

    def test_rank_local_prefetch_failure_falls_back_before_allocation(self) -> None:
        """A peer's failed L3 start cannot leave a rank-local protocol length."""

        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tp_size = 2
        mixin.gloo_group = object()
        mixin.tree_cache = SimpleNamespace(release_aborted_request=Mock())
        request = SimpleNamespace(rid="l3-fallback")
        prefix_match = self._prefix_match(
            storage_coverage=StoragePrefixCoverage(
                prefix_tokens=4,
                full_tokens=4,
                swa_tokens=3,
                mamba_slots=1,
            ),
            last_host_node=9,
        )
        prefix_match.prefetch_registered = True
        reduction_count = 0

        def agree_with_failed_peer(
            value: torch.Tensor,
            op: object,
            group: object,
        ) -> None:
            nonlocal reduction_count
            self.assertIs(group, mixin.gloo_group)
            self.assertIs(op, torch.distributed.ReduceOp.MIN)
            reduction_count += 1
            if reduction_count == 1:
                self.assertEqual(
                    value.tolist(),
                    [0, 4, -4, 4, -4, 3, -3, 1, -1, -1],
                )
                value[1] = 0
                value[3] = 0
                value[5] = 0
                value[7] = 0

        with patch(
            "torch.distributed.all_reduce",
            side_effect=agree_with_failed_peer,
        ):
            admitted, restore_budget = mixin._agree_hicache_admission(
                request,
                prefix_match,
                uses_separate_swa_allocator=True,
                full_required_tokens=10,
                full_allocatable_tokens=12,
                swa_required_tokens=0,
                swa_allocatable_tokens=0,
                mamba_required_slots=0,
                mamba_allocatable_slots=0,
            )

        self.assertTrue(admitted)
        self.assertEqual(reduction_count, 2)
        self.assertEqual(prefix_match.storage_coverage, StoragePrefixCoverage())
        self.assertIsNone(prefix_match.last_host_node)
        self.assertFalse(prefix_match.prefetch_registered)
        self.assertEqual(restore_budget, DecodeRestoreBudget())
        mixin.tree_cache.release_aborted_request.assert_called_once_with(request.rid)

    def test_declined_prefetch_removes_storage_coverage(self) -> None:
        """A rate-limited prefetch cannot remain promised to prefill."""

        node = SimpleNamespace(
            get_last_hash_value=Mock(return_value="hash"),
            get_prefix_hash_values=Mock(return_value=["prefix"]),
            parent=object(),
        )
        tree_cache = SimpleNamespace(
            resolve_node_handle=Mock(return_value=node),
            hicache_storage_pass_prefix_keys=True,
            prefetch_from_storage=Mock(),
            ongoing_prefetch={},
        )
        mixin = object.__new__(DecodeHiCachePreallocMixin)
        mixin.tree_cache = tree_cache
        mixin.tp_size = 1
        mixin.gloo_group = None
        request = SimpleNamespace(
            rid="declined-prefetch",
            origin_input_ids=list(range(8)),
        )
        prefix_match = self._prefix_match(
            full_tokens=2,
            storage_coverage=StoragePrefixCoverage(
                prefix_tokens=4,
                full_tokens=4,
            ),
            last_host_node=9,
        )

        mixin._start_hicache_prefetch(request, prefix_match)

        self.assertEqual(prefix_match.storage_coverage, StoragePrefixCoverage())
        self.assertFalse(prefix_match.prefetch_registered)


class TestDecodeRestoreTicketLifecycle(unittest.TestCase):
    """Tests ticket preparation, cleanup, and ownership transfer."""

    def test_merged_tickets_start_once_before_failed_ticket_cleanup(
        self,
    ) -> None:
        """A failed prepared ticket cannot start a partial merged queue."""

        first_prefix = torch.tensor([10, 11], dtype=torch.int64)
        second_prefix = torch.tensor([20, 21], dtype=torch.int64)
        first_match = DecodePrefixMatch(
            prefix_indices=first_prefix,
            l2_host_hit_length=2,
            last_device_node=17,
        )
        second_match = DecodePrefixMatch(
            prefix_indices=second_prefix,
            l2_host_hit_length=2,
            last_device_node=27,
        )
        first_request = _make_request(
            "valid",
            first_prefix,
            origin_length=4,
            host_hit_length=2,
        )
        second_request = _make_request(
            "failed",
            second_prefix,
            origin_length=4,
            last_node=27,
            host_hit_length=2,
        )
        valid_ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([12, 13]),
            restored_node=31,
            queued_components=frozenset({ComponentType.FULL}),
            full_tokens=2,
        )
        failed_ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([22]),
            restored_node=41,
            queued_components=frozenset({ComponentType.FULL}),
            full_tokens=1,
        )
        finish_event = Mock()
        tree_cache = SimpleNamespace(
            init_load_back=Mock(side_effect=[valid_ticket, failed_ticket]),
            inc_lock_ref=Mock(return_value=IncLockRefResult(delta=0)),
            dec_lock_ref=Mock(),
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(
                    producer_index=0,
                    num_counters=2,
                    events=[
                        SimpleNamespace(finish_event=Mock()),
                        SimpleNamespace(finish_event=finish_event),
                    ],
                )
            ),
            is_load_back_event_done=Mock(return_value=True),
            ready_to_load_host_cache=Mock(return_value=1),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        valid_decode_request = _make_decode_request(
            first_match,
            first_request,
        )
        failed_decode_request = _make_decode_request(
            second_match,
            second_request,
        )
        rematches = [
            SimpleNamespace(
                best_match_node=31,
                host_hit_length=2,
                device_indices=first_prefix,
                swa_host_hit_length=0,
                mamba_host_hit_length=0,
            ),
            SimpleNamespace(
                best_match_node=41,
                host_hit_length=2,
                device_indices=second_prefix,
                swa_host_hit_length=0,
                mamba_host_hit_length=0,
            ),
        ]

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            side_effect=rematches,
        ):
            mixin._process_hicache_local_restores(
                [valid_decode_request, failed_decode_request]
            )

        tree_cache.ready_to_load_host_cache.assert_called_once_with()
        self.assertEqual(valid_ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(failed_ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(valid_ticket.consumer_index, 1)
        self.assertEqual(failed_ticket.consumer_index, 1)
        self.assertEqual(
            failed_decode_request.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        tree_cache.dec_lock_ref.assert_not_called()

        mixin._clean_hicache_prefetch_resources(failed_decode_request)

        finish_event.synchronize.assert_called_once_with()
        self.assertEqual(failed_ticket.state, LoadBackTicketState.ABORTED)
        tree_cache.dec_lock_ref.assert_not_called()

    def test_merged_tickets_poll_shared_consumer_once(self) -> None:
        """One merged controller event requires one TP readiness collective."""

        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=1,
            last_device_node=17,
        )
        first_ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([12]),
            restored_node=31,
            queued_components=frozenset({ComponentType.FULL}),
            consumer_index=1,
            state=LoadBackTicketState.STARTED,
        )
        second_ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([13]),
            restored_node=32,
            queued_components=frozenset({ComponentType.FULL}),
            consumer_index=1,
            state=LoadBackTicketState.STARTED,
        )
        first = _make_decode_request(
            prefix_match,
            _make_request("first", prefix_match.prefix_indices),
            ticket=first_ticket,
        )
        second = _make_decode_request(
            prefix_match,
            _make_request("second", prefix_match.prefix_indices),
            ticket=second_ticket,
        )
        tree_cache = SimpleNamespace(
            is_load_back_event_done=Mock(return_value=True),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache

        mixin._process_hicache_local_restores([first, second])

        tree_cache.is_load_back_event_done.assert_called_once_with(1)
        self.assertEqual(first.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(second.hicache_restore_status, HiCacheRestoreResult.READY)

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_noop_ticket_uses_newly_device_resident_suffix(
        self,
        match_prefix: Mock,
    ) -> None:
        """A concurrent restore can satisfy the promise without another DMA."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=2,
            last_device_node=17,
            page_size=1,
        )
        request = _make_request(
            "no-op",
            prefix_indices,
            origin_length=4,
            host_hit_length=2,
        )
        rematched_indices = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        match_prefix.return_value = SimpleNamespace(
            best_match_node=31,
            host_hit_length=0,
            device_indices=rematched_indices,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=31,
        )
        tree_cache = SimpleNamespace(
            init_load_back=Mock(return_value=ticket),
            inc_lock_ref=Mock(return_value=IncLockRefResult(delta=0)),
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(producer_index=0, num_counters=2)
            ),
            is_load_back_event_done=Mock(return_value=True),
            ready_to_load_host_cache=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = _make_decode_request(prefix_match, request)

        mixin._process_hicache_local_restores([decode_request])

        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.PREPARED)
        self.assertEqual(
            ticket.restored_full_device_indices.tolist(),
            [12, 13],
        )
        tree_cache.ready_to_load_host_cache.assert_not_called()

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_synchronous_full_ticket_needs_no_controller_event(
        self,
        match_prefix: Mock,
    ) -> None:
        """Synchronous storage restores report coverage without queued work."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=2,
            last_device_node=17,
        )
        request = _make_request(
            "synchronous-full",
            prefix_indices,
            origin_length=4,
            host_hit_length=2,
        )
        match_prefix.return_value = SimpleNamespace(
            best_match_node=31,
            host_hit_length=2,
            device_indices=prefix_indices,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([12, 13], dtype=torch.int64),
            restored_node=31,
            full_tokens=2,
        )
        tree_cache = SimpleNamespace(
            init_load_back=Mock(return_value=ticket),
            inc_lock_ref=Mock(return_value=IncLockRefResult(delta=0)),
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(producer_index=0, num_counters=2)
            ),
            is_load_back_event_done=Mock(return_value=True),
            ready_to_load_host_cache=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = _make_decode_request(prefix_match, request)

        mixin._process_hicache_local_restores([decode_request])

        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.READY,
        )
        self.assertEqual(ticket.state, LoadBackTicketState.PREPARED)
        self.assertEqual(ticket.restored_full_device_indices.tolist(), [12, 13])
        tree_cache.ready_to_load_host_cache.assert_not_called()

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_noop_ticket_waits_for_same_round_published_indices(
        self,
        match_prefix: Mock,
    ) -> None:
        """A no-copy ticket waits for the merged copy that published its suffix."""

        first_prefix = torch.tensor([10, 11], dtype=torch.int64)
        second_prefix = torch.tensor([20, 21], dtype=torch.int64)
        first_match = DecodePrefixMatch(
            prefix_indices=first_prefix,
            l2_host_hit_length=2,
            last_device_node=17,
        )
        second_match = DecodePrefixMatch(
            prefix_indices=second_prefix,
            l2_host_hit_length=2,
            last_device_node=27,
        )
        first_request = _make_request(
            "publisher",
            first_prefix,
            origin_length=4,
            host_hit_length=2,
        )
        second_request = _make_request(
            "dependent",
            second_prefix,
            origin_length=4,
            last_node=27,
            host_hit_length=2,
        )
        publishing_ticket = LoadBackTicket(
            new_full_device_indices=torch.tensor([12, 13], dtype=torch.int64),
            restored_node=31,
            queued_components=frozenset({ComponentType.FULL}),
            full_tokens=2,
        )
        dependent_ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=41,
        )
        match_prefix.side_effect = [
            SimpleNamespace(
                best_match_node=31,
                host_hit_length=2,
                device_indices=first_prefix,
                swa_host_hit_length=0,
                mamba_host_hit_length=0,
            ),
            SimpleNamespace(
                best_match_node=41,
                host_hit_length=0,
                device_indices=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
                swa_host_hit_length=0,
                mamba_host_hit_length=0,
            ),
        ]
        tree_cache = SimpleNamespace(
            init_load_back=Mock(
                side_effect=[publishing_ticket, dependent_ticket],
            ),
            inc_lock_ref=Mock(return_value=IncLockRefResult(delta=0)),
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(producer_index=0, num_counters=2)
            ),
            is_load_back_event_done=Mock(return_value=True),
            ready_to_load_host_cache=Mock(return_value=1),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        publisher = _make_decode_request(first_match, first_request)
        dependent = _make_decode_request(second_match, second_request)

        mixin._process_hicache_local_restores([publisher, dependent])

        tree_cache.ready_to_load_host_cache.assert_called_once_with()
        self.assertEqual(publishing_ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(dependent_ticket.state, LoadBackTicketState.STARTED)
        self.assertEqual(publishing_ticket.consumer_index, 1)
        self.assertEqual(dependent_ticket.consumer_index, 1)
        self.assertEqual(
            dependent.hicache_restore_status,
            HiCacheRestoreResult.PENDING,
        )
        self.assertEqual(
            dependent_ticket.restored_full_device_indices.tolist(),
            [22, 23],
        )

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_promised_coverage_is_rejected_before_cache_mutation(
        self,
        match_prefix: Mock,
    ) -> None:
        """A shortened rematch cannot enter init_load_back."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=3,
            last_device_node=17,
        )
        request = _make_request(
            "short-rematch",
            prefix_indices,
            origin_length=5,
            host_hit_length=3,
        )
        original_lock_params = request.last_node_lock_params

        def mutate_request(
            _tree_cache: object,
            mutable_request: SimpleNamespace,
            _tokens: list[int],
            *,
            cow_mamba: bool,
            include_req: bool,
        ) -> SimpleNamespace:
            self.assertFalse(cow_mamba)
            self.assertTrue(include_req)
            mutable_request.prefix_indices = torch.tensor([99])
            mutable_request.last_node = 999
            mutable_request.last_node_lock_params = DecLockRefParams(
                swa_uuid_for_lock=999
            )
            mutable_request.host_hit_length = 999
            return SimpleNamespace(
                best_match_node=31,
                host_hit_length=2,
                device_indices=prefix_indices,
                swa_host_hit_length=0,
                mamba_host_hit_length=0,
            )

        match_prefix.side_effect = mutate_request
        tree_cache = SimpleNamespace(init_load_back=Mock())
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = _make_decode_request(prefix_match, request)

        queued = mixin._try_hicache_queue_load_back(decode_request)

        self.assertFalse(queued)
        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertIsNone(decode_request.hicache_load_back_ticket)
        tree_cache.init_load_back.assert_not_called()
        self.assertIs(request.prefix_indices, prefix_indices)
        self.assertEqual(request.last_node, 17)
        self.assertIs(request.last_node_lock_params, original_lock_params)
        self.assertEqual(request.host_hit_length, 3)

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_ticket_cannot_exceed_the_structured_mamba_reservation(
        self,
        match_prefix: Mock,
    ) -> None:
        """Post-rematch allocation demand stays within admitted component slots."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=0,
            last_device_node=17,
            mamba_host_hit_length=1,
        )
        request = _make_request(
            "oversized-mamba-ticket",
            prefix_indices,
            mamba_host_hit_length=1,
        )
        match_prefix.return_value = SimpleNamespace(
            best_match_node=31,
            host_hit_length=0,
            device_indices=prefix_indices,
            swa_host_hit_length=0,
            mamba_host_hit_length=1,
        )
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=31,
            queued_components=frozenset({ComponentType.MAMBA}),
            mamba_slots=2,
            mamba_device_slots=2,
        )
        tree_cache = SimpleNamespace(
            init_load_back=Mock(return_value=ticket),
            inc_lock_ref=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache
        decode_request = _make_decode_request(prefix_match, request)

        queued = mixin._try_hicache_queue_load_back(decode_request)

        self.assertFalse(queued)
        self.assertEqual(
            decode_request.hicache_restore_status,
            HiCacheRestoreResult.FAILED,
        )
        self.assertIs(decode_request.hicache_load_back_ticket, ticket)
        tree_cache.inc_lock_ref.assert_not_called()

    def test_abort_releases_prefetch_and_restored_lock_once(self) -> None:
        """Repeated cleanup is terminal for every ticket-owned resource."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=2,
            last_device_node=17,
            prefetch_registered=True,
        )
        restored_lock_params = DecLockRefParams(
            swa_uuid_for_lock=77,
            skip_lock_node_ids={ComponentType.SWA: {31}},
        )
        ticket = LoadBackTicket(
            new_full_device_indices=torch.empty(0, dtype=torch.int64),
            restored_node=31,
            restored_lock_params=restored_lock_params,
            owns_restored_lock=True,
        )
        request = _make_request("abort", prefix_indices)
        decode_request = _make_decode_request(prefix_match, request, ticket=ticket)
        tree_cache = SimpleNamespace(
            release_aborted_request=Mock(),
            dec_lock_ref=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache

        mixin._clean_hicache_prefetch_resources(decode_request)
        mixin._clean_hicache_prefetch_resources(decode_request)

        tree_cache.release_aborted_request.assert_called_once_with("abort")
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.dec_lock_ref.assert_called_once_with(
            31,
            restored_lock_params,
        )
        self.assertFalse(ticket.owns_restored_lock)
        self.assertEqual(ticket.state, LoadBackTicketState.ABORTED)

    def test_commit_transfers_restored_lock_ownership_to_request(self) -> None:
        """Commit releases only the old lock and makes cleanup a no-op."""

        prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        initial_lock_params = DecLockRefParams(
            swa_uuid_for_lock=41,
            skip_lock_node_ids={ComponentType.FULL: {17}},
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=2,
            last_device_node=17,
            last_device_lock_params=initial_lock_params,
        )
        restored_indices = torch.tensor([12, 13], dtype=torch.int64)
        restored_lock_params = DecLockRefParams(
            swa_uuid_for_lock=77,
            skip_lock_node_ids={ComponentType.SWA: {31}},
        )
        ticket = LoadBackTicket(
            new_full_device_indices=restored_indices.clone(),
            restored_node=31,
            restored_full_device_indices=restored_indices,
            restored_lock_params=restored_lock_params,
            owns_restored_lock=True,
            state=LoadBackTicketState.STARTED,
        )
        request = _make_request(
            "commit",
            prefix_indices,
            origin_length=4,
            host_hit_length=2,
        )
        decode_request = _make_decode_request(prefix_match, request, ticket=ticket)
        decode_request.hicache_restore_status = HiCacheRestoreResult.READY
        req_to_token_pool = SimpleNamespace(write=Mock())
        tree_cache = SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            dec_lock_ref=Mock(),
        )
        mixin = object.__new__(DecodeHiCacheTransferMixin)
        mixin.tree_cache = tree_cache

        mixin._commit_hicache_local_restore_to_req(decode_request)
        mixin._clean_hicache_prefetch_resources(decode_request)

        tree_cache.dec_lock_ref.assert_called_once_with(
            17,
            initial_lock_params,
        )
        req_to_token_pool.write.assert_called_once()
        write_target, write_values = req_to_token_pool.write.call_args.args
        self.assertEqual(write_target[0], 3)
        self.assertEqual(write_target[1], slice(2, 4))
        self.assertTrue(torch.equal(write_values, restored_indices))
        self.assertEqual(
            request.prefix_indices.tolist(),
            [10, 11, 12, 13],
        )
        self.assertEqual(request.last_node, 31)
        self.assertIs(
            request.last_node_lock_params,
            restored_lock_params,
        )
        self.assertEqual(request.swa_uuid_for_lock, 77)

        self.assertFalse(ticket.owns_restored_lock)
        self.assertEqual(ticket.state, LoadBackTicketState.COMMITTED)


class TestUnifiedRequestLockOwnership(unittest.TestCase):
    """Tests exact request-owned lock release and replacement."""

    def test_early_swa_release_and_final_cleanup_share_component_boundary(
        self,
    ) -> None:
        """Final cleanup must not revisit SWA or lower-priority ownership."""

        def component(component_type: ComponentType, priority: int) -> SimpleNamespace:
            return SimpleNamespace(
                component_type=component_type,
                eviction_priority=Mock(return_value=priority),
                release_component_lock=Mock(),
                release_window_lock=Mock(),
            )

        full = component(ComponentType.FULL, 2)
        swa = component(ComponentType.SWA, 1)
        mamba = component(ComponentType.MAMBA, 0)
        node = SimpleNamespace(id=31)
        core = object.__new__(UnifiedTreeCore)
        core.components = (full, swa, mamba)
        core.components_by_type = {
            ComponentType.FULL: full,
            ComponentType.SWA: swa,
            ComponentType.MAMBA: mamba,
        }
        core._node_arena = {node.id: node}
        core._update_evictable_leaf_sets = Mock()
        lock_params = DecLockRefParams(
            swa_uuid_for_lock=77,
            skip_lock_node_ids={ComponentType.MAMBA: {31}},
        )

        core.dec_swa_lock_only(node.id, lock_params)

        full.release_component_lock.assert_not_called()
        swa.release_window_lock.assert_called_once()
        self.assertIs(swa.release_window_lock.call_args.args[1], lock_params)
        mamba.release_component_lock.assert_called_once_with(node, lock_params)

        full.release_component_lock.reset_mock()
        swa.release_window_lock.reset_mock()
        mamba.release_component_lock.reset_mock()

        core.dec_lock_ref(node.id, lock_params, skip_swa=True)

        full.release_component_lock.assert_called_once_with(
            node=node,
            params=lock_params,
        )
        swa.release_component_lock.assert_not_called()
        mamba.release_component_lock.assert_not_called()

    def test_finished_request_releases_exact_committed_lock(self) -> None:
        """Finished cleanup replays component tombstones from acquisition."""

        lock_params = DecLockRefParams(
            swa_uuid_for_lock=77,
            skip_lock_node_ids={ComponentType.SWA: {31}},
        )
        request = SimpleNamespace(
            origin_input_ids=[1, 2],
            output_ids=[],
            req_pool_idx=0,
            cache_protected_len=2,
            last_node=31,
            last_node_lock_params=lock_params,
            swa_prefix_lock_released=False,
        )
        cache = object.__new__(UnifiedRadixCache)
        cache.session = SimpleNamespace(try_cache_finished_req=Mock(return_value=False))
        cache.disable = False
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11]], dtype=torch.int64)
        )
        cache.token_to_kv_pool_allocator = SimpleNamespace(free=Mock())
        cache._components_tuple = ()
        cache.dec_lock_ref = Mock()

        cache.cache_finished_req(
            request,
            is_insert=False,
            kv_len_to_handle=2,
        )

        cache.dec_lock_ref.assert_called_once_with(
            31,
            lock_params,
            skip_swa=False,
        )
        self.assertIsNone(request.last_node_lock_params)

    def test_unfinished_request_replaces_exact_lock_params(self) -> None:
        """Chunk cleanup transfers ownership to the rematched node lock."""

        old_lock_params = DecLockRefParams(
            swa_uuid_for_lock=77,
            skip_lock_node_ids={ComponentType.SWA: {31}},
        )
        new_lock_result = IncLockRefResult(
            swa_uuid_for_lock=88,
            skip_lock_node_ids={ComponentType.FULL: {41}},
        )
        request = SimpleNamespace(
            get_fill_ids=Mock(return_value=[1, 2]),
            req_pool_idx=0,
            cache_protected_len=2,
            extra_key=None,
            priority=0,
            prefix_indices=torch.tensor([10, 11]),
            last_node=31,
            last_node_lock_params=old_lock_params,
            swa_uuid_for_lock=77,
            swa_prefix_lock_released=False,
        )
        cache = object.__new__(UnifiedRadixCache)
        cache.session = SimpleNamespace(
            try_cache_unfinished_req=Mock(return_value=False)
        )
        cache.disable = False
        cache.tree_core = SimpleNamespace(is_eagle=False)
        cache.page_size = 1
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11]], dtype=torch.int64),
            write=Mock(),
        )
        cache.token_to_kv_pool_allocator = SimpleNamespace()
        cache._components_tuple = ()
        cache.insert = Mock(return_value=SimpleNamespace(prefix_len=2))
        cache.match_prefix = Mock(
            return_value=SimpleNamespace(
                device_indices=torch.tensor([10, 11]),
                last_device_node=41,
            )
        )
        cache.dec_lock_ref = Mock()
        cache.inc_lock_ref = Mock(return_value=new_lock_result)

        cache.cache_unfinished_req(request)

        cache.dec_lock_ref.assert_called_once_with(31, old_lock_params)
        self.assertEqual(request.last_node, 41)
        self.assertEqual(
            request.last_node_lock_params,
            new_lock_result.to_dec_params(),
        )
        self.assertEqual(request.swa_uuid_for_lock, 88)


class TestUnifiedLoadBackConsensus(unittest.TestCase):
    """Tests rank-consistent allocation and cancellation."""

    def test_empty_component_specs_do_not_queue_controller_work(self) -> None:
        """An empty component dictionary cannot strand an unstarted load."""

        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.empty(0, dtype=torch.int64),
        )
        tree_core = SimpleNamespace(
            build_load_back_spec=Mock(
                return_value=(kv_transfer, {ComponentType.SWA: []})
            ),
        )
        controller = SimpleNamespace(load=Mock())
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = tree_core
        cache.cache_controller = controller
        cache.load_back_threshold = 0
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache._admit_load_back = Mock()
        cache.dec_lock_ref = Mock()
        cache.dec_host_lock_ref = Mock()
        cache._all_reduce = Mock()
        ancestor_lock_params = DecLockRefParams(swa_uuid_for_lock=41)
        host_anchor_params = DecLockRefParams(swa_uuid_for_host_lock=42)

        queued = cache._load_back_transfers(
            node_id=31,
            mem_quota=None,
            req=SimpleNamespace(),
            result=IncLockRefResult(delta=0),
            ancestor_lock_params=ancestor_lock_params,
            host_anchor_params=host_anchor_params,
        )

        self.assertIsNone(queued)
        cache._admit_load_back.assert_not_called()
        controller.load.assert_not_called()
        cache.dec_lock_ref.assert_called_once_with(31, ancestor_lock_params)
        cache.dec_host_lock_ref.assert_called_once_with(31, host_anchor_params)

    def test_local_allocation_is_cancelled_when_a_peer_fails(self) -> None:
        """No rank publishes a tree mutation after TP allocation disagreement."""

        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([1, 2], dtype=torch.int64),
        )
        local_device_indices = torch.tensor([21, 22], dtype=torch.int64)
        tree_core = SimpleNamespace(
            build_load_back_spec=Mock(return_value=(kv_transfer, {})),
            commit_load_back=Mock(),
        )
        controller = SimpleNamespace(
            load=Mock(return_value=local_device_indices),
            cancel_pending_load=Mock(),
        )
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = tree_core
        cache.cache_controller = controller
        cache.load_back_threshold = 0
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache._admit_load_back = Mock(return_value=True)
        cache.dec_lock_ref = Mock()
        cache.dec_host_lock_ref = Mock()

        reduce_calls: int = 0

        def fail_second_rank(value: torch.Tensor, _op: object) -> None:
            nonlocal reduce_calls
            reduce_calls += 1
            if reduce_calls == 2:
                value.zero_()

        cache._all_reduce = Mock(side_effect=fail_second_rank)
        ancestor_lock_params = DecLockRefParams(swa_uuid_for_lock=41)
        host_anchor_params = DecLockRefParams(swa_uuid_for_host_lock=42)

        queued = cache._load_back_transfers(
            node_id=31,
            mem_quota=None,
            req=SimpleNamespace(),
            result=IncLockRefResult(delta=0),
            ancestor_lock_params=ancestor_lock_params,
            host_anchor_params=host_anchor_params,
        )

        self.assertIsNone(queued)
        controller.load.assert_called_once()
        controller.cancel_pending_load.assert_called_once()
        self.assertIs(
            controller.cancel_pending_load.call_args.args[0],
            local_device_indices,
        )
        cache.dec_lock_ref.assert_called_once_with(31, ancestor_lock_params)
        cache.dec_host_lock_ref.assert_called_once_with(31, host_anchor_params)
        tree_core.commit_load_back.assert_not_called()
        self.assertEqual(reduce_calls, 2)


class TestHybridControllerRollback(unittest.TestCase):
    """Tests atomic rollback of component-local device allocations."""

    def test_derived_transfers_are_cleared_when_a_later_source_is_missing(
        self,
    ) -> None:
        """A failed transfer graph restores every descriptor to unbound state."""

        swa_indices = torch.tensor([30], dtype=torch.int64)
        swa_allocator = SimpleNamespace(
            alloc=Mock(return_value=swa_indices),
            free=Mock(),
        )
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.SWA: SimpleNamespace(
                    device_alloc_fn=swa_allocator.alloc,
                    device_free_fn=swa_allocator.free,
                    device_evict_fn=None,
                    device_pool=None,
                )
            }
        )
        source = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )
        resolved_derived = PoolTransfer(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.SWA,
        )
        missing_derived = PoolTransfer(
            name=PoolName.DRAFT,
            indices_from_pool=PoolName.MAMBA,
        )

        result = controller._resolve_pool_transfers_allocation(
            [source, resolved_derived, missing_derived],
            alloc_host=False,
        )

        self.assertIsNone(result)
        swa_allocator.free.assert_called_once_with(swa_indices)
        self.assertIsNone(source.device_indices)
        self.assertIsNone(resolved_derived.device_indices)
        self.assertIsNone(missing_derived.device_indices)

    def test_partial_component_allocation_rolls_back_full_and_swa(self) -> None:
        """A later component failure returns every earlier allocation."""

        full_indices = torch.tensor([20, 21], dtype=torch.int64)
        swa_indices = torch.tensor([30], dtype=torch.int64)
        full_allocator = SimpleNamespace(
            alloc=Mock(return_value=full_indices),
            free=Mock(),
        )
        swa_allocator = SimpleNamespace(
            alloc=Mock(return_value=swa_indices),
            free=Mock(),
        )
        mamba_allocator = SimpleNamespace(
            alloc=Mock(return_value=None),
            free=Mock(),
        )
        controller = object.__new__(HybridCacheController)
        controller.device = torch.device("cpu")
        controller.mem_pool_device_allocator = SimpleNamespace(
            full_attn_allocator=full_allocator
        )
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.SWA: SimpleNamespace(
                    device_alloc_fn=swa_allocator.alloc,
                    device_free_fn=swa_allocator.free,
                    device_evict_fn=None,
                    device_pool=None,
                ),
                PoolName.MAMBA: SimpleNamespace(
                    device_alloc_fn=mamba_allocator.alloc,
                    device_free_fn=mamba_allocator.free,
                    device_evict_fn=None,
                    device_pool=None,
                ),
            }
        )
        controller.load_queue = []
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2], dtype=torch.int64),
        )

        result = controller.load(
            host_indices=torch.tensor([3, 4], dtype=torch.int64),
            node_id=31,
            extra_pools=[swa_transfer, mamba_transfer],
        )

        self.assertIsNone(result)
        full_allocator.free.assert_called_once()
        self.assertIs(full_allocator.free.call_args.args[0], full_indices)
        swa_allocator.free.assert_called_once()
        self.assertIs(swa_allocator.free.call_args.args[0], swa_indices)
        mamba_allocator.free.assert_not_called()
        self.assertIsNone(swa_transfer.device_indices)
        self.assertIsNone(mamba_transfer.device_indices)
        self.assertEqual(controller.load_queue, [])

    def test_cancel_pending_load_frees_full_and_component_slots(self) -> None:
        """A TP peer failure can cancel a locally successful preparation."""

        full_indices = torch.tensor([20, 21], dtype=torch.int64)
        swa_indices = torch.tensor([30], dtype=torch.int64)
        full_allocator = SimpleNamespace(
            alloc=Mock(return_value=full_indices),
            free=Mock(),
        )
        swa_allocator = SimpleNamespace(
            alloc=Mock(return_value=swa_indices),
            free=Mock(),
        )
        controller = object.__new__(HybridCacheController)
        controller.device = torch.device("cpu")
        controller.mem_pool_device_allocator = SimpleNamespace(
            full_attn_allocator=full_allocator
        )
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.SWA: SimpleNamespace(
                    device_alloc_fn=swa_allocator.alloc,
                    device_free_fn=swa_allocator.free,
                    device_evict_fn=None,
                    device_pool=None,
                )
            }
        )
        controller.load_queue = []
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([1], dtype=torch.int64),
        )

        result = controller.load(
            host_indices=torch.tensor([3, 4], dtype=torch.int64),
            node_id=31,
            extra_pools=[swa_transfer],
        )

        self.assertIs(result, full_indices)
        self.assertEqual(len(controller.load_queue), 1)
        self.assertIs(swa_transfer.device_indices, swa_indices)

        controller.cancel_pending_load(result)

        self.assertEqual(controller.load_queue, [])
        full_allocator.free.assert_called_once()
        self.assertIs(full_allocator.free.call_args.args[0], full_indices)
        swa_allocator.free.assert_called_once()
        self.assertIs(swa_allocator.free.call_args.args[0], swa_indices)


if __name__ == "__main__":
    unittest.main()
