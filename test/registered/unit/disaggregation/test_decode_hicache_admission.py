import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodePrefixMatch,
    DecodeRestoreBudget,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_swa_allocator(
    *,
    full_available: int,
    swa_available: int,
    page_size: int,
) -> SWATokenToKVPoolAllocator:
    """Build a CPU-free SWA allocator capacity fixture.

    :param full_available: Reported full-attention token availability.
    :param swa_available: Reported sliding-window token availability.
    :param page_size: Physical allocator page size.
    :returns: Minimal composite allocator fixture.
    """

    allocator = object.__new__(SWATokenToKVPoolAllocator)
    allocator.full_attn_allocator = SimpleNamespace(
        available_size=lambda: full_available
    )
    allocator.swa_attn_allocator = SimpleNamespace(available_size=lambda: swa_available)
    allocator.page_size = page_size
    allocator._size_full = full_available
    allocator._size_swa = swa_available
    return allocator


def _make_budget_queue(
    *,
    full_available: int,
    swa_available: int,
    page_size: int,
    uses_swa_tail_prealloc: bool,
) -> DecodePreallocQueue:
    """Build the minimal queue state needed by capacity accounting.

    :param full_available: Reported full-attention token availability.
    :param swa_available: Reported sliding-window token availability.
    :param page_size: Physical allocator page size.
    :param uses_swa_tail_prealloc: Whether admission uses separate tail demand.
    :returns: Decode preallocation queue fixture.
    """

    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.token_to_kv_pool_allocator = _make_swa_allocator(
        full_available=full_available,
        swa_available=swa_available,
        page_size=page_size,
    )
    queue.token_to_kv_pool = object()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=uses_swa_tail_prealloc)
    queue.scheduler = SimpleNamespace(
        enable_hisparse=False,
        last_batch=None,
        running_batch=SimpleNamespace(reqs=[]),
        server_args=SimpleNamespace(
            disable_radix_cache=False,
            disaggregation_decode_enable_radix_cache=False,
        ),
        sliding_window_size=1024,
        waiting_queue=[],
    )
    queue.transfer_queue = SimpleNamespace(queue=[])
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = 0
    return queue


class TestDecodePrefixRestoreIntent(unittest.TestCase):
    def test_unified_hicache_without_storage_has_no_l3_hit(self) -> None:
        root_node = object()
        tree_cache = object.__new__(UnifiedRadixCache)
        tree_cache.tree_core = SimpleNamespace(
            enable_storage=False,
            root_node=root_node,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(enable_decode_hicache=True)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=16)
        queue.tree_cache = tree_cache
        result = SimpleNamespace(
            device_indices=torch.arange(8, dtype=torch.int64),
            host_hit_length=17,
            swa_host_hit_length=33,
            mamba_host_hit_length=0,
            last_device_node=object(),
            last_host_node=root_node,
        )
        req = SimpleNamespace(origin_input_ids=list(range(64)))

        prefix_match = queue._build_decode_prefix_match(req, result)

        self.assertEqual(prefix_match.l2_host_hit_length, 17)
        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertIsNone(prefix_match.last_host_node)

    def test_prefix_match_carries_component_hits_and_allocator_page_size(self) -> None:
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(enable_decode_hicache=False)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=16)
        result = SimpleNamespace(
            device_indices=torch.arange(8, dtype=torch.int64),
            host_hit_length=17,
            swa_host_hit_length=33,
            mamba_host_hit_length=1,
            last_device_node=object(),
        )

        prefix_match = queue._build_decode_prefix_match(object(), result)

        self.assertEqual(prefix_match.l2_host_hit_length, 17)
        self.assertEqual(prefix_match.swa_host_hit_length, 33)
        self.assertEqual(prefix_match.mamba_host_hit_length, 1)
        self.assertEqual(prefix_match.page_size, 16)
        self.assertEqual(prefix_match.restore_budget.mamba_slots, 2)

    def test_swa_host_hit_requires_restore_when_full_prefix_is_on_device(self) -> None:
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(64, dtype=torch.int64),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=object(),
            swa_host_hit_length=17,
            page_size=16,
        )

        self.assertEqual(prefix_match.decode_prefix_len, 64)
        self.assertEqual(prefix_match.full_restore_token_count, 0)
        self.assertEqual(prefix_match.swa_restore_token_count, 32)
        self.assertTrue(prefix_match.needs_local_restore)

    def test_restore_demand_is_page_rounded_by_component(self) -> None:
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(8, dtype=torch.int64),
            l2_host_hit_length=17,
            l3_storage_hit_length=1,
            last_device_node=object(),
            swa_host_hit_length=33,
            page_size=16,
        )

        self.assertEqual(prefix_match.full_restore_token_count, 32)
        self.assertEqual(prefix_match.swa_restore_token_count, 48)


class TestDecodeRestoreAdmission(unittest.TestCase):
    def test_pending_budget_counts_only_unallocated_restores(self) -> None:
        pending = DecodePrefixMatch(
            prefix_indices=torch.arange(8, dtype=torch.int64),
            l2_host_hit_length=17,
            l3_storage_hit_length=0,
            last_device_node=object(),
            swa_host_hit_length=17,
            page_size=16,
        )
        swa_only = DecodePrefixMatch(
            prefix_indices=torch.arange(8, dtype=torch.int64),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=object(),
            swa_host_hit_length=1,
            page_size=16,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(enable_decode_hicache=True)
        queue.transfer_queue = SimpleNamespace(
            queue=[
                SimpleNamespace(
                    prefix_match=pending,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_load_back_ticket=None,
                ),
                SimpleNamespace(
                    prefix_match=swa_only,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_load_back_ticket=None,
                ),
                SimpleNamespace(
                    prefix_match=pending,
                    hicache_restore_status=HiCacheRestoreResult.PENDING,
                    hicache_load_back_ticket=object(),
                ),
                SimpleNamespace(
                    prefix_match=pending,
                    hicache_restore_status=HiCacheRestoreResult.READY,
                    hicache_load_back_ticket=None,
                ),
            ]
        )

        self.assertEqual(
            queue._hicache_pending_restore_budgets(),
            DecodeRestoreBudget(full_tokens=32, swa_tokens=48),
        )

    def test_page_one_capacity_subtracts_debt_before_joint_constraint(self) -> None:
        cases = (
            (100, 60, DecodeRestoreBudget(full_tokens=40), 60),
            (60, 100, DecodeRestoreBudget(swa_tokens=40), 60),
        )
        for full_available, swa_available, restore_budget, expected in cases:
            with self.subTest(restore_budget=restore_budget):
                queue = _make_budget_queue(
                    full_available=full_available,
                    swa_available=swa_available,
                    page_size=1,
                    uses_swa_tail_prealloc=False,
                )

                actual = queue._allocatable_token_budgets(
                    count_retracted=False,
                    reserved_tokens=0,
                    hicache_restore_budget=restore_budget,
                )

                self.assertEqual(actual, expected)

    def test_paged_tail_capacity_keeps_restore_components_independent(self) -> None:
        queue = _make_budget_queue(
            full_available=128,
            swa_available=96,
            page_size=16,
            uses_swa_tail_prealloc=True,
        )

        full_debt = queue._swa_aware_allocatable_token_budgets(
            count_retracted=False,
            hicache_restore_budget=DecodeRestoreBudget(full_tokens=32),
        )
        swa_debt = queue._swa_aware_allocatable_token_budgets(
            count_retracted=False,
            hicache_restore_budget=DecodeRestoreBudget(swa_tokens=48),
        )

        self.assertEqual(full_debt, (96, 96))
        self.assertEqual(swa_debt, (128, 48))

    def test_ordinary_preallocation_receives_pending_component_budget(self) -> None:
        restore_budget = DecodeRestoreBudget(full_tokens=32, swa_tokens=48)
        for uses_swa_tail_prealloc in (False, True):
            with self.subTest(uses_swa_tail_prealloc=uses_swa_tail_prealloc):
                queue = _make_budget_queue(
                    full_available=128,
                    swa_available=128,
                    page_size=16 if uses_swa_tail_prealloc else 1,
                    uses_swa_tail_prealloc=uses_swa_tail_prealloc,
                )
                queue.queue = []
                queue.pending_reqs = []
                queue._resolve_pending_reqs = MagicMock()
                queue._update_handshake_waiters = MagicMock()
                queue._hicache_pending_restore_budgets = MagicMock(
                    return_value=restore_budget
                )
                queue._allocatable_token_budgets = MagicMock(return_value=128)
                queue._swa_aware_allocatable_token_budgets = MagicMock(
                    return_value=(128, 128)
                )
                queue.scheduler.enable_priority_scheduling = False

                self.assertEqual(queue.pop_preallocated(), ([], []))

                if uses_swa_tail_prealloc:
                    queue._swa_aware_allocatable_token_budgets.assert_called_once_with(
                        retractable_tokens=0,
                        retractable_swa_tokens=0,
                        count_retracted=True,
                        hicache_restore_budget=restore_budget,
                    )
                    continue
                queue._allocatable_token_budgets.assert_called_once_with(
                    retractable_tokens=0,
                    count_retracted=True,
                    hicache_restore_budget=restore_budget,
                )

    def test_swa_only_match_is_reserved_before_transport_preallocation(self) -> None:
        queue = _make_budget_queue(
            full_available=100,
            swa_available=3,
            page_size=1,
            uses_swa_tail_prealloc=False,
        )
        last_device_node = object()
        request = SimpleNamespace(
            rid="swa-only-restore",
            origin_input_ids=list(range(8)),
            output_ids=[1],
            last_node=last_device_node,
            last_node_lock_params=DecLockRefParams(swa_uuid_for_lock=17),
            finished_reason=None,
            cache_protected_len=0,
            sampling_params=SimpleNamespace(max_new_tokens=0),
        )
        decode_req = SimpleNamespace(
            req=request,
            waiting_for_input=True,
            is_rebootstrap=False,
        )
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(
            return_value=DecodePrefixMatch(
                prefix_indices=torch.arange(8, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=0,
                last_device_node=last_device_node,
                swa_host_hit_length=4,
                page_size=1,
                last_device_lock_params=request.last_node_lock_params,
            )
        )
        queue._pre_alloc = MagicMock(
            side_effect=AssertionError("transport allocation must not start")
        )
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )
        queue.tree_cache = MagicMock()
        queue.tree_cache.evictable_size.return_value = 0
        queue.scheduler.enable_decode_hicache = True
        queue.scheduler.enable_priority_scheduling = False
        queue.scheduler.enable_hisparse = False
        queue.scheduler.server_args.disaggregation_decode_enable_radix_cache = True

        self.assertEqual(queue.pop_preallocated(), ([], []))

        queue._pre_alloc.assert_not_called()
        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            last_device_node,
            DecLockRefParams(swa_uuid_for_lock=17),
        )

    def test_reservation_admission_uses_asymmetric_page_one_capacity(self) -> None:
        cases = (
            (100, 60, DecodeRestoreBudget(full_tokens=40), False),
            (60, 100, DecodeRestoreBudget(swa_tokens=40), False),
            (100, 60, DecodeRestoreBudget(swa_tokens=32), True),
        )
        request = SimpleNamespace(
            origin_input_ids=list(range(48)),
            output_ids=[1],
        )
        for full_available, swa_available, restore_budget, should_refuse in cases:
            with self.subTest(
                restore_budget=restore_budget,
                should_refuse=should_refuse,
            ):
                queue = _make_budget_queue(
                    full_available=full_available,
                    swa_available=swa_available,
                    page_size=1,
                    uses_swa_tail_prealloc=False,
                )
                queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
                queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
                    available_size=lambda: 1
                )
                queue.max_total_num_tokens = 1024
                queue._unpublished_preallocated_child_count = MagicMock(return_value=0)
                queue._hicache_pending_restore_budgets = MagicMock(
                    return_value=restore_budget
                )

                if should_refuse:
                    with self.assertRaises(DecodeReservationAdmissionRefused) as ctx:
                        queue._validate_preallocated_capacity((request,))
                    self.assertEqual(ctx.exception.reason_code, "decode_kv_capacity")
                    continue
                queue._validate_preallocated_capacity((request,))


if __name__ == "__main__":
    unittest.main()
