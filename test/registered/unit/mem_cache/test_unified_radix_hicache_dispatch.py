import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.base_prefix_cache import (
    IncLockRefResult,
    InitLoadBackParams,
    LoadBackResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    StackBuildResult,
    StackStrategy,
    _apply_stack_result,
    _DeepSeekV4Strategy,
    _DsaStrategy,
    _MambaStrategy,
    _MiniMaxSparseStrategy,
    _PlainKvStrategy,
    _select_strategy,
    _SwaStrategy,
    register_stack_strategy,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingLoadBack,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_kvcache(cls):
    return MagicMock(spec=cls)


FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA


class TestUnifiedRadixHiCacheDispatch(unittest.TestCase):
    def test_strategy_registry_ordering(self):
        order = [type(s) for s in _STRATEGIES]
        # DeepSeekV4 inherits from SWAKVPool, so it must resolve before _SwaStrategy.
        self.assertLess(order.index(_DeepSeekV4Strategy), order.index(_SwaStrategy))
        self.assertLess(
            order.index(_MiniMaxSparseStrategy), order.index(_PlainKvStrategy)
        )
        self.assertEqual(order[-1], _PlainKvStrategy)

    def test_deepseek_v4_full_swa(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = _mock_kvcache(DeepSeekV4TokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _DeepSeekV4Strategy)

    def test_mamba(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = _mock_kvcache(HybridLinearKVPool)
        strategy = _select_strategy(kvcache, {FULL, MAMBA})
        self.assertIsInstance(strategy, _MambaStrategy)

    def test_swa(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = _mock_kvcache(SWAKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _SwaStrategy)

    def test_dsa(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = _mock_kvcache(DSATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _DsaStrategy)

    def test_minimax_sparse(self):
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        kvcache = _mock_kvcache(MiniMaxSparseKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _MiniMaxSparseStrategy)

    def test_minimax_sparse_build_registers_indexer_sidecar(self):
        strategy = _MiniMaxSparseStrategy()
        host_pool_group = MagicMock()
        kv_host_pool = object()
        host_pool_group.get_pool.return_value = kv_host_pool
        cache_controller = MagicMock()
        cache = MagicMock(page_size=4)
        kvcache = MagicMock()
        kvcache.index_k_pool = object()
        kvcache.main_pool.layer_num = 8
        params = MagicMock()
        params.tp_cache_group = None
        params.pp_rank = 0
        params.pp_size = 1
        server_args = MagicMock()

        with patch.object(
            hybrid_pool_assembler,
            "build_minimax_sparse_hicache_stack",
            return_value=(host_pool_group, cache_controller),
        ) as build_stack:
            result = strategy.build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=server_args,
                load_cache_event=object(),
            )

        build_stack.assert_called_once()
        self.assertIs(build_stack.call_args.kwargs["sparse_pool"], kvcache)
        self.assertIs(result.host_pool_group, host_pool_group)
        self.assertIs(result.cache_controller, cache_controller)
        self.assertIs(result.component_host_pools[FULL], kv_host_pool)
        self.assertEqual(result.pools_desc, "KV + INDEXER(k-only)")
        self.assertEqual(result.transfer_layer_num, 8)
        self.assertEqual(len(result.sidecars), 1)
        self.assertEqual(result.sidecars[0].pool_name, PoolName.INDEXER)
        self.assertEqual(result.sidecars[0].indices_from_pool, PoolName.KV)

    def test_plain_kv_fallback(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kvcache = _mock_kvcache(MHATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_mla_routes_to_plain(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        kvcache = _mock_kvcache(MLATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_unknown_combo_raises(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        for cls in (SWAKVPool, DeepSeekV4TokenToKVPool):
            kvcache = _mock_kvcache(cls)
            with self.assertRaises(AssertionError) as cm:
                _select_strategy(kvcache, {FULL})
            self.assertIn("No matching HiCache strategy", str(cm.exception))

    def test_register_custom_strategy_takes_precedence(self):
        class _CustomStrategy(StackStrategy):
            def matches(self, kvcache, components):
                return components == {FULL}

            def build(self, **_):
                raise NotImplementedError

        custom = _CustomStrategy()
        original = list(hybrid_pool_assembler._STRATEGIES)
        try:
            register_stack_strategy(custom)
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            kvcache = _mock_kvcache(MHATokenToKVPool)
            self.assertIs(_select_strategy(kvcache, {FULL}), custom)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = original


class TestApplyStackResult(unittest.TestCase):
    @staticmethod
    def _fake_cache(component_types):
        cache = MagicMock()
        cache.components = {ct: MagicMock() for ct in component_types}
        return cache

    def test_wires_components_sidecars_and_counters(self):
        full_host, swa_host, mamba_host = MagicMock(), MagicMock(), MagicMock()
        cache = self._fake_cache([FULL, SWA, MAMBA])
        kvcache = MagicMock()
        params = MagicMock()
        controller = MagicMock()
        sidecar = SidecarPoolSpec(
            pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV
        )
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=controller,
            component_host_pools={FULL: full_host, SWA: swa_host, MAMBA: mamba_host},
            sidecars=[sidecar],
            register_req_to_token_counter=True,
            transfer_layer_num=8,
            pools_desc="KV + SWA + MAMBA",
        )

        _apply_stack_result(cache, kvcache, params, result)

        self.assertIs(cache.host_pool_group, result.host_pool_group)
        self.assertIs(cache.cache_controller, controller)
        self.assertIs(cache.full_kv_pool_host, full_host)
        self.assertIs(cache.swa_kv_pool_host, swa_host)
        self.assertIs(cache.mamba_pool_host, mamba_host)
        self.assertIs(cache.components[FULL]._full_kv_pool_host, full_host)
        self.assertIs(cache.components[SWA]._swa_kv_pool_host, swa_host)
        self.assertIs(cache.components[MAMBA]._mamba_pool_host, mamba_host)
        cache.register_sidecar_pool.assert_called_once_with(sidecar)
        kvcache.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )
        params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )

    def test_skips_req_to_token_counter_when_flag_false(self):
        cache = self._fake_cache([FULL])
        kvcache = MagicMock()
        params = MagicMock()
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=MagicMock(),
            component_host_pools={FULL: MagicMock()},
            sidecars=[],
            register_req_to_token_counter=False,
            transfer_layer_num=1,
            pools_desc="KV",
        )

        _apply_stack_result(cache, kvcache, params, result)

        kvcache.register_layer_transfer_counter.assert_called_once()
        params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        cache.register_sidecar_pool.assert_not_called()


class _FullSWACompletionEvent:
    """Model the merged stream event owned by the hybrid cache controller."""

    full_complete: bool
    swa_complete: bool
    query_count: int
    synchronize_count: int

    def __init__(self) -> None:
        """Initialize an incomplete full-plus-SWA transfer event."""

        self.full_complete = False
        self.swa_complete = False
        self.query_count = 0
        self.synchronize_count = 0

    def query(self) -> bool:
        """Return whether both component copies have completed.

        :returns: Whether the merged controller event is signaled.
        """

        self.query_count += 1
        return self.full_complete and self.swa_complete

    def synchronize(self) -> None:
        """Acknowledge a signaled merged controller event."""

        if not self.query():
            raise AssertionError("cannot acknowledge an incomplete load-back event")
        self.synchronize_count += 1


class TestUnifiedLoadBackCompletion(unittest.TestCase):
    @staticmethod
    def _make_cache(
        finish_event: _FullSWACompletionEvent, node_id: int
    ) -> SimpleNamespace:
        """Build a cache double that runs the production reaping path.

        :param finish_event: Rank-local merged component event.
        :param node_id: Ongoing load-back operation identifier.
        :returns: Unified cache completion double.
        """

        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(
                    events=[SimpleNamespace(finish_event=finish_event)]
                ),
                ack_load_queue=[
                    HiCacheAck(
                        start_event=finish_event,
                        finish_event=finish_event,
                        node_ids=[node_id],
                    )
                ],
            ),
            ongoing_load_back={node_id: _OngoingLoadBack(node_id, object(), object())},
            dec_lock_ref=MagicMock(),
            dec_host_lock_ref=MagicMock(),
            validate_host_lock_ref=MagicMock(),
            buffer_pipeline=None,
            metrics_collector=None,
            pp_rank=0,
            tree_core=SimpleNamespace(
                finish_load_back=MagicMock(),
                write_back_duplicate_reclaim_digest=0,
            ),
            _all_reduce=MagicMock(),
        )
        cache._count_ready_acks = lambda queue: UnifiedRadixCache._count_ready_acks(
            cache, queue
        )
        cache.loading_check = MagicMock(
            side_effect=lambda: UnifiedRadixCache.loading_check(cache)
        )
        return cache

    @staticmethod
    def _make_decode_req() -> SimpleNamespace:
        """Build a request whose network KV transfer is complete.

        :returns: Pending local-restore request double.
        """

        receiver = MagicMock()
        receiver.poll.return_value = KVPoll.Success
        return SimpleNamespace(
            kv_receiver=receiver,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )

    @staticmethod
    def _poll_all(
        caches: list[SimpleNamespace], decode_reqs: list[SimpleNamespace]
    ) -> list[KVPoll]:
        """Advance local completion and poll each rank-local receiver gate.

        :param caches: Rank-local unified cache doubles.
        :param decode_reqs: Rank-local decode request doubles.
        :returns: Rank-local receiver poll states.
        """

        polls = []
        for cache, decode_req in zip(caches, decode_reqs, strict=True):
            if (
                decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
                and UnifiedRadixCache.is_load_back_event_done(cache, 0)
            ):
                decode_req.hicache_restore_status = HiCacheRestoreResult.READY
            polls.append(HiCacheRestoreGatedKVReceiver(decode_req).poll())
        return polls

    def test_negative_ticket_is_complete_without_reaping(self) -> None:
        event = _FullSWACompletionEvent()
        cache = self._make_cache(event, node_id=7)

        self.assertTrue(UnifiedRadixCache.is_load_back_event_done(cache, -1))
        self.assertEqual(event.query_count, 0)
        cache.loading_check.assert_not_called()
        self.assertEqual(len(cache.cache_controller.ack_load_queue), 1)

    def test_tp_ranks_wait_for_full_and_swa_before_receiver_success(self) -> None:
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                events = [_FullSWACompletionEvent() for _ in range(tp_size)]
                caches = [
                    self._make_cache(event, node_id=rank + 1)
                    for rank, event in enumerate(events)
                ]
                def reduce_ready(value: torch.Tensor, reduction: object) -> None:
                    del reduction
                    ready = min(
                        int(event.full_complete and event.swa_complete)
                        for event in events
                    )
                    if value.numel() == 1:
                        value.fill_(ready)
                        return
                    value[0] = ready

                for cache in caches:
                    cache._all_reduce.side_effect = reduce_ready
                decode_reqs = [self._make_decode_req() for _ in range(tp_size)]

                self.assertEqual(
                    self._poll_all(caches, decode_reqs),
                    [KVPoll.Transferring] * tp_size,
                )
                for event in events:
                    event.full_complete = True
                self.assertEqual(
                    self._poll_all(caches, decode_reqs),
                    [KVPoll.Transferring] * tp_size,
                )
                for cache in caches:
                    cache.loading_check.assert_not_called()
                    self.assertEqual(len(cache.cache_controller.ack_load_queue), 1)

                if tp_size == 2:
                    events[0].swa_complete = True
                    polls = self._poll_all(caches, decode_reqs)
                    self.assertEqual(polls, [KVPoll.Transferring] * tp_size)
                    for cache in caches:
                        cache.loading_check.assert_not_called()

                for event in events:
                    event.swa_complete = True
                self.assertEqual(
                    self._poll_all(caches, decode_reqs),
                    [KVPoll.Success] * tp_size,
                )
                for event, cache in zip(events, caches, strict=True):
                    cache.loading_check.assert_called_once_with()
                    self.assertEqual(event.synchronize_count, 1)
                    self.assertEqual(cache.cache_controller.ack_load_queue, [])
                    self.assertEqual(cache.ongoing_load_back, {})
                    cache.dec_lock_ref.assert_called_once()
                    cache.dec_host_lock_ref.assert_called_once()


class TestUnifiedLoadBackResult(unittest.TestCase):
    _BEST_NODE = 29
    _OLD_DEVICE_NODE = 11

    @classmethod
    def _restore(
        cls, full_physical_tokens: int, swa_physical_tokens: int
    ) -> tuple[
        LoadBackResult,
        UnifiedRadixCache,
        PoolTransfer,
        PoolTransfer | None,
    ]:
        """Run a unified load-back with controller-allocated component copies.

        :param full_physical_tokens: Physical full-KV indices in the transfer.
        :param swa_physical_tokens: Physical SWA indices in the transfer.
        :returns: Public result, cache, and queued full/SWA transfers.
        """

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.arange(full_physical_tokens, dtype=torch.int64),
        )
        swa_transfer: PoolTransfer | None = None
        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if swa_physical_tokens > 0:
            swa_transfer = PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(swa_physical_tokens, dtype=torch.int64),
            )
            component_transfers[SWA] = [swa_transfer]

        tree_core = MagicMock()
        tree_core.build_load_back_spec.return_value = (
            kv_transfer,
            component_transfers,
        )
        tree_core.commit_load_back.return_value = []
        tree_core.is_full_device_evicted.return_value = full_physical_tokens > 0
        tree_core.empty_match_result = SimpleNamespace(
            device_indices=torch.empty((0,), dtype=torch.int64)
        )
        cache.tree_core = tree_core

        controller = MagicMock()

        def load(
            *,
            host_indices: torch.Tensor,
            node_id: int,
            extra_pools: list[PoolTransfer] | None,
        ) -> torch.Tensor:
            """Allocate physical destinations for the queued transfers.

            :param host_indices: Full-KV source indices.
            :param node_id: Restored radix node.
            :param extra_pools: Auxiliary component transfers.
            :returns: Full-KV destination indices.
            """

            if node_id != cls._BEST_NODE:
                raise AssertionError(f"unexpected restored node {node_id}")
            transfers = extra_pools if extra_pools is not None else ()
            for transfer in transfers:
                if transfer.name == PoolName.SWA:
                    transfer.device_indices = torch.arange(
                        int(transfer.host_indices.numel()), dtype=torch.int64
                    )
            return torch.arange(int(host_indices.numel()), dtype=torch.int64)

        controller.load.side_effect = load
        cache.cache_controller = controller
        cache.buffer_pipeline = None
        cache._components_tuple = ()
        cache.inc_host_lock_ref = MagicMock(return_value=IncLockRefResult(delta=0))
        cache.inc_lock_ref = MagicMock(return_value=IncLockRefResult(delta=0))
        cache.dec_lock_ref = MagicMock()
        cache.dec_host_lock_ref = MagicMock()
        cache._apply_cache_actions = MagicMock()
        cache.token_to_kv_pool_allocator = SimpleNamespace(
            full_available_size=MagicMock(return_value=1024)
        )
        cache.sidecar_pool_specs = []
        cache.load_back_threshold = 0
        cache.is_swa_enabled = True
        cache.ongoing_load_back = {}

        req = SimpleNamespace(
            last_node=cls._OLD_DEVICE_NODE,
            swa_host_hit_length=int(swa_physical_tokens > 0),
            mamba_host_hit_length=0,
        )
        result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=cls._BEST_NODE,
                host_hit_length=int(full_physical_tokens > 0),
                req=req,
            )
        )
        return result, cache, kv_transfer, swa_transfer

    def test_full_and_swa_result_uses_queued_physical_sizes(self) -> None:
        result, cache, kv_transfer, swa_transfer = self._restore(6, 4)

        self.assertTrue(result.queued_any_component)
        self.assertEqual(result.restored_node, self._BEST_NODE)
        self.assertEqual(result.new_full_device_indices.numel(), 6)
        self.assertEqual(result.full_tokens, 6)
        self.assertEqual(result.swa_tokens, 4)
        self.assertIsNotNone(kv_transfer.device_indices)
        self.assertIsNotNone(swa_transfer)
        self.assertIsNotNone(swa_transfer.device_indices)
        self.assertEqual(kv_transfer.device_indices.numel(), 6)
        self.assertEqual(swa_transfer.device_indices.numel(), 4)
        cache.tree_core.collect_full_device_indices.assert_not_called()

    def test_swa_only_result_restores_node_without_new_full_indices(self) -> None:
        result, cache, kv_transfer, swa_transfer = self._restore(0, 4)

        self.assertTrue(result.queued_any_component)
        self.assertEqual(result.restored_node, self._BEST_NODE)
        self.assertEqual(result.new_full_device_indices.numel(), 0)
        self.assertEqual(result.full_tokens, 0)
        self.assertEqual(result.swa_tokens, 4)
        self.assertIsNotNone(kv_transfer.device_indices)
        self.assertIsNotNone(swa_transfer)
        self.assertIsNotNone(swa_transfer.device_indices)
        self.assertEqual(kv_transfer.device_indices.numel(), 0)
        self.assertEqual(swa_transfer.device_indices.numel(), 4)
        cache.tree_core.collect_full_device_indices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
