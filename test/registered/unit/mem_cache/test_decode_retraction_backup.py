import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    free_swa_out_of_window_slots,
    retraction_backup,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.streaming_session import SessionSlot
from sglang.srt.speculative.base_spec_worker import (
    HiCacheDraftMode,
    HiCacheDraftPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDecodeRetractionBackup(unittest.TestCase):
    pool_size = 32
    num_tokens = 8
    dtype = torch.bfloat16
    device = "cuda"

    def test_demotion_insert_disables_every_write_through_path(self) -> None:
        tree_core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        state = SimpleNamespace(params=InsertParams(trigger_backup=False))

        self.assertFalse(tree_core._should_backup_after_insert(state))

    def test_demotion_splits_node_at_swa_host_window_boundary(self) -> None:
        class RecordingTreeCore:
            """Record tree mutations made by SWA host publication."""

            def __init__(self) -> None:
                self.splits: list[tuple[int, int]] = []
                self.commits: list[tuple[int, torch.Tensor]] = []

            def _split_node(
                self,
                key: list[int],
                node: SimpleNamespace,
                split_len: int,
            ) -> tuple[SimpleNamespace, None]:
                self.splits.append((node.id, split_len))
                node.key = key[split_len:]
                return SimpleNamespace(), None

            def commit_backup(
                self,
                node_id: int,
                host_indices: torch.Tensor,
                transfers: dict[ComponentType, list[PoolTransfer]],
            ) -> None:
                self.assert_empty_full(host_indices)
                swa_transfer = transfers[ComponentType.SWA][0]
                assert swa_transfer.host_indices is not None
                self.commits.append((node_id, swa_transfer.host_indices.clone()))

            @staticmethod
            def assert_empty_full(host_indices: torch.Tensor) -> None:
                assert host_indices.numel() == 0

        class RecordingHostPoolGroup:
            """Record released host slices."""

            def __init__(self) -> None:
                self.frees: list[tuple[PoolName, torch.Tensor]] = []

            def free(
                self,
                indices: torch.Tensor,
                *,
                pool: PoolName,
            ) -> int:
                self.frees.append((pool, indices.clone()))
                return len(indices)

        def node(node_id: int, length: int) -> SimpleNamespace:
            return SimpleNamespace(
                id=node_id,
                key=list(range(length)),
                is_session_fringe=False,
                is_session_private=False,
                private_physical_length=None,
                component_data={
                    ComponentType.FULL: SimpleNamespace(
                        host_value=torch.arange(length)
                    ),
                    ComponentType.SWA: SimpleNamespace(host_value=None),
                },
            )

        prefix_node = node(1, 8)
        tail_node = node(2, 4)
        tree_core = RecordingTreeCore()
        host_pool_group = RecordingHostPoolGroup()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = tree_core
        cache.host_pool_group = host_pool_group

        def path(_last_node: int, _expected_len: int) -> list[SimpleNamespace]:
            return [prefix_node, tail_node]

        cache._streaming_session_path = path
        backup = RetractionBackup(
            host_indices=torch.arange(12),
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.arange(100, 108),
                )
            ],
        )

        cache._commit_streaming_session_host_path(tail_node.id, backup)

        self.assertEqual(tree_core.splits, [(prefix_node.id, 4)])
        self.assertEqual(len(prefix_node.key), 4)
        self.assertEqual(
            [(node_id, indices.tolist()) for node_id, indices in tree_core.commits],
            [
                (tail_node.id, [104, 105, 106, 107]),
                (prefix_node.id, [100, 101, 102, 103]),
            ],
        )

    def _make_pool(
        self, layer_num: int, *, size: int | None = None, page_size: int = 1
    ) -> MHATokenToKVPool:
        return MHATokenToKVPool(
            size=self.pool_size if size is None else size,
            page_size=page_size,
            head_num=2,
            head_dim=64,
            dtype=self.dtype,
            layer_num=layer_num,
            device=self.device,
            enable_memory_saver=False,
        )

    def _seed_pool(
        self, pool: MHATokenToKVPool, indices: torch.Tensor, base: int
    ) -> None:
        for layer_id, (key, value) in enumerate(
            zip(pool.k_buffer, pool.v_buffer, strict=True)
        ):
            pattern = torch.arange(
                key[indices].numel(), device=self.device, dtype=torch.float32
            ).reshape_as(key[indices])
            key[indices] = (pattern + base + layer_id * 100).to(self.dtype)
            value[indices] = (pattern + base + 50 + layer_id * 100).to(self.dtype)

    @staticmethod
    def _snapshot_pool(
        pool: MHATokenToKVPool, indices: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (key[indices].clone(), value[indices].clone())
            for key, value in zip(pool.k_buffer, pool.v_buffer, strict=True)
        ]

    def _assert_pool_equal(
        self,
        pool: MHATokenToKVPool,
        indices: torch.Tensor,
        expected: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for (key, value), (expected_key, expected_value) in zip(
            zip(pool.k_buffer, pool.v_buffer, strict=True), expected, strict=True
        ):
            self.assertTrue(torch.equal(key[indices], expected_key))
            self.assertTrue(torch.equal(value[indices], expected_value))

    def _build_cache(
        self,
        hicache_ratio: float,
        *,
        disable: bool = True,
        enable_session_radix_cache: bool = False,
        page_size: int = 1,
        sliding_window_size: int | None = None,
    ) -> SimpleNamespace:
        """Bring up a UnifiedRadixCache with a draft sidecar over fresh pools."""
        server_args = ServerArgs(
            model_path="dummy",
            page_size=page_size,
            hicache_ratio=hicache_ratio,
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
        )
        set_global_server_args_for_scheduler(server_args)

        pool_size = max(self.pool_size, page_size * 4)
        req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=pool_size,
            device=self.device,
            enable_memory_saver=False,
        )
        swa_pool = None
        if sliding_window_size is not None:
            kv_pool = SWAKVPool(
                size=pool_size,
                size_swa=pool_size,
                page_size=page_size,
                dtype=self.dtype,
                head_num=2,
                head_dim=64,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                device=self.device,
            )
            allocator = SWATokenToKVPoolAllocator(
                size=pool_size,
                size_swa=pool_size,
                page_size=page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=kv_pool,
                need_sort=False,
            )
            target_pool = kv_pool.full_kv_pool
            swa_pool = kv_pool.swa_kv_pool
            tree_components = (ComponentType.FULL, ComponentType.SWA)
        else:
            target_pool = self._make_pool(
                layer_num=2, size=pool_size, page_size=page_size
            )
            kv_pool = target_pool
            tree_components = (ComponentType.FULL,)
        if page_size == 1 and sliding_window_size is None:
            allocator = TokenToKVPoolAllocator(
                size=pool_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=kv_pool,
                need_sort=False,
            )
        elif sliding_window_size is None:
            allocator = PagedTokenToKVPoolAllocator(
                size=pool_size,
                page_size=page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=kv_pool,
                need_sort=False,
            )
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            is_eagle=True,
            enable_session_radix_cache=enable_session_radix_cache,
            sliding_window_size=sliding_window_size,
            tree_components=tree_components,
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(server_args, params)
        self.addCleanup(cache.release_host_resources)

        draft_pool = self._make_pool(layer_num=1, size=pool_size, page_size=page_size)
        maybe_register_hicache_draft(
            tree_cache=cache,
            draft_plan=HiCacheDraftPlan(
                mode=HiCacheDraftMode.SIDECAR,
                device_pools=(draft_pool,),
            ),
        )
        self.assertIn(PoolName.DRAFT, cache.host_pool_group.entry_map)
        cache.validate_retraction_host_capacity()
        return SimpleNamespace(
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            allocator=allocator,
            target_pool=target_pool,
            swa_pool=swa_pool,
            draft_pool=draft_pool,
            cache=cache,
        )

    def _admit_req(self, env, num_tokens: int):
        req = SimpleNamespace(rid="request", req_pool_idx=None, seqlen=num_tokens + 1)
        self.assertIsNotNone(env.req_to_token_pool.alloc([req]))
        allocated_tokens = (
            (num_tokens + env.server_args.page_size - 1)
            // env.server_args.page_size
            * env.server_args.page_size
        )
        if isinstance(env.allocator, SWATokenToKVPoolAllocator):
            source_indices = env.allocator.full_attn_allocator.alloc(allocated_tokens)
            swa_indices = env.allocator.swa_attn_allocator.alloc(allocated_tokens)
            self.assertIsNotNone(source_indices)
            self.assertIsNotNone(swa_indices)
            env.allocator.set_full_to_swa_mapping(source_indices, swa_indices)
        else:
            source_indices = env.allocator.alloc(allocated_tokens)
        self.assertIsNotNone(source_indices)
        env.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, num_tokens)), source_indices[:num_tokens]
        )
        return req, source_indices

    def _admit_streaming_session(
        self,
        env: SimpleNamespace,
        session_id: str,
        num_tokens: int | None = None,
    ) -> tuple[SimpleNamespace, torch.Tensor]:
        """Attach seeded target and draft KV to one detached session slot.

        :param env: Real CUDA cache fixture.
        :param session_id: Session identity to register.
        :returns: Synthetic request and its allocated device indices.
        """
        exact_tokens = self.num_tokens if num_tokens is None else num_tokens
        req, source_indices = self._admit_req(env, exact_tokens)
        self._seed_pool(env.target_pool, source_indices, base=1000)
        self._seed_pool(env.draft_pool, source_indices, base=3000)
        env.cache.session.slots[session_id] = SessionSlot(
            req_pool_idx=req.req_pool_idx,
            kv_committed_len=exact_tokens,
            kv=ReqKvInfo(kv_allocated_len=exact_tokens),
        )
        env.cache.open_radix_session(session_id)
        return req, source_indices

    @staticmethod
    def _streaming_match_req(session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            session=SimpleNamespace(
                streaming=True,
                session_id=session_id,
            ),
            kv=ReqKvInfo(),
        )

    def test_backup_declined_when_host_pool_too_small(self):
        # A backup-only host pool is deliberately smaller than the device pool,
        # so a large enough request cannot be preserved.
        env = self._build_cache(hicache_ratio=0.1)
        self.assertLess(env.cache.host_pool_group.available_size(), self.num_tokens)

        req, source_indices = self._admit_req(env, self.num_tokens)
        host_free_before = env.cache.host_pool_group.available_size()

        self.assertIsNone(env.cache.retraction_backup(req))
        # The declined backup must not leak host slots.
        self.assertEqual(env.cache.host_pool_group.available_size(), host_free_before)

        # This is the signal release_req propagates so retract_decode aborts.
        self.assertFalse(
            retraction_backup(
                req,
                env.cache,
                env.req_to_token_pool,
                env.allocator,
                "host_pool",
            )
        )

        env.allocator.free(source_indices)
        env.req_to_token_pool.free(req)

    def test_restores_target_and_draft_kv(self):
        env = self._build_cache(hicache_ratio=1.0)
        req_to_token_pool = env.req_to_token_pool
        allocator = env.allocator
        target_pool = env.target_pool
        draft_pool = env.draft_pool
        cache = env.cache

        req, source_indices = self._admit_req(env, self.num_tokens)

        self._seed_pool(target_pool, source_indices, base=1000)
        self._seed_pool(draft_pool, source_indices, base=3000)
        target_expected = self._snapshot_pool(target_pool, source_indices)
        draft_expected = self._snapshot_pool(draft_pool, source_indices)

        host_free_before = cache.host_pool_group.available_size()
        backup = cache.retraction_backup(req)
        self.assertEqual(
            {transfer.name for transfer in backup.pool_transfers or []},
            {PoolName.DRAFT},
        )
        self.assertLess(cache.host_pool_group.available_size(), host_free_before)

        for buffer in (*target_pool.k_buffer, *target_pool.v_buffer):
            buffer.fill_(-1)
        for buffer in (*draft_pool.k_buffer, *draft_pool.v_buffer):
            buffer.fill_(-2)

        allocator.free(source_indices)
        blocker_indices = allocator.alloc(self.num_tokens)
        destination_indices = allocator.alloc(self.num_tokens)
        self.assertIsNotNone(blocker_indices)
        self.assertIsNotNone(destination_indices)
        self.assertFalse(torch.equal(source_indices, destination_indices))
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, self.num_tokens)), destination_indices
        )

        cache.retraction_restore(req, backup)

        self._assert_pool_equal(target_pool, destination_indices, target_expected)
        self._assert_pool_equal(draft_pool, destination_indices, draft_expected)
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

        allocator.free(blocker_indices)
        allocator.free(destination_indices)
        req_to_token_pool.free(req)

    def test_streaming_session_demotion_publishes_and_releases_host_frontier(self):
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
        )
        cache = env.cache
        self._admit_streaming_session(env, "session")
        host_free_before = cache.host_pool_group.available_size()

        prepared = cache.prepare_streaming_session_demotion(
            "session",
            array("q", range(self.num_tokens + 1)),
            extra_key=None,
            cache_salt=None,
            priority=0,
        )

        self.assertEqual(prepared, self.num_tokens)
        committed = cache.commit_streaming_session_demotion("session")
        self.assertEqual(committed, self.num_tokens)
        self.assertNotIn("session", cache.session.slots)
        self.assertTrue(cache.is_streaming_session_demoted("session"))
        self.assertEqual(
            cache.streaming_session_cache_snapshot("session").full.host_backed_pages,
            self.num_tokens,
        )
        self.assertEqual(
            cache.streaming_session_cache_snapshot("session").full.device_pages,
            0,
        )
        demoted = cache.session.demoted["session"]
        node = cache.tree_core.node_by_id(demoted.last_node)
        self.assertTrue(node.evicted)
        self.assertEqual(
            node.component_data[ComponentType.FULL].host_lock_ref,
            1,
        )
        self.assertEqual(
            node.component_data[ComponentType.FULL].session_ref,
            1,
        )
        cache.sanity_check()

        cache.release_radix_session("session")
        cache.release_session("session")

        self.assertFalse(cache.is_streaming_session_demoted("session"))
        self.assertEqual(
            node.component_data[ComponentType.FULL].host_lock_ref,
            0,
        )
        self.assertEqual(
            node.component_data[ComponentType.FULL].session_ref,
            0,
        )
        cache.sanity_check()
        self.assertEqual(cache.evict_host(self.num_tokens), 0)
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

    def test_streaming_session_demotion_publishes_exact_private_fringe(self) -> None:
        page_size = 64
        exact_tokens = page_size + 1
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
        )
        cache = env.cache
        _, source_indices = self._admit_streaming_session(env, "session", exact_tokens)
        target_expected = self._snapshot_pool(env.target_pool, source_indices)
        draft_expected = self._snapshot_pool(env.draft_pool, source_indices)
        token_ids = array("q", range(exact_tokens + 1))

        prepared = cache.prepare_streaming_session_demotion(
            "session",
            token_ids,
            extra_key="namespace",
            cache_salt="tenant",
            priority=0,
        )
        committed = cache.commit_streaming_session_demotion("session")

        self.assertEqual(prepared, exact_tokens)
        self.assertEqual(committed, exact_tokens)
        state = cache.session.demoted["session"]
        fringe = cache.tree_core.node_by_id(state.last_node)
        self.assertTrue(fringe.is_session_fringe)
        self.assertEqual(fringe.private_session_id, "session")
        self.assertEqual(len(fringe.key), 1)
        aligned_private = fringe.parent
        self.assertTrue(aligned_private.is_session_private)
        self.assertEqual(aligned_private.private_session_id, "session")
        self.assertEqual(len(aligned_private.key), page_size)
        self.assertEqual(
            len(fringe.component_data[ComponentType.FULL].host_value),
            page_size,
        )
        self.assertEqual(
            cache.streaming_session_cache_snapshot("session").protected,
            exact_tokens,
        )
        self.assertEqual(
            cache.streaming_session_cache_snapshot("session").full.host_backed_pages,
            2,
        )

        def match(session_id: str, key_token_ids: array = token_ids):
            return cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        key_token_ids,
                        extra_key="namespace",
                        cache_salt="tenant",
                    ),
                    req=self._streaming_match_req(session_id),
                )
            )

        owner_match = match("session")
        owner_continuation_match = match(
            "session",
            array("q", range(exact_tokens + 12)),
        )
        foreign_match = match("other-session")
        self.assertEqual(owner_match.full_kv_hit_length, exact_tokens)
        self.assertEqual(owner_match.cache_protected_len, exact_tokens)
        self.assertEqual(owner_match.best_match_node, fringe.id)
        self.assertEqual(
            owner_continuation_match.full_kv_hit_length,
            exact_tokens,
        )
        self.assertEqual(
            owner_continuation_match.cache_protected_len,
            exact_tokens,
        )
        self.assertEqual(owner_continuation_match.best_match_node, fringe.id)
        self.assertEqual(foreign_match.full_kv_hit_length, 0)
        self.assertNotEqual(foreign_match.best_match_node, fringe.id)

        load_req = SimpleNamespace(
            last_node=owner_match.last_device_node,
            prefix_indices=owner_match.device_indices,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_pool_idx=None,
        )
        load_result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=owner_match.best_match_node,
                host_hit_length=owner_match.host_hit_length,
                req=load_req,
            )
        )
        self.assertEqual(len(load_result.new_full_device_indices), exact_tokens)
        self.assertEqual(load_result.full_tokens, page_size * 2)
        self.assertEqual(load_result.cache_protected_len, exact_tokens)
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)
        restored_indices = load_result.new_full_device_indices
        fringe_page = restored_indices[-1] + torch.arange(
            page_size,
            dtype=torch.int64,
            device=self.device,
        )
        restored_physical_indices = torch.cat(
            [restored_indices[:page_size], fringe_page]
        )
        self._assert_pool_equal(
            env.target_pool,
            restored_physical_indices,
            target_expected,
        )
        self._assert_pool_equal(
            env.draft_pool,
            restored_physical_indices,
            draft_expected,
        )

        continuation_indices = env.allocator.alloc_extend(
            torch.tensor([exact_tokens], dtype=torch.int64, device=self.device),
            torch.tensor([exact_tokens], dtype=torch.int64),
            torch.tensor([page_size * 2], dtype=torch.int64, device=self.device),
            torch.tensor([page_size * 2], dtype=torch.int64),
            restored_indices[-1:],
            page_size - 1,
        )
        self.assertTrue(torch.equal(continuation_indices, fringe_page[1:]))

        cache.sanity_check()
        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()

    def test_streaming_session_private_suffix_wins_collision_and_redemotes(
        self,
    ) -> None:
        page_size = 64
        exact_tokens = page_size + 1
        token_ids = array("q", range(exact_tokens + 1))
        key = RadixKey(
            token_ids,
            extra_key="namespace",
            cache_salt="tenant",
            is_bigram=True,
        )
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
        )
        cache = env.cache
        _, source_indices = self._admit_streaming_session(env, "session", exact_tokens)
        target_expected = self._snapshot_pool(env.target_pool, source_indices)
        draft_expected = self._snapshot_pool(env.draft_pool, source_indices)

        collision_indices = env.allocator.alloc(page_size)
        self.assertIsNotNone(collision_indices)
        self._seed_pool(env.target_pool, collision_indices, base=7000)
        self._seed_pool(env.draft_pool, collision_indices, base=9000)
        collision = cache.insert(
            InsertParams(
                key=key[:page_size],
                value=collision_indices,
                trigger_backup=False,
            )
        )
        host_free_before = cache.host_pool_group.available_size()

        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                "session",
                token_ids,
                extra_key="namespace",
                cache_salt="tenant",
                priority=0,
            ),
            exact_tokens,
        )
        self.assertEqual(
            cache.commit_streaming_session_demotion("session"), exact_tokens
        )
        first_state = cache.session.demoted["session"]
        first_private_ids: list[int] = []
        private_node = cache.tree_core.node_by_id(first_state.last_node)
        while private_node.is_session_private:
            first_private_ids.append(private_node.id)
            private_node = private_node.parent

        def match(session_id: str):
            return cache.match_prefix(
                MatchPrefixParams(
                    key=key,
                    req=self._streaming_match_req(session_id),
                )
            )

        owner_match = match("session")
        foreign_match = match("other-session")
        self.assertEqual(owner_match.full_kv_hit_length, exact_tokens)
        self.assertEqual(owner_match.best_match_node, first_state.last_node)
        self.assertEqual(foreign_match.full_kv_hit_length, page_size)
        self.assertEqual(foreign_match.best_match_node, collision.last_device_node)

        load_req = SimpleNamespace(
            last_node=owner_match.last_device_node,
            prefix_indices=owner_match.device_indices,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_pool_idx=None,
        )
        load_result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=owner_match.best_match_node,
                host_hit_length=owner_match.host_hit_length,
                req=load_req,
            )
        )
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)

        restored_indices = load_result.new_full_device_indices
        fringe_page = restored_indices[-1] + torch.arange(
            page_size,
            dtype=torch.int64,
            device=self.device,
        )
        restored_physical_indices = torch.cat(
            [restored_indices[:page_size], fringe_page]
        )
        self._assert_pool_equal(
            env.target_pool,
            restored_physical_indices,
            target_expected,
        )
        self._assert_pool_equal(
            env.draft_pool,
            restored_physical_indices,
            draft_expected,
        )

        restored_req = SimpleNamespace(rid="restored", req_pool_idx=None)
        self.assertIsNotNone(env.req_to_token_pool.alloc([restored_req]))
        env.req_to_token_pool.write(
            (restored_req.req_pool_idx, slice(0, exact_tokens)),
            restored_indices,
        )
        restored_lock = cache.inc_lock_ref(first_state.last_node)
        cache.session.slots["session"] = SessionSlot(
            req_pool_idx=restored_req.req_pool_idx,
            kv_committed_len=exact_tokens,
            kv=ReqKvInfo(kv_allocated_len=exact_tokens),
            last_node=first_state.last_node,
            cache_protected_len=exact_tokens,
            tree_protected_len=exact_tokens,
            swa_uuid_for_lock=restored_lock.swa_uuid_for_lock,
            skip_lock_node_ids=restored_lock.skip_lock_node_ids,
        )
        cache.session._release_demoted_state("session")

        restored_slot = cache.session.slots["session"]
        self.assertEqual(restored_slot.last_node, cache.tree_core.root_node.id)
        self.assertEqual(restored_slot.tree_protected_len, 0)
        self.assertTrue(
            all(
                node_id not in cache.tree_core._node_arena
                for node_id in first_private_ids
            )
        )
        self._assert_pool_equal(
            env.target_pool,
            restored_physical_indices,
            target_expected,
        )
        cache.sanity_check()

        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                "session",
                token_ids,
                extra_key="namespace",
                cache_salt="tenant",
                priority=0,
            ),
            exact_tokens,
        )
        self.assertEqual(
            cache.commit_streaming_session_demotion("session"), exact_tokens
        )
        cache.sanity_check()

        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

    def test_streaming_session_exact_fringe_restores_swa_page_mapping(self) -> None:
        page_size = 64
        exact_tokens = page_size + 1
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size * 2,
        )
        cache = env.cache
        _, source_indices = self._admit_streaming_session(env, "session", exact_tokens)
        source_swa_indices = env.allocator.translate_loc_from_full_to_swa(
            source_indices
        )
        self._seed_pool(env.swa_pool, source_swa_indices, base=5000)
        swa_expected = self._snapshot_pool(env.swa_pool, source_swa_indices)
        token_ids = array("q", range(exact_tokens + 1))

        prepared = cache.prepare_streaming_session_demotion(
            "session",
            token_ids,
            extra_key="namespace",
            cache_salt="tenant",
            priority=0,
        )
        committed = cache.commit_streaming_session_demotion("session")

        self.assertEqual(prepared, exact_tokens)
        self.assertEqual(committed, exact_tokens)
        owner_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    extra_key="namespace",
                    cache_salt="tenant",
                ),
                req=self._streaming_match_req("session"),
            )
        )
        load_req = SimpleNamespace(
            last_node=owner_match.last_device_node,
            prefix_indices=owner_match.device_indices,
            swa_host_hit_length=owner_match.swa_host_hit_length,
            mamba_host_hit_length=0,
            mamba_pool_idx=None,
        )
        load_result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=owner_match.best_match_node,
                host_hit_length=owner_match.host_hit_length,
                req=load_req,
            )
        )
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)

        restored_full_indices = load_result.new_full_device_indices
        fringe_full_page = restored_full_indices[-1] + torch.arange(
            page_size,
            dtype=torch.int64,
            device=self.device,
        )
        restored_physical_full_indices = torch.cat(
            [restored_full_indices[:page_size], fringe_full_page]
        )
        restored_swa_indices = env.allocator.translate_loc_from_full_to_swa(
            restored_physical_full_indices
        )
        self._assert_pool_equal(env.swa_pool, restored_swa_indices, swa_expected)
        fringe_swa_page = env.allocator.translate_loc_from_full_to_swa(fringe_full_page)
        self.assertTrue(bool((fringe_swa_page > 0).all()))

        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()

    def test_streaming_session_swa_adoption_balances_real_allocator(self) -> None:
        page_size = 64
        exact_tokens = page_size * 2 + 1
        below_transition = page_size * 3 - 1
        transition = page_size * 3
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        allocator = env.allocator
        full_available_baseline = allocator.full_available_size()
        swa_available_baseline = allocator.swa_available_size()
        request_available_baseline = env.req_to_token_pool.available_size()
        host_available_baseline = cache.host_pool_group.available_size()

        self._admit_streaming_session(env, "session", exact_tokens)
        token_ids = array("q", range(exact_tokens + 1))
        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                "session",
                token_ids,
                extra_key="namespace",
                cache_salt="tenant",
                priority=0,
            ),
            exact_tokens,
        )
        self.assertEqual(
            cache.commit_streaming_session_demotion("session"),
            exact_tokens,
        )
        cache.sanity_check()
        state = cache.session.demoted["session"]
        self.assertEqual(state.swa_evicted_seqlen, page_size)
        self.assertEqual(allocator.full_available_size(), full_available_baseline)
        self.assertEqual(allocator.swa_available_size(), swa_available_baseline)
        self.assertEqual(
            env.req_to_token_pool.available_size(), request_available_baseline
        )

        match_req = self._streaming_match_req("session")
        owner_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    extra_key="namespace",
                    cache_salt="tenant",
                ),
                req=match_req,
            )
        )
        self.assertEqual(owner_match.full_kv_hit_length, exact_tokens)
        self.assertEqual(match_req.kv.swa_evicted_seqlen, page_size)
        self.assertEqual(
            match_req.streaming_session_tree_protected_len,
            exact_tokens,
        )
        match_req.last_node = owner_match.last_device_node
        match_req.prefix_indices = owner_match.device_indices
        match_req.swa_host_hit_length = owner_match.swa_host_hit_length
        match_req.mamba_host_hit_length = 0
        match_req.mamba_pool_idx = None
        load_result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=owner_match.best_match_node,
                host_hit_length=owner_match.host_hit_length,
                req=match_req,
            )
        )
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)

        restored_req = SimpleNamespace(rid="restored", req_pool_idx=None)
        self.assertIsNotNone(env.req_to_token_pool.alloc([restored_req]))
        restored_indices = load_result.new_full_device_indices
        fringe_page = restored_indices[-1] + torch.arange(
            page_size,
            dtype=torch.int64,
            device=self.device,
        )
        env.req_to_token_pool.write(
            (restored_req.req_pool_idx, slice(0, exact_tokens)),
            restored_indices,
        )
        env.req_to_token_pool.write(
            (restored_req.req_pool_idx, slice(exact_tokens, transition)),
            fringe_page[1:],
        )
        match_req.req_pool_idx = restored_req.req_pool_idx
        match_req.is_holding_kv = True
        match_req.cache_protected_len = exact_tokens
        match_req.swa_evict_floor = 0
        match_req.streaming_session_floor = transition
        match_req.kv.kv_allocated_len = transition
        swa_available_after_restore = allocator.swa_available_size()

        for pre_len, expected_watermark in (
            (below_transition, page_size),
            (transition, page_size * 2),
        ):
            free_swa_out_of_window_slots(
                match_req,
                pre_len,
                sliding_window_size=page_size,
                page_size=page_size,
                req_to_token_pool=env.req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
            )
            self.assertEqual(
                match_req.kv.swa_evicted_seqlen,
                expected_watermark,
            )
            self.assertEqual(
                allocator.swa_available_size(),
                swa_available_after_restore,
            )

        restored_lock = cache.inc_lock_ref(state.last_node)
        cache.session.slots["session"] = SessionSlot(
            req_pool_idx=restored_req.req_pool_idx,
            kv_committed_len=transition,
            kv=match_req.kv,
            streaming_session_floor=transition,
            last_node=state.last_node,
            cache_protected_len=exact_tokens,
            tree_protected_len=exact_tokens,
            swa_uuid_for_lock=restored_lock.swa_uuid_for_lock,
            skip_lock_node_ids=restored_lock.skip_lock_node_ids,
        )

        self.assertTrue(cache.session._release_demoted_state("session"))

        slot = cache.session.slots["session"]
        self.assertEqual(slot.tree_protected_len, 0)
        self.assertEqual(slot.kv.swa_evicted_seqlen, page_size * 2)
        full_consumption = (
            full_available_baseline - allocator.full_available_size()
        )
        swa_consumption = swa_available_baseline - allocator.swa_available_size()
        self.assertEqual(cache.session.session_held_tokens(), full_consumption)
        self.assertEqual(cache.session.session_held_swa_tokens(), swa_consumption)
        self.assertEqual(swa_consumption, page_size)
        released_full_slots = env.req_to_token_pool.req_to_token[
            slot.req_pool_idx, page_size : page_size * 2
        ]
        retained_full_slots = env.req_to_token_pool.req_to_token[
            slot.req_pool_idx, page_size * 2 : transition
        ]
        self.assertTrue(
            bool(
                (
                    allocator.translate_loc_from_full_to_swa(released_full_slots)
                    == 0
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    allocator.translate_loc_from_full_to_swa(retained_full_slots)
                    > 0
                ).all()
            )
        )

        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()
        self.assertEqual(allocator.full_available_size(), full_available_baseline)
        self.assertEqual(allocator.swa_available_size(), swa_available_baseline)
        self.assertEqual(
            env.req_to_token_pool.available_size(), request_available_baseline
        )
        self.assertEqual(
            cache.host_pool_group.available_size(), host_available_baseline
        )

    def test_streaming_session_failed_vote_discards_private_stage(self) -> None:
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
        )
        cache = env.cache
        _, source_indices = self._admit_streaming_session(env, "session")
        target_expected = self._snapshot_pool(env.target_pool, source_indices)
        draft_expected = self._snapshot_pool(env.draft_pool, source_indices)
        host_free_before = cache.host_pool_group.available_size()

        prepared = cache.prepare_streaming_session_demotion(
            "session",
            array("q", range(self.num_tokens + 1)),
            extra_key=None,
            cache_salt=None,
            priority=0,
        )

        self.assertEqual(prepared, self.num_tokens)
        self.assertLess(cache.host_pool_group.available_size(), host_free_before)
        cache.discard_streaming_session_demotion("session")

        self.assertIn("session", cache.session.slots)
        self.assertFalse(cache.is_streaming_session_demoted("session"))
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)
        self._assert_pool_equal(env.target_pool, source_indices, target_expected)
        self._assert_pool_equal(env.draft_pool, source_indices, draft_expected)

        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
