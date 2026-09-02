import unittest
from array import array
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
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
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    _HostStageLedger,
    _StreamingSessionHostPathTransaction,
    UnifiedRadixCache,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.session.errors import StreamingSessionBusyError
from sglang.srt.session.streaming_session import SessionSlot
from sglang.srt.speculative.base_spec_worker import (
    HiCacheDraftMode,
    HiCacheDraftPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@dataclass(frozen=True)
class _PrivatePathLifecycleState:
    """One observable boundary in a private host path's lifecycle.

    :ivar private_node_ids: Nodes created for the session-private suffix.
    :ivar arena_node_ids: Nodes currently registered in the tree arena.
    :ivar component_state: Device presence, host presence, device lock, host
        lock, session reference, and exact host owners by private node and
        component.
    :ivar device_lru_ids: Registered device-LRU nodes by component.
    :ivar host_lru_ids: Registered host-LRU nodes by component.
    :ivar expected_device_leaf_ids: Private nodes satisfying device-leaf rules.
    :ivar expected_host_leaf_ids: Private nodes satisfying host-leaf rules.
    :ivar device_leaf_ids: Registered device-leaf nodes.
    :ivar host_leaf_ids: Registered host-leaf nodes.
    :ivar session_leaf_ids: Indexed session frontiers by component.
    :ivar owner_pages: Full-KV pages owned by the detached session slot.
    :ivar free_full_pages: Full-KV pages currently available to the allocator.
    """

    private_node_ids: frozenset[int]
    arena_node_ids: frozenset[int]
    component_state: dict[
        tuple[int, ComponentType],
        tuple[bool, bool, int, int, int, frozenset[int]],
    ]
    device_lru_ids: dict[ComponentType, frozenset[int]]
    host_lru_ids: dict[ComponentType, frozenset[int]]
    expected_device_leaf_ids: frozenset[int]
    expected_host_leaf_ids: frozenset[int]
    device_leaf_ids: frozenset[int]
    host_leaf_ids: frozenset[int]
    session_leaf_ids: dict[ComponentType, frozenset[int]]
    owner_pages: frozenset[int]
    free_full_pages: frozenset[int]


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
        """The SWA window boundary is split before publication, then each
        selected node receives exactly its staged SWA slice in path order."""

        def make_node(node_id: int, length: int) -> SimpleNamespace:
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

        class RecordingTreeCore:
            """Record the split and attach SWA host slices like the real core."""

            def __init__(self) -> None:
                self.nodes: dict[int, SimpleNamespace] = {}
                self.splits: list[tuple[int, int]] = []
                self.commits: list[tuple[int, torch.Tensor]] = []

            def _split_node(
                self,
                key: list[int],
                node: SimpleNamespace,
                split_len: int,
            ) -> tuple[SimpleNamespace, None]:
                self.splits.append((node.id, split_len))
                parent = make_node(max(self.nodes) + 1, split_len)
                node.key = key[split_len:]
                self.nodes[parent.id] = parent
                return parent, None

            def commit_backup(
                self,
                node_id: int,
                host_indices: torch.Tensor,
                transfers: dict[ComponentType, list[PoolTransfer]],
            ) -> None:
                assert host_indices.numel() == 0
                swa_transfer = transfers[ComponentType.SWA][0]
                assert swa_transfer.host_indices is not None
                self.commits.append((node_id, swa_transfer.host_indices.clone()))
                swa_data = self.nodes[node_id].component_data[ComponentType.SWA]
                swa_data.host_value = swa_transfer.host_indices.clone()

        prefix_node = make_node(1, 8)
        tail_node = make_node(2, 4)
        tree_core = RecordingTreeCore()
        tree_core.nodes = {prefix_node.id: prefix_node, tail_node.id: tail_node}
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = tree_core

        def path(_last_node: int, _expected_len: int) -> list[SimpleNamespace]:
            split_parents = [
                node
                for node in tree_core.nodes.values()
                if node is not prefix_node and node is not tail_node
            ]
            return [*split_parents, prefix_node, tail_node]

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

        cache._split_streaming_session_path_at(tail_node.id, 12, 4)
        transaction = _StreamingSessionHostPathTransaction(
            ledger=_HostStageLedger.from_backup(backup)
        )
        cache._commit_streaming_session_host_path(
            tail_node.id,
            backup,
            transaction,
            swa_window_start=4,
        )

        self.assertEqual(tree_core.splits, [(prefix_node.id, 4)])
        self.assertEqual(len(prefix_node.key), 4)
        self.assertEqual(
            [(node_id, indices.tolist()) for node_id, indices in tree_core.commits],
            [
                (prefix_node.id, [100, 101, 102, 103]),
                (tail_node.id, [104, 105, 106, 107]),
            ],
        )
        self.assertEqual(
            [
                (attachment.node_id, attachment.component_type)
                for attachment in transaction.attachments
            ],
            [
                (prefix_node.id, ComponentType.SWA),
                (tail_node.id, ComponentType.SWA),
            ],
        )
        self.assertTrue(
            all(attachment.stage_slice.tree_owned for attachment in transaction.attachments)
        )
        transaction.ledger.assert_fully_consumed()

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

    def _capture_private_path_lifecycle_state(
        self,
        env: SimpleNamespace,
        private_nodes: tuple[UnifiedTreeNode, ...],
        session_id: str,
    ) -> _PrivatePathLifecycleState:
        """Capture every tree and allocator index for one lifecycle boundary.

        :param env: Real CUDA cache fixture.
        :param private_nodes: Stable private-node objects retained across detach.
        :param session_id: Session whose indexes are inspected.
        :returns: Immutable identifiers and counters for assertions.
        """
        cache = env.cache
        tree_core = cache.tree_core
        private_node_ids = frozenset(node.id for node in private_nodes)
        arena_node_ids = frozenset(tree_core._node_arena)
        component_state = {
            (node.id, component_type): (
                node.component_data[component_type].value is not None,
                node.component_data[component_type].host_value is not None,
                node.component_data[component_type].lock_ref,
                node.component_data[component_type].host_lock_ref,
                node.component_data[component_type].session_ref,
                frozenset(node.component_data[component_type].host_lock_ids or ()),
            )
            for node in private_nodes
            for component_type in tree_core.component_types
        }
        device_lru_ids = {
            component_type: frozenset(tree_core.lru_lists[component_type].cache)
            for component_type in tree_core.component_types
        }
        host_lru_ids = {
            component_type: frozenset(tree_core.host_lru_lists[component_type].cache)
            for component_type in tree_core.component_types
        }
        session_leaf_ids = {
            component.component_type: frozenset(
                node.id for node in component._session_leaves.get(session_id, ())
            )
            for component in tree_core.components
        }

        slot = cache.session.slots.get(session_id)
        owner_pages: frozenset[int] = frozenset()
        if slot is not None and slot.is_holding_kv:
            owner_indices = env.req_to_token_pool.req_to_token[
                slot.req_pool_idx, : slot.kv.kv_allocated_len
            ]
            owner_pages = frozenset(
                int(page)
                for page in torch.unique(owner_indices // cache.page_size).tolist()
            )

        full_allocator = env.allocator.full_attn_allocator
        free_tensors = [full_allocator.free_pages]
        if len(full_allocator.release_pages) > 0:
            free_tensors.append(full_allocator.release_pages)
        free_full_pages = frozenset(
            int(page) for page in torch.cat(free_tensors).tolist()
        )

        def expects_device_leaf(node: UnifiedTreeNode) -> bool:
            full_data = node.component_data[ComponentType.FULL]
            return (
                node.id in arena_node_ids
                and full_data.value is not None
                and all(data.lock_ref == 0 for data in node.component_data)
                and all(
                    child.component_data[ComponentType.FULL].value is None
                    for child in node.children.values()
                )
            )

        def expects_host_leaf(node: UnifiedTreeNode) -> bool:
            full_data = node.component_data[ComponentType.FULL]
            return (
                node.id in arena_node_ids
                and full_data.value is None
                and full_data.host_value is not None
                and all(data.host_lock_ref == 0 for data in node.component_data)
                and len(node.children) == 0
            )

        return _PrivatePathLifecycleState(
            private_node_ids=private_node_ids,
            arena_node_ids=arena_node_ids,
            component_state=component_state,
            device_lru_ids=device_lru_ids,
            host_lru_ids=host_lru_ids,
            expected_device_leaf_ids=frozenset(
                node.id for node in private_nodes if expects_device_leaf(node)
            ),
            expected_host_leaf_ids=frozenset(
                node.id for node in private_nodes if expects_host_leaf(node)
            ),
            device_leaf_ids=frozenset(
                node.id for node in tree_core.evictable_device_leaves
            ),
            host_leaf_ids=frozenset(
                node.id for node in tree_core.evictable_host_leaves
            ),
            session_leaf_ids=session_leaf_ids,
            owner_pages=owner_pages,
            free_full_pages=free_full_pages,
        )

    def _exercise_private_host_path_lifecycle(
        self,
    ) -> dict[str, _PrivatePathLifecycleState]:
        """Demote, restore, adopt, and close one exact SWA session path.

        :returns: Bookkeeping snapshots at each externally meaningful boundary.
        """
        session_id = "ownership-session"
        page_size = 64
        exact_tokens = page_size * 2 + 1
        token_ids = array("q", range(exact_tokens + 1))
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        self._admit_streaming_session(env, session_id, exact_tokens)
        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                session_id,
                token_ids,
                extra_key="ownership-namespace",
                cache_salt="ownership-tenant",
                priority=0,
            ),
            exact_tokens,
        )
        self.assertEqual(
            cache.commit_streaming_session_demotion(session_id),
            exact_tokens,
        )
        demoted_state = cache.session.demoted[session_id]
        private_nodes: list[UnifiedTreeNode] = []
        node = cache.tree_core.node_by_id(demoted_state.last_node)
        while node.is_session_private:
            private_nodes.append(node)
            node = node.parent
        private_nodes.reverse()
        stable_private_nodes = tuple(private_nodes)
        self.assertGreater(len(stable_private_nodes), 0)

        cache.sanity_check()
        states = {
            "demoted": self._capture_private_path_lifecycle_state(
                env, stable_private_nodes, session_id
            )
        }

        match_req = self._streaming_match_req(session_id)
        owner_match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    extra_key="ownership-namespace",
                    cache_salt="ownership-tenant",
                ),
                req=match_req,
            )
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
        cache.sanity_check()
        states["restored"] = self._capture_private_path_lifecycle_state(
            env, stable_private_nodes, session_id
        )

        restored_req = SimpleNamespace(rid="restored-owner", req_pool_idx=None)
        self.assertIsNotNone(env.req_to_token_pool.alloc([restored_req]))
        env.req_to_token_pool.write(
            (restored_req.req_pool_idx, slice(0, exact_tokens)),
            load_result.new_full_device_indices,
        )
        restored_lock = cache.inc_lock_ref(demoted_state.last_node)
        cache.session.slots[session_id] = SessionSlot(
            req_pool_idx=restored_req.req_pool_idx,
            kv_committed_len=exact_tokens,
            kv=ReqKvInfo(
                kv_allocated_len=exact_tokens,
                swa_evicted_seqlen=demoted_state.swa_evicted_seqlen,
            ),
            last_node=demoted_state.last_node,
            cache_protected_len=exact_tokens,
            tree_protected_len=exact_tokens,
            swa_uuid_for_lock=restored_lock.swa_uuid_for_lock,
            skip_lock_node_ids=restored_lock.skip_lock_node_ids,
        )
        self.assertTrue(cache.session._release_demoted_state(session_id))
        cache.sanity_check()
        states["adopted"] = self._capture_private_path_lifecycle_state(
            env, stable_private_nodes, session_id
        )

        cache.release_radix_session(session_id)
        cache.release_session(session_id)
        cache.sanity_check()
        states["closed"] = self._capture_private_path_lifecycle_state(
            env, stable_private_nodes, session_id
        )
        return states

    def test_private_host_path_device_lru_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        demoted = states["demoted"]
        restored = states["restored"]
        for (node_id, component_type), state in restored.component_state.items():
            device_present = state[0]
            if component_type is ComponentType.FULL:
                self.assertNotIn(node_id, restored.device_lru_ids[component_type])
            elif device_present:
                self.assertIn(node_id, restored.device_lru_ids[component_type])
            else:
                self.assertNotIn(node_id, restored.device_lru_ids[component_type])
        for boundary in (demoted, states["adopted"], states["closed"]):
            for component_type, lru_ids in boundary.device_lru_ids.items():
                self.assertTrue(
                    boundary.private_node_ids.isdisjoint(lru_ids),
                    component_type,
                )

    def test_private_host_path_host_lru_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        for boundary_name in ("demoted", "restored"):
            boundary = states[boundary_name]
            for (node_id, component_type), state in boundary.component_state.items():
                device_present, host_present, _, host_lock_ref, _, _ = state
                if component_type is ComponentType.FULL:
                    self.assertNotIn(node_id, boundary.host_lru_ids[component_type])
                    continue
                host_lru_eligible = (
                    not device_present and host_present and host_lock_ref == 0
                )
                self.assertEqual(
                    node_id in boundary.host_lru_ids[component_type],
                    host_lru_eligible,
                )
        for boundary_name in ("adopted", "closed"):
            boundary = states[boundary_name]
            for lru_ids in boundary.host_lru_ids.values():
                self.assertTrue(boundary.private_node_ids.isdisjoint(lru_ids))

    def test_private_host_path_leaf_set_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        for boundary in states.values():
            self.assertEqual(
                boundary.private_node_ids & boundary.device_leaf_ids,
                boundary.expected_device_leaf_ids,
            )
            self.assertEqual(
                boundary.private_node_ids & boundary.host_leaf_ids,
                boundary.expected_host_leaf_ids,
            )

    def test_private_host_path_lock_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        demoted_host_locks = sum(
            state[3] for state in states["demoted"].component_state.values()
        )
        restored_host_locks = sum(
            state[3] for state in states["restored"].component_state.values()
        )
        self.assertGreater(demoted_host_locks, 0)
        self.assertEqual(restored_host_locks, demoted_host_locks)
        for boundary_name in ("demoted", "restored"):
            for state in states[boundary_name].component_state.values():
                self.assertEqual(state[3], len(state[5]))
        for boundary_name in ("adopted", "closed"):
            self.assertEqual(
                sum(
                    state[2] + state[3]
                    for state in states[boundary_name].component_state.values()
                ),
                0,
            )

    def test_private_host_path_session_tracker_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        for boundary_name in ("demoted", "restored"):
            boundary = states[boundary_name]
            for component_type, leaf_ids in boundary.session_leaf_ids.items():
                self.assertGreater(len(leaf_ids), 0, component_type)
                self.assertTrue(leaf_ids <= boundary.private_node_ids)
                for leaf_id in leaf_ids:
                    self.assertGreater(
                        boundary.component_state[(leaf_id, component_type)][4],
                        0,
                    )
        for boundary_name in ("adopted", "closed"):
            self.assertTrue(
                all(
                    len(leaf_ids) == 0
                    for leaf_ids in states[boundary_name].session_leaf_ids.values()
                )
            )

    def test_private_host_path_page_owner_registration(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        self.assertEqual(states["demoted"].owner_pages, frozenset())
        self.assertEqual(states["restored"].owner_pages, frozenset())
        adopted_owner_pages = states["adopted"].owner_pages
        self.assertGreater(len(adopted_owner_pages), 0)
        self.assertTrue(
            adopted_owner_pages.isdisjoint(states["adopted"].free_full_pages)
        )
        self.assertEqual(states["closed"].owner_pages, frozenset())
        self.assertTrue(adopted_owner_pages <= states["closed"].free_full_pages)

    def test_private_host_path_sanity_state_lifecycle(self) -> None:
        states = self._exercise_private_host_path_lifecycle()

        for boundary_name in ("demoted", "restored"):
            boundary = states[boundary_name]
            self.assertTrue(boundary.private_node_ids <= boundary.arena_node_ids)
        for boundary_name in ("adopted", "closed"):
            boundary = states[boundary_name]
            self.assertTrue(
                boundary.private_node_ids.isdisjoint(boundary.arena_node_ids)
            )

    def test_close_during_load_back_is_typed_and_preserves_both_owners(self) -> None:
        session_id = "close-during-load-back"
        page_size = 64
        exact_tokens = page_size * 2 + 1
        token_ids = array("q", range(exact_tokens + 1))
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        self._admit_streaming_session(env, session_id, exact_tokens)
        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                session_id,
                token_ids,
                extra_key="close-load-namespace",
                cache_salt="close-load-tenant",
                priority=0,
            ),
            exact_tokens,
        )
        cache.commit_streaming_session_demotion(session_id)

        match_req = self._streaming_match_req(session_id)
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    extra_key="close-load-namespace",
                    cache_salt="close-load-tenant",
                ),
                req=match_req,
            )
        )
        match_req.last_node = match.last_device_node
        match_req.prefix_indices = match.device_indices
        match_req.swa_host_hit_length = match.swa_host_hit_length
        match_req.mamba_host_hit_length = 0
        match_req.mamba_pool_idx = None
        self.assertIsNotNone(
            cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=match.best_match_node,
                    host_hit_length=match.host_hit_length,
                    req=match_req,
                )
            )
        )
        cache.ready_to_load_host_cache()

        with self.assertRaises(StreamingSessionBusyError):
            cache.release_session(session_id)

        self.assertTrue(cache.is_streaming_session_demoted(session_id))
        self.assertGreater(len(cache.ongoing_load_back), 0)
        cache.sanity_check()

        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)
        cache.release_session(session_id)
        cache.release_radix_session(session_id)
        cache.sanity_check()
        self.assertFalse(cache.is_streaming_session_demoted(session_id))

    def test_demotion_publication_failures_restore_the_prepared_slot(self) -> None:
        for failure_point in ("session_refs", "transition"):
            with self.subTest(failure_point=failure_point):
                session_id = f"demotion-{failure_point}-failure"
                page_size = 64
                exact_tokens = page_size * 2 + 1
                token_ids = array("q", range(exact_tokens + 1))
                env = self._build_cache(
                    hicache_ratio=1.0,
                    disable=False,
                    enable_session_radix_cache=True,
                    page_size=page_size,
                    sliding_window_size=page_size,
                )
                cache = env.cache
                self._admit_streaming_session(env, session_id, exact_tokens)
                available_before = cache.host_pool_group.available_size()
                self.assertEqual(
                    cache.prepare_streaming_session_demotion(
                        session_id,
                        token_ids,
                        extra_key="failure-namespace",
                        cache_salt="failure-tenant",
                        priority=0,
                    ),
                    exact_tokens,
                )
                if failure_point == "session_refs":
                    target = cache.session_refs
                    attribute = "register_streaming_session_frontier"
                else:
                    target = cache.session
                    attribute = "transition_to_demoted"

                with (
                    mock.patch.object(
                        target,
                        attribute,
                        side_effect=RuntimeError("injected publication failure"),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "injected publication failure",
                    ),
                ):
                    cache.commit_streaming_session_demotion(session_id)

                self.assertIn(session_id, cache.session.slots)
                self.assertNotIn(session_id, cache.session.demoted)
                self.assertNotIn(
                    session_id,
                    cache._pending_streaming_session_demotions,
                )
                self.assertFalse(
                    any(
                        node.private_session_id == session_id
                        for node in cache.tree_core._node_arena.values()
                    )
                )
                self.assertEqual(
                    cache.host_pool_group.available_size(),
                    available_before,
                )
                cache.sanity_check()

    def test_demotion_retirement_failure_keeps_the_stage(self) -> None:
        """A failure after source retirement began is indeterminate, never rolled back."""
        session_id = "demotion-retirement-failure"
        page_size = 64
        exact_tokens = page_size * 2 + 1
        token_ids = array("q", range(exact_tokens + 1))
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        self._admit_streaming_session(env, session_id, exact_tokens)
        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                session_id,
                token_ids,
                extra_key="retire-namespace",
                cache_salt="retire-tenant",
                priority=0,
            ),
            exact_tokens,
        )

        with (
            mock.patch.object(
                cache.session,
                "_release_slot_resources",
                side_effect=RuntimeError("injected retirement failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected retirement failure"),
        ):
            cache.commit_streaming_session_demotion(session_id)

        self.assertTrue(cache.session.demotion_retirement_started(session_id))
        self.assertIn(session_id, cache._pending_streaming_session_demotions)
        self.assertNotIn(session_id, cache.session.demoted)
        self.assertIn(session_id, cache.session.slots)

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

    def _dump_tree(self, cache) -> str:
        """Render every node's ownership for a failing shared-prefix step."""
        lines: list[str] = []
        stack = [(cache.tree_core.root_node, 0)]
        while stack:
            node, depth = stack.pop()
            if node is not cache.tree_core.root_node:
                full = node.component_data[ComponentType.FULL]
                swa = node.component_data[ComponentType.SWA]
                lines.append(
                    "  " * depth
                    + f"node {node.id} len={len(node.key)} private={node.private_session_id} "
                    f"full(dev={full.value is not None}, host={full.host_value is not None}, "
                    f"lock={full.lock_ref}, hlock={full.host_lock_ref}) "
                    f"swa(dev={swa.value is not None}, host={swa.host_value is not None}, "
                    f"lock={swa.lock_ref}, hlock={swa.host_lock_ref}) children={len(node.children)}"
                )
            for child in node.children.values():
                stack.append((child, depth + 1))
        return "\n".join(lines)

    def test_second_reload_of_a_shared_prefix_keeps_the_prefix(self) -> None:
        """Two demoted sharers must both restore the whole path (E3 F16)."""
        page_size = 64
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        tip = 2 * page_size
        prefix_req, prefix_indices = self._admit_req(env, page_size)
        self._seed_pool(env.target_pool, prefix_indices, base=5000)
        prefix_key = RadixKey(
            array("q", range(page_size + 1)),
            extra_key="namespace",
            cache_salt="tenant",
            is_bigram=True,
        )
        planted = cache.insert(
            InsertParams(key=prefix_key, value=prefix_indices, trigger_backup=True)
        )
        env.req_to_token_pool.free(prefix_req)
        prefix_node = planted.last_device_node
        cache.writing_check(write_back=True)
        self.assertTrue(cache.tree_core.node_by_id(prefix_node).backuped)

        for session_id in ("a", "b"):
            req, tail_indices = self._admit_streaming_session(env, session_id, page_size)
            row = req.req_pool_idx
            env.req_to_token_pool.write((row, slice(0, page_size)), prefix_indices)
            env.req_to_token_pool.write((row, slice(page_size, tip)), tail_indices)
            lock = cache.inc_lock_ref(prefix_node)
            slot = cache.session.slots[session_id]
            slot.last_node = prefix_node
            slot.cache_protected_len = page_size
            slot.tree_protected_len = page_size
            slot.kv_committed_len = tip
            slot.kv = ReqKvInfo(kv_allocated_len=tip)
            slot.swa_uuid_for_lock = lock.swa_uuid_for_lock
            slot.skip_lock_node_ids = lock.skip_lock_node_ids

        session_key = array("q", range(tip + 1))
        for session_id in ("a", "b"):
            self.assertEqual(
                cache.prepare_streaming_session_demotion(
                    session_id,
                    session_key,
                    extra_key="namespace",
                    cache_salt="tenant",
                    priority=0,
                ),
                tip,
            )
            self.assertEqual(cache.commit_streaming_session_demotion(session_id), tip)
        cache.sanity_check()
        after_demotions = self._dump_tree(cache)
        prefix_full = cache.tree_core.node_by_id(prefix_node).component_data[
            ComponentType.FULL
        ]
        if prefix_full.value is not None:
            evicted = cache.evict(EvictParams(num_tokens=page_size))
            self.assertGreaterEqual(evicted.num_tokens_evicted, page_size, after_demotions)
        self.assertIsNone(
            cache.tree_core.node_by_id(prefix_node).component_data[ComponentType.FULL].value,
            self._dump_tree(cache),
        )
        cache.sanity_check()
        full_available_before_reloads = env.allocator.full_available_size()

        def match(session_id: str):
            return cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        session_key,
                        extra_key="namespace",
                        cache_salt="tenant",
                        is_bigram=True,
                    ),
                    req=self._streaming_match_req(session_id),
                )
            )

        first = match("a")
        self.assertEqual(first.full_kv_hit_length, tip, self._dump_tree(cache))
        load_req = SimpleNamespace(
            last_node=first.last_device_node,
            prefix_indices=first.device_indices,
            swa_host_hit_length=first.swa_host_hit_length,
            mamba_host_hit_length=0,
            mamba_pool_idx=None,
        )
        load_result = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=first.best_match_node,
                host_hit_length=first.host_hit_length,
                req=load_req,
            )
        )
        restored = torch.cat([first.device_indices, load_result.new_full_device_indices])
        self.assertEqual(len(restored), tip, self._dump_tree(cache))
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)
        restored_lock = cache.inc_lock_ref(first.best_match_node)
        resumed = SimpleNamespace(rid="resumed-a", req_pool_idx=None, seqlen=tip + 1)
        self.assertIsNotNone(env.req_to_token_pool.alloc([resumed]))
        env.req_to_token_pool.write((resumed.req_pool_idx, slice(0, tip)), restored)
        # A resumed request inherits the demoted SWA watermark on admission.
        restored_watermark = cache.session.demoted["a"].swa_evicted_seqlen
        cache.session.slots["a"] = SessionSlot(
            req_pool_idx=resumed.req_pool_idx,
            kv_committed_len=tip,
            kv=ReqKvInfo(kv_allocated_len=tip, swa_evicted_seqlen=restored_watermark),
            last_node=first.best_match_node,
            cache_protected_len=tip,
            tree_protected_len=tip,
            swa_uuid_for_lock=restored_lock.swa_uuid_for_lock,
            skip_lock_node_ids=restored_lock.skip_lock_node_ids,
        )
        self.assertTrue(cache.session._release_demoted_state("a"))
        cache.sanity_check()
        after_first_reload = self._dump_tree(cache)
        adopted = cache.session.slots["a"]
        self.assertEqual(adopted.tree_protected_len, page_size, after_first_reload)
        self.assertEqual(adopted.last_node, prefix_node, after_first_reload)
        self.assertIsNotNone(
            cache.tree_core.node_by_id(prefix_node).component_data[ComponentType.FULL].value,
            after_first_reload,
        )

        second = match("b")
        second_summary = (
            f"full_kv_hit_length={second.full_kv_hit_length} "
            f"device_indices={len(second.device_indices)} "
            f"host_hit_length={second.host_hit_length} "
            f"swa_host_hit_length={second.swa_host_hit_length} "
            f"last_device_node={second.last_device_node} "
            f"best_match_node={second.best_match_node} "
            f"cache_protected_len={second.cache_protected_len}\n{after_first_reload}"
        )
        self.assertEqual(second.full_kv_hit_length, tip, second_summary)
        self.assertEqual(len(second.device_indices), page_size, second_summary)
        self.assertEqual(second.host_hit_length, page_size, second_summary)
        second_load_req = SimpleNamespace(
            last_node=second.last_device_node,
            prefix_indices=second.device_indices,
            swa_host_hit_length=second.swa_host_hit_length,
            mamba_host_hit_length=0,
            mamba_pool_idx=None,
        )
        second_load = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=second.best_match_node,
                host_hit_length=second.host_hit_length,
                req=second_load_req,
            )
        )
        self.assertEqual(len(second_load.new_full_device_indices), page_size, after_first_reload)
        self.assertEqual(second_load.cache_protected_len, tip, after_first_reload)
        cache.ready_to_load_host_cache()
        ack = cache.cache_controller.ack_load_queue[0]
        ack.finish_event.synchronize()
        cache.loading_check(finish_count=1)
        cache.sanity_check()
        self.assertEqual(
            env.allocator.full_available_size(),
            full_available_before_reloads - 3 * page_size,
            self._dump_tree(cache),
        )

        cache.release_radix_session("b")
        cache.release_session("b")
        cache.release_radix_session("a")
        cache.release_session("a")
        cache.sanity_check()

    def test_device_pressure_keeps_a_demoted_path_out_of_the_host_lru(self) -> None:
        """Evicting a shared prefix's window pages must not expose locked host pages (E3 F17)."""
        page_size = 64
        env = self._build_cache(
            hicache_ratio=1.0,
            disable=False,
            enable_session_radix_cache=True,
            page_size=page_size,
            sliding_window_size=page_size,
        )
        cache = env.cache
        tip = 2 * page_size
        prefix_req, prefix_indices = self._admit_req(env, page_size)
        self._seed_pool(env.target_pool, prefix_indices, base=5000)
        prefix_key = RadixKey(
            array("q", range(page_size + 1)),
            extra_key="namespace",
            cache_salt="tenant",
            is_bigram=True,
        )
        planted = cache.insert(
            InsertParams(key=prefix_key, value=prefix_indices, trigger_backup=True)
        )
        env.req_to_token_pool.free(prefix_req)
        prefix_node = planted.last_device_node
        cache.writing_check(write_back=True)

        req, tail_indices = self._admit_streaming_session(env, "a", page_size)
        row = req.req_pool_idx
        env.req_to_token_pool.write((row, slice(0, page_size)), prefix_indices)
        env.req_to_token_pool.write((row, slice(page_size, tip)), tail_indices)
        lock = cache.inc_lock_ref(prefix_node)
        slot = cache.session.slots["a"]
        slot.last_node = prefix_node
        slot.cache_protected_len = page_size
        slot.tree_protected_len = page_size
        slot.kv_committed_len = tip
        slot.kv = ReqKvInfo(kv_allocated_len=tip)
        slot.swa_uuid_for_lock = lock.swa_uuid_for_lock
        slot.skip_lock_node_ids = lock.skip_lock_node_ids
        self.assertEqual(
            cache.prepare_streaming_session_demotion(
                "a",
                array("q", range(tip + 1)),
                extra_key="namespace",
                cache_salt="tenant",
                priority=0,
            ),
            tip,
        )
        self.assertEqual(cache.commit_streaming_session_demotion("a"), tip)
        cache.sanity_check()

        def locked_host_lru_nodes() -> list[int]:
            lru = cache.tree_core.host_lru_lists[ComponentType.SWA]
            return sorted(
                node.id
                for node in lru.cache.values()
                if node.component_data[ComponentType.SWA].host_lock_ref > 0
                or node.is_session_private
            )

        self.assertEqual(locked_host_lru_nodes(), [], self._dump_tree(cache))
        evicted = cache.evict(EvictParams(num_tokens=page_size, swa_num_tokens=page_size))
        cache.sanity_check()
        self.assertEqual(
            locked_host_lru_nodes(),
            [],
            f"evicted={evicted}\n" + self._dump_tree(cache),
        )
        cache.release_radix_session("a")
        cache.release_session("a")
        cache.sanity_check()

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
        protected_swa_nodes = [
            node
            for node in cache.tree_core._collect_all_nodes()
            if node.component_data[ComponentType.SWA].host_lock_ref > 0
        ]
        self.assertGreater(len(protected_swa_nodes), 0)
        self.assertTrue(
            all(
                not cache.tree_core.host_lru_lists[ComponentType.SWA].in_list(node)
                for node in protected_swa_nodes
            )
        )
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
        cache.sanity_check()

        slot = cache.session.slots["session"]
        self.assertEqual(slot.tree_protected_len, 0)
        self.assertEqual(slot.kv.swa_evicted_seqlen, page_size * 2)
        full_consumption = full_available_baseline - allocator.full_available_size()
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
                    allocator.translate_loc_from_full_to_swa(released_full_slots) == 0
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    allocator.translate_loc_from_full_to_swa(retained_full_slots) > 0
                ).all()
            )
        )

        cache.release_radix_session("session")
        cache.release_session("session")
        cache.sanity_check()
        self.assertTrue(
            all(
                component_data.host_lock_ref == 0
                for node in cache.tree_core._collect_all_nodes()
                for component_data in node.component_data
            )
        )
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
