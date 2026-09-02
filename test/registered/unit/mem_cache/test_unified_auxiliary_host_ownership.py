"""CPU regressions for unified auxiliary host ownership."""

import unittest
from array import array
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    HostLockFootprint,
    HostLockOwner,
    HostLockOwnerKind,
    HostLockRange,
    IncLockRefResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentData,
    ComponentType,
    EvictLayer,
    PreparePrefetchResult,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingLoadBack,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _AuxiliaryTreeCore(UnifiedTreeCore):
    """Minimal tree state needed by auxiliary ownership transitions."""

    component_types = (
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    )

    def __init__(self) -> None:
        self.page_size = 1
        self.root_node = UnifiedTreeNode(self.component_types)
        self.root_node.key = RadixKey(array("q"))
        self.host_lru_lists = {
            component_type: UnifiedLRUList(
                component_type,
                self.component_types,
                use_host_ptr=True,
            )
            for component_type in self.component_types
        }
        self.lru_lists = {
            component_type: UnifiedLRUList(component_type, self.component_types)
            for component_type in self.component_types
        }
        self.component_evictable_size_ = defaultdict(int)
        self.component_protected_size_ = defaultdict(int)
        self.components_by_type = {}
        self.components = ()
        self._node_arena = {self.root_node.id: self.root_node}
        self.evictable_device_leaves = set()
        self.evictable_host_leaves = set()
        self.full_host_duplicates = {}
        self.is_write_back = False
        self.enable_session_radix_cache = False

    def add_child(self, token_ids: list[int]) -> UnifiedTreeNode:
        """Create one root child spanning ``token_ids``."""
        node = UnifiedTreeNode(self.component_types)
        node.parent = self.root_node
        node.key = RadixKey(array("q", token_ids))
        self.root_node.children[token_ids[0]] = node
        self._node_arena[node.id] = node
        return node

    def _reconcile_auxiliary_host_lru(
        self,
        node: UnifiedTreeNode,
        component_type: ComponentType,
    ) -> None:
        UnifiedTreeCore._reconcile_auxiliary_host_lru(self, node, component_type)

    @staticmethod
    def _auxiliary_host_lru_eligible(component_data: ComponentData) -> bool:
        """Apply the production host-LRU eligibility predicate."""
        return UnifiedTreeCore._auxiliary_host_lru_eligible(component_data)

    def component_logical_length(
        self,
        node: UnifiedTreeNode,
        component_type: ComponentType,
        *,
        host: bool = False,
    ) -> int:
        """Return the physical length used by these synthetic components."""
        component_data = node.component_data[component_type]
        value = component_data.host_value if host else component_data.value
        assert value is not None
        return len(value)

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """The focused tests do not maintain Full leaf-set bookkeeping."""


class TestUnifiedAuxiliaryHostOwnership(unittest.TestCase):
    def _build_component(
        self,
        component_type: ComponentType,
        *,
        length: int = 8,
    ) -> tuple[_AuxiliaryTreeCore, TreeComponent, UnifiedTreeNode]:
        core = _AuxiliaryTreeCore()
        node = core.add_child(list(range(1, length + 1)))
        full_data = node.component_data[BASE_COMPONENT_TYPE]
        full_data.value = torch.arange(length)
        full_data.host_value = torch.arange(length)
        component_data = node.component_data[component_type]
        component_data.value = torch.arange(length)
        component_data.host_value = torch.arange(length)
        core.component_evictable_size_[component_type] = length
        core.lru_lists[component_type].insert_mru(node)

        component_class = (
            SWAComponent if component_type is ComponentType.SWA else MambaComponent
        )
        component = object.__new__(component_class)
        component.tree_core = core
        if component_type is ComponentType.SWA:
            component.sliding_window_size = length
        full_component = object.__new__(FullComponent)
        full_component.tree_core = core
        core.components_by_type = {
            ComponentType.FULL: full_component,
            component_type: component,
        }
        core.components = (full_component, component)
        return core, component, node

    def _split_child(
        self,
        core: _AuxiliaryTreeCore,
        child: UnifiedTreeNode,
        split_len: int,
    ) -> UnifiedTreeNode:
        """Split one node through the production tree-core transition."""
        original_key = child.key
        assert original_key is not None
        new_parent, action = core._split_node(original_key, child, split_len)
        self.assertIsNone(action)
        return new_parent

    def test_protected_device_eviction_stays_out_of_auxiliary_host_lru(self) -> None:
        for component_type in (ComponentType.SWA, ComponentType.MAMBA):
            with self.subTest(component_type=component_type):
                core, component, node = self._build_component(component_type)
                lock_result = component.acquire_component_lock(
                    node,
                    IncLockRefResult(host_lock_id=1),
                    lock_host=True,
                )
                component_data = node.component_data[component_type]
                self.assertEqual(component_data.host_lock_ref, 1)
                self.assertFalse(core.host_lru_lists[component_type].in_list(node))

                component.evict_component(
                    node,
                    defaultdict(list),
                    defaultdict(list),
                    target=EvictLayer.DEVICE,
                )

                self.assertIsNone(component_data.value)
                self.assertIsNotNone(component_data.host_value)
                self.assertEqual(component_data.host_lock_ref, 1)
                self.assertFalse(core.host_lru_lists[component_type].in_list(node))

                component.release_component_lock(
                    node,
                    lock_result.to_dec_params(),
                    lock_host=True,
                )
                self.assertEqual(component_data.host_lock_ref, 0)
                self.assertIsNone(component_data.host_lock_ids)
                self.assertTrue(core.host_lru_lists[component_type].in_list(node))

    def test_full_host_lock_treats_the_root_as_a_non_owner(self) -> None:
        core = _AuxiliaryTreeCore()
        core.is_write_back = True
        component = object.__new__(FullComponent)
        component.tree_core = core
        core.components_by_type = {ComponentType.FULL: component}
        core.components = (component,)
        core._next_host_lock_id = 1

        lock_result = core.inc_host_lock_ref(core.root_node.id)

        self.assertEqual(lock_result.host_lock_footprints, {})
        self.assertIsNone(lock_result.host_lock_id)
        self.assertEqual(
            core.root_node.component_data[ComponentType.FULL].host_lock_ref,
            0,
        )

    def test_host_locked_swa_split_preserves_window_ownership(self) -> None:
        core, component, child = self._build_component(ComponentType.SWA)
        child_data = child.component_data[ComponentType.SWA]
        child_data.value = None
        core.lru_lists[ComponentType.SWA].remove_node(child)
        core.component_evictable_size_[ComponentType.SWA] = 0
        core._reconcile_auxiliary_host_lru(child, ComponentType.SWA)

        lock_result = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        )
        lock_params = lock_result.to_dec_params()
        lock_id = lock_params.host_lock_id
        self.assertIsNotNone(lock_id)

        new_parent = self._split_child(core, child, 4)

        parent_data = new_parent.component_data[ComponentType.SWA]
        self.assertEqual(parent_data.host_lock_ref, 1)
        self.assertEqual(child_data.host_lock_ref, 1)
        self.assertEqual(parent_data.host_lock_ids, {lock_id})
        self.assertEqual(child_data.host_lock_ids, {lock_id})
        self.assertEqual(
            component.host_lock_nodes(child, lock_params),
            (child, new_parent),
        )
        self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(new_parent))
        self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(child))

        parent_data.host_lock_ref = 0
        with self.assertRaisesRegex(AssertionError, "counter disagrees"):
            component.release_component_lock(child, lock_params, lock_host=True)
        self.assertEqual(child_data.host_lock_ref, 1)
        parent_data.host_lock_ref = 1

        parent_data.host_lock_ref = 0
        parent_data.host_lock_ids = None
        with self.assertRaisesRegex(AssertionError, "does not carry"):
            component.release_component_lock(child, lock_params, lock_host=True)
        self.assertEqual(child_data.host_lock_ref, 1)
        parent_data.host_lock_ref = 1
        parent_data.host_lock_ids = {lock_id}
        component.release_component_lock(child, lock_params, lock_host=True)

        self.assertEqual(parent_data.host_lock_ref, 0)
        self.assertEqual(child_data.host_lock_ref, 0)
        self.assertIsNone(parent_data.host_lock_ids)
        self.assertIsNone(child_data.host_lock_ids)
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(new_parent))
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(child))

    def test_host_locked_full_split_preserves_transfer_ownership(self) -> None:
        core = _AuxiliaryTreeCore()
        core.is_write_back = False
        child = core.add_child(list(range(1, 9)))
        child_data = child.component_data[ComponentType.FULL]
        child_data.value = torch.arange(8)
        child_data.host_value = torch.arange(8)
        component = object.__new__(FullComponent)
        component.tree_core = core
        core.components_by_type[ComponentType.FULL] = component
        core.components = (component,)

        lock_result = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        )
        lock_params = lock_result.to_dec_params()
        lock_id = lock_params.host_lock_id
        self.assertIsNotNone(lock_id)

        new_parent = self._split_child(core, child, 4)

        parent_data = new_parent.component_data[ComponentType.FULL]
        self.assertEqual(parent_data.host_lock_ref, 1)
        self.assertEqual(child_data.host_lock_ref, 1)
        self.assertEqual(parent_data.host_lock_ids, {lock_id})
        self.assertEqual(child_data.host_lock_ids, {lock_id})
        self.assertEqual(
            component.host_lock_nodes(child, lock_params),
            (child, new_parent),
        )

        parent_data.host_lock_ref = 0
        with self.assertRaisesRegex(AssertionError, "counter disagrees"):
            component.release_component_lock(child, lock_params, lock_host=True)
        self.assertEqual(child_data.host_lock_ref, 1)
        parent_data.host_lock_ref = 1

        parent_data.host_lock_ref = 0
        parent_data.host_lock_ids = None
        with self.assertRaisesRegex(AssertionError, "does not carry"):
            component.release_component_lock(child, lock_params, lock_host=True)
        self.assertEqual(child_data.host_lock_ref, 1)
        parent_data.host_lock_ref = 1
        parent_data.host_lock_ids = {lock_id}
        component.release_component_lock(child, lock_params, lock_host=True)

        self.assertEqual(parent_data.host_lock_ref, 0)
        self.assertEqual(child_data.host_lock_ref, 0)
        self.assertIsNone(parent_data.host_lock_ids)
        self.assertIsNone(child_data.host_lock_ids)

    def test_host_locked_mamba_split_follows_the_terminal_child(self) -> None:
        core, component, child = self._build_component(ComponentType.MAMBA)
        lock_params = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()
        new_parent = self._split_child(core, child, 4)

        self.assertEqual(component.host_lock_nodes(child, lock_params), (child,))
        self.assertEqual(
            new_parent.component_data[ComponentType.MAMBA].host_lock_ref,
            0,
        )
        component.release_component_lock(child, lock_params, lock_host=True)
        self.assertEqual(
            child.component_data[ComponentType.MAMBA].host_lock_ref,
            0,
        )

    def test_overlapping_full_host_owners_remain_distinct_after_split(self) -> None:
        core = _AuxiliaryTreeCore()
        core.is_write_back = False
        child = core.add_child(list(range(1, 9)))
        child_data = child.component_data[ComponentType.FULL]
        child_data.host_value = torch.arange(8)
        component = object.__new__(FullComponent)
        component.tree_core = core
        core.components_by_type[ComponentType.FULL] = component
        core.components = (component,)

        first = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()
        new_parent = self._split_child(core, child, 4)

        second = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=2),
            lock_host=True,
        ).to_dec_params()
        parent_data = new_parent.component_data[ComponentType.FULL]
        self.assertEqual(parent_data.host_lock_ids, {1})
        self.assertEqual(child_data.host_lock_ids, {1, 2})

        component.release_component_lock(child, first, lock_host=True)
        self.assertIsNone(parent_data.host_lock_ids)
        self.assertEqual(child_data.host_lock_ids, {2})
        self.assertEqual(child_data.host_lock_ref, 1)

        component.release_component_lock(child, second, lock_host=True)
        self.assertIsNone(child_data.host_lock_ids)
        self.assertEqual(child_data.host_lock_ref, 0)

    def test_overlapping_full_host_owners_release_newest_first(self) -> None:
        core = _AuxiliaryTreeCore()
        child = core.add_child(list(range(1, 9)))
        child_data = child.component_data[ComponentType.FULL]
        child_data.host_value = torch.arange(8)
        component = object.__new__(FullComponent)
        component.tree_core = core
        core.components_by_type = {ComponentType.FULL: component}
        core.components = (component,)

        first = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()
        new_parent = self._split_child(core, child, 4)
        second = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=2),
            lock_host=True,
        ).to_dec_params()

        component.release_component_lock(child, second, lock_host=True)
        self.assertEqual(
            new_parent.component_data[ComponentType.FULL].host_lock_ids,
            {1},
        )
        self.assertEqual(child_data.host_lock_ids, {1})
        component.release_component_lock(child, first, lock_host=True)
        self.assertIsNone(new_parent.component_data[ComponentType.FULL].host_lock_ids)
        self.assertIsNone(child_data.host_lock_ids)

    def test_overlapping_swa_host_owners_remain_distinct_after_split(self) -> None:
        core, component, child = self._build_component(ComponentType.SWA)
        child_data = child.component_data[ComponentType.SWA]
        child_data.value = None
        core.lru_lists[ComponentType.SWA].remove_node(child)
        core.component_evictable_size_[ComponentType.SWA] = 0
        core._reconcile_auxiliary_host_lru(child, ComponentType.SWA)
        first = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()
        new_parent = self._split_child(core, child, 4)

        component.sliding_window_size = 4
        second = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=2),
            lock_host=True,
        ).to_dec_params()
        parent_data = new_parent.component_data[ComponentType.SWA]
        self.assertEqual(parent_data.host_lock_ids, {1})
        self.assertEqual(child_data.host_lock_ids, {1, 2})

        component.release_component_lock(child, first, lock_host=True)
        self.assertIsNone(parent_data.host_lock_ids)
        self.assertEqual(child_data.host_lock_ids, {2})
        self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(child))
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(new_parent))

        component.release_component_lock(child, second, lock_host=True)
        self.assertIsNone(child_data.host_lock_ids)
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(child))

    def test_overlapping_swa_host_owners_release_newest_first(self) -> None:
        core, component, child = self._build_component(ComponentType.SWA)
        child_data = child.component_data[ComponentType.SWA]
        child_data.value = None
        core.lru_lists[ComponentType.SWA].remove_node(child)
        core.component_evictable_size_[ComponentType.SWA] = 0
        core._reconcile_auxiliary_host_lru(child, ComponentType.SWA)
        first = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()
        new_parent = self._split_child(core, child, 4)
        component.sliding_window_size = 4
        second = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=2),
            lock_host=True,
        ).to_dec_params()

        component.release_component_lock(child, second, lock_host=True)
        self.assertEqual(
            new_parent.component_data[ComponentType.SWA].host_lock_ids,
            {1},
        )
        self.assertEqual(child_data.host_lock_ids, {1})
        self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(new_parent))
        self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(child))
        component.release_component_lock(child, first, lock_host=True)
        self.assertIsNone(new_parent.component_data[ComponentType.SWA].host_lock_ids)
        self.assertIsNone(child_data.host_lock_ids)
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(new_parent))
        self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(child))

    def test_host_lock_owner_checker_rejects_all_mismatch_directions(self) -> None:
        core, component, node = self._build_component(ComponentType.SWA)
        lock_result = component.acquire_component_lock(
            node,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        )
        owner = HostLockOwner(
            kind=HostLockOwnerKind.DEMOTED_SESSION,
            owner_id="session",
            anchor_node_id=node.id,
            lock_params=lock_result.to_dec_params(),
        )
        reachable = {core.root_node, node}

        errors: list[str] = []
        UnifiedTreeCore._validate_host_lock_owners(
            core,
            [owner],
            reachable,
            errors.append,
        )
        self.assertEqual(errors, [])

        component_data = node.component_data[ComponentType.SWA]
        for actual, owners, expected_error in (
            (1, [], "orphan"),
            (0, [owner], "under-count"),
            (2, [owner], "over-count"),
        ):
            with self.subTest(expected_error=expected_error):
                component_data.host_lock_ref = actual
                errors = []
                UnifiedTreeCore._validate_host_lock_owners(
                    core,
                    owners,
                    reachable,
                    errors.append,
                )
                self.assertTrue(
                    any(expected_error in error for error in errors),
                    errors,
                )

        component_data.host_lock_ref = 1
        component_data.host_lock_ids = {999}
        errors = []
        UnifiedTreeCore._validate_host_lock_owners(
            core,
            [owner],
            reachable,
            errors.append,
        )
        self.assertTrue(
            any("owner-set mismatch" in error for error in errors),
            errors,
        )

    def test_host_lock_release_preflights_every_component_before_mutation(self) -> None:
        core = _AuxiliaryTreeCore()
        core.is_write_back = False
        node = core.add_child(list(range(1, 9)))
        full = object.__new__(FullComponent)
        full.tree_core = core
        swa = object.__new__(SWAComponent)
        swa.tree_core = core
        swa.sliding_window_size = 8
        core.components_by_type = {
            ComponentType.FULL: full,
            ComponentType.SWA: swa,
        }
        core.components = (full, swa)

        for component_type in (ComponentType.FULL, ComponentType.SWA):
            component_data = node.component_data[component_type]
            component_data.host_value = torch.arange(8)

        result = IncLockRefResult(host_lock_id=1)
        full.acquire_component_lock(node, result, lock_host=True)
        swa.acquire_component_lock(node, result, lock_host=True)
        params = result.to_dec_params()
        full_data = node.component_data[ComponentType.FULL]
        swa_data = node.component_data[ComponentType.SWA]

        swa_data.host_lock_ref = 0
        swa_data.host_lock_ids = None
        with self.assertRaisesRegex(AssertionError, "does not carry"):
            UnifiedTreeCore.dec_host_lock_ref(core, node.id, params)

        self.assertEqual(full_data.host_lock_ref, 1)
        self.assertEqual(full_data.host_lock_ids, {1})

    def test_host_lock_release_preflights_full_swa_and_mamba_as_one_unit(
        self,
    ) -> None:
        core = _AuxiliaryTreeCore()
        node = core.add_child(list(range(1, 9)))
        full = object.__new__(FullComponent)
        full.tree_core = core
        swa = object.__new__(SWAComponent)
        swa.tree_core = core
        swa.sliding_window_size = 8
        mamba = object.__new__(MambaComponent)
        mamba.tree_core = core
        core.components_by_type = {
            ComponentType.FULL: full,
            ComponentType.SWA: swa,
            ComponentType.MAMBA: mamba,
        }
        core.components = (full, swa, mamba)

        result = IncLockRefResult(host_lock_id=1)
        for component in core.components:
            data = node.component_data[component.component_type]
            data.host_value = torch.arange(8)
            component.acquire_component_lock(node, result, lock_host=True)
        params = result.to_dec_params()

        mamba_data = node.component_data[ComponentType.MAMBA]
        mamba_data.host_lock_ref = 0
        mamba_data.host_lock_ids = None
        with self.assertRaisesRegex(AssertionError, "does not carry"):
            core.dec_host_lock_ref(node.id, params)

        for component_type in (ComponentType.FULL, ComponentType.SWA):
            data = node.component_data[component_type]
            self.assertEqual(data.host_lock_ref, 1)
            self.assertEqual(data.host_lock_ids, {1})

    def test_full_host_receipt_records_residency_intent(self) -> None:
        for host_backed in (False, True):
            with self.subTest(host_backed=host_backed):
                core = _AuxiliaryTreeCore()
                core.is_write_back = True
                core._next_host_lock_id = 1
                node = core.add_child(list(range(1, 9)))
                data = node.component_data[ComponentType.FULL]
                data.value = torch.arange(8)
                if host_backed:
                    data.host_value = torch.arange(8)
                component = object.__new__(FullComponent)
                component.tree_core = core
                core.components_by_type = {ComponentType.FULL: component}
                core.components = (component,)

                params = core.inc_host_lock_ref(node.id).to_dec_params()
                footprint = params.host_lock_footprints[ComponentType.FULL]
                self.assertEqual(footprint.requires_host_value, host_backed)
                core.validate_host_lock_ref(node.id, params)
                core.dec_host_lock_ref(node.id, params)
                self.assertEqual(data.host_lock_ref, 0)
                self.assertIsNone(data.host_lock_ids)

    def test_host_lock_acquisition_rolls_back_earlier_components(self) -> None:
        class _FailingComponent:
            """Component stub that fails after Full has acquired its lock."""

            def acquire_component_lock(
                self, *args: object, **kwargs: object
            ) -> IncLockRefResult:
                raise RuntimeError("injected acquisition failure")

        core = _AuxiliaryTreeCore()
        core.is_write_back = False
        core._next_host_lock_id = 1
        node = core.add_child(list(range(1, 9)))
        full_data = node.component_data[ComponentType.FULL]
        full_data.host_value = torch.arange(8)
        full = object.__new__(FullComponent)
        full.tree_core = core
        core.components_by_type = {ComponentType.FULL: full}
        core.components = (full, _FailingComponent())

        with self.assertRaisesRegex(RuntimeError, "injected acquisition failure"):
            UnifiedTreeCore.inc_host_lock_ref(core, node.id)

        self.assertEqual(full_data.host_lock_ref, 0)
        self.assertIsNone(full_data.host_lock_ids)

    def test_owner_checker_does_not_derive_expected_node_from_owner_ids(self) -> None:
        core, component, node = self._build_component(ComponentType.SWA)
        sibling = core.add_child(list(range(20, 28)))
        sibling_data = sibling.component_data[ComponentType.SWA]
        sibling_data.host_value = torch.arange(8)
        lock_result = component.acquire_component_lock(
            node,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        )
        owner = HostLockOwner(
            kind=HostLockOwnerKind.DEMOTED_SESSION,
            owner_id="session",
            anchor_node_id=node.id,
            lock_params=lock_result.to_dec_params(),
        )

        node_data = node.component_data[ComponentType.SWA]
        node_data.host_lock_ref = 0
        node_data.host_lock_ids = None
        sibling_data.host_lock_ref = 1
        sibling_data.host_lock_ids = {1}
        errors: list[str] = []
        UnifiedTreeCore._validate_host_lock_owners(
            core,
            [owner],
            {core.root_node, node, sibling},
            errors.append,
        )

        self.assertTrue(any("under-count" in error for error in errors), errors)
        self.assertTrue(any("orphan" in error for error in errors), errors)

    def test_owner_checker_rejects_one_acquisition_split_across_lifecycles(
        self,
    ) -> None:
        core, component, node = self._build_component(ComponentType.SWA)
        result = component.acquire_component_lock(
            node,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        )
        params = result.to_dec_params()
        owners = [
            HostLockOwner(
                kind=HostLockOwnerKind.DEMOTED_SESSION,
                owner_id="session",
                anchor_node_id=node.id,
                lock_params=params,
            ),
            HostLockOwner(
                kind=HostLockOwnerKind.LOAD_BACK,
                owner_id=17,
                anchor_node_id=node.id,
                lock_params=params,
            ),
        ]

        errors: list[str] = []
        core._validate_host_lock_owners(
            owners,
            {core.root_node, node},
            errors.append,
        )

        self.assertTrue(any("claimed by both" in error for error in errors), errors)

    def test_session_ref_checker_rejects_unexplained_positive_counter(self) -> None:
        core = _AuxiliaryTreeCore()
        node = core.add_child(list(range(1, 9)))
        component = object.__new__(FullComponent)
        component.tree_core = core
        component._session_leaves = defaultdict(set)
        node.component_data[ComponentType.FULL].session_ref = 1

        errors: list[str] = []
        component.validate_session_state(
            {core.root_node, node},
            errors.append,
        )

        self.assertTrue(any("session_ref=1 expected=0" in error for error in errors))

    def test_repeated_production_splits_preserve_exact_swa_owner_footprint(
        self,
    ) -> None:
        core, component, child = self._build_component(ComponentType.SWA)
        child_data = child.component_data[ComponentType.SWA]
        child_data.value = None
        core.lru_lists[ComponentType.SWA].remove_node(child)
        core.component_evictable_size_[ComponentType.SWA] = 0
        core._reconcile_auxiliary_host_lru(child, ComponentType.SWA)
        params = component.acquire_component_lock(
            child,
            IncLockRefResult(host_lock_id=1),
            lock_host=True,
        ).to_dec_params()

        first_parent = self._split_child(core, child, 2)
        second_parent = self._split_child(core, child, 2)

        self.assertEqual(
            component.host_lock_nodes(child, params),
            (child, second_parent, first_parent),
        )
        for node in (first_parent, second_parent, child):
            data = node.component_data[ComponentType.SWA]
            self.assertEqual(data.host_lock_ref, 1)
            self.assertEqual(data.host_lock_ids, {1})
            self.assertFalse(core.host_lru_lists[ComponentType.SWA].in_list(node))

        core.dec_host_lock_ref(child.id, params)
        for node in (first_parent, second_parent, child):
            data = node.component_data[ComponentType.SWA]
            self.assertEqual(data.host_lock_ref, 0)
            self.assertIsNone(data.host_lock_ids)
            self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(node))

    def test_combined_host_owner_survives_repeated_production_splits(self) -> None:
        core = _AuxiliaryTreeCore()
        core._next_host_lock_id = 1
        child = core.add_child(list(range(1, 9)))
        full = object.__new__(FullComponent)
        full.tree_core = core
        swa = object.__new__(SWAComponent)
        swa.tree_core = core
        swa.sliding_window_size = 8
        mamba = object.__new__(MambaComponent)
        mamba.tree_core = core
        core.components_by_type = {
            ComponentType.FULL: full,
            ComponentType.SWA: swa,
            ComponentType.MAMBA: mamba,
        }
        core.components = (full, swa, mamba)
        for component_type in core.component_types:
            child.component_data[component_type].host_value = torch.arange(8)
        for component_type in (ComponentType.SWA, ComponentType.MAMBA):
            core.host_lru_lists[component_type].insert_mru(child)

        params = core.inc_host_lock_ref(child.id).to_dec_params()
        first_parent = self._split_child(core, child, 2)
        second_parent = self._split_child(core, child, 2)

        self.assertEqual(
            full.host_lock_nodes(child, params),
            (child, second_parent, first_parent),
        )
        self.assertEqual(
            swa.host_lock_nodes(child, params),
            (child, second_parent, first_parent),
        )
        self.assertEqual(mamba.host_lock_nodes(child, params), (child,))
        core.validate_host_lock_ref(child.id, params)
        core.dec_host_lock_ref(child.id, params)

        for node in (first_parent, second_parent, child):
            for component_type in (ComponentType.FULL, ComponentType.SWA):
                data = node.component_data[component_type]
                self.assertEqual(data.host_lock_ref, 0)
                self.assertIsNone(data.host_lock_ids)
            self.assertTrue(core.host_lru_lists[ComponentType.SWA].in_list(node))
        self.assertEqual(
            child.component_data[ComponentType.MAMBA].host_lock_ref,
            0,
        )
        self.assertIsNone(child.component_data[ComponentType.MAMBA].host_lock_ids)
        self.assertTrue(core.host_lru_lists[ComponentType.MAMBA].in_list(child))

    def test_device_only_full_host_owner_survives_split_and_release(self) -> None:
        core = _AuxiliaryTreeCore()
        core.is_write_back = True
        core._next_host_lock_id = 1
        child = core.add_child(list(range(1, 9)))
        child.component_data[ComponentType.FULL].value = torch.arange(8)
        full = object.__new__(FullComponent)
        full.tree_core = core
        core.components_by_type = {ComponentType.FULL: full}
        core.components = (full,)

        params = core.inc_host_lock_ref(child.id).to_dec_params()
        parent = self._split_child(core, child, 4)

        footprint = params.host_lock_footprints[ComponentType.FULL]
        self.assertFalse(footprint.requires_host_value)
        self.assertEqual(full.host_lock_nodes(child, params), (child, parent))
        core.validate_host_lock_ref(child.id, params)
        core.dec_host_lock_ref(child.id, params)
        for node in (parent, child):
            data = node.component_data[ComponentType.FULL]
            self.assertEqual(data.host_lock_ref, 0)
            self.assertIsNone(data.host_lock_ids)

    def test_lock_receipt_conversion_does_not_alias_mutable_acquisition_state(
        self,
    ) -> None:
        result = IncLockRefResult(
            host_lock_id=7,
            host_lock_footprints={
                ComponentType.SWA: HostLockFootprint(
                    ranges=(HostLockRange(4, 8), HostLockRange(0, 4)),
                    requires_host_value=True,
                )
            },
            skip_lock_node_ids={ComponentType.SWA: {11}},
        )

        params = result.to_dec_params()
        result.host_lock_footprints.clear()
        result.skip_lock_node_ids[ComponentType.SWA].add(12)

        self.assertEqual(params.host_lock_id, 7)
        self.assertEqual(
            params.host_lock_footprints[ComponentType.SWA].ranges,
            (HostLockRange(0, 4), HostLockRange(4, 8)),
        )
        self.assertEqual(params.skip_lock_node_ids[ComponentType.SWA], {11})

    def test_load_back_ack_preflights_all_host_receipts_before_commit(self) -> None:
        cache = object.__new__(UnifiedRadixCache)
        ack = SimpleNamespace(
            finish_event=Mock(),
            node_ids=[11, 12],
            num_tokens_by_pool={},
            num_bytes=0,
            timing_enabled=False,
        )
        cache.cache_controller = SimpleNamespace(ack_load_queue=[ack])
        cache.buffer_pipeline = None
        cache.metrics_collector = None
        cache.ongoing_load_back = {
            11: _OngoingLoadBack(11, DecLockRefParams(), DecLockRefParams()),
            12: _OngoingLoadBack(12, DecLockRefParams(), DecLockRefParams()),
        }
        cache.dec_lock_ref = Mock()
        cache.dec_host_lock_ref = Mock()
        cache.tree_core = SimpleNamespace(
            finish_load_back=Mock(),
            write_back_duplicate_reclaim_digest=0,
        )

        def reject_second(node_id: int, _params: DecLockRefParams) -> None:
            if node_id == 12:
                raise AssertionError("injected host receipt failure")

        cache.validate_host_lock_ref = Mock(side_effect=reject_second)
        with self.assertRaisesRegex(AssertionError, "injected host receipt failure"):
            cache.loading_check(finish_count=1)

        self.assertEqual(cache.cache_controller.ack_load_queue, [ack])
        self.assertEqual(set(cache.ongoing_load_back), {11, 12})
        cache.dec_lock_ref.assert_not_called()
        cache.dec_host_lock_ref.assert_not_called()
        cache.tree_core.finish_load_back.assert_not_called()

        cache.validate_host_lock_ref = Mock()
        cache.loading_check(finish_count=1)
        self.assertEqual(cache.cache_controller.ack_load_queue, [])
        self.assertEqual(cache.ongoing_load_back, {})
        self.assertEqual(cache.dec_lock_ref.call_count, 2)
        self.assertEqual(cache.dec_host_lock_ref.call_count, 2)
        self.assertEqual(cache.tree_core.finish_load_back.call_count, 2)

    def test_storage_backup_ack_retains_queue_head_until_receipt_release(self) -> None:
        cache = object.__new__(UnifiedRadixCache)
        operation = SimpleNamespace(id=31, completed_tokens=0)
        ack_backup_queue: Queue[SimpleNamespace] = Queue()
        ack_backup_queue.put(operation)
        cache.cache_controller = SimpleNamespace(
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=ack_backup_queue,
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
        )
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.ongoing_backup = {31: (7, DecLockRefParams())}
        cache.dec_host_lock_ref = Mock()
        cache.validate_host_lock_ref = Mock(
            side_effect=AssertionError("injected backup receipt failure")
        )
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None

        with self.assertRaisesRegex(AssertionError, "injected backup receipt failure"):
            cache._drain_storage_control_queues_impl(
                n_storage_hit=0,
                n_ack_prefetch=0,
                n_backup=1,
                n_release=0,
                extra_release_counts={},
                log_metrics=False,
            )

        self.assertIs(ack_backup_queue.queue[0], operation)
        self.assertIn(31, cache.ongoing_backup)
        cache.dec_host_lock_ref.assert_not_called()

        cache.validate_host_lock_ref = Mock()
        cache._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=1,
            n_release=0,
            extra_release_counts={},
            log_metrics=False,
        )
        self.assertTrue(ack_backup_queue.empty())
        self.assertEqual(cache.ongoing_backup, {})
        cache.dec_host_lock_ref.assert_called_once()

    def test_prefetch_setup_releases_every_prepublication_owner_on_failure(
        self,
    ) -> None:
        for failure_point in ("prepare", "build", "sidecar", "submit"):
            with self.subTest(failure_point=failure_point):
                cache = object.__new__(UnifiedRadixCache)
                host_buffer = torch.arange(8)
                component = SimpleNamespace(
                    prepare_prefetch=Mock(
                        return_value=PreparePrefetchResult(host_indices=host_buffer)
                    )
                )
                if failure_point == "prepare":
                    component.prepare_prefetch.side_effect = RuntimeError(
                        "injected prepare failure"
                    )

                controller = SimpleNamespace(
                    prefetch_rate_limited=Mock(return_value=False),
                    prefetch=Mock(return_value=SimpleNamespace()),
                    append_host_mem_release=Mock(),
                    prefetch_tokens_occupied=0,
                )
                if failure_point == "submit":
                    controller.prefetch.side_effect = RuntimeError(
                        "injected submit failure"
                    )

                transfer = PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_buffer,
                )
                tree_core = SimpleNamespace(
                    is_eagle=False,
                    prefetch_anchor_info=Mock(return_value=(None, None)),
                    build_hicache_transfers=Mock(return_value=[transfer]),
                )
                if failure_point == "build":
                    tree_core.build_hicache_transfers.side_effect = RuntimeError(
                        "injected build failure"
                    )

                cache.tree_core = tree_core
                cache.enable_storage = True
                cache.cache_controller = controller
                cache.host_memory_mode = "cache"
                cache.page_size = 1
                cache.prefetch_threshold = 1
                cache._prefetch_outcome_stats = defaultdict(int)
                cache._storage_prefetch_missed_rids = set()
                cache.ongoing_prefetch = {}
                cache.buffer_pipeline = None
                cache.tree_components = (
                    ComponentType.FULL,
                    ComponentType.SWA,
                )
                cache.components = {ComponentType.SWA: component}
                cache.inc_host_lock_ref = Mock(
                    return_value=IncLockRefResult(host_lock_id=91)
                )
                cache.dec_host_lock_ref = Mock()
                cache._build_sidecar_transfers = Mock(return_value=[])
                if failure_point == "sidecar":
                    cache._build_sidecar_transfers.side_effect = RuntimeError(
                        "injected sidecar failure"
                    )

                with self.assertRaisesRegex(RuntimeError, "injected"):
                    cache.prefetch_from_storage(
                        "request",
                        7,
                        list(range(8)),
                    )

                self.assertEqual(cache.ongoing_prefetch, {})
                cache.dec_host_lock_ref.assert_called_once()
                self.assertEqual(controller.prefetch_tokens_occupied, 0)
                if failure_point == "prepare":
                    controller.append_host_mem_release.assert_not_called()
                else:
                    released = controller.append_host_mem_release.call_args.kwargs[
                        "extra_pools"
                    ]
                    self.assertEqual(len(released), 1)
                    self.assertTrue(torch.equal(released[0].host_indices, host_buffer))

    def test_prefetch_setup_releases_prior_buffers_when_later_pool_is_full(
        self,
    ) -> None:
        cache = object.__new__(UnifiedRadixCache)
        host_buffer = torch.arange(8)
        swa = SimpleNamespace(
            prepare_prefetch=Mock(
                return_value=PreparePrefetchResult(host_indices=host_buffer)
            )
        )
        mamba = SimpleNamespace(
            prepare_prefetch=Mock(return_value=PreparePrefetchResult(alloc_failed=True))
        )
        controller = SimpleNamespace(
            prefetch_rate_limited=Mock(return_value=False),
            prefetch=Mock(),
            append_host_mem_release=Mock(),
            prefetch_tokens_occupied=0,
        )
        cache.tree_core = SimpleNamespace(
            is_eagle=False,
            prefetch_anchor_info=Mock(return_value=(None, None)),
            build_hicache_transfers=Mock(
                return_value=[PoolTransfer(name=PoolName.SWA, host_indices=host_buffer)]
            ),
        )
        cache.enable_storage = True
        cache.cache_controller = controller
        cache.host_memory_mode = "cache"
        cache.page_size = 1
        cache.prefetch_threshold = 1
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_missed_rids = set()
        cache.ongoing_prefetch = {}
        cache.buffer_pipeline = None
        cache.tree_components = (
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.MAMBA,
        )
        cache.components = {
            ComponentType.SWA: swa,
            ComponentType.MAMBA: mamba,
        }
        cache.inc_host_lock_ref = Mock(return_value=IncLockRefResult(host_lock_id=92))
        cache.dec_host_lock_ref = Mock()
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None

        cache.prefetch_from_storage("request", 7, list(range(8)))

        controller.prefetch.assert_not_called()
        cache.dec_host_lock_ref.assert_called_once()
        released = controller.append_host_mem_release.call_args.kwargs["extra_pools"]
        self.assertEqual(len(released), 1)
        self.assertTrue(torch.equal(released[0].host_indices, host_buffer))
        self.assertIn("request", cache._storage_prefetch_missed_rids)


if __name__ == "__main__":
    unittest.main()
