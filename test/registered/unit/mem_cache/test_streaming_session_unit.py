from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    KVComponentResidency,
    MatchResult,
)
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.session.streaming_session import (
    DemotedSessionState,
    SessionSlot,
    StreamingSession,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []
        self.freed_swa = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())

    def free_swa(self, free_index: torch.Tensor):
        self.freed_swa.append(free_index.clone())

    def get_kvcache(self) -> "_FakeAllocator":
        return self

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return kv_indices + 1


class _FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor) -> None:
        self.req_to_token = req_to_token
        self.free_slots = []

    def release_detached_request_slot(self, pool_idx: int) -> None:
        self.free_slots.append(pool_idx)

    def free(self, req: "_FakeReq") -> None:
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None


class _FakeInnerCache:
    def __init__(
        self,
        req_to_token_pool,
        allocator,
        page_size,
        match_results=None,
        sliding_window_size=None,
        protected_residency=None,
        private_parent=None,
        private_len=0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []
        self.dec_lock_ref_params = []
        self.dec_host_lock_ref_calls = []
        self.cleared_session_refs = []
        self.retired_private_paths = []
        self.adopted_private_paths = []
        self.private_path_detach_validations = []
        self.inc_lock_ref_calls = []
        self.sanity_checks = 0
        self.private_parent = private_parent
        self.private_len = private_len
        self.protected_residency = (
            protected_residency
            if protected_residency is not None
            else (KVComponentResidency(), KVComponentResidency())
        )

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)
        self.dec_lock_ref_params.append(args[0] if args else kwargs.get("params"))

    def inc_lock_ref(self, node):
        self.inc_lock_ref_calls.append(node)
        return IncLockRefResult(
            swa_uuid_for_lock=23,
            skip_lock_node_ids={ComponentType.SWA: {9}},
        )

    def dec_host_lock_ref(self, node, params):
        self.dec_host_lock_ref_calls.append((node, params))

    def clear_radix_session_refs(self, session_id: str) -> int:
        self.cleared_session_refs.append(session_id)
        return 1

    def retire_streaming_session_private_path(self, session_id: str, node) -> None:
        self.retired_private_paths.append((session_id, node))

    def validate_streaming_session_private_path_detach(
        self,
        session_id: str,
        node,
        owner_params: DecLockRefParams,
        *,
        allow_device_locks: bool,
    ) -> None:
        self.private_path_detach_validations.append(
            (session_id, node, owner_params, allow_device_locks)
        )

    def streaming_session_private_parent(self, node):
        return self.private_parent

    def adopt_streaming_session_private_path(self, session_id: str, node) -> int:
        self.adopted_private_paths.append((session_id, node))
        return self.private_len

    def supports_mamba(self):
        return False

    def supports_swa(self):
        return self.sliding_window_size is not None

    def streaming_session_protected_residency(self, node):
        return self.protected_residency

    def sanity_check(self):
        self.sanity_checks += 1
        return None


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            _inflight=True,
        )

        def finish_req(req: _FakeReq) -> None:
            assert req is self
            req.streaming_session_owns_inflight = False
            session._inflight = False

        def abort_req(req: _FakeReq) -> None:
            assert req is self
            req.streaming_session_owns_inflight = False
            session._inflight = False

        session.finish_req = finish_req
        session.abort_req = abort_req
        self.session = session
        self.streaming_session_owns_inflight = True
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv = ReqKvInfo(
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
        )
        self.origin_input_ids = list(range(committed))
        self.origin_input_ids_unpadded = self.origin_input_ids
        self.output_ids = []
        self.streaming_session_floor = 0
        self.streaming_session_commit_to = None
        self.streaming_session_truncate_to = None
        self.streaming_session_admitted = True
        self.streaming_session_preburst_mutation = False
        self.extra_key = None
        self.cache_salt = None
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.skip_lock_node_ids = {}
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None
        self.c4_state_alloc_offset = 0
        self.c128_state_alloc_offset = 0
        self.time_stats = SimpleNamespace(
            increment_streaming_session_abort_with_slot_preserved=lambda: None
        )


def test_per_session_cache_snapshot_reports_durable_slot_ownership():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            _FakeReqToTokenPool(req_to_token),
            _FakeAllocator(),
            page_size=16,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=91,
        kv=SimpleNamespace(kv_allocated_len=93, swa_evicted_seqlen=0),
        cache_protected_len=32,
    )

    snapshot = tree_cache.streaming_session_cache_snapshot("session-a")

    assert snapshot.protected == 32
    assert snapshot.held_tokens == 64
    assert snapshot.full == KVComponentResidency(device_pages=4)
    assert snapshot.swa == KVComponentResidency()
    assert tree_cache.streaming_session_cache_snapshot("missing").held_tokens == 0


def test_per_session_cache_snapshot_combines_tree_and_slot_residency():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            _FakeReqToTokenPool(req_to_token),
            _FakeAllocator(),
            page_size=16,
            sliding_window_size=64,
            protected_residency=(
                KVComponentResidency(device_pages=2, host_backed_pages=2),
                KVComponentResidency(device_pages=1, host_backed_pages=1),
            ),
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=96,
        kv=SimpleNamespace(kv_allocated_len=96, swa_evicted_seqlen=64),
        cache_protected_len=32,
        last_node=17,
    )

    snapshot = tree_cache.streaming_session_cache_snapshot("session-a")

    assert snapshot.full == KVComponentResidency(
        device_pages=6,
        host_backed_pages=2,
    )
    assert snapshot.swa == KVComponentResidency(
        device_pages=3,
        host_backed_pages=1,
    )


def test_demoted_snapshot_and_close_release_only_session_ownership() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    inner = _FakeInnerCache(
        _FakeReqToTokenPool(req_to_token),
        _FakeAllocator(),
        page_size=16,
        protected_residency=(
            KVComponentResidency(device_pages=0, host_backed_pages=6),
            KVComponentResidency(device_pages=0, host_backed_pages=4),
        ),
    )
    tree_cache = StreamingSession(inner)
    lock_params = DecLockRefParams(host_lock_id=17)
    tree_cache.demoted["session-a"] = DemotedSessionState(
        last_node=41,
        cache_protected_len=96,
        tree_protected_len=96,
        swa_evicted_seqlen=0,
        host_lock_params=lock_params,
    )
    tree_cache.slots["active-session"] = SessionSlot(req_pool_idx=0)

    snapshot = tree_cache.streaming_session_cache_snapshot("session-a")

    assert snapshot.protected == 96
    assert snapshot.held_tokens == 0
    assert snapshot.full == KVComponentResidency(host_backed_pages=6)
    assert snapshot.swa == KVComponentResidency(host_backed_pages=4)

    tree_cache.sanity_check()
    assert inner.sanity_checks == 1

    tree_cache.release_session("session-a")

    assert not tree_cache.is_demoted("session-a")
    assert inner.dec_host_lock_ref_calls == [(41, lock_params)]
    assert inner.cleared_session_refs == ["session-a"]
    assert inner.retired_private_paths == [("session-a", 41)]


def test_restored_private_suffix_reanchors_tree_lock_and_becomes_slot_owned() -> None:
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=page_size,
        private_parent=41,
        private_len=17,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=80,
        kv=SimpleNamespace(kv_allocated_len=80, swa_evicted_seqlen=0),
        last_node=42,
        cache_protected_len=65,
        tree_protected_len=65,
        swa_uuid_for_lock=17,
        skip_lock_node_ids={ComponentType.FULL: {7}},
    )
    host_lock_params = DecLockRefParams(host_lock_id=19)
    tree_cache.demoted["session-a"] = DemotedSessionState(
        last_node=42,
        cache_protected_len=65,
        tree_protected_len=48,
        swa_evicted_seqlen=0,
        host_lock_params=host_lock_params,
    )

    tree_cache._release_demoted_state("session-a")

    slot = tree_cache.slots["session-a"]
    assert slot.last_node == 41
    assert slot.cache_protected_len == 65
    assert slot.tree_protected_len == 48
    assert slot.swa_uuid_for_lock == 23
    assert slot.skip_lock_node_ids == {ComponentType.SWA: {9}}
    assert inner.dec_host_lock_ref_calls == [(42, host_lock_params)]
    assert inner.cleared_session_refs == ["session-a"]
    assert inner.inc_lock_ref_calls == [41]
    assert inner.dec_lock_ref_calls == [42]
    assert inner.dec_lock_ref_params == [
        DecLockRefParams(
            swa_uuid_for_lock=17,
            skip_lock_node_ids={ComponentType.FULL: {7}},
        )
    ]
    assert inner.adopted_private_paths == [("session-a", 42)]
    snapshot = tree_cache.streaming_session_cache_snapshot("session-a")
    assert snapshot.protected == 65
    assert snapshot.held_tokens == 32
    assert snapshot.full.device_pages == 2
    assert tree_cache.session_held_tokens() == 32

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [42, 41]
    assert inner.retired_private_paths == [("session-a", 41)]
    assert req_to_token_pool.free_slots == [0]
    assert allocator.freed[0].tolist() == list(range(48, 80))


@pytest.mark.parametrize(
    ("committed_len", "logical_watermark", "expected_freed"),
    [
        (79, 32, []),
        (80, 48, list(range(32, 48))),
    ],
)
def test_demoted_swa_adoption_reconciles_restored_private_ownership(
    committed_len: int,
    logical_watermark: int,
    expected_freed: list[int],
) -> None:
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=page_size,
        sliding_window_size=32,
        private_parent=41,
        private_len=49,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=committed_len,
        kv=SimpleNamespace(
            kv_allocated_len=80,
            swa_evicted_seqlen=logical_watermark,
        ),
        streaming_session_floor=committed_len,
        last_node=42,
        cache_protected_len=65,
        tree_protected_len=65,
        swa_uuid_for_lock=17,
    )
    tree_cache.demoted["session-a"] = DemotedSessionState(
        last_node=42,
        cache_protected_len=65,
        tree_protected_len=16,
        swa_evicted_seqlen=32,
        host_lock_params=DecLockRefParams(host_lock_id=19),
    )

    assert tree_cache._release_demoted_state("session-a")

    slot = tree_cache.slots["session-a"]
    assert slot.tree_protected_len == 16
    assert slot.kv.swa_evicted_seqlen == logical_watermark
    assert [
        index for freed in allocator.freed_swa for index in freed.tolist()
    ] == expected_freed
    assert tree_cache.session_held_swa_tokens() == 80 - logical_watermark


def test_demoted_reload_skips_ordinary_radix_insertion_until_adoption() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    inner = _FakeInnerCache(
        _FakeReqToTokenPool(req_to_token),
        _FakeAllocator(),
        page_size=16,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.demoted["session-a"] = DemotedSessionState(
        last_node=42,
        cache_protected_len=65,
        tree_protected_len=0,
        swa_evicted_seqlen=0,
        host_lock_params=DecLockRefParams(),
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=65, allocated=80)

    handled = tree_cache.try_cache_unfinished_req(req)

    assert handled
    assert tree_cache.is_demoted("session-a")
    assert "session-a" not in tree_cache.slots


def test_held_empty_slot_remains_owned_during_resume() -> None:
    slot = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=0,
        kv=ReqKvInfo(),
    )
    req = _FakeReq("session-a", req_pool_idx=1, committed=0, allocated=0)

    slot.restore_to_req(req)

    assert slot.is_holding_kv
    assert req.req_pool_idx == 0
    assert req.kv.is_released


def test_preabort_detaches_session_and_preserves_slot():
    """Pre-aborted req (to_finish set before match_prefix) is detached from
    the session: session=None, abort_req() called. Slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )
        ],
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv=SimpleNamespace(kv_allocated_len=48, swa_evicted_seqlen=0),
        cache_protected_len=16,
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    # Req detached from session.
    assert req.session is None
    # Slot untouched.
    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_first_mid_abort_nukes_ephemeral_slot():
    """First-request mid-processing abort: no slot exists yet, ephemeral
    slot is created from req state and nuked via release_session."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    req.finished_reason = FINISH_ABORT("input too long")

    tree_cache.cache_finished_req(req)

    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))


def test_first_mid_abort_preserves_complete_preburst_slot():
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=45)
    req.origin_input_ids = list(range(40))
    req.origin_input_ids_unpadded = req.origin_input_ids
    req.streaming_session_preburst_mutation = True
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 40
    assert slot.kv.kv_allocated_len == 40
    assert allocator.freed[0].tolist() == list(range(40, 45))
    assert req.req_pool_idx is None
    assert req.kv.is_released
    assert req_to_token_pool.free_slots == []


def test_nth_mid_abort_preserves_session_slot():
    """Nth-request abort rolls KV back to the last successful boundary."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # Session already has a slot from a previous turn.
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=None,
        cache_protected_len=0,
    )

    # Mid-processing abort: req has the SESSION slot's pool_idx (restore_to_req ran).
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    # Only the aborted request's tail is freed; the committed slot survives.
    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 50
    assert slot.kv.kv_allocated_len == 50
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(50, 65))
    assert req_to_token_pool.free_slots == []

    # Ownership moved back to the slot.
    assert req.req_pool_idx is None
    assert req.kv.is_released


def test_committed_append_becomes_abort_rollback_boundary():
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=70, allocated=75)
    req.streaming_session_commit_to = 60
    req.streaming_session_preburst_mutation = True
    req.streaming_session_floor = 60
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 70
    assert slot.kv.kv_allocated_len == 70
    assert slot.streaming_session_floor == 60
    assert allocator.freed[0].tolist() == list(range(70, 75))


def test_raw_append_becomes_abort_rollback_boundary():
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=70, allocated=75)
    req.origin_input_ids = list(range(70))
    req.origin_input_ids_unpadded = req.origin_input_ids
    req.streaming_session_preburst_mutation = True
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 70
    assert slot.kv.kv_allocated_len == 70
    assert allocator.freed[0].tolist() == list(range(70, 75))


def test_truncate_rebuild_becomes_abort_rollback_boundary():
    page_size = 16
    req_to_token = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=64,
        kv=SimpleNamespace(kv_allocated_len=64, swa_evicted_seqlen=0),
        cache_protected_len=64,
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=144, allocated=160)
    req.origin_input_ids = list(range(144))
    req.origin_input_ids_unpadded = req.origin_input_ids
    req.streaming_session_truncate_to = 128
    req.streaming_session_preburst_mutation = True
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 144
    assert slot.kv.kv_allocated_len == 144
    assert slot.cache_protected_len == 64
    assert allocator.freed[0].tolist() == list(range(144, 160))


def test_incomplete_committed_append_retires_physical_slot():
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.origin_input_ids = list(range(70))
    req.origin_input_ids_unpadded = req.origin_input_ids
    req.streaming_session_commit_to = 60
    req.streaming_session_preburst_mutation = True
    req.streaming_session_floor = 60
    req.finished_reason = FINISH_ABORT("prefill interrupted")

    tree_cache.cache_finished_req(req)

    assert "session-a" not in tree_cache.slots
    assert req.req_pool_idx is None
    assert req.kv.is_released
    assert req_to_token_pool.free_slots == [0]
    assert allocator.freed[0].tolist() == list(range(65))


def test_release_session_threads_mamba_skip_ids():
    """release_session must forward the slot's skip_lock_node_ids to
    dec_lock_ref. The first req's last_node may be full-only-locked (mamba
    skipped at inc), so without the skip set the release would drop a mamba
    lock the session never took -- another request's, on a shared node."""
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = StreamingSession(inner)

    lock_node = SimpleNamespace(id=42)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=lock_node,
        cache_protected_len=0,
        skip_lock_node_ids={ComponentType.MAMBA: {42}},
    )

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [lock_node]
    params = inner.dec_lock_ref_params[0]
    assert params is not None
    assert params.skip_lock_node_ids.get(ComponentType.MAMBA) == {42}


# Shrink tests removed: streaming sessions are append-only after the
# rollback fix in session_controller (rollback_aborted_req).  The shrink
# code path in cache_finished_req no longer exists.


def test_trim_overshoot_postcondition():
    """`_trim_overshoot` postcondition: every per-req KV field is capped at
    target = origin+finished_len, output_ids is truncated, and the tail
    KV slots are freed. Covers both non-SWA fields (kv_committed_len,
    kv_allocated_len, output_ids) and SWA bookkeeping (swa_evicted_seqlen)
    in one shot — same invariant `_free_tail` enforces on the match_prefix
    path.
    """
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    # Overshoot scenario: origin=26, finished_len=12 -> target=38.
    # committed=40 (overshoot 2), allocated=44, swa_evicted=42 (> target),
    # output_ids extended to 14 by the overshoot round.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 42

    tree_cache._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.swa_evicted_seqlen == target
    assert len(req.output_ids) == 12
    # Tail [38, 44) freed by _free_kv_aligned.
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(38, 44))


def test_truncate_session_frees_tail_and_clamps_cursors():
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=91,
        kv=SimpleNamespace(kv_allocated_len=96, swa_evicted_seqlen=70),
        cache_protected_len=32,
    )

    tree_cache.truncate_session("session-a", 53)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 53
    assert slot.kv.kv_allocated_len == 53
    assert slot.kv.swa_evicted_seqlen == 53
    assert allocator.freed[0].tolist() == list(range(64, 96))


def test_truncate_below_protected_reprefills_partial_shared_page():
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=80,
        kv=SimpleNamespace(kv_allocated_len=80, swa_evicted_seqlen=70),
        cache_protected_len=64,
        tree_protected_len=64,
    )

    tree_cache.truncate_session("session-a", 35)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 32
    assert slot.kv.kv_allocated_len == 32
    assert slot.kv.swa_evicted_seqlen == 32
    assert slot.cache_protected_len == 32
    assert slot.tree_protected_len == 64
    assert allocator.freed[0].tolist() == list(range(64, 80))


@pytest.mark.parametrize(("target", "expected_retained"), [(48, 32), (64, 48)])
def test_truncate_at_protected_page_reprefills_logit_reserve_page(
    target: int, expected_retained: int
) -> None:
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=80,
        kv=SimpleNamespace(kv_allocated_len=80, swa_evicted_seqlen=0),
        cache_protected_len=64,
        tree_protected_len=64,
    )

    tree_cache.truncate_session("session-a", target)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == expected_retained
    assert slot.kv.kv_allocated_len == expected_retained
    assert slot.cache_protected_len == expected_retained
    assert slot.tree_protected_len == 64
    assert allocator.freed[0].tolist() == list(range(64, 80))

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    match = tree_cache.try_match_prefix(
        SimpleNamespace(req=req, key=list(range(target - 1)))
    )

    assert match is not None
    assert len(match.device_indices) == expected_retained
    assert match.cache_protected_len == expected_retained


def test_page_one_truncate_reprefills_logit_reserve_token() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=80,
        kv=SimpleNamespace(kv_allocated_len=80, swa_evicted_seqlen=0),
        cache_protected_len=64,
        tree_protected_len=64,
    )

    tree_cache.truncate_session("session-a", 64)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 63
    assert slot.kv.kv_allocated_len == 63
    assert slot.cache_protected_len == 63
    assert allocator.freed[0].tolist() == list(range(64, 80))


def test_streaming_match_rejects_prefix_below_protected_boundary() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size=16)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=64,
        kv=SimpleNamespace(kv_allocated_len=64, swa_evicted_seqlen=0),
        cache_protected_len=64,
        tree_protected_len=64,
    )
    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)

    with pytest.raises(AssertionError, match="streaming session prefix shrank"):
        tree_cache.try_match_prefix(SimpleNamespace(req=req, key=list(range(63))))

    assert allocator.freed == []


def test_streaming_match_can_replay_session_owned_logit_reserve() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size=16)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=96,
        kv=SimpleNamespace(kv_allocated_len=96, swa_evicted_seqlen=0),
        cache_protected_len=64,
        tree_protected_len=64,
    )

    tree_cache.truncate_session("session-a", 80)
    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    match = tree_cache.try_match_prefix(SimpleNamespace(req=req, key=list(range(79))))

    assert match is not None
    assert len(match.device_indices) == 79
    assert match.cache_protected_len == 64
    assert tree_cache.slots["session-a"].kv_committed_len == 79
    assert tree_cache.slots["session-a"].kv.kv_allocated_len == 79
    assert allocator.freed[0].tolist() == list(range(80, 96))


def test_floor_round_trip_and_detached_swa_reconciliation():
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            req_to_token_pool,
            allocator,
            page_size,
            sliding_window_size=32,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=112,
        kv=SimpleNamespace(kv_allocated_len=112, swa_evicted_seqlen=32),
        streaming_session_floor=64,
        cache_protected_len=32,
    )
    assert tree_cache.session_held_swa_tokens() == 80

    tree_cache.commit_session("session-a", 96)
    tree_cache.commit_session("session-a", 96)

    slot = tree_cache.slots["session-a"]
    assert slot.streaming_session_floor == 96
    assert slot.kv.swa_evicted_seqlen == 64
    assert len(allocator.freed_swa) == 1
    assert allocator.freed_swa[0].tolist() == list(range(32, 64))
    assert tree_cache.session_held_swa_tokens() == 48

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    slot.restore_to_req(req)
    assert req.streaming_session_floor == 96

    tree_cache.commit_session("session-a", 112)
    assert slot.kv.swa_evicted_seqlen == 80
    assert tree_cache.session_held_swa_tokens() == 32


def test_active_swa_eviction_is_clamped_by_streaming_floor():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=0,
        swa_evict_floor=0,
        streaming_session_floor=80,
        streaming_session_tree_protected_len=0,
        kv=ReqKvInfo(kv_allocated_len=112, swa_evicted_seqlen=0),
    )

    free_swa_out_of_window_slots(
        req,
        112,
        sliding_window_size=32,
        page_size=16,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )

    assert req.kv.swa_evicted_seqlen == 48
    assert allocator.freed_swa[0].tolist() == list(range(48))


def test_held_empty_request_skips_swa_eviction_before_allocation() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=0,
        swa_evict_floor=0,
        streaming_session_floor=80,
        streaming_session_tree_protected_len=None,
        kv=ReqKvInfo(),
    )

    free_swa_out_of_window_slots(
        req,
        112,
        sliding_window_size=32,
        page_size=16,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )

    assert req.kv.is_released
    assert allocator.freed_swa == []


def test_streaming_floor_preserves_radix_owned_rollback_window():
    req_to_token = torch.arange(9_216, dtype=torch.int32).reshape(2, 4_608)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=2_048,
        swa_evict_floor=0,
        streaming_session_floor=2_048,
        streaming_session_tree_protected_len=2_048,
        kv=ReqKvInfo(kv_allocated_len=4_608, swa_evicted_seqlen=0),
    )

    free_swa_out_of_window_slots(
        req,
        4_608,
        sliding_window_size=1_024,
        page_size=64,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )

    assert req.kv.swa_evicted_seqlen == 1_024
    assert allocator.freed_swa == []


def test_streaming_exact_fringe_preserves_its_physical_swa_page() -> None:
    req_to_token = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=65,
        swa_evict_floor=0,
        streaming_session_floor=256,
        streaming_session_tree_protected_len=65,
        kv=ReqKvInfo(kv_allocated_len=256, swa_evicted_seqlen=0),
    )

    free_swa_out_of_window_slots(
        req,
        256,
        sliding_window_size=64,
        page_size=16,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )

    assert req.kv.swa_evicted_seqlen == 192
    assert len(allocator.freed_swa) == 1
    assert allocator.freed_swa[0].tolist() == list(range(80, 192))


def test_streaming_floor_rejects_unpersisted_forced_evict_prefix():
    req_to_token = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=64,
        swa_evict_floor=96,
        streaming_session_floor=256,
        streaming_session_tree_protected_len=64,
        kv=ReqKvInfo(kv_allocated_len=256, swa_evicted_seqlen=32),
    )

    with pytest.raises(RuntimeError, match="prefill-aware SWA"):
        free_swa_out_of_window_slots(
            req,
            256,
            sliding_window_size=64,
            page_size=16,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

    assert req.kv.swa_evicted_seqlen == 32
    assert allocator.freed_swa == []


def test_detached_floor_frees_only_session_owned_swa():
    req_to_token = torch.arange(9_216, dtype=torch.int32).reshape(2, 4_608)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            req_to_token_pool,
            allocator,
            page_size=64,
            sliding_window_size=1_024,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=4_608,
        kv=SimpleNamespace(kv_allocated_len=4_608, swa_evicted_seqlen=0),
        streaming_session_floor=2_048,
        cache_protected_len=2_048,
    )

    tree_cache.commit_session("session-a", 2_048)
    slot = tree_cache.slots["session-a"]
    assert slot.kv.swa_evicted_seqlen == 1_024
    assert allocator.freed_swa == []

    tree_cache.commit_session("session-a", 4_096)
    assert slot.kv.swa_evicted_seqlen == 3_072
    assert len(allocator.freed_swa) == 1
    assert allocator.freed_swa[0].tolist() == list(range(2_048, 3_072))


def test_truncate_rejects_logically_evicted_rollback_window():
    req_to_token = torch.arange(9_216, dtype=torch.int32).reshape(2, 4_608)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            req_to_token_pool,
            allocator,
            page_size=64,
            sliding_window_size=1_024,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=4_608,
        kv=SimpleNamespace(kv_allocated_len=4_608, swa_evicted_seqlen=2_048),
        streaming_session_floor=2_048,
        cache_protected_len=2_048,
    )

    with pytest.raises(AssertionError, match="SWA pin invariant"):
        tree_cache.truncate_session("session-a", 2_048)


def test_below_protected_truncate_rejects_evicted_rollback_window():
    req_to_token = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            req_to_token_pool,
            allocator,
            page_size=16,
            sliding_window_size=32,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=128,
        kv=SimpleNamespace(kv_allocated_len=128, swa_evicted_seqlen=32),
        streaming_session_floor=16,
        cache_protected_len=64,
    )

    with pytest.raises(AssertionError, match="SWA pin invariant"):
        tree_cache.truncate_session("session-a", 48)


def test_below_protected_truncate_keeps_page_aligned_rollback_window():
    req_to_token = torch.arange(512, dtype=torch.int32).reshape(2, 256)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            req_to_token_pool,
            allocator,
            page_size=16,
            sliding_window_size=32,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=128,
        kv=SimpleNamespace(kv_allocated_len=128, swa_evicted_seqlen=16),
        streaming_session_floor=16,
        cache_protected_len=64,
    )

    tree_cache.truncate_session("session-a", 48)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 32
    assert slot.kv.kv_allocated_len == 32
    assert slot.kv.swa_evicted_seqlen == 16
    assert slot.cache_protected_len == 32
    assert allocator.freed[0].tolist() == list(range(64, 128))


def test_non_session_swa_eviction_keeps_existing_frontier():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    req = SimpleNamespace(
        is_holding_kv=True,
        req_pool_idx=0,
        cache_protected_len=0,
        swa_evict_floor=0,
        streaming_session_floor=None,
        streaming_session_tree_protected_len=None,
        kv=ReqKvInfo(kv_allocated_len=112, swa_evicted_seqlen=0),
    )

    free_swa_out_of_window_slots(
        req,
        112,
        sliding_window_size=32,
        page_size=16,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )

    assert req.kv.swa_evicted_seqlen == 80
    assert allocator.freed_swa[0].tolist() == list(range(80))


def test_subpage_truncate_retires_slot_and_reallocates_fresh_owner():
    pool = ReqToTokenPool(
        size=1,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    first_owner = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([first_owner]) == [1]
    pool.req_to_token[1, :80] = torch.arange(80, dtype=torch.int32)

    allocator = _FakeAllocator()
    inner = _FakeInnerCache(pool, allocator, page_size=64)
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=1,
        kv_committed_len=80,
        kv=SimpleNamespace(kv_allocated_len=80, swa_evicted_seqlen=70),
        last_node="first-turn-node",
        cache_protected_len=64,
        tree_protected_len=64,
    )

    tree_cache.truncate_session("session-a", 35)

    assert "session-a" not in tree_cache.slots
    assert allocator.freed[0].tolist() == list(range(64, 80))
    assert pool.free_slots == [1]
    assert inner.dec_lock_ref_calls == ["first-turn-node"]

    replacement = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([replacement]) == [1]
    assert int(pool.req_generation[1].item()) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
