import dataclasses
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLeaseError,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeReqToTokenPool,
    DecodeRequest,
)
from sglang.srt.mem_cache.allocation_pin import AllocationPinnedError
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclasses.dataclass
class _Request:
    """Minimal live request carrying allocation identity."""

    req_pool_idx: int | None = None
    mamba_pool_idx: torch.Tensor | None = None
    inflight_middle_chunks: int = 0
    kv_committed_len: int = 0


@dataclasses.dataclass
class _IssuanceFixture:
    """CPU-only scheduler allocation state for issuance tests."""

    queue: DecodePreallocQueue
    request_pool: DecodeReqToTokenPool
    request: _Request
    decode_request: DecodeRequest


def _fixture(
    allocator: TokenToKVPoolAllocator | SWATokenToKVPoolAllocator,
    *,
    source_tp_size: int,
    sliding_window_size: int | None = None,
) -> _IssuanceFixture:
    """Build one exact live request and queue authority.

    :param allocator: Concrete or SWA composite KV allocator.
    :param source_tp_size: TP2 or TP4 prefill width.
    :param sliding_window_size: Active SWA window.
    :returns: CPU-only issuance fixture.
    """

    request_pool = DecodeReqToTokenPool(
        size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=2,
    )
    request = _Request()
    selected = request_pool.alloc([request])
    assert selected is not None
    receiver = SimpleNamespace(
        require_staging=True,
        prefill_info=SimpleNamespace(attn_tp_size=source_tp_size),
    )
    decode_request = DecodeRequest(req=request, kv_receiver=receiver)

    queue = object.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = request_pool
    queue.token_to_kv_pool_allocator = allocator
    queue.tp_size = 1
    queue.scheduler = SimpleNamespace(
        sliding_window_size=sliding_window_size,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gemma4")
        ),
    )
    queue.allocation_lease_authority = DecodeAllocationLeaseAuthority()
    return _IssuanceFixture(
        queue=queue,
        request_pool=request_pool,
        request=request,
        decode_request=decode_request,
    )


def _token_allocator(size: int = 32) -> TokenToKVPoolAllocator:
    """Build a static CPU token allocator.

    :param size: Allocator token capacity.
    :returns: Empty static allocator.
    """

    return TokenToKVPoolAllocator(
        size=size,
        dtype=torch.float16,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )


def _swa_allocator() -> SWATokenToKVPoolAllocator:
    """Build independent CPU FULL and SWA allocators.

    :returns: Empty SWA composite allocator.
    """

    kv_pool = SWAKVPool(
        size=32,
        size_swa=16,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=4,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    return SWATokenToKVPoolAllocator(
        size=32,
        size_swa=16,
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )


@pytest.mark.parametrize("source_tp_size", (2, 4))
def test_live_tp2_tp4_to_tp1_issuance_pins_exact_full_mapping(
    source_tp_size: int,
) -> None:
    """Asymmetric issuance binds source writers, slot generation, and FULL."""

    allocator = _token_allocator()
    fixture = _fixture(allocator, source_tp_size=source_tp_size)
    indices = allocator.alloc(8)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(0, 8)), indices)

    fixture.queue._acquire_decode_allocation_lease(
        fixture.decode_request,
        prefix_len=2,
        migration_end=8,
    )

    lease = fixture.decode_request.allocation_lease
    assert lease is not None
    snapshot = fixture.queue.allocation_lease_authority.snapshot(lease)
    full, swa, mamba = snapshot.components
    assert snapshot.writer_manifest.source_tp_size == source_tp_size
    assert snapshot.request_slot == slot
    assert snapshot.request_generation == int(
        fixture.request_pool.req_generation[slot].item()
    )
    assert full.component is DecodeAllocationComponent.FULL
    assert full.logical_start == 2
    assert full.logical_length == 6
    assert full.virtual_pages == tuple(indices[2:8].tolist())
    assert swa.zero_work
    assert mamba.zero_work

    with pytest.raises(AllocationPinnedError):
        allocator.free(indices[2:8])
    with pytest.raises(AllocationPinnedError):
        fixture.request_pool.free(fixture.request)


def test_live_swa_claim_uses_only_migration_owned_active_window() -> None:
    """SWA pins the live window intersection, not unrelated prompt KV."""

    allocator = _swa_allocator()
    fixture = _fixture(
        allocator,
        source_tp_size=4,
        sliding_window_size=4,
    )
    full_indices = allocator.alloc(8)
    assert full_indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(0, 8)), full_indices)

    fixture.queue._acquire_decode_allocation_lease(
        fixture.decode_request,
        prefix_len=2,
        migration_end=8,
    )

    lease = fixture.decode_request.allocation_lease
    assert lease is not None
    full, swa, mamba = fixture.queue.allocation_lease_authority.snapshot(
        lease
    ).components
    expected_swa = allocator.translate_loc_from_full_to_swa(full_indices[4:8])
    assert full.logical_start == 2
    assert full.logical_length == 6
    assert swa.logical_start == 4
    assert swa.logical_length == 4
    assert swa.virtual_pages == tuple(expected_swa.tolist())
    assert mamba.zero_work


def test_hicache_gap_must_have_final_mapping_before_issuance() -> None:
    """A promised but unmapped HiCache gap fails closed before publication."""

    allocator = _token_allocator()
    fixture = _fixture(allocator, source_tp_size=2)
    indices = allocator.alloc(3)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(5, 8)), indices)

    with pytest.raises(
        DecodeAllocationLeaseError,
        match="HiCache restore destinations must be final",
    ):
        fixture.queue._acquire_decode_allocation_lease(
            fixture.decode_request,
            prefix_len=2,
            migration_end=8,
        )
    assert fixture.decode_request.allocation_lease is None


def test_failed_child_rolls_back_every_prepared_child_lease() -> None:
    """A later child failure removes all earlier cohort migration pins."""

    allocator = _token_allocator()
    first = _fixture(allocator, source_tp_size=2)
    second_request = _Request()
    selected = first.request_pool.alloc([second_request])
    assert selected is not None
    second = DecodeRequest(
        req=second_request,
        kv_receiver=SimpleNamespace(
            require_staging=True,
            prefill_info=SimpleNamespace(attn_tp_size=2),
        ),
    )
    first_indices = allocator.alloc(4)
    second_indices = allocator.alloc(2)
    assert first_indices is not None
    assert second_indices is not None
    first_slot = first.request.req_pool_idx
    second_slot = second_request.req_pool_idx
    assert first_slot is not None
    assert second_slot is not None
    first.request_pool.write((first_slot, slice(0, 4)), first_indices)
    first.request_pool.write((second_slot, slice(2, 4)), second_indices)

    first.queue._acquire_decode_allocation_lease(
        first.decode_request,
        prefix_len=0,
        migration_end=4,
    )
    with pytest.raises(DecodeAllocationLeaseError, match="incomplete"):
        first.queue._acquire_decode_allocation_lease(
            second,
            prefix_len=0,
            migration_end=4,
        )
    first.queue._rollback_decode_allocation_leases([first.decode_request])

    assert first.decode_request.allocation_lease is None
    allocator.free(first_indices)
    first.request_pool.free(first.request)
