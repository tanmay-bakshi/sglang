import dataclasses
import sys
from types import SimpleNamespace
from typing import Never

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
    _DecodeAllocationPreparation,
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
    kv: object | None = None
    mamba_pool_idx: torch.Tensor | None = None
    inflight_middle_chunks: int = 0
    kv_committed_len: int = 0
    origin_input_ids: list[int] = dataclasses.field(
        default_factory=lambda: list(range(8))
    )
    output_ids: list[int] = dataclasses.field(default_factory=list)
    rid: str = "request"
    extend_range: range | None = None

    def set_extend_range(self, start: int, end: int) -> None:
        """Set the allocated request range.

        :param start: Inclusive range start.
        :param end: Exclusive range end.
        """

        self.extend_range = range(start, end)


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
    destination_tp_size: int = 1,
    destination_tp_rank: int = 0,
    sliding_window_size: int | None = None,
) -> _IssuanceFixture:
    """Build one exact live request and queue authority.

    :param allocator: Concrete or SWA composite KV allocator.
    :param source_tp_size: Supported packed prefill width.
    :param destination_tp_size: Decode attention TP width.
    :param destination_tp_rank: Decode attention TP rank.
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
    queue.kv_manager = SimpleNamespace(
        attn_tp_size=destination_tp_size,
        attn_tp_rank=destination_tp_rank,
    )
    queue.scheduler = SimpleNamespace(
        enable_hisparse=False,
        sliding_window_size=sliding_window_size,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="gemma4")),
        server_args=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False,
        ),
    )
    queue.allocation_lease_authority = DecodeAllocationLeaseAuthority()
    queue.token_to_kv_pool = object()
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


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_live_supported_source_to_tp1_issuance_pins_exact_full_mapping(
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
        migration_start=2,
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


@pytest.mark.parametrize("destination_tp_rank", (0, 1))
def test_live_tp2_to_tp2_issuance_binds_rank_local_writer(
    destination_tp_rank: int,
) -> None:
    """Each TP2 decoder rank leases only its routed source writer."""

    allocator = _token_allocator()
    fixture = _fixture(
        allocator,
        source_tp_size=2,
        destination_tp_size=2,
        destination_tp_rank=destination_tp_rank,
    )
    indices = allocator.alloc(8)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(0, 8)), indices)

    fixture.queue._acquire_decode_allocation_lease(
        fixture.decode_request,
        migration_start=0,
        migration_end=8,
    )

    lease = fixture.decode_request.allocation_lease
    assert lease is not None
    snapshot = fixture.queue.allocation_lease_authority.snapshot(lease)
    manifest = snapshot.writer_manifest
    assert manifest.destination_tp_size == 2
    assert manifest.destination_tp_rank == destination_tp_rank
    assert tuple(writer.source_attn_tp_rank for writer in manifest.writers) == (
        destination_tp_rank,
    )
    assert len(snapshot.writer_participation) == 3
    assert all(
        item.writer_id.source_attn_tp_rank == destination_tp_rank
        for item in snapshot.writer_participation
    )


def test_issuance_rejects_destination_tp_wider_than_source() -> None:
    """A destination width without complete source coverage fails before pinning."""

    allocator = _token_allocator()
    fixture = _fixture(
        allocator,
        source_tp_size=2,
        destination_tp_size=4,
        destination_tp_rank=0,
    )
    indices = allocator.alloc(8)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(0, 8)), indices)

    with pytest.raises(DecodeAllocationLeaseError, match="divisible"):
        fixture.queue._acquire_decode_allocation_lease(
            fixture.decode_request,
            migration_start=0,
            migration_end=8,
        )

    assert fixture.decode_request.allocation_lease is None


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_reserved_issuance_uses_authorized_source_before_receiver_handshake(
    source_tp_size: int,
) -> None:
    """Reserved preallocation issues its lease before receiver bootstrap."""

    allocator = _token_allocator()
    fixture = _fixture(allocator, source_tp_size=source_tp_size)
    fixture.request_pool.free(fixture.request)
    fixture.decode_request.kv_receiver = SimpleNamespace(require_staging=False)

    kv_indices = fixture.queue._pre_alloc(
        fixture.request,
        decode_req=fixture.decode_request,
        migration_end=len(fixture.request.origin_input_ids),
        source_tp_size=source_tp_size,
    )

    assert kv_indices.numel() == len(fixture.request.origin_input_ids)
    lease = fixture.decode_request.allocation_lease
    assert lease is not None
    snapshot = fixture.queue.allocation_lease_authority.snapshot(lease)
    assert snapshot.writer_manifest.source_tp_size == source_tp_size
    assert snapshot.request_slot == fixture.request.req_pool_idx


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
        migration_start=2,
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


def test_hicache_restore_gap_is_excluded_from_migration_ownership() -> None:
    """An unmapped restore gap is valid because prefill cannot write it."""

    allocator = _token_allocator()
    fixture = _fixture(allocator, source_tp_size=2)
    indices = allocator.alloc(3)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(5, 8)), indices)

    fixture.queue._acquire_decode_allocation_lease(
        fixture.decode_request,
        migration_start=5,
        migration_end=8,
    )

    lease = fixture.decode_request.allocation_lease
    assert lease is not None
    full = fixture.queue.allocation_lease_authority.snapshot(lease).components[0]
    assert full.logical_start == 5
    assert full.logical_length == 3
    assert full.virtual_pages == tuple(indices.tolist())


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
        migration_start=0,
        migration_end=4,
    )
    with pytest.raises(DecodeAllocationLeaseError, match="incomplete"):
        first.queue._acquire_decode_allocation_lease(
            second,
            migration_start=0,
            migration_end=4,
        )
    first.queue._rollback_decode_allocation_leases([first.decode_request])

    assert first.decode_request.allocation_lease is None
    allocator.free(first_indices)
    first.request_pool.free(first.request)


def test_post_acquire_prepublication_failure_rolls_back_entire_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any preparation failure releases current and earlier child leases."""

    allocator = _token_allocator()
    first = _fixture(allocator, source_tp_size=4)
    second_request = _Request()
    selected = first.request_pool.alloc([second_request])
    assert selected is not None
    second = DecodeRequest(
        req=second_request,
        kv_receiver=SimpleNamespace(
            require_staging=True,
            prefill_info=SimpleNamespace(attn_tp_size=4),
        ),
    )
    indices = allocator.alloc(8)
    assert indices is not None
    first_slot = first.request.req_pool_idx
    second_slot = second_request.req_pool_idx
    assert first_slot is not None
    assert second_slot is not None
    first.request_pool.write((first_slot, slice(0, 4)), indices[:4])
    first.request_pool.write((second_slot, slice(0, 4)), indices[4:])

    def fail_after_acquisition(
        rids_to_check: list[str] | None,
        preparation: _DecodeAllocationPreparation,
    ) -> Never:
        """Inject failure after both leases exist but before publication.

        :param rids_to_check: Unused request filter.
        :param preparation: Live cohort transaction.
        :raises RuntimeError: Always, after both leases are prepared.
        """

        del rids_to_check
        first.queue._acquire_decode_allocation_lease(
            first.decode_request,
            migration_start=0,
            migration_end=4,
        )
        preparation.record_prepared(first.decode_request)
        first.queue._acquire_decode_allocation_lease(
            second,
            migration_start=0,
            migration_end=4,
        )
        preparation.record_prepared(second)
        raise RuntimeError("injected state payload failure")

    monkeypatch.setattr(
        first.queue,
        "_prepare_and_publish_preallocated",
        fail_after_acquisition,
    )
    with pytest.raises(RuntimeError, match="injected state payload failure"):
        first.queue.pop_preallocated()

    assert first.decode_request.allocation_lease is None
    assert second.allocation_lease is None
    allocator.free(indices)
    first.request_pool.free(first.request)
    first.request_pool.free(second_request)


def test_failure_after_publication_boundary_retains_prepared_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A possibly armed writer keeps its allocation pinned fail closed."""

    allocator = _token_allocator()
    fixture = _fixture(allocator, source_tp_size=2)
    indices = allocator.alloc(4)
    assert indices is not None
    slot = fixture.request.req_pool_idx
    assert slot is not None
    fixture.request_pool.write((slot, slice(0, 4)), indices)

    def fail_after_publication_boundary(
        rids_to_check: list[str] | None,
        preparation: _DecodeAllocationPreparation,
    ) -> Never:
        """Inject failure after metadata may have armed a writer.

        :param rids_to_check: Unused request filter.
        :param preparation: Live cohort transaction.
        :raises RuntimeError: Always, after crossing the publication boundary.
        """

        del rids_to_check
        fixture.queue._acquire_decode_allocation_lease(
            fixture.decode_request,
            migration_start=0,
            migration_end=4,
        )
        preparation.record_prepared(fixture.decode_request)
        preparation.publication_started = True
        raise RuntimeError("injected post-publication failure")

    monkeypatch.setattr(
        fixture.queue,
        "_prepare_and_publish_preallocated",
        fail_after_publication_boundary,
    )
    with pytest.raises(RuntimeError, match="injected post-publication failure"):
        fixture.queue.pop_preallocated()

    assert fixture.decode_request.allocation_lease is not None
    with pytest.raises(AllocationPinnedError):
        allocator.free(indices)
    with pytest.raises(AllocationPinnedError):
        fixture.request_pool.free(fixture.request)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
