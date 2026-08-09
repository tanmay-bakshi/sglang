import dataclasses
import sys

import pytest
import torch

from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DECODE_ALLOCATION_COMPONENT_ORDER,
    DecodeAllocationComponent,
    DecodeAllocationComponentClaim,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLeaseError,
    DecodeAllocationLeaseState,
    DecodeAllocationLifecycleUnavailable,
    DecodeWriterManifest,
)
from sglang.srt.mem_cache.allocation_pin import (
    AllocationPinnedError,
    AllocationPinSnapshot,
)
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclasses.dataclass
class _Request:
    """Minimal request carrying request-pool allocation fields."""

    req_pool_idx: int | None = None
    inflight_middle_chunks: int = 0
    kv_committed_len: int = 0


@dataclasses.dataclass
class _LeaseFixture:
    """CPU-only local decode allocation owners used by receipt tests."""

    authority: DecodeAllocationLeaseAuthority
    lifecycle: object
    request: _Request
    request_pool: ReqToTokenPool
    full_allocator: TokenToKVPoolAllocator
    full_indices: torch.Tensor
    swa_allocator: TokenToKVPoolAllocator
    swa_indices: torch.Tensor
    mamba_allocator: MambaSlotAllocator
    mamba_indices: torch.Tensor


def _lease_fixture() -> _LeaseFixture:
    """Allocate independent CPU FULL, SWA, Mamba, and request-slot owners.

    :returns: Exact local allocation fixture.
    """

    request_pool = ReqToTokenPool(
        size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
    )
    request = _Request()
    selected_slots = request_pool.alloc([request])
    assert selected_slots is not None
    assert request.req_pool_idx is not None

    full_allocator = TokenToKVPoolAllocator(
        size=32,
        dtype=torch.float16,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    swa_allocator = TokenToKVPoolAllocator(
        size=16,
        dtype=torch.float16,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    mamba_allocator = MambaSlotAllocator(size=4, device="cpu")
    full_indices = full_allocator.alloc(4)
    swa_indices = swa_allocator.alloc(3)
    mamba_indices = mamba_allocator.alloc(1)
    assert full_indices is not None
    assert swa_indices is not None
    assert mamba_indices is not None
    lifecycle = object()
    return _LeaseFixture(
        authority=DecodeAllocationLeaseAuthority(lifecycle),
        lifecycle=lifecycle,
        request=request,
        request_pool=request_pool,
        full_allocator=full_allocator,
        full_indices=full_indices,
        swa_allocator=swa_allocator,
        swa_indices=swa_indices,
        mamba_allocator=mamba_allocator,
        mamba_indices=mamba_indices,
    )


def _claims(
    fixture: _LeaseFixture,
    *,
    include_swa: bool,
    include_mamba: bool,
) -> tuple[DecodeAllocationComponentClaim, ...]:
    """Build canonical component claims with explicit zero-work phases.

    :param fixture: Exact local allocation owners.
    :param include_swa: Whether SWA has local work.
    :param include_mamba: Whether Mamba has local work.
    :returns: FULL, SWA, and Mamba claims in fixed order.
    """

    swa_claim = (
        DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.SWA,
            logical_start=1,
            logical_length=int(fixture.swa_indices.numel()),
            allocator=fixture.swa_allocator,
            indices=fixture.swa_indices,
        )
        if include_swa
        else DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.SWA,
            logical_start=0,
            logical_length=0,
            allocator=None,
            indices=None,
        )
    )
    mamba_claim = (
        DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.MAMBA,
            logical_start=0,
            logical_length=int(fixture.mamba_indices.numel()),
            allocator=fixture.mamba_allocator,
            indices=fixture.mamba_indices,
        )
        if include_mamba
        else DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.MAMBA,
            logical_start=0,
            logical_length=0,
            allocator=None,
            indices=None,
        )
    )
    return (
        DecodeAllocationComponentClaim(
            component=DecodeAllocationComponent.FULL,
            logical_start=7,
            logical_length=int(fixture.full_indices.numel()),
            allocator=fixture.full_allocator,
            indices=fixture.full_indices,
        ),
        swa_claim,
        mamba_claim,
    )


def _acquire(
    fixture: _LeaseFixture,
    *,
    source_tp_size: int = 2,
    include_swa: bool = True,
    include_mamba: bool = False,
):
    """Acquire one generation-bound local allocation lease.

    :param fixture: Exact local allocation owners.
    :param source_tp_size: Exact supported packed source width.
    :param include_swa: Whether SWA has local work.
    :param include_mamba: Whether Mamba has local work.
    :returns: Opaque allocation lease.
    """

    request_slot = fixture.request.req_pool_idx
    assert request_slot is not None
    generation = int(fixture.request_pool.req_generation[request_slot].item())
    return fixture.authority.acquire(
        request_pool=fixture.request_pool,
        request_slot=request_slot,
        expected_request_generation=generation,
        writer_manifest=DecodeWriterManifest.for_tensor_parallel(source_tp_size),
        component_claims=_claims(
            fixture,
            include_swa=include_swa,
            include_mamba=include_mamba,
        ),
    )


def _complete_teardown(
    fixture: _LeaseFixture,
    lease: DecodeAllocationLease,
) -> None:
    """Advance one lease through authenticated terminal transport teardown.

    :param fixture: Exact local allocation owners and lifecycle authority.
    :param lease: Exact allocation lease to advance.
    """

    fixture.authority.record_publication(lease, fixture.lifecycle)
    fixture.authority.record_submission(lease, fixture.lifecycle)
    fixture.authority.record_writer_completion(lease, fixture.lifecycle)
    fixture.authority.record_scatter_completion(lease, fixture.lifecycle)
    snapshot = fixture.authority.snapshot(lease)
    fixture.authority.record_teardown_completion(
        lease,
        fixture.lifecycle,
        request_generation=snapshot.request_generation,
        writer_manifest_digest=snapshot.writer_manifest.digest,
        allocation_digest=snapshot.allocation_digest,
    )


def _alter_digest(digest: bytes) -> bytes:
    """Return a same-length digest differing in its first byte.

    :param digest: Source digest.
    :returns: Altered digest.
    """

    return bytes((digest[0] ^ 1,)) + digest[1:]


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_tp_manifest_and_component_participation_are_exact(
    source_tp_size: int,
) -> None:
    """Every supported source width retains exact ordered participation."""

    fixture = _lease_fixture()
    lease = _acquire(
        fixture,
        source_tp_size=source_tp_size,
        include_swa=True,
        include_mamba=False,
    )
    snapshot = fixture.authority.snapshot(lease)

    assert tuple(
        writer.source_attn_tp_rank for writer in snapshot.writer_manifest.writers
    ) == tuple(range(source_tp_size))
    assert tuple(receipt.component for receipt in snapshot.components) == (
        DECODE_ALLOCATION_COMPONENT_ORDER
    )
    assert len(snapshot.writer_participation) == source_tp_size * 3
    expected_zero_work = (False, False, True)
    for phase, component in enumerate(DECODE_ALLOCATION_COMPONENT_ORDER):
        participation = snapshot.writer_participation[
            phase * source_tp_size : (phase + 1) * source_tp_size
        ]
        assert all(item.generation_order == phase for item in participation)
        assert all(item.component is component for item in participation)
        assert all(
            item.zero_work is expected_zero_work[phase] for item in participation
        )
        assert tuple(
            item.writer_id.source_attn_tp_rank for item in participation
        ) == tuple(range(source_tp_size))


@pytest.mark.parametrize(
    ("source_tp_size", "destination_tp_size", "destination_tp_rank", "source_ranks"),
    (
        (1, 1, 0, (0,)),
        (2, 1, 0, (0, 1)),
        (4, 1, 0, (0, 1, 2, 3)),
        (2, 2, 0, (0,)),
        (2, 2, 1, (1,)),
        (4, 2, 0, (0, 1)),
        (4, 2, 1, (2, 3)),
    ),
)
def test_manifest_selects_exact_destination_local_writers(
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    source_ranks: tuple[int, ...],
) -> None:
    """Manifest membership follows the compatible TP routing topology."""

    manifest = DecodeWriterManifest.for_tensor_parallel(
        source_tp_size,
        destination_tp_size,
        destination_tp_rank,
    )

    assert (
        tuple(writer.source_attn_tp_rank for writer in manifest.writers) == source_ranks
    )


def test_manifest_rejects_missing_and_duplicate_writers() -> None:
    """Writer manifests reject incomplete and duplicated local membership."""

    canonical = DecodeWriterManifest.for_tensor_parallel(2)
    with pytest.raises(ValueError, match="writer count"):
        DecodeWriterManifest(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=canonical.writers[:1],
        )
    with pytest.raises(ValueError, match="duplicate"):
        DecodeWriterManifest(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
            writers=(canonical.writers[0], canonical.writers[0]),
        )


def test_receipt_uses_allocator_derived_static_physical_identity() -> None:
    """Static FULL and SWA allocators produce exact identity mappings."""

    fixture = _lease_fixture()
    lease = _acquire(fixture, include_swa=True, include_mamba=False)
    snapshot = fixture.authority.snapshot(lease)
    full, swa, mamba = snapshot.components

    assert full.virtual_pages == full.physical_pages == (1, 2, 3, 4)
    assert swa.virtual_pages == swa.physical_pages == (1, 2, 3)
    assert mamba.zero_work
    assert mamba.virtual_pages == ()
    assert mamba.physical_pages == ()


def test_receipt_preserves_request_logical_page_order() -> None:
    """Keep scatter order even when the pin registry canonicalizes page IDs."""

    claim = DecodeAllocationComponentClaim(
        component=DecodeAllocationComponent.FULL,
        logical_start=0,
        logical_length=4,
        allocator=object(),
        indices=torch.tensor([6, 7, 4, 5], dtype=torch.int64),
    )
    receipt = DecodeAllocationLeaseAuthority._component_receipt(
        claim,
        AllocationPinSnapshot(
            allocator_label="reordered",
            page_size=2,
            virtual_pages=(2, 3),
            physical_pages=(12, 13),
            quarantined=False,
        ),
    )

    assert receipt.virtual_pages == (3, 2)
    assert receipt.physical_pages == (13, 12)


def test_nonzero_mamba_is_pinned_and_gemma_zero_work_is_explicit() -> None:
    """Generic Mamba slots pin when present; Gemma-style absence stays explicit."""

    mamba_fixture = _lease_fixture()
    mamba_lease = _acquire(
        mamba_fixture,
        include_swa=False,
        include_mamba=True,
    )
    mamba_snapshot = mamba_fixture.authority.snapshot(mamba_lease)
    assert mamba_snapshot.components[2].virtual_pages == (1,)
    with pytest.raises(AllocationPinnedError, match="pinned"):
        mamba_fixture.mamba_allocator.free(mamba_fixture.mamba_indices)

    gemma_fixture = _lease_fixture()
    gemma_lease = _acquire(
        gemma_fixture,
        include_swa=True,
        include_mamba=False,
    )
    gemma_snapshot = gemma_fixture.authority.snapshot(gemma_lease)
    assert gemma_snapshot.components[2].zero_work
    assert all(
        participation.zero_work
        for participation in gemma_snapshot.writer_participation
        if participation.component is DecodeAllocationComponent.MAMBA
    )


def test_live_lease_blocks_request_and_component_reuse() -> None:
    """Migration pins reject every direct local ownership release."""

    fixture = _lease_fixture()
    _acquire(fixture)
    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.full_allocator.free(fixture.full_indices)
    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.swa_allocator.free(fixture.swa_indices)
    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.request_pool.free(fixture.request)
    with pytest.raises(AllocationPinnedError, match="clear"):
        fixture.full_allocator.clear()
    with pytest.raises(AllocationPinnedError, match="restore"):
        fixture.full_allocator.restore_state(fixture.full_allocator.backup_state())


def test_failed_component_pin_rolls_back_every_prior_owner() -> None:
    """Failed transactional preparation publishes no partial allocation pin."""

    fixture = _lease_fixture()
    conflicting_owner = object()
    conflicting_pin = fixture.swa_allocator.acquire_allocation_pin(
        fixture.swa_indices,
        conflicting_owner,
    )
    request_slot = fixture.request.req_pool_idx
    assert request_slot is not None
    generation = int(fixture.request_pool.req_generation[request_slot].item())

    with pytest.raises(AllocationPinnedError, match="already pinned"):
        fixture.authority.acquire(
            request_pool=fixture.request_pool,
            request_slot=request_slot,
            expected_request_generation=generation,
            writer_manifest=DecodeWriterManifest.for_tensor_parallel(2),
            component_claims=_claims(
                fixture,
                include_swa=True,
                include_mamba=False,
            ),
        )

    fixture.full_allocator.free(fixture.full_indices)
    fixture.request_pool.free(fixture.request)
    fixture.swa_allocator.release_allocation_pin(
        conflicting_pin,
        conflicting_owner,
    )


def test_stale_request_generation_fails_before_component_pins() -> None:
    """A stale engine generation cannot claim otherwise-live allocator pages."""

    fixture = _lease_fixture()
    request_slot = fixture.request.req_pool_idx
    assert request_slot is not None
    actual_generation = int(fixture.request_pool.req_generation[request_slot].item())
    with pytest.raises(AllocationPinnedError, match="generation changed"):
        fixture.authority.acquire(
            request_pool=fixture.request_pool,
            request_slot=request_slot,
            expected_request_generation=actual_generation + 1,
            writer_manifest=DecodeWriterManifest.for_tensor_parallel(2),
            component_claims=_claims(
                fixture,
                include_swa=True,
                include_mamba=False,
            ),
        )
    fixture.full_allocator.free(fixture.full_indices)
    fixture.request_pool.free(fixture.request)


def test_commit_returns_migration_pin_without_freeing_request_storage() -> None:
    """Successful teardown commits ownership without changing allocator capacity."""

    fixture = _lease_fixture()
    full_available_after_alloc = fixture.full_allocator.available_size()
    request_available_after_alloc = fixture.request_pool.available_size()
    lease = _acquire(fixture)
    _complete_teardown(fixture, lease)
    fixture.authority.commit_to_request_after_teardown(
        lease,
        fixture.lifecycle,
    )

    snapshot = fixture.authority.snapshot(lease)
    assert snapshot.state is DecodeAllocationLeaseState.COMMITTED_TO_REQUEST
    assert fixture.full_allocator.available_size() == full_available_after_alloc
    assert fixture.request_pool.available_size() == request_available_after_alloc

    fixture.authority.retire_terminal(lease)
    with pytest.raises(DecodeAllocationLeaseError, match="not registered"):
        fixture.authority.snapshot(lease)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)
    assert fixture.full_allocator.available_size() == 32
    assert fixture.request_pool.available_size() == 4


def test_legacy_consumption_commit_returns_published_pins() -> None:
    """A consumed legacy transfer returns ownership to normal request cleanup."""

    fixture = _lease_fixture()
    full_available_after_alloc = fixture.full_allocator.available_size()
    request_available_after_alloc = fixture.request_pool.available_size()
    lease = _acquire(fixture)
    fixture.authority.record_publication(lease, fixture.lifecycle)

    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.request_pool.free(fixture.request)
    with pytest.raises(DecodeAllocationLeaseError, match="exact lifecycle"):
        fixture.authority.commit_legacy_to_request_after_consumption(
            lease,
            object(),
        )
    assert (
        fixture.authority.snapshot(lease).state
        is DecodeAllocationLeaseState.PUBLISHED
    )

    fixture.authority.commit_legacy_to_request_after_consumption(
        lease,
        fixture.lifecycle,
    )
    assert (
        fixture.authority.snapshot(lease).state
        is DecodeAllocationLeaseState.COMMITTED_TO_REQUEST
    )
    assert fixture.full_allocator.available_size() == full_available_after_alloc
    assert fixture.request_pool.available_size() == request_available_after_alloc

    fixture.authority.retire_terminal(lease)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)
    assert fixture.full_allocator.available_size() == 32
    assert fixture.request_pool.available_size() == 4


def test_legacy_consumption_commit_cannot_bypass_lifecycle_edges() -> None:
    """Prepared and submitted transactions cannot use the legacy success edge."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    with pytest.raises(DecodeAllocationLeaseError, match="published transfer"):
        fixture.authority.commit_legacy_to_request_after_consumption(
            lease,
            fixture.lifecycle,
        )

    fixture.authority.record_publication(lease, fixture.lifecycle)
    fixture.authority.record_submission(lease, fixture.lifecycle)
    with pytest.raises(DecodeAllocationLeaseError, match="published transfer"):
        fixture.authority.commit_legacy_to_request_after_consumption(
            lease,
            fixture.lifecycle,
        )


def test_legacy_terminal_failure_authorizes_canonical_abort_cleanup() -> None:
    """A terminal legacy failure unpins only through a consumed abort permit."""

    fixture = _lease_fixture()
    full_available_after_alloc = fixture.full_allocator.available_size()
    request_available_after_alloc = fixture.request_pool.available_size()
    lease = _acquire(fixture)
    fixture.authority.record_publication(lease, fixture.lifecycle)

    with pytest.raises(DecodeAllocationLeaseError, match="exact lifecycle"):
        fixture.authority.authorize_legacy_abort_after_terminal_failure(
            lease,
            object(),
        )
    permit = fixture.authority.authorize_legacy_abort_after_terminal_failure(
        lease,
        fixture.lifecycle,
    )
    assert (
        fixture.authority.snapshot(lease).state
        is DecodeAllocationLeaseState.ABORT_AUTHORIZED
    )
    assert fixture.full_allocator.available_size() == full_available_after_alloc
    assert fixture.request_pool.available_size() == request_available_after_alloc
    with pytest.raises(DecodeAllocationLeaseError, match="permit consumption"):
        fixture.authority.retire_terminal(lease)

    fixture.authority.consume_abort_permit(lease, permit)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)
    fixture.authority.retire_terminal(lease)
    assert fixture.full_allocator.available_size() == 32
    assert fixture.request_pool.available_size() == 4


def test_legacy_terminal_failure_cannot_bypass_lifecycle_edges() -> None:
    """Only published legacy allocations can use the terminal failure edge."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    with pytest.raises(DecodeAllocationLeaseError, match="published transfer"):
        fixture.authority.authorize_legacy_abort_after_terminal_failure(
            lease,
            fixture.lifecycle,
        )

    fixture.authority.record_publication(lease, fixture.lifecycle)
    fixture.authority.record_submission(lease, fixture.lifecycle)
    with pytest.raises(DecodeAllocationLeaseError, match="published transfer"):
        fixture.authority.authorize_legacy_abort_after_terminal_failure(
            lease,
            fixture.lifecycle,
        )


def test_pre_submission_rollback_returns_pin_to_live_request() -> None:
    """Rollback removes migration ownership without performing allocator free."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    available_after_alloc = fixture.full_allocator.available_size()
    fixture.authority.rollback_to_request(lease)
    assert (
        fixture.authority.snapshot(lease).state
        is DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST
    )
    assert fixture.full_allocator.available_size() == available_after_alloc
    fixture.authority.retire_terminal(lease)
    with pytest.raises(DecodeAllocationLeaseError, match="not registered"):
        fixture.authority.snapshot(lease)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)


def test_pre_submission_abort_permit_is_take_once() -> None:
    """Prepared allocations require one consumed permit before engine cleanup."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    permit = fixture.authority.authorize_pre_submission_abort(lease)
    fixture.authority.consume_abort_permit(lease, permit)
    with pytest.raises(DecodeAllocationLeaseError, match="already consumed"):
        fixture.authority.consume_abort_permit(lease, permit)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)
    fixture.authority.retire_terminal(lease)


def test_post_submission_abort_requires_exact_authenticated_teardown() -> None:
    """Only exact terminal transport authority can authorize engine cleanup."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    _complete_teardown(fixture, lease)
    with pytest.raises(DecodeAllocationLeaseError, match="exact lifecycle"):
        fixture.authority.authorize_abort_after_teardown(lease, object())
    permit = fixture.authority.authorize_abort_after_teardown(
        lease,
        fixture.lifecycle,
    )
    fixture.authority.consume_abort_permit(lease, permit)
    fixture.full_allocator.free(fixture.full_indices)
    fixture.swa_allocator.free(fixture.swa_indices)
    fixture.request_pool.free(fixture.request)
    fixture.authority.retire_terminal(lease)


def test_teardown_authenticates_generation_manifest_and_allocation() -> None:
    """Terminal acknowledgement cannot be replayed across routes or storage."""

    fixture = _lease_fixture()
    lease = _acquire(fixture, source_tp_size=4)
    fixture.authority.record_publication(lease, fixture.lifecycle)
    fixture.authority.record_submission(lease, fixture.lifecycle)
    fixture.authority.record_writer_completion(lease, fixture.lifecycle)
    fixture.authority.record_scatter_completion(lease, fixture.lifecycle)
    snapshot = fixture.authority.snapshot(lease)

    with pytest.raises(DecodeAllocationLeaseError, match="generation"):
        fixture.authority.record_teardown_completion(
            lease,
            fixture.lifecycle,
            request_generation=snapshot.request_generation + 1,
            writer_manifest_digest=snapshot.writer_manifest.digest,
            allocation_digest=snapshot.allocation_digest,
        )
    with pytest.raises(DecodeAllocationLeaseError, match="writer manifest"):
        fixture.authority.record_teardown_completion(
            lease,
            fixture.lifecycle,
            request_generation=snapshot.request_generation,
            writer_manifest_digest=_alter_digest(snapshot.writer_manifest.digest),
            allocation_digest=snapshot.allocation_digest,
        )
    with pytest.raises(DecodeAllocationLeaseError, match="allocation digest"):
        fixture.authority.record_teardown_completion(
            lease,
            fixture.lifecycle,
            request_generation=snapshot.request_generation,
            writer_manifest_digest=snapshot.writer_manifest.digest,
            allocation_digest=_alter_digest(snapshot.allocation_digest),
        )
    fixture.authority.record_teardown_completion(
        lease,
        fixture.lifecycle,
        request_generation=snapshot.request_generation,
        writer_manifest_digest=snapshot.writer_manifest.digest,
        allocation_digest=snapshot.allocation_digest,
    )
    assert (
        fixture.authority.snapshot(lease).state
        is DecodeAllocationLeaseState.TEARDOWN_COMPLETED
    )


def test_unbound_transport_lifecycle_hard_gates_submission() -> None:
    """Allocator-only authorities cannot assert that native submission occurred."""

    fixture = _lease_fixture()
    authority = DecodeAllocationLeaseAuthority()
    request_slot = fixture.request.req_pool_idx
    assert request_slot is not None
    generation = int(fixture.request_pool.req_generation[request_slot].item())
    lease = authority.acquire(
        request_pool=fixture.request_pool,
        request_slot=request_slot,
        expected_request_generation=generation,
        writer_manifest=DecodeWriterManifest.for_tensor_parallel(2),
        component_claims=_claims(
            fixture,
            include_swa=True,
            include_mamba=False,
        ),
    )
    with pytest.raises(DecodeAllocationLifecycleUnavailable, match="not bound"):
        authority.record_publication(lease, object())


def test_phase_order_is_exact_and_one_shot() -> None:
    """Completion, scatter, and teardown cannot skip or replay lifecycle edges."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    with pytest.raises(DecodeAllocationLeaseError, match="invalid"):
        fixture.authority.record_writer_completion(lease, fixture.lifecycle)
    fixture.authority.record_publication(lease, fixture.lifecycle)
    assert (
        fixture.authority.snapshot(lease).state is DecodeAllocationLeaseState.PUBLISHED
    )
    fixture.authority.record_submission(lease, fixture.lifecycle)
    with pytest.raises(DecodeAllocationLeaseError, match="invalid"):
        fixture.authority.record_publication(lease, fixture.lifecycle)
    with pytest.raises(DecodeAllocationLeaseError, match="invalid"):
        fixture.authority.record_submission(lease, fixture.lifecycle)
    fixture.authority.record_writer_completion(lease, fixture.lifecycle)
    with pytest.raises(DecodeAllocationLeaseError, match="authenticated"):
        fixture.authority.commit_to_request_after_teardown(
            lease,
            fixture.lifecycle,
        )
    fixture.authority.record_scatter_completion(lease, fixture.lifecycle)
    snapshot = fixture.authority.snapshot(lease)
    fixture.authority.record_teardown_completion(
        lease,
        fixture.lifecycle,
        request_generation=snapshot.request_generation,
        writer_manifest_digest=snapshot.writer_manifest.digest,
        allocation_digest=snapshot.allocation_digest,
    )
    with pytest.raises(DecodeAllocationLeaseError, match="invalid"):
        fixture.authority.record_teardown_completion(
            lease,
            fixture.lifecycle,
            request_generation=snapshot.request_generation,
            writer_manifest_digest=snapshot.writer_manifest.digest,
            allocation_digest=snapshot.allocation_digest,
        )


def test_ambiguity_quarantines_complete_local_allocation() -> None:
    """Ambiguous submission permanently blocks local free, clear, and commit."""

    fixture = _lease_fixture()
    lease = _acquire(fixture)
    fixture.authority.record_publication(lease, fixture.lifecycle)
    fixture.authority.record_submission(lease, fixture.lifecycle)
    fixture.authority.quarantine(lease, "native handle terminality is unknown")
    snapshot = fixture.authority.snapshot(lease)
    assert snapshot.state is DecodeAllocationLeaseState.QUARANTINED
    assert snapshot.failure_reason == "native handle terminality is unknown"
    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.full_allocator.free(fixture.full_indices)
    with pytest.raises(AllocationPinnedError, match="pinned"):
        fixture.request_pool.free(fixture.request)
    with pytest.raises(DecodeAllocationLeaseError, match="authenticated"):
        fixture.authority.commit_to_request_after_teardown(
            lease,
            fixture.lifecycle,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
