import dataclasses
import hashlib
import sys
import threading
import uuid
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationLeaseError,
    DecodeAllocationLeaseState,
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryPlan,
    PackedRequestKey,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodePreparedAllocationCohort,
    DecodeRequest,
    _DecodeMetadataSubmission,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodePrefixMatch,
    DecodeRestoreBudget,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAttempt,
    DecodeReservationBootstrapEndpoint,
    DecodeReservationProcess,
    DecodeReservationState,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedRequestPublication,
)
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclasses.dataclass
class _FakeLease:
    """Minimal mutable migration lease used by queue lifecycle tests.

    :ivar lease_id: Exact 16-byte generation.
    :ivar request_slot: Reserved request slot.
    :ivar request_generation: Reserved slot generation.
    :ivar source_tp_size: Exact source writer width.
    :ivar state: Current fake allocation state.
    :ivar quarantined_reason: First retained failure reason.
    """

    lease_id: bytes
    request_slot: int
    request_generation: int
    source_tp_size: int
    state: DecodeAllocationLeaseState = DecodeAllocationLeaseState.PREPARED
    quarantined_reason: str | None = None


class _FakeAllocationAuthority:
    """Allocation authority with the exact transitions consumed by the queue."""

    lifecycle: object
    retired: list[_FakeLease]

    def __init__(self, lifecycle: object) -> None:
        """Initialize an empty fake authority.

        :param lifecycle: Exact trusted lifecycle object.
        """

        self.lifecycle = lifecycle
        self.retired = []

    def snapshot(self, lease: _FakeLease) -> SimpleNamespace:
        """Build one immutable-enough allocation snapshot.

        :param lease: Exact fake lease.
        :returns: Queue-consumed snapshot fields.
        """

        manifest = DecodeWriterManifest.for_tensor_parallel(lease.source_tp_size)
        allocation_digest = hashlib.sha256(lease.lease_id).digest()
        return SimpleNamespace(
            lease_id=lease.lease_id,
            request_slot=lease.request_slot,
            request_generation=lease.request_generation,
            writer_manifest=manifest,
            allocation_digest=allocation_digest,
            state=lease.state,
        )

    def record_publication(self, lease: _FakeLease, lifecycle: object) -> None:
        """Advance one prepared lease to irreversible publication.

        :param lease: Exact fake lease.
        :param lifecycle: Candidate trusted lifecycle.
        """

        assert lifecycle is self.lifecycle
        assert lease.state is DecodeAllocationLeaseState.PREPARED
        lease.state = DecodeAllocationLeaseState.PUBLISHED

    def rollback_to_request(self, lease: _FakeLease) -> None:
        """Release migration ownership back to an unpublished request.

        :param lease: Exact fake lease.
        """

        assert lease.state is DecodeAllocationLeaseState.PREPARED
        lease.state = DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST

    def retire_terminal(self, lease: _FakeLease) -> None:
        """Forget one terminal fake lease.

        :param lease: Exact fake lease.
        """

        assert lease.state in (
            DecodeAllocationLeaseState.ROLLED_BACK_TO_REQUEST,
            DecodeAllocationLeaseState.COMMITTED_TO_REQUEST,
        )
        self.retired.append(lease)

    def quarantine(self, lease: _FakeLease, reason: str) -> None:
        """Retain one ambiguous fake lease.

        :param lease: Exact fake lease.
        :param reason: Stable quarantine reason.
        """

        if lease.state is DecodeAllocationLeaseState.QUARANTINED:
            return
        lease.state = DecodeAllocationLeaseState.QUARANTINED
        lease.quarantined_reason = reason


class _FakePackedTransaction:
    """Request-scoped publication owner used by queue integration tests."""

    authority: _FakeAllocationAuthority
    cancel_count: int
    cancel_entered: threading.Event | None
    cancel_release: threading.Event | None
    fail_publication_after_transition: bool
    lease: _FakeLease
    metadata_buffer_index: int
    publication: PackedRequestPublication
    publish_count: int
    quarantine_count: int
    request_owner: DecodeRequest

    def __init__(
        self,
        *,
        room_id: int,
        request_owner: DecodeRequest,
        metadata_buffer_index: int,
        lease: _FakeLease,
        authority: _FakeAllocationAuthority,
    ) -> None:
        """Bind one fake transaction to an exact prepared lease.

        :param room_id: Decoder-minted bootstrap room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Reserved auxiliary destination slot.
        :param lease: Prepared fake allocation lease.
        :param authority: Exact fake allocation authority.
        """

        snapshot = authority.snapshot(lease)
        assert snapshot.state is DecodeAllocationLeaseState.PREPARED
        self.authority = authority
        self.cancel_count = 0
        self.cancel_entered = None
        self.cancel_release = None
        self.fail_publication_after_transition = False
        self.lease = lease
        self.metadata_buffer_index = metadata_buffer_index
        request_key = PackedRequestKey(
            room_id=room_id,
            request_generation=snapshot.lease_id,
        )
        self.publication = PackedRequestPublication(
            key=request_key,
            request_slot_generation=snapshot.request_generation,
            writer_manifest_digest=snapshot.writer_manifest.digest,
            allocation_digest=snapshot.allocation_digest,
            auxiliary_plan=PackedAuxiliaryPlan(
                key=request_key,
                request_slot_generation=snapshot.request_generation,
                metadata_buffer_index=metadata_buffer_index,
                metadata_slot_generation=snapshot.lease_id,
                destination_segments=(
                    PackedAuxiliaryDestinationSegment(
                        address=0x100000 + metadata_buffer_index * 0x1000,
                        item_length=256,
                    ),
                ),
                canonical_writer_id=snapshot.writer_manifest.writers[0],
                destination_process_generation=b"p" * 16,
                native_route_digest=hashlib.sha256(b"fake native route").digest(),
                runtime_cohort_digest=hashlib.sha256(b"fake runtime cohort").digest(),
            ),
            chunk_specs=(),
        )
        self.publish_count = 0
        self.quarantine_count = 0
        self.request_owner = request_owner

    def publish(self) -> PackedRequestPublication:
        """Cross the sole fake publication boundary.

        :returns: Exact immutable publication.
        """

        self.publish_count += 1
        self.authority.record_publication(self.lease, self.authority.lifecycle)
        if self.fail_publication_after_transition:
            raise RuntimeError("injected packed publication failure")
        return self.publication

    def cancel_unpublished(self) -> DecodeRequest:
        """Retire an unpublished fake lease and return its owner.

        :returns: Exact retained decode request.
        """

        self.cancel_count += 1
        if self.cancel_entered is not None:
            self.cancel_entered.set()
        if self.cancel_release is not None:
            assert self.cancel_release.wait(timeout=5)
        self.authority.rollback_to_request(self.lease)
        self.authority.retire_terminal(self.lease)
        return self.request_owner

    def quarantine(self, reason: str) -> None:
        """Retain the complete fake transaction after ambiguity.

        :param reason: Stable quarantine reason.
        """

        self.quarantine_count += 1
        self.authority.quarantine(self.lease, reason)


class _FakePackedRuntimeManager:
    """Explicit packed runtime seam consumed by the decode queue."""

    authority: _FakeAllocationAuthority
    metadata_publications: list[tuple[_FakePackedTransaction, object]]
    transactions: list[_FakePackedTransaction]

    def __init__(self, authority: _FakeAllocationAuthority) -> None:
        """Initialize an available fake runtime.

        :param authority: Exact fake allocation authority.
        """

        self.authority = authority
        self.attn_tp_rank = 0
        self.attn_tp_size = 1
        self.metadata_publications = []
        self.transactions = []

    def supports_packed_decode_request_transactions(self) -> bool:
        """Return whether request-scoped packed ownership is initialized.

        :returns: Always ``True`` for this fake runtime.
        """

        return True

    def prepare_packed_decode_request_transaction(
        self,
        *,
        room_id: int,
        request_owner: DecodeRequest,
        metadata_buffer_index: int,
        allocation_lease: _FakeLease,
        allocation_authority: _FakeAllocationAuthority,
        lifecycle_authority: object,
        source_tp_size: int,
    ) -> _FakePackedTransaction:
        """Construct one transaction while its lease is still prepared.

        :param room_id: Decoder-minted bootstrap room.
        :param request_owner: Exact retained decode request.
        :param metadata_buffer_index: Reserved auxiliary destination slot.
        :param allocation_lease: Prepared fake allocation lease.
        :param allocation_authority: Exact fake allocation authority.
        :param lifecycle_authority: Exact lifecycle transition authority.
        :param source_tp_size: Supported packed source width.
        :returns: Prepared fake transaction.
        """

        assert allocation_authority is self.authority
        assert lifecycle_authority is self.authority.lifecycle
        assert source_tp_size in (1, 2, 4)
        transaction = _FakePackedTransaction(
            room_id=room_id,
            request_owner=request_owner,
            metadata_buffer_index=metadata_buffer_index,
            lease=allocation_lease,
            authority=allocation_authority,
        )
        self.transactions.append(transaction)
        return transaction

    def cancel_unpublished_packed_decode_request_transaction(
        self,
        transaction: _FakePackedTransaction,
    ) -> DecodeRequest:
        """Cancel one unpublished transaction through the runtime owner.

        :param transaction: Exact fake transaction to retire.
        :returns: Exact retained decode request.
        """

        assert any(owned is transaction for owned in self.transactions)
        return transaction.cancel_unpublished()

    def send_packed_decode_request_metadata(
        self,
        *,
        transaction: _FakePackedTransaction,
        publication: PackedRequestPublication,
        receiver: object,
        page_indices: np.ndarray,
        metadata_buffer_index: int,
        state_indices: list[object] | None,
        decode_prefix_len: int,
    ) -> None:
        """Record metadata entry into the packed actor instead of legacy send.

        :param transaction: Exact published fake transaction.
        :param publication: Matching irreversible publication.
        :param receiver: Exact retained receiver.
        :param page_indices: Destination main-KV pages.
        :param metadata_buffer_index: Reserved auxiliary slot.
        :param state_indices: Destination state pages.
        :param decode_prefix_len: Decoder-reused prefix length.
        """

        del page_indices, state_indices, decode_prefix_len
        assert transaction.publication is publication
        assert transaction.metadata_buffer_index == metadata_buffer_index
        self.metadata_publications.append((transaction, receiver))


class _FakeReceiver:
    """Receiver recording initialization and cleanup without network work."""

    require_staging: bool = True
    prefill_info: SimpleNamespace
    init_ranks: list[int]
    clear_count: int
    abort_count: int
    metadata_count: int

    def __init__(self, source_tp_size: int) -> None:
        """Initialize one receiver.

        :param source_tp_size: Handshake-reported source width.
        """

        self.prefill_info = SimpleNamespace(attn_tp_size=source_tp_size)
        self.init_ranks = []
        self.clear_count = 0
        self.abort_count = 0
        self.metadata_count = 0

    def init(self, rank: int) -> None:
        """Record asynchronous receiver initialization.

        :param rank: Selected prefill DP rank.
        """

        self.init_ranks.append(rank)

    def clear(self) -> None:
        """Record terminal unpublished cleanup."""

        self.clear_count += 1

    def abort(self) -> None:
        """Record conservative quarantine activation."""

        self.abort_count += 1

    def send_metadata(self, *args: object, **kwargs: object) -> None:
        """Record one metadata publication.

        :param args: Positional metadata payload.
        :param kwargs: Keyword metadata payload.
        """

        del args, kwargs
        self.metadata_count += 1


class _FakeMetadataAllocator:
    """Small exact metadata-slot allocator with observable ownership."""

    free_slots: list[int]
    allocated: list[int]

    def __init__(self, size: int) -> None:
        """Initialize a finite allocator.

        :param size: Number of available metadata slots.
        """

        self.free_slots = list(range(size))
        self.allocated = []

    def available_size(self) -> int:
        """Return current free capacity.

        :returns: Number of free metadata slots.
        """

        return len(self.free_slots)

    def alloc(self) -> int | None:
        """Take one metadata slot.

        :returns: Fresh slot, otherwise ``None``.
        """

        if len(self.free_slots) == 0:
            return None
        slot = self.free_slots.pop(0)
        self.allocated.append(slot)
        return slot

    def free(self, slot: int) -> None:
        """Return one exact metadata slot.

        :param slot: Exact live slot.
        """

        self.allocated.remove(slot)
        self.free_slots.append(slot)


@dataclasses.dataclass
class _QueueFixture:
    """Minimal queue and observable resource state for lifecycle tests.

    :ivar queue: Queue under test.
    :ivar authority: Fake migration authority.
    :ivar packed_runtime: Fake request-scoped transfer runtime.
    :ivar metadata_allocator: Exact metadata allocator.
    :ivar receivers: Receivers created by preparation.
    :ivar released_request_ids: Requests released by rollback.
    :ivar pre_alloc_calls: Source widths passed into physical preparation.
    """

    queue: DecodePreallocQueue
    authority: _FakeAllocationAuthority
    packed_runtime: _FakePackedRuntimeManager
    metadata_allocator: _FakeMetadataAllocator
    receivers: list[_FakeReceiver]
    released_request_ids: list[str]
    pre_alloc_calls: list[int]


def _request(child_id: uuid.UUID, *, prompt_tokens: int = 8) -> Req:
    """Build one canonical minimally populated scheduler request.

    :param child_id: Exact request identity.
    :param prompt_tokens: Prompt token count.
    :returns: Canonical Req instance for queue admission.
    """

    req = object.__new__(Req)
    req.rid = str(child_id)
    req.req_pool_idx = None
    req.kv = None
    req.mamba_pool_idx = None
    req.bootstrap_host = None
    req.bootstrap_port = None
    req.bootstrap_room = None
    req.origin_input_ids = array("q", range(prompt_tokens))
    req.output_ids = array("q")
    req.sampling_params = SimpleNamespace(max_new_tokens=32)
    req.time_stats = MagicMock()
    req.extra_key = None
    req.last_node = None
    req.last_node_lock_params = None
    req.last_host_node = None
    req.best_match_node = None
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.mamba_host_hit_length = 0
    req.num_matched_prefix_tokens = 0
    req.cache_protected_len = 0
    req.finished_reason = None
    req.return_logprob = False
    return req


def _attempt(
    child_ids: tuple[uuid.UUID, ...],
    *,
    source_tp_size: int,
) -> DecodeReservationAttempt:
    """Build one exact reservation attempt for a child cohort.

    :param child_ids: Ordered request identities.
    :param source_tp_size: Supported packed source width.
    :returns: Reservation attempt consumed by the queue.
    """

    prefill_process = DecodeReservationProcess(
        url="http://prefill:30000",
        instance_id=uuid.uuid4(),
    )
    return DecodeReservationAttempt(
        prefill_process=prefill_process,
        prefill_bootstrap_endpoint=DecodeReservationBootstrapEndpoint(
            host="prefill.internal",
            port=8998,
        ),
        decoder_process=DecodeReservationProcess(
            url="http://decode:30001",
            instance_id=uuid.uuid4(),
        ),
        logical_request_chain_id=uuid.uuid4(),
        reservation_attempt_id=uuid.uuid4(),
        reserve_attempt_digest=b"r" * 32,
        source_tp_size=source_tp_size,
        prepared_ttl_ms=10_000,
        inference_route="/generate",
        request_shape="scalar" if len(child_ids) == 1 else "batch",
        base_request_body=b"{}",
        child_request_ids=child_ids,
    )


def _queue_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_slots: int = 16,
) -> _QueueFixture:
    """Build one queue with deterministic CPU-only resource ownership.

    :param monkeypatch: Pytest mutation fixture.
    :param metadata_slots: Available metadata slots.
    :returns: Queue and observable fake resources.
    """

    queue = object.__new__(DecodePreallocQueue)
    queue.tp_size = 1
    queue.enable_staging = True
    queue.transfer_backend = TransferBackend.NIXL
    queue.num_reserved_decode_tokens = 4
    queue.max_total_num_tokens = 4_096
    queue.queue = []
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue._prepared_cohort_nonce = object()
    queue._prepared_cohort_lock = threading.RLock()
    queue._prepared_cohorts = {}
    queue._prepared_grant_ids = {}
    queue._prepared_request_ids = {}
    queue._partial_preparation_quarantines = []
    queue._preparing_grant_ids = set()
    queue._preparing_request_ids = set()
    queue._seen_bootstrap_rooms = set()

    def free_request_slot(req: Req) -> None:
        """Release one partially prepared request slot.

        :param req: Exact request owning the slot.
        """

        req.req_pool_idx = None

    queue.req_to_token_pool = SimpleNamespace(
        available_size=lambda: 16,
        free=MagicMock(side_effect=free_request_slot),
    )
    metadata_allocator = _FakeMetadataAllocator(metadata_slots)
    queue.req_to_metadata_buffer_idx_allocator = metadata_allocator
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
    root_node = object()
    queue.tree_cache = SimpleNamespace(
        root_node_handle=MagicMock(return_value=root_node),
        dec_lock_ref=MagicMock(),
        release_aborted_request=MagicMock(),
    )
    queue.transfer_queue = SimpleNamespace(
        queue=[],
        enable_staging=False,
    )
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_transfer_backend="nixl",
            disaggregation_decode_enable_radix_cache=True,
            disable_radix_cache=False,
        ),
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        enable_priority_scheduling=False,
        enable_hisparse=False,
        enable_decode_hicache=False,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
    )
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue._allocatable_token_budgets = MagicMock(return_value=100_000)
    queue._hicache_pending_restore_budgets = MagicMock(
        return_value=DecodeRestoreBudget()
    )
    queue._resolve_prefill_dp_rank = MagicMock(return_value=None)

    def match_prefix(req: Req) -> DecodePrefixMatch:
        """Return a locked zero-length radix match for prepared admission.

        :param req: Request receiving the match state.
        :returns: Reservation-owned root match.
        """

        lock_params = DecLockRefParams()
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.last_node = root_node
        req.last_node_lock_params = lock_params
        req.last_host_node = root_node
        req.best_match_node = root_node
        return DecodePrefixMatch(
            prefix_indices=req.prefix_indices,
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=root_node,
            last_device_lock_params=lock_params,
        )

    queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match_prefix)

    lifecycle = object()
    authority = _FakeAllocationAuthority(lifecycle)
    queue.allocation_lifecycle_authority = lifecycle
    queue.allocation_lease_authority = authority
    packed_runtime = _FakePackedRuntimeManager(authority)
    queue.kv_manager = packed_runtime
    receivers: list[_FakeReceiver] = []
    released_request_ids: list[str] = []
    pre_alloc_calls: list[int] = []

    def create_receiver(req: Req, *, is_rebootstrap: bool = False) -> DecodeRequest:
        """Create one observable fake receiver.

        :param req: Exact canonical request.
        :param is_rebootstrap: Unused rebootstrap marker.
        :returns: Unenqueued decode request.
        """

        del is_rebootstrap
        receiver = _FakeReceiver(source_tp_size=2)
        receivers.append(receiver)
        return DecodeRequest(req=req, kv_receiver=receiver)

    next_slot = 1

    def pre_alloc(
        req: Req,
        *args: object,
        decode_req: DecodeRequest,
        migration_end: int,
        source_tp_size: int,
        **kwargs: object,
    ) -> np.ndarray:
        """Issue one deterministic fake migration lease.

        :param req: Exact canonical request.
        :param args: Unused positional allocation inputs.
        :param decode_req: Exact child receiving the lease.
        :param migration_end: Exact transferred prompt end.
        :param source_tp_size: Reservation-authorized writer width.
        :param kwargs: Unused keyword allocation inputs.
        :returns: Fake destination indices.
        """

        nonlocal next_slot
        del args, kwargs
        assert migration_end == len(req.origin_input_ids)
        assert req.last_node is not None
        pre_alloc_calls.append(source_tp_size)
        req.req_pool_idx = next_slot
        req.kv = object()
        decode_req.allocation_lease = _FakeLease(
            lease_id=next_slot.to_bytes(16, "big"),
            request_slot=next_slot,
            request_generation=1,
            source_tp_size=source_tp_size,
        )
        next_slot += 1
        return np.arange(migration_end, dtype=np.int64)

    def release(req: Req, tree_cache: object, *, is_insert: bool) -> None:
        """Release one fake request allocation.

        :param req: Exact allocated request.
        :param tree_cache: Exact fake tree owner.
        :param is_insert: Whether cleanup inserts cache state.
        """

        assert tree_cache is queue.tree_cache
        assert not is_insert
        released_request_ids.append(req.rid)
        req.req_pool_idx = None
        req.kv = None

    queue._create_receiver = create_receiver
    queue._pre_alloc = pre_alloc
    monkeypatch.setattr("sglang.srt.disaggregation.decode.release_kv_cache", release)
    return _QueueFixture(
        queue=queue,
        authority=authority,
        packed_runtime=packed_runtime,
        metadata_allocator=metadata_allocator,
        receivers=receivers,
        released_request_ids=released_request_ids,
        pre_alloc_calls=pre_alloc_calls,
    )


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_prepare_owns_exact_resources_without_queue_publication(
    monkeypatch: pytest.MonkeyPatch,
    source_tp_size: int,
) -> None:
    """Prepare reserves every child atomically without starting receivers."""

    fixture = _queue_fixture(monkeypatch)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    requests = tuple(_request(child_id) for child_id in child_ids)
    attempt = _attempt(child_ids, source_tp_size=source_tp_size)

    allocations, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=attempt,
        requests=requests,
    )

    assert type(cohort) is DecodePreparedAllocationCohort
    assert fixture.pre_alloc_calls == [source_tp_size, source_tp_size]
    assert fixture.queue.queue == []
    assert fixture.queue.pending_reqs == []
    assert all(receiver.init_ranks == [] for receiver in fixture.receivers)
    assert tuple(allocation.child_request_id for allocation in allocations) == child_ids
    assert len({allocation.bootstrap_room for allocation in allocations}) == 2
    assert all(allocation.bootstrap_room >= 1 << 63 for allocation in allocations)
    assert tuple(allocation.request_slot for allocation in allocations) == (1, 2)
    assert fixture.queue._seen_bootstrap_rooms == {
        allocation.bootstrap_room for allocation in allocations
    }
    assert len(fixture.packed_runtime.transactions) == 2
    assert [
        transaction.metadata_buffer_index
        for transaction in fixture.packed_runtime.transactions
    ] == [0, 1]
    assert all(
        transaction.publish_count == 0
        for transaction in fixture.packed_runtime.transactions
    )


def test_simultaneous_cohorts_keep_independent_keys_and_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many prepared cohorts coexist without room or request ownership reuse."""

    fixture = _queue_fixture(monkeypatch)
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    first_allocations, first = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((first_id,), source_tp_size=2),
        requests=(_request(first_id),),
    )
    second_allocations, second = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((second_id,), source_tp_size=4),
        requests=(_request(second_id),),
    )

    assert first is not second
    assert first_allocations[0].bootstrap_room != second_allocations[0].bootstrap_room
    assert fixture.pre_alloc_calls == [2, 4]
    assert len(fixture.queue._prepared_cohorts) == 2


def test_child_failure_rolls_back_the_complete_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later child failure releases every receiver, metadata slot, slot, and KV."""

    fixture = _queue_fixture(monkeypatch)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    requests = tuple(_request(child_id) for child_id in child_ids)
    successful_pre_alloc = fixture.queue._pre_alloc
    call_count = 0

    def fail_second(*args: object, **kwargs: object) -> np.ndarray:
        """Fail the second physical child preparation.

        :param args: Forwarded positional inputs.
        :param kwargs: Forwarded keyword inputs.
        :returns: First child's fake allocation.
        :raises RuntimeError: On the second child.
        """

        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected second-child failure")
        return successful_pre_alloc(*args, **kwargs)

    fixture.queue._pre_alloc = fail_second
    with pytest.raises(RuntimeError, match="injected second-child failure"):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=requests,
        )

    assert fixture.metadata_allocator.allocated == []
    assert [receiver.clear_count for receiver in fixture.receivers] == [1, 1]
    assert fixture.released_request_ids == [requests[0].rid]
    assert all(req.req_pool_idx is None for req in requests)
    assert fixture.queue._prepared_cohorts == {}
    assert fixture.queue._preparing_request_ids == set()
    assert fixture.queue._seen_bootstrap_rooms == {
        request.bootstrap_room for request in requests
    }
    assert len(fixture.authority.retired) == 1
    assert fixture.packed_runtime.transactions[0].cancel_count == 1


def test_rollback_failure_retains_partial_preparation_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous rollback owners block destructive idleness process-lifetime."""

    fixture = _queue_fixture(monkeypatch)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    requests = tuple(_request(child_id) for child_id in child_ids)
    successful_pre_alloc = fixture.queue._pre_alloc
    call_count = 0

    def fail_second(*args: object, **kwargs: object) -> np.ndarray:
        """Fail after the first child acquired its complete transaction.

        :param args: Forwarded positional inputs.
        :param kwargs: Forwarded keyword inputs.
        :returns: First child's fake allocation.
        :raises RuntimeError: On the second child.
        """

        nonlocal call_count
        call_count += 1
        if call_count == 2:
            fixture.packed_runtime.transactions[0].cancel_unpublished = MagicMock(
                side_effect=RuntimeError("injected cancellation failure")
            )
            raise RuntimeError("injected second-child failure")
        return successful_pre_alloc(*args, **kwargs)

    fixture.queue._pre_alloc = fail_second
    with pytest.raises(
        DecodeAllocationLeaseError,
        match="partial ownership was quarantined",
    ):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=requests,
        )

    assert fixture.queue.has_live_preallocated_cohorts()
    assert len(fixture.queue._partial_preparation_quarantines) == 1
    quarantine = fixture.queue._partial_preparation_quarantines[0]
    assert quarantine.request_ids == tuple(str(child_id) for child_id in child_ids)
    assert (
        quarantine.decode_reqs[0]
        is fixture.packed_runtime.transactions[0].request_owner
    )
    assert quarantine.packed_transactions == (fixture.packed_runtime.transactions[0],)
    assert fixture.queue._preparing_request_ids == set(quarantine.request_ids)
    assert fixture.queue._seen_bootstrap_rooms == {
        request.bootstrap_room for request in requests
    }
    assert fixture.metadata_allocator.allocated == [0]


def test_failure_before_kv_allocation_releases_partial_request_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback handles the request-slot-only prefix of physical preparation."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    request = _request(child_id)

    def fail_after_request_slot(*args: object, **kwargs: object) -> np.ndarray:
        """Acquire only a request slot before injecting failure.

        :param args: Allocation positional inputs.
        :param kwargs: Allocation keyword inputs.
        :returns: Never returns.
        :raises RuntimeError: Always after request-slot mutation.
        """

        del kwargs
        req = args[0]
        req.req_pool_idx = 7
        raise RuntimeError("injected pre-KV failure")

    fixture.queue._pre_alloc = fail_after_request_slot
    with pytest.raises(RuntimeError, match="injected pre-KV failure"):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((child_id,), source_tp_size=2),
            requests=(request,),
        )

    assert request.req_pool_idx is None
    fixture.queue.req_to_token_pool.free.assert_called_once_with(request)
    assert fixture.metadata_allocator.allocated == []
    assert fixture.receivers[0].clear_count == 1


def test_promote_authorizes_and_attach_is_take_once_queue_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion is non-runnable; attachment enqueues and initializes exactly once."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    request = _request(child_id)
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(request,),
    )
    decode_req = next(iter(fixture.queue._prepared_cohorts.values())).decode_reqs[0]

    fixture.queue.promote_preallocated(cohort)

    assert fixture.queue.queue == []
    assert fixture.queue.pending_reqs == []
    assert decode_req.allocation_lease.state is DecodeAllocationLeaseState.PUBLISHED
    assert fixture.packed_runtime.transactions[0].publish_count == 1
    assert fixture.receivers[0].init_ranks == []

    fixture.queue.attach_preallocated(cohort)

    assert fixture.queue.queue == [decode_req]
    assert fixture.queue.pending_reqs == [decode_req]
    assert fixture.receivers[0].init_ranks == []
    assert fixture.packed_runtime.transactions[0].publish_count == 1
    with pytest.raises(DecodeAllocationLeaseError, match="attachment is invalid"):
        fixture.queue.attach_preallocated(cohort)


def test_partial_transaction_publication_quarantines_the_complete_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later child publication failure leaves no runnable queue visibility."""

    fixture = _queue_fixture(monkeypatch)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt(child_ids, source_tp_size=4),
        requests=tuple(_request(child_id) for child_id in child_ids),
    )
    fixture.packed_runtime.transactions[1].fail_publication_after_transition = True

    with pytest.raises(RuntimeError, match="injected packed publication failure"):
        fixture.queue.promote_preallocated(cohort)

    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.state.value == "quarantined"
    assert fixture.queue.queue == []
    assert fixture.queue.pending_reqs == []
    assert [
        transaction.publish_count for transaction in record.packed_transactions
    ] == [
        1,
        1,
    ]
    assert [
        transaction.quarantine_count for transaction in record.packed_transactions
    ] == [1, 1]
    assert all(
        decode_req.allocation_lease.state is DecodeAllocationLeaseState.QUARANTINED
        for decode_req in record.decode_reqs
    )


def test_quarantine_callback_failure_still_closes_cohort_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed native quarantine callback cannot leave a published cohort live."""

    fixture = _queue_fixture(monkeypatch)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt(child_ids, source_tp_size=4),
        requests=tuple(_request(child_id) for child_id in child_ids),
    )
    fixture.packed_runtime.transactions[0].quarantine = MagicMock(
        side_effect=RuntimeError("injected quarantine failure")
    )
    fixture.packed_runtime.transactions[1].fail_publication_after_transition = True

    with pytest.raises(RuntimeError, match="injected quarantine failure"):
        fixture.queue.promote_preallocated(cohort)

    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.state.value == "quarantined"
    assert record.quarantine_reason is not None
    assert fixture.queue.queue == []
    assert fixture.queue.pending_reqs == []
    assert fixture.queue.has_live_preallocated_cohorts()
    assert (
        fixture.queue.quarantine_preallocated(cohort, "retry", None)
        is DecodeReservationState.QUARANTINED
    )


def test_cross_queue_cohort_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque cohort can never be consumed by another queue authority."""

    first = _queue_fixture(monkeypatch)
    second = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = first.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )

    with pytest.raises(DecodeAllocationLeaseError, match="another decode queue"):
        second.queue.promote_preallocated(cohort)


def test_pre_promotion_cancel_releases_resources_and_retains_room_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation frees owners without permitting transport-room reuse."""

    fixture = _queue_fixture(monkeypatch)
    first_id = uuid.uuid4()
    first_allocations, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((first_id,), source_tp_size=2),
        requests=(_request(first_id),),
    )
    assert fixture.queue.has_live_preallocated_cohorts()

    terminal = fixture.queue.cancel_preallocated(cohort)

    assert terminal is DecodeReservationState.CANCELLED
    assert fixture.metadata_allocator.allocated == []
    assert fixture.receivers[0].clear_count == 1
    assert fixture.queue._prepared_cohorts == {}
    assert fixture.queue._seen_bootstrap_rooms == {first_allocations[0].bootstrap_room}
    assert not fixture.queue.has_live_preallocated_cohorts()
    assert fixture.packed_runtime.transactions[0].cancel_count == 1

    second_id = uuid.uuid4()
    second_allocations, _ = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((second_id,), source_tp_size=2),
        requests=(_request(second_id),),
    )
    assert second_allocations[0].bootstrap_room != first_allocations[0].bootstrap_room
    assert fixture.queue._seen_bootstrap_rooms == {
        first_allocations[0].bootstrap_room,
        second_allocations[0].bootstrap_room,
    }


def test_bootstrap_room_collision_refuses_active_and_terminal_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A colliding grant cannot alias active or historical staging traffic."""

    fixture = _queue_fixture(monkeypatch)
    first_id = uuid.uuid4()
    first_allocations, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((first_id,), source_tp_size=2),
        requests=(_request(first_id),),
    )
    claimed_room = first_allocations[0].bootstrap_room
    pre_alloc_call_count = len(fixture.pre_alloc_calls)

    def collide_bootstrap_room(
        grant_id: uuid.UUID,
        child_request_ids: tuple[uuid.UUID, ...],
    ) -> tuple[int, ...]:
        """Return the active room for a distinct syntactically valid grant.

        :param grant_id: Ignored distinct grant identity.
        :param child_request_ids: Ignored distinct child identities.
        :returns: One deliberately colliding room.
        """

        del grant_id, child_request_ids
        return (claimed_room,)

    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode.derive_decode_reservation_bootstrap_rooms",
        collide_bootstrap_room,
    )

    second_id = uuid.uuid4()
    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_bootstrap_room_collision",
    ):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((second_id,), source_tp_size=2),
            requests=(_request(second_id),),
        )

    assert len(fixture.pre_alloc_calls) == pre_alloc_call_count
    assert fixture.queue._seen_bootstrap_rooms == {claimed_room}

    fixture.queue.cancel_preallocated(cohort)
    replacement_id = uuid.uuid4()
    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_bootstrap_room_collision",
    ):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((replacement_id,), source_tp_size=2),
            requests=(_request(replacement_id),),
        )
    assert fixture.queue._seen_bootstrap_rooms == {claimed_room}


def test_cancellation_serializes_against_transaction_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion cannot race an unpublished transaction through cancellation."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    transaction = fixture.packed_runtime.transactions[0]
    transaction.cancel_entered = threading.Event()
    transaction.cancel_release = threading.Event()
    cancellation_results: list[DecodeReservationState] = []
    cancellation_errors: list[Exception] = []
    promotion_started = threading.Event()
    promotion_done = threading.Event()
    promotion_errors: list[DecodeAllocationLeaseError] = []

    def cancel() -> None:
        """Drive cancellation while its transaction is deliberately paused."""

        try:
            cancellation_results.append(fixture.queue.cancel_preallocated(cohort))
        except (DecodeAllocationLeaseError, RuntimeError, AssertionError) as error:
            cancellation_errors.append(error)

    def promote() -> None:
        """Attempt publication after cancellation owns the cohort lock."""

        promotion_started.set()
        try:
            fixture.queue.promote_preallocated(cohort)
        except DecodeAllocationLeaseError as error:
            promotion_errors.append(error)
        finally:
            promotion_done.set()

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    assert transaction.cancel_entered.wait(timeout=5)
    promote_thread = threading.Thread(target=promote)
    promote_thread.start()
    assert promotion_started.wait(timeout=5)
    assert not promotion_done.wait(timeout=0.05)

    transaction.cancel_release.set()
    cancel_thread.join(timeout=5)
    promote_thread.join(timeout=5)

    assert not cancel_thread.is_alive()
    assert not promote_thread.is_alive()
    assert cancellation_errors == []
    assert cancellation_results == [DecodeReservationState.CANCELLED]
    assert len(promotion_errors) == 1
    assert isinstance(promotion_errors[0], DecodeAllocationLeaseError)
    assert transaction.publish_count == 0


def test_capacity_refusal_has_no_allocator_or_streaming_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission refusal occurs before receivers, allocations, or output streaming."""

    fixture = _queue_fixture(monkeypatch, metadata_slots=0)
    fixture.queue.scheduler.output_streamer = MagicMock()
    child_id = uuid.uuid4()

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_metadata_capacity",
    ):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((child_id,), source_tp_size=2),
            requests=(_request(child_id),),
        )

    assert fixture.receivers == []
    assert fixture.pre_alloc_calls == []
    fixture.queue.scheduler.output_streamer.stream_output.assert_not_called()


def test_unavailable_packed_runtime_refuses_before_native_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configuration-only staging flag cannot authorize reservations."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue.kv_manager.supports_packed_decode_request_transactions = MagicMock(
        return_value=False
    )
    child_id = uuid.uuid4()

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="packed_runtime_unavailable",
    ):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((child_id,), source_tp_size=2),
            requests=(_request(child_id),),
        )

    assert fixture.receivers == []
    assert fixture.pre_alloc_calls == []
    assert fixture.packed_runtime.transactions == []


def test_capacity_refusal_accounts_for_each_child_page_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cohort admission reserves physical pages rather than raw token lengths."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue.token_to_kv_pool_allocator.page_size = 4
    fixture.queue._allocatable_token_budgets.return_value = 7
    child_ids = (uuid.uuid4(), uuid.uuid4())

    with pytest.raises(DecodeReservationAdmissionRefused, match="decode_kv_capacity"):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=tuple(
                _request(child_id, prompt_tokens=3) for child_id in child_ids
            ),
        )

    assert len(fixture.receivers) == 2
    assert all(receiver.clear_count == 1 for receiver in fixture.receivers)
    assert fixture.pre_alloc_calls == []


def test_reserved_pop_waits_for_handshake_and_never_reallocates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attached reservations publish pre-owned metadata only after WaitingForInput."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    fixture.queue.attach_preallocated(cohort)
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    decode_req = record.decode_reqs[0]
    pre_alloc_count = len(fixture.pre_alloc_calls)
    metadata_allocations = tuple(fixture.metadata_allocator.allocated)
    fixture.queue._resolve_pending_reqs = MagicMock()
    fixture.queue._update_handshake_waiters = MagicMock()

    preallocated, failed = fixture.queue.pop_preallocated()

    assert preallocated == []
    assert failed == []
    assert fixture.queue.queue == [decode_req]

    decode_req.waiting_for_input = True
    fixture.queue._build_decode_metadata_submission = MagicMock(
        return_value=_DecodeMetadataSubmission(
            decode_req=decode_req,
            page_indices=np.array([1, 2], dtype=np.int32),
            state_indices=[],
            decode_prefix_len=0,
        )
    )
    preallocated, failed = fixture.queue.pop_preallocated()

    assert preallocated == [decode_req]
    assert failed == []
    assert fixture.queue.queue == []
    assert len(fixture.pre_alloc_calls) == pre_alloc_count
    assert tuple(fixture.metadata_allocator.allocated) == metadata_allocations
    assert fixture.receivers[0].metadata_count == 0
    assert fixture.packed_runtime.metadata_publications == [
        (fixture.packed_runtime.transactions[0], fixture.receivers[0])
    ]
    assert record.metadata_published


def test_prepared_l1_repeat_reuses_locked_prefix_through_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict repeat reserves and publishes only its uncached transfer tail."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    request = _request(child_id)
    last_node = object()
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=0,
        l3_storage_hit_length=0,
        last_device_node=last_node,
        last_device_lock_params=DecLockRefParams(),
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = last_node
        req.last_node_lock_params = prefix_match.last_device_lock_params
        return prefix_match

    fixture.queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match)
    pre_alloc = MagicMock(side_effect=fixture.queue._pre_alloc)
    fixture.queue._pre_alloc = pre_alloc

    allocations, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(request,),
    )

    assert pre_alloc.call_args.kwargs["prefix_len"] == 4
    assert pre_alloc.call_args.kwargs["total_prefix_len"] == 4
    assert allocations[0].reserved_kv_tokens == 8
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.decode_reqs[0].prefix_match is prefix_match
    assert request.cache_protected_len == 4

    fixture.queue.promote_preallocated(cohort)
    fixture.queue.attach_preallocated(cohort)
    decode_req = record.decode_reqs[0]
    decode_req.waiting_for_input = True
    fixture.queue._resolve_pending_reqs = MagicMock()
    fixture.queue._update_handshake_waiters = MagicMock()
    fixture.queue._build_decode_metadata_submission = MagicMock(
        return_value=_DecodeMetadataSubmission(
            decode_req=decode_req,
            page_indices=np.array([1], dtype=np.int32),
            state_indices=[],
            decode_prefix_len=4,
        )
    )

    fixture.queue.pop_preallocated()

    assert fixture.queue._build_decode_metadata_submission.call_args.kwargs == {
        "origin_input_len": 8,
        "prefix_len": 4,
        "total_prefix_len": 4,
        "dst_kv_indices": None,
        "allocate_metadata_index": False,
    }


def test_prepared_l2_restore_retains_restore_gap_without_transfer_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host hit remains restore-owned while packed migration starts after it."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue.scheduler.enable_decode_hicache = True
    child_id = uuid.uuid4()
    request = _request(child_id)
    last_node = object()
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=2,
        l3_storage_hit_length=0,
        last_device_node=last_node,
        swa_host_hit_length=2,
        last_device_lock_params=DecLockRefParams(),
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = last_node
        req.last_node_lock_params = prefix_match.last_device_lock_params
        return prefix_match

    fixture.queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match)
    fixture.queue._start_hicache_prefetch = MagicMock()
    pre_alloc = MagicMock(side_effect=fixture.queue._pre_alloc)
    fixture.queue._pre_alloc = pre_alloc

    fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=4),
        requests=(request,),
    )

    assert pre_alloc.call_args.kwargs["prefix_len"] == 4
    assert pre_alloc.call_args.kwargs["total_prefix_len"] == 6
    fixture.queue._start_hicache_prefetch.assert_called_once_with(
        request,
        prefix_match,
    )
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.decode_reqs[0].prefix_match is prefix_match
    assert request.cache_protected_len == 6


def test_migration_lease_excludes_hicache_restore_gap() -> None:
    """Packed ownership pins only rows that the prefill writers may mutate."""

    queue = object.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[11, 12, 13, 14, 0, 0, 21, 22]])
    )
    queue.token_to_kv_pool_allocator = object()
    queue.scheduler = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="gemma4"))
    )
    req = SimpleNamespace(req_pool_idx=0)
    decode_req = SimpleNamespace(req=req)

    claims = queue._migration_component_claims(
        decode_req,
        migration_start=6,
        migration_end=8,
    )

    full_claim = claims[0]
    assert full_claim.component is DecodeAllocationComponent.FULL
    assert full_claim.logical_start == 6
    assert full_claim.logical_length == 2
    assert torch.equal(full_claim.indices, torch.tensor([21, 22]))
    assert 0 not in full_claim.indices.tolist()


def test_prepared_prefetch_rollback_releases_each_owner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation aborts storage work and releases one retained allocation."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue.scheduler.enable_decode_hicache = True
    child_id = uuid.uuid4()
    request = _request(child_id)
    lock_params = DecLockRefParams()
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=0,
        l3_storage_hit_length=2,
        last_device_node=object(),
        last_host_node=object(),
        last_device_lock_params=lock_params,
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = prefix_match.last_device_node
        req.last_node_lock_params = prefix_match.last_device_lock_params
        return prefix_match

    def start_prefetch(req: Req, match_result: DecodePrefixMatch) -> None:
        assert req is request
        assert match_result is prefix_match
        match_result.prefetch_registered = True

    fixture.queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match)
    fixture.queue._start_hicache_prefetch = MagicMock(side_effect=start_prefetch)
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(request,),
    )

    fixture.queue.cancel_preallocated(cohort)

    fixture.queue.tree_cache.release_aborted_request.assert_called_once_with(
        request.rid
    )
    assert fixture.released_request_ids == [request.rid]
    assert fixture.packed_runtime.transactions[0].cancel_count == 1
    assert fixture.metadata_allocator.allocated == []
    assert fixture.receivers[0].clear_count == 1


def test_l3_prefetch_fallback_is_revalidated_as_a_larger_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L3 fallback cannot publish a suffix larger than exact admission allowed."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue.scheduler.enable_decode_hicache = True
    fixture.queue._allocatable_token_budgets.return_value = 3
    child_id = uuid.uuid4()
    request = _request(child_id)
    lock_params = DecLockRefParams()
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=0,
        l3_storage_hit_length=2,
        last_device_node=object(),
        last_host_node=object(),
        last_device_lock_params=lock_params,
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = prefix_match.last_device_node
        req.last_node_lock_params = prefix_match.last_device_lock_params
        return prefix_match

    def fail_prefetch(req: Req, match_result: DecodePrefixMatch) -> None:
        assert req is request
        match_result.l3_storage_hit_length = 0

    fixture.queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match)
    fixture.queue._start_hicache_prefetch = MagicMock(side_effect=fail_prefetch)
    fixture.queue._pre_alloc = MagicMock()

    with pytest.raises(DecodeReservationAdmissionRefused, match="decode_kv_capacity"):
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((child_id,), source_tp_size=2),
            requests=(request,),
        )

    fixture.queue._pre_alloc.assert_not_called()
    fixture.queue.tree_cache.dec_lock_ref.assert_called_once_with(
        prefix_match.last_device_node,
        lock_params,
    )
    assert fixture.receivers[0].clear_count == 1


def test_completion_with_live_transport_ownership_quarantines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion cannot outrun packed request teardown and lease clearing."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=4),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    fixture.queue.attach_preallocated(cohort)

    terminal = fixture.queue.complete_preallocated(cohort)

    assert terminal is DecodeReservationState.QUARANTINED
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.decode_reqs[0].allocation_lease.state is (
        DecodeAllocationLeaseState.QUARANTINED
    )
    assert fixture.queue._seen_bootstrap_rooms == {
        record.decode_reqs[0].req.bootstrap_room
    }
    assert fixture.metadata_allocator.allocated != []
    assert fixture.queue.has_live_preallocated_cohorts()


def test_completion_retires_cohort_after_transport_clears_every_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion succeeds only after packed teardown returns request ownership."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    fixture.queue.attach_preallocated(cohort)
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    decode_req = record.decode_reqs[0]
    lease = decode_req.allocation_lease
    lease.state = DecodeAllocationLeaseState.COMMITTED_TO_REQUEST
    fixture.authority.retire_terminal(lease)
    decode_req.allocation_lease = None

    terminal = fixture.queue.complete_preallocated(cohort)

    assert terminal is DecodeReservationState.COMPLETED
    assert fixture.queue._prepared_cohorts == {}
    assert fixture.queue._prepared_request_ids == {}
    assert fixture.queue._seen_bootstrap_rooms == {decode_req.req.bootstrap_room}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
