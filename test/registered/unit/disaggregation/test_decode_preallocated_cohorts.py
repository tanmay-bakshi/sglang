import dataclasses
import hashlib
import sys
import threading
import uuid
from array import array
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import (
    TerminalPrefillAuthorityMismatch,
    TerminalPrefillRequestAuthority,
)
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
    DecodeTransferQueue,
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
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
)
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    NixlTerminalPrefillRequestAuthority,
    _NixlPrefillPeer,
    _TerminalStartupPeerEnrollment,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedRequestPublication,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.request_registration import (
    PackedTerminalRequestRegistrationError,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TERMINAL_PREFILL_AUTHORITY_COVERAGE = {
    "TPA-COLD-001": "resolve frozen source authority before receiver construction",
    "TPA-ORDER-002": "refuse before request identity or resource ownership",
    "TPA-ATTACH-003": "attach from retained authority without discovery caches",
    "TPA-TP-004": "reject source TP drift as terminal",
    "TPA-GENERATION-005": "reject source generation drift as terminal",
    "TPA-MULTI-006": "retain deterministic authority for every request",
}


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
    state: PackedRequestTransactionState
    terminal_binding_digest_value: bytes | None

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
        self.state = PackedRequestTransactionState.PREPARED
        self.terminal_binding_digest_value = None

    def publish(self) -> PackedRequestPublication:
        """Cross the sole fake publication boundary.

        :returns: Exact immutable publication.
        """

        self.publish_count += 1
        self.authority.record_publication(self.lease, self.authority.lifecycle)
        if self.fail_publication_after_transition:
            raise RuntimeError("injected packed publication failure")
        self.state = PackedRequestTransactionState.PUBLISHED
        return self.publication

    @property
    def terminal_binding_digest(self) -> bytes | None:
        """Return no owner-driven authority for the legacy fixture.

        :returns: Always ``None``.
        """

        return self.terminal_binding_digest_value

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
        self.state = PackedRequestTransactionState.CANCELLED
        return self.request_owner

    def quarantine(self, reason: str) -> None:
        """Retain the complete fake transaction after ambiguity.

        :param reason: Stable quarantine reason.
        """

        self.quarantine_count += 1
        self.authority.quarantine(self.lease, reason)
        self.state = PackedRequestTransactionState.QUARANTINED


class _FakePackedRuntimeManager:
    """Explicit packed runtime seam consumed by the decode queue."""

    authority: _FakeAllocationAuthority
    metadata_allocator: "_FakeMetadataAllocator"
    metadata_publications: list[tuple[_FakePackedTransaction, object]]
    terminal_startup_binding: object | None
    transactions: list[_FakePackedTransaction]

    def __init__(
        self,
        authority: _FakeAllocationAuthority,
        metadata_allocator: "_FakeMetadataAllocator",
    ) -> None:
        """Initialize an available fake runtime.

        :param authority: Exact fake allocation authority.
        :param metadata_allocator: Exact adopted auxiliary row allocator.
        """

        self.authority = authority
        self.attn_tp_rank = 0
        self.attn_tp_size = 1
        self.metadata_allocator = metadata_allocator
        self.metadata_publications = []
        self.terminal_startup_binding = None
        self.transactions = []

    def supports_packed_decode_request_transactions(self) -> bool:
        """Return whether request-scoped packed ownership is initialized.

        :returns: Always ``True`` for this fake runtime.
        """

        return True

    def resolve_terminal_prefill_request_authority(
        self,
        *,
        bootstrap_addr: str,
        prefill_process_url: str,
        prefill_process_instance_id: uuid.UUID,
        prefill_dp_rank: int | None,
        source_tp_size: int,
    ) -> TerminalPrefillRequestAuthority:
        """Return a stable fake authority for terminal lifecycle tests.

        :param bootstrap_addr: Candidate source bootstrap address.
        :param prefill_process_url: Candidate source service URL.
        :param prefill_process_instance_id: Candidate source launch instance.
        :param prefill_dp_rank: Candidate explicit source DP rank.
        :param source_tp_size: Reservation-authenticated source TP width.
        :returns: Opaque fake source authority.
        """

        assert bootstrap_addr == "prefill.internal:8998"
        assert prefill_process_url == "http://prefill.internal:30000"
        assert prefill_process_instance_id == uuid.UUID(int=1)
        assert prefill_dp_rank is None
        assert source_tp_size in (1, 2, 4)
        return TerminalPrefillRequestAuthority()

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
        owner = transaction.cancel_unpublished()
        self.metadata_allocator.free(transaction.metadata_buffer_index)
        return owner

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
    terminal_authorities: list[TerminalPrefillRequestAuthority]
    conclude_state: int | None
    prefill_dp_rank: int

    def __init__(self, source_tp_size: int) -> None:
        """Initialize one receiver.

        :param source_tp_size: Handshake-reported source width.
        """

        self.prefill_info = SimpleNamespace(attn_tp_size=source_tp_size)
        self.init_ranks = []
        self.clear_count = 0
        self.abort_count = 0
        self.metadata_count = 0
        self.terminal_authorities = []
        self.conclude_state = None
        self.prefill_dp_rank = 0

    def init(self, rank: int) -> None:
        """Record asynchronous receiver initialization.

        :param rank: Selected prefill DP rank.
        """

        self.init_ranks.append(rank)

    def init_from_terminal_authority(
        self,
        authority: TerminalPrefillRequestAuthority,
    ) -> None:
        """Record one immutable terminal source authority.

        :param authority: PREPARE-retained source authority.
        """

        self.terminal_authorities.append(authority)

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


class _ForbiddenLookupDict(dict[str, object]):
    """Empty mapping that fails if a legacy discovery consumer reads it."""

    def __contains__(self, key: object) -> bool:
        """Reject membership lookup.

        :param key: Candidate legacy cache key.
        :returns: Never returns.
        :raises AssertionError: Always.
        """

        raise AssertionError(f"legacy cache membership lookup: {key!r}")

    def __getitem__(self, key: str) -> object:
        """Reject indexed lookup.

        :param key: Candidate legacy cache key.
        :returns: Never returns.
        :raises AssertionError: Always.
        """

        raise AssertionError(f"legacy cache indexed lookup: {key!r}")

    def get(self, key: str, default: object = None) -> object:
        """Reject optional lookup.

        :param key: Candidate legacy cache key.
        :param default: Unused fallback value.
        :returns: Never returns.
        :raises AssertionError: Always.
        """

        del default
        raise AssertionError(f"legacy cache optional lookup: {key!r}")


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
    req.last_host_node = None
    req.best_match_node = None
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.mamba_host_hit_length = 0
    req.num_matched_prefix_tokens = 0
    req.cache_protected_len = 0
    req.finished_reason = None
    req.return_logprob = False
    req.disagg_prefill_dp_rank = None
    return req


def _attempt(
    child_ids: tuple[uuid.UUID, ...],
    *,
    source_tp_size: int,
    prefill_instance_id: uuid.UUID = uuid.UUID(int=1),
) -> DecodeReservationAttempt:
    """Build one exact reservation attempt for a child cohort.

    :param child_ids: Ordered request identities.
    :param source_tp_size: Supported packed source width.
    :param prefill_instance_id: Reservation-authenticated source launch instance.
    :returns: Reservation attempt consumed by the queue.
    """

    prefill_process = DecodeReservationProcess(
        url="http://prefill.internal:30000",
        instance_id=prefill_instance_id,
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
    queue._terminal_decode_serving = None
    queue._terminal_dflash_boundary_pool = None

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

        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.last_node = root_node
        req.last_host_node = root_node
        req.best_match_node = root_node
        return DecodePrefixMatch(
            prefix_indices=req.prefix_indices,
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=root_node,
        )

    queue._match_preallocated_prefix_and_lock = MagicMock(side_effect=match_prefix)

    lifecycle = object()
    authority = _FakeAllocationAuthority(lifecycle)
    queue.allocation_lifecycle_authority = lifecycle
    queue.allocation_lease_authority = authority
    packed_runtime = _FakePackedRuntimeManager(authority, metadata_allocator)
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


def _terminal_rank(
    *,
    role: TerminalOwnerRole,
    tp_rank: int,
    tp_size: int,
    generation: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact terminal startup rank for authority tests.

    :param role: Source or decode role.
    :param tp_rank: Rank within the service.
    :param tp_size: Exact service TP width.
    :param generation: Non-nil process-generation integer.
    :returns: Immutable startup rank.
    """

    service_id = "prefill-a" if role is TerminalOwnerRole.SOURCE else "decode-a"
    metadata = f"{service_id}-metadata-{tp_rank}".encode("ascii")
    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=b"c" * 32,
        service_id=service_id,
        service_origin=(
            "http://prefill.internal:30000"
            if role is TerminalOwnerRole.SOURCE
            else "http://decode.internal:30001"
        ),
        role=role,
        launch_instance_id=uuid.UUID(
            int=1 if role is TerminalOwnerRole.SOURCE else 2
        ).bytes,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=f"{service_id}-agent-{tp_rank}",
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _install_terminal_prefill_authority(
    fixture: _QueueFixture,
) -> tuple[TerminalStartupRankBinding, _TerminalStartupPeerEnrollment]:
    """Install the real NIXL authority resolver over a cold fake runtime.

    :param fixture: Queue fixture receiving terminal startup state.
    :returns: Exact local binding and mutable test enrollment.
    """

    source_ranks = (
        _terminal_rank(
            role=TerminalOwnerRole.SOURCE,
            tp_rank=0,
            tp_size=2,
            generation=101,
        ),
        _terminal_rank(
            role=TerminalOwnerRole.SOURCE,
            tp_rank=1,
            tp_size=2,
            generation=102,
        ),
    )
    decode_rank = _terminal_rank(
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
        generation=201,
    )
    matrix = TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=b"c" * 32,
        ranks=(*source_ranks, decode_rank),
    )
    binding = TerminalStartupRankBinding(
        advertisement=decode_rank,
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id=decode_rank.service_id,
            local_tensor_parallel_rank=decode_rank.tensor_parallel_rank,
            first_producer_id=0,
        ),
    )
    enrollment = _TerminalStartupPeerEnrollment(
        binding=binding,
        expected_remote_ranks=source_ranks,
    )
    for rank in source_ranks:
        peer = _NixlPrefillPeer(
            bootstrap_addr="prefill.internal:8998",
            attn_dp_rank=0,
            attn_cp_rank=0,
            attn_tp_rank=rank.tensor_parallel_rank,
            pp_rank=0,
            transfer_source_rank=rank.tensor_parallel_rank,
            agent_name=rank.nixl_agent_name,
            metadata_sha256=rank.nixl_agent_metadata_sha256.hex(),
            process_generation=str(uuid.UUID(bytes=rank.process_generation)),
            control_endpoint=NetworkAddress(
                "127.0.0.1",
                31000 + rank.tensor_parallel_rank,
            ),
            handle=object(),
        )
        enrollment.prefill_peers[rank.key] = peer
    enrollment.frozen = True
    enrollment.frozen_event.set()

    manager = fixture.packed_runtime
    manager.terminal_startup_binding = binding
    manager._terminal_startup_peer_enrollment = enrollment
    manager._quarantined_remote_handles = set()
    manager.kv_args = SimpleNamespace(engine_rank=0, page_size=1)
    manager.enable_staging = True
    manager.prefill_info_table = {}
    manager.connection_pool = {}
    manager.resolve_terminal_prefill_request_authority = (
        NixlKVManager.resolve_terminal_prefill_request_authority.__get__(manager)
    )
    fixture.queue._terminal_decode_serving = object()
    fixture.queue.scheduler.enable_decode_hicache = False
    fixture.queue.scheduler.server_args.disable_radix_cache = True
    fixture.queue.scheduler.server_args.disaggregation_decode_enable_radix_cache = False
    return binding, enrollment


def _install_terminal_transfer_publication(
    fixture: _QueueFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> DecodeTransferQueue:
    """Install the production terminal registry with deterministic wire authority.

    :param fixture: Queue fixture receiving terminal transfer ownership.
    :param monkeypatch: Pytest mutation fixture.
    :returns: Production transfer registry used by the queue.
    """

    transfer_queue = object.__new__(DecodeTransferQueue)
    transfer_queue.queue = []
    transfer_queue._terminal_requests = {}
    fixture.queue.transfer_queue = transfer_queue
    fixture.queue._rebootstrap_prefill_len = lambda req: len(req.origin_input_ids)

    def build_submission(
        decode_req: DecodeRequest,
        **kwargs: object,
    ) -> _DecodeMetadataSubmission:
        """Build deterministic destination metadata for one prepared request.

        :param decode_req: Exact prepared request.
        :param kwargs: Production shape arguments, unused by the fixture.
        :returns: One immutable metadata submission.
        """

        del kwargs
        return _DecodeMetadataSubmission(
            decode_req=decode_req,
            page_indices=np.arange(len(decode_req.req.origin_input_ids)),
            state_indices=None,
            decode_prefix_len=0,
        )

    def build_authority(**kwargs: object) -> object:
        """Bind the terminal source plan before packed publication.

        :param kwargs: Exact production authority inputs.
        :returns: Opaque registered owner authority.
        """

        transaction = kwargs["transaction"]
        assert type(transaction) is _FakePackedTransaction
        binding_digest = hashlib.sha256(
            transaction.publication.key.request_generation
        ).digest()
        transaction.terminal_binding_digest_value = binding_digest
        transaction.publication = dataclasses.replace(
            transaction.publication,
            terminal_source_plan=b"sealed terminal source plan",
        )
        return object()

    fixture.queue._build_decode_metadata_submission = build_submission
    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=build_authority
    )
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode.register_packed_terminal_decode_request",
        lambda serving, authority: None,
    )
    return transfer_queue


def _install_canonical_scheduler_abort(
    fixture: _QueueFixture,
    transfer_queue: DecodeTransferQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Bind the production scheduler abort path to the queue fixture.

    :param fixture: Exact queue and observable fake resources.
    :param transfer_queue: Production terminal registry owned by the fixture.
    :param monkeypatch: Pytest mutation fixture.
    :returns: Observable tokenizer-output sender.
    """

    scheduler = fixture.queue.scheduler
    send_output = MagicMock()
    scheduler.abort_request = Scheduler.abort_request.__get__(scheduler)
    scheduler.chunked_req = None
    scheduler.enable_hicache_storage = False
    scheduler.tree_cache = fixture.queue.tree_cache
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=send_output)
    )
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = fixture.queue
    scheduler.disagg_decode_transfer_queue = transfer_queue
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.last_batch = None
    transfer_queue.scheduler = scheduler

    def release_scheduler_request(
        req: Req,
        tree_cache: object,
        is_insert: bool = True,
    ) -> None:
        """Release one scheduler-owned fake allocation.

        :param req: Exact scheduler-owned request.
        :param tree_cache: Exact fake tree owner.
        :param is_insert: Whether canonical cleanup inserts cache state.
        """

        assert tree_cache is fixture.queue.tree_cache
        assert is_insert
        fixture.released_request_ids.append(req.rid)
        req.req_pool_idx = None
        req.kv = None

    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.release_kv_cache",
        release_scheduler_request,
    )
    return send_output


def _capture_terminal_request_callbacks(
    fixture: _QueueFixture,
) -> dict[str, dict[str, object]]:
    """Capture each production terminal authority callback set by request ID.

    :param fixture: Exact queue fixture whose authority builder is wrapped.
    :returns: Mutable callback map populated during promotion.
    """

    callbacks: dict[str, dict[str, object]] = {}
    base_builder = fixture.packed_runtime.build_terminal_decode_request_authority

    def capture_authority(**kwargs: object) -> object:
        """Retain callbacks while preserving production fixture construction.

        :param kwargs: Exact terminal authority construction inputs.
        :returns: Opaque terminal authority.
        """

        transaction = kwargs["transaction"]
        assert type(transaction) is _FakePackedTransaction
        callbacks[transaction.request_owner.req.rid] = dict(kwargs)
        return base_builder(**kwargs)

    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=capture_authority
    )
    return callbacks


def _finalize_terminal_request(
    callbacks: dict[str, dict[str, object]],
    decode_req: DecodeRequest,
    transaction: _FakePackedTransaction,
) -> None:
    """Drive one request through its real queue-owned finalization callback.

    :param callbacks: Callback sets captured during terminal promotion.
    :param decode_req: Exact adopted request becoming runnable.
    :param transaction: Exact committed packed transaction.
    """

    decode_req.req.pd_dflash_boundary_token_id = torch.tensor([17])
    decode_req.req.pd_dflash_boundary_completion_event = object()
    transaction.state = PackedRequestTransactionState.COMMITTED
    finalize_request = callbacks[decode_req.req.rid]["finalize_request"]
    assert callable(finalize_request)
    finalize_request(decode_req)


def _assert_no_preparation_ownership(
    fixture: _QueueFixture,
    requests: tuple[Req, ...],
) -> None:
    """Assert refusal preceded every mutable preparation category.

    :param fixture: Exact queue and resource fixture.
    :param requests: Candidate requests which must remain untouched.
    """

    assert fixture.receivers == []
    assert fixture.pre_alloc_calls == []
    assert fixture.metadata_allocator.allocated == []
    assert fixture.queue._preparing_grant_ids == set()
    assert fixture.queue._preparing_request_ids == set()
    assert fixture.queue._prepared_grant_ids == {}
    assert fixture.queue._prepared_request_ids == {}
    assert all(req.bootstrap_host is None for req in requests)
    assert all(req.bootstrap_port is None for req in requests)
    assert all(req.bootstrap_room is None for req in requests)
    assert all(req.req_pool_idx is None for req in requests)


def test_terminal_prepare_resolves_cold_authority_before_receiver_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPA-COLD-001 and TPA-MULTI-006 retain deterministic cold authority."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    child_ids = (uuid.uuid4(), uuid.uuid4())
    requests = tuple(_request(child_id) for child_id in child_ids)
    events: list[str] = []
    real_resolver = fixture.packed_runtime.resolve_terminal_prefill_request_authority
    real_create_receiver = fixture.queue._create_receiver
    real_pre_alloc = fixture.queue._pre_alloc

    def resolve_authority(**kwargs: object) -> TerminalPrefillRequestAuthority:
        """Record real authority resolution without replacing its behavior.

        :param kwargs: Exact resolver keyword arguments.
        :returns: Generation-bound resolver result.
        """

        events.append("authority")
        return real_resolver(**kwargs)

    def create_receiver(req: Req, *, is_rebootstrap: bool = False) -> DecodeRequest:
        """Record the first receiver-owned side effect.

        :param req: Request receiving the receiver.
        :param is_rebootstrap: Whether this is a rebootstrap request.
        :returns: Created decode request.
        """

        events.append("receiver")
        decode_req = real_create_receiver(req, is_rebootstrap=is_rebootstrap)
        real_init = decode_req.kv_receiver.init_from_terminal_authority

        def initialize(authority: TerminalPrefillRequestAuthority) -> None:
            """Record consumption by the exact newly created receiver.

            :param authority: PREPARE-retained source authority.
            """

            events.append("receiver_init")
            real_init(authority)

        decode_req.kv_receiver.init_from_terminal_authority = initialize
        return decode_req

    def pre_alloc(*args: object, **kwargs: object) -> np.ndarray:
        """Record the first KV ownership operation.

        :param args: Allocation positional inputs.
        :param kwargs: Allocation keyword inputs.
        :returns: Fake destination indices.
        """

        events.append("kv_allocation")
        return real_pre_alloc(*args, **kwargs)

    fixture.packed_runtime.resolve_terminal_prefill_request_authority = (
        resolve_authority
    )
    fixture.queue._create_receiver = create_receiver
    fixture.queue._pre_alloc = pre_alloc
    assert fixture.packed_runtime.prefill_info_table == {}
    assert fixture.packed_runtime.connection_pool == {}
    assert fixture.receivers == []

    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt(child_ids, source_tp_size=2),
        requests=requests,
    )

    record = fixture.queue._prepared_cohorts[cohort._token]
    receiver_seals = record.prepared_receiver_seals
    assert receiver_seals is not None
    authorities = tuple(seal.authority for seal in receiver_seals)
    assert len(authorities) == 2
    assert tuple(seal.receiver for seal in receiver_seals) == tuple(fixture.receivers)
    assert all(
        type(value) is NixlTerminalPrefillRequestAuthority for value in authorities
    )
    assert all(
        tuple(peer.attn_tp_rank for peer in value.peers) == (0, 1)
        for value in authorities
    )
    assert events == [
        "authority",
        "authority",
        "receiver",
        "receiver_init",
        "receiver",
        "receiver_init",
        "kv_allocation",
        "kv_allocation",
    ]
    assert all(
        req.time_stats.set_bootstrap_done_time.call_count == 1 for req in requests
    )
    assert fixture.packed_runtime.prefill_info_table == {}
    assert fixture.packed_runtime.connection_pool == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        authorities[0].prefill_dp_rank = 1


def test_terminal_prepare_to_attach_uses_real_receiver_without_legacy_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPA-COLD-001 and TPA-ATTACH-003 compose through the real receiver."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    manager = fixture.packed_runtime
    manager.addr_to_rooms_tracker = defaultdict(set)
    manager.required_prefill_response_num_table = {}
    manager.update_status = MagicMock()
    manager.prefill_info_table = _ForbiddenLookupDict()
    manager.connection_pool = _ForbiddenLookupDict()
    created_receivers: list[NixlKVReceiver] = []

    def create_receiver(req: Req, *, is_rebootstrap: bool = False) -> DecodeRequest:
        """Construct the production NIXL receiver without publishing it.

        :param req: Exact request receiving terminal authority.
        :param is_rebootstrap: Whether the request is a rebootstrap.
        :returns: Exact request and production receiver owner.
        """

        receiver = NixlKVReceiver(
            mgr=manager,
            bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )
        created_receivers.append(receiver)
        return DecodeRequest(
            req=req,
            kv_receiver=receiver,
            is_rebootstrap=is_rebootstrap,
        )

    fixture.queue._create_receiver = create_receiver
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    request = _request(child_id)

    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(request,),
    )
    record = fixture.queue._prepared_cohorts[cohort._token]
    receiver = created_receivers[0]
    receiver_seals = record.prepared_receiver_seals
    assert receiver_seals is not None
    seal = receiver_seals[0]
    transaction = fixture.packed_runtime.transactions[0]
    assert record.packed_transactions[0] is transaction

    fixture.queue.promote_preallocated(cohort)

    assert record.state.value == "attached"
    assert record.metadata_published
    assert record.decode_reqs[0].packed_transaction is transaction
    assert transfer_queue.live_requests() == record.decode_reqs
    assert fixture.packed_runtime.metadata_publications == [(transaction, receiver)]
    publication_count = len(fixture.packed_runtime.metadata_publications)
    fixture.queue.attach_preallocated(cohort)

    assert seal.decode_req is record.decode_reqs[0]
    assert seal.receiver is receiver
    assert tuple(peer.attn_tp_rank for peer in seal.authority.peers) == (0, 1)
    assert receiver.prefill_peers == list(seal.authority.peers)
    assert receiver.prefill_info.attn_tp_size == 2
    assert fixture.queue.queue == []
    assert fixture.queue.pending_reqs == []
    assert transfer_queue.live_requests() == record.decode_reqs
    assert fixture.packed_runtime.metadata_publications == [(transaction, receiver)]
    assert len(fixture.packed_runtime.metadata_publications) == publication_count
    assert manager.prefill_info_table == {}
    assert manager.connection_pool == {}


def test_terminal_prepare_transient_authority_refusal_has_no_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPA-ORDER-002 refuses incomplete enrollment before all side effects."""

    fixture = _queue_fixture(monkeypatch)
    _, enrollment = _install_terminal_prefill_authority(fixture)
    enrollment.frozen = False
    enrollment.frozen_event.clear()
    fixture.queue.scheduler.metrics_reporter.enable_metrics = True
    fixture.queue.scheduler.metrics_collector = MagicMock()
    child_ids = (uuid.uuid4(),)
    requests = tuple(_request(child_id) for child_id in child_ids)

    with pytest.raises(DecodeReservationAdmissionRefused) as raised:
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=requests,
        )

    assert raised.value.reason_code == "terminal_prefill_authority_unavailable"
    assert (
        raised.value.disposition
        is DecodeReservationRefusalDisposition.RETRY_SAME_DECODER
    )
    fixture.queue.scheduler.metrics_collector.increment_bootstrap_failed_reqs.assert_called_once_with()
    _assert_no_preparation_ownership(fixture, requests)


def test_terminal_prepare_rejects_hicache_without_restore_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal ownership fails closed before an unsupported HiCache restore."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    fixture.queue.scheduler.enable_decode_hicache = True
    child_id = uuid.uuid4()
    requests = (_request(child_id),)

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="terminal_decode_hicache_not_supported",
    ) as raised:
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt((child_id,), source_tp_size=2),
            requests=requests,
        )

    assert raised.value.disposition is DecodeReservationRefusalDisposition.TERMINAL
    _assert_no_preparation_ownership(fixture, requests)


def test_terminal_receiver_initialization_failure_rolls_back_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receiver authority failure clears its partial owner and metric once."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    fixture.queue.scheduler.metrics_reporter.enable_metrics = True
    fixture.queue.scheduler.metrics_collector = MagicMock()
    real_create_receiver = fixture.queue._create_receiver

    def create_failing_receiver(
        req: Req,
        *,
        is_rebootstrap: bool = False,
    ) -> DecodeRequest:
        """Create a receiver whose first authority consumption fails.

        :param req: Request receiving the receiver.
        :param is_rebootstrap: Whether this is a rebootstrap request.
        :returns: Decode request with an injected terminal failure.
        """

        decode_req = real_create_receiver(req, is_rebootstrap=is_rebootstrap)
        decode_req.kv_receiver.init_from_terminal_authority = MagicMock(
            side_effect=TerminalPrefillAuthorityMismatch("injected receiver drift")
        )
        return decode_req

    fixture.queue._create_receiver = create_failing_receiver
    child_ids = (uuid.uuid4(),)
    request = _request(child_ids[0])

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="terminal_prefill_authority_mismatch",
    ) as raised:
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=(request,),
        )

    assert raised.value.disposition is DecodeReservationRefusalDisposition.TERMINAL
    assert raised.value.diagnostic == "injected receiver drift"
    assert len(fixture.receivers) == 1
    assert fixture.receivers[0].clear_count == 1
    assert fixture.pre_alloc_calls == []
    assert fixture.metadata_allocator.allocated == []
    assert fixture.queue._preparing_grant_ids == set()
    assert fixture.queue._preparing_request_ids == set()
    bootstrap_failure = (
        fixture.queue.scheduler.metrics_collector.increment_bootstrap_failed_reqs
    )
    bootstrap_failure.assert_called_once_with()


@pytest.mark.parametrize(
    "mismatch",
    ("tp", "peer_generation", "attempt_generation"),
)
def test_terminal_prepare_authority_mismatch_is_terminal_without_ownership(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    """TPA-TP-004 and TPA-GENERATION-005 reject frozen-authority drift."""

    fixture = _queue_fixture(monkeypatch)
    _, enrollment = _install_terminal_prefill_authority(fixture)
    source_tp_size = 4 if mismatch == "tp" else 2
    if mismatch == "peer_generation":
        rank = enrollment.expected_remote_ranks[0]
        peer = enrollment.prefill_peers[rank.key]
        enrollment.prefill_peers[rank.key] = dataclasses.replace(
            peer,
            process_generation=str(uuid.UUID(int=999)),
        )
    child_ids = (uuid.uuid4(),)
    requests = tuple(_request(child_id) for child_id in child_ids)
    prefill_instance_id = (
        uuid.UUID(int=999) if mismatch == "attempt_generation" else uuid.UUID(int=1)
    )

    with pytest.raises(DecodeReservationAdmissionRefused) as raised:
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(
                child_ids,
                source_tp_size=source_tp_size,
                prefill_instance_id=prefill_instance_id,
            ),
            requests=requests,
        )

    assert raised.value.reason_code == "terminal_prefill_authority_mismatch"
    assert raised.value.disposition is DecodeReservationRefusalDisposition.TERMINAL
    _assert_no_preparation_ownership(fixture, requests)


def test_terminal_receiver_consumes_retained_authority_with_caches_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPA-ATTACH-003 initializes directly from PREPARE-retained writers."""

    fixture = _queue_fixture(monkeypatch)
    binding, _ = _install_terminal_prefill_authority(fixture)
    manager = fixture.packed_runtime
    authority = manager.resolve_terminal_prefill_request_authority(
        bootstrap_addr="prefill.internal:8998",
        prefill_process_url="http://prefill.internal:30000",
        prefill_process_instance_id=uuid.UUID(int=1),
        prefill_dp_rank=None,
        source_tp_size=2,
    )
    manager.update_status = MagicMock()
    manager.required_prefill_response_num_table = {}
    receiver = object.__new__(NixlKVReceiver)
    receiver.kv_mgr = manager
    receiver.bootstrap_addr = "prefill.internal:8998"
    receiver.bootstrap_room = 41
    receiver.prefill_peers = []
    receiver.conclude_state = None
    receiver._terminal_authority_initialized = False

    assert manager.terminal_startup_binding is binding
    assert manager.prefill_info_table == {}
    assert manager.connection_pool == {}
    receiver.init_from_terminal_authority(authority)

    assert tuple(peer.attn_tp_rank for peer in receiver.prefill_peers) == (0, 1)
    assert receiver.required_prefill_response_num == 2
    assert manager.required_prefill_response_num_table == {41: 2}
    assert manager.prefill_info_table == {}
    assert manager.connection_pool == {}
    manager.update_status.assert_called_once_with(41, KVPoll.WaitingForInput)
    with pytest.raises(
        TerminalPrefillAuthorityMismatch,
        match="already initialized",
    ):
        receiver.init_from_terminal_authority(authority)
    manager.update_status.assert_called_once_with(41, KVPoll.WaitingForInput)


def test_terminal_structural_swa_oversize_is_terminal_without_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single impossible SWA request preserves legacy BAD_REQUEST semantics."""

    fixture = _queue_fixture(monkeypatch)
    fixture.queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
    fixture.queue._prealloc_required_tokens = MagicMock(return_value=(8, 65))
    fixture.queue.token_to_kv_pool_allocator.size_swa = 64
    child_ids = (uuid.uuid4(),)
    requests = tuple(_request(child_id) for child_id in child_ids)

    with pytest.raises(DecodeReservationAdmissionRefused) as raised:
        fixture.queue.prepare_preallocated(
            grant_id=uuid.uuid4(),
            attempt=_attempt(child_ids, source_tp_size=2),
            requests=requests,
        )

    assert raised.value.reason_code == "request_exceeds_decode_swa_capacity"
    assert raised.value.disposition is DecodeReservationRefusalDisposition.TERMINAL
    _assert_no_preparation_ownership(fixture, requests)


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


def test_terminal_registration_precedes_allocation_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish immutable evidence before registry and metadata visibility."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    record = fixture.queue._prepared_cohorts[cohort._token]
    transaction = fixture.packed_runtime.transactions[0]
    serving = fixture.queue._terminal_decode_serving
    events: list[str] = []
    callbacks: dict[str, object] = {}
    authority = object()

    def build_authority(**kwargs: object) -> object:
        """Capture production callbacks without replacing their queue owners.

        :param kwargs: Exact authority-construction inputs.
        :returns: Opaque fake authority consumed by the patched registrar.
        """

        events.append("build")
        callbacks.update(kwargs)
        return authority

    def register(serving_arg: object, authority_arg: object) -> None:
        """Model source-plan attachment at the full-serving boundary.

        :param serving_arg: Candidate bound serving.
        :param authority_arg: Candidate request authority.
        """

        assert serving_arg is serving
        assert authority_arg is authority
        events.append("register")
        transaction.terminal_binding_digest_value = b"t" * 32
        transaction.publication = dataclasses.replace(
            transaction.publication,
            terminal_source_plan=b"sealed terminal source plan",
        )

    original_publish = transaction.publish

    def publish() -> PackedRequestPublication:
        """Require registration-owned source authority before publication.

        :returns: Exact fake publication.
        """

        assert transaction.publication.terminal_source_plan is not None
        events.append("publish")
        return original_publish()

    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=build_authority
    )
    transaction.publish = MagicMock(side_effect=publish)
    original_transfer_registration = transfer_queue.register_terminal_requests

    def register_transfer(decode_reqs: tuple[DecodeRequest, ...]) -> None:
        """Require immutable publication and request binding before visibility.

        :param decode_reqs: Exact promoted request cohort.
        """

        events.append("transfer_registry")
        assert record.packed_publications == (transaction.publication,)
        assert record.state.value == "promoted"
        assert not record.metadata_published
        assert decode_reqs[0].packed_transaction is transaction
        original_transfer_registration(decode_reqs)

    transfer_queue.register_terminal_requests = register_transfer
    original_metadata_send = fixture.packed_runtime.send_packed_decode_request_metadata

    def send_metadata(**kwargs: object) -> None:
        """Require attached registry ownership before metadata visibility.

        :param kwargs: Exact packed metadata submission.
        """

        events.append("metadata")
        assert record.state.value == "attached"
        assert not record.metadata_published
        original_metadata_send(**kwargs)

    fixture.packed_runtime.send_packed_decode_request_metadata = send_metadata
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode.register_packed_terminal_decode_request",
        register,
    )

    fixture.queue.promote_preallocated(cohort)

    assert events == [
        "build",
        "register",
        "publish",
        "transfer_registry",
        "metadata",
    ]
    assert callbacks["transaction"] is transaction
    assert "destination_bindings" not in callbacks
    assert transaction.publish_count == 1
    assert record.state.value == "attached"
    assert record.metadata_published
    assert transfer_queue.live_requests() == record.decode_reqs


def test_terminal_callback_rejects_another_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback cannot mutate queues for an aliased mutable request owner."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    callbacks: dict[str, object] = {}

    def build_authority(**kwargs: object) -> object:
        """Retain exact callbacks for an adversarial identity invocation.

        :param kwargs: Exact authority-construction inputs.
        :returns: Opaque authority.
        """

        callbacks.update(kwargs)
        transaction = kwargs["transaction"]
        assert type(transaction) is _FakePackedTransaction
        transaction.terminal_binding_digest_value = b"t" * 32
        transaction.publication = dataclasses.replace(
            transaction.publication,
            terminal_source_plan=b"sealed terminal source plan",
        )
        return object()

    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=build_authority
    )
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode.register_packed_terminal_decode_request",
        lambda serving_arg, authority_arg: None,
    )
    fixture.queue.promote_preallocated(cohort)
    record = next(iter(fixture.queue._prepared_cohorts.values()))
    queue_before = tuple(fixture.queue.queue)
    pending_before = tuple(fixture.queue.pending_reqs)

    adopt_request = callbacks["adopt_request"]
    assert callable(adopt_request)
    with pytest.raises(DecodeAllocationLeaseError, match="another request owner"):
        adopt_request(object())

    assert tuple(fixture.queue.queue) == queue_before
    assert tuple(fixture.queue.pending_reqs) == pending_before
    assert record.state.value == "attached"
    assert fixture.packed_runtime.transactions[0].publish_count == 1


def test_terminal_inference_attachment_accepts_early_transport_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate immutable evidence after terminal transfer ownership retires."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    record = fixture.queue._prepared_cohorts[cohort._token]
    decode_req = record.decode_reqs[0]
    transaction = record.packed_transactions[0]
    publication_count = len(fixture.packed_runtime.metadata_publications)
    digest = transaction.terminal_binding_digest
    assert digest is not None
    assert transfer_queue._terminal_requests.pop(digest) is decode_req
    transaction.state = PackedRequestTransactionState.COMMITTED
    decode_req.packed_transaction = None
    decode_req.allocation_lease = None
    decode_req.kv_receiver = None
    decode_req.metadata_buffer_index = -1

    fixture.queue.attach_preallocated(cohort)

    assert len(fixture.packed_runtime.metadata_publications) == publication_count
    assert transfer_queue.live_requests() == ()
    assert record.state.value == "attached"


def test_terminal_abort_after_promotion_removes_callback_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort a published cohort before inference attachment, fail closed."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    callbacks: dict[str, object] = {}
    base_builder = fixture.packed_runtime.build_terminal_decode_request_authority

    def capture_authority(**kwargs: object) -> object:
        """Retain owner callbacks while preserving deterministic binding.

        :param kwargs: Exact production authority inputs.
        :returns: Opaque owner authority.
        """

        callbacks.update(kwargs)
        return base_builder(**kwargs)

    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=capture_authority
    )
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    record = fixture.queue._prepared_cohorts[cohort._token]
    decode_req = record.decode_reqs[0]
    transaction = record.packed_transactions[0]
    assert transfer_queue.live_requests() == (decode_req,)

    terminal = fixture.queue.abort_preallocated(
        cohort,
        "gateway_dispatch_failed",
        None,
    )

    assert terminal is DecodeReservationState.QUARANTINED
    assert record.state.value == "quarantined"
    assert transfer_queue.live_requests() == ()
    assert decode_req.packed_transaction is transaction
    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert decode_req.allocation_lease is not None
    assert decode_req.allocation_lease.state is DecodeAllocationLeaseState.QUARANTINED
    adopt_request = callbacks["adopt_request"]
    assert callable(adopt_request)
    with pytest.raises(DecodeAllocationLeaseError, match="attached prepared cohort"):
        adopt_request(decode_req)
    assert fixture.queue.scheduler.waiting_queue == []


def test_terminal_reservation_abort_after_finalization_uses_scheduler_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort a runnable terminal request through canonical scheduler cleanup."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    callbacks = _capture_terminal_request_callbacks(fixture)
    send_output = _install_canonical_scheduler_abort(
        fixture,
        transfer_queue,
        monkeypatch,
    )
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    record = fixture.queue._prepared_cohorts[cohort._token]
    decode_req = record.decode_reqs[0]
    transaction = record.packed_transactions[0]

    _finalize_terminal_request(callbacks, decode_req, transaction)

    assert fixture.queue.scheduler.waiting_queue == [decode_req.req]
    assert decode_req.allocation_lease is None
    terminal = fixture.queue.abort_preallocated(
        cohort,
        "gateway_dispatch_failed",
        None,
    )

    assert terminal is DecodeReservationState.ABORTED
    assert record.state.value == "aborted"
    assert fixture.queue.scheduler.waiting_queue == []
    assert fixture.released_request_ids == [decode_req.req.rid]
    assert transfer_queue.live_requests() == ()
    assert fixture.queue._prepared_cohorts == {}
    send_output.assert_called_once()
    abort_request, output_req = send_output.call_args.args
    assert type(abort_request) is AbortReq
    assert abort_request.rid == decode_req.req.rid
    assert abort_request.abort_message == (
        "Decode reservation aborted: gateway_dispatch_failed"
    )
    assert output_req is decode_req.req


def test_terminal_mixed_reservation_abort_cleans_finalized_child_before_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean a finalized child while retaining its live sibling fail closed."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    callbacks = _capture_terminal_request_callbacks(fixture)
    send_output = _install_canonical_scheduler_abort(
        fixture,
        transfer_queue,
        monkeypatch,
    )
    child_ids = (uuid.uuid4(), uuid.uuid4())
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt(child_ids, source_tp_size=2),
        requests=tuple(_request(child_id) for child_id in child_ids),
    )
    fixture.queue.promote_preallocated(cohort)
    record = fixture.queue._prepared_cohorts[cohort._token]
    finalized_req, live_req = record.decode_reqs
    finalized_transaction, live_transaction = record.packed_transactions

    _finalize_terminal_request(
        callbacks,
        finalized_req,
        finalized_transaction,
    )

    assert fixture.queue.scheduler.waiting_queue == [finalized_req.req]
    assert transfer_queue.live_requests() == (live_req,)
    terminal = fixture.queue.abort_preallocated(
        cohort,
        "gateway_dispatch_failed",
        None,
    )

    assert terminal is DecodeReservationState.QUARANTINED
    assert record.state.value == "quarantined"
    assert fixture.queue.scheduler.waiting_queue == []
    assert fixture.released_request_ids == [finalized_req.req.rid]
    assert transfer_queue.live_requests() == ()
    assert finalized_transaction.state is PackedRequestTransactionState.COMMITTED
    assert live_transaction.state is PackedRequestTransactionState.QUARANTINED
    assert live_req.allocation_lease is not None
    assert live_req.allocation_lease.state is DecodeAllocationLeaseState.QUARANTINED
    assert fixture.queue._prepared_cohorts[cohort._token] is record
    send_output.assert_called_once()


def test_terminal_explicit_quarantine_removes_callback_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit quarantine retains resources outside the callback registry."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.queue.promote_preallocated(cohort)
    record = fixture.queue._prepared_cohorts[cohort._token]

    first = fixture.queue.quarantine_preallocated(
        cohort,
        "operator_quarantine",
        None,
    )
    second = fixture.queue.quarantine_preallocated(
        cohort,
        "operator_quarantine",
        None,
    )

    assert first is DecodeReservationState.QUARANTINED
    assert second is DecodeReservationState.QUARANTINED
    assert transfer_queue.live_requests() == ()
    assert record.state.value == "quarantined"
    assert record.quarantine_reason == "operator_quarantine"


def test_terminal_promotion_without_full_serving_fails_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal startup cannot fall back to an unowned packed publication."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.packed_runtime.terminal_startup_binding = object()

    with pytest.raises(RuntimeError, match="full serving owner"):
        fixture.queue.promote_preallocated(cohort)

    record = next(iter(fixture.queue._prepared_cohorts.values()))
    assert record.state.value == "quarantined"
    assert fixture.packed_runtime.transactions[0].publish_count == 0


def test_terminal_metadata_failure_after_publication_quarantines_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promotion-time metadata failure cannot remain merely promoted."""

    fixture = _queue_fixture(monkeypatch)
    _install_terminal_prefill_authority(fixture)
    transfer_queue = _install_terminal_transfer_publication(fixture, monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    fixture.packed_runtime.send_packed_decode_request_metadata = MagicMock(
        side_effect=RuntimeError("injected metadata publication failure")
    )

    with pytest.raises(RuntimeError, match="metadata publication failure"):
        fixture.queue.promote_preallocated(cohort)

    record = fixture.queue._prepared_cohorts[cohort._token]
    transaction = record.packed_transactions[0]
    assert transaction.publish_count == 1
    assert record.state.value == "quarantined"
    assert not record.metadata_published
    assert transfer_queue.live_requests() == ()
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_terminal_tp2_without_cross_rank_bindings_fails_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion delegates TP identity and never fabricates peer bindings."""

    fixture = _queue_fixture(monkeypatch)
    child_id = uuid.uuid4()
    _, cohort = fixture.queue.prepare_preallocated(
        grant_id=uuid.uuid4(),
        attempt=_attempt((child_id,), source_tp_size=2),
        requests=(_request(child_id),),
    )
    serving = object.__new__(PackedTerminalDecodeServing)
    fixture.queue.bind_terminal_decode_serving(serving)
    fixture.packed_runtime.terminal_startup_binding = object()
    fixture.packed_runtime.attn_tp_size = 2

    def reject_missing_bindings(**kwargs: object) -> object:
        """Model the production builder's TP-greater-than-one invariant.

        :param kwargs: Exact promotion inputs.
        :returns: Never returns.
        :raises PackedTerminalRequestRegistrationError: Always without peers.
        """

        assert "destination_bindings" not in kwargs
        raise PackedTerminalRequestRegistrationError(
            "decode TP greater than one requires an exact cross-rank request "
            "binding manifest"
        )

    fixture.packed_runtime.build_terminal_decode_request_authority = MagicMock(
        side_effect=reject_missing_bindings
    )

    with pytest.raises(
        PackedTerminalRequestRegistrationError,
        match="cross-rank request binding manifest",
    ):
        fixture.queue.promote_preallocated(cohort)

    assert fixture.packed_runtime.transactions[0].publish_count == 0


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
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = last_node
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
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = last_node
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
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=0,
        l3_storage_hit_length=2,
        last_device_node=object(),
        last_host_node=object(),
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = prefix_match.last_device_node
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
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.arange(4, dtype=torch.int64),
        l2_host_hit_length=0,
        l3_storage_hit_length=2,
        last_device_node=object(),
        last_host_node=object(),
    )

    def match(req: Req) -> DecodePrefixMatch:
        req.prefix_indices = prefix_match.prefix_indices
        req.last_node = prefix_match.last_device_node
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
        prefix_match.last_device_node
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
