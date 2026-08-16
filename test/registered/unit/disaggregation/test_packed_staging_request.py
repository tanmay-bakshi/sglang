import dataclasses
import sys
import threading
import traceback

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationComponentClaim,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLeaseError,
    DecodeAllocationLeaseSnapshot,
    DecodeAllocationLeaseState,
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliaryAllocationError,
    PackedAuxiliaryAllocationLease,
    PackedAuxiliaryAllocationLeaseAuthority,
    PackedAuxiliaryAllocationLeaseSnapshot,
    PackedAuxiliaryAllocationState,
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_DIGEST_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedDFlashBoundaryMetadata,
    PackedDFlashBoundaryOutcome,
    PackedLayoutSpec,
    PackedLease,
    PackedPrepare,
    PackedProtocolError,
    PackedReady,
    PackedRequestTeardown,
    PackedRequestTeardownAck,
    PackedScatterPreparation,
    PackedTopology,
    PackedTransportPath,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityAction,
    PackedWriterVisibilityEvidence,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    decode_packed_message,
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBuffer,
    StagingComponentBufferRegistry,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedCopyExecutor,
    PackedDestinationOutcomeCoordinator,
    PackedDestinationVisibilityActionExecutor,
    PackedDestinationVisibilityPolicy,
    PackedGpuDirectFlushOptions,
    PackedGpuDirectFlushScope,
    PackedGpuDirectFlushTarget,
    PackedGpuDirectWritesOrdering,
    PackedPreparedScatterCopy,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedDFlashBoundaryDecodeAdoption,
    PackedRequestChunkPlan,
    PackedRequestPublication,
    PackedRequestTransactionError,
    PackedRequestTransactionState,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWA_COMPONENT = StagingComponentId(state_index=0, state_type=StateType.SWA)


@dataclasses.dataclass
class _RequestOwner:
    """Minimal request carrying a request-pool slot."""

    req_pool_idx: int | None = None
    inflight_middle_chunks: int = 0
    kv_committed_len: int = 0


class _StagingAllocator:
    """CPU-only packed staging allocator."""

    _next_lease_id: int
    releases: list[int]
    quarantines: list[int]

    def __init__(self) -> None:
        """Initialize an empty fake staging arena."""

        self._next_lease_id = 1
        self.releases = []
        self.quarantines = []

    def allocate(self, length_bytes: int) -> PackedLease:
        """Allocate one fake registered staging region.

        :param length_bytes: Exact canonical packed byte count.
        :returns: Fake non-overlapping staging lease.
        """

        lease_id = self._next_lease_id
        self._next_lease_id += 1
        return PackedLease(
            lease_id=lease_id,
            base_address=0x800000 + lease_id * 0x10000,
            length_bytes=length_bytes,
        )

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Record one quarantined staging lease.

        :param lease: Ambiguous staging allocation.
        :param reason: Stable failure reason.
        """

        del reason
        self.quarantines.append(lease.lease_id)

    def release(self, lease: PackedLease) -> None:
        """Record one terminal staging release.

        :param lease: Terminal staging allocation.
        """

        self.releases.append(lease.lease_id)


class _AuxiliaryAllocator:
    """CPU-only generation-bearing auxiliary row allocator."""

    _owner: object | None
    _reservation: object | None
    quarantines: list[object]
    releases: list[object]

    def __init__(
        self,
        segments: tuple[PackedAuxiliaryDestinationSegment, ...] | None = None,
    ) -> None:
        """Initialize one allocator with no live reservation.

        :param segments: Optional exact registered destination geometry.
        """

        self._owner = None
        self._reservation = None
        self.quarantines = []
        self.releases = []
        self._segments = (
            (
                PackedAuxiliaryDestinationSegment(
                    address=0xA00000,
                    item_length=128,
                ),
                PackedAuxiliaryDestinationSegment(
                    address=0xA01000,
                    item_length=64,
                ),
            )
            if segments is None
            else segments
        )

    def allocate_packed_auxiliary_slot(self, owner: object) -> object:
        """Allocate one fake metadata row.

        :param owner: Exact process-local reservation owner.
        :returns: Opaque fake reservation.
        """

        assert self._reservation is None
        self._owner = owner
        self._reservation = object()
        return self._reservation

    def packed_auxiliary_slot_reservation_snapshot(
        self,
        reservation: object,
    ) -> PackedAuxiliarySlotReservationSnapshot:
        """Return exact deterministic row identity and geometry.

        :param reservation: Exact fake reservation.
        :returns: Immutable fake row snapshot.
        """

        assert reservation is self._reservation
        return PackedAuxiliarySlotReservationSnapshot(
            metadata_buffer_index=17,
            metadata_slot_generation=bytes(range(PACKED_REQUEST_GENERATION_BYTES)),
            destination_segments=self._segments,
        )

    def release_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Record one exact row release.

        :param reservation: Exact fake reservation.
        :param owner: Exact process-local reservation owner.
        """

        assert reservation is self._reservation
        assert owner is self._owner
        self.releases.append(reservation)

    def quarantine_packed_auxiliary_slot(
        self,
        reservation: object,
        owner: object,
    ) -> None:
        """Record one exact row quarantine.

        :param reservation: Exact fake reservation.
        :param owner: Exact process-local reservation owner.
        """

        assert reservation is self._reservation
        assert owner is self._owner
        self.quarantines.append(reservation)


class _VisibilityExecutor(PackedDestinationVisibilityActionExecutor):
    """CPU-only direct CUDA event visibility executor."""

    def establish_cuda_stream_dependency(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
    ) -> None:
        """Validate one direct-event dependency without touching CUDA.

        :param writer_id: Exact authenticated writer.
        :param policy: Decode-selected visibility policy.
        :param evidence: Source terminal visibility evidence.
        """

        del writer_id
        policy.validate_evidence(evidence)

    def flush_gpudirect_writes(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
        target: PackedGpuDirectFlushTarget,
        scope: PackedGpuDirectFlushScope,
    ) -> None:
        """Reject the unused NIC visibility path.

        :param writer_id: Exact authenticated writer.
        :param policy: Decode-selected visibility policy.
        :param evidence: Source terminal visibility evidence.
        :param target: CUDA flush target.
        :param scope: CUDA flush scope.
        """

        del writer_id, policy, evidence, target, scope
        raise AssertionError("direct-event tests must not flush GPUDirect writes")


class _CopyExecutor(PackedCopyExecutor):
    """CPU-only prepared-scatter executor preserving production ownership."""

    fail_on_preparation: int | None
    preparations: list[PackedScatterPreparation]
    prepared_copies: list[PackedPreparedScatterCopy]
    quarantined_copies: list[PackedPreparedScatterCopy]

    def __init__(self) -> None:
        """Initialize deterministic request-local metadata ownership."""

        self._owner_token = object()
        self.fail_on_preparation = None
        self.preparations = []
        self.prepared_copies = []
        self.quarantined_copies = []

    def prepare_scatter_copy(
        self,
        preparation: PackedScatterPreparation,
    ) -> PackedPreparedScatterCopy:
        """Project protocol geometry without allocating CUDA resources.

        :param preparation: Exact protocol-owned destination geometry.
        :returns: Identity-bound prepared scatter metadata.
        """

        self.preparations.append(preparation)
        if self.fail_on_preparation == len(self.preparations):
            raise RuntimeError("injected prepared scatter construction failure")
        prepared_copy = PackedPreparedScatterCopy(
            key=preparation.key,
            layout=preparation.layout,
            destination_binding=preparation.destination_binding,
            groups=(),
            ready_event=object(),
            resources=(),
            _owner_token=self._owner_token,
        )
        self.prepared_copies.append(prepared_copy)
        return prepared_copy

    def quarantine_scatter_copy(
        self,
        prepared_copy: PackedPreparedScatterCopy,
    ) -> None:
        """Retain a projection whose construction lifetime became ambiguous.

        :param prepared_copy: Exact projection created by this executor.
        """

        if prepared_copy._owner_token is not self._owner_token:
            raise RuntimeError("prepared scatter copy belongs to another executor")
        if any(retained is prepared_copy for retained in self.quarantined_copies):
            return
        self.quarantined_copies.append(prepared_copy)


@dataclasses.dataclass
class _Fixture:
    """Complete CPU-only packed request transaction inputs."""

    allocation_authority: DecodeAllocationLeaseAuthority
    allocation_lease: DecodeAllocationLease
    allocation_snapshot: DecodeAllocationLeaseSnapshot
    auxiliary_allocator: _AuxiliaryAllocator
    auxiliary_authority: PackedAuxiliaryAllocationLeaseAuthority
    auxiliary_lease: PackedAuxiliaryAllocationLease
    coordinator: PackedDestinationOutcomeCoordinator
    consumer_authority: object
    copy_executor: _CopyExecutor
    full_allocator: BaseTokenToKVPoolAllocator
    lifecycle_authority: object
    owner: _RequestOwner
    plans: tuple[PackedRequestChunkPlan, ...]
    protocol: PackedDecodeProtocol
    request_pool: ReqToTokenPool
    staging_allocator: _StagingAllocator
    swa_allocator: BaseTokenToKVPoolAllocator


def _policy(writer_id: StagingWriterId) -> PackedDestinationVisibilityPolicy:
    """Build one deterministic direct-event route policy.

    :param writer_id: Exact writer owning the route.
    :returns: Immutable CPU-test visibility policy.
    """

    return PackedDestinationVisibilityPolicy(
        transport_path=PackedTransportPath.CUDA_IPC,
        lane_identifier=f"cuda-ipc-writer-{writer_id.source_attn_tp_rank}",
        completion_mechanism=(
            PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
        ),
        writes_ordering=PackedGpuDirectWritesOrdering.OWNER,
        flush_options=PackedGpuDirectFlushOptions.NONE,
        native_data_transport=None,
        native_data_device=None,
        native_runtime_artifact_digest=None,
    )


def _component_buffer(
    component_id: StagingComponentId,
    receipt_pages: tuple[int, ...],
    item_len: int,
    page_size: int,
) -> StagingComponentBuffer:
    """Build one fake destination registration from an allocation receipt.

    :param component_id: Main KV or SWA identity.
    :param receipt_pages: Exact ordered physical pages.
    :param item_len: Destination bytes per page.
    :param page_size: Allocation page size.
    :returns: Complete fake registered destination buffer.
    """

    pointer_offset = 0 if component_id == MAIN_KV_COMPONENT else 0x10000
    return StagingComponentBuffer(
        component_id=component_id,
        tensor_ptrs=(0x100000 + pointer_offset,),
        data_lens=(item_len * 64,),
        item_lens=(item_len,),
        layer_ids=(0,),
        page_size=page_size,
        page_array=np.asarray(receipt_pages, dtype=np.int32),
    )


def _build_plans(
    snapshot: DecodeAllocationLeaseSnapshot,
    room_id: int,
) -> tuple[PackedRequestChunkPlan, ...]:
    """Build fixed chunks covering FULL and final SWA pages exactly.

    :param snapshot: Exact decode allocation receipt.
    :param room_id: Decoder-minted request room.
    :returns: Canonically ordered request chunk plans.
    """

    receipts = {receipt.component: receipt for receipt in snapshot.components}
    full = receipts[DecodeAllocationComponent.FULL]
    swa = receipts[DecodeAllocationComponent.SWA]
    source_tp_size = snapshot.writer_manifest.source_tp_size
    destination_item_len = 16 * full.page_size
    source_item_len = destination_item_len // source_tp_size
    registry_components = [
        _component_buffer(
            MAIN_KV_COMPONENT,
            full.physical_pages,
            destination_item_len,
            full.page_size,
        )
    ]
    if not swa.zero_work:
        registry_components.append(
            _component_buffer(
                SWA_COMPONENT,
                swa.physical_pages,
                destination_item_len,
                swa.page_size,
            )
        )
    registry = StagingComponentBufferRegistry(tuple(registry_components))
    policies = tuple(
        (writer_id, _policy(writer_id))
        for writer_id in snapshot.writer_manifest.writers
    )
    if len(full.physical_pages) == 1:
        full_counts = (1,)
        full_offsets = (0,)
    else:
        full_split = len(full.physical_pages) // 2
        full_counts = (
            full_split,
            len(full.physical_pages) - full_split,
        )
        full_offsets = (0, full_split)
    plans: list[PackedRequestChunkPlan] = []
    for chunk_id in range(len(full_counts)):
        is_last = chunk_id == len(full_counts) - 1
        component_ids = [MAIN_KV_COMPONENT]
        spans = [
            StagingComponentSpan(
                component_id=MAIN_KV_COMPONENT,
                source_index_offset=0,
                destination_index_offset=full_offsets[chunk_id],
                logical_token_count=full_counts[chunk_id] * full.page_size,
                physical_token_count=full_counts[chunk_id] * full.page_size,
            )
        ]
        if is_last and not swa.zero_work:
            component_ids.append(SWA_COMPONENT)
            spans.append(
                StagingComponentSpan(
                    component_id=SWA_COMPONENT,
                    source_index_offset=0,
                    destination_index_offset=0,
                    logical_token_count=len(swa.physical_pages) * swa.page_size,
                    physical_token_count=len(swa.physical_pages) * swa.page_size,
                )
            )
        source_components = tuple(
            StagingComponentGeometry(
                component_id=component_id,
                item_lens=(source_item_len,),
                layer_ids=(0,),
                page_size=(
                    full.page_size
                    if component_id == MAIN_KV_COMPONENT
                    else swa.page_size
                ),
            )
            for component_id in component_ids
        )
        destination_components = tuple(
            StagingComponentGeometry(
                component_id=component_id,
                item_lens=(destination_item_len,),
                layer_ids=(0,),
                page_size=(
                    full.page_size
                    if component_id == MAIN_KV_COMPONENT
                    else swa.page_size
                ),
            )
            for component_id in component_ids
        )
        key = PackedChunkKey(
            room_id=room_id,
            chunk_id=chunk_id,
            request_generation=snapshot.lease_id,
        )
        spec = PackedLayoutSpec(
            chunk_id=chunk_id,
            is_last=is_last,
            spans=tuple(spans),
            source_components=source_components,
            destination_components=destination_components,
            writers=snapshot.writer_manifest.writers,
            topology=PackedTopology(
                source_tp_size=source_tp_size,
                destination_tp_size=1,
                destination_tp_rank=0,
            ),
        )
        plans.append(
            PackedRequestChunkPlan(
                key=key,
                spec=spec,
                destination_registry=registry,
                visibility_policies=policies,
            )
        )
    return tuple(plans)


def _fixture(
    source_tp_size: int = 2,
    room_id: int = 101,
    *,
    logical_start: int = 0,
    logical_length: int = 4,
    swa_logical_start: int = 2,
    swa_logical_length: int = 2,
    page_size: int = 1,
    auxiliary_segments: tuple[PackedAuxiliaryDestinationSegment, ...] | None = None,
) -> _Fixture:
    """Allocate one complete CPU request with FULL, SWA, and zero Mamba.

    :param source_tp_size: Exact supported packed source width.
    :param room_id: Decoder-minted request room.
    :param logical_start: Request-local FULL migration start.
    :param logical_length: Exact FULL migration length.
    :param swa_logical_start: Request-local SWA migration start.
    :param swa_logical_length: Exact SWA migration length.
    :param page_size: Tokens represented by one allocation page.
    :param auxiliary_segments: Optional exact auxiliary destination geometry.
    :returns: Complete transaction fixture.
    """

    request_pool = ReqToTokenPool(
        size=4,
        max_context_len=max(32, logical_start + logical_length),
        device="cpu",
        enable_memory_saver=False,
    )
    owner = _RequestOwner()
    slots = request_pool.alloc([owner])
    assert slots is not None
    assert owner.req_pool_idx is not None

    allocator_type = (
        TokenToKVPoolAllocator if page_size == 1 else PagedTokenToKVPoolAllocator
    )
    allocator_kwargs = {
        "size": max(1024, logical_length * 2),
        "dtype": torch.float16,
        "device": "cpu",
        "kvcache": object(),
        "need_sort": False,
    }
    if page_size > 1:
        allocator_kwargs["page_size"] = page_size
    full_allocator = allocator_type(**allocator_kwargs)
    swa_allocator = allocator_type(**allocator_kwargs)
    full_indices = full_allocator.alloc(logical_length)
    swa_indices = swa_allocator.alloc(swa_logical_length)
    assert full_indices is not None
    assert swa_indices is not None

    lifecycle_authority = object()
    consumer_authority = object()
    allocation_authority = DecodeAllocationLeaseAuthority(lifecycle_authority)
    request_generation = int(request_pool.req_generation[owner.req_pool_idx].item())
    allocation_lease = allocation_authority.acquire(
        request_pool=request_pool,
        request_slot=owner.req_pool_idx,
        expected_request_generation=request_generation,
        writer_manifest=DecodeWriterManifest.for_tensor_parallel(source_tp_size),
        component_claims=(
            DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.FULL,
                logical_start=logical_start,
                logical_length=int(full_indices.numel()),
                allocator=full_allocator,
                indices=full_indices,
            ),
            DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.SWA,
                logical_start=swa_logical_start,
                logical_length=int(swa_indices.numel()),
                allocator=swa_allocator,
                indices=swa_indices,
            ),
            DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.MAMBA,
                logical_start=0,
                logical_length=0,
                allocator=None,
                indices=None,
            ),
        ),
    )
    snapshot = allocation_authority.snapshot(allocation_lease)
    auxiliary_allocator = _AuxiliaryAllocator(auxiliary_segments)
    auxiliary_authority = PackedAuxiliaryAllocationLeaseAuthority(
        lifecycle_authority,
        consumer_authority,
    )
    auxiliary_lease = auxiliary_authority.acquire(auxiliary_allocator)
    staging_allocator = _StagingAllocator()
    protocol = PackedDecodeProtocol(staging_allocator)
    coordinator = PackedDestinationOutcomeCoordinator(
        protocol,
        _VisibilityExecutor(),
    )
    copy_executor = _CopyExecutor()
    return _Fixture(
        allocation_authority=allocation_authority,
        allocation_lease=allocation_lease,
        allocation_snapshot=snapshot,
        auxiliary_allocator=auxiliary_allocator,
        auxiliary_authority=auxiliary_authority,
        auxiliary_lease=auxiliary_lease,
        coordinator=coordinator,
        consumer_authority=consumer_authority,
        copy_executor=copy_executor,
        full_allocator=full_allocator,
        lifecycle_authority=lifecycle_authority,
        owner=owner,
        plans=_build_plans(snapshot, room_id),
        protocol=protocol,
        request_pool=request_pool,
        staging_allocator=staging_allocator,
        swa_allocator=swa_allocator,
    )


def _transaction(
    fixture: _Fixture,
    plans: tuple[PackedRequestChunkPlan, ...] | None = None,
) -> PackedDecodeRequestTransaction:
    """Construct one request transaction from its fixture.

    :param fixture: Exact allocation and chunk inputs.
    :param plans: Optional replacement chunk plans.
    :returns: Prepared request transaction.
    """

    selected_plans = fixture.plans if plans is None else plans
    return PackedDecodeRequestTransaction(
        room_id=selected_plans[0].key.room_id,
        request_owner=fixture.owner,
        allocation_lease=fixture.allocation_lease,
        allocation_authority=fixture.allocation_authority,
        lifecycle_authority=fixture.lifecycle_authority,
        protocol=fixture.protocol,
        outcome_coordinator=fixture.coordinator,
        copy_executor=fixture.copy_executor,
        chunk_plans=selected_plans,
        auxiliary_allocation_lease=fixture.auxiliary_lease,
        auxiliary_allocation_authority=fixture.auxiliary_authority,
        destination_process_generation=bytes(range(PACKED_REQUEST_GENERATION_BYTES)),
        native_route_digest=bytes(range(PACKED_REQUEST_DIGEST_BYTES)),
        runtime_cohort_digest=bytes(reversed(range(PACKED_REQUEST_DIGEST_BYTES))),
    )


def _prepare_chunk(
    transaction: PackedDecodeRequestTransaction,
    plan: PackedRequestChunkPlan,
) -> tuple[bytes, int]:
    """Reach READY consensus for one exact chunk.

    :param transaction: Published request transaction.
    :param plan: Exact decoder-authored chunk.
    :returns: Canonical layout digest and staging lease identity.
    """

    ready_messages = ()
    digest = plan.spec.build().digest
    for writer_id in plan.spec.writers:
        produced = transaction.handle_prepare(
            PackedPrepare(
                key=plan.key,
                writer_id=writer_id,
                spec=plan.spec,
                digest=digest,
            ),
            writer_id,
        )
        if len(produced) > 0:
            ready_messages = produced
    assert len(ready_messages) == len(plan.spec.writers)
    assert all(message.digest == digest for message in ready_messages)
    lease_ids = {message.lease_id for message in ready_messages}
    assert len(lease_ids) == 1
    return digest, next(iter(lease_ids))


def _complete_writers(
    transaction: PackedDecodeRequestTransaction,
    plan: PackedRequestChunkPlan,
    digest: bytes,
    lease_id: int,
) -> None:
    """Deliver successful terminal outcomes for every chunk writer.

    :param transaction: Published request transaction.
    :param plan: Exact chunk plan.
    :param digest: Canonical layout digest.
    :param lease_id: Exact staging lease identity.
    """

    for writer_id, policy in plan.visibility_policies:
        evidence = PackedWriterVisibilityEvidence(
            policy_digest=policy.digest,
            transport_path=policy.transport_path,
            lane_identifier=policy.lane_identifier,
            completion_mechanism=policy.completion_mechanism,
            writer_action=PackedWriterVisibilityAction.CUDA_EVENT_RECORDED,
        )
        transaction.handle_writer_outcome(
            PackedWriterOutcome(
                key=plan.key,
                writer_id=writer_id,
                digest=digest,
                lease_id=lease_id,
                status=PackedWriterOutcomeStatus.DONE,
                visibility=evidence,
                reason=None,
            ),
            writer_id,
        )


def _complete_auxiliary(
    transaction: PackedDecodeRequestTransaction,
    publication: PackedRequestPublication,
) -> PackedAuxiliaryOutcome:
    """Deliver one exact canonical auxiliary terminal outcome.

    :param transaction: Published request transaction.
    :param publication: Exact decoder-authored request publication.
    :returns: Accepted terminal auxiliary outcome.
    """

    plan = publication.auxiliary_plan
    outcome = PackedAuxiliaryOutcome(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_dram_handle_generation=29,
        descriptor_digest=b"d" * PACKED_REQUEST_DIGEST_BYTES,
        evidence_digest=b"e" * PACKED_REQUEST_DIGEST_BYTES,
    )
    assert transaction.handle_auxiliary_outcome(
        outcome,
        plan.canonical_writer_id,
    )
    return outcome


def _complete_dflash_boundary(
    transaction: PackedDecodeRequestTransaction,
    publication: PackedRequestPublication,
    *,
    boundary_token_id: int = 17,
    native_handle_generation: int = 29,
) -> PackedDFlashBoundaryOutcome:
    """Deliver one authenticated all-VRAM DFlash boundary outcome.

    :param transaction: Published request transaction.
    :param publication: Exact decoder-authored request publication.
    :param boundary_token_id: Source-authored first decode token.
    :param native_handle_generation: Exact terminal native handle generation.
    :returns: Accepted terminal DFlash outcome.
    """

    plan = publication.auxiliary_plan
    outcome = PackedDFlashBoundaryOutcome.create(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_handle_generation=native_handle_generation,
        descriptor_digest=b"d" * PACKED_REQUEST_DIGEST_BYTES,
        evidence_digest=b"e" * PACKED_REQUEST_DIGEST_BYTES,
        metadata=PackedDFlashBoundaryMetadata(
            boundary_token_id=boundary_token_id,
            cached_tokens=4,
            cached_tokens_device=3,
            cached_tokens_host=2,
            cached_tokens_storage=1,
            image_tokens=7,
            audio_tokens=8,
            video_tokens=9,
        ),
    )
    assert transaction.handle_dflash_boundary_outcome(
        outcome,
        plan.canonical_writer_id,
    )
    return outcome


def _commit_completed_request(
    transaction: PackedDecodeRequestTransaction,
) -> None:
    """Complete teardown and commit one fully transferred request.

    :param transaction: Scatter-completed request.
    """

    receipt = None
    for request in transaction.begin_teardown():
        candidate = transaction.handle_teardown_ack(
            _ack(request),
            request.writer_id,
        )
        if candidate is not None:
            receipt = candidate
    assert receipt is not None
    transaction.commit_on_scheduler_thread(receipt)


def _committed_dflash_transaction(
    *,
    room_id: int,
) -> tuple[
    _Fixture,
    PackedDecodeRequestTransaction,
    PackedDFlashBoundaryOutcome,
]:
    """Build one destination-consumption-waiting DFlash transaction.

    :param room_id: Decoder-minted request room.
    :returns: Fixture, committed transaction, and authenticated outcome.
    """

    fixture = _fixture(
        room_id=room_id,
        auxiliary_segments=(
            PackedAuxiliaryDestinationSegment(
                address=0xA00000,
                item_length=8,
            ),
        ),
    )
    transaction = _transaction(fixture)
    publication = transaction.publish()
    ready = [_prepare_chunk(transaction, plan) for plan in fixture.plans]
    for plan, (digest, lease_id) in zip(fixture.plans, ready, strict=True):
        _complete_writers(transaction, plan, digest, lease_id)
        scatter = transaction.begin_scatter(plan.key)
        transaction.complete_scatter(scatter)
    outcome = _complete_dflash_boundary(transaction, publication)
    _commit_completed_request(transaction)
    return fixture, transaction, outcome


def _complete_request_transfers(
    fixture: _Fixture,
    transaction: PackedDecodeRequestTransaction,
    publication: PackedRequestPublication,
) -> PackedAuxiliaryOutcome:
    """Complete every chunk and the request auxiliary transfer.

    :param fixture: Exact transaction fixture.
    :param transaction: Published request transaction.
    :param publication: Exact request publication.
    :returns: Accepted auxiliary terminal outcome.
    """

    ready = [_prepare_chunk(transaction, plan) for plan in fixture.plans]
    for plan, (digest, lease_id) in zip(fixture.plans, ready, strict=True):
        _complete_writers(transaction, plan, digest, lease_id)
        scatter = transaction.begin_scatter(plan.key)
        transaction.complete_scatter(scatter)
    return _complete_auxiliary(transaction, publication)


def _different_bytes(value: bytes) -> bytes:
    """Return an equal-length byte string with a distinct first byte.

    :param value: Non-empty source bytes.
    :returns: Deterministically distinct bytes.
    """

    if len(value) == 0:
        raise ValueError("source bytes must not be empty")
    return bytes((value[0] ^ 1,)) + value[1:]


def _stale_auxiliary_plan(
    plan: PackedAuxiliaryPlan,
    mismatch: str,
) -> PackedAuxiliaryPlan:
    """Change one exact auxiliary identity dimension.

    :param plan: Live decoder-authored auxiliary plan.
    :param mismatch: Identity dimension to replace.
    :returns: Structurally valid stale plan.
    """

    if mismatch == "request_generation":
        return dataclasses.replace(
            plan,
            key=dataclasses.replace(
                plan.key,
                request_generation=_different_bytes(plan.key.request_generation),
            ),
        )
    if mismatch == "request_slot_generation":
        return dataclasses.replace(
            plan,
            request_slot_generation=plan.request_slot_generation + 1,
        )
    if mismatch == "metadata_buffer_index":
        return dataclasses.replace(
            plan,
            metadata_buffer_index=plan.metadata_buffer_index + 1,
        )
    if mismatch == "metadata_slot_generation":
        return dataclasses.replace(
            plan,
            metadata_slot_generation=_different_bytes(plan.metadata_slot_generation),
        )
    if mismatch == "destination_process_generation":
        return dataclasses.replace(
            plan,
            destination_process_generation=_different_bytes(
                plan.destination_process_generation
            ),
        )
    if mismatch == "native_route_digest":
        return dataclasses.replace(
            plan,
            native_route_digest=_different_bytes(plan.native_route_digest),
        )
    if mismatch == "runtime_cohort_digest":
        return dataclasses.replace(
            plan,
            runtime_cohort_digest=_different_bytes(plan.runtime_cohort_digest),
        )
    if mismatch == "destination_segments":
        return dataclasses.replace(
            plan,
            destination_segments=tuple(reversed(plan.destination_segments)),
        )
    raise ValueError(f"unknown auxiliary mismatch: {mismatch}")


def _ack(request: PackedRequestTeardown) -> PackedRequestTeardownAck:
    """Build the exact acknowledgement for one teardown request.

    :param request: Exact destination teardown request.
    :returns: Matching source acknowledgement.
    """

    return PackedRequestTeardownAck(
        key=request.key,
        writer_id=request.writer_id,
        request_slot_generation=request.request_slot_generation,
        writer_manifest_digest=request.writer_manifest_digest,
        allocation_digest=request.allocation_digest,
        teardown_generation=request.teardown_generation,
        auxiliary_handle_generation=request.auxiliary_handle_generation,
    )


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_page64_exact_repeat_preserves_final_full_and_swa_page(
    source_tp_size: int,
) -> None:
    """Build a valid packed transaction for the exact-repeat migration tail.

    :param source_tp_size: Supported packed prefill width.
    """

    fixture = _fixture(
        source_tp_size,
        logical_start=448,
        logical_length=64,
        swa_logical_start=448,
        swa_logical_length=64,
        page_size=64,
    )
    receipts = {
        receipt.component: receipt for receipt in fixture.allocation_snapshot.components
    }
    full = receipts[DecodeAllocationComponent.FULL]
    swa = receipts[DecodeAllocationComponent.SWA]

    assert (full.logical_start, full.logical_length, full.page_size) == (448, 64, 64)
    assert (swa.logical_start, swa.logical_length, swa.page_size) == (448, 64, 64)
    assert len(fixture.plans) == 1
    final_components = {span.component_id for span in fixture.plans[0].spec.spans}
    assert final_components == {MAIN_KV_COMPONENT, SWA_COMPONENT}
    _transaction(fixture)


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_multichunk_success_waits_for_every_chunk_and_scheduler_commit(
    source_tp_size: int,
) -> None:
    """Supported cohorts complete out of order without an early commit."""

    fixture = _fixture(source_tp_size)
    transaction = _transaction(fixture)
    publication = transaction.publish()
    assert publication.key.request_generation == fixture.allocation_snapshot.lease_id

    ready = {
        plan.key.chunk_id: _prepare_chunk(transaction, plan) for plan in fixture.plans
    }
    second = fixture.plans[1]
    second_digest, second_lease_id = ready[1]
    _complete_writers(
        transaction,
        second,
        second_digest,
        second_lease_id,
    )
    second_scatter = transaction.begin_scatter(second.key)
    transaction.complete_scatter(second_scatter)
    assert transaction.state is PackedRequestTransactionState.SUBMITTED
    assert (
        fixture.allocation_authority.snapshot(fixture.allocation_lease).state
        is DecodeAllocationLeaseState.SUBMITTED
    )

    first = fixture.plans[0]
    first_digest, first_lease_id = ready[0]
    _complete_writers(
        transaction,
        first,
        first_digest,
        first_lease_id,
    )
    assert transaction.state is PackedRequestTransactionState.SUBMITTED
    first_scatter = transaction.begin_scatter(first.key)
    transaction.complete_scatter(first_scatter)
    assert transaction.state is PackedRequestTransactionState.SUBMITTED
    with pytest.raises(PackedRequestTransactionError, match="teardown"):
        transaction.begin_teardown()

    auxiliary_outcome = _complete_auxiliary(transaction, publication)
    assert transaction.state is PackedRequestTransactionState.SCATTER_COMPLETED
    assert transaction.snapshot().auxiliary_outcome == auxiliary_outcome
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.SUBMITTED
    )
    assert set(fixture.staging_allocator.releases) == {
        first_lease_id,
        second_lease_id,
    }

    teardown = transaction.begin_teardown()
    receipt = None
    for request in reversed(teardown):
        candidate = transaction.handle_teardown_ack(
            _ack(request),
            request.writer_id,
        )
        if candidate is not None:
            receipt = candidate
    assert receipt is not None
    assert transaction.state is PackedRequestTransactionState.COMMIT_READY

    errors: list[BaseException] = []

    def commit_from_wrong_thread() -> None:
        try:
            transaction.commit_on_scheduler_thread(receipt)
        except PackedRequestTransactionError as error:
            errors.append(error)

    thread = threading.Thread(target=commit_from_wrong_thread)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], PackedRequestTransactionError)
    assert transaction.state is PackedRequestTransactionState.COMMIT_READY

    assert transaction.commit_on_scheduler_thread(receipt) is fixture.owner
    assert (
        transaction.state
        is PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
    )
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST
    )
    with pytest.raises(DecodeAllocationLeaseError, match="not registered"):
        fixture.allocation_authority.snapshot(fixture.allocation_lease)
    transaction.complete_auxiliary_consumption_on_scheduler_thread(
        fixture.consumer_authority
    )
    assert transaction.state is PackedRequestTransactionState.COMMITTED
    assert len(fixture.auxiliary_allocator.releases) == 1
    with pytest.raises(PackedAuxiliaryAllocationError, match="not registered"):
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease)


@pytest.mark.parametrize(
    ("first_offset", "match"),
    ((3, "gap"), (1, "overlaps")),
)
def test_destination_coverage_rejects_gap_and_overlap(
    first_offset: int,
    match: str,
) -> None:
    """Reject decoder plans that do not partition destination pages exactly."""

    fixture = _fixture()
    second = fixture.plans[1]
    spans = list(second.spec.spans)
    spans[0] = dataclasses.replace(
        spans[0],
        destination_index_offset=first_offset,
    )
    invalid_spec = dataclasses.replace(second.spec, spans=tuple(spans))
    invalid_plans = (
        fixture.plans[0],
        dataclasses.replace(second, spec=invalid_spec),
    )
    with pytest.raises(ValueError, match=match):
        _transaction(fixture, invalid_plans)


def test_registry_pages_must_match_ordered_allocation_receipt() -> None:
    """Reject a registry that permutes or replaces destination physical pages."""

    fixture = _fixture()
    components = list(fixture.plans[0].destination_registry.components)
    main = components[0]
    wrong_pages = np.array(main.page_array, copy=True)
    wrong_pages[0] += 7
    components[0] = dataclasses.replace(main, page_array=wrong_pages)
    wrong_registry = StagingComponentBufferRegistry(tuple(components))
    invalid_plans = tuple(
        dataclasses.replace(plan, destination_registry=wrong_registry)
        for plan in fixture.plans
    )
    with pytest.raises(ValueError, match="ordered allocation receipt"):
        _transaction(fixture, invalid_plans)


def test_stale_generation_replay_is_rejected_without_mutation() -> None:
    """A prior room generation cannot address the live request transaction."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    transaction.publish()
    plan = fixture.plans[0]
    stale_key = dataclasses.replace(
        plan.key,
        request_generation=bytes(reversed(plan.key.request_generation)),
    )
    stale_prepare = PackedPrepare(
        key=stale_key,
        writer_id=plan.spec.writers[0],
        spec=plan.spec,
        digest=plan.spec.build().digest,
    )
    with pytest.raises(PackedRequestTransactionError, match="generation"):
        transaction.handle_prepare(stale_prepare, plan.spec.writers[0])
    assert transaction.state is PackedRequestTransactionState.PUBLISHED


def test_publication_is_irreversible_and_duplicate_publication_is_rejected() -> None:
    """External publication closes both rollback and duplicate publication."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    transaction.publish()
    with pytest.raises(PackedRequestTransactionError, match="publication"):
        transaction.publish()
    with pytest.raises(PackedRequestTransactionError, match="cancellation"):
        transaction.cancel_unpublished()
    assert (
        fixture.allocation_authority.snapshot(fixture.allocation_lease).state
        is DecodeAllocationLeaseState.PUBLISHED
    )
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.PUBLISHED
    )


def test_unpublished_cancellation_releases_pins_and_registrations() -> None:
    """Prepared cancellation removes all internal metadata and allocation pins."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    assert transaction.cancel_unpublished() is fixture.owner
    assert transaction.state is PackedRequestTransactionState.CANCELLED
    with pytest.raises(DecodeAllocationLeaseError, match="not registered"):
        fixture.allocation_authority.snapshot(fixture.allocation_lease)
    with pytest.raises(PackedAuxiliaryAllocationError, match="not registered"):
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease)
    assert len(fixture.auxiliary_allocator.releases) == 1
    for plan in fixture.plans:
        with pytest.raises(PackedProtocolError, match="not registered"):
            fixture.protocol.snapshot(plan.key)


def test_transaction_prepares_scatter_metadata_before_publication() -> None:
    """Each begun scatter retains its exact construction-time projection."""

    fixture = _fixture()
    transaction = _transaction(fixture)

    assert len(fixture.copy_executor.preparations) == len(fixture.plans)
    assert len(fixture.copy_executor.prepared_copies) == len(fixture.plans)
    assert fixture.copy_executor.quarantined_copies == []

    transaction.publish()
    plan = fixture.plans[0]
    digest, lease_id = _prepare_chunk(transaction, plan)
    _complete_writers(transaction, plan, digest, lease_id)
    scatter = transaction.begin_scatter(plan.key)
    preparation = fixture.copy_executor.preparations[0]
    prepared_copy = fixture.copy_executor.prepared_copies[0]

    assert scatter.prepared_copy is prepared_copy
    assert prepared_copy.key == preparation.key
    assert prepared_copy.layout is preparation.layout
    assert prepared_copy.destination_binding is preparation.destination_binding
    assert scatter.work.layout is prepared_copy.layout
    assert scatter.work.destination_binding is prepared_copy.destination_binding


def test_construction_failure_quarantines_every_built_scatter_projection() -> None:
    """A later registration failure cannot release earlier asynchronous metadata."""

    fixture = _fixture()
    fixture.copy_executor.fail_on_preparation = 2

    with pytest.raises(
        PackedRequestTransactionError,
        match="chunk registration failed",
    ):
        _transaction(fixture)

    assert len(fixture.copy_executor.preparations) == 2
    assert len(fixture.copy_executor.prepared_copies) == 1
    assert fixture.copy_executor.quarantined_copies == [
        fixture.copy_executor.prepared_copies[0]
    ]
    for plan in fixture.plans:
        with pytest.raises(PackedProtocolError, match="not registered"):
            fixture.protocol.snapshot(plan.key)


def test_transaction_rejects_a_lease_published_by_another_authority() -> None:
    """Construction cannot adopt an externally published allocation lease."""

    fixture = _fixture()
    fixture.allocation_authority.record_publication(
        fixture.allocation_lease,
        fixture.lifecycle_authority,
    )

    with pytest.raises(PackedRequestTransactionError, match="prepared"):
        _transaction(fixture)


def test_independent_tp1_replicas_own_disjoint_request_registries() -> None:
    """Separate TP1 replicas can serve independent generations concurrently."""

    left = _fixture(room_id=201)
    right = _fixture(room_id=202)
    left_transaction = _transaction(left)
    right_transaction = _transaction(right)
    left_publication = left_transaction.publish()
    right_publication = right_transaction.publish()
    assert left_publication.key != right_publication.key
    assert left.plans[0].destination_registry is not right.plans[0].destination_registry
    assert (
        tuple(left.plans[0].destination_registry.components[0].page_array)
        == left.allocation_snapshot.components[0].physical_pages
    )
    assert (
        tuple(right.plans[0].destination_registry.components[0].page_array)
        == right.allocation_snapshot.components[0].physical_pages
    )


def test_request_teardown_wire_round_trip_is_versioned_and_exact() -> None:
    """Round-trip request teardown and acknowledgement through strict msgpack."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    ready = [_prepare_chunk(transaction, plan) for plan in fixture.plans]
    for plan, (digest, lease_id) in zip(fixture.plans, ready, strict=True):
        _complete_writers(transaction, plan, digest, lease_id)
        scatter = transaction.begin_scatter(plan.key)
        transaction.complete_scatter(scatter)
    _complete_auxiliary(transaction, publication)
    request = transaction.begin_teardown()[0]
    acknowledgement = _ack(request)
    assert decode_packed_message(encode_packed_message(request)) == request
    assert (
        decode_packed_message(encode_packed_message(acknowledgement)) == acknowledgement
    )


@pytest.mark.parametrize(
    "mismatch",
    (
        "request_generation",
        "request_slot_generation",
        "metadata_buffer_index",
        "metadata_slot_generation",
        "destination_process_generation",
        "native_route_digest",
        "runtime_cohort_digest",
        "destination_segments",
    ),
)
def test_auxiliary_outcome_rejects_every_stale_identity_dimension(
    mismatch: str,
) -> None:
    """A structurally valid stale auxiliary plan quarantines both allocations."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    stale_plan = _stale_auxiliary_plan(publication.auxiliary_plan, mismatch)
    outcome = PackedAuxiliaryOutcome(
        plan=stale_plan,
        writer_id=stale_plan.canonical_writer_id,
        native_dram_handle_generation=31,
        descriptor_digest=b"d" * PACKED_REQUEST_DIGEST_BYTES,
        evidence_digest=b"e" * PACKED_REQUEST_DIGEST_BYTES,
    )

    with pytest.raises(PackedRequestTransactionError, match="exact request plan"):
        transaction.handle_auxiliary_outcome(
            outcome,
            stale_plan.canonical_writer_id,
        )

    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.QUARANTINED
    )
    assert (
        fixture.allocation_authority.snapshot(fixture.allocation_lease).state
        is DecodeAllocationLeaseState.QUARANTINED
    )


def test_auxiliary_outcome_is_exactly_idempotent_and_conflicts_quarantine() -> None:
    """Only a byte-for-byte duplicate auxiliary outcome is idempotent."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    outcome = _complete_auxiliary(transaction, publication)

    assert not transaction.handle_auxiliary_outcome(
        outcome,
        outcome.writer_id,
    )
    conflicting = dataclasses.replace(
        outcome,
        native_dram_handle_generation=outcome.native_dram_handle_generation + 1,
    )
    with pytest.raises(PackedRequestTransactionError, match="conflicting duplicate"):
        transaction.handle_auxiliary_outcome(
            conflicting,
            conflicting.writer_id,
        )
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_dflash_boundary_adoption_is_one_shot_and_gates_exact_row_release() -> None:
    """Release a DFlash row only through its exact issued adoption authority."""

    fixture, transaction, outcome = _committed_dflash_transaction(room_id=171)

    adoption = transaction.begin_dflash_boundary_adoption_on_scheduler_thread()

    assert type(adoption) is PackedDFlashBoundaryDecodeAdoption
    assert adoption.metadata is outcome.metadata
    assert adoption.outcome_digest == outcome.outcome_digest
    assert adoption.lease.state is PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST
    assert adoption.slot.metadata_buffer_index == 17
    with pytest.raises(PackedRequestTransactionError, match="already issued"):
        transaction.begin_dflash_boundary_adoption_on_scheduler_thread()

    transaction.complete_auxiliary_consumption_on_scheduler_thread(
        fixture.consumer_authority,
        dflash_adoption=adoption,
    )

    assert transaction.state is PackedRequestTransactionState.COMMITTED
    assert len(fixture.auxiliary_allocator.releases) == 1
    with pytest.raises(PackedAuxiliaryAllocationError, match="not registered"):
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease)


def test_dflash_boundary_release_without_exact_adoption_fails_closed() -> None:
    """Quarantine the row when scheduler consumption lacks copy authority."""

    fixture, transaction, outcome = _committed_dflash_transaction(room_id=172)
    adoption = transaction.begin_dflash_boundary_adoption_on_scheduler_thread()
    conflicting = dataclasses.replace(
        adoption,
        outcome_digest=_different_bytes(outcome.outcome_digest),
    )

    with pytest.raises(PackedRequestTransactionError, match="exact adoption"):
        transaction.complete_auxiliary_consumption_on_scheduler_thread(
            fixture.consumer_authority,
            dflash_adoption=conflicting,
        )

    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert len(fixture.auxiliary_allocator.releases) == 0
    assert len(fixture.auxiliary_allocator.quarantines) == 1


def test_dflash_boundary_outcome_is_idempotent_and_conflicts_quarantine() -> None:
    """Accept one exact DFlash outcome and quarantine conflicting evidence."""

    fixture = _fixture(
        room_id=173,
        auxiliary_segments=(
            PackedAuxiliaryDestinationSegment(
                address=0xA00000,
                item_length=8,
            ),
        ),
    )
    transaction = _transaction(fixture)
    publication = transaction.publish()
    outcome = _complete_dflash_boundary(transaction, publication)

    assert not transaction.handle_dflash_boundary_outcome(
        outcome,
        outcome.writer_id,
    )
    conflicting = PackedDFlashBoundaryOutcome.create(
        plan=outcome.plan,
        writer_id=outcome.writer_id,
        native_handle_generation=outcome.native_handle_generation + 1,
        descriptor_digest=outcome.descriptor_digest,
        evidence_digest=outcome.evidence_digest,
        metadata=outcome.metadata,
    )
    with pytest.raises(PackedRequestTransactionError, match="conflicting duplicate"):
        transaction.handle_dflash_boundary_outcome(
            conflicting,
            conflicting.writer_id,
        )
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


@pytest.mark.parametrize("fault", ("canonical_handle", "noncanonical_handle"))
def test_teardown_rejects_inexact_auxiliary_handle_evidence(fault: str) -> None:
    """Canonical and noncanonical writers must echo their exact handle shape."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    outcome = _complete_request_transfers(fixture, transaction, publication)
    requests = transaction.begin_teardown()
    if fault == "canonical_handle":
        request = next(
            candidate
            for candidate in requests
            if candidate.writer_id == publication.auxiliary_plan.canonical_writer_id
        )
        acknowledgement = dataclasses.replace(
            _ack(request),
            auxiliary_handle_generation=outcome.native_dram_handle_generation + 1,
        )
    else:
        request = next(
            candidate
            for candidate in requests
            if candidate.writer_id != publication.auxiliary_plan.canonical_writer_id
        )
        acknowledgement = dataclasses.replace(
            _ack(request),
            auxiliary_handle_generation=outcome.native_dram_handle_generation,
        )

    with pytest.raises(PackedRequestTransactionError, match="exact request"):
        transaction.handle_teardown_ack(
            acknowledgement,
            request.writer_id,
        )
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_teardown_acknowledgements_are_exactly_idempotent() -> None:
    """Exact acknowledgement replays are harmless and conflicts quarantine."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    _complete_request_transfers(fixture, transaction, publication)
    requests = transaction.begin_teardown()
    canonical_request = next(
        request
        for request in requests
        if request.writer_id == publication.auxiliary_plan.canonical_writer_id
    )
    canonical_ack = _ack(canonical_request)
    assert (
        transaction.handle_teardown_ack(
            canonical_ack,
            canonical_request.writer_id,
        )
        is None
    )
    assert (
        transaction.handle_teardown_ack(
            canonical_ack,
            canonical_request.writer_id,
        )
        is None
    )

    receipt = None
    for request in requests:
        if request is canonical_request:
            continue
        receipt = transaction.handle_teardown_ack(
            _ack(request),
            request.writer_id,
        )
    assert receipt is not None
    assert (
        transaction.handle_teardown_ack(
            canonical_ack,
            canonical_request.writer_id,
        )
        is None
    )

    conflicting = dataclasses.replace(
        canonical_ack,
        teardown_generation=_different_bytes(canonical_ack.teardown_generation),
    )
    with pytest.raises(PackedRequestTransactionError, match="conflicting teardown"):
        transaction.handle_teardown_ack(
            conflicting,
            canonical_request.writer_id,
        )
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_quarantine_after_main_commit_retains_only_unconsumed_auxiliary_row() -> None:
    """Post-commit failure retains metadata without revisiting retired KV pins."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    _complete_request_transfers(fixture, transaction, publication)
    receipt = None
    for request in transaction.begin_teardown():
        candidate = transaction.handle_teardown_ack(
            _ack(request),
            request.writer_id,
        )
        if candidate is not None:
            receipt = candidate
    assert receipt is not None
    transaction.commit_on_scheduler_thread(receipt)

    transaction.quarantine("scheduler metadata consumption abandoned")

    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.QUARANTINED
    )
    with pytest.raises(DecodeAllocationLeaseError, match="not registered"):
        fixture.allocation_authority.snapshot(fixture.allocation_lease)


def test_prepare_mutation_and_quarantine_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quarantine cannot cross a PREPARE transaction mutation in flight."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    transaction.publish()
    plan = fixture.plans[0]
    writer_id = plan.spec.writers[0]
    message = PackedPrepare(
        key=plan.key,
        writer_id=writer_id,
        spec=plan.spec,
        digest=plan.spec.build().digest,
    )
    entered = threading.Event()
    release = threading.Event()
    quarantine_started = threading.Event()
    quarantine_completed = threading.Event()
    errors: list[str] = []
    original_handle_prepare = fixture.protocol.handle_prepare

    def blocking_handle_prepare(
        candidate: PackedPrepare,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedReady, ...]:
        entered.set()
        assert release.wait(timeout=5)
        return original_handle_prepare(candidate, authenticated_writer_id)

    def run_prepare() -> None:
        try:
            transaction.handle_prepare(message, writer_id)
        except Exception:
            errors.append(traceback.format_exc())

    def run_quarantine() -> None:
        quarantine_started.set()
        try:
            transaction.quarantine("concurrent quarantine")
        except Exception:
            errors.append(traceback.format_exc())
        finally:
            quarantine_completed.set()

    monkeypatch.setattr(
        fixture.protocol,
        "handle_prepare",
        blocking_handle_prepare,
    )
    prepare_thread = threading.Thread(target=run_prepare)
    prepare_thread.start()
    assert entered.wait(timeout=5)
    quarantine_thread = threading.Thread(target=run_quarantine)
    quarantine_thread.start()
    assert quarantine_started.wait(timeout=5)
    assert not quarantine_completed.wait(timeout=0.05)
    release.set()
    prepare_thread.join(timeout=5)
    quarantine_thread.join(timeout=5)

    assert not prepare_thread.is_alive()
    assert not quarantine_thread.is_alive()
    assert errors == []
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_writer_outcome_mutation_and_quarantine_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quarantine cannot cross a writer-outcome mutation in flight."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    transaction.publish()
    plan = fixture.plans[0]
    digest, lease_id = _prepare_chunk(transaction, plan)
    writer_id, policy = plan.visibility_policies[0]
    message = PackedWriterOutcome(
        key=plan.key,
        writer_id=writer_id,
        digest=digest,
        lease_id=lease_id,
        status=PackedWriterOutcomeStatus.DONE,
        visibility=PackedWriterVisibilityEvidence(
            policy_digest=policy.digest,
            transport_path=policy.transport_path,
            lane_identifier=policy.lane_identifier,
            completion_mechanism=policy.completion_mechanism,
            writer_action=PackedWriterVisibilityAction.CUDA_EVENT_RECORDED,
        ),
        reason=None,
    )
    entered = threading.Event()
    release = threading.Event()
    quarantine_started = threading.Event()
    quarantine_completed = threading.Event()
    errors: list[str] = []
    original_handle_outcome = fixture.coordinator.handle_writer_outcome

    def blocking_handle_outcome(
        candidate: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        entered.set()
        assert release.wait(timeout=5)
        return original_handle_outcome(candidate, authenticated_writer_id)

    def run_outcome() -> None:
        try:
            transaction.handle_writer_outcome(message, writer_id)
        except Exception:
            errors.append(traceback.format_exc())

    def run_quarantine() -> None:
        quarantine_started.set()
        try:
            transaction.quarantine("concurrent quarantine")
        except Exception:
            errors.append(traceback.format_exc())
        finally:
            quarantine_completed.set()

    monkeypatch.setattr(
        fixture.coordinator,
        "handle_writer_outcome",
        blocking_handle_outcome,
    )
    outcome_thread = threading.Thread(target=run_outcome)
    outcome_thread.start()
    assert entered.wait(timeout=5)
    quarantine_thread = threading.Thread(target=run_quarantine)
    quarantine_thread.start()
    assert quarantine_started.wait(timeout=5)
    assert not quarantine_completed.wait(timeout=0.05)
    release.set()
    outcome_thread.join(timeout=5)
    quarantine_thread.join(timeout=5)

    assert not outcome_thread.is_alive()
    assert not quarantine_thread.is_alive()
    assert errors == []
    assert transaction.state is PackedRequestTransactionState.QUARANTINED


def test_quarantine_attempts_main_authority_after_auxiliary_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken auxiliary callback cannot skip main-allocation quarantine."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    transaction.publish()

    def fail_auxiliary_quarantine(
        reservation: object,
        owner: object,
    ) -> None:
        del reservation, owner
        raise LookupError("injected auxiliary quarantine fault")

    monkeypatch.setattr(
        fixture.auxiliary_allocator,
        "quarantine_packed_auxiliary_slot",
        fail_auxiliary_quarantine,
    )
    transaction.quarantine("forced transaction ambiguity")

    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert (
        fixture.auxiliary_authority.snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.QUARANTINED
    )
    assert (
        fixture.allocation_authority.snapshot(fixture.allocation_lease).state
        is DecodeAllocationLeaseState.QUARANTINED
    )
    assert "LookupError: injected auxiliary quarantine fault" in caplog.text


@pytest.mark.parametrize(
    ("fault", "match"),
    (
        ("unavailable", "snapshot is unavailable"),
        ("stale", "differs from its exact plan"),
    ),
)
def test_consumption_snapshot_fault_quarantines_the_retained_row(
    fault: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler consumption fails closed when retained-row proof is lost."""

    fixture = _fixture()
    transaction = _transaction(fixture)
    publication = transaction.publish()
    _complete_request_transfers(fixture, transaction, publication)
    receipt = None
    for request in transaction.begin_teardown():
        candidate = transaction.handle_teardown_ack(
            _ack(request),
            request.writer_id,
        )
        if candidate is not None:
            receipt = candidate
    assert receipt is not None
    transaction.commit_on_scheduler_thread(receipt)
    original_snapshot = fixture.auxiliary_authority.snapshot
    committed_snapshot = original_snapshot(fixture.auxiliary_lease)

    def faulty_snapshot(
        lease: PackedAuxiliaryAllocationLease,
    ) -> PackedAuxiliaryAllocationLeaseSnapshot:
        assert lease is fixture.auxiliary_lease
        if fault == "unavailable":
            raise LookupError("injected metadata snapshot fault")
        return dataclasses.replace(
            committed_snapshot,
            metadata_buffer_index=committed_snapshot.metadata_buffer_index + 1,
        )

    monkeypatch.setattr(
        fixture.auxiliary_authority,
        "snapshot",
        faulty_snapshot,
    )
    with pytest.raises(PackedRequestTransactionError, match=match):
        transaction.complete_auxiliary_consumption_on_scheduler_thread(
            fixture.consumer_authority
        )

    assert transaction.state is PackedRequestTransactionState.QUARANTINED
    assert (
        original_snapshot(fixture.auxiliary_lease).state
        is PackedAuxiliaryAllocationState.QUARANTINED
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
