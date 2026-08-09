import dataclasses
import logging
import uuid
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationComponentClaim,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedLease,
    PackedPrepare,
    PackedProtocolState,
    PackedReady,
    PackedRequestKey,
    PackedRequestTeardown,
    PackedRequestTeardownAck,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityEvidence,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingWriterId,
)
from sglang.srt.disaggregation.nixl import packed_runtime as runtime_module
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PackedControlSender,
    PackedDecodeControlSender,
    PackedDecodeRuntime,
    PackedPrefillRuntime,
    PackedPrefillSubmission,
    PackedRegistrationAdvertisement,
    build_same_host_visibility_policy,
    decode_packed_control_frames,
    encode_packed_control_frames,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    PackedDestinationRegistration,
    PackedNixlRuntimeArtifactCohort,
    PackedNixlRuntimeArtifactIdentity,
    PackedNixlRuntimeRoot,
    PackedPeerIdentity,
    build_prefill_chunk,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedRequestTransactionState,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class _FakeAgent:
    """Record exact source-handle releases."""

    name: str
    released_handles: list[object]

    def __init__(self, name: str) -> None:
        """Initialize one fake agent.

        :param name: Stable agent name.
        """

        self.name = name
        self.released_handles = []

    def release_xfer_handle(self, handle: object) -> None:
        """Record one terminal handle release.

        :param handle: Exact retained handle.
        """

        self.released_handles.append(handle)


class _FakeManager:
    """Provide the narrow manager surface used by actor-only tests."""

    agent: _FakeAgent
    agent_metadata: bytes
    attn_cp_rank: int
    attn_tp_rank: int
    attn_tp_size: int
    kv_args: KVArgs
    pp_rank: int
    process_generation: str
    transfer_source_rank: int
    failures: list[tuple[int, str]]
    statuses: list[tuple[int, int]]

    def __init__(self, attn_tp_size: int = 2) -> None:
        """Initialize canonical source-rank-zero state.

        :param attn_tp_size: Supported source attention TP width.
        """

        kv_args = KVArgs()
        kv_args.gpu_id = 0
        kv_args.aux_data_ptrs = [0x1000]
        kv_args.aux_item_lens = [8]
        self.agent = _FakeAgent("prefill-agent")
        self.agent_metadata = b"metadata"
        self.attn_cp_rank = 0
        self.attn_tp_rank = 0
        self.attn_tp_size = attn_tp_size
        self.kv_args = kv_args
        self.pp_rank = 0
        self.process_generation = str(uuid.uuid4())
        self.transfer_source_rank = 0
        self.failures = []
        self.statuses = []

    def _post_transfer_when_ready(self, handle: object, context: str) -> object:
        """Return the exact fake handle.

        :param handle: Exact fake handle.
        :param context: Diagnostic operation label.
        :returns: The supplied handle.
        """

        del context
        return handle

    def record_failure(self, room: int, reason: str) -> None:
        """Record one actor failure.

        :param room: Request room.
        :param reason: Stable failure reason.
        """

        self.failures.append((room, reason))

    def update_status(self, room: int, status: int) -> None:
        """Record one actor status transition.

        :param room: Request room.
        :param status: Transfer status.
        """

        self.statuses.append((room, status))


class _ReadyCoordinator:
    """Return one exact sentinel for authenticated READY dispatch."""

    transfer: object

    def __init__(self, transfer: object) -> None:
        """Initialize the coordinator.

        :param transfer: Sentinel source transfer.
        """

        self.transfer = transfer

    def handle_ready(self, message: PackedReady, peer: PackedPeerIdentity) -> object:
        """Return the retained sentinel.

        :param message: Valid READY payload.
        :param peer: Authenticated decoder process.
        :returns: Retained sentinel transfer.
        """

        del message, peer
        return self.transfer


@dataclasses.dataclass(frozen=True)
class _DecodeSnapshot:
    """Minimal transaction snapshot used by the runtime registry."""

    key: PackedRequestKey


class _DecodeTransaction:
    """Exercise decode outcome, acknowledgement, commit, and consumption calls."""

    request_owner: object
    auxiliary_allocation: object
    key: PackedRequestKey
    auxiliary_messages: list[PackedAuxiliaryOutcome]
    acknowledgements: list[PackedRequestTeardownAck]
    committed_receipts: list[object]
    consumers: list[object]
    state: PackedRequestTransactionState

    def __init__(
        self,
        key: PackedRequestKey,
        request_owner: object,
        auxiliary_allocation: object,
    ) -> None:
        """Initialize one fake transaction.

        :param key: Exact request identity.
        :param request_owner: Retained decode request owner.
        :param auxiliary_allocation: Adopted metadata row adapter.
        """

        self.key = key
        self.request_owner = request_owner
        self.auxiliary_allocation = auxiliary_allocation
        self.auxiliary_messages = []
        self.acknowledgements = []
        self.committed_receipts = []
        self.consumers = []
        self.state = PackedRequestTransactionState.PUBLISHED

    def snapshot(self) -> _DecodeSnapshot:
        """Return the exact request key.

        :returns: Minimal transaction snapshot.
        """

        return _DecodeSnapshot(self.key)

    def handle_auxiliary_outcome(
        self,
        message: PackedAuxiliaryOutcome,
        writer_id: StagingWriterId,
    ) -> bool:
        """Record one authenticated auxiliary outcome.

        :param message: Auxiliary outcome.
        :param writer_id: Authenticated writer.
        :returns: Whether the outcome was newly recorded.
        """

        assert message.writer_id == writer_id
        self.auxiliary_messages.append(message)
        return True

    def handle_teardown_ack(
        self,
        message: PackedRequestTeardownAck,
        writer_id: StagingWriterId,
    ) -> object:
        """Record one acknowledgement and issue a commit receipt.

        :param message: Teardown acknowledgement.
        :param writer_id: Authenticated writer.
        :returns: Opaque commit receipt.
        """

        assert message.writer_id == writer_id
        self.acknowledgements.append(message)
        self.state = PackedRequestTransactionState.COMMIT_READY
        return object()

    def commit_on_scheduler_thread(self, receipt: object) -> object:
        """Consume one actor-stored commit receipt.

        :param receipt: Opaque receipt.
        :returns: Exact request owner.
        """

        self.committed_receipts.append(receipt)
        self.state = PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
        return self.request_owner

    def complete_auxiliary_consumption_on_scheduler_thread(
        self,
        consumer: object,
    ) -> None:
        """Record consumption and release the fake metadata row.

        :param consumer: Exact scheduler consumer authority.
        """

        self.consumers.append(consumer)
        self.auxiliary_allocation.released = True


@dataclasses.dataclass(frozen=True)
class _DecodeLifecycleSnapshot:
    """Expose the actor-facing portion of one lifecycle snapshot."""

    key: PackedRequestKey
    chunk_states: tuple[PackedProtocolState, ...]
    scatter_terminal: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Scatter:
    """Carry the opaque scatter inputs consumed by the decode actor."""

    work: object
    proofs: tuple[object, ...]


class _Event:
    """Provide controllable asynchronous scatter completion."""

    complete: bool

    def __init__(self) -> None:
        """Initialize an incomplete event."""

        self.complete = False

    def query(self) -> bool:
        """Return the configured terminal state.

        :returns: Whether fake scatter work is complete.
        """

        return self.complete


@dataclasses.dataclass(frozen=True)
class _ScatterSubmission:
    """Retain the fake event returned by scatter submission."""

    event: _Event
    duration_ms: float = 7.0

    def elapsed_milliseconds(self) -> float:
        """Return deterministic terminal scatter device time.

        :returns: Fake CUDA duration in milliseconds.
        """

        assert self.event.complete
        return self.duration_ms


class _CopyExecutor:
    """Record one actor-owned scatter submission."""

    event: _Event
    submissions: list[tuple[object, tuple[object, ...]]]

    def __init__(self, event: _Event) -> None:
        """Initialize the executor.

        :param event: Completion event returned for each submission.
        """

        self.event = event
        self.submissions = []

    def scatter(
        self,
        work: object,
        proofs: tuple[object, ...],
    ) -> _ScatterSubmission:
        """Record and return one asynchronous scatter.

        :param work: Opaque protocol scatter work.
        :param proofs: Opaque visibility proofs.
        :returns: Fake asynchronous submission.
        """

        self.submissions.append((work, proofs))
        return _ScatterSubmission(self.event)


@dataclasses.dataclass(frozen=True)
class _Arena:
    """Expose the copy executor required by decode progress."""

    copy_executor: _CopyExecutor


class _DecodeLifecycleTransaction:
    """Exercise every decode actor dispatch and progress boundary."""

    key: PackedRequestKey
    request_owner: object
    ready: PackedReady
    teardown: PackedRequestTeardown
    state: PackedRequestTransactionState
    chunk_state: PackedProtocolState
    prepare_messages: list[tuple[PackedPrepare, StagingWriterId]]
    writer_outcomes: list[tuple[PackedWriterOutcome, StagingWriterId]]
    completed_scatters: list[_Scatter]
    acknowledgements: list[tuple[PackedRequestTeardownAck, StagingWriterId]]
    defer_scatter_completion_until_auxiliary: bool
    scatter_complete: bool

    def __init__(
        self,
        key: PackedRequestKey,
        request_owner: object,
        ready: PackedReady,
        teardown: PackedRequestTeardown,
        defer_scatter_completion_until_auxiliary: bool = False,
    ) -> None:
        """Initialize a published fake transaction.

        :param key: Exact request key.
        :param request_owner: Retained scheduler request.
        :param ready: READY emitted after PREPARE.
        :param teardown: Teardown emitted after scatter.
        :param defer_scatter_completion_until_auxiliary: Keep request state before
            scatter completion until the delayed auxiliary outcome arrives.
        """

        self.key = key
        self.request_owner = request_owner
        self.ready = ready
        self.teardown = teardown
        self.state = PackedRequestTransactionState.PUBLISHED
        self.chunk_state = PackedProtocolState.COLLECTING
        self.prepare_messages = []
        self.writer_outcomes = []
        self.completed_scatters = []
        self.acknowledgements = []
        self.defer_scatter_completion_until_auxiliary = (
            defer_scatter_completion_until_auxiliary
        )
        self.scatter_complete = False

    def snapshot(self) -> _DecodeLifecycleSnapshot:
        """Return current actor-facing transaction state.

        :returns: Current request and chunk state.
        """

        return _DecodeLifecycleSnapshot(
            self.key,
            (self.chunk_state,),
            (0,) if self.scatter_complete else (),
        )

    def handle_prepare(
        self,
        message: PackedPrepare,
        writer_id: StagingWriterId,
    ) -> tuple[PackedReady, ...]:
        """Accept PREPARE and produce READY.

        :param message: Authenticated writer PREPARE.
        :param writer_id: Authenticated writer identity.
        :returns: One READY message.
        """

        self.prepare_messages.append((message, writer_id))
        self.chunk_state = PackedProtocolState.READY
        self.state = PackedRequestTransactionState.SUBMITTED
        return (self.ready,)

    def handle_writer_outcome(
        self,
        message: PackedWriterOutcome,
        writer_id: StagingWriterId,
    ) -> bool:
        """Accept the main OUTCOME and make scatter eligible.

        :param message: Authenticated terminal outcome.
        :param writer_id: Authenticated writer identity.
        :returns: Whether the outcome was newly accepted.
        """

        self.writer_outcomes.append((message, writer_id))
        self.chunk_state = PackedProtocolState.SCATTER_READY
        self.state = PackedRequestTransactionState.WRITERS_COMPLETED
        return True

    def begin_scatter(self, key: PackedChunkKey) -> _Scatter:
        """Hand one scatter to the actor.

        :param key: Exact chunk key.
        :returns: Opaque fake scatter.
        """

        assert key == self.ready.key
        self.chunk_state = PackedProtocolState.SCATTERING
        return _Scatter(object(), ())

    def complete_scatter(self, scatter: _Scatter) -> None:
        """Record terminal scatter completion.

        :param scatter: Exact actor-owned scatter.
        """

        self.completed_scatters.append(scatter)
        self.scatter_complete = True
        self.chunk_state = PackedProtocolState.RELEASED
        if not self.defer_scatter_completion_until_auxiliary:
            self.state = PackedRequestTransactionState.SCATTER_COMPLETED

    def complete_auxiliary(self) -> None:
        """Advance request completion after a deliberately delayed outcome."""

        assert self.defer_scatter_completion_until_auxiliary
        assert self.scatter_complete
        assert self.state is PackedRequestTransactionState.WRITERS_COMPLETED
        self.state = PackedRequestTransactionState.SCATTER_COMPLETED

    def begin_teardown(self) -> tuple[PackedRequestTeardown, ...]:
        """Produce the request teardown.

        :returns: One writer teardown.
        """

        self.state = PackedRequestTransactionState.TEARDOWN_WAITING
        return (self.teardown,)

    def handle_teardown_ack(
        self,
        message: PackedRequestTeardownAck,
        writer_id: StagingWriterId,
    ) -> object:
        """Accept terminal teardown acknowledgement.

        :param message: Exact acknowledgement.
        :param writer_id: Authenticated writer identity.
        :returns: Opaque scheduler commit receipt.
        """

        self.acknowledgements.append((message, writer_id))
        self.state = PackedRequestTransactionState.COMMIT_READY
        return object()

    def commit_on_scheduler_thread(self, receipt: object) -> object:
        """Consume the opaque receipt and return the request owner.

        :param receipt: Actor-retained commit receipt.
        :returns: Retained scheduler request.
        """

        del receipt
        self.state = PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
        return self.request_owner


@dataclasses.dataclass
class _AuxiliaryAllocation:
    """Expose mutable release state for consumption tests."""

    released: bool = False


@dataclasses.dataclass
class _PackedRequestOwner:
    """Minimal request owner accepted by the request-slot allocator."""

    req_pool_idx: int | None = None
    inflight_middle_chunks: int = 0
    kv_committed_len: int = 0


@dataclasses.dataclass(frozen=True)
class _PackedTransferProjection:
    """Carry decoder-published page windows into source projection."""

    dst_kv_indices: tuple[int, ...]
    dst_state_indices: tuple[tuple[int, ...], ...]


class _PackedMetadataAllocator:
    """Record scheduler metadata rows released by terminal transactions."""

    released: list[int]

    def __init__(self) -> None:
        """Initialize an empty release record."""

        self.released = []

    def free(self, index: int) -> None:
        """Record one returned metadata row.

        :param index: Exact metadata row index.
        """

        self.released.append(index)


class _PackedLeaseAllocator:
    """Allocate deterministic CPU-only packed staging leases."""

    next_lease_id: int

    def __init__(self) -> None:
        """Initialize the first fake lease identity."""

        self.next_lease_id = 1

    def allocate(self, length_bytes: int) -> PackedLease:
        """Allocate one address-distinct fake lease.

        :param length_bytes: Required canonical packed byte count.
        :returns: Exact fake staging lease.
        """

        lease_id = self.next_lease_id
        self.next_lease_id += 1
        return PackedLease(
            lease_id=lease_id,
            base_address=0x800000 + lease_id * 0x10000,
            length_bytes=length_bytes,
        )

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Accept protocol quarantine without reusing the fake lease.

        :param lease: Ambiguous staging lease.
        :param reason: Stable quarantine reason.
        """

        del lease, reason

    def release(self, lease: PackedLease) -> None:
        """Accept terminal fake-lease release.

        :param lease: Terminal staging lease.
        """

        del lease


@dataclasses.dataclass(frozen=True)
class _PackedDecodeArena:
    """Expose the real protocol through the decode runtime arena surface."""

    protocol: PackedDecodeProtocol


class _PackedDecodeManager:
    """Provide decoder registration and lifecycle identity to the runtime."""

    attn_tp_rank: int
    attn_tp_size: int
    kv_args: KVArgs
    process_generation: str

    def __init__(
        self,
        kv_args: KVArgs,
        *,
        attn_tp_size: int = 1,
        attn_tp_rank: int = 0,
    ) -> None:
        """Initialize deterministic decoder runtime metadata.

        :param kv_args: Decode-local registered cache geometry.
        :param attn_tp_size: Destination attention TP width.
        :param attn_tp_rank: Destination attention TP rank.
        """

        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.kv_args = kv_args
        self.process_generation = str(uuid.uuid4())


def _packed_kv_args(item_length: int) -> KVArgs:
    """Build one main-KV plus SWA registration.

    :param item_length: Bytes occupied by one registered page.
    :returns: Complete packed runtime registration projection.
    """

    page_size = 64
    kv_args = KVArgs()
    kv_args.gpu_id = 0
    kv_args.page_size = page_size
    kv_args.kv_data_ptrs = [0x100000]
    kv_args.kv_data_lens = [item_length * 4096]
    kv_args.kv_item_lens = [item_length]
    kv_args.kv_layer_ids = [0]
    kv_args.state_types = [StateType.SWA]
    kv_args.state_data_ptrs = [[0x200000]]
    kv_args.state_data_lens = [[item_length * 4096]]
    kv_args.state_item_lens = [[item_length]]
    kv_args.state_layer_ids = [[0]]
    kv_args.aux_data_ptrs = [0x300000]
    kv_args.aux_item_lens = [128]
    return kv_args


def _cache_hit_allocation(
    manager: _PackedDecodeManager,
    source_tp_size: int = 2,
) -> tuple[
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLease,
    _PackedRequestOwner,
]:
    """Pin one 64-token FULL/SWA migration beginning at token 448.

    :param manager: Exact lifecycle authority retained by the lease.
    :param source_tp_size: Supported packed source TP width.
    :returns: Authority, exact allocation lease, and request owner.
    """

    request_pool = ReqToTokenPool(
        size=2,
        max_context_len=512,
        device="cpu",
        enable_memory_saver=False,
    )
    owner = _PackedRequestOwner()
    slots = request_pool.alloc([owner])
    assert slots is not None
    assert owner.req_pool_idx is not None
    allocator_kwargs = {
        "size": 1024,
        "page_size": 64,
        "dtype": torch.float16,
        "device": "cpu",
        "kvcache": object(),
        "need_sort": False,
    }
    full_allocator = PagedTokenToKVPoolAllocator(**allocator_kwargs)
    swa_allocator = PagedTokenToKVPoolAllocator(**allocator_kwargs)
    full_indices = full_allocator.alloc(64)
    swa_indices = swa_allocator.alloc(64)
    assert full_indices is not None
    assert swa_indices is not None
    authority = DecodeAllocationLeaseAuthority(manager)
    request_generation = int(request_pool.req_generation[owner.req_pool_idx].item())
    lease = authority.acquire(
        request_pool=request_pool,
        request_slot=owner.req_pool_idx,
        expected_request_generation=request_generation,
        writer_manifest=DecodeWriterManifest.for_tensor_parallel(
            source_tp_size,
            manager.attn_tp_size,
            manager.attn_tp_rank,
        ),
        component_claims=(
            DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.FULL,
                logical_start=448,
                logical_length=64,
                allocator=full_allocator,
                indices=full_indices,
            ),
            DecodeAllocationComponentClaim(
                component=DecodeAllocationComponent.SWA,
                logical_start=448,
                logical_length=64,
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
    return authority, lease, owner


def _runtime_artifacts() -> PackedNixlRuntimeArtifactCohort:
    """Build a path-independent valid runtime cohort.

    :returns: Complete fake NIXL/UCX artifact cohort.
    """

    return PackedNixlRuntimeArtifactCohort(
        roots=(
            PackedNixlRuntimeRoot(root_id="nixl", path="/tmp"),
            PackedNixlRuntimeRoot(root_id="ucx", path="/tmp"),
        ),
        artifacts=(
            PackedNixlRuntimeArtifactIdentity(
                component="libnixl",
                root_id="nixl",
                relative_path="libnixl.so",
                build_id="aa",
                version="1.3.2",
            ),
            PackedNixlRuntimeArtifactIdentity(
                component="libucp",
                root_id="ucx",
                relative_path="libucp.so",
                build_id="bb",
                version="1.21.0",
            ),
            PackedNixlRuntimeArtifactIdentity(
                component="libuct_cuda",
                root_id="ucx",
                relative_path="libuct_cuda.so",
                build_id="cc",
                version="1.21.0",
            ),
            PackedNixlRuntimeArtifactIdentity(
                component="ucx-plugin",
                root_id="nixl",
                relative_path="libplugin_UCX.so",
                build_id="dd",
                version="0.1.0",
            ),
        ),
    )


def _writer() -> StagingWriterId:
    """Return canonical TP0 writer identity.

    :returns: TP0/PP0/CP0 writer.
    """

    return StagingWriterId(
        transfer_source_rank=0,
        source_attn_tp_rank=0,
        source_pp_rank=0,
        source_cp_rank=0,
    )


def _plan(peer: PackedPeerIdentity) -> PackedAuxiliaryPlan:
    """Build one valid decoder-authored auxiliary plan.

    :param peer: Target decoder process.
    :returns: Valid request plan.
    """

    return PackedAuxiliaryPlan(
        key=PackedRequestKey(room_id=41, request_generation=b"r" * 16),
        request_slot_generation=7,
        metadata_buffer_index=3,
        metadata_slot_generation=b"m" * 16,
        destination_segments=(
            PackedAuxiliaryDestinationSegment(address=0x2000, item_length=8),
        ),
        canonical_writer_id=_writer(),
        destination_process_generation=peer.agent_generation,
        native_route_digest=b"n" * 32,
        runtime_cohort_digest=b"c" * 32,
    )


def _unvalidated_submission(
    plan: PackedAuxiliaryPlan,
    control: PackedDecodeControlSender,
) -> PackedPrefillSubmission:
    """Construct the control-only projection used by actor dispatch tests.

    :param plan: Decoder-authored plan.
    :param control: Authenticated decoder route.
    :returns: Submission whose GPU-only fields are intentionally absent.
    """

    submission = object.__new__(PackedPrefillSubmission)
    object.__setattr__(submission, "plan", plan)
    object.__setattr__(submission, "control", control)
    return submission


def _unvalidated_prepare(key: PackedChunkKey, writer: StagingWriterId) -> PackedPrepare:
    """Construct the actor-dispatch projection of PREPARE.

    Protocol-level PREPARE validation is covered by the lifecycle suite. This
    actor test needs only the already-validated fields consumed by dispatch.

    :param key: Exact chunk key.
    :param writer: Claimed writer identity.
    :returns: PREPARE projection accepted by actor dispatch.
    """

    prepare = object.__new__(PackedPrepare)
    object.__setattr__(prepare, "key", key)
    object.__setattr__(prepare, "writer_id", writer)
    object.__setattr__(prepare, "spec", None)
    object.__setattr__(prepare, "digest", b"d" * 32)
    return prepare


def test_main_transfer_initialization_failure_logs_native_traceback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve native diagnostics without changing pre-submit recovery."""

    native_error = RuntimeError("NIXL_ERR_BACKEND: injected initialization failure")
    source_descriptors = object()
    destination_descriptors = object()
    agent = Mock()
    agent.get_xfer_descs.side_effect = (
        source_descriptors,
        destination_descriptors,
    )
    agent.initialize_xfer.side_effect = native_error
    manager = Mock()
    manager.agent = agent
    manager.kv_args.gpu_id = 0
    runtime = object.__new__(PackedPrefillRuntime)
    runtime._manager = manager
    outcome = Mock(reason="packed main transfer initialization failed")
    lane = Mock(data_ptr=0x100000)
    lane.abort_before_submit.return_value = outcome
    executor = Mock()
    producer_stream = object()
    record = Mock()
    record.submission.producer_stream = producer_stream
    transfer = Mock(
        destination_address=0x200000,
        length_bytes=4096,
    )
    transfer.destination.route.destination_gpu_id = 1
    remote_handle = object()
    record.submission.control.remote_handle = remote_handle
    monkeypatch.setattr(runtime, "_acquire_lane", lambda _transfer: lane)
    monkeypatch.setattr(runtime, "_source_copy_executor", lambda: executor)
    caplog.set_level(logging.ERROR, logger=runtime_module.__name__)

    with pytest.raises(runtime_module.PackedGatherError) as raised:
        runtime._post_main_transfer(record, transfer)

    assert raised.value.__cause__ is native_error
    assert str(raised.value.__cause__) == (
        "NIXL_ERR_BACKEND: injected initialization failure"
    )
    assert raised.value.outcome is outcome
    diagnostic = "\n".join(caplog.messages)
    assert "Packed main transfer initialization failed:\nTraceback" in diagnostic
    assert (
        "RuntimeError: NIXL_ERR_BACKEND: injected initialization failure" in diagnostic
    )
    assert agent.get_xfer_descs.call_count == 2
    assert [call.args[1] for call in agent.get_xfer_descs.call_args_list] == [
        "VRAM",
        "VRAM",
    ]
    agent.initialize_xfer.assert_called_once_with(
        "WRITE",
        source_descriptors,
        destination_descriptors,
        remote_handle,
        b"",
    )
    executor.gather.assert_called_once_with(
        transfer=transfer,
        source_lane=lane,
        producer_stream=producer_stream,
    )
    lane.abort_before_submit.assert_called_once_with(
        "packed main transfer initialization failed"
    )
    lane.arm_submission.assert_not_called()
    lane.mark_submission_ambiguous.assert_not_called()
    manager._post_transfer_when_ready.assert_not_called()


def test_control_envelope_round_trips_generation_bound_ready() -> None:
    """Round-trip a READY through the closed multipart envelope."""

    message = PackedReady(
        key=PackedChunkKey(
            room_id=41,
            chunk_id=0,
            request_generation=b"r" * 16,
        ),
        writer_id=_writer(),
        digest=b"d" * 32,
        visibility_policy_digest=b"v" * 32,
        lease_id=9,
        lease_base_address=0x400000,
        projection_offset=0,
        projection_length=4096,
    )
    generation = str(uuid.uuid4())

    frames = encode_packed_control_frames("decoder-agent", generation, message)

    assert decode_packed_control_frames(frames) == (
        "decoder-agent",
        generation,
        message,
    )


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_prefill_capability_binds_supported_source_runtime_and_topology(
    source_tp_size: int,
) -> None:
    """Bind every supported source width only for the accepted runtime digest."""

    manager = _FakeManager(source_tp_size)
    artifacts = _runtime_artifacts()
    policy = build_same_host_visibility_policy(artifacts)
    runtime = PackedPrefillRuntime(manager, artifacts, policy)
    peer = PackedPeerIdentity("decoder-agent", b"p" * 16)
    advertisement = PackedRegistrationAdvertisement(
        base_address=0x400000,
        total_size=64 * 1024 * 1024,
        arena_generation=b"a" * 16,
        visibility_policy_digest=policy.digest,
        runtime_cohort_digest=artifacts.digest,
        page_size=1,
    )

    capability = runtime.build_destination_capability(
        advertisement=advertisement,
        decode_peer=peer,
        destination_gpu_id=1,
        destination_tp_size=1,
        destination_tp_rank=0,
        request_generation=b"r" * 16,
    )

    assert capability.route.peer == peer
    assert capability.route.topology.source_tp_size == source_tp_size
    assert capability.route.topology.destination_tp_size == 1
    assert capability.request_generation == b"r" * 16


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_cache_hit_swa_window_matches_decode_runtime_canonical_layout(
    source_tp_size: int,
) -> None:
    """Project only migration-owned SWA pages into a runtime PREPARE."""

    artifacts = _runtime_artifacts()
    policy = build_same_host_visibility_policy(artifacts)
    decode_kv_args = _packed_kv_args(item_length=1024)
    decode_manager = _PackedDecodeManager(decode_kv_args)
    protocol = PackedDecodeProtocol(_PackedLeaseAllocator())
    decode_runtime = PackedDecodeRuntime(
        decode_manager,
        _PackedDecodeArena(protocol),
        artifacts,
        policy,
    )
    decode_runtime.attach_scheduler(_PackedMetadataAllocator(), object())
    allocation_authority, allocation_lease, owner = _cache_hit_allocation(
        decode_manager,
        source_tp_size,
    )
    allocation_snapshot = allocation_authority.snapshot(allocation_lease)
    transaction = decode_runtime.prepare_transaction(
        room_id=91,
        request_owner=owner,
        metadata_buffer_index=3,
        allocation_lease=allocation_lease,
        allocation_authority=allocation_authority,
        lifecycle_authority=decode_manager,
        source_tp_size=source_tp_size,
    )
    publication = transaction.publish()
    receipts = {
        receipt.component: receipt for receipt in allocation_snapshot.components
    }
    full_receipt = receipts[DecodeAllocationComponent.FULL]
    swa_receipt = receipts[DecodeAllocationComponent.SWA]
    assert len(full_receipt.physical_pages) == 1
    assert len(swa_receipt.physical_pages) == 1

    source_kv_args = _packed_kv_args(item_length=1024 // source_tp_size)
    source_manager = object.__new__(NixlKVManager)
    source_manager.kv_args = source_kv_args
    destination_swa_window = (
        *tuple(range(100, 107)),
        *swa_receipt.physical_pages,
    )
    components = source_manager._packed_source_components(
        np.asarray([7], dtype=np.int32),
        _PackedTransferProjection(
            dst_kv_indices=full_receipt.physical_pages,
            dst_state_indices=(destination_swa_window,),
        ),
        [tuple(range(200, 208))],
    )
    swa_component = StagingComponentId(state_index=0, state_type=StateType.SWA)
    assert tuple(components[0].destination_pages) == full_receipt.physical_pages
    assert components[1].component_id == swa_component
    assert tuple(components[1].source_pages) == (207,)
    assert tuple(components[1].destination_pages) == swa_receipt.physical_pages

    key = PackedChunkKey(
        room_id=publication.key.room_id,
        chunk_id=0,
        request_generation=publication.key.request_generation,
    )
    source_spec, _ = build_prefill_chunk(
        key=key,
        is_last=True,
        kv_args=source_kv_args,
        destination_registration=PackedDestinationRegistration(
            main_item_lens=tuple(decode_kv_args.kv_item_lens),
            main_layer_ids=tuple(decode_kv_args.kv_layer_ids),
            state_item_lens=tuple(
                tuple(item_lens) for item_lens in decode_kv_args.state_item_lens
            ),
            state_layer_ids=tuple(
                tuple(layer_ids) for layer_ids in decode_kv_args.state_layer_ids
            ),
            page_size=decode_kv_args.page_size,
        ),
        components=components,
        source_tp_size=source_tp_size,
        destination_tp_size=1,
        destination_tp_rank=0,
        writers=allocation_snapshot.writer_manifest.writers,
    )
    assert len(source_spec.writers) == source_tp_size
    assert tuple(writer.source_attn_tp_rank for writer in source_spec.writers) == tuple(
        range(source_tp_size)
    )
    assert source_spec.build() == publication.chunk_specs[0].build()
    source_layout = source_spec.build()
    ready_messages: list[PackedReady] = []
    for writer in allocation_snapshot.writer_manifest.writers:
        ready_messages.extend(
            transaction.handle_prepare(
                PackedPrepare(
                    key=key,
                    writer_id=writer,
                    spec=source_spec,
                    digest=source_layout.digest,
                ),
                writer,
            )
        )

    assert tuple(message.writer_id for message in ready_messages) == (
        allocation_snapshot.writer_manifest.writers
    )


def test_tp2_rank_one_decode_runtime_uses_destination_local_writer() -> None:
    """Bind FULL, SWA, and draft KV geometry to source rank one only."""

    artifacts = _runtime_artifacts()
    policy = build_same_host_visibility_policy(artifacts)
    decode_kv_args = _packed_kv_args(item_length=512)
    decode_kv_args.kv_data_ptrs = [0x100000, 0x110000]
    decode_kv_args.kv_data_lens = [512 * 4096, 512 * 4096]
    decode_kv_args.kv_item_lens = [512, 512]
    decode_kv_args.kv_layer_ids = [0, 0]
    decode_manager = _PackedDecodeManager(
        decode_kv_args,
        attn_tp_size=2,
        attn_tp_rank=1,
    )
    protocol = PackedDecodeProtocol(_PackedLeaseAllocator())
    decode_runtime = PackedDecodeRuntime(
        decode_manager,
        _PackedDecodeArena(protocol),
        artifacts,
        policy,
    )
    decode_runtime.attach_scheduler(_PackedMetadataAllocator(), object())
    allocation_authority, allocation_lease, owner = _cache_hit_allocation(
        decode_manager
    )

    transaction = decode_runtime.prepare_transaction(
        room_id=92,
        request_owner=owner,
        metadata_buffer_index=3,
        allocation_lease=allocation_lease,
        allocation_authority=allocation_authority,
        lifecycle_authority=decode_manager,
        source_tp_size=2,
    )
    publication = transaction.publish()
    spec = publication.chunk_specs[0]

    assert spec.topology.destination_tp_size == 2
    assert spec.topology.destination_tp_rank == 1
    assert spec.writers == (
        StagingWriterId(
            transfer_source_rank=1,
            source_attn_tp_rank=1,
            source_pp_rank=0,
            source_cp_rank=0,
        ),
    )
    assert spec.destination_components[0].layer_ids == (0, 0)
    assert publication.auxiliary_plan.canonical_writer_id == spec.writers[0]


def test_prefill_ready_outcomes_teardown_releases_handles_and_acks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Drive READY ownership through exact source handle teardown and ACK."""

    manager = _FakeManager()
    artifacts = _runtime_artifacts()
    policy = build_same_host_visibility_policy(artifacts)
    runtime = PackedPrefillRuntime(manager, artifacts, policy)
    peer = PackedPeerIdentity("decoder-agent", b"p" * 16)
    sent_messages: list[object] = []
    control = PackedDecodeControlSender(
        peer=peer,
        remote_handle=object(),
        send_message=sent_messages.append,
    )
    plan = _plan(peer)
    submission = _unvalidated_submission(plan, control)
    chunk_key = PackedChunkKey(
        room_id=plan.key.room_id,
        chunk_id=0,
        request_generation=plan.key.request_generation,
    )
    record = runtime_module._PrefillRequestRecord(
        submission=submission,
        writer_id=_writer(),
        chunk_key=chunk_key,
    )
    runtime._records[plan.key] = record
    source_transfer = object()
    runtime._ready = _ReadyCoordinator(source_transfer)
    ready = PackedReady(
        key=chunk_key,
        writer_id=_writer(),
        digest=b"d" * 32,
        visibility_policy_digest=policy.digest,
        lease_id=9,
        lease_base_address=0x400000,
        projection_offset=0,
        projection_length=4096,
    )

    runtime.handle_control(peer, ready)

    assert record.source_transfer is source_transfer
    visibility = PackedWriterVisibilityEvidence(
        policy_digest=policy.digest,
        transport_path=policy.transport_path,
        lane_identifier=policy.lane_identifier,
        completion_mechanism=policy.completion_mechanism,
        writer_action=policy.expected_writer_action,
        native_handle_generation=11,
        native_descriptor_digest=b"d" * 32,
        native_evidence_digest=b"e" * 32,
    )
    record.main_outcome = PackedWriterOutcome(
        key=chunk_key,
        writer_id=_writer(),
        digest=b"d" * 32,
        lease_id=9,
        status=PackedWriterOutcomeStatus.DONE,
        visibility=visibility,
    )
    record.auxiliary_outcome = PackedAuxiliaryOutcome(
        plan=plan,
        writer_id=_writer(),
        native_dram_handle_generation=12,
        descriptor_digest=b"a" * 32,
        evidence_digest=b"b" * 32,
    )
    main_handle = object()
    auxiliary_handle = object()
    record.main_handle = main_handle
    record.auxiliary_handle = auxiliary_handle
    record.outcomes_sent = True
    record.ready_wait_duration_ms = 11.0
    record.source_gather_copy_duration_ms = 5.0
    record.main_transport_duration_ms = 19.0
    with caplog.at_level(logging.INFO, logger=runtime_module.__name__):
        runtime._emit_transfer_stats(record)
    teardown = PackedRequestTeardown(
        key=plan.key,
        writer_id=_writer(),
        request_slot_generation=7,
        writer_manifest_digest=b"w" * 32,
        allocation_digest=b"l" * 32,
        teardown_generation=b"t" * 16,
        auxiliary_handle_generation=12,
    )

    runtime.handle_control(peer, teardown)

    assert manager.agent.released_handles == [main_handle, auxiliary_handle]
    assert sent_messages == [
        PackedRequestTeardownAck(
            key=teardown.key,
            writer_id=teardown.writer_id,
            request_slot_generation=teardown.request_slot_generation,
            writer_manifest_digest=teardown.writer_manifest_digest,
            allocation_digest=teardown.allocation_digest,
            teardown_generation=teardown.teardown_generation,
            auxiliary_handle_generation=teardown.auxiliary_handle_generation,
        )
    ]
    assert plan.key not in runtime._records
    assert (
        "PackedTransferStats(room=41, role=prefill, source_rank=0, "
        "ready_wait_duration=11.000ms, "
        "source_gather_copy_duration=5.000ms, "
        "main_transport_duration=19.000ms)"
    ) in caplog.messages


def test_decode_outcome_ack_commit_and_metadata_consumption() -> None:
    """Drive decoder control evidence through scheduler commit and row release."""

    manager = _FakeManager()
    peer = PackedPeerIdentity("decoder-agent", b"p" * 16)
    plan = _plan(peer)
    writer = _writer()
    auxiliary = _AuxiliaryAllocation()
    request_owner = object()
    transaction = _DecodeTransaction(plan.key, request_owner, auxiliary)
    record = runtime_module._DecodeRequestRecord(
        transaction=transaction,
        auxiliary_allocation=auxiliary,
        chunk_keys=(
            PackedChunkKey(
                room_id=plan.key.room_id,
                chunk_id=0,
                request_generation=plan.key.request_generation,
            ),
        ),
        upstream_wait_duration_ms=1.0,
        finalize_started_at=runtime_module.time.perf_counter(),
    )
    runtime = object.__new__(PackedDecodeRuntime)
    runtime._manager = manager
    runtime._records = {plan.key: record}
    runtime._records_by_room = {plan.key.room_id: plan.key}
    runtime._lock = runtime_module.threading.RLock()
    consumer = object()
    runtime._consumer_authority = consumer
    runtime._poll_scatters = lambda owned: None
    runtime._begin_teardown_if_ready = lambda owned: None
    outcome = PackedAuxiliaryOutcome(
        plan=plan,
        writer_id=writer,
        native_dram_handle_generation=13,
        descriptor_digest=b"a" * 32,
        evidence_digest=b"b" * 32,
    )
    acknowledgement = PackedRequestTeardownAck(
        key=plan.key,
        writer_id=writer,
        request_slot_generation=7,
        writer_manifest_digest=b"w" * 32,
        allocation_digest=b"l" * 32,
        teardown_generation=b"t" * 16,
        auxiliary_handle_generation=13,
    )

    runtime.handle_control(writer, outcome)
    runtime.handle_control(writer, acknowledgement)

    assert transaction.auxiliary_messages == [outcome]
    assert transaction.acknowledgements == [acknowledgement]
    assert runtime.poll(transaction) == KVPoll.Success
    assert len(transaction.committed_receipts) == 1

    runtime.complete_metadata_consumption(transaction)

    assert transaction.consumers == [consumer]
    assert auxiliary.released
    assert plan.key not in runtime._records


@pytest.mark.parametrize(
    "defer_auxiliary",
    (False, True),
    ids=("auxiliary-before-scatter", "scatter-before-auxiliary"),
)
def test_decode_prepare_scatter_teardown_and_commit_dispatch(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    defer_auxiliary: bool,
) -> None:
    """Drive both auxiliary/scatter orderings from PREPARE through commit."""

    manager = _FakeManager()
    artifacts = _runtime_artifacts()
    policy = build_same_host_visibility_policy(artifacts)
    writer = _writer()
    peer = PackedPeerIdentity("decoder-agent", b"p" * 16)
    plan = _plan(peer)
    chunk_key = PackedChunkKey(
        room_id=plan.key.room_id,
        chunk_id=0,
        request_generation=plan.key.request_generation,
    )
    ready = PackedReady(
        key=chunk_key,
        writer_id=writer,
        digest=b"d" * 32,
        visibility_policy_digest=policy.digest,
        lease_id=9,
        lease_base_address=0x400000,
        projection_offset=0,
        projection_length=4096,
    )
    teardown = PackedRequestTeardown(
        key=plan.key,
        writer_id=writer,
        request_slot_generation=7,
        writer_manifest_digest=b"w" * 32,
        allocation_digest=b"l" * 32,
        teardown_generation=b"t" * 16,
    )
    request_owner = object()
    transaction = _DecodeLifecycleTransaction(
        plan.key,
        request_owner,
        ready,
        teardown,
        defer_scatter_completion_until_auxiliary=defer_auxiliary,
    )
    sent_messages: list[object] = []
    record = runtime_module._DecodeRequestRecord(
        transaction=transaction,
        auxiliary_allocation=_AuxiliaryAllocation(),
        chunk_keys=(chunk_key,),
        routes={writer: PackedControlSender(writer, sent_messages.append)},
        pipeline_started_at=1.0,
    )
    event = _Event()
    executor = _CopyExecutor(event)
    runtime = object.__new__(PackedDecodeRuntime)
    runtime._manager = manager
    runtime._arena = _Arena(executor)
    runtime._records = {plan.key: record}
    runtime._records_by_room = {plan.key.room_id: plan.key}
    runtime._lock = runtime_module.threading.RLock()
    runtime._consumer_authority = object()
    prepare = _unvalidated_prepare(chunk_key, writer)
    timestamps = iter((1.031, 1.038, 1.043))
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: next(timestamps))
    caplog.set_level(logging.INFO, logger=runtime_module.__name__)

    runtime.handle_control(writer, prepare)

    assert transaction.prepare_messages == [(prepare, writer)]
    assert sent_messages == [ready]
    visibility = PackedWriterVisibilityEvidence(
        policy_digest=policy.digest,
        transport_path=policy.transport_path,
        lane_identifier=policy.lane_identifier,
        completion_mechanism=policy.completion_mechanism,
        writer_action=policy.expected_writer_action,
        native_handle_generation=11,
        native_descriptor_digest=b"d" * 32,
        native_evidence_digest=b"e" * 32,
    )
    outcome = PackedWriterOutcome(
        key=chunk_key,
        writer_id=writer,
        digest=ready.digest,
        lease_id=ready.lease_id,
        status=PackedWriterOutcomeStatus.DONE,
        visibility=visibility,
    )

    runtime.handle_control(writer, outcome)

    assert transaction.writer_outcomes == [(outcome, writer)]
    assert runtime.poll(transaction) == KVPoll.Transferring
    assert len(executor.submissions) == 1
    event.complete = True

    assert runtime.poll(transaction) == KVPoll.Transferring
    assert len(transaction.completed_scatters) == 1
    assert record.finalize_started_at == 1.038
    if defer_auxiliary:
        assert sent_messages == [ready]
        transaction.complete_auxiliary()
        assert runtime.poll(transaction) == KVPoll.Transferring
    assert sent_messages == [ready, teardown]
    acknowledgement = PackedRequestTeardownAck(
        key=teardown.key,
        writer_id=teardown.writer_id,
        request_slot_generation=teardown.request_slot_generation,
        writer_manifest_digest=teardown.writer_manifest_digest,
        allocation_digest=teardown.allocation_digest,
        teardown_generation=teardown.teardown_generation,
        auxiliary_handle_generation=teardown.auxiliary_handle_generation,
    )

    runtime.handle_control(writer, acknowledgement)

    assert transaction.acknowledgements == [(acknowledgement, writer)]
    assert runtime.poll(transaction) == KVPoll.Success
    assert (
        transaction.state
        is PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
    )
    assert runtime.poll(transaction) == KVPoll.Success
    assert manager.failures == []
    assert (
        "PackedTransferStats(room=41, role=decode, destination_rank=0, "
        "upstream_wait_duration=31.000ms, "
        "destination_scatter_copy_duration=7.000ms, "
        "finalize_duration=5.000ms, packed_pipeline_duration=43.000ms)"
    ) in caplog.messages
