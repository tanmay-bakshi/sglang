import dataclasses
import uuid

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
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
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl import packed_runtime as runtime_module
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
    PackedNixlRuntimeArtifactCohort,
    PackedNixlRuntimeArtifactIdentity,
    PackedNixlRuntimeRoot,
    PackedPeerIdentity,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedRequestTransactionState,
)


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

    def __init__(self) -> None:
        """Initialize canonical TP2 source-rank-zero state."""

        kv_args = KVArgs()
        kv_args.gpu_id = 0
        kv_args.aux_data_ptrs = [0x1000]
        kv_args.aux_item_lens = [8]
        self.agent = _FakeAgent("prefill-agent")
        self.agent_metadata = b"metadata"
        self.attn_cp_rank = 0
        self.attn_tp_rank = 0
        self.attn_tp_size = 2
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

    def __init__(
        self,
        key: PackedRequestKey,
        request_owner: object,
        ready: PackedReady,
        teardown: PackedRequestTeardown,
    ) -> None:
        """Initialize a published fake transaction.

        :param key: Exact request key.
        :param request_owner: Retained scheduler request.
        :param ready: READY emitted after PREPARE.
        :param teardown: Teardown emitted after scatter.
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

    def snapshot(self) -> _DecodeLifecycleSnapshot:
        """Return current actor-facing transaction state.

        :returns: Current request and chunk state.
        """

        return _DecodeLifecycleSnapshot(self.key, (self.chunk_state,))

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
        self.chunk_state = PackedProtocolState.RELEASED
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


def test_prefill_capability_binds_registration_runtime_and_topology() -> None:
    """Build a TP2-to-TP1 capability only for the accepted runtime digest."""

    manager = _FakeManager()
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
    assert capability.route.topology.source_tp_size == 2
    assert capability.route.topology.destination_tp_size == 1
    assert capability.request_generation == b"r" * 16


def test_prefill_ready_outcomes_teardown_releases_handles_and_acks() -> None:
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


def test_decode_prepare_scatter_teardown_and_commit_dispatch() -> None:
    """Drive the production actor branches from PREPARE through commit."""

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
    )
    sent_messages: list[object] = []
    record = runtime_module._DecodeRequestRecord(
        transaction=transaction,
        auxiliary_allocation=_AuxiliaryAllocation(),
        chunk_keys=(chunk_key,),
        routes={writer: PackedControlSender(writer, sent_messages.append)},
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
