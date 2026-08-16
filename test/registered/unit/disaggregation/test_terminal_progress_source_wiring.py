import dataclasses
import hashlib
import inspect
import logging

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.evidence import (
    parse_terminal_progress_timing_log_line,
)
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlMemberTiming,
    GroupedNixlTerminalResult,
    GroupedNixlTransferMember,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerObservation,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
    FrozenTerminalGatewayPublication,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceCancellationDisposition,
    PackedTerminalSourceMetric,
    PackedTerminalSourceQuarantineRetentionError,
    PackedTerminalSourceSubmission,
    PackedTerminalSourceWiring,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LOCAL_PRODUCER_ID = 10
_LOCAL_RECEIPT_PRODUCER_ID = 11
_DECODER_CONTROL_PRODUCER_ID = 12
_DECODER_RECEIPT_PRODUCER_ID = 13
_PUBLISHER_RECEIPT_PRODUCER_ID = 14


@dataclasses.dataclass(frozen=True, slots=True)
class _Projection(FrozenTerminalGatewayOutputProjection):
    """Minimal immutable output projection fixture.

    :ivar payload: Exact bytes represented by this projection.
    """

    payload: bytes

    @property
    def digest(self) -> bytes:
        """Return a stable fixture digest.

        :returns: SHA-256 over the exact payload.
        """

        return hashlib.sha256(self.payload).digest()


@dataclasses.dataclass(frozen=True, slots=True)
class _SubmissionPayload:
    """Opaque transport submission fixture.

    :ivar label: Stable test identity.
    """

    label: str


class _Metrics:
    """Non-gating source metric ledger fixture."""

    values: list[PackedTerminalSourceMetric]
    fail: bool

    def __init__(self, *, fail: bool = False) -> None:
        """Create an empty metric sink.

        :param fail: Whether every emission raises.
        """

        self.values = []
        self.fail = fail

    def emit(self, metric: PackedTerminalSourceMetric) -> None:
        """Record or reject one metric.

        :param metric: Exact source metric.
        """

        if self.fail:
            raise RuntimeError("synthetic metric failure")
        self.values.append(metric)


class _Publisher:
    """Canonical publisher submission ledger fixture."""

    publications: list[FrozenTerminalGatewayPublication]

    def __init__(self) -> None:
        """Create an empty publisher ledger."""

        self.publications = []

    def submit(self, publication: FrozenTerminalGatewayPublication) -> bool:
        """Record one exact publication.

        :param publication: Immutable gateway handoff.
        :returns: Whether the publication was newly accepted.
        """

        self.publications.append(publication)
        return True


class _Clock:
    """Strictly increasing deterministic nanosecond clock."""

    value: int

    def __init__(self) -> None:
        """Start at one nonzero timestamp."""

        self.value = 10_000

    def now_ns(self) -> int:
        """Advance and return time.

        :returns: Strictly increasing synthetic nanoseconds.
        """

        self.value += 1
        return self.value


class _CudaCompletion:
    """Record exact source callback attachment ordering."""

    operations: list[tuple[str, int | bytes, bytes | None]]
    runtime_operations: list[tuple[object, ...]]

    def __init__(self, runtime_operations: list[tuple[object, ...]]) -> None:
        """Create a callback ledger sharing the runtime-order ledger.

        :param runtime_operations: Cross-producer operation order.
        """

        self.operations = []
        self.runtime_operations = runtime_operations

    def arm(self, binding_digest: bytes) -> None:
        """Record one armed lifecycle.

        :param binding_digest: Exact source binding.
        """

        operation = ("arm", binding_digest, None)
        self.operations.append(operation)
        self.runtime_operations.append(("cuda-arm", binding_digest))

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Record one stream-tail callback attachment.

        :param stream_handle: Exact source stream handle.
        :param binding_digest: Exact armed binding.
        """

        operation = ("submit", stream_handle, binding_digest)
        self.operations.append(operation)
        self.runtime_operations.append(("cuda-submit", stream_handle, binding_digest))

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Record authenticated decoder allocation authorization.

        :param binding_digest: Exact armed source binding.
        :returns: ``False`` because this fixture retains no native callback.
        """

        operation = ("authorize", binding_digest, None)
        self.operations.append(operation)
        self.runtime_operations.append(("cuda-authorize", binding_digest))
        return False


class _Runtime:
    """Strict process-runtime double with exact authority accounting."""

    authorities: dict[tuple[NativeTerminalProducerClass, bytes | None], int]
    operations: list[tuple[object, ...]]
    registrations: list[NativeTerminalLifecycleRegistration]
    fail_registration: bool
    fail_work_completion: bool
    fail_scheduler_completion: bool

    def __init__(
        self,
        identity: PackedTerminalSourceIdentityPlan,
    ) -> None:
        """Freeze every producer route before wiring construction.

        :param identity: Exact source identity graph.
        """

        local = identity.local_binding.owner
        self.authorities = {
            (NativeTerminalProducerClass.LOCAL, None): _LOCAL_PRODUCER_ID,
            (NativeTerminalProducerClass.RECEIPT, local.digest): (
                _LOCAL_RECEIPT_PRODUCER_ID
            ),
            (
                NativeTerminalProducerClass.CONTROL,
                identity.request_ready_issuer.digest,
            ): _DECODER_CONTROL_PRODUCER_ID,
            (
                NativeTerminalProducerClass.RECEIPT,
                identity.request_ready_issuer.digest,
            ): _DECODER_RECEIPT_PRODUCER_ID,
        }
        if local != identity.publisher_issuer:
            self.authorities[
                (
                    NativeTerminalProducerClass.RECEIPT,
                    identity.publisher_issuer.digest,
                )
            ] = _PUBLISHER_RECEIPT_PRODUCER_ID
        self.operations = []
        self.registrations = []
        self.fail_registration = False
        self.fail_work_completion = False
        self.fail_scheduler_completion = False

    def python_producer_id(
        self,
        producer_class: NativeTerminalProducerClass,
        authenticated_issuer: NativeTerminalProcessIdentity | None = None,
    ) -> int:
        """Resolve one frozen producer route.

        :param producer_class: Required producer authority.
        :param authenticated_issuer: Exact route issuer when required.
        :returns: Stable fixture producer identity.
        """

        issuer_digest = (
            None if authenticated_issuer is None else authenticated_issuer.digest
        )
        return self.authorities[(producer_class, issuer_digest)]

    def register_lifecycle(
        self,
        registration: NativeTerminalLifecycleRegistration,
    ) -> None:
        """Record one lifecycle registration.

        :param registration: Complete native registration.
        """

        self.operations.append(("register", registration.binding.digest))
        if self.fail_registration:
            raise RuntimeError("synthetic registration failure")
        self.registrations.append(registration)

    def submit(
        self,
        producer_id: int,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Record one local or control event.

        :param producer_id: Exact producer identity.
        :param binding_digest: Exact lifecycle digest.
        :param kind: Native event kind.
        :param receipt: Optional receipt authority.
        :param reason: Optional failure reason.
        :param enqueued_ns: Exact producer timestamp.
        """

        self.operations.append(
            ("submit", producer_id, binding_digest, kind, receipt, reason, enqueued_ns)
        )

    def submit_imported_receipt(
        self,
        producer_id: int,
        receipt: NativeTerminalReceipt,
        kind: NativeTerminalOwnerEventKind,
        *,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Record one imported receipt event.

        :param producer_id: Exact producer identity.
        :param receipt: Imported native authority.
        :param kind: Receipt-consuming event.
        :param reason: Optional failure reason.
        :param enqueued_ns: Exact producer timestamp.
        """

        self.operations.append(
            ("import", producer_id, receipt, kind, reason, enqueued_ns)
        )

    def complete_work_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Record one exact work completion.

        :param producer_id: Exact producer identity.
        :param action: Work action being consumed.
        :param followup_kind: Earned lifecycle event.
        :param receipt: Optional publication authority.
        :param reason: Optional failure reason.
        :param enqueued_ns: Exact producer timestamp.
        """

        if self.fail_work_completion:
            self.fail_work_completion = False
            raise RuntimeError("synthetic work completion failure")
        self.operations.append(
            (
                "work",
                producer_id,
                action,
                followup_kind,
                receipt,
                reason,
                enqueued_ns,
            )
        )

    def complete_scheduler_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        completion_receipt: NativeTerminalReceipt | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Record one exact scheduler completion.

        :param producer_id: Exact receipt producer identity.
        :param action: Scheduler action being consumed.
        :param followup_kind: Reclaim-consumed event.
        :param completion_receipt: Scheduler-minted receipt authority.
        :param enqueued_ns: Exact producer timestamp.
        """

        if self.fail_scheduler_completion:
            raise RuntimeError("synthetic scheduler completion failure")
        self.operations.append(
            (
                "scheduler",
                producer_id,
                action,
                followup_kind,
                completion_receipt,
                enqueued_ns,
            )
        )

    def fail_scheduler_action(
        self,
        action: NativeTerminalOwnerAction,
        reason: str,
    ) -> None:
        """Record exact fail-closed scheduler ownership.

        :param action: Ambiguous scheduler action.
        :param reason: Stable process-fatal reason.
        """

        self.operations.append(("scheduler-failed", action, reason))

    def acknowledge_consumed_action(self, action: NativeTerminalOwnerAction) -> None:
        """Record one consumed terminal action.

        :param action: Exact retirement or quarantine action.
        """

        self.operations.append(("ack", action))


@dataclasses.dataclass(slots=True)
class _Harness:
    """Source wiring fixture with one process-lifetime runtime."""

    runtime: _Runtime
    cuda_completion: _CudaCompletion
    wiring: PackedTerminalSourceWiring
    identity: PackedTerminalSourceIdentityPlan
    submission: PackedTerminalSourceSubmission
    metrics: _Metrics
    publisher: _Publisher | None
    clock: _Clock


def _identities(
    *,
    local_rank: int = 0,
    request_seed: int = 0x71,
) -> PackedTerminalSourceIdentityPlan:
    """Build one TP2 source and TP1 decode identity graph.

    :param local_rank: Source rank selected as local.
    :param request_seed: Byte distinguishing one request lifecycle.
    :returns: Exact rank-local source identity plan.
    """

    key = PackedRequestKey(
        room_id=request_seed,
        request_generation=bytes((request_seed,)) * 16,
    )
    sources = tuple(
        TerminalProcessIdentity(
            process_generation=bytes((0x10 + rank,)) * 16,
            role=TerminalOwnerRole.SOURCE,
            tp_rank=rank,
            tp_size=2,
        )
        for rank in range(2)
    )
    decoder = TerminalProcessIdentity(
        process_generation=bytes.fromhex("31" * 16),
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
    )
    bindings = tuple(
        TerminalRequestBinding(
            request_key=key,
            owner=source,
            rank_manifest_digest=bytes.fromhex("41" * 32),
            allocation_digest=bytes.fromhex("51" * 32),
        )
        for source in sources
    )
    publication = TerminalPublicationIdentity(
        request_key=key,
        publisher_process_generation=sources[0].process_generation,
        publication_generation=bytes((request_seed - 0x10,)) * 16,
    )
    return PackedTerminalSourceIdentityPlan(
        local_binding=bindings[local_rank],
        source_bindings=bindings,
        publication_identity=publication,
        request_ready_issuer=decoder,
        publisher_issuer=sources[0],
    )


def _submission(
    identity: PackedTerminalSourceIdentityPlan,
) -> PackedTerminalSourceSubmission:
    """Build one immutable source handoff fixture.

    :param identity: Exact source identity plan.
    :returns: Complete source submission.
    """

    return PackedTerminalSourceSubmission(
        identity=identity,
        output_projection=(
            _Projection(payload=b"output")
            if identity.local_binding.owner.tp_rank == 0
            else None
        ),
        producer_event_generation=bytes.fromhex("81" * 16),
        producer_stream_handle=81,
        transport_submission=_SubmissionPayload(label="transport"),
    )


def _harness(
    *,
    local_rank: int = 0,
    failing_metrics: bool = False,
    attach_completion: bool = True,
    packed_ready: bool = True,
) -> _Harness:
    """Construct source wiring and accept one immutable submission.

    :param local_rank: Exact source TP rank.
    :param failing_metrics: Whether metrics reject every projection.
    :param attach_completion: Whether the source CUDA callback is attached.
    :param packed_ready: Whether authenticated decoder allocation arrived.
    :returns: Complete source test fixture.
    """

    identity = _identities(local_rank=local_rank)
    runtime = _Runtime(identity)
    cuda_completion = _CudaCompletion(runtime.operations)
    metrics = _Metrics(fail=failing_metrics)
    publisher = _Publisher() if local_rank == 0 else None
    clock = _Clock()
    wiring = PackedTerminalSourceWiring(
        runtime=runtime,
        cuda_completion=cuda_completion,
        local_identity=identity.local_binding.owner,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=clock.now_ns,
    )
    submission = _submission(identity)
    wiring.accept_submission(submission)
    if attach_completion:
        wiring.attach_producer_completion(submission)
    if packed_ready:
        wiring.packed_ready(identity.local_binding.digest)
    return _Harness(
        runtime=runtime,
        cuda_completion=cuda_completion,
        wiring=wiring,
        identity=identity,
        submission=submission,
        metrics=metrics,
        publisher=publisher,
        clock=clock,
    )


def _submission_observation(
    harness: _Harness,
    *,
    binding: NativeTerminalRequestBinding | None = None,
    producer_id: int = _LOCAL_PRODUCER_ID,
    enqueued_ns: int = 10,
    completed_ns: int = 20,
) -> NativeTerminalOwnerObservation:
    """Build one exact actionless native submission observation.

    :param harness: Source lifecycle fixture.
    :param binding: Native binding override for validation tests.
    :param producer_id: Native producer identity carried by the observation.
    :param enqueued_ns: Producer enqueue timestamp.
    :param completed_ns: Native commit timestamp.
    :returns: Evidence-only source commit observation.
    """

    local_binding = harness.identity.local_binding
    native_binding = binding
    if native_binding is None:
        native_binding = NativeTerminalRequestBinding.from_binding(local_binding)
    return NativeTerminalOwnerObservation(
        binding=native_binding,
        owner_sequence=0,
        producer_id=producer_id,
        producer_sequence=0,
        producer_rank=local_binding.owner.tp_rank,
        event_kind=NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        enqueued_ns=enqueued_ns,
        completed_ns=completed_ns,
        role=NativeTerminalOwnerRole.SOURCE,
    )


def _action(
    harness: _Harness,
    kind: NativeTerminalOwnerActionKind,
    action_id: int,
    *,
    binding: TerminalRequestBinding | None = None,
) -> NativeTerminalOwnerAction:
    """Build one exact owner action for the local lifecycle.

    :param harness: Source lifecycle fixture.
    :param kind: Exact action kind.
    :param action_id: Stable one-shot identity.
    :param binding: Source binding override for multi-request tests.
    :returns: Native action matching the local binding.
    """

    source_binding = harness.identity.local_binding if binding is None else binding
    native_binding = NativeTerminalRequestBinding.from_binding(source_binding)
    receipt = None
    if kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
        receipt = NativeTerminalReceipt(
            binding=native_binding,
            issuer=NativeTerminalProcessIdentity.from_identity(source_binding.owner),
            kind=NativeTerminalReceiptKind.RECLAIM_AUTHORIZED,
            outcome=NativeTerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=action_id,
            nonce=action_id.to_bytes(16, "big"),
        )
    return NativeTerminalOwnerAction(
        action_id=action_id,
        kind=kind,
        binding=native_binding,
        commit_timestamp_ns=action_id,
        receipt=receipt,
    )


def _ready(
    harness: _Harness,
    *,
    identity: PackedTerminalSourceIdentityPlan | None = None,
) -> None:
    """Deliver one authenticated request-ready receipt.

    :param harness: Source lifecycle fixture.
    :param identity: Source identity override for multi-request tests.
    """

    source_identity = harness.identity if identity is None else identity
    issued = TerminalWireReceiptIssuer(source_identity.request_ready_issuer).issue(
        binding=source_identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    harness.wiring.request_ready(
        binding_digest=source_identity.local_binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=source_identity.request_ready_issuer,
    )


def _failed(
    harness: _Harness,
    *,
    reason: str = "synthetic request-global failure",
) -> IssuedTerminalWireReceipt:
    """Deliver one authenticated request-failure receipt.

    :param harness: Source lifecycle fixture.
    :param reason: Stable failure evidence.
    :returns: Exact local and wire authority delivered to the wiring.
    """

    issued = TerminalWireReceiptIssuer(harness.identity.request_ready_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.FAILURE,
        outcome=TerminalReceiptOutcome.FAILURE,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    harness.wiring.request_failed(
        binding_digest=harness.identity.local_binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=harness.identity.request_ready_issuer,
        reason=reason,
    )
    return issued


def _publication_result(
    harness: _Harness,
    publication: FrozenTerminalGatewayPublication,
    *,
    success: bool,
) -> TerminalGatewayPublicationSuccess | TerminalGatewayPublicationFailure:
    """Build one authenticated canonical publisher result.

    :param harness: Source lifecycle fixture.
    :param publication: Exact immutable publication attempt.
    :param success: Whether publication succeeded.
    :returns: Complete publisher result.
    """

    issuer = TerminalWireReceiptIssuer(harness.identity.publisher_issuer)
    kind = TerminalReceiptKind.GATEWAY_PUBLISHED
    outcome = TerminalReceiptOutcome.SUCCESS
    if not success:
        kind = TerminalReceiptKind.FAILURE
        outcome = TerminalReceiptOutcome.FAILURE
    timestamp_ns = harness.clock.now_ns()
    receipts = tuple(
        issuer.issue(
            binding=binding,
            kind=kind,
            outcome=outcome,
            terminal_timestamp_ns=timestamp_ns,
        )
        for binding in harness.identity.source_bindings
    )
    if success:
        return TerminalGatewayPublicationSuccess(
            publication=publication,
            completed_ns=timestamp_ns,
            source_receipts=receipts,
        )
    return TerminalGatewayPublicationFailure(
        publication=publication,
        failed_ns=timestamp_ns,
        source_receipts=receipts,
        reason="synthetic publication failure",
        formatted_traceback="synthetic traceback",
    )


def test_runtime_registration_precedes_first_source_event() -> None:
    """Publish lifecycle identity before any producer can target it."""

    harness = _harness()
    operations = harness.runtime.operations
    assert operations[0] == ("register", harness.identity.local_binding.digest)
    assert operations[1][0:4] == (
        "submit",
        _LOCAL_PRODUCER_ID,
        harness.identity.local_binding.digest,
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
    )
    assert operations[2:] == [
        ("cuda-arm", harness.identity.local_binding.digest),
        ("cuda-submit", 81, harness.identity.local_binding.digest),
        ("cuda-authorize", harness.identity.local_binding.digest),
    ]
    assert harness.cuda_completion.operations == [
        ("arm", harness.identity.local_binding.digest, None),
        ("submit", 81, harness.identity.local_binding.digest),
        ("authorize", harness.identity.local_binding.digest, None),
    ]
    assert harness.wiring.lifecycle_published(harness.identity.local_binding.digest)


def test_packed_ready_authorizes_attached_native_callback_once() -> None:
    """Authenticated allocation releases the already-attached native callback."""

    harness = _harness(packed_ready=False)
    digest = harness.identity.local_binding.digest
    operation_count = len(harness.runtime.operations)

    assert not harness.wiring.packed_ready(digest)
    assert harness.runtime.operations[-1] == ("cuda-authorize", digest)
    assert len(harness.runtime.operations) == operation_count + 1
    assert not harness.wiring.packed_ready(digest)
    assert len(harness.runtime.operations) == operation_count + 1


def test_packed_ready_before_attachment_is_released_after_submit() -> None:
    """A fast READY response cannot outrun callback registration."""

    harness = _harness(attach_completion=False, packed_ready=False)
    digest = harness.identity.local_binding.digest
    operation_count = len(harness.runtime.operations)

    assert not harness.wiring.packed_ready(digest)
    assert len(harness.runtime.operations) == operation_count
    harness.wiring.attach_producer_completion(harness.submission)

    assert harness.runtime.operations[-3:] == [
        ("cuda-arm", digest),
        ("cuda-submit", 81, digest),
        ("cuda-authorize", digest),
    ]


def test_source_timing_projects_two_main_members_and_one_dflash_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TP2 DFlash timing retains exactly three native transfer samples."""

    harnesses = (_harness(local_rank=0), _harness(local_rank=1))
    with caplog.at_level(
        logging.INFO,
        logger="sglang.srt.disaggregation.terminal_progress.source_wiring",
    ):
        for rank, harness in enumerate(harnesses):
            binding = harness.identity.local_binding
            harness.wiring.submission_committed(
                _submission_observation(
                    harness,
                    enqueued_ns=100 + rank,
                    completed_ns=110 + rank,
                )
            )
            timings = [
                GroupedNixlMemberTiming(
                    member=GroupedNixlTransferMember.MAIN,
                    owner_cookie=10 + rank,
                    post_started_ns=200 + rank,
                    native_terminal_ns=220 + rank,
                )
            ]
            if rank == 0:
                timings.append(
                    GroupedNixlMemberTiming(
                        member=GroupedNixlTransferMember.DFLASH_BOUNDARY,
                        owner_cookie=20,
                        post_started_ns=201,
                        native_terminal_ns=225,
                    )
                )
            result = GroupedNixlTerminalResult(
                binding_digest=binding.digest,
                successful=True,
                transfer_count=len(timings),
                native_timestamp_ns=max(
                    timing.native_terminal_ns for timing in timings
                ),
                reason=None,
                member_timings=tuple(timings),
            )
            harness.wiring.grouped_native_terminal(result)

    samples = tuple(
        sample
        for record in caplog.records
        if (sample := parse_terminal_progress_timing_log_line(record.getMessage()))
        is not None
    )
    native_samples = tuple(
        sample
        for sample in samples
        if sample.field.value == "native_terminal_delivery_ms"
    )

    assert len(samples) == 5
    assert len(native_samples) == 3
    assert {sample.sample_key for sample in native_samples} == {
        "main:writer-0",
        "main:writer-1",
        "boundary:writer-0",
    }


def test_submission_observation_requires_complete_binding_before_one_shot() -> None:
    """Malformed evidence cannot consume the valid submission projection."""

    harness = _harness()
    native_binding = NativeTerminalRequestBinding.from_binding(
        harness.identity.local_binding
    )
    malformed_binding = dataclasses.replace(
        native_binding,
        room_id=native_binding.room_id + 1,
    )

    with pytest.raises(ValueError, match="binding digest is inconsistent"):
        harness.wiring.submission_committed(
            _submission_observation(harness, binding=malformed_binding)
        )
    with pytest.raises(RuntimeError, match="another producer"):
        harness.wiring.submission_committed(
            _submission_observation(harness, producer_id=_LOCAL_PRODUCER_ID + 1)
        )

    valid = _submission_observation(harness)
    harness.wiring.submission_committed(valid)
    with pytest.raises(RuntimeError, match="observed twice"):
        harness.wiring.submission_committed(valid)


def test_unpublished_registration_failure_can_cancel_exact_submission() -> None:
    """Permit paired scheduler rollback only before native publication."""

    identity = _identities()
    runtime = _Runtime(identity)
    runtime.fail_registration = True
    wiring = PackedTerminalSourceWiring(
        runtime=runtime,
        cuda_completion=_CudaCompletion(runtime.operations),
        local_identity=identity.local_binding.owner,
        publisher=_Publisher(),
        metrics_sink=_Metrics(),
        clock_ns=_Clock().now_ns,
    )
    submission = _submission(identity)
    with pytest.raises(RuntimeError, match="synthetic registration failure"):
        wiring.accept_submission(submission)
    assert not wiring.lifecycle_published(identity.local_binding.digest)
    assert wiring.cancel_unpublished(identity.local_binding.digest) == submission
    assert wiring.inventory().active_binding_digests == ()


def test_client_cancellation_records_completion_without_native_failure() -> None:
    """Published ownership completes normally after client intent is recorded."""

    harness = _harness()
    binding = harness.identity.local_binding
    operation_count = len(harness.runtime.operations)

    assert harness.wiring.cancel_request(binding, "client disconnected") is (
        PackedTerminalSourceCancellationDisposition.COMPLETION_REQUIRED
    )
    assert len(harness.runtime.operations) == operation_count
    assert harness.wiring.inventory().completion_required_binding_digests == (
        binding.digest,
    )
    assert harness.wiring.cancel_request(binding, "duplicate disconnect") is (
        PackedTerminalSourceCancellationDisposition.ALREADY_RECORDED
    )
    assert len(harness.runtime.operations) == operation_count

    _ready(harness)
    assert harness.runtime.operations[-1][3] is (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY
    )


def test_request_ready_commits_before_late_client_cancellation() -> None:
    """A ready receipt wins the cutover without manufacturing rollback."""

    harness = _harness()
    _ready(harness)
    operation_count = len(harness.runtime.operations)

    assert (
        harness.wiring.cancel_request(
            harness.identity.local_binding,
            "client disconnected after ready",
        )
        is PackedTerminalSourceCancellationDisposition.TOO_LATE_FOR_ROLLBACK
    )
    assert len(harness.runtime.operations) == operation_count
    assert harness.wiring.inventory().completion_required_binding_digests == ()


@pytest.mark.parametrize(
    "cancellation_stage",
    ("accepted", "gathered", "outcomes_sent", "ack_sent"),
)
def test_completion_required_cancellation_retires_at_every_downstream_phase(
    cancellation_stage: str,
) -> None:
    """Client intent never suppresses the ordinary successful retirement chain."""

    harness = _harness()
    binding = harness.identity.local_binding

    def cancel_if(stage: str) -> None:
        """Record cancellation at the selected lifecycle phase.

        :param stage: Current synthetic source phase.
        """

        if cancellation_stage != stage:
            return
        assert harness.wiring.cancel_request(binding, "client disconnected") is (
            PackedTerminalSourceCancellationDisposition.COMPLETION_REQUIRED
        )

    cancel_if("accepted")
    harness.wiring.consume_gather_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_GATHER_READY, 70),
        lambda submission, action: None,
    )
    cancel_if("gathered")
    harness.wiring.consume_outcome_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY, 71),
        lambda submission, action: None,
    )
    cancel_if("outcomes_sent")
    harness.wiring.teardown_received(
        binding.digest, harness.identity.request_ready_issuer
    )
    harness.wiring.consume_ack_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_ACK_READY, 72),
        lambda submission, action: None,
    )
    cancel_if("ack_sent")
    _ready(harness)
    harness.wiring.consume_reclaim_authorized(
        _action(harness, NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED, 73),
        lambda submission: None,
    )
    publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        74,
    )
    publication = harness.wiring.consume_gateway_publication_ready(publication_action)
    assert publication is not None
    harness.wiring.publisher_result(
        _publication_result(harness, publication, success=True)
    )
    retirement_callbacks: list[PackedTerminalSourceSubmission] = []
    retired = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        75,
    )
    harness.wiring.consume_terminal_action(
        retired,
        lambda submission, action: retirement_callbacks.append(submission),
    )

    assert retirement_callbacks == [harness.submission]
    inventory = harness.wiring.inventory()
    assert inventory.active_binding_digests == ()
    assert inventory.completion_required_binding_digests == ()


def test_full_source_success_uses_runtime_completion_surfaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Join reclaim and publication before exact runtime retirement."""

    caplog.set_level(
        logging.INFO,
        logger="sglang.srt.disaggregation.terminal_progress.source_wiring",
    )
    harness = _harness()
    digest = harness.identity.local_binding.digest
    harness.wiring.consume_gather_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_GATHER_READY, 1),
        lambda submission, action: None,
    )
    harness.wiring.consume_outcome_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY, 2),
        lambda submission, action: None,
    )
    harness.wiring.teardown_received(digest, harness.identity.request_ready_issuer)
    harness.wiring.consume_ack_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_ACK_READY, 3),
        lambda submission, action: None,
    )
    _ready(harness)
    reclaim = _action(
        harness,
        NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        4,
    )
    publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        5,
    )
    publication = harness.wiring.consume_gateway_publication_ready(publication_action)
    assert publication is not None
    reclaim_receipt = harness.wiring.consume_reclaim_authorized(
        reclaim,
        lambda submission: None,
    )
    assert reclaim_receipt.wire_receipt.kind is TerminalReceiptKind.RECLAIM_CONSUMED
    result = _publication_result(harness, publication, success=True)
    local_receipt = next(
        receipt
        for receipt in result.source_receipts
        if receipt.wire_receipt.binding == harness.identity.local_binding
    )
    harness.wiring.publication_receipt(
        wire_receipt=local_receipt.wire_receipt,
        local_receipt=local_receipt.local_receipt,
        authenticated_issuer=harness.identity.publisher_issuer,
    )
    retired = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        6,
    )
    retired_callbacks: list[PackedTerminalSourceSubmission] = []
    assert (
        harness.wiring.consume_terminal_action(
            retired,
            lambda submission, action: retired_callbacks.append(submission),
        )
        == harness.submission
    )
    assert retired_callbacks == [harness.submission]
    assert harness.wiring.inventory().active_binding_digests == ()
    assert any(operation[0] == "scheduler" for operation in harness.runtime.operations)
    assert any(operation[0] == "work" for operation in harness.runtime.operations)
    assert harness.runtime.operations[-1] == ("ack", retired)
    samples = tuple(
        sample
        for record in caplog.records
        if (sample := parse_terminal_progress_timing_log_line(record.getMessage()))
        is not None
    )
    assert len(samples) == 1
    assert samples[0].field.value == "gateway_publication_ms"
    assert samples[0].sample_key == "canonical-source-publisher"


def test_retirement_side_effect_failure_retains_native_and_wiring_authority() -> None:
    """External retirement must succeed before action acknowledgement or deletion."""

    harness = _harness()
    digest = harness.identity.local_binding.digest
    harness.wiring.consume_gather_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_GATHER_READY, 10),
        lambda submission, action: None,
    )
    harness.wiring.consume_outcome_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY, 11),
        lambda submission, action: None,
    )
    harness.wiring.teardown_received(digest, harness.identity.request_ready_issuer)
    harness.wiring.consume_ack_ready(
        _action(harness, NativeTerminalOwnerActionKind.SOURCE_ACK_READY, 12),
        lambda submission, action: None,
    )
    _ready(harness)
    harness.wiring.consume_reclaim_authorized(
        _action(harness, NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED, 13),
        lambda submission: None,
    )
    publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        14,
    )
    publication = harness.wiring.consume_gateway_publication_ready(publication_action)
    assert publication is not None
    harness.wiring.publisher_result(
        _publication_result(harness, publication, success=True)
    )
    retired = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        15,
    )
    operation_count = len(harness.runtime.operations)

    def fail_retirement(
        submission: PackedTerminalSourceSubmission,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Reject the external actor/control-state retirement boundary.

        :param submission: Exact immutable source submission.
        :param action: Exact joined retirement authority.
        """

        raise RuntimeError("synthetic retirement failure")

    with pytest.raises(RuntimeError, match="synthetic retirement failure"):
        harness.wiring.consume_terminal_action(retired, fail_retirement)
    assert len(harness.runtime.operations) == operation_count
    assert harness.wiring.inventory().active_binding_digests == (digest,)


def test_reclaim_failure_enters_explicit_runtime_fatal_boundary() -> None:
    """Never lose scheduler authority behind a failed cleanup callback."""

    harness = _harness()
    reclaim = _action(
        harness,
        NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        20,
    )

    def fail_release(submission: PackedTerminalSourceSubmission) -> None:
        """Reject one synthetic scheduler-affine release.

        :param submission: Exact immutable source submission.
        """

        raise RuntimeError("synthetic release failure")

    with pytest.raises(RuntimeError, match="synthetic release failure"):
        harness.wiring.consume_reclaim_authorized(reclaim, fail_release)
    failure = harness.runtime.operations[-1]
    assert failure[0:2] == ("scheduler-failed", reclaim)
    assert "source reclaim consumption failed" in failure[2]


def test_source_work_failure_returns_request_failure_to_runtime() -> None:
    """Return failed off-loop work through the same runtime action ledger."""

    harness = _harness()
    action = _action(
        harness,
        NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        30,
    )

    def fail_gather(
        submission: PackedTerminalSourceSubmission,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Reject one synthetic gather.

        :param submission: Exact immutable source submission.
        :param action: Exact native owner authority.
        """

        del submission, action
        raise RuntimeError("synthetic gather failure")

    with pytest.raises(RuntimeError, match="synthetic gather failure"):
        harness.wiring.consume_gather_ready(action, fail_gather)
    completion = harness.runtime.operations[-1]
    assert completion[0:4] == (
        "work",
        _LOCAL_PRODUCER_ID,
        action,
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
    )
    assert "source gather publication failed" in completion[5]


def test_noncanonical_rank_consumes_its_exact_publication_receipt() -> None:
    """Keep per-rank authority instead of inferring canonical completion."""

    harness = _harness(local_rank=1)
    _ready(harness)
    action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        40,
    )
    assert harness.wiring.consume_gateway_publication_ready(action) is None
    canonical = _identities(local_rank=0)
    canonical_ready = TerminalWireReceiptIssuer(
        harness.identity.request_ready_issuer
    ).issue(
        binding=canonical.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    publication = FrozenTerminalGatewayPublication(
        identity=harness.identity.publication_identity,
        canonical_binding=harness.identity.source_bindings[0],
        source_bindings=harness.identity.source_bindings,
        request_ready_receipt=canonical_ready.local_receipt,
        output_projection=_Projection(payload=b"output"),
        enqueued_ns=harness.clock.now_ns(),
    )
    harness.wiring.publisher_result(
        _publication_result(harness, publication, success=True)
    )
    completion = harness.runtime.operations[-1]
    assert completion[0:4] == (
        "work",
        _PUBLISHER_RECEIPT_PRODUCER_ID,
        action,
        NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
    )


def test_noncanonical_publication_receipt_joins_late_native_action() -> None:
    """An authenticated remote outcome may outrun the local action inbox."""

    harness = _harness(local_rank=1)
    _ready(harness)
    canonical = _identities(local_rank=0)
    canonical_ready = TerminalWireReceiptIssuer(
        harness.identity.request_ready_issuer
    ).issue(
        binding=canonical.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    publication = FrozenTerminalGatewayPublication(
        identity=harness.identity.publication_identity,
        canonical_binding=harness.identity.source_bindings[0],
        source_bindings=harness.identity.source_bindings,
        request_ready_receipt=canonical_ready.local_receipt,
        output_projection=_Projection(payload=b"output"),
        enqueued_ns=harness.clock.now_ns(),
    )
    result = _publication_result(harness, publication, success=True)
    local_receipt = next(
        receipt
        for receipt in result.source_receipts
        if receipt.wire_receipt.binding == harness.identity.local_binding
    )
    operation_count = len(harness.runtime.operations)

    harness.wiring.publication_receipt(
        wire_receipt=local_receipt.wire_receipt,
        local_receipt=local_receipt.local_receipt,
        authenticated_issuer=harness.identity.publisher_issuer,
    )

    inventory = harness.wiring.inventory()
    assert len(harness.runtime.operations) == operation_count
    assert inventory.pending_publication_action_count == 0
    assert inventory.pending_publication_receipt_count == 1

    action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        41,
    )
    assert harness.wiring.consume_gateway_publication_ready(action) is None
    completion = harness.runtime.operations[-1]
    assert completion[0:4] == (
        "work",
        _PUBLISHER_RECEIPT_PRODUCER_ID,
        action,
        NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
    )
    inventory = harness.wiring.inventory()
    assert inventory.pending_publication_action_count == 0
    assert inventory.pending_publication_receipt_count == 0


def test_publication_failure_quarantines_identity_after_safe_reclaim() -> None:
    """Retain failed publication identity while storage is already reusable."""

    harness = _harness()
    _ready(harness)
    reclaim = _action(
        harness,
        NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        50,
    )
    harness.wiring.consume_reclaim_authorized(reclaim, lambda submission: None)
    publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        51,
    )
    publication = harness.wiring.consume_gateway_publication_ready(publication_action)
    assert publication is not None
    harness.wiring.publisher_result(
        _publication_result(harness, publication, success=False)
    )
    quarantine = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        52,
    )
    retained_actions: list[NativeTerminalOwnerAction] = []
    harness.wiring.consume_quarantine(
        quarantine,
        retained_actions.append,
    )
    assert retained_actions == [quarantine]
    inventory = harness.wiring.inventory()
    assert inventory.active_binding_digests == (harness.identity.local_binding.digest,)
    assert inventory.quarantined_binding_digests == (
        harness.identity.local_binding.digest,
    )
    assert harness.runtime.operations[-1] == ("ack", quarantine)


@pytest.mark.parametrize(
    "kind",
    (
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        NativeTerminalOwnerActionKind.PROCESS_FATAL,
    ),
)
def test_quarantine_marks_before_external_retention_and_acknowledgement(
    kind: NativeTerminalOwnerActionKind,
) -> None:
    """Wiring quarantine precedes external retention and native acknowledgement."""

    harness = _harness()
    digest = harness.identity.local_binding.digest
    action = _action(harness, kind, 53)
    callback_observations: list[tuple[NativeTerminalOwnerAction, bool, bool]] = []

    def retain_resources(candidate: NativeTerminalOwnerAction) -> None:
        """Capture ownership state at the external retention boundary.

        :param candidate: Exact fail-closed action being retained.
        """

        inventory = harness.wiring.inventory()
        callback_observations.append(
            (
                candidate,
                digest in inventory.quarantined_binding_digests,
                ("ack", candidate) in harness.runtime.operations,
            )
        )

    harness.wiring.consume_quarantine(action, retain_resources)

    assert callback_observations == [(action, True, False)]
    inventory = harness.wiring.inventory()
    assert inventory.active_binding_digests == (digest,)
    assert inventory.quarantined_binding_digests == (digest,)
    assert harness.runtime.operations[-1] == ("ack", action)


def test_quarantine_callback_failure_retains_record_and_claimed_action() -> None:
    """A failed external retention cannot acknowledge or redeliver authority."""

    harness = _harness()
    digest = harness.identity.local_binding.digest
    action = _action(
        harness,
        NativeTerminalOwnerActionKind.PROCESS_FATAL,
        54,
    )
    operation_count = len(harness.runtime.operations)

    def fail_retention(candidate: NativeTerminalOwnerAction) -> None:
        """Reject the external resource-retention boundary.

        :param candidate: Exact fail-closed action being retained.
        """

        assert candidate is action
        raise RuntimeError("synthetic resource retention failure")

    with pytest.raises(
        PackedTerminalSourceQuarantineRetentionError,
        match="source quarantine retained exact action authority",
    ):
        harness.wiring.consume_quarantine(action, fail_retention)

    inventory = harness.wiring.inventory()
    assert inventory.active_binding_digests == (digest,)
    assert inventory.quarantined_binding_digests == (digest,)
    assert inventory.retained_quarantine_action_ids == (action.action_id,)
    assert len(harness.runtime.operations) == operation_count
    with pytest.raises(RuntimeError, match="native source action was delivered twice"):
        harness.wiring.consume_quarantine(action, lambda candidate: None)


def test_distinct_quarantine_actions_idempotently_retain_one_record() -> None:
    """Distinct fail-closed authorities preserve one exact quarantined record."""

    harness = _harness()
    digest = harness.identity.local_binding.digest
    request_quarantine = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        55,
    )
    process_fatal = _action(
        harness,
        NativeTerminalOwnerActionKind.PROCESS_FATAL,
        56,
    )
    retained_actions: list[NativeTerminalOwnerAction] = []

    harness.wiring.consume_quarantine(request_quarantine, retained_actions.append)
    harness.wiring.consume_quarantine(process_fatal, retained_actions.append)

    inventory = harness.wiring.inventory()
    assert retained_actions == [request_quarantine, process_fatal]
    assert inventory.active_binding_digests == (digest,)
    assert inventory.quarantined_binding_digests == (digest,)
    assert harness.runtime.operations[-2:] == [
        ("ack", request_quarantine),
        ("ack", process_fatal),
    ]


def test_fail_closed_closure_transfers_exact_quarantined_actions_once() -> None:
    """Fail-closed closure transfers sorted publication authority exactly once."""

    harness = _harness()
    second_identity = _identities(request_seed=0x72)
    second_submission = _submission(second_identity)
    harness.wiring.accept_submission(second_submission)
    _ready(harness)
    _ready(harness, identity=second_identity)
    first_publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        72,
    )
    second_publication_action = _action(
        harness,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        62,
        binding=second_identity.local_binding,
    )
    assert (
        harness.wiring.consume_gateway_publication_ready(first_publication_action)
        is not None
    )
    assert (
        harness.wiring.consume_gateway_publication_ready(second_publication_action)
        is not None
    )
    harness.wiring.consume_quarantine(
        _action(
            harness,
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
            73,
        ),
        lambda candidate: None,
    )
    harness.wiring.consume_quarantine(
        _action(
            harness,
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            63,
            binding=second_identity.local_binding,
        ),
        lambda candidate: None,
    )

    closure = harness.wiring.take_fail_closed_closure()

    expected_digests = tuple(
        sorted(
            (
                harness.identity.local_binding.digest,
                second_identity.local_binding.digest,
            )
        )
    )
    assert closure.inventory.active_binding_digests == expected_digests
    assert closure.inventory.quarantined_binding_digests == expected_digests
    assert closure.inventory.pending_publication_action_ids == (62, 72)
    assert closure.retained_publication_actions == (
        second_publication_action,
        first_publication_action,
    )
    with pytest.raises(
        RuntimeError,
        match="source wiring fail-closed closure was already taken",
    ):
        harness.wiring.take_fail_closed_closure()


def test_metrics_failure_never_gates_runtime_progress() -> None:
    """Keep observability failure outside lifecycle authority."""

    harness = _harness(failing_metrics=True)
    _ready(harness)
    assert harness.metrics.values == []
    assert harness.runtime.operations[-1][3] is (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY
    )


def test_request_ready_rejects_wrong_issuer_before_runtime_submission() -> None:
    """Reject valid-looking authority from an unauthenticated route."""

    harness = _harness()
    wrong_issuer = harness.identity.publisher_issuer
    ready = TerminalWireReceiptIssuer(wrong_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    operation_count = len(harness.runtime.operations)
    with pytest.raises(RuntimeError, match="authenticated another issuer"):
        harness.wiring.request_ready(
            binding_digest=harness.identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=wrong_issuer,
        )
    assert len(harness.runtime.operations) == operation_count


def test_request_failure_retains_authority_and_submits_native_failure() -> None:
    """Preserve failure terminality as failure throughout source ingress."""

    harness = _harness()
    reason = "decode coordinator failed request-global coordination"
    issued = _failed(harness, reason=reason)

    operation = harness.runtime.operations[-1]
    assert operation[0] == "import"
    assert operation[1] == _DECODER_RECEIPT_PRODUCER_ID
    assert operation[2] == NativeTerminalReceipt.from_wire_receipt(issued.wire_receipt)
    assert operation[3] is NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED
    assert operation[4] == reason
    assert harness.metrics.values[-1].event_kind is (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED
    )
    operation_count = len(harness.runtime.operations)
    with pytest.raises(RuntimeError, match="terminality was delivered twice"):
        harness.wiring.request_failed(
            binding_digest=harness.identity.local_binding.digest,
            wire_receipt=issued.wire_receipt,
            local_receipt=issued.local_receipt,
            authenticated_issuer=harness.identity.request_ready_issuer,
            reason=reason,
        )
    assert len(harness.runtime.operations) == operation_count


def test_request_failure_rejects_wrong_route_before_runtime_submission() -> None:
    """Reject failure authority delivered by another authenticated route."""

    harness = _harness()
    wrong_issuer = harness.identity.publisher_issuer
    failure = TerminalWireReceiptIssuer(wrong_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.FAILURE,
        outcome=TerminalReceiptOutcome.FAILURE,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    operation_count = len(harness.runtime.operations)
    with pytest.raises(RuntimeError, match="authenticated another issuer"):
        harness.wiring.request_failed(
            binding_digest=harness.identity.local_binding.digest,
            wire_receipt=failure.wire_receipt,
            local_receipt=failure.local_receipt,
            authenticated_issuer=wrong_issuer,
            reason="synthetic remote failure",
        )
    assert len(harness.runtime.operations) == operation_count


def test_request_failure_rejects_ready_authority_without_poisoning_terminality() -> (
    None
):
    """Validate receipt kind before retaining request-terminal authority."""

    harness = _harness()
    ready = TerminalWireReceiptIssuer(harness.identity.request_ready_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    operation_count = len(harness.runtime.operations)
    with pytest.raises(RuntimeError, match="another authority"):
        harness.wiring.request_failed(
            binding_digest=harness.identity.local_binding.digest,
            wire_receipt=ready.wire_receipt,
            local_receipt=ready.local_receipt,
            authenticated_issuer=harness.identity.request_ready_issuer,
            reason="synthetic remote failure",
        )
    assert len(harness.runtime.operations) == operation_count
    _failed(harness)


def test_source_wiring_has_no_raw_owner_or_dynamic_registration() -> None:
    """Keep native ownership and producer registration in the runtime only."""

    source = inspect.getsource(PackedTerminalSourceWiring)
    assert "NativeTerminalOwner(" not in source
    assert "self._owner" not in source
    assert "register_producer" not in source
    assert "retire_python_producer" not in source
