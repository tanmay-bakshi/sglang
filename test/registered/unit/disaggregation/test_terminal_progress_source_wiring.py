import dataclasses
import hashlib
import inspect

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
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
    PackedTerminalSourceMetric,
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
    wiring: PackedTerminalSourceWiring
    identity: PackedTerminalSourceIdentityPlan
    submission: PackedTerminalSourceSubmission
    metrics: _Metrics
    publisher: _Publisher | None
    clock: _Clock


def _identities(*, local_rank: int = 0) -> PackedTerminalSourceIdentityPlan:
    """Build one TP2 source and TP1 decode identity graph.

    :param local_rank: Source rank selected as local.
    :returns: Exact rank-local source identity plan.
    """

    key = PackedRequestKey(room_id=71, request_generation=bytes.fromhex("71" * 16))
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
        publication_generation=bytes.fromhex("61" * 16),
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
        output_projection=_Projection(payload=b"output"),
        producer_event_generation=bytes.fromhex("81" * 16),
        transport_submission=_SubmissionPayload(label="transport"),
    )


def _harness(*, local_rank: int = 0, failing_metrics: bool = False) -> _Harness:
    """Construct source wiring and accept one immutable submission.

    :param local_rank: Exact source TP rank.
    :param failing_metrics: Whether metrics reject every projection.
    :returns: Complete source test fixture.
    """

    identity = _identities(local_rank=local_rank)
    runtime = _Runtime(identity)
    metrics = _Metrics(fail=failing_metrics)
    publisher = _Publisher() if local_rank == 0 else None
    clock = _Clock()
    wiring = PackedTerminalSourceWiring(
        runtime=runtime,
        local_identity=identity.local_binding.owner,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=clock.now_ns,
    )
    submission = _submission(identity)
    wiring.accept_submission(submission)
    return _Harness(
        runtime=runtime,
        wiring=wiring,
        identity=identity,
        submission=submission,
        metrics=metrics,
        publisher=publisher,
        clock=clock,
    )


def _action(
    harness: _Harness,
    kind: NativeTerminalOwnerActionKind,
    action_id: int,
) -> NativeTerminalOwnerAction:
    """Build one exact owner action for the local lifecycle.

    :param harness: Source lifecycle fixture.
    :param kind: Exact action kind.
    :param action_id: Stable one-shot identity.
    :returns: Native action matching the local binding.
    """

    native_binding = NativeTerminalRequestBinding.from_binding(
        harness.identity.local_binding
    )
    receipt = None
    if kind is NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
        receipt = NativeTerminalReceipt(
            binding=native_binding,
            issuer=NativeTerminalProcessIdentity.from_identity(
                harness.identity.local_binding.owner
            ),
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


def _ready(harness: _Harness) -> None:
    """Deliver one authenticated request-ready receipt.

    :param harness: Source lifecycle fixture.
    """

    issued = TerminalWireReceiptIssuer(harness.identity.request_ready_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    harness.wiring.request_ready(
        binding_digest=harness.identity.local_binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=harness.identity.request_ready_issuer,
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
    assert harness.wiring.lifecycle_published(harness.identity.local_binding.digest)


def test_unpublished_registration_failure_can_cancel_exact_submission() -> None:
    """Permit paired scheduler rollback only before native publication."""

    identity = _identities()
    runtime = _Runtime(identity)
    runtime.fail_registration = True
    wiring = PackedTerminalSourceWiring(
        runtime=runtime,
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


def test_full_source_success_uses_runtime_completion_surfaces() -> None:
    """Join reclaim and publication before exact runtime retirement."""

    harness = _harness()
    digest = harness.identity.local_binding.digest
    harness.wiring.producer_completed(digest)
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
    harness.wiring.publisher_result(
        _publication_result(harness, publication, success=True)
    )
    retired = _action(
        harness,
        NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        6,
    )
    assert harness.wiring.consume_terminal_action(retired) == harness.submission
    assert harness.wiring.inventory().active_binding_digests == ()
    assert any(operation[0] == "scheduler" for operation in harness.runtime.operations)
    assert any(operation[0] == "work" for operation in harness.runtime.operations)
    assert harness.runtime.operations[-1] == ("ack", retired)


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
    assert harness.wiring.consume_terminal_action(quarantine) is None
    inventory = harness.wiring.inventory()
    assert inventory.active_binding_digests == (harness.identity.local_binding.digest,)
    assert inventory.quarantined_binding_digests == (
        harness.identity.local_binding.digest,
    )
    assert harness.runtime.operations[-1] == ("ack", quarantine)


def test_metrics_failure_never_gates_runtime_progress() -> None:
    """Keep observability failure outside lifecycle authority."""

    harness = _harness(failing_metrics=True)
    harness.wiring.producer_completed(harness.identity.local_binding.digest)
    assert harness.metrics.values == []
    assert harness.runtime.operations[-1][3] is (
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED
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
