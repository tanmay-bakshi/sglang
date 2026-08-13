import dataclasses
import hashlib

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
    NativeTerminalRequestBinding,
    NativeTerminalReceipt,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationSuccess,
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
    TerminalWireReceiptIssuer,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


@dataclasses.dataclass(frozen=True, slots=True)
class _RuntimeSubmission:
    """One native runtime submission observation.

    :ivar producer: Exact producer fixture.
    :ivar binding_digest: Exact target lifecycle.
    :ivar kind: Submitted event kind.
    :ivar receipt: Optional native authority.
    :ivar reason: Optional stable failure evidence.
    """

    producer: object
    binding_digest: bytes
    kind: NativeTerminalOwnerEventKind
    receipt: NativeTerminalReceipt | None
    reason: str | None


class _Runtime:
    """Deterministic process-lifetime native runtime fixture."""

    local_events: object
    local_receipts: object
    registrations: list[
        tuple[
            TerminalRequestBinding,
            TerminalPublicationIdentity,
            tuple[TerminalProcessIdentity, ...],
        ]
    ]
    submissions: list[_RuntimeSubmission]
    completed_actions: list[NativeTerminalOwnerAction]
    controls: dict[bytes, object]

    def __init__(self) -> None:
        """Create independent stable producer fixtures."""

        self.local_events = object()
        self.local_receipts = object()
        self.registrations = []
        self.submissions = []
        self.completed_actions = []
        self.controls = {}

    def control_producer(self, issuer_digest: bytes) -> object:
        """Return one stable authenticated producer per issuer.

        :param issuer_digest: Exact remote process identity.
        :returns: Stable producer fixture.
        """

        producer = self.controls.get(issuer_digest)
        if producer is None:
            producer = object()
            self.controls[issuer_digest] = producer
        return producer

    def register_source(
        self,
        binding: TerminalRequestBinding,
        publication_identity: TerminalPublicationIdentity,
        trusted_issuers: tuple[TerminalProcessIdentity, ...],
    ) -> None:
        """Record one exact source registration.

        :param binding: Exact source binding.
        :param publication_identity: Exact publication identity.
        :param trusted_issuers: Canonical issuer set.
        """

        self.registrations.append(
            (binding, publication_identity, trusted_issuers)
        )

    def submit(
        self,
        producer: object,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
    ) -> None:
        """Record one producer-ordered owner event.

        :param producer: Exact producer fixture.
        :param binding_digest: Exact lifecycle identity.
        :param kind: Closed event kind.
        :param receipt: Optional native authority.
        :param reason: Optional stable failure evidence.
        """

        self.submissions.append(
            _RuntimeSubmission(
                producer=producer,
                binding_digest=binding_digest,
                kind=kind,
                receipt=receipt,
                reason=reason,
            )
        )

    def complete_scheduler_action(self, action: NativeTerminalOwnerAction) -> None:
        """Record exact reclaim consumption.

        :param action: Native scheduler authority.
        """

        self.completed_actions.append(action)


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

    publications: list[object]

    def __init__(self) -> None:
        """Create an empty publisher ledger."""

        self.publications = []

    def submit(self, publication: object) -> bool:
        """Record one exact publication.

        :param publication: Immutable gateway handoff.
        :returns: Always ``True`` for a new fixture publication.
        """

        self.publications.append(publication)
        return True


class _Clock:
    """Strictly increasing deterministic nanosecond clock."""

    value: int

    def __init__(self) -> None:
        """Start at one nonzero timestamp."""

        self.value = 100

    def now_ns(self) -> int:
        """Advance and return time.

        :returns: Strictly increasing synthetic nanoseconds.
        """

        self.value += 1
        return self.value


def _identities(
    *,
    local_rank: int = 0,
) -> tuple[
    PackedTerminalSourceIdentityPlan,
    TerminalProcessIdentity,
    TerminalProcessIdentity,
]:
    """Build one TP2 source and TP1 decode identity graph.

    :param local_rank: Source rank selected as local.
    :returns: Source plan, decode coordinator, and canonical publisher.
    """

    key = PackedRequestKey(room_id=71, request_generation=bytes.fromhex("71" * 16))
    source_processes = tuple(
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
    manifest = bytes.fromhex("41" * 32)
    allocation = bytes.fromhex("51" * 32)
    bindings = tuple(
        TerminalRequestBinding(
            request_key=key,
            owner=process,
            rank_manifest_digest=manifest,
            allocation_digest=allocation,
        )
        for process in source_processes
    )
    publication = TerminalPublicationIdentity(
        request_key=key,
        publisher_process_generation=source_processes[0].process_generation,
        publication_generation=bytes.fromhex("61" * 16),
    )
    return (
        PackedTerminalSourceIdentityPlan(
            local_binding=bindings[local_rank],
            source_bindings=bindings,
            publication_identity=publication,
            request_ready_issuer=decoder,
            publisher_issuer=source_processes[0],
        ),
        decoder,
        source_processes[0],
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


def _wiring(
    *,
    local_rank: int = 0,
    failing_metrics: bool = False,
) -> tuple[
    PackedTerminalSourceWiring,
    PackedTerminalSourceIdentityPlan,
    _Runtime,
    _Metrics,
    _Publisher | None,
]:
    """Create one accepted source request and its test owners.

    :param local_rank: Source rank selected as local.
    :param failing_metrics: Whether metrics reject every projection.
    :returns: Wiring, identity, runtime, metrics, and optional publisher.
    """

    identity, _, _ = _identities(local_rank=local_rank)
    runtime = _Runtime()
    metrics = _Metrics(fail=failing_metrics)
    publisher = _Publisher() if local_rank == 0 else None
    clock = _Clock()
    wiring = PackedTerminalSourceWiring(
        runtime=runtime,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=clock.now_ns,
    )
    wiring.accept_submission(_submission(identity))
    return wiring, identity, runtime, metrics, publisher


def test_submission_registers_before_first_native_event() -> None:
    """Bind source identity before accepting its immutable submission."""

    wiring, identity, runtime, metrics, _ = _wiring()
    del wiring
    assert runtime.registrations == [
        (
            identity.local_binding,
            identity.publication_identity,
            identity.trusted_issuers,
        )
    ]
    assert tuple(value.kind for value in runtime.submissions) == (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
    )
    assert tuple(value.event_kind for value in metrics.values) == (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
    )


def test_local_operations_preserve_exact_native_order() -> None:
    """Project producer, gather, outcome, teardown, and ACK events exactly."""

    wiring, identity, runtime, _, _ = _wiring()
    digest = identity.local_binding.digest
    wiring.producer_completed(digest)
    wiring.gather_posted(digest)
    wiring.outcomes_sent(digest)
    wiring.teardown_received(digest, identity.request_ready_issuer)
    wiring.teardown_ack_sent(digest)
    assert tuple(value.kind for value in runtime.submissions) == (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
        NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
    )
    assert runtime.submissions[-2].producer is runtime.control_producer(
        identity.request_ready_issuer.digest
    )


def test_request_ready_submits_native_authority_and_canonical_publication() -> None:
    """Publish only after authenticated request-global readiness."""

    wiring, identity, runtime, _, publisher = _wiring()
    if publisher is None:
        raise AssertionError("canonical fixture omitted its publisher")
    issued = TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=301,
    )
    wiring.request_ready(
        binding_digest=identity.local_binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=identity.request_ready_issuer,
    )
    assert runtime.submissions[-1].kind is NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY
    assert runtime.submissions[-1].receipt == NativeTerminalReceipt.from_wire_receipt(
        issued.wire_receipt
    )
    assert len(publisher.publications) == 1
    assert publisher.publications[0].request_ready_receipt is issued.local_receipt


def test_noncanonical_rank_never_submits_gateway_output() -> None:
    """Retain per-rank readiness without creating duplicate gateway output."""

    wiring, identity, _, _, publisher = _wiring(local_rank=1)
    assert publisher is None
    issued = TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=401,
    )
    wiring.request_ready(
        binding_digest=identity.local_binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=identity.request_ready_issuer,
    )


def test_reclaim_release_precedes_native_consumption_ack() -> None:
    """Keep resources pinned until exact reclaim authority reaches scheduler."""

    wiring, identity, runtime, _, _ = _wiring()
    source_issuer = TerminalWireReceiptIssuer(identity.local_binding.owner)
    issued = source_issuer.issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.RECLAIM_AUTHORIZED,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=501,
    )
    action = NativeTerminalOwnerAction(
        action_id=1,
        kind=NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        binding=NativeTerminalRequestBinding.from_binding(identity.local_binding),
        commit_timestamp_ns=501,
        receipt=NativeTerminalReceipt.from_wire_receipt(issued.wire_receipt),
    )
    observations: list[str] = []

    def release(submission: PackedTerminalSourceSubmission) -> None:
        assert submission.identity is identity
        assert runtime.completed_actions == []
        observations.append("released")

    wiring.consume_reclaim(action, release)
    assert observations == ["released"]
    assert runtime.completed_actions == [action]
    with pytest.raises(RuntimeError, match="consumed twice"):
        wiring.consume_reclaim(action, release)


def test_publisher_success_returns_native_publication_authority() -> None:
    """Join canonical publisher success into native source retirement."""

    wiring, identity, runtime, _, publisher = _wiring()
    if publisher is None:
        raise AssertionError("canonical fixture omitted its publisher")
    ready = TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=601,
    )
    wiring.request_ready(
        binding_digest=identity.local_binding.digest,
        wire_receipt=ready.wire_receipt,
        local_receipt=ready.local_receipt,
        authenticated_issuer=identity.request_ready_issuer,
    )
    publication = publisher.publications[0]
    publisher_issuer = TerminalWireReceiptIssuer(identity.publisher_issuer)
    receipts = tuple(
        publisher_issuer.issue(
            binding=binding,
            kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=701,
        )
        for binding in identity.source_bindings
    )
    wiring.publisher_result(
        TerminalGatewayPublicationSuccess(
            publication=publication,
            completed_ns=701,
            source_receipts=receipts,
        )
    )
    assert runtime.submissions[-1].kind is NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED
    assert runtime.submissions[-1].receipt == NativeTerminalReceipt.from_wire_receipt(
        receipts[0].wire_receipt
    )


def test_publisher_failure_is_process_fatal_not_storage_quarantine() -> None:
    """Project publisher death without fabricating a gateway success receipt."""

    wiring, identity, runtime, _, publisher = _wiring()
    if publisher is None:
        raise AssertionError("canonical fixture omitted its publisher")
    ready = TerminalWireReceiptIssuer(identity.request_ready_issuer).issue(
        binding=identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=801,
    )
    wiring.request_ready(
        binding_digest=identity.local_binding.digest,
        wire_receipt=ready.wire_receipt,
        local_receipt=ready.local_receipt,
        authenticated_issuer=identity.request_ready_issuer,
    )
    wiring.publisher_result(
        TerminalGatewayPublicationFailure(
            publication=publisher.publications[0],
            failed_ns=901,
            reason="synthetic publisher death",
            formatted_traceback="synthetic traceback",
        )
    )
    assert runtime.submissions[-1].kind is NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED
    assert runtime.submissions[-1].receipt is None
    assert runtime.submissions[-1].reason == "synthetic publisher death"


def test_metric_failure_does_not_gate_native_progress() -> None:
    """Keep metrics asynchronous and irrelevant to lifecycle authority."""

    wiring, identity, runtime, metrics, _ = _wiring(failing_metrics=True)
    wiring.producer_completed(identity.local_binding.digest)
    assert metrics.values == []
    assert tuple(value.kind for value in runtime.submissions) == (
        NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
    )


def test_identity_plan_rejects_cross_rank_allocation_drift() -> None:
    """Reject source manifests whose ranks do not bind the same allocation."""

    identity, decoder, publisher = _identities()
    drifted = dataclasses.replace(
        identity.source_bindings[1],
        allocation_digest=bytes.fromhex("ff" * 32),
    )
    with pytest.raises(ValueError, match="disagree on request allocation"):
        PackedTerminalSourceIdentityPlan(
            local_binding=identity.local_binding,
            source_bindings=(identity.source_bindings[0], drifted),
            publication_identity=identity.publication_identity,
            request_ready_issuer=decoder,
            publisher_issuer=publisher,
        )
