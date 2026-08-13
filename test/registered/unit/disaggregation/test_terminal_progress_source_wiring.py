import dataclasses
import hashlib
import select
import sys
import time

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
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
    PackedTerminalSourceProducer,
    PackedTerminalSourceProducerDirectory,
    PackedTerminalSourceSubmission,
    PackedTerminalSourceWiring,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="native terminal owner requires Linux eventfd and timerfd",
)

_LOCAL_PRODUCER_ID = 10
_LOCAL_RECEIPT_PRODUCER_ID = 11
_DECODER_CONTROL_PRODUCER_ID = 12
_DECODER_RECEIPT_PRODUCER_ID = 13
_PUBLISHER_RECEIPT_PRODUCER_ID = 14
_WAIT_SECONDS = 3.0


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
        :returns: Always ``True`` for a new fixture publication.
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


@dataclasses.dataclass(slots=True)
class _Harness:
    """Live native source-wiring integration fixture.

    :ivar owner: Concrete process-lifetime native owner.
    :ivar wiring: Source side-effect dispatcher under test.
    :ivar identity: Exact local source identity graph.
    :ivar producers: Immutable producer authority directory.
    :ivar metrics: Non-gating metric ledger.
    :ivar publisher: Canonical publisher, when local rank zero.
    :ivar clock: Deterministic producer clock.
    """

    owner: NativeTerminalOwner
    wiring: PackedTerminalSourceWiring
    identity: PackedTerminalSourceIdentityPlan
    producers: PackedTerminalSourceProducerDirectory
    metrics: _Metrics
    publisher: _Publisher | None
    clock: _Clock


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


def _producer_directory(
    identity: PackedTerminalSourceIdentityPlan,
) -> PackedTerminalSourceProducerDirectory:
    """Build the complete exact producer map for one source rank.

    :param identity: Rank-local source identity graph.
    :returns: Immutable authority-to-producer directory.
    """

    producers = [
        PackedTerminalSourceProducer(
            producer_id=_LOCAL_PRODUCER_ID,
            name="source-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            authenticated_issuer=None,
        ),
        PackedTerminalSourceProducer(
            producer_id=_LOCAL_RECEIPT_PRODUCER_ID,
            name="source-local-receipt",
            producer_class=NativeTerminalProducerClass.RECEIPT,
            authenticated_issuer=identity.local_binding.owner,
        ),
        PackedTerminalSourceProducer(
            producer_id=_DECODER_CONTROL_PRODUCER_ID,
            name="decode-control",
            producer_class=NativeTerminalProducerClass.CONTROL,
            authenticated_issuer=identity.request_ready_issuer,
        ),
        PackedTerminalSourceProducer(
            producer_id=_DECODER_RECEIPT_PRODUCER_ID,
            name="decode-receipt",
            producer_class=NativeTerminalProducerClass.RECEIPT,
            authenticated_issuer=identity.request_ready_issuer,
        ),
    ]
    if identity.local_binding.owner != identity.publisher_issuer:
        producers.append(
            PackedTerminalSourceProducer(
                producer_id=_PUBLISHER_RECEIPT_PRODUCER_ID,
                name="publisher-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                authenticated_issuer=identity.publisher_issuer,
            )
        )
    return PackedTerminalSourceProducerDirectory(
        local_identity=identity.local_binding.owner,
        producers=tuple(producers),
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


def _start_harness(
    *,
    local_rank: int = 0,
    failing_metrics: bool = False,
) -> _Harness:
    """Start one concrete native owner with an accepted source request.

    :param local_rank: Source rank selected as local.
    :param failing_metrics: Whether metrics reject every projection.
    :returns: Live integration fixture.
    """

    identity, _, _ = _identities(local_rank=local_rank)
    producers = _producer_directory(identity)
    owner = NativeTerminalOwner(
        input_capacity=128,
        output_capacity=128,
        owner_identity=NativeTerminalProcessIdentity.from_identity(
            identity.local_binding.owner
        ),
        testing=True,
    )
    metrics = _Metrics(fail=failing_metrics)
    publisher = _Publisher() if local_rank == 0 else None
    clock = _Clock()
    wiring = PackedTerminalSourceWiring(
        owner=owner,
        producers=producers,
        publisher=publisher,
        metrics_sink=metrics,
        clock_ns=clock.now_ns,
    )
    owner.start()
    wiring.accept_submission(_submission(identity))
    return _Harness(
        owner=owner,
        wiring=wiring,
        identity=identity,
        producers=producers,
        metrics=metrics,
        publisher=publisher,
        clock=clock,
    )


def _wait_for_actions(
    owner: NativeTerminalOwner,
    expected_kinds: tuple[NativeTerminalOwnerActionKind, ...],
) -> tuple[NativeTerminalOwnerAction, ...]:
    """Wait on the real owner eventfd for one exact action population.

    :param owner: Concrete native owner.
    :param expected_kinds: Exact action sequence expected from one event.
    :returns: Native actions in commit order.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    actions: list[NativeTerminalOwnerAction] = []
    while len(actions) < len(expected_kinds):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("native owner did not publish the expected actions")
        readable, _, _ = select.select([owner.output_fileno()], [], [], remaining)
        if len(readable) == 0:
            raise TimeoutError("native owner eventfd did not become readable")
        for output in owner.drain_outputs():
            if output.process_fatal:
                raise RuntimeError(
                    f"native owner became fatal: {output.fatal_code.name}"
                )
            actions.extend(output.actions)
    observed_kinds = tuple(action.kind for action in actions)
    if observed_kinds != expected_kinds:
        raise AssertionError(
            f"native action sequence {observed_kinds} differs from {expected_kinds}"
        )
    return tuple(actions)


def _wait_for_fatal(owner: NativeTerminalOwner) -> NativeTerminalOwnerAction:
    """Wait for the native process-fatal action.

    :param owner: Concrete native owner.
    :returns: Exact process-fatal action.
    """

    deadline = time.monotonic() + _WAIT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("native owner did not enter process-fatal authority")
        readable, _, _ = select.select([owner.output_fileno()], [], [], remaining)
        if len(readable) == 0:
            raise TimeoutError("native owner eventfd did not become readable")
        for output in owner.drain_outputs():
            matching = tuple(
                action
                for action in output.actions
                if action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL
            )
            if len(matching) == 1:
                return matching[0]


def _submit_native_terminal(harness: _Harness) -> None:
    """Submit the native event-channel terminality fixture.

    :param harness: Live source integration fixture.
    """

    harness.owner.submit(
        NativeTerminalOwnerEvent(
            producer_id=harness.producers.producer_id(
                NativeTerminalProducerClass.LOCAL,
                None,
            ),
            binding_digest=harness.identity.local_binding.digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
            enqueued_ns=harness.clock.now_ns(),
        )
    )


def _drive_to_request_ready(
    harness: _Harness,
) -> tuple[NativeTerminalOwnerAction, NativeTerminalOwnerAction]:
    """Drive one request through the action-owned source completion path.

    :param harness: Live source integration fixture.
    :returns: Reclaim and gateway-publication actions.
    """

    digest = harness.identity.local_binding.digest
    harness.wiring.producer_completed(digest)
    (gather_action,) = _wait_for_actions(
        harness.owner,
        (NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,),
    )
    harness.wiring.consume_gather_ready(gather_action, lambda submission: None)
    _submit_native_terminal(harness)
    (outcome_action,) = _wait_for_actions(
        harness.owner,
        (NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,),
    )
    harness.wiring.consume_outcome_ready(outcome_action, lambda submission: None)
    harness.wiring.teardown_received(
        digest,
        harness.identity.request_ready_issuer,
    )
    (ack_action,) = _wait_for_actions(
        harness.owner,
        (NativeTerminalOwnerActionKind.SOURCE_ACK_READY,),
    )
    harness.wiring.consume_ack_ready(ack_action, lambda submission: None)
    ready = TerminalWireReceiptIssuer(harness.identity.request_ready_issuer).issue(
        binding=harness.identity.local_binding,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=harness.clock.now_ns(),
    )
    harness.wiring.request_ready(
        binding_digest=digest,
        wire_receipt=ready.wire_receipt,
        local_receipt=ready.local_receipt,
        authenticated_issuer=harness.identity.request_ready_issuer,
    )
    reclaim, publication = _wait_for_actions(
        harness.owner,
        (
            NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
            NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        ),
    )
    return reclaim, publication


def _publication_result(
    harness: _Harness,
    publication: FrozenTerminalGatewayPublication,
    *,
    success: bool,
) -> TerminalGatewayPublicationSuccess | TerminalGatewayPublicationFailure:
    """Build one authenticated canonical publication result.

    :param harness: Live source integration fixture.
    :param publication: Exact immutable publication attempt.
    :param success: Whether to construct success or functional failure.
    :returns: Complete canonical publisher result.
    """

    publisher_issuer = TerminalWireReceiptIssuer(harness.identity.publisher_issuer)
    kind = TerminalReceiptKind.GATEWAY_PUBLISHED
    outcome = TerminalReceiptOutcome.SUCCESS
    if not success:
        kind = TerminalReceiptKind.FAILURE
        outcome = TerminalReceiptOutcome.FAILURE
    timestamp_ns = harness.clock.now_ns()
    receipts = tuple(
        publisher_issuer.issue(
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


def _close_clean(harness: _Harness) -> None:
    """Exercise ordered producer retirement, quiescence, and exact close.

    :param harness: Fully retired source integration fixture.
    """

    harness.owner.stop_admission()
    harness.wiring.retire_python_producers(_WAIT_SECONDS)
    assert harness.owner.join_producers()
    assert harness.owner.wait_for_output_quiescence(_WAIT_SECONDS)
    harness.owner.close()


def test_producer_directory_requires_exact_frozen_authority() -> None:
    """Reject ambiguous producer ids, issuers, and ordering before startup."""

    identity, _, _ = _identities()
    directory = _producer_directory(identity)
    assert directory.producer_id(NativeTerminalProducerClass.LOCAL, None) == (
        _LOCAL_PRODUCER_ID
    )
    with pytest.raises(RuntimeError, match="no exact authority"):
        directory.producer_id(
            NativeTerminalProducerClass.CONTROL,
            identity.publisher_issuer,
        )
    duplicate = dataclasses.replace(
        directory.producers[-1],
        producer_id=directory.producers[-1].producer_id + 1,
    )
    with pytest.raises(ValueError, match="authority mappings must be unique"):
        PackedTerminalSourceProducerDirectory(
            local_identity=directory.local_identity,
            producers=(*directory.producers, duplicate),
        )


def test_producer_registrations_are_frozen_before_owner_start() -> None:
    """Make post-start source-wiring construction fail at native registration."""

    identity, _, _ = _identities()
    owner = NativeTerminalOwner(
        input_capacity=32,
        output_capacity=32,
        owner_identity=NativeTerminalProcessIdentity.from_identity(
            identity.local_binding.owner
        ),
        testing=True,
    )
    owner.start()
    try:
        with pytest.raises(RuntimeError):
            PackedTerminalSourceWiring(
                owner=owner,
                producers=_producer_directory(identity),
                publisher=_Publisher(),
                metrics_sink=_Metrics(),
                clock_ns=_Clock().now_ns,
            )
    finally:
        owner.abort_and_close()


def test_full_source_success_path_uses_owner_actions_and_retires_cleanly() -> None:
    """Join reclaim and publication before the sole record-removal authority."""

    harness = _start_harness()
    try:
        reclaim_action, publication_action = _drive_to_request_ready(harness)
        publication = harness.wiring.consume_gateway_publication_ready(
            publication_action
        )
        assert publication is not None
        assert harness.publisher is not None
        assert harness.publisher.publications == [publication]
        assert harness.wiring.inventory().pending_publication_action_count == 1
        reclaim_receipt = harness.wiring.consume_reclaim_authorized(
            reclaim_action,
            lambda submission: None,
        )
        assert reclaim_receipt.wire_receipt.kind is (
            TerminalReceiptKind.RECLAIM_CONSUMED
        )
        assert harness.wiring.inventory().active_binding_digests == (
            harness.identity.local_binding.digest,
        )
        harness.wiring.publisher_result(
            _publication_result(harness, publication, success=True)
        )
        (retired_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.REQUEST_RETIRED,),
        )
        retired = harness.wiring.consume_terminal_action(retired_action)
        assert retired == _submission(harness.identity)
        assert harness.wiring.inventory().active_binding_digests == ()
        inventory = harness.owner.inventory()
        assert inventory.safely_retired_count == 1
        assert inventory.pending_action_count == 0
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.NONE
        _close_clean(harness)
    finally:
        harness.owner.abort_and_close()


def test_publication_action_remains_pending_until_authenticated_result() -> None:
    """Do not acknowledge publication authority merely because enqueue succeeded."""

    harness = _start_harness()
    try:
        reclaim_action, publication_action = _drive_to_request_ready(harness)
        harness.wiring.consume_reclaim_authorized(
            reclaim_action,
            lambda submission: None,
        )
        publication = harness.wiring.consume_gateway_publication_ready(
            publication_action
        )
        assert publication is not None
        assert harness.owner.inventory().pending_action_count == 1
        harness.wiring.publisher_result(
            _publication_result(harness, publication, success=True)
        )
        assert harness.wiring.inventory().pending_publication_action_count == 0
        (retired_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.REQUEST_RETIRED,),
        )
        harness.wiring.consume_terminal_action(retired_action)
        _close_clean(harness)
    finally:
        harness.owner.abort_and_close()


def test_noncanonical_rank_consumes_its_own_publication_receipt() -> None:
    """Require rank-local authority instead of inferring canonical completion."""

    harness = _start_harness(local_rank=1)
    try:
        reclaim_action, publication_action = _drive_to_request_ready(harness)
        assert (
            harness.wiring.consume_gateway_publication_ready(publication_action) is None
        )
        harness.wiring.consume_reclaim_authorized(
            reclaim_action,
            lambda submission: None,
        )
        canonical_identity, _, _ = _identities(local_rank=0)
        canonical_publication = FrozenTerminalGatewayPublication(
            identity=harness.identity.publication_identity,
            canonical_binding=harness.identity.source_bindings[0],
            source_bindings=harness.identity.source_bindings,
            request_ready_receipt=TerminalWireReceiptIssuer(
                harness.identity.request_ready_issuer
            )
            .issue(
                binding=canonical_identity.local_binding,
                kind=TerminalReceiptKind.REQUEST_READY,
                outcome=TerminalReceiptOutcome.SUCCESS,
                terminal_timestamp_ns=harness.clock.now_ns(),
            )
            .local_receipt,
            output_projection=_Projection(payload=b"output"),
            enqueued_ns=harness.clock.now_ns(),
        )
        harness.wiring.publisher_result(
            _publication_result(harness, canonical_publication, success=True)
        )
        (retired_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.REQUEST_RETIRED,),
        )
        harness.wiring.consume_terminal_action(retired_action)
        _close_clean(harness)
    finally:
        harness.owner.abort_and_close()


def test_functional_publication_failure_quarantines_identity_after_reclaim() -> None:
    """Keep storage reusable while retaining failed publication identity."""

    harness = _start_harness()
    try:
        reclaim_action, publication_action = _drive_to_request_ready(harness)
        harness.wiring.consume_reclaim_authorized(
            reclaim_action,
            lambda submission: None,
        )
        publication = harness.wiring.consume_gateway_publication_ready(
            publication_action
        )
        assert publication is not None
        harness.wiring.publisher_result(
            _publication_result(harness, publication, success=False)
        )
        (quarantined_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,),
        )
        assert harness.wiring.consume_terminal_action(quarantined_action) is None
        inventory = harness.wiring.inventory()
        assert inventory.active_binding_digests == (
            harness.identity.local_binding.digest,
        )
        assert inventory.quarantined_binding_digests == (
            harness.identity.local_binding.digest,
        )
        native_inventory = harness.owner.inventory()
        assert native_inventory.safely_retired_count == 0
        assert native_inventory.quarantined_count == 1
    finally:
        harness.owner.abort_and_close()


def test_publisher_thread_death_enters_process_fatal_native_lifecycle() -> None:
    """Keep publisher death distinct from authenticated functional failure."""

    harness = _start_harness()
    try:
        harness.wiring.publisher_died(
            harness.identity.local_binding.digest,
            "synthetic publisher death",
        )
        fatal_action = _wait_for_fatal(harness.owner)
        assert fatal_action.binding.digest == harness.identity.local_binding.digest
        inventory = harness.owner.inventory()
        assert inventory.fatal_code is NativeTerminalOwnerFatalCode.DEPENDENCY_DEATH
        assert inventory.quarantined_count == 1
    finally:
        harness.owner.abort_and_close()


def test_duplicate_and_wrong_kind_actions_run_no_second_side_effect() -> None:
    """Reject stale or mistyped actions before touching serving resources."""

    harness = _start_harness()
    calls: list[str] = []
    try:
        harness.wiring.producer_completed(harness.identity.local_binding.digest)
        (gather_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,),
        )
        with pytest.raises(ValueError, match="SOURCE_OUTCOME_READY"):
            harness.wiring.consume_outcome_ready(
                gather_action,
                lambda submission: calls.append("wrong"),
            )
        harness.wiring.consume_gather_ready(
            gather_action,
            lambda submission: calls.append("gather"),
        )
        with pytest.raises(RuntimeError, match="delivered twice"):
            harness.wiring.consume_gather_ready(
                gather_action,
                lambda submission: calls.append("duplicate"),
            )
        assert calls == ["gather"]
    finally:
        harness.owner.abort_and_close()


def test_side_effect_failure_enters_native_fail_closed_action_delivery() -> None:
    """Turn an accepted-but-undeliverable action into process-fatal authority."""

    harness = _start_harness()
    try:
        harness.wiring.producer_completed(harness.identity.local_binding.digest)
        (gather_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,),
        )

        def fail_gather(submission: PackedTerminalSourceSubmission) -> None:
            raise RuntimeError("synthetic gather failure")

        with pytest.raises(RuntimeError, match="synthetic gather failure"):
            harness.wiring.consume_gather_ready(gather_action, fail_gather)
        _wait_for_fatal(harness.owner)
        assert harness.owner.inventory().fatal_code is (
            NativeTerminalOwnerFatalCode.OUTPUT_QUEUE_OVERFLOW
        )
    finally:
        harness.owner.abort_and_close()


def test_metrics_failure_never_gates_native_progress() -> None:
    """Keep observability failure outside lifecycle authority."""

    harness = _start_harness(failing_metrics=True)
    try:
        harness.wiring.producer_completed(harness.identity.local_binding.digest)
        (gather_action,) = _wait_for_actions(
            harness.owner,
            (NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,),
        )
        assert gather_action.binding.digest == harness.identity.local_binding.digest
        assert harness.metrics.values == []
        assert harness.owner.inventory().fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        harness.owner.abort_and_close()


def test_request_ready_rejects_wrong_issuer_before_native_submission() -> None:
    """Reject valid-looking authority from an unauthenticated producer route."""

    harness = _start_harness()
    try:
        wrong_issuer = harness.identity.publisher_issuer
        ready = TerminalWireReceiptIssuer(wrong_issuer).issue(
            binding=harness.identity.local_binding,
            kind=TerminalReceiptKind.REQUEST_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=harness.clock.now_ns(),
        )
        with pytest.raises(RuntimeError, match="authenticated another issuer"):
            harness.wiring.request_ready(
                binding_digest=harness.identity.local_binding.digest,
                wire_receipt=ready.wire_receipt,
                local_receipt=ready.local_receipt,
                authenticated_issuer=wrong_issuer,
            )
        assert harness.owner.inventory().fatal_code is NativeTerminalOwnerFatalCode.NONE
    finally:
        harness.owner.abort_and_close()
