import dataclasses
import enum
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sglang.srt.disaggregation.terminal_progress.cuda_owner_producer import (
    TerminalCudaCompletionProducer,
)
from sglang.srt.disaggregation.terminal_progress.evidence import (
    TerminalProgressTimingRecorder,
    terminal_progress_timing_recorder,
)
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlTerminalResult,
    GroupedNixlTransferMember,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerObservation,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalPublicationIdentity,
    NativeTerminalReceipt,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerTimingField,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
    FrozenTerminalGatewayPublication,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationResult,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceipt,
    TerminalWireReceiptIssuer,
)

logger = logging.getLogger(__name__)


class PackedTerminalSourceCancellationDisposition(enum.StrEnum):
    """Scheduler cancellation disposition for one published source lifecycle."""

    COMPLETION_REQUIRED = "completion_required"
    ALREADY_RECORDED = "already_recorded"
    TOO_LATE_FOR_ROLLBACK = "too_late_for_rollback"


@runtime_checkable
class NativeTerminalSourceRuntime(Protocol):
    """Process-lifetime native runtime surface consumed by source wiring."""

    def python_producer_id(
        self,
        producer_class: NativeTerminalProducerClass,
        authenticated_issuer: NativeTerminalProcessIdentity | None = None,
    ) -> int:
        """Resolve one pre-registered Python producer authority.

        :param producer_class: Required local, control, or receipt authority.
        :param authenticated_issuer: Route-authenticated issuer when required.
        :returns: Exact registered producer identity.
        """

    def register_lifecycle(
        self,
        registration: NativeTerminalLifecycleRegistration,
    ) -> None:
        """Register one source lifecycle before its first event.

        :param registration: Complete native source registration.
        """

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
        """Submit one producer-ordered event to the native owner.

        :param producer_id: Exact registered producer identity.
        :param binding_digest: Exact lifecycle lookup identity.
        :param kind: Closed native event kind.
        :param receipt: Authenticated authority when required.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def submit_imported_receipt(
        self,
        producer_id: int,
        receipt: NativeTerminalReceipt,
        kind: NativeTerminalOwnerEventKind,
        *,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit route-authenticated imported receipt authority.

        :param producer_id: Producer bound to the receipt issuer.
        :param receipt: Exact imported native receipt.
        :param kind: Receipt-consuming source transition.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

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
        """Complete one source or publication work action exactly once.

        :param producer_id: Producer owning the continuation transition.
        :param action: Exact action previously drained from the runtime.
        :param followup_kind: Exact success or failure transition.
        :param receipt: Publication authority when required.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def complete_scheduler_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        completion_receipt: NativeTerminalReceipt | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Complete scheduler-affine reclaim under exact authority.

        :param producer_id: Local receipt producer identity.
        :param action: Exact reclaim action previously drained.
        :param followup_kind: Reclaim-consumed transition.
        :param completion_receipt: Scheduler-minted consumption authority.
        :param enqueued_ns: Optional exact native-clock timestamp.
        """

    def fail_scheduler_action(
        self,
        action: NativeTerminalOwnerAction,
        reason: str,
    ) -> None:
        """Fail one ambiguous scheduler action without losing its authority.

        :param action: Exact scheduler action which could not complete.
        :param reason: Stable process-fatal failure evidence.
        """

    def acknowledge_consumed_action(self, action: NativeTerminalOwnerAction) -> None:
        """Release runtime accounting after terminal action consumption.

        :param action: Exact terminal action accepted by source wiring.
        """


@runtime_checkable
class PackedTerminalSourceMetricsSink(Protocol):
    """Non-gating sink for source lifecycle timing projections."""

    def emit(self, metric: "PackedTerminalSourceMetric") -> None:
        """Consume one immutable exactly-once metric.

        :param metric: Source lifecycle event timing.
        """


@runtime_checkable
class PackedTerminalSourcePublisher(Protocol):
    """Exactly-once publisher boundary used by canonical source rank zero."""

    def submit(self, publication: FrozenTerminalGatewayPublication) -> bool:
        """Accept one immutable request-global publication.

        :param publication: Exact output handoff earned by native readiness.
        :returns: Whether a new publication was enqueued.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceSubmission:
    """Immutable handoff accepted before another forward may own progress.

    :ivar identity: Complete source binding and publication identity.
    :ivar output_projection: Pre-forward shell and stable producer result slot.
    :ivar producer_event_generation: Exact event generation covering the slot.
    :ivar producer_stream_handle: Stream tail immediately following that event.
    :ivar transport_submission: Opaque immutable packed transport submission.
    """

    identity: PackedTerminalSourceIdentityPlan
    output_projection: FrozenTerminalGatewayOutputProjection | None
    producer_event_generation: bytes
    producer_stream_handle: int
    transport_submission: object

    def __post_init__(self) -> None:
        """Validate immutable submission identities without duck typing."""

        if type(self.identity) is not PackedTerminalSourceIdentityPlan:
            raise TypeError("identity must be PackedTerminalSourceIdentityPlan")
        canonical_rank = self.identity.local_binding.owner.tp_rank == 0
        if canonical_rank and not isinstance(
            self.output_projection,
            FrozenTerminalGatewayOutputProjection,
        ):
            raise TypeError("canonical source requires an output projection")
        if not canonical_rank and self.output_projection is not None:
            raise ValueError("noncanonical source cannot own an output projection")
        if type(self.producer_event_generation) is not bytes:
            raise TypeError("producer_event_generation must be bytes")
        if len(self.producer_event_generation) != 16:
            raise ValueError("producer_event_generation must contain 16 bytes")
        if (
            type(self.producer_stream_handle) is not int
            or self.producer_stream_handle < 0
        ):
            raise ValueError("producer_stream_handle must be a non-negative integer")
        if self.transport_submission is None:
            raise ValueError("transport_submission must not be None")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceMetric:
    """Exactly-once non-authoritative projection of one source operation.

    :ivar binding_digest: Exact source lifecycle identity.
    :ivar event_kind: Native event represented by the operation.
    :ivar timestamp_ns: Local monotonic operation timestamp.
    """

    binding_digest: bytes
    event_kind: NativeTerminalOwnerEventKind
    timestamp_ns: int

    def __post_init__(self) -> None:
        """Validate one bounded source metric projection."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if type(self.event_kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("event_kind must be NativeTerminalOwnerEventKind")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceInventory:
    """Immutable serving-facing source side-effect inventory.

    :ivar active_binding_digests: Live identities retaining source records.
    :ivar quarantined_binding_digests: Fail-closed publication identities.
    :ivar completion_required_binding_digests: Client-cancelled identities whose
        already-published terminal protocol must still complete.
    :ivar active_result_slot_binding_digests: Canonical live pinned result slots.
    :ivar quarantined_result_slot_binding_digests: Pinned slots retained under
        request quarantine.
    :ivar pending_publication_action_count: Native actions held during publication.
    """

    active_binding_digests: tuple[bytes, ...]
    quarantined_binding_digests: tuple[bytes, ...]
    completion_required_binding_digests: tuple[bytes, ...]
    active_result_slot_binding_digests: tuple[bytes, ...]
    quarantined_result_slot_binding_digests: tuple[bytes, ...]
    pending_publication_action_count: int

    def __post_init__(self) -> None:
        """Validate deterministic identity and action counts."""

        identity_sets = (
            self.active_binding_digests,
            self.quarantined_binding_digests,
            self.completion_required_binding_digests,
            self.active_result_slot_binding_digests,
            self.quarantined_result_slot_binding_digests,
        )
        if any(type(values) is not tuple for values in identity_sets):
            raise TypeError("inventory identity collections must be tuples")
        if any(
            type(value) is not bytes or len(value) != 32
            for values in identity_sets
            for value in values
        ):
            raise ValueError("inventory identities must contain 32 bytes")
        if any(tuple(sorted(values)) != values for values in identity_sets):
            raise ValueError("inventory identities must use digest order")
        if not set(self.quarantined_binding_digests).issubset(
            self.active_binding_digests
        ):
            raise ValueError("quarantined identities must remain active")
        if not set(self.completion_required_binding_digests).issubset(
            self.active_binding_digests
        ):
            raise ValueError("completion-required identities must remain active")
        if not set(self.active_result_slot_binding_digests).issubset(
            self.active_binding_digests
        ):
            raise ValueError("active result slots must remain source-active")
        if not set(self.quarantined_result_slot_binding_digests).issubset(
            self.active_result_slot_binding_digests
        ):
            raise ValueError("quarantined result slots must remain slot-active")
        if (
            type(self.pending_publication_action_count) is not int
            or self.pending_publication_action_count < 0
        ):
            raise ValueError("pending publication action count must be non-negative")


@dataclasses.dataclass(slots=True)
class _SourceRecord:
    """Side-effect inventory keyed by native-owned lifecycle identity."""

    submission: PackedTerminalSourceSubmission
    lifecycle_published: bool = False
    submission_commit_observed: bool = False
    request_ready_receipt: TerminalReceipt | None = None
    request_failure_receipt: TerminalReceipt | None = None
    publication_action: NativeTerminalOwnerAction | None = None
    publication_submitted: bool = False
    reclaim_consumed: bool = False
    publication_terminal: bool = False
    publication_failed: bool = False
    quarantined: bool = False
    cancellation_reason: str | None = None
    producer_completion_attachment_started: bool = False
    producer_completion_attached: bool = False
    packed_transfer_ready: bool = False
    producer_delivery_authorized: bool = False
    claimed_action_ids: set[int] = dataclasses.field(default_factory=set)


class PackedTerminalSourceWiring:
    """Dispatch source side effects through one process-lifetime runtime.

    The native runtime alone owns the reducer, producer namespaces, native
    action routing, and conservation. This component retains immutable serving
    payloads and performs only side effects explicitly earned by runtime actions.
    """

    _runtime: NativeTerminalSourceRuntime
    _cuda_completion: TerminalCudaCompletionProducer
    _local_identity: TerminalProcessIdentity
    _local_producer_id: int
    _local_receipt_producer_id: int
    _reclaim_issuer: TerminalWireReceiptIssuer
    _publisher: PackedTerminalSourcePublisher | None
    _metrics_sink: PackedTerminalSourceMetricsSink
    _clock_ns: Callable[[], int]
    _timing: TerminalProgressTimingRecorder
    _records: dict[bytes, _SourceRecord]
    _emitted_metrics: set[tuple[bytes, NativeTerminalOwnerEventKind]]
    _lock: threading.Lock

    def __init__(
        self,
        *,
        runtime: NativeTerminalSourceRuntime,
        cuda_completion: TerminalCudaCompletionProducer,
        local_identity: TerminalProcessIdentity,
        publisher: PackedTerminalSourcePublisher | None,
        metrics_sink: PackedTerminalSourceMetricsSink,
        clock_ns: Callable[[], int],
    ) -> None:
        """Construct source wiring over the sole process-lifetime runtime.

        :param runtime: Sole authoritative native lifecycle runtime.
        :param cuda_completion: Direct source callback-to-owner producer.
        :param local_identity: Exact source process owned by this wiring.
        :param publisher: Canonical-rank publisher, otherwise ``None``.
        :param metrics_sink: Non-gating metric projection sink.
        :param clock_ns: Local monotonic nanosecond clock.
        """

        if not isinstance(runtime, NativeTerminalSourceRuntime):
            raise TypeError("runtime must satisfy NativeTerminalSourceRuntime")
        if not isinstance(cuda_completion, TerminalCudaCompletionProducer):
            raise TypeError(
                "cuda_completion must satisfy TerminalCudaCompletionProducer"
            )
        if type(local_identity) is not TerminalProcessIdentity:
            raise TypeError("local_identity must be TerminalProcessIdentity")
        if local_identity.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("local_identity must belong to a source owner")
        if publisher is not None and not isinstance(
            publisher,
            PackedTerminalSourcePublisher,
        ):
            raise TypeError("publisher must satisfy PackedTerminalSourcePublisher")
        if not isinstance(metrics_sink, PackedTerminalSourceMetricsSink):
            raise TypeError("metrics_sink must satisfy PackedTerminalSourceMetricsSink")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if local_identity.tp_rank == 0 and publisher is None:
            raise ValueError("canonical source rank requires a gateway publisher")
        if local_identity.tp_rank != 0 and publisher is not None:
            raise ValueError("only canonical source rank may own a gateway publisher")
        native_identity = NativeTerminalProcessIdentity.from_identity(local_identity)
        self._runtime = runtime
        self._cuda_completion = cuda_completion
        self._local_identity = local_identity
        self._local_producer_id = runtime.python_producer_id(
            NativeTerminalProducerClass.LOCAL
        )
        self._local_receipt_producer_id = runtime.python_producer_id(
            NativeTerminalProducerClass.RECEIPT,
            native_identity,
        )
        self._reclaim_issuer = TerminalWireReceiptIssuer(local_identity)
        self._publisher = publisher
        self._metrics_sink = metrics_sink
        self._clock_ns = clock_ns
        self._timing = terminal_progress_timing_recorder(logger, clock_ns)
        self._records = {}
        self._emitted_metrics = set()
        self._lock = threading.Lock()

    def accept_submission(self, submission: PackedTerminalSourceSubmission) -> None:
        """Register and accept one immutable source request exactly once.

        :param submission: Launch-bound transport and output handoff.
        """

        if type(submission) is not PackedTerminalSourceSubmission:
            raise TypeError("submission must be PackedTerminalSourceSubmission")
        identity = submission.identity
        binding = identity.local_binding
        if binding.owner != self._local_identity:
            raise RuntimeError("source submission belongs to another runtime")
        digest = binding.digest
        record = _SourceRecord(submission=submission)
        with self._lock:
            if digest in self._records:
                raise RuntimeError("source submission identity was reused")
            # Once accepted, this record is retained on every downstream error;
            # silently removing it could make scheduler-owned pages look reusable.
            self._records[digest] = record
        registration = NativeTerminalLifecycleRegistration(
            binding=NativeTerminalRequestBinding.from_binding(binding),
            publication_identity=NativeTerminalPublicationIdentity.from_identity(
                identity.publication_identity
            ),
            trusted_issuers=tuple(
                NativeTerminalProcessIdentity.from_identity(issuer)
                for issuer in identity.trusted_issuers
            ),
        )
        self._runtime.register_lifecycle(registration)
        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("source registry changed during lifecycle publish")
            current.lifecycle_published = True
        self._runtime.submit(
            self._local_producer_id,
            digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            enqueued_ns=self._clock_ns(),
        )
        self._emit_metric_once(
            digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        )

    def attach_producer_completion(
        self,
        submission: PackedTerminalSourceSubmission,
    ) -> None:
        """Attach one callback after the actor has published PREPARE.

        The attachment attempt becomes irreversible before entering CUDA. A
        partial arm or submit failure therefore cannot be retried against the
        same lifecycle and must enter the caller's process-fatal path.

        :param submission: Exact accepted source submission.
        """

        if type(submission) is not PackedTerminalSourceSubmission:
            raise TypeError("submission must be PackedTerminalSourceSubmission")
        digest = submission.identity.local_binding.digest
        record = self._record(digest)
        if record.submission is not submission:
            raise RuntimeError("source completion targets another submission")
        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("source record changed before callback attachment")
            if current.producer_completion_attachment_started:
                raise RuntimeError("source completion callback was already attempted")
            current.producer_completion_attachment_started = True
        self._cuda_completion.arm(digest)
        self._cuda_completion.submit(submission.producer_stream_handle, digest)
        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("source record changed during callback attachment")
            current.producer_completion_attached = True
            authorize = current.packed_transfer_ready
            if authorize:
                if current.producer_delivery_authorized:
                    raise RuntimeError("source producer delivery was already authorized")
                current.producer_delivery_authorized = True
        if authorize:
            self._cuda_completion.authorize_delivery(digest)

    def submission_committed(self, observation: NativeTerminalOwnerObservation) -> None:
        """Project one exact native submission commit into evidence.

        :param observation: Actionless native commit observation.
        """

        if type(observation) is not NativeTerminalOwnerObservation:
            raise TypeError("observation must be NativeTerminalOwnerObservation")
        binding = observation.binding.to_binding()
        expected_owner = NativeTerminalProcessIdentity.from_identity(
            self._local_identity
        )
        if observation.binding.owner != expected_owner:
            raise RuntimeError("submission observation belongs to another source rank")
        if observation.producer_id != self._local_producer_id:
            raise RuntimeError("submission observation belongs to another producer")
        digest = binding.digest
        record = self._record(digest)
        if binding != record.submission.identity.local_binding:
            raise RuntimeError("submission observation differs from its source binding")
        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("source record changed during commit observation")
            if current.submission_commit_observed:
                raise RuntimeError("source submission commit was observed twice")
            current.submission_commit_observed = True
        self._timing.emit_interval(
            binding=binding,
            field=TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF,
            sample_key=f"source-rank-{binding.owner.tp_rank}",
            started_ns=observation.enqueued_ns,
            completed_ns=observation.completed_ns,
        )

    def lifecycle_published(self, binding_digest: bytes) -> bool:
        """Return whether a lifecycle crossed the native publication boundary.

        :param binding_digest: Exact source lifecycle identity.
        :returns: Whether runtime registration completed.
        """

        return self._record(binding_digest).lifecycle_published

    def cancel_unpublished(
        self, binding_digest: bytes
    ) -> PackedTerminalSourceSubmission:
        """Remove one submission which never reached native registration.

        :param binding_digest: Exact unpublished source lifecycle identity.
        :returns: Immutable submission released from the source registry.
        """

        record = self._record(binding_digest)
        with self._lock:
            current = self._records.get(binding_digest)
            if current is not record:
                raise RuntimeError("source registry changed during cancellation")
            if current.lifecycle_published:
                raise RuntimeError("published source lifecycle cannot be cancelled")
            del self._records[binding_digest]
        return record.submission

    def packed_ready(self, binding_digest: bytes) -> bool:
        """Join authenticated decoder allocation with producer completion.

        The native CUDA producer retains a completed callback until this
        authorization arrives. Exact duplicate ``PackedReady`` delivery is
        idempotent because the transport actor authenticates and de-duplicates
        the message before this lifecycle join. A conflicting message never
        reaches this boundary.

        :param binding_digest: Exact accepted source binding.
        :returns: Whether this input published producer completion to native state.
        """

        record = self._record(binding_digest)
        with self._lock:
            current = self._records.get(binding_digest)
            if current is not record:
                raise RuntimeError("source record changed during PackedReady delivery")
            if current.packed_transfer_ready:
                return False
            current.packed_transfer_ready = True
            authorize = current.producer_completion_attached
            if authorize:
                if current.producer_delivery_authorized:
                    raise RuntimeError("source producer delivery was already authorized")
                current.producer_delivery_authorized = True
        if not authorize:
            return False
        return self._cuda_completion.authorize_delivery(binding_digest)

    def cancel_request(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> PackedTerminalSourceCancellationDisposition:
        """Record client cancellation without revoking published ownership.

        The scheduler exposes a request through its terminal registry only
        after the same scheduler-thread call has published PREPARE. Rollback is
        therefore no longer legal when this method can be reached. Recording
        intent leaves every native action and transfer authority intact so the
        lifecycle retires through its ordinary reclaim/publication join. The
        gateway independently drops output for a disconnected client.

        :param binding: Exact scheduler-retained source generation.
        :param reason: Stable client-cancellation reason.
        :returns: Whether completion was newly required, already recorded, or
            request-global readiness had already committed.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be a TerminalRequestBinding")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        record = self._record(binding.digest)
        if record.submission.identity.local_binding != binding:
            raise RuntimeError("source cancellation targets another binding")
        with self._lock:
            current = self._records.get(binding.digest)
            if current is not record:
                raise RuntimeError("source registry changed during cancellation")
            if current.cancellation_reason is not None:
                return PackedTerminalSourceCancellationDisposition.ALREADY_RECORDED
            if current.request_ready_receipt is not None:
                return (
                    PackedTerminalSourceCancellationDisposition.TOO_LATE_FOR_ROLLBACK
                )
            current.cancellation_reason = reason
        return PackedTerminalSourceCancellationDisposition.COMPLETION_REQUIRED

    def grouped_native_terminal(self, result: GroupedNixlTerminalResult) -> None:
        """Commit one request-level grouped NIXL terminal result.

        Main KV and the canonical DFlash boundary have independent native
        handles. Their grouped owner emits success only after the complete
        predeclared member set reaches terminality, or emits failure at the
        first failed member, so the native reducer sees exactly one source
        transition.

        :param result: Exact aggregate result claimed from the source channel.
        """

        if type(result) is not GroupedNixlTerminalResult:
            raise TypeError("result must be GroupedNixlTerminalResult")
        self._record(result.binding_digest)
        kind = NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL
        reason = None
        if not result.successful:
            kind = NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED
            reason = result.reason
        self._runtime.submit(
            self._local_producer_id,
            result.binding_digest,
            kind,
            reason=reason,
            enqueued_ns=result.native_timestamp_ns,
        )
        binding = self._record(result.binding_digest).submission.identity.local_binding
        for timing in result.member_timings:
            sample_key = f"main:writer-{binding.owner.tp_rank}"
            if timing.member is GroupedNixlTransferMember.DFLASH_BOUNDARY:
                sample_key = "boundary:writer-0"
            self._timing.emit_interval(
                binding=binding,
                field=TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY,
                sample_key=sample_key,
                started_ns=timing.post_started_ns,
                completed_ns=timing.native_terminal_ns,
            )
        self._emit_metric_once(result.binding_digest, kind)

    def consume_gather_ready(
        self,
        action: NativeTerminalOwnerAction,
        post_gather: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
    ) -> None:
        """Post gather only under exact native gather authority.

        :param action: One-shot ``SOURCE_GATHER_READY`` action.
        :param post_gather: Nonblocking gather and transfer-post side effect.
        """

        self._consume_followup_action(
            action=action,
            expected_kind=NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
            side_effect=post_gather,
            followup_kind=NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
            failure_label="source gather publication failed",
        )

    def consume_outcome_ready(
        self,
        action: NativeTerminalOwnerAction,
        send_outcomes: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
    ) -> None:
        """Send immutable outcomes only under exact native authority.

        :param action: One-shot ``SOURCE_OUTCOME_READY`` action.
        :param send_outcomes: Authenticated outcome-send side effect.
        """

        self._consume_followup_action(
            action=action,
            expected_kind=NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
            side_effect=send_outcomes,
            followup_kind=NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
            failure_label="source outcome publication failed",
        )

    def teardown_received(
        self,
        binding_digest: bytes,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Commit authenticated decode teardown delivery.

        :param binding_digest: Exact accepted source binding.
        :param authenticated_issuer: Process identity proved by control routing.
        """

        record = self._record(binding_digest)
        if authenticated_issuer != record.submission.identity.request_ready_issuer:
            raise RuntimeError("teardown issuer differs from the trusted decoder")
        self._submit_event(
            producer_class=NativeTerminalProducerClass.CONTROL,
            authenticated_issuer=authenticated_issuer,
            binding_digest=binding_digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        )
        self._emit_metric_once(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        )

    def consume_ack_ready(
        self,
        action: NativeTerminalOwnerAction,
        send_ack: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
    ) -> None:
        """Send the exact teardown ACK only under native authority.

        :param action: One-shot ``SOURCE_ACK_READY`` action.
        :param send_ack: Authenticated ACK-send side effect.
        """

        self._consume_followup_action(
            action=action,
            expected_kind=NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
            side_effect=send_ack,
            followup_kind=NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
            failure_label="source teardown acknowledgement failed",
        )

    def request_ready(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Store authenticated request readiness and submit it to native state.

        Publication and reclaim remain forbidden until the native owner returns
        their independent one-shot actions.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport authority submitted to native state.
        :param local_receipt: Matching process-local publisher authority.
        :param authenticated_issuer: Process identity proved by control routing.
        """

        self._submit_request_terminal_receipt(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
            event_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
            reason=None,
        )

    def request_failed(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        reason: str,
    ) -> None:
        """Store authenticated request failure and submit it to native state.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport failure authority submitted to native state.
        :param local_receipt: Matching process-local failure authority.
        :param authenticated_issuer: Process identity proved by control routing.
        :param reason: Stable request-global failure evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        self._submit_request_terminal_receipt(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
            event_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
            reason=reason,
        )

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
        release_resources: Callable[[PackedTerminalSourceSubmission], None],
    ) -> IssuedTerminalWireReceipt:
        """Release source resources and return exact consumption authority.

        :param action: Native one-shot reclaim authorization.
        :param release_resources: Scheduler-thread resource release operation.
        :returns: Locally issued ``RECLAIM_CONSUMED`` receipt.
        """

        if not callable(release_resources):
            raise TypeError("release_resources must be callable")
        record = self._claim_action(
            action,
            NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        )
        try:
            release_resources(record.submission)
            issued = self._reclaim_issuer.issue(
                binding=record.submission.identity.local_binding,
                kind=TerminalReceiptKind.RECLAIM_CONSUMED,
                outcome=TerminalReceiptOutcome.SUCCESS,
                terminal_timestamp_ns=self._clock_ns(),
            )
            self._runtime.complete_scheduler_action(
                self._local_receipt_producer_id,
                action,
                NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
                completion_receipt=NativeTerminalReceipt.from_wire_receipt(
                    issued.wire_receipt
                ),
                enqueued_ns=self._clock_ns(),
            )
        except Exception as error:
            self._fail_scheduler_action(
                action,
                "source reclaim consumption failed",
                error,
            )
            raise
        with self._lock:
            current = self._records.get(action.binding.digest)
            if current is not record:
                raise RuntimeError("source request registry changed during reclaim")
            current.reclaim_consumed = True
        self._emit_metric_once(
            action.binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
        )
        return issued

    def consume_gateway_publication_ready(
        self,
        action: NativeTerminalOwnerAction,
    ) -> FrozenTerminalGatewayPublication | None:
        """Retain publication authority and enqueue only on canonical rank.

        :param action: Native one-shot gateway-publication action.
        :returns: Canonical immutable publication, otherwise ``None``.
        """

        record = self._claim_action(
            action,
            NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
        )
        ready_receipt = record.request_ready_receipt
        if ready_receipt is None:
            error = RuntimeError("request-ready receipt is absent")
            self._complete_failed_work(
                action=action,
                producer_id=self._local_producer_id,
                followup_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                label="gateway publication preceded stored request readiness",
                error=error,
            )
            raise error
        with self._lock:
            if record.publication_action is not None:
                raise RuntimeError("source publication action was retained twice")
            # This action stays native-pending until the publisher returns an
            # authenticated result. Acknowledging at enqueue would permit a
            # publication timeout to lose its fail-closed owner.
            record.publication_action = action
        if self._local_identity.tp_rank != 0:
            return None
        publisher = self._publisher
        if publisher is None:
            error = RuntimeError("gateway publisher is absent")
            self._complete_failed_work(
                action=action,
                producer_id=self._local_producer_id,
                followup_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                label="canonical source rank has no gateway publisher",
                error=error,
            )
            self._mark_publication_action_failed(record, action)
            raise error
        submission = record.submission
        identity = submission.identity
        output_projection = submission.output_projection
        if output_projection is None:
            raise RuntimeError("canonical source output projection disappeared")
        try:
            publication = FrozenTerminalGatewayPublication(
                identity=identity.publication_identity,
                canonical_binding=identity.local_binding,
                source_bindings=identity.source_bindings,
                request_ready_receipt=ready_receipt,
                output_projection=output_projection,
                enqueued_ns=self._clock_ns(),
            )
            self._timing.capture(
                binding=identity.local_binding,
                field=TerminalOwnerTimingField.GATEWAY_PUBLICATION,
                sample_key="canonical-source-publisher",
                started_ns=publication.enqueued_ns,
            )
            with self._lock:
                record.publication_submitted = True
            accepted = publisher.submit(publication)
            if not accepted:
                raise RuntimeError("gateway publisher rejected a new publication")
        except Exception as error:
            self._complete_failed_work(
                action=action,
                producer_id=self._local_producer_id,
                followup_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                label="gateway publication enqueue failed",
                error=error,
            )
            self._mark_publication_action_failed(record, action)
            raise
        return publication

    def publisher_result(self, result: TerminalGatewayPublicationResult) -> None:
        """Project one publisher result back into native lifecycle authority.

        :param result: Exactly-once publisher success or functional failure.
        """

        if type(result) not in (
            TerminalGatewayPublicationSuccess,
            TerminalGatewayPublicationFailure,
        ):
            raise TypeError("result must be a terminal gateway publication result")
        publication = result.publication
        record, local_issued_receipt = self._local_publication_result(
            publication,
            result.source_receipts,
        )
        reason = None
        if type(result) is TerminalGatewayPublicationFailure:
            reason = result.reason
        self._consume_publication_receipt(
            record=record,
            wire_receipt=local_issued_receipt.wire_receipt,
            local_receipt=local_issued_receipt.local_receipt,
            authenticated_issuer=publication.canonical_binding.owner,
            reason=reason,
        )
        if type(result) is TerminalGatewayPublicationSuccess:
            self._timing.complete(
                binding=publication.canonical_binding,
                field=TerminalOwnerTimingField.GATEWAY_PUBLICATION,
                sample_key="canonical-source-publisher",
                completed_ns=result.completed_ns,
            )

    def publication_receipt(
        self,
        *,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Consume a direct same-service publication result.

        The startup-enrolled source route imports remote authority before this
        call. Canonical rank zero supplies its publisher-issued local authority
        through the same boundary, so every source owner advances identically.

        :param wire_receipt: Exact canonical publisher receipt.
        :param local_receipt: Matching process-local import authority.
        :param authenticated_issuer: Source rank proved by the enrolled route.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(local_receipt) is not TerminalReceipt:
            raise TypeError("local_receipt must be TerminalReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        record = self._record(wire_receipt.binding.digest)
        self._consume_publication_receipt(
            record=record,
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
            reason=(
                "gateway publication failed"
                if wire_receipt.kind is TerminalReceiptKind.FAILURE
                else None
            ),
        )
        if (
            self._local_identity.tp_rank == 0
            and wire_receipt.kind is TerminalReceiptKind.GATEWAY_PUBLISHED
        ):
            self._timing.complete(
                binding=wire_receipt.binding,
                field=TerminalOwnerTimingField.GATEWAY_PUBLICATION,
                sample_key="canonical-source-publisher",
                completed_ns=wire_receipt.terminal_timestamp_ns,
            )

    def publisher_died(self, binding_digest: bytes, reason: str) -> None:
        """Enter process-fatal source authority after publisher-thread death.

        :param binding_digest: Exact live canonical source binding.
        :param reason: Stable process-fatal publisher evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        record = self._record(binding_digest)
        if record.submission.identity.local_binding.owner.tp_rank != 0:
            raise RuntimeError("only canonical source rank owns publisher death")
        self._submit_event(
            producer_class=NativeTerminalProducerClass.LOCAL,
            authenticated_issuer=None,
            binding_digest=binding_digest,
            kind=NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED,
            reason=reason,
        )
        self._emit_metric_once(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED,
        )

    def consume_terminal_action(
        self,
        action: NativeTerminalOwnerAction,
        retire_submission: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
    ) -> PackedTerminalSourceSubmission | None:
        """Commit retirement side effects or retain quarantined authority.

        :param action: Exact ``REQUEST_RETIRED`` or ``REQUEST_QUARANTINED`` action.
        :param retire_submission: Transactional external retirement boundary run
            before native acknowledgement and wiring-record deletion.
        :returns: Retired immutable submission, otherwise ``None`` for quarantine.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in (
            NativeTerminalOwnerActionKind.REQUEST_RETIRED,
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        ):
            raise ValueError("source terminal consumption requires a terminal action")
        if not callable(retire_submission):
            raise TypeError("retire_submission must be callable")
        record = self._claim_action(action, action.kind)
        if action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED:
            with self._lock:
                record.quarantined = True
            self._runtime.acknowledge_consumed_action(action)
            return None
        with self._lock:
            if (
                not record.reclaim_consumed
                or not record.publication_terminal
                or record.publication_failed
                or record.publication_action is not None
            ):
                raise RuntimeError("native retirement preceded joined side effects")
        retire_submission(record.submission, action)
        self._runtime.acknowledge_consumed_action(action)
        with self._lock:
            current = self._records.get(action.binding.digest)
            if current is not record:
                raise RuntimeError("source request registry changed during retirement")
            # REQUEST_RETIRED is the sole successful deletion authority. Earlier
            # local observations cannot prove both reclaim and publication joins.
            del self._records[action.binding.digest]
        return record.submission

    def inventory(self) -> PackedTerminalSourceInventory:
        """Return exact live and quarantined source side-effect identities.

        :returns: Immutable health and teardown inventory.
        """

        with self._lock:
            active = tuple(sorted(self._records))
            quarantined = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.quarantined
                )
            )
            pending_publication_count = sum(
                record.publication_action is not None
                for record in self._records.values()
            )
            completion_required = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.cancellation_reason is not None
                )
            )
            active_result_slots = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.submission.output_projection is not None
                )
            )
            quarantined_result_slots = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.quarantined
                    and record.submission.output_projection is not None
                )
            )
        return PackedTerminalSourceInventory(
            active_binding_digests=active,
            quarantined_binding_digests=quarantined,
            completion_required_binding_digests=completion_required,
            active_result_slot_binding_digests=active_result_slots,
            quarantined_result_slot_binding_digests=quarantined_result_slots,
            pending_publication_action_count=pending_publication_count,
        )

    def _submit_request_terminal_receipt(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        event_kind: NativeTerminalOwnerEventKind,
        reason: str | None,
    ) -> None:
        """Join route authentication with one request-terminal authority.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport authority submitted to native state.
        :param local_receipt: Matching process-local authority retained by wiring.
        :param authenticated_issuer: Process identity proved by control routing.
        :param event_kind: Native ready or failed transition.
        :param reason: Stable failure evidence, otherwise ``None``.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(local_receipt) is not TerminalReceipt:
            raise TypeError("local_receipt must be TerminalReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if event_kind is NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY:
            expected_kind = TerminalReceiptKind.REQUEST_READY
            expected_outcome = TerminalReceiptOutcome.SUCCESS
            if reason is not None:
                raise ValueError("request readiness cannot carry a failure reason")
        elif event_kind is NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED:
            expected_kind = TerminalReceiptKind.FAILURE
            expected_outcome = TerminalReceiptOutcome.FAILURE
            if type(reason) is not str or len(reason) == 0:
                raise ValueError("request failure requires a non-empty reason")
        else:
            raise ValueError("source terminal ingress requires ready or failed event")
        if (
            wire_receipt.kind is not expected_kind
            or wire_receipt.outcome is not expected_outcome
            or local_receipt.kind is not expected_kind
            or local_receipt.outcome is not expected_outcome
        ):
            raise RuntimeError("source terminal ingress received another authority")
        record = self._record(binding_digest)
        identity = record.submission.identity
        binding = identity.local_binding
        if authenticated_issuer != identity.request_ready_issuer:
            raise RuntimeError("request-terminal route authenticated another issuer")
        if wire_receipt.issuer != authenticated_issuer:
            raise RuntimeError("request-terminal receipt asserts another issuer")
        if wire_receipt.binding != binding or local_receipt.binding != binding:
            raise RuntimeError("request-terminal authority targets another binding")
        if local_receipt.terminal_timestamp_ns != wire_receipt.terminal_timestamp_ns:
            raise RuntimeError(
                "local request-terminal authority differs from wire receipt"
            )
        with self._lock:
            current = self._records.get(binding_digest)
            if current is not record:
                raise RuntimeError("source request registry changed during terminality")
            if (
                current.request_ready_receipt is not None
                or current.request_failure_receipt is not None
            ):
                raise RuntimeError("request terminality was delivered twice")
            if event_kind is NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY:
                current.request_ready_receipt = local_receipt
            else:
                current.request_failure_receipt = local_receipt
        producer_id = self._runtime.python_producer_id(
            NativeTerminalProducerClass.RECEIPT,
            NativeTerminalProcessIdentity.from_identity(authenticated_issuer),
        )
        self._runtime.submit_imported_receipt(
            producer_id,
            NativeTerminalReceipt.from_wire_receipt(wire_receipt),
            event_kind,
            reason=reason,
            enqueued_ns=self._clock_ns(),
        )
        self._emit_metric_once(binding_digest, event_kind)

    def _consume_followup_action(
        self,
        *,
        action: NativeTerminalOwnerAction,
        expected_kind: NativeTerminalOwnerActionKind,
        side_effect: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
        followup_kind: NativeTerminalOwnerEventKind,
        failure_label: str,
    ) -> None:
        """Run one owner-earned side effect and return its terminal event.

        :param action: Exact one-shot native action.
        :param expected_kind: Required action authority.
        :param side_effect: Nonblocking serving operation.
        :param followup_kind: Event proving accepted side-effect completion.
        :param failure_label: Stable fail-closed evidence prefix.
        """

        if not callable(side_effect):
            raise TypeError("side_effect must be callable")
        record = self._claim_action(action, expected_kind)
        try:
            side_effect(record.submission, action)
            self._runtime.complete_work_action(
                self._local_producer_id,
                action,
                followup_kind,
                enqueued_ns=self._clock_ns(),
            )
        except Exception as error:
            self._complete_failed_work(
                action=action,
                producer_id=self._local_producer_id,
                followup_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                label=failure_label,
                error=error,
            )
            raise
        self._emit_metric_once(action.binding.digest, followup_kind)

    def _claim_action(
        self,
        action: NativeTerminalOwnerAction,
        expected_kind: NativeTerminalOwnerActionKind,
    ) -> _SourceRecord:
        """Claim one exact native action before any serving side effect.

        :param action: Candidate native action.
        :param expected_kind: Required source authority.
        :returns: Exact live side-effect record.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if type(expected_kind) is not NativeTerminalOwnerActionKind:
            raise TypeError("expected_kind must be NativeTerminalOwnerActionKind")
        if action.kind is not expected_kind:
            raise ValueError(
                "source operation requires "
                f"{expected_kind.name}, got {action.kind.name}"
            )
        record = self._record(action.binding.digest)
        native_binding = NativeTerminalRequestBinding.from_binding(
            record.submission.identity.local_binding
        )
        if action.binding != native_binding:
            raise RuntimeError("native action binding differs from the source record")
        with self._lock:
            if action.action_id in record.claimed_action_ids:
                raise RuntimeError("native source action was delivered twice")
            record.claimed_action_ids.add(action.action_id)
        return record

    def _local_publication_result(
        self,
        publication: FrozenTerminalGatewayPublication,
        source_receipts: tuple[IssuedTerminalWireReceipt, ...],
    ) -> tuple[_SourceRecord, IssuedTerminalWireReceipt]:
        """Resolve this rank's authenticated result without rank-zero inference.

        :param publication: Exact request-global publication attempt.
        :param source_receipts: Canonically ordered per-rank result authority.
        :returns: Local side-effect record and its exact issued receipt.
        """

        with self._lock:
            matches = tuple(
                (self._records[issued.wire_receipt.binding.digest], issued)
                for issued in source_receipts
                if issued.wire_receipt.binding.digest in self._records
            )
        if len(matches) != 1:
            raise RuntimeError(
                "publisher result must contain exactly one live local source receipt"
            )
        record, local_receipt = matches[0]
        identity = record.submission.identity
        if (
            publication.identity != identity.publication_identity
            or publication.canonical_binding != identity.source_bindings[0]
            or publication.source_bindings != identity.source_bindings
        ):
            raise RuntimeError("publisher result differs from the source identity plan")
        if record.request_ready_receipt is None:
            raise RuntimeError("publisher result preceded local request readiness")
        if record.publication_action is None:
            raise RuntimeError("publisher result preceded publication action")
        if (
            identity.local_binding.owner.tp_rank == 0
            and not record.publication_submitted
        ):
            raise RuntimeError("canonical publisher result preceded publication submit")
        if record.publication_terminal:
            raise RuntimeError("publisher result was delivered twice")
        wire_receipt = local_receipt.wire_receipt
        if wire_receipt.issuer != identity.publisher_issuer:
            raise RuntimeError("publisher result was signed by another process")
        return record, local_receipt

    def _consume_publication_receipt(
        self,
        *,
        record: _SourceRecord,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        reason: str | None,
    ) -> None:
        """Advance one local source owner from enrolled publisher authority.

        :param record: Exact live source side-effect record.
        :param wire_receipt: Canonical rank-zero publication result.
        :param local_receipt: Matching local import authority.
        :param authenticated_issuer: Rank proved by the source control route.
        :param reason: Stable publication failure reason, when failed.
        """

        identity = record.submission.identity
        binding = identity.local_binding
        if authenticated_issuer != identity.publisher_issuer:
            raise RuntimeError("publication route authenticated another issuer")
        if wire_receipt.issuer != authenticated_issuer:
            raise RuntimeError("publication receipt asserts another issuer")
        if wire_receipt.binding != binding or local_receipt.binding != binding:
            raise RuntimeError("publication authority targets another source binding")
        shared_fields = (
            local_receipt.kind is wire_receipt.kind,
            local_receipt.outcome is wire_receipt.outcome,
            local_receipt.terminal_timestamp_ns == wire_receipt.terminal_timestamp_ns,
        )
        if not all(shared_fields):
            raise RuntimeError("local publication authority differs from wire receipt")
        publication_failed = wire_receipt.kind is TerminalReceiptKind.FAILURE
        if publication_failed:
            if wire_receipt.outcome is not TerminalReceiptOutcome.FAILURE:
                raise RuntimeError("publication failure requires a failure outcome")
            if type(reason) is not str or len(reason) == 0:
                raise RuntimeError("publication failure requires stable evidence")
            kind = NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED
        else:
            if (
                wire_receipt.kind is not TerminalReceiptKind.GATEWAY_PUBLISHED
                or wire_receipt.outcome is not TerminalReceiptOutcome.SUCCESS
            ):
                raise RuntimeError("publication receipt has an invalid terminal shape")
            if reason is not None:
                raise RuntimeError(
                    "successful publication cannot carry failure evidence"
                )
            kind = NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED
        action = record.publication_action
        if action is None:
            raise RuntimeError("publication result preceded publication authority")
        if record.request_ready_receipt is None:
            raise RuntimeError("publication result preceded request readiness")
        if record.publication_terminal:
            raise RuntimeError("publication result was delivered twice")
        if self._local_identity.tp_rank == 0 and not record.publication_submitted:
            raise RuntimeError("canonical publication result preceded publisher submit")

        producer_id = self._runtime.python_producer_id(
            NativeTerminalProducerClass.RECEIPT,
            NativeTerminalProcessIdentity.from_identity(authenticated_issuer),
        )
        try:
            self._runtime.complete_work_action(
                producer_id,
                action,
                kind,
                receipt=NativeTerminalReceipt.from_wire_receipt(wire_receipt),
                reason=reason,
                enqueued_ns=self._clock_ns(),
            )
        except Exception as error:
            self._complete_failed_work(
                action=action,
                producer_id=self._local_producer_id,
                followup_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                label="gateway publication result delivery failed",
                error=error,
            )
            self._mark_publication_action_failed(record, action)
            raise
        with self._lock:
            current = self._records.get(binding.digest)
            if current is not record:
                raise RuntimeError("source request registry changed during publication")
            record.publication_action = None
            record.publication_terminal = True
            record.publication_failed = publication_failed
        self._emit_metric_once(binding.digest, kind)

    def _submit_local(
        self,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
    ) -> None:
        """Submit one source-local event after exact registry validation.

        :param binding_digest: Exact accepted source binding.
        :param kind: Source-local event kind.
        """

        self._record(binding_digest)
        self._submit_event(
            producer_class=NativeTerminalProducerClass.LOCAL,
            authenticated_issuer=None,
            binding_digest=binding_digest,
            kind=kind,
        )
        self._emit_metric_once(binding_digest, kind)

    def _submit_event(
        self,
        *,
        producer_class: NativeTerminalProducerClass,
        authenticated_issuer: TerminalProcessIdentity | None,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
    ) -> None:
        """Submit one exact event through its registered authority.

        :param producer_class: Required producer authority class.
        :param authenticated_issuer: Exact route identity, when required.
        :param binding_digest: Exact native lifecycle key.
        :param kind: Closed native source event.
        :param receipt: Authenticated one-shot authority when required.
        :param reason: Stable failure evidence when required.
        """

        native_issuer = (
            None
            if authenticated_issuer is None
            else NativeTerminalProcessIdentity.from_identity(authenticated_issuer)
        )
        producer_id = self._runtime.python_producer_id(
            producer_class,
            native_issuer,
        )
        self._runtime.submit(
            producer_id,
            binding_digest,
            kind,
            receipt=receipt,
            reason=reason,
            enqueued_ns=self._clock_ns(),
        )

    def _record(self, binding_digest: bytes) -> _SourceRecord:
        """Return one exact live side-effect record.

        :param binding_digest: Exact native lifecycle identity.
        :returns: Matching live source record.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        with self._lock:
            record = self._records.get(binding_digest)
        if record is None:
            raise RuntimeError("source operation targets an unknown binding")
        return record

    def _complete_failed_work(
        self,
        *,
        action: NativeTerminalOwnerAction,
        producer_id: int,
        followup_kind: NativeTerminalOwnerEventKind,
        label: str,
        error: BaseException,
        receipt: NativeTerminalReceipt | None = None,
    ) -> None:
        """Return one failed work action to the authoritative runtime.

        :param action: Exact one-shot action whose side effect did not complete.
        :param producer_id: Exact producer authorized to report the failure.
        :param followup_kind: Failure transition earned by the action.
        :param label: Stable evidence prefix.
        :param error: Original serving-boundary failure.
        :param receipt: Authenticated publication authority when required.
        """

        formatted_traceback = traceback.format_exc()
        logger.error("%s:\n%s", label, formatted_traceback)
        reason = f"{label}: {type(error).__name__}: {error}"
        try:
            self._runtime.complete_work_action(
                producer_id,
                action,
                followup_kind,
                receipt=receipt,
                reason=reason,
                enqueued_ns=self._clock_ns(),
            )
        except Exception:
            logger.critical(
                "Native runtime also rejected fail-closed work completion:\n%s",
                traceback.format_exc(),
            )
            raise

    def _fail_scheduler_action(
        self,
        action: NativeTerminalOwnerAction,
        label: str,
        error: BaseException,
    ) -> None:
        """Enter process-fatal runtime state for ambiguous scheduler cleanup.

        :param action: Exact reclaim action whose cleanup did not complete.
        :param label: Stable evidence prefix.
        :param error: Original scheduler-boundary failure.
        """

        logger.error("%s:\n%s", label, traceback.format_exc())
        reason = f"{label}: {type(error).__name__}: {error}"
        try:
            self._runtime.fail_scheduler_action(action, reason)
        except Exception:
            logger.critical(
                "Native runtime also rejected scheduler failure:\n%s",
                traceback.format_exc(),
            )
            raise

    def _mark_publication_action_failed(
        self,
        record: _SourceRecord,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release retained publication accounting after fail-closed return.

        :param record: Exact source record holding publication authority.
        :param action: Exact publication action returned as failed.
        """

        with self._lock:
            current = self._records.get(action.binding.digest)
            if current is not record or current.publication_action != action:
                raise RuntimeError(
                    "publication action changed during failed completion"
                )
            current.publication_action = None
            current.publication_terminal = True
            current.publication_failed = True

    def _emit_metric_once(
        self,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
    ) -> None:
        """Emit one non-gating metric without changing lifecycle authority.

        :param binding_digest: Exact source lifecycle identity.
        :param kind: Operation represented by the metric.
        """

        key = (binding_digest, kind)
        with self._lock:
            if key in self._emitted_metrics:
                raise RuntimeError("source metric event was emitted twice")
            self._emitted_metrics.add(key)
        try:
            self._metrics_sink.emit(
                PackedTerminalSourceMetric(
                    binding_digest=binding_digest,
                    event_kind=kind,
                    timestamp_ns=self._clock_ns(),
                )
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "Terminal source metric projection failed without gating progress:\n%s",
                traceback.format_exc(),
            )
