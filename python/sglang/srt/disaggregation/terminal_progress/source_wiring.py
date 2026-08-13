import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEventKind,
    NativeTerminalReceipt,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayPublication,
    FrozenTerminalGatewayOutputProjection,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationResult,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import TerminalReceipt
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

logger = logging.getLogger(__name__)


@runtime_checkable
class NativeTerminalSourceRuntime(Protocol):
    """Process-lifetime native owner boundary consumed by source wiring."""

    @property
    def local_events(self) -> object:
        """Return the source-local event producer.

        :returns: Opaque producer registered by the process runtime.
        """

    @property
    def local_receipts(self) -> object:
        """Return the source-local receipt producer.

        :returns: Opaque producer registered by the process runtime.
        """

    def control_producer(self, issuer_digest: bytes) -> object:
        """Return the authenticated producer for one remote issuer.

        :param issuer_digest: Exact remote process-identity digest.
        :returns: Opaque producer registered by the process runtime.
        """

    def register_source(
        self,
        binding: TerminalRequestBinding,
        publication_identity: TerminalPublicationIdentity,
        trusted_issuers: tuple[TerminalProcessIdentity, ...],
    ) -> None:
        """Register one source lifecycle before its first event.

        :param binding: Exact rank-local source binding.
        :param publication_identity: Exactly-once gateway publication identity.
        :param trusted_issuers: Complete authenticated receipt issuer set.
        """

    def submit(
        self,
        producer: object,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
    ) -> None:
        """Submit one producer-ordered event to the native owner.

        :param producer: Exact process-lifetime producer.
        :param binding_digest: Exact lifecycle lookup identity.
        :param kind: Closed native event kind.
        :param receipt: Authenticated authority when required.
        :param reason: Stable failure evidence when required.
        """

    def complete_scheduler_action(self, action: NativeTerminalOwnerAction) -> None:
        """Acknowledge exact scheduler consumption to the native owner.

        :param action: Reclaim authority whose resource release completed.
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
    :ivar transport_submission: Opaque immutable packed transport submission.
    """

    identity: PackedTerminalSourceIdentityPlan
    output_projection: FrozenTerminalGatewayOutputProjection
    producer_event_generation: bytes
    transport_submission: object

    def __post_init__(self) -> None:
        """Validate immutable submission identities without duck-typing payloads."""

        if type(self.identity) is not PackedTerminalSourceIdentityPlan:
            raise TypeError("identity must be PackedTerminalSourceIdentityPlan")
        if not isinstance(
            self.output_projection,
            FrozenTerminalGatewayOutputProjection,
        ):
            raise TypeError(
                "output_projection must inherit FrozenTerminalGatewayOutputProjection"
            )
        if type(self.producer_event_generation) is not bytes:
            raise TypeError("producer_event_generation must be bytes")
        if len(self.producer_event_generation) != 16:
            raise ValueError("producer_event_generation must contain 16 bytes")
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


@dataclasses.dataclass(slots=True)
class _SourceRecord:
    """Side-effect inventory keyed by native-owned lifecycle identity."""

    submission: PackedTerminalSourceSubmission
    request_ready_receipt: TerminalReceipt | None = None
    publication_submitted: bool = False
    reclaim_consumed: bool = False
    publication_terminal: bool = False


class PackedTerminalSourceWiring:
    """Coordinate source side effects around one authoritative native owner.

    This component stores immutable payloads and dispatches side effects. It does
    not reduce lifecycle phases, mint native authority, or infer completion from
    a status field. Every state-changing permission enters through the supplied
    process-lifetime native runtime.
    """

    _runtime: NativeTerminalSourceRuntime
    _publisher: PackedTerminalSourcePublisher | None
    _metrics_sink: PackedTerminalSourceMetricsSink
    _clock_ns: Callable[[], int]
    _records: dict[bytes, _SourceRecord]
    _emitted_metrics: set[tuple[bytes, NativeTerminalOwnerEventKind]]
    _lock: threading.Lock

    def __init__(
        self,
        *,
        runtime: NativeTerminalSourceRuntime,
        publisher: PackedTerminalSourcePublisher | None,
        metrics_sink: PackedTerminalSourceMetricsSink,
        clock_ns: Callable[[], int],
    ) -> None:
        """Construct source orchestration around existing process owners.

        :param runtime: Sole authoritative process-lifetime native runtime.
        :param publisher: Canonical rank publisher, otherwise ``None``.
        :param metrics_sink: Non-gating metric projection sink.
        :param clock_ns: Local monotonic nanosecond clock.
        """

        if not isinstance(runtime, NativeTerminalSourceRuntime):
            raise TypeError("runtime must satisfy NativeTerminalSourceRuntime")
        if publisher is not None and not isinstance(
            publisher,
            PackedTerminalSourcePublisher,
        ):
            raise TypeError("publisher must satisfy PackedTerminalSourcePublisher")
        if not isinstance(metrics_sink, PackedTerminalSourceMetricsSink):
            raise TypeError("metrics_sink must satisfy PackedTerminalSourceMetricsSink")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._runtime = runtime
        self._publisher = publisher
        self._metrics_sink = metrics_sink
        self._clock_ns = clock_ns
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
        digest = binding.digest
        with self._lock:
            if digest in self._records:
                raise RuntimeError("source submission identity was reused")
            self._runtime.register_source(
                binding,
                identity.publication_identity,
                identity.trusted_issuers,
            )
            self._runtime.submit(
                self._runtime.local_events,
                digest,
                NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
            )
            self._records[digest] = _SourceRecord(submission=submission)
        self._emit_metric_once(
            digest,
            NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED,
        )

    def producer_completed(self, binding_digest: bytes) -> None:
        """Commit completion of the exact producer event and stable slots.

        :param binding_digest: Exact accepted source binding.
        """

        self._submit_local(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_PRODUCER_COMPLETED,
        )

    def gather_posted(self, binding_digest: bytes) -> None:
        """Commit source gather completion and native transfer post.

        :param binding_digest: Exact accepted source binding.
        """

        self._submit_local(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
        )

    def outcomes_sent(self, binding_digest: bytes) -> None:
        """Commit immutable writer and auxiliary outcome publication.

        :param binding_digest: Exact accepted source binding.
        """

        self._submit_local(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
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
        producer = self._runtime.control_producer(authenticated_issuer.digest)
        self._runtime.submit(
            producer,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        )
        self._emit_metric_once(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        )

    def teardown_ack_sent(self, binding_digest: bytes) -> None:
        """Commit successful exact-generation teardown acknowledgement.

        :param binding_digest: Exact accepted source binding.
        """

        self._submit_local(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
        )

    def request_ready(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Consume authenticated request-global readiness and publish on rank zero.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport authority submitted to native state.
        :param local_receipt: Matching process-local authority for the publisher.
        :param authenticated_issuer: Process identity proved by control routing.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(local_receipt) is not TerminalReceipt:
            raise TypeError("local_receipt must be TerminalReceipt")
        record = self._record(binding_digest)
        identity = record.submission.identity
        binding = identity.local_binding
        if authenticated_issuer != identity.request_ready_issuer:
            raise RuntimeError("request-ready route authenticated another issuer")
        if wire_receipt.binding != binding or local_receipt.binding != binding:
            raise RuntimeError("request-ready authority targets another binding")
        native_receipt = NativeTerminalReceipt.from_wire_receipt(wire_receipt)
        producer = self._runtime.control_producer(authenticated_issuer.digest)
        self._runtime.submit(
            producer,
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
            receipt=native_receipt,
        )
        with self._lock:
            current = self._records.get(binding_digest)
            if current is not record:
                raise RuntimeError("source request registry changed during readiness")
            if current.request_ready_receipt is not None:
                raise RuntimeError("request-ready authority was delivered twice")
            current.request_ready_receipt = local_receipt
        self._emit_metric_once(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
        )
        if binding.owner.tp_rank == 0:
            self._submit_publication(record, local_receipt)

    def consume_reclaim(
        self,
        action: NativeTerminalOwnerAction,
        release_resources: Callable[[PackedTerminalSourceSubmission], None],
    ) -> None:
        """Release source resources only under exact native reclaim authority.

        :param action: Native one-shot reclaim action.
        :param release_resources: Scheduler-thread resource release operation.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
            raise ValueError("source reclaim requires RECLAIM_AUTHORIZED")
        if not callable(release_resources):
            raise TypeError("release_resources must be callable")
        binding_digest = action.binding.digest
        record = self._record(binding_digest)
        with self._lock:
            if record.reclaim_consumed:
                raise RuntimeError("source reclaim action was consumed twice")
        release_resources(record.submission)
        self._runtime.complete_scheduler_action(action)
        with self._lock:
            current = self._records.get(binding_digest)
            if current is not record:
                raise RuntimeError("source request registry changed during reclaim")
            current.reclaim_consumed = True
        self._emit_metric_once(
            binding_digest,
            NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
        )

    def publisher_result(self, result: TerminalGatewayPublicationResult) -> None:
        """Project one publisher result back into native lifecycle authority.

        :param result: Exactly-once publisher success or failure.
        """

        if type(result) is TerminalGatewayPublicationFailure:
            publication = result.publication
            record = self._record(publication.canonical_binding.digest)
            self._runtime.submit(
                self._runtime.local_events,
                publication.canonical_binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED,
                reason=result.reason,
            )
            with self._lock:
                record.publication_terminal = True
            self._emit_metric_once(
                publication.canonical_binding.digest,
                NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED,
            )
            return
        if type(result) is not TerminalGatewayPublicationSuccess:
            raise TypeError("result must be a terminal gateway publication result")
        publication = result.publication
        local_binding = publication.canonical_binding
        local_receipt = next(
            (
                issued.wire_receipt
                for issued in result.source_receipts
                if issued.wire_receipt.binding == local_binding
            ),
            None,
        )
        if local_receipt is None:
            raise RuntimeError("publisher success omitted the local source receipt")
        record = self._record(local_binding.digest)
        producer = self._runtime.control_producer(local_receipt.issuer.digest)
        self._runtime.submit(
            producer,
            local_binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
            receipt=NativeTerminalReceipt.from_wire_receipt(local_receipt),
        )
        with self._lock:
            record.publication_terminal = True
        self._emit_metric_once(
            local_binding.digest,
            NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
        )

    def retire(self, binding_digest: bytes) -> PackedTerminalSourceSubmission:
        """Forget side-effect payloads only after native retirement delivery.

        :param binding_digest: Exact retired source binding.
        :returns: Immutable submission released from the registry.
        """

        with self._lock:
            record = self._records.get(binding_digest)
            if record is None:
                raise RuntimeError("native retirement targets an unknown source")
            if not record.reclaim_consumed or not record.publication_terminal:
                raise RuntimeError("native retirement preceded joined side effects")
            del self._records[binding_digest]
            return record.submission

    def _submit_publication(
        self,
        record: _SourceRecord,
        request_ready_receipt: TerminalReceipt,
    ) -> None:
        """Submit canonical immutable output after request-global readiness.

        :param record: Exact source side-effect inventory.
        :param request_ready_receipt: Imported local publisher authority.
        """

        publisher = self._publisher
        if publisher is None:
            raise RuntimeError("canonical source rank has no gateway publisher")
        submission = record.submission
        identity = submission.identity
        publication = FrozenTerminalGatewayPublication(
            identity=identity.publication_identity,
            canonical_binding=identity.local_binding,
            source_bindings=identity.source_bindings,
            request_ready_receipt=request_ready_receipt,
            output_projection=submission.output_projection,
            enqueued_ns=self._clock_ns(),
        )
        publisher.submit(publication)
        with self._lock:
            if record.publication_submitted:
                raise RuntimeError("source gateway publication was submitted twice")
            record.publication_submitted = True

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
        self._runtime.submit(self._runtime.local_events, binding_digest, kind)
        self._emit_metric_once(binding_digest, kind)

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
        except Exception:
            logger.error(
                "Terminal source metric projection failed without gating progress:\n%s",
                traceback.format_exc(),
            )
