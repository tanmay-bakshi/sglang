import dataclasses
import threading

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayPublication,
    PackedTerminalOutputPublisher,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationResult,
    TerminalGatewayPublicationSuccess,
    TerminalGatewayPublisherDisposition,
    TerminalGatewayPublisherError,
    TerminalGatewaySink,
    TerminalGatewaySinkFactory,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Clock:
    """Locked deterministic monotonic clock for publisher tests."""

    _value: int
    _lock: threading.Lock

    def __init__(self) -> None:
        """Start at a stable nonzero timestamp."""

        self._value = 1_000
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        """Advance and return one monotonic timestamp.

        :returns: Strictly increasing synthetic nanoseconds.
        """

        with self._lock:
            self._value += 1
            return self._value


class _RecordingSink(TerminalGatewaySink):
    """Test sink recording thread affinity and immutable bytes."""

    payloads: list[bytes]
    created_thread: int
    closed_thread: int | None
    fail_send: bool
    send_entered: threading.Event
    send_release: threading.Event | None

    def __init__(
        self,
        *,
        fail_send: bool,
        send_release: threading.Event | None,
    ) -> None:
        """Create the sink on the publisher thread.

        :param fail_send: Whether every send raises.
        :param send_release: Optional barrier held during send.
        """

        self.payloads = []
        self.created_thread = threading.get_ident()
        self.closed_thread = None
        self.fail_send = fail_send
        self.send_entered = threading.Event()
        self.send_release = send_release

    def send(self, encoded_payload: bytes) -> None:
        """Record or fail one send.

        :param encoded_payload: Frozen IPC bytes.
        """

        self.send_entered.set()
        if self.send_release is not None:
            self.send_release.wait(timeout=2.0)
        if self.fail_send:
            raise RuntimeError("synthetic gateway send failure")
        self.payloads.append(encoded_payload)

    def close(self) -> None:
        """Record the execution context which closed the sink."""

        self.closed_thread = threading.get_ident()


class _RecordingSinkFactory(TerminalGatewaySinkFactory):
    """Create and expose exactly one publisher-thread-owned test sink."""

    _fail_send: bool
    _send_release: threading.Event | None
    sink: _RecordingSink | None
    created: threading.Event

    def __init__(
        self,
        *,
        fail_send: bool = False,
        send_release: threading.Event | None = None,
    ) -> None:
        """Configure one test sink.

        :param fail_send: Whether sends fail.
        :param send_release: Optional send barrier.
        """

        self._fail_send = fail_send
        self._send_release = send_release
        self.sink = None
        self.created = threading.Event()

    def create(self) -> TerminalGatewaySink:
        """Create the sink from the publisher thread.

        :returns: Newly created recording sink.
        """

        if self.sink is not None:
            raise RuntimeError("test sink factory was reused")
        self.sink = _RecordingSink(
            fail_send=self._fail_send,
            send_release=self._send_release,
        )
        self.created.set()
        return self.sink


def _source_bindings() -> tuple[TerminalRequestBinding, ...]:
    """Build one canonical TP2 source-rank manifest.

    :returns: Exact rank-zero and rank-one source bindings.
    """

    key = PackedRequestKey(room_id=31, request_generation=bytes.fromhex("31" * 16))
    return tuple(
        TerminalRequestBinding(
            request_key=key,
            owner=TerminalProcessIdentity(
                process_generation=bytes([rank + 1]) * 16,
                role=TerminalOwnerRole.SOURCE,
                tp_rank=rank,
                tp_size=2,
            ),
            rank_manifest_digest=bytes.fromhex("a1" * 32),
            allocation_digest=bytes.fromhex("b2" * 32),
        )
        for rank in range(2)
    )


def _publication(
    bindings: tuple[TerminalRequestBinding, ...],
    request_ready_issuer: TerminalReceiptIssuer,
    *,
    payload: bytes = b"frozen-output",
) -> FrozenTerminalGatewayPublication:
    """Build one request-global publication handoff.

    :param bindings: Complete source TP manifest.
    :param request_ready_issuer: Trusted imported coordinator authority.
    :param payload: Frozen downstream IPC bytes.
    :returns: Complete publisher input.
    """

    canonical = bindings[0]
    return FrozenTerminalGatewayPublication(
        identity=TerminalPublicationIdentity(
            request_key=canonical.request_key,
            publisher_process_generation=canonical.owner.process_generation,
            publication_generation=bytes.fromhex("c3" * 16),
        ),
        canonical_binding=canonical,
        source_bindings=bindings,
        request_ready_receipt=request_ready_issuer.issue(
            binding=canonical,
            kind=TerminalReceiptKind.REQUEST_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=999,
        ),
        encoded_payload=payload,
        enqueued_ns=1_000,
    )


def _publisher(
    factory: TerminalGatewaySinkFactory,
    bindings: tuple[TerminalRequestBinding, ...],
    request_ready_issuer: TerminalReceiptIssuer,
    results: list[TerminalGatewayPublicationResult],
    fatals: list[tuple[str, str | None]],
    clock: _Clock,
) -> PackedTerminalOutputPublisher:
    """Construct one publisher with captured callbacks.

    :param factory: Thread-owned sink factory.
    :param bindings: Complete source manifest.
    :param request_ready_issuer: Trusted coordinator receipt issuer.
    :param results: Mutable test-only result capture.
    :param fatals: Mutable test-only fatal capture.
    :param clock: Synthetic publisher clock.
    :returns: Unstarted publisher.
    """

    return PackedTerminalOutputPublisher(
        capacity=4,
        sink_factory=factory,
        wire_issuer=TerminalWireReceiptIssuer(bindings[0].owner),
        request_ready_authorities=frozenset((request_ready_issuer.authority,)),
        result_listener=results.append,
        fatal_listener=lambda reason, formatted: fatals.append((reason, formatted)),
        clock_ns=clock.now_ns,
    )


def test_exact_duplicate_coalesces_and_socket_lifecycle_stays_on_thread() -> None:
    """One exact generation sends once and owns its sink on one thread."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    publication = _publication(bindings, request_ready_issuer)
    factory = _RecordingSinkFactory()
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
    )

    publisher.start()
    assert publisher.submit(publication)
    assert not publisher.submit(publication)
    assert publisher.stop_admission_and_join(2.0)

    sink = factory.sink
    assert sink is not None
    assert sink.payloads == [b"frozen-output"]
    assert sink.created_thread == sink.closed_thread
    assert len(results) == 1
    assert type(results[0]) is TerminalGatewayPublicationSuccess
    assert (
        tuple(receipt.wire_receipt.binding for receipt in results[0].source_receipts)
        == bindings
    )
    assert fatals == []
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.STOPPED
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == (publication.identity,)


def test_conflicting_generation_is_process_fatal_without_second_send() -> None:
    """Same publication identity cannot acquire a second payload."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    release = threading.Event()
    factory = _RecordingSinkFactory(send_release=release)
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
    )
    first = _publication(bindings, request_ready_issuer)
    conflicting = dataclasses.replace(first, encoded_payload=b"conflicting-output")

    publisher.start()
    assert publisher.submit(first)
    assert factory.created.wait(timeout=2.0)
    sink = factory.sink
    assert sink is not None
    assert sink.send_entered.wait(timeout=2.0)
    try:
        publisher.submit(conflicting)
        raise AssertionError("conflicting publication unexpectedly succeeded")
    except TerminalGatewayPublisherError:
        pass
    release.set()
    assert publisher.stop_admission_and_join(2.0)

    assert len(sink.payloads) <= 1
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert len(fatals) == 1


def test_send_failure_quarantines_identity_and_kills_publisher() -> None:
    """A failed send emits failure evidence and cannot restart."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    factory = _RecordingSinkFactory(fail_send=True)
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
    )
    publication = _publication(bindings, request_ready_issuer)

    publisher.start()
    assert publisher.submit(publication)
    assert publisher.stop_admission_and_join(2.0)

    assert len(results) == 1
    assert type(results[0]) is TerminalGatewayPublicationFailure
    assert "synthetic gateway send failure" in results[0].formatted_traceback
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.failed_identities == (publication.identity,)
    assert len(fatals) == 1
    try:
        publisher.start()
        raise AssertionError("publisher restart unexpectedly succeeded")
    except TerminalGatewayPublisherError:
        pass
