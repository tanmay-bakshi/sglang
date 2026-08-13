import dataclasses
import threading

import pytest
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

    def advance_ns(self, delta_ns: int) -> None:
        """Advance the synthetic clock without producing a timestamp.

        :param delta_ns: Positive nanoseconds added to the current time.
        """

        if type(delta_ns) is not int or delta_ns <= 0:
            raise ValueError("delta_ns must be a positive integer")
        with self._lock:
            self._value += delta_ns


class _RecordingSink(TerminalGatewaySink):
    """Test sink recording thread affinity and immutable bytes."""

    payloads: list[bytes]
    created_thread: int
    closed_thread: int | None
    fail_send: bool
    fail_close: bool
    send_entered: threading.Event
    send_release: threading.Event | None
    timeout_seconds: list[float]

    def __init__(
        self,
        *,
        fail_send: bool,
        fail_close: bool,
        send_release: threading.Event | None,
    ) -> None:
        """Create the sink on the publisher thread.

        :param fail_send: Whether every send raises.
        :param fail_close: Whether sink close raises.
        :param send_release: Optional barrier held during send.
        """

        self.payloads = []
        self.created_thread = threading.get_ident()
        self.closed_thread = None
        self.fail_send = fail_send
        self.fail_close = fail_close
        self.send_entered = threading.Event()
        self.send_release = send_release
        self.timeout_seconds = []

    def send(self, encoded_payload: bytes, timeout_seconds: float) -> None:
        """Record or fail one send.

        :param encoded_payload: Frozen IPC bytes.
        :param timeout_seconds: Remaining hash-bound publication deadline.
        """

        self.timeout_seconds.append(timeout_seconds)
        self.send_entered.set()
        if self.send_release is not None:
            self.send_release.wait(timeout=2.0)
        if self.fail_send:
            raise RuntimeError("synthetic gateway send failure")
        self.payloads.append(encoded_payload)

    def close(self) -> None:
        """Record the execution context which closed the sink."""

        self.closed_thread = threading.get_ident()
        if self.fail_close:
            raise RuntimeError("synthetic gateway close failure")


class _RecordingSinkFactory(TerminalGatewaySinkFactory):
    """Create and expose exactly one publisher-thread-owned test sink."""

    _fail_send: bool
    _fail_close: bool
    _send_release: threading.Event | None
    _create_release: threading.Event | None
    _fail_create: bool
    sink: _RecordingSink | None
    create_entered: threading.Event
    created: threading.Event

    def __init__(
        self,
        *,
        fail_send: bool = False,
        fail_close: bool = False,
        send_release: threading.Event | None = None,
        create_release: threading.Event | None = None,
        fail_create: bool = False,
    ) -> None:
        """Configure one test sink.

        :param fail_send: Whether sends fail.
        :param fail_close: Whether sink close fails.
        :param send_release: Optional send barrier.
        :param create_release: Optional factory-construction barrier.
        :param fail_create: Whether factory creation fails after its barrier.
        """

        self._fail_send = fail_send
        self._fail_close = fail_close
        self._send_release = send_release
        self._create_release = create_release
        self._fail_create = fail_create
        self.sink = None
        self.create_entered = threading.Event()
        self.created = threading.Event()

    def create(self) -> TerminalGatewaySink:
        """Create the sink from the publisher thread.

        :returns: Newly created recording sink.
        """

        if self.sink is not None:
            raise RuntimeError("test sink factory was reused")
        self.create_entered.set()
        if self._create_release is not None:
            self._create_release.wait(timeout=2.0)
        if self._fail_create:
            raise RuntimeError("synthetic gateway factory failure")
        self.sink = _RecordingSink(
            fail_send=self._fail_send,
            fail_close=self._fail_close,
            send_release=self._send_release,
        )
        self.created.set()
        return self.sink


def _source_bindings(
    *,
    room_id: int = 31,
    request_generation_byte: int = 0x31,
) -> tuple[TerminalRequestBinding, ...]:
    """Build one canonical TP2 source-rank manifest.

    :param room_id: Stable request room identity.
    :param request_generation_byte: Byte repeated across request generation.
    :returns: Exact rank-zero and rank-one source bindings.
    """

    key = PackedRequestKey(
        room_id=room_id,
        request_generation=bytes((request_generation_byte,)) * 16,
    )
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
    publication_generation_byte: int = 0xC3,
) -> FrozenTerminalGatewayPublication:
    """Build one request-global publication handoff.

    :param bindings: Complete source TP manifest.
    :param request_ready_issuer: Trusted imported coordinator authority.
    :param payload: Frozen downstream IPC bytes.
    :param publication_generation_byte: Byte repeated across publication generation.
    :returns: Complete publisher input.
    """

    canonical = bindings[0]
    return FrozenTerminalGatewayPublication(
        identity=TerminalPublicationIdentity(
            request_key=canonical.request_key,
            publisher_process_generation=canonical.owner.process_generation,
            publication_generation=bytes((publication_generation_byte,)) * 16,
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
    *,
    capacity: int = 4,
    result_event: threading.Event | None = None,
    fail_result_listener: bool = False,
) -> PackedTerminalOutputPublisher:
    """Construct one publisher with captured callbacks.

    :param factory: Thread-owned sink factory.
    :param bindings: Complete source manifest.
    :param request_ready_issuer: Trusted coordinator receipt issuer.
    :param results: Mutable test-only result capture.
    :param fatals: Mutable test-only fatal capture.
    :param clock: Synthetic publisher clock.
    :param capacity: Maximum pending publication count.
    :param result_event: Optional result-delivery notification.
    :param fail_result_listener: Whether result delivery raises after notification.
    :returns: Unstarted publisher.
    """

    def record_result(result: TerminalGatewayPublicationResult) -> None:
        """Record one result and apply the requested listener disposition.

        :param result: Immutable publisher result.
        """

        results.append(result)
        if result_event is not None:
            result_event.set()
        if fail_result_listener:
            raise RuntimeError("synthetic publisher result listener failure")

    return PackedTerminalOutputPublisher(
        capacity=capacity,
        sink_factory=factory,
        wire_issuer=TerminalWireReceiptIssuer(bindings[0].owner),
        request_ready_authorities=frozenset((request_ready_issuer.authority,)),
        result_listener=record_result,
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
    assert publisher.stop_admission_and_join()

    sink = factory.sink
    assert sink is not None
    assert sink.payloads == [b"frozen-output"]
    assert len(sink.timeout_seconds) == 1
    assert 59.0 < sink.timeout_seconds[0] <= 60.0
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
    with pytest.raises(TerminalGatewayPublisherError):
        publisher.submit(conflicting)
    release.set()
    assert publisher.stop_admission_and_join()

    assert len(sink.payloads) <= 1
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.failed_identities == (first.identity,)
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
    assert publisher.stop_admission_and_join()

    assert len(results) == 1
    assert type(results[0]) is TerminalGatewayPublicationFailure
    assert "synthetic gateway send failure" in results[0].formatted_traceback
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.failed_identities == (publication.identity,)
    assert len(fatals) == 1
    with pytest.raises(TerminalGatewayPublisherError):
        publisher.start()


def test_publication_identity_must_name_the_canonical_publisher() -> None:
    """A stale publisher process generation cannot enter the handoff."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    publication = _publication(bindings, request_ready_issuer)
    stale_identity = dataclasses.replace(
        publication.identity,
        publisher_process_generation=bytes.fromhex("d4" * 16),
    )

    with pytest.raises(
        ValueError,
        match="publication identity belongs to another publisher",
    ):
        dataclasses.replace(publication, identity=stale_identity)


def test_factory_death_quarantines_every_accepted_publication() -> None:
    """A publisher that cannot own its sink fails all accepted identities."""

    first_bindings = _source_bindings()
    second_bindings = _source_bindings(
        room_id=32,
        request_generation_byte=0x32,
    )
    request_ready_issuer = TerminalReceiptIssuer()
    release = threading.Event()
    factory = _RecordingSinkFactory(
        create_release=release,
        fail_create=True,
    )
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        first_bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
    )
    first = _publication(first_bindings, request_ready_issuer)
    second = _publication(
        second_bindings,
        request_ready_issuer,
        publication_generation_byte=0xC4,
    )

    publisher.start()
    assert factory.create_entered.wait(timeout=2.0)
    assert publisher.submit(first)
    assert publisher.submit(second)
    release.set()
    assert publisher.stop_admission_and_join()

    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == ()
    assert snapshot.failed_identities == (first.identity, second.identity)
    assert results == []
    assert len(fatals) == 1
    assert fatals[0][1] is not None
    assert "synthetic gateway factory failure" in fatals[0][1]


def test_queue_overflow_fails_the_new_and_every_unpublished_identity() -> None:
    """Bound overflow closes the process without stranding pending work."""

    first_bindings = _source_bindings()
    second_bindings = _source_bindings(
        room_id=33,
        request_generation_byte=0x33,
    )
    request_ready_issuer = TerminalReceiptIssuer()
    release = threading.Event()
    factory = _RecordingSinkFactory(send_release=release)
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        first_bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
        capacity=1,
    )
    first = _publication(first_bindings, request_ready_issuer)
    second = _publication(
        second_bindings,
        request_ready_issuer,
        publication_generation_byte=0xC5,
    )

    publisher.start()
    assert publisher.submit(first)
    assert factory.created.wait(timeout=2.0)
    sink = factory.sink
    assert sink is not None
    assert sink.send_entered.wait(timeout=2.0)
    with pytest.raises(
        TerminalGatewayPublisherError,
        match="gateway publication queue exceeded capacity 1",
    ):
        publisher.submit(second)
    release.set()
    assert publisher.stop_admission_and_join()

    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == ()
    assert snapshot.failed_identities == (first.identity, second.identity)
    assert not any(
        type(result) is TerminalGatewayPublicationSuccess for result in results
    )
    assert len(fatals) == 1


def test_result_delivery_failure_is_fatal_and_conserves_queued_work() -> None:
    """Receipt fan-out failure quarantines current and queued publications."""

    first_bindings = _source_bindings()
    second_bindings = _source_bindings(
        room_id=34,
        request_generation_byte=0x34,
    )
    request_ready_issuer = TerminalReceiptIssuer()
    release = threading.Event()
    factory = _RecordingSinkFactory(send_release=release)
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    publisher = _publisher(
        factory,
        first_bindings,
        request_ready_issuer,
        results,
        fatals,
        _Clock(),
        fail_result_listener=True,
    )
    first = _publication(first_bindings, request_ready_issuer)
    second = _publication(
        second_bindings,
        request_ready_issuer,
        publication_generation_byte=0xC6,
    )

    publisher.start()
    assert publisher.submit(first)
    assert factory.created.wait(timeout=2.0)
    sink = factory.sink
    assert sink is not None
    assert sink.send_entered.wait(timeout=2.0)
    assert publisher.submit(second)
    release.set()
    assert publisher.stop_admission_and_join()

    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == ()
    assert snapshot.failed_identities == (first.identity, second.identity)
    assert len(fatals) == 1
    assert snapshot.fatal_traceback is not None
    assert "synthetic publisher result listener failure" in snapshot.fatal_traceback


def test_completed_identity_is_not_rolled_back_by_a_late_conflict() -> None:
    """A rejected post-send replay cannot undo an already delivered receipt."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    result_event = threading.Event()
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
        result_event=result_event,
    )
    publication = _publication(bindings, request_ready_issuer)
    conflicting = dataclasses.replace(
        publication,
        encoded_payload=b"late-conflicting-output",
    )

    publisher.start()
    assert publisher.submit(publication)
    assert result_event.wait(timeout=2.0)
    with pytest.raises(TerminalGatewayPublisherError):
        publisher.submit(conflicting)
    assert publisher.stop_admission_and_join()

    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == (publication.identity,)
    assert snapshot.failed_identities == ()
    assert len(results) == 1
    assert type(results[0]) is TerminalGatewayPublicationSuccess
    assert len(fatals) == 1


def test_expired_publication_deadline_fails_before_socket_send() -> None:
    """Queue delay consumes the one-shot publication deadline without reset."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    create_release = threading.Event()
    factory = _RecordingSinkFactory(create_release=create_release)
    results: list[TerminalGatewayPublicationResult] = []
    fatals: list[tuple[str, str | None]] = []
    clock = _Clock()
    publisher = _publisher(
        factory,
        bindings,
        request_ready_issuer,
        results,
        fatals,
        clock,
    )
    publication = _publication(bindings, request_ready_issuer)

    publisher.start()
    assert factory.create_entered.wait(timeout=2.0)
    assert publisher.submit(publication)
    clock.advance_ns(60_000_000_000)
    create_release.set()
    assert publisher.stop_admission_and_join()

    sink = factory.sink
    assert sink is not None
    assert sink.payloads == []
    assert sink.timeout_seconds == []
    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.failed_identities == (publication.identity,)


def test_close_failure_is_fatal_without_quarantining_published_identity() -> None:
    """Sink teardown failure preserves a fully delivered publication truth."""

    bindings = _source_bindings()
    request_ready_issuer = TerminalReceiptIssuer()
    factory = _RecordingSinkFactory(fail_close=True)
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
    assert publisher.stop_admission_and_join()

    snapshot = publisher.snapshot()
    assert snapshot.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL
    assert snapshot.pending_identities == ()
    assert snapshot.completed_identities == (publication.identity,)
    assert snapshot.failed_identities == ()
    assert len(results) == 1
    assert type(results[0]) is TerminalGatewayPublicationSuccess
    assert len(fatals) == 1
    assert snapshot.fatal_traceback is not None
    assert "synthetic gateway close failure" in snapshot.fatal_traceback
