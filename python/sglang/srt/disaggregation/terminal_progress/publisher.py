import abc
import dataclasses
import enum
import hashlib
import queue
import threading
import traceback
from collections.abc import Callable

import zmq
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptAuthority,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
    TerminalReceiptToken,
    terminal_receipt_token,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
    TerminalWireReceiptIssuer,
)
from sglang.srt.utils.network import get_zmq_socket


class TerminalGatewayPublisherError(RuntimeError):
    """Gateway publisher lifecycle or exactly-once invariant violation."""


class TerminalGatewayPublisherDisposition(enum.StrEnum):
    """Process-lifetime publisher disposition."""

    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    PROCESS_FATAL = "process_fatal"


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenTerminalGatewayPublication:
    """Immutable request-global output ready for publisher-owned I/O.

    :ivar identity: Exactly-once publication generation.
    :ivar canonical_binding: Source-rank binding owning the gateway endpoint.
    :ivar source_bindings: Complete source-rank receipt fan-out manifest.
    :ivar request_ready_receipt: Imported request-global readiness authority.
    :ivar encoded_payload: Complete IPC bytes frozen before submission.
    :ivar enqueued_ns: Publisher-process monotonic enqueue timestamp.
    """

    identity: TerminalPublicationIdentity
    canonical_binding: TerminalRequestBinding
    source_bindings: tuple[TerminalRequestBinding, ...]
    request_ready_receipt: TerminalReceipt
    encoded_payload: bytes
    enqueued_ns: int

    def __post_init__(self) -> None:
        """Validate one complete publication handoff."""

        if type(self.identity) is not TerminalPublicationIdentity:
            raise TypeError("identity must be TerminalPublicationIdentity")
        if type(self.canonical_binding) is not TerminalRequestBinding:
            raise TypeError("canonical_binding must be TerminalRequestBinding")
        if type(self.source_bindings) is not tuple or len(self.source_bindings) == 0:
            raise ValueError("source_bindings must be a non-empty tuple")
        if any(
            type(binding) is not TerminalRequestBinding
            for binding in self.source_bindings
        ):
            raise TypeError("source_bindings must contain TerminalRequestBinding")
        if type(self.request_ready_receipt) is not TerminalReceipt:
            raise TypeError("request_ready_receipt must be TerminalReceipt")
        if type(self.encoded_payload) is not bytes or len(self.encoded_payload) == 0:
            raise ValueError("encoded_payload must be non-empty bytes")
        if type(self.enqueued_ns) is not int or self.enqueued_ns < 0:
            raise ValueError("enqueued_ns must be a non-negative integer")

        request_key = self.identity.request_key
        if self.canonical_binding.request_key != request_key:
            raise ValueError("canonical binding belongs to another request")
        if self.canonical_binding.owner.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("canonical binding must belong to a source owner")
        if self.canonical_binding.owner.tp_rank != 0:
            raise ValueError("canonical publication owner must be source rank zero")
        if self.canonical_binding not in self.source_bindings:
            raise ValueError("source manifest omits the canonical binding")
        source_tp_size = self.canonical_binding.owner.tp_size
        source_ranks: list[int] = []
        for binding in self.source_bindings:
            owner = binding.owner
            if binding.request_key != request_key:
                raise ValueError("source manifest spans request generations")
            if owner.role is not TerminalOwnerRole.SOURCE:
                raise ValueError("source manifest contains a decode owner")
            if owner.tp_size != source_tp_size:
                raise ValueError("source manifest disagrees on tensor parallel width")
            source_ranks.append(owner.tp_rank)
        if tuple(source_ranks) != tuple(range(source_tp_size)):
            raise ValueError("source manifest must contain canonical TP-rank order")
        if self.request_ready_receipt.binding != self.canonical_binding:
            raise ValueError("request-ready authority targets another binding")
        if self.request_ready_receipt.kind is not TerminalReceiptKind.REQUEST_READY:
            raise ValueError("publication requires request-ready authority")
        if self.request_ready_receipt.outcome is not TerminalReceiptOutcome.SUCCESS:
            raise ValueError("publication requires successful request readiness")

    @property
    def digest(self) -> bytes:
        """Return the exact identity and payload digest used for deduplication.

        :returns: SHA-256 over immutable publication inputs.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.packed-terminal.gateway-publication.v1")
        digest.update(self.identity.digest)
        digest.update(len(self.source_bindings).to_bytes(4, "big"))
        for binding in self.source_bindings:
            digest.update(binding.digest)
        digest.update(hashlib.sha256(self.encoded_payload).digest())
        return digest.digest()


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalGatewayPublicationSuccess:
    """Successful socket publication and source-owner receipt fan-out.

    :ivar publication: Exact immutable handoff which was sent.
    :ivar completed_ns: Publisher-process monotonic send-completion timestamp.
    :ivar source_receipts: One gateway-published authority per source rank.
    """

    publication: FrozenTerminalGatewayPublication
    completed_ns: int
    source_receipts: tuple[IssuedTerminalWireReceipt, ...]

    def __post_init__(self) -> None:
        """Validate successful publication evidence."""

        if type(self.publication) is not FrozenTerminalGatewayPublication:
            raise TypeError("publication must be FrozenTerminalGatewayPublication")
        if type(self.completed_ns) is not int:
            raise TypeError("completed_ns must be an integer")
        if self.completed_ns < self.publication.enqueued_ns:
            raise ValueError("publication completion precedes enqueue")
        if type(self.source_receipts) is not tuple:
            raise TypeError("source_receipts must be a tuple")
        if len(self.source_receipts) != len(self.publication.source_bindings):
            raise ValueError("publication receipt fan-out is incomplete")
        for binding, receipt in zip(
            self.publication.source_bindings,
            self.source_receipts,
            strict=True,
        ):
            if type(receipt) is not IssuedTerminalWireReceipt:
                raise TypeError("source_receipts must contain issued wire receipts")
            wire_receipt = receipt.wire_receipt
            if (
                wire_receipt.binding != binding
                or wire_receipt.kind is not TerminalReceiptKind.GATEWAY_PUBLISHED
                or wire_receipt.outcome is not TerminalReceiptOutcome.SUCCESS
            ):
                raise ValueError("gateway receipt differs from its source binding")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalGatewayPublicationFailure:
    """Failed or ambiguous gateway publication retained for quarantine.

    :ivar publication: Exact immutable handoff which failed.
    :ivar failed_ns: Publisher-process monotonic failure timestamp.
    :ivar reason: Stable reader-facing failure context.
    :ivar formatted_traceback: Complete send-path traceback.
    """

    publication: FrozenTerminalGatewayPublication
    failed_ns: int
    reason: str
    formatted_traceback: str

    def __post_init__(self) -> None:
        """Validate one publication failure record."""

        if type(self.publication) is not FrozenTerminalGatewayPublication:
            raise TypeError("publication must be FrozenTerminalGatewayPublication")
        if type(self.failed_ns) is not int:
            raise TypeError("failed_ns must be an integer")
        if self.failed_ns < self.publication.enqueued_ns:
            raise ValueError("publication failure precedes enqueue")
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if (
            type(self.formatted_traceback) is not str
            or len(self.formatted_traceback) == 0
        ):
            raise ValueError("formatted_traceback must be a non-empty string")


TerminalGatewayPublicationResult = (
    TerminalGatewayPublicationSuccess | TerminalGatewayPublicationFailure
)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalGatewayPublisherSnapshot:
    """Immutable publisher liveness and retained-identity inventory.

    :ivar disposition: Current process-lifetime publisher disposition.
    :ivar admission_open: Whether new publications are accepted.
    :ivar thread_alive: Whether the publisher execution context is alive.
    :ivar pending_identities: Publication identities awaiting terminal result.
    :ivar completed_identities: Successfully published identities.
    :ivar failed_identities: Failed or ambiguous identities.
    :ivar fatal_reason: First process-fatal reason, if any.
    :ivar fatal_traceback: Complete first process-fatal traceback, if any.
    """

    disposition: TerminalGatewayPublisherDisposition
    admission_open: bool
    thread_alive: bool
    pending_identities: tuple[TerminalPublicationIdentity, ...]
    completed_identities: tuple[TerminalPublicationIdentity, ...]
    failed_identities: tuple[TerminalPublicationIdentity, ...]
    fatal_reason: str | None = None
    fatal_traceback: str | None = None


class TerminalGatewaySink(abc.ABC):
    """Thread-owned byte sink for frozen scheduler IPC payloads."""

    @abc.abstractmethod
    def send(self, encoded_payload: bytes) -> None:
        """Publish one already-serialized output.

        :param encoded_payload: Bytes accepted by the downstream IPC receiver.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Close the sink from the same thread which created it."""


class TerminalGatewaySinkFactory(abc.ABC):
    """Factory invoked inside the publisher execution context."""

    @abc.abstractmethod
    def create(self) -> TerminalGatewaySink:
        """Create one publisher-thread-owned sink.

        :returns: Open sink whose complete lifecycle belongs to this thread.
        """


class _ZmqTerminalGatewaySink(TerminalGatewaySink):
    """PUSH socket and context owned exclusively by the publisher thread."""

    _context: zmq.Context
    _socket: zmq.Socket

    def __init__(self, endpoint: str) -> None:
        """Connect one publisher-owned PUSH socket.

        :param endpoint: Existing downstream PULL endpoint.
        """

        self._context = zmq.Context(io_threads=1)
        socket = get_zmq_socket(
            self._context,
            zmq.PUSH,
            endpoint,
            False,
        )
        if not isinstance(socket, zmq.Socket):
            self._context.term()
            raise RuntimeError("explicit gateway endpoint returned no socket")
        self._socket = socket

    def send(self, encoded_payload: bytes) -> None:
        """Send one frozen payload without touching scheduler-owned objects.

        :param encoded_payload: Complete active-mode IPC encoding.
        """

        self._socket.send(encoded_payload)

    def close(self) -> None:
        """Close the thread-owned socket before terminating its context."""

        self._socket.close(linger=0)
        self._context.term()


@dataclasses.dataclass(frozen=True, slots=True)
class ZmqTerminalGatewaySinkFactory(TerminalGatewaySinkFactory):
    """Create one dedicated gateway PUSH socket inside the publisher thread.

    :ivar endpoint: Existing tokenizer or detokenizer PULL endpoint.
    """

    endpoint: str

    def __post_init__(self) -> None:
        """Validate one explicit ZeroMQ endpoint."""

        if type(self.endpoint) is not str or len(self.endpoint) == 0:
            raise ValueError("endpoint must be a non-empty string")

    def create(self) -> TerminalGatewaySink:
        """Create the thread-owned ZeroMQ sink.

        :returns: Connected publisher-owned sink.
        """

        return _ZmqTerminalGatewaySink(self.endpoint)


_PUBLISHER_STOP = object()


class PackedTerminalOutputPublisher:
    """Process-lifetime exactly-once publisher with no scheduler-owned socket.

    The scheduler freezes complete IPC bytes before enqueue. The publisher
    creates, uses, and closes its sink on its own thread. Any unexpected thread
    exit, queue overflow, conflicting generation, send failure, or listener
    failure is process-fatal and leaves the affected publication identities in
    explicit inventory.
    """

    _capacity: int
    _sink_factory: TerminalGatewaySinkFactory
    _wire_issuer: TerminalWireReceiptIssuer
    _request_ready_ledger: TerminalReceiptLedger
    _result_listener: Callable[[TerminalGatewayPublicationResult], None]
    _fatal_listener: Callable[[str, str | None], None]
    _clock_ns: Callable[[], int]
    _queue: queue.SimpleQueue[FrozenTerminalGatewayPublication | object]
    _known_digests: dict[bytes, bytes]
    _known_ready_tokens: dict[bytes, TerminalReceiptToken]
    _pending: dict[bytes, TerminalPublicationIdentity]
    _completed: dict[bytes, TerminalPublicationIdentity]
    _failed: dict[bytes, TerminalPublicationIdentity]
    _disposition: TerminalGatewayPublisherDisposition
    _admission_open: bool
    _started: bool
    _stop_requested: bool
    _fatal_reason: str | None
    _fatal_traceback: str | None
    _lock: threading.Lock
    _thread: threading.Thread

    def __init__(
        self,
        *,
        capacity: int,
        sink_factory: TerminalGatewaySinkFactory,
        wire_issuer: TerminalWireReceiptIssuer,
        request_ready_authorities: frozenset[TerminalReceiptAuthority],
        result_listener: Callable[[TerminalGatewayPublicationResult], None],
        fatal_listener: Callable[[str, str | None], None],
        clock_ns: Callable[[], int],
        thread_name: str = "packed-terminal-output-publisher",
    ) -> None:
        """Construct a publisher before any request becomes externally ready.

        :param capacity: Maximum pending publication population.
        :param sink_factory: Factory executed inside the publisher thread.
        :param wire_issuer: Canonical source-rank terminal receipt issuer.
        :param request_ready_authorities: Imported coordinator authorities trusted
            by this publisher.
        :param result_listener: Off-forward-loop result delivery callback.
        :param fatal_listener: Process-fatal lifecycle callback.
        :param clock_ns: Publisher-process monotonic clock.
        :param thread_name: Stable publisher thread identity.
        """

        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(sink_factory, TerminalGatewaySinkFactory):
            raise TypeError("sink_factory must inherit TerminalGatewaySinkFactory")
        if type(wire_issuer) is not TerminalWireReceiptIssuer:
            raise TypeError("wire_issuer must be TerminalWireReceiptIssuer")
        if wire_issuer.identity.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("gateway publisher must use a source-rank issuer")
        if wire_issuer.identity.tp_rank != 0:
            raise ValueError("gateway publisher issuer must be source rank zero")
        if (
            type(request_ready_authorities) is not frozenset
            or len(request_ready_authorities) == 0
        ):
            raise ValueError("request_ready_authorities must be non-empty")
        if any(
            type(authority) is not TerminalReceiptAuthority
            for authority in request_ready_authorities
        ):
            raise TypeError("request_ready_authorities contains an invalid value")
        if not callable(result_listener):
            raise TypeError("result_listener must be callable")
        if not callable(fatal_listener):
            raise TypeError("fatal_listener must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if type(thread_name) is not str or len(thread_name) == 0:
            raise ValueError("thread_name must be a non-empty string")

        self._capacity = capacity
        self._sink_factory = sink_factory
        self._wire_issuer = wire_issuer
        self._request_ready_ledger = TerminalReceiptLedger(
            authorities=request_ready_authorities
        )
        self._result_listener = result_listener
        self._fatal_listener = fatal_listener
        self._clock_ns = clock_ns
        self._queue = queue.SimpleQueue()
        self._known_digests = {}
        self._known_ready_tokens = {}
        self._pending = {}
        self._completed = {}
        self._failed = {}
        self._disposition = TerminalGatewayPublisherDisposition.CREATED
        self._admission_open = False
        self._started = False
        self._stop_requested = False
        self._fatal_reason = None
        self._fatal_traceback = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    def start(self) -> None:
        """Start the publisher exactly once."""

        with self._lock:
            if self._started:
                raise TerminalGatewayPublisherError("publisher cannot restart")
            self._started = True
            self._admission_open = True
            self._disposition = TerminalGatewayPublisherDisposition.RUNNING
        self._thread.start()

    def submit(self, publication: FrozenTerminalGatewayPublication) -> bool:
        """Submit one publication or coalesce its exact retransmission.

        :param publication: Complete request-global output handoff.
        :returns: Whether a new publication was enqueued.
        :raises TerminalGatewayPublisherError: If admission is closed, the
            generation conflicts, or the bounded queue overflows.
        """

        if type(publication) is not FrozenTerminalGatewayPublication:
            raise TypeError("publication must be FrozenTerminalGatewayPublication")
        identity_digest = publication.identity.digest
        publication_digest = publication.digest
        ready_token = terminal_receipt_token(publication.request_ready_receipt)
        fatal_reason: str | None = None
        with self._lock:
            if not self._admission_open:
                raise TerminalGatewayPublisherError("publisher admission is closed")
            existing_digest = self._known_digests.get(identity_digest)
            if existing_digest is not None:
                if (
                    existing_digest == publication_digest
                    and self._known_ready_tokens[identity_digest] == ready_token
                ):
                    return False
                fatal_reason = "publication identity was reused with conflicting bytes"
            elif len(self._pending) >= self._capacity:
                self._known_digests[identity_digest] = publication_digest
                self._known_ready_tokens[identity_digest] = ready_token
                self._failed[identity_digest] = publication.identity
                fatal_reason = (
                    f"gateway publication queue exceeded capacity {self._capacity}"
                )
            else:
                self._known_digests[identity_digest] = publication_digest
                self._known_ready_tokens[identity_digest] = ready_token
                self._pending[identity_digest] = publication.identity
                self._queue.put(publication)
        if fatal_reason is not None:
            self._enter_fatal(fatal_reason, None)
            raise TerminalGatewayPublisherError(fatal_reason)
        return True

    def stop_admission_and_join(self, timeout_seconds: float) -> bool:
        """Drain accepted publications and join without polling.

        :param timeout_seconds: Positive thread-join bound.
        :returns: Whether the publisher stopped within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        with self._lock:
            if not self._started:
                raise TerminalGatewayPublisherError("publisher was never started")
            if self._disposition is TerminalGatewayPublisherDisposition.STOPPED:
                return True
            if self._disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL:
                should_enqueue_stop = False
            elif not self._stop_requested:
                self._stop_requested = True
                self._admission_open = False
                self._disposition = TerminalGatewayPublisherDisposition.DRAINING
                should_enqueue_stop = True
            else:
                should_enqueue_stop = False
        if should_enqueue_stop:
            self._queue.put(_PUBLISHER_STOP)
        self._thread.join(timeout=timeout_seconds)
        return not self._thread.is_alive()

    def snapshot(self) -> TerminalGatewayPublisherSnapshot:
        """Return exact publisher liveness and retained identity inventory.

        :returns: Immutable process-lifetime publisher snapshot.
        """

        with self._lock:
            return TerminalGatewayPublisherSnapshot(
                disposition=self._disposition,
                admission_open=self._admission_open,
                thread_alive=self._thread.is_alive(),
                pending_identities=tuple(self._pending.values()),
                completed_identities=tuple(self._completed.values()),
                failed_identities=tuple(self._failed.values()),
                fatal_reason=self._fatal_reason,
                fatal_traceback=self._fatal_traceback,
            )

    def _run(self) -> None:
        """Own the sink and drive every accepted publication to terminality."""

        sink: TerminalGatewaySink | None = None
        current: FrozenTerminalGatewayPublication | None = None
        try:
            sink = self._sink_factory.create()
            while True:
                item = self._queue.get()
                if item is _PUBLISHER_STOP:
                    break
                with self._lock:
                    if (
                        self._disposition
                        is TerminalGatewayPublisherDisposition.PROCESS_FATAL
                    ):
                        break
                if type(item) is not FrozenTerminalGatewayPublication:
                    raise TerminalGatewayPublisherError(
                        "publisher queue contained an invalid item"
                    )
                current = item
                self._publish_one(sink, current)
                current = None
            sink.close()
            sink = None
            with self._lock:
                if (
                    self._disposition
                    is TerminalGatewayPublisherDisposition.PROCESS_FATAL
                ):
                    return
                if len(self._pending) > 0:
                    raise TerminalGatewayPublisherError(
                        "publisher stopped with pending identities"
                    )
                self._disposition = TerminalGatewayPublisherDisposition.STOPPED
        except Exception:
            formatted_traceback = traceback.format_exc()
            reason = "gateway publisher thread failed"
            if current is not None:
                failed_ns = self._clock_ns()
                failure = TerminalGatewayPublicationFailure(
                    publication=current,
                    failed_ns=failed_ns,
                    reason=reason,
                    formatted_traceback=formatted_traceback,
                )
                identity_digest = current.identity.digest
                with self._lock:
                    self._pending.pop(identity_digest, None)
                    self._failed[identity_digest] = current.identity
                try:
                    self._result_listener(failure)
                except Exception:
                    formatted_traceback += "\nresult listener failure:\n"
                    formatted_traceback += traceback.format_exc()
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    formatted_traceback += "\nsink close failure:\n"
                    formatted_traceback += traceback.format_exc()
            self._enter_fatal(reason, formatted_traceback)

    def _publish_one(
        self,
        sink: TerminalGatewaySink,
        publication: FrozenTerminalGatewayPublication,
    ) -> None:
        """Validate readiness, send once, and mint source-rank authority.

        :param sink: Publisher-thread-owned output sink.
        :param publication: Exact queued output handoff.
        """

        if self._wire_issuer.identity != publication.canonical_binding.owner:
            raise TerminalGatewayPublisherError(
                "gateway receipt issuer differs from the canonical source owner"
            )
        self._request_ready_ledger = self._request_ready_ledger.consume(
            publication.request_ready_receipt,
            publication.canonical_binding,
            TerminalReceiptKind.REQUEST_READY,
            TerminalReceiptOutcome.SUCCESS,
        )
        sink.send(publication.encoded_payload)
        completed_ns = self._clock_ns()
        receipts = tuple(
            self._wire_issuer.issue(
                binding=binding,
                kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
                outcome=TerminalReceiptOutcome.SUCCESS,
                terminal_timestamp_ns=completed_ns,
            )
            for binding in publication.source_bindings
        )
        success = TerminalGatewayPublicationSuccess(
            publication=publication,
            completed_ns=completed_ns,
            source_receipts=receipts,
        )
        self._result_listener(success)
        identity_digest = publication.identity.digest
        with self._lock:
            if self._pending.pop(identity_digest, None) is None:
                raise TerminalGatewayPublisherError(
                    "published identity was absent from pending inventory"
                )
            self._completed[identity_digest] = publication.identity

    def _enter_fatal(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Record first fatal state and notify process lifecycle ownership.

        :param reason: Stable first-failure reason.
        :param formatted_traceback: Complete traceback when failure was raised.
        """

        should_notify = False
        should_wake = False
        with self._lock:
            if (
                self._disposition
                is not TerminalGatewayPublisherDisposition.PROCESS_FATAL
            ):
                self._admission_open = False
                self._disposition = TerminalGatewayPublisherDisposition.PROCESS_FATAL
                self._fatal_reason = reason
                self._fatal_traceback = formatted_traceback
                should_notify = True
                should_wake = self._started
        if should_wake:
            self._queue.put(_PUBLISHER_STOP)
        if should_notify:
            try:
                self._fatal_listener(reason, formatted_traceback)
            except Exception:
                listener_traceback = traceback.format_exc()
                with self._lock:
                    existing = self._fatal_traceback
                    prefix = "" if existing is None else existing + "\n"
                    self._fatal_traceback = (
                        prefix + "fatal listener failure:\n" + listener_traceback
                    )
