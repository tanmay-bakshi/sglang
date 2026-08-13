import abc
import dataclasses
import enum
import hashlib
import math
import queue
import threading
import traceback
from collections.abc import Callable

import zmq
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    BoundTerminalDeadline,
    TerminalDeadlineKind,
    start_terminal_deadline,
)
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
        if (
            self.identity.publisher_process_generation
            != self.canonical_binding.owner.process_generation
        ):
            raise ValueError("publication identity belongs to another publisher")
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
                or wire_receipt.issuer != self.publication.canonical_binding.owner
                or wire_receipt.kind is not TerminalReceiptKind.GATEWAY_PUBLISHED
                or wire_receipt.outcome is not TerminalReceiptOutcome.SUCCESS
                or wire_receipt.terminal_timestamp_ns != self.completed_ns
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

    def __post_init__(self) -> None:
        """Validate one conservation-complete publisher snapshot."""

        if type(self.disposition) is not TerminalGatewayPublisherDisposition:
            raise TypeError("disposition must be TerminalGatewayPublisherDisposition")
        if type(self.admission_open) is not bool:
            raise TypeError("admission_open must be bool")
        if type(self.thread_alive) is not bool:
            raise TypeError("thread_alive must be bool")
        inventories = (
            self.pending_identities,
            self.completed_identities,
            self.failed_identities,
        )
        if any(type(inventory) is not tuple for inventory in inventories):
            raise TypeError("publisher inventories must be tuples")
        if any(
            type(identity) is not TerminalPublicationIdentity
            for inventory in inventories
            for identity in inventory
        ):
            raise TypeError("publisher inventories contain an invalid identity")
        inventory_digests = tuple(
            frozenset(identity.digest for identity in inventory)
            for inventory in inventories
        )
        if any(
            len(digests) != len(inventory)
            for digests, inventory in zip(inventory_digests, inventories, strict=True)
        ):
            raise ValueError("publisher inventory contains a duplicate identity")
        pending, completed, failed = inventory_digests
        if (
            len(pending & completed) > 0
            or len(pending & failed) > 0
            or len(completed & failed) > 0
        ):
            raise ValueError("publisher inventories are not disjoint")
        if self.admission_open and (
            self.disposition is not TerminalGatewayPublisherDisposition.RUNNING
        ):
            raise ValueError("publisher admission requires running disposition")
        if self.disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL:
            if type(self.fatal_reason) is not str or len(self.fatal_reason) == 0:
                raise ValueError("process-fatal publisher requires a reason")
            if (
                self.fatal_traceback is not None
                and type(self.fatal_traceback) is not str
            ):
                raise TypeError("fatal_traceback must be a string or None")
        elif self.fatal_reason is not None or self.fatal_traceback is not None:
            raise ValueError("healthy publisher cannot retain fatal evidence")


class TerminalGatewaySink(abc.ABC):
    """Thread-owned byte sink for frozen scheduler IPC payloads."""

    @abc.abstractmethod
    def send(self, encoded_payload: bytes, timeout_seconds: float) -> None:
        """Publish one already-serialized output.

        :param encoded_payload: Bytes accepted by the downstream IPC receiver.
        :param timeout_seconds: Remaining hash-bound publication deadline.
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

    def send(self, encoded_payload: bytes, timeout_seconds: float) -> None:
        """Send one frozen payload without touching scheduler-owned objects.

        :param encoded_payload: Complete active-mode IPC encoding.
        :param timeout_seconds: Remaining hash-bound publication deadline.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        timeout_ms = max(1, math.ceil(timeout_seconds * 1_000.0))
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
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
    _pending: dict[bytes, FrozenTerminalGatewayPublication]
    _completed: dict[bytes, TerminalPublicationIdentity]
    _failed: dict[bytes, TerminalPublicationIdentity]
    _disposition: TerminalGatewayPublisherDisposition
    _admission_open: bool
    _started: bool
    _thread_launched: bool
    _stop_requested: bool
    _shutdown_deadline: BoundTerminalDeadline | None
    _fatal_reason: str | None
    _fatal_traceback: str | None
    _transition_lock: threading.RLock
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
        self._thread_launched = False
        self._stop_requested = False
        self._shutdown_deadline = None
        self._fatal_reason = None
        self._fatal_traceback = None
        self._transition_lock = threading.RLock()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    def start(self) -> None:
        """Start the publisher exactly once."""

        with self._transition_lock:
            with self._lock:
                if self._started:
                    raise TerminalGatewayPublisherError("publisher cannot restart")
                self._started = True
                self._admission_open = True
                self._disposition = TerminalGatewayPublisherDisposition.RUNNING
            try:
                self._thread.start()
            except BaseException:
                formatted_traceback = traceback.format_exc()
                self._enter_fatal(
                    "gateway publisher thread failed to start",
                    formatted_traceback,
                )
                raise
            with self._lock:
                self._thread_launched = True

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
        should_notify = False
        should_wake = False
        with self._transition_lock, self._lock:
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
                fatal_reason = (
                    f"gateway publication queue exceeded capacity {self._capacity}"
                )
            else:
                self._known_digests[identity_digest] = publication_digest
                self._known_ready_tokens[identity_digest] = ready_token
                self._pending[identity_digest] = publication
                self._queue.put(publication)
            if fatal_reason is not None:
                additional_failed = ()
                if identity_digest not in self._completed:
                    additional_failed = (publication,)
                should_notify, should_wake = self._record_fatal_locked(
                    fatal_reason,
                    None,
                    additional_failed,
                )
        if fatal_reason is not None:
            self._finish_fatal_transition(
                should_notify,
                should_wake,
                fatal_reason,
                None,
            )
            raise TerminalGatewayPublisherError(fatal_reason)
        return True

    def stop_admission_and_join(self) -> bool:
        """Drain accepted publications under the frozen shutdown deadline.

        :returns: Whether the publisher stopped within the hash-bound deadline.
        """

        with self._transition_lock, self._lock:
            if not self._started:
                raise TerminalGatewayPublisherError("publisher was never started")
            if self._disposition is TerminalGatewayPublisherDisposition.STOPPED:
                return True
            if self._shutdown_deadline is None:
                self._shutdown_deadline = start_terminal_deadline(
                    TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
                    self._clock_ns(),
                )
            shutdown_deadline = self._shutdown_deadline
            if self._disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL:
                should_enqueue_stop = False
            elif not self._stop_requested:
                self._stop_requested = True
                self._admission_open = False
                self._disposition = TerminalGatewayPublisherDisposition.DRAINING
                should_enqueue_stop = True
            else:
                should_enqueue_stop = False
            thread_launched = self._thread_launched
        if should_enqueue_stop:
            self._queue.put(_PUBLISHER_STOP)
        if not thread_launched:
            return True
        timeout_seconds = self._remaining_deadline_seconds(shutdown_deadline)
        if timeout_seconds > 0.0:
            self._thread.join(timeout=timeout_seconds)
        if not self._thread.is_alive():
            return True
        reason = shutdown_deadline.spec.timeout_outcome
        self._enter_fatal(reason, None)
        return False

    def snapshot(self) -> TerminalGatewayPublisherSnapshot:
        """Return exact publisher liveness and retained identity inventory.

        :returns: Immutable process-lifetime publisher snapshot.
        """

        with self._lock:
            return TerminalGatewayPublisherSnapshot(
                disposition=self._disposition,
                admission_open=self._admission_open,
                thread_alive=self._thread.is_alive(),
                pending_identities=tuple(
                    publication.identity for publication in self._pending.values()
                ),
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
            closing_sink = sink
            sink = None
            closing_sink.close()
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
        # SystemExit from an injected sink or listener is still publisher death.
        except BaseException:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            additional_tracebacks: list[str] = []
            reason = "gateway publisher thread failed"
            self._enter_fatal(reason, formatted_traceback)
            if current is not None:
                try:
                    failed_ns = max(current.enqueued_ns, self._clock_ns())
                except BaseException:  # noqa: BLE001
                    timestamp_traceback = (
                        "failure timestamp error:\n" + traceback.format_exc()
                    )
                    additional_tracebacks.append(timestamp_traceback)
                    formatted_traceback += "\n" + timestamp_traceback
                    failed_ns = current.enqueued_ns
                failure = TerminalGatewayPublicationFailure(
                    publication=current,
                    failed_ns=failed_ns,
                    reason=reason,
                    formatted_traceback=formatted_traceback,
                )
                try:
                    with self._transition_lock:
                        self._result_listener(failure)
                except BaseException:  # noqa: BLE001
                    additional_tracebacks.append(
                        "result listener failure:\n" + traceback.format_exc()
                    )
            if sink is not None:
                closing_sink = sink
                sink = None
                try:
                    closing_sink.close()
                except BaseException:  # noqa: BLE001
                    additional_tracebacks.append(
                        "sink close failure:\n" + traceback.format_exc()
                    )
            if len(additional_tracebacks) > 0:
                self._append_fatal_traceback("\n".join(additional_tracebacks))
        finally:
            with self._lock:
                disposition = self._disposition
            if disposition not in (
                TerminalGatewayPublisherDisposition.STOPPED,
                TerminalGatewayPublisherDisposition.PROCESS_FATAL,
            ):
                self._enter_fatal(
                    "gateway publisher thread exited without a terminal disposition",
                    None,
                )

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
        identity_digest = publication.identity.digest
        with self._lock:
            if self._disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL:
                raise TerminalGatewayPublisherError(
                    "publisher entered process-fatal state before publication"
                )
            if self._pending.get(identity_digest) is not publication:
                raise TerminalGatewayPublisherError(
                    "queued identity differs from pending publication authority"
                )
        self._request_ready_ledger = self._request_ready_ledger.consume(
            publication.request_ready_receipt,
            publication.canonical_binding,
            TerminalReceiptKind.REQUEST_READY,
            TerminalReceiptOutcome.SUCCESS,
        )
        deadline = start_terminal_deadline(
            TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
            publication.enqueued_ns,
        )
        timeout_seconds = self._remaining_deadline_seconds(deadline)
        if timeout_seconds <= 0.0:
            raise TerminalGatewayPublisherError(deadline.spec.timeout_outcome)
        sink.send(publication.encoded_payload, timeout_seconds)
        completed_ns = self._clock_ns()
        if completed_ns >= deadline.expires_ns:
            raise TerminalGatewayPublisherError(deadline.spec.timeout_outcome)
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
        with self._transition_lock:
            with self._lock:
                if (
                    self._disposition
                    is TerminalGatewayPublisherDisposition.PROCESS_FATAL
                ):
                    raise TerminalGatewayPublisherError(
                        "publication completed after a process-fatal transition"
                    )
                if self._pending.get(identity_digest) is not publication:
                    raise TerminalGatewayPublisherError(
                        "published identity lost pending authority"
                    )
            self._result_listener(success)
            with self._lock:
                if (
                    self._disposition
                    is TerminalGatewayPublisherDisposition.PROCESS_FATAL
                ):
                    raise TerminalGatewayPublisherError(
                        "publication result delivery entered process-fatal state"
                    )
                if self._pending.pop(identity_digest, None) is not publication:
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

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        if formatted_traceback is not None and type(formatted_traceback) is not str:
            raise TypeError("formatted_traceback must be a string or None")
        with self._transition_lock, self._lock:
            should_notify, should_wake = self._record_fatal_locked(
                reason,
                formatted_traceback,
                (),
            )
        self._finish_fatal_transition(
            should_notify,
            should_wake,
            reason,
            formatted_traceback,
        )

    def _record_fatal_locked(
        self,
        reason: str,
        formatted_traceback: str | None,
        additional_failed: tuple[FrozenTerminalGatewayPublication, ...],
    ) -> tuple[bool, bool]:
        """Record one atomic fatal transition while lifecycle locks are held.

        :param reason: Stable first-failure reason.
        :param formatted_traceback: Complete traceback when failure was raised.
        :param additional_failed: Rejected publications not in pending inventory.
        :returns: Whether to notify lifecycle ownership and wake the thread.
        """

        if self._disposition is TerminalGatewayPublisherDisposition.PROCESS_FATAL:
            return (False, False)
        self._admission_open = False
        self._disposition = TerminalGatewayPublisherDisposition.PROCESS_FATAL
        self._fatal_reason = reason
        self._fatal_traceback = formatted_traceback
        for identity_digest, publication in self._pending.items():
            self._failed[identity_digest] = publication.identity
        self._pending.clear()
        for publication in additional_failed:
            identity_digest = publication.identity.digest
            if identity_digest in self._completed:
                continue
            self._failed[identity_digest] = publication.identity
        return (True, self._thread_launched)

    def _finish_fatal_transition(
        self,
        should_notify: bool,
        should_wake: bool,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Wake and notify after a fatal inventory transition releases locks.

        :param should_notify: Whether this is the first fatal transition.
        :param should_wake: Whether the publisher thread was launched.
        :param reason: Stable first-failure reason.
        :param formatted_traceback: Complete traceback when failure was raised.
        """

        if should_wake:
            self._queue.put(_PUBLISHER_STOP)
        if not should_notify:
            return
        try:
            self._fatal_listener(reason, formatted_traceback)
        # A lifecycle callback cannot silently kill the publication owner.
        except BaseException:  # noqa: BLE001
            listener_traceback = traceback.format_exc()
            self._append_fatal_traceback(
                "fatal listener failure:\n" + listener_traceback
            )

    def _append_fatal_traceback(self, formatted_traceback: str) -> None:
        """Retain additional fatal-path evidence without changing first cause.

        :param formatted_traceback: Additional complete failure context.
        """

        if type(formatted_traceback) is not str or len(formatted_traceback) == 0:
            raise ValueError("formatted_traceback must be a non-empty string")
        with self._lock:
            existing = self._fatal_traceback
            if existing is None:
                self._fatal_traceback = formatted_traceback
                return
            if formatted_traceback == existing:
                return
            self._fatal_traceback = existing + "\n" + formatted_traceback

    def _remaining_deadline_seconds(
        self,
        deadline: BoundTerminalDeadline,
    ) -> float:
        """Return a nonnegative remainder without resetting a frozen deadline.

        :param deadline: One-shot hash-bound phase deadline.
        :returns: Remaining seconds, or zero after expiry.
        """

        if type(deadline) is not BoundTerminalDeadline:
            raise TypeError("deadline must be BoundTerminalDeadline")
        now_ns = self._clock_ns()
        if now_ns < deadline.started_ns:
            raise TerminalGatewayPublisherError(
                "publisher clock precedes the frozen deadline anchor"
            )
        remaining_ns = max(0, deadline.expires_ns - now_ns)
        return float(remaining_ns) / 1_000_000_000.0
