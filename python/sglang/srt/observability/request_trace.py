import atexit
import dataclasses
import enum
import functools
import itertools
import json
import logging
import os
import socket
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

REQUEST_TRACE_ENV = "SGLANG_REQUEST_TRACE"
REQUEST_TRACE_PREFIX = "SGLANG_REQUEST_TRACE "
REQUEST_TRACE_SCHEMA_VERSION = 1
_REQUEST_TRACE_DRAIN_INTERVAL_SECONDS = 0.25


class RequestTraceEvent(enum.StrEnum):
    """Closed event names emitted by request-correlated tracing."""

    PREFILL_BATCH_COMPLETED = "prefill_batch_completed"
    PACKED_TRANSFER_BEGIN = "packed_transfer_begin"
    PACKED_TRANSFER_END = "packed_transfer_end"
    DECODE_HANDOFF_TOKEN_READY = "decode_handoff_token_ready"
    DECODE_FIRST_ISSUE = "decode_first_issue"
    DECODE_FIRST_RESULT = "decode_first_result"


class RequestTraceRole(enum.StrEnum):
    """Process roles participating in one request timeline."""

    PREFILL_SCHEDULER = "prefill_scheduler"
    PREFILL_TRANSFER = "prefill_transfer"
    DECODE_SCHEDULER = "decode_scheduler"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RequestTraceFields:
    """Typed optional fields shared by the closed request-trace events.

    :ivar process_rank: Role-local tensor-parallel rank.
    :ivar batch_size: Number of requests occupying the scheduled batch.
    :ivar batch_token_count: Total scheduled input or issued decode tokens.
    :ivar forward_iter: Process-local scheduler forward sequence.
    :ivar cuda_graph_active: Whether the measured forward used a CUDA graph.
    :ivar accepted_token_count: Tokens accepted by the first decode result.
    :ivar copy_group_count: Packed source copy groups carrying payload.
    :ivar payload_bytes: Logical packed payload bytes.
    :ivar transport_bytes: Physical packed transport bytes.
    """

    process_rank: int | None = None
    batch_size: int | None = None
    batch_token_count: int | None = None
    forward_iter: int | None = None
    cuda_graph_active: bool | None = None
    accepted_token_count: int | None = None
    copy_group_count: int | None = None
    payload_bytes: int | None = None
    transport_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate non-negative counters without coercing caller values."""

        integer_fields = (
            ("process_rank", self.process_rank),
            ("batch_size", self.batch_size),
            ("batch_token_count", self.batch_token_count),
            ("forward_iter", self.forward_iter),
            ("accepted_token_count", self.accepted_token_count),
            ("copy_group_count", self.copy_group_count),
            ("payload_bytes", self.payload_bytes),
            ("transport_bytes", self.transport_bytes),
        )
        for name, value in integer_fields:
            if value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.cuda_graph_active is not None
            and type(self.cuda_graph_active) is not bool
        ):
            raise TypeError("cuda_graph_active must be a boolean")


@functools.cache
def request_trace_enabled() -> bool:
    """Return whether request tracing was explicitly enabled for this process.

    :returns: ``True`` only for an accepted affirmative environment value.
    :raises ValueError: If the environment value is not part of the closed contract.
    """

    value = os.environ.get(REQUEST_TRACE_ENV)
    if value is None or len(value) == 0:
        return False
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{REQUEST_TRACE_ENV} must be one of 1/0, true/false, yes/no, or on/off"
    )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _QueuedRequestTrace:
    """Immutable trace boundary awaiting serialization and log emission.

    :ivar event: Closed event identifier.
    :ivar role: Process role emitting the event.
    :ivar monotonic_ns: Boundary timestamp on the machine-wide monotonic clock.
    :ivar sequence: Process-local boundary order.
    :ivar request_ids: Stable engine request identifiers.
    :ivar bootstrap_rooms: Decoder-minted bootstrap rooms.
    :ivar request_generations: Replay-safe packed request generations.
    :ivar fields: Event-specific typed measurements.
    """

    event: RequestTraceEvent
    role: RequestTraceRole
    monotonic_ns: int
    sequence: int
    request_ids: tuple[str, ...]
    bootstrap_rooms: tuple[int | None, ...]
    request_generations: tuple[str | None, ...]
    fields: RequestTraceFields


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _RequestTraceFlush:
    """FIFO barrier used to prove that preceding trace records are durable.

    :ivar completed: Event set after all preceding records have been logged.
    """

    completed: threading.Event


class _RequestTraceStop:
    """Terminal queue marker for orderly writer shutdown."""


_REQUEST_TRACE_STOP = _RequestTraceStop()


class _RequestTraceWriter:
    """Single lossless writer that keeps log I/O off inference threads."""

    _hostname: str
    _process_id: int
    _queue: deque[_QueuedRequestTrace | _RequestTraceFlush | _RequestTraceStop]
    _sequence: itertools.count
    _enqueue_lock: threading.Lock
    _closed: bool
    _wake_writer: threading.Event
    _thread: threading.Thread

    def __init__(self) -> None:
        """Start the process-local trace writer."""

        self._hostname = socket.gethostname()
        self._process_id = os.getpid()
        self._queue = deque()
        self._sequence = itertools.count()
        self._enqueue_lock = threading.Lock()
        self._closed = False
        self._wake_writer = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-request-trace-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def process_id(self) -> int:
        """Return the process that owns this writer.

        :returns: Owning process identifier.
        """

        return self._process_id

    def enqueue(
        self,
        event: RequestTraceEvent,
        role: RequestTraceRole,
        *,
        request_ids: tuple[str, ...],
        bootstrap_rooms: tuple[int | None, ...],
        request_generations: tuple[str | None, ...],
        fields: RequestTraceFields,
    ) -> None:
        """Capture and enqueue one boundary without formatting or log I/O.

        Timestamp, sequence assignment, and FIFO insertion share one lock so
        concurrent producers cannot reorder records within a process.

        :param event: Closed event identifier.
        :param role: Process role emitting the event.
        :param request_ids: Stable engine request identifiers.
        :param bootstrap_rooms: Decoder-minted bootstrap rooms.
        :param request_generations: Replay-safe packed request generations.
        :param fields: Event-specific typed measurements.
        :raises RuntimeError: If the process-local writer has already stopped.
        """

        with self._enqueue_lock:
            if self._closed:
                raise RuntimeError("request trace writer is closed")
            self._queue.append(
                _QueuedRequestTrace(
                    event=event,
                    role=role,
                    monotonic_ns=time.monotonic_ns(),
                    sequence=next(self._sequence),
                    request_ids=request_ids,
                    bootstrap_rooms=bootstrap_rooms,
                    request_generations=request_generations,
                    fields=fields,
                )
            )

    def flush(self, timeout_seconds: float) -> None:
        """Wait until every record enqueued before this call is logged.

        :param timeout_seconds: Maximum drain wait.
        :raises RuntimeError: If the writer has already stopped.
        :raises TimeoutError: If the writer does not reach the FIFO barrier.
        """

        completed = threading.Event()
        with self._enqueue_lock:
            if self._closed:
                raise RuntimeError("request trace writer is closed")
            self._queue.append(_RequestTraceFlush(completed=completed))
        self._wake_writer.set()
        if not completed.wait(timeout_seconds):
            raise TimeoutError(
                f"request trace writer did not drain within {timeout_seconds} seconds"
            )

    def close(self, timeout_seconds: float) -> None:
        """Drain all accepted records and stop the writer thread.

        :param timeout_seconds: Maximum writer shutdown wait.
        :raises TimeoutError: If the writer does not stop after the terminal marker.
        """

        with self._enqueue_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.append(_REQUEST_TRACE_STOP)
        self._wake_writer.set()
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError(
                f"request trace writer did not stop within {timeout_seconds} seconds"
            )

    def _run(self) -> None:
        """Serialize and log records in captured process order."""

        while True:
            self._wake_writer.wait(_REQUEST_TRACE_DRAIN_INTERVAL_SECONDS)
            self._wake_writer.clear()
            while True:
                try:
                    item = self._queue.popleft()
                except IndexError:
                    break
                if item is _REQUEST_TRACE_STOP:
                    return
                if isinstance(item, _RequestTraceFlush):
                    item.completed.set()
                    continue
                _log_request_trace(item, self._hostname, self._process_id)


_request_trace_writer_instance: _RequestTraceWriter | None = None
_request_trace_writer_lock = threading.Lock()


def _request_trace_writer() -> _RequestTraceWriter:
    """Return the lazy writer owned by the calling process.

    :returns: Process-local request trace writer.
    """

    global _request_trace_writer_instance
    writer = _request_trace_writer_instance
    process_id = os.getpid()
    if writer is not None and writer.process_id == process_id:
        return writer
    with _request_trace_writer_lock:
        writer = _request_trace_writer_instance
        if writer is not None and writer.process_id == process_id:
            return writer
        writer = _RequestTraceWriter()
        _request_trace_writer_instance = writer
        return writer


def _reset_request_trace_writer_after_fork() -> None:
    """Discard inherited writer state whose thread does not survive fork."""

    global _request_trace_writer_instance, _request_trace_writer_lock
    _request_trace_writer_instance = None
    _request_trace_writer_lock = threading.Lock()


def flush_request_traces(timeout_seconds: float = 5.0) -> None:
    """Make all request traces accepted so far durable in the service log.

    :param timeout_seconds: Maximum drain wait.
    """

    writer = _request_trace_writer_instance
    if writer is None or writer.process_id != os.getpid():
        return
    writer.flush(timeout_seconds)


def shutdown_request_trace_writer(timeout_seconds: float = 5.0) -> None:
    """Drain and stop the calling process's request-trace writer.

    :param timeout_seconds: Maximum writer shutdown wait.
    """

    global _request_trace_writer_instance
    with _request_trace_writer_lock:
        writer = _request_trace_writer_instance
        if writer is None or writer.process_id != os.getpid():
            return
        _request_trace_writer_instance = None
    writer.close(timeout_seconds)


def _shutdown_request_trace_writer_at_exit() -> None:
    """Best-effort orderly drain during normal interpreter shutdown."""

    try:
        shutdown_request_trace_writer()
    except TimeoutError:
        logger.exception("Request trace writer did not drain during shutdown")


os.register_at_fork(after_in_child=_reset_request_trace_writer_after_fork)
atexit.register(_shutdown_request_trace_writer_at_exit)


def emit_request_trace(
    event: RequestTraceEvent,
    role: RequestTraceRole,
    *,
    request_ids: tuple[str, ...] = (),
    bootstrap_rooms: tuple[int | None, ...] = (),
    request_generations: tuple[str | None, ...] = (),
    fields: RequestTraceFields | None = None,
) -> None:
    """Emit one compact JSON record on the machine-wide monotonic clock.

    The service logger remains the durability owner, so each worker's existing
    captured log is also its trace shard. ``bootstrap_rooms`` aligns with
    ``request_ids`` when both are present. Packed worker events may omit a
    request id and instead use the replay-safe room plus request generation.

    :param event: Closed event identifier.
    :param role: Process role emitting the event.
    :param request_ids: Stable engine request identifiers.
    :param bootstrap_rooms: Decoder-minted bootstrap rooms.
    :param request_generations: Replay-safe packed request generations.
    :param fields: Event-specific typed measurements.
    :raises TypeError: If an event or role is not a closed enum value.
    :raises ValueError: If request correlation fields are malformed.
    """

    if not request_trace_enabled():
        return
    if type(event) is not RequestTraceEvent:
        raise TypeError("event must be RequestTraceEvent")
    if type(role) is not RequestTraceRole:
        raise TypeError("role must be RequestTraceRole")
    _validate_correlations(request_ids, bootstrap_rooms, request_generations)

    resolved_fields = RequestTraceFields() if fields is None else fields
    _request_trace_writer().enqueue(
        event,
        role,
        request_ids=request_ids,
        bootstrap_rooms=bootstrap_rooms,
        request_generations=request_generations,
        fields=resolved_fields,
    )


def _log_request_trace(
    trace: _QueuedRequestTrace,
    hostname: str,
    process_id: int,
) -> None:
    """Materialize and log one trace record outside inference threads.

    :param trace: Captured immutable trace boundary.
    :param hostname: Kernel hostname cached by the process-local writer.
    :param process_id: Process identifier cached by the writer.
    """

    record: dict[str, object] = {
        "schema_version": REQUEST_TRACE_SCHEMA_VERSION,
        "event": trace.event.value,
        "role": trace.role.value,
        "monotonic_ns": trace.monotonic_ns,
        "sequence": trace.sequence,
        "hostname": hostname,
        "process_id": process_id,
        "request_ids": list(trace.request_ids),
        "bootstrap_rooms": list(trace.bootstrap_rooms),
        "request_generations": list(trace.request_generations),
    }
    fields = trace.fields
    if fields.process_rank is not None:
        record["process_rank"] = fields.process_rank
    if fields.batch_size is not None:
        record["batch_size"] = fields.batch_size
    if fields.batch_token_count is not None:
        record["batch_token_count"] = fields.batch_token_count
    if fields.forward_iter is not None:
        record["forward_iter"] = fields.forward_iter
    if fields.cuda_graph_active is not None:
        record["cuda_graph_active"] = fields.cuda_graph_active
    if fields.accepted_token_count is not None:
        record["accepted_token_count"] = fields.accepted_token_count
    if fields.copy_group_count is not None:
        record["copy_group_count"] = fields.copy_group_count
    if fields.payload_bytes is not None:
        record["payload_bytes"] = fields.payload_bytes
    if fields.transport_bytes is not None:
        record["transport_bytes"] = fields.transport_bytes
    logger.info(
        "%s%s",
        REQUEST_TRACE_PREFIX,
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    )


def _validate_correlations(
    request_ids: tuple[str, ...],
    bootstrap_rooms: tuple[int | None, ...],
    request_generations: tuple[str | None, ...],
) -> None:
    """Validate aligned request correlation vectors.

    :param request_ids: Stable engine request identifiers.
    :param bootstrap_rooms: Decoder-minted bootstrap rooms.
    :param request_generations: Replay-safe packed request generations.
    :raises ValueError: If a vector is empty, malformed, or misaligned.
    """

    if len(request_ids) == 0 and len(bootstrap_rooms) == 0:
        raise ValueError("request trace requires a request id or bootstrap room")
    if len(request_ids) > 0 and any(len(request_id) == 0 for request_id in request_ids):
        raise ValueError("request trace request ids must not be empty")
    if len(bootstrap_rooms) > 0:
        for room in bootstrap_rooms:
            if room is not None and (type(room) is not int or room < 0):
                raise ValueError("bootstrap rooms must be non-negative integers")
    if (
        len(request_ids) > 0
        and len(bootstrap_rooms) > 0
        and len(request_ids) != len(bootstrap_rooms)
    ):
        raise ValueError("request ids and bootstrap rooms must align")
    if len(request_generations) > 0:
        if len(request_generations) != len(bootstrap_rooms):
            raise ValueError("request generations and bootstrap rooms must align")
        for generation in request_generations:
            if generation is not None and len(generation) != 32:
                raise ValueError(
                    "packed request generations must be 16-byte hex strings"
                )
