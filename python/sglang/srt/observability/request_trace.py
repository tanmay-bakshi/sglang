import dataclasses
import enum
import functools
import itertools
import json
import logging
import os
import socket
import time

logger = logging.getLogger(__name__)

REQUEST_TRACE_ENV = "SGLANG_REQUEST_TRACE"
REQUEST_TRACE_PREFIX = "SGLANG_REQUEST_TRACE "
REQUEST_TRACE_SCHEMA_VERSION = 1


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


_sequence = itertools.count()


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

    record: dict[str, object] = {
        "schema_version": REQUEST_TRACE_SCHEMA_VERSION,
        "event": event.value,
        "role": role.value,
        "monotonic_ns": time.monotonic_ns(),
        "sequence": next(_sequence),
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "request_ids": list(request_ids),
        "bootstrap_rooms": list(bootstrap_rooms),
        "request_generations": list(request_generations),
    }
    resolved_fields = RequestTraceFields() if fields is None else fields
    for field in dataclasses.fields(resolved_fields):
        value = getattr(resolved_fields, field.name)
        if value is not None:
            record[field.name] = value
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
