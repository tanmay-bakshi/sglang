import dataclasses
import json
import logging
import math
import traceback
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_DIGEST_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TERMINAL_PROCESS_GENERATION_BYTES,
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerTimingField,
)

TERMINAL_PROGRESS_TIMING_SCHEMA_VERSION = 1
TERMINAL_PROGRESS_TIMING_LOG_PREFIX = "TerminalProgressTiming:"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalProgressTimingSample:
    """One process-local interval projected into terminal smoke evidence.

    :ivar binding: Exact request and process generation measured locally.
    :ivar field: Frozen terminal timing field represented by the interval.
    :ivar sample_key: Stable cardinality key within one binding and field.
    :ivar started_ns: Process-local ``CLOCK_MONOTONIC_RAW`` start timestamp.
    :ivar completed_ns: Process-local ``CLOCK_MONOTONIC_RAW`` end timestamp.
    """

    binding: TerminalRequestBinding
    field: TerminalOwnerTimingField
    sample_key: str
    started_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        """Validate one complete same-process timing interval."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.field) is not TerminalOwnerTimingField:
            raise TypeError("field must be TerminalOwnerTimingField")
        if type(self.sample_key) is not str or len(self.sample_key) == 0:
            raise ValueError("sample_key must be a non-empty string")
        if type(self.started_ns) is not int or self.started_ns < 0:
            raise ValueError("started_ns must be a non-negative integer")
        if type(self.completed_ns) is not int or self.completed_ns < self.started_ns:
            raise ValueError("completed_ns must not precede started_ns")

    @property
    def duration_ms(self) -> float:
        """Return the process-local interval in milliseconds.

        :returns: Finite non-negative interval duration.
        """

        return (self.completed_ns - self.started_ns) / 1_000_000.0

    @property
    def cardinality_key(self) -> tuple[bytes, TerminalOwnerTimingField, str]:
        """Return the unique sample identity within one request.

        :returns: Binding, field, and writer/handle/rank sample identity.
        """

        return (self.binding.digest, self.field, self.sample_key)

    def to_payload(self) -> dict[str, object]:
        """Serialize the stable production evidence record.

        :returns: JSON-compatible identity and interval fields.
        """

        owner = self.binding.owner
        request_key = self.binding.request_key
        return {
            "schema_version": TERMINAL_PROGRESS_TIMING_SCHEMA_VERSION,
            "room_id": request_key.room_id,
            "request_generation": request_key.request_generation.hex(),
            "binding_digest": self.binding.digest.hex(),
            "rank_manifest_digest": self.binding.rank_manifest_digest.hex(),
            "allocation_digest": self.binding.allocation_digest.hex(),
            "process_generation": owner.process_generation.hex(),
            "role": owner.role.value,
            "tp_rank": owner.tp_rank,
            "tp_size": owner.tp_size,
            "field": self.field.value,
            "sample_key": self.sample_key,
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "TerminalProgressTimingSample":
        """Parse and authenticate one stable evidence payload.

        :param payload: Candidate JSON object from a production log.
        :returns: Validated process-local interval.
        """

        expected_fields = {
            "schema_version",
            "room_id",
            "request_generation",
            "binding_digest",
            "rank_manifest_digest",
            "allocation_digest",
            "process_generation",
            "role",
            "tp_rank",
            "tp_size",
            "field",
            "sample_key",
            "started_ns",
            "completed_ns",
            "duration_ms",
        }
        if type(payload) is not dict or set(payload) != expected_fields:
            raise ValueError("terminal timing payload fields differ from schema")
        if payload["schema_version"] != TERMINAL_PROGRESS_TIMING_SCHEMA_VERSION:
            raise ValueError("terminal timing schema version differs")

        request_generation = _decode_hex(
            payload["request_generation"],
            PACKED_REQUEST_GENERATION_BYTES,
            "request_generation",
        )
        process_generation = _decode_hex(
            payload["process_generation"],
            TERMINAL_PROCESS_GENERATION_BYTES,
            "process_generation",
        )
        rank_manifest_digest = _decode_hex(
            payload["rank_manifest_digest"],
            PACKED_REQUEST_DIGEST_BYTES,
            "rank_manifest_digest",
        )
        allocation_digest = _decode_hex(
            payload["allocation_digest"],
            PACKED_REQUEST_DIGEST_BYTES,
            "allocation_digest",
        )
        binding_digest = _decode_hex(
            payload["binding_digest"],
            PACKED_REQUEST_DIGEST_BYTES,
            "binding_digest",
        )
        room_id = _require_integer(payload["room_id"], "room_id")
        tp_rank = _require_integer(payload["tp_rank"], "tp_rank")
        tp_size = _require_integer(payload["tp_size"], "tp_size")
        started_ns = _require_integer(payload["started_ns"], "started_ns")
        completed_ns = _require_integer(payload["completed_ns"], "completed_ns")
        role_value = _require_string(payload["role"], "role")
        field_value = _require_string(payload["field"], "field")
        sample_key = _require_string(payload["sample_key"], "sample_key")
        duration_ms = payload["duration_ms"]
        if type(duration_ms) is not float and type(duration_ms) is not int:
            raise TypeError("duration_ms must be a number")
        if not math.isfinite(float(duration_ms)) or float(duration_ms) < 0.0:
            raise ValueError("duration_ms must be finite and non-negative")

        binding = TerminalRequestBinding(
            request_key=PackedRequestKey(
                room_id=room_id,
                request_generation=request_generation,
            ),
            owner=TerminalProcessIdentity(
                process_generation=process_generation,
                role=TerminalOwnerRole(role_value),
                tp_rank=tp_rank,
                tp_size=tp_size,
            ),
            rank_manifest_digest=rank_manifest_digest,
            allocation_digest=allocation_digest,
        )
        if binding.digest != binding_digest:
            raise ValueError("terminal timing binding digest is inconsistent")
        sample = cls(
            binding=binding,
            field=TerminalOwnerTimingField(field_value),
            sample_key=sample_key,
            started_ns=started_ns,
            completed_ns=completed_ns,
        )
        if not math.isclose(
            sample.duration_ms,
            float(duration_ms),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("terminal timing duration differs from timestamps")
        return sample

    def to_log_record(self) -> str:
        """Render the parser-stable production log record.

        :returns: Prefix-delimited canonical JSON.
        """

        encoded = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return f"{TERMINAL_PROGRESS_TIMING_LOG_PREFIX}{encoded}"


@runtime_checkable
class TerminalProgressTimingSink(Protocol):
    """Non-gating consumer of request-correlated timing evidence."""

    def emit(self, sample: TerminalProgressTimingSample) -> None:
        """Consume one immutable process-local interval.

        :param sample: Exact interval and request identity.
        """


class TerminalProgressTimingLogger(TerminalProgressTimingSink):
    """Project immutable terminal timing samples into production logs."""

    _logger: logging.Logger

    def __init__(self, logger: logging.Logger) -> None:
        """Bind one process-local logger.

        :param logger: Logger captured by the owning serving composition.
        """

        if not isinstance(logger, logging.Logger):
            raise TypeError("logger must be logging.Logger")
        self._logger = logger

    def emit(self, sample: TerminalProgressTimingSample) -> None:
        """Write one parser-stable evidence line.

        :param sample: Exact interval and request identity.
        """

        if type(sample) is not TerminalProgressTimingSample:
            raise TypeError("sample must be TerminalProgressTimingSample")
        self._logger.info("%s", sample.to_log_record())


class TerminalProgressTimingEmitter:
    """Non-gating construction and projection boundary for timing evidence."""

    _sink: TerminalProgressTimingSink
    _failure_logger: logging.Logger

    def __init__(
        self,
        sink: TerminalProgressTimingSink,
        failure_logger: logging.Logger,
    ) -> None:
        """Bind the process evidence sink and its diagnostic logger.

        :param sink: Immutable timing sample consumer.
        :param failure_logger: Logger receiving full projection failures.
        """

        if not isinstance(sink, TerminalProgressTimingSink):
            raise TypeError("sink must satisfy TerminalProgressTimingSink")
        if not isinstance(failure_logger, logging.Logger):
            raise TypeError("failure_logger must be logging.Logger")
        self._sink = sink
        self._failure_logger = failure_logger

    def emit(
        self,
        *,
        binding: TerminalRequestBinding,
        field: TerminalOwnerTimingField,
        sample_key: str,
        started_ns: int,
        completed_ns: int,
    ) -> bool:
        """Project one interval without acquiring lifecycle authority.

        Evidence failures remain visible and make the smoke incomplete, but
        they cannot delay or reject an otherwise valid terminal transition.

        :param binding: Exact local request generation.
        :param field: Frozen interval field.
        :param sample_key: Stable rank, writer, or handle cardinality key.
        :param started_ns: Process-local monotonic start timestamp.
        :param completed_ns: Process-local monotonic completion timestamp.
        :returns: Whether the sink accepted the evidence sample.
        """

        try:
            sample = TerminalProgressTimingSample(
                binding=binding,
                field=field,
                sample_key=sample_key,
                started_ns=started_ns,
                completed_ns=completed_ns,
            )
            self._sink.emit(sample)
        except Exception:  # noqa: BLE001
            self._failure_logger.error(
                "Terminal timing evidence projection failed:\n%s",
                traceback.format_exc(),
            )
            return False
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalProgressRequestTimingSummary:
    """Request-level critical-path reduction over preserved raw samples.

    :ivar request_key: Exact request generation shared by every sample.
    :ivar critical_path_ms: Per-field maximum same-process interval.
    :ivar sample_counts: Number of distinct raw samples retained per field.
    """

    request_key: PackedRequestKey
    critical_path_ms: tuple[tuple[TerminalOwnerTimingField, float], ...]
    sample_counts: tuple[tuple[TerminalOwnerTimingField, int], ...]

    def __post_init__(self) -> None:
        """Validate one complete, ordered field reduction."""

        if type(self.request_key) is not PackedRequestKey:
            raise TypeError("request_key must be PackedRequestKey")
        fields = tuple(TerminalOwnerTimingField)
        if tuple(field for field, _ in self.critical_path_ms) != fields:
            raise ValueError("critical_path_ms must contain every timing field")
        if tuple(field for field, _ in self.sample_counts) != fields:
            raise ValueError("sample_counts must contain every timing field")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for _, value in self.critical_path_ms
        ):
            raise ValueError("critical-path values must be finite and non-negative")
        if any(type(count) is not int or count <= 0 for _, count in self.sample_counts):
            raise ValueError("every timing field must retain at least one sample")

    def to_payload(self) -> dict[str, object]:
        """Serialize the request-level critical-path summary.

        :returns: JSON-compatible field maxima and raw sample counts.
        """

        return {
            "room_id": self.request_key.room_id,
            "request_generation": self.request_key.request_generation.hex(),
            "critical_path_ms": {
                field.value: value for field, value in self.critical_path_ms
            },
            "sample_counts": {
                field.value: count for field, count in self.sample_counts
            },
        }


def parse_terminal_progress_timing_log_line(
    line: str,
) -> TerminalProgressTimingSample | None:
    """Parse one production log line when it carries terminal timing evidence.

    :param line: Complete service log line.
    :returns: Validated sample, or ``None`` when the marker is absent.
    """

    if type(line) is not str:
        raise TypeError("line must be a string")
    marker_index = line.find(TERMINAL_PROGRESS_TIMING_LOG_PREFIX)
    if marker_index < 0:
        return None
    encoded = line[marker_index + len(TERMINAL_PROGRESS_TIMING_LOG_PREFIX) :]
    decoder = json.JSONDecoder()
    payload, consumed = decoder.raw_decode(encoded)
    if len(encoded[consumed:].strip()) > 0:
        raise ValueError("terminal timing log line has trailing content")
    if type(payload) is not dict:
        raise ValueError("terminal timing log payload must be an object")
    return TerminalProgressTimingSample.from_payload(payload)


def summarize_terminal_progress_request(
    samples: Iterable[TerminalProgressTimingSample],
) -> TerminalProgressRequestTimingSummary:
    """Reduce one request while retaining every raw cardinality externally.

    Byte-identical duplicate log records coalesce. A conflicting duplicate or
    a missing mandatory field fails evidence construction.

    :param samples: Raw samples belonging to exactly one request generation.
    :returns: Per-field local critical-path maxima and cardinalities.
    """

    values = tuple(samples)
    if len(values) == 0:
        raise ValueError("request timing summary requires samples")
    if any(type(sample) is not TerminalProgressTimingSample for sample in values):
        raise TypeError("samples must contain TerminalProgressTimingSample values")
    request_key = values[0].binding.request_key
    if any(sample.binding.request_key != request_key for sample in values):
        raise ValueError("request timing samples span request generations")

    distinct: dict[
        tuple[bytes, TerminalOwnerTimingField, str], TerminalProgressTimingSample
    ] = {}
    for sample in values:
        existing = distinct.get(sample.cardinality_key)
        if existing is None:
            distinct[sample.cardinality_key] = sample
            continue
        if existing != sample:
            raise ValueError("terminal timing cardinality has conflicting samples")

    by_field: dict[TerminalOwnerTimingField, list[float]] = {
        field: [] for field in TerminalOwnerTimingField
    }
    for sample in distinct.values():
        by_field[sample.field].append(sample.duration_ms)
    missing = tuple(
        field.value for field, durations in by_field.items() if len(durations) == 0
    )
    if len(missing) > 0:
        raise ValueError(f"request timing fields are incomplete: {missing}")
    return TerminalProgressRequestTimingSummary(
        request_key=request_key,
        critical_path_ms=tuple(
            (field, max(by_field[field])) for field in TerminalOwnerTimingField
        ),
        sample_counts=tuple(
            (field, len(by_field[field])) for field in TerminalOwnerTimingField
        ),
    )


def _decode_hex(value: object, byte_count: int, label: str) -> bytes:
    """Decode one fixed-width lowercase hexadecimal field.

    :param value: Candidate encoded identity.
    :param byte_count: Required decoded width.
    :param label: Reader-facing field name.
    :returns: Validated bytes.
    """

    encoded = _require_string(value, label)
    if len(encoded) != byte_count * 2 or encoded.lower() != encoded:
        raise ValueError(f"{label} must be lowercase fixed-width hexadecimal")
    try:
        decoded = bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    if len(decoded) != byte_count:
        raise ValueError(f"{label} must decode to {byte_count} bytes")
    return decoded


def _require_integer(value: object, label: str) -> int:
    """Require one non-Boolean integer.

    :param value: Candidate value.
    :param label: Reader-facing field name.
    :returns: Validated integer.
    """

    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _require_string(value: object, label: str) -> str:
    """Require one non-empty string.

    :param value: Candidate value.
    :param label: Reader-facing field name.
    :returns: Validated string.
    """

    if type(value) is not str or len(value) == 0:
        raise ValueError(f"{label} must be a non-empty string")
    return value
