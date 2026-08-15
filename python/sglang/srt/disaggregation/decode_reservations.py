import base64
import copy
import dataclasses
import enum
import hashlib
import json
import logging
import secrets
import threading
import time
import traceback
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from typing import Protocol

import blake3

from sglang.srt.disaggregation.runtime_capabilities import (
    SUPPORTED_PACKED_SOURCE_TP_SIZES,
)

SCHEMA_VERSION = 1
GRANT_TOKEN_BYTES = 32
RESERVE_ATTEMPT_DIGEST_DOMAIN = b"sglang-pd-decoder-reserve-attempt-v1"
RESERVATION_DIGEST_DOMAIN = b"sglang-pd-decoder-reservation-v1"
GRANT_DIGEST_DOMAIN = b"sglang-pd-decoder-grant-v4"
RECEIPT_DIGEST_DOMAIN = b"sglang-pd-decoder-control-receipt-v1"
BOOTSTRAP_ROOM_DOMAIN = b"sglang-pd-decoder-bootstrap-room-v1"

_RESERVE_FIELDS = frozenset(
    {
        "schema_version",
        "prefill_process",
        "prefill_bootstrap_endpoint",
        "decoder_process",
        "logical_request_chain_id",
        "reservation_attempt_id",
        "reserve_attempt_digest",
        "source_tp_size",
        "prepared_ttl_ms",
        "inference_route",
        "request_shape",
        "base_request_body_json",
        "child_request_ids",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "grant_id",
        "reservation_attempt_id",
        "reserve_attempt_digest",
        "prefill_process",
        "prefill_bootstrap_endpoint",
        "decoder_process",
        "logical_request_chain_id",
        "source_tp_size",
        "inference_route",
        "request_shape",
        "prepared_ttl_ms",
        "prepared_expires_at_unix_ms",
        "child_request_ids",
        "decoder_slot_generations",
        "bootstrap_rooms",
        "reservation_digest",
        "grant_digest",
    }
)
_FAILURE_FIELDS = _BINDING_FIELDS | {"reason_code", "diagnostic"}
_UNBOUND_CANCELLATION_FIELDS = _BINDING_FIELDS - {"grant_digest"} | {
    "attempted_grant_digest"
}
_INFERENCE_ROUTES = frozenset({"/generate", "/v1/chat/completions", "/v1/completions"})
_REQUEST_SHAPES = frozenset({"scalar", "batch"})
_FAILURE_REASON_MAX_BYTES = 64
_FAILURE_DIAGNOSTIC_MAX_BYTES = 512
_FAILURE_REASON_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)

logger = logging.getLogger(__name__)


def derive_decode_reservation_bootstrap_rooms(
    grant_id: uuid.UUID,
    child_request_ids: tuple[uuid.UUID, ...],
) -> tuple[int, ...]:
    """Derive process-global staging rooms for one ordered grant cohort.

    :param grant_id: Tokenizer-issued process-global grant identity.
    :param child_request_ids: Exact ordered child request identities.
    :returns: Deterministic unsigned 64-bit staging rooms.
    """

    if type(grant_id) is not uuid.UUID or grant_id.int == 0:
        raise DecodeReservationValidationError("grant_id must be a non-nil UUID")
    if len(child_request_ids) == 0:
        raise DecodeReservationValidationError(
            "bootstrap room derivation requires at least one child"
        )
    if any(
        type(child_id) is not uuid.UUID or child_id.int == 0
        for child_id in child_request_ids
    ):
        raise DecodeReservationValidationError(
            "bootstrap room derivation requires non-nil UUID children"
        )

    rooms: list[int] = []
    for index, child_id in enumerate(child_request_ids):
        digest = hashlib.sha256()
        digest.update(BOOTSTRAP_ROOM_DOMAIN)
        digest.update(grant_id.bytes)
        digest.update(index.to_bytes(8, "little"))
        digest.update(child_id.bytes)
        room = int.from_bytes(digest.digest()[:8], "little") | (1 << 63)
        rooms.append(room)
    if len(set(rooms)) != len(rooms):
        raise DecodeReservationValidationError(
            "derived bootstrap rooms collide within one grant"
        )
    return tuple(rooms)


def _wall_clock_unix_ms() -> int:
    """Return current Unix time in milliseconds.

    :returns: Current wall-clock milliseconds.
    """

    return time.time_ns() // 1_000_000


def _monotonic_clock_ns() -> int:
    """Return current monotonic time in nanoseconds.

    :returns: Current monotonic nanoseconds.
    """

    return time.monotonic_ns()


class DecodeReservationError(RuntimeError):
    """Base decoder-reservation control error."""


class DecodeReservationValidationError(DecodeReservationError):
    """Invalid or contradictory decoder-reservation transcript."""


class DecodeReservationConflictError(DecodeReservationError):
    """A control operation conflicts with retained authoritative state."""


class DecodeReservationOperationInFlightError(DecodeReservationConflictError):
    """An exact operation is already executing outside the authority lock."""


class DecodeReservationExpiredError(DecodeReservationConflictError):
    """A prepared reservation passed its monotonic ownership deadline."""


class DecodeReservationNotFoundError(DecodeReservationError):
    """No decoder reservation owns the supplied identity."""


class DecodeReservationAuthenticationError(DecodeReservationError):
    """A control request does not own the required bearer capability."""


class DecodeReservationRefusalDisposition(enum.StrEnum):
    """Authoritative allocator admission refusal disposition."""

    RETRY_SAME_DECODER = "retry_same_decoder"
    RETRY_ANOTHER_DECODER = "retry_another_decoder"
    TERMINAL = "terminal"


class DecodeReservationAdmissionRefused(DecodeReservationError):
    """Take-once allocator admission refusal before ownership mutation."""

    reason_code: str
    diagnostic: str | None
    disposition: DecodeReservationRefusalDisposition

    def __init__(
        self,
        reason_code: str,
        disposition: DecodeReservationRefusalDisposition,
        diagnostic: str | None = None,
    ) -> None:
        """Initialize one authoritative admission refusal.

        :param reason_code: Stable machine-readable reason.
        :param disposition: Gateway retry disposition.
        :param diagnostic: Optional bounded diagnostic.
        """

        validated_reason, validated_diagnostic = _validate_failure_context_values(
            reason_code,
            diagnostic,
        )
        if type(disposition) is not DecodeReservationRefusalDisposition:
            raise TypeError("disposition must be DecodeReservationRefusalDisposition")
        self.reason_code = validated_reason
        self.diagnostic = validated_diagnostic
        self.disposition = disposition
        super().__init__(validated_reason)


class DecodeReservationState(enum.StrEnum):
    """Engine-owned decoder reservation lifecycle."""

    PREPARED_UNBOUND = "prepared_unbound"
    PREPARED_BOUND = "prepared_bound"
    PROMOTED = "promoted"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ABORTED = "aborted"
    QUARANTINED = "quarantined"


class DecodeInferenceAttachmentState(enum.StrEnum):
    """Tokenizer-local exact inference attachment state."""

    PREPARED_UNBOUND = "prepared_unbound"
    PREPARED_BOUND = "prepared_bound"
    PROMOTED = "promoted"
    ATTACHED = "attached"
    TERMINAL = "terminal"


class DecodeReservationOperation(enum.StrEnum):
    """Closed control operation set."""

    BIND = "bind"
    PROMOTE = "promote"
    CANCEL = "cancel"
    COMPLETE = "complete"
    ABORT = "abort"
    QUARANTINE = "quarantine"


class DecodeReservationTerminalAction(Protocol):
    """Scheduler-owned cleanup action for one exact reservation."""

    def __call__(
        self,
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Apply an exact terminal action.

        :param owner: Exact opaque allocation owner.
        :param reason_code: Stable machine-readable failure reason.
        :param diagnostic: Bounded diagnostic text.
        :returns: Authoritative terminal state.
        """

        ...


class DecodeReservationAllocator(Protocol):
    """Native scheduler allocation and transfer owner."""

    def prepare(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: "DecodeReservationAttempt",
        tokenized_requests: tuple[object, ...],
    ) -> tuple[tuple["DecodeReservationAllocation", ...], object]:
        """Prepare and retain one exact request cohort.

        :param grant_id: Tokenizer-issued grant identity.
        :param attempt: Exact reserve transcript.
        :param tokenized_requests: Exact canonical tokenizer outputs.
        :returns: Ordered allocation receipts and opaque retained cohort.
        """

        ...

    def promote(self, owner: object) -> None:
        """Publish one prepared cohort's transport ownership.

        :param owner: Exact retained cohort.
        """

        ...

    def attach(self, owner: object) -> None:
        """Attach exact inference ownership to one promoted cohort.

        :param owner: Exact retained cohort.
        """

        ...

    def cancel(self, owner: object) -> DecodeReservationState:
        """Release a prepared cohort exactly.

        :param owner: Exact retained cohort.
        :returns: Authoritative terminal state.
        """

        ...

    def complete(self, owner: object) -> DecodeReservationState:
        """Reconcile normal terminal completion.

        :param owner: Exact retained cohort.
        :returns: Authoritative terminal state.
        """

        ...

    def abort(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Reconcile a promoted abort.

        :param owner: Exact retained cohort.
        :param reason_code: Stable failure reason.
        :param diagnostic: Bounded diagnostic.
        :returns: Aborted or quarantined state.
        """

        ...

    def quarantine(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Retain ambiguous ownership process-lifetime.

        :param owner: Exact retained cohort.
        :param reason_code: Stable failure reason.
        :param diagnostic: Bounded diagnostic.
        :returns: Quarantined state.
        """

        ...


@dataclasses.dataclass(frozen=True)
class DecodeReservationProcess:
    """One process-generation identity.

    :ivar url: Canonical process HTTP origin.
    :ivar instance_id: Exact launch-generation UUID.
    """

    url: str
    instance_id: uuid.UUID

    @classmethod
    def from_value(
        cls,
        value: object,
        name: str,
    ) -> "DecodeReservationProcess":
        """Parse one strict process identity.

        :param value: Candidate JSON value.
        :param name: Field name used in errors.
        :returns: Validated process identity.
        """

        fields = _require_mapping(value, name)
        _require_exact_fields(fields, {"url", "instance_id"}, name)
        url = _require_nonempty_string(fields["url"], f"{name}.url")
        instance_id = _require_uuid(fields["instance_id"], f"{name}.instance_id")
        return cls(url=url, instance_id=instance_id)

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire representation.

        :returns: JSON-native process identity.
        """

        return {"url": self.url, "instance_id": str(self.instance_id)}


@dataclasses.dataclass(frozen=True)
class DecodeReservationBootstrapEndpoint:
    """Selected prefill bootstrap endpoint.

    :ivar host: Explicit nonempty host.
    :ivar port: TCP port.
    """

    host: str
    port: int

    @classmethod
    def from_value(
        cls,
        value: object,
    ) -> "DecodeReservationBootstrapEndpoint":
        """Parse one strict bootstrap endpoint.

        :param value: Candidate JSON value.
        :returns: Validated endpoint.
        """

        fields = _require_mapping(value, "prefill_bootstrap_endpoint")
        _require_exact_fields(
            fields,
            {"host", "port"},
            "prefill_bootstrap_endpoint",
        )
        host = _require_nonempty_string(
            fields["host"], "prefill_bootstrap_endpoint.host"
        )
        port = _require_integer(fields["port"], "prefill_bootstrap_endpoint.port")
        if not 1 <= port <= 65535:
            raise DecodeReservationValidationError(
                "prefill_bootstrap_endpoint.port must be between 1 and 65535"
            )
        return cls(host=host, port=port)

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire representation.

        :returns: JSON-native bootstrap endpoint.
        """

        return {"host": self.host, "port": self.port}


@dataclasses.dataclass(frozen=True)
class DecodeReservationAttempt:
    """Exact idempotent reserve-attempt transcript.

    :ivar prefill_process: Selected prefill process generation.
    :ivar prefill_bootstrap_endpoint: Selected prefill bootstrap endpoint.
    :ivar decoder_process: Selected decoder process generation.
    :ivar logical_request_chain_id: Stable logical retry-chain identity.
    :ivar reservation_attempt_id: Exact allocator-attempt identity.
    :ivar reserve_attempt_digest: Gateway-computed BLAKE3 transcript digest.
    :ivar source_tp_size: Supported packed prefill TP width.
    :ivar prepared_ttl_ms: Engine-clock-owned prepared TTL.
    :ivar inference_route: Exact inference HTTP route.
    :ivar request_shape: Scalar or batch request shape.
    :ivar base_request_body: Exact UTF-8 RID-enriched provisional request body.
    :ivar child_request_ids: Ordered child request identities.
    """

    prefill_process: DecodeReservationProcess
    prefill_bootstrap_endpoint: DecodeReservationBootstrapEndpoint
    decoder_process: DecodeReservationProcess
    logical_request_chain_id: uuid.UUID
    reservation_attempt_id: uuid.UUID
    reserve_attempt_digest: bytes
    source_tp_size: int
    prepared_ttl_ms: int
    inference_route: str
    request_shape: str
    base_request_body: bytes
    child_request_ids: tuple[uuid.UUID, ...]

    def __repr__(self) -> str:
        """Return a prompt-redacted representation.

        :returns: Safe diagnostic representation.
        """

        return (
            "DecodeReservationAttempt("
            f"reservation_attempt_id={self.reservation_attempt_id!s}, "
            f"logical_request_chain_id={self.logical_request_chain_id!s}, "
            f"inference_route={self.inference_route!r}, "
            f"request_shape={self.request_shape!r}, "
            f"source_tp_size={self.source_tp_size}, "
            f"child_count={len(self.child_request_ids)}, "
            f"base_request_body_bytes={len(self.base_request_body)})"
        )

    @classmethod
    def from_value(cls, value: object) -> "DecodeReservationAttempt":
        """Parse and authenticate one reserve request.

        :param value: Candidate decoded JSON object.
        :returns: Exact validated reserve attempt.
        """

        fields = _require_mapping(value, "reserve request")
        _require_exact_fields(fields, _RESERVE_FIELDS, "reserve request")
        _require_schema_version(fields["schema_version"])
        source_tp_size = _require_integer(fields["source_tp_size"], "source_tp_size")
        if source_tp_size not in SUPPORTED_PACKED_SOURCE_TP_SIZES:
            raise DecodeReservationValidationError(
                f"source_tp_size must be one of {SUPPORTED_PACKED_SOURCE_TP_SIZES}"
            )
        prepared_ttl_ms = _require_integer(fields["prepared_ttl_ms"], "prepared_ttl_ms")
        if prepared_ttl_ms <= 0:
            raise DecodeReservationValidationError("prepared_ttl_ms must be positive")
        inference_route = _require_nonempty_string(
            fields["inference_route"], "inference_route"
        )
        if inference_route not in _INFERENCE_ROUTES:
            raise DecodeReservationValidationError("unknown inference_route")
        request_shape = _require_nonempty_string(
            fields["request_shape"], "request_shape"
        )
        if request_shape not in _REQUEST_SHAPES:
            raise DecodeReservationValidationError("unknown request_shape")
        base_request_body_json = _require_nonempty_string(
            fields["base_request_body_json"], "base_request_body_json"
        )
        base_request_body = base_request_body_json.encode("utf-8")
        try:
            base_request = json.loads(base_request_body)
        except json.JSONDecodeError as error:
            raise DecodeReservationValidationError(
                "base_request_body_json must contain valid JSON"
            ) from error
        if type(base_request) is not dict:
            raise DecodeReservationValidationError(
                "base_request_body_json must contain a JSON object"
            )
        child_values = fields["child_request_ids"]
        if type(child_values) is not list or len(child_values) == 0:
            raise DecodeReservationValidationError(
                "child_request_ids must be a nonempty array"
            )
        child_request_ids = tuple(
            _require_uuid(value, f"child_request_ids[{index}]")
            for index, value in enumerate(child_values)
        )
        if len(set(child_request_ids)) != len(child_request_ids):
            raise DecodeReservationValidationError("child_request_ids must be unique")
        if request_shape == "scalar" and len(child_request_ids) != 1:
            raise DecodeReservationValidationError(
                "scalar reservations require exactly one child"
            )
        attempt = cls(
            prefill_process=DecodeReservationProcess.from_value(
                fields["prefill_process"], "prefill_process"
            ),
            prefill_bootstrap_endpoint=(
                DecodeReservationBootstrapEndpoint.from_value(
                    fields["prefill_bootstrap_endpoint"]
                )
            ),
            decoder_process=DecodeReservationProcess.from_value(
                fields["decoder_process"], "decoder_process"
            ),
            logical_request_chain_id=_require_uuid(
                fields["logical_request_chain_id"],
                "logical_request_chain_id",
            ),
            reservation_attempt_id=_require_uuid(
                fields["reservation_attempt_id"], "reservation_attempt_id"
            ),
            reserve_attempt_digest=_require_digest(
                fields["reserve_attempt_digest"], "reserve_attempt_digest"
            ),
            source_tp_size=source_tp_size,
            prepared_ttl_ms=prepared_ttl_ms,
            inference_route=inference_route,
            request_shape=request_shape,
            base_request_body=base_request_body,
            child_request_ids=child_request_ids,
        )
        expected_digest = attempt.compute_digest()
        if not secrets.compare_digest(attempt.reserve_attempt_digest, expected_digest):
            raise DecodeReservationValidationError(
                "reserve_attempt_digest does not authenticate the reserve transcript"
            )
        attempt._validate_base_request_ids(base_request)
        return attempt

    def compute_digest(self) -> bytes:
        """Compute the Rust-compatible reserve-attempt digest.

        :returns: Exact 32-byte BLAKE3 digest.
        """

        hasher = blake3.blake3()
        hasher.update(RESERVE_ATTEMPT_DIGEST_DOMAIN)
        hasher.update(self.reservation_attempt_id.bytes)
        _hash_bytes(hasher, self.inference_route.encode())
        _hash_bytes(hasher, self.request_shape.encode())
        hasher.update(self.prepared_ttl_ms.to_bytes(8, "little"))
        _hash_bytes(hasher, self.base_request_body)
        _hash_process(hasher, self.prefill_process)
        _hash_bytes(hasher, self.prefill_bootstrap_endpoint.host.encode())
        hasher.update(self.prefill_bootstrap_endpoint.port.to_bytes(8, "little"))
        hasher.update(self.logical_request_chain_id.bytes)
        hasher.update(self.source_tp_size.to_bytes(8, "little"))
        _hash_process(hasher, self.decoder_process)
        hasher.update(len(self.child_request_ids).to_bytes(8, "little"))
        for index, child_request_id in enumerate(self.child_request_ids):
            hasher.update(index.to_bytes(8, "little"))
            hasher.update(child_request_id.bytes)
        return hasher.digest()

    def _validate_base_request_ids(self, body: dict[str, object]) -> None:
        """Require the exact ordered child IDs in the provisional body.

        :param body: Parsed provisional request body.
        """

        rid = body.get("rid")
        expected: object
        if self.request_shape == "scalar":
            expected = str(self.child_request_ids[0])
        else:
            expected = [str(child_id) for child_id in self.child_request_ids]
        if rid != expected:
            raise DecodeReservationValidationError(
                "base request rid differs from child_request_ids"
            )


@dataclasses.dataclass(frozen=True)
class DecodeReservationAllocation:
    """One process-global engine-issued child allocation receipt.

    :ivar child_request_id: Ordered gateway child identity.
    :ivar decoder_slot_generation: Opaque process-global allocation generation.
    :ivar bootstrap_room: Process-global decoder transfer room.
    :ivar request_slot: Ordered child ordinal within the grant.
    :ivar request_generation: Stable process-global allocation generation value.
    :ivar writer_manifest_digest: Complete cross-rank writer-topology digest.
    :ivar allocation_digest: Process-global logical allocation digest.
    :ivar reserved_kv_tokens: Conservative reserved KV tokens.
    :ivar remaining_decode_tokens: Remaining decode-work estimate.
    """

    child_request_id: uuid.UUID
    decoder_slot_generation: uuid.UUID
    bootstrap_room: int
    request_slot: int
    request_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    reserved_kv_tokens: int
    remaining_decode_tokens: int

    def __post_init__(self) -> None:
        """Validate one allocator-issued receipt."""

        if self.child_request_id.int == 0:
            raise DecodeReservationValidationError("child_request_id cannot be nil")
        if self.decoder_slot_generation.int == 0:
            raise DecodeReservationValidationError(
                "decoder_slot_generation cannot be nil"
            )
        for name, value in (
            ("bootstrap_room", self.bootstrap_room),
            ("request_slot", self.request_slot),
            ("request_generation", self.request_generation),
            ("reserved_kv_tokens", self.reserved_kv_tokens),
            ("remaining_decode_tokens", self.remaining_decode_tokens),
        ):
            if type(value) is not int or value < 0 or value >= 1 << 64:
                raise DecodeReservationValidationError(
                    f"{name} must be an unsigned 64-bit integer"
                )
        for name, digest in (
            ("writer_manifest_digest", self.writer_manifest_digest),
            ("allocation_digest", self.allocation_digest),
        ):
            if type(digest) is not bytes or len(digest) != 32:
                raise DecodeReservationValidationError(
                    f"{name} must contain exactly 32 bytes"
                )

    def to_dict(self) -> dict[str, object]:
        """Return the strict wire representation.

        :returns: JSON-native allocation receipt.
        """

        return {
            "child_request_id": str(self.child_request_id),
            "decoder_slot_generation": str(self.decoder_slot_generation),
            "bootstrap_room": self.bootstrap_room,
            "request_slot": self.request_slot,
            "request_generation": self.request_generation,
            "writer_manifest_digest": self.writer_manifest_digest.hex(),
            "allocation_digest": self.allocation_digest.hex(),
            "reserved_kv_tokens": self.reserved_kv_tokens,
            "remaining_decode_tokens": self.remaining_decode_tokens,
        }


@dataclasses.dataclass(frozen=True, repr=False)
class DecodeReservationGrantToken:
    """Opaque grant-specific bearer token."""

    _value: str

    def __post_init__(self) -> None:
        """Validate one canonical 256-bit bearer token."""

        if type(self._value) is not str:
            raise TypeError("grant token must be a string")
        try:
            decoded = base64.urlsafe_b64decode(self._value + "==")
        except (ValueError, base64.binascii.Error) as error:
            raise DecodeReservationValidationError(
                "grant token must be unpadded base64url"
            ) from error
        if (
            len(decoded) != GRANT_TOKEN_BYTES
            or base64.urlsafe_b64encode(decoded).decode().rstrip("=") != self._value
        ):
            raise DecodeReservationValidationError(
                "grant token must be canonical unpadded 256-bit base64url"
            )

    @classmethod
    def generate(cls) -> "DecodeReservationGrantToken":
        """Generate one unpredictable grant token.

        :returns: New opaque grant token.
        """

        value = base64.urlsafe_b64encode(
            secrets.token_bytes(GRANT_TOKEN_BYTES)
        ).decode()
        return cls(value.rstrip("="))

    def expose_for_response(self) -> str:
        """Expose the token only for its one reserve response.

        :returns: Wire token value.
        """

        return self._value

    def authenticates(self, authorization_header: str | None) -> bool:
        """Check one exact bearer header in constant time.

        :param authorization_header: Candidate Authorization header.
        :returns: Whether the header owns this grant.
        """

        token = _extract_bearer_token(authorization_header)
        if token is None:
            return False
        return secrets.compare_digest(token, self._value)

    def __repr__(self) -> str:
        """Return a permanently redacted representation.

        :returns: Redacted token representation.
        """

        return "DecodeReservationGrantToken([REDACTED])"


@dataclasses.dataclass(frozen=True)
class DecodeInferenceAttachmentSnapshot:
    """Immutable tokenizer-local one-shot attachment snapshot.

    :ivar grant_id: Exact scheduler grant identity.
    :ivar reservation_attempt_id: Exact idempotent reserve-attempt identity.
    :ivar reserve_attempt_digest: Exact reserve transcript digest.
    :ivar inference_route: Exact normal inference HTTP route.
    :ivar child_request_ids: Ordered already-initialized request IDs.
    :ivar opaque_request: Exact normalized tokenizer request until terminal.
    :ivar state: Current tokenizer-local attachment state.
    :ivar has_bound_request: Whether exact inference bytes are retained.
    :ivar has_reserve_response: Whether the reserve response was published.
    """

    grant_id: uuid.UUID
    reservation_attempt_id: uuid.UUID
    reserve_attempt_digest: bytes
    inference_route: str
    child_request_ids: tuple[uuid.UUID, ...]
    opaque_request: object | None
    state: DecodeInferenceAttachmentState
    has_bound_request: bool
    has_reserve_response: bool

    def __repr__(self) -> str:
        """Return a prompt- and secret-redacted representation.

        :returns: Safe diagnostic representation.
        """

        return (
            "DecodeInferenceAttachmentSnapshot("
            f"grant_id={self.grant_id!s}, "
            f"reservation_attempt_id={self.reservation_attempt_id!s}, "
            f"inference_route={self.inference_route!r}, "
            f"child_count={len(self.child_request_ids)}, "
            f"state={self.state.value!r}, "
            f"has_bound_request={self.has_bound_request}, "
            f"has_reserve_response={self.has_reserve_response})"
        )


@dataclasses.dataclass
class _DecodeInferenceAttachmentRecord:
    """Private tokenizer-local one-shot attachment state."""

    grant_id: uuid.UUID
    reservation_attempt_id: uuid.UUID
    reserve_attempt_digest: bytes
    inference_route: str
    child_request_ids: tuple[uuid.UUID, ...]
    opaque_request: object | None
    grant_token: DecodeReservationGrantToken
    state: DecodeInferenceAttachmentState = (
        DecodeInferenceAttachmentState.PREPARED_UNBOUND
    )
    bound_request_body: bytes | None = None
    reserve_response: dict[str, object] | None = None

    def __repr__(self) -> str:
        """Return a prompt-redacted representation.

        :returns: Safe diagnostic representation.
        """

        return (
            "DecodeInferenceAttachment("
            f"grant_id={self.grant_id!s}, "
            f"reservation_attempt_id={self.reservation_attempt_id!s}, "
            f"inference_route={self.inference_route!r}, "
            f"child_count={len(self.child_request_ids)}, "
            f"state={self.state.value!r}, "
            f"has_bound_request={self.bound_request_body is not None})"
        )


class DecodeInferenceAttachmentRegistry:
    """Tokenizer-local one-shot route and exact-body attachment registry."""

    _attempt_owners: dict[uuid.UUID, uuid.UUID]
    _child_owners: dict[uuid.UUID, uuid.UUID]
    _lock: threading.Lock
    _promoted_bodies: dict[tuple[str, bytes], uuid.UUID]
    _records: dict[uuid.UUID, _DecodeInferenceAttachmentRecord]

    def __init__(self) -> None:
        """Initialize an empty attachment registry."""

        self._attempt_owners = {}
        self._child_owners = {}
        self._lock = threading.Lock()
        self._promoted_bodies = {}
        self._records = {}

    def register(
        self,
        *,
        grant_id: uuid.UUID,
        reservation_attempt_id: uuid.UUID,
        reserve_attempt_digest: bytes,
        inference_route: str,
        child_request_ids: tuple[uuid.UUID, ...],
        opaque_request: object,
        grant_token: DecodeReservationGrantToken | None = None,
    ) -> DecodeInferenceAttachmentSnapshot:
        """Register one reserve-created tokenizer request.

        :param grant_id: Exact scheduler grant identity.
        :param reservation_attempt_id: Exact idempotent reserve-attempt identity.
        :param reserve_attempt_digest: Exact reserve transcript digest.
        :param inference_route: Exact normal inference HTTP route.
        :param child_request_ids: Ordered already-initialized request IDs.
        :param opaque_request: Exact normalized tokenizer request.
        :param grant_token: Existing token for idempotent reconstruction.
        :returns: New local attachment capability.
        """

        if grant_id.int == 0:
            raise DecodeReservationValidationError("grant_id cannot be nil")
        if reservation_attempt_id.int == 0:
            raise DecodeReservationValidationError(
                "reservation_attempt_id cannot be nil"
            )
        if (
            type(reserve_attempt_digest) is not bytes
            or len(reserve_attempt_digest) != 32
        ):
            raise DecodeReservationValidationError(
                "reserve_attempt_digest must contain exactly 32 bytes"
            )
        if inference_route not in _INFERENCE_ROUTES:
            raise DecodeReservationValidationError("unknown inference_route")
        owned_ids = tuple(child_request_ids)
        if (
            len(owned_ids) == 0
            or any(child_request_id.int == 0 for child_request_id in owned_ids)
            or len(set(owned_ids)) != len(owned_ids)
        ):
            raise DecodeReservationValidationError(
                "attachment child IDs must be non-nil and unique"
            )
        if opaque_request is None:
            raise DecodeReservationValidationError(
                "attachment requires the exact tokenizer request"
            )
        record = _DecodeInferenceAttachmentRecord(
            grant_id=grant_id,
            reservation_attempt_id=reservation_attempt_id,
            reserve_attempt_digest=reserve_attempt_digest,
            inference_route=inference_route,
            child_request_ids=owned_ids,
            opaque_request=opaque_request,
            grant_token=(
                DecodeReservationGrantToken.generate()
                if grant_token is None
                else grant_token
            ),
        )
        with self._lock:
            if grant_id in self._records:
                raise DecodeReservationConflictError(
                    "grant attachment is already registered"
                )
            existing_attempt = self._find_attempt_locked(reservation_attempt_id)
            if existing_attempt is not None:
                raise DecodeReservationConflictError(
                    "reserve attempt attachment is already registered"
                )
            if any(
                child_request_id in self._child_owners for child_request_id in owned_ids
            ):
                raise DecodeReservationConflictError(
                    "attachment child request is already owned"
                )
            self._records[grant_id] = record
            self._attempt_owners[reservation_attempt_id] = grant_id
            for child_request_id in owned_ids:
                self._child_owners[child_request_id] = grant_id
            return self._snapshot_locked(record)

    def find_reserve_attempt(
        self,
        reservation_attempt_id: uuid.UUID,
        reserve_attempt_digest: bytes,
    ) -> DecodeInferenceAttachmentSnapshot | None:
        """Resolve an idempotent local reserve attempt.

        :param reservation_attempt_id: Exact reserve-attempt identity.
        :param reserve_attempt_digest: Exact reserve transcript digest.
        :returns: Retained local attempt, otherwise ``None``.
        """

        with self._lock:
            record = self._find_attempt_locked(reservation_attempt_id)
            if record is None:
                return None
            if not secrets.compare_digest(
                record.reserve_attempt_digest,
                reserve_attempt_digest,
            ):
                raise DecodeReservationConflictError(
                    "reserve attempt ID was reused with another digest"
                )
            return self._snapshot_locked(record)

    def publish_reserve_response(
        self,
        grant_id: uuid.UUID,
        response: Mapping[str, object],
    ) -> dict[str, object]:
        """Retain and return one idempotent reserve response.

        :param grant_id: Exact scheduler grant identity.
        :param response: Scheduler response without bearer secrets.
        :returns: Response including the grant token.
        """

        owned_response = copy.deepcopy(dict(response))
        if "grant_token" in owned_response:
            raise DecodeReservationValidationError(
                "scheduler reserve response cannot contain grant_token"
            )
        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.reserve_response is not None:
                if record.reserve_response != owned_response:
                    raise DecodeReservationConflictError(
                        "reserve response cannot be replaced"
                    )
            else:
                record.reserve_response = owned_response
            result = copy.deepcopy(record.reserve_response)
            result["grant_token"] = record.grant_token.expose_for_response()
            return result

    def retained_reserve_response(
        self,
        grant_id: uuid.UUID,
    ) -> dict[str, object] | None:
        """Return one previously published response with its token.

        :param grant_id: Exact grant identity.
        :returns: Idempotent response, otherwise ``None``.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.state is DecodeInferenceAttachmentState.TERMINAL:
                raise DecodeReservationConflictError(
                    "terminal reserve attempt cannot be replayed"
                )
            if record.reserve_response is None:
                return None
            result = copy.deepcopy(record.reserve_response)
            result["grant_token"] = record.grant_token.expose_for_response()
            return result

    def authenticate(
        self,
        grant_id: uuid.UUID,
        authorization_header: str | None,
    ) -> DecodeInferenceAttachmentSnapshot:
        """Authenticate one grant-specific HTTP control operation.

        :param grant_id: Candidate grant identity.
        :param authorization_header: Candidate grant bearer.
        :returns: Exact local attachment capability.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            if not record.grant_token.authenticates(authorization_header):
                raise DecodeReservationAuthenticationError(
                    "invalid decoder reservation bearer"
                )
            return self._snapshot_locked(record)

    def bind(self, grant_id: uuid.UUID, request_body: bytes) -> None:
        """Pin exact final inference bytes locally.

        :param grant_id: Exact prepared grant.
        :param request_body: Exact final request body.
        """

        owned_body = bytes(request_body)
        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.state is DecodeInferenceAttachmentState.PREPARED_BOUND:
                if record.bound_request_body != owned_body:
                    raise DecodeReservationConflictError(
                        "attachment request bytes cannot be replaced"
                    )
                return
            if record.state is not DecodeInferenceAttachmentState.PREPARED_UNBOUND:
                raise DecodeReservationConflictError(
                    f"attachment bind is invalid in state {record.state.value}"
                )
            record.bound_request_body = owned_body
            record.state = DecodeInferenceAttachmentState.PREPARED_BOUND

    def promote(self, grant_id: uuid.UUID) -> None:
        """Make one bound request attachable.

        :param grant_id: Exact bound grant.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.state is DecodeInferenceAttachmentState.PROMOTED:
                return
            if record.state is not DecodeInferenceAttachmentState.PREPARED_BOUND:
                raise DecodeReservationConflictError(
                    f"attachment promote is invalid in state {record.state.value}"
                )
            if record.bound_request_body is None:
                raise RuntimeError("bound attachment lost its request bytes")
            body_key = self._body_key(
                record.inference_route,
                record.bound_request_body,
            )
            existing_grant_id = self._promoted_bodies.get(body_key)
            if existing_grant_id is not None and existing_grant_id != grant_id:
                raise DecodeReservationConflictError(
                    "another promoted grant owns the same inference bytes"
                )
            self._promoted_bodies[body_key] = grant_id
            record.state = DecodeInferenceAttachmentState.PROMOTED

    def consume(
        self,
        inference_route: str,
        request_body: bytes,
    ) -> DecodeInferenceAttachmentSnapshot | None:
        """Consume one exact promoted route/body capability.

        :param inference_route: Actual normal inference route.
        :param request_body: Actual raw HTTP body bytes.
        :returns: Exact prepared request, or ``None`` when no grant matches.
        """

        owned_body = bytes(request_body)
        body_key = self._body_key(inference_route, owned_body)
        with self._lock:
            grant_id = self._promoted_bodies.get(body_key)
            if grant_id is None:
                return None
            record = self._require_record_locked(grant_id)
            if (
                record.state is not DecodeInferenceAttachmentState.PROMOTED
                or record.inference_route != inference_route
                or record.bound_request_body != owned_body
            ):
                raise DecodeReservationConflictError(
                    "promoted inference attachment index is inconsistent"
                )
            del self._promoted_bodies[body_key]
            record.state = DecodeInferenceAttachmentState.ATTACHED
            return self._snapshot_locked(record)

    def terminalize(
        self,
        grant_id: uuid.UUID,
    ) -> DecodeInferenceAttachmentSnapshot:
        """Mark a local capability terminal without releasing scheduler state.

        :param grant_id: Exact scheduler grant identity.
        :returns: Attachment snapshot immediately before terminalization.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            snapshot = self._snapshot_locked(record)
            if (
                record.state is DecodeInferenceAttachmentState.PROMOTED
                and record.bound_request_body is not None
            ):
                body_key = self._body_key(
                    record.inference_route,
                    record.bound_request_body,
                )
                self._promoted_bodies.pop(body_key, None)
            record.bound_request_body = None
            record.opaque_request = None
            record.state = DecodeInferenceAttachmentState.TERMINAL
            return snapshot

    def _require_record_locked(
        self,
        grant_id: uuid.UUID,
    ) -> _DecodeInferenceAttachmentRecord:
        record = self._records.get(grant_id)
        if record is None:
            raise DecodeReservationNotFoundError(
                "decoder inference attachment not found"
            )
        return record

    def _find_attempt_locked(
        self,
        reservation_attempt_id: uuid.UUID,
    ) -> _DecodeInferenceAttachmentRecord | None:
        grant_id = self._attempt_owners.get(reservation_attempt_id)
        if grant_id is None:
            return None
        return self._require_record_locked(grant_id)

    @staticmethod
    def _snapshot_locked(
        record: _DecodeInferenceAttachmentRecord,
    ) -> DecodeInferenceAttachmentSnapshot:
        return DecodeInferenceAttachmentSnapshot(
            grant_id=record.grant_id,
            reservation_attempt_id=record.reservation_attempt_id,
            reserve_attempt_digest=record.reserve_attempt_digest,
            inference_route=record.inference_route,
            child_request_ids=record.child_request_ids,
            opaque_request=record.opaque_request,
            state=record.state,
            has_bound_request=record.bound_request_body is not None,
            has_reserve_response=record.reserve_response is not None,
        )

    @staticmethod
    def _body_key(inference_route: str, request_body: bytes) -> tuple[str, bytes]:
        return inference_route, hashlib.sha256(request_body).digest()


@dataclasses.dataclass(frozen=True)
class DecodeReservationSnapshot:
    """Immutable redacted scheduler reservation snapshot.

    :ivar grant_id: Exact grant identity.
    :ivar reservation_attempt_id: Exact reserve-attempt identity.
    :ivar reserve_attempt_digest: Exact reserve transcript digest.
    :ivar prefill_process: Selected prefill process generation.
    :ivar prefill_bootstrap_endpoint: Selected bootstrap endpoint.
    :ivar decoder_process: Selected decoder process generation.
    :ivar logical_request_chain_id: Stable logical retry-chain identity.
    :ivar source_tp_size: Supported packed source width.
    :ivar prepared_ttl_ms: Prepared lease TTL.
    :ivar prepared_expires_at_unix_ms: Engine-clock expiry.
    :ivar inference_route: Exact normal inference route.
    :ivar request_shape: Scalar or batch shape.
    :ivar child_request_ids: Ordered child identities.
    :ivar allocations: Ordered immutable allocation receipts.
    :ivar reservation_digest: Exact reservation transcript digest.
    :ivar grant_digest: Bound request digest, when bound.
    :ivar state: Current authoritative state.
    :ivar inference_attached: Whether the exact promoted inference request was attached.
    """

    grant_id: uuid.UUID
    reservation_attempt_id: uuid.UUID
    reserve_attempt_digest: bytes
    prefill_process: DecodeReservationProcess
    prefill_bootstrap_endpoint: DecodeReservationBootstrapEndpoint
    decoder_process: DecodeReservationProcess
    logical_request_chain_id: uuid.UUID
    source_tp_size: int
    prepared_ttl_ms: int
    prepared_expires_at_unix_ms: int
    inference_route: str
    request_shape: str
    child_request_ids: tuple[uuid.UUID, ...]
    allocations: tuple[DecodeReservationAllocation, ...]
    reservation_digest: bytes
    grant_digest: bytes | None
    state: DecodeReservationState
    inference_attached: bool

    def reserve_response(self) -> dict[str, object]:
        """Build the Rust-compatible PREPARED response.

        :returns: Strict reserve response without bearer secrets.
        """

        if self.state not in (
            DecodeReservationState.PREPARED_UNBOUND,
            DecodeReservationState.PREPARED_BOUND,
            DecodeReservationState.PROMOTED,
        ):
            raise DecodeReservationConflictError(
                "terminal reservation cannot produce a prepared response"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "grant_id": str(self.grant_id),
            "prefill_process": self.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (self.prefill_bootstrap_endpoint.to_dict()),
            "decoder_process": self.decoder_process.to_dict(),
            "logical_request_chain_id": str(self.logical_request_chain_id),
            "reservation_attempt_id": str(self.reservation_attempt_id),
            "reserve_attempt_digest": self.reserve_attempt_digest.hex(),
            "source_tp_size": self.source_tp_size,
            "prepared_ttl_ms": self.prepared_ttl_ms,
            "inference_route": self.inference_route,
            "request_shape": self.request_shape,
            "reservation_digest": self.reservation_digest.hex(),
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "prepared_expires_at_unix_ms": self.prepared_expires_at_unix_ms,
        }


@dataclasses.dataclass(frozen=True)
class DecodeReservationExpirySweep:
    """Result of one scheduler-owned prepared-reservation expiry sweep.

    :ivar scanned_count: Number of authority records inspected.
    :ivar cancelled_grant_ids: Unbound grants released exactly.
    :ivar quarantined_grant_ids: Ambiguous promotions moved to safe quiescence.
    :ivar deferred_grant_ids: Expired grants with an executing callback.
    :ivar failed_grant_ids: Expired grants whose allocator action failed.
    """

    scanned_count: int
    cancelled_grant_ids: tuple[uuid.UUID, ...]
    quarantined_grant_ids: tuple[uuid.UUID, ...]
    deferred_grant_ids: tuple[uuid.UUID, ...]
    failed_grant_ids: tuple[uuid.UUID, ...]


@dataclasses.dataclass
class _DecodeReservationRecord:
    """Private scheduler-owned reservation state."""

    attempt: DecodeReservationAttempt
    grant_id: uuid.UUID
    allocations: tuple[DecodeReservationAllocation, ...]
    prepared_expires_at_unix_ms: int
    prepared_deadline_monotonic_ns: int
    reservation_digest: bytes
    opaque_allocation_owner: object | None
    state: DecodeReservationState = DecodeReservationState.PREPARED_UNBOUND
    bound_request_body: bytes | None = None
    grant_digest: bytes | None = None
    receipts: dict[DecodeReservationOperation, dict[str, object]] = dataclasses.field(
        default_factory=dict
    )
    reserve_refusal_receipt: dict[str, object] | None = None
    inflight_operation: DecodeReservationOperation | None = None
    inflight_reason_code: str | None = None
    inflight_diagnostic: str | None = None
    operation_executing: bool = False
    inference_attachment_executing: bool = False
    inference_attached: bool = False
    terminal_reason_code: str | None = None
    terminal_diagnostic: str | None = None

    def __repr__(self) -> str:
        """Return a prompt- and secret-redacted representation.

        :returns: Safe diagnostic representation.
        """

        return (
            "_DecodeReservationRecord("
            f"grant_id={self.grant_id!s}, "
            f"reservation_attempt_id={self.attempt.reservation_attempt_id!s}, "
            f"state={self.state.value!r}, "
            f"child_count={len(self.allocations)}, "
            f"has_bound_request={self.bound_request_body is not None})"
        )


@dataclasses.dataclass(frozen=True)
class _DecodeReservationExpiryClaim:
    """One exact allocator action claimed by the expiry sweeper."""

    grant_id: uuid.UUID
    operation: DecodeReservationOperation
    owner: object
    reason_code: str | None
    diagnostic: str | None


@dataclasses.dataclass(frozen=True)
class _DecodeReservationRefusalRecord:
    """One immutable take-once allocator admission refusal."""

    attempt: DecodeReservationAttempt
    response: dict[str, object]


class DecodeReservationAuthority:
    """Single scheduler-owned reservation state and allocation authority."""

    _abort: DecodeReservationTerminalAction
    _cancel: DecodeReservationTerminalAction
    _complete: DecodeReservationTerminalAction
    _expected_decoder_instance_id: uuid.UUID
    _lock: threading.Lock
    _monotonic_clock_ns: Callable[[], int]
    _prepared_grants: set[uuid.UUID]
    _promote: Callable[[object], None]
    _quarantine: DecodeReservationTerminalAction
    _records: dict[uuid.UUID, _DecodeReservationRecord]
    _refusals: dict[uuid.UUID, _DecodeReservationRefusalRecord]
    _reserve_attempts: dict[uuid.UUID, uuid.UUID]
    _wall_clock_unix_ms: Callable[[], int]

    def __init__(
        self,
        *,
        expected_decoder_instance_id: uuid.UUID,
        promote: Callable[[object], None],
        cancel: DecodeReservationTerminalAction,
        complete: DecodeReservationTerminalAction,
        abort: DecodeReservationTerminalAction,
        quarantine: DecodeReservationTerminalAction,
        wall_clock_unix_ms: Callable[[], int] = _wall_clock_unix_ms,
        monotonic_clock_ns: Callable[[], int] = _monotonic_clock_ns,
    ) -> None:
        """Initialize the process-local scheduler authority.

        :param expected_decoder_instance_id: This decoder launch generation.
        :param promote: Atomic cohort promotion action.
        :param cancel: Prepared allocation cleanup action.
        :param complete: Promoted request completion reconciliation.
        :param abort: Promoted request abort reconciliation.
        :param quarantine: Ambiguous ownership retention action.
        :param wall_clock_unix_ms: Wire-transcript wall clock.
        :param monotonic_clock_ns: Ownership-deadline monotonic clock.
        """

        if expected_decoder_instance_id.int == 0:
            raise ValueError("expected decoder instance ID cannot be nil")
        self._expected_decoder_instance_id = expected_decoder_instance_id
        self._promote = promote
        self._cancel = cancel
        self._complete = complete
        self._abort = abort
        self._quarantine = quarantine
        self._wall_clock_unix_ms = wall_clock_unix_ms
        self._monotonic_clock_ns = monotonic_clock_ns
        self._lock = threading.Lock()
        self._prepared_grants = set()
        self._records = {}
        self._refusals = {}
        self._reserve_attempts = {}

    def refuse(
        self,
        attempt: DecodeReservationAttempt,
        refusal: DecodeReservationAdmissionRefused,
    ) -> dict[str, object]:
        """Publish one take-once admission refusal without allocation ownership.

        :param attempt: Exact authenticated reserve attempt.
        :param refusal: Allocator-issued pre-mutation refusal.
        :returns: Stable Rust-compatible refusal receipt.
        """

        if attempt.decoder_process.instance_id != self._expected_decoder_instance_id:
            raise DecodeReservationValidationError(
                "decoder process generation differs from this engine"
            )
        if type(refusal) is not DecodeReservationAdmissionRefused:
            raise TypeError("refusal must be DecodeReservationAdmissionRefused")
        with self._lock:
            if attempt.reservation_attempt_id in self._reserve_attempts:
                raise DecodeReservationConflictError(
                    "live reservation already owns the refused attempt ID"
                )
            existing = self._refusals.get(attempt.reservation_attempt_id)
            if existing is not None:
                if not secrets.compare_digest(
                    existing.attempt.reserve_attempt_digest,
                    attempt.reserve_attempt_digest,
                ):
                    raise DecodeReservationConflictError(
                        "reservation attempt ID was reused with another digest"
                    )
                return self._copy_receipt(existing.response)
            response = self._admission_refusal_receipt(attempt, refusal)
            self._refusals[attempt.reservation_attempt_id] = (
                _DecodeReservationRefusalRecord(
                    attempt=attempt,
                    response=response,
                )
            )
            return self._copy_receipt(response)

    def prepare(
        self,
        attempt: DecodeReservationAttempt,
        grant_id: uuid.UUID,
        allocations: tuple[DecodeReservationAllocation, ...],
        opaque_allocation_owner: object,
        *,
        prepared_expires_at_unix_ms: int | None = None,
    ) -> DecodeReservationSnapshot:
        """Publish one allocator-prepared reservation idempotently.

        :param attempt: Exact authenticated reserve attempt.
        :param grant_id: Tokenizer-issued non-secret grant identity.
        :param allocations: Ordered allocator-issued child receipts.
        :param opaque_allocation_owner: Exact retained Req/DecodeRequest cohort.
        :param prepared_expires_at_unix_ms: Process-global prepared deadline.
        :returns: Immutable new or idempotently retained reservation snapshot.
        """

        if attempt.decoder_process.instance_id != self._expected_decoder_instance_id:
            raise DecodeReservationValidationError(
                "decoder process generation differs from this engine"
            )
        owned_allocations = tuple(allocations)
        if (
            tuple(allocation.child_request_id for allocation in owned_allocations)
            != attempt.child_request_ids
        ):
            raise DecodeReservationValidationError(
                "allocation children differ from reserve request order"
            )
        if opaque_allocation_owner is None:
            raise DecodeReservationValidationError(
                "prepared reservation requires an allocation owner"
            )
        now_unix_ms = self._wall_clock_unix_ms()
        now_monotonic_ns = self._monotonic_clock_ns()
        if type(now_unix_ms) is not int or now_unix_ms < 0:
            raise ValueError("wall_clock_unix_ms must return a non-negative integer")
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise ValueError("monotonic_clock_ns must return a non-negative integer")
        expires_at = prepared_expires_at_unix_ms
        if expires_at is None:
            expires_at = now_unix_ms + attempt.prepared_ttl_ms
        if type(expires_at) is not int or expires_at <= 0:
            raise DecodeReservationValidationError(
                "prepared reservation expiry must be a positive integer"
            )
        if expires_at >= 1 << 64:
            raise DecodeReservationValidationError(
                "prepared reservation expiry overflows u64 milliseconds"
            )
        remaining_ttl_ms = max(expires_at - now_unix_ms, 0)
        deadline_monotonic_ns = now_monotonic_ns + remaining_ttl_ms * 1_000_000

        with self._lock:
            existing_id = self._reserve_attempts.get(attempt.reservation_attempt_id)
            if existing_id is not None:
                existing = self._records[existing_id]
                if not secrets.compare_digest(
                    existing.attempt.reserve_attempt_digest,
                    attempt.reserve_attempt_digest,
                ):
                    raise DecodeReservationConflictError(
                        "reservation attempt ID was reused with another digest"
                    )
                if existing.grant_id != grant_id:
                    raise DecodeReservationConflictError(
                        "reservation attempt ID was reused with another grant ID"
                    )
                return self._snapshot_locked(existing)

            if grant_id.int == 0:
                raise DecodeReservationValidationError("grant_id cannot be nil")
            if grant_id in self._records:
                raise DecodeReservationConflictError(
                    "grant ID is already owned by another reserve attempt"
                )
            reservation_digest = _compute_reservation_digest(
                grant_id,
                attempt.reserve_attempt_digest,
                expires_at,
                owned_allocations,
            )
            record = _DecodeReservationRecord(
                attempt=attempt,
                grant_id=grant_id,
                allocations=owned_allocations,
                prepared_expires_at_unix_ms=expires_at,
                prepared_deadline_monotonic_ns=deadline_monotonic_ns,
                reservation_digest=reservation_digest,
                opaque_allocation_owner=opaque_allocation_owner,
            )
            self._records[grant_id] = record
            self._prepared_grants.add(grant_id)
            self._reserve_attempts[attempt.reservation_attempt_id] = grant_id
            return self._snapshot_locked(record)

    def bind(
        self,
        grant_id: uuid.UUID,
        request_body: bytes,
    ) -> dict[str, object]:
        """Bind exact final inference bytes without changing allocation.

        :param grant_id: Exact prepared grant.
        :param request_body: Exact final inference body bytes.
        :returns: Idempotent bind receipt.
        """

        owned_body = bytes(request_body)
        with self._lock:
            record = self._require_record_locked(grant_id)
            if self._is_prepared_expired_locked(record):
                raise DecodeReservationExpiredError(
                    "prepared reservation expired before bind"
                )
            if record.inflight_operation is not None:
                raise DecodeReservationConflictError(
                    f"{record.inflight_operation.value} reconciliation blocks bind"
                )
            if record.state is DecodeReservationState.PREPARED_BOUND:
                if record.bound_request_body != owned_body:
                    raise DecodeReservationConflictError(
                        "bound request bytes cannot be replaced"
                    )
                return self._copy_receipt(
                    record.receipts[DecodeReservationOperation.BIND]
                )
            if record.state is not DecodeReservationState.PREPARED_UNBOUND:
                raise DecodeReservationConflictError(
                    f"bind is invalid in state {record.state.value}"
                )
            self._validate_bound_request(record, owned_body)
            record.bound_request_body = owned_body
            record.grant_digest = _compute_grant_digest(
                record.reservation_digest,
                owned_body,
            )
            record.state = DecodeReservationState.PREPARED_BOUND
            receipt = self._receipt(
                record,
                DecodeReservationOperation.BIND,
                "prepared",
            )
            record.receipts[DecodeReservationOperation.BIND] = receipt
            return self._copy_receipt(receipt)

    def transition(
        self,
        grant_id: uuid.UUID,
        operation: DecodeReservationOperation,
        transcript: object,
    ) -> dict[str, object]:
        """Apply one authenticated post-bind operation.

        :param grant_id: Exact grant identity.
        :param operation: Closed lifecycle operation.
        :param transcript: Rust binding-control request.
        :returns: Idempotent exact receipt.
        """

        fields = _require_mapping(transcript, f"{operation.value} request")
        expected_fields = (
            _FAILURE_FIELDS
            if operation
            in (
                DecodeReservationOperation.ABORT,
                DecodeReservationOperation.QUARANTINE,
            )
            else _BINDING_FIELDS
        )
        _require_exact_fields(fields, expected_fields, f"{operation.value} request")
        reason_code, diagnostic = _failure_context(fields, operation)

        with self._lock:
            record = self._require_record_locked(grant_id)
            self._validate_binding_transcript(record, fields)
            existing = record.receipts.get(operation)
            if existing is not None:
                return self._copy_receipt(existing)
            self._begin_operation_locked(
                record,
                operation,
                reason_code=reason_code,
                diagnostic=diagnostic,
            )
            owner = record.opaque_allocation_owner
            if owner is None:
                raise RuntimeError("active reservation lost its allocation owner")

        operation_completed = False
        try:
            terminal = self._execute_operation(
                owner,
                operation,
                reason_code,
                diagnostic,
            )
            self._validate_operation_result(operation, terminal)
            operation_completed = True
        finally:
            if not operation_completed:
                self._release_failed_execution(grant_id, operation)

        with self._lock:
            record = self._require_record_locked(grant_id)
            if (
                record.inflight_operation is not operation
                or not record.operation_executing
            ):
                raise DecodeReservationConflictError(
                    "reservation operation ownership changed during execution"
                )
            self._apply_operation_result_locked(
                record,
                terminal,
                reason_code,
                diagnostic,
            )
            state = (
                "prepared"
                if operation is DecodeReservationOperation.BIND
                else terminal.value
            )
            receipt = self._receipt(record, operation, state)
            record.receipts[operation] = receipt
            record.inflight_operation = None
            record.inflight_reason_code = None
            record.inflight_diagnostic = None
            record.operation_executing = False
            return self._copy_receipt(receipt)

    def cancel_unbound(
        self,
        grant_id: uuid.UUID,
        transcript: object,
    ) -> dict[str, object]:
        """Cancel a PREPARED_UNBOUND allocation exactly.

        :param grant_id: Exact grant identity.
        :param transcript: Rust unbound-cancellation request.
        :returns: Idempotent cancellation receipt.
        """

        fields = _require_mapping(transcript, "unbound cancellation request")
        _require_exact_fields(
            fields,
            _UNBOUND_CANCELLATION_FIELDS,
            "unbound cancellation request",
        )
        with self._lock:
            record = self._require_record_locked(grant_id)
            existing = record.receipts.get(DecodeReservationOperation.CANCEL)
            if existing is not None:
                return self._copy_receipt(existing)
            self._validate_unbound_transcript(record, fields)
            if (
                record.state is DecodeReservationState.CANCELLED
                and record.inflight_operation is None
            ):
                receipt = self._unbound_cancellation_receipt(record, fields)
                record.receipts[DecodeReservationOperation.CANCEL] = receipt
                return self._copy_receipt(receipt)
            if record.state is not DecodeReservationState.PREPARED_UNBOUND:
                raise DecodeReservationConflictError(
                    f"unbound cancellation is invalid in state {record.state.value}"
                )
            self._begin_operation_locked(
                record,
                DecodeReservationOperation.CANCEL,
                expected_states=(DecodeReservationState.PREPARED_UNBOUND,),
            )
            owner = record.opaque_allocation_owner
            if owner is None:
                raise RuntimeError("active reservation lost its allocation owner")

        operation_completed = False
        try:
            terminal = self._cancel(owner, None, None)
            if terminal is not DecodeReservationState.CANCELLED:
                raise DecodeReservationConflictError(
                    "cancel action did not prove cancellation"
                )
            operation_completed = True
        finally:
            if not operation_completed:
                self._release_failed_execution(
                    grant_id,
                    DecodeReservationOperation.CANCEL,
                )

        with self._lock:
            record = self._require_record_locked(grant_id)
            if (
                record.inflight_operation is not DecodeReservationOperation.CANCEL
                or not record.operation_executing
            ):
                raise DecodeReservationConflictError(
                    "unbound cancellation ownership changed during execution"
                )
            self._apply_operation_result_locked(record, terminal, None, None)
            receipt = self._unbound_cancellation_receipt(record, fields)
            record.receipts[DecodeReservationOperation.CANCEL] = receipt
            record.inflight_operation = None
            record.inflight_reason_code = None
            record.inflight_diagnostic = None
            record.operation_executing = False
            return self._copy_receipt(receipt)

    def sweep_expired_prepared(self) -> DecodeReservationExpirySweep:
        """Reconcile every prepared reservation past its monotonic deadline.

        Clean PREPARED_UNBOUND and PREPARED_BOUND cohorts are cancelled because
        promotion is the only publication boundary. A failed promotion is
        quarantined because submission may have occurred. Allocator callbacks
        execute without the authority lock.

        :returns: Exact sweep outcome grouped by allocator result.
        """

        now_monotonic_ns = self._monotonic_clock_ns()
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise ValueError("monotonic_clock_ns must return a non-negative integer")
        with self._lock:
            grant_ids = tuple(self._prepared_grants)

        cancelled: list[uuid.UUID] = []
        quarantined: list[uuid.UUID] = []
        deferred: list[uuid.UUID] = []
        failed: list[uuid.UUID] = []
        for grant_id in grant_ids:
            with self._lock:
                claim, is_deferred = self._claim_expiry_locked(
                    grant_id,
                    now_monotonic_ns,
                )
            if is_deferred:
                deferred.append(grant_id)
                continue
            if claim is None:
                continue

            operation_completed = False
            try:
                terminal = self._execute_operation(
                    claim.owner,
                    claim.operation,
                    claim.reason_code,
                    claim.diagnostic,
                )
                self._validate_operation_result(claim.operation, terminal)
                operation_completed = True
            except Exception:  # noqa: BLE001
                logger.error(
                    "Prepared reservation expiry failed for grant %s\n%s",
                    claim.grant_id,
                    traceback.format_exc(),
                )
                failed.append(claim.grant_id)
            finally:
                if not operation_completed:
                    self._release_failed_execution(
                        claim.grant_id,
                        claim.operation,
                    )
            if not operation_completed:
                continue

            with self._lock:
                self._commit_expiry_locked(claim, terminal)
            if terminal is DecodeReservationState.CANCELLED:
                cancelled.append(claim.grant_id)
            else:
                quarantined.append(claim.grant_id)

        return DecodeReservationExpirySweep(
            scanned_count=len(grant_ids),
            cancelled_grant_ids=tuple(cancelled),
            quarantined_grant_ids=tuple(quarantined),
            deferred_grant_ids=tuple(deferred),
            failed_grant_ids=tuple(failed),
        )

    def snapshot(self, grant_id: uuid.UUID) -> DecodeReservationSnapshot:
        """Return an immutable redacted reservation snapshot.

        :param grant_id: Exact grant identity.
        :returns: Immutable reservation snapshot.
        """

        with self._lock:
            return self._snapshot_locked(self._require_record_locked(grant_id))

    def snapshot_for_attempt(
        self,
        reservation_attempt_id: uuid.UUID,
    ) -> DecodeReservationSnapshot | None:
        """Return an immutable snapshot for an idempotent reserve attempt.

        :param reservation_attempt_id: Exact reserve-attempt identity.
        :returns: Immutable reservation snapshot, otherwise ``None``.
        """

        with self._lock:
            grant_id = self._reserve_attempts.get(reservation_attempt_id)
            if grant_id is None:
                return None
            return self._snapshot_locked(self._records[grant_id])

    def claim_inference_attachment(
        self,
        grant_id: uuid.UUID,
        inference_route: str,
        request_body: bytes,
    ) -> object | None:
        """Claim exact promoted inference attachment for allocator publication.

        :param grant_id: Exact promoted grant identity.
        :param inference_route: Actual inference HTTP route.
        :param request_body: Actual raw inference body bytes.
        :returns: Exact retained allocation owner, or ``None`` after prior attachment.
        """

        owned_body = bytes(request_body)
        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.state is not DecodeReservationState.PROMOTED:
                raise DecodeReservationConflictError(
                    f"inference attachment is invalid in state {record.state.value}"
                )
            if record.attempt.inference_route != inference_route:
                raise DecodeReservationValidationError(
                    "inference attachment changed the reserved route"
                )
            if record.bound_request_body is None:
                raise RuntimeError("promoted reservation lost its bound request bytes")
            if not secrets.compare_digest(record.bound_request_body, owned_body):
                raise DecodeReservationValidationError(
                    "inference attachment changed the bound request bytes"
                )
            if record.inference_attached:
                return None
            if record.inference_attachment_executing:
                raise DecodeReservationOperationInFlightError(
                    "inference attachment is already executing"
                )
            if record.inflight_operation is not None:
                raise DecodeReservationConflictError(
                    f"{record.inflight_operation.value} reconciliation blocks "
                    "inference attachment"
                )
            owner = record.opaque_allocation_owner
            if owner is None:
                raise RuntimeError("promoted reservation lost its allocation owner")
            record.inference_attachment_executing = True
            return owner

    def commit_inference_attachment(self, grant_id: uuid.UUID) -> None:
        """Commit successful exact inference attachment.

        :param grant_id: Exact promoted grant identity.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            if (
                record.state is not DecodeReservationState.PROMOTED
                or not record.inference_attachment_executing
                or record.inference_attached
            ):
                raise DecodeReservationConflictError(
                    "inference attachment ownership changed during publication"
                )
            record.inference_attached = True
            record.inference_attachment_executing = False

    def quarantine_failed_inference_attachment(
        self,
        grant_id: uuid.UUID,
        diagnostic: str,
    ) -> None:
        """Record conservative quarantine after ambiguous attachment publication.

        :param grant_id: Exact promoted grant identity.
        :param diagnostic: Bounded publication failure diagnostic.
        """

        with self._lock:
            record = self._require_record_locked(grant_id)
            if not record.inference_attachment_executing:
                raise DecodeReservationConflictError(
                    "failed inference attachment no longer owns publication"
                )
            self._apply_operation_result_locked(
                record,
                DecodeReservationState.QUARANTINED,
                "inference_attachment_failed",
                diagnostic,
            )
            record.receipts[DecodeReservationOperation.QUARANTINE] = self._receipt(
                record,
                DecodeReservationOperation.QUARANTINE,
                DecodeReservationState.QUARANTINED.value,
            )
            record.inference_attachment_executing = False

    def reserve_retry_response(
        self,
        reservation_attempt_id: uuid.UUID,
        reserve_attempt_digest: bytes,
    ) -> dict[str, object] | None:
        """Return the current authoritative response for an exact retry.

        Live records return their original PREPARED transcript. Terminal
        records return a stable Rust-validated reserve-refusal receipt, never a
        synthetic live allocation.

        :param reservation_attempt_id: Exact reserve-attempt identity.
        :param reserve_attempt_digest: Exact reserve transcript digest.
        :returns: Prepared or refused response, otherwise ``None``.
        """

        with self._lock:
            refusal = self._refusals.get(reservation_attempt_id)
            if refusal is not None:
                if not secrets.compare_digest(
                    refusal.attempt.reserve_attempt_digest,
                    reserve_attempt_digest,
                ):
                    raise DecodeReservationConflictError(
                        "reservation attempt ID was reused with another digest"
                    )
                return self._copy_receipt(refusal.response)
            grant_id = self._reserve_attempts.get(reservation_attempt_id)
            if grant_id is None:
                return None
            record = self._records[grant_id]
            if not secrets.compare_digest(
                record.attempt.reserve_attempt_digest,
                reserve_attempt_digest,
            ):
                raise DecodeReservationConflictError(
                    "reservation attempt ID was reused with another digest"
                )
            if record.state in (
                DecodeReservationState.PREPARED_UNBOUND,
                DecodeReservationState.PREPARED_BOUND,
                DecodeReservationState.PROMOTED,
            ):
                if self._is_prepared_expired_locked(record):
                    raise DecodeReservationExpiredError(
                        "expired reservation has no authoritative terminal receipt"
                    )
                return self._snapshot_locked(record).reserve_response()
            if record.reserve_refusal_receipt is None:
                record.reserve_refusal_receipt = self._reserve_refusal_receipt(record)
            return self._copy_receipt(record.reserve_refusal_receipt)

    def _require_record_locked(
        self,
        grant_id: uuid.UUID,
    ) -> _DecodeReservationRecord:
        record = self._records.get(grant_id)
        if record is None:
            raise DecodeReservationNotFoundError("decoder reservation not found")
        return record

    @staticmethod
    def _snapshot_locked(
        record: _DecodeReservationRecord,
    ) -> DecodeReservationSnapshot:
        return DecodeReservationSnapshot(
            grant_id=record.grant_id,
            reservation_attempt_id=record.attempt.reservation_attempt_id,
            reserve_attempt_digest=record.attempt.reserve_attempt_digest,
            prefill_process=record.attempt.prefill_process,
            prefill_bootstrap_endpoint=(record.attempt.prefill_bootstrap_endpoint),
            decoder_process=record.attempt.decoder_process,
            logical_request_chain_id=record.attempt.logical_request_chain_id,
            source_tp_size=record.attempt.source_tp_size,
            prepared_ttl_ms=record.attempt.prepared_ttl_ms,
            prepared_expires_at_unix_ms=(record.prepared_expires_at_unix_ms),
            inference_route=record.attempt.inference_route,
            request_shape=record.attempt.request_shape,
            child_request_ids=record.attempt.child_request_ids,
            allocations=record.allocations,
            reservation_digest=record.reservation_digest,
            grant_digest=record.grant_digest,
            state=record.state,
            inference_attached=record.inference_attached,
        )

    def _begin_operation_locked(
        self,
        record: _DecodeReservationRecord,
        operation: DecodeReservationOperation,
        *,
        expected_states: tuple[DecodeReservationState, ...] | None = None,
        reason_code: str | None = None,
        diagnostic: str | None = None,
    ) -> None:
        if record.inference_attachment_executing:
            raise DecodeReservationConflictError(
                "inference attachment publication blocks control reconciliation"
            )
        if expected_states is None:
            expected_states = self._expected_operation_states(operation)
        if record.state not in expected_states:
            raise DecodeReservationConflictError(
                f"{operation.value} is invalid in state {record.state.value}"
            )
        if (
            operation is DecodeReservationOperation.PROMOTE
            and self._is_prepared_expired_locked(record)
        ):
            raise DecodeReservationExpiredError(
                "prepared reservation expired before promote"
            )
        if record.inflight_operation is not None:
            if record.inflight_operation is not operation:
                raise DecodeReservationConflictError(
                    f"{record.inflight_operation.value} reconciliation blocks "
                    f"{operation.value}"
                )
            if (
                record.inflight_reason_code != reason_code
                or record.inflight_diagnostic != diagnostic
            ):
                raise DecodeReservationConflictError(
                    f"{operation.value} reconciliation context changed"
                )
            if record.operation_executing:
                raise DecodeReservationOperationInFlightError(
                    f"{operation.value} is already executing"
                )
        record.inflight_operation = operation
        record.inflight_reason_code = reason_code
        record.inflight_diagnostic = diagnostic
        record.operation_executing = True

    def _claim_expiry_locked(
        self,
        grant_id: uuid.UUID,
        now_monotonic_ns: int,
    ) -> tuple[_DecodeReservationExpiryClaim | None, bool]:
        record = self._require_record_locked(grant_id)
        if (
            record.state
            not in (
                DecodeReservationState.PREPARED_UNBOUND,
                DecodeReservationState.PREPARED_BOUND,
            )
            or record.prepared_deadline_monotonic_ns > now_monotonic_ns
        ):
            return None, False
        if record.operation_executing:
            return None, True
        owner = record.opaque_allocation_owner
        if owner is None:
            raise RuntimeError("prepared reservation lost its allocation owner")

        operation: DecodeReservationOperation
        reason_code: str | None
        diagnostic: str | None
        if record.state is DecodeReservationState.PREPARED_UNBOUND:
            operation = DecodeReservationOperation.CANCEL
            if record.inflight_operation is DecodeReservationOperation.CANCEL:
                reason_code = record.inflight_reason_code
                diagnostic = record.inflight_diagnostic
            else:
                reason_code = "prepared_ttl_expired"
                diagnostic = None
        elif record.inflight_operation in (
            DecodeReservationOperation.CANCEL,
            DecodeReservationOperation.QUARANTINE,
        ):
            operation = record.inflight_operation
            reason_code = record.inflight_reason_code
            diagnostic = record.inflight_diagnostic
        elif record.inflight_operation is DecodeReservationOperation.PROMOTE:
            operation = DecodeReservationOperation.QUARANTINE
            reason_code = "promotion_reconciliation_expired"
            diagnostic = None
        else:
            operation = DecodeReservationOperation.CANCEL
            reason_code = "prepared_ttl_expired"
            diagnostic = None

        record.inflight_operation = operation
        record.inflight_reason_code = reason_code
        record.inflight_diagnostic = diagnostic
        record.operation_executing = True
        return (
            _DecodeReservationExpiryClaim(
                grant_id=grant_id,
                operation=operation,
                owner=owner,
                reason_code=reason_code,
                diagnostic=diagnostic,
            ),
            False,
        )

    def _commit_expiry_locked(
        self,
        claim: _DecodeReservationExpiryClaim,
        terminal: DecodeReservationState,
    ) -> None:
        record = self._require_record_locked(claim.grant_id)
        if (
            record.inflight_operation is not claim.operation
            or not record.operation_executing
        ):
            raise DecodeReservationConflictError(
                "expiry operation ownership changed during execution"
            )
        self._apply_operation_result_locked(
            record,
            terminal,
            claim.reason_code,
            claim.diagnostic,
        )
        if record.grant_digest is not None:
            receipt = self._receipt(record, claim.operation, terminal.value)
            record.receipts[claim.operation] = receipt
        record.inflight_operation = None
        record.inflight_reason_code = None
        record.inflight_diagnostic = None
        record.operation_executing = False

    def _is_prepared_expired_locked(
        self,
        record: _DecodeReservationRecord,
    ) -> bool:
        if record.state not in (
            DecodeReservationState.PREPARED_UNBOUND,
            DecodeReservationState.PREPARED_BOUND,
        ):
            return False
        now_monotonic_ns = self._monotonic_clock_ns()
        if type(now_monotonic_ns) is not int or now_monotonic_ns < 0:
            raise ValueError("monotonic_clock_ns must return a non-negative integer")
        return now_monotonic_ns >= record.prepared_deadline_monotonic_ns

    def _apply_operation_result_locked(
        self,
        record: _DecodeReservationRecord,
        terminal: DecodeReservationState,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> None:
        record.state = terminal
        self._prepared_grants.discard(record.grant_id)
        if terminal is DecodeReservationState.PROMOTED:
            return
        record.terminal_reason_code = reason_code
        record.terminal_diagnostic = diagnostic
        record.bound_request_body = None
        if terminal is not DecodeReservationState.QUARANTINED:
            record.opaque_allocation_owner = None

    def _release_failed_execution(
        self,
        grant_id: uuid.UUID,
        operation: DecodeReservationOperation,
    ) -> None:
        with self._lock:
            record = self._require_record_locked(grant_id)
            if record.inflight_operation is operation and record.operation_executing:
                record.operation_executing = False

    @staticmethod
    def _expected_operation_states(
        operation: DecodeReservationOperation,
    ) -> tuple[DecodeReservationState, ...]:
        if operation in (
            DecodeReservationOperation.PROMOTE,
            DecodeReservationOperation.CANCEL,
        ):
            return (DecodeReservationState.PREPARED_BOUND,)
        if operation in (
            DecodeReservationOperation.COMPLETE,
            DecodeReservationOperation.ABORT,
        ):
            return (DecodeReservationState.PROMOTED,)
        if operation is DecodeReservationOperation.QUARANTINE:
            return (
                DecodeReservationState.PREPARED_BOUND,
                DecodeReservationState.PROMOTED,
            )
        raise DecodeReservationValidationError(
            f"unsupported transition {operation.value}"
        )

    def _execute_operation(
        self,
        owner: object,
        operation: DecodeReservationOperation,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        if operation is DecodeReservationOperation.PROMOTE:
            self._promote(owner)
            return DecodeReservationState.PROMOTED
        if operation is DecodeReservationOperation.CANCEL:
            return self._cancel(owner, None, None)
        if operation is DecodeReservationOperation.COMPLETE:
            return self._complete(owner, None, None)
        if operation is DecodeReservationOperation.ABORT:
            if reason_code is None:
                raise RuntimeError("validated abort lost its reason code")
            return self._abort(owner, reason_code, diagnostic)
        if operation is DecodeReservationOperation.QUARANTINE:
            if reason_code is None:
                raise RuntimeError("validated quarantine lost its reason code")
            return self._quarantine(owner, reason_code, diagnostic)
        raise DecodeReservationValidationError(
            f"unsupported transition {operation.value}"
        )

    @staticmethod
    def _validate_operation_result(
        operation: DecodeReservationOperation,
        terminal: DecodeReservationState,
    ) -> None:
        expected: tuple[DecodeReservationState, ...]
        if operation is DecodeReservationOperation.PROMOTE:
            expected = (DecodeReservationState.PROMOTED,)
        elif operation is DecodeReservationOperation.CANCEL:
            expected = (DecodeReservationState.CANCELLED,)
        elif operation is DecodeReservationOperation.COMPLETE:
            expected = (
                DecodeReservationState.COMPLETED,
                DecodeReservationState.QUARANTINED,
            )
        elif operation is DecodeReservationOperation.ABORT:
            expected = (
                DecodeReservationState.ABORTED,
                DecodeReservationState.QUARANTINED,
            )
        elif operation is DecodeReservationOperation.QUARANTINE:
            expected = (DecodeReservationState.QUARANTINED,)
        else:
            raise DecodeReservationValidationError(
                f"unsupported transition {operation.value}"
            )
        if terminal not in expected:
            raise DecodeReservationConflictError(
                f"{operation.value} action returned state {terminal.value}"
            )

    @staticmethod
    def _validate_bound_request(
        record: _DecodeReservationRecord,
        request_body: bytes,
    ) -> None:
        try:
            bound = json.loads(request_body)
        except json.JSONDecodeError as error:
            raise DecodeReservationValidationError(
                "bound inference body must contain valid JSON"
            ) from error
        try:
            base = json.loads(record.attempt.base_request_body)
        except json.JSONDecodeError as error:
            raise RuntimeError("validated base request became invalid") from error
        if type(bound) is not dict:
            raise DecodeReservationValidationError(
                "bound inference body must contain a JSON object"
            )
        rooms = [allocation.bootstrap_room for allocation in record.allocations]
        child_count = len(record.allocations)
        expected_host: object = record.attempt.prefill_bootstrap_endpoint.host
        expected_port: object = record.attempt.prefill_bootstrap_endpoint.port
        expected_rooms: object = rooms[0]
        if record.attempt.request_shape == "batch":
            expected_host = [expected_host] * child_count
            expected_port = [expected_port] * child_count
            expected_rooms = rooms
        for key, expected in (
            ("bootstrap_host", expected_host),
            ("bootstrap_port", expected_port),
            ("bootstrap_room", expected_rooms),
        ):
            if bound.pop(key, None) != expected:
                raise DecodeReservationValidationError(
                    f"bound request contains an invalid {key}"
                )
        if bound != base:
            raise DecodeReservationValidationError(
                "bound request changed fields outside the grant binding"
            )

    @staticmethod
    def _validate_binding_transcript(
        record: _DecodeReservationRecord,
        fields: Mapping[str, object],
    ) -> None:
        _require_schema_version(fields["schema_version"])
        if record.grant_digest is None:
            raise DecodeReservationConflictError(
                "binding transcript requires a bound request"
            )
        expected: dict[str, object] = {
            "grant_id": str(record.grant_id),
            "reservation_attempt_id": str(record.attempt.reservation_attempt_id),
            "reserve_attempt_digest": record.attempt.reserve_attempt_digest.hex(),
            "prefill_process": record.attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                record.attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": record.attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(record.attempt.logical_request_chain_id),
            "source_tp_size": record.attempt.source_tp_size,
            "inference_route": record.attempt.inference_route,
            "request_shape": record.attempt.request_shape,
            "prepared_ttl_ms": record.attempt.prepared_ttl_ms,
            "prepared_expires_at_unix_ms": (record.prepared_expires_at_unix_ms),
            "child_request_ids": [
                str(value) for value in record.attempt.child_request_ids
            ],
            "decoder_slot_generations": [
                str(value.decoder_slot_generation) for value in record.allocations
            ],
            "bootstrap_rooms": [value.bootstrap_room for value in record.allocations],
            "reservation_digest": record.reservation_digest.hex(),
            "grant_digest": record.grant_digest.hex(),
        }
        for name, value in expected.items():
            if fields[name] != value:
                raise DecodeReservationValidationError(
                    f"control transcript changed {name}"
                )

    @staticmethod
    def _validate_unbound_transcript(
        record: _DecodeReservationRecord,
        fields: Mapping[str, object],
    ) -> None:
        _require_schema_version(fields["schema_version"])
        expected: dict[str, object] = {
            "grant_id": str(record.grant_id),
            "reservation_attempt_id": str(record.attempt.reservation_attempt_id),
            "reserve_attempt_digest": record.attempt.reserve_attempt_digest.hex(),
            "prefill_process": record.attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                record.attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": record.attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(record.attempt.logical_request_chain_id),
            "source_tp_size": record.attempt.source_tp_size,
            "inference_route": record.attempt.inference_route,
            "request_shape": record.attempt.request_shape,
            "prepared_ttl_ms": record.attempt.prepared_ttl_ms,
            "prepared_expires_at_unix_ms": (record.prepared_expires_at_unix_ms),
            "child_request_ids": [
                str(value) for value in record.attempt.child_request_ids
            ],
            "decoder_slot_generations": [
                str(value.decoder_slot_generation) for value in record.allocations
            ],
            "bootstrap_rooms": [value.bootstrap_room for value in record.allocations],
            "reservation_digest": record.reservation_digest.hex(),
        }
        for name, value in expected.items():
            if fields[name] != value:
                raise DecodeReservationValidationError(
                    f"unbound cancellation changed {name}"
                )
        attempted = fields["attempted_grant_digest"]
        if attempted is not None:
            _require_digest(attempted, "attempted_grant_digest")

    @staticmethod
    def _receipt(
        record: _DecodeReservationRecord,
        operation: DecodeReservationOperation,
        state: str,
    ) -> dict[str, object]:
        if record.grant_digest is None:
            raise RuntimeError("bound receipt requires a grant digest")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "grant_id": str(record.grant_id),
            "reservation_attempt_id": str(record.attempt.reservation_attempt_id),
            "reserve_attempt_digest": record.attempt.reserve_attempt_digest.hex(),
            "prefill_process": record.attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                record.attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": record.attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(record.attempt.logical_request_chain_id),
            "source_tp_size": record.attempt.source_tp_size,
            "inference_route": record.attempt.inference_route,
            "request_shape": record.attempt.request_shape,
            "prepared_ttl_ms": record.attempt.prepared_ttl_ms,
            "prepared_expires_at_unix_ms": (record.prepared_expires_at_unix_ms),
            "child_request_ids": [
                str(value) for value in record.attempt.child_request_ids
            ],
            "decoder_slot_generations": [
                str(value.decoder_slot_generation) for value in record.allocations
            ],
            "bootstrap_rooms": [value.bootstrap_room for value in record.allocations],
            "reservation_digest": record.reservation_digest.hex(),
            "grant_digest": record.grant_digest.hex(),
            "operation": operation.value,
            "state": state,
            "receipt_id": str(uuid.uuid4()),
            "take_once": True,
        }
        receipt["receipt_digest"] = _receipt_digest(receipt).hex()
        return receipt

    @staticmethod
    def _reserve_refusal_receipt(
        record: _DecodeReservationRecord,
    ) -> dict[str, object]:
        if record.state is DecodeReservationState.CANCELLED:
            reason_code = record.terminal_reason_code or "reservation_cancelled"
            disposition = "retry_same_decoder"
        elif record.state is DecodeReservationState.QUARANTINED:
            reason_code = record.terminal_reason_code or "reservation_quarantined"
            disposition = "retry_another_decoder"
        elif record.state is DecodeReservationState.COMPLETED:
            reason_code = "reservation_completed"
            disposition = "terminal"
        elif record.state is DecodeReservationState.ABORTED:
            reason_code = record.terminal_reason_code or "reservation_aborted"
            disposition = "terminal"
        else:
            raise RuntimeError("reserve refusal requires a terminal reservation")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "operation": "reserve",
            "state": "refused",
            "prefill_process": record.attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                record.attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": record.attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(record.attempt.logical_request_chain_id),
            "reservation_attempt_id": str(record.attempt.reservation_attempt_id),
            "reserve_attempt_digest": record.attempt.reserve_attempt_digest.hex(),
            "source_tp_size": record.attempt.source_tp_size,
            "prepared_ttl_ms": record.attempt.prepared_ttl_ms,
            "inference_route": record.attempt.inference_route,
            "request_shape": record.attempt.request_shape,
            "reason_code": reason_code,
            "diagnostic": record.terminal_diagnostic,
            "disposition": disposition,
            "receipt_id": str(uuid.uuid4()),
            "take_once": True,
        }
        receipt["receipt_digest"] = _receipt_digest(receipt).hex()
        return receipt

    @staticmethod
    def _admission_refusal_receipt(
        attempt: DecodeReservationAttempt,
        refusal: DecodeReservationAdmissionRefused,
    ) -> dict[str, object]:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "operation": "reserve",
            "state": "refused",
            "prefill_process": attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(attempt.logical_request_chain_id),
            "reservation_attempt_id": str(attempt.reservation_attempt_id),
            "reserve_attempt_digest": attempt.reserve_attempt_digest.hex(),
            "source_tp_size": attempt.source_tp_size,
            "prepared_ttl_ms": attempt.prepared_ttl_ms,
            "inference_route": attempt.inference_route,
            "request_shape": attempt.request_shape,
            "reason_code": refusal.reason_code,
            "diagnostic": refusal.diagnostic,
            "disposition": refusal.disposition.value,
            "receipt_id": str(uuid.uuid4()),
            "take_once": True,
        }
        receipt["receipt_digest"] = _receipt_digest(receipt).hex()
        return receipt

    @staticmethod
    def _copy_receipt(
        receipt: Mapping[str, object],
    ) -> dict[str, object]:
        return copy.deepcopy(dict(receipt))

    @staticmethod
    def _unbound_cancellation_receipt(
        record: _DecodeReservationRecord,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "grant_id": str(record.grant_id),
            "reservation_attempt_id": str(record.attempt.reservation_attempt_id),
            "reserve_attempt_digest": record.attempt.reserve_attempt_digest.hex(),
            "prefill_process": record.attempt.prefill_process.to_dict(),
            "prefill_bootstrap_endpoint": (
                record.attempt.prefill_bootstrap_endpoint.to_dict()
            ),
            "decoder_process": record.attempt.decoder_process.to_dict(),
            "logical_request_chain_id": str(record.attempt.logical_request_chain_id),
            "source_tp_size": record.attempt.source_tp_size,
            "inference_route": record.attempt.inference_route,
            "request_shape": record.attempt.request_shape,
            "prepared_ttl_ms": record.attempt.prepared_ttl_ms,
            "prepared_expires_at_unix_ms": (record.prepared_expires_at_unix_ms),
            "child_request_ids": [
                str(value) for value in record.attempt.child_request_ids
            ],
            "decoder_slot_generations": [
                str(value.decoder_slot_generation) for value in record.allocations
            ],
            "bootstrap_rooms": [value.bootstrap_room for value in record.allocations],
            "reservation_digest": record.reservation_digest.hex(),
            "attempted_grant_digest": request["attempted_grant_digest"],
            "operation": "cancel",
            "state": "cancelled",
            "receipt_id": str(uuid.uuid4()),
            "take_once": True,
        }
        receipt["receipt_digest"] = _receipt_digest(receipt).hex()
        return receipt


def authenticate_process_bearer(
    authorization_header: str | None,
    api_key: str | None,
) -> None:
    """Authenticate the reserve route with the process static API key.

    :param authorization_header: Candidate Authorization header.
    :param api_key: Configured decoder process API key.
    :raises DecodeReservationAuthenticationError: If authentication fails.
    """

    if api_key is None or len(api_key) == 0:
        raise DecodeReservationAuthenticationError(
            "decoder reservation service requires a process API key"
        )
    token = _extract_bearer_token(authorization_header)
    if token is None or not secrets.compare_digest(token, api_key):
        raise DecodeReservationAuthenticationError("invalid decoder process bearer")


def _failure_context(
    fields: Mapping[str, object],
    operation: DecodeReservationOperation,
) -> tuple[str | None, str | None]:
    if operation not in (
        DecodeReservationOperation.ABORT,
        DecodeReservationOperation.QUARANTINE,
    ):
        return None, None
    return _validate_failure_context_values(
        fields["reason_code"],
        fields["diagnostic"],
    )


def _validate_failure_context_values(
    reason_code_value: object,
    diagnostic_value: object,
) -> tuple[str, str | None]:
    reason_code = _require_nonempty_string(reason_code_value, "reason_code")
    if len(reason_code.encode()) > _FAILURE_REASON_MAX_BYTES:
        raise DecodeReservationValidationError("reason_code is too long")
    if any(character not in _FAILURE_REASON_CHARACTERS for character in reason_code):
        raise DecodeReservationValidationError(
            "reason_code contains an invalid character"
        )
    if diagnostic_value is None:
        return reason_code, None
    diagnostic = _require_nonempty_string(diagnostic_value, "diagnostic")
    if len(diagnostic.encode()) > _FAILURE_DIAGNOSTIC_MAX_BYTES:
        raise DecodeReservationValidationError("diagnostic is too long")
    if any(unicodedata.category(character) == "Cc" for character in diagnostic):
        raise DecodeReservationValidationError(
            "diagnostic cannot contain control characters"
        )
    return reason_code, diagnostic


def _compute_reservation_digest(
    grant_id: uuid.UUID,
    reserve_attempt_digest: bytes,
    prepared_expires_at_unix_ms: int,
    allocations: tuple[DecodeReservationAllocation, ...],
) -> bytes:
    hasher = blake3.blake3()
    hasher.update(RESERVATION_DIGEST_DOMAIN)
    hasher.update(grant_id.bytes)
    hasher.update(reserve_attempt_digest)
    hasher.update(prepared_expires_at_unix_ms.to_bytes(8, "little"))
    hasher.update(len(allocations).to_bytes(8, "little"))
    for index, allocation in enumerate(allocations):
        hasher.update(index.to_bytes(8, "little"))
        hasher.update(allocation.child_request_id.bytes)
        hasher.update(allocation.decoder_slot_generation.bytes)
        hasher.update(allocation.bootstrap_room.to_bytes(8, "little"))
        hasher.update(allocation.request_slot.to_bytes(8, "little"))
        hasher.update(allocation.request_generation.to_bytes(8, "little"))
        hasher.update(allocation.writer_manifest_digest)
        hasher.update(allocation.allocation_digest)
        hasher.update(allocation.reserved_kv_tokens.to_bytes(8, "little"))
        hasher.update(allocation.remaining_decode_tokens.to_bytes(8, "little"))
    return hasher.digest()


def _compute_grant_digest(
    reservation_digest: bytes,
    request_body: bytes,
) -> bytes:
    hasher = blake3.blake3()
    hasher.update(GRANT_DIGEST_DOMAIN)
    hasher.update(reservation_digest)
    _hash_bytes(hasher, request_body)
    return hasher.digest()


def _receipt_digest(receipt: Mapping[str, object]) -> bytes:
    encoded = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(RECEIPT_DIGEST_DOMAIN + encoded).digest()


def _hash_process(
    hasher: blake3.blake3,
    process: DecodeReservationProcess,
) -> None:
    _hash_bytes(hasher, process.url.encode())
    hasher.update(process.instance_id.bytes)


def _hash_bytes(hasher: blake3.blake3, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    if authorization_header is None:
        return None
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    if len(parts[1]) == 0:
        return None
    return parts[1]


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise DecodeReservationValidationError(f"{name} must be a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    if set(value) != set(expected):
        raise DecodeReservationValidationError(f"{name} contains invalid fields")


def _require_schema_version(value: object) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise DecodeReservationValidationError(
            f"schema_version must be {SCHEMA_VERSION}"
        )


def _require_nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or len(value) == 0:
        raise DecodeReservationValidationError(f"{name} must be a nonempty string")
    return value


def _require_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise DecodeReservationValidationError(f"{name} must be an integer")
    return value


def _require_uuid(value: object, name: str) -> uuid.UUID:
    text = _require_nonempty_string(value, name)
    try:
        parsed = uuid.UUID(text)
    except ValueError as error:
        raise DecodeReservationValidationError(
            f"{name} must be a canonical UUID"
        ) from error
    if parsed.int == 0 or str(parsed) != text:
        raise DecodeReservationValidationError(
            f"{name} must be a canonical non-nil UUID"
        )
    return parsed


def _require_digest(value: object, name: str) -> bytes:
    text = _require_nonempty_string(value, name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise DecodeReservationValidationError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return bytes.fromhex(text)
