import dataclasses
import hashlib
import hmac
import json
import re
import uuid

from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)

TERMINAL_STARTUP_ENROLLMENT_ACK_SCHEMA: str = (
    "packed-terminal-startup-enrollment-ack-v2"
)
TERMINAL_STARTUP_ENROLLMENT_ACK_TAG: bytes = b"TERMINAL_STARTUP_ENROLLMENT_ACK"
TERMINAL_STARTUP_ENROLLMENT_ACK_MAX_BYTES: int = 4096

_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}")
_NIXL_AGENT_NAME_MAX_BYTES = 256
_SHA256_BYTES = hashlib.sha256().digest_size


class TerminalStartupEnrollmentAckError(RuntimeError):
    """Invalid terminal startup enrollment acknowledgement."""


def _require_sha256(value: bytes, label: str) -> None:
    """Require one binary SHA-256 digest.

    :param value: Candidate digest.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != _SHA256_BYTES:
        raise ValueError(f"{label} must contain 32 bytes")


def _require_uuid_bytes(value: bytes, label: str) -> None:
    """Require one canonical non-nil UUID encoded as bytes.

    :param value: Candidate UUID bytes.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != 16:
        raise ValueError(f"{label} must contain 16 bytes")
    if uuid.UUID(bytes=value).int == 0:
        raise ValueError(f"{label} must be non-nil")


def _require_service_id(value: str, label: str) -> None:
    """Require one launcher-canonical service identifier.

    :param value: Candidate service identifier.
    :param label: Reader-facing field name.
    """

    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is not canonical")


def _require_agent_name(value: str) -> None:
    """Require one bounded ASCII NIXL agent name.

    :param value: Candidate native agent name.
    """

    if type(value) is not str or len(value) == 0:
        raise ValueError("source_nixl_agent_name must be nonempty")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("source_nixl_agent_name must be ASCII") from error
    if len(encoded) > _NIXL_AGENT_NAME_MAX_BYTES:
        raise ValueError("source_nixl_agent_name is too large")


def terminal_decoder_registration_multipart_sha256(
    frames: tuple[bytes, ...],
) -> bytes:
    """Digest the exact ordered decoder registration multipart message.

    Each frame is preceded by its unsigned eight-byte network-order length, so
    frame boundaries and empty frames are part of the authenticated value.

    :param frames: Complete immutable registration multipart message.
    :returns: SHA-256 of the exact length-delimited frames.
    """

    if type(frames) is not tuple or len(frames) == 0:
        raise ValueError("decoder registration frames must be a nonempty tuple")
    if any(type(frame) is not bytes for frame in frames):
        raise TypeError("decoder registration frames must contain bytes")

    digest = hashlib.sha256()
    for frame in frames:
        digest.update(len(frame).to_bytes(8, byteorder="big", signed=False))
        digest.update(frame)
    return digest.digest()


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupEnrollmentAck:
    """Source proof that one exact decoder registration is retained.

    :ivar startup_matrix_sha256: Digest of the sealed startup matrix.
    :ivar source_service_id: Source service that retained the registration.
    :ivar source_tensor_parallel_rank: Source rank within its service.
    :ivar source_process_generation: Exact source process incarnation.
    :ivar source_nixl_agent_name: Exact source native agent identity.
    :ivar source_nixl_agent_metadata_sha256: Digest of source agent metadata.
    :ivar target_decoder_service_id: Decoder service being acknowledged.
    :ivar target_decoder_tensor_parallel_rank: Decoder rank within its service.
    :ivar target_decoder_process_generation: Exact decoder process incarnation.
    :ivar decoder_registration_multipart_sha256: Digest of the complete decoder
        registration multipart message retained by the source.
    :ivar decoder_control_route_table_sha256: Digest of the complete immutable
        same-service decoder listener table retained by the source.
    """

    startup_matrix_sha256: bytes
    source_service_id: str
    source_tensor_parallel_rank: int
    source_process_generation: bytes
    source_nixl_agent_name: str
    source_nixl_agent_metadata_sha256: bytes
    target_decoder_service_id: str
    target_decoder_tensor_parallel_rank: int
    target_decoder_process_generation: bytes
    decoder_registration_multipart_sha256: bytes
    decoder_control_route_table_sha256: bytes

    def __post_init__(self) -> None:
        """Validate one structurally complete acknowledgement."""

        _require_sha256(self.startup_matrix_sha256, "startup_matrix_sha256")
        _require_service_id(self.source_service_id, "source_service_id")
        if (
            type(self.source_tensor_parallel_rank) is not int
            or self.source_tensor_parallel_rank < 0
        ):
            raise ValueError("source_tensor_parallel_rank must be nonnegative")
        _require_uuid_bytes(
            self.source_process_generation,
            "source_process_generation",
        )
        _require_agent_name(self.source_nixl_agent_name)
        _require_sha256(
            self.source_nixl_agent_metadata_sha256,
            "source_nixl_agent_metadata_sha256",
        )
        _require_service_id(
            self.target_decoder_service_id,
            "target_decoder_service_id",
        )
        if (
            type(self.target_decoder_tensor_parallel_rank) is not int
            or self.target_decoder_tensor_parallel_rank < 0
        ):
            raise ValueError("target_decoder_tensor_parallel_rank must be nonnegative")
        _require_uuid_bytes(
            self.target_decoder_process_generation,
            "target_decoder_process_generation",
        )
        _require_sha256(
            self.decoder_registration_multipart_sha256,
            "decoder_registration_multipart_sha256",
        )
        _require_sha256(
            self.decoder_control_route_table_sha256,
            "decoder_control_route_table_sha256",
        )

    @property
    def digest(self) -> bytes:
        """Return the canonical acknowledgement digest.

        :returns: SHA-256 of the exact schema-bearing wire bytes.
        """

        return hashlib.sha256(encode_terminal_startup_enrollment_ack(self)).digest()

    def require_matrix(self, matrix: TerminalStartupCohortMatrix) -> None:
        """Authenticate both endpoint identities against one sealed matrix.

        :param matrix: Complete generation-authenticated startup matrix.
        :raises TerminalStartupEnrollmentAckError: If an identity field differs.
        """

        if type(matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("matrix must be TerminalStartupCohortMatrix")
        if not hmac.compare_digest(self.startup_matrix_sha256, matrix.digest):
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement belongs to another startup matrix"
            )
        try:
            source = matrix.rank(
                self.source_service_id,
                self.source_tensor_parallel_rank,
            )
            decoder = matrix.rank(
                self.target_decoder_service_id,
                self.target_decoder_tensor_parallel_rank,
            )
        except TerminalStartupCohortError as error:
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement rank is absent from the startup matrix"
            ) from error

        if source.role is not TerminalOwnerRole.SOURCE:
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement issuer is not a source rank"
            )
        if (
            source.process_generation != self.source_process_generation
            or source.nixl_agent_name != self.source_nixl_agent_name
            or not hmac.compare_digest(
                source.nixl_agent_metadata_sha256,
                self.source_nixl_agent_metadata_sha256,
            )
        ):
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement source differs from the startup matrix"
            )
        if decoder.role is not TerminalOwnerRole.DECODE:
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement target is not a decoder rank"
            )
        if decoder.process_generation != self.target_decoder_process_generation:
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement decoder differs from the startup matrix"
            )

    def require_decoder_registration(
        self,
        frames: tuple[bytes, ...],
    ) -> None:
        """Authenticate the exact decoder registration multipart message.

        :param frames: Complete decoder registration sent during enrollment.
        :raises TerminalStartupEnrollmentAckError: If the registration differs.
        """

        actual = terminal_decoder_registration_multipart_sha256(frames)
        if not hmac.compare_digest(
            self.decoder_registration_multipart_sha256,
            actual,
        ):
            raise TerminalStartupEnrollmentAckError(
                "enrollment acknowledgement binds another decoder registration"
            )


def build_terminal_startup_enrollment_ack(
    matrix: TerminalStartupCohortMatrix,
    source: TerminalStartupRankAdvertisement,
    target_decoder: TerminalStartupRankAdvertisement,
    decoder_registration_frames: tuple[bytes, ...],
    decoder_control_route_table_sha256: bytes,
) -> TerminalStartupEnrollmentAck:
    """Build one matrix-authenticated source enrollment acknowledgement.

    :param matrix: Complete generation-authenticated startup matrix.
    :param source: Exact source rank retaining the registration.
    :param target_decoder: Exact decoder rank that issued the registration.
    :param decoder_registration_frames: Complete retained registration message.
    :param decoder_control_route_table_sha256: Digest of the frozen listener
        table for the target decoder service.
    :returns: Immutable acknowledgement bound to all three authorities.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    if type(source) is not TerminalStartupRankAdvertisement:
        raise TypeError("source must be TerminalStartupRankAdvertisement")
    if type(target_decoder) is not TerminalStartupRankAdvertisement:
        raise TypeError("target_decoder must be TerminalStartupRankAdvertisement")
    _require_sha256(
        decoder_control_route_table_sha256,
        "decoder_control_route_table_sha256",
    )
    try:
        matrix_source = matrix.rank(*source.key)
        matrix_decoder = matrix.rank(*target_decoder.key)
    except TerminalStartupCohortError as error:
        raise TerminalStartupEnrollmentAckError(
            "enrollment endpoint is absent from the startup matrix"
        ) from error
    if matrix_source != source or source.role is not TerminalOwnerRole.SOURCE:
        raise TerminalStartupEnrollmentAckError(
            "enrollment acknowledgement source is not the exact matrix source"
        )
    if (
        matrix_decoder != target_decoder
        or target_decoder.role is not TerminalOwnerRole.DECODE
    ):
        raise TerminalStartupEnrollmentAckError(
            "enrollment acknowledgement target is not the exact matrix decoder"
        )

    acknowledgement = TerminalStartupEnrollmentAck(
        startup_matrix_sha256=matrix.digest,
        source_service_id=source.service_id,
        source_tensor_parallel_rank=source.tensor_parallel_rank,
        source_process_generation=source.process_generation,
        source_nixl_agent_name=source.nixl_agent_name,
        source_nixl_agent_metadata_sha256=source.nixl_agent_metadata_sha256,
        target_decoder_service_id=target_decoder.service_id,
        target_decoder_tensor_parallel_rank=target_decoder.tensor_parallel_rank,
        target_decoder_process_generation=target_decoder.process_generation,
        decoder_registration_multipart_sha256=(
            terminal_decoder_registration_multipart_sha256(decoder_registration_frames)
        ),
        decoder_control_route_table_sha256=decoder_control_route_table_sha256,
    )
    acknowledgement.require_matrix(matrix)
    acknowledgement.require_decoder_registration(decoder_registration_frames)
    return acknowledgement


def _require_exact_fields(
    payload: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    """Reject unknown or missing JSON fields.

    :param payload: Parsed JSON object.
    :param expected: Exact accepted field set.
    :param label: Reader-facing object label.
    """

    if set(payload) != expected:
        raise TerminalStartupEnrollmentAckError(f"{label} field set is invalid")


def _parse_sha256(value: object, label: str) -> bytes:
    """Parse one canonical lowercase SHA-256 string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: Binary SHA-256 value.
    """

    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise TerminalStartupEnrollmentAckError(f"{label} is not canonical")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise TerminalStartupEnrollmentAckError(
            f"{label} is not hexadecimal"
        ) from error
    if len(parsed) != _SHA256_BYTES:
        raise TerminalStartupEnrollmentAckError(f"{label} is not SHA-256")
    return parsed


def _parse_uuid(value: object, label: str) -> bytes:
    """Parse one canonical non-nil UUID string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: UUID bytes.
    """

    if type(value) is not str:
        raise TerminalStartupEnrollmentAckError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise TerminalStartupEnrollmentAckError(f"{label} is not a UUID") from error
    if parsed.int == 0 or str(parsed) != value:
        raise TerminalStartupEnrollmentAckError(f"{label} is not canonical")
    return parsed.bytes


def _parse_rank(value: object, label: str) -> tuple[str, int, bytes]:
    """Parse one strict service-rank-generation endpoint.

    :param value: Candidate nested JSON object.
    :param label: Reader-facing endpoint label.
    :returns: Service identifier, TP rank, and process generation.
    """

    if type(value) is not dict:
        raise TerminalStartupEnrollmentAckError(f"{label} must be an object")
    _require_exact_fields(
        value,
        {"service_id", "tensor_parallel_rank", "process_generation"},
        label,
    )
    service_id = value["service_id"]
    tensor_parallel_rank = value["tensor_parallel_rank"]
    if type(service_id) is not str or type(tensor_parallel_rank) is not int:
        raise TerminalStartupEnrollmentAckError(f"{label} fields are malformed")
    return (
        service_id,
        tensor_parallel_rank,
        _parse_uuid(value["process_generation"], f"{label}.process_generation"),
    )


def _load_json(payload: bytes) -> dict[str, object]:
    """Decode one bounded duplicate-free JSON object.

    :param payload: Exact candidate wire bytes.
    :returns: Parsed top-level object.
    """

    if (
        type(payload) is not bytes
        or not 0 < len(payload) <= TERMINAL_STARTUP_ENROLLMENT_ACK_MAX_BYTES
    ):
        raise TerminalStartupEnrollmentAckError(
            "startup enrollment acknowledgement payload size is invalid"
        )

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Build one object while rejecting duplicate field names.

        :param pairs: Decoder-preserved JSON field pairs.
        :returns: Unique-key JSON object.
        """

        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalStartupEnrollmentAckError(
                    f"duplicate startup enrollment acknowledgement field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        """Reject non-finite JSON constants.

        :param value: Non-finite JSON spelling.
        :raises TerminalStartupEnrollmentAckError: For every input.
        """

        raise TerminalStartupEnrollmentAckError(
            f"non-finite startup enrollment acknowledgement value: {value}"
        )

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalStartupEnrollmentAckError(
            "startup enrollment acknowledgement is not valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise TerminalStartupEnrollmentAckError(
            "startup enrollment acknowledgement must be an object"
        )
    return decoded


def encode_terminal_startup_enrollment_ack(
    acknowledgement: TerminalStartupEnrollmentAck,
) -> bytes:
    """Encode one canonical terminal startup enrollment acknowledgement.

    :param acknowledgement: Exact matrix-bound source acknowledgement.
    :returns: Compact UTF-8 JSON without a newline.
    """

    if type(acknowledgement) is not TerminalStartupEnrollmentAck:
        raise TypeError("acknowledgement must be TerminalStartupEnrollmentAck")
    payload = {
        "schema": TERMINAL_STARTUP_ENROLLMENT_ACK_SCHEMA,
        "startup_matrix_sha256": acknowledgement.startup_matrix_sha256.hex(),
        "source": {
            "service_id": acknowledgement.source_service_id,
            "tensor_parallel_rank": acknowledgement.source_tensor_parallel_rank,
            "process_generation": str(
                uuid.UUID(bytes=acknowledgement.source_process_generation)
            ),
            "nixl_agent_name": acknowledgement.source_nixl_agent_name,
            "nixl_agent_metadata_sha256": (
                acknowledgement.source_nixl_agent_metadata_sha256.hex()
            ),
        },
        "target_decoder": {
            "service_id": acknowledgement.target_decoder_service_id,
            "tensor_parallel_rank": (
                acknowledgement.target_decoder_tensor_parallel_rank
            ),
            "process_generation": str(
                uuid.UUID(bytes=acknowledgement.target_decoder_process_generation)
            ),
        },
        "decoder_registration_multipart_sha256": (
            acknowledgement.decoder_registration_multipart_sha256.hex()
        ),
        "decoder_control_route_table_sha256": (
            acknowledgement.decoder_control_route_table_sha256.hex()
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > TERMINAL_STARTUP_ENROLLMENT_ACK_MAX_BYTES:
        raise ValueError("startup enrollment acknowledgement is too large")
    return encoded


def decode_terminal_startup_enrollment_ack(
    payload: bytes,
) -> TerminalStartupEnrollmentAck:
    """Decode one strict terminal startup enrollment acknowledgement.

    :param payload: Exact canonical wire bytes.
    :returns: Immutable matrix-bound source acknowledgement.
    :raises TerminalStartupEnrollmentAckError: If bytes are malformed.
    """

    decoded = _load_json(payload)
    _require_exact_fields(
        decoded,
        {
            "schema",
            "startup_matrix_sha256",
            "source",
            "target_decoder",
            "decoder_registration_multipart_sha256",
            "decoder_control_route_table_sha256",
        },
        "startup enrollment acknowledgement",
    )
    if decoded["schema"] != TERMINAL_STARTUP_ENROLLMENT_ACK_SCHEMA:
        raise TerminalStartupEnrollmentAckError(
            "unsupported startup enrollment acknowledgement schema"
        )

    raw_source = decoded["source"]
    if type(raw_source) is not dict:
        raise TerminalStartupEnrollmentAckError("source must be an object")
    _require_exact_fields(
        raw_source,
        {
            "service_id",
            "tensor_parallel_rank",
            "process_generation",
            "nixl_agent_name",
            "nixl_agent_metadata_sha256",
        },
        "source",
    )
    source_service_id = raw_source["service_id"]
    source_tensor_parallel_rank = raw_source["tensor_parallel_rank"]
    source_nixl_agent_name = raw_source["nixl_agent_name"]
    if (
        type(source_service_id) is not str
        or type(source_tensor_parallel_rank) is not int
        or type(source_nixl_agent_name) is not str
    ):
        raise TerminalStartupEnrollmentAckError("source fields are malformed")

    (
        target_decoder_service_id,
        target_decoder_tensor_parallel_rank,
        target_decoder_process_generation,
    ) = _parse_rank(decoded["target_decoder"], "target_decoder")
    try:
        acknowledgement = TerminalStartupEnrollmentAck(
            startup_matrix_sha256=_parse_sha256(
                decoded["startup_matrix_sha256"],
                "startup_matrix_sha256",
            ),
            source_service_id=source_service_id,
            source_tensor_parallel_rank=source_tensor_parallel_rank,
            source_process_generation=_parse_uuid(
                raw_source["process_generation"],
                "source.process_generation",
            ),
            source_nixl_agent_name=source_nixl_agent_name,
            source_nixl_agent_metadata_sha256=_parse_sha256(
                raw_source["nixl_agent_metadata_sha256"],
                "source.nixl_agent_metadata_sha256",
            ),
            target_decoder_service_id=target_decoder_service_id,
            target_decoder_tensor_parallel_rank=(target_decoder_tensor_parallel_rank),
            target_decoder_process_generation=target_decoder_process_generation,
            decoder_registration_multipart_sha256=_parse_sha256(
                decoded["decoder_registration_multipart_sha256"],
                "decoder_registration_multipart_sha256",
            ),
            decoder_control_route_table_sha256=_parse_sha256(
                decoded["decoder_control_route_table_sha256"],
                "decoder_control_route_table_sha256",
            ),
        )
    except (TypeError, ValueError) as error:
        raise TerminalStartupEnrollmentAckError(
            "startup enrollment acknowledgement fields are invalid"
        ) from error
    if encode_terminal_startup_enrollment_ack(acknowledgement) != payload:
        raise TerminalStartupEnrollmentAckError(
            "startup enrollment acknowledgement JSON is not canonical"
        )
    return acknowledgement
