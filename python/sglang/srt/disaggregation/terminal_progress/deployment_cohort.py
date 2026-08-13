import dataclasses
import enum
import hashlib
import json
import os
import re
import secrets
import stat
import uuid
from pathlib import Path

TERMINAL_DEPLOYMENT_COHORT_SCHEMA: str = "packed-terminal-deployment-cohort-v1"
MAX_TERMINAL_DEPLOYMENT_COHORT_BYTES: int = 256 * 1024
TERMINAL_DEPLOYMENT_COHORT_DIGEST_BYTES: int = hashlib.sha256().digest_size

_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}")
_BOOTSTRAP_HOST = re.compile(r"[A-Za-z0-9._:\[\]-]+")
_LOOPBACK_ORIGIN = re.compile(r"http://127\.0\.0\.1:(?P<port>[1-9][0-9]{0,4})")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_PREFILL_TP_SIZES = frozenset((1, 2, 4, 8))
_DECODE_TP_SIZES = frozenset((1, 2))


class TerminalDeploymentCohortError(ValueError):
    """Invalid immutable packed-terminal deployment cohort."""


class TerminalDeploymentRole(enum.StrEnum):
    """Role of one model service in a terminal deployment cohort."""

    PREFILL = "prefill"
    DECODE = "decode"


def _require_uuid(value: uuid.UUID, label: str) -> None:
    """Require one canonical non-nil UUID.

    :param value: Candidate UUID.
    :param label: Reader-facing field label.
    """

    if type(value) is not uuid.UUID:
        raise TypeError(f"{label} must be UUID")
    if value.int == 0:
        raise ValueError(f"{label} must be non-nil")


def _require_service_identity(
    *,
    service_id: str,
    launch_instance_id: uuid.UUID,
    origin: str,
    tensor_parallel_size: int,
    allowed_tp_sizes: frozenset[int],
    role: str,
) -> None:
    """Validate fields common to one immutable model service.

    :param service_id: Stable launcher service name.
    :param launch_instance_id: Launcher-assigned service incarnation.
    :param origin: Exact loopback HTTP origin.
    :param tensor_parallel_size: Complete service TP width.
    :param allowed_tp_sizes: Role-specific admitted TP widths.
    :param role: Reader-facing role label.
    """

    if type(service_id) is not str or _IDENTIFIER.fullmatch(service_id) is None:
        raise ValueError(f"{role} service_id is not canonical")
    _require_uuid(launch_instance_id, f"{role} launch_instance_id")
    if type(origin) is not str or _LOOPBACK_ORIGIN.fullmatch(origin) is None:
        raise ValueError(f"{role} origin must be a canonical loopback origin")
    if (
        type(tensor_parallel_size) is not int
        or tensor_parallel_size not in allowed_tp_sizes
    ):
        raise ValueError(f"{role} tensor_parallel_size is unsupported")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentBootstrapEndpoint:
    """Exact bootstrap route published by one prefill service.

    :ivar host: Canonical bootstrap listener host.
    :ivar port: Bootstrap listener TCP port.
    """

    host: str
    port: int

    def __post_init__(self) -> None:
        """Validate one immutable bootstrap endpoint."""

        if type(self.host) is not str or _BOOTSTRAP_HOST.fullmatch(self.host) is None:
            raise ValueError("bootstrap host is not canonical")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("bootstrap port must be in the TCP port range")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentPrefill:
    """One launcher-authorized prefill service incarnation.

    :ivar service_id: Stable service name within one deployment run.
    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar origin: Exact loopback HTTP origin used by the launcher.
    :ivar bootstrap_endpoint: Exact prefill rank-discovery endpoint.
    :ivar tensor_parallel_size: Complete prefill TP width.
    """

    service_id: str
    launch_instance_id: uuid.UUID
    origin: str
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate one exact static prefill identity."""

        _require_service_identity(
            service_id=self.service_id,
            launch_instance_id=self.launch_instance_id,
            origin=self.origin,
            tensor_parallel_size=self.tensor_parallel_size,
            allowed_tp_sizes=_PREFILL_TP_SIZES,
            role="prefill",
        )
        if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
            raise TypeError(
                "bootstrap_endpoint must be TerminalDeploymentBootstrapEndpoint"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentDecoder:
    """One launcher-authorized decode service incarnation.

    :ivar service_id: Stable service name within one deployment run.
    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar origin: Exact loopback HTTP origin used by the launcher.
    :ivar tensor_parallel_size: Complete decode TP width.
    """

    service_id: str
    launch_instance_id: uuid.UUID
    origin: str
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate one exact static decode identity."""

        _require_service_identity(
            service_id=self.service_id,
            launch_instance_id=self.launch_instance_id,
            origin=self.origin,
            tensor_parallel_size=self.tensor_parallel_size,
            allowed_tp_sizes=_DECODE_TP_SIZES,
            role="decode",
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentLocalService:
    """One exact local member selected from a hash-bound cohort.

    :ivar group_id: Owning static prefill group.
    :ivar cohort_digest: Exact launcher-authenticated group epoch.
    :ivar role: Local prefill or decode role.
    :ivar service_id: Stable launcher service name.
    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar origin: Exact loopback HTTP origin.
    :ivar tensor_parallel_size: Complete local service TP width.
    :ivar bootstrap_endpoint: Prefill endpoint, absent for decode.
    """

    group_id: str
    cohort_digest: bytes
    role: TerminalDeploymentRole
    service_id: str
    launch_instance_id: uuid.UUID
    origin: str
    tensor_parallel_size: int
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint | None

    def __post_init__(self) -> None:
        """Validate one selected local membership."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        if (
            type(self.cohort_digest) is not bytes
            or len(self.cohort_digest) != TERMINAL_DEPLOYMENT_COHORT_DIGEST_BYTES
        ):
            raise ValueError("cohort_digest must contain one SHA-256 digest")
        if type(self.role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        allowed_tp_sizes = (
            _PREFILL_TP_SIZES
            if self.role is TerminalDeploymentRole.PREFILL
            else _DECODE_TP_SIZES
        )
        _require_service_identity(
            service_id=self.service_id,
            launch_instance_id=self.launch_instance_id,
            origin=self.origin,
            tensor_parallel_size=self.tensor_parallel_size,
            allowed_tp_sizes=allowed_tp_sizes,
            role=self.role.value,
        )
        if self.role is TerminalDeploymentRole.PREFILL:
            if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
                raise ValueError("prefill membership requires a bootstrap endpoint")
            return
        if self.bootstrap_endpoint is not None:
            raise ValueError("decode membership cannot carry a bootstrap endpoint")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentRequestPlan:
    """One request-selected prefill/decode subset of a frozen cohort.

    :ivar cohort_digest: Exact static deployment epoch.
    :ivar group_id: Selected ownership group.
    :ivar prefill_service_id: Selected prefill service name.
    :ivar prefill_launch_instance_id: Selected prefill launch incarnation.
    :ivar decoder_service_id: Selected decoder service name.
    :ivar decoder_launch_instance_id: Selected decoder launch incarnation.
    """

    cohort_digest: bytes
    group_id: str
    prefill_service_id: str
    prefill_launch_instance_id: uuid.UUID
    decoder_service_id: str
    decoder_launch_instance_id: uuid.UUID

    def __post_init__(self) -> None:
        """Validate the shape of one request selection."""

        if (
            type(self.cohort_digest) is not bytes
            or len(self.cohort_digest) != TERMINAL_DEPLOYMENT_COHORT_DIGEST_BYTES
        ):
            raise ValueError("cohort_digest must contain one SHA-256 digest")
        identifiers = (self.group_id, self.prefill_service_id, self.decoder_service_id)
        if any(
            type(value) is not str or _IDENTIFIER.fullmatch(value) is None
            for value in identifiers
        ):
            raise ValueError("request-plan identifiers are not canonical")
        _require_uuid(self.prefill_launch_instance_id, "prefill_launch_instance_id")
        _require_uuid(self.decoder_launch_instance_id, "decoder_launch_instance_id")
        if self.prefill_service_id == self.decoder_service_id:
            raise ValueError("prefill and decoder service IDs must differ")
        if self.prefill_launch_instance_id == self.decoder_launch_instance_id:
            raise ValueError("prefill and decoder launch identities must differ")

    def contains(self, service: TerminalDeploymentLocalService) -> bool:
        """Return whether this plan contains one exact local member.

        :param service: Cohort-validated local service membership.
        :returns: Whether the member is one of this plan's two services.
        """

        if type(service) is not TerminalDeploymentLocalService:
            raise TypeError("service must be TerminalDeploymentLocalService")
        if service.group_id != self.group_id or not secrets.compare_digest(
            service.cohort_digest,
            self.cohort_digest,
        ):
            return False
        if service.role is TerminalDeploymentRole.PREFILL:
            return (
                service.service_id == self.prefill_service_id
                and service.launch_instance_id == self.prefill_launch_instance_id
            )
        return (
            service.service_id == self.decoder_service_id
            and service.launch_instance_id == self.decoder_launch_instance_id
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentCohort:
    """Complete immutable membership of one isolated prefill group.

    :ivar group_id: Stable topology-local ownership identifier.
    :ivar model_fingerprint: Exact model-weights compatibility fingerprint.
    :ivar logical_kv_layout_fingerprint: Exact TP-independent KV layout.
    :ivar prefill: Sole prefill service and bootstrap identity.
    :ivar decoders: Canonically ordered nonempty decoder membership.
    """

    group_id: str
    model_fingerprint: str
    logical_kv_layout_fingerprint: str
    prefill: TerminalDeploymentPrefill
    decoders: tuple[TerminalDeploymentDecoder, ...]

    def __post_init__(self) -> None:
        """Validate complete, ordered, collision-free group membership."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        fingerprints = (
            ("model_fingerprint", self.model_fingerprint),
            (
                "logical_kv_layout_fingerprint",
                self.logical_kv_layout_fingerprint,
            ),
        )
        for label, value in fingerprints:
            if type(value) is not str or _FINGERPRINT.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if type(self.prefill) is not TerminalDeploymentPrefill:
            raise TypeError("prefill must be TerminalDeploymentPrefill")
        if type(self.decoders) is not tuple or len(self.decoders) == 0:
            raise ValueError("decoders must be a nonempty tuple")
        if any(
            type(decoder) is not TerminalDeploymentDecoder for decoder in self.decoders
        ):
            raise TypeError("decoders must contain TerminalDeploymentDecoder")
        decoder_ids = tuple(decoder.service_id for decoder in self.decoders)
        if tuple(sorted(decoder_ids)) != decoder_ids:
            raise ValueError("decoders must use canonical service_id order")
        service_ids = (self.prefill.service_id, *decoder_ids)
        launch_ids = (
            self.prefill.launch_instance_id,
            *(decoder.launch_instance_id for decoder in self.decoders),
        )
        origins = (
            self.prefill.origin,
            *(decoder.origin for decoder in self.decoders),
        )
        unique_fields = (
            ("service_id", service_ids),
            ("launch_instance_id", launch_ids),
            ("origin", origins),
        )
        for label, values in unique_fields:
            if len(set(values)) != len(values):
                raise ValueError(f"service {label} values must be unique")

    @property
    def digest(self) -> bytes:
        """Return the canonical cohort SHA-256.

        :returns: Digest of the exact canonical cohort encoding.
        """

        return hashlib.sha256(encode_terminal_deployment_cohort(self)).digest()

    def require_local_service(
        self,
        *,
        service_id: str,
        role: TerminalDeploymentRole,
        launch_instance_id: uuid.UUID,
        tensor_parallel_size: int,
        origin: str,
        bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint | None,
    ) -> TerminalDeploymentLocalService:
        """Select one exact local service or reject configuration drift.

        :param service_id: Launcher-selected local service name.
        :param role: Expected local process role.
        :param launch_instance_id: Local launcher-assigned incarnation.
        :param tensor_parallel_size: Local runtime TP width.
        :param origin: Local HTTP origin.
        :param bootstrap_endpoint: Local prefill endpoint, absent for decode.
        :returns: Exact cohort membership for the local service.
        :raises TerminalDeploymentCohortError: If no exact member exists.
        """

        if type(role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        if role is TerminalDeploymentRole.PREFILL:
            expected_id = self.prefill.service_id
            expected_launch_id = self.prefill.launch_instance_id
            expected_tp_size = self.prefill.tensor_parallel_size
            expected_origin = self.prefill.origin
            expected_bootstrap = self.prefill.bootstrap_endpoint
        else:
            matching = tuple(
                decoder for decoder in self.decoders if decoder.service_id == service_id
            )
            if len(matching) != 1:
                raise TerminalDeploymentCohortError(
                    "local decoder identity is absent from its cohort"
                )
            decoder = matching[0]
            expected_id = decoder.service_id
            expected_launch_id = decoder.launch_instance_id
            expected_tp_size = decoder.tensor_parallel_size
            expected_origin = decoder.origin
            expected_bootstrap = None
        if (
            service_id != expected_id
            or launch_instance_id != expected_launch_id
            or tensor_parallel_size != expected_tp_size
            or origin != expected_origin
            or bootstrap_endpoint != expected_bootstrap
        ):
            raise TerminalDeploymentCohortError(
                "local service configuration differs from cohort membership"
            )
        return TerminalDeploymentLocalService(
            group_id=self.group_id,
            cohort_digest=self.digest,
            role=role,
            service_id=expected_id,
            launch_instance_id=expected_launch_id,
            origin=expected_origin,
            tensor_parallel_size=expected_tp_size,
            bootstrap_endpoint=expected_bootstrap,
        )

    def require_request_plan(
        self,
        plan: TerminalDeploymentRequestPlan,
    ) -> tuple[TerminalDeploymentPrefill, TerminalDeploymentDecoder]:
        """Validate one request-selected subset against exact membership.

        :param plan: Request-scoped deployment selection.
        :returns: Exact prefill and decoder cohort records.
        :raises TerminalDeploymentCohortError: If any selected fact is stale.
        """

        if type(plan) is not TerminalDeploymentRequestPlan:
            raise TypeError("plan must be TerminalDeploymentRequestPlan")
        if not secrets.compare_digest(plan.cohort_digest, self.digest):
            raise TerminalDeploymentCohortError(
                "request plan uses a stale deployment cohort"
            )
        if plan.group_id != self.group_id:
            raise TerminalDeploymentCohortError(
                "request plan belongs to another deployment group"
            )
        if (
            plan.prefill_service_id != self.prefill.service_id
            or plan.prefill_launch_instance_id != self.prefill.launch_instance_id
        ):
            raise TerminalDeploymentCohortError(
                "request plan prefill differs from cohort membership"
            )
        matching = tuple(
            decoder
            for decoder in self.decoders
            if decoder.service_id == plan.decoder_service_id
        )
        if len(matching) != 1:
            raise TerminalDeploymentCohortError(
                "request plan decoder is absent from cohort membership"
            )
        decoder = matching[0]
        if decoder.launch_instance_id != plan.decoder_launch_instance_id:
            raise TerminalDeploymentCohortError(
                "request plan decoder launch identity is stale"
            )
        return self.prefill, decoder


def _cohort_payload(cohort: TerminalDeploymentCohort) -> dict[str, object]:
    """Project one validated cohort into canonical JSON field order.

    :param cohort: Complete immutable per-group deployment cohort.
    :returns: JSON-native canonical payload.
    """

    if type(cohort) is not TerminalDeploymentCohort:
        raise TypeError("cohort must be TerminalDeploymentCohort")
    return {
        "schema": TERMINAL_DEPLOYMENT_COHORT_SCHEMA,
        "group_id": cohort.group_id,
        "model_fingerprint": cohort.model_fingerprint,
        "logical_kv_layout_fingerprint": cohort.logical_kv_layout_fingerprint,
        "prefill": {
            "id": cohort.prefill.service_id,
            "launch_instance_id": str(cohort.prefill.launch_instance_id),
            "origin": cohort.prefill.origin,
            "bootstrap_endpoint": {
                "host": cohort.prefill.bootstrap_endpoint.host,
                "port": cohort.prefill.bootstrap_endpoint.port,
            },
            "tensor_parallel_size": cohort.prefill.tensor_parallel_size,
        },
        "decoders": [
            {
                "id": decoder.service_id,
                "launch_instance_id": str(decoder.launch_instance_id),
                "origin": decoder.origin,
                "tensor_parallel_size": decoder.tensor_parallel_size,
            }
            for decoder in cohort.decoders
        ],
    }


def encode_terminal_deployment_cohort(cohort: TerminalDeploymentCohort) -> bytes:
    """Encode one cohort into its sole accepted byte representation.

    :param cohort: Complete immutable deployment cohort.
    :returns: Compact canonical UTF-8 JSON without a trailing newline.
    """

    return json.dumps(
        _cohort_payload(cohort),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON fields while decoding.

    :param pairs: Ordered object fields emitted by :mod:`json`.
    :returns: Exact object mapping.
    :raises TerminalDeploymentCohortError: If a field occurs twice.
    """

    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TerminalDeploymentCohortError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    """Reject nonstandard JSON numeric constants.

    :param value: Parsed non-finite numeric token.
    :raises TerminalDeploymentCohortError: Always.
    """

    raise TerminalDeploymentCohortError(f"invalid JSON constant: {value}")


def _mapping(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    """Require one object with an exact field set.

    :param value: Candidate decoded JSON value.
    :param fields: Exact accepted field names.
    :param label: Reader-facing object label.
    :returns: Validated mapping.
    """

    if type(value) is not dict:
        raise TerminalDeploymentCohortError(f"{label} must be an object")
    mapping: dict[str, object] = value
    if frozenset(mapping) != fields:
        raise TerminalDeploymentCohortError(f"{label} has an invalid field set")
    return mapping


def _uuid_from_json(value: object, label: str) -> uuid.UUID:
    """Parse one canonical non-nil UUID string.

    :param value: Candidate JSON scalar.
    :param label: Reader-facing field label.
    :returns: Parsed UUID.
    """

    if type(value) is not str:
        raise TerminalDeploymentCohortError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise TerminalDeploymentCohortError(f"{label} is not a UUID") from error
    if parsed.int == 0 or str(parsed) != value:
        raise TerminalDeploymentCohortError(f"{label} is not canonical and non-nil")
    return parsed


def _string(value: object, label: str) -> str:
    """Require one exact JSON string.

    :param value: Candidate JSON scalar.
    :param label: Reader-facing field label.
    :returns: Exact string.
    """

    if type(value) is not str:
        raise TerminalDeploymentCohortError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    """Require one exact JSON integer.

    :param value: Candidate JSON scalar.
    :param label: Reader-facing field label.
    :returns: Exact integer.
    """

    if type(value) is not int:
        raise TerminalDeploymentCohortError(f"{label} must be an integer")
    return value


def _decode_bootstrap(value: object) -> TerminalDeploymentBootstrapEndpoint:
    """Decode one strict bootstrap endpoint.

    :param value: Candidate endpoint object.
    :returns: Validated endpoint.
    """

    fields = _mapping(value, frozenset(("host", "port")), "bootstrap_endpoint")
    return TerminalDeploymentBootstrapEndpoint(
        host=_string(fields["host"], "bootstrap host"),
        port=_integer(fields["port"], "bootstrap port"),
    )


def _decode_prefill(value: object) -> TerminalDeploymentPrefill:
    """Decode the sole strict prefill service.

    :param value: Candidate prefill object.
    :returns: Validated prefill identity.
    """

    fields = _mapping(
        value,
        frozenset(
            (
                "id",
                "launch_instance_id",
                "origin",
                "bootstrap_endpoint",
                "tensor_parallel_size",
            )
        ),
        "prefill",
    )
    return TerminalDeploymentPrefill(
        service_id=_string(fields["id"], "prefill id"),
        launch_instance_id=_uuid_from_json(
            fields["launch_instance_id"], "prefill launch_instance_id"
        ),
        origin=_string(fields["origin"], "prefill origin"),
        bootstrap_endpoint=_decode_bootstrap(fields["bootstrap_endpoint"]),
        tensor_parallel_size=_integer(
            fields["tensor_parallel_size"], "prefill tensor_parallel_size"
        ),
    )


def _decode_decoder(value: object, index: int) -> TerminalDeploymentDecoder:
    """Decode one strict decoder service.

    :param value: Candidate decoder object.
    :param index: Decoder index used in diagnostics.
    :returns: Validated decoder identity.
    """

    fields = _mapping(
        value,
        frozenset(("id", "launch_instance_id", "origin", "tensor_parallel_size")),
        f"decoders[{index}]",
    )
    return TerminalDeploymentDecoder(
        service_id=_string(fields["id"], "decoder id"),
        launch_instance_id=_uuid_from_json(
            fields["launch_instance_id"], "decoder launch_instance_id"
        ),
        origin=_string(fields["origin"], "decoder origin"),
        tensor_parallel_size=_integer(
            fields["tensor_parallel_size"], "decoder tensor_parallel_size"
        ),
    )


def _decode_cohort_payload(value: object) -> TerminalDeploymentCohort:
    """Build immutable domain objects from one strict JSON value.

    :param value: Decoded root value.
    :returns: Validated deployment cohort.
    """

    root = _mapping(
        value,
        frozenset(
            (
                "schema",
                "group_id",
                "model_fingerprint",
                "logical_kv_layout_fingerprint",
                "prefill",
                "decoders",
            )
        ),
        "cohort",
    )
    if root["schema"] != TERMINAL_DEPLOYMENT_COHORT_SCHEMA:
        raise TerminalDeploymentCohortError("unsupported deployment cohort schema")
    raw_decoders = root["decoders"]
    if type(raw_decoders) is not list or len(raw_decoders) == 0:
        raise TerminalDeploymentCohortError("decoders must be a nonempty array")
    return TerminalDeploymentCohort(
        group_id=_string(root["group_id"], "group_id"),
        model_fingerprint=_string(root["model_fingerprint"], "model_fingerprint"),
        logical_kv_layout_fingerprint=_string(
            root["logical_kv_layout_fingerprint"],
            "logical_kv_layout_fingerprint",
        ),
        prefill=_decode_prefill(root["prefill"]),
        decoders=tuple(
            _decode_decoder(raw_decoder, index)
            for index, raw_decoder in enumerate(raw_decoders)
        ),
    )


def decode_terminal_deployment_cohort(payload: bytes) -> TerminalDeploymentCohort:
    """Decode and authenticate one canonical cohort document.

    :param payload: Exact canonical JSON bytes.
    :returns: Complete immutable deployment cohort.
    :raises TerminalDeploymentCohortError: If framing or membership is invalid.
    """

    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    if len(payload) == 0:
        raise TerminalDeploymentCohortError("deployment cohort must not be empty")
    if len(payload) > MAX_TERMINAL_DEPLOYMENT_COHORT_BYTES:
        raise TerminalDeploymentCohortError("deployment cohort exceeds bounded size")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TerminalDeploymentCohortError("deployment cohort is not UTF-8") from error
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
        cohort = _decode_cohort_payload(decoded)
    except TerminalDeploymentCohortError:
        raise
    except (TypeError, ValueError) as error:
        raise TerminalDeploymentCohortError(
            "deployment cohort membership is invalid"
        ) from error
    if encode_terminal_deployment_cohort(cohort) != payload:
        raise TerminalDeploymentCohortError("deployment cohort JSON is not canonical")
    return cohort


def load_terminal_deployment_cohort(
    path: Path,
    expected_digest: bytes,
) -> TerminalDeploymentCohort:
    """Load one immutable canonical cohort without following symlinks.

    :param path: Exact deployment-cohort path.
    :param expected_digest: Launcher-attested canonical SHA-256.
    :returns: Complete immutable deployment cohort.
    :raises TerminalDeploymentCohortError: If the artifact or digest is invalid.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if (
        type(expected_digest) is not bytes
        or len(expected_digest) != TERMINAL_DEPLOYMENT_COHORT_DIGEST_BYTES
    ):
        raise ValueError("expected_digest must contain one SHA-256 digest")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise TerminalDeploymentCohortError(
                    "deployment cohort path is not a regular file"
                )
            payload = stream.read(MAX_TERMINAL_DEPLOYMENT_COHORT_BYTES + 1)
    except OSError as error:
        raise TerminalDeploymentCohortError(
            "deployment cohort artifact cannot be opened safely"
        ) from error
    cohort = decode_terminal_deployment_cohort(payload)
    actual_digest = hashlib.sha256(payload).digest()
    if not secrets.compare_digest(expected_digest, actual_digest):
        raise TerminalDeploymentCohortError(
            "deployment cohort digest differs from launcher attestation"
        )
    return cohort
