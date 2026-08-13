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

TERMINAL_DEPLOYMENT_COHORT_SCHEMA: str = "pd-terminal-deployment-cohort-v1"
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
class TerminalDeploymentService:
    """One launcher-owned model-service incarnation.

    :ivar service_id: Stable service name within one deployment run.
    :ivar role: Prefill or decode role.
    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar tensor_parallel_size: Complete service TP width.
    :ivar origin: Exact loopback HTTP origin used by the launcher.
    :ivar port: HTTP listener port carried by ``origin``.
    """

    service_id: str
    role: TerminalDeploymentRole
    launch_instance_id: uuid.UUID
    tensor_parallel_size: int
    origin: str
    port: int

    def __post_init__(self) -> None:
        """Validate one exact static service identity."""

        if (
            type(self.service_id) is not str
            or _IDENTIFIER.fullmatch(self.service_id) is None
        ):
            raise ValueError("service_id is not canonical")
        if type(self.role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        _require_uuid(self.launch_instance_id, "launch_instance_id")
        allowed_tp_sizes = (
            _PREFILL_TP_SIZES
            if self.role is TerminalDeploymentRole.PREFILL
            else _DECODE_TP_SIZES
        )
        if (
            type(self.tensor_parallel_size) is not int
            or self.tensor_parallel_size not in allowed_tp_sizes
        ):
            raise ValueError("tensor_parallel_size is unsupported for the role")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("service port must be in the TCP port range")
        if type(self.origin) is not str:
            raise TypeError("origin must be str")
        match = _LOOPBACK_ORIGIN.fullmatch(self.origin)
        if match is None or int(match.group("port")) != self.port:
            raise ValueError("origin must be the exact loopback service port")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentLocalService:
    """One exact local member selected from a hash-bound cohort.

    :ivar group_id: Owning static prefill group.
    :ivar bootstrap_endpoint: Group prefill bootstrap endpoint.
    :ivar service: Exact launcher-owned service identity.
    """

    group_id: str
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint
    service: TerminalDeploymentService

    def __post_init__(self) -> None:
        """Validate one selected local membership."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
            raise TypeError(
                "bootstrap_endpoint must be TerminalDeploymentBootstrapEndpoint"
            )
        if type(self.service) is not TerminalDeploymentService:
            raise TypeError("service must be TerminalDeploymentService")


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
        if service.group_id != self.group_id:
            return False
        member = service.service
        if member.role is TerminalDeploymentRole.PREFILL:
            return (
                member.service_id == self.prefill_service_id
                and member.launch_instance_id == self.prefill_launch_instance_id
            )
        return (
            member.service_id == self.decoder_service_id
            and member.launch_instance_id == self.decoder_launch_instance_id
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentCohort:
    """Complete immutable membership of one isolated prefill group.

    :ivar group_id: Stable topology-local ownership identifier.
    :ivar model_fingerprint: Exact model-weights compatibility fingerprint.
    :ivar logical_kv_layout_fingerprint: Exact TP-independent KV layout.
    :ivar bootstrap_endpoint: Exact group prefill bootstrap endpoint.
    :ivar services: Prefill-first complete service membership.
    """

    group_id: str
    model_fingerprint: str
    logical_kv_layout_fingerprint: str
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint
    services: tuple[TerminalDeploymentService, ...]

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
        if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
            raise TypeError(
                "bootstrap_endpoint must be TerminalDeploymentBootstrapEndpoint"
            )
        if type(self.services) is not tuple or len(self.services) < 2:
            raise ValueError(
                "services must contain one prefill and at least one decoder"
            )
        if any(
            type(service) is not TerminalDeploymentService for service in self.services
        ):
            raise TypeError("services must contain TerminalDeploymentService")
        expected_roles = (
            TerminalDeploymentRole.PREFILL,
            *(TerminalDeploymentRole.DECODE for _ in self.services[1:]),
        )
        if tuple(service.role for service in self.services) != expected_roles:
            raise ValueError("services must order one prefill before every decoder")
        unique_fields = (
            ("service_id", tuple(service.service_id for service in self.services)),
            (
                "launch_instance_id",
                tuple(service.launch_instance_id for service in self.services),
            ),
            ("origin", tuple(service.origin for service in self.services)),
            ("port", tuple(service.port for service in self.services)),
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

    @property
    def prefill(self) -> TerminalDeploymentService:
        """Return the cohort's sole prefill service.

        :returns: First and sole prefill member.
        """

        return self.services[0]

    @property
    def decoders(self) -> tuple[TerminalDeploymentService, ...]:
        """Return the cohort's complete decoder membership.

        :returns: Canonically ordered decoder services.
        """

        return self.services[1:]

    def require_local_service(
        self,
        *,
        service_id: str,
        role: TerminalDeploymentRole,
        launch_instance_id: uuid.UUID,
        tensor_parallel_size: int,
        origin: str,
        port: int,
        bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint | None,
    ) -> TerminalDeploymentLocalService:
        """Select one exact local service or reject configuration drift.

        :param service_id: Launcher-selected local service name.
        :param role: Expected local process role.
        :param launch_instance_id: Local launcher-assigned incarnation.
        :param tensor_parallel_size: Local runtime TP width.
        :param origin: Local HTTP origin.
        :param port: Local HTTP listener port.
        :param bootstrap_endpoint: Local prefill endpoint, absent for decode.
        :returns: Exact cohort membership for the local service.
        :raises TerminalDeploymentCohortError: If no exact member exists.
        """

        if type(role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        matching = tuple(
            service for service in self.services if service.service_id == service_id
        )
        if len(matching) != 1:
            raise TerminalDeploymentCohortError(
                "local service identity is absent from its cohort"
            )
        expected = matching[0]
        claimed = TerminalDeploymentService(
            service_id=service_id,
            role=role,
            launch_instance_id=launch_instance_id,
            tensor_parallel_size=tensor_parallel_size,
            origin=origin,
            port=port,
        )
        if expected != claimed:
            raise TerminalDeploymentCohortError(
                "local service configuration differs from cohort membership"
            )
        if role is TerminalDeploymentRole.PREFILL:
            if bootstrap_endpoint != self.bootstrap_endpoint:
                raise TerminalDeploymentCohortError(
                    "local prefill bootstrap differs from cohort membership"
                )
        elif bootstrap_endpoint is not None:
            raise TerminalDeploymentCohortError(
                "local decoder cannot claim a prefill bootstrap endpoint"
            )
        return TerminalDeploymentLocalService(
            group_id=self.group_id,
            bootstrap_endpoint=self.bootstrap_endpoint,
            service=expected,
        )

    def require_request_plan(
        self,
        plan: TerminalDeploymentRequestPlan,
    ) -> tuple[TerminalDeploymentService, TerminalDeploymentService]:
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
        prefill = self.prefill
        if (
            plan.prefill_service_id != prefill.service_id
            or plan.prefill_launch_instance_id != prefill.launch_instance_id
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
        return prefill, decoder


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
        "bootstrap_endpoint": {
            "host": cohort.bootstrap_endpoint.host,
            "port": cohort.bootstrap_endpoint.port,
        },
        "services": [
            {
                "id": service.service_id,
                "role": service.role.value,
                "launch_instance_id": str(service.launch_instance_id),
                "tensor_parallel_size": service.tensor_parallel_size,
                "origin": service.origin,
                "port": service.port,
            }
            for service in cohort.services
        ],
    }


def encode_terminal_deployment_cohort(cohort: TerminalDeploymentCohort) -> bytes:
    """Encode one cohort into its sole accepted byte representation.

    :param cohort: Complete immutable deployment cohort.
    :returns: Compact canonical UTF-8 JSON without a trailing newline.
    """

    return json.dumps(
        _cohort_payload(cohort),
        ensure_ascii=False,
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
    host = fields["host"]
    if type(host) is not str:
        raise TerminalDeploymentCohortError("bootstrap host must be a string")
    return TerminalDeploymentBootstrapEndpoint(
        host=host,
        port=_integer(fields["port"], "bootstrap port"),
    )


def _decode_service(value: object, index: int) -> TerminalDeploymentService:
    """Decode one strict service member.

    :param value: Candidate service object.
    :param index: Service index used in diagnostics.
    :returns: Validated service identity.
    """

    fields = _mapping(
        value,
        frozenset(
            (
                "id",
                "role",
                "launch_instance_id",
                "tensor_parallel_size",
                "origin",
                "port",
            )
        ),
        f"services[{index}]",
    )
    service_id = fields["id"]
    role = fields["role"]
    origin = fields["origin"]
    if type(service_id) is not str:
        raise TerminalDeploymentCohortError("service id must be a string")
    if type(role) is not str:
        raise TerminalDeploymentCohortError("service role must be a string")
    if type(origin) is not str:
        raise TerminalDeploymentCohortError("service origin must be a string")
    try:
        parsed_role = TerminalDeploymentRole(role)
    except ValueError as error:
        raise TerminalDeploymentCohortError("service role is unsupported") from error
    return TerminalDeploymentService(
        service_id=service_id,
        role=parsed_role,
        launch_instance_id=_uuid_from_json(
            fields["launch_instance_id"], "service launch_instance_id"
        ),
        tensor_parallel_size=_integer(
            fields["tensor_parallel_size"], "service tensor_parallel_size"
        ),
        origin=origin,
        port=_integer(fields["port"], "service port"),
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
                "bootstrap_endpoint",
                "services",
            )
        ),
        "cohort",
    )
    if root["schema"] != TERMINAL_DEPLOYMENT_COHORT_SCHEMA:
        raise TerminalDeploymentCohortError("unsupported deployment cohort schema")
    group_id = root["group_id"]
    if type(group_id) is not str:
        raise TerminalDeploymentCohortError("group_id must be a string")
    model_fingerprint = root["model_fingerprint"]
    logical_kv_layout_fingerprint = root["logical_kv_layout_fingerprint"]
    if type(model_fingerprint) is not str:
        raise TerminalDeploymentCohortError("model_fingerprint must be a string")
    if type(logical_kv_layout_fingerprint) is not str:
        raise TerminalDeploymentCohortError(
            "logical_kv_layout_fingerprint must be a string"
        )
    raw_services = root["services"]
    if type(raw_services) is not list or len(raw_services) < 2:
        raise TerminalDeploymentCohortError(
            "services must contain one prefill and at least one decoder"
        )
    return TerminalDeploymentCohort(
        group_id=group_id,
        model_fingerprint=model_fingerprint,
        logical_kv_layout_fingerprint=logical_kv_layout_fingerprint,
        bootstrap_endpoint=_decode_bootstrap(root["bootstrap_endpoint"]),
        services=tuple(
            _decode_service(raw_service, index)
            for index, raw_service in enumerate(raw_services)
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
