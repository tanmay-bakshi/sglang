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

_GROUP_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
_BOOTSTRAP_HOST = re.compile(r"[A-Za-z0-9._:\[\]-]+")
_PREFILL_TP_SIZES = frozenset((1, 2, 4, 8))
_DECODE_TP_SIZES = frozenset((1, 2))


class TerminalDeploymentCohortError(ValueError):
    """Invalid immutable packed-terminal deployment cohort."""


class TerminalDeploymentRole(enum.StrEnum):
    """Role of one model service in a terminal deployment cohort."""

    PREFILL = "prefill"
    DECODE = "decode"


def _require_uuid(value: uuid.UUID, label: str) -> None:
    """Require one non-nil UUID.

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
class TerminalDeploymentPrefill:
    """Exact prefill service admitted to one deployment epoch.

    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar bootstrap_endpoint: Exact bootstrap endpoint for native peer preflight.
    :ivar tensor_parallel_size: Complete prefill TP width.
    """

    launch_instance_id: uuid.UUID
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate one prefill service identity."""

        _require_uuid(self.launch_instance_id, "prefill launch_instance_id")
        if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
            raise TypeError(
                "bootstrap_endpoint must be TerminalDeploymentBootstrapEndpoint"
            )
        if (
            type(self.tensor_parallel_size) is not int
            or self.tensor_parallel_size not in _PREFILL_TP_SIZES
        ):
            raise ValueError("prefill tensor_parallel_size is unsupported")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentDecoder:
    """Exact decode service admitted to one deployment epoch.

    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar tensor_parallel_size: Complete decode TP width.
    """

    launch_instance_id: uuid.UUID
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate one decode service identity."""

        _require_uuid(self.launch_instance_id, "decoder launch_instance_id")
        if (
            type(self.tensor_parallel_size) is not int
            or self.tensor_parallel_size not in _DECODE_TP_SIZES
        ):
            raise ValueError("decoder tensor_parallel_size is unsupported")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentGroup:
    """One immutable prefill ownership group and launch epoch.

    :ivar group_id: Stable topology-local ownership identifier.
    :ivar launch_epoch: Incarnation of the complete group membership.
    :ivar prefill: Exact prefill service for this group.
    :ivar decoders: Ordered nonempty decode-service membership.
    """

    group_id: str
    launch_epoch: uuid.UUID
    prefill: TerminalDeploymentPrefill
    decoders: tuple[TerminalDeploymentDecoder, ...]

    def __post_init__(self) -> None:
        """Validate complete and collision-free group membership."""

        if type(self.group_id) is not str or _GROUP_ID.fullmatch(self.group_id) is None:
            raise ValueError("group_id is not canonical")
        _require_uuid(self.launch_epoch, "group launch_epoch")
        if type(self.prefill) is not TerminalDeploymentPrefill:
            raise TypeError("prefill must be TerminalDeploymentPrefill")
        if type(self.decoders) is not tuple or len(self.decoders) == 0:
            raise ValueError("decoders must be a nonempty tuple")
        if any(
            type(decoder) is not TerminalDeploymentDecoder for decoder in self.decoders
        ):
            raise TypeError("decoders must contain TerminalDeploymentDecoder")
        launch_ids = (
            self.prefill.launch_instance_id,
            *(decoder.launch_instance_id for decoder in self.decoders),
        )
        if len(set(launch_ids)) != len(launch_ids):
            raise ValueError("service launch_instance_id values must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentLocalService:
    """Exact group member selected by one local model process.

    :ivar group_id: Owning deployment group.
    :ivar group_launch_epoch: Exact group membership incarnation.
    :ivar role: Local service role.
    :ivar launch_instance_id: Local launcher-assigned incarnation.
    :ivar tensor_parallel_size: Complete local service TP width.
    :ivar bootstrap_endpoint: Prefill bootstrap endpoint, absent for decode.
    """

    group_id: str
    group_launch_epoch: uuid.UUID
    role: TerminalDeploymentRole
    launch_instance_id: uuid.UUID
    tensor_parallel_size: int
    bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint | None

    def __post_init__(self) -> None:
        """Validate one fully selected local membership."""

        if type(self.group_id) is not str or _GROUP_ID.fullmatch(self.group_id) is None:
            raise ValueError("group_id is not canonical")
        _require_uuid(self.group_launch_epoch, "group_launch_epoch")
        if type(self.role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        _require_uuid(self.launch_instance_id, "launch_instance_id")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.role is TerminalDeploymentRole.PREFILL:
            if type(self.bootstrap_endpoint) is not TerminalDeploymentBootstrapEndpoint:
                raise ValueError("prefill membership requires a bootstrap endpoint")
            return
        if self.bootstrap_endpoint is not None:
            raise ValueError("decode membership cannot carry a bootstrap endpoint")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentRequestPlan:
    """One request-selected prefill/decode subset of a frozen group.

    :ivar group_id: Selected ownership group.
    :ivar group_launch_epoch: Exact group membership incarnation.
    :ivar prefill_launch_instance_id: Selected prefill service incarnation.
    :ivar prefill_bootstrap_endpoint: Exact prefill bootstrap endpoint.
    :ivar prefill_tensor_parallel_size: Complete selected prefill TP width.
    :ivar decoder_launch_instance_id: Selected decode service incarnation.
    :ivar decoder_tensor_parallel_size: Complete selected decode TP width.
    """

    group_id: str
    group_launch_epoch: uuid.UUID
    prefill_launch_instance_id: uuid.UUID
    prefill_bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint
    prefill_tensor_parallel_size: int
    decoder_launch_instance_id: uuid.UUID
    decoder_tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate the shape of one request selection."""

        if type(self.group_id) is not str or _GROUP_ID.fullmatch(self.group_id) is None:
            raise ValueError("group_id is not canonical")
        _require_uuid(self.group_launch_epoch, "group_launch_epoch")
        _require_uuid(self.prefill_launch_instance_id, "prefill_launch_instance_id")
        if (
            type(self.prefill_bootstrap_endpoint)
            is not TerminalDeploymentBootstrapEndpoint
        ):
            raise TypeError(
                "prefill_bootstrap_endpoint must be TerminalDeploymentBootstrapEndpoint"
            )
        if (
            type(self.prefill_tensor_parallel_size) is not int
            or self.prefill_tensor_parallel_size not in _PREFILL_TP_SIZES
        ):
            raise ValueError("prefill_tensor_parallel_size is unsupported")
        _require_uuid(self.decoder_launch_instance_id, "decoder_launch_instance_id")
        if (
            type(self.decoder_tensor_parallel_size) is not int
            or self.decoder_tensor_parallel_size not in _DECODE_TP_SIZES
        ):
            raise ValueError("decoder_tensor_parallel_size is unsupported")
        if self.prefill_launch_instance_id == self.decoder_launch_instance_id:
            raise ValueError("prefill and decoder launch identities must differ")

    def contains(self, service: TerminalDeploymentLocalService) -> bool:
        """Return whether this request selection contains one exact local member.

        :param service: Cohort-validated local service membership.
        :returns: Whether the member is one of this plan's two services.
        """

        if type(service) is not TerminalDeploymentLocalService:
            raise TypeError("service must be TerminalDeploymentLocalService")
        if (
            service.group_id != self.group_id
            or service.group_launch_epoch != self.group_launch_epoch
        ):
            return False
        if service.role is TerminalDeploymentRole.PREFILL:
            return (
                service.launch_instance_id == self.prefill_launch_instance_id
                and service.tensor_parallel_size == self.prefill_tensor_parallel_size
                and service.bootstrap_endpoint == self.prefill_bootstrap_endpoint
            )
        return (
            service.launch_instance_id == self.decoder_launch_instance_id
            and service.tensor_parallel_size == self.decoder_tensor_parallel_size
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeploymentCohort:
    """Complete immutable deployment membership consumed before runtime start.

    :ivar groups: Canonically ordered nonempty ownership groups.
    """

    groups: tuple[TerminalDeploymentGroup, ...]

    def __post_init__(self) -> None:
        """Validate deployment-global identity uniqueness and ordering."""

        if type(self.groups) is not tuple or len(self.groups) == 0:
            raise ValueError("groups must be a nonempty tuple")
        if any(type(group) is not TerminalDeploymentGroup for group in self.groups):
            raise TypeError("groups must contain TerminalDeploymentGroup")
        group_ids = tuple(group.group_id for group in self.groups)
        if tuple(sorted(group_ids)) != group_ids:
            raise ValueError("groups must use canonical group_id order")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("group_id values must be unique")
        launch_epochs = tuple(group.launch_epoch for group in self.groups)
        if len(set(launch_epochs)) != len(launch_epochs):
            raise ValueError("group launch_epoch values must be unique")
        service_launch_ids = tuple(
            launch_id
            for group in self.groups
            for launch_id in (
                group.prefill.launch_instance_id,
                *(decoder.launch_instance_id for decoder in group.decoders),
            )
        )
        if len(set(service_launch_ids)) != len(service_launch_ids):
            raise ValueError(
                "service launch_instance_id values must be deployment-global"
            )

    @property
    def digest(self) -> bytes:
        """Return the canonical cohort SHA-256.

        :returns: Digest of the exact canonical cohort encoding.
        """

        return hashlib.sha256(encode_terminal_deployment_cohort(self)).digest()

    def require_local_service(
        self,
        *,
        group_id: str,
        role: TerminalDeploymentRole,
        launch_instance_id: uuid.UUID,
        tensor_parallel_size: int,
        bootstrap_endpoint: TerminalDeploymentBootstrapEndpoint | None,
    ) -> TerminalDeploymentLocalService:
        """Select one exact local service or reject configuration drift.

        :param group_id: Expected ownership group.
        :param role: Expected process role.
        :param launch_instance_id: Local launcher-assigned incarnation.
        :param tensor_parallel_size: Local runtime TP width.
        :param bootstrap_endpoint: Local prefill endpoint, absent for decode.
        :returns: Exact cohort membership for the local service.
        :raises TerminalDeploymentCohortError: If no exact member exists.
        """

        if type(role) is not TerminalDeploymentRole:
            raise TypeError("role must be TerminalDeploymentRole")
        group = self._group(group_id)
        membership: TerminalDeploymentLocalService
        if role is TerminalDeploymentRole.PREFILL:
            prefill = group.prefill
            membership = TerminalDeploymentLocalService(
                group_id=group.group_id,
                group_launch_epoch=group.launch_epoch,
                role=role,
                launch_instance_id=prefill.launch_instance_id,
                tensor_parallel_size=prefill.tensor_parallel_size,
                bootstrap_endpoint=prefill.bootstrap_endpoint,
            )
        else:
            matching = tuple(
                decoder
                for decoder in group.decoders
                if decoder.launch_instance_id == launch_instance_id
            )
            if len(matching) != 1:
                raise TerminalDeploymentCohortError(
                    "local decoder launch identity is absent from its group"
                )
            decoder = matching[0]
            membership = TerminalDeploymentLocalService(
                group_id=group.group_id,
                group_launch_epoch=group.launch_epoch,
                role=role,
                launch_instance_id=decoder.launch_instance_id,
                tensor_parallel_size=decoder.tensor_parallel_size,
                bootstrap_endpoint=None,
            )
        claimed = TerminalDeploymentLocalService(
            group_id=group_id,
            group_launch_epoch=group.launch_epoch,
            role=role,
            launch_instance_id=launch_instance_id,
            tensor_parallel_size=tensor_parallel_size,
            bootstrap_endpoint=bootstrap_endpoint,
        )
        if membership != claimed:
            raise TerminalDeploymentCohortError(
                "local service configuration differs from cohort membership"
            )
        return membership

    def require_request_plan(
        self,
        plan: TerminalDeploymentRequestPlan,
    ) -> tuple[
        TerminalDeploymentGroup,
        TerminalDeploymentPrefill,
        TerminalDeploymentDecoder,
    ]:
        """Validate one request-selected subset against exact group membership.

        :param plan: Request-scoped deployment selection.
        :returns: Exact group, prefill, and decoder cohort records.
        :raises TerminalDeploymentCohortError: If any selected fact is stale.
        """

        if type(plan) is not TerminalDeploymentRequestPlan:
            raise TypeError("plan must be TerminalDeploymentRequestPlan")
        group = self._group(plan.group_id)
        if group.launch_epoch != plan.group_launch_epoch:
            raise TerminalDeploymentCohortError(
                "request plan uses a stale group launch epoch"
            )
        prefill = group.prefill
        if (
            prefill.launch_instance_id != plan.prefill_launch_instance_id
            or prefill.bootstrap_endpoint != plan.prefill_bootstrap_endpoint
            or prefill.tensor_parallel_size != plan.prefill_tensor_parallel_size
        ):
            raise TerminalDeploymentCohortError(
                "request plan prefill differs from group membership"
            )
        matching = tuple(
            decoder
            for decoder in group.decoders
            if decoder.launch_instance_id == plan.decoder_launch_instance_id
        )
        if len(matching) != 1:
            raise TerminalDeploymentCohortError(
                "request plan decoder is absent from its group"
            )
        decoder = matching[0]
        if decoder.tensor_parallel_size != plan.decoder_tensor_parallel_size:
            raise TerminalDeploymentCohortError(
                "request plan decoder TP differs from group membership"
            )
        return group, prefill, decoder

    def _group(self, group_id: str) -> TerminalDeploymentGroup:
        """Resolve one exact group identifier.

        :param group_id: Stable deployment group identity.
        :returns: Exactly one matching group.
        :raises TerminalDeploymentCohortError: If the group is absent.
        """

        if type(group_id) is not str:
            raise TypeError("group_id must be str")
        matching = tuple(group for group in self.groups if group.group_id == group_id)
        if len(matching) != 1:
            raise TerminalDeploymentCohortError("deployment group is absent")
        return matching[0]


def _cohort_payload(cohort: TerminalDeploymentCohort) -> dict[str, object]:
    """Project one validated cohort into canonical JSON field order.

    :param cohort: Complete immutable deployment cohort.
    :returns: JSON-native canonical payload.
    """

    if type(cohort) is not TerminalDeploymentCohort:
        raise TypeError("cohort must be TerminalDeploymentCohort")
    return {
        "schema": TERMINAL_DEPLOYMENT_COHORT_SCHEMA,
        "groups": [
            {
                "id": group.group_id,
                "launch_epoch": str(group.launch_epoch),
                "prefill": {
                    "launch_instance_id": str(group.prefill.launch_instance_id),
                    "bootstrap_endpoint": {
                        "host": group.prefill.bootstrap_endpoint.host,
                        "port": group.prefill.bootstrap_endpoint.port,
                    },
                    "tensor_parallel_size": group.prefill.tensor_parallel_size,
                },
                "decoders": [
                    {
                        "launch_instance_id": str(decoder.launch_instance_id),
                        "tensor_parallel_size": decoder.tensor_parallel_size,
                    }
                    for decoder in group.decoders
                ],
            }
            for group in cohort.groups
        ],
    }


def encode_terminal_deployment_cohort(cohort: TerminalDeploymentCohort) -> bytes:
    """Encode one cohort into its only accepted JSON representation.

    :param cohort: Complete immutable deployment cohort.
    :returns: Canonical compact UTF-8 JSON without a trailing newline.
    """

    return json.dumps(
        _cohort_payload(cohort),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Decode one JSON object while rejecting duplicate member names.

    :param pairs: Ordered JSON object members.
    :returns: Newly allocated mapping preserving wire order.
    :raises TerminalDeploymentCohortError: If a member name repeats.
    """

    value: dict[str, object] = {}
    for key, member in pairs:
        if key in value:
            raise TerminalDeploymentCohortError(f"duplicate JSON field: {key}")
        value[key] = member
    return value


def _reject_json_constant(value: str) -> None:
    """Reject non-finite JSON extensions.

    :param value: Decoder-supplied extension token.
    :raises TerminalDeploymentCohortError: Always.
    """

    raise TerminalDeploymentCohortError(f"invalid JSON constant: {value}")


def _mapping(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    """Require one object with an exact field set.

    :param value: Candidate JSON value.
    :param fields: Complete accepted field names.
    :param label: Reader-facing object label.
    :returns: Validated mapping.
    """

    if type(value) is not dict:
        raise TerminalDeploymentCohortError(f"{label} must be an object")
    mapping = value
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


def _decode_cohort_payload(value: object) -> TerminalDeploymentCohort:
    """Build immutable domain objects from one strict JSON value.

    :param value: Decoded root value.
    :returns: Validated deployment cohort.
    """

    root = _mapping(value, frozenset(("schema", "groups")), "cohort")
    if root["schema"] != TERMINAL_DEPLOYMENT_COHORT_SCHEMA:
        raise TerminalDeploymentCohortError("unsupported deployment cohort schema")
    raw_groups = root["groups"]
    if type(raw_groups) is not list or len(raw_groups) == 0:
        raise TerminalDeploymentCohortError("groups must be a nonempty array")
    groups: list[TerminalDeploymentGroup] = []
    for group_index, raw_group in enumerate(raw_groups):
        group_fields = _mapping(
            raw_group,
            frozenset(("id", "launch_epoch", "prefill", "decoders")),
            f"groups[{group_index}]",
        )
        group_id = group_fields["id"]
        if type(group_id) is not str:
            raise TerminalDeploymentCohortError("group id must be a string")
        prefill_fields = _mapping(
            group_fields["prefill"],
            frozenset(
                (
                    "launch_instance_id",
                    "bootstrap_endpoint",
                    "tensor_parallel_size",
                )
            ),
            f"groups[{group_index}].prefill",
        )
        raw_decoders = group_fields["decoders"]
        if type(raw_decoders) is not list or len(raw_decoders) == 0:
            raise TerminalDeploymentCohortError("decoders must be a nonempty array")
        decoders: list[TerminalDeploymentDecoder] = []
        for decoder_index, raw_decoder in enumerate(raw_decoders):
            decoder_fields = _mapping(
                raw_decoder,
                frozenset(("launch_instance_id", "tensor_parallel_size")),
                f"groups[{group_index}].decoders[{decoder_index}]",
            )
            decoders.append(
                TerminalDeploymentDecoder(
                    launch_instance_id=_uuid_from_json(
                        decoder_fields["launch_instance_id"],
                        "decoder launch_instance_id",
                    ),
                    tensor_parallel_size=_integer(
                        decoder_fields["tensor_parallel_size"],
                        "decoder tensor_parallel_size",
                    ),
                )
            )
        groups.append(
            TerminalDeploymentGroup(
                group_id=group_id,
                launch_epoch=_uuid_from_json(
                    group_fields["launch_epoch"], "group launch_epoch"
                ),
                prefill=TerminalDeploymentPrefill(
                    launch_instance_id=_uuid_from_json(
                        prefill_fields["launch_instance_id"],
                        "prefill launch_instance_id",
                    ),
                    bootstrap_endpoint=_decode_bootstrap(
                        prefill_fields["bootstrap_endpoint"]
                    ),
                    tensor_parallel_size=_integer(
                        prefill_fields["tensor_parallel_size"],
                        "prefill tensor_parallel_size",
                    ),
                ),
                decoders=tuple(decoders),
            )
        )
    return TerminalDeploymentCohort(groups=tuple(groups))


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
    if not secrets.compare_digest(
        expected_digest,
        actual_digest,
    ):
        raise TerminalDeploymentCohortError(
            "deployment cohort digest differs from launcher attestation"
        )
    return cohort
