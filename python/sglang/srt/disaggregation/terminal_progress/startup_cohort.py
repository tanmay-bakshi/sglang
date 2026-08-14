import dataclasses
import enum
import hashlib
import json
import math
import re
import threading
import time
import urllib.parse
import uuid

from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
)

TERMINAL_STARTUP_COHORT_SCHEMA: str = "packed-terminal-startup-cohort-v1"
TERMINAL_STARTUP_ADVERTISEMENT_SCHEMA: str = (
    "packed-terminal-startup-rank-advertisement-v1"
)
TERMINAL_STARTUP_COHORT_DIGEST_BYTES: int = hashlib.sha256().digest_size
TERMINAL_STARTUP_WIRE_MAX_BYTES: int = 1024 * 1024

_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}")
_NIXL_AGENT_NAME_MAX_BYTES = 256
_SERVICE_ORIGIN_MAX_BYTES = 2048


class TerminalStartupCohortError(RuntimeError):
    """Invalid or failed terminal startup cohort."""


class TerminalStartupCohortDisposition(enum.StrEnum):
    """Lifecycle of one deployment-epoch startup registry."""

    OPEN = "open"
    SEALED = "sealed"
    FAILED = "failed"


def _require_uuid_bytes(value: bytes, label: str) -> None:
    """Require one canonical non-nil UUID encoded as bytes.

    :param value: Candidate UUID bytes.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != 16:
        raise ValueError(f"{label} must contain 16 bytes")
    if uuid.UUID(bytes=value).int == 0:
        raise ValueError(f"{label} must be non-nil")


def _require_sha256(value: bytes, label: str) -> None:
    """Require one binary SHA-256 value.

    :param value: Candidate digest.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != TERMINAL_STARTUP_COHORT_DIGEST_BYTES:
        raise ValueError(f"{label} must contain 32 bytes")


def _require_service_origin(value: str) -> None:
    """Require one absolute HTTP service origin without route components.

    :param value: Candidate canonical service origin.
    """

    if type(value) is not str or len(value) == 0:
        raise ValueError("service_origin must be nonempty")
    try:
        encoded = value.encode("ascii")
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError("service_origin is malformed") from error
    if len(encoded) > _SERVICE_ORIGIN_MAX_BYTES:
        raise ValueError("service_origin is too large")
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname is None
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != ""
        or parsed.query != ""
        or parsed.fragment != ""
    ):
        raise ValueError("service_origin must contain only scheme, host, and port")


def _role_order(role: TerminalOwnerRole) -> int:
    """Return the canonical deployment role order.

    :param role: Source or decode service role.
    :returns: Source-first integer order.
    """

    if role is TerminalOwnerRole.SOURCE:
        return 0
    return 1


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupServiceExpectation:
    """One statically admitted model-service incarnation.

    :ivar service_id: Launcher-assigned service identifier.
    :ivar service_origin: Exact canonical model-service origin.
    :ivar role: Source or decode role.
    :ivar launch_instance_id: Launcher-assigned service incarnation.
    :ivar tensor_parallel_size: Exact number of terminal-owner ranks.
    """

    service_id: str
    service_origin: str
    role: TerminalOwnerRole
    launch_instance_id: bytes
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        """Validate exact static membership."""

        if (
            type(self.service_id) is not str
            or _IDENTIFIER.fullmatch(self.service_id) is None
        ):
            raise ValueError("service_id is not canonical")
        _require_service_origin(self.service_origin)
        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        _require_uuid_bytes(self.launch_instance_id, "launch_instance_id")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupCohortExpectation:
    """Launcher-authenticated membership for one ownership group.

    :ivar group_id: Static prefill ownership group.
    :ivar cohort_sha256: Digest of the canonical launcher cohort document.
    :ivar services: Source-first, service-ID-sorted exact membership.
    """

    group_id: str
    cohort_sha256: bytes
    services: tuple[TerminalStartupServiceExpectation, ...]

    def __post_init__(self) -> None:
        """Validate complete, canonical, collision-free membership."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        _require_sha256(self.cohort_sha256, "cohort_sha256")
        if type(self.services) is not tuple or len(self.services) < 2:
            raise ValueError("startup cohort requires source and decode services")
        if any(
            type(service) is not TerminalStartupServiceExpectation
            for service in self.services
        ):
            raise TypeError(
                "services must contain TerminalStartupServiceExpectation values"
            )
        expected_order = tuple(
            sorted(
                self.services,
                key=lambda service: (_role_order(service.role), service.service_id),
            )
        )
        if self.services != expected_order:
            raise ValueError("services are not in canonical source-first order")
        if (
            sum(service.role is TerminalOwnerRole.SOURCE for service in self.services)
            != 1
        ):
            raise ValueError("startup cohort requires exactly one source service")
        if not any(
            service.role is TerminalOwnerRole.DECODE for service in self.services
        ):
            raise ValueError("startup cohort requires at least one decode service")
        service_ids = tuple(service.service_id for service in self.services)
        service_origins = tuple(service.service_origin for service in self.services)
        launch_ids = tuple(service.launch_instance_id for service in self.services)
        if len(set(service_ids)) != len(service_ids):
            raise ValueError("service_id values must be unique")
        if len(set(service_origins)) != len(service_origins):
            raise ValueError("service_origin values must be unique")
        if len(set(launch_ids)) != len(launch_ids):
            raise ValueError("launch_instance_id values must be unique")

    @property
    def expected_rank_count(self) -> int:
        """Return the complete process-rank population.

        :returns: Sum of exact service TP widths.
        """

        return sum(service.tensor_parallel_size for service in self.services)

    def service(self, service_id: str) -> TerminalStartupServiceExpectation:
        """Resolve one exact static service.

        :param service_id: Launcher service identifier.
        :returns: Exact service expectation.
        :raises TerminalStartupCohortError: If the service is absent.
        """

        if type(service_id) is not str:
            raise TypeError("service_id must be a string")
        matches = tuple(
            service for service in self.services if service.service_id == service_id
        )
        if len(matches) != 1:
            raise TerminalStartupCohortError(
                f"service {service_id!r} is absent from startup cohort"
            )
        return matches[0]

    def require_advertisement(
        self,
        advertisement: "TerminalStartupRankAdvertisement",
    ) -> None:
        """Bind one observed native rank to exact static membership.

        :param advertisement: Observed per-rank NIXL identity.
        :raises TerminalStartupCohortError: If any static field differs.
        """

        if type(advertisement) is not TerminalStartupRankAdvertisement:
            raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
        if (
            advertisement.group_id != self.group_id
            or advertisement.cohort_sha256 != self.cohort_sha256
        ):
            raise TerminalStartupCohortError(
                "rank advertisement belongs to another deployment epoch"
            )
        service = self.service(advertisement.service_id)
        if (
            advertisement.service_origin != service.service_origin
            or advertisement.role is not service.role
            or advertisement.launch_instance_id != service.launch_instance_id
            or advertisement.tensor_parallel_size != service.tensor_parallel_size
            or advertisement.tensor_parallel_rank >= service.tensor_parallel_size
        ):
            raise TerminalStartupCohortError(
                "rank advertisement differs from static service membership"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupRankAdvertisement:
    """Observed per-rank native identity bound to one static service.

    :ivar group_id: Static ownership group.
    :ivar cohort_sha256: Canonical launcher cohort digest.
    :ivar service_id: Exact service member.
    :ivar service_origin: Exact canonical model-service origin.
    :ivar role: Source or decode role.
    :ivar launch_instance_id: Exact launcher service incarnation.
    :ivar tensor_parallel_rank: Rank within the service.
    :ivar tensor_parallel_size: Exact service TP width.
    :ivar process_generation: Native NIXL process incarnation.
    :ivar nixl_agent_name: Exact native agent name.
    :ivar nixl_agent_metadata_sha256: Digest of the complete agent metadata.
    """

    group_id: str
    cohort_sha256: bytes
    service_id: str
    service_origin: str
    role: TerminalOwnerRole
    launch_instance_id: bytes
    tensor_parallel_rank: int
    tensor_parallel_size: int
    process_generation: bytes
    nixl_agent_name: str
    nixl_agent_metadata_sha256: bytes

    def __post_init__(self) -> None:
        """Validate one complete advertised rank identity."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        _require_sha256(self.cohort_sha256, "cohort_sha256")
        if (
            type(self.service_id) is not str
            or _IDENTIFIER.fullmatch(self.service_id) is None
        ):
            raise ValueError("service_id is not canonical")
        _require_service_origin(self.service_origin)
        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        _require_uuid_bytes(self.launch_instance_id, "launch_instance_id")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if (
            type(self.tensor_parallel_rank) is not int
            or self.tensor_parallel_rank < 0
            or self.tensor_parallel_rank >= self.tensor_parallel_size
        ):
            raise ValueError("tensor_parallel_rank is outside its service")
        _require_uuid_bytes(self.process_generation, "process_generation")
        if type(self.nixl_agent_name) is not str or len(self.nixl_agent_name) == 0:
            raise ValueError("nixl_agent_name must be nonempty")
        try:
            encoded_name = self.nixl_agent_name.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("nixl_agent_name must be ASCII") from error
        if len(encoded_name) > _NIXL_AGENT_NAME_MAX_BYTES:
            raise ValueError("nixl_agent_name is too large")
        _require_sha256(
            self.nixl_agent_metadata_sha256,
            "nixl_agent_metadata_sha256",
        )

    @property
    def key(self) -> tuple[str, int]:
        """Return the static service-rank lookup key.

        :returns: Service identifier and TP rank.
        """

        return self.service_id, self.tensor_parallel_rank

    @property
    def terminal_identity(self) -> TerminalProcessIdentity:
        """Return the runtime owner/issuer identity.

        :returns: Exact terminal process identity.
        """

        return TerminalProcessIdentity(
            process_generation=self.process_generation,
            role=self.role,
            tp_rank=self.tensor_parallel_rank,
            tp_size=self.tensor_parallel_size,
        )


def _rank_order(
    rank: TerminalStartupRankAdvertisement,
) -> tuple[int, str, int]:
    """Return the canonical matrix rank order.

    :param rank: Candidate rank identity.
    :returns: Source-first service and rank order.
    """

    return _role_order(rank.role), rank.service_id, rank.tensor_parallel_rank


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupCohortMatrix:
    """Immutable observed native rank matrix for one deployment epoch.

    :ivar group_id: Static ownership group.
    :ivar cohort_sha256: Canonical launcher cohort digest.
    :ivar ranks: Complete source-first observed rank population.
    """

    group_id: str
    cohort_sha256: bytes
    ranks: tuple[TerminalStartupRankAdvertisement, ...]

    def __post_init__(self) -> None:
        """Validate canonical ordering and global native identity uniqueness."""

        if (
            type(self.group_id) is not str
            or _IDENTIFIER.fullmatch(self.group_id) is None
        ):
            raise ValueError("group_id is not canonical")
        _require_sha256(self.cohort_sha256, "cohort_sha256")
        if type(self.ranks) is not tuple or len(self.ranks) == 0:
            raise ValueError("ranks must be a nonempty tuple")
        if any(
            type(rank) is not TerminalStartupRankAdvertisement for rank in self.ranks
        ):
            raise TypeError("ranks must contain TerminalStartupRankAdvertisement")
        if tuple(sorted(self.ranks, key=_rank_order)) != self.ranks:
            raise ValueError("ranks are not in canonical order")
        if any(
            rank.group_id != self.group_id or rank.cohort_sha256 != self.cohort_sha256
            for rank in self.ranks
        ):
            raise ValueError("matrix ranks belong to another deployment epoch")
        keys = tuple(rank.key for rank in self.ranks)
        generations = tuple(rank.process_generation for rank in self.ranks)
        agent_names = tuple(rank.nixl_agent_name for rank in self.ranks)
        metadata_digests = tuple(rank.nixl_agent_metadata_sha256 for rank in self.ranks)
        if len(set(keys)) != len(keys):
            raise ValueError("matrix service-rank keys must be unique")
        if len(set(generations)) != len(generations):
            raise ValueError("matrix process generations must be unique")
        if len(set(agent_names)) != len(agent_names):
            raise ValueError("matrix NIXL agent names must be unique")
        if len(set(metadata_digests)) != len(metadata_digests):
            raise ValueError("matrix NIXL metadata digests must be unique")

    @property
    def digest(self) -> bytes:
        """Return the exact observed-matrix digest.

        :returns: SHA-256 of canonical matrix bytes.
        """

        return hashlib.sha256(encode_terminal_startup_cohort_matrix(self)).digest()

    def require_expectation(
        self,
        expectation: TerminalStartupCohortExpectation,
    ) -> None:
        """Prove complete correspondence to static membership.

        :param expectation: Launcher-authenticated cohort expectation.
        :raises TerminalStartupCohortError: If membership is missing or drifts.
        """

        if type(expectation) is not TerminalStartupCohortExpectation:
            raise TypeError("expectation must be TerminalStartupCohortExpectation")
        if (
            self.group_id != expectation.group_id
            or self.cohort_sha256 != expectation.cohort_sha256
            or len(self.ranks) != expectation.expected_rank_count
        ):
            raise TerminalStartupCohortError(
                "observed matrix differs from static cohort population"
            )
        for rank in self.ranks:
            expectation.require_advertisement(rank)
        expected_keys = {
            (service.service_id, rank)
            for service in expectation.services
            for rank in range(service.tensor_parallel_size)
        }
        if {rank.key for rank in self.ranks} != expected_keys:
            raise TerminalStartupCohortError(
                "observed matrix is incomplete or contains an extra rank"
            )

    def rank(
        self,
        service_id: str,
        tensor_parallel_rank: int,
    ) -> TerminalStartupRankAdvertisement:
        """Resolve one exact static rank.

        :param service_id: Launcher service identifier.
        :param tensor_parallel_rank: Rank within that service.
        :returns: Exact observed rank identity.
        :raises TerminalStartupCohortError: If the rank is absent.
        """

        if type(service_id) is not str:
            raise TypeError("service_id must be a string")
        if type(tensor_parallel_rank) is not int or tensor_parallel_rank < 0:
            raise ValueError("tensor_parallel_rank must be nonnegative")
        matches = tuple(
            rank
            for rank in self.ranks
            if rank.key == (service_id, tensor_parallel_rank)
        )
        if len(matches) != 1:
            raise TerminalStartupCohortError("startup matrix rank is absent")
        return matches[0]

    def rank_for_process_generation(
        self,
        process_generation: bytes,
    ) -> TerminalStartupRankAdvertisement:
        """Resolve the sole rank owning one native process generation.

        :param process_generation: Exact native incarnation.
        :returns: Static-member-bound observed rank.
        :raises TerminalStartupCohortError: If the generation is stale or unknown.
        """

        _require_uuid_bytes(process_generation, "process_generation")
        matches = tuple(
            rank for rank in self.ranks if rank.process_generation == process_generation
        )
        if len(matches) != 1:
            raise TerminalStartupCohortError(
                "native process generation is absent from the sealed matrix"
            )
        return matches[0]


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupCohortRegistrySnapshot:
    """Immutable startup-registry health and population.

    :ivar disposition: Open, sealed, or failed state.
    :ivar registered_rank_count: Exact accepted rank population.
    :ivar expected_rank_count: Static required rank population.
    :ivar matrix_digest: Sealed observed-matrix digest, when available.
    :ivar failure_reason: Sticky failure evidence, when failed.
    """

    disposition: TerminalStartupCohortDisposition
    registered_rank_count: int
    expected_rank_count: int
    matrix_digest: bytes | None
    failure_reason: str | None


class TerminalStartupCohortRegistry:
    """Event-driven join barrier for one immutable deployment epoch."""

    _expectation: TerminalStartupCohortExpectation
    _timeout_seconds: float
    _registrations: dict[tuple[str, int], TerminalStartupRankAdvertisement]
    _disposition: TerminalStartupCohortDisposition
    _matrix: TerminalStartupCohortMatrix | None
    _failure_reason: str | None
    _deadline_ns: int | None
    _condition: threading.Condition

    def __init__(
        self,
        expectation: TerminalStartupCohortExpectation,
        timeout_seconds: float,
    ) -> None:
        """Construct one dormant exact-membership registry.

        The deadline begins with the first accepted rank, not server creation,
        because model loading legitimately precedes the startup handshake.

        :param expectation: Launcher-authenticated static membership.
        :param timeout_seconds: Hash-bound maximum join duration.
        """

        if type(expectation) is not TerminalStartupCohortExpectation:
            raise TypeError("expectation must be TerminalStartupCohortExpectation")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self._expectation = expectation
        self._timeout_seconds = float(timeout_seconds)
        self._registrations = {}
        self._disposition = TerminalStartupCohortDisposition.OPEN
        self._matrix = None
        self._failure_reason = None
        self._deadline_ns = None
        self._condition = threading.Condition()

    @property
    def expectation(self) -> TerminalStartupCohortExpectation:
        """Return static membership bound to this registry.

        :returns: Immutable launcher expectation.
        """

        return self._expectation

    @property
    def timeout_seconds(self) -> float:
        """Return the hash-bound startup control deadline.

        :returns: Positive finite timeout in seconds.
        """

        return self._timeout_seconds

    def sealed_matrix_for(
        self,
        advertisement: TerminalStartupRankAdvertisement,
    ) -> TerminalStartupCohortMatrix:
        """Authenticate one exact rank against the already sealed matrix.

        This accessor cannot join, replace, or otherwise mutate membership. It
        lets later startup phases reuse the original generation authority.

        :param advertisement: Exact rank requesting a later startup phase.
        :returns: Complete immutable observed matrix.
        :raises TerminalStartupCohortError: If the cohort is unsealed or differs.
        """

        if type(advertisement) is not TerminalStartupRankAdvertisement:
            raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
        with self._condition:
            if self._disposition is TerminalStartupCohortDisposition.FAILED:
                self._raise_failure_locked()
            if (
                self._disposition is not TerminalStartupCohortDisposition.SEALED
                or self._matrix is None
            ):
                raise TerminalStartupCohortError(
                    "terminal startup cohort is not sealed"
                )
            if self._matrix.rank(*advertisement.key) != advertisement:
                raise TerminalStartupCohortError(
                    "startup phase requester differs from the sealed matrix"
                )
            return self._matrix

    def register_and_wait(
        self,
        advertisement: TerminalStartupRankAdvertisement,
    ) -> TerminalStartupCohortMatrix:
        """Register one rank and block on event-driven cohort sealing.

        No fixed-cadence query occurs. Every waiter sleeps on the registry
        condition and is woken by another rank, collective failure, or timeout.

        :param advertisement: Exact native identity for one static rank.
        :returns: Complete sealed observed matrix shared by every rank.
        :raises TerminalStartupCohortError: If membership conflicts or times out.
        """

        if type(advertisement) is not TerminalStartupRankAdvertisement:
            raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
        with self._condition:
            if self._disposition is TerminalStartupCohortDisposition.FAILED:
                self._raise_failure_locked()
            if (
                self._deadline_ns is not None
                and time.monotonic_ns() >= self._deadline_ns
                and self._disposition is TerminalStartupCohortDisposition.OPEN
            ):
                self._fail_locked(
                    "terminal startup cohort did not reach complete membership"
                )
                self._raise_failure_locked()
            try:
                self._expectation.require_advertisement(advertisement)
            except TerminalStartupCohortError as error:
                self._fail_locked(str(error))
                raise
            existing = self._registrations.get(advertisement.key)
            if existing is not None and existing != advertisement:
                self._fail_locked(
                    "static service rank attempted to change native generation"
                )
                self._raise_failure_locked()
            if self._disposition is TerminalStartupCohortDisposition.SEALED:
                if existing != advertisement or self._matrix is None:
                    self._fail_locked(
                        "sealed startup cohort rejected a replacement process"
                    )
                    self._raise_failure_locked()
                return self._matrix
            if existing is None:
                if any(
                    rank.process_generation == advertisement.process_generation
                    for rank in self._registrations.values()
                ):
                    self._fail_locked(
                        "native process generation was reused by another rank"
                    )
                    self._raise_failure_locked()
                if any(
                    rank.nixl_agent_name == advertisement.nixl_agent_name
                    for rank in self._registrations.values()
                ):
                    self._fail_locked("NIXL agent name was reused by another rank")
                    self._raise_failure_locked()
                self._registrations[advertisement.key] = advertisement
                if self._deadline_ns is None:
                    self._deadline_ns = time.monotonic_ns() + int(
                        self._timeout_seconds * 1_000_000_000
                    )
            if len(self._registrations) == self._expectation.expected_rank_count:
                self._seal_locked()
            while self._disposition is TerminalStartupCohortDisposition.OPEN:
                assert self._deadline_ns is not None
                remaining_ns = self._deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    self._fail_locked(
                        "terminal startup cohort did not reach complete membership"
                    )
                    break
                self._condition.wait(remaining_ns / 1_000_000_000)
            if self._disposition is TerminalStartupCohortDisposition.FAILED:
                self._raise_failure_locked()
            if self._matrix is None:
                raise TerminalStartupCohortError(
                    "sealed startup cohort has no observed matrix"
                )
            return self._matrix

    def fail(self, reason: str) -> None:
        """Fail the complete epoch and wake every joiner.

        :param reason: Stable fail-closed evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a nonempty string")
        with self._condition:
            self._fail_locked(reason)

    def snapshot(self) -> TerminalStartupCohortRegistrySnapshot:
        """Return exact registry health without mutating the barrier.

        :returns: Immutable health and population.
        """

        with self._condition:
            return TerminalStartupCohortRegistrySnapshot(
                disposition=self._disposition,
                registered_rank_count=len(self._registrations),
                expected_rank_count=self._expectation.expected_rank_count,
                matrix_digest=(None if self._matrix is None else self._matrix.digest),
                failure_reason=self._failure_reason,
            )

    def _seal_locked(self) -> None:
        """Seal complete membership under the registry condition."""

        matrix = TerminalStartupCohortMatrix(
            group_id=self._expectation.group_id,
            cohort_sha256=self._expectation.cohort_sha256,
            ranks=tuple(sorted(self._registrations.values(), key=_rank_order)),
        )
        matrix.require_expectation(self._expectation)
        self._matrix = matrix
        self._disposition = TerminalStartupCohortDisposition.SEALED
        self._condition.notify_all()

    def _fail_locked(self, reason: str) -> None:
        """Enter sticky collective failure under the registry condition.

        :param reason: Stable failure evidence.
        """

        if self._disposition is TerminalStartupCohortDisposition.FAILED:
            return
        self._failure_reason = reason
        self._disposition = TerminalStartupCohortDisposition.FAILED
        self._condition.notify_all()

    def _raise_failure_locked(self) -> None:
        """Raise the sticky failure under the registry condition."""

        reason = self._failure_reason
        if reason is None:
            reason = "terminal startup cohort failed without evidence"
        raise TerminalStartupCohortError(reason)


def _advertisement_payload(
    advertisement: TerminalStartupRankAdvertisement,
) -> dict[str, object]:
    """Return canonical JSON-compatible rank fields.

    :param advertisement: Exact rank identity.
    :returns: Frozen field-order payload.
    """

    return {
        "group_id": advertisement.group_id,
        "cohort_sha256": advertisement.cohort_sha256.hex(),
        "service_id": advertisement.service_id,
        "service_origin": advertisement.service_origin,
        "role": advertisement.role.value,
        "launch_instance_id": str(uuid.UUID(bytes=advertisement.launch_instance_id)),
        "tensor_parallel_rank": advertisement.tensor_parallel_rank,
        "tensor_parallel_size": advertisement.tensor_parallel_size,
        "process_generation": str(uuid.UUID(bytes=advertisement.process_generation)),
        "nixl_agent_name": advertisement.nixl_agent_name,
        "nixl_agent_metadata_sha256": (advertisement.nixl_agent_metadata_sha256.hex()),
    }


def _require_exact_fields(
    payload: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    """Reject unknown or missing JSON object fields.

    :param payload: Parsed JSON object.
    :param expected: Exact allowed field set.
    :param label: Reader-facing object label.
    """

    if set(payload) != expected:
        raise TerminalStartupCohortError(f"{label} field set is invalid")


def _parse_uuid(value: object, label: str) -> bytes:
    """Parse one canonical non-nil UUID string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: UUID bytes.
    """

    if type(value) is not str:
        raise TerminalStartupCohortError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise TerminalStartupCohortError(f"{label} is not a UUID") from error
    if parsed.int == 0 or str(parsed) != value:
        raise TerminalStartupCohortError(f"{label} is not canonical")
    return parsed.bytes


def _parse_sha256(value: object, label: str) -> bytes:
    """Parse one lowercase SHA-256 string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: Binary digest.
    """

    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise TerminalStartupCohortError(f"{label} is not canonical")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise TerminalStartupCohortError(f"{label} is not hexadecimal") from error
    _require_sha256(parsed, label)
    return parsed


def _parse_advertisement_payload(
    payload: dict[str, object],
) -> TerminalStartupRankAdvertisement:
    """Decode one strict advertisement object.

    :param payload: Parsed JSON object without schema wrapper.
    :returns: Validated rank identity.
    """

    _require_exact_fields(
        payload,
        {
            "group_id",
            "cohort_sha256",
            "service_id",
            "service_origin",
            "role",
            "launch_instance_id",
            "tensor_parallel_rank",
            "tensor_parallel_size",
            "process_generation",
            "nixl_agent_name",
            "nixl_agent_metadata_sha256",
        },
        "terminal startup advertisement",
    )
    role_value = payload["role"]
    if type(role_value) is not str:
        raise TerminalStartupCohortError("advertisement role must be a string")
    try:
        role = TerminalOwnerRole(role_value)
    except ValueError as error:
        raise TerminalStartupCohortError("advertisement role is invalid") from error
    tensor_parallel_rank = payload["tensor_parallel_rank"]
    tensor_parallel_size = payload["tensor_parallel_size"]
    if type(tensor_parallel_rank) is not int or type(tensor_parallel_size) is not int:
        raise TerminalStartupCohortError("advertisement ranks must be integers")
    group_id = payload["group_id"]
    service_id = payload["service_id"]
    service_origin = payload["service_origin"]
    agent_name = payload["nixl_agent_name"]
    if (
        type(group_id) is not str
        or type(service_id) is not str
        or type(service_origin) is not str
    ):
        raise TerminalStartupCohortError("advertisement identifiers must be strings")
    if type(agent_name) is not str:
        raise TerminalStartupCohortError("advertisement agent name must be a string")
    try:
        return TerminalStartupRankAdvertisement(
            group_id=group_id,
            cohort_sha256=_parse_sha256(
                payload["cohort_sha256"],
                "cohort_sha256",
            ),
            service_id=service_id,
            service_origin=service_origin,
            role=role,
            launch_instance_id=_parse_uuid(
                payload["launch_instance_id"],
                "launch_instance_id",
            ),
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_size=tensor_parallel_size,
            process_generation=_parse_uuid(
                payload["process_generation"],
                "process_generation",
            ),
            nixl_agent_name=agent_name,
            nixl_agent_metadata_sha256=_parse_sha256(
                payload["nixl_agent_metadata_sha256"],
                "nixl_agent_metadata_sha256",
            ),
        )
    except (TypeError, ValueError) as error:
        raise TerminalStartupCohortError(
            "terminal startup advertisement fields are invalid"
        ) from error


def _load_json(payload: bytes) -> dict[str, object]:
    """Decode one bounded duplicate-free JSON object.

    :param payload: Exact wire bytes.
    :returns: Parsed object.
    """

    if (
        type(payload) is not bytes
        or not 0 < len(payload) <= TERMINAL_STARTUP_WIRE_MAX_BYTES
    ):
        raise TerminalStartupCohortError("terminal startup payload size is invalid")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalStartupCohortError(
                    f"duplicate terminal startup JSON field: {key}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalStartupCohortError(
            "terminal startup payload is not valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise TerminalStartupCohortError("terminal startup payload must be an object")
    return decoded


def encode_terminal_startup_rank_advertisement(
    advertisement: TerminalStartupRankAdvertisement,
) -> bytes:
    """Encode one canonical rank advertisement.

    :param advertisement: Exact observed rank identity.
    :returns: Compact UTF-8 JSON without a newline.
    """

    if type(advertisement) is not TerminalStartupRankAdvertisement:
        raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
    payload = {
        "schema": TERMINAL_STARTUP_ADVERTISEMENT_SCHEMA,
        "rank": _advertisement_payload(advertisement),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def decode_terminal_startup_rank_advertisement(
    payload: bytes,
) -> TerminalStartupRankAdvertisement:
    """Decode one canonical rank advertisement.

    :param payload: Exact wire bytes.
    :returns: Validated observed rank identity.
    :raises TerminalStartupCohortError: If bytes are malformed or noncanonical.
    """

    decoded = _load_json(payload)
    _require_exact_fields(decoded, {"schema", "rank"}, "advertisement wrapper")
    if decoded["schema"] != TERMINAL_STARTUP_ADVERTISEMENT_SCHEMA:
        raise TerminalStartupCohortError("unsupported advertisement schema")
    rank_payload = decoded["rank"]
    if type(rank_payload) is not dict:
        raise TerminalStartupCohortError("advertisement rank must be an object")
    rank = _parse_advertisement_payload(rank_payload)
    if encode_terminal_startup_rank_advertisement(rank) != payload:
        raise TerminalStartupCohortError("advertisement JSON is not canonical")
    return rank


def encode_terminal_startup_cohort_matrix(
    matrix: TerminalStartupCohortMatrix,
) -> bytes:
    """Encode one canonical sealed matrix.

    :param matrix: Complete observed cohort.
    :returns: Compact UTF-8 JSON without a newline.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    payload = {
        "schema": TERMINAL_STARTUP_COHORT_SCHEMA,
        "group_id": matrix.group_id,
        "cohort_sha256": matrix.cohort_sha256.hex(),
        "ranks": [_advertisement_payload(rank) for rank in matrix.ranks],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def decode_terminal_startup_cohort_matrix(
    payload: bytes,
) -> TerminalStartupCohortMatrix:
    """Decode one canonical sealed matrix.

    :param payload: Exact wire bytes.
    :returns: Complete observed cohort.
    :raises TerminalStartupCohortError: If bytes are malformed or noncanonical.
    """

    decoded = _load_json(payload)
    _require_exact_fields(
        decoded,
        {"schema", "group_id", "cohort_sha256", "ranks"},
        "startup matrix",
    )
    if decoded["schema"] != TERMINAL_STARTUP_COHORT_SCHEMA:
        raise TerminalStartupCohortError("unsupported startup matrix schema")
    group_id = decoded["group_id"]
    raw_ranks = decoded["ranks"]
    if type(group_id) is not str or type(raw_ranks) is not list:
        raise TerminalStartupCohortError("startup matrix fields are malformed")
    ranks: list[TerminalStartupRankAdvertisement] = []
    for raw_rank in raw_ranks:
        if type(raw_rank) is not dict:
            raise TerminalStartupCohortError("startup matrix rank is not an object")
        ranks.append(_parse_advertisement_payload(raw_rank))
    try:
        matrix = TerminalStartupCohortMatrix(
            group_id=group_id,
            cohort_sha256=_parse_sha256(decoded["cohort_sha256"], "cohort_sha256"),
            ranks=tuple(ranks),
        )
    except (TypeError, ValueError) as error:
        raise TerminalStartupCohortError(
            "terminal startup matrix fields are invalid"
        ) from error
    if encode_terminal_startup_cohort_matrix(matrix) != payload:
        raise TerminalStartupCohortError("startup matrix JSON is not canonical")
    return matrix
