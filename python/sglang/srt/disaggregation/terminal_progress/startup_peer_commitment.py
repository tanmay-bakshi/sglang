import dataclasses
import enum
import hashlib
import json
import math
import threading
import time

from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TERMINAL_STARTUP_WIRE_MAX_BYTES,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
    decode_terminal_startup_rank_advertisement,
    encode_terminal_startup_rank_advertisement,
)

TERMINAL_STARTUP_PEER_COMMITMENT_SCHEMA: str = (
    "packed-terminal-startup-peer-commitment-v1"
)
TERMINAL_STARTUP_PEER_COMMITMENT_MATRIX_SCHEMA: str = (
    "packed-terminal-startup-peer-commitment-matrix-v1"
)
TERMINAL_STARTUP_PEER_ROSTER_DIGEST_DOMAIN: bytes = (
    b"packed-terminal-startup-opposite-role-roster-v1\x00"
)

_SHA256_BYTES = hashlib.sha256().digest_size


class TerminalStartupPeerCommitmentError(RuntimeError):
    """Invalid or failed terminal startup peer commitment barrier."""


class TerminalStartupPeerCommitmentDisposition(enum.StrEnum):
    """Lifecycle of one generation-bound peer commitment registry."""

    OPEN = "open"
    SEALED = "sealed"
    FAILED = "failed"


def _role_order(role: TerminalOwnerRole) -> int:
    """Return source-first role order.

    :param role: Source or decode owner role.
    :returns: Canonical integer order.
    """

    if role is TerminalOwnerRole.SOURCE:
        return 0
    return 1


def _rank_order(
    rank: TerminalStartupRankAdvertisement,
) -> tuple[int, str, int]:
    """Return canonical startup-rank order.

    :param rank: Exact startup matrix row.
    :returns: Source-first service and TP-rank order.
    """

    return _role_order(rank.role), rank.service_id, rank.tensor_parallel_rank


def _commitment_order(
    commitment: "TerminalStartupPeerCommitment",
) -> tuple[int, str, int]:
    """Return canonical commitment order.

    :param commitment: Per-rank peer enrollment commitment.
    :returns: Source-first service and TP-rank order.
    """

    return _rank_order(commitment.local_rank)


def _require_sha256(value: bytes, label: str) -> None:
    """Require one binary SHA-256 value.

    :param value: Candidate digest.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != _SHA256_BYTES:
        raise ValueError(f"{label} must contain 32 bytes")


def terminal_startup_peer_roster_sha256(
    ranks: tuple[TerminalStartupRankAdvertisement, ...],
) -> bytes:
    """Digest one canonical, single-role startup peer roster.

    The length-prefixed canonical advertisement bytes bind every peer's native
    process generation, agent identity, metadata digest, and static service row.

    :param ranks: Complete canonical ranks belonging to one owner role.
    :returns: Domain-separated SHA-256 roster digest.
    """

    if type(ranks) is not tuple or len(ranks) == 0:
        raise ValueError("peer roster must be a nonempty tuple")
    if any(type(rank) is not TerminalStartupRankAdvertisement for rank in ranks):
        raise TypeError("peer roster must contain startup rank advertisements")
    if tuple(sorted(ranks, key=_rank_order)) != ranks:
        raise ValueError("peer roster is not in canonical order")
    if len({rank.key for rank in ranks}) != len(ranks):
        raise ValueError("peer roster contains a duplicate service rank")
    if len({rank.process_generation for rank in ranks}) != len(ranks):
        raise ValueError("peer roster contains a duplicate process generation")
    if len({rank.role for rank in ranks}) != 1:
        raise ValueError("peer roster must contain exactly one owner role")

    hasher = hashlib.sha256()
    hasher.update(TERMINAL_STARTUP_PEER_ROSTER_DIGEST_DOMAIN)
    hasher.update(len(ranks).to_bytes(8, byteorder="big", signed=False))
    for rank in ranks:
        encoded = encode_terminal_startup_rank_advertisement(rank)
        hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        hasher.update(encoded)
    return hasher.digest()


def _opposite_role_roster(
    matrix: TerminalStartupCohortMatrix,
    local_role: TerminalOwnerRole,
) -> tuple[TerminalStartupRankAdvertisement, ...]:
    """Resolve the exact opposite-role roster from one sealed startup matrix.

    :param matrix: Complete generation-authenticated startup matrix.
    :param local_role: Role issuing the commitment.
    :returns: Canonically ordered opposite-role ranks.
    """

    return tuple(rank for rank in matrix.ranks if rank.role is not local_role)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupPeerCommitment:
    """One rank's immutable proof of complete opposite-role enrollment.

    :ivar startup_matrix_sha256: Digest of the exact sealed startup matrix.
    :ivar local_rank: Exact generation-bearing local matrix row.
    :ivar opposite_role_rank_count: Complete opposite-role peer population.
    :ivar opposite_role_roster_sha256: Digest of all enrolled opposite-role rows.
    """

    startup_matrix_sha256: bytes
    local_rank: TerminalStartupRankAdvertisement
    opposite_role_rank_count: int
    opposite_role_roster_sha256: bytes

    def __post_init__(self) -> None:
        """Validate one structurally complete commitment."""

        _require_sha256(self.startup_matrix_sha256, "startup_matrix_sha256")
        if type(self.local_rank) is not TerminalStartupRankAdvertisement:
            raise TypeError("local_rank must be TerminalStartupRankAdvertisement")
        if (
            type(self.opposite_role_rank_count) is not int
            or self.opposite_role_rank_count <= 0
        ):
            raise ValueError("opposite_role_rank_count must be positive")
        _require_sha256(
            self.opposite_role_roster_sha256,
            "opposite_role_roster_sha256",
        )

    @property
    def key(self) -> tuple[str, int]:
        """Return the exact static service-rank key.

        :returns: Service identifier and TP rank.
        """

        return self.local_rank.key

    @property
    def digest(self) -> bytes:
        """Return the canonical commitment digest.

        :returns: SHA-256 of the exact commitment wire bytes.
        """

        return hashlib.sha256(encode_terminal_startup_peer_commitment(self)).digest()


def build_terminal_startup_peer_commitment(
    matrix: TerminalStartupCohortMatrix,
    local_rank: TerminalStartupRankAdvertisement,
    enrolled_opposite_role_ranks: tuple[TerminalStartupRankAdvertisement, ...],
) -> TerminalStartupPeerCommitment:
    """Build a commitment after exact opposite-role enrollment completes.

    The explicit enrolled roster keeps construction coupled to the enrollment
    result. Native handle and route retention remain the caller's responsibility.

    :param matrix: Complete generation-authenticated startup matrix.
    :param local_rank: Exact local row from that matrix.
    :param enrolled_opposite_role_ranks: Canonical rows actually enrolled locally.
    :returns: Exact generation-bound peer commitment.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    if type(local_rank) is not TerminalStartupRankAdvertisement:
        raise TypeError("local_rank must be TerminalStartupRankAdvertisement")
    if matrix.rank(*local_rank.key) != local_rank:
        raise TerminalStartupPeerCommitmentError(
            "local commitment row differs from the sealed startup matrix"
        )
    expected_roster = _opposite_role_roster(matrix, local_rank.role)
    if len(expected_roster) == 0:
        raise TerminalStartupPeerCommitmentError(
            "startup matrix has no opposite-role peer roster"
        )
    if enrolled_opposite_role_ranks != expected_roster:
        raise TerminalStartupPeerCommitmentError(
            "enrolled peer roster differs from the sealed startup matrix"
        )
    return TerminalStartupPeerCommitment(
        startup_matrix_sha256=matrix.digest,
        local_rank=local_rank,
        opposite_role_rank_count=len(expected_roster),
        opposite_role_roster_sha256=terminal_startup_peer_roster_sha256(
            expected_roster
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupPeerCommitmentMatrix:
    """Sealed all-rank proof of complete cross-role startup enrollment.

    :ivar startup_matrix_sha256: Digest of the exact startup matrix.
    :ivar commitments: Complete canonical per-rank commitment population.
    """

    startup_matrix_sha256: bytes
    commitments: tuple[TerminalStartupPeerCommitment, ...]

    def __post_init__(self) -> None:
        """Validate canonical ordering and role-wide roster agreement."""

        _require_sha256(self.startup_matrix_sha256, "startup_matrix_sha256")
        if type(self.commitments) is not tuple or len(self.commitments) == 0:
            raise ValueError("commitments must be a nonempty tuple")
        if any(
            type(commitment) is not TerminalStartupPeerCommitment
            for commitment in self.commitments
        ):
            raise TypeError(
                "commitments must contain TerminalStartupPeerCommitment values"
            )
        if tuple(sorted(self.commitments, key=_commitment_order)) != self.commitments:
            raise ValueError("commitments are not in canonical order")
        if any(
            commitment.startup_matrix_sha256 != self.startup_matrix_sha256
            for commitment in self.commitments
        ):
            raise ValueError("commitments bind different startup matrices")
        keys = tuple(commitment.key for commitment in self.commitments)
        if len(set(keys)) != len(keys):
            raise ValueError("commitment service-rank keys must be unique")
        if {commitment.local_rank.role for commitment in self.commitments} != {
            TerminalOwnerRole.SOURCE,
            TerminalOwnerRole.DECODE,
        }:
            raise ValueError("commitment matrix requires source and decode ranks")
        for role in TerminalOwnerRole:
            role_commitments = tuple(
                commitment
                for commitment in self.commitments
                if commitment.local_rank.role is role
            )
            roster_claims = {
                (
                    commitment.opposite_role_rank_count,
                    commitment.opposite_role_roster_sha256,
                )
                for commitment in role_commitments
            }
            if len(roster_claims) != 1:
                raise ValueError(
                    "commitments of one local role disagree on the peer roster"
                )

    @property
    def digest(self) -> bytes:
        """Return the exact sealed commitment-matrix digest.

        :returns: SHA-256 of canonical matrix bytes.
        """

        return hashlib.sha256(
            encode_terminal_startup_peer_commitment_matrix(self)
        ).digest()

    def commitment(
        self,
        service_id: str,
        tensor_parallel_rank: int,
    ) -> TerminalStartupPeerCommitment:
        """Resolve one exact per-rank commitment.

        :param service_id: Static service identifier.
        :param tensor_parallel_rank: Rank within the service.
        :returns: Exact commitment for that rank.
        :raises TerminalStartupPeerCommitmentError: If the rank is absent.
        """

        if type(service_id) is not str:
            raise TypeError("service_id must be a string")
        if type(tensor_parallel_rank) is not int or tensor_parallel_rank < 0:
            raise ValueError("tensor_parallel_rank must be nonnegative")
        matches = tuple(
            commitment
            for commitment in self.commitments
            if commitment.key == (service_id, tensor_parallel_rank)
        )
        if len(matches) != 1:
            raise TerminalStartupPeerCommitmentError(
                "startup peer commitment is absent"
            )
        return matches[0]

    def require_startup_matrix(self, matrix: TerminalStartupCohortMatrix) -> None:
        """Prove every commitment against the exact sealed startup matrix.

        :param matrix: Generation-authenticated startup matrix.
        :raises TerminalStartupPeerCommitmentError: If any row or roster differs.
        """

        if type(matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("matrix must be TerminalStartupCohortMatrix")
        if self.startup_matrix_sha256 != matrix.digest or len(self.commitments) != len(
            matrix.ranks
        ):
            raise TerminalStartupPeerCommitmentError(
                "commitment matrix differs from the sealed startup population"
            )
        if {commitment.key for commitment in self.commitments} != {
            rank.key for rank in matrix.ranks
        }:
            raise TerminalStartupPeerCommitmentError(
                "commitment matrix is missing or contains an extra rank"
            )
        for commitment in self.commitments:
            try:
                expected_local_rank = matrix.rank(*commitment.key)
            except RuntimeError as error:
                raise TerminalStartupPeerCommitmentError(
                    "commitment local rank is absent from the startup matrix"
                ) from error
            if commitment.local_rank != expected_local_rank:
                raise TerminalStartupPeerCommitmentError(
                    "commitment local row differs from the sealed startup matrix"
                )
            expected_roster = _opposite_role_roster(
                matrix,
                commitment.local_rank.role,
            )
            if len(expected_roster) == 0:
                raise TerminalStartupPeerCommitmentError(
                    "startup matrix has no opposite-role peer roster"
                )
            if (
                commitment.opposite_role_rank_count != len(expected_roster)
                or commitment.opposite_role_roster_sha256
                != terminal_startup_peer_roster_sha256(expected_roster)
            ):
                raise TerminalStartupPeerCommitmentError(
                    "commitment peer roster differs from the sealed startup matrix"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupPeerCommitmentRegistrySnapshot:
    """Immutable peer-commitment barrier state.

    :ivar disposition: Open, sealed, or failed state.
    :ivar registered_rank_count: Exact accepted commitment population.
    :ivar expected_rank_count: Required startup matrix population.
    :ivar commitment_matrix_sha256: Sealed commitment-matrix digest, if available.
    :ivar failure_reason: Sticky collective failure evidence, if failed.
    """

    disposition: TerminalStartupPeerCommitmentDisposition
    registered_rank_count: int
    expected_rank_count: int
    commitment_matrix_sha256: bytes | None
    failure_reason: str | None


class TerminalStartupPeerCommitmentRegistry:
    """Event-driven all-rank barrier for exact startup peer enrollment."""

    _startup_matrix: TerminalStartupCohortMatrix
    _timeout_seconds: float
    _expected: dict[tuple[str, int], TerminalStartupPeerCommitment]
    _registrations: dict[tuple[str, int], TerminalStartupPeerCommitment]
    _registration_bytes: dict[tuple[str, int], bytes]
    _disposition: TerminalStartupPeerCommitmentDisposition
    _commitment_matrix: TerminalStartupPeerCommitmentMatrix | None
    _failure_reason: str | None
    _deadline_ns: int | None
    _condition: threading.Condition

    def __init__(
        self,
        startup_matrix: TerminalStartupCohortMatrix,
        timeout_seconds: float,
    ) -> None:
        """Construct one dormant, exact-matrix peer commitment barrier.

        The deadline begins with the first accepted commitment. Peer enrollment
        legitimately happens after the startup cohort itself has sealed.

        :param startup_matrix: Complete generation-authenticated rank matrix.
        :param timeout_seconds: Hash-bound maximum commitment duration.
        """

        if type(startup_matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("startup_matrix must be TerminalStartupCohortMatrix")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self._startup_matrix = startup_matrix
        self._timeout_seconds = float(timeout_seconds)
        self._expected = {}
        for rank in startup_matrix.ranks:
            expected_roster = _opposite_role_roster(startup_matrix, rank.role)
            if len(expected_roster) == 0:
                raise ValueError("startup matrix must contain both owner roles")
            self._expected[rank.key] = build_terminal_startup_peer_commitment(
                startup_matrix,
                rank,
                expected_roster,
            )
        self._registrations = {}
        self._registration_bytes = {}
        self._disposition = TerminalStartupPeerCommitmentDisposition.OPEN
        self._commitment_matrix = None
        self._failure_reason = None
        self._deadline_ns = None
        self._condition = threading.Condition()

    @property
    def startup_matrix(self) -> TerminalStartupCohortMatrix:
        """Return the exact matrix bound to this barrier.

        :returns: Immutable generation-authenticated startup matrix.
        """

        return self._startup_matrix

    def register_and_wait(
        self,
        commitment: TerminalStartupPeerCommitment,
    ) -> TerminalStartupPeerCommitmentMatrix:
        """Register one rank and wait for complete peer-enrollment commitment.

        Waiters sleep on a condition and wake only for registration, collective
        failure, sealing, or their shared deadline. No cadence polling occurs.

        :param commitment: Exact local peer-enrollment commitment.
        :returns: Shared sealed all-rank commitment matrix.
        :raises TerminalStartupPeerCommitmentError: If validation or sealing fails.
        """

        if type(commitment) is not TerminalStartupPeerCommitment:
            raise TypeError("commitment must be TerminalStartupPeerCommitment")
        encoded = encode_terminal_startup_peer_commitment(commitment)
        with self._condition:
            if self._disposition is TerminalStartupPeerCommitmentDisposition.FAILED:
                self._raise_failure_locked()
            if (
                self._deadline_ns is not None
                and time.monotonic_ns() >= self._deadline_ns
                and self._disposition is TerminalStartupPeerCommitmentDisposition.OPEN
            ):
                self._fail_locked(
                    "terminal startup peer commitments did not reach all ranks"
                )
                self._raise_failure_locked()

            existing_bytes = self._registration_bytes.get(commitment.key)
            if existing_bytes is not None and existing_bytes != encoded:
                self._fail_locked(
                    "startup rank submitted a conflicting peer commitment"
                )
                self._raise_failure_locked()
            if self._disposition is TerminalStartupPeerCommitmentDisposition.SEALED:
                if existing_bytes != encoded or self._commitment_matrix is None:
                    self._fail_locked(
                        "sealed peer commitment barrier rejected a replacement rank"
                    )
                    self._raise_failure_locked()
                return self._commitment_matrix

            if existing_bytes is None:
                try:
                    self._require_expected_locked(commitment)
                except TerminalStartupPeerCommitmentError as error:
                    self._fail_locked(str(error))
                    self._raise_failure_locked()
                self._registrations[commitment.key] = commitment
                self._registration_bytes[commitment.key] = encoded
                if self._deadline_ns is None:
                    self._deadline_ns = time.monotonic_ns() + int(
                        self._timeout_seconds * 1_000_000_000
                    )

            if len(self._registrations) == len(self._expected):
                self._seal_locked()
            while self._disposition is TerminalStartupPeerCommitmentDisposition.OPEN:
                assert self._deadline_ns is not None
                remaining_ns = self._deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    self._fail_locked(
                        "terminal startup peer commitments did not reach all ranks"
                    )
                    break
                self._condition.wait(remaining_ns / 1_000_000_000)
            if self._disposition is TerminalStartupPeerCommitmentDisposition.FAILED:
                self._raise_failure_locked()
            if self._commitment_matrix is None:
                raise TerminalStartupPeerCommitmentError(
                    "sealed peer commitment barrier has no commitment matrix"
                )
            return self._commitment_matrix

    def fail(self, reason: str) -> None:
        """Fail the complete commitment epoch and wake every waiter.

        :param reason: Stable fail-closed evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a nonempty string")
        with self._condition:
            self._fail_locked(reason)

    def snapshot(self) -> TerminalStartupPeerCommitmentRegistrySnapshot:
        """Return exact barrier health without mutating it.

        :returns: Immutable disposition and population snapshot.
        """

        with self._condition:
            return TerminalStartupPeerCommitmentRegistrySnapshot(
                disposition=self._disposition,
                registered_rank_count=len(self._registrations),
                expected_rank_count=len(self._expected),
                commitment_matrix_sha256=(
                    None
                    if self._commitment_matrix is None
                    else self._commitment_matrix.digest
                ),
                failure_reason=self._failure_reason,
            )

    def _require_expected_locked(
        self,
        commitment: TerminalStartupPeerCommitment,
    ) -> None:
        """Validate one new commitment under the registry condition.

        :param commitment: Candidate exact per-rank commitment.
        :raises TerminalStartupPeerCommitmentError: If any matrix field differs.
        """

        expected = self._expected.get(commitment.key)
        if expected is None:
            raise TerminalStartupPeerCommitmentError(
                "peer commitment rank is absent from the sealed startup matrix"
            )
        for accepted in self._registrations.values():
            if accepted.local_rank.role is commitment.local_rank.role and (
                accepted.opposite_role_rank_count != commitment.opposite_role_rank_count
                or accepted.opposite_role_roster_sha256
                != commitment.opposite_role_roster_sha256
            ):
                raise TerminalStartupPeerCommitmentError(
                    "commitments of one local role disagree on the peer roster"
                )
        if commitment.startup_matrix_sha256 != expected.startup_matrix_sha256:
            raise TerminalStartupPeerCommitmentError(
                "peer commitment binds another startup matrix"
            )
        if commitment.local_rank != expected.local_rank:
            raise TerminalStartupPeerCommitmentError(
                "commitment local row differs from the sealed startup matrix"
            )
        if (
            commitment.opposite_role_rank_count != expected.opposite_role_rank_count
            or commitment.opposite_role_roster_sha256
            != expected.opposite_role_roster_sha256
        ):
            raise TerminalStartupPeerCommitmentError(
                "commitment peer roster differs from the sealed startup matrix"
            )

    def _seal_locked(self) -> None:
        """Seal complete commitments under the registry condition."""

        commitment_matrix = TerminalStartupPeerCommitmentMatrix(
            startup_matrix_sha256=self._startup_matrix.digest,
            commitments=tuple(
                sorted(self._registrations.values(), key=_commitment_order)
            ),
        )
        commitment_matrix.require_startup_matrix(self._startup_matrix)
        self._commitment_matrix = commitment_matrix
        self._disposition = TerminalStartupPeerCommitmentDisposition.SEALED
        self._condition.notify_all()

    def _fail_locked(self, reason: str) -> None:
        """Enter sticky collective failure under the registry condition.

        :param reason: Stable failure evidence.
        """

        if self._disposition is TerminalStartupPeerCommitmentDisposition.FAILED:
            return
        self._failure_reason = reason
        self._disposition = TerminalStartupPeerCommitmentDisposition.FAILED
        self._condition.notify_all()

    def _raise_failure_locked(self) -> None:
        """Raise the sticky failure under the registry condition."""

        reason = self._failure_reason
        if reason is None:
            reason = "terminal startup peer commitment barrier failed without evidence"
        raise TerminalStartupPeerCommitmentError(reason)


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
        raise TerminalStartupPeerCommitmentError(f"{label} field set is invalid")


def _parse_sha256(value: object, label: str) -> bytes:
    """Parse one canonical lowercase SHA-256 string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: Binary SHA-256 value.
    """

    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise TerminalStartupPeerCommitmentError(f"{label} is not canonical")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise TerminalStartupPeerCommitmentError(
            f"{label} is not hexadecimal"
        ) from error
    if len(parsed) != _SHA256_BYTES:
        raise TerminalStartupPeerCommitmentError(f"{label} is not SHA-256")
    return parsed


def _load_json(payload: bytes) -> dict[str, object]:
    """Decode one bounded, duplicate-free JSON object.

    :param payload: Exact candidate wire bytes.
    :returns: Parsed top-level object.
    """

    if (
        type(payload) is not bytes
        or not 0 < len(payload) <= TERMINAL_STARTUP_WIRE_MAX_BYTES
    ):
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment payload size is invalid"
        )

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Build one object while rejecting duplicate field names.

        :param pairs: Decoder-preserved JSON field pairs.
        :returns: Unique-key JSON object.
        """

        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalStartupPeerCommitmentError(
                    f"duplicate startup peer commitment JSON field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        """Reject JSON constants outside the finite data model.

        :param value: Non-finite JSON constant spelling.
        :raises TerminalStartupPeerCommitmentError: For every input.
        """

        raise TerminalStartupPeerCommitmentError(
            f"non-finite startup peer commitment JSON value: {value}"
        )

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment payload is not valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment payload must be an object"
        )
    return decoded


def _advertisement_payload(
    rank: TerminalStartupRankAdvertisement,
) -> dict[str, object]:
    """Return the existing canonical advertisement wrapper as an object.

    :param rank: Exact startup matrix row.
    :returns: Canonical JSON-compatible advertisement wrapper.
    """

    decoded = json.loads(encode_terminal_startup_rank_advertisement(rank))
    if type(decoded) is not dict:
        raise RuntimeError("canonical startup rank advertisement is not an object")
    return decoded


def _parse_advertisement(value: object) -> TerminalStartupRankAdvertisement:
    """Parse one nested canonical startup rank advertisement.

    :param value: Candidate nested JSON value.
    :returns: Exact validated startup matrix row.
    """

    if type(value) is not dict:
        raise TerminalStartupPeerCommitmentError(
            "commitment local_rank must be an object"
        )
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    try:
        return decode_terminal_startup_rank_advertisement(encoded)
    except RuntimeError as error:
        raise TerminalStartupPeerCommitmentError(
            "commitment local_rank is invalid"
        ) from error


def _commitment_payload(
    commitment: TerminalStartupPeerCommitment,
) -> dict[str, object]:
    """Return canonical JSON-compatible commitment fields.

    :param commitment: Exact per-rank peer commitment.
    :returns: Frozen field-order payload without a schema wrapper.
    """

    return {
        "startup_matrix_sha256": commitment.startup_matrix_sha256.hex(),
        "local_rank": _advertisement_payload(commitment.local_rank),
        "opposite_role_rank_count": commitment.opposite_role_rank_count,
        "opposite_role_roster_sha256": (commitment.opposite_role_roster_sha256.hex()),
    }


def _parse_commitment_payload(
    payload: dict[str, object],
) -> TerminalStartupPeerCommitment:
    """Parse strict commitment fields without a schema wrapper.

    :param payload: Parsed commitment object.
    :returns: Exact validated per-rank commitment.
    """

    _require_exact_fields(
        payload,
        {
            "startup_matrix_sha256",
            "local_rank",
            "opposite_role_rank_count",
            "opposite_role_roster_sha256",
        },
        "startup peer commitment",
    )
    opposite_role_rank_count = payload["opposite_role_rank_count"]
    if type(opposite_role_rank_count) is not int:
        raise TerminalStartupPeerCommitmentError(
            "opposite_role_rank_count must be an integer"
        )
    try:
        return TerminalStartupPeerCommitment(
            startup_matrix_sha256=_parse_sha256(
                payload["startup_matrix_sha256"],
                "startup_matrix_sha256",
            ),
            local_rank=_parse_advertisement(payload["local_rank"]),
            opposite_role_rank_count=opposite_role_rank_count,
            opposite_role_roster_sha256=_parse_sha256(
                payload["opposite_role_roster_sha256"],
                "opposite_role_roster_sha256",
            ),
        )
    except (TypeError, ValueError) as error:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment fields are invalid"
        ) from error


def encode_terminal_startup_peer_commitment(
    commitment: TerminalStartupPeerCommitment,
) -> bytes:
    """Encode one canonical per-rank peer commitment.

    :param commitment: Exact generation-bound commitment.
    :returns: Compact UTF-8 JSON without a newline.
    """

    if type(commitment) is not TerminalStartupPeerCommitment:
        raise TypeError("commitment must be TerminalStartupPeerCommitment")
    payload = {
        "schema": TERMINAL_STARTUP_PEER_COMMITMENT_SCHEMA,
        **_commitment_payload(commitment),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def decode_terminal_startup_peer_commitment(
    payload: bytes,
) -> TerminalStartupPeerCommitment:
    """Decode one strict canonical per-rank peer commitment.

    :param payload: Exact wire bytes.
    :returns: Validated generation-bound commitment.
    :raises TerminalStartupPeerCommitmentError: If bytes are malformed.
    """

    decoded = _load_json(payload)
    _require_exact_fields(
        decoded,
        {
            "schema",
            "startup_matrix_sha256",
            "local_rank",
            "opposite_role_rank_count",
            "opposite_role_roster_sha256",
        },
        "startup peer commitment wrapper",
    )
    if decoded.pop("schema") != TERMINAL_STARTUP_PEER_COMMITMENT_SCHEMA:
        raise TerminalStartupPeerCommitmentError(
            "unsupported startup peer commitment schema"
        )
    commitment = _parse_commitment_payload(decoded)
    if encode_terminal_startup_peer_commitment(commitment) != payload:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment JSON is not canonical"
        )
    return commitment


def encode_terminal_startup_peer_commitment_matrix(
    matrix: TerminalStartupPeerCommitmentMatrix,
) -> bytes:
    """Encode one canonical sealed peer commitment matrix.

    :param matrix: Complete all-rank commitment result.
    :returns: Compact UTF-8 JSON without a newline.
    """

    if type(matrix) is not TerminalStartupPeerCommitmentMatrix:
        raise TypeError("matrix must be TerminalStartupPeerCommitmentMatrix")
    payload = {
        "schema": TERMINAL_STARTUP_PEER_COMMITMENT_MATRIX_SCHEMA,
        "startup_matrix_sha256": matrix.startup_matrix_sha256.hex(),
        "commitments": [
            _commitment_payload(commitment) for commitment in matrix.commitments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def decode_terminal_startup_peer_commitment_matrix(
    payload: bytes,
) -> TerminalStartupPeerCommitmentMatrix:
    """Decode one strict canonical sealed peer commitment matrix.

    :param payload: Exact wire bytes.
    :returns: Validated all-rank commitment result.
    :raises TerminalStartupPeerCommitmentError: If bytes are malformed.
    """

    decoded = _load_json(payload)
    _require_exact_fields(
        decoded,
        {"schema", "startup_matrix_sha256", "commitments"},
        "startup peer commitment matrix",
    )
    if decoded["schema"] != TERMINAL_STARTUP_PEER_COMMITMENT_MATRIX_SCHEMA:
        raise TerminalStartupPeerCommitmentError(
            "unsupported startup peer commitment matrix schema"
        )
    raw_commitments = decoded["commitments"]
    if type(raw_commitments) is not list:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitments must be a list"
        )
    commitments: list[TerminalStartupPeerCommitment] = []
    for raw_commitment in raw_commitments:
        if type(raw_commitment) is not dict:
            raise TerminalStartupPeerCommitmentError(
                "startup peer commitment matrix entry must be an object"
            )
        commitments.append(_parse_commitment_payload(raw_commitment))
    try:
        matrix = TerminalStartupPeerCommitmentMatrix(
            startup_matrix_sha256=_parse_sha256(
                decoded["startup_matrix_sha256"],
                "startup_matrix_sha256",
            ),
            commitments=tuple(commitments),
        )
    except (TypeError, ValueError) as error:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment matrix fields are invalid"
        ) from error
    if encode_terminal_startup_peer_commitment_matrix(matrix) != payload:
        raise TerminalStartupPeerCommitmentError(
            "startup peer commitment matrix JSON is not canonical"
        )
    return matrix
