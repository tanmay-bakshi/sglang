import base64
import binascii
import dataclasses
import hashlib
import json
import uuid

import requests
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TERMINAL_STARTUP_WIRE_MAX_BYTES,
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
    encode_terminal_startup_rank_advertisement,
)

TERMINAL_NIXL_SOURCE_ROSTER_ROUTE: str = "/terminal-source-roster"
TERMINAL_NIXL_SOURCE_ROSTER_SCHEMA: str = "packed-terminal-nixl-source-roster-v1"

_ROSTER_FIELDS = {
    "schema",
    "matrix_sha256",
    "requester_service_id",
    "requester_tensor_parallel_rank",
    "requester_process_generation",
    "routes",
}
_ROUTE_FIELDS = {
    "service_id",
    "tensor_parallel_rank",
    "tensor_parallel_size",
    "process_generation",
    "nixl_agent_name",
    "nixl_agent_metadata",
    "nixl_agent_metadata_sha256",
    "rank_ip",
    "rank_port",
    "attn_dp_rank",
    "attn_cp_rank",
    "attn_tp_rank",
    "pp_rank",
    "transfer_source_rank",
    "transport_protocol",
}


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
    """Require one binary SHA-256 digest.

    :param value: Candidate digest.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes or len(value) != hashlib.sha256().digest_size:
        raise ValueError(f"{label} must contain 32 bytes")


def _parse_uuid(value: object, label: str) -> bytes:
    """Parse one canonical non-nil UUID string.

    :param value: Candidate JSON value.
    :param label: Reader-facing field name.
    :returns: Canonical UUID bytes.
    :raises TerminalStartupCohortError: If the value is malformed.
    """

    if type(value) is not str:
        raise TerminalStartupCohortError(f"{label} must be a string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise TerminalStartupCohortError(f"{label} must be a UUID") from error
    if parsed.int == 0 or str(parsed) != value:
        raise TerminalStartupCohortError(f"{label} is not canonical")
    return parsed.bytes


def _parse_sha256(value: object, label: str) -> bytes:
    """Parse one lowercase hexadecimal SHA-256 value.

    :param value: Candidate JSON value.
    :param label: Reader-facing field name.
    :returns: Binary digest.
    :raises TerminalStartupCohortError: If the value is malformed.
    """

    if type(value) is not str or len(value) != 64:
        raise TerminalStartupCohortError(f"{label} must contain 64 hex digits")
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise TerminalStartupCohortError(
            f"{label} must be lowercase hexadecimal"
        ) from error
    if digest.hex() != value:
        raise TerminalStartupCohortError(f"{label} must be lowercase hexadecimal")
    return digest


def _parse_nonnegative_int(value: object, label: str) -> int:
    """Parse one exact nonnegative integer.

    :param value: Candidate JSON value.
    :param label: Reader-facing field name.
    :returns: Parsed integer.
    :raises TerminalStartupCohortError: If the value is invalid.
    """

    if type(value) is not int or value < 0:
        raise TerminalStartupCohortError(f"{label} must be nonnegative")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalNixlSourceRoute:
    """One complete native source route admitted by the startup matrix.

    :ivar service_id: Static source service identifier.
    :ivar tensor_parallel_rank: Rank within the source service.
    :ivar tensor_parallel_size: Exact source TP width.
    :ivar process_generation: Native source process incarnation.
    :ivar nixl_agent_name: Exact native agent name.
    :ivar nixl_agent_metadata: Complete native agent metadata.
    :ivar rank_ip: Source control listener address.
    :ivar rank_port: Source control listener port.
    :ivar attn_dp_rank: Source attention DP rank.
    :ivar attn_cp_rank: Source attention CP rank.
    :ivar attn_tp_rank: Source attention TP rank.
    :ivar pp_rank: Source pipeline rank.
    :ivar transfer_source_rank: Canonical source writer rank.
    :ivar transport_protocol: Exact NIXL peer protocol.
    """

    service_id: str
    tensor_parallel_rank: int
    tensor_parallel_size: int
    process_generation: bytes
    nixl_agent_name: str
    nixl_agent_metadata: bytes
    rank_ip: str
    rank_port: int
    attn_dp_rank: int
    attn_cp_rank: int
    attn_tp_rank: int
    pp_rank: int
    transfer_source_rank: int
    transport_protocol: str

    def __post_init__(self) -> None:
        """Validate one complete source route."""

        string_fields = (
            (self.service_id, "service_id"),
            (self.nixl_agent_name, "nixl_agent_name"),
            (self.rank_ip, "rank_ip"),
            (self.transport_protocol, "transport_protocol"),
        )
        for value, label in string_fields:
            if type(value) is not str or len(value) == 0:
                raise ValueError(f"{label} must be nonempty")
            try:
                value.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError(f"{label} must be ASCII") from error
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        rank_fields = (
            (self.tensor_parallel_rank, "tensor_parallel_rank"),
            (self.attn_dp_rank, "attn_dp_rank"),
            (self.attn_cp_rank, "attn_cp_rank"),
            (self.attn_tp_rank, "attn_tp_rank"),
            (self.pp_rank, "pp_rank"),
            (self.transfer_source_rank, "transfer_source_rank"),
        )
        for value, label in rank_fields:
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be nonnegative")
        if self.tensor_parallel_rank >= self.tensor_parallel_size:
            raise ValueError("tensor_parallel_rank is outside its source service")
        _require_uuid_bytes(self.process_generation, "process_generation")
        if type(self.nixl_agent_metadata) is not bytes:
            raise TypeError("nixl_agent_metadata must be bytes")
        if len(self.nixl_agent_metadata) == 0:
            raise ValueError("nixl_agent_metadata must be nonempty")
        if type(self.rank_port) is not int or not 1 <= self.rank_port <= 65535:
            raise ValueError("rank_port is invalid")

    @property
    def metadata_sha256(self) -> bytes:
        """Return the complete native metadata digest.

        :returns: Binary SHA-256 digest.
        """

        return hashlib.sha256(self.nixl_agent_metadata).digest()

    def bootstrap_info(self) -> dict[str, object]:
        """Return the request-path bootstrap shape without mutable authority.

        :returns: Full metadata route accepted by the NIXL manager.
        """

        return {
            "transport_protocol": self.transport_protocol,
            "nixl_agent_name": self.nixl_agent_name,
            "nixl_agent_metadata": base64.b64encode(self.nixl_agent_metadata).decode(
                "ascii"
            ),
            "nixl_agent_metadata_sha256": self.metadata_sha256.hex(),
            "process_generation": str(uuid.UUID(bytes=self.process_generation)),
            "attn_dp_rank": self.attn_dp_rank,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_tp_rank": self.attn_tp_rank,
            "pp_rank": self.pp_rank,
            "transfer_source_rank": self.transfer_source_rank,
            "rank_ip": self.rank_ip,
            "rank_port": self.rank_port,
            "is_dummy": False,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalNixlSourceRoster:
    """Canonical complete source route roster for one decoder rank.

    :ivar matrix_sha256: Exact observed startup matrix digest.
    :ivar requester_service_id: Decoder service receiving the roster.
    :ivar requester_tensor_parallel_rank: Decoder rank receiving the roster.
    :ivar requester_process_generation: Exact decoder process incarnation.
    :ivar routes: Complete source-first matrix-ordered native routes.
    """

    matrix_sha256: bytes
    requester_service_id: str
    requester_tensor_parallel_rank: int
    requester_process_generation: bytes
    routes: tuple[TerminalNixlSourceRoute, ...]

    def __post_init__(self) -> None:
        """Validate the closed roster shape."""

        _require_sha256(self.matrix_sha256, "matrix_sha256")
        if (
            type(self.requester_service_id) is not str
            or len(self.requester_service_id) == 0
        ):
            raise ValueError("requester_service_id must be nonempty")
        if (
            type(self.requester_tensor_parallel_rank) is not int
            or self.requester_tensor_parallel_rank < 0
        ):
            raise ValueError("requester_tensor_parallel_rank must be nonnegative")
        _require_uuid_bytes(
            self.requester_process_generation,
            "requester_process_generation",
        )
        if type(self.routes) is not tuple or len(self.routes) == 0:
            raise ValueError("routes must be a nonempty tuple")
        if any(type(route) is not TerminalNixlSourceRoute for route in self.routes):
            raise TypeError("routes must contain TerminalNixlSourceRoute values")
        keys = tuple(
            (route.service_id, route.tensor_parallel_rank) for route in self.routes
        )
        if len(set(keys)) != len(keys):
            raise ValueError("source route keys must be unique")

    def require_matrix(
        self,
        matrix: TerminalStartupCohortMatrix,
        requester: TerminalStartupRankAdvertisement,
        transport_protocol: str,
    ) -> None:
        """Authenticate the complete roster against one decoder binding.

        :param matrix: Complete sealed startup matrix.
        :param requester: Exact local decoder row.
        :param transport_protocol: Required native transport protocol.
        :raises TerminalStartupCohortError: If identity or topology drifts.
        """

        if type(matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("matrix must be TerminalStartupCohortMatrix")
        if type(requester) is not TerminalStartupRankAdvertisement:
            raise TypeError("requester must be TerminalStartupRankAdvertisement")
        if requester.role is not TerminalOwnerRole.DECODE:
            raise TerminalStartupCohortError(
                "only a decoder rank may receive the source roster"
            )
        if matrix.rank(*requester.key) != requester:
            raise TerminalStartupCohortError(
                "source roster requester differs from the sealed matrix"
            )
        if (
            self.matrix_sha256 != matrix.digest
            or self.requester_service_id != requester.service_id
            or self.requester_tensor_parallel_rank != requester.tensor_parallel_rank
            or self.requester_process_generation != requester.process_generation
        ):
            raise TerminalStartupCohortError(
                "source roster belongs to another matrix or decoder"
            )
        source_ranks = tuple(
            rank for rank in matrix.ranks if rank.role is TerminalOwnerRole.SOURCE
        )
        if len(self.routes) != len(source_ranks):
            raise TerminalStartupCohortError("source roster is incomplete")
        for route, rank in zip(self.routes, source_ranks, strict=True):
            if (
                route.service_id != rank.service_id
                or route.tensor_parallel_rank != rank.tensor_parallel_rank
                or route.tensor_parallel_size != rank.tensor_parallel_size
                or route.process_generation != rank.process_generation
                or route.nixl_agent_name != rank.nixl_agent_name
                or route.metadata_sha256 != rank.nixl_agent_metadata_sha256
                or route.attn_dp_rank != 0
                or route.attn_cp_rank != 0
                or route.pp_rank != 0
                or route.attn_tp_rank != rank.tensor_parallel_rank
                or route.transfer_source_rank != rank.tensor_parallel_rank
                or route.transport_protocol != transport_protocol
            ):
                raise TerminalStartupCohortError(
                    "source route differs from the sealed startup matrix"
                )


def _route_payload(route: TerminalNixlSourceRoute) -> dict[str, object]:
    """Return canonical JSON-compatible source route fields.

    :param route: Exact source route.
    :returns: Frozen field-order payload.
    """

    return {
        "service_id": route.service_id,
        "tensor_parallel_rank": route.tensor_parallel_rank,
        "tensor_parallel_size": route.tensor_parallel_size,
        "process_generation": str(uuid.UUID(bytes=route.process_generation)),
        "nixl_agent_name": route.nixl_agent_name,
        "nixl_agent_metadata": base64.b64encode(route.nixl_agent_metadata).decode(
            "ascii"
        ),
        "nixl_agent_metadata_sha256": route.metadata_sha256.hex(),
        "rank_ip": route.rank_ip,
        "rank_port": route.rank_port,
        "attn_dp_rank": route.attn_dp_rank,
        "attn_cp_rank": route.attn_cp_rank,
        "attn_tp_rank": route.attn_tp_rank,
        "pp_rank": route.pp_rank,
        "transfer_source_rank": route.transfer_source_rank,
        "transport_protocol": route.transport_protocol,
    }


def encode_terminal_nixl_source_roster(
    roster: TerminalNixlSourceRoster,
) -> bytes:
    """Encode one roster as bounded canonical JSON.

    :param roster: Complete authenticated source roster.
    :returns: Canonical UTF-8 JSON bytes.
    """

    if type(roster) is not TerminalNixlSourceRoster:
        raise TypeError("roster must be TerminalNixlSourceRoster")
    payload = {
        "schema": TERMINAL_NIXL_SOURCE_ROSTER_SCHEMA,
        "matrix_sha256": roster.matrix_sha256.hex(),
        "requester_service_id": roster.requester_service_id,
        "requester_tensor_parallel_rank": roster.requester_tensor_parallel_rank,
        "requester_process_generation": str(
            uuid.UUID(bytes=roster.requester_process_generation)
        ),
        "routes": [_route_payload(route) for route in roster.routes],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    if len(encoded) > TERMINAL_STARTUP_WIRE_MAX_BYTES:
        raise ValueError("source roster exceeds the startup wire limit")
    return encoded


def _decode_route(payload: object) -> TerminalNixlSourceRoute:
    """Decode one strict source route object.

    :param payload: Parsed JSON value.
    :returns: Validated native source route.
    :raises TerminalStartupCohortError: If the route is malformed.
    """

    if type(payload) is not dict or set(payload) != _ROUTE_FIELDS:
        raise TerminalStartupCohortError("source route field set is invalid")
    metadata_value = payload["nixl_agent_metadata"]
    if type(metadata_value) is not str or len(metadata_value) == 0:
        raise TerminalStartupCohortError(
            "nixl_agent_metadata must be a nonempty string"
        )
    try:
        metadata = base64.b64decode(metadata_value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TerminalStartupCohortError(
            "nixl_agent_metadata is not canonical base64"
        ) from error
    digest = _parse_sha256(
        payload["nixl_agent_metadata_sha256"],
        "nixl_agent_metadata_sha256",
    )
    if hashlib.sha256(metadata).digest() != digest:
        raise TerminalStartupCohortError("source route metadata digest mismatch")
    string_fields = {}
    for field_name in (
        "service_id",
        "nixl_agent_name",
        "rank_ip",
        "transport_protocol",
    ):
        value = payload[field_name]
        if type(value) is not str or len(value) == 0:
            raise TerminalStartupCohortError(f"{field_name} must be nonempty")
        string_fields[field_name] = value
    tensor_parallel_size = payload["tensor_parallel_size"]
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
        raise TerminalStartupCohortError("tensor_parallel_size must be positive")
    rank_port = payload["rank_port"]
    if type(rank_port) is not int or not 1 <= rank_port <= 65535:
        raise TerminalStartupCohortError("rank_port is invalid")
    return TerminalNixlSourceRoute(
        service_id=string_fields["service_id"],
        tensor_parallel_rank=_parse_nonnegative_int(
            payload["tensor_parallel_rank"],
            "tensor_parallel_rank",
        ),
        tensor_parallel_size=tensor_parallel_size,
        process_generation=_parse_uuid(
            payload["process_generation"],
            "process_generation",
        ),
        nixl_agent_name=string_fields["nixl_agent_name"],
        nixl_agent_metadata=metadata,
        rank_ip=string_fields["rank_ip"],
        rank_port=rank_port,
        attn_dp_rank=_parse_nonnegative_int(payload["attn_dp_rank"], "attn_dp_rank"),
        attn_cp_rank=_parse_nonnegative_int(payload["attn_cp_rank"], "attn_cp_rank"),
        attn_tp_rank=_parse_nonnegative_int(payload["attn_tp_rank"], "attn_tp_rank"),
        pp_rank=_parse_nonnegative_int(payload["pp_rank"], "pp_rank"),
        transfer_source_rank=_parse_nonnegative_int(
            payload["transfer_source_rank"],
            "transfer_source_rank",
        ),
        transport_protocol=string_fields["transport_protocol"],
    )


def decode_terminal_nixl_source_roster(data: bytes) -> TerminalNixlSourceRoster:
    """Decode bounded canonical source-roster JSON.

    :param data: Candidate wire bytes.
    :returns: Validated complete source roster.
    :raises TerminalStartupCohortError: If the payload is malformed or noncanonical.
    """

    if type(data) is not bytes or not 0 < len(data) <= TERMINAL_STARTUP_WIRE_MAX_BYTES:
        raise TerminalStartupCohortError("source roster payload size is invalid")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalStartupCohortError("source roster is not valid JSON") from error
    if type(payload) is not dict or set(payload) != _ROSTER_FIELDS:
        raise TerminalStartupCohortError("source roster field set is invalid")
    if payload["schema"] != TERMINAL_NIXL_SOURCE_ROSTER_SCHEMA:
        raise TerminalStartupCohortError("source roster schema is unsupported")
    routes_value = payload["routes"]
    if type(routes_value) is not list or len(routes_value) == 0:
        raise TerminalStartupCohortError("source roster routes must be nonempty")
    requester_service_id = payload["requester_service_id"]
    if type(requester_service_id) is not str or len(requester_service_id) == 0:
        raise TerminalStartupCohortError("requester_service_id must be nonempty")
    roster = TerminalNixlSourceRoster(
        matrix_sha256=_parse_sha256(payload["matrix_sha256"], "matrix_sha256"),
        requester_service_id=requester_service_id,
        requester_tensor_parallel_rank=_parse_nonnegative_int(
            payload["requester_tensor_parallel_rank"],
            "requester_tensor_parallel_rank",
        ),
        requester_process_generation=_parse_uuid(
            payload["requester_process_generation"],
            "requester_process_generation",
        ),
        routes=tuple(_decode_route(route) for route in routes_value),
    )
    if encode_terminal_nixl_source_roster(roster) != data:
        raise TerminalStartupCohortError("source roster encoding is not canonical")
    return roster


def fetch_terminal_nixl_source_roster(
    endpoint: str,
    requester: TerminalStartupRankAdvertisement,
    matrix: TerminalStartupCohortMatrix,
    transport_protocol: str,
    timeout_seconds: float,
) -> TerminalNixlSourceRoster:
    """Fetch one complete source roster without retry polling.

    Listener liveness and the initial rank join precede this call. A transport
    failure is therefore a collective startup failure rather than a mutable
    discovery condition.

    :param endpoint: Exact source-owned roster route.
    :param requester: Local decoder startup row.
    :param matrix: Complete sealed startup matrix.
    :param transport_protocol: Required native transport protocol.
    :param timeout_seconds: Hash-bound startup control deadline.
    :returns: Complete authenticated source route roster.
    :raises TerminalStartupCohortError: If transport or admission fails.
    """

    if type(endpoint) is not str or not endpoint.endswith(
        TERMINAL_NIXL_SOURCE_ROSTER_ROUTE
    ):
        raise ValueError("endpoint must select the terminal source-roster route")
    if type(requester) is not TerminalStartupRankAdvertisement:
        raise TypeError("requester must be TerminalStartupRankAdvertisement")
    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    if type(transport_protocol) is not str or len(transport_protocol) == 0:
        raise ValueError("transport_protocol must be nonempty")
    if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a positive float")
    try:
        response = requests.post(
            endpoint,
            data=encode_terminal_startup_rank_advertisement(requester),
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise TerminalStartupCohortError(
            "terminal source-roster transport failed"
        ) from error
    if response.status_code != 200:
        raise TerminalStartupCohortError(
            "terminal source-roster request failed with HTTP "
            f"{response.status_code}: {response.text[:512]}"
        )
    roster = decode_terminal_nixl_source_roster(response.content)
    roster.require_matrix(matrix, requester, transport_protocol)
    return roster
