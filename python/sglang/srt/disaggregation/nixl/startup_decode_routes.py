import dataclasses
import hashlib
import hmac
import json
import uuid

from sglang.srt.disaggregation.nixl.startup_enrollment_ack import (
    terminal_decoder_registration_multipart_sha256,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.utils.network import NetworkAddress

TERMINAL_DECODE_CONTROL_ROUTE_TABLE_SCHEMA: str = (
    "packed-terminal-decode-control-route-table-v1"
)
TERMINAL_DECODE_CONTROL_ROUTE_TABLE_MAX_BYTES: int = 64 * 1024

_SHA256_BYTES = hashlib.sha256().digest_size


class TerminalDecodeControlRouteError(RuntimeError):
    """Invalid or stale same-service decode control route authority."""


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDecodeControlRoute:
    """One decoder owner and its startup-authenticated control listener.

    :ivar startup_rank: Exact decoder row from the sealed startup matrix.
    :ivar endpoint: Manager-owned PULL listener retained by every source rank.
    :ivar registration_multipart_sha256: Digest of the guarded decoder
        registration which advertised the endpoint.
    """

    startup_rank: TerminalStartupRankAdvertisement
    endpoint: NetworkAddress
    registration_multipart_sha256: bytes

    def __post_init__(self) -> None:
        """Validate one generation-bound decoder route."""

        if type(self.startup_rank) is not TerminalStartupRankAdvertisement:
            raise TypeError("startup_rank must be TerminalStartupRankAdvertisement")
        if self.startup_rank.role is not TerminalOwnerRole.DECODE:
            raise ValueError("decode control routes require decoder ranks")
        if type(self.endpoint) is not NetworkAddress:
            raise TypeError("endpoint must be NetworkAddress")
        if type(self.endpoint.host) is not str or len(self.endpoint.host) == 0:
            raise ValueError("decode control endpoint host must be nonempty")
        try:
            self.endpoint.host.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("decode control endpoint host must be ASCII") from error
        if type(self.endpoint.port) is not int or not 1 <= self.endpoint.port <= 65535:
            raise ValueError("decode control endpoint port is invalid")
        if (
            type(self.registration_multipart_sha256) is not bytes
            or len(self.registration_multipart_sha256) != _SHA256_BYTES
        ):
            raise ValueError("registration_multipart_sha256 must contain 32 bytes")

    @property
    def identity(self) -> TerminalProcessIdentity:
        """Return the exact decoder process reached by this route.

        :returns: Generation-bound decoder process identity.
        """

        return self.startup_rank.terminal_identity


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDecodeControlRouteTable:
    """Complete immutable listener table for one decoder TP service.

    :ivar startup_matrix_sha256: Digest of the sealed startup matrix.
    :ivar decoder_service_id: Sole decoder service represented by the table.
    :ivar routes: Canonical TP-rank-ordered decoder control routes.
    """

    startup_matrix_sha256: bytes
    decoder_service_id: str
    routes: tuple[TerminalDecodeControlRoute, ...]

    def __post_init__(self) -> None:
        """Validate complete canonical membership and unique listeners."""

        if (
            type(self.startup_matrix_sha256) is not bytes
            or len(self.startup_matrix_sha256) != _SHA256_BYTES
        ):
            raise ValueError("startup_matrix_sha256 must contain 32 bytes")
        if type(self.decoder_service_id) is not str or len(self.decoder_service_id) == 0:
            raise ValueError("decoder_service_id must be nonempty")
        if type(self.routes) is not tuple or len(self.routes) == 0:
            raise ValueError("routes must be a nonempty tuple")
        if any(type(route) is not TerminalDecodeControlRoute for route in self.routes):
            raise TypeError("routes must contain TerminalDecodeControlRoute values")

        ranks = tuple(route.startup_rank for route in self.routes)
        tp_size = ranks[0].tensor_parallel_size
        if len(ranks) != tp_size:
            raise ValueError("decode control route table is incomplete")
        if any(
            rank.service_id != self.decoder_service_id
            or rank.role is not TerminalOwnerRole.DECODE
            or rank.tensor_parallel_size != tp_size
            for rank in ranks
        ):
            raise ValueError("decode control route table spans services or TP widths")
        if tuple(rank.tensor_parallel_rank for rank in ranks) != tuple(range(tp_size)):
            raise ValueError("decode control routes must use canonical TP-rank order")
        endpoints = tuple(
            (route.endpoint.host, route.endpoint.port) for route in self.routes
        )
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("decode control endpoints must be unique")

    @property
    def digest(self) -> bytes:
        """Return the canonical route-table digest.

        :returns: SHA-256 of the exact canonical wire bytes.
        """

        return hashlib.sha256(encode_terminal_decode_control_route_table(self)).digest()

    @property
    def identities(self) -> tuple[TerminalProcessIdentity, ...]:
        """Return the complete decoder TP membership.

        :returns: One generation-bound identity per TP rank.
        """

        return tuple(route.identity for route in self.routes)

    def route_for(self, identity: TerminalProcessIdentity) -> TerminalDecodeControlRoute:
        """Resolve one exact-generation decoder listener.

        :param identity: Decoder owner selected by request authority.
        :returns: Sole matching immutable route.
        :raises TerminalDecodeControlRouteError: If the identity is stale or absent.
        """

        if type(identity) is not TerminalProcessIdentity:
            raise TypeError("identity must be TerminalProcessIdentity")
        matches = tuple(route for route in self.routes if route.identity == identity)
        if len(matches) != 1:
            raise TerminalDecodeControlRouteError(
                "decoder identity is absent or stale in the control route table"
            )
        return matches[0]

    def require_matrix(self, matrix: TerminalStartupCohortMatrix) -> None:
        """Authenticate every route against one sealed startup matrix.

        :param matrix: Complete deployment-epoch rank matrix.
        :raises TerminalDecodeControlRouteError: If membership differs.
        """

        if type(matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("matrix must be TerminalStartupCohortMatrix")
        if not hmac.compare_digest(self.startup_matrix_sha256, matrix.digest):
            raise TerminalDecodeControlRouteError(
                "decode control routes belong to another startup matrix"
            )
        expected = tuple(
            rank
            for rank in matrix.ranks
            if rank.role is TerminalOwnerRole.DECODE
            and rank.service_id == self.decoder_service_id
        )
        if tuple(route.startup_rank for route in self.routes) != expected:
            raise TerminalDecodeControlRouteError(
                "decode control routes differ from the sealed service membership"
            )

    def require_local_registration(
        self,
        binding: TerminalStartupRankBinding,
        local_endpoint: NetworkAddress,
        registration_frames: tuple[bytes, ...],
    ) -> None:
        """Bind the table to the local listener and exact sent registration.

        :param binding: Local decoder startup authority.
        :param local_endpoint: Actual manager-owned PULL listener.
        :param registration_frames: Guarded registration sent to every source.
        :raises TerminalDecodeControlRouteError: If local authority differs.
        """

        if type(binding) is not TerminalStartupRankBinding:
            raise TypeError("binding must be TerminalStartupRankBinding")
        if type(local_endpoint) is not NetworkAddress:
            raise TypeError("local_endpoint must be NetworkAddress")
        local = binding.advertisement
        if local.role is not TerminalOwnerRole.DECODE:
            raise TerminalDecodeControlRouteError(
                "only a decoder can adopt a decode control route table"
            )
        self.require_matrix(binding.matrix)
        if local.service_id != self.decoder_service_id:
            raise TerminalDecodeControlRouteError(
                "decode control routes belong to another decoder service"
            )
        route = self.route_for(local.terminal_identity)
        if route.endpoint != local_endpoint:
            raise TerminalDecodeControlRouteError(
                "decode control route conflicts with the local manager listener"
            )
        actual_digest = terminal_decoder_registration_multipart_sha256(
            registration_frames
        )
        if not hmac.compare_digest(route.registration_multipart_sha256, actual_digest):
            raise TerminalDecodeControlRouteError(
                "decode control route binds another local registration"
            )


def build_terminal_decode_control_route_table(
    matrix: TerminalStartupCohortMatrix,
    decoder_service_id: str,
    registrations: tuple[
        tuple[
            TerminalStartupRankAdvertisement,
            NetworkAddress,
            tuple[bytes, ...],
        ],
        ...,
    ],
) -> TerminalDecodeControlRouteTable:
    """Freeze source-retained decoder registrations into one route table.

    :param matrix: Complete generation-authenticated startup matrix.
    :param decoder_service_id: Decoder TP service represented by registrations.
    :param registrations: Canonical rank, listener, and guarded registration rows.
    :returns: Complete matrix-bound route table.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    if type(decoder_service_id) is not str or len(decoder_service_id) == 0:
        raise ValueError("decoder_service_id must be nonempty")
    if type(registrations) is not tuple:
        raise TypeError("registrations must be a tuple")
    routes = tuple(
        TerminalDecodeControlRoute(
            startup_rank=rank,
            endpoint=endpoint,
            registration_multipart_sha256=(
                terminal_decoder_registration_multipart_sha256(frames)
            ),
        )
        for rank, endpoint, frames in registrations
    )
    table = TerminalDecodeControlRouteTable(
        startup_matrix_sha256=matrix.digest,
        decoder_service_id=decoder_service_id,
        routes=routes,
    )
    table.require_matrix(matrix)
    return table


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
        raise TerminalDecodeControlRouteError(f"{label} field set is invalid")


def _parse_sha256(value: object, label: str) -> bytes:
    """Parse one canonical lowercase SHA-256 string.

    :param value: Candidate JSON field.
    :param label: Reader-facing field name.
    :returns: Binary SHA-256 value.
    """

    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise TerminalDecodeControlRouteError(f"{label} is not canonical")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise TerminalDecodeControlRouteError(f"{label} is not hexadecimal") from error
    if len(parsed) != _SHA256_BYTES:
        raise TerminalDecodeControlRouteError(f"{label} is not SHA-256")
    return parsed


def _load_json(payload: bytes) -> dict[str, object]:
    """Decode one bounded duplicate-free JSON object.

    :param payload: Exact candidate route-table wire bytes.
    :returns: Parsed top-level object.
    """

    if (
        type(payload) is not bytes
        or not 0 < len(payload) <= TERMINAL_DECODE_CONTROL_ROUTE_TABLE_MAX_BYTES
    ):
        raise TerminalDecodeControlRouteError(
            "decode control route-table payload size is invalid"
        )

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Build one object while rejecting duplicate field names.

        :param pairs: Decoder-preserved JSON field pairs.
        :returns: Unique-key JSON object.
        """

        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalDecodeControlRouteError(
                    f"duplicate decode control route-table field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        """Reject non-finite JSON constants.

        :param value: Non-finite JSON spelling.
        :raises TerminalDecodeControlRouteError: For every input.
        """

        raise TerminalDecodeControlRouteError(
            f"non-finite decode control route-table value: {value}"
        )

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalDecodeControlRouteError(
            "decode control route table is not valid JSON"
        ) from error
    if type(decoded) is not dict:
        raise TerminalDecodeControlRouteError(
            "decode control route table must be an object"
        )
    return decoded


def encode_terminal_decode_control_route_table(
    table: TerminalDecodeControlRouteTable,
) -> bytes:
    """Encode one canonical immutable decoder route table.

    :param table: Complete matrix-bound decoder routes.
    :returns: Compact canonical UTF-8 JSON.
    """

    if type(table) is not TerminalDecodeControlRouteTable:
        raise TypeError("table must be TerminalDecodeControlRouteTable")
    payload = {
        "schema": TERMINAL_DECODE_CONTROL_ROUTE_TABLE_SCHEMA,
        "startup_matrix_sha256": table.startup_matrix_sha256.hex(),
        "decoder_service_id": table.decoder_service_id,
        "routes": [
            {
                "tensor_parallel_rank": route.startup_rank.tensor_parallel_rank,
                "process_generation": str(
                    uuid.UUID(bytes=route.startup_rank.process_generation)
                ),
                "nixl_agent_name": route.startup_rank.nixl_agent_name,
                "host": route.endpoint.host,
                "port": route.endpoint.port,
                "registration_multipart_sha256": (
                    route.registration_multipart_sha256.hex()
                ),
            }
            for route in table.routes
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > TERMINAL_DECODE_CONTROL_ROUTE_TABLE_MAX_BYTES:
        raise ValueError("decode control route table is too large")
    return encoded


def decode_terminal_decode_control_route_table(
    matrix: TerminalStartupCohortMatrix,
    payload: bytes,
) -> TerminalDecodeControlRouteTable:
    """Decode and authenticate one canonical decoder route table.

    :param matrix: Complete startup matrix used to resolve rank identities.
    :param payload: Exact canonical wire bytes.
    :returns: Immutable matrix-bound route table.
    :raises TerminalDecodeControlRouteError: If bytes or membership are invalid.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    decoded = _load_json(payload)
    _require_exact_fields(
        decoded,
        {"schema", "startup_matrix_sha256", "decoder_service_id", "routes"},
        "decode control route table",
    )
    if decoded["schema"] != TERMINAL_DECODE_CONTROL_ROUTE_TABLE_SCHEMA:
        raise TerminalDecodeControlRouteError(
            "unsupported decode control route-table schema"
        )
    service_id = decoded["decoder_service_id"]
    raw_routes = decoded["routes"]
    if type(service_id) is not str or len(service_id) == 0:
        raise TerminalDecodeControlRouteError("decoder_service_id is malformed")
    if type(raw_routes) is not list or len(raw_routes) == 0:
        raise TerminalDecodeControlRouteError("routes must be a nonempty array")

    routes: list[TerminalDecodeControlRoute] = []
    for raw_route in raw_routes:
        if type(raw_route) is not dict:
            raise TerminalDecodeControlRouteError("route must be an object")
        _require_exact_fields(
            raw_route,
            {
                "tensor_parallel_rank",
                "process_generation",
                "nixl_agent_name",
                "host",
                "port",
                "registration_multipart_sha256",
            },
            "decode control route",
        )
        rank_value = raw_route["tensor_parallel_rank"]
        generation_value = raw_route["process_generation"]
        agent_name = raw_route["nixl_agent_name"]
        host = raw_route["host"]
        port = raw_route["port"]
        if (
            type(rank_value) is not int
            or type(generation_value) is not str
            or type(agent_name) is not str
            or type(host) is not str
            or type(port) is not int
        ):
            raise TerminalDecodeControlRouteError("route fields are malformed")
        try:
            startup_rank = matrix.rank(service_id, rank_value)
            generation = uuid.UUID(generation_value)
        except (TerminalStartupCohortError, ValueError) as error:
            raise TerminalDecodeControlRouteError(
                "decode control route rank identity is invalid"
            ) from error
        if (
            generation.int == 0
            or str(generation) != generation_value
            or startup_rank.process_generation != generation.bytes
            or startup_rank.nixl_agent_name != agent_name
        ):
            raise TerminalDecodeControlRouteError(
                "decode control route differs from the startup rank identity"
            )
        try:
            route = TerminalDecodeControlRoute(
                startup_rank=startup_rank,
                endpoint=NetworkAddress(host, port),
                registration_multipart_sha256=_parse_sha256(
                    raw_route["registration_multipart_sha256"],
                    "registration_multipart_sha256",
                ),
            )
        except (TypeError, ValueError) as error:
            raise TerminalDecodeControlRouteError(
                "decode control route fields are invalid"
            ) from error
        routes.append(route)

    try:
        table = TerminalDecodeControlRouteTable(
            startup_matrix_sha256=_parse_sha256(
                decoded["startup_matrix_sha256"],
                "startup_matrix_sha256",
            ),
            decoder_service_id=service_id,
            routes=tuple(routes),
        )
        table.require_matrix(matrix)
    except (TypeError, ValueError) as error:
        raise TerminalDecodeControlRouteError(
            "decode control route table fields are invalid"
        ) from error
    if encode_terminal_decode_control_route_table(table) != payload:
        raise TerminalDecodeControlRouteError(
            "decode control route-table JSON is not canonical"
        )
    return table
