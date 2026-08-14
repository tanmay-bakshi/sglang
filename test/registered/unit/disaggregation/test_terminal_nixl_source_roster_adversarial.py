import base64
import dataclasses
import hashlib
import json
import uuid

import pytest
from sglang.srt.disaggregation.nixl.startup_source_roster import (
    TERMINAL_NIXL_SOURCE_ROSTER_SCHEMA,
    TerminalNixlSourceRoster,
    TerminalNixlSourceRoute,
    decode_terminal_nixl_source_roster,
    encode_terminal_nixl_source_roster,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TERMINAL_STARTUP_WIRE_MAX_BYTES,
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_SHA256 = hashlib.sha256(b"terminal-source-roster-adversarial").digest()
_TRANSPORT_PROTOCOL = "nixl-peer-handle-v1"


def _uuid_bytes(marker: int) -> bytes:
    """Build one stable non-nil UUID fixture.

    :param marker: Positive low-order integer.
    :returns: Exact UUID bytes.
    """

    return uuid.UUID(int=marker).bytes


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    generation_marker: int,
    agent_name: str,
    metadata: bytes,
    launch_marker: int,
    service_port: int,
) -> TerminalStartupRankAdvertisement:
    """Build one matrix-bound native rank.

    :param service_id: Static service identifier.
    :param role: Source or decoder owner role.
    :param tensor_parallel_rank: Rank within the service.
    :param tensor_parallel_size: Exact service width.
    :param generation_marker: Stable process-generation marker.
    :param agent_name: Exact NIXL agent name.
    :param metadata: Full metadata represented by the matrix digest.
    :param launch_marker: Stable service-incarnation marker.
    :param service_port: Canonical service-origin port.
    :returns: Exact startup advertisement.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{service_port}",
        role=role,
        launch_instance_id=_uuid_bytes(launch_marker),
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        process_generation=_uuid_bytes(generation_marker),
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build one TP2 source and TP1 decoder matrix.

    :returns: Complete canonical startup matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        ranks=(
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tensor_parallel_rank=0,
                tensor_parallel_size=2,
                generation_marker=101,
                agent_name="prefill-agent-0",
                metadata=b"source-metadata-0",
                launch_marker=1,
                service_port=32000,
            ),
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tensor_parallel_rank=1,
                tensor_parallel_size=2,
                generation_marker=102,
                agent_name="prefill-agent-1",
                metadata=b"source-metadata-1",
                launch_marker=1,
                service_port=32000,
            ),
            _rank(
                service_id="decode-a",
                role=TerminalOwnerRole.DECODE,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                generation_marker=201,
                agent_name="decode-agent-0",
                metadata=b"decode-metadata-0",
                launch_marker=2,
                service_port=32001,
            ),
        ),
    )


def _source_route(
    rank: TerminalStartupRankAdvertisement,
) -> TerminalNixlSourceRoute:
    """Build the full source route represented by one matrix row.

    :param rank: Exact source advertisement.
    :returns: Complete source route.
    """

    metadata = f"source-metadata-{rank.tensor_parallel_rank}".encode("ascii")
    return TerminalNixlSourceRoute(
        service_id=rank.service_id,
        tensor_parallel_rank=rank.tensor_parallel_rank,
        tensor_parallel_size=rank.tensor_parallel_size,
        process_generation=rank.process_generation,
        nixl_agent_name=rank.nixl_agent_name,
        nixl_agent_metadata=metadata,
        rank_ip="127.0.0.1",
        rank_port=33000 + rank.tensor_parallel_rank,
        attn_dp_rank=0,
        attn_cp_rank=0,
        attn_tp_rank=rank.tensor_parallel_rank,
        pp_rank=0,
        transfer_source_rank=rank.tensor_parallel_rank,
        transport_protocol=_TRANSPORT_PROTOCOL,
    )


def _roster() -> tuple[
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
    TerminalNixlSourceRoster,
]:
    """Build one valid matrix, requester, and full source roster.

    :returns: Bound matrix, decoder requester, and canonical roster.
    """

    matrix = _matrix()
    requester = matrix.rank("decode-a", 0)
    routes = tuple(
        _source_route(rank)
        for rank in matrix.ranks
        if rank.role is TerminalOwnerRole.SOURCE
    )
    roster = TerminalNixlSourceRoster(
        matrix_sha256=matrix.digest,
        requester_service_id=requester.service_id,
        requester_tensor_parallel_rank=requester.tensor_parallel_rank,
        requester_process_generation=requester.process_generation,
        routes=routes,
    )
    return matrix, requester, roster


def _payload(roster: TerminalNixlSourceRoster) -> dict[str, object]:
    """Return a mutable JSON payload for one canonical roster.

    :param roster: Source roster to encode.
    :returns: Parsed mutable JSON object.
    """

    payload = json.loads(encode_terminal_nixl_source_roster(roster))
    if type(payload) is not dict:
        raise AssertionError("encoded source roster is not an object")
    return payload


def _route_payload(
    payload: dict[str, object],
    route_index: int = 0,
) -> dict[str, object]:
    """Return one mutable route object from a roster payload.

    :param payload: Parsed roster payload.
    :param route_index: Route index to select.
    :returns: Selected mutable route object.
    """

    routes = payload["routes"]
    if type(routes) is not list:
        raise AssertionError("encoded source routes are not a list")
    route = routes[route_index]
    if type(route) is not dict:
        raise AssertionError("encoded source route is not an object")
    return route


def _canonical_wire(payload: dict[str, object]) -> bytes:
    """Encode one mutated payload with canonical JSON formatting.

    :param payload: Mutated JSON object.
    :returns: Canonically formatted ASCII JSON.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def test_canonical_roster_round_trip_and_matrix_binding() -> None:
    """Canonical bytes round-trip and retain the exact bootstrap route shape."""

    matrix, requester, roster = _roster()
    encoded = encode_terminal_nixl_source_roster(roster)
    decoded = decode_terminal_nixl_source_roster(encoded)

    assert decoded == roster
    decoded.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)
    assert decoded.routes[1].bootstrap_info() == {
        "transport_protocol": _TRANSPORT_PROTOCOL,
        "nixl_agent_name": "prefill-agent-1",
        "nixl_agent_metadata": base64.b64encode(b"source-metadata-1").decode("ascii"),
        "nixl_agent_metadata_sha256": hashlib.sha256(b"source-metadata-1").hexdigest(),
        "process_generation": str(uuid.UUID(int=102)),
        "attn_dp_rank": 0,
        "attn_cp_rank": 0,
        "attn_tp_rank": 1,
        "pp_rank": 0,
        "transfer_source_rank": 1,
        "rank_ip": "127.0.0.1",
        "rank_port": 33001,
        "is_dummy": False,
    }


@pytest.mark.parametrize(
    "roster_change",
    (
        {"matrix_sha256": hashlib.sha256(b"another-matrix").digest()},
        {"requester_service_id": "decode-b"},
        {"requester_tensor_parallel_rank": 1},
        {"requester_process_generation": _uuid_bytes(999)},
    ),
)
def test_roster_rejects_another_matrix_or_requester(
    roster_change: dict[str, object],
) -> None:
    """Roster targeting fields cannot select another matrix or requester.

    :param roster_change: Valid but unbound roster-field replacement.
    """

    matrix, requester, roster = _roster()
    changed = dataclasses.replace(roster, **roster_change)

    with pytest.raises(
        TerminalStartupCohortError,
        match="another matrix or requester",
    ):
        changed.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_source_rank_receives_its_own_matrix_bound_source_roster() -> None:
    """A source rank may enroll the same-service control route population."""

    matrix, _, decoder_roster = _roster()
    requester = matrix.rank("prefill-a", 0)
    roster = dataclasses.replace(
        decoder_roster,
        requester_service_id=requester.service_id,
        requester_tensor_parallel_rank=requester.tensor_parallel_rank,
        requester_process_generation=requester.process_generation,
    )

    roster.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_requester_generation_drift_is_rejected_by_matrix_lookup() -> None:
    """A locally mutated requester cannot borrow the matrix rank key."""

    matrix, requester, roster = _roster()
    changed_requester = dataclasses.replace(
        requester,
        process_generation=_uuid_bytes(999),
    )

    with pytest.raises(
        TerminalStartupCohortError,
        match="requester differs from the sealed matrix",
    ):
        roster.require_matrix(matrix, changed_requester, _TRANSPORT_PROTOCOL)


@pytest.mark.parametrize("tampered_field", ("metadata", "digest"))
def test_wire_rejects_metadata_or_digest_tampering(tampered_field: str) -> None:
    """Metadata bytes and their advertised digest are inseparable.

    :param tampered_field: Which side of the digest relation to corrupt.
    """

    _, _, roster = _roster()
    payload = _payload(roster)
    route = _route_payload(payload)
    if tampered_field == "metadata":
        route["nixl_agent_metadata"] = base64.b64encode(
            b"tampered-source-metadata"
        ).decode("ascii")
    else:
        route["nixl_agent_metadata_sha256"] = hashlib.sha256(
            b"another-metadata-image"
        ).hexdigest()

    with pytest.raises(
        TerminalStartupCohortError,
        match="metadata digest mismatch",
    ):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_self_consistent_metadata_replacement_still_fails_matrix_binding() -> None:
    """A recomputed wire digest cannot replace matrix-authenticated metadata."""

    matrix, requester, roster = _roster()
    payload = _payload(roster)
    route = _route_payload(payload)
    replacement = b"self-consistent-but-unauthorized"
    route["nixl_agent_metadata"] = base64.b64encode(replacement).decode("ascii")
    route["nixl_agent_metadata_sha256"] = hashlib.sha256(replacement).hexdigest()

    decoded = decode_terminal_nixl_source_roster(_canonical_wire(payload))
    with pytest.raises(
        TerminalStartupCohortError,
        match="differs from the sealed startup matrix",
    ):
        decoded.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_missing_source_route_is_rejected() -> None:
    """Every source matrix row must occur exactly once."""

    matrix, requester, roster = _roster()
    changed = dataclasses.replace(roster, routes=roster.routes[:-1])

    with pytest.raises(TerminalStartupCohortError, match="roster is incomplete"):
        changed.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_extra_source_route_is_rejected() -> None:
    """A syntactically valid extra route cannot extend the sealed roster."""

    matrix, requester, roster = _roster()
    extra = dataclasses.replace(
        roster.routes[-1],
        service_id="prefill-b",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        process_generation=_uuid_bytes(103),
        nixl_agent_name="prefill-agent-extra",
        nixl_agent_metadata=b"source-metadata-extra",
        rank_port=33002,
        attn_tp_rank=0,
        transfer_source_rank=0,
    )
    changed = dataclasses.replace(roster, routes=(*roster.routes, extra))

    with pytest.raises(TerminalStartupCohortError, match="roster is incomplete"):
        changed.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_reordered_source_routes_are_rejected() -> None:
    """Route ordering is the source-first matrix ordering, not a set."""

    matrix, requester, roster = _roster()
    changed = dataclasses.replace(roster, routes=tuple(reversed(roster.routes)))

    with pytest.raises(
        TerminalStartupCohortError,
        match="differs from the sealed startup matrix",
    ):
        changed.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


@pytest.mark.parametrize(
    "route_change",
    (
        {"service_id": "prefill-b"},
        {"tensor_parallel_rank": 2, "tensor_parallel_size": 3},
        {"tensor_parallel_size": 3},
        {"process_generation": _uuid_bytes(999)},
        {"nixl_agent_name": "another-prefill-agent"},
        {"attn_dp_rank": 1},
        {"attn_cp_rank": 1},
        {"attn_tp_rank": 1},
        {"pp_rank": 1},
        {"transfer_source_rank": 1},
        {"transport_protocol": "another-protocol"},
    ),
)
def test_route_identity_topology_and_protocol_drift_are_rejected(
    route_change: dict[str, object],
) -> None:
    """Every matrix and fixed-topology field is authenticated.

    :param route_change: Valid but unauthorized route-field replacement.
    """

    matrix, requester, roster = _roster()
    changed_route = dataclasses.replace(roster.routes[0], **route_change)
    changed = dataclasses.replace(
        roster,
        routes=(changed_route, *roster.routes[1:]),
    )

    with pytest.raises(
        TerminalStartupCohortError,
        match="differs from the sealed startup matrix",
    ):
        changed.require_matrix(matrix, requester, _TRANSPORT_PROTOCOL)


def test_required_transport_protocol_is_bound_by_the_consumer() -> None:
    """A caller cannot reinterpret an otherwise valid roster protocol."""

    matrix, requester, roster = _roster()

    with pytest.raises(
        TerminalStartupCohortError,
        match="differs from the sealed startup matrix",
    ):
        roster.require_matrix(matrix, requester, "another-protocol")


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        (
            "process_generation",
            str(uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")).upper(),
        ),
        ("process_generation", "{00000000-0000-0000-0000-000000000065}"),
        ("process_generation", str(uuid.UUID(int=0))),
    ),
)
def test_wire_rejects_noncanonical_route_uuid(
    field_name: str,
    value: str,
) -> None:
    """Route generations require canonical non-nil UUID strings.

    :param field_name: Route UUID field to mutate.
    :param value: Noncanonical UUID spelling.
    """

    _, _, roster = _roster()
    payload = _payload(roster)
    _route_payload(payload)[field_name] = value

    with pytest.raises(TerminalStartupCohortError):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_wire_rejects_noncanonical_requester_uuid() -> None:
    """Requester generations use the same canonical UUID contract."""

    _, _, roster = _roster()
    payload = _payload(roster)
    generation = payload["requester_process_generation"]
    if type(generation) is not str:
        raise AssertionError("encoded requester generation is not a string")
    payload["requester_process_generation"] = generation.upper()

    with pytest.raises(
        TerminalStartupCohortError,
        match="requester_process_generation is not canonical",
    ):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_wire_rejects_invalid_base64() -> None:
    """Whitespace and omitted padding cannot enter native metadata."""

    _, _, roster = _roster()
    payload = _payload(roster)
    route = _route_payload(payload)
    metadata = route["nixl_agent_metadata"]
    if type(metadata) is not str:
        raise AssertionError("encoded NIXL metadata is not a string")
    route["nixl_agent_metadata"] = metadata.rstrip("=")

    with pytest.raises(
        TerminalStartupCohortError,
        match="not canonical base64",
    ):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_wire_rejects_valid_but_noncanonical_base64() -> None:
    """Alternate pad bits cannot produce a second wire image for one value."""

    _, _, roster = _roster()
    payload = _payload(roster)
    route = _route_payload(payload)
    route["nixl_agent_metadata"] = "AB=="
    route["nixl_agent_metadata_sha256"] = hashlib.sha256(b"\x00").hexdigest()

    with pytest.raises(
        TerminalStartupCohortError,
        match="encoding is not canonical",
    ):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_wire_rejects_noncanonical_json_formatting() -> None:
    """Whitespace creates a distinct and therefore invalid roster wire image."""

    _, _, roster = _roster()
    pretty = json.dumps(
        _payload(roster),
        indent=2,
        ensure_ascii=True,
    ).encode("ascii")

    with pytest.raises(
        TerminalStartupCohortError,
        match="encoding is not canonical",
    ):
        decode_terminal_nixl_source_roster(pretty)


def test_wire_rejects_uppercase_matrix_digest() -> None:
    """The matrix digest has exactly one lowercase hexadecimal spelling."""

    _, _, roster = _roster()
    payload = _payload(roster)
    matrix_sha256 = payload["matrix_sha256"]
    if type(matrix_sha256) is not str:
        raise AssertionError("encoded matrix digest is not a string")
    payload["matrix_sha256"] = matrix_sha256.upper()

    with pytest.raises(
        TerminalStartupCohortError,
        match="lowercase hexadecimal",
    ):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


@pytest.mark.parametrize(
    "payload_change",
    (
        {"schema": "packed-terminal-nixl-source-roster-v2"},
        {"requester_tensor_parallel_rank": True},
        {"routes": []},
    ),
)
def test_wire_rejects_schema_and_structural_drift(
    payload_change: dict[str, object],
) -> None:
    """Schema, exact integer types, and nonempty routes are closed.

    :param payload_change: Invalid top-level payload replacement.
    """

    _, _, roster = _roster()
    payload = _payload(roster)
    payload.update(payload_change)

    with pytest.raises(TerminalStartupCohortError):
        decode_terminal_nixl_source_roster(_canonical_wire(payload))


def test_wire_rejects_missing_and_extra_fields() -> None:
    """Neither the roster nor route schemas admit extension fields."""

    _, _, roster = _roster()
    missing = _payload(roster)
    del missing["requester_service_id"]
    extra = _payload(roster)
    _route_payload(extra)["unexpected"] = "value"

    with pytest.raises(TerminalStartupCohortError, match="field set is invalid"):
        decode_terminal_nixl_source_roster(_canonical_wire(missing))
    with pytest.raises(TerminalStartupCohortError, match="field set is invalid"):
        decode_terminal_nixl_source_roster(_canonical_wire(extra))


def test_decoder_rejects_empty_and_oversized_wire_images() -> None:
    """Wire bounds are enforced before JSON parsing."""

    for payload in (b"", b"x" * (TERMINAL_STARTUP_WIRE_MAX_BYTES + 1)):
        with pytest.raises(
            TerminalStartupCohortError,
            match="payload size is invalid",
        ):
            decode_terminal_nixl_source_roster(payload)


def test_encoder_rejects_roster_larger_than_wire_limit() -> None:
    """Base64 expansion cannot bypass the bounded startup envelope."""

    _, _, roster = _roster()
    oversized_route = dataclasses.replace(
        roster.routes[0],
        nixl_agent_metadata=b"x" * TERMINAL_STARTUP_WIRE_MAX_BYTES,
    )
    oversized = dataclasses.replace(
        roster,
        routes=(oversized_route, *roster.routes[1:]),
    )

    with pytest.raises(ValueError, match="exceeds the startup wire limit"):
        encode_terminal_nixl_source_roster(oversized)


def test_canonical_schema_constant_is_emitted() -> None:
    """The encoder always emits the one admitted schema version."""

    _, _, roster = _roster()

    assert _payload(roster)["schema"] == TERMINAL_NIXL_SOURCE_ROSTER_SCHEMA
