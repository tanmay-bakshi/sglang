import dataclasses
import hashlib
import json
import uuid

import pytest

from sglang.srt.disaggregation.nixl.startup_decode_routes import (
    TERMINAL_DECODE_CONTROL_ROUTE_TABLE_SCHEMA,
    TerminalDecodeControlRoute,
    TerminalDecodeControlRouteError,
    TerminalDecodeControlRouteTable,
    build_terminal_decode_control_route_table,
    decode_terminal_decode_control_route_table,
    encode_terminal_decode_control_route_table,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_DIGEST = bytes.fromhex("11" * 32)


def _rank(
    service_id: str,
    role: TerminalOwnerRole,
    tp_rank: int,
    tp_size: int,
    generation: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact startup row.

    :param service_id: Static service identifier.
    :param role: Source or decoder role.
    :param tp_rank: Rank within its TP service.
    :param tp_size: Exact service TP width.
    :param generation: Non-nil process generation integer.
    :returns: Generation-bound startup advertisement.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        service_id=service_id,
        service_origin=(
            "http://127.0.0.1:32001"
            if role is TerminalOwnerRole.SOURCE
            else "http://127.0.0.1:32002"
        ),
        role=role,
        launch_instance_id=uuid.UUID(
            int=1 if role is TerminalOwnerRole.SOURCE else 2
        ).bytes,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=f"{role.value}-agent-{tp_rank}",
        nixl_agent_metadata_sha256=hashlib.sha256(
            f"{role.value}-metadata-{tp_rank}".encode("ascii")
        ).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build one TP2 source and TP4 decoder matrix.

    :returns: Complete canonical startup matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        ranks=(
            _rank("prefill-a", TerminalOwnerRole.SOURCE, 0, 2, 101),
            _rank("prefill-a", TerminalOwnerRole.SOURCE, 1, 2, 102),
            *tuple(
                _rank("decode-a", TerminalOwnerRole.DECODE, rank, 4, 201 + rank)
                for rank in range(4)
            ),
        ),
    )


def _registration(rank: int) -> tuple[bytes, ...]:
    """Build one exact guarded synthetic registration.

    :param rank: Decoder TP rank.
    :returns: Complete immutable multipart registration.
    """

    return (
        b"NixlMsgGuard",
        b"None",
        b"127.0.0.1",
        str(33000 + rank).encode("ascii"),
        f"decode-agent-{rank}".encode("ascii"),
    )


def _table() -> TerminalDecodeControlRouteTable:
    """Build one complete TP4 decoder listener table.

    :returns: Matrix-authenticated immutable routes.
    """

    matrix = _matrix()
    return build_terminal_decode_control_route_table(
        matrix,
        "decode-a",
        tuple(
            (
                matrix.rank("decode-a", rank),
                NetworkAddress("127.0.0.1", 33000 + rank),
                _registration(rank),
            )
            for rank in range(4)
        ),
    )


def _binding(rank: int) -> TerminalStartupRankBinding:
    """Build one decoder rank binding.

    :param rank: Local decoder TP rank.
    :returns: Complete startup and producer authority.
    """

    matrix = _matrix()
    return TerminalStartupRankBinding(
        advertisement=matrix.rank("decode-a", rank),
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id="decode-a",
            local_tensor_parallel_rank=rank,
            first_producer_id=0,
        ),
    )


def test_route_table_round_trip_binds_every_tp_rank_and_listener() -> None:
    """Canonical bytes preserve complete TP membership and exact endpoints."""

    table = _table()
    encoded = encode_terminal_decode_control_route_table(table)
    decoded = decode_terminal_decode_control_route_table(_matrix(), encoded)

    assert encoded.startswith(
        b'{"schema":"'
        + TERMINAL_DECODE_CONTROL_ROUTE_TABLE_SCHEMA.encode("ascii")
        + b'"'
    )
    assert decoded == table
    assert tuple(identity.tp_rank for identity in decoded.identities) == (0, 1, 2, 3)
    assert decoded.digest == hashlib.sha256(encoded).digest()
    assert encode_terminal_decode_control_route_table(decoded) == encoded


@pytest.mark.parametrize("rank", range(4))
def test_every_rank_authenticates_its_actual_listener_and_registration(rank: int) -> None:
    """Every consumer proves the source-agreed table contains its own route.

    :param rank: Local decoder TP rank.
    """

    _table().require_local_registration(
        _binding(rank),
        NetworkAddress("127.0.0.1", 33000 + rank),
        _registration(rank),
    )


def test_local_route_rejects_listener_and_registration_drift() -> None:
    """A source cannot redirect or substitute the local decoder listener."""

    table = _table()
    binding = _binding(2)
    with pytest.raises(TerminalDecodeControlRouteError, match="local manager"):
        table.require_local_registration(
            binding,
            NetworkAddress("127.0.0.1", 33999),
            _registration(2),
        )
    with pytest.raises(TerminalDecodeControlRouteError, match="registration"):
        table.require_local_registration(
            binding,
            NetworkAddress("127.0.0.1", 33002),
            (*_registration(2), b"drift"),
        )


def test_route_table_rejects_incomplete_duplicate_and_stale_membership() -> None:
    """The immutable table has one unique route for every exact TP rank."""

    table = _table()
    with pytest.raises(ValueError, match="incomplete"):
        dataclasses.replace(table, routes=table.routes[:-1])
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(
            table,
            routes=(
                table.routes[0],
                dataclasses.replace(
                    table.routes[1],
                    endpoint=table.routes[0].endpoint,
                ),
                *table.routes[2:],
            ),
        )
    stale = dataclasses.replace(
        table.routes[1].startup_rank,
        process_generation=uuid.UUID(int=999).bytes,
    )
    stale_table = dataclasses.replace(
        table,
        routes=(
            table.routes[0],
            dataclasses.replace(table.routes[1], startup_rank=stale),
            *table.routes[2:],
        ),
    )
    with pytest.raises(TerminalDecodeControlRouteError, match="membership"):
        stale_table.require_matrix(_matrix())


def test_codec_rejects_endpoint_drift_and_noncanonical_json() -> None:
    """Wire mutations cannot acquire control-route authority."""

    encoded = encode_terminal_decode_control_route_table(_table())
    payload = json.loads(encoded)
    payload["routes"][0]["process_generation"] = str(uuid.UUID(int=999))
    mutated = json.dumps(payload, separators=(",", ":")).encode()
    with pytest.raises(TerminalDecodeControlRouteError, match="startup rank"):
        decode_terminal_decode_control_route_table(_matrix(), mutated)

    alternate = json.dumps(json.loads(encoded), indent=2).encode()
    with pytest.raises(TerminalDecodeControlRouteError, match="not canonical"):
        decode_terminal_decode_control_route_table(_matrix(), alternate)


def test_route_value_rejects_source_role() -> None:
    """A source listener cannot enter a decoder route table."""

    matrix = _matrix()
    with pytest.raises(ValueError, match="decoder ranks"):
        TerminalDecodeControlRoute(
            startup_rank=matrix.rank("prefill-a", 0),
            endpoint=NetworkAddress("127.0.0.1", 33000),
            registration_multipart_sha256=bytes.fromhex("31" * 32),
        )
