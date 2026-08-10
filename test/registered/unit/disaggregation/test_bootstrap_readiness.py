import base64
import hashlib
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from sglang.srt.disaggregation.common.conn import (
    NIXL_AGENT_METADATA_MAX_BYTES,
    NIXL_AGENT_NAME_MAX_BYTES,
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    SERIALIZED_RANK_LIMIT,
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PACKED_SOURCE_GEOMETRY = (
    '{"components":[{"item_lens":[32768],"layer_ids":[0],"page_size":64,'
    '"state_index":null,"state_type":null}],"schema_version":1}'
)


def _free_port() -> int:
    """Reserve and release one loopback port for an in-process server.

    :returns: Available TCP port.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_liveness(base_url: str) -> None:
    """Wait for the bootstrap thread to bind its listener.

    :param base_url: Bootstrap service origin.
    """

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=0.2)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            return
    raise RuntimeError("bootstrap test server did not become live")


def _registration(
    tp_rank: int, *, rank_port: int, with_nixl_identity: bool = False
) -> dict[str, object]:
    """Build one TP2 prefill rank registration.

    :param tp_rank: Source tensor-parallel rank.
    :param rank_port: Rank-local transfer control port.
    :returns: Registration JSON.
    """

    registration: dict[str, object] = {
        "attn_tp_size": 2,
        "attn_tp_rank": tp_rank,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "127.0.0.1",
        "rank_port": rank_port,
        "page_size": 64,
        "kv_cache_dtype": "bfloat16",
        "load_balance_method": "follow_bootstrap_room",
        "enable_dsa_cache_layer_split": False,
        "prefill_http_port": 30000,
    }
    if with_nixl_identity:
        metadata = f"nixl-metadata-{tp_rank}".encode("ascii")
        registration.update(
            {
                "transport_protocol": NIXL_BOOTSTRAP_PEER_PROTOCOL,
                "nixl_agent_name": f"prefill-agent-{tp_rank}",
                "nixl_agent_metadata": base64.b64encode(metadata).decode("ascii"),
                "nixl_agent_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                "process_generation": f"00000000-0000-4000-8000-{tp_rank:012d}",
                "transfer_source_rank": tp_rank,
                "packed_source_geometry": PACKED_SOURCE_GEOMETRY,
                "packed_source_geometry_sha256": hashlib.sha256(
                    PACKED_SOURCE_GEOMETRY.encode("ascii")
                ).hexdigest(),
            }
        )
    return registration


def test_bootstrap_readiness_counts_unique_ranks() -> None:
    """Only an identical duplicate is idempotent for one process rank."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)

        first = requests.put(
            f"{base_url}/route",
            json=_registration(0, rank_port=31000),
            timeout=1,
        )
        assert first.status_code == 200
        assert server.registered_count == 1
        assert requests.get(f"{base_url}/ready", timeout=1).status_code == 503

        duplicate = requests.put(
            f"{base_url}/route",
            json=_registration(0, rank_port=31000),
            timeout=1,
        )
        assert duplicate.status_code == 200
        assert server.registered_count == 1
        assert requests.get(f"{base_url}/ready", timeout=1).status_code == 503

        conflicting = requests.put(
            f"{base_url}/route",
            json=_registration(0, rank_port=31001),
            timeout=1,
        )
        assert conflicting.status_code == 409
        assert server.registered_count == 1

        second = requests.put(
            f"{base_url}/route",
            json=_registration(1, rank_port=31002),
            timeout=1,
        )
        assert second.status_code == 200
        server.wait_until_ready(timeout_s=0.2)
        assert server.registered_count == 2
        assert requests.get(f"{base_url}/ready", timeout=1).status_code == 200
    finally:
        server.close()


def test_nixl_peer_identity_round_trips_for_every_rank() -> None:
    """A decoder can fetch exact metadata and source identity for each writer."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        for tp_rank in range(2):
            response = requests.put(
                f"{base_url}/route",
                json=_registration(
                    tp_rank,
                    rank_port=31000 + tp_rank,
                    with_nixl_identity=True,
                ),
                timeout=1,
            )
            assert response.status_code == 200

        for tp_rank in range(2):
            route = requests.get(
                f"{base_url}/route",
                params={
                    "prefill_dp_rank": 0,
                    "prefill_cp_rank": 0,
                    "target_tp_rank": tp_rank,
                    "target_pp_rank": 0,
                },
                timeout=1,
            )
            assert route.status_code == 200
            payload = route.json()
            assert payload["transport_protocol"] == NIXL_BOOTSTRAP_PEER_PROTOCOL
            assert payload["nixl_agent_name"] == f"prefill-agent-{tp_rank}"
            assert payload["transfer_source_rank"] == tp_rank
            assert payload["attn_tp_rank"] == tp_rank
            assert payload["process_generation"] == (
                f"00000000-0000-4000-8000-{tp_rank:012d}"
            )
            assert payload["packed_source_geometry"] == PACKED_SOURCE_GEOMETRY
    finally:
        server.close()


def test_bootstrap_rejects_cross_rank_source_geometry_disagreement() -> None:
    """Packed source component geometry is one cohort-wide bootstrap fact."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        first = _registration(0, rank_port=31000, with_nixl_identity=True)
        first_response = requests.put(
            f"{base_url}/route",
            json=first,
            timeout=1,
        )
        assert first_response.status_code == 200

        second = _registration(1, rank_port=31001, with_nixl_identity=True)
        divergent = PACKED_SOURCE_GEOMETRY.replace("32768", "16384")
        second["packed_source_geometry"] = divergent
        second["packed_source_geometry_sha256"] = hashlib.sha256(
            divergent.encode("ascii")
        ).hexdigest()
        response = requests.put(f"{base_url}/route", json=second, timeout=1)

        assert response.status_code == 409
        assert "inconsistent bootstrap packed_source_geometry" in response.text
        assert server.registered_count == 1
    finally:
        server.close()


def test_nixl_registration_rejects_missing_metadata() -> None:
    """A partial native identity never contributes to bootstrap readiness."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        registration = _registration(0, rank_port=31000, with_nixl_identity=True)
        del registration["nixl_agent_metadata"]

        response = requests.put(f"{base_url}/route", json=registration, timeout=1)

        assert response.status_code == 400
        assert server.registered_count == 0
    finally:
        server.close()


def test_nixl_registration_rejects_non_string_metadata() -> None:
    """Malformed native metadata receives a bounded client error, not a 500."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        registration = _registration(0, rank_port=31000, with_nixl_identity=True)
        registration["nixl_agent_metadata"] = 17

        response = requests.put(f"{base_url}/route", json=registration, timeout=1)

        assert response.status_code == 400
        assert server.registered_count == 0
    finally:
        server.close()


def test_bootstrap_rejects_non_uint32_ranks() -> None:
    """Serialized topology and source ranks are strict uint32 values."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    rank_fields = (
        "attn_tp_rank",
        "attn_cp_rank",
        "attn_dp_rank",
        "pp_rank",
        "system_dp_rank",
        "transfer_source_rank",
    )
    try:
        _wait_for_liveness(base_url)
        for field_name in rank_fields:
            for invalid_value in (True, "0", SERIALIZED_RANK_LIMIT):
                registration = _registration(
                    0,
                    rank_port=31000,
                    with_nixl_identity=True,
                )
                registration[field_name] = invalid_value

                response = requests.put(
                    f"{base_url}/route",
                    json=registration,
                    timeout=1,
                )

                assert response.status_code == 400, (
                    field_name,
                    invalid_value,
                    response.text,
                )
        assert server.registered_count == 0
    finally:
        server.close()


def test_bootstrap_accepts_maximum_uint32_rank() -> None:
    """The inclusive uint32 maximum remains a valid rank coordinate."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        registration = _registration(
            0,
            rank_port=31000,
            with_nixl_identity=True,
        )
        registration["attn_tp_size"] = SERIALIZED_RANK_LIMIT
        registration["attn_tp_rank"] = SERIALIZED_RANK_LIMIT - 1
        registration["transfer_source_rank"] = SERIALIZED_RANK_LIMIT - 1

        response = requests.put(
            f"{base_url}/route",
            json=registration,
            timeout=1,
        )

        assert response.status_code == 200
        assert server.registered_count == 1
    finally:
        server.close()


def test_bootstrap_rejects_unbounded_nixl_identity() -> None:
    """Peer names and metadata cannot create unbounded bootstrap state."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        for invalid_name in (
            "prefill-\N{SNOWMAN}",
            "a" * (NIXL_AGENT_NAME_MAX_BYTES + 1),
        ):
            registration = _registration(
                0,
                rank_port=31000,
                with_nixl_identity=True,
            )
            registration["nixl_agent_name"] = invalid_name
            response = requests.put(
                f"{base_url}/route",
                json=registration,
                timeout=1,
            )
            assert response.status_code == 400

        oversized_metadata = b"x" * (NIXL_AGENT_METADATA_MAX_BYTES + 1)
        registration = _registration(
            0,
            rank_port=31000,
            with_nixl_identity=True,
        )
        registration["nixl_agent_metadata"] = base64.b64encode(
            oversized_metadata
        ).decode("ascii")
        registration["nixl_agent_metadata_sha256"] = hashlib.sha256(
            oversized_metadata
        ).hexdigest()
        response = requests.put(
            f"{base_url}/route",
            json=registration,
            timeout=1,
        )

        assert response.status_code == 400
        assert server.registered_count == 0
    finally:
        server.close()


def test_nixl_registration_rejects_stale_process_generation() -> None:
    """A live rank route cannot be replaced by another process generation."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        registration = _registration(0, rank_port=31000, with_nixl_identity=True)
        first = requests.put(f"{base_url}/route", json=registration, timeout=1)
        assert first.status_code == 200

        stale = dict(registration)
        stale["process_generation"] = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        response = requests.put(f"{base_url}/route", json=stale, timeout=1)

        assert response.status_code == 409
        assert server.registered_count == 1
    finally:
        server.close()


def test_bootstrap_rejects_rank_metadata_disagreement() -> None:
    """One process generation cannot mix incompatible KV layouts."""

    port = _free_port()
    server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_liveness(base_url)
        assert (
            requests.put(
                f"{base_url}/route",
                json=_registration(0, rank_port=31000),
                timeout=1,
            ).status_code
            == 200
        )

        mismatched = _registration(1, rank_port=31001)
        mismatched["page_size"] = 32
        response = requests.put(
            f"{base_url}/route",
            json=mismatched,
            timeout=1,
        )

        assert response.status_code == 409
        assert server.registered_count == 1
    finally:
        server.close()


def test_registration_failure_aborts_scheduler_initialization() -> None:
    """A prefill scheduler cannot become ready without bootstrap ownership."""

    manager = object.__new__(CommonKVManager)
    manager.dist_init_addr = None
    manager.bootstrap_host = "127.0.0.1"
    manager.bootstrap_port = 8998
    manager.attn_tp_size = 1
    manager.attn_tp_rank = 0
    manager.attn_cp_size = 1
    manager.attn_cp_rank = 0
    manager.attn_dp_size = 1
    manager.attn_dp_rank = 0
    manager.pp_size = 1
    manager.pp_rank = 0
    manager.system_dp_size = 1
    manager.system_dp_rank = 0
    manager.local_ip = "127.0.0.1"
    manager.rank_port = 31000
    manager.kv_args = SimpleNamespace(page_size=64)
    manager.server_args = SimpleNamespace(
        load_balance_method="follow_bootstrap_room",
        enable_dsa_cache_layer_split=False,
        port=30000,
    )
    failed_response = SimpleNamespace(status_code=503, text="not ready")
    model = SimpleNamespace(kv_cache_dtype="bfloat16")

    with (
        patch(
            "sglang.srt.disaggregation.common.conn.requests.put",
            return_value=failed_response,
        ),
        patch("sglang.srt.disaggregation.common.conn.time.sleep"),
        patch(
            "sglang.srt.disaggregation.common.conn.get_model",
            return_value=model,
        ),
        pytest.raises(RuntimeError, match="failed to register"),
    ):
        manager.register_to_bootstrap()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
