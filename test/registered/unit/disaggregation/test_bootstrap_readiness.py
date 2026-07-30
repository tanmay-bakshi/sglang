import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def _registration(tp_rank: int, *, rank_port: int) -> dict[str, object]:
    """Build one TP2 prefill rank registration.

    :param tp_rank: Source tensor-parallel rank.
    :param rank_port: Rank-local transfer control port.
    :returns: Registration JSON.
    """

    return {
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


def test_bootstrap_readiness_counts_unique_ranks() -> None:
    """Duplicate registration replaces a rank without manufacturing readiness."""

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
            json=_registration(0, rank_port=31001),
            timeout=1,
        )
        assert duplicate.status_code == 200
        assert server.registered_count == 1
        assert requests.get(f"{base_url}/ready", timeout=1).status_code == 503

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
