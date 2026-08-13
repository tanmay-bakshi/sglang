import concurrent.futures
import hashlib
import socket
import time
import uuid

import requests

from sglang.srt.disaggregation.nixl.conn import NixlKVBootstrapServer
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortExpectation,
    TerminalStartupCohortRegistry,
    TerminalStartupRankAdvertisement,
    TerminalStartupServiceExpectation,
)
from sglang.srt.disaggregation.terminal_progress.startup_http import (
    TERMINAL_STARTUP_ROUTE,
    join_terminal_startup_cohort,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_SHA256 = hashlib.sha256(b"startup-http-cohort").digest()


def _uuid_bytes(marker: int) -> bytes:
    """Build one stable non-nil UUID fixture.

    :param marker: Positive low-order integer.
    :returns: Exact UUID bytes.
    """

    return uuid.UUID(int=marker).bytes


def _free_port() -> int:
    """Reserve and release one loopback port.

    :returns: Available TCP listener port.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _expectation() -> TerminalStartupCohortExpectation:
    """Build one source rank and one decode rank.

    :returns: Complete static startup population.
    """

    return TerminalStartupCohortExpectation(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        services=(
            TerminalStartupServiceExpectation(
                service_id="prefill-a",
                service_origin="http://127.0.0.1:32000",
                role=TerminalOwnerRole.SOURCE,
                launch_instance_id=_uuid_bytes(1),
                tensor_parallel_size=1,
            ),
            TerminalStartupServiceExpectation(
                service_id="decode-a",
                service_origin="http://127.0.0.1:32001",
                role=TerminalOwnerRole.DECODE,
                launch_instance_id=_uuid_bytes(2),
                tensor_parallel_size=1,
            ),
        ),
    )


def _advertisement(
    service: TerminalStartupServiceExpectation,
    generation_marker: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact native rank advertisement.

    :param service: Static service identity.
    :param generation_marker: Unique process-generation marker.
    :returns: Generation-bound rank advertisement.
    """

    agent_name = f"nixl-{service.service_id}"
    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        service_id=service.service_id,
        service_origin=service.service_origin,
        role=service.role,
        launch_instance_id=service.launch_instance_id,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        process_generation=_uuid_bytes(generation_marker),
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(agent_name.encode()).digest(),
    )


def _wait_for_server(base_url: str) -> None:
    """Wait for the in-process bootstrap listener.

    :param base_url: Exact loopback bootstrap origin.
    """

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=0.1).status_code == 200:
                return
        except requests.RequestException:
            continue
    raise AssertionError("terminal startup test listener did not become live")


def test_http_join_releases_complete_cohort_without_blocking_event_loop() -> None:
    """Earlier joins sleep off-loop while later ranks and health remain live."""

    expectation = _expectation()
    registry = TerminalStartupCohortRegistry(expectation, timeout_seconds=2.0)
    port = _free_port()
    server = NixlKVBootstrapServer(
        host="127.0.0.1",
        port=port,
        terminal_startup_registry=registry,
    )
    base_url = f"http://127.0.0.1:{port}"
    endpoint = f"{base_url}{TERMINAL_STARTUP_ROUTE}"
    source = _advertisement(expectation.services[0], 101)
    decode = _advertisement(expectation.services[1], 102)
    try:
        _wait_for_server(base_url)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            source_join = executor.submit(
                join_terminal_startup_cohort,
                endpoint,
                source,
                2.0,
            )
            deadline = time.monotonic() + 1.0
            while registry.snapshot().registered_rank_count != 1:
                if time.monotonic() >= deadline:
                    raise AssertionError("source join did not reach the registry")
                time.sleep(0.001)
            assert not source_join.done()
            assert requests.get(f"{base_url}/health", timeout=0.2).status_code == 200
            decode_join = executor.submit(
                join_terminal_startup_cohort,
                endpoint,
                decode,
                2.0,
            )
            source_matrix = source_join.result(timeout=2.0)
            decode_matrix = decode_join.result(timeout=2.0)
        assert source_matrix == decode_matrix
        source_matrix.require_expectation(expectation)
    finally:
        server.close()


def test_unconfigured_bootstrap_has_no_terminal_startup_route() -> None:
    """A direct baseline cannot accidentally expose cohort admission."""

    port = _free_port()
    server = NixlKVBootstrapServer(host="127.0.0.1", port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)
        response = requests.post(
            f"{base_url}{TERMINAL_STARTUP_ROUTE}",
            data=b"{}",
            timeout=0.2,
        )
        assert response.status_code == 404
    finally:
        server.close()
