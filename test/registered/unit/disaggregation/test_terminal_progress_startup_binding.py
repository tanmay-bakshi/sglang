import concurrent.futures
import hashlib
import socket
import time
import uuid

import requests
from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.terminal_progress.cohort_expectation import (
    build_terminal_startup_cohort_expectation,
)
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentBootstrapEndpoint,
    TerminalDeploymentCohort,
    TerminalDeploymentDecoder,
    TerminalDeploymentPrefill,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalProducerClass,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    build_terminal_startup_rank_advertisement,
    join_terminal_startup_rank,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _free_port() -> int:
    """Reserve and release one loopback TCP port.

    :returns: Available listener port.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _cohort(port: int) -> TerminalDeploymentCohort:
    """Build one TP1 source and TP1 decoder cohort.

    :param port: Source-owned bootstrap listener port.
    :returns: Exact immutable deployment cohort.
    """

    return TerminalDeploymentCohort(
        group_id="group-a",
        model_fingerprint="11" * 32,
        logical_kv_layout_fingerprint="22" * 32,
        prefill=TerminalDeploymentPrefill(
            service_id="prefill-a",
            launch_instance_id=uuid.UUID(int=1),
            origin="http://127.0.0.1:32001",
            bootstrap_endpoint=TerminalDeploymentBootstrapEndpoint(
                host="127.0.0.1",
                port=port,
            ),
            tensor_parallel_size=1,
        ),
        decoders=(
            TerminalDeploymentDecoder(
                service_id="decode-a",
                launch_instance_id=uuid.UUID(int=2),
                origin="http://127.0.0.1:32002",
                tensor_parallel_size=1,
            ),
        ),
    )


def _members(cohort: TerminalDeploymentCohort):
    """Select both exact local service records.

    :param cohort: Complete fixture cohort.
    :returns: Source and decoder local memberships.
    """

    source = cohort.require_local_service(
        service_id="prefill-a",
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=uuid.UUID(int=1),
        tensor_parallel_size=1,
        origin="http://127.0.0.1:32001",
        bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
    )
    decode = cohort.require_local_service(
        service_id="decode-a",
        role=TerminalDeploymentRole.DECODE,
        launch_instance_id=uuid.UUID(int=2),
        tensor_parallel_size=1,
        origin="http://127.0.0.1:32002",
        bootstrap_endpoint=None,
    )
    return source, decode


def _wait_for_server(port: int) -> None:
    """Wait for the fixture bootstrap listener.

    :param port: Expected listener port.
    """

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            continue
    raise AssertionError("startup binding listener did not become live")


def test_rank_advertisement_binds_frozen_metadata_and_static_membership() -> None:
    """Native identity cannot select or mutate its static service row."""

    cohort = _cohort(_free_port())
    source, _ = _members(cohort)
    expectation = build_terminal_startup_cohort_expectation(cohort, source)
    metadata = b"complete-source-agent-metadata"

    advertisement = build_terminal_startup_rank_advertisement(
        cohort,
        source,
        expectation,
        tensor_parallel_rank=0,
        process_generation=str(uuid.UUID(int=101)),
        nixl_agent_name="source-agent",
        nixl_agent_metadata=metadata,
    )

    assert advertisement.nixl_agent_metadata_sha256 == hashlib.sha256(
        metadata
    ).digest()
    assert advertisement.service_id == "prefill-a"
    assert advertisement.launch_instance_id == uuid.UUID(int=1).bytes


def test_complete_http_join_returns_one_matrix_and_frozen_producer_plans() -> None:
    """Every rank receives the same epoch before producer numbering begins."""

    port = _free_port()
    cohort = _cohort(port)
    source, decode = _members(cohort)
    expectation = build_terminal_startup_cohort_expectation(cohort, source)
    registry = TerminalStartupCohortRegistry(expectation, timeout_seconds=2.0)
    server = CommonKVBootstrapServer(
        host="127.0.0.1",
        port=port,
        terminal_startup_registry=registry,
    )
    try:
        _wait_for_server(port)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            source_future = executor.submit(
                join_terminal_startup_rank,
                cohort,
                source,
                expectation,
                tensor_parallel_rank=0,
                process_generation=str(uuid.UUID(int=101)),
                nixl_agent_name="source-agent",
                nixl_agent_metadata=b"source-metadata",
                timeout_seconds=2.0,
            )
            decode_future = executor.submit(
                join_terminal_startup_rank,
                cohort,
                decode,
                expectation,
                tensor_parallel_rank=0,
                process_generation=str(uuid.UUID(int=102)),
                nixl_agent_name="decode-agent",
                nixl_agent_metadata=b"decode-metadata",
                timeout_seconds=2.0,
            )
            source_binding = source_future.result(timeout=3.0)
            decode_binding = decode_future.result(timeout=3.0)
    finally:
        server.close()

    assert source_binding.matrix == decode_binding.matrix
    assert source_binding.matrix.digest == decode_binding.matrix.digest
    assert source_binding.python_producers.fatal_producer_id == 0
    assert decode_binding.python_producers.fatal_producer_id == 0
    assert {
        spec.registration.producer_class
        for spec in source_binding.python_producers.specs
    } == {
        NativeTerminalProducerClass.LOCAL,
        NativeTerminalProducerClass.RECEIPT,
        NativeTerminalProducerClass.CONTROL,
    }
