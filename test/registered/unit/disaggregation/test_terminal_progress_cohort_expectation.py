import dataclasses
import hashlib
import uuid

import pytest
from sglang.srt.disaggregation.terminal_progress.cohort_expectation import (
    build_terminal_startup_cohort_expectation,
)
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentCohort,
    TerminalDeploymentCohortError,
    TerminalDeploymentLocalService,
    TerminalDeploymentRole,
    decode_terminal_deployment_cohort,
    encode_terminal_deployment_cohort,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_LAUNCHER_CANONICAL_COHORT = (
    b'{"schema":"packed-terminal-deployment-cohort-v1","group_id":"group-a",'
    b'"model_fingerprint":"111111111111111111111111111111111111111111111111'
    b'1111111111111111","logical_kv_layout_fingerprint":"222222222222222222222222'
    b'2222222222222222222222222222222222222222","prefill":{"id":"prefill-a",'
    b'"launch_instance_id":"00000000-0000-0000-0000-000000000002","origin":'
    b'"http://127.0.0.1:32001","bootstrap_endpoint":{"host":"gemma-dev-1",'
    b'"port":32150},"tensor_parallel_size":2},"decoders":[{"id":"decode-a",'
    b'"launch_instance_id":"00000000-0000-0000-0000-000000000003","origin":'
    b'"http://127.0.0.1:32002","tensor_parallel_size":1},{"id":"decode-b",'
    b'"launch_instance_id":"00000000-0000-0000-0000-000000000004","origin":'
    b'"http://127.0.0.1:32003","tensor_parallel_size":1}]}'
)
_LAUNCHER_CANONICAL_COHORT_SHA256 = bytes.fromhex(
    "75cff91a961a3a2d749e84ae38a485da0d6f703bd0c47187b4de5a4a0a8acab8"
)


def _cohort() -> TerminalDeploymentCohort:
    """Load the launcher-owned golden deployment cohort.

    :returns: Canonical typed cohort.
    """

    return decode_terminal_deployment_cohort(_LAUNCHER_CANONICAL_COHORT)


def _local_service(
    cohort: TerminalDeploymentCohort,
    role: TerminalDeploymentRole,
) -> TerminalDeploymentLocalService:
    """Select one exact local service from the golden cohort.

    :param cohort: Canonical deployment cohort.
    :param role: Local service role to select.
    :returns: Exact local membership.
    """

    if role is TerminalDeploymentRole.PREFILL:
        return cohort.require_local_service(
            service_id=cohort.prefill.service_id,
            role=role,
            launch_instance_id=cohort.prefill.launch_instance_id,
            tensor_parallel_size=cohort.prefill.tensor_parallel_size,
            origin=cohort.prefill.origin,
            bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
        )
    decoder = cohort.decoders[1]
    return cohort.require_local_service(
        service_id=decoder.service_id,
        role=role,
        launch_instance_id=decoder.launch_instance_id,
        tensor_parallel_size=decoder.tensor_parallel_size,
        origin=decoder.origin,
        bootstrap_endpoint=None,
    )


def test_launcher_bytes_map_to_exact_startup_expectation() -> None:
    """Bind the launcher golden bytes to one complete startup expectation."""

    cohort = _cohort()
    local_service = _local_service(cohort, TerminalDeploymentRole.PREFILL)
    expectation = build_terminal_startup_cohort_expectation(cohort, local_service)

    assert encode_terminal_deployment_cohort(cohort) == _LAUNCHER_CANONICAL_COHORT
    assert hashlib.sha256(_LAUNCHER_CANONICAL_COHORT).digest() == (
        _LAUNCHER_CANONICAL_COHORT_SHA256
    )
    assert cohort.digest == _LAUNCHER_CANONICAL_COHORT_SHA256
    assert expectation.group_id == "group-a"
    assert expectation.cohort_sha256 == _LAUNCHER_CANONICAL_COHORT_SHA256
    assert expectation.expected_rank_count == 4
    assert tuple(service.service_id for service in expectation.services) == (
        "prefill-a",
        "decode-a",
        "decode-b",
    )
    assert tuple(service.role for service in expectation.services) == (
        TerminalOwnerRole.SOURCE,
        TerminalOwnerRole.DECODE,
        TerminalOwnerRole.DECODE,
    )
    assert tuple(service.launch_instance_id for service in expectation.services) == (
        uuid.UUID(int=2).bytes,
        uuid.UUID(int=3).bytes,
        uuid.UUID(int=4).bytes,
    )
    assert tuple(service.tensor_parallel_size for service in expectation.services) == (
        2,
        1,
        1,
    )


def test_every_exact_local_member_produces_the_same_complete_expectation() -> None:
    """Local role selection cannot narrow or alter static startup membership."""

    cohort = _cohort()
    source_expectation = build_terminal_startup_cohort_expectation(
        cohort,
        _local_service(cohort, TerminalDeploymentRole.PREFILL),
    )
    decode_expectation = build_terminal_startup_cohort_expectation(
        cohort,
        _local_service(cohort, TerminalDeploymentRole.DECODE),
    )

    assert source_expectation == decode_expectation
    decode_b = decode_expectation.service("decode-b")
    assert decode_b.service_origin == "http://127.0.0.1:32003"
    assert decode_b.role is TerminalOwnerRole.DECODE
    assert decode_b.launch_instance_id == uuid.UUID(int=4).bytes


def test_adapter_rejects_stale_epoch_and_local_row_drift() -> None:
    """Never guess or repair a local service identity at the model boundary."""

    cohort = _cohort()
    local_service = _local_service(cohort, TerminalDeploymentRole.DECODE)
    stale_epoch = dataclasses.replace(
        local_service,
        cohort_digest=bytes.fromhex("ff" * 32),
    )
    with pytest.raises(TerminalDeploymentCohortError, match="another deployment"):
        build_terminal_startup_cohort_expectation(cohort, stale_epoch)

    drifted_origin = dataclasses.replace(
        local_service,
        origin="http://127.0.0.1:32999",
    )
    with pytest.raises(TerminalDeploymentCohortError, match="differs"):
        build_terminal_startup_cohort_expectation(cohort, drifted_origin)
