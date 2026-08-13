import hashlib
import uuid
from pathlib import Path

import pytest
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentBootstrapEndpoint,
    TerminalDeploymentCohort,
    TerminalDeploymentDecoder,
    TerminalDeploymentPrefill,
    encode_terminal_deployment_cohort,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _cohort() -> TerminalDeploymentCohort:
    """Build one exact launcher cohort fixture.

    :returns: Valid per-group cohort.
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
                host="gemma-dev-1",
                port=32150,
            ),
            tensor_parallel_size=2,
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


def _write_cohort(tmp_path: Path) -> tuple[Path, str]:
    """Persist one canonical fixture and return its attestation.

    :param tmp_path: Isolated test directory.
    :returns: Artifact path and lowercase SHA-256.
    """

    payload = encode_terminal_deployment_cohort(_cohort())
    path = tmp_path / "cohort.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_prefill_server_args_bind_exact_local_cohort(tmp_path: Path) -> None:
    """Load and retain an exact prefill membership before runtime startup."""

    path, digest = _write_cohort(tmp_path)
    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="prefill",
        launch_instance_id=str(uuid.UUID(int=1)),
        host="0.0.0.0",
        port=32001,
        tp_size=2,
        disaggregation_bootstrap_port=32150,
        pd_prefill_bootstrap_advertise_host="gemma-dev-1",
        pd_model_fingerprint="11" * 32,
        pd_logical_kv_layout_fingerprint="22" * 32,
        pd_terminal_cohort_manifest=str(path),
        pd_terminal_cohort_sha256=digest,
        pd_terminal_local_service="prefill-a",
        pd_terminal_startup_timeout_seconds=60.0,
    )

    args._handle_pd_disaggregation()

    assert args.pd_terminal_deployment_cohort == _cohort()
    assert args.pd_terminal_local_membership is not None
    assert args.pd_terminal_local_membership.service_id == "prefill-a"


def test_decode_server_args_bind_exact_local_cohort(tmp_path: Path) -> None:
    """Load and retain an exact decoder membership before runtime startup."""

    path, digest = _write_cohort(tmp_path)
    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="decode",
        launch_instance_id=str(uuid.UUID(int=2)),
        port=32002,
        tp_size=1,
        pd_model_fingerprint="11" * 32,
        pd_logical_kv_layout_fingerprint="22" * 32,
        pd_terminal_cohort_manifest=str(path),
        pd_terminal_cohort_sha256=digest,
        pd_terminal_local_service="decode-a",
        pd_terminal_startup_timeout_seconds=60.0,
    )

    args._handle_pd_disaggregation()

    assert args.pd_terminal_local_membership is not None
    assert args.pd_terminal_local_membership.service_id == "decode-a"


def test_terminal_cohort_arguments_are_all_or_none(tmp_path: Path) -> None:
    """Reject a partial launcher attestation before any runtime construction."""

    path, _ = _write_cohort(tmp_path)
    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="decode",
        pd_terminal_cohort_manifest=str(path),
    )

    with pytest.raises(ValueError, match="configured together"):
        args._handle_pd_disaggregation()


def test_terminal_cohort_rejects_local_and_digest_drift(tmp_path: Path) -> None:
    """Fail closed on stale bytes or a mismatched local service incarnation."""

    path, _ = _write_cohort(tmp_path)
    digest_drift = ServerArgs(
        model_path="dummy",
        disaggregation_mode="decode",
        launch_instance_id=str(uuid.UUID(int=2)),
        port=32002,
        tp_size=1,
        pd_model_fingerprint="11" * 32,
        pd_logical_kv_layout_fingerprint="22" * 32,
        pd_terminal_cohort_manifest=str(path),
        pd_terminal_cohort_sha256="ff" * 32,
        pd_terminal_local_service="decode-a",
        pd_terminal_startup_timeout_seconds=60.0,
    )
    with pytest.raises(ValueError, match="digest differs"):
        digest_drift._handle_pd_disaggregation()

    _, digest = _write_cohort(tmp_path)
    identity_drift = ServerArgs(
        model_path="dummy",
        disaggregation_mode="decode",
        launch_instance_id=str(uuid.UUID(int=3)),
        port=32002,
        tp_size=1,
        pd_model_fingerprint="11" * 32,
        pd_logical_kv_layout_fingerprint="22" * 32,
        pd_terminal_cohort_manifest=str(path),
        pd_terminal_cohort_sha256=digest,
        pd_terminal_local_service="decode-a",
        pd_terminal_startup_timeout_seconds=60.0,
    )
    with pytest.raises(ValueError, match="differs"):
        identity_drift._handle_pd_disaggregation()
