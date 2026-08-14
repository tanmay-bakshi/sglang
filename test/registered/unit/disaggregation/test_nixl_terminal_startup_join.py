import types
import uuid
from unittest.mock import patch

import pytest
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentBootstrapEndpoint,
    TerminalDeploymentCohort,
    TerminalDeploymentDecoder,
    TerminalDeploymentPrefill,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _cohort() -> TerminalDeploymentCohort:
    """Build one source-TP2 and decoder-TP1 deployment.

    :returns: Exact immutable fixture cohort.
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
                port=31001,
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


def _source_manager() -> NixlKVManager:
    """Build the initialized fields consumed by the startup join.

    :returns: Source-rank manager shell.
    """

    cohort = _cohort()
    membership = cohort.require_local_service(
        service_id="prefill-a",
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=uuid.UUID(int=1),
        tensor_parallel_size=2,
        origin="http://127.0.0.1:32001",
        bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
    )
    manager = object.__new__(NixlKVManager)
    manager.server_args = types.SimpleNamespace(
        pd_terminal_deployment_cohort=cohort,
        pd_terminal_local_membership=membership,
        pd_terminal_startup_timeout_seconds=60.0,
    )
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.pp_size = 1
    manager.attn_cp_size = 1
    manager.attn_tp_size = 2
    manager.attn_tp_rank = 1
    manager.process_generation = str(uuid.UUID(int=101))
    manager.agent = types.SimpleNamespace(name="source-agent-rank-1")
    manager.agent_metadata = b"complete-source-agent-metadata"
    return manager


def test_manager_joins_after_complete_native_metadata_is_frozen() -> None:
    """The composition boundary forwards the exact initialized rank identity."""

    manager = _source_manager()
    binding = object()

    with patch(
        "sglang.srt.disaggregation.nixl.conn.join_terminal_startup_rank",
        return_value=binding,
    ) as join:
        observed = manager._join_terminal_startup_cohort()

    assert observed is binding
    expectation = join.call_args.args[2]
    assert (
        expectation.cohort_sha256
        == manager.server_args.pd_terminal_deployment_cohort.digest
    )
    join.assert_called_once_with(
        manager.server_args.pd_terminal_deployment_cohort,
        manager.server_args.pd_terminal_local_membership,
        expectation,
        tensor_parallel_rank=1,
        process_generation=str(uuid.UUID(int=101)),
        nixl_agent_name="source-agent-rank-1",
        nixl_agent_metadata=b"complete-source-agent-metadata",
        timeout_seconds=60.0,
    )


def test_manager_rejects_runtime_topology_drift_before_join() -> None:
    """A manager cannot claim a launcher-selected service with another TP width."""

    manager = _source_manager()
    manager.attn_tp_size = 4

    with pytest.raises(ValueError, match="TP width differs"):
        manager._join_terminal_startup_cohort()


def test_manager_skips_startup_join_only_when_fully_unconfigured() -> None:
    """Legacy deployments retain no partial terminal startup authority."""

    manager = _source_manager()
    manager.server_args = types.SimpleNamespace(
        pd_terminal_deployment_cohort=None,
        pd_terminal_local_membership=None,
        pd_terminal_startup_timeout_seconds=None,
    )

    assert manager._join_terminal_startup_cohort() is None


def test_manager_rejects_partial_terminal_configuration() -> None:
    """Missing one immutable startup input fails before native peers escape."""

    manager = _source_manager()
    manager.server_args.pd_terminal_local_membership = None

    with pytest.raises(ValueError, match="configured together"):
        manager._join_terminal_startup_cohort()
