import uuid

import pytest

from sglang.srt.disaggregation.base.conn import BaseKVBootstrapServer
from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentBootstrapEndpoint,
    TerminalDeploymentCohort,
    TerminalDeploymentDecoder,
    TerminalDeploymentPrefill,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortRegistry,
)
from sglang.srt.disaggregation.utils import KVClassType, TransferBackend
from sglang.srt.managers import disagg_service
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BootstrapServer(BaseKVBootstrapServer):
    """Capture the exact startup registry composed for one prefill service."""

    host: str
    port: int
    terminal_startup_registry: TerminalStartupCohortRegistry

    def __init__(
        self,
        host: str,
        port: int,
        terminal_startup_registry: TerminalStartupCohortRegistry,
    ) -> None:
        """Retain constructor inputs without opening a listener.

        :param host: Bootstrap listener host.
        :param port: Bootstrap listener port.
        :param terminal_startup_registry: Exact startup join registry.
        """

        self.host = host
        self.port = port
        self.terminal_startup_registry = terminal_startup_registry

    def wait_until_ready(self, timeout_s: float) -> None:
        """Satisfy the bootstrap interface without opening a listener.

        :param timeout_s: Unused readiness timeout.
        """


def _bootstrap_class(
    transfer_backend: TransferBackend,
    class_type: KVClassType,
) -> type[BaseKVBootstrapServer]:
    """Select the listener-free bootstrap fixture.

    :param transfer_backend: Requested transfer backend.
    :param class_type: Requested KV component kind.
    :returns: Listener-free bootstrap fixture class.
    """

    assert transfer_backend is TransferBackend.NIXL
    assert class_type is KVClassType.BOOTSTRAP_SERVER
    return _BootstrapServer


def _cohort() -> TerminalDeploymentCohort:
    """Build one exact startup deployment.

    :returns: Immutable prefill and decoder cohort.
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


def test_prefill_composes_startup_expectation_at_service_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project deployment membership only when constructing the NIXL service.

    :param monkeypatch: Isolated bootstrap-class replacement.
    """

    cohort = _cohort()
    membership = cohort.require_local_service(
        service_id="prefill-a",
        role=TerminalDeploymentRole.PREFILL,
        launch_instance_id=cohort.prefill.launch_instance_id,
        tensor_parallel_size=cohort.prefill.tensor_parallel_size,
        origin=cohort.prefill.origin,
        bootstrap_endpoint=cohort.prefill.bootstrap_endpoint,
    )
    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="nixl",
        host="0.0.0.0",
        disaggregation_bootstrap_port=32150,
        pd_terminal_startup_timeout_seconds=60.0,
    )
    args.pd_terminal_deployment_cohort = cohort
    args.pd_terminal_local_membership = membership
    monkeypatch.setattr(disagg_service, "get_kv_class", _bootstrap_class)

    bootstrap_server = disagg_service.start_disagg_service(args)

    assert type(bootstrap_server) is _BootstrapServer
    assert bootstrap_server.host == "0.0.0.0"
    assert bootstrap_server.port == 32150
    expectation = bootstrap_server.terminal_startup_registry.expectation
    assert expectation.group_id == "group-a"
    assert expectation.expected_rank_count == 3
    assert expectation.service("prefill-a").tensor_parallel_size == 2
    assert expectation.service("decode-a").tensor_parallel_size == 1


def test_prefill_rejects_partial_deployment_binding() -> None:
    """Fail before opening a listener when deployment binding is incomplete."""

    args = ServerArgs(
        model_path="dummy",
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="nixl",
    )
    args.pd_terminal_deployment_cohort = _cohort()

    with pytest.raises(ValueError, match="deployment binding is incomplete"):
        disagg_service.start_disagg_service(args)
