"""Start bootstrap/kv-store-related server"""

import os

from sglang.srt.disaggregation.base.conn import BaseKVBootstrapServer
from sglang.srt.disaggregation.nixl import NixlKVBootstrapServer
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortExpectation,
    TerminalStartupCohortRegistry,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs


def start_disagg_service(
    server_args: ServerArgs,
) -> BaseKVBootstrapServer | None:
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        startup_expectation = server_args.pd_terminal_startup_expectation
        if startup_expectation is not None:
            if type(startup_expectation) is not TerminalStartupCohortExpectation:
                raise TypeError(
                    "pd_terminal_startup_expectation has an invalid type"
                )
            if transfer_backend != TransferBackend.NIXL:
                raise ValueError(
                    "packed-terminal startup requires the NIXL transfer backend"
                )
            timeout_seconds = server_args.pd_terminal_startup_timeout_seconds
            if timeout_seconds is None:
                raise ValueError("packed-terminal startup timeout is absent")
            registry = TerminalStartupCohortRegistry(
                startup_expectation,
                timeout_seconds=timeout_seconds,
            )
            return NixlKVBootstrapServer(
                host=server_args.host,
                port=server_args.disaggregation_bootstrap_port,
                terminal_startup_registry=registry,
            )
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        is_create_store = (
            server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND
        )
        if is_create_store:
            try:
                from memfabric_hybrid import create_config_store

                ascend_url = os.getenv("ASCEND_MF_STORE_URL")
                create_config_store(ascend_url)
            except Exception as e:
                error_message = f"Failed create mf store, invalid ascend_url."
                error_message += f" With exception {e}"
                raise error_message

        return bootstrap_server
