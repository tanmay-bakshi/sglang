import sys
import uuid
from dataclasses import dataclass

import pytest

from sglang.srt.disaggregation.decode_reservation_scheduler import (
    DecodeReservationSchedulerControl,
    DecodeReservationUnavailableControl,
)
from sglang.srt.disaggregation.runtime_capabilities import KvTransferProtocol
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclass
class _ParallelState:
    """Minimal scheduler parallel topology.

    :ivar tp_size: Tensor-parallel width.
    :ivar attn_tp_size: Attention tensor-parallel width.
    :ivar pp_size: Pipeline-parallel width.
    """

    tp_size: int
    attn_tp_size: int
    pp_size: int


@dataclass
class _ServerArgs:
    """Minimal control-plane server arguments.

    :ivar dp_size: Data-parallel width.
    :ivar launch_instance_id: Canonical process-generation UUID.
    """

    dp_size: int
    launch_instance_id: str


class _KvManager:
    """Expose one initialized transfer protocol."""

    _packed_decode_runtime_live: bool
    _prepared_grant_protocol: str | None
    _protocol: KvTransferProtocol | None

    def __init__(
        self,
        protocol: KvTransferProtocol | None,
        prepared_grant_protocol: str | None,
        packed_decode_runtime_live: bool,
    ) -> None:
        """Initialize one manager double.

        :param protocol: Initialized transfer protocol.
        :param prepared_grant_protocol: Initialized prefill grant actor protocol.
        :param packed_decode_runtime_live: Whether decode request actors are live.
        """

        self._packed_decode_runtime_live = packed_decode_runtime_live
        self._prepared_grant_protocol = prepared_grant_protocol
        self._protocol = protocol

    def kv_transfer_protocol(self) -> KvTransferProtocol | None:
        """Return the configured runtime protocol.

        :returns: Configured transfer protocol.
        """

        return self._protocol

    def prepared_grant_protocol(self) -> str | None:
        """Return the initialized prefill grant actor protocol.

        :returns: Runtime-owned protocol, otherwise ``None``.
        """

        return self._prepared_grant_protocol

    def supports_packed_decode_request_transactions(self) -> bool:
        """Return whether complete decode request actors are initialized.

        :returns: Runtime-owned decode actor readiness.
        """

        return self._packed_decode_runtime_live


class _DisaggregationQueue:
    """Carry the runtime KV manager required by scheduler capability capture."""

    kv_manager: _KvManager

    def __init__(
        self,
        protocol: KvTransferProtocol | None,
        prepared_grant_protocol: str | None,
        packed_decode_runtime_live: bool,
    ) -> None:
        """Initialize one queue double.

        :param protocol: Initialized transfer protocol.
        :param prepared_grant_protocol: Initialized prefill grant actor protocol.
        :param packed_decode_runtime_live: Whether decode request actors are live.
        """

        self.kv_manager = _KvManager(
            protocol,
            prepared_grant_protocol,
            packed_decode_runtime_live,
        )


@dataclass
class _ModelRunner:
    """Minimal model-runner capability state.

    :ivar kv_cache_dtype_str: Runtime KV storage dtype.
    """

    kv_cache_dtype_str: str = "bfloat16"


@dataclass
class _TpWorker:
    """Minimal tensor-parallel worker state.

    :ivar model_runner: Runtime model-runner state.
    """

    model_runner: _ModelRunner


@dataclass
class _TokenAllocator:
    """Minimal token allocator state.

    :ivar page_size: Runtime KV page size.
    """

    page_size: int = 64


def _scheduler(
    *,
    mode: DisaggregationMode,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    kv_transfer_protocol: KvTransferProtocol | None = None,
    prepared_grant_protocol: str | None = None,
    packed_decode_runtime_live: bool = False,
) -> Scheduler:
    """Build the scheduler state required by control initialization.

    :param mode: PD role.
    :param tp_size: Tensor-parallel width.
    :param pp_size: Pipeline-parallel width.
    :param dp_size: Data-parallel width.
    :param kv_transfer_protocol: Runtime-attested data protocol.
    :param prepared_grant_protocol: Runtime-owned prefill grant protocol.
    :param packed_decode_runtime_live: Runtime-owned decode actor readiness.
    :returns: Minimally initialized scheduler.
    """

    scheduler = object.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.ps = _ParallelState(
        tp_size=tp_size,
        attn_tp_size=tp_size,
        pp_size=pp_size,
    )
    scheduler.server_args = _ServerArgs(
        dp_size=dp_size,
        launch_instance_id=str(uuid.uuid4()),
    )
    queue = _DisaggregationQueue(
        kv_transfer_protocol,
        prepared_grant_protocol,
        packed_decode_runtime_live,
    )
    scheduler.disagg_decode_prealloc_queue = (
        queue if mode is DisaggregationMode.DECODE else None
    )
    scheduler.disagg_prefill_bootstrap_queue = (
        queue if mode is DisaggregationMode.PREFILL else None
    )
    scheduler.tp_worker = _TpWorker(model_runner=_ModelRunner())
    scheduler.token_to_kv_pool_allocator = _TokenAllocator()
    return scheduler


@pytest.mark.parametrize(
    (
        "mode",
        "tp_size",
        "pp_size",
        "dp_size",
        "expected_live_control",
        "expected_protocol",
    ),
    (
        (DisaggregationMode.DECODE, 1, 1, 1, True, None),
        (DisaggregationMode.DECODE, 2, 1, 1, True, None),
        (DisaggregationMode.DECODE, 3, 1, 1, False, None),
        (DisaggregationMode.DECODE, 1, 2, 1, False, None),
        (DisaggregationMode.DECODE, 1, 1, 2, False, None),
        (DisaggregationMode.PREFILL, 2, 1, 1, False, None),
        (DisaggregationMode.PREFILL, 4, 1, 1, False, None),
        (DisaggregationMode.PREFILL, 1, 1, 1, False, None),
        (DisaggregationMode.PREFILL, 8, 1, 1, False, None),
        (DisaggregationMode.PREFILL, 2, 2, 1, False, None),
        (DisaggregationMode.PREFILL, 2, 1, 2, False, None),
    ),
)
def test_control_and_capability_gates_match_supported_topologies(
    mode: DisaggregationMode,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    expected_live_control: bool,
    expected_protocol: str | None,
) -> None:
    """Only a concrete initialized authority advertises prepared grants."""

    scheduler = _scheduler(
        mode=mode,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
    )

    scheduler.init_decode_reservation_control()
    scheduler.init_pd_runtime_capabilities()

    assert (scheduler.decode_reservation_control is not None) is expected_live_control
    expected_handler_type = (
        DecodeReservationSchedulerControl
        if expected_live_control
        else DecodeReservationUnavailableControl
    )
    assert type(scheduler.decode_reservation_request_control) is expected_handler_type
    assert scheduler.pd_runtime_capabilities is not None
    assert (
        scheduler.pd_runtime_capabilities.prepared_grant_protocol == expected_protocol
    )
    assert scheduler.pd_runtime_capabilities.advertises_pd_process is False


@pytest.mark.parametrize(
    ("mode", "tp_size"),
    (
        (DisaggregationMode.DECODE, 1),
        (DisaggregationMode.DECODE, 2),
        (DisaggregationMode.PREFILL, 2),
        (DisaggregationMode.PREFILL, 4),
    ),
)
def test_supported_topology_advertises_only_when_packed_transport_is_live(
    mode: DisaggregationMode,
    tp_size: int,
) -> None:
    """The control vertical cannot advertise an unwired data plane."""

    scheduler = _scheduler(
        mode=mode,
        tp_size=tp_size,
        pp_size=1,
        dp_size=1,
        kv_transfer_protocol="packed-v4",
        prepared_grant_protocol=(
            "control-v1" if mode is DisaggregationMode.PREFILL else None
        ),
        packed_decode_runtime_live=mode is DisaggregationMode.DECODE,
    )

    scheduler.init_decode_reservation_control()
    scheduler.init_pd_runtime_capabilities()

    assert scheduler.pd_runtime_capabilities is not None
    assert scheduler.pd_runtime_capabilities.advertises_pd_process is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
