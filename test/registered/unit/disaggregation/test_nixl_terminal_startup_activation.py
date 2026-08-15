import base64
import hashlib
import threading
import time
import uuid
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import zmq

from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedReady,
    PackedRequestKey,
    PackedTerminalReceipt,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.conn import (
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    NixlKVManager,
    _TerminalDecoderEnrollment,
)
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_CONTROL_TAG,
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedRegistrationAdvertisement,
    decode_packed_control_frames,
    encode_packed_control_frames,
)
from sglang.srt.disaggregation.nixl.startup_decode_routes import (
    TerminalDecodeControlRouteTable,
    build_terminal_decode_control_route_table,
    encode_terminal_decode_control_route_table,
)
from sglang.srt.disaggregation.nixl.startup_enrollment_ack import (
    TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
    build_terminal_startup_enrollment_ack,
    encode_terminal_startup_enrollment_ack,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeDeliveryTarget,
    PackedTerminalDecodeWireDelivery,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.serving_reactor import (
    PackedTerminalProcessReactorFailure,
    PackedTerminalProcessReactorFailureCause,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptIssuer,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_DIGEST = bytes.fromhex("11" * 32)


class _FakeRemoteHandle:
    """Stable native handle selected by one metadata blob."""

    name: str

    def __init__(self, name: str) -> None:
        """Construct one remote handle.

        :param name: Metadata-selected NIXL agent name.
        """

        self.name = name


class _FakeAgent:
    """Minimal native peer lifecycle used during startup activation."""

    name: str
    _metadata_names: dict[bytes, str]
    _handles: dict[bytes, _FakeRemoteHandle]
    add_calls: list[bytes]
    connection_calls: list[_FakeRemoteHandle]

    def __init__(self, name: str, metadata_names: dict[bytes, str]) -> None:
        """Construct an exact metadata-to-handle authority.

        :param name: Local NIXL agent name.
        :param metadata_names: Remote metadata and exact resolved names.
        """

        self.name = name
        self._metadata_names = metadata_names
        self._handles = {}
        self.add_calls = []
        self.connection_calls = []

    def add_remote_agent(self, metadata: bytes) -> _FakeRemoteHandle:
        """Resolve or create one stable native handle.

        :param metadata: Complete remote agent metadata.
        :returns: Metadata-owned native handle.
        """

        self.add_calls.append(metadata)
        handle = self._handles.get(metadata)
        if handle is not None:
            return handle
        handle = _FakeRemoteHandle(self._metadata_names[metadata])
        self._handles[metadata] = handle
        return handle

    def make_connection(self, handle: _FakeRemoteHandle) -> None:
        """Record one proactive native connection.

        :param handle: Exact retained remote handle.
        """

        self.connection_calls.append(handle)

    def remove_remote_agent(self, handle: _FakeRemoteHandle) -> None:
        """Remove one half-loaded remote handle.

        :param handle: Exact remote handle.
        """

        for metadata, candidate in tuple(self._handles.items()):
            if candidate is handle:
                del self._handles[metadata]
                return
        raise RuntimeError("remote handle is absent")


class _InprocInbox:
    """One isolated in-process ZeroMQ startup inbox."""

    context: zmq.Context
    pull: zmq.Socket
    push: zmq.Socket

    def __init__(self) -> None:
        """Bind one PULL/PUSH pair under a unique in-process address."""

        self.context = zmq.Context()
        self.pull = self.context.socket(zmq.PULL)
        self.push = self.context.socket(zmq.PUSH)
        endpoint = f"inproc://terminal-startup-{uuid.uuid4()}"
        self.pull.bind(endpoint)
        self.push.connect(endpoint)

    def send(self, frames: tuple[bytes, ...]) -> None:
        """Queue one complete startup multipart message.

        :param frames: Exact ordered message frames.
        """

        self.push.send_multipart(frames)

    def close(self) -> None:
        """Close both sockets and their isolated context."""

        self.push.close(linger=0)
        self.pull.close(linger=0)
        self.context.term()


class _BootstrapRoute:
    """Immutable-enough route facade consumed by decoder activation."""

    _info: dict[str, object]

    def __init__(self, info: dict[str, object]) -> None:
        """Retain one already authenticated source route.

        :param info: Complete source bootstrap route.
        """

        self._info = dict(info)

    def bootstrap_info(self) -> dict[str, object]:
        """Return an isolated route image for one registration send.

        :returns: Complete source bootstrap route.
        """

        return dict(self._info)


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    tp_rank: int,
    tp_size: int,
    generation: int,
    agent_name: str,
    metadata: bytes,
    launch_instance: int,
    port: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact startup matrix row.

    :param service_id: Static service identifier.
    :param role: Source or decode role.
    :param tp_rank: Rank within the service.
    :param tp_size: Exact service TP width.
    :param generation: Non-nil UUID integer.
    :param agent_name: Exact NIXL agent name.
    :param metadata: Complete NIXL metadata represented by the row digest.
    :param launch_instance: Static launch UUID integer.
    :param port: Model service origin port.
    :returns: Validated startup advertisement.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{port}",
        role=role,
        launch_instance_id=uuid.UUID(int=launch_instance).bytes,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build a TP2 source and two TP1 decoder services.

    :returns: Complete canonical cross-role matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        ranks=(
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tp_rank=0,
                tp_size=2,
                generation=101,
                agent_name="prefill-agent-0",
                metadata=b"prefill-metadata-0",
                launch_instance=1,
                port=32001,
            ),
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tp_rank=1,
                tp_size=2,
                generation=102,
                agent_name="prefill-agent-1",
                metadata=b"prefill-metadata-1",
                launch_instance=1,
                port=32001,
            ),
            _rank(
                service_id="decode-a",
                role=TerminalOwnerRole.DECODE,
                tp_rank=0,
                tp_size=1,
                generation=201,
                agent_name="decode-agent-a",
                metadata=b"decode-metadata-a",
                launch_instance=2,
                port=32002,
            ),
            _rank(
                service_id="decode-b",
                role=TerminalOwnerRole.DECODE,
                tp_rank=0,
                tp_size=1,
                generation=202,
                agent_name="decode-agent-b",
                metadata=b"decode-metadata-b",
                launch_instance=3,
                port=32003,
            ),
        ),
    )


def _decode_tp2_matrix() -> TerminalStartupCohortMatrix:
    """Build a source service and one TP2 decode service.

    :returns: Complete same-decode-service receipt-routing matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        ranks=(
            _rank(
                service_id="prefill-c",
                role=TerminalOwnerRole.SOURCE,
                tp_rank=0,
                tp_size=1,
                generation=301,
                agent_name="prefill-agent-c",
                metadata=b"prefill-metadata-c",
                launch_instance=4,
                port=32004,
            ),
            _rank(
                service_id="decode-c",
                role=TerminalOwnerRole.DECODE,
                tp_rank=0,
                tp_size=2,
                generation=401,
                agent_name="decode-agent-c-0",
                metadata=b"decode-metadata-c-0",
                launch_instance=5,
                port=32005,
            ),
            _rank(
                service_id="decode-c",
                role=TerminalOwnerRole.DECODE,
                tp_rank=1,
                tp_size=2,
                generation=402,
                agent_name="decode-agent-c-1",
                metadata=b"decode-metadata-c-1",
                launch_instance=5,
                port=32005,
            ),
        ),
    )


def _binding_for_matrix(
    matrix: TerminalStartupCohortMatrix,
    service_id: str,
    tp_rank: int,
) -> TerminalStartupRankBinding:
    """Bind one rank from an explicit startup matrix.

    :param matrix: Complete immutable deployment matrix.
    :param service_id: Exact local service.
    :param tp_rank: Exact local tensor-parallel rank.
    :returns: Rank binding and least-authority producer plan.
    """

    return TerminalStartupRankBinding(
        advertisement=matrix.rank(service_id, tp_rank),
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id=service_id,
            local_tensor_parallel_rank=tp_rank,
            first_producer_id=0,
        ),
    )


def _request_binding(owner: TerminalProcessIdentity) -> TerminalRequestBinding:
    """Build one stable request binding for a decode routing test.

    :param owner: Exact destination-rank lifecycle owner.
    :returns: Complete immutable request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=47,
            request_generation=bytes((19,)) * 16,
        ),
        owner=owner,
        rank_manifest_digest=bytes((23,)) * 32,
        allocation_digest=bytes((29,)) * 32,
    )


def _binding(service_id: str, tp_rank: int) -> TerminalStartupRankBinding:
    """Build one local binding over the shared fixture matrix.

    :param service_id: Local service identifier.
    :param tp_rank: Local service TP rank.
    :returns: Exact rank binding and least-authority producer plan.
    """

    matrix = _matrix()
    advertisement = matrix.rank(service_id, tp_rank)
    return TerminalStartupRankBinding(
        advertisement=advertisement,
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id=service_id,
            local_tensor_parallel_rank=tp_rank,
            first_producer_id=0,
        ),
    )


def _manager(
    binding: TerminalStartupRankBinding,
    metadata_names: dict[bytes, str],
) -> NixlKVManager:
    """Build one manager shell with sealed startup authority.

    :param binding: Exact local startup rank binding.
    :param metadata_names: Remote metadata-to-agent mapping.
    :returns: Manager with activation and peer lifecycle state.
    """

    manager = object.__new__(NixlKVManager)
    local = binding.advertisement
    manager.disaggregation_mode = (
        DisaggregationMode.PREFILL
        if local.role is TerminalOwnerRole.SOURCE
        else DisaggregationMode.DECODE
    )
    manager.process_generation = str(uuid.UUID(bytes=local.process_generation))
    manager.agent = _FakeAgent(local.nixl_agent_name, metadata_names)
    local_metadata = {
        "prefill-agent-0": b"prefill-metadata-0",
        "prefill-agent-1": b"prefill-metadata-1",
        "decode-agent-a": b"decode-metadata-a",
        "decode-agent-b": b"decode-metadata-b",
        "prefill-agent-c": b"prefill-metadata-c",
        "decode-agent-c-0": b"decode-metadata-c-0",
        "decode-agent-c-1": b"decode-metadata-c-1",
    }[local.nixl_agent_name]
    manager.agent_metadata = local_metadata
    manager.local_ip = "127.0.0.1"
    manager.rank_port = (
        _decode_listener_port(local)
        if local.role is TerminalOwnerRole.DECODE
        else 34000 + local.tensor_parallel_rank
    )
    manager.attn_tp_size = local.tensor_parallel_size
    manager.attn_tp_rank = local.tensor_parallel_rank
    manager.attn_cp_size = 1
    manager.pp_size = 1
    manager.server_args = SimpleNamespace(pd_terminal_startup_timeout_seconds=1.0)
    manager._terminal_startup_binding = None
    manager._terminal_startup_peer_enrollment = None
    manager._terminal_source_publication_control = None
    manager._terminal_decode_control_routes = None
    manager._terminal_runtime_activated = threading.Event()
    manager._terminal_activation_lock = threading.Lock()
    manager._terminal_activation_started = False
    manager._terminal_bootstrap_thread = None
    manager._runtime_workers_started = False
    manager._terminal_runtime_installation = None
    manager._terminal_runtime_enrollment = None
    manager._terminal_source_serving = None
    manager._terminal_decode_serving = None
    manager._terminal_process_reactor = None
    manager._terminal_output_publisher = None
    manager._terminal_source_receipt_importers = {}
    manager._terminal_control_thread = None
    manager._terminal_control_read_fd = None
    manager._terminal_control_write_fd = None
    manager._terminal_control_stop_requested = False
    manager._terminal_control_ready = threading.Event()
    manager._terminal_control_lock = threading.Lock()
    manager._packed_control_send_lock = threading.Lock()
    manager._terminal_runtime_close_started = False
    manager._terminal_runtime_closed = False
    manager._terminal_runtime_close_lock = threading.Lock()
    manager._terminal_process_fatal_reason = None
    manager._terminal_process_fatal_traceback = None
    manager._terminal_process_fatal_lock = threading.Lock()
    manager._prefill_peers = {}
    manager._prefill_peer_keys_by_addr = defaultdict(set)
    manager._prefill_peers_by_agent_name = {}
    manager._prefill_peers_by_handle = {}
    manager._prefill_peer_lock = threading.RLock()
    manager._quarantined_remote_handles = set()
    manager.decode_kv_args_table = {}
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager._packed_prefill_runtime = None
    if local.role is TerminalOwnerRole.SOURCE:
        writer_manifest = DecodeWriterManifest.for_tensor_parallel(
            local.tensor_parallel_size,
        )
        manager._packed_prefill_runtime = SimpleNamespace(
            writer_id=writer_manifest.writers[local.tensor_parallel_rank],
        )
    manager._prepare_payload_xfer = MagicMock()
    manager.install_terminal_startup_binding(binding)
    return manager


def _source_route(rank: TerminalStartupRankAdvertisement) -> dict[str, object]:
    """Build one full-metadata bootstrap route for a source rank.

    :param rank: Exact source matrix row.
    :returns: Complete route accepted by NIXL peer loading.
    """

    metadata = f"prefill-metadata-{rank.tensor_parallel_rank}".encode("ascii")
    return {
        "transport_protocol": NIXL_BOOTSTRAP_PEER_PROTOCOL,
        "nixl_agent_name": rank.nixl_agent_name,
        "nixl_agent_metadata": base64.b64encode(metadata).decode("ascii"),
        "nixl_agent_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "process_generation": str(uuid.UUID(bytes=rank.process_generation)),
        "attn_dp_rank": 0,
        "attn_cp_rank": 0,
        "attn_tp_rank": rank.tensor_parallel_rank,
        "pp_rank": 0,
        "transfer_source_rank": rank.tensor_parallel_rank,
        "rank_ip": "127.0.0.1",
        "rank_port": 31000 + rank.tensor_parallel_rank,
    }


def _source_routes(
    binding: TerminalStartupRankBinding,
) -> tuple[dict[str, object], ...]:
    """Build the canonical complete source route population.

    :param binding: Decoder binding whose matrix owns the source ranks.
    :returns: Canonically matrix-ordered source routes.
    """

    return tuple(
        _source_route(rank)
        for rank in binding.matrix.ranks
        if rank.role is TerminalOwnerRole.SOURCE
    )


def _decoder_registration_frames(
    binding: TerminalStartupRankBinding,
) -> tuple[bytes, ...]:
    """Serialize one complete packed decoder registration.

    :param binding: Exact decoder rank binding.
    :returns: Guarded process-lifetime registration frames.
    """

    rank = binding.advertisement
    manager = object.__new__(NixlKVManager)
    manager._terminal_startup_binding = binding
    manager.local_ip = "127.0.0.1"
    manager.rank_port = _decode_listener_port(rank)
    manager.agent = SimpleNamespace(name=rank.nixl_agent_name)
    manager.agent_metadata = (
        b"decode-metadata-a" if rank.service_id == "decode-a" else b"decode-metadata-b"
    )
    manager.process_generation = str(uuid.UUID(bytes=rank.process_generation))
    manager.attn_tp_size = rank.tensor_parallel_size
    manager.enable_staging = False
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000],
        kv_data_lens=[4096],
        kv_data_mem_kinds=["VRAM"],
        kv_item_lens=[256],
        kv_layer_ids=[3],
        aux_data_ptrs=[0x2000],
        state_data_ptrs=[],
        state_item_lens=[],
        state_dim_per_tensor=[],
        state_layer_ids=[],
        gpu_id=0,
        engine_rank=0,
    )
    manager._packed_decode_controller = SimpleNamespace(
        ready=True,
        prepared_grant_protocol=PACKED_PREPARED_GRANT_PROTOCOL,
        advertisement=PackedRegistrationAdvertisement(
            base_address=0x4000,
            total_size=8192,
            arena_generation=uuid.uuid5(uuid.NAMESPACE_DNS, rank.service_id).bytes,
            visibility_policy_digest=bytes.fromhex("22" * 32),
            runtime_cohort_digest=bytes.fromhex("33" * 32),
            page_size=64,
        ),
    )
    return manager._decode_registration_frames()


def _decode_listener_port(rank: TerminalStartupRankAdvertisement) -> int:
    """Return one stable synthetic decoder manager listener port.

    :param rank: Exact decoder startup row.
    :returns: Unique test listener port.
    """

    service_offset = {
        "decode-a": 0,
        "decode-b": 100,
        "decode-c": 200,
    }[rank.service_id]
    return 33000 + service_offset + rank.tensor_parallel_rank


def _decode_route_table(
    binding: TerminalStartupRankBinding,
    service_id: str,
    registration_frames: tuple[bytes, ...],
) -> TerminalDecodeControlRouteTable:
    """Build one complete synthetic same-service listener table.

    :param binding: Startup matrix authority.
    :param service_id: Decoder service represented by the table.
    :param registration_frames: Stable synthetic registration bytes.
    :returns: Immutable decoder control route table.
    """

    target_ranks = tuple(
        rank
        for rank in binding.matrix.ranks
        if rank.role is TerminalOwnerRole.DECODE and rank.service_id == service_id
    )
    return build_terminal_decode_control_route_table(
        binding.matrix,
        service_id,
        tuple(
            (
                rank,
                NetworkAddress("127.0.0.1", _decode_listener_port(rank)),
                registration_frames,
            )
            for rank in target_ranks
        ),
    )


def _ack_frames(
    binding: TerminalStartupRankBinding,
    source_rank: TerminalStartupRankAdvertisement,
    registration_frames: tuple[bytes, ...],
    *,
    target_service_id: str | None = None,
) -> tuple[bytes, ...]:
    """Build one source acknowledgement multipart message.

    :param binding: Local decoder binding.
    :param source_rank: Exact source rank issuing the acknowledgement.
    :param registration_frames: Registration retained by the source.
    :param target_service_id: Optional alternate matrix decoder target.
    :returns: Tagged canonical acknowledgement frames.
    """

    target = binding.matrix.rank(
        binding.advertisement.service_id
        if target_service_id is None
        else target_service_id,
        binding.advertisement.tensor_parallel_rank,
    )
    route_table = _decode_route_table(
        binding,
        target.service_id,
        registration_frames,
    )
    acknowledgement = build_terminal_startup_enrollment_ack(
        binding.matrix,
        source_rank,
        target,
        registration_frames,
        route_table.digest,
    )
    return (
        TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
        encode_terminal_startup_enrollment_ack(acknowledgement),
        encode_terminal_decode_control_route_table(route_table),
    )


def test_activation_is_one_shot_and_precommit_failure_starts_no_workers() -> None:
    """A failed first activation cannot compose either runtime implementation."""

    manager = _manager(_binding("prefill-a", 0), {})
    manager._activate_terminal_source = MagicMock(
        side_effect=RuntimeError("precommit enrollment failed")
    )
    manager._compose_terminal_runtime = MagicMock()
    manager._start_prefill_runtime_workers = MagicMock()

    with pytest.raises(RuntimeError, match="precommit enrollment failed"):
        manager.activate_terminal_startup()

    assert manager._terminal_activation_started
    assert not manager._terminal_runtime_activated.is_set()
    assert not manager._runtime_workers_started
    manager._compose_terminal_runtime.assert_not_called()
    manager._start_prefill_runtime_workers.assert_not_called()

    manager._activate_terminal_source = MagicMock()
    with pytest.raises(RuntimeError, match="cannot repeat"):
        manager.activate_terminal_startup()
    manager._activate_terminal_source.assert_not_called()
    manager._compose_terminal_runtime.assert_not_called()
    manager._start_prefill_runtime_workers.assert_not_called()


def test_runtime_composition_failure_is_sticky_and_starts_no_legacy_workers() -> None:
    """A post-enrollment composition failure cannot retry or fall back."""

    manager = _manager(_binding("prefill-a", 0), {})
    manager._activate_terminal_source = MagicMock()
    manager._terminal_startup_peer_enrollment = SimpleNamespace(frozen=True)
    manager._terminal_source_publication_control = object()
    manager._compose_terminal_runtime = MagicMock(
        side_effect=RuntimeError("terminal composition failed")
    )
    manager._start_prefill_runtime_workers = MagicMock()

    with pytest.raises(RuntimeError, match="terminal composition failed"):
        manager.activate_terminal_startup()

    assert manager._terminal_activation_started
    assert not manager._terminal_runtime_activated.is_set()
    assert manager._terminal_process_fatal_reason == (
        "terminal runtime composition failed"
    )
    manager._start_prefill_runtime_workers.assert_not_called()

    with pytest.raises(RuntimeError, match="cannot repeat"):
        manager.activate_terminal_startup()
    assert manager._compose_terminal_runtime.call_count == 1


def test_source_runtime_composition_binds_before_starting_exactly_one_stack() -> None:
    """One enrollment, serving owner, reactor, and receiver start in order."""

    binding = _binding("prefill-a", 0)
    manager = _manager(binding, {})
    manager._terminal_startup_peer_enrollment = SimpleNamespace(binding=binding)
    order: list[str] = []
    enrollment = SimpleNamespace()
    serving = MagicMock()
    publisher = MagicMock()
    reactor = MagicMock()
    serving.start.side_effect = lambda: order.append("serving-start")
    publisher.start.side_effect = lambda: order.append("publisher-start")
    reactor.start.side_effect = lambda timeout: order.append("reactor-start")
    manager._start_terminal_control_receiver = MagicMock(
        side_effect=lambda: order.append("receiver-start")
    )
    manager._compose_terminal_source = MagicMock(return_value=(serving, publisher))
    manager._terminal_runtime_installation = SimpleNamespace(
        terminal_request_capacity=7,
        bind_source_serving=lambda value: order.append("scheduler-bind"),
    )
    manager._start_prefill_runtime_workers = MagicMock()
    manager._start_decode_runtime_workers = MagicMock()

    with (
        patch(
            "sglang.srt.disaggregation.nixl.conn.TerminalRankRuntimeEnrollmentFactory"
        ) as enrollment_factory,
        patch(
            "sglang.srt.disaggregation.nixl.conn."
            "PackedTerminalProcessReactor.for_source",
            return_value=reactor,
        ) as reactor_factory,
    ):
        enrollment_factory.return_value.create.return_value = enrollment
        manager._compose_terminal_runtime()

    assert order == [
        "scheduler-bind",
        "publisher-start",
        "serving-start",
        "reactor-start",
        "receiver-start",
    ]
    enrollment_factory.assert_called_once_with(
        binding,
        manager._terminal_rank_runtime_config(7),
    )
    enrollment_factory.return_value.create.assert_called_once_with()
    manager._compose_terminal_source.assert_called_once_with(
        binding,
        enrollment,
        manager._terminal_runtime_installation,
    )
    reactor_factory.assert_called_once_with(
        serving,
        manager._terminal_reactor_failed,
    )
    assert manager._terminal_runtime_enrollment is enrollment
    assert manager._terminal_source_serving is serving
    assert manager._terminal_decode_serving is None
    assert manager._terminal_process_reactor is reactor
    assert manager._terminal_output_publisher is publisher
    manager._start_prefill_runtime_workers.assert_not_called()
    manager._start_decode_runtime_workers.assert_not_called()


def test_publisher_death_is_sticky_and_wakes_scheduler_once() -> None:
    """Publisher thread death becomes one process-lifetime fatal wake."""

    manager = _manager(_binding("prefill-a", 0), {})
    wake_scheduler = MagicMock()
    manager._terminal_runtime_installation = SimpleNamespace(
        owner_dead_handler=wake_scheduler
    )
    reactor = MagicMock()
    reactor.inventory.return_value = SimpleNamespace(
        started=True,
        admission_open=True,
    )
    manager._terminal_process_reactor = reactor

    manager._terminal_publisher_failed("publisher died", "publisher traceback")
    manager._terminal_publisher_failed("later publisher error", None)

    assert manager._terminal_process_fatal_reason == "publisher died"
    assert manager._terminal_process_fatal_traceback == "publisher traceback"
    reactor.stop_admission.assert_called_once_with()
    wake_scheduler.assert_called_once_with()


def test_reactor_death_is_sticky_and_wakes_scheduler_once() -> None:
    """Reactor thread death becomes one process-lifetime fatal wake."""

    manager = _manager(_binding("decode-a", 0), {})
    wake_scheduler = MagicMock()
    manager._terminal_runtime_installation = SimpleNamespace(
        owner_dead_handler=wake_scheduler
    )
    reactor = MagicMock()
    reactor.inventory.return_value = SimpleNamespace(
        started=True,
        admission_open=False,
    )
    manager._terminal_process_reactor = reactor
    failure = PackedTerminalProcessReactorFailure(
        cause=PackedTerminalProcessReactorFailureCause.SELECTOR_FAILURE,
        reason="selector died",
        formatted_traceback="reactor traceback",
        occurred_ns=17,
    )

    manager._terminal_reactor_failed(failure)

    assert manager._terminal_process_fatal_reason == str(failure)
    assert manager._terminal_process_fatal_traceback == "reactor traceback"
    reactor.stop_admission.assert_not_called()
    wake_scheduler.assert_called_once_with()


def test_terminal_runtime_closes_in_dependency_reverse_order() -> None:
    """Teardown keeps downstream consumers alive until their producers drain."""

    manager = _manager(_binding("prefill-a", 0), {})
    order: list[str] = []
    reactor = MagicMock()
    serving = MagicMock()
    publisher = MagicMock()
    publication_control = MagicMock()
    reactor.stop_admission.side_effect = lambda: order.append("reactor-admission")
    reactor.close.side_effect = lambda timeout: order.append("reactor-close")
    serving.stop_admission_and_retire_producers.side_effect = lambda: order.append(
        "producer-retirement"
    )
    serving.close_clean.side_effect = lambda timeout: order.append("serving-close")
    publisher.stop_admission_and_join.side_effect = lambda: (
        order.append("publisher-close") or True
    )
    publication_control.close_clean.side_effect = lambda: order.append(
        "publication-control-close"
    )
    manager.stop_terminal_control_receiver = lambda timeout: order.append(
        "receiver-close"
    )
    manager._terminal_process_reactor = reactor
    manager._terminal_source_serving = serving
    manager._terminal_output_publisher = publisher
    manager._terminal_source_publication_control = publication_control

    manager.close_terminal_runtime(process_fatal=False)
    manager.close_terminal_runtime(process_fatal=False)

    assert order == [
        "reactor-admission",
        "receiver-close",
        "producer-retirement",
        "reactor-close",
        "serving-close",
        "publisher-close",
        "publication-control-close",
    ]
    serving.abort_and_close.assert_not_called()


def test_terminal_control_receiver_stops_without_component_failure() -> None:
    """The dedicated wake joins a blocked receiver as an intentional stop."""

    binding = _binding("prefill-a", 0)
    manager = _manager(binding, {})
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull
    manager._require_terminal_startup_peer_enrollment = MagicMock(
        return_value=SimpleNamespace(binding=binding)
    )
    manager._record_terminal_component_failure = MagicMock()

    try:
        manager._start_terminal_control_receiver()
        manager.stop_terminal_control_receiver(1.0)
    finally:
        inbox.close()

    thread = manager._terminal_control_thread
    assert thread is not None
    assert not thread.is_alive()
    assert manager._terminal_control_read_fd is None
    assert manager._terminal_control_write_fd is None
    manager._record_terminal_component_failure.assert_not_called()


def test_terminal_control_receiver_rejects_unrelated_traffic_and_continues() -> None:
    """Foreign pre-request framing cannot kill the terminal socket owner."""

    binding = _binding("prefill-a", 0)
    manager = _manager(binding, {})
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull
    manager._require_terminal_startup_peer_enrollment = MagicMock(
        return_value=SimpleNamespace(binding=binding)
    )
    manager._record_terminal_component_failure = MagicMock()
    dispatched = threading.Event()
    manager._dispatch_terminal_source_control = MagicMock(
        side_effect=lambda frames: dispatched.set()
    )

    try:
        manager._start_terminal_control_receiver()
        inbox.send((b"legacy-runtime",))
        inbox.send((PACKED_CONTROL_TAG, b"opaque-terminal-payload"))
        assert dispatched.wait(1.0)
        manager.stop_terminal_control_receiver(1.0)
    finally:
        inbox.close()

    manager._dispatch_terminal_source_control.assert_called_once()
    manager._record_terminal_component_failure.assert_not_called()


def test_terminal_control_dispatch_failure_kills_receiver_fail_closed() -> None:
    """A structurally terminal lifecycle failure is process-fatal."""

    binding = _binding("prefill-a", 0)
    manager = _manager(binding, {})
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull
    manager._require_terminal_startup_peer_enrollment = MagicMock(
        return_value=SimpleNamespace(binding=binding)
    )
    failed = threading.Event()
    failures: list[tuple[str, str | None]] = []

    def record_failure(reason: str, formatted_traceback: str | None) -> None:
        failures.append((reason, formatted_traceback))
        failed.set()

    manager._record_terminal_component_failure = record_failure
    manager._dispatch_terminal_source_control = MagicMock(
        side_effect=RuntimeError("authenticated lifecycle was invalid")
    )

    try:
        manager._start_terminal_control_receiver()
        inbox.send((PACKED_CONTROL_TAG, b"opaque-terminal-payload"))
        assert failed.wait(1.0)
        manager.stop_terminal_control_receiver(1.0)
    finally:
        inbox.close()

    thread = manager._terminal_control_thread
    assert thread is not None
    assert not thread.is_alive()
    assert len(failures) == 1
    assert failures[0][0] == "terminal control receiver died"
    assert failures[0][1] is not None
    assert "authenticated lifecycle was invalid" in failures[0][1]


def test_source_packed_ready_authenticates_transport_before_owner_join() -> None:
    """Production control dispatch joins READY only after actor authentication."""

    local_binding = _binding("prefill-a", 0)
    local = local_binding.advertisement.terminal_identity
    decoder_rank = local_binding.matrix.rank("decode-a", 0)
    manager = _manager(local_binding, {})
    source_binding = _request_binding(local)
    order: list[str] = []
    actor = MagicMock()
    actor.deliver_terminal_owner_ready.side_effect = (
        lambda peer, message: order.append("actor") or source_binding
    )
    serving = MagicMock()
    serving.packed_ready.side_effect = lambda digest: order.append("owner_join")
    manager._packed_prefill_runtime = actor
    manager._terminal_source_serving = serving
    manager._authenticated_terminal_control_rank = MagicMock(
        return_value=decoder_rank
    )
    manager._require_terminal_startup_peer_enrollment = MagicMock(
        return_value=SimpleNamespace(
            decoder_peers={
                decoder_rank.key: SimpleNamespace(
                    agent_name=decoder_rank.nixl_agent_name,
                    process_generation=str(
                        uuid.UUID(bytes=decoder_rank.process_generation)
                    ),
                    remote_handle=object(),
                )
            }
        )
    )
    ready = PackedReady(
        key=PackedChunkKey(
            room_id=47,
            chunk_id=0,
            request_generation=source_binding.request_key.request_generation,
        ),
        writer_id=StagingWriterId(
            transfer_source_rank=0,
            source_attn_tp_rank=0,
            source_pp_rank=0,
            source_cp_rank=0,
        ),
        digest=bytes((31,)) * 32,
        visibility_policy_digest=bytes((37,)) * 32,
        lease_id=1,
        lease_base_address=0x1000,
        projection_offset=0,
        projection_length=4096,
    )

    with patch(
        "sglang.srt.disaggregation.nixl.conn.decode_packed_control_frames",
        return_value=(
            decoder_rank.nixl_agent_name,
            str(uuid.UUID(bytes=decoder_rank.process_generation)),
            ready,
        ),
    ):
        manager._dispatch_terminal_source_control([b"authenticated-control"])

    assert order == ["actor", "owner_join"]
    serving.packed_ready.assert_called_once_with(source_binding.digest)


def test_decode_rank_zero_accepts_authenticated_local_ready_from_peer_rank() -> None:
    """Coordinator fan-in preserves the remote rank's binding identity."""

    matrix = _decode_tp2_matrix()
    local_binding = _binding_for_matrix(matrix, "decode-c", 0)
    remote = matrix.rank("decode-c", 1).terminal_identity
    manager = _manager(_binding("decode-a", 0), {})
    manager._terminal_startup_binding = local_binding
    manager._terminal_startup_peer_enrollment = SimpleNamespace(binding=local_binding)
    manager._terminal_decode_serving = MagicMock()
    manager._terminal_process_reactor = MagicMock()
    receipt = (
        TerminalWireReceiptIssuer(remote)
        .issue(
            binding=_request_binding(remote),
            kind=TerminalReceiptKind.LOCAL_DECODE_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=31,
        )
        .wire_receipt
    )

    manager.receive_terminal_decode_receipt(receipt, remote)

    manager._terminal_decode_serving.coordinator_receipt_received.assert_called_once_with(
        receipt,
        remote,
    )
    manager._terminal_process_reactor.notify_coordinator_deadline_changed.assert_called_once_with()
    manager._terminal_decode_serving.request_terminal_received.assert_not_called()


def test_decode_peer_accepts_request_ready_only_from_canonical_rank() -> None:
    """Coordinator fan-out targets the local owner under rank-zero authority."""

    matrix = _decode_tp2_matrix()
    local_binding = _binding_for_matrix(matrix, "decode-c", 1)
    coordinator = matrix.rank("decode-c", 0).terminal_identity
    local = local_binding.advertisement.terminal_identity
    manager = _manager(_binding("decode-a", 0), {})
    manager._terminal_startup_binding = local_binding
    manager._terminal_startup_peer_enrollment = SimpleNamespace(binding=local_binding)
    manager._terminal_decode_serving = MagicMock()
    receipt = (
        TerminalWireReceiptIssuer(coordinator)
        .issue(
            binding=_request_binding(local),
            kind=TerminalReceiptKind.REQUEST_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=37,
        )
        .wire_receipt
    )

    manager.receive_terminal_decode_receipt(receipt, coordinator)

    manager._terminal_decode_serving.request_terminal_received.assert_called_once_with(
        receipt,
        coordinator,
    )
    manager._terminal_decode_serving.coordinator_receipt_received.assert_not_called()


def test_decode_delivery_uses_frozen_rank_listener_without_collective_relay() -> None:
    """LOCAL_DECODE_READY reaches rank zero through the exact startup route."""

    matrix = _decode_tp2_matrix()
    binding = _binding_for_matrix(matrix, "decode-c", 1)
    local = binding.advertisement.terminal_identity
    coordinator = matrix.rank("decode-c", 0).terminal_identity
    manager = _manager(binding, {})
    manager._terminal_decode_control_routes = _decode_route_table(
        binding,
        "decode-c",
        (b"guarded-registration",),
    )
    socket = MagicMock()
    manager._connect = MagicMock(return_value=socket)
    receipt = (
        TerminalWireReceiptIssuer(local)
        .issue(
            binding=_request_binding(local),
            kind=TerminalReceiptKind.LOCAL_DECODE_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=47,
        )
        .wire_receipt
    )

    manager._send_terminal_decode_delivery(
        PackedTerminalDecodeWireDelivery(
            target=PackedTerminalDecodeDeliveryTarget.COORDINATOR,
            recipient=coordinator,
            receipt=receipt,
        )
    )

    manager._connect.assert_called_once_with("tcp://127.0.0.1:33200", is_ipv6=False)
    frames = socket.send_multipart.call_args.args[0]
    agent_name, generation, message = decode_packed_control_frames(list(frames))
    assert agent_name == binding.advertisement.nixl_agent_name
    assert generation == str(uuid.UUID(bytes=binding.advertisement.process_generation))
    assert type(message) is PackedTerminalReceipt
    assert message.key == receipt.binding.request_key
    assert message.receipt_payload == receipt.encode()


def test_rank_zero_fans_request_ready_to_decode_and_source_owners_directly() -> None:
    """Request-global terminality uses exact destination and source listeners."""

    matrix = _decode_tp2_matrix()
    binding = _binding_for_matrix(matrix, "decode-c", 0)
    coordinator = binding.advertisement.terminal_identity
    decode_peer = matrix.rank("decode-c", 1).terminal_identity
    source_rank = matrix.rank("prefill-c", 0)
    manager = _manager(binding, {})
    manager._terminal_decode_control_routes = _decode_route_table(
        binding,
        "decode-c",
        (b"guarded-registration",),
    )
    source_handle = object()
    enrollment = manager._terminal_startup_peer_enrollment
    assert enrollment is not None
    enrollment.prefill_peers[source_rank.key] = SimpleNamespace(
        handle=source_handle,
        control_endpoint=NetworkAddress("127.0.0.1", 31000),
    )
    sockets: dict[str, MagicMock] = {}

    def connect(endpoint: str, *, is_ipv6: bool) -> MagicMock:
        """Return one endpoint-specific fake PUSH socket.

        :param endpoint: Exact target URL.
        :param is_ipv6: Address-family discriminator.
        :returns: Stable synthetic socket.
        """

        assert not is_ipv6
        return sockets.setdefault(endpoint, MagicMock())

    manager._connect = connect
    for owner in (decode_peer, source_rank.terminal_identity):
        receipt = (
            TerminalWireReceiptIssuer(coordinator)
            .issue(
                binding=_request_binding(owner),
                kind=TerminalReceiptKind.REQUEST_READY,
                outcome=TerminalReceiptOutcome.SUCCESS,
                terminal_timestamp_ns=53,
            )
            .wire_receipt
        )
        manager._send_terminal_decode_delivery(
            PackedTerminalDecodeWireDelivery(
                target=PackedTerminalDecodeDeliveryTarget.OWNER,
                recipient=owner,
                receipt=receipt,
            )
        )

    assert set(sockets) == {
        "tcp://127.0.0.1:31000",
        "tcp://127.0.0.1:33201",
    }
    assert all(socket.send_multipart.call_count == 1 for socket in sockets.values())


def test_decode_control_ingress_authenticates_same_service_rank_receipt() -> None:
    """The blocking manager owner joins wire claims with its frozen route table."""

    matrix = _decode_tp2_matrix()
    binding = _binding_for_matrix(matrix, "decode-c", 0)
    remote_rank = matrix.rank("decode-c", 1)
    remote = remote_rank.terminal_identity
    manager = _manager(binding, {})
    manager._terminal_decode_control_routes = _decode_route_table(
        binding,
        "decode-c",
        (b"guarded-registration",),
    )
    manager._terminal_decode_serving = MagicMock()
    manager._terminal_process_reactor = MagicMock()
    receipt = (
        TerminalWireReceiptIssuer(remote)
        .issue(
            binding=_request_binding(remote),
            kind=TerminalReceiptKind.LOCAL_DECODE_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=59,
        )
        .wire_receipt
    )
    frames = encode_packed_control_frames(
        remote_rank.nixl_agent_name,
        str(uuid.UUID(bytes=remote_rank.process_generation)),
        PackedTerminalReceipt(
            key=receipt.binding.request_key,
            receipt_payload=receipt.encode(),
        ),
    )

    manager._dispatch_terminal_decode_control(frames)

    manager._terminal_decode_serving.coordinator_receipt_received.assert_called_once_with(
        receipt,
        remote,
    )
    manager._terminal_process_reactor.notify_coordinator_deadline_changed.assert_called_once_with()


def test_source_routes_authenticated_failure_to_failure_ingress() -> None:
    """Keep decode failure terminality distinct from request readiness."""

    local_binding = _binding("prefill-a", 0)
    local = local_binding.advertisement.terminal_identity
    decoder = local_binding.matrix.rank("decode-a", 0).terminal_identity
    manager = _manager(local_binding, {})
    manager._terminal_startup_peer_enrollment = SimpleNamespace(binding=local_binding)
    manager._terminal_source_serving = MagicMock()
    importer = MagicMock()
    manager._terminal_source_receipt_importers = {decoder: importer}
    issued = TerminalWireReceiptIssuer(decoder).issue(
        binding=_request_binding(local),
        kind=TerminalReceiptKind.FAILURE,
        outcome=TerminalReceiptOutcome.FAILURE,
        terminal_timestamp_ns=41,
    )
    importer.import_receipt.return_value = issued.local_receipt

    manager.receive_terminal_source_receipt(issued.wire_receipt, decoder)

    manager._terminal_source_serving.request_failed.assert_called_once_with(
        binding_digest=issued.wire_receipt.binding.digest,
        wire_receipt=issued.wire_receipt,
        local_receipt=issued.local_receipt,
        authenticated_issuer=decoder,
        reason="request-global coordination failed",
    )
    manager._terminal_source_serving.request_ready.assert_not_called()
    importer.retire_binding.assert_called_once_with(issued.wire_receipt.binding)


def test_source_rejects_mismatched_failure_outcome_before_import() -> None:
    """Reject malformed terminality before consuming importer authority."""

    local_binding = _binding("prefill-a", 0)
    local = local_binding.advertisement.terminal_identity
    decoder = local_binding.matrix.rank("decode-a", 0).terminal_identity
    manager = _manager(local_binding, {})
    manager._terminal_startup_peer_enrollment = SimpleNamespace(binding=local_binding)
    importer = MagicMock()
    manager._terminal_source_receipt_importers = {decoder: importer}
    receipt = (
        TerminalWireReceiptIssuer(decoder)
        .issue(
            binding=_request_binding(local),
            kind=TerminalReceiptKind.FAILURE,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=43,
        )
        .wire_receipt
    )

    with pytest.raises(RuntimeError, match="another receipt kind"):
        manager.receive_terminal_source_receipt(receipt, decoder)

    importer.import_receipt.assert_not_called()


def test_source_retains_exact_decoder_roster_before_runtime_composition() -> None:
    """Source activation freezes all decoders before terminal composition."""

    binding = _binding("prefill-a", 0)
    manager = _manager(
        binding,
        {
            b"decode-metadata-a": "decode-agent-a",
            b"decode-metadata-b": "decode-agent-b",
        },
    )
    decoder_ranks = tuple(
        rank for rank in binding.matrix.ranks if rank.role is TerminalOwnerRole.DECODE
    )
    registration_by_service = {
        rank.service_id: _decoder_registration_frames(
            _binding(rank.service_id, rank.tensor_parallel_rank)
        )
        for rank in decoder_ranks
    }
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull
    events: list[tuple[str, object]] = []
    publication_control = object()

    def enroll_publication_routes(deadline: float) -> object:
        assert deadline > time.monotonic()
        assert manager.terminal_peer_enrollment_frozen is False
        assert not manager._terminal_runtime_activated.is_set()
        events.append(("publication-roster", None))
        return publication_control

    def send_ack(
        enrollment: _TerminalDecoderEnrollment,
        route_table: TerminalDecodeControlRouteTable,
    ) -> None:
        """Record one source acknowledgement after route-table freeze.

        :param enrollment: Exact target decoder registration.
        :param route_table: Complete target-service listener table.
        """

        assert manager.terminal_peer_enrollment_frozen
        assert manager.terminal_source_publication_control is publication_control
        assert route_table.decoder_service_id == enrollment.rank.service_id
        assert route_table.route_for(enrollment.rank.terminal_identity).endpoint == (
            NetworkAddress(
                enrollment.registration.endpoint,
                enrollment.registration.dst_port,
            )
        )
        assert set(manager.decode_kv_args_table) == {
            "decode-agent-a",
            "decode-agent-b",
        }
        assert not manager._terminal_runtime_activated.is_set()
        assert not manager._runtime_workers_started
        events.append(("ack", enrollment.rank.key))

    def compose_runtime() -> None:
        assert not manager._terminal_runtime_activated.is_set()
        assert [event[0] for event in events] == [
            "publication-roster",
            "ack",
            "ack",
        ]
        events.append(("runtime", None))

    manager._enroll_terminal_source_publication_routes = enroll_publication_routes
    manager._send_terminal_startup_ack = send_ack
    manager._compose_terminal_runtime = compose_runtime
    manager._start_prefill_runtime_workers = MagicMock()

    try:
        for rank in reversed(decoder_ranks):
            inbox.send(registration_by_service[rank.service_id])
        manager.activate_terminal_startup()
    finally:
        inbox.close()

    assert events == [
        ("publication-roster", None),
        ("ack", ("decode-a", 0)),
        ("ack", ("decode-b", 0)),
        ("runtime", None),
    ]
    assert manager.agent.add_calls == [
        b"decode-metadata-b",
        b"decode-metadata-a",
    ]
    assert manager._terminal_runtime_activated.is_set()
    assert not manager._runtime_workers_started
    manager._start_prefill_runtime_workers.assert_not_called()


def test_decoder_freezes_sources_before_registration_and_requires_all_acks() -> None:
    """Decoder activation publishes only after freeze and commits after all ACKs."""

    binding = _binding("decode-a", 0)
    manager = _manager(
        binding,
        {
            b"prefill-metadata-0": "prefill-agent-0",
            b"prefill-metadata-1": "prefill-agent-1",
        },
    )
    routes = _source_routes(binding)
    registration_frames = _decoder_registration_frames(binding)
    source_ranks = tuple(
        rank for rank in binding.matrix.ranks if rank.role is TerminalOwnerRole.SOURCE
    )
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull
    events: list[tuple[str, object]] = []

    def enroll_sources(deadline: float) -> SimpleNamespace:
        assert deadline > time.monotonic()
        assert not manager._terminal_runtime_activated.is_set()
        peers = manager.enroll_terminal_prefill_routes(
            "127.0.0.1:31000",
            routes,
        )
        events.append(("roster", tuple(peer.agent_name for peer in peers)))
        return SimpleNamespace(
            routes=tuple(_BootstrapRoute(route) for route in routes),
        )

    def registration_image() -> tuple[bytes, ...]:
        assert manager.terminal_peer_enrollment_frozen
        events.append(("registration-image", None))
        return registration_frames

    def send_registration(
        route: dict[str, object],
        frames: tuple[bytes, ...],
    ) -> None:
        assert manager.terminal_peer_enrollment_frozen
        assert not manager._terminal_runtime_activated.is_set()
        assert frames == registration_frames
        events.append(("registration", route["nixl_agent_name"]))

    def compose_runtime() -> None:
        assert not manager._terminal_runtime_activated.is_set()
        events.append(("runtime", None))

    manager._enroll_terminal_source_routes = enroll_sources
    manager._decode_registration_frames = registration_image
    manager._send_terminal_decoder_registration = send_registration
    manager._compose_terminal_runtime = compose_runtime
    manager._start_decode_runtime_workers = MagicMock()

    try:
        for source_rank in reversed(source_ranks):
            inbox.send(_ack_frames(binding, source_rank, registration_frames))
        manager.activate_terminal_startup()
    finally:
        inbox.close()

    assert events == [
        ("roster", ("prefill-agent-0", "prefill-agent-1")),
        ("registration-image", None),
        ("registration", "prefill-agent-0"),
        ("registration", "prefill-agent-1"),
        ("runtime", None),
    ]
    assert manager.agent.add_calls == [
        b"prefill-metadata-0",
        b"prefill-metadata-1",
    ]
    assert manager._terminal_runtime_activated.is_set()
    assert not manager._runtime_workers_started
    manager._start_decode_runtime_workers.assert_not_called()


def test_decoder_rejects_duplicate_source_ack() -> None:
    """One source rank cannot satisfy another rank's acknowledgement slot."""

    binding = _binding("decode-a", 0)
    manager = _manager(binding, {})
    registration_frames = _decoder_registration_frames(binding)
    source = binding.matrix.rank("prefill-a", 0)
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull

    try:
        duplicate = _ack_frames(binding, source, registration_frames)
        inbox.send(duplicate)
        inbox.send(duplicate)
        with pytest.raises(RuntimeError, match="received twice"):
            manager._wait_for_terminal_source_acks(
                registration_frames,
                time.monotonic() + 1.0,
            )
    finally:
        inbox.close()


def test_decoder_rejects_ack_targeting_another_matrix_decoder() -> None:
    """A valid matrix ACK for another decoder cannot commit this process."""

    binding = _binding("decode-a", 0)
    manager = _manager(binding, {})
    registration_frames = _decoder_registration_frames(binding)
    source = binding.matrix.rank("prefill-a", 0)
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull

    try:
        inbox.send(
            _ack_frames(
                binding,
                source,
                registration_frames,
                target_service_id="decode-b",
            )
        )
        with pytest.raises(RuntimeError, match="targets another decoder"):
            manager._wait_for_terminal_source_acks(
                registration_frames,
                time.monotonic() + 1.0,
            )
    finally:
        inbox.close()


def test_decoder_does_not_commit_with_a_missing_source_ack() -> None:
    """A partial source ACK population expires without setting readiness."""

    binding = _binding("decode-a", 0)
    manager = _manager(binding, {})
    registration_frames = _decoder_registration_frames(binding)
    source = binding.matrix.rank("prefill-a", 0)
    inbox = _InprocInbox()
    manager.server_socket = inbox.pull

    try:
        inbox.send(_ack_frames(binding, source, registration_frames))
        with pytest.raises(RuntimeError, match="activation timed out"):
            manager._wait_for_terminal_source_acks(
                registration_frames,
                time.monotonic() + 0.02,
            )
    finally:
        inbox.close()

    assert not manager._terminal_runtime_activated.is_set()
    assert not manager._runtime_workers_started
