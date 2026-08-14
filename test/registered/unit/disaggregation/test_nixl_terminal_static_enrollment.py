import base64
import hashlib
import inspect
import threading
import uuid
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sglang.srt.disaggregation.nixl.conn import (
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
)
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedRegistrationAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
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
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_DIGEST = bytes.fromhex("11" * 32)


def test_terminal_production_composition_uses_monotonic_raw_clock() -> None:
    """Every timing-bearing production composition uses the canonical raw clock."""

    methods = (
        NixlKVManager._compose_terminal_source,
        NixlKVManager._compose_terminal_decode,
        NixlKVManager._compose_terminal_runtime,
    )
    sources = tuple(inspect.getsource(method) for method in methods)

    assert all("time.monotonic_ns" not in source for source in sources)
    assert sum(source.count("SystemTerminalOwnerClock") for source in sources) == 3


class _FakeRemoteHandle:
    """Stable native handle selected by one metadata blob."""

    name: str

    def __init__(self, name: str) -> None:
        """Construct one remote handle.

        :param name: Metadata-selected NIXL agent name.
        """

        self.name = name


class _FakeAgent:
    """Minimal native peer lifecycle used by static enrollment tests."""

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
    """Build one manager shell and install its sealed startup authority.

    :param binding: Exact local startup rank binding.
    :param metadata_names: Remote metadata-to-agent mapping.
    :returns: Manager with native peer lifecycle state.
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
    manager.agent_metadata = (
        b"prefill-metadata-0"
        if local.nixl_agent_name == "prefill-agent-0"
        else b"decode-metadata-a"
    )
    manager.attn_tp_size = local.tensor_parallel_size
    manager.attn_tp_rank = local.tensor_parallel_rank
    manager._terminal_startup_binding = None
    manager._terminal_startup_peer_enrollment = None
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


def _decoder_registration(
    rank: TerminalStartupRankAdvertisement,
) -> KVArgsRegisterInfo:
    """Build one full-metadata decoder registration.

    :param rank: Exact decoder matrix row.
    :returns: Registration suitable for native peer retention.
    """

    suffix = rank.service_id.removeprefix("decode-")
    return KVArgsRegisterInfo(
        room="None",
        endpoint="127.0.0.1",
        dst_port=33000 if suffix == "a" else 33001,
        agent_name=rank.nixl_agent_name,
        agent_metadata=f"decode-metadata-{suffix}".encode("ascii"),
        dst_kv_ptrs=[0x1000],
        dst_kv_mem_kinds=["VRAM"],
        dst_aux_ptrs=[0x2000],
        dst_state_data_ptrs=[],
        gpu_id=0,
        decode_tp_size=rank.tensor_parallel_size,
        decode_tp_rank=rank.tensor_parallel_rank,
        dst_kv_item_len=256,
        dst_kv_item_lens=[256],
        process_generation=str(uuid.UUID(bytes=rank.process_generation)),
        registration_digest=f"registration-{suffix}",
    )


def test_decode_eagerly_loads_and_freezes_complete_source_roster() -> None:
    """All source handles exist before a terminal request can resolve them."""

    binding = _binding("decode-a", 0)
    manager = _manager(
        binding,
        {
            b"prefill-metadata-0": "prefill-agent-0",
            b"prefill-metadata-1": "prefill-agent-1",
        },
    )
    routes = tuple(
        _source_route(rank)
        for rank in binding.matrix.ranks
        if rank.role is TerminalOwnerRole.SOURCE
    )
    manager._decode_registration_frames = MagicMock(
        return_value=(b"NixlMsgGuard", b"None")
    )
    manager._send_terminal_decoder_registration = MagicMock()

    peers = manager.enroll_terminal_prefill_routes("127.0.0.1:31000", routes)

    assert tuple(peer.agent_name for peer in peers) == (
        "prefill-agent-0",
        "prefill-agent-1",
    )
    assert manager.terminal_peer_enrollment_frozen
    manager.wait_for_terminal_peer_enrollment(0.01)
    assert manager.agent.add_calls == [
        b"prefill-metadata-0",
        b"prefill-metadata-1",
    ]
    manager._send_terminal_decoder_registration.assert_not_called()


def test_terminal_receiver_resolves_only_pre_enrolled_peers_without_send() -> None:
    """Request initialization cannot mutate or re-register a terminal roster."""

    binding = _binding("decode-a", 0)
    manager = _manager(
        binding,
        {
            b"prefill-metadata-0": "prefill-agent-0",
            b"prefill-metadata-1": "prefill-agent-1",
        },
    )
    routes = tuple(
        _source_route(rank)
        for rank in binding.matrix.ranks
        if rank.role is TerminalOwnerRole.SOURCE
    )
    manager._decode_registration_frames = MagicMock(return_value=(b"guard",))
    manager._send_terminal_decoder_registration = MagicMock()
    manager.enroll_terminal_prefill_routes("127.0.0.1:31000", routes)
    manager.record_failure = MagicMock()
    manager.update_status = MagicMock()

    receiver = object.__new__(NixlKVReceiver)
    receiver.kv_mgr = manager
    receiver.bootstrap_addr = "127.0.0.1:31000"
    receiver.bootstrap_room = 41
    receiver.bootstrap_infos = [dict(route, is_dummy=False) for route in routes]
    receiver.prefill_peers = []
    receiver.conclude_state = None
    receiver._connect_to_bootstrap_server = MagicMock()

    assert receiver._register_kv_args()
    assert tuple(peer.agent_name for peer in receiver.prefill_peers) == (
        "prefill-agent-0",
        "prefill-agent-1",
    )
    receiver._connect_to_bootstrap_server.assert_not_called()
    assert manager.agent.add_calls == [
        b"prefill-metadata-0",
        b"prefill-metadata-1",
    ]
    manager._send_terminal_decoder_registration.assert_not_called()


def test_decode_rejects_matrix_drift_before_creating_native_handle() -> None:
    """A full route with another generation cannot spend native authority."""

    binding = _binding("decode-a", 0)
    manager = _manager(
        binding,
        {
            b"prefill-metadata-0": "prefill-agent-0",
            b"prefill-metadata-1": "prefill-agent-1",
        },
    )
    routes = [
        _source_route(rank)
        for rank in binding.matrix.ranks
        if rank.role is TerminalOwnerRole.SOURCE
    ]
    routes[1]["process_generation"] = str(uuid.UUID(int=999))

    with pytest.raises(RuntimeError, match="differs from sealed matrix"):
        manager.enroll_terminal_prefill_routes(
            "127.0.0.1:31000",
            tuple(routes),
        )

    assert manager.agent.add_calls == []
    assert not manager.terminal_peer_enrollment_frozen


def test_source_freezes_only_after_every_decoder_rank_is_retained() -> None:
    """Source roster completion is exact across independent decoder services."""

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

    manager._add_remote_peer(_decoder_registration(decoder_ranks[0]))
    assert not manager.terminal_peer_enrollment_frozen
    manager._add_remote_peer(_decoder_registration(decoder_ranks[1]))

    assert manager.terminal_peer_enrollment_frozen
    manager.wait_for_terminal_peer_enrollment(0.01)
    assert set(manager.decode_kv_args_table) == {
        "decode-agent-a",
        "decode-agent-b",
    }
    assert manager.agent.add_calls == [b"decode-metadata-a", b"decode-metadata-b"]

    with pytest.raises(RuntimeError, match="roster is frozen"):
        manager._add_remote_peer(_decoder_registration(decoder_ranks[0]))


def test_source_rejects_decoder_tp_drift_before_native_registration() -> None:
    """Agent metadata cannot claim another rank or TP width."""

    binding = _binding("prefill-a", 0)
    manager = _manager(binding, {b"decode-metadata-a": "decode-agent-a"})
    decoder_rank = binding.matrix.rank("decode-a", 0)
    registration = _decoder_registration(decoder_rank)
    registration.decode_tp_size = 2

    with pytest.raises(RuntimeError, match="differs from sealed matrix"):
        manager._add_remote_peer(registration)

    assert manager.agent.add_calls == []
    assert manager.decode_kv_args_table == {}


def test_binding_install_rejects_local_metadata_drift() -> None:
    """A matrix cannot authorize a manager with another frozen metadata image."""

    binding = _binding("decode-a", 0)
    manager = object.__new__(NixlKVManager)
    local = binding.advertisement
    manager.disaggregation_mode = DisaggregationMode.DECODE
    manager.process_generation = str(uuid.UUID(bytes=local.process_generation))
    manager.agent = _FakeAgent(local.nixl_agent_name, {})
    manager.agent_metadata = b"different-local-metadata"
    manager.attn_tp_size = 1
    manager.attn_tp_rank = 0
    manager._terminal_startup_binding = None
    manager._terminal_startup_peer_enrollment = None

    with pytest.raises(RuntimeError, match="another NIXL metadata"):
        manager.install_terminal_startup_binding(binding)


def test_terminal_decoder_registration_serializes_one_complete_frozen_image() -> None:
    """Process-lifetime registration has one exact multipart image."""

    binding = _binding("decode-a", 0)
    manager = _manager(binding, {})
    manager.local_ip = "127.0.0.1"
    manager.rank_port = 33000
    manager.enable_staging = False
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[4096, 4096],
        kv_data_mem_kinds=["VRAM", "VRAM"],
        kv_item_lens=[256, 256],
        kv_layer_ids=[3, 4],
        aux_data_ptrs=[0x3000],
        state_data_ptrs=[],
        state_item_lens=[],
        state_dim_per_tensor=[],
        state_layer_ids=[],
        gpu_id=0,
        engine_rank=0,
    )
    advertisement = PackedRegistrationAdvertisement(
        base_address=0x4000,
        total_size=8192,
        arena_generation=uuid.UUID(int=301).bytes,
        visibility_policy_digest=bytes.fromhex("22" * 32),
        runtime_cohort_digest=bytes.fromhex("33" * 32),
        page_size=64,
    )
    manager._packed_decode_controller = SimpleNamespace(
        ready=True,
        prepared_grant_protocol=PACKED_PREPARED_GRANT_PROTOCOL,
        advertisement=advertisement,
    )

    frames = manager._decode_registration_frames()
    registration = KVArgsRegisterInfo.from_zmq(list(frames[1:]))

    assert registration.agent_name == "decode-agent-a"
    assert registration.agent_metadata == b"decode-metadata-a"
    assert registration.process_generation == str(uuid.UUID(int=201))
    assert registration.decode_tp_size == 1
    assert registration.decode_tp_rank == 0
    assert registration.dst_kv_ptrs == [0x1000, 0x2000]
    assert registration.dst_kv_item_lens == [256, 256]
    assert registration.dst_kv_layer_ids == [3, 4]
    assert registration.packed_advertisement == advertisement
