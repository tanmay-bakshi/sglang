import base64
import hashlib
import threading
import time
import uuid
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.nixl.conn import (
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    NixlKVManager,
    _TerminalDecoderEnrollment,
)
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedRegistrationAdvertisement,
)
from sglang.srt.disaggregation.nixl.startup_enrollment_ack import (
    TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
    build_terminal_startup_enrollment_ack,
    encode_terminal_startup_enrollment_ack,
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
    }[local.nixl_agent_name]
    manager.agent_metadata = local_metadata
    manager.attn_tp_size = local.tensor_parallel_size
    manager.attn_tp_rank = local.tensor_parallel_rank
    manager.attn_cp_size = 1
    manager.pp_size = 1
    manager.server_args = SimpleNamespace(pd_terminal_startup_timeout_seconds=1.0)
    manager._terminal_startup_binding = None
    manager._terminal_startup_peer_enrollment = None
    manager._terminal_source_publication_control = None
    manager._terminal_runtime_activated = threading.Event()
    manager._terminal_activation_lock = threading.Lock()
    manager._terminal_activation_started = False
    manager._terminal_bootstrap_thread = None
    manager._runtime_workers_started = False
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
    manager.rank_port = 33000 + (0 if rank.service_id == "decode-a" else 1)
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
    acknowledgement = build_terminal_startup_enrollment_ack(
        binding.matrix,
        source_rank,
        target,
        registration_frames,
    )
    return (
        TERMINAL_STARTUP_ENROLLMENT_ACK_TAG,
        encode_terminal_startup_enrollment_ack(acknowledgement),
    )


def test_activation_is_one_shot_and_precommit_failure_starts_no_workers() -> None:
    """A failed first activation permanently fails closed before worker launch."""

    manager = _manager(_binding("prefill-a", 0), {})
    manager._activate_terminal_source = MagicMock(
        side_effect=RuntimeError("precommit enrollment failed")
    )
    manager._start_prefill_runtime_workers = MagicMock()

    with pytest.raises(RuntimeError, match="precommit enrollment failed"):
        manager.activate_terminal_startup()

    assert manager._terminal_activation_started
    assert not manager._terminal_runtime_activated.is_set()
    assert not manager._runtime_workers_started
    manager._start_prefill_runtime_workers.assert_not_called()

    manager._activate_terminal_source = MagicMock()
    with pytest.raises(RuntimeError, match="cannot repeat"):
        manager.activate_terminal_startup()
    manager._activate_terminal_source.assert_not_called()
    manager._start_prefill_runtime_workers.assert_not_called()


def test_source_retains_exact_decoder_roster_before_acks_and_workers() -> None:
    """Source activation freezes all decoders before any ACK or worker starts."""

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

    def send_ack(enrollment: _TerminalDecoderEnrollment) -> None:
        assert manager.terminal_peer_enrollment_frozen
        assert manager.terminal_source_publication_control is publication_control
        assert set(manager.decode_kv_args_table) == {
            "decode-agent-a",
            "decode-agent-b",
        }
        assert not manager._terminal_runtime_activated.is_set()
        assert not manager._runtime_workers_started
        events.append(("ack", enrollment.rank.key))

    def start_workers() -> None:
        assert manager._terminal_runtime_activated.is_set()
        assert [event[0] for event in events] == [
            "publication-roster",
            "ack",
            "ack",
        ]
        manager._runtime_workers_started = True
        events.append(("workers", None))

    manager._enroll_terminal_source_publication_routes = enroll_publication_routes
    manager._send_terminal_startup_ack = send_ack
    manager._start_prefill_runtime_workers = start_workers

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
        ("workers", None),
    ]
    assert manager.agent.add_calls == [
        b"decode-metadata-b",
        b"decode-metadata-a",
    ]
    assert manager._terminal_runtime_activated.is_set()
    assert manager._runtime_workers_started


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

    def start_workers() -> None:
        assert manager._terminal_runtime_activated.is_set()
        manager._runtime_workers_started = True
        events.append(("workers", None))

    manager._enroll_terminal_source_routes = enroll_sources
    manager._decode_registration_frames = registration_image
    manager._send_terminal_decoder_registration = send_registration
    manager._start_decode_runtime_workers = start_workers

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
        ("workers", None),
    ]
    assert manager.agent.add_calls == [
        b"prefill-metadata-0",
        b"prefill-metadata-1",
    ]
    assert manager._terminal_runtime_activated.is_set()
    assert manager._runtime_workers_started


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
