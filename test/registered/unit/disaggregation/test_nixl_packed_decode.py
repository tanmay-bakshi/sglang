import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedReady,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.decode import PackedNixlDecodeController
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_PREPARED_GRANT_PROTOCOL,
    decode_packed_control_frames,
    encode_packed_control_frames,
)
from sglang.srt.disaggregation.nixl.packed_staging import PackedPeerIdentity


class _Manager:
    """Expose decoder control identity used by the controller."""

    process_generation: str

    def __init__(self, process_generation: str) -> None:
        """Initialize one process identity.

        :param process_generation: Exact process generation UUID.
        """

        self.process_generation = process_generation


class _Runtime:
    """Record decode actor dispatch and progress calls."""

    ready: bool
    controls: list[tuple[StagingWriterId, object]]
    polls: list[object]

    def __init__(self, ready: bool) -> None:
        """Initialize a fake actor.

        :param ready: Initial scheduler attachment readiness.
        """

        self.ready = ready
        self.controls = []
        self.polls = []

    def handle_control(
        self,
        writer_id: StagingWriterId,
        message: object,
    ) -> None:
        """Record one authenticated control dispatch.

        :param writer_id: Authenticated source writer.
        :param message: Decoded wire payload.
        """

        self.controls.append((writer_id, message))

    def poll(self, transaction: object) -> KVPoll:
        """Record and complete one scheduler poll.

        :param transaction: Exact transaction owner.
        :returns: Successful transfer state.
        """

        self.polls.append(transaction)
        return KVPoll.Success


def _writer() -> StagingWriterId:
    """Return canonical TP0 source identity.

    :returns: Canonical source writer.
    """

    return StagingWriterId(
        transfer_source_rank=0,
        source_attn_tp_rank=0,
        source_pp_rank=0,
        source_cp_rank=0,
    )


def _ready(writer: StagingWriterId) -> PackedReady:
    """Build one valid control payload.

    :param writer: Exact source writer.
    :returns: Valid READY message.
    """

    return PackedReady(
        key=PackedChunkKey(
            room_id=17,
            chunk_id=0,
            request_generation=b"r" * 16,
        ),
        writer_id=writer,
        digest=b"d" * 32,
        visibility_policy_digest=b"v" * 32,
        lease_id=9,
        lease_base_address=0x400000,
        projection_offset=0,
        projection_length=4096,
    )


def _controller(
    manager: _Manager,
    runtime: _Runtime,
    peer: PackedPeerIdentity,
) -> PackedNixlDecodeController:
    """Construct the control-only projection of a controller.

    :param manager: Decoder process identity.
    :param runtime: Fake decode actor.
    :param peer: Decoder NIXL identity.
    :returns: Controller projection used by actor-boundary tests.
    """

    controller = object.__new__(PackedNixlDecodeController)
    controller._manager = manager
    controller._runtime = runtime
    controller._peer = peer
    return controller


def test_protocol_is_advertised_only_after_scheduler_attachment() -> None:
    """Keep prepared-grant capability closed until actor readiness."""

    generation = str(uuid.uuid4())
    peer = PackedPeerIdentity("decode-agent", uuid.UUID(generation).bytes)
    runtime = _Runtime(ready=False)
    controller = _controller(_Manager(generation), runtime, peer)

    assert controller.prepared_grant_protocol is None

    runtime.ready = True

    assert controller.prepared_grant_protocol == PACKED_PREPARED_GRANT_PROTOCOL


def test_controller_constructs_for_tp2_rank_one() -> None:
    """Create a destination-rank-local actor for the second TP2 rank."""

    generation = str(uuid.uuid4())
    manager = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=1,
        attn_cp_rank=0,
        pp_rank=0,
        process_generation=generation,
        agent=SimpleNamespace(name="decode-rank-1"),
        kv_args=SimpleNamespace(gpu_id=1),
    )
    arena = MagicMock()
    runtime = MagicMock()
    with (
        patch(
            "sglang.srt.disaggregation.nixl.decode.load_exact_nixl_runtime_artifacts",
            return_value=object(),
        ),
        patch(
            "sglang.srt.disaggregation.nixl.decode.build_same_host_visibility_policy",
            return_value=object(),
        ),
        patch(
            "sglang.srt.disaggregation.nixl.decode.PackedStagingArena",
            return_value=arena,
        ),
        patch(
            "sglang.srt.disaggregation.nixl.decode.PackedDecodeRuntime",
            return_value=runtime,
        ) as runtime_class,
    ):
        controller = PackedNixlDecodeController(manager, object(), object())

    runtime_class.assert_called_once()
    assert controller._runtime is runtime


def test_control_frames_require_generation_bound_native_peer() -> None:
    """Dispatch source control only after exact native-peer authentication."""

    decode_generation = str(uuid.uuid4())
    source_generation = str(uuid.uuid4())
    decode_peer = PackedPeerIdentity(
        "decode-agent",
        uuid.UUID(decode_generation).bytes,
    )
    source_peer = PackedPeerIdentity(
        "source-agent",
        uuid.UUID(source_generation).bytes,
    )
    runtime = _Runtime(ready=True)
    controller = _controller(_Manager(decode_generation), runtime, decode_peer)
    writer = _writer()
    message = _ready(writer)
    frames = encode_packed_control_frames(
        source_peer.agent_name,
        source_generation,
        message,
    )

    controller.handle_control_frames(frames, source_peer, writer)

    assert runtime.controls == [(writer, message)]
    stale_peer = PackedPeerIdentity("source-agent", uuid.uuid4().bytes)
    with pytest.raises(
        RuntimeError,
        match="generation differs from native peer",
    ):
        controller.handle_control_frames(frames, stale_peer, writer)


def test_control_sender_binds_decoder_identity_and_actor_progress() -> None:
    """Encode decoder identity on outbound control and delegate polling."""

    generation = str(uuid.uuid4())
    peer = PackedPeerIdentity("decode-agent", uuid.UUID(generation).bytes)
    manager = _Manager(generation)
    runtime = _Runtime(ready=True)
    controller = _controller(manager, runtime, peer)
    writer = _writer()
    message = _ready(writer)
    sent_frames: list[list[bytes]] = []
    sender = controller.build_control_sender(writer, sent_frames.append)

    sender.send_message(message)

    assert sender.writer_id == writer
    assert decode_packed_control_frames(sent_frames[0]) == (
        peer.agent_name,
        generation,
        message,
    )
    transaction = object()
    assert controller.poll(transaction) == KVPoll.Success
    assert runtime.polls == [transaction]
