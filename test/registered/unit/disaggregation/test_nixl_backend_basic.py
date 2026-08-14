"""Basic CPU unit tests for NIXL disaggregation control paths."""

import base64
import hashlib
import struct
import sys
import threading
import types
import unittest
import uuid
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    NIXL_AGENT_METADATA_MAX_BYTES,
    NIXL_AGENT_NAME_MAX_BYTES,
    NIXL_BOOTSTRAP_PEER_PROTOCOL,
    SERIALIZED_RANK_LIMIT,
    CommonKVManager,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedReady,
    PackedRequestKey,
)
from sglang.srt.disaggregation.common.packed_staging_wire import encode_packed_message
from sglang.srt.disaggregation.common.staging_handler import PrefillStagingContext
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    NIXL_DIRECT_KV_MAX_COHORT_DESCRIPTORS,
    NIXL_RMA_MAX_DESCRIPTORS,
    NIXL_RMA_SEGMENT_BYTES,
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
    TransferKVChunk,
    TransferStatus,
    _build_contiguous_rma_requests,
)
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PACKED_KV_TRANSFER_PROTOCOL,
    PACKED_PREPARED_GRANT_PROTOCOL,
    PackedLegacyAuxiliarySource,
    PackedPrefillLaunchPlan,
    PackedRegistrationAdvertisement,
    encode_packed_control_frames,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


class FakeRemoteHandle:
    def __init__(self, name, identity=1, generation=1):
        self.name = name
        self.identity = identity
        self.generation = generation


class PointerEqualRemoteHandle(FakeRemoteHandle):
    def __eq__(self, other: object) -> bool:
        if type(other) is not PointerEqualRemoteHandle:
            return False
        return self.identity == other.identity and self.generation == other.generation

    def __hash__(self) -> int:
        return hash((self.identity, self.generation))


class NotificationFakeAgent:
    def __init__(self, messages, peer_handle=None):
        self.messages = messages
        self.peer_handle = peer_handle or FakeRemoteHandle("prefill")
        self.name = "decode-agent"

    def get_new_notifs(self):
        return {self.peer_handle: [msg.encode("ascii") for msg in self.messages]}


class StagingFakeAgent:
    def __init__(self, register_result=None):
        self.register_result = (
            register_result if register_result is not None else ["desc"]
        )
        self.register_memory_calls = []
        self.get_xfer_descs_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []
        self.release_xfer_handle_calls = []

    def register_memory(self, addrs, mem_type):
        self.register_memory_calls.append((addrs, mem_type))
        return self.register_result

    def get_xfer_descs(self, reqs, mem_type):
        self.get_xfer_descs_calls.append((reqs, mem_type))
        return f"{mem_type}_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        self.initialize_xfer_calls.append(args)
        return "handle"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.release_xfer_handle_calls.append(handle)


class TransferFailureFakeAgent(StagingFakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        transport = SimpleNamespace(transport="rc_mlx5", device="mlx5_3:1")
        self.attestation = SimpleNamespace(
            state="FAILED",
            status="NIXL_ERR_BACKEND",
            error="ucp_ep_flush_nbx failed",
            backend="UCX",
            submissionSealed=True,
            completionClaimed=False,
            segments=(
                SimpleNamespace(
                    index=0,
                    localAddress=0x9000,
                    remoteAddress=0x100000,
                    localDeviceId=1,
                    remoteDeviceId=4,
                    length=32 * 1024 * 1024,
                    posted=True,
                    endpointIdentity="endpoint-7",
                    requestInfo="ucx-request-9",
                    selectedTransports=(transport,),
                ),
            ),
            endpoints=(
                SimpleNamespace(
                    workerId=3,
                    workerIdentity="worker-3",
                    endpointIdentity="endpoint-7",
                    segmentIndices=(0,),
                    flushPosted=True,
                    remoteFlushed=False,
                    transports=(transport,),
                ),
            ),
        )

    def transfer(self, handle):
        self.events.append("transfer")
        raise RuntimeError("NIXL_ERR_BACKEND")

    def query_xfer_attestation(self, handle):
        self.events.append("attestation")
        return self.attestation

    def release_xfer_handle(self, handle):
        self.events.append("release")
        super().release_xfer_handle(handle)


class MetadataSnapshotFakeAgent:
    def __init__(self) -> None:
        self.registered_addresses: list[int] = []
        self.metadata_snapshots: list[bytes] = []

    def create_backend(self, backend: str, parameters: dict[str, str]) -> None:
        pass

    def get_plugin_list(self) -> list[str]:
        return ["UCX"]

    def register_memory(
        self, addresses: list[tuple[int, int, int, str]], memory_type: str
    ) -> list[str]:
        self.registered_addresses.extend(address for address, _, _, _ in addresses)
        return ["descriptor"]

    def get_agent_metadata(self) -> bytes:
        snapshot = ",".join(
            str(address) for address in self.registered_addresses
        ).encode("ascii")
        self.metadata_snapshots.append(snapshot)
        return snapshot


class PeerLifecycleFakeAgent:
    def __init__(self, metadata_names, *, connection_error=None, removal_error=None):
        self.metadata_names = metadata_names
        self.connection_error = connection_error
        self.removal_error = removal_error
        self.handles = {}
        self.generations = defaultdict(int)
        self.add_calls = []
        self.connection_calls = []
        self.removal_calls = []

    def add_remote_agent(self, metadata):
        self.add_calls.append(metadata)
        existing = self.handles.get(metadata)
        if existing is not None:
            return existing
        self.generations[metadata] += 1
        handle = FakeRemoteHandle(
            self.metadata_names[metadata],
            identity=len(self.handles) + 1,
            generation=self.generations[metadata],
        )
        self.handles[metadata] = handle
        return handle

    def make_connection(self, handle):
        self.connection_calls.append(handle)
        if self.connection_error is not None:
            raise self.connection_error

    def remove_remote_agent(self, handle):
        self.removal_calls.append(handle)
        if self.removal_error is not None:
            raise self.removal_error
        for metadata, candidate in list(self.handles.items()):
            if candidate is handle:
                del self.handles[metadata]
                return
        raise RuntimeError("stale handle")


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeTensor:
    shape = (1, 1, 8)

    def element_size(self):
        return 2


class SizedFakeTensor:
    """Tensor geometry stub that carries no backing allocation."""

    shape: tuple[int, int, int]
    _element_size: int

    def __init__(self, head_dim: int, element_size: int = 1) -> None:
        """
        :param head_dim: Synthetic final tensor dimension.
        :param element_size: Synthetic element size in bytes.
        """

        self.shape = (1, 1, head_dim)
        self._element_size = element_size

    def element_size(self) -> int:
        """:returns: Synthetic element size in bytes."""

        return self._element_size


class FakeStagingBuffer:
    def __init__(self, ptr=0x9000, size=1 << 20):
        self.ptr = ptr
        self.size = size

    def fits(self, required_bytes):
        return required_bytes <= self.size

    def get_ptr(self):
        return self.ptr


class FakeStagingAllocator:
    ALLOC_OVERSIZED = -2


def _fake_staging_buffer_module(mock_gather=None):
    module = types.ModuleType("sglang.srt.disaggregation.common.staging_buffer")
    module.StagingAllocator = FakeStagingAllocator
    module.compute_head_slice_params = lambda *args: (0, 1, 0, 1)
    module.compute_staging_layout = lambda *args: (2, [256, 256], 512)
    module.resolve_total_kv_heads = lambda kv_args, attn_tp_size: 2
    module.gather_all_layers_to_staging = mock_gather or MagicMock()
    return module


def _packed_advertisement() -> PackedRegistrationAdvertisement:
    """Build one stable packed decoder advertisement.

    :returns: A valid process-lifetime packed registration.
    """

    return PackedRegistrationAdvertisement(
        base_address=0x200000,
        total_size=1 << 20,
        arena_generation=b"a" * 16,
        visibility_policy_digest=b"v" * 32,
        runtime_cohort_digest=b"c" * 32,
        page_size=16,
    )


def _packed_writer(source_rank: int = 0) -> StagingWriterId:
    """Build one TP writer identity.

    :param source_rank: Source TP and transfer rank.
    :returns: A PP0/CP0 source writer.
    """

    return StagingWriterId(
        transfer_source_rank=source_rank,
        source_attn_tp_rank=source_rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )


def _packed_plan(
    destination_generation: str,
    room_id: int = 41,
    canonical_source_rank: int = 0,
) -> PackedAuxiliaryPlan:
    """Build one valid decoder-authored auxiliary plan.

    :param destination_generation: Destination process generation UUID.
    :param room_id: Decoder-minted bootstrap room.
    :param canonical_source_rank: Source writer routed to the destination rank.
    :returns: A generation-bound auxiliary transfer plan.
    """

    advertisement = _packed_advertisement()
    return PackedAuxiliaryPlan(
        key=PackedRequestKey(
            room_id=room_id,
            request_generation=b"r" * 16,
        ),
        request_slot_generation=7,
        metadata_buffer_index=3,
        metadata_slot_generation=b"m" * 16,
        destination_segments=(
            PackedAuxiliaryDestinationSegment(
                address=0x210000,
                item_length=8,
            ),
        ),
        canonical_writer_id=_packed_writer(canonical_source_rank),
        destination_process_generation=uuid.UUID(destination_generation).bytes,
        native_route_digest=advertisement.visibility_policy_digest,
        runtime_cohort_digest=advertisement.runtime_cohort_digest,
    )


def _packed_ready(writer: StagingWriterId) -> PackedReady:
    """Build one valid packed READY message.

    :param writer: Exact source writer encoded in the message.
    :returns: A valid control payload.
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


class TestNixlTransferInfo(CustomTestCase):
    def test_from_zmq_parses_required_fields(self):
        kv_indices = np.array([3, 5, 8], dtype=np.int32)
        state_indices = [[1, 2], [], [9]]
        msg = [
            b"7",
            b"127.0.0.1",
            b"12345",
            b"decode_agent",
            kv_indices.tobytes(),
            b"4",
            b"2",
            pack_int_lists(state_indices, "i"),
            b"11",
            b"00000000-0000-4000-8000-000000000001",
        ]

        info = TransferInfo.from_zmq(msg)

        self.assertEqual(info.room, 7)
        self.assertEqual(info.endpoint, "127.0.0.1")
        self.assertEqual(info.dst_port, 12345)
        self.assertEqual(info.agent_name, "decode_agent")
        np.testing.assert_array_equal(info.dst_kv_indices, kv_indices)
        self.assertEqual(info.dst_aux_index, 4)
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.dst_state_indices, state_indices)
        self.assertEqual(info.decode_prefix_len, 11)
        self.assertEqual(
            info.process_generation,
            "00000000-0000-4000-8000-000000000001",
        )

    def test_from_zmq_defaults_optional_fields(self):
        info = TransferInfo.from_zmq(
            [
                b"8",
                b"127.0.0.1",
                b"12346",
                b"agent",
                np.array([1], dtype=np.int32).tobytes(),
                b"0",
                b"1",
            ]
        )

        self.assertEqual(info.dst_state_indices, [])
        self.assertIsNone(info.decode_prefix_len)

    def test_decode_radix_full_hit_is_not_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12347",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
            ]
        )

        self.assertFalse(info.is_dummy())

    def test_empty_indices_without_decode_prefix_is_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"10",
                b"127.0.0.1",
                b"12348",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"0",
            ]
        )

        self.assertTrue(info.is_dummy())

    def test_from_zmq_rejects_unbounded_agent_names(self):
        base_message = [
            b"29",
            b"127.0.0.1",
            b"12348",
            b"agent",
            np.array([], dtype=np.int32).tobytes(),
            b"2",
            b"1",
        ]
        non_ascii = list(base_message)
        non_ascii[3] = b"agent-\xff"
        oversized = list(base_message)
        oversized[3] = b"a" * (NIXL_AGENT_NAME_MAX_BYTES + 1)

        with self.assertRaises(UnicodeDecodeError):
            TransferInfo.from_zmq(non_ascii)
        with self.assertRaisesRegex(ValueError, "agent name exceeds"):
            TransferInfo.from_zmq(oversized)


class TestNixlKVArgsRegisterInfo(CustomTestCase):
    def test_from_zmq_preserves_unsigned_pointers_and_optional_fields(self):
        high_ptr = 0xFFFF_81AB_54E0_1000
        kv_ptrs = [high_ptr, high_ptr + 0x1000]
        aux_ptrs = [0x1000, 0x2000]
        state_ptrs = [[high_ptr + 0x2000], [high_ptr + 0x3000, high_ptr + 0x4000]]
        state_item_lens = [[64], [128, 256]]
        state_dims = [[16], [32, 64]]
        staging_ptr = high_ptr + 0x5000

        msg = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"agent_with_large_ptr",
            b"metadata",
            b"".join(struct.pack("Q", ptr) for ptr in kv_ptrs),
            b"".join(struct.pack("Q", ptr) for ptr in aux_ptrs),
            pack_int_lists(state_ptrs, "Q"),
            b"3",
            b"4",
            b"1",
            b"1024",
            pack_int_lists(state_item_lens, "I"),
            pack_int_lists(state_dims, "I"),
            struct.pack("Q", staging_ptr),
            b"1048576",
            b"64",
            b"DRAM,DRAM",
            b"".join(struct.pack("Q", item_len) for item_len in [1024, 2048]),
            b"",
            b"",
            b"00000000-0000-4000-8000-000000000002",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.room, "None")
        self.assertEqual(info.endpoint, "10.0.0.2")
        self.assertEqual(info.dst_port, 23456)
        self.assertEqual(info.agent_name, "agent_with_large_ptr")
        self.assertEqual(info.agent_metadata, b"metadata")
        self.assertEqual(info.dst_kv_ptrs, kv_ptrs)
        self.assertEqual(info.dst_aux_ptrs, aux_ptrs)
        self.assertEqual(info.dst_state_data_ptrs, state_ptrs)
        self.assertEqual(info.gpu_id, 3)
        self.assertEqual(info.decode_tp_size, 4)
        self.assertEqual(info.decode_tp_rank, 1)
        self.assertEqual(info.dst_kv_item_len, 1024)
        self.assertEqual(info.dst_kv_item_lens, [1024, 2048])
        self.assertEqual(info.dst_num_slots, 64)
        self.assertEqual(info.dst_kv_mem_kinds, ["DRAM", "DRAM"])
        self.assertEqual(info.dst_state_item_lens, state_item_lens)
        self.assertEqual(info.dst_state_dim_per_tensor, state_dims)
        self.assertIsNotNone(info.staging)
        self.assertEqual(info.staging.base_ptr, staging_ptr)
        self.assertEqual(info.staging.total_size, 1048576)
        self.assertEqual(
            info.process_generation,
            "00000000-0000-4000-8000-000000000002",
        )

    def test_from_zmq_allows_missing_state_and_staging_fields(self):
        msg = [
            b"None",
            b"10.0.0.3",
            b"23457",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"0",
            b"256",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_state_data_ptrs, [])
        self.assertEqual(info.dst_state_item_lens, [])
        self.assertEqual(info.dst_state_dim_per_tensor, [])
        self.assertEqual(info.dst_kv_item_lens, [256])
        self.assertIsNone(info.staging)

    def test_from_zmq_rejects_unbounded_peer_identity(self):
        base_message = [
            b"None",
            b"10.0.0.3",
            b"23457",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"0",
            b"256",
        ]
        non_ascii_name = list(base_message)
        non_ascii_name[3] = b"agent-\xff"
        oversized_name = list(base_message)
        oversized_name[3] = b"a" * (NIXL_AGENT_NAME_MAX_BYTES + 1)
        oversized_metadata = list(base_message)
        oversized_metadata[4] = b"x" * (NIXL_AGENT_METADATA_MAX_BYTES + 1)

        with self.assertRaises(UnicodeDecodeError):
            KVArgsRegisterInfo.from_zmq(non_ascii_name)
        with self.assertRaisesRegex(ValueError, "agent name exceeds"):
            KVArgsRegisterInfo.from_zmq(oversized_name)
        with self.assertRaisesRegex(ValueError, "metadata exceeds"):
            KVArgsRegisterInfo.from_zmq(oversized_metadata)


class TestNixlPackedRegistration(CustomTestCase):
    """Validate the manager-produced persistent packed registration tail."""

    @staticmethod
    def _base_frames(process_generation: str) -> list[bytes]:
        """Build the fixed legacy registration prefix.

        :param process_generation: Decoder process generation UUID.
        :returns: The 22 frames preceding an optional packed tail.
        """

        return [
            b"None",
            b"127.0.0.1",
            b"23457",
            b"decode-agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"1",
            b"1",
            b"0",
            b"256",
            b"",
            b"",
            b"",
            b"",
            b"64",
            b"VRAM",
            struct.pack("Q", 256),
            b"",
            struct.pack("I", 3),
            process_generation.encode("ascii"),
        ]

    @staticmethod
    def _registration_tail() -> tuple[bytes, ...]:
        """Serialize one ready decode controller.

        :returns: The closed eight-frame packed registration tail.
        """

        manager = object.__new__(NixlKVManager)
        manager._packed_decode_controller = SimpleNamespace(
            ready=True,
            prepared_grant_protocol=PACKED_PREPARED_GRANT_PROTOCOL,
            advertisement=_packed_advertisement(),
        )
        return manager._packed_decode_registration_frames()

    def test_ready_decode_registration_round_trips(self) -> None:
        """Parse the exact tail produced by a ready decode manager."""

        generation = str(uuid.uuid4())
        info = KVArgsRegisterInfo.from_zmq(
            self._base_frames(generation) + list(self._registration_tail())
        )

        self.assertEqual(
            info.packed_transfer_protocol,
            PACKED_KV_TRANSFER_PROTOCOL,
        )
        self.assertEqual(
            info.prepared_grant_protocol,
            PACKED_PREPARED_GRANT_PROTOCOL,
        )
        self.assertEqual(info.packed_advertisement, _packed_advertisement())
        self.assertEqual(info.process_generation, generation)

    def test_malformed_packed_registration_frames_are_rejected(self) -> None:
        """Reject truncated, unsupported, and malformed packed tails."""

        frames = self._base_frames(str(uuid.uuid4())) + list(self._registration_tail())
        cases: tuple[tuple[str, list[bytes], str], ...] = (
            (
                "truncated",
                frames[:-1],
                "invalid frame count",
            ),
            (
                "unsupported protocol",
                frames[:22] + [b"packed-v3"] + frames[23:],
                "unsupported packed decoder transfer protocol",
            ),
            (
                "short generation",
                frames[:26] + [b"short"] + frames[27:],
                "arena generation must contain 16 bytes",
            ),
        )

        for case_name, malformed_frames, expected_error in cases:
            with self.subTest(case_name=case_name):
                with self.assertRaisesRegex(ValueError, expected_error):
                    KVArgsRegisterInfo.from_zmq(malformed_frames)


class TestNixlTransferStatus(CustomTestCase):
    def test_not_done_until_aux_and_expected_count_arrive(self):
        status = TransferStatus()

        self.assertFalse(status.is_done())

        status.received_aux = True
        self.assertFalse(status.is_done())

        status.num_source_writers_expected = 1
        self.assertFalse(status.is_done())

        status.expected_kvs_per_source[0] = 1
        self.assertFalse(status.is_done())

        status.received_kvs_per_source[0].add(0)
        self.assertTrue(status.is_done())


class TestNixlPeerLifecycle(CustomTestCase):
    @staticmethod
    def _route(
        *,
        agent_name,
        metadata,
        tp_rank,
        process_generation="00000000-0000-4000-8000-000000000000",
    ):
        return {
            "transport_protocol": NIXL_BOOTSTRAP_PEER_PROTOCOL,
            "nixl_agent_name": agent_name,
            "nixl_agent_metadata": base64.b64encode(metadata).decode("ascii"),
            "nixl_agent_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "process_generation": process_generation,
            "attn_dp_rank": 0,
            "attn_cp_rank": 0,
            "attn_tp_rank": tp_rank,
            "pp_rank": 0,
            "transfer_source_rank": tp_rank,
        }

    @staticmethod
    def _manager(agent):
        manager = object.__new__(NixlKVManager)
        manager.agent = agent
        manager.disaggregation_mode = DisaggregationMode.DECODE
        manager._prefill_peers = {}
        manager._prefill_peer_keys_by_addr = defaultdict(set)
        manager._prefill_peers_by_agent_name = {}
        manager._prefill_peers_by_handle = {}
        manager._prefill_peer_lock = threading.RLock()
        manager._quarantined_remote_handles = set()
        return manager

    def test_loads_and_connects_multiple_prefill_peers(self):
        agent = PeerLifecycleFakeAgent(
            {b"metadata-a": "prefill-a", b"metadata-b": "prefill-b"}
        )
        manager = self._manager(agent)

        first = manager._load_prefill_peer(
            "prefill:8998",
            self._route(agent_name="prefill-a", metadata=b"metadata-a", tp_rank=0),
        )
        second = manager._load_prefill_peer(
            "prefill:8998",
            self._route(agent_name="prefill-b", metadata=b"metadata-b", tp_rank=1),
        )

        self.assertIsNot(first.handle, second.handle)
        self.assertEqual(agent.connection_calls, [first.handle, second.handle])
        self.assertEqual(len(manager._prefill_peers), 2)

    def test_duplicate_identical_route_reuses_exact_handle(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "prefill"})
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)

        first = manager._load_prefill_peer("prefill:8998", route)
        second = manager._load_prefill_peer("prefill:8998", dict(route))

        self.assertIs(first, second)
        self.assertIs(first.handle, second.handle)
        self.assertEqual(agent.add_calls, [b"metadata"])
        self.assertEqual(agent.connection_calls, [first.handle, first.handle])

    def test_route_rejects_non_uint32_ranks(self):
        agent = PeerLifecycleFakeAgent({})
        manager = self._manager(agent)
        rank_fields = (
            "attn_dp_rank",
            "attn_cp_rank",
            "attn_tp_rank",
            "pp_rank",
            "transfer_source_rank",
        )

        for field_name in rank_fields:
            for invalid_value in (True, "0", SERIALIZED_RANK_LIMIT):
                with self.subTest(field_name=field_name, value=invalid_value):
                    route = self._route(
                        agent_name="prefill",
                        metadata=b"metadata",
                        tp_rank=0,
                    )
                    route[field_name] = invalid_value

                    with self.assertRaisesRegex(
                        RuntimeError, f"Invalid prefill route {field_name}"
                    ):
                        manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(agent.add_calls, [])

    def test_route_accepts_maximum_uint32_rank(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "prefill"})
        manager = self._manager(agent)
        route = self._route(
            agent_name="prefill",
            metadata=b"metadata",
            tp_rank=SERIALIZED_RANK_LIMIT - 1,
        )

        peer = manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(peer.attn_tp_rank, SERIALIZED_RANK_LIMIT - 1)
        self.assertEqual(peer.transfer_source_rank, SERIALIZED_RANK_LIMIT - 1)

    def test_route_rejects_unbounded_nixl_identity(self):
        agent = PeerLifecycleFakeAgent({})
        manager = self._manager(agent)
        invalid_names = (
            "prefill-\N{SNOWMAN}",
            "a" * (NIXL_AGENT_NAME_MAX_BYTES + 1),
        )

        for invalid_name in invalid_names:
            with self.subTest(agent_name=invalid_name):
                route = self._route(
                    agent_name=invalid_name,
                    metadata=b"metadata",
                    tp_rank=0,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Invalid prefill NIXL agent name"
                ):
                    manager._load_prefill_peer("prefill:8998", route)

        oversized_metadata = b"x" * (NIXL_AGENT_METADATA_MAX_BYTES + 1)
        route = self._route(
            agent_name="prefill",
            metadata=oversized_metadata,
            tp_rank=0,
        )
        with self.assertRaisesRegex(
            RuntimeError, "Invalid prefill NIXL agent metadata"
        ):
            manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(agent.add_calls, [])

    def test_concurrent_identical_load_publishes_one_handle(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "prefill"})
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)
        results = []

        threads = [
            threading.Thread(
                target=lambda: results.append(
                    manager._load_prefill_peer("prefill:8998", dict(route))
                )
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 4)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(agent.add_calls, [b"metadata"])

    def test_stale_process_generation_is_rejected(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "prefill"})
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)
        manager._load_prefill_peer("prefill:8998", route)
        stale = dict(route)
        stale["process_generation"] = "ffffffff-ffff-4fff-8fff-ffffffffffff"

        with self.assertRaisesRegex(RuntimeError, "stale prefill"):
            manager._load_prefill_peer("prefill:8998", stale)

    def test_spoofed_agent_name_is_rejected(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "metadata-owner"})
        manager = self._manager(agent)
        route = self._route(agent_name="spoofed-name", metadata=b"metadata", tp_rank=0)

        with self.assertRaisesRegex(RuntimeError, "different agent"):
            manager._load_prefill_peer("prefill:8998", route)

    def test_remove_then_reload_uses_new_native_generation(self):
        agent = PeerLifecycleFakeAgent({b"metadata": "prefill"})
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)
        first = manager._load_prefill_peer("prefill:8998", route)

        manager._remove_prefill_peers("prefill:8998")
        second = manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(agent.removal_calls, [first.handle])
        self.assertIsNot(first.handle, second.handle)
        self.assertGreater(second.handle.generation, first.handle.generation)

    def test_connection_failure_removes_half_loaded_handle(self):
        agent = PeerLifecycleFakeAgent(
            {b"metadata": "prefill"},
            connection_error=RuntimeError("connect failed"),
        )
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)

        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(agent.removal_calls, [agent.connection_calls[0]])
        self.assertEqual(manager._prefill_peers, {})
        self.assertEqual(manager._quarantined_remote_handles, set())

    def test_failed_half_load_cleanup_quarantines_exact_handle(self):
        agent = PeerLifecycleFakeAgent(
            {b"metadata": "prefill"},
            connection_error=RuntimeError("connect failed"),
            removal_error=RuntimeError("remove failed"),
        )
        manager = self._manager(agent)
        route = self._route(agent_name="prefill", metadata=b"metadata", tp_rank=0)

        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            manager._load_prefill_peer("prefill:8998", route)

        self.assertEqual(
            manager._quarantined_remote_handles,
            {agent.connection_calls[0]},
        )


class TestNixlDecoderPeerLifecycle(CustomTestCase):
    @staticmethod
    def _registration(agent_name, metadata, generation, digest):
        return SimpleNamespace(
            agent_name=agent_name,
            agent_metadata=metadata,
            decode_tp_size=1,
            decode_tp_rank=0,
            process_generation=generation,
            registration_digest=digest,
            packed_advertisement=None,
            packed_transfer_protocol=None,
            prepared_grant_protocol=None,
            remote_handle=None,
        )

    @staticmethod
    def _manager(agent):
        manager = object.__new__(NixlKVManager)
        manager.agent = agent
        manager.decode_kv_args_table = {}
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.attn_tp_size = 1
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager._prefill_peers = {}
        manager._prefill_peer_keys_by_addr = defaultdict(set)
        manager._prefill_peers_by_agent_name = {}
        manager._prefill_peers_by_handle = {}
        manager._prefill_peer_lock = threading.RLock()
        manager._quarantined_remote_handles = set()
        manager._prepare_payload_xfer = MagicMock()
        return manager

    def test_prefill_retains_each_decoder_native_handle(self):
        agent = PeerLifecycleFakeAgent(
            {b"decoder-a": "decode-a", b"decoder-b": "decode-b"}
        )
        manager = self._manager(agent)
        generation_a = "00000000-0000-4000-8000-000000000001"
        generation_b = "00000000-0000-4000-8000-000000000002"

        manager._add_remote_peer(
            self._registration("decode-a", b"decoder-a", generation_a, "digest-a")
        )
        manager._add_remote_peer(
            self._registration("decode-b", b"decoder-b", generation_b, "digest-b")
        )

        first = manager.decode_kv_args_table["decode-a"].remote_handle
        second = manager.decode_kv_args_table["decode-b"].remote_handle
        self.assertIsNot(first, second)
        self.assertEqual(agent.connection_calls, [first, second])

    def test_programmatic_registration_rejects_unbounded_peer_identity(self):
        agent = PeerLifecycleFakeAgent({})
        manager = self._manager(agent)
        generation = "00000000-0000-4000-8000-000000000001"
        invalid_registrations = (
            self._registration(
                "decode-\N{SNOWMAN}",
                b"metadata",
                generation,
                "digest-a",
            ),
            self._registration(
                "a" * (NIXL_AGENT_NAME_MAX_BYTES + 1),
                b"metadata",
                generation,
                "digest-b",
            ),
            self._registration(
                "decode",
                b"x" * (NIXL_AGENT_METADATA_MAX_BYTES + 1),
                generation,
                "digest-c",
            ),
        )

        for registration in invalid_registrations:
            with self.subTest(registration=registration):
                with self.assertRaisesRegex(
                    RuntimeError, "Invalid decoder NIXL peer identity"
                ):
                    manager._add_remote_peer(registration)

        self.assertEqual(agent.add_calls, [])

    def test_duplicate_decoder_registration_must_be_exact(self):
        agent = PeerLifecycleFakeAgent({b"decoder": "decode"})
        manager = self._manager(agent)
        generation = "00000000-0000-4000-8000-000000000001"
        first = self._registration(
            "decode", b"decoder", generation, "registration-digest"
        )
        duplicate = self._registration(
            "decode", b"decoder", generation, "registration-digest"
        )

        manager._add_remote_peer(first)
        manager._add_remote_peer(duplicate)

        self.assertIs(manager.decode_kv_args_table["decode"], first)
        self.assertIs(first.remote_handle, agent.handles[b"decoder"])

    def test_conflicting_decoder_generation_is_rejected(self):
        agent = PeerLifecycleFakeAgent({b"decoder": "decode"})
        manager = self._manager(agent)
        manager._add_remote_peer(
            self._registration(
                "decode",
                b"decoder",
                "00000000-0000-4000-8000-000000000001",
                "registration-digest",
            )
        )
        conflicting = self._registration(
            "decode",
            b"decoder",
            "00000000-0000-4000-8000-000000000002",
            "other-digest",
        )

        with self.assertRaisesRegex(RuntimeError, "stale decoder"):
            manager._add_remote_peer(conflicting)


class TestNixlCapabilityAdmission(CustomTestCase):
    def test_not_ready_retries_the_same_transfer_handle(self):
        manager = object.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(
            transfer=MagicMock(side_effect=["NOT_READY", "NOT_READY", "PROC"])
        )
        handle = object()

        with patch("sglang.srt.disaggregation.nixl.conn.time.sleep") as sleep:
            result = manager._post_transfer_when_ready(handle, "test transfer")

        self.assertIs(result, handle)
        self.assertEqual(manager.agent.transfer.call_count, 3)
        for invocation in manager.agent.transfer.call_args_list:
            self.assertIs(invocation.args[0], handle)
        self.assertEqual(sleep.call_count, 2)

    def test_not_ready_has_a_bounded_monotonic_deadline(self):
        manager = object.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(transfer=MagicMock(return_value="NOT_READY"))

        with patch(
            "sglang.srt.disaggregation.nixl.conn.time.monotonic",
            side_effect=[10.0, 15.0],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "test transfer capability admission timed out after 5.000s",
            ):
                manager._post_transfer_when_ready(object(), "test transfer")

    def test_terminal_post_never_retries_static_capability_failure(self):
        """A terminal cohort contradiction fails on its first native result."""

        manager = object.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(transfer=MagicMock(return_value="NOT_READY"))
        handle = object()

        with patch("sglang.srt.disaggregation.nixl.conn.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "one-shot post"):
                manager._post_terminal_transfer_once(handle, "terminal transfer")

        manager.agent.transfer.assert_called_once_with(handle)
        sleep.assert_not_called()

    def test_terminal_post_accepts_only_native_terminal_states(self):
        """Return the exact handle after a successful one-shot post."""

        manager = object.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(transfer=MagicMock(return_value="PROC"))
        handle = object()

        self.assertIs(
            manager._post_terminal_transfer_once(handle, "terminal transfer"),
            handle,
        )
        manager.agent.transfer.assert_called_once_with(handle)


class TestNixlTransferStatusCompletion(CustomTestCase):
    def test_zero_kv_aux_only_completion(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_source_writers_expected = 1
        status.expected_kvs_per_source[0] = 0

        self.assertTrue(status.is_done())

    def test_multi_pp_requires_each_rank_expected_chunks(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_source_writers_expected = 2
        status.expected_kvs_per_source[0] = 1
        status.received_kvs_per_source[0].add(0)

        self.assertFalse(status.is_done())

        status.expected_kvs_per_source[1] = 2
        status.received_kvs_per_source[1].update({0, 1})
        self.assertTrue(status.is_done())

    def test_state_completion_waits_for_every_writer_component_pair(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_source_writers_expected = 2
        status.expected_kvs_per_source[2] = 0
        status.expected_kvs_per_source[5] = 0
        status.expected_state_indices.update({0, 3})

        self.assertFalse(status.is_done())

        status.received_state_components.update({(2, 0), (2, 3), (5, 0)})
        self.assertFalse(status.is_done())

        status.received_state_components.add((5, 3))
        self.assertTrue(status.is_done())

    def test_empty_state_components_do_not_delay_completion(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_source_writers_expected = 1
        status.expected_kvs_per_source[0] = 0

        self.assertTrue(status.is_done())


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))

    def test_sender_preserves_producer_event_in_manager_submission(self) -> None:
        """The sender cannot replace the exact scheduler-owned event."""

        producer_event = object()
        manager = SimpleNamespace(
            enable_all_cp_ranks_for_transfer=False,
            server_args=SimpleNamespace(enable_dsa_cache_layer_split=False),
            is_dummy_cp_rank=False,
            add_transfer_request=MagicMock(),
        )
        sender = object.__new__(NixlKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 41
        sender.curr_idx = 0
        sender.num_kv_indices = 1
        sender.chunk_id = 0
        sender.aux_index = 7
        sender._send_failed = False
        sender._transfer_start_time = None
        sender._transfer_num_kv_indices = 0
        sender._transfer_num_state_indices = 0

        sender.send(
            np.array([3], dtype=np.int32),
            producer_event=producer_event,
        )

        manager.add_transfer_request.assert_called_once()
        self.assertIs(
            manager.add_transfer_request.call_args.args[-1],
            producer_event,
        )


class TestNixlProducerEventQueue(CustomTestCase):
    def _make_manager(self) -> NixlKVManager:
        """Build a packed source manager without starting worker threads.

        :returns: Manager double with one observable transfer queue.
        """

        manager = object.__new__(NixlKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager._packed_source_route = MagicMock(return_value=(object(), object()))
        manager.enable_staging = False
        manager.transfer_queues = [SimpleNamespace(put=MagicMock())]
        return manager

    def test_final_chunk_queue_owns_the_exact_producer_event(self) -> None:
        """The in-process worker queue retains immutable event identity."""

        manager = self._make_manager()
        producer_event = object()

        manager.add_transfer_request(
            41,
            np.array([3], dtype=np.int32),
            slice(0, 1),
            True,
            0,
            7,
            None,
            producer_event,
        )

        queued = manager.transfer_queues[0].put.call_args.args[0]
        self.assertIs(queued.producer_event, producer_event)

    def test_final_packed_chunk_without_event_fails_before_enqueue(self) -> None:
        """Packed transfer cannot fall back to sampling a mutable stream."""

        manager = self._make_manager()

        with self.assertRaisesRegex(RuntimeError, "no producer event"):
            manager.add_transfer_request(
                41,
                np.array([3], dtype=np.int32),
                slice(0, 1),
                True,
                0,
                7,
                None,
                None,
            )

        manager.transfer_queues[0].put.assert_not_called()

    def test_nonpacked_final_chunk_remains_valid_without_event(self) -> None:
        """The new dependency is required only by the packed source actor."""

        manager = self._make_manager()
        manager._packed_source_route.return_value = None

        manager.add_transfer_request(
            41,
            np.array([3], dtype=np.int32),
            slice(0, 1),
            True,
            0,
            7,
            None,
            None,
        )

        queued = manager.transfer_queues[0].put.call_args.args[0]
        self.assertIsNone(queued.producer_event)


class TestNixlAbortHandling(CustomTestCase):
    def _make_manager(self, request_status=None):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status or {})
        mgr._connect = MagicMock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        return mgr

    def test_given_known_incomplete_room_when_abort_arrives_then_room_fails_without_ack(
        self,
    ):
        mgr = self._make_manager({11: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"11", b"127.0.0.1", b"5555"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)
        self.assertEqual(
            mgr.failure_records[11],
            "Aborted by decode-side abort notification.",
        )
        mgr._connect.assert_not_called()

    def test_given_successful_room_when_abort_arrives_then_status_is_preserved(self):
        mgr = self._make_manager({12: KVPoll.Success})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"12", b"127.0.0.1", b"5556"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[12], KVPoll.Success)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_unknown_room_when_abort_arrives_then_status_remains_absent(self):
        mgr = self._make_manager()

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"14", b"127.0.0.1", b"5557"]
        )

        self.assertTrue(handled)
        self.assertNotIn(14, mgr.request_status)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_malformed_abort_when_handled_then_no_exception_or_ack(self):
        mgr = self._make_manager({13: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [b"ABORT", b"invalid-room", b"127.0.0.1", b"5558"]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[13], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()


class TestNixlUpdateStatus(CustomTestCase):
    def _make_manager(self, request_status):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status)
        return mgr

    def test_given_failed_room_when_status_is_promoted_then_failed_is_preserved(self):
        for status in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(status=status):
                mgr = self._make_manager({17: KVPoll.Failed})

                mgr.update_status(17, status)

                self.assertEqual(mgr.request_status[17], KVPoll.Failed)

    def test_given_missing_room_when_failed_update_arrives_then_room_is_not_resurrected(
        self,
    ):
        mgr = self._make_manager({})

        mgr.update_status(18, KVPoll.Failed)

        self.assertNotIn(18, mgr.request_status)


class TestNixlMetadataSnapshot(CustomTestCase):
    @staticmethod
    def _common_manager_init(
        manager: NixlKVManager,
        args: SimpleNamespace,
        disaggregation_mode: DisaggregationMode,
        server_args: SimpleNamespace,
        is_mla_backend: bool,
        defer_prefill_bootstrap_registration: bool,
    ) -> None:
        manager.kv_args = args
        manager.server_args = server_args
        manager.disaggregation_mode = disaggregation_mode
        manager.attn_tp_size = server_args.tp_size
        manager.attn_tp_rank = args.engine_rank
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = args.pp_rank

    @staticmethod
    def _register_payload_memory(manager: NixlKVManager) -> None:
        manager.agent.register_memory([(0x1000, 4096, 0, "")], "VRAM")

    @staticmethod
    def _register_decode_staging_memory(manager: NixlKVManager) -> None:
        staging_tensor = object()
        manager._staging_ctx = SimpleNamespace(
            allocator=SimpleNamespace(
                buffer=SimpleNamespace(buffer=staging_tensor),
            )
        )
        manager._decode_staging_registration = manager._register_staging_memory(
            0x2000,
            4096,
            0,
        )

    def test_decode_staging_registration_precedes_agent_metadata_snapshot(self) -> None:
        agent = MetadataSnapshotFakeAgent()
        args = SimpleNamespace(
            engine_rank=0,
            kv_data_lens=[4096],
            kv_data_mem_kinds=["VRAM"],
            kv_data_ptrs=[0x1000],
            kv_item_lens=[64],
            pp_rank=0,
        )
        server_args = SimpleNamespace(
            tp_size=1,
            pd_terminal_deployment_cohort=None,
            pd_terminal_local_membership=None,
            pd_terminal_startup_timeout_seconds=None,
        )

        with (
            patch.object(
                CommonKVManager,
                "__init__",
                new=self._common_manager_init,
            ),
            patch.object(
                NixlKVManager,
                "register_buffer_to_engine",
                new=self._register_payload_memory,
            ),
            patch.object(
                NixlKVManager,
                "_init_staging_decode_ctx",
                new=self._register_decode_staging_memory,
            ),
            patch.object(NixlKVManager, "_start_decode_staging_thread"),
            patch.object(NixlKVManager, "_start_heartbeat_checker_thread"),
            patch.object(
                envs.SGLANG_DISAGGREGATION_NIXL_BACKEND,
                "get",
                return_value="UCX",
            ),
            patch.object(
                envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS,
                "get",
                return_value="{}",
            ),
            patch.object(
                envs.SGLANG_DISAGG_STAGING_BUFFER,
                "get",
                return_value=True,
            ),
            patch("sglang.srt.disaggregation.nixl.conn.PackedNixlDecodeController"),
            patch("nixl._api.nixl_agent", return_value=agent),
            patch("nixl._api.nixl_agent_config", return_value=object()),
        ):
            manager = NixlKVManager(
                args,
                DisaggregationMode.DECODE,
                server_args,
            )

        self.assertEqual(manager.agent_metadata, b"4096,8192")
        self.assertEqual(agent.metadata_snapshots, [b"4096,8192"])

    def test_terminal_dflash_rows_register_before_agent_metadata_snapshot(self) -> None:
        """Advertise manager-owned DFlash VRAM before terminal identity freezes."""

        agent = MetadataSnapshotFakeAgent()
        args = SimpleNamespace(
            engine_rank=0,
            gpu_id=0,
            kv_data_lens=[4096],
            kv_data_mem_kinds=["VRAM"],
            kv_data_ptrs=[0x1000],
            kv_item_lens=[64],
            pp_rank=0,
            terminal_request_capacity=4,
        )
        server_args = SimpleNamespace(
            tp_size=1,
            speculative_algorithm="DFLASH",
            pd_terminal_deployment_cohort=object(),
            pd_terminal_local_membership=object(),
            pd_terminal_startup_timeout_seconds=60.0,
        )
        boundary_pool = object()

        def register_boundary_pool(
            pool_agent: MetadataSnapshotFakeAgent,
            *,
            row_capacity: int,
            device: object,
        ) -> object:
            """Model the real pool's NIXL registration side effect.

            :param pool_agent: Exact process-local fake NIXL agent.
            :param row_capacity: Configured in-flight request bound.
            :param device: Explicit indexed CUDA device.
            :returns: Stable fake process-lifetime pool.
            """

            self.assertIs(pool_agent, agent)
            self.assertEqual(row_capacity, 4)
            self.assertEqual(str(device), "cuda:0")
            pool_agent.register_memory([(0x3000, row_capacity * 8, 0, "")], "VRAM")
            return boundary_pool

        with (
            patch.object(
                CommonKVManager,
                "__init__",
                new=self._common_manager_init,
            ),
            patch.object(
                NixlKVManager,
                "register_buffer_to_engine",
                new=self._register_payload_memory,
            ),
            patch.object(
                NixlKVManager,
                "_init_staging_decode_ctx",
                new=self._register_decode_staging_memory,
            ),
            patch.object(
                NixlKVManager,
                "_join_terminal_startup_cohort",
                return_value=None,
            ),
            patch.object(NixlKVManager, "_start_decode_runtime_workers"),
            patch.object(
                envs.SGLANG_DISAGGREGATION_NIXL_BACKEND,
                "get",
                return_value="UCX",
            ),
            patch.object(
                envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS,
                "get",
                return_value="{}",
            ),
            patch.object(
                envs.SGLANG_DISAGG_STAGING_BUFFER,
                "get",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.nixl.conn.DFlashBoundaryDeviceRowPool",
                side_effect=register_boundary_pool,
            ),
            patch(
                "sglang.srt.disaggregation.nixl.conn.PackedNixlDecodeController"
            ) as controller_constructor,
            patch("nixl._api.nixl_agent", return_value=agent),
            patch("nixl._api.nixl_agent_config", return_value=object()),
        ):
            manager = NixlKVManager(
                args,
                DisaggregationMode.DECODE,
                server_args,
            )

        self.assertEqual(manager.agent_metadata, b"4096,12288,8192")
        self.assertEqual(agent.metadata_snapshots, [b"4096,12288,8192"])
        self.assertIs(manager.terminal_dflash_boundary_pool(), boundary_pool)
        controller_constructor.assert_called_once_with(
            manager,
            manager._staging_ctx.allocator.buffer.buffer,
            manager._decode_staging_registration,
            boundary_pool,
        )

    def test_supported_prefill_widths_initialize_live_packed_runtime(self) -> None:
        """Select the real packed source actor for TP1, TP2, and TP4."""

        artifacts = object()
        visibility_policy = object()
        for source_tp_size in (1, 2, 4):
            with self.subTest(source_tp_size=source_tp_size):
                agent = MetadataSnapshotFakeAgent()
                args = SimpleNamespace(
                    engine_rank=0,
                    kv_data_lens=[4096],
                    kv_data_mem_kinds=["VRAM"],
                    kv_data_ptrs=[0x1000],
                    kv_item_lens=[64],
                    pp_rank=0,
                )
                server_args = SimpleNamespace(
                    tp_size=source_tp_size,
                    pd_terminal_deployment_cohort=None,
                    pd_terminal_local_membership=None,
                    pd_terminal_startup_timeout_seconds=None,
                )
                runtime = object()
                with (
                    patch.object(
                        CommonKVManager,
                        "__init__",
                        new=self._common_manager_init,
                    ),
                    patch.object(NixlKVManager, "register_buffer_to_engine"),
                    patch.object(NixlKVManager, "_init_staging_prefill_ctx"),
                    patch.object(NixlKVManager, "_init_staging_buffers"),
                    patch.object(NixlKVManager, "register_to_bootstrap"),
                    patch.object(NixlKVManager, "_start_bootstrap_thread"),
                    patch.object(
                        envs.SGLANG_DISAGGREGATION_NIXL_BACKEND,
                        "get",
                        return_value="UCX",
                    ),
                    patch.object(
                        envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS,
                        "get",
                        return_value="{}",
                    ),
                    patch.object(
                        envs.SGLANG_DISAGG_STAGING_BUFFER,
                        "get",
                        return_value=True,
                    ),
                    patch.object(
                        envs.SGLANG_DISAGGREGATION_QUEUE_SIZE,
                        "get",
                        return_value=0,
                    ),
                    patch(
                        "sglang.srt.disaggregation.nixl.conn."
                        "load_exact_nixl_runtime_artifacts",
                        return_value=artifacts,
                    ),
                    patch(
                        "sglang.srt.disaggregation.nixl.conn."
                        "build_same_host_visibility_policy",
                        return_value=visibility_policy,
                    ),
                    patch(
                        "sglang.srt.disaggregation.nixl.conn.PackedPrefillRuntime",
                        return_value=runtime,
                    ) as runtime_constructor,
                    patch("nixl._api.nixl_agent", return_value=agent),
                    patch("nixl._api.nixl_agent_config", return_value=object()),
                ):
                    manager = NixlKVManager(
                        args,
                        DisaggregationMode.PREFILL,
                        server_args,
                    )

                runtime_constructor.assert_called_once_with(
                    manager,
                    artifacts,
                    visibility_policy,
                )
                self.assertIs(manager._packed_prefill_runtime, runtime)
                self.assertEqual(
                    manager.kv_transfer_protocol(),
                    PACKED_KV_TRANSFER_PROTOCOL,
                )
                self.assertEqual(
                    manager.prepared_grant_protocol(),
                    PACKED_PREPARED_GRANT_PROTOCOL,
                )


class TestNixlTransferWorker(CustomTestCase):
    def _make_manager(self, room):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.transfer_infos = {
            room: {
                "agent": TransferInfo(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=5555,
                    agent_name="agent",
                    dst_kv_indices=np.array([2], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(
                decode_tp_size=1,
                dst_kv_ptrs=[0],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging=None,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                remote_handle=FakeRemoteHandle("agent"),
            )
        }
        mgr.req_to_decode_prefix_len = {room: 4}
        mgr.enable_staging = False
        mgr._staging_ctx = None
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = 0
        mgr.attn_cp_rank = 0
        mgr.pp_rank = 0
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[0])
        mgr.exceptions = {}
        mgr._direct_kv_transfer_lock = threading.Lock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._packed_prefill_runtime = None

        def check_xfer_state(_handle):
            mgr.update_status(room, KVPoll.Failed)
            return "DONE"

        mgr.agent = SimpleNamespace(
            check_xfer_state=check_xfer_state,
            release_xfer_handle=MagicMock(),
            send_notif=MagicMock(),
        )
        return mgr

    def _make_chunk(self, room, prefill_kv_indices, is_last_chunk):
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array(prefill_kv_indices, dtype=np.int32),
            index_slice=slice(0, len(prefill_kv_indices)),
            is_last_chunk=is_last_chunk,
            chunk_id=0,
            prefill_aux_index=0 if is_last_chunk else None,
            state_indices=None,
        )

    def _run_worker_once(self, mgr, chunk):
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

    def test_packed_worker_passes_final_chunk_event_to_source_actor(self) -> None:
        """Worker dispatch preserves the event owned by the final work unit."""

        room = 28
        manager = self._make_manager(room)
        transfer_info = object()
        registration = object()
        manager._packed_source_route = MagicMock(
            return_value=(transfer_info, registration)
        )
        manager._execute_packed_source_request = MagicMock()
        producer_event = object()
        chunk = self._make_chunk(room, [4, 5], is_last_chunk=True)
        chunk.producer_event = producer_event

        self._run_worker_once(manager, chunk)

        submission = manager._execute_packed_source_request.call_args.kwargs
        self.assertIs(submission["transfer_info"], transfer_info)
        self.assertIs(submission["registration"], registration)
        self.assertIs(submission["producer_event"], producer_event)
        self.assertNotIn(room, manager.transfer_infos)

    def test_transfer_handles_are_released_once_after_all_complete(self) -> None:
        """Completed outer transfer handles remain valid until every handle is done."""

        room = 29
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        mgr.send_aux = MagicMock(return_value="aux_handle")
        events: list[str] = []
        states = {
            "kv_handle": ["PROC", "DONE"],
            "aux_handle": ["DONE", "DONE"],
        }

        def check_xfer_state(handle: str) -> str:
            state = states[handle].pop(0)
            events.append(f"check:{handle}:{state}")
            return state

        def release_xfer_handle(handle: str) -> None:
            events.append(f"release:{handle}")

        mgr.agent.check_xfer_state = check_xfer_state
        mgr.agent.release_xfer_handle = release_xfer_handle
        chunk = self._make_chunk(room, [1], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(
            events,
            [
                "check:kv_handle:PROC",
                "check:aux_handle:DONE",
                "check:kv_handle:DONE",
                "check:aux_handle:DONE",
                "release:kv_handle",
                "release:aux_handle",
            ],
        )

    def test_last_asymmetric_kv_releases_before_state_and_auxiliary_post(self) -> None:
        """Direct asymmetric KV completes before the final state transfer phase."""

        room = 30
        mgr = self._make_manager(room)
        mgr.attn_tp_size = 2
        dst_info = mgr.decode_kv_args_table["agent"]
        dst_info.decode_tp_rank = 0
        dst_info.dst_state_data_ptrs = [0]
        dst_info.dst_state_item_lens = [1]
        dst_info.dst_state_dim_per_tensor = [1]
        dst_info.dst_state_layer_ids = [0]
        events: list[str] = []

        def send_kvcache_slice(*_args: object) -> str:
            events.append("post:kv")
            return "kv_handle"

        def send_state(*_args: object, **_kwargs: object) -> list[str]:
            events.append("post:state")
            return ["state_handle"]

        def send_aux(*_args: object) -> str:
            events.append("post:aux")
            return "aux_handle"

        def check_xfer_state(handle: str) -> str:
            events.append(f"check:{handle}:DONE")
            return "DONE"

        def release_xfer_handle(handle: str) -> None:
            events.append(f"release:{handle}")

        mgr.send_kvcache_slice = send_kvcache_slice
        mgr.maybe_send_extra = send_state
        mgr.send_aux = send_aux
        mgr.agent.check_xfer_state = check_xfer_state
        mgr.agent.release_xfer_handle = release_xfer_handle
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[1],
        )

        self._run_worker_once(mgr, chunk)

        self.assertEqual(
            events,
            [
                "post:kv",
                "check:kv_handle:DONE",
                "release:kv_handle",
                "post:state",
                "post:aux",
                "check:state_handle:DONE",
                "check:aux_handle:DONE",
                "release:state_handle",
                "release:aux_handle",
            ],
        )

    def test_direct_asymmetric_transaction_lock_spans_final_release(self) -> None:
        """A rank holds its direct lock through KV, state, and auxiliary release."""

        room = 31
        mgr = self._make_manager(room)
        mgr.attn_tp_size = 2
        dst_info = mgr.decode_kv_args_table["agent"]
        dst_info.decode_tp_rank = 0
        dst_info.dst_state_data_ptrs = [0]
        dst_info.dst_state_item_lens = [1]
        dst_info.dst_state_dim_per_tensor = [1]
        dst_info.dst_state_layer_ids = [0]
        events: list[str] = []

        class RecordingLock:
            """Record the transaction lock lifetime used by the worker."""

            def acquire(self) -> bool:
                events.append("lock:acquire")
                return True

            def release(self) -> None:
                events.append("lock:release")

        mgr._direct_kv_transfer_lock = RecordingLock()

        def send_kvcache_slice(*_args: object) -> str:
            events.append("post:kv")
            return "kv_handle"

        def send_state(*_args: object, **_kwargs: object) -> list[str]:
            events.append("post:state")
            return ["state_handle"]

        def send_aux(*_args: object) -> str:
            events.append("post:aux")
            return "aux_handle"

        def check_xfer_state(handle: str) -> str:
            events.append(f"check:{handle}:DONE")
            return "DONE"

        def release_xfer_handle(handle: str) -> None:
            events.append(f"release:{handle}")

        mgr.send_kvcache_slice = send_kvcache_slice
        mgr.maybe_send_extra = send_state
        mgr.send_aux = send_aux
        mgr.agent = SimpleNamespace(
            check_xfer_state=check_xfer_state,
            release_xfer_handle=release_xfer_handle,
            send_notif=MagicMock(),
        )
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[1],
        )

        self._run_worker_once(mgr, chunk)

        self.assertEqual(
            events,
            [
                "lock:acquire",
                "post:kv",
                "check:kv_handle:DONE",
                "release:kv_handle",
                "post:state",
                "post:aux",
                "check:state_handle:DONE",
                "check:aux_handle:DONE",
                "release:state_handle",
                "release:aux_handle",
                "lock:release",
            ],
        )

    def test_prepped_slice_transfer_uses_direct_kv_cohort_bound(self) -> None:
        """TP2 and TP4 direct KV parts stay within the transport cohort."""

        expected_descriptor_limits = {2: 8 * 1024, 4: 4 * 1024}
        for source_tp_size in (2, 4):
            with self.subTest(source_tp_size=source_tp_size):
                mgr = object.__new__(NixlKVManager)
                mgr.attn_tp_size = source_tp_size
                mgr.kv_args = SimpleNamespace(page_size=1)
                mgr.prep_handle_slice_src = ("src", 1, 1, 1)
                mgr.prep_handles_slice_dst = {"peer": ("dst", 1, 0)}
                events: list[str] = []
                calls: list[tuple[int, int, bytes]] = []

                def make_prepped_xfer(
                    _operation: str,
                    _src_handle: str,
                    src_indices: np.ndarray,
                    _dst_handle: str,
                    dst_indices: np.ndarray,
                    notification: bytes,
                ) -> str:
                    handle = f"handle-{len(calls)}"
                    calls.append((src_indices.size, dst_indices.size, notification))
                    events.append(f"make:{handle}")
                    return handle

                def transfer(handle: str) -> str:
                    events.append(f"post:{handle}")
                    return "PROC"

                def check_xfer_state(handle: str) -> str:
                    events.append(f"check:{handle}:DONE")
                    return "DONE"

                def release_xfer_handle(handle: str) -> None:
                    events.append(f"release:{handle}")

                mgr.agent = SimpleNamespace(
                    make_prepped_xfer=make_prepped_xfer,
                    transfer=transfer,
                    check_xfer_state=check_xfer_state,
                    release_xfer_handle=release_xfer_handle,
                )
                descriptor_limit = min(
                    NIXL_RMA_MAX_DESCRIPTORS,
                    NIXL_DIRECT_KV_MAX_COHORT_DESCRIPTORS // source_tp_size,
                )
                self.assertEqual(
                    descriptor_limit,
                    expected_descriptor_limits[source_tp_size],
                )
                indices = np.arange(descriptor_limit + 3, dtype=np.int32)

                final_handle = mgr.send_kvcache_slice(
                    "peer",
                    indices,
                    indices,
                    "31_kv_0_1_0",
                )

                self.assertEqual(
                    calls,
                    [
                        (descriptor_limit, descriptor_limit, b""),
                        (3, 3, b"31_kv_0_1_0"),
                    ],
                )
                self.assertEqual(
                    events,
                    [
                        "make:handle-0",
                        "post:handle-0",
                        "check:handle-0:DONE",
                        "release:handle-0",
                        "make:handle-1",
                        "post:handle-1",
                    ],
                )
                self.assertEqual(final_handle, "handle-1")

    def test_given_last_chunk_aborts_mid_transfer_when_worker_finishes_then_failed_status_is_preserved(
        self,
    ):
        room = 21
        mgr = self._make_manager(room)
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        mgr.send_aux.assert_called_once()

    def test_tp2_canonical_writer_sends_kv_and_auxiliary_payload(self):
        room = 25
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [1], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        mgr.send_kvcache.assert_called_once()
        mgr.send_aux.assert_called_once_with("agent", 0, [0], 0, "25_aux")
        mgr.agent.send_notif.assert_not_called()

    def test_tp2_noncanonical_writer_sends_kv_without_auxiliary_payload(self):
        room = 26
        mgr = self._make_manager(room)
        mgr.attn_tp_rank = 1
        mgr.transfer_source_rank = 1
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [1], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        mgr.send_kvcache.assert_called_once()
        mgr.send_aux.assert_not_called()
        mgr.agent.send_notif.assert_not_called()

    def test_tp2_noncanonical_zero_kv_writer_sends_notification_only_marker(self):
        room = 27
        mgr = self._make_manager(room)
        mgr.attn_tp_rank = 1
        mgr.transfer_source_rank = 1
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        mgr.send_aux.assert_not_called()
        mgr.agent.send_notif.assert_called_once_with(
            mgr.decode_kv_args_table["agent"].remote_handle,
            b"27_aux_nokv_1",
        )

    def test_auxiliary_writer_requires_tp_cp_and_pp_leadership(self):
        room = 28
        for field in ("attn_tp_rank", "attn_cp_rank", "pp_rank"):
            with self.subTest(field=field):
                mgr = self._make_manager(room)
                setattr(mgr, field, 1)

                self.assertFalse(mgr._is_canonical_aux_writer())

    def test_given_non_last_chunk_aborts_mid_transfer_when_worker_finishes_then_failed_status_is_preserved(
        self,
    ):
        room = 22
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertIn(room, mgr.transfer_infos)
        self.assertIn(room, mgr.req_to_decode_prefix_len)
        mgr.send_kvcache.assert_called_once()

    def test_worker_error_notifies_decode_peer_and_marks_sender_failed(self):
        room = 23
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(side_effect=RuntimeError("invalid geometry"))
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertEqual(str(mgr.exceptions[room]), "invalid geometry")
        mgr.agent.send_notif.assert_called_once_with(
            mgr.decode_kv_args_table["agent"].remote_handle,
            b"23_failure_0",
        )

    def test_notification_error_does_not_mask_original_transfer_failure(self):
        room = 24
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(side_effect=RuntimeError("invalid geometry"))
        mgr.agent.send_notif.side_effect = ValueError("notification failed")
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertEqual(str(mgr.exceptions[room]), "invalid geometry")
        self.assertEqual(mgr.failure_records[room], "invalid geometry")


class TestNixlNotifications(CustomTestCase):
    def _make_manager(self, messages, required=None, source_rank=0, peer_handle=None):
        mgr = object.__new__(NixlKVManager)
        agent = NotificationFakeAgent(messages, peer_handle=peer_handle)
        mgr.agent = agent
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = required or {}
        rooms = {int(message.split("_", 1)[0]) for message in messages}
        mgr.request_status = {room: KVPoll.WaitingForInput for room in rooms}
        process_generation = "00000000-0000-4000-8000-000000000000"
        mgr._prefill_peer_lock = threading.RLock()
        mgr._prefill_peers_by_handle = {
            agent.peer_handle: SimpleNamespace(process_generation=process_generation)
        }
        mgr._quarantined_remote_handles = set()
        for room in rooms:
            status = mgr.transfer_statuses[room]
            status.expected_source_ranks[agent.peer_handle] = source_rank
            status.expected_source_generations[agent.peer_handle] = process_generation
            status.canonical_aux_source = agent.peer_handle
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        return mgr

    def test_kv_last_notification_sets_expected_count(self):
        mgr = self._make_manager(["5_kv_2_1_0"])

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_source[0], {2})
        self.assertEqual(status.expected_kvs_per_source[0], 3)
        self.assertEqual(status.num_source_writers_expected, 1)

    def test_staging_notification_preserves_agent_name_with_underscores(self):
        mgr = self._make_manager(["5_stg_0_1_0_2_4_8_part_0_1_agent_with_underscores"])
        mgr.agent.name = "agent_with_underscores"
        calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(5, 2, 4, 8, "agent_with_underscores")])
        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_source[0], {0})
        self.assertEqual(status.expected_kvs_per_source[0], 1)

    def test_staging_parts_scatter_once_after_every_part_arrives(self):
        mgr = self._make_manager(
            [
                "28_stg_3_1_0_2_4_8_part_1_2_decode-agent",
                "28_stg_3_1_0_2_4_8_part_0_2_decode-agent",
            ]
        )
        calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(28, 2, 4, 8, "decode-agent")])
        status = mgr.transfer_statuses[28]
        self.assertEqual(status.received_kvs_per_source[0], {3})
        self.assertEqual(status.expected_kvs_per_source[0], 4)
        self.assertEqual(status.staging_parts_per_source, {})
        self.assertEqual(status.completed_staging_chunks, {(0, 3)})

    def test_duplicate_staging_part_fails_without_scatter(self):
        message = "29_stg_0_1_0_2_4_8_part_0_2_decode-agent"
        mgr = self._make_manager([message, message])
        mgr._handle_staging_chunk_arrived = MagicMock()

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[29], KVPoll.Failed)
        self.assertIn("duplicate staging part", mgr.failure_records[29])
        mgr._handle_staging_chunk_arrived.assert_not_called()

    def test_staging_part_geometry_mismatch_fails_without_scatter(self):
        mgr = self._make_manager(
            [
                "30_stg_0_1_0_2_4_8_part_0_2_decode-agent",
                "30_stg_0_1_0_2_5_8_part_1_2_decode-agent",
            ]
        )
        mgr._handle_staging_chunk_arrived = MagicMock()

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[30], KVPoll.Failed)
        self.assertIn("immutable chunk metadata", mgr.failure_records[30])
        mgr._handle_staging_chunk_arrived.assert_not_called()

    def test_aux_nokv_marks_zero_expected_chunks_for_pp_rank(self):
        mgr = self._make_manager(["6_aux_nokv_3"], required={6: 4}, source_rank=3)

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[6]
        self.assertTrue(status.received_aux)
        self.assertEqual(status.expected_kvs_per_source[3], 0)
        self.assertEqual(status.num_source_writers_expected, 4)

    def test_state_notification_records_writer_and_component(self):
        mgr = self._make_manager(["7_state_2_3"], source_rank=2)

        mgr.update_transfer_status()

        self.assertEqual(
            mgr.transfer_statuses[7].received_state_components,
            {(2, 3)},
        )

    def test_failure_notification_marks_known_room_failed(self):
        mgr = self._make_manager(["9_failure_3"], source_rank=3)
        mgr.request_status[9] = KVPoll.Transferring

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[9], KVPoll.Failed)
        self.assertEqual(
            mgr.failure_records[9],
            "Prefill source rank 3 reported transfer failure",
        )

    def test_aux_nokv_allows_full_hit_completion(self):
        mgr = self._make_manager(["8_aux_nokv_0"], required={8: 1})

        mgr.update_transfer_status()

        self.assertTrue(mgr.transfer_statuses[8].is_done())

    def test_fresh_wrapper_for_same_native_handle_retains_aux_authority(self):
        retained_handle = PointerEqualRemoteHandle(
            "prefill",
            identity=7,
            generation=3,
        )
        delivered_handle = PointerEqualRemoteHandle(
            "prefill",
            identity=7,
            generation=3,
        )
        mgr = self._make_manager(["27_aux"], peer_handle=retained_handle)
        mgr.agent.get_new_notifs = lambda: {delivered_handle: [b"27_aux"]}

        mgr.update_transfer_status()

        self.assertTrue(mgr.transfer_statuses[27].received_aux)
        self.assertNotEqual(mgr.request_status[27], KVPoll.Failed)

    def test_wrong_native_handle_cannot_fail_unbound_room(self):
        expected_handle = FakeRemoteHandle("expected")
        wrong_handle = FakeRemoteHandle("wrong")
        mgr = self._make_manager(["10_kv_0_1_0"], peer_handle=wrong_handle)
        status = mgr.transfer_statuses[10]
        status.expected_source_ranks.clear()
        status.expected_source_generations.clear()
        status.expected_source_ranks[expected_handle] = 0
        status.expected_source_generations[expected_handle] = (
            "00000000-0000-4000-8000-000000000000"
        )

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[10], KVPoll.WaitingForInput)
        self.assertNotIn(10, mgr.failure_records)

    def test_spoofed_source_rank_fails_room(self):
        mgr = self._make_manager(["11_kv_0_1_3"], source_rank=2)

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[11], KVPoll.Failed)
        self.assertIn("source-rank mismatch", mgr.failure_records[11])

    def test_spoofed_staging_decoder_name_fails_room(self):
        mgr = self._make_manager(["12_stg_0_1_0_2_4_8_part_0_1_spoofed_decode"])
        mgr._handle_staging_chunk_arrived = MagicMock()

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[12], KVPoll.Failed)
        mgr._handle_staging_chunk_arrived.assert_not_called()

    def test_malformed_known_room_notification_fails_without_escaping(self):
        mgr = self._make_manager(["13_kv_not-an-int_1_0"])

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[13], KVPoll.Failed)
        self.assertIn("Rejected NIXL notification", mgr.failure_records[13])

    def test_non_ascii_known_room_notification_fails_without_escaping(self):
        mgr = self._make_manager(["17_kv_0_1_0"])
        peer_handle = mgr.agent.peer_handle
        mgr.agent.get_new_notifs = lambda: {peer_handle: [b"17_kv_0_1_0_\xff"]}

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[17], KVPoll.Failed)
        self.assertIn("payload is not ASCII", mgr.failure_records[17])

    def test_unparseable_room_fails_every_room_bound_to_peer(self):
        mgr = self._make_manager(["18_kv_0_1_0", "19_kv_0_1_0"])
        peer_handle = mgr.agent.peer_handle
        mgr.agent.get_new_notifs = lambda: {peer_handle: [b"not-a-room_kv_0_1_0"]}

        mgr.update_transfer_status()

        for room in (18, 19):
            with self.subTest(room=room):
                self.assertEqual(mgr.request_status[room], KVPoll.Failed)
                self.assertIn("room is not parseable", mgr.failure_records[room])

    def test_non_ascii_unparseable_room_fails_every_room_bound_to_peer(self):
        mgr = self._make_manager(["20_kv_0_1_0", "21_kv_0_1_0"])
        peer_handle = mgr.agent.peer_handle
        mgr.agent.get_new_notifs = lambda: {peer_handle: [b"\xff_kv_0_1_0"]}

        mgr.update_transfer_status()

        for room in (20, 21):
            with self.subTest(room=room):
                self.assertEqual(mgr.request_status[room], KVPoll.Failed)
                self.assertIn("room is not parseable", mgr.failure_records[room])

    def test_unparseable_room_enforces_bound_process_generation(self):
        mgr = self._make_manager(["26_kv_0_1_0"])
        peer_handle = mgr.agent.peer_handle
        mgr.transfer_statuses[26].expected_source_generations[
            peer_handle
        ] = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        mgr.agent.get_new_notifs = lambda: {peer_handle: [b"not-a-room_kv_0_1_0"]}

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[26], KVPoll.Failed)
        self.assertIn("stale native peer generation", mgr.failure_records[26])

    def test_non_ascii_wrong_handle_cannot_fail_unbound_room(self):
        expected_handle = FakeRemoteHandle("expected")
        wrong_handle = FakeRemoteHandle("wrong")
        mgr = self._make_manager(["22_kv_0_1_0"], peer_handle=wrong_handle)
        status = mgr.transfer_statuses[22]
        status.expected_source_ranks.clear()
        status.expected_source_generations.clear()
        status.expected_source_ranks[expected_handle] = 0
        status.expected_source_generations[expected_handle] = (
            "00000000-0000-4000-8000-000000000000"
        )
        mgr.agent.get_new_notifs = lambda: {wrong_handle: [b"22_kv_0_1_0_\xff"]}

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[22], KVPoll.WaitingForInput)
        self.assertNotIn(22, mgr.failure_records)

    def test_parseable_unknown_room_cannot_fail_another_room(self):
        mgr = self._make_manager(["23_kv_0_1_0"])
        peer_handle = mgr.agent.peer_handle
        mgr.agent.get_new_notifs = lambda: {peer_handle: [b"24_kv_0_1_0"]}

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[23], KVPoll.WaitingForInput)
        self.assertNotIn(23, mgr.failure_records)

    def test_unknown_tag_fails_bound_room(self):
        mgr = self._make_manager(["25_unknown"])

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[25], KVPoll.Failed)
        self.assertIn("unknown notification tag", mgr.failure_records[25])

    def test_stale_process_generation_fails_room(self):
        mgr = self._make_manager(["14_kv_0_1_0"])
        mgr.transfer_statuses[14].expected_source_generations[
            mgr.agent.peer_handle
        ] = "ffffffff-ffff-4fff-8fff-ffffffffffff"

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[14], KVPoll.Failed)
        self.assertIn("stale native peer generation", mgr.failure_records[14])

    def test_noncanonical_aux_writer_fails_room(self):
        mgr = self._make_manager(["15_aux"])
        mgr.transfer_statuses[15].canonical_aux_source = FakeRemoteHandle("canonical")

        mgr.update_transfer_status()

        self.assertEqual(mgr.request_status[15], KVPoll.Failed)
        self.assertIn("noncanonical writer", mgr.failure_records[15])

    def test_noncanonical_no_kv_marker_does_not_claim_aux_ownership(self):
        mgr = self._make_manager(["16_aux_nokv_0"])
        mgr.transfer_statuses[16].canonical_aux_source = FakeRemoteHandle("canonical")

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[16]
        self.assertFalse(status.received_aux)
        self.assertEqual(status.expected_kvs_per_source[0], 0)
        self.assertEqual(mgr.request_status[16], KVPoll.WaitingForInput)


class TestNixlReceiverPoll(CustomTestCase):
    def _make_receiver(self, status=KVPoll.WaitingForInput):
        mgr = MagicMock()
        mgr.waiting_timeout = 5
        mgr.check_status.return_value = status
        mgr.check_transfer_done.return_value = False
        mgr.transfer_statuses = {}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["prefill:8998"].add(11)

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.bootstrap_addr = "prefill:8998"
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.abort_notified = False
        receiver.prefill_peers = []
        return receiver, mgr

    def test_failed_cohort_load_reuses_only_fully_validated_peers(self):
        agent = PeerLifecycleFakeAgent(
            {
                b"metadata-a": "prefill-a",
                b"metadata-b": "prefill-b",
            }
        )
        mgr = TestNixlPeerLifecycle._manager(agent)
        mgr.transfer_statuses = defaultdict(TransferStatus)
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_addr = "prefill:8998"
        receiver.bootstrap_room = 28
        receiver.prefill_peers = []
        first_route = TestNixlPeerLifecycle._route(
            agent_name="prefill-a",
            metadata=b"metadata-a",
            tp_rank=0,
        )
        spoofed_second_route = TestNixlPeerLifecycle._route(
            agent_name="spoofed-b",
            metadata=b"metadata-b",
            tp_rank=1,
        )
        receiver.bootstrap_infos = [first_route, spoofed_second_route]

        with self.assertRaisesRegex(RuntimeError, "different agent"):
            receiver._load_bootstrap_peers()

        self.assertEqual(receiver.prefill_peers, [])
        self.assertEqual(len(mgr._prefill_peers), 1)
        retained_first_peer = next(iter(mgr._prefill_peers.values()))
        self.assertEqual(retained_first_peer.agent_name, "prefill-a")
        self.assertEqual(
            mgr.transfer_statuses[28].expected_source_ranks,
            {},
        )
        self.assertEqual(len(agent.removal_calls), 1)
        removed_second_handle = agent.removal_calls[0]
        self.assertEqual(removed_second_handle.name, "prefill-b")

        valid_second_route = dict(spoofed_second_route)
        valid_second_route["nixl_agent_name"] = "prefill-b"
        receiver.bootstrap_infos = [first_route, valid_second_route]

        receiver._load_bootstrap_peers()

        self.assertIs(receiver.prefill_peers[0], retained_first_peer)
        self.assertEqual(receiver.prefill_peers[1].agent_name, "prefill-b")
        self.assertIsNot(receiver.prefill_peers[1].handle, removed_second_handle)
        self.assertGreater(
            receiver.prefill_peers[1].handle.generation,
            removed_second_handle.generation,
        )
        self.assertEqual(
            agent.add_calls,
            [b"metadata-a", b"metadata-b", b"metadata-b"],
        )

    def test_send_metadata_records_only_nonempty_state_components(self):
        receiver, mgr = self._make_receiver()
        receiver.bootstrap_infos = []
        mgr.enable_staging = False
        mgr.transfer_statuses = defaultdict(TransferStatus)

        receiver.send_metadata(
            np.array([], dtype=np.int32),
            state_indices=[[4], [], None],
        )

        self.assertEqual(
            mgr.transfer_statuses[11].expected_state_indices,
            {0},
        )
        self.assertFalse(receiver.started_transfer)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_send_metadata_binds_tp4_writer_handles_and_canonical_aux(self):
        receiver, mgr = self._make_receiver()
        generation = "00000000-0000-4000-8000-000000000010"
        handles = [FakeRemoteHandle(f"prefill-{rank}") for rank in range(4)]
        receiver.bootstrap_infos = [
            {
                "rank_ip": "127.0.0.1",
                "rank_port": 31000 + rank,
                "is_dummy": False,
            }
            for rank in range(4)
        ]
        receiver.prefill_peers = [
            SimpleNamespace(
                handle=handles[rank],
                transfer_source_rank=rank,
                process_generation=generation,
                attn_tp_rank=rank,
                attn_cp_rank=0,
                pp_rank=0,
            )
            for rank in range(4)
        ]
        receiver.required_dst_info_num = 1
        mgr.enable_staging = False
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.agent = SimpleNamespace(name="decode-agent")
        mgr.process_generation = generation
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 32000
        sockets = [MagicMock() for _ in range(4)]
        receiver._connect_to_bootstrap_server = MagicMock(
            side_effect=[(sock, threading.Lock()) for sock in sockets]
        )

        receiver.send_metadata(np.array([1], dtype=np.int32), aux_index=0)

        status = mgr.transfer_statuses[11]
        self.assertEqual(
            status.expected_source_ranks,
            {handle: rank for rank, handle in enumerate(handles)},
        )
        self.assertIs(status.canonical_aux_source, handles[0])
        self.assertTrue(receiver.started_transfer)
        for sock in sockets:
            frames = sock.send_multipart.call_args.args[0]
            self.assertEqual(frames[-1], generation.encode("ascii"))

    def test_returns_existing_conclude_state_without_polling_manager(self):
        receiver, mgr = self._make_receiver()
        receiver.conclude_state = KVPoll.Success

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.check_status.assert_not_called()

    def test_returns_bootstrap_status_before_transfer_starts(self):
        receiver, mgr = self._make_receiver(status=KVPoll.Bootstrapping)

        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        mgr.update_transfer_status.assert_not_called()

    def test_manager_success_or_failed_status_is_terminal(self):
        for terminal_status in (KVPoll.Success, KVPoll.Failed):
            receiver, _ = self._make_receiver(status=terminal_status)

            self.assertEqual(receiver.poll(), terminal_status)
            self.assertEqual(receiver.conclude_state, terminal_status)

    def test_failure_drained_from_notifications_is_immediately_terminal(self):
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        mgr.check_status.side_effect = [KVPoll.WaitingForInput, KVPoll.Failed]

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        mgr.update_transfer_status.assert_called_once_with()
        mgr.check_transfer_done.assert_not_called()

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_waiting_timeout_records_failure(self, mock_time):
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        mgr.record_failure.assert_called_once()
        self.assertIn("timed out", mgr.record_failure.call_args[0][1])
        mgr.update_status.assert_called_once_with(11, KVPoll.Failed)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_queued_completion_wins_over_waiting_timeout(self, mock_time):
        # Past the deadline, but the completion is already queued/observed:
        # draining before the timeout check must yield Success, not a false
        # timeout, and must not send an abort.
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        mgr.transfer_statuses = {11: TransferStatus()}
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.update_transfer_status.assert_called_once_with()
        mgr.record_failure.assert_not_called()
        mgr.update_status.assert_not_called()
        self.assertNotIn(11, mgr.transfer_statuses)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_transfer_done_returns_success_and_cleans_room_state(self, mock_time):
        mock_time.return_value = 12.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        status = TransferStatus()
        status.received_aux = True
        status.num_source_writers_expected = 1
        status.expected_kvs_per_source[0] = 0
        mgr.transfer_statuses = {11: status}
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertNotIn(11, mgr.transfer_statuses)
        self.assertNotIn(11, mgr.addr_to_rooms_tracker["prefill:8998"])
        self.assertEqual(receiver.conclude_state, KVPoll.Success)

    def test_clear_releases_failed_room_bookkeeping(self):
        receiver, mgr = self._make_receiver(status=KVPoll.Failed)
        mgr.request_status = {11: KVPoll.Failed}
        mgr.required_prefill_response_num_table = {11: 1}
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.prefill_response_tracker[11].add(0)
        mgr.transfer_statuses = {11: TransferStatus()}
        mgr.agent = SimpleNamespace(remove_remote_agent=MagicMock())

        receiver.clear()

        self.assertNotIn(11, mgr.request_status)
        self.assertNotIn(11, mgr.required_prefill_response_num_table)
        self.assertNotIn(11, mgr.prefill_response_tracker)
        self.assertNotIn(11, mgr.transfer_statuses)
        self.assertNotIn(11, mgr.addr_to_rooms_tracker["prefill:8998"])
        mgr.agent.remove_remote_agent.assert_not_called()


class TestNixlNodeFailure(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.connection_lock = threading.Lock()
        # Connection keys are "{addr}_{dp_rank}_{cp_rank}_{tp_rank}".
        mgr.connection_pool = {
            "10.0.0.1:8998_0_0_0": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.1:8998_0_0_1": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.2:8998_0_0_0": [{"rank_ip": "10.0.0.2"}],
        }
        mgr.prefill_info_table = {
            "10.0.0.1:8998": object(),
            "10.0.0.2:8998": object(),
        }
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["10.0.0.1:8998"] = {3, 4, 5}
        mgr.request_status = {
            3: KVPoll.WaitingForInput,
            4: KVPoll.Transferring,
            5: KVPoll.Success,
        }
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.update_status = CommonKVManager.update_status.__get__(mgr, CommonKVManager)
        mgr.check_status = CommonKVManager.check_status.__get__(mgr, CommonKVManager)
        mgr.record_failure = CommonKVManager.record_failure.__get__(
            mgr, CommonKVManager
        )
        mgr.agent = SimpleNamespace(remove_remote_agent=MagicMock())
        mgr._prefill_peer_keys_by_addr = defaultdict(set)
        mgr._prefill_peers = {}
        mgr._prefill_peers_by_agent_name = {}
        mgr._prefill_peers_by_handle = {}
        mgr._prefill_peer_lock = threading.RLock()
        mgr._quarantined_remote_handles = set()
        return mgr

    def test_handle_node_failure_removes_connections_and_marks_pending_rooms(self):
        mgr = self._make_manager()

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998_0_0_1", mgr.connection_pool)
        self.assertIn("10.0.0.2:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998", mgr.addr_to_rooms_tracker)
        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertEqual(mgr.request_status[4], KVPoll.Failed)
        self.assertEqual(mgr.request_status[5], KVPoll.Success)
        self.assertIn(3, mgr.failure_records)
        self.assertIn(4, mgr.failure_records)
        self.assertNotIn(5, mgr.failure_records)

    def test_late_failed_update_does_not_resurrect_cleared_room(self):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {}

        CommonKVManager.update_status(mgr, 9, KVPoll.Failed)

        self.assertNotIn(9, mgr.request_status)


class TestNixlStaging(CustomTestCase):
    def _make_manager(self, agent=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent or StagingFakeAgent()
        mgr.decode_kv_args_table = {
            "peer": SimpleNamespace(remote_handle=FakeRemoteHandle("peer"))
        }
        mgr.attn_tp_size = 2
        mgr.is_mla_backend = False
        mgr.transfer_source_rank = 1
        mgr.kv_args = SimpleNamespace(
            gpu_id=1,
            engine_rank=1,
            page_size=2,
            total_kv_head_num=2,
            kv_head_num=1,
        )
        mgr.server_args = SimpleNamespace(chunked_prefill_size=4)
        return mgr

    def test_register_buffer_to_engine_groups_kv_memory_kinds_in_one_pass(self):
        agent = StagingFakeAgent(register_result=["desc"])
        mgr = self._make_manager(agent)
        mgr.kv_args.kv_data_ptrs = [0x1000, 0x2000, 0x3000]
        mgr.kv_args.kv_data_lens = [64, 128, 256]
        mgr.kv_args.kv_data_mem_kinds = ["VRAM", "DRAM", "VRAM"]
        mgr.kv_args.aux_data_ptrs = [0x4000]
        mgr.kv_args.aux_data_lens = [32]
        mgr.kv_args.state_data_ptrs = []
        mgr.kv_args.state_data_lens = []

        mgr.register_buffer_to_engine()

        self.assertEqual(
            agent.register_memory_calls,
            [
                (
                    [(0x1000, 64, 1, ""), (0x3000, 256, 1, "")],
                    "VRAM",
                ),
                ([(0x2000, 128, 0, "")], "DRAM"),
                ([(0x4000, 32, 0, "")], "DRAM"),
            ],
        )
        self.assertEqual(mgr.kv_descs, [["desc"], ["desc"]])
        self.assertEqual(mgr.aux_descs, ["desc"])

    def test_post_failure_logs_attestation_before_releasing_handle(self) -> None:
        agent = TransferFailureFakeAgent()
        mgr = self._make_manager(agent)

        with (
            patch(
                "sglang.srt.disaggregation.nixl.conn._NIXL_TRANSPORT_ERRORS",
                (RuntimeError,),
            ),
            self.assertLogs(
                "sglang.srt.disaggregation.nixl.conn", level="ERROR"
            ) as logs,
            self.assertRaisesRegex(RuntimeError, "NIXL_ERR_BACKEND"),
        ):
            mgr._post_transfer_when_ready("failed-handle", "staging part 2/7")

        self.assertEqual(agent.events, ["transfer", "attestation", "release"])
        self.assertEqual(agent.release_xfer_handle_calls, ["failed-handle"])
        evidence = "\n".join(logs.output)
        self.assertIn("staging part 2/7", evidence)
        self.assertIn('"status": "NIXL_ERR_BACKEND"', evidence)
        self.assertIn('"error": "ucp_ep_flush_nbx failed"', evidence)
        self.assertIn('"segment_count": 1', evidence)
        self.assertIn('"posted_segment_count": 1', evidence)
        self.assertIn('"local_address": "0x9000"', evidence)
        self.assertIn('"length": 33554432', evidence)
        self.assertIn('"transport": "rc_mlx5"', evidence)
        self.assertIn('"flush_posted": true', evidence)
        self.assertIn('"remote_flushed": false', evidence)

    def test_register_staging_memory_uses_vram_and_fails_on_empty_descs(self):
        agent = StagingFakeAgent(register_result=["staging"])
        mgr = self._make_manager(agent)

        mgr._register_staging_memory(0x1000, 4096, 3)

        self.assertEqual(
            agent.register_memory_calls,
            [([(0x1000, 4096, 3, "")], "VRAM")],
        )

        mgr = self._make_manager(StagingFakeAgent(register_result=[]))
        with self.assertRaisesRegex(RuntimeError, "staging buffer"):
            mgr._register_staging_memory(0x1000, 4096, 3)

    def test_prefetch_staging_reqs_noops_when_disabled_or_missing_kv_buffers(self):
        mgr = self._make_manager()
        mgr.enable_staging = False
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}

        mgr._prefetch_staging_reqs(3)

        mgr.enable_staging = True
        mgr.kv_buffer_tensors = None
        mgr._prefetch_staging_reqs(3)

    def test_prefetch_staging_reqs_marks_room_when_no_peer_needs_staging(self):
        mgr = self._make_manager()
        mgr.enable_staging = True
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}
        mgr._staging_ctx = PrefillStagingContext()
        mgr.transfer_infos = {
            3: {
                "agent": TransferInfo(
                    room=3,
                    endpoint="127.0.0.1",
                    dst_port=1000,
                    agent_name="agent",
                    dst_kv_indices=np.array([1], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(decode_tp_size=2),
        }

        mgr._prefetch_staging_reqs(3)

        self.assertIn(3, mgr._staging_ctx.prefetched_rooms)

    def test_do_staging_transfer_requeues_when_allocation_not_ready(self):
        mgr = self._make_manager()
        mgr._staging_ctx = PrefillStagingContext()
        strategy = MagicMock()
        strategy.check_ready.return_value = (False, 0, -1, 0, -1)
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=False,
            chunk_id=0,
            prefill_aux_index=None,
            state_indices=None,
        )
        req = SimpleNamespace(room=3, agent_name="decode_agent")
        queue = FakeQueue()

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            completed, deferred = mgr._do_staging_transfer(
                strategy,
                kv_chunk,
                kv_chunk.prefill_kv_indices,
                req,
                SimpleNamespace(),
                queue,
            )

        self.assertFalse(completed)
        self.assertTrue(deferred)
        self.assertEqual(queue.items, [kv_chunk])

    def test_do_staging_transfer_raises_for_oversized_allocation(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (
            False,
            0,
            FakeStagingAllocator.ALLOC_OVERSIZED,
            0,
            -1,
        )
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            chunk_id=0,
            prefill_aux_index=None,
            state_indices=None,
        )

        with self.assertRaisesRegex(RuntimeError, "ring buffer total size"):
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.common.staging_buffer": (
                        _fake_staging_buffer_module()
                    )
                },
            ):
                mgr._do_staging_transfer(
                    strategy,
                    kv_chunk,
                    kv_chunk.prefill_kv_indices,
                    SimpleNamespace(room=3, agent_name="decode_agent"),
                    SimpleNamespace(),
                    FakeQueue(),
                )

    def test_do_staging_transfer_builds_staging_notification(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (True, 2, 128, 0, 512)
        strategy.staging_buffer = FakeStagingBuffer()
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(4, 6),
            is_last_chunk=True,
            chunk_id=7,
            prefill_aux_index=0,
            state_indices=None,
        )
        dst_info = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=1000,
            agent_name="decode_agent",
            agent_metadata=b"",
            dst_kv_ptrs=[],
            dst_kv_mem_kinds=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            gpu_id=5,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=128,
            dst_kv_item_lens=[],
            staging=SimpleNamespace(base_ptr=0x8000, total_size=4096),
        )
        calls = []
        mgr.send_kvcache_staged = lambda *args, **kwargs: (
            calls.append((args, kwargs)) or True
        )

        completed, deferred = mgr._do_staging_transfer(
            strategy,
            kv_chunk,
            kv_chunk.prefill_kv_indices,
            SimpleNamespace(room=3, agent_name="decode_agent"),
            dst_info,
            FakeQueue(),
        )

        self.assertTrue(completed)
        self.assertFalse(deferred)
        self.assertEqual(calls[0][0][8], "3_stg_7_1_1_2_4_2_decode_agent")

    def test_segmented_rma_geometry_preserves_contiguous_coverage(self) -> None:
        src_base_ptr = 0x9000
        dst_base_ptr = 0x100000
        cases = (
            (55, 2, 23),
            (220, 7, 28),
            (880, 28, 16),
        )

        for size_mib, expected_count, expected_tail_mib in cases:
            with self.subTest(size_mib=size_mib):
                total_bytes = size_mib * 1024 * 1024
                src_reqs, dst_reqs = _build_contiguous_rma_requests(
                    src_base_ptr,
                    dst_base_ptr,
                    total_bytes,
                    src_gpu_id=3,
                    dst_gpu_id=6,
                )

                self.assertEqual(src_reqs.shape, (expected_count, 3))
                self.assertEqual(dst_reqs.shape, (expected_count, 3))
                np.testing.assert_array_equal(src_reqs[:, 1], dst_reqs[:, 1])
                np.testing.assert_array_equal(
                    src_reqs[:, 2], np.full(expected_count, 3, dtype=np.int64)
                )
                np.testing.assert_array_equal(
                    dst_reqs[:, 2], np.full(expected_count, 6, dtype=np.int64)
                )
                self.assertEqual(int(src_reqs[:, 1].sum()), total_bytes)
                self.assertLessEqual(int(src_reqs[:, 1].max()), NIXL_RMA_SEGMENT_BYTES)
                self.assertEqual(int(src_reqs[-1, 1]), expected_tail_mib * 1024 * 1024)
                np.testing.assert_array_equal(
                    src_reqs[1:, 0], src_reqs[:-1, 0] + src_reqs[:-1, 1]
                )
                np.testing.assert_array_equal(
                    dst_reqs[1:, 0], dst_reqs[:-1, 0] + dst_reqs[:-1, 1]
                )
                self.assertEqual(int(src_reqs[0, 0]), src_base_ptr)
                self.assertEqual(int(dst_reqs[0, 0]), dst_base_ptr)
                self.assertEqual(
                    int(src_reqs[-1, 0] + src_reqs[-1, 1]),
                    src_base_ptr + total_bytes,
                )
                self.assertEqual(
                    int(dst_reqs[-1, 0] + dst_reqs[-1, 1]),
                    dst_base_ptr + total_bytes,
                )

    def test_segmented_rma_geometry_rejects_invalid_and_overflowing_ranges(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "total_bytes must be positive"):
            _build_contiguous_rma_requests(0x1000, 0x2000, 0, 1, 2)
        with self.assertRaisesRegex(ValueError, "max_segment_bytes must be positive"):
            _build_contiguous_rma_requests(0x1000, 0x2000, 1, 1, 2, max_segment_bytes=0)

        int64_max = int(np.iinfo(np.int64).max)
        with self.assertRaisesRegex(OverflowError, "source RMA range"):
            _build_contiguous_rma_requests(int64_max - 15, 0x2000, 32, 1, 2)
        with self.assertRaisesRegex(OverflowError, "destination RMA range"):
            _build_contiguous_rma_requests(0x1000, int64_max - 15, 32, 1, 2)

    def test_send_kvcache_staged_uses_one_bulk_vram_write(self):
        mock_gather = MagicMock()
        agent = StagingFakeAgent()
        mgr = self._make_manager(agent)
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module(mock_gather)
                )
            },
        ):
            completed = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0x100000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif="3_stg_0_1_1_0_0_2_peer",
                staging_buffer=FakeStagingBuffer(ptr=0x9000, size=1 << 20),
            )

        self.assertTrue(completed)
        self.assertEqual(agent.release_xfer_handle_calls, ["handle"])
        mock_gather.assert_called_once()
        src_reqs, src_mem = agent.get_xfer_descs_calls[0]
        dst_reqs, dst_mem = agent.get_xfer_descs_calls[1]
        self.assertEqual(src_mem, "VRAM")
        self.assertEqual(dst_mem, "VRAM")
        self.assertEqual(src_reqs.shape, (1, 3))
        self.assertEqual(dst_reqs.shape, (1, 3))
        self.assertTrue(np.issubdtype(src_reqs.dtype, np.integer))
        self.assertTrue(np.issubdtype(dst_reqs.dtype, np.integer))
        self.assertEqual(int(src_reqs[0, 0]), 0x9000)
        self.assertGreaterEqual(int(dst_reqs[0, 0]), 0x100000)
        self.assertEqual(agent.initialize_xfer_calls[0][0], "WRITE")
        self.assertEqual(
            agent.initialize_xfer_calls[0][-1],
            b"3_stg_0_1_1_0_0_2_part_0_1_peer",
        )

    def test_send_kvcache_staged_segments_large_bulk_vram_writes(self) -> None:
        cases = (
            (55, 2, 23),
            (220, 7, 28),
            (880, 28, 16),
        )

        for size_mib, expected_count, expected_tail_mib in cases:
            with self.subTest(size_mib=size_mib):
                total_bytes = size_mib * 1024 * 1024
                mock_gather = MagicMock()
                agent = StagingFakeAgent()
                mgr = self._make_manager(agent)
                synthetic_tensor = SizedFakeTensor(total_bytes // 2)
                mgr.kv_buffer_tensors = {
                    "k_buffers": [synthetic_tensor],
                    "v_buffers": [synthetic_tensor],
                    "page_size": 1,
                }

                with patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.disaggregation.common.staging_buffer": (
                            _fake_staging_buffer_module(mock_gather)
                        )
                    },
                ):
                    completed = mgr.send_kvcache_staged(
                        "peer",
                        np.array([1], dtype=np.int32),
                        dst_staging_ptr=0x100000,
                        dst_staging_size=total_bytes,
                        dst_gpu_id=4,
                        dst_tp_rank=0,
                        dst_attn_tp_size=1,
                        dst_kv_item_len=128,
                        notif=f"3_stg_0_1_1_0_0_1_peer",
                        staging_buffer=FakeStagingBuffer(ptr=0x9000, size=total_bytes),
                    )

                self.assertTrue(completed)
                self.assertEqual(
                    agent.release_xfer_handle_calls,
                    ["handle"] * expected_count,
                )
                mock_gather.assert_called_once()
                src_reqs = np.concatenate(
                    [
                        request
                        for request, memory_type in agent.get_xfer_descs_calls[0::2]
                        if memory_type == "VRAM"
                    ]
                )
                dst_reqs = np.concatenate(
                    [
                        request
                        for request, memory_type in agent.get_xfer_descs_calls[1::2]
                        if memory_type == "VRAM"
                    ]
                )
                self.assertEqual(src_reqs.shape, (expected_count, 3))
                self.assertEqual(dst_reqs.shape, (expected_count, 3))
                self.assertTrue(
                    all(
                        requests.shape == (1, 3)
                        for requests, _ in agent.get_xfer_descs_calls
                    )
                )
                np.testing.assert_array_equal(src_reqs[:, 1], dst_reqs[:, 1])
                np.testing.assert_array_equal(
                    src_reqs[1:, 0], src_reqs[:-1, 0] + src_reqs[:-1, 1]
                )
                np.testing.assert_array_equal(
                    dst_reqs[1:, 0], dst_reqs[:-1, 0] + dst_reqs[:-1, 1]
                )
                self.assertEqual(int(src_reqs[:, 1].sum()), total_bytes)
                self.assertEqual(int(src_reqs[-1, 1]), expected_tail_mib * 1024 * 1024)
                self.assertEqual(
                    int(src_reqs[-1, 0] + src_reqs[-1, 1]),
                    0x9000 + total_bytes,
                )
                self.assertEqual(
                    int(dst_reqs[-1, 0] + dst_reqs[-1, 1]),
                    0x100000 + 256 + total_bytes,
                )
                self.assertEqual(len(agent.initialize_xfer_calls), expected_count)
                self.assertEqual(
                    [call[-1] for call in agent.initialize_xfer_calls],
                    [
                        f"3_stg_0_1_1_0_0_1_part_{part_idx}_{expected_count}_peer".encode(
                            "ascii"
                        )
                        for part_idx in range(expected_count)
                    ],
                )

    def test_send_kvcache_staged_falls_back_when_prefill_buffer_too_small(self):
        mgr = self._make_manager()
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            completed = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0xA000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif="notif",
                staging_buffer=FakeStagingBuffer(size=1),
            )

        self.assertFalse(completed)


class TestNixlPackedManagerIntegration(CustomTestCase):
    """Exercise the packed manager seams around the actor implementations."""

    def test_manager_lifecycle_methods_delegate_to_ready_controller(self) -> None:
        """Keep request lifecycle ownership inside one ready controller."""

        controller = MagicMock()
        controller.ready = True
        transaction = object()
        request_owner = object()
        metadata_allocator = object()
        consumer_authority = object()
        allocation_lease = object()
        allocation_authority = object()
        lifecycle_authority = object()
        controller.prepare_transaction.return_value = transaction
        controller.poll.return_value = KVPoll.Success
        controller.cancel_unpublished.return_value = request_owner
        manager = object.__new__(NixlKVManager)
        manager._packed_decode_controller = controller

        manager.attach_packed_decode_scheduler(
            metadata_allocator,
            consumer_authority,
        )
        prepared = manager.prepare_packed_decode_request_transaction(
            room_id=41,
            request_owner=request_owner,
            metadata_buffer_index=3,
            allocation_lease=allocation_lease,
            allocation_authority=allocation_authority,
            lifecycle_authority=lifecycle_authority,
            source_tp_size=4,
        )
        polled = manager.poll_packed_decode_request_transaction(transaction)
        cancelled = manager.cancel_unpublished_packed_decode_request_transaction(
            transaction
        )
        manager.complete_packed_decode_request_metadata_consumption(transaction)
        manager.quarantine_packed_decode_request_transaction(transaction, "failed")

        controller.attach_scheduler.assert_called_once_with(
            metadata_allocator,
            consumer_authority,
        )
        controller.prepare_transaction.assert_called_once_with(
            room_id=41,
            request_owner=request_owner,
            metadata_buffer_index=3,
            allocation_lease=allocation_lease,
            allocation_authority=allocation_authority,
            lifecycle_authority=lifecycle_authority,
            source_tp_size=4,
        )
        self.assertIs(prepared, transaction)
        self.assertEqual(polled, KVPoll.Success)
        self.assertIs(cancelled, request_owner)
        controller.poll.assert_called_once_with(transaction)
        controller.cancel_unpublished.assert_called_once_with(transaction)
        controller.complete_metadata_consumption.assert_called_once_with(transaction)
        controller.quarantine.assert_called_once_with(transaction, "failed")

    def test_capabilities_follow_initialized_runtime_ownership(self) -> None:
        """Advertise data and grant protocols only from their owning actors."""

        prefill = object.__new__(NixlKVManager)
        prefill.disaggregation_mode = DisaggregationMode.PREFILL
        prefill._packed_prefill_runtime = object()
        prefill._packed_decode_controller = None
        decode = object.__new__(NixlKVManager)
        decode.disaggregation_mode = DisaggregationMode.DECODE
        decode._packed_prefill_runtime = None
        decode._packed_decode_controller = SimpleNamespace(ready=True)

        self.assertEqual(
            prefill.kv_transfer_protocol(),
            PACKED_KV_TRANSFER_PROTOCOL,
        )
        self.assertEqual(
            prefill.prepared_grant_protocol(),
            PACKED_PREPARED_GRANT_PROTOCOL,
        )
        self.assertEqual(
            decode.kv_transfer_protocol(),
            PACKED_KV_TRANSFER_PROTOCOL,
        )
        self.assertIsNone(decode.prepared_grant_protocol())

        prefill._packed_prefill_runtime = None
        decode._packed_decode_controller.ready = False

        self.assertIsNone(prefill.kv_transfer_protocol())
        self.assertIsNone(prefill.prepared_grant_protocol())
        self.assertIsNone(decode.kv_transfer_protocol())


class TestNixlPackedMetadataIntegration(CustomTestCase):
    """Validate generation-bound packed request metadata through conn.py."""

    def test_auxiliary_plan_round_trips_through_receiver_metadata(self) -> None:
        """Append and parse the decoder-authored plan without a side channel."""

        generation = str(uuid.uuid4())
        plan = _packed_plan(generation)
        handle = FakeRemoteHandle("prefill")
        manager = SimpleNamespace(
            enable_staging=False,
            transfer_statuses=defaultdict(TransferStatus),
            agent=SimpleNamespace(name="decode-agent"),
            process_generation=generation,
            local_ip="127.0.0.1",
            rank_port=32000,
            kv_args=SimpleNamespace(engine_rank=0),
        )
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_addr = "prefill:8998"
        receiver.bootstrap_room = 41
        receiver.bootstrap_infos = [
            {
                "rank_ip": "127.0.0.1",
                "rank_port": 31000,
                "is_dummy": False,
            }
        ]
        receiver.prefill_peers = [
            SimpleNamespace(
                handle=handle,
                transfer_source_rank=0,
                process_generation=str(uuid.uuid4()),
                attn_tp_rank=0,
                attn_cp_rank=0,
                pp_rank=0,
            )
        ]
        receiver.required_dst_info_num = 1
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )

        receiver.send_metadata(
            np.array([5, 8], dtype=np.int32),
            aux_index=3,
            decode_prefix_len=11,
            packed_plan=plan,
        )

        frames = socket.send_multipart.call_args.args[0]
        parsed = TransferInfo.from_zmq(frames[1:])
        self.assertEqual(frames[-1], encode_packed_message(plan))
        self.assertEqual(parsed.packed_plan, plan)
        self.assertEqual(parsed.process_generation, generation)
        self.assertTrue(receiver.started_transfer)

    def test_tp2_rank_one_plan_owns_rank_one_auxiliary_source(self) -> None:
        """Bind packed auxiliary completion to the routed TP2 source rank."""

        generation = str(uuid.uuid4())
        plan = _packed_plan(generation, canonical_source_rank=1)
        handle = FakeRemoteHandle("prefill-1")
        manager = SimpleNamespace(
            enable_staging=False,
            transfer_statuses=defaultdict(TransferStatus),
            agent=SimpleNamespace(name="decode-agent"),
            process_generation=generation,
            local_ip="127.0.0.1",
            rank_port=32001,
            kv_args=SimpleNamespace(engine_rank=1),
        )
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_addr = "prefill:8998"
        receiver.bootstrap_room = 41
        receiver.bootstrap_infos = [
            {
                "rank_ip": "127.0.0.1",
                "rank_port": 31001,
                "is_dummy": False,
            }
        ]
        receiver.prefill_peers = [
            SimpleNamespace(
                handle=handle,
                transfer_source_rank=1,
                process_generation=str(uuid.uuid4()),
                attn_tp_rank=1,
                attn_cp_rank=0,
                pp_rank=0,
            )
        ]
        receiver.required_dst_info_num = 1
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )

        receiver.send_metadata(
            np.array([5, 8], dtype=np.int32),
            aux_index=3,
            packed_plan=plan,
        )

        status = manager.transfer_statuses[41]
        self.assertIs(status.canonical_aux_source, handle)
        self.assertTrue(receiver.started_transfer)


class TestNixlPackedControlAuthentication(CustomTestCase):
    """Reject packed control claims that differ from native peer authority."""

    def test_decode_rejects_stale_prefill_generation(self) -> None:
        """Do not dispatch prefill control from another process generation."""

        expected_generation = str(uuid.uuid4())
        handle = FakeRemoteHandle("prefill")
        controller = MagicMock()
        controller.ready = True
        manager = object.__new__(NixlKVManager)
        manager._prefill_peer_lock = threading.RLock()
        manager._prefill_peers_by_agent_name = {
            "prefill-agent": SimpleNamespace(
                agent_name="prefill-agent",
                process_generation=expected_generation,
                handle=handle,
                transfer_source_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                pp_rank=0,
            )
        }
        manager._quarantined_remote_handles = set()
        manager._packed_decode_controller = controller
        frames = encode_packed_control_frames(
            "prefill-agent",
            str(uuid.uuid4()),
            _packed_ready(_packed_writer()),
        )

        manager._handle_packed_decode_control(frames)

        controller.handle_control_frames.assert_not_called()

    def test_prefill_rejects_stale_decoder_generation(self) -> None:
        """Do not dispatch decode control from another process generation."""

        expected_generation = str(uuid.uuid4())
        runtime = MagicMock()
        manager = object.__new__(NixlKVManager)
        manager.decode_kv_args_table = {
            "decode-agent": SimpleNamespace(
                agent_name="decode-agent",
                process_generation=expected_generation,
                remote_handle=FakeRemoteHandle("decode"),
                packed_transfer_protocol=PACKED_KV_TRANSFER_PROTOCOL,
                prepared_grant_protocol=PACKED_PREPARED_GRANT_PROTOCOL,
                packed_advertisement=_packed_advertisement(),
            )
        }
        manager._prefill_peer_lock = threading.RLock()
        manager._quarantined_remote_handles = set()
        manager._packed_prefill_runtime = runtime
        frames = encode_packed_control_frames(
            "decode-agent",
            str(uuid.uuid4()),
            _packed_ready(_packed_writer()),
        )

        manager._handle_packed_prefill_control(frames)

        runtime.handle_control.assert_not_called()


class TestNixlPackedSourceIntegration(CustomTestCase):
    """Validate the production source submission assembled by conn.py."""

    def test_tp2_rank_one_registration_matches_source_rank_one(self) -> None:
        """Accept only the destination rank connected to this source actor."""

        manager = object.__new__(NixlKVManager)
        manager.attn_tp_size = 2
        manager._packed_prefill_runtime = SimpleNamespace(writer_id=_packed_writer(1))

        manifest = manager._packed_destination_manifest(
            SimpleNamespace(decode_tp_size=2, decode_tp_rank=1)
        )

        self.assertEqual(manifest.writers, (_packed_writer(1),))
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            manager._packed_destination_manifest(
                SimpleNamespace(decode_tp_size=2, decode_tp_rank=0)
            )

    def test_source_submission_uses_decoder_registration_and_page_projection(
        self,
    ) -> None:
        """Construct one packed submission from authenticated decoder metadata."""

        decode_generation = str(uuid.uuid4())
        source_generation = str(uuid.uuid4())
        plan = _packed_plan(decode_generation)
        remote_handle = FakeRemoteHandle("decode")
        registration = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=33000,
            agent_name="decode-agent",
            agent_metadata=b"metadata",
            dst_kv_ptrs=[0x1000],
            dst_kv_mem_kinds=["VRAM"],
            dst_aux_ptrs=[0x2000],
            dst_state_data_ptrs=[],
            gpu_id=1,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=256,
            dst_kv_item_lens=[256],
            dst_kv_layer_ids=[3],
            process_generation=decode_generation,
            packed_transfer_protocol=PACKED_KV_TRANSFER_PROTOCOL,
            prepared_grant_protocol=PACKED_PREPARED_GRANT_PROTOCOL,
            packed_advertisement=_packed_advertisement(),
            remote_handle=remote_handle,
        )
        transfer_info = TransferInfo(
            room=41,
            endpoint="127.0.0.1",
            dst_port=33000,
            agent_name="decode-agent",
            dst_kv_indices=np.array([9, 10], dtype=np.int32),
            dst_aux_index=3,
            required_dst_info_num=1,
            dst_state_indices=[],
            process_generation=decode_generation,
            packed_plan=plan,
        )
        runtime = MagicMock()
        destination = object()
        runtime.build_destination_capability.return_value = destination
        manager = object.__new__(NixlKVManager)
        manager.attn_cp_rank = 0
        manager.attn_tp_rank = 0
        manager.pp_rank = 0
        manager._packed_prefill_runtime = runtime
        manager.kv_args = SimpleNamespace(state_types=[])
        manager.agent = SimpleNamespace(name="prefill-agent")
        manager.process_generation = source_generation
        manager._send_packed_control_frames = MagicMock()
        producer_event = object()
        submission = object()
        launch_plan = MagicMock(spec=PackedPrefillLaunchPlan)
        launch_plan.bind_producer_event.return_value = submission

        with patch(
            "sglang.srt.disaggregation.nixl.conn.PackedPrefillLaunchPlan",
            return_value=launch_plan,
        ) as launch_plan_factory:
            manager._execute_packed_source_request(
                transfer_info=transfer_info,
                registration=registration,
                source_main_pages=np.array([1, 2], dtype=np.int32),
                auxiliary_source_index=7,
                state_indices=None,
                producer_event=producer_event,
            )

        runtime.build_destination_capability.assert_called_once_with(
            advertisement=_packed_advertisement(),
            decode_peer=ANY,
            destination_gpu_id=1,
            destination_tp_size=1,
            destination_tp_rank=0,
            request_generation=plan.key.request_generation,
        )
        runtime.execute.assert_called_once_with(submission)
        launch_plan.bind_producer_event.assert_called_once_with(producer_event)
        submission_kwargs = launch_plan_factory.call_args.kwargs
        self.assertIs(submission_kwargs["plan"], plan)
        self.assertIs(submission_kwargs["destination"], destination)
        self.assertEqual(
            submission_kwargs["auxiliary_source"],
            PackedLegacyAuxiliarySource(row_index=7),
        )
        self.assertEqual(
            submission_kwargs["destination_registration"].main_item_lens,
            (256,),
        )
        self.assertEqual(
            submission_kwargs["destination_registration"].main_layer_ids,
            (3,),
        )
        component = submission_kwargs["components"][0]
        np.testing.assert_array_equal(component.source_pages, [1, 2])
        np.testing.assert_array_equal(component.destination_pages, [9, 10])
        control = submission_kwargs["control"]
        self.assertIs(control.remote_handle, remote_handle)
        self.assertEqual(
            control.peer.agent_generation,
            uuid.UUID(decode_generation).bytes,
        )


class TestNixlPackedControlRoutes(CustomTestCase):
    """Bind every supported native writer cohort to packed routes."""

    def test_supported_routes_preserve_writer_and_socket_ownership(self) -> None:
        """Keep each canonically ordered writer bound to its own socket."""

        for source_tp_size in (1, 2, 4):
            with self.subTest(source_tp_size=source_tp_size):
                receiver = object.__new__(NixlKVReceiver)
                receiver.bootstrap_infos = [
                    {
                        "rank_ip": "127.0.0.1",
                        "rank_port": 31000 + rank,
                        "is_dummy": False,
                    }
                    for rank in range(source_tp_size)
                ]
                receiver.prefill_peers = [
                    SimpleNamespace(
                        transfer_source_rank=rank,
                        attn_tp_rank=rank,
                        attn_cp_rank=0,
                        pp_rank=0,
                    )
                    for rank in range(source_tp_size)
                ]
                sockets = [MagicMock() for _ in range(source_tp_size)]
                receiver._connect_to_bootstrap_server = MagicMock(
                    side_effect=[(socket, threading.Lock()) for socket in sockets]
                )
                controller = MagicMock()
                controller.build_control_sender.side_effect = (
                    lambda writer_id, send_frames: SimpleNamespace(
                        writer_id=writer_id,
                        send_frames=send_frames,
                    )
                )

                routes = receiver.build_packed_control_routes(controller)

                self.assertEqual(
                    [route.writer_id for route in routes],
                    [_packed_writer(rank) for rank in range(source_tp_size)],
                )
                for rank, route in enumerate(routes):
                    frames = [f"writer-{rank}".encode("ascii")]
                    route.send_frames(frames)
                    sockets[rank].send_multipart.assert_called_once_with(frames)
                self.assertEqual(
                    controller.build_control_sender.call_count,
                    source_tp_size,
                )


class TestNixlSliceTransferBounds(CustomTestCase):
    """Keep heterogeneous-TP prepped requests within cohort budgets."""

    @staticmethod
    def _manager(source_tp_size: int) -> NixlKVManager:
        """Build the narrow manager used by bounded slice sends.

        :param source_tp_size: Source attention TP width.
        :returns: A manager with prepared source and destination dlists.
        """

        manager = object.__new__(NixlKVManager)
        manager.prep_handle_slice_src = ("src", 1, 1, 65536)
        manager.prep_handles_slice_dst = {"decode": ("dst", 65536, 0)}
        manager.kv_args = SimpleNamespace(page_size=1)
        manager.attn_tp_size = source_tp_size
        manager.agent = SimpleNamespace(
            make_prepped_xfer=MagicMock(
                side_effect=lambda *args: f"handle-{args[2][0]}"
            ),
            check_xfer_state=MagicMock(return_value="DONE"),
            release_xfer_handle=MagicMock(),
        )
        manager._post_transfer_when_ready = MagicMock(
            side_effect=lambda handle, _context: handle
        )
        return manager

    def test_tp2_splits_20460_descriptors_and_releases_intermediate(self) -> None:
        """Split at 8,192 and retain only the final 4,076-descriptor handle."""

        manager = self._manager(source_tp_size=2)
        indices = np.arange(20460, dtype=np.int32)

        final_handle = manager.send_kvcache_slice(
            "decode",
            indices,
            indices,
            "room_kv_0_1_0",
        )

        calls = manager.agent.make_prepped_xfer.call_args_list
        self.assertEqual(
            [len(call.args[2]) for call in calls],
            [8192, 8192, 4076],
        )
        self.assertEqual(
            [len(call.args[4]) for call in calls],
            [8192, 8192, 4076],
        )
        self.assertEqual(
            [call.args[5] for call in calls],
            [b"", b"", b"room_kv_0_1_0"],
        )
        self.assertEqual(final_handle, "handle-16384")
        self.assertEqual(
            [call.args[0] for call in manager.agent.check_xfer_state.call_args_list],
            ["handle-0", "handle-8192"],
        )
        self.assertEqual(
            [call.args[0] for call in manager.agent.release_xfer_handle.call_args_list],
            ["handle-0", "handle-8192"],
        )

    def test_tp4_uses_4096_descriptor_cohort_limit(self) -> None:
        """Apply the per-rank 4,096 descriptor budget for TP4 sources."""

        manager = self._manager(source_tp_size=4)
        indices = np.arange(8193, dtype=np.int32)

        final_handle = manager.send_kvcache_slice(
            "decode",
            indices,
            indices,
            "final",
        )

        calls = manager.agent.make_prepped_xfer.call_args_list
        self.assertEqual([len(call.args[2]) for call in calls], [4096, 4096, 1])
        self.assertEqual([call.args[5] for call in calls], [b"", b"", b"final"])
        self.assertEqual(final_handle, "handle-8192")
        self.assertEqual(
            [call.args[0] for call in manager.agent.release_xfer_handle.call_args_list],
            ["handle-0", "handle-4096"],
        )

    def test_mismatched_expanded_descriptor_counts_are_rejected(self) -> None:
        """Reject unequal source and destination descriptor projections."""

        manager = self._manager(source_tp_size=2)

        with self.assertRaisesRegex(
            ValueError,
            "Prepped slice transfer index count mismatch: source=2, destination=1",
        ):
            manager.send_kvcache_slice(
                "decode",
                np.array([1, 2], dtype=np.int32),
                np.array([3], dtype=np.int32),
                "final",
            )

        manager.agent.make_prepped_xfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
