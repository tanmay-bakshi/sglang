import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.common.utils import (
    TensorParallelShard,
    compute_tensor_parallel_shard,
)
from sglang.srt.disaggregation.nixl.conn import (
    NIXL_RMA_MAX_DESCRIPTORS,
    NixlKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingNixlAgent:
    """Records NIXL descriptor and transfer calls without requiring a transport."""

    descriptor_requests: list[tuple[np.ndarray, str]]
    initialize_calls: list[tuple[object, ...]]
    remote_handle: object
    release_xfer_handle_calls: list[object]
    transfer_calls: list[object]

    def __init__(self) -> None:
        self.descriptor_requests = []
        self.initialize_calls = []
        self.remote_handle = object()
        self.release_xfer_handle_calls = []
        self.transfer_calls = []

    def get_xfer_descs(self, requests: np.ndarray, memory_kind: str) -> tuple[str, int]:
        request_copy = np.asarray(requests, dtype=np.uint64).copy()
        self.descriptor_requests.append((request_copy, memory_kind))
        return ("descriptors", len(self.descriptor_requests))

    def initialize_xfer(self, *args: object) -> object:
        self.initialize_calls.append(args)
        return object()

    def transfer(self, handle: object) -> str:
        self.transfer_calls.append(handle)
        return "PROC"

    def check_xfer_state(self, handle: object) -> str:
        return "DONE"

    def release_xfer_handle(self, handle: object) -> None:
        self.release_xfer_handle_calls.append(handle)


class TestComputeTensorParallelShard(unittest.TestCase):
    """Tests byte-exact tensor-parallel shard mapping."""

    def test_gathers_four_source_ranks_into_one_destination(self) -> None:
        """Each source rank fills its ordered quarter of the destination token."""

        actual = [
            compute_tensor_parallel_shard(
                source_token_bytes=1024,
                destination_token_bytes=4096,
                source_parallel_size=4,
                destination_parallel_size=1,
                source_rank=source_rank,
                destination_rank=0,
            )
            for source_rank in range(4)
        ]
        expected = [
            TensorParallelShard(0, destination_offset, 1024)
            for destination_offset in (0, 1024, 2048, 3072)
        ]
        self.assertEqual(actual, expected)

    def test_scatters_one_source_rank_into_two_destinations(self) -> None:
        """Each destination receives its ordered half of the source token."""

        actual = [
            compute_tensor_parallel_shard(
                source_token_bytes=2048,
                destination_token_bytes=1024,
                source_parallel_size=2,
                destination_parallel_size=4,
                source_rank=0,
                destination_rank=destination_rank,
            )
            for destination_rank in (0, 1)
        ]
        expected = [
            TensorParallelShard(0, 0, 1024),
            TensorParallelShard(1024, 0, 1024),
        ]
        self.assertEqual(actual, expected)

    def test_preserves_equal_width_rank_layout(self) -> None:
        """Equal tensor-parallel widths transfer the complete local token."""

        actual = compute_tensor_parallel_shard(
            source_token_bytes=2048,
            destination_token_bytes=2048,
            source_parallel_size=2,
            destination_parallel_size=2,
            source_rank=1,
            destination_rank=1,
        )
        self.assertEqual(actual, TensorParallelShard(0, 0, 2048))

    def test_rejects_replicated_or_incompatible_byte_geometry(self) -> None:
        """Aggregate source and destination bytes must describe one partition."""

        with self.assertRaisesRegex(ValueError, "non-replicated contiguous partition"):
            compute_tensor_parallel_shard(
                source_token_bytes=1024,
                destination_token_bytes=4096,
                source_parallel_size=4,
                destination_parallel_size=2,
                source_rank=0,
                destination_rank=0,
            )

    def test_rejects_rank_pairs_outside_the_transfer_mapping(self) -> None:
        """A source rank cannot write into an unrelated destination rank."""

        with self.assertRaisesRegex(ValueError, "maps to destination rank 0"):
            compute_tensor_parallel_shard(
                source_token_bytes=1024,
                destination_token_bytes=2048,
                source_parallel_size=4,
                destination_parallel_size=2,
                source_rank=1,
                destination_rank=1,
            )


class TestNixlTensorParallelStateTransfer(unittest.TestCase):
    """Tests the byte-exact NIXL request arrays for TP-sharded state."""

    @staticmethod
    def _make_manager(
        *,
        source_tp_size: int,
        source_engine_rank: int,
        page_size: int,
    ) -> tuple[NixlKVManager, RecordingNixlAgent]:
        agent = RecordingNixlAgent()
        manager = object.__new__(NixlKVManager)
        manager.agent = agent
        manager.attn_tp_size = source_tp_size
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            engine_rank=source_engine_rank,
            gpu_id=3,
            page_size=page_size,
        )
        manager.decode_kv_args_table = {
            "decode-peer": SimpleNamespace(remote_handle=agent.remote_handle)
        }
        return manager, agent

    @staticmethod
    def _call_transfer(
        manager: NixlKVManager,
        *,
        source_indices: list[int],
        source_data_ptrs: list[int],
        source_item_lens: list[int],
        destination_indices: list[int],
        destination_data_ptrs: list[int],
        destination_item_lens: list[int],
        destination_tp_size: int,
        destination_engine_rank: int,
        source_layer_ids: list[int] | None = None,
        destination_layer_ids: list[int] | None = None,
    ) -> object | None:
        return manager._send_tp_sharded_state(
            peer_name="decode-peer",
            source_indices=source_indices,
            source_data_ptrs=source_data_ptrs,
            source_item_lens=source_item_lens,
            destination_indices=destination_indices,
            destination_data_ptrs=destination_data_ptrs,
            destination_item_lens=destination_item_lens,
            destination_gpu_id=7,
            destination_tp_size=destination_tp_size,
            destination_engine_rank=destination_engine_rank,
            notification="room_state_0",
            source_layer_ids=source_layer_ids,
            destination_layer_ids=destination_layer_ids,
        )

    def test_page_size_two_routes_exact_layers_tokens_and_descriptors(self) -> None:
        """Every page token and repeated layer id reaches its exact byte range."""

        manager, agent = self._make_manager(
            source_tp_size=2,
            source_engine_rank=1,
            page_size=2,
        )

        handle = self._call_transfer(
            manager,
            source_indices=[3, 7],
            source_data_ptrs=[0x1000, 0x2000, 0x3000, 0x4000],
            source_item_lens=[16, 16, 16, 16],
            destination_indices=[5, 9],
            destination_data_ptrs=[0x8000, 0x9000, 0xA000, 0xB000],
            destination_item_lens=[32, 32, 32, 32],
            destination_tp_size=1,
            destination_engine_rank=0,
            source_layer_ids=[10, 20, 10, 20],
            destination_layer_ids=[20, 10, 20, 10],
        )

        expected_source = np.array(
            [
                [0x1030, 8, 3],
                [0x1038, 8, 3],
                [0x1070, 8, 3],
                [0x1078, 8, 3],
                [0x2030, 8, 3],
                [0x2038, 8, 3],
                [0x2070, 8, 3],
                [0x2078, 8, 3],
                [0x3030, 8, 3],
                [0x3038, 8, 3],
                [0x3070, 8, 3],
                [0x3078, 8, 3],
                [0x4030, 8, 3],
                [0x4038, 8, 3],
                [0x4070, 8, 3],
                [0x4078, 8, 3],
            ],
            dtype=np.uint64,
        )
        expected_destination = np.array(
            [
                [0x90A8, 8, 7],
                [0x90B8, 8, 7],
                [0x9128, 8, 7],
                [0x9138, 8, 7],
                [0x80A8, 8, 7],
                [0x80B8, 8, 7],
                [0x8128, 8, 7],
                [0x8138, 8, 7],
                [0xB0A8, 8, 7],
                [0xB0B8, 8, 7],
                [0xB128, 8, 7],
                [0xB138, 8, 7],
                [0xA0A8, 8, 7],
                [0xA0B8, 8, 7],
                [0xA128, 8, 7],
                [0xA138, 8, 7],
            ],
            dtype=np.uint64,
        )

        self.assertIsNotNone(handle)
        self.assertEqual(len(agent.descriptor_requests), 2)
        np.testing.assert_array_equal(agent.descriptor_requests[0][0], expected_source)
        np.testing.assert_array_equal(
            agent.descriptor_requests[1][0], expected_destination
        )
        self.assertEqual(
            [memory_kind for _, memory_kind in agent.descriptor_requests],
            ["VRAM", "VRAM"],
        )
        self.assertEqual(len(agent.initialize_calls), 1)
        initialize_call = agent.initialize_calls[0]
        self.assertEqual(initialize_call[0], "WRITE")
        self.assertEqual(initialize_call[1], ("descriptors", 1))
        self.assertEqual(initialize_call[2], ("descriptors", 2))
        self.assertIs(initialize_call[3], agent.remote_handle)
        self.assertEqual(initialize_call[4], b"room_state_0")
        self.assertEqual(agent.transfer_calls, [handle])

    def test_large_state_uses_independently_progressed_bounded_handles(self) -> None:
        """Descriptor-heavy SWA posts bounded parts and notifies only at the end."""

        manager, agent = self._make_manager(
            source_tp_size=2,
            source_engine_rank=1,
            page_size=1,
        )
        descriptor_count = NIXL_RMA_MAX_DESCRIPTORS + 3
        indices = list(range(descriptor_count))

        final_handle = self._call_transfer(
            manager,
            source_indices=indices,
            source_data_ptrs=[0x100000000],
            source_item_lens=[2048],
            destination_indices=indices,
            destination_data_ptrs=[0x200000000],
            destination_item_lens=[4096],
            destination_tp_size=1,
            destination_engine_rank=0,
        )

        self.assertEqual(len(agent.descriptor_requests), 4)
        self.assertEqual(
            [requests.shape for requests, _ in agent.descriptor_requests],
            [
                (NIXL_RMA_MAX_DESCRIPTORS, 3),
                (NIXL_RMA_MAX_DESCRIPTORS, 3),
                (3, 3),
                (3, 3),
            ],
        )
        self.assertEqual(len(agent.initialize_calls), 2)
        self.assertEqual(
            [call[4] for call in agent.initialize_calls],
            [b"", b"room_state_0"],
        )
        self.assertEqual(len(agent.transfer_calls), 2)
        self.assertEqual(
            agent.release_xfer_handle_calls,
            [agent.transfer_calls[0]],
        )
        self.assertIs(final_handle, agent.transfer_calls[1])

    def test_tp_four_to_two_routes_only_connected_source_halves(self) -> None:
        """Source ranks two and three fill ordered halves of decode rank one."""

        expected_destination_addresses = {2: 0x8040, 3: 0x8048}
        for (
            source_rank,
            expected_destination_address,
        ) in expected_destination_addresses.items():
            with self.subTest(source_rank=source_rank):
                manager, agent = self._make_manager(
                    source_tp_size=4,
                    source_engine_rank=source_rank,
                    page_size=1,
                )

                self._call_transfer(
                    manager,
                    source_indices=[2],
                    source_data_ptrs=[0x1000],
                    source_item_lens=[8],
                    destination_indices=[4],
                    destination_data_ptrs=[0x8000],
                    destination_item_lens=[16],
                    destination_tp_size=2,
                    destination_engine_rank=1,
                )

                np.testing.assert_array_equal(
                    agent.descriptor_requests[0][0],
                    np.array([[0x1010, 8, 3]], dtype=np.uint64),
                )
                np.testing.assert_array_equal(
                    agent.descriptor_requests[1][0],
                    np.array([[expected_destination_address, 8, 7]], dtype=np.uint64),
                )

    def test_tp_two_to_four_routes_one_source_into_two_destinations(self) -> None:
        """Decode ranks two and three read ordered halves of source rank one."""

        expected_source_addresses = {2: 0x1020, 3: 0x1028}
        for (
            destination_rank,
            expected_source_address,
        ) in expected_source_addresses.items():
            with self.subTest(destination_rank=destination_rank):
                manager, agent = self._make_manager(
                    source_tp_size=2,
                    source_engine_rank=1,
                    page_size=1,
                )

                self._call_transfer(
                    manager,
                    source_indices=[2],
                    source_data_ptrs=[0x1000],
                    source_item_lens=[16],
                    destination_indices=[4],
                    destination_data_ptrs=[0x8000],
                    destination_item_lens=[8],
                    destination_tp_size=4,
                    destination_engine_rank=destination_rank,
                )

                np.testing.assert_array_equal(
                    agent.descriptor_requests[0][0],
                    np.array([[expected_source_address, 8, 3]], dtype=np.uint64),
                )
                np.testing.assert_array_equal(
                    agent.descriptor_requests[1][0],
                    np.array([[0x8020, 8, 7]], dtype=np.uint64),
                )

    def test_normalizes_global_engine_ranks_to_attention_tp_ranks(self) -> None:
        """PP or DP rank offsets must not alter the connected TP shard."""

        manager, agent = self._make_manager(
            source_tp_size=4,
            source_engine_rank=7,
            page_size=1,
        )

        self._call_transfer(
            manager,
            source_indices=[2],
            source_data_ptrs=[0x1000],
            source_item_lens=[8],
            destination_indices=[4],
            destination_data_ptrs=[0x8000],
            destination_item_lens=[16],
            destination_tp_size=2,
            destination_engine_rank=5,
        )

        np.testing.assert_array_equal(
            agent.descriptor_requests[0][0],
            np.array([[0x1010, 8, 3]], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            agent.descriptor_requests[1][0],
            np.array([[0x8048, 8, 7]], dtype=np.uint64),
        )

    def test_rejects_unrelated_peer_before_descriptor_construction(self) -> None:
        """Outer routing must not present a source rank to an unrelated peer."""

        manager, agent = self._make_manager(
            source_tp_size=4,
            source_engine_rank=1,
            page_size=1,
        )

        with self.assertRaisesRegex(ValueError, "maps to destination rank 0"):
            self._call_transfer(
                manager,
                source_indices=[0],
                source_data_ptrs=[0x1000],
                source_item_lens=[8],
                destination_indices=[0],
                destination_data_ptrs=[0x8000],
                destination_item_lens=[16],
                destination_tp_size=2,
                destination_engine_rank=1,
            )

        self.assertEqual(agent.descriptor_requests, [])

    def test_rejects_negative_source_and_destination_page_indices(self) -> None:
        """Negative page ids must not wrap into uint64 addresses."""

        for source_indices, destination_indices, expected_message in (
            ([-1], [0], "Source"),
            ([0], [-1], "Destination"),
        ):
            with self.subTest(
                source_indices=source_indices,
                destination_indices=destination_indices,
            ):
                manager, agent = self._make_manager(
                    source_tp_size=2,
                    source_engine_rank=0,
                    page_size=2,
                )

                with self.assertRaisesRegex(
                    ValueError, f"{expected_message}.*non-negative"
                ):
                    self._call_transfer(
                        manager,
                        source_indices=source_indices,
                        source_data_ptrs=[0x1000],
                        source_item_lens=[16],
                        destination_indices=destination_indices,
                        destination_data_ptrs=[0x8000],
                        destination_item_lens=[32],
                        destination_tp_size=1,
                        destination_engine_rank=0,
                    )

                self.assertEqual(agent.descriptor_requests, [])

    def test_asymmetric_swa_dispatches_to_tp_sharded_transfer(self) -> None:
        """Unequal-TP SWA passes peer geometry to the sharded state path."""

        manager = object.__new__(NixlKVManager)
        manager.attn_tp_size = 2
        manager.pp_size = 1
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.SWA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[16]],
            state_dim_per_tensor=[[]],
            state_conv_shard_groups=[[]],
            state_slice_outer_counts=[[]],
            state_layer_ids=[[12]],
        )
        handle = object()
        manager._send_tp_sharded_state = MagicMock(return_value=handle)

        actual = manager.maybe_send_extra(
            peer_name="decode-peer",
            prefill_state_indices=[[3]],
            dst_state_data_ptrs=[[0x8000]],
            dst_state_indices=[[5]],
            dst_gpu_id=7,
            notif="19_state_1",
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_state_item_lens=[[32]],
            dst_state_layer_ids=[[12]],
        )

        self.assertEqual(actual, [handle])
        manager._send_tp_sharded_state.assert_called_once_with(
            peer_name="decode-peer",
            source_indices=[3],
            source_data_ptrs=[0x1000],
            source_item_lens=[16],
            destination_indices=[5],
            destination_data_ptrs=[0x8000],
            destination_item_lens=[32],
            destination_gpu_id=7,
            destination_tp_size=1,
            destination_engine_rank=0,
            notification="19_state_1_0",
            source_layer_ids=[12],
            destination_layer_ids=[12],
        )


class TestNixlAsymmetricSwaTopology(unittest.TestCase):
    """Tests topology validation before asymmetric SWA bootstrap routing."""

    @staticmethod
    def _make_manager(
        *,
        decode_tp_size: int = 1,
        decode_pp_size: int = 1,
        decode_cp_size: int = 1,
    ) -> NixlKVManager:
        manager = object.__new__(NixlKVManager)
        manager.attn_tp_size = decode_tp_size
        manager.attn_cp_size = decode_cp_size
        manager.pp_size = decode_pp_size
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(state_types=[StateType.SWA])
        return manager

    @staticmethod
    def _make_prefill_info(
        *,
        prefill_tp_size: int = 2,
        prefill_pp_size: int = 1,
        prefill_cp_size: int = 1,
    ) -> PrefillServerInfo:
        return PrefillServerInfo(
            attn_tp_size=prefill_tp_size,
            attn_cp_size=prefill_cp_size,
            dp_size=1,
            pp_size=prefill_pp_size,
            page_size=1,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )

    def test_unsupported_pp_or_cp_rejects_before_base_rank_mapping(self) -> None:
        """Every asymmetric SWA endpoint must use PP1 and CP1."""

        cases = (
            ("decode PP", {"decode_pp_size": 2}, {}),
            ("prefill PP", {}, {"prefill_pp_size": 2}),
            ("decode CP", {"decode_cp_size": 2}, {}),
            ("prefill CP", {}, {"prefill_cp_size": 2}),
        )
        for name, manager_kwargs, info_kwargs in cases:
            with self.subTest(name=name):
                manager = self._make_manager(**manager_kwargs)
                info = self._make_prefill_info(**info_kwargs)

                with patch.object(
                    CommonKVManager,
                    "_resolve_rank_mapping",
                    autospec=True,
                ) as base_resolver:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "requires PP=1 and CP=1",
                    ):
                        manager._resolve_rank_mapping(info)

                    base_resolver.assert_not_called()

    def test_pp1_cp1_asymmetric_swa_delegates_to_base_mapping(self) -> None:
        """The supported topology continues into the common rank resolver."""

        manager = self._make_manager()
        info = self._make_prefill_info()

        with patch.object(
            CommonKVManager,
            "_resolve_rank_mapping",
            autospec=True,
        ) as base_resolver:
            manager._resolve_rank_mapping(info)

        base_resolver.assert_called_once_with(manager, info)

    def test_equal_tp_swa_is_not_rejected_by_asymmetric_guard(self) -> None:
        """Equal TP retains the common resolver's existing PP behavior."""

        manager = self._make_manager(decode_tp_size=2, decode_pp_size=2)
        info = self._make_prefill_info(prefill_tp_size=2, prefill_pp_size=2)

        with patch.object(
            CommonKVManager,
            "_resolve_rank_mapping",
            autospec=True,
        ) as base_resolver:
            manager._resolve_rank_mapping(info)

        base_resolver.assert_called_once_with(manager, info)


class TestTensorParallelPeerRouting(unittest.TestCase):
    """Tests that bootstrap routing gives each prefill only connected peers."""

    @staticmethod
    def _resolve_mapping(
        *, decode_tp_size: int, decode_tp_rank: int, prefill_tp_size: int
    ) -> PrefillServerInfo:
        manager = object.__new__(CommonKVManager)
        manager.attn_tp_size = decode_tp_size
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.kv_args = SimpleNamespace(engine_rank=decode_tp_rank)
        manager.server_args = SimpleNamespace(enable_dsa_cache_layer_split=False)
        info = PrefillServerInfo(
            attn_tp_size=prefill_tp_size,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
        )

        manager._resolve_rank_mapping(info)
        return info

    def test_tp_four_to_two_metadata_targets_connected_prefill_pair(self) -> None:
        """Each TP2 decode rank sends metadata to its exact pair of TP4 ranks."""

        rank_zero = self._resolve_mapping(
            decode_tp_size=2,
            decode_tp_rank=0,
            prefill_tp_size=4,
        )
        rank_one = self._resolve_mapping(
            decode_tp_size=2,
            decode_tp_rank=1,
            prefill_tp_size=4,
        )

        self.assertEqual(rank_zero.target_tp_ranks, [0, 1])
        self.assertEqual(rank_one.target_tp_ranks, [2, 3])
        self.assertEqual(rank_zero.required_prefill_response_num, 2)
        self.assertEqual(rank_one.required_prefill_response_num, 2)

    def test_tp_two_to_four_metadata_targets_connected_prefill_rank(self) -> None:
        """Each TP4 decode rank targets the TP2 rank containing its shard."""

        target_ranks: list[list[int]] = []
        for decode_rank in range(4):
            info = self._resolve_mapping(
                decode_tp_size=4,
                decode_tp_rank=decode_rank,
                prefill_tp_size=2,
            )
            target_ranks.append(info.target_tp_ranks)
            self.assertEqual(info.required_dst_info_num, 2)
            self.assertEqual(info.required_prefill_response_num, 1)

        self.assertEqual(target_ranks, [[0], [0], [1], [1]])


if __name__ == "__main__":
    unittest.main()
