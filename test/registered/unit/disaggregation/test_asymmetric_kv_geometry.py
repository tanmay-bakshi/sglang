"""CPU tests for asymmetric main-KV entry geometry admission."""

import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.common.asymmetric_kv_geometry import (
    require_uniform_asymmetric_kv_entry_geometry,
)
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeWriterManifest,
)
from sglang.srt.disaggregation.nixl.conn import KVArgsRegisterInfo, NixlKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAsymmetricKVGeometry(CustomTestCase):
    def test_uniform_entries_support_asymmetric_tp(self) -> None:
        require_uniform_asymmetric_kv_entry_geometry(
            source_item_lens=(128, 128),
            destination_item_lens=(512, 512),
            source_tp_size=4,
            destination_tp_size=1,
        )

    def test_heterogeneous_entries_support_equal_tp(self) -> None:
        require_uniform_asymmetric_kv_entry_geometry(
            source_item_lens=(512, 128),
            destination_item_lens=(512, 128),
            source_tp_size=1,
            destination_tp_size=1,
        )

    def test_heterogeneous_entries_require_per_entry_slicing(self) -> None:
        geometries = (
            ((128, 64), (512, 256)),
            ((128, 128), (512, 256)),
            ((128, 64), (512, 512)),
        )

        for source_item_lens, destination_item_lens in geometries:
            with (
                self.subTest(
                    source_item_lens=source_item_lens,
                    destination_item_lens=destination_item_lens,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "requires per-entry byte-range slicing",
                ),
            ):
                require_uniform_asymmetric_kv_entry_geometry(
                    source_item_lens=source_item_lens,
                    destination_item_lens=destination_item_lens,
                    source_tp_size=4,
                    destination_tp_size=1,
                )

    def test_aligned_entry_counts_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry counts differ"):
            require_uniform_asymmetric_kv_entry_geometry(
                source_item_lens=(128, 128),
                destination_item_lens=(512,),
                source_tp_size=4,
                destination_tp_size=1,
            )


class _RemoteHandle:
    def __init__(self, name: str) -> None:
        self.name = name


class TestNixlAsymmetricKVGeometryAdmission(CustomTestCase):
    @staticmethod
    def _manager(source_tp_size: int) -> NixlKVManager:
        handle = _RemoteHandle("decode-agent")
        manager = object.__new__(NixlKVManager)
        manager.agent = SimpleNamespace(
            add_remote_agent=MagicMock(return_value=handle),
            make_connection=MagicMock(),
        )
        manager.attn_tp_size = source_tp_size
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            kv_item_lens=[128, 128, 64, 64],
            kv_layer_ids=[5, 5, 0, 0],
        )
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager._packed_prefill_runtime = SimpleNamespace(
            writer_id=DecodeWriterManifest.for_tensor_parallel(
                source_tp_size
            ).writers[0]
        )
        manager._prefill_peer_lock = threading.RLock()
        manager._quarantined_remote_handles = set()
        manager.decode_kv_args_table = {}
        manager._prepare_payload_xfer = MagicMock()
        return manager

    @staticmethod
    def _registration(*, packed: bool) -> KVArgsRegisterInfo:
        return KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=39000,
            agent_name="decode-agent",
            agent_metadata=b"decode-metadata",
            dst_kv_ptrs=[0x1000, 0x2000, 0x3000, 0x4000],
            dst_kv_mem_kinds=["VRAM"] * 4,
            dst_aux_ptrs=[0x5000],
            dst_state_data_ptrs=[],
            gpu_id=1,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=512,
            dst_kv_item_lens=[512, 512, 256, 256],
            dst_kv_layer_ids=[5, 5, 0, 0],
            process_generation=str(uuid.uuid4()),
            packed_advertisement=object() if packed else None,
        )

    def test_legacy_registration_rejects_before_native_peer_creation(self) -> None:
        for source_tp_size in (2, 4):
            manager = self._manager(source_tp_size)
            with (
                self.subTest(source_tp_size=source_tp_size),
                self.assertRaisesRegex(
                    ValueError,
                    "requires per-entry byte-range slicing",
                ),
            ):
                manager._add_remote_peer(self._registration(packed=False))

            manager.agent.add_remote_agent.assert_not_called()
            manager._prepare_payload_xfer.assert_not_called()
            self.assertEqual(manager.decode_kv_args_table, {})

    def test_packed_registration_accepts_per_entry_geometry(self) -> None:
        for source_tp_size in (2, 4):
            with self.subTest(source_tp_size=source_tp_size):
                manager = self._manager(source_tp_size)
                registration = self._registration(packed=True)

                manager._add_remote_peer(registration)

                manager.agent.add_remote_agent.assert_called_once_with(
                    b"decode-metadata"
                )
                manager.agent.make_connection.assert_called_once_with(
                    manager.agent.add_remote_agent.return_value
                )
                manager._prepare_payload_xfer.assert_not_called()
                self.assertIs(
                    manager.decode_kv_args_table["decode-agent"], registration
                )


if __name__ == "__main__":
    unittest.main()
