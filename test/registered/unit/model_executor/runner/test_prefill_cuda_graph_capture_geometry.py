import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _build_context_aware_capture_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeGraphSlot:
    """Minimal static-buffer slot used by capture-geometry tests."""

    _buffer: torch.Tensor

    def __init__(self, buffer: torch.Tensor) -> None:
        """Initialize a test slot.

        :param buffer: Tensor exposed through :meth:`slice_for`.
        """

        self._buffer = buffer

    def slice_for(self, batch_size: int, num_tokens: int) -> torch.Tensor:
        """Return the token-axis slice expected by the prefill runner.

        :param batch_size: Live request count, unused for token-axis slots.
        :param num_tokens: Requested token-axis extent.
        :returns: Prefix view of the test buffer.
        """

        del batch_size
        return self._buffer[:num_tokens]


class _FakeBufferRegistry:
    """Token-only buffer registry for a CPU capture_prepare call."""

    _slots: dict[str, _FakeGraphSlot]

    def __init__(self, num_tokens: int) -> None:
        """Allocate the three required prefill token slots.

        :param num_tokens: Maximum token extent exposed by the registry.
        """

        self._slots = {
            "input_ids": _FakeGraphSlot(torch.zeros(num_tokens, dtype=torch.int64)),
            "out_cache_loc": _FakeGraphSlot(torch.zeros(num_tokens, dtype=torch.int64)),
            "positions": _FakeGraphSlot(torch.zeros(num_tokens, dtype=torch.int64)),
        }

    def has_slot(self, name: str) -> bool:
        """Return whether a named test slot exists.

        :param name: Registry slot name.
        :returns: Whether the slot is present.
        """

        return name in self._slots

    def get_slot(self, name: str) -> _FakeGraphSlot:
        """Return one required test slot.

        :param name: Registry slot name.
        :returns: Registered test slot.
        """

        return self._slots[name]


class _FakeTboPlugin:
    """No-op TBO capture hook."""

    captured_shapes: list[tuple[int, int]]

    def __init__(self) -> None:
        """Initialize an empty capture log."""

        self.captured_shapes = []

    def capture_one_batch_size(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> None:
        """Record the request and token dimensions.

        :param forward_batch: Synthetic forward batch.
        :param num_tokens: Aggregate capture tokens.
        """

        batch_size = int(forward_batch.batch_size)
        self.captured_shapes.append((batch_size, num_tokens))


class TestContextAwarePrefillCaptureGeometry(CustomTestCase):
    def _make_runner(self) -> PrefillCudaGraphRunner:
        """Build a minimally initialized tc-piecewise runner.

        :returns: CPU-only runner suitable for geometry gates.
        """

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [8192, 16384]
        runner.max_num_tokens = 16384
        runner.max_bs = 32
        runner._tc_piecewise_max_sequence_tokens = 8450
        runner._tc_piecewise_capture_geometries = {
            num_tokens: _build_context_aware_capture_geometry(
                num_tokens, 8450, runner.max_bs
            )
            for num_tokens in runner.capture_num_tokens
        }
        return runner

    def _make_forward_batch(
        self,
        *,
        raw_num_tokens: int,
        sequence_lengths: list[int] | None,
        extend_lengths: list[int] | None,
    ) -> SimpleNamespace:
        """Build the host metadata consumed by graph eligibility.

        :param raw_num_tokens: Aggregate unpadded token count.
        :param sequence_lengths: Per-request total sequence lengths.
        :param extend_lengths: Per-request extension lengths.
        :returns: Forward-batch-shaped namespace for eligibility tests.
        """

        batch_size = len(extend_lengths) if extend_lengths is not None else 2
        return SimpleNamespace(
            batch_size=batch_size,
            input_embeds=None,
            replace_embeds=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=range(raw_num_tokens),
            seq_lens_cpu=(
                torch.tensor(sequence_lengths, dtype=torch.int64)
                if sequence_lengths is not None
                else None
            ),
            extend_seq_lens_cpu=extend_lengths,
        )

    def test_m16k_capture_uses_two_legal_page_table_rows(self) -> None:
        geometry = _build_context_aware_capture_geometry(16384, 8450, 32)

        self.assertEqual(geometry.sequence_lengths, (8192, 8192))
        self.assertEqual(geometry.start_locations, (0, 8192))
        self.assertEqual(geometry.num_tokens, 16384)
        runtime_page_table_width = (8450 + 64 - 1) // 64
        self.assertEqual(runtime_page_table_width, 133)
        self.assertEqual(
            tuple((length + 64 - 1) // 64 for length in geometry.sequence_lengths),
            (128, 128),
        )

    def test_existing_buckets_keep_one_request_geometry(self) -> None:
        for num_tokens in (1024, 2048, 4096, 8192):
            with self.subTest(num_tokens=num_tokens):
                geometry = _build_context_aware_capture_geometry(num_tokens, 8450, 32)
                self.assertEqual(geometry.sequence_lengths, (num_tokens,))
                self.assertEqual(geometry.start_locations, (0,))

    def test_illegal_capture_geometries_fail_closed(self) -> None:
        invalid_inputs = (
            (0, 8450, 32),
            (16384, 0, 32),
            (16384, 8450, 0),
            (16384, 8450, 1),
        )
        for num_tokens, max_sequence_tokens, max_request_slots in invalid_inputs:
            with (
                self.subTest(
                    num_tokens=num_tokens,
                    max_sequence_tokens=max_sequence_tokens,
                    max_request_slots=max_request_slots,
                ),
                self.assertRaises(ValueError),
            ):
                _build_context_aware_capture_geometry(
                    num_tokens,
                    max_sequence_tokens,
                    max_request_slots,
                )

    def test_capture_prepare_materializes_two_request_metadata(self) -> None:
        runner = self._make_runner()
        runner._capture_req_slots = 1
        runner.device = torch.device("cpu")
        runner._prefill_static_buffers = None
        runner.buffer_registry = _FakeBufferRegistry(16384)
        runner.require_mlp_tp_gather = False
        runner.require_attn_tp_gather = False
        runner.dp_size = 1
        runner.mamba_track_enabled = False
        runner.static_draft_hidden_states = None
        runner._capture_lora = False
        runner.capture_return_pooled_hidden_states = False
        runner.tbo_plugin = _FakeTboPlugin()
        attention_backend = object()
        runner.model_runner = SimpleNamespace(attn_backend=attention_backend)

        with patch.object(
            runner,
            "_next_token_logits_buffer",
            return_value=torch.zeros((2, 1), dtype=torch.float32),
        ):
            forward_batch, returned_backend = runner.capture_prepare(16384)

        self.assertEqual(forward_batch.batch_size, 2)
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [8192, 8192])
        self.assertEqual(forward_batch.seq_lens.tolist(), [8192, 8192])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [8192, 8192])
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [8192, 8192])
        self.assertEqual(forward_batch.extend_start_loc.tolist(), [0, 8192])
        self.assertEqual(forward_batch.seq_lens_sum, 16384)
        self.assertIs(returned_backend, attention_backend)
        self.assertEqual(runner.tbo_plugin.captured_shapes, [(2, 16384)])

    def test_m16k_replay_accepts_legal_live_request_geometry(self) -> None:
        runner = self._make_runner()
        exact = self._make_forward_batch(
            raw_num_tokens=16384,
            sequence_lengths=[8192, 8192],
            extend_lengths=[8192, 8192],
        )
        differently_partitioned = self._make_forward_batch(
            raw_num_tokens=9000,
            sequence_lengths=[8450, 550],
            extend_lengths=[8450, 550],
        )

        self.assertTrue(runner.can_run_graph(exact))
        self.assertTrue(runner.can_run_graph(differently_partitioned))

    def test_m16k_replay_rejects_illegal_or_incomplete_metadata(self) -> None:
        runner = self._make_runner()
        over_context = self._make_forward_batch(
            raw_num_tokens=9000,
            sequence_lengths=[8451, 549],
            extend_lengths=[8451, 549],
        )
        mismatched_total = self._make_forward_batch(
            raw_num_tokens=16384,
            sequence_lengths=[8192, 8192],
            extend_lengths=[8192, 8000],
        )
        missing_host_lengths = self._make_forward_batch(
            raw_num_tokens=16384,
            sequence_lengths=None,
            extend_lengths=[8192, 8192],
        )

        self.assertFalse(runner.can_run_graph(over_context))
        self.assertFalse(runner.can_run_graph(mismatched_total))
        self.assertFalse(runner.can_run_graph(missing_host_lengths))

    def test_existing_bucket_replay_retains_previous_metadata_contract(self) -> None:
        runner = self._make_runner()
        existing_bucket = self._make_forward_batch(
            raw_num_tokens=8192,
            sequence_lengths=None,
            extend_lengths=None,
        )
        existing_bucket.batch_size = 1

        self.assertTrue(runner.can_run_graph(existing_bucket))

    def test_sequence_capacity_uses_tighter_runtime_bound(self) -> None:
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(context_len=8450),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((32, 8512), dtype=torch.int32)
            ),
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_capture_sequence_capacity(model_runner),
            8450,
        )

        model_runner.req_to_token_pool.req_to_token = torch.empty(
            (32, 8000), dtype=torch.int32
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_capture_sequence_capacity(model_runner),
            8000,
        )


if __name__ == "__main__":
    unittest.main()
