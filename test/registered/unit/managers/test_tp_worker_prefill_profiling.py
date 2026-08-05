import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TestTpWorkerPrefillProfiling(unittest.TestCase):
    """Validate that TP-worker phase spans are prefill-only."""

    def _run_forward(self, *, is_prefill: bool) -> list[str]:
        """Run the host-only TP-worker dispatch path.

        :param is_prefill: Whether the synthetic batch is a prefill batch.
        :returns: Emitted profiler span names.
        """
        spans: list[str] = []

        @contextmanager
        def capture_range(name: str) -> Iterator[None]:
            """Capture one synthetic profiler range.

            :param name: Range name.
            :yields: Control to the profiled block.
            """
            spans.append(name)
            yield

        forward_mode = MagicMock()
        forward_mode.is_prefill.return_value = is_prefill
        batch = MagicMock()
        batch.forward_mode = forward_mode
        batch.hicache_consumer_index = 0

        forward_batch = MagicMock()
        forward_batch.forward_mode = forward_mode
        logits_output = MagicMock()
        model_runner = MagicMock()
        model_runner.forward.return_value = SimpleNamespace(
            logits_output=logits_output,
            can_run_graph=False,
            expert_distribution_metrics=None,
        )

        worker = TpModelWorker.__new__(TpModelWorker)
        worker._model_runner = model_runner
        worker.dllm_algorithm = None
        worker.hicache_layer_transfer_counter = None
        worker.pp_group = SimpleNamespace(is_last_rank=False)

        with (
            patch.object(ForwardBatch, "init_new", return_value=forward_batch),
            patch(
                "sglang.srt.managers.tp_worker.profile_range", new=capture_range
            ),
        ):
            worker.forward_batch_generation(batch)

        model_runner.forward.assert_called_once_with(
            forward_batch,
            pp_proxy_tensors=None,
        )
        return spans

    def test_prefill_emits_host_dispatch_spans(self) -> None:
        """Prefill emits both host-dispatch phase spans."""
        self.assertEqual(
            self._run_forward(is_prefill=True),
            [
                "sglang.prefill.forward_batch_init",
                "sglang.prefill.model_runner_dispatch",
            ],
        )

    def test_decode_does_not_emit_prefill_spans(self) -> None:
        """Decode does not pay profiler-range dispatch overhead."""
        self.assertEqual(self._run_forward(is_prefill=False), [])


if __name__ == "__main__":
    unittest.main()
