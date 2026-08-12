import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillTransferCompletionProgress(unittest.TestCase):
    """Verify transfer completion is observed during an enqueued forward."""

    @staticmethod
    def _scheduler(terminal_poll: int | None) -> SimpleNamespace:
        """Build a scheduler double with deterministic transfer progress.

        :param terminal_poll: Poll that empties the in-flight queue, or
            ``None`` when the transfer remains active.
        :returns: Scheduler state and a poll-counting progress function.
        """

        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[object()],
            poll_count=0,
        )

        def process() -> None:
            scheduler.poll_count += 1
            if terminal_poll == scheduler.poll_count:
                scheduler.disagg_prefill_inflight_queue.clear()

        def forward_pending(forward_completion: Mock) -> bool:
            return not forward_completion.query()

        scheduler.process_disagg_prefill_inflight_queue = process
        scheduler.disagg_prefill_forward_pending_on_all_ranks = forward_pending
        return scheduler

    def test_active_forward_hides_repeated_polls_until_transfer_completes(
        self,
    ) -> None:
        """A long forward keeps polling until the prior transfer is terminal."""

        scheduler = self._scheduler(terminal_poll=3)
        forward_completion = Mock()
        forward_completion.query.return_value = False

        SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
            scheduler,
            forward_completion,
        )

        self.assertEqual(scheduler.poll_count, 3)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(forward_completion.query.call_count, 2)

    def test_rank_consensus_prevents_an_asymmetric_collective_exit(self) -> None:
        """The decision reduces local event state across both rank dimensions."""

        scheduler = SimpleNamespace(
            attn_tp_cpu_group=object(),
            attn_cp_cpu_group=object(),
        )
        forward_completion = Mock()
        forward_completion.query.return_value = False
        groups: list[tuple[torch.Tensor, object, object]] = []

        def all_reduce(
            tensor: torch.Tensor,
            *,
            op: object,
            group: object,
        ) -> None:
            groups.append((tensor.clone(), op, group))
            if group is scheduler.attn_cp_cpu_group:
                tensor.fill_(0)

        with patch(
            "sglang.srt.disaggregation.prefill.dist.all_reduce",
            side_effect=all_reduce,
        ):
            method = (
                SchedulerDisaggregationPrefillMixin
                .disagg_prefill_forward_pending_on_all_ranks
            )
            pending = method(scheduler, forward_completion)

        self.assertFalse(pending)
        self.assertEqual(
            [group for _, _, group in groups],
            [scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group],
        )
        self.assertTrue(
            all(op is dist.ReduceOp.MIN for _, op, _ in groups)
        )

    def test_completed_forward_bounds_nonterminal_transfer_progress(self) -> None:
        """Polling returns control as soon as the current forward completes."""

        scheduler = self._scheduler(terminal_poll=None)
        forward_completion = Mock()
        forward_completion.query.side_effect = [False, True]

        SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
            scheduler,
            forward_completion,
        )

        self.assertEqual(scheduler.poll_count, 2)
        self.assertEqual(len(scheduler.disagg_prefill_inflight_queue), 1)
        self.assertEqual(forward_completion.query.call_count, 2)

    def test_absent_forward_preserves_one_opportunistic_poll(self) -> None:
        """An idle iteration does not spin without GPU work to hide it."""

        scheduler = self._scheduler(terminal_poll=None)

        SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
            scheduler,
            None,
        )

        self.assertEqual(scheduler.poll_count, 1)
        self.assertEqual(len(scheduler.disagg_prefill_inflight_queue), 1)


if __name__ == "__main__":
    unittest.main()
