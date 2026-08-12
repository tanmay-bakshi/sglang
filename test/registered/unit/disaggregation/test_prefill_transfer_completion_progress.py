import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.prefill import (
    DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS,
    DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS,
    DISAGG_PREFILL_TRANSFER_PROGRESS_TIME_BUDGET_SECONDS,
    SchedulerDisaggregationPrefillMixin,
)
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

        def progress_may_continue(
            forward_completion: Mock,
            progress_deadline: float,
        ) -> bool:
            del progress_deadline
            return (
                len(scheduler.disagg_prefill_inflight_queue) > 0
                and not forward_completion.query()
            )

        scheduler.process_disagg_prefill_inflight_queue = process
        scheduler.disagg_prefill_progress_may_continue_on_all_ranks = (
            progress_may_continue
        )
        return scheduler

    def test_active_forward_hides_repeated_polls_until_transfer_completes(
        self,
    ) -> None:
        """A long forward keeps polling until the prior transfer is terminal."""

        scheduler = self._scheduler(terminal_poll=3)
        forward_completion = Mock()
        forward_completion.query.return_value = False

        with patch("sglang.srt.disaggregation.prefill.time.sleep") as sleep:
            SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
                scheduler,
                forward_completion,
            )

        self.assertEqual(scheduler.poll_count, 3)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(forward_completion.query.call_count, 2)
        self.assertEqual(
            sleep.call_args_list,
            [call(DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS)] * 2,
        )

    def test_rank_consensus_prevents_an_asymmetric_collective_exit(self) -> None:
        """The decision reduces local event state across both rank dimensions."""

        scheduler = SimpleNamespace(
            attn_tp_cpu_group=object(),
            attn_cp_cpu_group=object(),
            disagg_prefill_inflight_queue=[object()],
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

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_reduce",
                side_effect=all_reduce,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.time.monotonic",
                return_value=1.0,
            ),
        ):
            method = (
                SchedulerDisaggregationPrefillMixin
                .disagg_prefill_progress_may_continue_on_all_ranks
            )
            pending = method(scheduler, forward_completion, 2.0)

        self.assertFalse(pending)
        self.assertEqual(
            [group for _, _, group in groups],
            [scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group],
        )
        self.assertTrue(
            all(op is dist.ReduceOp.MIN for _, op, _ in groups)
        )

    def test_empty_local_queue_still_participates_in_exit_consensus(self) -> None:
        """Queue skew cannot make one rank skip the exit collectives."""

        scheduler = SimpleNamespace(
            attn_tp_cpu_group=object(),
            attn_cp_cpu_group=object(),
            disagg_prefill_inflight_queue=[],
        )
        forward_completion = Mock()
        groups: list[object] = []

        def all_reduce(
            tensor: torch.Tensor,
            *,
            op: object,
            group: object,
        ) -> None:
            del tensor, op
            groups.append(group)

        with patch(
            "sglang.srt.disaggregation.prefill.dist.all_reduce",
            side_effect=all_reduce,
        ):
            method = (
                SchedulerDisaggregationPrefillMixin
                .disagg_prefill_progress_may_continue_on_all_ranks
            )
            pending = method(scheduler, forward_completion, 2.0)

        self.assertFalse(pending)
        self.assertEqual(
            groups,
            [scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group],
        )
        forward_completion.query.assert_not_called()

    def test_completed_forward_bounds_nonterminal_transfer_progress(self) -> None:
        """Polling returns control as soon as the current forward completes."""

        scheduler = self._scheduler(terminal_poll=None)
        forward_completion = Mock()
        forward_completion.query.side_effect = [False, True]

        with patch("sglang.srt.disaggregation.prefill.time.sleep") as sleep:
            SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
                scheduler,
                forward_completion,
            )

        self.assertEqual(scheduler.poll_count, 2)
        self.assertEqual(len(scheduler.disagg_prefill_inflight_queue), 1)
        self.assertEqual(forward_completion.query.call_count, 2)
        self.assertEqual(
            sleep.call_args_list,
            [call(DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS)],
        )

    def test_terminal_initial_poll_does_not_sleep(self) -> None:
        """A transfer completed by the initial poll returns without delay."""

        scheduler = self._scheduler(terminal_poll=1)
        forward_completion = Mock()
        forward_completion.query.return_value = False

        with patch("sglang.srt.disaggregation.prefill.time.sleep") as sleep:
            SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
                scheduler,
                forward_completion,
            )

        self.assertEqual(scheduler.poll_count, 1)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        forward_completion.query.assert_not_called()
        sleep.assert_not_called()

    def test_expired_time_budget_still_reaches_rank_consensus(self) -> None:
        """Every rank enters the decision collective before a timed exit."""

        scheduler = SimpleNamespace(
            attn_tp_cpu_group=object(),
            attn_cp_cpu_group=object(),
            disagg_prefill_inflight_queue=[object()],
        )
        forward_completion = Mock()
        groups: list[object] = []

        def all_reduce(
            tensor: torch.Tensor,
            *,
            op: object,
            group: object,
        ) -> None:
            del tensor, op
            groups.append(group)

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_reduce",
                side_effect=all_reduce,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.time.monotonic",
                return_value=2.0,
            ),
        ):
            method = (
                SchedulerDisaggregationPrefillMixin
                .disagg_prefill_progress_may_continue_on_all_ranks
            )
            pending = method(scheduler, forward_completion, 1.0)

        self.assertFalse(pending)
        self.assertEqual(
            groups,
            [scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group],
        )
        forward_completion.query.assert_not_called()

    def test_absent_forward_preserves_one_opportunistic_poll(self) -> None:
        """An idle iteration does not spin without GPU work to hide it."""

        scheduler = self._scheduler(terminal_poll=None)

        SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
            scheduler,
            None,
        )

        self.assertEqual(scheduler.poll_count, 1)
        self.assertEqual(len(scheduler.disagg_prefill_inflight_queue), 1)

    def test_poll_budget_bounds_an_anomalously_long_forward(self) -> None:
        """A nonterminal transfer cannot poll Gloo without a fixed bound."""

        scheduler = self._scheduler(terminal_poll=None)
        forward_completion = Mock()
        forward_completion.query.return_value = False

        with patch("sglang.srt.disaggregation.prefill.time.sleep") as sleep:
            SchedulerDisaggregationPrefillMixin.progress_disagg_prefill_transfers_during_forward(
                scheduler,
                forward_completion,
            )

        self.assertEqual(
            scheduler.poll_count,
            DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS,
        )
        self.assertEqual(
            forward_completion.query.call_count,
            DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS - 1,
        )
        self.assertEqual(
            sleep.call_count,
            DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS - 1,
        )
        self.assertEqual(len(scheduler.disagg_prefill_inflight_queue), 1)

    def test_progress_budgets_have_one_consistent_upper_bound(self) -> None:
        """The poll cadence cannot outlive either registered budget."""

        repeated_poll_budget = (
            (DISAGG_PREFILL_TRANSFER_PROGRESS_MAX_POLLS - 1)
            * DISAGG_PREFILL_TRANSFER_PROGRESS_POLL_INTERVAL_SECONDS
        )
        self.assertLessEqual(
            repeated_poll_budget,
            DISAGG_PREFILL_TRANSFER_PROGRESS_TIME_BUDGET_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
