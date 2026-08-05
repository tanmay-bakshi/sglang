import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import (
    SingletonPollProgressPolicy,
    _reduce_poll_values,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSingletonPollProgressPolicy(unittest.TestCase):
    """Verify bounded TP1 yields and mandatory multi-rank synchronization."""

    @staticmethod
    def _group(size: int) -> Mock:
        """Create a process-group mock with a stable size.

        :param size: Reported process-group size.
        :returns: Configured process-group mock.
        """

        group = Mock(spec=dist.ProcessGroup)
        group.size.return_value = size
        return group

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    @patch("sglang.srt.disaggregation.utils.torch.tensor")
    @patch("sglang.srt.disaggregation.utils.time.perf_counter_ns")
    @patch("sglang.srt.disaggregation.utils.os.sched_yield")
    def test_explicit_yield_mode_calls_scheduler_yield_after_one_stalled_poll(
        self,
        sched_yield_mock: Mock,
        clock_mock: Mock,
        tensor_mock: Mock,
        all_reduce_mock: Mock,
    ) -> None:
        """Verify repeated singleton polls invoke a cooperative scheduler
        yield.
        """

        clock_mock.side_effect = [100, 175]
        policy = SingletonPollProgressPolicy(stream_name="transfer", mode="sched_yield")
        group = self._group(1)

        self.assertEqual(_reduce_poll_values([2], group, policy), [2])
        self.assertEqual(_reduce_poll_values([2], group, policy), [2])

        sched_yield_mock.assert_called_once_with()
        tensor_mock.assert_not_called()
        all_reduce_mock.assert_not_called()
        self.assertEqual(
            policy.snapshot().as_dict(),
            {
                "polls": 2,
                "progress_transitions": 0,
                "stalled_polls": 1,
                "yields": 1,
                "skipped_collectives": 2,
                "cumulative_requested_sleep_us": 0,
                "cumulative_observed_yield_ns": 75,
                "max_consecutive_stalled_polls": 1,
            },
        )

    @patch("sglang.srt.disaggregation.utils.time.perf_counter_ns")
    @patch("sglang.srt.disaggregation.utils.time.sleep")
    def test_complete_vector_progress_resets_stalled_cadence(
        self,
        sleep_mock: Mock,
        clock_mock: Mock,
    ) -> None:
        """Verify value or length changes reset the bounded stall cadence."""

        clock_mock.side_effect = [200, 260]
        policy = SingletonPollProgressPolicy(
            stream_name="preallocation",
            mode="yield",
            yield_cadence=2,
        )

        policy.observe([0])
        policy.observe([0])
        policy.observe([0, 0])
        policy.observe([0, 1])
        policy.observe([0, 1])
        policy.observe([0, 1])

        sleep_mock.assert_called_once_with(0)
        stats = policy.snapshot()
        self.assertEqual(stats.polls, 6)
        self.assertEqual(stats.progress_transitions, 2)
        self.assertEqual(stats.stalled_polls, 3)
        self.assertEqual(stats.yields, 1)
        self.assertEqual(stats.max_consecutive_stalled_polls, 2)

    @patch("sglang.srt.disaggregation.utils.time.perf_counter_ns")
    @patch("sglang.srt.disaggregation.utils.time.sleep")
    def test_positive_sleep_records_requested_and_observed_duration(
        self,
        sleep_mock: Mock,
        clock_mock: Mock,
    ) -> None:
        """Verify configured sleep and both cumulative duration counters."""

        clock_mock.side_effect = [1_000, 1_175]
        policy = SingletonPollProgressPolicy(
            stream_name="transfer",
            mode="yield",
            yield_sleep_us=25,
        )

        policy.observe([2])
        policy.observe([2])

        sleep_mock.assert_called_once_with(25 / 1_000_000)
        stats = policy.snapshot()
        self.assertEqual(stats.cumulative_requested_sleep_us, 25)
        self.assertEqual(stats.cumulative_observed_yield_ns, 175)

    @patch("sglang.srt.disaggregation.utils.time.sleep")
    def test_mark_idle_resets_comparison_without_resetting_counters(
        self,
        sleep_mock: Mock,
    ) -> None:
        """Verify a new busy period starts without inheriting stale cadence."""

        policy = SingletonPollProgressPolicy(stream_name="transfer", mode="yield")
        policy.observe([2])
        policy.mark_idle()
        policy.observe([2])

        sleep_mock.assert_not_called()
        stats = policy.snapshot()
        self.assertEqual(stats.polls, 2)
        self.assertEqual(stats.progress_transitions, 0)
        self.assertEqual(stats.stalled_polls, 0)
        self.assertEqual(stats.yields, 0)

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    @patch("sglang.srt.disaggregation.utils.torch.tensor")
    def test_default_gloo_mode_observes_and_retains_collective(
        self,
        tensor_mock: Mock,
        all_reduce_mock: Mock,
    ) -> None:
        """Verify production-default Gloo mode records comparable poll counters."""

        group = self._group(1)
        policy = SingletonPollProgressPolicy(stream_name="transfer")
        reduced_tensor = Mock()
        reduced_tensor.tolist.return_value = [2]
        tensor_mock.return_value = reduced_tensor

        self.assertEqual(_reduce_poll_values([2], group, policy), [2])

        all_reduce_mock.assert_called_once_with(
            reduced_tensor,
            op=dist.ReduceOp.MIN,
            group=group,
        )
        state = policy.diagnostic_state()
        self.assertEqual(state["mode"], "gloo")
        self.assertEqual(state["polls"], 1)
        self.assertEqual(state["skipped_collectives"], 0)
        self.assertEqual(state["yields"], 0)

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    @patch("sglang.srt.disaggregation.utils.torch.tensor")
    def test_singleton_without_explicit_policy_retains_gloo_collective(
        self,
        tensor_mock: Mock,
        all_reduce_mock: Mock,
    ) -> None:
        """Verify an absent policy cannot silently enable the rejected shortcut."""

        group = self._group(1)
        reduced_tensor = Mock()
        reduced_tensor.tolist.return_value = [2]
        tensor_mock.return_value = reduced_tensor

        self.assertEqual(_reduce_poll_values([2], group), [2])

        tensor_mock.assert_called_once_with([2], dtype=torch.uint8, device="cpu")
        all_reduce_mock.assert_called_once_with(
            reduced_tensor,
            op=dist.ReduceOp.MIN,
            group=group,
        )

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    @patch("sglang.srt.disaggregation.utils.torch.tensor")
    def test_multi_rank_always_reduces_and_does_not_advance_policy(
        self,
        tensor_mock: Mock,
        all_reduce_mock: Mock,
    ) -> None:
        """Verify multi-rank MIN reduction remains mandatory with a policy present."""

        group = self._group(2)
        policy = SingletonPollProgressPolicy(stream_name="transfer", mode="yield")
        reduced_tensor = Mock()
        reduced_tensor.tolist.return_value = [1, 2]
        tensor_mock.return_value = reduced_tensor

        self.assertEqual(_reduce_poll_values([2, 2], group, policy), [1, 2])

        tensor_mock.assert_called_once_with([2, 2], dtype=torch.uint8, device="cpu")
        all_reduce_mock.assert_called_once_with(
            reduced_tensor,
            op=dist.ReduceOp.MIN,
            group=group,
        )
        self.assertEqual(policy.snapshot().polls, 0)

    def test_live_diagnostics_include_both_queue_snapshots(self) -> None:
        """Verify existing server diagnostics can serialize point-in-time counters."""

        preallocation = SingletonPollProgressPolicy(stream_name="preallocation")
        transfer = SingletonPollProgressPolicy(stream_name="transfer", mode="yield")
        preallocation.observe([0])
        transfer.observe([2])
        scheduler = SchedulerDisaggregationDecodeMixin()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            tp1_poll_progress_policy=preallocation
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            tp1_poll_progress_policy=transfer
        )

        stats = scheduler.get_decode_poll_progress_stats()

        self.assertEqual(stats["preallocation"]["polls"], 1)
        self.assertEqual(stats["transfer"]["polls"], 1)
        self.assertEqual(stats["preallocation"]["mode"], "gloo")
        self.assertEqual(stats["transfer"]["mode"], "yield")
        self.assertEqual(stats["preallocation"]["skipped_collectives"], 0)
        self.assertEqual(stats["transfer"]["skipped_collectives"], 1)

    def test_unsafe_configuration_is_rejected(self) -> None:
        """Verify cadence and sleep bounds cannot disable cooperative progress."""

        with self.assertRaises(ValueError):
            SingletonPollProgressPolicy(stream_name="transfer", yield_cadence=0)
        with self.assertRaises(ValueError):
            SingletonPollProgressPolicy(stream_name="transfer", yield_sleep_us=-1)
        with self.assertRaises(ValueError):
            SingletonPollProgressPolicy(
                stream_name="transfer", mode="sched_yield", yield_sleep_us=1
            )
        with self.assertRaises(ValueError):
            SingletonPollProgressPolicy(stream_name="transfer", mode="invalid")


if __name__ == "__main__":
    unittest.main()
