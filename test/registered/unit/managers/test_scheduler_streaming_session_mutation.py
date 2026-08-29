import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.io_struct import (
    GetSessionInfoReqErrorOutput,
    GetSessionInfoReqInput,
)
from sglang.srt.managers.scheduler import (
    Scheduler,
    _validate_streaming_session_topology,
)
from sglang.srt.session.errors import StreamingSessionInfoUnavailableError

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEmptyStreamingSessionMutation(CustomTestCase):
    """Exercise the scheduler's no-forward completion path."""

    def test_finishes_without_queueing_or_model_resources(self):
        """A zero-context mutation completes without owning model resources."""
        finished = False

        def update_finish_state() -> None:
            """Mark the synthetic request finished."""
            nonlocal finished
            finished = True

        session = SimpleNamespace(streaming=True, finish_req=MagicMock())
        time_stats = MagicMock()
        req = SimpleNamespace(
            session=session,
            sampling_params=SimpleNamespace(max_new_tokens=0),
            req_pool_idx=None,
            kv=None,
            time_stats=time_stats,
            update_finish_state=update_finish_state,
            finished=lambda: finished,
            return_logprob=False,
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.output_streamer = MagicMock()

        Scheduler._finish_empty_streaming_session_mutation(scheduler, req)

        self.assertTrue(finished)
        session.finish_req.assert_called_once_with(req)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        time_stats.set_wait_queue_entry_time.assert_called_once()
        timestamp = time_stats.set_wait_queue_entry_time.call_args.args[0]
        time_stats.set_forward_entry_time.assert_called_once_with(timestamp)
        time_stats.set_prefill_finished_time.assert_called_once_with(timestamp)
        time_stats.set_completion_time.assert_called_once_with(timestamp)


class TestStreamingSessionTopologyAndMetrics(CustomTestCase):
    """Exercise topology fail-closed and exactly-once conflict accounting."""

    def test_builtin_dp_fails_closed_at_scheduler_construction(self):
        """Reject a topology that cannot preserve one durable session owner."""
        with self.assertRaisesRegex(
            ValueError,
            "Streaming sessions require dp_size == 1",
        ):
            _validate_streaming_session_topology(
                SimpleNamespace(enable_streaming_session=True),
                ParallelState.trivial(dp_rank=0, dp_size=2),
            )

        _validate_streaming_session_topology(
            SimpleNamespace(enable_streaming_session=True),
            ParallelState.trivial(dp_rank=0, dp_size=1),
        )
        _validate_streaming_session_topology(
            SimpleNamespace(enable_streaming_session=False),
            ParallelState.trivial(dp_rank=0, dp_size=8),
        )

    def test_tp_conflict_counter_increments_on_exactly_one_rank(self):
        """Count once when attention-DP repeats its local stats rank."""
        time_stats_by_rank = [MagicMock() for _ in range(4)]

        for tp_rank, time_stats in enumerate(time_stats_by_rank):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.ps = ParallelState.trivial(
                tp_rank=tp_rank,
                tp_size=4,
                dp_rank=None,
                dp_size=2,
                attn_tp_rank=tp_rank % 2,
                attn_tp_size=2,
                attn_dp_rank=tp_rank // 2,
                attn_dp_size=2,
            )
            req = SimpleNamespace(time_stats=time_stats)

            Scheduler._record_streaming_session_idempotency_conflict(scheduler, req)

        self.assertEqual(
            sum(
                stats.increment_streaming_session_idempotency_conflict.call_count
                for stats in time_stats_by_rank
            ),
            1,
        )
        time_stats_by_rank[
            0
        ].increment_streaming_session_idempotency_conflict.assert_called_once_with()
        for time_stats in time_stats_by_rank[1:]:
            time_stats.increment_streaming_session_idempotency_conflict.assert_not_called()

    def test_non_streaming_info_error_is_typed_on_unique_output_rank(self):
        """Map controller rejection to one correlated IPC error."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = ParallelState.trivial()
        scheduler.session_controller = MagicMock()
        scheduler.session_controller.get_info.side_effect = (
            StreamingSessionInfoUnavailableError(
                "Session ordinary-session is not a streaming session."
            )
        )
        request = GetSessionInfoReqInput(
            correlation_id="correlation-a",
            session_id="ordinary-session",
        )

        output = Scheduler.get_session_info(scheduler, request)

        self.assertEqual(
            output,
            GetSessionInfoReqErrorOutput(
                correlation_id="correlation-a",
                message="Session ordinary-session is not a streaming session.",
            ),
        )

        scheduler.ps = ParallelState.trivial(
            tp_rank=2,
            tp_size=4,
            dp_rank=None,
            dp_size=2,
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_dp_rank=1,
            attn_dp_size=2,
        )
        scheduler.session_controller.reset_mock()

        self.assertIsNone(Scheduler.get_session_info(scheduler, request))
        scheduler.session_controller.get_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
