import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

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


if __name__ == "__main__":
    unittest.main()
