import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.io_struct import (
    GetSessionInfoReqErrorOutput,
    GetSessionInfoReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import (
    Scheduler,
    _validate_streaming_session_topology,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.errors import StreamingSessionInfoUnavailableError
from sglang.srt.session.session_controller import Session

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tokenized_request(
    rid: str,
    input_ids: list[int],
    *,
    max_new_tokens: int = 1,
    truncate_to: int | None = None,
    commit_to: int | None = None,
) -> TokenizedGenerateReqInput:
    """Build one real scheduler input for a streaming-session turn.

    :param rid: Request identity.
    :param input_ids: Raw token delta.
    :param max_new_tokens: Decode limit.
    :param truncate_to: Optional durable rollback target.
    :param commit_to: Optional rollback floor.
    :returns: Canonical tokenized scheduler input.
    """
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(
            id="session",
            truncate_to=truncate_to,
            commit_to=commit_to,
        ),
    )


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
            is_holding_kv=False,
            kv=ReqKvInfo(),
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


class TestStreamingSessionAdmission(CustomTestCase):
    """Exercise the scheduler-owned transaction boundary."""

    @staticmethod
    def _request(*, admitted: bool = False) -> SimpleNamespace:
        session = SimpleNamespace(
            streaming=True,
            inflight=True,
            durable_tip=12,
            slot_committed=10,
            abort_req=MagicMock(),
            commit_prepared_req=MagicMock(),
        )
        req = SimpleNamespace(
            rid="session-request",
            streaming_session_owns_inflight=True,
            session=session,
            streaming_session_admitted=admitted,
            priority=None,
            finished_reason=None,
            to_finish=None,
            origin_input_ids=[1, 2, 3],
            return_logprob=False,
            mamba_pool_idx=None,
            time_stats=SimpleNamespace(
                trace_ctx=MagicMock(),
                wait_queue_entry_time=0.0,
                set_wait_queue_entry_time=MagicMock(),
            ),
        )
        req.finished = lambda: req.finished_reason is not None

        def abort(request: SimpleNamespace) -> None:
            assert request is req
            request.streaming_session_owns_inflight = False
            session.inflight = False

        session.abort_req.side_effect = abort
        session.commit_prepared_req.side_effect = lambda request, tree_cache: setattr(
            request, "streaming_session_admitted", True
        )
        return req

    def test_admission_commits_once(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        req = self._request()

        Scheduler._admit_streaming_session_request(scheduler, req)
        Scheduler._admit_streaming_session_request(scheduler, req)

        req.session.commit_prepared_req.assert_called_once_with(
            req, scheduler.tree_cache
        )
        self.assertTrue(req.streaming_session_admitted)
        req.session.abort_req.assert_not_called()

    def test_queue_rejection_precedes_transaction_commit(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler._set_or_validate_priority = MagicMock(return_value=True)
        scheduler._abort_on_queued_limit = MagicMock(return_value=True)
        scheduler.waiting_queue = []
        req = self._request()

        Scheduler._add_request_to_queue(scheduler, req)

        req.session.commit_prepared_req.assert_not_called()
        self.assertFalse(req.streaming_session_admitted)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_pending_finish_detaches_without_committing(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler._set_or_validate_priority = MagicMock(return_value=True)
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler._prefetch_kvcache = MagicMock()
        scheduler.waiting_queue = []
        req = self._request()
        session = req.session
        req.to_finish = FINISH_ABORT("rejected before queue admission")

        Scheduler._add_request_to_queue(scheduler, req)

        session.commit_prepared_req.assert_not_called()
        session.abort_req.assert_called_once_with(req)
        self.assertIsNone(req.session)
        self.assertFalse(req.streaming_session_admitted)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_concurrent_rejection_cannot_release_active_owner(self):
        session = Session(
            capacity_of_str_len=0,
            session_id="session",
            streaming=True,
        )
        active = session.create_req(
            _tokenized_request("active", [1]),
            tokenizer=None,
            vocab_size=1024,
        )
        with get_parallel().override(tp_rank=0):
            rejected = session.create_req(
                _tokenized_request("rejected", [1]),
                tokenizer=None,
                vocab_size=1024,
            )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler._set_or_validate_priority = MagicMock(return_value=True)
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler._prefetch_kvcache = MagicMock()
        scheduler.waiting_queue = []

        Scheduler._add_request_to_queue(scheduler, rejected)

        self.assertTrue(active.streaming_session_owns_inflight)
        self.assertTrue(session._inflight)
        self.assertIsNone(rejected.session)
        self.assertFalse(rejected.streaming_session_owns_inflight)
        with get_parallel().override(tp_rank=0):
            third = session.create_req(
                _tokenized_request("third", [1]),
                tokenizer=None,
                vocab_size=1024,
            )
        self.assertIsNotNone(third.to_finish)
        self.assertIn("already has an active request", third.to_finish.message)
        session.abort_req(active)

    def test_queue_admitted_mutation_survives_abort_before_prefill(self):
        session = Session(
            capacity_of_str_len=0,
            session_id="session",
            streaming=True,
            manual_commit=True,
        )
        first = session.create_req(
            _tokenized_request("first", [1, 2, 3, 4, 5, 6]),
            tokenizer=None,
            vocab_size=1024,
        )
        first.output_ids.extend([7, 8])
        first._refresh_fill_ids()
        session.finish_req(first)

        cache = SimpleNamespace(slot_committed=8, floor=0)

        def truncate_session(session_id: str, target: int) -> None:
            self.assertEqual(session_id, "session")
            cache.slot_committed = target

        def commit_session(session_id: str, floor: int) -> None:
            self.assertEqual(session_id, "session")
            cache.floor = floor

        cache.truncate_session = truncate_session
        cache.commit_session = commit_session
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = cache
        scheduler._set_or_validate_priority = MagicMock(return_value=True)
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler._prefetch_kvcache = MagicMock()
        scheduler.waiting_queue = []
        queued = session.create_req(
            _tokenized_request(
                "queued",
                [20, 21],
                truncate_to=6,
                commit_to=6,
            ),
            tokenizer=None,
            vocab_size=1024,
        )

        Scheduler._add_request_to_queue(scheduler, queued)

        expected_context = [1, 2, 3, 4, 5, 6, 20, 21]
        self.assertEqual(scheduler.waiting_queue, [queued])
        self.assertEqual(session.current_tip(), len(expected_context))
        self.assertEqual(cache.slot_committed, 6)
        self.assertEqual(cache.floor, 6)

        Scheduler._abort_queued_streaming_session(queued, "Aborted")

        self.assertTrue(queued.finished())
        self.assertFalse(queued.streaming_session_owns_inflight)
        self.assertFalse(session._inflight)
        self.assertEqual(session.current_tip(), len(expected_context))
        self.assertEqual(session.floor, 6)
        self.assertEqual(cache.slot_committed, 6)
        self.assertEqual(cache.floor, 6)

        continuation = session.create_req(
            _tokenized_request("continuation", [22]),
            tokenizer=None,
            vocab_size=1024,
        )
        self.assertEqual(
            list(continuation.origin_input_ids),
            expected_context + [22],
        )
        session.abort_req(continuation)

    def test_queued_abort_terminalizes_without_rewriting_durable_state(self):
        req = self._request(admitted=True)
        session = req.session
        origin_before = list(req.origin_input_ids)
        durable_tip_before = session.durable_tip
        slot_committed_before = session.slot_committed

        Scheduler._abort_queued_streaming_session(req, "Aborted")

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.origin_input_ids, origin_before)
        self.assertEqual(session.durable_tip, durable_tip_before)
        self.assertEqual(session.slot_committed, slot_committed_before)
        self.assertFalse(session.inflight)
        self.assertIs(req.session, session)

    def test_queue_acceptance_commits_before_publication(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = MagicMock()
        scheduler._set_or_validate_priority = MagicMock(return_value=True)
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler.waiting_queue = []
        req = self._request()

        def assert_admitted_before_prefetch(request) -> None:
            self.assertIs(request, req)
            self.assertTrue(request.streaming_session_admitted)

        scheduler._prefetch_kvcache = MagicMock(
            side_effect=assert_admitted_before_prefetch
        )

        Scheduler._add_request_to_queue(scheduler, req)

        req.session.commit_prepared_req.assert_called_once_with(
            req, scheduler.tree_cache
        )
        scheduler._prefetch_kvcache.assert_called_once_with(req)
        self.assertEqual(scheduler.waiting_queue, [req])
        req.time_stats.set_wait_queue_entry_time.assert_called_once_with()

    def test_pre_admission_rejection_releases_session_owner(self):
        req = self._request()
        session = req.session

        Scheduler._release_unadmitted_streaming_session(req)

        session.abort_req.assert_called_once_with(req)
        session.commit_prepared_req.assert_not_called()
        self.assertIsNone(req.session)

    def test_post_admission_cleanup_does_not_roll_back_transaction(self):
        req = self._request(admitted=True)

        Scheduler._release_unadmitted_streaming_session(req)

        req.session.abort_req.assert_not_called()

    def test_hicache_prefetch_never_touches_unadmitted_session_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hicache_storage = True
        scheduler.tree_cache = MagicMock()
        req = self._request()
        req.init_next_round_input = MagicMock()

        Scheduler._prefetch_kvcache(scheduler, req)

        req.init_next_round_input.assert_not_called()
        scheduler.tree_cache.prefetch_from_storage.assert_not_called()


class TestStreamingSessionTopologyAndMetrics(CustomTestCase):
    """Exercise topology fail-closed and exactly-once conflict accounting."""

    def test_builtin_dp_fails_closed_at_scheduler_construction(self):
        """Reject a topology that cannot preserve one durable session owner."""
        with self.assertRaisesRegex(
            ValueError,
            "Streaming sessions require dp_size == 1",
        ):
            _validate_streaming_session_topology(
                SimpleNamespace(
                    enable_streaming_session=True,
                    disaggregation_mode="null",
                ),
                ParallelState.trivial(dp_rank=0, dp_size=2),
            )

        _validate_streaming_session_topology(
            SimpleNamespace(
                enable_streaming_session=True,
                disaggregation_mode="null",
            ),
            ParallelState.trivial(dp_rank=0, dp_size=1),
        )
        _validate_streaming_session_topology(
            SimpleNamespace(
                enable_streaming_session=False,
                disaggregation_mode="prefill",
            ),
            ParallelState.trivial(dp_rank=0, dp_size=8),
        )

    def test_disaggregated_streaming_session_fails_closed(self):
        for disaggregation_mode in ("prefill", "decode"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                with self.assertRaisesRegex(
                    ValueError,
                    "Streaming sessions do not support disaggregation",
                ):
                    _validate_streaming_session_topology(
                        SimpleNamespace(
                            enable_streaming_session=True,
                            disaggregation_mode=disaggregation_mode,
                        ),
                        ParallelState.trivial(dp_rank=0, dp_size=1),
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
            scheduler.metrics_collector_context = SimpleNamespace(
                streaming_session_metrics_enabled=tp_rank == 0
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

    def test_all_scheduler_metrics_escape_counts_each_rank(self):
        """Let the explicit all-schedulers mode expose rank-local counters."""
        time_stats_by_rank = [MagicMock(), MagicMock()]

        for time_stats in time_stats_by_rank:
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.metrics_collector_context = SimpleNamespace(
                streaming_session_metrics_enabled=True
            )
            req = SimpleNamespace(time_stats=time_stats)

            Scheduler._record_streaming_session_idempotency_conflict(scheduler, req)

        for time_stats in time_stats_by_rank:
            time_stats.increment_streaming_session_idempotency_conflict.assert_called_once_with()

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
