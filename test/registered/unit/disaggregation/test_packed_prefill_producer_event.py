import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StopLoop(Exception):
    """Terminate an event-loop double after its decisive observation."""


class _FakeCudaEvent:
    """Record the exact stream and ordering used by one event-loop test."""

    created: list["_FakeCudaEvent"] = []
    timeline: list[str] = []

    stream: object | None

    def __init__(self) -> None:
        self.stream = None
        self.created.append(self)

    def record(self, stream: object) -> None:
        self.stream = stream
        self.timeline.append(f"record:{stream}")


class _Batch:
    """Minimal rank-local batch snapshot used by the overlap-loop test."""

    name: str
    disagg_kv_producer_event: _FakeCudaEvent | None

    def __init__(self, name: str) -> None:
        self.name = name
        self.disagg_kv_producer_event = None

    def copy(self) -> "_Batch":
        snapshot = _Batch(self.name)
        snapshot.disagg_kv_producer_event = self.disagg_kv_producer_event
        return snapshot


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    """State-only scheduler double for exact forward-boundary tests."""


def _base_scheduler() -> _Scheduler:
    """Build common packed-prefill event-loop state.

    :returns: Scheduler double configured for packed-v4 event ownership.
    """

    scheduler = _Scheduler()
    manager = SimpleNamespace(kv_transfer_protocol=lambda: "packed-v4")
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        kv_manager=manager,
        pop_bootstrapped=lambda: [],
    )
    scheduler.request_receiver = SimpleNamespace(recv_requests=lambda: [])
    scheduler.process_input_requests = lambda _requests: None
    scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler.running_batch = object()
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda batch, chunked_req: batch
    )
    scheduler.enable_staging = False
    scheduler.cur_batch_for_debug = None
    scheduler.device = 0
    scheduler.forward_stream = "forward-stream"
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.drain_terminal_scheduler_receipts = lambda: None
    scheduler.maybe_prefetch_staging_for_batch = lambda _batch: None
    scheduler.on_idle = lambda: None
    return scheduler


class TestPackedPrefillProducerEvent(unittest.TestCase):
    """Protect exact, immutable source-KV producer ownership."""

    def setUp(self) -> None:
        _FakeCudaEvent.created.clear()
        _FakeCudaEvent.timeline.clear()

    def test_normal_loop_records_before_processing_the_same_batch(self) -> None:
        """Normal scheduling binds completion before any result can send KV."""

        scheduler = _base_scheduler()
        batch = _Batch("A")
        scheduler.get_next_disagg_prefill_batch_to_run = lambda **_kwargs: (
            SimpleNamespace(batch_to_run=batch, running_batch=object())
        )

        def run_batch(observed_batch: _Batch) -> object:
            self.assertIs(observed_batch, batch)
            _FakeCudaEvent.timeline.append("run:A")
            return object()

        def process_batch_result(observed_batch: _Batch, _result: object) -> None:
            _FakeCudaEvent.timeline.append("process:A")
            self.assertIs(
                observed_batch.disagg_kv_producer_event,
                _FakeCudaEvent.created[0],
            )
            raise _StopLoop

        scheduler.run_batch = run_batch
        scheduler.process_batch_result = process_batch_result

        with (
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.Event",
                _FakeCudaEvent,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.current_stream",
                return_value="schedule-stream",
            ),
            self.assertRaises(_StopLoop),
        ):
            scheduler.event_loop_normal_disagg_prefill()

        self.assertEqual(
            _FakeCudaEvent.timeline,
            ["run:A", "record:schedule-stream", "process:A"],
        )

    def test_overlap_loop_carries_batch_a_event_past_batch_b_launch(self) -> None:
        """Processing A after launching B cannot replace A's dependency."""

        scheduler = _base_scheduler()
        batches = iter((_Batch("A"), _Batch("B")))
        scheduler.get_next_disagg_prefill_batch_to_run = lambda **_kwargs: (
            SimpleNamespace(batch_to_run=next(batches), running_batch=object())
        )

        def run_batch(batch: _Batch) -> object:
            _FakeCudaEvent.timeline.append(f"run:{batch.name}")
            return SimpleNamespace(name=batch.name)

        def apply_war_barrier() -> None:
            _FakeCudaEvent.timeline.append("war")

        def process_batch_result(batch: _Batch, result: object) -> None:
            _FakeCudaEvent.timeline.append(f"process:{batch.name}")
            self.assertEqual(result.name, "A")
            self.assertIs(batch.disagg_kv_producer_event, _FakeCudaEvent.created[0])
            self.assertIsNot(batch.disagg_kv_producer_event, _FakeCudaEvent.created[1])
            raise _StopLoop

        scheduler.run_batch = run_batch
        scheduler._apply_war_barrier = apply_war_barrier
        scheduler.process_batch_result = process_batch_result
        scheduler.progress_disagg_prefill_transfers_during_forward = lambda _event: None
        scheduler.launch_batch_sample_if_needed = lambda _result, batch: (
            _FakeCudaEvent.timeline.append(f"sample:{batch.name}")
        )

        with (
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.Event",
                _FakeCudaEvent,
            ),
            self.assertRaises(_StopLoop),
        ):
            scheduler.event_loop_overlap_disagg_prefill()

        self.assertEqual(
            _FakeCudaEvent.timeline,
            [
                "run:A",
                "record:forward-stream",
                "war",
                "sample:A",
                "run:B",
                "record:forward-stream",
                "war",
                "process:A",
            ],
        )

    def test_schedule_batch_copy_preserves_event_identity(self) -> None:
        """The overlap queue owns the exact event, not a reconstructed fence."""

        producer_event = object()
        batch = ScheduleBatch(reqs=[], disagg_kv_producer_event=producer_event)

        snapshot = batch.copy()

        self.assertIs(snapshot.disagg_kv_producer_event, producer_event)

    def test_nonpacked_transport_does_not_record_cuda_event(self) -> None:
        """Backends outside packed-v4 retain their existing launch behavior."""

        scheduler = _base_scheduler()
        scheduler.disagg_prefill_bootstrap_queue.kv_manager.kv_transfer_protocol = (
            lambda: None
        )
        batch = _Batch("A")

        with patch(
            "sglang.srt.disaggregation.prefill.torch.cuda.Event",
            _FakeCudaEvent,
        ):
            scheduler.record_disagg_prefill_producer_event(batch, "stream")

        self.assertIsNone(batch.disagg_kv_producer_event)
        self.assertEqual(_FakeCudaEvent.created, [])

    def test_deferred_final_send_transfers_and_clears_escrowed_event(self) -> None:
        """Optimistic bootstrap delay retains exactly one final producer event."""

        producer_event = object()
        sender = SimpleNamespace(
            should_send_kv_chunk=lambda _count, _last: True,
            send=Mock(),
        )
        req = SimpleNamespace(
            rid="request-a",
            start_send_idx=0,
            origin_input_ids=[1],
            extend_range=SimpleNamespace(end=1),
            req_pool_idx=0,
            disagg_kv_sender=sender,
        )
        scheduler = _Scheduler()
        scheduler.disagg_prefill_deferred_producer_events = {req.rid: producer_event}
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[0]], dtype=torch.int64)
        )
        scheduler.disagg_metadata_buffers = SimpleNamespace(set_buf=Mock())
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        )

        scheduler.send_kv_chunk(req, last_chunk=True)

        self.assertEqual(scheduler.disagg_prefill_deferred_producer_events, {})
        sender.send.assert_called_once()
        self.assertIs(sender.send.call_args.kwargs["producer_event"], producer_event)

    def test_retry_clears_deferred_event_escrow(self) -> None:
        """A requeued request cannot retain a stale forward dependency."""

        producer_event = object()
        req = SimpleNamespace(
            rid="request-a",
            reset_for_retract=Mock(),
            output_ids=None,
            start_send_idx=1,
            tmp_end_idx=1,
            hidden_states_tensor=object(),
            output_dsa_topk_indices=object(),
            pending_bootstrap=False,
            prefill_attempt_count=0,
            time_stats=SimpleNamespace(
                reset_prefill_retry_time=Mock(),
                set_wait_queue_entry_time=Mock(),
            ),
        )
        scheduler = _Scheduler()
        scheduler.disagg_prefill_deferred_producer_events = {req.rid: producer_event}
        scheduler.server_args = SimpleNamespace(optimistic_prefill_attempts=2)
        scheduler.tree_cache = object()
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.waiting_queue = []
        scheduler.req_to_metadata_buffer_idx_allocator = object()

        with (
            patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
            patch("sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"),
        ):
            scheduler.optimistic_release_and_requeue(req)

        self.assertEqual(scheduler.disagg_prefill_deferred_producer_events, {})
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_transfer_failure_clears_deferred_event_escrow(self) -> None:
        """Failure terminality releases scheduler-local event ownership."""

        req = SimpleNamespace(
            rid="request-a",
            bootstrap_room=41,
            disagg_kv_sender=SimpleNamespace(failure_exception=lambda: None),
            time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=Mock())),
            finished_reason=None,
        )
        scheduler = _Scheduler()
        scheduler.disagg_prefill_deferred_producer_events = {req.rid: object()}
        scheduler.ps = SimpleNamespace(tp_rank=0)
        scheduler.tree_cache = object()
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)

        with (
            patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
            patch("sglang.srt.disaggregation.prefill.prepare_abort"),
        ):
            scheduler.handle_inflight_transfer_failure(req)

        self.assertEqual(scheduler.disagg_prefill_deferred_producer_events, {})


if __name__ == "__main__":
    unittest.main()
