import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import DecodeRestoreBudget
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TerminalFailureReceiver(FakeReceiver):
    def failure_exception(self) -> None:
        """Expose one authoritative terminal transport failure.

        :raises RuntimeError: Always, because this receiver is terminal.
        """

        raise RuntimeError("terminal transfer failure")


class TestDecodeQueueCleanup(CustomTestCase):
    @staticmethod
    def _idle_decode_scheduler(
        *,
        retracted_requests: list[object],
        has_live_preallocated_cohorts: bool,
    ) -> Scheduler:
        """Build an otherwise idle decode scheduler.

        :param retracted_requests: Native retraction queue contents.
        :param has_live_preallocated_cohorts: Keyed reservation ownership state.
        :returns: Minimally initialized scheduler.
        """

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[],
            retracted_queue=retracted_requests,
            has_live_preallocated_cohorts=(lambda: has_live_preallocated_cohorts),
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[],
            live_requests=lambda: (),
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        return scheduler

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_budgets = MagicMock(
            return_value=DecodeRestoreBudget()
        )

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_budgets = MagicMock(
            return_value=DecodeRestoreBudget()
        )

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_before_removing_request(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
            allocation_lease=None,
            packed_transaction=None,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.tp1_poll_progress_policy = MagicMock()
        queue.enable_staging = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_terminal_legacy_failure_unpins_before_request_cleanup(
        self,
        mock_prepare_abort: MagicMock,
        mock_release_kv_cache: MagicMock,
    ) -> None:
        events: list[str] = []
        receiver = TerminalFailureReceiver()
        req = SimpleNamespace(
            rid="terminal-legacy-failure",
            bootstrap_room=19,
            return_logprob=False,
        )
        lease = object()
        permit = object()
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=2,
            hicache_restore_status=HiCacheRestoreResult.READY,
            allocation_lease=lease,
            packed_transaction=None,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.tp1_poll_progress_policy = MagicMock()
        queue.enable_staging = True
        queue._poll_with_staging = MagicMock(return_value=[KVPoll.Failed])
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[0] * 4)
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.staging_handler = MagicMock()
        queue.staging_handler.is_staging_room.side_effect = [True, False]
        queue.staging_handler.unregister_decode_req.side_effect = (
            lambda bootstrap_room: events.append(f"unregister:{bootstrap_room}")
        )
        queue.allocation_lifecycle_authority = object()
        queue.allocation_lease_authority = MagicMock()
        queue.allocation_lease_authority.authorize_legacy_abort_after_terminal_failure.side_effect = (
            lambda failed_lease, lifecycle: (
                events.append("authorize"),
                permit,
            )[1]
        )
        queue.allocation_lease_authority.consume_abort_permit.side_effect = (
            lambda failed_lease, abort_permit: events.append("consume")
        )
        queue.allocation_lease_authority.retire_terminal.side_effect = (
            lambda failed_lease: events.append("retire")
        )
        mock_release_kv_cache.side_effect = (
            lambda failed_req, tree_cache, is_insert: events.append("release")
        )

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertEqual(
            events,
            ["unregister:19", "authorize", "consume", "retire", "release"],
        )
        queue.allocation_lease_authority.authorize_legacy_abort_after_terminal_failure.assert_called_once_with(
            lease,
            queue.allocation_lifecycle_authority,
        )
        queue.allocation_lease_authority.consume_abort_permit.assert_called_once_with(
            lease,
            permit,
        )
        queue.allocation_lease_authority.quarantine.assert_not_called()
        self.assertIsNone(decode_req.allocation_lease)
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        mock_prepare_abort.assert_called_once()

    def test_ambiguous_legacy_failure_quarantines_without_reuse(self) -> None:
        lease = object()
        authority = MagicMock()
        decode_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=23),
            allocation_lease=lease,
            packed_transaction=None,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = True
        queue.staging_handler = MagicMock()
        queue.allocation_lease_authority = authority
        queue.allocation_lifecycle_authority = object()

        cleanup_authorized = queue._reconcile_failed_allocation(
            decode_req,
            terminal_receiver_failure=False,
        )

        self.assertFalse(cleanup_authorized)
        authority.quarantine.assert_called_once_with(
            lease,
            "legacy transfer failure was not terminal at the receiver",
        )
        authority.authorize_legacy_abort_after_terminal_failure.assert_not_called()
        queue.staging_handler.unregister_decode_req.assert_not_called()
        self.assertIs(decode_req.allocation_lease, lease)

    def test_failed_packed_allocation_bypasses_legacy_reconciliation(self) -> None:
        lease = object()
        authority = MagicMock()
        decode_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=29),
            allocation_lease=lease,
            packed_transaction=object(),
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.allocation_lease_authority = authority
        queue.allocation_lifecycle_authority = object()

        cleanup_authorized = queue._reconcile_failed_allocation(
            decode_req,
            terminal_receiver_failure=True,
        )

        self.assertFalse(cleanup_authorized)
        authority.authorize_legacy_abort_after_terminal_failure.assert_not_called()
        authority.quarantine.assert_not_called()
        self.assertIs(decode_req.allocation_lease, lease)

    def test_consumed_legacy_allocation_commits_exactly_once(self) -> None:
        lease = object()
        lifecycle = object()
        authority = MagicMock()
        decode_req = SimpleNamespace(
            allocation_lease=lease,
            packed_transaction=None,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.allocation_lease_authority = authority
        queue.allocation_lifecycle_authority = lifecycle

        queue._commit_consumed_allocation(decode_req)
        queue._commit_consumed_allocation(decode_req)

        authority.commit_legacy_to_request_after_consumption.assert_called_once_with(
            lease,
            lifecycle,
        )
        authority.retire_terminal.assert_called_once_with(lease)
        self.assertIsNone(decode_req.allocation_lease)

    def test_consumed_packed_allocation_bypasses_legacy_commit(self) -> None:
        lease = object()
        authority = MagicMock()
        decode_req = SimpleNamespace(
            allocation_lease=lease,
            packed_transaction=object(),
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.allocation_lease_authority = authority
        queue.allocation_lifecycle_authority = object()

        queue._commit_consumed_allocation(decode_req)

        authority.commit_legacy_to_request_after_consumption.assert_not_called()
        authority.retire_terminal.assert_not_called()
        self.assertIs(decode_req.allocation_lease, lease)

    def test_retracted_decode_requests_keep_scheduler_non_idle(self) -> None:
        scheduler = self._idle_decode_scheduler(
            retracted_requests=[object()],
            has_live_preallocated_cohorts=False,
        )

        self.assertFalse(scheduler.is_fully_idle())

    def test_live_preallocated_cohort_keeps_scheduler_non_idle(self) -> None:
        """Unpublished and quarantined keyed ownership blocks destructive idleness."""

        scheduler = self._idle_decode_scheduler(
            retracted_requests=[],
            has_live_preallocated_cohorts=True,
        )

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
