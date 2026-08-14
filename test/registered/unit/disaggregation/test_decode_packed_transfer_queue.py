import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodeRequest,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingReceiver:
    """Observe whether packed completion falls back to legacy receiver work."""

    clear_count: int
    failure_exception_count: int

    def __init__(self) -> None:
        """Initialize empty call counts."""

        self.clear_count = 0
        self.failure_exception_count = 0

    def clear(self) -> None:
        """Record successful receiver retirement."""

        self.clear_count += 1

    def failure_exception(self) -> None:
        """Reject legacy failure inspection for an actor-owned transaction.

        :raises AssertionError: Always, because the packed actor is authoritative.
        """

        self.failure_exception_count += 1
        raise AssertionError("packed failure must not inspect the legacy receiver")


class _FakeMetadataBuffers:
    """Provide one auxiliary metadata row with observable reset state."""

    bootstrap_room: list[int]
    _room_id: int

    def __init__(self, *, size: int, room_id: int, row_index: int) -> None:
        """Initialize one populated row.

        :param size: Number of metadata rows.
        :param room_id: Expected decoder-minted room.
        :param row_index: Populated metadata row.
        """

        self.bootstrap_room = [0] * size
        self.bootstrap_room[row_index] = room_id
        self._room_id = room_id

    def get_buf(self, index: int) -> tuple[object, ...]:
        """Return the populated handoff metadata.

        :param index: Exact adopted metadata row.
        :returns: Metadata tuple consumed by :class:`DecodeTransferQueue`.
        """

        if self.bootstrap_room[index] != self._room_id:
            raise AssertionError("metadata row was reset before scheduler copy")
        unused = object()
        return (
            torch.tensor([17]),
            torch.tensor([4, 3, 2, 1, 0, 0, 0]),
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            torch.tensor([self._room_id]),
        )


class TestDecodePackedTransferQueue(CustomTestCase):
    @staticmethod
    def _request(
        *,
        room_id: int,
        metadata_buffer_index: int,
        receiver: _RecordingReceiver,
        allocation_lease: object,
        packed_transaction: object,
    ) -> DecodeRequest:
        """Build one production-shaped actor-owned decode request.

        :param room_id: Decoder-minted bootstrap room.
        :param metadata_buffer_index: Adopted auxiliary metadata row.
        :param receiver: Legacy receiver retained for publication ownership.
        :param allocation_lease: Packed allocation lease.
        :param packed_transaction: Request-scoped packed actor transaction.
        :returns: Decode request ready for transfer polling.
        """

        req = SimpleNamespace(
            rid="packed-transfer",
            bootstrap_room=room_id,
            bootstrap_host="127.0.0.1",
            return_logprob=False,
            return_sampling_mask=False,
            output_ids=[],
            finished_reason=None,
            pd_rebootstrap_forced_output_id=None,
            time_stats=MagicMock(),
        )
        return DecodeRequest(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=metadata_buffer_index,
            allocation_lease=allocation_lease,
            packed_transaction=packed_transaction,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

    @staticmethod
    def _queue(
        *,
        decode_req: DecodeRequest,
        metadata_buffers: _FakeMetadataBuffers,
        metadata_allocator: MagicMock,
        lifecycle_authority: MagicMock,
        allocation_lease_authority: MagicMock,
    ) -> DecodeTransferQueue:
        """Build a transfer queue whose legacy transport paths are forbidden.

        :param decode_req: Exact packed request under test.
        :param metadata_buffers: Observable auxiliary metadata storage.
        :param metadata_allocator: Existing scheduler row allocator.
        :param lifecycle_authority: Common KV manager owning the packed actor.
        :param allocation_lease_authority: Legacy allocation transition authority.
        :returns: Minimally initialized decode transfer queue.
        """

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.server_args.disaggregation_transfer_backend = "nixl"
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.tp1_poll_progress_policy = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = metadata_allocator
        queue.tp_rank = 0
        queue.metadata_buffers = metadata_buffers
        queue.scheduler = scheduler
        queue.tree_cache = MagicMock()
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue.enable_staging = True
        queue.staging_handler = MagicMock()
        queue.allocation_lease_authority = allocation_lease_authority
        queue.allocation_lifecycle_authority = lifecycle_authority
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue._commit_hicache_local_restore_to_req = MagicMock()
        queue._poll_with_staging = MagicMock(
            side_effect=AssertionError(
                "packed transactions must be polled by the packed actor"
            )
        )
        queue._poll_with_metadata_gate = MagicMock(
            side_effect=AssertionError(
                "packed transactions must be polled by the packed actor"
            )
        )
        return queue

    def test_actor_poll_commits_before_metadata_consumption_and_lease_clear(
        self,
    ) -> None:
        room_id = 41
        metadata_buffer_index = 2
        allocation_lease = object()
        packed_transaction = object()
        receiver = _RecordingReceiver()
        metadata_buffers = _FakeMetadataBuffers(
            size=4,
            room_id=room_id,
            row_index=metadata_buffer_index,
        )
        metadata_allocator = MagicMock()
        allocation_lease_authority = MagicMock()
        lifecycle_authority = MagicMock()
        decode_req = self._request(
            room_id=room_id,
            metadata_buffer_index=metadata_buffer_index,
            receiver=receiver,
            allocation_lease=allocation_lease,
            packed_transaction=packed_transaction,
        )
        queue = self._queue(
            decode_req=decode_req,
            metadata_buffers=metadata_buffers,
            metadata_allocator=metadata_allocator,
            lifecycle_authority=lifecycle_authority,
            allocation_lease_authority=allocation_lease_authority,
        )
        events: list[str] = []
        polls = iter(
            (
                (KVPoll.Transferring, "Transferring"),
                (KVPoll.Success, "Success"),
            )
        )

        def poll(transaction: object) -> KVPoll:
            """Return actor progress while checking pre-commit ownership.

            :param transaction: Exact request-scoped transaction.
            :returns: Next actor state.
            """

            self.assertIs(transaction, packed_transaction)
            self.assertIs(decode_req.allocation_lease, allocation_lease)
            result, label = next(polls)
            events.append(f"poll:{label}")
            return result

        def complete_metadata_consumption(transaction: object) -> None:
            """Model actor-owned row release after the scheduler copied it.

            :param transaction: Exact request-scoped transaction.
            """

            self.assertIs(transaction, packed_transaction)
            self.assertEqual(decode_req.req.output_ids, [17])
            self.assertEqual(decode_req.req.cached_tokens, 4)
            self.assertIs(decode_req.allocation_lease, allocation_lease)
            self.assertEqual(
                metadata_buffers.bootstrap_room[metadata_buffer_index],
                0,
            )
            events.append("complete_metadata_consumption")
            metadata_allocator.free(metadata_buffer_index)

        lifecycle_authority.poll_packed_decode_request_transaction.side_effect = poll
        complete_method = (
            lifecycle_authority.complete_packed_decode_request_metadata_consumption
        )
        complete_method.side_effect = complete_metadata_consumption

        first = queue.pop_transferred()

        self.assertEqual(first, [])
        self.assertEqual(events, ["poll:Transferring"])
        self.assertEqual(queue.queue, [decode_req])
        self.assertIs(decode_req.allocation_lease, allocation_lease)
        complete_method.assert_not_called()

        second = queue.pop_transferred()

        self.assertEqual(second, [decode_req.req])
        self.assertEqual(
            events,
            [
                "poll:Transferring",
                "poll:Success",
                "complete_metadata_consumption",
            ],
        )
        self.assertEqual(queue.queue, [])
        self.assertIsNone(decode_req.allocation_lease)
        self.assertIsNone(decode_req.packed_transaction)
        self.assertEqual(decode_req.metadata_buffer_index, -1)
        self.assertEqual(receiver.clear_count, 1)
        self.assertEqual(receiver.failure_exception_count, 0)
        metadata_allocator.free.assert_called_once_with(metadata_buffer_index)
        allocation_lease_authority.commit_legacy_to_request_after_consumption.assert_not_called()
        allocation_lease_authority.retire_terminal.assert_not_called()
        queue._poll_with_staging.assert_not_called()
        queue._poll_with_metadata_gate.assert_not_called()

    def test_terminal_adoption_keeps_metadata_pinned_until_finalization(
        self,
    ) -> None:
        """Split copy from runnable visibility around actor row consumption."""

        room_id = 47
        metadata_buffer_index = 1
        allocation_lease = object()
        transaction = SimpleNamespace(request_owner=None)
        receiver = _RecordingReceiver()
        metadata_buffers = _FakeMetadataBuffers(
            size=4,
            room_id=room_id,
            row_index=metadata_buffer_index,
        )
        metadata_allocator = MagicMock()
        decode_req = self._request(
            room_id=room_id,
            metadata_buffer_index=metadata_buffer_index,
            receiver=receiver,
            allocation_lease=allocation_lease,
            packed_transaction=transaction,
        )
        transaction.request_owner = decode_req
        queue = self._queue(
            decode_req=decode_req,
            metadata_buffers=metadata_buffers,
            metadata_allocator=metadata_allocator,
            lifecycle_authority=MagicMock(),
            allocation_lease_authority=MagicMock(),
        )
        queue.scheduler.waiting_queue = []

        queue.adopt_terminal_request(decode_req, transaction)

        self.assertEqual(decode_req.req.output_ids, [17])
        self.assertEqual(metadata_buffers.bootstrap_room[metadata_buffer_index], 0)
        self.assertEqual(queue.queue, [decode_req])
        self.assertEqual(queue.scheduler.waiting_queue, [])
        self.assertIs(decode_req.allocation_lease, allocation_lease)
        self.assertIs(decode_req.packed_transaction, transaction)
        self.assertEqual(decode_req.metadata_buffer_index, metadata_buffer_index)
        self.assertEqual(receiver.clear_count, 0)
        metadata_allocator.free.assert_not_called()

        queue.finalize_terminal_request(decode_req, transaction)

        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.scheduler.waiting_queue, [decode_req.req])
        self.assertIsNone(decode_req.allocation_lease)
        self.assertIsNone(decode_req.packed_transaction)
        self.assertEqual(decode_req.metadata_buffer_index, -1)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertEqual(receiver.clear_count, 1)
        metadata_allocator.free.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_terminal_actor_failure_quarantines_without_legacy_cleanup(
        self,
        mock_prepare_abort: MagicMock,
        mock_release_kv_cache: MagicMock,
    ) -> None:
        room_id = 43
        metadata_buffer_index = 3
        allocation_lease = object()
        packed_transaction = object()
        receiver = _RecordingReceiver()
        metadata_buffers = _FakeMetadataBuffers(
            size=4,
            room_id=room_id,
            row_index=metadata_buffer_index,
        )
        metadata_allocator = MagicMock()
        allocation_lease_authority = MagicMock()
        lifecycle_authority = MagicMock()
        lifecycle_authority.poll_packed_decode_request_transaction.return_value = (
            KVPoll.Failed
        )
        decode_req = self._request(
            room_id=room_id,
            metadata_buffer_index=metadata_buffer_index,
            receiver=receiver,
            allocation_lease=allocation_lease,
            packed_transaction=packed_transaction,
        )
        queue = self._queue(
            decode_req=decode_req,
            metadata_buffers=metadata_buffers,
            metadata_allocator=metadata_allocator,
            lifecycle_authority=lifecycle_authority,
            allocation_lease_authority=allocation_lease_authority,
        )

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        lifecycle_authority.poll_packed_decode_request_transaction.assert_called_once_with(
            packed_transaction
        )
        lifecycle_authority.quarantine_packed_decode_request_transaction.assert_called_once()
        quarantined_transaction, reason = (
            lifecycle_authority.quarantine_packed_decode_request_transaction.call_args.args
        )
        self.assertIs(quarantined_transaction, packed_transaction)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)
        lifecycle_authority.complete_packed_decode_request_metadata_consumption.assert_not_called()
        self.assertIs(decode_req.allocation_lease, allocation_lease)
        self.assertIs(decode_req.packed_transaction, packed_transaction)
        self.assertEqual(decode_req.metadata_buffer_index, metadata_buffer_index)
        self.assertEqual(
            metadata_buffers.bootstrap_room[metadata_buffer_index],
            room_id,
        )
        self.assertEqual(receiver.clear_count, 0)
        self.assertEqual(receiver.failure_exception_count, 0)
        metadata_allocator.free.assert_not_called()
        allocation_lease_authority.authorize_legacy_abort_after_terminal_failure.assert_not_called()
        allocation_lease_authority.consume_abort_permit.assert_not_called()
        allocation_lease_authority.retire_terminal.assert_not_called()
        allocation_lease_authority.quarantine.assert_not_called()
        queue.staging_handler.unregister_decode_req.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        mock_prepare_abort.assert_called_once()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req],
            decode_req.req.return_logprob,
        )
        queue._poll_with_staging.assert_not_called()
        queue._poll_with_metadata_gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
