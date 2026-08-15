import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodePreparedAllocationCohort,
    DecodeRequest,
    DecodeTransferQueue,
    _DecodeMetadataSubmission,
    _DecodePreparedCohortRecord,
    _DecodePreparedCohortState,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler


def _transaction(
    *,
    binding_digest: bytes | None,
) -> PackedDecodeRequestTransaction:
    """Build the immutable transaction fields used by ownership routing.

    :param binding_digest: Exact terminal authority, otherwise ``None``.
    :returns: Minimal concrete packed transaction.
    """

    transaction = object.__new__(PackedDecodeRequestTransaction)
    transaction._lock = threading.RLock()
    transaction._terminal_binding_digest = binding_digest
    transaction._request_owner = None
    return transaction


def _request(
    rid: str,
    *,
    transaction: PackedDecodeRequestTransaction | None,
) -> DecodeRequest:
    """Build one transfer request with observable resource ownership.

    :param rid: Stable request identity.
    :param transaction: Optional packed transaction.
    :returns: Decode request ready for ownership handoff.
    """

    req = SimpleNamespace(
        rid=rid,
        bootstrap_host="127.0.0.1",
        to_finish=None,
        pd_dflash_boundary_token_id=None,
        pd_dflash_boundary_completion_event=None,
        time_stats=MagicMock(),
    )
    decode_req = DecodeRequest(
        req=req,
        kv_receiver=MagicMock(),
        metadata_buffer_index=3,
        allocation_lease=object(),
        packed_transaction=transaction,
    )
    if transaction is not None:
        transaction._request_owner = decode_req
    return decode_req


def _queue() -> DecodeTransferQueue:
    """Build an empty queue with both structural ownership stores.

    :returns: Minimal transfer queue.
    """

    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = []
    queue._terminal_requests = {}
    queue.tp1_poll_progress_policy = MagicMock()
    queue.scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        waiting_queue=[],
        server_args=object(),
    )
    queue.enable_staging = False
    queue.gloo_group = object()
    queue.metadata_buffers = object()
    return queue


def test_terminal_handoff_never_enters_legacy_polling_or_collectives() -> None:
    """Terminal-only completion is structurally invisible to legacy polling."""

    transaction = _transaction(binding_digest=b"t" * 32)
    decode_req = _request("terminal-only", transaction=transaction)
    queue = _queue()

    queue.register_terminal_requests((decode_req,))

    assert queue.queue == []
    assert queue.live_requests() == (decode_req,)
    with patch("sglang.srt.disaggregation.decode.poll_and_all_reduce") as collective:
        assert queue.pop_transferred() == []
    collective.assert_not_called()
    queue.tp1_poll_progress_policy.mark_idle.assert_called()


def test_mixed_handoff_collectively_polls_only_legacy_requests() -> None:
    """A mixed batch gives each request to exactly one completion owner."""

    terminal_transaction = _transaction(binding_digest=b"t" * 32)
    terminal = _request("terminal", transaction=terminal_transaction)
    legacy = _request("legacy", transaction=None)
    queue = _queue()
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.tree_cache = MagicMock()
    queue.tp_rank = 0
    legacy_receiver = legacy.kv_receiver

    queue.register_terminal_requests((terminal,))
    queue.extend([legacy])

    assert queue.queue == [legacy]
    assert queue.live_requests() == (legacy, terminal)
    with patch(
        "sglang.srt.disaggregation.decode.poll_and_all_reduce",
        return_value=[KVPoll.Transferring],
    ) as collective:
        assert queue.pop_transferred() == []
    collective.assert_called_once()
    call = collective.call_args
    assert call.args[0] == [legacy_receiver]
    assert call.kwargs["decode_reqs"] == [legacy]
    assert terminal_transaction.terminal_binding_digest == b"t" * 32
    assert queue.live_requests() == (legacy, terminal)


def test_terminal_client_abort_retires_through_owner_registry() -> None:
    """Client intent cannot abort the receiver or revoke terminal ownership."""

    transaction = _transaction(binding_digest=b"t" * 32)
    decode_req = _request("terminal-abort", transaction=transaction)
    queue = _queue()
    queue.register_terminal_requests((decode_req,))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler.waiting_queue = []
    scheduler.enable_hicache_storage = False
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[],
        retracted_queue=[],
    )
    scheduler.disagg_decode_transfer_queue = queue
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = None
    scheduler.last_batch = None

    Scheduler.abort_request(scheduler, AbortReq(rid="terminal-abort"))

    assert isinstance(decode_req.req.to_finish, FINISH_ABORT)
    decode_req.kv_receiver.abort.assert_not_called()
    assert queue.live_requests() == (decode_req,)
    assert decode_req.packed_transaction is transaction
    assert decode_req.allocation_lease is not None

    decode_req.req.pd_dflash_boundary_token_id = object()
    decode_req.req.pd_dflash_boundary_completion_event = object()
    receiver = decode_req.kv_receiver
    queue.finalize_terminal_request(decode_req, transaction)

    assert queue.live_requests() == ()
    assert decode_req.packed_transaction is None
    assert decode_req.allocation_lease is None
    assert decode_req.kv_receiver is None
    receiver.clear.assert_called_once_with()
    assert queue.scheduler.waiting_queue == [decode_req.req]
    assert isinstance(decode_req.req.to_finish, FINISH_ABORT)


def test_terminal_handoff_rejects_duplicate_binding_without_partial_visibility() -> (
    None
):
    """A reused terminal identity leaves both ownership stores unchanged."""

    first_transaction = _transaction(binding_digest=b"t" * 32)
    second_transaction = _transaction(binding_digest=b"t" * 32)
    first = _request("first", transaction=first_transaction)
    second = _request("second", transaction=second_transaction)
    queue = _queue()

    try:
        queue.register_terminal_requests((first, second))
    except RuntimeError as error:
        assert "identity was reused" in str(error)
    else:
        raise AssertionError("duplicate terminal binding was accepted")

    assert queue.queue == []
    assert queue.live_requests() == ()


def test_terminal_attachment_bypasses_preallocation_handshake_and_queue() -> None:
    """Promotion attaches directly to owner progress before metadata publication."""

    transaction = _transaction(binding_digest=b"t" * 32)
    decode_req = _request("terminal-attach", transaction=None)
    transaction._request_owner = decode_req
    decode_req.kv_receiver.prefill_info = SimpleNamespace(attn_tp_size=2)
    transfer_queue = _queue()

    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = []
    queue.pending_reqs = []
    queue.transfer_queue = transfer_queue
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_transfer_backend="nixl")
    )
    queue.kv_manager = MagicMock()
    queue._terminal_decode_serving = object()
    queue._prepared_cohort_lock = threading.RLock()
    queue._prepared_cohort_nonce = object()
    queue._resolve_prefill_dp_rank = MagicMock(return_value=0)
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._rebootstrap_prefill_len = MagicMock(return_value=8)
    submission = _DecodeMetadataSubmission(
        decode_req=decode_req,
        page_indices=np.array([1, 2], dtype=np.int32),
        state_indices=None,
        decode_prefix_len=0,
    )
    queue._build_decode_metadata_submission = MagicMock(return_value=submission)

    handle = object.__new__(DecodePreparedAllocationCohort)
    handle._queue_nonce = queue._prepared_cohort_nonce
    handle._token = object()
    record = _DecodePreparedCohortRecord(
        handle=handle,
        grant_id=uuid.uuid4(),
        reservation_attempt_id=uuid.uuid4(),
        source_tp_size=2,
        decode_reqs=(decode_req,),
        packed_transactions=(transaction,),
        allocations=(),
        packed_publications=(object(),),
        state=_DecodePreparedCohortState.PROMOTED,
    )
    queue._prepared_cohorts = {handle._token: record}

    with patch("sglang.srt.disaggregation.decode.poll_and_all_reduce") as collective:
        queue.attach_preallocated(handle)

    collective.assert_not_called()
    queue._resolve_pending_reqs.assert_not_called()
    queue._update_handshake_waiters.assert_not_called()
    assert queue.queue == []
    assert queue.pending_reqs == []
    assert transfer_queue.queue == []
    assert transfer_queue.live_requests() == (decode_req,)
    assert record.state is _DecodePreparedCohortState.ATTACHED
    assert record.metadata_published
    assert decode_req.packed_transaction is transaction
    decode_req.kv_receiver.init.assert_called_once_with(0)
    queue.kv_manager.send_packed_decode_request_metadata.assert_called_once()
