import sys
import uuid
from array import array
from typing import cast

import pytest

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodePreparedAllocationCohort,
)
from sglang.srt.disaggregation.decode_reservation_allocator import (
    DecodePreallocQueueReservationAllocator,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAllocation,
    DecodeReservationAttempt,
    DecodeReservationBootstrapEndpoint,
    DecodeReservationProcess,
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
    derive_decode_reservation_bootstrap_rooms,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_PREFILL_INSTANCE_ID = uuid.UUID("11111111-2222-4333-8444-555555555555")
_DECODER_INSTANCE_ID = uuid.UUID("12345678-9abc-4def-8123-456789abcdef")
_CHAIN_ID = uuid.UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
_ATTEMPT_ID = uuid.UUID("fedcba98-7654-4321-8fed-cba987654321")
_CHILD_ID = uuid.UUID("01020304-0506-4708-890a-0b0c0d0e0f10")
_GRANT_ID = uuid.UUID("12345678-1234-4678-9234-567812345678")
_BASE_BODY = b'{"input_ids":[1,2,3],"rid":"01020304-0506-4708-890a-0b0c0d0e0f10"}'


class _FakeQueue:
    """Record native queue adapter calls without allocating GPU state."""

    cohort: DecodePreparedAllocationCohort
    events: list[tuple[str, object]]
    local_allocations: tuple[DecodeReservationAllocation, ...]
    local_rank: int

    def __init__(self, *, local_rank: int = 0) -> None:
        """Initialize one empty native queue double.

        :param local_rank: Rank identity used to vary process-local evidence.
        """

        self.cohort = object.__new__(DecodePreparedAllocationCohort)
        self.events = []
        self.local_allocations = ()
        self.local_rank = local_rank

    def prepare_preallocated(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        requests: tuple[Req, ...],
    ) -> tuple[
        tuple[DecodeReservationAllocation, ...],
        DecodePreparedAllocationCohort,
    ]:
        """Record exact native preparation inputs.

        :param grant_id: Exact grant identity.
        :param attempt: Exact reserve attempt.
        :param requests: Canonical scheduler requests.
        :returns: Empty receipts and retained cohort.
        """

        self.events.append(("prepare", (grant_id, attempt, requests)))
        rooms = derive_decode_reservation_bootstrap_rooms(
            grant_id,
            attempt.child_request_ids,
        )
        self.local_allocations = tuple(
            DecodeReservationAllocation(
                child_request_id=child_request_id,
                decoder_slot_generation=uuid.UUID(int=self.local_rank + index + 1),
                bootstrap_room=room,
                request_slot=self.local_rank * 100 + index,
                request_generation=self.local_rank + 1,
                writer_manifest_digest=bytes([0x10 + self.local_rank]) * 32,
                allocation_digest=bytes([0x20 + self.local_rank]) * 32,
                reserved_kv_tokens=128,
                remaining_decode_tokens=4,
            )
            for index, (child_request_id, room) in enumerate(
                zip(attempt.child_request_ids, rooms, strict=True)
            )
        )
        return self.local_allocations, self.cohort

    def promote_preallocated(self, cohort: DecodePreparedAllocationCohort) -> None:
        """Record promotion.

        :param cohort: Exact retained cohort.
        """

        self.events.append(("promote", cohort))

    def attach_preallocated(self, cohort: DecodePreparedAllocationCohort) -> None:
        """Record attachment.

        :param cohort: Exact retained cohort.
        """

        self.events.append(("attach", cohort))

    def cancel_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> DecodeReservationState:
        """Record cancellation.

        :param cohort: Exact retained cohort.
        :returns: Cancelled state.
        """

        self.events.append(("cancel", cohort))
        return DecodeReservationState.CANCELLED

    def complete_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
    ) -> DecodeReservationState:
        """Record completion.

        :param cohort: Exact retained cohort.
        :returns: Completed state.
        """

        self.events.append(("complete", cohort))
        return DecodeReservationState.COMPLETED

    def abort_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Record abort.

        :param cohort: Exact retained cohort.
        :param reason_code: Stable reason.
        :param diagnostic: Optional diagnostic.
        :returns: Aborted state.
        """

        self.events.append(("abort", (cohort, reason_code, diagnostic)))
        return DecodeReservationState.ABORTED

    def quarantine_preallocated(
        self,
        cohort: DecodePreparedAllocationCohort,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Record quarantine.

        :param cohort: Exact retained cohort.
        :param reason_code: Stable reason.
        :param diagnostic: Optional diagnostic.
        :returns: Quarantined state.
        """

        self.events.append(("quarantine", (cohort, reason_code, diagnostic)))
        return DecodeReservationState.QUARANTINED


class _RequestHarness:
    """Record canonical scheduler construction and validation calls."""

    built: list[TokenizedGenerateReqInput]
    prepared: list[tuple[TokenizedGenerateReqInput, Req]]
    request: Req
    validation_result: bool

    def __init__(self, *, validation_result: bool = True) -> None:
        """Initialize one request-construction harness.

        :param validation_result: Scheduler validation outcome.
        """

        self.built = []
        self.prepared = []
        self.request = cast(Req, object())
        self.validation_result = validation_result

    def build(self, request: TokenizedGenerateReqInput) -> Req:
        """Record canonical request construction.

        :param request: Tokenizer-produced request.
        :returns: Stable request sentinel.
        """

        self.built.append(request)
        return self.request

    def prepare(
        self,
        tokenized_request: TokenizedGenerateReqInput,
        request: Req,
    ) -> bool:
        """Record non-publishing scheduler validation.

        :param tokenized_request: Tokenizer-produced request.
        :param request: Canonical request sentinel.
        :returns: Configured validation result.
        """

        self.prepared.append((tokenized_request, request))
        return self.validation_result


def _attempt() -> DecodeReservationAttempt:
    """Build one exact TP4 prefill decode attempt.

    :returns: Deterministic reserve attempt.
    """

    return DecodeReservationAttempt(
        prefill_process=DecodeReservationProcess(
            url="https://prefill.example:8443",
            instance_id=_PREFILL_INSTANCE_ID,
        ),
        prefill_bootstrap_endpoint=DecodeReservationBootstrapEndpoint(
            host="10.20.30.40",
            port=50051,
        ),
        decoder_process=DecodeReservationProcess(
            url="http://decode.example:30001",
            instance_id=_DECODER_INSTANCE_ID,
        ),
        logical_request_chain_id=_CHAIN_ID,
        reservation_attempt_id=_ATTEMPT_ID,
        reserve_attempt_digest=bytes(32),
        source_tp_size=4,
        prepared_ttl_ms=2500,
        inference_route="/generate",
        request_shape="scalar",
        base_request_body=_BASE_BODY,
        child_request_ids=(_CHILD_ID,),
    )


def _tokenized_request(
    *,
    session_id: str | None = None,
    bootstrap_host: str | None = None,
    bootstrap_port: int | None = None,
    bootstrap_room: int | None = None,
    decode_tp_size: int | None = None,
) -> TokenizedGenerateReqInput:
    """Build one canonical tokenizer output.

    :param session_id: Optional unsupported session identity.
    :param bootstrap_host: Optional preassigned host.
    :param bootstrap_port: Optional preassigned port.
    :param bootstrap_room: Optional preassigned room.
    :param decode_tp_size: Optional destination TP width.
    :returns: Tokenized generation request.
    """

    return TokenizedGenerateReqInput(
        rid=str(_CHILD_ID),
        input_text=None,
        input_ids=array("l", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_id=session_id,
        bootstrap_host=bootstrap_host,
        bootstrap_port=bootstrap_port,
        bootstrap_room=bootstrap_room,
        decode_tp_size=decode_tp_size,
    )


def _allocator(
    queue: _FakeQueue,
    harness: _RequestHarness,
    destination_tp_size: int = 1,
) -> DecodePreallocQueueReservationAllocator:
    """Build an adapter around deterministic doubles.

    :param queue: Native queue double.
    :param harness: Scheduler request harness.
    :param destination_tp_size: Authoritative attention-TP destination width.
    :returns: Reservation allocator adapter.
    """

    return DecodePreallocQueueReservationAllocator(
        queue=cast(DecodePreallocQueue, queue),
        destination_tp_size=destination_tp_size,
        build_request=harness.build,
        prepare_request=harness.prepare,
    )


@pytest.mark.parametrize("destination_tp_size", (1, 2))
def test_prepare_binds_coordinates_and_uses_exact_scheduler_request(
    destination_tp_size: int,
) -> None:
    """Preparation delegates the exact canonical object to the native queue."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness, destination_tp_size)
    tokenized_request = _tokenized_request()
    attempt = _attempt()

    allocations, owner = allocator.prepare(
        grant_id=_GRANT_ID,
        attempt=attempt,
        tokenized_requests=(tokenized_request,),
    )

    assert len(allocations) == 1
    assert allocations[0].child_request_id == _CHILD_ID
    assert allocations[0].bootstrap_room >= 1 << 63
    assert (
        allocations[0].allocation_digest != queue.local_allocations[0].allocation_digest
    )
    assert owner is queue.cohort
    assert harness.built == [tokenized_request]
    assert harness.prepared == [(tokenized_request, harness.request)]
    assert tokenized_request.bootstrap_host == "10.20.30.40"
    assert tokenized_request.bootstrap_port == 50051
    assert tokenized_request.decode_tp_size == destination_tp_size
    assert queue.events == [
        ("prepare", (_GRANT_ID, attempt, (harness.request,))),
    ]


def test_tp2_ranks_emit_identical_process_global_rooms_and_receipts() -> None:
    """Rank-local lease evidence never enters the process-global receipt."""

    attempt = _attempt()
    rank_receipts: list[tuple[DecodeReservationAllocation, ...]] = []
    rank_requests: list[TokenizedGenerateReqInput] = []
    queues = (_FakeQueue(local_rank=0), _FakeQueue(local_rank=1))
    for queue in queues:
        tokenized_request = _tokenized_request()
        receipts, _ = _allocator(
            queue,
            _RequestHarness(),
            destination_tp_size=2,
        ).prepare(
            grant_id=_GRANT_ID,
            attempt=attempt,
            tokenized_requests=(tokenized_request,),
        )
        rank_receipts.append(receipts)
        rank_requests.append(tokenized_request)

    assert queues[0].local_allocations != queues[1].local_allocations
    assert rank_receipts[0] == rank_receipts[1]
    assert rank_requests[0].decode_tp_size == 2
    assert rank_requests[1].decode_tp_size == 2
    assert rank_receipts[0][0].bootstrap_room >= 1 << 63


def test_scheduler_validation_failure_never_reaches_native_queue() -> None:
    """Invalid canonical requests cannot acquire queue-owned resources."""

    queue = _FakeQueue()
    harness = _RequestHarness(validation_result=False)
    allocator = _allocator(queue, harness)

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_request_validation_failed",
    ) as refusal:
        allocator.prepare(
            grant_id=_GRANT_ID,
            attempt=_attempt(),
            tokenized_requests=(_tokenized_request(),),
        )

    assert refusal.value.disposition is DecodeReservationRefusalDisposition.TERMINAL
    assert queue.events == []


def test_sessions_are_rejected_before_scheduler_construction() -> None:
    """Prepared reservation ownership excludes session request semantics."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness)

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_session_not_supported",
    ):
        allocator.prepare(
            grant_id=_GRANT_ID,
            attempt=_attempt(),
            tokenized_requests=(_tokenized_request(session_id="session"),),
        )

    assert harness.built == []
    assert queue.events == []


@pytest.mark.parametrize(
    "tokenized_request",
    (
        _tokenized_request(bootstrap_room=7),
        _tokenized_request(bootstrap_host="10.20.30.41"),
        _tokenized_request(bootstrap_port=50052),
    ),
)
def test_preassigned_or_mismatched_bootstrap_is_rejected(
    tokenized_request: TokenizedGenerateReqInput,
) -> None:
    """Only authenticated attempt coordinates may reach native allocation."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness)

    with pytest.raises(DecodeReservationAdmissionRefused):
        allocator.prepare(
            grant_id=_GRANT_ID,
            attempt=_attempt(),
            tokenized_requests=(tokenized_request,),
        )

    assert harness.built == []
    assert queue.events == []


@pytest.mark.parametrize(
    ("destination_tp_size", "request_tp_size"),
    ((1, 2), (2, 1), (2, 3)),
)
def test_mismatched_destination_is_retryable_on_another_decoder(
    destination_tp_size: int,
    request_tp_size: int,
) -> None:
    """Preassigned destination width must match the local decode replica."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness, destination_tp_size)

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_tp_mismatch",
    ) as refusal:
        allocator.prepare(
            grant_id=_GRANT_ID,
            attempt=_attempt(),
            tokenized_requests=(_tokenized_request(decode_tp_size=request_tp_size),),
        )

    assert (
        refusal.value.disposition
        is DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER
    )
    assert harness.built == []
    assert queue.events == []


@pytest.mark.parametrize("destination_tp_size", (0, 3, True))
def test_allocator_rejects_unsupported_destination_width(
    destination_tp_size: int,
) -> None:
    """Only implemented destination topologies can expose reservation control."""

    with pytest.raises(
        ValueError,
        match="destination TP size must be 1 or 2",
    ):
        _allocator(_FakeQueue(), _RequestHarness(), destination_tp_size)


def test_lifecycle_methods_forward_exact_cohort_and_terminal_context() -> None:
    """Every reservation transition retains native cohort identity."""

    queue = _FakeQueue()
    allocator = _allocator(queue, _RequestHarness())
    cohort = queue.cohort

    allocator.promote(cohort)
    allocator.attach(cohort)
    assert allocator.cancel(cohort) is DecodeReservationState.CANCELLED
    assert allocator.complete(cohort) is DecodeReservationState.COMPLETED
    assert (
        allocator.abort(cohort, "transport_failed", "bounded")
        is DecodeReservationState.ABORTED
    )
    assert (
        allocator.quarantine(cohort, "ownership_ambiguous", None)
        is DecodeReservationState.QUARANTINED
    )
    assert queue.events == [
        ("promote", cohort),
        ("attach", cohort),
        ("cancel", cohort),
        ("complete", cohort),
        ("abort", (cohort, "transport_failed", "bounded")),
        ("quarantine", (cohort, "ownership_ambiguous", None)),
    ]


def test_lifecycle_methods_reject_non_native_owner_types() -> None:
    """Reservation owners cannot be substituted by structurally similar objects."""

    queue = _FakeQueue()
    allocator = _allocator(queue, _RequestHarness())

    with pytest.raises(TypeError, match="prepared allocation cohort"):
        allocator.promote(object())

    assert queue.events == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
