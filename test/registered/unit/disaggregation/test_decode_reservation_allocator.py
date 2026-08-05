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
    DecodeReservationAttempt,
    DecodeReservationBootstrapEndpoint,
    DecodeReservationProcess,
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
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

    def __init__(self) -> None:
        """Initialize one empty native queue double."""

        self.cohort = object.__new__(DecodePreparedAllocationCohort)
        self.events = []

    def prepare_preallocated(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        requests: tuple[Req, ...],
    ) -> tuple[tuple[object, ...], DecodePreparedAllocationCohort]:
        """Record exact native preparation inputs.

        :param grant_id: Exact grant identity.
        :param attempt: Exact reserve attempt.
        :param requests: Canonical scheduler requests.
        :returns: Empty receipts and retained cohort.
        """

        self.events.append(("prepare", (grant_id, attempt, requests)))
        return (), self.cohort

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
    """Build one exact TP4 prefill-to-TP1 decode attempt.

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
) -> DecodePreallocQueueReservationAllocator:
    """Build an adapter around deterministic doubles.

    :param queue: Native queue double.
    :param harness: Scheduler request harness.
    :returns: Reservation allocator adapter.
    """

    return DecodePreallocQueueReservationAllocator(
        queue=cast(DecodePreallocQueue, queue),
        build_request=harness.build,
        prepare_request=harness.prepare,
    )


def test_prepare_binds_coordinates_and_uses_exact_scheduler_request() -> None:
    """Preparation delegates the exact canonical object to the native queue."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness)
    tokenized_request = _tokenized_request()
    attempt = _attempt()

    allocations, owner = allocator.prepare(
        grant_id=_GRANT_ID,
        attempt=attempt,
        tokenized_requests=(tokenized_request,),
    )

    assert allocations == ()
    assert owner is queue.cohort
    assert harness.built == [tokenized_request]
    assert harness.prepared == [(tokenized_request, harness.request)]
    assert tokenized_request.bootstrap_host == "10.20.30.40"
    assert tokenized_request.bootstrap_port == 50051
    assert tokenized_request.decode_tp_size == 1
    assert queue.events == [
        ("prepare", (_GRANT_ID, attempt, (harness.request,))),
    ]


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


def test_non_tp1_destination_is_retryable_on_another_decoder() -> None:
    """Asymmetric prepared decode remains explicitly scoped to TP1."""

    queue = _FakeQueue()
    harness = _RequestHarness()
    allocator = _allocator(queue, harness)

    with pytest.raises(
        DecodeReservationAdmissionRefused,
        match="decode_tp_not_supported",
    ) as refusal:
        allocator.prepare(
            grant_id=_GRANT_ID,
            attempt=_attempt(),
            tokenized_requests=(_tokenized_request(decode_tp_size=2),),
        )

    assert (
        refusal.value.disposition
        is DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER
    )
    assert harness.built == []
    assert queue.events == []


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
