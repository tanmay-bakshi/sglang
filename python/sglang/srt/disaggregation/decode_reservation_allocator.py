import uuid
from collections.abc import Callable

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodePreparedAllocationCohort,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAllocation,
    DecodeReservationAttempt,
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req


class DecodePreallocQueueReservationAllocator:
    """Adapt the keyed decode preallocation queue to reservation ownership."""

    _build_request: Callable[[TokenizedGenerateReqInput], Req]
    _prepare_request: Callable[[TokenizedGenerateReqInput, Req], bool]
    _queue: DecodePreallocQueue

    def __init__(
        self,
        *,
        queue: DecodePreallocQueue,
        build_request: Callable[[TokenizedGenerateReqInput], Req],
        prepare_request: Callable[[TokenizedGenerateReqInput, Req], bool],
    ) -> None:
        """Initialize an adapter around one scheduler-local decode queue.

        :param queue: Native decode allocation and publication authority.
        :param build_request: Canonical scheduler request constructor.
        :param prepare_request: Non-publishing scheduler request validator.
        """

        self._queue = queue
        self._build_request = build_request
        self._prepare_request = prepare_request

    def prepare(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        tokenized_requests: tuple[object, ...],
    ) -> tuple[tuple[DecodeReservationAllocation, ...], object]:
        """Prepare one exact native allocation cohort without publishing it.

        :param grant_id: Tokenizer-issued grant identity.
        :param attempt: Authenticated reserve attempt.
        :param tokenized_requests: Exact canonical tokenizer outputs.
        :returns: Ordered allocation receipts and retained cohort owner.
        """

        requests = self._build_prepared_requests(
            attempt,
            _tokenized_generate_requests(tokenized_requests),
        )
        allocations, cohort = self._queue.prepare_preallocated(
            grant_id=grant_id,
            attempt=attempt,
            requests=requests,
        )
        return allocations, cohort

    def _build_prepared_requests(
        self,
        attempt: DecodeReservationAttempt,
        tokenized_requests: tuple[TokenizedGenerateReqInput, ...],
    ) -> tuple[Req, ...]:
        """Build and validate canonical scheduler requests without publication.

        :param attempt: Authenticated reserve attempt.
        :param tokenized_requests: Exact canonical tokenizer outputs.
        :returns: Ordered fully prepared scheduler requests.
        """

        requests: list[Req] = []
        for tokenized_request in tokenized_requests:
            _bind_reservation_bootstrap(tokenized_request, attempt)
            request = self._build_request(tokenized_request)
            if not self._prepare_request(tokenized_request, request):
                raise DecodeReservationAdmissionRefused(
                    "decode_request_validation_failed",
                    DecodeReservationRefusalDisposition.TERMINAL,
                    "scheduler rejected the canonical reserved request",
                )
            requests.append(request)
        return tuple(requests)

    def promote(self, owner: object) -> None:
        """Authorize a prepared cohort without publishing runnable work.

        :param owner: Exact retained cohort.
        """

        self._queue.promote_preallocated(_prepared_cohort(owner))

    def attach(self, owner: object) -> None:
        """Publish a promoted cohort through the native decode queue.

        :param owner: Exact retained cohort.
        """

        self._queue.attach_preallocated(_prepared_cohort(owner))

    def cancel(self, owner: object) -> DecodeReservationState:
        """Release a prepared cohort exactly.

        :param owner: Exact retained cohort.
        :returns: Authoritative terminal state.
        """

        return self._queue.cancel_preallocated(_prepared_cohort(owner))

    def complete(self, owner: object) -> DecodeReservationState:
        """Reconcile normal terminal completion.

        :param owner: Exact retained cohort.
        :returns: Authoritative terminal state.
        """

        return self._queue.complete_preallocated(_prepared_cohort(owner))

    def abort(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Reconcile an exact promoted abort.

        :param owner: Exact retained cohort.
        :param reason_code: Stable failure reason.
        :param diagnostic: Bounded diagnostic text.
        :returns: Authoritative terminal state.
        """

        return self._queue.abort_preallocated(
            _prepared_cohort(owner),
            reason_code,
            diagnostic,
        )

    def quarantine(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        """Retain ambiguous native ownership process-lifetime.

        :param owner: Exact retained cohort.
        :param reason_code: Stable failure reason.
        :param diagnostic: Bounded diagnostic text.
        :returns: Authoritative quarantined state.
        """

        return self._queue.quarantine_preallocated(
            _prepared_cohort(owner),
            reason_code,
            diagnostic,
        )


def _tokenized_generate_requests(
    requests: tuple[object, ...],
) -> tuple[TokenizedGenerateReqInput, ...]:
    """Validate exact canonical generation request ownership.

    :param requests: Candidate tokenizer outputs.
    :returns: Exact typed request tuple.
    """

    typed_requests: list[TokenizedGenerateReqInput] = []
    for request in requests:
        if not isinstance(request, TokenizedGenerateReqInput):
            raise TypeError(
                "decode reservation requires TokenizedGenerateReqInput children"
            )
        typed_requests.append(request)
    return tuple(typed_requests)


def _bind_reservation_bootstrap(
    request: TokenizedGenerateReqInput,
    attempt: DecodeReservationAttempt,
) -> None:
    """Bind authenticated prefill coordinates before scheduler construction.

    :param request: Exact tokenizer-produced child request.
    :param attempt: Authenticated reserve attempt.
    """

    if request.session_id is not None or request.session_params is not None:
        raise DecodeReservationAdmissionRefused(
            "decode_session_not_supported",
            DecodeReservationRefusalDisposition.TERMINAL,
        )
    if request.bootstrap_room is not None:
        raise DecodeReservationAdmissionRefused(
            "decode_bootstrap_room_already_assigned",
            DecodeReservationRefusalDisposition.TERMINAL,
        )
    endpoint = attempt.prefill_bootstrap_endpoint
    if request.bootstrap_host not in (None, endpoint.host):
        raise DecodeReservationAdmissionRefused(
            "decode_bootstrap_host_mismatch",
            DecodeReservationRefusalDisposition.TERMINAL,
        )
    if request.bootstrap_port not in (None, endpoint.port):
        raise DecodeReservationAdmissionRefused(
            "decode_bootstrap_port_mismatch",
            DecodeReservationRefusalDisposition.TERMINAL,
        )
    if request.decode_tp_size not in (None, 1):
        raise DecodeReservationAdmissionRefused(
            "decode_tp_not_supported",
            DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
        )
    request.bootstrap_host = endpoint.host
    request.bootstrap_port = endpoint.port
    request.decode_tp_size = 1


def _prepared_cohort(owner: object) -> DecodePreparedAllocationCohort:
    """Require the exact native cohort type.

    :param owner: Candidate reservation owner.
    :returns: Exact keyed decode cohort.
    """

    if type(owner) is not DecodePreparedAllocationCohort:
        raise TypeError("decode reservation owner must be a prepared allocation cohort")
    return owner
