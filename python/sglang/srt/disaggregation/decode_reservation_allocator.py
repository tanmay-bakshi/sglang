import hashlib
import uuid
from collections.abc import Callable

from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeWriterManifest,
)
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
    DecodeReservationValidationError,
    derive_decode_reservation_bootstrap_rooms,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req

_ALLOCATION_GENERATION_DOMAIN = b"sglang-pd-decoder-allocation-generation-v1"
_ALLOCATION_DIGEST_DOMAIN = b"sglang-pd-decoder-logical-allocation-v1"
_WRITER_TOPOLOGY_DOMAIN = b"sglang-pd-decoder-writer-topology-v1"


class DecodePreallocQueueReservationAllocator:
    """Adapt the keyed decode preallocation queue to reservation ownership."""

    _build_request: Callable[[TokenizedGenerateReqInput], Req]
    _destination_tp_size: int
    _prepare_request: Callable[[TokenizedGenerateReqInput, Req], bool]
    _queue: DecodePreallocQueue

    def __init__(
        self,
        *,
        queue: DecodePreallocQueue,
        destination_tp_size: int,
        build_request: Callable[[TokenizedGenerateReqInput], Req],
        prepare_request: Callable[[TokenizedGenerateReqInput, Req], bool],
    ) -> None:
        """Initialize an adapter around one scheduler-local decode queue.

        :param queue: Native decode allocation and publication authority.
        :param destination_tp_size: Authoritative attention-TP destination width.
        :param build_request: Canonical scheduler request constructor.
        :param prepare_request: Non-publishing scheduler request validator.
        """

        if type(destination_tp_size) is not int or destination_tp_size not in (1, 2):
            raise ValueError("decode reservation destination TP size must be 1 or 2")
        self._queue = queue
        self._destination_tp_size = destination_tp_size
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
        local_allocations, cohort = self._queue.prepare_preallocated(
            grant_id=grant_id,
            attempt=attempt,
            requests=requests,
        )
        try:
            allocations = _project_process_global_allocations(
                grant_id=grant_id,
                attempt=attempt,
                destination_tp_size=self._destination_tp_size,
                local_allocations=local_allocations,
            )
        except (DecodeReservationValidationError, ValueError):
            self._queue.cancel_preallocated(cohort)
            raise
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
            _bind_reservation_bootstrap(
                tokenized_request,
                attempt,
                self._destination_tp_size,
            )
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
        """Publish a prepared cohort's transport ownership.

        :param owner: Exact retained cohort.
        """

        self._queue.promote_preallocated(_prepared_cohort(owner))

    def attach(self, owner: object) -> None:
        """Attach exact inference ownership to a promoted cohort.

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


def _project_process_global_allocations(
    *,
    grant_id: uuid.UUID,
    attempt: DecodeReservationAttempt,
    destination_tp_size: int,
    local_allocations: tuple[DecodeReservationAllocation, ...],
) -> tuple[DecodeReservationAllocation, ...]:
    """Project rank-local allocation evidence into one process-global receipt.

    :param grant_id: Tokenizer-issued process-global grant identity.
    :param attempt: Exact authenticated reserve attempt.
    :param destination_tp_size: Authoritative attention-TP destination width.
    :param local_allocations: Ordered rank-local allocator receipts.
    :returns: Ordered rank-independent control-plane receipts.
    """

    if len(local_allocations) != len(attempt.child_request_ids):
        raise DecodeReservationValidationError(
            "local allocation count differs from reserve child count"
        )
    rooms = derive_decode_reservation_bootstrap_rooms(
        grant_id,
        attempt.child_request_ids,
    )
    writer_manifest_digest = _process_writer_manifest_digest(
        attempt.source_tp_size,
        destination_tp_size,
    )
    projected: list[DecodeReservationAllocation] = []
    for index, (child_request_id, room, local) in enumerate(
        zip(
            attempt.child_request_ids,
            rooms,
            local_allocations,
            strict=True,
        )
    ):
        if local.child_request_id != child_request_id:
            raise DecodeReservationValidationError(
                "local allocation changed its ordered child identity"
            )
        if local.bootstrap_room != room:
            raise DecodeReservationValidationError(
                "local allocation changed its process-global bootstrap room"
            )
        generation = _process_allocation_generation(
            grant_id,
            child_request_id,
            index,
        )
        request_generation = int.from_bytes(generation.bytes[8:], "little")
        allocation_digest = _process_allocation_digest(
            grant_id=grant_id,
            child_request_id=child_request_id,
            generation=generation,
            bootstrap_room=room,
            child_index=index,
            source_tp_size=attempt.source_tp_size,
            destination_tp_size=destination_tp_size,
            writer_manifest_digest=writer_manifest_digest,
            reserved_kv_tokens=local.reserved_kv_tokens,
            remaining_decode_tokens=local.remaining_decode_tokens,
        )
        projected.append(
            DecodeReservationAllocation(
                child_request_id=child_request_id,
                decoder_slot_generation=generation,
                bootstrap_room=room,
                request_slot=index,
                request_generation=request_generation,
                writer_manifest_digest=writer_manifest_digest,
                allocation_digest=allocation_digest,
                reserved_kv_tokens=local.reserved_kv_tokens,
                remaining_decode_tokens=local.remaining_decode_tokens,
            )
        )
    return tuple(projected)


def _process_allocation_generation(
    grant_id: uuid.UUID,
    child_request_id: uuid.UUID,
    child_index: int,
) -> uuid.UUID:
    """Derive one opaque process-global allocation generation.

    :param grant_id: Exact grant identity.
    :param child_request_id: Exact child request identity.
    :param child_index: Ordered child ordinal.
    :returns: Deterministic UUID-shaped allocation generation.
    """

    digest = hashlib.sha256()
    digest.update(_ALLOCATION_GENERATION_DOMAIN)
    digest.update(grant_id.bytes)
    digest.update(child_index.to_bytes(8, "little"))
    digest.update(child_request_id.bytes)
    value = bytearray(digest.digest()[:16])
    value[6] = (value[6] & 0x0F) | 0x80
    value[8] = (value[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(value))


def _process_writer_manifest_digest(
    source_tp_size: int,
    destination_tp_size: int,
) -> bytes:
    """Digest complete destination-rank writer membership.

    :param source_tp_size: Authenticated prefill attention-TP width.
    :param destination_tp_size: Authoritative decode attention-TP width.
    :returns: Stable process-global writer-topology digest.
    """

    digest = hashlib.sha256()
    digest.update(_WRITER_TOPOLOGY_DOMAIN)
    digest.update(source_tp_size.to_bytes(4, "little"))
    digest.update(destination_tp_size.to_bytes(4, "little"))
    for destination_tp_rank in range(destination_tp_size):
        manifest = DecodeWriterManifest.for_tensor_parallel(
            source_tp_size,
            destination_tp_size,
            destination_tp_rank,
        )
        digest.update(destination_tp_rank.to_bytes(4, "little"))
        digest.update(manifest.digest)
    return digest.digest()


def _process_allocation_digest(
    *,
    grant_id: uuid.UUID,
    child_request_id: uuid.UUID,
    generation: uuid.UUID,
    bootstrap_room: int,
    child_index: int,
    source_tp_size: int,
    destination_tp_size: int,
    writer_manifest_digest: bytes,
    reserved_kv_tokens: int,
    remaining_decode_tokens: int,
) -> bytes:
    """Digest globally agreed logical allocation evidence.

    :param grant_id: Exact grant identity.
    :param child_request_id: Exact child request identity.
    :param generation: Process-global allocation generation.
    :param bootstrap_room: Process-global staging room.
    :param child_index: Ordered child ordinal.
    :param source_tp_size: Authenticated prefill attention-TP width.
    :param destination_tp_size: Authoritative decode attention-TP width.
    :param writer_manifest_digest: Complete writer-topology digest.
    :param reserved_kv_tokens: Logical KV-token reservation.
    :param remaining_decode_tokens: Remaining logical decode work.
    :returns: Stable process-global logical allocation digest.
    """

    digest = hashlib.sha256()
    digest.update(_ALLOCATION_DIGEST_DOMAIN)
    digest.update(grant_id.bytes)
    digest.update(child_request_id.bytes)
    digest.update(generation.bytes)
    for value in (
        bootstrap_room,
        child_index,
        source_tp_size,
        destination_tp_size,
        reserved_kv_tokens,
        remaining_decode_tokens,
    ):
        digest.update(value.to_bytes(8, "little"))
    digest.update(writer_manifest_digest)
    return digest.digest()


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
    destination_tp_size: int,
) -> None:
    """Bind authenticated prefill coordinates before scheduler construction.

    :param request: Exact tokenizer-produced child request.
    :param attempt: Authenticated reserve attempt.
    :param destination_tp_size: Authoritative attention-TP destination width.
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
    if request.decode_tp_size not in (None, destination_tp_size):
        raise DecodeReservationAdmissionRefused(
            "decode_tp_mismatch",
            DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
        )
    request.bootstrap_host = endpoint.host
    request.bootstrap_port = endpoint.port
    request.decode_tp_size = destination_tp_size


def _prepared_cohort(owner: object) -> DecodePreparedAllocationCohort:
    """Require the exact native cohort type.

    :param owner: Candidate reservation owner.
    :returns: Exact keyed decode cohort.
    """

    if type(owner) is not DecodePreparedAllocationCohort:
        raise TypeError("decode reservation owner must be a prepared allocation cohort")
    return owner
