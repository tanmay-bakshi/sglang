import logging
import threading
import traceback
import uuid
from collections.abc import Callable, Mapping

from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAllocator,
    DecodeReservationAttempt,
    DecodeReservationAuthority,
    DecodeReservationConflictError,
    DecodeReservationExpirySweep,
    DecodeReservationOperation,
    DecodeReservationSnapshot,
    DecodeReservationState,
)

logger = logging.getLogger(__name__)


class DecodeReservationSchedulerService:
    """Scheduler-local reservation orchestration around one native allocator."""

    _allocator: DecodeReservationAllocator
    _authority: DecodeReservationAuthority
    _prepare_lock: threading.Lock

    def __init__(
        self,
        *,
        expected_decoder_instance_id: uuid.UUID,
        allocator: DecodeReservationAllocator,
        wall_clock_unix_ms: Callable[[], int],
        monotonic_clock_ns: Callable[[], int],
    ) -> None:
        """Initialize one scheduler-local reservation service.

        :param expected_decoder_instance_id: This decoder launch generation.
        :param allocator: Native prepare, publication, and cleanup owner.
        :param wall_clock_unix_ms: Wire-transcript wall clock.
        :param monotonic_clock_ns: Ownership-deadline monotonic clock.
        """

        self._allocator = allocator
        self._authority = DecodeReservationAuthority(
            expected_decoder_instance_id=expected_decoder_instance_id,
            promote=self._promote,
            cancel=self._cancel,
            complete=self._complete,
            abort=self._abort,
            quarantine=self._quarantine,
            wall_clock_unix_ms=wall_clock_unix_ms,
            monotonic_clock_ns=monotonic_clock_ns,
        )
        self._prepare_lock = threading.Lock()

    def prepare(
        self,
        *,
        attempt: DecodeReservationAttempt,
        grant_id: uuid.UUID,
        tokenized_requests: tuple[object, ...],
        prepared_expires_at_unix_ms: int | None = None,
    ) -> dict[str, object]:
        """Prepare one exact cohort or return its authoritative retry outcome.

        The allocator must retain canonical request objects without publishing
        metadata or putting them in runnable queues. Promotion is the only
        publication boundary.

        :param attempt: Exact authenticated reserve transcript.
        :param grant_id: Tokenizer-issued non-secret grant identity.
        :param tokenized_requests: Exact canonical tokenizer outputs.
        :param prepared_expires_at_unix_ms: Process-global prepared deadline.
        :returns: Rust-compatible prepared or reserve-refusal response.
        """

        if type(attempt) is not DecodeReservationAttempt:
            raise TypeError("attempt must be DecodeReservationAttempt")
        if type(grant_id) is not uuid.UUID or grant_id.int == 0:
            raise ValueError("grant_id must be a non-nil UUID")
        owned_requests = tuple(tokenized_requests)
        if len(owned_requests) != len(attempt.child_request_ids):
            raise ValueError("tokenized request count differs from reserve child count")

        with self._prepare_lock:
            retained = self._authority.reserve_retry_response(
                attempt.reservation_attempt_id,
                attempt.reserve_attempt_digest,
            )
            if retained is not None:
                return retained

            try:
                allocations, owner = self._allocator.prepare(
                    grant_id=grant_id,
                    attempt=attempt,
                    tokenized_requests=owned_requests,
                )
            except DecodeReservationAdmissionRefused as refusal:
                return self._authority.refuse(attempt, refusal)
            if owner is None:
                raise RuntimeError("allocator returned no owner for a prepared cohort")
            try:
                snapshot = self._authority.prepare(
                    attempt,
                    grant_id,
                    tuple(allocations),
                    owner,
                    prepared_expires_at_unix_ms=prepared_expires_at_unix_ms,
                )
            except Exception:
                publication_traceback = traceback.format_exc()
                self._discard_unpublished_owner(
                    owner,
                    publication_traceback,
                )
                raise
            return snapshot.reserve_response()

    def bind(
        self,
        grant_id: uuid.UUID,
        request_body: bytes,
    ) -> dict[str, object]:
        """Bind exact final inference bytes.

        :param grant_id: Exact prepared grant.
        :param request_body: Exact final inference body bytes.
        :returns: Idempotent bind receipt.
        """

        return self._authority.bind(grant_id, request_body)

    def attach(
        self,
        grant_id: uuid.UUID,
        inference_route: str,
        request_body: bytes,
    ) -> None:
        """Attach one exact promoted inference request to its response owner.

        :param grant_id: Exact promoted grant identity.
        :param inference_route: Actual normal inference route.
        :param request_body: Actual raw inference body bytes.
        """

        owner = self._authority.claim_inference_attachment(
            grant_id,
            inference_route,
            request_body,
        )
        if owner is None:
            return
        try:
            self._allocator.attach(owner)
        except Exception as attachment_error:
            attachment_traceback = traceback.format_exc()
            diagnostic = "allocator failed while attaching inference ownership"
            logger.error(
                "Inference ownership attachment failed; quarantining the exact "
                "reservation owner:\n%s",
                attachment_traceback,
            )
            try:
                terminal = self._allocator.quarantine(
                    owner,
                    "inference_attachment_failed",
                    diagnostic,
                )
            except Exception as quarantine_error:
                logger.critical(
                    "Inference attachment and quarantine both failed. Attachment "
                    "traceback:\n%s\nQuarantine traceback:\n%s",
                    attachment_traceback,
                    traceback.format_exc(),
                )
                raise DecodeReservationConflictError(
                    "inference attachment ownership is unresolved"
                ) from quarantine_error
            if terminal is not DecodeReservationState.QUARANTINED:
                raise DecodeReservationConflictError(
                    "failed inference attachment returned an invalid quarantine state"
                ) from attachment_error
            self._authority.quarantine_failed_inference_attachment(
                grant_id,
                diagnostic,
            )
            raise DecodeReservationConflictError(
                "inference attachment failed and was quarantined"
            ) from attachment_error
        self._authority.commit_inference_attachment(grant_id)

    def transition(
        self,
        grant_id: uuid.UUID,
        operation: DecodeReservationOperation,
        transcript: Mapping[str, object],
    ) -> dict[str, object]:
        """Apply one exact bound control operation.

        :param grant_id: Exact grant identity.
        :param operation: Closed lifecycle operation.
        :param transcript: Exact Rust control transcript.
        :returns: Idempotent operation receipt.
        """

        return self._authority.transition(grant_id, operation, transcript)

    def cancel_unbound(
        self,
        grant_id: uuid.UUID,
        transcript: Mapping[str, object],
    ) -> dict[str, object]:
        """Cancel one exact unbound cohort.

        :param grant_id: Exact grant identity.
        :param transcript: Exact Rust unbound cancellation transcript.
        :returns: Idempotent cancellation receipt.
        """

        return self._authority.cancel_unbound(grant_id, transcript)

    def sweep_expired_prepared(self) -> DecodeReservationExpirySweep:
        """Reconcile every prepared cohort past its monotonic deadline.

        :returns: Exact grouped sweep outcome.
        """

        return self._authority.sweep_expired_prepared()

    def snapshot(self, grant_id: uuid.UUID) -> DecodeReservationSnapshot:
        """Return one immutable reservation snapshot.

        :param grant_id: Exact grant identity.
        :returns: Immutable scheduler-owned snapshot.
        """

        return self._authority.snapshot(grant_id)

    def _promote(self, owner: object) -> None:
        self._allocator.promote(owner)

    def _cancel(
        self,
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        del reason_code, diagnostic
        return self._allocator.cancel(owner)

    def _complete(
        self,
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        del reason_code, diagnostic
        return self._allocator.complete(owner)

    def _abort(
        self,
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        if reason_code is None:
            raise RuntimeError("validated abort lost its reason code")
        return self._allocator.abort(owner, reason_code, diagnostic)

    def _quarantine(
        self,
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        if reason_code is None:
            raise RuntimeError("validated quarantine lost its reason code")
        return self._allocator.quarantine(owner, reason_code, diagnostic)

    def _discard_unpublished_owner(
        self,
        owner: object,
        publication_traceback: str,
    ) -> None:
        try:
            terminal = self._allocator.cancel(owner)
        except Exception as cancellation_error:
            cancellation_traceback = traceback.format_exc()
            try:
                self._allocator.quarantine(
                    owner,
                    "reservation_publication_failed",
                    cancellation_traceback,
                )
            except Exception as quarantine_error:
                logger.critical(
                    "Reservation publication, cancellation, and quarantine "
                    "all failed. Publication traceback:\n%s\nCancellation "
                    "traceback:\n%s\nQuarantine traceback:\n%s",
                    publication_traceback,
                    cancellation_traceback,
                    traceback.format_exc(),
                )
                raise DecodeReservationConflictError(
                    "unpublished allocation ownership is unresolved"
                ) from quarantine_error
            raise DecodeReservationConflictError(
                "unpublished allocation was quarantined after cancellation failed"
            ) from cancellation_error
        if terminal is DecodeReservationState.CANCELLED:
            return
        self._allocator.quarantine(
            owner,
            "reservation_publication_failed",
            f"cancel returned {terminal.value}",
        )
        raise DecodeReservationConflictError(
            "unpublished allocation was quarantined after invalid cancellation"
        )
