import logging
import traceback
import uuid
from collections.abc import Callable

from sglang.srt.disaggregation.decode_reservation_service import (
    DecodeReservationSchedulerService,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAttempt,
    DecodeReservationOperation,
    DecodeReservationValidationError,
)
from sglang.srt.managers.io_struct import (
    DecodeInferenceAttachReqInput,
    DecodeReservationBindReqInput,
    DecodeReservationCancelUnboundReqInput,
    DecodeReservationControlReqOutput,
    DecodeReservationExpiryReqOutput,
    DecodeReservationPrepareReqInput,
    DecodeReservationTransitionReqInput,
)

logger = logging.getLogger(__name__)

_EXPIRY_SWEEP_INTERVAL_S = 0.1
_TERMINAL_OPERATIONS = frozenset(
    {
        DecodeReservationOperation.CANCEL,
        DecodeReservationOperation.COMPLETE,
        DecodeReservationOperation.ABORT,
        DecodeReservationOperation.QUARANTINE,
    }
)


class DecodeReservationSchedulerControl:
    """Translate correlated scheduler IPC into reservation service calls."""

    _grant_output_ipcs: dict[uuid.UUID, str | None]
    _next_expiry_sweep_s: float
    _service: DecodeReservationSchedulerService

    def __init__(self, service: DecodeReservationSchedulerService) -> None:
        """Initialize scheduler control around one native reservation service.

        :param service: Scheduler-local reservation authority.
        """

        self._service = service
        self._grant_output_ipcs = {}
        self._next_expiry_sweep_s = 0.0

    def handle_prepare(
        self,
        request: DecodeReservationPrepareReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Prepare one canonical request cohort.

        :param request: Correlated prepare IPC.
        :returns: Correlated authoritative response.
        """

        def prepare() -> dict[str, object]:
            grant_id = _canonical_uuid(request.grant_id, "grant_id")
            response = self._service.prepare(
                attempt=DecodeReservationAttempt.from_value(request.attempt),
                grant_id=grant_id,
                tokenized_requests=tuple(request.tokenized_requests),
            )
            if response.get("state") == "prepared":
                if response.get("grant_id") != str(grant_id):
                    raise RuntimeError(
                        "prepared decoder reservation changed its grant identity"
                    )
                self._grant_output_ipcs[grant_id] = request.http_worker_ipc
            return response

        return self._execute(
            request.correlation_id,
            "reserve",
            prepare,
        )

    def handle_bind(
        self,
        request: DecodeReservationBindReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Bind exact final inference bytes.

        :param request: Correlated bind IPC.
        :returns: Correlated authoritative response.
        """

        return self._execute(
            request.correlation_id,
            "bind",
            lambda: self._service.bind(
                _canonical_uuid(request.grant_id, "grant_id"),
                bytes(request.request_body),
            ),
        )

    def handle_transition(
        self,
        request: DecodeReservationTransitionReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Apply one bound lifecycle transition.

        :param request: Correlated transition IPC.
        :returns: Correlated authoritative response.
        """

        operation_name = request.operation

        def transition() -> dict[str, object]:
            grant_id = _canonical_uuid(request.grant_id, "grant_id")
            operation = _reservation_operation(operation_name)
            response = self._service.transition(
                grant_id,
                operation,
                request.transcript,
            )
            if operation in _TERMINAL_OPERATIONS:
                self._grant_output_ipcs.pop(grant_id, None)
            return response

        return self._execute(
            request.correlation_id,
            operation_name,
            transition,
        )

    def handle_cancel_unbound(
        self,
        request: DecodeReservationCancelUnboundReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Cancel one cohort before final-body binding.

        :param request: Correlated unbound-cancel IPC.
        :returns: Correlated authoritative response.
        """

        def cancel_unbound() -> dict[str, object]:
            grant_id = _canonical_uuid(request.grant_id, "grant_id")
            response = self._service.cancel_unbound(grant_id, request.transcript)
            self._grant_output_ipcs.pop(grant_id, None)
            return response

        return self._execute(
            request.correlation_id,
            "cancel_unbound",
            cancel_unbound,
        )

    def handle_attach(
        self,
        request: DecodeInferenceAttachReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Attach exact promoted inference bytes to native publication.

        :param request: Correlated inference-attachment IPC.
        :returns: Correlated authoritative response.
        """

        def attach() -> dict[str, object]:
            grant_id = _canonical_uuid(request.grant_id, "grant_id")
            self._service.attach(
                grant_id,
                request.inference_route,
                bytes(request.request_body),
            )
            self._grant_output_ipcs.pop(grant_id, None)
            return {}

        return self._execute(
            request.correlation_id,
            "inference_attach",
            attach,
        )

    def sweep_expired_if_due(
        self,
        now_monotonic_s: float,
    ) -> tuple[DecodeReservationExpiryReqOutput, ...]:
        """Reconcile expired preparations on a throttled scheduler-loop cadence.

        :param now_monotonic_s: Scheduler-loop monotonic timestamp.
        :returns: Prompt-free terminal notifications routed to exact tokenizers.
        """

        if now_monotonic_s < self._next_expiry_sweep_s:
            return ()
        self._next_expiry_sweep_s = now_monotonic_s + _EXPIRY_SWEEP_INTERVAL_S
        sweep = self._service.sweep_expired_prepared()
        if len(sweep.failed_grant_ids) > 0:
            logger.error(
                "Decoder reservation expiry reconciliation failed for grants %s",
                tuple(str(grant_id) for grant_id in sweep.failed_grant_ids),
            )

        outputs: list[DecodeReservationExpiryReqOutput] = []
        for grant_ids, cancelled in (
            (sweep.cancelled_grant_ids, True),
            (sweep.quarantined_grant_ids, False),
        ):
            for grant_id in grant_ids:
                if grant_id not in self._grant_output_ipcs:
                    raise RuntimeError(
                        "expired decoder reservation lost its tokenizer output owner"
                    )
                output_ipc = self._grant_output_ipcs.pop(grant_id)
                grant_id_value = str(grant_id)
                outputs.append(
                    DecodeReservationExpiryReqOutput(
                        http_worker_ipc=output_ipc,
                        cancelled_grant_ids=(grant_id_value,) if cancelled else (),
                        quarantined_grant_ids=(() if cancelled else (grant_id_value,)),
                    )
                )
        return tuple(outputs)

    @staticmethod
    def _execute(
        correlation_id: str,
        operation: str,
        action: Callable[[], dict[str, object]],
    ) -> DecodeReservationControlReqOutput:
        try:
            response = action()
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Decoder reservation scheduler operation %s failed:\n%s",
                operation,
                traceback.format_exc(),
            )
            return DecodeReservationControlReqOutput(
                correlation_id=correlation_id,
                operation=operation,
                success=False,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        return DecodeReservationControlReqOutput(
            correlation_id=correlation_id,
            operation=operation,
            success=True,
            response=response,
        )


class DecodeReservationUnavailableControl:
    """Return correlated failures when prepared grants are unsupported."""

    def handle_prepare(
        self,
        request: DecodeReservationPrepareReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Refuse unavailable preparation.

        :param request: Correlated prepare IPC.
        :returns: Correlated unavailable response.
        """

        return self._response(request.correlation_id, "reserve")

    def handle_bind(
        self,
        request: DecodeReservationBindReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Refuse unavailable binding.

        :param request: Correlated bind IPC.
        :returns: Correlated unavailable response.
        """

        return self._response(request.correlation_id, "bind")

    def handle_transition(
        self,
        request: DecodeReservationTransitionReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Refuse an unavailable lifecycle transition.

        :param request: Correlated transition IPC.
        :returns: Correlated unavailable response.
        """

        return self._response(request.correlation_id, request.operation)

    def handle_cancel_unbound(
        self,
        request: DecodeReservationCancelUnboundReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Refuse unavailable unbound cancellation.

        :param request: Correlated cancellation IPC.
        :returns: Correlated unavailable response.
        """

        return self._response(request.correlation_id, "cancel_unbound")

    def handle_attach(
        self,
        request: DecodeInferenceAttachReqInput,
    ) -> DecodeReservationControlReqOutput:
        """Refuse unavailable inference attachment.

        :param request: Correlated attachment IPC.
        :returns: Correlated unavailable response.
        """

        return self._response(request.correlation_id, "inference_attach")

    @staticmethod
    def _response(
        correlation_id: str,
        operation: str,
    ) -> DecodeReservationControlReqOutput:
        """Build one prompt-free unavailable response.

        :param correlation_id: Exact tokenizer correlation identity.
        :param operation: Exact requested operation.
        :returns: Correlated unavailable response.
        """

        return DecodeReservationControlReqOutput(
            correlation_id=correlation_id,
            operation=operation,
            success=False,
            error_type="DecodeReservationUnavailableError",
            error_message="decoder reservation control is unavailable on this topology",
        )


def _canonical_uuid(value: str, name: str) -> uuid.UUID:
    """Parse one canonical non-nil UUID string.

    :param value: Candidate UUID string.
    :param name: Field name for diagnostics.
    :returns: Parsed UUID.
    """

    if type(value) is not str:
        raise DecodeReservationValidationError(f"{name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise DecodeReservationValidationError(
            f"{name} must be a canonical UUID"
        ) from error
    if parsed.int == 0 or str(parsed) != value:
        raise DecodeReservationValidationError(f"{name} must be a canonical UUID")
    return parsed


def _reservation_operation(value: str) -> DecodeReservationOperation:
    """Parse one closed lifecycle operation.

    :param value: Candidate operation string.
    :returns: Parsed lifecycle operation.
    """

    if type(value) is not str:
        raise DecodeReservationValidationError(
            "decoder reservation operation must be a string"
        )
    try:
        return DecodeReservationOperation(value)
    except ValueError as error:
        raise DecodeReservationValidationError(
            "unknown decoder reservation operation"
        ) from error
