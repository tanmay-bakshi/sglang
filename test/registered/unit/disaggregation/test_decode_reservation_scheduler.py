import dataclasses
import logging
import sys
import uuid

import pytest

from sglang.srt.disaggregation.decode_reservation_scheduler import (
    DecodeReservationSchedulerControl,
    DecodeReservationUnavailableControl,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAttempt,
    DecodeReservationBootstrapEndpoint,
    DecodeReservationExpirySweep,
    DecodeReservationOperation,
    DecodeReservationProcess,
)
from sglang.srt.managers.io_struct import (
    DecodeInferenceAttachReqInput,
    DecodeReservationBindReqInput,
    DecodeReservationCancelUnboundReqInput,
    DecodeReservationPrepareReqInput,
    DecodeReservationTransitionReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PREFILL_INSTANCE_ID = uuid.UUID("11111111-2222-4333-8444-555555555555")
_DECODER_INSTANCE_ID = uuid.UUID("12345678-9abc-4def-8123-456789abcdef")
_CHAIN_ID = uuid.UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
_ATTEMPT_ID = uuid.UUID("fedcba98-7654-4321-8fed-cba987654321")
_CHILD_ID = uuid.UUID("01020304-0506-4708-890a-0b0c0d0e0f10")
_GRANT_ID = uuid.UUID("12345678-1234-4678-9234-567812345678")
_BASE_BODY = b'{"input_ids":[1,2,3],"rid":"01020304-0506-4708-890a-0b0c0d0e0f10"}'


class _FakeService:
    """Record scheduler-control calls without native allocation."""

    events: list[tuple[str, object]]
    failure: RuntimeError | None
    prepare_response: dict[str, object] | None
    prepared_expires_at_unix_ms: int | None
    sweep: DecodeReservationExpirySweep
    sweep_calls: int

    def __init__(self) -> None:
        """Initialize an empty service double."""

        self.events = []
        self.failure = None
        self.prepare_response = None
        self.prepared_expires_at_unix_ms = None
        self.sweep = DecodeReservationExpirySweep(
            scanned_count=0,
            cancelled_grant_ids=(),
            quarantined_grant_ids=(),
            deferred_grant_ids=(),
            failed_grant_ids=(),
        )
        self.sweep_calls = 0

    def _record(self, operation: str, value: object) -> dict[str, object]:
        """Record one operation or raise the configured failure.

        :param operation: Stable operation name.
        :param value: Exact received arguments.
        :returns: Deterministic fake receipt.
        """

        self.events.append((operation, value))
        if self.failure is not None:
            raise self.failure
        return {"operation": operation}

    def prepare(
        self,
        *,
        attempt: DecodeReservationAttempt,
        grant_id: uuid.UUID,
        tokenized_requests: tuple[object, ...],
        prepared_expires_at_unix_ms: int | None = None,
    ) -> dict[str, object]:
        """Record a prepare call.

        :param attempt: Parsed attempt.
        :param grant_id: Parsed grant identity.
        :param tokenized_requests: Exact nested requests.
        :param prepared_expires_at_unix_ms: Process-global prepared deadline.
        :returns: Fake receipt.
        """

        self.prepared_expires_at_unix_ms = prepared_expires_at_unix_ms
        response = self._record("reserve", (attempt, grant_id, tokenized_requests))
        return response if self.prepare_response is None else self.prepare_response

    def bind(self, grant_id: uuid.UUID, request_body: bytes) -> dict[str, object]:
        """Record a bind call.

        :param grant_id: Parsed grant identity.
        :param request_body: Exact body bytes.
        :returns: Fake receipt.
        """

        return self._record("bind", (grant_id, request_body))

    def transition(
        self,
        grant_id: uuid.UUID,
        operation: DecodeReservationOperation,
        transcript: dict[str, object],
    ) -> dict[str, object]:
        """Record a lifecycle transition.

        :param grant_id: Parsed grant identity.
        :param operation: Parsed operation.
        :param transcript: Exact transcript.
        :returns: Fake receipt.
        """

        return self._record("transition", (grant_id, operation, transcript))

    def cancel_unbound(
        self,
        grant_id: uuid.UUID,
        transcript: dict[str, object],
    ) -> dict[str, object]:
        """Record an unbound cancellation.

        :param grant_id: Parsed grant identity.
        :param transcript: Exact transcript.
        :returns: Fake receipt.
        """

        return self._record("cancel_unbound", (grant_id, transcript))

    def attach(
        self,
        grant_id: uuid.UUID,
        inference_route: str,
        request_body: bytes,
    ) -> None:
        """Record inference attachment.

        :param grant_id: Parsed grant identity.
        :param inference_route: Exact route.
        :param request_body: Exact body bytes.
        """

        self._record("inference_attach", (grant_id, inference_route, request_body))

    def sweep_expired_prepared(self) -> DecodeReservationExpirySweep:
        """Return the configured expiry result.

        :returns: Configured expiry sweep.
        """

        self.sweep_calls += 1
        return self.sweep


def _attempt() -> DecodeReservationAttempt:
    provisional = DecodeReservationAttempt(
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
    return dataclasses.replace(
        provisional,
        reserve_attempt_digest=provisional.compute_digest(),
    )


def _attempt_wire() -> dict[str, object]:
    attempt = _attempt()
    return {
        "schema_version": 1,
        "prefill_process": attempt.prefill_process.to_dict(),
        "prefill_bootstrap_endpoint": attempt.prefill_bootstrap_endpoint.to_dict(),
        "decoder_process": attempt.decoder_process.to_dict(),
        "logical_request_chain_id": str(attempt.logical_request_chain_id),
        "reservation_attempt_id": str(attempt.reservation_attempt_id),
        "reserve_attempt_digest": attempt.reserve_attempt_digest.hex(),
        "source_tp_size": attempt.source_tp_size,
        "prepared_ttl_ms": attempt.prepared_ttl_ms,
        "inference_route": attempt.inference_route,
        "request_shape": attempt.request_shape,
        "base_request_body_json": attempt.base_request_body.decode(),
        "child_request_ids": [str(child_id) for child_id in attempt.child_request_ids],
    }


def test_prepare_parses_identity_and_preserves_nested_requests() -> None:
    service = _FakeService()
    control = DecodeReservationSchedulerControl(service)
    tokenized_requests = (object(),)

    output = control.handle_prepare(
        DecodeReservationPrepareReqInput(
            correlation_id="prepare-correlation",
            grant_id=str(_GRANT_ID),
            attempt=_attempt_wire(),
            tokenized_requests=tokenized_requests,
            prepared_expires_at_unix_ms=1_900_000_002_500,
        )
    )

    assert output.correlation_id == "prepare-correlation"
    assert output.operation == "reserve"
    assert output.success is True
    assert output.response == {"operation": "reserve"}
    _, (attempt, grant_id, received_requests) = service.events[0]
    assert attempt == _attempt()
    assert grant_id == _GRANT_ID
    assert received_requests == tokenized_requests
    assert service.prepared_expires_at_unix_ms == 1_900_000_002_500


def test_bind_cancel_and_attach_preserve_exact_payloads() -> None:
    service = _FakeService()
    control = DecodeReservationSchedulerControl(service)

    bind = control.handle_bind(
        DecodeReservationBindReqInput(
            correlation_id="bind-correlation",
            grant_id=str(_GRANT_ID),
            request_body=b"bound body",
        )
    )
    cancel = control.handle_cancel_unbound(
        DecodeReservationCancelUnboundReqInput(
            correlation_id="cancel-correlation",
            grant_id=str(_GRANT_ID),
            transcript={"receipt": "cancel"},
        )
    )
    attach = control.handle_attach(
        DecodeInferenceAttachReqInput(
            correlation_id="attach-correlation",
            grant_id=str(_GRANT_ID),
            inference_route="/generate",
            request_body=b"attached body",
        )
    )

    assert bind.success is True
    assert cancel.success is True
    assert attach.success is True
    assert attach.response == {}
    assert service.events == [
        ("bind", (_GRANT_ID, b"bound body")),
        ("cancel_unbound", (_GRANT_ID, {"receipt": "cancel"})),
        ("inference_attach", (_GRANT_ID, "/generate", b"attached body")),
    ]


def test_transition_requires_a_closed_operation() -> None:
    service = _FakeService()
    control = DecodeReservationSchedulerControl(service)

    output = control.handle_transition(
        DecodeReservationTransitionReqInput(
            correlation_id="transition-correlation",
            grant_id=str(_GRANT_ID),
            operation="invented",
            transcript={},
        )
    )

    assert output.operation == "invented"
    assert output.success is False
    assert output.error_type == "DecodeReservationValidationError"
    assert service.events == []


def test_noncanonical_grant_id_fails_without_calling_service() -> None:
    service = _FakeService()
    control = DecodeReservationSchedulerControl(service)

    output = control.handle_bind(
        DecodeReservationBindReqInput(
            correlation_id="bind-correlation",
            grant_id=str(_DECODER_INSTANCE_ID).upper(),
            request_body=b"bound body",
        )
    )

    assert output.success is False
    assert output.error_type == "DecodeReservationValidationError"
    assert service.events == []


def test_service_failure_returns_correlated_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _FakeService()
    service.failure = RuntimeError("native allocator failed")
    control = DecodeReservationSchedulerControl(service)

    with caplog.at_level(logging.ERROR):
        output = control.handle_bind(
            DecodeReservationBindReqInput(
                correlation_id="bind-correlation",
                grant_id=str(_GRANT_ID),
                request_body=b"prompt must not be logged",
            )
        )

    assert output.correlation_id == "bind-correlation"
    assert output.operation == "bind"
    assert output.success is False
    assert output.error_type == "RuntimeError"
    assert output.error_message == "native allocator failed"
    assert "Traceback (most recent call last)" in caplog.text
    assert "prompt must not be logged" not in caplog.text


def test_expiry_sweep_is_throttled_and_routes_terminal_grant() -> None:
    service = _FakeService()
    service.prepare_response = {
        "state": "prepared",
        "grant_id": str(_GRANT_ID),
    }
    control = DecodeReservationSchedulerControl(service)
    prepare = control.handle_prepare(
        DecodeReservationPrepareReqInput(
            correlation_id="prepare-correlation",
            grant_id=str(_GRANT_ID),
            attempt=_attempt_wire(),
            tokenized_requests=(object(),),
            http_worker_ipc="ipc:///tmp/tokenizer-worker",
        )
    )
    assert prepare.success is True
    service.sweep = DecodeReservationExpirySweep(
        scanned_count=1,
        cancelled_grant_ids=(_GRANT_ID,),
        quarantined_grant_ids=(),
        deferred_grant_ids=(),
        failed_grant_ids=(),
    )

    outputs = control.sweep_expired_if_due(10.0)

    assert len(outputs) == 1
    assert outputs[0].http_worker_ipc == "ipc:///tmp/tokenizer-worker"
    assert outputs[0].cancelled_grant_ids == (str(_GRANT_ID),)
    assert outputs[0].quarantined_grant_ids == ()
    assert control.sweep_expired_if_due(10.05) == ()
    assert service.sweep_calls == 1


def test_failed_attachment_retains_expiry_route() -> None:
    """A non-terminal attachment rejection cannot orphan tokenizer cleanup."""

    service = _FakeService()
    service.prepare_response = {
        "state": "prepared",
        "grant_id": str(_GRANT_ID),
    }
    control = DecodeReservationSchedulerControl(service)
    control.handle_prepare(
        DecodeReservationPrepareReqInput(
            correlation_id="prepare-correlation",
            grant_id=str(_GRANT_ID),
            attempt=_attempt_wire(),
            tokenized_requests=(object(),),
            http_worker_ipc="ipc:///tmp/tokenizer-worker",
        )
    )
    service.failure = RuntimeError("attachment rejected before publication")

    attachment = control.handle_attach(
        DecodeInferenceAttachReqInput(
            correlation_id="attach-correlation",
            grant_id=str(_GRANT_ID),
            inference_route="/generate",
            request_body=b"attached body",
        )
    )

    assert attachment.success is False
    service.failure = None
    service.sweep = DecodeReservationExpirySweep(
        scanned_count=1,
        cancelled_grant_ids=(_GRANT_ID,),
        quarantined_grant_ids=(),
        deferred_grant_ids=(),
        failed_grant_ids=(),
    )
    outputs = control.sweep_expired_if_due(10.0)
    assert len(outputs) == 1
    assert outputs[0].http_worker_ipc == "ipc:///tmp/tokenizer-worker"
    assert outputs[0].cancelled_grant_ids == (str(_GRANT_ID),)


def test_unavailable_control_fails_every_operation_closed() -> None:
    """Unsupported topologies return correlated errors for every control IPC."""

    control = DecodeReservationUnavailableControl()
    outputs = (
        control.handle_prepare(
            DecodeReservationPrepareReqInput(
                correlation_id="prepare-correlation",
                grant_id=str(_GRANT_ID),
                attempt={},
                tokenized_requests=(),
            )
        ),
        control.handle_bind(
            DecodeReservationBindReqInput(
                correlation_id="bind-correlation",
                grant_id=str(_GRANT_ID),
                request_body=b"body",
            )
        ),
        control.handle_transition(
            DecodeReservationTransitionReqInput(
                correlation_id="transition-correlation",
                grant_id=str(_GRANT_ID),
                operation="promote",
                transcript={},
            )
        ),
        control.handle_cancel_unbound(
            DecodeReservationCancelUnboundReqInput(
                correlation_id="cancel-correlation",
                grant_id=str(_GRANT_ID),
                transcript={},
            )
        ),
        control.handle_attach(
            DecodeInferenceAttachReqInput(
                correlation_id="attach-correlation",
                grant_id=str(_GRANT_ID),
                inference_route="/generate",
                request_body=b"body",
            )
        ),
    )

    assert tuple(output.operation for output in outputs) == (
        "reserve",
        "bind",
        "promote",
        "cancel_unbound",
        "inference_attach",
    )
    assert all(output.success is False for output in outputs)
    assert all(
        output.error_type == "DecodeReservationUnavailableError" for output in outputs
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
