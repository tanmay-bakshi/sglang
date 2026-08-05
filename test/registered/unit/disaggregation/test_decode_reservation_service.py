import dataclasses
import sys
import uuid

import pytest

from sglang.srt.disaggregation.decode_reservation_service import (
    DecodeReservationSchedulerService,
)
from sglang.srt.disaggregation.decode_reservations import (
    DecodeReservationAdmissionRefused,
    DecodeReservationAllocation,
    DecodeReservationAttempt,
    DecodeReservationBootstrapEndpoint,
    DecodeReservationConflictError,
    DecodeReservationOperation,
    DecodeReservationProcess,
    DecodeReservationRefusalDisposition,
    DecodeReservationState,
    DecodeReservationValidationError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PREFILL_INSTANCE_ID = uuid.UUID("11111111-2222-4333-8444-555555555555")
_DECODER_INSTANCE_ID = uuid.UUID("12345678-9abc-4def-8123-456789abcdef")
_CHAIN_ID = uuid.UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
_ATTEMPT_ID = uuid.UUID("fedcba98-7654-4321-8fed-cba987654321")
_CHILD_ID = uuid.UUID("01020304-0506-4708-890a-0b0c0d0e0f10")
_BASE_BODY = b'{"input_ids":[1,2,3],"rid":"01020304-0506-4708-890a-0b0c0d0e0f10"}'
_BOUND_BODY = (
    b'{"input_ids":[1,2,3],"rid":'
    b'"01020304-0506-4708-890a-0b0c0d0e0f10",'
    b'"bootstrap_host":"10.20.30.40","bootstrap_port":50051,'
    b'"bootstrap_room":41}'
)


@dataclasses.dataclass
class _FakeClock:
    """Controllable service clocks.

    :ivar unix_ms: Current wall-clock milliseconds.
    :ivar monotonic_ns: Current monotonic nanoseconds.
    """

    unix_ms: int = 1_900_000_000_000
    monotonic_ns: int = 5_000_000_000

    def wall_clock_unix_ms(self) -> int:
        """Return controlled wall time.

        :returns: Current wall-clock milliseconds.
        """

        return self.unix_ms

    def monotonic_clock_ns(self) -> int:
        """Return controlled monotonic time.

        :returns: Current monotonic nanoseconds.
        """

        return self.monotonic_ns


class _FakeAllocator:
    """Deterministic native-allocation service test double."""

    events: list[tuple[str, object]]
    invalid_child: bool
    fail_attach: bool
    refuse_prepare: bool
    owner: object

    def __init__(
        self,
        *,
        invalid_child: bool = False,
        fail_attach: bool = False,
        refuse_prepare: bool = False,
    ) -> None:
        """Initialize an empty fake allocator.

        :param invalid_child: Return a contradictory allocation identity.
        :param fail_attach: Fail after attachment publication is claimed.
        :param refuse_prepare: Refuse admission before retaining ownership.
        """

        self.events = []
        self.invalid_child = invalid_child
        self.fail_attach = fail_attach
        self.refuse_prepare = refuse_prepare
        self.owner = object()

    def prepare(
        self,
        *,
        grant_id: uuid.UUID,
        attempt: DecodeReservationAttempt,
        tokenized_requests: tuple[object, ...],
    ) -> tuple[tuple[DecodeReservationAllocation, ...], object]:
        del grant_id
        self.events.append(("prepare", tokenized_requests))
        if self.refuse_prepare:
            raise DecodeReservationAdmissionRefused(
                "decode_capacity_exhausted",
                DecodeReservationRefusalDisposition.RETRY_ANOTHER_DECODER,
            )
        child_request_id = (
            uuid.uuid4() if self.invalid_child else attempt.child_request_ids[0]
        )
        allocation = DecodeReservationAllocation(
            child_request_id=child_request_id,
            decoder_slot_generation=uuid.uuid4(),
            bootstrap_room=41,
            request_slot=7,
            request_generation=3,
            writer_manifest_digest=bytes([0x11]) * 32,
            allocation_digest=bytes([0x22]) * 32,
            reserved_kv_tokens=123,
            remaining_decode_tokens=32,
        )
        return (allocation,), self.owner

    def promote(self, owner: object) -> None:
        self.events.append(("promote", owner))

    def attach(self, owner: object) -> None:
        self.events.append(("attach", owner))
        if self.fail_attach:
            raise RuntimeError("attachment publication failed")

    def cancel(self, owner: object) -> DecodeReservationState:
        self.events.append(("cancel", owner))
        return DecodeReservationState.CANCELLED

    def complete(self, owner: object) -> DecodeReservationState:
        self.events.append(("complete", owner))
        return DecodeReservationState.COMPLETED

    def abort(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        self.events.append(("abort", (owner, reason_code, diagnostic)))
        return DecodeReservationState.ABORTED

    def quarantine(
        self,
        owner: object,
        reason_code: str,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        self.events.append(("quarantine", (owner, reason_code, diagnostic)))
        return DecodeReservationState.QUARANTINED


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


def _service(
    allocator: _FakeAllocator,
    clock: _FakeClock,
) -> DecodeReservationSchedulerService:
    return DecodeReservationSchedulerService(
        expected_decoder_instance_id=_DECODER_INSTANCE_ID,
        allocator=allocator,
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )


def _binding_request(
    service: DecodeReservationSchedulerService,
    grant_id: uuid.UUID,
) -> dict[str, object]:
    snapshot = service.snapshot(grant_id)
    assert snapshot.grant_digest is not None
    return {
        "schema_version": 1,
        "grant_id": str(grant_id),
        "reservation_attempt_id": str(_ATTEMPT_ID),
        "reserve_attempt_digest": snapshot.reserve_attempt_digest.hex(),
        "prefill_process": {
            "url": "https://prefill.example:8443",
            "instance_id": str(_PREFILL_INSTANCE_ID),
        },
        "prefill_bootstrap_endpoint": {
            "host": "10.20.30.40",
            "port": 50051,
        },
        "decoder_process": {
            "url": "http://decode.example:30001",
            "instance_id": str(_DECODER_INSTANCE_ID),
        },
        "logical_request_chain_id": str(_CHAIN_ID),
        "source_tp_size": 4,
        "inference_route": "/generate",
        "request_shape": "scalar",
        "prepared_ttl_ms": 2500,
        "prepared_expires_at_unix_ms": (snapshot.prepared_expires_at_unix_ms),
        "child_request_ids": [str(_CHILD_ID)],
        "decoder_slot_generations": [
            str(snapshot.allocations[0].decoder_slot_generation)
        ],
        "bootstrap_rooms": [41],
        "reservation_digest": snapshot.reservation_digest.hex(),
        "grant_digest": snapshot.grant_digest.hex(),
    }


def test_prepare_is_idempotent_without_duplicate_native_allocation() -> None:
    allocator = _FakeAllocator()
    clock = _FakeClock()
    service = _service(allocator, clock)
    attempt = _attempt()
    grant_id = uuid.uuid4()
    tokenized_request = object()

    first = service.prepare(
        attempt=attempt,
        grant_id=grant_id,
        tokenized_requests=(tokenized_request,),
    )
    second = service.prepare(
        attempt=attempt,
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )

    assert first == second
    assert first is not second
    assert [event[0] for event in allocator.events] == ["prepare"]
    assert allocator.events[0][1] == (tokenized_request,)


def test_allocator_admission_refusal_is_take_once_and_idempotent() -> None:
    allocator = _FakeAllocator(refuse_prepare=True)
    service = _service(allocator, _FakeClock())
    attempt = _attempt()

    first = service.prepare(
        attempt=attempt,
        grant_id=uuid.uuid4(),
        tokenized_requests=(object(),),
    )
    second = service.prepare(
        attempt=attempt,
        grant_id=uuid.uuid4(),
        tokenized_requests=(object(),),
    )

    assert first == second
    assert first is not second
    assert first["state"] == "refused"
    assert first["reason_code"] == "decode_capacity_exhausted"
    assert first["disposition"] == "retry_another_decoder"
    assert first["take_once"] is True
    assert [event[0] for event in allocator.events] == ["prepare"]


def test_bind_is_local_and_promote_is_the_only_publication_boundary() -> None:
    allocator = _FakeAllocator()
    clock = _FakeClock()
    service = _service(allocator, clock)
    attempt = _attempt()
    grant_id = uuid.uuid4()
    service.prepare(
        attempt=attempt,
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )

    service.bind(grant_id, _BOUND_BODY)
    assert [event[0] for event in allocator.events] == ["prepare"]
    binding = _binding_request(service, grant_id)
    service.transition(
        grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )
    assert [event[0] for event in allocator.events] == [
        "prepare",
        "promote",
    ]
    assert allocator.events[1][1] is allocator.owner


def test_exact_inference_attachment_publishes_once_after_promotion() -> None:
    allocator = _FakeAllocator()
    service = _service(allocator, _FakeClock())
    grant_id = uuid.uuid4()
    service.prepare(
        attempt=_attempt(),
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )
    service.bind(grant_id, _BOUND_BODY)
    service.transition(
        grant_id,
        DecodeReservationOperation.PROMOTE,
        _binding_request(service, grant_id),
    )

    service.attach(grant_id, "/generate", _BOUND_BODY)
    service.attach(grant_id, "/generate", _BOUND_BODY)

    assert [event[0] for event in allocator.events] == [
        "prepare",
        "promote",
        "attach",
    ]
    assert service.snapshot(grant_id).inference_attached


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/v1/completions", _BOUND_BODY),
        ("/generate", _BOUND_BODY + b" "),
    ],
)
def test_inference_attachment_requires_exact_route_and_body(
    route: str,
    body: bytes,
) -> None:
    allocator = _FakeAllocator()
    service = _service(allocator, _FakeClock())
    grant_id = uuid.uuid4()
    service.prepare(
        attempt=_attempt(),
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )
    service.bind(grant_id, _BOUND_BODY)
    service.transition(
        grant_id,
        DecodeReservationOperation.PROMOTE,
        _binding_request(service, grant_id),
    )

    with pytest.raises(DecodeReservationValidationError):
        service.attach(grant_id, route, body)

    assert [event[0] for event in allocator.events] == ["prepare", "promote"]


def test_ambiguous_inference_attachment_is_quarantined() -> None:
    allocator = _FakeAllocator(fail_attach=True)
    service = _service(allocator, _FakeClock())
    grant_id = uuid.uuid4()
    service.prepare(
        attempt=_attempt(),
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )
    service.bind(grant_id, _BOUND_BODY)
    service.transition(
        grant_id,
        DecodeReservationOperation.PROMOTE,
        _binding_request(service, grant_id),
    )

    with pytest.raises(
        DecodeReservationConflictError,
        match="failed and was quarantined",
    ):
        service.attach(grant_id, "/generate", _BOUND_BODY)

    assert [event[0] for event in allocator.events] == [
        "prepare",
        "promote",
        "attach",
        "quarantine",
    ]
    assert service.snapshot(grant_id).state is DecodeReservationState.QUARANTINED


def test_expired_prepare_retry_after_sweep_returns_refusal() -> None:
    allocator = _FakeAllocator()
    clock = _FakeClock()
    service = _service(allocator, clock)
    attempt = _attempt()
    grant_id = uuid.uuid4()
    service.prepare(
        attempt=attempt,
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )
    clock.unix_ms -= 60_000
    clock.monotonic_ns += 2_500_000_000
    sweep = service.sweep_expired_prepared()

    retry = service.prepare(
        attempt=attempt,
        grant_id=grant_id,
        tokenized_requests=(object(),),
    )

    assert sweep.cancelled_grant_ids == (grant_id,)
    assert retry["state"] == "refused"
    assert retry["reason_code"] == "prepared_ttl_expired"
    assert [event[0] for event in allocator.events] == [
        "prepare",
        "cancel",
    ]


def test_invalid_allocator_receipt_releases_unpublished_owner() -> None:
    allocator = _FakeAllocator(invalid_child=True)
    service = _service(allocator, _FakeClock())

    with pytest.raises(
        DecodeReservationValidationError,
        match="allocation children differ",
    ):
        service.prepare(
            attempt=_attempt(),
            grant_id=uuid.uuid4(),
            tokenized_requests=(object(),),
        )

    assert [event[0] for event in allocator.events] == [
        "prepare",
        "cancel",
    ]
    assert allocator.events[1][1] is allocator.owner


def test_reused_attempt_with_another_digest_never_reaches_allocator() -> None:
    allocator = _FakeAllocator()
    service = _service(allocator, _FakeClock())
    attempt = _attempt()
    service.prepare(
        attempt=attempt,
        grant_id=uuid.uuid4(),
        tokenized_requests=(object(),),
    )
    contradictory = dataclasses.replace(
        attempt,
        reserve_attempt_digest=bytes([0xAA]) * 32,
    )

    with pytest.raises(DecodeReservationConflictError):
        service.prepare(
            attempt=contradictory,
            grant_id=uuid.uuid4(),
            tokenized_requests=(object(),),
        )
    assert [event[0] for event in allocator.events] == ["prepare"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
