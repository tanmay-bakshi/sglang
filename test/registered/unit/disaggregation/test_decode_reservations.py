import concurrent.futures
import copy
import dataclasses
import gc
import sys
import threading
import uuid
import weakref
from collections.abc import Callable

import pytest

from sglang.srt.disaggregation.decode_reservations import (
    DecodeInferenceAttachmentRegistry,
    DecodeInferenceAttachmentSnapshot,
    DecodeInferenceAttachmentState,
    DecodeReservationAllocation,
    DecodeReservationAttempt,
    DecodeReservationAuthenticationError,
    DecodeReservationAuthority,
    DecodeReservationConflictError,
    DecodeReservationExpiredError,
    DecodeReservationExpirySweep,
    DecodeReservationOperation,
    DecodeReservationOperationInFlightError,
    DecodeReservationSnapshot,
    DecodeReservationState,
    DecodeReservationValidationError,
    _compute_grant_digest,
    _compute_reservation_digest,
    authenticate_process_bearer,
    derive_decode_reservation_bootstrap_rooms,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREFILL_INSTANCE_ID = "11111111-2222-4333-8444-555555555555"
_DECODER_INSTANCE_ID = "12345678-9abc-4def-8123-456789abcdef"
_CHAIN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_ATTEMPT_ID = "fedcba98-7654-4321-8fed-cba987654321"
_CHILD_IDS = (
    "01020304-0506-4708-890a-0b0c0d0e0f10",
    "f0e0d0c0-b0a0-4908-8706-050403020100",
)
_BASE_BODY = (
    b'{"input_ids":[[1,2,3],[4,5]],"rid":'
    b'["01020304-0506-4708-890a-0b0c0d0e0f10",'
    b'"f0e0d0c0-b0a0-4908-8706-050403020100"]}'
)
_BOUND_BODY = (
    b'{"input_ids":[[1,2,3],[4,5]],"rid":'
    b'["01020304-0506-4708-890a-0b0c0d0e0f10",'
    b'"f0e0d0c0-b0a0-4908-8706-050403020100"],'
    b'"bootstrap_host":["10.20.30.40","10.20.30.40"],'
    b'"bootstrap_port":[50051,50051],"bootstrap_room":[41,42]}'
)
_RESERVE_DIGEST = "1673ccc0b56472cbcf512f2caa4fb2989ecb82729791f3438a677af1b2582c14"
_NOW_UNIX_MS = 1_900_000_000_000

Transition = tuple[str, object, str | None, str | None]


@dataclasses.dataclass
class _FakeClock:
    """Controllable wall and monotonic clocks.

    :ivar unix_ms: Current wire-transcript wall time.
    :ivar monotonic_ns: Current ownership-deadline time.
    """

    unix_ms: int = _NOW_UNIX_MS
    monotonic_ns: int = 5_000_000_000

    def wall_clock_unix_ms(self) -> int:
        """Return the controlled wall time.

        :returns: Current wall-clock milliseconds.
        """

        return self.unix_ms

    def monotonic_clock_ns(self) -> int:
        """Return the controlled monotonic time.

        :returns: Current monotonic nanoseconds.
        """

        return self.monotonic_ns


def _reserve_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "prefill_process": {
            "url": "https://prefill.example:8443",
            "instance_id": _PREFILL_INSTANCE_ID,
        },
        "prefill_bootstrap_endpoint": {
            "host": "10.20.30.40",
            "port": 50051,
        },
        "decoder_process": {
            "url": "http://decode.example:30001",
            "instance_id": _DECODER_INSTANCE_ID,
        },
        "logical_request_chain_id": _CHAIN_ID,
        "reservation_attempt_id": _ATTEMPT_ID,
        "reserve_attempt_digest": _RESERVE_DIGEST,
        "source_tp_size": 4,
        "prepared_ttl_ms": 2500,
        "inference_route": "/generate",
        "request_shape": "batch",
        "base_request_body_json": _BASE_BODY.decode(),
        "child_request_ids": list(_CHILD_IDS),
    }


def _allocations() -> tuple[DecodeReservationAllocation, ...]:
    return (
        DecodeReservationAllocation(
            child_request_id=uuid.UUID(_CHILD_IDS[0]),
            decoder_slot_generation=uuid.UUID("10203040-5060-4780-8900-a0b0c0d0e0f0"),
            bootstrap_room=41,
            request_slot=7,
            request_generation=3,
            writer_manifest_digest=bytes([0x11]) * 32,
            allocation_digest=bytes([0x22]) * 32,
            reserved_kv_tokens=12_345,
            remaining_decode_tokens=321,
        ),
        DecodeReservationAllocation(
            child_request_id=uuid.UUID(_CHILD_IDS[1]),
            decoder_slot_generation=uuid.UUID("0f1e2d3c-4b5a-4978-8695-a4b3c2d1e0ff"),
            bootstrap_room=42,
            request_slot=8,
            request_generation=9,
            writer_manifest_digest=bytes([0x33]) * 32,
            allocation_digest=bytes([0x44]) * 32,
            reserved_kv_tokens=67_890,
            remaining_decode_tokens=654,
        ),
    )


def _terminal_action(
    transitions: list[Transition],
    state: DecodeReservationState,
) -> Callable[[object, str | None, str | None], DecodeReservationState]:
    def apply(
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        transitions.append((state.value, owner, reason_code, diagnostic))
        return state

    return apply


def _authority(
    transitions: list[Transition],
    clock: _FakeClock | None = None,
    *,
    complete_state: DecodeReservationState = DecodeReservationState.COMPLETED,
) -> DecodeReservationAuthority:
    def promote(owner: object) -> None:
        transitions.append(("promote", owner, None, None))

    owned_clock = _FakeClock() if clock is None else clock
    return DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=promote,
        cancel=_terminal_action(
            transitions,
            DecodeReservationState.CANCELLED,
        ),
        complete=_terminal_action(
            transitions,
            complete_state,
        ),
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=owned_clock.wall_clock_unix_ms,
        monotonic_clock_ns=owned_clock.monotonic_clock_ns,
    )


def _prepare(
    authority: DecodeReservationAuthority,
    owner: object,
    *,
    grant_id: uuid.UUID | None = None,
) -> DecodeReservationSnapshot:
    return authority.prepare(
        DecodeReservationAttempt.from_value(_reserve_request()),
        uuid.uuid4() if grant_id is None else grant_id,
        _allocations(),
        owner,
    )


def _binding_request(
    snapshot: DecodeReservationSnapshot,
) -> dict[str, object]:
    assert snapshot.grant_digest is not None
    return {
        "schema_version": 1,
        "grant_id": str(snapshot.grant_id),
        "reservation_attempt_id": _ATTEMPT_ID,
        "reserve_attempt_digest": _RESERVE_DIGEST,
        "prefill_process": {
            "url": "https://prefill.example:8443",
            "instance_id": _PREFILL_INSTANCE_ID,
        },
        "prefill_bootstrap_endpoint": {
            "host": "10.20.30.40",
            "port": 50051,
        },
        "decoder_process": {
            "url": "http://decode.example:30001",
            "instance_id": _DECODER_INSTANCE_ID,
        },
        "logical_request_chain_id": _CHAIN_ID,
        "source_tp_size": 4,
        "inference_route": "/generate",
        "request_shape": "batch",
        "prepared_ttl_ms": 2500,
        "prepared_expires_at_unix_ms": (snapshot.prepared_expires_at_unix_ms),
        "child_request_ids": list(_CHILD_IDS),
        "decoder_slot_generations": [
            str(value.decoder_slot_generation) for value in snapshot.allocations
        ],
        "bootstrap_rooms": [41, 42],
        "reservation_digest": snapshot.reservation_digest.hex(),
        "grant_digest": snapshot.grant_digest.hex(),
    }


def _bind(
    authority: DecodeReservationAuthority,
    snapshot: DecodeReservationSnapshot,
) -> tuple[DecodeReservationSnapshot, dict[str, object]]:
    authority.bind(snapshot.grant_id, _BOUND_BODY)
    bound = authority.snapshot(snapshot.grant_id)
    return bound, _binding_request(bound)


def _unbound_cancellation_request(
    snapshot: DecodeReservationSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "grant_id": str(snapshot.grant_id),
        "reservation_attempt_id": _ATTEMPT_ID,
        "reserve_attempt_digest": _RESERVE_DIGEST,
        "prefill_process": {
            "url": "https://prefill.example:8443",
            "instance_id": _PREFILL_INSTANCE_ID,
        },
        "prefill_bootstrap_endpoint": {
            "host": "10.20.30.40",
            "port": 50051,
        },
        "decoder_process": {
            "url": "http://decode.example:30001",
            "instance_id": _DECODER_INSTANCE_ID,
        },
        "logical_request_chain_id": _CHAIN_ID,
        "source_tp_size": 4,
        "inference_route": "/generate",
        "request_shape": "batch",
        "prepared_ttl_ms": 2500,
        "prepared_expires_at_unix_ms": (snapshot.prepared_expires_at_unix_ms),
        "child_request_ids": list(_CHILD_IDS),
        "decoder_slot_generations": [
            str(value.decoder_slot_generation) for value in snapshot.allocations
        ],
        "bootstrap_rooms": [41, 42],
        "reservation_digest": snapshot.reservation_digest.hex(),
        "attempted_grant_digest": None,
    }


def _attachment_registry(
    *,
    opaque_request: object,
) -> tuple[
    DecodeInferenceAttachmentRegistry,
    DecodeInferenceAttachmentSnapshot,
]:
    registry = DecodeInferenceAttachmentRegistry()
    attachment = registry.register(
        grant_id=uuid.uuid4(),
        reservation_attempt_id=uuid.UUID(_ATTEMPT_ID),
        reserve_attempt_digest=bytes.fromhex(_RESERVE_DIGEST),
        inference_route="/generate",
        child_request_ids=tuple(uuid.UUID(value) for value in _CHILD_IDS),
        opaque_request=opaque_request,
    )
    return registry, attachment


def test_cross_language_digest_vectors() -> None:
    attempt = DecodeReservationAttempt.from_value(_reserve_request())
    assert attempt.compute_digest().hex() == _RESERVE_DIGEST

    reservation_digest = _compute_reservation_digest(
        uuid.UUID("00112233-4455-4677-8899-aabbccddeeff"),
        attempt.reserve_attempt_digest,
        1_900_000_000_123,
        _allocations(),
    )
    assert (
        reservation_digest.hex()
        == "d0b0b05dea2236839cc9bef079325e2ff0be11d93bcf9c97aa4718cfe5de495a"
    )
    assert (
        _compute_grant_digest(reservation_digest, _BOUND_BODY).hex()
        == "1a47879143d21f3e0945673cd4b207d2a347cc83d326897967d8937568d0cd73"
    )


@pytest.mark.parametrize("source_tp_size", (1, 2, 4))
def test_reserve_wire_accepts_every_supported_packed_source_width(
    source_tp_size: int,
) -> None:
    """Authenticate control-v1 reserve transcripts for every source width."""

    request = _reserve_request()
    base_attempt = DecodeReservationAttempt.from_value(request)
    candidate = dataclasses.replace(base_attempt, source_tp_size=source_tp_size)
    request["source_tp_size"] = source_tp_size
    request["reserve_attempt_digest"] = candidate.compute_digest().hex()

    attempt = DecodeReservationAttempt.from_value(request)

    assert attempt.source_tp_size == source_tp_size
    assert attempt.reserve_attempt_digest == candidate.compute_digest()


def test_process_global_expiry_produces_identical_rank_receipts() -> None:
    """Rank-local clock sampling cannot enter a TP decoder grant digest."""

    attempt = DecodeReservationAttempt.from_value(_reserve_request())
    grant_id = uuid.UUID("00112233-4455-4677-8899-aabbccddeeff")
    expires_at_unix_ms = _NOW_UNIX_MS + attempt.prepared_ttl_ms
    snapshots: list[DecodeReservationSnapshot] = []
    for clock_offset_ms in (0, 17):
        clock = _FakeClock(unix_ms=_NOW_UNIX_MS + clock_offset_ms)
        authority = _authority([], clock)
        snapshots.append(
            authority.prepare(
                attempt,
                grant_id,
                _allocations(),
                object(),
                prepared_expires_at_unix_ms=expires_at_unix_ms,
            )
        )

    assert snapshots[0].prepared_expires_at_unix_ms == expires_at_unix_ms
    assert snapshots[1].prepared_expires_at_unix_ms == expires_at_unix_ms
    assert snapshots[0].reservation_digest == snapshots[1].reservation_digest
    assert snapshots[0].reserve_response() == snapshots[1].reserve_response()


@pytest.mark.parametrize("expires_at_unix_ms", (0, -1, 1 << 64, True))
def test_process_global_expiry_requires_positive_u64(
    expires_at_unix_ms: object,
) -> None:
    """A prepared receipt accepts only an exact positive u64 deadline."""

    authority = _authority([], _FakeClock())
    with pytest.raises(
        DecodeReservationValidationError,
        match="prepared reservation expiry",
    ):
        authority.prepare(
            DecodeReservationAttempt.from_value(_reserve_request()),
            uuid.uuid4(),
            _allocations(),
            object(),
            prepared_expires_at_unix_ms=expires_at_unix_ms,
        )


def test_bootstrap_rooms_are_deterministic_ordered_and_staging_scoped() -> None:
    """Every TP rank derives the same collision-resistant room vector."""

    grant_id = uuid.UUID("00112233-4455-4677-8899-aabbccddeeff")
    child_ids = tuple(uuid.UUID(value) for value in _CHILD_IDS)
    first = derive_decode_reservation_bootstrap_rooms(grant_id, child_ids)
    second = derive_decode_reservation_bootstrap_rooms(grant_id, child_ids)

    assert first == second
    assert len(set(first)) == len(child_ids)
    assert all(room >= 1 << 63 for room in first)
    assert (
        derive_decode_reservation_bootstrap_rooms(
            grant_id,
            tuple(reversed(child_ids)),
        )
        != first
    )
    assert derive_decode_reservation_bootstrap_rooms(uuid.uuid4(), child_ids) != first


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"unexpected": True}),
        lambda value: value.update({"source_tp_size": 3}),
        lambda value: value.update({"reserve_attempt_digest": "00" * 32}),
        lambda value: value.update({"child_request_ids": list(reversed(_CHILD_IDS))}),
    ),
)
def test_reserve_request_is_strict(
    mutation: Callable[[dict[str, object]], None],
) -> None:
    request = _reserve_request()
    mutation(request)
    with pytest.raises(DecodeReservationValidationError):
        DecodeReservationAttempt.from_value(request)


def test_state_machine_owns_exact_cohort_and_returns_immutable_snapshots() -> None:
    transitions: list[Transition] = []
    authority = _authority(transitions)
    owner = object()
    prepared = _prepare(authority, owner)

    assert prepared.state is DecodeReservationState.PREPARED_UNBOUND
    assert (
        "opaque_allocation_owner" not in DecodeReservationSnapshot.__dataclass_fields__
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.state = DecodeReservationState.COMPLETED  # type: ignore[misc]

    bound, binding = _bind(authority, prepared)
    promote_receipt = authority.transition(
        prepared.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )
    complete_receipt = authority.transition(
        prepared.grant_id,
        DecodeReservationOperation.COMPLETE,
        binding,
    )

    assert prepared.state is DecodeReservationState.PREPARED_UNBOUND
    assert bound.state is DecodeReservationState.PREPARED_BOUND
    assert promote_receipt["state"] == "promoted"
    assert complete_receipt["state"] == "completed"
    assert authority.snapshot(prepared.grant_id).state is (
        DecodeReservationState.COMPLETED
    )
    assert [transition[0] for transition in transitions] == [
        "promote",
        "completed",
    ]
    assert all(transition[1] is owner for transition in transitions)


def test_completion_quarantine_terminalizes_authority_ownership() -> None:
    """A live packed lease conservatively terminalizes both ownership layers."""

    transitions: list[Transition] = []
    authority = _authority(
        transitions,
        complete_state=DecodeReservationState.QUARANTINED,
    )
    prepared = _prepare(authority, object())
    bound, binding = _bind(authority, prepared)
    authority.transition(
        bound.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )

    first = authority.transition(
        bound.grant_id,
        DecodeReservationOperation.COMPLETE,
        binding,
    )
    second = authority.transition(
        bound.grant_id,
        DecodeReservationOperation.COMPLETE,
        binding,
    )

    assert first["state"] == "quarantined"
    assert second == first
    assert authority.snapshot(bound.grant_id).state is (
        DecodeReservationState.QUARANTINED
    )
    assert [transition[0] for transition in transitions] == [
        "promote",
        "quarantined",
    ]


def test_retries_are_idempotent_and_receipts_are_isolated_copies() -> None:
    authority = _authority([])
    owner = object()
    attempt = DecodeReservationAttempt.from_value(_reserve_request())
    grant_id = uuid.uuid4()
    prepared = authority.prepare(
        attempt,
        grant_id,
        _allocations(),
        owner,
    )
    repeated = authority.prepare(
        attempt,
        grant_id,
        _allocations(),
        object(),
    )
    assert repeated == prepared
    assert repeated is not prepared

    first_receipt = authority.bind(grant_id, _BOUND_BODY)
    first_receipt["child_request_ids"][0] = str(uuid.uuid4())
    first_receipt["prefill_process"]["url"] = "https://mutated.invalid"
    second_receipt = authority.bind(grant_id, _BOUND_BODY)
    assert second_receipt["child_request_ids"] == list(_CHILD_IDS)
    assert second_receipt["prefill_process"]["url"] == ("https://prefill.example:8443")
    assert second_receipt is not first_receipt

    changed = _BOUND_BODY.replace(b"[41,42]", b"[41,43]")
    with pytest.raises(DecodeReservationConflictError):
        authority.bind(grant_id, changed)


def test_callback_runs_outside_lock_and_concurrent_operations_fail_closed() -> None:
    started = threading.Event()
    release = threading.Event()
    owner = object()

    def blocking_promote(callback_owner: object) -> None:
        assert callback_owner is owner
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the promote callback")

    transitions: list[Transition] = []
    clock = _FakeClock()
    authority = DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=blocking_promote,
        cancel=_terminal_action(
            transitions,
            DecodeReservationState.CANCELLED,
        ),
        complete=_terminal_action(
            transitions,
            DecodeReservationState.COMPLETED,
        ),
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )
    prepared = _prepare(authority, owner)
    bound, binding = _bind(authority, prepared)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        promote = executor.submit(
            authority.transition,
            bound.grant_id,
            DecodeReservationOperation.PROMOTE,
            binding,
        )
        assert started.wait(timeout=2)
        try:
            current = executor.submit(
                authority.snapshot,
                bound.grant_id,
            ).result(timeout=1)
            assert current.state is DecodeReservationState.PREPARED_BOUND

            retry = executor.submit(
                authority.transition,
                bound.grant_id,
                DecodeReservationOperation.PROMOTE,
                binding,
            )
            with pytest.raises(DecodeReservationOperationInFlightError):
                retry.result(timeout=1)

            conflict = executor.submit(
                authority.transition,
                bound.grant_id,
                DecodeReservationOperation.CANCEL,
                binding,
            )
            with pytest.raises(DecodeReservationConflictError):
                conflict.result(timeout=1)
        finally:
            release.set()
        assert promote.result(timeout=2)["state"] == "promoted"


def test_failed_operation_retries_only_the_same_reconciliation() -> None:
    attempts = 0
    owner = object()

    def flaky_promote(callback_owner: object) -> None:
        nonlocal attempts
        assert callback_owner is owner
        attempts += 1
        if attempts == 1:
            raise RuntimeError("ambiguous allocator result")

    transitions: list[Transition] = []
    clock = _FakeClock()
    authority = DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=flaky_promote,
        cancel=_terminal_action(
            transitions,
            DecodeReservationState.CANCELLED,
        ),
        complete=_terminal_action(
            transitions,
            DecodeReservationState.COMPLETED,
        ),
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )
    prepared = _prepare(authority, owner)
    bound, binding = _bind(authority, prepared)

    with pytest.raises(RuntimeError, match="ambiguous allocator result"):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.PROMOTE,
            binding,
        )
    assert authority.snapshot(bound.grant_id).state is (
        DecodeReservationState.PREPARED_BOUND
    )
    with pytest.raises(DecodeReservationConflictError, match="blocks cancel"):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.CANCEL,
            binding,
        )

    receipt = authority.transition(
        bound.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )
    assert receipt["state"] == "promoted"
    assert attempts == 2


def test_invalid_callback_result_remains_reconcilable() -> None:
    attempts = 0

    def promote(owner: object) -> None:
        del owner

    def invalid_then_complete(
        owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        nonlocal attempts
        del owner, reason_code, diagnostic
        attempts += 1
        if attempts == 1:
            return DecodeReservationState.ABORTED
        return DecodeReservationState.COMPLETED

    transitions: list[Transition] = []
    clock = _FakeClock()
    authority = DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=promote,
        cancel=_terminal_action(
            transitions,
            DecodeReservationState.CANCELLED,
        ),
        complete=invalid_then_complete,
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )
    prepared = _prepare(authority, object())
    bound, binding = _bind(authority, prepared)
    authority.transition(
        bound.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )

    with pytest.raises(DecodeReservationConflictError, match="returned state"):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.COMPLETE,
            binding,
        )
    receipt = authority.transition(
        bound.grant_id,
        DecodeReservationOperation.COMPLETE,
        binding,
    )
    assert receipt["state"] == "completed"
    assert attempts == 2


def test_unbound_cancel_is_idempotent_and_owns_exact_cohort() -> None:
    transitions: list[Transition] = []
    authority = _authority(transitions)
    owner = object()
    prepared = _prepare(authority, owner)
    request = _unbound_cancellation_request(prepared)

    first = authority.cancel_unbound(prepared.grant_id, request)
    first["decoder_slot_generations"][0] = str(uuid.uuid4())
    second = authority.cancel_unbound(prepared.grant_id, request)

    assert second["state"] == "cancelled"
    assert second["decoder_slot_generations"][0] == str(
        _allocations()[0].decoder_slot_generation
    )
    assert transitions == [("cancelled", owner, None, None)]


def test_expiry_sweep_indexes_only_live_prepared_grants() -> None:
    clock = _FakeClock()
    transitions: list[Transition] = []
    authority = _authority(transitions, clock)
    prepared = _prepare(authority, object())

    assert authority.sweep_expired_prepared().scanned_count == 1
    bound, binding = _bind(authority, prepared)
    assert authority.sweep_expired_prepared().scanned_count == 1
    authority.transition(
        bound.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )
    assert authority.sweep_expired_prepared().scanned_count == 0


def test_monotonic_expiry_cancels_unbound_and_refuses_reserve_retry() -> None:
    clock = _FakeClock()
    transitions: list[Transition] = []
    authority = _authority(transitions, clock)
    owner = object()
    prepared = _prepare(authority, owner)
    cancellation = _unbound_cancellation_request(prepared)
    assert prepared.prepared_expires_at_unix_ms == _NOW_UNIX_MS + 2500

    clock.unix_ms -= 60_000
    clock.monotonic_ns += 2_500_000_000
    sweep = authority.sweep_expired_prepared()

    assert isinstance(sweep, DecodeReservationExpirySweep)
    assert sweep.cancelled_grant_ids == (prepared.grant_id,)
    assert len(sweep.quarantined_grant_ids) == 0
    assert authority.snapshot(prepared.grant_id).state is (
        DecodeReservationState.CANCELLED
    )
    assert transitions == [("cancelled", owner, None, None)]

    first_retry = authority.reserve_retry_response(
        prepared.reservation_attempt_id,
        prepared.reserve_attempt_digest,
    )
    assert first_retry is not None
    assert first_retry["operation"] == "reserve"
    assert first_retry["state"] == "refused"
    assert first_retry["reason_code"] == "prepared_ttl_expired"
    assert first_retry["disposition"] == "retry_same_decoder"
    first_retry["prefill_process"]["url"] = "https://mutated.invalid"
    second_retry = authority.reserve_retry_response(
        prepared.reservation_attempt_id,
        prepared.reserve_attempt_digest,
    )
    assert second_retry is not None
    assert second_retry["prefill_process"]["url"] == ("https://prefill.example:8443")
    assert second_retry["receipt_id"] == first_retry["receipt_id"]

    cancel_receipt = authority.cancel_unbound(
        prepared.grant_id,
        cancellation,
    )
    assert cancel_receipt["state"] == "cancelled"
    assert transitions == [("cancelled", owner, None, None)]


def test_reserve_retry_rejects_expiry_until_periodic_sweep() -> None:
    clock = _FakeClock()
    transitions: list[Transition] = []
    authority = _authority(transitions, clock)
    owner = object()
    prepared = _prepare(authority, owner)
    clock.monotonic_ns += 2_500_000_000

    with pytest.raises(DecodeReservationExpiredError):
        authority.reserve_retry_response(
            prepared.reservation_attempt_id,
            prepared.reserve_attempt_digest,
        )

    assert authority.snapshot(prepared.grant_id).state is (
        DecodeReservationState.PREPARED_UNBOUND
    )
    assert transitions == []

    sweep = authority.sweep_expired_prepared()
    assert sweep.cancelled_grant_ids == (prepared.grant_id,)
    assert authority.snapshot(prepared.grant_id).state is (
        DecodeReservationState.CANCELLED
    )
    assert transitions == [("cancelled", owner, None, None)]


def test_monotonic_expiry_cancels_clean_bound_and_blocks_late_promotion() -> None:
    clock = _FakeClock()
    transitions: list[Transition] = []
    authority = _authority(transitions, clock)
    owner = object()
    prepared = _prepare(authority, owner)
    bound, binding = _bind(authority, prepared)

    clock.monotonic_ns += 2_500_000_000
    with pytest.raises(DecodeReservationExpiredError):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.PROMOTE,
            binding,
        )
    sweep = authority.sweep_expired_prepared()

    assert sweep.cancelled_grant_ids == (bound.grant_id,)
    assert len(sweep.quarantined_grant_ids) == 0
    assert authority.snapshot(bound.grant_id).state is (
        DecodeReservationState.CANCELLED
    )
    assert transitions == [("cancelled", owner, None, None)]
    retry = authority.reserve_retry_response(
        bound.reservation_attempt_id,
        bound.reserve_attempt_digest,
    )
    assert retry is not None
    assert retry["state"] == "refused"
    assert retry["reason_code"] == "prepared_ttl_expired"
    assert retry["disposition"] == "retry_same_decoder"


def test_expiry_quarantines_only_ambiguous_promotion() -> None:
    clock = _FakeClock()
    owner = object()
    transitions: list[Transition] = []

    def ambiguous_promote(callback_owner: object) -> None:
        assert callback_owner is owner
        raise RuntimeError("submission outcome is ambiguous")

    authority = DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=ambiguous_promote,
        cancel=_terminal_action(
            transitions,
            DecodeReservationState.CANCELLED,
        ),
        complete=_terminal_action(
            transitions,
            DecodeReservationState.COMPLETED,
        ),
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )
    prepared = _prepare(authority, owner)
    bound, binding = _bind(authority, prepared)
    with pytest.raises(RuntimeError, match="submission outcome is ambiguous"):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.PROMOTE,
            binding,
        )

    clock.monotonic_ns += 2_500_000_000
    sweep = authority.sweep_expired_prepared()

    assert sweep.quarantined_grant_ids == (bound.grant_id,)
    assert len(sweep.cancelled_grant_ids) == 0
    assert authority.snapshot(bound.grant_id).state is (
        DecodeReservationState.QUARANTINED
    )
    assert transitions == [
        (
            "quarantined",
            owner,
            "promotion_reconciliation_expired",
            None,
        )
    ]


def test_expiry_callback_runs_outside_authority_lock() -> None:
    clock = _FakeClock()
    started = threading.Event()
    release = threading.Event()
    owner = object()

    def blocking_cancel(
        callback_owner: object,
        reason_code: str | None,
        diagnostic: str | None,
    ) -> DecodeReservationState:
        assert callback_owner is owner
        assert reason_code is None
        assert diagnostic is None
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release expiry cancellation")
        return DecodeReservationState.CANCELLED

    def promote(callback_owner: object) -> None:
        del callback_owner

    transitions: list[Transition] = []
    authority = DecodeReservationAuthority(
        expected_decoder_instance_id=uuid.UUID(_DECODER_INSTANCE_ID),
        promote=promote,
        cancel=blocking_cancel,
        complete=_terminal_action(
            transitions,
            DecodeReservationState.COMPLETED,
        ),
        abort=_terminal_action(
            transitions,
            DecodeReservationState.ABORTED,
        ),
        quarantine=_terminal_action(
            transitions,
            DecodeReservationState.QUARANTINED,
        ),
        wall_clock_unix_ms=clock.wall_clock_unix_ms,
        monotonic_clock_ns=clock.monotonic_clock_ns,
    )
    prepared = _prepare(authority, owner)
    clock.monotonic_ns += 2_500_000_000

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        sweep_future = executor.submit(authority.sweep_expired_prepared)
        assert started.wait(timeout=2)
        try:
            current = executor.submit(
                authority.snapshot,
                prepared.grant_id,
            ).result(timeout=1)
            assert current.state is DecodeReservationState.PREPARED_UNBOUND
        finally:
            release.set()
        sweep = sweep_future.result(timeout=2)
    assert sweep.cancelled_grant_ids == (prepared.grant_id,)


def test_reserve_and_grant_auth_domains_do_not_overlap() -> None:
    process_key = "process-static-secret"
    authenticate_process_bearer(f"Bearer {process_key}", process_key)

    authority = _authority([])
    prepared = _prepare(authority, object())
    registry, attachment = _attachment_registry(opaque_request=object())
    response = registry.publish_reserve_response(
        attachment.grant_id,
        prepared.reserve_response(),
    )
    grant_key = response["grant_token"]
    assert type(grant_key) is str
    authenticated = registry.authenticate(
        attachment.grant_id,
        f"Bearer {grant_key}",
    )
    assert authenticated.grant_id == attachment.grant_id

    with pytest.raises(DecodeReservationAuthenticationError):
        authenticate_process_bearer(f"Bearer {grant_key}", process_key)
    with pytest.raises(DecodeReservationAuthenticationError):
        registry.authenticate(
            attachment.grant_id,
            f"Bearer {process_key}",
        )
    with pytest.raises(DecodeReservationAuthenticationError):
        authenticate_process_bearer(None, process_key)


def test_debug_representations_redact_secrets_and_prompts() -> None:
    attempt = DecodeReservationAttempt.from_value(_reserve_request())
    opaque_request = {"input_ids": [[1, 2, 3]]}
    registry, attachment = _attachment_registry(
        opaque_request=opaque_request,
    )
    response = registry.publish_reserve_response(
        attachment.grant_id,
        {"nested": {"value": "safe"}},
    )
    token = response["grant_token"]
    assert type(token) is str

    representations = (repr(attempt), repr(attachment))
    assert all(token not in representation for representation in representations)
    assert all("input_ids" not in representation for representation in representations)
    assert "base_request_body_bytes=" in representations[0]
    assert "opaque_request" not in representations[1]


@pytest.mark.parametrize(
    ("reason_code", "diagnostic"),
    (
        ("contains space", None),
        ("slash/is/invalid", None),
        ("valid-code", "line one\nline two"),
        ("valid-code", "control\u0085character"),
    ),
)
def test_failure_context_matches_rust_validation(
    reason_code: str,
    diagnostic: str | None,
) -> None:
    authority = _authority([])
    prepared = _prepare(authority, object())
    bound, binding = _bind(authority, prepared)
    authority.transition(
        bound.grant_id,
        DecodeReservationOperation.PROMOTE,
        binding,
    )
    failure = dict(binding)
    failure["reason_code"] = reason_code
    failure["diagnostic"] = diagnostic

    with pytest.raises(DecodeReservationValidationError):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.ABORT,
            failure,
        )


def test_control_transcript_is_exact() -> None:
    authority = _authority([])
    prepared = _prepare(authority, object())
    bound, binding = _bind(authority, prepared)
    changed = copy.deepcopy(binding)
    changed["bootstrap_rooms"][1] = 43

    with pytest.raises(DecodeReservationValidationError):
        authority.transition(
            bound.grant_id,
            DecodeReservationOperation.PROMOTE,
            changed,
        )


def test_promoted_inference_attach_consumes_exact_bytes_once() -> None:
    request = object()
    registry, attachment = _attachment_registry(opaque_request=request)
    registry.bind(attachment.grant_id, _BOUND_BODY)
    registry.promote(attachment.grant_id)

    retained = registry.find_reserve_attempt(
        attachment.reservation_attempt_id,
        attachment.reserve_attempt_digest,
    )
    assert retained is not None
    assert retained.state is DecodeInferenceAttachmentState.PROMOTED
    with pytest.raises(dataclasses.FrozenInstanceError):
        retained.state = DecodeInferenceAttachmentState.TERMINAL  # type: ignore[misc]

    consumed = registry.consume("/generate", _BOUND_BODY)
    assert consumed is not None
    assert consumed.opaque_request is request
    assert consumed.state is DecodeInferenceAttachmentState.ATTACHED
    assert attachment.state is DecodeInferenceAttachmentState.PREPARED_UNBOUND
    assert registry.consume("/generate", _BOUND_BODY) is None


def test_attachment_attempt_index_rejects_conflicting_registration() -> None:
    registry, attachment = _attachment_registry(opaque_request=object())
    retained = registry.find_reserve_attempt(
        attachment.reservation_attempt_id,
        attachment.reserve_attempt_digest,
    )
    assert retained is not None
    assert retained.grant_id == attachment.grant_id

    with pytest.raises(DecodeReservationConflictError):
        registry.register(
            grant_id=uuid.uuid4(),
            reservation_attempt_id=attachment.reservation_attempt_id,
            reserve_attempt_digest=b"x" * 32,
            inference_route="/generate",
            child_request_ids=(uuid.uuid4(),),
            opaque_request=object(),
        )


def test_terminal_attachment_releases_prompt_object() -> None:
    class PromptRequest:
        pass

    request = PromptRequest()
    request_reference = weakref.ref(request)
    registry, attachment = _attachment_registry(opaque_request=request)
    grant_id = attachment.grant_id
    attempt_id = attachment.reservation_attempt_id
    attempt_digest = attachment.reserve_attempt_digest

    preterminal = registry.terminalize(grant_id)
    assert preterminal.opaque_request is request
    assert preterminal.state is DecodeInferenceAttachmentState.PREPARED_UNBOUND
    del attachment
    del preterminal
    del request
    gc.collect()

    assert request_reference() is None
    retained = registry.find_reserve_attempt(attempt_id, attempt_digest)
    assert retained is not None
    assert retained.state is DecodeInferenceAttachmentState.TERMINAL
    assert retained.opaque_request is None


def test_attachment_responses_are_isolated_copies() -> None:
    registry, attachment = _attachment_registry(opaque_request=object())
    source = {"nested": {"values": [1, 2, 3]}}
    first = registry.publish_reserve_response(attachment.grant_id, source)
    source["nested"]["values"][0] = 99
    first["nested"]["values"][1] = 98

    retained = registry.retained_reserve_response(attachment.grant_id)
    assert retained is not None
    assert retained["nested"]["values"] == [1, 2, 3]


def test_inference_attach_rejects_route_or_body_drift() -> None:
    registry, attachment = _attachment_registry(opaque_request=object())
    registry.bind(attachment.grant_id, _BOUND_BODY)
    registry.promote(attachment.grant_id)

    assert registry.consume("/v1/completions", _BOUND_BODY) is None
    assert registry.consume("/generate", _BOUND_BODY + b" ") is None
    assert registry.consume("/generate", _BOUND_BODY) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
