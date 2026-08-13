import concurrent.futures
import dataclasses
import hashlib
import json
import time
import uuid

import pytest
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalProducerClass,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortDisposition,
    TerminalStartupCohortError,
    TerminalStartupCohortExpectation,
    TerminalStartupCohortMatrix,
    TerminalStartupCohortRegistry,
    TerminalStartupRankAdvertisement,
    TerminalStartupServiceExpectation,
    decode_terminal_startup_cohort_matrix,
    decode_terminal_startup_rank_advertisement,
    encode_terminal_startup_cohort_matrix,
    encode_terminal_startup_rank_advertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_SHA256 = hashlib.sha256(b"terminal-startup-test-cohort").digest()


def _uuid_bytes(marker: int) -> bytes:
    """Build one readable non-nil UUID fixture.

    :param marker: Positive low-order UUID value.
    :returns: Canonical UUID bytes.
    """

    return uuid.UUID(int=marker).bytes


def _expectation() -> TerminalStartupCohortExpectation:
    """Build one TP2 source with two independent TP1 decoders.

    :returns: Complete static group expectation.
    """

    return TerminalStartupCohortExpectation(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        services=(
            TerminalStartupServiceExpectation(
                service_id="prefill-a",
                service_origin="http://127.0.0.1:31000",
                role=TerminalOwnerRole.SOURCE,
                launch_instance_id=_uuid_bytes(1),
                tensor_parallel_size=2,
            ),
            TerminalStartupServiceExpectation(
                service_id="decode-a",
                service_origin="http://127.0.0.1:31001",
                role=TerminalOwnerRole.DECODE,
                launch_instance_id=_uuid_bytes(2),
                tensor_parallel_size=1,
            ),
            TerminalStartupServiceExpectation(
                service_id="decode-b",
                service_origin="http://127.0.0.1:31002",
                role=TerminalOwnerRole.DECODE,
                launch_instance_id=_uuid_bytes(3),
                tensor_parallel_size=1,
            ),
        ),
    )


def _advertisements() -> tuple[TerminalStartupRankAdvertisement, ...]:
    """Build the complete observed native identity population.

    :returns: Canonically ordered startup advertisements.
    """

    expectation = _expectation()
    advertisements: list[TerminalStartupRankAdvertisement] = []
    generation_marker = 100
    for service in expectation.services:
        for rank in range(service.tensor_parallel_size):
            agent_name = f"nixl-{service.service_id}-rank-{rank}"
            advertisements.append(
                TerminalStartupRankAdvertisement(
                    group_id=expectation.group_id,
                    cohort_sha256=expectation.cohort_sha256,
                    service_id=service.service_id,
                    service_origin=service.service_origin,
                    role=service.role,
                    launch_instance_id=service.launch_instance_id,
                    tensor_parallel_rank=rank,
                    tensor_parallel_size=service.tensor_parallel_size,
                    process_generation=_uuid_bytes(generation_marker),
                    nixl_agent_name=agent_name,
                    nixl_agent_metadata_sha256=hashlib.sha256(
                        agent_name.encode("ascii")
                    ).digest(),
                )
            )
            generation_marker += 1
    return tuple(advertisements)


def _wait_for_registered_count(
    registry: TerminalStartupCohortRegistry,
    expected_count: int,
) -> None:
    """Wait for test workers to enter the registry.

    :param registry: Registry under concurrent test.
    :param expected_count: Exact population that must be visible.
    """

    deadline = time.monotonic() + 2.0
    while registry.snapshot().registered_rank_count != expected_count:
        if time.monotonic() >= deadline:
            raise AssertionError("startup registry workers did not arrive")
        time.sleep(0.001)


def _seal_registry(
    registry: TerminalStartupCohortRegistry,
) -> TerminalStartupCohortMatrix:
    """Join every fixture rank and return their shared sealed matrix.

    :param registry: Empty compatible startup registry.
    :returns: Complete immutable matrix.
    """

    advertisements = _advertisements()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(advertisements)
    ) as executor:
        futures = tuple(
            executor.submit(registry.register_and_wait, advertisement)
            for advertisement in advertisements
        )
        matrices = tuple(future.result(timeout=2.0) for future in futures)
    assert all(matrix == matrices[0] for matrix in matrices)
    return matrices[0]


def test_complete_matrix_seals_and_releases_every_waiter_together() -> None:
    """No rank observes a partial native producer population."""

    advertisements = _advertisements()
    registry = TerminalStartupCohortRegistry(_expectation(), timeout_seconds=2.0)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(advertisements)
    ) as executor:
        waiting = tuple(
            executor.submit(registry.register_and_wait, advertisement)
            for advertisement in advertisements[:-1]
        )
        _wait_for_registered_count(registry, len(advertisements) - 1)
        assert all(not future.done() for future in waiting)

        final = executor.submit(registry.register_and_wait, advertisements[-1])
        matrices = tuple(future.result(timeout=2.0) for future in (*waiting, final))

    assert all(matrix == matrices[0] for matrix in matrices)
    matrices[0].require_expectation(_expectation())
    snapshot = registry.snapshot()
    assert snapshot.disposition is TerminalStartupCohortDisposition.SEALED
    assert snapshot.matrix_digest == matrices[0].digest


def test_timeout_is_sticky_and_wakes_every_waiter() -> None:
    """One missing rank fails the whole deployment epoch."""

    registry = TerminalStartupCohortRegistry(_expectation(), timeout_seconds=0.05)
    advertisements = _advertisements()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(registry.register_and_wait, advertisement)
            for advertisement in advertisements[:2]
        )
        errors: list[TerminalStartupCohortError] = []
        for future in futures:
            with pytest.raises(TerminalStartupCohortError) as captured:
                future.result(timeout=2.0)
            errors.append(captured.value)

    assert len({str(error) for error in errors}) == 1
    assert "complete membership" in str(errors[0])
    snapshot = registry.snapshot()
    assert snapshot.disposition is TerminalStartupCohortDisposition.FAILED
    assert snapshot.registered_rank_count == 2


def test_conflicting_generation_collectively_fails_open_waiters() -> None:
    """A static rank cannot replace its process while the epoch joins."""

    registry = TerminalStartupCohortRegistry(_expectation(), timeout_seconds=2.0)
    first = _advertisements()[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(registry.register_and_wait, first)
        _wait_for_registered_count(registry, 1)
        replacement = dataclasses.replace(
            first,
            process_generation=_uuid_bytes(999),
        )
        with pytest.raises(
            TerminalStartupCohortError, match="change native generation"
        ):
            registry.register_and_wait(replacement)
        with pytest.raises(
            TerminalStartupCohortError, match="change native generation"
        ):
            waiting.result(timeout=2.0)

    assert registry.snapshot().disposition is TerminalStartupCohortDisposition.FAILED


def test_static_service_identity_drift_fails_the_complete_epoch() -> None:
    """Origins and launch identities cannot be chosen by live processes."""

    registry = TerminalStartupCohortRegistry(_expectation(), timeout_seconds=2.0)
    advertisement = dataclasses.replace(
        _advertisements()[0],
        service_origin="http://127.0.0.1:31999",
    )

    with pytest.raises(TerminalStartupCohortError, match="static service membership"):
        registry.register_and_wait(advertisement)
    assert registry.snapshot().disposition is TerminalStartupCohortDisposition.FAILED


def test_sealed_retry_is_idempotent_but_replacement_fails_closed() -> None:
    """Only byte-identical rank retries survive cohort sealing."""

    registry = TerminalStartupCohortRegistry(_expectation(), timeout_seconds=2.0)
    matrix = _seal_registry(registry)
    original = _advertisements()[0]

    assert registry.register_and_wait(original) is matrix
    replacement = dataclasses.replace(
        original,
        process_generation=_uuid_bytes(999),
    )
    with pytest.raises(TerminalStartupCohortError, match="change native generation"):
        registry.register_and_wait(replacement)
    assert registry.snapshot().disposition is TerminalStartupCohortDisposition.FAILED


def test_wire_codecs_require_exact_canonical_duplicate_free_json() -> None:
    """Startup identities have one accepted byte representation."""

    advertisement = _advertisements()[0]
    encoded_advertisement = encode_terminal_startup_rank_advertisement(advertisement)
    assert (
        decode_terminal_startup_rank_advertisement(encoded_advertisement)
        == advertisement
    )
    with pytest.raises(TerminalStartupCohortError, match="not canonical"):
        decode_terminal_startup_rank_advertisement(b" " + encoded_advertisement)
    duplicate_schema = encoded_advertisement.replace(
        b'{"schema":',
        b'{"schema":"duplicate","schema":',
        1,
    )
    with pytest.raises(TerminalStartupCohortError, match="duplicate"):
        decode_terminal_startup_rank_advertisement(duplicate_schema)

    unknown_advertisement = json.loads(encoded_advertisement)
    unknown_advertisement["rank"]["unknown"] = 1
    with pytest.raises(TerminalStartupCohortError, match="field set"):
        decode_terminal_startup_rank_advertisement(
            json.dumps(
                unknown_advertisement,
                separators=(",", ":"),
            ).encode()
        )
    malformed_origin = json.loads(encoded_advertisement)
    malformed_origin["rank"]["service_origin"] = "http://127.0.0.1:31000/route"
    with pytest.raises(TerminalStartupCohortError, match="fields are invalid"):
        decode_terminal_startup_rank_advertisement(
            json.dumps(malformed_origin, separators=(",", ":")).encode()
        )

    matrix = TerminalStartupCohortMatrix(
        group_id=_expectation().group_id,
        cohort_sha256=_expectation().cohort_sha256,
        ranks=_advertisements(),
    )
    matrix.require_expectation(_expectation())
    encoded_matrix = encode_terminal_startup_cohort_matrix(matrix)
    assert decode_terminal_startup_cohort_matrix(encoded_matrix) == matrix
    unknown_matrix = json.loads(encoded_matrix)
    unknown_matrix["unknown"] = 1
    with pytest.raises(TerminalStartupCohortError, match="field set"):
        decode_terminal_startup_cohort_matrix(
            json.dumps(unknown_matrix, separators=(",", ":")).encode()
        )


def test_producer_plan_excludes_unrelated_decode_replicas() -> None:
    """Producer authority follows role edges and exact TP service membership."""

    matrix = TerminalStartupCohortMatrix(
        group_id=_expectation().group_id,
        cohort_sha256=_expectation().cohort_sha256,
        ranks=_advertisements(),
    )
    source_plan = build_terminal_startup_python_producer_plan(
        matrix,
        local_service_id="prefill-a",
        local_tensor_parallel_rank=0,
        first_producer_id=10,
    )
    source_classes = tuple(
        spec.registration.producer_class for spec in source_plan.specs
    )
    assert source_classes.count(NativeTerminalProducerClass.LOCAL) == 1
    assert source_classes.count(NativeTerminalProducerClass.CONTROL) == 2
    assert source_classes.count(NativeTerminalProducerClass.RECEIPT) == 4
    assert source_plan.fatal_producer_id == 10
    assert source_plan.next_producer_id == 17

    decode_plan = build_terminal_startup_python_producer_plan(
        matrix,
        local_service_id="decode-a",
        local_tensor_parallel_rank=0,
        first_producer_id=30,
    )
    decode_classes = tuple(
        spec.registration.producer_class for spec in decode_plan.specs
    )
    assert decode_classes.count(NativeTerminalProducerClass.LOCAL) == 1
    assert decode_classes.count(NativeTerminalProducerClass.CONTROL) == 2
    assert decode_classes.count(NativeTerminalProducerClass.RECEIPT) == 3
    decode_b_generation = matrix.rank("decode-b", 0).process_generation
    authenticated_generations = {
        spec.registration.authenticated_issuer.process_generation
        for spec in decode_plan.specs
        if spec.registration.authenticated_issuer is not None
    }
    assert decode_b_generation not in authenticated_generations
    assert decode_plan.next_producer_id == 36
