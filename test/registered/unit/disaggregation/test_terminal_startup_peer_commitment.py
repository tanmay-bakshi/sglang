import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import threading
import uuid
from collections.abc import Callable

import pytest
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_peer_commitment import (
    TerminalStartupPeerCommitment,
    TerminalStartupPeerCommitmentDisposition,
    TerminalStartupPeerCommitmentError,
    TerminalStartupPeerCommitmentMatrix,
    TerminalStartupPeerCommitmentRegistry,
    build_terminal_startup_peer_commitment,
    decode_terminal_startup_peer_commitment,
    decode_terminal_startup_peer_commitment_matrix,
    encode_terminal_startup_peer_commitment,
    encode_terminal_startup_peer_commitment_matrix,
    terminal_startup_peer_roster_sha256,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_SHA256 = hashlib.sha256(b"startup-peer-commitment-test").digest()


def _uuid_bytes(marker: int) -> bytes:
    """Build one readable non-nil UUID fixture.

    :param marker: Positive low-order UUID value.
    :returns: Canonical UUID bytes.
    """

    return uuid.UUID(int=marker).bytes


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    tp_rank: int,
    tp_size: int,
    generation: int,
    launch_instance: int,
    port: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact startup matrix row.

    :param service_id: Static model-service identifier.
    :param role: Source or decode service role.
    :param tp_rank: Rank within the service.
    :param tp_size: Exact service TP width.
    :param generation: Native process-generation UUID marker.
    :param launch_instance: Static launch-incarnation UUID marker.
    :param port: Canonical service-origin port.
    :returns: Validated startup advertisement.
    """

    agent_name = f"nixl-{service_id}-rank-{tp_rank}"
    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{port}",
        role=role,
        launch_instance_id=_uuid_bytes(launch_instance),
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        process_generation=_uuid_bytes(generation),
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(
            f"metadata-{agent_name}".encode("ascii")
        ).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build one TP2 source and two independent TP1 decoders.

    :returns: Complete canonical startup matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_SHA256,
        ranks=(
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tp_rank=0,
                tp_size=2,
                generation=101,
                launch_instance=1,
                port=32000,
            ),
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                tp_rank=1,
                tp_size=2,
                generation=102,
                launch_instance=1,
                port=32000,
            ),
            _rank(
                service_id="decode-a",
                role=TerminalOwnerRole.DECODE,
                tp_rank=0,
                tp_size=1,
                generation=201,
                launch_instance=2,
                port=32001,
            ),
            _rank(
                service_id="decode-b",
                role=TerminalOwnerRole.DECODE,
                tp_rank=0,
                tp_size=1,
                generation=202,
                launch_instance=3,
                port=32002,
            ),
        ),
    )


def _commitments(
    matrix: TerminalStartupCohortMatrix | None = None,
) -> tuple[TerminalStartupPeerCommitment, ...]:
    """Build every exact commitment for one fixture matrix.

    :param matrix: Optional matrix override.
    :returns: Complete canonical commitment population.
    """

    selected_matrix = _matrix() if matrix is None else matrix
    return tuple(
        build_terminal_startup_peer_commitment(
            selected_matrix,
            rank,
            tuple(peer for peer in selected_matrix.ranks if peer.role is not rank.role),
        )
        for rank in selected_matrix.ranks
    )


def _wait_for_registered_count(
    registry: TerminalStartupPeerCommitmentRegistry,
    expected_count: int,
) -> None:
    """Wait until concurrent test workers enter the registry.

    :param registry: Registry under concurrent qualification.
    :param expected_count: Exact visible registration population.
    """

    deadline = threading.Event()
    for _ in range(2_000):
        if registry.snapshot().registered_rank_count == expected_count:
            return
        deadline.wait(0.001)
    raise AssertionError("peer commitment workers did not enter the registry")


def _seal_registry(
    registry: TerminalStartupPeerCommitmentRegistry,
    commitments: tuple[TerminalStartupPeerCommitment, ...],
) -> tuple[TerminalStartupPeerCommitmentMatrix, ...]:
    """Submit a complete population concurrently.

    :param registry: Empty compatible registry.
    :param commitments: Complete commitment population in arrival order.
    :returns: Result observed by every waiter.
    """

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(commitments)
    ) as executor:
        futures = tuple(
            executor.submit(registry.register_and_wait, commitment)
            for commitment in commitments
        )
        return tuple(future.result(timeout=2.0) for future in futures)


def test_builder_binds_exact_generation_complete_opposite_role_roster() -> None:
    """The commitment cannot hide a missing, reordered, or stale peer row."""

    matrix = _matrix()
    source = matrix.rank("prefill-a", 0)
    decoder_roster = tuple(
        rank for rank in matrix.ranks if rank.role is TerminalOwnerRole.DECODE
    )
    commitment = build_terminal_startup_peer_commitment(
        matrix,
        source,
        decoder_roster,
    )

    assert commitment.local_rank is source
    assert commitment.startup_matrix_sha256 == matrix.digest
    assert commitment.opposite_role_rank_count == 2
    assert commitment.opposite_role_roster_sha256 == (
        terminal_startup_peer_roster_sha256(decoder_roster)
    )
    with pytest.raises(TerminalStartupPeerCommitmentError, match="differs"):
        build_terminal_startup_peer_commitment(
            matrix,
            source,
            decoder_roster[:-1],
        )
    with pytest.raises(TerminalStartupPeerCommitmentError, match="differs"):
        build_terminal_startup_peer_commitment(
            matrix,
            source,
            tuple(reversed(decoder_roster)),
        )

    stale_decoder = dataclasses.replace(
        decoder_roster[0],
        process_generation=_uuid_bytes(999),
    )
    stale_roster = (stale_decoder, decoder_roster[1])
    assert terminal_startup_peer_roster_sha256(stale_roster) != (
        commitment.opposite_role_roster_sha256
    )


def test_all_rank_barrier_seals_and_replay_is_byte_idempotent() -> None:
    """Every rank receives one shared result and exact replay is harmless."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    registry = TerminalStartupPeerCommitmentRegistry(matrix, timeout_seconds=2.0)
    results = _seal_registry(registry, commitments)

    assert all(result is results[0] for result in results)
    results[0].require_startup_matrix(matrix)
    assert registry.register_and_wait(commitments[0]) is results[0]
    snapshot = registry.snapshot()
    assert snapshot.disposition is TerminalStartupPeerCommitmentDisposition.SEALED
    assert snapshot.registered_rank_count == len(matrix.ranks)
    assert snapshot.commitment_matrix_sha256 == results[0].digest


def test_every_arrival_permutation_seals_identical_canonical_bytes() -> None:
    """Scheduling order cannot influence the sealed commitment matrix."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    expected = encode_terminal_startup_peer_commitment_matrix(
        TerminalStartupPeerCommitmentMatrix(
            startup_matrix_sha256=matrix.digest,
            commitments=commitments,
        )
    )

    for arrival_order in itertools.permutations(commitments):
        registry = TerminalStartupPeerCommitmentRegistry(
            matrix,
            timeout_seconds=2.0,
        )
        results = _seal_registry(registry, arrival_order)
        assert encode_terminal_startup_peer_commitment_matrix(results[0]) == expected


def test_timeout_and_manual_failure_are_sticky() -> None:
    """The first collective failure remains authoritative for every caller."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    registry = TerminalStartupPeerCommitmentRegistry(matrix, timeout_seconds=0.02)

    with pytest.raises(TerminalStartupPeerCommitmentError) as timeout:
        registry.register_and_wait(commitments[0])
    assert "did not reach all ranks" in str(timeout.value)
    registry.fail("replacement failure must not win")
    with pytest.raises(TerminalStartupPeerCommitmentError) as replay:
        registry.register_and_wait(commitments[1])
    assert str(replay.value) == str(timeout.value)
    snapshot = registry.snapshot()
    assert snapshot.disposition is TerminalStartupPeerCommitmentDisposition.FAILED
    assert snapshot.failure_reason == str(timeout.value)


def test_conflicting_duplicate_fails_the_previously_sealed_epoch() -> None:
    """Only a byte-identical commitment may replay one static rank key."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    registry = TerminalStartupPeerCommitmentRegistry(matrix, timeout_seconds=2.0)
    _seal_registry(registry, commitments)
    conflicting = dataclasses.replace(
        commitments[0],
        opposite_role_roster_sha256=bytes.fromhex("ab" * 32),
    )

    with pytest.raises(TerminalStartupPeerCommitmentError, match="conflicting"):
        registry.register_and_wait(conflicting)
    with pytest.raises(TerminalStartupPeerCommitmentError, match="conflicting"):
        registry.register_and_wait(commitments[0])
    assert (
        registry.snapshot().disposition
        is TerminalStartupPeerCommitmentDisposition.FAILED
    )


def test_same_role_roster_disagreement_fails_and_wakes_existing_waiter() -> None:
    """Every source and every decoder must independently agree by local role."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    registry = TerminalStartupPeerCommitmentRegistry(matrix, timeout_seconds=2.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(registry.register_and_wait, commitments[0])
        _wait_for_registered_count(registry, 1)
        disagreement = dataclasses.replace(
            commitments[1],
            opposite_role_roster_sha256=bytes.fromhex("cd" * 32),
        )
        with pytest.raises(
            TerminalStartupPeerCommitmentError,
            match="one local role disagree",
        ):
            registry.register_and_wait(disagreement)
        with pytest.raises(
            TerminalStartupPeerCommitmentError,
            match="one local role disagree",
        ):
            waiting.result(timeout=2.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda commitment: dataclasses.replace(
                commitment,
                startup_matrix_sha256=bytes.fromhex("01" * 32),
            ),
            "another startup matrix",
        ),
        (
            lambda commitment: dataclasses.replace(
                commitment,
                local_rank=dataclasses.replace(
                    commitment.local_rank,
                    process_generation=_uuid_bytes(999),
                ),
            ),
            "local row differs",
        ),
        (
            lambda commitment: dataclasses.replace(
                commitment,
                opposite_role_rank_count=commitment.opposite_role_rank_count + 1,
            ),
            "peer roster differs",
        ),
        (
            lambda commitment: dataclasses.replace(
                commitment,
                opposite_role_roster_sha256=bytes.fromhex("02" * 32),
            ),
            "peer roster differs",
        ),
    ),
)
def test_registry_rejects_every_dimension_of_matrix_drift(
    mutate: Callable[[TerminalStartupPeerCommitment], TerminalStartupPeerCommitment],
    message: str,
) -> None:
    """A locally plausible commitment cannot drift from the sealed matrix.

    :param mutate: Typed fixture mutation callable supplied by pytest.
    :param message: Expected validation evidence.
    """

    matrix = _matrix()
    commitment = _commitments(matrix)[0]
    registry = TerminalStartupPeerCommitmentRegistry(matrix, timeout_seconds=2.0)
    mutated = mutate(commitment)

    with pytest.raises(TerminalStartupPeerCommitmentError, match=message):
        registry.register_and_wait(mutated)
    assert (
        registry.snapshot().disposition
        is TerminalStartupPeerCommitmentDisposition.FAILED
    )


def test_commitment_matrix_rejects_same_role_disagreement_structurally() -> None:
    """A decoded matrix cannot represent split-brain source roster claims."""

    matrix = _matrix()
    commitments = list(_commitments(matrix))
    commitments[1] = dataclasses.replace(
        commitments[1],
        opposite_role_roster_sha256=bytes.fromhex("ef" * 32),
    )

    with pytest.raises(ValueError, match="one local role disagree"):
        TerminalStartupPeerCommitmentMatrix(
            startup_matrix_sha256=matrix.digest,
            commitments=tuple(commitments),
        )


def test_wire_codecs_are_strict_canonical_and_duplicate_free() -> None:
    """Commitments and sealed matrices each have one accepted byte encoding."""

    matrix = _matrix()
    commitments = _commitments(matrix)
    commitment = commitments[0]
    encoded = encode_terminal_startup_peer_commitment(commitment)
    assert decode_terminal_startup_peer_commitment(encoded) == commitment
    with pytest.raises(TerminalStartupPeerCommitmentError, match="not canonical"):
        decode_terminal_startup_peer_commitment(b" " + encoded)
    duplicate_schema = encoded.replace(
        b'{"schema":',
        b'{"schema":"duplicate","schema":',
        1,
    )
    with pytest.raises(TerminalStartupPeerCommitmentError, match="duplicate"):
        decode_terminal_startup_peer_commitment(duplicate_schema)

    unknown = json.loads(encoded)
    unknown["unknown"] = True
    with pytest.raises(TerminalStartupPeerCommitmentError, match="field set"):
        decode_terminal_startup_peer_commitment(
            json.dumps(unknown, separators=(",", ":")).encode()
        )
    boolean_count = json.loads(encoded)
    boolean_count["opposite_role_rank_count"] = True
    with pytest.raises(TerminalStartupPeerCommitmentError, match="integer"):
        decode_terminal_startup_peer_commitment(
            json.dumps(boolean_count, separators=(",", ":")).encode()
        )

    sealed = TerminalStartupPeerCommitmentMatrix(
        startup_matrix_sha256=matrix.digest,
        commitments=commitments,
    )
    encoded_matrix = encode_terminal_startup_peer_commitment_matrix(sealed)
    decoded_matrix = decode_terminal_startup_peer_commitment_matrix(encoded_matrix)
    assert decoded_matrix == sealed
    decoded_matrix.require_startup_matrix(matrix)
    noncanonical_order = json.loads(encoded_matrix)
    noncanonical_order["commitments"].reverse()
    with pytest.raises(
        TerminalStartupPeerCommitmentError,
        match="fields are invalid",
    ):
        decode_terminal_startup_peer_commitment_matrix(
            json.dumps(noncanonical_order, separators=(",", ":")).encode()
        )


def test_internally_agreed_but_wrong_roster_cannot_validate_against_matrix() -> None:
    """Role-wide agreement does not substitute for the expected exact roster."""

    matrix = _matrix()
    commitments = tuple(
        dataclasses.replace(
            commitment,
            opposite_role_roster_sha256=(
                bytes.fromhex("11" * 32)
                if commitment.local_rank.role is TerminalOwnerRole.SOURCE
                else bytes.fromhex("22" * 32)
            ),
        )
        for commitment in _commitments(matrix)
    )
    sealed = TerminalStartupPeerCommitmentMatrix(
        startup_matrix_sha256=matrix.digest,
        commitments=commitments,
    )

    with pytest.raises(TerminalStartupPeerCommitmentError, match="peer roster differs"):
        sealed.require_startup_matrix(matrix)
