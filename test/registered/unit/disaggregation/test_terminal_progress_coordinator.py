import dataclasses

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.coordinator import (
    TerminalRequestCoordinator,
    TerminalRequestCoordinatorDisposition,
    TerminalRequestCoordinatorError,
    TerminalRequestCoordinatorManifest,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptImportNamespace,
    TerminalWireReceiptIssuer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _identity(
    seed: int, role: TerminalOwnerRole, rank: int, size: int
) -> TerminalProcessIdentity:
    return TerminalProcessIdentity(
        process_generation=bytes((seed,)) * 16,
        role=role,
        tp_rank=rank,
        tp_size=size,
    )


def _binding(
    key: PackedRequestKey,
    owner: TerminalProcessIdentity,
    allocation_seed: int,
) -> TerminalRequestBinding:
    return TerminalRequestBinding(
        request_key=key,
        owner=owner,
        rank_manifest_digest=b"m" * 32,
        allocation_digest=bytes((allocation_seed,)) * 32,
    )


def _coordinator() -> tuple[
    TerminalRequestCoordinator,
    tuple[TerminalRequestBinding, ...],
    tuple[TerminalRequestBinding, ...],
    tuple[TerminalWireReceiptIssuer, ...],
]:
    key = PackedRequestKey(room_id=73, request_generation=b"g" * 16)
    destinations = tuple(
        _binding(
            key,
            _identity(10 + rank, TerminalOwnerRole.DECODE, rank, 2),
            20 + rank,
        )
        for rank in range(2)
    )
    sources = tuple(
        _binding(
            key,
            _identity(30 + rank, TerminalOwnerRole.SOURCE, rank, 2),
            40 + rank,
        )
        for rank in range(2)
    )
    local_ready_issuers = tuple(
        TerminalWireReceiptIssuer(binding.owner) for binding in destinations
    )
    coordinator_issuer = TerminalWireReceiptIssuer(destinations[0].owner)
    importers = tuple(
        TerminalWireReceiptImportNamespace(binding.owner) for binding in destinations
    )
    manifest = TerminalRequestCoordinatorManifest(
        request_key=key,
        destination_bindings=destinations,
        recipient_bindings=(*destinations, *sources),
    )
    return (
        TerminalRequestCoordinator(
            manifest=manifest,
            issuer=coordinator_issuer,
            importers=importers,
        ),
        destinations,
        sources,
        local_ready_issuers,
    )


def _local_ready(
    issuer: TerminalWireReceiptIssuer,
    binding: TerminalRequestBinding,
    timestamp_ns: int,
):
    return issuer.issue(
        binding=binding,
        kind=TerminalReceiptKind.LOCAL_DECODE_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=timestamp_ns,
    ).wire_receipt


def test_coordinator_fans_out_only_after_complete_rank_manifest() -> None:
    coordinator, destinations, sources, issuers = _coordinator()
    rank_one = _local_ready(issuers[1], destinations[1], 100)
    rank_zero = _local_ready(issuers[0], destinations[0], 110)

    first = coordinator.accept(rank_one, destinations[1].owner, 1_000)
    complete = coordinator.accept(rank_zero, destinations[0].owner, 1_250)

    assert first.disposition is TerminalRequestCoordinatorDisposition.COLLECTING
    assert first.accepted_rank_count == 1
    assert first.emissions == ()
    assert complete.disposition is TerminalRequestCoordinatorDisposition.READY
    assert complete.accepted_rank_count == 2
    assert complete.newly_terminal
    assert complete.timing is not None
    assert complete.timing.duration_ns == 250
    assert tuple(emission.recipient for emission in complete.emissions) == (
        *destinations,
        *sources,
    )
    assert all(
        emission.receipt.wire_receipt.kind is TerminalReceiptKind.REQUEST_READY
        for emission in complete.emissions
    )
    assert all(
        emission.receipt.wire_receipt.outcome is TerminalReceiptOutcome.SUCCESS
        for emission in complete.emissions
    )

    duplicate = coordinator.accept(rank_one, destinations[1].owner, 1_300)
    assert duplicate.disposition is TerminalRequestCoordinatorDisposition.READY
    assert not duplicate.newly_terminal
    assert duplicate.emissions == ()

    coordinator.close()


def test_failure_wins_over_incomplete_success() -> None:
    coordinator, destinations, _, issuers = _coordinator()
    rank_zero = _local_ready(issuers[0], destinations[0], 100)
    coordinator.accept(rank_zero, destinations[0].owner, 1_000)
    failure = (
        issuers[1]
        .issue(
            binding=destinations[1],
            kind=TerminalReceiptKind.FAILURE,
            outcome=TerminalReceiptOutcome.FAILURE,
            terminal_timestamp_ns=110,
        )
        .wire_receipt
    )

    result = coordinator.accept(failure, destinations[1].owner, 1_100)

    assert result.disposition is TerminalRequestCoordinatorDisposition.FAILED
    assert result.newly_terminal
    assert all(
        emission.receipt.wire_receipt.kind is TerminalReceiptKind.FAILURE
        for emission in result.emissions
    )


def test_coordinator_deadline_is_anchored_once_at_first_receipt() -> None:
    coordinator, destinations, _, issuers = _coordinator()
    assert coordinator.deadline_expires_ns is None
    first = _local_ready(issuers[0], destinations[0], 100)
    coordinator.accept(first, destinations[0].owner, 1_000)
    timeout_ns = terminal_deadline_spec(
        TerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY
    ).duration_ns
    assert coordinator.deadline_expires_ns == 1_000 + timeout_ns

    before = coordinator.expire(1_000 + timeout_ns - 1)
    expired = coordinator.expire(1_000 + timeout_ns)

    assert before.disposition is TerminalRequestCoordinatorDisposition.COLLECTING
    assert expired.disposition is TerminalRequestCoordinatorDisposition.FAILED
    assert expired.newly_terminal
    assert expired.timing is not None
    assert expired.timing.duration_ns == timeout_ns


def test_unpublished_coordinator_cancellation_releases_replay_state() -> None:
    """A coordinator with no input has one explicit safe rollback boundary."""

    coordinator, destinations, _, issuers = _coordinator()

    coordinator.cancel_unpublished()

    with pytest.raises(TerminalRequestCoordinatorError, match="closed"):
        coordinator.accept(
            _local_ready(issuers[0], destinations[0], 100),
            destinations[0].owner,
            1_000,
        )


def test_published_coordinator_cannot_use_unpublished_rollback() -> None:
    """Receipt admission permanently closes the nonterminal rollback path."""

    coordinator, destinations, _, issuers = _coordinator()
    coordinator.accept(
        _local_ready(issuers[0], destinations[0], 100),
        destinations[0].owner,
        1_000,
    )

    with pytest.raises(TerminalRequestCoordinatorError, match="published"):
        coordinator.cancel_unpublished()


def test_conflicting_rank_receipt_fails_closed() -> None:
    coordinator, destinations, _, issuers = _coordinator()
    first = _local_ready(issuers[0], destinations[0], 100)
    coordinator.accept(first, destinations[0].owner, 1_000)
    conflicting = dataclasses.replace(first, terminal_timestamp_ns=101)

    with pytest.raises(TerminalRequestCoordinatorError, match="conflicting"):
        coordinator.accept(conflicting, destinations[0].owner, 1_001)

    assert coordinator.disposition is TerminalRequestCoordinatorDisposition.FAILED
    assert len(coordinator.terminal_emissions) == 4


def test_manifest_requires_complete_canonical_destination_ranks() -> None:
    key = PackedRequestKey(room_id=73, request_generation=b"g" * 16)
    rank_one = _binding(
        key,
        _identity(11, TerminalOwnerRole.DECODE, 1, 2),
        21,
    )

    with pytest.raises(ValueError, match="every TP rank"):
        TerminalRequestCoordinatorManifest(
            request_key=key,
            destination_bindings=(rank_one,),
            recipient_bindings=(rank_one,),
        )
