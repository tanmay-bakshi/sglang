import dataclasses
import itertools

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.executor import (
    DeterministicEmission,
    DeterministicExecutor,
    DeterministicTransition,
    replay_deterministic_trace,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DECODE_RESOURCE_KINDS,
    SOURCE_RESOURCE_KINDS,
    DecodeLifecycle,
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
    DecodeLifecyclePhase,
    SourceLifecycle,
    SourceLifecycleEvent,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
    TerminalProcessDisposition,
    TerminalResourceInventory,
    TerminalResourceKind,
    create_decode_lifecycle,
    create_source_lifecycle,
    reduce_decode_lifecycle,
    reduce_source_lifecycle,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
_PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_PUBLICATION_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")


@dataclasses.dataclass(frozen=True, slots=True)
class _OwnerPairState:
    """Complete immutable source and decode model state."""

    source: SourceLifecycle
    decode: DecodeLifecycle


@dataclasses.dataclass(frozen=True, slots=True)
class _OwnerPairEvent:
    """Exactly one source-side or decode-side lifecycle event."""

    source: SourceLifecycleEvent | None = None
    decode: DecodeLifecycleEvent | None = None

    def __post_init__(self) -> None:
        """Require one and only one event branch."""

        if (self.source is None) == (self.decode is None):
            raise ValueError("exactly one owner event branch is required")


def _binding(role: TerminalOwnerRole) -> TerminalRequestBinding:
    """Create one exact role-specific request binding.

    :param role: Source or decode owner role.
    :returns: Immutable request and allocation binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=71,
            request_generation=_REQUEST_GENERATION,
        ),
        owner=TerminalProcessIdentity(
            process_generation=_PROCESS_GENERATION,
            role=role,
            tp_rank=0,
            tp_size=2,
        ),
        rank_manifest_digest=b"r" * 32,
        allocation_digest=b"a" * 32,
    )


def _initial_state(
    issuer: TerminalReceiptIssuer,
) -> _OwnerPairState:
    """Create all-live source and decode lifecycle records.

    :param issuer: Authority trusted by both local lifecycle models.
    :returns: Complete initial pair state.
    """

    source_binding = _binding(TerminalOwnerRole.SOURCE)
    authority = TerminalReceiptLedger(authorities=frozenset((issuer.authority,)))
    return _OwnerPairState(
        source=create_source_lifecycle(
            binding=source_binding,
            publication_identity=TerminalPublicationIdentity(
                request_key=source_binding.request_key,
                publisher_process_generation=_PROCESS_GENERATION,
                publication_generation=_PUBLICATION_GENERATION,
            ),
            receipt_ledger=authority,
        ),
        decode=create_decode_lifecycle(
            binding=_binding(TerminalOwnerRole.DECODE),
            receipt_ledger=authority,
        ),
    )


def _receipt_event(
    issuer: TerminalReceiptIssuer,
    binding: TerminalRequestBinding,
    kind: TerminalReceiptKind,
    event_kind: SourceLifecycleEventKind | DecodeLifecycleEventKind,
) -> SourceLifecycleEvent | DecodeLifecycleEvent:
    """Build one receipt-bearing event of the requested owner role.

    :param issuer: Trusted receipt authority.
    :param binding: Exact target lifecycle binding.
    :param kind: Authority represented by the receipt.
    :param event_kind: Lifecycle transition consuming the receipt.
    :returns: Immutable source or decode event.
    """

    receipt = issuer.issue(
        binding=binding,
        kind=kind,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=1,
    )
    if type(event_kind) is SourceLifecycleEventKind:
        return SourceLifecycleEvent(kind=event_kind, receipt=receipt)
    if type(event_kind) is DecodeLifecycleEventKind:
        return DecodeLifecycleEvent(kind=event_kind, receipt=receipt)
    raise TypeError("event_kind must be a source or decode lifecycle event")


def _source_path(
    issuer: TerminalReceiptIssuer,
    binding: TerminalRequestBinding,
    join_order: tuple[str, str],
) -> tuple[SourceLifecycleEvent, ...]:
    """Build one complete source path with an explicit terminal join order.

    :param issuer: Trusted receipt authority.
    :param binding: Exact source lifecycle binding.
    :param join_order: Reclaim and publication terminal-join order.
    :returns: Complete immutable source event path.
    """

    forward = tuple(
        SourceLifecycleEvent(kind=kind)
        for kind in (
            SourceLifecycleEventKind.SUBMISSION_ACCEPTED,
            SourceLifecycleEventKind.PRODUCER_COMPLETED,
            SourceLifecycleEventKind.GATHER_COMPLETED_AND_NATIVE_POSTED,
            SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL,
            SourceLifecycleEventKind.OUTCOMES_SENT,
            SourceLifecycleEventKind.TEARDOWN_RECEIVED,
            SourceLifecycleEventKind.ACK_SENT,
        )
    )
    ready = _receipt_event(
        issuer,
        binding,
        TerminalReceiptKind.REQUEST_READY,
        SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
    )
    joins = {
        "reclaim": _receipt_event(
            issuer,
            binding,
            TerminalReceiptKind.RECLAIM_CONSUMED,
            SourceLifecycleEventKind.RECLAIM_CONSUMED,
        ),
        "publish": _receipt_event(
            issuer,
            binding,
            TerminalReceiptKind.GATEWAY_PUBLISHED,
            SourceLifecycleEventKind.GATEWAY_PUBLISHED,
        ),
    }
    assert type(ready) is SourceLifecycleEvent
    return (*forward, ready, *(joins[name] for name in join_order))


def _decode_path(
    issuer: TerminalReceiptIssuer,
    binding: TerminalRequestBinding,
) -> tuple[DecodeLifecycleEvent, ...]:
    """Build one complete decode adoption and request-ready path.

    :param issuer: Trusted receipt authority.
    :param binding: Exact decode lifecycle binding.
    :returns: Complete immutable decode event path.
    """

    forward = tuple(
        DecodeLifecycleEvent(kind=kind)
        for kind in (
            DecodeLifecycleEventKind.ALLOCATION_PUBLISHED,
            DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
            DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
            DecodeLifecycleEventKind.SCATTER_STARTED,
            DecodeLifecycleEventKind.SCATTER_TERMINAL,
            DecodeLifecycleEventKind.TEARDOWN_SENT,
            DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
            DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
        )
    )
    adoption = _receipt_event(
        issuer,
        binding,
        TerminalReceiptKind.ADOPTION_READY,
        DecodeLifecycleEventKind.ADOPTION_CONSUMED,
    )
    metadata = DecodeLifecycleEvent(kind=DecodeLifecycleEventKind.METADATA_CONSUMED)
    request_ready = _receipt_event(
        issuer,
        binding,
        TerminalReceiptKind.REQUEST_READY,
        DecodeLifecycleEventKind.REQUEST_READY_RECEIVED,
    )
    assert type(adoption) is DecodeLifecycleEvent
    assert type(metadata) is DecodeLifecycleEvent
    assert type(request_ready) is DecodeLifecycleEvent
    return (
        *forward,
        adoption,
        metadata,
        DecodeLifecycleEvent(kind=DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED),
        request_ready,
    )


def _reduce_pair(
    state: _OwnerPairState,
    event: _OwnerPairEvent,
) -> DeterministicTransition[_OwnerPairState, _OwnerPairEvent]:
    """Apply one exact event to its immutable owner state.

    :param state: Complete lifecycle pair before the event.
    :param event: Exact source or decode event.
    :returns: Complete lifecycle pair after the event.
    """

    if event.source is not None:
        current = reduce_source_lifecycle(state.source, event.source).current
        return DeterministicTransition(state=dataclasses.replace(state, source=current))
    if event.decode is None:
        raise RuntimeError("validated pair event lost both branches")
    current = reduce_decode_lifecycle(state.decode, event.decode).current
    return DeterministicTransition(state=dataclasses.replace(state, decode=current))


def _assert_conserved(inventory: TerminalResourceInventory) -> None:
    """Assert that every generation-bound resource has one disposition.

    :param inventory: Exact owner inventory to validate.
    """

    assert inventory.live.isdisjoint(inventory.safely_retired)
    assert inventory.live.isdisjoint(inventory.quarantined)
    assert inventory.safely_retired.isdisjoint(inventory.quarantined)
    assert (
        inventory.live | inventory.safely_retired | inventory.quarantined
        == inventory.universe
    )


@pytest.mark.parametrize(
    "join_order",
    tuple(itertools.permutations(("reclaim", "publish"))),
)
def test_deterministic_interleavings_replay_with_exact_resource_conservation(
    join_order: tuple[str, str],
) -> None:
    """Real source/decode reducers conserve every resource under replay."""

    issuer = TerminalReceiptIssuer()
    initial = _initial_state(issuer)
    source_events = _source_path(issuer, initial.source.binding, join_order)
    decode_events = _decode_path(issuer, initial.decode.binding)
    emissions: list[DeterministicEmission[_OwnerPairEvent]] = []
    for source_event, decode_event in itertools.zip_longest(
        source_events,
        decode_events,
    ):
        if source_event is not None:
            emissions.append(
                DeterministicEmission(
                    label=f"source.{source_event.kind.value}",
                    event=_OwnerPairEvent(source=source_event),
                )
            )
        if decode_event is not None:
            emissions.append(
                DeterministicEmission(
                    label=f"decode.{decode_event.kind.value}",
                    event=_OwnerPairEvent(decode=decode_event),
                )
            )
    roots = tuple(emissions)
    executor: DeterministicExecutor[_OwnerPairState, _OwnerPairEvent] = (
        DeterministicExecutor.create(initial)
    )
    drained = executor.enqueue_many(roots).drain(
        _reduce_pair,
        maximum_steps=len(roots),
    )

    assert drained.state.source.phase is SourceLifecyclePhase.RETIRED
    assert drained.state.decode.phase is DecodeLifecyclePhase.RETIRED
    assert drained.state.source.inventory.safely_retired == SOURCE_RESOURCE_KINDS
    assert drained.state.decode.inventory.safely_retired == DECODE_RESOURCE_KINDS
    assert TerminalResourceKind.SOURCE_DFLASH_AUX_VRAM_ROWS in (
        drained.state.source.inventory.universe
    )
    assert TerminalResourceKind.DECODE_DFLASH_AUX_VRAM_ROWS in (
        drained.state.decode.inventory.universe
    )
    for entry in drained.trace:
        _assert_conserved(entry.state_before.source.inventory)
        _assert_conserved(entry.state_before.decode.inventory)
        _assert_conserved(entry.state_after.source.inventory)
        _assert_conserved(entry.state_after.decode.inventory)

    replayed = replay_deterministic_trace(
        initial_state=initial,
        root_emissions=roots,
        reducer=_reduce_pair,
        expected_trace=drained.trace,
        maximum_steps=len(roots),
    )
    assert replayed == drained


@pytest.mark.parametrize(
    "join_order",
    tuple(itertools.permutations(("reclaim", "publish"))),
)
def test_source_owner_death_at_every_prefix_conserves_all_resources(
    join_order: tuple[str, str],
) -> None:
    """Every nonterminal source prefix has a complete fail-closed inventory."""

    for prefix_length in range(10):
        issuer = TerminalReceiptIssuer()
        initial = _initial_state(issuer)
        events = _source_path(issuer, initial.source.binding, join_order)
        current = initial.source
        for event in events[:prefix_length]:
            current = reduce_source_lifecycle(current, event).current
        failed = reduce_source_lifecycle(
            current,
            SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.OWNER_DIED,
                reason="synthetic owner loss",
            ),
        ).current
        _assert_conserved(failed.inventory)
        assert len(failed.inventory.live) == 0
        assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL


def test_decode_owner_death_at_every_prefix_conserves_all_resources() -> None:
    """Every nonterminal decode prefix has a complete fail-closed inventory."""

    for prefix_length in range(12):
        issuer = TerminalReceiptIssuer()
        initial = _initial_state(issuer)
        events = _decode_path(issuer, initial.decode.binding)
        current = initial.decode
        for event in events[:prefix_length]:
            current = reduce_decode_lifecycle(current, event).current
        failed = reduce_decode_lifecycle(
            current,
            DecodeLifecycleEvent(
                kind=DecodeLifecycleEventKind.OWNER_DIED,
                reason="synthetic owner loss",
            ),
        ).current
        _assert_conserved(failed.inventory)
        assert len(failed.inventory.live) == 0
        assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
