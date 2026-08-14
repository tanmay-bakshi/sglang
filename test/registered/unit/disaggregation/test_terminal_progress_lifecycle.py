import itertools

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DECODE_RESOURCE_KINDS,
    SOURCE_RECLAIMABLE_RESOURCES,
    SOURCE_RESOURCE_KINDS,
    DecodeLifecycle,
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
    DecodeLifecyclePhase,
    SourceLifecycle,
    SourceLifecycleEvent,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
    TerminalLifecycleError,
    TerminalProcessDisposition,
    TerminalResourceInventory,
    TerminalResourceKind,
    create_decode_lifecycle,
    create_source_lifecycle,
    reduce_decode_lifecycle,
    reduce_source_lifecycle,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptError,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
PUBLICATION_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
RANK_MANIFEST_DIGEST = b"r" * 32
ALLOCATION_DIGEST = b"a" * 32

SOURCE_FORWARD_EVENTS = (
    SourceLifecycleEventKind.SUBMISSION_ACCEPTED,
    SourceLifecycleEventKind.PRODUCER_COMPLETED,
    SourceLifecycleEventKind.GATHER_COMPLETED_AND_NATIVE_POSTED,
    SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL,
    SourceLifecycleEventKind.OUTCOMES_SENT,
    SourceLifecycleEventKind.TEARDOWN_RECEIVED,
    SourceLifecycleEventKind.ACK_SENT,
)
DECODE_FORWARD_EVENTS = (
    DecodeLifecycleEventKind.ALLOCATION_PUBLISHED,
    DecodeLifecycleEventKind.WRITER_AGGREGATION_STARTED,
    DecodeLifecycleEventKind.WRITER_MANIFEST_COMPLETED,
    DecodeLifecycleEventKind.SCATTER_STARTED,
    DecodeLifecycleEventKind.SCATTER_TERMINAL,
    DecodeLifecycleEventKind.TEARDOWN_SENT,
    DecodeLifecycleEventKind.ACK_AGGREGATION_STARTED,
    DecodeLifecycleEventKind.ACK_MANIFEST_COMPLETED,
)


def make_binding(
    role: TerminalOwnerRole, *, room_id: int = 17
) -> TerminalRequestBinding:
    """Create one exact terminal-owner request binding.

    :param role: Source or decode owner role.
    :param room_id: Stable packed room identity.
    :returns: Valid immutable request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=REQUEST_GENERATION,
        ),
        owner=TerminalProcessIdentity(
            process_generation=PROCESS_GENERATION,
            role=role,
            tp_rank=0,
            tp_size=2,
        ),
        rank_manifest_digest=RANK_MANIFEST_DIGEST,
        allocation_digest=ALLOCATION_DIGEST,
    )


def make_source(
    issuer: TerminalReceiptIssuer,
    *,
    binding: TerminalRequestBinding | None = None,
) -> SourceLifecycle:
    """Create one initial source lifecycle trusting an exact issuer.

    :param issuer: Trusted request-local receipt issuer.
    :param binding: Optional source request binding.
    :returns: Initial source lifecycle.
    """

    source_binding = binding or make_binding(TerminalOwnerRole.SOURCE)
    return create_source_lifecycle(
        binding=source_binding,
        publication_identity=TerminalPublicationIdentity(
            request_key=source_binding.request_key,
            publisher_process_generation=PROCESS_GENERATION,
            publication_generation=PUBLICATION_GENERATION,
        ),
        receipt_ledger=TerminalReceiptLedger(
            authorities=frozenset((issuer.authority,))
        ),
    )


def make_decode(issuer: TerminalReceiptIssuer) -> DecodeLifecycle:
    """Create one initial decode lifecycle trusting an exact issuer.

    :param issuer: Trusted request-local receipt issuer.
    :returns: Initial decode lifecycle.
    """

    return create_decode_lifecycle(
        binding=make_binding(TerminalOwnerRole.DECODE),
        receipt_ledger=TerminalReceiptLedger(
            authorities=frozenset((issuer.authority,))
        ),
    )


def issue_receipt(
    issuer: TerminalReceiptIssuer,
    binding: TerminalRequestBinding,
    kind: TerminalReceiptKind,
    *,
    outcome: TerminalReceiptOutcome = TerminalReceiptOutcome.SUCCESS,
    timestamp_ns: int = 1,
) -> TerminalReceipt:
    """Issue one exact lifecycle receipt.

    :param issuer: Trusted receipt issuer.
    :param binding: Exact request binding.
    :param kind: Receipt authority kind.
    :param outcome: Terminal receipt outcome.
    :param timestamp_ns: Issuer-local terminal timestamp.
    :returns: Immutable receipt.
    """

    return issuer.issue(binding, kind, outcome, timestamp_ns)


def advance_source_to_ready(
    lifecycle: SourceLifecycle, issuer: TerminalReceiptIssuer
) -> SourceLifecycle:
    """Advance one source lifecycle through authenticated request readiness.

    :param lifecycle: Initial source lifecycle.
    :param issuer: Trusted receipt issuer.
    :returns: Request-ready source lifecycle.
    """

    current = lifecycle
    for kind in SOURCE_FORWARD_EVENTS:
        current = reduce_source_lifecycle(
            current, SourceLifecycleEvent(kind=kind)
        ).current
    receipt = issue_receipt(
        issuer,
        current.binding,
        TerminalReceiptKind.REQUEST_READY,
        timestamp_ns=10,
    )
    return reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
            receipt=receipt,
        ),
    ).current


def advance_decode_to_adoption(
    lifecycle: DecodeLifecycle,
) -> DecodeLifecycle:
    """Advance one decode lifecycle through adoption authority issuance.

    :param lifecycle: Initial decode lifecycle.
    :returns: Adoption-ready decode lifecycle.
    """

    current = lifecycle
    for kind in DECODE_FORWARD_EVENTS:
        current = reduce_decode_lifecycle(
            current, DecodeLifecycleEvent(kind=kind)
        ).current
    return current


def assert_conserved(inventory: TerminalResourceInventory) -> None:
    """Assert exact one-of-three conservation for every inventory resource.

    :param inventory: Request-local resource inventory.
    """

    assert inventory.live.isdisjoint(inventory.safely_retired)
    assert inventory.live.isdisjoint(inventory.quarantined)
    assert inventory.safely_retired.isdisjoint(inventory.quarantined)
    assert (
        inventory.live | inventory.safely_retired | inventory.quarantined
        == inventory.universe
    )


@pytest.mark.parametrize(
    "join_order", tuple(itertools.permutations(("reclaim", "publish")))
)
def test_source_retires_only_after_both_terminal_join_branches(
    join_order: tuple[str, str],
) -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_source(issuer)
    current = advance_source_to_ready(initial, issuer)
    reclaim_receipt = issue_receipt(
        issuer,
        current.binding,
        TerminalReceiptKind.RECLAIM_CONSUMED,
        timestamp_ns=11,
    )
    publish_receipt = issue_receipt(
        issuer,
        current.binding,
        TerminalReceiptKind.GATEWAY_PUBLISHED,
        timestamp_ns=12,
    )
    events = {
        "reclaim": SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=reclaim_receipt,
        ),
        "publish": SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.GATEWAY_PUBLISHED,
            receipt=publish_receipt,
        ),
    }

    first = reduce_source_lifecycle(current, events[join_order[0]])
    assert first.previous is current
    assert first.current.phase is SourceLifecyclePhase.REQUEST_READY_RECEIVED
    assert first.current.phase is not SourceLifecyclePhase.RETIRED
    final = reduce_source_lifecycle(first.current, events[join_order[1]]).current

    assert final.phase is SourceLifecyclePhase.RETIRED
    assert final.reclaim_consumed
    assert final.gateway_published
    assert final.inventory.safely_retired == SOURCE_RESOURCE_KINDS
    assert_conserved(final.inventory)
    assert initial.phase is SourceLifecyclePhase.FROZEN
    assert initial.inventory.live == SOURCE_RESOURCE_KINDS


def test_publication_failure_after_reclaim_quarantines_only_publication_identity() -> (
    None
):
    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
                timestamp_ns=11,
            ),
        ),
    ).current
    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLICATION_FAILED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.FAILURE,
                outcome=TerminalReceiptOutcome.FAILURE,
                timestamp_ns=12,
            ),
            reason="gateway socket failed",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.inventory.safely_retired == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.quarantined == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert len(failed.inventory.live) == 0
    assert not failed.gateway_published
    assert failed.process_disposition is TerminalProcessDisposition.HEALTHY
    assert_conserved(failed.inventory)


def test_publisher_death_is_process_fatal_and_preserves_proved_reclaim() -> None:
    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
            ),
        ),
    ).current
    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited",
        ),
    ).current

    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.inventory.safely_retired == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.quarantined == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )


def test_publisher_death_before_request_ready_quarantines_all_live_resources() -> None:
    issuer = TerminalReceiptIssuer()
    current = make_source(issuer)

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory.quarantined == SOURCE_RESOURCE_KINDS
    assert len(failed.inventory.live) == 0
    assert len(failed.inventory.safely_retired) == 0
    assert_conserved(failed.inventory)


def test_publisher_death_after_request_ready_preserves_live_source_storage() -> None:
    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory.live == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.quarantined == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert len(failed.inventory.safely_retired) == 0
    assert_conserved(failed.inventory)


def test_publisher_death_preserves_an_existing_publication_quarantine() -> None:
    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLICATION_FAILED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.FAILURE,
                outcome=TerminalReceiptOutcome.FAILURE,
                timestamp_ns=11,
            ),
            reason="gateway socket failed",
        ),
    ).current
    inventory_before_death = current.inventory

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory == inventory_before_death
    assert failed.inventory.live == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.quarantined == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert_conserved(failed.inventory)


def test_publisher_death_preserves_reclaimed_storage_after_publication_failure() -> (
    None
):
    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
                timestamp_ns=11,
            ),
        ),
    ).current
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLICATION_FAILED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.FAILURE,
                outcome=TerminalReceiptOutcome.FAILURE,
                timestamp_ns=12,
            ),
            reason="gateway socket failed",
        ),
    ).current
    inventory_before_death = current.inventory

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory == inventory_before_death
    assert failed.inventory.safely_retired == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.quarantined == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert len(failed.inventory.live) == 0
    assert_conserved(failed.inventory)


def test_publisher_death_preserves_completed_publication_and_allows_reclaim() -> None:
    """Publisher process death cannot revoke already-issued terminal proof."""

    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.GATEWAY_PUBLISHED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.GATEWAY_PUBLISHED,
                timestamp_ns=11,
            ),
        ),
    ).current
    inventory_before_death = current.inventory

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLISHER_DIED,
            reason="publisher thread exited after publication",
        ),
    ).current

    assert failed.phase is SourceLifecyclePhase.REQUEST_READY_RECEIVED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory == inventory_before_death
    assert failed.gateway_published
    assert failed.inventory.live == SOURCE_RECLAIMABLE_RESOURCES
    assert failed.inventory.safely_retired == frozenset(
        (TerminalResourceKind.PUBLICATION_IDENTITY,)
    )
    assert len(failed.inventory.quarantined) == 0
    assert_conserved(failed.inventory)

    retired = reduce_source_lifecycle(
        failed,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=issue_receipt(
                issuer,
                failed.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
            ),
        ),
    ).current
    assert retired.phase is SourceLifecyclePhase.RETIRED
    assert retired.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert retired.inventory.safely_retired == SOURCE_RESOURCE_KINDS


def test_owner_death_after_publication_quarantine_preserves_closed_inventory() -> None:
    """A fatal owner loss remains representable after every resource is closed."""

    issuer = TerminalReceiptIssuer()
    current = advance_source_to_ready(make_source(issuer), issuer)
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.RECLAIM_CONSUMED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
            ),
        ),
    ).current
    current = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.PUBLICATION_FAILED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.FAILURE,
                outcome=TerminalReceiptOutcome.FAILURE,
            ),
            reason="gateway publication failed",
        ),
    ).current
    assert len(current.inventory.live) == 0

    failed = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.OWNER_DIED,
            reason="owner exited during fail-closed drain",
        ),
    ).current
    assert failed.phase is SourceLifecyclePhase.PUBLICATION_QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory == current.inventory


@pytest.mark.parametrize(
    "fatal_kind",
    (
        SourceLifecycleEventKind.OWNER_DIED,
        SourceLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    ),
)
def test_source_lifecycle_fatal_events_quarantine_every_live_resource(
    fatal_kind: SourceLifecycleEventKind,
) -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_source(issuer)
    failed = reduce_source_lifecycle(
        initial,
        SourceLifecycleEvent(kind=fatal_kind, reason="fatal lifecycle loss"),
    ).current

    assert failed.phase is SourceLifecyclePhase.QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory.quarantined == SOURCE_RESOURCE_KINDS
    assert len(failed.inventory.live) == 0
    assert_conserved(failed.inventory)


def test_source_rejects_wrong_order_binding_and_semantic_duplicates() -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_source(issuer)
    with pytest.raises(TerminalLifecycleError, match="illegal"):
        reduce_source_lifecycle(
            initial,
            SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.NATIVE_TRANSFER_TERMINAL
            ),
        )
    assert initial.phase is SourceLifecyclePhase.FROZEN
    assert len(initial.receipt_ledger.consumed_tokens) == 0

    current = initial
    for kind in SOURCE_FORWARD_EVENTS:
        current = reduce_source_lifecycle(
            current, SourceLifecycleEvent(kind=kind)
        ).current
    wrong_binding = make_binding(TerminalOwnerRole.SOURCE, room_id=18)
    wrong_receipt = issue_receipt(
        issuer,
        wrong_binding,
        TerminalReceiptKind.REQUEST_READY,
    )
    with pytest.raises(TerminalReceiptError, match="another request"):
        reduce_source_lifecycle(
            current,
            SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt=wrong_receipt,
            ),
        )
    assert current.phase is SourceLifecyclePhase.ACK_SENT
    assert len(current.receipt_ledger.consumed_tokens) == 0

    first_receipt = issue_receipt(
        issuer,
        current.binding,
        TerminalReceiptKind.REQUEST_READY,
    )
    ready = reduce_source_lifecycle(
        current,
        SourceLifecycleEvent(
            kind=SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
            receipt=first_receipt,
        ),
    ).current
    conflicting_receipt = issue_receipt(
        issuer,
        ready.binding,
        TerminalReceiptKind.REQUEST_READY,
        timestamp_ns=999,
    )
    with pytest.raises(TerminalLifecycleError, match="illegal"):
        reduce_source_lifecycle(
            ready,
            SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.REQUEST_READY_RECEIVED,
                receipt=conflicting_receipt,
            ),
        )
    assert len(ready.receipt_ledger.consumed_tokens) == 1


def test_decode_valid_path_retires_all_resources() -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_decode(issuer)
    current = advance_decode_to_adoption(initial)
    current = reduce_decode_lifecycle(
        current,
        DecodeLifecycleEvent(
            kind=DecodeLifecycleEventKind.ADOPTION_CONSUMED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.ADOPTION_READY,
                timestamp_ns=20,
            ),
        ),
    ).current
    current = reduce_decode_lifecycle(
        current,
        DecodeLifecycleEvent(
            kind=DecodeLifecycleEventKind.METADATA_CONSUMED,
        ),
    ).current
    current = reduce_decode_lifecycle(
        current,
        DecodeLifecycleEvent(kind=DecodeLifecycleEventKind.LOCAL_DECODE_READY_ISSUED),
    ).current
    final = reduce_decode_lifecycle(
        current,
        DecodeLifecycleEvent(
            kind=DecodeLifecycleEventKind.REQUEST_READY_RECEIVED,
            receipt=issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.REQUEST_READY,
                timestamp_ns=22,
            ),
        ),
    ).current

    assert final.phase is DecodeLifecyclePhase.RETIRED
    assert final.inventory.safely_retired == DECODE_RESOURCE_KINDS
    assert len(final.receipt_ledger.consumed_tokens) == 2
    assert_conserved(final.inventory)
    assert initial.phase is DecodeLifecyclePhase.PREPARED


def test_decode_cancel_retires_but_postpublication_failure_quarantines() -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_decode(issuer)
    cancelled = reduce_decode_lifecycle(
        initial,
        DecodeLifecycleEvent(
            kind=DecodeLifecycleEventKind.CANCEL_UNPUBLISHED,
            reason="client cancelled before publication",
        ),
    ).current
    assert cancelled.phase is DecodeLifecyclePhase.RETIRED
    assert cancelled.inventory.safely_retired == DECODE_RESOURCE_KINDS

    published = reduce_decode_lifecycle(
        initial,
        DecodeLifecycleEvent(kind=DecodeLifecycleEventKind.ALLOCATION_PUBLISHED),
    ).current
    failed = reduce_decode_lifecycle(
        published,
        DecodeLifecycleEvent(
            kind=DecodeLifecycleEventKind.REQUEST_FAILED,
            receipt=issue_receipt(
                issuer,
                published.binding,
                TerminalReceiptKind.FAILURE,
                outcome=TerminalReceiptOutcome.FAILURE,
            ),
            reason="writer failed after publication",
        ),
    ).current
    assert failed.phase is DecodeLifecyclePhase.QUARANTINED
    assert failed.inventory.quarantined == DECODE_RESOURCE_KINDS
    assert failed.process_disposition is TerminalProcessDisposition.HEALTHY
    assert_conserved(failed.inventory)


@pytest.mark.parametrize(
    "fatal_kind",
    (
        DecodeLifecycleEventKind.OWNER_DIED,
        DecodeLifecycleEventKind.SCHEDULER_INBOX_OVERFLOW,
    ),
)
def test_decode_lifecycle_fatal_events_share_one_process_fatal_path(
    fatal_kind: DecodeLifecycleEventKind,
) -> None:
    issuer = TerminalReceiptIssuer()
    initial = make_decode(issuer)
    failed = reduce_decode_lifecycle(
        initial,
        DecodeLifecycleEvent(kind=fatal_kind, reason="fatal lifecycle loss"),
    ).current

    assert failed.phase is DecodeLifecyclePhase.QUARANTINED
    assert failed.process_disposition is TerminalProcessDisposition.PROCESS_FATAL
    assert failed.inventory.quarantined == DECODE_RESOURCE_KINDS
    assert_conserved(failed.inventory)


def test_resource_inventory_rejects_missing_overlap_and_double_transition() -> None:
    inventory = TerminalResourceInventory.all_live(SOURCE_RESOURCE_KINDS)
    publication = frozenset((TerminalResourceKind.PUBLICATION_IDENTITY,))
    retired = inventory.safely_retire(publication)

    with pytest.raises(TerminalLifecycleError, match="only live"):
        retired.quarantine(publication)
    with pytest.raises(ValueError, match="exactly one disposition"):
        TerminalResourceInventory(
            universe=SOURCE_RESOURCE_KINDS,
            live=SOURCE_RECLAIMABLE_RESOURCES,
            safely_retired=frozenset(),
            quarantined=frozenset(),
        )
    with pytest.raises(ValueError, match="overlap"):
        TerminalResourceInventory(
            universe=SOURCE_RESOURCE_KINDS,
            live=SOURCE_RESOURCE_KINDS,
            safely_retired=publication,
            quarantined=frozenset(),
        )


def test_dflash_boundary_rows_are_explicit_vram_lifetime_resources() -> None:
    """Neither owner can hide DFlash boundary rows in generic metadata."""

    assert TerminalResourceKind.SOURCE_DFLASH_BOUNDARY_VRAM_ROWS in (
        SOURCE_RECLAIMABLE_RESOURCES
    )
    assert TerminalResourceKind.DECODE_DFLASH_BOUNDARY_VRAM_ROWS in (
        DECODE_RESOURCE_KINDS
    )
    assert all(
        "metadata_row" not in resource.value for resource in TerminalResourceKind
    )


def test_generated_source_join_interleavings_preserve_conservation() -> None:
    for join_order in itertools.permutations(("reclaim", "publish")):
        issuer = TerminalReceiptIssuer()
        current = advance_source_to_ready(make_source(issuer), issuer)
        receipts = {
            "reclaim": issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.RECLAIM_CONSUMED,
                timestamp_ns=31,
            ),
            "publish": issue_receipt(
                issuer,
                current.binding,
                TerminalReceiptKind.GATEWAY_PUBLISHED,
                timestamp_ns=32,
            ),
        }
        kinds = {
            "reclaim": SourceLifecycleEventKind.RECLAIM_CONSUMED,
            "publish": SourceLifecycleEventKind.GATEWAY_PUBLISHED,
        }
        for name in join_order:
            current = reduce_source_lifecycle(
                current,
                SourceLifecycleEvent(kind=kinds[name], receipt=receipts[name]),
            ).current
            assert_conserved(current.inventory)
        assert current.phase is SourceLifecyclePhase.RETIRED


def test_decode_rejects_every_out_of_order_forward_event_without_mutation() -> None:
    issuer = TerminalReceiptIssuer()
    current = make_decode(issuer)
    snapshots: list[tuple[DecodeLifecycle, DecodeLifecycleEventKind]] = []
    for correct_kind in DECODE_FORWARD_EVENTS:
        snapshots.append((current, correct_kind))
        current = reduce_decode_lifecycle(
            current, DecodeLifecycleEvent(kind=correct_kind)
        ).current

    for snapshot, correct_kind in snapshots:
        for candidate_kind in DECODE_FORWARD_EVENTS:
            if candidate_kind is correct_kind:
                continue
            with pytest.raises(TerminalLifecycleError, match="illegal"):
                reduce_decode_lifecycle(
                    snapshot,
                    DecodeLifecycleEvent(kind=candidate_kind),
                )
            assert_conserved(snapshot.inventory)
            assert len(snapshot.receipt_ledger.consumed_tokens) == 0
