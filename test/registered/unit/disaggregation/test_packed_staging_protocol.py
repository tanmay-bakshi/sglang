import dataclasses
import sys
import weakref

import msgspec
import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES,
    PACKED_REQUEST_DIGEST_BYTES,
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedLayoutSpec,
    PackedLease,
    PackedPrepare,
    PackedProtocolError,
    PackedProtocolState,
    PackedRequestKey,
    PackedTopology,
    PackedTransportPath,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityAction,
    PackedWriterVisibilityEvidence,
    _PackedWriterOutcomeTicketIssuer,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    MAX_PACKED_WIRE_BYTES,
    PackedWireError,
    decode_packed_message,
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBuffer,
    StagingComponentBufferRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAIN_KV = StagingComponentId(state_index=None, state_type=None)
SWA = StagingComponentId(state_index=0, state_type=StateType.SWA)
REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
KEY = PackedChunkKey(
    room_id=17,
    chunk_id=3,
    request_generation=REQUEST_GENERATION,
)


class RecordingAllocator:
    """CPU-only allocator recording every lease lifecycle transition."""

    allocations: list[int]
    quarantines: list[tuple[int, str]]
    releases: list[int]
    next_lease_id: int
    length_adjustment: int

    def __init__(self, *, length_adjustment: int = 0) -> None:
        """Initialize an empty allocator history.

        :param length_adjustment: Bytes added to the requested lease capacity.
        """

        self.allocations = []
        self.quarantines = []
        self.releases = []
        self.next_lease_id = 41
        self.length_adjustment = length_adjustment

    def allocate(self, length_bytes: int) -> PackedLease:
        """Return one fake registered contiguous allocation.

        :param length_bytes: Required packed bytes.
        :returns: Fake contiguous lease.
        """

        self.allocations.append(length_bytes)
        lease = PackedLease(
            lease_id=self.next_lease_id,
            base_address=0x800000,
            length_bytes=length_bytes + self.length_adjustment,
        )
        self.next_lease_id += 1
        return lease

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Record a quarantined allocation.

        :param lease: Failed lease.
        :param reason: First failure reason.
        """

        self.quarantines.append((lease.lease_id, reason))

    def release(self, lease: PackedLease) -> None:
        """Record a terminal lease release.

        :param lease: Lease safe for reuse.
        """

        self.releases.append(lease.lease_id)


class FaultInjectingAllocator(RecordingAllocator):
    """Transactional allocator that can fail before changing ownership."""

    quarantine_failures: int
    release_failures: int
    reenter_on_quarantine: bool
    protocol: PackedDecodeProtocol | None

    def __init__(self) -> None:
        """Initialize disabled fault injection."""

        super().__init__()
        self.quarantine_failures = 0
        self.release_failures = 0
        self.reenter_on_quarantine = False
        self.protocol = None

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Fail transactionally or record quarantine.

        :param lease: Failed lease.
        :param reason: First failure reason.
        """

        if self.reenter_on_quarantine:
            protocol = self.protocol
            if protocol is None:
                raise RuntimeError("fault allocator has no protocol")
            protocol.snapshot(KEY)
        if self.quarantine_failures > 0:
            self.quarantine_failures -= 1
            raise RuntimeError("injected quarantine failure")
        super().quarantine(lease, reason)

    def release(self, lease: PackedLease) -> None:
        """Fail transactionally or record release.

        :param lease: Lease safe for reuse.
        """

        if self.release_failures > 0:
            self.release_failures -= 1
            raise RuntimeError("injected release failure")
        super().release(lease)


def writer(rank: int) -> StagingWriterId:
    """Build one authenticated TP writer identity.

    :param rank: Source attention and transfer rank.
    :returns: Initial PP1/CP1 writer identity.
    """

    return StagingWriterId(
        transfer_source_rank=rank,
        source_attn_tp_rank=rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )


WRITERS = (writer(0), writer(1))
POLICY_DIGESTS = {
    WRITERS[0]: bytes.fromhex("10" * 32),
    WRITERS[1]: bytes.fromhex("20" * 32),
}
OUTCOME_TICKET_ISSUERS: weakref.WeakKeyDictionary[
    PackedDecodeProtocol,
    _PackedWriterOutcomeTicketIssuer,
] = weakref.WeakKeyDictionary()


def auxiliary_plan() -> PackedAuxiliaryPlan:
    """Build one exact plan with intentionally non-address-sorted segments.

    :returns: Deterministic decoder-authored auxiliary plan.
    """

    return PackedAuxiliaryPlan(
        key=PackedRequestKey.from_chunk_key(KEY),
        request_slot_generation=73,
        metadata_buffer_index=11,
        metadata_slot_generation=bytes.fromhex("ffeeddccbbaa99887766554433221100"),
        destination_segments=(
            PackedAuxiliaryDestinationSegment(
                address=0xA01000,
                item_length=64,
            ),
            PackedAuxiliaryDestinationSegment(
                address=0xA00000,
                item_length=128,
            ),
        ),
        canonical_writer_id=WRITERS[0],
        destination_process_generation=bytes.fromhex(
            "102132435465768798a9bacbdcedfe0f"
        ),
        native_route_digest=bytes(range(PACKED_REQUEST_DIGEST_BYTES)),
        runtime_cohort_digest=bytes(reversed(range(PACKED_REQUEST_DIGEST_BYTES))),
    )


def geometry(
    component_id: StagingComponentId,
    *,
    source: bool,
) -> StagingComponentGeometry:
    """Build a two-entry TP2-to-TP1 component geometry.

    :param component_id: Main-KV or SWA identity.
    :param source: Whether to build the TP2 source geometry.
    :returns: K-then-V component geometry.
    """

    item_len = 16 if source else 32
    layer_id = 5 if component_id == MAIN_KV else 1
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=(item_len, item_len),
        layer_ids=(layer_id, layer_id),
        page_size=4,
    )


def span(component_id: StagingComponentId) -> StagingComponentSpan:
    """Build one two-page active component span.

    :param component_id: Main-KV or SWA identity.
    :returns: Exact page-rounded component span.
    """

    return StagingComponentSpan(
        component_id=component_id,
        source_index_offset=0,
        destination_index_offset=0,
        logical_token_count=7,
        physical_token_count=8,
    )


def layout_spec(
    component_ids: tuple[StagingComponentId, ...] = (MAIN_KV,),
) -> PackedLayoutSpec:
    """Build one complete immutable packed layout input.

    :param component_ids: Active component shape.
    :returns: TP2-to-TP1 packed layout spec.
    """

    return PackedLayoutSpec(
        chunk_id=KEY.chunk_id,
        is_last=True,
        spans=tuple(span(component_id) for component_id in component_ids),
        source_components=tuple(
            geometry(component_id, source=True) for component_id in component_ids
        ),
        destination_components=tuple(
            geometry(component_id, source=False) for component_id in component_ids
        ),
        writers=WRITERS,
        topology=PackedTopology(
            source_tp_size=2,
            destination_tp_size=1,
            destination_tp_rank=0,
        ),
    )


def buffer(
    component_geometry: StagingComponentGeometry,
    *,
    page_array: tuple[int, ...] = (2, 4),
) -> StagingComponentBuffer:
    """Build one CPU-only decode component registration.

    :param component_geometry: Decode-side geometry.
    :param page_array: Request-local physical pages.
    :returns: Fake registered component buffers.
    """

    entry_count = len(component_geometry.item_lens)
    return StagingComponentBuffer(
        component_id=component_geometry.component_id,
        tensor_ptrs=tuple(
            0x100000 + entry_index * 0x1000 for entry_index in range(entry_count)
        ),
        data_lens=tuple(item_len * 16 for item_len in component_geometry.item_lens),
        item_lens=component_geometry.item_lens,
        layer_ids=component_geometry.layer_ids,
        page_size=component_geometry.page_size,
        page_array=np.asarray(page_array, dtype=np.int32),
    )


def protocol_fixture(
    component_ids: tuple[StagingComponentId, ...] = (MAIN_KV,),
    *,
    allocator: RecordingAllocator | None = None,
) -> tuple[
    PackedDecodeProtocol,
    RecordingAllocator,
    PackedLayoutSpec,
]:
    """Register one CPU-only protocol chunk.

    :param component_ids: Active component shape.
    :param allocator: Allocator override used for lifecycle fault injection.
    :returns: Protocol, recording allocator, and canonical spec.
    """

    spec = layout_spec(component_ids)
    registry = StagingComponentBufferRegistry(
        tuple(
            buffer(component_geometry)
            for component_geometry in spec.destination_components
        )
    )
    selected_allocator = RecordingAllocator() if allocator is None else allocator
    protocol = PackedDecodeProtocol(selected_allocator)
    OUTCOME_TICKET_ISSUERS[protocol] = protocol._claim_writer_outcome_ticket_issuer()
    protocol.register_chunk(KEY, spec, registry, POLICY_DIGESTS)
    return protocol, selected_allocator, spec


def admit_writer_outcome(
    protocol: PackedDecodeProtocol,
    message: PackedWriterOutcome,
    authenticated_writer_id: StagingWriterId,
) -> bool:
    """Admit protocol-unit-test DONE with its exact destination ticket.

    :param protocol: Protocol under test.
    :param message: Terminal writer outcome.
    :param authenticated_writer_id: Transport-authenticated writer.
    :returns: Whether this outcome newly made scatter eligible.
    """

    ticket = (
        OUTCOME_TICKET_ISSUERS[protocol]._issue(
            message,
            authenticated_writer_id,
        )
        if message.status is PackedWriterOutcomeStatus.DONE
        else None
    )
    return protocol.handle_writer_outcome(
        message,
        authenticated_writer_id,
        ticket,
    )


def prepare(
    writer_id: StagingWriterId,
    spec: PackedLayoutSpec,
    *,
    digest: bytes | None = None,
) -> PackedPrepare:
    """Build one PREPARE message.

    :param writer_id: Claimed writer.
    :param spec: Complete immutable layout spec.
    :param digest: Digest override.
    :returns: PREPARE payload.
    """

    canonical_digest = spec.build().digest if digest is None else digest
    return PackedPrepare(
        key=KEY,
        writer_id=writer_id,
        spec=spec,
        digest=canonical_digest,
    )


def outcome(
    writer_id: StagingWriterId,
    digest: bytes,
    lease_id: int,
    *,
    status: PackedWriterOutcomeStatus = PackedWriterOutcomeStatus.DONE,
    visibility: PackedWriterVisibilityEvidence | None = None,
    reason: str | None = None,
) -> PackedWriterOutcome:
    """Build one terminal-DMA writer outcome message.

    :param writer_id: Claimed writer.
    :param digest: READY layout digest.
    :param lease_id: READY allocation identity.
    :param status: Proven terminal status.
    :param visibility: Successful source visibility evidence override.
    :param reason: Failure reason for an error status.
    :returns: writer outcome payload.
    """

    selected_visibility = visibility
    if status is PackedWriterOutcomeStatus.DONE and selected_visibility is None:
        selected_visibility = visibility_evidence(writer_id)
    return PackedWriterOutcome(
        key=KEY,
        writer_id=writer_id,
        digest=digest,
        lease_id=lease_id,
        status=status,
        visibility=selected_visibility,
        reason=reason,
    )


def visibility_evidence(
    writer_id: StagingWriterId,
    *,
    policy_digest: bytes | None = None,
    transport_path: PackedTransportPath = PackedTransportPath.NIC_RDMA,
    lane_identifier: str | None = None,
    completion_mechanism: PackedWriterCompletionMechanism | None = None,
) -> PackedWriterVisibilityEvidence:
    """Build exact bounded writer evidence for one policy route.

    :param writer_id: Canonical writer owning the route.
    :param policy_digest: Route-policy digest override.
    :param transport_path: Selected CUDA IPC or NIC path.
    :param lane_identifier: Pinned lane override.
    :param completion_mechanism: Exact source completion primitive.
    :returns: Validated writer evidence.
    """

    selected_digest = (
        POLICY_DIGESTS[writer_id] if policy_digest is None else policy_digest
    )
    selected_lane = (
        f"mlx5_{writer_id.transfer_source_rank}/1:ucx-rc"
        if lane_identifier is None
        else lane_identifier
    )
    selected_mechanism = completion_mechanism
    if selected_mechanism is None:
        selected_mechanism = (
            PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            if transport_path is PackedTransportPath.CUDA_IPC
            else PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        )
    writer_action = (
        PackedWriterVisibilityAction.CUDA_EVENT_RECORDED
        if selected_mechanism
        is PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
        else PackedWriterVisibilityAction.TRANSPORT_HANDLE_TERMINAL
    )
    native_completion = (
        selected_mechanism
        is PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
    )
    return PackedWriterVisibilityEvidence(
        policy_digest=selected_digest,
        transport_path=transport_path,
        lane_identifier=selected_lane,
        completion_mechanism=selected_mechanism,
        writer_action=writer_action,
        native_handle_generation=1 if native_completion else None,
        native_descriptor_digest=(
            bytes.fromhex("11" * 32) if native_completion else None
        ),
        native_evidence_digest=bytes.fromhex("22" * 32) if native_completion else None,
    )


def reach_ready(
    protocol: PackedDecodeProtocol,
    spec: PackedLayoutSpec,
) -> tuple[bytes, int]:
    """Submit complete unique PREPARE consensus.

    :param protocol: Registered decode protocol.
    :param spec: Canonical spec.
    :returns: READY digest and lease identity.
    """

    assert protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0]) == ()
    ready_messages = protocol.handle_prepare(
        prepare(WRITERS[1], spec),
        WRITERS[1],
    )
    assert len(ready_messages) == 2
    assert tuple(
        message.visibility_policy_digest for message in ready_messages
    ) == tuple(POLICY_DIGESTS[writer_id] for writer_id in WRITERS)
    return ready_messages[0].digest, ready_messages[0].lease_id


def reach_scatter_ready(
    protocol: PackedDecodeProtocol,
    spec: PackedLayoutSpec,
) -> tuple[bytes, int]:
    """Submit complete PREPARE and writer outcome consensus.

    :param protocol: Registered decode protocol.
    :param spec: Canonical spec.
    :returns: READY digest and lease identity.
    """

    digest, lease_id = reach_ready(protocol, spec)
    assert not admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    return digest, lease_id


def test_partial_prepare_consensus_does_not_allocate() -> None:
    """Neither partial consensus nor an identical duplicate allocates."""

    protocol, allocator, spec = protocol_fixture()
    first_prepare = prepare(WRITERS[0], spec)

    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()

    assert allocator.allocations == []
    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.COLLECTING
    assert snapshot.prepared_writers == (WRITERS[0],)


def test_complete_prepare_consensus_allocates_once_and_projects_ready() -> None:
    """The final unique PREPARE produces one projection per canonical writer."""

    protocol, allocator, spec = protocol_fixture()
    first_prepare = prepare(WRITERS[0], spec)
    second_prepare = prepare(WRITERS[1], spec)

    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    ready_messages = protocol.handle_prepare(second_prepare, WRITERS[1])
    duplicate_result = protocol.handle_prepare(first_prepare, WRITERS[0])

    assert len(allocator.allocations) == 1
    assert tuple(message.writer_id for message in ready_messages) == WRITERS
    assert duplicate_result == ()
    assert len(allocator.allocations) == 1
    assert ready_messages[0].projection_offset == 0
    assert ready_messages[1].projection_offset >= ready_messages[0].projection_length
    assert all(message.projection_length > 0 for message in ready_messages)


def test_duplicate_prepare_never_reexposes_a_ready_lease() -> None:
    """READY is emitted once even after writer outcome consensus and begun scatter."""

    protocol, allocator, spec = protocol_fixture()
    first_prepare = prepare(WRITERS[0], spec)
    digest, lease_id = reach_ready(protocol, spec)

    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    protocol.begin_scatter(KEY)
    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    assert len(allocator.allocations) == 1


def test_prepare_claim_cannot_spoof_authenticated_writer() -> None:
    """Wire identity cannot contribute consensus for another transport peer."""

    protocol, allocator, spec = protocol_fixture()

    with pytest.raises(PackedProtocolError, match="authenticated peer"):
        protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[1])

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_conflicting_duplicate_prepare_fails_before_ready() -> None:
    """A writer cannot replace its accepted immutable digest."""

    protocol, allocator, spec = protocol_fixture()
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])

    with pytest.raises(PackedProtocolError, match="digest"):
        protocol.handle_prepare(
            prepare(WRITERS[0], spec, digest=b"x" * 32),
            WRITERS[0],
        )

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_conflicting_topology_fails_partial_consensus() -> None:
    """Every writer must declare the decode-local explicit topology."""

    protocol, allocator, spec = protocol_fixture()
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])
    conflicting_spec = dataclasses.replace(
        spec,
        topology=dataclasses.replace(spec.topology, alignment_bytes=512),
    )

    with pytest.raises(PackedProtocolError, match="topology"):
        protocol.handle_prepare(
            prepare(WRITERS[1], conflicting_spec),
            WRITERS[1],
        )

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_conflicting_prepare_after_ready_quarantines_until_dma_terminal() -> None:
    """A post-READY conflict cannot release a lease still exposed to writers."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    with pytest.raises(PackedProtocolError, match="digest"):
        protocol.handle_prepare(
            prepare(WRITERS[0], spec, digest=b"x" * 32),
            WRITERS[0],
        )

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.releases == []
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]


def test_writer_outcome_before_ready_fails_without_allocating() -> None:
    """A writer outcome cannot manufacture a lease or imply PREPARE consensus."""

    protocol, allocator, spec = protocol_fixture()
    digest = spec.build().digest

    with pytest.raises(PackedProtocolError, match="before READY"):
        admit_writer_outcome(protocol, outcome(WRITERS[0], digest, 41), WRITERS[0])

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_duplicate_writer_outcome_does_not_advance_consensus() -> None:
    """An identical terminal-DMA report counts for one writer exactly once."""

    protocol, _, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    first_outcome = outcome(WRITERS[0], digest, lease_id)

    assert not admit_writer_outcome(protocol, first_outcome, WRITERS[0])
    assert not admit_writer_outcome(protocol, first_outcome, WRITERS[0])
    assert protocol.snapshot(KEY).state is PackedProtocolState.READY
    assert admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert protocol.snapshot(KEY).state is PackedProtocolState.SCATTER_READY


def test_conflicting_duplicate_writer_outcome_quarantines_remaining_dma_owners() -> (
    None
):
    """A writer cannot replace its accepted terminal transfer identity."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )

    with pytest.raises(PackedProtocolError, match="digest"):
        admit_writer_outcome(
            protocol,
            outcome(WRITERS[0], b"x" * 32, lease_id),
            WRITERS[0],
        )

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.releases == []
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]


def test_writer_outcome_policy_digest_must_match_issued_ready() -> None:
    """A terminal writer cannot substitute a different route policy."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    forged_visibility = visibility_evidence(
        WRITERS[0],
        policy_digest=b"\xee" * 32,
    )

    with pytest.raises(PackedProtocolError, match="visibility policy"):
        admit_writer_outcome(
            protocol,
            outcome(
                WRITERS[0],
                digest,
                lease_id,
                visibility=forged_visibility,
            ),
            WRITERS[0],
        )

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, snapshot.failure_reason)]
    assert allocator.releases == []


def test_writer_outcome_claim_cannot_spoof_authenticated_writer() -> None:
    """writer outcome coverage is keyed by the authenticated transport peer."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    with pytest.raises(PackedProtocolError, match="authenticated peer"):
        protocol.handle_writer_outcome(
            outcome(WRITERS[0], digest, lease_id),
            WRITERS[1],
        )

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, snapshot.failure_reason)]
    assert allocator.releases == []


def test_writer_outcome_ticket_issuer_has_one_coordinator_owner() -> None:
    """A protocol cannot hand DONE-admission authority to two coordinators."""

    protocol = PackedDecodeProtocol(RecordingAllocator())

    issuer = protocol._claim_writer_outcome_ticket_issuer()

    assert "claim_writer_outcome_ticket_issuer" not in dir(protocol)
    assert "issue" not in dir(issuer)

    with pytest.raises(RuntimeError, match="already been claimed"):
        protocol._claim_writer_outcome_ticket_issuer()


def test_new_done_requires_a_protocol_bound_visibility_ticket() -> None:
    """Direct DONE ingress cannot bypass destination CUDA visibility."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    with pytest.raises(PackedProtocolError, match="visibility ticket"):
        protocol.handle_writer_outcome(
            outcome(WRITERS[0], digest, lease_id),
            WRITERS[0],
        )

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, snapshot.failure_reason)]


def test_done_ticket_is_exact_message_and_protocol_bound() -> None:
    """A visibility ticket cannot admit another message or protocol instance."""

    first, _, first_spec = protocol_fixture()
    second, second_allocator, second_spec = protocol_fixture()
    first_digest, first_lease_id = reach_ready(first, first_spec)
    second_digest, second_lease_id = reach_ready(second, second_spec)
    first_message = outcome(WRITERS[0], first_digest, first_lease_id)
    first_ticket = OUTCOME_TICKET_ISSUERS[first]._issue(
        first_message,
        WRITERS[0],
    )

    with pytest.raises(PackedProtocolError, match="another protocol"):
        second.handle_writer_outcome(
            outcome(WRITERS[0], second_digest, second_lease_id),
            WRITERS[0],
            first_ticket,
        )

    second_snapshot = second.snapshot(KEY)
    assert second_snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert second_allocator.quarantines == [
        (second_lease_id, second_snapshot.failure_reason)
    ]

    third, third_allocator, third_spec = protocol_fixture()
    third_digest, third_lease_id = reach_ready(third, third_spec)
    writer_zero_message = outcome(WRITERS[0], third_digest, third_lease_id)
    writer_zero_ticket = OUTCOME_TICKET_ISSUERS[third]._issue(
        writer_zero_message,
        WRITERS[0],
    )

    with pytest.raises(PackedProtocolError, match="another message"):
        third.handle_writer_outcome(
            outcome(WRITERS[1], third_digest, third_lease_id),
            WRITERS[1],
            writer_zero_ticket,
        )

    third_snapshot = third.snapshot(KEY)
    assert third_snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert third_allocator.quarantines == [
        (third_lease_id, third_snapshot.failure_reason)
    ]


def test_duplicate_done_is_idempotent_without_reusing_its_ticket() -> None:
    """An admitted duplicate neither needs nor consumes another ticket."""

    protocol, _, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    message = outcome(WRITERS[0], digest, lease_id)
    ticket = OUTCOME_TICKET_ISSUERS[protocol]._issue(message, WRITERS[0])

    assert not protocol.handle_writer_outcome(message, WRITERS[0], ticket)
    assert not protocol.handle_writer_outcome(message, WRITERS[0])
    assert protocol.snapshot(KEY).writer_outcomes == (message,)


def test_late_writer_outcomes_release_failed_ready_lease() -> None:
    """A failed lease remains quarantined until every possible DMA completes."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    protocol.fail_chunk(KEY, "READY delivery failed")
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, "READY delivery failed")]
    assert allocator.releases == []

    first_outcome = outcome(WRITERS[0], digest, lease_id)
    assert not admit_writer_outcome(protocol, first_outcome, WRITERS[0])
    assert not admit_writer_outcome(protocol, first_outcome, WRITERS[0])
    assert allocator.releases == []

    assert not admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_terminal_writer_error_quarantines_exact_lease_until_all_writers_terminal() -> (
    None
):
    """A typed ERR proves one writer terminal without covering another writer."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    error_outcome = outcome(
        WRITERS[0],
        digest,
        lease_id,
        status=PackedWriterOutcomeStatus.ERROR,
        reason="terminal NIXL transport error",
    )

    assert not admit_writer_outcome(protocol, error_outcome, WRITERS[0])

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert snapshot.writer_outcomes == (error_outcome,)
    assert allocator.quarantines == [
        (lease_id, f"writer {WRITERS[0]} failed: terminal NIXL transport error")
    ]
    assert allocator.releases == []

    assert not admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_writer_outcome_status_and_reason_are_one_exact_identity() -> None:
    """Conflicting duplicate terminal status cannot replace accepted proof."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    done = outcome(WRITERS[0], digest, lease_id)
    admit_writer_outcome(protocol, done, WRITERS[0])

    with pytest.raises(PackedProtocolError, match="conflicting duplicate"):
        admit_writer_outcome(
            protocol,
            outcome(
                WRITERS[0],
                digest,
                lease_id,
                status=PackedWriterOutcomeStatus.ERROR,
                reason="late conflicting error",
            ),
            WRITERS[0],
        )

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.releases == []
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]


def test_ready_delivery_failure_requires_outcome_or_trusted_quiescence() -> None:
    """A writer whose READY reply failed still remains a possible DMA owner."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    protocol.fail_chunk(KEY, "one READY reply failed")

    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert allocator.releases == []

    protocol.quiesce_writer(KEY, WRITERS[1])
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_scatter_cannot_begin_before_all_unique_writer_outcomes() -> None:
    """Scatter ownership remains unavailable during partial writer outcome consensus."""

    protocol, _, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )

    with pytest.raises(PackedProtocolError, match="scatter cannot begin"):
        protocol.begin_scatter(KEY)

    assert protocol.snapshot(KEY).state is PackedProtocolState.READY


def test_failed_begun_scatter_requires_explicit_scatter_quiescence() -> None:
    """Completed writer DMAs do not imply that asynchronous scatter is terminal."""

    protocol, allocator, spec = protocol_fixture()
    _, lease_id = reach_scatter_ready(protocol, spec)
    work = protocol.begin_scatter(KEY)
    assert work.lease.lease_id == lease_id

    protocol.fail_scatter(KEY, "scatter kernel failed")

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert snapshot.scatter_started
    assert not snapshot.scatter_terminal
    assert allocator.quarantines == [(lease_id, "scatter kernel failed")]
    assert allocator.releases == []

    protocol.quiesce_scatter(KEY)

    assert allocator.releases == [lease_id]
    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_RELEASED
    assert snapshot.scatter_terminal


def test_retirement_requires_terminal_async_ownership() -> None:
    """Terminal retirement drops specs and page snapshots, never live leases."""

    protocol, _, spec = protocol_fixture()
    reach_ready(protocol, spec)

    with pytest.raises(PackedProtocolError, match="cannot retire"):
        protocol.retire_chunk(KEY)

    protocol.fail_chunk(KEY, "room cleanup")
    with pytest.raises(PackedProtocolError, match="cannot retire"):
        protocol.retire_chunk(KEY)
    for writer_id in WRITERS:
        protocol.quiesce_writer(KEY, writer_id)

    protocol.retire_chunk(KEY)

    with pytest.raises(PackedProtocolError, match="not registered"):
        protocol.snapshot(KEY)


def test_quarantine_callback_failure_is_transactionally_retryable() -> None:
    """A failed quarantine callback leaves the lease owned and retryable."""

    allocator = FaultInjectingAllocator()
    protocol, _, spec = protocol_fixture(allocator=allocator)
    _, lease_id = reach_ready(protocol, spec)
    allocator.quarantine_failures = 1

    with pytest.raises(RuntimeError, match="injected quarantine failure"):
        protocol.fail_chunk(KEY, "transport failed")

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == []
    assert allocator.releases == []

    protocol.fail_chunk(KEY, "retry quarantine")
    assert allocator.quarantines == [(lease_id, "transport failed")]
    for writer_id in WRITERS:
        protocol.quiesce_writer(KEY, writer_id)
    assert allocator.releases == [lease_id]


def test_release_callback_failure_is_transactionally_retryable() -> None:
    """A failed release callback cannot mark or double-release the lease."""

    allocator = FaultInjectingAllocator()
    protocol, _, spec = protocol_fixture(allocator=allocator)
    _, lease_id = reach_scatter_ready(protocol, spec)
    protocol.begin_scatter(KEY)
    allocator.release_failures = 1

    with pytest.raises(RuntimeError, match="injected release failure"):
        protocol.complete_scatter(KEY)

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.SCATTERING
    assert snapshot.scatter_terminal
    assert allocator.releases == []

    protocol.complete_scatter(KEY)
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.RELEASED


def test_duplicate_terminal_outcome_retries_failed_quarantine_release() -> None:
    """An idempotent duplicate can finish the final transactional release."""

    allocator = FaultInjectingAllocator()
    protocol, _, spec = protocol_fixture(allocator=allocator)
    digest, lease_id = reach_ready(protocol, spec)
    protocol.fail_chunk(KEY, "transport failed")
    admit_writer_outcome(
        protocol,
        outcome(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    final_outcome = outcome(WRITERS[1], digest, lease_id)
    allocator.release_failures = 1

    with pytest.raises(RuntimeError, match="injected release failure"):
        admit_writer_outcome(protocol, final_outcome, WRITERS[1])

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.releases == []

    assert not protocol.handle_writer_outcome(final_outcome, WRITERS[1])
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_allocator_reentry_is_rejected_before_protocol_mutation() -> None:
    """Allocator callbacks cannot recursively observe or mutate chunk state."""

    allocator = FaultInjectingAllocator()
    protocol, _, spec = protocol_fixture(allocator=allocator)
    _, lease_id = reach_ready(protocol, spec)
    allocator.protocol = protocol
    allocator.reenter_on_quarantine = True

    with pytest.raises(RuntimeError, match="must not reenter"):
        protocol.fail_chunk(KEY, "transport failed")

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == []
    allocator.reenter_on_quarantine = False
    protocol.fail_chunk(KEY, "retry quarantine")
    for writer_id in WRITERS:
        protocol.quiesce_writer(KEY, writer_id)
    assert allocator.releases == [lease_id]


def test_undersized_lease_release_failure_retains_protocol_ownership() -> None:
    """Malformed allocation cleanup remains retryable if release first fails."""

    allocator = FaultInjectingAllocator()
    allocator.length_adjustment = -1
    allocator.release_failures = 1
    protocol, _, spec = protocol_fixture(allocator=allocator)
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])

    with pytest.raises(PackedProtocolError, match="release failed"):
        protocol.handle_prepare(prepare(WRITERS[1], spec), WRITERS[1])

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert not snapshot.ready_issued
    assert allocator.releases == []

    protocol.fail_chunk(KEY, "retry malformed lease cleanup")
    assert allocator.releases == [snapshot.lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


@pytest.mark.parametrize(
    "component_ids",
    [
        (MAIN_KV,),
        (MAIN_KV, SWA),
        (SWA,),
    ],
    ids=["main-only", "main-and-swa", "swa-only"],
)
def test_component_shapes_complete_through_scatter(
    component_ids: tuple[StagingComponentId, ...],
) -> None:
    """Intermediate and final component shapes share one exact lifecycle."""

    protocol, allocator, spec = protocol_fixture(component_ids)
    _, lease_id = reach_scatter_ready(protocol, spec)

    work = protocol.begin_scatter(KEY)

    assert (
        tuple(
            active.component.component_id
            for active in work.destination_binding.components
        )
        == component_ids
    )
    assert (
        tuple(active_span.component_id for active_span in work.layout.component_spans)
        == component_ids
    )
    protocol.complete_scatter(KEY)
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.RELEASED


def test_destination_page_bounds_rejected_before_prepare() -> None:
    """Decode cannot register a span exceeding its local page-array snapshot."""

    spec = layout_spec()
    registry = StagingComponentBufferRegistry(
        (buffer(spec.destination_components[0], page_array=(2,)),)
    )
    allocator = RecordingAllocator()
    protocol = PackedDecodeProtocol(allocator)

    with pytest.raises(ValueError, match="bounds overflow"):
        protocol.register_chunk(KEY, spec, registry, POLICY_DIGESTS)

    assert allocator.allocations == []


def test_registration_requires_one_policy_digest_per_canonical_writer() -> None:
    """READY cannot be issued without a pinned policy for every writer route."""

    spec = layout_spec()
    registry = StagingComponentBufferRegistry(
        tuple(
            buffer(component_geometry)
            for component_geometry in spec.destination_components
        )
    )
    protocol = PackedDecodeProtocol(RecordingAllocator())

    with pytest.raises(ValueError, match="exactly cover canonical writers"):
        protocol.register_chunk(
            KEY,
            spec,
            registry,
            {WRITERS[0]: POLICY_DIGESTS[WRITERS[0]]},
        )


def test_received_geometry_must_match_decode_registration() -> None:
    """A digest-valid but different endpoint geometry fails the room."""

    protocol, allocator, spec = protocol_fixture()
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])
    different_source = dataclasses.replace(
        spec.source_components[0],
        item_lens=(32, 32),
    )
    different_destination = dataclasses.replace(
        spec.destination_components[0],
        item_lens=(64, 64),
    )
    conflicting_spec = dataclasses.replace(
        spec,
        source_components=(different_source,),
        destination_components=(different_destination,),
    )

    with pytest.raises(PackedProtocolError, match="decode-local canonical"):
        protocol.handle_prepare(
            prepare(WRITERS[1], conflicting_spec),
            WRITERS[1],
        )

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_auxiliary_plan_and_outcome_wire_round_trip_are_exact() -> None:
    """Wire v6 preserves plan order, generations, and terminal evidence."""

    plan = auxiliary_plan()
    outcome = PackedAuxiliaryOutcome(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_dram_handle_generation=47,
        descriptor_digest=bytes.fromhex("31" * PACKED_REQUEST_DIGEST_BYTES),
        evidence_digest=bytes.fromhex("42" * PACKED_REQUEST_DIGEST_BYTES),
    )

    for message in (plan, outcome):
        payload = encode_packed_message(message)
        decoded = decode_packed_message(payload)
        assert decoded == message
        assert encode_packed_message(decoded) == payload
    decoded_plan = decode_packed_message(encode_packed_message(plan))
    assert isinstance(decoded_plan, PackedAuxiliaryPlan)
    assert decoded_plan.metadata_slot_generation == plan.metadata_slot_generation
    assert decoded_plan.destination_segments == plan.destination_segments


def test_auxiliary_plan_rejects_duplicate_overlap_and_noncanonical_writer() -> None:
    """Metadata destinations and their destination-local writer are unambiguous."""

    plan = auxiliary_plan()
    segment = plan.destination_segments[0]
    with pytest.raises(ValueError, match="duplicate"):
        dataclasses.replace(
            plan,
            destination_segments=(segment, segment),
        )
    with pytest.raises(ValueError, match="overlapping"):
        dataclasses.replace(
            plan,
            destination_segments=(
                segment,
                PackedAuxiliaryDestinationSegment(
                    address=segment.address + 1,
                    item_length=segment.item_length,
                ),
            ),
        )
    rank_one_plan = dataclasses.replace(plan, canonical_writer_id=WRITERS[1])
    assert rank_one_plan.canonical_writer_id == WRITERS[1]
    with pytest.raises(ValueError, match="PP0 and CP0"):
        dataclasses.replace(
            plan,
            canonical_writer_id=dataclasses.replace(
                WRITERS[1],
                source_pp_rank=1,
            ),
        )
    with pytest.raises(ValueError, match="canonical writer"):
        PackedAuxiliaryOutcome(
            plan=plan,
            writer_id=WRITERS[1],
            native_dram_handle_generation=47,
            descriptor_digest=bytes.fromhex("31" * PACKED_REQUEST_DIGEST_BYTES),
            evidence_digest=bytes.fromhex("42" * PACKED_REQUEST_DIGEST_BYTES),
        )


def test_auxiliary_wire_rejects_nested_schema_drift() -> None:
    """The closed auxiliary envelope rejects unknown nested plan fields."""

    envelope = msgspec.msgpack.decode(encode_packed_message(auxiliary_plan()))
    envelope["plan"]["unexpected"] = 1
    with pytest.raises(PackedWireError, match="invalid packed wire"):
        decode_packed_message(msgspec.msgpack.encode(envelope))


@pytest.mark.parametrize(
    "component_ids",
    [
        (MAIN_KV,),
        (MAIN_KV, SWA),
        (SWA,),
    ],
    ids=["main-only", "main-and-swa", "swa-only"],
)
def test_prepare_wire_round_trip_is_deterministic(
    component_ids: tuple[StagingComponentId, ...],
) -> None:
    """Every active component shape has one stable PREPARE encoding."""

    message = prepare(WRITERS[0], layout_spec(component_ids))

    payload = encode_packed_message(message)
    decoded = decode_packed_message(payload)

    assert decoded == message
    assert encode_packed_message(decoded) == payload


def test_tp1_prepare_wire_round_trip_preserves_single_writer_geometry() -> None:
    """Round-trip one exact TP1-to-TP1 packed-v4 PREPARE."""

    base_spec = layout_spec()
    spec = dataclasses.replace(
        base_spec,
        source_components=base_spec.destination_components,
        writers=(WRITERS[0],),
        topology=PackedTopology(
            source_tp_size=1,
            destination_tp_size=1,
            destination_tp_rank=0,
        ),
    )
    message = prepare(WRITERS[0], spec)

    payload = encode_packed_message(message)
    decoded = decode_packed_message(payload)

    assert decoded == message
    assert decoded.spec.topology.source_tp_size == 1
    assert decoded.spec.writers == (WRITERS[0],)
    assert decoded.spec.build() == spec.build()


@pytest.mark.parametrize(
    "visibility",
    [
        visibility_evidence(
            WRITERS[0],
            transport_path=PackedTransportPath.CUDA_IPC,
            lane_identifier="cuda-ipc:gpu3->gpu6",
        ),
        visibility_evidence(
            WRITERS[0],
            transport_path=PackedTransportPath.CUDA_IPC,
            lane_identifier="nixl-ucx-sha256:" + "3a" * 32,
            completion_mechanism=(
                PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
            ),
        ),
        visibility_evidence(
            WRITERS[0],
            lane_identifier="mlx5_0/1:ucx-ordered",
        ),
        visibility_evidence(
            WRITERS[0],
            lane_identifier="mlx5_1/1:ucx-unordered",
        ),
    ],
    ids=[
        "direct-cuda-ipc",
        "nixl-cuda-ipc",
        "ordered-nic",
        "host-flushed-nic",
    ],
)
def test_ready_and_writer_outcome_wire_round_trip(
    visibility: PackedWriterVisibilityEvidence,
) -> None:
    """READY and every successful visibility-evidence shape round-trip exactly."""

    protocol, _, spec = protocol_fixture()
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])
    ready_messages = protocol.handle_prepare(
        prepare(WRITERS[1], spec),
        WRITERS[1],
    )
    outcome_message = outcome(
        WRITERS[0],
        ready_messages[0].digest,
        ready_messages[0].lease_id,
        visibility=visibility,
    )
    error_message = outcome(
        WRITERS[1],
        ready_messages[1].digest,
        ready_messages[1].lease_id,
        status=PackedWriterOutcomeStatus.ERROR,
        reason="terminal transport error",
    )

    for message in (*ready_messages, outcome_message, error_message):
        payload = encode_packed_message(message)
        assert decode_packed_message(payload) == message
        assert encode_packed_message(decode_packed_message(payload)) == payload


def test_request_generation_and_outcome_reason_are_strict_domain_values() -> None:
    """Replay identity and terminal status cannot be represented ambiguously."""

    with pytest.raises(ValueError, match="request_generation"):
        PackedChunkKey(
            room_id=KEY.room_id,
            chunk_id=KEY.chunk_id,
            request_generation=b"short",
        )
    with pytest.raises(ValueError, match="must not contain a reason"):
        outcome(
            WRITERS[0],
            layout_spec().build().digest,
            41,
            reason="success cannot carry an error",
        )
    with pytest.raises(ValueError, match="non-empty reason"):
        outcome(
            WRITERS[0],
            layout_spec().build().digest,
            41,
            status=PackedWriterOutcomeStatus.ERROR,
        )
    with pytest.raises(TypeError, match="visibility evidence"):
        PackedWriterOutcome(
            key=KEY,
            writer_id=WRITERS[0],
            digest=layout_spec().build().digest,
            lease_id=41,
            status=PackedWriterOutcomeStatus.DONE,
            visibility=None,
        )
    with pytest.raises(ValueError, match="lane_identifier exceeds"):
        dataclasses.replace(
            visibility_evidence(WRITERS[0]),
            lane_identifier="x" * (MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES + 1),
        )
    with pytest.raises(TypeError, match="PackedWriterCompletionMechanism"):
        dataclasses.replace(
            visibility_evidence(WRITERS[0]),
            completion_mechanism="free-form completion text",
        )
    with pytest.raises(ValueError, match="does not match its completion mechanism"):
        dataclasses.replace(
            visibility_evidence(WRITERS[0]),
            writer_action=PackedWriterVisibilityAction.CUDA_EVENT_RECORDED,
        )
    native_evidence = visibility_evidence(WRITERS[0])
    with pytest.raises(ValueError, match="NIC RDMA visibility requires"):
        dataclasses.replace(
            native_evidence,
            completion_mechanism=(
                PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            ),
            writer_action=PackedWriterVisibilityAction.CUDA_EVENT_RECORDED,
            native_handle_generation=None,
            native_descriptor_digest=None,
            native_evidence_digest=None,
        )
    with pytest.raises(ValueError, match="handle generation"):
        dataclasses.replace(
            native_evidence,
            native_handle_generation=None,
        )
    with pytest.raises(ValueError, match="must not contain native NIXL"):
        dataclasses.replace(
            visibility_evidence(
                WRITERS[0],
                transport_path=PackedTransportPath.CUDA_IPC,
                lane_identifier="cuda-ipc:gpu3->gpu6",
            ),
            native_handle_generation=1,
            native_descriptor_digest=bytes.fromhex("11" * 32),
            native_evidence_digest=bytes.fromhex("22" * 32),
        )


def test_wire_rejects_unknown_kind_version_and_fields() -> None:
    """Envelope dispatch is versioned and forbids schema drift."""

    payload = encode_packed_message(prepare(WRITERS[0], layout_spec()))
    envelope = msgspec.msgpack.decode(payload)

    unknown_kind = dict(envelope)
    unknown_kind["kind"] = "surprise"
    with pytest.raises(PackedWireError, match="invalid packed wire"):
        decode_packed_message(msgspec.msgpack.encode(unknown_kind))

    unknown_version = dict(envelope)
    unknown_version["version"] = 99
    with pytest.raises(PackedWireError, match="unsupported packed wire version"):
        decode_packed_message(msgspec.msgpack.encode(unknown_version))

    unknown_field = dict(envelope)
    unknown_field["surprise"] = 1
    with pytest.raises(PackedWireError, match="invalid packed wire"):
        decode_packed_message(msgspec.msgpack.encode(unknown_field))


def test_wire_rejects_invalid_domain_values_and_frame_bounds() -> None:
    """Strict decoding rejects malformed identities and oversized frames."""

    payload = encode_packed_message(prepare(WRITERS[0], layout_spec()))
    envelope = msgspec.msgpack.decode(payload)
    envelope["spec"]["spans"][0]["component_id"]["state_type"] = "not-a-state"

    with pytest.raises(PackedWireError, match="invalid packed wire"):
        decode_packed_message(msgspec.msgpack.encode(envelope))

    unsupported_topology = msgspec.msgpack.decode(payload)
    unsupported_topology["spec"]["topology"]["source_tp_size"] = 3
    with pytest.raises(PackedWireError, match="invalid packed wire"):
        decode_packed_message(msgspec.msgpack.encode(unsupported_topology))
    with pytest.raises(PackedWireError, match="must not be empty"):
        decode_packed_message(b"")
    with pytest.raises(PackedWireError, match="exceeds"):
        decode_packed_message(b"x" * (MAX_PACKED_WIRE_BYTES + 1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
