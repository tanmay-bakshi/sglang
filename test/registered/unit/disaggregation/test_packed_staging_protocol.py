import dataclasses

import msgspec
import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedCommit,
    PackedDecodeProtocol,
    PackedLayoutSpec,
    PackedLease,
    PackedPrepare,
    PackedProtocolError,
    PackedProtocolState,
    PackedTopology,
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

MAIN_KV = StagingComponentId(state_index=None, state_type=None)
SWA = StagingComponentId(state_index=0, state_type=StateType.SWA)
KEY = PackedChunkKey(room_id=17, chunk_id=3)


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
    protocol.register_chunk(KEY, spec, registry)
    return protocol, selected_allocator, spec


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


def commit(writer_id: StagingWriterId, digest: bytes, lease_id: int) -> PackedCommit:
    """Build one terminal-DMA COMMIT message.

    :param writer_id: Claimed writer.
    :param digest: READY layout digest.
    :param lease_id: READY allocation identity.
    :returns: COMMIT payload.
    """

    return PackedCommit(
        key=KEY,
        writer_id=writer_id,
        digest=digest,
        lease_id=lease_id,
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
    return ready_messages[0].digest, ready_messages[0].lease_id


def reach_scatter_ready(
    protocol: PackedDecodeProtocol,
    spec: PackedLayoutSpec,
) -> tuple[bytes, int]:
    """Submit complete PREPARE and COMMIT consensus.

    :param protocol: Registered decode protocol.
    :param spec: Canonical spec.
    :returns: READY digest and lease identity.
    """

    digest, lease_id = reach_ready(protocol, spec)
    assert not protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
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
    """READY is emitted once even after COMMIT consensus and begun scatter."""

    protocol, allocator, spec = protocol_fixture()
    first_prepare = prepare(WRITERS[0], spec)
    digest, lease_id = reach_ready(protocol, spec)

    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert protocol.handle_prepare(first_prepare, WRITERS[0]) == ()
    protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
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
    protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]


def test_commit_before_ready_fails_without_allocating() -> None:
    """A COMMIT cannot manufacture a lease or imply PREPARE consensus."""

    protocol, allocator, spec = protocol_fixture()
    digest = spec.build().digest

    with pytest.raises(PackedProtocolError, match="before READY"):
        protocol.handle_commit(commit(WRITERS[0], digest, 41), WRITERS[0])

    assert allocator.allocations == []
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_duplicate_commit_does_not_advance_consensus() -> None:
    """An identical terminal-DMA report counts for one writer exactly once."""

    protocol, _, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    first_commit = commit(WRITERS[0], digest, lease_id)

    assert not protocol.handle_commit(first_commit, WRITERS[0])
    assert not protocol.handle_commit(first_commit, WRITERS[0])
    assert protocol.snapshot(KEY).state is PackedProtocolState.READY
    assert protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert protocol.snapshot(KEY).state is PackedProtocolState.SCATTER_READY


def test_conflicting_duplicate_commit_quarantines_remaining_dma_owners() -> None:
    """A writer cannot replace its accepted terminal transfer identity."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )

    with pytest.raises(PackedProtocolError, match="digest"):
        protocol.handle_commit(
            commit(WRITERS[0], b"x" * 32, lease_id),
            WRITERS[0],
        )

    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.releases == []
    protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]


def test_commit_claim_cannot_spoof_authenticated_writer() -> None:
    """COMMIT coverage is keyed by the authenticated transport peer."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    with pytest.raises(PackedProtocolError, match="authenticated peer"):
        protocol.handle_commit(
            commit(WRITERS[0], digest, lease_id),
            WRITERS[1],
        )

    snapshot = protocol.snapshot(KEY)
    assert snapshot.state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, snapshot.failure_reason)]
    assert allocator.releases == []


def test_late_commits_release_failed_ready_lease() -> None:
    """A failed lease remains quarantined until every possible DMA completes."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)

    protocol.fail_chunk(KEY, "READY delivery failed")
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_QUARANTINED
    assert allocator.quarantines == [(lease_id, "READY delivery failed")]
    assert allocator.releases == []

    first_commit = commit(WRITERS[0], digest, lease_id)
    assert not protocol.handle_commit(first_commit, WRITERS[0])
    assert not protocol.handle_commit(first_commit, WRITERS[0])
    assert allocator.releases == []

    assert not protocol.handle_commit(
        commit(WRITERS[1], digest, lease_id),
        WRITERS[1],
    )
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_ready_delivery_failure_requires_commit_or_trusted_quiescence() -> None:
    """A writer whose READY reply failed still remains a possible DMA owner."""

    protocol, allocator, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    protocol.fail_chunk(KEY, "one READY reply failed")

    protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
        WRITERS[0],
    )
    assert allocator.releases == []

    protocol.quiesce_writer(KEY, WRITERS[1])
    assert allocator.releases == [lease_id]
    assert protocol.snapshot(KEY).state is PackedProtocolState.FAILED_RELEASED


def test_scatter_cannot_begin_before_all_unique_commits() -> None:
    """Scatter ownership remains unavailable during partial COMMIT consensus."""

    protocol, _, spec = protocol_fixture()
    digest, lease_id = reach_ready(protocol, spec)
    protocol.handle_commit(
        commit(WRITERS[0], digest, lease_id),
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
        protocol.register_chunk(KEY, spec, registry)

    assert allocator.allocations == []


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


def test_ready_and_commit_wire_round_trip() -> None:
    """READY projections and terminal COMMIT identities round-trip exactly."""

    protocol, _, spec = protocol_fixture()
    protocol.handle_prepare(prepare(WRITERS[0], spec), WRITERS[0])
    ready_messages = protocol.handle_prepare(
        prepare(WRITERS[1], spec),
        WRITERS[1],
    )
    commit_message = commit(
        WRITERS[0],
        ready_messages[0].digest,
        ready_messages[0].lease_id,
    )

    for message in (*ready_messages, commit_message):
        payload = encode_packed_message(message)
        assert decode_packed_message(payload) == message
        assert encode_packed_message(decode_packed_message(payload)) == payload


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
    with pytest.raises(PackedWireError, match="must not be empty"):
        decode_packed_message(b"")
    with pytest.raises(PackedWireError, match="exceeds"):
        decode_packed_message(b"x" * (MAX_PACKED_WIRE_BYTES + 1))
