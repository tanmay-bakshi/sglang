import contextlib
import dataclasses
import inspect
from unittest.mock import Mock

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryPlan,
    PackedDFlashBoundaryMetadata,
    PackedDFlashBoundaryOutcome,
    PackedRequestKey,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    decode_packed_message,
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.terminal_progress import dflash_auxiliary
from sglang.srt.disaggregation.terminal_progress.dflash_auxiliary import (
    DFLASH_BOUNDARY_MEMORY_KIND,
    DFLASH_BOUNDARY_ROW_BYTES,
    DFlashBoundaryDeviceRowPool,
    DFlashBoundaryNixlRegistration,
    DFlashBoundaryRegisteredRow,
    DFlashBoundaryRegistration,
    DFlashBoundaryRemoteRow,
    DFlashBoundaryRowAllocator,
    DFlashBoundaryRowLease,
    DFlashBoundaryRowLeaseState,
    DFlashBoundarySourceTransfer,
    DFlashBoundarySourceTransportOwner,
    DFlashBoundaryTransportAccounting,
    build_dflash_boundary_nixl_descriptors,
)
from sglang.srt.disaggregation.terminal_progress.output_projection import (
    TerminalGatewayResultSlot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingAgent:
    """CPU-only recorder for exact NIXL registration and transfer calls."""

    name: str
    registration_calls: list[tuple[list[tuple[int, int, int, str]], str]]
    descriptor_calls: list[tuple[list[tuple[int, int, int]], str]]
    initialize_calls: list[tuple[str, object, object, object, bytes]]
    registration_handle: object
    transfer_handle: object

    def __init__(self) -> None:
        """Initialize one deterministic native boundary."""

        self.name = "source-agent"
        self.registration_calls = []
        self.descriptor_calls = []
        self.initialize_calls = []
        self.registration_handle = object()
        self.transfer_handle = object()

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Record one process-lifetime registration.

        :param addresses: Exact registration descriptors.
        :param memory_kind: Requested NIXL memory kind.
        :returns: Configured opaque registration handle.
        """

        self.registration_calls.append((addresses, memory_kind))
        return self.registration_handle

    def get_xfer_descs(
        self,
        addresses: list[tuple[int, int, int]],
        memory_kind: str,
    ) -> object:
        """Record one registered-row descriptor projection.

        :param addresses: Exact transfer requests.
        :param memory_kind: Requested NIXL memory kind.
        :returns: Deterministic opaque descriptor collection.
        """

        self.descriptor_calls.append((addresses, memory_kind))
        return (len(self.descriptor_calls), tuple(addresses), memory_kind)

    def initialize_xfer(
        self,
        operation: str,
        source_descriptors: object,
        destination_descriptors: object,
        remote_handle: object,
        notification: bytes,
    ) -> object:
        """Record one exact device-to-device transfer initialization.

        :param operation: Requested NIXL operation.
        :param source_descriptors: Exact local descriptors.
        :param destination_descriptors: Exact remote descriptors.
        :param remote_handle: Exact remote agent handle.
        :param notification: Exact notification payload.
        :returns: Deterministic native transfer handle.
        """

        self.initialize_calls.append(
            (
                operation,
                source_descriptors,
                destination_descriptors,
                remote_handle,
                notification,
            )
        )
        return self.transfer_handle


class _FakeStream:
    """Indexed CUDA stream identity without a CUDA context."""

    device_index: int

    def __init__(self, device_index: int) -> None:
        """Initialize one fake stream.

        :param device_index: Owning fake CUDA device.
        """

        self.device_index = device_index


class _FakeEvent:
    """CUDA event double which appends its record to one timeline."""

    timeline: list[str]

    def __init__(self, timeline: list[str] | None = None, **_kwargs: object) -> None:
        """Initialize one fake event.

        :param timeline: Optional shared ordering log.
        :param _kwargs: Ignored CUDA event controls.
        """

        self.timeline = [] if timeline is None else timeline

    def record(self, stream: _FakeStream) -> None:
        """Record exact stream ordering.

        :param stream: Exact producing stream.
        """

        self.timeline.append(f"event:{stream.device_index}")


class _FakeDestinationRow:
    """Device-row double which records D2D copy ordering."""

    timeline: list[str]

    def __init__(self, timeline: list[str]) -> None:
        """Bind one shared ordering timeline.

        :param timeline: Exact test ordering log.
        """

        self.timeline = timeline

    def copy_(self, source: object, *, non_blocking: bool) -> None:
        """Record one nonblocking row copy.

        :param source: Exact source projection.
        :param non_blocking: Required asynchronous-copy control.
        """

        if not non_blocking:
            raise AssertionError("boundary row copy was blocking")
        self.timeline.append(f"vram:{source}")

    def clone(self) -> "_FakeDestinationRow":
        """Return a request-owned device-row identity.

        :returns: Independent fake device row.
        """

        self.timeline.append("clone")
        return _FakeDestinationRow(self.timeline)

    def cpu(self) -> None:
        raise AssertionError("decoder adoption attempted a host copy")

    def item(self) -> None:
        raise AssertionError("decoder adoption attempted a host scalar read")


class _FakeRows:
    """Indexable stable device-row allocation double."""

    row: _FakeDestinationRow

    def __init__(self, row: _FakeDestinationRow) -> None:
        """Retain one fake row.

        :param row: Exact row returned for every valid index.
        """

        self.row = row

    def __getitem__(self, _row_index: int) -> _FakeDestinationRow:
        return self.row


class _FakeBoundaryToken:
    """One source-token double with an observable reshape projection."""

    def reshape(self, length: int) -> str:
        """Return one deterministic projected token identity.

        :param length: Required one-element row length.
        :returns: Stable projected token label.
        """

        if length != 1:
            raise AssertionError("boundary token was projected to the wrong shape")
        return "boundary-token[1]"

    def __str__(self) -> str:
        """Return the stable gateway-facing token label.

        :returns: Deterministic token label.
        """

        return "boundary-token"


@dataclasses.dataclass(frozen=True, slots=True)
class _AdoptedValue:
    """CPU-only result double for one device-resident adoption."""

    boundary_token_id: object
    completion_event: object


class _GatewaySlot(TerminalGatewayResultSlot):
    """Gateway-facing D2H result slot recorder."""

    timeline: list[str]

    def __init__(self, timeline: list[str]) -> None:
        """Bind one shared ordering timeline.

        :param timeline: Exact test ordering log.
        """

        self.timeline = timeline

    @property
    def generation(self) -> bytes:
        """Return one stable slot generation.

        :returns: Fixed-width test generation.
        """

        return b"g" * PACKED_REQUEST_GENERATION_BYTES

    def enqueue_copy(self, boundary_token_id: object) -> None:
        """Record the sole allowed device-to-host projection.

        :param boundary_token_id: Exact token tensor identity.
        """

        self.timeline.append(f"gateway:{boundary_token_id}")

    def read_next_token_id(self) -> int:
        """Return one deterministic completed token.

        :returns: Test boundary token.
        """

        return 17


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeEnum:
    """Native-shaped named enum fixture."""

    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeSegment:
    """Native-shaped exact transfer segment fixture."""

    index: int
    length: int
    localAddress: int
    posted: bool
    remoteAddress: int


@dataclasses.dataclass(frozen=True, slots=True)
class _NativeReceipt:
    """Native-shaped successful completion receipt fixture."""

    handleIdentity: int
    generation: int
    segments: tuple[_NativeSegment, ...]
    state: _NativeEnum = dataclasses.field(
        default_factory=lambda: _NativeEnum("NIXL_XFER_ATTESTATION_REMOTE_FLUSHED")
    )
    status: _NativeEnum = dataclasses.field(
        default_factory=lambda: _NativeEnum("NIXL_SUCCESS")
    )
    submissionSealed: bool = True
    completionClaimed: bool = True
    backend: str = "UCX"
    localAgent: str = "source-agent"
    remoteAgent: str = "decoder-agent"
    operation: _NativeEnum = dataclasses.field(
        default_factory=lambda: _NativeEnum("NIXL_WRITE")
    )
    localMemoryType: _NativeEnum = dataclasses.field(
        default_factory=lambda: _NativeEnum("VRAM_SEG")
    )
    remoteMemoryType: _NativeEnum = dataclasses.field(
        default_factory=lambda: _NativeEnum("VRAM_SEG")
    )
    descriptorDigest: str = "11" * 32
    evidenceDigest: str = "22" * 32
    error: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class _DirectTransfer:
    """Direct-owner exact transfer generation fixture."""

    binding_digest: bytes
    generation: int = 7
    handle_identity: int = 41


class _DirectOwner:
    """Deterministic direct native-owner adapter fixture."""

    receipt: _NativeReceipt | None
    calls: list[str]
    native_handle: object | None

    def __init__(self) -> None:
        """Initialize one clean direct owner."""

        self.receipt = None
        self.calls = []
        self.native_handle = None

    def arm_transfer(
        self,
        handle: object,
        binding_digest: bytes,
    ) -> _DirectTransfer:
        """Retain one exact native handle before posting.

        :param handle: Exact initialized handle.
        :param binding_digest: Exact source lifecycle binding.
        :returns: Deterministic transfer generation.
        """

        self.calls.append("arm")
        self.native_handle = handle
        return _DirectTransfer(binding_digest)

    def post_transfer(
        self,
        transfer: _DirectTransfer,
        post: object,
    ) -> object:
        """Invoke one exact post operation after arming.

        :param transfer: Exact armed transfer.
        :param post: One-shot native posting callable.
        :returns: Native post result.
        """

        if transfer.binding_digest is None or not callable(post):
            raise AssertionError("invalid direct post")
        self.calls.append("post")
        return post(self.native_handle)

    def settle_success(
        self,
        _transfer: _DirectTransfer,
        _action: object,
    ) -> _NativeReceipt:
        """Return one configured take-once success receipt.

        :param _transfer: Exact terminal transfer.
        :param _action: Exact source owner action.
        :returns: Configured native receipt.
        """

        self.calls.append("settle")
        if self.receipt is None:
            raise AssertionError("native receipt was not configured")
        return self.receipt

    def settle_failure(self, _transfer: _DirectTransfer, _action: object) -> None:
        """Record terminal failure settlement.

        :param _transfer: Exact failed transfer.
        :param _action: Matching owner action.
        """

        self.calls.append("failure")

    def release_transfer(self, _transfer: _DirectTransfer) -> None:
        """Record exact handle release.

        :param _transfer: Exact settled transfer.
        """

        self.calls.append("release")


def _registration(
    *,
    address_offset: int = 0,
    device_index: int = 3,
    row_capacity: int = 8,
) -> DFlashBoundaryRegistration:
    """Build deterministic CPU-only boundary geometry.

    :param address_offset: Offset separating independent process addresses.
    :param device_index: Exact CUDA device identity.
    :param row_capacity: Stable row count.
    :returns: Valid boundary registration receipt.
    """

    return DFlashBoundaryRegistration(
        base_address=0x1000 + address_offset,
        allocation_length=row_capacity * DFLASH_BOUNDARY_ROW_BYTES,
        device_index=device_index,
        row_capacity=row_capacity,
    )


def _allocator(
    *,
    address_offset: int = 0,
    device_index: int = 3,
    row_capacity: int = 8,
) -> tuple[_RecordingAgent, DFlashBoundaryRowAllocator]:
    """Build one independently registered CPU-only row allocator.

    :param address_offset: Offset separating process addresses.
    :param device_index: Exact CUDA device identity.
    :param row_capacity: Stable row count.
    :returns: Recording agent and generation-bound allocator.
    """

    agent = _RecordingAgent()
    registration = DFlashBoundaryNixlRegistration(
        agent,
        _registration(
            address_offset=address_offset,
            device_index=device_index,
            row_capacity=row_capacity,
        ),
    )
    return agent, DFlashBoundaryRowAllocator(registration)


def _pool_without_cuda(
    allocator: DFlashBoundaryRowAllocator,
    row: _FakeDestinationRow | None = None,
) -> DFlashBoundaryDeviceRowPool:
    """Build a device-pool shell over deterministic CPU-only authority.

    :param allocator: Exact registered row allocator.
    :param row: Optional fake device row.
    :returns: Device pool shell which never allocates a CUDA context.
    """

    pool = object.__new__(DFlashBoundaryDeviceRowPool)
    pool._registration = allocator._registration
    pool._row_allocator = allocator
    pool._boundary_token_ids = _FakeRows(
        _FakeDestinationRow([]) if row is None else row
    )
    return pool


def _writer() -> StagingWriterId:
    """Return the canonical source writer.

    :returns: TP0/PP0/CP0 writer identity.
    """

    return StagingWriterId(0, 0, 0, 0)


def _plan(
    *,
    destination_address: int = 0x9000,
    writer: StagingWriterId | None = None,
) -> PackedAuxiliaryPlan:
    """Build one exact decoder-authored boundary-row plan.

    :param destination_address: Exact remote row address.
    :param writer: Optional canonical writer identity.
    :returns: Valid immutable boundary plan.
    """

    return PackedAuxiliaryPlan(
        key=PackedRequestKey(room_id=41, request_generation=b"r" * 16),
        request_slot_generation=7,
        metadata_buffer_index=3,
        metadata_slot_generation=b"m" * 16,
        destination_segments=(
            PackedAuxiliaryDestinationSegment(
                address=destination_address,
                item_length=DFLASH_BOUNDARY_ROW_BYTES,
            ),
        ),
        canonical_writer_id=_writer() if writer is None else writer,
        destination_process_generation=b"d" * 16,
        native_route_digest=b"n" * 32,
        runtime_cohort_digest=b"c" * 32,
    )


def _metadata(boundary_token_id: int = 17) -> PackedDFlashBoundaryMetadata:
    """Build deterministic frozen boundary control metadata.

    :param boundary_token_id: Exact sampled target token.
    :returns: Complete scalar control metadata.
    """

    return PackedDFlashBoundaryMetadata(
        boundary_token_id=boundary_token_id,
        cached_tokens=8000,
        cached_tokens_device=7000,
        cached_tokens_host=900,
        cached_tokens_storage=100,
        image_tokens=11,
        audio_tokens=12,
        video_tokens=13,
    )


def _registered_row(
    allocator: DFlashBoundaryRowAllocator,
    owner: object,
) -> tuple[object, DFlashBoundaryRegisteredRow]:
    """Lease and prove one exact live boundary row.

    :param allocator: Owning generation-bound row allocator.
    :param owner: Exact reservation owner.
    :returns: Opaque reservation and registered-row proof.
    """

    reservation = allocator.allocate_packed_auxiliary_slot(owner)
    snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
    return reservation, allocator.registered_row(snapshot)


def _posted_transport() -> tuple[
    DFlashBoundaryDeviceRowPool,
    DFlashBoundarySourceTransportOwner,
    _DirectOwner,
    DFlashBoundaryRowLease,
    DFlashBoundarySourceTransfer,
]:
    """Build one posted all-VRAM source transfer with a valid receipt.

    :returns: Pool, owner, direct owner, source lease, and transfer identity.
    """

    agent, allocator = _allocator(row_capacity=1)
    pool = _pool_without_cuda(allocator)
    direct_owner = _DirectOwner()
    owner = DFlashBoundarySourceTransportOwner(
        pool=pool,
        agent=agent,
        direct_owner=direct_owner,
        writer_id=_writer(),
        post=lambda handle: handle,
    )
    source_lease = owner.lease_source_row()
    plan = _plan()
    transfer = owner.post(
        plan=plan,
        source_lease=source_lease,
        destination_device_index=6,
        remote_handle=object(),
        remote_agent_name="decoder-agent",
        binding_digest=b"b" * 32,
    )
    source_segment = source_lease.snapshot.destination_segments[0]
    destination_segment = plan.destination_segments[0]
    direct_owner.receipt = _NativeReceipt(
        handleIdentity=41,
        generation=7,
        segments=(
            _NativeSegment(
                index=0,
                length=DFLASH_BOUNDARY_ROW_BYTES,
                localAddress=source_segment.address,
                posted=True,
                remoteAddress=destination_segment.address,
            ),
        ),
    )
    return pool, owner, direct_owner, source_lease, transfer


def test_registration_is_one_exact_eight_byte_vram_allocation() -> None:
    """The process-lifetime pool exposes only one boundary-token allocation."""

    receipt = _registration(row_capacity=4)
    agent = _RecordingAgent()

    registration = DFlashBoundaryNixlRegistration(agent, receipt)

    assert registration.receipt is receipt
    assert agent.registration_calls == [
        ([(0x1000, 4 * DFLASH_BOUNDARY_ROW_BYTES, 3, "")], "VRAM")
    ]
    assert receipt.segment_for_row(3) == PackedAuxiliaryDestinationSegment(
        address=0x1000 + 3 * DFLASH_BOUNDARY_ROW_BYTES,
        item_length=DFLASH_BOUNDARY_ROW_BYTES,
    )
    assert receipt.transfer_request_for_row(3) == (
        0x1000 + 3 * DFLASH_BOUNDARY_ROW_BYTES,
        DFLASH_BOUNDARY_ROW_BYTES,
        3,
    )


@pytest.mark.parametrize("handle", [None, []], ids=("none", "empty"))
def test_registration_requires_owned_native_authority(handle: object) -> None:
    """An absent native registration cannot back reusable device rows."""

    agent = _RecordingAgent()
    agent.registration_handle = handle

    with pytest.raises(RuntimeError, match="returned no DFlash boundary"):
        DFlashBoundaryNixlRegistration(agent, _registration())


def test_row_allocator_exhaustion_release_and_quarantine_conserve() -> None:
    """Every physical row remains free, active, or quarantined exactly once."""

    _, allocator = _allocator(row_capacity=2)
    owner = object()
    first = allocator.allocate_packed_auxiliary_slot(owner)
    second = allocator.allocate_packed_auxiliary_slot(owner)

    assert allocator.inventory() == (0, 2, 0)
    with pytest.raises(RuntimeError, match="pool is exhausted"):
        allocator.allocate_packed_auxiliary_slot(owner)

    allocator.release_packed_auxiliary_slot(first, owner)
    allocator.quarantine_packed_auxiliary_slot(second, owner)

    assert allocator.inventory() == (1, 0, 1)
    reused = allocator.allocate_packed_auxiliary_slot(owner)
    assert allocator.inventory() == (0, 1, 1)
    allocator.release_packed_auxiliary_slot(reused, owner)
    assert allocator.inventory() == (1, 0, 1)


def test_row_allocator_changes_generation_before_physical_reuse() -> None:
    """Repeated reuse of one physical row cannot recreate stale authority."""

    _, allocator = _allocator(row_capacity=1)
    owner = object()
    generations: set[bytes] = set()

    for _ in range(256):
        reservation = allocator.allocate_packed_auxiliary_slot(owner)
        snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
        assert snapshot.metadata_buffer_index == 0
        assert len(snapshot.metadata_slot_generation) == (
            PACKED_REQUEST_GENERATION_BYTES
        )
        assert snapshot.metadata_slot_generation not in generations
        generations.add(snapshot.metadata_slot_generation)
        allocator.release_packed_auxiliary_slot(reservation, owner)


def test_row_allocator_rejects_stale_owner_generation_and_geometry() -> None:
    """Every reusable-row identity dimension is independently authoritative."""

    _, allocator = _allocator(row_capacity=1)
    owner = object()
    reservation = allocator.allocate_packed_auxiliary_slot(owner)
    snapshot = allocator.packed_auxiliary_slot_reservation_snapshot(reservation)

    with pytest.raises(RuntimeError, match="owner is stale"):
        allocator.release_packed_auxiliary_slot(reservation, object())
    with pytest.raises(RuntimeError, match="generation is stale"):
        allocator.require_snapshot(
            dataclasses.replace(
                snapshot,
                metadata_slot_generation=b"x" * PACKED_REQUEST_GENERATION_BYTES,
            )
        )
    with pytest.raises(RuntimeError, match="geometry was altered"):
        allocator.require_snapshot(
            dataclasses.replace(
                snapshot,
                destination_segments=(
                    PackedAuxiliaryDestinationSegment(
                        address=snapshot.destination_segments[0].address + 8,
                        item_length=DFLASH_BOUNDARY_ROW_BYTES,
                    ),
                ),
            )
        )

    allocator.release_packed_auxiliary_slot(reservation, owner)

    with pytest.raises(RuntimeError, match="reservation is stale"):
        allocator.packed_auxiliary_slot_reservation_snapshot(reservation)
    with pytest.raises(RuntimeError, match="snapshot is stale"):
        allocator.require_snapshot(snapshot)


def test_remote_row_requires_one_exact_eight_byte_segment() -> None:
    """Multi-field EAGLE geometry cannot enter the DFlash boundary channel."""

    plan = _plan()
    remote = DFlashBoundaryRemoteRow.from_plan(plan, device_index=6)

    assert remote.transfer_request == (
        plan.destination_segments[0].address,
        DFLASH_BOUNDARY_ROW_BYTES,
        6,
    )

    eagle_segments = (
        PackedAuxiliaryDestinationSegment(address=0x9000, item_length=64),
        PackedAuxiliaryDestinationSegment(address=0xA000, item_length=128),
        PackedAuxiliaryDestinationSegment(address=0xB000, item_length=512),
    )
    with pytest.raises(ValueError, match="exactly one segment"):
        DFlashBoundaryRemoteRow.from_plan(
            dataclasses.replace(plan, destination_segments=eagle_segments),
            device_index=6,
        )
    with pytest.raises(ValueError, match="not eight bytes"):
        DFlashBoundaryRemoteRow.from_plan(
            dataclasses.replace(
                plan,
                destination_segments=(
                    PackedAuxiliaryDestinationSegment(
                        address=0x9000,
                        item_length=64,
                    ),
                ),
            ),
            device_index=6,
        )


def test_boundary_outcome_rejects_multi_segment_and_eagle_payloads() -> None:
    """The authenticated wire type itself enforces boundary-token geometry."""

    plan = _plan()
    outcome_fields = {
        "writer_id": plan.canonical_writer_id,
        "native_handle_generation": 7,
        "descriptor_digest": b"d" * 32,
        "evidence_digest": b"e" * 32,
        "metadata": _metadata(),
    }
    multi_segment_plan = dataclasses.replace(
        plan,
        destination_segments=(
            PackedAuxiliaryDestinationSegment(address=0x9000, item_length=64),
            PackedAuxiliaryDestinationSegment(address=0xA000, item_length=128),
            PackedAuxiliaryDestinationSegment(address=0xB000, item_length=512),
        ),
    )
    one_eagle_field_plan = dataclasses.replace(
        plan,
        destination_segments=(
            PackedAuxiliaryDestinationSegment(address=0x9000, item_length=64),
        ),
    )

    for rejected_plan in (multi_segment_plan, one_eagle_field_plan):
        with pytest.raises(ValueError, match="one eight-byte segment"):
            PackedDFlashBoundaryOutcome.create(
                plan=rejected_plan,
                **outcome_fields,
            )


def test_descriptor_pair_is_vram_to_vram_with_zero_host_bytes() -> None:
    """The sole boundary token travels directly between registered devices."""

    agent, source_allocator = _allocator(address_offset=0x10000, device_index=2)
    _, destination_allocator = _allocator(
        address_offset=0x20000,
        device_index=6,
    )
    _, source = _registered_row(source_allocator, object())
    _, destination = _registered_row(destination_allocator, object())

    descriptors = build_dflash_boundary_nixl_descriptors(
        agent,
        source,
        destination,
    )

    assert agent.descriptor_calls == [
        ([source.transfer_request], DFLASH_BOUNDARY_MEMORY_KIND),
        ([destination.transfer_request], DFLASH_BOUNDARY_MEMORY_KIND),
    ]
    assert descriptors.source_request[2] == 2
    assert descriptors.destination_request[2] == 6
    assert descriptors.accounting == DFlashBoundaryTransportAccounting(
        source_vram_bytes=DFLASH_BOUNDARY_ROW_BYTES,
        destination_vram_bytes=DFLASH_BOUNDARY_ROW_BYTES,
        vram_transport_bytes=DFLASH_BOUNDARY_ROW_BYTES,
        dram_transport_bytes=0,
    )
    with pytest.raises(ValueError, match="zero DRAM bytes"):
        dataclasses.replace(descriptors.accounting, dram_transport_bytes=1)


def test_source_projection_orders_vram_gateway_and_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One producer event covers both projections from the same source token."""

    _, allocator = _allocator(row_capacity=1)
    timeline: list[str] = []
    pool = _pool_without_cuda(allocator, _FakeDestinationRow(timeline))
    owner = object()
    reservation = pool.allocate_packed_auxiliary_slot(owner)
    snapshot = pool.packed_auxiliary_slot_reservation_snapshot(reservation)
    stream = _FakeStream(3)
    event = _FakeEvent(timeline)
    gateway_slot = _GatewaySlot(timeline)
    pool._validate_boundary_token = Mock()
    monkeypatch.setattr(dflash_auxiliary.torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(dflash_auxiliary.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        dflash_auxiliary.torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )

    pool.enqueue_source_projection(
        snapshot,
        _FakeBoundaryToken(),
        gateway_slot,
        stream=stream,
        producer_event=event,
    )

    assert timeline == [
        "vram:boundary-token[1]",
        "gateway:boundary-token",
        "event:3",
    ]


def test_destination_adoption_is_device_only_and_never_reads_a_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoder adoption clones the row without any host-visible operation."""

    _, allocator = _allocator(row_capacity=1)
    timeline: list[str] = []
    pool = _pool_without_cuda(allocator, _FakeDestinationRow(timeline))
    owner = object()
    reservation = pool.allocate_packed_auxiliary_slot(owner)
    snapshot = pool.packed_auxiliary_slot_reservation_snapshot(reservation)
    stream = _FakeStream(3)
    events: list[_FakeEvent] = []

    def event_factory(**_kwargs: object) -> _FakeEvent:
        event = _FakeEvent(timeline)
        events.append(event)
        return event

    monkeypatch.setattr(dflash_auxiliary.torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(dflash_auxiliary.torch.cuda, "Event", event_factory)
    monkeypatch.setattr(
        dflash_auxiliary.torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )
    monkeypatch.setattr(dflash_auxiliary, "DFlashBoundaryAdoptedValue", _AdoptedValue)

    adopted = pool.enqueue_destination_adoption(snapshot, stream=stream)

    assert type(adopted) is _AdoptedValue
    assert isinstance(adopted.boundary_token_id, _FakeDestinationRow)
    assert adopted.completion_event is events[0]
    assert timeline == ["clone", "event:3"]
    source = inspect.getsource(DFlashBoundaryDeviceRowPool.enqueue_destination_adoption)
    assert ".cpu(" not in source
    assert ".item(" not in source


def test_terminal_boundary_path_contains_no_legacy_metadata_channel() -> None:
    """Dead EAGLE helpers and polling cannot re-enter terminal DFlash code."""

    source = inspect.getsource(dflash_auxiliary)
    forbidden_fragments = (
        "MetadataBuffers",
        "aux_data_ptrs",
        "aux_item_lens",
        "auxiliary_source_index",
        "output_topk_p",
        "output_topk_index",
        "output_hidden_states",
        "time.sleep(",
    )

    for fragment in forbidden_fragments:
        assert fragment not in source


def test_scalar_metadata_and_outcome_digest_bind_every_counter() -> None:
    """Every frozen scalar changes both metadata and outcome authentication."""

    plan = _plan()
    metadata = _metadata()
    outcome = PackedDFlashBoundaryOutcome.create(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_handle_generation=7,
        descriptor_digest=b"d" * 32,
        evidence_digest=b"e" * 32,
        metadata=metadata,
    )

    for field in dataclasses.fields(metadata):
        field_name = field.name
        changed_metadata = dataclasses.replace(
            metadata,
            **{field_name: getattr(metadata, field_name) + 1},
        )
        changed_outcome = PackedDFlashBoundaryOutcome.create(
            plan=plan,
            writer_id=plan.canonical_writer_id,
            native_handle_generation=7,
            descriptor_digest=b"d" * 32,
            evidence_digest=b"e" * 32,
            metadata=changed_metadata,
        )
        assert changed_metadata.digest != metadata.digest
        assert changed_outcome.outcome_digest != outcome.outcome_digest
        with pytest.raises(ValueError, match="digest differs"):
            dataclasses.replace(outcome, metadata=changed_metadata)


def test_outcome_digest_rejects_plan_and_native_evidence_tampering() -> None:
    """Plan identity and both native attestations are authenticated exactly."""

    plan = _plan()
    outcome = PackedDFlashBoundaryOutcome.create(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_handle_generation=7,
        descriptor_digest=b"d" * 32,
        evidence_digest=b"e" * 32,
        metadata=_metadata(),
    )
    tampered_fields = (
        {"plan": dataclasses.replace(plan, request_slot_generation=8)},
        {"native_handle_generation": 8},
        {"descriptor_digest": b"x" * 32},
        {"evidence_digest": b"y" * 32},
    )

    for changed in tampered_fields:
        with pytest.raises(ValueError, match="digest differs"):
            dataclasses.replace(outcome, **changed)


@pytest.mark.parametrize("value", [-1, 1 << 64, True])
def test_scalar_metadata_requires_exact_uint64_values(value: int) -> None:
    """Control metadata rejects negative, overflowing, and boolean scalars."""

    with pytest.raises(ValueError, match="must be a uint64"):
        dataclasses.replace(_metadata(), cached_tokens=value)


def test_boundary_outcome_wire_round_trip_is_exact() -> None:
    """Wire v8 preserves one-segment geometry and authenticated scalar state."""

    plan = _plan()
    outcome = PackedDFlashBoundaryOutcome.create(
        plan=plan,
        writer_id=plan.canonical_writer_id,
        native_handle_generation=7,
        descriptor_digest=b"d" * 32,
        evidence_digest=b"e" * 32,
        metadata=_metadata(),
    )

    payload = encode_packed_message(outcome)
    decoded = decode_packed_message(payload)

    assert decoded == outcome
    assert type(decoded) is PackedDFlashBoundaryOutcome
    assert len(decoded.plan.destination_segments) == 1
    assert decoded.plan.destination_segments[0].item_length == (
        DFLASH_BOUNDARY_ROW_BYTES
    )
    assert encode_packed_message(decoded) == payload


def test_source_owner_retains_row_and_handle_until_ack_release() -> None:
    """Successful settlement is not storage or native-handle retirement."""

    pool, owner, direct_owner, source_lease, transfer = _posted_transport()

    outcome = owner.settle(transfer, Mock(), _metadata())

    assert type(outcome) is PackedDFlashBoundaryOutcome
    assert outcome.metadata == _metadata()
    assert source_lease.state is DFlashBoundaryRowLeaseState.ACTIVE
    assert pool.inventory() == (0, 1, 0)
    assert owner.inventory().active_count == 1
    assert owner.inventory().settled_count == 1
    assert direct_owner.calls == ["arm", "post", "settle"]

    owner.release(transfer)

    assert source_lease.state is DFlashBoundaryRowLeaseState.RELEASED
    assert pool.inventory() == (1, 0, 0)
    assert owner.inventory().active_count == 0
    assert owner.inventory().released_count == 1
    assert direct_owner.calls == ["arm", "post", "settle", "release"]


def test_receipt_mismatch_quarantines_row_and_retains_native_authority() -> None:
    """Ambiguous completion evidence permanently removes the source row."""

    pool, owner, direct_owner, source_lease, transfer = _posted_transport()
    receipt = direct_owner.receipt
    assert receipt is not None
    direct_owner.receipt = dataclasses.replace(receipt, backend="TCP")

    with pytest.raises(ValueError, match="backend is not UCX"):
        owner.settle(transfer, Mock(), _metadata())

    inventory = owner.inventory()
    assert source_lease.state is DFlashBoundaryRowLeaseState.QUARANTINED
    assert pool.inventory() == (0, 0, 1)
    assert inventory.active_count == 1
    assert inventory.quarantined_count == 1
    assert "release" not in direct_owner.calls


def test_post_failure_quarantines_row_without_disowning_transfer() -> None:
    """A post exception preserves ambiguous native and row authority."""

    agent, allocator = _allocator(row_capacity=1)
    pool = _pool_without_cuda(allocator)
    direct_owner = _DirectOwner()

    def fail_post(_handle: object) -> object:
        raise RuntimeError("injected post failure")

    owner = DFlashBoundarySourceTransportOwner(
        pool=pool,
        agent=agent,
        direct_owner=direct_owner,
        writer_id=_writer(),
        post=fail_post,
    )
    source_lease = owner.lease_source_row()

    with pytest.raises(RuntimeError, match="injected post failure"):
        owner.post(
            plan=_plan(),
            source_lease=source_lease,
            destination_device_index=6,
            remote_handle=object(),
            remote_agent_name="decoder-agent",
            binding_digest=b"b" * 32,
        )

    inventory = owner.inventory()
    assert source_lease.state is DFlashBoundaryRowLeaseState.QUARANTINED
    assert pool.inventory() == (0, 0, 1)
    assert inventory.active_count == 1
    assert inventory.quarantined_count == 1
    assert inventory.unowned_native_handle_count == 0
    assert direct_owner.calls == ["arm", "post"]
