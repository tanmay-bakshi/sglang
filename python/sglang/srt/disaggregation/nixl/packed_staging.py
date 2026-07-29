import dataclasses
import logging
import threading
import traceback

import numpy as np
import numpy.typing as npt
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedChunkKey,
    PackedCommit,
    PackedLayoutSpec,
    PackedLease,
    PackedPrepare,
    PackedReady,
    PackedScatterWork,
    PackedTopology,
)
from sglang.srt.disaggregation.common.staging_layout import (
    DEFAULT_STAGING_ALIGNMENT_BYTES,
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
    StagingWriterLayout,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBuffer,
    StagingComponentBufferRegistry,
    StagingEndpoint,
    StagingEndpointBufferBinding,
    bind_staging_endpoint_buffers,
)

logger = logging.getLogger(__name__)

MAIN_KV_COMPONENT = StagingComponentId(state_index=None, state_type=None)
_UINT64_LIMIT = 1 << 64


def _align_up(value: int, alignment: int) -> int:
    """Round a positive byte count up to allocator alignment.

    :param value: Positive byte count.
    :param alignment: Positive byte alignment.
    :returns: Aligned byte count.
    """

    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


@dataclasses.dataclass(frozen=True)
class PackedDestinationRegistration:
    """Destination geometry advertised to one prefill writer.

    :ivar main_item_lens: Main-KV bytes per page in registration order.
    :ivar main_layer_ids: Main-KV global layer IDs in registration order.
    :ivar state_item_lens: State-component bytes per page.
    :ivar state_layer_ids: State-component global layer IDs.
    :ivar page_size: Tokens represented by one page index.
    """

    main_item_lens: tuple[int, ...]
    main_layer_ids: tuple[int, ...]
    state_item_lens: tuple[tuple[int, ...], ...]
    state_layer_ids: tuple[tuple[int, ...], ...]
    page_size: int

    def __post_init__(self) -> None:
        """Own immutable copies of all registration metadata."""

        object.__setattr__(self, "main_item_lens", tuple(self.main_item_lens))
        object.__setattr__(self, "main_layer_ids", tuple(self.main_layer_ids))
        object.__setattr__(
            self,
            "state_item_lens",
            tuple(tuple(item_lens) for item_lens in self.state_item_lens),
        )
        object.__setattr__(
            self,
            "state_layer_ids",
            tuple(tuple(layer_ids) for layer_ids in self.state_layer_ids),
        )
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if len(self.state_item_lens) != len(self.state_layer_ids):
            raise ValueError(
                "destination state item-length/layer-id component counts differ"
            )


@dataclasses.dataclass(frozen=True)
class PackedComponentPages:
    """Source and destination pages for one active packed component.

    :ivar component_id: Exact main-KV or SWA state identity.
    :ivar source_pages: Source request-local physical pages.
    :ivar destination_pages: Destination request-local physical pages.
    :ivar destination_index_offset: Offset into the decode room's page array.
    """

    component_id: StagingComponentId
    source_pages: npt.NDArray[np.int32]
    destination_pages: npt.NDArray[np.int32]
    destination_index_offset: int

    def __post_init__(self) -> None:
        """Own immutable page-array snapshots."""

        source_pages = _immutable_page_array(self.source_pages, "source")
        destination_pages = _immutable_page_array(
            self.destination_pages,
            "destination",
        )
        if len(source_pages) == 0:
            raise ValueError("packed component must contain at least one page")
        if len(source_pages) != len(destination_pages):
            raise ValueError(
                "source/destination packed page counts differ: "
                f"{len(source_pages)} and {len(destination_pages)}"
            )
        if self.destination_index_offset < 0:
            raise ValueError(
                "destination_index_offset must be non-negative, got "
                f"{self.destination_index_offset}"
            )
        object.__setattr__(self, "source_pages", source_pages)
        object.__setattr__(self, "destination_pages", destination_pages)


def _immutable_page_array(
    pages: npt.NDArray[np.int32],
    label: str,
) -> npt.NDArray[np.int32]:
    """Return one immutable contiguous int32 page snapshot.

    :param pages: Page array to copy.
    :param label: Reader-facing endpoint label.
    :returns: Immutable page snapshot.
    """

    if not isinstance(pages, np.ndarray):
        raise TypeError(f"{label} pages must be a NumPy array")
    if pages.dtype != np.dtype(np.int32):
        raise TypeError(f"{label} pages must have dtype int32, got {pages.dtype}")
    if pages.ndim != 1:
        raise TypeError(f"{label} pages must be one-dimensional")
    snapshot = np.array(pages, order="C", copy=True)
    if np.any(snapshot < 0):
        raise ValueError(f"{label} pages contain a negative index")
    snapshot.setflags(write=False)
    return snapshot


def _component_id(kv_args: KVArgs, state_index: int | None) -> StagingComponentId:
    """Resolve an exact local component identity.

    :param kv_args: Local registered KV metadata.
    :param state_index: State component index, or ``None`` for main KV.
    :returns: Exact component identity.
    """

    if state_index is None:
        return MAIN_KV_COMPONENT
    if state_index < 0 or state_index >= len(kv_args.state_types):
        raise ValueError(f"state_index is out of range: {state_index}")
    return StagingComponentId(
        state_index=state_index,
        state_type=kv_args.state_types[state_index],
    )


def local_component_geometry(
    kv_args: KVArgs,
    component_id: StagingComponentId,
) -> StagingComponentGeometry:
    """Return local registered geometry for one packable component.

    :param kv_args: Local registered KV metadata.
    :param component_id: Exact main-KV or state identity.
    :returns: Immutable registered component geometry.
    :raises ValueError: If the component is missing or not SWA-packable.
    """

    state_index = component_id.state_index
    if state_index is None:
        if component_id != MAIN_KV_COMPONENT:
            raise ValueError("invalid main-KV component identity")
        return StagingComponentGeometry(
            component_id=component_id,
            item_lens=tuple(kv_args.kv_item_lens),
            layer_ids=tuple(kv_args.kv_layer_ids),
            page_size=kv_args.page_size,
        )
    expected_component = _component_id(kv_args, state_index)
    if component_id != expected_component:
        raise ValueError(
            f"state component identity differs from local registration: {component_id}"
        )
    if component_id.state_type is not StateType.SWA:
        raise ValueError(
            f"packed staging supports only SWA state, got {component_id.state_type}"
        )
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=tuple(kv_args.state_item_lens[state_index]),
        layer_ids=tuple(kv_args.state_layer_ids[state_index]),
        page_size=kv_args.page_size,
    )


def destination_component_geometry(
    registration: PackedDestinationRegistration,
    state_types: tuple[StateType, ...],
    component_id: StagingComponentId,
) -> StagingComponentGeometry:
    """Return advertised destination geometry for one active component.

    :param registration: Destination registration metadata.
    :param state_types: Source-local state ordering expected at the destination.
    :param component_id: Exact main-KV or SWA state identity.
    :returns: Immutable destination geometry.
    """

    state_index = component_id.state_index
    if state_index is None:
        if component_id != MAIN_KV_COMPONENT:
            raise ValueError("invalid main-KV component identity")
        return StagingComponentGeometry(
            component_id=component_id,
            item_lens=registration.main_item_lens,
            layer_ids=registration.main_layer_ids,
            page_size=registration.page_size,
        )
    if state_index < 0 or state_index >= len(state_types):
        raise ValueError(f"state_index is out of range: {state_index}")
    if state_types[state_index] is not StateType.SWA:
        raise ValueError(
            f"packed staging supports only SWA state, got {state_types[state_index]}"
        )
    if component_id.state_type is not state_types[state_index]:
        raise ValueError("state component identity differs from source registration")
    if state_index >= len(registration.state_item_lens):
        raise ValueError(f"destination state component is missing: {state_index}")
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=registration.state_item_lens[state_index],
        layer_ids=registration.state_layer_ids[state_index],
        page_size=registration.page_size,
    )


def derive_source_geometry(
    destination: StagingComponentGeometry,
    source_tp_size: int,
    destination_tp_size: int,
) -> StagingComponentGeometry:
    """Derive the only valid non-replicated source TP geometry.

    :param destination: Decode-local component geometry.
    :param source_tp_size: Source attention tensor-parallel width.
    :param destination_tp_size: Destination attention tensor-parallel width.
    :returns: Expected source per-rank geometry.
    :raises ValueError: If aggregate bytes cannot be partitioned exactly.
    """

    if source_tp_size <= 0 or destination_tp_size <= 0:
        raise ValueError("tensor-parallel sizes must be positive")
    source_item_lens: list[int] = []
    for destination_item_len in destination.item_lens:
        aggregate_item_len = destination_item_len * destination_tp_size
        if aggregate_item_len % source_tp_size != 0:
            raise ValueError(
                "destination geometry cannot be exactly partitioned by source TP: "
                f"{aggregate_item_len} bytes across {source_tp_size} ranks"
            )
        source_item_lens.append(aggregate_item_len // source_tp_size)
    return StagingComponentGeometry(
        component_id=destination.component_id,
        item_lens=tuple(source_item_lens),
        layer_ids=destination.layer_ids,
        page_size=destination.page_size,
    )


def build_component_buffer_registry(
    kv_args: KVArgs,
    page_arrays: dict[StagingComponentId, npt.NDArray[np.int32]],
) -> StagingComponentBufferRegistry:
    """Bind local registered pointers to request-local component pages.

    :param kv_args: Local registered KV metadata.
    :param page_arrays: Exact active component page arrays.
    :returns: Immutable component buffer registry.
    """

    components: list[StagingComponentBuffer] = []
    for component_id, page_array in page_arrays.items():
        geometry = local_component_geometry(kv_args, component_id)
        state_index = component_id.state_index
        if state_index is None:
            tensor_ptrs = tuple(kv_args.kv_data_ptrs)
            data_lens = tuple(kv_args.kv_data_lens)
        else:
            tensor_ptrs = tuple(kv_args.state_data_ptrs[state_index])
            data_lens = tuple(kv_args.state_data_lens[state_index])
        components.append(
            StagingComponentBuffer(
                component_id=component_id,
                tensor_ptrs=tensor_ptrs,
                data_lens=data_lens,
                item_lens=geometry.item_lens,
                layer_ids=geometry.layer_ids,
                page_size=geometry.page_size,
                page_array=page_array,
            )
        )
    return StagingComponentBufferRegistry(tuple(components))


def active_destination_page_arrays(
    kv_args: KVArgs,
    kv_indices: npt.NDArray[np.int32],
    state_indices: list[npt.NDArray[np.int32] | None] | None,
) -> dict[StagingComponentId, npt.NDArray[np.int32]]:
    """Snapshot every destination component eligible for packed transfer.

    :param kv_args: Decode-local registered KV metadata.
    :param kv_indices: Complete room-local main-KV page array.
    :param state_indices: Complete room-local state page arrays.
    :returns: Page arrays keyed by exact component identity.
    """

    page_arrays: dict[StagingComponentId, npt.NDArray[np.int32]] = {
        MAIN_KV_COMPONENT: _immutable_page_array(kv_indices, "destination main-KV")
    }
    if state_indices is None:
        return page_arrays
    if len(state_indices) != len(kv_args.state_types):
        raise ValueError(
            "destination state index count differs from registered state types: "
            f"{len(state_indices)} and {len(kv_args.state_types)}"
        )
    for state_index, pages in enumerate(state_indices):
        if pages is None or len(pages) == 0:
            continue
        if kv_args.state_types[state_index] is not StateType.SWA:
            continue
        component_id = _component_id(kv_args, state_index)
        page_arrays[component_id] = _immutable_page_array(
            pages,
            f"destination state[{state_index}]",
        )
    return page_arrays


def _validate_component_shape(
    *,
    is_last: bool,
    component_ids: tuple[StagingComponentId, ...],
    required_final_components: frozenset[StagingComponentId],
) -> None:
    """Validate the supported intermediate and final component shapes.

    :param is_last: Whether the chunk completes the room.
    :param component_ids: Active components.
    :param required_final_components: Destination state components required at
        final completion.
    :raises ValueError: If the shape is unsupported or omits required state.
    """

    component_set = set(component_ids)
    if len(component_set) != len(component_ids):
        raise ValueError("packed component identities must be unique")
    for component_id in component_ids:
        if component_id == MAIN_KV_COMPONENT:
            continue
        if component_id.state_type is not StateType.SWA:
            raise ValueError(
                f"packed staging supports only SWA state, got {component_id}"
            )
    if not is_last:
        if component_set != {MAIN_KV_COMPONENT}:
            raise ValueError("intermediate packed chunks must be main-KV-only")
        return
    if len(component_set) == 0:
        raise ValueError("final packed chunk must contain main KV or SWA")
    if not required_final_components.issubset(component_set):
        missing = sorted(
            required_final_components - component_set,
            key=lambda component_id: (
                component_id.state_index if component_id.state_index is not None else -1
            ),
        )
        raise ValueError(f"final packed chunk omits required components: {missing}")


def build_prefill_chunk(
    *,
    key: PackedChunkKey,
    is_last: bool,
    kv_args: KVArgs,
    destination_registration: PackedDestinationRegistration,
    components: tuple[PackedComponentPages, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    writers: tuple[StagingWriterId, ...],
) -> tuple[PackedLayoutSpec, StagingEndpointBufferBinding]:
    """Build and bind one source-authored packed chunk.

    :param key: Request and chunk identity.
    :param is_last: Whether the chunk completes the room.
    :param kv_args: Source-local registered KV metadata.
    :param destination_registration: Decode geometry from bootstrap metadata.
    :param components: Active source and destination page projections.
    :param source_tp_size: Source attention tensor-parallel width.
    :param destination_tp_size: Destination attention tensor-parallel width.
    :param destination_tp_rank: Destination attention tensor-parallel rank.
    :param writers: Complete writer set connected to this destination.
    :returns: Canonical layout spec and immutable source binding.
    """

    component_ids = tuple(component.component_id for component in components)
    required_final_components = frozenset(
        component_id
        for component_id in component_ids
        if component_id != MAIN_KV_COMPONENT
    )
    _validate_component_shape(
        is_last=is_last,
        component_ids=component_ids,
        required_final_components=required_final_components,
    )
    source_components = tuple(
        local_component_geometry(kv_args, component_id)
        for component_id in component_ids
    )
    state_types = tuple(kv_args.state_types)
    destination_components = tuple(
        destination_component_geometry(
            destination_registration,
            state_types,
            component_id,
        )
        for component_id in component_ids
    )
    spans = tuple(
        StagingComponentSpan(
            component_id=component.component_id,
            source_index_offset=0,
            destination_index_offset=component.destination_index_offset,
            logical_token_count=len(component.source_pages) * kv_args.page_size,
            physical_token_count=len(component.source_pages) * kv_args.page_size,
        )
        for component in components
    )
    spec = PackedLayoutSpec(
        chunk_id=key.chunk_id,
        is_last=is_last,
        spans=spans,
        source_components=source_components,
        destination_components=destination_components,
        writers=writers,
        topology=PackedTopology(
            source_tp_size=source_tp_size,
            destination_tp_size=destination_tp_size,
            destination_tp_rank=destination_tp_rank,
        ),
    )
    source_registry = build_component_buffer_registry(
        kv_args,
        {component.component_id: component.source_pages for component in components},
    )
    source_binding = bind_staging_endpoint_buffers(
        spec.build(),
        StagingEndpoint.SOURCE,
        source_registry,
    )
    return spec, source_binding


def build_decode_spec(
    *,
    chunk_id: int,
    is_last: bool,
    spans: tuple[StagingComponentSpan, ...],
    kv_args: KVArgs,
    expected_writers: tuple[StagingWriterId, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    required_final_components: frozenset[StagingComponentId],
) -> PackedLayoutSpec:
    """Build decode-local canonical truth before accepting PREPARE.

    :param chunk_id: Trusted request-local chunk identifier.
    :param is_last: Trusted final-chunk marker.
    :param spans: Trusted component spans derived from room metadata.
    :param kv_args: Decode-local registered KV metadata.
    :param expected_writers: Writers authenticated by bootstrap routing.
    :param source_tp_size: Expected source attention TP width.
    :param destination_tp_size: Decode attention TP width.
    :param destination_tp_rank: Decode attention TP rank.
    :param required_final_components: Non-empty room-local SWA components.
    :returns: Trusted canonical layout input.
    """

    expected_topology = PackedTopology(
        source_tp_size=source_tp_size,
        destination_tp_size=destination_tp_size,
        destination_tp_rank=destination_tp_rank,
    )
    component_ids = tuple(span.component_id for span in spans)
    _validate_component_shape(
        is_last=is_last,
        component_ids=component_ids,
        required_final_components=required_final_components,
    )
    for span in spans:
        if span.source_index_offset != 0:
            raise ValueError("packed source spans must start at request-local offset 0")
        if (
            span.component_id != MAIN_KV_COMPONENT
            and span.destination_index_offset != 0
        ):
            raise ValueError("packed SWA destination spans must start at offset 0")
        if span.logical_token_count != span.physical_token_count:
            raise ValueError(
                "packed NIXL spans currently require complete physical token rows"
            )
    destination_components = tuple(
        local_component_geometry(kv_args, component_id)
        for component_id in component_ids
    )
    source_components = tuple(
        derive_source_geometry(
            destination,
            source_tp_size,
            destination_tp_size,
        )
        for destination in destination_components
    )
    return PackedLayoutSpec(
        chunk_id=chunk_id,
        is_last=is_last,
        spans=spans,
        source_components=source_components,
        destination_components=destination_components,
        writers=expected_writers,
        topology=expected_topology,
    )


class PackedIntervalLeaseAllocator:
    """Non-overlapping first-fit allocator for one registered GPU buffer."""

    _alignment_bytes: int
    _allocations: dict[int, tuple[int, int]]
    _base_address: int
    _free_intervals: list[tuple[int, int]]
    _lock: threading.Lock
    _next_lease_id: int
    _quarantined: set[int]
    _total_size: int

    def __init__(
        self,
        *,
        base_address: int,
        total_size: int,
        alignment_bytes: int = DEFAULT_STAGING_ALIGNMENT_BYTES,
    ) -> None:
        """Initialize one empty contiguous allocation arena.

        :param base_address: Registered GPU buffer base pointer.
        :param total_size: Registered buffer capacity.
        :param alignment_bytes: Allocation alignment.
        """

        if base_address <= 0:
            raise ValueError(f"base_address must be positive, got {base_address}")
        if total_size <= 0:
            raise ValueError(f"total_size must be positive, got {total_size}")
        if alignment_bytes <= 0:
            raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
        if base_address % alignment_bytes != 0:
            raise ValueError(
                "base_address must satisfy allocator alignment: "
                f"{base_address} % {alignment_bytes} != 0"
            )
        self._alignment_bytes = alignment_bytes
        self._allocations = {}
        self._base_address = base_address
        self._free_intervals = [(0, total_size)]
        self._lock = threading.Lock()
        self._next_lease_id = 0
        self._quarantined = set()
        self._total_size = total_size

    @property
    def total_size(self) -> int:
        """Return registered arena capacity.

        :returns: Total bytes.
        """

        return self._total_size

    def allocate(self, length_bytes: int) -> PackedLease:
        """Allocate one non-overlapping aligned interval without waiting.

        :param length_bytes: Minimum required bytes.
        :returns: Contiguous registered lease.
        :raises MemoryError: If no sufficiently large interval is free.
        """

        aligned_length = _align_up(length_bytes, self._alignment_bytes)
        with self._lock:
            selected_index: int | None = None
            selected_offset = 0
            selected_length = 0
            for interval_index, (offset, free_length) in enumerate(
                self._free_intervals
            ):
                if free_length < aligned_length:
                    continue
                selected_index = interval_index
                selected_offset = offset
                selected_length = free_length
                break
            if selected_index is None:
                raise MemoryError(
                    f"packed staging pool cannot allocate {aligned_length} bytes"
                )
            remaining = selected_length - aligned_length
            if remaining == 0:
                self._free_intervals.pop(selected_index)
            else:
                self._free_intervals[selected_index] = (
                    selected_offset + aligned_length,
                    remaining,
                )
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            self._allocations[lease_id] = (selected_offset, aligned_length)
            return PackedLease(
                lease_id=lease_id,
                base_address=self._base_address + selected_offset,
                length_bytes=aligned_length,
            )

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Mark one live lease as failed without making it allocatable.

        :param lease: Failed live lease.
        :param reason: First protocol failure reason.
        """

        if len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        with self._lock:
            self._validate_live_lease(lease)
            self._quarantined.add(lease.lease_id)

    def release(self, lease: PackedLease) -> None:
        """Release and coalesce one terminally quiescent lease.

        :param lease: Live lease safe for immediate reuse.
        """

        with self._lock:
            offset, length = self._validate_live_lease(lease)
            candidate_intervals = sorted((*self._free_intervals, (offset, length)))
            merged: list[tuple[int, int]] = []
            for free_offset, free_length in candidate_intervals:
                if len(merged) == 0:
                    merged.append((free_offset, free_length))
                    continue
                previous_offset, previous_length = merged[-1]
                previous_end = previous_offset + previous_length
                if previous_end == free_offset:
                    merged[-1] = (
                        previous_offset,
                        previous_length + free_length,
                    )
                    continue
                if previous_end > free_offset:
                    raise RuntimeError("packed allocator free intervals overlap")
                merged.append((free_offset, free_length))
            del self._allocations[lease.lease_id]
            self._quarantined.discard(lease.lease_id)
            self._free_intervals = merged

    def _validate_live_lease(self, lease: PackedLease) -> tuple[int, int]:
        """Validate lease identity without mutating allocator ownership.

        :param lease: Lease expected to be live.
        :returns: Internal offset and aligned length.
        :raises ValueError: If identity or geometry differs.
        """

        allocation = self._allocations.get(lease.lease_id)
        if allocation is None:
            raise ValueError(f"packed lease is not live: {lease.lease_id}")
        offset, length = allocation
        if lease.base_address != self._base_address + offset:
            raise ValueError("packed lease base address differs from allocation")
        if lease.length_bytes != length:
            raise ValueError("packed lease length differs from allocation")
        return allocation


def writer_layout_for(
    layout: StagingChunkLayout,
    writer_id: StagingWriterId,
) -> StagingWriterLayout:
    """Return one writer's canonical projection.

    :param layout: Canonical packed layout.
    :param writer_id: Exact writer identity.
    :returns: Writer projection.
    :raises ValueError: If the writer is absent.
    """

    for writer_layout in layout.writers:
        if writer_layout.writer_id == writer_id:
            return writer_layout
    raise ValueError(f"writer is absent from packed layout: {writer_id}")


def _validate_source_binding(
    layout: StagingChunkLayout,
    binding: StagingEndpointBufferBinding,
) -> None:
    """Validate a source binding against one canonical layout.

    :param layout: Canonical packed layout.
    :param binding: Source-side registered buffer binding.
    :raises ValueError: If endpoint, geometry, ordering, or page counts differ.
    """

    if binding.endpoint is not StagingEndpoint.SOURCE:
        raise ValueError("packed source coordinator requires a source binding")
    if len(binding.components) != len(layout.component_spans):
        raise ValueError("packed source binding component count differs from layout")
    for geometry, span, active in zip(
        layout.source_components,
        layout.component_spans,
        binding.components,
        strict=True,
    ):
        if active.component.component_id != span.component_id:
            raise ValueError(
                "packed source binding component order differs from layout"
            )
        if active.component.geometry != geometry:
            raise ValueError("packed source binding geometry differs from layout")
        expected_page_count = span.physical_token_count // geometry.page_size
        if active.page_offset != span.source_index_offset:
            raise ValueError("packed source binding page offset differs from layout")
        if active.page_count != expected_page_count:
            raise ValueError("packed source binding page count differs from layout")
        if len(active.page_array) != expected_page_count:
            raise ValueError("packed source binding page snapshot is incomplete")


def _checked_uint64_region(address: int, length_bytes: int, label: str) -> int:
    """Validate one address region representable by NIXL descriptors.

    :param address: Region base address.
    :param length_bytes: Positive region length.
    :param label: Reader-facing region label.
    :returns: Exclusive region end.
    """

    if type(address) is not int or address <= 0:
        raise ValueError(f"{label} address must be a positive integer")
    if type(length_bytes) is not int or length_bytes <= 0:
        raise ValueError(f"{label} length must be a positive integer")
    end = address + length_bytes
    if address >= _UINT64_LIMIT or end > _UINT64_LIMIT:
        raise ValueError(f"{label} exceeds the uint64 address space")
    return end


class PackedReadyError(RuntimeError):
    """Source-side rejection of an untrusted READY message."""


@dataclasses.dataclass(frozen=True)
class PackedSourceTransfer:
    """Canonical one-shot source work produced by a validated READY.

    :ivar key: Request and chunk identity.
    :ivar decode_peer_name: Authenticated NIXL destination peer.
    :ivar destination_gpu_id: Bootstrap-derived destination CUDA device.
    :ivar layout: Locally rebuilt canonical packed layout.
    :ivar writer_id: Local authenticated writer identity.
    :ivar source_binding: Immutable request-local source pages and buffers.
    :ivar lease_id: Decode lease identity copied into COMMIT.
    :ivar destination_address: Exact destination projection base address.
    :ivar length_bytes: Exact local canonical DMA length.
    :ivar commit: Completion notification emitted by the terminal NIXL write.
    """

    key: PackedChunkKey
    decode_peer_name: str
    destination_gpu_id: int
    layout: StagingChunkLayout
    writer_id: StagingWriterId
    source_binding: StagingEndpointBufferBinding
    lease_id: int
    destination_address: int
    length_bytes: int
    commit: PackedCommit


@dataclasses.dataclass(frozen=True)
class _PendingPackedSource:
    """Immutable canonical truth retained until exactly one READY is accepted."""

    decode_peer_name: str
    destination_gpu_id: int
    layout: StagingChunkLayout
    writer_id: StagingWriterId
    source_binding: StagingEndpointBufferBinding


class PackedReadyCoordinator:
    """Thread-safe one-shot validation of decode READY messages."""

    _lock: threading.Lock
    _pending: dict[tuple[PackedChunkKey, str], _PendingPackedSource]

    def __init__(self) -> None:
        """Initialize an empty source-side READY registry."""

        self._lock = threading.Lock()
        self._pending = {}

    def register_chunk(
        self,
        *,
        key: PackedChunkKey,
        decode_peer_name: str,
        destination_gpu_id: int,
        writer_id: StagingWriterId,
        spec: PackedLayoutSpec,
        source_binding: StagingEndpointBufferBinding,
    ) -> PackedPrepare:
        """Register local truth and build the PREPARE sent to decode.

        :param key: Request and chunk identity.
        :param decode_peer_name: Exact bootstrap-authenticated decode peer.
        :param destination_gpu_id: Bootstrap-derived destination CUDA device.
        :param writer_id: Local transfer writer identity.
        :param spec: Source-built canonical layout input.
        :param source_binding: Immutable source registration and page snapshots.
        :returns: PREPARE carrying the canonical layout digest.
        """

        if type(decode_peer_name) is not str or len(decode_peer_name) == 0:
            raise ValueError("decode_peer_name must be a non-empty string")
        if type(destination_gpu_id) is not int or destination_gpu_id < 0:
            raise ValueError("destination_gpu_id must be a non-negative integer")
        if key.chunk_id != spec.chunk_id:
            raise ValueError(
                f"chunk key/spec mismatch: {key.chunk_id} and {spec.chunk_id}"
            )
        layout = spec.build()
        writer_layout_for(layout, writer_id)
        _validate_source_binding(layout, source_binding)
        pending = _PendingPackedSource(
            decode_peer_name=decode_peer_name,
            destination_gpu_id=destination_gpu_id,
            layout=layout,
            writer_id=writer_id,
            source_binding=source_binding,
        )
        route_key = (key, decode_peer_name)
        with self._lock:
            if route_key in self._pending:
                raise ValueError(
                    "packed source route is already registered: "
                    f"{key} via {decode_peer_name}"
                )
            self._pending[route_key] = pending
        return PackedPrepare(
            key=key,
            writer_id=writer_id,
            spec=spec,
            digest=layout.digest,
        )

    def handle_ready(
        self,
        message: PackedReady,
        authenticated_decode_peer: str,
    ) -> PackedSourceTransfer:
        """Consume one validated READY and hand out exactly one DMA submission.

        Every address and shape field from READY is checked against locally
        retained canonical truth. The returned DMA length and gather layout are
        local values, never values selected by the peer.

        :param message: Untrusted READY payload.
        :param authenticated_decode_peer: NIXL peer bound to the exact route.
        :returns: Canonical gather, destination, and completion-notification work.
        :raises PackedReadyError: If READY conflicts with local truth.
        """

        with self._lock:
            route_key = (message.key, authenticated_decode_peer)
            pending = self._pending.get(route_key)
            if pending is None:
                raise PackedReadyError(
                    "packed source route is not pending: "
                    f"{message.key} via {authenticated_decode_peer}"
                )
            try:
                transfer = self._validate_ready(
                    message, authenticated_decode_peer, pending
                )
            except ValueError as error:
                raise PackedReadyError(
                    f"packed READY rejected for {message.key}: {error}"
                ) from error
            del self._pending[route_key]
            return transfer

    def retire_pending(self, key: PackedChunkKey, decode_peer_name: str) -> None:
        """Forget one route that cannot receive READY.

        :param key: Request and chunk identity.
        :param decode_peer_name: Exact registered decode peer.
        """

        with self._lock:
            self._pending.pop((key, decode_peer_name), None)

    @staticmethod
    def _validate_ready(
        message: PackedReady,
        authenticated_decode_peer: str,
        pending: _PendingPackedSource,
    ) -> PackedSourceTransfer:
        """Validate READY without mutating coordinator ownership.

        :param message: Untrusted READY payload.
        :param authenticated_decode_peer: Authenticated route peer.
        :param pending: Locally retained canonical source truth.
        :returns: Canonical one-shot transfer work.
        """

        if authenticated_decode_peer != pending.decode_peer_name:
            raise ValueError("decode peer does not match the registered route")
        if message.writer_id != pending.writer_id:
            raise ValueError("writer identity differs from local transfer writer")
        if message.digest != pending.layout.digest:
            raise ValueError("digest differs from the local canonical layout")
        writer_layout = writer_layout_for(pending.layout, pending.writer_id)
        if message.projection_offset != writer_layout.lease_offset:
            raise ValueError("projection offset differs from the canonical layout")
        if message.projection_length != writer_layout.length_bytes:
            raise ValueError("projection length differs from the canonical layout")
        projection_end = writer_layout.lease_offset + writer_layout.length_bytes
        if projection_end > pending.layout.total_bytes:
            raise ValueError("canonical writer projection exceeds the packed lease")
        _checked_uint64_region(
            message.lease_base_address,
            pending.layout.total_bytes,
            "packed lease",
        )
        destination_address = message.lease_base_address + writer_layout.lease_offset
        _checked_uint64_region(
            destination_address,
            writer_layout.length_bytes,
            "packed writer projection",
        )
        commit = PackedCommit(
            key=message.key,
            writer_id=pending.writer_id,
            digest=pending.layout.digest,
            lease_id=message.lease_id,
        )
        return PackedSourceTransfer(
            key=message.key,
            decode_peer_name=pending.decode_peer_name,
            destination_gpu_id=pending.destination_gpu_id,
            layout=pending.layout,
            writer_id=pending.writer_id,
            source_binding=pending.source_binding,
            lease_id=message.lease_id,
            destination_address=destination_address,
            length_bytes=writer_layout.length_bytes,
            commit=commit,
        )


@triton.jit
def _gather_packed_bytes_kernel(
    entry_ptrs,
    page_indices,
    staging,
    staging_offset,
    physical_tokens,
    source_token_bytes,
    source_offset_bytes,
    copy_bytes_per_token: tl.constexpr,
    page_size: tl.constexpr,
    bytes_per_entry,
    block_size: tl.constexpr,
):
    entry_index = tl.program_id(0)
    block_index = tl.program_id(1)
    offsets = block_index * block_size + tl.arange(0, block_size)
    mask = offsets < bytes_per_entry
    token_index = offsets // copy_bytes_per_token
    token_byte = offsets % copy_bytes_per_token
    page_index = token_index // page_size
    page_offset = token_index % page_size
    physical_page = tl.load(page_indices + page_index, mask=mask, other=0)
    entry_ptr = tl.load(entry_ptrs + entry_index).to(staging.dtype)
    source_offsets = (
        physical_page * page_size * source_token_bytes
        + page_offset * source_token_bytes
        + source_offset_bytes
        + token_byte
    )
    values = tl.load(entry_ptr + source_offsets, mask=mask)
    destination_offsets = (
        staging_offset + entry_index * physical_tokens * copy_bytes_per_token + offsets
    )
    tl.store(staging + destination_offsets, values, mask=mask)


@triton.jit
def _scatter_packed_bytes_kernel(
    entry_ptrs,
    page_indices,
    staging,
    staging_offset,
    physical_tokens,
    destination_token_bytes,
    destination_offset_bytes,
    copy_bytes_per_token: tl.constexpr,
    page_size: tl.constexpr,
    bytes_per_entry,
    block_size: tl.constexpr,
):
    entry_index = tl.program_id(0)
    block_index = tl.program_id(1)
    offsets = block_index * block_size + tl.arange(0, block_size)
    mask = offsets < bytes_per_entry
    token_index = offsets // copy_bytes_per_token
    token_byte = offsets % copy_bytes_per_token
    page_index = token_index // page_size
    page_offset = token_index % page_size
    physical_page = tl.load(page_indices + page_index, mask=mask, other=0)
    entry_ptr = tl.load(entry_ptrs + entry_index).to(staging.dtype)
    source_offsets = (
        staging_offset + entry_index * physical_tokens * copy_bytes_per_token + offsets
    )
    values = tl.load(staging + source_offsets, mask=mask)
    destination_offsets = (
        physical_page * page_size * destination_token_bytes
        + page_offset * destination_token_bytes
        + destination_offset_bytes
        + token_byte
    )
    tl.store(entry_ptr + destination_offsets, values, mask=mask)


class PackedSourceBuffer:
    """Lazily grown registered source buffer owned by one transfer lane.

    A lane must have at most one NIXL read in flight. Source ranks that fan out
    to multiple decode routes need a distinct buffer per concurrently active
    route; gathering another projection into a buffer still read by NIXL would
    corrupt the earlier transfer.
    """

    _agent: object
    _capacity: int
    _device: torch.device
    _gpu_id: int
    _registration: object | None
    _tensor: torch.Tensor | None

    def __init__(self, agent: object, gpu_id: int) -> None:
        """Initialize an allocation-free worker buffer.

        :param agent: NIXL agent used for memory registration.
        :param gpu_id: Source CUDA device identifier.
        """

        self._agent = agent
        self._capacity = 0
        self._device = torch.device(f"cuda:{gpu_id}")
        self._gpu_id = gpu_id
        self._registration = None
        self._tensor = None

    @property
    def tensor(self) -> torch.Tensor:
        """Return the currently allocated byte tensor.

        :returns: Registered source byte tensor.
        :raises RuntimeError: If capacity has not been requested.
        """

        if self._tensor is None:
            raise RuntimeError("packed source buffer is not allocated")
        return self._tensor

    @property
    def data_ptr(self) -> int:
        """Return the registered source pointer.

        :returns: Source GPU base pointer.
        """

        return self.tensor.data_ptr()

    def ensure_capacity(self, required_bytes: int) -> None:
        """Grow to a power-of-two capacity while the lane has no DMA in flight.

        :param required_bytes: Minimum projection capacity.
        """

        if required_bytes <= self._capacity:
            return
        minimum_capacity = max(1 << 20, required_bytes)
        capacity = 1 << (minimum_capacity - 1).bit_length()
        torch.cuda.set_device(self._gpu_id)
        tensor = torch.empty(capacity, dtype=torch.uint8, device=self._device)
        registration = self._agent.register_memory(
            [(tensor.data_ptr(), capacity, self._gpu_id, "")],
            "VRAM",
        )
        if not registration:
            raise RuntimeError(
                f"NIXL failed to register packed source buffer of {capacity} bytes"
            )
        previous_registration = self._registration
        if previous_registration is not None:
            try:
                self._agent.deregister_memory(previous_registration)
            except Exception:
                logger.error(
                    "Failed to replace a packed source-buffer registration:\n%s",
                    traceback.format_exc(),
                )
                try:
                    self._agent.deregister_memory(registration)
                except Exception:  # noqa: BLE001
                    logger.error(
                        "Failed to roll back the replacement registration:\n%s",
                        traceback.format_exc(),
                    )
                raise
        self._tensor = tensor
        self._registration = registration
        self._capacity = capacity


@dataclasses.dataclass(frozen=True)
class PackedScatterSubmission:
    """Resources retained until one asynchronous scatter event is terminal.

    :ivar event: CUDA completion event.
    :ivar resources: Temporary pointer and page tensors used by kernels.
    """

    event: torch.cuda.Event
    resources: tuple[torch.Tensor, ...]


class PackedCopyExecutor:
    """Component-aware raw-byte gather and scatter kernel dispatcher."""

    _device: torch.device
    _gpu_id: int
    _scatter_base_address: int | None
    _scatter_buffer: torch.Tensor | None
    _scatter_stream: torch.cuda.Stream
    _source_stream: torch.cuda.Stream

    def __init__(
        self,
        *,
        gpu_id: int,
        scatter_buffer: torch.Tensor | None = None,
    ) -> None:
        """Initialize dedicated gather and scatter streams.

        :param gpu_id: CUDA device identifier.
        :param scatter_buffer: Decode staging-pool byte tensor.
        """

        self._device = torch.device(f"cuda:{gpu_id}")
        self._gpu_id = gpu_id
        self._scatter_buffer = scatter_buffer
        self._scatter_base_address = (
            scatter_buffer.data_ptr() if scatter_buffer is not None else None
        )
        torch.cuda.set_device(gpu_id)
        self._source_stream = torch.cuda.Stream(device=self._device)
        self._scatter_stream = torch.cuda.Stream(device=self._device)

    def gather(
        self,
        *,
        layout: StagingChunkLayout,
        writer_id: StagingWriterId,
        source_binding: StagingEndpointBufferBinding,
        source_buffer: PackedSourceBuffer,
    ) -> int:
        """Gather one writer projection and wait until NIC-visible.

        The caller owns the source-buffer lane until the corresponding NIXL
        transfer is terminal. This method synchronizes the gather kernel, not a
        later transport read from the registered buffer.

        :param layout: Canonical packed layout.
        :param writer_id: Local writer identity.
        :param source_binding: Immutable source page snapshots.
        :param source_buffer: Worker-owned registered staging buffer.
        :returns: Exact contiguous DMA length.
        """

        writer_layout = writer_layout_for(layout, writer_id)
        source_buffer.ensure_capacity(writer_layout.length_bytes)
        retained: list[torch.Tensor] = []
        with torch.cuda.stream(self._source_stream):
            source_buffer.tensor[: writer_layout.length_bytes].zero_()
            for group in writer_layout.copy_groups:
                active = source_binding.require(group.component_id)
                entry_ptrs = torch.tensor(
                    [
                        active.component.tensor_ptrs[entry_index]
                        for entry_index in group.source_entry_indices
                    ],
                    dtype=torch.int64,
                    device=self._device,
                )
                page_indices = torch.from_numpy(
                    np.array(active.page_array, dtype=np.int32, copy=True)
                ).to(self._device)
                retained.extend((entry_ptrs, page_indices))
                physical_tokens = group.page_count * active.component.page_size
                bytes_per_entry = physical_tokens * group.copy_bytes_per_token
                grid = (
                    len(group.source_entry_indices),
                    triton.cdiv(bytes_per_entry, 256),
                )
                _gather_packed_bytes_kernel[grid](
                    entry_ptrs,
                    page_indices,
                    source_buffer.tensor,
                    group.packed_offset - writer_layout.lease_offset,
                    physical_tokens,
                    group.source_token_bytes,
                    group.source_offset_bytes,
                    copy_bytes_per_token=group.copy_bytes_per_token,
                    page_size=active.component.page_size,
                    bytes_per_entry=bytes_per_entry,
                    block_size=256,
                )
        self._source_stream.synchronize()
        return writer_layout.length_bytes

    def scatter(self, work: PackedScatterWork) -> PackedScatterSubmission:
        """Launch component-aware scatter without blocking the caller.

        :param work: Protocol-owned immutable scatter inputs.
        :returns: Completion event and retained temporary tensors.
        """

        scatter_buffer = self._scatter_buffer
        scatter_base_address = self._scatter_base_address
        if scatter_buffer is None or scatter_base_address is None:
            raise RuntimeError("decode packed scatter buffer is not configured")
        if work.lease.length_bytes < work.layout.total_bytes:
            raise ValueError("packed lease is smaller than its canonical layout")
        lease_offset = work.lease.base_address - scatter_base_address
        if lease_offset < 0:
            raise ValueError("packed lease precedes decode staging buffer")
        if lease_offset + work.lease.length_bytes > scatter_buffer.numel():
            raise ValueError("packed lease exceeds decode staging buffer")

        retained: list[torch.Tensor] = []
        with torch.cuda.stream(self._scatter_stream):
            for writer_layout in work.layout.writers:
                for group in writer_layout.copy_groups:
                    active = work.destination_binding.require(group.component_id)
                    entry_ptrs = torch.tensor(
                        [
                            active.component.tensor_ptrs[entry_index]
                            for entry_index in group.destination_entry_indices
                        ],
                        dtype=torch.int64,
                        device=self._device,
                    )
                    page_indices = torch.from_numpy(
                        np.array(active.page_array, dtype=np.int32, copy=True)
                    ).to(self._device)
                    retained.extend((entry_ptrs, page_indices))
                    physical_tokens = group.page_count * active.component.page_size
                    bytes_per_entry = physical_tokens * group.copy_bytes_per_token
                    grid = (
                        len(group.destination_entry_indices),
                        triton.cdiv(bytes_per_entry, 256),
                    )
                    _scatter_packed_bytes_kernel[grid](
                        entry_ptrs,
                        page_indices,
                        scatter_buffer,
                        lease_offset + group.packed_offset,
                        physical_tokens,
                        group.destination_token_bytes,
                        group.destination_offset_bytes,
                        copy_bytes_per_token=group.copy_bytes_per_token,
                        page_size=active.component.page_size,
                        bytes_per_entry=bytes_per_entry,
                        block_size=256,
                    )
            event = torch.cuda.Event()
            event.record(self._scatter_stream)
        return PackedScatterSubmission(event=event, resources=tuple(retained))

    def synchronize_scatter(self) -> None:
        """Wait until every submitted scatter on the dedicated stream is terminal."""

        self._scatter_stream.synchronize()
