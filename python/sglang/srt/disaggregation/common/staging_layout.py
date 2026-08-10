import dataclasses
import hashlib
import json

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import compute_tensor_parallel_shard

DEFAULT_STAGING_ALIGNMENT_BYTES: int = 256


@dataclasses.dataclass(frozen=True)
class StagingComponentId:
    """Stable identity for one independently indexed KV component.

    :ivar state_index: Position in ``KVArgs.state_types``, or ``None`` for main KV.
    :ivar state_type: State kind, or ``None`` for main KV.
    """

    state_index: int | None
    state_type: StateType | None


@dataclasses.dataclass(frozen=True)
class StagingComponentSpan:
    """Logical and physical page span transferred for one component.

    :ivar component_id: Component owning the span.
    :ivar source_index_offset: First source page-array entry used by the span.
    :ivar destination_index_offset: First destination page-array entry used by
        the span.
    :ivar logical_token_count: Tokens visible to attention.
    :ivar physical_token_count: Page-rounded tokens copied by the transport.
    """

    component_id: StagingComponentId
    source_index_offset: int
    destination_index_offset: int
    logical_token_count: int
    physical_token_count: int


@dataclasses.dataclass(frozen=True)
class StagingComponentGeometry:
    """Registered tensor geometry for one page-indexed component.

    :ivar component_id: Component described by the geometry.
    :ivar item_lens: Bytes occupied by one page in each registered tensor.
    :ivar layer_ids: Global layer identifier for each registered tensor.
    :ivar page_size: Complete token rows in one page.
    """

    component_id: StagingComponentId
    item_lens: tuple[int, ...]
    layer_ids: tuple[int, ...]
    page_size: int

    def __post_init__(self) -> None:
        """Own immutable copies of all registration sequences."""

        object.__setattr__(self, "item_lens", tuple(self.item_lens))
        object.__setattr__(self, "layer_ids", tuple(self.layer_ids))


@dataclasses.dataclass(frozen=True, order=True)
class StagingWriterId:
    """Identity of one independently writing prefill rank.

    :ivar transfer_source_rank: Global rank used by the transfer backend.
    :ivar source_attn_tp_rank: Attention tensor-parallel rank selecting the
        source byte shard.
    :ivar source_pp_rank: Pipeline-parallel rank.
    :ivar source_cp_rank: Context-parallel rank.
    """

    transfer_source_rank: int
    source_attn_tp_rank: int
    source_pp_rank: int
    source_cp_rank: int


@dataclasses.dataclass(frozen=True)
class StagingCopyGroup:
    """Entries sharing one gather or scatter kernel geometry.

    :ivar component_id: Component copied by the group.
    :ivar source_entry_indices: Registered source-tensor indices.
    :ivar destination_entry_indices: Corresponding destination-tensor indices.
    :ivar packed_offset: Absolute byte offset within the decode lease.
    :ivar page_count: Number of complete physical pages copied per entry.
    :ivar source_token_bytes: Bytes per token in one source entry.
    :ivar destination_token_bytes: Bytes per token in one destination entry.
    :ivar source_offset_bytes: TP slice offset within a source token.
    :ivar destination_offset_bytes: TP slice offset within a destination token.
    :ivar copy_bytes_per_token: Bytes copied per entry and token.
    :ivar length_bytes: Packed bytes occupied by the complete group.
    """

    component_id: StagingComponentId
    source_entry_indices: tuple[int, ...]
    destination_entry_indices: tuple[int, ...]
    packed_offset: int
    page_count: int
    source_token_bytes: int
    destination_token_bytes: int
    source_offset_bytes: int
    destination_offset_bytes: int
    copy_bytes_per_token: int
    length_bytes: int

    def __post_init__(self) -> None:
        """Own immutable copies of all registration-index sequences."""

        object.__setattr__(
            self, "source_entry_indices", tuple(self.source_entry_indices)
        )
        object.__setattr__(
            self, "destination_entry_indices", tuple(self.destination_entry_indices)
        )


@dataclasses.dataclass(frozen=True)
class StagingWriterLayout:
    """Packed lease projection owned by one prefill writer.

    :ivar writer_id: Writer owning the projection.
    :ivar lease_offset: First byte of the writer projection.
    :ivar length_bytes: Page-aligned projection length.
    :ivar copy_groups: Gather and scatter groups in packed order.
    """

    writer_id: StagingWriterId
    lease_offset: int
    length_bytes: int
    copy_groups: tuple[StagingCopyGroup, ...]

    def __post_init__(self) -> None:
        """Own an immutable copy of the writer's copy groups."""

        object.__setattr__(self, "copy_groups", tuple(self.copy_groups))


@dataclasses.dataclass(frozen=True)
class StagingChunkLayout:
    """Immutable component-aware staging plan for one transfer chunk.

    :ivar chunk_id: Request-local monotonically increasing chunk identifier.
    :ivar is_last: Whether this is the request's final transfer chunk.
    :ivar component_spans: Component-local page spans.
    :ivar source_components: Immutable source geometry for every active
        component.
    :ivar destination_components: Immutable destination geometry for every
        active component.
    :ivar writers: Canonically ordered writer projections.
    :ivar total_bytes: Complete decode lease size.
    :ivar digest: SHA-256 digest of the immutable plan.
    """

    chunk_id: int
    is_last: bool
    component_spans: tuple[StagingComponentSpan, ...]
    source_components: tuple[StagingComponentGeometry, ...]
    destination_components: tuple[StagingComponentGeometry, ...]
    writers: tuple[StagingWriterLayout, ...]
    total_bytes: int
    digest: bytes

    def __post_init__(self) -> None:
        """Own immutable copies of every digest-bearing layout sequence."""

        object.__setattr__(self, "component_spans", tuple(self.component_spans))
        object.__setattr__(self, "source_components", tuple(self.source_components))
        object.__setattr__(
            self, "destination_components", tuple(self.destination_components)
        )
        object.__setattr__(self, "writers", tuple(self.writers))
        object.__setattr__(self, "digest", bytes(self.digest))


@dataclasses.dataclass(frozen=True)
class _EntryPair:
    """Paired source and destination registration entries."""

    source_index: int
    destination_index: int
    layer_id: int
    occurrence: int


def _align_up(value: int, alignment: int) -> int:
    """Round a non-negative value up to an alignment.

    :param value: Value to align.
    :param alignment: Positive byte alignment.
    :returns: Aligned value.
    :raises ValueError: If the value is negative or alignment is not positive.
    """

    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _component_sort_key(component_id: StagingComponentId) -> tuple[int, int, str]:
    """Return a deterministic main-KV-first component key.

    :param component_id: Component to order.
    :returns: Sort key.
    """

    if component_id.state_index is None:
        return (0, -1, "")
    state_type = component_id.state_type
    if state_type is None:
        raise ValueError("state component must declare state_type")
    return (1, component_id.state_index, state_type.value)


def _validate_component_id(component_id: StagingComponentId) -> None:
    """Validate main-KV and auxiliary-state identity.

    :param component_id: Component identity to validate.
    :raises ValueError: If the identity mixes main and state fields.
    """

    if component_id.state_index is None:
        if component_id.state_type is not None:
            raise ValueError("main KV cannot declare a state_type")
        return
    if component_id.state_index < 0:
        raise ValueError(
            f"state_index must be non-negative, got {component_id.state_index}"
        )
    if component_id.state_type is None:
        raise ValueError("state component must declare state_type")


def validate_staging_component_geometry(
    geometry: StagingComponentGeometry, label: str
) -> None:
    """Validate a registered component geometry.

    :param geometry: Geometry to validate.
    :param label: Reader-facing source or destination label.
    :raises ValueError: If registration metadata is incomplete or inconsistent.
    """

    _validate_component_id(geometry.component_id)
    if geometry.page_size <= 0:
        raise ValueError(
            f"{label} page_size must be positive, got {geometry.page_size}"
        )
    if len(geometry.item_lens) == 0:
        raise ValueError(f"{label} component must contain at least one entry")
    if len(geometry.item_lens) != len(geometry.layer_ids):
        raise ValueError(
            f"{label} item-length/layer-id count mismatch: "
            f"{len(geometry.item_lens)} and {len(geometry.layer_ids)}"
        )
    for layer_id in geometry.layer_ids:
        if layer_id < 0:
            raise ValueError(
                f"{label} layer identifiers must be non-negative, got {layer_id}"
            )
    for item_len in geometry.item_lens:
        if item_len <= 0:
            raise ValueError(f"{label} item lengths must be positive, got {item_len}")
        if item_len % geometry.page_size != 0:
            raise ValueError(
                f"{label} item length {item_len} is not divisible by page size "
                f"{geometry.page_size}"
            )


def _validate_span(
    span: StagingComponentSpan,
    page_size: int,
) -> None:
    """Validate component-local logical and physical token counts.

    :param span: Span to validate.
    :param page_size: Component page size.
    :raises ValueError: If indices or page-rounded capacity are invalid.
    """

    _validate_component_id(span.component_id)
    if span.source_index_offset < 0 or span.destination_index_offset < 0:
        raise ValueError("component page-array offsets must be non-negative")
    if span.logical_token_count <= 0 or span.physical_token_count <= 0:
        raise ValueError("included component token counts must be positive")
    if span.logical_token_count > span.physical_token_count:
        raise ValueError(
            "logical token count cannot exceed physical token count: "
            f"{span.logical_token_count} > {span.physical_token_count}"
        )
    expected_physical_tokens = _align_up(span.logical_token_count, page_size)
    if span.physical_token_count != expected_physical_tokens:
        raise ValueError(
            "physical token count must be the exact page-rounded logical count: "
            f"expected {expected_physical_tokens}, got {span.physical_token_count}"
        )


def _index_geometries(
    geometries: tuple[StagingComponentGeometry, ...],
    label: str,
) -> dict[StagingComponentId, StagingComponentGeometry]:
    """Index unique component geometries.

    :param geometries: Geometries to index.
    :param label: Reader-facing source or destination label.
    :returns: Mapping keyed by component identity.
    :raises ValueError: If a component is duplicated.
    """

    result: dict[StagingComponentId, StagingComponentGeometry] = {}
    state_indices: set[int] = set()
    for geometry in geometries:
        validate_staging_component_geometry(geometry, label)
        if geometry.component_id in result:
            raise ValueError(f"{label} component geometry is duplicated")
        state_index = geometry.component_id.state_index
        if state_index is not None:
            if state_index in state_indices:
                raise ValueError(
                    f"{label} state_index {state_index} is assigned more than once"
                )
            state_indices.add(state_index)
        result[geometry.component_id] = geometry
    return result


def _entry_keys(layer_ids: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """Index repeated layer identifiers by occurrence.

    :param layer_ids: Registered global layer identifiers.
    :returns: Entry index keyed by ``(layer_id, occurrence)``.
    """

    occurrences: dict[int, int] = {}
    result: dict[tuple[int, int], int] = {}
    for entry_index, layer_id in enumerate(layer_ids):
        occurrence = occurrences.get(layer_id, 0)
        occurrences[layer_id] = occurrence + 1
        result[(layer_id, occurrence)] = entry_index
    return result


def _pair_entries(
    source: StagingComponentGeometry,
    destination: StagingComponentGeometry,
) -> tuple[_EntryPair, ...]:
    """Pair source and destination entries by layer and occurrence.

    :param source: Source registration geometry.
    :param destination: Destination registration geometry.
    :returns: Canonically ordered entry pairs.
    :raises ValueError: If registrations describe different tensor sets.
    """

    source_entries = _entry_keys(source.layer_ids)
    destination_entries = _entry_keys(destination.layer_ids)
    if source_entries.keys() != destination_entries.keys():
        raise ValueError(
            "source and destination registrations contain different layer entries"
        )
    pairs: list[_EntryPair] = []
    for (layer_id, occurrence), source_index in source_entries.items():
        pairs.append(
            _EntryPair(
                source_index=source_index,
                destination_index=destination_entries[(layer_id, occurrence)],
                layer_id=layer_id,
                occurrence=occurrence,
            )
        )
    return tuple(pairs)


def source_tp_ranks_for_destination(
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
) -> tuple[int, ...]:
    """Return source attention ranks connected to one destination rank.

    :param source_tp_size: Source attention TP width.
    :param destination_tp_size: Destination attention TP width.
    :param destination_tp_rank: Destination attention TP rank.
    :returns: Connected source attention ranks.
    :raises ValueError: If TP widths or destination rank are invalid.
    """

    if source_tp_size <= 0 or destination_tp_size <= 0:
        raise ValueError("tensor-parallel sizes must be positive")
    if destination_tp_rank < 0 or destination_tp_rank >= destination_tp_size:
        raise ValueError(
            f"destination_tp_rank must be in [0, {destination_tp_size}), got "
            f"{destination_tp_rank}"
        )
    if source_tp_size >= destination_tp_size:
        if source_tp_size % destination_tp_size != 0:
            raise ValueError(
                "source tensor-parallel width must be divisible by destination width"
            )
        sources_per_destination = source_tp_size // destination_tp_size
        first_source_rank = destination_tp_rank * sources_per_destination
        return tuple(
            range(first_source_rank, first_source_rank + sources_per_destination)
        )
    if destination_tp_size % source_tp_size != 0:
        raise ValueError(
            "destination tensor-parallel width must be divisible by source width"
        )
    destinations_per_source = destination_tp_size // source_tp_size
    return (destination_tp_rank // destinations_per_source,)


def _source_replication_for_entry(
    source_token_bytes: int,
    destination_token_bytes: int,
    source_tp_size: int,
    destination_tp_size: int,
) -> int:
    """Derive physical source replication from one pinned entry pair.

    Packed decode currently admits TP1 and TP2 destinations whose Gemma KV
    components are not replicated. Any excess physical source bytes therefore
    represent equal consecutive source replicas, not additional logical data.

    :param source_token_bytes: Per-rank source bytes for one token.
    :param destination_token_bytes: Per-rank destination bytes for one token.
    :param source_tp_size: Physical source attention TP width.
    :param destination_tp_size: Physical destination attention TP width.
    :returns: Consecutive physical source ranks per logical shard.
    :raises ValueError: If the pinned geometries cannot describe exact source
        replication of the destination.
    """

    source_physical_bytes = source_token_bytes * source_tp_size
    destination_physical_bytes = destination_token_bytes * destination_tp_size
    if source_physical_bytes % destination_physical_bytes != 0:
        raise ValueError(
            "source and destination entry geometries do not form exact "
            "source replication"
        )
    replication = source_physical_bytes // destination_physical_bytes
    if replication <= 0 or source_tp_size % replication != 0:
        raise ValueError("source entry replication does not divide source TP width")
    return replication


def _canonical_writers(
    writers: tuple[StagingWriterId, ...],
    expected_source_ranks: tuple[int, ...],
) -> tuple[StagingWriterId, ...]:
    """Validate and order the complete writer set.

    :param writers: Submitted writer identities.
    :param expected_source_ranks: Source attention ranks required by routing.
    :returns: Canonically ordered writers.
    :raises ValueError: If a writer is duplicated, unsupported, or missing.
    """

    if len(writers) == 0:
        raise ValueError("at least one writer is required")
    if len(set(writers)) != len(writers):
        raise ValueError("writer identities must be unique")
    source_ranks: list[int] = []
    transfer_source_ranks: list[int] = []
    for writer in writers:
        if writer.transfer_source_rank < 0 or writer.source_attn_tp_rank < 0:
            raise ValueError("writer ranks must be non-negative")
        if writer.source_pp_rank != 0 or writer.source_cp_rank != 0:
            raise ValueError("initial staging contract requires PP=1 and CP=1")
        source_ranks.append(writer.source_attn_tp_rank)
        transfer_source_ranks.append(writer.transfer_source_rank)
    if len(set(source_ranks)) != len(source_ranks):
        raise ValueError("source attention TP ranks must be unique")
    if len(set(transfer_source_ranks)) != len(transfer_source_ranks):
        raise ValueError("transfer source ranks must be unique")
    if tuple(sorted(source_ranks)) != expected_source_ranks:
        raise ValueError(
            "writer set does not match bootstrap routing: "
            f"expected {expected_source_ranks}, got {tuple(sorted(source_ranks))}"
        )
    return tuple(sorted(writers, key=lambda writer: writer.source_attn_tp_rank))


def _layout_digest_payload(
    chunk_id: int,
    is_last: bool,
    spans: tuple[StagingComponentSpan, ...],
    writers: tuple[StagingWriterLayout, ...],
    total_bytes: int,
    source_components: dict[StagingComponentId, StagingComponentGeometry],
    destination_components: dict[StagingComponentId, StagingComponentGeometry],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    alignment_bytes: int,
) -> dict[str, object]:
    """Build the canonical JSON-compatible digest payload.

    :param chunk_id: Chunk identifier.
    :param is_last: Final-chunk marker.
    :param spans: Canonically ordered component spans.
    :param writers: Canonically ordered writer layouts.
    :param total_bytes: Complete decode lease size.
    :param source_components: Source geometries keyed by component.
    :param destination_components: Destination geometries keyed by component.
    :param source_tp_size: Source attention TP width.
    :param destination_tp_size: Destination attention TP width.
    :param destination_tp_rank: Destination attention TP rank.
    :param alignment_bytes: Packed-region byte alignment.
    :returns: JSON-compatible plan payload.
    """

    geometry_values: list[dict[str, object]] = []
    for component_id in sorted(source_components, key=_component_sort_key):
        source_geometry = source_components[component_id]
        destination_geometry = destination_components[component_id]
        geometry_values.append(
            {
                "component": {
                    "state_index": component_id.state_index,
                    "state_type": (
                        component_id.state_type.value
                        if component_id.state_type is not None
                        else None
                    ),
                },
                "destination": {
                    "item_lens": list(destination_geometry.item_lens),
                    "layer_ids": list(destination_geometry.layer_ids),
                    "page_size": destination_geometry.page_size,
                },
                "source": {
                    "item_lens": list(source_geometry.item_lens),
                    "layer_ids": list(source_geometry.layer_ids),
                    "page_size": source_geometry.page_size,
                },
            }
        )

    span_values: list[dict[str, object]] = []
    for span in spans:
        span_values.append(
            {
                "component": {
                    "state_index": span.component_id.state_index,
                    "state_type": (
                        span.component_id.state_type.value
                        if span.component_id.state_type is not None
                        else None
                    ),
                },
                "destination_index_offset": span.destination_index_offset,
                "logical_token_count": span.logical_token_count,
                "physical_token_count": span.physical_token_count,
                "source_index_offset": span.source_index_offset,
            }
        )

    writer_values: list[dict[str, object]] = []
    for writer_layout in writers:
        group_values: list[dict[str, object]] = []
        for group in writer_layout.copy_groups:
            group_values.append(
                {
                    "component": {
                        "state_index": group.component_id.state_index,
                        "state_type": (
                            group.component_id.state_type.value
                            if group.component_id.state_type is not None
                            else None
                        ),
                    },
                    "copy_bytes_per_token": group.copy_bytes_per_token,
                    "destination_entry_indices": list(group.destination_entry_indices),
                    "destination_offset_bytes": group.destination_offset_bytes,
                    "destination_token_bytes": group.destination_token_bytes,
                    "length_bytes": group.length_bytes,
                    "packed_offset": group.packed_offset,
                    "page_count": group.page_count,
                    "source_entry_indices": list(group.source_entry_indices),
                    "source_offset_bytes": group.source_offset_bytes,
                    "source_token_bytes": group.source_token_bytes,
                }
            )
        writer_values.append(
            {
                "copy_groups": group_values,
                "lease_offset": writer_layout.lease_offset,
                "length_bytes": writer_layout.length_bytes,
                "writer": {
                    "source_attn_tp_rank": (
                        writer_layout.writer_id.source_attn_tp_rank
                    ),
                    "source_cp_rank": writer_layout.writer_id.source_cp_rank,
                    "source_pp_rank": writer_layout.writer_id.source_pp_rank,
                    "transfer_source_rank": (
                        writer_layout.writer_id.transfer_source_rank
                    ),
                },
            }
        )
    return {
        "alignment_bytes": alignment_bytes,
        "chunk_id": chunk_id,
        "component_geometries": geometry_values,
        "component_spans": span_values,
        "destination_tp_rank": destination_tp_rank,
        "destination_tp_size": destination_tp_size,
        "is_last": is_last,
        "source_tp_size": source_tp_size,
        "total_bytes": total_bytes,
        "writers": writer_values,
    }


def build_staging_chunk_layout(
    *,
    chunk_id: int,
    is_last: bool,
    spans: tuple[StagingComponentSpan, ...],
    source_components: tuple[StagingComponentGeometry, ...],
    destination_components: tuple[StagingComponentGeometry, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    writers: tuple[StagingWriterId, ...],
    alignment_bytes: int = DEFAULT_STAGING_ALIGNMENT_BYTES,
) -> StagingChunkLayout:
    """Build a deterministic component-aware packed staging plan.

    :param chunk_id: Request-local monotonically increasing chunk identifier.
    :param is_last: Whether the chunk completes the request.
    :param spans: Exact component-local logical and physical page spans.
    :param source_components: Source registration geometries.
    :param destination_components: Destination registration geometries.
    :param source_tp_size: Source attention TP width.
    :param destination_tp_size: Destination attention TP width.
    :param destination_tp_rank: Destination attention TP rank.
    :param writers: Complete writer set connected by bootstrap routing.
    :param alignment_bytes: Byte alignment for packed groups and writers.
    :returns: Immutable staging layout and its canonical digest.
    :raises ValueError: If geometry, routing, or component coverage is invalid.
    """

    if chunk_id < 0:
        raise ValueError(f"chunk_id must be non-negative, got {chunk_id}")
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")

    source_by_component = _index_geometries(source_components, "source")
    destination_by_component = _index_geometries(destination_components, "destination")
    if source_by_component.keys() != destination_by_component.keys():
        raise ValueError(
            "source and destination registrations contain different components"
        )
    if len(spans) == 0:
        raise ValueError("a staging chunk must contain at least one component span")
    span_by_component: dict[StagingComponentId, StagingComponentSpan] = {}
    for span in spans:
        if span.component_id in span_by_component:
            raise ValueError("component spans must be unique")
        if span.component_id not in source_by_component:
            raise ValueError("component span has no registered geometry")
        source_geometry = source_by_component[span.component_id]
        destination_geometry = destination_by_component[span.component_id]
        if source_geometry.page_size != destination_geometry.page_size:
            raise ValueError("source and destination component page sizes must match")
        _validate_span(span, source_geometry.page_size)
        span_by_component[span.component_id] = span

    ordered_spans = tuple(
        span_by_component[component_id]
        for component_id in sorted(span_by_component, key=_component_sort_key)
    )
    active_source_components = {
        span.component_id: source_by_component[span.component_id]
        for span in ordered_spans
    }
    active_destination_components = {
        span.component_id: destination_by_component[span.component_id]
        for span in ordered_spans
    }
    expected_source_ranks = source_tp_ranks_for_destination(
        source_tp_size,
        destination_tp_size,
        destination_tp_rank,
    )
    ordered_writers = _canonical_writers(writers, expected_source_ranks)

    entry_pairs: dict[StagingComponentId, tuple[_EntryPair, ...]] = {}
    for span in ordered_spans:
        source_geometry = source_by_component[span.component_id]
        destination_geometry = destination_by_component[span.component_id]
        entry_pairs[span.component_id] = _pair_entries(
            source_geometry,
            destination_geometry,
        )

    next_offset = 0
    writer_layouts: list[StagingWriterLayout] = []
    destination_ranges: dict[tuple[StagingComponentId, int], list[tuple[int, int]]] = {}
    for writer in ordered_writers:
        writer_offset = _align_up(next_offset, alignment_bytes)
        group_offset = writer_offset
        groups: list[StagingCopyGroup] = []
        for span in ordered_spans:
            source_geometry = source_by_component[span.component_id]
            destination_geometry = destination_by_component[span.component_id]
            page_count = span.physical_token_count // source_geometry.page_size
            grouped_pairs: dict[
                tuple[int, int, int, int, int],
                list[_EntryPair],
            ] = {}
            for pair in entry_pairs[span.component_id]:
                source_token_bytes = (
                    source_geometry.item_lens[pair.source_index]
                    // source_geometry.page_size
                )
                destination_token_bytes = (
                    destination_geometry.item_lens[pair.destination_index]
                    // destination_geometry.page_size
                )
                source_replication = _source_replication_for_entry(
                    source_token_bytes,
                    destination_token_bytes,
                    source_tp_size,
                    destination_tp_size,
                )
                if writer.source_attn_tp_rank % source_replication != 0:
                    continue
                logical_source_tp_size = source_tp_size // source_replication
                shard = compute_tensor_parallel_shard(
                    source_token_bytes=source_token_bytes,
                    destination_token_bytes=destination_token_bytes,
                    source_parallel_size=logical_source_tp_size,
                    destination_parallel_size=destination_tp_size,
                    source_rank=(
                        writer.source_attn_tp_rank // source_replication
                    ),
                    destination_rank=destination_tp_rank,
                )
                geometry_key = (
                    source_token_bytes,
                    destination_token_bytes,
                    shard.source_offset_bytes,
                    shard.destination_offset_bytes,
                    shard.length_bytes,
                )
                grouped_pairs.setdefault(geometry_key, []).append(pair)
                destination_ranges.setdefault(
                    (span.component_id, pair.destination_index), []
                ).append(
                    (
                        shard.destination_offset_bytes,
                        shard.destination_offset_bytes + shard.length_bytes,
                    )
                )

            for geometry_key in sorted(grouped_pairs):
                (
                    source_token_bytes,
                    destination_token_bytes,
                    source_offset_bytes,
                    destination_offset_bytes,
                    copy_bytes_per_token,
                ) = geometry_key
                pairs = grouped_pairs[geometry_key]
                group_offset = _align_up(group_offset, alignment_bytes)
                length_bytes = (
                    len(pairs) * span.physical_token_count * copy_bytes_per_token
                )
                groups.append(
                    StagingCopyGroup(
                        component_id=span.component_id,
                        source_entry_indices=tuple(pair.source_index for pair in pairs),
                        destination_entry_indices=tuple(
                            pair.destination_index for pair in pairs
                        ),
                        packed_offset=group_offset,
                        page_count=page_count,
                        source_token_bytes=source_token_bytes,
                        destination_token_bytes=destination_token_bytes,
                        source_offset_bytes=source_offset_bytes,
                        destination_offset_bytes=destination_offset_bytes,
                        copy_bytes_per_token=copy_bytes_per_token,
                        length_bytes=length_bytes,
                    )
                )
                group_offset += length_bytes

        writer_length = _align_up(group_offset - writer_offset, alignment_bytes)
        if writer_length == 0:
            writer_length = alignment_bytes
        writer_layouts.append(
            StagingWriterLayout(
                writer_id=writer,
                lease_offset=writer_offset,
                length_bytes=writer_length,
                copy_groups=tuple(groups),
            )
        )
        next_offset = writer_offset + writer_length

    for component_id, destination_geometry in active_destination_components.items():
        for destination_index, destination_item_len in enumerate(
            destination_geometry.item_lens
        ):
            destination_token_bytes = (
                destination_item_len // destination_geometry.page_size
            )
            ranges = sorted(destination_ranges[(component_id, destination_index)])
            expected_start = 0
            for range_start, range_end in ranges:
                if range_start != expected_start:
                    raise ValueError(
                        "writer byte ranges do not exactly cover a destination "
                        f"token: expected offset {expected_start}, got {range_start}"
                    )
                expected_start = range_end
            if expected_start != destination_token_bytes:
                raise ValueError(
                    "writer byte ranges do not exactly cover a destination token: "
                    f"expected {destination_token_bytes} bytes, got {expected_start}"
                )

    total_bytes = _align_up(next_offset, alignment_bytes)
    immutable_writers = tuple(writer_layouts)
    digest_payload = _layout_digest_payload(
        chunk_id,
        is_last,
        ordered_spans,
        immutable_writers,
        total_bytes,
        active_source_components,
        active_destination_components,
        source_tp_size,
        destination_tp_size,
        destination_tp_rank,
        alignment_bytes,
    )
    encoded_payload = json.dumps(
        digest_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return StagingChunkLayout(
        chunk_id=chunk_id,
        is_last=is_last,
        component_spans=ordered_spans,
        source_components=tuple(
            active_source_components[span.component_id] for span in ordered_spans
        ),
        destination_components=tuple(
            active_destination_components[span.component_id] for span in ordered_spans
        ),
        writers=immutable_writers,
        total_bytes=total_bytes,
        digest=hashlib.sha256(encoded_payload).digest(),
    )
