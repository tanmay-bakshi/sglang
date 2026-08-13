import msgspec

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedLayoutSpec,
    PackedPrepare,
    PackedReady,
    PackedRequestKey,
    PackedRequestTeardown,
    PackedRequestTeardownAck,
    PackedTerminalReceipt,
    PackedTopology,
    PackedTransportPath,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityAction,
    PackedWriterVisibilityEvidence,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
)

PACKED_WIRE_VERSION: int = 7
MAX_PACKED_WIRE_BYTES: int = 1024 * 1024


class PackedWireError(ValueError):
    """Invalid or unsupported packed staging wire payload."""


class _WireChunkKey(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`PackedChunkKey`."""

    room_id: int
    chunk_id: int
    request_generation: bytes


class _WireRequestKey(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`PackedRequestKey`."""

    room_id: int
    request_generation: bytes


class _WireComponentId(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`StagingComponentId`."""

    state_index: int | None
    state_type: str | None


class _WireComponentSpan(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`StagingComponentSpan`."""

    component_id: _WireComponentId
    source_index_offset: int
    destination_index_offset: int
    logical_token_count: int
    physical_token_count: int


class _WireComponentGeometry(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`StagingComponentGeometry`."""

    component_id: _WireComponentId
    item_lens: tuple[int, ...]
    layer_ids: tuple[int, ...]
    page_size: int


class _WireWriterId(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`StagingWriterId`."""

    transfer_source_rank: int
    source_attn_tp_rank: int
    source_pp_rank: int
    source_cp_rank: int


class _WireAuxiliaryDestinationSegment(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of one auxiliary destination segment."""

    address: int
    item_length: int


class _WireAuxiliaryPlanFields(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of one decoder-authored auxiliary plan."""

    key: _WireRequestKey
    request_slot_generation: int
    metadata_buffer_index: int
    metadata_slot_generation: bytes
    destination_segments: tuple[_WireAuxiliaryDestinationSegment, ...]
    canonical_writer_id: _WireWriterId
    destination_process_generation: bytes
    native_route_digest: bytes
    runtime_cohort_digest: bytes


class _WireTopology(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`PackedTopology`."""

    source_tp_size: int
    destination_tp_size: int
    destination_tp_rank: int
    alignment_bytes: int


class _WireLayoutSpec(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of :class:`PackedLayoutSpec`."""

    chunk_id: int
    is_last: bool
    spans: tuple[_WireComponentSpan, ...]
    source_components: tuple[_WireComponentGeometry, ...]
    destination_components: tuple[_WireComponentGeometry, ...]
    writers: tuple[_WireWriterId, ...]
    topology: _WireTopology


class _WirePrepare(
    msgspec.Struct,
    tag="prepare",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned PREPARE envelope."""

    version: int
    key: _WireChunkKey
    writer_id: _WireWriterId
    spec: _WireLayoutSpec
    digest: bytes


class _WireReady(
    msgspec.Struct,
    tag="ready",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned READY envelope."""

    version: int
    key: _WireChunkKey
    writer_id: _WireWriterId
    digest: bytes
    visibility_policy_digest: bytes
    lease_id: int
    lease_base_address: int
    projection_offset: int
    projection_length: int


class _WireWriterVisibilityEvidence(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Wire representation of source-side visibility evidence."""

    policy_digest: bytes
    transport_path: str
    lane_identifier: str
    completion_mechanism: str
    writer_action: str
    native_handle_generation: int | None
    native_descriptor_digest: bytes | None
    native_evidence_digest: bytes | None


class _WireWriterOutcome(
    msgspec.Struct,
    tag="writer_outcome",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned terminal writer-outcome envelope."""

    version: int
    key: _WireChunkKey
    writer_id: _WireWriterId
    digest: bytes
    lease_id: int
    status: str
    visibility: _WireWriterVisibilityEvidence | None
    reason: str | None


class _WireAuxiliaryPlan(
    msgspec.Struct,
    tag="auxiliary_plan",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned decoder-authored auxiliary-plan envelope."""

    version: int
    plan: _WireAuxiliaryPlanFields


class _WireAuxiliaryOutcome(
    msgspec.Struct,
    tag="auxiliary_outcome",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned terminal auxiliary-outcome envelope."""

    version: int
    plan: _WireAuxiliaryPlanFields
    writer_id: _WireWriterId
    native_dram_handle_generation: int
    descriptor_digest: bytes
    evidence_digest: bytes


class _WireRequestTeardown(
    msgspec.Struct,
    tag="request_teardown",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned request teardown envelope."""

    version: int
    key: _WireRequestKey
    writer_id: _WireWriterId
    request_slot_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    teardown_generation: bytes
    auxiliary_handle_generation: int | None


class _WireRequestTeardownAck(
    msgspec.Struct,
    tag="request_teardown_ack",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned request teardown acknowledgement envelope."""

    version: int
    key: _WireRequestKey
    writer_id: _WireWriterId
    request_slot_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    teardown_generation: bytes
    auxiliary_handle_generation: int | None


class _WireTerminalReceipt(
    msgspec.Struct,
    tag="terminal_receipt",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned terminal-owner receipt envelope."""

    version: int
    key: _WireRequestKey
    receipt_payload: bytes


PackedWireMessage = (
    PackedAuxiliaryPlan
    | PackedAuxiliaryOutcome
    | PackedPrepare
    | PackedReady
    | PackedWriterOutcome
    | PackedRequestTeardown
    | PackedRequestTeardownAck
    | PackedTerminalReceipt
)
_WireMessage = (
    _WireAuxiliaryPlan
    | _WireAuxiliaryOutcome
    | _WirePrepare
    | _WireReady
    | _WireWriterOutcome
    | _WireRequestTeardown
    | _WireRequestTeardownAck
    | _WireTerminalReceipt
)
_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(_WireMessage, strict=True)


def _encode_chunk_key(key: PackedChunkKey) -> _WireChunkKey:
    """Convert a domain chunk key into its wire shape.

    :param key: Domain chunk identity.
    :returns: Immutable wire identity.
    """

    return _WireChunkKey(
        room_id=key.room_id,
        chunk_id=key.chunk_id,
        request_generation=key.request_generation,
    )


def _decode_chunk_key(key: _WireChunkKey) -> PackedChunkKey:
    """Convert a wire chunk key into its validated domain shape.

    :param key: Wire chunk identity.
    :returns: Validated domain identity.
    """

    return PackedChunkKey(
        room_id=key.room_id,
        chunk_id=key.chunk_id,
        request_generation=key.request_generation,
    )


def _encode_request_key(key: PackedRequestKey) -> _WireRequestKey:
    """Convert a domain request key into its wire shape.

    :param key: Domain request identity.
    :returns: Immutable wire identity.
    """

    return _WireRequestKey(
        room_id=key.room_id,
        request_generation=key.request_generation,
    )


def _decode_request_key(key: _WireRequestKey) -> PackedRequestKey:
    """Convert a wire request key into its validated domain shape.

    :param key: Wire request identity.
    :returns: Validated domain identity.
    """

    return PackedRequestKey(
        room_id=key.room_id,
        request_generation=key.request_generation,
    )


def _encode_component_id(component_id: StagingComponentId) -> _WireComponentId:
    """Convert a domain component identity into its wire shape.

    :param component_id: Main-KV or auxiliary-state identity.
    :returns: Immutable wire identity.
    """

    state_type = component_id.state_type
    return _WireComponentId(
        state_index=component_id.state_index,
        state_type=state_type.value if state_type is not None else None,
    )


def _decode_component_id(component_id: _WireComponentId) -> StagingComponentId:
    """Convert a wire component identity into its validated domain shape.

    :param component_id: Wire component identity.
    :returns: Main-KV or auxiliary-state identity.
    """

    state_type = (
        StateType(component_id.state_type)
        if component_id.state_type is not None
        else None
    )
    return StagingComponentId(
        state_index=component_id.state_index,
        state_type=state_type,
    )


def _encode_span(span: StagingComponentSpan) -> _WireComponentSpan:
    """Convert a domain component span into its wire shape.

    :param span: Domain component span.
    :returns: Immutable wire span.
    """

    return _WireComponentSpan(
        component_id=_encode_component_id(span.component_id),
        source_index_offset=span.source_index_offset,
        destination_index_offset=span.destination_index_offset,
        logical_token_count=span.logical_token_count,
        physical_token_count=span.physical_token_count,
    )


def _decode_span(span: _WireComponentSpan) -> StagingComponentSpan:
    """Convert a wire component span into its domain shape.

    :param span: Wire component span.
    :returns: Domain component span.
    """

    return StagingComponentSpan(
        component_id=_decode_component_id(span.component_id),
        source_index_offset=span.source_index_offset,
        destination_index_offset=span.destination_index_offset,
        logical_token_count=span.logical_token_count,
        physical_token_count=span.physical_token_count,
    )


def _encode_geometry(
    geometry: StagingComponentGeometry,
) -> _WireComponentGeometry:
    """Convert a domain component geometry into its wire shape.

    :param geometry: Domain registration geometry.
    :returns: Immutable wire geometry.
    """

    return _WireComponentGeometry(
        component_id=_encode_component_id(geometry.component_id),
        item_lens=geometry.item_lens,
        layer_ids=geometry.layer_ids,
        page_size=geometry.page_size,
    )


def _decode_geometry(
    geometry: _WireComponentGeometry,
) -> StagingComponentGeometry:
    """Convert a wire component geometry into its domain shape.

    :param geometry: Wire registration geometry.
    :returns: Domain registration geometry.
    """

    return StagingComponentGeometry(
        component_id=_decode_component_id(geometry.component_id),
        item_lens=geometry.item_lens,
        layer_ids=geometry.layer_ids,
        page_size=geometry.page_size,
    )


def _encode_writer_id(writer_id: StagingWriterId) -> _WireWriterId:
    """Convert a domain writer identity into its wire shape.

    :param writer_id: Domain writer identity.
    :returns: Immutable wire identity.
    """

    return _WireWriterId(
        transfer_source_rank=writer_id.transfer_source_rank,
        source_attn_tp_rank=writer_id.source_attn_tp_rank,
        source_pp_rank=writer_id.source_pp_rank,
        source_cp_rank=writer_id.source_cp_rank,
    )


def _decode_writer_id(writer_id: _WireWriterId) -> StagingWriterId:
    """Convert a wire writer identity into its domain shape.

    :param writer_id: Wire writer identity.
    :returns: Domain writer identity.
    """

    return StagingWriterId(
        transfer_source_rank=writer_id.transfer_source_rank,
        source_attn_tp_rank=writer_id.source_attn_tp_rank,
        source_pp_rank=writer_id.source_pp_rank,
        source_cp_rank=writer_id.source_cp_rank,
    )


def _encode_auxiliary_plan(
    plan: PackedAuxiliaryPlan,
) -> _WireAuxiliaryPlanFields:
    """Convert an auxiliary plan into its exact wire shape.

    :param plan: Decoder-authored domain plan.
    :returns: Immutable wire plan fields.
    """

    return _WireAuxiliaryPlanFields(
        key=_encode_request_key(plan.key),
        request_slot_generation=plan.request_slot_generation,
        metadata_buffer_index=plan.metadata_buffer_index,
        metadata_slot_generation=plan.metadata_slot_generation,
        destination_segments=tuple(
            _WireAuxiliaryDestinationSegment(
                address=segment.address,
                item_length=segment.item_length,
            )
            for segment in plan.destination_segments
        ),
        canonical_writer_id=_encode_writer_id(plan.canonical_writer_id),
        destination_process_generation=plan.destination_process_generation,
        native_route_digest=plan.native_route_digest,
        runtime_cohort_digest=plan.runtime_cohort_digest,
    )


def _decode_auxiliary_plan(
    plan: _WireAuxiliaryPlanFields,
) -> PackedAuxiliaryPlan:
    """Convert untrusted auxiliary plan fields into the domain shape.

    :param plan: Untrusted wire plan fields.
    :returns: Validated decoder-authored plan.
    """

    return PackedAuxiliaryPlan(
        key=_decode_request_key(plan.key),
        request_slot_generation=plan.request_slot_generation,
        metadata_buffer_index=plan.metadata_buffer_index,
        metadata_slot_generation=plan.metadata_slot_generation,
        destination_segments=tuple(
            PackedAuxiliaryDestinationSegment(
                address=segment.address,
                item_length=segment.item_length,
            )
            for segment in plan.destination_segments
        ),
        canonical_writer_id=_decode_writer_id(plan.canonical_writer_id),
        destination_process_generation=plan.destination_process_generation,
        native_route_digest=plan.native_route_digest,
        runtime_cohort_digest=plan.runtime_cohort_digest,
    )


def _encode_visibility_evidence(
    evidence: PackedWriterVisibilityEvidence,
) -> _WireWriterVisibilityEvidence:
    """Convert source visibility evidence into its wire shape.

    :param evidence: Validated domain evidence.
    :returns: Immutable wire evidence.
    """

    return _WireWriterVisibilityEvidence(
        policy_digest=evidence.policy_digest,
        transport_path=evidence.transport_path.value,
        lane_identifier=evidence.lane_identifier,
        completion_mechanism=evidence.completion_mechanism.value,
        writer_action=evidence.writer_action.value,
        native_handle_generation=evidence.native_handle_generation,
        native_descriptor_digest=evidence.native_descriptor_digest,
        native_evidence_digest=evidence.native_evidence_digest,
    )


def _decode_visibility_evidence(
    evidence: _WireWriterVisibilityEvidence,
) -> PackedWriterVisibilityEvidence:
    """Convert wire visibility evidence into its validated domain shape.

    :param evidence: Untrusted wire evidence.
    :returns: Validated domain evidence.
    """

    return PackedWriterVisibilityEvidence(
        policy_digest=evidence.policy_digest,
        transport_path=PackedTransportPath(evidence.transport_path),
        lane_identifier=evidence.lane_identifier,
        completion_mechanism=PackedWriterCompletionMechanism(
            evidence.completion_mechanism
        ),
        writer_action=PackedWriterVisibilityAction(evidence.writer_action),
        native_handle_generation=evidence.native_handle_generation,
        native_descriptor_digest=evidence.native_descriptor_digest,
        native_evidence_digest=evidence.native_evidence_digest,
    )


def _encode_topology(topology: PackedTopology) -> _WireTopology:
    """Convert a domain topology into its wire shape.

    :param topology: Domain tensor-parallel topology.
    :returns: Immutable wire topology.
    """

    return _WireTopology(
        source_tp_size=topology.source_tp_size,
        destination_tp_size=topology.destination_tp_size,
        destination_tp_rank=topology.destination_tp_rank,
        alignment_bytes=topology.alignment_bytes,
    )


def _decode_topology(topology: _WireTopology) -> PackedTopology:
    """Convert a wire topology into its validated domain shape.

    :param topology: Wire tensor-parallel topology.
    :returns: Validated domain topology.
    """

    return PackedTopology(
        source_tp_size=topology.source_tp_size,
        destination_tp_size=topology.destination_tp_size,
        destination_tp_rank=topology.destination_tp_rank,
        alignment_bytes=topology.alignment_bytes,
    )


def _encode_layout_spec(spec: PackedLayoutSpec) -> _WireLayoutSpec:
    """Convert a domain layout spec into its wire shape.

    :param spec: Complete domain layout input.
    :returns: Immutable wire layout input.
    """

    return _WireLayoutSpec(
        chunk_id=spec.chunk_id,
        is_last=spec.is_last,
        spans=tuple(_encode_span(span) for span in spec.spans),
        source_components=tuple(
            _encode_geometry(geometry) for geometry in spec.source_components
        ),
        destination_components=tuple(
            _encode_geometry(geometry) for geometry in spec.destination_components
        ),
        writers=tuple(_encode_writer_id(writer_id) for writer_id in spec.writers),
        topology=_encode_topology(spec.topology),
    )


def _decode_layout_spec(spec: _WireLayoutSpec) -> PackedLayoutSpec:
    """Convert a wire layout spec into its validated domain shape.

    :param spec: Complete wire layout input.
    :returns: Immutable domain layout input.
    """

    return PackedLayoutSpec(
        chunk_id=spec.chunk_id,
        is_last=spec.is_last,
        spans=tuple(_decode_span(span) for span in spec.spans),
        source_components=tuple(
            _decode_geometry(geometry) for geometry in spec.source_components
        ),
        destination_components=tuple(
            _decode_geometry(geometry) for geometry in spec.destination_components
        ),
        writers=tuple(_decode_writer_id(writer_id) for writer_id in spec.writers),
        topology=_decode_topology(spec.topology),
    )


def encode_packed_message(message: PackedWireMessage) -> bytes:
    """Encode one versioned packed staging envelope.

    ZMQ supplies message framing, so the returned bytes contain exactly one
    deterministic msgpack object and no delimiter protocol.

    :param message: PREPARE, READY, or terminal writer-outcome domain payload.
    :returns: Versioned msgpack frame.
    :raises TypeError: If the message type is unsupported.
    :raises PackedWireError: If the encoded frame exceeds the protocol bound.
    """

    wire_message: _WireMessage
    if type(message) is PackedAuxiliaryPlan:
        wire_message = _WireAuxiliaryPlan(
            version=PACKED_WIRE_VERSION,
            plan=_encode_auxiliary_plan(message),
        )
    elif type(message) is PackedAuxiliaryOutcome:
        wire_message = _WireAuxiliaryOutcome(
            version=PACKED_WIRE_VERSION,
            plan=_encode_auxiliary_plan(message.plan),
            writer_id=_encode_writer_id(message.writer_id),
            native_dram_handle_generation=(message.native_dram_handle_generation),
            descriptor_digest=message.descriptor_digest,
            evidence_digest=message.evidence_digest,
        )
    elif type(message) is PackedPrepare:
        wire_message = _WirePrepare(
            version=PACKED_WIRE_VERSION,
            key=_encode_chunk_key(message.key),
            writer_id=_encode_writer_id(message.writer_id),
            spec=_encode_layout_spec(message.spec),
            digest=message.digest,
        )
    elif type(message) is PackedReady:
        wire_message = _WireReady(
            version=PACKED_WIRE_VERSION,
            key=_encode_chunk_key(message.key),
            writer_id=_encode_writer_id(message.writer_id),
            digest=message.digest,
            visibility_policy_digest=message.visibility_policy_digest,
            lease_id=message.lease_id,
            lease_base_address=message.lease_base_address,
            projection_offset=message.projection_offset,
            projection_length=message.projection_length,
        )
    elif type(message) is PackedWriterOutcome:
        wire_message = _WireWriterOutcome(
            version=PACKED_WIRE_VERSION,
            key=_encode_chunk_key(message.key),
            writer_id=_encode_writer_id(message.writer_id),
            digest=message.digest,
            lease_id=message.lease_id,
            status=message.status.value,
            visibility=(
                _encode_visibility_evidence(message.visibility)
                if message.visibility is not None
                else None
            ),
            reason=message.reason,
        )
    elif type(message) is PackedRequestTeardown:
        wire_message = _WireRequestTeardown(
            version=PACKED_WIRE_VERSION,
            key=_encode_request_key(message.key),
            writer_id=_encode_writer_id(message.writer_id),
            request_slot_generation=message.request_slot_generation,
            writer_manifest_digest=message.writer_manifest_digest,
            allocation_digest=message.allocation_digest,
            teardown_generation=message.teardown_generation,
            auxiliary_handle_generation=message.auxiliary_handle_generation,
        )
    elif type(message) is PackedRequestTeardownAck:
        wire_message = _WireRequestTeardownAck(
            version=PACKED_WIRE_VERSION,
            key=_encode_request_key(message.key),
            writer_id=_encode_writer_id(message.writer_id),
            request_slot_generation=message.request_slot_generation,
            writer_manifest_digest=message.writer_manifest_digest,
            allocation_digest=message.allocation_digest,
            teardown_generation=message.teardown_generation,
            auxiliary_handle_generation=message.auxiliary_handle_generation,
        )
    elif type(message) is PackedTerminalReceipt:
        wire_message = _WireTerminalReceipt(
            version=PACKED_WIRE_VERSION,
            key=_encode_request_key(message.key),
            receipt_payload=message.receipt_payload,
        )
    else:
        raise TypeError(f"unsupported packed wire message: {type(message)!r}")

    payload = _ENCODER.encode(wire_message)
    if len(payload) > MAX_PACKED_WIRE_BYTES:
        raise PackedWireError(
            f"packed wire payload exceeds {MAX_PACKED_WIRE_BYTES} bytes"
        )
    return payload


def decode_packed_message(payload: bytes) -> PackedWireMessage:
    """Decode and validate one versioned packed staging envelope.

    :param payload: Complete msgpack ZMQ frame.
    :returns: PREPARE, READY, or terminal writer-outcome domain payload.
    :raises PackedWireError: If framing, schema, version, or domain values fail.
    """

    if type(payload) is not bytes:
        raise PackedWireError(
            f"packed wire payload must be bytes, got {type(payload)!r}"
        )
    if len(payload) == 0:
        raise PackedWireError("packed wire payload must not be empty")
    if len(payload) > MAX_PACKED_WIRE_BYTES:
        raise PackedWireError(
            f"packed wire payload exceeds {MAX_PACKED_WIRE_BYTES} bytes"
        )
    try:
        wire_message = _DECODER.decode(payload)
        if wire_message.version != PACKED_WIRE_VERSION:
            raise PackedWireError(
                "unsupported packed wire version "
                f"{wire_message.version}; expected {PACKED_WIRE_VERSION}"
            )
        if type(wire_message) is _WireAuxiliaryPlan:
            return _decode_auxiliary_plan(wire_message.plan)
        if type(wire_message) is _WireAuxiliaryOutcome:
            return PackedAuxiliaryOutcome(
                plan=_decode_auxiliary_plan(wire_message.plan),
                writer_id=_decode_writer_id(wire_message.writer_id),
                native_dram_handle_generation=(
                    wire_message.native_dram_handle_generation
                ),
                descriptor_digest=wire_message.descriptor_digest,
                evidence_digest=wire_message.evidence_digest,
            )
        if type(wire_message) is _WirePrepare:
            return PackedPrepare(
                key=_decode_chunk_key(wire_message.key),
                writer_id=_decode_writer_id(wire_message.writer_id),
                spec=_decode_layout_spec(wire_message.spec),
                digest=wire_message.digest,
            )
        if type(wire_message) is _WireReady:
            return PackedReady(
                key=_decode_chunk_key(wire_message.key),
                writer_id=_decode_writer_id(wire_message.writer_id),
                digest=wire_message.digest,
                visibility_policy_digest=wire_message.visibility_policy_digest,
                lease_id=wire_message.lease_id,
                lease_base_address=wire_message.lease_base_address,
                projection_offset=wire_message.projection_offset,
                projection_length=wire_message.projection_length,
            )
        if type(wire_message) is _WireWriterOutcome:
            return PackedWriterOutcome(
                key=_decode_chunk_key(wire_message.key),
                writer_id=_decode_writer_id(wire_message.writer_id),
                digest=wire_message.digest,
                lease_id=wire_message.lease_id,
                status=PackedWriterOutcomeStatus(wire_message.status),
                visibility=(
                    _decode_visibility_evidence(wire_message.visibility)
                    if wire_message.visibility is not None
                    else None
                ),
                reason=wire_message.reason,
            )
        if type(wire_message) is _WireRequestTeardown:
            return PackedRequestTeardown(
                key=_decode_request_key(wire_message.key),
                writer_id=_decode_writer_id(wire_message.writer_id),
                request_slot_generation=wire_message.request_slot_generation,
                writer_manifest_digest=wire_message.writer_manifest_digest,
                allocation_digest=wire_message.allocation_digest,
                teardown_generation=wire_message.teardown_generation,
                auxiliary_handle_generation=wire_message.auxiliary_handle_generation,
            )
        if type(wire_message) is _WireRequestTeardownAck:
            return PackedRequestTeardownAck(
                key=_decode_request_key(wire_message.key),
                writer_id=_decode_writer_id(wire_message.writer_id),
                request_slot_generation=wire_message.request_slot_generation,
                writer_manifest_digest=wire_message.writer_manifest_digest,
                allocation_digest=wire_message.allocation_digest,
                teardown_generation=wire_message.teardown_generation,
                auxiliary_handle_generation=wire_message.auxiliary_handle_generation,
            )
        if type(wire_message) is _WireTerminalReceipt:
            return PackedTerminalReceipt(
                key=_decode_request_key(wire_message.key),
                receipt_payload=wire_message.receipt_payload,
            )
        raise PackedWireError(
            f"unsupported packed wire message: {type(wire_message)!r}"
        )
    except PackedWireError:
        raise
    except (msgspec.DecodeError, TypeError, ValueError) as error:
        raise PackedWireError(f"invalid packed wire payload: {error}") from error
