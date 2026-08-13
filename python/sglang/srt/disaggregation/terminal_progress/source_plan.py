import dataclasses

import msgspec
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_DIGEST_BYTES,
    PackedRequestKey,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)

PACKED_TERMINAL_SOURCE_PLAN_VERSION: int = 1
MAX_PACKED_TERMINAL_SOURCE_PLAN_BYTES: int = 64 * 1024


class PackedTerminalSourcePlanError(ValueError):
    """Invalid or unsupported packed terminal source plan."""


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceIdentityPlan:
    """Complete cross-rank identity required before source submission.

    :ivar local_binding: Exact source-rank lifecycle binding.
    :ivar source_bindings: Canonically ordered complete source TP manifest.
    :ivar publication_identity: Exactly-once gateway publication identity.
    :ivar request_ready_issuer: Request coordinator authenticated on control.
    :ivar publisher_issuer: Canonical source-rank publisher identity.
    """

    local_binding: TerminalRequestBinding
    source_bindings: tuple[TerminalRequestBinding, ...]
    publication_identity: TerminalPublicationIdentity
    request_ready_issuer: TerminalProcessIdentity
    publisher_issuer: TerminalProcessIdentity

    def __post_init__(self) -> None:
        """Validate the complete immutable source identity graph."""

        if type(self.local_binding) is not TerminalRequestBinding:
            raise TypeError("local_binding must be TerminalRequestBinding")
        if type(self.source_bindings) is not tuple or len(self.source_bindings) == 0:
            raise ValueError("source_bindings must be a non-empty tuple")
        if any(
            type(binding) is not TerminalRequestBinding
            for binding in self.source_bindings
        ):
            raise TypeError("source_bindings must contain TerminalRequestBinding")
        if type(self.publication_identity) is not TerminalPublicationIdentity:
            raise TypeError("publication_identity must be TerminalPublicationIdentity")
        if type(self.request_ready_issuer) is not TerminalProcessIdentity:
            raise TypeError("request_ready_issuer must be TerminalProcessIdentity")
        if type(self.publisher_issuer) is not TerminalProcessIdentity:
            raise TypeError("publisher_issuer must be TerminalProcessIdentity")

        key = self.local_binding.request_key
        local_owner = self.local_binding.owner
        if local_owner.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("local binding must belong to a source owner")
        source_tp_size = local_owner.tp_size
        if len(self.source_bindings) != source_tp_size:
            raise ValueError("source binding count differs from source TP size")
        expected_ranks = tuple(range(source_tp_size))
        observed_ranks: list[int] = []
        for binding in self.source_bindings:
            owner = binding.owner
            if binding.request_key != key:
                raise ValueError("source bindings span request generations")
            if owner.role is not TerminalOwnerRole.SOURCE:
                raise ValueError("source bindings contain a decode owner")
            if owner.tp_size != source_tp_size:
                raise ValueError("source bindings disagree on TP size")
            if (
                binding.rank_manifest_digest
                != self.local_binding.rank_manifest_digest
                or binding.allocation_digest != self.local_binding.allocation_digest
            ):
                raise ValueError("source bindings disagree on request allocation")
            observed_ranks.append(owner.tp_rank)
        if tuple(observed_ranks) != expected_ranks:
            raise ValueError("source bindings must use canonical TP-rank order")
        if self.local_binding not in self.source_bindings:
            raise ValueError("source manifest omits the local binding")
        canonical = self.source_bindings[0].owner
        if self.publisher_issuer != canonical:
            raise ValueError("publisher issuer must be canonical source rank zero")
        if self.publication_identity.request_key != key:
            raise ValueError("publication identity belongs to another request")
        if (
            self.publication_identity.publisher_process_generation
            != canonical.process_generation
        ):
            raise ValueError("publication identity belongs to another publisher")
        if self.request_ready_issuer.role is not TerminalOwnerRole.DECODE:
            raise ValueError("request-ready issuer must belong to decode")
        if self.request_ready_issuer.tp_rank != 0:
            raise ValueError("request-ready issuer must be decoder rank zero")

    @property
    def request_key(self) -> PackedRequestKey:
        """Return the shared packed request identity.

        :returns: Request key carried by every binding.
        """

        return self.local_binding.request_key

    @property
    def trusted_issuers(self) -> tuple[TerminalProcessIdentity, ...]:
        """Return the canonical native trusted-issuer set.

        :returns: Coordinator followed by publisher, without duplicates.
        """

        if self.request_ready_issuer == self.publisher_issuer:
            return (self.request_ready_issuer,)
        return (self.request_ready_issuer, self.publisher_issuer)


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceWriter:
    """Explicit transport-writer to terminal-owner association.

    :ivar writer_id: Exact packed transport writer identity.
    :ivar process_identity: Exact source owner for that writer and TP rank.
    """

    writer_id: StagingWriterId
    process_identity: TerminalProcessIdentity

    def __post_init__(self) -> None:
        """Validate one source writer association."""

        if type(self.writer_id) is not StagingWriterId:
            raise TypeError("writer_id must be StagingWriterId")
        if type(self.process_identity) is not TerminalProcessIdentity:
            raise TypeError("process_identity must be TerminalProcessIdentity")
        if self.process_identity.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("terminal source writer must belong to source")
        if self.writer_id.source_attn_tp_rank != self.process_identity.tp_rank:
            raise ValueError("writer TP rank differs from its terminal owner")
        if self.writer_id.source_pp_rank != 0 or self.writer_id.source_cp_rank != 0:
            raise ValueError("terminal source writers require PP0 and CP0")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourcePlan:
    """Decoder-authored identity plan required before source submission.

    :ivar request_key: Exact packed request generation.
    :ivar writers: Canonically ordered complete source writer-owner manifest.
    :ivar rank_manifest_digest: Exact participating-writer manifest digest.
    :ivar allocation_digest: Exact destination allocation identity.
    :ivar publication_identity: Exactly-once canonical source publication.
    :ivar request_ready_issuer: Destination coordinator which can mint readiness.
    """

    request_key: PackedRequestKey
    writers: tuple[PackedTerminalSourceWriter, ...]
    rank_manifest_digest: bytes
    allocation_digest: bytes
    publication_identity: TerminalPublicationIdentity
    request_ready_issuer: TerminalProcessIdentity

    def __post_init__(self) -> None:
        """Validate one complete request identity graph."""

        if type(self.request_key) is not PackedRequestKey:
            raise TypeError("request_key must be PackedRequestKey")
        if type(self.writers) is not tuple or len(self.writers) == 0:
            raise ValueError("writers must be a non-empty tuple")
        if any(type(writer) is not PackedTerminalSourceWriter for writer in self.writers):
            raise TypeError("writers must contain PackedTerminalSourceWriter")
        for label, value in (
            ("rank_manifest_digest", self.rank_manifest_digest),
            ("allocation_digest", self.allocation_digest),
        ):
            if type(value) is not bytes:
                raise TypeError(f"{label} must be bytes")
            if len(value) != PACKED_REQUEST_DIGEST_BYTES:
                raise ValueError(
                    f"{label} must contain {PACKED_REQUEST_DIGEST_BYTES} bytes"
                )
        if type(self.publication_identity) is not TerminalPublicationIdentity:
            raise TypeError("publication_identity must be TerminalPublicationIdentity")
        if type(self.request_ready_issuer) is not TerminalProcessIdentity:
            raise TypeError("request_ready_issuer must be TerminalProcessIdentity")
        if self.publication_identity.request_key != self.request_key:
            raise ValueError("publication identity belongs to another request")
        if self.request_ready_issuer.role is not TerminalOwnerRole.DECODE:
            raise ValueError("request-ready issuer must belong to decode")
        if self.request_ready_issuer.tp_rank != 0:
            raise ValueError("request-ready issuer must be decoder rank zero")

        source_tp_size = len(self.writers)
        process_ranks: list[int] = []
        writer_ids: list[StagingWriterId] = []
        process_generations: list[bytes] = []
        for writer in self.writers:
            identity = writer.process_identity
            if identity.tp_size != source_tp_size:
                raise ValueError("source writer identity disagrees on TP size")
            process_ranks.append(identity.tp_rank)
            writer_ids.append(writer.writer_id)
            process_generations.append(identity.process_generation)
        if tuple(process_ranks) != tuple(range(source_tp_size)):
            raise ValueError("source writers must use canonical TP-rank order")
        if len(set(writer_ids)) != source_tp_size:
            raise ValueError("source writer identities must be unique")
        if len(set(process_generations)) != source_tp_size:
            raise ValueError("source process generations must be unique")
        publisher = self.writers[0].process_identity
        if (
            self.publication_identity.publisher_process_generation
            != publisher.process_generation
        ):
            raise ValueError("publication identity belongs to another source process")

    @property
    def source_bindings(self) -> tuple[TerminalRequestBinding, ...]:
        """Build the canonical source-rank binding manifest.

        :returns: One exact binding per source writer in TP-rank order.
        """

        return tuple(
            TerminalRequestBinding(
                request_key=self.request_key,
                owner=writer.process_identity,
                rank_manifest_digest=self.rank_manifest_digest,
                allocation_digest=self.allocation_digest,
            )
            for writer in self.writers
        )

    def identity_for_writer(
        self,
        writer_id: StagingWriterId,
    ) -> PackedTerminalSourceIdentityPlan:
        """Project this wire plan to one rank-local source identity plan.

        :param writer_id: Exact local packed writer identity.
        :returns: Complete source identity graph with that writer selected.
        """

        if type(writer_id) is not StagingWriterId:
            raise TypeError("writer_id must be StagingWriterId")
        matching = tuple(writer for writer in self.writers if writer.writer_id == writer_id)
        if len(matching) != 1:
            raise PackedTerminalSourcePlanError(
                "terminal source plan does not contain exactly one local writer"
            )
        bindings = self.source_bindings
        local_rank = matching[0].process_identity.tp_rank
        return PackedTerminalSourceIdentityPlan(
            local_binding=bindings[local_rank],
            source_bindings=bindings,
            publication_identity=self.publication_identity,
            request_ready_issuer=self.request_ready_issuer,
            publisher_issuer=self.writers[0].process_identity,
        )


class _WireRequestKey(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict wire request identity."""

    room_id: int
    request_generation: bytes


class _WireProcessIdentity(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict wire process identity."""

    process_generation: bytes
    role: str
    tp_rank: int
    tp_size: int


class _WireWriterId(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict wire transport writer identity."""

    transfer_source_rank: int
    source_attn_tp_rank: int
    source_pp_rank: int
    source_cp_rank: int


class _WireSourceWriter(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict wire writer-owner association."""

    writer_id: _WireWriterId
    process_identity: _WireProcessIdentity


class _WirePublicationIdentity(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Strict wire publication identity."""

    request_key: _WireRequestKey
    publisher_process_generation: bytes
    publication_generation: bytes


class _WireTerminalSourcePlan(
    msgspec.Struct,
    tag="terminal_source_plan",
    tag_field="kind",
    frozen=True,
    forbid_unknown_fields=True,
):
    """Versioned strict terminal source plan envelope."""

    version: int
    request_key: _WireRequestKey
    writers: tuple[_WireSourceWriter, ...]
    rank_manifest_digest: bytes
    allocation_digest: bytes
    publication_identity: _WirePublicationIdentity
    request_ready_issuer: _WireProcessIdentity


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(_WireTerminalSourcePlan, strict=True)


def _encode_request_key(key: PackedRequestKey) -> _WireRequestKey:
    """Encode one request key.

    :param key: Domain request identity.
    :returns: Strict wire identity.
    """

    return _WireRequestKey(
        room_id=key.room_id,
        request_generation=key.request_generation,
    )


def _decode_request_key(key: _WireRequestKey) -> PackedRequestKey:
    """Decode one request key.

    :param key: Strict wire identity.
    :returns: Validated domain request identity.
    """

    return PackedRequestKey(
        room_id=key.room_id,
        request_generation=key.request_generation,
    )


def _encode_process(identity: TerminalProcessIdentity) -> _WireProcessIdentity:
    """Encode one process identity.

    :param identity: Validated domain process identity.
    :returns: Strict wire identity.
    """

    return _WireProcessIdentity(
        process_generation=identity.process_generation,
        role=identity.role.value,
        tp_rank=identity.tp_rank,
        tp_size=identity.tp_size,
    )


def _decode_process(identity: _WireProcessIdentity) -> TerminalProcessIdentity:
    """Decode one process identity.

    :param identity: Strict wire identity.
    :returns: Validated domain process identity.
    """

    return TerminalProcessIdentity(
        process_generation=identity.process_generation,
        role=TerminalOwnerRole(identity.role),
        tp_rank=identity.tp_rank,
        tp_size=identity.tp_size,
    )


def _encode_writer_id(writer_id: StagingWriterId) -> _WireWriterId:
    """Encode one packed writer identity.

    :param writer_id: Domain writer identity.
    :returns: Strict wire identity.
    """

    return _WireWriterId(
        transfer_source_rank=writer_id.transfer_source_rank,
        source_attn_tp_rank=writer_id.source_attn_tp_rank,
        source_pp_rank=writer_id.source_pp_rank,
        source_cp_rank=writer_id.source_cp_rank,
    )


def _decode_writer_id(writer_id: _WireWriterId) -> StagingWriterId:
    """Decode one packed writer identity.

    :param writer_id: Strict wire identity.
    :returns: Domain writer identity.
    """

    return StagingWriterId(
        transfer_source_rank=writer_id.transfer_source_rank,
        source_attn_tp_rank=writer_id.source_attn_tp_rank,
        source_pp_rank=writer_id.source_pp_rank,
        source_cp_rank=writer_id.source_cp_rank,
    )


def encode_packed_terminal_source_plan(plan: PackedTerminalSourcePlan) -> bytes:
    """Encode one complete strict terminal source plan.

    :param plan: Decoder-authored request identity graph.
    :returns: Bounded versioned msgpack payload.
    """

    if type(plan) is not PackedTerminalSourcePlan:
        raise TypeError("plan must be PackedTerminalSourcePlan")
    publication = plan.publication_identity
    wire = _WireTerminalSourcePlan(
        version=PACKED_TERMINAL_SOURCE_PLAN_VERSION,
        request_key=_encode_request_key(plan.request_key),
        writers=tuple(
            _WireSourceWriter(
                writer_id=_encode_writer_id(writer.writer_id),
                process_identity=_encode_process(writer.process_identity),
            )
            for writer in plan.writers
        ),
        rank_manifest_digest=plan.rank_manifest_digest,
        allocation_digest=plan.allocation_digest,
        publication_identity=_WirePublicationIdentity(
            request_key=_encode_request_key(publication.request_key),
            publisher_process_generation=(
                publication.publisher_process_generation
            ),
            publication_generation=publication.publication_generation,
        ),
        request_ready_issuer=_encode_process(plan.request_ready_issuer),
    )
    payload = _ENCODER.encode(wire)
    if len(payload) > MAX_PACKED_TERMINAL_SOURCE_PLAN_BYTES:
        raise PackedTerminalSourcePlanError(
            "terminal source plan exceeds its bounded wire size"
        )
    return payload


def decode_packed_terminal_source_plan(payload: bytes) -> PackedTerminalSourcePlan:
    """Decode one strict terminal source plan.

    :param payload: Complete untrusted versioned payload.
    :returns: Validated immutable source identity graph.
    :raises PackedTerminalSourcePlanError: If framing, schema, or identity fails.
    """

    if type(payload) is not bytes:
        raise TypeError("terminal source plan payload must be bytes")
    if len(payload) == 0:
        raise PackedTerminalSourcePlanError("terminal source plan must not be empty")
    if len(payload) > MAX_PACKED_TERMINAL_SOURCE_PLAN_BYTES:
        raise PackedTerminalSourcePlanError(
            "terminal source plan exceeds its bounded wire size"
        )
    try:
        wire = _DECODER.decode(payload)
        if wire.version != PACKED_TERMINAL_SOURCE_PLAN_VERSION:
            raise PackedTerminalSourcePlanError(
                f"unsupported terminal source plan version {wire.version}"
            )
        publication = TerminalPublicationIdentity(
            request_key=_decode_request_key(wire.publication_identity.request_key),
            publisher_process_generation=(
                wire.publication_identity.publisher_process_generation
            ),
            publication_generation=wire.publication_identity.publication_generation,
        )
        return PackedTerminalSourcePlan(
            request_key=_decode_request_key(wire.request_key),
            writers=tuple(
                PackedTerminalSourceWriter(
                    writer_id=_decode_writer_id(writer.writer_id),
                    process_identity=_decode_process(writer.process_identity),
                )
                for writer in wire.writers
            ),
            rank_manifest_digest=wire.rank_manifest_digest,
            allocation_digest=wire.allocation_digest,
            publication_identity=publication,
            request_ready_issuer=_decode_process(wire.request_ready_issuer),
        )
    except PackedTerminalSourcePlanError:
        raise
    except (msgspec.DecodeError, TypeError, ValueError) as error:
        raise PackedTerminalSourcePlanError(
            f"invalid terminal source plan: {error}"
        ) from error
