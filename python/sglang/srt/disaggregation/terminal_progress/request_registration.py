import dataclasses
import secrets
from collections.abc import Callable

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestPublication,
)
from sglang.srt.disaggregation.terminal_progress.coordinator import (
    TerminalRequestCoordinatorManifest,
)
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeSchedulerRegistration,
)
from sglang.srt.disaggregation.terminal_progress.decode_serving import (
    PackedTerminalDecodeServing,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TERMINAL_PUBLICATION_GENERATION_BYTES,
    TerminalOwnerRole,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourceIdentityPlan,
    PackedTerminalSourcePlan,
    PackedTerminalSourceWriter,
    encode_packed_terminal_source_plan,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupRankAdvertisement,
)


class PackedTerminalRequestRegistrationError(RuntimeError):
    """Invalid production request authority derived from sealed startup state."""


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeRequestAuthority:
    """Complete prepublication decode authority for one packed request.

    :ivar binding: Exact rank-local decode lifecycle identity.
    :ivar source_plan: Request-global source writer and publication authority.
    :ivar coordinator_manifest: Exact destination fan-in and owner fan-out.
    :ivar registration: Scheduler callbacks retaining the mutable request owner.
    :ivar encoded_source_plan: Canonical bytes carried in packed metadata.
    """

    binding: TerminalRequestBinding
    source_plan: PackedTerminalSourcePlan
    coordinator_manifest: TerminalRequestCoordinatorManifest
    registration: PackedTerminalDecodeSchedulerRegistration
    encoded_source_plan: bytes

    def __post_init__(self) -> None:
        """Validate one internally identical authority graph."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.source_plan) is not PackedTerminalSourcePlan:
            raise TypeError("source_plan must be PackedTerminalSourcePlan")
        if type(self.coordinator_manifest) is not TerminalRequestCoordinatorManifest:
            raise TypeError(
                "coordinator_manifest must be TerminalRequestCoordinatorManifest"
            )
        if type(self.registration) is not PackedTerminalDecodeSchedulerRegistration:
            raise TypeError(
                "registration must be PackedTerminalDecodeSchedulerRegistration"
            )
        if type(self.encoded_source_plan) is not bytes:
            raise TypeError("encoded_source_plan must be bytes")
        if self.registration.binding != self.binding:
            raise ValueError("scheduler registration carries another binding")
        if self.registration.source_plan != self.source_plan:
            raise ValueError("scheduler registration carries another source plan")
        if self.coordinator_manifest.request_key != self.binding.request_key:
            raise ValueError("coordinator manifest carries another request")
        if encode_packed_terminal_source_plan(self.source_plan) != (
            self.encoded_source_plan
        ):
            raise ValueError("encoded source plan is not canonical")


def _rank_population(
    binding: TerminalStartupRankBinding,
    role: TerminalOwnerRole,
    *,
    service_id: str | None = None,
) -> tuple[TerminalStartupRankAdvertisement, ...]:
    """Select one canonical role and service population.

    :param binding: Complete sealed startup authority.
    :param role: Source or decode rank population.
    :param service_id: Optional exact service restriction.
    :returns: Canonically ordered advertisements.
    """

    ranks = tuple(
        rank
        for rank in binding.matrix.ranks
        if rank.role is role and (service_id is None or rank.service_id == service_id)
    )
    if len(ranks) == 0:
        raise PackedTerminalRequestRegistrationError(
            "startup matrix has no matching terminal rank population"
        )
    service_ids = {rank.service_id for rank in ranks}
    if len(service_ids) != 1:
        raise PackedTerminalRequestRegistrationError(
            "terminal rank population spans multiple services"
        )
    expected_size = ranks[0].tensor_parallel_size
    if len(ranks) != expected_size or tuple(
        rank.tensor_parallel_rank for rank in ranks
    ) != tuple(range(expected_size)):
        raise PackedTerminalRequestRegistrationError(
            "terminal rank population is not a complete canonical TP group"
        )
    return ranks


def _publication_writers(
    publication: PackedRequestPublication,
) -> tuple[StagingWriterId, ...]:
    """Return the exact writer manifest shared by every request chunk.

    :param publication: Prepared decoder-authored publication.
    :returns: Canonical source writer IDs.
    """

    if type(publication) is not PackedRequestPublication:
        raise TypeError("publication must be PackedRequestPublication")
    if len(publication.chunk_specs) == 0:
        raise PackedTerminalRequestRegistrationError(
            "packed publication has no request chunks"
        )
    writers = publication.chunk_specs[0].writers
    if len(writers) == 0:
        raise PackedTerminalRequestRegistrationError(
            "packed publication has no source writers"
        )
    if any(spec.writers != writers for spec in publication.chunk_specs[1:]):
        raise PackedTerminalRequestRegistrationError(
            "packed request chunks disagree on source writers"
        )
    return writers


def _source_writers(
    binding: TerminalStartupRankBinding,
    publication: PackedRequestPublication,
) -> tuple[PackedTerminalSourceWriter, ...]:
    """Join exact transaction writers to sealed source process identities.

    :param binding: Decode rank's sealed startup authority.
    :param publication: Exact prepared transaction publication.
    :returns: Source writers in source TP-rank order.
    """

    source_ranks = _rank_population(binding, TerminalOwnerRole.SOURCE)
    writer_ids = _publication_writers(publication)
    if len(writer_ids) != len(source_ranks):
        raise PackedTerminalRequestRegistrationError(
            "transaction writer count differs from the sealed source TP group"
        )
    writers_by_rank: dict[int, StagingWriterId] = {}
    for writer_id in writer_ids:
        if writer_id.source_pp_rank != 0 or writer_id.source_cp_rank != 0:
            raise PackedTerminalRequestRegistrationError(
                "terminal source writers require PP0 and CP0"
            )
        rank = writer_id.source_attn_tp_rank
        if rank in writers_by_rank:
            raise PackedTerminalRequestRegistrationError(
                "transaction has duplicate source attention TP writers"
            )
        writers_by_rank[rank] = writer_id
    if tuple(sorted(writers_by_rank)) != tuple(range(len(source_ranks))):
        raise PackedTerminalRequestRegistrationError(
            "transaction writer ranks differ from the sealed source TP group"
        )
    return tuple(
        PackedTerminalSourceWriter(
            writer_id=writers_by_rank[rank.tensor_parallel_rank],
            process_identity=rank.terminal_identity,
        )
        for rank in source_ranks
    )


def _destination_bindings(
    binding: TerminalStartupRankBinding,
    local_binding: TerminalRequestBinding,
    supplied: tuple[TerminalRequestBinding, ...] | None,
) -> tuple[TerminalRequestBinding, ...]:
    """Resolve an honest request-global destination manifest.

    :param binding: Local decode startup authority.
    :param local_binding: Exact transaction-derived local binding.
    :param supplied: Cross-rank request bindings when decode TP exceeds one.
    :returns: Complete canonical destination binding population.
    """

    local = binding.advertisement
    decode_ranks = _rank_population(
        binding,
        TerminalOwnerRole.DECODE,
        service_id=local.service_id,
    )
    if supplied is None:
        if len(decode_ranks) != 1:
            raise PackedTerminalRequestRegistrationError(
                "decode TP greater than one requires an exact cross-rank request "
                "binding manifest"
            )
        return (local_binding,)
    if type(supplied) is not tuple or len(supplied) != len(decode_ranks):
        raise PackedTerminalRequestRegistrationError(
            "destination binding count differs from the sealed decode TP group"
        )
    if any(type(candidate) is not TerminalRequestBinding for candidate in supplied):
        raise TypeError("destination_bindings must contain TerminalRequestBinding")
    if supplied[local.tensor_parallel_rank] != local_binding:
        raise PackedTerminalRequestRegistrationError(
            "destination manifest carries another local transaction binding"
        )
    expected_owners = tuple(rank.terminal_identity for rank in decode_ranks)
    if tuple(candidate.owner for candidate in supplied) != expected_owners:
        raise PackedTerminalRequestRegistrationError(
            "destination manifest differs from the sealed decode TP group"
        )
    if any(
        candidate.request_key != local_binding.request_key for candidate in supplied
    ):
        raise PackedTerminalRequestRegistrationError(
            "destination manifest spans request generations"
        )
    return supplied


def build_packed_terminal_decode_request_authority(
    *,
    startup_binding: TerminalStartupRankBinding,
    transaction: PackedDecodeRequestTransaction,
    adopt_request: Callable[[object], None],
    finalize_request: Callable[[object], None],
    cancel_request: Callable[[object], None],
    quarantine_request: Callable[[object, str], None],
    destination_bindings: tuple[TerminalRequestBinding, ...] | None = None,
    publication_generation: bytes | None = None,
) -> PackedTerminalDecodeRequestAuthority:
    """Build one exact decode registration without publishing allocation state.

    :param startup_binding: Complete immutable local startup authority.
    :param transaction: Exact prepared packed transaction.
    :param adopt_request: Scheduler request-adoption callback.
    :param finalize_request: Scheduler post-adoption finalization callback.
    :param cancel_request: Safe unpublished cancellation callback.
    :param quarantine_request: Ambiguous scheduler retention callback.
    :param destination_bindings: Exact cross-rank decode bindings for TP greater
        than one. TP1 derives its sole binding locally.
    :param publication_generation: Optional pre-minted publication generation.
    :returns: Complete immutable authority ready for serving registration.
    """

    if type(startup_binding) is not TerminalStartupRankBinding:
        raise TypeError("startup_binding must be TerminalStartupRankBinding")
    if type(transaction) is not PackedDecodeRequestTransaction:
        raise TypeError("transaction must be PackedDecodeRequestTransaction")
    local = startup_binding.advertisement
    if local.role is not TerminalOwnerRole.DECODE:
        raise PackedTerminalRequestRegistrationError(
            "decode request authority requires a decode startup binding"
        )
    publication = transaction.prepared_publication()
    if publication.terminal_source_plan is not None:
        raise PackedTerminalRequestRegistrationError(
            "transaction already carries terminal source authority"
        )
    local_binding = TerminalRequestBinding(
        request_key=publication.key,
        owner=local.terminal_identity,
        rank_manifest_digest=publication.writer_manifest_digest,
        allocation_digest=publication.allocation_digest,
    )
    destinations = _destination_bindings(
        startup_binding,
        local_binding,
        destination_bindings,
    )
    request_ready_issuer = destinations[0].owner
    if request_ready_issuer.tp_rank != 0:
        raise PackedTerminalRequestRegistrationError(
            "request-ready authority must belong to decode rank zero"
        )
    source_writers = _source_writers(startup_binding, publication)
    generation = (
        secrets.token_bytes(TERMINAL_PUBLICATION_GENERATION_BYTES)
        if publication_generation is None
        else publication_generation
    )
    source_plan = PackedTerminalSourcePlan(
        request_key=publication.key,
        writers=source_writers,
        rank_manifest_digest=publication.writer_manifest_digest,
        allocation_digest=publication.allocation_digest,
        publication_identity=TerminalPublicationIdentity(
            request_key=publication.key,
            publisher_process_generation=(
                source_writers[0].process_identity.process_generation
            ),
            publication_generation=generation,
        ),
        request_ready_issuer=request_ready_issuer,
    )
    coordinator_manifest = TerminalRequestCoordinatorManifest(
        request_key=publication.key,
        destination_bindings=destinations,
        recipient_bindings=(*destinations, *source_plan.source_bindings),
    )
    registration = PackedTerminalDecodeSchedulerRegistration(
        binding=local_binding,
        source_plan=source_plan,
        transaction=transaction,
        request_owner=transaction.request_owner,
        adopt_request=adopt_request,
        finalize_request=finalize_request,
        cancel_request=cancel_request,
        quarantine_request=quarantine_request,
    )
    return PackedTerminalDecodeRequestAuthority(
        binding=local_binding,
        source_plan=source_plan,
        coordinator_manifest=coordinator_manifest,
        registration=registration,
        encoded_source_plan=encode_packed_terminal_source_plan(source_plan),
    )


def register_packed_terminal_decode_request(
    serving: PackedTerminalDecodeServing,
    authority: PackedTerminalDecodeRequestAuthority,
) -> None:
    """Register scheduler and native ownership before allocation publication.

    :param serving: Already-constructed process-lifetime decode serving owner.
    :param authority: Exact prepared request authority.
    """

    if type(serving) is not PackedTerminalDecodeServing:
        raise TypeError("serving must be PackedTerminalDecodeServing")
    if type(authority) is not PackedTerminalDecodeRequestAuthority:
        raise TypeError("authority must be PackedTerminalDecodeRequestAuthority")
    serving.register_request(
        authority.registration,
        authority.coordinator_manifest,
    )
    publication = authority.registration.transaction.prepared_publication()
    if publication.terminal_source_plan != authority.encoded_source_plan:
        raise RuntimeError("terminal actor bound another encoded source-plan authority")


def project_packed_terminal_source_authority(
    *,
    startup_binding: TerminalStartupRankBinding,
    source_plan: PackedTerminalSourcePlan,
    local_writer_id: StagingWriterId,
    destination_process_generation: bytes,
) -> PackedTerminalSourceIdentityPlan:
    """Validate wire authority against sealed membership and select local source.

    :param startup_binding: Exact source rank startup authority.
    :param source_plan: Decoder-authored and codec-validated source plan.
    :param local_writer_id: Exact local packed transport writer.
    :param destination_process_generation: Decoder generation from auxiliary
        transport metadata.
    :returns: Complete rank-local source identity plan.
    """

    if type(startup_binding) is not TerminalStartupRankBinding:
        raise TypeError("startup_binding must be TerminalStartupRankBinding")
    if type(source_plan) is not PackedTerminalSourcePlan:
        raise TypeError("source_plan must be PackedTerminalSourcePlan")
    if type(local_writer_id) is not StagingWriterId:
        raise TypeError("local_writer_id must be StagingWriterId")
    local = startup_binding.advertisement
    if local.role is not TerminalOwnerRole.SOURCE:
        raise PackedTerminalRequestRegistrationError(
            "source authority projection requires a source startup binding"
        )
    source_ranks = _rank_population(startup_binding, TerminalOwnerRole.SOURCE)
    expected_identities = tuple(rank.terminal_identity for rank in source_ranks)
    if tuple(writer.process_identity for writer in source_plan.writers) != (
        expected_identities
    ):
        raise PackedTerminalRequestRegistrationError(
            "source plan writer identities differ from the sealed source TP group"
        )
    destination = startup_binding.matrix.rank_for_process_generation(
        destination_process_generation
    )
    if (
        destination.role is not TerminalOwnerRole.DECODE
        or destination.tensor_parallel_rank != 0
        or source_plan.request_ready_issuer != destination.terminal_identity
    ):
        raise PackedTerminalRequestRegistrationError(
            "source plan request-ready issuer differs from the destination route"
        )
    identity_plan = source_plan.identity_for_writer(local_writer_id)
    if identity_plan.local_binding.owner != local.terminal_identity:
        raise PackedTerminalRequestRegistrationError(
            "source plan local writer belongs to another startup rank"
        )
    return identity_plan


def require_source_plan_request_key(
    source_plan: PackedTerminalSourcePlan,
    request_key: PackedRequestKey,
) -> None:
    """Require terminal and packed metadata to name one request generation.

    :param source_plan: Decoded terminal authority.
    :param request_key: Exact packed auxiliary request key.
    """

    if type(source_plan) is not PackedTerminalSourcePlan:
        raise TypeError("source_plan must be PackedTerminalSourcePlan")
    if type(request_key) is not PackedRequestKey:
        raise TypeError("request_key must be PackedRequestKey")
    if source_plan.request_key != request_key:
        raise PackedTerminalRequestRegistrationError(
            "terminal source plan belongs to another packed request generation"
        )
