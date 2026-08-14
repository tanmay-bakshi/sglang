import dataclasses
import hashlib
import uuid

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.nixl.source_publication_control import (
    TerminalSourcePublicationControl,
    TerminalSourcePublicationControlError,
    TerminalSourcePublicationDelivery,
    TerminalSourcePublicationRouteRoster,
    encode_terminal_source_publication_receipt,
)
from sglang.srt.disaggregation.nixl.startup_source_roster import (
    TerminalNixlSourceRoster,
    TerminalNixlSourceRoute,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    FrozenTerminalGatewayOutputProjection,
    FrozenTerminalGatewayPublication,
    TerminalGatewayPublicationFailure,
    TerminalGatewayPublicationSuccess,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceiptIssuer,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_DIGEST = bytes.fromhex("11" * 32)
_TRANSPORT_PROTOCOL = "nixl-peer-handle-v1"


@dataclasses.dataclass(frozen=True, slots=True)
class _Projection(FrozenTerminalGatewayOutputProjection):
    """Minimal immutable output projection for publication control tests.

    :ivar payload: Stable synthetic output bytes.
    """

    payload: bytes

    @property
    def digest(self) -> bytes:
        """Return the stable projection digest.

        :returns: SHA-256 of the synthetic output bytes.
        """

        return hashlib.sha256(self.payload).digest()


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    tp_rank: int,
    tp_size: int,
    generation: int,
    metadata: bytes,
) -> TerminalStartupRankAdvertisement:
    """Build one generation-bound startup row.

    :param service_id: Static service identifier.
    :param role: Source or decode owner role.
    :param tp_rank: Rank within its service.
    :param tp_size: Exact service TP width.
    :param generation: Nonzero process-generation UUID integer.
    :param metadata: Complete synthetic NIXL metadata.
    :returns: Exact startup advertisement.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        service_id=service_id,
        service_origin=(
            "http://127.0.0.1:32001"
            if role is TerminalOwnerRole.SOURCE
            else "http://127.0.0.1:32002"
        ),
        role=role,
        launch_instance_id=uuid.UUID(
            int=1 if role is TerminalOwnerRole.SOURCE else 2
        ).bytes,
        tensor_parallel_rank=tp_rank,
        tensor_parallel_size=tp_size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=f"{role.value}-agent-{tp_rank}",
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _matrix(source_tp_size: int) -> TerminalStartupCohortMatrix:
    """Build one TP2 or TP4 source service and TP1 decoder.

    :param source_tp_size: Exact source TP width.
    :returns: Complete canonical startup matrix.
    """

    sources = tuple(
        _rank(
            service_id="prefill-a",
            role=TerminalOwnerRole.SOURCE,
            tp_rank=rank,
            tp_size=source_tp_size,
            generation=101 + rank,
            metadata=f"source-metadata-{rank}".encode("ascii"),
        )
        for rank in range(source_tp_size)
    )
    decoder = _rank(
        service_id="decode-a",
        role=TerminalOwnerRole.DECODE,
        tp_rank=0,
        tp_size=1,
        generation=201,
        metadata=b"decode-metadata-0",
    )
    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        ranks=(*sources, decoder),
    )


def _binding(
    matrix: TerminalStartupCohortMatrix,
    source_rank: int,
) -> TerminalStartupRankBinding:
    """Build one source rank binding over the shared matrix.

    :param matrix: Complete startup matrix.
    :param source_rank: Local source TP rank.
    :returns: Exact startup binding and Python producer plan.
    """

    advertisement = matrix.rank("prefill-a", source_rank)
    return TerminalStartupRankBinding(
        advertisement=advertisement,
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id="prefill-a",
            local_tensor_parallel_rank=source_rank,
            first_producer_id=0,
        ),
    )


def _source_routes(
    matrix: TerminalStartupCohortMatrix,
) -> tuple[TerminalNixlSourceRoute, ...]:
    """Build the actual manager listener roster for every source rank.

    :param matrix: Complete startup matrix.
    :returns: Canonical generation-bound source routes.
    """

    source_ranks = tuple(
        rank for rank in matrix.ranks if rank.role is TerminalOwnerRole.SOURCE
    )
    return tuple(
        TerminalNixlSourceRoute(
            service_id=rank.service_id,
            tensor_parallel_rank=rank.tensor_parallel_rank,
            tensor_parallel_size=rank.tensor_parallel_size,
            process_generation=rank.process_generation,
            nixl_agent_name=rank.nixl_agent_name,
            nixl_agent_metadata=(
                f"source-metadata-{rank.tensor_parallel_rank}".encode("ascii")
            ),
            rank_ip="127.0.0.1",
            rank_port=33000 + rank.tensor_parallel_rank,
            attn_dp_rank=0,
            attn_cp_rank=0,
            attn_tp_rank=rank.tensor_parallel_rank,
            pp_rank=0,
            transfer_source_rank=rank.tensor_parallel_rank,
            transport_protocol=_TRANSPORT_PROTOCOL,
        )
        for rank in source_ranks
    )


def _nixl_roster(
    matrix: TerminalStartupCohortMatrix,
    requester: TerminalStartupRankAdvertisement,
    routes: tuple[TerminalNixlSourceRoute, ...] | None = None,
) -> TerminalNixlSourceRoster:
    """Build one requester-bound source listener response.

    :param matrix: Complete startup matrix.
    :param requester: Exact source rank fetching same-service routes.
    :param routes: Optional adversarial route population.
    :returns: Complete source roster.
    """

    return TerminalNixlSourceRoster(
        matrix_sha256=matrix.digest,
        requester_service_id=requester.service_id,
        requester_tensor_parallel_rank=requester.tensor_parallel_rank,
        requester_process_generation=requester.process_generation,
        routes=_source_routes(matrix) if routes is None else routes,
    )


def _route_roster(
    matrix: TerminalStartupCohortMatrix,
    source_rank: int,
) -> TerminalSourcePublicationRouteRoster:
    """Enroll one local rank against the actual manager route table.

    :param matrix: Complete startup matrix.
    :param source_rank: Local source TP rank.
    :returns: Complete immutable publication routes.
    """

    binding = _binding(matrix, source_rank)
    return TerminalSourcePublicationRouteRoster.from_startup_roster(
        binding,
        _nixl_roster(matrix, binding.advertisement),
        NetworkAddress("127.0.0.1", 33000 + source_rank),
    )


def _publication(
    roster: TerminalSourcePublicationRouteRoster,
) -> tuple[
    FrozenTerminalGatewayPublication,
    tuple[TerminalRequestBinding, ...],
]:
    """Build one complete request-global source publication.

    :param roster: Exact source route population.
    :returns: Publication and canonical source bindings.
    """

    key = PackedRequestKey(
        room_id=91,
        request_generation=bytes.fromhex("21" * 16),
    )
    bindings = tuple(
        TerminalRequestBinding(
            request_key=key,
            owner=identity,
            rank_manifest_digest=bytes.fromhex("31" * 32),
            allocation_digest=bytes.fromhex("41" * 32),
        )
        for identity in roster.identities
    )
    ready = TerminalReceiptIssuer().issue(
        binding=bindings[0],
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=100,
    )
    return (
        FrozenTerminalGatewayPublication(
            identity=TerminalPublicationIdentity(
                request_key=key,
                publisher_process_generation=roster.canonical_identity.process_generation,
                publication_generation=bytes.fromhex("51" * 16),
            ),
            canonical_binding=bindings[0],
            source_bindings=bindings,
            request_ready_receipt=ready,
            output_projection=_Projection(payload=b"output"),
            enqueued_ns=90,
        ),
        bindings,
    )


def _success(
    publication: FrozenTerminalGatewayPublication,
) -> TerminalGatewayPublicationSuccess:
    """Mint one canonical successful publication fan-out.

    :param publication: Exact immutable publication.
    :returns: Complete successful publisher result.
    """

    issuer = TerminalWireReceiptIssuer(publication.canonical_binding.owner)
    receipts = tuple(
        issuer.issue(
            binding=binding,
            kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=110,
        )
        for binding in publication.source_bindings
    )
    return TerminalGatewayPublicationSuccess(
        publication=publication,
        completed_ns=110,
        source_receipts=receipts,
    )


def _failure(
    publication: FrozenTerminalGatewayPublication,
) -> TerminalGatewayPublicationFailure:
    """Mint one canonical failed publication fan-out.

    :param publication: Exact immutable publication.
    :returns: Complete failed publisher result.
    """

    issuer = TerminalWireReceiptIssuer(publication.canonical_binding.owner)
    receipts = tuple(
        issuer.issue(
            binding=binding,
            kind=TerminalReceiptKind.FAILURE,
            outcome=TerminalReceiptOutcome.FAILURE,
            terminal_timestamp_ns=115,
        )
        for binding in publication.source_bindings
    )
    return TerminalGatewayPublicationFailure(
        publication=publication,
        failed_ns=115,
        source_receipts=receipts,
        reason="gateway write failed",
        formatted_traceback="synthetic gateway write traceback",
    )


@pytest.mark.parametrize("source_tp_size", (2, 4))
def test_tp_source_publisher_fans_out_directly_to_every_rank(
    source_tp_size: int,
) -> None:
    """TP2 and TP4 use actual rank listeners without relay or collective."""

    matrix = _matrix(source_tp_size)
    route_rosters = tuple(_route_roster(matrix, rank) for rank in range(source_tp_size))
    assert all(roster == route_rosters[0] for roster in route_rosters)
    controls: list[TerminalSourcePublicationControl] = []
    deliveries: list[list[TerminalSourcePublicationDelivery]] = [
        [] for _ in range(source_tp_size)
    ]
    endpoint_to_rank = {
        route.endpoint: route.identity.tp_rank for route in route_rosters[0].routes
    }

    def send_frames(endpoint: NetworkAddress, frames: tuple[bytes, ...]) -> None:
        """Deliver one fake point-to-point manager send.

        :param endpoint: Exact destination manager listener.
        :param frames: Closed source publication message.
        """

        assert controls[endpoint_to_rank[endpoint]].receive_frames(frames)

    for rank, roster in enumerate(route_rosters):
        control = TerminalSourcePublicationControl(
            roster,
            roster.identities[rank],
            send_frames,
        )
        control.bind_listener(deliveries[rank].append, lambda reason: None)
        controls.append(control)

    publication, bindings = _publication(route_rosters[0])
    for control, binding in zip(controls, bindings, strict=True):
        control.register_binding(binding)
    controls[0].publish_result(_success(publication))

    assert [len(rank_deliveries) for rank_deliveries in deliveries] == [
        1
    ] * source_tp_size
    for rank, rank_deliveries in enumerate(deliveries):
        delivery = rank_deliveries[0]
        assert delivery.wire_receipt.binding == bindings[rank]
        assert delivery.authenticated_issuer == route_rosters[0].canonical_identity
    for control, binding in zip(controls, bindings, strict=True):
        control.retire_binding(binding)
        control.close_clean()
        assert control.inventory().closed


def test_tp2_publication_failure_fans_out_directly_to_every_rank() -> None:
    """Functional gateway failure reaches every source owner directly."""

    matrix = _matrix(2)
    route_rosters = tuple(_route_roster(matrix, rank) for rank in range(2))
    controls: list[TerminalSourcePublicationControl] = []
    deliveries: list[list[TerminalSourcePublicationDelivery]] = [[], []]
    endpoint_to_rank = {
        route.endpoint: route.identity.tp_rank for route in route_rosters[0].routes
    }

    def send_frames(endpoint: NetworkAddress, frames: tuple[bytes, ...]) -> None:
        """Deliver one fake point-to-point manager send.

        :param endpoint: Exact destination manager listener.
        :param frames: Closed source publication message.
        """

        assert controls[endpoint_to_rank[endpoint]].receive_frames(frames)

    for rank, roster in enumerate(route_rosters):
        control = TerminalSourcePublicationControl(
            roster,
            roster.identities[rank],
            send_frames,
        )
        control.bind_listener(deliveries[rank].append, lambda reason: None)
        controls.append(control)

    publication, bindings = _publication(route_rosters[0])
    for control, binding in zip(controls, bindings, strict=True):
        control.register_binding(binding)
    controls[0].publish_result(_failure(publication))

    assert [len(rank_deliveries) for rank_deliveries in deliveries] == [1, 1]
    assert all(
        rank_deliveries[0].wire_receipt.kind is TerminalReceiptKind.FAILURE
        for rank_deliveries in deliveries
    )
    assert all(
        rank_deliveries[0].wire_receipt.outcome is TerminalReceiptOutcome.FAILURE
        for rank_deliveries in deliveries
    )


def test_route_enrollment_rejects_stale_missing_and_conflicting_membership() -> None:
    """No source endpoint can escape exact sealed matrix authority."""

    matrix = _matrix(2)
    binding = _binding(matrix, 1)
    routes = _source_routes(matrix)
    stale = dataclasses.replace(
        routes[1],
        process_generation=uuid.UUID(int=999).bytes,
    )
    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="sealed startup matrix",
    ):
        TerminalSourcePublicationRouteRoster.from_startup_roster(
            binding,
            _nixl_roster(matrix, binding.advertisement, (routes[0], stale)),
            NetworkAddress("127.0.0.1", 33001),
        )

    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="sealed startup matrix",
    ):
        TerminalSourcePublicationRouteRoster.from_startup_roster(
            binding,
            _nixl_roster(matrix, binding.advertisement, routes[:-1]),
            NetworkAddress("127.0.0.1", 33001),
        )

    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="conflicts with the local manager listener",
    ):
        TerminalSourcePublicationRouteRoster.from_startup_roster(
            binding,
            _nixl_roster(matrix, binding.advertisement),
            NetworkAddress("127.0.0.1", 33999),
        )


def test_route_enrollment_rejects_duplicate_or_missing_endpoint() -> None:
    """Endpoint population is complete, nonempty, and collision-free."""

    matrix = _matrix(2)
    requester = matrix.rank("prefill-a", 0)
    routes = _source_routes(matrix)
    duplicate = dataclasses.replace(
        routes[1],
        rank_ip=routes[0].rank_ip,
        rank_port=routes[0].rank_port,
    )
    with pytest.raises(ValueError, match="endpoints must be unique"):
        _nixl_roster(matrix, requester, (routes[0], duplicate))
    with pytest.raises(ValueError, match="rank_ip must be nonempty"):
        dataclasses.replace(routes[1], rank_ip="")


def test_stale_publisher_generation_enters_process_fatal_ownership() -> None:
    """Authentication failure closes route admission through its fatal owner."""

    matrix = _matrix(2)
    roster = _route_roster(matrix, 1)
    sent: list[tuple[bytes, ...]] = []
    canonical = TerminalSourcePublicationControl(
        _route_roster(matrix, 0),
        roster.canonical_identity,
        lambda endpoint, frames: sent.append(frames),
    )
    remote = TerminalSourcePublicationControl(
        roster,
        roster.identities[1],
        lambda endpoint, frames: None,
    )
    fatal_reasons: list[str] = []
    canonical.bind_listener(lambda delivery: None, lambda reason: None)
    remote.bind_listener(lambda delivery: None, fatal_reasons.append)
    publication, bindings = _publication(roster)
    canonical.register_binding(bindings[0])
    remote.register_binding(bindings[1])
    canonical.publish_result(_success(publication))
    assert len(sent) == 1

    stale_frames = list(sent[0])
    stale_frames[2] = uuid.UUID(int=999).bytes
    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="stale publisher generation",
    ):
        remote.receive_frames(tuple(stale_frames))
    assert len(fatal_reasons) == 1
    assert "stale publisher generation" in fatal_reasons[0]
    assert remote.inventory().fatal_reason == fatal_reasons[0]
    with pytest.raises(TerminalSourcePublicationControlError):
        remote.receive_frames(sent[0])
    assert len(fatal_reasons) == 1


def test_duplicate_coalesces_and_conflicting_receipt_fails_closed() -> None:
    """Byte-identical replay coalesces while another outcome is fatal."""

    matrix = _matrix(2)
    roster = _route_roster(matrix, 1)
    sent: list[tuple[bytes, ...]] = []
    canonical = TerminalSourcePublicationControl(
        _route_roster(matrix, 0),
        roster.canonical_identity,
        lambda endpoint, frames: sent.append(frames),
    )
    remote = TerminalSourcePublicationControl(
        roster,
        roster.identities[1],
        lambda endpoint, frames: None,
    )
    fatal_reasons: list[str] = []
    canonical.bind_listener(lambda delivery: None, lambda reason: None)
    remote.bind_listener(lambda delivery: None, fatal_reasons.append)
    publication, bindings = _publication(roster)
    canonical.register_binding(bindings[0])
    remote.register_binding(bindings[1])
    canonical.publish_result(_success(publication))
    assert len(sent) == 1
    assert remote.receive_frames(sent[0])
    assert not remote.receive_frames(sent[0])

    failure_issuer = TerminalWireReceiptIssuer(roster.canonical_identity)
    conflicting = failure_issuer.issue(
        binding=bindings[1],
        kind=TerminalReceiptKind.FAILURE,
        outcome=TerminalReceiptOutcome.FAILURE,
        terminal_timestamp_ns=120,
    )
    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="conflicting outcomes",
    ):
        remote.receive_frames(
            encode_terminal_source_publication_receipt(
                roster,
                conflicting.wire_receipt,
            )
        )
    assert remote.inventory().fatal_reason is not None
    assert fatal_reasons == [remote.inventory().fatal_reason]


def test_clean_teardown_requires_terminal_delivery_and_binding_retirement() -> None:
    """Route teardown cannot discard active or replay-bearing authority."""

    matrix = _matrix(2)
    roster = _route_roster(matrix, 1)
    control = TerminalSourcePublicationControl(
        roster,
        roster.identities[1],
        lambda endpoint, frames: None,
    )
    control.bind_listener(lambda delivery: None, lambda reason: None)
    _, bindings = _publication(roster)
    control.register_binding(bindings[1])
    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="retains request authority",
    ):
        control.close_clean()

    issuer = TerminalWireReceiptIssuer(roster.canonical_identity)
    issued = issuer.issue(
        binding=bindings[1],
        kind=TerminalReceiptKind.GATEWAY_PUBLISHED,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=110,
    )
    assert control.receive_frames(
        encode_terminal_source_publication_receipt(roster, issued.wire_receipt)
    )
    with pytest.raises(
        TerminalSourcePublicationControlError,
        match="retains request authority",
    ):
        control.close_clean()
    control.retire_binding(bindings[1])
    control.close_clean()
    assert control.inventory().active_binding_digests == ()
