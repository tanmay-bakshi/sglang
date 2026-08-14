import dataclasses
import hashlib
import threading

import numpy as np
import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedAuxiliaryDestinationSegment,
    PackedAuxiliaryPlan,
    PackedLayoutSpec,
    PackedRequestKey,
    PackedTopology,
)
from sglang.srt.disaggregation.common.packed_staging_wire import (
    encode_packed_message,
)
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.conn import TransferInfo
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestPublication,
    PackedRequestTransactionError,
    PackedRequestTransactionState,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.request_registration import (
    PackedTerminalDecodeRequestAuthority,
    PackedTerminalRequestRegistrationError,
    build_packed_terminal_decode_request_authority,
    project_packed_terminal_source_authority,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    encode_packed_terminal_source_plan,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortError,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _advertisement(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    rank: int,
    size: int,
    marker: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact sealed startup row.

    :param service_id: Static service identity.
    :param role: Source or decode role.
    :param rank: Rank within the service TP group.
    :param size: Service TP width.
    :param marker: Unique identity marker.
    :returns: Valid startup advertisement.
    """

    metadata = f"metadata-{marker}".encode("ascii")
    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=b"c" * 32,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{32000 + marker}",
        role=role,
        launch_instance_id=marker.to_bytes(16, "big"),
        tensor_parallel_rank=rank,
        tensor_parallel_size=size,
        process_generation=(marker + 100).to_bytes(16, "big"),
        nixl_agent_name=f"agent-{marker}",
        nixl_agent_metadata_sha256=hashlib.sha256(metadata).digest(),
    )


def _matrix(*, decode_tp_size: int = 1) -> TerminalStartupCohortMatrix:
    """Build one TP2 source and configurable decode startup matrix.

    :param decode_tp_size: Decode service TP width.
    :returns: Canonical immutable startup matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=b"c" * 32,
        ranks=(
            _advertisement(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                rank=0,
                size=2,
                marker=1,
            ),
            _advertisement(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                rank=1,
                size=2,
                marker=2,
            ),
            *tuple(
                _advertisement(
                    service_id="decode-a",
                    role=TerminalOwnerRole.DECODE,
                    rank=rank,
                    size=decode_tp_size,
                    marker=10 + rank,
                )
                for rank in range(decode_tp_size)
            ),
        ),
    )


def _binding(
    matrix: TerminalStartupCohortMatrix,
    *,
    service_id: str,
    rank: int,
) -> TerminalStartupRankBinding:
    """Build one immutable local binding from the shared matrix.

    :param matrix: Complete startup matrix.
    :param service_id: Exact local service.
    :param rank: Local TP rank.
    :returns: Rank-local startup authority.
    """

    advertisement = matrix.rank(service_id, rank)
    return TerminalStartupRankBinding(
        advertisement=advertisement,
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id=service_id,
            local_tensor_parallel_rank=rank,
            first_producer_id=0,
        ),
    )


def _writer(rank: int) -> StagingWriterId:
    """Build one transaction-authored source writer.

    :param rank: Source attention TP rank.
    :returns: Exact transport writer identity.
    """

    return StagingWriterId(
        transfer_source_rank=6 + rank,
        source_attn_tp_rank=rank,
        source_pp_rank=0,
        source_cp_rank=0,
    )


def _publication(key: PackedRequestKey) -> PackedRequestPublication:
    """Build one immutable prepared transaction publication.

    :param key: Exact request generation.
    :returns: Decoder-authored metadata with TP2 writers.
    """

    writers = (_writer(0), _writer(1))
    return PackedRequestPublication(
        key=key,
        request_slot_generation=7,
        writer_manifest_digest=b"w" * 32,
        allocation_digest=b"a" * 32,
        auxiliary_plan=PackedAuxiliaryPlan(
            key=key,
            request_slot_generation=7,
            metadata_buffer_index=3,
            metadata_slot_generation=b"m" * 16,
            destination_segments=(
                PackedAuxiliaryDestinationSegment(
                    address=0x2000,
                    item_length=64,
                ),
            ),
            canonical_writer_id=writers[0],
            destination_process_generation=(110).to_bytes(16, "big"),
            native_route_digest=b"n" * 32,
            runtime_cohort_digest=b"r" * 32,
        ),
        chunk_specs=(
            PackedLayoutSpec(
                chunk_id=0,
                is_last=True,
                spans=(),
                source_components=(),
                destination_components=(),
                writers=writers,
                topology=PackedTopology(
                    source_tp_size=2,
                    destination_tp_size=1,
                    destination_tp_rank=0,
                ),
            ),
        ),
    )


def _transaction(owner: object) -> PackedDecodeRequestTransaction:
    """Build a type-exact prepared transaction shell.

    :param owner: Exact retained scheduler request.
    :returns: Minimal transaction exercising immutable publication APIs.
    """

    key = PackedRequestKey(room_id=41, request_generation=b"g" * 16)
    transaction = object.__new__(PackedDecodeRequestTransaction)
    transaction._request_key = key
    transaction._request_owner = owner
    transaction._scheduler_thread_id = threading.get_ident()
    transaction._lock = threading.RLock()
    transaction._state = PackedRequestTransactionState.PREPARED
    transaction._publication = _publication(key)
    transaction._terminal_binding_digest = None
    transaction._protocol = object()
    transaction._chunks = ()
    transaction._teardown_acks = set()
    transaction._auxiliary_outcome = None
    transaction._auxiliary_plan = transaction._publication.auxiliary_plan
    return transaction


def _authority(
    *,
    matrix: TerminalStartupCohortMatrix | None = None,
) -> tuple[PackedDecodeRequestTransaction, PackedTerminalDecodeRequestAuthority]:
    """Build one production-shaped TP2-to-TP1 request authority.

    :param matrix: Optional startup matrix override.
    :returns: Exact transaction and request authority.
    """

    startup_matrix = _matrix() if matrix is None else matrix
    transaction = _transaction(object())
    authority = build_packed_terminal_decode_request_authority(
        startup_binding=_binding(
            startup_matrix,
            service_id="decode-a",
            rank=0,
        ),
        transaction=transaction,
        adopt_request=lambda owner: None,
        finalize_request=lambda owner: None,
        cancel_request=lambda owner: None,
        quarantine_request=lambda owner, reason: None,
        publication_generation=b"p" * 16,
    )
    return transaction, authority


def test_builder_joins_exact_transaction_to_sealed_startup_matrix() -> None:
    """Derive writer, publisher, issuer, and allocation identity without discovery."""

    transaction, authority = _authority()

    assert authority.registration.transaction is transaction
    assert authority.binding.owner.role is TerminalOwnerRole.DECODE
    assert authority.binding.owner.tp_rank == 0
    assert authority.binding.rank_manifest_digest == b"w" * 32
    assert authority.binding.allocation_digest == b"a" * 32
    assert tuple(writer.writer_id for writer in authority.source_plan.writers) == (
        _writer(0),
        _writer(1),
    )
    assert tuple(
        writer.process_identity.process_generation
        for writer in authority.source_plan.writers
    ) == ((101).to_bytes(16, "big"), (102).to_bytes(16, "big"))
    assert authority.source_plan.request_ready_issuer == authority.binding.owner
    assert authority.source_plan.publication_identity.publisher_process_generation == (
        101
    ).to_bytes(16, "big")
    assert authority.coordinator_manifest.destination_bindings == (authority.binding,)
    assert authority.coordinator_manifest.recipient_bindings == (
        authority.binding,
        *authority.source_plan.source_bindings,
    )


def test_transaction_binds_exact_payload_only_with_owner_registration() -> None:
    """Make allocation publication structurally depend on terminal owner binding."""

    transaction, authority = _authority()
    transaction.bind_terminal_owner_authority(
        authority.encoded_source_plan,
        authority.binding.digest,
    )

    assert (
        transaction.prepared_publication().terminal_source_plan
        == authority.encoded_source_plan
    )
    with pytest.raises(PackedRequestTransactionError, match="already bound"):
        transaction.bind_terminal_owner_authority(
            authority.encoded_source_plan,
            authority.binding.digest,
        )


def test_publication_rejects_terminal_payload_without_owner_registration() -> None:
    """Reject a source plan attached without actor and native lifecycle authority."""

    transaction, authority = _authority()
    transaction._publication = dataclasses.replace(
        transaction._publication,
        terminal_source_plan=authority.encoded_source_plan,
    )

    with pytest.raises(PackedRequestTransactionError, match="incomplete"):
        transaction.publish()


def test_source_projection_selects_local_writer_from_sealed_matrix() -> None:
    """Resolve source rank one without request-time peer discovery."""

    matrix = _matrix()
    _, authority = _authority(matrix=matrix)
    projected = project_packed_terminal_source_authority(
        startup_binding=_binding(matrix, service_id="prefill-a", rank=1),
        source_plan=authority.source_plan,
        local_writer_id=_writer(1),
        destination_process_generation=(110).to_bytes(16, "big"),
    )

    assert projected.local_binding == authority.source_plan.source_bindings[1]
    assert (
        projected.publisher_issuer == authority.source_plan.writers[0].process_identity
    )


def test_source_projection_rejects_route_generation_drift() -> None:
    """Refuse a request-ready issuer that differs from the metadata route."""

    matrix = _matrix()
    _, authority = _authority(matrix=matrix)

    with pytest.raises(TerminalStartupCohortError, match="absent"):
        project_packed_terminal_source_authority(
            startup_binding=_binding(matrix, service_id="prefill-a", rank=0),
            source_plan=authority.source_plan,
            local_writer_id=_writer(0),
            destination_process_generation=b"z" * 16,
        )


def test_decode_tp_requires_real_cross_rank_allocation_bindings() -> None:
    """Never manufacture peer allocation identity from one local transaction."""

    matrix = _matrix(decode_tp_size=2)
    transaction = _transaction(object())

    with pytest.raises(
        PackedTerminalRequestRegistrationError,
        match="cross-rank request binding manifest",
    ):
        build_packed_terminal_decode_request_authority(
            startup_binding=_binding(matrix, service_id="decode-a", rank=0),
            transaction=transaction,
            adopt_request=lambda owner: None,
            finalize_request=lambda owner: None,
            cancel_request=lambda owner: None,
            quarantine_request=lambda owner, reason: None,
        )


def test_transfer_info_carries_exact_source_plan_on_existing_metadata_path() -> None:
    """Round-trip terminal authority beside the packed auxiliary frame."""

    transaction, authority = _authority()
    publication = transaction.prepared_publication()
    frames = [
        b"41",
        b"127.0.0.1",
        b"32010",
        b"decode-agent",
        np.array([1, 2], dtype=np.int32).tobytes(),
        b"3",
        b"1",
        b"",
        b"0",
        b"00000000-0000-0000-0000-00000000006e",
        encode_packed_message(publication.auxiliary_plan),
        authority.encoded_source_plan,
    ]

    transfer_info = TransferInfo.from_zmq(frames)

    assert transfer_info.terminal_source_plan_payload == authority.encoded_source_plan
    assert transfer_info.decode_terminal_source_plan() == authority.source_plan


def test_transfer_info_rejects_cross_generation_source_plan() -> None:
    """Reject terminal authority which names another packed request generation."""

    transaction, authority = _authority()
    publication = transaction.prepared_publication()
    other_key = PackedRequestKey(room_id=42, request_generation=b"h" * 16)
    other_plan = dataclasses.replace(
        authority.source_plan,
        request_key=other_key,
        publication_identity=dataclasses.replace(
            authority.source_plan.publication_identity,
            request_key=other_key,
        ),
    )

    frames = [
        b"41",
        b"127.0.0.1",
        b"32010",
        b"decode-agent",
        np.array([1], dtype=np.int32).tobytes(),
        b"3",
        b"1",
        b"",
        b"0",
        b"00000000-0000-0000-0000-00000000006e",
        encode_packed_message(publication.auxiliary_plan),
        encode_packed_terminal_source_plan(other_plan),
    ]

    with pytest.raises(PackedTerminalRequestRegistrationError, match="another"):
        TransferInfo.from_zmq(frames)
