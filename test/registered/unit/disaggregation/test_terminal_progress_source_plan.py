import dataclasses

import msgspec
import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    MAX_PACKED_TERMINAL_SOURCE_PLAN_BYTES,
    PackedTerminalSourcePlan,
    PackedTerminalSourcePlanError,
    PackedTerminalSourceWriter,
    decode_packed_terminal_source_plan,
    encode_packed_terminal_source_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _source_process(rank: int, *, tp_size: int = 2) -> TerminalProcessIdentity:
    """Build one exact source process fixture.

    :param rank: Source tensor-parallel rank.
    :param tp_size: Complete source tensor-parallel width.
    :returns: Valid source process identity.
    """

    return TerminalProcessIdentity(
        process_generation=bytes((0x10 + rank,)) * 16,
        role=TerminalOwnerRole.SOURCE,
        tp_rank=rank,
        tp_size=tp_size,
    )


def _writer(rank: int, *, tp_size: int = 2) -> PackedTerminalSourceWriter:
    """Build one explicitly associated source writer fixture.

    :param rank: Source tensor-parallel rank.
    :param tp_size: Complete source tensor-parallel width.
    :returns: Writer and terminal-owner association.
    """

    return PackedTerminalSourceWriter(
        writer_id=StagingWriterId(
            transfer_source_rank=6 + rank,
            source_attn_tp_rank=rank,
            source_pp_rank=0,
            source_cp_rank=0,
        ),
        process_identity=_source_process(rank, tp_size=tp_size),
    )


def _plan() -> PackedTerminalSourcePlan:
    """Build one complete TP2 source identity plan.

    :returns: Valid decoder-authored source plan.
    """

    request_key = PackedRequestKey(
        room_id=73,
        request_generation=bytes.fromhex("73" * 16),
    )
    writers = (_writer(0), _writer(1))
    return PackedTerminalSourcePlan(
        request_key=request_key,
        writers=writers,
        rank_manifest_digest=bytes.fromhex("41" * 32),
        allocation_digest=bytes.fromhex("51" * 32),
        publication_identity=TerminalPublicationIdentity(
            request_key=request_key,
            publisher_process_generation=(
                writers[0].process_identity.process_generation
            ),
            publication_generation=bytes.fromhex("61" * 16),
        ),
        request_ready_issuer=TerminalProcessIdentity(
            process_generation=bytes.fromhex("31" * 16),
            role=TerminalOwnerRole.DECODE,
            tp_rank=0,
            tp_size=1,
        ),
    )


def test_source_plan_round_trip_is_exact_and_canonical() -> None:
    """Preserve the complete source identity graph byte-for-byte."""

    plan = _plan()
    payload = encode_packed_terminal_source_plan(plan)

    assert decode_packed_terminal_source_plan(payload) == plan
    assert encode_packed_terminal_source_plan(
        decode_packed_terminal_source_plan(payload)
    ) == payload


def test_source_plan_preserves_explicit_writer_process_association() -> None:
    """Select rank-local authority by exact transport writer identity."""

    plan = _plan()
    selected = plan.identity_for_writer(plan.writers[1].writer_id)

    assert selected.local_binding.owner == plan.writers[1].process_identity
    assert selected.local_binding == plan.source_bindings[1]
    assert selected.source_bindings == plan.source_bindings
    assert selected.publisher_issuer == plan.writers[0].process_identity


def test_source_plan_rejects_unknown_fields_and_versions() -> None:
    """Reject schema drift and unsupported wire semantics."""

    envelope = msgspec.msgpack.decode(encode_packed_terminal_source_plan(_plan()))

    unknown_field = dict(envelope)
    unknown_field["surprise"] = 1
    with pytest.raises(PackedTerminalSourcePlanError, match="invalid"):
        decode_packed_terminal_source_plan(msgspec.msgpack.encode(unknown_field))

    unknown_version = dict(envelope)
    unknown_version["version"] = 99
    with pytest.raises(PackedTerminalSourcePlanError, match="unsupported.*version"):
        decode_packed_terminal_source_plan(msgspec.msgpack.encode(unknown_version))


def test_source_plan_rejects_wrong_rank_order_and_tp_size() -> None:
    """Require one canonical, complete tensor-parallel source manifest."""

    plan = _plan()
    with pytest.raises(ValueError, match="canonical TP-rank order"):
        dataclasses.replace(plan, writers=tuple(reversed(plan.writers)))

    wrong_size = _writer(1, tp_size=3)
    with pytest.raises(ValueError, match="disagrees on TP size"):
        dataclasses.replace(plan, writers=(plan.writers[0], wrong_size))


def test_source_writer_rejects_mismatched_transport_rank() -> None:
    """Forbid implicit writer-to-process rank inference."""

    writer = _writer(0)
    with pytest.raises(ValueError, match="writer TP rank differs"):
        dataclasses.replace(
            writer,
            writer_id=dataclasses.replace(
                writer.writer_id,
                source_attn_tp_rank=1,
            ),
        )


def test_source_plan_rejects_wrong_publisher_generation() -> None:
    """Bind publication authority to canonical source rank zero."""

    plan = _plan()
    with pytest.raises(ValueError, match="another source process"):
        dataclasses.replace(
            plan,
            publication_identity=dataclasses.replace(
                plan.publication_identity,
                publisher_process_generation=bytes.fromhex("ff" * 16),
            ),
        )


def test_source_plan_rejects_missing_local_writer() -> None:
    """Fail closed when local transport identity is absent from the manifest."""

    missing = StagingWriterId(
        transfer_source_rank=99,
        source_attn_tp_rank=0,
        source_pp_rank=0,
        source_cp_rank=0,
    )
    with pytest.raises(PackedTerminalSourcePlanError, match="local writer"):
        _plan().identity_for_writer(missing)


def test_source_plan_rejects_malformed_truncated_and_oversized_payloads() -> None:
    """Bound and strictly validate every untrusted wire payload."""

    payload = encode_packed_terminal_source_plan(_plan())
    with pytest.raises(PackedTerminalSourcePlanError, match="invalid"):
        decode_packed_terminal_source_plan(payload[:-1])
    with pytest.raises(PackedTerminalSourcePlanError, match="must not be empty"):
        decode_packed_terminal_source_plan(b"")
    with pytest.raises(PackedTerminalSourcePlanError, match="exceeds"):
        decode_packed_terminal_source_plan(
            b"x" * (MAX_PACKED_TERMINAL_SOURCE_PLAN_BYTES + 1)
        )
    with pytest.raises(PackedTerminalSourcePlanError, match="invalid"):
        decode_packed_terminal_source_plan(b"\xc1")
