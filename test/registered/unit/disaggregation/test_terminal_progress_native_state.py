import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    PACKED_TERMINAL_DEADLINES,
    terminal_deadline_table_digest,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NATIVE_DECODE_RESOURCE_MASK,
    NATIVE_SOURCE_RECLAIMABLE_MASK,
    NATIVE_SOURCE_RESOURCE_MASK,
    NativeSourceLifecyclePhase,
    NativeTerminalDeadlineKind,
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerFatalCode,
    NativeTerminalOwnerOutput,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalPublicationIdentity,
    NativeTerminalReceipt,
    NativeTerminalReceiptKind,
    NativeTerminalReceiptOutcome,
    NativeTerminalRequestBinding,
    NativeTerminalResource,
    canonical_native_terminal_deadlines,
    native_terminal_deadline_table_digest,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
SOURCE_PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
DECODE_PROCESS_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
PUBLICATION_GENERATION = bytes.fromhex("0123456789abcdeffedcba9876543210")
RECEIPT_NONCE = bytes.fromhex("55" * 16)


def make_process_identity(role: TerminalOwnerRole) -> TerminalProcessIdentity:
    """Create one canonical process identity.

    :param role: Source or decode owner role.
    :returns: Exact rank-local identity.
    """

    generation = SOURCE_PROCESS_GENERATION
    if role is TerminalOwnerRole.DECODE:
        generation = DECODE_PROCESS_GENERATION
    return TerminalProcessIdentity(
        process_generation=generation,
        role=role,
        tp_rank=1,
        tp_size=4,
    )


def make_binding(role: TerminalOwnerRole) -> TerminalRequestBinding:
    """Create one canonical request binding.

    :param role: Source or decode owner role.
    :returns: Exact binding with full manifest and allocation identity.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=71,
            request_generation=REQUEST_GENERATION,
        ),
        owner=make_process_identity(role),
        rank_manifest_digest=b"r" * 32,
        allocation_digest=b"a" * 32,
    )


def make_publication_identity() -> TerminalPublicationIdentity:
    """Create the source request's canonical publication identity.

    :returns: Exact publication generation.
    """

    source_binding = make_binding(TerminalOwnerRole.SOURCE)
    return TerminalPublicationIdentity(
        request_key=source_binding.request_key,
        publisher_process_generation=SOURCE_PROCESS_GENERATION,
        publication_generation=PUBLICATION_GENERATION,
    )


def test_native_binding_retains_complete_identity_and_digest() -> None:
    binding = make_binding(TerminalOwnerRole.SOURCE)
    native = NativeTerminalRequestBinding.from_binding(binding)

    assert native.room_id == binding.request_key.room_id
    assert native.request_generation == binding.request_key.request_generation
    assert native.owner.process_generation == binding.owner.process_generation
    assert native.owner.role is NativeTerminalOwnerRole.SOURCE
    assert native.owner.tp_rank == binding.owner.tp_rank
    assert native.owner.tp_size == binding.owner.tp_size
    assert native.owner.digest == binding.owner.digest
    assert native.rank_manifest_digest == binding.rank_manifest_digest
    assert native.allocation_digest == binding.allocation_digest
    assert native.digest == binding.digest
    assert native.to_native()["digest"] == binding.digest


def test_native_deadline_table_is_exactly_the_packaged_hash_bound_table() -> None:
    native = canonical_native_terminal_deadlines()

    assert len(native) == len(PACKED_TERMINAL_DEADLINES) == 10
    assert tuple(value.kind for value in native) == tuple(NativeTerminalDeadlineKind)
    assert tuple(value.duration_ns for value in native) == tuple(
        spec.duration_ns for spec in PACKED_TERMINAL_DEADLINES
    )
    assert tuple(value.process_fatal for value in native) == (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    )
    assert tuple(value.starts_at for value in native) == tuple(
        spec.starts_at for spec in PACKED_TERMINAL_DEADLINES
    )
    assert tuple(value.timeout_outcome for value in native) == tuple(
        spec.timeout_outcome for spec in PACKED_TERMINAL_DEADLINES
    )
    assert native_terminal_deadline_table_digest() == terminal_deadline_table_digest()


def test_native_resource_vocabulary_matches_canonical_dflash_lifetimes() -> None:
    assert NativeTerminalResource.SOURCE_DFLASH_AUX_VRAM_ROWS.value == 1 << 7
    assert NativeTerminalResource.DECODE_DFLASH_AUX_VRAM_ROWS.value == 1 << 13
    assert "SOURCE_METADATA" not in NativeTerminalResource.__members__
    assert "DECODE_METADATA_ROW" not in NativeTerminalResource.__members__


def test_native_registration_enforces_role_specific_publication_identity() -> None:
    source_binding = NativeTerminalRequestBinding.from_binding(
        make_binding(TerminalOwnerRole.SOURCE)
    )
    decode_binding = NativeTerminalRequestBinding.from_binding(
        make_binding(TerminalOwnerRole.DECODE)
    )
    publication = NativeTerminalPublicationIdentity.from_identity(
        make_publication_identity()
    )
    source_issuer = NativeTerminalProcessIdentity.from_identity(
        make_process_identity(TerminalOwnerRole.SOURCE)
    )

    registration = NativeTerminalLifecycleRegistration(
        binding=source_binding,
        publication_identity=publication,
        trusted_issuers=(source_issuer,),
    )

    assert registration.to_native()["binding"]["digest"] == source_binding.digest
    with pytest.raises(ValueError, match="source registration requires"):
        NativeTerminalLifecycleRegistration(
            binding=source_binding,
            publication_identity=None,
            trusted_issuers=(source_issuer,),
        )
    with pytest.raises(ValueError, match="decode registration cannot"):
        NativeTerminalLifecycleRegistration(
            binding=decode_binding,
            publication_identity=publication,
            trusted_issuers=(source_issuer,),
        )
    with pytest.raises(ValueError, match="must be unique"):
        NativeTerminalLifecycleRegistration(
            binding=source_binding,
            publication_identity=publication,
            trusted_issuers=(source_issuer, source_issuer),
        )


def test_native_receipt_preserves_authenticated_wire_authority() -> None:
    binding = make_binding(TerminalOwnerRole.SOURCE)
    issuer = make_process_identity(TerminalOwnerRole.DECODE)
    wire = TerminalWireReceipt(
        binding=binding,
        issuer=issuer,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=99,
        receipt_nonce=RECEIPT_NONCE,
    )

    native = NativeTerminalReceipt.from_wire_receipt(wire)

    assert native.binding.digest == binding.digest
    assert native.issuer.digest == issuer.digest
    assert native.kind is NativeTerminalReceiptKind.REQUEST_READY
    assert native.outcome is NativeTerminalReceiptOutcome.SUCCESS
    assert native.nonce == RECEIPT_NONCE
    roundtrip = NativeTerminalReceipt.from_native(native.to_native())
    assert roundtrip == native


def test_native_event_has_no_producer_selected_phase() -> None:
    binding = NativeTerminalRequestBinding.from_binding(
        make_binding(TerminalOwnerRole.SOURCE)
    )
    event = NativeTerminalOwnerEvent(
        producer_id=4,
        binding_digest=binding.digest,
        kind=NativeTerminalOwnerEventKind.SOURCE_NATIVE_TERMINAL,
        enqueued_ns=123,
    )

    assert set(event.to_native()) == {
        "producer_id",
        "binding_digest",
        "kind",
        "enqueued_ns",
        "receipt",
        "reason",
    }


def test_native_output_requires_complete_exclusive_resource_partition() -> None:
    binding = NativeTerminalRequestBinding.from_binding(
        make_binding(TerminalOwnerRole.SOURCE)
    )
    reclaim_receipt = NativeTerminalReceipt(
        binding=binding,
        issuer=NativeTerminalProcessIdentity.from_identity(
            make_process_identity(TerminalOwnerRole.SOURCE)
        ),
        kind=NativeTerminalReceiptKind.RECLAIM_AUTHORIZED,
        outcome=NativeTerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=456,
        nonce=RECEIPT_NONCE,
    )
    reclaim_action = NativeTerminalOwnerAction(
        action_id=1,
        kind=NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        binding=binding,
        commit_timestamp_ns=456,
        receipt=reclaim_receipt,
    )
    output = NativeTerminalOwnerOutput(
        binding=binding,
        owner_sequence=8,
        producer_id=4,
        producer_sequence=12,
        event_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
        enqueued_ns=123,
        completed_ns=456,
        role=NativeTerminalOwnerRole.SOURCE,
        previous_phase=int(NativeSourceLifecyclePhase.ACK_SENT),
        phase=int(NativeSourceLifecyclePhase.REQUEST_READY_RECEIVED),
        live_resources=NATIVE_SOURCE_RESOURCE_MASK,
        retired_resources=0,
        quarantined_resources=0,
        actions=(reclaim_action,),
        armed_deadline_mask=0,
        process_fatal=False,
        fatal_code=NativeTerminalOwnerFatalCode.NONE,
    )

    assert output.latency_ns == 333
    assert NATIVE_SOURCE_RECLAIMABLE_MASK < NATIVE_SOURCE_RESOURCE_MASK
    assert NATIVE_SOURCE_RESOURCE_MASK & NATIVE_DECODE_RESOURCE_MASK == 0
    with pytest.raises(ValueError, match="partitions overlap"):
        NativeTerminalOwnerOutput(
            binding=binding,
            owner_sequence=8,
            producer_id=4,
            producer_sequence=12,
            event_kind=NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
            enqueued_ns=123,
            completed_ns=456,
            role=NativeTerminalOwnerRole.SOURCE,
            previous_phase=int(NativeSourceLifecyclePhase.ACK_SENT),
            phase=int(NativeSourceLifecyclePhase.REQUEST_READY_RECEIVED),
            live_resources=NATIVE_SOURCE_RESOURCE_MASK,
            retired_resources=1,
            quarantined_resources=0,
            actions=(reclaim_action,),
            armed_deadline_mask=0,
            process_fatal=False,
            fatal_code=NativeTerminalOwnerFatalCode.NONE,
        )
