import dataclasses

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    PACKED_TERMINAL_DEADLINES,
    BoundTerminalDeadline,
    TerminalDeadlineKind,
    canonical_terminal_deadline_table,
    start_terminal_deadline,
    terminal_deadline_spec,
    terminal_deadline_table_digest,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptError,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REQUEST_GENERATION = bytes.fromhex("00112233445566778899aabbccddeeff")
PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
PUBLICATION_GENERATION = bytes.fromhex("ffeeddccbbaa99887766554433221100")
RANK_MANIFEST_DIGEST = b"r" * 32
ALLOCATION_DIGEST = b"a" * 32


def make_binding(
    role: TerminalOwnerRole = TerminalOwnerRole.SOURCE,
    *,
    room_id: int = 17,
) -> TerminalRequestBinding:
    """Create one exact terminal-owner request binding.

    :param role: Owner role to bind.
    :param room_id: Stable packed room identity.
    :returns: Valid immutable request binding.
    """

    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=room_id,
            request_generation=REQUEST_GENERATION,
        ),
        owner=TerminalProcessIdentity(
            process_generation=PROCESS_GENERATION,
            role=role,
            tp_rank=0,
            tp_size=2,
        ),
        rank_manifest_digest=RANK_MANIFEST_DIGEST,
        allocation_digest=ALLOCATION_DIGEST,
    )


def test_identity_digests_are_stable_and_values_are_immutable() -> None:
    binding = make_binding()
    publication = TerminalPublicationIdentity(
        request_key=binding.request_key,
        publisher_process_generation=PROCESS_GENERATION,
        publication_generation=PUBLICATION_GENERATION,
    )

    assert binding.owner.digest.hex() == (
        "0673986bb18792728fd025b0e3c500961ede5af355263a6764afda54fa3fe20a"
    )
    assert binding.digest.hex() == (
        "aec4b123c26646ec43d9e0f438fa9b3670ae9a97a69427c8c4fdf3c2de040232"
    )
    assert publication.digest.hex() == (
        "c163b0434174823c90fa33809dd924d620a351c0110cea32da34aa4110d8ab2a"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.rank_manifest_digest = b"x" * 32


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("process_generation", b"short"),
        ("rank_manifest_digest", b"short"),
        ("allocation_digest", b"short"),
    ),
)
def test_identity_widths_are_exact(field: str, value: bytes) -> None:
    binding = make_binding()
    if field == "process_generation":
        with pytest.raises(ValueError, match="process_generation"):
            TerminalProcessIdentity(
                process_generation=value,
                role=TerminalOwnerRole.SOURCE,
                tp_rank=0,
                tp_size=1,
            )
        return
    kwargs = {
        "request_key": binding.request_key,
        "owner": binding.owner,
        "rank_manifest_digest": binding.rank_manifest_digest,
        "allocation_digest": binding.allocation_digest,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        TerminalRequestBinding(**kwargs)


def test_receipt_ledger_enforces_trust_exact_binding_and_one_shot_consumption() -> None:
    binding = make_binding()
    other_binding = make_binding(room_id=18)
    trusted_issuer = TerminalReceiptIssuer()
    untrusted_issuer = TerminalReceiptIssuer()
    ledger = TerminalReceiptLedger(authorities=frozenset((trusted_issuer.authority,)))
    receipt = trusted_issuer.issue(
        binding,
        TerminalReceiptKind.REQUEST_READY,
        TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=100,
    )

    consumed = ledger.consume(
        receipt,
        binding,
        TerminalReceiptKind.REQUEST_READY,
    )
    assert len(ledger.consumed_tokens) == 0
    assert len(consumed.consumed_tokens) == 1
    with pytest.raises(TerminalReceiptError, match="already consumed"):
        consumed.consume(receipt, binding, TerminalReceiptKind.REQUEST_READY)
    with pytest.raises(TerminalReceiptError, match="another request"):
        ledger.consume(receipt, other_binding, TerminalReceiptKind.REQUEST_READY)

    untrusted_receipt = untrusted_issuer.issue(
        binding,
        TerminalReceiptKind.REQUEST_READY,
        TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=101,
    )
    with pytest.raises(TerminalReceiptError, match="untrusted issuer"):
        ledger.consume(
            untrusted_receipt,
            binding,
            TerminalReceiptKind.REQUEST_READY,
        )


def test_receipts_are_immutable_and_kind_outcome_are_exact() -> None:
    binding = make_binding()
    issuer = TerminalReceiptIssuer()
    ledger = TerminalReceiptLedger(authorities=frozenset((issuer.authority,)))
    receipt = issuer.issue(
        binding,
        TerminalReceiptKind.FAILURE,
        TerminalReceiptOutcome.FAILURE,
        terminal_timestamp_ns=55,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.terminal_timestamp_ns = 56
    with pytest.raises(TerminalReceiptError, match="does not authorize"):
        ledger.consume(receipt, binding, TerminalReceiptKind.REQUEST_READY)
    with pytest.raises(TerminalReceiptError, match="outcome"):
        ledger.consume(
            receipt,
            binding,
            TerminalReceiptKind.FAILURE,
            TerminalReceiptOutcome.CANCELLED,
        )


def test_deadline_table_is_complete_hash_bound_and_one_shot() -> None:
    expected_seconds = {
        TerminalDeadlineKind.EXISTING_NIXL_CAPABILITY_READY: 5.0,
        TerminalDeadlineKind.EXISTING_PACKED_CONTROL: 60.0,
        TerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER: 60.0,
        TerminalDeadlineKind.OWNER_NATIVE_TRANSFER: 60.0,
        TerminalDeadlineKind.OWNER_DECODE_SCATTER: 60.0,
        TerminalDeadlineKind.OWNER_TEARDOWN_ACK: 60.0,
        TerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY: 60.0,
        TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION: 60.0,
        TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION: 60.0,
        TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN: 60.0,
    }

    assert len(PACKED_TERMINAL_DEADLINES) == len(TerminalDeadlineKind)
    assert {
        spec.kind: spec.seconds for spec in PACKED_TERMINAL_DEADLINES
    } == expected_seconds
    assert len(canonical_terminal_deadline_table()) == len(expected_seconds)
    assert terminal_deadline_table_digest().hex() == (
        "093dc590ed62aead9216de256ae6a66f089e8ff8a3f6d902850c040e1594e34f"
    )

    deadline = start_terminal_deadline(
        TerminalDeadlineKind.OWNER_NATIVE_TRANSFER,
        started_ns=1_000,
    )
    assert isinstance(deadline, BoundTerminalDeadline)
    assert deadline.spec is terminal_deadline_spec(
        TerminalDeadlineKind.OWNER_NATIVE_TRANSFER
    )
    assert deadline.expires_ns == 60_000_001_000
    assert not deadline.expired(60_000_000_999)
    assert deadline.expired(60_000_001_000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        deadline.started_ns = 2_000
