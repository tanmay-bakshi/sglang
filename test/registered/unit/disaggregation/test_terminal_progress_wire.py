import dataclasses

import pytest

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceipt,
    TerminalWireReceiptError,
    TerminalWireReceiptImportNamespace,
    TerminalWireReceiptIssuer,
)


def _identity(
    seed: int, role: TerminalOwnerRole, rank: int, size: int
) -> TerminalProcessIdentity:
    return TerminalProcessIdentity(
        process_generation=bytes((seed,)) * 16,
        role=role,
        tp_rank=rank,
        tp_size=size,
    )


def _binding(owner: TerminalProcessIdentity, seed: int = 3) -> TerminalRequestBinding:
    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=41,
            request_generation=bytes((seed,)) * 16,
        ),
        owner=owner,
        rank_manifest_digest=b"r" * 32,
        allocation_digest=b"a" * 32,
    )


def test_wire_receipt_round_trip_is_canonical() -> None:
    issuer_identity = _identity(1, TerminalOwnerRole.DECODE, 0, 2)
    target = _binding(_identity(2, TerminalOwnerRole.SOURCE, 1, 2))
    issued = TerminalWireReceiptIssuer(issuer_identity).issue(
        binding=target,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=123_456,
    )

    payload = issued.wire_receipt.encode()
    decoded = TerminalWireReceipt.decode(payload)

    assert decoded == issued.wire_receipt
    assert decoded.encode() == payload
    assert decoded.digest == issued.wire_receipt.digest


def test_authenticated_import_preserves_one_shot_local_authority() -> None:
    remote = _identity(1, TerminalOwnerRole.DECODE, 0, 1)
    target = _binding(_identity(2, TerminalOwnerRole.SOURCE, 0, 2))
    issued = TerminalWireReceiptIssuer(remote).issue(
        binding=target,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=99,
    )
    namespace = TerminalWireReceiptImportNamespace(remote)
    namespace.register_binding(target)

    first = namespace.import_receipt(issued.wire_receipt, remote)
    duplicate = namespace.import_receipt(issued.wire_receipt, remote)
    ledger = TerminalReceiptLedger(authorities=frozenset((namespace.authority,)))

    assert first is duplicate
    consumed = ledger.consume(
        first,
        target,
        TerminalReceiptKind.REQUEST_READY,
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        consumed.consume(
            duplicate,
            target,
            TerminalReceiptKind.REQUEST_READY,
        )


def test_import_rejects_unauthenticated_and_inactive_bindings() -> None:
    remote = _identity(1, TerminalOwnerRole.DECODE, 0, 1)
    impostor = _identity(9, TerminalOwnerRole.DECODE, 0, 1)
    target = _binding(_identity(2, TerminalOwnerRole.SOURCE, 0, 2))
    issued = TerminalWireReceiptIssuer(remote).issue(
        binding=target,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=99,
    )
    namespace = TerminalWireReceiptImportNamespace(remote)

    with pytest.raises(TerminalWireReceiptError, match="another issuer"):
        namespace.import_receipt(issued.wire_receipt, impostor)
    with pytest.raises(TerminalWireReceiptError, match="inactive"):
        namespace.import_receipt(issued.wire_receipt, remote)

    namespace.register_binding(target)
    namespace.import_receipt(issued.wire_receipt, remote)
    namespace.retire_binding(target)
    with pytest.raises(TerminalWireReceiptError, match="inactive"):
        namespace.import_receipt(issued.wire_receipt, remote)


def test_import_rejects_conflicting_nonce_reuse() -> None:
    remote = _identity(1, TerminalOwnerRole.DECODE, 0, 1)
    target = _binding(_identity(2, TerminalOwnerRole.SOURCE, 0, 2))
    issued = TerminalWireReceiptIssuer(remote).issue(
        binding=target,
        kind=TerminalReceiptKind.REQUEST_READY,
        outcome=TerminalReceiptOutcome.SUCCESS,
        terminal_timestamp_ns=99,
    )
    namespace = TerminalWireReceiptImportNamespace(remote)
    namespace.register_binding(target)
    namespace.import_receipt(issued.wire_receipt, remote)
    conflicting = dataclasses.replace(
        issued.wire_receipt,
        terminal_timestamp_ns=100,
    )

    with pytest.raises(TerminalWireReceiptError, match="conflicting fields"):
        namespace.import_receipt(conflicting, remote)


def test_decoder_rejects_unknown_codes_and_shapes() -> None:
    remote = _identity(1, TerminalOwnerRole.DECODE, 0, 1)
    target = _binding(_identity(2, TerminalOwnerRole.SOURCE, 0, 2))
    payload = (
        TerminalWireReceiptIssuer(remote)
        .issue(
            binding=target,
            kind=TerminalReceiptKind.REQUEST_READY,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=99,
        )
        .wire_receipt.encode()
    )

    with pytest.raises(TerminalWireReceiptError, match="byte length"):
        TerminalWireReceipt.decode(payload[:-1])
    corrupted_magic = b"BAD!" + payload[4:]
    with pytest.raises(TerminalWireReceiptError, match="magic"):
        TerminalWireReceipt.decode(corrupted_magic)
