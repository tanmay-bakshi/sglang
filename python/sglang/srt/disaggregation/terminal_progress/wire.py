import dataclasses
import enum
import hashlib
import secrets
import struct
import threading

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TERMINAL_RECEIPT_NONCE_BYTES,
    TerminalReceipt,
    TerminalReceiptAuthority,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)

TERMINAL_WIRE_RECEIPT_MAGIC = b"PTR1"

_TERMINAL_WIRE_RECEIPT = struct.Struct("!4sQ16s16sBII32s32s16sBIIBBQ16s")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class TerminalWireReceiptError(RuntimeError):
    """Authenticated terminal-receipt wire invariant violation."""


class _TerminalOwnerRoleCode(enum.IntEnum):
    """Stable wire code for a terminal owner role."""

    SOURCE = 1
    DECODE = 2


class _TerminalReceiptKindCode(enum.IntEnum):
    """Stable wire code for a terminal receipt kind."""

    ADOPTION_READY = 1
    METADATA_CONSUMED = 2
    LOCAL_DECODE_READY = 3
    REQUEST_READY = 4
    RECLAIM_AUTHORIZED = 5
    RECLAIM_CONSUMED = 6
    GATEWAY_PUBLISHED = 7
    FAILURE = 8


class _TerminalReceiptOutcomeCode(enum.IntEnum):
    """Stable wire code for a terminal receipt outcome."""

    SUCCESS = 1
    FAILURE = 2
    CANCELLED = 3


_ROLE_TO_CODE = {
    TerminalOwnerRole.SOURCE: _TerminalOwnerRoleCode.SOURCE,
    TerminalOwnerRole.DECODE: _TerminalOwnerRoleCode.DECODE,
}
_CODE_TO_ROLE = {value: key for key, value in _ROLE_TO_CODE.items()}
_KIND_TO_CODE = {
    TerminalReceiptKind.ADOPTION_READY: _TerminalReceiptKindCode.ADOPTION_READY,
    TerminalReceiptKind.METADATA_CONSUMED: _TerminalReceiptKindCode.METADATA_CONSUMED,
    TerminalReceiptKind.LOCAL_DECODE_READY: (
        _TerminalReceiptKindCode.LOCAL_DECODE_READY
    ),
    TerminalReceiptKind.REQUEST_READY: _TerminalReceiptKindCode.REQUEST_READY,
    TerminalReceiptKind.RECLAIM_AUTHORIZED: (
        _TerminalReceiptKindCode.RECLAIM_AUTHORIZED
    ),
    TerminalReceiptKind.RECLAIM_CONSUMED: (_TerminalReceiptKindCode.RECLAIM_CONSUMED),
    TerminalReceiptKind.GATEWAY_PUBLISHED: (_TerminalReceiptKindCode.GATEWAY_PUBLISHED),
    TerminalReceiptKind.FAILURE: _TerminalReceiptKindCode.FAILURE,
}
_CODE_TO_KIND = {value: key for key, value in _KIND_TO_CODE.items()}
_OUTCOME_TO_CODE = {
    TerminalReceiptOutcome.SUCCESS: _TerminalReceiptOutcomeCode.SUCCESS,
    TerminalReceiptOutcome.FAILURE: _TerminalReceiptOutcomeCode.FAILURE,
    TerminalReceiptOutcome.CANCELLED: _TerminalReceiptOutcomeCode.CANCELLED,
}
_CODE_TO_OUTCOME = {value: key for key, value in _OUTCOME_TO_CODE.items()}


def _require_uint(value: int, maximum: int, label: str) -> None:
    """Validate an unsigned integer field.

    :param value: Candidate integer.
    :param maximum: Inclusive upper bound.
    :param label: Reader-facing field name.
    """

    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{label} must be an unsigned integer at most {maximum}")


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalWireReceipt:
    """Fixed-width receipt transported over an authenticated owner route.

    The wire value does not prove who sent it. The receiving control path must
    supply the independently authenticated process identity to an import
    namespace. This keeps transport authentication and one-shot lifecycle
    authority as two explicit, joined proofs.

    :ivar binding: Exact target owner and request generation.
    :ivar issuer: Process identity authenticated by the control route.
    :ivar kind: Authority represented by this receipt.
    :ivar outcome: Terminal outcome represented by this receipt.
    :ivar terminal_timestamp_ns: Issuer-local monotonic timestamp.
    :ivar receipt_nonce: Issuer-local one-shot wire identity.
    """

    binding: TerminalRequestBinding
    issuer: TerminalProcessIdentity
    kind: TerminalReceiptKind
    outcome: TerminalReceiptOutcome
    terminal_timestamp_ns: int
    receipt_nonce: bytes

    def __post_init__(self) -> None:
        """Validate one complete fixed-width wire receipt."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.issuer) is not TerminalProcessIdentity:
            raise TypeError("issuer must be TerminalProcessIdentity")
        if type(self.kind) is not TerminalReceiptKind:
            raise TypeError("kind must be TerminalReceiptKind")
        if type(self.outcome) is not TerminalReceiptOutcome:
            raise TypeError("outcome must be TerminalReceiptOutcome")
        _require_uint(
            self.terminal_timestamp_ns,
            _UINT64_MAX,
            "terminal_timestamp_ns",
        )
        if type(self.receipt_nonce) is not bytes:
            raise TypeError("receipt_nonce must be bytes")
        if len(self.receipt_nonce) != TERMINAL_RECEIPT_NONCE_BYTES:
            raise ValueError(
                f"receipt_nonce must contain {TERMINAL_RECEIPT_NONCE_BYTES} bytes"
            )

    def encode(self) -> bytes:
        """Encode this receipt into its canonical fixed-width representation.

        :returns: Canonical wire bytes.
        """

        target = self.binding.owner
        _require_uint(self.binding.request_key.room_id, _UINT64_MAX, "room_id")
        _require_uint(target.tp_rank, _UINT32_MAX, "target tp_rank")
        _require_uint(target.tp_size, _UINT32_MAX, "target tp_size")
        _require_uint(self.issuer.tp_rank, _UINT32_MAX, "issuer tp_rank")
        _require_uint(self.issuer.tp_size, _UINT32_MAX, "issuer tp_size")
        return _TERMINAL_WIRE_RECEIPT.pack(
            TERMINAL_WIRE_RECEIPT_MAGIC,
            self.binding.request_key.room_id,
            self.binding.request_key.request_generation,
            target.process_generation,
            int(_ROLE_TO_CODE[target.role]),
            target.tp_rank,
            target.tp_size,
            self.binding.rank_manifest_digest,
            self.binding.allocation_digest,
            self.issuer.process_generation,
            int(_ROLE_TO_CODE[self.issuer.role]),
            self.issuer.tp_rank,
            self.issuer.tp_size,
            int(_KIND_TO_CODE[self.kind]),
            int(_OUTCOME_TO_CODE[self.outcome]),
            self.terminal_timestamp_ns,
            self.receipt_nonce,
        )

    @classmethod
    def decode(cls, payload: bytes) -> "TerminalWireReceipt":
        """Decode and validate one canonical receipt payload.

        :param payload: Untrusted fixed-width payload.
        :returns: Validated immutable wire receipt.
        :raises TerminalWireReceiptError: If a code or shape is invalid.
        """

        if type(payload) is not bytes:
            raise TypeError("payload must be bytes")
        if len(payload) != _TERMINAL_WIRE_RECEIPT.size:
            raise TerminalWireReceiptError(
                "terminal receipt payload has an invalid byte length"
            )
        (
            magic,
            room_id,
            request_generation,
            target_process_generation,
            target_role_code,
            target_tp_rank,
            target_tp_size,
            rank_manifest_digest,
            allocation_digest,
            issuer_process_generation,
            issuer_role_code,
            issuer_tp_rank,
            issuer_tp_size,
            kind_code,
            outcome_code,
            terminal_timestamp_ns,
            receipt_nonce,
        ) = _TERMINAL_WIRE_RECEIPT.unpack(payload)
        if magic != TERMINAL_WIRE_RECEIPT_MAGIC:
            raise TerminalWireReceiptError("terminal receipt magic is invalid")
        try:
            target_role = _CODE_TO_ROLE[_TerminalOwnerRoleCode(target_role_code)]
            issuer_role = _CODE_TO_ROLE[_TerminalOwnerRoleCode(issuer_role_code)]
            kind = _CODE_TO_KIND[_TerminalReceiptKindCode(kind_code)]
            outcome = _CODE_TO_OUTCOME[_TerminalReceiptOutcomeCode(outcome_code)]
        except ValueError as error:
            raise TerminalWireReceiptError(
                "terminal receipt contains an unknown enum code"
            ) from error
        return cls(
            binding=TerminalRequestBinding(
                request_key=PackedRequestKey(
                    room_id=room_id,
                    request_generation=request_generation,
                ),
                owner=TerminalProcessIdentity(
                    process_generation=target_process_generation,
                    role=target_role,
                    tp_rank=target_tp_rank,
                    tp_size=target_tp_size,
                ),
                rank_manifest_digest=rank_manifest_digest,
                allocation_digest=allocation_digest,
            ),
            issuer=TerminalProcessIdentity(
                process_generation=issuer_process_generation,
                role=issuer_role,
                tp_rank=issuer_tp_rank,
                tp_size=issuer_tp_size,
            ),
            kind=kind,
            outcome=outcome,
            terminal_timestamp_ns=terminal_timestamp_ns,
            receipt_nonce=receipt_nonce,
        )

    @property
    def digest(self) -> bytes:
        """Return the canonical wire-receipt digest.

        :returns: SHA-256 over the fixed-width representation.
        """

        return hashlib.sha256(self.encode()).digest()


@dataclasses.dataclass(frozen=True, slots=True)
class IssuedTerminalWireReceipt:
    """Local authority and matching transport representation.

    :ivar local_receipt: Process-local one-shot authority.
    :ivar wire_receipt: Authenticated-route representation for remote owners.
    """

    local_receipt: TerminalReceipt
    wire_receipt: TerminalWireReceipt

    def __post_init__(self) -> None:
        """Validate the public receipt fields shared by both forms."""

        if type(self.local_receipt) is not TerminalReceipt:
            raise TypeError("local_receipt must be TerminalReceipt")
        if type(self.wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        wire = self.wire_receipt
        local = self.local_receipt
        if (
            wire.binding != local.binding
            or wire.kind is not local.kind
            or wire.outcome is not local.outcome
            or wire.terminal_timestamp_ns != local.terminal_timestamp_ns
        ):
            raise ValueError("local and wire receipt fields differ")


class TerminalWireReceiptIssuer:
    """Process-bound issuer for local and transported receipt authority."""

    _identity: TerminalProcessIdentity
    _local_issuer: TerminalReceiptIssuer
    _wire_nonces: set[bytes]
    _lock: threading.Lock

    def __init__(self, identity: TerminalProcessIdentity) -> None:
        """Initialize one exact process-local wire issuer.

        :param identity: Authenticated process identity placed on every receipt.
        """

        if type(identity) is not TerminalProcessIdentity:
            raise TypeError("identity must be TerminalProcessIdentity")
        self._identity = identity
        self._local_issuer = TerminalReceiptIssuer()
        self._wire_nonces = set()
        self._lock = threading.Lock()

    @property
    def identity(self) -> TerminalProcessIdentity:
        """Return the exact public issuer identity.

        :returns: Process identity bound to this issuer.
        """

        return self._identity

    @property
    def authority(self) -> TerminalReceiptAuthority:
        """Return the matching process-local receipt authority.

        :returns: Authority suitable for a local receipt ledger.
        """

        return self._local_issuer.authority

    def issue(
        self,
        binding: TerminalRequestBinding,
        kind: TerminalReceiptKind,
        outcome: TerminalReceiptOutcome,
        terminal_timestamp_ns: int,
    ) -> IssuedTerminalWireReceipt:
        """Mint matching local and wire one-shot receipts.

        :param binding: Exact target owner and request binding.
        :param kind: Authority represented by the receipt.
        :param outcome: Terminal outcome represented by the receipt.
        :param terminal_timestamp_ns: Issuer-local monotonic timestamp.
        :returns: Joined local and wire receipt forms.
        """

        local_receipt = self._local_issuer.issue(
            binding=binding,
            kind=kind,
            outcome=outcome,
            terminal_timestamp_ns=terminal_timestamp_ns,
        )
        while True:
            wire_nonce = secrets.token_bytes(TERMINAL_RECEIPT_NONCE_BYTES)
            with self._lock:
                if wire_nonce in self._wire_nonces:
                    continue
                self._wire_nonces.add(wire_nonce)
                break
        return IssuedTerminalWireReceipt(
            local_receipt=local_receipt,
            wire_receipt=TerminalWireReceipt(
                binding=binding,
                issuer=self._identity,
                kind=kind,
                outcome=outcome,
                terminal_timestamp_ns=terminal_timestamp_ns,
                receipt_nonce=wire_nonce,
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _ImportedReceipt:
    """Exact imported wire value and its stable local authority."""

    wire_receipt: TerminalWireReceipt
    local_receipt: TerminalReceipt


class TerminalWireReceiptImportNamespace:
    """Authenticated-route importer for one exact remote process identity.

    A binding must be registered while its request generation is live. Retiring
    the binding removes its replay cache; any later packet is rejected because
    the generation is no longer active. This bounds memory without weakening
    replay protection for a live request.
    """

    _remote_issuer: TerminalProcessIdentity
    _local_issuer: TerminalReceiptIssuer
    _active_bindings: dict[bytes, TerminalRequestBinding]
    _imported: dict[tuple[bytes, bytes], _ImportedReceipt]
    _lock: threading.Lock

    def __init__(self, remote_issuer: TerminalProcessIdentity) -> None:
        """Initialize one authenticated remote-issuer namespace.

        :param remote_issuer: Exact identity authenticated by the control route.
        """

        if type(remote_issuer) is not TerminalProcessIdentity:
            raise TypeError("remote_issuer must be TerminalProcessIdentity")
        self._remote_issuer = remote_issuer
        self._local_issuer = TerminalReceiptIssuer()
        self._active_bindings = {}
        self._imported = {}
        self._lock = threading.Lock()

    @property
    def remote_issuer(self) -> TerminalProcessIdentity:
        """Return the only remote identity accepted by this namespace.

        :returns: Exact authenticated remote issuer.
        """

        return self._remote_issuer

    @property
    def authority(self) -> TerminalReceiptAuthority:
        """Return the local authority representing validated imports.

        :returns: Authority suitable for a local receipt ledger.
        """

        return self._local_issuer.authority

    @property
    def active_binding_count(self) -> int:
        """Return the exact live binding count.

        :returns: Number of request generations accepting imports.
        """

        with self._lock:
            return len(self._active_bindings)

    @property
    def imported_receipt_count(self) -> int:
        """Return the exact retained replay-entry count.

        :returns: Number of receipts imported for live bindings.
        """

        with self._lock:
            return len(self._imported)

    def register_binding(self, binding: TerminalRequestBinding) -> None:
        """Register one target request generation before network admission.

        :param binding: Exact live target binding.
        :raises TerminalWireReceiptError: If the digest is already bound to a
            different value.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        digest = binding.digest
        with self._lock:
            current = self._active_bindings.get(digest)
            if current is None:
                self._active_bindings[digest] = binding
                return
            if current != binding:
                raise TerminalWireReceiptError(
                    "binding digest is already registered to another identity"
                )

    def import_receipt(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> TerminalReceipt:
        """Import a receipt only after exact route authentication.

        :param wire_receipt: Decoded untrusted receipt value.
        :param authenticated_issuer: Sender identity proved by the control route.
        :returns: Stable process-local authority for the wire receipt.
        :raises TerminalWireReceiptError: If identity, liveness, or replay
            invariants fail.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        if authenticated_issuer != self._remote_issuer:
            raise TerminalWireReceiptError("control route authenticated another issuer")
        if wire_receipt.issuer != authenticated_issuer:
            raise TerminalWireReceiptError(
                "wire receipt issuer differs from the authenticated route"
            )

        binding_digest = wire_receipt.binding.digest
        key = (binding_digest, wire_receipt.receipt_nonce)
        with self._lock:
            active_binding = self._active_bindings.get(binding_digest)
            if active_binding is None or active_binding != wire_receipt.binding:
                raise TerminalWireReceiptError(
                    "wire receipt targets an inactive request binding"
                )
            imported = self._imported.get(key)
            if imported is not None:
                if imported.wire_receipt != wire_receipt:
                    raise TerminalWireReceiptError(
                        "wire receipt nonce was reused with conflicting fields"
                    )
                return imported.local_receipt

            local_receipt = self._local_issuer.issue(
                binding=wire_receipt.binding,
                kind=wire_receipt.kind,
                outcome=wire_receipt.outcome,
                terminal_timestamp_ns=wire_receipt.terminal_timestamp_ns,
            )
            self._imported[key] = _ImportedReceipt(
                wire_receipt=wire_receipt,
                local_receipt=local_receipt,
            )
            return local_receipt

    def retire_binding(self, binding: TerminalRequestBinding) -> None:
        """Stop admission and release replay state for one terminal binding.

        :param binding: Exact request binding reaching terminal local ownership.
        :raises TerminalWireReceiptError: If the binding is not active.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        digest = binding.digest
        with self._lock:
            current = self._active_bindings.get(digest)
            if current is None or current != binding:
                raise TerminalWireReceiptError("request binding is not active")
            del self._active_bindings[digest]
            stale_keys = tuple(key for key in self._imported if key[0] == digest)
            for key in stale_keys:
                del self._imported[key]
