import dataclasses
import enum
import secrets
import threading

from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalRequestBinding,
)

TERMINAL_RECEIPT_NONCE_BYTES = 16

_RECEIPT_CONSTRUCTION_SEAL = object()
_AUTHORITY_CONSTRUCTION_SEAL = object()


class TerminalReceiptError(RuntimeError):
    """Terminal authority receipt invariant violation."""


class TerminalReceiptKind(enum.StrEnum):
    """Cross-owner and scheduler authority carried by a receipt."""

    ADOPTION_READY = "adoption_ready"
    METADATA_CONSUMED = "metadata_consumed"
    LOCAL_DECODE_READY = "local_decode_ready"
    REQUEST_READY = "request_ready"
    RECLAIM_AUTHORIZED = "reclaim_authorized"
    RECLAIM_CONSUMED = "reclaim_consumed"
    GATEWAY_PUBLISHED = "gateway_published"
    FAILURE = "failure"


class TerminalReceiptOutcome(enum.StrEnum):
    """Terminal outcome authenticated by an authority receipt."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class TerminalReceipt:
    """Immutable one-shot request authority minted by a trusted issuer.

    :ivar binding: Exact request, owner, rank, and allocation binding.
    :ivar kind: Authority represented by this receipt.
    :ivar outcome: Authenticated terminal outcome.
    :ivar terminal_timestamp_ns: Issuer-local monotonic terminal timestamp.
    """

    binding: TerminalRequestBinding
    kind: TerminalReceiptKind
    outcome: TerminalReceiptOutcome
    terminal_timestamp_ns: int
    _issuer_nonce: object = dataclasses.field(repr=False, compare=False)
    _receipt_nonce: bytes = dataclasses.field(repr=False)

    def __init__(
        self,
        binding: TerminalRequestBinding,
        kind: TerminalReceiptKind,
        outcome: TerminalReceiptOutcome,
        terminal_timestamp_ns: int,
        issuer_nonce: object,
        receipt_nonce: bytes,
        construction_seal: object,
    ) -> None:
        """Construct one issuer-owned authority receipt.

        :param binding: Exact request-local binding.
        :param kind: Authority represented by this receipt.
        :param outcome: Authenticated terminal outcome.
        :param terminal_timestamp_ns: Issuer-local monotonic timestamp.
        :param issuer_nonce: Private issuer identity.
        :param receipt_nonce: Private one-shot nonce.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _RECEIPT_CONSTRUCTION_SEAL:
            raise TypeError("terminal receipts are issuer owned")
        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(kind) is not TerminalReceiptKind:
            raise TypeError("kind must be TerminalReceiptKind")
        if type(outcome) is not TerminalReceiptOutcome:
            raise TypeError("outcome must be TerminalReceiptOutcome")
        if type(terminal_timestamp_ns) is not int or terminal_timestamp_ns < 0:
            raise ValueError("terminal_timestamp_ns must be a non-negative integer")
        if type(receipt_nonce) is not bytes:
            raise TypeError("receipt nonce must be bytes")
        if len(receipt_nonce) != TERMINAL_RECEIPT_NONCE_BYTES:
            raise ValueError(
                f"receipt nonce must contain {TERMINAL_RECEIPT_NONCE_BYTES} bytes"
            )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "terminal_timestamp_ns", terminal_timestamp_ns)
        object.__setattr__(self, "_issuer_nonce", issuer_nonce)
        object.__setattr__(self, "_receipt_nonce", receipt_nonce)


TerminalReceiptToken = tuple[object, bytes]


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class TerminalReceiptAuthority:
    """Opaque authority identifying one trusted receipt issuer."""

    _issuer_nonce: object = dataclasses.field(repr=False)

    def __init__(self, issuer_nonce: object, construction_seal: object) -> None:
        """Construct one issuer-owned authority.

        :param issuer_nonce: Private issuer identity.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _AUTHORITY_CONSTRUCTION_SEAL:
            raise TypeError("terminal receipt authorities are issuer owned")
        object.__setattr__(self, "_issuer_nonce", issuer_nonce)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalReceiptLedger:
    """Immutable trusted-issuer set and one-shot receipt-consumption ledger.

    :ivar authorities: Receipt issuers trusted by one lifecycle owner.
    :ivar consumed_tokens: Receipt identities already consumed by that owner.
    """

    authorities: frozenset[TerminalReceiptAuthority]
    consumed_tokens: frozenset[TerminalReceiptToken] = dataclasses.field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        """Validate one immutable receipt ledger."""

        if type(self.authorities) is not frozenset:
            raise TypeError("authorities must be a frozenset")
        if len(self.authorities) == 0:
            raise ValueError("at least one trusted receipt authority is required")
        if any(
            type(authority) is not TerminalReceiptAuthority
            for authority in self.authorities
        ):
            raise TypeError("authorities must contain TerminalReceiptAuthority values")
        if type(self.consumed_tokens) is not frozenset:
            raise TypeError("consumed_tokens must be a frozenset")

    def consume(
        self,
        receipt: TerminalReceipt,
        binding: TerminalRequestBinding,
        kind: TerminalReceiptKind,
        outcome: TerminalReceiptOutcome = TerminalReceiptOutcome.SUCCESS,
    ) -> "TerminalReceiptLedger":
        """Validate and consume one exact receipt without mutating this ledger.

        :param receipt: Candidate authority receipt.
        :param binding: Exact request-local binding.
        :param kind: Required authority kind.
        :param outcome: Required terminal outcome.
        :returns: A new ledger containing the consumed one-shot identity.
        :raises TerminalReceiptError: If the receipt is untrusted or replayed.
        """

        validate_terminal_receipt(receipt, binding, kind, outcome)
        token = terminal_receipt_token(receipt)
        issuer_nonce = token[0]
        trusted = any(
            authority._issuer_nonce is issuer_nonce for authority in self.authorities
        )
        if not trusted:
            raise TerminalReceiptError("receipt was minted by an untrusted issuer")
        if token in self.consumed_tokens:
            raise TerminalReceiptError("receipt authority was already consumed")
        return dataclasses.replace(
            self,
            consumed_tokens=self.consumed_tokens | frozenset((token,)),
        )


class TerminalReceiptIssuer:
    """Process-local namespace which mints immutable one-shot receipts."""

    _issuer_nonce: object
    _authority: TerminalReceiptAuthority
    _issued_nonces: set[bytes]
    _lock: threading.Lock

    def __init__(self) -> None:
        """Initialize one independent receipt namespace."""

        self._issuer_nonce = object()
        self._authority = TerminalReceiptAuthority(
            issuer_nonce=self._issuer_nonce,
            construction_seal=_AUTHORITY_CONSTRUCTION_SEAL,
        )
        self._issued_nonces = set()
        self._lock = threading.Lock()

    @property
    def authority(self) -> TerminalReceiptAuthority:
        """Return the opaque authority used to trust this issuer.

        :returns: Process-local receipt authority.
        """

        return self._authority

    def issue(
        self,
        binding: TerminalRequestBinding,
        kind: TerminalReceiptKind,
        outcome: TerminalReceiptOutcome,
        terminal_timestamp_ns: int,
    ) -> TerminalReceipt:
        """Mint one request-local authority receipt.

        :param binding: Exact request-local binding.
        :param kind: Authority represented by the receipt.
        :param outcome: Authenticated terminal outcome.
        :param terminal_timestamp_ns: Issuer-local monotonic timestamp.
        :returns: Immutable one-shot receipt.
        """

        while True:
            receipt_nonce = secrets.token_bytes(TERMINAL_RECEIPT_NONCE_BYTES)
            with self._lock:
                if receipt_nonce in self._issued_nonces:
                    continue
                self._issued_nonces.add(receipt_nonce)
                break
        return TerminalReceipt(
            binding=binding,
            kind=kind,
            outcome=outcome,
            terminal_timestamp_ns=terminal_timestamp_ns,
            issuer_nonce=self._issuer_nonce,
            receipt_nonce=receipt_nonce,
            construction_seal=_RECEIPT_CONSTRUCTION_SEAL,
        )


def validate_terminal_receipt(
    receipt: TerminalReceipt,
    binding: TerminalRequestBinding,
    kind: TerminalReceiptKind,
    outcome: TerminalReceiptOutcome = TerminalReceiptOutcome.SUCCESS,
) -> None:
    """Validate one receipt against its exact expected authority.

    :param receipt: Candidate receipt.
    :param binding: Exact request-local binding.
    :param kind: Required authority kind.
    :param outcome: Required terminal outcome.
    :raises TerminalReceiptError: If any authority field differs.
    """

    if type(receipt) is not TerminalReceipt:
        raise TypeError("receipt must be TerminalReceipt")
    if receipt.binding != binding:
        raise TerminalReceiptError("receipt belongs to another request binding")
    if receipt.kind is not kind:
        raise TerminalReceiptError(
            f"receipt kind {receipt.kind.value} does not authorize {kind.value}"
        )
    if receipt.outcome is not outcome:
        raise TerminalReceiptError(
            f"receipt outcome {receipt.outcome.value} does not authorize "
            f"{outcome.value}"
        )


def terminal_receipt_token(receipt: TerminalReceipt) -> TerminalReceiptToken:
    """Return the private replay identity of one trusted receipt.

    :param receipt: Exact receipt minted by a trusted issuer.
    :returns: Process-local issuer and one-shot nonce identity.
    """

    if type(receipt) is not TerminalReceipt:
        raise TypeError("receipt must be TerminalReceipt")
    return (receipt._issuer_nonce, receipt._receipt_nonce)
