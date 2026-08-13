import dataclasses
import enum
import hashlib

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_DIGEST_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PackedRequestKey,
)

TERMINAL_PROCESS_GENERATION_BYTES = PACKED_REQUEST_GENERATION_BYTES
TERMINAL_PUBLICATION_GENERATION_BYTES = PACKED_REQUEST_GENERATION_BYTES
TERMINAL_BINDING_DIGEST_BYTES = PACKED_REQUEST_DIGEST_BYTES


def _require_exact_bytes(value: bytes, length: int, label: str) -> None:
    """Validate one fixed-width identity.

    :param value: Candidate identity.
    :param length: Required byte count.
    :param label: Reader-facing field label.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} bytes, got {len(value)}")


def _digest_fields(domain: bytes, fields: tuple[bytes, ...]) -> bytes:
    """Hash length-delimited fields under one protocol domain.

    :param domain: Protocol-specific digest domain.
    :param fields: Ordered byte fields.
    :returns: SHA-256 digest.
    """

    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for field in fields:
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)
    return digest.digest()


class TerminalOwnerRole(enum.StrEnum):
    """Packed terminal owner role within one disaggregated request."""

    SOURCE = "source"
    DECODE = "decode"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalProcessIdentity:
    """Exact process and tensor-parallel rank owning one local state machine.

    :ivar process_generation: Process incarnation preventing restart replay.
    :ivar role: Source or decode owner role.
    :ivar tp_rank: Local tensor-parallel rank.
    :ivar tp_size: Tensor-parallel width of the local replica.
    """

    process_generation: bytes
    role: TerminalOwnerRole
    tp_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        """Validate one process-local owner identity."""

        _require_exact_bytes(
            self.process_generation,
            TERMINAL_PROCESS_GENERATION_BYTES,
            "process_generation",
        )
        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        if type(self.tp_size) is not int or self.tp_size <= 0:
            raise ValueError("tp_size must be a positive integer")
        if (
            type(self.tp_rank) is not int
            or self.tp_rank < 0
            or self.tp_rank >= self.tp_size
        ):
            raise ValueError("tp_rank must identify one rank within tp_size")

    @property
    def digest(self) -> bytes:
        """Return the canonical owner-identity digest.

        :returns: SHA-256 owner digest.
        """

        return _digest_fields(
            b"sglang.packed-terminal.process-identity.v1",
            (
                self.process_generation,
                self.role.value.encode("ascii"),
                self.tp_rank.to_bytes(4, "big"),
                self.tp_size.to_bytes(4, "big"),
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRequestBinding:
    """Immutable identity carried by every request-local authority receipt.

    :ivar request_key: Stable packed request key.
    :ivar owner: Process-local owner identity.
    :ivar rank_manifest_digest: Exact participating-rank manifest digest.
    :ivar allocation_digest: Exact request allocation digest.
    """

    request_key: PackedRequestKey
    owner: TerminalProcessIdentity
    rank_manifest_digest: bytes
    allocation_digest: bytes

    def __post_init__(self) -> None:
        """Validate one complete request-local authority binding."""

        if type(self.request_key) is not PackedRequestKey:
            raise TypeError("request_key must be PackedRequestKey")
        if type(self.owner) is not TerminalProcessIdentity:
            raise TypeError("owner must be TerminalProcessIdentity")
        _require_exact_bytes(
            self.rank_manifest_digest,
            PACKED_REQUEST_DIGEST_BYTES,
            "rank_manifest_digest",
        )
        _require_exact_bytes(
            self.allocation_digest,
            PACKED_REQUEST_DIGEST_BYTES,
            "allocation_digest",
        )

    @property
    def digest(self) -> bytes:
        """Return the canonical request-binding digest.

        :returns: SHA-256 binding digest.
        """

        return _digest_fields(
            b"sglang.packed-terminal.request-binding.v1",
            (
                self.request_key.room_id.to_bytes(8, "big"),
                self.request_key.request_generation,
                self.owner.digest,
                self.rank_manifest_digest,
                self.allocation_digest,
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalPublicationIdentity:
    """Exactly-once gateway publication identity.

    :ivar request_key: Stable packed request key.
    :ivar publisher_process_generation: Canonical publisher process incarnation.
    :ivar publication_generation: Request-local publication generation.
    """

    request_key: PackedRequestKey
    publisher_process_generation: bytes
    publication_generation: bytes

    def __post_init__(self) -> None:
        """Validate one gateway publication identity."""

        if type(self.request_key) is not PackedRequestKey:
            raise TypeError("request_key must be PackedRequestKey")
        _require_exact_bytes(
            self.publisher_process_generation,
            TERMINAL_PROCESS_GENERATION_BYTES,
            "publisher_process_generation",
        )
        _require_exact_bytes(
            self.publication_generation,
            TERMINAL_PUBLICATION_GENERATION_BYTES,
            "publication_generation",
        )

    @property
    def digest(self) -> bytes:
        """Return the canonical publication-identity digest.

        :returns: SHA-256 publication digest.
        """

        return _digest_fields(
            b"sglang.packed-terminal.publication-identity.v1",
            (
                self.request_key.room_id.to_bytes(8, "big"),
                self.request_key.request_generation,
                self.publisher_process_generation,
                self.publication_generation,
            ),
        )
