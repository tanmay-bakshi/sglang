import dataclasses
import enum
from collections.abc import Mapping

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    PACKED_TERMINAL_DEADLINES,
    TerminalDeadlineKind,
    terminal_deadline_table_digest,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalPublicationIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TERMINAL_RECEIPT_NONCE_BYTES,
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

_DIGEST_BYTES = 32
_GENERATION_BYTES = 16
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _require_exact_bytes(value: bytes, length: int, label: str) -> None:
    """Validate one fixed-width native ABI field.

    :param value: Candidate bytes.
    :param length: Required byte count.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} bytes")


def _require_uint(value: int, maximum: int, label: str) -> None:
    """Validate one fixed-width unsigned ABI integer.

    :param value: Candidate integer.
    :param maximum: Inclusive maximum value.
    :param label: Reader-facing field name.
    """

    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{label} must be an unsigned integer at most {maximum}")


class NativeTerminalOwnerRole(enum.IntEnum):
    """Stable native code for a terminal lifecycle owner role."""

    SOURCE = 1
    DECODE = 2


class NativeTerminalProducerClass(enum.IntEnum):
    """Stable native producer authority class."""

    LOCAL = 1
    RECEIPT = 2
    CONTROL = 3
    QUALIFICATION = 4


class NativeTerminalOwnerEventKind(enum.IntEnum):
    """Closed producer vocabulary reduced by the native owner."""

    SOURCE_SUBMISSION_ACCEPTED = 10
    SOURCE_PRODUCER_COMPLETED = 11
    SOURCE_GATHER_POSTED = 12
    SOURCE_NATIVE_TERMINAL = 13
    SOURCE_OUTCOMES_SENT = 14
    SOURCE_TEARDOWN_RECEIVED = 15
    SOURCE_ACK_SENT = 16
    SOURCE_REQUEST_READY = 17
    SOURCE_RECLAIM_CONSUMED = 18
    SOURCE_GATEWAY_PUBLISHED = 19
    SOURCE_PUBLICATION_FAILED = 20
    SOURCE_REQUEST_FAILED = 21
    SOURCE_OWNER_DIED = 22
    SOURCE_PUBLISHER_DIED = 23
    SOURCE_INBOX_OVERFLOW = 24
    DECODE_ALLOCATION_PUBLISHED = 40
    DECODE_WRITER_AGGREGATION_STARTED = 41
    DECODE_WRITER_MANIFEST_COMPLETED = 42
    DECODE_SCATTER_STARTED = 43
    DECODE_SCATTER_TERMINAL = 44
    DECODE_TEARDOWN_SENT = 45
    DECODE_ACK_AGGREGATION_STARTED = 46
    DECODE_ACK_MANIFEST_COMPLETED = 47
    DECODE_ADOPTION_CONSUMED = 48
    DECODE_METADATA_CONSUMED = 49
    DECODE_LOCAL_READY_ISSUED = 50
    DECODE_REQUEST_READY = 51
    DECODE_CANCEL_UNPUBLISHED = 52
    DECODE_REQUEST_FAILED = 53
    DECODE_OWNER_DIED = 54
    DECODE_INBOX_OVERFLOW = 55


class NativeTerminalReceiptKind(enum.IntEnum):
    """Stable native code for receipt authority."""

    ADOPTION_READY = 1
    METADATA_CONSUMED = 2
    LOCAL_DECODE_READY = 3
    REQUEST_READY = 4
    RECLAIM_AUTHORIZED = 5
    RECLAIM_CONSUMED = 6
    GATEWAY_PUBLISHED = 7
    FAILURE = 8


class NativeTerminalReceiptOutcome(enum.IntEnum):
    """Stable native code for a receipt outcome."""

    SUCCESS = 1
    FAILURE = 2
    CANCELLED = 3


class NativeTerminalDeadlineKind(enum.IntEnum):
    """Stable native code for every hash-bound owner deadline."""

    EXISTING_NIXL_CAPABILITY_READY = 1
    EXISTING_PACKED_CONTROL = 2
    OWNER_PRODUCER_AND_GATHER = 3
    OWNER_NATIVE_TRANSFER = 4
    OWNER_DECODE_SCATTER = 5
    OWNER_TEARDOWN_ACK = 6
    OWNER_REQUEST_GLOBAL_READY = 7
    OWNER_SCHEDULER_RECEIPT_CONSUMPTION = 8
    OWNER_GATEWAY_PUBLICATION = 9
    OWNER_SHUTDOWN_DRAIN = 10


class NativeTerminalResource(enum.IntFlag):
    """Stable resource bits conserved by the native lifecycle owner."""

    SOURCE_KV_PAGES = 1 << 0
    SOURCE_PRODUCER_RESULTS = 1 << 1
    SOURCE_PACKED_LANE = 1 << 2
    SOURCE_REMOTE_REGISTRATIONS = 1 << 3
    SOURCE_NIXL_HANDLES = 1 << 4
    SOURCE_RESULT_SLOT = 1 << 5
    SOURCE_REQUEST_IDENTITY = 1 << 6
    SOURCE_DFLASH_BOUNDARY_VRAM_ROWS = 1 << 7
    PUBLICATION_IDENTITY = 1 << 8
    DECODE_FULL_PAGES = 1 << 9
    DECODE_SWA_PAGES = 1 << 10
    DECODE_REQUEST_SLOT = 1 << 11
    DECODE_STAGING_LEASE = 1 << 12
    DECODE_DFLASH_BOUNDARY_VRAM_ROWS = 1 << 13
    DECODE_WRITER_STATE = 1 << 14
    DECODE_NATIVE_IDENTITY = 1 << 15


NATIVE_SOURCE_RECLAIMABLE_MASK = int(
    NativeTerminalResource.SOURCE_KV_PAGES
    | NativeTerminalResource.SOURCE_PRODUCER_RESULTS
    | NativeTerminalResource.SOURCE_PACKED_LANE
    | NativeTerminalResource.SOURCE_REMOTE_REGISTRATIONS
    | NativeTerminalResource.SOURCE_NIXL_HANDLES
    | NativeTerminalResource.SOURCE_RESULT_SLOT
    | NativeTerminalResource.SOURCE_REQUEST_IDENTITY
    | NativeTerminalResource.SOURCE_DFLASH_BOUNDARY_VRAM_ROWS
)
NATIVE_SOURCE_RESOURCE_MASK = int(
    NativeTerminalResource(NATIVE_SOURCE_RECLAIMABLE_MASK)
    | NativeTerminalResource.PUBLICATION_IDENTITY
)
NATIVE_DECODE_RESOURCE_MASK = int(
    NativeTerminalResource.DECODE_FULL_PAGES
    | NativeTerminalResource.DECODE_SWA_PAGES
    | NativeTerminalResource.DECODE_REQUEST_SLOT
    | NativeTerminalResource.DECODE_STAGING_LEASE
    | NativeTerminalResource.DECODE_DFLASH_BOUNDARY_VRAM_ROWS
    | NativeTerminalResource.DECODE_WRITER_STATE
    | NativeTerminalResource.DECODE_NATIVE_IDENTITY
)


class NativeSourceLifecyclePhase(enum.IntEnum):
    """Stable native source lifecycle phases."""

    FROZEN = 1
    WAITING_FOR_PRODUCER = 2
    GATHERING = 3
    NATIVE_IN_FLIGHT = 4
    LOCAL_TRANSFER_TERMINAL = 5
    OUTCOMES_SENT = 6
    TEARDOWN_RECEIVED = 7
    ACK_SENT = 8
    REQUEST_READY_RECEIVED = 9
    PUBLICATION_QUARANTINED = 10
    RETIRED = 11
    QUARANTINED = 12


class NativeDecodeLifecyclePhase(enum.IntEnum):
    """Stable native decode lifecycle phases."""

    PREPARED = 1
    PUBLISHED = 2
    WRITER_AGGREGATING = 3
    SCATTER_READY = 4
    SCATTER_IN_FLIGHT = 5
    SCATTER_TERMINAL = 6
    TEARDOWN_SENT = 7
    ACK_AGGREGATING = 8
    ADOPTION_READY = 9
    ADOPTED_BY_SCHEDULER = 10
    METADATA_CONSUMED = 11
    LOCAL_DECODE_READY = 12
    REQUEST_READY = 13
    RETIRED = 14
    QUARANTINED = 15


class NativeTerminalOwnerActionKind(enum.IntEnum):
    """Immutable side effects earned by one native state commit."""

    RECLAIM_AUTHORIZED = 1
    ADOPTION_READY = 2
    LOCAL_DECODE_READY = 3
    REQUEST_RETIRED = 4
    REQUEST_QUARANTINED = 5
    PROCESS_FATAL = 6
    SOURCE_GATHER_READY = 7
    SOURCE_OUTCOME_READY = 8
    SOURCE_ACK_READY = 9
    DECODE_SCATTER_READY = 10
    DECODE_TEARDOWN_READY = 11
    GATEWAY_PUBLICATION_READY = 12


class NativeTerminalOwnerFatalCode(enum.IntEnum):
    """Sticky native owner failures which stop process continuation."""

    NONE = 0
    INPUT_QUEUE_OVERFLOW = 1
    OUTPUT_QUEUE_OVERFLOW = 2
    EVENTFD_FAILURE = 3
    TIMERFD_FAILURE = 4
    PRODUCER_SEQUENCE = 5
    DUPLICATE_BINDING = 6
    UNKNOWN_BINDING = 7
    ILLEGAL_TRANSITION = 8
    RECEIPT_AUTHORITY = 9
    RECEIPT_REPLAY = 10
    UNKNOWN_ACTION = 11
    ACTION_REPLAY = 12
    DEADLINE_EXPIRY = 13
    DEPENDENCY_DEATH = 14
    CLOSE_WITH_RETAINED_INVENTORY = 15
    INTERNAL_ERROR = 16


_ROLE_TO_NATIVE = {
    TerminalOwnerRole.SOURCE: NativeTerminalOwnerRole.SOURCE,
    TerminalOwnerRole.DECODE: NativeTerminalOwnerRole.DECODE,
}
_NATIVE_TO_ROLE = {value: key for key, value in _ROLE_TO_NATIVE.items()}
_RECEIPT_KIND_TO_NATIVE = {
    TerminalReceiptKind.ADOPTION_READY: NativeTerminalReceiptKind.ADOPTION_READY,
    TerminalReceiptKind.METADATA_CONSUMED: (
        NativeTerminalReceiptKind.METADATA_CONSUMED
    ),
    TerminalReceiptKind.LOCAL_DECODE_READY: (
        NativeTerminalReceiptKind.LOCAL_DECODE_READY
    ),
    TerminalReceiptKind.REQUEST_READY: NativeTerminalReceiptKind.REQUEST_READY,
    TerminalReceiptKind.RECLAIM_AUTHORIZED: (
        NativeTerminalReceiptKind.RECLAIM_AUTHORIZED
    ),
    TerminalReceiptKind.RECLAIM_CONSUMED: (NativeTerminalReceiptKind.RECLAIM_CONSUMED),
    TerminalReceiptKind.GATEWAY_PUBLISHED: (
        NativeTerminalReceiptKind.GATEWAY_PUBLISHED
    ),
    TerminalReceiptKind.FAILURE: NativeTerminalReceiptKind.FAILURE,
}
_RECEIPT_OUTCOME_TO_NATIVE = {
    TerminalReceiptOutcome.SUCCESS: NativeTerminalReceiptOutcome.SUCCESS,
    TerminalReceiptOutcome.FAILURE: NativeTerminalReceiptOutcome.FAILURE,
    TerminalReceiptOutcome.CANCELLED: NativeTerminalReceiptOutcome.CANCELLED,
}
_NATIVE_TO_RECEIPT_KIND = {
    value: key for key, value in _RECEIPT_KIND_TO_NATIVE.items()
}
_NATIVE_TO_RECEIPT_OUTCOME = {
    value: key for key, value in _RECEIPT_OUTCOME_TO_NATIVE.items()
}
_DEADLINE_KIND_TO_NATIVE = {
    TerminalDeadlineKind.EXISTING_NIXL_CAPABILITY_READY: (
        NativeTerminalDeadlineKind.EXISTING_NIXL_CAPABILITY_READY
    ),
    TerminalDeadlineKind.EXISTING_PACKED_CONTROL: (
        NativeTerminalDeadlineKind.EXISTING_PACKED_CONTROL
    ),
    TerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER: (
        NativeTerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER
    ),
    TerminalDeadlineKind.OWNER_NATIVE_TRANSFER: (
        NativeTerminalDeadlineKind.OWNER_NATIVE_TRANSFER
    ),
    TerminalDeadlineKind.OWNER_DECODE_SCATTER: (
        NativeTerminalDeadlineKind.OWNER_DECODE_SCATTER
    ),
    TerminalDeadlineKind.OWNER_TEARDOWN_ACK: (
        NativeTerminalDeadlineKind.OWNER_TEARDOWN_ACK
    ),
    TerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY: (
        NativeTerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY
    ),
    TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION: (
        NativeTerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION
    ),
    TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION: (
        NativeTerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION
    ),
    TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN: (
        NativeTerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
    ),
}
_EVENT_RECEIPT_REQUIREMENTS = {
    NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY: (
        NativeTerminalReceiptKind.REQUEST_READY,
        NativeTerminalReceiptOutcome.SUCCESS,
    ),
    NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED: (
        NativeTerminalReceiptKind.RECLAIM_CONSUMED,
        NativeTerminalReceiptOutcome.SUCCESS,
    ),
    NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED: (
        NativeTerminalReceiptKind.GATEWAY_PUBLISHED,
        NativeTerminalReceiptOutcome.SUCCESS,
    ),
    NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED: (
        NativeTerminalReceiptKind.FAILURE,
        NativeTerminalReceiptOutcome.FAILURE,
    ),
    NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED: (
        NativeTerminalReceiptKind.FAILURE,
        NativeTerminalReceiptOutcome.FAILURE,
    ),
    NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED: (
        NativeTerminalReceiptKind.ADOPTION_READY,
        NativeTerminalReceiptOutcome.SUCCESS,
    ),
    NativeTerminalOwnerEventKind.DECODE_REQUEST_READY: (
        NativeTerminalReceiptKind.REQUEST_READY,
        NativeTerminalReceiptOutcome.SUCCESS,
    ),
    NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED: (
        NativeTerminalReceiptKind.FAILURE,
        NativeTerminalReceiptOutcome.FAILURE,
    ),
}
_EVENT_OPTIONAL_LOCAL_FAILURE_RECEIPTS = frozenset(
    (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
        NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
    )
)
_EVENT_REASON_KINDS = frozenset(
    (
        NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED,
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
        NativeTerminalOwnerEventKind.SOURCE_OWNER_DIED,
        NativeTerminalOwnerEventKind.SOURCE_PUBLISHER_DIED,
        NativeTerminalOwnerEventKind.SOURCE_INBOX_OVERFLOW,
        NativeTerminalOwnerEventKind.DECODE_CANCEL_UNPUBLISHED,
        NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
        NativeTerminalOwnerEventKind.DECODE_OWNER_DIED,
        NativeTerminalOwnerEventKind.DECODE_INBOX_OVERFLOW,
    )
)
_ACTION_RECEIPT_REQUIREMENTS = {
    NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED: (
        NativeTerminalReceiptKind.RECLAIM_AUTHORIZED
    ),
    NativeTerminalOwnerActionKind.ADOPTION_READY: (
        NativeTerminalReceiptKind.ADOPTION_READY
    ),
    NativeTerminalOwnerActionKind.LOCAL_DECODE_READY: (
        NativeTerminalReceiptKind.LOCAL_DECODE_READY
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalDeadlineSpec:
    """Hash-bound deadline supplied to the native reactor at construction.

    :ivar kind: Stable native deadline code.
    :ivar duration_ns: Exact positive integer duration.
    :ivar process_fatal: Whether expiry makes process continuation unsafe.
    :ivar starts_at: Frozen one-shot start anchor.
    :ivar timeout_outcome: Frozen fail-closed outcome.
    """

    kind: NativeTerminalDeadlineKind
    duration_ns: int
    process_fatal: bool
    starts_at: str
    timeout_outcome: str

    def __post_init__(self) -> None:
        """Validate one complete native deadline specification."""

        if type(self.kind) is not NativeTerminalDeadlineKind:
            raise TypeError("kind must be NativeTerminalDeadlineKind")
        _require_uint(self.duration_ns, _UINT64_MAX, "duration_ns")
        if self.duration_ns == 0:
            raise ValueError("duration_ns must be positive")
        if type(self.process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")
        strings = (self.starts_at, self.timeout_outcome)
        if any(type(value) is not str or len(value) == 0 for value in strings):
            raise ValueError("deadline anchor and outcome must be non-empty strings")

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible deadline specification.

        :returns: Native deadline fields.
        """

        return {
            "kind": int(self.kind),
            "duration_ns": self.duration_ns,
            "process_fatal": self.process_fatal,
            "starts_at": self.starts_at,
            "timeout_outcome": self.timeout_outcome,
        }


def canonical_native_terminal_deadlines() -> tuple[NativeTerminalDeadlineSpec, ...]:
    """Project the packaged frozen deadline table into the native ABI.

    :returns: Deadline records ordered by stable native code.
    """

    values = tuple(
        NativeTerminalDeadlineSpec(
            kind=_DEADLINE_KIND_TO_NATIVE[spec.kind],
            duration_ns=spec.duration_ns,
            process_fatal=(
                spec.kind
                in (
                    TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
                    TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
                    TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
                )
            ),
            starts_at=spec.starts_at,
            timeout_outcome=spec.timeout_outcome,
        )
        for spec in PACKED_TERMINAL_DEADLINES
    )
    return tuple(sorted(values, key=lambda value: int(value.kind)))


def native_terminal_deadline_table_digest() -> bytes:
    """Return the canonical digest attested beside the native table.

    :returns: Existing packaged deadline-table SHA-256 digest.
    """

    return terminal_deadline_table_digest()


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalProcessIdentity:
    """Fixed-width process identity registered with the native owner.

    :ivar process_generation: Exact process incarnation.
    :ivar role: Source or decode role.
    :ivar tp_rank: Exact local rank.
    :ivar tp_size: Tensor-parallel width.
    :ivar digest: Canonical full process-identity digest.
    """

    process_generation: bytes
    role: NativeTerminalOwnerRole
    tp_rank: int
    tp_size: int
    digest: bytes

    def __post_init__(self) -> None:
        """Validate one complete native process identity."""

        _require_exact_bytes(
            self.process_generation, _GENERATION_BYTES, "process_generation"
        )
        if type(self.role) is not NativeTerminalOwnerRole:
            raise TypeError("role must be NativeTerminalOwnerRole")
        _require_uint(self.tp_rank, _UINT32_MAX, "tp_rank")
        _require_uint(self.tp_size, _UINT32_MAX, "tp_size")
        if self.tp_size == 0 or self.tp_rank >= self.tp_size:
            raise ValueError("tp_rank must identify one rank within tp_size")
        _require_exact_bytes(self.digest, _DIGEST_BYTES, "process identity digest")

    @classmethod
    def from_identity(
        cls, identity: TerminalProcessIdentity
    ) -> "NativeTerminalProcessIdentity":
        """Convert a canonical Python process identity.

        :param identity: Existing exact process identity.
        :returns: Fixed-width native representation.
        """

        if type(identity) is not TerminalProcessIdentity:
            raise TypeError("identity must be TerminalProcessIdentity")
        return cls(
            process_generation=identity.process_generation,
            role=_ROLE_TO_NATIVE[identity.role],
            tp_rank=identity.tp_rank,
            tp_size=identity.tp_size,
            digest=identity.digest,
        )

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible fixed-width record.

        :returns: Native process identity fields.
        """

        return {
            "process_generation": self.process_generation,
            "role": int(self.role),
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "digest": self.digest,
        }

    def to_identity(self) -> TerminalProcessIdentity:
        """Recover the canonical Python identity with digest verification.

        :returns: Exact process identity represented by this native value.
        """

        identity = TerminalProcessIdentity(
            process_generation=self.process_generation,
            role=_NATIVE_TO_ROLE[self.role],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        if identity.digest != self.digest:
            raise ValueError("native process identity digest is inconsistent")
        return identity


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalRequestBinding:
    """Complete request binding retained behind its 32-byte native key.

    :ivar room_id: Stable packed room identity.
    :ivar request_generation: Exact request generation.
    :ivar owner: Exact rank-local owner identity.
    :ivar rank_manifest_digest: Frozen participant manifest.
    :ivar allocation_digest: Frozen resource allocation.
    :ivar digest: Canonical full binding digest used as the native map key.
    """

    room_id: int
    request_generation: bytes
    owner: NativeTerminalProcessIdentity
    rank_manifest_digest: bytes
    allocation_digest: bytes
    digest: bytes

    def __post_init__(self) -> None:
        """Validate one collision-checkable native request binding."""

        _require_uint(self.room_id, _UINT64_MAX, "room_id")
        _require_exact_bytes(
            self.request_generation, _GENERATION_BYTES, "request_generation"
        )
        if type(self.owner) is not NativeTerminalProcessIdentity:
            raise TypeError("owner must be NativeTerminalProcessIdentity")
        _require_exact_bytes(
            self.rank_manifest_digest, _DIGEST_BYTES, "rank_manifest_digest"
        )
        _require_exact_bytes(self.allocation_digest, _DIGEST_BYTES, "allocation_digest")
        _require_exact_bytes(self.digest, _DIGEST_BYTES, "binding digest")

    @classmethod
    def from_binding(
        cls, binding: TerminalRequestBinding
    ) -> "NativeTerminalRequestBinding":
        """Convert one canonical Python request binding.

        :param binding: Existing request binding.
        :returns: Fixed-width native representation.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        return cls(
            room_id=binding.request_key.room_id,
            request_generation=binding.request_key.request_generation,
            owner=NativeTerminalProcessIdentity.from_identity(binding.owner),
            rank_manifest_digest=binding.rank_manifest_digest,
            allocation_digest=binding.allocation_digest,
            digest=binding.digest,
        )

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible fixed-width record.

        :returns: Native request binding fields.
        """

        return {
            "room_id": self.room_id,
            "request_generation": self.request_generation,
            "owner": self.owner.to_native(),
            "rank_manifest_digest": self.rank_manifest_digest,
            "allocation_digest": self.allocation_digest,
            "digest": self.digest,
        }

    def to_binding(self) -> TerminalRequestBinding:
        """Recover the canonical Python binding with digest verification.

        :returns: Exact request binding represented by this native value.
        """

        binding = TerminalRequestBinding(
            request_key=PackedRequestKey(
                room_id=self.room_id,
                request_generation=self.request_generation,
            ),
            owner=self.owner.to_identity(),
            rank_manifest_digest=self.rank_manifest_digest,
            allocation_digest=self.allocation_digest,
        )
        if binding.digest != self.digest:
            raise ValueError("native request binding digest is inconsistent")
        return binding


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalPublicationIdentity:
    """Complete exactly-once publication identity retained by native state.

    :ivar room_id: Stable packed room identity.
    :ivar request_generation: Exact request generation.
    :ivar publisher_process_generation: Canonical publisher incarnation.
    :ivar publication_generation: Exact publication generation.
    :ivar digest: Canonical publication identity digest.
    """

    room_id: int
    request_generation: bytes
    publisher_process_generation: bytes
    publication_generation: bytes
    digest: bytes

    def __post_init__(self) -> None:
        """Validate one complete native publication identity."""

        _require_uint(self.room_id, _UINT64_MAX, "room_id")
        _require_exact_bytes(
            self.request_generation, _GENERATION_BYTES, "request_generation"
        )
        _require_exact_bytes(
            self.publisher_process_generation,
            _GENERATION_BYTES,
            "publisher_process_generation",
        )
        _require_exact_bytes(
            self.publication_generation,
            _GENERATION_BYTES,
            "publication_generation",
        )
        _require_exact_bytes(self.digest, _DIGEST_BYTES, "publication digest")

    @classmethod
    def from_identity(
        cls, identity: TerminalPublicationIdentity
    ) -> "NativeTerminalPublicationIdentity":
        """Convert one canonical Python publication identity.

        :param identity: Existing publication identity.
        :returns: Fixed-width native representation.
        """

        if type(identity) is not TerminalPublicationIdentity:
            raise TypeError("identity must be TerminalPublicationIdentity")
        return cls(
            room_id=identity.request_key.room_id,
            request_generation=identity.request_key.request_generation,
            publisher_process_generation=identity.publisher_process_generation,
            publication_generation=identity.publication_generation,
            digest=identity.digest,
        )

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible publication record.

        :returns: Native publication identity fields.
        """

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalReceipt:
    """Authenticated fixed-width receipt consumed or emitted by native state.

    :ivar binding: Exact target request binding.
    :ivar issuer: Authenticated process identity.
    :ivar kind: One-shot authority kind.
    :ivar outcome: Authenticated terminal outcome.
    :ivar terminal_timestamp_ns: Issuer-local terminal timestamp.
    :ivar nonce: Sixteen-byte one-shot receipt identity.
    """

    binding: NativeTerminalRequestBinding
    issuer: NativeTerminalProcessIdentity
    kind: NativeTerminalReceiptKind
    outcome: NativeTerminalReceiptOutcome
    terminal_timestamp_ns: int
    nonce: bytes

    def __post_init__(self) -> None:
        """Validate one native receipt without weakening wire identity."""

        if type(self.binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        if type(self.issuer) is not NativeTerminalProcessIdentity:
            raise TypeError("issuer must be NativeTerminalProcessIdentity")
        if type(self.kind) is not NativeTerminalReceiptKind:
            raise TypeError("kind must be NativeTerminalReceiptKind")
        if type(self.outcome) is not NativeTerminalReceiptOutcome:
            raise TypeError("outcome must be NativeTerminalReceiptOutcome")
        _require_uint(self.terminal_timestamp_ns, _UINT64_MAX, "terminal_timestamp_ns")
        _require_exact_bytes(self.nonce, TERMINAL_RECEIPT_NONCE_BYTES, "nonce")

    @classmethod
    def from_wire_receipt(cls, receipt: TerminalWireReceipt) -> "NativeTerminalReceipt":
        """Convert a receipt already authenticated by its control route.

        :param receipt: Decoded canonical wire receipt.
        :returns: Fixed-width native receipt.
        """

        if type(receipt) is not TerminalWireReceipt:
            raise TypeError("receipt must be TerminalWireReceipt")
        return cls(
            binding=NativeTerminalRequestBinding.from_binding(receipt.binding),
            issuer=NativeTerminalProcessIdentity.from_identity(receipt.issuer),
            kind=_RECEIPT_KIND_TO_NATIVE[receipt.kind],
            outcome=_RECEIPT_OUTCOME_TO_NATIVE[receipt.outcome],
            terminal_timestamp_ns=receipt.terminal_timestamp_ns,
            nonce=receipt.receipt_nonce,
        )

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible receipt record.

        :returns: Exact native receipt fields.
        """

        return {
            "binding": self.binding.to_native(),
            "issuer": self.issuer.to_native(),
            "kind": int(self.kind),
            "outcome": int(self.outcome),
            "terminal_timestamp_ns": self.terminal_timestamp_ns,
            "nonce": self.nonce,
        }

    def to_wire_receipt(self) -> TerminalWireReceipt:
        """Recover the canonical authenticated-route receipt value.

        :returns: Exact wire receipt represented by this native value.
        """

        return TerminalWireReceipt(
            binding=self.binding.to_binding(),
            issuer=self.issuer.to_identity(),
            kind=_NATIVE_TO_RECEIPT_KIND[self.kind],
            outcome=_NATIVE_TO_RECEIPT_OUTCOME[self.outcome],
            terminal_timestamp_ns=self.terminal_timestamp_ns,
            receipt_nonce=self.nonce,
        )

    @classmethod
    def from_native(cls, value: Mapping[str, object]) -> "NativeTerminalReceipt":
        """Parse one immutable native receipt output.

        :param value: Mapping returned by the native bridge.
        :returns: Validated immutable receipt.
        """

        binding_value = value["binding"]
        issuer_value = value["issuer"]
        if not isinstance(binding_value, Mapping):
            raise TypeError("native receipt binding must be a mapping")
        if not isinstance(issuer_value, Mapping):
            raise TypeError("native receipt issuer must be a mapping")
        return cls(
            binding=_binding_from_native(binding_value),
            issuer=_process_identity_from_native(issuer_value),
            kind=NativeTerminalReceiptKind(int(value["kind"])),
            outcome=NativeTerminalReceiptOutcome(int(value["outcome"])),
            terminal_timestamp_ns=int(value["terminal_timestamp_ns"]),
            nonce=bytes(value["nonce"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerAction:
    """One-shot production action emitted after an authoritative commit.

    :ivar action_id: Gap-free native action identity.
    :ivar kind: Closed consumer action kind.
    :ivar binding: Exact request generation earning the action.
    :ivar commit_timestamp_ns: Native post-commit timestamp.
    :ivar receipt: Newly minted authority when the action transfers permission.
    """

    action_id: int
    kind: NativeTerminalOwnerActionKind
    binding: NativeTerminalRequestBinding
    commit_timestamp_ns: int
    receipt: NativeTerminalReceipt | None

    def __post_init__(self) -> None:
        """Validate one exact action and optional receipt authority."""

        _require_uint(self.action_id, _UINT64_MAX, "action_id")
        if type(self.kind) is not NativeTerminalOwnerActionKind:
            raise TypeError("kind must be NativeTerminalOwnerActionKind")
        if type(self.binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        _require_uint(self.commit_timestamp_ns, _UINT64_MAX, "commit_timestamp_ns")
        expected_receipt_kind = _ACTION_RECEIPT_REQUIREMENTS.get(self.kind)
        if expected_receipt_kind is None:
            if self.receipt is not None:
                raise ValueError(f"{self.kind.name} does not carry a receipt")
            return
        if type(self.receipt) is not NativeTerminalReceipt:
            raise ValueError(f"{self.kind.name} requires a native receipt")
        if (
            self.receipt.binding != self.binding
            or self.receipt.kind is not expected_receipt_kind
            or self.receipt.outcome is not NativeTerminalReceiptOutcome.SUCCESS
            or self.receipt.terminal_timestamp_ns != self.commit_timestamp_ns
        ):
            raise ValueError("native action receipt does not match its authority")

    @classmethod
    def from_native(cls, value: Mapping[str, object]) -> "NativeTerminalOwnerAction":
        """Parse one immutable native action output.

        :param value: Native action mapping.
        :returns: Validated immutable action.
        """

        binding_value = value["binding"]
        receipt_value = value["receipt"]
        if not isinstance(binding_value, Mapping):
            raise TypeError("native action binding must be a mapping")
        receipt: NativeTerminalReceipt | None = None
        if receipt_value is not None:
            if not isinstance(receipt_value, Mapping):
                raise TypeError("native action receipt must be a mapping")
            receipt = NativeTerminalReceipt.from_native(receipt_value)
        return cls(
            action_id=int(value["action_id"]),
            kind=NativeTerminalOwnerActionKind(int(value["kind"])),
            binding=_binding_from_native(binding_value),
            commit_timestamp_ns=int(value["commit_timestamp_ns"]),
            receipt=receipt,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalLifecycleRegistration:
    """Complete lifecycle registration accepted before producer events.

    :ivar binding: Exact source or decode binding.
    :ivar publication_identity: Source publication identity, absent for decode.
    :ivar trusted_issuers: Authenticated receipt issuers accepted by this record.
    """

    binding: NativeTerminalRequestBinding
    publication_identity: NativeTerminalPublicationIdentity | None
    trusted_issuers: tuple[NativeTerminalProcessIdentity, ...]

    def __post_init__(self) -> None:
        """Validate role-specific registration shape and issuer uniqueness."""

        if type(self.binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        if self.binding.owner.role is NativeTerminalOwnerRole.SOURCE:
            if type(self.publication_identity) is not NativeTerminalPublicationIdentity:
                raise ValueError("source registration requires publication identity")
            if (
                self.publication_identity.room_id != self.binding.room_id
                or self.publication_identity.request_generation
                != self.binding.request_generation
            ):
                raise ValueError("publication identity belongs to another request")
        elif self.publication_identity is not None:
            raise ValueError("decode registration cannot carry publication identity")
        if type(self.trusted_issuers) is not tuple:
            raise TypeError("trusted_issuers must be a tuple")
        if any(
            type(issuer) is not NativeTerminalProcessIdentity
            for issuer in self.trusted_issuers
        ):
            raise TypeError("trusted_issuers contain an invalid identity")
        digests = tuple(issuer.digest for issuer in self.trusted_issuers)
        if len(set(digests)) != len(digests):
            raise ValueError("trusted issuer identities must be unique")

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible lifecycle registration.

        :returns: Native registration fields.
        """

        publication = self.publication_identity
        return {
            "binding": self.binding.to_native(),
            "publication_identity": (
                None if publication is None else publication.to_native()
            ),
            "trusted_issuers": tuple(
                issuer.to_native() for issuer in self.trusted_issuers
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalProducerRegistration:
    """Process-lifetime producer identity and route authentication.

    :ivar producer_id: Exact native producer namespace.
    :ivar name: Stable evidence-facing producer name.
    :ivar producer_class: Events this producer is authorized to submit.
    :ivar allowed_role: Lifecycle role this producer may mutate.
    :ivar authenticated_issuer: Process identity proved by this route, if any.
    """

    producer_id: int
    name: str
    producer_class: NativeTerminalProducerClass
    allowed_role: NativeTerminalOwnerRole
    authenticated_issuer: NativeTerminalProcessIdentity | None

    def __post_init__(self) -> None:
        """Validate producer authority before native registration."""

        _require_uint(self.producer_id, _UINT64_MAX, "producer_id")
        if type(self.name) is not str or len(self.name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(self.producer_class) is not NativeTerminalProducerClass:
            raise TypeError("producer_class must be NativeTerminalProducerClass")
        if type(self.allowed_role) is not NativeTerminalOwnerRole:
            raise TypeError("allowed_role must be NativeTerminalOwnerRole")
        if self.authenticated_issuer is not None and (
            type(self.authenticated_issuer) is not NativeTerminalProcessIdentity
        ):
            raise TypeError("authenticated_issuer has an invalid identity")
        if self.producer_class is NativeTerminalProducerClass.LOCAL:
            if self.authenticated_issuer is not None:
                raise ValueError("local native producers cannot assert a wire issuer")
            return
        if self.authenticated_issuer is None:
            raise ValueError("non-local producers require an authenticated issuer")

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible producer registration.

        :returns: Native producer registration fields.
        """

        issuer = self.authenticated_issuer
        return {
            "producer_id": self.producer_id,
            "name": self.name,
            "producer_class": int(self.producer_class),
            "allowed_role": int(self.allowed_role),
            "authenticated_issuer": None if issuer is None else issuer.to_native(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerEvent:
    """Producer-bound event submitted to the authoritative native reducer.

    :ivar producer_id: Registered native producer identity.
    :ivar binding_digest: Exact 32-byte lifecycle lookup key.
    :ivar kind: Closed lifecycle event, never a producer-selected phase.
    :ivar enqueued_ns: Producer-side ``CLOCK_MONOTONIC_RAW`` timestamp.
    :ivar receipt: Authenticated one-shot authority when required.
    :ivar reason: Stable failure evidence when required.
    """

    producer_id: int
    binding_digest: bytes
    kind: NativeTerminalOwnerEventKind
    enqueued_ns: int
    receipt: NativeTerminalReceipt | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate one structurally complete native owner event."""

        _require_uint(self.producer_id, _UINT64_MAX, "producer_id")
        _require_exact_bytes(self.binding_digest, _DIGEST_BYTES, "binding_digest")
        if type(self.kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("kind must be NativeTerminalOwnerEventKind")
        _require_uint(self.enqueued_ns, _UINT64_MAX, "enqueued_ns")
        if self.receipt is not None and type(self.receipt) is not NativeTerminalReceipt:
            raise TypeError("receipt must be NativeTerminalReceipt")
        receipt_requirement = _EVENT_RECEIPT_REQUIREMENTS.get(self.kind)
        if receipt_requirement is None:
            if self.receipt is not None:
                raise ValueError(f"{self.kind.name} does not accept a receipt")
        else:
            if (
                self.receipt is None
                and self.kind not in _EVENT_OPTIONAL_LOCAL_FAILURE_RECEIPTS
            ):
                raise ValueError(f"{self.kind.name} requires a receipt")
            if self.receipt is not None and (
                self.receipt.binding.digest != self.binding_digest
                or self.receipt.kind is not receipt_requirement[0]
                or self.receipt.outcome is not receipt_requirement[1]
            ):
                raise ValueError("receipt does not authorize the exact native event")
        if self.kind in _EVENT_REASON_KINDS:
            if type(self.reason) is not str or len(self.reason) == 0:
                raise ValueError(f"{self.kind.name} requires a non-empty reason")
        elif self.reason is not None:
            raise ValueError(f"{self.kind.name} does not accept a reason")

    def to_native(self) -> dict[str, object]:
        """Return a pybind-compatible event record.

        :returns: Native event fields.
        """

        receipt = self.receipt
        return {
            "producer_id": self.producer_id,
            "binding_digest": self.binding_digest,
            "kind": int(self.kind),
            "enqueued_ns": self.enqueued_ns,
            "receipt": None if receipt is None else receipt.to_native(),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerObservation:
    """Immutable non-authoritative evidence for one actionless native commit.

    This value cannot authorize lifecycle progress. It carries the exact
    source-submission commit interval across the native boundary without
    weakening :class:`NativeTerminalOwnerOutput`'s non-empty action invariant.

    :ivar binding: Exact lifecycle identity observed by the commit.
    :ivar owner_sequence: Process-local authoritative commit sequence.
    :ivar producer_id: Producer namespace which supplied the accepted event.
    :ivar producer_sequence: Accepted producer-local sequence.
    :ivar producer_rank: Source TP rank which owns the producer commit.
    :ivar event_kind: Exact actionless event observed by the native owner.
    :ivar enqueued_ns: Producer-side ``CLOCK_MONOTONIC_RAW`` timestamp.
    :ivar completed_ns: Native commit ``CLOCK_MONOTONIC_RAW`` timestamp.
    :ivar role: Lifecycle role carried by the exact binding.
    """

    binding: NativeTerminalRequestBinding
    owner_sequence: int
    producer_id: int
    producer_sequence: int
    producer_rank: int
    event_kind: NativeTerminalOwnerEventKind
    enqueued_ns: int
    completed_ns: int
    role: NativeTerminalOwnerRole

    def __post_init__(self) -> None:
        """Validate one exact, evidence-only submission observation."""

        if type(self.binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        counts = (
            self.owner_sequence,
            self.producer_id,
            self.producer_sequence,
            self.producer_rank,
            self.enqueued_ns,
            self.completed_ns,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("native observation integer fields must be non-negative")
        if self.completed_ns < self.enqueued_ns:
            raise ValueError("native completion cannot precede event enqueue")
        if (
            self.event_kind
            is not NativeTerminalOwnerEventKind.SOURCE_SUBMISSION_ACCEPTED
        ):
            raise ValueError("native observation carries another event kind")
        if self.role is not NativeTerminalOwnerRole.SOURCE:
            raise ValueError("native submission observation must belong to source")
        if self.binding.owner.role is not self.role:
            raise ValueError("observation role differs from its exact binding")
        if self.producer_rank != self.binding.owner.tp_rank:
            raise ValueError("observation producer rank differs from its binding")

    @property
    def latency_ns(self) -> int:
        """Return producer enqueue to native commit latency.

        :returns: Native owner handoff latency in nanoseconds.
        """

        return self.completed_ns - self.enqueued_ns

    @classmethod
    def from_native(
        cls, value: Mapping[str, object]
    ) -> "NativeTerminalOwnerObservation":
        """Parse one evidence-only observation returned by the native bridge.

        :param value: Native observation mapping.
        :returns: Validated immutable observation.
        """

        binding_value = value["binding"]
        if not isinstance(binding_value, Mapping):
            raise TypeError("native observation binding must be a mapping")
        return cls(
            binding=_binding_from_native(binding_value),
            owner_sequence=int(value["owner_sequence"]),
            producer_id=int(value["producer_id"]),
            producer_sequence=int(value["producer_sequence"]),
            producer_rank=int(value["producer_rank"]),
            event_kind=NativeTerminalOwnerEventKind(int(value["event_kind"])),
            enqueued_ns=int(value["enqueued_ns"]),
            completed_ns=int(value["completed_ns"]),
            role=NativeTerminalOwnerRole(int(value["role"])),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerObservationBatch:
    """Typed result of one non-gating native observation drain.

    :ivar observations: Every native record which crossed validation.
    :ivar dropped_count: Records rejected independently at the Python ABI
        projection boundary.
    """

    observations: tuple[NativeTerminalOwnerObservation, ...]
    dropped_count: int

    def __post_init__(self) -> None:
        """Validate one complete observation-drain result."""

        if type(self.observations) is not tuple or any(
            type(value) is not NativeTerminalOwnerObservation
            for value in self.observations
        ):
            raise TypeError(
                "observations must contain NativeTerminalOwnerObservation values"
            )
        if type(self.dropped_count) is not int or self.dropped_count < 0:
            raise ValueError("dropped_count must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerOutput:
    """Immutable production action earned by an authoritative native commit.

    Commits which earn no external action stay entirely native. This queue may
    never become a Python-drain dependency for native lifecycle progress.

    :ivar binding: Exact lifecycle identity affected by the commit.
    :ivar owner_sequence: Gap-free process-local commit sequence.
    :ivar producer_id: Producer which supplied the accepted event.
    :ivar producer_sequence: Accepted producer-local sequence.
    :ivar event_kind: Closed producer event which earned this commit.
    :ivar enqueued_ns: Producer-side monotonic timestamp.
    :ivar completed_ns: Native post-commit monotonic timestamp.
    :ivar role: Source or decode lifecycle role.
    :ivar previous_phase: Stable phase code before the reduction.
    :ivar phase: Stable native source or decode phase code.
    :ivar live_resources: Resource bits still pinned.
    :ivar retired_resources: Resource bits carrying exact reuse proof.
    :ivar quarantined_resources: Resource bits retained fail-closed.
    :ivar actions: One-shot production actions earned by this commit.
    :ivar armed_deadline_mask: Deadlines remaining active after this commit.
    :ivar process_fatal: Whether process continuation became unsafe.
    :ivar fatal_code: Sticky first fatal code, or none.
    """

    binding: NativeTerminalRequestBinding
    owner_sequence: int
    producer_id: int
    producer_sequence: int
    event_kind: NativeTerminalOwnerEventKind
    enqueued_ns: int
    completed_ns: int
    role: NativeTerminalOwnerRole
    previous_phase: int
    phase: int
    live_resources: int
    retired_resources: int
    quarantined_resources: int
    actions: tuple[NativeTerminalOwnerAction, ...]
    armed_deadline_mask: int
    process_fatal: bool
    fatal_code: NativeTerminalOwnerFatalCode

    def __post_init__(self) -> None:
        """Validate timing, output identity, and resource conservation."""

        if type(self.binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        counts = (
            self.owner_sequence,
            self.producer_id,
            self.producer_sequence,
            self.enqueued_ns,
            self.completed_ns,
            self.previous_phase,
            self.phase,
            self.live_resources,
            self.retired_resources,
            self.quarantined_resources,
            self.armed_deadline_mask,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("native output integer fields must be non-negative")
        if self.completed_ns < self.enqueued_ns:
            raise ValueError("native completion cannot precede event enqueue")
        if type(self.event_kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("event_kind must be NativeTerminalOwnerEventKind")
        if type(self.role) is not NativeTerminalOwnerRole:
            raise TypeError("role must be NativeTerminalOwnerRole")
        if self.binding.owner.role is not self.role:
            raise ValueError("output role differs from its exact binding")
        if type(self.actions) is not tuple or len(self.actions) == 0:
            raise ValueError("actions must be a non-empty tuple")
        if any(
            type(action) is not NativeTerminalOwnerAction for action in self.actions
        ):
            raise TypeError("actions must contain NativeTerminalOwnerAction values")
        if any(action.binding != self.binding for action in self.actions):
            raise ValueError("native output action belongs to another binding")
        if any(
            action.commit_timestamp_ns != self.completed_ns for action in self.actions
        ):
            raise ValueError("native output action timestamp differs from its commit")
        action_ids = tuple(action.action_id for action in self.actions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("native output action identities must be unique")
        if type(self.process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")
        if type(self.fatal_code) is not NativeTerminalOwnerFatalCode:
            raise TypeError("fatal_code must be NativeTerminalOwnerFatalCode")
        if self.process_fatal != (
            self.fatal_code is not NativeTerminalOwnerFatalCode.NONE
        ):
            raise ValueError("process fatal disposition and code must agree")
        if self.process_fatal != (
            NativeTerminalOwnerActionKind.PROCESS_FATAL
            in tuple(action.kind for action in self.actions)
        ):
            raise ValueError("process fatal disposition and action must agree")
        phase_values: tuple[int, ...]
        if self.role is NativeTerminalOwnerRole.SOURCE:
            phase_values = tuple(int(value) for value in NativeSourceLifecyclePhase)
            if int(self.event_kind) >= 40:
                raise ValueError("source output carries a decode event")
        else:
            phase_values = tuple(int(value) for value in NativeDecodeLifecyclePhase)
            if int(self.event_kind) < 40:
                raise ValueError("decode output carries a source event")
        if self.previous_phase not in phase_values or self.phase not in phase_values:
            raise ValueError("native output carries an unknown role-specific phase")
        masks = (
            self.live_resources,
            self.retired_resources,
            self.quarantined_resources,
        )
        if (
            masks[0] & masks[1] != 0
            or masks[0] & masks[2] != 0
            or masks[1] & masks[2] != 0
        ):
            raise ValueError("native resource partitions overlap")
        expected = (
            NATIVE_SOURCE_RESOURCE_MASK
            if self.role is NativeTerminalOwnerRole.SOURCE
            else NATIVE_DECODE_RESOURCE_MASK
        )
        if (masks[0] | masks[1] | masks[2]) != expected:
            raise ValueError("native resource partitions are not conservative")

    @property
    def latency_ns(self) -> int:
        """Return producer enqueue to authoritative commit latency.

        :returns: Native owner latency in nanoseconds.
        """

        return self.completed_ns - self.enqueued_ns

    @classmethod
    def from_native(cls, value: Mapping[str, object]) -> "NativeTerminalOwnerOutput":
        """Parse one immutable output returned by the native bridge.

        :param value: Native output mapping.
        :returns: Validated immutable output.
        """

        binding_value = value["binding"]
        if not isinstance(binding_value, Mapping):
            raise TypeError("native output binding must be a mapping")
        actions_value = value["actions"]
        if type(actions_value) not in (list, tuple):
            raise TypeError("native output actions must be a sequence")
        actions: list[NativeTerminalOwnerAction] = []
        for action_value in actions_value:
            if not isinstance(action_value, Mapping):
                raise TypeError("native output action must be a mapping")
            actions.append(NativeTerminalOwnerAction.from_native(action_value))
        return cls(
            binding=_binding_from_native(binding_value),
            owner_sequence=int(value["owner_sequence"]),
            producer_id=int(value["producer_id"]),
            producer_sequence=int(value["producer_sequence"]),
            event_kind=NativeTerminalOwnerEventKind(int(value["event_kind"])),
            enqueued_ns=int(value["enqueued_ns"]),
            completed_ns=int(value["completed_ns"]),
            role=NativeTerminalOwnerRole(int(value["role"])),
            previous_phase=int(value["previous_phase"]),
            phase=int(value["phase"]),
            live_resources=int(value["live_resources"]),
            retired_resources=int(value["retired_resources"]),
            quarantined_resources=int(value["quarantined_resources"]),
            actions=tuple(actions),
            armed_deadline_mask=int(value["armed_deadline_mask"]),
            process_fatal=bool(value["process_fatal"]),
            fatal_code=NativeTerminalOwnerFatalCode(int(value["fatal_code"])),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalQualificationTrace:
    """Test-only trace of one real native reducer commit.

    :ivar machine_index: Closed-loop qualification machine identity.
    :ivar generation_index: Machine-local request generation index.
    :ivar hop_index: Correlated production hop within the seven-hop sample.
    :ivar binding_digest: Exact request binding reduced by the commit.
    :ivar event_kind: Real source event applied by the native reducer.
    :ivar previous_phase: Native source phase before the commit.
    :ivar phase: Native source phase after the commit.
    :ivar enqueued_ns: Native producer timestamp.
    :ivar completed_ns: Authoritative post-commit timestamp.
    """

    machine_index: int
    generation_index: int
    hop_index: int
    binding_digest: bytes
    event_kind: NativeTerminalOwnerEventKind
    previous_phase: NativeSourceLifecyclePhase
    phase: NativeSourceLifecyclePhase
    enqueued_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        """Validate one correlated real-reducer trace."""

        counts = (
            self.machine_index,
            self.generation_index,
            self.hop_index,
            self.enqueued_ns,
            self.completed_ns,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("qualification trace counts must be non-negative")
        if self.hop_index >= 7:
            raise ValueError("qualification hop index must be in the seven-hop path")
        _require_exact_bytes(self.binding_digest, _DIGEST_BYTES, "binding_digest")
        if type(self.event_kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("event_kind must be NativeTerminalOwnerEventKind")
        phases = (self.previous_phase, self.phase)
        if any(type(value) is not NativeSourceLifecyclePhase for value in phases):
            raise TypeError("qualification traces require source lifecycle phases")
        if self.completed_ns < self.enqueued_ns:
            raise ValueError("qualification completion cannot precede enqueue")

    @property
    def latency_ns(self) -> int:
        """Return enqueue-to-authoritative-commit latency.

        :returns: Native reducer latency in nanoseconds.
        """

        return self.completed_ns - self.enqueued_ns

    @classmethod
    def from_native(
        cls, value: Mapping[str, object]
    ) -> "NativeTerminalQualificationTrace":
        """Parse one test-only trace from the native bridge.

        :param value: Native trace mapping.
        :returns: Validated immutable trace.
        """

        return cls(
            machine_index=int(value["machine_index"]),
            generation_index=int(value["generation_index"]),
            hop_index=int(value["hop_index"]),
            binding_digest=bytes(value["binding_digest"]),
            event_kind=NativeTerminalOwnerEventKind(int(value["event_kind"])),
            previous_phase=NativeSourceLifecyclePhase(int(value["previous_phase"])),
            phase=NativeSourceLifecyclePhase(int(value["phase"])),
            enqueued_ns=int(value["enqueued_ns"]),
            completed_ns=int(value["completed_ns"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalOwnerInventory:
    """Complete reactor, lifecycle, queue, and fatal-state inventory.

    :ivar input_capacity: Bounded native input capacity.
    :ivar output_capacity: Bounded production-action capacity.
    :ivar fatal_output_capacity: Reserved fail-closed action capacity.
    :ivar observation_capacity: Bounded evidence-only commit capacity.
    :ivar queued_input_count: Inputs not yet committed.
    :ivar queued_output_count: Earned actions not yet consumed.
    :ivar queued_fatal_output_count: Reserved fail-closed actions not consumed.
    :ivar queued_observation_count: Evidence records not yet consumed.
    :ivar delivered_observation_count: Evidence records drained by the sole
        observation consumer.
    :ivar dropped_observation_count: Evidence records omitted without changing
        lifecycle progress.
    :ivar observation_count: Total actionless commits selected for evidence.
    :ivar observation_eventfd_error_count: Non-gating observation wake or drain
        failures.
    :ivar pending_action_count: Native actions not yet acknowledged.
    :ivar registered_producer_count: Exact producer registry size.
    :ivar joined_producer_count: Producers explicitly joined at drain.
    :ivar active_source_count: Source generations with live resources.
    :ivar active_decode_count: Decode generations with live resources.
    :ivar safely_retired_count: Fully retired request generations.
    :ivar quarantined_count: Generations retaining quarantined resources.
    :ivar quarantined_binding_digests: Exact retained quarantine identities.
    :ivar armed_deadline_count: Exact active deadline count.
    :ivar transition_count: Authoritative native reducer commit count.
    :ivar action_count: Production actions earned since construction.
    :ivar qualification_trace_count: Test-only retained correlated traces.
    :ivar started: Whether the native reactor started.
    :ivar admission_open: Whether new lifecycle registrations remain accepted.
    :ivar event_admission_open: Whether registered producers may still drain
        existing lifecycles.
    :ivar draining: Whether fail-closed drain began.
    :ivar closed: Whether exact clean closure completed.
    :ivar input_eventfd_open: Whether the native input descriptor remains open.
    :ivar output_eventfd_open: Whether the action descriptor remains open.
    :ivar observation_eventfd_open: Whether the evidence descriptor remains open.
    :ivar output_drain_active: Whether the sole consumer owns a swapped batch.
    :ivar fatal_code: Sticky first process-fatal code.
    :ivar fatal_binding_digest: Exact triggering binding when request-local.
    :ivar deadline_table_digest: Hash-bound table retained by the reactor.
    """

    input_capacity: int
    output_capacity: int
    fatal_output_capacity: int
    observation_capacity: int
    queued_input_count: int
    queued_output_count: int
    queued_fatal_output_count: int
    queued_observation_count: int
    delivered_observation_count: int
    dropped_observation_count: int
    observation_count: int
    observation_eventfd_error_count: int
    pending_action_count: int
    registered_producer_count: int
    joined_producer_count: int
    active_source_count: int
    active_decode_count: int
    safely_retired_count: int
    quarantined_count: int
    quarantined_binding_digests: tuple[bytes, ...]
    armed_deadline_count: int
    transition_count: int
    action_count: int
    qualification_trace_count: int
    started: bool
    admission_open: bool
    event_admission_open: bool
    draining: bool
    closed: bool
    input_eventfd_open: bool
    output_eventfd_open: bool
    observation_eventfd_open: bool
    output_drain_active: bool
    fatal_code: NativeTerminalOwnerFatalCode
    fatal_binding_digest: bytes | None
    deadline_table_digest: bytes

    def __post_init__(self) -> None:
        """Validate one conservative native owner inventory."""

        counts = (
            self.input_capacity,
            self.output_capacity,
            self.fatal_output_capacity,
            self.observation_capacity,
            self.queued_input_count,
            self.queued_output_count,
            self.queued_fatal_output_count,
            self.queued_observation_count,
            self.delivered_observation_count,
            self.dropped_observation_count,
            self.observation_count,
            self.observation_eventfd_error_count,
            self.pending_action_count,
            self.registered_producer_count,
            self.joined_producer_count,
            self.active_source_count,
            self.active_decode_count,
            self.safely_retired_count,
            self.quarantined_count,
            self.armed_deadline_count,
            self.transition_count,
            self.action_count,
            self.qualification_trace_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("native inventory counts must be non-negative")
        if (
            self.input_capacity == 0
            or self.output_capacity == 0
            or self.fatal_output_capacity == 0
            or self.observation_capacity == 0
        ):
            raise ValueError("native inventory capacities must be positive")
        if self.queued_input_count > self.input_capacity:
            raise ValueError("native input queue exceeds its physical capacity")
        normal_output_count = self.queued_output_count - self.queued_fatal_output_count
        if normal_output_count < 0 or normal_output_count > self.output_capacity:
            raise ValueError("native output queue exceeds its physical capacity")
        if self.queued_fatal_output_count > self.fatal_output_capacity:
            raise ValueError("native fatal output queue exceeds its reserve")
        if self.queued_observation_count > self.observation_capacity:
            raise ValueError("native observation queue exceeds its physical capacity")
        if self.observation_count != (
            self.queued_observation_count
            + self.delivered_observation_count
            + self.dropped_observation_count
        ):
            raise ValueError("native observation inventory violates conservation")
        if self.joined_producer_count > self.registered_producer_count:
            raise ValueError("joined producer count exceeds registered producers")
        if type(self.quarantined_binding_digests) is not tuple or any(
            type(digest) is not bytes or len(digest) != _DIGEST_BYTES
            for digest in self.quarantined_binding_digests
        ):
            raise ValueError("quarantined binding digests must contain 32 bytes")
        if len(set(self.quarantined_binding_digests)) != len(
            self.quarantined_binding_digests
        ):
            raise ValueError("quarantined binding digests must be unique")
        if len(self.quarantined_binding_digests) != self.quarantined_count:
            raise ValueError("quarantine count and identities disagree")
        flags = (
            self.started,
            self.admission_open,
            self.event_admission_open,
            self.draining,
            self.closed,
            self.input_eventfd_open,
            self.output_eventfd_open,
            self.observation_eventfd_open,
            self.output_drain_active,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("native inventory lifecycle flags must be bool values")
        if type(self.fatal_code) is not NativeTerminalOwnerFatalCode:
            raise TypeError("fatal_code must be NativeTerminalOwnerFatalCode")
        if self.fatal_binding_digest is not None:
            _require_exact_bytes(
                self.fatal_binding_digest, _DIGEST_BYTES, "fatal_binding_digest"
            )
        _require_exact_bytes(
            self.deadline_table_digest, _DIGEST_BYTES, "deadline_table_digest"
        )
        if self.closed:
            if self.admission_open or self.event_admission_open:
                raise ValueError("closed native inventory cannot accept work")
            if (
                self.fatal_code is NativeTerminalOwnerFatalCode.NONE
                and self.joined_producer_count != self.registered_producer_count
            ):
                raise ValueError("closed native inventory has unjoined producers")
            if (
                self.queued_input_count != 0
                or self.queued_output_count != 0
                or self.queued_observation_count != 0
                or self.pending_action_count != 0
                or self.active_source_count != 0
                or self.active_decode_count != 0
                or self.armed_deadline_count != 0
                or self.output_drain_active
            ):
                raise ValueError("closed native inventory retains active work")
            if (
                self.input_eventfd_open
                or self.output_eventfd_open
                or self.observation_eventfd_open
            ):
                raise ValueError("closed native inventory retains an eventfd")

    @classmethod
    def from_native(cls, value: Mapping[str, object]) -> "NativeTerminalOwnerInventory":
        """Parse one complete native inventory mapping.

        :param value: Native inventory fields.
        :returns: Validated immutable inventory.
        """

        fatal_binding_value = value["fatal_binding_digest"]
        fatal_binding_digest: bytes | None = None
        if fatal_binding_value is not None:
            fatal_binding_digest = bytes(fatal_binding_value)
        return cls(
            input_capacity=int(value["input_capacity"]),
            output_capacity=int(value["output_capacity"]),
            fatal_output_capacity=int(value["fatal_output_capacity"]),
            observation_capacity=int(value["observation_capacity"]),
            queued_input_count=int(value["queued_input_count"]),
            queued_output_count=int(value["queued_output_count"]),
            queued_fatal_output_count=int(value["queued_fatal_output_count"]),
            queued_observation_count=int(value["queued_observation_count"]),
            delivered_observation_count=int(value["delivered_observation_count"]),
            dropped_observation_count=int(value["dropped_observation_count"]),
            observation_count=int(value["observation_count"]),
            observation_eventfd_error_count=int(
                value["observation_eventfd_error_count"]
            ),
            pending_action_count=int(value["pending_action_count"]),
            registered_producer_count=int(value["registered_producer_count"]),
            joined_producer_count=int(value["joined_producer_count"]),
            active_source_count=int(value["active_source_count"]),
            active_decode_count=int(value["active_decode_count"]),
            safely_retired_count=int(value["safely_retired_count"]),
            quarantined_count=int(value["quarantined_count"]),
            quarantined_binding_digests=tuple(
                bytes(digest) for digest in value["quarantined_binding_digests"]
            ),
            armed_deadline_count=int(value["armed_deadline_count"]),
            transition_count=int(value["transition_count"]),
            action_count=int(value["action_count"]),
            qualification_trace_count=int(value["qualification_trace_count"]),
            started=bool(value["started"]),
            admission_open=bool(value["admission_open"]),
            event_admission_open=bool(value["event_admission_open"]),
            draining=bool(value["draining"]),
            closed=bool(value["closed"]),
            input_eventfd_open=bool(value["input_eventfd_open"]),
            output_eventfd_open=bool(value["output_eventfd_open"]),
            observation_eventfd_open=bool(value["observation_eventfd_open"]),
            output_drain_active=bool(value["output_drain_active"]),
            fatal_code=NativeTerminalOwnerFatalCode(int(value["fatal_code"])),
            fatal_binding_digest=fatal_binding_digest,
            deadline_table_digest=bytes(value["deadline_table_digest"]),
        )


def _process_identity_from_native(
    value: Mapping[str, object],
) -> NativeTerminalProcessIdentity:
    """Parse one native process identity mapping.

    :param value: Native process identity fields.
    :returns: Validated immutable identity.
    """

    return NativeTerminalProcessIdentity(
        process_generation=bytes(value["process_generation"]),
        role=NativeTerminalOwnerRole(int(value["role"])),
        tp_rank=int(value["tp_rank"]),
        tp_size=int(value["tp_size"]),
        digest=bytes(value["digest"]),
    )


def _binding_from_native(
    value: Mapping[str, object],
) -> NativeTerminalRequestBinding:
    """Parse one native request binding mapping.

    :param value: Native request binding fields.
    :returns: Validated immutable binding.
    """

    owner_value = value["owner"]
    if not isinstance(owner_value, Mapping):
        raise TypeError("native binding owner must be a mapping")
    return NativeTerminalRequestBinding(
        room_id=int(value["room_id"]),
        request_generation=bytes(value["request_generation"]),
        owner=_process_identity_from_native(owner_value),
        rank_manifest_digest=bytes(value["rank_manifest_digest"]),
        allocation_digest=bytes(value["allocation_digest"]),
        digest=bytes(value["digest"]),
    )
