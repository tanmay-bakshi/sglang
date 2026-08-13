import dataclasses
import enum
import hashlib
import json


class TerminalDeadlineKind(enum.StrEnum):
    """Hash-bound timeout phase in packed terminal ownership."""

    EXISTING_NIXL_CAPABILITY_READY = "existing_nixl_capability_ready"
    EXISTING_PACKED_CONTROL = "existing_packed_control"
    OWNER_PRODUCER_AND_GATHER = "owner_producer_and_gather"
    OWNER_NATIVE_TRANSFER = "owner_native_transfer"
    OWNER_DECODE_SCATTER = "owner_decode_scatter"
    OWNER_TEARDOWN_ACK = "owner_teardown_ack"
    OWNER_REQUEST_GLOBAL_READY = "owner_request_global_ready"
    OWNER_SCHEDULER_RECEIPT_CONSUMPTION = "owner_scheduler_receipt_consumption"
    OWNER_GATEWAY_PUBLICATION = "owner_gateway_publication"
    OWNER_SHUTDOWN_DRAIN = "owner_shutdown_drain"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalDeadlineSpec:
    """Immutable effective timeout value, anchor, and failure outcome.

    :ivar kind: Stable timeout phase identity.
    :ivar seconds: Frozen timeout duration in seconds.
    :ivar starts_at: Exact one-shot start anchor.
    :ivar timeout_outcome: Fail-closed timeout disposition.
    """

    kind: TerminalDeadlineKind
    seconds: float
    starts_at: str
    timeout_outcome: str

    def __post_init__(self) -> None:
        """Validate one complete timeout specification."""

        if type(self.kind) is not TerminalDeadlineKind:
            raise TypeError("kind must be TerminalDeadlineKind")
        if type(self.seconds) is not float or self.seconds <= 0.0:
            raise ValueError("seconds must be a positive float")
        if type(self.starts_at) is not str or len(self.starts_at) == 0:
            raise ValueError("starts_at must be a non-empty string")
        if type(self.timeout_outcome) is not str or len(self.timeout_outcome) == 0:
            raise ValueError("timeout_outcome must be a non-empty string")

    @property
    def duration_ns(self) -> int:
        """Return the exact integer nanosecond duration.

        :returns: Positive timeout duration in nanoseconds.
        """

        return int(self.seconds * 1_000_000_000)


PACKED_TERMINAL_DEADLINES = (
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.EXISTING_NIXL_CAPABILITY_READY,
        seconds=5.0,
        starts_at="remote_route_binding_is_published",
        timeout_outcome="route_failure_and_bound_request_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.EXISTING_PACKED_CONTROL,
        seconds=60.0,
        starts_at="phase_specific_control_wait_begins",
        timeout_outcome="request_failure_and_phase_owned_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_PRODUCER_AND_GATHER,
        seconds=60.0,
        starts_at="bound_submission_is_enqueued_to_source_owner",
        timeout_outcome="source_request_failure_and_source_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_NATIVE_TRANSFER,
        seconds=60.0,
        starts_at="exact_nixl_handle_is_posted",
        timeout_outcome="native_cancel_attempt_then_source_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_DECODE_SCATTER,
        seconds=60.0,
        starts_at="exact_decode_scatter_is_enqueued",
        timeout_outcome="decode_request_failure_and_decode_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_TEARDOWN_ACK,
        seconds=60.0,
        starts_at="last_writer_teardown_message_is_sent",
        timeout_outcome="request_failure_and_source_decode_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_REQUEST_GLOBAL_READY,
        seconds=60.0,
        starts_at="first_local_decode_ready_enters_request_coordinator",
        timeout_outcome="request_failure_and_incomplete_rank_resource_quarantine",
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
        seconds=60.0,
        starts_at="adoption_ready_receipt_is_enqueued_before_eventfd_signal",
        timeout_outcome=(
            "process_fatal_scheduler_progress_failure_and_request_quarantine"
        ),
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
        seconds=60.0,
        starts_at="request_ready_is_delivered_to_canonical_publisher",
        timeout_outcome=(
            "process_fatal_publisher_failure_and_publication_identity_quarantine"
        ),
    ),
    TerminalDeadlineSpec(
        kind=TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
        seconds=60.0,
        starts_at="admission_is_closed_for_shutdown",
        timeout_outcome="unresolved_resource_quarantine_and_fail_closed_teardown",
    ),
)


def terminal_deadline_spec(kind: TerminalDeadlineKind) -> TerminalDeadlineSpec:
    """Return the exact packaged specification for one timeout phase.

    :param kind: Timeout phase identity.
    :returns: Frozen timeout specification.
    """

    if type(kind) is not TerminalDeadlineKind:
        raise TypeError("kind must be TerminalDeadlineKind")
    matches = tuple(spec for spec in PACKED_TERMINAL_DEADLINES if spec.kind is kind)
    if len(matches) != 1:
        raise RuntimeError(
            f"deadline table has {len(matches)} entries for {kind.value}"
        )
    return matches[0]


def canonical_terminal_deadline_table() -> tuple[dict[str, object], ...]:
    """Return the canonical packaged deadline table.

    :returns: Stable records ordered by phase identity.
    """

    return tuple(
        {
            "kind": spec.kind.value,
            "seconds": spec.seconds,
            "starts_at": spec.starts_at,
            "timeout_outcome": spec.timeout_outcome,
        }
        for spec in sorted(PACKED_TERMINAL_DEADLINES, key=lambda item: item.kind.value)
    )


def terminal_deadline_table_digest() -> bytes:
    """Return the SHA-256 digest of the canonical effective table.

    :returns: Canonical deadline-table digest.
    """

    payload = json.dumps(
        canonical_terminal_deadline_table(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


@dataclasses.dataclass(frozen=True, slots=True)
class BoundTerminalDeadline:
    """One timeout started exactly once at its frozen anchor.

    :ivar spec: Frozen timeout phase specification.
    :ivar started_ns: Monotonic timestamp at the exact start anchor.
    """

    spec: TerminalDeadlineSpec
    started_ns: int

    def __post_init__(self) -> None:
        """Validate one one-shot bound deadline."""

        if type(self.spec) is not TerminalDeadlineSpec:
            raise TypeError("spec must be TerminalDeadlineSpec")
        if type(self.started_ns) is not int or self.started_ns < 0:
            raise ValueError("started_ns must be a non-negative integer")

    @property
    def expires_ns(self) -> int:
        """Return the exact monotonic expiration timestamp.

        :returns: Start timestamp plus frozen duration.
        """

        return self.started_ns + self.spec.duration_ns

    def expired(self, now_ns: int) -> bool:
        """Return whether the bound deadline has expired.

        :param now_ns: Current monotonic timestamp.
        :returns: Whether current time reached the expiration timestamp.
        """

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        return now_ns >= self.expires_ns


def start_terminal_deadline(
    kind: TerminalDeadlineKind,
    started_ns: int,
) -> BoundTerminalDeadline:
    """Bind one packaged deadline to its exact start timestamp.

    :param kind: Timeout phase identity.
    :param started_ns: Monotonic timestamp at the frozen start anchor.
    :returns: One-shot immutable deadline.
    """

    return BoundTerminalDeadline(
        spec=terminal_deadline_spec(kind),
        started_ns=started_ns,
    )
