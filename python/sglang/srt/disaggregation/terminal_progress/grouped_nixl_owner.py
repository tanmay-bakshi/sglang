import dataclasses
import enum
import threading
from collections.abc import Callable
from typing import Protocol

from nixl._api import nixl_xfer_completion_receipt, nixl_xfer_handle
from nixl._bindings import NIXL_IN_PROG, NIXL_SUCCESS, nixl_status_t

from sglang.srt.disaggregation.terminal_progress.clock import (
    SystemTerminalOwnerClock,
    TerminalOwnerClock,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
)
from sglang.srt.disaggregation.terminal_progress.nixl_adapter import (
    NixlTerminalAgentBoundary,
    NixlTerminalCancelOutcome,
    NixlTerminalChannelInventory,
    NixlTerminalEventAdapter,
    NixlTerminalLifecycleError,
    NixlTerminalProcessFatalError,
    NixlTerminalSubscription,
    NixlTerminalSubscriptionBinding,
    NixlTransferTerminalEvent,
)
from sglang.srt.disaggregation.terminal_progress.nixl_owner_boundary import (
    NixlTerminalOwnerBoundary,
)

_UINT64_MAX = (1 << 64) - 1


class GroupedNixlTerminalAgentBoundary(NixlTerminalAgentBoundary, Protocol):
    """Qualified NIXL API used by request-grouped terminal ownership."""

    def take_xfer_completion_receipt(
        self,
        handle: nixl_xfer_handle,
    ) -> nixl_xfer_completion_receipt | None:
        """Take one successful exact-generation completion receipt.

        :param handle: Exact terminal transfer handle.
        :returns: Take-once completion authority when terminal success exists.
        """

        ...


class GroupedNixlTransferHandleBoundary(Protocol):
    """Transfer-handle lifetime surface retained by grouped ownership."""

    def release(self) -> None:
        """Release one settled transfer handle."""


class GroupedNixlTransferState(enum.StrEnum):
    """Exact grouped-transfer lifecycle retained through source ACK."""

    ARMED = "armed"
    POSTING = "posting"
    POSTED = "posted"
    AMBIGUOUS = "ambiguous"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"
    SETTLED_SUCCESS = "settled_success"
    QUARANTINED = "quarantined"
    RELEASED = "released"


class GroupedNixlTransferMember(enum.StrEnum):
    """Closed semantic membership of one source request group."""

    MAIN = "main"
    DFLASH_BOUNDARY = "dflash_boundary"


class GroupedNixlResultState(enum.StrEnum):
    """Delivery state of one request-level native terminal result."""

    ABSENT = "absent"
    PENDING = "pending"
    CLAIMED = "claimed"
    ACKNOWLEDGED = "acknowledged"


def grouped_nixl_source_members(
    canonical_dflash_writer: bool,
) -> tuple[GroupedNixlTransferMember, ...]:
    """Return one source rank's exact request-group schema.

    :param canonical_dflash_writer: Whether this rank owns the sole boundary
        token transfer.
    :returns: Main KV plus canonical-only DFlash membership.
    """

    if type(canonical_dflash_writer) is not bool:
        raise TypeError("canonical_dflash_writer must be bool")
    if canonical_dflash_writer:
        return (
            GroupedNixlTransferMember.MAIN,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        )
    return (GroupedNixlTransferMember.MAIN,)


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedNixlMemberTiming:
    """One member's exact post-to-native-terminal interval.

    :ivar member: Main KV or DFlash boundary semantic member.
    :ivar owner_cookie: Exact native subscription correlation identity.
    :ivar post_started_ns: Local monotonic timestamp immediately before post.
    :ivar native_terminal_ns: Native terminal-token publication timestamp.
    """

    member: GroupedNixlTransferMember
    owner_cookie: int
    post_started_ns: int
    native_terminal_ns: int

    def __post_init__(self) -> None:
        """Validate one same-process member interval."""

        if type(self.member) is not GroupedNixlTransferMember:
            raise TypeError("member must be GroupedNixlTransferMember")
        if type(self.owner_cookie) is not int or self.owner_cookie <= 0:
            raise ValueError("owner_cookie must be a positive integer")
        if type(self.post_started_ns) is not int or self.post_started_ns < 0:
            raise ValueError("post_started_ns must be non-negative")
        if (
            type(self.native_terminal_ns) is not int
            or self.native_terminal_ns < self.post_started_ns
        ):
            raise ValueError("native terminality cannot precede transfer post")


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedNixlTerminalResult:
    """One aggregate transfer result eligible for native lifecycle ingress.

    :ivar binding_digest: Exact source lifecycle represented by the group.
    :ivar successful: Whether every expected transfer completed successfully.
    :ivar transfer_count: Exact expected and observed transfer population.
    :ivar native_timestamp_ns: Latest member terminal publication timestamp.
    :ivar reason: Stable failure evidence, otherwise ``None`` on success.
    :ivar member_timings: Distinct post-to-terminal intervals for members whose
        terminal tokens contributed to this result.
    """

    binding_digest: bytes
    successful: bool
    transfer_count: int
    native_timestamp_ns: int
    reason: str | None
    member_timings: tuple[GroupedNixlMemberTiming, ...]

    def __post_init__(self) -> None:
        """Validate one complete aggregate result."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if type(self.successful) is not bool:
            raise TypeError("successful must be bool")
        if type(self.transfer_count) is not int or self.transfer_count <= 0:
            raise ValueError("transfer_count must be a positive integer")
        if type(self.native_timestamp_ns) is not int or self.native_timestamp_ns < 0:
            raise ValueError("native_timestamp_ns must be non-negative")
        if type(self.member_timings) is not tuple or len(self.member_timings) == 0:
            raise ValueError("member_timings must be a non-empty tuple")
        if any(
            type(timing) is not GroupedNixlMemberTiming
            for timing in self.member_timings
        ):
            raise TypeError("member_timings contains an invalid interval")
        timing_members = tuple(timing.member for timing in self.member_timings)
        if len(set(timing_members)) != len(timing_members):
            raise ValueError("member_timings must contain unique semantic members")
        if len(self.member_timings) > self.transfer_count:
            raise ValueError("member timings exceed the expected transfer population")
        if self.successful:
            if self.reason is not None:
                raise ValueError("successful grouped result cannot carry a reason")
            if len(self.member_timings) != self.transfer_count:
                raise ValueError(
                    "successful grouped result requires every member timing"
                )
            if self.native_timestamp_ns != max(
                timing.native_terminal_ns for timing in self.member_timings
            ):
                raise ValueError("successful result timestamp differs from its members")
            return
        if type(self.reason) is not str or len(self.reason) == 0:
            raise ValueError("failed grouped result requires a non-empty reason")
        if self.native_timestamp_ns not in tuple(
            timing.native_terminal_ns for timing in self.member_timings
        ):
            raise ValueError("failed result timestamp lacks a member terminal token")


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedNixlTerminalOwnerInventory:
    """Conservation-complete request-group and native-channel inventory.

    :ivar native: Qualified native terminal-channel inventory.
    :ivar admission_open: Whether new request groups may be created.
    :ivar closed: Whether exact-zero clean closure completed.
    :ivar active_group_count: Groups still retaining lifecycle authority.
    :ivar sealed_group_count: Groups whose expected membership is immutable.
    :ivar pending_result_count: Aggregate results not yet acknowledged.
    :ivar acknowledged_result_count: Live groups accepted by native lifecycle.
    :ivar active_transfer_count: Transfer records retained by live groups.
    :ivar terminal_transfer_count: Transfers with exact native terminality.
    :ivar settled_transfer_count: Successful receipts consumed under owner action.
    :ivar quarantined_transfer_count: Transfers retained after ambiguous failure.
    :ivar released_transfer_count: Successful handles released since creation.
    :ivar unowned_handle_count: Handles retained after subscription ambiguity.
    """

    native: NixlTerminalChannelInventory
    admission_open: bool
    closed: bool
    active_group_count: int
    sealed_group_count: int
    pending_result_count: int
    acknowledged_result_count: int
    active_transfer_count: int
    terminal_transfer_count: int
    settled_transfer_count: int
    quarantined_transfer_count: int
    released_transfer_count: int
    unowned_handle_count: int

    def __post_init__(self) -> None:
        """Validate exact non-negative inventory accounting."""

        if type(self.native) is not NixlTerminalChannelInventory:
            raise TypeError("native must be NixlTerminalChannelInventory")
        if type(self.admission_open) is not bool or type(self.closed) is not bool:
            raise TypeError("inventory lifecycle flags must be bool")
        counts = (
            self.active_group_count,
            self.sealed_group_count,
            self.pending_result_count,
            self.acknowledged_result_count,
            self.active_transfer_count,
            self.terminal_transfer_count,
            self.settled_transfer_count,
            self.quarantined_transfer_count,
            self.released_transfer_count,
            self.unowned_handle_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("grouped NIXL inventory counts must be non-negative")
        if self.sealed_group_count > self.active_group_count:
            raise ValueError("sealed groups exceed active groups")
        if self.pending_result_count + self.acknowledged_result_count > (
            self.active_group_count
        ):
            raise ValueError("group result accounting exceeds active groups")
        if (
            self.terminal_transfer_count
            + self.settled_transfer_count
            + self.quarantined_transfer_count
            > self.active_transfer_count
        ):
            raise ValueError("terminal transfer accounting exceeds active transfers")


class GroupedNixlTerminalTransfer:
    """Opaque exact-generation transfer authority issued by one group owner."""

    __slots__ = ("_owner_nonce", "_token")

    _owner_nonce: object
    _token: object

    def __init__(
        self,
        owner_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one owner-private transfer identity.

        :param owner_nonce: Exact issuing owner identity.
        :param token: Owner-private registry key.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _TRANSFER_CONSTRUCTION_SEAL:
            raise TypeError("grouped NIXL transfers are owner constructed")
        self._owner_nonce = owner_nonce
        self._token = token


@dataclasses.dataclass(slots=True)
class _GroupedTransferRecord:
    """Mutable exact-generation transfer retained through source ACK."""

    public: GroupedNixlTerminalTransfer
    binding_digest: bytes
    member: GroupedNixlTransferMember
    owner_cookie: int
    handle: GroupedNixlTransferHandleBoundary
    subscription: NixlTerminalSubscription | None
    subscription_binding: NixlTerminalSubscriptionBinding
    handle_identity: int
    generation: int
    state: GroupedNixlTransferState
    post_started_ns: int | None = None
    native_status: nixl_status_t | None = None
    native_timestamp_ns: int | None = None
    completion_receipt: nixl_xfer_completion_receipt | None = None
    settlement_action_id: int | None = None
    cancel_requested: bool = False


@dataclasses.dataclass(slots=True)
class _GroupedRequestRecord:
    """Mutable request-level aggregation authority."""

    binding_digest: bytes
    expected_members: tuple[GroupedNixlTransferMember, ...]
    transfers: dict[GroupedNixlTransferMember, _GroupedTransferRecord]
    sealed: bool = False
    quarantined: bool = False
    quarantine_reason: str | None = None
    result: GroupedNixlTerminalResult | None = None
    result_state: GroupedNixlResultState = GroupedNixlResultState.ABSENT
    result_delivery_token: object | None = None
    settlement_action_id: int | None = None
    release_action_id: int | None = None


_TRANSFER_CONSTRUCTION_SEAL = object()


class GroupedNixlTerminalEndpoint(NixlTerminalOwnerBoundary):
    """Fixed-member NIXL ownership boundary for one grouped source lane."""

    _owner: "GroupedNixlTerminalOwner"
    _member: GroupedNixlTransferMember

    def __init__(
        self,
        owner: "GroupedNixlTerminalOwner",
        member: GroupedNixlTransferMember,
    ) -> None:
        """Bind one semantic member to its sole process owner.

        :param owner: Exact request-group owner.
        :param member: Immutable main or DFlash membership.
        """

        if type(member) is not GroupedNixlTransferMember:
            raise TypeError("member must be GroupedNixlTransferMember")
        self._owner = owner
        self._member = member

    def arm_transfer(self, handle: object, binding_digest: bytes) -> object:
        """Arm this endpoint's exact member before posting.

        :param handle: Initialized but unposted NIXL handle.
        :param binding_digest: Exact request-group lifecycle digest.
        :returns: Opaque member transfer authority.
        """

        return self._owner._arm_transfer(self._member, handle, binding_digest)

    def post_transfer(
        self,
        transfer: object,
        post: Callable[[object], object],
    ) -> object:
        """Post this endpoint's exact armed member.

        :param transfer: Exact member authority.
        :param post: Existing one-shot post operation.
        :returns: Existing post result.
        """

        return self._owner._post_transfer(self._member, transfer, post)

    def settle_success(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> object:
        """Take this member's receipt under aggregate success authority.

        :param transfer: Exact member authority.
        :param action: Matching source outcome action.
        :returns: Native take-once completion receipt.
        """

        return self._owner._settle_success(self._member, transfer, action)

    def settle_failure(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Quarantine this member under exact failure authority.

        :param transfer: Exact member authority.
        :param action: Matching quarantine or process-fatal action.
        """

        self._owner._settle_failure(self._member, transfer, action)

    def cancel_transfer(self, transfer: object) -> None:
        """Request cancellation of this exact member.

        :param transfer: Exact member authority.
        """

        self._owner._cancel_transfer(self._member, transfer)

    def release_transfer(
        self,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release this member under exact source ACK authority.

        :param transfer: Exact settled member authority.
        :param action: Matching one-shot source ACK action.
        """

        self._owner._release_transfer(self._member, transfer, action)


class GroupedNixlTerminalOwner:
    """Aggregate independently posted NIXL handles into one source transition.

    One native terminal channel carries a distinct exact-generation
    subscription for every handle. Request membership is frozen before the
    channel is drained, and exactly one aggregate result becomes visible only
    after all expected members reach terminality. The source process reactor is
    the sole drain owner, so immediate completion cannot outrun the preceding
    ``SOURCE_GATHER_POSTED`` transition.
    """

    _agent: GroupedNixlTerminalAgentBoundary
    _adapter: NixlTerminalEventAdapter
    _clock: TerminalOwnerClock
    _owner_nonce: object
    _groups: dict[bytes, _GroupedRequestRecord]
    _transfers: dict[object, _GroupedTransferRecord]
    _transfers_by_binding: dict[NixlTerminalSubscriptionBinding, _GroupedTransferRecord]
    _unowned_handles: list[GroupedNixlTransferHandleBoundary]
    _next_owner_cookie: int
    _released_transfer_count: int
    _admission_open: bool
    _closed: bool
    _lock: threading.RLock
    _main_endpoint: GroupedNixlTerminalEndpoint
    _dflash_endpoint: GroupedNixlTerminalEndpoint

    def __init__(
        self,
        agent: GroupedNixlTerminalAgentBoundary,
        channel_capacity: int,
        clock: TerminalOwnerClock | None = None,
    ) -> None:
        """Create one process-lifetime request-grouped terminal channel.

        :param agent: Qualified NIXL terminal-channel and receipt API.
        :param channel_capacity: Maximum simultaneously queued member events.
        :param clock: ``CLOCK_MONOTONIC_RAW`` post timestamp source.
        """

        if type(channel_capacity) is not int or channel_capacity <= 0:
            raise ValueError("channel_capacity must be a positive integer")
        if clock is not None and not isinstance(clock, TerminalOwnerClock):
            raise TypeError("clock must inherit TerminalOwnerClock")
        self._agent = agent
        self._adapter = NixlTerminalEventAdapter(agent, channel_capacity)
        self._clock = SystemTerminalOwnerClock() if clock is None else clock
        self._owner_nonce = object()
        self._groups = {}
        self._transfers = {}
        self._transfers_by_binding = {}
        self._unowned_handles = []
        self._next_owner_cookie = 1
        self._released_transfer_count = 0
        self._admission_open = True
        self._closed = False
        self._lock = threading.RLock()
        self._main_endpoint = GroupedNixlTerminalEndpoint(
            self,
            GroupedNixlTransferMember.MAIN,
        )
        self._dflash_endpoint = GroupedNixlTerminalEndpoint(
            self,
            GroupedNixlTransferMember.DFLASH_BOUNDARY,
        )

    @property
    def main_endpoint(self) -> GroupedNixlTerminalEndpoint:
        """Return the sole main-KV member ownership boundary.

        :returns: Process-lifetime main transfer endpoint.
        """

        return self._main_endpoint

    @property
    def dflash_endpoint(self) -> GroupedNixlTerminalEndpoint:
        """Return the sole DFlash-boundary member ownership boundary.

        :returns: Process-lifetime DFlash transfer endpoint.
        """

        return self._dflash_endpoint

    def fileno(self) -> int:
        """Return the native completion eventfd owned by the process reactor.

        :returns: Open borrowed native terminal-channel descriptor.
        """

        with self._lock:
            self._require_not_closed_locked()
            return self._adapter.fileno()

    def begin_group(
        self,
        binding_digest: bytes,
        expected_members: tuple[GroupedNixlTransferMember, ...],
    ) -> None:
        """Create one empty request group before any member is armed.

        :param binding_digest: Exact registered source lifecycle digest.
        :param expected_members: Immutable semantic member population.
        """

        self._validate_binding_digest(binding_digest)
        if type(expected_members) is not tuple or len(expected_members) == 0:
            raise ValueError("expected_members must be a non-empty tuple")
        if any(
            type(member) is not GroupedNixlTransferMember for member in expected_members
        ):
            raise TypeError("expected_members contains an invalid member")
        if len(set(expected_members)) != len(expected_members):
            raise ValueError("expected_members must be unique")
        if expected_members[0] is not GroupedNixlTransferMember.MAIN:
            raise ValueError("main KV must be the first expected group member")
        if expected_members not in (
            (GroupedNixlTransferMember.MAIN,),
            (
                GroupedNixlTransferMember.MAIN,
                GroupedNixlTransferMember.DFLASH_BOUNDARY,
            ),
        ):
            raise ValueError("source group has an unsupported member schema")
        with self._lock:
            self._require_accepting_locked()
            if binding_digest in self._groups:
                raise NixlTerminalLifecycleError(
                    "grouped NIXL lifecycle binding is already active"
                )
            self._groups[binding_digest] = _GroupedRequestRecord(
                binding_digest=binding_digest,
                expected_members=expected_members,
                transfers={},
            )

    def seal_group(self, binding_digest: bytes) -> None:
        """Freeze exact request membership after every member has posted.

        :param binding_digest: Exact active source lifecycle digest.
        """

        with self._lock:
            group = self._require_group_locked(binding_digest)
            if group.sealed:
                raise NixlTerminalLifecycleError("group membership was already sealed")
            if set(group.transfers) != set(group.expected_members):
                raise NixlTerminalLifecycleError(
                    "group membership differs from its expected semantic schema"
                )
            if any(
                transfer.state is GroupedNixlTransferState.ARMED
                for transfer in group.transfers.values()
            ):
                raise NixlTerminalLifecycleError(
                    "group cannot seal before every member post begins"
                )
            group.sealed = True
            self._materialize_result_locked(group)

    def quarantine_group(self, binding_digest: bytes, reason: str) -> None:
        """Retain one partially posted group after functional ambiguity.

        The corresponding work-action failure owns native lifecycle ingress;
        this method suppresses a later duplicate aggregate result while native
        member events continue to drain to exact terminal inventory.

        :param binding_digest: Exact active source lifecycle digest.
        :param reason: Stable ambiguity evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            group = self._require_group_locked(binding_digest)
            group.quarantined = True
            group.quarantine_reason = reason
            for transfer in group.transfers.values():
                if transfer.state is GroupedNixlTransferState.RELEASED:
                    raise NixlTerminalLifecycleError(
                        "quarantine cannot retain an already released member"
                    )
                transfer.state = GroupedNixlTransferState.QUARANTINED

    def _arm_transfer(
        self,
        member: GroupedNixlTransferMember,
        handle: object,
        binding_digest: bytes,
    ) -> GroupedNixlTerminalTransfer:
        """Subscribe one exact group member before its transfer post.

        :param member: Exact semantic group member.
        :param handle: Initialized but unposted NIXL transfer handle.
        :param binding_digest: Exact active request-group digest.
        :returns: Opaque exact-generation member authority.
        """

        if handle is None:
            raise ValueError("handle must not be None")
        with self._lock:
            self._require_accepting_locked()
            group = self._require_group_locked(binding_digest)
            if group.sealed or group.quarantined:
                raise NixlTerminalLifecycleError(
                    "group does not accept another transfer member"
                )
            if member not in group.expected_members:
                raise NixlTerminalLifecycleError(
                    "transfer member is absent from the request schema"
                )
            if member in group.transfers:
                raise NixlTerminalLifecycleError(
                    "transfer member is already armed for this request"
                )
            owner_cookie = self._allocate_owner_cookie_locked()
            try:
                subscription = self._adapter.subscribe_transfer(
                    handle,
                    owner_cookie,
                )
            except BaseException:
                self._unowned_handles.append(handle)
                raise
            token = object()
            public = GroupedNixlTerminalTransfer(
                self._owner_nonce,
                token,
                _TRANSFER_CONSTRUCTION_SEAL,
            )
            binding = subscription.binding
            record = _GroupedTransferRecord(
                public=public,
                binding_digest=binding_digest,
                member=member,
                owner_cookie=owner_cookie,
                handle=handle,
                subscription=subscription,
                subscription_binding=binding,
                handle_identity=binding.identity,
                generation=binding.generation,
                state=GroupedNixlTransferState.ARMED,
            )
            group.transfers[member] = record
            self._transfers[token] = record
            self._transfers_by_binding[binding] = record
            return public

    def _post_transfer(
        self,
        member: GroupedNixlTransferMember,
        transfer: object,
        post: Callable[[object], object],
    ) -> object:
        """Post one armed member without polling for terminality.

        :param member: Exact semantic group member.
        :param transfer: Exact authority returned by :meth:`_arm_transfer`.
        :param post: Existing one-shot NIXL post operation.
        :returns: Existing post operation result.
        """

        if not callable(post):
            raise TypeError("post must be callable")
        record = self._require_transfer(transfer)
        with self._lock:
            self._require_member_locked(record, member)
            self._require_accepting_locked()
            if record.state is not GroupedNixlTransferState.ARMED:
                raise NixlTerminalLifecycleError(
                    "group member can be posted exactly once after arming"
                )
            record.state = GroupedNixlTransferState.POSTING
            post_started_ns = self._clock.now_ns()
            if type(post_started_ns) is not int or post_started_ns < 0:
                raise NixlTerminalProcessFatalError(
                    "grouped transfer post clock returned an invalid timestamp",
                    self._adapter.query_inventory(),
                )
            record.post_started_ns = post_started_ns
            handle = record.handle
        post_returned = False
        try:
            result = post(handle)
            post_returned = True
            return result
        finally:
            with self._lock:
                if record.state is GroupedNixlTransferState.POSTING:
                    record.state = (
                        GroupedNixlTransferState.POSTED
                        if post_returned
                        else GroupedNixlTransferState.AMBIGUOUS
                    )

    def drain(self) -> tuple[GroupedNixlTerminalResult, ...]:
        """Drain native member events and claim newly complete group results.

        :returns: Aggregate results awaiting explicit lifecycle-ingress ACK.
        """

        with self._lock:
            self._require_not_closed_locked()
            batch = self._adapter.drain()
            touched: dict[bytes, _GroupedRequestRecord] = {}
            for event in batch.events:
                if type(event) is not NixlTransferTerminalEvent:
                    raise NixlTerminalProcessFatalError(
                        "grouped transfer channel carried a capability event",
                        batch.inventory,
                    )
                record = self._transfers_by_binding.get(event.binding)
                if record is None:
                    raise NixlTerminalProcessFatalError(
                        "grouped transfer event has no exact member owner",
                        batch.inventory,
                    )
                if (
                    event.identity != record.handle_identity
                    or event.generation != record.generation
                ):
                    raise NixlTerminalProcessFatalError(
                        "grouped transfer event changed exact member generation",
                        batch.inventory,
                    )
                if record.native_status is not None:
                    raise NixlTerminalProcessFatalError(
                        "grouped transfer terminal event was replayed",
                        batch.inventory,
                    )
                record.subscription = None
                record.native_status = event.status
                record.native_timestamp_ns = event.native_timestamp_ns
                if event.status == NIXL_SUCCESS:
                    if record.state is not GroupedNixlTransferState.QUARANTINED:
                        record.state = GroupedNixlTransferState.TERMINAL_SUCCESS
                elif event.status == NIXL_IN_PROG:
                    raise NixlTerminalProcessFatalError(
                        "grouped terminal event carried an in-progress status",
                        batch.inventory,
                    )
                else:
                    if record.state is not GroupedNixlTransferState.QUARANTINED:
                        record.state = GroupedNixlTransferState.TERMINAL_FAILURE
                touched[record.binding_digest] = self._require_group_locked(
                    record.binding_digest
                )
            for group in touched.values():
                self._materialize_result_locked(group)
            claimed: list[GroupedNixlTerminalResult] = []
            for group in self._groups.values():
                if group.result_state is not GroupedNixlResultState.PENDING:
                    continue
                result = group.result
                if result is None:
                    raise RuntimeError("pending group result disappeared")
                group.result_state = GroupedNixlResultState.CLAIMED
                group.result_delivery_token = object()
                claimed.append(result)
            return tuple(claimed)

    def acknowledge_result(self, result: GroupedNixlTerminalResult) -> None:
        """Commit successful delivery of one aggregate result to lifecycle.

        :param result: Exact object returned by :meth:`drain`.
        """

        if type(result) is not GroupedNixlTerminalResult:
            raise TypeError("result must be GroupedNixlTerminalResult")
        with self._lock:
            group = self._require_group_locked(result.binding_digest)
            if group.result is not result:
                raise NixlTerminalLifecycleError(
                    "aggregate result belongs to another request group"
                )
            if group.result_state is not GroupedNixlResultState.CLAIMED:
                raise NixlTerminalLifecycleError(
                    "aggregate result is absent, stale, or already acknowledged"
                )
            group.result_state = GroupedNixlResultState.ACKNOWLEDGED
            group.result_delivery_token = None

    def _settle_success(
        self,
        member: GroupedNixlTransferMember,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> nixl_xfer_completion_receipt:
        """Take one member receipt under aggregate outcome authority.

        :param member: Exact semantic group member.
        :param transfer: Exact group member authority.
        :param action: Matching ``SOURCE_OUTCOME_READY`` action.
        :returns: Native take-once member completion receipt.
        """

        record = self._require_transfer(transfer)
        with self._lock:
            self._require_member_locked(record, member)
            group = self._settlement_group_locked(
                record,
                action,
                NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
            )
            if group.result is None or not group.result.successful:
                raise NixlTerminalLifecycleError(
                    "successful settlement lacks aggregate success authority"
                )
            if group.result_state is not GroupedNixlResultState.ACKNOWLEDGED:
                raise NixlTerminalLifecycleError(
                    "successful settlement preceded lifecycle ingress"
                )
            if record.state is not GroupedNixlTransferState.TERMINAL_SUCCESS:
                raise NixlTerminalLifecycleError(
                    "successful settlement requires member terminal success"
                )
            receipt = self._agent.take_xfer_completion_receipt(record.handle)
            if receipt is None:
                raise NixlTerminalLifecycleError(
                    "aggregate terminality has no member completion receipt"
                )
            self._validate_completion_receipt(record, receipt)
            record.completion_receipt = receipt
            record.state = GroupedNixlTransferState.SETTLED_SUCCESS
            return receipt

    def _settle_failure(
        self,
        member: GroupedNixlTransferMember,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Quarantine one member under exact native failure authority.

        :param member: Exact semantic group member.
        :param transfer: Exact group member authority.
        :param action: Matching quarantine or process-fatal action.
        """

        record = self._require_transfer(transfer)
        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in (
            NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
            NativeTerminalOwnerActionKind.PROCESS_FATAL,
        ):
            raise ValueError("failure settlement requires quarantine authority")
        with self._lock:
            self._require_member_locked(record, member)
            group = self._require_group_locked(record.binding_digest)
            self._bind_settlement_action_locked(group, record, action)
            group.quarantined = True
            if group.quarantine_reason is None:
                group.quarantine_reason = "native source lifecycle quarantined"
            record.state = GroupedNixlTransferState.QUARANTINED

    def _cancel_transfer(
        self,
        member: GroupedNixlTransferMember,
        transfer: object,
    ) -> None:
        """Request exact member cancellation without releasing ambiguity.

        :param member: Exact semantic group member.
        :param transfer: Exact group member authority.
        """

        record = self._require_transfer(transfer)
        with self._lock:
            self._require_member_locked(record, member)
            if record.cancel_requested:
                raise NixlTerminalLifecycleError(
                    "group member cancellation was already requested"
                )
            subscription = record.subscription
            if subscription is None:
                record.cancel_requested = True
                record.state = GroupedNixlTransferState.QUARANTINED
                return
            outcome = self._adapter.cancel(subscription)
            record.cancel_requested = True
            record.state = GroupedNixlTransferState.QUARANTINED
            if outcome is NixlTerminalCancelOutcome.RELEASED:
                record.subscription = None

    def _release_transfer(
        self,
        member: GroupedNixlTransferMember,
        transfer: object,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release one settled member after source teardown authority.

        :param member: Exact semantic group member.
        :param transfer: Exact successfully settled group member.
        :param action: Matching one-shot source ACK action.
        """

        record = self._require_transfer(transfer)
        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
            raise ValueError("group member release requires SOURCE_ACK_READY")
        with self._lock:
            self._require_member_locked(record, member)
            if action.binding.digest != record.binding_digest:
                raise NixlTerminalLifecycleError(
                    "release action belongs to another request group"
                )
            if record.state is not GroupedNixlTransferState.SETTLED_SUCCESS:
                raise NixlTerminalLifecycleError(
                    "group member release requires successful settlement"
                )
            if record.subscription is not None:
                raise NixlTerminalLifecycleError(
                    "group member release retained a native subscription"
                )
            group = self._require_group_locked(record.binding_digest)
            if group.release_action_id is None:
                group.release_action_id = action.action_id
            elif group.release_action_id != action.action_id:
                raise NixlTerminalLifecycleError(
                    "request group changed source ACK authority"
                )
            record.handle.release()
            record.state = GroupedNixlTransferState.RELEASED
            del self._transfers[record.public._token]
            del self._transfers_by_binding[record.subscription_binding]
            self._released_transfer_count += 1
            if all(
                member.state is GroupedNixlTransferState.RELEASED
                for member in group.transfers.values()
            ):
                del self._groups[group.binding_digest]

    def stop_admission(self) -> None:
        """Permanently reject new groups and member subscriptions."""

        with self._lock:
            self._require_not_closed_locked()
            self._admission_open = False

    def close_clean(self) -> GroupedNixlTerminalOwnerInventory:
        """Close the native channel only at exact-zero grouped authority.

        :returns: Final clean grouped and native inventory.
        """

        with self._lock:
            self._require_not_closed_locked()
            self._admission_open = False
            if (
                len(self._groups) != 0
                or len(self._transfers) != 0
                or len(self._transfers_by_binding) != 0
                or len(self._unowned_handles) != 0
            ):
                raise NixlTerminalLifecycleError(
                    "grouped owner cannot close with retained authority"
                )
            self._adapter.close()
            self._closed = True
            return self._inventory_locked()

    def inventory(self) -> GroupedNixlTerminalOwnerInventory:
        """Return exact grouped and native channel ownership evidence.

        :returns: Immutable conservation-complete inventory.
        """

        with self._lock:
            return self._inventory_locked()

    def _materialize_result_locked(self, group: _GroupedRequestRecord) -> None:
        """Create one aggregate result when sealed membership is terminal.

        :param group: Exact mutable request group.
        """

        if (
            not group.sealed
            or group.quarantined
            or group.result is not None
            or set(group.transfers) != set(group.expected_members)
        ):
            return
        ordered_transfers = tuple(
            group.transfers[member] for member in group.expected_members
        )
        failed_transfers = tuple(
            transfer
            for transfer in ordered_transfers
            if transfer.native_status is not None
            and transfer.native_status != NIXL_SUCCESS
        )
        if len(failed_transfers) > 0:
            failure_timestamps = tuple(
                transfer.native_timestamp_ns for transfer in failed_transfers
            )
            if any(timestamp is None for timestamp in failure_timestamps):
                raise RuntimeError("terminal member failure lost its timestamp")
            native_timestamp_ns = min(
                timestamp for timestamp in failure_timestamps if timestamp is not None
            )
            failed = tuple(
                f"cookie={transfer.owner_cookie}:status={transfer.native_status}"
                for transfer in failed_transfers
            )
            successful = False
            reason = "grouped NIXL member failure: " + ", ".join(failed)
        else:
            timestamps = tuple(
                transfer.native_timestamp_ns for transfer in ordered_transfers
            )
            statuses = tuple(transfer.native_status for transfer in ordered_transfers)
            if any(timestamp is None for timestamp in timestamps) or any(
                status is None for status in statuses
            ):
                return
            native_timestamp_ns = max(
                timestamp for timestamp in timestamps if timestamp is not None
            )
            successful = True
            reason = None
        group.result = GroupedNixlTerminalResult(
            binding_digest=group.binding_digest,
            successful=successful,
            transfer_count=len(group.expected_members),
            native_timestamp_ns=native_timestamp_ns,
            reason=reason,
            member_timings=tuple(
                GroupedNixlMemberTiming(
                    member=transfer.member,
                    owner_cookie=transfer.owner_cookie,
                    post_started_ns=self._require_post_timestamp(transfer),
                    native_terminal_ns=self._require_native_timestamp(transfer),
                )
                for transfer in ordered_transfers
                if transfer.native_timestamp_ns is not None
            ),
        )
        group.result_state = GroupedNixlResultState.PENDING

    def _settlement_group_locked(
        self,
        record: _GroupedTransferRecord,
        action: NativeTerminalOwnerAction,
        expected_kind: NativeTerminalOwnerActionKind,
    ) -> _GroupedRequestRecord:
        """Validate one member and request-level settlement action.

        :param record: Exact member record.
        :param action: Candidate lifecycle action.
        :param expected_kind: Required action kind.
        :returns: Matching request group.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not expected_kind:
            raise ValueError(f"settlement requires {expected_kind.name}")
        group = self._require_group_locked(record.binding_digest)
        self._bind_settlement_action_locked(group, record, action)
        return group

    @staticmethod
    def _bind_settlement_action_locked(
        group: _GroupedRequestRecord,
        record: _GroupedTransferRecord,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Bind every member settlement to one exact owner action.

        :param group: Exact request-level owner.
        :param record: Exact member owner.
        :param action: Candidate lifecycle action.
        """

        if action.binding.digest != group.binding_digest:
            raise NixlTerminalLifecycleError(
                "owner action belongs to another request group"
            )
        if group.settlement_action_id is None:
            group.settlement_action_id = action.action_id
        elif group.settlement_action_id != action.action_id:
            raise NixlTerminalLifecycleError(
                "request group changed settlement action authority"
            )
        if record.settlement_action_id is None:
            record.settlement_action_id = action.action_id
        elif record.settlement_action_id != action.action_id:
            raise NixlTerminalLifecycleError(
                "group member changed settlement action authority"
            )

    @staticmethod
    def _validate_completion_receipt(
        record: _GroupedTransferRecord,
        receipt: nixl_xfer_completion_receipt,
    ) -> None:
        """Validate take-once member success against its subscription.

        :param record: Exact terminal member record.
        :param receipt: Native completion authority.
        """

        if int(receipt.handleIdentity) != record.handle_identity:
            raise NixlTerminalLifecycleError(
                "completion receipt changed member handle identity"
            )
        if int(receipt.generation) != record.generation:
            raise NixlTerminalLifecycleError(
                "completion receipt changed member generation"
            )
        if not bool(receipt.submissionSealed) or not bool(receipt.completionClaimed):
            raise NixlTerminalLifecycleError(
                "completion receipt lacks sealed take-once authority"
            )
        if receipt.status.name != "NIXL_SUCCESS":
            raise NixlTerminalLifecycleError(
                "completion receipt does not carry successful terminal status"
            )

    @staticmethod
    def _require_post_timestamp(record: _GroupedTransferRecord) -> int:
        """Return one posted member's validated timing anchor.

        :param record: Exact terminal member record.
        :returns: Same-process post timestamp.
        """

        timestamp = record.post_started_ns
        if timestamp is None:
            raise RuntimeError("terminal group member was never posted")
        return timestamp

    @staticmethod
    def _require_native_timestamp(record: _GroupedTransferRecord) -> int:
        """Return one member's validated native terminal timestamp.

        :param record: Exact terminal member record.
        :returns: Native terminal-token timestamp.
        """

        timestamp = record.native_timestamp_ns
        if timestamp is None:
            raise RuntimeError("terminal group member lacks native terminality")
        return timestamp

    def _require_transfer(self, transfer: object) -> _GroupedTransferRecord:
        """Resolve one exact owner-private transfer token.

        :param transfer: Candidate transfer authority.
        :returns: Matching live transfer record.
        """

        if type(transfer) is not GroupedNixlTerminalTransfer:
            raise TypeError("transfer must be GroupedNixlTerminalTransfer")
        if transfer._owner_nonce is not self._owner_nonce:
            raise NixlTerminalLifecycleError(
                "grouped transfer belongs to another owner"
            )
        with self._lock:
            record = self._transfers.get(transfer._token)
            if record is None or record.public is not transfer:
                raise NixlTerminalLifecycleError(
                    "grouped transfer is absent, stale, or already released"
                )
            return record

    @staticmethod
    def _require_member_locked(
        record: _GroupedTransferRecord,
        member: GroupedNixlTransferMember,
    ) -> None:
        """Reject use of one transfer through another semantic endpoint.

        :param record: Exact retained transfer record.
        :param member: Endpoint member used by the caller.
        """

        if type(member) is not GroupedNixlTransferMember:
            raise TypeError("member must be GroupedNixlTransferMember")
        if record.member is not member:
            raise NixlTerminalLifecycleError(
                "grouped transfer was routed through another member endpoint"
            )

    def _require_group_locked(
        self,
        binding_digest: bytes,
    ) -> _GroupedRequestRecord:
        """Resolve one exact active request group while holding the lock.

        :param binding_digest: Candidate lifecycle digest.
        :returns: Matching live request group.
        """

        self._validate_binding_digest(binding_digest)
        group = self._groups.get(binding_digest)
        if group is None:
            raise NixlTerminalLifecycleError(
                "request group is absent, stale, or already released"
            )
        return group

    def _allocate_owner_cookie_locked(self) -> int:
        """Allocate one never-reused positive native owner cookie.

        :returns: Unique process-lifetime native correlation cookie.
        """

        owner_cookie = self._next_owner_cookie
        if owner_cookie > _UINT64_MAX:
            raise OverflowError("grouped NIXL owner cookie space is exhausted")
        self._next_owner_cookie += 1
        return owner_cookie

    def _inventory_locked(self) -> GroupedNixlTerminalOwnerInventory:
        """Build immutable inventory while holding the owner lock.

        :returns: Current conservation-complete inventory.
        """

        native = self._adapter.query_inventory()
        groups = tuple(self._groups.values())
        transfers = tuple(self._transfers.values())
        terminal_states = (
            GroupedNixlTransferState.TERMINAL_SUCCESS,
            GroupedNixlTransferState.TERMINAL_FAILURE,
        )
        return GroupedNixlTerminalOwnerInventory(
            native=native,
            admission_open=self._admission_open,
            closed=self._closed,
            active_group_count=len(groups),
            sealed_group_count=sum(group.sealed for group in groups),
            pending_result_count=sum(
                group.result_state
                in (
                    GroupedNixlResultState.PENDING,
                    GroupedNixlResultState.CLAIMED,
                )
                for group in groups
            ),
            acknowledged_result_count=sum(
                group.result_state is GroupedNixlResultState.ACKNOWLEDGED
                for group in groups
            ),
            active_transfer_count=len(transfers),
            terminal_transfer_count=sum(
                transfer.state in terminal_states for transfer in transfers
            ),
            settled_transfer_count=sum(
                transfer.state is GroupedNixlTransferState.SETTLED_SUCCESS
                for transfer in transfers
            ),
            quarantined_transfer_count=sum(
                transfer.state is GroupedNixlTransferState.QUARANTINED
                for transfer in transfers
            ),
            released_transfer_count=self._released_transfer_count,
            unowned_handle_count=len(self._unowned_handles),
        )

    def _require_accepting_locked(self) -> None:
        """Require open request and member admission."""

        self._require_not_closed_locked()
        if not self._admission_open:
            raise NixlTerminalLifecycleError(
                "grouped NIXL terminal admission is closed"
            )

    def _require_not_closed_locked(self) -> None:
        """Reject operations after exact-zero channel closure."""

        if self._closed:
            raise NixlTerminalLifecycleError(
                "grouped NIXL terminal owner is already closed"
            )

    @staticmethod
    def _validate_binding_digest(binding_digest: bytes) -> None:
        """Validate one exact source lifecycle digest.

        :param binding_digest: Candidate lifecycle identity.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
