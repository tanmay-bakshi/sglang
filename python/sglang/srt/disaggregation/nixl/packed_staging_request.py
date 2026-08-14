import dataclasses
import enum
import logging
import secrets
import threading
import traceback

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.decode_allocation_lease import (
    DecodeAllocationComponent,
    DecodeAllocationComponentReceipt,
    DecodeAllocationLease,
    DecodeAllocationLeaseAuthority,
    DecodeAllocationLeaseError,
    DecodeAllocationLeaseSnapshot,
    DecodeAllocationLeaseState,
)
from sglang.srt.disaggregation.common.packed_auxiliary_allocation import (
    PackedAuxiliaryAllocationLease,
    PackedAuxiliaryAllocationLeaseAuthority,
    PackedAuxiliaryAllocationLeaseSnapshot,
    PackedAuxiliaryAllocationState,
    PackedAuxiliarySlotReservationSnapshot,
)
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_TEARDOWN_GENERATION_BYTES,
    PackedAuxiliaryOutcome,
    PackedAuxiliaryPlan,
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedDFlashBoundaryMetadata,
    PackedDFlashBoundaryOutcome,
    PackedLayoutSpec,
    PackedPrepare,
    PackedProtocolError,
    PackedProtocolState,
    PackedReady,
    PackedRequestKey,
    PackedRequestTeardown,
    PackedRequestTeardownAck,
    PackedScatterWork,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
)
from sglang.srt.disaggregation.common.staging_layout import (
    StagingComponentId,
    StagingWriterId,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBufferRegistry,
)
from sglang.srt.disaggregation.nixl.packed_staging import (
    MAIN_KV_COMPONENT,
    PackedDestinationOutcomeCoordinator,
    PackedDestinationVisibilityError,
    PackedDestinationVisibilityPolicy,
    PackedDestinationVisibilityProof,
)

logger = logging.getLogger(__name__)

_SCATTER_CONSTRUCTION_SEAL = object()
_COMMIT_CONSTRUCTION_SEAL = object()

PackedTerminalAuxiliaryOutcome = (
    PackedAuxiliaryOutcome | PackedDFlashBoundaryOutcome
)


class PackedRequestTransactionError(RuntimeError):
    """Request-scoped packed staging invariant violation."""


class PackedRequestTransactionState(enum.StrEnum):
    """Decode-side ownership state spanning every request chunk."""

    PREPARED = "prepared"
    PUBLISHED = "published"
    SUBMITTED = "submitted"
    WRITERS_COMPLETED = "writers_completed"
    SCATTER_COMPLETED = "scatter_completed"
    TEARDOWN_WAITING = "teardown_waiting"
    COMMIT_READY = "commit_ready"
    DESTINATION_CONSUMPTION_WAITING = "destination_consumption_waiting"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


@dataclasses.dataclass(frozen=True)
class PackedRequestChunkPlan:
    """Decoder-authored immutable plan for one request chunk.

    :ivar key: Exact request and chunk generation.
    :ivar spec: Decode-local canonical packed layout input.
    :ivar destination_registry: Complete allocation-derived destination pages.
    :ivar visibility_policies: Canonical writer route policies.
    """

    key: PackedChunkKey
    spec: PackedLayoutSpec
    destination_registry: StagingComponentBufferRegistry
    visibility_policies: tuple[
        tuple[StagingWriterId, PackedDestinationVisibilityPolicy], ...
    ]

    def __post_init__(self) -> None:
        """Own and validate one immutable chunk declaration."""

        if type(self.key) is not PackedChunkKey:
            raise TypeError("chunk plan key must be PackedChunkKey")
        if type(self.spec) is not PackedLayoutSpec:
            raise TypeError("chunk plan spec must be PackedLayoutSpec")
        if type(self.destination_registry) is not StagingComponentBufferRegistry:
            raise TypeError(
                "chunk plan destination_registry must be StagingComponentBufferRegistry"
            )
        policies = tuple(self.visibility_policies)
        object.__setattr__(self, "visibility_policies", policies)
        for writer_id, policy in policies:
            if type(writer_id) is not StagingWriterId:
                raise TypeError("chunk plan policy writer must be StagingWriterId")
            if type(policy) is not PackedDestinationVisibilityPolicy:
                raise TypeError(
                    "chunk plan policy must be PackedDestinationVisibilityPolicy"
                )

    @property
    def policy_map(
        self,
    ) -> dict[StagingWriterId, PackedDestinationVisibilityPolicy]:
        """Return an exact mutable copy for protocol registration.

        :returns: Writer policy mapping.
        """

        policies = dict(self.visibility_policies)
        if len(policies) != len(self.visibility_policies):
            raise ValueError("chunk plan contains duplicate writer policies")
        return policies


@dataclasses.dataclass(frozen=True)
class PackedRequestPublication:
    """Metadata made externally visible for one prepared request.

    :ivar key: Exact request generation.
    :ivar request_slot_generation: Decode request-slot reuse generation.
    :ivar writer_manifest_digest: Exact source writer membership.
    :ivar allocation_digest: Exact destination allocation identity.
    :ivar auxiliary_plan: Exact authority-derived metadata transfer plan.
    :ivar chunk_specs: Decoder-prescribed fixed chunk boundaries.
    :ivar terminal_source_plan: Exact encoded terminal source authority, when
        terminal serving owns this request.
    """

    key: PackedRequestKey
    request_slot_generation: int
    writer_manifest_digest: bytes
    allocation_digest: bytes
    auxiliary_plan: PackedAuxiliaryPlan
    chunk_specs: tuple[PackedLayoutSpec, ...]
    terminal_source_plan: bytes | None = None

    def __post_init__(self) -> None:
        """Own and validate externally visible request metadata."""

        if type(self.key) is not PackedRequestKey:
            raise TypeError("publication key must be PackedRequestKey")
        if type(self.auxiliary_plan) is not PackedAuxiliaryPlan:
            raise TypeError("publication auxiliary_plan must be PackedAuxiliaryPlan")
        if self.auxiliary_plan.key != self.key:
            raise ValueError("publication auxiliary plan key differs from request")
        if self.auxiliary_plan.request_slot_generation != self.request_slot_generation:
            raise ValueError(
                "publication auxiliary slot generation differs from request"
            )
        object.__setattr__(self, "chunk_specs", tuple(self.chunk_specs))
        if self.terminal_source_plan is not None:
            if type(self.terminal_source_plan) is not bytes:
                raise TypeError("terminal source plan must be bytes")
            if len(self.terminal_source_plan) == 0:
                raise ValueError("terminal source plan must not be empty")


class PackedRequestScatter:
    """Opaque ownership of one begun destination scatter."""

    __slots__ = ("_token", "_transaction_nonce", "proofs", "work")

    _transaction_nonce: object
    _token: object
    proofs: tuple[PackedDestinationVisibilityProof, ...]
    work: PackedScatterWork

    def __init__(
        self,
        transaction_nonce: object,
        token: object,
        work: PackedScatterWork,
        proofs: tuple[PackedDestinationVisibilityProof, ...],
        construction_seal: object,
    ) -> None:
        """Construct one transaction-owned scatter permit.

        :param transaction_nonce: Exact issuing transaction.
        :param token: Transaction-private chunk record identity.
        :param work: Protocol-owned scatter work.
        :param proofs: Exact canonical visibility proofs.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _SCATTER_CONSTRUCTION_SEAL:
            raise TypeError("packed request scatters are transaction owned")
        self._transaction_nonce = transaction_nonce
        self._token = token
        self.work = work
        self.proofs = tuple(proofs)


class PackedRequestCommitReceipt:
    """Opaque one-shot proof that request-level teardown is complete."""

    __slots__ = ("_token", "_transaction_nonce")

    _transaction_nonce: object
    _token: object

    def __init__(
        self,
        transaction_nonce: object,
        token: object,
        construction_seal: object,
    ) -> None:
        """Construct one transaction-owned commit receipt.

        :param transaction_nonce: Exact issuing transaction.
        :param token: Exact one-shot receipt identity.
        :param construction_seal: Module-private construction authority.
        """

        if construction_seal is not _COMMIT_CONSTRUCTION_SEAL:
            raise TypeError("packed request commit receipts are transaction owned")
        self._transaction_nonce = transaction_nonce
        self._token = token


@dataclasses.dataclass(frozen=True)
class PackedRequestTransactionSnapshot:
    """Immutable diagnostic view of one request transaction.

    :ivar key: Exact request identity.
    :ivar state: Request-level ownership state.
    :ivar chunk_states: Canonical chunk protocol states.
    :ivar scatter_started: Chunk IDs with handed-off scatter ownership.
    :ivar scatter_terminal: Chunk IDs with terminal successful scatter.
    :ivar teardown_acks: Canonically ordered acknowledged writers.
    :ivar auxiliary_outcome: Exact terminal auxiliary outcome, if received.
    :ivar auxiliary_teardown_acknowledged: Whether its exact handle was retired.
    """

    key: PackedRequestKey
    state: PackedRequestTransactionState
    chunk_states: tuple[PackedProtocolState, ...]
    scatter_started: tuple[int, ...]
    scatter_terminal: tuple[int, ...]
    teardown_acks: tuple[StagingWriterId, ...]
    auxiliary_outcome: PackedTerminalAuxiliaryOutcome | None
    auxiliary_teardown_acknowledged: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PackedDFlashBoundaryDecodeAdoption:
    """Authenticated destination row and scalar state for scheduler adoption.

    :ivar metadata: Source-authored DFlash boundary and request counters.
    :ivar lease: Exact committed destination VRAM row generation and state.
    :ivar outcome_digest: Identity of the authenticated native transfer outcome.
    """

    metadata: PackedDFlashBoundaryMetadata
    lease: PackedAuxiliaryAllocationLeaseSnapshot
    outcome_digest: bytes

    def __post_init__(self) -> None:
        """Validate immutable scheduler adoption authority."""

        if type(self.metadata) is not PackedDFlashBoundaryMetadata:
            raise TypeError("metadata must be PackedDFlashBoundaryMetadata")
        if type(self.lease) is not PackedAuxiliaryAllocationLeaseSnapshot:
            raise TypeError("lease must be PackedAuxiliaryAllocationLeaseSnapshot")
        if self.lease.state is not PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST:
            raise ValueError("DFlash boundary row is not committed to its request")
        if type(self.outcome_digest) is not bytes or len(self.outcome_digest) != 32:
            raise ValueError("outcome_digest must contain 32 bytes")

    @property
    def slot(self) -> PackedAuxiliarySlotReservationSnapshot:
        """Project the committed lease to its live pool reservation geometry.

        :returns: Exact row index, generation, and registered segment.
        """

        return PackedAuxiliarySlotReservationSnapshot(
            metadata_buffer_index=self.lease.metadata_buffer_index,
            metadata_slot_generation=self.lease.metadata_slot_generation,
            destination_segments=self.lease.destination_segments,
        )


@dataclasses.dataclass
class _PackedRequestChunk:
    """Mutable request-owned state for one registered protocol chunk."""

    plan: PackedRequestChunkPlan
    token: object = dataclasses.field(default_factory=object)
    scatter: PackedRequestScatter | None = None
    scatter_terminal: bool = False
    retired: bool = False


class PackedDecodeRequestTransaction:
    """Aggregate all packed chunks into one decode allocation transaction."""

    _auxiliary_allocation_authority: PackedAuxiliaryAllocationLeaseAuthority
    _auxiliary_allocation_lease: PackedAuxiliaryAllocationLease
    _auxiliary_outcome: PackedTerminalAuxiliaryOutcome | None
    _auxiliary_plan: PackedAuxiliaryPlan
    _auxiliary_committed_to_request: bool
    _auxiliary_released: bool
    _auxiliary_retired: bool
    _auxiliary_submission_recorded: bool
    _auxiliary_teardown_recorded: bool
    _dflash_boundary_adoption: PackedDFlashBoundaryDecodeAdoption | None
    _allocation_authority: DecodeAllocationLeaseAuthority
    _allocation_committed: bool
    _allocation_lease: DecodeAllocationLease
    _allocation_retired: bool
    _allocation_snapshot: DecodeAllocationLeaseSnapshot
    _chunks: tuple[_PackedRequestChunk, ...]
    _chunks_by_key: dict[PackedChunkKey, _PackedRequestChunk]
    _commit_receipt: PackedRequestCommitReceipt | None
    _lifecycle_authority: object
    _lock: threading.RLock
    _outcome_coordinator: PackedDestinationOutcomeCoordinator
    _protocol: PackedDecodeProtocol
    _publication: PackedRequestPublication
    _request_key: PackedRequestKey
    _request_owner: object
    _scheduler_thread_id: int
    _state: PackedRequestTransactionState
    _submission_recorded: bool
    _teardown_acks: set[StagingWriterId]
    _teardown_requests: tuple[PackedRequestTeardown, ...]
    _terminal_binding_digest: bytes | None
    _transaction_nonce: object
    _writer_completion_recorded: bool

    def __init__(
        self,
        *,
        room_id: int,
        request_owner: object,
        allocation_lease: DecodeAllocationLease,
        allocation_authority: DecodeAllocationLeaseAuthority,
        lifecycle_authority: object,
        protocol: PackedDecodeProtocol,
        outcome_coordinator: PackedDestinationOutcomeCoordinator,
        chunk_plans: tuple[PackedRequestChunkPlan, ...],
        auxiliary_allocation_lease: PackedAuxiliaryAllocationLease,
        auxiliary_allocation_authority: PackedAuxiliaryAllocationLeaseAuthority,
        destination_process_generation: bytes,
        native_route_digest: bytes,
        runtime_cohort_digest: bytes,
        scheduler_thread_id: int | None = None,
    ) -> None:
        """Validate and register one complete decoder-authored request plan.

        :param room_id: Decoder-minted non-recycled bootstrap room.
        :param request_owner: Exact retained decode request.
        :param allocation_lease: Exact migration-pinned decode allocation.
        :param allocation_authority: Allocation lease authority.
        :param lifecycle_authority: Exact authority bound to allocation transitions.
        :param protocol: Shared arena chunk protocol.
        :param outcome_coordinator: Shared destination visibility coordinator.
        :param chunk_plans: Every fixed request chunk in canonical order.
        :param auxiliary_allocation_lease: Exact retained metadata-row lease.
        :param auxiliary_allocation_authority: Metadata lease authority.
        :param destination_process_generation: Exact destination process
            generation.
        :param native_route_digest: Decode-selected native route digest.
        :param runtime_cohort_digest: Exact loaded runtime cohort digest.
        :param scheduler_thread_id: Owning scheduler thread, defaults to the caller.
        """

        if request_owner is None:
            raise ValueError("packed request owner must not be None")
        if type(allocation_lease) is not DecodeAllocationLease:
            raise TypeError("allocation_lease must be DecodeAllocationLease")
        if type(allocation_authority) is not DecodeAllocationLeaseAuthority:
            raise TypeError(
                "allocation_authority must be DecodeAllocationLeaseAuthority"
            )
        if type(auxiliary_allocation_lease) is not PackedAuxiliaryAllocationLease:
            raise TypeError(
                "auxiliary_allocation_lease must be PackedAuxiliaryAllocationLease"
            )
        if (
            type(auxiliary_allocation_authority)
            is not PackedAuxiliaryAllocationLeaseAuthority
        ):
            raise TypeError(
                "auxiliary_allocation_authority must be "
                "PackedAuxiliaryAllocationLeaseAuthority"
            )
        if type(protocol) is not PackedDecodeProtocol:
            raise TypeError("protocol must be PackedDecodeProtocol")
        if type(outcome_coordinator) is not PackedDestinationOutcomeCoordinator:
            raise TypeError(
                "outcome_coordinator must be PackedDestinationOutcomeCoordinator"
            )
        owner_thread_id = (
            threading.get_ident()
            if scheduler_thread_id is None
            else scheduler_thread_id
        )
        if type(owner_thread_id) is not int or owner_thread_id <= 0:
            raise ValueError("scheduler_thread_id must be a positive integer")

        snapshot = allocation_authority.snapshot(allocation_lease)
        if snapshot.state is not DecodeAllocationLeaseState.PREPARED:
            raise PackedRequestTransactionError(
                "packed request requires a prepared decode allocation"
            )
        request_key = PackedRequestKey(
            room_id=room_id,
            request_generation=snapshot.lease_id,
        )
        auxiliary_snapshot = auxiliary_allocation_authority.snapshot(
            auxiliary_allocation_lease
        )
        if auxiliary_snapshot.state is not PackedAuxiliaryAllocationState.PREPARED:
            raise PackedRequestTransactionError(
                "packed request requires a prepared auxiliary allocation"
            )
        plans = tuple(chunk_plans)
        self._validate_plans(request_key, snapshot, plans)
        canonical_writer = self._canonical_auxiliary_writer(snapshot)
        auxiliary_plan = PackedAuxiliaryPlan(
            key=request_key,
            request_slot_generation=snapshot.request_generation,
            metadata_buffer_index=auxiliary_snapshot.metadata_buffer_index,
            metadata_slot_generation=auxiliary_snapshot.metadata_slot_generation,
            destination_segments=auxiliary_snapshot.destination_segments,
            canonical_writer_id=canonical_writer,
            destination_process_generation=destination_process_generation,
            native_route_digest=native_route_digest,
            runtime_cohort_digest=runtime_cohort_digest,
        )

        chunks = tuple(_PackedRequestChunk(plan=plan) for plan in plans)
        self._auxiliary_allocation_authority = auxiliary_allocation_authority
        self._auxiliary_allocation_lease = auxiliary_allocation_lease
        self._auxiliary_outcome = None
        self._auxiliary_plan = auxiliary_plan
        self._auxiliary_committed_to_request = False
        self._auxiliary_released = False
        self._auxiliary_retired = False
        self._auxiliary_submission_recorded = False
        self._auxiliary_teardown_recorded = False
        self._dflash_boundary_adoption = None
        self._allocation_authority = allocation_authority
        self._allocation_committed = False
        self._allocation_lease = allocation_lease
        self._allocation_retired = False
        self._allocation_snapshot = snapshot
        self._chunks = chunks
        self._chunks_by_key = {chunk.plan.key: chunk for chunk in chunks}
        self._commit_receipt = None
        self._lifecycle_authority = lifecycle_authority
        self._lock = threading.RLock()
        self._outcome_coordinator = outcome_coordinator
        self._protocol = protocol
        self._publication = PackedRequestPublication(
            key=request_key,
            request_slot_generation=snapshot.request_generation,
            writer_manifest_digest=snapshot.writer_manifest.digest,
            allocation_digest=snapshot.allocation_digest,
            auxiliary_plan=auxiliary_plan,
            chunk_specs=tuple(plan.spec for plan in plans),
        )
        self._request_key = request_key
        self._request_owner = request_owner
        self._scheduler_thread_id = owner_thread_id
        self._state = PackedRequestTransactionState.PREPARED
        self._submission_recorded = False
        self._teardown_acks = set()
        self._teardown_requests = ()
        self._terminal_binding_digest = None
        self._transaction_nonce = object()
        self._writer_completion_recorded = False
        self._register_chunks()

    @property
    def allocation_lease(self) -> DecodeAllocationLease:
        """Return the exact retained decode allocation lease.

        :returns: Process-local allocation lease.
        """

        return self._allocation_lease

    @property
    def request_owner(self) -> object:
        """Return the exact retained decode request owner.

        :returns: Identity-stable request owner.
        """

        return self._request_owner

    @property
    def state(self) -> PackedRequestTransactionState:
        """Return a stable request transaction state.

        :returns: Current transaction state.
        """

        with self._lock:
            return self._state

    def prepared_publication(self) -> PackedRequestPublication:
        """Return immutable metadata while publication remains reversible.

        :returns: Exact decoder-authored metadata before wire visibility.
        """

        self._require_scheduler_thread()
        with self._lock:
            if self._state is not PackedRequestTransactionState.PREPARED:
                raise PackedRequestTransactionError(
                    "prepared publication is available only before publication"
                )
            return self._publication

    def bind_terminal_owner_authority(
        self,
        encoded_source_plan: bytes,
        binding_digest: bytes,
    ) -> None:
        """Bind exact wire authority to a registered terminal owner.

        The packed actor calls this only after validating the source plan and
        reserving the rank-local terminal binding. Keeping the payload and the
        owner marker in the transaction makes publication itself enforce the
        ordering boundary instead of relying on caller discipline.

        :param encoded_source_plan: Canonical source-plan codec output.
        :param binding_digest: Exact rank-local terminal binding digest.
        """

        self._require_scheduler_thread()
        if type(encoded_source_plan) is not bytes or len(encoded_source_plan) == 0:
            raise ValueError("encoded_source_plan must be nonempty bytes")
        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        with self._lock:
            if self._state is not PackedRequestTransactionState.PREPARED:
                raise PackedRequestTransactionError(
                    "terminal owner authority must bind before publication"
                )
            if self._terminal_binding_digest is not None:
                raise PackedRequestTransactionError(
                    "terminal owner authority was already bound"
                )
            if self._publication.terminal_source_plan is not None:
                raise PackedRequestTransactionError(
                    "terminal source plan was already bound"
                )
            self._publication = dataclasses.replace(
                self._publication,
                terminal_source_plan=encoded_source_plan,
            )
            self._terminal_binding_digest = binding_digest

    def publish(self) -> PackedRequestPublication:
        """Make the exact allocation generation irreversible before wire I/O.

        :returns: Immutable decoder-authored request metadata.
        """

        self._require_scheduler_thread()
        with self._lock:
            if self._state is not PackedRequestTransactionState.PREPARED:
                raise PackedRequestTransactionError(
                    f"publication is invalid in state {self._state.value}"
                )
            terminal_payload = self._publication.terminal_source_plan
            terminal_binding = self._terminal_binding_digest
            if (terminal_payload is None) != (terminal_binding is None):
                raise PackedRequestTransactionError(
                    "terminal source plan and owner registration are incomplete"
                )
            try:
                self._auxiliary_allocation_authority.record_publication(
                    self._auxiliary_allocation_lease,
                    self._lifecycle_authority,
                )
                self._allocation_authority.record_publication(
                    self._allocation_lease,
                    self._lifecycle_authority,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                self._quarantine_locked("packed request publication failed")
                raise PackedRequestTransactionError(
                    "packed request publication failed"
                ) from error
            self._state = PackedRequestTransactionState.PUBLISHED
            return self._publication

    def cancel_unpublished(self) -> object:
        """Rollback one exact request before metadata can be externally visible.

        :returns: Exact retained request whose transaction ownership was retired.
        """

        self._require_scheduler_thread()
        with self._lock:
            if self._state is not PackedRequestTransactionState.PREPARED:
                raise PackedRequestTransactionError(
                    f"unpublished cancellation is invalid in state {self._state.value}"
                )
            try:
                for chunk in reversed(self._chunks):
                    self._outcome_coordinator.cancel_unpublished_chunk(chunk.plan.key)
                    chunk.retired = True
                self._auxiliary_allocation_authority.cancel_unpublished(
                    self._auxiliary_allocation_lease
                )
                self._auxiliary_released = True
                self._auxiliary_allocation_authority.retire_terminal(
                    self._auxiliary_allocation_lease
                )
                self._auxiliary_retired = True
                self._allocation_authority.rollback_to_request(self._allocation_lease)
                self._allocation_authority.retire_terminal(self._allocation_lease)
                self._allocation_retired = True
            except (RuntimeError, TypeError, ValueError) as error:
                reason = "unpublished packed request cancellation failed"
                self._quarantine_locked(reason)
                raise PackedRequestTransactionError(reason) from error
            self._state = PackedRequestTransactionState.CANCELLED
            return self._request_owner

    def handle_prepare(
        self,
        message: PackedPrepare,
        authenticated_writer_id: StagingWriterId,
    ) -> tuple[PackedReady, ...]:
        """Accept one authenticated source declaration after publication.

        :param message: Untrusted writer PREPARE.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: READY messages after final unique-writer consensus.
        """

        with self._lock:
            self._require_active_chunk_locked(message.key)
            if self._state not in (
                PackedRequestTransactionState.PUBLISHED,
                PackedRequestTransactionState.SUBMITTED,
            ):
                raise PackedRequestTransactionError(
                    f"PREPARE is invalid in state {self._state.value}"
                )
            try:
                return self._protocol.handle_prepare(
                    message,
                    authenticated_writer_id,
                )
            except PackedProtocolError as error:
                self._quarantine_locked(error.reason)
                raise

    def handle_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Record one terminal writer result and aggregate request completion.

        :param message: Authenticated terminal writer outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this exact chunk newly became scatter eligible.
        """

        with self._lock:
            self._require_active_chunk_locked(message.key)
            if self._state not in (
                PackedRequestTransactionState.PUBLISHED,
                PackedRequestTransactionState.SUBMITTED,
                PackedRequestTransactionState.WRITERS_COMPLETED,
                PackedRequestTransactionState.QUARANTINED,
            ):
                raise PackedRequestTransactionError(
                    f"writer outcome is invalid in state {self._state.value}"
                )
            try:
                admission_required = self._protocol.preflight_writer_outcome(
                    message,
                    authenticated_writer_id,
                )
                if (
                    admission_required
                    and message.status is PackedWriterOutcomeStatus.DONE
                    and not self._submission_recorded
                ):
                    self._allocation_authority.record_submission(
                        self._allocation_lease,
                        self._lifecycle_authority,
                    )
                    self._submission_recorded = True
                    self._state = PackedRequestTransactionState.SUBMITTED
                scatter_ready = self._outcome_coordinator.handle_writer_outcome(
                    message,
                    authenticated_writer_id,
                )
                self._advance_writer_completion_locked()
                return scatter_ready
            except (
                DecodeAllocationLeaseError,
                PackedProtocolError,
                PackedDestinationVisibilityError,
            ) as error:
                self._quarantine_locked(str(error))
                raise

    def handle_auxiliary_outcome(
        self,
        message: PackedAuxiliaryOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Bind one exact terminal metadata transfer to request ownership.

        :param message: Untrusted terminal auxiliary outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this exact outcome was newly accepted.
        """

        if type(message) is not PackedAuxiliaryOutcome:
            raise TypeError("message must be PackedAuxiliaryOutcome")
        return self._handle_terminal_auxiliary_outcome(
            message,
            authenticated_writer_id,
        )

    def handle_dflash_boundary_outcome(
        self,
        message: PackedDFlashBoundaryOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Bind one authenticated all-VRAM DFlash boundary outcome.

        :param message: Untrusted terminal DFlash boundary outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this exact outcome was newly accepted.
        """

        if type(message) is not PackedDFlashBoundaryOutcome:
            raise TypeError("message must be PackedDFlashBoundaryOutcome")
        return self._handle_terminal_auxiliary_outcome(
            message,
            authenticated_writer_id,
        )

    def _handle_terminal_auxiliary_outcome(
        self,
        message: PackedTerminalAuxiliaryOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Apply shared authenticated auxiliary lifecycle transitions.

        :param message: Validated legacy or DFlash terminal outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether this exact outcome was newly accepted.
        """

        if type(authenticated_writer_id) is not StagingWriterId:
            raise TypeError("authenticated_writer_id must be StagingWriterId")
        with self._lock:
            if self._state in (
                PackedRequestTransactionState.CANCELLED,
                PackedRequestTransactionState.COMMITTED,
            ):
                raise PackedRequestTransactionError(
                    f"auxiliary outcome is invalid in state {self._state.value}"
                )
            failure_reason: str | None = None
            if message.plan != self._auxiliary_plan:
                failure_reason = (
                    "auxiliary outcome plan differs from the exact request plan"
                )
            elif message.writer_id != self._auxiliary_plan.canonical_writer_id:
                failure_reason = (
                    "auxiliary outcome writer differs from the canonical writer"
                )
            elif authenticated_writer_id != self._auxiliary_plan.canonical_writer_id:
                failure_reason = (
                    "auxiliary outcome peer differs from the canonical writer"
                )
            elif message.writer_id != authenticated_writer_id:
                failure_reason = (
                    "auxiliary outcome writer differs from its authenticated peer"
                )
            if failure_reason is not None:
                self._quarantine_locked(failure_reason)
                raise PackedRequestTransactionError(failure_reason)
            if self._auxiliary_outcome is not None:
                if message == self._auxiliary_outcome:
                    return False
                failure_reason = "conflicting duplicate auxiliary outcome"
                self._quarantine_locked(failure_reason)
                raise PackedRequestTransactionError(failure_reason)
            if self._state is PackedRequestTransactionState.PREPARED:
                failure_reason = "auxiliary outcome arrived before publication"
                self._quarantine_locked(failure_reason)
                raise PackedRequestTransactionError(failure_reason)
            if self._state is PackedRequestTransactionState.QUARANTINED:
                raise PackedRequestTransactionError(
                    "auxiliary outcome cannot mutate a quarantined request"
                )
            self._auxiliary_outcome = message
            try:
                if not self._auxiliary_submission_recorded:
                    self._auxiliary_allocation_authority.record_submission(
                        self._auxiliary_allocation_lease,
                        self._lifecycle_authority,
                    )
                    self._auxiliary_submission_recorded = True
                if not self._submission_recorded:
                    self._allocation_authority.record_submission(
                        self._allocation_lease,
                        self._lifecycle_authority,
                    )
                    self._submission_recorded = True
            except (RuntimeError, TypeError, ValueError) as error:
                self._quarantine_locked(
                    "auxiliary outcome submission transition failed"
                )
                raise PackedRequestTransactionError(
                    "auxiliary outcome submission transition failed"
                ) from error
            self._state = PackedRequestTransactionState.SUBMITTED
            self._advance_writer_completion_locked()
            return True

    def begin_scatter(self, key: PackedChunkKey) -> PackedRequestScatter:
        """Hand one scatter-ready chunk to asynchronous destination work.

        :param key: Exact request-local chunk identity.
        :returns: Opaque scatter ownership and canonical work inputs.
        """

        with self._lock:
            chunk = self._require_active_chunk_locked(key)
            if self._state in (
                PackedRequestTransactionState.PREPARED,
                PackedRequestTransactionState.QUARANTINED,
                PackedRequestTransactionState.CANCELLED,
                PackedRequestTransactionState.COMMITTED,
            ):
                raise PackedRequestTransactionError(
                    f"scatter is invalid in state {self._state.value}"
                )
            if chunk.scatter is not None:
                raise PackedRequestTransactionError(
                    f"chunk {key.chunk_id} scatter already owns asynchronous work"
                )
            proofs = self._outcome_coordinator.proofs(
                key,
                self._allocation_snapshot.writer_manifest.writers,
            )
            work = self._protocol.begin_scatter(key)
            scatter = PackedRequestScatter(
                self._transaction_nonce,
                chunk.token,
                work,
                proofs,
                _SCATTER_CONSTRUCTION_SEAL,
            )
            chunk.scatter = scatter
            return scatter

    def complete_scatter(self, scatter: PackedRequestScatter) -> None:
        """Record exact terminal completion of one successful scatter.

        :param scatter: Exact transaction-owned scatter.
        """

        with self._lock:
            chunk = self._validate_scatter_locked(scatter)
            if chunk.scatter_terminal:
                raise PackedRequestTransactionError(
                    f"chunk {chunk.plan.key.chunk_id} scatter already completed"
                )
            self._protocol.complete_scatter(chunk.plan.key)
            chunk.scatter_terminal = True
            self._advance_scatter_completion_locked()

    def fail_scatter(
        self,
        scatter: PackedRequestScatter,
        reason: str,
        *,
        quiesced: bool,
    ) -> None:
        """Fail one begun scatter and retain unsafe ownership conservatively.

        :param scatter: Exact transaction-owned scatter.
        :param reason: Stable failure reason.
        :param quiesced: Whether the destination scatter stream is terminal.
        """

        if len(reason) == 0:
            raise ValueError("scatter failure reason must not be empty")
        with self._lock:
            chunk = self._validate_scatter_locked(scatter)
            self._protocol.fail_scatter(chunk.plan.key, reason)
            if quiesced:
                self._protocol.quiesce_scatter(chunk.plan.key)
            self._quarantine_locked(reason)

    def begin_teardown(self) -> tuple[PackedRequestTeardown, ...]:
        """Create one request-level teardown barrier entry per source writer.

        :returns: Canonically ordered teardown requests.
        """

        with self._lock:
            if self._state is not PackedRequestTransactionState.SCATTER_COMPLETED:
                raise PackedRequestTransactionError(
                    f"teardown is invalid in state {self._state.value}"
                )
            outcome = self._auxiliary_outcome
            if outcome is None:
                raise PackedRequestTransactionError(
                    "teardown requires a terminal auxiliary outcome"
                )
            generation = secrets.token_bytes(PACKED_TEARDOWN_GENERATION_BYTES)
            snapshot = self._allocation_snapshot
            self._teardown_requests = tuple(
                PackedRequestTeardown(
                    key=self._request_key,
                    writer_id=writer_id,
                    request_slot_generation=snapshot.request_generation,
                    writer_manifest_digest=snapshot.writer_manifest.digest,
                    allocation_digest=snapshot.allocation_digest,
                    teardown_generation=generation,
                    auxiliary_handle_generation=(
                        self._terminal_auxiliary_handle_generation(outcome)
                        if writer_id == self._auxiliary_plan.canonical_writer_id
                        else None
                    ),
                )
                for writer_id in snapshot.writer_manifest.writers
            )
            self._state = PackedRequestTransactionState.TEARDOWN_WAITING
            return self._teardown_requests

    def handle_teardown_ack(
        self,
        message: PackedRequestTeardownAck,
        authenticated_writer_id: StagingWriterId,
    ) -> PackedRequestCommitReceipt | None:
        """Authenticate one writer acknowledgement and issue the final receipt.

        :param message: Untrusted teardown acknowledgement.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: One-shot commit receipt after final unique acknowledgement.
        """

        if type(message) is not PackedRequestTeardownAck:
            raise TypeError("message must be PackedRequestTeardownAck")
        with self._lock:
            if self._state not in (
                PackedRequestTransactionState.TEARDOWN_WAITING,
                PackedRequestTransactionState.COMMIT_READY,
            ):
                raise PackedRequestTransactionError(
                    f"teardown acknowledgement is invalid in state {self._state.value}"
                )
            writers = self._allocation_snapshot.writer_manifest.writers
            if authenticated_writer_id not in writers:
                raise PackedRequestTransactionError(
                    "teardown writer is absent from the request manifest"
                )
            expected = self._teardown_requests[writers.index(authenticated_writer_id)]
            expected_ack = self._teardown_ack(expected)
            if self._state is PackedRequestTransactionState.COMMIT_READY:
                if (
                    authenticated_writer_id in self._teardown_acks
                    and message == expected_ack
                ):
                    return None
                self._quarantine_locked(
                    "conflicting teardown evidence after barrier completion"
                )
                raise PackedRequestTransactionError(
                    "conflicting teardown evidence after barrier completion"
                )
            if message.writer_id != authenticated_writer_id:
                self._quarantine_locked(
                    "teardown writer differs from authenticated peer"
                )
                raise PackedRequestTransactionError(
                    "teardown writer differs from authenticated peer"
                )
            if message != expected_ack:
                self._quarantine_locked(
                    "teardown acknowledgement differs from its exact request"
                )
                raise PackedRequestTransactionError(
                    "teardown acknowledgement differs from its exact request"
                )
            if authenticated_writer_id in self._teardown_acks:
                return None
            if authenticated_writer_id == self._auxiliary_plan.canonical_writer_id:
                outcome = self._auxiliary_outcome
                if outcome is None:
                    self._quarantine_locked(
                        "canonical teardown lacks an auxiliary outcome"
                    )
                    raise PackedRequestTransactionError(
                        "canonical teardown lacks an auxiliary outcome"
                    )
                try:
                    self._auxiliary_allocation_authority.record_teardown_completion(
                        self._auxiliary_allocation_lease,
                        self._lifecycle_authority,
                        metadata_slot_generation=(
                            self._auxiliary_plan.metadata_slot_generation
                        ),
                        native_dram_handle_generation=(
                            self._terminal_auxiliary_handle_generation(outcome)
                        ),
                        descriptor_digest=outcome.descriptor_digest,
                        evidence_digest=outcome.evidence_digest,
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    self._quarantine_locked(
                        "canonical auxiliary teardown transition failed"
                    )
                    raise PackedRequestTransactionError(
                        "canonical auxiliary teardown transition failed"
                    ) from error
                self._auxiliary_teardown_recorded = True
            self._teardown_acks.add(authenticated_writer_id)
            if self._teardown_acks != set(writers):
                return None

            self._validate_auxiliary_teardown_snapshot_locked()
            snapshot = self._allocation_snapshot
            try:
                self._allocation_authority.record_teardown_completion(
                    self._allocation_lease,
                    self._lifecycle_authority,
                    request_generation=snapshot.request_generation,
                    writer_manifest_digest=snapshot.writer_manifest.digest,
                    allocation_digest=snapshot.allocation_digest,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                self._quarantine_locked("request teardown transition failed")
                raise PackedRequestTransactionError(
                    "request teardown transition failed"
                ) from error
            receipt = PackedRequestCommitReceipt(
                self._transaction_nonce,
                object(),
                _COMMIT_CONSTRUCTION_SEAL,
            )
            self._commit_receipt = receipt
            self._state = PackedRequestTransactionState.COMMIT_READY
            return receipt

    def commit_on_scheduler_thread(
        self,
        receipt: PackedRequestCommitReceipt,
    ) -> object:
        """Consume teardown authority and return the exact request owner.

        :param receipt: Exact one-shot commit-ready receipt.
        :returns: Exact retained request whose allocation lease must be cleared.
        """

        self._require_scheduler_thread()
        with self._lock:
            if type(receipt) is not PackedRequestCommitReceipt:
                raise TypeError("receipt must be PackedRequestCommitReceipt")
            if (
                receipt._transaction_nonce is not self._transaction_nonce
                or self._commit_receipt is not receipt
            ):
                raise PackedRequestTransactionError(
                    "commit receipt belongs to another request transaction"
                )
            if self._state is not PackedRequestTransactionState.COMMIT_READY:
                raise PackedRequestTransactionError(
                    f"request commit is invalid in state {self._state.value}"
                )
            if not self._auxiliary_committed_to_request:
                self._validate_auxiliary_teardown_snapshot_locked()
                self._auxiliary_allocation_authority.commit_to_request_after_teardown(
                    self._auxiliary_allocation_lease,
                    self._lifecycle_authority,
                )
                self._auxiliary_committed_to_request = True
            else:
                self._validate_auxiliary_request_commit_snapshot_locked()
            for chunk in self._chunks:
                if chunk.retired:
                    continue
                self._outcome_coordinator.retire_chunk(chunk.plan.key)
                chunk.retired = True
            if not self._allocation_committed:
                self._allocation_authority.commit_to_request_after_teardown(
                    self._allocation_lease,
                    self._lifecycle_authority,
                )
                self._allocation_committed = True
            if not self._allocation_retired:
                self._allocation_authority.retire_terminal(self._allocation_lease)
                self._allocation_retired = True
            self._commit_receipt = None
            self._state = PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
            return self._request_owner

    def begin_dflash_boundary_adoption_on_scheduler_thread(
        self,
    ) -> PackedDFlashBoundaryDecodeAdoption:
        """Issue the authenticated DFlash row generation for one D2D adoption.

        This operation does not release the destination row. The scheduler must
        enqueue a request-owned device copy and hand its completion authority to
        the terminal owner before metadata consumption can release the row.

        :returns: Exact committed row and authenticated scalar metadata.
        """

        self._require_scheduler_thread()
        with self._lock:
            if (
                self._state
                is not PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
            ):
                raise PackedRequestTransactionError(
                    "DFlash boundary adoption is invalid in state "
                    f"{self._state.value}"
                )
            if self._dflash_boundary_adoption is not None:
                raise PackedRequestTransactionError(
                    "DFlash boundary adoption authority was already issued"
                )
            outcome = self._auxiliary_outcome
            if type(outcome) is not PackedDFlashBoundaryOutcome:
                raise PackedRequestTransactionError(
                    "terminal request has no authenticated DFlash boundary outcome"
                )
            slot = self._validate_auxiliary_request_commit_snapshot_locked()
            adoption = PackedDFlashBoundaryDecodeAdoption(
                metadata=outcome.metadata,
                lease=slot,
                outcome_digest=outcome.outcome_digest,
            )
            self._dflash_boundary_adoption = adoption
            return adoption

    def complete_auxiliary_consumption_on_scheduler_thread(
        self,
        consumer_authority: object,
        dflash_adoption: PackedDFlashBoundaryDecodeAdoption | None = None,
    ) -> None:
        """Release the exact metadata row after scheduler copy completes.

        :param consumer_authority: Exact configured scheduler metadata consumer.
        :param dflash_adoption: Exact issued DFlash adoption after its device
            copy completion became terminal.
        """

        self._require_scheduler_thread()
        with self._lock:
            if (
                self._state
                is not PackedRequestTransactionState.DESTINATION_CONSUMPTION_WAITING
            ):
                raise PackedRequestTransactionError(
                    "metadata consumption completion is invalid in state "
                    f"{self._state.value}"
                )
            if type(self._auxiliary_outcome) is PackedDFlashBoundaryOutcome:
                if dflash_adoption is not self._dflash_boundary_adoption:
                    self._quarantine_locked(
                        "DFlash metadata consumption lacks exact adoption authority"
                    )
                    raise PackedRequestTransactionError(
                        "DFlash metadata consumption lacks exact adoption authority"
                    )
            elif dflash_adoption is not None:
                self._quarantine_locked(
                    "legacy metadata consumption received DFlash authority"
                )
                raise PackedRequestTransactionError(
                    "legacy metadata consumption received DFlash authority"
                )
            self._validate_auxiliary_request_commit_snapshot_locked()
            try:
                if not self._auxiliary_released:
                    self._auxiliary_allocation_authority.release_after_consumption(
                        self._auxiliary_allocation_lease,
                        consumer_authority,
                    )
                    self._auxiliary_released = True
                if not self._auxiliary_retired:
                    self._auxiliary_allocation_authority.retire_terminal(
                        self._auxiliary_allocation_lease
                    )
                    self._auxiliary_retired = True
            except (RuntimeError, TypeError, ValueError) as error:
                self._quarantine_locked("metadata consumption release failed")
                raise PackedRequestTransactionError(
                    "metadata consumption release failed"
                ) from error
            self._dflash_boundary_adoption = None
            self._state = PackedRequestTransactionState.COMMITTED

    def quarantine(self, reason: str) -> None:
        """Quarantine the complete allocation when ownership becomes unprovable.

        :param reason: Stable failure reason.
        """

        if len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        with self._lock:
            self._quarantine_locked(reason)

    def snapshot(self) -> PackedRequestTransactionSnapshot:
        """Return an immutable request-level diagnostic snapshot.

        :returns: Current aggregate transaction state.
        """

        with self._lock:
            return PackedRequestTransactionSnapshot(
                key=self._request_key,
                state=self._state,
                chunk_states=tuple(
                    self._protocol.snapshot(chunk.plan.key).state
                    for chunk in self._chunks
                    if not chunk.retired
                ),
                scatter_started=tuple(
                    chunk.plan.key.chunk_id
                    for chunk in self._chunks
                    if chunk.scatter is not None
                ),
                scatter_terminal=tuple(
                    chunk.plan.key.chunk_id
                    for chunk in self._chunks
                    if chunk.scatter_terminal
                ),
                teardown_acks=tuple(sorted(self._teardown_acks)),
                auxiliary_outcome=self._auxiliary_outcome,
                auxiliary_teardown_acknowledged=(
                    self._auxiliary_plan.canonical_writer_id in self._teardown_acks
                ),
            )

    @staticmethod
    def _terminal_auxiliary_handle_generation(
        outcome: PackedTerminalAuxiliaryOutcome,
    ) -> int:
        """Return the native handle generation from either auxiliary schema.

        :param outcome: Authenticated legacy or DFlash terminal outcome.
        :returns: Exact native handle generation.
        """

        if type(outcome) is PackedAuxiliaryOutcome:
            return outcome.native_dram_handle_generation
        if type(outcome) is PackedDFlashBoundaryOutcome:
            return outcome.native_handle_generation
        raise TypeError("unsupported terminal auxiliary outcome")

    @staticmethod
    def _teardown_ack(request: PackedRequestTeardown) -> PackedRequestTeardownAck:
        """Project one exact teardown request to its acknowledgement shape.

        :param request: Decoder-authored teardown request.
        :returns: Exact authenticated acknowledgement shape.
        """

        return PackedRequestTeardownAck(
            key=request.key,
            writer_id=request.writer_id,
            request_slot_generation=request.request_slot_generation,
            writer_manifest_digest=request.writer_manifest_digest,
            allocation_digest=request.allocation_digest,
            teardown_generation=request.teardown_generation,
            auxiliary_handle_generation=request.auxiliary_handle_generation,
        )

    def _validate_auxiliary_teardown_snapshot_locked(
        self,
    ) -> PackedAuxiliaryAllocationLeaseSnapshot:
        """Require exact retained row and canonical source-handle retirement.

        :returns: Exact teardown-complete metadata lease snapshot.
        """

        outcome = self._auxiliary_outcome
        failure_reason: str | None = None
        if outcome is None:
            failure_reason = "auxiliary teardown has no terminal outcome"
        elif not self._auxiliary_teardown_recorded:
            failure_reason = "auxiliary teardown was not recorded"
        try:
            snapshot = self._auxiliary_allocation_authority.snapshot(
                self._auxiliary_allocation_lease
            )
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Packed auxiliary teardown snapshot failed:\n%s",
                traceback.format_exc(),
            )
            failure_reason = "auxiliary teardown allocation snapshot is unavailable"
            self._quarantine_locked(failure_reason)
            raise PackedRequestTransactionError(failure_reason) from error
        plan = self._auxiliary_plan
        if snapshot.metadata_buffer_index != plan.metadata_buffer_index:
            failure_reason = "auxiliary teardown metadata index is stale"
        elif snapshot.metadata_slot_generation != plan.metadata_slot_generation:
            failure_reason = "auxiliary teardown metadata generation is stale"
        elif snapshot.destination_segments != plan.destination_segments:
            failure_reason = "auxiliary teardown metadata segments are stale"
        elif snapshot.state is not PackedAuxiliaryAllocationState.TEARDOWN_COMPLETED:
            failure_reason = "auxiliary allocation has not reached teardown completion"
        elif outcome is not None and (
            snapshot.native_dram_handle_generation
            != self._terminal_auxiliary_handle_generation(outcome)
            or snapshot.descriptor_digest != outcome.descriptor_digest
            or snapshot.evidence_digest != outcome.evidence_digest
        ):
            failure_reason = "auxiliary teardown terminal evidence is stale"
        if failure_reason is not None:
            self._quarantine_locked(failure_reason)
            raise PackedRequestTransactionError(failure_reason)
        return snapshot

    def _validate_auxiliary_request_commit_snapshot_locked(
        self,
    ) -> PackedAuxiliaryAllocationLeaseSnapshot:
        """Require the exact metadata row retained for scheduler consumption.

        :returns: Exact request-committed metadata lease snapshot.
        """

        try:
            snapshot = self._auxiliary_allocation_authority.snapshot(
                self._auxiliary_allocation_lease
            )
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Packed auxiliary consumption snapshot failed:\n%s",
                traceback.format_exc(),
            )
            failure_reason = "request consumption metadata snapshot is unavailable"
            self._quarantine_locked(failure_reason)
            raise PackedRequestTransactionError(failure_reason) from error
        plan = self._auxiliary_plan
        if (
            snapshot.metadata_buffer_index != plan.metadata_buffer_index
            or snapshot.metadata_slot_generation != plan.metadata_slot_generation
            or snapshot.destination_segments != plan.destination_segments
            or snapshot.state is not PackedAuxiliaryAllocationState.COMMITTED_TO_REQUEST
        ):
            failure_reason = (
                "request consumption metadata lease differs from its exact plan"
            )
            self._quarantine_locked(failure_reason)
            raise PackedRequestTransactionError(failure_reason)
        return snapshot

    def _register_chunks(self) -> None:
        """Register every validated chunk before publication is possible."""

        protocol_registered: list[PackedChunkKey] = []
        coordinator_registered: list[PackedChunkKey] = []
        try:
            for chunk in self._chunks:
                plan = chunk.plan
                policies = plan.policy_map
                self._protocol.register_chunk(
                    plan.key,
                    plan.spec,
                    plan.destination_registry,
                    {
                        writer_id: policy.digest
                        for writer_id, policy in policies.items()
                    },
                )
                protocol_registered.append(plan.key)
                self._outcome_coordinator.register_chunk(plan.key, policies)
                coordinator_registered.append(plan.key)
        except (RuntimeError, TypeError, ValueError) as error:
            try:
                for key in reversed(coordinator_registered):
                    self._outcome_coordinator.cancel_unpublished_chunk(key)
                coordinator_keys = set(coordinator_registered)
                for key in reversed(protocol_registered):
                    if key not in coordinator_keys:
                        self._protocol.cancel_unpublished_chunk(key)
            except (RuntimeError, TypeError, ValueError) as cleanup_error:
                raise PackedRequestTransactionError(
                    "packed request registration cleanup failed"
                ) from cleanup_error
            raise PackedRequestTransactionError(
                "packed request chunk registration failed"
            ) from error

    @staticmethod
    def _canonical_auxiliary_writer(
        snapshot: DecodeAllocationLeaseSnapshot,
    ) -> StagingWriterId:
        """Resolve the first writer in the destination-local cohort.

        :param snapshot: Exact allocation-derived writer manifest.
        :returns: Canonical request auxiliary writer.
        """

        writers = snapshot.writer_manifest.writers
        if len(writers) == 0:
            raise ValueError("packed auxiliary plan requires a source writer")
        return writers[0]

    @staticmethod
    def _validate_plans(
        request_key: PackedRequestKey,
        snapshot: DecodeAllocationLeaseSnapshot,
        plans: tuple[PackedRequestChunkPlan, ...],
    ) -> None:
        """Validate complete topology, pages, and fixed destination coverage.

        :param request_key: Exact allocation-derived packed request identity.
        :param snapshot: Immutable allocation receipt.
        :param plans: Candidate decoder-authored chunks.
        """

        if len(plans) == 0:
            raise ValueError("packed request must contain at least one chunk")
        expected_chunk_ids = tuple(range(len(plans)))
        actual_chunk_ids = tuple(plan.key.chunk_id for plan in plans)
        if actual_chunk_ids != expected_chunk_ids:
            raise ValueError("packed request chunk IDs must be contiguous from zero")

        receipts = {receipt.component: receipt for receipt in snapshot.components}
        mamba = receipts[DecodeAllocationComponent.MAMBA]
        if not mamba.zero_work:
            raise ValueError("initial Gemma packed path requires zero-work Mamba")
        full = receipts[DecodeAllocationComponent.FULL]
        if full.zero_work:
            raise ValueError("packed request FULL component must not be empty")
        swa = receipts[DecodeAllocationComponent.SWA]
        coverage: dict[DecodeAllocationComponent, list[tuple[int, int]]] = {
            DecodeAllocationComponent.FULL: [],
            DecodeAllocationComponent.SWA: [],
        }

        for plan_index, plan in enumerate(plans):
            if PackedRequestKey.from_chunk_key(plan.key) != request_key:
                raise ValueError(
                    "packed chunk room or generation differs from allocation"
                )
            if plan.key.chunk_id != plan.spec.chunk_id:
                raise ValueError("packed chunk key/spec identity differs")
            expected_last = plan_index == len(plans) - 1
            if plan.spec.is_last is not expected_last:
                raise ValueError("packed request must contain one exact final chunk")
            if plan.spec.writers != snapshot.writer_manifest.writers:
                raise ValueError("packed chunk writers differ from allocation manifest")
            topology = plan.spec.topology
            manifest = snapshot.writer_manifest
            if (
                topology.source_tp_size != manifest.source_tp_size
                or topology.destination_tp_size != manifest.destination_tp_size
                or topology.destination_tp_rank != manifest.destination_tp_rank
            ):
                raise ValueError(
                    "packed chunk topology differs from allocation manifest"
                )
            policies = plan.policy_map
            if tuple(policies) != manifest.writers:
                raise ValueError(
                    "packed visibility policies must follow exact writer order"
                )
            PackedDecodeRequestTransaction._validate_registry(
                plan.destination_registry,
                full,
                swa,
            )

            seen_components: set[DecodeAllocationComponent] = set()
            for span in plan.spec.spans:
                component = PackedDecodeRequestTransaction._allocation_component(
                    span.component_id
                )
                if component in seen_components:
                    raise ValueError("packed chunk duplicates an allocation component")
                seen_components.add(component)
                receipt = receipts[component]
                if receipt.zero_work:
                    raise ValueError(
                        "packed chunk includes a zero-work allocation component"
                    )
                if span.source_index_offset != 0:
                    raise ValueError(
                        "packed source chunk spans must start at local offset zero"
                    )
                if span.logical_token_count != span.physical_token_count:
                    raise ValueError(
                        "packed request spans require complete physical token rows"
                    )
                if span.physical_token_count % receipt.page_size != 0:
                    raise ValueError("packed span token count is not page aligned")
                page_count = span.physical_token_count // receipt.page_size
                coverage[component].append(
                    (
                        span.destination_index_offset,
                        span.destination_index_offset + page_count,
                    )
                )
            if DecodeAllocationComponent.FULL not in seen_components:
                raise ValueError("every packed chunk must contain FULL KV")
            swa_present = DecodeAllocationComponent.SWA in seen_components
            if not expected_last and swa_present:
                raise ValueError("intermediate packed chunks must not contain SWA")
            if expected_last and swa.zero_work and swa_present:
                raise ValueError("final packed chunk contains zero-work SWA")
            if expected_last and not swa.zero_work and not swa_present:
                raise ValueError("final packed chunk omits nonzero SWA")

        PackedDecodeRequestTransaction._validate_coverage(
            full,
            coverage[full.component],
        )
        PackedDecodeRequestTransaction._validate_coverage(
            swa,
            coverage[swa.component],
        )

    @staticmethod
    def _validate_registry(
        registry: StagingComponentBufferRegistry,
        full: DecodeAllocationComponentReceipt,
        swa: DecodeAllocationComponentReceipt,
    ) -> None:
        """Validate registry arrays against exact ordered physical receipts.

        :param registry: Candidate complete destination registry.
        :param full: FULL allocation receipt.
        :param swa: SWA allocation receipt.
        """

        expected = {DecodeAllocationComponent.FULL: full}
        if not swa.zero_work:
            expected[DecodeAllocationComponent.SWA] = swa
        actual_components = tuple(
            PackedDecodeRequestTransaction._allocation_component(component.component_id)
            for component in registry.components
        )
        if set(actual_components) != set(expected):
            raise ValueError(
                "destination registry components differ from allocation receipts"
            )
        if len(actual_components) != len(expected):
            raise ValueError(
                "destination registry allocation components are duplicated"
            )
        for component in registry.components:
            allocation_component = PackedDecodeRequestTransaction._allocation_component(
                component.component_id
            )
            receipt = expected[allocation_component]
            if component.page_size != receipt.page_size:
                raise ValueError(
                    "destination registry page size differs from allocation receipt"
                )
            pages = tuple(int(value) for value in component.page_array.tolist())
            if pages != receipt.physical_pages:
                raise ValueError(
                    "destination registry pages differ from ordered allocation receipt"
                )

    @staticmethod
    def _validate_coverage(
        receipt: DecodeAllocationComponentReceipt,
        intervals: list[tuple[int, int]],
    ) -> None:
        """Require exact gap-free non-overlapping destination page coverage.

        :param receipt: Component allocation receipt.
        :param intervals: Candidate request chunk page intervals.
        """

        if receipt.zero_work:
            if len(intervals) != 0:
                raise ValueError("zero-work component has destination coverage")
            return
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                if start < cursor:
                    raise ValueError("packed destination page coverage overlaps")
                raise ValueError("packed destination page coverage contains a gap")
            if end <= start:
                raise ValueError("packed destination page interval is empty")
            cursor = end
        if cursor != len(receipt.physical_pages):
            raise ValueError("packed destination page coverage differs from allocation")

    @staticmethod
    def _allocation_component(
        component_id: StagingComponentId,
    ) -> DecodeAllocationComponent:
        """Map one staging component to its allocation phase.

        :param component_id: Exact staging component identity.
        :returns: FULL or SWA allocation component.
        """

        if component_id == MAIN_KV_COMPONENT:
            return DecodeAllocationComponent.FULL
        if (
            component_id.state_index is not None
            and component_id.state_type is StateType.SWA
        ):
            return DecodeAllocationComponent.SWA
        raise ValueError(
            f"packed request contains unsupported component {component_id}"
        )

    def _advance_writer_completion_locked(self) -> None:
        """Advance only after every writer for every chunk is terminal."""

        failed_reason: str | None = None
        all_done = (
            self._auxiliary_outcome is not None and self._auxiliary_submission_recorded
        )
        for chunk in self._chunks:
            snapshot = self._protocol.snapshot(chunk.plan.key)
            if snapshot.state in (
                PackedProtocolState.FAILED_QUARANTINED,
                PackedProtocolState.FAILED_RELEASED,
            ):
                failed_reason = snapshot.failure_reason or "packed chunk failed"
                break
            outcomes = snapshot.writer_outcomes
            if len(outcomes) != len(
                self._allocation_snapshot.writer_manifest.writers
            ) or any(
                outcome.status is not PackedWriterOutcomeStatus.DONE
                for outcome in outcomes
            ):
                all_done = False
        if failed_reason is not None:
            self._quarantine_locked(failed_reason)
            return
        if not all_done or self._writer_completion_recorded:
            return
        if not self._submission_recorded:
            raise PackedRequestTransactionError(
                "all-writer completion has no recorded submission"
            )
        self._allocation_authority.record_writer_completion(
            self._allocation_lease,
            self._lifecycle_authority,
        )
        self._writer_completion_recorded = True
        self._state = PackedRequestTransactionState.WRITERS_COMPLETED
        self._advance_scatter_completion_locked()

    def _advance_scatter_completion_locked(self) -> None:
        """Advance only after every expected chunk scatter is terminal."""

        if not self._writer_completion_recorded:
            return
        if not all(chunk.scatter_terminal for chunk in self._chunks):
            return
        if self._state is PackedRequestTransactionState.SCATTER_COMPLETED:
            return
        self._allocation_authority.record_scatter_completion(
            self._allocation_lease,
            self._lifecycle_authority,
        )
        self._state = PackedRequestTransactionState.SCATTER_COMPLETED

    def _quarantine_locked(self, reason: str) -> None:
        """Fail all live chunks and retain exact allocation pins.

        :param reason: Stable request failure reason.
        """

        if self._state is PackedRequestTransactionState.QUARANTINED:
            return
        if self._state in (
            PackedRequestTransactionState.CANCELLED,
            PackedRequestTransactionState.COMMITTED,
        ):
            raise PackedRequestTransactionError(
                f"quarantine is invalid in state {self._state.value}"
            )
        self._state = PackedRequestTransactionState.QUARANTINED

        if not self._auxiliary_released and not self._auxiliary_retired:
            try:
                self._auxiliary_allocation_authority.quarantine(
                    self._auxiliary_allocation_lease,
                    reason,
                )
            except Exception:
                logger.error(
                    "Packed auxiliary allocation quarantine failed for %s:\n%s",
                    reason,
                    traceback.format_exc(),
                )
        if not self._allocation_committed and not self._allocation_retired:
            try:
                self._allocation_authority.quarantine(
                    self._allocation_lease,
                    reason,
                )
            except Exception:
                logger.error(
                    "Packed decode allocation quarantine failed for %s:\n%s",
                    reason,
                    traceback.format_exc(),
                )
        for chunk in self._chunks:
            if chunk.retired:
                continue
            try:
                self._protocol.fail_chunk(chunk.plan.key, reason)
            except Exception:
                logger.error(
                    "Packed chunk quarantine failed for %s:\n%s",
                    chunk.plan.key,
                    traceback.format_exc(),
                )

    def _require_active_chunk_locked(
        self,
        key: PackedChunkKey,
    ) -> _PackedRequestChunk:
        """Resolve one exact request chunk without accepting stale replay.

        :param key: Candidate request-local chunk identity.
        :returns: Exact mutable chunk record.
        """

        chunk = self._chunks_by_key.get(key)
        if chunk is None:
            raise PackedRequestTransactionError(
                "packed chunk does not belong to this request generation"
            )
        if chunk.retired:
            raise PackedRequestTransactionError("packed chunk is already retired")
        return chunk

    def _validate_scatter_locked(
        self,
        scatter: PackedRequestScatter,
    ) -> _PackedRequestChunk:
        """Resolve exact transaction-owned scatter authority.

        :param scatter: Candidate scatter ownership.
        :returns: Exact mutable chunk record.
        """

        if type(scatter) is not PackedRequestScatter:
            raise TypeError("scatter must be PackedRequestScatter")
        if scatter._transaction_nonce is not self._transaction_nonce:
            raise PackedRequestTransactionError(
                "scatter belongs to another request transaction"
            )
        for chunk in self._chunks:
            if chunk.token is scatter._token and chunk.scatter is scatter:
                return chunk
        raise PackedRequestTransactionError("scatter is not live")

    def _require_scheduler_thread(self) -> None:
        """Require the exact scheduler thread captured at construction."""

        if threading.get_ident() != self._scheduler_thread_id:
            raise PackedRequestTransactionError(
                "request lifecycle mutation requires the scheduler thread"
            )
