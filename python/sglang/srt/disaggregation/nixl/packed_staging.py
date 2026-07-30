import dataclasses
import enum
import hashlib
import logging
import threading
import traceback
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
import triton
import triton.language as tl
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.packed_staging_protocol import (
    MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES,
    PACKED_REQUEST_GENERATION_BYTES,
    PACKED_VISIBILITY_POLICY_DIGEST_BYTES,
    PackedChunkKey,
    PackedDecodeProtocol,
    PackedLayoutSpec,
    PackedLease,
    PackedPrepare,
    PackedReady,
    PackedScatterWork,
    PackedTopology,
    PackedTransportPath,
    PackedWriterCompletionMechanism,
    PackedWriterOutcome,
    PackedWriterOutcomeStatus,
    PackedWriterVisibilityAction,
    PackedWriterVisibilityEvidence,
    _PackedWriterOutcomeTicket,
    _PackedWriterOutcomeTicketIssuer,
)
from sglang.srt.disaggregation.common.staging_layout import (
    DEFAULT_STAGING_ALIGNMENT_BYTES,
    StagingChunkLayout,
    StagingComponentGeometry,
    StagingComponentId,
    StagingComponentSpan,
    StagingWriterId,
    StagingWriterLayout,
)
from sglang.srt.disaggregation.common.staging_runtime import (
    StagingComponentBuffer,
    StagingComponentBufferRegistry,
    StagingEndpoint,
    StagingEndpointBufferBinding,
    bind_staging_endpoint_buffers,
)

logger = logging.getLogger(__name__)

MAIN_KV_COMPONENT = StagingComponentId(state_index=None, state_type=None)
_UINT64_LIMIT = 1 << 64
_CAPABILITY_GENERATION_BYTES = 16


class PackedMemoryAgent(Protocol):
    """NIXL memory-registration operations required by packed owners."""

    def register_memory(
        self,
        addresses: list[tuple[int, int, int, str]],
        memory_kind: str,
    ) -> object:
        """Register one or more memory regions.

        :param addresses: NIXL address descriptors.
        :param memory_kind: NIXL memory-kind name.
        :returns: Opaque registration handle.
        """

        ...

    def deregister_memory(self, registration: object) -> None:
        """Deregister one opaque registration handle.

        :param registration: Handle returned by :meth:`register_memory`.
        """

        ...


@dataclasses.dataclass(frozen=True)
class PackedPeerIdentity:
    """Authenticated NIXL agent process identity.

    :ivar agent_name: Transport-authenticated NIXL agent name.
    :ivar agent_generation: Bootstrap generation for that exact agent process.
    """

    agent_name: str
    agent_generation: bytes

    def __post_init__(self) -> None:
        """Own and validate immutable process identity."""

        if type(self.agent_generation) is not bytes:
            raise TypeError("agent_generation must be bytes")
        if type(self.agent_name) is not str or len(self.agent_name) == 0:
            raise ValueError("agent_name must be a non-empty string")
        if len(self.agent_generation) != _CAPABILITY_GENERATION_BYTES:
            raise ValueError(
                "agent_generation must contain "
                f"{_CAPABILITY_GENERATION_BYTES} bytes, got "
                f"{len(self.agent_generation)}"
            )


class PackedCudaDeviceAttribute(enum.IntEnum):
    """CUDA device attributes required to select destination visibility."""

    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118


class PackedGpuDirectWritesOrdering(enum.IntEnum):
    """CUDA GPUDirect RDMA writes-ordering attribute values."""

    NONE = 0
    OWNER = 100
    ALL_DEVICES = 200


class PackedGpuDirectFlushOptions(enum.IntFlag):
    """CUDA GPUDirect RDMA flush-options attribute bits."""

    NONE = 0
    HOST = 1
    MEMOPS = 2


class PackedGpuDirectFlushTarget(enum.IntEnum):
    """CUDA GPUDirect flush target used by packed destination staging."""

    CURRENT_CONTEXT = 0


class PackedGpuDirectFlushScope(enum.IntEnum):
    """CUDA GPUDirect flush scope used by packed destination staging."""

    OWNER = 100


class PackedDestinationVisibilityAction(enum.StrEnum):
    """Visibility operation proven complete before scatter submission."""

    CUDA_STREAM_DEPENDENCY = "cuda_stream_dependency"
    GPUDIRECT_HOST_FLUSH = "gpudirect_host_flush"
    OWNER_ORDERING = "owner_ordering"


@dataclasses.dataclass(frozen=True)
class PackedDestinationVisibilityPolicy:
    """Path-specific CUDA consumer-visibility policy.

    Multiple RDMA hardware paths are separate ordering domains. Callers must
    record the exact UCX lane or CUDA IPC path selected for each transfer.

    :ivar transport_path: Data path into destination memory.
    :ivar lane_identifier: Stable diagnostic identity of the selected lane.
    :ivar completion_mechanism: Exact source completion primitive.
    :ivar writes_ordering: CUDA GPUDirect writes-ordering attribute.
    :ivar flush_options: CUDA GPUDirect flush-options attribute bits.
    """

    transport_path: PackedTransportPath
    lane_identifier: str
    completion_mechanism: PackedWriterCompletionMechanism
    writes_ordering: PackedGpuDirectWritesOrdering
    flush_options: PackedGpuDirectFlushOptions

    def __post_init__(self) -> None:
        """Reject incomplete or unsafe CUDA visibility policies."""

        if type(self.transport_path) is not PackedTransportPath:
            raise TypeError("transport_path must be PackedTransportPath")
        if type(self.completion_mechanism) is not PackedWriterCompletionMechanism:
            raise TypeError(
                "completion_mechanism must be PackedWriterCompletionMechanism"
            )
        if type(self.writes_ordering) is not PackedGpuDirectWritesOrdering:
            raise TypeError("writes_ordering must be PackedGpuDirectWritesOrdering")
        if type(self.flush_options) is not PackedGpuDirectFlushOptions:
            raise TypeError("flush_options must be PackedGpuDirectFlushOptions")
        known_flush_options = (
            PackedGpuDirectFlushOptions.HOST | PackedGpuDirectFlushOptions.MEMOPS
        )
        unknown_flush_options = int(self.flush_options) & ~int(known_flush_options)
        if unknown_flush_options != 0:
            raise ValueError(
                "flush_options contains unsupported CUDA attribute bits: "
                f"{unknown_flush_options:#x}"
            )
        if type(self.lane_identifier) is not str or len(self.lane_identifier) == 0:
            raise ValueError("visibility lane_identifier must not be empty")
        if (
            len(self.lane_identifier.encode("utf-8"))
            > MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES
        ):
            raise ValueError(
                "visibility lane_identifier exceeds "
                f"{MAX_PACKED_VISIBILITY_LANE_IDENTIFIER_BYTES} UTF-8 bytes"
            )
        if self.transport_path is PackedTransportPath.CUDA_IPC:
            if (
                self.completion_mechanism
                is not PackedWriterCompletionMechanism.EXPORTED_CUDA_EVENT_RECORDED
            ):
                raise ValueError(
                    "CUDA IPC requires an exported CUDA event completion mechanism"
                )
            return
        if (
            self.completion_mechanism
            is not PackedWriterCompletionMechanism.NIXL_TRANSFER_HANDLE_TERMINAL
        ):
            raise ValueError(
                "NIC RDMA requires a terminal NIXL transfer-handle mechanism"
            )
        if self.writes_ordering >= PackedGpuDirectWritesOrdering.OWNER:
            return
        host_flush_available = (
            self.flush_options & PackedGpuDirectFlushOptions.HOST
        ) == PackedGpuDirectFlushOptions.HOST
        if not host_flush_available:
            raise ValueError(
                "unordered GPUDirect writes require the CUDA HOST flush option"
            )

    @property
    def required_action(self) -> PackedDestinationVisibilityAction:
        """Return the visibility action required for this exact path.

        :returns: Path-specific CUDA visibility action.
        """

        if self.transport_path is PackedTransportPath.CUDA_IPC:
            return PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY
        if self.writes_ordering >= PackedGpuDirectWritesOrdering.OWNER:
            return PackedDestinationVisibilityAction.OWNER_ORDERING
        return PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH

    @property
    def expected_writer_action(self) -> PackedWriterVisibilityAction:
        """Return the source-side action required for this transport path.

        :returns: Exact writer action accepted in terminal evidence.
        """

        if self.transport_path is PackedTransportPath.CUDA_IPC:
            return PackedWriterVisibilityAction.CUDA_EVENT_RECORDED
        return PackedWriterVisibilityAction.TRANSPORT_HANDLE_TERMINAL

    @property
    def digest(self) -> bytes:
        """Return a deterministic digest of the selected path and CUDA policy.

        :returns: SHA-256 route-policy identity.
        """

        fields = (
            b"sglang-packed-visibility-policy-v1",
            self.transport_path.value.encode("utf-8"),
            self.lane_identifier.encode("utf-8"),
            self.completion_mechanism.value.encode("utf-8"),
            int(self.writes_ordering).to_bytes(4, "big"),
            int(self.flush_options).to_bytes(4, "big"),
        )
        digest = hashlib.sha256()
        for field in fields:
            digest.update(len(field).to_bytes(4, "big"))
            digest.update(field)
        return digest.digest()

    def validate_evidence(self, evidence: PackedWriterVisibilityEvidence) -> None:
        """Authenticate source evidence against this decode-local policy.

        :param evidence: Untrusted writer evidence from a terminal outcome.
        """

        if type(evidence) is not PackedWriterVisibilityEvidence:
            raise TypeError(
                "visibility evidence must be PackedWriterVisibilityEvidence"
            )
        if evidence.policy_digest != self.digest:
            raise ValueError("visibility evidence policy digest differs from READY")
        if evidence.transport_path is not self.transport_path:
            raise ValueError("visibility evidence transport path differs from policy")
        if evidence.lane_identifier != self.lane_identifier:
            raise ValueError("visibility evidence lane differs from pinned policy")
        if evidence.completion_mechanism is not self.completion_mechanism:
            raise ValueError(
                "visibility evidence completion mechanism differs from policy"
            )
        if evidence.writer_action is not self.expected_writer_action:
            raise ValueError("visibility evidence writer action differs from policy")


_DESTINATION_VISIBILITY_ACTION_RECEIPT = object()


@dataclasses.dataclass(frozen=True)
class PackedDestinationVisibilityProof:
    """Per-writer evidence that the selected visibility policy was completed.

    :ivar writer_id: Exact writer whose transfer became visible.
    :ivar policy: Path and queried CUDA attribute policy.
    :ivar evidence: Authenticated terminal writer evidence.
    :ivar completed_action: Visibility action completed before construction.
    """

    writer_id: StagingWriterId
    policy: PackedDestinationVisibilityPolicy
    evidence: PackedWriterVisibilityEvidence
    completed_action: PackedDestinationVisibilityAction
    _action_receipt: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject evidence that does not satisfy its path policy."""

        if type(self.policy) is not PackedDestinationVisibilityPolicy:
            raise TypeError("policy must be PackedDestinationVisibilityPolicy")
        if self._action_receipt is not _DESTINATION_VISIBILITY_ACTION_RECEIPT:
            raise TypeError(
                "visibility proof requires a destination action executor receipt"
            )
        self.policy.validate_evidence(self.evidence)
        if type(self.completed_action) is not PackedDestinationVisibilityAction:
            raise TypeError(
                "completed_action must be PackedDestinationVisibilityAction"
            )
        if self.completed_action is not self.policy.required_action:
            raise ValueError(
                "completed visibility action does not satisfy the selected policy"
            )


class PackedDestinationVisibilityActionExecutor(Protocol):
    """Destination-owned CUDA visibility operations."""

    def establish_cuda_stream_dependency(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
    ) -> None:
        """Make CUDA IPC writes precede the destination scatter stream.

        :param writer_id: Authenticated writer used to locate its imported event.
        :param policy: Decode-local pinned path policy.
        :param evidence: Authenticated terminal writer evidence.
        """

        ...

    def flush_gpudirect_writes(
        self,
        writer_id: StagingWriterId,
        policy: PackedDestinationVisibilityPolicy,
        evidence: PackedWriterVisibilityEvidence,
        target: PackedGpuDirectFlushTarget,
        scope: PackedGpuDirectFlushScope,
    ) -> None:
        """Flush unordered GPUDirect RDMA writes for the destination context.

        :param writer_id: Authenticated writer owning the selected NIC lane.
        :param policy: Decode-local pinned path policy.
        :param evidence: Authenticated terminal writer evidence.
        :param target: CUDA flush target.
        :param scope: CUDA flush scope.
        """

        ...


def derive_destination_visibility_proof(
    *,
    writer_id: StagingWriterId,
    evidence: PackedWriterVisibilityEvidence,
    policy: PackedDestinationVisibilityPolicy,
    action_executor: PackedDestinationVisibilityActionExecutor,
) -> PackedDestinationVisibilityProof:
    """Complete destination-owned visibility and construct trusted proof.

    :param writer_id: Authenticated canonical writer identity.
    :param evidence: Bounded evidence from the writer's terminal outcome.
    :param policy: Decode-local route and queried CUDA attribute policy.
    :param action_executor: Trusted destination CUDA action implementation.
    :returns: Proof safe to pass to destination scatter.
    """

    policy.validate_evidence(evidence)
    action = policy.required_action
    if action is PackedDestinationVisibilityAction.CUDA_STREAM_DEPENDENCY:
        action_executor.establish_cuda_stream_dependency(
            writer_id,
            policy,
            evidence,
        )
    elif action is PackedDestinationVisibilityAction.GPUDIRECT_HOST_FLUSH:
        action_executor.flush_gpudirect_writes(
            writer_id,
            policy,
            evidence,
            PackedGpuDirectFlushTarget.CURRENT_CONTEXT,
            PackedGpuDirectFlushScope.OWNER,
        )
    return PackedDestinationVisibilityProof(
        writer_id=writer_id,
        policy=policy,
        evidence=evidence,
        completed_action=action,
        _action_receipt=_DESTINATION_VISIBILITY_ACTION_RECEIPT,
    )


class PackedDestinationVisibilityError(RuntimeError):
    """Destination visibility failed before DONE could enter consensus."""


@dataclasses.dataclass
class _PackedDestinationVisibilityAttempt:
    """One per-writer visibility action executing outside registry locks."""

    message: PackedWriterOutcome
    completed: threading.Event = dataclasses.field(default_factory=threading.Event)
    visibility_failed: bool = False


class PackedDestinationOutcomeCoordinator:
    """Order destination visibility before successful writer consensus."""

    _action_executor: PackedDestinationVisibilityActionExecutor
    _active_done_handlers: dict[PackedChunkKey, int]
    _attempts: dict[
        tuple[PackedChunkKey, StagingWriterId],
        _PackedDestinationVisibilityAttempt,
    ]
    _lock: threading.Lock
    _policies: dict[
        PackedChunkKey,
        dict[StagingWriterId, PackedDestinationVisibilityPolicy],
    ]
    _proofs: dict[
        tuple[PackedChunkKey, StagingWriterId],
        tuple[
            PackedWriterOutcome,
            PackedDestinationVisibilityProof,
            _PackedWriterOutcomeTicket,
        ],
    ]
    _protocol: PackedDecodeProtocol
    _retiring: set[PackedChunkKey]
    _ticket_issuer: _PackedWriterOutcomeTicketIssuer

    def __init__(
        self,
        protocol: PackedDecodeProtocol,
        action_executor: PackedDestinationVisibilityActionExecutor,
    ) -> None:
        """Bind one decode protocol to its trusted CUDA action executor.

        :param protocol: Arena-owned decode protocol.
        :param action_executor: Destination CUDA visibility implementation.
        """

        self._action_executor = action_executor
        self._active_done_handlers = {}
        self._attempts = {}
        self._lock = threading.Lock()
        self._policies = {}
        self._proofs = {}
        self._protocol = protocol
        self._retiring = set()
        self._ticket_issuer = protocol._claim_writer_outcome_ticket_issuer()

    def register_chunk(
        self,
        key: PackedChunkKey,
        policies: dict[StagingWriterId, PackedDestinationVisibilityPolicy],
    ) -> None:
        """Retain decode-local policy truth for one chunk's writer routes.

        :param key: Exact request and chunk identity.
        :param policies: Canonical writer-to-policy mapping.
        """

        if len(policies) == 0:
            raise ValueError("destination visibility policies must not be empty")
        owned: dict[StagingWriterId, PackedDestinationVisibilityPolicy] = {}
        for writer_id, policy in policies.items():
            if type(writer_id) is not StagingWriterId:
                raise TypeError("visibility policy key must be StagingWriterId")
            if type(policy) is not PackedDestinationVisibilityPolicy:
                raise TypeError(
                    "visibility policy must be PackedDestinationVisibilityPolicy"
                )
            owned[writer_id] = policy
        protocol_policy_digests = self._protocol._writer_visibility_policy_digests(key)
        coordinator_policy_digests = {
            writer_id: policy.digest for writer_id, policy in owned.items()
        }
        if coordinator_policy_digests != protocol_policy_digests:
            raise ValueError(
                "destination visibility policies differ from protocol registration"
            )
        with self._lock:
            if key in self._retiring:
                raise ValueError(f"destination visibility chunk is retiring: {key}")
            if key in self._policies:
                raise ValueError(
                    f"destination visibility chunk is already registered: {key}"
                )
            self._policies[key] = owned
            self._active_done_handlers[key] = 0

    def handle_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Validate visibility before allowing DONE to enter the protocol.

        ERROR outcomes enter the protocol directly. A destination action failure
        quarantines the chunk without recording that writer terminal, so the
        lease cannot be reused until the exact action later succeeds.

        :param message: Authenticated terminal writer outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether successful consensus newly became scatter-ready.
        """

        if message.status is PackedWriterOutcomeStatus.ERROR:
            return self._protocol.handle_writer_outcome(
                message,
                authenticated_writer_id,
            )
        handler_reserved = False
        with self._lock:
            if message.key in self._retiring:
                raise PackedDestinationVisibilityError(
                    "destination visibility chunk retirement is in progress"
                )
            if message.key in self._policies:
                active_handlers = self._active_done_handlers[message.key]
                self._active_done_handlers[message.key] = active_handlers + 1
                handler_reserved = True
        try:
            return self._handle_done_writer_outcome(
                message,
                authenticated_writer_id,
            )
        finally:
            if handler_reserved:
                with self._lock:
                    active_handlers = self._active_done_handlers.get(message.key)
                    if active_handlers is None or active_handlers <= 0:
                        raise RuntimeError(
                            "destination visibility handler reservation drifted"
                        )
                    self._active_done_handlers[message.key] = active_handlers - 1

    def _handle_done_writer_outcome(
        self,
        message: PackedWriterOutcome,
        authenticated_writer_id: StagingWriterId,
    ) -> bool:
        """Validate and commit one coordinator-admitted DONE outcome.

        :param message: Authenticated successful writer outcome.
        :param authenticated_writer_id: Writer bound to the transport peer.
        :returns: Whether successful consensus newly became scatter-ready.
        """

        admission_required = self._protocol.preflight_writer_outcome(
            message,
            authenticated_writer_id,
        )
        if not admission_required:
            return self._protocol.handle_writer_outcome(
                message,
                authenticated_writer_id,
            )

        cache_key = (message.key, authenticated_writer_id)
        attempt: _PackedDestinationVisibilityAttempt | None = None
        policy: PackedDestinationVisibilityPolicy | None = None
        failure_reason: str | None = None
        cached_outcome = False
        cached_ticket: _PackedWriterOutcomeTicket | None = None
        conflicting_attempt = False
        while (
            attempt is None
            and failure_reason is None
            and not cached_outcome
            and not conflicting_attempt
        ):
            wait_for: _PackedDestinationVisibilityAttempt | None = None
            with self._lock:
                policies = self._policies.get(message.key)
                if policies is None:
                    failure_reason = "destination visibility policy is not registered"
                else:
                    policy = policies.get(authenticated_writer_id)
                    if policy is None:
                        failure_reason = (
                            "destination visibility writer policy is missing"
                        )
                if failure_reason is not None:
                    continue
                existing = self._attempts.get(cache_key)
                if existing is not None:
                    if existing.message != message:
                        conflicting_attempt = True
                        continue
                    wait_for = existing
                else:
                    cached = self._proofs.get(cache_key)
                    if cached is not None:
                        if cached[0] != message:
                            conflicting_attempt = True
                            continue
                        cached_outcome = True
                        cached_ticket = cached[2]
                        continue
                    attempt = _PackedDestinationVisibilityAttempt(message=message)
                    self._attempts[cache_key] = attempt
            if wait_for is None:
                continue
            wait_for.completed.wait()
            if wait_for.visibility_failed:
                raise PackedDestinationVisibilityError(
                    "destination visibility action failed"
                )
            admission_required = self._protocol.preflight_writer_outcome(
                message,
                authenticated_writer_id,
            )
            if not admission_required:
                return self._protocol.handle_writer_outcome(
                    message,
                    authenticated_writer_id,
                )

        if cached_outcome:
            if cached_ticket is None:
                raise RuntimeError("cached destination proof lost its DONE ticket")
            return self._protocol.handle_writer_outcome(
                message,
                authenticated_writer_id,
                cached_ticket,
            )
        if failure_reason is not None:
            self._protocol.fail_chunk(message.key, failure_reason)
            raise PackedDestinationVisibilityError(failure_reason)
        if conflicting_attempt:
            return self._protocol.handle_writer_outcome(
                message,
                authenticated_writer_id,
            )
        if attempt is None or policy is None:
            raise RuntimeError("destination visibility attempt initialization drifted")

        try:
            visibility = message.visibility
            if visibility is None:
                raise RuntimeError("DONE outcome lost its required visibility evidence")
            proof = derive_destination_visibility_proof(
                writer_id=authenticated_writer_id,
                evidence=visibility,
                policy=policy,
                action_executor=self._action_executor,
            )
            ticket = self._ticket_issuer._issue(message, authenticated_writer_id)
        except Exception as error:
            logger.error(
                "Packed destination visibility failed:\n%s",
                traceback.format_exc(),
            )
            with self._lock:
                current = self._attempts.get(cache_key)
                if current is attempt:
                    attempt.visibility_failed = True
            try:
                self._protocol.fail_chunk(
                    message.key,
                    "destination visibility action failed",
                )
            except Exception as protocol_error:
                logger.error(
                    "Packed destination visibility quarantine failed:\n%s",
                    traceback.format_exc(),
                )
                raise PackedDestinationVisibilityError(
                    "destination visibility action and chunk quarantine failed"
                ) from protocol_error
            finally:
                with self._lock:
                    current = self._attempts.get(cache_key)
                    if current is attempt:
                        del self._attempts[cache_key]
                        attempt.completed.set()
            raise PackedDestinationVisibilityError(
                "destination visibility action failed"
            ) from error

        try:
            with self._lock:
                current = self._attempts.get(cache_key)
                if current is not attempt:
                    raise RuntimeError(
                        "destination visibility attempt ownership changed"
                    )
                self._proofs[cache_key] = (message, proof, ticket)
            return self._protocol.handle_writer_outcome(
                message,
                authenticated_writer_id,
                ticket,
            )
        finally:
            with self._lock:
                current = self._attempts.get(cache_key)
                if current is attempt:
                    del self._attempts[cache_key]
                    attempt.completed.set()

    def proofs(
        self,
        key: PackedChunkKey,
        writers: tuple[StagingWriterId, ...],
    ) -> tuple[PackedDestinationVisibilityProof, ...]:
        """Return exact canonical proofs after successful consensus.

        :param key: Exact request and chunk identity.
        :param writers: Canonical writer order.
        :returns: Destination proofs in canonical writer order.
        """

        with self._lock:
            try:
                return tuple(self._proofs[(key, writer_id)][1] for writer_id in writers)
            except KeyError as error:
                raise PackedDestinationVisibilityError(
                    "destination visibility proofs are incomplete"
                ) from error

    def retire_chunk(self, key: PackedChunkKey) -> None:
        """Retire terminal protocol state, policy, and proof metadata.

        :param key: Retired protocol chunk identity.
        """

        with self._lock:
            if key in self._retiring:
                raise PackedDestinationVisibilityError(
                    "destination visibility chunk retirement is already in progress"
                )
            if self._active_done_handlers.get(key, 0) != 0:
                raise PackedDestinationVisibilityError(
                    "destination visibility DONE handling is still in progress"
                )
            if any(attempt_key[0] == key for attempt_key in self._attempts):
                raise PackedDestinationVisibilityError(
                    "destination visibility action is still in progress"
                )
            self._retiring.add(key)
        protocol_retired = False
        try:
            self._protocol.retire_chunk(key)
            protocol_retired = True
        finally:
            if not protocol_retired:
                with self._lock:
                    self._retiring.remove(key)
        with self._lock:
            policies = self._policies.pop(key, None)
            self._active_done_handlers.pop(key, None)
            if policies is not None:
                for writer_id in policies:
                    self._proofs.pop((key, writer_id), None)
            self._retiring.remove(key)


def _align_up(value: int, alignment: int) -> int:
    """Round a positive byte count up to allocator alignment.

    :param value: Positive byte count.
    :param alignment: Positive byte alignment.
    :returns: Aligned byte count.
    """

    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


@dataclasses.dataclass(frozen=True)
class PackedDestinationRegistration:
    """Destination geometry advertised to one prefill writer.

    :ivar main_item_lens: Main-KV bytes per page in registration order.
    :ivar main_layer_ids: Main-KV global layer IDs in registration order.
    :ivar state_item_lens: State-component bytes per page.
    :ivar state_layer_ids: State-component global layer IDs.
    :ivar page_size: Tokens represented by one page index.
    """

    main_item_lens: tuple[int, ...]
    main_layer_ids: tuple[int, ...]
    state_item_lens: tuple[tuple[int, ...], ...]
    state_layer_ids: tuple[tuple[int, ...], ...]
    page_size: int

    def __post_init__(self) -> None:
        """Own immutable copies of all registration metadata."""

        object.__setattr__(self, "main_item_lens", tuple(self.main_item_lens))
        object.__setattr__(self, "main_layer_ids", tuple(self.main_layer_ids))
        object.__setattr__(
            self,
            "state_item_lens",
            tuple(tuple(item_lens) for item_lens in self.state_item_lens),
        )
        object.__setattr__(
            self,
            "state_layer_ids",
            tuple(tuple(layer_ids) for layer_ids in self.state_layer_ids),
        )
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if len(self.state_item_lens) != len(self.state_layer_ids):
            raise ValueError(
                "destination state item-length/layer-id component counts differ"
            )


@dataclasses.dataclass(frozen=True)
class PackedComponentPages:
    """Source and destination pages for one active packed component.

    :ivar component_id: Exact main-KV or SWA state identity.
    :ivar source_pages: Source request-local physical pages.
    :ivar destination_pages: Destination request-local physical pages.
    :ivar destination_index_offset: Offset into the decode room's page array.
    """

    component_id: StagingComponentId
    source_pages: npt.NDArray[np.int32]
    destination_pages: npt.NDArray[np.int32]
    destination_index_offset: int

    def __post_init__(self) -> None:
        """Own immutable page-array snapshots."""

        source_pages = _immutable_page_array(self.source_pages, "source")
        destination_pages = _immutable_page_array(
            self.destination_pages,
            "destination",
        )
        if len(source_pages) == 0:
            raise ValueError("packed component must contain at least one page")
        if len(source_pages) != len(destination_pages):
            raise ValueError(
                "source/destination packed page counts differ: "
                f"{len(source_pages)} and {len(destination_pages)}"
            )
        if self.destination_index_offset < 0:
            raise ValueError(
                "destination_index_offset must be non-negative, got "
                f"{self.destination_index_offset}"
            )
        object.__setattr__(self, "source_pages", source_pages)
        object.__setattr__(self, "destination_pages", destination_pages)


def _immutable_page_array(
    pages: npt.NDArray[np.int32],
    label: str,
) -> npt.NDArray[np.int32]:
    """Return one immutable contiguous int32 page snapshot.

    :param pages: Page array to copy.
    :param label: Reader-facing endpoint label.
    :returns: Immutable page snapshot.
    """

    if not isinstance(pages, np.ndarray):
        raise TypeError(f"{label} pages must be a NumPy array")
    if pages.dtype != np.dtype(np.int32):
        raise TypeError(f"{label} pages must have dtype int32, got {pages.dtype}")
    if pages.ndim != 1:
        raise TypeError(f"{label} pages must be one-dimensional")
    snapshot = np.array(pages, order="C", copy=True)
    if np.any(snapshot < 0):
        raise ValueError(f"{label} pages contain a negative index")
    snapshot.setflags(write=False)
    return snapshot


def _component_id(kv_args: KVArgs, state_index: int | None) -> StagingComponentId:
    """Resolve an exact local component identity.

    :param kv_args: Local registered KV metadata.
    :param state_index: State component index, or ``None`` for main KV.
    :returns: Exact component identity.
    """

    if state_index is None:
        return MAIN_KV_COMPONENT
    if state_index < 0 or state_index >= len(kv_args.state_types):
        raise ValueError(f"state_index is out of range: {state_index}")
    return StagingComponentId(
        state_index=state_index,
        state_type=kv_args.state_types[state_index],
    )


def local_component_geometry(
    kv_args: KVArgs,
    component_id: StagingComponentId,
) -> StagingComponentGeometry:
    """Return local registered geometry for one packable component.

    :param kv_args: Local registered KV metadata.
    :param component_id: Exact main-KV or state identity.
    :returns: Immutable registered component geometry.
    :raises ValueError: If the component is missing or not SWA-packable.
    """

    state_index = component_id.state_index
    if state_index is None:
        if component_id != MAIN_KV_COMPONENT:
            raise ValueError("invalid main-KV component identity")
        return StagingComponentGeometry(
            component_id=component_id,
            item_lens=tuple(kv_args.kv_item_lens),
            layer_ids=tuple(kv_args.kv_layer_ids),
            page_size=kv_args.page_size,
        )
    expected_component = _component_id(kv_args, state_index)
    if component_id != expected_component:
        raise ValueError(
            f"state component identity differs from local registration: {component_id}"
        )
    if component_id.state_type is not StateType.SWA:
        raise ValueError(
            f"packed staging supports only SWA state, got {component_id.state_type}"
        )
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=tuple(kv_args.state_item_lens[state_index]),
        layer_ids=tuple(kv_args.state_layer_ids[state_index]),
        page_size=kv_args.page_size,
    )


def destination_component_geometry(
    registration: PackedDestinationRegistration,
    state_types: tuple[StateType, ...],
    component_id: StagingComponentId,
) -> StagingComponentGeometry:
    """Return advertised destination geometry for one active component.

    :param registration: Destination registration metadata.
    :param state_types: Source-local state ordering expected at the destination.
    :param component_id: Exact main-KV or SWA state identity.
    :returns: Immutable destination geometry.
    """

    state_index = component_id.state_index
    if state_index is None:
        if component_id != MAIN_KV_COMPONENT:
            raise ValueError("invalid main-KV component identity")
        return StagingComponentGeometry(
            component_id=component_id,
            item_lens=registration.main_item_lens,
            layer_ids=registration.main_layer_ids,
            page_size=registration.page_size,
        )
    if state_index < 0 or state_index >= len(state_types):
        raise ValueError(f"state_index is out of range: {state_index}")
    if state_types[state_index] is not StateType.SWA:
        raise ValueError(
            f"packed staging supports only SWA state, got {state_types[state_index]}"
        )
    if component_id.state_type is not state_types[state_index]:
        raise ValueError("state component identity differs from source registration")
    if state_index >= len(registration.state_item_lens):
        raise ValueError(f"destination state component is missing: {state_index}")
    return StagingComponentGeometry(
        component_id=component_id,
        item_lens=registration.state_item_lens[state_index],
        layer_ids=registration.state_layer_ids[state_index],
        page_size=registration.page_size,
    )


def derive_source_geometry(
    destination: StagingComponentGeometry,
    source_tp_size: int,
    destination_tp_size: int,
) -> StagingComponentGeometry:
    """Derive the only valid non-replicated source TP geometry.

    :param destination: Decode-local component geometry.
    :param source_tp_size: Source attention tensor-parallel width.
    :param destination_tp_size: Destination attention tensor-parallel width.
    :returns: Expected source per-rank geometry.
    :raises ValueError: If aggregate bytes cannot be partitioned exactly.
    """

    if source_tp_size <= 0 or destination_tp_size <= 0:
        raise ValueError("tensor-parallel sizes must be positive")
    source_item_lens: list[int] = []
    for destination_item_len in destination.item_lens:
        aggregate_item_len = destination_item_len * destination_tp_size
        if aggregate_item_len % source_tp_size != 0:
            raise ValueError(
                "destination geometry cannot be exactly partitioned by source TP: "
                f"{aggregate_item_len} bytes across {source_tp_size} ranks"
            )
        source_item_lens.append(aggregate_item_len // source_tp_size)
    return StagingComponentGeometry(
        component_id=destination.component_id,
        item_lens=tuple(source_item_lens),
        layer_ids=destination.layer_ids,
        page_size=destination.page_size,
    )


def build_component_buffer_registry(
    kv_args: KVArgs,
    page_arrays: dict[StagingComponentId, npt.NDArray[np.int32]],
) -> StagingComponentBufferRegistry:
    """Bind local registered pointers to request-local component pages.

    :param kv_args: Local registered KV metadata.
    :param page_arrays: Exact active component page arrays.
    :returns: Immutable component buffer registry.
    """

    components: list[StagingComponentBuffer] = []
    for component_id, page_array in page_arrays.items():
        geometry = local_component_geometry(kv_args, component_id)
        state_index = component_id.state_index
        if state_index is None:
            tensor_ptrs = tuple(kv_args.kv_data_ptrs)
            data_lens = tuple(kv_args.kv_data_lens)
        else:
            tensor_ptrs = tuple(kv_args.state_data_ptrs[state_index])
            data_lens = tuple(kv_args.state_data_lens[state_index])
        components.append(
            StagingComponentBuffer(
                component_id=component_id,
                tensor_ptrs=tensor_ptrs,
                data_lens=data_lens,
                item_lens=geometry.item_lens,
                layer_ids=geometry.layer_ids,
                page_size=geometry.page_size,
                page_array=page_array,
            )
        )
    return StagingComponentBufferRegistry(tuple(components))


def active_destination_page_arrays(
    kv_args: KVArgs,
    kv_indices: npt.NDArray[np.int32],
    state_indices: list[npt.NDArray[np.int32] | None] | None,
) -> dict[StagingComponentId, npt.NDArray[np.int32]]:
    """Snapshot every destination component eligible for packed transfer.

    :param kv_args: Decode-local registered KV metadata.
    :param kv_indices: Complete room-local main-KV page array.
    :param state_indices: Complete room-local state page arrays.
    :returns: Page arrays keyed by exact component identity.
    """

    page_arrays: dict[StagingComponentId, npt.NDArray[np.int32]] = {
        MAIN_KV_COMPONENT: _immutable_page_array(kv_indices, "destination main-KV")
    }
    if state_indices is None:
        return page_arrays
    if len(state_indices) != len(kv_args.state_types):
        raise ValueError(
            "destination state index count differs from registered state types: "
            f"{len(state_indices)} and {len(kv_args.state_types)}"
        )
    for state_index, pages in enumerate(state_indices):
        if pages is None or len(pages) == 0:
            continue
        if kv_args.state_types[state_index] is not StateType.SWA:
            continue
        component_id = _component_id(kv_args, state_index)
        page_arrays[component_id] = _immutable_page_array(
            pages,
            f"destination state[{state_index}]",
        )
    return page_arrays


def _validate_component_shape(
    *,
    is_last: bool,
    component_ids: tuple[StagingComponentId, ...],
    required_final_components: frozenset[StagingComponentId],
) -> None:
    """Validate the supported intermediate and final component shapes.

    :param is_last: Whether the chunk completes the room.
    :param component_ids: Active components.
    :param required_final_components: Destination state components required at
        final completion.
    :raises ValueError: If the shape is unsupported or omits required state.
    """

    component_set = set(component_ids)
    if len(component_set) != len(component_ids):
        raise ValueError("packed component identities must be unique")
    for component_id in component_ids:
        if component_id == MAIN_KV_COMPONENT:
            continue
        if component_id.state_type is not StateType.SWA:
            raise ValueError(
                f"packed staging supports only SWA state, got {component_id}"
            )
    if not is_last:
        if component_set != {MAIN_KV_COMPONENT}:
            raise ValueError("intermediate packed chunks must be main-KV-only")
        return
    if MAIN_KV_COMPONENT not in component_set:
        raise ValueError("final packed chunk must contain its main-KV slice")
    if not required_final_components.issubset(component_set):
        missing = sorted(
            required_final_components - component_set,
            key=lambda component_id: (
                component_id.state_index if component_id.state_index is not None else -1
            ),
        )
        raise ValueError(f"final packed chunk omits required components: {missing}")
    allowed_components = set(required_final_components)
    allowed_components.add(MAIN_KV_COMPONENT)
    unexpected = component_set - allowed_components
    if len(unexpected) > 0:
        raise ValueError(
            f"final packed chunk contains unregistered components: {unexpected}"
        )


def _required_final_components(
    kv_args: KVArgs,
) -> frozenset[StagingComponentId]:
    """Derive non-empty registered SWA components from endpoint-local truth.

    :param kv_args: Endpoint-local registered KV metadata.
    :returns: Exact SWA component set required in the final packed chunk.
    """

    state_collections = (
        kv_args.state_data_ptrs,
        kv_args.state_data_lens,
        kv_args.state_item_lens,
        kv_args.state_layer_ids,
    )
    required: set[StagingComponentId] = set()
    for state_index, state_type in enumerate(kv_args.state_types):
        if state_type is not StateType.SWA:
            continue
        if any(state_index >= len(collection) for collection in state_collections):
            raise ValueError(
                f"SWA state registration metadata is missing at index {state_index}"
            )
        lengths = tuple(
            len(collection[state_index]) for collection in state_collections
        )
        if all(length == 0 for length in lengths):
            continue
        if len(set(lengths)) != 1 or lengths[0] == 0:
            raise ValueError(
                "SWA state registration metadata has inconsistent entry counts "
                f"at index {state_index}: {lengths}"
            )
        required.add(_component_id(kv_args, state_index))
    return frozenset(required)


def build_prefill_chunk(
    *,
    key: PackedChunkKey,
    is_last: bool,
    kv_args: KVArgs,
    destination_registration: PackedDestinationRegistration,
    components: tuple[PackedComponentPages, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
    writers: tuple[StagingWriterId, ...],
) -> tuple[PackedLayoutSpec, StagingEndpointBufferBinding]:
    """Build and bind one source-authored packed chunk.

    :param key: Request and chunk identity.
    :param is_last: Whether the chunk completes the room.
    :param kv_args: Source-local registered KV metadata.
    :param destination_registration: Decode geometry from bootstrap metadata.
    :param components: Active source and destination page projections.
    :param source_tp_size: Source attention tensor-parallel width.
    :param destination_tp_size: Destination attention tensor-parallel width.
    :param destination_tp_rank: Destination attention tensor-parallel rank.
    :param writers: Complete writer set connected to this destination.
    :returns: Canonical layout spec and immutable source binding.
    """

    component_ids = tuple(component.component_id for component in components)
    required_final_components = _required_final_components(kv_args)
    _validate_component_shape(
        is_last=is_last,
        component_ids=component_ids,
        required_final_components=required_final_components,
    )
    source_components = tuple(
        local_component_geometry(kv_args, component_id)
        for component_id in component_ids
    )
    state_types = tuple(kv_args.state_types)
    destination_components = tuple(
        destination_component_geometry(
            destination_registration,
            state_types,
            component_id,
        )
        for component_id in component_ids
    )
    spans = tuple(
        StagingComponentSpan(
            component_id=component.component_id,
            source_index_offset=0,
            destination_index_offset=component.destination_index_offset,
            logical_token_count=len(component.source_pages) * kv_args.page_size,
            physical_token_count=len(component.source_pages) * kv_args.page_size,
        )
        for component in components
    )
    spec = PackedLayoutSpec(
        chunk_id=key.chunk_id,
        is_last=is_last,
        spans=spans,
        source_components=source_components,
        destination_components=destination_components,
        writers=writers,
        topology=PackedTopology(
            source_tp_size=source_tp_size,
            destination_tp_size=destination_tp_size,
            destination_tp_rank=destination_tp_rank,
        ),
    )
    source_registry = build_component_buffer_registry(
        kv_args,
        {component.component_id: component.source_pages for component in components},
    )
    source_binding = bind_staging_endpoint_buffers(
        spec.build(),
        StagingEndpoint.SOURCE,
        source_registry,
    )
    return spec, source_binding


def build_decode_spec(
    *,
    chunk_id: int,
    is_last: bool,
    spans: tuple[StagingComponentSpan, ...],
    kv_args: KVArgs,
    expected_writers: tuple[StagingWriterId, ...],
    source_tp_size: int,
    destination_tp_size: int,
    destination_tp_rank: int,
) -> PackedLayoutSpec:
    """Build decode-local canonical truth before accepting PREPARE.

    :param chunk_id: Trusted request-local chunk identifier.
    :param is_last: Trusted final-chunk marker.
    :param spans: Trusted component spans derived from room metadata.
    :param kv_args: Decode-local registered KV metadata.
    :param expected_writers: Writers authenticated by bootstrap routing.
    :param source_tp_size: Expected source attention TP width.
    :param destination_tp_size: Decode attention TP width.
    :param destination_tp_rank: Decode attention TP rank.
    :returns: Trusted canonical layout input.
    """

    expected_topology = PackedTopology(
        source_tp_size=source_tp_size,
        destination_tp_size=destination_tp_size,
        destination_tp_rank=destination_tp_rank,
    )
    component_ids = tuple(span.component_id for span in spans)
    _validate_component_shape(
        is_last=is_last,
        component_ids=component_ids,
        required_final_components=_required_final_components(kv_args),
    )
    for span in spans:
        if span.source_index_offset != 0:
            raise ValueError("packed source spans must start at request-local offset 0")
        if (
            span.component_id != MAIN_KV_COMPONENT
            and span.destination_index_offset != 0
        ):
            raise ValueError("packed SWA destination spans must start at offset 0")
        if span.logical_token_count != span.physical_token_count:
            raise ValueError(
                "packed NIXL spans currently require complete physical token rows"
            )
    destination_components = tuple(
        local_component_geometry(kv_args, component_id)
        for component_id in component_ids
    )
    source_components = tuple(
        derive_source_geometry(
            destination,
            source_tp_size,
            destination_tp_size,
        )
        for destination in destination_components
    )
    return PackedLayoutSpec(
        chunk_id=chunk_id,
        is_last=is_last,
        spans=spans,
        source_components=source_components,
        destination_components=destination_components,
        writers=expected_writers,
        topology=expected_topology,
    )


class PackedIntervalLeaseAllocator:
    """Non-overlapping first-fit allocator for one registered GPU buffer."""

    _alignment_bytes: int
    _allocations: dict[int, tuple[int, int]]
    _base_address: int
    _free_intervals: list[tuple[int, int]]
    _lock: threading.Lock
    _next_lease_id: int
    _quarantined: set[int]
    _total_size: int

    def __init__(
        self,
        *,
        base_address: int,
        total_size: int,
        alignment_bytes: int = DEFAULT_STAGING_ALIGNMENT_BYTES,
    ) -> None:
        """Initialize one empty contiguous allocation arena.

        :param base_address: Registered GPU buffer base pointer.
        :param total_size: Registered buffer capacity.
        :param alignment_bytes: Allocation alignment.
        """

        if type(base_address) is not int or base_address <= 0:
            raise ValueError(f"base_address must be positive, got {base_address}")
        if type(total_size) is not int or total_size <= 0:
            raise ValueError(f"total_size must be positive, got {total_size}")
        if type(alignment_bytes) is not int or alignment_bytes <= 0:
            raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
        if base_address % alignment_bytes != 0:
            raise ValueError(
                "base_address must satisfy allocator alignment: "
                f"{base_address} % {alignment_bytes} != 0"
            )
        _checked_uint64_region(base_address, total_size, "packed allocator arena")
        self._alignment_bytes = alignment_bytes
        self._allocations = {}
        self._base_address = base_address
        self._free_intervals = [(0, total_size)]
        self._lock = threading.Lock()
        self._next_lease_id = 0
        self._quarantined = set()
        self._total_size = total_size

    @property
    def alignment_bytes(self) -> int:
        """Return immutable lease alignment.

        :returns: Alignment in bytes.
        """

        return self._alignment_bytes

    @property
    def base_address(self) -> int:
        """Return immutable registered arena base.

        :returns: Base GPU address.
        """

        return self._base_address

    @property
    def live_lease_count(self) -> int:
        """Return allocated lease count.

        :returns: Live and quarantined lease count.
        """

        with self._lock:
            return len(self._allocations)

    @property
    def total_size(self) -> int:
        """Return registered arena capacity.

        :returns: Total bytes.
        """

        return self._total_size

    def allocate(self, length_bytes: int) -> PackedLease:
        """Allocate one non-overlapping aligned interval without waiting.

        :param length_bytes: Minimum required bytes.
        :returns: Contiguous registered lease.
        :raises MemoryError: If no sufficiently large interval is free.
        """

        aligned_length = _align_up(length_bytes, self._alignment_bytes)
        with self._lock:
            selected_index: int | None = None
            selected_offset = 0
            selected_length = 0
            for interval_index, (offset, free_length) in enumerate(
                self._free_intervals
            ):
                if free_length < aligned_length:
                    continue
                selected_index = interval_index
                selected_offset = offset
                selected_length = free_length
                break
            if selected_index is None:
                raise MemoryError(
                    f"packed staging pool cannot allocate {aligned_length} bytes"
                )
            remaining = selected_length - aligned_length
            if remaining == 0:
                self._free_intervals.pop(selected_index)
            else:
                self._free_intervals[selected_index] = (
                    selected_offset + aligned_length,
                    remaining,
                )
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            self._allocations[lease_id] = (selected_offset, aligned_length)
            return PackedLease(
                lease_id=lease_id,
                base_address=self._base_address + selected_offset,
                length_bytes=aligned_length,
            )

    def quarantine(self, lease: PackedLease, reason: str) -> None:
        """Mark one live lease as failed without making it allocatable.

        :param lease: Failed live lease.
        :param reason: First protocol failure reason.
        """

        if len(reason) == 0:
            raise ValueError("quarantine reason must not be empty")
        with self._lock:
            self._validate_live_lease(lease)
            self._quarantined.add(lease.lease_id)

    def release(self, lease: PackedLease) -> None:
        """Release and coalesce one terminally quiescent lease.

        :param lease: Live lease safe for immediate reuse.
        """

        with self._lock:
            offset, length = self._validate_live_lease(lease)
            candidate_intervals = sorted((*self._free_intervals, (offset, length)))
            merged: list[tuple[int, int]] = []
            for free_offset, free_length in candidate_intervals:
                if len(merged) == 0:
                    merged.append((free_offset, free_length))
                    continue
                previous_offset, previous_length = merged[-1]
                previous_end = previous_offset + previous_length
                if previous_end == free_offset:
                    merged[-1] = (
                        previous_offset,
                        previous_length + free_length,
                    )
                    continue
                if previous_end > free_offset:
                    raise RuntimeError("packed allocator free intervals overlap")
                merged.append((free_offset, free_length))
            del self._allocations[lease.lease_id]
            self._quarantined.discard(lease.lease_id)
            self._free_intervals = merged

    def _validate_live_lease(self, lease: PackedLease) -> tuple[int, int]:
        """Validate lease identity without mutating allocator ownership.

        :param lease: Lease expected to be live.
        :returns: Internal offset and aligned length.
        :raises ValueError: If identity or geometry differs.
        """

        allocation = self._allocations.get(lease.lease_id)
        if allocation is None:
            raise ValueError(f"packed lease is not live: {lease.lease_id}")
        offset, length = allocation
        if lease.base_address != self._base_address + offset:
            raise ValueError("packed lease base address differs from allocation")
        if lease.length_bytes != length:
            raise ValueError("packed lease length differs from allocation")
        return allocation


def writer_layout_for(
    layout: StagingChunkLayout,
    writer_id: StagingWriterId,
) -> StagingWriterLayout:
    """Return one writer's canonical projection.

    :param layout: Canonical packed layout.
    :param writer_id: Exact writer identity.
    :returns: Writer projection.
    :raises ValueError: If the writer is absent.
    """

    for writer_layout in layout.writers:
        if writer_layout.writer_id == writer_id:
            return writer_layout
    raise ValueError(f"writer is absent from packed layout: {writer_id}")


def _validate_source_binding(
    layout: StagingChunkLayout,
    binding: StagingEndpointBufferBinding,
) -> None:
    """Validate a source binding against one canonical layout.

    :param layout: Canonical packed layout.
    :param binding: Source-side registered buffer binding.
    :raises ValueError: If endpoint, geometry, ordering, or page counts differ.
    """

    if binding.endpoint is not StagingEndpoint.SOURCE:
        raise ValueError("packed source coordinator requires a source binding")
    if len(binding.components) != len(layout.component_spans):
        raise ValueError("packed source binding component count differs from layout")
    for geometry, span, active in zip(
        layout.source_components,
        layout.component_spans,
        binding.components,
        strict=True,
    ):
        if active.component.component_id != span.component_id:
            raise ValueError(
                "packed source binding component order differs from layout"
            )
        if active.component.geometry != geometry:
            raise ValueError("packed source binding geometry differs from layout")
        expected_page_count = span.physical_token_count // geometry.page_size
        if active.page_offset != span.source_index_offset:
            raise ValueError("packed source binding page offset differs from layout")
        if active.page_count != expected_page_count:
            raise ValueError("packed source binding page count differs from layout")
        if len(active.page_array) != expected_page_count:
            raise ValueError("packed source binding page snapshot is incomplete")


def _checked_uint64_region(address: int, length_bytes: int, label: str) -> int:
    """Validate one address region representable by NIXL descriptors.

    :param address: Region base address.
    :param length_bytes: Positive region length.
    :param label: Reader-facing region label.
    :returns: Exclusive region end.
    """

    if type(address) is not int or address <= 0:
        raise ValueError(f"{label} address must be a positive integer")
    if type(length_bytes) is not int or length_bytes <= 0:
        raise ValueError(f"{label} length must be a positive integer")
    end = address + length_bytes
    if address >= _UINT64_LIMIT or end > _UINT64_LIMIT:
        raise ValueError(f"{label} exceeds the uint64 address space")
    return end


@dataclasses.dataclass(frozen=True)
class PackedDestinationRouteBinding:
    """Process-lifetime source-lane authority for one destination route.

    :ivar peer: Exact authenticated destination agent process.
    :ivar arena_generation: Generation of the retained staging registration.
    :ivar destination_gpu_id: Bootstrap-derived destination CUDA device.
    :ivar topology: Exact asymmetric transfer topology.
    :ivar visibility_policy_digest: Decode-selected route-policy identity.
    :ivar base_address: Registered staging allocation base.
    :ivar total_size: Registered staging allocation capacity.
    :ivar alignment_bytes: Required lease and writer-projection alignment.
    """

    peer: PackedPeerIdentity
    arena_generation: bytes
    destination_gpu_id: int
    topology: PackedTopology
    visibility_policy_digest: bytes
    base_address: int
    total_size: int
    alignment_bytes: int

    def __post_init__(self) -> None:
        """Validate immutable process-lifetime route authority."""

        if type(self.peer) is not PackedPeerIdentity:
            raise TypeError("route peer must be PackedPeerIdentity")
        if type(self.topology) is not PackedTopology:
            raise TypeError("route topology must be PackedTopology")
        if type(self.arena_generation) is not bytes:
            raise TypeError("arena_generation must be bytes")
        if len(self.arena_generation) != _CAPABILITY_GENERATION_BYTES:
            raise ValueError(
                "arena_generation must contain "
                f"{_CAPABILITY_GENERATION_BYTES} bytes, got "
                f"{len(self.arena_generation)}"
            )
        if type(self.visibility_policy_digest) is not bytes:
            raise TypeError("visibility_policy_digest must be bytes")
        if len(self.visibility_policy_digest) != PACKED_VISIBILITY_POLICY_DIGEST_BYTES:
            raise ValueError(
                "visibility_policy_digest must contain "
                f"{PACKED_VISIBILITY_POLICY_DIGEST_BYTES} bytes, got "
                f"{len(self.visibility_policy_digest)}"
            )
        if type(self.destination_gpu_id) is not int or self.destination_gpu_id < 0:
            raise ValueError("destination_gpu_id must be a non-negative integer")
        if type(self.alignment_bytes) is not int or self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be a positive integer")
        if self.topology.alignment_bytes != self.alignment_bytes:
            raise ValueError("route alignment differs from its bound topology")
        _checked_uint64_region(
            self.base_address,
            self.total_size,
            "packed destination route",
        )
        if self.base_address % self.alignment_bytes != 0:
            raise ValueError("route base address is not aligned")


@dataclasses.dataclass(frozen=True)
class PackedDestinationCapability:
    """Request-scoped authority derived from a process-lifetime route.

    :ivar route: Exact process, registration, topology, and policy authority.
    :ivar request_generation: Request generation authorized to use the route.
    """

    route: PackedDestinationRouteBinding
    request_generation: bytes

    def __post_init__(self) -> None:
        """Validate one immutable request-scoped capability."""

        if type(self.route) is not PackedDestinationRouteBinding:
            raise TypeError("capability route must be PackedDestinationRouteBinding")
        if type(self.request_generation) is not bytes:
            raise TypeError("request_generation must be bytes")
        if len(self.request_generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "request_generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes, got "
                f"{len(self.request_generation)}"
            )

    def validate_ready(
        self,
        message: PackedReady,
        layout: StagingChunkLayout,
        writer_layout: StagingWriterLayout,
    ) -> int:
        """Validate READY entirely inside this registered capability.

        :param message: Untrusted READY payload.
        :param layout: Source-rebuilt canonical layout.
        :param writer_layout: Source-local canonical writer projection.
        :returns: Exact registered writer destination address.
        """

        if message.key.request_generation != self.request_generation:
            raise ValueError("READY request generation differs from capability")
        route = self.route
        if message.visibility_policy_digest != route.visibility_policy_digest:
            raise ValueError("READY visibility policy differs from capability")
        capability_end = _checked_uint64_region(
            route.base_address,
            route.total_size,
            "packed destination capability",
        )
        lease_end = _checked_uint64_region(
            message.lease_base_address,
            layout.total_bytes,
            "packed lease",
        )
        if message.lease_base_address % route.alignment_bytes != 0:
            raise ValueError("READY lease base address is not aligned")
        if (
            message.lease_base_address < route.base_address
            or lease_end > capability_end
        ):
            raise ValueError(
                "READY lease exceeds the registered destination capability"
            )

        destination_address = message.lease_base_address + writer_layout.lease_offset
        projection_end = _checked_uint64_region(
            destination_address,
            writer_layout.length_bytes,
            "packed writer projection",
        )
        if projection_end > lease_end:
            raise ValueError("READY writer projection exceeds its packed lease")
        if destination_address % route.alignment_bytes != 0:
            raise ValueError("READY writer projection address is not aligned")
        return destination_address


class PackedReadyError(RuntimeError):
    """Source-side rejection of an untrusted READY message."""


@dataclasses.dataclass(frozen=True)
class PackedSourceTransfer:
    """Canonical one-shot source work produced by a validated READY.

    :ivar key: Request and chunk identity.
    :ivar destination: Bootstrap-derived registered destination capability.
    :ivar layout: Locally rebuilt canonical packed layout.
    :ivar writer_id: Local authenticated writer identity.
    :ivar source_binding: Immutable request-local source pages and buffers.
    :ivar lease_id: Decode lease identity copied into the terminal outcome.
    :ivar destination_address: Exact destination projection base address.
    :ivar length_bytes: Exact local canonical DMA length.
    """

    key: PackedChunkKey
    destination: PackedDestinationCapability
    layout: StagingChunkLayout
    writer_id: StagingWriterId
    source_binding: StagingEndpointBufferBinding
    lease_id: int
    destination_address: int
    length_bytes: int


@dataclasses.dataclass(frozen=True)
class _PendingPackedSource:
    """Immutable canonical truth retained until exactly one READY is accepted."""

    destination: PackedDestinationCapability
    layout: StagingChunkLayout
    writer_id: StagingWriterId
    source_binding: StagingEndpointBufferBinding


class PackedReadyCoordinator:
    """Thread-safe one-shot validation of decode READY messages."""

    _lock: threading.Lock
    _pending: dict[tuple[PackedChunkKey, PackedPeerIdentity], _PendingPackedSource]

    def __init__(self) -> None:
        """Initialize an empty source-side READY registry."""

        self._lock = threading.Lock()
        self._pending = {}

    def register_chunk(
        self,
        *,
        key: PackedChunkKey,
        destination: PackedDestinationCapability,
        writer_id: StagingWriterId,
        spec: PackedLayoutSpec,
        source_binding: StagingEndpointBufferBinding,
    ) -> PackedPrepare:
        """Register local truth and build the PREPARE sent to decode.

        :param key: Request and chunk identity.
        :param destination: Exact bootstrap-derived registered destination.
        :param writer_id: Local transfer writer identity.
        :param spec: Source-built canonical layout input.
        :param source_binding: Immutable source registration and page snapshots.
        :returns: PREPARE carrying the canonical layout digest.
        """

        if key.chunk_id != spec.chunk_id:
            raise ValueError(
                f"chunk key/spec mismatch: {key.chunk_id} and {spec.chunk_id}"
            )
        if key.request_generation != destination.request_generation:
            raise ValueError("chunk request generation differs from destination")
        if spec.topology != destination.route.topology:
            raise ValueError("packed topology differs from destination capability")
        layout = spec.build()
        writer_layout_for(layout, writer_id)
        _validate_source_binding(layout, source_binding)
        pending = _PendingPackedSource(
            destination=destination,
            layout=layout,
            writer_id=writer_id,
            source_binding=source_binding,
        )
        route_key = (key, destination.route.peer)
        with self._lock:
            if route_key in self._pending:
                raise ValueError(
                    "packed source route is already registered: "
                    f"{key} via {destination.route.peer}"
                )
            self._pending[route_key] = pending
        return PackedPrepare(
            key=key,
            writer_id=writer_id,
            spec=spec,
            digest=layout.digest,
        )

    def handle_ready(
        self,
        message: PackedReady,
        authenticated_decode_peer: PackedPeerIdentity,
    ) -> PackedSourceTransfer:
        """Consume one validated READY and hand out exactly one DMA submission.

        Every address and shape field from READY is checked against locally
        retained canonical truth. The returned DMA length and gather layout are
        local values, never values selected by the peer.

        :param message: Untrusted READY payload.
        :param authenticated_decode_peer: NIXL process bound to the exact route.
        :returns: Canonical gather, destination, and completion-notification work.
        :raises PackedReadyError: If READY conflicts with local truth.
        """

        with self._lock:
            route_key = (message.key, authenticated_decode_peer)
            pending = self._pending.get(route_key)
            if pending is None:
                raise PackedReadyError(
                    "packed source route is not pending: "
                    f"{message.key} via {authenticated_decode_peer}"
                )
            try:
                transfer = self._validate_ready(
                    message, authenticated_decode_peer, pending
                )
            except ValueError as error:
                raise PackedReadyError(
                    f"packed READY rejected for {message.key}: {error}"
                ) from error
            del self._pending[route_key]
            return transfer

    def retire_pending(
        self,
        key: PackedChunkKey,
        decode_peer: PackedPeerIdentity,
    ) -> None:
        """Forget one route that cannot receive READY.

        :param key: Request and chunk identity.
        :param decode_peer: Exact registered decode process.
        """

        with self._lock:
            self._pending.pop((key, decode_peer), None)

    @staticmethod
    def _validate_ready(
        message: PackedReady,
        authenticated_decode_peer: PackedPeerIdentity,
        pending: _PendingPackedSource,
    ) -> PackedSourceTransfer:
        """Validate READY without mutating coordinator ownership.

        :param message: Untrusted READY payload.
        :param authenticated_decode_peer: Authenticated route peer.
        :param pending: Locally retained canonical source truth.
        :returns: Canonical one-shot transfer work.
        """

        if authenticated_decode_peer != pending.destination.route.peer:
            raise ValueError("decode peer does not match the registered route")
        if message.writer_id != pending.writer_id:
            raise ValueError("writer identity differs from local transfer writer")
        if message.digest != pending.layout.digest:
            raise ValueError("digest differs from the local canonical layout")
        writer_layout = writer_layout_for(pending.layout, pending.writer_id)
        if message.projection_offset != writer_layout.lease_offset:
            raise ValueError("projection offset differs from the canonical layout")
        if message.projection_length != writer_layout.length_bytes:
            raise ValueError("projection length differs from the canonical layout")
        projection_end = writer_layout.lease_offset + writer_layout.length_bytes
        if projection_end > pending.layout.total_bytes:
            raise ValueError("canonical writer projection exceeds the packed lease")
        destination_address = pending.destination.validate_ready(
            message,
            pending.layout,
            writer_layout,
        )
        return PackedSourceTransfer(
            key=message.key,
            destination=pending.destination,
            layout=pending.layout,
            writer_id=pending.writer_id,
            source_binding=pending.source_binding,
            lease_id=message.lease_id,
            destination_address=destination_address,
            length_bytes=writer_layout.length_bytes,
        )


@triton.jit
def _gather_packed_bytes_kernel(
    entry_ptrs,
    page_indices,
    staging,
    staging_offset,
    physical_tokens,
    source_token_bytes,
    source_offset_bytes,
    copy_bytes_per_token: tl.constexpr,
    page_size: tl.constexpr,
    bytes_per_entry,
    block_size: tl.constexpr,
):
    entry_index = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1).to(tl.int64)
    offsets = block_index * block_size + tl.arange(0, block_size).to(tl.int64)
    mask = offsets < bytes_per_entry
    token_index = offsets // copy_bytes_per_token
    token_byte = offsets % copy_bytes_per_token
    page_index = token_index // page_size
    page_offset = token_index % page_size
    physical_page = tl.load(page_indices + page_index, mask=mask, other=0).to(tl.int64)
    entry_ptr = tl.load(entry_ptrs + entry_index).to(staging.dtype)
    source_offsets = (
        physical_page * page_size * source_token_bytes.to(tl.int64)
        + page_offset * source_token_bytes.to(tl.int64)
        + source_offset_bytes.to(tl.int64)
        + token_byte
    )
    values = tl.load(entry_ptr + source_offsets, mask=mask)
    destination_offsets = (
        staging_offset.to(tl.int64)
        + entry_index * physical_tokens.to(tl.int64) * copy_bytes_per_token
        + offsets
    )
    tl.store(staging + destination_offsets, values, mask=mask)


@triton.jit
def _scatter_packed_bytes_kernel(
    entry_ptrs,
    page_indices,
    staging,
    staging_offset,
    physical_tokens,
    destination_token_bytes,
    destination_offset_bytes,
    copy_bytes_per_token: tl.constexpr,
    page_size: tl.constexpr,
    bytes_per_entry,
    block_size: tl.constexpr,
):
    entry_index = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1).to(tl.int64)
    offsets = block_index * block_size + tl.arange(0, block_size).to(tl.int64)
    mask = offsets < bytes_per_entry
    token_index = offsets // copy_bytes_per_token
    token_byte = offsets % copy_bytes_per_token
    page_index = token_index // page_size
    page_offset = token_index % page_size
    physical_page = tl.load(page_indices + page_index, mask=mask, other=0).to(tl.int64)
    entry_ptr = tl.load(entry_ptrs + entry_index).to(staging.dtype)
    source_offsets = (
        staging_offset.to(tl.int64)
        + entry_index * physical_tokens.to(tl.int64) * copy_bytes_per_token
        + offsets
    )
    values = tl.load(staging + source_offsets, mask=mask)
    destination_offsets = (
        physical_page * page_size * destination_token_bytes.to(tl.int64)
        + page_offset * destination_token_bytes.to(tl.int64)
        + destination_offset_bytes.to(tl.int64)
        + token_byte
    )
    tl.store(entry_ptr + destination_offsets, values, mask=mask)


def _validate_packed_byte_tensor(tensor: torch.Tensor, label: str) -> None:
    """Validate one retained contiguous byte allocation.

    :param tensor: Candidate packed storage.
    :param label: Reader-facing owner label.
    """

    if tensor.dtype is not torch.uint8:
        raise TypeError(f"{label} tensor must have dtype uint8")
    if tensor.ndim != 1:
        raise ValueError(f"{label} tensor must be one-dimensional")
    if not tensor.is_contiguous():
        raise ValueError(f"{label} tensor must be contiguous")
    if tensor.numel() <= 0:
        raise ValueError(f"{label} tensor must not be empty")


def _register_packed_tensor(
    agent: PackedMemoryAgent,
    tensor: torch.Tensor,
    gpu_id: int,
) -> object:
    """Register one retained byte tensor with NIXL.

    :param agent: NIXL registration agent.
    :param tensor: Retained byte tensor.
    :param gpu_id: CUDA device identifier.
    :returns: Opaque registration handle.
    """

    registration = agent.register_memory(
        [(tensor.data_ptr(), tensor.numel(), gpu_id, "")],
        "VRAM",
    )
    if registration is None:
        raise RuntimeError("NIXL returned no packed-memory registration")
    if isinstance(registration, list) and len(registration) == 0:
        raise RuntimeError("NIXL returned an empty packed-memory registration")
    return registration


@dataclasses.dataclass(frozen=True)
class _QuarantinedPackedRegistration:
    """Strongly retained registration whose safe deregistration is unproven."""

    tensor: torch.Tensor
    registration: object
    reason: str
    owners: tuple[object, ...] = ()


class PackedRegistrationQuarantine:
    """Process-lifetime retention for ambiguous NIXL registrations."""

    _lock: threading.Lock
    _resources: list[_QuarantinedPackedRegistration]

    def __init__(self) -> None:
        """Initialize an empty strong-retention set."""

        self._lock = threading.Lock()
        self._resources = []

    def retain(
        self,
        tensor: torch.Tensor,
        registration: object,
        reason: str,
        owners: tuple[object, ...] = (),
    ) -> None:
        """Retain one registration without attempting unsafe cleanup.

        :param tensor: Allocation possibly still referenced by NIXL or CUDA.
        :param registration: Opaque NIXL registration handle.
        :param reason: Stable quarantine reason.
        :param owners: Additional asynchronous owners requiring retention.
        """

        if len(reason) == 0:
            raise ValueError("registration quarantine reason must not be empty")
        resource = _QuarantinedPackedRegistration(
            tensor=tensor,
            registration=registration,
            reason=reason,
            owners=tuple(owners),
        )
        with self._lock:
            self._resources.append(resource)

    @property
    def count(self) -> int:
        """Return the number of strongly retained registrations.

        :returns: Quarantined registration count.
        """

        with self._lock:
            return len(self._resources)

    def retains(self, owner: object) -> bool:
        """Return whether an exact asynchronous owner is strongly retained.

        :param owner: Object whose identity must remain process-live.
        :returns: Whether any quarantined cohort retains the object.
        """

        with self._lock:
            return any(
                retained_owner is owner
                for resource in self._resources
                for retained_owner in resource.owners
            )


PACKED_REGISTRATION_QUARANTINE = PackedRegistrationQuarantine()


class PackedTransferLaneState(enum.StrEnum):
    """Ownership state of one presized source transfer lane."""

    IDLE = "idle"
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    POISONED = "poisoned"
    CLOSED = "closed"


@dataclasses.dataclass(frozen=True)
class PackedSourceVisibilityCompletion:
    """Typed source-executor receipt for one exact submitted handle.

    Transport adapters construct this only after the bound NIXL handle reaches
    terminal DONE or the exported CUDA event has been recorded.

    :ivar transport_handle: Exact handle whose ownership moved into the lane.
    :ivar evidence: Bounded wire evidence produced by that source action.
    """

    transport_handle: object
    evidence: PackedWriterVisibilityEvidence

    def __post_init__(self) -> None:
        """Validate one typed source completion receipt."""

        if self.transport_handle is None:
            raise ValueError("source visibility transport_handle must not be None")
        if type(self.evidence) is not PackedWriterVisibilityEvidence:
            raise TypeError(
                "source visibility evidence must be PackedWriterVisibilityEvidence"
            )


class PackedTransferLane:
    """Presized per-route source buffer and exact transfer ownership state.

    One lane owns one tensor and NIXL registration for its entire lifetime.
    It cannot grow, cannot be reused while reserved or in flight, and emits a
    terminal wire outcome only after no-submit proof or terminal transport
    completion. Ambiguous submission poisons the lane and emits no outcome.
    """

    _active_transfer: PackedSourceTransfer | None
    _agent: PackedMemoryAgent
    _destination_route: PackedDestinationRouteBinding
    _gpu_id: int
    _lock: threading.Lock
    _poison_reason: str | None
    _quarantine: PackedRegistrationQuarantine
    _quarantine_retained: bool
    _registration: object | None
    _state: PackedTransferLaneState
    _tensor: torch.Tensor | None
    _transport_handle: object | None
    _transport_owners: tuple[object, ...]
    _visibility_policy: PackedDestinationVisibilityPolicy

    def __init__(
        self,
        *,
        agent: PackedMemoryAgent,
        destination_route: PackedDestinationRouteBinding,
        visibility_policy: PackedDestinationVisibilityPolicy,
        gpu_id: int,
        tensor: torch.Tensor,
        quarantine: PackedRegistrationQuarantine = PACKED_REGISTRATION_QUARANTINE,
    ) -> None:
        """Register one presized source lane.

        :param agent: NIXL memory-registration agent.
        :param destination_route: Process-lifetime route owned by this lane.
        :param visibility_policy: Exact pinned path and CUDA policy.
        :param gpu_id: Source CUDA device identifier.
        :param tensor: Presized route-owned byte tensor.
        :param quarantine: Strong-retention owner for ambiguous cleanup.
        """

        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer")
        if type(destination_route) is not PackedDestinationRouteBinding:
            raise TypeError("destination_route must be PackedDestinationRouteBinding")
        if type(visibility_policy) is not PackedDestinationVisibilityPolicy:
            raise TypeError(
                "visibility_policy must be PackedDestinationVisibilityPolicy"
            )
        if visibility_policy.digest != destination_route.visibility_policy_digest:
            raise ValueError(
                "transfer lane visibility policy differs from destination route"
            )
        _validate_packed_byte_tensor(tensor, "packed transfer lane")
        registration = _register_packed_tensor(agent, tensor, gpu_id)
        self._active_transfer = None
        self._agent = agent
        self._destination_route = destination_route
        self._gpu_id = gpu_id
        self._lock = threading.Lock()
        self._poison_reason = None
        self._quarantine = quarantine
        self._quarantine_retained = False
        self._registration = registration
        self._state = PackedTransferLaneState.IDLE
        self._tensor = tensor
        self._transport_handle = None
        self._transport_owners = ()
        self._visibility_policy = visibility_policy

    @property
    def capacity(self) -> int:
        """Return immutable registered lane capacity.

        :returns: Byte capacity.
        """

        tensor = self._require_tensor()
        return tensor.numel()

    @property
    def data_ptr(self) -> int:
        """Return the registered source address.

        :returns: Source lane base pointer.
        """

        return self._require_tensor().data_ptr()

    @property
    def state(self) -> PackedTransferLaneState:
        """Return a stable lane-state snapshot.

        :returns: Current ownership state.
        """

        with self._lock:
            return self._state

    @property
    def tensor(self) -> torch.Tensor:
        """Return retained registered source storage.

        :returns: Presized byte tensor.
        """

        return self._require_tensor()

    def reserve(self, transfer: PackedSourceTransfer) -> None:
        """Reserve this route lane for one exact canonical transfer.

        :param transfer: Source transfer produced by validated READY.
        """

        with self._lock:
            if self._state is not PackedTransferLaneState.IDLE:
                raise RuntimeError(
                    f"packed transfer lane cannot reserve in state {self._state.value}"
                )
            tensor = self._require_tensor_locked()
            if transfer.destination.route != self._destination_route:
                raise ValueError(
                    "packed transfer destination differs from its route lane"
                )
            if (
                transfer.key.request_generation
                != transfer.destination.request_generation
            ):
                raise ValueError(
                    "packed transfer key generation differs from its capability"
                )
            if transfer.length_bytes > tensor.numel():
                raise ValueError(
                    "packed transfer exceeds presized lane capacity: "
                    f"{transfer.length_bytes} > {tensor.numel()}"
                )
            self._active_transfer = transfer
            self._state = PackedTransferLaneState.RESERVED

    def arm_submission(
        self,
        transport_handle: object,
        *,
        owners: tuple[object, ...] = (),
    ) -> None:
        """Take exact DMA ownership before the external transfer/post call.

        Runtime integration must call this after handle initialization and
        before invoking NIXL transfer/post. Any exception from that external
        call is ambiguous unless the transport explicitly proves no submission.

        :param transport_handle: Initialized opaque NIXL transfer handle.
        :param owners: Agent endpoint or progress owners needed by the handle.
        """

        with self._lock:
            if self._state is not PackedTransferLaneState.RESERVED:
                raise RuntimeError(
                    "packed transfer submission requires a reserved lane"
                )
            if transport_handle is None:
                raise ValueError("transport_handle must not be None")
            self._transport_handle = transport_handle
            self._transport_owners = tuple(owners)
            self._state = PackedTransferLaneState.IN_FLIGHT

    def abort_armed_without_submit(self, reason: str) -> PackedWriterOutcome:
        """Disarm an initialized handle after explicit no-submit proof.

        :param reason: Bounded terminal failure reason.
        :returns: Exact error outcome proving no destination DMA was posted.
        """

        with self._lock:
            if self._state is not PackedTransferLaneState.IN_FLIGHT:
                raise RuntimeError(
                    "armed no-submit abort requires in-flight lane ownership"
                )
            transfer = self._require_active_transfer_locked()
            if self._transport_handle is None:
                raise RuntimeError("armed lane has no initialized transport handle")
            outcome = self._outcome_locked(
                transfer,
                PackedWriterOutcomeStatus.ERROR,
                None,
                reason,
            )
            self._active_transfer = None
            self._transport_handle = None
            self._transport_owners = ()
            self._state = PackedTransferLaneState.IDLE
            return outcome

    def abort_before_submit(
        self,
        reason: str,
        *,
        source_stream_quiesced: bool = True,
    ) -> PackedWriterOutcome:
        """Produce terminal error after proving no destination DMA was submitted.

        :param reason: Bounded terminal failure reason.
        :param source_stream_quiesced: Whether source gather work is terminal.
        :returns: Exact authenticated error outcome for decode.
        """

        with self._lock:
            if self._state is not PackedTransferLaneState.RESERVED:
                raise RuntimeError("pre-submit abort requires a reserved lane")
            transfer = self._require_active_transfer_locked()
            outcome = self._outcome_locked(
                transfer,
                PackedWriterOutcomeStatus.ERROR,
                None,
                reason,
            )
            self._active_transfer = None
            if source_stream_quiesced:
                self._state = PackedTransferLaneState.IDLE
                return outcome
            self._poison_and_retain_locked(
                f"{reason}; source gather stream did not quiesce"
            )
            return outcome

    def mark_submission_ambiguous(self, reason: str) -> None:
        """Poison a possibly submitted lane without claiming terminal DMA.

        :param reason: Ambiguous submission or connection-loss reason.
        """

        if len(reason) == 0:
            raise ValueError("poison reason must not be empty")
        with self._lock:
            if self._state not in (
                PackedTransferLaneState.RESERVED,
                PackedTransferLaneState.IN_FLIGHT,
            ):
                raise RuntimeError(
                    "ambiguous submission requires reserved or in-flight ownership"
                )
            self._poison_and_retain_locked(reason)

    def mark_transport_terminal(
        self,
        error: str | None = None,
        *,
        completion: PackedSourceVisibilityCompletion | None = None,
    ) -> PackedWriterOutcome:
        """Produce an exact outcome after terminal NIXL DONE or ERR.

        :param error: Terminal transport error, or ``None`` for success.
        :param completion: Receipt produced by the terminal transport executor.
        :returns: Exact authenticated terminal outcome for decode.
        """

        with self._lock:
            if self._state not in (
                PackedTransferLaneState.IN_FLIGHT,
                PackedTransferLaneState.POISONED,
            ):
                raise RuntimeError(
                    "transport terminality requires in-flight or poisoned ownership"
                )
            if self._transport_handle is None:
                raise RuntimeError(
                    "transport terminality requires the exact submitted handle"
                )
            transfer = self._require_active_transfer_locked()
            status = (
                PackedWriterOutcomeStatus.DONE
                if error is None
                else PackedWriterOutcomeStatus.ERROR
            )
            if status is PackedWriterOutcomeStatus.DONE:
                if type(completion) is not PackedSourceVisibilityCompletion:
                    raise TypeError(
                        "successful transport terminality requires a typed "
                        "source completion"
                    )
                if completion.transport_handle is not self._transport_handle:
                    raise ValueError(
                        "source completion belongs to another transport handle"
                    )
                visibility = completion.evidence
                self._visibility_policy.validate_evidence(visibility)
            else:
                visibility = None
            if status is PackedWriterOutcomeStatus.ERROR and completion is not None:
                raise ValueError(
                    "terminal transport error must not contain a source completion"
                )
            outcome = self._outcome_locked(transfer, status, visibility, error)
            self._active_transfer = None
            self._transport_handle = None
            self._transport_owners = ()
            if self._poison_reason is None:
                self._state = PackedTransferLaneState.IDLE
            else:
                self._state = PackedTransferLaneState.POISONED
            return outcome

    def close(self) -> None:
        """Terminally close or leak-safely quarantine this registration.

        Closing is idempotent. Outstanding or poisoned ownership is retained
        process-wide instead of being deregistered speculatively.
        """

        with self._lock:
            if self._state is PackedTransferLaneState.CLOSED:
                return
            tensor = self._require_tensor_locked()
            registration = self._require_registration_locked()
            if self._active_transfer is not None or self._poison_reason is not None:
                reason = (
                    self._poison_reason
                    or "packed transfer lane closed with outstanding ownership"
                )
                self._poison_and_retain_locked(reason)
                self._state = PackedTransferLaneState.CLOSED
                raise RuntimeError(reason)
            try:
                self._agent.deregister_memory(registration)
            except Exception:
                reason = "packed transfer lane deregistration failed"
                logger.error("%s:\n%s", reason, traceback.format_exc())
                self._retain_locked(tensor, registration, reason)
                self._state = PackedTransferLaneState.CLOSED
                raise
            self._registration = None
            self._tensor = None
            self._state = PackedTransferLaneState.CLOSED

    def _outcome_locked(
        self,
        transfer: PackedSourceTransfer,
        status: PackedWriterOutcomeStatus,
        visibility: PackedWriterVisibilityEvidence | None,
        reason: str | None,
    ) -> PackedWriterOutcome:
        """Build exact terminal identity while lane ownership is serialized.

        :param transfer: Active validated transfer.
        :param status: Proven terminal status.
        :param visibility: Successful writer-side evidence.
        :param reason: Error reason for ``ERROR``.
        :returns: Exact terminal writer outcome.
        """

        return PackedWriterOutcome(
            key=transfer.key,
            writer_id=transfer.writer_id,
            digest=transfer.layout.digest,
            lease_id=transfer.lease_id,
            status=status,
            visibility=visibility,
            reason=reason,
        )

    def _poison_and_retain_locked(self, reason: str) -> None:
        """Poison and strongly retain this lane while its lock is held.

        :param reason: Stable quarantine reason.
        """

        self._poison_reason = reason
        tensor = self._require_tensor_locked()
        registration = self._require_registration_locked()
        self._retain_locked(tensor, registration, reason)
        self._state = PackedTransferLaneState.POISONED

    def _retain_locked(
        self,
        tensor: torch.Tensor,
        registration: object,
        reason: str,
    ) -> None:
        """Strongly retain a resource exactly once.

        :param tensor: Registered tensor.
        :param registration: Opaque registration handle.
        :param reason: Stable quarantine reason.
        """

        if self._quarantine_retained:
            return
        transport_handle = self._transport_handle
        transport_owners = (transport_handle,) if transport_handle is not None else ()
        self._quarantine.retain(
            tensor,
            registration,
            reason,
            owners=(self._agent, *self._transport_owners, *transport_owners),
        )
        self._quarantine_retained = True

    def _require_active_transfer_locked(self) -> PackedSourceTransfer:
        """Return the active exact transfer while the lane lock is held.

        :returns: Reserved or in-flight transfer.
        """

        transfer = self._active_transfer
        if transfer is None:
            raise RuntimeError("packed transfer lane has no active transfer")
        return transfer

    def _require_registration_locked(self) -> object:
        """Return the registration while the lane lock is held.

        :returns: Opaque live or quarantined registration.
        """

        registration = self._registration
        if registration is None:
            raise RuntimeError("packed transfer lane registration is closed")
        return registration

    def _require_tensor(self) -> torch.Tensor:
        """Return retained storage under the lane lock.

        :returns: Retained byte tensor.
        """

        with self._lock:
            return self._require_tensor_locked()

    def _require_tensor_locked(self) -> torch.Tensor:
        """Return retained storage while the lane lock is held.

        :returns: Retained byte tensor.
        """

        tensor = self._tensor
        if tensor is None:
            raise RuntimeError("packed transfer lane storage is closed")
        return tensor


class PackedGatherError(RuntimeError):
    """Gather failure carrying an exact no-DMA terminal outcome."""

    outcome: PackedWriterOutcome

    def __init__(self, outcome: PackedWriterOutcome) -> None:
        """Initialize a source-gather terminal failure.

        :param outcome: Exact error outcome safe to deliver to decode.
        """

        self.outcome = outcome
        super().__init__(outcome.reason)


@dataclasses.dataclass(frozen=True)
class PackedScatterSubmission:
    """Resources retained until one asynchronous scatter event is terminal.

    :ivar event: CUDA completion event.
    :ivar resources: Temporary pointer and page tensors used by kernels.
    """

    event: torch.cuda.Event
    resources: tuple[torch.Tensor, ...]


class PackedCopyExecutor:
    """Component-aware raw-byte gather and scatter kernel dispatcher."""

    _device: torch.device
    _failed_scatter_resources: list[torch.Tensor]
    _gpu_id: int
    _scatter_base_address: int | None
    _scatter_buffer: torch.Tensor | None
    _scatter_stream: torch.cuda.Stream
    _source_stream: torch.cuda.Stream

    def __init__(
        self,
        *,
        gpu_id: int,
        scatter_buffer: torch.Tensor | None = None,
    ) -> None:
        """Initialize dedicated gather and scatter streams.

        :param gpu_id: CUDA device identifier.
        :param scatter_buffer: Decode staging-pool byte tensor.
        """

        self._device = torch.device(f"cuda:{gpu_id}")
        self._failed_scatter_resources = []
        self._gpu_id = gpu_id
        self._scatter_buffer = scatter_buffer
        self._scatter_base_address = (
            scatter_buffer.data_ptr() if scatter_buffer is not None else None
        )
        torch.cuda.set_device(gpu_id)
        self._source_stream = torch.cuda.Stream(device=self._device)
        self._scatter_stream = torch.cuda.Stream(device=self._device)

    def gather(
        self,
        *,
        transfer: PackedSourceTransfer,
        source_lane: PackedTransferLane,
        producer_event: torch.cuda.Event | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> int:
        """Gather one writer projection and wait until NIC-visible.

        Exactly one producer dependency is required. The caller owns the lane
        from successful return until exact transport terminality. This method
        synchronizes gather work, not a later NIXL read.

        :param transfer: Canonical work produced by validated READY.
        :param source_lane: Presized route-owned registered staging lane.
        :param producer_event: Event recorded after every source KV write.
        :param producer_stream: Stream containing every source KV write.
        :returns: Exact contiguous DMA length.
        :raises PackedGatherError: If no destination DMA was submitted.
        """

        if (producer_event is None) == (producer_stream is None):
            raise ValueError(
                "packed gather requires exactly one producer event or stream"
            )
        writer_layout = writer_layout_for(transfer.layout, transfer.writer_id)
        if transfer.length_bytes != writer_layout.length_bytes:
            raise ValueError("source transfer length differs from canonical writer")
        source_lane.reserve(transfer)
        retained: list[torch.Tensor] = []
        try:
            with torch.cuda.stream(self._source_stream):
                if producer_event is not None:
                    self._source_stream.wait_event(producer_event)
                else:
                    if producer_stream is None:
                        raise RuntimeError("producer dependency validation drifted")
                    self._source_stream.wait_stream(producer_stream)
                source_lane.tensor[: writer_layout.length_bytes].zero_()
                for group in writer_layout.copy_groups:
                    active = transfer.source_binding.require(group.component_id)
                    entry_ptrs = torch.tensor(
                        [
                            active.component.tensor_ptrs[entry_index]
                            for entry_index in group.source_entry_indices
                        ],
                        dtype=torch.int64,
                        device=self._device,
                    )
                    page_indices = torch.from_numpy(
                        np.array(active.page_array, dtype=np.int32, copy=True)
                    ).to(self._device)
                    retained.extend((entry_ptrs, page_indices))
                    physical_tokens = group.page_count * active.component.page_size
                    bytes_per_entry = physical_tokens * group.copy_bytes_per_token
                    grid = (
                        len(group.source_entry_indices),
                        triton.cdiv(bytes_per_entry, 256),
                    )
                    _gather_packed_bytes_kernel[grid](
                        entry_ptrs,
                        page_indices,
                        source_lane.tensor,
                        group.packed_offset - writer_layout.lease_offset,
                        physical_tokens,
                        group.source_token_bytes,
                        group.source_offset_bytes,
                        copy_bytes_per_token=group.copy_bytes_per_token,
                        page_size=active.component.page_size,
                        bytes_per_entry=bytes_per_entry,
                        block_size=256,
                    )
            self._source_stream.synchronize()
        except Exception as error:
            logger.error("Packed source gather failed:\n%s", traceback.format_exc())
            source_stream_quiesced = False
            try:
                self._source_stream.synchronize()
                source_stream_quiesced = True
            except Exception:  # noqa: BLE001
                logger.error(
                    "Packed source gather stream did not quiesce:\n%s",
                    traceback.format_exc(),
                )
            outcome = source_lane.abort_before_submit(
                "packed source gather failed",
                source_stream_quiesced=source_stream_quiesced,
            )
            raise PackedGatherError(outcome) from error
        return writer_layout.length_bytes

    def scatter(
        self,
        work: PackedScatterWork,
        visibility: tuple[PackedDestinationVisibilityProof, ...],
    ) -> PackedScatterSubmission:
        """Launch component-aware scatter without blocking the caller.

        :param work: Protocol-owned immutable scatter inputs.
        :param visibility: Per-writer CUDA consumer-visibility evidence.
        :returns: Completion event and retained temporary tensors.
        """

        visibility_by_writer = {evidence.writer_id: evidence for evidence in visibility}
        expected_writers = {
            writer_layout.writer_id for writer_layout in work.layout.writers
        }
        if len(visibility_by_writer) != len(visibility):
            raise ValueError("destination visibility contains duplicate writers")
        if set(visibility_by_writer) != expected_writers:
            raise ValueError(
                "destination visibility must exactly cover canonical writers"
            )
        scatter_buffer = self._scatter_buffer
        scatter_base_address = self._scatter_base_address
        if scatter_buffer is None or scatter_base_address is None:
            raise RuntimeError("decode packed scatter buffer is not configured")
        if work.lease.length_bytes < work.layout.total_bytes:
            raise ValueError("packed lease is smaller than its canonical layout")
        lease_offset = work.lease.base_address - scatter_base_address
        if lease_offset < 0:
            raise ValueError("packed lease precedes decode staging buffer")
        if lease_offset + work.lease.length_bytes > scatter_buffer.numel():
            raise ValueError("packed lease exceeds decode staging buffer")

        retained: list[torch.Tensor] = []
        try:
            with torch.cuda.stream(self._scatter_stream):
                for writer_layout in work.layout.writers:
                    for group in writer_layout.copy_groups:
                        active = work.destination_binding.require(group.component_id)
                        entry_ptrs = torch.tensor(
                            [
                                active.component.tensor_ptrs[entry_index]
                                for entry_index in group.destination_entry_indices
                            ],
                            dtype=torch.int64,
                            device=self._device,
                        )
                        page_indices = torch.from_numpy(
                            np.array(active.page_array, dtype=np.int32, copy=True)
                        ).to(self._device)
                        retained.extend((entry_ptrs, page_indices))
                        physical_tokens = group.page_count * active.component.page_size
                        bytes_per_entry = physical_tokens * group.copy_bytes_per_token
                        grid = (
                            len(group.destination_entry_indices),
                            triton.cdiv(bytes_per_entry, 256),
                        )
                        _scatter_packed_bytes_kernel[grid](
                            entry_ptrs,
                            page_indices,
                            scatter_buffer,
                            lease_offset + group.packed_offset,
                            physical_tokens,
                            group.destination_token_bytes,
                            group.destination_offset_bytes,
                            copy_bytes_per_token=group.copy_bytes_per_token,
                            page_size=active.component.page_size,
                            bytes_per_entry=bytes_per_entry,
                            block_size=256,
                        )
                event = torch.cuda.Event()
                event.record(self._scatter_stream)
        except Exception:
            logger.error(
                "Packed destination scatter failed:\n%s", traceback.format_exc()
            )
            try:
                self._scatter_stream.synchronize()
            except Exception:  # noqa: BLE001
                self._failed_scatter_resources.extend(retained)
                logger.error(
                    "Packed destination scatter stream did not quiesce:\n%s",
                    traceback.format_exc(),
                )
            raise
        return PackedScatterSubmission(event=event, resources=tuple(retained))

    def synchronize_scatter(self) -> None:
        """Wait until every submitted scatter on the dedicated stream is terminal."""

        self._scatter_stream.synchronize()


class PackedStagingArena:
    """Single owner of decode storage, registration, allocation, and copying."""

    _agent: PackedMemoryAgent
    _allocator: PackedIntervalLeaseAllocator
    _arena_generation: bytes
    _closed: bool
    _copy_executor: PackedCopyExecutor | None
    _gpu_id: int
    _lock: threading.Lock
    _peer: PackedPeerIdentity
    _protocol: PackedDecodeProtocol
    _quarantine: PackedRegistrationQuarantine
    _quarantine_retained: bool
    _registration: object | None
    _tensor: torch.Tensor | None

    def __init__(
        self,
        *,
        agent: PackedMemoryAgent,
        tensor: torch.Tensor,
        gpu_id: int,
        peer: PackedPeerIdentity,
        arena_generation: bytes,
        alignment_bytes: int = DEFAULT_STAGING_ALIGNMENT_BYTES,
        quarantine: PackedRegistrationQuarantine = PACKED_REGISTRATION_QUARANTINE,
    ) -> None:
        """Register and bind one immutable decode staging arena.

        :param agent: NIXL memory-registration agent.
        :param tensor: Decode staging-pool byte tensor.
        :param gpu_id: Destination CUDA device identifier.
        :param peer: This exact destination agent process identity.
        :param arena_generation: Generation of this registration.
        :param alignment_bytes: Lease and writer-projection alignment.
        :param quarantine: Strong-retention owner for ambiguous cleanup.
        """

        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer")
        if type(arena_generation) is not bytes:
            raise TypeError("arena_generation must be bytes")
        generation = arena_generation
        if len(generation) != _CAPABILITY_GENERATION_BYTES:
            raise ValueError(
                "arena_generation must contain "
                f"{_CAPABILITY_GENERATION_BYTES} bytes, got {len(generation)}"
            )
        _validate_packed_byte_tensor(tensor, "packed staging arena")
        allocator = PackedIntervalLeaseAllocator(
            base_address=tensor.data_ptr(),
            total_size=tensor.numel(),
            alignment_bytes=alignment_bytes,
        )
        registration = _register_packed_tensor(agent, tensor, gpu_id)
        self._agent = agent
        self._allocator = allocator
        self._arena_generation = generation
        self._closed = False
        self._copy_executor = None
        self._gpu_id = gpu_id
        self._lock = threading.Lock()
        self._peer = peer
        self._protocol = PackedDecodeProtocol(allocator)
        self._quarantine = quarantine
        self._quarantine_retained = False
        self._registration = registration
        self._tensor = tensor

    @property
    def copy_executor(self) -> PackedCopyExecutor:
        """Return the lazily constructed executor bound to this exact tensor.

        :returns: Arena-owned copy executor.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("packed staging arena is closed")
            executor = self._copy_executor
            if executor is None:
                tensor = self._require_tensor_locked()
                executor = PackedCopyExecutor(
                    gpu_id=self._gpu_id,
                    scatter_buffer=tensor,
                )
                self._copy_executor = executor
            return executor

    @property
    def protocol(self) -> PackedDecodeProtocol:
        """Return the decode protocol sharing this arena's allocator.

        :returns: Arena-bound decode protocol.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("packed staging arena is closed")
            return self._protocol

    def capability(
        self,
        *,
        request_generation: bytes,
        topology: PackedTopology,
        visibility_policy: PackedDestinationVisibilityPolicy,
    ) -> PackedDestinationCapability:
        """Issue request-scoped authority for this exact registration.

        :param request_generation: Bootstrap request generation.
        :param topology: Exact transfer topology.
        :param visibility_policy: Decode-selected path and CUDA attribute policy.
        :returns: Immutable destination capability advertised to source.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("packed staging arena is closed")
            if topology.alignment_bytes != self._allocator.alignment_bytes:
                raise ValueError("topology alignment differs from packed arena")
            if type(visibility_policy) is not PackedDestinationVisibilityPolicy:
                raise TypeError(
                    "visibility_policy must be PackedDestinationVisibilityPolicy"
                )
            tensor = self._require_tensor_locked()
            return PackedDestinationCapability(
                route=PackedDestinationRouteBinding(
                    peer=self._peer,
                    arena_generation=self._arena_generation,
                    destination_gpu_id=self._gpu_id,
                    topology=topology,
                    visibility_policy_digest=visibility_policy.digest,
                    base_address=tensor.data_ptr(),
                    total_size=tensor.numel(),
                    alignment_bytes=self._allocator.alignment_bytes,
                ),
                request_generation=request_generation,
            )

    def close(self) -> None:
        """Idempotently close only after every arena owner is terminal."""

        with self._lock:
            if self._closed:
                return
            tensor = self._require_tensor_locked()
            registration = self._require_registration_locked()
            executor = self._copy_executor
            if self._allocator.live_lease_count != 0:
                reason = "packed staging arena still owns live leases"
                owners: tuple[object, ...] = (executor,) if executor is not None else ()
                self._retain_locked(
                    tensor,
                    registration,
                    reason,
                    owners=owners,
                )
                self._closed = True
                raise RuntimeError(reason)
            if executor is not None:
                try:
                    executor.synchronize_scatter()
                except Exception:
                    reason = "packed staging arena scatter stream did not quiesce"
                    logger.error("%s:\n%s", reason, traceback.format_exc())
                    self._retain_locked(
                        tensor,
                        registration,
                        reason,
                        owners=(executor,),
                    )
                    self._closed = True
                    raise
            try:
                self._agent.deregister_memory(registration)
            except Exception:
                reason = "packed staging arena deregistration failed"
                logger.error("%s:\n%s", reason, traceback.format_exc())
                owners: tuple[object, ...] = (executor,) if executor is not None else ()
                self._retain_locked(
                    tensor,
                    registration,
                    reason,
                    owners=owners,
                )
                self._closed = True
                raise
            self._registration = None
            self._tensor = None
            self._closed = True

    def _retain_locked(
        self,
        tensor: torch.Tensor,
        registration: object,
        reason: str,
        *,
        owners: tuple[object, ...],
    ) -> None:
        """Strongly retain arena resources exactly once.

        :param tensor: Registered arena storage.
        :param registration: Opaque NIXL registration.
        :param reason: Stable quarantine reason.
        :param owners: Additional asynchronous owners.
        """

        if self._quarantine_retained:
            return
        self._quarantine.retain(
            tensor,
            registration,
            reason,
            owners=(self._agent, self._allocator, self._protocol, *owners),
        )
        self._quarantine_retained = True

    def _require_registration_locked(self) -> object:
        """Return arena registration while its lock is held.

        :returns: Opaque registration handle.
        """

        registration = self._registration
        if registration is None:
            raise RuntimeError("packed staging arena registration is closed")
        return registration

    def _require_tensor_locked(self) -> torch.Tensor:
        """Return arena storage while its lock is held.

        :returns: Retained arena tensor.
        """

        tensor = self._tensor
        if tensor is None:
            raise RuntimeError("packed staging arena storage is closed")
        return tensor
