import dataclasses
import enum
import logging
import threading
import traceback
from collections.abc import Callable
from typing import TypeVar

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.common.packed_staging_wire import PackedWireMessage
from sglang.srt.disaggregation.common.staging_layout import StagingWriterId
from sglang.srt.disaggregation.nixl.packed_runtime import (
    PackedControlSender,
    PackedDecodeOwnerInventory,
    PackedDecodeOwnerSignal,
    PackedDecodeRuntime,
    PackedDecodeScatterCompletionProducer,
)
from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
    PackedRequestPublication,
)
from sglang.srt.disaggregation.terminal_progress.coordinator import (
    TerminalRequestCoordinator,
    TerminalRequestCoordinatorDisposition,
    TerminalRequestCoordinatorEmission,
    TerminalRequestCoordinatorError,
    TerminalRequestCoordinatorManifest,
    TerminalRequestCoordinationTiming,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.evidence import (
    TerminalProgressTimingRecorder,
    terminal_progress_timing_recorder,
)
from sglang.srt.disaggregation.terminal_progress.decode_scheduler_consumer import (
    PackedTerminalDecodeSchedulerInventory,
    PackedTerminalDecodeSchedulerRegistration,
    PackedTerminalDecodeServingComposition,
)
from sglang.srt.disaggregation.terminal_progress.decode_wiring import (
    PackedTerminalDecodeWiring,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerOutput,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerTimingField,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceiptKind,
    TerminalReceiptOutcome,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeSnapshot,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerServing,
    TerminalSchedulerServingInventory,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourcePlan,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    TerminalWireReceipt,
    TerminalWireReceiptImportNamespace,
    TerminalWireReceiptIssuer,
)

logger = logging.getLogger(__name__)

LaunchResultT = TypeVar("LaunchResultT")
BoundLaunchResultT = TypeVar("BoundLaunchResultT")


class PackedTerminalDecodeDeliveryTarget(enum.StrEnum):
    """Execution context receiving one point-to-point terminal receipt."""

    COORDINATOR = "coordinator"
    OWNER = "owner"


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeWireDelivery:
    """One immutable point-to-point decode terminal delivery.

    Transport authentication is deliberately absent from this value. The
    receiver must join it with the process identity proved by the route before
    calling the corresponding composition ingress.

    :ivar target: Coordinator fan-in or lifecycle-owner fan-out context.
    :ivar recipient: Exact process which must receive the wire value.
    :ivar receipt: Owner or coordinator minted terminal authority.
    """

    target: PackedTerminalDecodeDeliveryTarget
    recipient: TerminalProcessIdentity
    receipt: TerminalWireReceipt

    def __post_init__(self) -> None:
        """Validate one unambiguous transport dispatch."""

        if type(self.target) is not PackedTerminalDecodeDeliveryTarget:
            raise TypeError("target must be PackedTerminalDecodeDeliveryTarget")
        if type(self.recipient) is not TerminalProcessIdentity:
            raise TypeError("recipient must be TerminalProcessIdentity")
        if type(self.receipt) is not TerminalWireReceipt:
            raise TypeError("receipt must be TerminalWireReceipt")
        receipt = self.receipt
        if self.target is PackedTerminalDecodeDeliveryTarget.COORDINATOR:
            if (
                self.recipient.role is not TerminalOwnerRole.DECODE
                or self.recipient.tp_rank != 0
            ):
                raise ValueError("coordinator delivery requires decode rank zero")
            if receipt.binding.owner.role is not TerminalOwnerRole.DECODE:
                raise ValueError("coordinator receipt requires a decode binding")
            if receipt.binding.owner.tp_size != self.recipient.tp_size:
                raise ValueError("coordinator delivery crosses decode TP groups")
            if receipt.issuer != receipt.binding.owner:
                raise ValueError("coordinator receipt must be issued by its rank")
            if receipt.kind not in (
                TerminalReceiptKind.LOCAL_DECODE_READY,
                TerminalReceiptKind.FAILURE,
            ):
                raise ValueError("coordinator delivery carries another receipt kind")
            return
        if self.recipient != receipt.binding.owner:
            raise ValueError("owner delivery targets another process")
        if (
            receipt.issuer.role is not TerminalOwnerRole.DECODE
            or receipt.issuer.tp_rank != 0
        ):
            raise ValueError("request terminality requires a decode coordinator")
        if receipt.kind not in (
            TerminalReceiptKind.REQUEST_READY,
            TerminalReceiptKind.FAILURE,
        ):
            raise ValueError("owner delivery carries another receipt kind")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeWork:
    """Algorithm-neutral work executed outside the scheduler loop.

    :ivar send_delivery: Send one immutable point-to-point terminal receipt.
    :ivar observe_output: Consume non-gating native output evidence.
    """

    send_delivery: Callable[[PackedTerminalDecodeWireDelivery], None]
    observe_output: Callable[[NativeTerminalOwnerOutput], None]

    def __post_init__(self) -> None:
        """Validate process-lifetime decode work callbacks."""

        if not callable(self.send_delivery):
            raise TypeError("send_delivery must be callable")
        if not callable(self.observe_output):
            raise TypeError("observe_output must be callable")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeServingInventory:
    """Complete decode runtime, actor, scheduler, and coordinator evidence.

    :ivar runtime: Authoritative process-lifetime runtime snapshot.
    :ivar actor: Packed decode actor resource inventory.
    :ivar scheduler_consumer: Scheduler-affine request ownership.
    :ivar scheduler_serving: Qualified scheduler receipt ownership.
    :ivar active_binding_digests: Composition-owned immutable request records.
    :ivar active_coordinator_manifest_digests: Canonical-rank fan-in records.
    :ivar ready_coordinator_count: Successfully completed coordinators.
    :ivar failed_coordinator_count: Failure-terminal coordinators.
    :ivar owner_dead_marked: Whether scheduler failure wake was published.
    :ivar native_producers_retired: Whether native producer contexts joined.
    """

    runtime: NativeTerminalRuntimeSnapshot
    actor: PackedDecodeOwnerInventory
    scheduler_consumer: PackedTerminalDecodeSchedulerInventory
    scheduler_serving: TerminalSchedulerServingInventory
    active_binding_digests: tuple[bytes, ...]
    active_coordinator_manifest_digests: tuple[bytes, ...]
    ready_coordinator_count: int
    failed_coordinator_count: int
    owner_dead_marked: bool
    native_producers_retired: bool

    def __post_init__(self) -> None:
        """Validate one conservative cross-component inventory."""

        if type(self.runtime) is not NativeTerminalRuntimeSnapshot:
            raise TypeError("runtime must be NativeTerminalRuntimeSnapshot")
        if type(self.actor) is not PackedDecodeOwnerInventory:
            raise TypeError("actor must be PackedDecodeOwnerInventory")
        if type(self.scheduler_consumer) is not PackedTerminalDecodeSchedulerInventory:
            raise TypeError(
                "scheduler_consumer must be PackedTerminalDecodeSchedulerInventory"
            )
        if type(self.scheduler_serving) is not TerminalSchedulerServingInventory:
            raise TypeError(
                "scheduler_serving must be TerminalSchedulerServingInventory"
            )
        populations = (
            self.active_binding_digests,
            self.active_coordinator_manifest_digests,
        )
        if any(type(population) is not tuple for population in populations):
            raise TypeError("decode serving identity populations must be tuples")
        if any(
            type(digest) is not bytes or len(digest) != 32
            for population in populations
            for digest in population
        ):
            raise ValueError("decode serving identities must contain 32 bytes")
        if any(tuple(sorted(population)) != population for population in populations):
            raise ValueError("decode serving identities must use digest order")
        counts = (self.ready_coordinator_count, self.failed_coordinator_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("coordinator counts must be non-negative integers")
        if type(self.owner_dead_marked) is not bool:
            raise TypeError("owner_dead_marked must be bool")
        if type(self.native_producers_retired) is not bool:
            raise TypeError("native_producers_retired must be bool")


@dataclasses.dataclass(frozen=True, slots=True)
class _DecodeServingRequest:
    """Immutable request identity retained until native retirement."""

    binding: TerminalRequestBinding
    source_plan: PackedTerminalSourcePlan
    coordinator_manifest_digest: bytes | None


@dataclasses.dataclass(slots=True)
class _DecodeCoordinatorRecord:
    """Request-local coordinator and its immutable manifest."""

    manifest: TerminalRequestCoordinatorManifest
    coordinator: TerminalRequestCoordinator


class PackedTerminalDecodeServing:
    """Compose one decode process around the sole native terminal runtime.

    Runtime queues are drained only from their fd-owning execution contexts.
    Scheduler mutation remains behind the qualified scheduler inbox and its
    launch handoff. Rank consistency uses authenticated point-to-point receipt
    fan-in and fan-out; this class contains no status cadence or collective.
    """

    _actor: PackedDecodeRuntime
    _runtime: NativeTerminalRuntime
    _local_identity: TerminalProcessIdentity
    _wiring: PackedTerminalDecodeWiring
    _decode_composition: PackedTerminalDecodeServingComposition
    _scheduler_serving: TerminalSchedulerServing
    _coordinator_issuer: TerminalWireReceiptIssuer | None
    _coordinator_importers: tuple[TerminalWireReceiptImportNamespace, ...]
    _clock_ns: Callable[[], int]
    _timing: TerminalProgressTimingRecorder
    _work: PackedTerminalDecodeWork
    _retire_native_producers: Callable[[], None]
    _requests: dict[bytes, _DecodeServingRequest]
    _coordinators: dict[PackedRequestKey, _DecodeCoordinatorRecord]
    _ready_coordinator_count: int
    _failed_coordinator_count: int
    _owner_dead_marked: bool
    _native_producers_retired: bool
    _started: bool
    _closed: bool
    _lock: threading.RLock

    def __init__(
        self,
        *,
        actor: PackedDecodeRuntime,
        runtime: NativeTerminalRuntime,
        cuda_completion: PackedDecodeScatterCompletionProducer,
        local_identity: TerminalProcessIdentity,
        coordinator_issuer: TerminalWireReceiptIssuer | None,
        coordinator_importers: tuple[TerminalWireReceiptImportNamespace, ...],
        clock_ns: Callable[[], int],
        physical_capacity: int,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
        work: PackedTerminalDecodeWork,
        retire_native_producers: Callable[[], None],
    ) -> None:
        """Construct a dormant decode serving composition.

        :param actor: Process-lifetime packed decode transaction actor.
        :param runtime: Sole authoritative native lifecycle runtime.
        :param cuda_completion: Direct scatter callback-to-owner producer.
        :param local_identity: Exact decode process owned by this composition.
        :param coordinator_issuer: Rank-zero request-global receipt issuer.
        :param coordinator_importers: Rank-zero local-ready import namespaces.
        :param clock_ns: Local monotonic nanosecond clock.
        :param physical_capacity: Maximum configured in-flight generations.
        :param process_fatal_handler: Scheduler-affine fail-closed handler.
        :param work: Point-to-point delivery and observation callbacks.
        :param retire_native_producers: Native event-channel retirement fence.
        """

        if type(actor) is not PackedDecodeRuntime:
            raise TypeError("actor must be PackedDecodeRuntime")
        if type(runtime) is not NativeTerminalRuntime:
            raise TypeError("runtime must be NativeTerminalRuntime")
        if type(local_identity) is not TerminalProcessIdentity:
            raise TypeError("local_identity must be TerminalProcessIdentity")
        if local_identity.role is not TerminalOwnerRole.DECODE:
            raise ValueError("local_identity must belong to decode")
        if type(coordinator_importers) is not tuple or any(
            type(importer) is not TerminalWireReceiptImportNamespace
            for importer in coordinator_importers
        ):
            raise TypeError("coordinator_importers must contain wire import namespaces")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if type(physical_capacity) is not int or physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        if type(work) is not PackedTerminalDecodeWork:
            raise TypeError("work must be PackedTerminalDecodeWork")
        if not callable(retire_native_producers):
            raise TypeError("retire_native_producers must be callable")
        self._validate_coordinator_authority(
            local_identity,
            coordinator_issuer,
            coordinator_importers,
        )
        wiring = PackedTerminalDecodeWiring(
            actor=actor,
            runtime=runtime,
            cuda_completion=cuda_completion,
            local_identity=local_identity,
            clock_ns=clock_ns,
        )
        decode_composition = PackedTerminalDecodeServingComposition(
            wiring=wiring,
            physical_capacity=physical_capacity,
            process_fatal_handler=process_fatal_handler,
        )
        self._actor = actor
        self._runtime = runtime
        self._local_identity = local_identity
        self._wiring = wiring
        self._decode_composition = decode_composition
        self._scheduler_serving = decode_composition.scheduler_serving
        self._coordinator_issuer = coordinator_issuer
        self._coordinator_importers = coordinator_importers
        self._clock_ns = clock_ns
        self._timing = terminal_progress_timing_recorder(logger, clock_ns)
        self._work = work
        self._retire_native_producers = retire_native_producers
        self._requests = {}
        self._coordinators = {}
        self._ready_coordinator_count = 0
        self._failed_coordinator_count = 0
        self._owner_dead_marked = False
        self._native_producers_retired = False
        self._started = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def wiring(self) -> PackedTerminalDecodeWiring:
        """Return the authenticated decode control and work boundary.

        :returns: Process-owned decode wiring.
        """

        return self._wiring

    @property
    def scheduler_composition(self) -> PackedTerminalDecodeServingComposition:
        """Return the concrete scheduler binding surface.

        :returns: Paired decode consumer and qualified scheduler inbox.
        """

        return self._decode_composition

    @property
    def scheduler_fileno(self) -> int:
        """Return the qualified scheduler receipt descriptor.

        :returns: Scheduler-owned readable wake descriptor.
        """

        return self._scheduler_serving.fileno()

    @property
    def runtime_filenos(self) -> tuple[int, ...]:
        """Return every decode runtime descriptor in stable drain order.

        :returns: Scheduler, coordinator, decode-work, lifecycle, and
            observation descriptors.
        """

        return (
            self._runtime.scheduler_actions.fileno(),
            self._runtime.coordinator_actions.fileno(),
            self._runtime.decode_work_actions.fileno(),
            self._runtime.lifecycle_actions.fileno(),
            self._runtime.observations.fileno(),
        )

    @property
    def next_coordinator_deadline_ns(self) -> int | None:
        """Return the earliest armed request-global timer deadline.

        A process reactor re-arms its timer source after coordinator ingress or
        drain. This value is never sampled on a fixed cadence.

        :returns: Earliest expiration timestamp, otherwise ``None``.
        """

        with self._lock:
            deadlines = tuple(
                deadline
                for record in self._coordinators.values()
                if (deadline := record.coordinator.deadline_expires_ns) is not None
            )
        if len(deadlines) == 0:
            return None
        return min(deadlines)

    def start(self) -> None:
        """Start the sole native runtime exactly once."""

        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("decode serving cannot restart")
            self._runtime.start()
            self._started = True

    def register_request(
        self,
        registration: PackedTerminalDecodeSchedulerRegistration,
        coordinator_manifest: TerminalRequestCoordinatorManifest | None,
    ) -> None:
        """Retain coordinator and scheduler identity before publication.

        Canonical rank zero receives the complete manifest. Other ranks carry
        no shadow coordinator state and forward their local-ready receipt to
        the canonical process.

        :param registration: Exact scheduler and packed-actor ownership.
        :param coordinator_manifest: Rank-zero request-global membership.
        """

        self._require_open()
        if type(registration) is not PackedTerminalDecodeSchedulerRegistration:
            raise TypeError(
                "registration must be PackedTerminalDecodeSchedulerRegistration"
            )
        binding = registration.binding
        if binding.owner != self._local_identity:
            raise ValueError("decode registration belongs to another process")
        self._validate_request_manifest(registration, coordinator_manifest)
        with self._lock:
            if binding.digest in self._requests:
                raise RuntimeError("decode serving request identity was reused")
            if (
                coordinator_manifest is not None
                and coordinator_manifest.request_key in self._coordinators
            ):
                raise RuntimeError("request coordinator identity was reused")

        coordinator: TerminalRequestCoordinator | None = None
        if coordinator_manifest is not None:
            issuer = self._coordinator_issuer
            if issuer is None:
                raise RuntimeError("canonical decode rank has no coordinator issuer")
            coordinator = TerminalRequestCoordinator(
                manifest=coordinator_manifest,
                issuer=issuer,
                importers=self._coordinator_importers,
            )
        try:
            self._decode_composition.register(registration)
        except Exception:
            if coordinator is not None:
                try:
                    coordinator.cancel_unpublished()
                except Exception:  # noqa: BLE001
                    logger.error(
                        "Decode coordinator registration rollback failed:\n%s",
                        traceback.format_exc(),
                    )
            self._component_failed("decode request registration failed")
            raise

        context = _DecodeServingRequest(
            binding=binding,
            source_plan=registration.source_plan,
            coordinator_manifest_digest=(
                None if coordinator_manifest is None else coordinator_manifest.digest
            ),
        )
        with self._lock:
            self._requests[binding.digest] = context
            if coordinator_manifest is not None and coordinator is not None:
                self._coordinators[coordinator_manifest.request_key] = (
                    _DecodeCoordinatorRecord(
                        manifest=coordinator_manifest,
                        coordinator=coordinator,
                    )
                )

    def cancel_unpublished(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> None:
        """Cancel an unexposed allocation and its dormant coordinator.

        :param binding: Exact local unpublished lifecycle identity.
        :param reason: Stable scheduler cancellation evidence.
        """

        self._require_open()
        context = self._request_context(binding.digest)
        if context.binding != binding:
            raise RuntimeError("decode cancellation binding changed")
        try:
            self._decode_composition.cancel_unpublished(binding, reason)
            self._cancel_request_coordinator(context)
        except Exception:
            self._component_failed("decode unpublished cancellation failed")
            raise

    def allocation_published(
        self,
        transaction: PackedDecodeRequestTransaction,
        publication: PackedRequestPublication,
        routes: tuple[PackedControlSender, ...],
    ) -> None:
        """Publish the post-forward allocation through the bound owner path.

        :param transaction: Exact scheduler-retained packed transaction.
        :param publication: Matching irreversible allocation publication.
        :param routes: Complete authenticated writer routes.
        """

        self._require_open()
        try:
            self._wiring.allocation_published(transaction, publication, routes)
        except Exception:
            self._component_failed("decode allocation publication failed")
            raise

    def control_received(
        self,
        authenticated_writer_id: StagingWriterId,
        message: PackedWireMessage,
    ) -> tuple[PackedDecodeOwnerSignal, ...]:
        """Deliver authenticated packed control without scheduler polling.

        :param authenticated_writer_id: Writer proved by the control route.
        :param message: Validated packed control payload.
        :returns: Exact native transitions submitted for this message.
        """

        self._require_open()
        try:
            return self._wiring.control_received(authenticated_writer_id, message)
        except Exception:
            self._component_failed("decode control ingress failed")
            raise

    def coordinator_receipt_received(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Deliver one authenticated destination-rank receipt to rank zero.

        :param wire_receipt: Local-ready or failure authority.
        :param authenticated_issuer: Decode rank proved by the control route.
        """

        self._require_open()
        if self._local_identity.tp_rank != 0:
            raise RuntimeError("only decode rank zero owns request coordination")
        try:
            self._accept_coordinator_receipt(wire_receipt, authenticated_issuer)
        except Exception:
            self._component_failed("decode coordinator ingress failed")
            raise

    def request_terminal_received(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Deliver request-global readiness or failure to this decode owner.

        :param wire_receipt: Coordinator-minted owner receipt.
        :param authenticated_issuer: Coordinator proved by the control route.
        """

        self._require_open()
        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        binding = wire_receipt.binding
        context = self._request_context(binding.digest)
        if binding != context.binding or binding.owner != self._local_identity:
            raise RuntimeError("request terminality targets another decode owner")
        try:
            if (
                wire_receipt.kind is TerminalReceiptKind.REQUEST_READY
                and wire_receipt.outcome is TerminalReceiptOutcome.SUCCESS
            ):
                self._wiring.request_ready(
                    binding_digest=binding.digest,
                    wire_receipt=wire_receipt,
                    authenticated_issuer=authenticated_issuer,
                )
                return
            if (
                wire_receipt.kind is TerminalReceiptKind.FAILURE
                and wire_receipt.outcome is TerminalReceiptOutcome.FAILURE
            ):
                self._wiring.request_failed(
                    binding_digest=binding.digest,
                    wire_receipt=wire_receipt,
                    authenticated_issuer=authenticated_issuer,
                    reason="request-global coordination failed",
                )
                return
            raise RuntimeError("decode owner received another terminal receipt kind")
        except Exception:
            self._component_failed("decode request-terminal ingress failed")
            raise

    def expire_coordinators(self, now_ns: int) -> int:
        """Consume one reactor timer wake for all deadlines due at ``now_ns``.

        :param now_ns: Exact monotonic timestamp supplied by the timer reactor.
        :returns: Number of coordinators which became terminal.
        """

        self._require_open()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        if self._local_identity.tp_rank != 0:
            raise RuntimeError("only decode rank zero owns coordinator deadlines")
        with self._lock:
            records = tuple(self._coordinators.values())
        terminal_count = 0
        try:
            for record in records:
                result = record.coordinator.expire(now_ns)
                if not result.newly_terminal:
                    continue
                self._emit_coordination_timing(record, result.timing)
                self._deliver_coordinator_emissions(record, result.emissions)
                terminal_count += 1
        except Exception:
            self._component_failed("decode coordinator deadline handling failed")
            raise
        return terminal_count

    def drain_scheduler_actions(self) -> int:
        """Move native adoption authority into the qualified scheduler inbox.

        :returns: Number of scheduler actions accepted or aborted.
        """

        self._require_open()
        actions = self._runtime.scheduler_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                self._scheduler_serving.publish_action(action)
            except Exception as error:
                reason = (
                    "decode scheduler action publication failed: "
                    f"{type(error).__name__}: {error}"
                )
                try:
                    self._runtime.fail_scheduler_action(action, reason)
                except Exception:  # noqa: BLE001
                    logger.error(
                        "Decode scheduler failure publication also failed:\n%s",
                        traceback.format_exc(),
                    )
                self._component_failed(reason)
                raise
        self._propagate_runtime_fatal()
        return len(actions)

    def drain_coordinator_actions(self) -> int:
        """Consume owner-minted local-ready actions without a collective.

        :returns: Number of coordinator actions accepted or aborted.
        """

        self._require_open()
        actions = self._runtime.coordinator_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                self._consume_local_ready_action(action)
                self._runtime.acknowledge_consumed_action(action)
            except Exception:
                self._component_failed("decode local-ready dispatch failed")
                raise
        self._propagate_runtime_fatal()
        return len(actions)

    def drain_decode_work_actions(self) -> int:
        """Execute scatter and teardown continuations off the scheduler loop.

        :returns: Number of decode work actions consumed or aborted.
        """

        self._require_open()
        actions = self._runtime.decode_work_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                if action.kind is NativeTerminalOwnerActionKind.DECODE_SCATTER_READY:
                    self._wiring.consume_scatter_action(action)
                elif action.kind is NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY:
                    self._wiring.consume_teardown_action(action)
                else:
                    raise RuntimeError("decode work inbox carried another action kind")
            except Exception:
                self._component_failed("decode work action failed")
                raise
        self._propagate_runtime_fatal()
        return len(actions)

    def drain_lifecycle_actions(self) -> int:
        """Consume retirement, quarantine, and process-fatal authority.

        :returns: Number of lifecycle actions accepted or aborted.
        """

        self._require_open()
        actions = self._runtime.lifecycle_actions.drain()
        for action in actions:
            if action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL:
                self._mark_owner_dead()
                if self._aborting:
                    self._runtime.acknowledge_aborted_action(action)
                else:
                    self._runtime.acknowledge_consumed_action(action)
                continue
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            if action.kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED:
                try:
                    self._retire_request(action)
                except Exception:
                    self._component_failed("decode request retirement failed")
                    raise
                continue
            if action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED:
                self._runtime.acknowledge_consumed_action(action)
                try:
                    self._mark_owner_dead()
                finally:
                    self._runtime.begin_abort()
                continue
            raise RuntimeError("decode lifecycle inbox carried another action kind")
        self._propagate_runtime_fatal()
        return len(actions)

    def drain_observations(self) -> int:
        """Project non-gating native evidence without functional backpressure.

        :returns: Number of observations removed from the bounded inbox.
        """

        self._require_open()
        observations = self._runtime.observations.drain()
        for output in observations:
            try:
                self._wiring.observe_native_output(output)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Decode terminal timing observation failed without gating "
                    "progress:\n%s",
                    traceback.format_exc(),
                )
            try:
                self._work.observe_output(output)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Decode terminal observation callback failed without gating "
                    "progress:\n%s",
                    traceback.format_exc(),
                )
        self._propagate_runtime_fatal()
        return len(observations)

    def drain_runtime_actions(self) -> int:
        """Drain all currently readable runtime contexts once.

        :returns: Total immutable actions and observations consumed.
        """

        consumed = 0
        consumed += self.drain_scheduler_actions()
        consumed += self.drain_coordinator_actions()
        consumed += self.drain_decode_work_actions()
        consumed += self.drain_lifecycle_actions()
        consumed += self.drain_observations()
        return consumed

    def drain_scheduler_at_loop_entry(
        self,
    ) -> tuple[NativeTerminalOwnerAction, ...]:
        """Consume qualified adoption authority on the scheduler thread.

        :returns: Exact adoption actions successfully consumed.
        """

        self._require_open()
        try:
            return self._scheduler_serving.drain_at_loop_entry()
        except Exception:
            self._component_failed("decode scheduler loop-entry drain failed")
            raise

    def launch_handoff(self, submit: Callable[[], LaunchResultT]) -> LaunchResultT:
        """Order owner receipts against one host forward submission.

        :param submit: Narrow model-worker submission callback.
        :returns: Exact launch result.
        """

        self._require_open()
        try:
            return self._scheduler_serving.launch_handoff(submit)
        except Exception:
            self._component_failed("decode scheduler launch handoff failed")
            raise

    def launch_and_bind_handoff(
        self,
        submit: Callable[[], LaunchResultT],
        bind: Callable[[LaunchResultT], BoundLaunchResultT],
    ) -> BoundLaunchResultT:
        """Submit and bind post-forward ownership inside the launch gate.

        :param submit: Narrow model-worker submission callback.
        :param bind: Immediate immutable post-forward binding callback.
        :returns: Exact value returned by ``bind``.
        """

        self._require_open()
        try:
            return self._scheduler_serving.launch_and_bind_handoff(submit, bind)
        except Exception:
            self._component_failed("decode scheduler launch binding failed")
            raise

    def inventory(self) -> PackedTerminalDecodeServingInventory:
        """Return complete process-lifetime decode ownership evidence.

        :returns: Runtime, actor, scheduler, and coordinator inventory.
        """

        with self._lock:
            active = tuple(sorted(self._requests))
            coordinator_digests = tuple(
                sorted(record.manifest.digest for record in self._coordinators.values())
            )
            ready_count = self._ready_coordinator_count
            failed_count = self._failed_coordinator_count
            owner_dead_marked = self._owner_dead_marked
            native_producers_retired = self._native_producers_retired
        return PackedTerminalDecodeServingInventory(
            runtime=self._runtime.snapshot(),
            actor=self._actor.terminal_owner_inventory(),
            scheduler_consumer=self._decode_composition.consumer.inventory(),
            scheduler_serving=self._scheduler_serving.inventory(),
            active_binding_digests=active,
            active_coordinator_manifest_digests=coordinator_digests,
            ready_coordinator_count=ready_count,
            failed_coordinator_count=failed_count,
            owner_dead_marked=owner_dead_marked,
            native_producers_retired=native_producers_retired,
        )

    def stop_admission_and_retire_producers(self) -> None:
        """Close admission and join every native and Python producer context."""

        self._require_open()
        self._runtime.stop_admission()
        self._retire_all_producers()

    def close_clean(self, timeout_seconds: float) -> None:
        """Close a fully retired decode composition with exact-zero inventory.

        :param timeout_seconds: Hash-bound native projection fence timeout.
        """

        self._require_open()
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        self._scheduler_serving.fence_runtime_teardown(
            self._runtime,
            timeout_seconds,
        )
        self.drain_runtime_actions()
        self.drain_scheduler_at_loop_entry()
        self._scheduler_serving.fence_runtime_teardown(
            self._runtime,
            timeout_seconds,
        )
        self.drain_runtime_actions()
        inventory = self.inventory()
        if self._retains_clean_authority(inventory):
            raise RuntimeError("clean decode serving close retains lifecycle authority")
        self._runtime.close_clean()
        self._scheduler_serving.close()
        with self._lock:
            self._closed = True

    def abort_and_close(self) -> PackedTerminalDecodeServingInventory:
        """Drain fail-closed authority and preserve ambiguous decode resources.

        :returns: Final combined fail-closed inventory before descriptor close.
        """

        self._require_open()
        self._runtime.begin_abort()
        self._retire_all_producers()
        self._mark_owner_dead()
        shutdown_timeout = terminal_deadline_spec(
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
        ).seconds
        if not self._runtime.wait_for_output_projection_quiescence(shutdown_timeout):
            raise RuntimeError("abort retained unrouted decode terminal authority")
        self.drain_runtime_actions()
        self._scheduler_serving.close_fail_closed()
        inventory = self.inventory()
        self._runtime.finish_abort_close()
        with self._lock:
            self._closed = True
        return inventory

    def _consume_local_ready_action(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Deliver one owner-minted local-ready receipt to its coordinator.

        :param action: Exact local-ready action from the native runtime.
        """

        if (
            action.kind is not NativeTerminalOwnerActionKind.LOCAL_DECODE_READY
            or action.receipt is None
        ):
            raise RuntimeError("coordinator inbox carried another action kind")
        context = self._request_context(action.binding.digest)
        wire_receipt = action.receipt.to_wire_receipt()
        coordinator = context.source_plan.request_ready_issuer
        if coordinator == self._local_identity:
            self._accept_coordinator_receipt(wire_receipt, self._local_identity)
            return
        self._work.send_delivery(
            PackedTerminalDecodeWireDelivery(
                target=PackedTerminalDecodeDeliveryTarget.COORDINATOR,
                recipient=coordinator,
                receipt=wire_receipt,
            )
        )

    def _accept_coordinator_receipt(
        self,
        wire_receipt: TerminalWireReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Apply one rank receipt and fan out any newly terminal result.

        :param wire_receipt: Authenticated local-ready or failure authority.
        :param authenticated_issuer: Sender identity proved by the route.
        """

        if type(wire_receipt) is not TerminalWireReceipt:
            raise TypeError("wire_receipt must be TerminalWireReceipt")
        if type(authenticated_issuer) is not TerminalProcessIdentity:
            raise TypeError("authenticated_issuer must be TerminalProcessIdentity")
        with self._lock:
            record = self._coordinators.get(wire_receipt.binding.request_key)
        if record is None:
            raise KeyError("request coordinator is not retained")
        try:
            result = record.coordinator.accept(
                wire_receipt,
                authenticated_issuer,
                self._clock_ns(),
            )
        except TerminalRequestCoordinatorError:
            emissions = record.coordinator.terminal_emissions
            if len(emissions) == 0:
                raise
            self._deliver_coordinator_emissions(record, emissions)
            return
        if result.newly_terminal:
            self._emit_coordination_timing(record, result.timing)
            self._deliver_coordinator_emissions(record, result.emissions)

    def _emit_coordination_timing(
        self,
        record: _DecodeCoordinatorRecord,
        timing: TerminalRequestCoordinationTiming | None,
    ) -> None:
        """Project one newly terminal request-global coordination interval.

        :param record: Exact canonical coordinator and immutable manifest.
        :param timing: Same-process first-receipt-to-terminal interval.
        """

        try:
            if type(timing) is not TerminalRequestCoordinationTiming:
                raise RuntimeError("terminal coordinator timing is absent")
            local_bindings = tuple(
                binding
                for binding in record.manifest.destination_bindings
                if binding.owner == self._local_identity
            )
            if len(local_bindings) != 1:
                raise RuntimeError("coordinator timing has no unique local binding")
            binding = local_bindings[0]
            self._timing.emit_interval(
                binding=binding,
                field=TerminalOwnerTimingField.REQUEST_GLOBAL_COORDINATION,
                sample_key=f"decode-rank-{binding.owner.tp_rank}",
                started_ns=timing.first_local_ready_received_ns,
                completed_ns=timing.terminal_emitted_ns,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode coordination timing projection failed without gating "
                "progress:\n%s",
                traceback.format_exc(),
            )

    def _deliver_coordinator_emissions(
        self,
        record: _DecodeCoordinatorRecord,
        emissions: tuple[TerminalRequestCoordinatorEmission, ...],
    ) -> None:
        """Deliver complete fan-out before releasing coordinator replay state.

        :param record: Exact terminal request coordinator.
        :param emissions: One immutable result per frozen recipient.
        """

        if len(emissions) == 0:
            raise RuntimeError("terminal coordinator emitted no recipients")
        for emission in emissions:
            delivery = PackedTerminalDecodeWireDelivery(
                target=PackedTerminalDecodeDeliveryTarget.OWNER,
                recipient=emission.recipient.owner,
                receipt=emission.receipt.wire_receipt,
            )
            if delivery.recipient == self._local_identity:
                self.request_terminal_received(
                    delivery.receipt,
                    delivery.receipt.issuer,
                )
                continue
            self._work.send_delivery(delivery)
        record.coordinator.close()
        disposition = record.coordinator.disposition
        with self._lock:
            current = self._coordinators.get(record.manifest.request_key)
            if current is not record:
                raise RuntimeError(
                    "request coordinator registry changed during fan-out"
                )
            del self._coordinators[record.manifest.request_key]
            if disposition is TerminalRequestCoordinatorDisposition.READY:
                self._ready_coordinator_count += 1
            elif disposition is TerminalRequestCoordinatorDisposition.FAILED:
                self._failed_coordinator_count += 1
            else:
                raise RuntimeError("retired request coordinator is not terminal")

    def _retire_request(self, action: NativeTerminalOwnerAction) -> None:
        """Retire actor and composition identity under native authority.

        :param action: Exact native request-retirement action.
        """

        context = self._request_context(action.binding.digest)
        with self._lock:
            if context.source_plan.request_key in self._coordinators:
                raise RuntimeError("decode retirement preceded coordinator fan-out")
        self._wiring.retire(action)
        with self._lock:
            current = self._requests.get(action.binding.digest)
            if current is not context:
                raise RuntimeError("decode request registry changed during retirement")
            del self._requests[action.binding.digest]

    def _cancel_request_coordinator(self, context: _DecodeServingRequest) -> None:
        """Release a coordinator which never accepted external authority.

        :param context: Exact unpublished request context.
        """

        if context.coordinator_manifest_digest is None:
            return
        with self._lock:
            record = self._coordinators.get(context.source_plan.request_key)
        if record is None:
            raise RuntimeError("unpublished request lost its coordinator")
        if record.manifest.digest != context.coordinator_manifest_digest:
            raise RuntimeError("unpublished coordinator manifest changed")
        record.coordinator.cancel_unpublished()
        with self._lock:
            current = self._coordinators.get(context.source_plan.request_key)
            if current is not record:
                raise RuntimeError("request coordinator changed during cancellation")
            del self._coordinators[context.source_plan.request_key]

    def _request_context(self, binding_digest: bytes) -> _DecodeServingRequest:
        """Resolve one exact immutable serving request.

        :param binding_digest: Exact local lifecycle identity.
        :returns: Retained request context.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        with self._lock:
            context = self._requests.get(binding_digest)
        if context is None:
            raise KeyError("decode serving request is not retained")
        return context

    def _retire_all_producers(self) -> None:
        """Join external contexts and retire every frozen Python authority."""

        with self._lock:
            already_retired = self._native_producers_retired
        if not already_retired:
            self._retire_native_producers()
            with self._lock:
                self._native_producers_retired = True
        snapshot = self._runtime.snapshot()
        if snapshot.producers_joined:
            return
        for producer_id in self._runtime.python_producer_ids:
            self._runtime.retire_python_producer(producer_id)
        self._runtime.join_producers()

    @property
    def _aborting(self) -> bool:
        """Return whether functional side effects must no longer execute.

        :returns: Whether runtime authority is in fail-closed drain.
        """

        return self._runtime.disposition in (
            NativeTerminalRuntimeDisposition.ABORT_DRAINING,
            NativeTerminalRuntimeDisposition.PROCESS_FATAL,
        )

    def _propagate_runtime_fatal(self) -> None:
        """Wake scheduler fail-closed handling after runtime failure."""

        if self._runtime.snapshot().fatal_reason is not None:
            self._mark_owner_dead()

    def _component_failed(self, label: str) -> None:
        """Enter process-fatal drain after a functional component failure.

        :param label: Stable failure boundary label.
        """

        logger.error("%s:\n%s", label, traceback.format_exc())
        try:
            self._mark_owner_dead()
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode scheduler owner-death wake also failed:\n%s",
                traceback.format_exc(),
            )
        try:
            self._runtime.begin_abort()
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode runtime abort entry also failed:\n%s",
                traceback.format_exc(),
            )

    def _mark_owner_dead(self) -> None:
        """Publish one sticky scheduler wake for owner or component death."""

        with self._lock:
            if self._owner_dead_marked:
                return
            self._scheduler_serving.mark_owner_dead()
            self._owner_dead_marked = True

    @staticmethod
    def _retains_clean_authority(
        inventory: PackedTerminalDecodeServingInventory,
    ) -> bool:
        """Return whether any functional or quarantined authority remains.

        :param inventory: Complete decode serving inventory.
        :returns: Whether exact-zero clean close must be rejected.
        """

        runtime = inventory.runtime
        return (
            runtime.scheduler_live_count != 0
            or runtime.scheduler_pending_count != 0
            or runtime.consumer_pending_count != 0
            or len(runtime.quarantined_binding_digests) != 0
            or len(inventory.actor.active_bindings) != 0
            or len(inventory.actor.quarantined_bindings) != 0
            or len(inventory.scheduler_consumer.active_binding_digests) != 0
            or inventory.scheduler_serving.inbox.live_count != 0
            or len(inventory.active_binding_digests) != 0
            or len(inventory.active_coordinator_manifest_digests) != 0
        )

    @staticmethod
    def _validate_coordinator_authority(
        local_identity: TerminalProcessIdentity,
        coordinator_issuer: TerminalWireReceiptIssuer | None,
        coordinator_importers: tuple[TerminalWireReceiptImportNamespace, ...],
    ) -> None:
        """Validate immutable rank-zero coordination authority.

        :param local_identity: Exact process owning the composition.
        :param coordinator_issuer: Optional canonical receipt issuer.
        :param coordinator_importers: Complete destination-rank import roster.
        """

        if local_identity.tp_rank != 0:
            if coordinator_issuer is not None or len(coordinator_importers) != 0:
                raise ValueError("noncanonical decode ranks cannot own coordination")
            return
        if type(coordinator_issuer) is not TerminalWireReceiptIssuer:
            raise TypeError("decode rank zero requires a coordinator issuer")
        if coordinator_issuer.identity != local_identity:
            raise ValueError("coordinator issuer belongs to another process")
        identities = tuple(importer.remote_issuer for importer in coordinator_importers)
        if len(identities) != local_identity.tp_size:
            raise ValueError("coordinator importer count differs from decode TP size")
        if any(
            identity.role is not TerminalOwnerRole.DECODE
            or identity.tp_size != local_identity.tp_size
            for identity in identities
        ):
            raise ValueError("coordinator importers cross decode TP groups")
        if tuple(identity.tp_rank for identity in identities) != tuple(
            range(local_identity.tp_size)
        ):
            raise ValueError("coordinator importers must use canonical rank order")
        if identities[0] != local_identity:
            raise ValueError(
                "coordinator importer rank zero differs from local process"
            )

    def _validate_request_manifest(
        self,
        registration: PackedTerminalDecodeSchedulerRegistration,
        manifest: TerminalRequestCoordinatorManifest | None,
    ) -> None:
        """Validate exact per-request fan-in and fan-out membership.

        :param registration: Local scheduler and source identity plan.
        :param manifest: Canonical rank-zero request membership.
        """

        if self._local_identity.tp_rank != 0:
            if manifest is not None:
                raise ValueError(
                    "noncanonical decode rank cannot register a coordinator"
                )
            return
        if type(manifest) is not TerminalRequestCoordinatorManifest:
            raise TypeError("decode rank zero requires a coordinator manifest")
        binding = registration.binding
        source_plan = registration.source_plan
        if manifest.request_key != binding.request_key:
            raise ValueError("coordinator manifest belongs to another request")
        if manifest.destination_bindings[0].owner != self._local_identity:
            raise ValueError("coordinator manifest belongs to another canonical rank")
        if manifest.destination_bindings[0] != binding:
            raise ValueError("coordinator local binding differs from registration")
        if source_plan.request_ready_issuer != self._local_identity:
            raise ValueError("source plan names another request coordinator")
        expected_recipients = (
            *manifest.destination_bindings,
            *source_plan.source_bindings,
        )
        if manifest.recipient_bindings != expected_recipients:
            raise ValueError("coordinator fan-out differs from decode and source plans")

    def _require_open(self) -> None:
        """Require one started, non-closed decode serving composition."""

        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("decode serving is not open")
