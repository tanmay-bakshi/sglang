import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.cuda_owner_producer import (
    TerminalCudaCompletionProducer,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    TerminalDeadlineKind,
    terminal_deadline_spec,
)
from sglang.srt.disaggregation.terminal_progress.grouped_nixl_owner import (
    GroupedNixlTerminalOwner,
    GroupedNixlTerminalOwnerInventory,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerObservation,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    TerminalGatewayPublicationResult,
)
from sglang.srt.disaggregation.terminal_progress.receipts import TerminalReceipt
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalObservation,
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
    TerminalSchedulerServingRole,
)
from sglang.srt.disaggregation.terminal_progress.source_gather_worker import (
    PackedTerminalSourceGatherWorker,
    PackedTerminalSourceGatherWorkerInventory,
)
from sglang.srt.disaggregation.terminal_progress.source_scheduler_consumer import (
    PackedTerminalSourceSchedulerConsumer,
    PackedTerminalSourceSchedulerInventory,
    PackedTerminalSourceSchedulerRelease,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceCancellationDisposition,
    PackedTerminalSourceInventory,
    PackedTerminalSourceMetricsSink,
    PackedTerminalSourcePublisher,
    PackedTerminalSourceSubmission,
    PackedTerminalSourceWiring,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceResourceInventory:
    """Exact actor, DFlash, and pre-lifecycle quarantine ownership.

    :ivar actor_active_binding_digests: Live packed source actor identities.
    :ivar actor_quarantined_binding_digests: Actor identities held fail closed.
    :ivar actor_waiting_for_ready_binding_digests: PREPARE state awaiting READY.
    :ivar actor_main_handle_binding_digests: Live main NIXL handles.
    :ivar actor_auxiliary_handle_binding_digests: Live auxiliary NIXL handles.
    :ivar actor_lane_binding_digests: Live packed transfer lanes.
    :ivar request_ready_import_binding_digests: Live decoder receipt replay routes.
    :ivar publication_control_active_binding_digests: Live publisher routes.
    :ivar publication_control_terminal_binding_digests: Routes retaining one
        terminal publisher outcome.
    :ivar source_transfer_info_room_ids: Bootstrap transfer metadata rooms.
    :ivar source_prefix_length_room_ids: Decoder prefix-length metadata rooms.
    :ivar source_prefetched_room_ids: Staging rooms retained as prefetched.
    :ivar source_prefetch_requested_room_ids: Staging prefetch request rooms.
    :ivar dflash_active_transfer_count: Retained DFlash transfer identities.
    :ivar dflash_posted_transfer_count: DFlash transfers awaiting terminality.
    :ivar dflash_settled_transfer_count: DFlash transfers retained through ACK.
    :ivar dflash_released_transfer_count: Cumulative clean transfer releases.
    :ivar dflash_quarantined_transfer_count: Ambiguous DFlash transfers.
    :ivar dflash_unowned_native_handle_count: Handles retained after arm failure.
    :ivar dflash_free_row_count: Reusable device-side boundary rows.
    :ivar dflash_active_row_count: Leased device-side boundary rows.
    :ivar dflash_quarantined_row_count: Permanently non-reusable boundary rows.
    :ivar unpublished_quarantined_binding_digests: CUDA-touched submissions which
        failed before native lifecycle publication, including result slots.
    :ivar unpublished_quarantined_result_slot_binding_digests: Canonical pinned
        result slots retained with pre-lifecycle submissions.
    """

    actor_active_binding_digests: tuple[bytes, ...]
    actor_quarantined_binding_digests: tuple[bytes, ...]
    actor_waiting_for_ready_binding_digests: tuple[bytes, ...]
    actor_main_handle_binding_digests: tuple[bytes, ...]
    actor_auxiliary_handle_binding_digests: tuple[bytes, ...]
    actor_lane_binding_digests: tuple[bytes, ...]
    request_ready_import_binding_digests: tuple[bytes, ...]
    publication_control_active_binding_digests: tuple[bytes, ...]
    publication_control_terminal_binding_digests: tuple[bytes, ...]
    source_transfer_info_room_ids: tuple[int, ...]
    source_prefix_length_room_ids: tuple[int, ...]
    source_prefetched_room_ids: tuple[int, ...]
    source_prefetch_requested_room_ids: tuple[int, ...]
    dflash_active_transfer_count: int
    dflash_posted_transfer_count: int
    dflash_settled_transfer_count: int
    dflash_released_transfer_count: int
    dflash_quarantined_transfer_count: int
    dflash_unowned_native_handle_count: int
    dflash_free_row_count: int
    dflash_active_row_count: int
    dflash_quarantined_row_count: int
    unpublished_quarantined_binding_digests: tuple[bytes, ...]
    unpublished_quarantined_result_slot_binding_digests: tuple[bytes, ...]

    def __post_init__(self) -> None:
        """Validate conservation-complete source resource evidence."""

        binding_collections = (
            self.actor_active_binding_digests,
            self.actor_quarantined_binding_digests,
            self.actor_waiting_for_ready_binding_digests,
            self.actor_main_handle_binding_digests,
            self.actor_auxiliary_handle_binding_digests,
            self.actor_lane_binding_digests,
            self.request_ready_import_binding_digests,
            self.publication_control_active_binding_digests,
            self.publication_control_terminal_binding_digests,
            self.unpublished_quarantined_binding_digests,
            self.unpublished_quarantined_result_slot_binding_digests,
        )
        if any(type(values) is not tuple for values in binding_collections):
            raise TypeError("source resource binding collections must be tuples")
        if any(
            type(value) is not bytes or len(value) != 32
            for values in binding_collections
            for value in values
        ):
            raise ValueError("source resource bindings must contain 32 bytes")
        if any(values != tuple(sorted(values)) for values in binding_collections):
            raise ValueError("source resource bindings must use digest order")
        actor_active = set(self.actor_active_binding_digests)
        if any(
            not set(values).issubset(actor_active)
            for values in binding_collections[1:6]
        ):
            raise ValueError("actor resource bindings must remain actor-active")
        if not set(self.publication_control_terminal_binding_digests).issubset(
            self.publication_control_active_binding_digests
        ):
            raise ValueError("terminal publisher routes must remain active")
        if not set(self.unpublished_quarantined_result_slot_binding_digests).issubset(
            self.unpublished_quarantined_binding_digests
        ):
            raise ValueError(
                "unpublished result slots must remain submission-quarantined"
            )
        room_collections = (
            self.source_transfer_info_room_ids,
            self.source_prefix_length_room_ids,
            self.source_prefetched_room_ids,
            self.source_prefetch_requested_room_ids,
        )
        if any(type(values) is not tuple for values in room_collections):
            raise TypeError("source room collections must be tuples")
        if any(
            type(value) is not int or value < 0
            for values in room_collections
            for value in values
        ):
            raise ValueError("source rooms must be non-negative integers")
        if any(values != tuple(sorted(set(values))) for values in room_collections):
            raise ValueError("source room collections must be sorted and unique")
        counts = (
            self.dflash_active_transfer_count,
            self.dflash_posted_transfer_count,
            self.dflash_settled_transfer_count,
            self.dflash_released_transfer_count,
            self.dflash_quarantined_transfer_count,
            self.dflash_unowned_native_handle_count,
            self.dflash_free_row_count,
            self.dflash_active_row_count,
            self.dflash_quarantined_row_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("source resource counts must be non-negative integers")
        if self.dflash_active_transfer_count != (
            self.dflash_posted_transfer_count
            + self.dflash_settled_transfer_count
            + self.dflash_quarantined_transfer_count
        ):
            raise ValueError("DFlash transfer inventory does not conserve")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceWork:
    """Algorithm-neutral source work earned by native lifecycle actions.

    :ivar post_gather: Post the immutable gather and transport submission.
    :ivar send_outcomes: Send immutable writer outcome messages.
    :ivar send_ack: Send one exact-generation teardown acknowledgement.
    :ivar quarantine: Retain actor and transport resources under failure authority.
    :ivar observe_output: Consume non-gating native output evidence.
    """

    post_gather: Callable[
        [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
    ]
    send_outcomes: Callable[
        [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
    ]
    send_ack: Callable[
        [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
    ]
    quarantine: Callable[[NativeTerminalOwnerAction], None]
    observe_output: Callable[[NativeTerminalObservation], None]

    def __post_init__(self) -> None:
        """Validate every process-lifetime source work boundary."""

        callbacks = (
            self.post_gather,
            self.send_outcomes,
            self.send_ack,
            self.quarantine,
            self.observe_output,
        )
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("source work callbacks must be callable")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceServingInventory:
    """Combined source runtime, wiring, and scheduler ownership evidence.

    :ivar runtime: Authoritative process-lifetime runtime snapshot.
    :ivar wiring: Immutable source side-effect inventory.
    :ivar scheduler_consumer: Scheduler-affine resource inventory.
    :ivar scheduler_serving: Qualified scheduler receipt inventory.
    :ivar gather_worker: Dedicated blocking source gather execution context.
    :ivar grouped_nixl: Request-level main and DFlash transfer inventory.
    :ivar resources: Packed actor, DFlash, and pre-lifecycle quarantine inventory.
    :ivar owner_dead_marked: Whether scheduler failure wake was published.
    :ivar native_producers_retired: Whether native producer contexts joined.
    """

    runtime: NativeTerminalRuntimeSnapshot
    wiring: PackedTerminalSourceInventory
    scheduler_consumer: PackedTerminalSourceSchedulerInventory
    scheduler_serving: TerminalSchedulerServingInventory
    gather_worker: PackedTerminalSourceGatherWorkerInventory
    grouped_nixl: GroupedNixlTerminalOwnerInventory
    resources: PackedTerminalSourceResourceInventory
    owner_dead_marked: bool
    native_producers_retired: bool

    def __post_init__(self) -> None:
        """Validate one exact cross-component serving inventory."""

        if type(self.runtime) is not NativeTerminalRuntimeSnapshot:
            raise TypeError("runtime must be NativeTerminalRuntimeSnapshot")
        if type(self.wiring) is not PackedTerminalSourceInventory:
            raise TypeError("wiring must be PackedTerminalSourceInventory")
        if type(self.scheduler_consumer) is not PackedTerminalSourceSchedulerInventory:
            raise TypeError(
                "scheduler_consumer must be PackedTerminalSourceSchedulerInventory"
            )
        if type(self.scheduler_serving) is not TerminalSchedulerServingInventory:
            raise TypeError(
                "scheduler_serving must be TerminalSchedulerServingInventory"
            )
        if type(self.gather_worker) is not PackedTerminalSourceGatherWorkerInventory:
            raise TypeError(
                "gather_worker must be PackedTerminalSourceGatherWorkerInventory"
            )
        if type(self.grouped_nixl) is not GroupedNixlTerminalOwnerInventory:
            raise TypeError("grouped_nixl must be GroupedNixlTerminalOwnerInventory")
        if type(self.resources) is not PackedTerminalSourceResourceInventory:
            raise TypeError("resources must be PackedTerminalSourceResourceInventory")
        if type(self.owner_dead_marked) is not bool:
            raise TypeError("owner_dead_marked must be bool")
        if type(self.native_producers_retired) is not bool:
            raise TypeError("native_producers_retired must be bool")

    @property
    def retained_resource_count(self) -> int:
        """Count every resource population consulted by clean closure.

        The count is intentionally conservative and may count one request in
        several ownership domains. Its contract is exact at zero: a zero value
        is equivalent to the absence of all clean-close retention authority.

        :returns: Conservative retained resource population.
        """

        runtime = self.runtime
        owner = runtime.owner
        native = self.grouped_nixl.native
        backend_lifecycle = native.backend_lifecycle
        runtime_inboxes = (
            runtime.scheduler,
            runtime.coordinator,
            runtime.lifecycle,
            runtime.source_gather,
            runtime.source_work,
            runtime.decode_scatter,
            runtime.decode_work,
            runtime.publisher,
        )
        return sum(
            (
                owner.queued_input_count,
                owner.queued_output_count,
                owner.queued_fatal_output_count,
                owner.pending_action_count,
                owner.active_source_count,
                owner.active_decode_count,
                owner.quarantined_count,
                owner.armed_deadline_count,
                int(owner.output_drain_active),
                runtime.scheduler_live_count,
                runtime.scheduler_pending_count,
                runtime.consumer_pending_count,
                len(runtime.quarantined_binding_digests),
                int(runtime.fatal_reason is not None),
                sum(inbox.queued_count for inbox in runtime_inboxes),
                len(self.wiring.active_binding_digests),
                len(self.wiring.active_result_slot_binding_digests),
                self.wiring.pending_publication_action_count,
                self.wiring.pending_publication_receipt_count,
                len(self.scheduler_consumer.active_binding_digests),
                self.scheduler_serving.inbox.live_count,
                len(self.scheduler_serving.retained_action_ids),
                self.gather_worker.retained_action_count,
                int(self.gather_worker.fatal_reason is not None),
                self.grouped_nixl.active_group_count,
                self.grouped_nixl.active_transfer_count,
                self.grouped_nixl.quarantined_transfer_count,
                self.grouped_nixl.unowned_handle_count,
                native.queued_channel_events,
                native.active_channel_subscriptions,
                native.retained_public_subscriptions,
                native.active_callback_slots,
                native.queued_owner_continuations,
                backend_lifecycle.source_deliveries_outstanding,
                backend_lifecycle.source_local_pending,
                backend_lifecycle.source_receipt_pending,
                backend_lifecycle.destination_pending,
                backend_lifecycle.destination_admitting,
                backend_lifecycle.destination_committed,
                backend_lifecycle.destination_replaying,
                backend_lifecycle.destination_quarantined,
                backend_lifecycle.active_native_deadlines,
                len(backend_lifecycle.source_deliveries),
                len(backend_lifecycle.destination_deliveries),
                len(backend_lifecycle.native_deadlines),
                int(native.fatal != 0),
                int(native.eventfd_error != 0),
                len(self.resources.actor_active_binding_digests),
                len(self.resources.request_ready_import_binding_digests),
                len(self.resources.publication_control_active_binding_digests),
                len(self.resources.publication_control_terminal_binding_digests),
                len(self.resources.source_transfer_info_room_ids),
                len(self.resources.source_prefix_length_room_ids),
                len(self.resources.source_prefetched_room_ids),
                len(self.resources.source_prefetch_requested_room_ids),
                self.resources.dflash_active_transfer_count,
                self.resources.dflash_quarantined_transfer_count,
                self.resources.dflash_unowned_native_handle_count,
                self.resources.dflash_active_row_count,
                self.resources.dflash_quarantined_row_count,
                len(self.resources.unpublished_quarantined_binding_digests),
                len(self.resources.unpublished_quarantined_result_slot_binding_digests),
            )
        )


class PackedTerminalSourceServing:
    """Compose one source process around the sole native terminal runtime.

    Runtime inboxes are deliberately exposed through explicit drain methods.
    The process reactor registers their file descriptors and invokes these
    methods when readable. Scheduler mutation still occurs only through
    :meth:`drain_scheduler_at_loop_entry` on the scheduler thread.
    """

    _runtime: NativeTerminalRuntime
    _wiring: PackedTerminalSourceWiring
    _scheduler_consumer: PackedTerminalSourceSchedulerConsumer
    _scheduler_serving: TerminalSchedulerServing
    _gather_worker: PackedTerminalSourceGatherWorker
    _grouped_nixl: GroupedNixlTerminalOwner
    _work: PackedTerminalSourceWork
    _retire_native_producers: Callable[[], None]
    _resource_inventory: Callable[[], PackedTerminalSourceResourceInventory]
    _retire_submission: Callable[
        [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
    ]
    _owner_dead_marked: bool
    _native_producers_retired: bool
    _started: bool
    _closed: bool
    _lock: threading.Lock

    def __init__(
        self,
        *,
        runtime: NativeTerminalRuntime,
        cuda_completion: TerminalCudaCompletionProducer,
        local_identity: TerminalProcessIdentity,
        publisher: PackedTerminalSourcePublisher | None,
        metrics_sink: PackedTerminalSourceMetricsSink,
        clock_ns: Callable[[], int],
        physical_capacity: int,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
        grouped_nixl: GroupedNixlTerminalOwner,
        work: PackedTerminalSourceWork,
        bind_gather_cuda_device: Callable[[], None],
        retire_native_producers: Callable[[], None],
        resource_inventory: Callable[[], PackedTerminalSourceResourceInventory],
        retire_submission: Callable[
            [PackedTerminalSourceSubmission, NativeTerminalOwnerAction], None
        ],
    ) -> None:
        """Construct a dormant source serving composition.

        :param runtime: Sole process-lifetime native lifecycle runtime.
        :param cuda_completion: Direct source callback-to-owner producer.
        :param local_identity: Exact source process owned by the runtime.
        :param publisher: Canonical-rank publisher, otherwise ``None``.
        :param metrics_sink: Non-gating source metric projection.
        :param clock_ns: Local monotonic nanosecond clock.
        :param physical_capacity: Maximum configured in-flight generations.
        :param process_fatal_handler: Scheduler-affine fail-closed handler.
        :param grouped_nixl: Sole request-grouped native completion owner.
        :param work: Algorithm-neutral source work callbacks.
        :param bind_gather_cuda_device: Worker-thread CUDA device binding.
        :param retire_native_producers: Native event-channel retirement fence.
        :param resource_inventory: Exact external actor and DFlash ownership probe.
        :param retire_submission: Process-lifetime control-state retirement
            after native lifecycle retirement succeeds.
        """

        if type(runtime) is not NativeTerminalRuntime:
            raise TypeError("runtime must be NativeTerminalRuntime")
        if type(local_identity) is not TerminalProcessIdentity:
            raise TypeError("local_identity must be TerminalProcessIdentity")
        if local_identity.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("local_identity must belong to source")
        if type(physical_capacity) is not int or physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        if runtime.source_gather_actions.snapshot().capacity != physical_capacity:
            raise ValueError(
                "source gather inbox capacity must equal physical request capacity"
            )
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        if not isinstance(grouped_nixl, GroupedNixlTerminalOwner):
            raise TypeError("grouped_nixl must be GroupedNixlTerminalOwner")
        if type(work) is not PackedTerminalSourceWork:
            raise TypeError("work must be PackedTerminalSourceWork")
        if not callable(bind_gather_cuda_device):
            raise TypeError("bind_gather_cuda_device must be callable")
        if not callable(retire_native_producers):
            raise TypeError("retire_native_producers must be callable")
        if not callable(resource_inventory):
            raise TypeError("resource_inventory must be callable")
        if not callable(retire_submission):
            raise TypeError("retire_submission must be callable")
        wiring = PackedTerminalSourceWiring(
            runtime=runtime,
            cuda_completion=cuda_completion,
            local_identity=local_identity,
            publisher=publisher,
            metrics_sink=metrics_sink,
            clock_ns=clock_ns,
        )
        scheduler_consumer = PackedTerminalSourceSchedulerConsumer(
            wiring=wiring,
            process_fatal_handler=process_fatal_handler,
        )
        scheduler_serving = TerminalSchedulerServing(
            role=TerminalSchedulerServingRole.SOURCE,
            physical_capacity=physical_capacity,
            source_consumer=scheduler_consumer,
        )
        self._runtime = runtime
        self._wiring = wiring
        self._scheduler_consumer = scheduler_consumer
        self._scheduler_serving = scheduler_serving
        self._grouped_nixl = grouped_nixl
        self._work = work
        self._retire_native_producers = retire_native_producers
        self._resource_inventory = resource_inventory
        self._retire_submission = retire_submission
        self._owner_dead_marked = False
        self._native_producers_retired = False
        self._started = False
        self._closed = False
        self._lock = threading.Lock()
        self._gather_worker = PackedTerminalSourceGatherWorker(
            runtime=runtime,
            consume_action=self._consume_gather_action,
            bind_cuda_device=bind_gather_cuda_device,
            fatal_listener=self._gather_worker_failed,
        )

    @property
    def wiring(self) -> PackedTerminalSourceWiring:
        """Return the source control and publication boundary.

        :returns: Process-owned source wiring.
        """

        return self._wiring

    @property
    def scheduler_fileno(self) -> int:
        """Return the scheduler's qualified receipt wake descriptor.

        :returns: Readable scheduler inbox descriptor.
        """

        return self._scheduler_serving.fileno()

    @property
    def scheduler_serving(self) -> TerminalSchedulerServing:
        """Return the qualified source scheduler binding surface.

        :returns: Process-lifetime source scheduler serving adapter.
        """

        return self._scheduler_serving

    @property
    def runtime_filenos(self) -> tuple[int, ...]:
        """Return every source runtime descriptor owned by the process reactor.

        :returns: Scheduler-pump, source-work, publication, lifecycle, and
            observation descriptors in stable order.
        """

        return (
            self._runtime.scheduler_actions.fileno(),
            self._runtime.source_work_actions.fileno(),
            self._runtime.publisher_actions.fileno(),
            self._runtime.lifecycle_actions.fileno(),
            self._runtime.observations.fileno(),
            self._grouped_nixl.fileno(),
        )

    def start(self) -> None:
        """Start the native runtime and its dedicated gather execution context."""

        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("source serving cannot restart")
            self._started = True
        try:
            self._runtime.start()
            self._gather_worker.start()
        except BaseException:
            startup_traceback = traceback.format_exc()
            try:
                self._runtime.begin_abort("source serving startup failed")
            except BaseException:  # noqa: BLE001
                logger.critical(
                    "Source runtime abort after startup failure also failed:\n%s",
                    traceback.format_exc(),
                )
            try:
                self._mark_owner_dead()
            except BaseException:  # noqa: BLE001
                logger.critical(
                    "Source owner-death publication after startup failure also "
                    "failed:\n%s",
                    traceback.format_exc(),
                )
            logger.critical(
                "Source serving failed during startup:\n%s",
                startup_traceback,
            )
            raise

    def bind_submission(
        self,
        submission: PackedTerminalSourceSubmission,
        release_resources: Callable[[PackedTerminalSourceSubmission], None],
    ) -> None:
        """Bind scheduler ownership before publishing native lifecycle state.

        :param submission: Exact immutable post-forward source handoff.
        :param release_resources: Scheduler-affine one-shot resource release.
        """

        self._require_open()
        if type(submission) is not PackedTerminalSourceSubmission:
            raise TypeError("submission must be PackedTerminalSourceSubmission")
        if not callable(release_resources):
            raise TypeError("release_resources must be callable")
        binding = submission.identity.local_binding
        release = PackedTerminalSourceSchedulerRelease(
            binding=binding,
            release_resources=release_resources,
        )
        self._scheduler_consumer.register_release(release)
        try:
            self._scheduler_serving.register_request(binding)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._scheduler_consumer.cancel_unpublished(binding)
            raise
        try:
            self._wiring.accept_submission(submission)
        except (OSError, RuntimeError, TypeError, ValueError):
            if self._can_cancel_unpublished(binding):
                self._scheduler_consumer.cancel_unpublished(binding)
                self._scheduler_serving.cancel_unpublished_request(binding)
                raise
            self._mark_owner_dead()
            self._runtime.begin_abort()
            raise

    def attach_producer_completion(
        self,
        submission: PackedTerminalSourceSubmission,
    ) -> None:
        """Attach direct terminal delivery after PREPARE is published.

        :param submission: Exact accepted source submission.
        """

        self._require_open()
        self._wiring.attach_producer_completion(submission)

    def packed_ready(self, binding_digest: bytes) -> bool:
        """Deliver authenticated decoder allocation into the source join.

        :param binding_digest: Exact source lifecycle made transport-ready.
        :returns: Whether readiness released producer completion to native state.
        """

        self._require_open()
        return self._wiring.packed_ready(binding_digest)

    def drain_runtime_actions(self) -> int:
        """Drain every currently readable runtime execution-context inbox.

        :returns: Total immutable actions and observations consumed.
        """

        self._require_open()
        consumed = 0
        consumed += self._drain_scheduler_actions()
        consumed += self._drain_source_work_actions()
        consumed += self._drain_grouped_nixl()
        consumed += self._drain_publisher_actions()
        consumed += self._drain_lifecycle_actions()
        observations = self._runtime.observations.drain()
        for observation in observations:
            try:
                if type(observation) is NativeTerminalOwnerObservation:
                    self._wiring.submission_committed(observation)
                else:
                    self._work.observe_output(observation)
            except Exception:  # noqa: BLE001
                formatted_traceback = traceback.format_exc()
                self._runtime.report_observation_loss(observation)
                logger.error(
                    "Source terminal observation failed without gating lifecycle: %s",
                    formatted_traceback,
                )
        consumed += len(observations)
        self._propagate_runtime_fatal()
        return consumed

    def drain_scheduler_at_loop_entry(
        self,
    ) -> tuple[NativeTerminalOwnerAction, ...]:
        """Consume qualified reclaim authority on the scheduler thread.

        :returns: Exact reclaim actions successfully consumed.
        """

        self._require_open()
        return self._scheduler_serving.drain_at_loop_entry()

    def inventory(self) -> PackedTerminalSourceServingInventory:
        """Return complete cross-component source ownership evidence.

        :returns: Runtime, wiring, scheduler consumer, and receipt inventories.
        """

        with self._lock:
            owner_dead_marked = self._owner_dead_marked
            native_producers_retired = self._native_producers_retired
        return PackedTerminalSourceServingInventory(
            runtime=self._runtime.snapshot(),
            wiring=self._wiring.inventory(),
            scheduler_consumer=self._scheduler_consumer.inventory(),
            scheduler_serving=self._scheduler_serving.inventory(),
            gather_worker=self._gather_worker.inventory(),
            grouped_nixl=self._grouped_nixl.inventory(),
            resources=self._resource_inventory(),
            owner_dead_marked=owner_dead_marked,
            native_producers_retired=native_producers_retired,
        )

    def stop_admission_and_retire_producers(self) -> None:
        """Fence native gathers before retiring their downstream producers."""

        self._require_open()
        self._runtime.stop_admission()
        self._retire_native_producers()
        with self._lock:
            self._native_producers_retired = True
        shutdown_timeout: float = terminal_deadline_spec(
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
        ).seconds
        if not self._runtime.wait_for_output_projection(shutdown_timeout):
            self._runtime.begin_abort()
            raise RuntimeError(
                "source shutdown retained native output before gather drain"
            )
        self._gather_worker.stop_and_join(shutdown_timeout, abort=False)
        self._grouped_nixl.stop_admission()
        for producer_id in self._runtime.python_producer_ids:
            self._runtime.retire_python_producer(producer_id)
        self._runtime.join_producers()

    def close_clean(self, timeout_seconds: float) -> None:
        """Close a fully retired source composition with exact-zero inventory.

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
        if inventory.retained_resource_count != 0:
            raise RuntimeError("clean source serving close retains lifecycle authority")
        self._grouped_nixl.close_clean()
        self._runtime.close_clean()
        self._scheduler_serving.close()
        with self._lock:
            self._closed = True

    def abort_and_close(self) -> PackedTerminalSourceServingInventory:
        """Drain fail-closed authority and preserve every ambiguous resource.

        :returns: Final combined fail-closed inventory before descriptors close.
        """

        self._require_open()
        self._runtime.begin_abort()
        shutdown_timeout: float = terminal_deadline_spec(
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
        ).seconds
        snapshot = self._runtime.snapshot()
        if not snapshot.producers_joined:
            if not self._native_producers_retired:
                self._retire_native_producers()
                with self._lock:
                    self._native_producers_retired = True
            if not self._runtime.wait_for_output_projection(shutdown_timeout):
                raise RuntimeError("abort retained unrouted source terminal authority")
            self._gather_worker.stop_and_join(shutdown_timeout, abort=True)
            self._grouped_nixl.stop_admission()
            for producer_id in self._runtime.python_producer_ids:
                self._runtime.retire_python_producer(producer_id)
            self._runtime.join_producers()
        else:
            self._gather_worker.stop_and_join(shutdown_timeout, abort=True)
        self._mark_owner_dead()
        if not self._runtime.wait_for_output_projection_quiescence(shutdown_timeout):
            raise RuntimeError("abort retained unrouted source terminal authority")
        self.drain_runtime_actions()
        self._scheduler_serving.close_fail_closed()
        inventory = self.inventory()
        if (
            inventory.grouped_nixl.active_group_count == 0
            and inventory.grouped_nixl.active_transfer_count == 0
            and inventory.grouped_nixl.unowned_handle_count == 0
        ):
            self._grouped_nixl.close_clean()
        self._runtime.finish_abort_close()
        with self._lock:
            self._closed = True
        return inventory

    def begin_fail_closed_abort(self) -> None:
        """Stop functional side effects and wake scheduler-owned teardown.

        This boundary is used when scheduler-local state can no longer be
        reconciled with already-published native ownership. It deliberately
        preserves every live request for process teardown quarantine.
        """

        self._require_open()
        self._runtime.begin_abort()
        self._mark_owner_dead()

    def publisher_result(self, result: TerminalGatewayPublicationResult) -> None:
        """Return one exactly-once publisher outcome to source authority.

        :param result: Exact success or functional failure result.
        """

        self._require_open()
        self._wiring.publisher_result(result)

    def request_ready(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Deliver authenticated request readiness into source authority.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport readiness authority.
        :param local_receipt: Matching process-local authority.
        :param authenticated_issuer: Decode coordinator proved by routing.
        """

        self._require_open()
        self._wiring.request_ready(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
        )

    def request_failed(
        self,
        *,
        binding_digest: bytes,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
        reason: str,
    ) -> None:
        """Deliver authenticated request failure into source authority.

        :param binding_digest: Exact accepted source binding.
        :param wire_receipt: Transport failure authority.
        :param local_receipt: Matching process-local authority.
        :param authenticated_issuer: Decode coordinator proved by routing.
        :param reason: Stable request-global failure evidence.
        """

        self._require_open()
        self._wiring.request_failed(
            binding_digest=binding_digest,
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
            reason=reason,
        )

    def cancel_submission(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> PackedTerminalSourceCancellationDisposition:
        """Record scheduler cancellation after source publication cutover.

        :param binding: Exact scheduler-retained source generation.
        :param reason: Stable client-cancellation reason.
        :returns: Completion-required or too-late-for-rollback disposition.
        """

        self._require_open()
        return self._wiring.cancel_request(binding, reason)

    def publication_receipt(
        self,
        *,
        wire_receipt: TerminalWireReceipt,
        local_receipt: TerminalReceipt,
        authenticated_issuer: TerminalProcessIdentity,
    ) -> None:
        """Deliver one startup-route-authenticated publisher outcome.

        :param wire_receipt: Exact canonical source publisher receipt.
        :param local_receipt: Matching local import authority.
        :param authenticated_issuer: Source rank proved by control enrollment.
        """

        self._require_open()
        self._wiring.publication_receipt(
            wire_receipt=wire_receipt,
            local_receipt=local_receipt,
            authenticated_issuer=authenticated_issuer,
        )

    def _drain_scheduler_actions(self) -> int:
        """Transfer runtime reclaim actions into the qualified scheduler inbox.

        :returns: Number of actions accepted or aborted.
        """

        actions = self._runtime.scheduler_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                self._scheduler_serving.publish_action(action)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                reason = (
                    "source scheduler action publication failed: "
                    f"{type(error).__name__}: {error}"
                )
                self._runtime.fail_scheduler_action(action, reason)
                self._mark_owner_dead()
                raise
        return len(actions)

    def _drain_source_work_actions(self) -> int:
        """Execute ACKs before outcomes on the forward-independent reactor.

        :returns: Number of actions consumed or aborted.
        """

        actions = self._runtime.source_work_actions.drain()
        acknowledgements = tuple(
            action
            for action in actions
            if action.kind is NativeTerminalOwnerActionKind.SOURCE_ACK_READY
        )
        outcomes = tuple(
            action
            for action in actions
            if action.kind is NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY
        )
        if len(acknowledgements) + len(outcomes) != len(actions):
            raise RuntimeError("source work inbox carried another action kind")
        for action in (*acknowledgements, *outcomes):
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                if action.kind is NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
                    self._wiring.consume_outcome_ready(action, self._work.send_outcomes)
                elif action.kind is NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
                    self._wiring.consume_ack_ready(action, self._work.send_ack)
                else:
                    raise RuntimeError("source work inbox carried another action kind")
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error("Source work action failed:\n%s", traceback.format_exc())
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
        return len(actions)

    def _consume_gather_action(self, action: NativeTerminalOwnerAction) -> None:
        """Consume one direct-inbox gather on its dedicated worker thread.

        :param action: Exact native ``SOURCE_GATHER_READY`` authority.
        """

        self._wiring.consume_gather_ready(action, self._work.post_gather)

    def _gather_worker_failed(
        self,
        reason: str,
        formatted_traceback: str | None,
    ) -> None:
        """Publish worker death into scheduler-owned process-fatal handling.

        :param reason: Stable gather worker failure evidence.
        :param formatted_traceback: Complete worker traceback, if available.
        """

        traceback_text = formatted_traceback
        if traceback_text is None:
            traceback_text = "no traceback available"
        logger.critical(
            "Source gather worker entered process-fatal state: %s\n%s",
            reason,
            traceback_text,
        )
        self._mark_owner_dead()

    def _drain_grouped_nixl(self) -> int:
        """Commit complete request-level native transfer results.

        This drain runs after source work in the same process reactor. Even if
        a member completes inside its post call, ``SOURCE_GATHER_POSTED`` is
        therefore producer-ordered before the aggregate terminal transition.

        :returns: Number of aggregate results committed to native lifecycle.
        """

        results = self._grouped_nixl.drain()
        for result in results:
            try:
                self._wiring.grouped_native_terminal(result)
                self._grouped_nixl.acknowledge_result(result)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error(
                    "Grouped NIXL lifecycle ingress failed:\n%s",
                    traceback.format_exc(),
                )
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
        return len(results)

    def _drain_publisher_actions(self) -> int:
        """Transfer publication authority into the dedicated publisher.

        :returns: Number of publication actions consumed or aborted.
        """

        actions = self._runtime.publisher_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                self._wiring.consume_gateway_publication_ready(action)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error(
                    "Source publication action failed:\n%s", traceback.format_exc()
                )
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
        return len(actions)

    def _drain_lifecycle_actions(self) -> int:
        """Consume retirement, quarantine, and process-fatal authority.

        :returns: Number of lifecycle actions accepted or aborted.
        """

        actions = self._runtime.lifecycle_actions.drain()
        for action in actions:
            if action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL:
                self._mark_owner_dead()
                self._work.quarantine(action)
                if self._aborting:
                    self._runtime.acknowledge_aborted_action(action)
                else:
                    self._runtime.acknowledge_consumed_action(action)
                continue
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                if action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED:
                    self._work.quarantine(action)
                self._wiring.consume_terminal_action(
                    action,
                    self._retire_submission,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error(
                    "Source lifecycle retirement failed:\n%s",
                    traceback.format_exc(),
                )
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
        return len(actions)

    @property
    def _aborting(self) -> bool:
        """Return whether runtime authority is in fail-closed drain.

        :returns: Whether functional side effects must no longer execute.
        """

        disposition = self._runtime.disposition
        return disposition in (
            NativeTerminalRuntimeDisposition.ABORT_DRAINING,
            NativeTerminalRuntimeDisposition.PROCESS_FATAL,
        )

    def _can_cancel_unpublished(self, binding: TerminalRequestBinding) -> bool:
        """Return whether paired scheduler rollback remains legal.

        :param binding: Exact source lifecycle whose bind failed.
        :returns: Whether native lifecycle publication never completed.
        """

        inventory = self._wiring.inventory()
        if binding.digest not in inventory.active_binding_digests:
            return True
        return not self._wiring.lifecycle_published(binding.digest)

    def _propagate_runtime_fatal(self) -> None:
        """Wake scheduler fail-closed handling after any runtime inbox failure."""

        snapshot = self._runtime.snapshot()
        if snapshot.fatal_reason is not None:
            self._mark_owner_dead()

    def _mark_owner_dead(self) -> None:
        """Publish one sticky scheduler wake for runtime or component death."""

        with self._lock:
            if self._owner_dead_marked:
                return
            self._owner_dead_marked = True
        self._scheduler_serving.mark_owner_dead()

    def _require_open(self) -> None:
        """Require one started, non-closed source serving composition."""

        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("source serving is not open")
