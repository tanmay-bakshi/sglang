import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable

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
    NativeTerminalOwnerOutput,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    TerminalGatewayPublicationResult,
)
from sglang.srt.disaggregation.terminal_progress.receipts import TerminalReceipt
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
    TerminalSchedulerServingRole,
)
from sglang.srt.disaggregation.terminal_progress.source_scheduler_consumer import (
    PackedTerminalSourceSchedulerConsumer,
    PackedTerminalSourceSchedulerInventory,
    PackedTerminalSourceSchedulerRelease,
)
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceInventory,
    PackedTerminalSourceMetricsSink,
    PackedTerminalSourcePublisher,
    PackedTerminalSourceSubmission,
    PackedTerminalSourceWiring,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

logger = logging.getLogger(__name__)


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
    observe_output: Callable[[NativeTerminalOwnerOutput], None]

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
    :ivar grouped_nixl: Request-level main and DFlash transfer inventory.
    :ivar owner_dead_marked: Whether scheduler failure wake was published.
    :ivar native_producers_retired: Whether native producer contexts joined.
    """

    runtime: NativeTerminalRuntimeSnapshot
    wiring: PackedTerminalSourceInventory
    scheduler_consumer: PackedTerminalSourceSchedulerInventory
    scheduler_serving: TerminalSchedulerServingInventory
    grouped_nixl: GroupedNixlTerminalOwnerInventory
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
        if type(self.grouped_nixl) is not GroupedNixlTerminalOwnerInventory:
            raise TypeError("grouped_nixl must be GroupedNixlTerminalOwnerInventory")
        if type(self.owner_dead_marked) is not bool:
            raise TypeError("owner_dead_marked must be bool")
        if type(self.native_producers_retired) is not bool:
            raise TypeError("native_producers_retired must be bool")


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
    _grouped_nixl: GroupedNixlTerminalOwner
    _work: PackedTerminalSourceWork
    _retire_native_producers: Callable[[], None]
    _retire_submission: Callable[[PackedTerminalSourceSubmission], None]
    _owner_dead_marked: bool
    _native_producers_retired: bool
    _started: bool
    _closed: bool
    _lock: threading.Lock

    def __init__(
        self,
        *,
        runtime: NativeTerminalRuntime,
        local_identity: TerminalProcessIdentity,
        publisher: PackedTerminalSourcePublisher | None,
        metrics_sink: PackedTerminalSourceMetricsSink,
        clock_ns: Callable[[], int],
        physical_capacity: int,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
        grouped_nixl: GroupedNixlTerminalOwner,
        work: PackedTerminalSourceWork,
        retire_native_producers: Callable[[], None],
        retire_submission: Callable[[PackedTerminalSourceSubmission], None],
    ) -> None:
        """Construct a dormant source serving composition.

        :param runtime: Sole process-lifetime native lifecycle runtime.
        :param local_identity: Exact source process owned by the runtime.
        :param publisher: Canonical-rank publisher, otherwise ``None``.
        :param metrics_sink: Non-gating source metric projection.
        :param clock_ns: Local monotonic nanosecond clock.
        :param physical_capacity: Maximum configured in-flight generations.
        :param process_fatal_handler: Scheduler-affine fail-closed handler.
        :param grouped_nixl: Sole request-grouped native completion owner.
        :param work: Algorithm-neutral source work callbacks.
        :param retire_native_producers: Native event-channel retirement fence.
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
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        if type(grouped_nixl) is not GroupedNixlTerminalOwner:
            raise TypeError("grouped_nixl must be GroupedNixlTerminalOwner")
        if type(work) is not PackedTerminalSourceWork:
            raise TypeError("work must be PackedTerminalSourceWork")
        if not callable(retire_native_producers):
            raise TypeError("retire_native_producers must be callable")
        if not callable(retire_submission):
            raise TypeError("retire_submission must be callable")
        wiring = PackedTerminalSourceWiring(
            runtime=runtime,
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
        self._retire_submission = retire_submission
        self._owner_dead_marked = False
        self._native_producers_retired = False
        self._started = False
        self._closed = False
        self._lock = threading.Lock()

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
        """Start the sole native runtime exactly once."""

        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("source serving cannot restart")
            self._runtime.start()
            self._started = True

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
        for output in observations:
            self._work.observe_output(output)
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
            grouped_nixl=self._grouped_nixl.inventory(),
            owner_dead_marked=owner_dead_marked,
            native_producers_retired=native_producers_retired,
        )

    def stop_admission_and_retire_producers(self) -> None:
        """Close admission and join every native and Python producer context."""

        self._require_open()
        self._grouped_nixl.stop_admission()
        self._runtime.stop_admission()
        self._retire_native_producers()
        with self._lock:
            self._native_producers_retired = True
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
        if (
            inventory.runtime.scheduler_live_count != 0
            or inventory.runtime.scheduler_pending_count != 0
            or inventory.runtime.consumer_pending_count != 0
            or len(inventory.wiring.active_binding_digests) != 0
            or len(inventory.scheduler_consumer.active_binding_digests) != 0
            or inventory.scheduler_serving.inbox.live_count != 0
            or inventory.grouped_nixl.active_group_count != 0
            or inventory.grouped_nixl.active_transfer_count != 0
        ):
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
        snapshot = self._runtime.snapshot()
        if not snapshot.producers_joined:
            if not self._native_producers_retired:
                self._retire_native_producers()
                with self._lock:
                    self._native_producers_retired = True
            for producer_id in self._runtime.python_producer_ids:
                self._runtime.retire_python_producer(producer_id)
            self._runtime.join_producers()
        self._mark_owner_dead()
        shutdown_timeout: float = terminal_deadline_spec(
            TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
        ).seconds
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
        """Execute owner-earned source work outside the scheduler loop.

        :returns: Number of actions consumed or aborted.
        """

        actions = self._runtime.source_work_actions.drain()
        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                continue
            try:
                if action.kind is NativeTerminalOwnerActionKind.SOURCE_GATHER_READY:
                    self._wiring.consume_gather_ready(action, self._work.post_gather)
                elif action.kind is NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
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
                retired = self._wiring.consume_terminal_action(action)
                if retired is not None:
                    self._retire_submission(retired)
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
