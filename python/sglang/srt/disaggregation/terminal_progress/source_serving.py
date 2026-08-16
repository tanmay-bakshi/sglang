import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
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
    NativeTerminalActionClaimError,
    NativeTerminalObservation,
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeSnapshot,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerActionPublicationError,
    TerminalSchedulerDeliveryLease,
    TerminalSchedulerFailClosedClosure,
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
    PackedTerminalSourceFailClosedClosure,
    PackedTerminalSourceInventory,
    PackedTerminalSourceMetricsSink,
    PackedTerminalSourcePublicationRetentionError,
    PackedTerminalSourcePublisher,
    PackedTerminalSourceQuarantineRetentionError,
    PackedTerminalSourceSubmission,
    PackedTerminalSourceWiring,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

logger = logging.getLogger(__name__)

_SOURCE_CAUSAL_DELIVERY_ACTIONS = frozenset(
    (
        NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,
    )
)


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
class PackedTerminalSourceDeliveryLeaseInventory:
    """Exact request-scoped launch exclusions retained by source delivery.

    :ivar active_binding_digests: Requests which still exclude another launch.
    :ivar outcomes_sent_binding_digests: Active requests with durable outcomes.
    :ivar publication_owned_binding_digests: Active requests whose publication
        has a durable owner.
    """

    active_binding_digests: tuple[bytes, ...]
    outcomes_sent_binding_digests: tuple[bytes, ...]
    publication_owned_binding_digests: tuple[bytes, ...]

    def __post_init__(self) -> None:
        """Validate one exact causal-delivery inventory."""

        populations = (
            self.active_binding_digests,
            self.outcomes_sent_binding_digests,
            self.publication_owned_binding_digests,
        )
        if any(type(population) is not tuple for population in populations):
            raise TypeError("delivery lease populations must be tuples")
        if any(
            type(digest) is not bytes or len(digest) != 32
            for population in populations
            for digest in population
        ):
            raise ValueError("delivery lease identities must contain 32 bytes")
        if any(
            population != tuple(sorted(set(population))) for population in populations
        ):
            raise ValueError("delivery lease identities must be sorted and unique")
        active = set(self.active_binding_digests)
        if not set(self.outcomes_sent_binding_digests).issubset(active):
            raise ValueError("outcome milestones require active delivery leases")
        if not set(self.publication_owned_binding_digests).issubset(active):
            raise ValueError("publication milestones require active delivery leases")


@dataclasses.dataclass(slots=True)
class _PackedTerminalSourceDeliveryLeaseRecord:
    """Mutable two-milestone state for one request launch exclusion.

    :ivar binding: Exact source request generation.
    :ivar lease: Take-once scheduler launch exclusion.
    :ivar outcomes_sent: Whether ``SOURCE_OUTCOMES_SENT`` committed.
    :ivar publication_owned: Whether gateway publication has a durable owner.
    """

    binding: TerminalRequestBinding
    lease: TerminalSchedulerDeliveryLease
    outcomes_sent: bool = False
    publication_owned: bool = False

    def __post_init__(self) -> None:
        """Validate one live source delivery record."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if type(self.lease) is not TerminalSchedulerDeliveryLease:
            raise TypeError("lease must be TerminalSchedulerDeliveryLease")
        if self.lease.binding != self.binding:
            raise ValueError("delivery lease belongs to another request generation")
        if type(self.outcomes_sent) is not bool:
            raise TypeError("outcomes_sent must be bool")
        if type(self.publication_owned) is not bool:
            raise TypeError("publication_owned must be bool")


class _PackedTerminalSourceDeliveryLeases:
    """Transfer native handoff authority into request-scoped launch intents."""

    _scheduler_serving: TerminalSchedulerServing
    _records: dict[PackedRequestKey, _PackedTerminalSourceDeliveryLeaseRecord]
    _process_fatal: bool
    _scheduler_fatal: bool
    _lock: threading.Lock

    def __init__(self, scheduler_serving: TerminalSchedulerServing) -> None:
        """Create an empty source delivery registry.

        :param scheduler_serving: Source launch gate owning external intents.
        """

        if type(scheduler_serving) is not TerminalSchedulerServing:
            raise TypeError("scheduler_serving must be TerminalSchedulerServing")
        if scheduler_serving.role is not TerminalSchedulerServingRole.SOURCE:
            raise ValueError("source delivery leases require source scheduler serving")
        self._scheduler_serving = scheduler_serving
        self._records = {}
        self._process_fatal = False
        self._scheduler_fatal = False
        self._lock = threading.Lock()

    def acquire_for_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> None:
        """Establish every request intent before native claims are released.

        :param actions: Exact action population about to cross inbox delivery.
        """

        if type(actions) is not tuple or any(
            type(action) is not NativeTerminalOwnerAction for action in actions
        ):
            raise TypeError("actions must contain native terminal actions")
        try:
            self._acquire_for_actions(actions)
        except Exception as error:
            formatted_traceback = traceback.format_exc()
            try:
                self._retain_failure_barrier(actions)
            except Exception:  # noqa: BLE001
                logger.critical(
                    "Source delivery failure could not retain its launch barrier:\n%s",
                    traceback.format_exc(),
                )
            raise RuntimeError(
                f"source delivery lease acquisition failed:\n{formatted_traceback}"
            ) from error

    def _acquire_for_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> None:
        """Allocate one prevalidated request batch under the registry lock.

        :param actions: Exact action population about to cross inbox delivery.
        """

        bindings: dict[PackedRequestKey, TerminalRequestBinding] = {}
        action_kinds: dict[
            PackedRequestKey,
            set[NativeTerminalOwnerActionKind],
        ] = {}
        for action in actions:
            if action.kind not in _SOURCE_CAUSAL_DELIVERY_ACTIONS:
                continue
            binding = action.binding.to_binding()
            request_key = binding.request_key
            existing = bindings.get(request_key)
            if existing is not None and existing != binding:
                raise RuntimeError("delivery batch aliases a request generation")
            bindings[request_key] = binding
            action_kinds.setdefault(request_key, set()).add(action.kind)
        if len(bindings) == 0:
            return
        with self._lock:
            if self._process_fatal:
                raise RuntimeError("source delivery lease owner is process-fatal")
            for request_key, binding in bindings.items():
                existing = self._records.get(request_key)
                if existing is not None and existing.binding != binding:
                    raise RuntimeError("delivery lease aliases a request generation")
                if (
                    existing is None
                    and NativeTerminalOwnerActionKind.SOURCE_GATHER_READY
                    not in action_kinds[request_key]
                ):
                    raise RuntimeError("source delivery began after its gather handoff")
            new_records: dict[
                PackedRequestKey,
                _PackedTerminalSourceDeliveryLeaseRecord,
            ] = {}
            try:
                for request_key, binding in bindings.items():
                    if request_key in self._records:
                        continue
                    new_records[request_key] = _PackedTerminalSourceDeliveryLeaseRecord(
                        binding=binding,
                        lease=self._scheduler_serving.begin_delivery_lease(binding),
                    )
            except Exception as error:
                formatted_traceback = traceback.format_exc()
                self._records.update(new_records)
                raise RuntimeError(
                    "source delivery lease batch allocation failed:\n"
                    f"{formatted_traceback}"
                ) from error
            for request_key, record in new_records.items():
                if request_key in self._records:
                    for allocated in new_records.values():
                        if allocated.lease.active:
                            allocated.lease.complete()
                    raise RuntimeError("delivery lease changed during allocation")
                self._records[request_key] = record

    def _retain_failure_barrier(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
    ) -> None:
        """Keep launch exclusion active until outer owner-death is durable.

        This boundary must not take scheduler state. It may run while the
        pending-call handoff has interrupted that exact lock; the outer source
        failure path marks scheduler death only after the native claim releases
        the interrupted thread.

        :param actions: Exact failed delivery population.
        """

        causal = tuple(
            action
            for action in actions
            if action.kind in _SOURCE_CAUSAL_DELIVERY_ACTIONS
        )
        if len(causal) == 0:
            raise RuntimeError("delivery failure lacks a causal source action")
        binding = causal[0].binding.to_binding()
        with self._lock:
            if self._scheduler_fatal:
                return
            self._process_fatal = True
            if len(self._records) > 0:
                return
            self._records[binding.request_key] = (
                _PackedTerminalSourceDeliveryLeaseRecord(
                    binding=binding,
                    lease=self._scheduler_serving.begin_delivery_lease(binding),
                )
            )

    def mark_outcomes_sent(self, binding: TerminalRequestBinding) -> None:
        """Commit the source-outcome milestone and release a complete join.

        :param binding: Exact request whose outcome send committed natively.
        """

        with self._lock:
            record = self._require_record_locked(binding)
            if record.outcomes_sent:
                raise RuntimeError("source delivery outcomes committed twice")
            record.outcomes_sent = True
            self._complete_ready_locked(record)

    def mark_publication_owned(self, binding: TerminalRequestBinding) -> None:
        """Commit durable gateway ownership and release a complete join.

        :param binding: Exact request accepted by its publication owner.
        """

        with self._lock:
            record = self._require_record_locked(binding)
            if record.publication_owned:
                raise RuntimeError("source publication ownership committed twice")
            record.publication_owned = True
            self._complete_ready_locked(record)

    def release_quarantined(self, binding: TerminalRequestBinding) -> bool:
        """Release one intent after request-local quarantine is durable.

        :param binding: Exact quarantined source request generation.
        :returns: Whether the request retained a delivery lease.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        with self._lock:
            record = self._records.get(binding.request_key)
            if record is None:
                return False
            if record.binding != binding:
                raise RuntimeError("quarantine aliases a delivery lease generation")
            record.lease.complete()
            del self._records[binding.request_key]
            return True

    def release_process_fatal(self) -> None:
        """Release every intent after scheduler process-fatal became sticky."""

        with self._lock:
            self._process_fatal = True
            self._scheduler_fatal = True
            records = tuple(self._records.values())
            for record in records:
                if record.lease.active:
                    record.lease.complete()
            self._records.clear()

    def inventory(self) -> PackedTerminalSourceDeliveryLeaseInventory:
        """Return every active request and committed join milestone.

        :returns: Immutable causal-delivery inventory.
        """

        with self._lock:
            records = tuple(self._records.values())
        return PackedTerminalSourceDeliveryLeaseInventory(
            active_binding_digests=tuple(
                sorted(record.binding.digest for record in records)
            ),
            outcomes_sent_binding_digests=tuple(
                sorted(
                    record.binding.digest for record in records if record.outcomes_sent
                )
            ),
            publication_owned_binding_digests=tuple(
                sorted(
                    record.binding.digest
                    for record in records
                    if record.publication_owned
                )
            ),
        )

    def _require_record_locked(
        self,
        binding: TerminalRequestBinding,
    ) -> _PackedTerminalSourceDeliveryLeaseRecord:
        """Return one exact record while the registry lock is held.

        :param binding: Expected request generation.
        :returns: Matching live delivery record.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        record = self._records.get(binding.request_key)
        if record is None or record.binding != binding:
            raise RuntimeError("source delivery milestone lacks an exact lease")
        return record

    def _complete_ready_locked(
        self,
        record: _PackedTerminalSourceDeliveryLeaseRecord,
    ) -> None:
        """Release a two-milestone join while the registry lock is held.

        :param record: Exact active delivery record.
        """

        if not record.outcomes_sent or not record.publication_owned:
            return
        record.lease.complete()
        del self._records[record.binding.request_key]


@dataclasses.dataclass(frozen=True, slots=True)
class _PackedTerminalSourceRuntimeActionBatch:
    """One process-reactor turn claimed before any downstream execution.

    :ivar scheduler: Scheduler-affine reclaim actions.
    :ivar source_work: Forward-independent ACK and outcome actions.
    :ivar publisher: Forward-independent gateway publication actions.
    :ivar lifecycle: Retirement, quarantine, and process-fatal actions.
    """

    scheduler: tuple[NativeTerminalOwnerAction, ...]
    source_work: tuple[NativeTerminalOwnerAction, ...]
    publisher: tuple[NativeTerminalOwnerAction, ...]
    lifecycle: tuple[NativeTerminalOwnerAction, ...]

    def __post_init__(self) -> None:
        """Validate the immutable populations claimed in one reactor turn."""

        populations = (
            self.scheduler,
            self.source_work,
            self.publisher,
            self.lifecycle,
        )
        if any(type(population) is not tuple for population in populations):
            raise TypeError("source runtime action populations must be tuples")
        if any(
            type(action) is not NativeTerminalOwnerAction
            for population in populations
            for action in population
        ):
            raise TypeError(
                "source runtime action populations must contain native actions"
            )

    @property
    def action_count(self) -> int:
        """Return the total claimed action population.

        :returns: Number of actions claimed in this reactor turn.
        """

        return sum(
            (
                len(self.scheduler),
                len(self.source_work),
                len(self.publisher),
                len(self.lifecycle),
            )
        )

    @property
    def all_actions(self) -> tuple[NativeTerminalOwnerAction, ...]:
        """Return every claimed action in semantic execution order.

        :returns: Flattened immutable action population.
        """

        return (
            *self.scheduler,
            *self.source_work,
            *self.publisher,
            *self.lifecycle,
        )


class _PackedTerminalSourceRuntimeActionExecution:
    """Track exact local ownership while one claimed batch executes."""

    _actions: _PackedTerminalSourceRuntimeActionBatch
    _locally_owned_by_id: dict[int, NativeTerminalOwnerAction]

    def __init__(self, actions: _PackedTerminalSourceRuntimeActionBatch) -> None:
        """Retain every claimed action until its consumer accepts ownership.

        :param actions: Complete immutable population claimed for execution.
        """

        if type(actions) is not _PackedTerminalSourceRuntimeActionBatch:
            raise TypeError("actions must be a source runtime action batch")
        locally_owned_by_id = {
            action.action_id: action for action in actions.all_actions
        }
        if len(locally_owned_by_id) != actions.action_count:
            raise ValueError(
                "source runtime action batch contains duplicate identities"
            )
        self._actions = actions
        self._locally_owned_by_id = locally_owned_by_id

    def transfer(self, action: NativeTerminalOwnerAction) -> None:
        """Transfer one exact action after downstream acceptance.

        :param action: Action whose authority left the process-reactor turn.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        locally_owned = self._locally_owned_by_id.get(action.action_id)
        if locally_owned != action:
            raise RuntimeError(
                "source runtime action is absent, aliased, or already transferred"
            )
        del self._locally_owned_by_id[action.action_id]

    @property
    def locally_owned_actions(self) -> tuple[NativeTerminalOwnerAction, ...]:
        """Return locally retained actions in semantic execution order.

        :returns: Immutable population still eligible for reconciliation.
        """

        return tuple(
            action
            for action in self._actions.all_actions
            if action.action_id in self._locally_owned_by_id
        )


class _PackedTerminalSourceRuntimeActionBatchClaimError(RuntimeError):
    """A source reactor batch failed after removing one or more actions."""

    actions: _PackedTerminalSourceRuntimeActionBatch
    formatted_traceback: str

    def __init__(
        self,
        actions: _PackedTerminalSourceRuntimeActionBatch,
        formatted_traceback: str,
    ) -> None:
        """Retain every incrementally removed source action population.

        :param actions: Complete local authority removed before the failure.
        :param formatted_traceback: Original claim failure traceback.
        """

        if type(actions) is not _PackedTerminalSourceRuntimeActionBatch:
            raise TypeError("actions must be a source runtime action batch")
        if type(formatted_traceback) is not str or len(formatted_traceback) == 0:
            raise ValueError("formatted_traceback must be a non-empty string")
        self.actions = actions
        self.formatted_traceback = formatted_traceback
        super().__init__("source runtime action batch claim failed")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceServingInventory:
    """Combined source runtime, wiring, and scheduler ownership evidence.

    :ivar runtime: Authoritative process-lifetime runtime snapshot.
    :ivar wiring: Immutable source side-effect inventory.
    :ivar scheduler_consumer: Scheduler-affine resource inventory.
    :ivar scheduler_serving: Qualified scheduler receipt inventory.
    :ivar delivery_leases: Request-scoped launch exclusions and join state.
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
    delivery_leases: PackedTerminalSourceDeliveryLeaseInventory
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
        if type(self.delivery_leases) is not PackedTerminalSourceDeliveryLeaseInventory:
            raise TypeError(
                "delivery_leases must be PackedTerminalSourceDeliveryLeaseInventory"
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
                len(self.delivery_leases.active_binding_digests),
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
    _delivery_leases: _PackedTerminalSourceDeliveryLeases
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
        delivery_leases = _PackedTerminalSourceDeliveryLeases(scheduler_serving)
        runtime.bind_source_delivery_lease_acquirer(delivery_leases.acquire_for_actions)
        self._runtime = runtime
        self._wiring = wiring
        self._scheduler_consumer = scheduler_consumer
        self._scheduler_serving = scheduler_serving
        self._delivery_leases = delivery_leases
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
        actions: _PackedTerminalSourceRuntimeActionBatch | None = None
        execution: _PackedTerminalSourceRuntimeActionExecution | None = None
        try:
            with self._runtime.source_action_delivery_fence():
                actions = self._claim_runtime_action_batch()
                execution = _PackedTerminalSourceRuntimeActionExecution(actions)
                self._consume_scheduler_actions(actions.scheduler, execution)
            consumed = actions.action_count
            self._consume_source_work_actions(actions.source_work, execution)
            consumed += self._drain_grouped_nixl()
            self._consume_publisher_actions(actions.publisher, execution)
            self._consume_lifecycle_actions(actions.lifecycle, execution)
        except Exception as error:
            formatted_traceback = traceback.format_exc()
            if type(error) is _PackedTerminalSourceRuntimeActionBatchClaimError:
                actions = error.actions
                execution = _PackedTerminalSourceRuntimeActionExecution(actions)
                formatted_traceback = error.formatted_traceback
            if not self._aborting:
                try:
                    self._runtime.begin_abort(
                        "source claimed action batch execution failed"
                    )
                except Exception:  # noqa: BLE001
                    logger.critical(
                        "Source claimed action batch could not begin abort:\n%s",
                        traceback.format_exc(),
                    )
            if execution is not None:
                self._reconcile_aborted_action_execution(execution)
            self._mark_owner_dead()
            logger.error(
                "Source claimed action batch failed closed:\n%s",
                formatted_traceback,
            )
            raise
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

    def _claim_runtime_action_batch(
        self,
    ) -> _PackedTerminalSourceRuntimeActionBatch:
        """Claim every process-reactor action before invoking a consumer.

        A CPython pending call may interrupt the scheduler while its receipt
        inbox lock is held. Claiming the complete cross-inbox population first
        lets every forward-independent sibling release that handoff before a
        scheduler-affine publication can wait on the interrupted lock.

        :returns: Immutable, already-claimed action populations.
        """

        inboxes = (
            self._runtime.scheduler_actions,
            self._runtime.source_work_actions,
            self._runtime.publisher_actions,
            self._runtime.lifecycle_actions,
        )
        populations: list[tuple[NativeTerminalOwnerAction, ...]] = []
        try:
            for inbox in inboxes:
                try:
                    population = inbox.drain()
                except NativeTerminalActionClaimError as error:
                    populations.append(error.locally_claimed_actions)
                    raise
                populations.append(population)
        except Exception as error:
            populations.extend(() for _ in range(len(inboxes) - len(populations)))
            actions = _PackedTerminalSourceRuntimeActionBatch(
                scheduler=populations[0],
                source_work=populations[1],
                publisher=populations[2],
                lifecycle=populations[3],
            )
            formatted_traceback = traceback.format_exc()
            if type(error) is NativeTerminalActionClaimError:
                formatted_traceback = error.formatted_traceback
            raise _PackedTerminalSourceRuntimeActionBatchClaimError(
                actions,
                formatted_traceback,
            ) from error
        return _PackedTerminalSourceRuntimeActionBatch(
            scheduler=populations[0],
            source_work=populations[1],
            publisher=populations[2],
            lifecycle=populations[3],
        )

    def _reconcile_aborted_action_execution(
        self,
        execution: _PackedTerminalSourceRuntimeActionExecution,
    ) -> None:
        """Release every locally claimed action still pending after failure.

        Downstream consumers which already accepted an action retain their own
        fail-closed authority. This reconciliation releases only the runtime's
        local consumer accounting, including actions later in the immutable
        batch which were never executed.

        :param execution: Exact ownership remaining in the failed turn.
        """

        for action in execution.locally_owned_actions:
            try:
                self._runtime.acknowledge_aborted_action_if_pending(action)
            except Exception:  # noqa: BLE001
                logger.critical(
                    "Source claimed action reconciliation failed for action %d:\n%s",
                    action.action_id,
                    traceback.format_exc(),
                )

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
            delivery_leases=self._delivery_leases.inventory(),
            gather_worker=self._gather_worker.inventory(),
            grouped_nixl=self._grouped_nixl.inventory(),
            resources=self._resource_inventory(),
            owner_dead_marked=owner_dead_marked,
            native_producers_retired=native_producers_retired,
        )

    def stop_admission_and_retire_producers(self) -> None:
        """Retire native work and every Python ingress producer."""

        self.stop_admission_and_retire_native_producers()
        self.retire_python_producers()

    def stop_admission_and_retire_native_producers(self) -> None:
        """Fence native gathers while publication-result ingress stays live."""

        self._require_open()
        self._runtime.stop_admission()
        with self._lock:
            native_producers_retired = self._native_producers_retired
        if not native_producers_retired:
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

    def retire_python_producers(self) -> None:
        """Close result ingress after every publisher and receiver joined."""

        self._require_open()
        with self._lock:
            if not self._native_producers_retired:
                raise RuntimeError(
                    "Python producer retirement preceded native producer drain"
                )
        if self._runtime.snapshot().producers_joined:
            return
        for producer_id in self._runtime.unretired_python_producer_ids:
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

    def drain_shutdown_actions(self, timeout_seconds: float) -> None:
        """Drain projected work while its scheduler and publisher remain live.

        This boundary is called after the process reactor joins and on both
        sides of publisher shutdown. The first pass delivers every projected
        publication before publisher admission closes. The second consumes
        publisher results before their Python producer namespaces retire.

        :param timeout_seconds: Hash-bound native projection fence timeout.
        """

        self._require_open()
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        if not self._runtime.wait_for_output_projection(timeout_seconds):
            self._runtime.begin_abort("source shutdown projection fence timed out")
            raise RuntimeError("source shutdown projection fence timed out")
        self.drain_runtime_actions()
        self.drain_scheduler_at_loop_entry()
        if not self._runtime.wait_for_output_projection(timeout_seconds):
            self._runtime.begin_abort("source shutdown projection fence timed out")
            raise RuntimeError("source shutdown projection fence timed out")
        self.drain_runtime_actions()

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
            for producer_id in self._runtime.unretired_python_producer_ids:
                self._runtime.retire_python_producer(producer_id)
            self._runtime.join_producers()
        else:
            self._gather_worker.stop_and_join(shutdown_timeout, abort=True)
        self._mark_owner_dead()
        if not self._runtime.wait_for_output_projection_quiescence(shutdown_timeout):
            raise RuntimeError("abort retained unrouted source terminal authority")
        self.drain_runtime_actions()
        scheduler_closure = self._scheduler_serving.close_fail_closed()
        wiring_closure = self._wiring.take_fail_closed_closure()
        self._surrender_fail_closed_downstream_actions(
            scheduler_closure,
            wiring_closure,
        )
        inventory = dataclasses.replace(
            self.inventory(),
            scheduler_serving=scheduler_closure.inventory,
            wiring=wiring_closure.inventory,
        )
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

    def _surrender_fail_closed_downstream_actions(
        self,
        scheduler_closure: TerminalSchedulerFailClosedClosure,
        wiring_closure: PackedTerminalSourceFailClosedClosure,
    ) -> None:
        """Release runtime accounting after downstream quarantine is durable.

        Scheduler and publisher consumers retain accepted actions beyond the
        process-reactor turn. At process-fatal close, their own inventories are
        the durable authority. Runtime accounting can be surrendered only
        after those inventories prove the corresponding request quarantined.

        :param scheduler_closure: Take-once scheduler quarantine authority.
        :param wiring_closure: Take-once publisher quarantine authority.
        """

        if type(scheduler_closure) is not TerminalSchedulerFailClosedClosure:
            raise TypeError(
                "scheduler_closure must be TerminalSchedulerFailClosedClosure"
            )
        if type(wiring_closure) is not PackedTerminalSourceFailClosedClosure:
            raise TypeError(
                "wiring_closure must be PackedTerminalSourceFailClosedClosure"
            )
        scheduler_inventory = self._scheduler_consumer.inventory()
        scheduler_quarantined = frozenset(
            scheduler_inventory.quarantined_binding_digests
        )
        scheduler_actions = scheduler_closure.retained_actions
        publisher_actions = wiring_closure.retained_publication_actions
        quarantine_actions = wiring_closure.retained_quarantine_actions
        for action in scheduler_actions:
            if action.kind is not NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
                raise RuntimeError("source scheduler retained another action kind")
            if action.binding.digest not in scheduler_quarantined:
                raise RuntimeError(
                    "source scheduler action lacks durable quarantine ownership"
                )
        actions = (*scheduler_actions, *publisher_actions, *quarantine_actions)
        action_ids = frozenset(action.action_id for action in actions)
        if len(action_ids) != len(actions):
            raise RuntimeError("fail-closed downstream action identities overlap")
        surrender = self._runtime.surrender_fail_closed_actions(actions)
        if surrender.action_ids != tuple(action.action_id for action in actions):
            raise RuntimeError("runtime surrender receipt differs from closure order")

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

    def _consume_scheduler_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
        execution: _PackedTerminalSourceRuntimeActionExecution,
    ) -> None:
        """Publish one already-claimed scheduler action population.

        :param actions: Immutable actions claimed before downstream execution.
        :param execution: Exact process-reactor ownership ledger.
        """

        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                execution.transfer(action)
                continue
            try:
                self._scheduler_serving.publish_action(action)
            except TerminalSchedulerActionPublicationError as error:
                reason = (
                    "source scheduler action publication failed: "
                    f"{type(error).__name__}: {error}"
                )
                if not error.scheduler_retains_action:
                    self._runtime.fail_scheduler_action(action, reason)
                execution.transfer(action)
                self._mark_owner_dead()
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                reason = (
                    "source scheduler action publication failed: "
                    f"{type(error).__name__}: {error}"
                )
                self._runtime.fail_scheduler_action(action, reason)
                execution.transfer(action)
                self._mark_owner_dead()
                raise
            execution.transfer(action)

    def _drain_source_work_actions(self) -> int:
        """Execute ACKs before outcomes on the forward-independent reactor.

        :returns: Number of actions consumed or aborted.
        """

        actions = self._runtime.source_work_actions.drain()
        execution = _PackedTerminalSourceRuntimeActionExecution(
            _PackedTerminalSourceRuntimeActionBatch(
                scheduler=(),
                source_work=actions,
                publisher=(),
                lifecycle=(),
            )
        )
        self._consume_source_work_actions(actions, execution)
        return len(actions)

    def _consume_source_work_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
        execution: _PackedTerminalSourceRuntimeActionExecution,
    ) -> None:
        """Execute one already-claimed source-work population.

        :param actions: Immutable actions claimed before downstream execution.
        :param execution: Exact process-reactor ownership ledger.
        """

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
                execution.transfer(action)
                continue
            try:
                if action.kind is NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY:
                    self._wiring.consume_outcome_ready(action, self._work.send_outcomes)
                    if self._runtime.forward_independent_handoff_enabled:
                        self._delivery_leases.mark_outcomes_sent(
                            action.binding.to_binding()
                        )
                elif action.kind is NativeTerminalOwnerActionKind.SOURCE_ACK_READY:
                    self._wiring.consume_ack_ready(action, self._work.send_ack)
                else:
                    raise RuntimeError("source work inbox carried another action kind")
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error("Source work action failed:\n%s", traceback.format_exc())
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
            execution.transfer(action)

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

    def _consume_publisher_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
        execution: _PackedTerminalSourceRuntimeActionExecution,
    ) -> None:
        """Publish one already-claimed gateway action population.

        :param actions: Immutable actions claimed before downstream execution.
        :param execution: Exact process-reactor ownership ledger.
        """

        for action in actions:
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                execution.transfer(action)
                continue
            try:
                self._wiring.consume_gateway_publication_ready(action)
                execution.transfer(action)
                if self._runtime.forward_independent_handoff_enabled:
                    self._delivery_leases.mark_publication_owned(
                        action.binding.to_binding()
                    )
            except PackedTerminalSourcePublicationRetentionError:
                execution.transfer(action)
                logger.error(
                    "Source publication retained exact authority on failure:\n%s",
                    traceback.format_exc(),
                )
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.error(
                    "Source publication action failed:\n%s", traceback.format_exc()
                )
                self._mark_owner_dead()
                self._runtime.begin_abort()
                raise

    def _consume_lifecycle_actions(
        self,
        actions: tuple[NativeTerminalOwnerAction, ...],
        execution: _PackedTerminalSourceRuntimeActionExecution,
    ) -> None:
        """Consume one already-claimed lifecycle action population.

        :param actions: Immutable actions claimed before downstream execution.
        :param execution: Exact process-reactor ownership ledger.
        """

        for action in actions:
            if action.kind in (
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
                NativeTerminalOwnerActionKind.PROCESS_FATAL,
            ):
                try:
                    if action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL:
                        self._mark_owner_dead()
                    self._wiring.consume_quarantine(
                        action,
                        self._work.quarantine,
                    )
                    if (
                        action.kind is NativeTerminalOwnerActionKind.REQUEST_QUARANTINED
                        and self._runtime.forward_independent_handoff_enabled
                    ):
                        self._delivery_leases.release_quarantined(
                            action.binding.to_binding()
                        )
                except PackedTerminalSourceQuarantineRetentionError:
                    execution.transfer(action)
                    logger.error(
                        "Source lifecycle quarantine retained exact authority:\n%s",
                        traceback.format_exc(),
                    )
                    self._mark_owner_dead()
                    self._runtime.begin_abort()
                    raise
                except (OSError, RuntimeError, TypeError, ValueError):
                    logger.error(
                        "Source lifecycle quarantine failed:\n%s",
                        traceback.format_exc(),
                    )
                    self._mark_owner_dead()
                    self._runtime.begin_abort()
                    raise
                execution.transfer(action)
                continue
            if self._aborting:
                self._runtime.acknowledge_aborted_action(action)
                execution.transfer(action)
                continue
            try:
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
            execution.transfer(action)

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
            self._scheduler_serving.mark_owner_dead()
            if self._runtime.forward_independent_handoff_enabled:
                self._delivery_leases.release_process_fatal()
            self._owner_dead_marked = True

    def _require_open(self) -> None:
        """Require one started, non-closed source serving composition."""

        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("source serving is not open")
