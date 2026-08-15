import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sglang.srt.disaggregation.nixl.packed_staging_request import (
    PackedDecodeRequestTransaction,
)
from sglang.srt.disaggregation.terminal_progress.decode_adoption import (
    TerminalDFlashDecodeAdoption,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxInventory,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_serving import (
    TerminalSchedulerServing,
    TerminalSchedulerServingRole,
)
from sglang.srt.disaggregation.terminal_progress.source_plan import (
    PackedTerminalSourcePlan,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class TerminalDecodeAdoptionWiring(Protocol):
    """Decode lifecycle boundary consumed by the scheduler adapter."""

    def bind_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        binding: TerminalRequestBinding,
        source_plan: PackedTerminalSourcePlan,
    ) -> None:
        """Bind native lifecycle identity before allocation publication.

        :param transaction: Exact retained packed request transaction.
        :param binding: Rank-local decode lifecycle identity.
        :param source_plan: Complete source-writer and coordinator plan.
        """

    def consume_adoption_action(
        self,
        action: NativeTerminalOwnerAction,
        adopt_request: Callable[[object], TerminalDFlashDecodeAdoption],
        finalize_request: Callable[[object], None],
    ) -> object:
        """Consume allocation and metadata authority around scheduler callbacks.

        :param action: Exact owner-minted adoption authority.
        :param adopt_request: Scheduler metadata-copy and queue-adoption callback.
        :param finalize_request: Scheduler transaction-finalization callback.
        :returns: Exact retained request owner.
        """

    def cancel_unpublished(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> object:
        """Return one unpublished request owner under transaction authority.

        :param transaction: Exact retained packed request transaction.
        :param reason: Stable cancellation evidence.
        :returns: Exact retained request owner.
        """

    def quarantine_transaction(
        self,
        transaction: PackedDecodeRequestTransaction,
        reason: str,
    ) -> None:
        """Retain one ambiguous transaction against reuse.

        :param transaction: Exact retained packed request transaction.
        :param reason: Stable fail-closed evidence.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeSchedulerRegistration:
    """Exact scheduler-owned decode request retained before publication.

    The callbacks close over mutable scheduler structures but never leave the
    scheduler thread. Native owner threads carry only the immutable binding and
    action. The transaction is retained here solely as process-local lifecycle
    authority and is never inferred from a request identifier.

    :ivar binding: Exact rank-local decode lifecycle identity.
    :ivar source_plan: Complete authenticated source and coordinator plan.
    :ivar transaction: Request-scoped packed allocation owner.
    :ivar request_owner: Exact mutable request owner retained by the transaction.
    :ivar adopt_request: Copy metadata and install the exact request into
        scheduler-owned structures while its metadata row remains pinned.
    :ivar finalize_request: Clear transaction-local scheduler state after
        metadata release and before local-ready publication.
    :ivar cancel_request: Reconcile the exact request after safe unpublished
        transaction cancellation.
    :ivar quarantine_request: Retain scheduler-owned state after ambiguity.
    """

    binding: TerminalRequestBinding
    source_plan: PackedTerminalSourcePlan
    transaction: PackedDecodeRequestTransaction
    request_owner: object
    adopt_request: Callable[[object], TerminalDFlashDecodeAdoption]
    finalize_request: Callable[[object], None]
    cancel_request: Callable[[object], None]
    quarantine_request: Callable[[object, str], None]

    def __post_init__(self) -> None:
        """Validate one identity-stable scheduler registration."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if self.binding.owner.role is not TerminalOwnerRole.DECODE:
            raise ValueError("decode scheduler registration requires a decode binding")
        if type(self.source_plan) is not PackedTerminalSourcePlan:
            raise TypeError("source_plan must be PackedTerminalSourcePlan")
        if type(self.transaction) is not PackedDecodeRequestTransaction:
            raise TypeError("transaction must be PackedDecodeRequestTransaction")
        if self.transaction.request_owner is not self.request_owner:
            raise ValueError("transaction retains another scheduler request owner")
        snapshot = self.transaction.snapshot()
        if snapshot.key != self.binding.request_key:
            raise ValueError("decode binding differs from transaction generation")
        if self.source_plan.request_key != self.binding.request_key:
            raise ValueError("source plan differs from decode request generation")
        if (
            self.source_plan.rank_manifest_digest != self.binding.rank_manifest_digest
            or self.source_plan.allocation_digest != self.binding.allocation_digest
        ):
            raise ValueError("source plan differs from decode allocation identity")
        callbacks = (
            self.adopt_request,
            self.finalize_request,
            self.cancel_request,
            self.quarantine_request,
        )
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("decode scheduler registration callbacks must be callable")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalDecodeSchedulerInventory:
    """Complete decode scheduler ownership and failure inventory.

    :ivar active_binding_digests: Registrations retaining request ownership.
    :ivar bound_binding_digests: Registrations bound to native lifecycle state.
    :ivar consuming_binding_digests: Registrations executing adoption.
    :ivar scheduler_adopted_binding_digests: Requests installed before a later
        ambiguous transition.
    :ivar scheduler_finalized_binding_digests: Requests whose local scheduler
        finalization returned before a later ambiguous transition.
    :ivar cancelled_binding_digests: Safely cancelled resources awaiting paired
        scheduler-inbox retirement.
    :ivar quarantined_binding_digests: Ambiguous registrations retained.
    :ivar adopted_count: Fully consumed adoption registrations.
    :ivar cancelled_count: Fully paired unpublished cancellations.
    :ivar fatal_inventory: Sticky scheduler-inbox failure evidence.
    """

    active_binding_digests: tuple[bytes, ...]
    bound_binding_digests: tuple[bytes, ...]
    consuming_binding_digests: tuple[bytes, ...]
    scheduler_adopted_binding_digests: tuple[bytes, ...]
    scheduler_finalized_binding_digests: tuple[bytes, ...]
    cancelled_binding_digests: tuple[bytes, ...]
    quarantined_binding_digests: tuple[bytes, ...]
    adopted_count: int
    cancelled_count: int
    fatal_inventory: SchedulerReceiptInboxInventory | None

    def __post_init__(self) -> None:
        """Validate conservation of retained scheduler request identities."""

        populations = (
            self.active_binding_digests,
            self.bound_binding_digests,
            self.consuming_binding_digests,
            self.scheduler_adopted_binding_digests,
            self.scheduler_finalized_binding_digests,
            self.cancelled_binding_digests,
            self.quarantined_binding_digests,
        )
        if any(type(population) is not tuple for population in populations):
            raise TypeError("decode scheduler inventory populations must be tuples")
        if any(
            type(digest) is not bytes or len(digest) != 32
            for population in populations
            for digest in population
        ):
            raise ValueError("decode scheduler identities must contain 32 bytes")
        if any(tuple(sorted(population)) != population for population in populations):
            raise ValueError("decode scheduler identities must use digest order")
        active = set(self.active_binding_digests)
        for population in populations[1:]:
            if not set(population).issubset(active):
                raise ValueError("decode scheduler subpopulation exceeds active set")
        if not set(self.scheduler_finalized_binding_digests).issubset(
            set(self.scheduler_adopted_binding_digests)
        ):
            raise ValueError("finalized decode requests must first be adopted")
        if type(self.adopted_count) is not int or self.adopted_count < 0:
            raise ValueError("adopted_count must be a non-negative integer")
        if type(self.cancelled_count) is not int or self.cancelled_count < 0:
            raise ValueError("cancelled_count must be a non-negative integer")
        if self.fatal_inventory is not None:
            if type(self.fatal_inventory) is not SchedulerReceiptInboxInventory:
                raise TypeError("fatal_inventory must be scheduler inbox evidence")
            if self.fatal_inventory.fatal_cause is None:
                raise ValueError("fatal_inventory must carry a fatal cause")
            if set(self.quarantined_binding_digests) != active:
                raise ValueError("process-fatal decode requests must all be retained")


@dataclasses.dataclass(slots=True)
class _DecodeSchedulerRecord:
    """Mutable scheduler-thread state for one retained decode registration."""

    registration: PackedTerminalDecodeSchedulerRegistration
    bound: bool = False
    consuming: bool = False
    scheduler_adopted: bool = False
    scheduler_finalized: bool = False
    cancelled: bool = False
    quarantined: bool = False
    quarantine_callback_delivered: bool = False


class PackedTerminalDecodeSchedulerConsumer:
    """Consume decode adoption authority on the owning scheduler thread.

    The consumer conserves the exact mutable request across native action
    delivery. It does not accept a replacement request from a callback and it
    never infers scheduler ownership from a room or request id. Any partial
    adoption remains retained and quarantined for process-fatal teardown.
    """

    _wiring: TerminalDecodeAdoptionWiring
    _process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None]
    _scheduler_thread_id: int
    _records: dict[bytes, _DecodeSchedulerRecord]
    _adopted_count: int
    _cancelled_count: int
    _fatal_inventory: SchedulerReceiptInboxInventory | None
    _lock: threading.Lock

    def __init__(
        self,
        *,
        wiring: TerminalDecodeAdoptionWiring,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
    ) -> None:
        """Construct a scheduler-affine decode adoption consumer.

        :param wiring: Native decode lifecycle and transaction boundary.
        :param process_fatal_handler: Scheduler-owned admission-stop and teardown
            operation.
        """

        if not isinstance(wiring, TerminalDecodeAdoptionWiring):
            raise TypeError("wiring must satisfy TerminalDecodeAdoptionWiring")
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        self._wiring = wiring
        self._process_fatal_handler = process_fatal_handler
        self._scheduler_thread_id = threading.get_ident()
        self._records = {}
        self._adopted_count = 0
        self._cancelled_count = 0
        self._fatal_inventory = None
        self._lock = threading.Lock()

    def register_adoption(
        self,
        registration: PackedTerminalDecodeSchedulerRegistration,
    ) -> None:
        """Retain one exact request before native lifecycle publication.

        :param registration: Complete scheduler request ownership.
        """

        self._require_scheduler_thread()
        if type(registration) is not PackedTerminalDecodeSchedulerRegistration:
            raise TypeError(
                "registration must be PackedTerminalDecodeSchedulerRegistration"
            )
        digest = registration.binding.digest
        with self._lock:
            self._require_healthy_locked()
            if digest in self._records:
                raise RuntimeError("decode scheduler registration identity was reused")
            self._records[digest] = _DecodeSchedulerRecord(registration=registration)

    def bind_adoption(self, binding: TerminalRequestBinding) -> None:
        """Bind actor and native lifecycle after scheduler retention exists.

        :param binding: Exact retained decode lifecycle identity.
        """

        self._require_scheduler_thread()
        record = self._require_record(binding)
        if record.bound:
            raise RuntimeError("decode scheduler registration was already bound")
        try:
            self._wiring.bind_transaction(
                record.registration.transaction,
                record.registration.binding,
                record.registration.source_plan,
            )
        except Exception:
            self._quarantine_record(
                record,
                "decode lifecycle binding failed",
            )
            logger.error(
                "Decode lifecycle binding failed closed:\n%s",
                traceback.format_exc(),
            )
            raise
        record.bound = True

    def consume_adoption_ready(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Install one exact decode request under owner-minted authority.

        :param action: Exact adoption action transferred by the native owner.
        """

        self._require_scheduler_thread()
        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.ADOPTION_READY:
            raise ValueError("decode scheduler consumer requires adoption authority")
        record = self._require_native_record(action.binding)
        if not record.bound:
            raise RuntimeError("decode adoption preceded lifecycle binding")
        if record.consuming:
            raise RuntimeError("decode adoption authority is already consuming")
        if record.cancelled:
            raise RuntimeError("cancelled decode resources cannot be adopted")
        if record.quarantined:
            raise RuntimeError("quarantined decode resources cannot be adopted")
        record.consuming = True
        registration = record.registration

        def adopt_request(owner: object) -> TerminalDFlashDecodeAdoption:
            """Install the retained request and return exact row-copy authority.

            :param owner: Exact scheduler request retained by the transaction.
            :returns: Authenticated DFlash row-copy completion authority.
            """

            self._require_scheduler_thread()
            if owner is not registration.request_owner:
                raise RuntimeError("decode adoption returned another request owner")
            if record.scheduler_adopted:
                raise RuntimeError("decode scheduler request was adopted twice")
            adoption = registration.adopt_request(owner)
            if type(adoption) is not TerminalDFlashDecodeAdoption:
                raise RuntimeError(
                    "decode scheduler adoption returned invalid DFlash authority"
                )
            record.scheduler_adopted = True
            return adoption

        def finalize_request(owner: object) -> None:
            """Finish scheduler state before local-ready publication."""

            self._require_scheduler_thread()
            if owner is not registration.request_owner:
                raise RuntimeError("decode finalization returned another request owner")
            if not record.scheduler_adopted:
                raise RuntimeError("decode finalization preceded scheduler adoption")
            if record.scheduler_finalized:
                raise RuntimeError("decode scheduler request was finalized twice")
            registration.finalize_request(owner)
            record.scheduler_finalized = True

        try:
            owner = self._wiring.consume_adoption_action(
                action,
                adopt_request,
                finalize_request,
            )
            if owner is not registration.request_owner:
                raise RuntimeError("decode wiring returned another request owner")
            if not record.scheduler_adopted or not record.scheduler_finalized:
                raise RuntimeError(
                    "decode wiring returned before scheduler adoption completed"
                )
        except Exception:
            record.consuming = False
            self._quarantine_record(record, "decode scheduler adoption failed")
            logger.error(
                "Decode scheduler consumption failed closed:\n%s",
                traceback.format_exc(),
            )
            raise

        digest = registration.binding.digest
        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("decode scheduler registry changed during adoption")
            del self._records[digest]
            self._adopted_count += 1

    def begin_unpublished_cancellation(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> None:
        """Safely cancel transaction resources but retain inbox pairing state.

        :param binding: Exact unpublished decode lifecycle identity.
        :param reason: Stable cancellation evidence.
        """

        self._require_scheduler_thread()
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        record = self._require_record(binding)
        if not record.bound:
            raise RuntimeError("unbound decode registration cannot cancel transaction")
        if record.consuming or record.quarantined or record.cancelled:
            raise RuntimeError("decode registration cannot enter cancellation")
        try:
            owner = self._wiring.cancel_unpublished(
                record.registration.transaction,
                reason,
            )
            if owner is not record.registration.request_owner:
                raise RuntimeError("decode cancellation returned another request owner")
            record.registration.cancel_request(owner)
        except Exception:
            self._quarantine_record(record, "decode unpublished cancellation failed")
            logger.error(
                "Decode unpublished cancellation failed closed:\n%s",
                traceback.format_exc(),
            )
            raise
        record.cancelled = True

    def finish_unpublished_cancellation(
        self,
        binding: TerminalRequestBinding,
    ) -> None:
        """Retire cancellation only after scheduler inbox unregistration.

        :param binding: Exact safely cancelled decode lifecycle identity.
        """

        self._require_scheduler_thread()
        record = self._require_record(binding)
        if not record.cancelled:
            raise RuntimeError("decode cancellation resources are not terminal")
        if record.quarantined:
            raise RuntimeError("quarantined decode cancellation cannot retire")
        with self._lock:
            current = self._records.get(binding.digest)
            if current is not record:
                raise RuntimeError(
                    "decode scheduler registry changed during cancellation"
                )
            del self._records[binding.digest]
            self._cancelled_count += 1

    def quarantine_registration(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> None:
        """Retain one registration after composition or inbox ambiguity.

        :param binding: Exact retained decode lifecycle identity.
        :param reason: Stable fail-closed evidence.
        """

        self._require_scheduler_thread()
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        self._quarantine_record(self._require_record(binding), reason)

    def process_fatal(self, inventory: SchedulerReceiptInboxInventory) -> None:
        """Quarantine every retained decode request and stop admission.

        :param inventory: Complete sticky scheduler-inbox failure evidence.
        """

        self._require_scheduler_thread()
        if type(inventory) is not SchedulerReceiptInboxInventory:
            raise TypeError("inventory must be SchedulerReceiptInboxInventory")
        if inventory.fatal_cause is None:
            raise ValueError("process-fatal handling requires a fatal cause")
        with self._lock:
            if self._fatal_inventory is not None:
                if self._fatal_inventory != inventory:
                    raise RuntimeError("decode scheduler fatal evidence changed")
                return
            self._fatal_inventory = inventory
            records = tuple(self._records.values())
        for record in records:
            self._quarantine_record(record, "decode scheduler inbox is process-fatal")
        self._process_fatal_handler(inventory)

    def inventory(self) -> PackedTerminalDecodeSchedulerInventory:
        """Return exact request ownership and fail-closed evidence.

        :returns: Immutable scheduler-side decode lifecycle inventory.
        """

        with self._lock:
            records = tuple(self._records.items())
            active = tuple(sorted(digest for digest, _ in records))
            inventory = PackedTerminalDecodeSchedulerInventory(
                active_binding_digests=active,
                bound_binding_digests=tuple(
                    sorted(digest for digest, record in records if record.bound)
                ),
                consuming_binding_digests=tuple(
                    sorted(digest for digest, record in records if record.consuming)
                ),
                scheduler_adopted_binding_digests=tuple(
                    sorted(
                        digest for digest, record in records if record.scheduler_adopted
                    )
                ),
                scheduler_finalized_binding_digests=tuple(
                    sorted(
                        digest
                        for digest, record in records
                        if record.scheduler_finalized
                    )
                ),
                cancelled_binding_digests=tuple(
                    sorted(digest for digest, record in records if record.cancelled)
                ),
                quarantined_binding_digests=tuple(
                    sorted(digest for digest, record in records if record.quarantined)
                ),
                adopted_count=self._adopted_count,
                cancelled_count=self._cancelled_count,
                fatal_inventory=self._fatal_inventory,
            )
        return inventory

    def _require_record(
        self,
        binding: TerminalRequestBinding,
    ) -> _DecodeSchedulerRecord:
        """Resolve one exact retained Python binding.

        :param binding: Candidate scheduler lifecycle identity.
        :returns: Exact retained scheduler record.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        with self._lock:
            self._require_healthy_locked()
            record = self._records.get(binding.digest)
        if record is None or record.registration.binding != binding:
            raise KeyError("decode scheduler registration is not retained")
        return record

    def _require_native_record(
        self,
        binding: NativeTerminalRequestBinding,
    ) -> _DecodeSchedulerRecord:
        """Resolve one exact retained native binding.

        :param binding: Owner action binding.
        :returns: Exact retained scheduler record.
        """

        if type(binding) is not NativeTerminalRequestBinding:
            raise TypeError("binding must be NativeTerminalRequestBinding")
        with self._lock:
            self._require_healthy_locked()
            record = self._records.get(binding.digest)
        if record is None:
            raise KeyError("decode adoption has no retained scheduler request")
        expected = NativeTerminalRequestBinding.from_binding(
            record.registration.binding
        )
        if binding != expected:
            raise RuntimeError("decode adoption differs from retained binding")
        return record

    def _quarantine_record(
        self,
        record: _DecodeSchedulerRecord,
        reason: str,
    ) -> None:
        """Retain transaction and scheduler state after ambiguity.

        :param record: Exact retained scheduler record.
        :param reason: Stable first fail-closed evidence.
        """

        record.quarantined = True
        if record.quarantine_callback_delivered:
            return
        if record.cancelled:
            # Transaction and scheduler resources already crossed their safe
            # unpublished rollback boundary. A later inbox bookkeeping failure
            # quarantines this lifecycle identity, not storage which has been
            # authoritatively returned to its allocators.
            record.quarantine_callback_delivered = True
            return
        registration = record.registration
        try:
            self._wiring.quarantine_transaction(registration.transaction, reason)
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode transaction quarantine failed:\n%s",
                traceback.format_exc(),
            )
        try:
            registration.quarantine_request(registration.request_owner, reason)
        except Exception:  # noqa: BLE001
            logger.error(
                "Decode scheduler request quarantine failed:\n%s",
                traceback.format_exc(),
            )
        record.quarantine_callback_delivered = True

    def _require_scheduler_thread(self) -> None:
        """Reject mutable request ownership from every other thread."""

        if threading.get_ident() != self._scheduler_thread_id:
            raise RuntimeError("decode scheduler resources crossed thread affinity")

    def _require_healthy_locked(self) -> None:
        """Reject lifecycle mutation after process-fatal entry."""

        if self._fatal_inventory is not None:
            raise RuntimeError("decode scheduler consumer is process-fatal")


class PackedTerminalDecodeServingComposition:
    """Pair decode request retention with the qualified scheduler inbox.

    Construction hides the otherwise circular relationship between the
    concrete decode consumer and :class:`TerminalSchedulerServing`. Request
    registration, native lifecycle binding, and unpublished cancellation are
    exposed only as paired operations, so allocation publication cannot race a
    missing scheduler identity and safe rollback cannot leave a live inbox key.
    """

    _consumer: PackedTerminalDecodeSchedulerConsumer
    _scheduler_serving: TerminalSchedulerServing

    def __init__(
        self,
        *,
        wiring: TerminalDecodeAdoptionWiring,
        physical_capacity: int,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
    ) -> None:
        """Construct one process-lifetime decode serving composition.

        :param wiring: Native decode lifecycle and transaction boundary.
        :param physical_capacity: Maximum configured in-flight generations.
        :param process_fatal_handler: Scheduler-owned admission-stop and teardown
            operation.
        """

        consumer = PackedTerminalDecodeSchedulerConsumer(
            wiring=wiring,
            process_fatal_handler=process_fatal_handler,
        )
        scheduler_serving = TerminalSchedulerServing(
            role=TerminalSchedulerServingRole.DECODE,
            physical_capacity=physical_capacity,
            decode_consumer=consumer,
        )
        self._consumer = consumer
        self._scheduler_serving = scheduler_serving

    @property
    def consumer(self) -> PackedTerminalDecodeSchedulerConsumer:
        """Return the concrete scheduler-affine request consumer.

        :returns: Process-lifetime decode consumer.
        """

        return self._consumer

    @property
    def scheduler_serving(self) -> TerminalSchedulerServing:
        """Return the qualified inbox adapter bound into the scheduler.

        :returns: Decode-role scheduler serving adapter.
        """

        return self._scheduler_serving

    def register(
        self,
        registration: PackedTerminalDecodeSchedulerRegistration,
    ) -> None:
        """Retain and bind one request before allocation publication.

        Native lifecycle binding occurs only after the exact mutable request is
        retained. Scheduler-inbox registration then completes the publication
        precondition. A failure before that point either leaves an explicit
        quarantine or executes the transaction's safe unpublished rollback.

        :param registration: Complete scheduler request ownership.
        """

        self._consumer.register_adoption(registration)
        self._consumer.bind_adoption(registration.binding)
        try:
            self._scheduler_serving.register_request(registration.binding)
        except Exception:
            try:
                self._consumer.begin_unpublished_cancellation(
                    registration.binding,
                    "scheduler inbox registration failed",
                )
                self._consumer.finish_unpublished_cancellation(registration.binding)
            except Exception:  # noqa: BLE001
                logger.error(
                    "Decode registration rollback failed closed:\n%s",
                    traceback.format_exc(),
                )
            raise

    def cancel_unpublished(
        self,
        binding: TerminalRequestBinding,
        reason: str,
    ) -> None:
        """Pair safe transaction rollback with scheduler-inbox retirement.

        :param binding: Exact unpublished decode lifecycle identity.
        :param reason: Stable cancellation evidence.
        """

        self._consumer.begin_unpublished_cancellation(binding, reason)
        try:
            self._scheduler_serving.cancel_unpublished_request(binding)
        except Exception:
            self._consumer.quarantine_registration(
                binding,
                "scheduler inbox cancellation failed after safe rollback",
            )
            logger.error(
                "Decode scheduler inbox cancellation failed closed:\n%s",
                traceback.format_exc(),
            )
            raise
        self._consumer.finish_unpublished_cancellation(binding)

    def inventory(
        self,
    ) -> tuple[
        PackedTerminalDecodeSchedulerInventory,
        SchedulerReceiptInboxInventory,
    ]:
        """Return paired request and scheduler-inbox conservation evidence.

        :returns: Concrete consumer inventory followed by qualified inbox state.
        """

        return (
            self._consumer.inventory(),
            self._scheduler_serving.inventory().inbox,
        )
