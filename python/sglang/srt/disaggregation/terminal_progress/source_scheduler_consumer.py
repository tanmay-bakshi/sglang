import dataclasses
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Protocol, runtime_checkable

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
from sglang.srt.disaggregation.terminal_progress.source_wiring import (
    PackedTerminalSourceSubmission,
)
from sglang.srt.disaggregation.terminal_progress.wire import (
    IssuedTerminalWireReceipt,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class TerminalSourceReclaimWiring(Protocol):
    """Source lifecycle boundary which consumes native reclaim authority."""

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
        release_resources: Callable[[PackedTerminalSourceSubmission], None],
    ) -> IssuedTerminalWireReceipt:
        """Release exact source resources under native authority.

        :param action: Owner-minted one-shot reclaim authority.
        :param release_resources: Scheduler-affine release operation.
        :returns: Exact reclaim-consumed receipt.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceSchedulerRelease:
    """One scheduler-affine source resource release operation.

    The callable may retain a mutable scheduler request, tree-cache ownership,
    and packed transport resources. This value is owned and invoked only by the
    scheduler thread. Owner and publisher threads see only the immutable
    request binding carried by their native actions.

    :ivar binding: Exact source lifecycle whose resources are retained.
    :ivar release_resources: One-shot scheduler-affine release operation.
    """

    binding: TerminalRequestBinding
    release_resources: Callable[[PackedTerminalSourceSubmission], None]

    def __post_init__(self) -> None:
        """Validate one exact scheduler release authority."""

        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if self.binding.owner.role is not TerminalOwnerRole.SOURCE:
            raise ValueError("source scheduler release requires a source binding")
        if not callable(self.release_resources):
            raise TypeError("release_resources must be callable")


@dataclasses.dataclass(frozen=True, slots=True)
class PackedTerminalSourceSchedulerInventory:
    """Complete source scheduler-release ownership inventory.

    :ivar active_binding_digests: Release operations retaining source resources.
    :ivar consuming_binding_digests: Operations executing on the scheduler.
    :ivar quarantined_binding_digests: Operations retained after ambiguity.
    :ivar resource_release_completed_binding_digests: Quarantined lifecycle
        identities whose resource release returned before a later failure.
    :ivar released_count: Successfully consumed release operations.
    :ivar fatal_inventory: Sticky scheduler-inbox failure evidence.
    """

    active_binding_digests: tuple[bytes, ...]
    consuming_binding_digests: tuple[bytes, ...]
    quarantined_binding_digests: tuple[bytes, ...]
    resource_release_completed_binding_digests: tuple[bytes, ...]
    released_count: int
    fatal_inventory: SchedulerReceiptInboxInventory | None

    def __post_init__(self) -> None:
        """Validate source release conservation and fail-closed retention."""

        populations = (
            self.active_binding_digests,
            self.consuming_binding_digests,
            self.quarantined_binding_digests,
            self.resource_release_completed_binding_digests,
        )
        if any(type(population) is not tuple for population in populations):
            raise TypeError("source scheduler inventory populations must be tuples")
        if any(
            type(digest) is not bytes or len(digest) != 32
            for population in populations
            for digest in population
        ):
            raise ValueError("source scheduler identities must contain 32 bytes")
        if any(tuple(sorted(population)) != population for population in populations):
            raise ValueError("source scheduler identities must use digest order")
        active = set(self.active_binding_digests)
        if not set(self.consuming_binding_digests).issubset(active):
            raise ValueError("consuming source identities must remain active")
        if not set(self.quarantined_binding_digests).issubset(active):
            raise ValueError("quarantined source identities must remain active")
        if not set(self.resource_release_completed_binding_digests).issubset(active):
            raise ValueError("completed source releases must remain active")
        if type(self.released_count) is not int or self.released_count < 0:
            raise ValueError("released_count must be a non-negative integer")
        if self.fatal_inventory is not None and (
            type(self.fatal_inventory) is not SchedulerReceiptInboxInventory
        ):
            raise TypeError("fatal_inventory must be scheduler inbox evidence")
        if self.fatal_inventory is not None:
            if self.fatal_inventory.fatal_cause is None:
                raise ValueError("fatal_inventory must carry a fatal cause")
            if set(self.quarantined_binding_digests) != active:
                raise ValueError("process-fatal source resources must all be retained")


@dataclasses.dataclass(slots=True)
class _SourceSchedulerReleaseRecord:
    """Mutable scheduler-thread state for one retained release operation."""

    release: PackedTerminalSourceSchedulerRelease
    consuming: bool = False
    quarantined: bool = False
    resources_released: bool = False


class PackedTerminalSourceSchedulerConsumer:
    """Consume source reclaim actions on their owning scheduler thread.

    Native progress transports only immutable actions into this component. The
    exact callback which captures mutable request and cache state never leaves
    the scheduler-owned registry. A failed or ambiguous consumption retains
    that callback and therefore retains every underlying resource fail closed.
    """

    _wiring: TerminalSourceReclaimWiring
    _process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None]
    _scheduler_thread_id: int
    _records: dict[bytes, _SourceSchedulerReleaseRecord]
    _released_count: int
    _fatal_inventory: SchedulerReceiptInboxInventory | None
    _lock: threading.Lock

    def __init__(
        self,
        *,
        wiring: TerminalSourceReclaimWiring,
        process_fatal_handler: Callable[[SchedulerReceiptInboxInventory], None],
    ) -> None:
        """Construct a scheduler-affine source reclaim consumer.

        :param wiring: Native source lifecycle and reclaim boundary.
        :param process_fatal_handler: Scheduler-owned stop-admission and teardown
            operation invoked after resources become quarantined.
        """

        if not isinstance(wiring, TerminalSourceReclaimWiring):
            raise TypeError("wiring must satisfy TerminalSourceReclaimWiring")
        if not callable(process_fatal_handler):
            raise TypeError("process_fatal_handler must be callable")
        self._wiring = wiring
        self._process_fatal_handler = process_fatal_handler
        self._scheduler_thread_id = threading.get_ident()
        self._records = {}
        self._released_count = 0
        self._fatal_inventory = None
        self._lock = threading.Lock()

    def register_release(
        self,
        release: PackedTerminalSourceSchedulerRelease,
    ) -> None:
        """Retain one exact release before native authority can arrive.

        :param release: Scheduler-owned lifecycle binding and release operation.
        """

        self._require_scheduler_thread()
        if type(release) is not PackedTerminalSourceSchedulerRelease:
            raise TypeError("release must be PackedTerminalSourceSchedulerRelease")
        digest = release.binding.digest
        with self._lock:
            self._require_healthy_locked()
            if digest in self._records:
                raise RuntimeError("source scheduler release identity was reused")
            self._records[digest] = _SourceSchedulerReleaseRecord(release=release)

    def cancel_unpublished(self, binding: TerminalRequestBinding) -> None:
        """Drop a release whose lifecycle cannot publish reclaim authority.

        This operation is valid only before native publication. The caller must
        pair it with removal from the scheduler receipt inbox. Once an owner
        action exists, only :meth:`consume_reclaim_authorized` may release or
        retire the record.

        :param binding: Exact unpublished source lifecycle.
        """

        self._require_scheduler_thread()
        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        with self._lock:
            self._require_healthy_locked()
            record = self._records.get(binding.digest)
            if record is None or record.release.binding != binding:
                raise KeyError("source scheduler release is not registered")
            if record.consuming or record.quarantined:
                raise RuntimeError("published or ambiguous source release cannot cancel")
            del self._records[binding.digest]

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release exact source resources under owner-minted authority.

        :param action: Exact reclaim authority transferred by the native owner.
        """

        self._require_scheduler_thread()
        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind is not NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED:
            raise ValueError("source scheduler consumer requires reclaim authority")
        digest = action.binding.digest
        with self._lock:
            self._require_healthy_locked()
            record = self._records.get(digest)
            if record is None:
                raise KeyError("reclaim authority has no retained source release")
            if record.consuming:
                raise RuntimeError("source reclaim authority is already consuming")
            if record.quarantined:
                raise RuntimeError("quarantined source resources cannot be released")
            expected = NativeTerminalRequestBinding.from_binding(
                record.release.binding
            )
            if action.binding != expected:
                raise RuntimeError("reclaim authority differs from retained binding")
            record.consuming = True

        release_called = False

        def release_resources(submission: PackedTerminalSourceSubmission) -> None:
            """Invoke the exact retained operation once on this scheduler."""

            nonlocal release_called
            self._require_scheduler_thread()
            if release_called:
                raise RuntimeError("source release operation was invoked twice")
            if type(submission) is not PackedTerminalSourceSubmission:
                raise TypeError("source release requires a packed submission")
            if submission.identity.local_binding != record.release.binding:
                raise RuntimeError("source release submission identity changed")
            release_called = True
            record.release.release_resources(submission)
            with self._lock:
                current = self._records.get(digest)
                if current is not record:
                    raise RuntimeError(
                        "source release registry changed during resource release"
                    )
                current.resources_released = True

        try:
            self._wiring.consume_reclaim_authorized(action, release_resources)
            if not release_called:
                raise RuntimeError("source wiring returned without releasing resources")
        except Exception:
            with self._lock:
                current = self._records.get(digest)
                if current is record:
                    current.consuming = False
                    current.quarantined = True
            logger.error(
                "Source scheduler reclaim failed closed:\n%s",
                traceback.format_exc(),
            )
            raise

        with self._lock:
            current = self._records.get(digest)
            if current is not record:
                raise RuntimeError("source release registry changed during consumption")
            del self._records[digest]
            self._released_count += 1

    def process_fatal(self, inventory: SchedulerReceiptInboxInventory) -> None:
        """Retain every ambiguous source resource and stop scheduler admission.

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
                    raise RuntimeError("source scheduler fatal evidence changed")
                return
            self._fatal_inventory = inventory
            for record in self._records.values():
                record.quarantined = True
        self._process_fatal_handler(inventory)

    def inventory(self) -> PackedTerminalSourceSchedulerInventory:
        """Return exact active, consuming, and quarantined ownership.

        :returns: Immutable scheduler-side source lifecycle inventory.
        """

        with self._lock:
            active = tuple(sorted(self._records))
            consuming = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.consuming
                )
            )
            quarantined = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.quarantined
                )
            )
            resource_release_completed = tuple(
                sorted(
                    digest
                    for digest, record in self._records.items()
                    if record.resources_released
                )
            )
            released_count = self._released_count
            fatal_inventory = self._fatal_inventory
        return PackedTerminalSourceSchedulerInventory(
            active_binding_digests=active,
            consuming_binding_digests=consuming,
            quarantined_binding_digests=quarantined,
            resource_release_completed_binding_digests=(
                resource_release_completed
            ),
            released_count=released_count,
            fatal_inventory=fatal_inventory,
        )

    def _require_scheduler_thread(self) -> None:
        """Reject mutable request ownership from every other thread."""

        if threading.get_ident() != self._scheduler_thread_id:
            raise RuntimeError("source scheduler resources crossed thread affinity")

    def _require_healthy_locked(self) -> None:
        """Reject new lifecycle mutation after process-fatal entry."""

        if self._fatal_inventory is not None:
            raise RuntimeError("source scheduler consumer is process-fatal")
