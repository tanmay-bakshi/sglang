import dataclasses
import enum
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.clock import SystemTerminalOwnerClock
from sglang.srt.disaggregation.terminal_progress.evidence import (
    TerminalProgressTimingRecorder,
    terminal_progress_timing_recorder,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import SchedulerInboxError
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalReceipt,
    NativeTerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerTimingField,
)
from sglang.srt.disaggregation.terminal_progress.scheduler_inbox import (
    SchedulerReceiptInboxFatalError,
    SchedulerReceiptInboxInventory,
    SchedulerReceiptPublishResult,
    TerminalReceiptInbox,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

LaunchResultT = TypeVar("LaunchResultT")
BoundLaunchResultT = TypeVar("BoundLaunchResultT")
logger = logging.getLogger(__name__)


class TerminalSchedulerServingRole(enum.StrEnum):
    """Scheduler-side terminal authority consumed by one process role."""

    SOURCE = "source"
    DECODE = "decode"


class TerminalSchedulerActionPublicationDisposition(enum.StrEnum):
    """Exact native-action owner after a failed scheduler publication."""

    CALLER_RETAINS = "caller_retains"
    SCHEDULER_RETAINS = "scheduler_retains"


@runtime_checkable
class TerminalSourceSchedulerConsumer(Protocol):
    """Scheduler-affine source resource and lifecycle boundary.

    The concrete source adapter retains its exact ``release_resources``
    operation before native publication, then delegates this call to
    ``PackedTerminalSourceWiring.consume_reclaim_authorized`` with both the
    action and retained operation. Returning means the scheduler-affine release
    and its reclaim-consumed authority both completed.
    """

    def consume_reclaim_authorized(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Release exact source resources under owner-minted authority.

        :param action: Exact reclaim action transferred by the native owner.
        """

    def process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Enter fail-closed source teardown on the scheduler thread.

        :param inventory: Complete sticky scheduler-inbox evidence.
        """


@runtime_checkable
class TerminalDecodeSchedulerConsumer(Protocol):
    """Scheduler-affine decode adoption and lifecycle boundary.

    The concrete decode adapter consumes the owner returned by the lower-level
    adoption call and inserts it into scheduler-owned request structures before
    this method returns. The generic inbox never owns or drops that mutable
    request value.
    """

    def consume_adoption_ready(
        self,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Adopt exact decode resources under owner-minted authority.

        :param action: Exact adoption action transferred by the native owner.
        """

    def process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Enter fail-closed decode teardown on the scheduler thread.

        :param inventory: Complete sticky scheduler-inbox evidence.
        """


@runtime_checkable
class TerminalSchedulerRuntimeFence(Protocol):
    """Native runtime fence required before scheduler-consumer teardown."""

    def wait_for_output_projection_quiescence(
        self,
        timeout_seconds: float,
    ) -> bool:
        """Fence native authority through Python output routing.

        :param timeout_seconds: Hash-bound shutdown-drain timeout.
        :returns: Whether native and projected output both became quiescent.
        """

    def begin_abort(self) -> None:
        """Enter fail-closed runtime drain while consumers remain alive."""


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSchedulerServingInventory:
    """Complete serving adapter and qualified inbox inventory.

    :ivar role: Scheduler process role owning these actions.
    :ivar inbox: Qualified bounded-inbox inventory.
    :ivar retained_action_ids: Native actions awaiting scheduler consumption.
    :ivar fatal_delivered: Whether fail-closed handling ran on the scheduler.
    """

    role: TerminalSchedulerServingRole
    inbox: SchedulerReceiptInboxInventory
    retained_action_ids: tuple[int, ...]
    fatal_delivered: bool

    def __post_init__(self) -> None:
        """Validate exact cross-layer action conservation."""

        if type(self.role) is not TerminalSchedulerServingRole:
            raise TypeError("role must be TerminalSchedulerServingRole")
        if type(self.inbox) is not SchedulerReceiptInboxInventory:
            raise TypeError("inbox must be SchedulerReceiptInboxInventory")
        if type(self.retained_action_ids) is not tuple or any(
            type(action_id) is not int or action_id < 0
            for action_id in self.retained_action_ids
        ):
            raise ValueError("retained_action_ids must contain unsigned integers")
        if len(set(self.retained_action_ids)) != len(self.retained_action_ids):
            raise ValueError("retained action identities must be unique")
        if len(self.retained_action_ids) > self.inbox.live_count:
            raise ValueError("retained scheduler actions exceed live requests")
        if type(self.fatal_delivered) is not bool:
            raise TypeError("fatal_delivered must be bool")


class TerminalSchedulerActionPublicationError(SchedulerReceiptInboxFatalError):
    """Typed failed publication with an unambiguous action disposition."""

    action: NativeTerminalOwnerAction
    disposition: TerminalSchedulerActionPublicationDisposition
    serving_inventory: TerminalSchedulerServingInventory

    def __init__(
        self,
        *,
        action: NativeTerminalOwnerAction,
        disposition: TerminalSchedulerActionPublicationDisposition,
        inventory: TerminalSchedulerServingInventory,
    ) -> None:
        """Create one exact failed-publication disposition.

        :param action: Native action whose publication failed.
        :param disposition: Component retaining the exact action after failure.
        :param inventory: Complete serving inventory after reconciliation.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if type(disposition) is not TerminalSchedulerActionPublicationDisposition:
            raise TypeError(
                "disposition must be TerminalSchedulerActionPublicationDisposition"
            )
        if type(inventory) is not TerminalSchedulerServingInventory:
            raise TypeError("inventory must be TerminalSchedulerServingInventory")
        cause = inventory.inbox.fatal_cause
        if cause is None:
            raise ValueError("failed scheduler publication must be process-fatal")
        scheduler_retains = action.action_id in inventory.retained_action_ids
        if scheduler_retains != (
            disposition
            is TerminalSchedulerActionPublicationDisposition.SCHEDULER_RETAINS
        ):
            raise ValueError("publication disposition differs from retained inventory")
        super().__init__(cause, inventory.inbox)
        self.action = action
        self.disposition = disposition
        self.serving_inventory = inventory
        self.args = (
            "terminal scheduler action publication failed: "
            f"{disposition.value} (cause={cause.value})",
        )

    @property
    def scheduler_retains_action(self) -> bool:
        """Return whether scheduler ownership linearized before failure.

        :returns: Whether the scheduler retains the exact native action.
        """

        return (
            self.disposition
            is TerminalSchedulerActionPublicationDisposition.SCHEDULER_RETAINS
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalSchedulerFailClosedClosure:
    """Take-once scheduler authority retained by fail-closed teardown.

    :ivar inventory: Final closed scheduler serving inventory.
    :ivar retained_actions: Exact actions transferred out of the adapter.
    """

    inventory: TerminalSchedulerServingInventory
    retained_actions: tuple[NativeTerminalOwnerAction, ...]

    def __post_init__(self) -> None:
        """Validate exact closure action conservation."""

        if type(self.inventory) is not TerminalSchedulerServingInventory:
            raise TypeError("inventory must be TerminalSchedulerServingInventory")
        if type(self.retained_actions) is not tuple or any(
            type(action) is not NativeTerminalOwnerAction
            for action in self.retained_actions
        ):
            raise TypeError("retained_actions must contain native actions")
        action_ids = tuple(action.action_id for action in self.retained_actions)
        if action_ids != tuple(sorted(action_ids)):
            raise ValueError("retained actions must use native identity order")
        if action_ids != self.inventory.retained_action_ids:
            raise ValueError("closure actions differ from retained inventory")


def _terminal_wire_receipt(receipt: NativeTerminalReceipt) -> TerminalWireReceipt:
    """Project one owner-minted receipt into the qualified inbox wire value.

    :param receipt: Exact native authority carried by a scheduler action.
    :returns: Canonical fixed-width scheduler receipt.
    """

    if type(receipt) is not NativeTerminalReceipt:
        raise TypeError("receipt must be NativeTerminalReceipt")
    return receipt.to_wire_receipt()


class TerminalSchedulerServing:
    """Bind native terminal actions to real scheduler launch boundaries.

    Owner threads publish immutable actions, but only the scheduler thread may
    mutate request, cache, or allocation state. This adapter retains each exact
    native action behind the qualified receipt inbox. The inbox orders receipt
    consumption against host submission and supplies the only fd used to wake
    an idle scheduler.
    """

    _role: TerminalSchedulerServingRole
    _source_consumer: TerminalSourceSchedulerConsumer | None
    _decode_consumer: TerminalDecodeSchedulerConsumer | None
    _inbox: TerminalReceiptInbox
    _timing: TerminalProgressTimingRecorder
    _actions_by_receipt: dict[bytes, NativeTerminalOwnerAction]
    _receipt_by_request: dict[PackedRequestKey, bytes]
    _fatal_delivered: bool
    _fail_closed_closure_taken: bool
    _lock: threading.Lock

    def __init__(
        self,
        *,
        role: TerminalSchedulerServingRole,
        physical_capacity: int,
        source_consumer: TerminalSourceSchedulerConsumer | None = None,
        decode_consumer: TerminalDecodeSchedulerConsumer | None = None,
    ) -> None:
        """Construct one role-specific serving adapter.

        :param role: Source reclaim or decode adoption role.
        :param physical_capacity: Maximum configured in-flight generations.
        :param source_consumer: Source scheduler-affine resource consumer.
        :param decode_consumer: Decode scheduler-affine resource consumer.
        """

        if type(role) is not TerminalSchedulerServingRole:
            raise TypeError("role must be TerminalSchedulerServingRole")
        if role is TerminalSchedulerServingRole.SOURCE:
            if not isinstance(source_consumer, TerminalSourceSchedulerConsumer):
                raise TypeError(
                    "source_consumer must satisfy TerminalSourceSchedulerConsumer"
                )
            if decode_consumer is not None:
                raise ValueError("source serving cannot carry a decode consumer")
        else:
            if not isinstance(decode_consumer, TerminalDecodeSchedulerConsumer):
                raise TypeError(
                    "decode_consumer must satisfy TerminalDecodeSchedulerConsumer"
                )
            if source_consumer is not None:
                raise ValueError("decode serving cannot carry a source consumer")
        self._role = role
        self._source_consumer = source_consumer
        self._decode_consumer = decode_consumer
        self._inbox = TerminalReceiptInbox(physical_capacity=physical_capacity)
        self._timing = terminal_progress_timing_recorder(
            logger,
            SystemTerminalOwnerClock().now_ns,
        )
        self._actions_by_receipt = {}
        self._receipt_by_request = {}
        self._fatal_delivered = False
        self._fail_closed_closure_taken = False
        self._lock = threading.Lock()

    @property
    def role(self) -> TerminalSchedulerServingRole:
        """Return the exact scheduler serving role.

        :returns: Source or decode role.
        """

        return self._role

    def fileno(self) -> int:
        """Return the qualified inbox wake descriptor.

        :returns: Open nonblocking read-side descriptor.
        """

        return self._inbox.fileno()

    def inventory(self) -> TerminalSchedulerServingInventory:
        """Return complete receipt and native-action conservation evidence.

        :returns: Immutable serving inventory.
        """

        with self._lock:
            action_ids = tuple(
                sorted(action.action_id for action in self._actions_by_receipt.values())
            )
            fatal_delivered = self._fatal_delivered
        return TerminalSchedulerServingInventory(
            role=self._role,
            inbox=self._inbox.inventory(),
            retained_action_ids=action_ids,
            fatal_delivered=fatal_delivered,
        )

    def register_request(self, binding: TerminalRequestBinding) -> None:
        """Register one exact scheduler generation before owner publication.

        :param binding: Exact scheduler-owned request binding.
        """

        self._require_binding_role(binding)
        with self._lock:
            self._inbox.register_live(binding)

    def register_native_request(
        self,
        binding: NativeTerminalRequestBinding,
    ) -> None:
        """Register one exact native-runtime request binding.

        :param binding: Native representation owned by the terminal runtime.
        """

        self.register_request(binding.to_binding())

    def cancel_unpublished_request(self, binding: TerminalRequestBinding) -> None:
        """Remove one request which can no longer receive an owner action.

        :param binding: Exact live request binding without receipt work.
        """

        self._require_binding_role(binding)
        with self._lock:
            self._inbox.unregister_live(binding)

    def publish_action(
        self,
        action: NativeTerminalOwnerAction,
    ) -> SchedulerReceiptPublishResult:
        """Publish one exact native scheduler authority and actively wake.

        :param action: Owner-minted reclaim or adoption action.
        :returns: Whether the qualified receipt queued or coalesced.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        wire_receipt: TerminalWireReceipt | None = None
        encoded: bytes | None = None
        request_key: PackedRequestKey | None = None
        inserted = False
        try:
            expected_kind = NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED
            if self._role is TerminalSchedulerServingRole.DECODE:
                expected_kind = NativeTerminalOwnerActionKind.ADOPTION_READY
            if action.kind is not expected_kind or action.receipt is None:
                raise ValueError("native action does not match scheduler serving role")
            wire_receipt = _terminal_wire_receipt(action.receipt)
            self._require_binding_role(wire_receipt.binding)
            encoded = wire_receipt.encode()
            request_key = wire_receipt.binding.request_key

            def retain_action() -> None:
                """Retain authority before making its receipt visible."""

                nonlocal inserted
                with self._lock:
                    retained_receipt = self._receipt_by_request.get(request_key)
                    if retained_receipt is not None and retained_receipt != encoded:
                        raise SchedulerInboxError(
                            "request generation maps to conflicting native actions"
                        )
                    existing = self._actions_by_receipt.get(encoded)
                    if existing is not None and existing != action:
                        raise SchedulerInboxError(
                            "scheduler receipt maps to conflicting native actions"
                        )
                    if existing is None:
                        self._actions_by_receipt[encoded] = action
                        self._receipt_by_request[request_key] = encoded
                        inserted = True

            if self._role is TerminalSchedulerServingRole.DECODE:
                self._timing.capture(
                    binding=wire_receipt.binding,
                    field=TerminalOwnerTimingField.SCHEDULER_INBOX_DELAY,
                    sample_key=f"decode-rank-{wire_receipt.binding.owner.tp_rank}",
                )
            return self._inbox.publish_after_retention(
                wire_receipt,
                retain_action,
            )
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Terminal scheduler action publication failed:\n%s",
                traceback.format_exc(),
            )
            if wire_receipt is not None:
                self._timing.discard_binding(wire_receipt.binding.digest)
            inventory = self._inbox.inventory()
            if encoded is not None and request_key is not None:
                self._discard_unpublished_action(
                    encoded=encoded,
                    action=action,
                    inserted=inserted,
                    request_key=request_key,
                    inventory=inventory,
                )
            self._enter_process_fatal()
            raise self._action_publication_error(action) from error

    def drain_at_loop_entry(self) -> tuple[NativeTerminalOwnerAction, ...]:
        """Consume every ready authority before scheduler batch selection.

        :returns: Exact FIFO native actions successfully consumed.
        """

        consumed: list[NativeTerminalOwnerAction] = []

        def consume(receipt: TerminalWireReceipt) -> None:
            consumed.append(self._consume_receipt(receipt))

        try:
            self._inbox.drain_at_loop_entry(consume)
        except SchedulerReceiptInboxFatalError as error:
            self._deliver_process_fatal(error.inventory)
            raise
        except Exception:
            inventory = self._inbox.inventory()
            if inventory.fatal_cause is not None:
                self._deliver_process_fatal(inventory)
            logger.error(
                "Terminal scheduler loop-entry consumption failed:\n%s",
                traceback.format_exc(),
            )
            raise
        return tuple(consumed)

    def launch_handoff(
        self,
        submit: Callable[[], LaunchResultT],
    ) -> LaunchResultT:
        """Order terminal consumption against one host forward submission.

        :param submit: Narrow callback returning after host submission.
        :returns: Exact launch result.
        """

        if not callable(submit):
            raise TypeError("submit must be callable")
        try:
            return self._inbox.launch_handoff(
                submit=submit,
                consume=self._consume_receipt,
            )
        except SchedulerReceiptInboxFatalError as error:
            self._deliver_process_fatal(error.inventory)
            raise
        except Exception:
            inventory = self._inbox.inventory()
            if inventory.fatal_cause is not None:
                self._deliver_process_fatal(inventory)
            logger.error(
                "Terminal scheduler launch handoff failed:\n%s",
                traceback.format_exc(),
            )
            raise

    def launch_and_bind_handoff(
        self,
        submit: Callable[[], LaunchResultT],
        bind: Callable[[LaunchResultT], BoundLaunchResultT],
    ) -> BoundLaunchResultT:
        """Submit and bind immutable terminal ownership inside one handoff.

        ``bind`` is the only post-submit work allowed inside the gate. It must
        freeze the exact result, enqueue owner-visible device work, and return;
        mutable scheduler result processing remains outside this method.

        :param submit: Narrow model-worker submission callback.
        :param bind: Immediate immutable terminal-submission binder.
        :returns: Exact value returned by ``bind``.
        """

        if not callable(submit):
            raise TypeError("submit must be callable")
        if not callable(bind):
            raise TypeError("bind must be callable")
        try:
            return self._inbox.launch_and_bind_handoff(
                submit=submit,
                bind=bind,
                consume=self._consume_receipt,
            )
        except SchedulerReceiptInboxFatalError as error:
            self._deliver_process_fatal(error.inventory)
            raise
        except Exception:
            inventory = self._inbox.inventory()
            if inventory.fatal_cause is not None:
                self._deliver_process_fatal(inventory)
            logger.error(
                "Terminal scheduler launch binding failed:\n%s",
                traceback.format_exc(),
            )
            raise

    def mark_owner_dead(self) -> SchedulerReceiptInboxInventory:
        """Wake the scheduler after native owner or publisher death.

        :returns: Complete sticky fatal inbox evidence.
        """

        return self._inbox.mark_owner_dead()

    def fence_runtime_teardown(
        self,
        runtime: TerminalSchedulerRuntimeFence,
        timeout_seconds: float,
    ) -> None:
        """Fence native output before scheduler-consumer teardown.

        A failed fence begins native fail-closed drain but deliberately keeps
        this adapter and its wake descriptors alive. Final quarantine actions
        still need their scheduler consumer before a higher-level owner may
        close the runtime and this adapter.

        :param runtime: Authoritative native runtime owning output projection.
        :param timeout_seconds: Hash-bound shutdown-drain timeout.
        :raises RuntimeError: If projection does not become quiescent.
        """

        if not isinstance(runtime, TerminalSchedulerRuntimeFence):
            raise TypeError("runtime must satisfy TerminalSchedulerRuntimeFence")
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        try:
            quiescent = runtime.wait_for_output_projection_quiescence(timeout_seconds)
        except Exception:
            runtime.begin_abort()
            logger.error(
                "Terminal runtime projection fence failed:\n%s",
                traceback.format_exc(),
            )
            raise
        if quiescent:
            return
        runtime.begin_abort()
        raise RuntimeError(
            "terminal runtime projection fence timed out and began fail-closed drain"
        )

    def close(self) -> None:
        """Close a clean serving adapter with no retained native action."""

        with self._lock:
            if len(self._actions_by_receipt) > 0 or len(self._receipt_by_request) > 0:
                raise RuntimeError("scheduler serving retains native actions")
        self._inbox.close()

    def close_fail_closed(self) -> TerminalSchedulerFailClosedClosure:
        """Close and transfer retained scheduler authority exactly once.

        :returns: Immutable final inventory and retained action authority.
        """

        with self._lock:
            if self._fail_closed_closure_taken:
                raise RuntimeError("scheduler fail-closed closure was already taken")
            self._fail_closed_closure_taken = True
        inventory = self._inbox.inventory()
        if inventory.fatal_cause is None:
            inventory = self._inbox.mark_owner_dead()
        try:
            self._deliver_process_fatal(inventory)
        finally:
            self._inbox.close_fail_closed()
        inbox_inventory = self._inbox.inventory()
        with self._lock:
            retained_actions = tuple(
                sorted(
                    self._actions_by_receipt.values(),
                    key=lambda action: action.action_id,
                )
            )
            retained_action_ids = tuple(action.action_id for action in retained_actions)
            closure_inventory = TerminalSchedulerServingInventory(
                role=self._role,
                inbox=inbox_inventory,
                retained_action_ids=retained_action_ids,
                fatal_delivered=self._fatal_delivered,
            )
            self._actions_by_receipt.clear()
            self._receipt_by_request.clear()
        return TerminalSchedulerFailClosedClosure(
            inventory=closure_inventory,
            retained_actions=retained_actions,
        )

    def _consume_receipt(
        self,
        receipt: TerminalWireReceipt,
    ) -> NativeTerminalOwnerAction:
        """Dispatch one qualified receipt to its role-specific scheduler owner.

        :param receipt: Exact receipt selected by the launch handoff.
        :returns: Matching native action after successful side effects.
        """

        encoded = receipt.encode()
        with self._lock:
            action = self._actions_by_receipt.get(encoded)
        if action is None:
            raise RuntimeError("scheduler receipt has no retained native action")
        if self._role is TerminalSchedulerServingRole.DECODE:
            self._timing.complete(
                binding=receipt.binding,
                field=TerminalOwnerTimingField.SCHEDULER_INBOX_DELAY,
                sample_key=f"decode-rank-{receipt.binding.owner.tp_rank}",
            )
        if self._role is TerminalSchedulerServingRole.SOURCE:
            consumer = self._source_consumer
            if consumer is None:
                raise RuntimeError("source scheduler consumer is unavailable")
            consumer.consume_reclaim_authorized(action)
        else:
            consumer = self._decode_consumer
            if consumer is None:
                raise RuntimeError("decode scheduler consumer is unavailable")
            consumer.consume_adoption_ready(action)
        with self._lock:
            current = self._actions_by_receipt.get(encoded)
            if current != action:
                raise RuntimeError("native scheduler action changed during consumption")
            request_key = receipt.binding.request_key
            if self._receipt_by_request.get(request_key) != encoded:
                raise RuntimeError(
                    "native scheduler request changed during consumption"
                )
            del self._actions_by_receipt[encoded]
            del self._receipt_by_request[request_key]
        return action

    def _discard_unpublished_action(
        self,
        *,
        encoded: bytes,
        action: NativeTerminalOwnerAction,
        inserted: bool,
        request_key: PackedRequestKey,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Discard authority only when its receipt never became visible.

        :param encoded: Canonical receipt bytes used as the retention key.
        :param action: Exact action retained by this publication attempt.
        :param inserted: Whether this attempt created the retained entry.
        :param request_key: Exact request generation targeted by the receipt.
        :param inventory: Inbox evidence after the failed publication.
        """

        if not inserted:
            return
        active_keys = {
            *inventory.pending_request_keys,
            *inventory.consuming_request_keys,
        }
        if request_key in active_keys:
            return
        with self._lock:
            current = self._actions_by_receipt.get(encoded)
            if current != action:
                raise RuntimeError(
                    "native scheduler action changed during publication rollback"
                )
            if self._receipt_by_request.get(request_key) != encoded:
                raise RuntimeError(
                    "native scheduler request changed during publication rollback"
                )
            del self._actions_by_receipt[encoded]
            del self._receipt_by_request[request_key]

    def _deliver_process_fatal(
        self,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Invoke the role-specific fail-closed boundary exactly once.

        :param inventory: Complete sticky fatal inbox evidence.
        """

        with self._lock:
            if self._fatal_delivered:
                return
            self._fatal_delivered = True
        if self._role is TerminalSchedulerServingRole.SOURCE:
            consumer = self._source_consumer
            if consumer is None:
                raise RuntimeError("source scheduler consumer is unavailable")
            consumer.process_fatal(inventory)
            return
        consumer = self._decode_consumer
        if consumer is None:
            raise RuntimeError("decode scheduler consumer is unavailable")
        consumer.process_fatal(inventory)

    def _enter_process_fatal(self) -> None:
        """Enter the qualified inbox's shared sticky owner-death path."""

        inventory = self._inbox.inventory()
        if inventory.fatal_cause is None:
            self._inbox.mark_owner_dead()

    def _fatal_error(self) -> SchedulerReceiptInboxFatalError:
        """Build one typed error from the sticky qualified inbox state.

        :returns: Complete process-fatal scheduler error.
        """

        inventory = self._inbox.inventory()
        cause = inventory.fatal_cause
        if cause is None:
            raise RuntimeError("scheduler serving has no process-fatal cause")
        return SchedulerReceiptInboxFatalError(cause, inventory)

    def _action_publication_error(
        self,
        action: NativeTerminalOwnerAction,
    ) -> TerminalSchedulerActionPublicationError:
        """Build one exact post-reconciliation publication disposition.

        :param action: Native action supplied to the failed publication.
        :returns: Typed failure naming its current authority owner.
        """

        with self._lock:
            scheduler_retains = any(
                retained_action == action
                for retained_action in self._actions_by_receipt.values()
            )
        disposition = TerminalSchedulerActionPublicationDisposition.CALLER_RETAINS
        if scheduler_retains:
            disposition = (
                TerminalSchedulerActionPublicationDisposition.SCHEDULER_RETAINS
            )
        return TerminalSchedulerActionPublicationError(
            action=action,
            disposition=disposition,
            inventory=self.inventory(),
        )

    def _require_binding_role(self, binding: TerminalRequestBinding) -> None:
        """Require one binding to belong to this scheduler process role.

        :param binding: Candidate exact request binding.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        expected = TerminalOwnerRole.SOURCE
        if self._role is TerminalSchedulerServingRole.DECODE:
            expected = TerminalOwnerRole.DECODE
        if binding.owner.role is not expected:
            raise ValueError("terminal scheduler binding belongs to another role")
