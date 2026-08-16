import collections
import dataclasses
import enum
import errno
import os
import threading
from collections.abc import Callable
from typing import TypeVar

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.inbox import (
    SchedulerInboxError,
    SchedulerInboxFatalCause,
)
from sglang.srt.disaggregation.terminal_progress.wire import TerminalWireReceipt

LaunchResultT = TypeVar("LaunchResultT")
BoundLaunchResultT = TypeVar("BoundLaunchResultT")


class SchedulerReceiptPublishResult(enum.StrEnum):
    """Disposition of one receipt publication attempt."""

    QUEUED = "queued"
    COALESCED = "coalesced"


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerDeliveryIntent:
    """One exact launch exclusion owned by an external delivery path.

    :ivar identity: Monotonic process-local intent identity.
    :ivar binding: Exact request generation excluded from another host launch.
    """

    identity: int
    binding: TerminalRequestBinding

    def __post_init__(self) -> None:
        """Validate one process-local delivery intent."""

        if type(self.identity) is not int or self.identity < 0:
            raise ValueError("delivery intent identity must be non-negative")
        if type(self.binding) is not TerminalRequestBinding:
            raise TypeError("delivery intent binding must be TerminalRequestBinding")


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerReceiptInboxInventory:
    """Immutable scheduler receipt-inbox inventory.

    :ivar physical_capacity: Maximum registered live request generations.
    :ivar live_bindings: Exact request generations awaiting scheduler handling.
    :ivar pending_request_keys: FIFO request generations queued for handling.
    :ivar consuming_request_keys: Request generations inside the consumer call.
    :ivar outstanding_publications: Producer calls announced but not returned.
    :ivar active_delivery_intents: Exact external launch exclusions. An intent
        may outlive its scheduler-live generation after reclaim consumption and
        until the independent delivery owner commits its terminal milestone.
    :ivar wake_armed: Whether the readable fd carries an unconsumed wake hint.
    :ivar fatal_cause: First process-fatal invariant violation, when present.
    :ivar closed: Whether the runtime descriptors are closed.
    """

    physical_capacity: int
    live_bindings: tuple[TerminalRequestBinding, ...]
    pending_request_keys: tuple[PackedRequestKey, ...]
    consuming_request_keys: tuple[PackedRequestKey, ...]
    outstanding_publications: int
    active_delivery_intents: tuple[SchedulerDeliveryIntent, ...]
    wake_armed: bool
    fatal_cause: SchedulerInboxFatalCause | None
    closed: bool

    def __post_init__(self) -> None:
        """Validate the complete runtime inventory."""

        if type(self.physical_capacity) is not int or self.physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        if type(self.live_bindings) is not tuple:
            raise TypeError("live_bindings must be a tuple")
        if type(self.pending_request_keys) is not tuple:
            raise TypeError("pending_request_keys must be a tuple")
        if type(self.consuming_request_keys) is not tuple:
            raise TypeError("consuming_request_keys must be a tuple")
        if (
            type(self.outstanding_publications) is not int
            or self.outstanding_publications < 0
        ):
            raise ValueError("outstanding_publications must be non-negative")
        if type(self.active_delivery_intents) is not tuple or any(
            type(intent) is not SchedulerDeliveryIntent
            for intent in self.active_delivery_intents
        ):
            raise TypeError(
                "active_delivery_intents must contain SchedulerDeliveryIntent values"
            )
        if type(self.wake_armed) is not bool:
            raise TypeError("wake_armed must be bool")
        if (
            self.fatal_cause is not None
            and type(self.fatal_cause) is not SchedulerInboxFatalCause
        ):
            raise TypeError("fatal_cause must be SchedulerInboxFatalCause")
        if type(self.closed) is not bool:
            raise TypeError("closed must be bool")
        live_keys = tuple(binding.request_key for binding in self.live_bindings)
        if len(set(live_keys)) != len(live_keys):
            raise ValueError("live request generations must be unique")
        pending_keys = self.pending_request_keys
        consuming_keys = self.consuming_request_keys
        if len(set(pending_keys)) != len(pending_keys):
            raise ValueError("pending request generations must be unique")
        if len(set(consuming_keys)) != len(consuming_keys):
            raise ValueError("consuming request generations must be unique")
        if len(set(pending_keys) & set(consuming_keys)) > 0:
            raise ValueError("pending and consuming generations overlap")
        if not set((*pending_keys, *consuming_keys)).issubset(set(live_keys)):
            raise ValueError("every queued or consuming generation must be live")
        if len(pending_keys) > len(live_keys):
            raise ValueError("pending receipts exceed live requests")
        if len(live_keys) > self.physical_capacity:
            raise ValueError("live requests exceed physical capacity")
        delivery_identities = tuple(
            intent.identity for intent in self.active_delivery_intents
        )
        delivery_keys = tuple(
            intent.binding.request_key for intent in self.active_delivery_intents
        )
        if len(set(delivery_identities)) != len(delivery_identities):
            raise ValueError("delivery intent identities must be unique")
        if len(set(delivery_keys)) != len(delivery_keys):
            raise ValueError("delivery request generations must be unique")
        if len(delivery_keys) > self.physical_capacity:
            raise ValueError("delivery intents exceed physical capacity")

    @property
    def live_count(self) -> int:
        """Return the exact live request-generation count.

        :returns: Number of live scheduler request generations.
        """

        return len(self.live_bindings)

    @property
    def pending_count(self) -> int:
        """Return the exact pending receipt count.

        :returns: Number of receipts waiting for scheduler consumption.
        """

        return len(self.pending_request_keys)


class SchedulerReceiptInboxFatalError(SchedulerInboxError):
    """Sticky process-fatal scheduler inbox disposition."""

    cause: SchedulerInboxFatalCause
    inventory: SchedulerReceiptInboxInventory

    def __init__(
        self,
        cause: SchedulerInboxFatalCause,
        inventory: SchedulerReceiptInboxInventory,
    ) -> None:
        """Create one exact process-fatal error.

        :param cause: First fatal scheduler inbox cause.
        :param inventory: Complete fail-closed runtime inventory.
        """

        if type(cause) is not SchedulerInboxFatalCause:
            raise TypeError("cause must be SchedulerInboxFatalCause")
        if type(inventory) is not SchedulerReceiptInboxInventory:
            raise TypeError("inventory must be SchedulerReceiptInboxInventory")
        self.cause = cause
        self.inventory = inventory
        super().__init__(f"scheduler receipt inbox is process-fatal: {cause.value}")


@dataclasses.dataclass(frozen=True, slots=True)
class _PendingSchedulerReceipt:
    """Canonical receipt bytes retained beside their decoded value.

    :ivar receipt: Decoded immutable scheduler receipt.
    :ivar encoded: Canonical fixed-width receipt bytes.
    """

    receipt: TerminalWireReceipt
    encoded: bytes

    def __post_init__(self) -> None:
        """Require the bytes to be the receipt's canonical representation."""

        if type(self.receipt) is not TerminalWireReceipt:
            raise TypeError("receipt must be TerminalWireReceipt")
        if type(self.encoded) is not bytes:
            raise TypeError("encoded must be bytes")
        if self.receipt.encode() != self.encoded:
            raise ValueError("encoded bytes are not canonical for receipt")

    @property
    def request_key(self) -> PackedRequestKey:
        """Return the receipt's exact request-generation key.

        :returns: Stable packed request-generation identity.
        """

        return self.receipt.binding.request_key


class TerminalReceiptInbox:
    """Fd-backed one-receipt-per-generation scheduler runtime.

    Producers announce publication before entering the launch gate, append the
    canonical receipt before signalling the fd, and retain that announcement
    until publication returns. The scheduler snapshots announcements that race
    with host submission and waits only for those CPU publications before its
    post-launch drain. No operation waits for an unrelated CUDA forward.
    """

    _physical_capacity: int
    _read_fd: int
    _write_fd: int
    _live: dict[PackedRequestKey, TerminalRequestBinding]
    _pending: collections.deque[_PendingSchedulerReceipt]
    _active_encoded: dict[PackedRequestKey, bytes]
    _consuming: dict[PackedRequestKey, _PendingSchedulerReceipt]
    _wake_armed: bool
    _fatal_cause: SchedulerInboxFatalCause | None
    _closed: bool
    _state_lock: threading.Lock
    _consumer_lock: threading.Lock
    _intent_condition: threading.Condition
    _next_intent: int
    _active_intents: set[int]
    _active_delivery_intents: dict[int, SchedulerDeliveryIntent]

    def __init__(self, physical_capacity: int) -> None:
        """Create one bounded runtime and its coalesced wake pipe.

        :param physical_capacity: Maximum configured in-flight generations.
        """

        if type(physical_capacity) is not int or physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        self._physical_capacity = physical_capacity
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._live = {}
        self._pending = collections.deque()
        self._active_encoded = {}
        self._consuming = {}
        self._wake_armed = False
        self._fatal_cause = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._consumer_lock = threading.Lock()
        self._intent_condition = threading.Condition(threading.Lock())
        self._next_intent = 0
        self._active_intents = set()
        self._active_delivery_intents = {}

    def fileno(self) -> int:
        """Return the scheduler's readable wake descriptor.

        :returns: Open nonblocking read-side descriptor.
        :raises SchedulerInboxError: If the inbox is closed.
        """

        with self._state_lock:
            self._require_open_locked()
            return self._read_fd

    def inventory(self) -> SchedulerReceiptInboxInventory:
        """Return complete live, queued, and process disposition evidence.

        :returns: Immutable runtime inventory.
        """

        with self._state_lock:
            return self._inventory_locked()

    def register_live(self, binding: TerminalRequestBinding) -> None:
        """Register one exact generation before its receipt may arrive.

        :param binding: Exact scheduler-owned request binding.
        :raises SchedulerInboxError: If the generation conflicts or the
            configured in-flight bound is exhausted.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        fatal_error: SchedulerReceiptInboxFatalError | None = None
        with self._state_lock:
            self._require_operational_locked()
            current = self._live.get(binding.request_key)
            if current is not None:
                if current == binding:
                    return
                raise SchedulerInboxError(
                    "request generation is registered to another binding"
                )
            if len(self._live) >= self._physical_capacity:
                self._enter_process_fatal_locked(
                    SchedulerInboxFatalCause.PHYSICAL_CAPACITY
                )
                fatal_error = self._fatal_error_locked()
            else:
                self._live[binding.request_key] = binding
                self._validate_bounds_locked()
        if fatal_error is not None:
            raise fatal_error

    def unregister_live(self, binding: TerminalRequestBinding) -> None:
        """Cancel one live generation which has no scheduler receipt.

        :param binding: Exact live binding to remove.
        :raises SchedulerInboxError: If the binding is absent or has receipt
            work pending or consuming.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        with self._state_lock:
            self._require_operational_locked()
            current = self._live.get(binding.request_key)
            if current is None or current != binding:
                raise SchedulerInboxError("request binding is not live")
            if binding.request_key in self._active_encoded:
                raise SchedulerInboxError(
                    "cannot unregister a request with active receipt work"
                )
            del self._live[binding.request_key]
            self._validate_bounds_locked()

    def publish(
        self,
        receipt: TerminalWireReceipt,
    ) -> SchedulerReceiptPublishResult:
        """Publish one canonical receipt and then signal scheduler readiness.

        Byte-identical retransmissions coalesce. Any other receipt for the same
        live request generation enters the shared process-fatal path before it
        can alter the queue.

        :param receipt: Canonical fixed-width scheduler receipt.
        :returns: Whether the receipt was queued or coalesced.
        :raises SchedulerReceiptInboxFatalError: If a bound or duplicate
            invariant enters process-fatal disposition.
        :raises SchedulerInboxError: If the receipt does not target a live
            exact binding.
        """

        return self.publish_after_retention(receipt, lambda: None)

    def publish_after_retention(
        self,
        receipt: TerminalWireReceipt,
        retain: Callable[[], None],
    ) -> SchedulerReceiptPublishResult:
        """Publish after retaining scheduler-affine authority.

        The publication intent is announced before ``retain`` runs, and the
        receipt cannot become visible until ``retain`` returns successfully.
        This lets adapters retain an opaque native action without opening a
        race between host submission and scheduler consumption.

        :param receipt: Canonical fixed-width scheduler receipt.
        :param retain: Nonblocking callback which retains matching authority.
        :returns: Whether the receipt was queued or coalesced.
        :raises SchedulerReceiptInboxFatalError: If a bound or duplicate
            invariant enters process-fatal disposition.
        :raises SchedulerInboxError: If the receipt does not target a live
            exact binding.
        """

        if type(receipt) is not TerminalWireReceipt:
            raise TypeError("receipt must be TerminalWireReceipt")
        if not callable(retain):
            raise TypeError("retain must be callable")
        intent = self._begin_publication_intent()
        fatal_error: SchedulerReceiptInboxFatalError | None = None
        result: SchedulerReceiptPublishResult | None = None
        try:
            retain()
            encoded = receipt.encode()
            request_key = receipt.binding.request_key
            with self._state_lock:
                self._require_operational_locked()
                binding = self._live.get(request_key)
                if binding is None or binding != receipt.binding:
                    raise SchedulerInboxError(
                        "receipt does not bind to an exact live request generation"
                    )
                active_encoded = self._active_encoded.get(request_key)
                if active_encoded is not None:
                    if active_encoded == encoded:
                        result = SchedulerReceiptPublishResult.COALESCED
                    else:
                        self._enter_process_fatal_locked(
                            SchedulerInboxFatalCause.CONFLICTING_DUPLICATE
                        )
                        fatal_error = self._fatal_error_locked()
                elif len(self._pending) >= self._physical_capacity:
                    self._enter_process_fatal_locked(
                        SchedulerInboxFatalCause.PHYSICAL_CAPACITY
                    )
                    fatal_error = self._fatal_error_locked()
                elif len(self._pending) >= len(self._live):
                    self._enter_process_fatal_locked(
                        SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT
                    )
                    fatal_error = self._fatal_error_locked()
                else:
                    publication = _PendingSchedulerReceipt(
                        receipt=receipt,
                        encoded=encoded,
                    )
                    self._pending.append(publication)
                    self._active_encoded[request_key] = encoded
                    self._validate_bounds_locked()
                    self._signal_locked()
                    if self._fatal_cause is None:
                        result = SchedulerReceiptPublishResult.QUEUED
                    else:
                        fatal_error = self._fatal_error_locked()
        finally:
            self._complete_publication_intent(intent)
        if fatal_error is not None:
            raise fatal_error
        if result is None:
            raise SchedulerInboxError("receipt publication produced no disposition")
        return result

    def begin_delivery_intent(
        self,
        binding: TerminalRequestBinding,
    ) -> SchedulerDeliveryIntent:
        """Exclude a host launch while an external delivery remains causal.

        This boundary deliberately touches only the publication-intent
        condition. A delivery owner may call it while a native pending call
        has interrupted the scheduler inside the receipt state lock; taking
        that lock here would recreate the handoff deadlock this intent avoids.

        :param binding: Exact request generation protected by the intent.
        :returns: Take-once intent consumed after durable delivery.
        """

        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        with self._intent_condition:
            if len(self._active_delivery_intents) >= self._physical_capacity:
                raise SchedulerInboxError(
                    "delivery intents exceed scheduler physical capacity"
                )
            if any(
                intent.binding.request_key == binding.request_key
                for intent in self._active_delivery_intents.values()
            ):
                raise SchedulerInboxError(
                    "request generation already owns a delivery intent"
                )
            identity = self._next_intent
            self._next_intent += 1
            intent = SchedulerDeliveryIntent(identity=identity, binding=binding)
            self._active_intents.add(identity)
            self._active_delivery_intents[identity] = intent
            return intent

    def complete_delivery_intent(self, intent: SchedulerDeliveryIntent) -> None:
        """Release one external launch exclusion after durable delivery.

        :param intent: Exact active delivery intent.
        """

        if type(intent) is not SchedulerDeliveryIntent:
            raise TypeError("intent must be SchedulerDeliveryIntent")
        with self._intent_condition:
            active = self._active_delivery_intents.get(intent.identity)
            if active != intent or intent.identity not in self._active_intents:
                raise SchedulerInboxError(
                    "delivery intent is absent, forged, or already completed"
                )
            del self._active_delivery_intents[intent.identity]
            self._active_intents.remove(intent.identity)
            self._intent_condition.notify_all()

    def drain_at_loop_entry(
        self,
        consume: Callable[[TerminalWireReceipt], None],
    ) -> tuple[TerminalWireReceipt, ...]:
        """Consume every receipt ready before scheduler batch selection.

        :param consume: Scheduler-affine receipt consumer.
        :returns: Exact FIFO receipts successfully consumed.
        """

        if not callable(consume):
            raise TypeError("consume must be callable")
        with self._consumer_lock:
            return self._drain_ready(consume)

    def launch_handoff(
        self,
        submit: Callable[[], LaunchResultT],
        consume: Callable[[TerminalWireReceipt], None],
    ) -> LaunchResultT:
        """Drain both sides of one serialized host launch handoff.

        The ``submit`` callback must return immediately after host submission;
        it must not wait for forward completion. Publications announced before
        that return are completed and drained before this method returns.

        :param submit: Narrow host-submission callback.
        :param consume: Scheduler-affine receipt consumer.
        :returns: Exact result returned by ``submit``.
        """

        def retain_result(result: LaunchResultT) -> LaunchResultT:
            """Return an unmodified launch result.

            :param result: Exact host-submission result.
            :returns: The same result.
            """

            return result

        return self._launch_and_bind_handoff(
            submit=submit,
            bind=retain_result,
            consume=consume,
        )

    def launch_and_bind_handoff(
        self,
        submit: Callable[[], LaunchResultT],
        bind: Callable[[LaunchResultT], BoundLaunchResultT],
        consume: Callable[[TerminalWireReceipt], None],
    ) -> BoundLaunchResultT:
        """Bind post-submit ownership inside the serialized handoff.

        Binding runs after the state mutex protecting host submission is
        released, because the binder registers the new live generation through
        this inbox. The scheduler-consumer mutex remains held until binding,
        racing publication completion, and receipt draining all finish.

        :param submit: Narrow host-submission callback.
        :param bind: Immediate immutable post-submission binder.
        :param consume: Scheduler-affine receipt consumer.
        :returns: Exact result returned by ``bind``.
        """

        return self._launch_and_bind_handoff(
            submit=submit,
            bind=bind,
            consume=consume,
        )

    def _launch_and_bind_handoff(
        self,
        *,
        submit: Callable[[], LaunchResultT],
        bind: Callable[[LaunchResultT], BoundLaunchResultT],
        consume: Callable[[TerminalWireReceipt], None],
    ) -> BoundLaunchResultT:
        """Implement one submission, binding, and receipt-consumption gate.

        :param submit: Narrow host-submission callback.
        :param bind: Immediate immutable post-submission binder.
        :param consume: Scheduler-affine receipt consumer.
        :returns: Exact result returned by ``bind``.
        """

        if not callable(submit):
            raise TypeError("submit must be callable")
        if not callable(bind):
            raise TypeError("bind must be callable")
        if not callable(consume):
            raise TypeError("consume must be callable")
        with self._consumer_lock:
            while True:
                publication: _PendingSchedulerReceipt | None = None
                publication_intents: frozenset[int]
                with self._state_lock:
                    self._require_operational_locked()
                    self._clear_wake_locked()
                    publication_intents = self._publication_intent_snapshot()
                    if len(publication_intents) == 0:
                        if len(self._pending) > 0:
                            publication = self._take_next_locked()
                        else:
                            launch_result = submit()
                            publication_target = self._publication_intent_snapshot()
                            break
                if len(publication_intents) > 0:
                    self._wait_for_publication_intents(publication_intents)
                    continue
                if publication is not None:
                    self._consume_publication(publication, consume)
            bound_result = bind(launch_result)
            publication_target = publication_target.union(
                self._publication_intent_snapshot()
            )
            self._wait_for_publication_intents(publication_target)
            self._drain_ready(consume)
            return bound_result

    def mark_owner_dead(self) -> SchedulerReceiptInboxInventory:
        """Wake the scheduler into sticky process-fatal owner-death state.

        :returns: Complete fatal runtime inventory.
        """

        with self._state_lock:
            self._require_open_locked()
            self._enter_process_fatal_locked(SchedulerInboxFatalCause.OWNER_DEATH)
            return self._inventory_locked()

    def close(self) -> None:
        """Close a clean, quiescent scheduler receipt inbox.

        :raises SchedulerInboxError: If any live, queued, consuming, or
            publishing identity remains.
        """

        with self._consumer_lock:
            with self._state_lock:
                if self._closed:
                    return
                with self._intent_condition:
                    outstanding_publications = len(self._active_intents)
                if (
                    len(self._live) > 0
                    or len(self._pending) > 0
                    or len(self._consuming) > 0
                    or outstanding_publications > 0
                ):
                    raise SchedulerInboxError(
                        "cannot close scheduler inbox with retained inventory"
                    )
                self._clear_wake_locked()
                self._closed = True
                read_fd = self._read_fd
                write_fd = self._write_fd
            os.close(read_fd)
            os.close(write_fd)

    def close_fail_closed(self) -> SchedulerReceiptInboxInventory:
        """Close wake descriptors while retaining process-fatal evidence.

        This operation is reserved for process teardown. It never retires a
        live binding or active receipt because ambiguous resources must remain
        quarantined in the final inventory.

        :returns: Complete retained inventory after descriptor closure.
        :raises SchedulerInboxError: If no process-fatal cause is present or a
            publication is still inside its linearization boundary.
        """

        with self._consumer_lock:
            with self._state_lock:
                if self._closed:
                    return self._inventory_locked()
                if self._fatal_cause is None:
                    raise SchedulerInboxError(
                        "fail-closed scheduler inbox teardown requires a fatal cause"
                    )
                with self._intent_condition:
                    if len(self._active_intents) > 0:
                        raise SchedulerInboxError(
                            "cannot close scheduler inbox with an active launch intent"
                        )
                self._clear_wake_locked()
                self._closed = True
                read_fd = self._read_fd
                write_fd = self._write_fd
                inventory = self._inventory_locked()
            os.close(read_fd)
            os.close(write_fd)
            return inventory

    def _begin_publication_intent(self) -> int:
        """Announce a publication before it can block on the launch gate.

        :returns: Monotonic process-local publication intent identity.
        """

        with self._intent_condition:
            intent = self._next_intent
            self._next_intent += 1
            self._active_intents.add(intent)
            return intent

    def _complete_publication_intent(self, intent: int) -> None:
        """Release one publication announcement and wake gate waiters.

        :param intent: Exact active publication intent identity.
        """

        with self._intent_condition:
            if intent in self._active_delivery_intents:
                raise SchedulerInboxError(
                    "publication completion referenced a delivery intent"
                )
            if intent not in self._active_intents:
                raise SchedulerInboxError("publication intent completed twice")
            self._active_intents.remove(intent)
            self._intent_condition.notify_all()

    def _publication_intent_snapshot(self) -> frozenset[int]:
        """Capture every producer which began before host submission returned.

        :returns: Immutable set of active publication intent identities.
        """

        with self._intent_condition:
            return frozenset(self._active_intents)

    def _wait_for_publication_intents(self, intents: frozenset[int]) -> None:
        """Wait actively for only the producer set racing this launch.

        :param intents: Finite publication set captured at a handoff boundary.
        """

        if type(intents) is not frozenset:
            raise TypeError("intents must be a frozenset")
        with self._intent_condition:
            self._intent_condition.wait_for(
                lambda: len(intents & self._active_intents) == 0
            )

    def _drain_ready(
        self,
        consume: Callable[[TerminalWireReceipt], None],
    ) -> tuple[TerminalWireReceipt, ...]:
        """Drain every publication visible through the authoritative queue.

        :param consume: Scheduler-affine receipt consumer.
        :returns: Exact FIFO receipts successfully consumed.
        """

        consumed: list[TerminalWireReceipt] = []
        while True:
            with self._state_lock:
                self._require_operational_locked()
                self._clear_wake_locked()
                if len(self._pending) == 0:
                    return tuple(consumed)
                publication = self._take_next_locked()
            self._consume_publication(publication, consume)
            consumed.append(publication.receipt)

    def _take_next_locked(self) -> _PendingSchedulerReceipt:
        """Move the oldest pending publication into scheduler consumption.

        :returns: Oldest pending publication under exclusive state ownership.
        """

        publication = self._pending.popleft()
        request_key = publication.request_key
        if request_key in self._consuming:
            self._enter_process_fatal_locked(
                SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT
            )
            raise self._fatal_error_locked()
        self._consuming[request_key] = publication
        self._validate_bounds_locked()
        return publication

    def _consume_publication(
        self,
        publication: _PendingSchedulerReceipt,
        consume: Callable[[TerminalWireReceipt], None],
    ) -> None:
        """Invoke the scheduler consumer and retire its exact live generation.

        :param publication: Exact publication owned by scheduler consumption.
        :param consume: Scheduler-affine receipt consumer.
        """

        succeeded = False
        try:
            consume(publication.receipt)
            succeeded = True
        finally:
            with self._state_lock:
                if succeeded:
                    request_key = publication.request_key
                    current = self._consuming.get(request_key)
                    if current != publication:
                        self._enter_process_fatal_locked(
                            SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT
                        )
                        raise self._fatal_error_locked()
                    del self._consuming[request_key]
                    del self._active_encoded[request_key]
                    del self._live[request_key]
                    self._validate_bounds_locked()
                else:
                    self._enter_process_fatal_locked(
                        SchedulerInboxFatalCause.OWNER_DEATH
                    )

    def _validate_bounds_locked(self) -> None:
        """Fail closed if mutable runtime storage violates conservation."""

        if len(self._pending) > len(self._live):
            self._enter_process_fatal_locked(
                SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT
            )
            raise self._fatal_error_locked()
        if len(self._live) > self._physical_capacity:
            self._enter_process_fatal_locked(SchedulerInboxFatalCause.PHYSICAL_CAPACITY)
            raise self._fatal_error_locked()
        active_keys = set(self._active_encoded)
        runtime_keys = {
            *(publication.request_key for publication in self._pending),
            *self._consuming,
        }
        if active_keys != runtime_keys or not active_keys.issubset(self._live):
            self._enter_process_fatal_locked(
                SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT
            )
            raise self._fatal_error_locked()

    def _enter_process_fatal_locked(
        self,
        cause: SchedulerInboxFatalCause,
    ) -> None:
        """Enter the one sticky fatal path and actively wake the scheduler.

        :param cause: Fatal cause retained as the process disposition.
        """

        if type(cause) is not SchedulerInboxFatalCause:
            raise TypeError("cause must be SchedulerInboxFatalCause")
        if self._fatal_cause is None:
            self._fatal_cause = cause
        self._signal_locked()

    def _require_open_locked(self) -> None:
        """Reject access after descriptor closure."""

        if self._closed:
            raise SchedulerInboxError("scheduler receipt inbox is closed")

    def _require_operational_locked(self) -> None:
        """Reject mutations and consumption after a fatal disposition."""

        self._require_open_locked()
        if self._fatal_cause is not None:
            raise self._fatal_error_locked()

    def _fatal_error_locked(self) -> SchedulerReceiptInboxFatalError:
        """Build one error carrying the first fatal cause and full inventory.

        :returns: Exact sticky process-fatal error.
        """

        cause = self._fatal_cause
        if cause is None:
            raise SchedulerInboxError("scheduler inbox has no fatal cause")
        return SchedulerReceiptInboxFatalError(cause, self._inventory_locked())

    def _inventory_locked(self) -> SchedulerReceiptInboxInventory:
        """Build one immutable inventory while the state lock is held.

        :returns: Complete immutable runtime inventory.
        """

        with self._intent_condition:
            active_delivery_intents = tuple(
                sorted(
                    self._active_delivery_intents.values(),
                    key=lambda intent: intent.identity,
                )
            )
            outstanding_publications = len(self._active_intents) - len(
                active_delivery_intents
            )
        return SchedulerReceiptInboxInventory(
            physical_capacity=self._physical_capacity,
            live_bindings=tuple(self._live.values()),
            pending_request_keys=tuple(
                publication.request_key for publication in self._pending
            ),
            consuming_request_keys=tuple(self._consuming),
            outstanding_publications=outstanding_publications,
            active_delivery_intents=active_delivery_intents,
            wake_armed=self._wake_armed,
            fatal_cause=self._fatal_cause,
            closed=self._closed,
        )

    def _signal_locked(self) -> None:
        """Publish one coalesced fd wake after authoritative state mutation."""

        if self._wake_armed or self._closed:
            return
        while True:
            try:
                os.write(self._write_fd, b"\x01")
                self._wake_armed = True
                return
            except BlockingIOError:
                if self._fatal_cause is None:
                    self._fatal_cause = SchedulerInboxFatalCause.PHYSICAL_CAPACITY
                return
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                if self._fatal_cause is None:
                    self._fatal_cause = SchedulerInboxFatalCause.OWNER_DEATH
                return

    def _clear_wake_locked(self) -> None:
        """Consume every coalesced wake byte without sleep polling."""

        if not self._wake_armed:
            return
        while True:
            try:
                data = os.read(self._read_fd, 4096)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                self._enter_process_fatal_locked(SchedulerInboxFatalCause.OWNER_DEATH)
                raise self._fatal_error_locked() from error
            if len(data) == 0 or len(data) < 4096:
                break
        self._wake_armed = False
