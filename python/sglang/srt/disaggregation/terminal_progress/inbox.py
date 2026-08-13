import dataclasses
import enum

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PackedRequestKey,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    terminal_receipt_token,
)


class SchedulerInboxError(RuntimeError):
    """Bounded scheduler inbox invariant violation."""


class SchedulerInboxDisposition(enum.StrEnum):
    """Process disposition owned by the scheduler inbox model."""

    HEALTHY = "healthy"
    PROCESS_FATAL = "process_fatal"


class SchedulerInboxFatalCause(enum.StrEnum):
    """First cause entering the shared process-fatal disposition."""

    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    PHYSICAL_CAPACITY = "physical_capacity"
    PENDING_EXCEEDS_INFLIGHT = "pending_exceeds_inflight"
    OWNER_DEATH = "owner_death"


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerInboxOverflow:
    """Observed queue counts proving one overflow class.

    :ivar pending_count: Entries that would be pending after the operation.
    :ivar live_inflight_count: Exact live request-generation count.
    :ivar physical_capacity: Configured physical entry capacity.
    """

    pending_count: int
    live_inflight_count: int
    physical_capacity: int

    def __post_init__(self) -> None:
        """Require evidence of a logical or physical overflow."""

        counts = (
            (self.pending_count, "pending_count"),
            (self.live_inflight_count, "live_inflight_count"),
            (self.physical_capacity, "physical_capacity"),
        )
        for count, label in counts:
            if type(count) is not int or count < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.physical_capacity <= 0:
            raise ValueError("physical_capacity must be positive")
        if (
            self.pending_count <= self.live_inflight_count
            and self.pending_count <= self.physical_capacity
        ):
            raise ValueError("counts do not describe an inbox overflow")

    @property
    def fatal_cause(self) -> SchedulerInboxFatalCause:
        """Return the shared fatal cause classification for this evidence.

        :returns: Physical overflow first, otherwise logical overflow.
        """

        if self.pending_count > self.physical_capacity:
            return SchedulerInboxFatalCause.PHYSICAL_CAPACITY
        return SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerInboxEntry:
    """One exact authority receipt pending scheduler consumption.

    :ivar receipt: Immutable one-shot receipt for one live request generation.
    """

    receipt: TerminalReceipt

    def __post_init__(self) -> None:
        """Validate one pending receipt entry."""

        if type(self.receipt) is not TerminalReceipt:
            raise TypeError("receipt must be TerminalReceipt")

    @property
    def request_key(self) -> PackedRequestKey:
        """Return the stable live request-generation key.

        :returns: Packed request key carried by the receipt.
        """

        return self.receipt.binding.request_key

    def is_exact_duplicate(self, receipt: TerminalReceipt) -> bool:
        """Return whether another value is the exact same issued receipt.

        :param receipt: Candidate duplicate receipt.
        :returns: Whether public fields and private one-shot token agree.
        """

        if type(receipt) is not TerminalReceipt:
            raise TypeError("receipt must be TerminalReceipt")
        return self.receipt == receipt and terminal_receipt_token(
            self.receipt
        ) == terminal_receipt_token(receipt)


@dataclasses.dataclass(frozen=True, slots=True)
class BoundedSchedulerInbox:
    """Immutable one-receipt-per-generation scheduler inbox.

    Overflow from either the logical generation bound or physical queue bound
    enters one process-fatal disposition. The first fatal cause is retained for
    evidence, while the pending queue remains unchanged for fail-closed drain.

    :ivar physical_capacity: Maximum physically representable pending entries.
    :ivar live_bindings: Exact request generations currently in flight.
    :ivar pending: FIFO receipts awaiting scheduler consumption.
    :ivar disposition: Healthy or process-fatal lifecycle disposition.
    :ivar fatal_cause: First cause of a process-fatal disposition.
    """

    physical_capacity: int
    live_bindings: tuple[TerminalRequestBinding, ...] = ()
    pending: tuple[SchedulerInboxEntry, ...] = ()
    disposition: SchedulerInboxDisposition = SchedulerInboxDisposition.HEALTHY
    fatal_cause: SchedulerInboxFatalCause | None = None

    def __post_init__(self) -> None:
        """Validate queue, identity, and disposition conservation."""

        if type(self.physical_capacity) is not int or self.physical_capacity <= 0:
            raise ValueError("physical_capacity must be a positive integer")
        if type(self.live_bindings) is not tuple:
            raise TypeError("live_bindings must be a tuple")
        if type(self.pending) is not tuple:
            raise TypeError("pending must be a tuple")
        if type(self.disposition) is not SchedulerInboxDisposition:
            raise TypeError("disposition must be SchedulerInboxDisposition")

        live_by_key: dict[PackedRequestKey, TerminalRequestBinding] = {}
        for binding in self.live_bindings:
            if type(binding) is not TerminalRequestBinding:
                raise TypeError("live_bindings entries must be TerminalRequestBinding")
            request_key = binding.request_key
            if request_key in live_by_key:
                raise ValueError("live request generations must be unique")
            live_by_key[request_key] = binding

        pending_keys: set[PackedRequestKey] = set()
        for entry in self.pending:
            if type(entry) is not SchedulerInboxEntry:
                raise TypeError("pending entries must be SchedulerInboxEntry")
            request_key = entry.request_key
            if request_key in pending_keys:
                raise ValueError(
                    "pending receipts must be unique by request generation"
                )
            binding = live_by_key.get(request_key)
            if binding is None or binding != entry.receipt.binding:
                raise ValueError("every pending receipt must bind to a live request")
            pending_keys.add(request_key)

        if len(self.pending) > len(self.live_bindings):
            raise ValueError("pending receipts cannot exceed live requests")
        if len(self.pending) > self.physical_capacity:
            raise ValueError("pending receipts cannot exceed physical capacity")

        if self.disposition is SchedulerInboxDisposition.HEALTHY:
            if self.fatal_cause is not None:
                raise ValueError("a healthy inbox cannot carry a fatal cause")
            return
        if type(self.fatal_cause) is not SchedulerInboxFatalCause:
            raise ValueError("a process-fatal inbox requires one fatal cause")

    @property
    def live_count(self) -> int:
        """Return the exact in-flight request count.

        :returns: Number of live request generations.
        """

        return len(self.live_bindings)

    @property
    def pending_count(self) -> int:
        """Return the exact pending receipt count.

        :returns: Number of receipts awaiting scheduler consumption.
        """

        return len(self.pending)

    def _require_healthy(self) -> None:
        """Reject mutations after the process entered fatal disposition.

        :raises SchedulerInboxError: If the process must terminate.
        """

        if self.disposition is SchedulerInboxDisposition.PROCESS_FATAL:
            raise SchedulerInboxError("scheduler inbox is process-fatal")

    def _fatal(
        self,
        cause: SchedulerInboxFatalCause,
    ) -> "BoundedSchedulerInbox":
        """Enter the shared process-fatal disposition without queue mutation.

        :param cause: First fatal invariant violation.
        :returns: New process-fatal inbox state.
        """

        if type(cause) is not SchedulerInboxFatalCause:
            raise TypeError("cause must be SchedulerInboxFatalCause")
        self._require_healthy()
        return dataclasses.replace(
            self,
            disposition=SchedulerInboxDisposition.PROCESS_FATAL,
            fatal_cause=cause,
        )

    def register_live(
        self,
        binding: TerminalRequestBinding,
    ) -> "BoundedSchedulerInbox":
        """Register one request generation before a receipt can arrive.

        :param binding: Exact live request-local binding.
        :returns: New inbox with the request registered.
        :raises SchedulerInboxError: If the key is already bound differently.
        """

        self._require_healthy()
        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        for live_binding in self.live_bindings:
            if live_binding.request_key != binding.request_key:
                continue
            if live_binding == binding:
                return self
            raise SchedulerInboxError(
                "request generation is already registered to another binding"
            )
        return dataclasses.replace(
            self,
            live_bindings=(*self.live_bindings, binding),
        )

    def unregister_live(
        self,
        binding: TerminalRequestBinding,
    ) -> "BoundedSchedulerInbox":
        """Remove one request only after its receipt was consumed.

        :param binding: Exact live request-local binding.
        :returns: New inbox without the request.
        :raises SchedulerInboxError: If the binding is absent or still pending.
        """

        self._require_healthy()
        if type(binding) is not TerminalRequestBinding:
            raise TypeError("binding must be TerminalRequestBinding")
        if binding not in self.live_bindings:
            raise SchedulerInboxError("request binding is not live")
        if any(entry.request_key == binding.request_key for entry in self.pending):
            raise SchedulerInboxError(
                "cannot unregister a request with a pending receipt"
            )
        return dataclasses.replace(
            self,
            live_bindings=tuple(
                live_binding
                for live_binding in self.live_bindings
                if live_binding != binding
            ),
        )

    def enqueue(self, receipt: TerminalReceipt) -> "BoundedSchedulerInbox":
        """Queue one live receipt or coalesce its exact retransmission.

        A conflicting second receipt for one live request is a logical overflow.
        A receipt beyond the configured storage is a physical overflow. Both
        enter the same process-fatal disposition before changing the queue.

        :param receipt: Exact issued scheduler authority receipt.
        :returns: New healthy or process-fatal inbox state.
        :raises SchedulerInboxError: If the receipt does not bind to a live
            request generation.
        """

        self._require_healthy()
        if type(receipt) is not TerminalReceipt:
            raise TypeError("receipt must be TerminalReceipt")

        matching_bindings = tuple(
            binding
            for binding in self.live_bindings
            if binding.request_key == receipt.binding.request_key
        )
        if len(matching_bindings) != 1 or matching_bindings[0] != receipt.binding:
            raise SchedulerInboxError(
                "receipt does not bind to an exact live request generation"
            )

        for entry in self.pending:
            if entry.request_key != receipt.binding.request_key:
                continue
            if entry.is_exact_duplicate(receipt):
                return self
            return self._fatal(SchedulerInboxFatalCause.CONFLICTING_DUPLICATE)

        if len(self.pending) >= self.physical_capacity:
            return self._fatal(SchedulerInboxFatalCause.PHYSICAL_CAPACITY)
        if len(self.pending) >= len(self.live_bindings):
            return self._fatal(SchedulerInboxFatalCause.PENDING_EXCEEDS_INFLIGHT)
        return dataclasses.replace(
            self,
            pending=(*self.pending, SchedulerInboxEntry(receipt=receipt)),
        )

    def consume_next(
        self,
    ) -> tuple["BoundedSchedulerInbox", TerminalReceipt]:
        """Consume the oldest pending receipt in deterministic FIFO order.

        :returns: New inbox and the exact consumed receipt.
        :raises SchedulerInboxError: If the inbox is empty or process-fatal.
        """

        self._require_healthy()
        if len(self.pending) == 0:
            raise SchedulerInboxError("scheduler inbox is empty")
        entry = self.pending[0]
        return dataclasses.replace(self, pending=self.pending[1:]), entry.receipt

    def mark_owner_dead(self) -> "BoundedSchedulerInbox":
        """Enter process-fatal disposition after owner-thread death.

        :returns: New process-fatal inbox preserving drain evidence.
        """

        return self._fatal(SchedulerInboxFatalCause.OWNER_DEATH)

    def observe_overflow(
        self,
        overflow: SchedulerInboxOverflow,
    ) -> "BoundedSchedulerInbox":
        """Map external logical or physical overflow to process-fatal state.

        Runtime storage may detect overflow before it can materialize an
        invalid immutable inbox. This method gives both bounds the same
        fail-closed lifecycle transition while retaining valid drain evidence.

        :param overflow: Counts observed at the failed enqueue boundary.
        :returns: New process-fatal inbox preserving pending entries.
        """

        if type(overflow) is not SchedulerInboxOverflow:
            raise TypeError("overflow must be SchedulerInboxOverflow")
        if overflow.live_inflight_count != len(self.live_bindings):
            raise SchedulerInboxError(
                "overflow live count does not match the exact inbox population"
            )
        if overflow.physical_capacity != self.physical_capacity:
            raise SchedulerInboxError(
                "overflow capacity does not match the configured inbox"
            )
        return self._fatal(overflow.fatal_cause)
