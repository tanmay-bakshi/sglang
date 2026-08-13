import collections
import selectors
import threading
import time
import traceback
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.clock import (
    SystemTerminalOwnerClock,
    TerminalOwnerClock,
)
from sglang.srt.disaggregation.terminal_progress.deadlines import (
    BoundTerminalDeadline,
    TerminalDeadlineKind,
    start_terminal_deadline,
)
from sglang.srt.disaggregation.terminal_progress.event_source import (
    TerminalOwnerPulseEventSource,
    TerminalOwnerQueueEventSource,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.lifecycle import (
    DecodeLifecycle,
    DecodeLifecycleEvent,
    DecodeLifecycleEventKind,
    DecodeLifecyclePhase,
    SourceLifecycle,
    SourceLifecycleEvent,
    SourceLifecycleEventKind,
    SourceLifecyclePhase,
    create_decode_lifecycle,
    create_source_lifecycle,
    reduce_decode_lifecycle,
    reduce_source_lifecycle,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    AcknowledgeTerminalReceipt,
    ApplyDecodeLifecycleEvent,
    ApplySourceLifecycleEvent,
    BeginTerminalOwnerShutdown,
    CancelTerminalDeadline,
    InjectTerminalOwnerFailure,
    RegisterDecodeLifecycle,
    RegisterSourceLifecycle,
    RetireTerminalOwnerShutdown,
    ScheduleTerminalDeadline,
    TerminalOwnerClosedError,
    TerminalOwnerCommandValue,
    TerminalOwnerDispatchObserver,
    TerminalOwnerDisposition,
    TerminalOwnerError,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerEventSourceRegistration,
    TerminalOwnerFatalCause,
    TerminalOwnerOutput,
    TerminalOwnerOverflowError,
    TerminalOwnerPulse,
    TerminalOwnerQuarantineEntry,
    TerminalOwnerReceiptEmission,
    TerminalOwnerSnapshot,
    TerminalOwnerTimingAnchor,
    TerminalOwnerTimingSample,
)
from sglang.srt.disaggregation.terminal_progress.receipts import (
    TerminalReceipt,
    TerminalReceiptAuthority,
    TerminalReceiptIssuer,
    TerminalReceiptKind,
    TerminalReceiptLedger,
    TerminalReceiptOutcome,
    TerminalReceiptToken,
    terminal_receipt_token,
)

_INTERNAL_SOURCE_NAME = "packed-terminal-owner-submissions"


class PackedTerminalProgressOwner:
    """Process-lifetime single owner for packed terminal progress.

    Native progress, CUDA callbacks, control receivers, schedulers, and the
    gateway publisher enter through immutable fd-backed event sources. Only the
    reactor thread advances request state, mints authority, or changes resource
    disposition. Consumers receive immutable outputs and explicitly
    acknowledge every one-shot authority receipt.
    """

    _submission_source: TerminalOwnerQueueEventSource
    _deadline_pulse_source: TerminalOwnerPulseEventSource
    _output_pulse_source: TerminalOwnerPulseEventSource
    _registrations: tuple[TerminalOwnerEventSourceRegistration, ...]
    _sources_by_name: dict[str, TerminalOwnerEventSource]
    _dispatch_observers_by_name: dict[str, TerminalOwnerDispatchObserver]
    _source_next_sequence: dict[str, int]
    _output_capacity: int
    _outputs: collections.deque[TerminalOwnerOutput]
    _source_lifecycles: dict[TerminalRequestBinding, SourceLifecycle]
    _decode_lifecycles: dict[TerminalRequestBinding, DecodeLifecycle]
    _known_bindings: set[TerminalRequestBinding]
    _retired_bindings: set[TerminalRequestBinding]
    _quarantine_reasons: dict[TerminalRequestBinding, str]
    _deadlines: dict[
        tuple[TerminalRequestBinding | None, TerminalDeadlineKind],
        BoundTerminalDeadline,
    ]
    _pending_receipts: dict[TerminalReceiptToken, TerminalReceipt]
    _receipt_issuer: TerminalReceiptIssuer
    _clock: TerminalOwnerClock
    _condition: threading.Condition
    _thread: threading.Thread
    _started: bool
    _reactor_alive: bool
    _admission_open: bool
    _disposition: TerminalOwnerDisposition
    _owner_transition_count: int
    _fatal_cause: TerminalOwnerFatalCause | None
    _fatal_reason: str | None
    _fatal_traceback: str | None
    _shutdown_producers_retired: bool

    def __init__(
        self,
        submission_capacity: int,
        output_capacity: int,
        event_sources: tuple[TerminalOwnerEventSourceRegistration, ...] = (),
        clock: TerminalOwnerClock | None = None,
        thread_name: str = "packed-terminal-progress-owner",
    ) -> None:
        """Construct one owner before any producer begins publication.

        :param submission_capacity: Maximum scheduler/control commands retained.
        :param output_capacity: Maximum undrained timing and receipt outputs.
        :param event_sources: Native-neutral fd sources registered before start.
        :param clock: Same-process monotonic clock used for deadlines and evidence.
        :param thread_name: Stable reader-facing reactor thread name.
        """

        if type(submission_capacity) is not int or submission_capacity <= 0:
            raise ValueError("submission_capacity must be a positive integer")
        if type(output_capacity) is not int or output_capacity <= 0:
            raise ValueError("output_capacity must be a positive integer")
        if type(event_sources) is not tuple:
            raise TypeError("event_sources must be a tuple")
        if any(
            type(registration) is not TerminalOwnerEventSourceRegistration
            for registration in event_sources
        ):
            raise TypeError(
                "event_sources must contain TerminalOwnerEventSourceRegistration"
            )
        if clock is not None and not isinstance(clock, TerminalOwnerClock):
            raise TypeError("clock must inherit TerminalOwnerClock")
        if type(thread_name) is not str or len(thread_name) == 0:
            raise ValueError("thread_name must be a non-empty string")

        submission_source = TerminalOwnerQueueEventSource(
            name=_INTERNAL_SOURCE_NAME,
            capacity=submission_capacity,
        )
        deadline_pulse_source = TerminalOwnerPulseEventSource(
            name="packed-terminal-owner-deadline-pulse"
        )
        output_pulse_source = TerminalOwnerPulseEventSource(
            name="packed-terminal-owner-output-pulse"
        )
        registrations = (
            TerminalOwnerEventSourceRegistration(
                source=submission_source,
                close_on_shutdown=True,
            ),
            TerminalOwnerEventSourceRegistration(
                source=deadline_pulse_source,
                close_on_shutdown=True,
            ),
            *event_sources,
        )
        names = tuple(registration.source.name for registration in registrations)
        fds = tuple(registration.source.fileno() for registration in registrations)
        if len(set(names)) != len(names):
            submission_source.close()
            deadline_pulse_source.close()
            output_pulse_source.close()
            raise ValueError("event source names must be unique")
        if len(set(fds)) != len(fds):
            submission_source.close()
            deadline_pulse_source.close()
            output_pulse_source.close()
            raise ValueError("event source file descriptors must be unique")

        self._submission_source = submission_source
        self._deadline_pulse_source = deadline_pulse_source
        self._output_pulse_source = output_pulse_source
        self._registrations = registrations
        self._sources_by_name = {
            registration.source.name: registration.source
            for registration in registrations
        }
        self._dispatch_observers_by_name = {
            registration.source.name: registration.dispatch_observer
            for registration in registrations
            if registration.dispatch_observer is not None
        }
        self._source_next_sequence = {name: 0 for name in names}
        self._output_capacity = output_capacity
        self._outputs = collections.deque()
        self._source_lifecycles = {}
        self._decode_lifecycles = {}
        self._known_bindings = set()
        self._retired_bindings = set()
        self._quarantine_reasons = {}
        self._deadlines = {}
        self._pending_receipts = {}
        self._receipt_issuer = TerminalReceiptIssuer()
        self._clock = SystemTerminalOwnerClock() if clock is None else clock
        self._condition = threading.Condition()
        self._started = False
        self._reactor_alive = False
        self._admission_open = False
        self._disposition = TerminalOwnerDisposition.CREATED
        self._owner_transition_count = 0
        self._fatal_cause = None
        self._fatal_reason = None
        self._fatal_traceback = None
        self._shutdown_producers_retired = False
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    @property
    def receipt_authority(self) -> TerminalReceiptAuthority:
        """Return the opaque authority used to validate owner-minted receipts.

        :returns: Process-local terminal receipt authority.
        """

        return self._receipt_issuer.authority

    def start(self) -> None:
        """Start the process-lifetime reactor exactly once."""

        with self._condition:
            if self._started:
                raise TerminalOwnerError("terminal progress owner cannot restart")
            self._started = True
        self._thread.start()

    def submit(
        self,
        command: TerminalOwnerCommandValue,
        enqueued_ns: int | None = None,
    ) -> TerminalOwnerEventEnvelope:
        """Publish one immutable scheduler or control command.

        :param command: Exact owner command.
        :param enqueued_ns: Optional same-process timing anchor.
        :returns: Producer-sequenced envelope placed in the bounded source.
        """

        with self._condition:
            if not self._started:
                raise TerminalOwnerClosedError("terminal progress owner is not started")
            if self._disposition in (
                TerminalOwnerDisposition.STOPPED,
                TerminalOwnerDisposition.PROCESS_FATAL,
            ):
                raise TerminalOwnerClosedError(
                    f"terminal progress owner is {self._disposition.value}"
                )
            if self._disposition is TerminalOwnerDisposition.DRAINING and isinstance(
                command,
                (RegisterSourceLifecycle, RegisterDecodeLifecycle),
            ):
                raise TerminalOwnerClosedError("request admission is closed")
        return self._submission_source.publish(command, enqueued_ns=enqueued_ns)

    def begin_shutdown(self, started_ns: int | None = None) -> None:
        """Close admission and begin explicit fail-closed drain.

        :param started_ns: Optional exact monotonic shutdown anchor.
        """

        timestamp_ns = self._clock.now_ns() if started_ns is None else started_ns
        self.submit(BeginTerminalOwnerShutdown(started_ns=timestamp_ns))

    def retire_shutdown_producers(self) -> None:
        """Declare every external event producer joined after drain begins."""

        self.submit(RetireTerminalOwnerShutdown())

    def notify_clock_advanced(self) -> None:
        """Wake deadline evaluation after an externally controlled clock step."""

        self._deadline_pulse_source.signal()

    def drain_outputs(
        self, maximum_items: int | None = None
    ) -> tuple[TerminalOwnerOutput, ...]:
        """Drain immutable owner outputs without consuming receipt authority.

        :param maximum_items: Optional positive output count bound.
        :returns: FIFO timing samples and receipt emissions.
        """

        if maximum_items is not None and (
            type(maximum_items) is not int or maximum_items <= 0
        ):
            raise ValueError("maximum_items must be a positive integer")
        try:
            self._output_pulse_source.drain()
        except TerminalOwnerClosedError:
            pass
        should_wake = False
        with self._condition:
            count = len(self._outputs)
            if maximum_items is not None:
                count = min(count, maximum_items)
            outputs = tuple(self._outputs.popleft() for _ in range(count))
            should_wake = self._disposition is TerminalOwnerDisposition.DRAINING
            self._condition.notify_all()
        if should_wake:
            try:
                self._submission_source.publish(TerminalOwnerPulse())
            except TerminalOwnerClosedError:
                pass
        return outputs

    def output_fileno(self) -> int:
        """Return the fd signalling newly queued immutable outputs.

        Queue insertion is authoritative and the fd is only a coalesced wake
        hint. Consumers drain all available outputs whenever it becomes
        readable, then dispatch each receipt to its exact scheduler, publisher,
        coordinator, or metrics inbox.

        :returns: Open readable output-notification descriptor.
        """

        return self._output_pulse_source.fileno()

    def snapshot(self) -> TerminalOwnerSnapshot:
        """Return an immutable process-lifetime inventory snapshot.

        :returns: Current active, retired, quarantined, and receipt evidence.
        """

        with self._condition:
            return self._snapshot_locked()

    def wait_for_snapshot(
        self,
        predicate: Callable[[TerminalOwnerSnapshot], bool],
        timeout_seconds: float,
    ) -> TerminalOwnerSnapshot:
        """Wait on owner state changes until one explicit predicate passes.

        :param predicate: Condition evaluated against immutable snapshots.
        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: First snapshot satisfying the predicate.
        :raises TimeoutError: If the condition does not pass within the bound.
        """

        if not callable(predicate):
            raise TypeError("predicate must be callable")
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        expires_at = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                snapshot = self._snapshot_locked()
                if predicate(snapshot):
                    return snapshot
                remaining = expires_at - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("terminal owner snapshot wait expired")
                self._condition.wait(timeout=remaining)

    def wait_for_output_count(
        self,
        minimum_count: int,
        timeout_seconds: float,
    ) -> int:
        """Wait until the undrained immutable output queue reaches a count.

        :param minimum_count: Positive number of outputs required.
        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: Exact output count at the satisfying state.
        :raises TimeoutError: If the count does not arrive within the bound.
        """

        if type(minimum_count) is not int or minimum_count <= 0:
            raise ValueError("minimum_count must be a positive integer")
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        expires_at = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                if len(self._outputs) >= minimum_count:
                    return len(self._outputs)
                if self._disposition is TerminalOwnerDisposition.PROCESS_FATAL:
                    raise TerminalOwnerError("terminal owner became process-fatal")
                remaining = expires_at - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("terminal owner output wait expired")
                self._condition.wait(timeout=remaining)

    def join(self, timeout_seconds: float | None = None) -> bool:
        """Join the process-lifetime reactor thread.

        :param timeout_seconds: Optional non-negative join bound.
        :returns: Whether the reactor has stopped.
        """

        if timeout_seconds is not None and (
            type(timeout_seconds) is not float or timeout_seconds < 0.0
        ):
            raise ValueError("timeout_seconds must be a non-negative float")
        self._thread.join(timeout=timeout_seconds)
        return not self._thread.is_alive()

    def _run(self) -> None:
        """Own every lifecycle transition until clean drain or fatal exit."""

        selector = selectors.DefaultSelector()
        try:
            for registration in self._registrations:
                selector.register(
                    registration.source.fileno(),
                    selectors.EVENT_READ,
                    registration.source,
                )
            with self._condition:
                self._reactor_alive = True
                self._admission_open = True
                self._disposition = TerminalOwnerDisposition.RUNNING
                self._condition.notify_all()

            while True:
                with self._condition:
                    self._expire_deadlines_locked(self._clock.now_ns())
                    self._finish_drain_if_complete_locked()
                    if self._disposition in (
                        TerminalOwnerDisposition.STOPPED,
                        TerminalOwnerDisposition.PROCESS_FATAL,
                    ):
                        break
                    timeout_seconds = self._next_selector_timeout_locked()
                ready = selector.select(timeout=timeout_seconds)
                for key, _ in ready:
                    source = key.data
                    if not isinstance(source, TerminalOwnerEventSource):
                        raise TerminalOwnerError(
                            "selector returned an unknown event-source adapter"
                        )
                    self._drain_source(source)
                    with self._condition:
                        if self._disposition is TerminalOwnerDisposition.PROCESS_FATAL:
                            break
        except Exception:
            formatted_traceback = traceback.format_exc()
            with self._condition:
                self._enter_fatal_locked(
                    cause=TerminalOwnerFatalCause.OWNER_EXCEPTION,
                    reason="terminal progress owner reactor raised unexpectedly",
                    formatted_traceback=formatted_traceback,
                )
        finally:
            selector.close()
            for registration in self._registrations:
                if registration.close_on_shutdown:
                    registration.source.close()
            self._output_pulse_source.close()
            with self._condition:
                self._reactor_alive = False
                self._admission_open = False
                if self._disposition not in (
                    TerminalOwnerDisposition.STOPPED,
                    TerminalOwnerDisposition.PROCESS_FATAL,
                ):
                    self._enter_fatal_locked(
                        cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
                        reason="terminal progress owner exited outside shutdown",
                    )
                self._condition.notify_all()

    def _drain_source(self, source: TerminalOwnerEventSource) -> None:
        """Validate and dispatch one complete source wake.

        :param source: Exact registered event-source adapter.
        """

        try:
            envelopes = source.drain()
        except TerminalOwnerOverflowError as error:
            formatted_traceback = traceback.format_exc()
            with self._condition:
                self._inventory_overflow_registrations_locked(error)
                self._enter_fatal_locked(
                    cause=TerminalOwnerFatalCause.SUBMISSION_QUEUE_OVERFLOW,
                    reason=f"event source {source.name} overflowed",
                    formatted_traceback=formatted_traceback,
                )
            return
        except Exception:
            formatted_traceback = traceback.format_exc()
            with self._condition:
                self._enter_fatal_locked(
                    cause=TerminalOwnerFatalCause.EVENT_SOURCE_FAILURE,
                    reason=f"event source {source.name} failed while draining",
                    formatted_traceback=formatted_traceback,
                )
            return

        for envelope in envelopes:
            expected = self._source_next_sequence[source.name]
            if envelope.producer_sequence != expected:
                with self._condition:
                    self._enter_fatal_locked(
                        cause=TerminalOwnerFatalCause.EVENT_SOURCE_ORDER,
                        reason=(
                            f"event source {source.name} delivered sequence "
                            f"{envelope.producer_sequence}, expected {expected}"
                        ),
                    )
                return
            self._source_next_sequence[source.name] = expected + 1
            try:
                self._dispatch(envelope)
            except Exception:
                formatted_traceback = traceback.format_exc()
                with self._condition:
                    self._enter_fatal_locked(
                        cause=TerminalOwnerFatalCause.OWNER_EXCEPTION,
                        reason=(
                            f"terminal owner rejected {source.name} sequence "
                            f"{envelope.producer_sequence}"
                        ),
                        formatted_traceback=formatted_traceback,
                    )
                return
            observer = self._dispatch_observers_by_name.get(source.name)
            if observer is None:
                continue
            try:
                observer.acknowledge_dispatch(envelope)
            except TerminalOwnerOverflowError as error:
                formatted_traceback = traceback.format_exc()
                with self._condition:
                    self._inventory_overflow_registrations_locked(error)
                    self._enter_fatal_locked(
                        cause=TerminalOwnerFatalCause.SUBMISSION_QUEUE_OVERFLOW,
                        reason=(
                            f"event source {source.name} overflowed while "
                            "acknowledging owner dispatch"
                        ),
                        formatted_traceback=formatted_traceback,
                    )
                return
            except Exception:
                formatted_traceback = traceback.format_exc()
                with self._condition:
                    self._enter_fatal_locked(
                        cause=TerminalOwnerFatalCause.EVENT_SOURCE_FAILURE,
                        reason=(
                            f"event source {source.name} failed while "
                            "acknowledging owner dispatch"
                        ),
                        formatted_traceback=formatted_traceback,
                    )
                return

    def _dispatch(self, envelope: TerminalOwnerEventEnvelope) -> None:
        """Apply one validated event-source envelope atomically.

        :param envelope: Exact source-ordered immutable command.
        """

        owner_started_ns = self._clock.now_ns()
        with self._condition:
            command = envelope.command
            owner_sequence = self._owner_transition_count
            if type(command) is RegisterSourceLifecycle:
                self._register_source_locked(command)
            elif type(command) is RegisterDecodeLifecycle:
                self._register_decode_locked(command)
            elif type(command) is ApplySourceLifecycleEvent:
                self._apply_source_locked(command, owner_started_ns, owner_sequence)
            elif type(command) is ApplyDecodeLifecycleEvent:
                self._apply_decode_locked(command, owner_started_ns, owner_sequence)
            elif type(command) is ScheduleTerminalDeadline:
                self._schedule_deadline_locked(command)
            elif type(command) is CancelTerminalDeadline:
                self._cancel_deadline_locked(command)
            elif type(command) is AcknowledgeTerminalReceipt:
                self._acknowledge_receipt_locked(command)
            elif type(command) is BeginTerminalOwnerShutdown:
                self._begin_shutdown_locked(command)
            elif type(command) is RetireTerminalOwnerShutdown:
                self._retire_shutdown_producers_locked()
            elif type(command) is InjectTerminalOwnerFailure:
                self._enter_fatal_locked(command.cause, command.reason)
            elif type(command) is TerminalOwnerPulse:
                pass
            else:
                raise TypeError("unknown terminal owner command")
            self._owner_transition_count += 1
            self._expire_deadlines_locked(self._clock.now_ns())
            self._finish_drain_if_complete_locked()
            self._condition.notify_all()

    def _register_source_locked(self, command: RegisterSourceLifecycle) -> None:
        """Create one owner-issued source lifecycle.

        :param command: Exact source registration.
        """

        self._require_admission_locked(command.binding)
        authorities = command.trusted_authorities | frozenset(
            (self._receipt_issuer.authority,)
        )
        self._source_lifecycles[command.binding] = create_source_lifecycle(
            binding=command.binding,
            publication_identity=command.publication_identity,
            receipt_ledger=TerminalReceiptLedger(authorities=authorities),
        )
        self._known_bindings.add(command.binding)

    def _register_decode_locked(self, command: RegisterDecodeLifecycle) -> None:
        """Create one owner-issued decode lifecycle.

        :param command: Exact decode registration.
        """

        self._require_admission_locked(command.binding)
        authorities = command.trusted_authorities | frozenset(
            (self._receipt_issuer.authority,)
        )
        self._decode_lifecycles[command.binding] = create_decode_lifecycle(
            binding=command.binding,
            receipt_ledger=TerminalReceiptLedger(authorities=authorities),
        )
        self._known_bindings.add(command.binding)

    def _inventory_overflow_registrations_locked(
        self, error: TerminalOwnerOverflowError
    ) -> None:
        """Retain resources whose registration was accepted or rejected at overflow.

        The event-source emergency slot is not work capacity. It exists solely
        so fail-closed ownership can account for the command which crossed the
        bound instead of losing its generation and leased-resource identity.

        :param error: Sticky overflow carrying accepted and rejected envelopes.
        """

        envelopes = error.pending_envelopes
        if error.rejected_envelope is not None:
            envelopes = (*envelopes, error.rejected_envelope)
        for envelope in envelopes:
            command = envelope.command
            if type(command) is RegisterSourceLifecycle:
                if command.binding in self._known_bindings:
                    continue
                authorities = command.trusted_authorities | frozenset(
                    (self._receipt_issuer.authority,)
                )
                self._source_lifecycles[command.binding] = create_source_lifecycle(
                    binding=command.binding,
                    publication_identity=command.publication_identity,
                    receipt_ledger=TerminalReceiptLedger(authorities=authorities),
                )
                self._known_bindings.add(command.binding)
                continue
            if type(command) is RegisterDecodeLifecycle:
                if command.binding in self._known_bindings:
                    continue
                authorities = command.trusted_authorities | frozenset(
                    (self._receipt_issuer.authority,)
                )
                self._decode_lifecycles[command.binding] = create_decode_lifecycle(
                    binding=command.binding,
                    receipt_ledger=TerminalReceiptLedger(authorities=authorities),
                )
                self._known_bindings.add(command.binding)

    def _require_admission_locked(self, binding: TerminalRequestBinding) -> None:
        """Require open admission and a never-before-seen generation.

        :param binding: Candidate lifecycle binding.
        """

        if not self._admission_open:
            raise TerminalOwnerClosedError("request admission is closed")
        if binding in self._known_bindings:
            raise TerminalOwnerError("request binding was already registered")

    def _apply_source_locked(
        self,
        command: ApplySourceLifecycleEvent,
        owner_started_ns: int,
        owner_sequence: int,
    ) -> None:
        """Advance one source lifecycle and emit newly earned authority.

        :param command: Exact source event command.
        :param owner_started_ns: Owner dispatch timestamp.
        :param owner_sequence: Current owner transition sequence.
        """

        lifecycle = self._source_lifecycles.get(command.binding)
        if lifecycle is None:
            raise TerminalOwnerError("source lifecycle is not registered")
        transition = reduce_source_lifecycle(lifecycle, command.event)
        self._source_lifecycles[command.binding] = transition.current
        completed_ns = self._clock.now_ns()
        if (
            transition.previous.phase is not SourceLifecyclePhase.REQUEST_READY_RECEIVED
            and transition.current.phase is SourceLifecyclePhase.REQUEST_READY_RECEIVED
        ):
            self._emit_receipt_locked(
                binding=command.binding,
                kind=TerminalReceiptKind.RECLAIM_AUTHORIZED,
                timestamp_ns=completed_ns,
                owner_sequence=owner_sequence,
            )
        self._emit_timing_locked(
            binding=command.binding,
            timing_anchor=command.timing_anchor,
            owner_started_ns=owner_started_ns,
            completed_ns=completed_ns,
            owner_sequence=owner_sequence,
        )
        self._reconcile_source_locked(command.binding, transition.current)

    def _apply_decode_locked(
        self,
        command: ApplyDecodeLifecycleEvent,
        owner_started_ns: int,
        owner_sequence: int,
    ) -> None:
        """Advance one decode lifecycle and emit newly earned authority.

        :param command: Exact decode event command.
        :param owner_started_ns: Owner dispatch timestamp.
        :param owner_sequence: Current owner transition sequence.
        """

        lifecycle = self._decode_lifecycles.get(command.binding)
        if lifecycle is None:
            raise TerminalOwnerError("decode lifecycle is not registered")
        transition = reduce_decode_lifecycle(lifecycle, command.event)
        self._decode_lifecycles[command.binding] = transition.current
        completed_ns = self._clock.now_ns()
        if transition.current.phase is DecodeLifecyclePhase.ADOPTION_READY:
            self._emit_receipt_locked(
                binding=command.binding,
                kind=TerminalReceiptKind.ADOPTION_READY,
                timestamp_ns=completed_ns,
                owner_sequence=owner_sequence,
            )
        if transition.current.phase is DecodeLifecyclePhase.LOCAL_DECODE_READY:
            self._emit_receipt_locked(
                binding=command.binding,
                kind=TerminalReceiptKind.LOCAL_DECODE_READY,
                timestamp_ns=completed_ns,
                owner_sequence=owner_sequence,
            )
        self._emit_timing_locked(
            binding=command.binding,
            timing_anchor=command.timing_anchor,
            owner_started_ns=owner_started_ns,
            completed_ns=completed_ns,
            owner_sequence=owner_sequence,
        )
        self._reconcile_decode_locked(command.binding, transition.current)

    def _emit_receipt_locked(
        self,
        binding: TerminalRequestBinding,
        kind: TerminalReceiptKind,
        timestamp_ns: int,
        owner_sequence: int,
    ) -> None:
        """Mint and queue one immutable downstream authority.

        :param binding: Exact request lifecycle earning authority.
        :param kind: Newly proven authority kind.
        :param timestamp_ns: Owner-local proof timestamp.
        :param owner_sequence: Current owner transition sequence.
        """

        self._require_output_capacity_locked()
        receipt = self._receipt_issuer.issue(
            binding=binding,
            kind=kind,
            outcome=TerminalReceiptOutcome.SUCCESS,
            terminal_timestamp_ns=timestamp_ns,
        )
        token = terminal_receipt_token(receipt)
        self._pending_receipts[token] = receipt
        self._outputs.append(
            TerminalOwnerReceiptEmission(
                receipt=receipt,
                emitted_ns=timestamp_ns,
                owner_sequence=owner_sequence,
            )
        )
        self._output_pulse_source.signal()

    def _emit_timing_locked(
        self,
        binding: TerminalRequestBinding,
        timing_anchor: TerminalOwnerTimingAnchor | None,
        owner_started_ns: int,
        completed_ns: int,
        owner_sequence: int,
    ) -> None:
        """Queue one optional same-process timing decomposition sample.

        :param binding: Exact request lifecycle represented by the sample.
        :param timing_anchor: External phase start, when attribution is required.
        :param owner_started_ns: Owner dispatch timestamp.
        :param completed_ns: Owner transition completion timestamp.
        :param owner_sequence: Current owner transition sequence.
        """

        if timing_anchor is None:
            return
        self._require_output_capacity_locked()
        self._outputs.append(
            TerminalOwnerTimingSample(
                binding=binding,
                field=timing_anchor.field,
                sample_key=timing_anchor.sample_key,
                started_ns=timing_anchor.started_ns,
                owner_started_ns=owner_started_ns,
                completed_ns=completed_ns,
                owner_sequence=owner_sequence,
            )
        )
        self._output_pulse_source.signal()

    def _require_output_capacity_locked(self) -> None:
        """Fail before silently dropping one owner output."""

        if len(self._outputs) >= self._output_capacity:
            self._enter_fatal_locked(
                cause=TerminalOwnerFatalCause.OUTPUT_QUEUE_OVERFLOW,
                reason=(
                    f"terminal owner output queue exceeded capacity "
                    f"{self._output_capacity}"
                ),
            )
            raise TerminalOwnerOverflowError("terminal owner output queue overflowed")

    def _acknowledge_receipt_locked(self, command: AcknowledgeTerminalReceipt) -> None:
        """Remove one exact emitted receipt after downstream consumption.

        :param command: Exact consumer acknowledgement.
        """

        token = terminal_receipt_token(command.receipt)
        pending = self._pending_receipts.get(token)
        if pending is None or pending != command.receipt:
            raise TerminalOwnerError("receipt acknowledgement is absent or conflicting")
        del self._pending_receipts[token]

    def _schedule_deadline_locked(self, command: ScheduleTerminalDeadline) -> None:
        """Start one deadline once at its frozen anchor.

        :param command: Exact binding, phase, and start timestamp.
        """

        if command.binding not in self._known_bindings:
            raise TerminalOwnerError("deadline belongs to an unknown request")
        key = (command.binding, command.kind)
        if key in self._deadlines:
            raise TerminalOwnerError("deadline phase was already started")
        self._deadlines[key] = start_terminal_deadline(
            kind=command.kind,
            started_ns=command.started_ns,
        )

    def _cancel_deadline_locked(self, command: CancelTerminalDeadline) -> None:
        """Cancel one exact active phase deadline.

        :param command: Exact binding and phase.
        """

        key = (command.binding, command.kind)
        if self._deadlines.pop(key, None) is None:
            raise TerminalOwnerError("deadline phase is not active")

    def _begin_shutdown_locked(self, command: BeginTerminalOwnerShutdown) -> None:
        """Close admission and start explicit bounded drain.

        :param command: Exact shutdown anchor.
        """

        if self._disposition is not TerminalOwnerDisposition.RUNNING:
            raise TerminalOwnerError("terminal owner shutdown already began")
        self._admission_open = False
        self._disposition = TerminalOwnerDisposition.DRAINING
        shutdown_key = (None, TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN)
        self._deadlines[shutdown_key] = start_terminal_deadline(
            kind=TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN,
            started_ns=command.started_ns,
        )

    def _retire_shutdown_producers_locked(self) -> None:
        """Record explicit join of every external event producer."""

        if self._disposition is not TerminalOwnerDisposition.DRAINING:
            raise TerminalOwnerError("producer retirement requires active drain")
        self._shutdown_producers_retired = True

    def _expire_deadlines_locked(self, now_ns: int) -> None:
        """Apply every deadline which expired at the current clock value.

        :param now_ns: Current same-process monotonic timestamp.
        """

        expired = tuple(
            sorted(
                (
                    (key, deadline)
                    for key, deadline in self._deadlines.items()
                    if deadline.expired(now_ns)
                ),
                key=lambda item: (item[1].expires_ns, item[0][1].value),
            )
        )
        for key, deadline in expired:
            if key not in self._deadlines:
                continue
            del self._deadlines[key]
            binding, kind = key
            if kind is TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN:
                self._enter_fatal_locked(
                    cause=TerminalOwnerFatalCause.SHUTDOWN_DEADLINE,
                    reason=deadline.spec.timeout_outcome,
                )
                return
            if binding is None:
                raise TerminalOwnerError("request deadline lost its binding")
            if kind in (
                TerminalDeadlineKind.OWNER_SCHEDULER_RECEIPT_CONSUMPTION,
                TerminalDeadlineKind.OWNER_GATEWAY_PUBLICATION,
            ):
                self._enter_fatal_locked(
                    cause=TerminalOwnerFatalCause.OWNER_DEPENDENCY_DEATH,
                    reason=deadline.spec.timeout_outcome,
                )
                return
            self._fail_request_deadline_locked(
                binding=binding,
                reason=deadline.spec.timeout_outcome,
                timestamp_ns=now_ns,
            )

    def _fail_request_deadline_locked(
        self,
        binding: TerminalRequestBinding,
        reason: str,
        timestamp_ns: int,
    ) -> None:
        """Quarantine one request after a non-process-fatal phase timeout.

        :param binding: Exact timed-out lifecycle binding.
        :param reason: Frozen timeout outcome.
        :param timestamp_ns: Owner-local failure timestamp.
        """

        failure_receipt = self._receipt_issuer.issue(
            binding=binding,
            kind=TerminalReceiptKind.FAILURE,
            outcome=TerminalReceiptOutcome.FAILURE,
            terminal_timestamp_ns=timestamp_ns,
        )
        source = self._source_lifecycles.get(binding)
        if source is not None and len(source.inventory.live) > 0:
            event = SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.REQUEST_FAILED,
                receipt=failure_receipt,
                reason=reason,
            )
            current = reduce_source_lifecycle(source, event).current
            self._source_lifecycles[binding] = current
            self._quarantine_reasons[binding] = reason
            self._reconcile_source_locked(binding, current)
            return
        decode = self._decode_lifecycles.get(binding)
        if decode is not None and len(decode.inventory.live) > 0:
            event = DecodeLifecycleEvent(
                kind=DecodeLifecycleEventKind.REQUEST_FAILED,
                receipt=failure_receipt,
                reason=reason,
            )
            current = reduce_decode_lifecycle(decode, event).current
            self._decode_lifecycles[binding] = current
            self._quarantine_reasons[binding] = reason
            self._reconcile_decode_locked(binding, current)

    def _next_selector_timeout_locked(self) -> float | None:
        """Return a kernel-wait timeout for the nearest active deadline.

        :returns: Seconds until the nearest deadline, or no timeout.
        """

        if len(self._deadlines) == 0:
            return None
        now_ns = self._clock.now_ns()
        nearest_ns = min(deadline.expires_ns for deadline in self._deadlines.values())
        return max(0.0, (nearest_ns - now_ns) / 1_000_000_000)

    def _finish_drain_if_complete_locked(self) -> None:
        """Stop only after every lifecycle and authority is accounted for."""

        if self._disposition is not TerminalOwnerDisposition.DRAINING:
            return
        if not self._shutdown_producers_retired:
            return
        active = any(
            len(lifecycle.inventory.live) > 0
            for lifecycle in (
                *self._source_lifecycles.values(),
                *self._decode_lifecycles.values(),
            )
        )
        queued_submissions = sum(
            source.pending_count for source in self._sources_by_name.values()
        )
        if (
            active
            or len(self._pending_receipts) > 0
            or len(self._outputs) > 0
            or queued_submissions > 0
        ):
            return
        self._deadlines.clear()
        self._disposition = TerminalOwnerDisposition.STOPPED

    def _enter_fatal_locked(
        self,
        cause: TerminalOwnerFatalCause,
        reason: str,
        formatted_traceback: str | None = None,
    ) -> None:
        """Close admission and quarantine every still-live owner resource.

        :param cause: First process-fatal cause.
        :param reason: Stable reader-facing failure reason.
        :param formatted_traceback: Complete unexpected-failure traceback.
        """

        if self._disposition is TerminalOwnerDisposition.PROCESS_FATAL:
            return
        self._admission_open = False
        self._disposition = TerminalOwnerDisposition.PROCESS_FATAL
        self._fatal_cause = cause
        self._fatal_reason = reason
        self._fatal_traceback = formatted_traceback
        for binding, lifecycle in tuple(self._source_lifecycles.items()):
            if len(lifecycle.inventory.live) == 0:
                continue
            event = SourceLifecycleEvent(
                kind=SourceLifecycleEventKind.OWNER_DIED,
                reason=reason,
            )
            current = reduce_source_lifecycle(lifecycle, event).current
            self._source_lifecycles[binding] = current
            self._quarantine_reasons[binding] = reason
        for binding, lifecycle in tuple(self._decode_lifecycles.items()):
            if len(lifecycle.inventory.live) == 0:
                continue
            event = DecodeLifecycleEvent(
                kind=DecodeLifecycleEventKind.OWNER_DIED,
                reason=reason,
            )
            current = reduce_decode_lifecycle(lifecycle, event).current
            self._decode_lifecycles[binding] = current
            self._quarantine_reasons[binding] = reason
        self._deadlines.clear()
        self._condition.notify_all()

    def _reconcile_source_locked(
        self, binding: TerminalRequestBinding, lifecycle: SourceLifecycle
    ) -> None:
        """Update terminal sets after one source transition.

        :param binding: Exact source lifecycle binding.
        :param lifecycle: Complete source state after the transition.
        """

        if lifecycle.phase is SourceLifecyclePhase.RETIRED:
            self._retired_bindings.add(binding)
            self._cancel_binding_deadlines_locked(binding)
        if len(lifecycle.inventory.quarantined) > 0:
            self._quarantine_reasons.setdefault(binding, lifecycle.phase.value)
        if lifecycle.phase is SourceLifecyclePhase.QUARANTINED:
            self._cancel_binding_deadlines_locked(binding)

    def _reconcile_decode_locked(
        self, binding: TerminalRequestBinding, lifecycle: DecodeLifecycle
    ) -> None:
        """Update terminal sets after one decode transition.

        :param binding: Exact decode lifecycle binding.
        :param lifecycle: Complete decode state after the transition.
        """

        if lifecycle.phase is DecodeLifecyclePhase.RETIRED:
            self._retired_bindings.add(binding)
            self._cancel_binding_deadlines_locked(binding)
        if len(lifecycle.inventory.quarantined) > 0:
            self._quarantine_reasons.setdefault(binding, lifecycle.phase.value)
        if lifecycle.phase is DecodeLifecyclePhase.QUARANTINED:
            self._cancel_binding_deadlines_locked(binding)

    def _cancel_binding_deadlines_locked(self, binding: TerminalRequestBinding) -> None:
        """Remove every now-irrelevant request-local deadline.

        :param binding: Lifecycle which reached a terminal resource disposition.
        """

        for key in tuple(self._deadlines):
            if (
                key[0] == binding
                and key[1] is not TerminalDeadlineKind.OWNER_SHUTDOWN_DRAIN
            ):
                del self._deadlines[key]

    def _snapshot_locked(self) -> TerminalOwnerSnapshot:
        """Build one immutable snapshot while holding the owner condition.

        :returns: Complete process-lifetime inventory view.
        """

        source_active = tuple(
            sorted(
                (
                    binding
                    for binding, lifecycle in self._source_lifecycles.items()
                    if len(lifecycle.inventory.live) > 0
                ),
                key=lambda binding: binding.digest,
            )
        )
        decode_active = tuple(
            sorted(
                (
                    binding
                    for binding, lifecycle in self._decode_lifecycles.items()
                    if len(lifecycle.inventory.live) > 0
                ),
                key=lambda binding: binding.digest,
            )
        )
        quarantined: list[TerminalOwnerQuarantineEntry] = []
        for binding, lifecycle in self._source_lifecycles.items():
            if len(lifecycle.inventory.quarantined) == 0:
                continue
            quarantined.append(
                TerminalOwnerQuarantineEntry(
                    binding=binding,
                    resources=lifecycle.inventory.quarantined,
                    reason=self._quarantine_reasons[binding],
                )
            )
        for binding, lifecycle in self._decode_lifecycles.items():
            if len(lifecycle.inventory.quarantined) == 0:
                continue
            quarantined.append(
                TerminalOwnerQuarantineEntry(
                    binding=binding,
                    resources=lifecycle.inventory.quarantined,
                    reason=self._quarantine_reasons[binding],
                )
            )
        queued_submissions = sum(
            source.pending_count for source in self._sources_by_name.values()
        )
        return TerminalOwnerSnapshot(
            disposition=self._disposition,
            admission_open=self._admission_open,
            reactor_alive=self._reactor_alive,
            source_active=source_active,
            decode_active=decode_active,
            safely_retired=tuple(
                sorted(self._retired_bindings, key=lambda binding: binding.digest)
            ),
            quarantined=tuple(
                sorted(quarantined, key=lambda entry: entry.binding.digest)
            ),
            pending_receipts=tuple(self._pending_receipts.values()),
            queued_submission_count=queued_submissions,
            queued_output_count=len(self._outputs),
            owner_transition_count=self._owner_transition_count,
            fatal_cause=self._fatal_cause,
            fatal_reason=self._fatal_reason,
            fatal_traceback=self._fatal_traceback,
        )
