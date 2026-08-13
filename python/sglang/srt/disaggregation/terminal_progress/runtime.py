import collections
import dataclasses
import enum
import errno
import os
import selectors
import threading
import time
import traceback
from collections.abc import Sequence

from sglang.srt.disaggregation.terminal_progress.native_owner import (
    NativeTerminalOwner,
)
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerActionKind,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerEventKind,
    NativeTerminalOwnerInventory,
    NativeTerminalOwnerOutput,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
    NativeTerminalReceipt,
)


class NativeTerminalRuntimeError(RuntimeError):
    """Process-lifetime native terminal runtime invariant violation."""


class NativeTerminalRuntimeClosedError(NativeTerminalRuntimeError):
    """Operation attempted outside the runtime's open lifecycle."""


class NativeTerminalRuntimeOverflowError(NativeTerminalRuntimeError):
    """A bounded runtime-owned action queue crossed its frozen capacity."""


class NativeTerminalProducerDelivery(enum.StrEnum):
    """Execution domain which owns one registered producer sequence."""

    PYTHON = "python"
    NATIVE = "native"


class NativeTerminalRuntimeDisposition(enum.StrEnum):
    """Process-level lifecycle of one terminal runtime."""

    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    ABORT_DRAINING = "abort_draining"
    STOPPED = "stopped"
    PROCESS_FATAL = "process_fatal"


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalRuntimeProducerSpec:
    """One producer registration and its exclusive submission domain.

    :ivar registration: Authority registered before native owner startup.
    :ivar delivery: Python or native ownership of its gap-free sequence.
    """

    registration: NativeTerminalProducerRegistration
    delivery: NativeTerminalProducerDelivery

    def __post_init__(self) -> None:
        """Validate the complete producer specification."""

        if type(self.registration) is not NativeTerminalProducerRegistration:
            raise TypeError("registration must be NativeTerminalProducerRegistration")
        if type(self.delivery) is not NativeTerminalProducerDelivery:
            raise TypeError("delivery must be NativeTerminalProducerDelivery")


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalNativeProducerBinding:
    """Opaque C ABI binding for one native-owned producer.

    :ivar producer_id: Exact producer namespace registered with the owner.
    :ivar producer_api: Versioned native producer API capsule.
    :ivar producer_context: Producer-specific native context capsule.
    """

    producer_id: int
    producer_api: object
    producer_context: object

    def __post_init__(self) -> None:
        """Validate the stable public producer identity."""

        if type(self.producer_id) is not int or self.producer_id < 0:
            raise ValueError("producer_id must be a non-negative integer")


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalActionInboxSnapshot:
    """Immutable bounded-inbox health and population.

    :ivar name: Stable consumer inbox identity.
    :ivar capacity: Frozen physical queue capacity.
    :ivar queued_count: Actions not yet drained by the consumer.
    :ivar closed: Whether the wake descriptors are closed.
    :ivar fatal_reason: Sticky process-fatal runtime reason, when present.
    """

    name: str
    capacity: int
    queued_count: int
    closed: bool
    fatal_reason: str | None

    def __post_init__(self) -> None:
        """Validate one conservative queue snapshot."""

        if type(self.name) is not str or len(self.name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            type(self.queued_count) is not int
            or self.queued_count < 0
            or self.queued_count > self.capacity
        ):
            raise ValueError("queued_count must fit within capacity")
        if type(self.closed) is not bool:
            raise TypeError("closed must be bool")
        if self.fatal_reason is not None and (
            type(self.fatal_reason) is not str or len(self.fatal_reason) == 0
        ):
            raise ValueError("fatal_reason must be a non-empty string")


class _BoundedFdInbox[ValueT]:
    """Bounded FIFO whose queue insertion precedes a coalesced fd wake."""

    _name: str
    _capacity: int
    _read_fd: int
    _write_fd: int
    _pending: collections.deque[ValueT]
    _wake_armed: bool
    _closed: bool
    _fatal_reason: str | None
    _lock: threading.Lock

    def __init__(self, name: str, capacity: int) -> None:
        """Construct one process-local bounded inbox.

        :param name: Stable consumer identity.
        :param capacity: Maximum queued immutable values.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        self._name = name
        self._capacity = capacity
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._pending = collections.deque()
        self._wake_armed = False
        self._closed = False
        self._fatal_reason = None
        self._lock = threading.Lock()

    def fileno(self) -> int:
        """Return the readable wake descriptor.

        :returns: Open inbox descriptor.
        """

        with self._lock:
            if self._closed:
                raise NativeTerminalRuntimeClosedError(
                    f"runtime inbox {self._name} is closed"
                )
            return self._read_fd

    def snapshot(self) -> NativeTerminalActionInboxSnapshot:
        """Return immutable queue health and population.

        :returns: Current bounded-inbox snapshot.
        """

        with self._lock:
            return NativeTerminalActionInboxSnapshot(
                name=self._name,
                capacity=self._capacity,
                queued_count=len(self._pending),
                closed=self._closed,
                fatal_reason=self._fatal_reason,
            )

    def drain(self, maximum_items: int | None = None) -> tuple[ValueT, ...]:
        """Drain a FIFO prefix after consuming the coalesced wake.

        :param maximum_items: Optional positive value-count bound.
        :returns: Immutable FIFO population.
        """

        if maximum_items is not None and (
            type(maximum_items) is not int or maximum_items <= 0
        ):
            raise ValueError("maximum_items must be a positive integer")
        self._drain_wake()
        with self._lock:
            if self._closed:
                raise NativeTerminalRuntimeClosedError(
                    f"runtime inbox {self._name} is closed"
                )
            count = len(self._pending)
            if maximum_items is not None:
                count = min(count, maximum_items)
            values = tuple(self._pending.popleft() for _ in range(count))
            self._wake_armed = False
            if len(self._pending) > 0:
                self._signal_locked()
            return values

    def _enqueue(self, value: ValueT) -> None:
        """Publish one value or fail before mutating a full queue.

        :param value: Immutable consumer value.
        """

        with self._lock:
            if self._closed:
                raise NativeTerminalRuntimeClosedError(
                    f"runtime inbox {self._name} is closed"
                )
            if len(self._pending) >= self._capacity:
                raise NativeTerminalRuntimeOverflowError(
                    f"runtime inbox {self._name} exceeded capacity {self._capacity}"
                )
            self._pending.append(value)
            self._signal_locked()

    def _mark_fatal(self, reason: str) -> None:
        """Wake consumers with one sticky process-fatal reason.

        :param reason: Exact runtime failure evidence.
        """

        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            if self._closed:
                return
            if self._fatal_reason is None:
                self._fatal_reason = reason
            self._signal_locked()

    def _close(self, *, require_empty: bool) -> None:
        """Close both descriptors at the exact consumer boundary.

        :param require_empty: Whether retained queue entries reject closure.
        """

        with self._lock:
            if self._closed:
                return
            if require_empty and len(self._pending) > 0:
                raise NativeTerminalRuntimeError(
                    f"runtime inbox {self._name} retained {len(self._pending)} values"
                )
            self._closed = True
            read_fd = self._read_fd
            write_fd = self._write_fd
        os.close(read_fd)
        os.close(write_fd)

    def _signal_locked(self) -> None:
        """Arm one wake byte while the queue lock protects coalescing."""

        if self._wake_armed:
            return
        try:
            os.write(self._write_fd, b"\x01")
        except BlockingIOError as error:
            raise NativeTerminalRuntimeOverflowError(
                f"runtime inbox {self._name} wake pipe is full"
            ) from error
        self._wake_armed = True

    def _drain_wake(self) -> None:
        """Consume every available wake byte without a cadence loop."""

        while True:
            try:
                value = os.read(self._read_fd, 4096)
            except BlockingIOError:
                return
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                raise
            if len(value) == 0 or len(value) < 4096:
                return


class NativeTerminalActionInbox(_BoundedFdInbox[NativeTerminalOwnerAction]):
    """Fd-signalled immutable action queue for one execution context."""


class NativeTerminalObservationInbox(_BoundedFdInbox[NativeTerminalOwnerOutput]):
    """Non-authoritative output-observation queue for metrics and evidence."""


class _RuntimeProducer:
    """Runtime-owned gap-free sequence for one Python producer."""

    registration: NativeTerminalProducerRegistration
    delivery: NativeTerminalProducerDelivery
    _next_sequence: int
    _retirement_requested: bool
    _lock: threading.Lock

    def __init__(self, spec: NativeTerminalRuntimeProducerSpec) -> None:
        """Construct one producer from its frozen pre-start specification.

        :param spec: Exact producer registration and execution domain.
        """

        self.registration = spec.registration
        self.delivery = spec.delivery
        self._next_sequence = 0
        self._retirement_requested = False
        self._lock = threading.Lock()


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalRuntimeSnapshot:
    """Complete process-lifetime runtime health projection.

    :ivar disposition: Runtime lifecycle state.
    :ivar owner: Authoritative native inventory.
    :ivar scheduler: Scheduler action queue state.
    :ivar coordinator: Request-coordinator action queue state.
    :ivar lifecycle: Teardown and health action queue state.
    :ivar source_work: Source gather, outcome, and ACK work queue state.
    :ivar decode_work: Decode scatter and teardown work queue state.
    :ivar publisher: Gateway publication work queue state.
    :ivar observations: Non-authoritative observation queue state.
    :ivar scheduler_live_count: Exact request generations awaiting scheduler
        authority consumption.
    :ivar scheduler_pending_count: Scheduler actions queued or handed out but
        not explicitly completed.
    :ivar consumer_pending_count: All functional actions awaiting explicit
        downstream acceptance or completion.
    :ivar quarantined_binding_digests: Fail-closed identities accepted by
        lifecycle consumers.
    :ivar output_reactor_alive: Whether the sole output consumer is alive.
    :ivar producers_joined: Whether external producer shutdown was attested.
    :ivar dropped_observation_count: Metrics observations omitted under
        bounded backpressure without gating functional progress.
    :ivar fatal_reason: Sticky runtime-side failure evidence.
    """

    disposition: NativeTerminalRuntimeDisposition
    owner: NativeTerminalOwnerInventory
    scheduler: NativeTerminalActionInboxSnapshot
    coordinator: NativeTerminalActionInboxSnapshot
    lifecycle: NativeTerminalActionInboxSnapshot
    source_work: NativeTerminalActionInboxSnapshot
    decode_work: NativeTerminalActionInboxSnapshot
    publisher: NativeTerminalActionInboxSnapshot
    observations: NativeTerminalActionInboxSnapshot
    scheduler_live_count: int
    scheduler_pending_count: int
    consumer_pending_count: int
    quarantined_binding_digests: tuple[bytes, ...]
    output_reactor_alive: bool
    producers_joined: bool
    dropped_observation_count: int
    fatal_reason: str | None

    def __post_init__(self) -> None:
        """Validate cross-layer runtime conservation."""

        if type(self.disposition) is not NativeTerminalRuntimeDisposition:
            raise TypeError("disposition must be NativeTerminalRuntimeDisposition")
        if type(self.owner) is not NativeTerminalOwnerInventory:
            raise TypeError("owner must be NativeTerminalOwnerInventory")
        inboxes = (
            self.scheduler,
            self.coordinator,
            self.lifecycle,
            self.source_work,
            self.decode_work,
            self.publisher,
            self.observations,
        )
        if any(
            type(inbox) is not NativeTerminalActionInboxSnapshot for inbox in inboxes
        ):
            raise TypeError("runtime inbox snapshots have an invalid type")
        counts = (
            self.scheduler_live_count,
            self.scheduler_pending_count,
            self.consumer_pending_count,
            self.dropped_observation_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("runtime counts must be non-negative integers")
        if self.scheduler_pending_count > self.scheduler_live_count:
            raise ValueError("scheduler pending actions exceed live requests")
        if type(self.quarantined_binding_digests) is not tuple or any(
            type(digest) is not bytes or len(digest) != 32
            for digest in self.quarantined_binding_digests
        ):
            raise ValueError("quarantined binding digests must contain 32 bytes")
        if len(set(self.quarantined_binding_digests)) != len(
            self.quarantined_binding_digests
        ):
            raise ValueError("quarantined binding digests must be unique")
        if type(self.output_reactor_alive) is not bool:
            raise TypeError("output_reactor_alive must be bool")
        if type(self.producers_joined) is not bool:
            raise TypeError("producers_joined must be bool")
        if self.fatal_reason is not None and (
            type(self.fatal_reason) is not str or len(self.fatal_reason) == 0
        ):
            raise ValueError("fatal_reason must be a non-empty string")


_SCHEDULER_ACTIONS = frozenset(
    (
        NativeTerminalOwnerActionKind.RECLAIM_AUTHORIZED,
        NativeTerminalOwnerActionKind.ADOPTION_READY,
    )
)
_COORDINATOR_ACTIONS = frozenset((NativeTerminalOwnerActionKind.LOCAL_DECODE_READY,))
_LIFECYCLE_ACTIONS = frozenset(
    (
        NativeTerminalOwnerActionKind.REQUEST_RETIRED,
        NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
        NativeTerminalOwnerActionKind.PROCESS_FATAL,
    )
)
_SOURCE_WORK_ACTIONS = frozenset(
    (
        NativeTerminalOwnerActionKind.SOURCE_GATHER_READY,
        NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY,
        NativeTerminalOwnerActionKind.SOURCE_ACK_READY,
    )
)
_DECODE_WORK_ACTIONS = frozenset(
    (
        NativeTerminalOwnerActionKind.DECODE_SCATTER_READY,
        NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY,
    )
)
_PUBLISHER_ACTIONS = frozenset(
    (NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY,)
)
_LOCAL_FAILURE_EVENTS = frozenset(
    (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
        NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
    )
)
_CONTROL_EVENTS = frozenset(
    (
        NativeTerminalOwnerEventKind.SOURCE_TEARDOWN_RECEIVED,
        NativeTerminalOwnerEventKind.DECODE_WRITER_AGGREGATION_STARTED,
        NativeTerminalOwnerEventKind.DECODE_WRITER_MANIFEST_COMPLETED,
        NativeTerminalOwnerEventKind.DECODE_ACK_AGGREGATION_STARTED,
        NativeTerminalOwnerEventKind.DECODE_ACK_MANIFEST_COMPLETED,
    )
)
_RECEIPT_EVENTS = frozenset(
    (
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_READY,
        NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED,
        NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
        NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED,
        NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
        NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED,
        NativeTerminalOwnerEventKind.DECODE_REQUEST_READY,
        NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
    )
)


class NativeTerminalRuntime:
    """Process-lifetime service around the authoritative native owner.

    The native reducer remains the sole lifecycle authority. This service owns
    producer namespaces, one output-drain thread, bounded fd-signalled consumer
    queues, and clean or fail-closed process teardown. It never invokes
    scheduler, coordinator, publisher, or metrics code on its reactor thread.
    """

    _owner: NativeTerminalOwner
    _owner_identity: NativeTerminalProcessIdentity
    _producers: dict[int, _RuntimeProducer]
    _producer_ids_by_name: dict[str, int]
    _python_producer_ids_by_authority: dict[
        tuple[NativeTerminalProducerClass, bytes | None], int
    ]
    _fatal_producer_id: int
    _scheduler_actions: NativeTerminalActionInbox
    _coordinator_actions: NativeTerminalActionInbox
    _lifecycle_actions: NativeTerminalActionInbox
    _source_work_actions: NativeTerminalActionInbox
    _decode_work_actions: NativeTerminalActionInbox
    _publisher_actions: NativeTerminalActionInbox
    _observations: NativeTerminalObservationInbox
    _scheduler_live: dict[bytes, NativeTerminalLifecycleRegistration]
    _scheduler_pending: dict[bytes, NativeTerminalOwnerAction]
    _consumer_pending: dict[int, NativeTerminalOwnerAction]
    _known_bindings: dict[bytes, NativeTerminalLifecycleRegistration]
    _quarantined_bindings: set[bytes]
    _condition: threading.Condition
    _disposition: NativeTerminalRuntimeDisposition
    _fatal_reason: str | None
    _output_reactor_alive: bool
    _producers_joined: bool
    _dropped_observation_count: int
    _stop_read_fd: int
    _stop_write_fd: int
    _output_thread: threading.Thread

    _OUTPUT_QUIESCENCE_TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        owner_identity: NativeTerminalProcessIdentity,
        producer_specs: tuple[NativeTerminalRuntimeProducerSpec, ...],
        fatal_producer_id: int,
        input_capacity: int,
        output_capacity: int,
        maximum_live_lifecycles: int,
        scheduler_capacity: int,
        coordinator_capacity: int,
        lifecycle_capacity: int,
        source_work_capacity: int,
        decode_work_capacity: int,
        publisher_capacity: int,
        observation_capacity: int,
    ) -> None:
        """Construct a dormant runtime and every process-lifetime queue.

        :param owner_identity: Exact role and TP-rank process incarnation.
        :param producer_specs: Complete producer registry frozen before start.
        :param fatal_producer_id: Python-local producer used for dependency
            death and bounded-inbox overflow.
        :param input_capacity: Native owner input capacity.
        :param output_capacity: Native owner action capacity.
        :param maximum_live_lifecycles: Bound for lifecycle admission and
            complete fail-closed output reserve.
        :param scheduler_capacity: Scheduler inbox capacity.
        :param coordinator_capacity: Request-coordinator inbox capacity.
        :param lifecycle_capacity: Teardown and health inbox capacity.
        :param source_work_capacity: Source continuation inbox capacity.
        :param decode_work_capacity: Decode continuation inbox capacity.
        :param publisher_capacity: Gateway publication inbox capacity.
        :param observation_capacity: Non-gating metrics queue capacity.
        """

        if type(owner_identity) is not NativeTerminalProcessIdentity:
            raise TypeError("owner_identity must be NativeTerminalProcessIdentity")
        if type(producer_specs) is not tuple or len(producer_specs) == 0:
            raise ValueError("producer_specs must be a non-empty tuple")
        if any(
            type(spec) is not NativeTerminalRuntimeProducerSpec
            for spec in producer_specs
        ):
            raise TypeError(
                "producer_specs must contain NativeTerminalRuntimeProducerSpec values"
            )
        capacities = (
            input_capacity,
            output_capacity,
            maximum_live_lifecycles,
            scheduler_capacity,
            coordinator_capacity,
            lifecycle_capacity,
            source_work_capacity,
            decode_work_capacity,
            publisher_capacity,
            observation_capacity,
        )
        if any(type(value) is not int or value <= 0 for value in capacities):
            raise ValueError("runtime capacities must be positive integers")
        producers: dict[int, _RuntimeProducer] = {}
        producer_ids_by_name: dict[str, int] = {}
        python_producer_ids_by_authority: dict[
            tuple[NativeTerminalProducerClass, bytes | None], int
        ] = {}
        for spec in producer_specs:
            registration = spec.registration
            if registration.allowed_role is not owner_identity.role:
                raise ValueError("every producer must target the runtime owner role")
            if registration.producer_id in producers:
                raise ValueError("runtime producer IDs must be unique")
            producers[registration.producer_id] = _RuntimeProducer(spec)
            if registration.name in producer_ids_by_name:
                raise ValueError("runtime producer names must be unique")
            producer_ids_by_name[registration.name] = registration.producer_id
            if spec.delivery is NativeTerminalProducerDelivery.PYTHON:
                issuer = registration.authenticated_issuer
                authority_key = (
                    registration.producer_class,
                    None if issuer is None else issuer.digest,
                )
                if authority_key in python_producer_ids_by_authority:
                    raise ValueError("Python producer authority routes must be unique")
                python_producer_ids_by_authority[authority_key] = (
                    registration.producer_id
                )
        fatal_producer = producers.get(fatal_producer_id)
        if fatal_producer is None:
            raise ValueError("fatal_producer_id is absent from producer_specs")
        if (
            fatal_producer.delivery is not NativeTerminalProducerDelivery.PYTHON
            or fatal_producer.registration.producer_class
            is not NativeTerminalProducerClass.LOCAL
        ):
            raise ValueError("fatal producer must be a Python-local producer")

        owner = NativeTerminalOwner(
            input_capacity=input_capacity,
            output_capacity=output_capacity,
            owner_identity=owner_identity,
            maximum_live_lifecycles=maximum_live_lifecycles,
        )
        for spec in producer_specs:
            owner.register_producer(spec.registration)
        stop_read_fd, stop_write_fd = os.pipe()
        os.set_blocking(stop_read_fd, False)
        os.set_blocking(stop_write_fd, False)
        os.set_inheritable(stop_read_fd, False)
        os.set_inheritable(stop_write_fd, False)

        self._owner = owner
        self._owner_identity = owner_identity
        self._producers = producers
        self._producer_ids_by_name = producer_ids_by_name
        self._python_producer_ids_by_authority = python_producer_ids_by_authority
        self._fatal_producer_id = fatal_producer_id
        self._scheduler_actions = NativeTerminalActionInbox(
            "packed-terminal-scheduler-actions", scheduler_capacity
        )
        self._coordinator_actions = NativeTerminalActionInbox(
            "packed-terminal-coordinator-actions", coordinator_capacity
        )
        self._lifecycle_actions = NativeTerminalActionInbox(
            "packed-terminal-lifecycle-actions", lifecycle_capacity
        )
        self._source_work_actions = NativeTerminalActionInbox(
            "packed-terminal-source-work-actions", source_work_capacity
        )
        self._decode_work_actions = NativeTerminalActionInbox(
            "packed-terminal-decode-work-actions", decode_work_capacity
        )
        self._publisher_actions = NativeTerminalActionInbox(
            "packed-terminal-publisher-actions", publisher_capacity
        )
        self._observations = NativeTerminalObservationInbox(
            "packed-terminal-observations", observation_capacity
        )
        self._scheduler_live = {}
        self._scheduler_pending = {}
        self._consumer_pending = {}
        self._known_bindings = {}
        self._quarantined_bindings = set()
        self._condition = threading.Condition()
        self._disposition = NativeTerminalRuntimeDisposition.CREATED
        self._fatal_reason = None
        self._output_reactor_alive = False
        self._producers_joined = False
        self._dropped_observation_count = 0
        self._stop_read_fd = stop_read_fd
        self._stop_write_fd = stop_write_fd
        self._output_thread = threading.Thread(
            target=self._run_output_reactor,
            name="packed-terminal-native-output-reactor",
            daemon=False,
        )

    @property
    def scheduler_actions(self) -> NativeTerminalActionInbox:
        """Return the scheduler-affine action inbox.

        :returns: Bounded fd-signalled scheduler queue.
        """

        return self._scheduler_actions

    @property
    def coordinator_actions(self) -> NativeTerminalActionInbox:
        """Return the request-coordinator action inbox.

        :returns: Bounded fd-signalled coordinator queue.
        """

        return self._coordinator_actions

    @property
    def lifecycle_actions(self) -> NativeTerminalActionInbox:
        """Return the teardown and health action inbox.

        :returns: Bounded fd-signalled lifecycle queue.
        """

        return self._lifecycle_actions

    @property
    def observations(self) -> NativeTerminalObservationInbox:
        """Return the non-gating metrics and evidence inbox.

        :returns: Bounded fd-signalled observation queue.
        """

        return self._observations

    @property
    def source_work_actions(self) -> NativeTerminalActionInbox:
        """Return owner-earned source gather, outcome, and ACK work.

        :returns: Bounded fd-signalled source worker queue.
        """

        return self._source_work_actions

    @property
    def decode_work_actions(self) -> NativeTerminalActionInbox:
        """Return owner-earned decode scatter and teardown work.

        :returns: Bounded fd-signalled decode worker queue.
        """

        return self._decode_work_actions

    @property
    def publisher_actions(self) -> NativeTerminalActionInbox:
        """Return request-global gateway publication work.

        :returns: Bounded fd-signalled publisher queue.
        """

        return self._publisher_actions

    def start(self) -> None:
        """Start the native owner and its sole output consumer exactly once."""

        with self._condition:
            if self._disposition is not NativeTerminalRuntimeDisposition.CREATED:
                raise NativeTerminalRuntimeClosedError("runtime cannot restart")
            self._owner.start()
            self._disposition = NativeTerminalRuntimeDisposition.RUNNING
            self._output_thread.start()
            while (
                not self._output_reactor_alive
                and self._disposition is NativeTerminalRuntimeDisposition.RUNNING
            ):
                self._condition.wait()
            if self._disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL:
                raise NativeTerminalRuntimeError(
                    "native output reactor failed during startup"
                )
            self._condition.notify_all()

    def python_producer_id(
        self,
        producer_class: NativeTerminalProducerClass,
        authenticated_issuer: NativeTerminalProcessIdentity | None = None,
    ) -> int:
        """Resolve one pre-registered Python authority route.

        :param producer_class: Exact authority class required by the event.
        :param authenticated_issuer: Process identity proved by a receipt or
            control route. Local producers require ``None``.
        :returns: Stable producer namespace frozen before runtime startup.
        """

        if type(producer_class) is not NativeTerminalProducerClass:
            raise TypeError("producer_class must be NativeTerminalProducerClass")
        if authenticated_issuer is not None and (
            type(authenticated_issuer) is not NativeTerminalProcessIdentity
        ):
            raise TypeError("authenticated_issuer has an invalid type")
        issuer_digest = (
            None if authenticated_issuer is None else authenticated_issuer.digest
        )
        producer_id = self._python_producer_ids_by_authority.get(
            (producer_class, issuer_digest)
        )
        if producer_id is None:
            raise KeyError("Python producer authority route is not registered")
        return producer_id

    def native_producer_binding(
        self, producer_name: str
    ) -> NativeTerminalNativeProducerBinding:
        """Return native ABI capsules for one exclusively native producer.

        :param producer_name: Stable pre-registered native producer name.
        :returns: API and producer-context capsules retained by the caller.
        """

        if type(producer_name) is not str or len(producer_name) == 0:
            raise ValueError("producer_name must be a non-empty string")
        producer_id = self._producer_ids_by_name.get(producer_name)
        if producer_id is None:
            raise KeyError(f"terminal producer {producer_name!r} is not registered")
        producer = self._require_producer(producer_id)
        if producer.delivery is not NativeTerminalProducerDelivery.NATIVE:
            raise NativeTerminalRuntimeError("producer sequence is Python-owned")
        return NativeTerminalNativeProducerBinding(
            producer_id=producer_id,
            producer_api=self._owner.producer_api(),
            producer_context=self._owner.producer_capsule(producer_id),
        )

    def register_lifecycle(
        self, registration: NativeTerminalLifecycleRegistration
    ) -> None:
        """Register one request before any producer event can target it.

        :param registration: Complete role-local lifecycle registration.
        """

        if type(registration) is not NativeTerminalLifecycleRegistration:
            raise TypeError("registration must be NativeTerminalLifecycleRegistration")
        if registration.binding.owner != self._owner_identity:
            raise ValueError("lifecycle belongs to another runtime owner")
        binding_digest = registration.binding.digest
        with self._condition:
            self._require_running_locked()
            existing = self._known_bindings.get(binding_digest)
            if existing is not None:
                if existing == registration:
                    raise NativeTerminalRuntimeError(
                        "lifecycle registration cannot be replayed"
                    )
                raise NativeTerminalRuntimeError(
                    "binding digest is already registered to another lifecycle"
                )
            self._known_bindings[binding_digest] = registration
            self._scheduler_live[binding_digest] = registration
        try:
            self._owner.register_lifecycle(registration)
        except Exception:
            with self._condition:
                self._known_bindings.pop(binding_digest, None)
                self._scheduler_live.pop(binding_digest, None)
            raise

    def submit(
        self,
        producer_id: int,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit one Python-owned producer event in gap-free order.

        :param producer_id: Exact Python-owned producer namespace.
        :param binding_digest: Exact lifecycle lookup digest.
        :param kind: Closed native lifecycle event.
        :param receipt: Authenticated one-shot receipt when required.
        :param reason: Stable failure evidence when required.
        :param enqueued_ns: Optional exact ``CLOCK_MONOTONIC_RAW`` timestamp.
        """

        with self._condition:
            self._require_event_admission_locked()
        producer = self._require_producer(producer_id)
        self._submit_with_producer(
            producer=producer,
            binding_digest=binding_digest,
            kind=kind,
            receipt=receipt,
            reason=reason,
            enqueued_ns=enqueued_ns,
        )

    def complete_scheduler_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        completion_receipt: NativeTerminalReceipt | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit exact scheduler consumption proof and retire its inbox slot.

        :param producer_id: Python receipt producer authenticated as this owner.
        :param action: Adoption or reclaim action previously drained.
        :param followup_kind: Matching adoption- or reclaim-consumed event.
        :param completion_receipt: Scheduler-minted reclaim-consumed authority.
            Decode adoption consumes the owner-minted action receipt directly.
        :param enqueued_ns: Optional exact monotonic timestamp.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind not in _SCHEDULER_ACTIONS or action.receipt is None:
            raise ValueError("action does not carry scheduler authority")
        expected_kind = NativeTerminalOwnerEventKind.SOURCE_RECLAIM_CONSUMED
        receipt = completion_receipt
        if action.kind is NativeTerminalOwnerActionKind.ADOPTION_READY:
            expected_kind = NativeTerminalOwnerEventKind.DECODE_ADOPTION_CONSUMED
            if completion_receipt is not None:
                raise ValueError("decode adoption uses the owner-minted receipt")
            receipt = action.receipt
        elif completion_receipt is None:
            raise ValueError("source reclaim requires a consumed receipt")
        if followup_kind is not expected_kind:
            raise ValueError("followup kind does not consume this scheduler action")
        binding_digest = action.binding.digest
        with self._condition:
            pending = self._scheduler_pending.get(binding_digest)
            if pending != action:
                raise NativeTerminalRuntimeError(
                    "scheduler action is absent, stale, or already completed"
                )
        self.submit(
            producer_id=producer_id,
            binding_digest=binding_digest,
            kind=followup_kind,
            receipt=receipt,
            enqueued_ns=enqueued_ns,
        )
        with self._condition:
            current = self._scheduler_pending.get(binding_digest)
            if current != action:
                self._enter_runtime_fatal_locked(
                    "scheduler action changed during exact consumption"
                )
                raise NativeTerminalRuntimeError(
                    "scheduler action changed during exact consumption"
                )
            del self._scheduler_pending[binding_digest]
            self._consumer_pending.pop(action.action_id)
            self._scheduler_live.pop(binding_digest, None)
            self._condition.notify_all()

    def submit_imported_receipt(
        self,
        producer_id: int,
        receipt: NativeTerminalReceipt,
        kind: NativeTerminalOwnerEventKind,
        *,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Submit one receipt already authenticated by its control route.

        :param producer_id: Producer bound to the receipt's exact issuer.
        :param receipt: Validated native receipt authority.
        :param kind: Receipt-consuming lifecycle event.
        :param reason: Failure evidence for failure receipts.
        :param enqueued_ns: Optional exact producer timestamp.
        """

        if type(receipt) is not NativeTerminalReceipt:
            raise TypeError("receipt must be NativeTerminalReceipt")
        self.submit(
            producer_id=producer_id,
            binding_digest=receipt.binding.digest,
            kind=kind,
            receipt=receipt,
            reason=reason,
            enqueued_ns=enqueued_ns,
        )

    def acknowledge_consumed_action(self, action: NativeTerminalOwnerAction) -> None:
        """Retire one non-scheduler action after downstream acceptance.

        The native action ledger is acknowledged immediately after bounded
        inbox admission. This method releases the runtime's consumer-side
        generation accounting only after the receiving component accepted the
        immutable action.

        :param action: Action previously drained from a runtime inbox.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if action.kind in _SCHEDULER_ACTIONS:
            raise ValueError("scheduler actions require complete_scheduler_action")
        with self._condition:
            pending = self._consumer_pending.get(action.action_id)
            if pending != action:
                raise NativeTerminalRuntimeError(
                    "consumer action is absent, stale, or already acknowledged"
                )
            del self._consumer_pending[action.action_id]
            if action.kind in (
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
                NativeTerminalOwnerActionKind.PROCESS_FATAL,
            ):
                binding_digest = action.binding.digest
                self._quarantined_bindings.add(binding_digest)
                if binding_digest not in self._scheduler_pending:
                    self._scheduler_live.pop(binding_digest, None)
            self._condition.notify_all()

    def acknowledge_aborted_action(self, action: NativeTerminalOwnerAction) -> None:
        """Discard one consumer action after fail-closed quarantine wins.

        :param action: Exact action drained during abort processing.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        with self._condition:
            if self._disposition is not NativeTerminalRuntimeDisposition.ABORT_DRAINING:
                raise NativeTerminalRuntimeError(
                    "aborted action acknowledgement requires fail-closed drain"
                )
            pending = self._consumer_pending.get(action.action_id)
            if pending != action:
                raise NativeTerminalRuntimeError(
                    "aborted action is absent, stale, or already acknowledged"
                )
            del self._consumer_pending[action.action_id]
            binding_digest = action.binding.digest
            if action.kind in _SCHEDULER_ACTIONS:
                self._scheduler_pending.pop(binding_digest, None)
            if action.kind in (
                NativeTerminalOwnerActionKind.REQUEST_QUARANTINED,
                NativeTerminalOwnerActionKind.PROCESS_FATAL,
            ):
                self._quarantined_bindings.add(binding_digest)
            if binding_digest in self._quarantined_bindings:
                if binding_digest not in self._scheduler_pending:
                    self._scheduler_live.pop(binding_digest, None)
            self._condition.notify_all()

    def complete_work_action(
        self,
        producer_id: int,
        action: NativeTerminalOwnerAction,
        followup_kind: NativeTerminalOwnerEventKind,
        *,
        receipt: NativeTerminalReceipt | None = None,
        reason: str | None = None,
        enqueued_ns: int | None = None,
    ) -> None:
        """Commit the exact lifecycle event earned by one work action.

        :param producer_id: Producer owning the continuation transition.
        :param action: Source, decode, or publisher work action.
        :param followup_kind: Exact successful or failed followup event.
        :param receipt: Publication authority when the followup requires it.
        :param reason: Stable failure evidence when the followup requires it.
        :param enqueued_ns: Optional exact producer timestamp.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        allowed = {
            NativeTerminalOwnerActionKind.SOURCE_GATHER_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.SOURCE_GATHER_POSTED,
                    NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                )
            ),
            NativeTerminalOwnerActionKind.SOURCE_OUTCOME_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.SOURCE_OUTCOMES_SENT,
                    NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                )
            ),
            NativeTerminalOwnerActionKind.SOURCE_ACK_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.SOURCE_ACK_SENT,
                    NativeTerminalOwnerEventKind.SOURCE_REQUEST_FAILED,
                )
            ),
            NativeTerminalOwnerActionKind.DECODE_SCATTER_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.DECODE_SCATTER_STARTED,
                    NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
                )
            ),
            NativeTerminalOwnerActionKind.DECODE_TEARDOWN_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.DECODE_TEARDOWN_SENT,
                    NativeTerminalOwnerEventKind.DECODE_REQUEST_FAILED,
                )
            ),
            NativeTerminalOwnerActionKind.GATEWAY_PUBLICATION_READY: frozenset(
                (
                    NativeTerminalOwnerEventKind.SOURCE_GATEWAY_PUBLISHED,
                    NativeTerminalOwnerEventKind.SOURCE_PUBLICATION_FAILED,
                )
            ),
        }
        allowed_kinds = allowed.get(action.kind)
        if allowed_kinds is None or followup_kind not in allowed_kinds:
            raise ValueError("followup kind was not earned by this work action")
        with self._condition:
            pending = self._consumer_pending.get(action.action_id)
            if pending != action:
                raise NativeTerminalRuntimeError(
                    "work action is absent, stale, or already completed"
                )
        self.submit(
            producer_id=producer_id,
            binding_digest=action.binding.digest,
            kind=followup_kind,
            receipt=receipt,
            reason=reason,
            enqueued_ns=enqueued_ns,
        )
        self.acknowledge_consumed_action(action)

    def stop_admission(self) -> None:
        """Close request and event admission before producer drain."""

        with self._condition:
            if self._disposition is NativeTerminalRuntimeDisposition.RUNNING:
                self._disposition = NativeTerminalRuntimeDisposition.DRAINING
            elif self._disposition not in (
                NativeTerminalRuntimeDisposition.DRAINING,
                NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            ):
                raise NativeTerminalRuntimeClosedError(
                    "runtime cannot begin drain from its current state"
                )
            self._owner.stop_admission()
            self._condition.notify_all()

    def retire_python_producer(self, producer_id: int) -> None:
        """Retire one Python producer after its execution context joined.

        :param producer_id: Exact pre-registered Python producer namespace.
        """

        producer = self._require_producer(producer_id)
        if producer.delivery is not NativeTerminalProducerDelivery.PYTHON:
            raise NativeTerminalRuntimeError(
                "native producer must retire through its API capsule"
            )
        with self._condition:
            if self._disposition not in (
                NativeTerminalRuntimeDisposition.DRAINING,
                NativeTerminalRuntimeDisposition.ABORT_DRAINING,
                NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            ):
                raise NativeTerminalRuntimeError(
                    "producer retirement requires closed lifecycle admission"
                )
        with producer._lock:
            if producer._retirement_requested:
                raise NativeTerminalRuntimeError("producer was already retired")
            self._owner.retire_python_producer(producer_id)
            producer._retirement_requested = True

    def join_producers(self) -> None:
        """Verify every registered producer retired and close event admission."""

        with self._condition:
            if self._disposition not in (
                NativeTerminalRuntimeDisposition.DRAINING,
                NativeTerminalRuntimeDisposition.ABORT_DRAINING,
                NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            ):
                raise NativeTerminalRuntimeError(
                    "producer join requires closed admission"
                )
            if not self._owner.join_producers():
                raise NativeTerminalRuntimeError(
                    "producer join rejected a live producer namespace"
                )
            self._producers_joined = True
            self._condition.notify_all()

    def snapshot(self) -> NativeTerminalRuntimeSnapshot:
        """Return native authority and all runtime-owned queue inventories.

        :returns: Complete immutable runtime snapshot.
        """

        owner = self._owner.inventory()
        with self._condition:
            return NativeTerminalRuntimeSnapshot(
                disposition=self._disposition,
                owner=owner,
                scheduler=self._scheduler_actions.snapshot(),
                coordinator=self._coordinator_actions.snapshot(),
                lifecycle=self._lifecycle_actions.snapshot(),
                source_work=self._source_work_actions.snapshot(),
                decode_work=self._decode_work_actions.snapshot(),
                publisher=self._publisher_actions.snapshot(),
                observations=self._observations.snapshot(),
                scheduler_live_count=len(self._scheduler_live),
                scheduler_pending_count=len(self._scheduler_pending),
                consumer_pending_count=len(self._consumer_pending),
                quarantined_binding_digests=tuple(sorted(self._quarantined_bindings)),
                output_reactor_alive=self._output_reactor_alive,
                producers_joined=self._producers_joined,
                dropped_observation_count=self._dropped_observation_count,
                fatal_reason=self._fatal_reason,
            )

    def close_clean(self) -> None:
        """Close only after exact-zero native and consumer inventories."""

        with self._condition:
            if self._disposition is not NativeTerminalRuntimeDisposition.DRAINING:
                raise NativeTerminalRuntimeError("clean close requires runtime drain")
            if not self._producers_joined:
                raise NativeTerminalRuntimeError(
                    "clean close requires external producer join"
                )
        inboxes: Sequence[_BoundedFdInbox[object]] = (
            self._scheduler_actions,
            self._coordinator_actions,
            self._lifecycle_actions,
            self._source_work_actions,
            self._decode_work_actions,
            self._publisher_actions,
            self._observations,
        )
        if not self._owner.wait_for_output_quiescence(
            self._OUTPUT_QUIESCENCE_TIMEOUT_SECONDS
        ):
            self.begin_abort()
            raise NativeTerminalRuntimeError(
                "clean close timed out and entered fail-closed drain"
            )
        with self._condition:
            if (
                len(self._scheduler_live) != 0
                or len(self._scheduler_pending) != 0
                or len(self._consumer_pending) != 0
            ):
                raise NativeTerminalRuntimeError(
                    "clean close acquired new consumer authority during drain"
                )
        if any(inbox.snapshot().queued_count != 0 for inbox in inboxes):
            raise NativeTerminalRuntimeError(
                "clean close acquired new consumer actions during drain"
            )
        inventory = self._owner.inventory()
        if (
            inventory.queued_input_count != 0
            or inventory.queued_output_count != 0
            or inventory.pending_action_count != 0
            or inventory.active_source_count != 0
            or inventory.active_decode_count != 0
            or inventory.quarantined_count != 0
            or inventory.armed_deadline_count != 0
            or inventory.output_drain_active
        ):
            raise NativeTerminalRuntimeError("clean close retains native work")
        self._stop_output_reactor()
        self._owner.close()
        self._close_inboxes(require_empty=True)
        with self._condition:
            self._disposition = NativeTerminalRuntimeDisposition.STOPPED
            self._condition.notify_all()

    def begin_abort(self) -> None:
        """Publish final quarantine authority while consumers remain alive."""

        with self._condition:
            if self._disposition is NativeTerminalRuntimeDisposition.STOPPED:
                raise NativeTerminalRuntimeClosedError("runtime is stopped")
            if self._disposition is NativeTerminalRuntimeDisposition.ABORT_DRAINING:
                return
            if self._disposition is NativeTerminalRuntimeDisposition.PROCESS_FATAL:
                self._disposition = NativeTerminalRuntimeDisposition.ABORT_DRAINING
                self._condition.notify_all()
            else:
                self._enter_runtime_fatal_locked(
                    self._fatal_reason or "runtime aborted before clean close"
                )
                self._disposition = NativeTerminalRuntimeDisposition.ABORT_DRAINING
        self._owner.begin_abort()

    def finish_abort_close(self) -> None:
        """Close only after consumers accepted all final native authority."""

        with self._condition:
            if self._disposition is NativeTerminalRuntimeDisposition.STOPPED:
                return
            if self._disposition is not NativeTerminalRuntimeDisposition.ABORT_DRAINING:
                raise NativeTerminalRuntimeError(
                    "abort finish requires fail-closed drain"
                )
            if not self._producers_joined:
                raise NativeTerminalRuntimeError(
                    "abort finish requires external producer join"
                )
        inboxes: Sequence[_BoundedFdInbox[object]] = (
            self._scheduler_actions,
            self._coordinator_actions,
            self._lifecycle_actions,
            self._source_work_actions,
            self._decode_work_actions,
            self._publisher_actions,
            self._observations,
        )
        if not self._owner.wait_for_output_quiescence(
            self._OUTPUT_QUIESCENCE_TIMEOUT_SECONDS
        ):
            raise NativeTerminalRuntimeError(
                "abort retained unrouted native terminal authority"
            )
        with self._condition:
            if (
                len(self._scheduler_live) != 0
                or len(self._scheduler_pending) != 0
                or len(self._consumer_pending) != 0
            ):
                raise NativeTerminalRuntimeError(
                    "abort finish retains unaccepted consumer authority"
                )
        if any(inbox.snapshot().queued_count != 0 for inbox in inboxes):
            raise NativeTerminalRuntimeError("abort finish retains consumer actions")
        inventory = self._owner.inventory()
        if set(inventory.quarantined_binding_digests) != self._quarantined_bindings:
            raise NativeTerminalRuntimeError(
                "native and consumer quarantine identities disagree"
            )
        self._stop_output_reactor()
        self._owner.close_aborted()
        self._close_inboxes(require_empty=True)
        with self._condition:
            self._disposition = NativeTerminalRuntimeDisposition.STOPPED
            self._condition.notify_all()

    def abort_and_close(self) -> None:
        """Begin fail-closed drain and finish it when already quiescent."""

        if self._disposition is NativeTerminalRuntimeDisposition.STOPPED:
            return
        self.begin_abort()
        self.finish_abort_close()

    def _require_producer(self, producer_id: int) -> _RuntimeProducer:
        """Resolve one exact registered producer.

        :param producer_id: Candidate producer namespace.
        :returns: Runtime-owned producer state.
        """

        if type(producer_id) is not int or producer_id < 0:
            raise ValueError("producer_id must be a non-negative integer")
        producer = self._producers.get(producer_id)
        if producer is None:
            raise KeyError(f"terminal producer {producer_id} is not registered")
        return producer

    def _submit_with_producer(
        self,
        producer: _RuntimeProducer,
        binding_digest: bytes,
        kind: NativeTerminalOwnerEventKind,
        receipt: NativeTerminalReceipt | None,
        reason: str | None,
        enqueued_ns: int | None,
    ) -> None:
        """Submit one event while holding its producer sequence lock.

        :param producer: Exact Python-owned producer state.
        :param binding_digest: Lifecycle lookup digest.
        :param kind: Closed native lifecycle event.
        :param receipt: Optional authenticated authority.
        :param reason: Optional stable failure evidence.
        :param enqueued_ns: Optional exact producer timestamp.
        """

        if producer.delivery is not NativeTerminalProducerDelivery.PYTHON:
            raise NativeTerminalRuntimeError("native producer rejects Python submit")
        if type(kind) is not NativeTerminalOwnerEventKind:
            raise TypeError("kind must be NativeTerminalOwnerEventKind")
        if receipt is not None and type(receipt) is not NativeTerminalReceipt:
            raise TypeError("receipt must be NativeTerminalReceipt")
        registration = producer.registration
        is_source_event = int(kind) < 40
        if is_source_event != (
            registration.allowed_role is NativeTerminalOwnerRole.SOURCE
        ):
            raise ValueError("producer role and event kind disagree")
        if receipt is not None and (
            registration.producer_class is NativeTerminalProducerClass.LOCAL
        ):
            raise ValueError("local producer cannot assert receipt authority")
        if (
            receipt is None
            and kind in _LOCAL_FAILURE_EVENTS
            and (registration.producer_class is not NativeTerminalProducerClass.LOCAL)
        ):
            raise ValueError("receipt-free request failure requires a local producer")
        local_failure = kind in _LOCAL_FAILURE_EVENTS and receipt is None
        if kind in _RECEIPT_EVENTS and not local_failure:
            if registration.producer_class is not NativeTerminalProducerClass.RECEIPT:
                raise ValueError("receipt event requires a receipt producer")
        elif kind in _CONTROL_EVENTS:
            if registration.producer_class is not NativeTerminalProducerClass.CONTROL:
                raise ValueError("control event requires a control producer")
        elif registration.producer_class is not NativeTerminalProducerClass.LOCAL:
            raise ValueError("local lifecycle event requires a local producer")
        timestamp_ns = enqueued_ns
        if timestamp_ns is None:
            timestamp_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        with producer._lock:
            if producer._retirement_requested:
                raise NativeTerminalRuntimeClosedError(
                    "retired producer cannot submit another event"
                )
            event = NativeTerminalOwnerEvent(
                producer_id=registration.producer_id,
                producer_sequence=producer._next_sequence,
                binding_digest=binding_digest,
                kind=kind,
                enqueued_ns=timestamp_ns,
                receipt=receipt,
                reason=reason,
            )
            self._owner.submit(event)
            producer._next_sequence += 1

    def _run_output_reactor(self) -> None:
        """Drain native actions into bounded execution-context inboxes."""

        selector = selectors.DefaultSelector()
        current_binding: bytes | None = None
        try:
            selector.register(self._owner.output_fileno(), selectors.EVENT_READ, 1)
            selector.register(self._stop_read_fd, selectors.EVENT_READ, 2)
            with self._condition:
                self._output_reactor_alive = True
                self._condition.notify_all()
            while True:
                ready = selector.select()
                should_stop = False
                for key, _ in ready:
                    if key.data == 2:
                        self._drain_stop_wake()
                        should_stop = True
                        continue
                    outputs = self._owner.drain_outputs()
                    for output in outputs:
                        current_binding = output.binding.digest
                        self._route_output(output)
                        current_binding = None
                if should_stop:
                    break
        except Exception:  # noqa: BLE001
            formatted_traceback = traceback.format_exc()
            self._fail_output_reactor(current_binding, formatted_traceback)
        finally:
            selector.close()
            with self._condition:
                self._output_reactor_alive = False
                expected_stop = self._disposition in (
                    NativeTerminalRuntimeDisposition.DRAINING,
                    NativeTerminalRuntimeDisposition.ABORT_DRAINING,
                    NativeTerminalRuntimeDisposition.PROCESS_FATAL,
                    NativeTerminalRuntimeDisposition.STOPPED,
                )
                if not expected_stop:
                    self._enter_runtime_fatal_locked(
                        "native output reactor exited outside lifecycle shutdown"
                    )
                self._condition.notify_all()

    def _route_output(self, output: NativeTerminalOwnerOutput) -> None:
        """Publish every native action before acknowledging its authority.

        :param output: One action-bearing native commit.
        """

        if type(output) is not NativeTerminalOwnerOutput:
            raise TypeError("output must be NativeTerminalOwnerOutput")
        if output.process_fatal:
            with self._condition:
                self._enter_runtime_fatal_locked(
                    f"native terminal owner entered {output.fatal_code.name}"
                )
        for action in output.actions:
            try:
                self._route_action(action)
            except Exception:  # noqa: BLE001
                formatted_traceback = traceback.format_exc()
                self._owner.fail_action_delivery(
                    action,
                    "runtime consumer rejected a native terminal action",
                )
                with self._condition:
                    self._enter_runtime_fatal_locked(formatted_traceback)
                continue
            self._owner.acknowledge_action(action)
        try:
            self._observations._enqueue(output)
        except NativeTerminalRuntimeOverflowError:
            with self._condition:
                self._dropped_observation_count += 1
                self._condition.notify_all()

    def _route_action(self, action: NativeTerminalOwnerAction) -> None:
        """Route one action to its sole owning execution context.

        :param action: Exact native authority or lifecycle observation.
        """

        binding_digest = action.binding.digest
        if action.kind is NativeTerminalOwnerActionKind.PROCESS_FATAL:
            with self._condition:
                self._retain_consumer_action_locked(action)
                try:
                    self._lifecycle_actions._enqueue(action)
                except Exception:
                    self._consumer_pending.pop(action.action_id, None)
                    raise
                self._condition.notify_all()
            return
        if action.kind in _SCHEDULER_ACTIONS:
            with self._condition:
                if binding_digest not in self._scheduler_live:
                    raise NativeTerminalRuntimeOverflowError(
                        "scheduler action targets a non-live request"
                    )
                if binding_digest in self._scheduler_pending:
                    raise NativeTerminalRuntimeOverflowError(
                        "scheduler action duplicates one request generation"
                    )
                if len(self._scheduler_pending) >= len(self._scheduler_live):
                    raise NativeTerminalRuntimeOverflowError(
                        "scheduler actions exceed in-flight request count"
                    )
                self._retain_consumer_action_locked(action)
                try:
                    self._scheduler_actions._enqueue(action)
                except Exception:
                    self._consumer_pending.pop(action.action_id, None)
                    raise
                self._scheduler_pending[binding_digest] = action
                self._condition.notify_all()
            return
        if action.kind in _COORDINATOR_ACTIONS:
            self._enqueue_consumer_action(self._coordinator_actions, action)
            return
        if action.kind in _SOURCE_WORK_ACTIONS:
            self._enqueue_consumer_action(self._source_work_actions, action)
            return
        if action.kind in _DECODE_WORK_ACTIONS:
            self._enqueue_consumer_action(self._decode_work_actions, action)
            return
        if action.kind in _PUBLISHER_ACTIONS:
            self._enqueue_consumer_action(self._publisher_actions, action)
            return
        if action.kind not in _LIFECYCLE_ACTIONS:
            raise NativeTerminalRuntimeError(
                f"native action {action.kind.name} has no consumer route"
            )
        self._enqueue_consumer_action(self._lifecycle_actions, action)
        if action.kind is NativeTerminalOwnerActionKind.REQUEST_RETIRED:
            with self._condition:
                if binding_digest not in self._scheduler_pending:
                    self._scheduler_live.pop(binding_digest, None)
                self._condition.notify_all()

    def _enqueue_consumer_action(
        self,
        inbox: NativeTerminalActionInbox,
        action: NativeTerminalOwnerAction,
    ) -> None:
        """Retain consumer ownership before making an action observable.

        :param inbox: Exact bounded destination queue.
        :param action: Native action transferring to that consumer.
        """

        with self._condition:
            self._retain_consumer_action_locked(action)
            try:
                inbox._enqueue(action)
            except Exception:
                self._consumer_pending.pop(action.action_id, None)
                raise
            self._condition.notify_all()

    def _retain_consumer_action_locked(self, action: NativeTerminalOwnerAction) -> None:
        """Record one globally unique native action under the runtime lock.

        :param action: Newly routed action.
        """

        if action.action_id in self._consumer_pending:
            raise NativeTerminalRuntimeOverflowError(
                "native action identity was routed more than once"
            )
        self._consumer_pending[action.action_id] = action

    def _fail_output_reactor(
        self, binding_digest: bytes | None, formatted_traceback: str
    ) -> None:
        """Enter native and runtime fatal state after output-path failure.

        :param binding_digest: Exact action binding, when routing had begun.
        :param formatted_traceback: Complete output-reactor stack trace.
        """

        with self._condition:
            if binding_digest is None and len(self._known_bindings) > 0:
                binding_digest = next(iter(self._known_bindings))
        if binding_digest is not None:
            producer = self._producers[self._fatal_producer_id]
            kind = NativeTerminalOwnerEventKind.SOURCE_INBOX_OVERFLOW
            if self._owner_identity.role is NativeTerminalOwnerRole.DECODE:
                kind = NativeTerminalOwnerEventKind.DECODE_INBOX_OVERFLOW
            try:
                self._submit_with_producer(
                    producer=producer,
                    binding_digest=binding_digest,
                    kind=kind,
                    receipt=None,
                    reason="native terminal output reactor failed",
                    enqueued_ns=None,
                )
            except Exception:  # noqa: BLE001
                formatted_traceback = (
                    f"{formatted_traceback}\n"
                    "Native fatal submission also failed:\n"
                    f"{traceback.format_exc()}"
                )
        with self._condition:
            self._enter_runtime_fatal_locked(formatted_traceback)

    def _enter_runtime_fatal_locked(self, reason: str) -> None:
        """Record one sticky runtime fatal and wake every consumer.

        :param reason: Complete failure evidence.
        """

        if self._fatal_reason is None:
            self._fatal_reason = reason
        self._disposition = NativeTerminalRuntimeDisposition.PROCESS_FATAL
        self._scheduler_actions._mark_fatal(self._fatal_reason)
        self._coordinator_actions._mark_fatal(self._fatal_reason)
        self._lifecycle_actions._mark_fatal(self._fatal_reason)
        self._source_work_actions._mark_fatal(self._fatal_reason)
        self._decode_work_actions._mark_fatal(self._fatal_reason)
        self._publisher_actions._mark_fatal(self._fatal_reason)
        self._observations._mark_fatal(self._fatal_reason)
        self._condition.notify_all()

    def _require_running_locked(self) -> None:
        """Require open non-fatal runtime admission."""

        if self._disposition is not NativeTerminalRuntimeDisposition.RUNNING:
            raise NativeTerminalRuntimeClosedError(
                f"runtime is {self._disposition.value}"
            )

    def _require_event_admission_locked(self) -> None:
        """Require a lifecycle-draining runtime with live producers."""

        if (
            self._disposition
            not in (
                NativeTerminalRuntimeDisposition.RUNNING,
                NativeTerminalRuntimeDisposition.DRAINING,
            )
            or self._producers_joined
        ):
            raise NativeTerminalRuntimeClosedError(
                f"runtime event admission is closed in {self._disposition.value}"
            )

    def _stop_output_reactor(self) -> None:
        """Wake and join the output reactor without cadence polling."""

        if self._output_thread.is_alive():
            os.write(self._stop_write_fd, b"\x01")
            self._output_thread.join()
        os.close(self._stop_read_fd)
        os.close(self._stop_write_fd)

    def _drain_stop_wake(self) -> None:
        """Consume the one-shot output-reactor stop wake."""

        while True:
            try:
                value = os.read(self._stop_read_fd, 4096)
            except BlockingIOError:
                return
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                raise
            if len(value) == 0 or len(value) < 4096:
                return

    def _close_inboxes(self, *, require_empty: bool) -> None:
        """Close every consumer descriptor under one lifecycle decision.

        :param require_empty: Whether any retained entry rejects closure.
        """

        self._scheduler_actions._close(require_empty=require_empty)
        self._coordinator_actions._close(require_empty=require_empty)
        self._lifecycle_actions._close(require_empty=require_empty)
        self._source_work_actions._close(require_empty=require_empty)
        self._decode_work_actions._close(require_empty=require_empty)
        self._publisher_actions._close(require_empty=require_empty)
        self._observations._close(require_empty=require_empty)
