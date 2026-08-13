import dataclasses
import threading
import time

from sglang.srt.disaggregation.terminal_progress.cuda_bridge import (
    CudaCompletionBridge,
    CudaCompletionFatalCode,
    CudaCompletionIdentity,
    CudaCompletionInventory,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TERMINAL_OWNER_COMMAND_TYPES,
    TerminalOwnerClosedError,
    TerminalOwnerCommandValue,
    TerminalOwnerError,
    TerminalOwnerEventEnvelope,
    TerminalOwnerEventSource,
    TerminalOwnerEventSourceFatalError,
)


@dataclasses.dataclass(frozen=True, slots=True)
class CudaTerminalOwnerEventSourceInventory:
    """Complete native and route inventory for one CUDA completion source.

    :ivar native: Latest native callback-bridge inventory.
    :ivar registered_routes: Identities awaiting exact callback completion.
    :ivar observed_unrouted: Completed identities observed without a route.
    :ivar admission_open: Whether new completion identities may be armed.
    :ivar closed: Whether exact-zero bridge closure completed.
    :ivar fatal_reason: Sticky source-fatal reason, when present.
    """

    native: CudaCompletionInventory
    registered_routes: tuple[CudaCompletionIdentity, ...]
    observed_unrouted: tuple[CudaCompletionIdentity, ...]
    admission_open: bool
    closed: bool
    fatal_reason: str | None

    @property
    def retained_count(self) -> int:
        """Return every owner or orphan completion identity retained.

        :returns: Combined route and unrouted identity count.
        """

        return len(self.registered_routes) + len(self.observed_unrouted)

    @property
    def producers_joined(self) -> bool:
        """Return the native callback-producer join attestation.

        :returns: Whether all callback and registration producers returned.
        """

        return self.native.producers_joined


class CudaTerminalOwnerEventSourceFatalError(TerminalOwnerEventSourceFatalError):
    """Process-fatal CUDA source result with complete typed inventory."""

    inventory: CudaTerminalOwnerEventSourceInventory

    def __init__(
        self,
        source_name: str,
        reason: str,
        inventory: CudaTerminalOwnerEventSourceInventory,
    ) -> None:
        """Retain native and exact routing evidence on a source fatal.

        :param source_name: Stable owner event-source name.
        :param reason: Exact native or routing failure.
        :param inventory: Complete typed inventory at failure observation.
        """

        if type(inventory) is not CudaTerminalOwnerEventSourceInventory:
            raise TypeError("inventory must be CudaTerminalOwnerEventSourceInventory")
        self.inventory = inventory
        identities = set(inventory.registered_routes) | set(inventory.observed_unrouted)
        if inventory.native.fatal_identity is not None:
            identities.add(inventory.native.fatal_identity)
        labels = tuple(
            _cuda_identity_label(identity)
            for identity in sorted(identities, key=_cuda_identity_sort_key)
        )
        super().__init__(source_name, reason, labels)


def _cuda_identity_sort_key(
    identity: CudaCompletionIdentity,
) -> tuple[int, bytes]:
    """Return deterministic inventory order for one callback identity.

    :param identity: Exact callback identity.
    :returns: Stable sortable fields.
    """

    return identity.cookie, identity.generation


def _cuda_identity_label(identity: CudaCompletionIdentity) -> str:
    """Render one lossless callback identity for process-fatal evidence.

    :param identity: Exact callback identity.
    :returns: Stable identity label.
    """

    return f"cuda:cookie={identity.cookie}:generation={identity.generation.hex()}"


class CudaTerminalOwnerEventSource(TerminalOwnerEventSource):
    """Zero-poll adapter from native CUDA callbacks into owner commands.

    Route insertion precedes callback submission. The CUDA callback remains a
    POD enqueue and eventfd wake; only this adapter's owner-side drain resolves
    the exact cookie and request generation into an immutable command.
    """

    _name: str
    _bridge: CudaCompletionBridge
    _routes: dict[CudaCompletionIdentity, TerminalOwnerCommandValue]
    _observed_unrouted: set[CudaCompletionIdentity]
    _next_sequence: int
    _admission_open: bool
    _closed: bool
    _last_native_inventory: CudaCompletionInventory
    _fatal_error: CudaTerminalOwnerEventSourceFatalError | None
    _lock: threading.Lock

    def __init__(self, name: str, bridge: CudaCompletionBridge) -> None:
        """Take exclusive ownership of one empty CUDA completion bridge.

        :param name: Stable source identity used in owner evidence.
        :param bridge: Empty process-lifetime native callback bridge.
        """

        if type(name) is not str or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        if type(bridge) is not CudaCompletionBridge:
            raise TypeError("bridge must be CudaCompletionBridge")
        inventory = bridge.inventory()
        if (
            inventory.live_count != 0
            or inventory.queued_count != 0
            or inventory.active_callback_count != 0
            or inventory.active_registration_count != 0
        ):
            raise ValueError("CUDA owner source requires an empty native bridge")
        if inventory.fatal_code is not CudaCompletionFatalCode.NONE:
            raise ValueError("CUDA owner source requires a healthy native bridge")
        if inventory.closed or not inventory.eventfd_open:
            raise ValueError("CUDA owner source requires an open native bridge")
        self._name = name
        self._bridge = bridge
        self._routes = {}
        self._observed_unrouted = set()
        self._next_sequence = 0
        self._admission_open = True
        self._closed = False
        self._last_native_inventory = inventory
        self._fatal_error = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return the stable registered source identity.

        :returns: Source name supplied at construction.
        """

        return self._name

    @property
    def pending_count(self) -> int:
        """Return the exact native callback-token count awaiting drain.

        :returns: Native queued callback count.
        """

        with self._lock:
            self._refresh_inventory_locked()
            return self._last_native_inventory.queued_count

    def fileno(self) -> int:
        """Return the native eventfd without creating a poll loop.

        :returns: Open native callback eventfd.
        """

        with self._lock:
            self._require_not_closed_locked()
            descriptor = self._bridge.fileno()
            if descriptor < 0:
                raise TerminalOwnerClosedError(
                    "CUDA terminal owner source descriptor is closed"
                )
            return descriptor

    def arm(
        self,
        identity: CudaCompletionIdentity,
        command: TerminalOwnerCommandValue,
    ) -> None:
        """Arm one exact callback identity and its immutable command route.

        :param identity: Opaque cookie and exact request generation.
        :param command: Command delivered only for this exact completion.
        """

        if type(identity) is not CudaCompletionIdentity:
            raise TypeError("identity must be CudaCompletionIdentity")
        if type(command) not in TERMINAL_OWNER_COMMAND_TYPES:
            raise TypeError("command must be an exact terminal owner command")
        with self._lock:
            self._require_accepting_locked()
            if identity in self._routes:
                raise ValueError("CUDA completion identity route is not unique")
            try:
                self._bridge.arm(identity)
            except RuntimeError as error:
                raise self._native_call_error_locked(
                    "native CUDA completion arm failed", error
                ) from error
            self._routes[identity] = command

    def submit(self, stream_handle: int, identity: CudaCompletionIdentity) -> None:
        """Attach a host callback after the exact scatter stream work.

        :param stream_handle: Raw ``cudaStream_t`` value owned by the caller.
        :param identity: Exact previously armed route identity.
        """

        if type(stream_handle) is not int or stream_handle < 0:
            raise ValueError("stream_handle must be a non-negative integer")
        if type(identity) is not CudaCompletionIdentity:
            raise TypeError("identity must be CudaCompletionIdentity")
        with self._lock:
            self._require_accepting_locked()
            if identity not in self._routes:
                raise ValueError("CUDA completion submission has no exact owner route")
            try:
                self._bridge.submit(stream_handle, identity)
            except RuntimeError as error:
                raise self._native_call_error_locked(
                    "native CUDA completion submission failed", error
                ) from error

    def drain(self) -> tuple[TerminalOwnerEventEnvelope, ...]:
        """Translate one complete callback wake into exact owner commands.

        :returns: Immutable owner envelopes in native callback queue order.
        :raises CudaTerminalOwnerEventSourceFatalError: If native health or
            exact route ownership is ambiguous.
        """

        with self._lock:
            self._require_not_closed_locked()
            if self._fatal_error is not None:
                raise self._fatal_error
            try:
                drained = self._bridge.drain()
            except RuntimeError as error:
                raise self._native_call_error_locked(
                    "native CUDA completion drain failed", error
                ) from error
            self._last_native_inventory = drained.inventory
            if drained.inventory.fatal_code is not CudaCompletionFatalCode.NONE:
                raise self._record_fatal_locked(
                    "native CUDA completion bridge is fatal"
                )

            commands: list[TerminalOwnerCommandValue] = []
            for identity in drained.identities:
                command = self._routes.get(identity)
                if command is None:
                    self._observed_unrouted.add(identity)
                    raise self._record_fatal_locked(
                        "CUDA completion has no exact owner route"
                    )
                commands.append(command)

            observed_ns = time.monotonic_ns()
            envelopes = tuple(
                TerminalOwnerEventEnvelope(
                    producer_sequence=self._next_sequence + index,
                    enqueued_ns=observed_ns,
                    command=command,
                )
                for index, command in enumerate(commands)
            )
            self._next_sequence += len(envelopes)
            for identity in drained.identities:
                del self._routes[identity]
            return envelopes

    def stop_submissions(self) -> None:
        """Permanently stop callback admission without waiting or polling."""

        with self._lock:
            if self._closed:
                return
            self._admission_open = False
            self._bridge.stop_submissions()
            self._refresh_inventory_locked()

    def join_producers(self) -> bool:
        """Try one native producer join without a fixed-cadence loop.

        :returns: Whether every callback and registration producer returned.
        """

        with self._lock:
            if self._closed:
                return True
            self._admission_open = False
            joined = self._bridge.join_producers()
            self._refresh_inventory_locked()
            return joined

    def inventory(self) -> CudaTerminalOwnerEventSourceInventory:
        """Return complete current native and routing inventory.

        :returns: Immutable teardown and fatal evidence.
        """

        with self._lock:
            self._refresh_inventory_locked()
            return self._inventory_locked()

    def close(self) -> None:
        """Stop, join once, and close only at exact zero inventory."""

        with self._lock:
            if self._closed:
                return
            self._admission_open = False
            self._bridge.stop_submissions()
            joined = self._bridge.join_producers()
            self._refresh_inventory_locked()
            if self._fatal_error is not None:
                raise self._fatal_error
            if not joined:
                raise TerminalOwnerError(
                    "CUDA owner source cannot close before producers join"
                )
            inventory = self._inventory_locked()
            if (
                inventory.retained_count != 0
                or inventory.native.live_count != 0
                or inventory.native.queued_count != 0
            ):
                raise TerminalOwnerError(
                    "CUDA owner source cannot close with retained identities"
                )
            try:
                self._bridge.close()
            except RuntimeError as error:
                raise self._native_call_error_locked(
                    "native CUDA completion close failed", error
                ) from error
            self._closed = True
            self._last_native_inventory = self._bridge.inventory()

    def _native_call_error_locked(
        self, reason: str, error: RuntimeError
    ) -> RuntimeError:
        """Translate a native exception only when its sticky state is fatal.

        :param reason: Native operation which raised.
        :param error: Original native runtime error.
        :returns: Typed source fatal, or the unchanged non-fatal error.
        """

        self._refresh_inventory_locked()
        if self._last_native_inventory.fatal_code is CudaCompletionFatalCode.NONE:
            return error
        return self._record_fatal_locked(
            f"{reason}: {self._last_native_inventory.fatal_code.value}"
        )

    def _refresh_inventory_locked(self) -> None:
        """Refresh native inventory and retain its first sticky fatal."""

        self._last_native_inventory = self._bridge.inventory()
        fatal_code = self._last_native_inventory.fatal_code
        if fatal_code is CudaCompletionFatalCode.NONE or self._fatal_error is not None:
            return
        self._record_fatal_locked(
            f"native CUDA completion bridge is fatal: {fatal_code.value}"
        )

    def _record_fatal_locked(
        self, reason: str
    ) -> CudaTerminalOwnerEventSourceFatalError:
        """Store the first source fatal with exact route and native inventory.

        :param reason: Precise native or routing invariant failure.
        :returns: Sticky typed source-fatal object.
        """

        if self._fatal_error is None:
            self._admission_open = False
            self._fatal_error = CudaTerminalOwnerEventSourceFatalError(
                source_name=self._name,
                reason=reason,
                inventory=self._inventory_locked(fatal_reason=reason),
            )
        return self._fatal_error

    def _inventory_locked(
        self, fatal_reason: str | None = None
    ) -> CudaTerminalOwnerEventSourceInventory:
        """Project complete immutable source inventory while holding the lock.

        :param fatal_reason: Prospective first fatal reason during construction.
        :returns: Complete typed inventory.
        """

        reason = fatal_reason
        if reason is None and self._fatal_error is not None:
            reason = self._fatal_error.reason
        return CudaTerminalOwnerEventSourceInventory(
            native=self._last_native_inventory,
            registered_routes=tuple(sorted(self._routes, key=_cuda_identity_sort_key)),
            observed_unrouted=tuple(
                sorted(self._observed_unrouted, key=_cuda_identity_sort_key)
            ),
            admission_open=self._admission_open,
            closed=self._closed,
            fatal_reason=reason,
        )

    def _require_accepting_locked(self) -> None:
        """Require healthy open callback admission."""

        self._require_not_closed_locked()
        if self._fatal_error is not None:
            raise self._fatal_error
        if not self._admission_open:
            raise TerminalOwnerClosedError(
                "CUDA terminal owner source admission is closed"
            )

    def _require_not_closed_locked(self) -> None:
        """Reject use after exact-zero bridge closure."""

        if self._closed:
            raise TerminalOwnerClosedError("CUDA terminal owner source is closed")
