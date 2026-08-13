import dataclasses
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from sglang.srt.disaggregation.terminal_progress.native_state import (
    NATIVE_DECODE_RESOURCE_MASK,
    NATIVE_SOURCE_RESOURCE_MASK,
    NativeTerminalLifecycleRegistration,
    NativeTerminalOwnerAction,
    NativeTerminalOwnerEvent,
    NativeTerminalOwnerInventory,
    NativeTerminalOwnerOutput,
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerRegistration,
    canonical_native_terminal_deadlines,
    native_terminal_deadline_table_digest,
)
from torch.utils.cpp_extension import load


class _NativeTerminalOwnerBridge(Protocol):
    """Typed boundary implemented by the native terminal-owner extension."""

    def register_producer(
        self,
        producer_id: int,
        name: str,
        producer_class: int,
        allowed_role: int,
        authenticated_issuer: Mapping[str, object] | None,
    ) -> None:
        """Register one process-lifetime event producer before startup.

        :param producer_id: Exact producer namespace.
        :param name: Stable evidence-facing producer name.
        :param producer_class: Native authority class code.
        :param allowed_role: Lifecycle role this producer may mutate.
        :param authenticated_issuer: Route-authenticated issuer, when required.
        """

    def register_source(self, registration: Mapping[str, object]) -> int:
        """Enqueue one source lifecycle in the ordered reactor domain.

        :param registration: Exact source binding and publication identity.
        :returns: Zero on ordered admission, otherwise a positive errno value.
        """

    def register_decode(self, registration: Mapping[str, object]) -> int:
        """Enqueue one decode lifecycle in the ordered reactor domain.

        :param registration: Exact decode binding and trusted issuers.
        :returns: Zero on ordered admission, otherwise a positive errno value.
        """

    def start(self) -> None:
        """Start the process-lifetime native reactor."""

    def submit_event(self, event: Mapping[str, object]) -> int:
        """Submit one producer-ordered lifecycle event.

        :param event: Fixed-width event record.
        :returns: Zero on acceptance, otherwise a positive errno value.
        """

    def producer_api(self) -> object:
        """Return the versioned native producer API capsule.

        :returns: PyCapsule carrying the producer ABI table.
        """

    def producer_capsule(self, producer_id: int) -> object:
        """Return the exact context capsule for one registered producer.

        :param producer_id: Registered producer identity.
        :returns: Producer context PyCapsule.
        """

    def output_fileno(self) -> int:
        """Return the production-action eventfd.

        :returns: Open pollable descriptor.
        """

    def drain_outputs(self) -> list[Mapping[str, object]]:
        """Drain every production action committed before the wake.

        :returns: Immutable output records.
        """

    def acknowledge_action(self, action_id: int) -> None:
        """Consume one exact production action identity.

        :param action_id: One-shot native action identity.
        """

    def fail_action_delivery(self, action_id: int, reason: str) -> None:
        """Reject one action and atomically enter fail-closed authority.

        :param action_id: Exact action which no consumer could accept.
        :param reason: Stable consumer-boundary failure evidence.
        """

    def inventory(self) -> Mapping[str, object]:
        """Return the complete process-lifetime owner inventory.

        :returns: Native inventory mapping.
        """

    def lifecycle_snapshot(self, binding_digest: bytes) -> Mapping[str, object]:
        """Return one lifecycle from a native test build.

        :param binding_digest: Exact request-generation digest.
        :returns: Immutable native lifecycle mapping.
        """

    def wait_for_lifecycle_registration(
        self, binding_digest: bytes, timeout_seconds: float
    ) -> bool:
        """Wait for ordered reactor registration of one binding.

        :param binding_digest: Exact request-generation digest.
        :param timeout_seconds: Positive wall-clock bound.
        :returns: Whether the binding became reactor-visible.
        """

    def enable_test_clock(self, now_ns: int) -> None:
        """Enable the deterministic clock before reactor startup.

        :param now_ns: Positive initial monotonic time.
        """

    def set_test_clock(self, now_ns: int) -> None:
        """Advance the deterministic clock monotonically.

        :param now_ns: New monotonic time.
        """

    def expire_deadlines_for_test(self) -> None:
        """Evaluate all armed deadlines at deterministic test time."""

    def abort_active_qualification_for_test(self) -> None:
        """Synchronously stop an active qualification test population."""

    def start_qualification(
        self,
        machine_count: int,
        minimum_duration_seconds: float,
        minimum_transition_count: int,
    ) -> None:
        """Start the closed-loop real-reducer qualification producer.

        :param machine_count: Concurrent native state machines.
        :param minimum_duration_seconds: Sustained-duration floor.
        :param minimum_transition_count: Transition-count floor.
        """

    def qualification_join(self, timeout_seconds: float) -> bool:
        """Wait for the native qualification producer to retire.

        :param timeout_seconds: Positive wait bound.
        :returns: Whether qualification completed within the bound.
        """

    def qualification_summary(self) -> Mapping[str, object]:
        """Return native population statistics and bounded audit samples.

        :returns: Exact native summary mapping.
        """

    def stop_admission(self) -> None:
        """Close lifecycle and event admission."""

    def retire_python_producer(self, producer_id: int) -> int:
        """Order one Python producer's retirement behind accepted events.

        :param producer_id: Exact registered producer identity.
        :returns: Zero on ordered admission, otherwise a positive errno value.
        """

    def wait_for_producer_retirement(
        self, producer_id: int, timeout_seconds: float
    ) -> bool:
        """Wait for an ordered producer retirement to commit.

        :param producer_id: Exact registered producer identity.
        :param timeout_seconds: Positive wall-clock bound.
        :returns: Whether retirement committed within the bound.
        """

    def join_producers(self) -> bool:
        """Verify that every registered producer retired.

        :returns: Whether the complete producer registry is retired.
        """

    def wait_for_output_quiescence(self, timeout_seconds: float) -> bool:
        """Wait for all swapped native actions to finish routing.

        :param timeout_seconds: Positive hash-bound shutdown timeout.
        :returns: Whether no queued, swapped, or unacknowledged output remains.
        """

    def begin_abort(self) -> None:
        """Quarantine unresolved lifecycles and stop the native reactor."""

    def close_aborted(self) -> None:
        """Close descriptors after final quarantine actions were routed."""

    def close(self) -> None:
        """Close a fully drained and retired owner."""

    def abort_and_close(self) -> None:
        """Fail closed and release the reactor after an incomplete run."""


@dataclasses.dataclass(frozen=True, slots=True)
class NativeTerminalLifecycleSnapshot:
    """Test-visible immutable projection of one native lifecycle.

    :ivar binding_digest: Exact request-generation identity.
    :ivar role: Native owner role.
    :ivar phase: Stable role-specific phase code.
    :ivar live_resources: Resources still pinned by the owner.
    :ivar retired_resources: Resources carrying exact reuse proof.
    :ivar quarantined_resources: Resources retained fail-closed.
    :ivar armed_deadline_mask: Deadlines active for this lifecycle.
    :ivar process_fatal: Whether process continuation became unsafe.
    """

    binding_digest: bytes
    role: NativeTerminalOwnerRole
    phase: int
    live_resources: int
    retired_resources: int
    quarantined_resources: int
    armed_deadline_mask: int
    process_fatal: bool

    def __post_init__(self) -> None:
        """Validate exact identity and resource conservation."""

        if type(self.binding_digest) is not bytes or len(self.binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if type(self.role) is not NativeTerminalOwnerRole:
            raise TypeError("role must be NativeTerminalOwnerRole")
        values = (
            self.phase,
            self.live_resources,
            self.retired_resources,
            self.quarantined_resources,
            self.armed_deadline_mask,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("native lifecycle fields must be non-negative integers")
        if type(self.process_fatal) is not bool:
            raise TypeError("process_fatal must be bool")
        partitions = (
            self.live_resources,
            self.retired_resources,
            self.quarantined_resources,
        )
        if (
            partitions[0] & partitions[1] != 0
            or partitions[0] & partitions[2] != 0
            or partitions[1] & partitions[2] != 0
        ):
            raise ValueError("native lifecycle resource partitions overlap")
        expected = (
            NATIVE_SOURCE_RESOURCE_MASK
            if self.role is NativeTerminalOwnerRole.SOURCE
            else NATIVE_DECODE_RESOURCE_MASK
        )
        if partitions[0] | partitions[1] | partitions[2] != expected:
            raise ValueError("native lifecycle resource partitions are incomplete")

    @classmethod
    def from_native(
        cls, value: Mapping[str, object]
    ) -> "NativeTerminalLifecycleSnapshot":
        """Parse one native inventory lifecycle.

        :param value: Raw immutable lifecycle mapping from the bridge.
        :returns: Validated lifecycle snapshot.
        """

        binding_value = value["binding"]
        if not isinstance(binding_value, Mapping):
            raise TypeError("native lifecycle binding must be a mapping")
        return cls(
            binding_digest=bytes(binding_value["digest"]),
            role=NativeTerminalOwnerRole(int(value["role"])),
            phase=int(value["phase"]),
            live_resources=int(value["live_resources"]),
            retired_resources=int(value["retired_resources"]),
            quarantined_resources=int(value["quarantined_resources"]),
            armed_deadline_mask=int(value["armed_deadline_mask"]),
            process_fatal=bool(value["process_fatal"]),
        )


def _native_source_path() -> Path:
    """Return the packaged authoritative native owner source.

    :returns: Absolute C++ source path.
    """

    return Path(__file__).with_name("native_owner_bridge.cpp")


def _native_producer_header_path() -> Path:
    """Return the shared native producer ABI header.

    :returns: Absolute packaged header path.
    """

    return Path(__file__).with_name("native_producer_api.h")


@lru_cache(maxsize=2)
def load_native_terminal_owner_module(*, testing: bool = False) -> ModuleType:
    """Compile and load the CPU-only authoritative owner bridge.

    The module name binds the source digest and build variant. A test build may
    expose deterministic clock controls, while production builds retain native
    ``CLOCK_MONOTONIC_RAW`` ownership.

    :param testing: Whether native test-only controls are required.
    :returns: Loaded pybind11 extension module.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("native terminal owner requires Linux eventfd and timerfd")
    source_path = _native_source_path()
    header_path = _native_producer_header_path()
    for required_path in (source_path, header_path):
        if not required_path.is_file():
            raise RuntimeError(
                f"native terminal owner source is absent: {required_path}"
            )
    source_hasher = hashlib.sha256()
    for required_path in (source_path, header_path):
        source_hasher.update(required_path.name.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(required_path.read_bytes())
    source_digest = source_hasher.hexdigest()[:12]
    variant = "test" if testing else "runtime"
    module_name = f"sglang_terminal_owner_bridge_{source_digest}_{variant}"
    extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
    if testing:
        extra_cflags.append("-DSGLANG_TERMINAL_OWNER_TESTING")
    build_directory_value = os.environ.get("SGLANG_TERMINAL_OWNER_BUILD_DIR")
    build_directory: str | None = None
    if build_directory_value is not None:
        build_path = Path(build_directory_value).resolve() / module_name
        build_path.mkdir(parents=True, exist_ok=True)
        build_directory = str(build_path)
    return load(
        name=module_name,
        sources=[str(source_path)],
        extra_cflags=extra_cflags,
        extra_include_paths=[str(source_path.parent)],
        extra_ldflags=["-pthread"],
        build_directory=build_directory,
        with_cuda=False,
        verbose=False,
    )


class NativeTerminalOwner:
    """Typed process-lifetime facade over the authoritative native reducer."""

    _native: _NativeTerminalOwnerBridge
    _closed: bool
    _testing: bool

    def __init__(
        self,
        input_capacity: int,
        output_capacity: int,
        owner_identity: NativeTerminalProcessIdentity,
        *,
        maximum_live_lifecycles: int | None = None,
        testing: bool = False,
    ) -> None:
        """Construct one owner before producer and lifecycle registration.

        :param input_capacity: Bounded native event capacity.
        :param output_capacity: Bounded production-action capacity.
        :param owner_identity: Exact process and tensor-parallel identity.
        :param maximum_live_lifecycles: Bound for complete fail-closed output.
        :param testing: Whether the native test variant is required.
        """

        if maximum_live_lifecycles is None:
            maximum_live_lifecycles = input_capacity
        capacities = (input_capacity, output_capacity, maximum_live_lifecycles)
        if any(type(value) is not int or value <= 0 for value in capacities):
            raise ValueError("native owner capacities must be positive integers")
        if type(owner_identity) is not NativeTerminalProcessIdentity:
            raise TypeError("owner_identity must be NativeTerminalProcessIdentity")
        deadline_table = tuple(
            deadline.to_native() for deadline in canonical_native_terminal_deadlines()
        )
        if any("process_fatal" not in deadline for deadline in deadline_table):
            raise RuntimeError(
                "native deadline ABI lacks its process-fatal disposition"
            )
        module = load_native_terminal_owner_module(testing=testing)
        self._native = cast(
            _NativeTerminalOwnerBridge,
            module.NativeTerminalOwnerBridge(
                input_capacity,
                output_capacity,
                maximum_live_lifecycles,
                owner_identity.to_native(),
                deadline_table,
                native_terminal_deadline_table_digest(),
            ),
        )
        self._closed = False
        self._testing = testing

    def register_producer(
        self, registration: NativeTerminalProducerRegistration
    ) -> None:
        """Register one producer before the reactor starts.

        :param registration: Complete producer identity and authority.
        """

        if type(registration) is not NativeTerminalProducerRegistration:
            raise TypeError("registration must be NativeTerminalProducerRegistration")
        value = registration.to_native()
        issuer = value["authenticated_issuer"]
        if issuer is not None and not isinstance(issuer, Mapping):
            raise TypeError("native producer issuer must be a mapping")
        self._native.register_producer(
            int(value["producer_id"]),
            str(value["name"]),
            int(value["producer_class"]),
            int(value["allowed_role"]),
            issuer,
        )

    def register_lifecycle(
        self, registration: NativeTerminalLifecycleRegistration
    ) -> None:
        """Enqueue one dynamic lifecycle before its first ordered event.

        :param registration: Complete role-specific lifecycle identity.
        """

        if type(registration) is not NativeTerminalLifecycleRegistration:
            raise TypeError("registration must be NativeTerminalLifecycleRegistration")
        value = registration.to_native()
        if registration.binding.owner.role is NativeTerminalOwnerRole.SOURCE:
            status = int(self._native.register_source(value))
            if status != 0:
                raise OSError(status, os.strerror(status))
            return
        status = int(self._native.register_decode(value))
        if status != 0:
            raise OSError(status, os.strerror(status))

    def start(self) -> None:
        """Start the process-lifetime native reactor."""

        self._native.start()

    def enable_test_clock(self, now_ns: int) -> None:
        """Enable deterministic deadline time before test-owner startup.

        :param now_ns: Positive initial monotonic timestamp.
        :raises RuntimeError: If this owner is not a test build.
        """

        if not self._testing:
            raise RuntimeError("deterministic clocks require a native test build")
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError("now_ns must be a positive integer")
        self._native.enable_test_clock(now_ns)

    def set_test_clock(self, now_ns: int) -> None:
        """Advance deterministic test-owner time monotonically.

        :param now_ns: New monotonic timestamp.
        :raises RuntimeError: If this owner is not a test build.
        """

        if not self._testing:
            raise RuntimeError("deterministic clocks require a native test build")
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError("now_ns must be a positive integer")
        self._native.set_test_clock(now_ns)

    def expire_deadlines_for_testing(self) -> None:
        """Evaluate armed deadlines without sleeping in a native test build.

        :raises RuntimeError: If this owner is not a test build.
        """

        if not self._testing:
            raise RuntimeError("deterministic clocks require a native test build")
        self._native.expire_deadlines_for_test()

    def abort_active_qualification_for_testing(self) -> None:
        """Synchronously stop an active native qualification population.

        This test-only hook proves that destructor-equivalent shutdown cannot
        race the closed-loop producer into replenishing its input queue.

        :raises RuntimeError: If this owner is not a test build.
        """

        if not self._testing:
            raise RuntimeError("qualification abort requires a native test build")
        self._native.abort_active_qualification_for_test()
        self._closed = True

    def submit(self, event: NativeTerminalOwnerEvent) -> None:
        """Submit one exact event or surface the native errno result.

        :param event: Producer-bound lifecycle event.
        :raises OSError: If native admission rejects the event synchronously.
        """

        if type(event) is not NativeTerminalOwnerEvent:
            raise TypeError("event must be NativeTerminalOwnerEvent")
        status = int(self._native.submit_event(event.to_native()))
        if status != 0:
            raise OSError(status, os.strerror(status))

    def submit_unchecked_for_testing(self, event: Mapping[str, object]) -> None:
        """Submit a structurally adversarial event to a native test build.

        Production callers use :meth:`submit`, whose immutable Python types
        reject malformed authority before the native boundary. Differential
        qualification deliberately needs to present forged bindings, receipt
        kinds, outcomes, and replays to the native reducer itself.

        :param event: Complete raw native event mapping.
        :raises RuntimeError: If this owner was not built for testing.
        :raises OSError: If native admission rejects the event synchronously.
        """

        if not self._testing:
            raise RuntimeError("unchecked submission requires a native test build")
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        status = int(self._native.submit_event(dict(event)))
        if status != 0:
            raise OSError(status, os.strerror(status))

    def output_fileno(self) -> int:
        """Return the production-action eventfd.

        :returns: Open pollable descriptor.
        """

        return int(self._native.output_fileno())

    def drain_outputs(self) -> tuple[NativeTerminalOwnerOutput, ...]:
        """Drain and validate every newly earned production action.

        :returns: Immutable typed native commit outputs.
        """

        values = self._native.drain_outputs()
        return tuple(NativeTerminalOwnerOutput.from_native(value) for value in values)

    def acknowledge_action(self, action: NativeTerminalOwnerAction) -> None:
        """Acknowledge one exact one-shot production action.

        :param action: Action previously returned by :meth:`drain_outputs`.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        self._native.acknowledge_action(action.action_id)

    def fail_action_delivery(
        self, action: NativeTerminalOwnerAction, reason: str
    ) -> None:
        """Reject one action and enter the native process-fatal path.

        :param action: Action which failed bounded consumer admission.
        :param reason: Complete consumer-boundary failure evidence.
        """

        if type(action) is not NativeTerminalOwnerAction:
            raise TypeError("action must be NativeTerminalOwnerAction")
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("reason must be a non-empty string")
        self._native.fail_action_delivery(action.action_id, reason)

    def inventory(self) -> NativeTerminalOwnerInventory:
        """Return the complete validated native owner inventory.

        :returns: Immutable inventory projection.
        """

        return NativeTerminalOwnerInventory.from_native(self._native.inventory())

    def lifecycle_snapshot_for_testing(
        self, binding_digest: bytes
    ) -> NativeTerminalLifecycleSnapshot:
        """Return one exact lifecycle projection from a native test build.

        Actionless commits intentionally never enter the production output
        queue. This read-only projection lets the differential qualification
        compare those commits without creating a Python-owned transition path.

        :param binding_digest: Exact registered request-generation digest.
        :returns: Validated immutable native lifecycle snapshot.
        :raises RuntimeError: If this owner was not built for testing.
        :raises KeyError: If the exact lifecycle is absent.
        """

        if not self._testing:
            raise RuntimeError("lifecycle snapshots require a native test build")
        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        try:
            native_snapshot = self._native.lifecycle_snapshot(binding_digest)
        except ValueError as error:
            raise KeyError("native lifecycle snapshot binding is unknown") from error
        return NativeTerminalLifecycleSnapshot.from_native(native_snapshot)

    def wait_for_lifecycle_registration(
        self, binding_digest: bytes, timeout_seconds: float
    ) -> bool:
        """Wait for one ordered lifecycle registration to commit.

        This is a synchronization receipt for callers which need to know that
        the reactor has consumed registration. First events do not need this
        wait when submitted through the same ordered input domain.

        :param binding_digest: Exact request-generation digest.
        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: Whether the lifecycle became reactor-visible.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(
            self._native.wait_for_lifecycle_registration(
                binding_digest,
                timeout_seconds,
            )
        )

    def producer_api(self) -> object:
        """Return the versioned producer ABI capsule.

        :returns: Native producer API PyCapsule.
        """

        return self._native.producer_api()

    def producer_capsule(self, producer_id: int) -> object:
        """Return the context capsule for one registered producer.

        :param producer_id: Exact registered producer identity.
        :returns: Native producer context PyCapsule.
        """

        if type(producer_id) is not int or producer_id < 0:
            raise ValueError("producer_id must be a non-negative integer")
        return self._native.producer_capsule(producer_id)

    def start_qualification(
        self,
        machine_count: int,
        minimum_duration_seconds: float,
        minimum_transition_count: int,
    ) -> None:
        """Start native closed-loop qualification on a dedicated owner.

        :param machine_count: Concurrent lifecycle machine count.
        :param minimum_duration_seconds: Sustained wall-duration floor.
        :param minimum_transition_count: Measured seven-hop transition floor.
        """

        if type(machine_count) is not int or machine_count <= 0:
            raise ValueError("machine_count must be a positive integer")
        if (
            type(minimum_duration_seconds) is not float
            or minimum_duration_seconds <= 0.0
        ):
            raise ValueError("minimum_duration_seconds must be a positive float")
        if type(minimum_transition_count) is not int or minimum_transition_count <= 0:
            raise ValueError("minimum_transition_count must be a positive integer")
        self._native.start_qualification(
            machine_count,
            minimum_duration_seconds,
            minimum_transition_count,
        )

    def qualification_join(self, timeout_seconds: float) -> bool:
        """Wait without the GIL for native qualification completion.

        :param timeout_seconds: Positive wall-clock wait bound.
        :returns: Whether the complete population retired within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(self._native.qualification_join(timeout_seconds))

    def qualification_summary(self) -> Mapping[str, object]:
        """Return exact native statistics and bounded audit samples.

        The bridge computes nearest-rank order statistics over the complete
        measured population. Only first/last per-machine audit traces cross the
        Python boundary, so evidence size is independent of run throughput.

        :returns: Immutable native evidence summary mapping.
        """

        summary = self._native.qualification_summary()
        if not isinstance(summary, Mapping):
            raise TypeError("native qualification summary must be a mapping")
        return summary

    def stop_admission(self) -> None:
        """Close lifecycle and event admission before drain."""

        self._native.stop_admission()

    def retire_python_producer(self, producer_id: int) -> None:
        """Order one Python producer's retirement after its context joins.

        :param producer_id: Exact registered Python producer namespace.
        :raises OSError: If retirement cannot enter the ordered input domain.
        """

        if type(producer_id) is not int or producer_id < 0:
            raise ValueError("producer_id must be a non-negative integer")
        status = int(self._native.retire_python_producer(producer_id))
        if status != 0:
            raise OSError(status, os.strerror(status))

    def wait_for_producer_retirement(
        self, producer_id: int, timeout_seconds: float
    ) -> bool:
        """Wait for one ordered producer retirement fence to commit.

        :param producer_id: Exact registered producer identity.
        :param timeout_seconds: Positive wall-clock bound.
        :returns: Whether retirement committed within the bound.
        """

        if type(producer_id) is not int or producer_id < 0:
            raise ValueError("producer_id must be a non-negative integer")
        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(
            self._native.wait_for_producer_retirement(
                producer_id,
                timeout_seconds,
            )
        )

    def join_producers(self) -> bool:
        """Verify that every native and Python producer retired.

        :returns: Whether event admission closed after exact retirement.
        """

        return bool(self._native.join_producers())

    def wait_for_output_quiescence(self, timeout_seconds: float) -> bool:
        """Wait for the sole output consumer to finish native routing.

        :param timeout_seconds: Positive hash-bound shutdown timeout.
        :returns: Whether no queued, swapped, or unacknowledged output remains.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(self._native.wait_for_output_quiescence(timeout_seconds))

    def begin_abort(self) -> None:
        """Quarantine unresolved lifecycles while output routing stays alive."""

        self._native.begin_abort()

    def close_aborted(self) -> None:
        """Close a fatal owner after final native authority was routed."""

        if self._closed:
            return
        self._native.close_aborted()
        self._closed = True

    def close(self) -> None:
        """Close a fully drained owner exactly once."""

        if self._closed:
            return
        self._native.close()
        self._closed = True

    def abort_and_close(self) -> None:
        """Fail closed and release an incomplete owner exactly once."""

        if self._closed:
            return
        self._native.abort_and_close()
        self._closed = True


def parse_native_owner_outputs(
    values: Sequence[Mapping[str, object]],
) -> tuple[NativeTerminalOwnerOutput, ...]:
    """Parse raw bridge outputs for bounded integration tests.

    :param values: Raw immutable mappings returned by the native drain.
    :returns: Validated typed commit outputs.
    """

    return tuple(NativeTerminalOwnerOutput.from_native(value) for value in values)
