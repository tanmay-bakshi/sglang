import dataclasses
import enum
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from torch.utils.cpp_extension import load

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
)


class CudaCompletionFatalCode(enum.StrEnum):
    """Sticky process-fatal CUDA completion bridge outcomes."""

    NONE = "none"
    QUEUE_OVERFLOW = "queue_overflow"
    EVENTFD_WRITE_FAILURE = "eventfd_write_failure"
    EVENTFD_READ_FAILURE = "eventfd_read_failure"
    DUPLICATE_IDENTITY = "duplicate_identity"
    EXACT_GENERATION_MISMATCH = "exact_generation_mismatch"
    SUBMISSION_WITHOUT_ARM = "submission_without_arm"
    INVALID_IDENTITY_STATE = "invalid_identity_state"
    CUDA_CALLBACK_REGISTRATION_FAILURE = "cuda_callback_registration_failure"
    SUBMISSION_AFTER_STOP = "submission_after_stop"
    CONCURRENT_DRAIN = "concurrent_drain"
    CLOSE_WITH_ACTIVE_CALLBACKS = "close_with_active_callbacks"
    CLOSE_WITH_RETAINED_INVENTORY = "close_with_retained_inventory"
    CLOSE_BEFORE_PRODUCER_JOIN = "close_before_producer_join"


@dataclasses.dataclass(frozen=True, slots=True)
class CudaCompletionIdentity:
    """Opaque owner cookie bound to one exact request generation.

    :ivar cookie: Process-local owner routing cookie.
    :ivar generation: Exact request generation protected from stale callbacks.
    """

    cookie: int
    generation: bytes

    def __post_init__(self) -> None:
        """Validate one callback identity."""

        if type(self.cookie) is not int or self.cookie < 0 or self.cookie >= 2**64:
            raise ValueError("cookie must be an unsigned 64-bit integer")
        if type(self.generation) is not bytes:
            raise TypeError("generation must be bytes")
        if len(self.generation) != PACKED_REQUEST_GENERATION_BYTES:
            raise ValueError(
                "generation must contain "
                f"{PACKED_REQUEST_GENERATION_BYTES} bytes"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CudaCompletionInventory:
    """Complete native callback and token lifecycle inventory.

    :ivar capacity: Physical token-queue capacity.
    :ivar armed_count: Identities armed but not submitted.
    :ivar submitted_identity_count: Submitted identities awaiting drain.
    :ivar active_callback_count: CUDA callbacks not yet invoked.
    :ivar active_registration_count: Host-callback registrations still returning.
    :ivar queued_count: Completed tokens awaiting take-once drain.
    :ivar live_count: All armed or submitted exact identities.
    :ivar total_submissions: Successfully registered callback submissions.
    :ivar total_enqueued: Callback tokens admitted to the queue.
    :ivar total_drained: Tokens whose exact identities were consumed.
    :ivar overflow_count: Callback tokens rejected by the physical bound.
    :ivar eventfd_failure_count: Failed eventfd reads or writes.
    :ivar successful_wake_count: Successful callback or fatal wake writes.
    :ivar consumed_wake_count: Eventfd counter units consumed by drains.
    :ivar rejected_token_count: Drained tokens denied exact authority.
    :ivar producers_joined: Whether submission is stopped and callbacks joined.
    :ivar admission_open: Whether new identities may be armed.
    :ivar closed: Whether the bridge closed successfully.
    :ivar eventfd_open: Whether the pollable descriptor remains open.
    :ivar fatal_code: First sticky process-fatal outcome.
    :ivar fatal_system_error: Captured errno or CUDA status value.
    :ivar fatal_identity: Exact identity which triggered the fatal outcome.
    """

    capacity: int
    armed_count: int
    submitted_identity_count: int
    active_callback_count: int
    active_registration_count: int
    queued_count: int
    live_count: int
    total_submissions: int
    total_enqueued: int
    total_drained: int
    overflow_count: int
    eventfd_failure_count: int
    successful_wake_count: int
    consumed_wake_count: int
    rejected_token_count: int
    producers_joined: bool
    admission_open: bool
    closed: bool
    eventfd_open: bool
    fatal_code: CudaCompletionFatalCode
    fatal_system_error: int
    fatal_identity: CudaCompletionIdentity | None

    @classmethod
    def from_native(cls, value: dict[str, object]) -> "CudaCompletionInventory":
        """Construct a typed inventory from the native snapshot.

        :param value: Native inventory mapping.
        :returns: Fully typed lifecycle inventory.
        """

        fatal_cookie = value["fatal_cookie"]
        fatal_generation = value["fatal_generation"]
        fatal_identity: CudaCompletionIdentity | None = None
        if fatal_cookie is not None or fatal_generation is not None:
            if type(fatal_cookie) is not int or type(fatal_generation) is not bytes:
                raise RuntimeError("native fatal identity is incomplete")
            fatal_identity = CudaCompletionIdentity(
                cookie=fatal_cookie,
                generation=fatal_generation,
            )
        return cls(
            capacity=int(value["capacity"]),
            armed_count=int(value["armed_count"]),
            submitted_identity_count=int(value["submitted_identity_count"]),
            active_callback_count=int(value["active_callback_count"]),
            active_registration_count=int(value["active_registration_count"]),
            queued_count=int(value["queued_count"]),
            live_count=int(value["live_count"]),
            total_submissions=int(value["total_submissions"]),
            total_enqueued=int(value["total_enqueued"]),
            total_drained=int(value["total_drained"]),
            overflow_count=int(value["overflow_count"]),
            eventfd_failure_count=int(value["eventfd_failure_count"]),
            successful_wake_count=int(value["successful_wake_count"]),
            consumed_wake_count=int(value["consumed_wake_count"]),
            rejected_token_count=int(value["rejected_token_count"]),
            producers_joined=bool(value["producers_joined"]),
            admission_open=bool(value["admission_open"]),
            closed=bool(value["closed"]),
            eventfd_open=bool(value["eventfd_open"]),
            fatal_code=CudaCompletionFatalCode(str(value["fatal_code"])),
            fatal_system_error=int(value["fatal_system_error"]),
            fatal_identity=fatal_identity,
        )

    @property
    def retained_count(self) -> int:
        """Return all lifecycle identities retained by the bridge.

        :returns: Exact retained-identity count.
        """

        return self.live_count


@dataclasses.dataclass(frozen=True, slots=True)
class CudaCompletionDrain:
    """One take-once native queue drain.

    :ivar wake_count: Coalesced eventfd counter consumed before queue drain.
    :ivar identities: Exact completed identities retired by this drain.
    :ivar inventory: Native lifecycle state after every returned token retired.
    """

    wake_count: int
    identities: tuple[CudaCompletionIdentity, ...]
    inventory: CudaCompletionInventory


def _native_source_path() -> Path:
    """Return the packaged native bridge source.

    :returns: Absolute C++ source path.
    """

    return Path(__file__).with_name("cuda_completion_bridge.cpp")


class _NativeCudaCompletionBridge(Protocol):
    """Typed boundary implemented by the pybind11 bridge object."""

    def arm(self, cookie: int, generation: bytes) -> None:
        """Arm one native identity.

        :param cookie: Process-local owner cookie.
        :param generation: Exact request generation.
        """

    def submit(self, stream_handle: int, cookie: int, generation: bytes) -> None:
        """Attach one native callback.

        :param stream_handle: Raw CUDA stream handle.
        :param cookie: Process-local owner cookie.
        :param generation: Exact request generation.
        """

    def fileno(self) -> int:
        """Return the native eventfd.

        :returns: Pollable descriptor.
        """

    def drain(self) -> dict[str, object]:
        """Drain native tokens.

        :returns: Native drain mapping.
        """

    def stop_submissions(self) -> None:
        """Stop native submission admission."""

    def join_producers(self) -> bool:
        """Try to join all native callback producers.

        :returns: Whether every producer returned.
        """

    def close(self) -> None:
        """Close a fully drained native bridge."""

    def inventory(self) -> dict[str, object]:
        """Return native inventory.

        :returns: Complete native lifecycle mapping.
        """

    def _complete_synchronously_for_test(
        self, cookie: int, generation: bytes
    ) -> None:
        """Complete one armed test identity.

        :param cookie: Process-local owner cookie.
        :param generation: Exact request generation.
        """

    def _break_eventfd_for_test(self) -> None:
        """Break the native eventfd in a test build."""


@lru_cache(maxsize=2)
def _load_native_cuda_completion_bridge(testing: bool = False) -> ModuleType:
    """Compile and load the CUDA-runtime bridge extension once per process.

    :param testing: Whether to expose deterministic native-only test hooks.
    :returns: Loaded pybind11 extension module.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("CUDA completion bridge requires Linux eventfd")
    source_path = _native_source_path()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    variant = "test" if testing else "runtime"
    module_name = f"sglang_cuda_completion_bridge_{source_digest}_{variant}"
    extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
    if testing:
        extra_cflags.append("-DSGLANG_CUDA_COMPLETION_BRIDGE_TESTING")
    build_directory_value = os.environ.get("SGLANG_CUDA_BRIDGE_BUILD_DIR")
    build_directory: str | None = None
    if build_directory_value is not None:
        build_path = Path(build_directory_value).resolve() / module_name
        build_path.mkdir(parents=True, exist_ok=True)
        build_directory = str(build_path)
    return load(
        name=module_name,
        sources=[str(source_path)],
        extra_cflags=extra_cflags,
        build_directory=build_directory,
        with_cuda=True,
        verbose=False,
    )


class CudaCompletionBridge:
    """Process-lifetime bridge from CUDA host callbacks to one fd consumer."""

    _native: _NativeCudaCompletionBridge
    _testing: bool

    def __init__(self, capacity: int, *, testing: bool = False) -> None:
        """Create one bounded process-local callback bridge.

        :param capacity: Maximum completed tokens awaiting a drain.
        :param testing: Whether deterministic native test hooks are required.
        """

        if type(capacity) is not int or capacity < 2:
            raise ValueError("capacity must be an integer of at least two")
        native_module = _load_native_cuda_completion_bridge(testing=testing)
        self._native = cast(
            _NativeCudaCompletionBridge,
            native_module.CudaCompletionBridge(capacity),
        )
        self._testing = testing

    def arm(self, identity: CudaCompletionIdentity) -> None:
        """Reserve one exact callback identity before scatter submission.

        :param identity: Owner cookie and generation which the callback must carry.
        """

        if type(identity) is not CudaCompletionIdentity:
            raise TypeError("identity must be CudaCompletionIdentity")
        self._native.arm(identity.cookie, identity.generation)

    def submit(self, stream_handle: int, identity: CudaCompletionIdentity) -> None:
        """Attach the armed identity after all prior work on one CUDA stream.

        :param stream_handle: Raw ``cudaStream_t`` value owned by the caller.
        :param identity: The exact identity already armed on this bridge.
        """

        if type(stream_handle) is not int or stream_handle < 0:
            raise ValueError("stream_handle must be a non-negative integer")
        if type(identity) is not CudaCompletionIdentity:
            raise TypeError("identity must be CudaCompletionIdentity")
        self._native.submit(stream_handle, identity.cookie, identity.generation)

    def fileno(self) -> int:
        """Return the pollable nonblocking eventfd.

        :returns: Open eventfd, or ``-1`` after a successful close.
        """

        return int(self._native.fileno())

    def drain(self) -> CudaCompletionDrain:
        """Take every currently completed token exactly once.

        :returns: Completed identities and the post-drain lifecycle inventory.
        """

        value: dict[str, object] = self._native.drain()
        raw_tokens = value["tokens"]
        if type(raw_tokens) is not list:
            raise RuntimeError("native completion tokens must be a list")
        identities: list[CudaCompletionIdentity] = []
        for raw_token in raw_tokens:
            if type(raw_token) is not tuple or len(raw_token) != 2:
                raise RuntimeError("native completion token has invalid shape")
            cookie, generation = raw_token
            if type(cookie) is not int or type(generation) is not bytes:
                raise RuntimeError("native completion token has invalid fields")
            identities.append(
                CudaCompletionIdentity(cookie=cookie, generation=generation)
            )
        inventory_value = value["inventory"]
        if type(inventory_value) is not dict:
            raise RuntimeError("native completion inventory must be a dict")
        return CudaCompletionDrain(
            wake_count=int(value["wake_count"]),
            identities=tuple(identities),
            inventory=CudaCompletionInventory.from_native(inventory_value),
        )

    def stop_submissions(self) -> None:
        """Permanently stop new callback identities from entering the bridge."""

        self._native.stop_submissions()

    def join_producers(self) -> bool:
        """Stop submission and attest whether every callback has returned.

        :returns: Whether the callback producer count reached zero.
        """

        return bool(self._native.join_producers())

    def close(self) -> None:
        """Close only after producers joined and all authority was drained.

        :raises RuntimeError: If callbacks or lifecycle inventory remain.
        """

        self._native.close()

    def inventory(self) -> CudaCompletionInventory:
        """Return a complete native lifecycle snapshot.

        :returns: Callback, queue, descriptor, and fatal-state inventory.
        """

        value: dict[str, object] = self._native.inventory()
        return CudaCompletionInventory.from_native(value)

    def complete_synchronously_for_test(
        self, identity: CudaCompletionIdentity
    ) -> None:
        """Drive the callback publication body without CUDA in a test build.

        :param identity: Exact identity already armed on this bridge.
        """

        if type(identity) is not CudaCompletionIdentity:
            raise TypeError("identity must be CudaCompletionIdentity")
        if not self._testing:
            raise RuntimeError("deterministic completion requires a test build")
        self._native._complete_synchronously_for_test(
            identity.cookie,
            identity.generation,
        )

    def break_eventfd_for_test(self) -> None:
        """Close the native eventfd before a deterministic failure test."""

        if not self._testing:
            raise RuntimeError("eventfd fault injection requires a test build")
        self._native._break_eventfd_for_test()
