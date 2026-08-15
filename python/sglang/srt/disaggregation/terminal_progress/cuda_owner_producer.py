import dataclasses
import enum
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalNativeProducerBinding,
    NativeTerminalRuntime,
)
from torch.utils.cpp_extension import CUDA_HOME, load


@dataclasses.dataclass(frozen=True, slots=True)
class CudaTerminalProducerInventory:
    """Complete CUDA callback-to-owner producer inventory.

    :ivar armed_count: Bindings armed but not yet attached to a stream.
    :ivar submitted_count: Bindings submitted to CUDA but not yet delivered.
    :ivar pending_authorization_count: Completed source callbacks retained until
        authenticated decoder allocation authorizes native delivery.
    :ivar active_callback_count: CUDA host callbacks still outstanding.
    :ivar active_registration_count: Callback registrations still returning.
    :ivar total_submissions: Successfully registered callbacks.
    :ivar total_delivered: Events admitted directly into the native owner.
    :ivar owner_submit_failure_count: Native owner submission failures.
    :ivar admission_open: Whether new bindings may be armed.
    :ivar retirement_requested: Whether ordered producer retirement entered.
    :ivar joined: Whether ordered retirement committed.
    :ivar closed: Whether exact-zero closure completed.
    :ivar fatal_code: Sticky first producer failure.
    :ivar fatal_status: Native errno or CUDA status for the first failure.
    :ivar fatal_binding: Exact binding associated with the first failure.
    """

    armed_count: int
    submitted_count: int
    pending_authorization_count: int
    active_callback_count: int
    active_registration_count: int
    total_submissions: int
    total_delivered: int
    owner_submit_failure_count: int
    admission_open: bool
    retirement_requested: bool
    joined: bool
    closed: bool
    fatal_code: str
    fatal_status: int
    fatal_binding: bytes | None

    @classmethod
    def from_native(cls, value: dict[str, object]) -> "CudaTerminalProducerInventory":
        """Parse one native inventory snapshot.

        :param value: Native inventory mapping.
        :returns: Validated typed inventory.
        """

        fatal_binding_value = value["fatal_binding"]
        fatal_binding: bytes | None = None
        if fatal_binding_value is not None:
            if type(fatal_binding_value) is not bytes:
                raise TypeError("fatal_binding must be bytes or None")
            if len(fatal_binding_value) != 32:
                raise ValueError("fatal_binding must contain 32 bytes")
            fatal_binding = fatal_binding_value
        return cls(
            armed_count=int(value["armed_count"]),
            submitted_count=int(value["submitted_count"]),
            pending_authorization_count=int(value["pending_authorization_count"]),
            active_callback_count=int(value["active_callback_count"]),
            active_registration_count=int(value["active_registration_count"]),
            total_submissions=int(value["total_submissions"]),
            total_delivered=int(value["total_delivered"]),
            owner_submit_failure_count=int(value["owner_submit_failure_count"]),
            admission_open=bool(value["admission_open"]),
            retirement_requested=bool(value["retirement_requested"]),
            joined=bool(value["joined"]),
            closed=bool(value["closed"]),
            fatal_code=str(value["fatal_code"]),
            fatal_status=int(value["fatal_status"]),
            fatal_binding=fatal_binding,
        )

    @property
    def retained_count(self) -> int:
        """Return every binding still owned by the producer.

        :returns: Armed and callback-submitted binding count.
        """

        return self.armed_count + self.submitted_count


class CudaTerminalEventKind(enum.IntEnum):
    """Native lifecycle transition emitted by one CUDA callback owner."""

    SOURCE_PRODUCER_COMPLETED = 11
    DECODE_SCATTER_TERMINAL = 44


@runtime_checkable
class TerminalCudaCompletionProducer(Protocol):
    """Direct native CUDA-callback producer for one lifecycle boundary."""

    def arm(self, binding_digest: bytes) -> None:
        """Arm one exact lifecycle before callback registration.

        :param binding_digest: Exact local lifecycle identity.
        """

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Attach terminal delivery after prior work on a CUDA stream.

        :param stream_handle: Raw CUDA stream carrying the completed work.
        :param binding_digest: Exact lifecycle armed for this callback.
        """

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Authorize a gated callback to enter the native owner.

        :param binding_digest: Exact armed source lifecycle.
        :returns: Whether a completed callback was released immediately.
        """


class _NativeCudaTerminalProducer(Protocol):
    """Typed boundary implemented by the CUDA producer extension."""

    def arm(self, binding_digest: bytes) -> None:
        """Arm one exact owner binding.

        :param binding_digest: Exact request lifecycle digest.
        """

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Attach terminal delivery after prior work on a CUDA stream.

        :param stream_handle: Raw ``cudaStream_t`` value.
        :param binding_digest: Exact armed request lifecycle digest.
        """

    def stop_admission(self) -> None:
        """Permanently stop callback admission."""

    def join(self, timeout_seconds: float) -> bool:
        """Join callbacks and ordered producer retirement.

        :param timeout_seconds: Positive native wait bound.
        :returns: Whether ordered producer retirement committed.
        """

    def close(self) -> None:
        """Close a joined producer with exact-zero retained inventory."""

    def inventory(self) -> dict[str, object]:
        """Return complete producer inventory.

        :returns: Native inventory mapping.
        """

    def _complete_synchronously_for_test(self, binding_digest: bytes) -> None:
        """Deliver one armed binding without CUDA.

        :param binding_digest: Exact armed lifecycle digest.
        """

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Authorize one gated source callback for native delivery.

        :param binding_digest: Exact armed lifecycle digest.
        :returns: Whether a completed callback was released immediately.
        """

    def _begin_held_callback_for_test(self, binding_digest: bytes) -> None:
        """Begin one callback retained until an explicit test release.

        :param binding_digest: Exact armed lifecycle digest.
        """

    def _complete_held_callback_for_test(self, binding_digest: bytes) -> None:
        """Complete one explicitly held callback.

        :param binding_digest: Exact submitted lifecycle digest.
        """

    def _complete_concurrently_for_test(self, binding_digests: list[bytes]) -> None:
        """Deliver multiple bindings from concurrent native threads.

        :param binding_digests: Exact armed lifecycle digests.
        """


def _native_source_path() -> Path:
    """Return the packaged CUDA producer source.

    :returns: Absolute C++ source path.
    """

    return Path(__file__).with_name("cuda_owner_producer.cpp")


def _native_header_path() -> Path:
    """Return the shared terminal-owner producer ABI header.

    :returns: Absolute header path.
    """

    return Path(__file__).with_name("native_producer_api.h")


@lru_cache(maxsize=2)
def _load_native_cuda_terminal_producer(*, testing: bool) -> ModuleType:
    """Compile and load the direct CUDA-to-owner producer.

    :param testing: Whether deterministic native test controls are required.
    :returns: Loaded extension module.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("CUDA terminal producer requires Linux")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA terminal producer requires a CUDA toolkit")
    cuda_home = Path(CUDA_HOME).resolve()
    source_path = _native_source_path()
    header_path = _native_header_path()
    source_hasher = hashlib.sha256()
    for required_path in (source_path, header_path):
        if not required_path.is_file():
            raise RuntimeError(
                f"CUDA terminal producer source is absent: {required_path}"
            )
        source_hasher.update(required_path.name.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(required_path.read_bytes())
    source_digest = source_hasher.hexdigest()[:12]
    variant = "test" if testing else "runtime"
    module_name = f"sglang_cuda_terminal_producer_{source_digest}_{variant}"
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
        extra_cflags=[*extra_cflags, f"-I{cuda_home / 'include'}"],
        extra_include_paths=[str(source_path.parent)],
        extra_ldflags=[f"-L{cuda_home / 'lib64'}", "-lcudart", "-pthread"],
        build_directory=build_directory,
        with_cuda=False,
        verbose=False,
    )


def cuda_terminal_producer_abi(*, testing: bool = False) -> dict[str, object]:
    """Return the CUDA DSO's independently compiled owner ABI layout.

    :param testing: Whether to inspect the native test variant.
    :returns: Version, flags, structure sizes, and event-field offsets.
    """

    module = _load_native_cuda_terminal_producer(testing=testing)
    value = module.compiled_abi()
    if type(value) is not dict:
        raise TypeError("CUDA terminal producer ABI must be a dictionary")
    offsets = value["event_offsets"]
    if type(offsets) is not dict:
        raise TypeError("CUDA terminal producer ABI offsets must be a dictionary")
    return {
        "abi_version": int(value["abi_version"]),
        "api_struct_size": int(value["api_struct_size"]),
        "event_struct_size": int(value["event_struct_size"]),
        "required_flags": int(value["required_flags"]),
        "event_offsets": {str(key): int(item) for key, item in offsets.items()},
    }


class CudaTerminalProducer:
    """Process-lifetime direct CUDA callback producer for one native owner."""

    _binding: NativeTerminalNativeProducerBinding
    _native: _NativeCudaTerminalProducer
    _testing: bool

    def __init__(
        self,
        binding: NativeTerminalNativeProducerBinding,
        event_kind: CudaTerminalEventKind,
        *,
        testing: bool = False,
    ) -> None:
        """Bind one registered producer namespace to CUDA callbacks.

        :param binding: Runtime-owned API and producer-context capsules.
        :param event_kind: Exact lifecycle transition emitted on completion.
        :param testing: Whether native test controls are required.
        """

        if type(binding) is not NativeTerminalNativeProducerBinding:
            raise TypeError("binding must be NativeTerminalNativeProducerBinding")
        if type(event_kind) is not CudaTerminalEventKind:
            raise TypeError("event_kind must be CudaTerminalEventKind")
        if type(testing) is not bool:
            raise TypeError("testing must be bool")
        module = _load_native_cuda_terminal_producer(testing=testing)
        self._native = cast(
            _NativeCudaTerminalProducer,
            module.CudaTerminalProducer(
                binding.producer_api,
                binding.producer_context,
                int(event_kind),
            ),
        )
        self._binding = binding
        self._testing = testing

    @classmethod
    def from_runtime(
        cls,
        runtime: NativeTerminalRuntime,
        producer_name: str,
        event_kind: CudaTerminalEventKind,
        *,
        testing: bool = False,
    ) -> "CudaTerminalProducer":
        """Construct from one runtime-registered native producer.

        :param runtime: Dormant or running immutable owner runtime.
        :param producer_name: Stable pre-registered native producer name.
        :param event_kind: Exact lifecycle transition emitted on completion.
        :param testing: Whether native test controls are required.
        :returns: Bound direct CUDA terminal producer.
        """

        if type(runtime) is not NativeTerminalRuntime:
            raise TypeError("runtime must be NativeTerminalRuntime")
        return cls(
            runtime.native_producer_binding(producer_name),
            event_kind,
            testing=testing,
        )

    @property
    def producer_id(self) -> int:
        """Return the exact native owner producer namespace.

        :returns: Runtime-registered producer ID.
        """

        return self._binding.producer_id

    def arm(self, binding_digest: bytes) -> None:
        """Arm one exact lifecycle before scatter submission.

        :param binding_digest: Exact 32-byte native lifecycle digest.
        """

        self._require_binding(binding_digest)
        self._native.arm(binding_digest)

    def submit(self, stream_handle: int, binding_digest: bytes) -> None:
        """Attach direct owner delivery after scatter on one stream.

        :param stream_handle: Raw ``cudaStream_t`` value.
        :param binding_digest: Exact armed native lifecycle digest.
        """

        if type(stream_handle) is not int or stream_handle < 0:
            raise ValueError("stream_handle must be a non-negative integer")
        self._require_binding(binding_digest)
        self._native.submit(stream_handle, binding_digest)

    def authorize_delivery(self, binding_digest: bytes) -> bool:
        """Authorize one source completion after authenticated allocation.

        Decode scatter producers are ungated and reject this operation. Source
        producers retain the original callback timestamp when authorization
        arrives after CUDA completion.

        :param binding_digest: Exact armed source lifecycle digest.
        :returns: Whether a completed callback was released immediately.
        """

        self._require_binding(binding_digest)
        return bool(self._native.authorize_delivery(binding_digest))

    def stop_admission(self) -> None:
        """Permanently stop new callback bindings."""

        self._native.stop_admission()

    def join(self, timeout_seconds: float) -> bool:
        """Join callbacks and commit ordered producer retirement.

        :param timeout_seconds: Positive native wait bound.
        :returns: Whether retirement committed within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(self._native.join(timeout_seconds))

    def close(self) -> None:
        """Close after exact-zero joined retirement."""

        self._native.close()

    def inventory(self) -> CudaTerminalProducerInventory:
        """Return complete callback, binding, and retirement inventory.

        :returns: Typed producer inventory.
        """

        return CudaTerminalProducerInventory.from_native(self._native.inventory())

    def complete_synchronously_for_testing(self, binding_digest: bytes) -> None:
        """Deliver one armed binding without CUDA.

        :param binding_digest: Exact armed lifecycle digest.
        :raises RuntimeError: If this is not a test build.
        """

        self._require_testing()
        self._require_binding(binding_digest)
        self._native._complete_synchronously_for_test(binding_digest)

    def begin_held_callback_for_testing(self, binding_digest: bytes) -> None:
        """Retain one callback until explicitly completed by its test.

        :param binding_digest: Exact armed lifecycle digest.
        :raises RuntimeError: If this is not a test build.
        """

        self._require_testing()
        self._require_binding(binding_digest)
        self._native._begin_held_callback_for_test(binding_digest)

    def complete_held_callback_for_testing(self, binding_digest: bytes) -> None:
        """Release one callback previously held by its test.

        :param binding_digest: Exact submitted lifecycle digest.
        :raises RuntimeError: If this is not a test build.
        """

        self._require_testing()
        self._require_binding(binding_digest)
        self._native._complete_held_callback_for_test(binding_digest)

    def complete_concurrently_for_testing(
        self, binding_digests: tuple[bytes, ...]
    ) -> None:
        """Deliver exact bindings from concurrent native threads.

        :param binding_digests: Exact armed lifecycle digests.
        :raises RuntimeError: If this is not a test build.
        """

        self._require_testing()
        if type(binding_digests) is not tuple:
            raise TypeError("binding_digests must be a tuple")
        for binding_digest in binding_digests:
            self._require_binding(binding_digest)
        self._native._complete_concurrently_for_test(list(binding_digests))

    def _require_testing(self) -> None:
        """Require deterministic native test controls."""

        if not self._testing:
            raise RuntimeError("CUDA terminal producer test control is unavailable")

    @staticmethod
    def _require_binding(binding_digest: bytes) -> None:
        """Validate one exact native lifecycle digest.

        :param binding_digest: Candidate lifecycle digest.
        """

        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
