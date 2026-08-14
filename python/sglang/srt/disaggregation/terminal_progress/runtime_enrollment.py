import dataclasses
import enum
import math
import threading
import time
import traceback
from collections.abc import Callable

from sglang.srt.disaggregation.terminal_progress.cuda_owner_producer import (
    CudaTerminalEventKind,
    CudaTerminalProducer,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalProducerDelivery,
    NativeTerminalRuntime,
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)

TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME = "terminal-native-cuda-source"
TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME = "terminal-native-cuda-scatter"


class TerminalRankRuntimeEnrollmentError(RuntimeError):
    """Rank-lifetime runtime construction or retirement invariant violation."""


class TerminalRankNativeProducerDisposition(enum.StrEnum):
    """Unified lifetime of one rank's native producer population."""

    OPEN = "open"
    RETIRING = "retiring"
    RETIRED = "retired"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRankRuntimeConfig:
    """Frozen capacities and retirement bound for one rank runtime.

    :ivar input_capacity: Native owner input capacity.
    :ivar output_capacity: Native owner action capacity.
    :ivar maximum_live_lifecycles: Maximum admitted request generations.
    :ivar scheduler_capacity: Scheduler action inbox capacity.
    :ivar coordinator_capacity: Request coordinator inbox capacity.
    :ivar lifecycle_capacity: Teardown and health inbox capacity.
    :ivar source_work_capacity: Source continuation inbox capacity.
    :ivar decode_work_capacity: Decode continuation inbox capacity.
    :ivar publisher_capacity: Gateway publication inbox capacity.
    :ivar observation_capacity: Non-gating observation inbox capacity.
    :ivar native_producer_retirement_timeout_seconds: One absolute bound shared
        by every native producer retirement fence.
    """

    input_capacity: int
    output_capacity: int
    maximum_live_lifecycles: int
    scheduler_capacity: int
    coordinator_capacity: int
    lifecycle_capacity: int
    source_work_capacity: int
    decode_work_capacity: int
    publisher_capacity: int
    observation_capacity: int
    native_producer_retirement_timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate every physical bound before native allocation."""

        capacities = (
            self.input_capacity,
            self.output_capacity,
            self.maximum_live_lifecycles,
            self.scheduler_capacity,
            self.coordinator_capacity,
            self.lifecycle_capacity,
            self.source_work_capacity,
            self.decode_work_capacity,
            self.publisher_capacity,
            self.observation_capacity,
        )
        if any(type(value) is not int or value <= 0 for value in capacities):
            raise ValueError("runtime capacities must be positive integers")
        timeout_seconds = self.native_producer_retirement_timeout_seconds
        if (
            type(timeout_seconds) is not float
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise ValueError(
                "native_producer_retirement_timeout_seconds must be a "
                "positive finite float"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalRankNativeProducerPlan:
    """Role-specific native producer namespaces appended after Python.

    :ivar role: Local source or decode owner role.
    :ivar specs: Complete one-producer native suffix.
    :ivar cuda_source_producer_id: Source CUDA completion namespace, if any.
    :ivar cuda_scatter_producer_id: Decode CUDA completion namespace, if any.
    :ivar next_producer_id: First unallocated producer identifier.
    """

    role: TerminalOwnerRole
    specs: tuple[NativeTerminalRuntimeProducerSpec, ...]
    cuda_source_producer_id: int | None
    cuda_scatter_producer_id: int | None
    next_producer_id: int

    def __post_init__(self) -> None:
        """Validate role shape, delivery ownership, and numbering."""

        if type(self.role) is not TerminalOwnerRole:
            raise TypeError("role must be TerminalOwnerRole")
        expected_count = 1
        if type(self.specs) is not tuple or len(self.specs) != expected_count:
            raise ValueError("native producer population disagrees with its role")
        if any(
            type(spec) is not NativeTerminalRuntimeProducerSpec for spec in self.specs
        ):
            raise TypeError("specs contain an invalid producer specification")
        first_producer_id = self.specs[0].registration.producer_id
        producer_ids = tuple(spec.registration.producer_id for spec in self.specs)
        if producer_ids != tuple(
            range(first_producer_id, first_producer_id + expected_count)
        ):
            raise ValueError("native producer IDs must be contiguous")
        if self.next_producer_id != first_producer_id + expected_count:
            raise ValueError("next_producer_id must follow every native producer")
        if any(
            spec.delivery is not NativeTerminalProducerDelivery.NATIVE
            for spec in self.specs
        ):
            raise ValueError("native producer plan contains Python delivery")
        if any(
            spec.registration.producer_class is not NativeTerminalProducerClass.LOCAL
            for spec in self.specs
        ):
            raise ValueError("direct native producers require local authority")
        expected_native_role = _native_role(self.role)
        if any(
            spec.registration.allowed_role is not expected_native_role
            for spec in self.specs
        ):
            raise ValueError("native producer role disagrees with the local owner")
        expected_cuda_source_id = (
            first_producer_id if self.role is TerminalOwnerRole.SOURCE else None
        )
        expected_cuda_scatter_id = (
            first_producer_id if self.role is TerminalOwnerRole.DECODE else None
        )
        if self.cuda_source_producer_id != expected_cuda_source_id:
            raise ValueError("source CUDA producer disagrees with the local role")
        if self.cuda_scatter_producer_id != expected_cuda_scatter_id:
            raise ValueError("scatter CUDA producer disagrees with the local role")


def _native_role(role: TerminalOwnerRole) -> NativeTerminalOwnerRole:
    """Convert the stable Python role to its native owner code.

    :param role: Source or decode owner role.
    :returns: Stable native role.
    """

    if role is TerminalOwnerRole.SOURCE:
        return NativeTerminalOwnerRole.SOURCE
    return NativeTerminalOwnerRole.DECODE


def _native_producer_spec(
    producer_id: int,
    name: str,
    role: TerminalOwnerRole,
) -> NativeTerminalRuntimeProducerSpec:
    """Build one exclusively native local producer specification.

    :param producer_id: Stable process-local producer identifier.
    :param name: Evidence-facing producer name.
    :param role: Local owner role.
    :returns: Complete native producer specification.
    """

    return NativeTerminalRuntimeProducerSpec(
        registration=NativeTerminalProducerRegistration(
            producer_id=producer_id,
            name=name,
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=_native_role(role),
            authenticated_issuer=None,
        ),
        delivery=NativeTerminalProducerDelivery.NATIVE,
    )


def _require_consistent_binding(binding: TerminalStartupRankBinding) -> None:
    """Prove the Python producer prefix belongs to the exact sealed rank.

    :param binding: Candidate startup authority.
    """

    if type(binding) is not TerminalStartupRankBinding:
        raise TypeError("binding must be TerminalStartupRankBinding")
    local = binding.advertisement
    matrix_local = binding.matrix.rank(*local.key)
    if matrix_local != local:
        raise ValueError("startup matrix local rank differs from its advertisement")
    first_producer_id = binding.python_producers.specs[0].registration.producer_id
    expected_python_plan = build_terminal_startup_python_producer_plan(
        binding.matrix,
        local_service_id=local.service_id,
        local_tensor_parallel_rank=local.tensor_parallel_rank,
        first_producer_id=first_producer_id,
    )
    if binding.python_producers != expected_python_plan:
        raise ValueError("Python producer plan belongs to another startup authority")


def build_terminal_rank_native_producer_plan(
    binding: TerminalStartupRankBinding,
) -> TerminalRankNativeProducerPlan:
    """Append the exact role-specific native producer suffix.

    :param binding: Complete immutable startup rank authority.
    :returns: Contiguous native producer plan.
    """

    _require_consistent_binding(binding)
    local = binding.advertisement
    cuda_producer_id = binding.python_producers.next_producer_id
    cuda_source_producer_id: int | None = None
    cuda_scatter_producer_id: int | None = None
    if local.role is TerminalOwnerRole.SOURCE:
        cuda_source_producer_id = cuda_producer_id
        cuda_producer_name = TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME
    else:
        cuda_scatter_producer_id = cuda_producer_id
        cuda_producer_name = TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME
    specs = (_native_producer_spec(cuda_producer_id, cuda_producer_name, local.role),)
    return TerminalRankNativeProducerPlan(
        role=local.role,
        specs=specs,
        cuda_source_producer_id=cuda_source_producer_id,
        cuda_scatter_producer_id=cuda_scatter_producer_id,
        next_producer_id=cuda_producer_id + len(specs),
    )


def _bind_cuda_source_producer(
    runtime: NativeTerminalRuntime,
    *,
    testing: bool,
) -> CudaTerminalProducer:
    """Bind the source rank's direct CUDA completion producer.

    :param runtime: Dormant immutable native runtime.
    :param testing: Whether deterministic native test controls are required.
    :returns: Runtime-owned direct CUDA producer.
    """

    return CudaTerminalProducer.from_runtime(
        runtime,
        TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME,
        CudaTerminalEventKind.SOURCE_PRODUCER_COMPLETED,
        testing=testing,
    )


def _bind_cuda_scatter_producer(
    runtime: NativeTerminalRuntime,
    *,
    testing: bool,
) -> CudaTerminalProducer:
    """Bind the decode rank's direct CUDA scatter producer.

    :param runtime: Dormant immutable native runtime.
    :param testing: Whether deterministic native test controls are required.
    :returns: Runtime-owned direct CUDA producer.
    """

    return CudaTerminalProducer.from_runtime(
        runtime,
        TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME,
        CudaTerminalEventKind.DECODE_SCATTER_TERMINAL,
        testing=testing,
    )


class TerminalRankRuntimeEnrollment:
    """One rank's sole runtime and lifecycle-owned native producers."""

    _binding: TerminalStartupRankBinding
    _config: TerminalRankRuntimeConfig
    _native_plan: TerminalRankNativeProducerPlan
    _runtime: NativeTerminalRuntime
    _cuda_source: CudaTerminalProducer | None
    _cuda_scatter: CudaTerminalProducer | None
    _native_producer_disposition: TerminalRankNativeProducerDisposition
    _retirement_failure: str | None
    _retirement_lock: threading.Lock

    def __init__(
        self,
        *,
        binding: TerminalStartupRankBinding,
        config: TerminalRankRuntimeConfig,
        native_plan: TerminalRankNativeProducerPlan,
        runtime: NativeTerminalRuntime,
        cuda_source: CudaTerminalProducer | None,
        cuda_scatter: CudaTerminalProducer | None,
    ) -> None:
        """Retain exact process-lifetime ownership.

        :param binding: Complete immutable startup authority.
        :param config: Frozen queue and retirement bounds.
        :param native_plan: Role-specific native producer namespace.
        :param runtime: Sole dormant native owner runtime.
        :param cuda_source: Source-only CUDA completion producer.
        :param cuda_scatter: Decode-only CUDA scatter producer.
        """

        self._binding = binding
        self._config = config
        self._native_plan = native_plan
        self._runtime = runtime
        self._cuda_source = cuda_source
        self._cuda_scatter = cuda_scatter
        self._native_producer_disposition = TerminalRankNativeProducerDisposition.OPEN
        self._retirement_failure = None
        self._retirement_lock = threading.Lock()

    @property
    def binding(self) -> TerminalStartupRankBinding:
        """Return the exact startup authority used for construction.

        :returns: Immutable rank binding.
        """

        return self._binding

    @property
    def native_plan(self) -> TerminalRankNativeProducerPlan:
        """Return the exact appended native producer namespace.

        :returns: Frozen role-specific producer plan.
        """

        return self._native_plan

    @property
    def runtime(self) -> NativeTerminalRuntime:
        """Return the sole process-lifetime native runtime.

        :returns: Dormant runtime until startup activation commits.
        """

        return self._runtime

    @property
    def cuda_scatter(self) -> CudaTerminalProducer | None:
        """Return the decode CUDA producer when this is a decode rank.

        :returns: Decode-only CUDA completion producer.
        """

        return self._cuda_scatter

    @property
    def cuda_source(self) -> CudaTerminalProducer | None:
        """Return the source CUDA producer when this is a source rank.

        :returns: Source-only CUDA completion producer.
        """

        return self._cuda_source

    @property
    def native_producer_disposition(self) -> TerminalRankNativeProducerDisposition:
        """Return the unified native producer lifetime.

        :returns: Open, retiring, retired, or sticky failed state.
        """

        with self._retirement_lock:
            return self._native_producer_disposition

    @property
    def retirement_failure(self) -> str | None:
        """Return sticky native producer retirement evidence.

        :returns: Complete traceback after a fail-closed retirement.
        """

        with self._retirement_lock:
            return self._retirement_failure

    def retire_native_producers(self) -> None:
        """Stop, join, and close every native producer exactly once.

        Runtime lifecycle admission must already be closed. Any partial or
        timed-out retirement enters the runtime's fail-closed abort path and
        makes the failure sticky. A completed retirement is idempotent.
        """

        with self._retirement_lock:
            if (
                self._native_producer_disposition
                is TerminalRankNativeProducerDisposition.RETIRED
            ):
                return
            if (
                self._native_producer_disposition
                is TerminalRankNativeProducerDisposition.FAILED
            ):
                raise TerminalRankRuntimeEnrollmentError(
                    "native producer retirement already failed:\n"
                    f"{self._retirement_failure}"
                )
            if (
                self._native_producer_disposition
                is TerminalRankNativeProducerDisposition.RETIRING
            ):
                raise TerminalRankRuntimeEnrollmentError(
                    "native producer retirement re-entered"
                )
            allowed_runtime_dispositions = (
                NativeTerminalRuntimeDisposition.DRAINING,
                NativeTerminalRuntimeDisposition.ABORT_DRAINING,
                NativeTerminalRuntimeDisposition.PROCESS_FATAL,
            )
            if self._runtime.disposition not in allowed_runtime_dispositions:
                raise TerminalRankRuntimeEnrollmentError(
                    "runtime admission must close before native producer retirement"
                )
            self._native_producer_disposition = (
                TerminalRankNativeProducerDisposition.RETIRING
            )
            try:
                self._retire_native_producers_locked()
            except Exception as error:
                failure = traceback.format_exc()
                self._native_producer_disposition = (
                    TerminalRankNativeProducerDisposition.FAILED
                )
                try:
                    self._runtime.begin_abort()
                except Exception:  # noqa: BLE001
                    failure += "\nRuntime abort also failed:\n" + traceback.format_exc()
                self._retirement_failure = failure
                raise TerminalRankRuntimeEnrollmentError(
                    "native producer retirement failed closed"
                ) from error
            self._native_producer_disposition = (
                TerminalRankNativeProducerDisposition.RETIRED
            )

    def _retire_native_producers_locked(self) -> None:
        """Perform one ordered native producer retirement under the lock."""

        cuda_source = self._cuda_source
        cuda_scatter = self._cuda_scatter
        if cuda_source is not None:
            cuda_source.stop_admission()
        if cuda_scatter is not None:
            cuda_scatter.stop_admission()
        deadline = (
            time.monotonic() + self._config.native_producer_retirement_timeout_seconds
        )
        if cuda_source is not None and not self._join_before_deadline(
            cuda_source.join,
            deadline,
        ):
            raise TimeoutError("source CUDA terminal producer retirement timed out")
        if cuda_scatter is not None and not self._join_before_deadline(
            cuda_scatter.join,
            deadline,
        ):
            raise TimeoutError("CUDA terminal producer retirement timed out")
        if cuda_source is not None:
            cuda_source.close()
        if cuda_scatter is not None:
            cuda_scatter.close()

    @staticmethod
    def _join_before_deadline(
        join: Callable[[float], bool],
        deadline: float,
    ) -> bool:
        """Invoke one producer join against the shared absolute deadline.

        :param join: Bound producer join callable.
        :param deadline: Shared absolute monotonic deadline.
        :returns: Whether producer retirement committed in time.
        """

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0.0:
            return False
        return bool(join(float(remaining_seconds)))


class TerminalRankRuntimeEnrollmentFactory:
    """One-shot constructor for a sealed rank's sole dormant runtime."""

    _binding: TerminalStartupRankBinding
    _config: TerminalRankRuntimeConfig
    _native_plan: TerminalRankNativeProducerPlan
    _owner_identity: NativeTerminalProcessIdentity
    _consumed: bool
    _enrollment: TerminalRankRuntimeEnrollment | None
    _lock: threading.Lock

    def __init__(
        self,
        binding: TerminalStartupRankBinding,
        config: TerminalRankRuntimeConfig,
    ) -> None:
        """Freeze rank identity, producer numbering, and capacities.

        :param binding: Complete immutable startup rank authority.
        :param config: Frozen runtime capacities and retirement deadline.
        """

        if type(config) is not TerminalRankRuntimeConfig:
            raise TypeError("config must be TerminalRankRuntimeConfig")
        native_plan = build_terminal_rank_native_producer_plan(binding)
        self._binding = binding
        self._config = config
        self._native_plan = native_plan
        self._owner_identity = NativeTerminalProcessIdentity.from_identity(
            binding.advertisement.terminal_identity
        )
        self._consumed = False
        self._enrollment = None
        self._lock = threading.Lock()

    @property
    def native_plan(self) -> TerminalRankNativeProducerPlan:
        """Return the frozen native producer suffix before construction.

        :returns: Role-specific native producer plan.
        """

        return self._native_plan

    @property
    def enrollment(self) -> TerminalRankRuntimeEnrollment | None:
        """Return the sole successfully constructed enrollment.

        :returns: Enrollment after successful one-shot construction.
        """

        with self._lock:
            return self._enrollment

    def create(
        self,
        *,
        cuda_testing: bool = False,
    ) -> TerminalRankRuntimeEnrollment:
        """Construct one dormant runtime and bind its native producers.

        Construction is consumed before the first native allocation. A failed
        attempt cannot be retried against a partially initialized owner.

        :param cuda_testing: Whether decode CUDA test controls are required.
        :returns: Sole rank-lifetime runtime enrollment.
        """

        if type(cuda_testing) is not bool:
            raise TypeError("cuda_testing must be bool")
        with self._lock:
            if self._consumed:
                raise TerminalRankRuntimeEnrollmentError(
                    "runtime enrollment factory is one-shot"
                )
            self._consumed = True
            producer_specs = (
                *self._binding.python_producers.specs,
                *self._native_plan.specs,
            )
            config = self._config
            runtime = NativeTerminalRuntime(
                owner_identity=self._owner_identity,
                producer_specs=producer_specs,
                fatal_producer_id=self._binding.python_producers.fatal_producer_id,
                input_capacity=config.input_capacity,
                output_capacity=config.output_capacity,
                maximum_live_lifecycles=config.maximum_live_lifecycles,
                scheduler_capacity=config.scheduler_capacity,
                coordinator_capacity=config.coordinator_capacity,
                lifecycle_capacity=config.lifecycle_capacity,
                source_work_capacity=config.source_work_capacity,
                decode_work_capacity=config.decode_work_capacity,
                publisher_capacity=config.publisher_capacity,
                observation_capacity=config.observation_capacity,
            )
            cuda_source: CudaTerminalProducer | None = None
            cuda_scatter: CudaTerminalProducer | None = None
            if self._native_plan.role is TerminalOwnerRole.SOURCE:
                cuda_source = _bind_cuda_source_producer(
                    runtime,
                    testing=cuda_testing,
                )
            else:
                cuda_scatter = _bind_cuda_scatter_producer(
                    runtime,
                    testing=cuda_testing,
                )
            enrollment = TerminalRankRuntimeEnrollment(
                binding=self._binding,
                config=config,
                native_plan=self._native_plan,
                runtime=runtime,
                cuda_source=cuda_source,
                cuda_scatter=cuda_scatter,
            )
            self._enrollment = enrollment
            return enrollment
