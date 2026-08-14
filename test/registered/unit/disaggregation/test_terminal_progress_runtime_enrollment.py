import dataclasses
import hashlib
import uuid

import pytest

from sglang.srt.disaggregation.terminal_progress import (
    runtime_enrollment as enrollment_module,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalProcessIdentity,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalRuntimeDisposition,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.runtime_enrollment import (
    TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME,
    TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME,
    TerminalRankNativeProducerDisposition,
    TerminalRankRuntimeConfig,
    TerminalRankRuntimeEnrollmentError,
    TerminalRankRuntimeEnrollmentFactory,
    build_terminal_rank_native_producer_plan,
)
from sglang.srt.disaggregation.terminal_progress.startup_binding import (
    TerminalStartupRankBinding,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    build_terminal_startup_python_producer_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_COHORT_DIGEST = b"c" * 32
_FIRST_PRODUCER_ID = 7


class _FakeRuntime:
    """Constructor-exact dormant runtime fixture."""

    owner_identity: NativeTerminalProcessIdentity
    producer_specs: tuple[NativeTerminalRuntimeProducerSpec, ...]
    fatal_producer_id: int
    capacities: tuple[int, ...]
    disposition: NativeTerminalRuntimeDisposition
    start_calls: int
    begin_abort_calls: int

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
        """Capture the complete runtime construction.

        :param owner_identity: Exact local owner.
        :param producer_specs: Frozen producer directory.
        :param fatal_producer_id: Local fail-closed producer.
        :param input_capacity: Native input capacity.
        :param output_capacity: Native output capacity.
        :param maximum_live_lifecycles: Lifecycle population bound.
        :param scheduler_capacity: Scheduler inbox capacity.
        :param coordinator_capacity: Coordinator inbox capacity.
        :param lifecycle_capacity: Lifecycle inbox capacity.
        :param source_work_capacity: Source work inbox capacity.
        :param decode_work_capacity: Decode work inbox capacity.
        :param publisher_capacity: Publisher inbox capacity.
        :param observation_capacity: Observation inbox capacity.
        """

        self.owner_identity = owner_identity
        self.producer_specs = producer_specs
        self.fatal_producer_id = fatal_producer_id
        self.capacities = (
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
        self.disposition = NativeTerminalRuntimeDisposition.CREATED
        self.start_calls = 0
        self.begin_abort_calls = 0

    def producer_id(self, name: str) -> int:
        """Resolve one captured producer name.

        :param name: Stable producer name.
        :returns: Exact captured producer ID.
        """

        matches = tuple(
            spec.registration.producer_id
            for spec in self.producer_specs
            if spec.registration.name == name
        )
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def start(self) -> None:
        """Record a forbidden factory-side start."""

        self.start_calls += 1

    def begin_abort(self) -> None:
        """Record fail-closed retirement."""

        self.begin_abort_calls += 1
        self.disposition = NativeTerminalRuntimeDisposition.ABORT_DRAINING


class _FakeNativeProducer:
    """Ordered native producer retirement fixture."""

    producer_id: int
    name: str
    calls: list[str]
    join_result: bool

    def __init__(self, producer_id: int, name: str, calls: list[str]) -> None:
        """Construct one open producer.

        :param producer_id: Exact runtime namespace.
        :param name: Stable call-log prefix.
        :param calls: Shared ordered call log.
        """

        self.producer_id = producer_id
        self.name = name
        self.calls = calls
        self.join_result = True

    def stop_admission(self) -> None:
        """Record admission closure."""

        self.calls.append(f"{self.name}:stop")

    def join(self, timeout_seconds: float) -> bool:
        """Record the shared-deadline join.

        :param timeout_seconds: Remaining absolute-deadline budget.
        :returns: Prospective fixture result.
        """

        assert timeout_seconds > 0.0
        self.calls.append(f"{self.name}:join")
        return self.join_result

    def close(self) -> None:
        """Record exact-zero close."""

        self.calls.append(f"{self.name}:close")


@dataclasses.dataclass(slots=True)
class _Construction:
    """Captured fake runtime and producer population.

    :ivar runtimes: Every constructed runtime.
    :ivar cuda: Bound CUDA producers.
    :ivar calls: Shared producer lifecycle call log.
    :ivar cuda_testing: Role-specific CUDA construction modes.
    """

    runtimes: list[_FakeRuntime] = dataclasses.field(default_factory=list)
    cuda: list[_FakeNativeProducer] = dataclasses.field(default_factory=list)
    calls: list[str] = dataclasses.field(default_factory=list)
    cuda_testing: list[bool] = dataclasses.field(default_factory=list)


def _rank(
    *,
    service_id: str,
    role: TerminalOwnerRole,
    rank: int,
    size: int,
    generation: int,
    agent_name: str,
    port: int,
) -> TerminalStartupRankAdvertisement:
    """Build one exact startup matrix row.

    :param service_id: Static model service identifier.
    :param role: Source or decode owner role.
    :param rank: Tensor-parallel rank.
    :param size: Tensor-parallel width.
    :param generation: Exact process UUID integer.
    :param agent_name: Unique NIXL agent name.
    :param port: Service origin port.
    :returns: Complete startup advertisement.
    """

    return TerminalStartupRankAdvertisement(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        service_id=service_id,
        service_origin=f"http://127.0.0.1:{port}",
        role=role,
        launch_instance_id=uuid.UUID(int=port).bytes,
        tensor_parallel_rank=rank,
        tensor_parallel_size=size,
        process_generation=uuid.UUID(int=generation).bytes,
        nixl_agent_name=agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(agent_name.encode()).digest(),
    )


def _matrix() -> TerminalStartupCohortMatrix:
    """Build one TP2 source and one TP1 decoder.

    :returns: Canonically ordered sealed rank matrix.
    """

    return TerminalStartupCohortMatrix(
        group_id="group-a",
        cohort_sha256=_COHORT_DIGEST,
        ranks=(
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                rank=0,
                size=2,
                generation=101,
                agent_name="prefill-agent-0",
                port=32001,
            ),
            _rank(
                service_id="prefill-a",
                role=TerminalOwnerRole.SOURCE,
                rank=1,
                size=2,
                generation=102,
                agent_name="prefill-agent-1",
                port=32001,
            ),
            _rank(
                service_id="decode-a",
                role=TerminalOwnerRole.DECODE,
                rank=0,
                size=1,
                generation=201,
                agent_name="decode-agent-0",
                port=32002,
            ),
        ),
    )


def _binding(service_id: str, rank: int) -> TerminalStartupRankBinding:
    """Build one internally consistent rank binding.

    :param service_id: Local service identifier.
    :param rank: Local tensor-parallel rank.
    :returns: Complete immutable rank binding.
    """

    matrix = _matrix()
    return TerminalStartupRankBinding(
        advertisement=matrix.rank(service_id, rank),
        matrix=matrix,
        python_producers=build_terminal_startup_python_producer_plan(
            matrix,
            local_service_id=service_id,
            local_tensor_parallel_rank=rank,
            first_producer_id=_FIRST_PRODUCER_ID,
        ),
    )


def _config() -> TerminalRankRuntimeConfig:
    """Build distinct frozen capacities for forwarding assertions.

    :returns: Complete runtime configuration.
    """

    return TerminalRankRuntimeConfig(
        input_capacity=101,
        output_capacity=102,
        maximum_live_lifecycles=103,
        scheduler_capacity=104,
        coordinator_capacity=105,
        lifecycle_capacity=106,
        source_work_capacity=107,
        decode_work_capacity=108,
        publisher_capacity=109,
        observation_capacity=110,
        native_producer_retirement_timeout_seconds=3.0,
    )


def _install_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> _Construction:
    """Replace native allocation with constructor-exact fakes.

    :param monkeypatch: Pytest mutation fixture.
    :returns: Captured construction population.
    """

    construction = _Construction()

    class _CapturingRuntime(_FakeRuntime):
        """Runtime fixture which records every instance."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Construct and retain one fake runtime.

            :param args: Positional constructor values.
            :param kwargs: Named constructor values.
            """

            super().__init__(*args, **kwargs)
            construction.runtimes.append(self)

    def bind_cuda(
        runtime: _FakeRuntime,
        *,
        testing: bool,
        producer_name: str,
    ) -> _FakeNativeProducer:
        """Bind one captured role-specific CUDA producer.

        :param runtime: Captured rank runtime.
        :param testing: Prospective native test mode.
        :returns: Fake direct CUDA producer.
        """

        construction.cuda_testing.append(testing)
        producer = _FakeNativeProducer(
            runtime.producer_id(producer_name),
            "cuda",
            construction.calls,
        )
        construction.cuda.append(producer)
        return producer

    monkeypatch.setattr(enrollment_module, "NativeTerminalRuntime", _CapturingRuntime)
    monkeypatch.setattr(
        enrollment_module,
        "_bind_cuda_source_producer",
        lambda runtime, *, testing: bind_cuda(
            runtime,
            testing=testing,
            producer_name=TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME,
        ),
    )
    monkeypatch.setattr(
        enrollment_module,
        "_bind_cuda_scatter_producer",
        lambda runtime, *, testing: bind_cuda(
            runtime,
            testing=testing,
            producer_name=TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME,
        ),
    )
    return construction


def test_native_producer_numbering_is_a_role_specific_contiguous_suffix() -> None:
    """Native producers append exactly after each rank's Python authority."""

    source_binding = _binding("prefill-a", 0)
    source = build_terminal_rank_native_producer_plan(source_binding)
    assert source_binding.python_producers.next_producer_id == 12
    assert source.cuda_source_producer_id == 12
    assert source.cuda_scatter_producer_id is None
    assert source.next_producer_id == 13
    assert tuple(spec.registration.name for spec in source.specs) == (
        TERMINAL_NATIVE_CUDA_SOURCE_PRODUCER_NAME,
    )

    decode_binding = _binding("decode-a", 0)
    decode = build_terminal_rank_native_producer_plan(decode_binding)
    assert decode_binding.python_producers.next_producer_id == 13
    assert decode.cuda_source_producer_id is None
    assert decode.cuda_scatter_producer_id == 13
    assert decode.next_producer_id == 14
    assert tuple(spec.registration.name for spec in decode.specs) == (
        TERMINAL_NATIVE_CUDA_SCATTER_PRODUCER_NAME,
    )


def test_factory_rejects_a_python_plan_from_another_matrix_rank() -> None:
    """A type-valid producer prefix cannot replace local startup authority."""

    source = _binding("prefill-a", 0)
    decode = _binding("decode-a", 0)
    forged = TerminalStartupRankBinding(
        advertisement=source.advertisement,
        matrix=source.matrix,
        python_producers=decode.python_producers,
    )

    with pytest.raises(ValueError, match="another startup authority"):
        TerminalRankRuntimeEnrollmentFactory(forged, _config())


@pytest.mark.parametrize(
    ("service_id", "rank", "expected_native_ids"),
    (
        ("prefill-a", 1, (12,)),
        ("decode-a", 0, (13,)),
    ),
)
def test_factory_constructs_one_dormant_role_exact_runtime(
    monkeypatch: pytest.MonkeyPatch,
    service_id: str,
    rank: int,
    expected_native_ids: tuple[int, ...],
) -> None:
    """Construction freezes exact identity and never crosses activation.

    :param monkeypatch: Pytest mutation fixture.
    :param service_id: Local matrix service.
    :param rank: Local service rank.
    :param expected_native_ids: Role-specific native suffix.
    """

    construction = _install_construction(monkeypatch)
    binding = _binding(service_id, rank)
    factory = TerminalRankRuntimeEnrollmentFactory(binding, _config())
    enrollment = factory.create(cuda_testing=True)

    assert factory.enrollment is enrollment
    assert len(construction.runtimes) == 1
    runtime = construction.runtimes[0]
    assert runtime.owner_identity == NativeTerminalProcessIdentity.from_identity(
        binding.advertisement.terminal_identity
    )
    assert runtime.producer_specs == (
        *binding.python_producers.specs,
        *factory.native_plan.specs,
    )
    assert (
        tuple(spec.registration.producer_id for spec in factory.native_plan.specs)
        == expected_native_ids
    )
    assert runtime.capacities == tuple(range(101, 111))
    assert runtime.disposition is NativeTerminalRuntimeDisposition.CREATED
    assert runtime.start_calls == 0
    has_source_cuda = binding.advertisement.role is TerminalOwnerRole.SOURCE
    assert (enrollment.cuda_source is not None) is has_source_cuda
    assert (enrollment.cuda_scatter is not None) is (not has_source_cuda)
    assert len(construction.cuda) == 1
    assert construction.cuda_testing == [True]

    with pytest.raises(TerminalRankRuntimeEnrollmentError, match="one-shot"):
        factory.create(cuda_testing=True)
    assert len(construction.runtimes) == 1


def test_decode_native_retirement_is_ordered_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One owner stops all producers before shared-deadline join and close.

    :param monkeypatch: Pytest mutation fixture.
    """

    construction = _install_construction(monkeypatch)
    enrollment = TerminalRankRuntimeEnrollmentFactory(
        _binding("decode-a", 0),
        _config(),
    ).create()
    runtime = construction.runtimes[0]
    runtime.disposition = NativeTerminalRuntimeDisposition.DRAINING

    enrollment.retire_native_producers()
    enrollment.retire_native_producers()

    assert construction.calls == [
        "cuda:stop",
        "cuda:join",
        "cuda:close",
    ]
    assert (
        enrollment.native_producer_disposition
        is TerminalRankNativeProducerDisposition.RETIRED
    )
    assert enrollment.retirement_failure is None
    assert runtime.begin_abort_calls == 0


def test_retirement_timeout_is_sticky_and_enters_runtime_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial native retirement cannot be retried into apparent health.

    :param monkeypatch: Pytest mutation fixture.
    """

    construction = _install_construction(monkeypatch)
    enrollment = TerminalRankRuntimeEnrollmentFactory(
        _binding("prefill-a", 0),
        _config(),
    ).create()
    runtime = construction.runtimes[0]
    runtime.disposition = NativeTerminalRuntimeDisposition.DRAINING
    construction.cuda[0].join_result = False

    with pytest.raises(
        TerminalRankRuntimeEnrollmentError,
        match="failed closed",
    ):
        enrollment.retire_native_producers()
    assert runtime.begin_abort_calls == 1
    assert (
        enrollment.native_producer_disposition
        is TerminalRankNativeProducerDisposition.FAILED
    )
    assert enrollment.retirement_failure is not None
    assert "TimeoutError" in enrollment.retirement_failure

    with pytest.raises(
        TerminalRankRuntimeEnrollmentError,
        match="already failed",
    ):
        enrollment.retire_native_producers()
    assert runtime.begin_abort_calls == 1
    assert construction.calls == ["cuda:stop", "cuda:join"]


def test_runtime_config_is_frozen_and_rejects_implicit_capacity() -> None:
    """Every runtime bound is explicit and immutable."""

    config = _config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.input_capacity = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive integers"):
        dataclasses.replace(config, publisher_capacity=0)
