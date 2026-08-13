import dataclasses
import hashlib
import math
import secrets
import uuid

from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentCohort,
    TerminalDeploymentLocalService,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortExpectation,
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)
from sglang.srt.disaggregation.terminal_progress.startup_http import (
    TERMINAL_STARTUP_ROUTE,
    join_terminal_startup_cohort,
)
from sglang.srt.disaggregation.terminal_progress.startup_producers import (
    TerminalStartupPythonProducerPlan,
    build_terminal_startup_python_producer_plan,
)
from sglang.srt.utils.network import NetworkAddress


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupRankBinding:
    """One rank's immutable deployment epoch and producer authority.

    :ivar advertisement: Exact locally observed native identity.
    :ivar matrix: Complete generation-authenticated rank population.
    :ivar python_producers: Least-authority process-local producer namespace.
    """

    advertisement: TerminalStartupRankAdvertisement
    matrix: TerminalStartupCohortMatrix
    python_producers: TerminalStartupPythonProducerPlan

    def __post_init__(self) -> None:
        """Validate one internally consistent startup result."""

        if type(self.advertisement) is not TerminalStartupRankAdvertisement:
            raise TypeError("advertisement must be TerminalStartupRankAdvertisement")
        if type(self.matrix) is not TerminalStartupCohortMatrix:
            raise TypeError("matrix must be TerminalStartupCohortMatrix")
        if type(self.python_producers) is not TerminalStartupPythonProducerPlan:
            raise TypeError(
                "python_producers must be TerminalStartupPythonProducerPlan"
            )
        if self.matrix.rank(*self.advertisement.key) != self.advertisement:
            raise ValueError("startup matrix does not contain the exact local rank")


def build_terminal_startup_rank_advertisement(
    cohort: TerminalDeploymentCohort,
    local_service: TerminalDeploymentLocalService,
    expectation: TerminalStartupCohortExpectation,
    *,
    tensor_parallel_rank: int,
    process_generation: str,
    nixl_agent_name: str,
    nixl_agent_metadata: bytes,
) -> TerminalStartupRankAdvertisement:
    """Bind one observed NIXL identity to launcher-authenticated membership.

    :param cohort: Exact canonical deployment cohort.
    :param local_service: Exact local service selected from that cohort.
    :param expectation: Complete startup membership derived from the cohort.
    :param tensor_parallel_rank: Local rank within the service TP group.
    :param process_generation: Canonical NIXL process-generation UUID.
    :param nixl_agent_name: Exact initialized NIXL agent name.
    :param nixl_agent_metadata: Complete frozen native metadata bytes.
    :returns: Validated canonical startup advertisement.
    """

    if type(cohort) is not TerminalDeploymentCohort:
        raise TypeError("cohort must be TerminalDeploymentCohort")
    if type(local_service) is not TerminalDeploymentLocalService:
        raise TypeError("local_service must be TerminalDeploymentLocalService")
    if type(expectation) is not TerminalStartupCohortExpectation:
        raise TypeError("expectation must be TerminalStartupCohortExpectation")
    if not secrets.compare_digest(cohort.digest, local_service.cohort_digest):
        raise ValueError("local service belongs to another deployment cohort")
    if not secrets.compare_digest(cohort.digest, expectation.cohort_sha256):
        raise ValueError("startup expectation belongs to another deployment cohort")
    if (
        type(tensor_parallel_rank) is not int
        or tensor_parallel_rank < 0
        or tensor_parallel_rank >= local_service.tensor_parallel_size
    ):
        raise ValueError("tensor_parallel_rank is outside the local service")
    if type(process_generation) is not str:
        raise TypeError("process_generation must be a string")
    parsed_generation = uuid.UUID(process_generation)
    if parsed_generation.int == 0 or str(parsed_generation) != process_generation:
        raise ValueError("process_generation must be a canonical non-nil UUID")
    if type(nixl_agent_metadata) is not bytes or len(nixl_agent_metadata) == 0:
        raise ValueError("nixl_agent_metadata must be nonempty bytes")

    role = (
        TerminalOwnerRole.SOURCE
        if local_service.role is TerminalDeploymentRole.PREFILL
        else TerminalOwnerRole.DECODE
    )
    advertisement = TerminalStartupRankAdvertisement(
        group_id=local_service.group_id,
        cohort_sha256=local_service.cohort_digest,
        service_id=local_service.service_id,
        service_origin=local_service.origin,
        role=role,
        launch_instance_id=local_service.launch_instance_id.bytes,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=local_service.tensor_parallel_size,
        process_generation=parsed_generation.bytes,
        nixl_agent_name=nixl_agent_name,
        nixl_agent_metadata_sha256=hashlib.sha256(nixl_agent_metadata).digest(),
    )
    expectation.require_advertisement(advertisement)
    return advertisement


def join_terminal_startup_rank(
    cohort: TerminalDeploymentCohort,
    local_service: TerminalDeploymentLocalService,
    expectation: TerminalStartupCohortExpectation,
    *,
    tensor_parallel_rank: int,
    process_generation: str,
    nixl_agent_name: str,
    nixl_agent_metadata: bytes,
    timeout_seconds: float,
    first_producer_id: int = 0,
) -> TerminalStartupRankBinding:
    """Join one rank and freeze its complete Python producer authority.

    :param cohort: Exact canonical deployment cohort.
    :param local_service: Exact local service selected from that cohort.
    :param expectation: Complete startup membership derived from the cohort.
    :param tensor_parallel_rank: Local rank within the service TP group.
    :param process_generation: Canonical NIXL process-generation UUID.
    :param nixl_agent_name: Exact initialized NIXL agent name.
    :param nixl_agent_metadata: Complete frozen native metadata bytes.
    :param timeout_seconds: Hash-bound join deadline.
    :param first_producer_id: First process-local Python producer identifier.
    :returns: Complete immutable rank binding.
    """

    if (
        type(timeout_seconds) is not float
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be a positive finite float")
    advertisement = build_terminal_startup_rank_advertisement(
        cohort,
        local_service,
        expectation,
        tensor_parallel_rank=tensor_parallel_rank,
        process_generation=process_generation,
        nixl_agent_name=nixl_agent_name,
        nixl_agent_metadata=nixl_agent_metadata,
    )
    bootstrap = cohort.prefill.bootstrap_endpoint
    endpoint = (
        NetworkAddress(bootstrap.host, bootstrap.port).to_url()
        + TERMINAL_STARTUP_ROUTE
    )
    matrix = join_terminal_startup_cohort(
        endpoint,
        advertisement,
        timeout_seconds,
    )
    matrix.require_expectation(expectation)
    if matrix.rank(local_service.service_id, tensor_parallel_rank) != advertisement:
        raise RuntimeError("startup registry returned another local rank identity")
    python_producers = build_terminal_startup_python_producer_plan(
        matrix,
        local_service_id=local_service.service_id,
        local_tensor_parallel_rank=tensor_parallel_rank,
        first_producer_id=first_producer_id,
    )
    return TerminalStartupRankBinding(
        advertisement=advertisement,
        matrix=matrix,
        python_producers=python_producers,
    )
