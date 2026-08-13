import secrets

from sglang.srt.disaggregation.terminal_progress.deployment_cohort import (
    TerminalDeploymentCohort,
    TerminalDeploymentCohortError,
    TerminalDeploymentLocalService,
    TerminalDeploymentRole,
)
from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortExpectation,
    TerminalStartupServiceExpectation,
)


def build_terminal_startup_cohort_expectation(
    cohort: TerminalDeploymentCohort,
    local_service: TerminalDeploymentLocalService,
) -> TerminalStartupCohortExpectation:
    """Project an authenticated deployment cohort into startup membership.

    :param cohort: Canonical launcher-authenticated deployment cohort.
    :param local_service: Exact local member selected while loading the cohort.
    :returns: Complete source-first startup membership for the deployment epoch.
    :raises TerminalDeploymentCohortError: If the local member is not exact.
    """

    if type(cohort) is not TerminalDeploymentCohort:
        raise TypeError("cohort must be TerminalDeploymentCohort")
    if type(local_service) is not TerminalDeploymentLocalService:
        raise TypeError("local_service must be TerminalDeploymentLocalService")
    if local_service.group_id != cohort.group_id or not secrets.compare_digest(
        local_service.cohort_digest,
        cohort.digest,
    ):
        raise TerminalDeploymentCohortError(
            "local service belongs to another deployment cohort"
        )

    exact_local_service = cohort.require_local_service(
        service_id=local_service.service_id,
        role=local_service.role,
        launch_instance_id=local_service.launch_instance_id,
        tensor_parallel_size=local_service.tensor_parallel_size,
        origin=local_service.origin,
        bootstrap_endpoint=local_service.bootstrap_endpoint,
    )
    if exact_local_service != local_service:
        raise TerminalDeploymentCohortError(
            "local service differs from exact deployment membership"
        )

    source = TerminalStartupServiceExpectation(
        service_id=cohort.prefill.service_id,
        service_origin=cohort.prefill.origin,
        role=TerminalOwnerRole.SOURCE,
        launch_instance_id=cohort.prefill.launch_instance_id.bytes,
        tensor_parallel_size=cohort.prefill.tensor_parallel_size,
    )
    decoders = tuple(
        TerminalStartupServiceExpectation(
            service_id=decoder.service_id,
            service_origin=decoder.origin,
            role=TerminalOwnerRole.DECODE,
            launch_instance_id=decoder.launch_instance_id.bytes,
            tensor_parallel_size=decoder.tensor_parallel_size,
        )
        for decoder in cohort.decoders
    )
    expectation = TerminalStartupCohortExpectation(
        group_id=cohort.group_id,
        cohort_sha256=cohort.digest,
        services=(source, *decoders),
    )

    local_expectation = expectation.service(local_service.service_id)
    expected_role = (
        TerminalOwnerRole.SOURCE
        if local_service.role is TerminalDeploymentRole.PREFILL
        else TerminalOwnerRole.DECODE
    )
    if (
        local_expectation.service_origin != local_service.origin
        or local_expectation.role is not expected_role
        or local_expectation.launch_instance_id
        != local_service.launch_instance_id.bytes
        or local_expectation.tensor_parallel_size != local_service.tensor_parallel_size
    ):
        raise TerminalDeploymentCohortError(
            "local startup row differs from deployment membership"
        )
    return expectation
