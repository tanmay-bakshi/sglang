import dataclasses

from sglang.srt.disaggregation.terminal_progress.identity import TerminalOwnerRole
from sglang.srt.disaggregation.terminal_progress.native_state import (
    NativeTerminalOwnerRole,
    NativeTerminalProcessIdentity,
    NativeTerminalProducerClass,
    NativeTerminalProducerRegistration,
)
from sglang.srt.disaggregation.terminal_progress.runtime import (
    NativeTerminalProducerDelivery,
    NativeTerminalRuntimeProducerSpec,
)
from sglang.srt.disaggregation.terminal_progress.startup_cohort import (
    TerminalStartupCohortMatrix,
    TerminalStartupRankAdvertisement,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TerminalStartupPythonProducerPlan:
    """Least-privilege Python producers derived from a sealed rank matrix.

    :ivar specs: Deterministically numbered local and exact-issuer producers.
    :ivar fatal_producer_id: Local producer used for process-fatal transitions.
    :ivar next_producer_id: First identifier available to native producers.
    """

    specs: tuple[NativeTerminalRuntimeProducerSpec, ...]
    fatal_producer_id: int
    next_producer_id: int

    def __post_init__(self) -> None:
        """Validate producer-plan numbering and fatal authority."""

        if type(self.specs) is not tuple or len(self.specs) < 2:
            raise ValueError("specs must contain local and receipt producers")
        if any(
            type(spec) is not NativeTerminalRuntimeProducerSpec for spec in self.specs
        ):
            raise TypeError("specs contain an invalid producer specification")
        producer_ids = tuple(spec.registration.producer_id for spec in self.specs)
        if len(set(producer_ids)) != len(producer_ids):
            raise ValueError("producer IDs must be unique")
        if any(
            spec.delivery is not NativeTerminalProducerDelivery.PYTHON
            for spec in self.specs
        ):
            raise ValueError("startup producer plan must be Python-owned")
        if self.fatal_producer_id != producer_ids[0]:
            raise ValueError("fatal producer must be the first local producer")
        if (
            self.specs[0].registration.producer_class
            is not NativeTerminalProducerClass.LOCAL
        ):
            raise ValueError("fatal producer must have local authority")
        if self.next_producer_id != max(producer_ids) + 1:
            raise ValueError("next_producer_id must follow every Python producer")
        if producer_ids != tuple(range(producer_ids[0], self.next_producer_id)):
            raise ValueError("startup producer IDs must be contiguous")


def _native_role(rank: TerminalStartupRankAdvertisement) -> NativeTerminalOwnerRole:
    """Convert one startup role to the stable native owner code.

    :param rank: Observed local rank.
    :returns: Native source or decode owner role.
    """

    if rank.role is TerminalOwnerRole.SOURCE:
        return NativeTerminalOwnerRole.SOURCE
    return NativeTerminalOwnerRole.DECODE


def _producer_spec(
    *,
    producer_id: int,
    name: str,
    producer_class: NativeTerminalProducerClass,
    allowed_role: NativeTerminalOwnerRole,
    authenticated_issuer: NativeTerminalProcessIdentity | None,
) -> NativeTerminalRuntimeProducerSpec:
    """Build one Python-owned exact-authority producer.

    :param producer_id: Stable process-local producer ID.
    :param name: Evidence-facing producer name.
    :param producer_class: Local, control, or receipt authority.
    :param allowed_role: Local lifecycle role.
    :param authenticated_issuer: Exact remote or local receipt issuer.
    :returns: Complete runtime producer specification.
    """

    return NativeTerminalRuntimeProducerSpec(
        registration=NativeTerminalProducerRegistration(
            producer_id=producer_id,
            name=name,
            producer_class=producer_class,
            allowed_role=allowed_role,
            authenticated_issuer=authenticated_issuer,
        ),
        delivery=NativeTerminalProducerDelivery.PYTHON,
    )


def build_terminal_startup_python_producer_plan(
    matrix: TerminalStartupCohortMatrix,
    *,
    local_service_id: str,
    local_tensor_parallel_rank: int,
    first_producer_id: int,
) -> TerminalStartupPythonProducerPlan:
    """Freeze every Python producer before native runtime construction.

    Cross-role peers receive exact CONTROL and RECEIPT namespaces. Same-role
    peers receive RECEIPT only when they belong to the same TP service, which
    covers source publication fan-out and decode request coordination without
    trusting unrelated decoder replicas.

    :param matrix: Complete observed deployment-epoch rank matrix.
    :param local_service_id: Local static service identifier.
    :param local_tensor_parallel_rank: Local rank within that service.
    :param first_producer_id: First available nonnegative producer ID.
    :returns: Deterministic least-privilege Python producer plan.
    """

    if type(matrix) is not TerminalStartupCohortMatrix:
        raise TypeError("matrix must be TerminalStartupCohortMatrix")
    if type(first_producer_id) is not int or first_producer_id < 0:
        raise ValueError("first_producer_id must be nonnegative")
    local = matrix.rank(local_service_id, local_tensor_parallel_rank)
    allowed_role = _native_role(local)
    local_identity = NativeTerminalProcessIdentity.from_identity(
        local.terminal_identity
    )
    next_id = first_producer_id
    specs: list[NativeTerminalRuntimeProducerSpec] = []
    specs.append(
        _producer_spec(
            producer_id=next_id,
            name="terminal-startup-local",
            producer_class=NativeTerminalProducerClass.LOCAL,
            allowed_role=allowed_role,
            authenticated_issuer=None,
        )
    )
    fatal_producer_id = next_id
    next_id += 1
    specs.append(
        _producer_spec(
            producer_id=next_id,
            name="terminal-startup-local-receipt",
            producer_class=NativeTerminalProducerClass.RECEIPT,
            allowed_role=allowed_role,
            authenticated_issuer=local_identity,
        )
    )
    next_id += 1

    for remote in matrix.ranks:
        if remote.key == local.key:
            continue
        cross_role = remote.role is not local.role
        same_service_peer = remote.service_id == local.service_id
        if not cross_role and not same_service_peer:
            continue
        remote_identity = NativeTerminalProcessIdentity.from_identity(
            remote.terminal_identity
        )
        producer_prefix = (
            f"terminal-startup-{remote.role.value}-{remote.service_id}"
            f"-rank-{remote.tensor_parallel_rank}"
        )
        if cross_role:
            specs.append(
                _producer_spec(
                    producer_id=next_id,
                    name=f"{producer_prefix}-control",
                    producer_class=NativeTerminalProducerClass.CONTROL,
                    allowed_role=allowed_role,
                    authenticated_issuer=remote_identity,
                )
            )
            next_id += 1
        specs.append(
            _producer_spec(
                producer_id=next_id,
                name=f"{producer_prefix}-receipt",
                producer_class=NativeTerminalProducerClass.RECEIPT,
                allowed_role=allowed_role,
                authenticated_issuer=remote_identity,
            )
        )
        next_id += 1

    return TerminalStartupPythonProducerPlan(
        specs=tuple(specs),
        fatal_producer_id=fatal_producer_id,
        next_producer_id=next_id,
    )
