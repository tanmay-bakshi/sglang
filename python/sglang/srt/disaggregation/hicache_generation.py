"""Ordered tensor-parallel generations for decode-local HiCache work."""

import hashlib
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

import torch


class HiCacheGenerationPhase(str, Enum):
    """Lifecycle phase shared by every rank in one HiCache generation."""

    PLAN = "plan"
    PREPARE = "prepare"
    START = "start"
    WAIT_READY = "wait_ready"
    PUBLISH = "publish"
    SEALED = "sealed"
    ROLLED_BACK = "rolled_back"


class HiCacheGenerationOutcome(str, Enum):
    """Externally visible result of advancing one generation."""

    PENDING = "pending"
    SEALED = "sealed"
    NO_WORK = "no_work"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class HiCacheGenerationPlan:
    """Pure rank-local plan participating in TP identity agreement.

    :ivar generation_id: Monotonic generation number.
    :ivar request_ids: Stable request order represented by the generation.
    :ivar local_work: Whether this rank has data movement or publication work.
    """

    generation_id: int
    request_ids: tuple[str, ...]
    local_work: bool


PlanT = TypeVar("PlanT", bound=HiCacheGenerationPlan)
PreparedT = TypeVar("PreparedT")
StartedT = TypeVar("StartedT")
PublishedT = TypeVar("PublishedT")


@dataclass
class HiCacheGenerationState(Generic[PlanT, PreparedT, StartedT, PublishedT]):
    """Mutable ownership record for one ordered TP generation.

    :ivar generation_id: Monotonic generation number.
    :ivar phase: Current lifecycle phase.
    :ivar plan: Successful local plan, if planning completed.
    :ivar prepared: Reversible local preparation receipt.
    :ivar started: Local asynchronous-start receipt.
    :ivar published: Reversible local publication receipt.
    :ivar local_error: Retained traceback from the first local failure.
    :ivar cancel_requested: Whether local cleanup requested rollback.
    :ivar vote_count: Number of ordered TP votes entered by this rank.
    """

    generation_id: int
    phase: HiCacheGenerationPhase = HiCacheGenerationPhase.PLAN
    plan: PlanT | None = None
    prepared: PreparedT | None = None
    started: StartedT | None = None
    published: PublishedT | None = None
    local_error: str | None = None
    cancel_requested: bool = False
    vote_count: int = 0


class HiCacheGenerationHooks(Protocol[PlanT, PreparedT, StartedT, PublishedT]):
    """Rank-local operations driven by :class:`HiCacheGenerationCoordinator`."""

    def plan(self, generation_id: int) -> PlanT:
        """Build a collective-free plan.

        :param generation_id: Generation being planned.
        :returns: Pure local plan.
        """

        ...

    def prepare(self, plan: PlanT) -> PreparedT:
        """Perform reversible local preparation.

        :param plan: Agreed local plan.
        :returns: Immutable ownership receipt for prepared resources.
        """

        ...

    def start(self, prepared: PreparedT) -> StartedT:
        """Start one local event, including an explicit no-work event.

        This method must either return a drainable receipt or fail before any
        asynchronous operation starts.

        :param prepared: Prepared local resources.
        :returns: Local asynchronous-start receipt.
        """

        ...

    def is_ready(self, started: StartedT) -> bool:
        """Query local completion without entering a collective.

        :param started: Local asynchronous-start receipt.
        :returns: Whether local work is safe to publish or roll back.
        """

        ...

    def publish(self, prepared: PreparedT, started: StartedT) -> PublishedT:
        """Publish local state reversibly and acquire request ownership.

        :param prepared: Prepared local resources.
        :param started: Completed local start receipt.
        :returns: Reversible publication receipt.
        """

        ...

    def seal(
        self,
        prepared: PreparedT,
        started: StartedT,
        published: PublishedT,
    ) -> None:
        """Irreversibly seal a globally accepted publication.

        This method is a noexcept ownership transfer. All fallible validation
        belongs in :meth:`publish` before the publication vote.

        :param prepared: Prepared local resources.
        :param started: Completed local start receipt.
        :param published: Accepted publication receipt.
        """

        ...

    def rollback(
        self,
        state: HiCacheGenerationState[PlanT, PreparedT, StartedT, PublishedT],
    ) -> None:
        """Release every locally owned resource without collectives.

        This method is noexcept and idempotent.

        :param state: Complete local ownership record.
        """

        ...


class HiCacheGenerationCoordinator(Generic[PlanT, PreparedT, StartedT, PublishedT]):
    """Drive one fixed-order TP generation through commit or rollback."""

    def __init__(
        self,
        generation_id: int,
        group: torch.distributed.ProcessGroup | None,
        hooks: HiCacheGenerationHooks[PlanT, PreparedT, StartedT, PublishedT],
    ) -> None:
        """Initialize one generation.

        :param generation_id: Monotonic generation number.
        :param group: CPU-capable TP process group, or ``None`` for TP1.
        :param hooks: Rank-local lifecycle implementation.
        """

        self.group = group
        self.hooks = hooks
        self.state = HiCacheGenerationState[PlanT, PreparedT, StartedT, PublishedT](
            generation_id=generation_id
        )

    def request_cancel(self, reason: str) -> None:
        """Request rollback at the next ordered vote.

        :param reason: Diagnostic retained with the local generation state.
        """

        if self.state.phase in (
            HiCacheGenerationPhase.SEALED,
            HiCacheGenerationPhase.ROLLED_BACK,
        ):
            return
        self.state.cancel_requested = True
        if self.state.local_error is None:
            self.state.local_error = reason

    def advance(self) -> HiCacheGenerationOutcome:
        """Advance through non-blocking phases until completion or DMA wait.

        :returns: Current externally visible generation outcome.
        """

        while True:
            if self.state.phase == HiCacheGenerationPhase.PLAN:
                outcome = self._advance_plan()
                if outcome is not None:
                    return outcome
                continue

            if self.state.phase == HiCacheGenerationPhase.PREPARE:
                outcome = self._advance_prepare()
                if outcome is not None:
                    return outcome
                continue

            if self.state.phase == HiCacheGenerationPhase.START:
                self._advance_start()
                continue

            if self.state.phase == HiCacheGenerationPhase.WAIT_READY:
                outcome = self._advance_readiness()
                if outcome is not None:
                    return outcome
                continue

            if self.state.phase == HiCacheGenerationPhase.PUBLISH:
                return self._advance_publication()

            if self.state.phase == HiCacheGenerationPhase.SEALED:
                return HiCacheGenerationOutcome.SEALED

            if self.state.phase == HiCacheGenerationPhase.ROLLED_BACK:
                return HiCacheGenerationOutcome.ROLLED_BACK

            raise AssertionError(f"Unsupported HiCache phase {self.state.phase}")

    def _advance_plan(self) -> HiCacheGenerationOutcome | None:
        local_success = not self.state.cancel_requested
        plan: PlanT | None = None
        if local_success:
            try:
                plan = self.hooks.plan(self.state.generation_id)
                if plan.generation_id != self.state.generation_id:
                    raise ValueError(
                        "HiCache plan generation does not match its coordinator"
                    )
            except Exception:
                self._retain_current_exception()
                local_success = False

        agreed, any_work = self._agree_plan(local_success, plan)
        if not agreed:
            self._rollback()
            return HiCacheGenerationOutcome.ROLLED_BACK

        assert plan is not None
        self.state.plan = plan
        if not any_work:
            self.state.phase = HiCacheGenerationPhase.SEALED
            return HiCacheGenerationOutcome.NO_WORK

        self.state.phase = HiCacheGenerationPhase.PREPARE
        return None

    def _advance_prepare(self) -> HiCacheGenerationOutcome | None:
        plan = self.state.plan
        assert plan is not None
        local_success = not self.state.cancel_requested
        if local_success:
            try:
                prepared = self.hooks.prepare(plan)
                if prepared is None:
                    raise RuntimeError(
                        "Successful HiCache preparation returned no receipt"
                    )
                self.state.prepared = prepared
            except Exception:
                self._retain_current_exception()
                local_success = False

        if not self._agree_success(local_success):
            self._rollback()
            return HiCacheGenerationOutcome.ROLLED_BACK

        assert self.state.prepared is not None
        self.state.phase = HiCacheGenerationPhase.START
        return None

    def _advance_start(self) -> None:
        prepared = self.state.prepared
        assert prepared is not None
        if not self.state.cancel_requested:
            try:
                started = self.hooks.start(prepared)
                if started is None:
                    raise RuntimeError("Successful HiCache start returned no receipt")
                self.state.started = started
            except Exception:
                self._retain_current_exception()
        self.state.phase = HiCacheGenerationPhase.WAIT_READY

    def _advance_readiness(self) -> HiCacheGenerationOutcome | None:
        local_healthy = (
            self.state.local_error is None and not self.state.cancel_requested
        )
        local_ready = True
        if self.state.started is not None:
            try:
                local_ready = self.hooks.is_ready(self.state.started)
            except Exception:
                self._retain_current_exception()
                local_healthy = False
                local_ready = True

        globally_ready, globally_healthy = self._agree_readiness(
            local_ready, local_healthy
        )
        if not globally_ready:
            return HiCacheGenerationOutcome.PENDING
        if not globally_healthy:
            self._rollback()
            return HiCacheGenerationOutcome.ROLLED_BACK

        if self.state.started is None:
            raise RuntimeError("Healthy HiCache generation has no start receipt")
        self.state.phase = HiCacheGenerationPhase.PUBLISH
        return None

    def _advance_publication(self) -> HiCacheGenerationOutcome:
        prepared = self.state.prepared
        started = self.state.started
        assert prepared is not None
        assert started is not None

        local_success = not self.state.cancel_requested
        if local_success:
            try:
                published = self.hooks.publish(prepared, started)
                if published is None:
                    raise RuntimeError(
                        "Successful HiCache publication returned no receipt"
                    )
                self.state.published = published
            except Exception:
                self._retain_current_exception()
                local_success = False

        if not self._agree_success(local_success):
            self._rollback()
            return HiCacheGenerationOutcome.ROLLED_BACK

        published = self.state.published
        assert published is not None
        self.hooks.seal(prepared, started, published)
        self.state.phase = HiCacheGenerationPhase.SEALED
        return HiCacheGenerationOutcome.SEALED

    def _agree_plan(self, local_success: bool, plan: PlanT | None) -> tuple[bool, bool]:
        effective_plan = (
            plan
            if plan is not None
            else HiCacheGenerationPlan(
                generation_id=self.state.generation_id,
                request_ids=(),
                local_work=False,
            )
        )
        hash_high, hash_low = self._request_fingerprint(effective_plan.request_ids)
        packed = torch.tensor(
            [
                int(local_success),
                -int(effective_plan.local_work),
                effective_plan.generation_id,
                -effective_plan.generation_id,
                len(effective_plan.request_ids),
                -len(effective_plan.request_ids),
                hash_high,
                -hash_high,
                hash_low,
                -hash_low,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._reduce_min(packed)
        identity_agreed = all(
            int(packed[positive].item()) == -int(packed[negative].item())
            for positive, negative in ((2, 3), (4, 5), (6, 7), (8, 9))
        )
        return bool(packed[0].item()) and identity_agreed, packed[1].item() < 0

    def _agree_success(self, local_success: bool) -> bool:
        packed = torch.tensor(int(local_success), dtype=torch.int32, device="cpu")
        self._reduce_min(packed)
        return bool(packed.item())

    def _agree_readiness(
        self, local_ready: bool, local_healthy: bool
    ) -> tuple[bool, bool]:
        packed = torch.tensor(
            [int(local_ready), int(local_healthy)],
            dtype=torch.int32,
            device="cpu",
        )
        self._reduce_min(packed)
        return bool(packed[0].item()), bool(packed[1].item())

    def _reduce_min(self, value: torch.Tensor) -> None:
        self.state.vote_count += 1
        if not torch.distributed.is_initialized():
            return
        if self.group is None:
            return
        if torch.distributed.get_world_size(group=self.group) <= 1:
            return
        torch.distributed.all_reduce(
            value,
            op=torch.distributed.ReduceOp.MIN,
            group=self.group,
        )

    def _rollback(self) -> None:
        if self.state.phase == HiCacheGenerationPhase.ROLLED_BACK:
            return
        self.hooks.rollback(self.state)
        self.state.phase = HiCacheGenerationPhase.ROLLED_BACK

    def _retain_current_exception(self) -> None:
        if self.state.local_error is None:
            self.state.local_error = traceback.format_exc()

    @staticmethod
    def _request_fingerprint(request_ids: tuple[str, ...]) -> tuple[int, int]:
        digest = hashlib.blake2b(digest_size=16)
        for request_id in request_ids:
            encoded = request_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
            digest.update(encoded)
        raw = digest.digest()
        mask = (1 << 63) - 1
        return (
            int.from_bytes(raw[:8], byteorder="little", signed=False) & mask,
            int.from_bytes(raw[8:], byteorder="little", signed=False) & mask,
        )
