import datetime
import multiprocessing as mp
import os
import queue
import tempfile
import time
import unittest
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import cast

import torch

from sglang.srt.disaggregation.hicache_generation import (
    HiCacheGenerationCoordinator,
    HiCacheGenerationOutcome,
    HiCacheGenerationPlan,
    HiCacheGenerationState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _PreparedReceipt:
    """Fake immutable preparation receipt.

    :ivar local_work: Whether this rank represents queued cache work.
    """

    local_work: bool


@dataclass(frozen=True)
class _StartedReceipt:
    """Fake immutable asynchronous-start receipt.

    :ivar local_work: Whether this rank represents queued cache work.
    """

    local_work: bool


@dataclass(frozen=True)
class _PublishedReceipt:
    """Fake immutable publication receipt.

    :ivar local_work: Whether this rank represents published cache work.
    """

    local_work: bool


_RankResult = tuple[str, str, int, tuple[str, ...], bool]


class _GenerationHooks:
    """Fault-injectable rank-local lifecycle used with a real Gloo group."""

    def __init__(self, rank: int, scenario: str) -> None:
        """Initialize fake lifecycle state.

        :param rank: Process-group rank.
        :param scenario: Fault or asymmetry scenario name.
        """

        self.rank = rank
        self.scenario = scenario
        self.trace: list[str] = []
        self.readiness_queries = 0

    def plan(self, generation_id: int) -> HiCacheGenerationPlan:
        """Build the scenario's local plan.

        :param generation_id: Generation being planned.
        :returns: Rank-local plan.
        """

        self.trace.append("plan")
        if self.scenario == "plan_failure" and self.rank == 1:
            raise RuntimeError("injected plan failure")

        if self.scenario == "queued_noop":
            local_work = self.rank == 0
        elif self.scenario == "no_restore_queued":
            local_work = self.rank == 1
        elif self.scenario == "no_work":
            local_work = False
        else:
            local_work = True

        request_ids = ("request-a",)
        if self.scenario == "identity_mismatch" and self.rank == 1:
            request_ids = ("request-b",)
        if self.scenario == "plan_generation_mismatch" and self.rank == 1:
            generation_id += 1
        return HiCacheGenerationPlan(
            generation_id=generation_id,
            request_ids=request_ids,
            local_work=local_work,
        )

    def prepare(self, plan: HiCacheGenerationPlan) -> _PreparedReceipt:
        """Create a reversible fake preparation receipt.

        :param plan: Agreed local plan.
        :returns: Fake preparation receipt.
        """

        self.trace.append("prepare")
        if self.scenario == "preparation_failure" and self.rank == 1:
            raise RuntimeError("injected preparation failure")
        if self.scenario == "preparation_none" and self.rank == 1:
            return cast(_PreparedReceipt, None)
        return _PreparedReceipt(local_work=plan.local_work)

    def start(self, prepared: _PreparedReceipt) -> _StartedReceipt:
        """Start queued work or the rank's explicit no-op event.

        :param prepared: Fake preparation receipt.
        :returns: Fake start receipt.
        """

        self.trace.append("start")
        if self.scenario == "start_failure" and self.rank == 1:
            raise RuntimeError("injected start failure")
        if self.scenario == "start_none" and self.rank == 1:
            return cast(_StartedReceipt, None)
        return _StartedReceipt(local_work=prepared.local_work)

    def is_ready(self, started: _StartedReceipt) -> bool:
        """Return scenario-controlled local readiness.

        :param started: Fake start receipt.
        :returns: Whether the fake operation is locally complete.
        """

        self.trace.append("ready")
        self.readiness_queries += 1
        if (
            self.scenario in ("ready_delayed", "cancel_started")
            and self.rank == 1
            and self.readiness_queries == 1
        ):
            return False
        if self.scenario == "readiness_failure" and self.rank == 1:
            raise RuntimeError("injected readiness failure")
        return True

    def publish(
        self,
        prepared: _PreparedReceipt,
        started: _StartedReceipt,
    ) -> _PublishedReceipt:
        """Create a reversible fake publication receipt.

        :param prepared: Fake preparation receipt.
        :param started: Fake start receipt.
        :returns: Fake publication receipt.
        """

        self.trace.append("publish")
        if self.scenario == "publication_failure" and self.rank == 1:
            raise RuntimeError("injected publication failure")
        if self.scenario == "publication_none" and self.rank == 1:
            return cast(_PublishedReceipt, None)
        if prepared.local_work != started.local_work:
            raise AssertionError("start receipt changed local-work ownership")
        return _PublishedReceipt(local_work=prepared.local_work)

    def seal(
        self,
        prepared: _PreparedReceipt,
        started: _StartedReceipt,
        published: _PublishedReceipt,
    ) -> None:
        """Seal the fake publication.

        :param prepared: Fake preparation receipt.
        :param started: Fake start receipt.
        :param published: Fake publication receipt.
        """

        if not (prepared.local_work == started.local_work == published.local_work):
            raise AssertionError("publication receipts disagree about ownership")
        self.trace.append("seal")

    def rollback(
        self,
        state: HiCacheGenerationState[
            HiCacheGenerationPlan,
            _PreparedReceipt,
            _StartedReceipt,
            _PublishedReceipt,
        ],
    ) -> None:
        """Record the exact receipt state presented for cleanup.

        :param state: Complete fake ownership record.
        """

        self.trace.append(
            "rollback:"
            f"prepared={int(state.prepared is not None)}:"
            f"started={int(state.started is not None)}:"
            f"published={int(state.published is not None)}"
        )


def _run_generation_rank(
    rank: int,
    world_size: int,
    store_path: str,
    scenarios: tuple[str, ...],
    result_queue: Queue,
) -> None:
    """Run all fault scenarios in one real Gloo rank.

    :param rank: Process-group rank.
    :param world_size: Number of ranks in the process group.
    :param store_path: File-store rendezvous path.
    :param scenarios: Ordered fault or asymmetry scenario names.
    :param result_queue: Parent result queue.
    """

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        for generation_id, scenario in enumerate(scenarios, start=7):
            hooks = _GenerationHooks(rank, scenario)
            coordinator = HiCacheGenerationCoordinator(
                generation_id=generation_id,
                group=torch.distributed.group.WORLD,
                hooks=hooks,
            )

            outcome = HiCacheGenerationOutcome.PENDING
            iterations = 0
            while outcome == HiCacheGenerationOutcome.PENDING and iterations < 8:
                outcome = coordinator.advance()
                iterations += 1
                if scenario == "cancel_started" and iterations == 1 and rank == 1:
                    coordinator.request_cancel("injected cleanup while started")

            if outcome == HiCacheGenerationOutcome.PENDING:
                raise AssertionError(
                    f"generation did not terminate in scenario {scenario}"
                )

            terminal_trace = tuple(hooks.trace)
            repeated_outcome = coordinator.advance()
            if tuple(hooks.trace) != terminal_trace:
                raise AssertionError(
                    f"terminal generation repeated work in scenario {scenario}"
                )
            if outcome == HiCacheGenerationOutcome.NO_WORK:
                if repeated_outcome != HiCacheGenerationOutcome.SEALED:
                    raise AssertionError(
                        f"no-work generation changed in scenario {scenario}"
                    )
            elif repeated_outcome != outcome:
                raise AssertionError(f"terminal outcome changed in scenario {scenario}")

            result_queue.put(
                (
                    scenario,
                    rank,
                    outcome.value,
                    coordinator.state.phase.value,
                    coordinator.state.vote_count,
                    terminal_trace,
                    coordinator.state.local_error is not None,
                )
            )
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _run_two_rank_scenarios(
    scenarios: tuple[str, ...],
) -> dict[str, dict[int, _RankResult]]:
    """Run all scenarios in one spawned two-rank process group.

    :param scenarios: Ordered fault or asymmetry scenario names.
    :returns: Results keyed by scenario and rank.
    """

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix="sglang-hicache-gloo-") as temp_dir:
        store_path = os.path.join(temp_dir, "store")
        processes = [
            context.Process(
                target=_run_generation_rank,
                args=(rank, 2, store_path, scenarios, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 40.0
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))

        live_processes = [process for process in processes if process.is_alive()]
        for process in live_processes:
            process.terminate()
        for process in live_processes:
            process.join(5.0)
        if len(live_processes) > 0:
            raise AssertionError(
                "ordered-collective deadlock in generation suite: "
                f"pids={[process.pid for process in live_processes]}"
            )

        failed = [
            (process.pid, process.exitcode)
            for process in processes
            if process.exitcode != 0
        ]
        if len(failed) > 0:
            raise AssertionError(f"generation workers failed: {failed}")

        results: dict[str, dict[int, _RankResult]] = {
            scenario: {} for scenario in scenarios
        }
        expected_result_count = len(scenarios) * len(processes)
        for _ in range(expected_result_count):
            try:
                result = result_queue.get(timeout=2.0)
            except queue.Empty as error:
                raise AssertionError("missing generation worker result") from error

            scenario = str(result[0])
            rank = int(result[1])
            scenario_results = results.get(scenario)
            if scenario_results is None:
                raise AssertionError(f"worker returned unknown scenario {scenario}")
            if rank in scenario_results:
                raise AssertionError(
                    f"worker returned duplicate rank {rank} for {scenario}"
                )
            scenario_results[rank] = (
                str(result[2]),
                str(result[3]),
                int(result[4]),
                tuple(str(event) for event in result[5]),
                bool(result[6]),
            )

        result_queue.close()
        result_queue.join_thread()

    return results


class TestDecodeHiCacheGenerationCoordinator(unittest.TestCase):
    """Tests real two-rank collective ordering and receipt cleanup."""

    def test_two_rank_phase_order_and_fault_boundaries(self) -> None:
        """Every asymmetry reaches the same terminal phase without deadlock."""

        expectations = {
            "queued_noop": (HiCacheGenerationOutcome.SEALED, 4),
            "no_restore_queued": (HiCacheGenerationOutcome.SEALED, 4),
            "no_work": (HiCacheGenerationOutcome.NO_WORK, 1),
            "plan_failure": (HiCacheGenerationOutcome.ROLLED_BACK, 1),
            "plan_generation_mismatch": (
                HiCacheGenerationOutcome.ROLLED_BACK,
                1,
            ),
            "identity_mismatch": (HiCacheGenerationOutcome.ROLLED_BACK, 1),
            "preparation_failure": (HiCacheGenerationOutcome.ROLLED_BACK, 2),
            "preparation_none": (HiCacheGenerationOutcome.ROLLED_BACK, 2),
            "start_failure": (HiCacheGenerationOutcome.ROLLED_BACK, 3),
            "start_none": (HiCacheGenerationOutcome.ROLLED_BACK, 3),
            "readiness_failure": (HiCacheGenerationOutcome.ROLLED_BACK, 3),
            "ready_delayed": (HiCacheGenerationOutcome.SEALED, 5),
            "cancel_started": (HiCacheGenerationOutcome.ROLLED_BACK, 4),
            "publication_failure": (HiCacheGenerationOutcome.ROLLED_BACK, 4),
            "publication_none": (HiCacheGenerationOutcome.ROLLED_BACK, 4),
        }
        rank_one_error_scenarios = {
            "plan_failure",
            "plan_generation_mismatch",
            "preparation_failure",
            "preparation_none",
            "start_failure",
            "start_none",
            "readiness_failure",
            "cancel_started",
            "publication_failure",
            "publication_none",
        }
        results_by_scenario = _run_two_rank_scenarios(tuple(expectations))
        self.assertEqual(set(results_by_scenario), set(expectations))

        for scenario, (expected_outcome, expected_votes) in expectations.items():
            with self.subTest(scenario=scenario):
                results = results_by_scenario[scenario]
                self.assertEqual(set(results), {0, 1})
                for rank, result in results.items():
                    outcome, phase, vote_count, trace, has_error = result
                    self.assertEqual(outcome, expected_outcome.value)
                    self.assertEqual(vote_count, expected_votes)
                    self.assertEqual(
                        has_error,
                        rank == 1 and scenario in rank_one_error_scenarios,
                    )
                    if expected_outcome == HiCacheGenerationOutcome.ROLLED_BACK:
                        self.assertEqual(phase, "rolled_back")
                        self.assertTrue(str(trace[-1]).startswith("rollback:"))
                    else:
                        self.assertEqual(phase, "sealed")

                if scenario in (
                    "plan_failure",
                    "plan_generation_mismatch",
                    "identity_mismatch",
                ):
                    for result in results.values():
                        self.assertIn(
                            "rollback:prepared=0:started=0:published=0",
                            result[3],
                        )
                if scenario in ("preparation_failure", "preparation_none"):
                    self.assertIn(
                        "rollback:prepared=1:started=0:published=0",
                        results[0][3],
                    )
                    self.assertIn(
                        "rollback:prepared=0:started=0:published=0",
                        results[1][3],
                    )
                if scenario in ("start_failure", "start_none"):
                    self.assertIn(
                        "rollback:prepared=1:started=1:published=0",
                        results[0][3],
                    )
                    self.assertIn(
                        "rollback:prepared=1:started=0:published=0",
                        results[1][3],
                    )
                if scenario in ("readiness_failure", "cancel_started"):
                    for result in results.values():
                        self.assertIn(
                            "rollback:prepared=1:started=1:published=0",
                            result[3],
                        )
                if scenario in ("publication_failure", "publication_none"):
                    self.assertIn(
                        "rollback:prepared=1:started=1:published=1",
                        results[0][3],
                    )
                    self.assertIn(
                        "rollback:prepared=1:started=1:published=0",
                        results[1][3],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
