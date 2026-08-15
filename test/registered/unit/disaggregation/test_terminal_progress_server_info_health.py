from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.managers.scheduler import Scheduler


class _HealthManager:
    """Expose one immutable rank-local terminal health payload."""

    _health: dict[str, object] | None

    def __init__(self, health: dict[str, object] | None) -> None:
        """Create one manager projection.

        :param health: Rank-local terminal lifecycle health.
        """

        self._health = health

    def packed_terminal_health(self) -> dict[str, object] | None:
        """Return the configured local health payload.

        :returns: Rank-local health, if activated.
        """

        return self._health


class _GatherGroup:
    """Record one control-plane all-gather and return a frozen population."""

    _population: list[object]
    observed: list[object]

    def __init__(self, population: list[object]) -> None:
        """Create one deterministic TP CPU-group fixture.

        :param population: Objects returned in TP-rank order.
        """

        self._population = population
        self.observed = []

    def all_gather_object(self, value: object) -> list[object]:
        """Record the local entry and return every rank contribution.

        :param value: Local scheduler contribution.
        :returns: Frozen rank-ordered population.
        """

        self.observed.append(value)
        return self._population


def _scheduler(
    *,
    tp_rank: int,
    tp_size: int,
    local_health: dict[str, object] | None,
    gathered: list[object] | None = None,
) -> SimpleNamespace:
    """Build the scheduler fields required by the health projection.

    :param tp_rank: Local TP rank.
    :param tp_size: Local service TP width.
    :param local_health: Local manager health.
    :param gathered: Optional deterministic all-rank population.
    :returns: Scheduler-shaped fixture.
    """

    manager = _HealthManager(local_health)
    group = _GatherGroup([] if gathered is None else gathered)
    return SimpleNamespace(
        server_args=SimpleNamespace(pd_terminal_local_membership="membership"),
        ps=SimpleNamespace(tp_rank=tp_rank, tp_size=tp_size),
        tp_group=group,
        terminal_nixl_manager=MagicMock(return_value=manager),
    )


def test_terminal_health_tp1_returns_the_local_owner_inventory() -> None:
    """TP1 exposes the same rank-tagged schema without a collective."""

    health = {"role": "decode"}
    scheduler = _scheduler(tp_rank=0, tp_size=1, local_health=health)

    result = Scheduler.packed_terminal_health_by_tp_rank(scheduler)

    assert result == [{"tp_rank": 0, "packed_terminal_health": health}]
    assert scheduler.tp_group.observed == []


def test_terminal_health_tp2_gathers_every_rank_in_order() -> None:
    """The control plane returns one identity-correlatable entry per TP rank."""

    rank_zero_health: dict[str, object] = {"role": "source"}
    rank_one_health: dict[str, object] = {"role": "source"}
    rank_zero = {"tp_rank": 0, "packed_terminal_health": rank_zero_health}
    rank_one = {"tp_rank": 1, "packed_terminal_health": rank_one_health}
    gathered_rank_zero = {**rank_zero, "error": None}
    gathered_rank_one = {**rank_one, "error": None}
    scheduler = _scheduler(
        tp_rank=1,
        tp_size=2,
        local_health=rank_one_health,
        gathered=[gathered_rank_zero, gathered_rank_one],
    )

    result = Scheduler.packed_terminal_health_by_tp_rank(scheduler)

    assert result == [rank_zero, rank_one]
    assert scheduler.tp_group.observed == [gathered_rank_one]


@pytest.mark.parametrize(
    "population,error",
    [
        (
            [
                {
                    "tp_rank": 0,
                    "packed_terminal_health": {"role": "source"},
                    "error": None,
                }
            ],
            "incomplete",
        ),
        (
            [
                {
                    "tp_rank": 1,
                    "packed_terminal_health": {"role": "source"},
                    "error": None,
                },
                {
                    "tp_rank": 0,
                    "packed_terminal_health": {"role": "source"},
                    "error": None,
                },
            ],
            "order differs",
        ),
        (
            [
                {
                    "tp_rank": 0,
                    "packed_terminal_health": {"role": "source"},
                    "error": None,
                },
                {
                    "tp_rank": 1,
                    "packed_terminal_health": None,
                    "error": None,
                },
            ],
            "unavailable",
        ),
    ],
)
def test_terminal_health_fails_closed_on_incomplete_rank_evidence(
    population: list[object],
    error: str,
) -> None:
    """Missing, reordered, or uninitialized rank evidence cannot be reported.

    :param population: Candidate TP health population.
    :param error: Expected fail-closed diagnostic fragment.
    """

    scheduler = _scheduler(
        tp_rank=0,
        tp_size=2,
        local_health={"role": "source"},
        gathered=population,
    )

    with pytest.raises(RuntimeError, match=error):
        Scheduler.packed_terminal_health_by_tp_rank(scheduler)


def test_nonterminal_server_info_avoids_health_collectives() -> None:
    """Ordinary serving deployments preserve the existing control path."""

    manager = MagicMock()
    group = MagicMock()
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pd_terminal_local_membership=None),
        ps=SimpleNamespace(tp_rank=0, tp_size=2),
        tp_group=group,
        terminal_nixl_manager=manager,
    )

    assert Scheduler.packed_terminal_health_by_tp_rank(scheduler) is None
    manager.assert_not_called()
    group.all_gather_object.assert_not_called()


def test_rank_local_health_failure_still_enters_the_tp_collective() -> None:
    """A failed local snapshot cannot strand peer ranks inside all-gather."""

    manager = MagicMock(side_effect=RuntimeError("synthetic health failure"))
    peer = {
        "tp_rank": 1,
        "packed_terminal_health": {"role": "source"},
        "error": None,
    }

    def gather_with_peer(local: object) -> list[object]:
        """Return the failed local contribution beside one healthy peer.

        :param local: Local rank contribution.
        :returns: Complete TP2 population.
        """

        return [local, peer]

    group = MagicMock()
    group.all_gather_object.side_effect = gather_with_peer
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pd_terminal_local_membership="membership"),
        ps=SimpleNamespace(tp_rank=0, tp_size=2),
        tp_group=group,
        terminal_nixl_manager=manager,
    )

    with pytest.raises(RuntimeError, match="failed on one TP rank"):
        Scheduler.packed_terminal_health_by_tp_rank(scheduler)

    group.all_gather_object.assert_called_once()
    local = group.all_gather_object.call_args.args[0]
    assert local["tp_rank"] == 0
    assert local["packed_terminal_health"] is None
    assert "synthetic health failure" in local["error"]


def _decode_manager_with_rows(
    *,
    free_count: int,
    active_count: int,
    quarantined_count: int,
) -> NixlKVManager:
    """Build a decode manager with one exact row-pool inventory.

    :param free_count: Reusable registered row count.
    :param active_count: Live request-owned row count.
    :param quarantined_count: Process-lifetime retained row count.
    :returns: Manager-shaped owner with complete health collaborators.
    """

    pool = MagicMock()
    pool.row_capacity = free_count + active_count + quarantined_count
    pool.inventory.return_value = (
        free_count,
        active_count,
        quarantined_count,
    )
    inventory = SimpleNamespace(
        retained_resource_count=1,
        active_binding_digests=(b"a" * 32,),
        active_coordinator_manifest_digests=(),
        actor=SimpleNamespace(
            active_bindings=(b"a" * 32,),
            quarantined_bindings=(),
        ),
        scheduler_consumer=SimpleNamespace(
            active_binding_digests=(b"a" * 32,),
            quarantined_binding_digests=(),
        ),
        scheduler_serving=SimpleNamespace(
            inbox=SimpleNamespace(live_count=0),
        ),
        runtime=SimpleNamespace(
            quarantined_binding_digests=(),
            owner=SimpleNamespace(
                pending_action_count=0,
                active_source_count=0,
                active_decode_count=1,
                quarantined_count=0,
                queued_input_count=0,
                queued_output_count=0,
                queued_fatal_output_count=0,
                armed_deadline_count=0,
            ),
            scheduler=SimpleNamespace(queued_count=0),
            coordinator=SimpleNamespace(queued_count=0),
            lifecycle=SimpleNamespace(queued_count=0),
            source_work=SimpleNamespace(queued_count=0),
            decode_work=SimpleNamespace(queued_count=0),
            publisher=SimpleNamespace(queued_count=0),
            scheduler_live_count=1,
            consumer_pending_count=0,
            scheduler_pending_count=0,
            fatal_reason=None,
            output_reactor_alive=True,
        ),
        owner_dead_marked=False,
    )
    manager = object.__new__(NixlKVManager)
    manager._terminal_source_serving = None
    manager._terminal_decode_serving = SimpleNamespace(
        inventory=lambda: inventory,
    )
    manager._terminal_dflash_boundary_pool = pool
    return manager


def _source_manager_with_grouped_quarantine(
    *,
    native_fatal: int = 0,
    eventfd_error: int = 0,
) -> NixlKVManager:
    """Build source health with grouped quarantine outside wiring ownership.

    :param native_fatal: Sticky grouped-channel fatal flags.
    :param eventfd_error: Grouped-channel eventfd errno.
    :returns: Manager-shaped source owner with conservative retained evidence.
    """

    digest = b"s" * 32
    inventory = SimpleNamespace(
        retained_resource_count=7,
        wiring=SimpleNamespace(
            active_binding_digests=(),
            quarantined_binding_digests=(),
            completion_required_binding_digests=(),
            active_result_slot_binding_digests=(),
            quarantined_result_slot_binding_digests=(),
        ),
        scheduler_consumer=SimpleNamespace(
            active_binding_digests=(),
            quarantined_binding_digests=(),
        ),
        grouped_nixl=SimpleNamespace(
            native=SimpleNamespace(
                fatal=native_fatal,
                eventfd_error=eventfd_error,
            ),
            quarantined_transfer_count=1,
            unowned_handle_count=1,
        ),
        resources=SimpleNamespace(
            actor_active_binding_digests=(digest,),
            actor_quarantined_binding_digests=(),
            request_ready_import_binding_digests=(),
            publication_control_active_binding_digests=(),
            dflash_active_transfer_count=0,
            dflash_quarantined_transfer_count=0,
            dflash_active_row_count=0,
            dflash_quarantined_row_count=0,
            dflash_unowned_native_handle_count=0,
            unpublished_quarantined_binding_digests=(),
            unpublished_quarantined_result_slot_binding_digests=(),
        ),
        runtime=SimpleNamespace(
            owner=SimpleNamespace(pending_action_count=0),
            scheduler_pending_count=0,
            consumer_pending_count=0,
            quarantined_binding_digests=(),
            fatal_reason=None,
            output_reactor_alive=True,
        ),
        owner_dead_marked=False,
    )
    manager = object.__new__(NixlKVManager)
    manager._terminal_source_serving = SimpleNamespace(inventory=lambda: inventory)
    manager._terminal_decode_serving = None
    return manager


def test_source_health_includes_every_grouped_retention_domain() -> None:
    """Grouped quarantine and unowned handles cannot hide behind idle wiring."""

    health = _source_manager_with_grouped_quarantine().packed_terminal_health()

    assert health is not None
    assert health["active_binding_digests"] == [(b"s" * 32).hex()]
    assert health["retained_resource_count"] == 7
    assert health["grouped_nixl_quarantined_transfer_count"] == 1
    assert health["grouped_nixl_unowned_handle_count"] == 1
    assert health["quarantine_count"] == 0


def test_source_health_surfaces_grouped_native_channel_failure() -> None:
    """Native channel fatal state cannot disappear behind a healthy runtime."""

    health = _source_manager_with_grouped_quarantine(
        native_fatal=2,
        eventfd_error=5,
    ).packed_terminal_health()

    assert health is not None
    assert health["fatal_reason"] == "grouped NIXL channel fatal=2 eventfd_error=5"


def test_decode_health_projects_real_dflash_row_inventory() -> None:
    """Decode health never replaces live or quarantined VRAM rows with zeros."""

    manager = _decode_manager_with_rows(
        free_count=5,
        active_count=2,
        quarantined_count=1,
    )

    health = manager.packed_terminal_health()

    assert health is not None
    assert health["dflash_active_row_count"] == 2
    assert health["dflash_quarantined_row_count"] == 1
    assert health["retained_resource_count"] == 4
    assert health["quarantine_count"] == 0


def test_decode_dflash_teardown_requires_terminal_row_dispositions() -> None:
    """Clean and fail-closed teardown distinguish active from quarantine."""

    active = _decode_manager_with_rows(
        free_count=5,
        active_count=1,
        quarantined_count=0,
    )
    with pytest.raises(RuntimeError, match="active DFlash rows"):
        active._require_terminal_decode_dflash_teardown(process_fatal=True)

    retained = _decode_manager_with_rows(
        free_count=5,
        active_count=0,
        quarantined_count=1,
    )
    with pytest.raises(RuntimeError, match="retains DFlash quarantine"):
        retained._require_terminal_decode_dflash_teardown(process_fatal=False)
    retained._require_terminal_decode_dflash_teardown(process_fatal=True)
