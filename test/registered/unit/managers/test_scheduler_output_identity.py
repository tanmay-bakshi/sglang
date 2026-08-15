import dataclasses

import pytest
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.distributed.scheduler_output_identity import SchedulerOutputIdentity
from sglang.srt.managers.scheduler import build_scheduler_parallel_state
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _production_parallel_state(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    dp_rank: int | None = None,
    dp_size: int = 1,
    enable_dp_attention: bool = False,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
) -> ParallelState:
    """Build rank state through the scheduler's production assembly path.

    :param tp_rank: Tensor-parallel process rank.
    :param tp_size: Tensor-parallel width.
    :param dp_rank: Controller rank, routed-rank source, or ``None``.
    :param dp_size: Data-parallel width.
    :param enable_dp_attention: Whether attention partitions the DP domain.
    :param pp_rank: Pipeline-parallel process rank.
    :param pp_size: Pipeline-parallel width.
    :param attn_cp_rank: Attention context-parallel process rank.
    :param attn_cp_size: Attention context-parallel width.
    :returns: Exact process-local scheduler rank state.
    """

    server_args = ServerArgs(
        model_path="dummy",
        tp_size=tp_size,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
        pp_size=pp_size,
        attn_cp_size=attn_cp_size,
    )
    return build_scheduler_parallel_state(
        server_args,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        moe_ep_rank=0,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=0,
        dp_rank=dp_rank,
    )


def _unchecked_parallel_state(**overrides: int | None) -> ParallelState:
    """Build deliberately invalid state for constructor-boundary tests.

    :param overrides: Rank coordinates differing from TP-only rank zero.
    :returns: Unvalidated process-local rank state.
    """

    return dataclasses.replace(
        ParallelState.trivial(dp_rank=None),
        **overrides,
    )


@pytest.mark.parametrize(
    (
        "tp_rank",
        "tp_size",
        "dp_rank",
        "dp_size",
        "enable_dp_attention",
        "attn_cp_rank",
        "attn_cp_size",
        "routed_dp_rank",
        "expected_dp_rank",
        "expected_publisher",
    ),
    (
        (0, 2, None, 1, False, 0, 1, None, None, True),
        (1, 2, None, 1, False, 0, 1, None, None, False),
        (0, 2, None, 1, False, 0, 1, "7", 7, True),
        (1, 2, None, 1, False, 0, 1, "7", 7, False),
        (0, 2, 1, 2, False, 0, 1, None, 1, True),
        (1, 2, 1, 2, False, 0, 1, None, 1, False),
        (2, 4, 1, 2, True, 0, 1, None, 1, True),
        (3, 4, 1, 2, True, 0, 1, None, 1, False),
        (0, 4, None, 1, False, 0, 2, None, None, True),
        (2, 4, None, 1, False, 1, 2, None, None, False),
        (4, 8, 1, 2, True, 0, 2, None, 1, True),
        (6, 8, 1, 2, True, 1, 2, None, 1, False),
    ),
)
def test_parallel_state_rank_domain_matrix(
    tp_rank: int,
    tp_size: int,
    dp_rank: int | None,
    dp_size: int,
    enable_dp_attention: bool,
    attn_cp_rank: int,
    attn_cp_size: int,
    routed_dp_rank: str | None,
    expected_dp_rank: int | None,
    expected_publisher: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production construction preserves TP-only, DP, and attention-DP ranks."""

    if routed_dp_rank is None:
        monkeypatch.delenv("SGLANG_DP_RANK", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DP_RANK", routed_dp_rank)
    parallel_state = _production_parallel_state(
        tp_rank=tp_rank,
        tp_size=tp_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
    )
    identity = SchedulerOutputIdentity.from_parallel_state(parallel_state)

    assert identity.dp_rank == expected_dp_rank
    assert identity.is_gateway_publisher is expected_publisher


@pytest.mark.parametrize("tp_rank", (0, 1))
def test_routed_rank_uses_process_environment(
    tp_rank: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An external routed identity survives the real process resolution path."""

    monkeypatch.setenv("SGLANG_DP_RANK", "7")
    identity = SchedulerOutputIdentity.from_parallel_state(
        _production_parallel_state(
            dp_rank=None,
            dp_size=1,
            tp_rank=tp_rank,
            tp_size=2,
        )
    )

    assert identity.dp_rank == 7
    assert identity.is_gateway_publisher is (tp_rank == 0)


def test_controller_rank_precedes_routed_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A true DP controller identity cannot be replaced by router process state."""

    monkeypatch.setenv("SGLANG_DP_RANK", "7")
    identity = SchedulerOutputIdentity.from_parallel_state(
        _production_parallel_state(dp_rank=1, dp_size=2)
    )

    assert identity.dp_rank == 1


def test_invalid_routed_process_environment_fails_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed router process state fails before scheduler construction."""

    monkeypatch.setenv("SGLANG_DP_RANK", "not-an-integer")

    with pytest.raises(ValueError, match="SGLANG_DP_RANK must be an integer"):
        _production_parallel_state(dp_rank=None)


@pytest.mark.parametrize(
    "field",
    ("dp_size", "tp_size", "pp_size", "attn_tp_size", "attn_cp_size"),
)
def test_size_fields_require_exact_positive_integers(field: str) -> None:
    """Every output rank size is constructor-validated."""

    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        SchedulerOutputIdentity.from_parallel_state(
            dataclasses.replace(_unchecked_parallel_state(), **{field: True})
        )
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        SchedulerOutputIdentity.from_parallel_state(
            dataclasses.replace(_unchecked_parallel_state(), **{field: 0})
        )


@pytest.mark.parametrize(
    ("rank_field", "size_field"),
    (
        ("tp_rank", "tp_size"),
        ("pp_rank", "pp_size"),
        ("attn_tp_rank", "attn_tp_size"),
        ("attn_cp_rank", "attn_cp_size"),
    ),
)
def test_rank_fields_require_exact_in_range_integers(
    rank_field: str,
    size_field: str,
) -> None:
    """Every output rank coordinate is constructor-validated."""

    base = _unchecked_parallel_state()
    with pytest.raises(TypeError, match=f"{rank_field} must be an integer"):
        SchedulerOutputIdentity.from_parallel_state(
            dataclasses.replace(base, **{rank_field: True})
        )
    with pytest.raises(ValueError, match=f"{rank_field} must identify one rank"):
        SchedulerOutputIdentity.from_parallel_state(
            dataclasses.replace(
                base,
                **{rank_field: getattr(base, size_field)},
            )
        )


def test_output_identity_rejects_unsupported_pipeline_parallelism() -> None:
    """The publication identity admits PP1 only."""

    with pytest.raises(ValueError, match="requires pp_size == 1"):
        SchedulerOutputIdentity.from_parallel_state(
            _production_parallel_state(pp_size=2)
        )


def test_data_parallel_rank_domain_is_constructor_enforced() -> None:
    """DP rank nullability and range retain routed-rank semantics."""

    with pytest.raises(TypeError, match="dp_rank must be an integer or None"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(dp_rank=True)
        )
    with pytest.raises(ValueError, match="dp_rank must be non-negative"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(dp_rank=-1)
        )
    with pytest.raises(ValueError, match="dp_rank may be None"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(dp_size=2)
        )
    with pytest.raises(ValueError, match="within dp_size"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(dp_rank=2, dp_size=2)
        )

    routed_identity = SchedulerOutputIdentity.from_parallel_state(
        _production_parallel_state(dp_rank=7, dp_size=1)
    )
    assert routed_identity.dp_rank == 7


def test_attention_partition_invariants_are_constructor_enforced() -> None:
    """Attention publication coordinates cannot disagree with TP topology."""

    with pytest.raises(ValueError, match="must partition"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(tp_size=3, attn_tp_size=2)
        )
    with pytest.raises(ValueError, match="attn_tp_rank is inconsistent"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(tp_rank=1, tp_size=2, attn_tp_size=2)
        )
    with pytest.raises(ValueError, match="attn_cp_rank is inconsistent"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(
                tp_rank=2,
                tp_size=4,
                attn_tp_size=2,
                attn_cp_size=2,
            )
        )
    with pytest.raises(ValueError, match="dp_size is inconsistent"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(
                dp_rank=1,
                dp_size=3,
                tp_rank=2,
                tp_size=4,
                attn_tp_size=2,
            )
        )
    with pytest.raises(ValueError, match="dp_rank is inconsistent"):
        SchedulerOutputIdentity.from_parallel_state(
            _unchecked_parallel_state(
                dp_rank=0,
                dp_size=2,
                tp_rank=2,
                tp_size=4,
                attn_tp_size=2,
            )
        )


def test_terminal_binding_requires_exact_rank_and_authority() -> None:
    """Authenticated terminal startup cannot define another output publisher."""

    identity = SchedulerOutputIdentity.from_parallel_state(
        _production_parallel_state(tp_size=2)
    )
    identity.require_terminal_binding(
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
    )

    with pytest.raises(ValueError, match="startup binding disagrees"):
        identity.require_terminal_binding(
            tensor_parallel_rank=1,
            tensor_parallel_size=2,
        )
    with pytest.raises(ValueError, match="startup binding disagrees"):
        identity.require_terminal_binding(
            tensor_parallel_rank=0,
            tensor_parallel_size=4,
        )

    attention_dp_identity = SchedulerOutputIdentity.from_parallel_state(
        _production_parallel_state(
            dp_rank=1,
            dp_size=2,
            tp_rank=2,
            tp_size=4,
            enable_dp_attention=True,
        )
    )
    with pytest.raises(ValueError, match="publication authorities disagree"):
        attention_dp_identity.require_terminal_binding(
            tensor_parallel_rank=2,
            tensor_parallel_size=4,
        )


def test_output_identity_is_immutable() -> None:
    """Publication identity cannot drift after scheduler startup."""

    identity = SchedulerOutputIdentity.from_parallel_state(_production_parallel_state())

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.dp_rank = 1
