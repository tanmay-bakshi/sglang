import argparse
from dataclasses import dataclass, replace
from typing import cast

import pytest
import torch
import torch.distributed as dist

from sglang.srt import distributed, runtime_context
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.server_args import ServerArgs

_TEST_WORLD_SIZE = 2
_TEST_HIDDEN_SIZE = 32
_TEST_MAX_TOKENS = 4


def test_dedicated_cli_flag_defaults_off_and_parses_independently() -> None:
    """The multimem flag must not alias either NCCL communication policy."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    disabled = parser.parse_args(["--model-path", "dummy"])
    enabled = parser.parse_args(
        ["--model-path", "dummy", "--enable-multimem-all-gather"]
    )

    assert not disabled.enable_multimem_all_gather
    assert enabled.enable_multimem_all_gather
    assert not enabled.enable_symm_mem
    assert not enabled.enable_nccl_nvls


@dataclass(frozen=True)
class _ServerArgs:
    """Runtime configuration used by the policy tests.

    :ivar enable_multimem_all_gather: Dedicated multimem policy.
    :ivar nnodes: Deployment node count.
    :ivar enable_symm_mem: Independent NCCL symmetric-memory policy.
    :ivar enable_nccl_nvls: Independent NCCL NVLS policy.
    """

    enable_multimem_all_gather: bool
    nnodes: int = 1
    enable_symm_mem: bool = False
    enable_nccl_nvls: bool = False


@dataclass(frozen=True)
class _TensorParallelGroup:
    """Minimal tensor-parallel topology used by the policy tests.

    :ivar world_size: Number of tensor-parallel ranks.
    :ivar rank_in_group: This process's tensor-parallel rank.
    :ivar cpu_group: Control-plane process group placeholder.
    :ivar device_group: Device process group placeholder.
    """

    world_size: int = _TEST_WORLD_SIZE
    rank_in_group: int = 0
    cpu_group: object = object()
    device_group: object = object()


@dataclass(frozen=True)
class _SymmetricMemoryHandle:
    """Minimal symmetric-memory handle.

    :ivar multicast_ptr: Nonzero when CUDA multicast is available.
    """

    multicast_ptr: int


def _make_state(
    *,
    max_tokens: int = _TEST_MAX_TOKENS,
    hidden_size: int = _TEST_HIDDEN_SIZE,
    multicast_ptr: int = 1,
) -> triton_symm_mem_ag.MultimemAllGatherState:
    """Build CPU-backed state for dispatch-only tests.

    :param max_tokens: Symmetric-buffer token capacity.
    :param hidden_size: Gathered hidden width.
    :param multicast_ptr: Simulated multicast pointer.
    :returns: Multimem state whose kernel calls must be mocked.
    """
    return triton_symm_mem_ag.MultimemAllGatherState(
        group=cast(dist.ProcessGroup, object()),
        rank_in_group=0,
        world_size=_TEST_WORLD_SIZE,
        device=torch.device("cpu"),
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16),
        symm_mem_hdl=_SymmetricMemoryHandle(multicast_ptr=multicast_ptr),
    )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    server_args: _ServerArgs,
    tp_group: _TensorParallelGroup | None = None,
) -> None:
    """Install deterministic runtime accessors.

    :param monkeypatch: Pytest patching fixture.
    :param server_args: Runtime policy returned to the gatherer.
    :param tp_group: Tensor-parallel topology returned to the gatherer.
    """

    def get_server_args() -> _ServerArgs:
        """Return the test runtime policy.

        :returns: Test server arguments.
        """
        return server_args

    def get_tp_group() -> _TensorParallelGroup:
        """Return the test tensor-parallel topology.

        :returns: Test tensor-parallel group.
        """
        if tp_group is None:
            raise AssertionError(
                "disabled multimem unexpectedly requested the TP group"
            )
        return tp_group

    monkeypatch.setattr(runtime_context, "get_server_args", get_server_args)
    monkeypatch.setattr(distributed, "get_tp_group", get_tp_group)


def _install_ordinary_all_gather(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[torch.Tensor],
) -> None:
    """Install a recording ordinary all-gather.

    :param monkeypatch: Pytest patching fixture.
    :param calls: Destination for rank-local inputs.
    """

    def ordinary_all_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
        """Record and represent an NCCL all-gather.

        :param x: Rank-local input.
        :param dim: Gathered dimension.
        :returns: Distinct output identifying the ordinary path.
        """
        assert dim == -1
        calls.append(x)
        return x.clone()

    monkeypatch.setattr(
        distributed,
        "tensor_model_parallel_all_gather",
        ordinary_all_gather,
    )


def _install_replicated_consensus(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[object],
) -> None:
    """Install a two-rank consensus returning identical values.

    :param monkeypatch: Pytest patching fixture.
    :param calls: Destination for consensus values.
    """

    def replicated_consensus(
        value: object,
        group: dist.ProcessGroup,
        world_size: int,
    ) -> list[object]:
        """Replicate one value across the fake group.

        :param value: Rank-local consensus value.
        :param group: Unused process-group placeholder.
        :param world_size: Expected group size.
        :returns: Two identical values.
        """
        del group
        assert world_size == _TEST_WORLD_SIZE
        calls.append(value)
        return [value, value]

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_all_gather_consensus_values",
        replicated_consensus,
    )


def test_dedicated_flag_is_independent_from_nccl_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL symmetric memory and NVLS must not enable Triton multimem.

    :param monkeypatch: Pytest patching fixture.
    """
    tp_group = _TensorParallelGroup()
    _install_runtime(
        monkeypatch,
        _ServerArgs(
            enable_multimem_all_gather=False,
            enable_symm_mem=True,
            enable_nccl_nvls=True,
        ),
        tp_group,
    )
    consensus_calls: list[object] = []
    _install_replicated_consensus(monkeypatch, consensus_calls)
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)

    output = gatherer(x)

    assert len(ordinary_calls) == 1
    assert torch.equal(output, x)
    assert gatherer._initialized
    assert gatherer._state is None
    assert len(consensus_calls) == 1
    record = cast(triton_symm_mem_ag._InitializationRecord, consensus_calls[0])
    assert not record.process_enabled


def test_mixed_dedicated_flag_is_fatal_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed process flags must fail only after every rank enters consensus.

    :param monkeypatch: Pytest patching fixture.
    """
    _install_runtime(
        monkeypatch,
        _ServerArgs(enable_multimem_all_gather=False),
        _TensorParallelGroup(),
    )

    def mixed_flag_consensus(
        value: object,
        group: dist.ProcessGroup,
        world_size: int,
    ) -> list[object]:
        """Return records representing one disabled and one enabled rank.

        :param value: Rank-local initialization record.
        :param group: Unused process-group placeholder.
        :param world_size: Expected group size.
        :returns: Initialization records with divergent process flags.
        """
        del group
        assert world_size == _TEST_WORLD_SIZE
        record = cast(triton_symm_mem_ag._InitializationRecord, value)
        return [record, replace(record, process_enabled=not record.process_enabled)]

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_all_gather_consensus_values",
        mixed_flag_consensus,
    )

    def unexpected_create_state(*args: object, **kwargs: object) -> None:
        """Reject rendezvous after process-flag consensus fails.

        :param args: Forbidden positional arguments.
        :param kwargs: Forbidden keyword arguments.
        """
        del args, kwargs
        raise AssertionError("mixed process flags reached rendezvous")

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "create_state",
        unexpected_create_state,
    )
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)

    with pytest.raises(RuntimeError, match="contract differs across TP ranks"):
        gatherer(x)

    assert len(ordinary_calls) == 0
    assert not gatherer._initialized


def test_initialization_consensus_builds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible replicated initialization must build once and stay multimem.

    :param monkeypatch: Pytest patching fixture.
    """
    _install_runtime(
        monkeypatch,
        _ServerArgs(enable_multimem_all_gather=True),
        _TensorParallelGroup(),
    )
    consensus_calls: list[object] = []
    _install_replicated_consensus(monkeypatch, consensus_calls)
    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_static_ineligibility_reason",
        lambda record: None,
    )
    state = _make_state()
    build_calls: list[tuple[int, int, torch.device]] = []

    def create_state(
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_tokens: int,
        hidden_size: int,
        device: torch.device,
    ) -> triton_symm_mem_ag.MultimemAllGatherState:
        """Record one simulated symmetric-memory rendezvous.

        :param group: Device process-group placeholder.
        :param rank_in_group: Tensor-parallel rank.
        :param max_tokens: Symmetric-buffer token capacity.
        :param hidden_size: Gathered hidden width.
        :param device: Input device.
        :returns: CPU-backed multimem state.
        """
        del group
        assert rank_in_group == 0
        build_calls.append((max_tokens, hidden_size, device))
        return state

    kernel_calls: list[torch.Tensor] = []

    def all_gather_inner(
        actual_state: triton_symm_mem_ag.MultimemAllGatherState,
        x: torch.Tensor,
        tp_hidden_dim: int,
        skip_entry_sync: bool,
        safe: bool,
    ) -> torch.Tensor:
        """Record one simulated multimem launch.

        :param actual_state: Committed state.
        :param x: Prepared rank-local input.
        :param tp_hidden_dim: Gathered hidden width.
        :param skip_entry_sync: Entry-barrier policy.
        :param safe: Output ownership policy.
        :returns: Input tensor as a dispatch marker.
        """
        assert actual_state is state
        assert tp_hidden_dim == _TEST_HIDDEN_SIZE
        assert not skip_entry_sync
        assert not safe
        kernel_calls.append(x)
        return x

    monkeypatch.setattr(triton_symm_mem_ag, "create_state", create_state)
    monkeypatch.setattr(triton_symm_mem_ag, "all_gather_inner", all_gather_inner)
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
        skip_entry_sync=False,
    )
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)

    first_output = gatherer(x)
    second_output = gatherer(x)

    assert first_output is x
    assert second_output is x
    assert build_calls == [(_TEST_MAX_TOKENS, _TEST_HIDDEN_SIZE, torch.device("cpu"))]
    assert len(consensus_calls) == _TEST_WORLD_SIZE
    assert len(kernel_calls) == _TEST_WORLD_SIZE
    assert len(ordinary_calls) == 0


def test_contract_mismatch_is_fatal_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Divergent rank contracts must not select different collectives.

    :param monkeypatch: Pytest patching fixture.
    """
    _install_runtime(
        monkeypatch,
        _ServerArgs(enable_multimem_all_gather=True),
        _TensorParallelGroup(),
    )

    def divergent_consensus(
        value: object,
        group: dist.ProcessGroup,
        world_size: int,
    ) -> list[object]:
        """Return two different initialization records.

        :param value: Rank-local initialization record.
        :param group: Unused process-group placeholder.
        :param world_size: Expected group size.
        :returns: Divergent records.
        """
        del group
        assert world_size == _TEST_WORLD_SIZE
        record = cast(triton_symm_mem_ag._InitializationRecord, value)
        return [record, replace(record, caller_name="other-caller")]

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_all_gather_consensus_values",
        divergent_consensus,
    )

    def unexpected_create_state(*args: object, **kwargs: object) -> None:
        """Reject rendezvous after failed consensus.

        :param args: Forbidden positional arguments.
        :param kwargs: Forbidden keyword arguments.
        """
        del args, kwargs
        raise AssertionError("divergent contracts reached rendezvous")

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "create_state",
        unexpected_create_state,
    )
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)

    with pytest.raises(RuntimeError, match="contract differs across TP ranks"):
        gatherer(x)


def test_rendezvous_failure_is_not_converted_to_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-consensus rendezvous failures must remain fatal.

    :param monkeypatch: Pytest patching fixture.
    """
    _install_runtime(
        monkeypatch,
        _ServerArgs(enable_multimem_all_gather=True),
        _TensorParallelGroup(),
    )
    consensus_calls: list[object] = []
    _install_replicated_consensus(monkeypatch, consensus_calls)
    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_static_ineligibility_reason",
        lambda record: None,
    )

    def failed_create_state(*args: object, **kwargs: object) -> None:
        """Simulate a post-consensus rendezvous failure.

        :param args: Rendezvous positional arguments.
        :param kwargs: Rendezvous keyword arguments.
        """
        del args, kwargs
        raise RuntimeError("rendezvous failed")

    monkeypatch.setattr(triton_symm_mem_ag, "create_state", failed_create_state)
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)

    with pytest.raises(RuntimeError, match="rendezvous failed"):
        gatherer(x)

    assert len(ordinary_calls) == 0
    assert not gatherer._initialized


@pytest.mark.parametrize(
    ("availability", "error_match"),
    [
        ([False, False], None),
        ([True, False], "availability differs across TP ranks"),
    ],
)
def test_multicast_availability_has_one_replicated_outcome(
    monkeypatch: pytest.MonkeyPatch,
    availability: list[bool],
    error_match: str | None,
) -> None:
    """Multicast discovery must globally fall back or globally fail.

    :param monkeypatch: Pytest patching fixture.
    :param availability: Simulated post-rendezvous rank availability.
    :param error_match: Expected error text, or ``None`` for global fallback.
    """
    _install_runtime(
        monkeypatch,
        _ServerArgs(enable_multimem_all_gather=True),
        _TensorParallelGroup(),
    )
    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_static_ineligibility_reason",
        lambda record: None,
    )
    consensus_count = 0

    def consensus(
        value: object,
        group: dist.ProcessGroup,
        world_size: int,
    ) -> list[object]:
        """Return replicated records and configured multicast availability.

        :param value: Rank-local record or availability.
        :param group: Unused process-group placeholder.
        :param world_size: Expected group size.
        :returns: Simulated consensus result.
        """
        nonlocal consensus_count
        del group
        assert world_size == _TEST_WORLD_SIZE
        consensus_count += 1
        if consensus_count == 1:
            return [value, value]
        return list(availability)

    monkeypatch.setattr(
        triton_symm_mem_ag,
        "_all_gather_consensus_values",
        consensus,
    )
    state = _make_state(multicast_ptr=1)
    monkeypatch.setattr(
        triton_symm_mem_ag,
        "create_state",
        lambda **kwargs: state,
    )
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    x = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)

    if error_match is not None:
        with pytest.raises(RuntimeError, match=error_match):
            gatherer(x)
        assert len(ordinary_calls) == 0
        return

    output = gatherer(x)
    assert torch.equal(output, x)
    assert len(ordinary_calls) == 1
    assert gatherer._initialized
    assert gatherer._state is None


def test_oversized_replicated_batch_uses_ordinary_all_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later replicated token count may safely exceed multimem capacity.

    :param monkeypatch: Pytest patching fixture.
    """
    state = _make_state(max_tokens=_TEST_MAX_TOKENS)
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    gatherer._initialized = True
    gatherer._state = state
    kernel_calls: list[torch.Tensor] = []

    def all_gather_inner(
        actual_state: triton_symm_mem_ag.MultimemAllGatherState,
        x: torch.Tensor,
        tp_hidden_dim: int,
        skip_entry_sync: bool,
        safe: bool,
    ) -> torch.Tensor:
        """Record the in-capacity multimem call.

        :param actual_state: Committed state.
        :param x: Rank-local input.
        :param tp_hidden_dim: Gathered hidden width.
        :param skip_entry_sync: Entry-barrier policy.
        :param safe: Output ownership policy.
        :returns: Input tensor as a dispatch marker.
        """
        assert actual_state is state
        assert tp_hidden_dim == _TEST_HIDDEN_SIZE
        assert not skip_entry_sync
        assert not safe
        kernel_calls.append(x)
        return x

    monkeypatch.setattr(triton_symm_mem_ag, "all_gather_inner", all_gather_inner)
    ordinary_calls: list[torch.Tensor] = []
    _install_ordinary_all_gather(monkeypatch, ordinary_calls)
    in_capacity = torch.arange(64, dtype=torch.bfloat16).reshape(4, 16)
    oversized = torch.arange(80, dtype=torch.bfloat16).reshape(5, 16)

    assert gatherer(in_capacity) is in_capacity
    oversized_output = gatherer(oversized)

    assert torch.equal(oversized_output, oversized)
    assert len(kernel_calls) == 1
    assert ordinary_calls == [oversized]
    assert gatherer._state is state


@pytest.mark.parametrize("layout", ["misaligned", "noncontiguous"])
def test_rank_local_layout_is_repaired_without_changing_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """Rank-local layout must remain on the committed multimem path.

    :param monkeypatch: Pytest patching fixture.
    :param layout: Rank-local layout defect to exercise.
    """
    state = _make_state()
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    gatherer._initialized = True
    gatherer._state = state
    prepared_inputs: list[torch.Tensor] = []

    def all_gather_inner(
        actual_state: triton_symm_mem_ag.MultimemAllGatherState,
        x: torch.Tensor,
        tp_hidden_dim: int,
        skip_entry_sync: bool,
        safe: bool,
    ) -> torch.Tensor:
        """Capture the repaired multimem input.

        :param actual_state: Committed state.
        :param x: Prepared rank-local input.
        :param tp_hidden_dim: Gathered hidden width.
        :param skip_entry_sync: Entry-barrier policy.
        :param safe: Output ownership policy.
        :returns: Prepared input.
        """
        del tp_hidden_dim, skip_entry_sync, safe
        assert actual_state is state
        prepared_inputs.append(x)
        return x

    monkeypatch.setattr(triton_symm_mem_ag, "all_gather_inner", all_gather_inner)

    def unexpected_ordinary_all_gather(*args: object, **kwargs: object) -> None:
        """Reject rank-local layout fallback.

        :param args: Forbidden positional arguments.
        :param kwargs: Forbidden keyword arguments.
        """
        del args, kwargs
        raise AssertionError("rank-local layout changed collective dispatch")

    monkeypatch.setattr(
        distributed,
        "tensor_model_parallel_all_gather",
        unexpected_ordinary_all_gather,
    )
    if layout == "misaligned":
        base = torch.arange(33, dtype=torch.bfloat16)
        input_x = base[1:33].reshape(2, 16)
        assert input_x.is_contiguous()
        assert input_x.data_ptr() % 16 != 0
    else:
        input_x = torch.arange(32, dtype=torch.bfloat16).reshape(16, 2).T
        assert not input_x.is_contiguous()

    output = gatherer(input_x)

    assert len(prepared_inputs) == 1
    prepared = prepared_inputs[0]
    assert prepared.is_contiguous()
    assert prepared.data_ptr() % 16 == 0
    assert prepared.data_ptr() != input_x.data_ptr()
    assert torch.equal(output, input_x)


@pytest.mark.parametrize(
    ("x", "error_match"),
    [
        (
            torch.arange(16, dtype=torch.float32).reshape(1, 16),
            "dtype changed",
        ),
        (
            torch.arange(8, dtype=torch.bfloat16).reshape(1, 8),
            "hidden width changed",
        ),
    ],
)
def test_committed_static_contract_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    x: torch.Tensor,
    error_match: str,
) -> None:
    """Static contract violations must fail instead of splitting collectives.

    :param monkeypatch: Pytest patching fixture.
    :param x: Contract-violating input.
    :param error_match: Expected error text.
    """
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=_TEST_MAX_TOKENS,
        name="policy-test",
    )
    gatherer._initialized = True
    gatherer._state = _make_state()

    def unexpected_ordinary_all_gather(*args: object, **kwargs: object) -> None:
        """Reject static-contract fallback.

        :param args: Forbidden positional arguments.
        :param kwargs: Forbidden keyword arguments.
        """
        del args, kwargs
        raise AssertionError("static contract violation fell back")

    monkeypatch.setattr(
        distributed,
        "tensor_model_parallel_all_gather",
        unexpected_ordinary_all_gather,
    )

    with pytest.raises(RuntimeError, match=error_match):
        gatherer(x)
