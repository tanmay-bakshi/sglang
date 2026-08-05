from dataclasses import dataclass

import pytest
import torch

import sglang.srt.distributed as distributed
import sglang.srt.runtime_context as runtime_context
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag


@dataclass(frozen=True)
class _ServerArgs:
    """Runtime configuration used by the multimem policy test.

    :ivar enable_symm_mem: Whether CUDA symmetric memory is enabled.
    :ivar nnodes: Number of participating nodes.
    """

    enable_symm_mem: bool
    nnodes: int


@dataclass(frozen=True)
class _TensorParallelGroup:
    """Tensor-parallel topology used by the multimem policy test.

    :ivar world_size: Number of tensor-parallel ranks.
    """

    world_size: int


def test_disabled_symmetric_memory_never_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the disabled policy selects the ordinary all-gather directly.

    :param monkeypatch: Pytest patching fixture.
    """

    def get_server_args() -> _ServerArgs:
        """Return a configuration which administratively disables multicast.

        :returns: Disabled symmetric-memory configuration.
        """

        return _ServerArgs(enable_symm_mem=False, nnodes=1)

    def get_tp_group() -> _TensorParallelGroup:
        """Return a multi-rank topology which would otherwise use multimem.

        :returns: Two-rank tensor-parallel topology.
        """

        return _TensorParallelGroup(world_size=2)

    def unexpected_rendezvous(*args: object, **kwargs: object) -> None:
        """Fail if the disabled path attempts to create multicast state.

        :param args: Positional arguments passed to the forbidden operation.
        :param kwargs: Keyword arguments passed to the forbidden operation.
        """

        raise AssertionError("disabled symmetric memory attempted a rendezvous")

    def ordinary_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Represent the non-multicast tensor-parallel all-gather.

        :param tensor: Local tensor-parallel shard.
        :param dim: Dimension gathered across tensor-parallel ranks.
        :returns: A distinct tensor identifying the selected fallback.
        """

        assert dim == -1
        return tensor.clone()

    monkeypatch.setattr(runtime_context, "get_server_args", get_server_args)
    monkeypatch.setattr(distributed, "get_tp_group", get_tp_group)
    monkeypatch.setattr(triton_symm_mem_ag, "create_state", unexpected_rendezvous)
    monkeypatch.setattr(
        distributed, "tensor_model_parallel_all_gather", ordinary_all_gather
    )

    gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=128)
    local_tensor = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

    gathered_tensor = gatherer(local_tensor)

    assert torch.equal(gathered_tensor, local_tensor)
    assert gathered_tensor is not local_tensor
