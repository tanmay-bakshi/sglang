import atexit
import logging
import os

import sglang.srt.distributed.parallel_state as parallel_state
import torch
import torch.distributed as dist
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires two GPUs, self-skips in CI",
)

DTYPE = torch.bfloat16
HIDDEN_SIZE = 5376
TOKEN_COUNTS = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
MAX_WORKSPACE_BYTES = 128 * 1024 * 1024
PROVIDERS = ["nccl", "jit-eager-128mib"]


@cache_once
def init_cpu_group() -> dist.ProcessGroup:
    """Initialize the process groups shared by both providers.

    :returns: Gloo group used to construct the JIT communicator.
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    parallel_state._WORLD = coordinator = parallel_state.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coordinator.cpu_group


@cache_once
def init_nccl_group() -> dist.ProcessGroup:
    """Create the NCCL process group used by the serving fallback.

    :returns: NCCL process group bound to the local device.
    """

    init_cpu_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    if not isinstance(group, dist.ProcessGroup):
        raise RuntimeError("NCCL group initialization returned a non-member sentinel")
    return group


class NCCLBackend:
    """NCCL all-reduce path used after the production 16 MiB cutoff."""

    def __init__(self) -> None:
        self._group = init_nccl_group()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce a tensor in place.

        :param tensor: Rank-local BF16 input.
        :returns: Reduced tensor.
        """

        dist.all_reduce(tensor, group=self._group)
        return tensor


class JITBackend:
    """Blackwell-tuned JIT all-reduce with the full prefill workspace."""

    def __init__(self) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        self._communicator = CustomAllReduceV2(
            init_cpu_group(),
            device,
            max_size=MAX_WORKSPACE_BYTES,
        )
        if self._communicator.disabled:
            raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
        register_comm_cleanup(self._communicator)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce a tensor with the tuned eager pull kernel.

        :param tensor: Rank-local BF16 input.
        :returns: Reduced tensor.
        :raises RuntimeError: If a production shape leaves the custom path.
        """

        if not self._communicator.should_custom_ar(tensor):
            raise RuntimeError(
                f"JIT all-reduce rejected a {tensor.nbytes}-byte production shape"
            )
        return self._communicator.custom_all_reduce(tensor)


@cache_once
def init_nccl_backend() -> NCCLBackend:
    """Construct the NCCL backend once per rank.

    :returns: Initialized NCCL backend.
    """

    return NCCLBackend()


@cache_once
def init_jit_backend() -> JITBackend:
    """Construct the JIT backend once per rank.

    :returns: Initialized JIT backend.
    """

    return JITBackend()


BACKEND_FACTORIES = {
    "nccl": init_nccl_backend,
    "jit-eager-128mib": init_jit_backend,
}


@cache_once
def init_backends() -> None:
    """Build communicators before any measured iteration."""

    if int(os.environ["LOCAL_RANK"]) == 0:
        logging.basicConfig(level=logging.INFO)
    for factory in BACKEND_FACTORIES.values():
        factory()
    logging.getLogger().setLevel(logging.WARNING)


def validate_result(
    backend: NCCLBackend | JITBackend,
    token_count: int,
    device: torch.device,
) -> None:
    """Validate the provider on the exact shape before timing it.

    :param backend: Provider under test.
    :param token_count: Prefill-wave token count.
    :param device: Rank-local CUDA device.
    :raises RuntimeError: If the reduced values are incorrect.
    """

    value = float(dist.get_rank(group=init_cpu_group()) + 1)
    tensor = torch.full(
        (token_count, HIDDEN_SIZE),
        value,
        dtype=DTYPE,
        device=device,
    )
    output = backend.all_reduce(tensor)
    torch.cuda.synchronize(device)
    expected = float(sum(range(1, dist.get_world_size(init_cpu_group()) + 1)))
    if not bool(torch.all(output == expected)):
        raise RuntimeError("all-reduce correctness check failed")


@marker.parametrize("token_count", TOKEN_COUNTS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(token_count: int, provider: str) -> marker.BenchResult:
    """Measure the exact Gemma 4 row-parallel prefill collective.

    :param token_count: Tokens in the prefill wave.
    :param provider: Collective implementation.
    :returns: Benchmark latency and effective-bandwidth result.
    """

    init_backends()
    backend = BACKEND_FACTORIES[provider]()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    validate_result(backend, token_count, device)
    tensor = torch.randn(
        (token_count, HIDDEN_SIZE),
        dtype=DTYPE,
        device=device,
    )
    world_size = dist.get_world_size(init_cpu_group())
    effective_bytes = int(tensor.nbytes * 2 * (world_size - 1) / world_size)
    return marker.do_bench(
        backend.all_reduce,
        input_args=(tensor,),
        use_cuda_graph=False,
        warmup_iters=100,
        replay_iters=500,
        sync_multigpu_fn=lambda: dist.barrier(init_nccl_group()),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=effective_bytes,
    )


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=[2],
        main_fn=benchmark.run,
        timeout=900,
    )
