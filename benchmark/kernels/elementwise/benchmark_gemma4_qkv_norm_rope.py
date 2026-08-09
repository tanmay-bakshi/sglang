import argparse
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.gemma4_qkv_norm_rope import (
    gemma4_qkv_norm_rope,
)
from sglang.kernels.ops.attention.rope import (
    apply_rope_with_cos_sin_cache_inplace,
)
from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm

HEAD_DIM = 256
CONTEXT_LENGTH = 8192


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and numerical comparison for one Gemma 4 attention shape.

    :ivar tensor_parallel_size: Tensor-parallel degree.
    :ivar token_count: Flattened token count.
    :ivar rope_type: RoPE cache family.
    :ivar baseline_us: Median incumbent kernel time in microseconds.
    :ivar fused_us: Median fused kernel time in microseconds.
    :ivar exact_fraction: Fraction of BF16 outputs that match bit-for-bit.
    :ivar max_absolute_error: Largest output difference after BF16 conversion.
    """

    tensor_parallel_size: int
    token_count: int
    rope_type: str
    baseline_us: float
    fused_us: float
    exact_fraction: float
    max_absolute_error: float

    @property
    def speedup(self) -> float:
        """Return incumbent time divided by fused time.

        :returns: Kernel-time speedup.
        """
        return self.baseline_us / self.fused_us


def _build_rope_cache(rope_type: str, device: torch.device) -> torch.Tensor:
    """Build the exact Gemma 4 FP32 cache for one attention family.

    :param rope_type: ``sliding`` or ``proportional``.
    :param device: CUDA device receiving the cache.
    :returns: Contiguous ``[context_length, head_dim]`` cosine/sine cache.
    """
    if rope_type == "sliding":
        inverse_frequency = 1.0 / (
            10000.0
            ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
    elif rope_type == "proportional":
        rotated_frequency = 1.0 / (
            1000000.0
            ** (torch.arange(0, 64, 2, dtype=torch.float32) / HEAD_DIM)
        )
        inverse_frequency = torch.cat(
            (rotated_frequency, torch.zeros(96, dtype=torch.float32))
        )
    else:
        raise ValueError(f"unsupported RoPE type: {rope_type}")

    positions = torch.arange(CONTEXT_LENGTH, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequency)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(device)


def _time_cuda(operation: Callable[[], None], repeats: int) -> float:
    """Measure median CUDA execution time without per-iteration synchronizes.

    :param operation: Mutation-only CUDA operation to time.
    :param repeats: Number of measured launches.
    :returns: Median elapsed time in microseconds.
    """
    for _ in range(10):
        operation()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    )


def _run_shape(
    tensor_parallel_size: int,
    token_count: int,
    rope_type: str,
    repeats: int,
    device: torch.device,
) -> BenchmarkResult:
    """Compare the incumbent and fused paths for one exact QKV layout.

    :param tensor_parallel_size: Tensor-parallel degree.
    :param token_count: Flattened token count.
    :param rope_type: RoPE cache family.
    :param repeats: Number of measured launches.
    :param device: CUDA device executing the benchmark.
    :returns: Timing and numerical comparison.
    """
    num_q_heads = 32 // tensor_parallel_size
    num_kv_heads = 16 // tensor_parallel_size
    q_size = num_q_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv_width = q_size + 2 * kv_size

    generator = torch.Generator(device=device)
    generator.manual_seed(20260729 + tensor_parallel_size * 10000 + token_count)
    source = torch.randn(
        (token_count, qkv_width),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_weight = torch.randn(
        (HEAD_DIM,),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_weight = torch.randn(
        (HEAD_DIM,),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = (
        torch.arange(token_count, dtype=torch.int64, device=device) + 17
    ) % CONTEXT_LENGTH
    cache = _build_rope_cache(rope_type, device)

    baseline_output = source.clone()
    baseline_q, baseline_k, baseline_v = baseline_output.split(
        (q_size, kv_size, kv_size), dim=-1
    )

    def baseline() -> None:
        gemma_qkv_rmsnorm(
            baseline_q,
            baseline_k,
            baseline_v,
            q_weight,
            k_weight,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
        )
        apply_rope_with_cos_sin_cache_inplace(
            baseline_q.view(token_count, num_q_heads, HEAD_DIM),
            baseline_k.view(token_count, num_kv_heads, HEAD_DIM),
            cache,
            positions,
            is_neox=True,
            rope_dim=HEAD_DIM,
        )

    fused_output = source.clone()

    def fused() -> None:
        gemma4_qkv_norm_rope(
            fused_output,
            q_weight,
            k_weight,
            cache,
            positions,
            num_q_heads,
            num_kv_heads,
            1e-6,
        )

    baseline()
    fused()
    difference = (baseline_output.float() - fused_output.float()).abs()
    exact_fraction = float((baseline_output == fused_output).float().mean().item())
    max_absolute_error = float(difference.max().item())

    baseline_timing_input = source.clone()
    baseline_q, baseline_k, baseline_v = baseline_timing_input.split(
        (q_size, kv_size, kv_size), dim=-1
    )
    fused_output = source.clone()
    baseline_us = _time_cuda(baseline, repeats)
    fused_us = _time_cuda(fused, repeats)
    return BenchmarkResult(
        tensor_parallel_size=tensor_parallel_size,
        token_count=token_count,
        rope_type=rope_type,
        baseline_us=baseline_us,
        fused_us=fused_us,
        exact_fraction=exact_fraction,
        max_absolute_error=max_absolute_error,
    )


def _parse_ints(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated integer sequence.

    :param raw_values: Comma-separated integers.
    :returns: Parsed integer tuple.
    """
    return tuple(int(value) for value in raw_values.split(","))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Gemma 4 fused attention post-projection benchmark.

    :param argv: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", default="1,2,4")
    parser.add_argument("--tokens", default="1,8,1024,8192")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    results = [
        _run_shape(tp, token_count, rope_type, args.repeats, device)
        for tp in _parse_ints(args.tp)
        for token_count in _parse_ints(args.tokens)
        for rope_type in ("sliding", "proportional")
    ]

    print(
        "tp\ttokens\trope\tbaseline_us\tfused_us\tspeedup\texact_fraction\tmax_abs_error"
    )
    for result in results:
        print(
            f"{result.tensor_parallel_size}\t{result.token_count}\t{result.rope_type}"
            f"\t{result.baseline_us:.3f}\t{result.fused_us:.3f}"
            f"\t{result.speedup:.3f}\t{result.exact_fraction:.8f}"
            f"\t{result.max_absolute_error:.8f}"
        )


if __name__ == "__main__":
    main()
