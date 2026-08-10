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
class RegionMismatchCounts:
    """Bitwise mismatch counts partitioned by the fused operation boundary.

    :ivar q_first_half: Query mismatches in the first NeoX half.
    :ivar q_second_half: Query mismatches in the second NeoX half.
    :ivar k_first_half: Key mismatches in the first NeoX half.
    :ivar k_second_half: Key mismatches in the second NeoX half.
    :ivar v: Value mismatches after RMSNorm.
    """

    q_first_half: int
    q_second_half: int
    k_first_half: int
    k_second_half: int
    v: int

    @property
    def total(self) -> int:
        """Return the total mismatch count.

        :returns: Mismatches across every QKV region.
        """
        return (
            self.q_first_half
            + self.q_second_half
            + self.k_first_half
            + self.k_second_half
            + self.v
        )


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
    :ivar region_mismatches: Mismatch counts partitioned by QKV and NeoX half.
    """

    tensor_parallel_size: int
    token_count: int
    rope_type: str
    baseline_us: float
    fused_us: float
    exact_fraction: float
    max_absolute_error: float
    region_mismatches: RegionMismatchCounts

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
            10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
    elif rope_type == "proportional":
        rotated_frequency = 1.0 / (
            1000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / HEAD_DIM)
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


def _count_region_mismatches(
    baseline_output: torch.Tensor,
    fused_output: torch.Tensor,
    q_size: int,
    kv_size: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> RegionMismatchCounts:
    """Partition bitwise mismatches across QKV and the two NeoX halves.

    :param baseline_output: Incumbent QKV result.
    :param fused_output: Candidate QKV result.
    :param q_size: Flattened query width.
    :param kv_size: Flattened key or value width.
    :param num_q_heads: Query head count.
    :param num_kv_heads: Key and value head count.
    :returns: Regional mismatch counts.
    """
    baseline_q, baseline_k, baseline_v = baseline_output.split(
        (q_size, kv_size, kv_size), dim=-1
    )
    fused_q, fused_k, fused_v = fused_output.split((q_size, kv_size, kv_size), dim=-1)
    baseline_q = baseline_q.view(-1, num_q_heads, HEAD_DIM)
    fused_q = fused_q.view(-1, num_q_heads, HEAD_DIM)
    baseline_k = baseline_k.view(-1, num_kv_heads, HEAD_DIM)
    fused_k = fused_k.view(-1, num_kv_heads, HEAD_DIM)
    half_head_dim = HEAD_DIM // 2

    def mismatch_count(lhs: torch.Tensor, rhs: torch.Tensor) -> int:
        return int(torch.count_nonzero(lhs != rhs).item())

    return RegionMismatchCounts(
        q_first_half=mismatch_count(
            baseline_q[..., :half_head_dim], fused_q[..., :half_head_dim]
        ),
        q_second_half=mismatch_count(
            baseline_q[..., half_head_dim:], fused_q[..., half_head_dim:]
        ),
        k_first_half=mismatch_count(
            baseline_k[..., :half_head_dim], fused_k[..., :half_head_dim]
        ),
        k_second_half=mismatch_count(
            baseline_k[..., half_head_dim:], fused_k[..., half_head_dim:]
        ),
        v=mismatch_count(baseline_v, fused_v),
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
    region_mismatches = _count_region_mismatches(
        baseline_output,
        fused_output,
        q_size,
        kv_size,
        num_q_heads,
        num_kv_heads,
    )
    exact_fraction = 1.0 - region_mismatches.total / baseline_output.numel()
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
        region_mismatches=region_mismatches,
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
    parser.add_argument("--tp", default="1,2,4,8")
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
        "tp\ttokens\trope\tbaseline_us\tfused_us\tspeedup\texact_fraction"
        "\tmax_abs_error\tq_first_mismatches\tq_second_mismatches"
        "\tk_first_mismatches\tk_second_mismatches\tv_mismatches"
    )
    for result in results:
        print(
            f"{result.tensor_parallel_size}\t{result.token_count}\t{result.rope_type}"
            f"\t{result.baseline_us:.3f}\t{result.fused_us:.3f}"
            f"\t{result.speedup:.3f}\t{result.exact_fraction:.8f}"
            f"\t{result.max_absolute_error:.8f}"
            f"\t{result.region_mismatches.q_first_half}"
            f"\t{result.region_mismatches.q_second_half}"
            f"\t{result.region_mismatches.k_first_half}"
            f"\t{result.region_mismatches.k_second_half}"
            f"\t{result.region_mismatches.v}"
        )


if __name__ == "__main__":
    main()
