import argparse
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from sglang.kernels.ops.layernorm import fused_add_rmsnorm
from sglang.srt.layers.quantization.fp4_utils import (
    fp4_quantize,
    fused_add_rmsnorm_fp4_quantize,
)

HIDDEN_SIZE = 5376
EPSILON = 1.0e-6


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and numerical comparison for one Gemma 4 dense-MLP boundary.

    :ivar token_count: Flattened token count.
    :ivar baseline_us: Median incumbent kernel-chain time in microseconds.
    :ivar fused_us: Median fused kernel time in microseconds.
    :ivar residual_exact_fraction: Fraction of updated residual values that match.
    :ivar packed_exact_fraction: Fraction of packed FP4 bytes that match.
    :ivar scale_exact_fraction: Fraction of E4M3 scale bytes that match.
    """

    token_count: int
    baseline_us: float
    fused_us: float
    residual_exact_fraction: float
    packed_exact_fraction: float
    scale_exact_fraction: float

    @property
    def speedup(self) -> float:
        """Return incumbent time divided by fused time.

        :returns: Kernel-chain speedup.
        """

        return self.baseline_us / self.fused_us


def _exact_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the elementwise exact-match fraction.

    :param left: First tensor.
    :param right: Second tensor with the same shape and dtype.
    :returns: Fraction of exactly equal elements.
    """

    if left.shape != right.shape or left.dtype != right.dtype:
        return 0.0
    return float((left == right).to(torch.float32).mean().item())


def _byte_exact_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the exact-match fraction over raw tensor bytes.

    :param left: First contiguous tensor.
    :param right: Second contiguous tensor with the same byte size.
    :returns: Fraction of exactly equal bytes.
    """

    left_bytes = left.contiguous().view(torch.uint8).flatten()
    right_bytes = right.contiguous().view(torch.uint8).flatten()
    if left_bytes.shape != right_bytes.shape:
        return 0.0
    return float((left_bytes == right_bytes).to(torch.float32).mean().item())


def _time_cuda(operation: Callable[[], None], repeats: int) -> float:
    """Measure median CUDA execution time without per-iteration synchronizes.

    :param operation: CUDA operation to time.
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
    token_count: int,
    repeats: int,
    device: torch.device,
) -> BenchmarkResult:
    """Compare the incumbent and fused paths for one token bucket.

    :param token_count: Flattened token count.
    :param repeats: Number of measured launches.
    :param device: CUDA device executing the benchmark.
    :returns: Timing and exactness comparison.
    :raises RuntimeError: If the fused FlashInfer operator is unavailable.
    """

    if fused_add_rmsnorm_fp4_quantize is None:
        raise RuntimeError("fused RMSNorm FP4 quantization is unavailable")
    if fp4_quantize is None:
        raise RuntimeError("FP4 quantization is unavailable")

    generator = torch.Generator(device=device)
    generator.manual_seed(20260729 + token_count)
    source_input = torch.randn(
        (token_count, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    source_residual = torch.randn(
        (token_count, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        (HIDDEN_SIZE,),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    global_scale = torch.tensor([21.503999], dtype=torch.float32, device=device)

    baseline_input = source_input.clone()
    baseline_residual = source_residual.clone()
    fused_input = source_input.clone()
    fused_residual = source_residual.clone()

    fused_add_rmsnorm(
        baseline_input,
        baseline_residual,
        weight,
        EPSILON,
    )
    baseline_packed, baseline_scale = fp4_quantize(
        baseline_input,
        global_scale,
    )
    fused_packed, fused_scale = fused_add_rmsnorm_fp4_quantize(
        fused_input,
        fused_residual,
        weight,
        global_scale,
        EPSILON,
    )
    torch.cuda.synchronize()

    residual_exact_fraction = _exact_fraction(
        baseline_residual,
        fused_residual,
    )
    packed_exact_fraction = _exact_fraction(baseline_packed, fused_packed)
    scale_exact_fraction = _byte_exact_fraction(baseline_scale, fused_scale)

    baseline_input = source_input.clone()
    baseline_residual = source_residual.clone()
    fused_input = source_input.clone()
    fused_residual = source_residual.clone()

    def baseline() -> None:
        fused_add_rmsnorm(
            baseline_input,
            baseline_residual,
            weight,
            EPSILON,
        )
        fp4_quantize(baseline_input, global_scale)

    def fused() -> None:
        fused_add_rmsnorm_fp4_quantize(
            fused_input,
            fused_residual,
            weight,
            global_scale,
            EPSILON,
        )

    return BenchmarkResult(
        token_count=token_count,
        baseline_us=_time_cuda(baseline, repeats),
        fused_us=_time_cuda(fused, repeats),
        residual_exact_fraction=residual_exact_fraction,
        packed_exact_fraction=packed_exact_fraction,
        scale_exact_fraction=scale_exact_fraction,
    )


def _parse_ints(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated integer sequence.

    :param raw_values: Comma-separated integers.
    :returns: Parsed integer tuple.
    """

    return tuple(int(value) for value in raw_values.split(","))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Gemma 4 fused RMSNorm-to-FP4 benchmark.

    :param argv: Optional command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,8,1024,8192")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    results = [
        _run_shape(token_count, args.repeats, device)
        for token_count in _parse_ints(args.tokens)
    ]
    print(
        "tokens\tbaseline_us\tfused_us\tspeedup\tresidual_exact_fraction"
        "\tpacked_exact_fraction\tscale_exact_fraction"
    )
    for result in results:
        print(
            f"{result.token_count}\t{result.baseline_us:.3f}"
            f"\t{result.fused_us:.3f}\t{result.speedup:.3f}"
            f"\t{result.residual_exact_fraction:.8f}"
            f"\t{result.packed_exact_fraction:.8f}"
            f"\t{result.scale_exact_fraction:.8f}"
        )


if __name__ == "__main__":
    main()
