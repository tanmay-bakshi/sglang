import argparse
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from sglang.kernels.ops.quantization.gelu_tanh_and_mul_fp4_quant import (
    gelu_tanh_and_mul_fp4_quant,
)
from sglang.srt.layers.activation import gelu_tanh_and_mul
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

INTERMEDIATE_SIZE_BY_TP = {
    1: 21504,
    2: 10752,
    4: 5376,
}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and ABI comparison for one Gemma 4 dense-MLP shape.

    :ivar tensor_parallel_size: Tensor-parallel degree represented by the shape.
    :ivar token_count: Flattened token count.
    :ivar hidden_size: Per-rank intermediate width.
    :ivar baseline_us: Median incumbent path time in microseconds.
    :ivar fused_us: Median fused path time in microseconds.
    :ivar packed_exact_fraction: Fraction of packed FP4 bytes matching exactly.
    :ivar scale_exact_fraction: Fraction of swizzled scale bytes matching exactly.
    """

    tensor_parallel_size: int
    token_count: int
    hidden_size: int
    baseline_us: float
    fused_us: float
    packed_exact_fraction: float
    scale_exact_fraction: float

    @property
    def speedup(self) -> float:
        """Return incumbent time divided by fused time.

        :returns: End-to-end operator speedup.
        """
        return self.baseline_us / self.fused_us


QuantizedOutput = tuple[torch.Tensor, torch.Tensor]


def _baseline(input: torch.Tensor, global_scale: torch.Tensor) -> QuantizedOutput:
    """Run the incumbent GeGLU followed by FlashInfer FP4 quantization.

    :param input: Contiguous BF16 gate/up tensor.
    :param global_scale: ModelOpt inverse activation scale.
    :returns: Packed FP4 activations and swizzled block scales.
    :raises RuntimeError: If FlashInfer FP4 quantization is unavailable.
    """
    if fp4_quantize is None:
        raise RuntimeError("FlashInfer FP4 quantization is unavailable")
    activated = gelu_tanh_and_mul(input)
    return fp4_quantize(activated, global_scale)


def _time_cuda(operation: Callable[[], QuantizedOutput], repeats: int) -> float:
    """Measure median CUDA execution time without per-iteration synchronization.

    :param operation: CUDA operation to measure.
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


def _exact_fraction(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compare two tensors as their underlying byte sequences.

    :param reference: Incumbent ABI tensor.
    :param candidate: Fused-kernel ABI tensor.
    :returns: Fraction of bytes matching exactly.
    :raises ValueError: If the byte counts differ.
    """
    reference_bytes = reference.contiguous().view(torch.uint8).flatten()
    candidate_bytes = candidate.contiguous().view(torch.uint8).flatten()
    if reference_bytes.numel() != candidate_bytes.numel():
        raise ValueError(
            "ABI byte counts differ: "
            f"reference={reference_bytes.numel()}, candidate={candidate_bytes.numel()}"
        )
    return float((reference_bytes == candidate_bytes).float().mean().item())


def _run_shape(
    tensor_parallel_size: int,
    token_count: int,
    repeats: int,
    global_scale_value: float,
    device: torch.device,
) -> BenchmarkResult:
    """Compare incumbent and fused paths for one production Gemma 4 shape.

    :param tensor_parallel_size: Tensor-parallel degree represented by the shape.
    :param token_count: Flattened token count.
    :param repeats: Number of measured launches.
    :param global_scale_value: ModelOpt inverse activation scale value.
    :param device: CUDA device executing the benchmark.
    :returns: Timing and byte-exactness result.
    """
    hidden_size = INTERMEDIATE_SIZE_BY_TP[tensor_parallel_size]
    generator = torch.Generator(device=device)
    generator.manual_seed(20260729 + tensor_parallel_size * 10000 + token_count)
    input = torch.randn(
        (token_count, hidden_size * 2),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    global_scale = torch.tensor(
        [global_scale_value],
        dtype=torch.float32,
        device=device,
    )

    reference_packed, reference_scales = _baseline(input, global_scale)
    candidate_packed, candidate_scales = gelu_tanh_and_mul_fp4_quant(
        input,
        global_scale,
    )
    packed_exact_fraction = _exact_fraction(reference_packed, candidate_packed)
    scale_exact_fraction = _exact_fraction(reference_scales, candidate_scales)

    baseline_us = _time_cuda(lambda: _baseline(input, global_scale), repeats)
    fused_us = _time_cuda(
        lambda: gelu_tanh_and_mul_fp4_quant(input, global_scale),
        repeats,
    )
    return BenchmarkResult(
        tensor_parallel_size=tensor_parallel_size,
        token_count=token_count,
        hidden_size=hidden_size,
        baseline_us=baseline_us,
        fused_us=fused_us,
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
    """Run the Gemma 4 fused GeGLU-to-FP4 benchmark.

    :param argv: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", default="1,2,4")
    parser.add_argument("--tokens", default="1,8,1024,8192")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--global-scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    tensor_parallel_sizes = _parse_ints(args.tp)
    unsupported = set(tensor_parallel_sizes) - set(INTERMEDIATE_SIZE_BY_TP)
    if len(unsupported) > 0:
        raise ValueError(f"unsupported tensor-parallel sizes: {sorted(unsupported)}")

    device = torch.device("cuda", torch.cuda.current_device())
    results = [
        _run_shape(
            tensor_parallel_size,
            token_count,
            args.repeats,
            args.global_scale,
            device,
        )
        for tensor_parallel_size in tensor_parallel_sizes
        for token_count in _parse_ints(args.tokens)
    ]

    print(
        "tp\ttokens\thidden\tbaseline_us\tfused_us\tspeedup"
        "\tpacked_exact_fraction\tscale_exact_fraction"
    )
    for result in results:
        print(
            f"{result.tensor_parallel_size}\t{result.token_count}"
            f"\t{result.hidden_size}\t{result.baseline_us:.3f}"
            f"\t{result.fused_us:.3f}\t{result.speedup:.3f}"
            f"\t{result.packed_exact_fraction:.8f}"
            f"\t{result.scale_exact_fraction:.8f}"
        )


if __name__ == "__main__":
    main()
