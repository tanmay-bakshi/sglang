import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
)

PAGE_SIZE = 64
KV_HEADS = 16
HEAD_DIM = 256
ELEMENT_SIZE_BYTES = 2


@dataclass(frozen=True, slots=True)
class TransferGeometry:
    """One production Gemma 4 KV-cache family.

    :ivar name: Human-readable cache family.
    :ivar layer_count: Number of layers using this cache family.
    :ivar page_count: Number of resident pages represented by the transfer.
    """

    name: str
    layer_count: int
    page_count: int

    @property
    def token_count(self) -> int:
        """Return the number of tokens represented by the pages.

        :returns: Transferred token count.
        """
        return self.page_count * PAGE_SIZE

    @property
    def byte_count(self) -> int:
        """Return the combined K and V payload size.

        :returns: Transfer payload bytes.
        """
        return (
            self.token_count
            * self.layer_count
            * KV_HEADS
            * HEAD_DIM
            * ELEMENT_SIZE_BYTES
            * 2
        )


@dataclass(slots=True)
class TransferBuffers:
    """Page-first host and layer-first device buffers for one geometry.

    :ivar host_k: Page-first pinned host K cache.
    :ivar host_v: Page-first pinned host V cache.
    :ivar device_k: Layer-first device K tensors.
    :ivar device_v: Layer-first device V tensors.
    :ivar host_indices: Host-cache token indices.
    :ivar device_indices: Device-cache token indices.
    :ivar flat_host: Contiguous pinned host ceiling buffer.
    :ivar flat_device: Contiguous device ceiling buffer.
    """

    host_k: torch.Tensor
    host_v: torch.Tensor
    device_k: list[torch.Tensor]
    device_v: list[torch.Tensor]
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    flat_host: torch.Tensor
    flat_device: torch.Tensor


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Wall-time result for one transfer path.

    :ivar family: Cache family.
    :ivar direction: ``h2d`` or ``d2h``.
    :ivar path: Batched page-first path or contiguous ceiling.
    :ivar median_ms: Median submission-through-completion wall time.
    :ivar bandwidth_gib_s: Effective payload bandwidth in GiB/s.
    """

    family: str
    direction: str
    path: str
    median_ms: float
    bandwidth_gib_s: float


PRODUCTION_GEOMETRIES = {
    "full": TransferGeometry(name="full", layer_count=10, page_count=128),
    "sliding": TransferGeometry(name="sliding", layer_count=50, page_count=16),
}


def _allocate_buffers(
    geometry: TransferGeometry,
    device: torch.device,
) -> TransferBuffers:
    """Allocate the exact page-first and layer-first transfer layouts.

    :param geometry: Cache-family geometry.
    :param device: CUDA device receiving the layer-first cache.
    :returns: Allocated transfer buffers and production-shaped indices.
    """
    host_shape = (
        geometry.page_count,
        geometry.layer_count,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
    )
    device_shape = (geometry.token_count, KV_HEADS, HEAD_DIM)
    host_k = torch.empty(host_shape, dtype=torch.bfloat16, pin_memory=True)
    host_v = torch.empty(host_shape, dtype=torch.bfloat16, pin_memory=True)
    device_k = [
        torch.empty(device_shape, dtype=torch.bfloat16, device=device)
        for _ in range(geometry.layer_count)
    ]
    device_v = [
        torch.empty(device_shape, dtype=torch.bfloat16, device=device)
        for _ in range(geometry.layer_count)
    ]
    host_indices = torch.arange(geometry.token_count, dtype=torch.int64)
    device_indices = host_indices.to(device)
    flat_host = torch.empty(
        (geometry.byte_count,),
        dtype=torch.uint8,
        pin_memory=True,
    )
    flat_device = torch.empty(
        (geometry.byte_count,),
        dtype=torch.uint8,
        device=device,
    )
    return TransferBuffers(
        host_k=host_k,
        host_v=host_v,
        device_k=device_k,
        device_v=device_v,
        host_indices=host_indices,
        device_indices=device_indices,
        flat_host=flat_host,
        flat_device=flat_device,
    )


def _measure_wall_ms(
    operation: Callable[[], None],
    warmup: int,
    repeats: int,
) -> float:
    """Measure submission-through-completion CUDA wall time.

    :param operation: Asynchronous transfer submission.
    :param warmup: Number of warm-up transfers.
    :param repeats: Number of measured transfers.
    :returns: Median elapsed wall time in milliseconds.
    """
    for _ in range(warmup):
        operation()
        torch.cuda.synchronize()

    durations_ms: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(durations_ms)


def _benchmark_geometry(
    geometry: TransferGeometry,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[TransferResult]:
    """Benchmark both directions at one production cache geometry.

    :param geometry: Cache-family geometry.
    :param device: CUDA device used for the transfer.
    :param warmup: Number of warm-up transfers.
    :param repeats: Number of measured transfers.
    :returns: Page-first and contiguous-ceiling results.
    """
    buffers = _allocate_buffers(geometry, device)
    transfer_stream = torch.cuda.Stream(device=device)

    def batched_d2h() -> None:
        with torch.cuda.stream(transfer_stream):
            transfer_kv_all_layer_direct_lf_pf(
                buffers.device_k + buffers.device_v,
                [buffers.host_k, buffers.host_v],
                buffers.device_indices,
                buffers.host_indices,
                PAGE_SIZE,
            )

    def batched_h2d() -> None:
        with torch.cuda.stream(transfer_stream):
            for layer_id in range(geometry.layer_count):
                transfer_kv_per_layer_direct_pf_lf(
                    [buffers.host_k, buffers.host_v],
                    [buffers.device_k[layer_id], buffers.device_v[layer_id]],
                    buffers.host_indices,
                    buffers.device_indices,
                    layer_id,
                    PAGE_SIZE,
                )

    def contiguous_d2h() -> None:
        with torch.cuda.stream(transfer_stream):
            buffers.flat_host.copy_(buffers.flat_device, non_blocking=True)

    def contiguous_h2d() -> None:
        with torch.cuda.stream(transfer_stream):
            buffers.flat_device.copy_(buffers.flat_host, non_blocking=True)

    operations = (
        ("d2h", "page_first_batch", batched_d2h),
        ("h2d", "page_first_batch", batched_h2d),
        ("d2h", "contiguous_ceiling", contiguous_d2h),
        ("h2d", "contiguous_ceiling", contiguous_h2d),
    )
    results: list[TransferResult] = []
    for direction, path, operation in operations:
        median_ms = _measure_wall_ms(operation, warmup, repeats)
        bandwidth_gib_s = geometry.byte_count / (1024**3) / (median_ms / 1000.0)
        results.append(
            TransferResult(
                family=geometry.name,
                direction=direction,
                path=path,
                median_ms=median_ms,
                bandwidth_gib_s=bandwidth_gib_s,
            )
        )
    return results


def _parse_families(raw_value: str) -> tuple[str, ...]:
    """Parse and validate comma-separated cache families.

    :param raw_value: Comma-separated family names.
    :returns: Validated family tuple.
    """
    families = tuple(value.strip() for value in raw_value.split(","))
    unsupported = set(families) - set(PRODUCTION_GEOMETRIES)
    if len(unsupported) > 0:
        raise ValueError(f"unsupported cache families: {sorted(unsupported)}")
    return families


def main(argv: Sequence[str] | None = None) -> None:
    """Benchmark Gemma 4 CPU KV-cache transfer geometry.

    :param argv: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="full,sliding")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    families = _parse_families(args.families)
    results = [
        result
        for family in families
        for result in _benchmark_geometry(
            PRODUCTION_GEOMETRIES[family],
            device,
            args.warmup,
            args.repeats,
        )
    ]

    print("family\tdirection\tpath\tmedian_ms\tbandwidth_gib_s")
    for result in results:
        print(
            f"{result.family}\t{result.direction}\t{result.path}"
            f"\t{result.median_ms:.3f}\t{result.bandwidth_gib_s:.3f}"
        )


if __name__ == "__main__":
    main()
