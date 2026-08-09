from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_flashinfer_available,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import (
    register_custom_op,
    register_custom_op_from_extern,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


FusedAddRMSNormFp4Quantize = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
    tuple[torch.Tensor, torch.Tensor],
]


def _round_up(x: int, multiple: int) -> int:
    """Round an integer up to a positive multiple.

    :param x: Integer to round.
    :param multiple: Positive alignment.
    :returns: The least aligned integer greater than or equal to ``x``.
    """
    return ((x + multiple - 1) // multiple) * multiple


def _fused_add_rmsnorm_fp4_quantize_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe fused Gemma MLP input quantization to FakeTensor.

    :param input: Token-major activation tensor.
    :param residual: Residual tensor mutated by the real operator.
    :param weight: RMSNorm weight.
    :param global_scale: ModelOpt activation scale.
    :param eps: RMSNorm epsilon.
    :returns: Packed FP4 activations and 128x4-swizzled block scales.
    """
    del residual, weight, global_scale, eps
    if input.dim() != 2:
        raise ValueError("fused FP4 RMSNorm requires a two-dimensional input")
    token_count, hidden_size = input.shape
    scale_rows = _round_up(token_count, 128)
    scale_columns = _round_up(hidden_size // 16, 4)
    return (
        input.new_empty((token_count, hidden_size // 2), dtype=torch.uint8),
        input.new_empty((scale_rows, scale_columns), dtype=torch.float8_e4m3fn),
    )


fused_add_rmsnorm_fp4_quantize: FusedAddRMSNormFp4Quantize | None = None

if is_cuda() and is_sm100_supported() and is_flashinfer_available():
    from flashinfer import add_rmsnorm_fp4quant as _flashinfer_add_rmsnorm_fp4quant

    @register_custom_op(
        op_name="flashinfer_add_rmsnorm_fp4_quantize",
        mutates_args=["residual"],
        fake_impl=_fused_add_rmsnorm_fp4_quantize_fake,
    )
    def _fused_add_rmsnorm_fp4_quantize(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse residual addition, RMSNorm, and ModelOpt FP4 quantization.

        :param input: Token-major BF16 activation tensor.
        :param residual: Contiguous BF16 residual updated in place.
        :param weight: BF16 RMSNorm weight.
        :param global_scale: ModelOpt inverse activation scale.
        :param eps: RMSNorm epsilon.
        :returns: Packed FP4 activations and swizzled E4M3 block scales.
        :raises ValueError: If the tensors do not satisfy the kernel ABI.
        """
        if input.dim() != 2:
            raise ValueError("fused FP4 RMSNorm requires a two-dimensional input")
        if input.shape != residual.shape:
            raise ValueError("input and residual must have identical shapes")
        if not input.is_contiguous() or not residual.is_contiguous():
            raise ValueError("input and residual must be contiguous")
        if input.dtype != residual.dtype or input.dtype != weight.dtype:
            raise ValueError("input, residual, and weight must share a dtype")
        if input.shape[1] % 16 != 0:
            raise ValueError("the hidden dimension must be divisible by 16")

        output, block_scale = _flashinfer_add_rmsnorm_fp4quant(
            input=input,
            residual=residual,
            weight=weight,
            global_scale=global_scale,
            eps=eps,
            block_size=16,
            scale_format="e4m3",
            is_sf_swizzled_layout=True,
            enable_pdl=True,
        )
        token_count, hidden_size = input.shape
        scale_rows = _round_up(token_count, 128)
        scale_columns = _round_up(hidden_size // 16, 4)
        return output.view(torch.uint8), block_scale.view(scale_rows, scale_columns)

    fused_add_rmsnorm_fp4_quantize = _fused_add_rmsnorm_fp4_quantize


fp4_quantize = None
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
        )

    def _flashinfer_fp4_quantize_fake(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_column_major = input.stride(-2) == 1
        if is_column_major:
            m = input.shape[-1]
            K = input.shape[-2]
        else:
            m = input.numel() // input.shape[-1]
            K = input.shape[-1]
        if is_column_major:
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)
        else:
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            sf_rows = _round_up(m, row_size)
            sf_cols = _round_up(K // sf_vec_size, 4)
        else:
            sf_rows = m
            sf_cols = K // sf_vec_size
        if is_column_major:
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)
        else:
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)
        return x_q, sf

    fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
except ImportError:
    fp4_quantize = None


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    MARLIN = "marlin"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_marlin(self) -> bool:
        return self == Fp4GemmRunnerBackend.MARLIN

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:
            return "cute-dsl"
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend
    if backend == "auto":
        if is_sm100_supported():
            backend = "flashinfer_cutedsl"
        elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
            backend = "marlin"
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND
