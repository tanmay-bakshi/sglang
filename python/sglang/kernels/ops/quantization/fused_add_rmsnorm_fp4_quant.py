import torch
from tvm_ffi.module import Module

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op


def _round_up(value: int, multiple: int) -> int:
    """Round an integer up to a positive multiple.

    :param value: Integer to round.
    :param multiple: Positive alignment.
    :returns: The least aligned integer greater than or equal to ``value``.
    """

    return ((value + multiple - 1) // multiple) * multiple


@cache_once
def _jit_module(use_pdl: bool) -> Module:
    """Compile the exact fused RMSNorm-to-NVFP4 operator.

    :param use_pdl: Whether the launch participates in PDL scheduling.
    :returns: Loaded TVM-FFI kernel module.
    """

    template_args = make_cpp_args(use_pdl)
    return load_jit(
        "fused_add_rmsnorm_fp4_quant",
        *template_args,
        cuda_files=["elementwise/fused_add_rmsnorm_fp4_quant.cuh"],
        cuda_wrappers=[
            (
                "fused_add_rmsnorm_fp4_quant",
                f"FusedAddRMSNormFP4QuantKernel<{template_args}>::run",
            )
        ],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP4"],
        extra_dependencies=["flashinfer_trtllm"],
    )


@register_custom_op(mutates_args=["residual", "output", "output_scale"])
def _fused_add_rmsnorm_fp4_quant_inplace(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    epsilon: float,
) -> None:
    """Launch exact residual-add, RMSNorm, and NVFP4 quantization.

    :param input: Contiguous token-major BF16 input.
    :param residual: Contiguous BF16 residual updated in place.
    :param weight: BF16 RMSNorm weight.
    :param global_scale: Scalar ModelOpt inverse activation scale.
    :param output: Packed FP4 output buffer.
    :param output_scale: Swizzled E4M3 block-scale output buffer.
    :param epsilon: RMSNorm epsilon.
    """

    module = _jit_module(is_arch_support_pdl())
    module.fused_add_rmsnorm_fp4_quant(
        input,
        residual,
        weight,
        global_scale,
        output,
        output_scale,
        epsilon,
    )


def fused_add_rmsnorm_fp4_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Gemma's dense-MLP input normalization and NVFP4 quantization.

    The residual sum and normalized activation are rounded at the same BF16
    boundaries as the incumbent two-kernel path. This makes both packed FP4
    values and swizzled scale bytes exact, while avoiding the normalized BF16
    global-memory round trip.

    :param input: Contiguous ``[M, H]`` BF16 input.
    :param residual: Contiguous ``[M, H]`` BF16 residual updated in place.
    :param weight: Contiguous ``[H]`` BF16 RMSNorm weight.
    :param global_scale: Scalar ModelOpt inverse activation scale.
    :param epsilon: RMSNorm epsilon.
    :returns: Packed FP4 activations and 128x4-swizzled E4M3 block scales.
    :raises ValueError: If a tensor does not satisfy the production ABI.
    """

    if input.dim() != 2 or residual.dim() != 2:
        raise ValueError("fused RMSNorm FP4 quantization requires 2D inputs")
    if input.shape != residual.shape:
        raise ValueError("input and residual must have identical shapes")
    if input.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("input and residual must use BF16")
    if weight.dtype != torch.bfloat16 or weight.dim() != 1:
        raise ValueError("weight must be a one-dimensional BF16 tensor")
    if weight.shape[0] != input.shape[1]:
        raise ValueError("weight width must match the hidden dimension")
    if not input.is_contiguous() or not residual.is_contiguous():
        raise ValueError("input and residual must be contiguous")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if input.device != residual.device or input.device != weight.device:
        raise ValueError("input, residual, and weight must share a device")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("global_scale must be a scalar float32 tensor")
    if global_scale.device != input.device:
        raise ValueError("global_scale must share the input device")

    token_count, hidden_size = input.shape
    if hidden_size == 0 or hidden_size % 16 != 0:
        raise ValueError("hidden_size must be a positive multiple of 16")
    if hidden_size > 16384:
        raise ValueError("hidden_size must not exceed 16384")

    output = torch.empty(
        (token_count, hidden_size // 2),
        dtype=torch.uint8,
        device=input.device,
    )
    output_scale = torch.empty(
        (_round_up(token_count, 128), _round_up(hidden_size // 16, 4)),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    if token_count == 0:
        return output, output_scale

    _fused_add_rmsnorm_fp4_quant_inplace(
        input,
        residual,
        weight,
        global_scale,
        output,
        output_scale,
        epsilon,
    )
    return output, output_scale


__all__ = ["fused_add_rmsnorm_fp4_quant"]
