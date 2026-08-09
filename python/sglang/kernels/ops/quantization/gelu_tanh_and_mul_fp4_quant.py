import torch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op
from tvm_ffi.module import Module


def _round_up(value: int, multiple: int) -> int:
    """Round an integer up to a positive multiple.

    :param value: Integer to round.
    :param multiple: Positive alignment.
    :returns: The least aligned integer greater than or equal to ``value``.
    """
    return ((value + multiple - 1) // multiple) * multiple


@cache_once
def _jit_module(use_pdl: bool) -> Module:
    """Compile the dense Gemma GeGLU-to-NVFP4 operator.

    :param use_pdl: Whether the launch participates in PDL scheduling.
    :returns: Loaded TVM-FFI kernel module.
    """
    template_args = make_cpp_args(use_pdl)
    return load_jit(
        "gelu_tanh_and_mul_fp4_quant",
        *template_args,
        cuda_files=["elementwise/gelu_tanh_and_mul_fp4_quant.cuh"],
        cuda_wrappers=[
            (
                "gelu_tanh_and_mul_fp4_quant",
                f"GeluTanhMulFP4QuantKernel<{template_args}>::run",
            )
        ],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP4"],
        extra_dependencies=["flashinfer"],
    )


@register_custom_op(mutates_args=["output", "output_scale"])
def _gelu_tanh_and_mul_fp4_quant_inplace(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    """Launch fused GeGLU and ModelOpt-compatible NVFP4 quantization.

    :param input: Contiguous ``[M, 2 * H]`` BF16 gate/up tensor.
    :param global_scale: Scalar ModelOpt inverse activation scale.
    :param output: Packed FP4 output buffer.
    :param output_scale: Swizzled E4M3 block-scale output buffer.
    """
    module = _jit_module(is_arch_support_pdl())
    module.gelu_tanh_and_mul_fp4_quant(input, global_scale, output, output_scale)


def gelu_tanh_and_mul_fp4_quant(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Gemma's GeGLU activation with downstream NVFP4 quantization.

    The activation result is rounded to BF16 before FP4 conversion, matching
    the unfused ``gelu_tanh_and_mul`` followed by ``fp4_quantize`` path.

    :param input: Contiguous ``[M, 2 * H]`` BF16 gate/up tensor.
    :param global_scale: Scalar ModelOpt inverse activation scale.
    :returns: Packed FP4 activations and 128x4-swizzled E4M3 block scales.
    :raises ValueError: If the input does not satisfy the production ABI.
    """
    if input.dim() != 2:
        raise ValueError("fused GeGLU FP4 quantization requires a 2D input")
    if input.dtype != torch.bfloat16:
        raise ValueError("fused GeGLU FP4 quantization requires BF16 input")
    if not input.is_contiguous():
        raise ValueError("fused GeGLU FP4 quantization requires contiguous input")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("global_scale must be a scalar float32 tensor")

    token_count, fused_width = input.shape
    if fused_width % 2 != 0:
        raise ValueError("input must contain equal gate and up halves")
    hidden_size = fused_width // 2
    if hidden_size % 16 != 0:
        raise ValueError("hidden size must be divisible by 16")

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
    _gelu_tanh_and_mul_fp4_quant_inplace(
        input,
        global_scale,
        output,
        output_scale,
    )
    return output, output_scale


__all__ = ["gelu_tanh_and_mul_fp4_quant"]
