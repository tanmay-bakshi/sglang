import logging
import traceback

import torch
from tvm_ffi.module import Module

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_HEAD_DIM = 256


@cache_once
def _jit_gemma4_qkv_norm_rope_module(position_dtype: torch.dtype) -> Module:
    """Build the Gemma 4 QKV normalization and RoPE specialization.

    :param position_dtype: Integer dtype used by the position tensor.
    :returns: Loaded JIT module for the requested position dtype.
    """
    arguments = make_cpp_args(position_dtype, _HEAD_DIM, is_arch_support_pdl())
    return load_jit(
        "gemma4_qkv_norm_rope",
        *arguments,
        cuda_files=["elementwise/gemma4_qkv_norm_rope.cuh"],
        cuda_wrappers=[
            (
                "gemma4_qkv_norm_rope",
                f"gemma4_qkv_norm_rope<{arguments}>",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_gemma4_qkv_norm_rope(
    position_dtype: torch.dtype,
    activation_dtype: torch.dtype,
    head_dim: int,
) -> bool:
    """Return whether the exact Gemma 4 fused path is available.

    :param position_dtype: Integer dtype used by the position tensor.
    :param activation_dtype: QKV activation dtype.
    :param head_dim: Per-head QKV width.
    :returns: Whether the specialization compiled successfully.
    """
    if position_dtype not in (torch.int32, torch.int64):
        return False
    if activation_dtype != torch.bfloat16 or head_dim != _HEAD_DIM:
        return False

    try:
        _jit_gemma4_qkv_norm_rope_module(position_dtype)
    except Exception:
        logger.warning(
            "Failed to build the Gemma 4 fused QKV normalization and RoPE kernel:\n%s",
            traceback.format_exc(),
        )
        return False
    return True


@register_custom_op(
    op_name="gemma4_qkv_norm_rope_out",
    mutates_args=["qkv"],
)
def gemma4_qkv_norm_rope_out(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> None:
    """Normalize Q/K/V and rotate Q/K in place over a fused QKV row.

    The kernel preserves the incumbent BF16 rounding point between RMSNorm and
    RoPE. The FP32 RoPE cache encodes both standard sliding attention and Gemma
    4 proportional cross-mixing without a separate arithmetic mode.

    :param qkv: Contiguous BF16 tensor shaped ``[tokens, qkv_width]``.
    :param q_weight: Per-head Q RMSNorm weight.
    :param k_weight: Per-head K RMSNorm weight.
    :param cos_sin_cache: FP32 Gemma 4 RoPE cache.
    :param positions: Per-token position indices.
    :param num_q_heads: Tensor-parallel query head count.
    :param num_kv_heads: Tensor-parallel key and value head count.
    :param eps: RMSNorm epsilon.
    """
    module = _jit_gemma4_qkv_norm_rope_module(positions.dtype)
    module.gemma4_qkv_norm_rope(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_q_heads,
        num_kv_heads,
        eps,
    )


def gemma4_qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> None:
    """Apply the fused Gemma 4 attention post-projection transform.

    :param qkv: Contiguous BF16 tensor shaped ``[tokens, qkv_width]``.
    :param q_weight: Per-head Q RMSNorm weight.
    :param k_weight: Per-head K RMSNorm weight.
    :param cos_sin_cache: FP32 Gemma 4 RoPE cache.
    :param positions: Per-token position indices.
    :param num_q_heads: Tensor-parallel query head count.
    :param num_kv_heads: Tensor-parallel key and value head count.
    :param eps: RMSNorm epsilon.
    """
    gemma4_qkv_norm_rope_out(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_q_heads,
        num_kv_heads,
        eps,
    )
