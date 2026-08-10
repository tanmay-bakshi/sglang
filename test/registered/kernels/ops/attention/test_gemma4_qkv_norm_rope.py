import sys

import pytest
import torch
from sglang.kernels.ops.attention.gemma4_qkv_norm_rope import (
    gemma4_qkv_norm_rope,
)
from sglang.kernels.ops.attention.rope import (
    apply_rope_with_cos_sin_cache_inplace,
)
from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 256
CONTEXT_LENGTH = 8192


def _build_rope_cache(rope_type: str, device: torch.device) -> torch.Tensor:
    """Build the Gemma 4 FP32 cosine/sine cache.

    :param rope_type: ``sliding`` or ``proportional``.
    :param device: CUDA device receiving the cache.
    :returns: Contiguous ``[context_length, head_dim]`` cache.
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


def _mismatch_summary(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> str:
    """Summarize bitwise mismatches by QKV region.

    :param baseline: Incumbent QKV output.
    :param candidate: Fused QKV output.
    :param q_size: Flattened query width.
    :param kv_size: Flattened key or value width.
    :returns: Compact mismatch summary for an assertion failure.
    """
    baseline_parts = baseline.split((q_size, kv_size, kv_size), dim=-1)
    candidate_parts = candidate.split((q_size, kv_size, kv_size), dim=-1)
    counts = [
        int(torch.count_nonzero(lhs != rhs).item())
        for lhs, rhs in zip(baseline_parts, candidate_parts, strict=True)
    ]
    return f"Q={counts[0]}, K={counts[1]}, V={counts[2]}"


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("token_count", [1, 1024])
@pytest.mark.parametrize("rope_type", ["sliding", "proportional"])
@torch.inference_mode()
def test_fused_qkv_norm_rope_is_bitwise_exact(
    tensor_parallel_size: int,
    token_count: int,
    rope_type: str,
) -> None:
    """Match the production RMSNorm and RoPE sequence bit-for-bit.

    :param tensor_parallel_size: Tensor-parallel degree.
    :param token_count: Flattened token count.
    :param rope_type: RoPE cache family.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    num_q_heads = 32 // tensor_parallel_size
    num_kv_heads = 16 // tensor_parallel_size
    q_size = num_q_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM

    generator = torch.Generator(device=device)
    generator.manual_seed(20260729 + tensor_parallel_size * 10000 + token_count)
    source = torch.randn(
        (token_count, q_size + 2 * kv_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_weight = torch.randn(
        (HEAD_DIM,), dtype=torch.bfloat16, device=device, generator=generator
    )
    k_weight = torch.randn(
        (HEAD_DIM,), dtype=torch.bfloat16, device=device, generator=generator
    )
    positions = (
        torch.arange(token_count, dtype=torch.int64, device=device) + 17
    ) % CONTEXT_LENGTH
    cache = _build_rope_cache(rope_type, device)

    baseline = source.clone()
    baseline_q, baseline_k, baseline_v = baseline.split(
        (q_size, kv_size, kv_size), dim=-1
    )
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

    candidate = source.clone()
    gemma4_qkv_norm_rope(
        candidate,
        q_weight,
        k_weight,
        cache,
        positions,
        num_q_heads,
        num_kv_heads,
        1e-6,
    )

    assert torch.equal(candidate, baseline), _mismatch_summary(
        baseline, candidate, q_size, kv_size
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
