import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.layernorm import (
    RMSNorm,
    _flashinfer_add_rmsnorm_fp4quant_fake,
    _flashinfer_add_rmsnorm_fp4quant_impl,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRMSNormFp4Fusion(unittest.TestCase):
    """Validate the prequantized output contract without requiring a GPU."""

    def test_fake_outputs_match_swizzled_layout(self) -> None:
        """The compile-time fake should describe FlashInfer's output layout."""
        x = torch.empty((17, 5376), dtype=torch.bfloat16)
        residual = torch.empty_like(x)
        weight = torch.empty(5376, dtype=torch.bfloat16)
        global_scale = torch.ones(1, dtype=torch.float32)

        packed, scales = _flashinfer_add_rmsnorm_fp4quant_fake(
            x, residual, weight, global_scale, 1e-5
        )

        self.assertEqual(packed.shape, (17, 2688))
        self.assertEqual(packed.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(scales.shape, (43008,))
        self.assertEqual(scales.dtype, torch.float8_e4m3fn)

    def test_prequantized_contract_and_kernel_arguments(self) -> None:
        """The fused path should preserve residual and FP4 consumer semantics."""
        x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
        residual = torch.ones_like(x)
        expected_residual = x + residual
        global_scale = torch.tensor(0.75, dtype=torch.float32)
        packed = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
        scales = torch.empty(512, dtype=torch.float8_e4m3fn)
        norm = RMSNorm(16, eps=1e-5, weight_dtype=torch.bfloat16)
        calls: list[dict[str, object]] = []

        def fake_add_rmsnorm_fp4quant(
            input_tensor: torch.Tensor,
            residual_tensor: torch.Tensor,
            weight: torch.Tensor,
            *,
            global_scale: torch.Tensor,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Record the opaque operator contract.

            :param input_tensor: Input activation.
            :param residual_tensor: Residual updated in place.
            :param weight: RMSNorm weight.
            :param global_scale: ModelOpt activation scale.
            :param eps: RMSNorm epsilon.
            :returns: Fixed packed values and scales.
            """
            residual_tensor.add_(input_tensor)
            calls.append(
                {
                    "input": input_tensor,
                    "residual": residual_tensor,
                    "weight": weight,
                    "global_scale": global_scale,
                    "eps": eps,
                }
            )
            return packed, scales

        with patch(
            "sglang.srt.layers.layernorm._flashinfer_add_rmsnorm_fp4quant",
            new=fake_add_rmsnorm_fp4quant,
        ):
            quantized, residual_out = norm.forward_with_nvfp4_quant_fusion(
                x, residual, global_scale
            )

        packed_out, scales_out = quantized
        self.assertTrue(torch.equal(packed_out, packed))
        self.assertEqual(packed_out.dtype, torch.uint8)
        self.assertIs(scales_out, scales)
        self.assertIs(residual_out, residual)
        self.assertTrue(torch.equal(residual_out, expected_residual))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["input"], x)
        self.assertIs(calls[0]["residual"], residual)
        weight_argument = calls[0]["weight"]
        self.assertIsInstance(weight_argument, torch.Tensor)
        assert isinstance(weight_argument, torch.Tensor)
        self.assertEqual(weight_argument.data_ptr(), norm.weight.data_ptr())
        global_scale_argument = calls[0]["global_scale"]
        self.assertIsInstance(global_scale_argument, torch.Tensor)
        assert isinstance(global_scale_argument, torch.Tensor)
        self.assertEqual(global_scale_argument.shape, (1,))
        self.assertEqual(global_scale_argument.data_ptr(), global_scale.data_ptr())
        self.assertEqual(calls[0]["eps"], 1e-5)

    def test_flashinfer_kernel_arguments(self) -> None:
        """The opaque implementation should select the ModelOpt FP4 layout."""
        x = torch.empty((2, 16), dtype=torch.bfloat16)
        residual = torch.empty_like(x)
        weight = torch.empty(16, dtype=torch.bfloat16)
        global_scale = torch.tensor([0.75], dtype=torch.float32)
        packed = torch.empty((2, 8), dtype=torch.uint8)
        scales = torch.empty(512, dtype=torch.float8_e4m3fn)
        calls: list[dict[str, object]] = []

        def fake_add_rmsnorm_fp4quant(
            input_tensor: torch.Tensor,
            residual_tensor: torch.Tensor,
            weight_tensor: torch.Tensor,
            *,
            global_scale: torch.Tensor,
            eps: float,
            block_size: int,
            scale_format: str,
            is_sf_swizzled_layout: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Record the FlashInfer kernel arguments.

            :param input_tensor: Input activation.
            :param residual_tensor: Residual updated in place.
            :param weight_tensor: RMSNorm weight.
            :param global_scale: ModelOpt activation scale.
            :param eps: RMSNorm epsilon.
            :param block_size: FP4 quantization block size.
            :param scale_format: Scale-factor encoding.
            :param is_sf_swizzled_layout: Whether scales use the GEMM layout.
            :returns: Fixed packed values and scales.
            """
            calls.append(
                {
                    "input": input_tensor,
                    "residual": residual_tensor,
                    "weight": weight_tensor,
                    "global_scale": global_scale,
                    "eps": eps,
                    "block_size": block_size,
                    "scale_format": scale_format,
                    "is_sf_swizzled_layout": is_sf_swizzled_layout,
                }
            )
            return packed, scales

        with patch(
            "sglang.srt.layers.layernorm._get_flashinfer_add_rmsnorm_fp4quant",
            return_value=fake_add_rmsnorm_fp4quant,
        ):
            result = _flashinfer_add_rmsnorm_fp4quant_impl(
                x, residual, weight, global_scale, 1e-5
            )

        self.assertIs(result[0], packed)
        self.assertIs(result[1], scales)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["eps"], 1e-5)
        self.assertEqual(calls[0]["block_size"], 16)
        self.assertEqual(calls[0]["scale_format"], "e4m3")
        self.assertTrue(calls[0]["is_sf_swizzled_layout"])


if __name__ == "__main__":
    unittest.main()
