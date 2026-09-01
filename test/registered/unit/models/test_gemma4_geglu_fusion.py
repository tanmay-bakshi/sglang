import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.gemma4_causal import _should_use_fused_geglu_fp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma4GeGLUFusion(unittest.TestCase):
    def test_non_speculative_extend_remains_eligible(self) -> None:
        self.assertTrue(
            _should_use_fused_geglu_fp4(
                ForwardMode.EXTEND,
                num_tokens=1,
                enable_verify=False,
            )
        )

    def test_target_verify_requires_flag_and_minimum_batch(self) -> None:
        self.assertFalse(
            _should_use_fused_geglu_fp4(
                ForwardMode.TARGET_VERIFY,
                num_tokens=17,
                enable_verify=False,
            )
        )
        self.assertFalse(
            _should_use_fused_geglu_fp4(
                ForwardMode.TARGET_VERIFY,
                num_tokens=15,
                enable_verify=True,
            )
        )
        self.assertTrue(
            _should_use_fused_geglu_fp4(
                ForwardMode.TARGET_VERIFY,
                num_tokens=16,
                enable_verify=True,
            )
        )

    def test_decode_never_uses_fusion(self) -> None:
        self.assertFalse(
            _should_use_fused_geglu_fp4(
                ForwardMode.DECODE,
                num_tokens=64,
                enable_verify=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
