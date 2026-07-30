"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    adjust_hybrid_swa_layer_ids,
    compute_attention_and_moe_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))


class TestAdjustHybridSWALayerIds(unittest.TestCase):
    """Validate hybrid-attention registration against PP layer ownership."""

    def test_excludes_exclusive_pipeline_boundary(self) -> None:
        """The next pipeline stage's first layer is never registered locally."""

        full_attention_layer_ids = list(range(5, 60, 6))
        swa_attention_layer_ids = [
            layer_id
            for layer_id in range(60)
            if layer_id not in full_attention_layer_ids
        ]

        for end_layer in (30, 35):
            with self.subTest(end_layer=end_layer):
                model_config = SimpleNamespace(
                    is_deepseek_v4_arch=False,
                    full_attention_layer_ids=full_attention_layer_ids.copy(),
                    swa_attention_layer_ids=swa_attention_layer_ids.copy(),
                )

                adjust_hybrid_swa_layer_ids(
                    model_config=model_config,
                    start_layer=0,
                    end_layer=end_layer,
                    is_hybrid_swa=True,
                )

                registered_layer_ids = sorted(
                    model_config.full_attention_layer_ids
                    + model_config.swa_attention_layer_ids
                )
                self.assertEqual(registered_layer_ids, list(range(end_layer)))


if __name__ == "__main__":
    unittest.main()
