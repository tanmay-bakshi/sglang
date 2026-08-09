import ast
import unittest
from pathlib import Path

from sglang.srt.layers.quantization import gemma4_prefill_bf16
from sglang.srt.layers.quantization.gemma4_prefill_bf16 import (
    GEMMA4_TP2_PCG_PREFILL_BF16_TACTICS,
    GEMMA4_TP2_PCG_PREFILL_M_BUCKETS,
    TGV_TACTIC_COUNT,
    Gemma4PrefillBf16Shape,
    Gemma4PrefillBf16Tactic,
    StaticGemma4PrefillBf16TacticCache,
    gemma4_prefill_bf16_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGemma4PrefillBf16(unittest.TestCase):
    def test_production_pcg_buckets_resolve_exact_tactic(self) -> None:
        for m in GEMMA4_TP2_PCG_PREFILL_M_BUCKETS:
            with self.subTest(m=m):
                shape = Gemma4PrefillBf16Shape(
                    m=m,
                    n=8192,
                    k=5376,
                    has_bias=False,
                )
                self.assertEqual(
                    GEMMA4_TP2_PCG_PREFILL_BF16_TACTICS.resolve(
                        shape,
                        x_is_contiguous=True,
                        weight_is_contiguous=True,
                        bias_is_contiguous=True,
                    ),
                    24,
                )

    def test_shape_derivation_flattens_prefix_dimensions(self) -> None:
        self.assertEqual(
            gemma4_prefill_bf16_shape(
                (2, 512, 5376),
                (8192, 5376),
                has_bias=False,
            ),
            Gemma4PrefillBf16Shape(
                m=1024,
                n=8192,
                k=5376,
                has_bias=False,
            ),
        )

    def test_shape_derivation_rejects_incompatible_linear_abi(self) -> None:
        cases = (
            ((), (8192, 5376)),
            ((0, 5376), (8192, 5376)),
            ((1024, 0), (8192, 0)),
            ((1024, 5376), (8192,)),
            ((1024, 5376), (8192, 4096)),
        )
        for input_shape, weight_shape in cases:
            with self.subTest(
                input_shape=input_shape,
                weight_shape=weight_shape,
            ):
                self.assertIsNone(
                    gemma4_prefill_bf16_shape(
                        input_shape,
                        weight_shape,
                        has_bias=False,
                    )
                )

    def test_cache_falls_back_for_every_non_exact_shape_axis(self) -> None:
        cases = (
            Gemma4PrefillBf16Shape(2049, 8192, 5376, False),
            Gemma4PrefillBf16Shape(2048, 8193, 5376, False),
            Gemma4PrefillBf16Shape(2048, 8192, 5377, False),
            Gemma4PrefillBf16Shape(2048, 8192, 5376, True),
        )
        for shape in cases:
            with self.subTest(shape=shape):
                self.assertIsNone(
                    GEMMA4_TP2_PCG_PREFILL_BF16_TACTICS.resolve(
                        shape,
                        x_is_contiguous=True,
                        weight_is_contiguous=True,
                        bias_is_contiguous=True,
                    )
                )

    def test_cache_falls_back_for_incompatible_layout(self) -> None:
        shape = Gemma4PrefillBf16Shape(2048, 8192, 5376, False)
        layouts = (
            (False, True, True),
            (True, False, True),
        )
        for x_contiguous, weight_contiguous, bias_contiguous in layouts:
            with self.subTest(
                x_contiguous=x_contiguous,
                weight_contiguous=weight_contiguous,
                bias_contiguous=bias_contiguous,
            ):
                self.assertIsNone(
                    GEMMA4_TP2_PCG_PREFILL_BF16_TACTICS.resolve(
                        shape,
                        x_is_contiguous=x_contiguous,
                        weight_is_contiguous=weight_contiguous,
                        bias_is_contiguous=bias_contiguous,
                    )
                )

    def test_biased_entry_requires_contiguous_bias(self) -> None:
        shape = Gemma4PrefillBf16Shape(1024, 8192, 5376, True)
        cache = StaticGemma4PrefillBf16TacticCache(
            (Gemma4PrefillBf16Tactic(shape, 24),)
        )
        self.assertIsNone(
            cache.resolve(
                shape,
                x_is_contiguous=True,
                weight_is_contiguous=True,
                bias_is_contiguous=False,
            )
        )

    def test_cache_rejects_duplicate_shape_entries(self) -> None:
        shape = Gemma4PrefillBf16Shape(1024, 8192, 5376, False)
        with self.assertRaisesRegex(ValueError, "duplicate BF16 tactic entry"):
            StaticGemma4PrefillBf16TacticCache(
                (
                    Gemma4PrefillBf16Tactic(shape, 23),
                    Gemma4PrefillBf16Tactic(shape, 24),
                )
            )

    def test_entries_reject_invalid_dimensions_and_tactics(self) -> None:
        with self.assertRaisesRegex(ValueError, "m must be positive"):
            Gemma4PrefillBf16Shape(0, 8192, 5376, False)
        shape = Gemma4PrefillBf16Shape(1024, 8192, 5376, False)
        for tactic in (-1, TGV_TACTIC_COUNT):
            with (
                self.subTest(tactic=tactic),
                self.assertRaisesRegex(
                    ValueError,
                    "tactic must be in",
                ),
            ):
                Gemma4PrefillBf16Tactic(shape, tactic)

    def test_dispatch_table_has_no_gpu_runtime_import(self) -> None:
        module_path = Path(gemma4_prefill_bf16.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        self.assertTrue(
            imported_roots.isdisjoint({"cuda", "cutlass", "torch"}),
            imported_roots,
        )

    def test_tactic_count_matches_custom_op_abi(self) -> None:
        module_path = Path(gemma4_prefill_bf16.__file__)
        kernel_path = (
            module_path.parents[3] / "kernels" / "ops" / "gemm" / "cutedsl_bf16_gemm.py"
        )
        tree = ast.parse(kernel_path.read_text(encoding="utf-8"))
        tactic_values = None
        for node in tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if not isinstance(node.target, ast.Name):
                continue
            if node.target.id != "_TGV_CUTE_EXT_TACTIC_CONFIGS":
                continue
            tactic_values = ast.literal_eval(node.value)
            break
        self.assertIsNotNone(tactic_values)
        self.assertEqual(len(tactic_values), TGV_TACTIC_COUNT)


if __name__ == "__main__":
    unittest.main()
