import unittest

from sglang.srt.entrypoints.warmup import GEMMA4_STREAMING_SESSION_WARMUPS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.server_fixtures.gemma4_streaming_session_fixture import (
    GEMMA4_DFLASH_MODEL_PATH,
    GEMMA4_MODEL_PATH,
    GEMMA4_PREFILL_GRAPH_BUCKETS,
    GEMMA4_SERVED_MODEL_NAME,
    Gemma4StreamingSessionArm,
    assert_gemma4_streaming_session_server_info,
    build_gemma4_streaming_session_server_args,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _argument_value(arguments: list[str], flag: str) -> str:
    """Return the single value following one command flag.

    :param arguments: Command arguments to inspect.
    :param flag: Flag whose value is required.
    :returns: Value immediately following the flag.
    """
    index = arguments.index(flag)
    return arguments[index + 1]


def _server_info(arm: Gemma4StreamingSessionArm) -> dict[str, object]:
    """Build one exact resolved server-info fixture.

    :param arm: Qualification arm represented by the response.
    :returns: Exact fields consumed by production attestation.
    """
    info: dict[str, object] = {
        "model_path": GEMMA4_MODEL_PATH,
        "served_model_name": GEMMA4_SERVED_MODEL_NAME,
        "load_format": "runai_streamer",
        "quantization": "modelopt_fp4",
        "tp_size": 1,
        "context_length": 131_072,
        "mem_fraction_static": 0.7,
        "max_running_requests": 48,
        "chunked_prefill_size": arm.chunked_prefill_size,
        "page_size": 64,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "tc_piecewise",
        "cuda_graph_bs_prefill": list(GEMMA4_PREFILL_GRAPH_BUCKETS),
        "attention_backend": "trtllm_mha",
        "schedule_policy": "lpm",
        "radix_eviction_policy": "lru",
        "stream_interval": 1,
        "enable_request_time_stats_logging": True,
        "enable_metrics": True,
        "enable_streaming_session": True,
        "warmups": list(GEMMA4_STREAMING_SESSION_WARMUPS),
        "speculative_algorithm": None,
    }
    if arm.use_dflash:
        info.update(
            {
                "speculative_algorithm": "DFLASH",
                "speculative_draft_model_path": GEMMA4_DFLASH_MODEL_PATH,
                "speculative_draft_load_format": "auto",
                "speculative_draft_model_quantization": None,
                "speculative_draft_attention_backend": "fa4",
                "speculative_dflash_block_size": 16,
            }
        )
    return info


class Gemma4StreamingSessionFixtureTest(unittest.TestCase):
    def test_dflash_arguments_pin_production_shape(self) -> None:
        arm = Gemma4StreamingSessionArm(
            name="dflash-chunked",
            port=32_311,
            chunked_prefill_size=1_024,
            use_dflash=True,
        )

        arguments = build_gemma4_streaming_session_server_args(arm)

        self.assertEqual(_argument_value(arguments, "--tp-size"), "1")
        self.assertEqual(_argument_value(arguments, "--context-length"), "131072")
        self.assertEqual(_argument_value(arguments, "--page-size"), "64")
        self.assertEqual(
            _argument_value(arguments, "--chunked-prefill-size"),
            "1024",
        )
        self.assertEqual(
            _argument_value(arguments, "--speculative-draft-model-path"),
            GEMMA4_DFLASH_MODEL_PATH,
        )
        self.assertEqual(
            _argument_value(arguments, "--speculative-dflash-block-size"),
            "16",
        )
        graph_start = arguments.index("--cuda-graph-bs-prefill") + 1
        graph_end = arguments.index("--attention-backend")
        self.assertEqual(
            arguments[graph_start:graph_end],
            [str(bucket) for bucket in GEMMA4_PREFILL_GRAPH_BUCKETS],
        )
        self.assertIn("--enable-streaming-session", arguments)
        self.assertIn("--enable-metrics", arguments)
        self.assertEqual(
            _argument_value(arguments, "--warmups"),
            ",".join(GEMMA4_STREAMING_SESSION_WARMUPS),
        )

    def test_no_spec_arguments_exclude_every_dflash_flag(self) -> None:
        arm = Gemma4StreamingSessionArm(
            name="no-spec-one-shot",
            port=32_312,
            chunked_prefill_size=16_384,
            use_dflash=False,
        )

        arguments = build_gemma4_streaming_session_server_args(arm)

        self.assertNotIn("--speculative-algorithm", arguments)
        self.assertFalse(
            any(argument.startswith("--speculative-") for argument in arguments)
        )

    def test_attestation_accepts_exact_resolved_values(self) -> None:
        arm = Gemma4StreamingSessionArm(
            name="dflash-one-shot",
            port=32_310,
            chunked_prefill_size=16_384,
            use_dflash=True,
        )

        assert_gemma4_streaming_session_server_info(_server_info(arm), arm)

    def test_attestation_rejects_configuration_drift(self) -> None:
        arm = Gemma4StreamingSessionArm(
            name="no-spec-chunked",
            port=32_313,
            chunked_prefill_size=1_024,
            use_dflash=False,
        )
        server_info = _server_info(arm)
        server_info["chunked_prefill_size"] = 16_384

        with self.assertRaisesRegex(
            AssertionError,
            "chunked_prefill_size",
        ):
            assert_gemma4_streaming_session_server_info(server_info, arm)


if __name__ == "__main__":
    unittest.main()
