"""Production Gemma-4 fixture for raw-token streaming-session qualification."""

import contextlib
import subprocess
from dataclasses import dataclass
from typing import Any, ClassVar

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

GEMMA4_MODEL_PATH = "/models/gemma-4-31B-it-NVFP4"
GEMMA4_DFLASH_MODEL_PATH = "/models/gemma-4-31B-it-DFlash"
GEMMA4_SERVED_MODEL_NAME = "gemma-4-31B-it-NVFP4"
GEMMA4_PREFILL_GRAPH_BUCKETS = (
    1_024,
    2_048,
    3_072,
    4_096,
    5_120,
    6_144,
    7_168,
    8_192,
)
GEMMA4_SERVER_LAUNCH_TIMEOUT = min(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1_800.0)


@dataclass(frozen=True)
class Gemma4StreamingSessionArm:
    """One production Gemma-4 streaming-session qualification arm.

    :ivar name: Stable arm name used in receipts.
    :ivar port: HTTP listener port.
    :ivar chunked_prefill_size: Maximum tokens scheduled in one prefill chunk.
    :ivar use_dflash: Whether DFlash speculative decoding is enabled.
    """

    name: str
    port: int
    chunked_prefill_size: int
    use_dflash: bool


def build_gemma4_streaming_session_server_args(
    arm: Gemma4StreamingSessionArm,
) -> list[str]:
    """Build the exact production server arguments for one arm.

    :param arm: Qualification arm to launch.
    :returns: Ordered server command arguments.
    """
    args = [
        "--served-model-name",
        GEMMA4_SERVED_MODEL_NAME,
        "--load-format",
        "runai_streamer",
        "--quantization",
        "modelopt",
        "--base-gpu-id",
        "0",
        "--tp-size",
        "1",
        "--context-length",
        "131072",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "48",
        "--chunked-prefill-size",
        str(arm.chunked_prefill_size),
        "--page-size",
        "64",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-backend-prefill",
        "tc_piecewise",
        "--cuda-graph-bs-prefill",
        *[str(bucket) for bucket in GEMMA4_PREFILL_GRAPH_BUCKETS],
        "--attention-backend",
        "trtllm_mha",
        "--schedule-policy",
        "lpm",
        "--radix-eviction-policy",
        "lru",
        "--stream-interval",
        "1",
        "--enable-request-time-stats-logging",
        "--enable-metrics",
        "--enable-streaming-session",
        "--warmups",
        "streaming_session_small_extend",
        "--trust-remote-code",
        "--nccl-port",
        str(arm.port + 2_000),
        "--log-level",
        "info",
    ]
    if arm.use_dflash:
        args.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                GEMMA4_DFLASH_MODEL_PATH,
                "--speculative-draft-load-format",
                "auto",
                "--speculative-draft-model-quantization",
                "unquant",
                "--speculative-draft-attention-backend",
                "fa4",
                "--speculative-dflash-block-size",
                "16",
            ]
        )
    return args


def _assert_server_info_value(
    server_info: dict[str, Any],
    field: str,
    expected: object,
) -> None:
    """Assert one resolved server-info field.

    :param server_info: Live server-info response.
    :param field: Field to inspect.
    :param expected: Required value.
    """
    if field not in server_info:
        raise AssertionError(f"/server_info omitted required field {field!r}")
    actual = server_info[field]
    if actual != expected:
        raise AssertionError(
            f"/server_info field {field!r}: expected={expected!r}, actual={actual!r}"
        )


def assert_gemma4_streaming_session_server_info(
    server_info: dict[str, Any],
    arm: Gemma4StreamingSessionArm,
) -> None:
    """Attest one live server against its frozen production arm.

    :param server_info: Decoded server-info response.
    :param arm: Arm whose resolved configuration is required.
    """
    expected_fields: dict[str, object] = {
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
    }
    for field, expected in expected_fields.items():
        _assert_server_info_value(server_info, field, expected)

    warmups = server_info.get("warmups")
    if warmups not in (
        "streaming_session_small_extend",
        ["streaming_session_small_extend"],
    ):
        raise AssertionError(f"unexpected resolved warmups: {warmups!r}")

    speculative_algorithm = server_info.get("speculative_algorithm")
    if arm.use_dflash is False:
        if speculative_algorithm is not None:
            raise AssertionError(
                "non-spec arm unexpectedly enabled speculative decoding: "
                f"{speculative_algorithm!r}"
            )
        return

    dflash_fields: dict[str, object] = {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": GEMMA4_DFLASH_MODEL_PATH,
        "speculative_draft_load_format": "auto",
        "speculative_draft_model_quantization": None,
        "speculative_draft_attention_backend": "fa4",
        "speculative_dflash_block_size": 16,
    }
    for field, expected in dflash_fields.items():
        _assert_server_info_value(server_info, field, expected)


def _stop_gemma4_streaming_session_server(
    process: subprocess.Popen[bytes],
) -> None:
    """Stop and reap one complete server process tree.

    :param process: Launcher process whose descendants own the server.
    """
    kill_process_tree(process.pid, wait_timeout=120)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"server process {process.pid} did not exit after teardown"
        ) from error


class Gemma4StreamingSessionServerBase(CustomTestCase):
    """Launch-only fixture for one production Gemma-4 session arm."""

    arm: ClassVar[Gemma4StreamingSessionArm | None] = None
    base_url: ClassVar[str]
    process: ClassVar[subprocess.Popen[bytes]]
    server_info: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        """Launch and attest the configured production server."""
        arm = cls.arm
        if arm is None:
            raise RuntimeError(f"{cls.__name__} must define an arm")

        cls.base_url = f"http://127.0.0.1:{arm.port}"
        with contextlib.ExitStack() as cleanup:
            process = popen_launch_server(
                GEMMA4_MODEL_PATH,
                cls.base_url,
                timeout=GEMMA4_SERVER_LAUNCH_TIMEOUT,
                other_args=build_gemma4_streaming_session_server_args(arm),
            )
            cleanup.callback(_stop_gemma4_streaming_session_server, process)
            response = requests.get(cls.base_url + "/server_info", timeout=60)
            response.raise_for_status()
            server_info = response.json()
            if not isinstance(server_info, dict):
                raise AssertionError(
                    f"/server_info returned {type(server_info).__name__}, expected object"
                )
            assert_gemma4_streaming_session_server_info(server_info, arm)
            cls.process = process
            cls.server_info = server_info
            cleanup.pop_all()

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the server process tree and wait for process exit."""
        _stop_gemma4_streaming_session_server(cls.process)
