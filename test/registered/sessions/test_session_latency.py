"""
Benchmark: Streaming Session Inter-Turn Latency

Tests:
  1. Stability (bs=8):  streaming only, assert tail_avg / head_avg <= 1.15
  2. Correctness (bs=1): regular vs streaming, assert output equal
  3. Random lengths (bs=8): streaming only, random input/output lens, no crash
"""

import random
import statistics
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from tabulate import tabulate

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")

NUM_TURNS = 150
INPUT_LEN = 16
GEN_LEN = 8
NUM_CONCURRENT = 8
HEAD_TURNS = 10
TAIL_TURNS = 10
SAMPLE_TURNS = 8

COLD_PREFILL_DEPTH = 40_000
COLD_PREFILL_SHORT_SAMPLES = 7
COLD_PREFILL_SHORT_INPUT_LEN = 16
COLD_PREFILL_P50_RATIO_LIMIT = 3.0

NUM_TURNS_RANDOM = 50
RANDOM_INPUT_LEN_RANGE = (8, 64)
RANDOM_OUTPUT_LEN_RANGE = (4, 32)

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 200

SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": GEN_LEN,
    "no_stop_trim": True,
    "skip_special_tokens": False,
    "ignore_eos": True,
}


@dataclass
class TurnResult:
    turn: int
    context_len: int
    cached_tokens: int
    client_latency_ms: float
    e2e_latency_ms: float


@dataclass
class ModeResult:
    mode: str
    turns: List[TurnResult] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def _generate_input_chunks(
    tokenizer, num_turns: int, input_len: int, offset: int = 0
) -> List[List[int]]:
    all_ids = tokenizer.encode(FILLER_TEXT)
    if all_ids and all_ids[0] == tokenizer.bos_token_id:
        all_ids = all_ids[1:]

    start = offset * num_turns * input_len
    needed = start + num_turns * input_len
    while len(all_ids) < needed:
        all_ids = all_ids + all_ids
    chunks = [
        all_ids[start + i * input_len : start + (i + 1) * input_len]
        for i in range(num_turns)
    ]

    if tokenizer.bos_token_id is not None:
        chunks[0] = [tokenizer.bos_token_id] + chunks[0]

    return chunks


def _generate_random_input_chunks(
    tokenizer,
    num_turns: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
    offset: int = 0,
) -> List[List[int]]:
    all_ids = tokenizer.encode(FILLER_TEXT)
    if all_ids and all_ids[0] == tokenizer.bos_token_id:
        all_ids = all_ids[1:]

    total_max = offset * num_turns * max_len + num_turns * max_len
    while len(all_ids) < total_max:
        all_ids = all_ids + all_ids

    chunks: List[List[int]] = []
    pos = offset * num_turns * max_len
    for i in range(num_turns):
        length = rng.randint(min_len, max_len)
        chunk = all_ids[pos : pos + length]
        pos += length
        chunks.append(chunk)

    if tokenizer.bos_token_id is not None:
        chunks[0] = [tokenizer.bos_token_id] + chunks[0]

    return chunks


def _send_generate(base_url: str, payload: dict) -> dict:
    resp = requests.post(base_url + "/generate", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Generate failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _record_turn(
    turn_idx: int, context_len: int, meta: dict, client_latency_ms: float
) -> TurnResult:
    return TurnResult(
        turn=turn_idx + 1,
        context_len=context_len,
        cached_tokens=meta["cached_tokens"],
        client_latency_ms=client_latency_ms,
        e2e_latency_ms=meta.get("e2e_latency", 0) * 1000,
    )


def _run_one_session(
    base_url: str,
    chunks: List[List[int]],
    streaming: bool = False,
    per_turn_gen_lens: Optional[List[int]] = None,
) -> ModeResult:
    mode = "streaming_session" if streaming else "regular_session"
    result = ModeResult(mode=mode)

    default_gen = GEN_LEN
    if per_turn_gen_lens is not None:
        max_gen = max(per_turn_gen_lens)
    else:
        max_gen = default_gen
    capacity = sum(len(c) for c in chunks) + len(chunks) * max_gen + 1024

    open_payload: dict = {"capacity_of_str_len": capacity}
    if streaming:
        open_payload["streaming"] = True
    session_id = requests.post(base_url + "/open_session", json=open_payload).json()

    rid = None
    context_len = 0

    for turn_idx, chunk_ids in enumerate(chunks):
        context_len += len(chunk_ids)

        if per_turn_gen_lens is not None:
            sp = {**SAMPLING_PARAMS, "max_new_tokens": per_turn_gen_lens[turn_idx]}
        else:
            sp = SAMPLING_PARAMS

        t0 = time.perf_counter()
        response = _send_generate(
            base_url,
            {
                "input_ids": chunk_ids,
                "session_params": {"id": session_id, "rid": rid},
                "sampling_params": sp,
            },
        )
        client_lat = (time.perf_counter() - t0) * 1000

        meta = response["meta_info"]
        rid = meta["id"]
        context_len += meta["completion_tokens"]

        result.turns.append(_record_turn(turn_idx, context_len, meta, client_lat))
        result.outputs.append(response["text"])

    requests.post(base_url + "/close_session", json={"session_id": session_id})
    return result


def _collect_latencies(
    results: List[ModeResult],
    last_n: Optional[int] = None,
    first_n: Optional[int] = None,
) -> List[float]:
    lats = []
    for r in results:
        if last_n is not None:
            turns = r.turns[-last_n:]
        elif first_n is not None:
            # Skip turn 1 (includes prefill), then take next `first_n` turns.
            turns = r.turns[1 : 1 + first_n]
        else:
            turns = r.turns[1:]  # skip turn 1
        lats.extend(t.client_latency_ms for t in turns)
    return lats


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_mode_table(result: ModeResult, label: str = ""):
    tag = f"{result.mode} ({label})" if label else result.mode
    print(f"\n  [{tag}]  {len(result.turns)} turns")

    n = len(result.turns)
    if n <= SAMPLE_TURNS * 2:
        indices = list(range(n))
    else:
        indices = list(range(SAMPLE_TURNS)) + [-1] + list(range(n - SAMPLE_TURNS, n))

    rows = []
    for idx in indices:
        if idx == -1:
            rows.append(["..."] * 5)
            continue
        t = result.turns[idx]
        rows.append(
            [
                t.turn,
                t.context_len,
                t.cached_tokens,
                f"{t.client_latency_ms:.1f}ms",
                f"{t.e2e_latency_ms:.1f}ms",
            ]
        )
    print(
        tabulate(
            rows,
            headers=["Turn", "Context", "Cached", "Client Lat", "E2E Lat"],
            colalign=("right",) * 5,
        )
    )


class TestSessionLatency(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "openai/gpt-oss-20b"
        cls.base_url = DEFAULT_URL_FOR_TEST
        # NOTE: Overlap scheduling commits KV cache one step ahead,
        # so the last decode token is cached (unlike non-overlap).
        # Disable overlap to keep session cache behavior consistent.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--enable-streaming-session",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-backend-prefill=disabled",
                "--page-size",
                "4",
                "--chunked-prefill-size",
                "2048",
            ],
        )
        cls.tokenizer = get_tokenizer(cls.model)

        requests.post(cls.base_url + "/flush_cache")
        _send_generate(
            cls.base_url,
            {
                "input_ids": cls.tokenizer.encode("Hello world"),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_concurrent_session(
        self,
        streaming: bool = False,
        num_concurrent: int = NUM_CONCURRENT,
        num_turns: int = NUM_TURNS,
        input_len: int = INPUT_LEN,
        per_turn_gen_lens: Optional[List[int]] = None,
        random_input_chunks: bool = False,
        rng: Optional[random.Random] = None,
    ) -> List[ModeResult]:
        requests.post(self.base_url + "/flush_cache")

        def run_one(session_idx):
            if random_input_chunks and rng is not None:
                per_session_rng = random.Random(rng.randint(0, 2**32) + session_idx)
                chunks = _generate_random_input_chunks(
                    self.tokenizer,
                    num_turns,
                    RANDOM_INPUT_LEN_RANGE[0],
                    RANDOM_INPUT_LEN_RANGE[1],
                    per_session_rng,
                    offset=session_idx,
                )
            else:
                chunks = _generate_input_chunks(
                    self.tokenizer, num_turns, input_len, offset=session_idx
                )
            return _run_one_session(
                self.base_url,
                chunks,
                streaming=streaming,
                per_turn_gen_lens=per_turn_gen_lens,
            )

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            return list(pool.map(run_one, range(num_concurrent)))

    def test_streaming_session(self):
        """Stability: streaming reuses KV across turns, so tail/head latency
        should stay flat. Skip turn 1 (prefill) when computing head."""
        results = self._run_concurrent_session(streaming=True)
        _print_mode_table(results[0], label="session 0")

        head_avg = _avg(_collect_latencies(results, first_n=HEAD_TURNS))
        tail_avg = _avg(_collect_latencies(results, last_n=TAIL_TURNS))
        ratio = tail_avg / head_avg if head_avg > 0 else float("inf")
        print(
            f"\n  streaming_session  "
            f"head_avg(first {HEAD_TURNS})={head_avg:.1f}ms  "
            f"tail_avg(last {TAIL_TURNS})={tail_avg:.1f}ms  "
            f"ratio={ratio:.2f}"
        )
        self.assertLessEqual(
            ratio,
            1.15,
            f"streaming latency should stay flat across turns "
            f"(head={head_avg:.1f}ms, tail={tail_avg:.1f}ms, ratio={ratio:.2f} > 1.15)",
        )

    def test_streaming_session_correctness(self):
        """Correctness test: bs=1, assert regular and streaming outputs match."""
        correctness_turns = 30
        reg = self._run_concurrent_session(
            streaming=False, num_concurrent=1, num_turns=correctness_turns
        )
        stm = self._run_concurrent_session(
            streaming=True, num_concurrent=1, num_turns=correctness_turns
        )

        _print_mode_table(reg[0], label="correctness regular")
        _print_mode_table(stm[0], label="correctness streaming")

        reg_out = reg[0].outputs
        stm_out = stm[0].outputs
        mismatches = sum(1 for a, b in zip(reg_out, stm_out) if a != b)
        self.assertEqual(
            mismatches,
            0,
            f"regular vs streaming (bs=1): {mismatches}/{len(reg_out)} turns differ",
        )

    def test_streaming_session_random_lengths(self):
        """Stress test: bs=8, streaming only, random input/output lens."""
        rng = random.Random(42)
        gen_lens = [
            rng.randint(*RANDOM_OUTPUT_LEN_RANGE) for _ in range(NUM_TURNS_RANDOM)
        ]

        results = self._run_concurrent_session(
            streaming=True,
            num_turns=NUM_TURNS_RANDOM,
            per_turn_gen_lens=gen_lens,
            random_input_chunks=True,
            rng=random.Random(42),
        )

        for i, r in enumerate(results):
            self.assertEqual(
                len(r.turns),
                NUM_TURNS_RANDOM,
                f"session {i}: expected {NUM_TURNS_RANDOM} turns, got {len(r.turns)}",
            )
        _print_mode_table(results[0], label="random streaming session 0")

    def test_chunked_cold_prefill_preserves_short_request_latency(self) -> None:
        """Bound short-request p50 while a 40k session prefill is in flight."""
        solo_latencies: list[float] = []
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        for sample_idx in range(COLD_PREFILL_SHORT_SAMPLES):
            latency, _ = self._timed_short_request(sample_idx)
            solo_latencies.append(latency)

        requests.post(self.base_url + "/flush_cache").raise_for_status()
        deep_input_ids = _generate_input_chunks(
            self.tokenizer,
            num_turns=1,
            input_len=COLD_PREFILL_DEPTH,
            offset=0,
        )[0]
        open_response = requests.post(
            self.base_url + "/open_session",
            json={
                "capacity_of_str_len": len(deep_input_ids) + 16,
                "streaming": True,
            },
        )
        open_response.raise_for_status()
        session_id = open_response.json()

        try:
            with ThreadPoolExecutor(
                max_workers=COLD_PREFILL_SHORT_SAMPLES + 1
            ) as pool:
                deep_future = pool.submit(
                    self._run_deep_session_prefill,
                    session_id,
                    deep_input_ids,
                )
                self._wait_for_session_inflight(session_id, deep_future)
                short_futures = [
                    pool.submit(self._timed_short_request, sample_idx + 1000)
                    for sample_idx in range(COLD_PREFILL_SHORT_SAMPLES)
                ]
                short_results = [future.result() for future in short_futures]
                deep_response, deep_finished_at = deep_future.result()
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            ).raise_for_status()

        concurrent_latencies = [latency for latency, _ in short_results]
        short_finished_at = [finished_at for _, finished_at in short_results]
        solo_p50 = statistics.median(solo_latencies)
        concurrent_p50 = statistics.median(concurrent_latencies)
        ratio = concurrent_p50 / solo_p50
        print(
            "\n  chunked_cold_prefill  "
            f"solo_p50={solo_p50:.1f}ms  "
            f"concurrent_p50={concurrent_p50:.1f}ms  ratio={ratio:.2f}"
        )

        self.assertGreaterEqual(
            deep_response["meta_info"]["prompt_tokens"],
            COLD_PREFILL_DEPTH,
        )
        self.assertLess(
            min(short_finished_at),
            deep_finished_at,
            "a short request must finish before the deep session prefill",
        )
        self.assertLess(
            ratio,
            COLD_PREFILL_P50_RATIO_LIMIT,
            f"short-request p50 regressed under cold prefill: {ratio:.2f}x",
        )

    def _timed_short_request(self, sample_idx: int) -> tuple[float, float]:
        """Issue one cold short request and record latency and completion time.

        :param sample_idx: Offset used to keep prompts distinct across samples.
        :returns: Client latency in milliseconds and monotonic completion time.
        """
        input_ids = _generate_input_chunks(
            self.tokenizer,
            num_turns=1,
            input_len=COLD_PREFILL_SHORT_INPUT_LEN,
            offset=sample_idx,
        )[0]
        started_at = time.perf_counter()
        _send_generate(
            self.base_url,
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
        )
        finished_at = time.perf_counter()
        return (finished_at - started_at) * 1000, finished_at

    def _run_deep_session_prefill(
        self,
        session_id: str,
        input_ids: list[int],
    ) -> tuple[dict[str, object], float]:
        """Prefill one deep streaming session to a durable boundary.

        :param session_id: Open streaming-session identity.
        :param input_ids: Cold token history exceeding the qualification depth.
        :returns: Generate response and monotonic completion time.
        """
        response = _send_generate(
            self.base_url,
            {
                "input_ids": input_ids,
                "session_params": {"id": session_id, "rid": None},
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
        )
        return response, time.perf_counter()

    def _wait_for_session_inflight(
        self,
        session_id: str,
        deep_future: Future[tuple[dict[str, object], float]],
    ) -> None:
        """Wait until the scheduler has admitted the deep session mutation.

        :param session_id: Session whose atomic state is polled.
        :param deep_future: Concurrent deep-request future used for early failure.
        """
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
            )
            response.raise_for_status()
            if response.json()["inflight"]:
                return
            if deep_future.done():
                self.fail("deep session request finished before inflight observation")
            time.sleep(0.01)
        self.fail("deep session request was not admitted within 10 seconds")


if __name__ == "__main__":
    unittest.main()
