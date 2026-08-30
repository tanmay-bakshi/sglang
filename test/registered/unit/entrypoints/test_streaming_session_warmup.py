import unittest
from collections.abc import AsyncIterator

from sglang.srt.entrypoints.warmup import (
    GEMMA4_STREAMING_SESSION_WARMUPS,
    execute_warmups,
)
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    GenerateReqInput,
    OpenSessionReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _WarmupFailure(RuntimeError):
    """Sentinel failure raised by the fake tokenizer manager."""


class _TokenizerManager:
    """Record warmup lifecycle while emulating multi-yield generation."""

    events: list[str]
    requests: list[GenerateReqInput]
    open_requests: list[OpenSessionReqInput]
    close_requests: list[CloseSessionReqInput]
    fail_request: int | None
    failure: _WarmupFailure

    def __init__(self, fail_request: int | None = None) -> None:
        """Initialize the deterministic warmup manager.

        :param fail_request: One-based generation request that should fail.
        """
        self.events = []
        self.requests = []
        self.open_requests = []
        self.close_requests = []
        self.fail_request = fail_request
        self.failure = _WarmupFailure("warmup generation failed")

    async def open_session(
        self,
        request: OpenSessionReqInput,
        raw_request: object | None,
    ) -> str:
        """Record session creation.

        :param request: Session-open request.
        :param raw_request: Unused HTTP request.
        :returns: Stable session identifier.
        """
        self.events.append("open")
        self.open_requests.append(request)
        self.assert_no_raw_request(raw_request)
        return "warmup-session"

    async def generate_request(
        self,
        request: GenerateReqInput,
        raw_request: object | None,
    ) -> AsyncIterator[dict[str, int]]:
        """Yield three chunks or fail after the first chunk.

        :param request: Generation request under qualification.
        :param raw_request: Unused HTTP request.
        :yields: Three synthetic response chunks on success.
        :raises _WarmupFailure: For the configured request after one yield.
        """
        self.assert_no_raw_request(raw_request)
        request_number = len(self.requests) + 1
        self.requests.append(request)
        self.events.append(f"generate-{request_number}-start")
        for chunk_number in range(3):
            self.events.append(f"generate-{request_number}-yield-{chunk_number}")
            yield {"chunk": chunk_number}
            if self.fail_request == request_number and chunk_number == 0:
                self.events.append(f"generate-{request_number}-failure")
                raise self.failure
        self.events.append(f"generate-{request_number}-terminal")

    async def close_session(
        self,
        request: CloseSessionReqInput,
        raw_request: object | None,
    ) -> None:
        """Record session cleanup.

        :param request: Session-close request.
        :param raw_request: Unused HTTP request.
        """
        self.events.append("close")
        self.close_requests.append(request)
        self.assert_no_raw_request(raw_request)

    @staticmethod
    def assert_no_raw_request(raw_request: object | None) -> None:
        """Assert warmup calls do not carry an HTTP request.

        :param raw_request: Request object supplied by the warmup.
        """
        if raw_request is not None:
            raise AssertionError("warmup unexpectedly supplied an HTTP request")


class StreamingSessionSmallExtendWarmupTest(unittest.IsolatedAsyncioTestCase):
    async def test_drains_both_requests_and_closes_session(self) -> None:
        manager = _TokenizerManager()

        await execute_warmups(
            "null",
            ["streaming_session_small_extend"],
            manager,
        )

        self.assertEqual(
            manager.events,
            [
                "open",
                "generate-1-start",
                "generate-1-yield-0",
                "generate-1-yield-1",
                "generate-1-yield-2",
                "generate-1-terminal",
                "generate-2-start",
                "generate-2-yield-0",
                "generate-2-yield-1",
                "generate-2-yield-2",
                "generate-2-terminal",
                "close",
            ],
        )
        self.assertEqual(len(manager.open_requests), 1)
        self.assertEqual(len(manager.close_requests), 1)
        self.assertEqual(manager.close_requests[0].session_id, "warmup-session")
        self.assertEqual(len(manager.requests), 2)

        open_request = manager.open_requests[0]
        self.assertEqual(open_request.capacity_of_str_len, 0)
        self.assertTrue(open_request.streaming)

        prefix_request, extend_request = manager.requests
        self.assertEqual(len(prefix_request.input_ids), 40_960)
        self.assertEqual(prefix_request.sampling_params["max_new_tokens"], 0)
        self.assertFalse(prefix_request.stream)
        self.assertFalse(prefix_request.log_metrics)
        self.assertEqual(
            prefix_request.session_params,
            {"id": "warmup-session", "rid": None},
        )

        self.assertEqual(len(extend_request.input_ids), 64)
        self.assertEqual(extend_request.sampling_params["temperature"], 0.0)
        self.assertEqual(extend_request.sampling_params["max_new_tokens"], 8)
        self.assertTrue(extend_request.sampling_params["ignore_eos"])
        self.assertTrue(extend_request.stream)
        self.assertFalse(extend_request.log_metrics)
        self.assertEqual(prefix_request.extra_key, extend_request.extra_key)

    async def test_closes_once_and_preserves_first_request_failure(self) -> None:
        await self._assert_generation_failure_closes_once(fail_request=1)

    async def test_closes_once_and_preserves_second_request_failure(self) -> None:
        await self._assert_generation_failure_closes_once(fail_request=2)

    async def test_rejects_disaggregation_before_opening_session(self) -> None:
        manager = _TokenizerManager()

        with self.assertRaisesRegex(
            ValueError,
            "requires disaggregation_mode='null'",
        ):
            await execute_warmups(
                "prefill",
                ["streaming_session_small_extend"],
                manager,
            )

        self.assertEqual(manager.events, [])
        self.assertEqual(manager.open_requests, [])
        self.assertEqual(manager.requests, [])
        self.assertEqual(manager.close_requests, [])

    async def _assert_generation_failure_closes_once(
        self,
        fail_request: int,
    ) -> None:
        """Assert one generation failure propagates through cleanup.

        :param fail_request: One-based generation request that should fail.
        """
        manager = _TokenizerManager(fail_request=fail_request)

        with self.assertRaises(_WarmupFailure) as raised:
            await execute_warmups(
                "null",
                ["streaming_session_small_extend"],
                manager,
            )

        self.assertIs(raised.exception, manager.failure)
        self.assertEqual(len(manager.close_requests), 1)
        self.assertEqual(manager.events[-1], "close")


class StreamingSessionShallowEagerExtendWarmupTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_reproduces_cached_prefix_eager_continuation(self) -> None:
        manager = _TokenizerManager()

        await execute_warmups(
            "null",
            ["streaming_session_shallow_eager_extend"],
            manager,
        )

        self.assertEqual(len(manager.open_requests), 1)
        self.assertEqual(len(manager.requests), 2)
        self.assertEqual(len(manager.close_requests), 1)
        self.assertEqual(manager.events[-1], "close")

        seed_request, continuation_request = manager.requests
        self.assertEqual(len(seed_request.input_ids), 96)
        self.assertEqual(seed_request.session_params["expected_tip"], 0)
        self.assertEqual(seed_request.sampling_params["temperature"], 0.0)
        self.assertEqual(seed_request.sampling_params["max_new_tokens"], 4)
        self.assertTrue(seed_request.sampling_params["ignore_eos"])
        self.assertFalse(seed_request.stream)
        self.assertFalse(seed_request.log_metrics)

        self.assertEqual(len(continuation_request.input_ids), 16)
        self.assertEqual(continuation_request.session_params["expected_tip"], 100)
        self.assertEqual(continuation_request.sampling_params["temperature"], 0.0)
        self.assertEqual(continuation_request.sampling_params["max_new_tokens"], 4)
        self.assertTrue(continuation_request.sampling_params["ignore_eos"])
        self.assertFalse(continuation_request.stream)
        self.assertFalse(continuation_request.log_metrics)
        self.assertEqual(seed_request.extra_key, continuation_request.extra_key)

    async def test_closes_once_and_preserves_generation_failures(self) -> None:
        for fail_request in (1, 2):
            with self.subTest(fail_request=fail_request):
                manager = _TokenizerManager(fail_request=fail_request)

                with self.assertRaises(_WarmupFailure) as raised:
                    await execute_warmups(
                        "null",
                        ["streaming_session_shallow_eager_extend"],
                        manager,
                    )

                self.assertIs(raised.exception, manager.failure)
                self.assertEqual(len(manager.close_requests), 1)
                self.assertEqual(manager.events[-1], "close")

    async def test_rejects_disaggregation_before_opening_session(self) -> None:
        manager = _TokenizerManager()

        with self.assertRaisesRegex(
            ValueError,
            "requires disaggregation_mode='null'",
        ):
            await execute_warmups(
                "prefill",
                ["streaming_session_shallow_eager_extend"],
                manager,
            )

        self.assertEqual(manager.events, [])


class StochasticSamplingFirstUseWarmupTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_production_sampling_parameters(self) -> None:
        manager = _TokenizerManager()

        await execute_warmups(
            "null",
            ["stochastic_sampling_first_use"],
            manager,
        )

        self.assertEqual(len(manager.requests), 1)
        request = manager.requests[0]
        self.assertEqual(len(request.input_ids), 8)
        self.assertEqual(request.sampling_params["temperature"], 0.4)
        self.assertEqual(request.sampling_params["top_k"], 64)
        self.assertEqual(request.sampling_params["top_p"], 0.95)
        self.assertEqual(request.sampling_params["min_p"], 0.0)
        self.assertEqual(request.sampling_params["max_new_tokens"], 1)
        self.assertTrue(request.sampling_params["ignore_eos"])
        self.assertFalse(request.stream)
        self.assertFalse(request.log_metrics)
        self.assertEqual(manager.open_requests, [])
        self.assertEqual(manager.close_requests, [])

    async def test_rejects_disaggregation_before_generation(self) -> None:
        manager = _TokenizerManager()

        with self.assertRaisesRegex(
            ValueError,
            "requires disaggregation_mode='null'",
        ):
            await execute_warmups(
                "decode",
                ["stochastic_sampling_first_use"],
                manager,
            )

        self.assertEqual(manager.events, [])


class Gemma4WarmupListTest(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_list_resolves_and_executes_in_order(self) -> None:
        self.assertEqual(
            GEMMA4_STREAMING_SESSION_WARMUPS,
            (
                "streaming_session_small_extend",
                "streaming_session_shallow_eager_extend",
                "stochastic_sampling_first_use",
            ),
        )
        manager = _TokenizerManager()

        await execute_warmups(
            "null",
            list(GEMMA4_STREAMING_SESSION_WARMUPS),
            manager,
        )

        self.assertEqual(len(manager.open_requests), 2)
        self.assertEqual(len(manager.requests), 5)
        self.assertEqual(len(manager.close_requests), 2)
        self.assertNotEqual(
            manager.requests[0].extra_key,
            manager.requests[2].extra_key,
        )
        self.assertEqual(
            manager.events,
            [
                "open",
                "generate-1-start",
                "generate-1-yield-0",
                "generate-1-yield-1",
                "generate-1-yield-2",
                "generate-1-terminal",
                "generate-2-start",
                "generate-2-yield-0",
                "generate-2-yield-1",
                "generate-2-yield-2",
                "generate-2-terminal",
                "close",
                "open",
                "generate-3-start",
                "generate-3-yield-0",
                "generate-3-yield-1",
                "generate-3-yield-2",
                "generate-3-terminal",
                "generate-4-start",
                "generate-4-yield-0",
                "generate-4-yield-1",
                "generate-4-yield-2",
                "generate-4-terminal",
                "close",
                "generate-5-start",
                "generate-5-yield-0",
                "generate-5-yield-1",
                "generate-5-yield-2",
                "generate-5-terminal",
            ],
        )


if __name__ == "__main__":
    unittest.main()
