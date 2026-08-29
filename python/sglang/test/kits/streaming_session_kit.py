"""Streaming-session test method mixins.

Pair these with `StreamingSessionServerBase` (from sglang.test.server_fixtures.streaming_session_fixture)
to assemble a concrete test class. Per the sglang fixture/kit split:
the fixture only launches the server; the kit owns the `test_*` methods.

- `StreamingSessionKitMixin`: KV-inheritance + chunked-prefill + abort-recovery
  + concurrent-logprob/stress test methods.
- `AbortLeakReproKitMixin`: single test method for abort-heavy chunked-prefill leak repro.
"""

import asyncio
import json
import time
import uuid

import requests
from sglang.test.server_fixtures.streaming_session_fixture import (
    _abort_repro_run_all,
    _concurrent_logprob_run,
    _stress_run_all,
)

_SESSION_INFO_FIELDS = frozenset(
    {
        "exists",
        "tip",
        "floor",
        "protected",
        "inflight",
        "held_tokens",
        "last_rid",
    }
)
_CONFLICT_ERROR_FIELDS = frozenset(
    {"message", "type", "code", "retryable", "correlation_id"}
)
_SESSION_TIMEOUT_SECONDS = 1.25
_SESSION_REAP_INTERVAL_SECONDS = 1.0
_SESSION_INFO_POLL_SECONDS = 0.1


def _assert_session_conflict(body: object, correlation_id: str) -> None:
    """Assert the exact streaming-session conflict envelope.

    :param body: Decoded HTTP or SSE error body.
    :param correlation_id: Top-level request identifier sent by the client.
    """
    assert isinstance(body, dict), body
    assert set(body) == {"error"}, body
    error = body["error"]
    assert isinstance(error, dict), error
    assert set(error) == _CONFLICT_ERROR_FIELDS, error
    assert isinstance(error["message"], str) and len(error["message"]) > 0
    assert error["type"] == "streaming_session_conflict"
    assert error["code"] == 409
    assert error["retryable"] is False
    assert error["correlation_id"] == correlation_id


def _read_sse_data(response: requests.Response) -> list[str]:
    """Read all SSE data fields and reject any other event fields.

    :param response: Streaming HTTP response.
    :returns: Ordered SSE data payloads.
    """
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    assert content_type == "text/event-stream", response.headers

    payloads: list[str] = []
    for raw_line in response.iter_lines():
        if len(raw_line) == 0:
            continue
        line = raw_line.decode("utf-8")
        assert line.startswith("data:"), line
        payloads.append(line[len("data:") :].strip())
    return payloads


class StreamingSessionKitMixin:
    """Streaming-session KV-inheritance + retract/abort-recovery suite."""

    # Allowed inherited-cache offsets vs the previous turn's total. Non-overlap
    # spec decode can be off by 1: the bonus token's KV is only computed by the
    # next forward, which sync skips at finish (overlap drains it, so it's 0).
    kv_inherit_offsets = (0,)

    def test_kv_cache_inheritance(self, gen_len=12):
        """Each turn's cached_tokens must equal previous turn's prompt+completion
        (modulo kv_inherit_offsets)."""
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
        ]
        chunks_ids = [self.tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == self.tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # === Part 1: streaming session — check KV inheritance ===
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
        ).json()
        rid = None

        prev_kv_len = 0
        for turn_idx, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {"id": session_id, "rid": rid},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": gen_len,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            cached = response["meta_info"]["cached_tokens"]
            prompt_tokens = response["meta_info"]["prompt_tokens"]
            completion_tokens = response["meta_info"]["completion_tokens"]

            if turn_idx == 0:
                # Turn 1: cache flushed, no hit.
                self.assertEqual(cached, 0, "Turn 1: clean start, no cache hit")
            else:
                # Turns 2+: cached_tokens reflects KV inherited from previous turn
                # (via inherit_kv_states, not radix tree matching).
                allowed = {prev_kv_len + off for off in self.kv_inherit_offsets}
                self.assertIn(
                    cached,
                    allowed,
                    f"Turn {turn_idx + 1}: inherited {cached} not in {sorted(allowed)}",
                )
            prev_kv_len = prompt_tokens + completion_tokens

        # Close the session.
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

    def test_leak_logprob_concurrent(self) -> None:
        """Concurrent multi-session × 3 logprob modes (output / input / none),
        watch for KV leak."""
        requests.post(self.base_url + "/flush_cache")
        # Output logprob
        asyncio.run(
            _concurrent_logprob_run(self.base_url, self.tokenizer, return_logprob=True)
        )
        # Input logprob (logprob_start_len=0)
        asyncio.run(
            _concurrent_logprob_run(
                self.base_url,
                self.tokenizer,
                return_logprob=True,
                logprob_start_len=0,
            )
        )
        # No logprob
        asyncio.run(_concurrent_logprob_run(self.base_url, self.tokenizer))
        time.sleep(3)
        assert (
            requests.get(self.base_url + "/health").status_code == 200
        ), "Server unhealthy after concurrent logprob sessions."

    def test_stress_concurrent_sessions(self) -> None:
        """High concurrency streaming + non-streaming with retract pressure;
        scheduler must roll back streaming KV without leaking."""
        requests.post(self.base_url + "/flush_cache")
        asyncio.run(_stress_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-stress cleanup {i}.")
            requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
            )

        time.sleep(5)
        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after concurrent stress test — "
            "likely a token leak from retract/mixed-chunk + streaming session.",
        )

    def test_nth_mid_abort_recovery(self) -> None:
        """Abort an Nth-turn request mid-decode; session rolls back to last
        successful turn."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_1.status_code, 200, resp_1.text)
            data_1 = resp_1.json()
            turn_1_total = (
                data_1["meta_info"]["prompt_tokens"]
                + data_1["meta_info"]["completion_tokens"]
            )

            # Turn 2: long generate, then abort mid-decode.
            ids_2 = self.tokenizer.encode(" Continue the story in great detail.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 2 should have returned")
            data_2 = result[0].json()
            self.assertEqual(
                data_2["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 2 should be aborted, not finished normally",
            )

            # Turn 3: recovery. Rolls back to turn 1.
            ids_3 = self.tokenizer.encode(" What happens next?")
            for attempt in range(20):
                resp_3 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_3,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_3.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            self.assertEqual(
                data_3["meta_info"]["cached_tokens"],
                turn_1_total,
                "Abort must preserve the pre-burst KV slot",
            )
            # prompt_tokens = turn_1_total + append (BOS stripped).
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (no stale abort context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_first_mid_abort_recovery(self) -> None:
        """Abort the very first request mid-decode (no slot yet; ephemeral
        slot is created and nuked). Session must still be usable."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_1,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 100000,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()
            time.sleep(0.5)
            abort_resp = requests.post(
                self.base_url + "/abort_request",
                json={"rid": "", "abort_all": True},
                timeout=10,
            )
            self.assertEqual(abort_resp.status_code, 200, abort_resp.text)
            t.join(timeout=30)

            self.assertIsNotNone(result[0], "Turn 1 should have returned")
            data_1 = result[0].json()
            self.assertEqual(
                data_1["meta_info"]["finish_reason"]["type"],
                "abort",
                "Turn 1 should be aborted, not finished normally",
            )

            # Turn 2: recovery. No inherited context (req_nodes empty).
            ids_2 = self.tokenizer.encode("Tell me a short joke.")
            for attempt in range(20):
                resp_2 = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=30,
                )
                if resp_2.status_code == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(resp_2.status_code, 200, resp_2.text)
            data_2 = resp_2.json()
            self.assertEqual(
                data_2["meta_info"]["prompt_tokens"],
                len(ids_2),
                "prompt_tokens must equal turn 2 input only (no inherited context)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_preabort_recovery(self) -> None:
        """Pre-abort (rejected by create_req) preserves the slot; next turn
        inherits correctly."""
        requests.post(self.base_url + "/flush_cache")

        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot.
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_1.status_code, 200, resp_1.text)
            data_1 = resp_1.json()
            turn_1_total = (
                data_1["meta_info"]["prompt_tokens"]
                + data_1["meta_info"]["completion_tokens"]
            )

            # Turn 2: pre-aborted via unsupported offset parameter.
            ids_2 = self.tokenizer.encode(" This should be rejected.")
            resp_2 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_2,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "session_params": {
                        "id": session_id,
                        "rid": None,
                        "offset": 1,
                    },
                },
                timeout=30,
            )
            self.assertIn(resp_2.status_code, (200, 400), resp_2.text)

            # Turn 3: normal append. Slot should be intact from turn 1.
            ids_3 = self.tokenizer.encode(" What happens next?")
            resp_3 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_3,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            bos = 1 if ids_3[0] == self.tokenizer.bos_token_id else 0
            expected_prompt_3 = turn_1_total + len(ids_3) - bos
            self.assertEqual(
                data_3["meta_info"]["prompt_tokens"],
                expected_prompt_3,
                "prompt_tokens must equal turn_1_total + append (slot preserved)",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)

    def test_idempotency_and_session_info(self) -> None:
        """Idempotent retries conflict without touching recovery state."""
        requests.post(self.base_url + "/flush_cache", timeout=30)
        open_response = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
            timeout=30,
        )
        self.assertEqual(open_response.status_code, 200, open_response.text)
        session_id = open_response.json()
        hot_extra_key = "session-idempotency-hot-" + uuid.uuid4().hex

        try:
            initial_response = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            )
            self.assertEqual(initial_response.status_code, 200, initial_response.text)
            initial = initial_response.json()
            self.assertEqual(set(initial), _SESSION_INFO_FIELDS)
            self.assertEqual(
                initial,
                {
                    "exists": True,
                    "tip": 0,
                    "floor": 0,
                    "protected": 0,
                    "inflight": False,
                    "held_tokens": 0,
                    "last_rid": None,
                },
            )

            request_rid = "session-idempotency-" + uuid.uuid4().hex
            seed_input_ids = self.tokenizer.encode(
                "The first idempotent streaming-session request."
            )
            request_payload = {
                "rid": request_rid,
                "input_ids": seed_input_ids,
                "extra_key": hot_extra_key,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "expected_tip": 0,
                },
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "stream": False,
            }
            accepted_response = requests.post(
                self.base_url + "/generate",
                json=request_payload,
                timeout=30,
            )
            self.assertEqual(accepted_response.status_code, 200, accepted_response.text)
            accepted = accepted_response.json()
            self.assertEqual(accepted["meta_info"]["id"], request_rid)
            accepted_output_ids = accepted["output_ids"]
            self.assertIsInstance(accepted_output_ids, list)

            info_response = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            )
            self.assertEqual(info_response.status_code, 200, info_response.text)
            stable = info_response.json()
            self.assertEqual(set(stable), _SESSION_INFO_FIELDS)
            tip = (
                accepted["meta_info"]["prompt_tokens"]
                + accepted["meta_info"]["completion_tokens"]
            )
            self.assertEqual(stable["tip"], tip)
            self.assertEqual(stable["floor"], tip)
            self.assertEqual(stable["last_rid"], request_rid)
            self.assertFalse(stable["inflight"])

            retry_response = requests.post(
                self.base_url + "/generate",
                json=request_payload,
                timeout=30,
            )
            self.assertEqual(retry_response.status_code, 409, retry_response.text)
            _assert_session_conflict(retry_response.json(), request_rid)

            low_after = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            ).json()
            self.assertEqual(low_after, stable)

            stream_rid = "session-idempotency-stream-" + uuid.uuid4().hex
            stream_payload = {
                "rid": stream_rid,
                "input_ids": self.tokenizer.encode("A stale high-tip append."),
                "extra_key": hot_extra_key,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "expected_tip": tip + 1,
                },
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "stream": True,
            }
            with requests.post(
                self.base_url + "/generate",
                json=stream_payload,
                stream=True,
                timeout=30,
            ) as stream_response:
                self.assertEqual(stream_response.status_code, 200, stream_response.text)
                sse_data = _read_sse_data(stream_response)
            self.assertEqual(len(sse_data), 2, sse_data)
            self.assertEqual(sse_data[1], "[DONE]")
            _assert_session_conflict(json.loads(sse_data[0]), stream_rid)

            high_after = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            ).json()
            self.assertEqual(high_after, stable)

            recovery_rid = "session-idempotency-recovery-" + uuid.uuid4().hex
            recovery_input_ids = self.tokenizer.encode("A valid exact-tip append.")
            recovery_delta = recovery_input_ids
            if (
                len(recovery_delta) > 0
                and recovery_delta[0] == self.tokenizer.bos_token_id
            ):
                recovery_delta = recovery_delta[1:]
            recovery_response = requests.post(
                self.base_url + "/generate",
                json={
                    "rid": recovery_rid,
                    "input_ids": recovery_input_ids,
                    "extra_key": hot_extra_key,
                    "session_params": {
                        "id": session_id,
                        "rid": None,
                        "expected_tip": tip,
                    },
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(recovery_response.status_code, 200, recovery_response.text)
            recovery = recovery_response.json()
            self.assertEqual(recovery["meta_info"]["id"], recovery_rid)

            cold_open_response = requests.post(
                self.base_url + "/open_session",
                json={"capacity_of_str_len": 1000, "streaming": True},
                timeout=30,
            )
            self.assertEqual(
                cold_open_response.status_code,
                200,
                cold_open_response.text,
            )
            cold_session_id = cold_open_response.json()
            try:
                cold_response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "rid": "session-idempotency-cold-" + uuid.uuid4().hex,
                        "input_ids": (
                            seed_input_ids + accepted_output_ids + recovery_delta
                        ),
                        "extra_key": ("session-idempotency-cold-" + uuid.uuid4().hex),
                        "session_params": {"id": cold_session_id, "rid": None},
                        "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                    },
                    timeout=30,
                )
                self.assertEqual(
                    cold_response.status_code,
                    200,
                    cold_response.text,
                )
                cold = cold_response.json()
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                self.assertEqual(
                    cold["meta_info"]["prompt_tokens"],
                    recovery["meta_info"]["prompt_tokens"],
                )
                self.assertEqual(cold["output_ids"], recovery["output_ids"])
            finally:
                cold_close_response = requests.post(
                    self.base_url + "/close_session",
                    json={"session_id": cold_session_id},
                    timeout=30,
                )
                self.assertEqual(
                    cold_close_response.status_code,
                    200,
                    cold_close_response.text,
                )
        finally:
            close_response = requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
                timeout=30,
            )
            self.assertEqual(close_response.status_code, 200, close_response.text)

        closed_response = requests.get(
            self.base_url + "/session_info",
            params={"session_id": session_id},
            timeout=30,
        )
        self.assertEqual(closed_response.status_code, 200, closed_response.text)
        closed = closed_response.json()
        self.assertEqual(set(closed), _SESSION_INFO_FIELDS)
        self.assertEqual(
            closed,
            {
                "exists": False,
                "tip": 0,
                "floor": 0,
                "protected": 0,
                "inflight": False,
                "held_tokens": 0,
                "last_rid": None,
            },
        )

    def test_session_info_does_not_refresh_timeout(self) -> None:
        """Repeated recovery reads do not keep an idle session alive."""
        open_response = requests.post(
            self.base_url + "/open_session",
            json={
                "capacity_of_str_len": 1000,
                "streaming": True,
                "timeout": _SESSION_TIMEOUT_SECONDS,
            },
            timeout=30,
        )
        self.assertEqual(open_response.status_code, 200, open_response.text)
        session_id = open_response.json()
        started = time.monotonic()
        live_reads = 0
        info: dict[str, object] = {"exists": True}
        try:
            deadline = (
                started
                + _SESSION_TIMEOUT_SECONDS
                + _SESSION_REAP_INTERVAL_SECONDS
                + 1.0
            )
            while time.monotonic() < deadline:
                info_response = requests.get(
                    self.base_url + "/session_info",
                    params={"session_id": session_id},
                    timeout=30,
                )
                self.assertEqual(info_response.status_code, 200, info_response.text)
                info = info_response.json()
                self.assertEqual(set(info), _SESSION_INFO_FIELDS)
                if info["exists"] is False:
                    break
                live_reads += 1
                time.sleep(_SESSION_INFO_POLL_SECONDS)

            self.assertGreaterEqual(live_reads, 2)
            elapsed = time.monotonic() - started
            self.assertFalse(
                info["exists"],
                "session_info reads refreshed the inactivity timeout: "
                f"elapsed={elapsed:.3f}s",
            )
            self.assertGreaterEqual(
                elapsed,
                _SESSION_TIMEOUT_SECONDS - _SESSION_INFO_POLL_SECONDS,
                f"session reaped before its inactivity lease: elapsed={elapsed:.3f}s",
            )
            self.assertLessEqual(
                elapsed,
                _SESSION_TIMEOUT_SECONDS + _SESSION_REAP_INTERVAL_SECONDS + 1.5,
                f"session reaping exceeded its bounded lease: elapsed={elapsed:.3f}s",
            )
            self.assertEqual(
                info,
                {
                    "exists": False,
                    "tip": 0,
                    "floor": 0,
                    "protected": 0,
                    "inflight": False,
                    "held_tokens": 0,
                    "last_rid": None,
                },
            )
        finally:
            if info["exists"] is True:
                requests.post(
                    self.base_url + "/close_session",
                    json={"session_id": session_id},
                    timeout=30,
                )


class AbortLeakReproKitMixin:
    """Abort-heavy chunked-prefill leak repro."""

    def test_abort_heavy_chunked_prefill_does_not_leak(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_abort_repro_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-session cleanup request {i}.")
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during abort-heavy streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after abort-heavy streaming session cleanup.",
        )
