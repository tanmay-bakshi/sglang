"""HTTP lifecycle coverage for host-resident streaming sessions."""

import unittest
import uuid
from typing import Any, ClassVar

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


class TestDemotedStreamingSessionNoop(StreamingSessionServerBase):
    """Qualify an empty no-op without restoring a host-resident session."""

    extra_args: ClassVar[list[str]] = [
        "--attention-backend",
        "triton",
        "--cuda-graph-backend-decode=disabled",
        "--cuda-graph-backend-prefill=disabled",
        "--chunked-prefill-size",
        "512",
        "--mem-fraction-static",
        "0.7",
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1",
        "--hicache-write-policy",
        "write_through",
        "--enable-session-radix-cache",
    ]

    def test_empty_noop_preserves_demotion_then_resumes(self) -> None:
        requests.post(self.base_url + "/flush_cache", timeout=30).raise_for_status()
        open_response = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 0, "streaming": True},
            timeout=30,
        )
        self.assertEqual(open_response.status_code, 200, open_response.text)
        session_id: str = open_response.json()
        extra_key = "demoted-empty-noop-" + uuid.uuid4().hex

        try:
            seed_input_ids = self.tokenizer.encode(
                "A host-resident session must survive an empty append."
            )
            seed_response = requests.post(
                self.base_url + "/generate",
                json={
                    "rid": "seed-" + uuid.uuid4().hex,
                    "input_ids": seed_input_ids,
                    "extra_key": extra_key,
                    "session_params": {"id": session_id, "rid": None},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                },
                timeout=30,
            )
            self.assertEqual(seed_response.status_code, 200, seed_response.text)
            seed: dict[str, Any] = seed_response.json()
            seed_tip = (
                seed["meta_info"]["prompt_tokens"]
                + seed["meta_info"]["completion_tokens"]
            )

            demote_response = requests.post(
                self.base_url + "/demote_session",
                json={"session_id": session_id},
                timeout=30,
            )
            self.assertEqual(demote_response.status_code, 200, demote_response.text)
            self.assertEqual(
                demote_response.json()["host_backed_tokens"],
                seed_tip,
            )
            before_noop: dict[str, Any] = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            ).json()
            self.assertEqual(before_noop["tip"], seed_tip)
            self.assertEqual(before_noop["protected"], seed_tip)
            self.assertEqual(before_noop["held_tokens"], 0)
            self.assertFalse(before_noop["inflight"])

            noop_rid = "noop-" + uuid.uuid4().hex
            noop_response = requests.post(
                self.base_url + "/generate",
                json={
                    "rid": noop_rid,
                    "input_ids": [],
                    "extra_key": extra_key,
                    "session_params": {"id": session_id, "rid": None},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 0,
                    },
                },
                timeout=30,
            )
            self.assertEqual(noop_response.status_code, 200, noop_response.text)
            noop: dict[str, Any] = noop_response.json()
            self.assertEqual(noop["output_ids"], [])
            self.assertEqual(noop["meta_info"]["prompt_tokens"], seed_tip)
            self.assertEqual(noop["meta_info"]["completion_tokens"], 0)

            after_noop: dict[str, Any] = requests.get(
                self.base_url + "/session_info",
                params={"session_id": session_id},
                timeout=30,
            ).json()
            for field in ("tip", "floor", "protected", "held_tokens"):
                self.assertEqual(after_noop[field], before_noop[field])
            self.assertFalse(after_noop["inflight"])
            self.assertEqual(after_noop["last_rid"], noop_rid)

            continuation_ids = self.tokenizer.encode(" Continue normally.")
            if continuation_ids[0] == self.tokenizer.bos_token_id:
                continuation_ids = continuation_ids[1:]
            continuation_response = requests.post(
                self.base_url + "/generate",
                json={
                    "rid": "continuation-" + uuid.uuid4().hex,
                    "input_ids": continuation_ids,
                    "extra_key": extra_key,
                    "session_params": {"id": session_id, "rid": None},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                },
                timeout=30,
            )
            self.assertEqual(
                continuation_response.status_code,
                200,
                continuation_response.text,
            )
            continuation: dict[str, Any] = continuation_response.json()
            self.assertEqual(
                continuation["meta_info"]["cached_tokens"],
                seed_tip,
            )
            self.assertEqual(
                continuation["meta_info"]["prompt_tokens"],
                seed_tip + len(continuation_ids),
            )
            health = requests.get(self.base_url + "/health", timeout=30)
            self.assertEqual(health.status_code, 200, health.text)
        finally:
            close_response = requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
                timeout=30,
            )
            self.assertEqual(close_response.status_code, 200, close_response.text)


if __name__ == "__main__":
    unittest.main()
