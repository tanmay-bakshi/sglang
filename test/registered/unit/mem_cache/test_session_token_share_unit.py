"""Unit tests for the streaming-session in-place token-array share protocol
(`Session.create_req` / `finish_req` / `abort_req`):

- token arrays are extended in place and shared across turns (no per-turn copy);
- committed_* lengths recorded at finish_req trim away tokens appended by a
  turn that aborted before finishing (mid-turn and first-turn aborts);
- max_new_tokens overshoot falls back to a fill_ids rebuild instead of
  carrying an inconsistent array.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace

from sglang.srt.managers.io_struct import OpenSessionReqInput
from sglang.srt.mem_cache.base_prefix_cache import StreamingSessionCacheSnapshot
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.runtime_context import get_parallel
from sglang.srt.session.errors import (
    STREAMING_SESSION_CONFLICT_ERROR_TYPE,
    StreamingSessionConflictError,
)
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(
    rid,
    input_ids,
    max_new_tokens=8,
    truncate_to=None,
    commit_to=None,
    expected_tip=None,
):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s",
            rid=None,
            offset=None,
            replace=False,
            drop_previous_output=False,
            truncate_to=truncate_to,
            commit_to=commit_to,
            expected_tip=expected_tip,
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestSessionTokenShare(CustomTestCase):

    def setUp(self):
        self.session = Session(
            capacity_of_str_len=0,
            session_id="s",
            streaming=True,
            manual_commit=True,
        )

    def test_conflict_error_preserves_request_correlation(self):
        error = StreamingSessionConflictError("stale session", "request-rid")

        self.assertEqual(str(error), "stale session")
        self.assertEqual(error.correlation_id, "request-rid")

    def test_controller_captures_recurrent_cache_capability(self):
        tree_cache = SimpleNamespace(supports_mamba=lambda: True)
        controller = SessionController(tree_cache)

        result = controller.open(
            OpenSessionReqInput(
                capacity_of_str_len=0,
                session_id="mamba-session",
                streaming=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(controller.get("mamba-session").supports_mamba)

    def _create(
        self,
        rid,
        input_ids,
        max_new_tokens=8,
        truncate_to=None,
        commit_to=None,
        expected_tip=None,
    ):
        return self.session.create_req(
            _recv(
                rid,
                input_ids,
                max_new_tokens=max_new_tokens,
                truncate_to=truncate_to,
                commit_to=commit_to,
                expected_tip=expected_tip,
            ),
            tokenizer=None,
            vocab_size=VOCAB,
        )

    def _decode_and_finish(self, req, output, baked=None):
        """Simulate decode then a successful finish.

        `baked` output tokens are folded into the fill array before the rest
        arrive (mix_with_running refreshes mid-decode, so the bake is often
        partial).
        """
        if baked is None:
            baked = len(output)
        req.output_ids.extend(output[:baked])
        req._refresh_fill_ids()
        req.output_ids.extend(output[baked:])
        self.session.finish_req(req)

    def test_normal_multi_turn_share_and_carry(self):
        in1, out1 = list(range(100, 110)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self.assertEqual(list(r1.origin_input_ids), in1)
        self._decode_and_finish(r1, out1, baked=2)  # partial bake
        self.assertEqual(self.session.committed_origin_len, len(in1))
        self.assertEqual(self.session.committed_fill_len, len(in1) + 2)

        in2, out2 = [7, 8], [4, 5]
        r2 = self._create("r2", in2)
        # In-place share: same objects, extended to the new prompt.
        self.assertIs(r2.origin_input_ids, r1.origin_input_ids)
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + in2)
        # Carry: the fill array handed over and equal to the new origin.
        self.assertIs(r2.full_untruncated_fill_ids, r1.full_untruncated_fill_ids)
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))
        self._decode_and_finish(r2, out2)

        r3 = self._create("r3", [9])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + in2 + out2 + [9])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

    def test_mid_turn_abort_then_continue(self):
        in1, out1 = list(range(200, 210)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self._decode_and_finish(r1, out1)

        # Turn 2 extends the shared arrays, decodes a bit, then aborts:
        # finish_req never runs, req_nodes still points at r1.
        r2 = self._create("r2", [50, 51])
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + [50, 51])
        r2.output_ids.extend([6, 7])
        r2._refresh_fill_ids()
        self.session.abort_req()
        self.assertEqual(self.session.committed_origin_len, len(in1))

        # Turn 3 must see exactly r1's history — no [50, 51], no doubled out1.
        r3 = self._create("r3", [60])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + [60])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

        # Two aborted attempts in a row heal idempotently.
        self.session.abort_req()
        r4 = self._create("r4", [70])
        self.assertEqual(list(r4.origin_input_ids), in1 + out1 + [70])
        self.assertEqual(list(r4.full_untruncated_fill_ids), list(r4.origin_input_ids))

    def test_first_turn_abort(self):
        self._create("r1", [1, 2, 3])
        self.assertTrue(self.session._inflight)
        self.session.abort_req()
        self.assertFalse(self.session._inflight)
        # No finish_req ran: nothing committed, next turn starts from scratch.
        self.assertIsNone(self.session.committed_origin_len)
        r2 = self._create("r2", [4, 5])
        self.assertEqual(list(r2.origin_input_ids), [4, 5])
        self._decode_and_finish(r2, [9])
        r3 = self._create("r3", [6])
        self.assertEqual(list(r3.origin_input_ids), [4, 5, 9, 6])

    def test_max_new_tokens_overshoot_falls_back(self):
        in1 = list(range(300, 310))
        r1 = self._create("r1", in1, max_new_tokens=4)
        # Spec-decode overshoot: 6 tokens decoded and baked into the fill
        # array, then output trimmed to finished_len (like _trim_overshoot)
        # before finish.
        r1.output_ids.extend([1, 2, 3, 4, 5, 6])
        r1._refresh_fill_ids()
        del r1.output_ids[4:]
        self.session.finish_req(r1)
        self.assertEqual(
            self.session.committed_fill_len, len(in1) + 6
        )  # fill kept the overshoot

        # Next turn: out_tail is output[:max_new]; the carried fill has more
        # baked than out_tail, so the carry is dropped and the fill rebuilds.
        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [1, 2, 3, 4] + [50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)  # carry skipped
        r2._refresh_fill_ids()
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))

    def test_truncate_is_prepared_then_committed(self):
        r1 = self._create("r1", [10, 11, 12, 13])
        self._decode_and_finish(r1, [20, 21, 22])
        original = list(r1.origin_input_ids)

        r2 = self._create("r2", [30, 31], truncate_to=5)
        self.assertEqual(list(r2.origin_input_ids), [10, 11, 12, 13, 20, 30, 31])
        self.assertEqual(list(r1.origin_input_ids), original)

        cache = SimpleNamespace(calls=[])
        cache.truncate_session = lambda session_id, target: cache.calls.append(
            (session_id, target)
        )
        self.session.commit_prepared_req(r2, cache)
        self.assertEqual(cache.calls, [("s", 5)])
        self.assertEqual(list(r1.origin_input_ids), [10, 11, 12, 13, 20])
        self.assertEqual(list(r1.output_ids), [])

        self.session.abort_req()
        r3 = self._create("r3", [40])
        self.assertEqual(list(r3.origin_input_ids), [10, 11, 12, 13, 20, 40])

    def test_invalid_truncate_is_non_destructive(self):
        r1 = self._create("r1", [1, 2, 3])
        self._decode_and_finish(r1, [4, 5])
        origin_before = list(r1.origin_input_ids)
        output_before = list(r1.output_ids)

        with get_parallel().override(tp_rank=0):
            invalid = self._create("bad", [9], truncate_to=99)
        self.assertIsNotNone(invalid.to_finish)
        self.assertFalse(self.session._inflight)
        self.assertEqual(list(r1.origin_input_ids), origin_before)
        self.assertEqual(list(r1.output_ids), output_before)

    def test_expected_tip_conflict_is_typed_and_non_destructive(self):
        first = self._create("r1", [1, 2, 3])
        self._decode_and_finish(first, [4, 5])
        self.assertEqual(self.session.current_tip(), 5)
        self.assertEqual(self.session.last_rid, "r1")

        origin_before = list(first.origin_input_ids)
        output_before = list(first.output_ids)
        last_active_before = self.session.last_active_time
        with get_parallel().override(tp_rank=0):
            conflict = self._create(
                "stale-rid",
                [9],
                truncate_to=0,
                commit_to=0,
                expected_tip=4,
            )

        finish = conflict.to_finish
        self.assertEqual(finish.status_code, HTTPStatus.CONFLICT)
        self.assertEqual(finish.err_type, STREAMING_SESSION_CONFLICT_ERROR_TYPE)
        self.assertEqual(conflict.rid, "stale-rid")
        self.assertEqual(
            finish.message,
            "Streaming session expected_tip conflict for session s: expected 4, "
            "current tip is 5.",
        )
        self.assertEqual(list(first.origin_input_ids), origin_before)
        self.assertEqual(list(first.output_ids), output_before)
        self.assertEqual(self.session.current_tip(), 5)
        self.assertEqual(self.session.floor, 0)
        self.assertEqual(self.session.last_rid, "r1")
        self.assertEqual(self.session.last_active_time, last_active_before)
        self.assertFalse(self.session._inflight)

    def test_matching_expected_tip_tracks_only_durable_context(self):
        first = self._create("r1", [1, 2, 3])
        self._decode_and_finish(first, [4])

        second = self._create("r2", [5, 6], expected_tip=4)
        self.assertIsNone(second.finished_reason)
        self.assertEqual(self.session.current_tip(), 4)
        self.assertEqual(self.session.last_rid, "r1")
        self.session.abort_req()
        self.assertEqual(self.session.current_tip(), 4)
        self.assertEqual(self.session.last_rid, "r1")

    def test_inflight_rejection_precedes_expected_tip_conflict(self):
        first = self._create("r1", [1, 2, 3])
        self._decode_and_finish(first, [4])
        active = self._create("active", [5], expected_tip=4)
        self.assertIsNone(active.to_finish)

        with get_parallel().override(tp_rank=0):
            rejected = self._create("concurrent", [6], expected_tip=0)

        self.assertEqual(rejected.to_finish.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(rejected.to_finish.err_type, "BadRequestError")
        self.assertIn("already has an active request", rejected.to_finish.message)
        self.session.abort_req()

    def test_durable_tip_and_last_rid_follow_prepared_mutations(self):
        first = self._create("r1", [1, 2, 3, 4])
        self._decode_and_finish(first, [5, 6])
        self.assertEqual(
            (self.session.current_tip(), self.session.last_rid),
            (6, "r1"),
        )

        truncate = self._create("truncate", [], max_new_tokens=0, truncate_to=4)
        cache = SimpleNamespace(
            truncate_session=lambda session_id, target: None,
            commit_session=lambda session_id, floor: None,
        )
        self.session.commit_prepared_req(truncate, cache)
        self.assertEqual(
            (self.session.current_tip(), self.session.last_rid),
            (4, "truncate"),
        )
        self.session.abort_req()
        self.assertEqual(
            (self.session.current_tip(), self.session.last_rid),
            (4, "truncate"),
        )

        commit = self._create("commit", [7, 8], max_new_tokens=0, commit_to=6)
        self.session.commit_prepared_req(commit, cache)
        self.assertEqual(
            (self.session.current_tip(), self.session.last_rid),
            (6, "commit"),
        )
        self.session.abort_req()

    def test_default_mode_auto_commits_successful_tip(self):
        session = Session(
            capacity_of_str_len=0,
            session_id="auto",
            streaming=True,
        )
        first = session.create_req(
            _recv("r1", [1, 2, 3]),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        first.output_ids.extend([4, 5])
        first._refresh_fill_ids()
        session.finish_req(first)

        self.assertEqual(session.floor, 5)
        self.assertEqual(first.streaming_session_floor, 5)

    def test_manual_commit_persists_appended_context_across_abort(self):
        first = self._create("r1", [1, 2, 3])
        self._decode_and_finish(first, [4, 5])
        self.assertEqual(self.session.floor, 0)

        second = self._create("r2", [6, 7], commit_to=7)
        cache = SimpleNamespace(calls=[])
        cache.truncate_session = lambda session_id, target: cache.calls.append(
            ("truncate", session_id, target)
        )
        cache.commit_session = lambda session_id, floor: cache.calls.append(
            ("commit", session_id, floor)
        )
        self.session.commit_prepared_req(second, cache)

        self.assertEqual(cache.calls, [("commit", "s", 7)])
        self.assertEqual(self.session.floor, 7)
        second.output_ids.extend([8, 9])
        second._refresh_fill_ids()
        self.session.abort_req()

        third = self._create("r3", [10])
        self.assertEqual(list(third.origin_input_ids), list(range(1, 8)) + [10])
        self.assertEqual(third.streaming_session_floor, 7)

    def test_commit_and_truncate_validation_are_non_destructive(self):
        first = self._create("r1", [1, 2, 3, 4])
        self._decode_and_finish(first, [5, 6])

        commit = self._create("commit", [], max_new_tokens=0, commit_to=4)
        cache = SimpleNamespace(
            truncate_session=lambda session_id, target: None,
            commit_session=lambda session_id, floor: None,
        )
        self.session.commit_prepared_req(commit, cache)
        commit.update_finish_state()
        self.session.finish_req(commit)
        self.assertEqual(self.session.floor, 4)

        origin_before = list(commit.origin_input_ids)
        output_before = list(commit.output_ids)
        with get_parallel().override(tp_rank=0):
            below_floor = self._create("below", [9], truncate_to=3)
        self.assertIsNotNone(below_floor.to_finish)
        self.assertFalse(self.session._inflight)
        self.assertEqual(list(commit.origin_input_ids), origin_before)
        self.assertEqual(list(commit.output_ids), output_before)

        with get_parallel().override(tp_rank=0):
            past_post_append_tip = self._create("past", [9], commit_to=8)
        self.assertIsNotNone(past_post_append_tip.to_finish)
        self.assertFalse(self.session._inflight)
        self.assertEqual(list(commit.origin_input_ids), origin_before)
        self.assertEqual(list(commit.output_ids), output_before)

    def test_recurrent_session_rejects_rewind_non_destructively(self):
        session = Session(
            capacity_of_str_len=0,
            session_id="mamba-session",
            streaming=True,
            supports_mamba=True,
        )
        first = session.create_req(
            _recv("r1", [1, 2, 3]),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        first.output_ids.extend([4, 5])
        first._refresh_fill_ids()
        session.finish_req(first)
        origin_before = list(first.origin_input_ids)
        output_before = list(first.output_ids)

        with get_parallel().override(tp_rank=0):
            rejected = session.create_req(
                _recv("rewind", [9], truncate_to=4),
                tokenizer=None,
                vocab_size=VOCAB,
            )

        self.assertIsNotNone(rejected.to_finish)
        self.assertFalse(session._inflight)
        self.assertEqual(list(first.origin_input_ids), origin_before)
        self.assertEqual(list(first.output_ids), output_before)

    def test_recurrent_session_allows_truncate_to_tip_noop(self):
        session = Session(
            capacity_of_str_len=0,
            session_id="mamba-session",
            streaming=True,
            supports_mamba=True,
        )
        first = session.create_req(
            _recv("r1", [1, 2, 3]),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        first.output_ids.extend([4, 5])
        first._refresh_fill_ids()
        session.finish_req(first)

        no_op = session.create_req(
            _recv("no-op", [], max_new_tokens=0, truncate_to=5),
            tokenizer=None,
            vocab_size=VOCAB,
        )

        self.assertIsNone(no_op.to_finish)
        self.assertTrue(session._inflight)
        self.assertEqual(list(no_op.origin_input_ids), [1, 2, 3, 4, 5])

    def test_non_streaming_session_rejects_empty_token_input(self):
        session = Session(capacity_of_str_len=0, session_id="ordinary")

        with get_parallel().override(tp_rank=0):
            rejected = session.create_req(
                _recv("empty", [], max_new_tokens=0),
                tokenizer=None,
                vocab_size=VOCAB,
            )

        self.assertIsNotNone(rejected.to_finish)
        self.assertFalse(session._inflight)

    def test_non_streaming_session_rejects_streaming_mutation_riders(self):
        for riders in (
            {"truncate_to": 0},
            {"commit_to": 0},
            {"expected_tip": 0},
        ):
            with self.subTest(riders=riders):
                session = Session(capacity_of_str_len=0, session_id="ordinary")
                with get_parallel().override(tp_rank=0):
                    rejected = session.create_req(
                        _recv("bad", [1], **riders),
                        tokenizer=None,
                        vocab_size=VOCAB,
                    )

                self.assertEqual(
                    rejected.to_finish.status_code,
                    HTTPStatus.BAD_REQUEST,
                )
                self.assertEqual(
                    rejected.to_finish.err_type,
                    "BadRequestError",
                )
                self.assertEqual(len(session.req_nodes), 0)

    def test_empty_first_turn_is_prepared_only_for_mutation_completion(self):
        empty = self._create("empty", [], max_new_tokens=0, truncate_to=0)

        self.assertIsNone(empty.to_finish)
        self.assertTrue(self.session._inflight)
        self.assertEqual(list(empty.origin_input_ids), [])

        cache = SimpleNamespace(calls=[])
        cache.truncate_session = lambda session_id, target: cache.calls.append(
            (session_id, target)
        )
        self.session.commit_prepared_req(empty, cache)
        empty.update_finish_state()
        self.session.finish_req(empty)

        self.assertEqual(cache.calls, [("s", 0)])
        self.assertTrue(empty.finished())
        self.assertFalse(self.session._inflight)

    def test_empty_context_cannot_start_decode_burst(self):
        with get_parallel().override(tp_rank=0):
            rejected = self._create("empty-decode", [], max_new_tokens=1)

        self.assertIsNotNone(rejected.to_finish)
        self.assertFalse(self.session._inflight)

    def test_controller_info_is_read_only_and_durable(self):
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: False,
            streaming_session_cache_snapshot=lambda session_id: (
                StreamingSessionCacheSnapshot(protected=64, held_tokens=192)
            ),
        )
        controller = SessionController(tree_cache)
        controller.open(
            OpenSessionReqInput(
                capacity_of_str_len=0,
                session_id="info-session",
                streaming=True,
                manual_commit=True,
            )
        )
        session = controller.get("info-session")
        session.committed_origin_len = 224
        session.committed_output_len = 32
        session.floor = 128
        session.last_rid = "last-rid"
        session._inflight = True
        last_active_before = session.last_active_time

        info = controller.get_info("info-session")

        self.assertEqual(
            info,
            type(info)(
                exists=True,
                tip=256,
                floor=128,
                protected=64,
                inflight=True,
                held_tokens=192,
                last_rid="last-rid",
            ),
        )
        self.assertEqual(session.last_active_time, last_active_before)
        self.assertEqual(
            controller.get_info("missing"),
            type(info)(
                exists=False,
                tip=0,
                floor=0,
                protected=0,
                inflight=False,
                held_tokens=0,
                last_rid=None,
            ),
        )


if __name__ == "__main__":
    unittest.main()
