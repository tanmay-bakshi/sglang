import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashDisaggregation(unittest.TestCase):
    """Validate first-iteration DFlash state on a disaggregated decoder."""

    def test_overlap_seeds_token_relay_and_sequence_length_publication(self) -> None:
        """The first decode consumes the transferred token and committed lengths."""

        request_indices = torch.tensor([1, 3], dtype=torch.int64)
        sequence_lengths = torch.tensor([7, 11], dtype=torch.int64)
        prefill_tokens = torch.tensor([101, 202], dtype=torch.int64)
        batch = SimpleNamespace(
            device="cpu",
            enable_overlap=True,
            req_pool_indices=request_indices,
            seq_lens=sequence_lengths,
        )
        future_map = MagicMock()

        draft_input = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
            batch,
            SimpleNamespace(),
            prefill_tokens,
            future_map,
        )

        self.assertIsInstance(draft_input, DFlashDraftInputV2)
        self.assertTrue(torch.equal(draft_input.bonus_tokens, prefill_tokens))
        self.assertTrue(torch.equal(draft_input.new_seq_lens, sequence_lengths))
        self.assertIs(draft_input.future_indices, request_indices)

        future_map.publish.assert_called_once()
        published_indices, published_lengths = future_map.publish.call_args.args
        self.assertIs(published_indices, request_indices)
        self.assertIs(published_lengths, sequence_lengths)

        future_map.stash.assert_called_once()
        stashed_indices, payload = future_map.stash.call_args.args
        self.assertIs(stashed_indices, request_indices)
        self.assertIsInstance(payload, RelayPayload)
        self.assertIs(payload.bonus_tokens, prefill_tokens)

    def test_non_overlap_carries_first_iteration_state_directly(self) -> None:
        """Synchronous decode receives the same token and length seed directly."""

        sequence_lengths = torch.tensor([13], dtype=torch.int32)
        prefill_tokens = torch.tensor([303], dtype=torch.int32)
        batch = SimpleNamespace(
            device="cpu",
            enable_overlap=False,
            req_pool_indices=torch.tensor([2], dtype=torch.int64),
            seq_lens=sequence_lengths,
        )
        future_map = MagicMock()

        draft_input = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
            batch,
            SimpleNamespace(),
            prefill_tokens,
            future_map,
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, torch.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, torch.int64)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [303])
        self.assertEqual(draft_input.new_seq_lens.tolist(), [13])
        self.assertIsNone(draft_input.future_indices)
        future_map.publish.assert_not_called()
        future_map.stash.assert_not_called()

    def test_dspark_remains_outside_the_dflash_pd_contract(self) -> None:
        """DFlash initialization does not imply unvalidated DSpark PD support."""

        future_map = MagicMock()
        draft_input = SpeculativeAlgorithm.DSPARK.build_disagg_draft_input(
            SimpleNamespace(
                device="cpu",
                enable_overlap=True,
                req_pool_indices=torch.tensor([1], dtype=torch.int64),
                seq_lens=torch.tensor([5], dtype=torch.int64),
            ),
            SimpleNamespace(),
            torch.tensor([404], dtype=torch.int64),
            future_map,
        )

        self.assertIsNone(draft_input)
        future_map.publish.assert_not_called()
        future_map.stash.assert_not_called()


if __name__ == "__main__":
    unittest.main()
