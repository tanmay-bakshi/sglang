import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
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

    def test_prebuilt_terminal_batch_consumes_exact_request_owned_device_tokens(
        self,
    ) -> None:
        """Wait each adoption event and pass the same device tensors to DFlash."""

        timeline: list[str] = []
        stream = MagicMock()
        first_event = object()
        second_event = object()
        stream.wait_event.side_effect = lambda event: timeline.append(
            f"wait:{id(event)}"
        )
        first_token = MagicMock()
        second_token = MagicMock()
        first_token.record_stream.side_effect = lambda owner: timeline.append(
            f"record:first:{id(owner)}"
        )
        second_token.record_stream.side_effect = lambda owner: timeline.append(
            f"record:second:{id(owner)}"
        )
        requests = [
            SimpleNamespace(
                output_ids=[101],
                grammar=None,
                pd_dflash_boundary_token_id=first_token,
                pd_dflash_boundary_completion_event=first_event,
            ),
            SimpleNamespace(
                output_ids=[202],
                grammar=None,
                pd_dflash_boundary_token_id=second_token,
                pd_dflash_boundary_completion_event=second_event,
            ),
        ]
        draft_input = object()
        concatenated_tokens = object()
        spec_algorithm = MagicMock()
        spec_algorithm.is_dflash.return_value = True
        spec_algorithm.build_disagg_draft_input.return_value = draft_input
        batch = SimpleNamespace(
            reqs=requests,
            tree_cache=object(),
            spec_algorithm=spec_algorithm,
            device="cuda:0",
            req_pool_indices=object(),
        )
        server_args = object()
        future_map = MagicMock()
        device_module = SimpleNamespace(current_stream=MagicMock(return_value=stream))

        with (
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.maybe_cache_unfinished_req"
            ),
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.torch.get_device_module",
                return_value=device_module,
            ),
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.torch.cat",
                return_value=concatenated_tokens,
            ) as concatenate,
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.torch.tensor",
                side_effect=AssertionError(
                    "terminal DFlash rebuilt its device tokens through a host path"
                ),
            ),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch,
                server_args,
                future_map,
            )

        self.assertEqual(
            timeline,
            [
                f"wait:{id(first_event)}",
                f"record:first:{id(stream)}",
                f"wait:{id(second_event)}",
                f"record:second:{id(stream)}",
            ],
        )
        concatenate.assert_called_once_with((first_token, second_token), dim=0)
        spec_algorithm.build_disagg_draft_input.assert_called_once_with(
            batch,
            server_args,
            concatenated_tokens,
            future_map,
        )
        self.assertIs(batch.spec_info, draft_input)
        for request in requests:
            self.assertIsNone(request.pd_dflash_boundary_token_id)
            self.assertIsNone(request.pd_dflash_boundary_completion_event)

    def test_prebuilt_batch_rejects_mixed_terminal_and_legacy_tokens(self) -> None:
        """Refuse a batch whose DFlash initialization has two ownership models."""

        terminal_token = object()
        terminal_event = object()
        requests = [
            SimpleNamespace(
                output_ids=[101],
                grammar=None,
                pd_dflash_boundary_token_id=terminal_token,
                pd_dflash_boundary_completion_event=terminal_event,
            ),
            SimpleNamespace(
                output_ids=[202],
                grammar=None,
                pd_dflash_boundary_token_id=None,
                pd_dflash_boundary_completion_event=None,
            ),
        ]
        batch = SimpleNamespace(
            reqs=requests,
            tree_cache=object(),
            spec_algorithm=MagicMock(),
            device="cuda:0",
        )

        with (
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.maybe_cache_unfinished_req"
            ),
            self.assertRaisesRegex(RuntimeError, "cannot mix terminal and legacy"),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch,
                object(),
                MagicMock(),
            )

        batch.spec_algorithm.build_disagg_draft_input.assert_not_called()
        self.assertIs(requests[0].pd_dflash_boundary_token_id, terminal_token)
        self.assertIs(requests[0].pd_dflash_boundary_completion_event, terminal_event)

    def test_prebuilt_terminal_tensor_path_has_no_host_or_eagle_access(self) -> None:
        """Keep first-token construction device-only and DFlash-specific."""

        source = inspect.getsource(
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt
        )
        forbidden_fragments = (
            ".cpu(",
            ".item(",
            "time.sleep(",
            "MetadataBuffers",
            "output_topk_p",
            "output_topk_index",
            "output_hidden_states",
        )

        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
