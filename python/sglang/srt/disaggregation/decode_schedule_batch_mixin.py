from __future__ import annotations

import logging
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING

import torch
from sglang.srt.disaggregation.utils import _is_fake_transfer
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


class ScheduleBatchDisaggregationDecodeMixin:

    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """

        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_range.length for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)
            pre_len = len(req.prefix_indices)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                pre_len : pre_len + req.extend_range.length
            ]
            assert (
                offset + req.extend_range.length <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_range.length={req.extend_range.length}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_range.length] = chunk
            offset += req.extend_range.length

            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_range.length
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_range.length={req.extend_range.length}"

            if not req.retracted_stain:
                # Clamp to avoid double-counting: already_computed is seeded from
                # the prefill-reported cached_tokens in _commit_transfer_to_req, so
                # a decode-side prefix shorter than the prefill report must not
                # subtract from cached_tokens.
                delta = max(0, pre_len - req.already_computed)
                req.cached_tokens += delta
                req.cached_tokens_device += delta
                req.already_computed = seq_len
            req.is_retracted = False
            if getattr(req, "pd_rebootstrap_in_progress", False):
                req.pd_rebootstrap_in_progress = False
            pre_lens.append(pre_len)

        # Set fields
        self.input_ids = torch.tensor(
            sum(input_ids, array("q")), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_range.length for r in reqs]
        self.extend_logprob_start_lens = None
        self.extend_input_logprob_token_ids = None
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        terminal_boundary_tokens = tuple(
            req.pd_dflash_boundary_token_id for req in self.reqs
        )
        terminal_boundary_events = tuple(
            req.pd_dflash_boundary_completion_event for req in self.reqs
        )
        terminal_token_count = sum(
            token is not None for token in terminal_boundary_tokens
        )
        if terminal_token_count > 0:
            if not self.spec_algorithm.is_dflash():
                raise RuntimeError(
                    "request-owned terminal boundary tokens require DFlash"
                )
            for req, token, event in zip(
                self.reqs,
                terminal_boundary_tokens,
                terminal_boundary_events,
                strict=True,
            ):
                scheduler_local = _is_fake_transfer(req, server_args)
                if token is None:
                    if event is not None:
                        raise RuntimeError(
                            "terminal DFlash completion event lost its device token"
                        )
                    if not scheduler_local:
                        raise RuntimeError(
                            "terminal DFlash can compose only owner-published and "
                            "scheduler-local fake requests"
                        )
                    continue
                if scheduler_local:
                    raise RuntimeError(
                        "scheduler-local fake request retained terminal DFlash "
                        "ownership"
                    )
                if event is None:
                    raise RuntimeError(
                        "terminal DFlash boundary token lost its completion event"
                    )
        elif any(event is not None for event in terminal_boundary_events):
            raise RuntimeError("terminal DFlash completion event lost its device token")

        last_tokens: list[int] = []
        for req in self.reqs:
            last_tokens.append(req.output_ids[-1])
            maybe_cache_unfinished_req(req, self.tree_cache)
            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    # if it is not None, then the grammar is from a retracted request, and we should not
                    # accept the token as it's already accepted
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    from sglang.srt.managers.schedule_batch import FINISH_ABORT

                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    # Use to_finish (not finished_reason) so that process_batch_result_prebuilt
                    # handles the release via update_finish_state -> release_kv_cache in one place.
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    req.to_finish = FINISH_ABORT(
                        error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        if terminal_token_count > 0:
            current_stream = torch.get_device_module(self.device).current_stream()
            boundary_rows: list[torch.Tensor] = []
            for last_token, token, event in zip(
                last_tokens,
                terminal_boundary_tokens,
                terminal_boundary_events,
                strict=True,
            ):
                if token is None:
                    boundary_rows.append(
                        torch.tensor(
                            [last_token],
                            dtype=torch.int64,
                            device=self.device,
                        )
                    )
                    continue
                assert event is not None
                current_stream.wait_event(event)
                token.record_stream(current_stream)
                boundary_rows.append(token)
            last_tokens_tensor = torch.cat(tuple(boundary_rows), dim=0)
        else:
            last_tokens_tensor = torch.tensor(
                last_tokens, dtype=torch.int64, device=self.device
            )

        spec_info = self.spec_algorithm.build_disagg_draft_input(
            self,
            server_args,
            last_tokens_tensor,
            future_map,
        )
        if spec_info is not None:
            self.spec_info = spec_info
        else:
            # Non-spec: stash last token into the relay so the first DECODE's
            # resolve_forward_inputs gathers it like any other decode iter.
            future_map.stash(
                self.req_pool_indices, RelayPayload(bonus_tokens=last_tokens_tensor)
            )
            self.input_ids = None

        if terminal_token_count > 0:
            for req in self.reqs:
                req.pd_dflash_boundary_token_id = None
                req.pd_dflash_boundary_completion_event = None
