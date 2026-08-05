from typing import TYPE_CHECKING

import torch
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def build_dflash_disagg_draft_input(
    batch: "ScheduleBatch",
    last_tokens_tensor: torch.Tensor,
    future_map: "FutureMap",
) -> DFlashDraftInputV2:
    """Seed DFlash from the prefill result received by a decode worker.

    :param batch: Decode-side prebuilt batch after the transferred KV is committed.
    :param last_tokens_tensor: Target tokens sampled by the prefill worker.
    :param future_map: Overlap relay owned by the decode scheduler.
    :returns: DFlash state for the first decode iteration.
    """

    draft_input = DFlashDraftInputV2.from_next_tokens(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if not batch.enable_overlap:
        return draft_input

    draft_input.future_indices = batch.req_pool_indices
    future_map.publish(draft_input.future_indices, batch.seq_lens)
    future_map.stash(
        draft_input.future_indices,
        RelayPayload(bonus_tokens=last_tokens_tensor),
    )
    return draft_input
