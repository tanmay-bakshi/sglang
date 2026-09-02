# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import hashlib
import logging
import struct
import time
import uuid
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from http import HTTPStatus
from itertools import chain
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.session.errors import (
    STREAMING_SESSION_CONFLICT_ERROR_TYPE,
    StreamingSessionDemotionError,
    StreamingSessionInfoUnavailableError,
)
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import (
        BasePrefixCache,
        KVComponentResidency,
    )

StreamingSessionReapCause = Literal["close", "timeout"]
StreamingSessionReapObserver = Callable[[StreamingSessionReapCause], None]


logger = logging.getLogger(__name__)


_LINEAGE_DIGEST_DOMAIN = b"sglang.streaming-session.token-history:v1\x00"
_LINEAGE_DIGEST_PREFIX = "sha256:v1:"
_TOKEN_ID_STRUCT = struct.Struct(">q")


def compute_lineage_digest(token_ids: Iterable[int]) -> str:
    """Compute the canonical digest for a streaming-session token history.

    :param token_ids: Token identifiers in absolute lineage order.
    :returns: Versioned SHA-256 digest over canonical signed 64-bit token IDs.
    """
    digest = hashlib.sha256()
    digest.update(_LINEAGE_DIGEST_DOMAIN)
    for token_id in token_ids:
        digest.update(_TOKEN_ID_STRUCT.pack(token_id))
    return _LINEAGE_DIGEST_PREFIX + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StreamingSessionInfo:
    """Read-only durable state of a streaming session.

    :ivar exists: Whether the requested session is open.
    :ivar tip: Absolute token offset of the durable lineage tip.
    :ivar lineage_digest: Canonical digest of the durable token history.
    :ivar floor: Earliest token offset to which the session may roll back.
    :ivar protected: Tokens protected through shared radix-tree ownership.
    :ivar inflight: Whether a mutation currently owns the session.
    :ivar held_tokens: Tokens held exclusively by the session cache slot.
    :ivar last_rid: Request that established the latest durable boundary.
    """

    exists: bool
    tip: int
    lineage_digest: str | None
    floor: int
    protected: int
    inflight: bool
    held_tokens: int
    last_rid: str | None


@dataclass(frozen=True, slots=True)
class StreamingSessionInventory:
    """Recovery inventory entry for one open streaming session.

    :ivar session_id: Public session identifier.
    :ivar lineage_generation: Generation incremented by history rewrites.
    :ivar tip: Absolute token offset of the durable lineage tip.
    :ivar lineage_digest: Canonical digest of the durable token history.
    :ivar floor: Earliest token offset to which the session may roll back.
    :ivar full: Full-attention KV page residency.
    :ivar swa: Sliding-window-attention KV page residency.
    """

    session_id: str
    lineage_generation: int
    tip: int
    lineage_digest: str
    floor: int
    full: KVComponentResidency
    swa: KVComponentResidency


@dataclass(frozen=True, slots=True)
class StreamingSessionDemotionContext:
    """Immutable inputs and recovery coordinates for one demotion transaction.

    :ivar session_id: Public session identifier.
    :ivar token_ids: Complete committed token lineage.
    :ivar extra_key: Radix cache classification key.
    :ivar cache_salt: Radix cache namespace salt.
    :ivar priority: Cache eviction priority inherited from the last request.
    :ivar tip: Absolute durable token offset.
    :ivar lineage_digest: Canonical digest of the durable token history.
    :ivar lineage_generation: Generation incremented by history rewrites.
    :ivar already_demoted: Whether the transaction has already committed.
    """

    session_id: str
    token_ids: array
    extra_key: str | None
    cache_salt: str | None
    priority: int
    tip: int
    lineage_digest: str
    lineage_generation: int
    already_demoted: bool


class SessionReqNode:
    def __init__(
        self,
        req: Req,
        parent: Optional[SessionReqNode] = None,
        children=None,
    ):
        self.req = req
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = [] if not children else children

    def clear_children(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)
        self.children = []

    def clear(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)

        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()
        del req_dict[self.req.rid]

    def abort(self):
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()

    def __str__(self):
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        if len(self.children) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.children[0].req.rid
            ret = self.children[0]._str_helper(prefix)
            for child in self.children[1:]:
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
        supports_mamba: bool = False,
        manual_commit: bool = False,
    ):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.streaming = streaming
        self.timeout = timeout
        self.supports_mamba = supports_mamba
        self.manual_commit = manual_commit
        self.lineage_generation: int = 0
        self.floor: int = 0
        self.last_rid: str | None = None
        self.last_active_time: float = time.monotonic()
        self.req_nodes: Dict[str, SessionReqNode] = {}
        self.close_on_finish_cause: StreamingSessionReapCause | None = None
        self._inflight: bool = False
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        self.committed_origin_len: Optional[int] = None
        self.committed_unpadded_len: Optional[int] = None
        self.committed_fill_len: Optional[int] = None
        self.committed_output_len: Optional[int] = None
        self._lineage_digest: str = compute_lineage_digest(())

    def is_timed_out(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    def current_tip(self) -> int:
        """Return the durable context length in constant time.

        :returns: The tip at the latest successful or prepared durable boundary.
        """
        if self.committed_origin_len is None:
            assert self.committed_output_len is None
            return 0
        assert self.committed_output_len is not None
        return self.committed_origin_len + self.committed_output_len

    def current_digest(self) -> str:
        """Return the digest of the latest durable token history.

        :returns: Versioned canonical lineage digest.
        """
        return self._lineage_digest

    def committed_token_ids(self) -> array:
        """Return a private copy of the complete durable token lineage.

        :returns: Committed token identifiers in absolute lineage order.
        """
        if len(self.req_nodes) == 0:
            return array("q")
        [last_req_node] = self.req_nodes.values()
        token_ids, _ = self._committed_token_arrays(last_req_node.req)
        return token_ids

    @staticmethod
    def _strip_tokenized_bos_token(req: TokenizedGenerateReqInput, tokenizer) -> None:
        """Trim a tokenizer-added BOS on an appended text turn.

        :param req: Tokenized text request whose leading BOS is synthetic.
        :param tokenizer: Tokenizer that may have added the BOS token.
        """
        if not (
            tokenizer is not None
            and req.input_ids
            and req.input_ids[0] == tokenizer.bos_token_id
        ):
            return
        req.input_ids = req.input_ids[1:]
        if req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if item.offsets:
                    if any(s == 0 for s, _ in item.offsets):
                        logging.warning(
                            "mm_item offset starts at 0 (BOS position), "
                            "clamping to 0 after BOS strip"
                        )
                    item.offsets = [
                        (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets
                    ]

    def _share_token_arrays(self, last_req: Req, new_input_ids):
        """Plain streaming append: reuse last_req's token arrays in place.

        Trims each array back to its committed length first — an earlier turn
        may have appended its tokens and then aborted before finish_req, and
        req_nodes still points at last_req, so anything beyond the committed
        lengths is unconfirmed. Then extends with last turn's output and the
        new input. Returns (input_ids, input_ids_unpadded, carry_fill);
        carry_fill (== the new origin) spares the first fill_ids rebuild.
        """
        assert self.committed_output_len is not None
        out_tail = last_req.output_ids[: self.committed_output_len]

        input_ids = last_req.origin_input_ids
        del input_ids[self.committed_origin_len :]
        if last_req.origin_input_ids_unpadded is input_ids:
            input_ids_unpadded = input_ids
        else:
            input_ids_unpadded = last_req.origin_input_ids_unpadded
            del input_ids_unpadded[self.committed_unpadded_len :]

        carry_fill = last_req.full_untruncated_fill_ids
        if (
            not isinstance(carry_fill, array)
            or carry_fill is input_ids
            or carry_fill is input_ids_unpadded
        ):
            # Unexpected type or aliased with an origin array (extending it
            # below would double-append): let _refresh_fill_ids rebuild.
            carry_fill = None
        else:
            del carry_fill[self.committed_fill_len :]
            baked = len(carry_fill) - len(input_ids)
            if 0 <= baked <= len(out_tail):
                carry_fill.extend(out_tail[baked:])
                carry_fill.extend(new_input_ids)
            else:
                carry_fill = None

        input_ids.extend(out_tail)
        input_ids.extend(new_input_ids)
        if input_ids_unpadded is not input_ids:
            input_ids_unpadded.extend(out_tail)
            input_ids_unpadded.extend(new_input_ids)
        return input_ids, input_ids_unpadded, carry_fill

    def _committed_token_arrays(self, last_req: Req) -> tuple[array, array]:
        """Materialize the last successful context from its split arrays."""
        assert self.committed_origin_len is not None
        assert self.committed_unpadded_len is not None
        assert self.committed_output_len is not None

        output_ids = last_req.output_ids[: self.committed_output_len]
        input_ids = array("q", last_req.origin_input_ids[: self.committed_origin_len])
        input_ids.extend(output_ids)

        input_ids_unpadded = array(
            "q",
            last_req.origin_input_ids_unpadded[: self.committed_unpadded_len],
        )
        input_ids_unpadded.extend(output_ids)
        return input_ids, input_ids_unpadded

    def _truncate_token_arrays(self, target: int) -> None:
        """Move the last successful logical context to ``target``."""
        if len(self.req_nodes) == 0:
            assert target == 0
            return

        [last_req_node] = self.req_nodes.values()
        last_req = last_req_node.req
        input_ids, input_ids_unpadded = self._committed_token_arrays(last_req)
        del input_ids[target:]
        del input_ids_unpadded[target:]

        last_req.origin_input_ids = input_ids
        last_req.origin_input_ids_unpadded = input_ids_unpadded
        last_req.output_ids = array("q")
        last_req.full_untruncated_fill_ids = array("q", input_ids)
        self.committed_origin_len = len(input_ids)
        self.committed_unpadded_len = len(input_ids_unpadded)
        self.committed_fill_len = len(input_ids)
        self.committed_output_len = 0

    def _concat_token_arrays(
        self, last_req: Req, req: TokenizedGenerateReqInput, session_params
    ):
        """Copy-based assembly for replace/offset/drop_previous_output turns."""
        output_len = (
            self.committed_output_len
            if self.streaming
            else last_req.sampling_params.max_new_tokens
        )
        assert output_len is not None
        out_tail = last_req.output_ids[:output_len]

        input_ids = last_req.origin_input_ids + out_tail
        if session_params.drop_previous_output:
            input_ids = last_req.origin_input_ids[:]
        if session_params.offset is not None:
            input_ids = input_ids[: session_params.offset] + req.input_ids
        else:
            input_ids += req.input_ids

        input_ids_unpadded = last_req.origin_input_ids_unpadded + out_tail
        if session_params.drop_previous_output:
            input_ids_unpadded = last_req.origin_input_ids_unpadded[:]
        if session_params.offset is not None:
            input_ids_unpadded = (
                input_ids_unpadded[: session_params.offset] + req.input_ids
            )
        else:
            input_ids_unpadded += req.input_ids
        return input_ids, input_ids_unpadded

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
    ):
        assert req.session_params is not None
        session_params = req.session_params

        last_req_node = None
        last_req = None
        abort = False
        abort_message = ""
        abort_status_code: HTTPStatus | int = HTTPStatus.BAD_REQUEST
        abort_err_type = "BadRequestError"
        abort_error_data: dict[str, object] | None = None
        if self.streaming:
            if self._inflight:
                abort = True
                abort_message = "Streaming session already has an active request."
            elif (
                session_params.expected_tip is not None
                and session_params.expected_tip != self.current_tip()
            ):
                current_tip = self.current_tip()
                abort = True
                abort_message = (
                    "Streaming session expected_tip conflict for session "
                    f"{self.session_id}: expected {session_params.expected_tip}, "
                    f"current tip is {current_tip}."
                )
                abort_status_code = HTTPStatus.CONFLICT
                abort_err_type = STREAMING_SESSION_CONFLICT_ERROR_TYPE
                abort_error_data = {
                    "observed_tip": current_tip,
                    "observed_digest": self.current_digest(),
                    "lineage_generation": self.lineage_generation,
                }
            elif (
                session_params.expected_digest is not None
                and session_params.expected_digest != self.current_digest()
            ):
                current_tip = self.current_tip()
                current_digest = self.current_digest()
                abort = True
                abort_message = (
                    "Streaming session expected_digest conflict for session "
                    f"{self.session_id}: expected {session_params.expected_digest}, "
                    f"current digest is {current_digest}."
                )
                abort_status_code = HTTPStatus.CONFLICT
                abort_err_type = STREAMING_SESSION_CONFLICT_ERROR_TYPE
                abort_error_data = {
                    "observed_tip": current_tip,
                    "observed_digest": current_digest,
                    "lineage_generation": self.lineage_generation,
                }
            elif session_params.replace:
                abort = True
                abort_message = "Streaming sessions do not support replace."
            elif session_params.drop_previous_output:
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset is not None:
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif (
                req.sampling_params.stop_strs is not None
                and len(req.sampling_params.stop_strs) > 0
            ) or (
                req.sampling_params.stop_regex_strs is not None
                and len(req.sampling_params.stop_regex_strs) > 0
            ):
                abort = True
                abort_message = (
                    "Streaming sessions support stop_token_ids only; string and "
                    "regular-expression stop conditions require application-side "
                    "token handling."
                )
            elif self.req_nodes:
                assert len(self.req_nodes) == 1
                # Peek (don't pop) the single req_node. req_nodes is updated
                # only in finish_req after the request completes successfully.
                [last_req_node] = self.req_nodes.values()
                last_req = last_req_node.req

            if last_req is not None and req.input_text is not None:
                self._strip_tokenized_bos_token(req, tokenizer)

            if not abort:
                tip = self.current_tip()
                truncate_target = session_params.truncate_to
                if truncate_target is None:
                    truncate_target = tip

            if not abort and session_params.truncate_to is not None:
                if not 0 <= session_params.truncate_to <= tip:
                    abort = True
                    abort_message = (
                        "Streaming session truncate_to must be between the commit "
                        f"floor ({self.floor}) and "
                        f"the current tip ({tip}), got {session_params.truncate_to}."
                    )
                elif session_params.truncate_to < self.floor:
                    abort = True
                    abort_message = (
                        "Streaming session truncate_to must be between the commit "
                        f"floor ({self.floor}) and the current tip ({tip}), got "
                        f"{session_params.truncate_to}."
                    )
                elif self.supports_mamba and session_params.truncate_to < tip:
                    abort = True
                    abort_message = (
                        "Streaming sessions backed by recurrent state do not support "
                        "truncate_to below the current tip."
                    )

            if not abort and session_params.commit_to is not None:
                post_append_tip = truncate_target + len(req.input_ids)
                if not self.floor <= session_params.commit_to <= post_append_tip:
                    abort = True
                    abort_message = (
                        "Streaming session commit_to must be between the current "
                        f"commit floor ({self.floor}) and the post-append tip "
                        f"({post_append_tip}), got {session_params.commit_to}."
                    )
        elif (
            session_params.truncate_to is not None
            or session_params.commit_to is not None
            or session_params.expected_tip is not None
            or session_params.expected_digest is not None
        ):
            abort = True
            abort_message = (
                "Non-streaming sessions do not support truncate_to, commit_to, "
                "expected_tip, or expected_digest."
            )
        elif len(req.input_ids) == 0:
            abort = True
            abort_message = "Non-streaming sessions do not support empty input_ids."
        elif session_params.replace:
            if session_params.rid is None:
                for _, req_node in self.req_nodes.items():
                    req_node.clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_children(self.req_nodes)
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        carry_fill = None
        if last_req is not None:
            # In-place sharing is only safe for the plain streaming append:
            # streaming sessions allow a single inflight request, last_req has
            # finished, and the committed_* lengths recorded by finish_req let
            # _share_token_arrays trim away tokens appended by an aborted turn.
            # offset / drop_previous_output rewrite history and must copy.
            can_share_token_arrays = (
                self.streaming
                and not abort
                and self.committed_origin_len is not None
                and not session_params.drop_previous_output
                and session_params.offset is None
                and session_params.truncate_to is None
                and session_params.commit_to is None
            )
            if self.streaming and session_params.truncate_to is not None and not abort:
                input_ids, input_ids_unpadded = self._committed_token_arrays(last_req)
                del input_ids[session_params.truncate_to :]
                del input_ids_unpadded[session_params.truncate_to :]
                input_ids.extend(req.input_ids)
                input_ids_unpadded.extend(req.input_ids)
            elif can_share_token_arrays:
                input_ids, input_ids_unpadded, carry_fill = self._share_token_arrays(
                    last_req, req.input_ids
                )
            else:
                input_ids, input_ids_unpadded = self._concat_token_arrays(
                    last_req, req, session_params
                )
        else:
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        if (
            not abort
            and self.streaming
            and len(input_ids) == 0
            and req.sampling_params.max_new_tokens != 0
        ):
            abort = True
            abort_message = (
                "Streaming sessions cannot decode from an empty token context."
            )

        new_req = Req(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            return_sampling_mask=req.return_sampling_mask,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            priority=req.priority,
            routing_key=req.routing_key,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.multimodal_inputs
        new_req.tokenizer = tokenizer
        if carry_fill is not None:
            new_req.full_untruncated_fill_ids = carry_fill
        if not abort:
            new_req.streaming_session_truncate_to = session_params.truncate_to
            new_req.streaming_session_commit_to = session_params.commit_to
            new_req.streaming_session_preburst_mutation = (
                len(req.input_ids) > 0
                or session_params.truncate_to is not None
                or session_params.commit_to is not None
            )
            new_req.streaming_session_floor = (
                session_params.commit_to
                if session_params.commit_to is not None
                else self.floor
            )

        if abort:
            new_req.set_finish_with_abort(
                abort_message,
                status_code=abort_status_code,
                err_type=abort_err_type,
                error_data=abort_error_data,
            )
        elif self.streaming:
            # req_nodes is NOT updated here — finish_req() handles it.
            self.last_active_time = time.monotonic()
            new_req.streaming_session_owns_inflight = True
            self._inflight = True
        else:
            self.last_active_time = time.monotonic()
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node

        return new_req

    def commit_prepared_req(self, req: Req, tree_cache: BasePrefixCache) -> None:
        """Commit irreversible session mutations after scheduler validation."""
        assert req.streaming_session_owns_inflight
        assert req.streaming_session_admitted is False
        req.streaming_session_admitted = True

        truncate_target = req.streaming_session_truncate_to
        if truncate_target is not None:
            rewrites_lineage = truncate_target < self.current_tip()
            tree_cache.truncate_session(self.session_id, truncate_target)
            self._truncate_token_arrays(truncate_target)
            if rewrites_lineage:
                self.lineage_generation += 1
            req.time_stats.increment_streaming_session_truncation()

        commit_target = req.streaming_session_commit_to
        if commit_target is not None:
            self.floor = commit_target
            req.streaming_session_floor = commit_target
            tree_cache.commit_session(self.session_id, commit_target)
            req.time_stats.increment_streaming_session_commit()

        if req.streaming_session_preburst_mutation:
            self._adopt_preburst_context(req)

    def _adopt_preburst_context(self, req: Req) -> None:
        """Make an admitted context mutation the durable abort boundary."""
        if len(self.req_nodes) > 0:
            [prev_node] = self.req_nodes.values()
            if prev_node.req is not req:
                prev_node.req.session = None
            self.req_nodes.clear()
        self.req_nodes[req.rid] = SessionReqNode(req)

        if len(req.full_untruncated_fill_ids) != len(req.origin_input_ids):
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)
        self.committed_output_len = 0
        self._lineage_digest = compute_lineage_digest(req.origin_input_ids)
        self.last_rid = req.rid

    def finish_req(self, req):
        """Update req_nodes after a streaming request finishes successfully."""
        assert req.streaming_session_owns_inflight
        req.streaming_session_owns_inflight = False
        self._inflight = False
        if self.req_nodes:
            [prev_node] = self.req_nodes.values()
            if prev_node.req is not req:
                prev_node.req.session = None
            self.req_nodes.clear()
        self.req_nodes[req.rid] = SessionReqNode(req)

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        tip = len(req.origin_input_ids) + finished_len
        self.last_rid = req.rid
        if not self.manual_commit:
            self.floor = tip
        req.streaming_session_floor = self.floor

        # Confirm this req's token arrays as the session's rollback point.
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)
        self.committed_output_len = finished_len
        self._lineage_digest = compute_lineage_digest(
            chain(req.origin_input_ids, req.output_ids[:finished_len])
        )

    def abort_req(self, req: Req) -> None:
        """Release the exact request that owns the in-flight session turn.

        :param req: Request whose prepared or admitted turn is being aborted.
        """
        assert req.session is self
        assert req.streaming_session_owns_inflight
        req.streaming_session_owns_inflight = False
        self._inflight = False


class SessionController:
    def __init__(
        self,
        tree_cache: BasePrefixCache,
        reap_observer: StreamingSessionReapObserver | None = None,
    ):
        self.sessions: Dict[str, Session] = {}
        self._last_reap_time: float = 0.0
        self.tree_cache = tree_cache
        self.reap_observer = reap_observer

    def __contains__(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def get_info(self, session_id: str) -> StreamingSessionInfo:
        """Return a durable session snapshot without refreshing its timeout.

        :param session_id: Session identifier to inspect.
        :returns: Current durable state, or an explicit missing-session snapshot.
        """
        session = self.sessions.get(session_id)
        if session is None:
            return StreamingSessionInfo(
                exists=False,
                tip=0,
                lineage_digest=None,
                floor=0,
                protected=0,
                inflight=False,
                held_tokens=0,
                last_rid=None,
            )
        if not session.streaming:
            raise StreamingSessionInfoUnavailableError(
                f"Session {session_id} is not a streaming session."
            )

        cache = self.tree_cache.streaming_session_cache_snapshot(session_id)
        return StreamingSessionInfo(
            exists=True,
            tip=session.current_tip(),
            lineage_digest=session.current_digest(),
            floor=session.floor,
            protected=cache.protected,
            inflight=session._inflight,
            held_tokens=cache.held_tokens,
            last_rid=session.last_rid,
        )

    def list_info(self) -> list[StreamingSessionInventory]:
        """Return recovery inventory for every open streaming session.

        :returns: Session entries ordered by identifier for stable polling.
        """
        inventory: list[StreamingSessionInventory] = []
        for session_id in sorted(self.sessions):
            session = self.sessions[session_id]
            if not session.streaming:
                continue
            cache = self.tree_cache.streaming_session_cache_snapshot(session_id)
            inventory.append(
                StreamingSessionInventory(
                    session_id=session_id,
                    lineage_generation=session.lineage_generation,
                    tip=session.current_tip(),
                    lineage_digest=session.current_digest(),
                    floor=session.floor,
                    full=cache.full,
                    swa=cache.swa,
                )
            )
        return inventory

    def demotion_context(self, session_id: str) -> StreamingSessionDemotionContext:
        """Validate and freeze the inputs for a streaming-session demotion.

        :param session_id: Session identifier to demote.
        :returns: Immutable cache identity and recovery coordinates.
        :raises StreamingSessionDemotionError: If the session cannot be demoted.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise StreamingSessionDemotionError(f"Session {session_id} does not exist.")
        if not session.streaming:
            raise StreamingSessionDemotionError(
                f"Session {session_id} is not a streaming session."
            )
        if session._inflight:
            raise StreamingSessionDemotionError(
                f"Session {session_id} has an active request."
            )
        if not self.tree_cache.supports_streaming_session_demotion():
            raise StreamingSessionDemotionError(
                "Session demotion requires UnifiedRadixCache with cache-mode HiCache "
                "and session radix references enabled."
            )
        if len(session.req_nodes) != 1:
            raise StreamingSessionDemotionError(
                f"Session {session_id} has no committed KV frontier."
            )

        [last_req_node] = session.req_nodes.values()
        last_req = last_req_node.req
        if not last_req.finished():
            raise StreamingSessionDemotionError(
                f"Session {session_id} has an unfinished request."
            )
        return StreamingSessionDemotionContext(
            session_id=session_id,
            token_ids=session.committed_token_ids(),
            extra_key=last_req.extra_key,
            cache_salt=last_req.cache_salt,
            priority=0 if last_req.priority is None else last_req.priority,
            tip=session.current_tip(),
            lineage_digest=session.current_digest(),
            lineage_generation=session.lineage_generation,
            already_demoted=self.tree_cache.is_streaming_session_demoted(session_id),
        )

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
                supports_mamba=self.tree_cache.supports_mamba(),
                manual_commit=recv_req.manual_commit,
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"
            )
            return OpenSessionReqOutput(session_id=session_id, success=True)

    def close(self, recv_req: CloseSessionReqInput):
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id, cause="close")

    def _close(self, session_id: str, cause: StreamingSessionReapCause):
        session = self.sessions[session_id]
        req = None
        has_unfinished_request = False
        if session.streaming and session._inflight:
            has_unfinished_request = True
        elif session.streaming and session.req_nodes:
            assert len(session.req_nodes) == 1
            [last_node] = session.req_nodes.values()
            req = last_node.req
            if not req.finished():
                has_unfinished_request = True

        if has_unfinished_request:
            # An in-flight request is still decoding on this session's KV
            # memory. Freeing now would corrupt the scheduler. Mark the
            # session for deferred cleanup: the request keeps its session
            # reference so cache_finished_req takes the streaming path,
            # and we schedule release_session for after it completes.
            if session.close_on_finish_cause is None:
                session.close_on_finish_cause = cause
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,
            )
            return

        # No owning request -- safe to release immediately.
        if session.streaming and session.req_nodes:
            req = next(iter(session.req_nodes.values())).req
            req.session = None

        # Release multimodal features held by session requests.
        # Session reqs skip the normal mm cleanup path (scheduler and
        # output_processor) so features stay alive until the session closes.
        seen_mm = set()
        for node in session.req_nodes.values():
            mm = node.req.multimodal_inputs
            if mm is not None and id(mm) not in seen_mm:
                seen_mm.add(id(mm))
                mm.release_features()
            node.req.multimodal_inputs = None

        self.tree_cache.release_session(session_id)
        self.tree_cache.release_radix_session(session_id)
        del self.sessions[session_id]
        if session.streaming and self.reap_observer is not None:
            self.reap_observer(cause)
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"
        )

    def maybe_reap(self, now: float, interval: float = 1.0):
        # reap sessions every second
        if now - self._last_reap_time > interval:
            self._last_reap_time = now

            # Finish deferred closes for sessions whose requests completed.
            pending = [
                sid
                for sid, session in self.sessions.items()
                if session.close_on_finish_cause is not None
                and self._all_requests_finished(session)
            ]
            for sid in pending:
                log_info_on_rank0(
                    logger, f"Deferred close ready for session {sid}, releasing."
                )
                cause = self.sessions[sid].close_on_finish_cause
                assert cause is not None
                self.sessions[sid].close_on_finish_cause = None
                self._close(sid, cause=cause)

            timed_out = [
                sid for sid, session in self.sessions.items() if session.is_timed_out()
            ]
            for sid in timed_out:
                log_info_on_rank0(logger, f"Session {sid} timed out, closing.")
                self._close(sid, cause="timeout")

    @staticmethod
    def _all_requests_finished(session: Session) -> bool:
        if not session.req_nodes:
            return True
        return all(node.req.finished() for node in session.req_nodes.values())

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
