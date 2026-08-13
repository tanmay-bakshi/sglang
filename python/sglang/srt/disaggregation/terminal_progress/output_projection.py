import abc
import dataclasses
import hashlib
import json
from array import array

import torch

from sglang.srt.disaggregation.common.packed_staging_protocol import (
    PACKED_REQUEST_GENERATION_BYTES,
)
from sglang.srt.disaggregation.terminal_progress.publisher import (
    EncodedTerminalGatewayPayload,
    FrozenTerminalGatewayOutputProjection,
    TerminalGatewayPayloadEncoder,
)
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    CachedTokensDetails,
    encode_ipc_payload,
)
from sglang.srt.managers.schedule_batch import (
    INIT_INCREMENTAL_DETOKENIZATION_OFFSET,
    Req,
)


def _require_generation(value: bytes, label: str) -> None:
    """Validate one request-local generation.

    :param value: Candidate generation bytes.
    :param label: Reader-facing field name.
    """

    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != PACKED_REQUEST_GENERATION_BYTES:
        raise ValueError(
            f"{label} must contain {PACKED_REQUEST_GENERATION_BYTES} bytes"
        )


class TerminalGatewayResultSlot(abc.ABC):
    """Generation-bound stable host storage read by the gateway publisher."""

    @property
    @abc.abstractmethod
    def generation(self) -> bytes:
        """Return the exact result-slot generation.

        :returns: Fixed-width request-local generation.
        """

    @abc.abstractmethod
    def read_next_token_id(self) -> int:
        """Read the producer-complete next-token result.

        The request-global readiness receipt presented to the publisher proves
        that the producer event covering this slot has completed. Reading the
        slot before that receipt exists is a lifecycle violation enforced by
        the terminal owner, not a second readiness channel on this object.

        :returns: One signed 64-bit token identifier.
        """


class PinnedTerminalGatewayResultSlot(TerminalGatewayResultSlot):
    """One pinned-host next-token row retained for a request generation."""

    _generation: bytes
    _next_token_id: torch.Tensor

    def __init__(self, generation: bytes) -> None:
        """Allocate one stable result row before model submission.

        :param generation: Exact request-local row generation.
        """

        _require_generation(generation, "result slot generation")
        self._generation = generation
        self._next_token_id = torch.empty(
            (1,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )

    @property
    def generation(self) -> bytes:
        """Return the exact result-slot generation.

        :returns: Fixed-width request-local generation.
        """

        return self._generation

    @property
    def storage(self) -> torch.Tensor:
        """Return the stable row targeted by the producing CUDA stream.

        The scheduler may enqueue exactly one nonblocking device-to-host copy
        into this row before recording the producer event. The row remains
        owned by its terminal lifecycle until gateway publication and source
        retirement complete.

        :returns: One-element pinned-host ``int64`` tensor.
        """

        return self._next_token_id

    def enqueue_copy(self, next_token_id: torch.Tensor) -> None:
        """Enqueue the result copy covered by the producer event.

        :param next_token_id: One-element device tensor produced by the model.
        """

        if type(next_token_id) is not torch.Tensor:
            raise TypeError("next_token_id must be a torch.Tensor")
        if next_token_id.numel() != 1:
            raise ValueError("next_token_id must contain exactly one element")
        if next_token_id.device.type != "cuda":
            raise ValueError("next_token_id must reside in CUDA memory")
        self._next_token_id.copy_(next_token_id.reshape(1), non_blocking=True)

    def read_next_token_id(self) -> int:
        """Read the producer-complete next-token result.

        :returns: One signed 64-bit token identifier.
        """

        return int(self._next_token_id[0].item())


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenPrefillGatewayOutputShell:
    """Immutable non-logprob prefill response state frozen before submission.

    :ivar rid: Request identity owned by the tokenizer process.
    :ivar http_worker_ipc: Optional multi-tokenizer return route.
    :ivar origin_tail_ids: Prompt suffix required for incremental detokenization.
    :ivar read_offset: Offset into ``origin_tail_ids`` where decoding begins.
    :ivar decoded_text: Text already committed before this response.
    :ivar skip_special_tokens: Tokenizer special-token behavior.
    :ivar spaces_between_special_tokens: Tokenizer spacing behavior.
    :ivar no_stop_trim: Whether terminal stop text remains untrimmed.
    :ivar prompt_tokens: Unpadded prompt token count.
    :ivar reasoning_tokens: Reasoning token count at the prefill boundary.
    :ivar cached_tokens: Aggregate prefix-cache hit count.
    :ivar cached_tokens_details: Immutable cache-tier breakdown.
    :ivar image_tokens: Expanded image-token count.
    :ivar audio_tokens: Expanded audio-token count.
    :ivar video_tokens: Expanded video-token count.
    :ivar retraction_count: Number of prior scheduler retractions.
    :ivar dp_rank: Data-parallel rank which produced the request.
    :ivar speculative: Whether speculative counters are present in the active
        output schema.
    :ivar spec_verify_ct: Speculative verification count.
    :ivar spec_num_correct_drafts: Accepted draft-token count.
    :ivar spec_num_block_accept_tokens: Block-accepted draft-token count.
    :ivar spec_num_cap_tokens: Capped speculative-token count.
    :ivar spec_correct_drafts_histogram: Accepted-draft histogram.
    :ivar spec_cap_lens_histogram: Speculative cap-length histogram.
    """

    rid: str
    http_worker_ipc: str | None
    origin_tail_ids: tuple[int, ...]
    read_offset: int
    decoded_text: str
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    no_stop_trim: bool
    prompt_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cached_tokens_details: tuple[tuple[str, int | str], ...] | None
    image_tokens: int
    audio_tokens: int
    video_tokens: int
    retraction_count: int
    dp_rank: int
    speculative: bool
    spec_verify_ct: int
    spec_num_correct_drafts: int
    spec_num_block_accept_tokens: int
    spec_num_cap_tokens: int
    spec_correct_drafts_histogram: tuple[int, ...]
    spec_cap_lens_histogram: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate one complete immutable prefill response shell."""

        if type(self.rid) is not str or len(self.rid) == 0:
            raise ValueError("rid must be a non-empty string")
        if self.http_worker_ipc is not None and type(self.http_worker_ipc) is not str:
            raise TypeError("http_worker_ipc must be a string or None")
        if type(self.origin_tail_ids) is not tuple or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.origin_tail_ids
        ):
            raise ValueError("origin_tail_ids must contain non-negative integers")
        if (
            type(self.read_offset) is not int
            or self.read_offset < 0
            or self.read_offset > len(self.origin_tail_ids)
        ):
            raise ValueError("read_offset must index origin_tail_ids")
        if type(self.decoded_text) is not str:
            raise TypeError("decoded_text must be a string")
        boolean_values = (
            self.skip_special_tokens,
            self.spaces_between_special_tokens,
            self.no_stop_trim,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("tokenizer controls must be booleans")
        if type(self.speculative) is not bool:
            raise TypeError("speculative must be a boolean")
        counters = (
            self.prompt_tokens,
            self.reasoning_tokens,
            self.cached_tokens,
            self.image_tokens,
            self.audio_tokens,
            self.video_tokens,
            self.retraction_count,
            self.dp_rank,
            self.spec_verify_ct,
            self.spec_num_correct_drafts,
            self.spec_num_block_accept_tokens,
            self.spec_num_cap_tokens,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("output shell counters must be non-negative integers")
        if self.cached_tokens_details is not None:
            if type(self.cached_tokens_details) is not tuple:
                raise TypeError("cached_tokens_details must be a tuple or None")
            if any(
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or type(entry[1]) not in (int, str)
                for entry in self.cached_tokens_details
            ):
                raise TypeError("cached_tokens_details contains an invalid entry")
            keys = tuple(entry[0] for entry in self.cached_tokens_details)
            if len(set(keys)) != len(keys):
                raise ValueError("cached_tokens_details contains duplicate keys")
        histograms = (
            self.spec_correct_drafts_histogram,
            self.spec_cap_lens_histogram,
        )
        if any(
            type(histogram) is not tuple
            or any(type(value) is not int or value < 0 for value in histogram)
            for histogram in histograms
        ):
            raise ValueError("speculative histograms must contain non-negative integers")

    @property
    def digest(self) -> bytes:
        """Return the canonical immutable-shell digest.

        :returns: SHA-256 over every shell field.
        """

        encoded = json.dumps(
            dataclasses.asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"sglang.packed-terminal.prefill-output-shell.v1")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        return digest.digest()


def freeze_prefill_gateway_output_shell(
    req: Req,
    *,
    cached_tokens_details: CachedTokensDetails | None,
    dp_rank: int,
    speculative: bool,
) -> FrozenPrefillGatewayOutputShell:
    """Freeze scheduler-owned response state without mutating the request.

    The publisher appends the producer-complete boundary token from its stable
    result slot. Every other response field is captured before model
    submission, so neither the terminal owner nor the publisher ever reads the
    mutable request.

    :param req: Scheduler-owned prefill request before model submission.
    :param cached_tokens_details: Exact cache-tier breakdown reported for the
        request.
    :param dp_rank: Data-parallel rank producing the response.
    :param speculative: Whether the active scheduler emits speculative
        counters.
    :returns: Immutable non-logprob gateway response shell.
    :raises ValueError: If the request uses an unsupported result mode or has
        already crossed an output-streaming boundary.
    """

    if not isinstance(req, Req):
        raise TypeError("req must be a Req")
    unsupported_result_mode = (
        req.return_logprob
        or req.return_hidden_states
        or req.return_routed_experts
        or req.return_indexer_topk
        or req.return_sampling_mask
    )
    if unsupported_result_mode:
        raise ValueError(
            "terminal prefill publication requires plain token output"
        )
    if req.finished_reason is not None or req.finished_output is not None:
        raise ValueError("terminal prefill shell must be frozen before completion")
    if type(speculative) is not bool:
        raise TypeError("speculative must be a boolean")
    if (
        len(req.output_ids_through_stop) != 0
        or req.send_token_offset != 0
        or req.send_decode_id_offset != 0
        or req.send_output_token_logprobs_offset != 0
        or req.send_output_sampling_mask_offset != 0
        or req.surr_offset is not None
        or req.read_offset is not None
    ):
        raise ValueError(
            "terminal prefill shell requires the first output boundary"
        )

    absolute_read_offset = len(req.origin_input_ids_unpadded)
    surrounding_offset = max(
        absolute_read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET,
        0,
    )
    origin_tail_ids = tuple(req.origin_input_ids_unpadded[surrounding_offset:])
    read_offset = absolute_read_offset - surrounding_offset

    frozen_cache_details = None
    if cached_tokens_details is not None:
        frozen_cache_details = tuple(sorted(cached_tokens_details.items()))

    if (
        req.mm_image_tokens > 0
        or req.mm_audio_tokens > 0
        or req.mm_video_tokens > 0
    ):
        image_tokens = req.mm_image_tokens
        audio_tokens = req.mm_audio_tokens
        video_tokens = req.mm_video_tokens
    elif req.multimodal_inputs is not None:
        image_tokens, audio_tokens, video_tokens = (
            req.multimodal_inputs.compute_mm_token_counts()
        )
    else:
        image_tokens = 0
        audio_tokens = 0
        video_tokens = 0

    return FrozenPrefillGatewayOutputShell(
        rid=req.rid,
        http_worker_ipc=req.http_worker_ipc,
        origin_tail_ids=origin_tail_ids,
        read_offset=read_offset,
        decoded_text=req.decoded_text,
        skip_special_tokens=req.sampling_params.skip_special_tokens,
        spaces_between_special_tokens=(
            req.sampling_params.spaces_between_special_tokens
        ),
        no_stop_trim=req.sampling_params.no_stop_trim,
        prompt_tokens=len(req.origin_input_ids),
        reasoning_tokens=req.reasoning_tokens,
        cached_tokens=req.cached_tokens,
        cached_tokens_details=frozen_cache_details,
        image_tokens=image_tokens,
        audio_tokens=audio_tokens,
        video_tokens=video_tokens,
        retraction_count=req.retraction_count,
        dp_rank=dp_rank,
        speculative=speculative,
        spec_verify_ct=req.spec_verify_ct,
        spec_num_correct_drafts=req.spec_num_correct_drafts,
        spec_num_block_accept_tokens=req.spec_num_block_accept_tokens,
        spec_num_cap_tokens=req.spec_num_cap_tokens,
        spec_correct_drafts_histogram=tuple(req.spec_correct_drafts_histogram),
        spec_cap_lens_histogram=tuple(req.spec_cap_lens_histogram),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PrefillTerminalGatewayOutputProjection(FrozenTerminalGatewayOutputProjection):
    """Exact shell, result row, and producing-event generation binding.

    :ivar shell: Static response fields frozen before model submission.
    :ivar result_slot: Stable pinned row populated on the producing stream.
    :ivar producer_event_generation: Generation of the event covering the row.
    """

    shell: FrozenPrefillGatewayOutputShell
    result_slot: TerminalGatewayResultSlot
    producer_event_generation: bytes

    def __post_init__(self) -> None:
        """Validate one complete output projection."""

        if type(self.shell) is not FrozenPrefillGatewayOutputShell:
            raise TypeError("shell must be FrozenPrefillGatewayOutputShell")
        if not isinstance(self.result_slot, TerminalGatewayResultSlot):
            raise TypeError("result_slot must inherit TerminalGatewayResultSlot")
        _require_generation(
            self.result_slot.generation,
            "result slot generation",
        )
        _require_generation(
            self.producer_event_generation,
            "producer event generation",
        )

    @property
    def digest(self) -> bytes:
        """Return the shell, slot, and producer-event binding.

        :returns: SHA-256 over every immutable projection identity.
        """

        digest = hashlib.sha256()
        digest.update(b"sglang.packed-terminal.prefill-output-projection.v1")
        digest.update(self.shell.digest)
        digest.update(self.result_slot.generation)
        digest.update(self.producer_event_generation)
        return digest.digest()


class PrefillTerminalGatewayPayloadEncoder(TerminalGatewayPayloadEncoder):
    """Encode one non-logprob prefill completion on the publisher thread."""

    def encode(
        self,
        projection: FrozenTerminalGatewayOutputProjection,
    ) -> EncodedTerminalGatewayPayload:
        """Encode one producer-complete prefill output projection.

        :param projection: Exact immutable prefill projection.
        :returns: Active IPC bytes bound to the projection digest.
        """

        if type(projection) is not PrefillTerminalGatewayOutputProjection:
            raise TypeError(
                "prefill encoder requires PrefillTerminalGatewayOutputProjection"
            )
        shell = projection.shell
        next_token_id = projection.result_slot.read_next_token_id()
        if type(next_token_id) is not int or next_token_id < 0:
            raise ValueError("result slot returned an invalid token identifier")
        decode_ids = array("q", (*shell.origin_tail_ids, next_token_id))
        output_ids = array("q", (next_token_id,))
        cached_tokens_details = None
        if shell.cached_tokens_details is not None:
            cached_tokens_details = [dict(shell.cached_tokens_details)]
        spec_verify_ct = [shell.spec_verify_ct] if shell.speculative else []
        spec_num_correct_drafts = (
            [shell.spec_num_correct_drafts] if shell.speculative else []
        )
        spec_num_block_accept_tokens = (
            [shell.spec_num_block_accept_tokens] if shell.speculative else []
        )
        spec_num_cap_tokens = (
            [shell.spec_num_cap_tokens] if shell.speculative else []
        )
        spec_correct_drafts_histogram = (
            [list(shell.spec_correct_drafts_histogram)]
            if shell.speculative
            else []
        )
        spec_cap_lens_histogram = (
            [list(shell.spec_cap_lens_histogram)] if shell.speculative else []
        )
        payload = BatchTokenIDOutput(
            rids=[shell.rid],
            http_worker_ipcs=[shell.http_worker_ipc],
            finished_reasons=[{"type": "length", "length": 0}],
            decoded_texts=[shell.decoded_text],
            decode_ids=[decode_ids],
            read_offsets=[shell.read_offset],
            output_ids=[output_ids],
            skip_special_tokens=[shell.skip_special_tokens],
            spaces_between_special_tokens=[shell.spaces_between_special_tokens],
            no_stop_trim=[shell.no_stop_trim],
            prompt_tokens=[shell.prompt_tokens],
            reasoning_tokens=[shell.reasoning_tokens],
            completion_tokens=[1],
            cached_tokens=[shell.cached_tokens],
            cached_tokens_details=cached_tokens_details,
            image_tokens=[shell.image_tokens],
            audio_tokens=[shell.audio_tokens],
            video_tokens=[shell.video_tokens],
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=None,
            output_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            output_top_logprobs_val=None,
            output_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            output_token_ids_logprobs_val=None,
            output_token_ids_logprobs_idx=None,
            output_token_entropy_val=None,
            output_token_sampling_mask=None,
            output_token_sampling_logprobs=None,
            output_hidden_states=None,
            routed_experts=None,
            indexer_topk=None,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=[shell.retraction_count],
            token_steps=None,
            customized_info=None,
            dp_ranks=[shell.dp_rank],
            time_stats=None,
            spec_verify_ct=spec_verify_ct,
            spec_num_correct_drafts=spec_num_correct_drafts,
            spec_num_block_accept_tokens=spec_num_block_accept_tokens,
            spec_num_cap_tokens=spec_num_cap_tokens,
            spec_correct_drafts_histogram=spec_correct_drafts_histogram,
            spec_cap_lens_histogram=spec_cap_lens_histogram,
        )
        return EncodedTerminalGatewayPayload(
            projection_digest=projection.digest,
            encoded_payload=encode_ipc_payload(payload),
        )
