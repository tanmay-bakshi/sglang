from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    GenerateReqInput,
    OpenSessionReqInput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}

_STREAMING_SESSION_PREFIX_TOKENS: Final = 40_960
_STREAMING_SESSION_DELTA_TOKENS: Final = 64
_STREAMING_SESSION_BURST_TOKENS: Final = 8
_STREAMING_SESSION_TOKEN_BASE: Final = 2_048
_STREAMING_SESSION_TOKEN_SPAN: Final = 4_096
_STREAMING_SESSION_EXTRA_KEY: Final = "streaming-session-small-extend-warmup-v1"


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


def _streaming_session_warmup_tokens(length: int, offset: int = 0) -> list[int]:
    """Build deterministic in-vocabulary token IDs for session warmup.

    :param length: Number of token IDs to build.
    :param offset: Offset within the deterministic token cycle.
    :returns: Stable token IDs independent of tokenizer state.
    """
    return [
        _STREAMING_SESSION_TOKEN_BASE
        + ((offset + index) % _STREAMING_SESSION_TOKEN_SPAN)
        for index in range(length)
    ]


async def _drain_streaming_session_warmup_request(
    tokenizer_manager: TokenizerManager,
    request: GenerateReqInput,
) -> None:
    """Consume one warmup request through its terminal response.

    :param tokenizer_manager: Manager serving the warmup request.
    :param request: Raw-token generation request to drain.
    """
    async for _ in tokenizer_manager.generate_request(request, None):
        pass


@warmup("streaming_session_small_extend")
async def streaming_session_small_extend(
    disaggregation_mode: str,
    tokenizer_manager: TokenizerManager,
) -> None:
    """Warm the deep-session 64-token extend and 8-token decode shape.

    :param disaggregation_mode: Active prefill/decode disaggregation mode.
    :param tokenizer_manager: Manager that owns the streaming session.
    :raises ValueError: If invoked for a disaggregated server.
    """
    if disaggregation_mode != "null":
        raise ValueError(
            "streaming_session_small_extend requires disaggregation_mode='null'."
        )

    session_id = await tokenizer_manager.open_session(
        OpenSessionReqInput(
            capacity_of_str_len=0,
            streaming=True,
        ),
        None,
    )
    if session_id is None:
        raise RuntimeError("Failed to open the streaming-session warmup session.")

    try:
        await _drain_streaming_session_warmup_request(
            tokenizer_manager,
            GenerateReqInput(
                input_ids=_streaming_session_warmup_tokens(
                    _STREAMING_SESSION_PREFIX_TOKENS
                ),
                session_params={"id": session_id, "rid": None},
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 0,
                },
                stream=False,
                log_metrics=False,
                extra_key=_STREAMING_SESSION_EXTRA_KEY,
            ),
        )
        await _drain_streaming_session_warmup_request(
            tokenizer_manager,
            GenerateReqInput(
                input_ids=_streaming_session_warmup_tokens(
                    _STREAMING_SESSION_DELTA_TOKENS,
                    offset=_STREAMING_SESSION_PREFIX_TOKENS,
                ),
                session_params={"id": session_id, "rid": None},
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": _STREAMING_SESSION_BURST_TOKENS,
                    "ignore_eos": True,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                stream=True,
                log_metrics=False,
                extra_key=_STREAMING_SESSION_EXTRA_KEY,
            ),
        )
    finally:
        await tokenizer_manager.close_session(
            CloseSessionReqInput(session_id=session_id),
            None,
        )


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()
