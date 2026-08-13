import dataclasses
from array import array

import pytest
import zmq

from sglang.srt.disaggregation.terminal_progress.output_projection import (
    FrozenPrefillGatewayOutputShell,
    PrefillTerminalGatewayOutputProjection,
    PrefillTerminalGatewayPayloadEncoder,
    TerminalGatewayResultSlot,
    freeze_prefill_gateway_output_shell,
)
from sglang.srt.managers.io_struct import BatchTokenIDOutput, sock_recv
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StableResultSlot(TerminalGatewayResultSlot):
    """Deterministic stable result slot for CPU-only projection tests."""

    _generation: bytes
    _next_token_id: int

    def __init__(self, generation: bytes, next_token_id: int) -> None:
        """Create one immutable test slot.

        :param generation: Exact result-slot generation.
        :param next_token_id: Producer-complete token identifier.
        """

        self._generation = generation
        self._next_token_id = next_token_id

    @property
    def generation(self) -> bytes:
        """Return the exact result-slot generation.

        :returns: Fixed-width test generation.
        """

        return self._generation

    def read_next_token_id(self) -> int:
        """Read the stable token identifier.

        :returns: Producer-complete token identifier.
        """

        return self._next_token_id


def _shell() -> FrozenPrefillGatewayOutputShell:
    """Build one production-shaped non-logprob output shell.

    :returns: Complete immutable prefill shell.
    """

    return FrozenPrefillGatewayOutputShell(
        rid="request-17",
        http_worker_ipc="ipc:///tmp/http-worker-17",
        origin_tail_ids=(71, 72, 73, 74, 75),
        read_offset=5,
        decoded_text="",
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
        prompt_tokens=8192,
        reasoning_tokens=0,
        cached_tokens=128,
        cached_tokens_details=(("device", 128), ("host", 0)),
        image_tokens=0,
        audio_tokens=0,
        video_tokens=0,
        retraction_count=0,
        dp_rank=0,
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        spec_num_block_accept_tokens=0,
        spec_num_cap_tokens=0,
        spec_correct_drafts_histogram=(),
        spec_cap_lens_histogram=(),
    )


def _projection() -> PrefillTerminalGatewayOutputProjection:
    """Build one exact shell, slot, and event binding.

    :returns: Complete production-shaped projection.
    """

    return PrefillTerminalGatewayOutputProjection(
        shell=_shell(),
        result_slot=_StableResultSlot(bytes.fromhex("31" * 16), 42),
        producer_event_generation=bytes.fromhex("52" * 16),
    )


def _request() -> Req:
    """Build one scheduler-owned request at its pre-forward boundary.

    :returns: Plain-token request supported by terminal publication.
    """

    request = Req(
        rid="request-17",
        origin_input_text="",
        origin_input_ids=array("q", (10, 11, 12, 13, 14, 15, 16)),
        origin_input_ids_unpadded=array("q", (10, 11, 12, 13, 14, 15, 16)),
        sampling_params=SamplingParams(
            max_new_tokens=16,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            no_stop_trim=True,
        ),
        http_worker_ipc="ipc:///tmp/http-worker-17",
    )
    request.cached_tokens = 4
    request.reasoning_tokens = 2
    request.retraction_count = 1
    request.spec_verify_ct = 3
    request.spec_num_correct_drafts = 4
    request.spec_num_block_accept_tokens = 5
    request.spec_num_cap_tokens = 6
    request.spec_correct_drafts_histogram = [7, 8]
    request.spec_cap_lens_histogram = [9, 10]
    return request


def _decode_payload(encoded_payload: bytes) -> BatchTokenIDOutput:
    """Decode active IPC bytes through the production socket receiver.

    :param encoded_payload: Bytes emitted by the publisher encoder.
    :returns: Decoded scheduler output object.
    """

    context = zmq.Context(io_threads=1)
    sender = context.socket(zmq.PAIR)
    receiver = context.socket(zmq.PAIR)
    endpoint = "inproc://terminal-output-projection"
    receiver.bind(endpoint)
    sender.connect(endpoint)
    try:
        sender.send(encoded_payload)
        payload = sock_recv(receiver)
    finally:
        sender.close(linger=0)
        receiver.close(linger=0)
        context.term()
    if type(payload) is not BatchTokenIDOutput:
        raise TypeError("encoded projection did not produce BatchTokenIDOutput")
    return payload


def test_encoder_reconstructs_exact_prefill_completion_without_req() -> None:
    """The publisher needs only immutable shell state and a stable result row."""

    projection = _projection()
    encoded = PrefillTerminalGatewayPayloadEncoder().encode(projection)
    payload = _decode_payload(encoded.encoded_payload)

    assert encoded.projection_digest == projection.digest
    assert payload.rids == ["request-17"]
    assert payload.http_worker_ipcs == ["ipc:///tmp/http-worker-17"]
    assert payload.finished_reasons == [{"type": "length", "length": 0}]
    assert list(payload.decode_ids[0]) == [71, 72, 73, 74, 75, 42]
    assert payload.read_offsets == [5]
    assert list(payload.output_ids[0]) == [42]
    assert payload.prompt_tokens == [8192]
    assert payload.completion_tokens == [1]
    assert payload.cached_tokens_details == [{"device": 128, "host": 0}]
    assert payload.time_stats is None


def test_shell_builder_freezes_request_without_crossing_stream_boundary() -> None:
    """Pre-forward projection must not mutate incremental detokenizer state."""

    request = _request()
    request.output_ids.extend((31, 32))
    before_output_ids = tuple(request.output_ids)

    shell = freeze_prefill_gateway_output_shell(
        request,
        cached_tokens_details={"host": 0, "device": 4},
        dp_rank=2,
    )

    assert shell.rid == "request-17"
    assert shell.origin_tail_ids == (12, 13, 14, 15, 16, 31, 32)
    assert shell.read_offset == 5
    assert shell.prompt_tokens == 7
    assert shell.cached_tokens_details == (("device", 4), ("host", 0))
    assert shell.reasoning_tokens == 2
    assert shell.dp_rank == 2
    assert shell.spec_verify_ct == 3
    assert shell.spec_correct_drafts_histogram == (7, 8)
    assert shell.spaces_between_special_tokens is False
    assert shell.no_stop_trim is True
    assert request.surr_offset is None
    assert request.read_offset is None
    assert tuple(request.output_ids) == before_output_ids


def test_shell_builder_rejects_result_modes_without_stable_slots() -> None:
    """Unsupported mutable result fields cannot fall back to scheduler reads."""

    request = _request()
    request.return_logprob = True

    with pytest.raises(ValueError, match="plain token output"):
        freeze_prefill_gateway_output_shell(
            request,
            cached_tokens_details=None,
            dp_rank=0,
        )


def test_projection_digest_binds_shell_slot_and_producer_generations() -> None:
    """No result row or producing event can be substituted under one digest."""

    projection = _projection()
    changed_shell = dataclasses.replace(
        projection,
        shell=dataclasses.replace(projection.shell, cached_tokens=127),
    )
    changed_slot = dataclasses.replace(
        projection,
        result_slot=_StableResultSlot(bytes.fromhex("32" * 16), 42),
    )
    changed_event = dataclasses.replace(
        projection,
        producer_event_generation=bytes.fromhex("53" * 16),
    )

    assert len({
        projection.digest,
        changed_shell.digest,
        changed_slot.digest,
        changed_event.digest,
    }) == 4


def test_projection_rejects_malformed_generations() -> None:
    """Slot and producer generations are fixed-width lifecycle identities."""

    with pytest.raises(ValueError, match="result slot generation"):
        PrefillTerminalGatewayOutputProjection(
            shell=_shell(),
            result_slot=_StableResultSlot(b"short", 42),
            producer_event_generation=bytes.fromhex("52" * 16),
        )
    with pytest.raises(ValueError, match="producer event generation"):
        PrefillTerminalGatewayOutputProjection(
            shell=_shell(),
            result_slot=_StableResultSlot(bytes.fromhex("31" * 16), 42),
            producer_event_generation=b"short",
        )


def test_shell_rejects_mutable_or_incomplete_identity_fields() -> None:
    """Mutable list state and ambiguous cache details cannot cross threads."""

    with pytest.raises(ValueError, match="origin_tail_ids"):
        dataclasses.replace(_shell(), origin_tail_ids=[71, 72])
    with pytest.raises(ValueError, match="duplicate keys"):
        dataclasses.replace(
            _shell(),
            cached_tokens_details=(("device", 128), ("device", 127)),
        )


def test_encoder_rejects_another_projection_family() -> None:
    """The concrete encoder cannot reinterpret an unrelated projection."""

    with pytest.raises(
        TypeError,
        match="PrefillTerminalGatewayOutputProjection",
    ):
        PrefillTerminalGatewayPayloadEncoder().encode(object())
