"""Gemma-4 production qualification kit for the raw-token streaming-session API."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_PATH = "/models/gemma-4-31B-it-NVFP4"
SESSION_INFO_FIELDS = frozenset(
    {
        "exists",
        "tip",
        "floor",
        "protected",
        "inflight",
        "held_tokens",
        "last_rid",
    }
)
CONFLICT_ERROR_FIELDS = frozenset(
    {"message", "type", "code", "retryable", "correlation_id"}
)
CONFLICT_METRIC_NAME = "sglang:streaming_session_idempotency_conflicts_total"
SESSION_TIMEOUT_SECONDS = 1.25
SESSION_REAP_INTERVAL_SECONDS = 1.0
SESSION_INFO_POLL_SECONDS = 0.1


def _build_gemma4_context() -> tuple[list[int], list[int], PreTrainedTokenizerBase]:
    """Build a natural-token context with rollback-sensitive sentinels.

    :returns: A 4,608-token context, a raw-token question delta, and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context = [tokenizer.bos_token_id]
    context.extend(tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False))
    filler = tokenizer.encode(
        "The ledger contains ordinary archival prose used to preserve a stable "
        "natural-language inference context.\n",
        add_special_tokens=False,
    )

    def append_until(position: int) -> None:
        while len(context) < position:
            remaining = position - len(context)
            context.extend(filler[:remaining])

    context.extend(
        tokenizer.encode(
            "The current sentinel code is FIR.\n", add_special_tokens=False
        )
    )
    append_until(1_880)
    context.extend(
        tokenizer.encode(
            "The current sentinel code is ORCHID.\n", add_special_tokens=False
        )
    )
    append_until(2_185)
    context.extend(
        tokenizer.encode(
            "The current sentinel code is MAPLE.\n", add_special_tokens=False
        )
    )
    append_until(4_480)
    context.extend(
        tokenizer.encode(
            "The current sentinel code is CEDAR.\n", add_special_tokens=False
        )
    )
    append_until(4_608)
    assert len(context) == 4_608

    delta = tokenizer.encode(
        "\nRepeat only the most recent sentinel code from the context."
        "<end_of_turn>\n<start_of_turn>model\n",
        add_special_tokens=False,
    )
    return context, delta, tokenizer


@dataclass(frozen=True)
class GenerateResult:
    """Normalized raw-token generation result.

    :ivar output_ids: Sampled token IDs.
    :ivar prompt_tokens: Logical prompt length.
    :ivar completion_tokens: Number of sampled tokens.
    :ivar cached_tokens: Number of prompt tokens served from KV cache.
    :ivar rid: Request identifier returned by the server.
    """

    output_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    rid: str

    @property
    def tip(self) -> int:
        """Return the context length after this generation.

        :returns: Post-request logical context length.
        """
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class SessionInfo:
    """Normalized streaming-session recovery state.

    :ivar exists: Whether the session is present.
    :ivar tip: Current logical context length, or zero when absent.
    :ivar floor: Current rollback floor, or zero when absent.
    :ivar protected: Radix-owned prefix boundary, or zero when absent.
    :ivar inflight: Whether one request currently owns the session.
    :ivar held_tokens: KV tokens held by the session, or zero when absent.
    :ivar last_rid: Last successfully adopted request identifier.
    """

    exists: bool
    tip: int
    floor: int
    protected: int
    inflight: bool
    held_tokens: int
    last_rid: str | None

    @classmethod
    def from_body(cls, body: Any) -> "SessionInfo":
        """Validate and normalize one ``/session_info`` response.

        :param body: Decoded response body.
        :returns: Validated session snapshot.
        """
        assert isinstance(body, dict), f"session_info body is not an object: {body!r}"
        assert set(body) == SESSION_INFO_FIELDS, (
            "session_info fields changed: "
            f"expected={sorted(SESSION_INFO_FIELDS)}, actual={sorted(body)}"
        )
        assert type(body["exists"]) is bool
        assert type(body["inflight"]) is bool

        for field in ("tip", "floor", "protected", "held_tokens"):
            value = body[field]
            assert (
                type(value) is int
            ), f"session_info {field} must be an integer, got {value!r}"
            assert value >= 0

        last_rid = body["last_rid"]
        assert last_rid is None or isinstance(last_rid, str)
        if body["exists"] is False:
            assert body == {
                "exists": False,
                "tip": 0,
                "floor": 0,
                "protected": 0,
                "inflight": False,
                "held_tokens": 0,
                "last_rid": None,
            }, f"missing-session sentinel changed: {body!r}"

        return cls(
            exists=body["exists"],
            tip=body["tip"],
            floor=body["floor"],
            protected=body["protected"],
            inflight=body["inflight"],
            held_tokens=body["held_tokens"],
            last_rid=last_rid,
        )


def _assert_conflict_error(body: Any, correlation_id: str) -> None:
    """Assert the exact public idempotency-conflict envelope.

    :param body: Decoded HTTP or SSE error body.
    :param correlation_id: Top-level request identifier sent by the client.
    """
    assert isinstance(body, dict), f"conflict body is not an object: {body!r}"
    assert set(body) == {"error"}, f"unexpected conflict envelope: {body!r}"
    error = body["error"]
    assert isinstance(error, dict), f"conflict error is not an object: {error!r}"
    assert set(error) == CONFLICT_ERROR_FIELDS, (
        "conflict error fields changed: "
        f"expected={sorted(CONFLICT_ERROR_FIELDS)}, actual={sorted(error)}"
    )
    assert isinstance(error["message"], str) and len(error["message"]) > 0
    assert error["type"] == "streaming_session_conflict"
    assert error["code"] == 409
    assert error["retryable"] is False
    assert error["correlation_id"] == correlation_id


def _read_sse_data(response: requests.Response) -> list[str]:
    """Read and validate all data fields from an SSE response.

    :param response: Streaming HTTP response.
    :returns: Ordered SSE data payloads.
    """
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    assert content_type == "text/event-stream", response.headers

    payloads: list[str] = []
    for raw_line in response.iter_lines():
        if len(raw_line) == 0:
            continue
        line = raw_line.decode("utf-8")
        assert line.startswith("data:"), f"unexpected SSE field: {line!r}"
        payloads.append(line[len("data:") :].strip())
    return payloads


def _read_counter(base_url: str, metric_name: str) -> float | None:
    """Read one Prometheus counter, summing across its label series.

    :param base_url: Live inference server URL.
    :param metric_name: Exact Prometheus metric name.
    :returns: Counter total, or ``None`` when the metric is not exposed.
    """
    response = requests.get(base_url.rstrip("/") + "/metrics", timeout=30)
    if response.status_code != 200:
        return None

    found = False
    total = 0.0
    for line in response.text.splitlines():
        if len(line) == 0 or line.startswith("#"):
            continue
        sample_name = line.split("{", 1)[0].split(" ", 1)[0]
        if sample_name != metric_name:
            continue
        sample_tail = (
            line.rsplit("}", 1)[-1] if "}" in line else line[len(metric_name) :]
        )
        total += float(sample_tail.strip().split()[0])
        found = True
    return total if found else None


def _read_counter_with_label(
    base_url: str,
    metric_name: str,
    label_name: str,
    label_value: str,
) -> float | None:
    """Read one Prometheus counter series selected by an exact label.

    :param base_url: Live inference server URL.
    :param metric_name: Exact Prometheus metric name.
    :param label_name: Label key that selects the series.
    :param label_value: Exact label value.
    :returns: Matching counter total, or ``None`` when no series exists.
    """
    response = requests.get(base_url.rstrip("/") + "/metrics", timeout=30)
    if response.status_code != 200:
        return None

    selector = f'{label_name}="{label_value}"'
    found = False
    total = 0.0
    for line in response.text.splitlines():
        if len(line) == 0 or line.startswith("#"):
            continue
        sample_name = line.split("{", 1)[0].split(" ", 1)[0]
        if sample_name != metric_name or "{" not in line:
            continue
        labels = line.split("{", 1)[1].split("}", 1)[0].split(",")
        if selector not in labels:
            continue
        total += float(line.rsplit("}", 1)[-1].strip().split()[0])
        found = True
    return total if found else None


class SessionClient:
    """Small HTTP client that preserves raw token IDs exactly."""

    _base_url: str

    def __init__(self, base_url: str) -> None:
        """Initialize the client.

        :param base_url: Inference server base URL.
        """
        self._base_url = base_url.rstrip("/")

    def open(
        self,
        *,
        manual_commit: bool = False,
        timeout: float | None = None,
    ) -> str:
        """Open one streaming session.

        :param manual_commit: Whether the session uses an explicit commit floor.
        :param timeout: Optional inactivity timeout in seconds.
        :returns: Server-generated session identifier.
        """
        payload: dict[str, Any] = {
            "capacity_of_str_len": 0,
            "streaming": True,
        }
        if manual_commit:
            payload["manual_commit"] = True
        if timeout is not None:
            payload["timeout"] = timeout
        response = requests.post(
            self._base_url + "/open_session",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json())

    def session_info(self, session_id: str) -> SessionInfo:
        """Read one session without refreshing its inactivity timeout.

        :param session_id: Session to inspect.
        :returns: Exact recovery snapshot.
        """
        response = requests.get(
            self._base_url + "/session_info",
            params={"session_id": session_id},
            timeout=30,
        )
        response.raise_for_status()
        return SessionInfo.from_body(response.json())

    def abort(self, request_rid: str) -> None:
        """Abort one in-flight request.

        :param request_rid: Top-level request identifier to abort.
        """
        response = requests.post(
            self._base_url + "/abort_request",
            json={"rid": request_rid},
            timeout=30,
        )
        response.raise_for_status()

    def close(self, session_id: str) -> None:
        """Close one streaming session.

        :param session_id: Session to close.
        """
        response = requests.post(
            self._base_url + "/close_session",
            json={"session_id": session_id},
            timeout=30,
        )
        response.raise_for_status()

    def generate_payload(
        self,
        session_id: str,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        truncate_to: int | None = None,
        commit_to: int | None = None,
        expected_tip: int | None = None,
        extra_key: str | None = None,
        ignore_eos: bool = True,
        request_rid: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build one deterministic raw-token request body.

        :param session_id: Session to mutate.
        :param input_ids: Raw token delta.
        :param max_new_tokens: Size of the decode burst.
        :param truncate_to: Optional absolute truncation target.
        :param commit_to: Optional commit-floor target.
        :param expected_tip: Optional idempotency precondition.
        :param extra_key: Cache namespace isolating an oracle arm.
        :param ignore_eos: Whether generation continues after an end-of-sequence token.
        :param request_rid: Optional explicit top-level request identifier.
        :param stream: Whether the server should stream the response.
        :returns: JSON request body.
        """
        session_params: dict[str, Any] = {"id": session_id, "rid": None}
        if truncate_to is not None:
            session_params["truncate_to"] = truncate_to
        if commit_to is not None:
            session_params["commit_to"] = commit_to
        if expected_tip is not None:
            session_params["expected_tip"] = expected_tip

        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "extra_key": extra_key,
            "session_params": session_params,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos,
                "no_stop_trim": True,
                "skip_special_tokens": False,
            },
            "stream": stream,
        }
        if request_rid is not None:
            payload["rid"] = request_rid
        return payload

    def post_generate(
        self,
        payload: dict[str, Any],
        *,
        stream_response: bool = False,
    ) -> requests.Response:
        """Submit a prebuilt generation body without interpreting errors.

        :param payload: Exact JSON request body.
        :param stream_response: Whether to expose the response as a byte stream.
        :returns: Raw HTTP response.
        """
        return requests.post(
            self._base_url + "/generate",
            json=payload,
            stream=stream_response,
            timeout=300,
        )

    @staticmethod
    def parse_generate(response: requests.Response) -> GenerateResult:
        """Normalize a successful non-stream generation response.

        :param response: Completed HTTP response.
        :returns: Normalized generation result.
        """
        response.raise_for_status()
        body = response.json()
        meta = body["meta_info"]
        return GenerateResult(
            output_ids=list(body.get("output_ids", [])),
            prompt_tokens=int(meta["prompt_tokens"]),
            completion_tokens=int(meta["completion_tokens"]),
            cached_tokens=int(meta["cached_tokens"]),
            rid=str(meta["id"]),
        )

    def generate(
        self,
        session_id: str,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        truncate_to: int | None = None,
        commit_to: int | None = None,
        expected_tip: int | None = None,
        extra_key: str | None = None,
        ignore_eos: bool = True,
        request_rid: str | None = None,
    ) -> GenerateResult:
        """Run one deterministic raw-token session request.

        :param session_id: Session to mutate.
        :param input_ids: Raw token delta.
        :param max_new_tokens: Size of the decode burst.
        :param truncate_to: Optional absolute truncation target.
        :param commit_to: Optional commit-floor target.
        :param expected_tip: Optional idempotency precondition.
        :param extra_key: Cache namespace isolating an oracle arm.
        :param ignore_eos: Whether generation continues after an end-of-sequence token.
        :param request_rid: Optional explicit top-level request identifier.
        :returns: Normalized generation result.
        """
        payload = self.generate_payload(
            session_id,
            input_ids,
            max_new_tokens=max_new_tokens,
            truncate_to=truncate_to,
            commit_to=commit_to,
            expected_tip=expected_tip,
            extra_key=extra_key,
            ignore_eos=ignore_eos,
            request_rid=request_rid,
        )
        return self.parse_generate(self.post_generate(payload))


def _qualify_truncate_case(
    client: SessionClient,
    base_context: list[int],
    target: int,
    delta: list[int],
) -> None:
    """Compare a hot truncation with a fresh greedy oracle.

    :param client: Live session client.
    :param base_context: Token context before truncation.
    :param target: Absolute truncation target.
    :param delta: Tokens appended after truncation.
    """
    hot_session = client.open(manual_commit=True)
    fresh_session = client.open()
    hot_key = "truncate-hot-" + uuid.uuid4().hex
    fresh_key = "truncate-fresh-" + uuid.uuid4().hex
    try:
        seed = client.generate(
            hot_session,
            base_context,
            max_new_tokens=8,
            extra_key=hot_key,
        )
        exact_context = base_context + seed.output_ids
        hot = client.generate(
            hot_session,
            delta,
            max_new_tokens=16,
            truncate_to=target,
            extra_key=hot_key,
            ignore_eos=False,
        )
        fresh = client.generate(
            fresh_session,
            exact_context[:target] + delta,
            max_new_tokens=16,
            extra_key=fresh_key,
            ignore_eos=False,
        )
        assert (
            fresh.cached_tokens == 0
        ), f"fresh truncate oracle inherited {fresh.cached_tokens} cached tokens"
        assert hot.output_ids == fresh.output_ids, (
            f"greedy mismatch at target={target}: "
            f"hot={hot.output_ids}, fresh={fresh.output_ids}"
        )
        assert hot.prompt_tokens == target + len(delta)
    finally:
        client.close(hot_session)
        client.close(fresh_session)


def run_truncate_qualification(base_url: str) -> None:
    """Run Stage 2 truncation acceptance cases.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    full_context, delta, _ = _build_gemma4_context()
    base_context = full_context[:2_048]

    _qualify_truncate_case(client, base_context, 1_900, delta)
    _qualify_truncate_case(client, base_context, 35, delta)

    session_id = client.open(manual_commit=True)
    try:
        seed = client.generate(session_id, base_context, max_new_tokens=8)
        no_op = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            truncate_to=seed.tip,
        )
        assert no_op.prompt_tokens == seed.tip

        response = requests.post(
            base_url.rstrip("/") + "/generate",
            json={
                "input_ids": delta,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "truncate_to": seed.tip + 1,
                },
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
            timeout=30,
        )
        assert response.status_code == 400, response.text

        recovery = client.generate(session_id, delta, max_new_tokens=4)
        assert recovery.cached_tokens in {seed.tip - 1, seed.tip}

        empty = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            truncate_to=0,
        )
        assert empty.prompt_tokens == 0
        assert empty.completion_tokens == 0

        rebuilt = client.generate(session_id, delta, max_new_tokens=4)
        assert rebuilt.prompt_tokens == len(delta)
        assert rebuilt.cached_tokens == 0
    finally:
        client.close(session_id)


def _qualify_commit_case(
    client: SessionClient,
    base_context: list[int],
    floor: int,
    target: int,
    delta: list[int],
    tokenizer: PreTrainedTokenizerBase,
    expected_sentinel: str,
) -> None:
    """Compare a floor-pinned rewind with an exact fresh-session oracle.

    :param client: Live session client.
    :param base_context: Context extending more than one SWA window past floor.
    :param floor: Explicit rollback floor.
    :param target: Truncation target at or above floor.
    :param delta: Tokens appended after truncation.
    :param tokenizer: Tokenizer used to inspect the sampled answer.
    :param expected_sentinel: Context-dependent answer expected at the target.
    """
    hot_session = client.open(manual_commit=True)
    fresh_session = client.open()
    hot_key = "commit-hot-" + uuid.uuid4().hex
    fresh_key = "commit-fresh-" + uuid.uuid4().hex
    try:
        client.generate(
            hot_session,
            base_context[:floor],
            max_new_tokens=0,
            commit_to=floor,
            extra_key=hot_key,
        )
        extended = client.generate(
            hot_session,
            base_context[floor:],
            max_new_tokens=0,
            extra_key=hot_key,
        )
        assert extended.tip == len(base_context)

        hot = client.generate(
            hot_session,
            delta,
            max_new_tokens=16,
            truncate_to=target,
            extra_key=hot_key,
            ignore_eos=False,
        )
        fresh = client.generate(
            fresh_session,
            base_context[:target] + delta,
            max_new_tokens=16,
            extra_key=fresh_key,
            ignore_eos=False,
        )
        assert hot.cached_tokens == target
        assert (
            fresh.cached_tokens == 0
        ), f"fresh commit oracle inherited {fresh.cached_tokens} cached tokens"
        assert hot.output_ids == fresh.output_ids, (
            f"floor-pinned greedy mismatch at floor={floor}, target={target}: "
            f"hot={hot.output_ids}, fresh={fresh.output_ids}"
        )
        decoded = tokenizer.decode(hot.output_ids, skip_special_tokens=False)
        assert expected_sentinel in decoded, (
            f"wrong sentinel at target={target}: expected={expected_sentinel}, "
            f"decoded={decoded!r}"
        )
    finally:
        client.close(hot_session)
        client.close(fresh_session)


def run_commit_qualification(base_url: str) -> None:
    """Run Stage 3 commit-floor and SWA-pin acceptance cases.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    base_context, delta, tokenizer = _build_gemma4_context()
    floor = 2_048
    assert len(base_context) - floor > 1_024

    expected_sentinels = {
        floor: "ORCHID",
        floor + 257: "MAPLE",
        len(base_context) - 1: "CEDAR",
    }
    for target, expected_sentinel in expected_sentinels.items():
        _qualify_commit_case(
            client,
            base_context,
            floor,
            target,
            delta,
            tokenizer,
            expected_sentinel,
        )

    session_id = client.open(manual_commit=True)
    try:
        seeded = client.generate(
            session_id,
            base_context,
            max_new_tokens=0,
            commit_to=floor,
        )
        committed = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            commit_to=seeded.tip,
        )
        assert committed.tip == seeded.tip

        response = requests.post(
            base_url.rstrip("/") + "/generate",
            json={
                "input_ids": [],
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "commit_to": floor - 1,
                },
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            },
            timeout=30,
        )
        assert response.status_code == 400, response.text

        recovery = client.generate(session_id, delta, max_new_tokens=4)
        assert recovery.cached_tokens in {seeded.tip - 1, seeded.tip}
    finally:
        client.close(session_id)


def _assert_conflict_preserves_state(
    client: SessionClient,
    session_id: str,
    before: SessionInfo,
    payload: dict[str, Any],
) -> None:
    """Submit one stale request and prove its rejection is non-destructive.

    :param client: Live session client.
    :param session_id: Session guarded by ``expected_tip``.
    :param before: Recovery snapshot before the request.
    :param payload: Exact generation body to submit.
    """
    request_rid = payload.get("rid")
    assert isinstance(request_rid, str)
    is_streaming = payload.get("stream") is True
    with client.post_generate(
        payload,
        stream_response=is_streaming,
    ) as response:
        if is_streaming:
            assert response.status_code == 200, response.text
            sse_data = _read_sse_data(response)
            assert (
                len(sse_data) == 2
            ), f"conflict stream must contain one event and [DONE], got {sse_data!r}"
            assert sse_data[1] == "[DONE]", sse_data
            _assert_conflict_error(json.loads(sse_data[0]), request_rid)
        else:
            assert response.status_code == 409, response.text
            _assert_conflict_error(response.json(), request_rid)

    after = client.session_info(session_id)
    assert (
        after == before
    ), f"stale request mutated session state: before={before}, after={after}"


def _wait_for_inflight(
    client: SessionClient,
    session_id: str,
    expected: bool,
    timeout: float,
) -> SessionInfo:
    """Wait until the session's in-flight state reaches one value.

    :param client: Live session client.
    :param session_id: Session to inspect.
    :param expected: Desired in-flight value.
    :param timeout: Maximum wait in seconds.
    :returns: First matching recovery snapshot.
    """
    deadline = time.monotonic() + timeout
    last_info: SessionInfo | None = None
    while time.monotonic() < deadline:
        last_info = client.session_info(session_id)
        assert last_info.exists
        if last_info.inflight is expected:
            return last_info
        time.sleep(0.02)
    raise AssertionError(
        f"session inflight did not become {expected}; last_info={last_info}"
    )


def _is_terminal_abort(body: dict[str, Any]) -> bool:
    """Return whether one streamed response carries an abort finish reason.

    :param body: Decoded streaming response chunk.
    :returns: Whether the chunk is the terminal abort response.
    """
    meta_info = body.get("meta_info")
    if not isinstance(meta_info, dict):
        return False
    finish_reason = meta_info.get("finish_reason")
    return isinstance(finish_reason, dict) and finish_reason.get("type") == "abort"


def _abort_request_and_assert_recovery(
    client: SessionClient,
    session_id: str,
    payload: dict[str, Any],
    *,
    expected_tip: int,
    expected_floor: int,
    expected_last_rid: str,
) -> SessionInfo:
    """Abort one long burst and prove its durable state and cache heal.

    :param client: Live session client.
    :param session_id: Session mutated by the request.
    :param payload: Long-running generation request.
    :param expected_tip: Durable tip after prepared mutations apply.
    :param expected_floor: Durable floor after prepared mutations apply.
    :param expected_last_rid: Durable request identity after mutations apply.
    :returns: Healed post-abort session state.
    """
    request_rid = payload.get("rid")
    assert isinstance(request_rid, str)
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_response: requests.Response | None = None
    streamed_bodies: list[dict[str, Any]] = []
    abort_sent = False
    try:
        stream_response = client.post_generate(
            stream_payload,
            stream_response=True,
        )
        assert stream_response.status_code == 200, stream_response.text
        stream_lines = stream_response.iter_lines()
        for raw_line in stream_lines:
            if len(raw_line) == 0:
                continue
            line = raw_line.decode("utf-8")
            assert line.startswith("data:"), line
            data = line[len("data:") :].strip()
            assert data != "[DONE]", "long abort request ended before decode"
            body = json.loads(data)
            assert isinstance(body, dict), body
            streamed_bodies.append(body)
            output_ids = body.get("output_ids")
            if isinstance(output_ids, list) and len(output_ids) > 0:
                break

        deadline = time.monotonic() + 30
        inflight = client.session_info(session_id)
        while time.monotonic() < deadline:
            inflight = client.session_info(session_id)
            assert inflight.exists
            if (
                inflight.inflight
                and inflight.tip == expected_tip
                and inflight.floor == expected_floor
                and inflight.last_rid == expected_last_rid
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                "session did not expose its prepared durable state: "
                f"tip={expected_tip}, floor={expected_floor}, "
                f"last_rid={expected_last_rid}, actual={inflight}"
            )

        with ThreadPoolExecutor(max_workers=8) as info_executor:
            concurrent_info = list(
                info_executor.map(client.session_info, [session_id] * 16)
            )
        assert concurrent_info == [inflight] * 16, (
            "session_info returned a torn in-flight snapshot: "
            f"expected={inflight}, actual={concurrent_info}"
        )

        assert client.session_info(session_id).inflight
        client.abort(request_rid)
        abort_sent = True
        for raw_line in stream_lines:
            if len(raw_line) == 0:
                continue
            line = raw_line.decode("utf-8")
            assert line.startswith("data:"), line
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            body = json.loads(data)
            assert isinstance(body, dict), body
            streamed_bodies.append(body)
        else:
            raise AssertionError("aborted stream omitted [DONE]")
    finally:
        try:
            if stream_response is not None and abort_sent is False:
                client.abort(request_rid)
        finally:
            if stream_response is not None:
                stream_response.close()

    assert any(
        _is_terminal_abort(body) for body in streamed_bodies
    ), f"aborted stream omitted its terminal reason: {streamed_bodies!r}"
    _wait_for_inflight(client, session_id, False, timeout=30)
    healed = client.session_info(session_id)
    assert healed.exists
    assert healed.inflight is False
    assert healed.tip == expected_tip
    assert healed.floor == expected_floor
    assert healed.last_rid == expected_last_rid
    return healed


def _qualify_timeout_reads(client: SessionClient) -> None:
    """Prove recovery reads do not extend a session's inactivity lease.

    :param client: Live session client.
    """
    session_id = client.open(timeout=SESSION_TIMEOUT_SECONDS)
    started = time.monotonic()
    reads_while_live = 0
    last_info = client.session_info(session_id)
    try:
        deadline = (
            started + SESSION_TIMEOUT_SECONDS + SESSION_REAP_INTERVAL_SECONDS + 1.0
        )
        while time.monotonic() < deadline:
            last_info = client.session_info(session_id)
            if last_info.exists is False:
                break
            reads_while_live += 1
            time.sleep(SESSION_INFO_POLL_SECONDS)

        assert reads_while_live >= 2
        elapsed = time.monotonic() - started
        assert last_info.exists is False, (
            "session_info reads refreshed the inactivity timeout: "
            f"elapsed={elapsed:.3f}s"
        )
        assert (
            elapsed >= SESSION_TIMEOUT_SECONDS - SESSION_INFO_POLL_SECONDS
        ), f"session reaped before its inactivity lease: elapsed={elapsed:.3f}s"
        assert elapsed <= (
            SESSION_TIMEOUT_SECONDS + SESSION_REAP_INTERVAL_SECONDS + 1.5
        ), f"session reaping exceeded its bounded lease: elapsed={elapsed:.3f}s"
        assert last_info == SessionInfo(
            exists=False,
            tip=0,
            floor=0,
            protected=0,
            inflight=False,
            held_tokens=0,
            last_rid=None,
        )
    finally:
        if last_info.exists:
            client.close(session_id)


def _qualify_prepared_mutation_abort_recovery(
    client: SessionClient,
    context: list[int],
) -> None:
    """Exercise truncate and commit durability through live abort completion.

    :param client: Live session client.
    :param context: Natural Gemma-4 token context.
    """
    truncate_session_id = client.open(manual_commit=True)
    truncate_key = "stage4-truncate-abort-" + uuid.uuid4().hex
    try:
        seeded = client.generate(
            truncate_session_id,
            context[:256],
            max_new_tokens=4,
            extra_key=truncate_key,
        )
        seeded_info = client.session_info(truncate_session_id)
        assert seeded_info.tip == seeded.tip
        assert seeded_info.floor == 0

        truncate_target = 128
        truncate_rid = "stage4-truncate-abort-" + uuid.uuid4().hex
        truncate_payload = client.generate_payload(
            truncate_session_id,
            [],
            max_new_tokens=100_000,
            truncate_to=truncate_target,
            expected_tip=seeded.tip,
            extra_key=truncate_key,
            ignore_eos=True,
            request_rid=truncate_rid,
        )
        truncated = _abort_request_and_assert_recovery(
            client,
            truncate_session_id,
            truncate_payload,
            expected_tip=truncate_target,
            expected_floor=0,
            expected_last_rid=truncate_rid,
        )
        assert truncated.tip == truncate_target
        assert truncated.last_rid == truncate_rid

        truncate_resume = client.generate(
            truncate_session_id,
            context[256:260],
            max_new_tokens=1,
            expected_tip=truncate_target,
            extra_key=truncate_key,
        )
        assert truncate_resume.cached_tokens in {
            truncate_target - 1,
            truncate_target,
        }
    finally:
        client.close(truncate_session_id)

    commit_session_id = client.open(manual_commit=True)
    commit_key = "stage4-commit-abort-" + uuid.uuid4().hex
    try:
        seeded = client.generate(
            commit_session_id,
            context[:128],
            max_new_tokens=4,
            extra_key=commit_key,
        )
        seeded_info = client.session_info(commit_session_id)
        assert seeded_info.tip == seeded.tip
        assert seeded_info.floor == 0

        commit_delta = context[128:160]
        commit_target = seeded.tip + len(commit_delta)
        commit_rid = "stage4-commit-abort-" + uuid.uuid4().hex
        commit_payload = client.generate_payload(
            commit_session_id,
            commit_delta,
            max_new_tokens=100_000,
            commit_to=commit_target,
            expected_tip=seeded.tip,
            extra_key=commit_key,
            ignore_eos=True,
            request_rid=commit_rid,
        )
        committed = _abort_request_and_assert_recovery(
            client,
            commit_session_id,
            commit_payload,
            expected_tip=commit_target,
            expected_floor=commit_target,
            expected_last_rid=commit_rid,
        )
        assert committed.tip == commit_target
        assert committed.floor == commit_target
        assert committed.last_rid == commit_rid

        commit_resume = client.generate(
            commit_session_id,
            context[160:164],
            max_new_tokens=1,
            expected_tip=commit_target,
            extra_key=commit_key,
        )
        assert commit_resume.cached_tokens in {commit_target - 1, commit_target}
    finally:
        client.close(commit_session_id)


def run_recovery_qualification(base_url: str) -> None:
    """Run Stage 4 idempotency and recovery acceptance cases.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, _, _ = _build_gemma4_context()
    session_id = client.open()
    try:
        hot_extra_key = "stage4-recovery-hot-" + uuid.uuid4().hex
        conflict_count = 0
        conflict_metric_before = _read_counter(base_url, CONFLICT_METRIC_NAME)
        assert (
            conflict_metric_before is not None
        ), f"required metric is not exposed: {CONFLICT_METRIC_NAME}"

        empty = client.session_info(session_id)
        assert empty == SessionInfo(
            exists=True,
            tip=0,
            floor=0,
            protected=0,
            inflight=False,
            held_tokens=0,
            last_rid=None,
        )

        seed_rid = "stage4-seed-" + uuid.uuid4().hex
        seed_payload = client.generate_payload(
            session_id,
            context[:96],
            max_new_tokens=4,
            expected_tip=0,
            extra_key=hot_extra_key,
            request_rid=seed_rid,
        )
        seed = client.parse_generate(client.post_generate(seed_payload))
        assert seed.rid == seed_rid

        stable = client.session_info(session_id)
        assert stable.exists
        assert stable.tip == seed.tip
        assert stable.floor == seed.tip
        assert stable.protected <= stable.floor
        assert stable.held_tokens > 0
        assert stable.inflight is False
        assert stable.last_rid == seed_rid

        _assert_conflict_preserves_state(
            client,
            session_id,
            stable,
            seed_payload,
        )
        conflict_count += 1

        high_rid = "stage4-high-" + uuid.uuid4().hex
        high_payload = client.generate_payload(
            session_id,
            context[96:112],
            max_new_tokens=4,
            expected_tip=stable.tip + 1,
            extra_key=hot_extra_key,
            request_rid=high_rid,
        )
        _assert_conflict_preserves_state(
            client,
            session_id,
            stable,
            high_payload,
        )
        conflict_count += 1

        low_stream_rid = "stage4-low-stream-" + uuid.uuid4().hex
        low_stream_payload = client.generate_payload(
            session_id,
            context[96:112],
            max_new_tokens=4,
            expected_tip=stable.tip - 1,
            extra_key=hot_extra_key,
            request_rid=low_stream_rid,
            stream=True,
        )
        _assert_conflict_preserves_state(
            client,
            session_id,
            stable,
            low_stream_payload,
        )
        conflict_count += 1

        high_stream_rid = "stage4-high-stream-" + uuid.uuid4().hex
        high_stream_payload = client.generate_payload(
            session_id,
            context[96:112],
            max_new_tokens=4,
            expected_tip=stable.tip + 1,
            extra_key=hot_extra_key,
            request_rid=high_stream_rid,
            stream=True,
        )
        _assert_conflict_preserves_state(
            client,
            session_id,
            stable,
            high_stream_payload,
        )
        conflict_count += 1

        conflict_metric_after = _read_counter(base_url, CONFLICT_METRIC_NAME)
        assert (
            conflict_metric_after is not None
        ), f"required metric disappeared: {CONFLICT_METRIC_NAME}"
        assert conflict_metric_after - conflict_metric_before == conflict_count

        accepted_delta = context[112:128]
        accepted_rid = "stage4-accepted-" + uuid.uuid4().hex
        accepted = client.generate(
            session_id,
            accepted_delta,
            max_new_tokens=4,
            expected_tip=stable.tip,
            extra_key=hot_extra_key,
            request_rid=accepted_rid,
        )
        assert accepted.rid == accepted_rid

        cold_session_id = client.open()
        cold_extra_key = "stage4-recovery-cold-" + uuid.uuid4().hex
        try:
            cold = client.generate(
                cold_session_id,
                context[:96] + seed.output_ids + accepted_delta,
                max_new_tokens=4,
                extra_key=cold_extra_key,
                request_rid="stage4-cold-" + uuid.uuid4().hex,
            )
            assert cold.cached_tokens == 0, (
                "cold recovery oracle inherited radix state: "
                f"cached_tokens={cold.cached_tokens}"
            )
            assert cold.prompt_tokens == accepted.prompt_tokens
            assert cold.output_ids == accepted.output_ids, (
                "conflict recovery changed greedy content: "
                f"hot={accepted.output_ids}, cold={cold.output_ids}"
            )
        finally:
            client.close(cold_session_id)

        accepted_info = client.session_info(session_id)
        assert accepted_info.tip == accepted.tip
        assert accepted_info.last_rid == accepted_rid

        with ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_info = list(executor.map(client.session_info, [session_id] * 32))
        assert concurrent_info == [accepted_info] * 32

        abort_rid = "stage4-abort-" + uuid.uuid4().hex
        abort_payload = client.generate_payload(
            session_id,
            context[128:144],
            max_new_tokens=100_000,
            expected_tip=accepted_info.tip,
            extra_key=hot_extra_key,
            ignore_eos=True,
            request_rid=abort_rid,
        )
        assert accepted_info.last_rid is not None
        healed = _abort_request_and_assert_recovery(
            client,
            session_id,
            abort_payload,
            expected_tip=accepted_info.tip,
            expected_floor=accepted_info.floor,
            expected_last_rid=accepted_info.last_rid,
        )
        assert (
            healed == accepted_info
        ), f"aborted request did not heal to its pre-burst state: {healed}"

        recovery_rid = "stage4-recovery-" + uuid.uuid4().hex
        recovery = client.generate(
            session_id,
            context[144:160],
            max_new_tokens=4,
            expected_tip=accepted_info.tip,
            extra_key=hot_extra_key,
            request_rid=recovery_rid,
        )
        assert recovery.rid == recovery_rid
        recovered_info = client.session_info(session_id)
        assert recovered_info.tip == recovery.tip
        assert recovered_info.last_rid == recovery_rid
    finally:
        client.close(session_id)

    closed = client.session_info(session_id)
    assert closed == SessionInfo(
        exists=False,
        tip=0,
        floor=0,
        protected=0,
        inflight=False,
        held_tokens=0,
        last_rid=None,
    )
    _qualify_prepared_mutation_abort_recovery(client, context)
    _qualify_timeout_reads(client)


SESSION_METRIC_NAMES = (
    "sglang:num_streaming_sessions",
    "sglang:streaming_session_held_tokens",
    "sglang:streaming_session_held_swa_tokens",
    "sglang:kv_used_tokens",
    "sglang:swa_used_tokens",
)
TRUNCATION_METRIC_NAME = "sglang:streaming_session_truncations_total"
COMMIT_METRIC_NAME = "sglang:streaming_session_commits_total"
ABORT_PRESERVED_METRIC_NAME = (
    "sglang:streaming_session_aborts_with_slot_preserved_total"
)
REAP_METRIC_NAME = "sglang:streaming_session_reaps_total"
FIRST_REAL_SMALL_EXTEND_LIMIT_SECONDS = 0.5
METRIC_SETTLE_TIMEOUT_SECONDS = 45.0
METRIC_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class SessionMetricSnapshot:
    """Streaming-session and active-KV gauge snapshot.

    :ivar sessions: Number of open streaming sessions.
    :ivar held_tokens: Full-attention tokens held by detached session slots.
    :ivar held_swa_tokens: Sliding-window tokens held by detached session slots.
    :ivar kv_used_tokens: Active full-attention KV allocation.
    :ivar swa_used_tokens: Active sliding-window KV allocation.
    """

    sessions: float
    held_tokens: float
    held_swa_tokens: float
    kv_used_tokens: float
    swa_used_tokens: float

    @classmethod
    def read(cls, base_url: str) -> "SessionMetricSnapshot":
        """Read all required gauge values.

        :param base_url: Live inference server URL.
        :returns: Current aggregate gauge snapshot.
        """
        values: list[float] = []
        for metric_name in SESSION_METRIC_NAMES:
            value = _read_counter(base_url, metric_name)
            if value is None:
                raise AssertionError(f"required metric is absent: {metric_name}")
            values.append(value)
        return cls(*values)

    @classmethod
    def zero(cls) -> "SessionMetricSnapshot":
        """Return the fully quiescent expected snapshot.

        :returns: All-zero session and active-KV gauges.
        """
        return cls(
            sessions=0.0,
            held_tokens=0.0,
            held_swa_tokens=0.0,
            kv_used_tokens=0.0,
            swa_used_tokens=0.0,
        )


def _read_required_metric(base_url: str, metric_name: str) -> float:
    """Read one required aggregate metric.

    :param base_url: Live inference server URL.
    :param metric_name: Prometheus metric to read.
    :returns: Current aggregate value.
    """
    value = _read_counter(base_url, metric_name)
    if value is None:
        raise AssertionError(f"required metric is absent: {metric_name}")
    return value


def _wait_for_metric(
    base_url: str,
    metric_name: str,
    expected: float,
    timeout: float = METRIC_SETTLE_TIMEOUT_SECONDS,
) -> float:
    """Wait for one metric to reach an exact value.

    :param base_url: Live inference server URL.
    :param metric_name: Prometheus metric to read.
    :param expected: Exact expected value.
    :param timeout: Maximum wait in seconds.
    :returns: Matching metric value.
    """
    deadline = time.monotonic() + timeout
    actual = _read_required_metric(base_url, metric_name)
    while time.monotonic() < deadline:
        actual = _read_required_metric(base_url, metric_name)
        if actual == expected:
            return actual
        time.sleep(METRIC_POLL_SECONDS)
    raise AssertionError(
        f"metric {metric_name} did not settle: expected={expected}, actual={actual}"
    )


def _wait_for_labeled_metric(
    base_url: str,
    metric_name: str,
    label_name: str,
    label_value: str,
    expected: float,
    timeout: float = METRIC_SETTLE_TIMEOUT_SECONDS,
) -> float:
    """Wait for one exact labeled counter series.

    :param base_url: Live inference server URL.
    :param metric_name: Prometheus metric to read.
    :param label_name: Label key that selects the series.
    :param label_value: Exact label value.
    :param expected: Exact expected value.
    :param timeout: Maximum wait in seconds.
    :returns: Matching counter value.
    """
    deadline = time.monotonic() + timeout
    actual = _read_counter_with_label(base_url, metric_name, label_name, label_value)
    while time.monotonic() < deadline:
        actual = _read_counter_with_label(
            base_url, metric_name, label_name, label_value
        )
        if actual == expected:
            return actual
        time.sleep(METRIC_POLL_SECONDS)
    raise AssertionError(
        f"metric {metric_name}{{{label_name}={label_value!r}}} did not settle: "
        f"expected={expected}, actual={actual}"
    )


def _wait_for_metric_snapshot(
    base_url: str,
    expected: SessionMetricSnapshot,
    timeout: float = METRIC_SETTLE_TIMEOUT_SECONDS,
) -> SessionMetricSnapshot:
    """Wait for all session and active-KV gauges to match.

    :param base_url: Live inference server URL.
    :param expected: Exact expected snapshot.
    :param timeout: Maximum wait in seconds.
    :returns: Matching snapshot.
    """
    deadline = time.monotonic() + timeout
    actual = SessionMetricSnapshot.read(base_url)
    while time.monotonic() < deadline:
        actual = SessionMetricSnapshot.read(base_url)
        if actual == expected:
            return actual
        time.sleep(METRIC_POLL_SECONDS)
    raise AssertionError(
        f"session metrics did not settle: expected={expected}, actual={actual}"
    )


def _flush_to_quiescent_metrics(base_url: str) -> SessionMetricSnapshot:
    """Flush evictable cache state and prove all session allocations are gone.

    :param base_url: Live inference server URL.
    :returns: Quiescent all-zero metric snapshot.
    """
    response = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
    assert response.status_code == 200, response.text
    return _wait_for_metric_snapshot(base_url, SessionMetricSnapshot.zero())


def _deterministic_warmup_tokens(length: int, offset: int = 0) -> list[int]:
    """Build the same stable raw-token cycle as the startup warmup.

    :param length: Number of tokens to build.
    :param offset: Offset into the token cycle.
    :returns: Deterministic in-vocabulary token IDs.
    """
    return [2_048 + ((offset + index) % 4_096) for index in range(length)]


def _run_first_real_small_extend(base_url: str) -> float:
    """Measure the first deep-session small extend after startup warmup.

    :param base_url: Live inference server URL.
    :returns: Small-extend wall latency in seconds.
    """
    client = SessionClient(base_url)
    session_id = client.open()
    extra_key = "first-real-small-extend-" + uuid.uuid4().hex
    try:
        seeded = client.generate(
            session_id,
            _deterministic_warmup_tokens(40_960),
            max_new_tokens=0,
            extra_key=extra_key,
        )
        assert seeded.tip == 40_960

        started = time.perf_counter()
        extended = client.generate(
            session_id,
            _deterministic_warmup_tokens(64, offset=40_960),
            max_new_tokens=8,
            ignore_eos=True,
            extra_key=extra_key,
        )
        elapsed = time.perf_counter() - started
        assert extended.prompt_tokens == 41_024
        assert extended.cached_tokens == 40_960
        assert elapsed <= FIRST_REAL_SMALL_EXTEND_LIMIT_SECONDS, (
            "first real small extend missed its warmed latency bound: "
            f"elapsed={elapsed:.6f}s, "
            f"limit={FIRST_REAL_SMALL_EXTEND_LIMIT_SECONDS:.6f}s"
        )
        return elapsed
    finally:
        client.close(session_id)


def _run_deep_abort_preservation(base_url: str) -> None:
    """Prove a deep mid-burst abort preserves the exact pre-burst slot.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, delta, _ = _build_gemma4_context()
    session_id = client.open()
    extra_key = "deep-abort-" + uuid.uuid4().hex
    abort_before = _read_required_metric(base_url, ABORT_PRESERVED_METRIC_NAME)
    try:
        seeded = client.generate(
            session_id,
            context,
            max_new_tokens=0,
            extra_key=extra_key,
        )
        before = client.session_info(session_id)
        assert before.tip == seeded.tip == len(context)
        assert before.floor == before.tip
        assert before.last_rid is not None

        abort_rid = "deep-abort-" + uuid.uuid4().hex
        payload = client.generate_payload(
            session_id,
            [],
            max_new_tokens=100_000,
            expected_tip=before.tip,
            extra_key=extra_key,
            ignore_eos=True,
            request_rid=abort_rid,
            stream=True,
        )
        healed = _abort_request_and_assert_recovery(
            client,
            session_id,
            payload,
            expected_tip=before.tip,
            expected_floor=before.floor,
            expected_last_rid=before.last_rid,
        )
        assert healed == before

        resumed = client.generate(
            session_id,
            delta,
            max_new_tokens=16,
            expected_tip=before.tip,
            extra_key=extra_key,
            ignore_eos=False,
        )
        assert resumed.cached_tokens in {before.tip - 1, before.tip}
        assert resumed.prompt_tokens == before.tip + len(delta)

        fresh_session_id = client.open()
        fresh_key = "deep-abort-fresh-" + uuid.uuid4().hex
        try:
            fresh_seeded = client.generate(
                fresh_session_id,
                context,
                max_new_tokens=0,
                extra_key=fresh_key,
            )
            assert fresh_seeded.tip == before.tip

            fresh = client.generate(
                fresh_session_id,
                delta,
                max_new_tokens=16,
                extra_key=fresh_key,
                ignore_eos=False,
            )
            assert fresh.cached_tokens in {before.tip - 1, before.tip}
            assert resumed.output_ids == fresh.output_ids, (
                "aborted sampled tokens changed the schedule-matched continuation: "
                f"hot={resumed.output_ids}, fresh={fresh.output_ids}"
            )
        finally:
            client.close(fresh_session_id)
    finally:
        client.close(session_id)

    _wait_for_metric(
        base_url,
        ABORT_PRESERVED_METRIC_NAME,
        abort_before + 1,
    )


def _run_full_log_recovery(base_url: str) -> None:
    """Close a session and rebuild it solely from the application token log.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, delta, _ = _build_gemma4_context()
    first_session = client.open()
    first_key = "full-log-source-" + uuid.uuid4().hex
    try:
        first = client.generate(
            first_session,
            context,
            max_new_tokens=8,
            extra_key=first_key,
            ignore_eos=False,
        )
        full_log = context + first.output_ids
        assert len(first.output_ids) > 0
        assert first.tip == len(full_log)
    finally:
        client.close(first_session)

    assert client.session_info(first_session).exists is False

    recovered_session = client.open()
    recovered_key = "full-log-recovered-" + uuid.uuid4().hex
    try:
        recovered = client.generate(
            recovered_session,
            full_log,
            max_new_tokens=0,
            expected_tip=0,
            extra_key=recovered_key,
        )
        assert recovered.cached_tokens == 0
        assert recovered.tip == len(full_log)
        recovered_info = client.session_info(recovered_session)
        assert recovered_info.tip == len(full_log)
        assert recovered_info.floor == len(full_log)

        resumed = client.generate(
            recovered_session,
            delta,
            max_new_tokens=8,
            expected_tip=recovered_info.tip,
            extra_key=recovered_key,
            ignore_eos=False,
        )
        assert resumed.cached_tokens == recovered_info.tip
        assert resumed.prompt_tokens == recovered_info.tip + len(delta)
        assert len(resumed.output_ids) > 0
    finally:
        client.close(recovered_session)


def _run_pin_accounting(base_url: str) -> None:
    """Prove lagging-floor SWA retention and commit-time collapse.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, _, _ = _build_gemma4_context()
    floor = 2_048
    committed_swa_tokens = 1_024
    tip = len(context)
    lag = tip - floor
    session_id = client.open(manual_commit=True)
    extra_key = "pin-manual-" + uuid.uuid4().hex
    try:
        client.generate(
            session_id,
            context[:floor],
            max_new_tokens=0,
            commit_to=floor,
            extra_key=extra_key,
        )
        sessions = _wait_for_metric(
            base_url,
            "sglang:num_streaming_sessions",
            1.0,
        )
        assert sessions == 1.0
        before_extend = _read_required_metric(
            base_url,
            "sglang:streaming_session_held_swa_tokens",
        )

        extended = client.generate(
            session_id,
            context[floor:],
            max_new_tokens=0,
            extra_key=extra_key,
        )
        assert extended.tip == tip
        after_extend = _wait_for_metric(
            base_url,
            "sglang:streaming_session_held_swa_tokens",
            before_extend + lag,
        )
        assert after_extend - before_extend == lag

        client.generate(
            session_id,
            [],
            max_new_tokens=0,
            commit_to=tip,
            extra_key=extra_key,
        )
        after_commit = _wait_for_metric(
            base_url,
            "sglang:streaming_session_held_swa_tokens",
            committed_swa_tokens,
        )
        assert after_extend > after_commit
    finally:
        client.close(session_id)

    _flush_to_quiescent_metrics(base_url)

    default_session = client.open()
    default_key = "pin-default-" + uuid.uuid4().hex
    try:
        default = client.generate(
            default_session,
            context,
            max_new_tokens=0,
            extra_key=default_key,
        )
        assert default.tip == tip
        default_swa = _wait_for_metric(
            base_url,
            "sglang:streaming_session_held_swa_tokens",
            committed_swa_tokens,
        )
        assert default_swa == after_commit
    finally:
        client.close(default_session)


def _run_edge_cases(base_url: str) -> None:
    """Exercise pure mutations, protected-prefix truncation, and rejections.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, delta, _ = _build_gemma4_context()
    cache_key = "edge-protected-" + uuid.uuid4().hex

    prime = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "input_ids": context[:1_024],
            "extra_key": cache_key,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        },
        timeout=300,
    )
    assert prime.status_code == 200, prime.text

    session_id = client.open(manual_commit=True)
    try:
        seeded = client.generate(
            session_id,
            context[:2_048],
            max_new_tokens=0,
            extra_key=cache_key,
        )
        seeded_info = client.session_info(session_id)
        assert seeded_info.protected > 0
        assert seeded_info.floor == 0

        no_op = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            truncate_to=seeded.tip,
            extra_key=cache_key,
        )
        assert no_op.tip == seeded.tip

        below_protected = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            truncate_to=35,
            extra_key=cache_key,
        )
        assert below_protected.tip == 35
        truncated_info = client.session_info(session_id)
        assert truncated_info.tip == 35
        assert truncated_info.floor == 0

        burst = client.generate(
            session_id,
            [],
            max_new_tokens=1,
            expected_tip=35,
            extra_key=cache_key,
            ignore_eos=False,
        )
        assert burst.prompt_tokens == 35

        before_high_rejection = client.session_info(session_id)
        high_payload = client.generate_payload(
            session_id,
            delta,
            max_new_tokens=0,
            truncate_to=before_high_rejection.tip + 1,
            extra_key=cache_key,
        )
        high_response = client.post_generate(high_payload)
        assert high_response.status_code == 400, high_response.text
        assert client.session_info(session_id) == before_high_rejection

        committed = client.generate(
            session_id,
            [],
            max_new_tokens=0,
            commit_to=before_high_rejection.tip,
            expected_tip=before_high_rejection.tip,
            extra_key=cache_key,
        )
        committed_info = client.session_info(session_id)
        assert committed_info.tip == committed.tip
        assert committed_info.floor == committed.tip

        low_commit_payload = client.generate_payload(
            session_id,
            [],
            max_new_tokens=0,
            commit_to=committed_info.floor - 1,
            extra_key=cache_key,
        )
        low_commit_response = client.post_generate(low_commit_payload)
        assert low_commit_response.status_code == 400, low_commit_response.text
        assert client.session_info(session_id) == committed_info

        low_truncate_payload = client.generate_payload(
            session_id,
            [],
            max_new_tokens=0,
            truncate_to=committed_info.floor - 1,
            extra_key=cache_key,
        )
        low_truncate_response = client.post_generate(low_truncate_payload)
        assert low_truncate_response.status_code == 400, low_truncate_response.text
        assert client.session_info(session_id) == committed_info
    finally:
        client.close(session_id)


def _run_leak_closure(base_url: str) -> None:
    """Exercise truncate and abort lifecycles before an exact leak check.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, _, _ = _build_gemma4_context()

    truncate_session = client.open(manual_commit=True)
    truncate_key = "leak-truncate-" + uuid.uuid4().hex
    try:
        client.generate(
            truncate_session,
            context,
            max_new_tokens=0,
            extra_key=truncate_key,
        )
        client.generate(
            truncate_session,
            [],
            max_new_tokens=0,
            truncate_to=2_048,
            extra_key=truncate_key,
        )
    finally:
        client.close(truncate_session)

    abort_session = client.open()
    abort_key = "leak-abort-" + uuid.uuid4().hex
    try:
        seeded = client.generate(
            abort_session,
            context,
            max_new_tokens=0,
            extra_key=abort_key,
        )
        before = client.session_info(abort_session)
        assert before.last_rid is not None
        payload = client.generate_payload(
            abort_session,
            [],
            max_new_tokens=100_000,
            expected_tip=seeded.tip,
            extra_key=abort_key,
            ignore_eos=True,
            request_rid="leak-abort-" + uuid.uuid4().hex,
            stream=True,
        )
        _abort_request_and_assert_recovery(
            client,
            abort_session,
            payload,
            expected_tip=before.tip,
            expected_floor=before.floor,
            expected_last_rid=before.last_rid,
        )
    finally:
        client.close(abort_session)


class Gemma4StreamingSessionOracleKitMixin:
    """Load-bearing truncate oracle shared by every production matrix arm."""

    base_url: str

    def test_10_truncate_greedy_equivalence(self) -> None:
        """Match hot floor-pinned truncation to an isolated fresh session."""
        _flush_to_quiescent_metrics(self.base_url)
        truncations_before = _read_required_metric(
            self.base_url,
            TRUNCATION_METRIC_NAME,
        )
        commits_before = _read_required_metric(
            self.base_url,
            COMMIT_METRIC_NAME,
        )
        run_commit_qualification(self.base_url)
        _wait_for_metric(
            self.base_url,
            TRUNCATION_METRIC_NAME,
            truncations_before + 3,
        )
        _wait_for_metric(
            self.base_url,
            COMMIT_METRIC_NAME,
            commits_before + 5,
        )
        _flush_to_quiescent_metrics(self.base_url)


class Gemma4StreamingSessionFullKitMixin(Gemma4StreamingSessionOracleKitMixin):
    """Complete production Gemma-4 raw-token session qualification."""

    def test_00_first_real_small_extend_is_warmed(self) -> None:
        """Bound the first real 64-token extend plus 8-token burst."""
        _flush_to_quiescent_metrics(self.base_url)
        elapsed = _run_first_real_small_extend(self.base_url)
        print(f"first_real_small_extend_seconds={elapsed:.6f}")
        _flush_to_quiescent_metrics(self.base_url)

    def test_20_deep_abort_preserves_exact_slot(self) -> None:
        """Keep the exact deep cache boundary after a streamed abort."""
        _flush_to_quiescent_metrics(self.base_url)
        _run_deep_abort_preservation(self.base_url)
        _flush_to_quiescent_metrics(self.base_url)

    def test_30_idempotency_and_recovery(self) -> None:
        """Qualify conflict envelopes, introspection, leases, and re-prime."""
        _flush_to_quiescent_metrics(self.base_url)
        close_before = _read_counter_with_label(
            self.base_url,
            REAP_METRIC_NAME,
            "cause",
            "close",
        )
        timeout_before = _read_counter_with_label(
            self.base_url,
            REAP_METRIC_NAME,
            "cause",
            "timeout",
        )
        assert close_before is not None
        assert timeout_before is not None

        run_recovery_qualification(self.base_url)
        _run_full_log_recovery(self.base_url)
        _wait_for_labeled_metric(
            self.base_url,
            REAP_METRIC_NAME,
            "cause",
            "close",
            close_before + 6,
        )
        _wait_for_labeled_metric(
            self.base_url,
            REAP_METRIC_NAME,
            "cause",
            "timeout",
            timeout_before + 1,
        )
        _flush_to_quiescent_metrics(self.base_url)

    def test_40_swa_pin_accounting(self) -> None:
        """Account for the lagging floor and release it on commit."""
        _flush_to_quiescent_metrics(self.base_url)
        _run_pin_accounting(self.base_url)
        _flush_to_quiescent_metrics(self.base_url)

    def test_50_mutation_edge_cases(self) -> None:
        """Qualify zero-token mutations and non-destructive boundaries."""
        _flush_to_quiescent_metrics(self.base_url)
        _run_edge_cases(self.base_url)
        _flush_to_quiescent_metrics(self.base_url)

    def test_60_truncate_and_abort_leak_closure(self) -> None:
        """Return every session and active-KV gauge to its baseline."""
        _flush_to_quiescent_metrics(self.base_url)
        _run_leak_closure(self.base_url)
        _flush_to_quiescent_metrics(self.base_url)
