"""Live qualification probe for the raw-token streaming-session API."""

import argparse
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
    :ivar tip: Current logical context length, when present.
    :ivar floor: Current rollback floor, when present.
    :ivar protected: Radix-owned prefix boundary, when present.
    :ivar inflight: Whether one request currently owns the session.
    :ivar held_tokens: KV tokens held by the session, when present.
    :ivar last_rid: Last successfully adopted request identifier.
    """

    exists: bool
    tip: int | None
    floor: int | None
    protected: int | None
    inflight: bool
    held_tokens: int | None
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
            assert value is None or type(value) is int, (
                f"session_info {field} must be an integer or null, got {value!r}"
            )
            assert value is None or value >= 0

        last_rid = body["last_rid"]
        assert last_rid is None or isinstance(last_rid, str)
        if body["exists"]:
            assert type(body["tip"]) is int
            assert type(body["floor"]) is int
            assert type(body["protected"]) is int
            assert type(body["held_tokens"]) is int

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
        )
        fresh = client.generate(
            fresh_session,
            exact_context[:target] + delta,
            max_new_tokens=16,
            extra_key=fresh_key,
        )
        assert fresh.cached_tokens == 0, (
            f"fresh truncate oracle inherited {fresh.cached_tokens} cached tokens"
        )
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
        assert fresh.cached_tokens == 0, (
            f"fresh commit oracle inherited {fresh.cached_tokens} cached tokens"
        )
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
            assert len(sse_data) == 2, (
                f"conflict stream must contain one event and [DONE], got {sse_data!r}"
            )
            assert sse_data[1] == "[DONE]", sse_data
            _assert_conflict_error(json.loads(sse_data[0]), request_rid)
        else:
            assert response.status_code == 409, response.text
            _assert_conflict_error(response.json(), request_rid)

    after = client.session_info(session_id)
    assert after == before, (
        f"stale request mutated session state: before={before}, after={after}"
    )


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


def _qualify_timeout_reads(client: SessionClient) -> None:
    """Prove recovery reads do not extend a session's inactivity lease.

    :param client: Live session client.
    """
    session_timeout = 1.25
    session_id = client.open(timeout=session_timeout)
    started = time.monotonic()
    reads_while_live = 0
    last_info = client.session_info(session_id)
    try:
        while time.monotonic() - started < 5.0:
            last_info = client.session_info(session_id)
            if not last_info.exists:
                break
            reads_while_live += 1
            time.sleep(0.1)

        assert reads_while_live >= 2
        assert not last_info.exists, (
            "session_info reads refreshed the inactivity timeout: "
            f"elapsed={time.monotonic() - started:.3f}s"
        )
        assert last_info.inflight is False
        assert last_info.held_tokens in {None, 0}
        assert last_info.last_rid is None
    finally:
        if last_info.exists:
            client.close(session_id)


def run_recovery_qualification(base_url: str) -> None:
    """Run Stage 4 idempotency and recovery acceptance cases.

    :param base_url: Live inference server URL.
    """
    client = SessionClient(base_url)
    context, _, _ = _build_gemma4_context()
    session_id = client.open()
    conflict_count = 0
    conflict_metric_before = _read_counter(base_url, CONFLICT_METRIC_NAME)

    try:
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
            extra_key="stage4-recovery-" + uuid.uuid4().hex,
            request_rid=seed_rid,
        )
        seed = client.parse_generate(client.post_generate(seed_payload))
        assert seed.rid == seed_rid

        stable = client.session_info(session_id)
        assert stable.exists
        assert stable.tip == seed.tip
        assert stable.floor == seed.tip
        assert stable.protected is not None
        assert stable.floor is not None
        assert stable.protected <= stable.floor
        assert stable.held_tokens is not None and stable.held_tokens > 0
        assert stable.inflight is False
        assert stable.last_rid == seed_rid

        _assert_conflict_preserves_state(
            client,
            session_id,
            stable,
            seed_payload,
        )
        conflict_count += 1

        assert stable.tip is not None
        high_rid = "stage4-high-" + uuid.uuid4().hex
        high_payload = client.generate_payload(
            session_id,
            context[96:112],
            max_new_tokens=4,
            expected_tip=stable.tip + 1,
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
        if conflict_metric_before is not None or conflict_metric_after is not None:
            assert conflict_metric_before is not None
            assert conflict_metric_after is not None
            assert conflict_metric_after - conflict_metric_before == conflict_count

        accepted_rid = "stage4-accepted-" + uuid.uuid4().hex
        accepted = client.generate(
            session_id,
            context[112:128],
            max_new_tokens=4,
            expected_tip=stable.tip,
            request_rid=accepted_rid,
        )
        assert accepted.rid == accepted_rid
        accepted_info = client.session_info(session_id)
        assert accepted_info.tip == accepted.tip
        assert accepted_info.last_rid == accepted_rid

        with ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_info = list(executor.map(client.session_info, [session_id] * 32))
        assert concurrent_info == [accepted_info] * 32

        assert accepted_info.tip is not None
        abort_rid = "stage4-abort-" + uuid.uuid4().hex
        abort_payload = client.generate_payload(
            session_id,
            context[128:144],
            max_new_tokens=100_000,
            expected_tip=accepted_info.tip,
            ignore_eos=True,
            request_rid=abort_rid,
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            abort_future = executor.submit(client.post_generate, abort_payload)
            _wait_for_inflight(client, session_id, True, timeout=30)

            inflight_info = list(executor.map(client.session_info, [session_id] * 16))
            assert all(info.exists and info.inflight for info in inflight_info)
            assert all(info.tip == accepted_info.tip for info in inflight_info)

            client.abort(abort_rid)
            abort_response = abort_future.result(timeout=60)

        assert abort_response.status_code == 200, abort_response.text
        abort_body = abort_response.json()
        assert abort_body["meta_info"]["finish_reason"]["type"] == "abort"
        _wait_for_inflight(client, session_id, False, timeout=30)
        healed = client.session_info(session_id)
        assert healed == accepted_info, (
            f"aborted request did not heal to its pre-burst state: {healed}"
        )

        recovery_rid = "stage4-recovery-" + uuid.uuid4().hex
        recovery = client.generate(
            session_id,
            context[144:160],
            max_new_tokens=4,
            expected_tip=accepted_info.tip,
            request_rid=recovery_rid,
        )
        assert recovery.rid == recovery_rid
        recovered_info = client.session_info(session_id)
        assert recovered_info.tip == recovery.tip
        assert recovered_info.last_rid == recovery_rid
    finally:
        client.close(session_id)

    closed = client.session_info(session_id)
    assert not closed.exists
    assert closed.inflight is False
    assert closed.held_tokens in {None, 0}
    assert closed.last_rid is None
    _qualify_timeout_reads(client)


def main() -> None:
    """Run the requested live qualification stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32300")
    parser.add_argument(
        "--stage",
        choices=("truncate", "commit", "recovery"),
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "truncate":
        run_truncate_qualification(args.base_url)
    elif args.stage == "commit":
        run_commit_qualification(args.base_url)
    elif args.stage == "recovery":
        run_recovery_qualification(args.base_url)
    print(f"streaming-session {args.stage} qualification passed")


if __name__ == "__main__":
    main()
