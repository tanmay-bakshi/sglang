"""Live qualification probe for the raw-token streaming-session API."""

import argparse
import uuid
from dataclasses import dataclass
from typing import Any

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_PATH = "/models/gemma-4-31B-it-NVFP4"


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


class SessionClient:
    """Small HTTP client that preserves raw token IDs exactly."""

    _base_url: str

    def __init__(self, base_url: str) -> None:
        """Initialize the client.

        :param base_url: Inference server base URL.
        """
        self._base_url = base_url.rstrip("/")

    def open(self, *, manual_commit: bool = False) -> str:
        """Open one streaming session.

        :param manual_commit: Whether the session uses an explicit commit floor.
        :returns: Server-generated session identifier.
        """
        payload: dict[str, Any] = {
            "capacity_of_str_len": 0,
            "streaming": True,
        }
        if manual_commit:
            payload["manual_commit"] = True
        response = requests.post(
            self._base_url + "/open_session",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json())

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
        :returns: Normalized generation result.
        """
        session_params: dict[str, Any] = {"id": session_id, "rid": None}
        if truncate_to is not None:
            session_params["truncate_to"] = truncate_to
        if commit_to is not None:
            session_params["commit_to"] = commit_to
        if expected_tip is not None:
            session_params["expected_tip"] = expected_tip

        response = requests.post(
            self._base_url + "/generate",
            json={
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
            },
            timeout=300,
        )
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


def main() -> None:
    """Run the requested live qualification stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32300")
    parser.add_argument("--stage", choices=("truncate", "commit"), required=True)
    args = parser.parse_args()

    if args.stage == "truncate":
        run_truncate_qualification(args.base_url)
    elif args.stage == "commit":
        run_commit_qualification(args.base_url)
    print(f"streaming-session {args.stage} qualification passed")


if __name__ == "__main__":
    main()
