import importlib.util
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "observability"
    / "request_trace.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "sglang_request_trace_unit",
    _MODULE_PATH,
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
request_trace = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = request_trace
_MODULE_SPEC.loader.exec_module(request_trace)

REQUEST_TRACE_PREFIX = request_trace.REQUEST_TRACE_PREFIX
RequestTraceEvent = request_trace.RequestTraceEvent
RequestTraceFields = request_trace.RequestTraceFields
RequestTraceRole = request_trace.RequestTraceRole
emit_request_trace = request_trace.emit_request_trace


@pytest.fixture(autouse=True)
def clear_request_trace_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Reset the process-cached opt-in around every test.

    :param monkeypatch: Pytest environment owner.
    """

    monkeypatch.delenv(request_trace.REQUEST_TRACE_ENV, raising=False)
    request_trace.request_trace_enabled.cache_clear()
    yield
    request_trace.request_trace_enabled.cache_clear()


def test_disabled_trace_does_not_read_clock_or_validate_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep the disabled path to one cached branch and no log work."""

    def unexpected_clock() -> int:
        raise AssertionError("disabled tracing read the clock")

    monkeypatch.setattr(request_trace.time, "monotonic_ns", unexpected_clock)
    with caplog.at_level(logging.INFO, logger=request_trace.__name__):
        emit_request_trace(
            RequestTraceEvent.DECODE_FIRST_ISSUE,
            RequestTraceRole.DECODE_SCHEDULER,
            fields=RequestTraceFields(),
        )

    assert caplog.messages == []


def test_enabled_trace_emits_machine_monotonic_correlation_record(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bind ordered request identities, rooms, and graph shape in JSON."""

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "1")
    monkeypatch.setattr(request_trace.time, "monotonic_ns", lambda: 123_456_789)
    monkeypatch.setattr(request_trace.socket, "gethostname", lambda: "gemma-dev-test")
    monkeypatch.setattr(request_trace.os, "getpid", lambda: 314)
    request_trace.request_trace_enabled.cache_clear()

    with caplog.at_level(logging.INFO, logger=request_trace.__name__):
        emit_request_trace(
            RequestTraceEvent.PREFILL_BATCH_COMPLETED,
            RequestTraceRole.PREFILL_SCHEDULER,
            request_ids=("request-a", "request-b"),
            bootstrap_rooms=(41, 42),
            fields=RequestTraceFields(
                process_rank=1,
                batch_size=2,
                batch_token_count=16_384,
                forward_iter=7,
                cuda_graph_active=False,
            ),
        )

    assert len(caplog.messages) == 1
    message = caplog.messages[0]
    assert message.startswith(REQUEST_TRACE_PREFIX)
    record = json.loads(message.removeprefix(REQUEST_TRACE_PREFIX))
    assert record == {
        "schema_version": 1,
        "event": "prefill_batch_completed",
        "role": "prefill_scheduler",
        "monotonic_ns": 123_456_789,
        "sequence": record["sequence"],
        "hostname": "gemma-dev-test",
        "process_id": 314,
        "request_ids": ["request-a", "request-b"],
        "bootstrap_rooms": [41, 42],
        "request_generations": [],
        "process_rank": 1,
        "batch_size": 2,
        "batch_token_count": 16_384,
        "forward_iter": 7,
        "cuda_graph_active": False,
    }
    assert type(record["sequence"]) is int


def test_packed_trace_accepts_replay_safe_room_generation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Correlate transfer workers without inventing a second request id."""

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "true")
    request_trace.request_trace_enabled.cache_clear()

    with caplog.at_level(logging.INFO, logger=request_trace.__name__):
        emit_request_trace(
            RequestTraceEvent.PACKED_TRANSFER_BEGIN,
            RequestTraceRole.PREFILL_TRANSFER,
            bootstrap_rooms=(41,),
            request_generations=("01" * 16,),
            fields=RequestTraceFields(
                process_rank=0,
                copy_group_count=2,
                payload_bytes=230 * 1024 * 1024,
                transport_bytes=232 * 1024 * 1024,
            ),
        )

    record = json.loads(caplog.messages[0].removeprefix(REQUEST_TRACE_PREFIX))
    assert record["request_ids"] == []
    assert record["bootstrap_rooms"] == [41]
    assert record["request_generations"] == ["01" * 16]
    assert record["copy_group_count"] == 2


def test_trace_rejects_misaligned_request_and_room_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject correlation records that cannot be joined deterministically."""

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "on")
    request_trace.request_trace_enabled.cache_clear()

    with pytest.raises(ValueError, match="must align"):
        emit_request_trace(
            RequestTraceEvent.DECODE_HANDOFF_TOKEN_READY,
            RequestTraceRole.DECODE_SCHEDULER,
            request_ids=("request-a", "request-b"),
            bootstrap_rooms=(41,),
            fields=RequestTraceFields(),
        )


def test_trace_rejects_unknown_enable_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed on a misspelled instrumentation contract."""

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "probably")
    request_trace.request_trace_enabled.cache_clear()

    with pytest.raises(ValueError, match=request_trace.REQUEST_TRACE_ENV):
        request_trace.request_trace_enabled()
