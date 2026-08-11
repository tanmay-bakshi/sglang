import importlib.util
import json
import logging
import sys
import threading
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
    request_trace.shutdown_request_trace_writer()
    yield
    request_trace.shutdown_request_trace_writer()
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
        request_trace.flush_request_traces()

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
        request_trace.flush_request_traces()

    record = json.loads(caplog.messages[0].removeprefix(REQUEST_TRACE_PREFIX))
    assert record["request_ids"] == []
    assert record["bootstrap_rooms"] == [41]
    assert record["request_generations"] == ["01" * 16]
    assert record["copy_group_count"] == 2


def test_enabled_trace_moves_log_io_off_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not make inference threads wait for trace materialization or log I/O.

    :param monkeypatch: Pytest mutation fixture.
    """

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "1")
    request_trace.request_trace_enabled.cache_clear()
    writer_entered = threading.Event()
    allow_writer_to_finish = threading.Event()
    producer_finished = threading.Event()

    def blocking_log(*args: object, **kwargs: object) -> None:
        writer_entered.set()
        assert allow_writer_to_finish.wait(1.0)

    monkeypatch.setattr(request_trace.logger, "info", blocking_log)

    def produce() -> None:
        emit_request_trace(
            RequestTraceEvent.DECODE_FIRST_RESULT,
            RequestTraceRole.DECODE_SCHEDULER,
            request_ids=("request-a",),
            bootstrap_rooms=(41,),
            fields=RequestTraceFields(
                process_rank=0,
                batch_size=1,
                batch_token_count=1,
                forward_iter=7,
                cuda_graph_active=True,
                accepted_token_count=16,
            ),
        )
        producer_finished.set()

    producer = threading.Thread(target=produce)
    producer.start()
    assert writer_entered.wait(1.0)
    assert producer_finished.wait(0.1)
    allow_writer_to_finish.set()
    producer.join(1.0)
    request_trace.flush_request_traces()

    assert not producer.is_alive()


def test_writer_shutdown_drains_every_accepted_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make terminal scheduler cleanup a lossless trace barrier.

    :param monkeypatch: Pytest mutation fixture.
    """

    monkeypatch.setenv(request_trace.REQUEST_TRACE_ENV, "1")
    request_trace.request_trace_enabled.cache_clear()
    messages: list[str] = []

    def capture_log(template: str, prefix: str, payload: str) -> None:
        messages.append(template % (prefix, payload))

    monkeypatch.setattr(request_trace.logger, "info", capture_log)
    for index in range(32):
        emit_request_trace(
            RequestTraceEvent.DECODE_FIRST_ISSUE,
            RequestTraceRole.DECODE_SCHEDULER,
            request_ids=(f"request-{index}",),
            bootstrap_rooms=(index,),
        )

    request_trace.shutdown_request_trace_writer()

    assert len(messages) == 32
    records = [
        json.loads(message.removeprefix(REQUEST_TRACE_PREFIX)) for message in messages
    ]
    assert [record["sequence"] for record in records] == list(range(32))
    assert request_trace._request_trace_writer_instance is None


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
