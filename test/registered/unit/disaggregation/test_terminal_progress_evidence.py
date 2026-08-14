import io
import logging

import pytest
from sglang.srt.disaggregation.common.packed_staging_protocol import PackedRequestKey
from sglang.srt.disaggregation.terminal_progress.evidence import (
    TERMINAL_PROGRESS_TIMING_LOG_PREFIX,
    TerminalProgressTimingEmitter,
    TerminalProgressTimingLogger,
    TerminalProgressTimingSample,
    parse_terminal_progress_timing_log_line,
    summarize_terminal_progress_request,
)
from sglang.srt.disaggregation.terminal_progress.identity import (
    TerminalOwnerRole,
    TerminalProcessIdentity,
    TerminalRequestBinding,
)
from sglang.srt.disaggregation.terminal_progress.owner_events import (
    TerminalOwnerTimingField,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_PROCESS_GENERATION = bytes.fromhex("102132435465768798a9bacbdcedfe0f")


class _RejectingSink:
    """Timing sink fixture which rejects every sample."""

    def emit(self, sample: TerminalProgressTimingSample) -> None:
        """Reject one otherwise valid sample.

        :param sample: Exact sample presented to the fixture.
        """

        if type(sample) is not TerminalProgressTimingSample:
            raise TypeError("sample must be TerminalProgressTimingSample")
        raise RuntimeError("synthetic timing sink failure")


def _binding(
    field: TerminalOwnerTimingField,
    *,
    tp_rank: int = 0,
) -> TerminalRequestBinding:
    """Build one role-correct fixture binding for a timing field.

    :param field: Timing field represented by the sample.
    :param tp_rank: Local TP rank.
    :returns: Exact request binding.
    """

    source_fields = {
        TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF,
        TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY,
        TerminalOwnerTimingField.GATEWAY_PUBLICATION,
    }
    role = (
        TerminalOwnerRole.SOURCE if field in source_fields else TerminalOwnerRole.DECODE
    )
    return TerminalRequestBinding(
        request_key=PackedRequestKey(
            room_id=17,
            request_generation=bytes.fromhex("00112233445566778899aabbccddeeff"),
        ),
        owner=TerminalProcessIdentity(
            process_generation=_PROCESS_GENERATION,
            role=role,
            tp_rank=tp_rank,
            tp_size=2,
        ),
        rank_manifest_digest=bytes.fromhex("11" * 32),
        allocation_digest=bytes.fromhex("22" * 32),
    )


def _sample(
    field: TerminalOwnerTimingField,
    *,
    sample_key: str | None = None,
    started_ns: int = 10_000_000,
    completed_ns: int = 12_500_000,
    tp_rank: int = 0,
) -> TerminalProgressTimingSample:
    """Build one exact timing sample.

    :param field: Frozen interval field.
    :param sample_key: Optional explicit cardinality key.
    :param started_ns: Local interval start.
    :param completed_ns: Local interval completion.
    :param tp_rank: Local TP rank.
    :returns: Validated timing sample.
    """

    key = field.value if sample_key is None else sample_key
    return TerminalProgressTimingSample(
        binding=_binding(field, tp_rank=tp_rank),
        field=field,
        sample_key=key,
        started_ns=started_ns,
        completed_ns=completed_ns,
    )


def test_timing_log_round_trip_preserves_complete_identity() -> None:
    sample = _sample(
        TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY,
        sample_key="main:writer-0:handle-7",
    )
    line = f"INFO terminal {sample.to_log_record()}\n"

    parsed = parse_terminal_progress_timing_log_line(line)

    assert parsed == sample
    assert sample.duration_ms == 2.5
    assert TERMINAL_PROGRESS_TIMING_LOG_PREFIX in line


def test_timing_logger_emits_one_parser_stable_record() -> None:
    stream = io.StringIO()
    logger = logging.Logger("terminal-progress-evidence-test")
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    sink = TerminalProgressTimingLogger(logger)
    sample = _sample(TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF)

    sink.emit(sample)

    assert parse_terminal_progress_timing_log_line(stream.getvalue()) == sample


def test_timing_emitter_keeps_evidence_failure_off_lifecycle_path() -> None:
    stream = io.StringIO()
    logger = logging.Logger("terminal-progress-evidence-failure-test")
    logger.addHandler(logging.StreamHandler(stream))
    emitter = TerminalProgressTimingEmitter(_RejectingSink(), logger)
    field = TerminalOwnerTimingField.GATEWAY_PUBLICATION

    accepted = emitter.emit(
        binding=_binding(field),
        field=field,
        sample_key="canonical-source-publisher",
        started_ns=10,
        completed_ns=20,
    )

    assert accepted is False
    assert "synthetic timing sink failure" in stream.getvalue()


def test_parser_rejects_identity_and_duration_tampering() -> None:
    sample = _sample(TerminalOwnerTimingField.SCATTER_CALLBACK_DELIVERY)
    payload = sample.to_payload()
    payload["duration_ms"] = 9.0

    with pytest.raises(ValueError, match="duration differs"):
        TerminalProgressTimingSample.from_payload(payload)

    payload = sample.to_payload()
    payload["binding_digest"] = "00" * 32
    with pytest.raises(ValueError, match="binding digest"):
        TerminalProgressTimingSample.from_payload(payload)


def test_request_summary_preserves_cardinality_and_uses_local_maximum() -> None:
    samples = [
        _sample(field, completed_ns=11_000_000) for field in TerminalOwnerTimingField
    ]
    samples.extend(
        (
            _sample(
                TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY,
                sample_key="boundary:writer-0:handle-8",
                completed_ns=14_000_000,
            ),
            _sample(
                TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF,
                sample_key="source-rank-1",
                completed_ns=13_000_000,
                tp_rank=1,
            ),
        )
    )

    summary = summarize_terminal_progress_request((*samples, samples[0]))
    payload = summary.to_payload()

    assert (
        payload["critical_path_ms"][
            TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY.value
        ]
        == 4.0
    )
    assert (
        payload["critical_path_ms"][
            TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF.value
        ]
        == 3.0
    )
    assert (
        payload["sample_counts"][
            TerminalOwnerTimingField.NATIVE_TERMINAL_DELIVERY.value
        ]
        == 2
    )
    assert (
        payload["sample_counts"][
            TerminalOwnerTimingField.PRODUCER_TO_OWNER_HANDOFF.value
        ]
        == 2
    )


def test_request_summary_rejects_missing_or_conflicting_cardinality() -> None:
    samples = [_sample(field) for field in TerminalOwnerTimingField]

    with pytest.raises(ValueError, match="incomplete"):
        summarize_terminal_progress_request(samples[:-1])

    conflict = _sample(
        samples[0].field,
        sample_key=samples[0].sample_key,
        completed_ns=samples[0].completed_ns + 1,
    )
    with pytest.raises(ValueError, match="conflicting"):
        summarize_terminal_progress_request((*samples, conflict))


def test_timing_sample_rejects_crossed_local_interval() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        _sample(
            TerminalOwnerTimingField.METADATA_CONSUMPTION,
            started_ns=12,
            completed_ns=11,
        )
