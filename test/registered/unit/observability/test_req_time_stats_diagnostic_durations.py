from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats


def test_prefill_diagnostic_durations_are_exact_stage_boundaries() -> None:
    """Prefill output separates model execution from packed transfer."""
    stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.PREFILL)
    stats.prefill_bootstrap_queue_entry_time = 1.0
    stats.wait_queue_entry_time = 2.0
    stats.forward_entry_time = 3.0
    stats.prefill_finished_time = 5.0
    stats.prefill_transfer_queue_entry_time = 6.0
    stats.prefill_kv_transfer_finish_time = 9.0
    stats.completion_time = 10.0

    output = stats.convert_to_duration()

    assert "compute_duration=2000.00ms" in output
    assert "transfer_duration=3000.00ms" in output


def test_decode_diagnostic_duration_ends_at_first_decode_finish() -> None:
    """Decode output isolates the first-token forward boundary."""
    stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
    stats.decode_prealloc_queue_entry_time = 1.0
    stats.decode_transfer_queue_entry_time = 2.0
    stats.wait_queue_entry_time = 3.0
    stats.forward_entry_time = 4.0
    stats.completion_time = 10.0

    stats.set_last_decode_finish_time(5.5)
    stats.set_last_decode_finish_time(7.0)

    output = stats.convert_to_duration()

    assert "first_token_forward_duration=1500.00ms" in output
