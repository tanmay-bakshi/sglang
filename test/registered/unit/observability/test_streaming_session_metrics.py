import functools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import prometheus_client
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStreamingSessionMetrics(unittest.TestCase):
    """Exercise the initial streaming-session metric surface."""

    def test_counters_are_exposed_at_zero_with_production_labels(self) -> None:
        """Expose every labeled counter before its first mutation."""
        registry = prometheus_client.CollectorRegistry()
        labels = {
            "model_name": "gemma-4",
            "engine_type": "unified",
            "tp_rank": 0,
            "pp_rank": 0,
            "moe_ep_rank": 0,
        }

        with (
            patch.object(
                prometheus_client,
                "Counter",
                functools.partial(prometheus_client.Counter, registry=registry),
            ),
            patch.object(
                prometheus_client,
                "Gauge",
                functools.partial(prometheus_client.Gauge, registry=registry),
            ),
            patch.object(
                prometheus_client,
                "Histogram",
                functools.partial(prometheus_client.Histogram, registry=registry),
            ),
            patch.object(
                prometheus_client,
                "Summary",
                functools.partial(prometheus_client.Summary, registry=registry),
            ),
        ):
            SchedulerMetricsCollector(
                labels=labels,
                enable_streaming_session=True,
                server_args=SimpleNamespace(
                    prefill_delayer_max_delay_passes=200,
                    prefill_delayer_forward_passes_buckets=None,
                    prefill_delayer_wait_seconds_buckets=None,
                ),
            )

        samples = {
            sample.name: sample
            for metric in registry.collect()
            for sample in metric.samples
        }
        expected_names = {
            "sglang:streaming_session_truncations_total",
            "sglang:streaming_session_commits_total",
            "sglang:streaming_session_aborts_with_slot_preserved_total",
            "sglang:streaming_session_idempotency_conflicts_total",
        }
        for name in expected_names:
            sample = samples[name]
            self.assertEqual(sample.value, 0)
            self.assertEqual(
                sample.labels,
                {key: str(value) for key, value in labels.items()},
            )


if __name__ == "__main__":
    unittest.main()
