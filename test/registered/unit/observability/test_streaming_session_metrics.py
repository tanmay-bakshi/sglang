import functools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import prometheus_client
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_SCHEDULER,
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ContextOnlyCollector(SchedulerMetricsCollector):
    """Capture construction arguments without registering Prometheus metrics."""

    def __init__(
        self,
        *,
        streaming_session_metrics_enabled: bool,
        **_: object,
    ) -> None:
        """Store the session-family emission capability.

        :param streaming_session_metrics_enabled: Whether this scheduler may emit
            logical streaming-session metrics.
        :param _: Unused production collector arguments.
        """
        self.streaming_session_metrics_enabled = streaming_session_metrics_enabled


class TestStreamingSessionMetrics(unittest.TestCase):
    """Exercise streaming-session metric ownership and surface."""

    def setUp(self) -> None:
        """Disable unrelated MoE reporting in isolated collector fixtures."""
        balancedness_patch = patch(
            "sglang.srt.observability.metrics_collector."
            "exports_expert_balancedness_to_prometheus",
            return_value=False,
        )
        balancedness_patch.start()
        self.addCleanup(balancedness_patch.stop)
        schedule_patch = patch(
            "sglang.srt.observability.metrics_collector.get_schedule",
            return_value=SimpleNamespace(
                prefill_delayer_max_delay_passes=200,
                prefill_delayer_forward_passes_buckets=None,
                prefill_delayer_wait_seconds_buckets=None,
            ),
        )
        schedule_patch.start()
        self.addCleanup(schedule_patch.stop)

    @staticmethod
    def _server_args(
        *,
        enable_metrics: bool = True,
        enable_streaming_session: bool = True,
        enable_metrics_for_all_schedulers: bool = False,
    ) -> SimpleNamespace:
        """Build the metric initialization arguments used by the scheduler.

        :param enable_metrics: Whether Prometheus collection is enabled.
        :param enable_streaming_session: Whether the session API is enabled.
        :param enable_metrics_for_all_schedulers: Whether every scheduler may emit.
        :returns: Minimal production-shaped server arguments.
        """
        return SimpleNamespace(
            enable_metrics=enable_metrics,
            enable_streaming_session=enable_streaming_session,
            enable_metrics_for_all_schedulers=enable_metrics_for_all_schedulers,
            kv_events_config=None,
            disaggregation_mode="null",
            served_model_name="gemma-4",
            extra_metric_labels=None,
            stat_loggers={STAT_LOGGER_ROLE_SCHEDULER: _ContextOnlyCollector},
        )

    @staticmethod
    def _collector(
        *,
        tp_rank: int,
        streaming_session_metrics_enabled: bool,
    ) -> tuple[SchedulerMetricsCollector, prometheus_client.CollectorRegistry]:
        """Build one isolated production collector.

        :param tp_rank: Tensor-parallel rank label.
        :param streaming_session_metrics_enabled: Whether session metrics emit.
        :returns: Collector and its isolated Prometheus registry.
        """
        registry = prometheus_client.CollectorRegistry()
        metric_classes = (
            ("Counter", prometheus_client.Counter),
            ("Gauge", prometheus_client.Gauge),
            ("Histogram", prometheus_client.Histogram),
            ("Summary", prometheus_client.Summary),
        )
        patches = [
            patch.object(
                prometheus_client,
                name,
                functools.partial(metric_class, registry=registry),
            )
            for name, metric_class in metric_classes
        ]
        for metric_patch in patches:
            metric_patch.start()
        try:
            collector = SchedulerMetricsCollector(
                labels={
                    "model_name": "gemma-4",
                    "engine_type": "unified",
                    "tp_rank": tp_rank,
                    "pp_rank": 0,
                    "moe_ep_rank": 0,
                },
                enable_streaming_session=True,
                streaming_session_metrics_enabled=streaming_session_metrics_enabled,
                server_args=SimpleNamespace(
                    prefill_delayer_max_delay_passes=200,
                    prefill_delayer_forward_passes_buckets=None,
                    prefill_delayer_wait_seconds_buckets=None,
                ),
            )
        finally:
            for metric_patch in reversed(patches):
                metric_patch.stop()
        return collector, registry

    def test_context_derives_session_emission_capability(self) -> None:
        """Derive one default emitter while preserving the all-rank escape."""
        ps = SimpleNamespace(
            attn_tp_rank=1,
            pp_rank=0,
            attn_cp_rank=0,
            moe_ep_rank=0,
        )
        cases = (
            (True, True, False, False, False),
            (True, True, True, False, True),
            (True, True, False, True, True),
            (True, False, True, True, False),
            (False, True, True, True, False),
        )

        for (
            enable_metrics,
            enable_streaming_session,
            is_output_rank,
            enable_all_schedulers,
            expected,
        ) in cases:
            with self.subTest(
                enable_metrics=enable_metrics,
                enable_streaming_session=enable_streaming_session,
                is_output_rank=is_output_rank,
                enable_all_schedulers=enable_all_schedulers,
            ):
                observability = SimpleNamespace(
                    enable_metrics=enable_metrics,
                    enable_metrics_for_all_schedulers=enable_all_schedulers,
                    kv_events_config=None,
                    extra_metric_labels=None,
                    stat_loggers={STAT_LOGGER_ROLE_SCHEDULER: _ContextOnlyCollector},
                )
                serving = SimpleNamespace(
                    enable_streaming_session=enable_streaming_session,
                    served_model_name="gemma-4",
                )
                disaggregation = SimpleNamespace(disaggregation_mode="null")
                with (
                    patch(
                        "sglang.srt.observability.metrics_collector.get_observability",
                        return_value=observability,
                    ),
                    patch(
                        "sglang.srt.observability.metrics_collector.get_serving",
                        return_value=serving,
                    ),
                    patch(
                        "sglang.srt.observability.metrics_collector.get_disagg",
                        return_value=disaggregation,
                    ),
                    patch(
                        "sglang.srt.observability.metrics_collector."
                        "resolve_collector_class",
                        return_value=_ContextOnlyCollector,
                    ),
                ):
                    context = SchedulerMetricsCollector.init_new(
                        server_args=self._server_args(
                            enable_metrics=enable_metrics,
                            enable_streaming_session=enable_streaming_session,
                            enable_metrics_for_all_schedulers=enable_all_schedulers,
                        ),
                        ps=ps,
                        tp_rank=1,
                        pp_rank=0,
                        dp_rank=None,
                        enable_priority_scheduling=False,
                        enable_lora=False,
                        enable_hierarchical_cache=False,
                        is_streaming_session_output_rank=is_output_rank,
                    )

                self.assertEqual(
                    context.streaming_session_metrics_enabled,
                    expected,
                )
                if context.collector is not None:
                    self.assertEqual(
                        context.collector.streaming_session_metrics_enabled,
                        expected,
                    )

    def test_non_owner_suppresses_all_session_events_and_gauges(self) -> None:
        """Keep every logical session metric silent on a non-owner rank."""
        collector, registry = self._collector(
            tp_rank=1,
            streaming_session_metrics_enabled=False,
        )
        collector.increment_streaming_session_truncation()
        collector.increment_streaming_session_commit()
        collector.increment_streaming_session_abort_with_slot_preserved()
        collector.increment_streaming_session_idempotency_conflict()
        collector.increment_streaming_session_reap("close")
        collector.log_stats(
            SchedulerStats(
                num_streaming_sessions=3,
                streaming_session_held_tokens=4_096,
                streaming_session_held_swa_tokens=1_024,
            )
        )

        samples = {
            sample.name: sample
            for metric in registry.collect()
            for sample in metric.samples
        }
        counter_names = {
            "sglang:streaming_session_truncations_total",
            "sglang:streaming_session_commits_total",
            "sglang:streaming_session_aborts_with_slot_preserved_total",
            "sglang:streaming_session_idempotency_conflicts_total",
        }
        for name in counter_names:
            self.assertEqual(samples[name].value, 0)
        reap_samples = [
            sample
            for metric in registry.collect()
            for sample in metric.samples
            if sample.name == "sglang:streaming_session_reaps_total"
        ]
        self.assertTrue(all(sample.value == 0 for sample in reap_samples))
        self.assertNotIn("sglang:num_streaming_sessions", samples)
        self.assertNotIn("sglang:streaming_session_held_tokens", samples)
        self.assertNotIn("sglang:streaming_session_held_swa_tokens", samples)

    def test_owner_emits_all_session_events_and_gauges(self) -> None:
        """Publish every logical session metric on the designated owner."""
        collector, registry = self._collector(
            tp_rank=0,
            streaming_session_metrics_enabled=True,
        )
        collector.increment_streaming_session_truncation()
        collector.increment_streaming_session_commit()
        collector.increment_streaming_session_abort_with_slot_preserved()
        collector.increment_streaming_session_idempotency_conflict()
        collector.increment_streaming_session_reap("close")
        collector.log_stats(
            SchedulerStats(
                num_streaming_sessions=3,
                streaming_session_held_tokens=4_096,
                streaming_session_held_swa_tokens=1_024,
            )
        )

        all_samples = [
            sample for metric in registry.collect() for sample in metric.samples
        ]
        samples = {sample.name: sample for sample in all_samples}
        for name in (
            "sglang:streaming_session_truncations_total",
            "sglang:streaming_session_commits_total",
            "sglang:streaming_session_aborts_with_slot_preserved_total",
            "sglang:streaming_session_idempotency_conflicts_total",
        ):
            self.assertEqual(samples[name].value, 1)
        close_reap = next(
            sample
            for sample in all_samples
            if sample.name == "sglang:streaming_session_reaps_total"
            and sample.labels["cause"] == "close"
        )
        self.assertEqual(close_reap.value, 1)
        self.assertEqual(samples["sglang:num_streaming_sessions"].value, 3)
        self.assertEqual(samples["sglang:streaming_session_held_tokens"].value, 4_096)
        self.assertEqual(
            samples["sglang:streaming_session_held_swa_tokens"].value,
            1_024,
        )

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

        all_samples = [
            sample for metric in registry.collect() for sample in metric.samples
        ]
        samples = {sample.name: sample for sample in all_samples}
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
        reap_samples = [
            sample
            for sample in all_samples
            if sample.name == "sglang:streaming_session_reaps_total"
        ]
        self.assertEqual(len(reap_samples), 2)
        self.assertEqual(
            {sample.labels["cause"] for sample in reap_samples},
            {"close", "timeout"},
        )
        for sample in reap_samples:
            self.assertEqual(sample.value, 0)
            self.assertEqual(
                {key: sample.labels[key] for key in labels},
                {key: str(value) for key, value in labels.items()},
            )


if __name__ == "__main__":
    unittest.main()
