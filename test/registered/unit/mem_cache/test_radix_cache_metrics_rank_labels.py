import unittest
from collections.abc import Iterable
from typing import ClassVar
from unittest.mock import patch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector


class RecordingCollector:
    """Record labels passed to the radix-cache metrics collector."""

    labels: dict[str, object]

    def __init__(self, labels: dict[str, object]) -> None:
        """Store one collector label set.

        :param labels: Prometheus series labels.
        """

        self.labels = labels


class CacheStub:
    """Expose the collector field initialized by ``BasePrefixCache``."""

    metrics_collector: object | None

    def __init__(self) -> None:
        """Create one cache stub without a collector."""

        self.metrics_collector = None


class ObservabilityStub:
    """Expose operator-owned extra metric labels."""

    extra_metric_labels: dict[str, object]

    def __init__(self, extra_metric_labels: dict[str, object]) -> None:
        """Create one observability-config stub.

        :param extra_metric_labels: Candidate extra Prometheus labels.
        """

        self.extra_metric_labels = extra_metric_labels


class ParallelStub:
    """Expose stable pipeline- and tensor-parallel identity."""

    pp_rank: int
    tp_rank: int

    def __init__(self, pp_rank: int, tp_rank: int) -> None:
        """Create one parallel-context stub.

        :param pp_rank: Pipeline-parallel rank.
        :param tp_rank: Tensor-parallel rank.
        """

        self.pp_rank = pp_rank
        self.tp_rank = tp_rank


class RecordingMetric:
    """Record metric child labels and values without a Prometheus registry."""

    instances: ClassVar[list["RecordingMetric"]] = []

    name: str
    labelnames: tuple[str, ...]
    child_labels: list[dict[str, object]]
    increments: list[float]
    observations: list[float]

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Iterable[str],
        buckets: object | None = None,
    ) -> None:
        """Create one recording metric.

        :param name: Prometheus metric name.
        :param documentation: Prometheus help text.
        :param labelnames: Declared child label names.
        :param buckets: Optional histogram buckets.
        """

        del documentation, buckets
        self.name = name
        self.labelnames = tuple(labelnames)
        self.child_labels = []
        self.increments = []
        self.observations = []
        self.instances.append(self)

    def labels(self, **labels: object) -> "RecordingMetric":
        """Record one child label set.

        :param labels: Bound child labels.
        :returns: This recording child.
        """

        self.child_labels.append(labels)
        return self

    def inc(self, value: float) -> None:
        """Record one counter increment.

        :param value: Increment value.
        """

        self.increments.append(value)

    def observe(self, value: float) -> None:
        """Record one histogram observation.

        :param value: Observed value.
        """

        self.observations.append(value)


class RankLocalRadixCacheMetricsTests(unittest.TestCase):
    """Validate stable rank identity on rank-local cache metrics."""

    def test_collector_labels_include_unoverrideable_tp_and_pp_rank(self) -> None:
        """Eviction and load-back series retain their owning cache rank."""

        cache = CacheStub()
        observability = ObservabilityStub(
            {
                "deployment": "acceptance",
                "pp_rank": "operator-value",
                "tp_rank": "operator-value",
            }
        )
        parallel = ParallelStub(pp_rank=0, tp_rank=1)
        with (
            patch(
                "sglang.srt.mem_cache.base_prefix_cache.get_observability",
                return_value=observability,
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.mem_cache.base_prefix_cache.resolve_collector_class",
                return_value=RecordingCollector,
            ),
        ):
            BasePrefixCache.init_metrics_collector(cache)

        collector = cache.metrics_collector
        self.assertIsInstance(collector, RecordingCollector)
        self.assertEqual(
            collector.labels,
            {
                "cache_type": "CacheStub",
                "deployment": "acceptance",
                "pp_rank": 0,
                "tp_rank": 1,
            },
        )

    def test_eviction_and_load_back_emit_the_same_rank_identity(self) -> None:
        """Both lifecycle counters bind the owning TP and PP rank labels."""

        RecordingMetric.instances.clear()
        labels = {
            "cache_type": "UnifiedRadixCache",
            "pp_rank": 0,
            "tp_rank": 1,
        }
        with (
            patch.object(
                RadixCacheMetricsCollector,
                "_counter_cls",
                RecordingMetric,
            ),
            patch.object(
                RadixCacheMetricsCollector,
                "_histogram_cls",
                RecordingMetric,
            ),
        ):
            collector = RadixCacheMetricsCollector(labels=labels)
            collector.increment_eviction_num_tokens(1024)
            collector.increment_load_back_num_tokens(448, pool="kv")

        metrics = {metric.name: metric for metric in RecordingMetric.instances}
        eviction = metrics["sglang:evicted_tokens_total"]
        load_back = metrics["sglang:load_back_tokens_total"]
        self.assertEqual(eviction.labelnames, tuple(labels))
        self.assertEqual(load_back.labelnames, (*labels, "pool"))
        self.assertEqual(eviction.child_labels, [labels])
        self.assertEqual(load_back.child_labels, [labels | {"pool": "kv"}])
        self.assertEqual(eviction.increments, [1024])
        self.assertEqual(load_back.increments, [448])


if __name__ == "__main__":
    unittest.main()
