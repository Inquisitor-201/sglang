"""Unit tests for HiCache L3 observability metrics."""

from collections import defaultdict
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    StorageMetrics,
    StorageMetricsCollector,
)


class _BoundMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = tuple(sorted(labels.items()))

    def inc(self, value=1):
        self.metric.values[self.labels] += value

    def set(self, value):
        self.metric.values[self.labels] = value

    def observe(self, value):
        self.metric.observations[self.labels].append(value)


class _RecordingMetric:
    def __init__(self, *args, **kwargs):
        self.values = defaultdict(float)
        self.observations = defaultdict(list)

    def labels(self, **labels):
        return _BoundMetric(self, labels)


class _RecordingStorageMetricsCollector(StorageMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


def _value(metric, **labels):
    return metric.values[tuple(sorted(labels.items()))]


def _observations(metric, **labels):
    return metric.observations[tuple(sorted(labels.items()))]


def test_prefetch_snapshot_is_exported_as_deltas():
    collector = _RecordingStorageMetricsCollector(labels={"tp_rank": "0"})
    first = StorageMetrics(
        prefetch_stats={
            "attempts": 2,
            "issued": 1,
            "resolved_partial_hit": 1,
            "l1l2_miss_tokens": 100,
            "l3_miss_tokens": 40,
            "occupancy_ratio": 0.25,
        }
    )

    collector.log_storage_metrics(first)
    collector.log_storage_metrics(first)

    assert (
        _value(
            collector.hicache_prefetch_requests_total,
            tp_rank="0",
            outcome="attempts",
        )
        == 2
    )
    assert (
        _value(
            collector.hicache_l3_query_tokens_total,
            tp_rank="0",
            result="hit",
        )
        == 60
    )
    assert (
        _value(
            collector.hicache_l3_query_tokens_total,
            tp_rank="0",
            result="miss",
        )
        == 40
    )
    assert _value(collector.hicache_prefetch_occupancy_ratio, tp_rank="0") == 0.25

    collector.log_storage_metrics(
        StorageMetrics(
            prefetch_stats={
                "attempts": 3,
                "issued": 2,
                "resolved_partial_hit": 1,
                "resolved_full_hit": 1,
                "l1l2_miss_tokens": 200,
                "l3_miss_tokens": 60,
                "occupancy_ratio": 0.5,
            }
        )
    )

    assert (
        _value(
            collector.hicache_prefetch_requests_total,
            tp_rank="0",
            outcome="attempts",
        )
        == 3
    )
    assert (
        _value(
            collector.hicache_l3_query_tokens_total,
            tp_rank="0",
            result="hit",
        )
        == 140
    )
    assert (
        _value(
            collector.hicache_l3_query_tokens_total,
            tp_rank="0",
            result="miss",
        )
        == 60
    )
    assert _value(collector.hicache_prefetch_occupancy_ratio, tp_rank="0") == 0.5


def test_query_and_fetch_histograms_are_observed():
    collector = _RecordingStorageMetricsCollector(labels={"tp_rank": "0"})

    collector.observe_l3_query(hit_tokens=4096, duration_seconds=0.012)
    collector.observe_l3_fetch(duration_seconds=0.034)

    assert _observations(collector.histogram_l3_query_duration, tp_rank="0") == [0.012]
    assert _observations(collector.histogram_l3_first_miss_tokens, tp_rank="0") == [
        4096
    ]
    assert _observations(collector.histogram_l3_fetch_duration, tp_rank="0") == [0.034]


def test_hybrid_prefetch_operation_has_timing_fields():
    operation = HybridPrefetchOperation("rid", [1, 2, 3])

    assert operation.query_pages == 0
    assert operation.query_duration_seconds == 0.0
    assert operation.fetch_duration_seconds == 0.0


def test_prefetch_accounting_tolerates_legacy_operation_without_timing_fields():
    collector = _RecordingStorageMetricsCollector(labels={"tp_rank": "0"})
    cache = object.__new__(UnifiedRadixCache)
    cache.enable_storage_metrics = True
    cache.storage_metrics_collector = collector
    cache._prefetch_outcome_stats = defaultdict(int)

    operation = SimpleNamespace(
        request_id="legacy-rid",
        stats_requested_tokens=32,
        stats_total_tokens=32,
        storage_hit_count=16,
    )

    cache._account_prefetch_outcome(operation, revoked=False)

    assert _observations(collector.histogram_l3_query_duration, tp_rank="0") == [0.0]
    assert _observations(collector.histogram_l3_first_miss_tokens, tp_rank="0") == [16]
