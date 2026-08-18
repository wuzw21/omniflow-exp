from __future__ import annotations
from src.experiment.performance_metrics import (
    PerformanceMetrics,
    aggregate_performance_metrics,
    summarize_durations,
)


def test_summarize_durations_reports_interpolated_percentiles() -> None:
    summary = summarize_durations([1, 2, 3, 4], failed_count=1)

    assert summary["count"] == 4
    assert summary["total_ms"] == 10.0
    assert summary["mean_ms"] == 2.5
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == 3.85
    assert summary["failed_count"] == 1


def test_metrics_keep_failed_calls_and_method_wall_time() -> None:
    metrics = PerformanceMetrics(energy_enabled=False)
    metrics.start()
    metrics.record("observe", 4.0)
    metrics.record("act", 8.0, success=False)
    metrics.finish(method_wall_sec=1.25)

    payload = metrics.to_dict()

    assert payload["method_wall_sec"] == 1.25
    assert payload["timing"]["observe"]["count"] == 1
    assert payload["timing"]["act"]["failed_count"] == 1
    assert payload["energy"]["measurement_available"] is False


def test_aggregate_performance_metrics_groups_methods() -> None:
    rows = [
        {
            "method": "omniflow",
            "performance_metrics": {
                "method_wall_sec": 2.0,
                "timing": {
                    "observe": {
                        "samples_ms": [10.0, 20.0],
                        "failed_count": 0,
                    }
                },
                "energy": {
                    "measurement_available": True,
                    "estimated_mwh": 3.0,
                },
            },
        },
        {
            "method": "official_androidworld",
            "performance_metrics": {
                "method_wall_sec": 4.0,
                "timing": {
                    "observe": {
                        "samples_ms": [30.0],
                        "failed_count": 0,
                    }
                },
                "energy": {
                    "measurement_available": False,
                    "estimated_mwh": None,
                },
            },
        },
    ]

    summary = aggregate_performance_metrics(rows)

    assert summary["method_wall_sec"]["mean_sec"] == 3.0
    assert summary["timing"]["observe"]["p50_ms"] == 20.0
    assert summary["energy"]["measurement_available_count"] == 1
    assert summary["energy"]["estimated_mwh_total"] == 3.0
    assert summary["by_method"]["omniflow"]["method_wall_sec"]["p95_sec"] == 2.0
