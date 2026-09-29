"""Unit tests for comparative evaluation analysis."""

from __future__ import annotations

import json
from typing import Any

import pytest

from nearfield360.perception.comparison import (
    ComparisonError,
    EvaluationComparison,
    MetricDelta,
    compare_evaluations,
    compute_metric_delta,
)


def test_compute_metric_delta_basic() -> None:
    delta = compute_metric_delta(0.5, 0.7)
    assert isinstance(delta, MetricDelta)
    assert delta.baseline == 0.5
    assert delta.candidate == 0.7
    assert delta.delta == pytest.approx(0.2)
    assert delta.percent_change == pytest.approx(40.0)
    assert delta.ratio is None


def test_compute_metric_delta_with_ratio() -> None:
    delta = compute_metric_delta(20.0, 25.0, compute_ratio=True)
    assert delta.delta == pytest.approx(5.0)
    assert delta.percent_change == pytest.approx(25.0)
    assert delta.ratio == pytest.approx(1.25)


def test_compute_metric_delta_none_values() -> None:
    delta_none_candidate = compute_metric_delta(0.5, None)
    assert delta_none_candidate.baseline == 0.5
    assert delta_none_candidate.candidate is None
    assert delta_none_candidate.delta is None
    assert delta_none_candidate.percent_change is None

    delta_none_baseline = compute_metric_delta(None, 0.8)
    assert delta_none_baseline.baseline is None
    assert delta_none_baseline.candidate == 0.8
    assert delta_none_baseline.delta is None

    delta_both_none = compute_metric_delta(None, None)
    assert delta_both_none.baseline is None
    assert delta_both_none.candidate is None
    assert delta_both_none.delta is None


def test_compute_metric_delta_zero_baseline() -> None:
    delta = compute_metric_delta(0.0, 0.5, compute_ratio=True)
    assert delta.delta == pytest.approx(0.5)
    assert delta.percent_change is None
    assert delta.ratio is None


def _make_seg_report(
    *,
    mean_iou: float = 0.65,
    pixel_acc: float = 0.88,
    classes: dict[str, dict[str, Any]] | None = None,
    ece: float | None = None,
    timing_mean_ms: float | None = None,
) -> dict[str, Any]:
    if classes is None:
        classes = {
            "road": {"iou": 0.90, "target_pixels": 1000},
            "lanemarks": {"iou": 0.40, "target_pixels": 200},
        }
    metrics: dict[str, Any] = {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_acc,
        "image_count": 10,
        "pixel_count": 50000,
        "classes": classes,
    }
    if ece is not None:
        metrics["confidence_analysis"] = {
            "num_bins": 5,
            "pixel_count": 50000,
            "mean_confidence": 0.75,
            "ece": ece,
            "bins": [],
        }
    report: dict[str, Any] = {
        "samples": {"expected": 10, "evaluated": 10, "missing_predictions": []},
        "metrics": metrics,
    }
    if timing_mean_ms is not None:
        report["timing"] = {
            "samples": 10,
            "total_ms": timing_mean_ms * 10,
            "mean_ms": timing_mean_ms,
            "p50_ms": timing_mean_ms,
            "p95_ms": timing_mean_ms * 1.2,
            "min_ms": timing_mean_ms * 0.8,
            "max_ms": timing_mean_ms * 1.5,
            "samples_per_second": 1000.0 / timing_mean_ms,
        }
    return report


def _make_det_report(
    *,
    map_score: float = 0.55,
    classes: dict[str, dict[str, Any]] | None = None,
    timing_mean_ms: float | None = None,
) -> dict[str, Any]:
    if classes is None:
        classes = {
            "vehicles": {"average_precision": 0.70, "predictions": 15, "targets": 12},
            "person": {"average_precision": 0.40, "predictions": 8, "targets": 10},
        }
    metrics: dict[str, Any] = {
        "mean_average_precision": map_score,
        "iou_threshold": 0.5,
        "image_count": 5,
        "classes": classes,
    }
    report: dict[str, Any] = {
        "samples": {"expected": 5, "evaluated": 5, "missing_predictions": []},
        "metrics": metrics,
    }
    if timing_mean_ms is not None:
        report["timing"] = {
            "samples": 5,
            "total_ms": timing_mean_ms * 5,
            "mean_ms": timing_mean_ms,
            "p50_ms": timing_mean_ms,
            "p95_ms": timing_mean_ms * 1.1,
            "min_ms": timing_mean_ms * 0.9,
            "max_ms": timing_mean_ms * 1.2,
            "samples_per_second": 1000.0 / timing_mean_ms,
        }
    return report


def test_compare_segmentation_reports() -> None:
    baseline = _make_seg_report(mean_iou=0.60, pixel_acc=0.85)
    candidate = _make_seg_report(
        mean_iou=0.68,
        pixel_acc=0.89,
        classes={
            "road": {"iou": 0.92, "target_pixels": 1000},
            "lanemarks": {"iou": 0.44, "target_pixels": 200},
        },
    )

    cmp = compare_evaluations(baseline, candidate)
    assert isinstance(cmp, EvaluationComparison)
    assert cmp.task == "segmentation"
    assert cmp.summary_deltas["mean_iou"].delta == pytest.approx(0.08)
    assert cmp.summary_deltas["mean_iou"].percent_change == pytest.approx(13.33)
    assert cmp.summary_deltas["pixel_accuracy"].delta == pytest.approx(0.04)
    assert cmp.class_deltas["road"].delta == pytest.approx(0.02)
    assert cmp.class_deltas["lanemarks"].delta == pytest.approx(0.04)

    payload = cmp.as_dict()
    assert payload["task"] == "segmentation"
    assert json.dumps(payload)  # JSON-serializable


def test_compare_detection_reports() -> None:
    baseline = _make_det_report(map_score=0.50)
    candidate = _make_det_report(map_score=0.58)

    cmp = compare_evaluations(baseline, candidate)
    assert isinstance(cmp, EvaluationComparison)
    assert cmp.task == "detection"
    assert cmp.summary_deltas["mean_average_precision"].delta == pytest.approx(0.08)
    assert cmp.summary_deltas["mean_average_precision"].percent_change == pytest.approx(16.0)
    assert "vehicles" in cmp.class_deltas
    assert "person" in cmp.class_deltas

    payload = cmp.as_dict()
    assert payload["task"] == "detection"
    assert json.dumps(payload)


def test_compare_with_calibration() -> None:
    baseline = _make_seg_report(ece=0.08)
    candidate = _make_seg_report(ece=0.03)

    cmp = compare_evaluations(baseline, candidate)
    assert cmp.calibration_deltas is not None
    assert cmp.calibration_deltas["ece"].delta == pytest.approx(-0.05)
    assert cmp.calibration_deltas["ece"].percent_change == pytest.approx(-62.5)


def test_compare_with_timing() -> None:
    baseline = _make_seg_report(timing_mean_ms=20.0)
    candidate = _make_seg_report(timing_mean_ms=16.0)

    cmp = compare_evaluations(baseline, candidate)
    assert cmp.timing_deltas is not None
    assert cmp.timing_deltas["mean_ms"].delta == pytest.approx(-4.0)
    assert cmp.timing_deltas["mean_ms"].ratio == pytest.approx(0.8)


def test_compare_disjoint_classes() -> None:
    baseline = _make_seg_report(
        classes={
            "road": {"iou": 0.90, "target_pixels": 1000},
            "lanemarks": {"iou": 0.40, "target_pixels": 200},
        }
    )
    candidate = _make_seg_report(
        classes={
            "road": {"iou": 0.92, "target_pixels": 1000},
            "curb": {"iou": 0.55, "target_pixels": 300},
        }
    )

    cmp = compare_evaluations(baseline, candidate)
    assert set(cmp.class_deltas.keys()) == {"road", "lanemarks", "curb"}
    assert cmp.class_deltas["road"].delta == pytest.approx(0.02)
    assert cmp.class_deltas["lanemarks"].candidate is None
    assert cmp.class_deltas["lanemarks"].delta is None
    assert cmp.class_deltas["curb"].baseline is None
    assert cmp.class_deltas["curb"].delta is None


def test_compare_rejects_missing_metrics() -> None:
    with pytest.raises(ComparisonError, match="Baseline report is missing 'metrics'"):
        compare_evaluations({}, {"metrics": {"mean_iou": 0.5}})

    with pytest.raises(ComparisonError, match="Candidate report is missing 'metrics'"):
        compare_evaluations({"metrics": {"mean_iou": 0.5}}, {})


def test_compare_rejects_task_mismatch() -> None:
    baseline = _make_seg_report()
    candidate = _make_det_report()
    with pytest.raises(ComparisonError, match="Cannot compare segmentation report with detection"):
        compare_evaluations(baseline, candidate)


def test_compare_rejects_unrecognized_metrics() -> None:
    with pytest.raises(ComparisonError, match="Could not determine evaluation task"):
        compare_evaluations({"metrics": {"foo": 1}}, {"metrics": {"foo": 2}})
