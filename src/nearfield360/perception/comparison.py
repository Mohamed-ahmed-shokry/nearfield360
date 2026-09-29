"""Comparative analysis between evaluation report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ComparisonError(ValueError):
    """Raised when evaluation reports are incompatible, malformed, or missing sections."""


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Represents a comparative difference between baseline and candidate metric values."""

    baseline: float | None
    candidate: float | None
    delta: float | None
    percent_change: float | None = None
    ratio: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "percent_change": self.percent_change,
            "ratio": self.ratio,
        }


def compute_metric_delta(
    baseline: float | int | None,
    candidate: float | int | None,
    *,
    compute_percent: bool = True,
    compute_ratio: bool = False,
    precision: int = 4,
) -> MetricDelta:
    """Compute difference, percent change, and ratio between two scalar values."""
    if baseline is None or candidate is None:
        b_val = float(baseline) if baseline is not None else None
        c_val = float(candidate) if candidate is not None else None
        return MetricDelta(baseline=b_val, candidate=c_val, delta=None)

    b = float(baseline)
    c = float(candidate)
    delta = round(c - b, precision)

    percent_change: float | None = None
    if compute_percent and b != 0.0:
        percent_change = round(((c - b) / abs(b)) * 100.0, 2)

    ratio: float | None = None
    if compute_ratio and b != 0.0:
        ratio = round(c / b, 3)

    return MetricDelta(
        baseline=round(b, precision),
        candidate=round(c, precision),
        delta=delta,
        percent_change=percent_change,
        ratio=ratio,
    )


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Structured delta comparison between two evaluation reports."""

    task: str
    summary_deltas: dict[str, MetricDelta]
    class_deltas: dict[str, MetricDelta]
    timing_deltas: dict[str, MetricDelta] | None
    calibration_deltas: dict[str, MetricDelta] | None
    baseline_samples: dict[str, Any] | None
    candidate_samples: dict[str, Any] | None
    baseline_model: dict[str, Any] | None
    candidate_model: dict[str, Any] | None
    baseline_split: dict[str, Any] | None
    candidate_split: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible dictionary."""
        return {
            "task": self.task,
            "summary_deltas": {k: v.as_dict() for k, v in self.summary_deltas.items()},
            "class_deltas": {k: v.as_dict() for k, v in self.class_deltas.items()},
            "timing_deltas": (
                {k: v.as_dict() for k, v in self.timing_deltas.items()}
                if self.timing_deltas is not None
                else None
            ),
            "calibration_deltas": (
                {k: v.as_dict() for k, v in self.calibration_deltas.items()}
                if self.calibration_deltas is not None
                else None
            ),
            "baseline_samples": self.baseline_samples,
            "candidate_samples": self.candidate_samples,
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "baseline_split": self.baseline_split,
            "candidate_split": self.candidate_split,
        }


def _determine_task(metrics: dict[str, Any], report_label: str) -> str:
    if "mean_iou" in metrics:
        return "segmentation"
    if "mean_average_precision" in metrics:
        return "detection"
    raise ComparisonError(f"Could not determine evaluation task from {report_label} report metrics")


def compare_evaluations(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> EvaluationComparison:
    """Compare two evaluation report dictionaries and produce an EvaluationComparison."""
    if not isinstance(baseline, dict) or not isinstance(baseline.get("metrics"), dict):
        raise ComparisonError("Baseline report is missing 'metrics' section")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("metrics"), dict):
        raise ComparisonError("Candidate report is missing 'metrics' section")

    b_metrics = baseline["metrics"]
    c_metrics = candidate["metrics"]

    b_task = _determine_task(b_metrics, "baseline")
    c_task = _determine_task(c_metrics, "candidate")

    if b_task != c_task:
        raise ComparisonError(f"Cannot compare {b_task} report with {c_task} report: task mismatch")

    summary_deltas: dict[str, MetricDelta] = {}
    class_deltas: dict[str, MetricDelta] = {}

    if b_task == "segmentation":
        summary_deltas["mean_iou"] = compute_metric_delta(
            b_metrics.get("mean_iou"), c_metrics.get("mean_iou")
        )
        summary_deltas["pixel_accuracy"] = compute_metric_delta(
            b_metrics.get("pixel_accuracy"), c_metrics.get("pixel_accuracy")
        )
        summary_deltas["image_count"] = compute_metric_delta(
            b_metrics.get("image_count"), c_metrics.get("image_count"), precision=0
        )
        summary_deltas["pixel_count"] = compute_metric_delta(
            b_metrics.get("pixel_count"), c_metrics.get("pixel_count"), precision=0
        )

        b_classes = b_metrics.get("classes") or {}
        c_classes = c_metrics.get("classes") or {}
        all_classes = sorted(set(b_classes) | set(c_classes))
        for cls_name in all_classes:
            b_entry = b_classes.get(cls_name)
            c_entry = c_classes.get(cls_name)
            b_iou = b_entry.get("iou") if isinstance(b_entry, dict) else None
            c_iou = c_entry.get("iou") if isinstance(c_entry, dict) else None
            class_deltas[cls_name] = compute_metric_delta(b_iou, c_iou)

    else:  # detection
        summary_deltas["mean_average_precision"] = compute_metric_delta(
            b_metrics.get("mean_average_precision"),
            c_metrics.get("mean_average_precision"),
        )
        summary_deltas["image_count"] = compute_metric_delta(
            b_metrics.get("image_count"), c_metrics.get("image_count"), precision=0
        )
        if b_metrics.get("iou_threshold") is not None or c_metrics.get("iou_threshold") is not None:
            summary_deltas["iou_threshold"] = compute_metric_delta(
                b_metrics.get("iou_threshold"), c_metrics.get("iou_threshold")
            )

        b_classes = b_metrics.get("classes") or {}
        c_classes = c_metrics.get("classes") or {}
        all_classes = sorted(set(b_classes) | set(c_classes))
        for cls_name in all_classes:
            b_entry = b_classes.get(cls_name)
            c_entry = c_classes.get(cls_name)
            b_ap = b_entry.get("average_precision") if isinstance(b_entry, dict) else None
            c_ap = c_entry.get("average_precision") if isinstance(c_entry, dict) else None
            class_deltas[cls_name] = compute_metric_delta(b_ap, c_ap)

    # Calibration comparison (if confidence_analysis present in either)
    b_conf = b_metrics.get("confidence_analysis")
    c_conf = c_metrics.get("confidence_analysis")
    calibration_deltas: dict[str, MetricDelta] | None = None
    if isinstance(b_conf, dict) and isinstance(c_conf, dict):
        calibration_deltas = {
            "ece": compute_metric_delta(b_conf.get("ece"), c_conf.get("ece")),
            "mean_confidence": compute_metric_delta(
                b_conf.get("mean_confidence"), c_conf.get("mean_confidence")
            ),
        }

    # Timing comparison (if timing present in both)
    b_timing = baseline.get("timing")
    c_timing = candidate.get("timing")
    timing_deltas: dict[str, MetricDelta] | None = None
    if isinstance(b_timing, dict) and isinstance(c_timing, dict):
        timing_deltas = {
            "mean_ms": compute_metric_delta(
                b_timing.get("mean_ms"), c_timing.get("mean_ms"), compute_ratio=True
            ),
            "p50_ms": compute_metric_delta(
                b_timing.get("p50_ms"), c_timing.get("p50_ms"), compute_ratio=True
            ),
            "p95_ms": compute_metric_delta(
                b_timing.get("p95_ms"), c_timing.get("p95_ms"), compute_ratio=True
            ),
            "samples_per_second": compute_metric_delta(
                b_timing.get("samples_per_second"),
                c_timing.get("samples_per_second"),
                compute_ratio=True,
            ),
        }

    return EvaluationComparison(
        task=b_task,
        summary_deltas=summary_deltas,
        class_deltas=class_deltas,
        timing_deltas=timing_deltas,
        calibration_deltas=calibration_deltas,
        baseline_samples=baseline.get("samples"),
        candidate_samples=candidate.get("samples"),
        baseline_model=baseline.get("model"),
        candidate_model=candidate.get("model"),
        baseline_split=baseline.get("split"),
        candidate_split=candidate.get("split"),
    )


__all__ = [
    "ComparisonError",
    "EvaluationComparison",
    "MetricDelta",
    "compare_evaluations",
    "compute_metric_delta",
]
