from __future__ import annotations

from typing import Any

import pytest

from nearfield360.perception.comparison import compare_evaluations
from nearfield360.perception.dashboard import (
    DashboardError,
    generate_evaluation_html_dashboard,
)


@pytest.fixture
def segmentation_report() -> dict[str, Any]:
    return {
        "environment": {
            "nearfield360_version": "0.1.0",
            "python_version": "3.12.1",
            "platform": "TestOS",
        },
        "model": {
            "path": "models/seg.onnx",
            "backend": "opencv",
            "device": "cpu",
        },
        "split": {
            "split": "val",
            "manifest": "splits.json",
        },
        "samples": {
            "expected": 10,
            "evaluated": 10,
            "missing_predictions": [],
        },
        "timing": {
            "samples": 10,
            "total_ms": 250.0,
            "mean_ms": 25.0,
            "p50_ms": 24.5,
            "p95_ms": 28.2,
            "p99_ms": 29.8,
            "min_ms": 20.1,
            "max_ms": 30.5,
            "samples_per_second": 40.0,
        },
        "metrics": {
            "mean_iou": 0.725,
            "pixel_accuracy": 0.912,
            "image_count": 10,
            "pixel_count": 100000,
            "classes": {
                "road": {"iou": 0.88, "pixels": 50000},
                "lanemarks": {"iou": 0.65, "pixels": 15000},
                "curb": {"iou": 0.42, "pixels": 5000},
            },
            "confidence_analysis": {
                "num_bins": 5,
                "pixel_count": 100000,
                "mean_confidence": 0.85,
                "ece": 0.024,
                "bins": [
                    {
                        "bin_index": 0,
                        "bin_lower": 0.0,
                        "bin_upper": 0.2,
                        "pixels": 5000,
                        "accuracy": 0.15,
                        "mean_confidence": 0.12,
                    },
                    {
                        "bin_index": 1,
                        "bin_lower": 0.2,
                        "bin_upper": 0.4,
                        "pixels": 10000,
                        "accuracy": 0.32,
                        "mean_confidence": 0.31,
                    },
                    {
                        "bin_index": 2,
                        "bin_lower": 0.4,
                        "bin_upper": 0.6,
                        "pixels": 25000,
                        "accuracy": 0.55,
                        "mean_confidence": 0.52,
                    },
                    {
                        "bin_index": 3,
                        "bin_lower": 0.6,
                        "bin_upper": 0.8,
                        "pixels": 30000,
                        "accuracy": 0.74,
                        "mean_confidence": 0.71,
                    },
                    {
                        "bin_index": 4,
                        "bin_lower": 0.8,
                        "bin_upper": 1.0,
                        "pixels": 30000,
                        "accuracy": 0.94,
                        "mean_confidence": 0.92,
                    },
                ],
            },
        },
    }


@pytest.fixture
def detection_report() -> dict[str, Any]:
    return {
        "environment": {
            "nearfield360_version": "0.1.0",
            "python_version": "3.12.1",
            "platform": "TestOS",
        },
        "model": {
            "path": "models/det.onnx",
            "backend": "onnxruntime",
            "device": "cuda",
        },
        "samples": {
            "expected": 20,
            "evaluated": 20,
            "missing_predictions": [],
        },
        "timing": {
            "samples": 20,
            "total_ms": 300.0,
            "mean_ms": 15.0,
            "p50_ms": 14.8,
            "p95_ms": 18.0,
            "p99_ms": 19.5,
            "min_ms": 12.0,
            "max_ms": 20.0,
            "samples_per_second": 66.7,
        },
        "metrics": {
            "mean_average_precision": 0.642,
            "iou_threshold": 0.5,
            "image_count": 20,
            "classes": {
                "vehicles": {"average_precision": 0.82, "predictions": 45, "targets": 40},
                "pedestrians": {"average_precision": 0.51, "predictions": 22, "targets": 25},
                "cyclists": {"average_precision": 0.40, "predictions": 10, "targets": 12},
            },
            "confidence_analysis": {
                "iou_threshold": 0.5,
                "targets": 77,
                "thresholds": [
                    {
                        "confidence": 0.3,
                        "predictions": 100,
                        "true_positives": 65,
                        "precision": 0.65,
                        "recall": 0.84,
                        "f1": 0.73,
                    },
                    {
                        "confidence": 0.5,
                        "predictions": 80,
                        "true_positives": 58,
                        "precision": 0.725,
                        "recall": 0.75,
                        "f1": 0.74,
                    },
                    {
                        "confidence": 0.8,
                        "predictions": 50,
                        "true_positives": 45,
                        "precision": 0.90,
                        "recall": 0.58,
                        "f1": 0.71,
                    },
                ],
                "pr_curves": {
                    "vehicles": [
                        {"recall": 0.0, "precision": 1.0},
                        {"recall": 0.5, "precision": 0.90},
                        {"recall": 1.0, "precision": 0.75},
                    ],
                    "pedestrians": [
                        {"recall": 0.0, "precision": 0.85},
                        {"recall": 0.5, "precision": 0.60},
                        {"recall": 1.0, "precision": 0.35},
                    ],
                },
            },
        },
    }


def test_generate_segmentation_dashboard(segmentation_report: dict[str, Any]) -> None:
    html_output = generate_evaluation_html_dashboard(segmentation_report)

    assert "<!DOCTYPE html>" in html_output
    assert "Semantic Segmentation Evaluation Report" in html_output
    assert "Mean IoU" in html_output
    assert "72.50%" in html_output
    assert "Pixel Accuracy" in html_output
    assert "91.20%" in html_output
    assert "Expected Calibration Error" in html_output
    assert "0.0240" in html_output
    assert "Mean Confidence" in html_output
    assert "85.00%" in html_output
    assert "road" in html_output
    assert "lanemarks" in html_output
    assert "curb" in html_output
    assert "<svg" in html_output
    assert "Segmentation Reliability Diagram" in html_output
    assert "Inference Latency Profile" in html_output

    # Zero external CDN / remote asset verification
    assert "<script" not in html_output
    assert 'href="http' not in html_output
    assert 'src="http' not in html_output


def test_generate_detection_dashboard(detection_report: dict[str, Any]) -> None:
    html_output = generate_evaluation_html_dashboard(detection_report, title="Custom Det Dashboard")

    assert "<!DOCTYPE html>" in html_output
    assert "Custom Det Dashboard" in html_output
    assert "DETECTION" in html_output
    assert "mAP @ IoU 0.5" in html_output
    assert "64.20%" in html_output
    assert "vehicles" in html_output
    assert "pedestrians" in html_output
    assert "Detection Precision-Recall Curves" in html_output
    assert "Detection Operating Points" in html_output
    assert "Inference Latency Profile" in html_output
    assert "66.7 FPS" in html_output

    # Zero external CDN / remote asset verification
    assert "<script" not in html_output
    assert 'href="http' not in html_output
    assert 'src="http' not in html_output


def test_generate_comparative_dashboard(segmentation_report: dict[str, Any]) -> None:
    candidate_report = {
        "environment": segmentation_report["environment"],
        "metrics": {
            "mean_iou": 0.765,
            "pixel_accuracy": 0.930,
            "classes": {
                "road": {"iou": 0.90, "pixels": 50000},
                "lanemarks": {"iou": 0.70, "pixels": 15000},
                "curb": {"iou": 0.48, "pixels": 5000},
            },
        },
    }

    html_output = generate_evaluation_html_dashboard(
        candidate_report, baseline_report=segmentation_report
    )

    assert "Comparative Semantic Segmentation Evaluation Report" in html_output
    assert "Summary Metric Deltas (Baseline vs Candidate)" in html_output
    assert "Class Performance Deltas (IoU)" in html_output
    assert "+0.0400" in html_output  # mIoU delta: 0.765 - 0.725
    assert "+5.52%" in html_output


def test_generate_dashboard_with_regression_gates(segmentation_report: dict[str, Any]) -> None:
    gates = [
        {
            "gate": "fail_under_miou_delta",
            "threshold": 0.05,
            "actual": 0.04,
            "status": "fail",
            "detail": "mIoU delta +0.0400 (min required: +0.0500)",
        },
        {
            "gate": "fail_over_latency_ratio",
            "threshold": 1.2,
            "actual": 1.05,
            "status": "pass",
            "detail": "Latency ratio 1.050x (max: 1.200x)",
        },
    ]

    candidate_report = {
        "metrics": {
            "mean_iou": 0.765,
            "classes": {"road": {"iou": 0.90}},
        },
    }

    html_output = generate_evaluation_html_dashboard(
        candidate_report,
        baseline_report=segmentation_report,
        gate_results=gates,
    )

    assert "Regression Gate Verification" in html_output
    assert "GATE REGRESSION DETECTED" in html_output
    assert "[FAIL]" in html_output
    assert "[PASS]" in html_output
    assert "fail_under_miou_delta" in html_output
    assert "fail_over_latency_ratio" in html_output


def test_generate_dashboard_from_comparison_artifact_directly(
    segmentation_report: dict[str, Any],
) -> None:
    candidate_report = {
        "environment": segmentation_report["environment"],
        "metrics": {
            "mean_iou": 0.750,
            "pixel_accuracy": 0.920,
            "classes": {"road": {"iou": 0.89}},
        },
    }
    cmp_obj = compare_evaluations(segmentation_report, candidate_report)
    cmp_dict = cmp_obj.as_dict()

    html_output = generate_evaluation_html_dashboard(cmp_dict)
    assert "<!DOCTYPE html>" in html_output
    assert "Comparative Semantic Segmentation Evaluation Report" in html_output
    assert "Summary Metric Deltas" in html_output


def test_generate_dashboard_rejects_malformed_inputs(
    segmentation_report: dict[str, Any], detection_report: dict[str, Any]
) -> None:
    with pytest.raises(DashboardError, match="Report artifact must be a mapping"):
        generate_evaluation_html_dashboard(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(DashboardError, match="Report missing 'metrics' section"):
        generate_evaluation_html_dashboard({"empty": True})

    with pytest.raises(DashboardError, match="Unrecognized evaluation report format"):
        generate_evaluation_html_dashboard({"metrics": {"unknown_score": 0.5}})

    with pytest.raises(DashboardError, match="Comparative dashboard error"):
        generate_evaluation_html_dashboard(detection_report, baseline_report=segmentation_report)
