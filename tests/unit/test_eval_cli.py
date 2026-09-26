import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.perception.inference.test_utils import (
    create_dummy_detection_onnx,
    create_dummy_segmentation_onnx,
)
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


@pytest.fixture
def dummy_seg_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "models" / "seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)
    return model_path


@pytest.fixture
def dummy_det_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "models" / "det.onnx"
    create_dummy_detection_onnx(model_path, num_classes=5, num_boxes=3, height=32, width=32)
    return model_path


def _write_rgb(root: Path, name: str = "00001_FV.png") -> None:
    image_dir = root / "rgb_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(image_dir / name), np.zeros((3, 4, 3), dtype=np.uint8))


def _write_mask(root: Path, name: str = "00001_FV.png") -> None:
    mask_dir = root / "semantic_annotations/gtLabels"
    mask_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(mask_dir / name), np.ones((3, 4), dtype=np.uint8))


def _write_detection(root: Path, name: str = "00001_FV.txt") -> None:
    detection_dir = root / "detection_annotations"
    detection_dir.mkdir(parents=True, exist_ok=True)
    (detection_dir / name).write_text("vehicles,0,0,0,4,3\n", encoding="utf-8")


def _write_prediction_mask(predictions: Path) -> None:
    predictions.mkdir(parents=True)
    assert cv2.imwrite(str(predictions / "00001_FV.png"), np.ones((3, 4), dtype=np.uint8))


def _write_prediction_detection(predictions: Path) -> None:
    predictions.mkdir(parents=True)
    (predictions / "00001_FV.txt").write_text("vehicles,0,0,0,4,3,0.95\n", encoding="utf-8")


def test_eval_segmentation_writes_reproducible_report(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_mask(predictions)
    output = tmp_path / "reports/segmentation.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Evaluated 1/1 samples" in result.stdout
    payload = read_json(output)
    assert payload["metrics"]["mean_iou"] == 1.0
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}
    assert payload["metrics"]["classes"]["road"]["iou"] == 1.0
    assert payload["environment"]["nearfield360_version"]
    assert payload["config"]["runtime"]["seed"] == 42


def test_eval_segmentation_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_mask(predictions)
    output = tmp_path / "report.json"

    first = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )
    second = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Artifact error" in second.stderr


def test_eval_segmentation_reports_missing_predictions(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "Missing predictions for 1 sample(s): 00001_FV" in result.stderr
    assert not output.exists()


def test_eval_segmentation_requires_semantic_masks(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "r.json"),
        ],
    )

    assert result.exit_code == 1
    assert "no semantic masks" in result.stderr


def test_eval_detection_writes_report_with_map(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)
    output = tmp_path / "detection.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Evaluated 1/1 samples; mAP=1.000" in result.stdout
    payload = read_json(output)
    assert payload["metrics"]["mean_average_precision"] == 1.0
    assert payload["metrics"]["classes"]["vehicles"]["average_precision"] == 1.0
    assert payload["metrics"]["iou_threshold"] == 0.5
    assert json.dumps(payload)  # fully JSON-serializable


def test_eval_detection_rejects_incomplete_predictions(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "d.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Missing predictions" in result.stderr


def test_eval_segmentation_limit_bounds_annotated_samples(tmp_path: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    _write_rgb(tmp_path, "00002_FV.png")
    _write_mask(tmp_path, "00002_FV.png")
    predictions = tmp_path / "predictions"
    _write_prediction_mask(predictions)  # only 00001_FV present; 00002_FV missing

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "limited.json"),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = read_json(tmp_path / "limited.json")
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}


def test_eval_detection_limit_bounds_annotated_samples(tmp_path: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")
    _write_rgb(tmp_path, "00002_FV.png")
    _write_detection(tmp_path, "00002_FV.txt")
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)  # only 00001_FV present; 00002_FV missing

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "limited.json"),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = read_json(tmp_path / "limited.json")
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}


def test_eval_segmentation_limit_zero_evaluates_all(tmp_path: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    _write_rgb(tmp_path, "00002_FV.png")
    _write_mask(tmp_path, "00002_FV.png")
    predictions = tmp_path / "predictions"
    _write_prediction_mask(predictions)  # only 00001_FV present; 00002_FV missing

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "all.json"),
            "--limit",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "Missing predictions for 1 sample(s): 00002_FV" in result.stderr


def test_eval_segmentation_model_scores_annotations(tmp_path: Path, dummy_seg_model: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    output = tmp_path / "model_report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluated 1/1 samples" in result.stdout
    payload = read_json(output)
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}
    assert payload["model"] == {
        "path": str(dummy_seg_model),
        "backend": "opencv",
        "device": "cpu",
    }
    assert 0.0 <= payload["metrics"]["mean_iou"] <= 1.0
    assert json.dumps(payload)  # fully JSON-serializable


def test_eval_segmentation_model_limit_bounds_samples(
    tmp_path: Path, dummy_seg_model: Path
) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    _write_rgb(tmp_path, "00002_FV.png")
    _write_mask(tmp_path, "00002_FV.png")

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(tmp_path / "limited.json"),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(tmp_path / "limited.json")
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}


def test_eval_segmentation_rejects_predictions_with_model(
    tmp_path: Path, dummy_seg_model: Path
) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "exactly one of --predictions or --model" in result.stderr
    assert not output.exists()


def test_eval_segmentation_requires_predictions_or_model(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "exactly one of --predictions or --model" in result.stderr
    assert not output.exists()


def test_eval_segmentation_model_file_must_exist(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(tmp_path / "missing.onnx"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 2


def test_eval_segmentation_model_missing_onnxruntime(tmp_path: Path, dummy_seg_model: Path) -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("onnxruntime is installed; missing-package path not testable")

    _write_rgb(tmp_path)
    _write_mask(tmp_path)

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(tmp_path / "report.json"),
            "--backend",
            "onnxruntime",
        ],
    )

    assert result.exit_code == 1
    assert "onnxruntime is not installed" in result.stderr


def test_eval_detection_model_scores_annotations(tmp_path: Path, dummy_det_model: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")
    output = tmp_path / "model_report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_det_model),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluated 1/1 samples" in result.stdout
    assert "Timing: 1 samples" in result.stdout
    payload = read_json(output)
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}
    assert payload["model"] == {
        "path": str(dummy_det_model),
        "backend": "opencv",
        "device": "cpu",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
    }
    assert payload["timing"]["samples"] == 1
    assert payload["timing"]["min_ms"] <= payload["timing"]["p95_ms"]
    assert payload["timing"]["p99_ms"] <= payload["timing"]["max_ms"]
    assert 0.0 <= payload["metrics"]["mean_average_precision"] <= 1.0
    assert json.dumps(payload)  # fully JSON-serializable


def test_eval_detection_model_iou_threshold_matches_centered_box(
    tmp_path: Path, dummy_det_model: Path
) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_det_model),
            "--output",
            str(tmp_path / "report.json"),
            "--iou-threshold",
            "0.01",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(tmp_path / "report.json")
    assert payload["metrics"]["mean_average_precision"] == 1.0


def test_eval_detection_model_threshold_overrides(tmp_path: Path, dummy_det_model: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_det_model),
            "--output",
            str(tmp_path / "report.json"),
            "--confidence-threshold",
            "0.3",
            "--nms-threshold",
            "0.6",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(tmp_path / "report.json")
    assert payload["model"]["confidence_threshold"] == 0.3
    assert payload["model"]["nms_threshold"] == 0.6


def test_eval_detection_model_limit_bounds_samples(tmp_path: Path, dummy_det_model: Path) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")
    _write_rgb(tmp_path, "00002_FV.png")
    _write_detection(tmp_path, "00002_FV.txt")

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_det_model),
            "--output",
            str(tmp_path / "limited.json"),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(tmp_path / "limited.json")
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}


def test_eval_detection_rejects_predictions_with_model(
    tmp_path: Path, dummy_det_model: Path
) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--model",
            str(dummy_det_model),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "exactly one of --predictions or --model" in result.stderr
    assert not output.exists()


def test_eval_detection_requires_predictions_or_model(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "exactly one of --predictions or --model" in result.stderr
    assert not output.exists()


def test_eval_detection_empty_prediction_file_scores_zero(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "00001_FV.txt").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(tmp_path / "report.json")
    assert payload["metrics"]["mean_average_precision"] == 0.0
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}


def test_eval_segmentation_model_save_predictions_round_trip(
    tmp_path: Path, dummy_seg_model: Path
) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    saved = tmp_path / "saved_predictions"

    model_run = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(tmp_path / "model.json"),
            "--save-predictions",
            str(saved),
        ],
    )
    assert model_run.exit_code == 0, model_run.output
    assert (saved / "00001_FV.png").is_file()

    file_run = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(saved),
            "--output",
            str(tmp_path / "file.json"),
        ],
    )
    assert file_run.exit_code == 0, file_run.output
    model_payload = read_json(tmp_path / "model.json")
    file_payload = read_json(tmp_path / "file.json")
    assert model_payload["metrics"] == file_payload["metrics"]


def test_eval_detection_model_save_predictions_round_trip(
    tmp_path: Path, dummy_det_model: Path
) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_detection(tmp_path, "00001_FV.txt")
    saved = tmp_path / "saved_predictions"

    model_run = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_det_model),
            "--output",
            str(tmp_path / "model.json"),
            "--save-predictions",
            str(saved),
        ],
    )
    assert model_run.exit_code == 0, model_run.output
    assert (saved / "00001_FV.txt").is_file()

    file_run = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(saved),
            "--output",
            str(tmp_path / "file.json"),
        ],
    )
    assert file_run.exit_code == 0, file_run.output
    model_payload = read_json(tmp_path / "model.json")
    file_payload = read_json(tmp_path / "file.json")
    assert model_payload["metrics"] == file_payload["metrics"]


def test_eval_save_predictions_requires_model(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "report.json"),
            "--save-predictions",
            str(tmp_path / "saved"),
        ],
    )

    assert result.exit_code == 1
    assert "--save-predictions requires --model" in result.stderr


def test_eval_segmentation_model_report_includes_timing(
    tmp_path: Path, dummy_seg_model: Path
) -> None:
    _write_rgb(tmp_path, "00001_FV.png")
    _write_mask(tmp_path, "00001_FV.png")
    output = tmp_path / "model_report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--model",
            str(dummy_seg_model),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Timing: 1 samples" in result.stdout
    payload = read_json(output)
    timing = payload["timing"]
    assert timing["samples"] == 1
    assert timing["total_ms"] >= 0.0
    assert timing["mean_ms"] >= 0.0
    assert (
        timing["min_ms"]
        <= timing["p50_ms"]
        <= timing["p95_ms"]
        <= timing["p99_ms"]
        <= timing["max_ms"]
    )
    assert timing["samples_per_second"] >= 0.0


def test_eval_file_mode_report_has_null_model_and_timing(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_mask(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_mask(predictions)
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "segmentation",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(output)
    assert payload["model"] is None
    assert payload["timing"] is None


def test_eval_detection_confidence_thresholds_add_analysis(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)
    output = tmp_path / "confidence.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--confidence-thresholds",
            "0.7,0.3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(output)
    analysis = payload["metrics"]["confidence_analysis"]
    assert analysis["iou_threshold"] == 0.5
    assert analysis["targets"] == 1
    assert [point["confidence"] for point in analysis["thresholds"]] == [0.3, 0.7]
    for point in analysis["thresholds"]:
        assert point["predictions"] == 1
        assert point["true_positives"] == 1
        assert point["precision"] == 1.0
        assert point["recall"] == 1.0
        assert point["f1"] == 1.0
    assert len(analysis["pr_curves"]["vehicles"]) == 101
    assert analysis["pr_curves"]["person"] is None
    assert json.dumps(payload)  # fully JSON-serializable


def test_eval_detection_report_omits_confidence_analysis_by_default(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)
    output = tmp_path / "plain.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = read_json(output)
    assert "confidence_analysis" not in payload["metrics"]


def test_eval_detection_confidence_thresholds_dedupe_and_sort(tmp_path: Path) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)
    output = tmp_path / "dedup.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--confidence-thresholds",
            "0.9,0.1,0.9",
        ],
    )

    assert result.exit_code == 0, result.output
    analysis = read_json(output)["metrics"]["confidence_analysis"]
    assert [point["confidence"] for point in analysis["thresholds"]] == [0.1, 0.9]


@pytest.mark.parametrize("raw", ["abc", "1.5", "", "0.5,,0.7", "nan"])
def test_eval_detection_rejects_invalid_confidence_thresholds(tmp_path: Path, raw: str) -> None:
    _write_rgb(tmp_path)
    _write_detection(tmp_path)
    predictions = tmp_path / "predictions"
    _write_prediction_detection(predictions)
    output = tmp_path / "invalid.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "detection",
            "--root",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--confidence-thresholds",
            raw,
        ],
    )

    assert result.exit_code == 2
    assert not output.exists()
