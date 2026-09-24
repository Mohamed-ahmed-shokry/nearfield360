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
    payload = read_json(output)
    assert payload["samples"] == {"expected": 1, "evaluated": 1, "missing_predictions": []}
    assert payload["model"] == {
        "path": str(dummy_det_model),
        "backend": "opencv",
        "device": "cpu",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
    }
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
