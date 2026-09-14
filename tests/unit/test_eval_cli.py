import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


def _write_rgb(root: Path, name: str = "00001_FV.png") -> None:
    image_dir = root / "rgb_images"
    image_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / name), np.zeros((3, 4, 3), dtype=np.uint8))


def _write_mask(root: Path, name: str = "00001_FV.png") -> None:
    mask_dir = root / "semantic_annotations/gtLabels"
    mask_dir.mkdir(parents=True)
    assert cv2.imwrite(str(mask_dir / name), np.ones((3, 4), dtype=np.uint8))


def _write_detection(root: Path, name: str = "00001_FV.txt") -> None:
    detection_dir = root / "detection_annotations"
    detection_dir.mkdir(parents=True)
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
