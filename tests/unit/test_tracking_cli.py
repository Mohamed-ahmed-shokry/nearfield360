from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()

_DOWN_QUATERNION = [1.0, 0.0, 0.0, 0.0]


def _setup_dataset(
    root: Path,
    frames: list[tuple[str, str]],  # (frame_id, camera)
    with_detections: bool = True,
) -> None:
    (root / "rgb_images").mkdir(parents=True, exist_ok=True)
    (root / "calibration_data").mkdir(parents=True, exist_ok=True)
    if with_detections:
        (root / "detection_annotations").mkdir(parents=True, exist_ok=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    for frame_id, cam in frames:
        stem = f"{frame_id}_{cam}"
        img_path = root / f"rgb_images/{stem}.png"
        cv2.imwrite(str(img_path), img)

        calib_path = root / f"calibration_data/{stem}.json"
        calib_data = {
            "extrinsic": {
                "quaternion": _DOWN_QUATERNION,
                "translation": [0.0, 0.0, 1.5],
            },
            "intrinsic": {
                "aspect_ratio": 1.0,
                "cx_offset": 0.0,
                "cy_offset": 0.0,
                "height": 100,
                "k1": 50.0,
                "k2": 0.0,
                "k3": 0.0,
                "k4": 0.0,
                "model": "radial_poly",
                "poly_order": 4,
                "width": 100,
            },
            "name": cam,
        }
        calib_path.write_text(json.dumps(calib_data), encoding="utf-8")

        if with_detections:
            det_path = root / f"detection_annotations/{stem}.txt"
            # detection format: class_name,class_id,x_min,y_min,x_max,y_max
            det_path.write_text(
                "vehicles,0,40,40,60,80\nperson,1,10,20,30,50\n",
                encoding="utf-8",
            )


def test_track_help() -> None:
    result = runner.invoke(app, ["track", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_track_run_help() -> None:
    result = runner.invoke(app, ["track", "run", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.stdout
    assert "--all-cameras" in result.stdout
    assert "--png" in result.stdout
    assert "--backend" in result.stdout


def test_track_run_single_camera(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _setup_dataset(dataset_dir, [("00001", "FV"), ("00002", "FV")])

    output = tmp_path / "tracking_report.json"
    png_path = tmp_path / "tracking_overlay.png"

    result = runner.invoke(
        app,
        [
            "track",
            "run",
            "--root",
            str(dataset_dir),
            "--camera",
            "FV",
            "--output",
            str(output),
            "--png",
            str(png_path),
            "--max-frames",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "Tracking complete: 2 frames evaluated" in result.stdout
    assert output.is_file()
    assert png_path.is_file()

    payload = read_json(output)
    assert payload["camera_mode"] == "FV"
    assert payload["frames_evaluated"] == 2
    assert "summary" in payload
    assert "total_active_tracks" in payload["summary"]
    assert "tracks" in payload
    assert "forecasts" in payload
    assert len(payload["tracks"]) > 0


def test_track_run_all_cameras(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _setup_dataset(
        dataset_dir,
        [
            ("00001", "FV"),
            ("00001", "RV"),
            ("00001", "MVL"),
            ("00001", "MVR"),
            ("00002", "FV"),
            ("00002", "RV"),
            ("00002", "MVL"),
            ("00002", "MVR"),
        ],
    )

    output = tmp_path / "multi_cam_tracking.json"

    result = runner.invoke(
        app,
        [
            "track",
            "run",
            "--root",
            str(dataset_dir),
            "--all-cameras",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert "Tracking complete: 2 frames evaluated" in result.stdout
    payload = read_json(output)
    assert payload["camera_mode"] == "all"
    assert payload["frames_evaluated"] == 2


def test_track_run_overwrite_protection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _setup_dataset(dataset_dir, [("00001", "FV")])
    output = tmp_path / "tracking_report.json"

    result1 = runner.invoke(
        app,
        ["track", "run", "--root", str(dataset_dir), "--output", str(output)],
    )
    assert result1.exit_code == 0

    # Refuses overwrite
    result2 = runner.invoke(
        app,
        ["track", "run", "--root", str(dataset_dir), "--output", str(output)],
    )
    assert result2.exit_code == 1
    assert "Artifact error" in result2.stderr

    # Succeeds with --overwrite
    result3 = runner.invoke(
        app,
        [
            "track",
            "run",
            "--root",
            str(dataset_dir),
            "--output",
            str(output),
            "--overwrite",
        ],
    )
    assert result3.exit_code == 0


def test_track_run_no_matching_samples(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _setup_dataset(dataset_dir, [("00001", "RV")])
    output = tmp_path / "tracking_report.json"

    # Search for FV when only RV exists
    result = runner.invoke(
        app,
        [
            "track",
            "run",
            "--root",
            str(dataset_dir),
            "--camera",
            "FV",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "No matching samples found" in result.stderr


def test_track_run_with_live_detection_model(tmp_path: Path) -> None:
    from nearfield360.perception.inference.test_utils import create_dummy_detection_onnx

    model_path = tmp_path / "models" / "det.onnx"
    create_dummy_detection_onnx(model_path, num_classes=5, num_boxes=2, height=32, width=32)

    dataset_dir = tmp_path / "dataset"
    _setup_dataset(dataset_dir, [("00001", "FV"), ("00002", "FV")], with_detections=False)

    output = tmp_path / "tracking_report.json"
    result = runner.invoke(
        app,
        [
            "track",
            "run",
            "--root",
            str(dataset_dir),
            "--camera",
            "FV",
            "--output",
            str(output),
            "--model",
            str(model_path),
            "--backend",
            "opencv",
        ],
    )
    assert result.exit_code == 0
    payload = read_json(output)
    assert payload["frames_evaluated"] == 2
    assert payload["model"] == str(model_path)
    assert "summary" in payload
