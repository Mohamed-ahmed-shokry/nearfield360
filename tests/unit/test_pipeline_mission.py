"""Integration tests for four-camera pipeline runner with --mission flag."""

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
_CAMERAS = ("FV", "RV", "MVL", "MVR")


def _write_four_camera_frame(
    root: Path,
    frame_id: str,
    *,
    with_detections: bool = True,
    with_semantic: bool = True,
) -> None:
    (root / "rgb_images").mkdir(parents=True, exist_ok=True)
    (root / "calibration_data").mkdir(parents=True, exist_ok=True)
    if with_semantic:
        (root / "semantic_annotations/gtLabels").mkdir(parents=True, exist_ok=True)
    if with_detections:
        (root / "detection_annotations").mkdir(parents=True, exist_ok=True)

    img = np.zeros((2, 3, 3), dtype=np.uint8)
    mask = np.array([[1, 1, 1], [6, 6, 6]], dtype=np.uint8)

    for cam in _CAMERAS:
        stem = f"{frame_id}_{cam}"
        cv2.imwrite(str(root / f"rgb_images/{stem}.png"), img)
        if with_semantic:
            cv2.imwrite(str(root / f"semantic_annotations/gtLabels/{stem}.png"), mask)
        calib = root / f"calibration_data/{stem}.json"
        calib.write_text(
            json.dumps(
                {
                    "extrinsic": {
                        "quaternion": _DOWN_QUATERNION,
                        "translation": [0.0, 0.0, 1.0],
                    },
                    "intrinsic": {
                        "aspect_ratio": 1.0,
                        "cx_offset": 0.0,
                        "cy_offset": 0.0,
                        "height": 2,
                        "k1": 100.0,
                        "k2": 0.0,
                        "k3": 0.0,
                        "k4": 0.0,
                        "model": "radial_poly",
                        "poly_order": 4,
                        "width": 3,
                    },
                    "name": cam,
                }
            ),
            encoding="utf-8",
        )
        if with_detections:
            (root / f"detection_annotations/{stem}.txt").write_text(
                "vehicles,0,40,40,60,80\n",
                encoding="utf-8",
            )


def test_pipeline_run_with_mission(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    output = tmp_path / "pipeline_mission_report.json"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--mission",
        ],
    )

    assert result.exit_code == 0
    assert "Pipeline complete" in result.stdout
    payload = read_json(output)
    assert payload["samples"]["mission"] is True
    assert payload["samples"]["plan_parking"] is True
    assert payload["samples"]["simulate_control"] is True
    assert "slots" in payload
    assert "plan" in payload
    if "mission" in payload:
        assert "mission_id" in payload["mission"]
        assert "final_state" in payload["mission"]
        assert "mission_ms" in payload["timings"]


def test_pipeline_run_with_mission_and_png(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    output = tmp_path / "pipeline_mission_report.json"
    png_path = tmp_path / "pipeline_mission.png"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--mission",
            "--png",
            str(png_path),
        ],
    )

    assert result.exit_code == 0
    assert png_path.is_file()
    assert png_path.stat().st_size > 0


def test_pipeline_run_with_mission_and_health_aware(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    output = tmp_path / "pipeline_mission_health_report.json"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--mission",
            "--health-aware",
        ],
    )

    assert result.exit_code == 0
    payload = read_json(output)
    assert payload["samples"]["mission"] is True
    assert payload["samples"]["health_aware"] is True
    assert "health" in payload
