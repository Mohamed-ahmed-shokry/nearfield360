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


def _strip_cameras(root: Path, frame_id: str, cameras: tuple[str, ...]) -> None:
    for cam in cameras:
        stem = f"{frame_id}_{cam}"
        (root / f"rgb_images/{stem}.png").unlink(missing_ok=True)
        (root / f"calibration_data/{stem}.json").unlink(missing_ok=True)
        (root / f"semantic_annotations/gtLabels/{stem}.png").unlink(missing_ok=True)
        (root / f"detection_annotations/{stem}.txt").unlink(missing_ok=True)


def test_pipeline_help() -> None:
    result = runner.invoke(app, ["pipeline", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_pipeline_run_help() -> None:
    result = runner.invoke(app, ["pipeline", "run", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.stdout
    assert "--health-aware" in result.stdout
    assert "--samples" in result.stdout


def test_pipeline_run_writes_timed_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    _write_four_camera_frame(dataset, "00002")
    output = tmp_path / "reports/pipeline.json"
    png = tmp_path / "occupancy.png"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--png",
            str(png),
        ],
    )

    assert result.exit_code == 0
    assert "processed frame 00001" in result.stdout
    assert "processed frame 00002" in result.stdout
    assert "Pipeline complete: 2 frames" in result.stdout

    payload = read_json(output)
    assert payload["samples"]["evaluated"] == 2
    assert payload["samples"]["camera"] == "all"
    assert payload["samples"]["per_camera"] == {"FV": 2, "RV": 2, "MVL": 2, "MVR": 2}
    assert payload["environment"]["nearfield360_version"]
    assert payload["evidence"]["observed_cells"] > 0
    assert len(payload["risk"]) == 6
    assert len(payload["zones"]) == 6
    assert "summary" in payload
    assert "tracks" in payload
    assert "forecasts" in payload

    timings = payload["timings"]
    for key in (
        "discovery_ms",
        "perception_ms",
        "occupancy_ms",
        "tracking_ms",
        "risk_ms",
        "forecast_ms",
        "total_ms",
        "frames_per_second",
    ):
        assert key in timings
        assert timings[key] >= 0.0

    assert png.is_file()
    assert json.dumps(payload)


def test_pipeline_run_health_aware(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    output = tmp_path / "health_pipeline.json"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--health-aware",
        ],
    )

    assert result.exit_code == 0
    payload = read_json(output)
    assert payload["samples"]["health_aware"] is True
    assert "health" in payload
    assert len(payload["health"]) == 4


def test_pipeline_run_refuses_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    output = tmp_path / "report.json"

    first = runner.invoke(
        app,
        ["pipeline", "run", "--root", str(dataset), "--output", str(output)],
    )
    second = runner.invoke(
        app,
        ["pipeline", "run", "--root", str(dataset), "--output", str(output)],
    )
    third = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--overwrite",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Artifact error" in second.stderr
    assert third.exit_code == 0


def test_pipeline_run_fails_without_complete_frames(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    # Only FV camera present for frame 00001 — incomplete surround.
    _write_four_camera_frame(dataset, "00001")
    _strip_cameras(dataset, "00001", ("RV", "MVL", "MVR"))

    result = runner.invoke(
        app,
        ["pipeline", "run", "--root", str(dataset), "--output", str(tmp_path / "out.json")],
    )

    assert result.exit_code == 1
    assert "No frames with all four cameras" in result.stderr


def test_pipeline_run_skips_incomplete_frames(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    # Incomplete frame 00001 (only FV remains), complete 00002.
    _write_four_camera_frame(dataset, "00001")
    _strip_cameras(dataset, "00001", ("RV", "MVL", "MVR"))
    _write_four_camera_frame(dataset, "00002")

    output = tmp_path / "partial.json"
    result = runner.invoke(
        app,
        ["pipeline", "run", "--root", str(dataset), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert "Skipping frame 00001: missing cameras" in result.stderr
    assert "processed frame 00002" in result.stdout
    payload = read_json(output)
    assert payload["samples"]["evaluated"] == 1


def test_pipeline_run_limits_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")
    _write_four_camera_frame(dataset, "00002")
    output = tmp_path / "limited.json"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--samples",
            "5",
        ],
    )

    assert result.exit_code == 1
    assert "only 2 complete multi-camera frames" in result.stderr


def test_pipeline_run_requires_detections(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001", with_detections=False)

    result = runner.invoke(
        app,
        ["pipeline", "run", "--root", str(dataset), "--output", str(tmp_path / "out.json")],
    )

    assert result.exit_code == 1
    assert "No frames with all four cameras" in result.stderr
