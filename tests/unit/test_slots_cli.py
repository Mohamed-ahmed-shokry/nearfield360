"""Unit and integration tests for the nearfield360 slots CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()

_DOWN_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def _write_slots_dataset(
    root: Path,
    frame_ids: list[str] | None = None,
    cameras: list[str] | None = None,
    with_markings: bool = True,
) -> None:
    if frame_ids is None:
        frame_ids = ["00001"]
    if cameras is None:
        cameras = ["FV"]

    (root / "rgb_images").mkdir(parents=True, exist_ok=True)
    (root / "semantic_annotations/gtLabels").mkdir(parents=True, exist_ok=True)
    (root / "calibration_data").mkdir(parents=True, exist_ok=True)
    (root / "box_data").mkdir(parents=True, exist_ok=True)

    for frame_id in frame_ids:
        for cam in cameras:
            stem = f"{frame_id}_{cam}"
            # RGB dummy
            cv2.imwrite(str(root / f"rgb_images/{stem}.png"), np.zeros((4, 4, 3), dtype=np.uint8))

            # Semantic mask with road=0, lanemarks=1, curb=2
            mask = np.zeros((4, 4), dtype=np.uint8)
            if with_markings:
                mask[1:3, 1:3] = 1  # lanemarks
            cv2.imwrite(str(root / f"semantic_annotations/gtLabels/{stem}.png"), mask)

            # Calibration pointing downward to ground z=0
            calib = root / f"calibration_data/{stem}.json"
            calib.write_text(
                json.dumps(
                    {
                        "extrinsic": {
                            "quaternion": list(_DOWN_QUATERNION),
                            "translation": [0.0, 0.0, 1.0],
                        },
                        "intrinsic": {
                            "aspect_ratio": 1.0,
                            "cx_offset": 0.0,
                            "cy_offset": 0.0,
                            "height": 4,
                            "width": 4,
                            "k1": 100.0,
                            "k2": 0.0,
                            "k3": 0.0,
                            "k4": 0.0,
                            "model": "radial_poly",
                            "poly_order": 4,
                        },
                        "name": cam,
                    }
                ),
                encoding="utf-8",
            )

            # 2D detection box dummy
            boxes = root / f"box_data/{stem}.json"
            boxes.write_text(
                json.dumps(
                    [
                        {
                            "2d_box": [1, 1, 3, 3],
                            "label": "vehicles",
                        }
                    ]
                ),
                encoding="utf-8",
            )


def test_slots_help_screens() -> None:
    result = runner.invoke(app, ["slots", "--help"])
    assert result.exit_code == 0
    assert "3D metric parking slot detection" in result.stdout

    detect_help = runner.invoke(app, ["slots", "detect", "--help"])
    assert detect_help.exit_code == 0
    assert "--output" in detect_help.stdout
    assert "--png" in detect_help.stdout
    assert "--vacant-only" in detect_help.stdout


def test_slots_detect_single_camera(tmp_path: Path) -> None:
    _write_slots_dataset(tmp_path, cameras=["FV"])
    out_json = tmp_path / "outputs/slots.json"

    result = runner.invoke(
        app,
        ["slots", "detect", "--root", str(tmp_path), "--output", str(out_json)],
    )

    assert result.exit_code == 0
    assert out_json.is_file()

    data = read_json(out_json)
    assert "summary" in data
    assert "slots" in data
    assert "metadata" in data
    assert data["summary"]["total_slots"] >= 0


def test_slots_detect_all_cameras(tmp_path: Path) -> None:
    _write_slots_dataset(tmp_path, cameras=["FV", "RV", "MVL", "MVR"])
    out_json = tmp_path / "outputs/surround_slots.json"
    out_png = tmp_path / "outputs/surround_slots.png"

    result = runner.invoke(
        app,
        [
            "slots",
            "detect",
            "--root",
            str(tmp_path),
            "--all-cameras",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )

    assert result.exit_code == 0
    assert out_json.is_file()
    assert out_png.is_file()

    data = read_json(out_json)
    assert set(data["camera_sources"]) == {"FV", "RV", "MVL", "MVR"}


def test_slots_detect_vacant_only_filter(tmp_path: Path) -> None:
    _write_slots_dataset(tmp_path, cameras=["FV"])
    out_json = tmp_path / "outputs/vacant_slots.json"

    result = runner.invoke(
        app,
        [
            "slots",
            "detect",
            "--root",
            str(tmp_path),
            "--output",
            str(out_json),
            "--vacant-only",
        ],
    )

    assert result.exit_code == 0
    data = read_json(out_json)
    for slot in data["slots"]:
        assert slot["status"] == "vacant"


def test_slots_detect_overwrite_protection(tmp_path: Path) -> None:
    _write_slots_dataset(tmp_path, cameras=["FV"])
    out_json = tmp_path / "outputs/slots.json"
    out_json.parent.mkdir(parents=True)
    out_json.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["slots", "detect", "--root", str(tmp_path), "--output", str(out_json)],
    )

    assert result.exit_code == 1
    assert "Artifact error" in result.stderr

    # With --overwrite, succeeds
    res_overwrite = runner.invoke(
        app,
        [
            "slots",
            "detect",
            "--root",
            str(tmp_path),
            "--output",
            str(out_json),
            "--overwrite",
        ],
    )
    assert res_overwrite.exit_code == 0


def test_slots_detect_missing_complete_surround_frames(tmp_path: Path) -> None:
    _write_slots_dataset(tmp_path, cameras=["FV", "RV"])  # Missing MVL, MVR

    result = runner.invoke(
        app,
        [
            "slots",
            "detect",
            "--root",
            str(tmp_path),
            "--all-cameras",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    assert result.exit_code == 1
    assert "No complete multi-camera frames found" in result.stderr
