"""Integration tests for four-camera surround pipeline runner with --gate option."""

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


def _write_four_camera_frame(root: Path, frame_id: str) -> None:
    (root / "rgb_images").mkdir(parents=True, exist_ok=True)
    (root / "calibration_data").mkdir(parents=True, exist_ok=True)
    (root / "semantic_annotations/gtLabels").mkdir(parents=True, exist_ok=True)
    (root / "detection_annotations").mkdir(parents=True, exist_ok=True)

    img = np.zeros((2, 3, 3), dtype=np.uint8)
    mask = np.array([[1, 1, 1], [6, 6, 6]], dtype=np.uint8)

    for cam in _CAMERAS:
        stem = f"{frame_id}_{cam}"
        cv2.imwrite(str(root / f"rgb_images/{stem}.png"), img)
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
        (root / f"detection_annotations/{stem}.txt").write_text(
            "vehicles,0,40,40,60,80\n",
            encoding="utf-8",
        )


def test_pipeline_run_with_gate(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_four_camera_frame(dataset, "00001")

    output = tmp_path / "pipeline_gate_report.json"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--root",
            str(dataset),
            "--output",
            str(output),
            "--gate",
        ],
    )

    assert result.exit_code == 0
    assert "Pipeline complete" in result.stdout
    payload = read_json(output)

    assert "gate" in payload
    assert payload["gate"]["gate_id"] == "GATE_ENTRY_01"
    assert payload["gate"]["vehicle_id"] == "veh-pipeline"
    assert payload["gate"]["decision"]["granted"] is True
    assert payload["gate"]["transit_duration_s"] > 0.0
    assert payload["gate"]["tailgating_detected"] is False
    assert payload["gate"]["barrier_state"] == "lowered_closed"
    assert "audit_report" in payload["gate"]
    assert payload["gate"]["audit_report"]["total_transits"] == 1
    assert payload["gate"]["audit_report"]["granted_count"] == 1
    assert "gate_ms" in payload["timings"]
    assert payload["samples"]["gate"] is True
