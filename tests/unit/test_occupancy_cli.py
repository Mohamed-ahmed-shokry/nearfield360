import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()

_DOWN_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def _write_dataset(root: Path, *, camera: str = "FV", with_annotations: bool = True) -> None:
    (root / "rgb_images").mkdir(parents=True)
    assert cv2.imwrite(str(root / "rgb_images/00001_FV.png"), np.zeros((2, 3, 3), dtype=np.uint8))
    if with_annotations:
        (root / "semantic_annotations/gtLabels").mkdir(parents=True)
        mask = np.array([[1, 1, 1], [6, 6, 6]], dtype=np.uint8)
        assert cv2.imwrite(str(root / "semantic_annotations/gtLabels/00001_FV.png"), mask)
        calibration = root / "calibration_data/00001_FV.json"
        calibration.parent.mkdir(parents=True)
        calibration.write_text(
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
                        "height": 2,
                        "k1": 100.0,
                        "k2": 0.0,
                        "k3": 0.0,
                        "k4": 0.0,
                        "model": "radial_poly",
                        "poly_order": 4,
                        "width": 3,
                    },
                    "name": camera,
                }
            ),
            encoding="utf-8",
        )


def test_occupancy_layer_fuses_evidence_and_writes_report(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    output = tmp_path / "reports/occupancy.json"
    png = tmp_path / "occupancy.png"

    result = runner.invoke(
        app,
        ["occupancy", "layer", "--root", str(tmp_path), "--output", str(output), "--png", str(png)],
    )

    assert result.exit_code == 0
    assert "fused 00001_FV" in result.stdout
    payload = read_json(output)
    assert payload["samples"] == {"requested": 1, "evaluated": 1, "camera": "FV"}
    assert payload["grid"]["resolution"] == 0.05
    assert payload["evidence"]["observed_cells"] > 0
    assert {"forward_corridor", "near_circle", "warning_circle"} == {
        entry["name"] for entry in payload["zones"]
    }
    assert len(payload["risk"]) == 3
    corridor = next(entry for entry in payload["risk"] if entry["name"] == "forward_corridor")
    assert corridor["cells"] > 0
    assert corridor["occupied_cells"] >= 0
    assert json.dumps(payload)  # fully JSON-serializable
    rendered = cv2.imread(str(png))
    assert rendered is not None and rendered.shape[:2] == (240, 320)


def test_occupancy_layer_refuses_to_overwrite_and_limits_samples(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    output = tmp_path / "report.json"
    first = runner.invoke(
        app,
        ["occupancy", "layer", "--root", str(tmp_path), "--output", str(output)],
    )
    second = runner.invoke(
        app,
        ["occupancy", "layer", "--root", str(tmp_path), "--output", str(output)],
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Artifact error" in second.stderr

    limited = runner.invoke(
        app,
        [
            "occupancy",
            "layer",
            "--root",
            str(tmp_path),
            "--samples",
            "5",
            "--output",
            str(tmp_path / "x.json"),
        ],
    )
    assert limited.exit_code == 1
    assert "only 1 FV samples" in limited.stderr


def test_occupancy_layer_rejects_missing_annotations(tmp_path: Path) -> None:
    _write_dataset(tmp_path, with_annotations=False)

    result = runner.invoke(
        app,
        ["occupancy", "layer", "--root", str(tmp_path), "--output", str(tmp_path / "r.json")],
    )

    assert result.exit_code == 1
    assert "lacks calibration or semantic mask" in result.stderr


def test_occupancy_zones_reports_configured_geometry(tmp_path: Path) -> None:
    result = runner.invoke(app, ["occupancy", "zones", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["grid"]["width"] == 320
    names = [zone["name"] for zone in payload["zones"]]
    assert "forward_corridor" in names
    assert all(zone["cells"] > 0 for zone in payload["zones"])
