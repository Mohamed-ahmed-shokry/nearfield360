import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app

runner = CliRunner()


def _calibration_path(tmp_path: Path) -> Path:
    payload = {
        "extrinsic": {
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 1.0],
        },
        "intrinsic": {
            "aspect_ratio": 1.0,
            "cx_offset": 0.0,
            "cy_offset": 0.0,
            "height": 480,
            "k1": 100.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "model": "radial_poly",
            "poly_order": 4,
            "width": 640,
        },
        "name": "FV",
    }
    path = tmp_path / "00001_FV.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_geometry_info_reports_calibration(tmp_path: Path) -> None:
    calibration = _calibration_path(tmp_path)

    result = runner.invoke(
        app, ["geometry", "info", "--calibration", str(calibration), "--theta-max", "2.2"]
    )

    assert result.exit_code == 0, result.output
    assert "Camera: FV" in result.stdout
    assert "Theta max" in result.stdout


def test_geometry_info_json_uses_configured_theta_max(tmp_path: Path) -> None:
    calibration = _calibration_path(tmp_path)

    result = runner.invoke(app, ["geometry", "info", "--calibration", str(calibration), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["camera"] == "FV"
    assert payload["theta_max"] == 2.2
    assert payload["image"] == {"width": 640, "height": 480}


def test_geometry_ground_maps_offset_pixel_to_ground(tmp_path: Path) -> None:
    calibration = _calibration_path(tmp_path)
    # Principal (319.5, 239.5) looks straight up with identity extrinsics;
    # a 200-pixel offset selects theta=2.0 rad and must hit z=0.
    principal = "319.5,239.5"
    offset = "519.5,239.5"

    result = runner.invoke(
        app,
        [
            "geometry",
            "ground",
            "--calibration",
            str(calibration),
            "--pixel",
            principal,
            "--pixel",
            offset,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [item["valid"] for item in payload["footprints"]] == [False, True]
    assert payload["footprints"][1]["ground"][2] == 0.0


def test_geometry_ground_requires_pixels(tmp_path: Path) -> None:
    calibration = _calibration_path(tmp_path)

    result = runner.invoke(app, ["geometry", "ground", "--calibration", str(calibration)])

    assert result.exit_code == 2
    assert "at least one --pixel" in result.stderr


def test_geometry_bev_reports_configured_grid() -> None:
    result = runner.invoke(app, ["geometry", "bev", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["width"] == 320
    assert payload["height"] == 240
    assert payload["shape"] == [240, 320]


def test_geometry_info_rejects_invalid_calibration(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["geometry", "info", "--calibration", str(bad)])

    assert result.exit_code == 2
    assert "Calibration error" in result.stderr
