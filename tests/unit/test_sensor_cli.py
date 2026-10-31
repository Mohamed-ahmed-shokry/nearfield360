"""Unit tests for nearfield360 sensor CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_sensor_info(tmp_path: Path) -> None:
    result = runner.invoke(app, ["sensor", "info"])
    assert result.exit_code == 0
    assert "NearField360 Sensor Simulation Suite Configuration:" in result.stdout
    assert "Radar Carrier Frequency" in result.stdout

    # Test JSON output
    json_result = runner.invoke(app, ["sensor", "info", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["carrier_frequency_ghz"] == 77.0
    assert payload["num_ultrasonic_sensors"] == 12

    # Test file output
    out_file = tmp_path / "sensor_info.json"
    file_result = runner.invoke(app, ["sensor", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()


def test_cli_sensor_radar(tmp_path: Path) -> None:
    out_file = tmp_path / "radar_points.json"
    result = runner.invoke(
        app,
        [
            "sensor",
            "radar",
            "--range",
            "10.0",
            "--velocity",
            "-2.0",
            "--rcs",
            "15.0",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "FMCW Radar 2D FFT & CA-CFAR Detection Result:" in result.stdout
    assert out_file.is_file()

    # Test JSON stdout
    json_result = runner.invoke(app, ["sensor", "radar", "--range", "8.0", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert "points" in payload
    assert "sensor_id" in payload


def test_cli_sensor_ultrasonic(tmp_path: Path) -> None:
    out_file = tmp_path / "uss_frame.json"
    result = runner.invoke(
        app,
        [
            "sensor",
            "ultrasonic",
            "--front-dist",
            "0.75",
            "--rear-dist",
            "1.50",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Perimeter Ultrasonic Transducer Array Echoes:" in result.stdout
    assert out_file.is_file()

    json_result = runner.invoke(app, ["sensor", "ultrasonic", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert "echoes" in payload
    assert len(payload["echoes"]) == 12


def test_cli_sensor_fuse(tmp_path: Path) -> None:
    out_file = tmp_path / "fused_report.json"
    result = runner.invoke(
        app,
        [
            "sensor",
            "fuse",
            "--range",
            "10.0",
            "--velocity",
            "-1.5",
            "--uss-dist",
            "0.90",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Cross-Modal Radar-Ultrasonic Fused Obstacles:" in result.stdout
    assert out_file.is_file()

    json_result = runner.invoke(app, ["sensor", "fuse", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert "fused_obstacles" in payload


def test_cli_sensor_render(tmp_path: Path) -> None:
    out_png = tmp_path / "sensor_viz.png"
    result = runner.invoke(
        app,
        [
            "sensor",
            "render",
            "--range",
            "12.0",
            "--velocity",
            "-2.0",
            "--uss-dist",
            "0.80",
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "Rendered sensor dashboard PNG:" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 5000


def test_cli_sensor_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "sensor_audit.json"
    result = runner.invoke(app, ["sensor", "audit", "--output", str(out_file)])
    assert result.exit_code == 0
    assert "Sensor Suite Audit Status: PASS" in result.stdout
    assert out_file.is_file()

    json_result = runner.invoke(app, ["sensor", "audit", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["status"] == "PASS"
    assert payload["fmcw_radar_77ghz"]["passed"] is True
    assert payload["ultrasonic_array"]["passed"] is True
    assert payload["cross_modal_fusion"]["passed"] is True
