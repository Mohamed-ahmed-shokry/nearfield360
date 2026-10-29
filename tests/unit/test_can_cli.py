"""Unit tests for nearfield360 can CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_can_info(tmp_path: Path) -> None:
    result = runner.invoke(app, ["can", "info"])
    assert result.exit_code == 0
    assert "Vehicle CAN Bus Gateway Configuration (ISO 11898):" in result.stdout
    assert "Nominal Bitrate" in result.stdout

    # Test JSON output
    json_result = runner.invoke(app, ["can", "info", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["bus_bitrate_kbps"] == 500
    assert payload["max_bus_load_percent"] == 85.0

    # Test file output
    out_file = tmp_path / "can_info.json"
    file_result = runner.invoke(app, ["can", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()


def test_cli_can_encode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["can", "encode", "--steer", "0.25", "--throttle", "30.0", "--gear", "D", "--speed", "5.5"],
    )
    assert result.exit_code == 0
    assert "Serialized Vehicle Control CAN Frame:" in result.stdout
    assert "Arbitration ID : 0x120" in result.stdout
    assert "DLC            : 8 bytes" in result.stdout

    # Test JSON output to file
    out_file = tmp_path / "can_encoded.json"
    json_res = runner.invoke(
        app,
        ["can", "encode", "--steer", "-0.1", "--gear", "R", "--json", "-o", str(out_file)],
    )
    assert json_res.exit_code == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["arbitration_id"] == "0x120"
    assert "crc8" in data


def test_cli_can_simulate(tmp_path: Path) -> None:
    out_json = tmp_path / "hil_report.json"
    out_png = tmp_path / "hil_oscilloscope.png"
    result = runner.invoke(
        app,
        [
            "can",
            "simulate",
            "--duration-s",
            "0.5",
            "--fault-rate",
            "0.10",
            "--seed",
            "123",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "HIL CAN Bus Simulation Summary" in result.stdout
    assert "Frames Transmitted" in result.stdout
    assert out_json.is_file()
    assert out_png.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["duration_s"] == 0.5
    assert len(data["target_ecus"]) == 3
    assert "telemetry" in data


def test_cli_can_render(tmp_path: Path) -> None:
    out_png = tmp_path / "rendered_hil.png"
    result = runner.invoke(
        app,
        ["can", "render", "-o", str(out_png), "-d", "0.3", "-f", "0.05", "-s", "7"],
    )
    assert result.exit_code == 0
    assert "Rendered tactical HIL oscilloscope dashboard" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 5000


def test_cli_can_audit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["can", "audit"])
    assert result.exit_code == 0
    assert "Vehicle CAN Bus Gateway Subsystem Audit (ISO 11898):" in result.stdout
    assert "[PASS]" in result.stdout

    # Audit JSON export
    out_file = tmp_path / "can_audit.json"
    json_res = runner.invoke(app, ["can", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    payload = json.loads(json_res.stdout)
    assert payload["status"] == "pass"
    assert out_file.is_file()
