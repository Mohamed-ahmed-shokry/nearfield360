"""Unit tests for nearfield360 teleop CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_teleop_info() -> None:
    result = runner.invoke(app, ["teleop", "info"])
    assert result.exit_code == 0
    assert "Remote Teleoperation & Safety MRM Configuration:" in result.stdout
    assert "Heartbeat Timeout:" in result.stdout
    assert "Max Latency (Direct Drive):" in result.stdout

    json_result = runner.invoke(app, ["teleop", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "heartbeat_timeout_s" in data
    assert "max_rtt_ms" in data
    assert "critical_rtt_ms" in data
    assert "video_target_fps" in data


def test_cli_teleop_session(tmp_path: Path) -> None:
    out_file = tmp_path / "session.json"
    png_file = tmp_path / "session.png"
    result = runner.invoke(
        app,
        [
            "teleop",
            "session",
            "--vehicle-id",
            "veh-avp-tele-01",
            "--operator-id",
            "op-desk-7",
            "--rtt",
            "55.0",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Remote Teleoperation Session Execution:" in result.stdout
    assert "veh-avp-tele-01" in result.stdout
    assert "TELE_GUIDANCE" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["session"]["vehicle_id"] == "veh-avp-tele-01"
    assert data["session"]["operator_id"] == "op-desk-7"
    assert data["command_dispatched"] is True


def test_cli_teleop_session_emergency_stop(tmp_path: Path) -> None:
    out_file = tmp_path / "estop.json"
    png_file = tmp_path / "estop.png"
    result = runner.invoke(
        app,
        [
            "teleop",
            "session",
            "--vehicle-id",
            "veh-avp-estop-01",
            "--emergency-stop",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "MRM_SAFE_STOP" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["session"]["active_mode"] == "mrm_safe_stop"
    assert data["mrm_status"]["parking_brake_engaged"] is True


def test_cli_teleop_fallback(tmp_path: Path) -> None:
    out_file = tmp_path / "fallback.json"
    png_file = tmp_path / "fallback.png"
    result = runner.invoke(
        app,
        [
            "teleop",
            "fallback",
            "--vehicle-id",
            "veh-crit-01",
            "--latency",
            "320.0",
            "--packet-loss",
            "0.30",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Teleoperation Video Fallback & Watchdog Execution:" in result.stdout
    assert "CRITICAL" in result.stdout or "DEGRADED" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["command_accepted"] is False
    assert data["streaming_recommendation"]["direct_driving_permitted"] is False


def test_cli_teleop_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "teleop_audit.json"
    png_file = tmp_path / "teleop_audit.png"
    result = runner.invoke(
        app,
        [
            "teleop",
            "audit",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Teleoperation Facility Operations & Safety Audit:" in result.stdout
    assert "Active Sessions:" in result.stdout
    assert "MRM Executions:" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["total_interventions_count"] >= 1
    assert data["mrm_executions_count"] >= 1
    assert "average_rtt_ms" in data
