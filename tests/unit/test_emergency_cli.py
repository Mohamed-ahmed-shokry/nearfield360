"""Unit tests for nearfield360 emergency CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_emergency_info() -> None:
    result = runner.invoke(app, ["emergency", "info"])
    assert result.exit_code == 0
    assert "[Emergency & Life Safety Configuration]" in result.stdout
    assert "Preemption Radius:" in result.stdout
    assert "Hazard Buffer Radius:" in result.stdout
    assert "Evacuation Speed Limit:" in result.stdout
    assert "Fire Door Clearance Timeout:" in result.stdout

    json_result = runner.invoke(app, ["emergency", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["hazard_buffer_radius_m"] == 12.0
    assert data["evacuation_speed_limit_mps"] == 4.0
    assert data["fire_door_clearance_timeout_s"] == 15.0


def test_cli_emergency_preempt(tmp_path: Path) -> None:
    out_file = tmp_path / "preempt.json"
    png_file = tmp_path / "preempt.png"

    result = runner.invoke(
        app,
        [
            "emergency",
            "preempt",
            "--responder-id",
            "fd-engine-01",
            "--type",
            "fire_engine",
            "--start-x",
            "0.0",
            "--start-y",
            "10.0",
            "--target-x",
            "60.0",
            "--target-y",
            "10.0",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Emergency Vehicle Preemption Evaluation]" in result.stdout
    assert "fd-engine-01" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["responder"]["vehicle_id"] == "fd-engine-01"
    assert payload["is_preemption_active"] is True
    assert "veh-avp-01" in payload["evaluations"]


def test_cli_emergency_hazard(tmp_path: Path) -> None:
    out_file = tmp_path / "hazard.json"
    png_file = tmp_path / "hazard.png"

    result = runner.invoke(
        app,
        [
            "emergency",
            "hazard",
            "--hazard-id",
            "hz-test-01",
            "--type",
            "ev_thermal_runaway",
            "--severity",
            "full_facility_evacuation",
            "--x",
            "20.0",
            "--y",
            "20.0",
            "--radius",
            "8.0",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Hazard Containment & Exclusion Zone]" in result.stdout
    assert "hz-test-01" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["hazard"]["hazard_id"] == "hz-test-01"
    assert payload["hazard"]["radius_m"] == 8.0
    assert "veh-avp-01" in payload["points_in_exclusion"]


def test_cli_emergency_evacuate(tmp_path: Path) -> None:
    out_file = tmp_path / "evacuate.json"
    png_file = tmp_path / "evacuate.png"

    result = runner.invoke(
        app,
        [
            "emergency",
            "evacuate",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Facility Multi-Wave Evacuation Dispatch]" in result.stdout
    assert "Total Waves:" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(payload["waves"]) >= 1
    assert "veh-avp-01" in payload["routes"]


def test_cli_emergency_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    png_file = tmp_path / "audit.png"

    result = runner.invoke(
        app,
        [
            "emergency",
            "audit",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Life Safety & Emergency Compliance Audit]" in result.stdout
    assert "Audit Status:     PASS" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["preemption_verified"] is True
    assert payload["fire_door_interlock_verified"] is True
    assert payload["evacuation_routing_verified"] is True
