"""Unit tests for nearfield360 floor CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_floor_info() -> None:
    result = runner.invoke(app, ["floor", "info"])
    assert result.exit_code == 0
    assert "[Multi-Floor Vertical Transit Configuration]" in result.stdout
    assert "Default Floor Height:" in result.stdout
    assert "Max Ramp Grade:" in result.stdout
    assert "Elevator Hoist Speed:" in result.stdout

    json_result = runner.invoke(app, ["floor", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["default_floor_height_m"] == 3.5
    assert data["max_ramp_grade_pct"] == 15.0
    assert data["elevator_hoist_speed_mps"] == 1.2


def test_cli_floor_route(tmp_path: Path) -> None:
    out_file = tmp_path / "route.json"
    png_file = tmp_path / "route.png"

    result = runner.invoke(
        app,
        [
            "floor",
            "route",
            "--origin",
            "L1-entry",
            "--destination",
            "L2-bay-42",
            "--vehicle-id",
            "veh-avp-01",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Multi-Floor Route:" in result.stdout
    assert "Origin Floor:         L1" in result.stdout
    assert "Destination Floor:    L2" in result.stdout
    assert "ramp-L1-to-L2" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["origin_floor_id"] == "L1"
    assert payload["destination_floor_id"] == "L2"
    assert "ramp-L1-to-L2" in payload["portals_sequence"]
    assert len(payload["waypoints"]) >= 4


def test_cli_floor_ramp(tmp_path: Path) -> None:
    out_file = tmp_path / "ramp.json"
    png_file = tmp_path / "ramp.png"

    result = runner.invoke(
        app,
        [
            "floor",
            "ramp",
            "--vehicle-id",
            "veh-ramp-test",
            "--grade",
            "12.0",
            "--elevation",
            "1.75",
            "--spiral-radius",
            "12.0",
            "--velocity",
            "2.5",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Ramp Dynamics Simulation: veh-ramp-test]" in result.stdout
    assert "Incline Grade:        12.0%" in result.stdout
    assert "Chassis Grounding:    SAFE" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["pitch_state"]["grade_pct"] == 12.0
    assert payload["spiral_kinematics"]["radius_m"] == 12.0


def test_cli_floor_elevator(tmp_path: Path) -> None:
    out_file = tmp_path / "elevator.json"
    png_file = tmp_path / "elevator.png"

    result = runner.invoke(
        app,
        [
            "floor",
            "elevator",
            "--elevator-id",
            "lift-01",
            "--origin-floor",
            "L1",
            "--destination-floor",
            "L2",
            "--vehicle-id",
            "veh-lift-test",
            "--weight",
            "1900.0",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Elevator Dispatch Cycle: lift-01]" in result.stdout
    assert "veh-lift-test (1900.0 kg)" in result.stdout
    assert "Drive-in Permitted:   YES" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["elevator_id"] == "lift-01"
    assert payload["drive_in_permitted"] is True
    assert payload["cabin_state"]["occupied_vehicle_id"] == "veh-lift-test"


def test_cli_floor_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    png_file = tmp_path / "audit.png"

    result = runner.invoke(
        app,
        [
            "floor",
            "audit",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "[Multi-Floor Facility Audit:" in result.stdout
    assert "Total Floors:" in result.stdout
    assert "Total Capacity:" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["floors_count"] == 3
    assert payload["total_bays"] > 0
