"""Unit tests for nearfield360 curbside CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_curbside_info() -> None:
    result = runner.invoke(app, ["curbside", "info"])
    assert result.exit_code == 0
    assert "Curbside Staging & Pedestrian Safety Configuration:" in result.stdout
    assert "Staging Bays Capacity:" in result.stdout
    assert "Pedestrian Safety Radius:" in result.stdout

    json_result = runner.invoke(app, ["curbside", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "staging_capacity" in data
    assert "max_dropoff_dwell_s" in data
    assert "pedestrian_safety_radius_m" in data
    assert "overstay_penalty_per_min" in data


def test_cli_curbside_dropoff(tmp_path: Path) -> None:
    out_file = tmp_path / "dropoff.json"
    png_file = tmp_path / "dropoff.png"
    result = runner.invoke(
        app,
        [
            "curbside",
            "dropoff",
            "--vehicle-id",
            "veh-avp-99",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Curbside Drop-off Handover Execution:" in result.stdout
    assert "veh-avp-99" in result.stdout
    assert "COMPLETED" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["mission"]["vehicle_id"] == "veh-avp-99"
    assert data["mission"]["handover_type"] == "dropoff"
    assert data["mission"]["is_completed"] is True


def test_cli_curbside_dropoff_inhibited(tmp_path: Path) -> None:
    out_file = tmp_path / "dropoff_inhibited.json"
    png_file = tmp_path / "dropoff_inhibited.png"
    result = runner.invoke(
        app,
        [
            "curbside",
            "dropoff",
            "--vehicle-id",
            "veh-avp-blocked",
            "--doors-open",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "veh-avp-blocked" in result.stdout
    assert "INHIBITED (Safety Interlock Engaged)" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["mission"]["vehicle_id"] == "veh-avp-blocked"
    assert data["mission"]["is_completed"] is False


def test_cli_curbside_summon(tmp_path: Path) -> None:
    out_file = tmp_path / "summon.json"
    png_file = tmp_path / "summon.png"
    result = runner.invoke(
        app,
        [
            "curbside",
            "summon",
            "--vehicle-id",
            "veh-vip-02",
            "--dwell",
            "45.0",
            "--pedestrian-distance",
            "4.0",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Curbside Pick-up Summon Execution:" in result.stdout
    assert "veh-vip-02" in result.stdout
    assert "COMPLETED" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["mission"]["vehicle_id"] == "veh-vip-02"
    assert data["mission"]["handover_type"] == "pickup"
    assert data["mission"]["is_completed"] is True


def test_cli_curbside_summon_pedestrian_near(tmp_path: Path) -> None:
    out_file = tmp_path / "summon_ped.json"
    png_file = tmp_path / "summon_ped.png"
    result = runner.invoke(
        app,
        [
            "curbside",
            "summon",
            "--vehicle-id",
            "veh-vip-03",
            "--pedestrian-distance",
            "1.2",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "INHIBITED (Pedestrian in Critical Zone)" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["mission"]["is_completed"] is False


def test_cli_curbside_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "audit.json"
    png_file = tmp_path / "audit.png"
    result = runner.invoke(
        app,
        [
            "curbside",
            "audit",
            "--output",
            str(out_file),
            "--png",
            str(png_file),
        ],
    )
    assert result.exit_code == 0
    assert "Curbside Handover Facility Audit Report:" in result.stdout
    assert "Total Staging Bays:" in result.stdout
    assert "Completed Missions:" in result.stdout
    assert out_file.is_file()
    assert png_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["total_bays_count"] >= 1
    assert data["completed_missions_count"] >= 1
    assert "total_overstay_penalties_usd" in data
