"""Unit tests for nearfield360 ops CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_ops_info(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ops", "info"])
    assert result.exit_code == 0
    assert "Facility Operations Control Tower Configuration:" in result.stdout
    assert "Day Duration Horizon" in result.stdout

    # Test JSON output
    json_result = runner.invoke(app, ["ops", "info", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["day_duration_hours"] == 24.0
    assert payload["default_slot_capacity"] == 120

    # Test file output
    out_file = tmp_path / "ops_info.json"
    file_result = runner.invoke(app, ["ops", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()


def test_cli_ops_simulate(tmp_path: Path) -> None:
    out_json = tmp_path / "day_report.json"
    out_png = tmp_path / "day_dashboard.png"
    result = runner.invoke(
        app,
        [
            "ops",
            "simulate",
            "--seed",
            "7",
            "--capacity",
            "50",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "Facility Operations Simulation Summary" in result.stdout
    assert "Total Customer Arrivals" in result.stdout
    assert "Total Facility Revenue" in result.stdout
    assert out_json.is_file()
    assert out_png.is_file()

    # Verify JSON content
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["seed"] == 7
    assert data["slot_capacity"] == 50
    assert len(data["snapshots"]) == 24
    assert len(data["journeys"]) > 0
    assert "kpis" in data

    # Test --json flag
    json_res = runner.invoke(app, ["ops", "simulate", "--seed", "7", "--capacity", "50", "--json"])
    assert json_res.exit_code == 0
    parsed = json.loads(json_res.stdout)
    assert parsed["seed"] == 7


def test_cli_ops_render(tmp_path: Path) -> None:
    out_png = tmp_path / "rendered_ops.png"
    result = runner.invoke(app, ["ops", "render", "-o", str(out_png), "--seed", "11", "-c", "40"])
    assert result.exit_code == 0
    assert "Rendered tactical operations dashboard" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 5000


def test_cli_ops_audit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ops", "audit"])
    assert result.exit_code == 0
    assert "Operations Control Tower Subsystem Audit:" in result.stdout
    assert "[PASS]" in result.stdout

    # Audit JSON export
    out_file = tmp_path / "ops_audit.json"
    json_res = runner.invoke(app, ["ops", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    payload = json.loads(json_res.stdout)
    assert payload["status"] == "pass"
    assert out_file.is_file()
