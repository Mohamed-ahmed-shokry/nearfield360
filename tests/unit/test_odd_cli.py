"""Unit and integration tests for nearfield360 odd CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


def test_odd_help() -> None:
    result = runner.invoke(app, ["odd", "--help"])
    assert result.exit_code == 0
    assert "info" in result.stdout
    assert "evaluate" in result.stdout
    assert "simulate" in result.stdout
    assert "render" in result.stdout
    assert "audit" in result.stdout


def test_odd_info_json(tmp_path: Path) -> None:
    out_file = tmp_path / "info.json"
    result = runner.invoke(app, ["odd", "info", "--json", "--output", str(out_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["min_ambient_lux"] == 2.0
    assert data["nominal_speed_limit_mps"] == 4.5
    assert data["enable_mrm_interlock"] is True
    assert out_file.is_file()


def test_odd_evaluate_nominal(tmp_path: Path) -> None:
    out_file = tmp_path / "eval_nominal.json"
    result = runner.invoke(
        app,
        [
            "odd",
            "evaluate",
            "--lux",
            "600.0",
            "--friction",
            "0.85",
            "--rain-mmph",
            "0.0",
            "--json",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state"] == "nominal"
    assert payload["governor_speed_limit_mps"] == 4.5
    assert payload["safety_headway_multiplier"] == 1.0
    assert payload["mrm_required"] is False


def test_odd_evaluate_degraded() -> None:
    result = runner.invoke(
        app,
        [
            "odd",
            "evaluate",
            "--lux",
            "50.0",
            "--friction",
            "0.55",
            "--rain-mmph",
            "4.0",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state"] == "advisory_degraded"
    assert payload["governor_speed_limit_mps"] == 2.5
    assert payload["safety_headway_multiplier"] == 1.5


def test_odd_simulate_scenario(tmp_path: Path) -> None:
    out_json = tmp_path / "sim_report.json"
    out_png = tmp_path / "sim_hud.png"

    result = runner.invoke(
        app,
        [
            "odd",
            "simulate",
            "--scenario",
            "heavy_rain_glare",
            "--steps",
            "10",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    assert out_png.is_file()
    report = read_json(out_json)
    assert report["total_samples"] == 10
    assert "latest_evaluation" in report


def test_odd_render_from_report(tmp_path: Path) -> None:
    report_file = tmp_path / "odd_rep.json"
    hud_file = tmp_path / "hud_render.png"

    # Generate a report first
    runner.invoke(
        app,
        ["odd", "simulate", "--scenario", "nominal", "--steps", "5", "--output", str(report_file)],
    )

    result = runner.invoke(
        app,
        ["odd", "render", "--input", str(report_file), "--output", str(hud_file)],
    )
    assert result.exit_code == 0
    assert hud_file.is_file()
    assert hud_file.stat().st_size > 5000


def test_odd_audit_command(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit.json"
    result = runner.invoke(app, ["odd", "audit", "--json", "--output", str(audit_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "pass"
    assert data["checks_passed"] == data["checks_total"]
