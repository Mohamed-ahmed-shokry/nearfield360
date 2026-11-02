"""Unit tests for nearfield360 safety CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_safety_info(tmp_path: Path) -> None:
    result = runner.invoke(app, ["safety", "info", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["heartbeat_timeout_ms"] == 150.0
    assert payload["max_consecutive_misses"] == 3
    assert "vehicle_controller" in payload["critical_safety_nodes"]
    assert "fail_silent_latch" in payload["supported_safety_states"]

    out_file = tmp_path / "safety_info.json"
    file_result = runner.invoke(app, ["safety", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()


def test_cli_safety_watchdog(tmp_path: Path) -> None:
    # Dropped critical node
    result = runner.invoke(app, ["safety", "watchdog", "-d", "vehicle_controller", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["is_critical_failure"] is True
    assert "vehicle_controller" in payload["critical_failure_reason"]

    # Dropped non-critical node
    res_teleop = runner.invoke(app, ["safety", "watchdog", "-d", "teleop_supervisor", "--json"])
    assert res_teleop.exit_code == 0
    p_teleop = json.loads(res_teleop.stdout)
    assert p_teleop["is_critical_failure"] is False

    out_file = tmp_path / "watchdog.json"
    res_out = runner.invoke(app, ["safety", "watchdog", "-o", str(out_file)])
    assert res_out.exit_code == 0
    assert out_file.is_file()


def test_cli_safety_plausibility(tmp_path: Path) -> None:
    # Nominal kinematics
    res_nom = runner.invoke(
        app,
        [
            "safety",
            "plausibility",
            "--steering-rate",
            "0.2",
            "--acceleration",
            "1.0",
            "--speed-diff",
            "0.2",
            "--json",
        ],
    )
    assert res_nom.exit_code == 0
    p_nom = json.loads(res_nom.stdout)
    assert p_nom["is_plausible"] is True
    assert p_nom["violations_count"] == 0

    # Excessive steering rate
    res_viol = runner.invoke(
        app,
        ["safety", "plausibility", "--steering-rate", "1.5", "--json"],
    )
    assert res_viol.exit_code == 0
    p_viol = json.loads(res_viol.stdout)
    assert p_viol["is_plausible"] is False
    assert p_viol["violations_count"] >= 1


def test_cli_safety_simulate(tmp_path: Path) -> None:
    out_json = tmp_path / "sim_report.json"
    out_png = tmp_path / "sim_hud.png"

    # Nominal scenario
    res_nom = runner.invoke(
        app,
        ["safety", "simulate", "-s", "nominal", "-o", str(out_json), "-p", str(out_png)],
    )
    assert res_nom.exit_code == 0
    assert out_json.is_file()
    assert out_png.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["active_state"] == "normal_operation"

    # Critical timeout scenario
    res_crit = runner.invoke(app, ["safety", "simulate", "-s", "critical_timeout", "--json"])
    assert res_crit.exit_code == 0
    p_crit = json.loads(res_crit.stdout)
    assert p_crit["active_state"] == "emergency_safe_stop"


def test_cli_safety_render(tmp_path: Path) -> None:
    out_png = tmp_path / "custom_safety_hud.png"
    result = runner.invoke(app, ["safety", "render", "-s", "fail_latch", "-p", str(out_png)])
    assert result.exit_code == 0
    assert "Rendered tactical safety HUD dashboard" in result.stdout
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_cli_safety_audit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["safety", "audit", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["overall_compliant"] is True
    assert payload["total_checks"] >= 5
    assert payload["failed_checks"] == 0

    out_file = tmp_path / "audit.json"
    res_out = runner.invoke(app, ["safety", "audit", "-o", str(out_file)])
    assert res_out.exit_code == 0
    assert out_file.is_file()
