"""Unit tests for nearfield360 inspect CLI command group (ISO 23374 / ECE R43)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_inspect_info(tmp_path: Path) -> None:
    # 1. Plain text output
    res = runner.invoke(app, ["inspect", "info"])
    assert res.exit_code == 0
    assert "Vehicle Condition Inspection Configuration" in res.stdout
    assert "1.6 mm" in res.stdout

    # 2. JSON output
    json_res = runner.invoke(app, ["inspect", "info", "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["min_legal_tread_depth_mm"] == 1.6
    assert data["enable_cryptographic_certificate"] is True

    # 3. File output
    out_file = tmp_path / "inspect_info.json"
    file_res = runner.invoke(app, ["inspect", "info", "--output", str(out_file)])
    assert file_res.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["nominal_cold_pressure_bar"] == 2.40


def test_cli_inspect_scan(tmp_path: Path) -> None:
    # 1. Normal scan (no defects)
    res = runner.invoke(app, ["inspect", "scan"])
    assert res.exit_code == 0
    assert "Condition Score 100.0%" in res.stdout

    # 2. Scan with defects
    defects_res = runner.invoke(app, ["inspect", "scan", "--with-defects"])
    assert defects_res.exit_code == 0
    assert "Anomalies: 2" in defects_res.stdout
    assert "SCRATCH" in defects_res.stdout

    # 3. JSON output and file saving
    out_file = tmp_path / "scan_out.json"
    json_res = runner.invoke(
        app,
        ["inspect", "scan", "--with-defects", "--json", "-o", str(out_file)],
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["vin"] == "WP0ZZZ99ZTS392001"
    assert len(data["anomalies"]) == 2
    assert out_file.is_file()


def test_cli_inspect_tire(tmp_path: Path) -> None:
    # 1. Default tires
    res = runner.invoke(app, ["inspect", "tire"])
    assert res.exit_code == 0
    assert "4-Wheel Tire Profiling" in res.stdout
    assert "FRONT_LEFT" in res.stdout

    # 2. Worn tire with custom depth and low pressure
    worn_res = runner.invoke(
        app,
        ["inspect", "tire", "--fl", "1.2", "--pressure", "1.80"],
    )
    assert worn_res.exit_code == 0
    assert "CRITICAL_REPLACE" in worn_res.stdout

    # 3. JSON output
    out_file = tmp_path / "tire_out.json"
    json_res = runner.invoke(app, ["inspect", "tire", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert len(data["tires"]) == 4
    assert out_file.is_file()


def test_cli_inspect_glass(tmp_path: Path) -> None:
    # 1. Clean glass
    res = runner.invoke(app, ["inspect", "glass"])
    assert res.exit_code == 0
    assert "COMPLIANT" in res.stdout

    # 2. Cracked glass
    crack_res = runner.invoke(app, ["inspect", "glass", "--crack"])
    assert crack_res.exit_code == 0
    assert "NON-COMPLIANT" in crack_res.stdout

    # 3. JSON output
    out_file = tmp_path / "glass_out.json"
    json_res = runner.invoke(app, ["inspect", "glass", "--crack", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["ece_r43_compliant"] is False
    assert out_file.is_file()


def test_cli_inspect_certificate(tmp_path: Path) -> None:
    # 1. Certificate generation
    res = runner.invoke(app, ["inspect", "certificate"])
    assert res.exit_code == 0
    assert "Liability Certificate Issued" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "cert.json"
    json_res = runner.invoke(
        app,
        ["inspect", "certificate", "--vin", "TESTVIN123", "--json", "-o", str(out_file)],
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["vin"] == "TESTVIN123"
    assert "payload_sha256" in data
    assert "signature_hmac" in data
    assert out_file.is_file()


def test_cli_inspect_compare(tmp_path: Path) -> None:
    # 1. No new damage
    res = runner.invoke(app, ["inspect", "compare"])
    assert res.exit_code == 0
    assert "NO_NEW_DAMAGE" in res.stdout

    # 2. New damage detected
    dmg_res = runner.invoke(app, ["inspect", "compare", "--damage"])
    assert dmg_res.exit_code == 0
    assert "NEW_DAMAGE_DETECTED" in dmg_res.stdout

    # 3. JSON output
    out_file = tmp_path / "comp.json"
    json_res = runner.invoke(app, ["inspect", "compare", "--damage", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["new_damage_count"] == 1
    assert out_file.is_file()


def test_cli_inspect_render(tmp_path: Path) -> None:
    out_png = tmp_path / "test_dashboard.png"
    res = runner.invoke(app, ["inspect", "render", "-o", str(out_png)])
    assert res.exit_code == 0
    assert out_png.is_file()
    assert out_png.stat().st_size > 0

    # JSON render summary
    json_png = tmp_path / "test_dashboard2.png"
    json_res = runner.invoke(app, ["inspect", "render", "-o", str(json_png), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["canvas_shape"] == [720, 1280, 3]


def test_cli_inspect_audit(tmp_path: Path) -> None:
    # 1. Text output
    res = runner.invoke(app, ["inspect", "audit"])
    assert res.exit_code == 0
    assert "COMPLIANT" in res.stdout
    assert "PASS" in res.stdout

    # 2. JSON output
    out_file = tmp_path / "audit.json"
    json_res = runner.invoke(app, ["inspect", "audit", "--json", "-o", str(out_file)])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["compliant"] is True
    assert data["total_checks"] == 7
    assert out_file.is_file()
