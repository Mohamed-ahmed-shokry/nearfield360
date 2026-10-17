"""Unit tests for nearfield360 gate CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_gate_info(tmp_path: Path) -> None:
    # 1. Plain text output
    result = runner.invoke(app, ["gate", "info"])
    assert result.exit_code == 0
    assert "Facility Access Gateway Configuration:" in result.stdout
    assert "Portal Clearance Limits:" in result.stdout
    assert "Barrier Arm Length:" in result.stdout

    # 2. JSON stdout
    json_result = runner.invoke(app, ["gate", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["portal_clearance_height_m"] == 2.10
    assert data["portal_clearance_width_m"] == 2.60

    # 3. Output file write
    out_file = tmp_path / "gate_info.json"
    file_result = runner.invoke(app, ["gate", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["alpr_confidence_threshold"] == 0.85


def test_cli_gate_verify_granted(tmp_path: Path) -> None:
    out_file = tmp_path / "verify_granted.json"
    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--plate",
            "ABC-1234",
            "--confidence",
            "0.96",
            "--height",
            "1.70",
            "--width",
            "1.90",
            "--ground-clearance",
            "0.15",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Gate Access Verification: [GRANTED]" in result.stdout
    assert "PASS-ABC1234" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["granted"] is True
    assert data["status"] == "granted"
    assert data["headroom_margin_m"] > 0.0


def test_cli_gate_verify_rejected_over_height() -> None:
    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--plate",
            "ABC-1234",
            "--height",
            "2.85",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["granted"] is False
    assert data["status"] == "rejected_over_height"
    assert "exceeds portal limit" in data["message"]


def test_cli_gate_verify_rejected_credential() -> None:
    result = runner.invoke(
        app,
        [
            "gate",
            "verify",
            "--plate",
            "UNKNOWN-99",
            "--confidence",
            "0.40",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["granted"] is False
    assert data["status"] == "rejected_unknown_credential"


def test_cli_gate_transit_normal(tmp_path: Path) -> None:
    out_file = tmp_path / "transit.json"
    result = runner.invoke(
        app,
        [
            "gate",
            "transit",
            "--vehicle-id",
            "veh-test-01",
            "--plate",
            "ABC-1234",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Gate Transit Simulation Cycle" in result.stdout
    assert "Granted: True" in result.stdout
    assert "None (Secure)" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["granted"] is True
    assert data["tailgating_detected"] is False
    assert data["duration_s"] > 0.0


def test_cli_gate_transit_tailgating_and_obstruction() -> None:
    # 1. Tailgating simulation
    tg_result = runner.invoke(
        app,
        [
            "gate",
            "transit",
            "--tailgating",
            "--json",
        ],
    )
    assert tg_result.exit_code == 0
    tg_data = json.loads(tg_result.stdout)
    assert tg_data["tailgating_detected"] is True

    # 2. Obstruction simulation
    obs_result = runner.invoke(
        app,
        [
            "gate",
            "transit",
            "--obstruction",
            "--json",
        ],
    )
    assert obs_result.exit_code == 0
    obs_data = json.loads(obs_result.stdout)
    assert obs_data["safety_reversal_triggered"] is True


def test_cli_gate_render(tmp_path: Path) -> None:
    png_file = tmp_path / "gate_map.png"
    result = runner.invoke(
        app,
        [
            "gate",
            "render",
            "--png",
            str(png_file),
            "--barrier-angle",
            "45.0",
            "--loop",
            "loop_b_presence",
            "--tailgating",
        ],
    )
    assert result.exit_code == 0
    assert "Gateway tactical dashboard successfully rendered" in result.stdout
    assert png_file.is_file()
    assert png_file.stat().st_size > 5000


def test_cli_gate_audit(tmp_path: Path) -> None:
    out_file = tmp_path / "gate_audit.json"
    result = runner.invoke(
        app,
        [
            "gate",
            "audit",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Gateway Portal Operational Audit Report" in result.stdout
    assert "Total Transits:" in result.stdout
    assert out_file.is_file()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["total_transits"] == 3
    assert data["granted_count"] == 1
    assert data["rejected_count"] == 2
    assert data["clearance_violations"] == 1
