"""Unit tests for nearfield360 csms CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_csms_info(tmp_path: Path) -> None:
    # 1. Plain text output
    result = runner.invoke(app, ["csms", "info"])
    assert result.exit_code == 0
    assert (
        "UN R155 / ISO/SAE 21434 CSMS & Intrusion Detection System Configuration:" in result.stdout
    )
    assert "Max CAN Frame Rate" in result.stdout
    assert "Shannon Entropy Range" in result.stdout

    # 2. JSON output
    json_result = runner.invoke(app, ["csms", "info", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert data["max_can_frame_rate_per_sec"] == 250.0
    assert data["replay_window_tolerance_ms"] == 50.0
    assert data["enable_defensive_mitigation"] is True

    # 3. Output to file
    out_file = tmp_path / "csms_info.json"
    file_result = runner.invoke(app, ["csms", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["quarantine_duration_s"] == 15.0


def test_cli_csms_analyze(tmp_path: Path) -> None:
    # 1. Nominal frame
    result = runner.invoke(
        app,
        [
            "csms",
            "analyze",
            "--arbitration-id",
            "0x120",
            "--payload-hex",
            "01020304",
            "--speed",
            "0.0",
        ],
    )
    assert result.exit_code == 0
    assert "CAN Frame IDS Inspection Result: [NOMINAL]" in result.stdout
    assert "Arbitration ID : 0x120" in result.stdout

    # 2. Diagnostic SecurityAccess while moving (Critical Anomaly)
    sec_result = runner.invoke(
        app,
        [
            "csms",
            "analyze",
            "--arbitration-id",
            "0x7DF",
            "--payload-hex",
            "0227010000000000",
            "--speed",
            "12.5",
            "--json",
        ],
    )
    assert sec_result.exit_code == 0
    sec_data = json.loads(sec_result.stdout)
    assert sec_data["anomaly_detected"] is True
    assert sec_data["event"]["category"] == "uds_unauthorized_diag"
    assert sec_data["event"]["severity"] == "critical"
    assert sec_data["event"]["defense_action"] == "lockout_diagnostics"

    # 3. Output to file
    out_file = tmp_path / "analysis.json"
    runner.invoke(
        app,
        ["csms", "analyze", "--output", str(out_file)],
    )
    assert out_file.is_file()
    res = json.loads(out_file.read_text(encoding="utf-8"))
    assert "shannon_entropy_bits" in res


def test_cli_csms_quarantine(tmp_path: Path) -> None:
    # 1. Inspect empty/active quarantine list
    res = runner.invoke(app, ["csms", "quarantine"])
    assert res.exit_code == 0
    assert "Active CSMS Network Quarantines:" in res.stdout

    # 2. Add quarantine
    res_add = runner.invoke(
        app,
        [
            "csms",
            "quarantine",
            "--add",
            "0x120",
            "--reason",
            "Test DoS Injection",
            "--duration-s",
            "30.0",
        ],
    )
    assert res_add.exit_code == 0
    assert "Entity [0x120]: Test DoS Injection" in res_add.stdout

    # 3. Release quarantine
    res_rel = runner.invoke(app, ["csms", "quarantine", "--release", "0x120", "--json"])
    assert res_rel.exit_code == 0
    data = json.loads(res_rel.stdout)
    assert isinstance(data, list)
    assert len(data) == 0

    # 4. Output to file
    out_file = tmp_path / "quarantine.json"
    runner.invoke(
        app,
        ["csms", "quarantine", "--add", "0x250", "--output", str(out_file)],
    )
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["entity_id"] == "0x250"


def test_cli_csms_simulate(tmp_path: Path) -> None:
    # 1. DoS Flood simulation with HUD png export
    png_path = tmp_path / "dos_hud.png"
    out_file = tmp_path / "dos_sim.json"
    res = runner.invoke(
        app,
        [
            "csms",
            "simulate",
            "--scenario",
            "dos_flood",
            "--frames",
            "50",
            "--png",
            str(png_path),
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert "UN R155 CSMS Cyber Simulation Summary: [DOS_FLOOD]" in res.stdout
    assert "Frames Evaluated     : 50" in res.stdout
    assert png_path.is_file()
    assert png_path.stat().st_size > 1000
    assert out_file.is_file()

    # 2. Replay attack scenario with JSON output
    res_replay = runner.invoke(
        app,
        ["csms", "simulate", "--scenario", "replay_attack", "--frames", "30", "--json"],
    )
    assert res_replay.exit_code == 0
    data_replay = json.loads(res_replay.stdout)
    assert data_replay["threat_assessment"]["anomalous_frames_detected"] > 0
    assert data_replay["report_id"] == "VSOC-SIM-REPLAY_ATTACK"

    # 3. Nominal scenario (0 anomalies)
    res_nominal = runner.invoke(
        app,
        ["csms", "simulate", "--scenario", "nominal", "--frames", "25", "--json"],
    )
    assert res_nominal.exit_code == 0
    data_nom = json.loads(res_nominal.stdout)
    assert data_nom["threat_assessment"]["anomalous_frames_detected"] == 0


def test_cli_csms_render(tmp_path: Path) -> None:
    png_file = tmp_path / "test_csms_render.png"
    out_json = tmp_path / "render_meta.json"

    res = runner.invoke(
        app,
        [
            "csms",
            "render",
            "--png",
            str(png_file),
            "--scenario",
            "dos_flood",
            "--output",
            str(out_json),
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] == "rendered"
    assert png_file.is_file()
    assert png_file.stat().st_size > 1000
    assert out_json.is_file()


def test_cli_csms_audit(tmp_path: Path) -> None:
    # 1. Plain text audit
    res = runner.invoke(app, ["csms", "audit"])
    assert res.exit_code == 0
    assert "UN R155 CSMS Regulatory Compliance Audit: [COMPLIANT]" in res.stdout
    assert "[PASS] un_r155_csms_rate_bounds_configured" in res.stdout
    assert "[PASS] defensive_containment_quarantine_enabled" in res.stdout

    # 2. JSON audit and file output
    out_file = tmp_path / "audit.json"
    res_json = runner.invoke(app, ["csms", "audit", "--json", "--output", str(out_file)])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["compliant"] is True
    assert data["checks_total"] == 5
    assert data["checks_passed"] == 5
    assert out_file.is_file()
