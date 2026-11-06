"""Unit tests for nearfield360 ota CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_ota_info(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ota", "info"])
    assert result.exit_code == 0
    assert "UN R156 Software Update Management System (SUMS) Configuration:" in result.stdout
    assert "Minimum Battery SoC" in result.stdout
    assert "Anti-Rollback Monotonic" in result.stdout

    # Test JSON output
    json_result = runner.invoke(app, ["ota", "info", "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["min_battery_soc_pct"] == 30.0
    assert payload["max_package_size_mb"] == 500.0
    assert payload["enforce_monotonic_version"] is True

    # Test file output
    out_file = tmp_path / "ota_info.json"
    file_result = runner.invoke(app, ["ota", "info", "--output", str(out_file)])
    assert file_result.exit_code == 0
    assert out_file.is_file()


def test_cli_ota_manifest(tmp_path: Path) -> None:
    out_file = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "ota",
            "manifest",
            "--target-ecu",
            "ecu-front-camera-01",
            "--version",
            "v3.0.0",
            "--revision",
            "50",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "Generated UN R156 Firmware Manifest for [ecu-front-camera-01]:" in result.stdout
    assert "Target Version     : v3.0.0" in result.stdout
    assert "Monotonic Revision : 50" in result.stdout
    assert out_file.is_file()

    # Test JSON output
    json_result = runner.invoke(app, ["ota", "manifest", "--json"])
    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    assert "signature_hex" in data
    assert data["target_ecu"] == "ecu-perception-01"


def test_cli_ota_verify(tmp_path: Path) -> None:
    # Default verification
    result = runner.invoke(app, ["ota", "verify", "--current-revision", "40"])
    assert result.exit_code == 0
    assert "UN R156 Cryptographic Firmware Verification Result:" in result.stdout
    assert "Overall Valid      : True" in result.stdout

    # Verify against written manifest file
    manifest_file = tmp_path / "sample_manifest.json"
    runner.invoke(
        app,
        [
            "ota",
            "manifest",
            "--target-ecu",
            "ecu-perception-01",
            "--version",
            "v2.4.0",
            "--revision",
            "45",
            "--output",
            str(manifest_file),
        ],
    )
    assert manifest_file.is_file()

    verify_result = runner.invoke(
        app,
        [
            "ota",
            "verify",
            "--manifest",
            str(manifest_file),
            "--current-revision",
            "40",
            "--json",
        ],
    )
    assert verify_result.exit_code == 0
    payload = json.loads(verify_result.stdout)
    assert payload["valid"] is True
    assert payload["manifest_valid"] is True
    assert payload["anti_rollback_valid"] is True


def test_cli_ota_preflight(tmp_path: Path) -> None:
    # Nominal pass
    result = runner.invoke(
        app,
        [
            "ota",
            "preflight",
            "--gear",
            "PARK",
            "--speed",
            "0.0",
            "--soc",
            "80.0",
        ],
    )
    assert result.exit_code == 0
    assert "UN R156 Safety Preflight Interlock Check: [READY]" in result.stdout
    assert "Gear Is Park       : True" in result.stdout

    # Blocked by speed and gear
    out_blocked = tmp_path / "blocked.json"
    blocked_result = runner.invoke(
        app,
        [
            "ota",
            "preflight",
            "--gear",
            "DRIVE",
            "--speed",
            "12.5",
            "--soc",
            "15.0",
            "--json",
            "-o",
            str(out_blocked),
        ],
    )
    assert blocked_result.exit_code == 0
    assert out_blocked.is_file()
    data = json.loads(out_blocked.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert len(data["reasons"]) >= 2


def test_cli_ota_apply(tmp_path: Path) -> None:
    out_json = tmp_path / "ota_campaign.json"
    out_png = tmp_path / "ota_campaign_hud.png"

    # Nominal update success
    result = runner.invoke(
        app,
        [
            "ota",
            "apply",
            "--campaign-id",
            "CAMP-TEST-001",
            "--target-ecu",
            "ecu-perception-01",
            "--version",
            "v2.4.0",
            "--revision",
            "42",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
        ],
    )
    assert result.exit_code == 0
    assert "UN R156 OTA Campaign Execution Report: [CAMP-TEST-001]" in result.stdout
    assert "Final State        : COMMITTED" in result.stdout
    assert "Campaign Succeeded : True" in result.stdout
    assert out_json.is_file()
    assert out_png.is_file()

    # Simulated post-boot failure with automated rollback
    fail_json = tmp_path / "ota_rollback.json"
    fail_result = runner.invoke(
        app,
        [
            "ota",
            "apply",
            "--campaign-id",
            "CAMP-TEST-FAIL",
            "--simulate-failure",
            "--json",
            "-o",
            str(fail_json),
        ],
    )
    assert fail_result.exit_code == 0
    data = json.loads(fail_json.read_text(encoding="utf-8"))
    assert data["active_state"] == "rolled_back"
    assert data["success"] is False
    assert data["partition_state"]["rollback_count"] == 1


def test_cli_ota_render(tmp_path: Path) -> None:
    out_png = tmp_path / "rendered_hud.png"
    out_json = tmp_path / "render_meta.json"
    result = runner.invoke(
        app,
        [
            "ota",
            "render",
            "--png",
            str(out_png),
            "--output",
            str(out_json),
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert out_png.is_file()
    assert out_json.is_file()
    meta = json.loads(out_json.read_text(encoding="utf-8"))
    assert meta["status"] == "rendered"


def test_cli_ota_audit(tmp_path: Path) -> None:
    out_audit = tmp_path / "ota_audit.json"
    result = runner.invoke(app, ["ota", "audit", "--output", str(out_audit)])
    assert result.exit_code == 0
    assert "UN R156 SUMS Regulatory Audit: [COMPLIANT]" in result.stdout
    assert "[PASS] un_r156_sums_configured" in result.stdout
    assert "[PASS] cryptographic_signatures_enabled" in result.stdout
    assert "[PASS] anti_rollback_monotonic_guard" in result.stdout
    assert "[PASS] safety_preflight_interlocks" in result.stdout
    assert "[PASS] dual_partition_ab_failover" in result.stdout

    assert out_audit.is_file()
    report = json.loads(out_audit.read_text(encoding="utf-8"))
    assert report["compliant"] is True
    assert report["checks_total"] == 5
    assert report["checks_passed"] == 5
