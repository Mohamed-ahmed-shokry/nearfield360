"""Unit tests for nearfield360 digitalkey CLI command group."""

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_digitalkey_info_json() -> None:
    result = runner.invoke(app, ["digitalkey", "info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["enable_uwb_sts"] is True
    assert data["carrier_frequency_mhz"] == 6489.6
    assert data["num_uwb_anchors"] == 4


def test_digitalkey_pair_and_authenticate(tmp_path: Path) -> None:
    tok_file = tmp_path / "token.json"
    vin = "WP0ZZZ99ZTS392001"

    # Pair
    pair_res = runner.invoke(
        app,
        [
            "digitalkey",
            "pair",
            "--vin",
            vin,
            "--user",
            "usr-test",
            "--device",
            "dev-iphone",
            "--scopes",
            "AVP_OPERATE,CABIN_LOCK",
            "--output",
            str(tok_file),
        ],
    )
    assert pair_res.exit_code == 0
    assert tok_file.is_file()

    # Authenticate valid
    auth_res = runner.invoke(
        app,
        [
            "digitalkey",
            "authenticate",
            "--token-file",
            str(tok_file),
            "--vin",
            vin,
            "--scope",
            "AVP_OPERATE",
            "--json",
        ],
    )
    assert auth_res.exit_code == 0
    auth_data = json.loads(auth_res.stdout)
    assert auth_data["is_valid"] is True

    # Authenticate invalid scope (e.g. TRUNK_ACCESS not granted)
    auth_res_bad = runner.invoke(
        app,
        [
            "digitalkey",
            "authenticate",
            "--token-file",
            str(tok_file),
            "--vin",
            vin,
            "--scope",
            "TRUNK_ACCESS",
            "--json",
        ],
    )
    assert auth_res_bad.exit_code == 2
    bad_data = json.loads(auth_res_bad.stdout)
    assert bad_data["is_valid"] is False


def test_digitalkey_range_and_render_png(tmp_path: Path) -> None:
    out_json = tmp_path / "report.json"
    out_png = tmp_path / "hud.png"

    result = runner.invoke(
        app,
        [
            "digitalkey",
            "range",
            "--x",
            "2.2",
            "--y",
            "1.0",
            "--z",
            "1.0",
            "--output",
            str(out_json),
            "--png",
            str(out_png),
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    assert out_png.is_file()
    assert out_png.stat().st_size > 5000

    data = json.loads(result.stdout)
    assert "device_position" in data
    assert data["device_position"]["confidence"] > 0.7


def test_digitalkey_interlock_touch_and_relay(tmp_path: Path) -> None:
    # 1. Close distance in TOUCH zone -> standstill command active
    res1 = runner.invoke(app, ["digitalkey", "interlock", "--distance", "1.0", "--json"])
    assert res1.exit_code == 0
    data1 = json.loads(res1.stdout)
    assert data1["standstill_command"] is True
    assert data1["proximity_zone"] == "TOUCH"

    # 2. Simulated relay attack -> emergency standstill command active
    res2 = runner.invoke(
        app,
        ["digitalkey", "interlock", "--distance", "5.0", "--relay-attack", "--json"],
    )
    assert res2.exit_code == 0
    data2 = json.loads(res2.stdout)
    assert data2["standstill_command"] is True
    assert "Relay attack" in data2["reason"]


def test_digitalkey_handover_actions() -> None:
    for action in ("delegate", "pickup-ready", "takeover", "manual"):
        res = runner.invoke(app, ["digitalkey", "handover", "--action", action, "--json"])
        assert res.exit_code == 0
        data = json.loads(res.stdout)
        assert data["action"] == action
        assert data["success"] is True


def test_digitalkey_render_command(tmp_path: Path) -> None:
    out_png = tmp_path / "dashboard.png"
    res = runner.invoke(app, ["digitalkey", "render", "--png", str(out_png), "--json"])
    assert res.exit_code == 0
    assert out_png.is_file()


def test_digitalkey_audit_command() -> None:
    res = runner.invoke(app, ["digitalkey", "audit", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["overall_status"] == "PASS"
    assert data["total_checks"] >= 4
