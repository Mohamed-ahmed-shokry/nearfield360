from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nearfield360 import __version__
from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


def test_release_help() -> None:
    result = runner.invoke(app, ["release", "--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout


def test_release_audit_help() -> None:
    result = runner.invoke(app, ["release", "audit", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.stdout
    assert "--output" in result.stdout


def test_release_audit_passes_on_repository(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"

    result = runner.invoke(
        app,
        ["release", "audit", "--json", "--output", str(output)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["failed"] == 0
    assert payload["nearfield360_version"] == __version__
    names = {c["name"] for c in payload["checks"]}
    assert {
        "project_root",
        "license_file",
        "project_name",
        "version_consistency",
        "license_metadata",
        "readme_metadata",
        "console_script",
        "readme_exists",
        "py_typed",
        "default_config",
        "cli_surface",
    } <= names
    assert all(c["status"] == "pass" for c in payload["checks"])

    written = read_json(output)
    assert written["ok"] is True
    assert written["passed"] == payload["passed"]


def test_release_audit_human_output(tmp_path: Path) -> None:
    result = runner.invoke(app, ["release", "audit"])

    assert result.exit_code == 0
    assert "[PASS]" in result.stdout
    assert "Release audit passed." in result.stdout
    assert "cli_surface" in result.stdout


def test_release_audit_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"

    first = runner.invoke(
        app,
        ["release", "audit", "--json", "--output", str(output)],
    )
    second = runner.invoke(
        app,
        ["release", "audit", "--json", "--output", str(output)],
    )
    third = runner.invoke(
        app,
        ["release", "audit", "--json", "--output", str(output), "--overwrite"],
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Artifact error" in second.stderr
    assert third.exit_code == 0


def test_release_audit_detects_version_mismatch(monkeypatch: object) -> None:
    import nearfield360.cli.release as release_mod

    monkeypatch.setattr(release_mod, "__version__", "9.9.9")

    result = runner.invoke(app, ["release", "audit", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    failed = [c for c in payload["checks"] if c["status"] == "fail"]
    assert any(c["name"] == "version_consistency" for c in failed)
