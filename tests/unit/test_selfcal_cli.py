"""Unit tests for the nearfield360 selfcal CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from typer.testing import CliRunner

from nearfield360.cli.app import app

runner = CliRunner()


def test_cli_selfcal_info(tmp_path: Path) -> None:
    res = runner.invoke(app, ["selfcal", "info"])
    assert res.exit_code == 0
    assert "Extrinsic Self-Calibration" in res.stdout

    json_res = runner.invoke(app, ["selfcal", "info", "--json"])
    data = json.loads(json_res.stdout)
    assert data["warning_deg"] == 0.3
    assert data["fault_deg"] == 3.0

    out_file = tmp_path / "info.json"
    assert runner.invoke(app, ["selfcal", "info", "-o", str(out_file)]).exit_code == 0
    assert json.loads(out_file.read_text(encoding="utf-8"))["min_correspondences"] == 12


def test_cli_selfcal_estimate_recovers_truth_and_writes_artifact(tmp_path: Path) -> None:
    out_file = tmp_path / "estimate.json"

    res = runner.invoke(
        app,
        [
            "selfcal",
            "estimate",
            "--outliers",
            "0.2",
            "--points",
            "60",
            "--json",
            "-o",
            str(out_file),
        ],
    )

    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] == "WARNING"
    assert data["estimate"]["observable"] is True
    assert abs(data["estimate"]["delta"]["roll_rad"] - 0.00873) < 0.001
    assert json.loads(out_file.read_text(encoding="utf-8")) == data


def test_cli_selfcal_estimate_text_and_unobservable() -> None:
    text = runner.invoke(app, ["selfcal", "estimate", "--camera", "RV"])
    assert text.exit_code == 0
    assert "RV" in text.stdout and "RMS" in text.stdout

    sparse = runner.invoke(app, ["selfcal", "estimate", "--points", "5", "--json"])
    assert sparse.exit_code == 0
    payload = json.loads(sparse.stdout)
    assert payload["status"] == "UNOBSERVABLE"
    assert payload["estimate"]["sigma_deg"] is None


def test_cli_selfcal_estimate_rejects_bad_options() -> None:
    assert runner.invoke(app, ["selfcal", "estimate", "--camera", "XX"]).exit_code != 0
    assert runner.invoke(app, ["selfcal", "estimate", "--outliers", "1.5"]).exit_code != 0


def test_cli_selfcal_monitor_flags_only_the_drifting_camera() -> None:
    res = runner.invoke(
        app, ["selfcal", "monitor", "--camera", "MVR", "--pitch", "4.0", "--frames", "6", "--json"]
    )

    assert res.exit_code == 0
    reports = json.loads(res.stdout)["reports"]
    assert reports["MVR"]["status"] in {"RECALIBRATE", "FAULT"}
    assert {reports[c]["status"] for c in ("FV", "RV", "MVL")} == {"NOMINAL"}
    history = json.loads(res.stdout)["history_deg"]["MVR"]
    assert history == sorted(history) and len(history) == 6


def test_cli_selfcal_monitor_table_output() -> None:
    res = runner.invoke(app, ["selfcal", "monitor", "--frames", "3"])

    assert res.exit_code == 0
    assert "Drift monitor" in res.stdout


def test_cli_selfcal_render_writes_png(tmp_path: Path) -> None:
    target = tmp_path / "dash.png"

    res = runner.invoke(app, ["selfcal", "render", "-o", str(target), "--json"])

    assert res.exit_code == 0
    assert json.loads(res.stdout)["canvas_shape"] == [720, 1280, 3]
    image = cv2.imread(str(target))
    assert image is not None and image.shape == (720, 1280, 3)

    plain = runner.invoke(app, ["selfcal", "render", "-o", str(tmp_path / "plain.png")])
    assert plain.exit_code == 0 and "1280x720" in plain.stdout


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_selfcal_audit_is_compliant(as_json: bool, tmp_path: Path) -> None:
    args = ["selfcal", "audit", "-o", str(tmp_path / "audit.json")]
    res = runner.invoke(app, [*args, "--json"] if as_json else args)

    assert res.exit_code == 0
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["compliant"] is True
    assert saved["passed_checks"] == saved["total_checks"] == 9
    assert ("COMPLIANT" in res.stdout) or as_json


def test_release_audit_includes_selfcal_group() -> None:
    res = runner.invoke(app, ["release", "audit"])

    assert res.exit_code == 0
    assert "selfcal" in res.stdout
