from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()


def _create_sharp_image(path: Path) -> None:
    # High-frequency checkerboard pattern with contrast
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.sin(yy) > 0).astype(np.uint8) * 200 + 25
    bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), bgr)


def _create_blurred_image(path: Path) -> None:
    # Extremely blurred / low contrast image
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_health_help() -> None:
    result = runner.invoke(app, ["health", "--help"])
    assert result.exit_code == 0
    assert "assess" in result.stdout


def test_health_assess_help() -> None:
    result = runner.invoke(app, ["health", "assess", "--help"])
    assert result.exit_code == 0
    assert "--image" in result.stdout
    assert "--json" in result.stdout
    assert "--output" in result.stdout


def test_health_assess_human_readable(tmp_path: Path) -> None:
    img_path = tmp_path / "camera_fv.png"
    _create_sharp_image(img_path)

    result = runner.invoke(
        app,
        ["health", "assess", "--image", str(img_path), "--camera", "FV"],
    )
    assert result.exit_code == 0
    assert "Camera FV: HEALTHY" in result.stdout
    assert "Confidence: 1.00" in result.stdout
    assert "Discount: 1.00" in result.stdout
    assert "Blur score:" in result.stdout


def test_health_assess_json_stdout(tmp_path: Path) -> None:
    img_path = tmp_path / "camera_rv.png"
    _create_sharp_image(img_path)

    result = runner.invoke(
        app,
        ["health", "assess", "--image", str(img_path), "--camera", "RV", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["camera"] == "RV"
    assert data["status"] == "healthy"
    assert data["discount_weight"] == 1.0
    assert "metrics" in data
    assert data["metrics"]["blur_score"] > 0


def test_health_assess_writes_output_and_protects_overwrite(tmp_path: Path) -> None:
    img_path = tmp_path / "camera_mvl.png"
    _create_sharp_image(img_path)
    output_path = tmp_path / "health_report.json"

    result = runner.invoke(
        app,
        ["health", "assess", "--image", str(img_path), "--output", str(output_path)],
    )
    assert result.exit_code == 0
    assert output_path.is_file()
    saved = read_json(output_path)
    assert saved["status"] == "healthy"

    # Refuse overwrite without flag
    result_refuse = runner.invoke(
        app,
        ["health", "assess", "--image", str(img_path), "--output", str(output_path)],
    )
    assert result_refuse.exit_code == 1
    assert "Artifact error" in result_refuse.stderr

    # Succeeds with overwrite
    result_overwrite = runner.invoke(
        app,
        [
            "health",
            "assess",
            "--image",
            str(img_path),
            "--output",
            str(output_path),
            "--overwrite",
        ],
    )
    assert result_overwrite.exit_code == 0


def test_health_assess_detects_degradation(tmp_path: Path) -> None:
    img_path = tmp_path / "camera_blurred.png"
    _create_blurred_image(img_path)

    result = runner.invoke(
        app,
        ["health", "assess", "--image", str(img_path), "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] != "healthy"
    assert data["discount_weight"] < 1.0


def test_health_assess_missing_image_fails(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.png"
    result = runner.invoke(
        app,
        ["health", "assess", "--image", str(missing)],
    )
    assert result.exit_code != 0
