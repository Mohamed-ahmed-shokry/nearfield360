import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app

runner = CliRunner()


def _write_rgb(root: Path, name: str = "00001_FV.png") -> None:
    image_dir = root / "rgb_images"
    image_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / name), np.zeros((2, 3, 3), dtype=np.uint8))


def test_data_verify_requires_a_configured_root() -> None:
    result = runner.invoke(app, ["data", "verify"])

    assert result.exit_code == 2
    assert "Dataset root is not configured" in result.stderr


def test_data_verify_accepts_root_override(tmp_path: Path) -> None:
    _write_rgb(tmp_path)

    result = runner.invoke(app, ["data", "verify", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "WoodScape dataset: VALID" in result.stdout
    assert "Samples: 1" in result.stdout
    assert "FV=1" in result.stdout


def test_data_verify_uses_environment_config_and_emits_json(tmp_path: Path) -> None:
    _write_rgb(tmp_path, name="00001_RV.png")

    result = runner.invoke(
        app,
        ["data", "verify", "--json"],
        env={"NEARFIELD360_PATHS__DATASET_ROOT": str(tmp_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["sample_count"] == 1
    assert payload["camera_counts"]["RV"] == 1
    assert payload["available"] == {
        "calibrations": 0,
        "previous_images": 0,
        "semantic_masks": 0,
    }


def test_data_verify_returns_one_for_integrity_failure(tmp_path: Path) -> None:
    _write_rgb(tmp_path)

    result = runner.invoke(
        app,
        ["data", "verify", "--root", str(tmp_path), "--require-semantic"],
    )

    assert result.exit_code == 1
    assert "WoodScape dataset: INVALID" in result.stdout
    assert "missing_semantic_mask" in result.stdout


def test_data_verify_reports_truncated_findings(tmp_path: Path) -> None:
    _write_rgb(tmp_path)

    result = runner.invoke(
        app,
        [
            "data",
            "verify",
            "--root",
            str(tmp_path),
            "--require-previous",
            "--require-semantic",
            "--max-issues",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "Additional issues were truncated" in result.stdout


def test_data_verify_returns_two_for_invalid_layout(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "verify", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "Dataset discovery error:" in result.stderr
