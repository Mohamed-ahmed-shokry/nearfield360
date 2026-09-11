import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app

runner = CliRunner()


def _dataset_with_masks(root: Path) -> None:
    rgb_dir = root / "rgb_images"
    mask_dir = root / "semantic_annotations" / "gtLabels"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    # Two samples with masks and one without, exercising skip accounting.
    for stem, color in (("00001_FV", 1), ("00002_RV", 6)):
        rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
        assert cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb)
        mask = np.full((8, 8), color, dtype=np.uint8)
        assert cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
    rgb_only = np.full((8, 8, 3), 64, dtype=np.uint8)
    assert cv2.imwrite(str(rgb_dir / "00003_FV.png"), rgb_only)


def test_data_viz_writes_overlays_and_reports_skips(tmp_path: Path) -> None:
    _dataset_with_masks(tmp_path)
    output = tmp_path / "viz"

    result = runner.invoke(
        app, ["data", "viz", "--root", str(tmp_path), "--output", str(output), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert sorted(payload["written"]) == ["00001_FV", "00002_RV"]
    assert payload["skipped_no_mask"] == 1
    assert (output / "00001_FV_overlay.png").is_file()
    assert (output / "00002_RV_overlay.png").is_file()


def test_data_viz_skips_existing_without_overwrite(tmp_path: Path) -> None:
    _dataset_with_masks(tmp_path)
    output = tmp_path / "viz"
    assert (
        runner.invoke(
            app, ["data", "viz", "--root", str(tmp_path), "--output", str(output)]
        ).exit_code
        == 0
    )

    second = runner.invoke(
        app, ["data", "viz", "--root", str(tmp_path), "--output", str(output), "--json"]
    )

    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["skipped_existing"] == 2

    overwritten = runner.invoke(
        app,
        ["data", "viz", "--root", str(tmp_path), "--output", str(output), "--overwrite"],
    )
    assert overwritten.exit_code == 0, overwritten.output


def test_data_viz_respects_max_images(tmp_path: Path) -> None:
    _dataset_with_masks(tmp_path)
    output = tmp_path / "viz"

    result = runner.invoke(
        app,
        ["data", "viz", "--root", str(tmp_path), "--output", str(output), "--max-images", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "1 written" in result.stdout
    assert len(list(output.glob("*_overlay.png"))) == 1


def test_data_viz_reports_shape_mismatch(tmp_path: Path) -> None:
    import cv2 as cv2_module
    import numpy as np

    rgb_dir = tmp_path / "rgb_images"
    mask_dir = tmp_path / "semantic_annotations" / "gtLabels"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    assert cv2_module.imwrite(str(rgb_dir / "00001_FV.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    assert cv2_module.imwrite(str(mask_dir / "00001_FV.png"), np.zeros((4, 4), dtype=np.uint8))

    result = runner.invoke(
        app, ["data", "viz", "--root", str(tmp_path), "--output", str(tmp_path / "viz")]
    )

    assert result.exit_code == 1
    assert "does not match RGB shape" in result.stderr
