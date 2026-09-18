"""Unit tests for nearfield360 robustness CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from nearfield360.cli import app
from nearfield360.utils.artifacts import read_json

runner = CliRunner()

_DOWN_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def _write_dataset(root: Path, *, camera: str = "FV", with_annotations: bool = True) -> None:
    (root / "rgb_images").mkdir(parents=True, exist_ok=True)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / f"rgb_images/00001_{camera}.png"), img)
    if with_annotations:
        (root / "semantic_annotations/gtLabels").mkdir(parents=True, exist_ok=True)
        mask = np.full((20, 20), fill_value=6, dtype=np.uint8)
        mask[0:5, :] = 1
        assert cv2.imwrite(str(root / f"semantic_annotations/gtLabels/00001_{camera}.png"), mask)
        calib_file = root / f"calibration_data/00001_{camera}.json"
        calib_file.parent.mkdir(parents=True, exist_ok=True)
        calib_file.write_text(
            json.dumps(
                {
                    "extrinsic": {
                        "quaternion": list(_DOWN_QUATERNION),
                        "translation": [0.0, 0.0, 1.0],
                    },
                    "intrinsic": {
                        "aspect_ratio": 1.0,
                        "cx_offset": 0.0,
                        "cy_offset": 0.0,
                        "height": 20,
                        "k1": 100.0,
                        "k2": 0.0,
                        "k3": 0.0,
                        "k4": 0.0,
                        "model": "radial_poly",
                        "poly_order": 4,
                        "width": 20,
                    },
                    "name": camera,
                }
            ),
            encoding="utf-8",
        )


def test_robustness_help() -> None:
    result = runner.invoke(app, ["robustness", "--help"])
    assert result.exit_code == 0
    assert "corrupt" in result.stdout
    assert "perturb-calibration" in result.stdout
    assert "benchmark" in result.stdout
    assert "plot" in result.stdout


def test_robustness_corrupt_command(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    out_path = tmp_path / "corrupted.png"
    cv2.imwrite(str(img_path), np.full((32, 32, 3), 128, dtype=np.uint8))

    result = runner.invoke(
        app,
        [
            "robustness",
            "corrupt",
            "--image",
            str(img_path),
            "--type",
            "fog",
            "--severity",
            "2",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()
    corrupted_img = cv2.imread(str(out_path))
    assert corrupted_img is not None
    assert corrupted_img.shape == (32, 32, 3)

    overwrite_fail = runner.invoke(
        app,
        [
            "robustness",
            "corrupt",
            "--image",
            str(img_path),
            "--output",
            str(out_path),
        ],
    )
    assert overwrite_fail.exit_code == 1
    assert "already exists" in overwrite_fail.stderr

    overwrite_success = runner.invoke(
        app,
        [
            "robustness",
            "corrupt",
            "--image",
            str(img_path),
            "--output",
            str(out_path),
            "--overwrite",
        ],
    )
    assert overwrite_success.exit_code == 0


def test_robustness_corrupt_missing_image(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "robustness",
            "corrupt",
            "--image",
            str(tmp_path / "nonexistent.png"),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert result.exit_code != 0


def test_robustness_perturb_calibration_command(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    calib_in = tmp_path / "calibration_data/00001_FV.json"
    calib_out = tmp_path / "perturbed.json"

    result = runner.invoke(
        app,
        [
            "robustness",
            "perturb-calibration",
            "--calibration",
            str(calib_in),
            "--output",
            str(calib_out),
            "--roll",
            "2.5",
            "--pitch",
            "-1.0",
            "--yaw",
            "4.0",
            "--dx",
            "0.1",
            "--dy",
            "-0.05",
            "--dz",
            "0.02",
        ],
    )
    assert result.exit_code == 0
    assert "Wrote perturbed calibration" in result.stdout

    data = read_json(calib_out)
    assert data["name"] == "FV"
    assert data["extrinsic"]["translation"] != [0.0, 0.0, 1.0]

    refuse_result = runner.invoke(
        app,
        [
            "robustness",
            "perturb-calibration",
            "--calibration",
            str(calib_in),
            "--output",
            str(calib_out),
        ],
    )
    assert refuse_result.exit_code == 1
    assert "Artifact error" in refuse_result.stderr


def test_robustness_benchmark_command(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    report_path = tmp_path / "reports/robustness_benchmark.json"

    result = runner.invoke(
        app,
        [
            "robustness",
            "benchmark",
            "--root",
            str(tmp_path),
            "--camera",
            "FV",
            "--samples",
            "1",
            "--output",
            str(report_path),
        ],
    )
    assert result.exit_code == 0
    assert "Robustness benchmark complete" in result.stdout

    report = read_json(report_path)
    assert "environment" in report
    assert "config" in report
    assert len(report["corruption_sweeps"]) > 0
    assert len(report["calibration_sweeps"]) > 0

    dup_result = runner.invoke(
        app,
        [
            "robustness",
            "benchmark",
            "--root",
            str(tmp_path),
            "--output",
            str(report_path),
        ],
    )
    assert dup_result.exit_code == 1


def test_robustness_benchmark_no_annotations(tmp_path: Path) -> None:
    _write_dataset(tmp_path, with_annotations=False)
    report_path = tmp_path / "reports/robustness_benchmark.json"

    result = runner.invoke(
        app,
        [
            "robustness",
            "benchmark",
            "--root",
            str(tmp_path),
            "--output",
            str(report_path),
        ],
    )
    assert result.exit_code == 1
    assert "No valid annotated samples found" in result.stderr


def test_robustness_plot_command(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    report_path = tmp_path / "robustness_report.json"
    plot_dir = tmp_path / "plots"

    bench_res = runner.invoke(
        app,
        [
            "robustness",
            "benchmark",
            "--root",
            str(tmp_path),
            "--output",
            str(report_path),
        ],
    )
    assert bench_res.exit_code == 0

    plot_res = runner.invoke(
        app,
        [
            "robustness",
            "plot",
            "--report",
            str(report_path),
            "--output-dir",
            str(plot_dir),
        ],
    )
    assert plot_res.exit_code == 0
    assert (plot_dir / "index.html").exists()
    assert (plot_dir / "corruption_degradation.svg").exists()
    assert (plot_dir / "calibration_sensitivity.svg").exists()
    assert (plot_dir / "corruption_degradation.png").exists()
    assert (plot_dir / "calibration_sensitivity.png").exists()

    plot_dir_svg_only = tmp_path / "plots_svg"
    plot_res2 = runner.invoke(
        app,
        [
            "robustness",
            "plot",
            "--report",
            str(report_path),
            "--output-dir",
            str(plot_dir_svg_only),
            "--no-html",
            "--no-png",
        ],
    )
    assert plot_res2.exit_code == 0
    assert not (plot_dir_svg_only / "index.html").exists()
    assert not (plot_dir_svg_only / "corruption_degradation.png").exists()
    assert (plot_dir_svg_only / "corruption_degradation.svg").exists()
