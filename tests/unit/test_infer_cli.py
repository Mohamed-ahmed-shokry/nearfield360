"""Unit tests for 'nearfield360 infer' CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from nearfield360.cli.app import app
from nearfield360.perception.inference.test_utils import (
    create_dummy_detection_onnx,
    create_dummy_segmentation_onnx,
)

runner = CliRunner()


@pytest.fixture
def dummy_seg_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "models" / "seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)
    return model_path


@pytest.fixture
def dummy_det_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "models" / "det.onnx"
    create_dummy_detection_onnx(model_path, num_classes=5, num_boxes=3, height=32, width=32)
    return model_path


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "images" / "fisheye.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_infer_help() -> None:
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "semantic" in result.stdout
    assert "detection" in result.stdout
    assert "benchmark" in result.stdout
    assert "inspect" in result.stdout


def test_infer_semantic_success(dummy_seg_model: Path, sample_image: Path, tmp_path: Path) -> None:
    out_mask_png = tmp_path / "output" / "mask.png"
    out_json = tmp_path / "output" / "summary.json"

    # Test PNG mask output
    result = runner.invoke(
        app,
        [
            "infer",
            "semantic",
            "--model",
            str(dummy_seg_model),
            "--image",
            str(sample_image),
            "--output",
            str(out_mask_png),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert out_mask_png.is_file()
    assert "Segmentation complete" in result.stdout

    # Test JSON summary output with --json flag
    result = runner.invoke(
        app,
        [
            "infer",
            "semantic",
            "--model",
            str(dummy_seg_model),
            "--image",
            str(sample_image),
            "--output",
            str(out_json),
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    data = json.loads(result.stdout)
    assert "latency_ms" in data
    assert "classes_present" in data


def test_infer_semantic_invalid_device(dummy_seg_model: Path, sample_image: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "semantic",
            "--model",
            str(dummy_seg_model),
            "--image",
            str(sample_image),
            "--device",
            "invalid_device",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid device" in result.stderr or "Invalid device" in result.stdout


def test_infer_detection_success(dummy_det_model: Path, sample_image: Path, tmp_path: Path) -> None:
    out_det_png = tmp_path / "output" / "det.png"
    out_json = tmp_path / "output" / "detections.json"

    # Test annotated image output
    result = runner.invoke(
        app,
        [
            "infer",
            "detection",
            "--model",
            str(dummy_det_model),
            "--image",
            str(sample_image),
            "--output",
            str(out_det_png),
            "--confidence-threshold",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    assert out_det_png.is_file()
    assert "Detection complete" in result.stdout

    # Test JSON output
    result = runner.invoke(
        app,
        [
            "infer",
            "detection",
            "--model",
            str(dummy_det_model),
            "--image",
            str(sample_image),
            "--output",
            str(out_json),
            "--confidence-threshold",
            "0.1",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    data = json.loads(result.stdout)
    assert "detections" in data
    assert "latency_ms" in data


def test_infer_benchmark_success(dummy_seg_model: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "output" / "bench.json"
    result = runner.invoke(
        app,
        [
            "infer",
            "benchmark",
            "--model",
            str(dummy_seg_model),
            "--iterations",
            "5",
            "--warmup",
            "2",
            "--height",
            "32",
            "--width",
            "32",
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    assert "Benchmark Complete" in result.stdout
    with out_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["iterations"] == 5
    assert data["fps"] > 0


def test_infer_inspect_success(dummy_seg_model: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "output" / "meta.json"
    result = runner.invoke(
        app,
        [
            "infer",
            "inspect",
            "--model",
            str(dummy_seg_model),
            "--output",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    data = json.loads(result.stdout)
    assert data["backend"] == "opencv"


def test_infer_parity_success(dummy_seg_model: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "parity",
            "--model-a",
            str(dummy_seg_model),
            "--model-b",
            str(dummy_seg_model),
            "--height",
            "32",
            "--width",
            "32",
        ],
    )
    assert result.exit_code == 0
    assert "Parity check PASSED" in result.stdout
    assert "Max Absolute Diff" in result.stdout


def test_infer_parity_fails_on_different_models(
    dummy_seg_model: Path, dummy_det_model: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "parity",
            "--model-a",
            str(dummy_seg_model),
            "--model-b",
            str(dummy_det_model),
            "--height",
            "32",
            "--width",
            "32",
        ],
    )
    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "Parity comparison failed" in output


def test_infer_parity_invalid_device(dummy_seg_model: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "parity",
            "--model-a",
            str(dummy_seg_model),
            "--model-b",
            str(dummy_seg_model),
            "--device",
            "invalid_device",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid device" in result.stderr or "Invalid device" in result.stdout


def test_infer_parity_in_help() -> None:
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "parity" in result.stdout


def test_infer_semantic_accepts_backend_flag(dummy_seg_model: Path, sample_image: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "semantic",
            "--model",
            str(dummy_seg_model),
            "--image",
            str(sample_image),
            "--backend",
            "opencv",
        ],
    )
    assert result.exit_code == 0
    assert "Segmentation complete" in result.stdout


def test_infer_detection_accepts_backend_flag(dummy_det_model: Path, sample_image: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "detection",
            "--model",
            str(dummy_det_model),
            "--image",
            str(sample_image),
            "--backend",
            "opencv",
            "--confidence-threshold",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    assert "Detection complete" in result.stdout


def test_infer_semantic_onnxruntime_backend_fails_cleanly(
    dummy_seg_model: Path, sample_image: Path
) -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        result = runner.invoke(
            app,
            [
                "infer",
                "semantic",
                "--model",
                str(dummy_seg_model),
                "--image",
                str(sample_image),
                "--backend",
                "onnxruntime",
            ],
        )
        assert result.exit_code == 1
        output = result.stdout + result.stderr
        assert "onnxruntime is not installed" in output
    else:
        pytest.skip("onnxruntime installed; missing-package path not testable")


def test_infer_inspect_accepts_backend_flag(dummy_seg_model: Path) -> None:
    result = runner.invoke(
        app,
        [
            "infer",
            "inspect",
            "--model",
            str(dummy_seg_model),
            "--backend",
            "opencv",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["backend"] == "opencv"
