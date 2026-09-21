"""Tests for inference benchmarking and numerical parity verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nearfield360.perception.inference.backend import OpenCVDNNBackend
from nearfield360.perception.inference.benchmark import (
    ParityComparison,
    benchmark_inference,
    compare_numerical_parity,
    verify_numerical_parity,
)
from nearfield360.perception.inference.models import BenchmarkSummary
from nearfield360.perception.inference.test_utils import create_dummy_segmentation_onnx


@pytest.fixture
def dummy_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "dummy_seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=5, height=32, width=32)
    return model_path


def test_compare_numerical_parity_success() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.00001, 2.0, 2.99999], dtype=np.float32)

    comp = compare_numerical_parity(a, b, atol=1e-4, rtol=1e-4)
    assert isinstance(comp, ParityComparison)
    assert comp.is_match is True
    assert comp.max_abs_diff < 1e-4
    assert verify_numerical_parity(a, b, atol=1e-4, rtol=1e-4) is True


def test_compare_numerical_parity_mismatch() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.5, 2.0, 3.0], dtype=np.float32)

    comp = compare_numerical_parity(a, b, atol=1e-2, rtol=1e-2)
    assert comp.is_match is False
    assert comp.max_abs_diff == pytest.approx(0.5, abs=1e-5)
    assert verify_numerical_parity(a, b, atol=1e-2, rtol=1e-2) is False


def test_compare_numerical_parity_tuples() -> None:
    a1 = np.ones((2, 2), dtype=np.float32)
    a2 = np.zeros((3,), dtype=np.float32)
    b1 = np.ones((2, 2), dtype=np.float32)
    b2 = np.zeros((3,), dtype=np.float32)

    comp = compare_numerical_parity((a1, a2), (b1, b2))
    assert comp.is_match is True
    assert comp.max_abs_diff == 0.0

    # Length mismatch
    with pytest.raises(ValueError, match="Output tuple length mismatch"):
        compare_numerical_parity((a1, a2), (b1,))


def test_compare_numerical_parity_errors() -> None:
    # Type error
    with pytest.raises(TypeError, match="Expected numpy arrays or tuples of arrays"):
        compare_numerical_parity("not an array", np.zeros((2,)))  # type: ignore[arg-type]

    # Shape mismatch
    with pytest.raises(ValueError, match="Array shape mismatch"):
        compare_numerical_parity(np.zeros((2, 2)), np.zeros((3, 3)))


def test_benchmark_inference_validation(dummy_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_model)
    with pytest.raises(ValueError, match="iterations must be positive"):
        benchmark_inference(backend, iterations=0)
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        benchmark_inference(backend, warmup=-1)


def test_benchmark_inference_run(dummy_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_model)
    summary = benchmark_inference(
        backend,
        input_shape=(1, 3, 32, 32),
        iterations=5,
        warmup=2,
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.iterations == 5
    assert summary.warmup == 2
    assert summary.input_shape == (1, 3, 32, 32)
    assert summary.mean_latency_ms > 0.0
    assert summary.median_latency_ms > 0.0
    assert summary.fps > 0.0
    assert summary.min_latency_ms <= summary.max_latency_ms
    assert summary.backend == "opencv"
    assert summary.device == "cpu"


def test_benchmark_inference_with_dummy_input(dummy_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_model)
    dummy_tensor = np.zeros((1, 3, 32, 32), dtype=np.float32)
    summary = benchmark_inference(
        backend,
        dummy_input=dummy_tensor,
        iterations=3,
        warmup=1,
    )
    assert summary.iterations == 3
    assert summary.warmup == 1
    assert summary.input_shape == (1, 3, 32, 32)
