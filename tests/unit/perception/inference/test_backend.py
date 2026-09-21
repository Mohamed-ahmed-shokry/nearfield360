from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nearfield360.config import InferenceConfig
from nearfield360.perception.inference.backend import (
    InferenceBackend,
    InferenceError,
    OpenCVDNNBackend,
    create_backend,
)
from nearfield360.perception.inference.models import InferenceDevice
from nearfield360.perception.inference.test_utils import (
    create_dummy_detection_onnx,
    create_dummy_segmentation_onnx,
)


def test_backend_missing_model_raises(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.onnx"
    with pytest.raises(InferenceError, match="Model file not found"):
        OpenCVDNNBackend(missing)


def test_backend_corrupt_model_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an onnx file")
    with pytest.raises(InferenceError, match="OpenCV failed to parse ONNX"):
        OpenCVDNNBackend(corrupt)


def test_backend_segmentation_forward(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)

    backend = OpenCVDNNBackend(model_path, device=InferenceDevice.CPU)
    assert isinstance(backend, InferenceBackend)

    meta = backend.metadata
    assert meta.backend == "opencv"
    assert meta.device == "cpu"
    assert "output" in meta.output_names

    backend.warmup(iterations=2, sample_shape=(1, 3, 32, 32))

    dummy_input = np.ones((1, 3, 32, 32), dtype=np.float32)
    output = backend.forward(dummy_input)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 10, 32, 32)
    assert output.dtype == np.float32


def test_backend_detection_forward(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_det.onnx"
    create_dummy_detection_onnx(model_path, num_classes=5, num_boxes=8, height=32, width=32)

    backend = OpenCVDNNBackend(model_path, device=InferenceDevice.CPU)
    output = backend.forward(np.zeros((1, 3, 32, 32), dtype=np.float32))

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 8, 9)


def test_backend_forward_input_validation(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)
    backend = OpenCVDNNBackend(model_path)

    with pytest.raises(InferenceError, match="must be a numpy ndarray"):
        backend.forward([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(InferenceError, match="must have 4 dimensions"):
        backend.forward(np.zeros((3, 32, 32), dtype=np.float32))


def test_create_backend_factory(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)

    config = InferenceConfig(backend="opencv", device="cpu")
    backend = create_backend(config, model_path)
    assert isinstance(backend, OpenCVDNNBackend)


def test_backend_cuda_fallback_graceful(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_seg.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=10, height=32, width=32)

    # CUDA device may fail on environments without CUDA-enabled OpenCV, must fallback or succeed
    backend = OpenCVDNNBackend(model_path, device=InferenceDevice.CUDA)
    out = backend.forward(np.zeros((1, 3, 32, 32), dtype=np.float32))
    assert out.shape == (1, 10, 32, 32)
