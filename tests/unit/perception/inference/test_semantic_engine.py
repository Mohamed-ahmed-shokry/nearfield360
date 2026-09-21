"""Tests for SemanticSegmentationEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nearfield360.perception.inference.backend import InferenceError, OpenCVDNNBackend
from nearfield360.perception.inference.preprocessor import FisheyeImagePreprocessor
from nearfield360.perception.inference.semantic import SemanticSegmentationEngine
from nearfield360.perception.inference.test_utils import create_dummy_segmentation_onnx


@pytest.fixture
def dummy_seg_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "dummy_segmentation.onnx"
    create_dummy_segmentation_onnx(model_path, num_classes=5, height=32, width=32)
    return model_path


def test_semantic_engine_properties(dummy_seg_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_seg_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32))
    engine = SemanticSegmentationEngine(backend, preprocessor=preprocessor, num_classes=5)

    assert engine.backend is backend
    assert engine.preprocessor is preprocessor
    assert engine.num_classes == 5


def test_semantic_engine_predict_success(dummy_seg_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_seg_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32), preserve_aspect_ratio=False)
    engine = SemanticSegmentationEngine(backend, preprocessor=preprocessor, num_classes=5)

    # Input image of size (64, 64, 3)
    image = np.ones((64, 64, 3), dtype=np.uint8) * 128
    mask, conf = engine.predict(image)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert (mask < 5).all()

    assert conf.shape == (64, 64)
    assert conf.dtype == np.float32
    assert (conf >= 0.0).all()
    assert (conf <= 1.0).all()


def test_semantic_engine_predict_with_letterbox(dummy_seg_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_seg_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32), preserve_aspect_ratio=True)
    engine = SemanticSegmentationEngine(backend, preprocessor=preprocessor, num_classes=5)

    # Non-square input image
    image = np.full((30, 60, 3), 200, dtype=np.uint8)
    mask, conf = engine.predict(image)

    assert mask.shape == (30, 60)
    assert conf.shape == (30, 60)
    assert (conf >= 0.0).all()
    assert (conf <= 1.0).all()


def test_semantic_engine_class_mismatch(dummy_seg_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_seg_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32))
    # Expect 10 classes, but dummy model outputs 5
    engine = SemanticSegmentationEngine(backend, preprocessor=preprocessor, num_classes=10)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(InferenceError, match="Expected 10 classes, got 5"):
        engine.predict(image)


def test_semantic_engine_invalid_output_shape() -> None:
    mock_backend = MagicMock()
    # Return 2D array instead of 3D/4D
    mock_backend.forward.return_value = np.zeros((32, 32), dtype=np.float32)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32))
    engine = SemanticSegmentationEngine(mock_backend, preprocessor=preprocessor)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(InferenceError, match="Expected 3D or 4D segmentation logits tensor"):
        engine.predict(image)

    # Batch size != 1 for single predict
    mock_backend.forward.return_value = np.zeros((2, 5, 32, 32), dtype=np.float32)
    with pytest.raises(InferenceError, match="Expected single-item batch in logits"):
        engine.predict(image)


def test_semantic_engine_batch_predict(dummy_seg_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_seg_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32))
    engine = SemanticSegmentationEngine(backend, preprocessor=preprocessor, num_classes=5)

    # Empty batch
    assert engine.predict_batch([]) == []

    # Batch of 2
    images = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((40, 50, 3), dtype=np.uint8) * 50,
    ]
    results = engine.predict_batch(images)
    assert len(results) == 2

    mask0, conf0 = results[0]
    mask1, conf1 = results[1]
    assert mask0.shape == (32, 32)
    assert conf0.shape == (32, 32)
    assert mask1.shape == (40, 50)
    assert conf1.shape == (40, 50)


def test_semantic_engine_batch_errors() -> None:
    mock_backend = MagicMock()
    preprocessor = FisheyeImagePreprocessor(target_size=(32, 32))
    engine = SemanticSegmentationEngine(mock_backend, preprocessor=preprocessor, num_classes=5)

    # Batch output shape wrong
    mock_backend.forward.return_value = np.zeros((1, 5, 32, 32), dtype=np.float32)
    images = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8),
    ]
    with pytest.raises(InferenceError, match="Expected 4D logits tensor with batch size 2"):
        engine.predict_batch(images)

    # Class mismatch in batch
    mock_backend.forward.return_value = np.zeros((2, 3, 32, 32), dtype=np.float32)
    with pytest.raises(InferenceError, match="Expected 5 classes, got 3"):
        engine.predict_batch(images)
