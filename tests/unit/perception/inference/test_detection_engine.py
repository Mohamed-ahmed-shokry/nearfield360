"""Tests for ObjectDetectionEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nearfield360.data.detection import DetectionAnnotation, DetectionPrediction
from nearfield360.perception.inference.backend import InferenceError, OpenCVDNNBackend
from nearfield360.perception.inference.detection import ObjectDetectionEngine
from nearfield360.perception.inference.preprocessor import FisheyeImagePreprocessor
from nearfield360.perception.inference.test_utils import create_dummy_detection_onnx


@pytest.fixture
def dummy_det_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "dummy_detection.onnx"
    # 5 classes, 1 box with cx=0.5, cy=0.5, w=0.2, h=0.2, class 0 score=0.9
    boxes_scores = np.zeros((1, 2, 9), dtype=np.float32)
    # Box 0: center, class 0 (vehicles), score 0.9
    boxes_scores[0, 0, 0] = 0.5  # cx
    boxes_scores[0, 0, 1] = 0.5  # cy
    boxes_scores[0, 0, 2] = 0.2  # w
    boxes_scores[0, 0, 3] = 0.2  # h
    boxes_scores[0, 0, 4] = 0.9  # score class 0

    # Box 1: low confidence (0.1)
    boxes_scores[0, 1, 0] = 0.2
    boxes_scores[0, 1, 1] = 0.2
    boxes_scores[0, 1, 2] = 0.1
    boxes_scores[0, 1, 3] = 0.1
    boxes_scores[0, 1, 5] = 0.1  # score class 1

    create_dummy_detection_onnx(
        model_path,
        num_classes=5,
        num_boxes=2,
        height=64,
        width=64,
        boxes_and_scores=boxes_scores,
    )
    return model_path


def test_detection_engine_validation() -> None:
    mock_backend = MagicMock()
    with pytest.raises(ValueError, match="confidence_threshold must be in"):
        ObjectDetectionEngine(mock_backend, confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="confidence_threshold must be in"):
        ObjectDetectionEngine(mock_backend, confidence_threshold=1.5)
    with pytest.raises(ValueError, match="nms_threshold must be in"):
        ObjectDetectionEngine(mock_backend, nms_threshold=-0.1)
    with pytest.raises(ValueError, match="nms_threshold must be in"):
        ObjectDetectionEngine(mock_backend, nms_threshold=1.1)
    with pytest.raises(ValueError, match="num_classes must be positive"):
        ObjectDetectionEngine(mock_backend, num_classes=0)


def test_detection_engine_properties(dummy_det_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_det_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64))
    engine = ObjectDetectionEngine(
        backend,
        preprocessor=preprocessor,
        confidence_threshold=0.3,
        nms_threshold=0.5,
        num_classes=5,
    )

    assert engine.backend is backend
    assert engine.preprocessor is preprocessor
    assert engine.confidence_threshold == 0.3
    assert engine.nms_threshold == 0.5
    assert engine.num_classes == 5


def test_detection_engine_predict(dummy_det_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_det_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64), preserve_aspect_ratio=False)
    engine = ObjectDetectionEngine(
        backend,
        preprocessor=preprocessor,
        confidence_threshold=0.5,
    )

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    preds = engine.predict(image)

    # Box 0 has score 0.9 >= 0.5; Box 1 has score 0.1 < 0.5 and is filtered out
    assert len(preds) == 1
    p = preds[0]
    assert isinstance(p, DetectionPrediction)
    assert p.class_id == 0
    assert p.class_name == "vehicles"
    assert p.score == pytest.approx(0.9, abs=1e-4)

    # In target size (64, 64): cx=32, cy=32, w=12.8, h=12.8
    # Direct resize to orig (100, 200): cx=100, cy=50, w=40, h=20
    # x_min = 100 - 20 = 80, x_max = 100 + 20 = 120
    # y_min = 50 - 10 = 40, y_max = 50 + 10 = 60
    assert p.x_min == pytest.approx(80.0, abs=1.0)
    assert p.x_max == pytest.approx(120.0, abs=1.0)
    assert p.y_min == pytest.approx(40.0, abs=1.0)
    assert p.y_max == pytest.approx(60.0, abs=1.0)


def test_detection_engine_high_threshold_filters_all(dummy_det_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_det_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64))
    engine = ObjectDetectionEngine(
        backend,
        preprocessor=preprocessor,
        confidence_threshold=0.95,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    preds = engine.predict(image)
    assert preds == ()


def test_detection_engine_to_annotations(dummy_det_model: Path) -> None:
    backend = OpenCVDNNBackend(dummy_det_model)
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64))
    engine = ObjectDetectionEngine(backend, preprocessor=preprocessor, confidence_threshold=0.5)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    annotations = engine.predict_annotations(image)

    assert len(annotations) == 1
    ann = annotations[0]
    assert isinstance(ann, DetectionAnnotation)
    assert ann.class_id == 0
    assert ann.class_name == "vehicles"


def test_detection_engine_transposed_and_empty() -> None:
    mock_backend = MagicMock()
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64))
    engine = ObjectDetectionEngine(mock_backend, preprocessor=preprocessor, num_classes=5)

    # Empty box tensor
    mock_backend.forward.return_value = np.zeros((1, 0, 9), dtype=np.float32)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert engine.predict(image) == ()

    # Transposed YOLO tensor: shape (1, 9, 10)
    tensor = np.zeros((1, 9, 10), dtype=np.float32)
    # Box 3 has high confidence for person (class 1 -> index 4 + 1 = 5)
    tensor[0, 0, 3] = 32.0  # cx (pixel)
    tensor[0, 1, 3] = 32.0  # cy (pixel)
    tensor[0, 2, 3] = 16.0  # w
    tensor[0, 3, 3] = 16.0  # h
    tensor[0, 5, 3] = 0.85  # class 1 score
    mock_backend.forward.return_value = tensor

    preds = engine.predict(image)
    assert len(preds) == 1
    assert preds[0].class_id == 1
    assert preds[0].class_name == "person"
    assert preds[0].score == pytest.approx(0.85, abs=1e-3)


def test_detection_engine_errors() -> None:
    mock_backend = MagicMock()
    preprocessor = FisheyeImagePreprocessor(target_size=(64, 64))
    engine = ObjectDetectionEngine(mock_backend, preprocessor=preprocessor, num_classes=5)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # 4D output tensor
    mock_backend.forward.return_value = np.zeros((1, 2, 3, 4), dtype=np.float32)
    with pytest.raises(InferenceError, match="Expected 2D or 3D detection tensor"):
        engine.predict(image)

    # Batch size > 1
    mock_backend.forward.return_value = np.zeros((2, 5, 9), dtype=np.float32)
    with pytest.raises(InferenceError, match="Expected single-item batch"):
        engine.predict(image)

    # Feature dimension too small
    mock_backend.forward.return_value = np.zeros((1, 5, 7), dtype=np.float32)
    with pytest.raises(InferenceError, match="Expected output dimension at least 9"):
        engine.predict(image)
