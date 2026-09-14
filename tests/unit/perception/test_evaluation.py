import math

import numpy as np
import pytest

from nearfield360.data.detection import DetectionAnnotation
from nearfield360.perception import (
    EvaluationError,
    environment_metadata,
    evaluate_detection,
    evaluate_semantic,
)


def _mask(values: list[list[int]]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8)


def test_semantic_evaluation_aggregates_confusion_across_images() -> None:
    evaluation = evaluate_semantic(
        [
            (_mask([[0, 1], [1, 1]]), _mask([[0, 0], [1, 1]])),
            (_mask([[0, 0], [0, 1]]), _mask([[0, 0], [1, 1]])),
        ]
    )

    assert evaluation.image_count == 2
    assert evaluation.pixel_count == 8
    assert evaluation.confusion[0, 0] == 3
    assert evaluation.confusion[1, 1] == 3
    assert evaluation.confusion[1, 0] == 1
    assert evaluation.confusion[0, 1] == 1
    assert evaluation.confusion.shape == (10, 10)
    assert evaluation.per_class_iou["road"] == pytest.approx(0.6)
    assert evaluation.per_class_iou["void"] == pytest.approx(0.6)
    assert evaluation.per_class_iou["curb"] is None
    assert evaluation.mean_iou == pytest.approx(0.6)
    assert evaluation.pixel_accuracy == pytest.approx(0.75)
    assert evaluation.target_pixel_counts["road"] == 4
    assert evaluation.target_pixel_counts["void"] == 4
    assert evaluation.as_dict()["confusion"][0][0] == 3


def test_semantic_evaluation_reports_none_for_absent_classes() -> None:
    target = _mask([[0, 5]])
    prediction = _mask([[0, 5]])

    evaluation = evaluate_semantic([(prediction, target)])

    assert evaluation.per_class_iou["void"] == pytest.approx(1.0)
    assert evaluation.per_class_iou["rider"] == pytest.approx(1.0)
    assert evaluation.per_class_iou["person"] is None
    assert evaluation.mean_iou == pytest.approx(1.0)
    payload = evaluation.as_dict()
    assert payload["classes"]["person"]["iou"] is None


def test_semantic_evaluation_rejects_empty_or_misaligned_input() -> None:
    with pytest.raises(EvaluationError, match="at least one"):
        evaluate_semantic([])
    with pytest.raises(EvaluationError, match="must match target shape"):
        evaluate_semantic([(_mask([[0]]), _mask([[0], [1]]))])
    with pytest.raises(ValueError, match="unknown labels"):
        evaluate_semantic([(_mask([[42]]), _mask([[0]]))])


def test_detection_evaluation_pools_matches_across_images() -> None:
    predictions = [
        (np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([0.9]), np.array([0])),
        (np.array([[5.0, 5.0, 7.0, 7.0]]), np.array([0.8]), np.array([0])),
    ]
    targets = [
        (DetectionAnnotation(0, "vehicles", 0.0, 0.0, 2.0, 2.0),),
        (DetectionAnnotation(0, "vehicles", 5.0, 5.0, 7.0, 7.0),),
    ]

    evaluation = evaluate_detection(predictions, targets)

    assert evaluation.image_count == 2
    assert evaluation.per_class_ap["vehicles"] == pytest.approx(1.0)
    assert evaluation.mean_average_precision == pytest.approx(1.0)
    assert evaluation.target_counts["vehicles"] == 2
    assert evaluation.prediction_counts["vehicles"] == 2
    assert evaluation.per_class_ap["person"] is None
    payload = evaluation.as_dict()
    assert payload["mean_average_precision"] == pytest.approx(1.0)


def test_detection_evaluation_accepts_empty_batches() -> None:
    empty_predictions = [
        (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)),
        (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)),
    ]
    targets = [[], []]

    evaluation = evaluate_detection(empty_predictions, targets)

    assert evaluation.image_count == 2
    assert all(value is None for value in evaluation.per_class_ap.values())
    assert math.isnan(evaluation.mean_average_precision)


def test_detection_evaluation_rejects_misaligned_batches() -> None:
    with pytest.raises(EvaluationError, match="must match target batches"):
        evaluate_detection([(np.empty((0, 4)), np.empty((0,)), np.empty((0,)))], [])
    with pytest.raises(EvaluationError, match="at least one"):
        evaluate_detection([], [])
    with pytest.raises(EvaluationError, match="must align per batch"):
        evaluate_detection(
            [(np.zeros((1, 4)), np.array([0.5]), np.empty((0,), dtype=np.int64))],
            [()],
        )
    with pytest.raises(EvaluationError, match=r"\[0, 1\]"):
        evaluate_detection(
            [(np.zeros((1, 4)), np.array([2.5]), np.empty((1,), dtype=np.int64))],
            [()],
        )


def test_environment_metadata_contains_version_keys() -> None:
    metadata = environment_metadata()

    assert set(metadata) == {
        "nearfield360_version",
        "python_version",
        "numpy_version",
        "platform",
    }
    assert metadata["nearfield360_version"]
