import math

import numpy as np
import pytest

from nearfield360.perception.metrics import (
    detection_box_iou,
    detection_iou_matrix,
    mean_iou,
    semantic_confusion_matrix,
    semantic_iou,
    woodscape_semantic_scores,
)


def test_confusion_matrix_counts_target_rows_and_prediction_columns() -> None:
    prediction = np.array([[0, 1], [1, 1]])
    target = np.array([[0, 0], [1, 2]])

    confusion = semantic_confusion_matrix(prediction, target, num_classes=3)

    assert confusion.tolist() == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert confusion.dtype == np.int64


def test_confusion_matrix_rejects_shape_and_label_mismatches() -> None:
    with pytest.raises(ValueError, match="must match"):
        semantic_confusion_matrix([[0, 1]], [[0]], num_classes=2)
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        semantic_confusion_matrix([[0, 2]], [[0, 1]], num_classes=2)
    with pytest.raises(ValueError, match="integer"):
        semantic_confusion_matrix([[0.5, 1]], [[0, 1]], num_classes=2)
    with pytest.raises(ValueError, match="num_classes"):
        semantic_confusion_matrix([[0]], [[0]], num_classes=0)


def test_semantic_iou_reports_nan_for_absent_classes() -> None:
    confusion = np.array([[2, 1], [1, 2]])

    iou = semantic_iou(confusion)

    assert iou.tolist() == pytest.approx([0.5, 0.5])
    absent = semantic_iou(np.zeros((2, 2)))
    assert math.isnan(float(absent[0]))
    assert math.isnan(float(absent[1]))


def test_semantic_iou_rejects_malformed_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        semantic_iou([[1, 0, 0]])
    with pytest.raises(ValueError, match="non-negative"):
        semantic_iou([[-1, 0], [0, 1]])
    with pytest.raises(ValueError, match="finite"):
        semantic_iou([[float("inf"), 0], [0, 1]])


def test_mean_iou_ignores_absent_classes() -> None:
    assert mean_iou([0.5, float("nan"), 1.0]) == pytest.approx(0.75)
    assert math.isnan(mean_iou([float("nan"), float("nan")]))
    with pytest.raises(ValueError, match="must not be empty"):
        mean_iou([])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mean_iou([1.5])


def test_woodscape_scores_use_official_names_and_none_for_absent() -> None:
    prediction = np.zeros((4, 4), dtype=np.uint8)
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0, 0] = 1
    prediction[0, 0] = 1
    prediction[1, 1] = 6

    scores = woodscape_semantic_scores(prediction, target)

    assert scores["road"] == pytest.approx(1.0)
    assert scores["vehicles"] == pytest.approx(0.0)
    assert scores["person"] is None
    assert set(scores) == {
        "void",
        "road",
        "lanemarks",
        "curb",
        "person",
        "rider",
        "vehicles",
        "bicycle",
        "motorcycle",
        "traffic_sign",
    }
    with pytest.raises(ValueError, match="unknown labels"):
        woodscape_semantic_scores(np.full((2, 2), 42), target[:2, :2])


def test_detection_box_iou_matches_geometric_overlap() -> None:
    assert detection_box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert detection_box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == pytest.approx(0.0)
    # Intersection 1x1 over union 4+4-1=7.
    assert detection_box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1.0 / 7.0)


def test_detection_box_iou_rejects_degenerate_boxes() -> None:
    with pytest.raises(ValueError, match="positive width"):
        detection_box_iou([1, 1, 1, 2], [0, 0, 2, 2])
    with pytest.raises(ValueError, match="finite"):
        detection_box_iou([0, 0, 1, float("inf")], [0, 0, 2, 2])
    with pytest.raises(ValueError, match="non-negative"):
        detection_box_iou([-1, 0, 1, 1], [0, 0, 2, 2])


def test_detection_iou_matrix_shape_and_empty_corners() -> None:
    predictions = [[0, 0, 2, 2], [5, 5, 6, 6]]
    targets = [[0, 0, 2, 2], [1, 1, 3, 3]]

    matrix = detection_iou_matrix(predictions, targets)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(1.0 / 7.0)
    assert matrix[1, 0] == pytest.approx(0.0)
    assert detection_iou_matrix([], []).shape == (0, 0)
    assert detection_iou_matrix(predictions, []).shape == (2, 0)
    assert detection_iou_matrix([], targets).shape == (0, 2)
