import math

import numpy as np
import pytest

from nearfield360.perception.metrics import (
    detection_average_precision,
    detection_box_iou,
    detection_confidence_metrics,
    detection_iou_matrix,
    mean_iou,
    semantic_confidence_metrics,
    semantic_confusion_matrix,
    semantic_iou,
    woodscape_detection_scores,
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


def test_detection_scores_perfect_pooled_detections() -> None:
    boxes = np.array([[0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 12.0, 12.0]])
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])

    scores_result = woodscape_detection_scores(boxes, scores, classes, boxes, classes)

    assert scores_result["vehicles"] == pytest.approx(1.0)
    assert scores_result["person"] is None
    assert set(scores_result) == {
        "vehicles",
        "person",
        "bicycle",
        "traffic_light",
        "traffic_sign",
    }
    assert detection_average_precision(boxes, scores, classes, boxes, classes) == pytest.approx(1.0)


def test_detection_average_precision_from_pr_steps() -> None:
    targets = np.array([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 7.0]])
    target_classes = np.array([0, 0])
    predictions = np.array([[0.0, 0.0, 2.0, 2.0], [20.0, 20.0, 21.0, 21.0]])
    pred_classes = np.array([0, 0])
    pred_scores = np.array([0.9, 0.4])

    result = woodscape_detection_scores(
        predictions, pred_scores, pred_classes, targets, target_classes
    )

    assert result["vehicles"] == pytest.approx(0.5)


def test_detection_scores_without_targets_or_predictions() -> None:
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]])
    scores = np.array([0.9])
    classes = np.array([0])

    with_targets = woodscape_detection_scores(
        boxes, scores, classes, np.empty((0, 4)), np.empty((0,), dtype=np.int64)
    )
    assert with_targets["vehicles"] == 0.0

    empty_targets = np.empty((0, 4))
    empty_classes = np.empty((0,), dtype=np.int64)
    no_and_no = woodscape_detection_scores(
        np.empty((0, 4)), np.empty((0,)), empty_classes, empty_targets, empty_classes
    )
    assert no_and_no["vehicles"] is None


def test_detection_average_precision_is_nan_when_all_classes_absent() -> None:
    empty_boxes = np.empty((0, 4))
    empty_scores = np.empty((0,))
    empty_classes = np.empty((0,), dtype=np.int64)

    assert math.isnan(
        detection_average_precision(
            empty_boxes, empty_scores, empty_classes, empty_boxes, empty_classes
        )
    )


def test_detection_scores_pool_predictions_across_images() -> None:
    first_targets = np.array([[0.0, 0.0, 2.0, 2.0]])
    second_targets = np.array([[5.0, 5.0, 7.0, 7.0]])
    combined = np.array([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 7.0]])

    pooled = woodscape_detection_scores(
        combined,
        np.array([1.0, 1.0]),
        np.array([0, 0]),
        np.concatenate((first_targets, second_targets)),
        np.array([0, 0]),
    )
    assert pooled["vehicles"] == pytest.approx(1.0)


def test_detection_scores_respect_iou_threshold_boundary() -> None:
    ground_truth = np.array([[0.0, 0.0, 2.0, 2.0]])
    prediction = np.array([[0.0, 0.0, 2.0, 1.0]])
    scores = np.array([1.0])
    classes = np.array([0])

    at_threshold = woodscape_detection_scores(
        prediction, scores, classes, ground_truth, classes, iou_threshold=0.5
    )
    assert at_threshold["vehicles"] == pytest.approx(1.0)

    above_threshold = woodscape_detection_scores(
        prediction, scores, classes, ground_truth, classes, iou_threshold=0.6
    )
    assert above_threshold["vehicles"] == pytest.approx(0.0)


def test_detection_scores_prefer_higher_scored_match() -> None:
    ground_truth = np.array([[0.0, 0.0, 6.0, 6.0]])
    target_class = np.array([0])
    predictions = np.array([[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 6.0, 5.0]])
    classes = np.array([0, 0])

    first = woodscape_detection_scores(
        predictions, np.array([0.95, 0.1]), classes, ground_truth, target_class
    )
    second = woodscape_detection_scores(
        predictions, np.array([0.1, 0.95]), classes, ground_truth, target_class
    )

    assert first["vehicles"] == pytest.approx(1.0)
    assert second["vehicles"] == pytest.approx(1.0)


def test_detection_scores_reject_malformed_inputs() -> None:
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match="matching lengths"):
        woodscape_detection_scores(boxes, np.array([1.0]), np.array([0, 0]), boxes, np.array([0]))
    with pytest.raises(ValueError, match="iou_threshold"):
        woodscape_detection_scores(
            boxes, np.array([1.0]), np.array([0]), boxes, np.array([0]), iou_threshold=1.5
        )
    with pytest.raises(ValueError, match="Unknown WoodScape detection class"):
        woodscape_detection_scores(boxes, np.array([1.0]), np.array([9]), boxes, np.array([0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        woodscape_detection_scores(boxes, np.array([2.0]), np.array([0]), boxes, np.array([0]))


def _confidence_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One matching and one non-matching vehicle prediction plus one person FP."""
    pred_boxes = np.array(
        [
            [0.0, 0.0, 2.0, 2.0],  # matches the vehicle target
            [10.0, 10.0, 12.0, 12.0],  # vehicle FP
            [20.0, 20.0, 21.0, 21.0],  # person FP (class has no targets)
        ]
    )
    pred_scores = np.array([0.9, 0.4, 0.3])
    pred_classes = np.array([0, 0, 1])
    target_boxes = np.array([[0.0, 0.0, 2.0, 2.0]])
    target_classes = np.array([0])
    return pred_boxes, pred_scores, pred_classes, target_boxes, target_classes


def test_confidence_metrics_pooled_operating_points() -> None:
    pred_boxes, pred_scores, pred_classes, target_boxes, target_classes = _confidence_fixture()

    result = detection_confidence_metrics(
        pred_boxes,
        pred_scores,
        pred_classes,
        target_boxes,
        target_classes,
        thresholds=[0.5, 0.3],
    )

    assert result["iou_threshold"] == 0.5
    assert result["targets"] == 1
    points = result["thresholds"]
    assert [point["confidence"] for point in points] == [0.3, 0.5]
    high, low = points[1], points[0]
    # 0.5 keeps only the matching vehicle: perfect precision and recall.
    assert high["predictions"] == 1
    assert high["true_positives"] == 1
    assert high["precision"] == 1.0
    assert high["recall"] == 1.0
    assert high["f1"] == 1.0
    # 0.3 also keeps the vehicle FP and the person FP (no person targets).
    assert low["predictions"] == 3
    assert low["true_positives"] == 1
    assert low["precision"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert low["recall"] == 1.0
    assert low["f1"] == pytest.approx(0.5, abs=1e-6)


def test_confidence_metrics_reports_pr_grids() -> None:
    pred_boxes, pred_scores, pred_classes, target_boxes, target_classes = _confidence_fixture()

    result = detection_confidence_metrics(
        pred_boxes,
        pred_scores,
        pred_classes,
        target_boxes,
        target_classes,
        thresholds=[0.5],
    )

    vehicles = result["pr_curves"]["vehicles"]
    assert vehicles is not None
    assert len(vehicles) == 101
    assert [point["recall"] for point in vehicles] == [round(step / 100, 2) for step in range(101)]
    # The single vehicle target is matched by the top-scored prediction, so the
    # envelope stays at perfect precision across every reachable recall level.
    assert all(point["precision"] == 1.0 for point in vehicles)
    # Classes without ground-truth targets report a null curve.
    for absent in ("person", "bicycle", "traffic_light", "traffic_sign"):
        assert result["pr_curves"][absent] is None


def test_confidence_metrics_targets_without_predictions_are_zero() -> None:
    empty_boxes = np.empty((0, 4))
    empty_scores = np.empty((0,))
    empty_classes = np.empty((0,), dtype=np.int64)
    target_boxes = np.array([[0.0, 0.0, 2.0, 2.0]])
    target_classes = np.array([0])

    result = detection_confidence_metrics(
        empty_boxes,
        empty_scores,
        empty_classes,
        target_boxes,
        target_classes,
        thresholds=[0.0, 0.5],
    )

    assert result["targets"] == 1
    for point in result["thresholds"]:
        assert point["predictions"] == 0
        assert point["precision"] == 0.0
        assert point["recall"] == 0.0
        assert point["f1"] == 0.0
    vehicles = result["pr_curves"]["vehicles"]
    assert vehicles is not None
    assert len(vehicles) == 101
    assert all(point["precision"] == 0.0 for point in vehicles)


def test_confidence_metrics_deduplicates_and_sorts_thresholds() -> None:
    pred_boxes, pred_scores, pred_classes, target_boxes, target_classes = _confidence_fixture()

    result = detection_confidence_metrics(
        pred_boxes,
        pred_scores,
        pred_classes,
        target_boxes,
        target_classes,
        thresholds=[0.9, 0.1, 0.9, 0],
    )

    assert [point["confidence"] for point in result["thresholds"]] == [0.0, 0.1, 0.9]


def test_confidence_metrics_rejects_invalid_thresholds() -> None:
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]])
    score = np.array([0.9])
    classes = np.array([0])
    with pytest.raises(ValueError, match="at least one"):
        detection_confidence_metrics(boxes, score, classes, boxes, classes, thresholds=[])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        detection_confidence_metrics(boxes, score, classes, boxes, classes, thresholds=[1.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        detection_confidence_metrics(
            boxes, score, classes, boxes, classes, thresholds=[float("nan")]
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        detection_confidence_metrics(boxes, score, classes, boxes, classes, thresholds=["0.5"])


def test_semantic_confidence_metrics_bins_and_ece() -> None:
    confidences = np.array([0.9, 0.6, 0.4, 0.1])
    predictions = np.zeros(4, dtype=np.uint8)
    targets = np.array([0, 0, 1, 1], dtype=np.uint8)

    result = semantic_confidence_metrics(confidences, predictions, targets, num_bins=10)

    assert result["num_bins"] == 10
    assert result["pixel_count"] == 4
    assert result["mean_confidence"] == 0.5
    # 0.25 * (|0.9-1| + |0.6-1| + |0.4-0| + |0.1-0|)
    assert result["ece"] == 0.25
    assert len(result["bins"]) == 10
    high = result["bins"][9]
    assert (high["lower"], high["upper"], high["pixels"]) == (0.9, 1.0, 1)
    assert high["mean_confidence"] == 0.9
    assert high["accuracy"] == 1.0
    empty = result["bins"][0]
    assert empty["pixels"] == 0
    assert empty["mean_confidence"] is None
    assert empty["accuracy"] is None
    first = result["bins"][1]  # confidence 0.1, prediction wrong
    assert first["pixels"] == 1
    assert first["accuracy"] == 0.0


def test_semantic_confidence_metrics_clips_unit_confidence_into_last_bin() -> None:
    confidences = np.array([1.0, 1.0])

    result = semantic_confidence_metrics(
        confidences,
        np.zeros(2, dtype=np.uint8),
        np.zeros(2, dtype=np.uint8),
        num_bins=4,
    )

    assert result["ece"] == 0.0
    last = result["bins"][3]
    assert last["pixels"] == 2
    assert last["accuracy"] == 1.0
    assert last["upper"] == 1.0


def test_semantic_confidence_metrics_accepts_mask_shaped_inputs() -> None:
    confidences = np.full((2, 3), 0.75)
    predictions = np.zeros((2, 3), dtype=np.uint8)
    targets = np.ones((2, 3), dtype=np.uint8)

    result = semantic_confidence_metrics(confidences, predictions, targets, num_bins=8)

    assert result["pixel_count"] == 6
    assert result["bins"][6]["pixels"] == 6  # floor(0.75 * 8) = 6
    assert result["bins"][6]["accuracy"] == 0.0
    assert result["ece"] == 0.75  # every pixel overconfident by 0.75


def test_semantic_confidence_metrics_rejects_invalid_inputs() -> None:
    conf = np.array([0.5])
    pred = np.array([0], dtype=np.uint8)
    with pytest.raises(ValueError, match="num_bins"):
        semantic_confidence_metrics(conf, pred, pred, num_bins=0)
    with pytest.raises(ValueError, match="num_bins"):
        semantic_confidence_metrics(conf, pred, pred, num_bins=True)
    with pytest.raises(ValueError, match="num_bins"):
        semantic_confidence_metrics(conf, pred, pred, num_bins=1001)
    with pytest.raises(ValueError, match="must not be empty"):
        semantic_confidence_metrics(
            np.empty(0), np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        semantic_confidence_metrics(np.array([1.5]), pred, pred)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        semantic_confidence_metrics(np.array([float("nan")]), pred, pred)
    with pytest.raises(ValueError, match="align"):
        semantic_confidence_metrics(conf, np.zeros(2, dtype=np.uint8), pred)
    with pytest.raises(ValueError, match="integer labels"):
        semantic_confidence_metrics(conf, np.array([0.0]), pred)
    with pytest.raises(ValueError, match=r"labels must lie in \[0, 10\)"):
        semantic_confidence_metrics(conf, np.array([99], dtype=np.uint8), pred)
    with pytest.raises(ValueError, match="real numeric array"):
        semantic_confidence_metrics(np.array(["abc"]), pred, pred)
