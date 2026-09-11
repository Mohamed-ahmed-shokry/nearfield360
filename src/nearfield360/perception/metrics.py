"""Deterministic label-space metrics for segmentation and detection.

These helpers compare integer labels and XYXY boxes only. They never load
images, run models, or require accelerators, so unit tests and CI stay
CPU-only and synthetic. Prediction quality on real data still requires a
documented dataset split, and absent classes report NaN rather than a
perfect or zero score.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    validate_semantic_mask,
)


def _label_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer label array") from exc
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer labels")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.int64)
    if result.ndim != 2 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array, got {result.shape}")
    return result


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ValueError("num_classes must be a positive integer")
    if num_classes <= 0 or num_classes > 1024:
        raise ValueError("num_classes must be in the range 1..1024")
    return num_classes


def semantic_confusion_matrix(
    prediction: ArrayLike, target: ArrayLike, *, num_classes: int
) -> NDArray[np.int64]:
    """Count ``(target, prediction)`` pairs over two equal-shape label maps.

    Rows are ground-truth classes and columns are predicted classes. Labels
    outside ``[0, num_classes)`` raise instead of being silently dropped, so a
    shifted taxonomy surfaces as an error rather than a misleading score.
    """
    classes = _validate_num_classes(num_classes)
    pred = _label_array(prediction, "prediction")
    truth = _label_array(target, "target")
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} must match target {truth.shape}")
    if pred.min() < 0 or pred.max() >= classes or truth.min() < 0 or truth.max() >= classes:
        raise ValueError(f"labels must lie in [0, {classes})")
    flat = truth.ravel() * classes + pred.ravel()
    return np.bincount(flat, minlength=classes * classes).reshape(classes, classes).astype(np.int64)


def semantic_iou(confusion: ArrayLike) -> NDArray[np.float64]:
    """Return per-class IoU for a square confusion matrix.

    Classes with zero union (absent from both prediction and target) yield NaN
    so ``mean_iou`` can ignore them instead of rewarding empty classes.
    """
    try:
        matrix = np.asarray(confusion, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confusion must be a numeric square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"confusion must be a non-empty square matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("confusion must contain finite non-negative counts")
    if matrix.shape[0] > 1024:
        raise ValueError("confusion must not exceed 1024 classes")
    with np.errstate(all="ignore"):
        true_positives = np.diag(matrix)
        false_positives = matrix.sum(axis=0) - true_positives
        false_negatives = matrix.sum(axis=1) - true_positives
        union = true_positives + false_positives + false_negatives
        iou = true_positives / union
        iou[union == 0.0] = math.nan
    return np.ascontiguousarray(iou, dtype=np.float64)


def mean_iou(iou: ArrayLike) -> float:
    """Average per-class IoU while ignoring NaN (absent) classes.

    Returns NaN when every class is absent rather than reporting zero.
    """
    try:
        values = np.asarray(iou, dtype=np.float64).ravel()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("iou must be a numeric array") from exc
    if values.size == 0:
        raise ValueError("iou must not be empty")
    if np.any((~np.isnan(values)) & ((values < 0.0) | (values > 1.0))):
        raise ValueError("iou values must lie in [0, 1] or be NaN")
    if np.all(np.isnan(values)):
        return math.nan
    with np.errstate(all="ignore"):
        average = float(np.nanmean(values))
    return average


def woodscape_semantic_scores(prediction: ArrayLike, target: ArrayLike) -> dict[str, float | None]:
    """Score two validated WoodScape masks, keyed by class name.

    Absent classes map to ``None`` (JSON-safe) instead of NaN. Inputs pass
    through the same strict mask contract used by the data layer, so unknown
    label IDs raise before they can pollute a reported metric.
    """
    pred = validate_semantic_mask(np.asarray(prediction))
    truth = validate_semantic_mask(np.asarray(target))
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} must match target {truth.shape}")
    classes = len(WOODSCAPE_SEMANTIC_CLASSES)
    confusion = semantic_confusion_matrix(pred, truth, num_classes=classes)
    scores = semantic_iou(confusion)
    result: dict[str, float | None] = {}
    for semantic_class, score in zip(WOODSCAPE_SEMANTIC_CLASSES, scores, strict=True):
        result[semantic_class.name] = None if math.isnan(float(score)) else float(score)
    return result


def _box_array(values: ArrayLike, name: str, *, batched: bool) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric XYXY box array") from exc
    if array.dtype.kind != "f":
        raise ValueError(f"{name} must be a real numeric XYXY box array")
    if batched and (array.ndim != 2 or array.shape[1] != 4):
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not batched and array.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if np.any(array < 0.0):
        raise ValueError(f"{name} coordinates must be non-negative")
    if np.any(array[..., 0] >= array[..., 2]) or np.any(array[..., 1] >= array[..., 3]):
        raise ValueError(f"{name} boxes must have positive width and height")
    return np.ascontiguousarray(array, dtype=np.float64)


def detection_box_iou(first: ArrayLike, second: ArrayLike) -> float:
    """Return the IoU of two XYXY boxes in stored-image pixels.

    Coordinates use differences for width/height, without an inclusive-pixel
    ``+1`` convention, matching the detection annotation contract.
    """
    first_box = _box_array(first, "first", batched=False)
    second_box = _box_array(second, "second", batched=False)
    x_min = max(float(first_box[0]), float(second_box[0]))
    y_min = max(float(first_box[1]), float(second_box[1]))
    x_max = min(float(first_box[2]), float(second_box[2]))
    y_max = min(float(first_box[3]), float(second_box[3]))
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = float((first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))
    second_area = float((second_box[2] - second_box[0]) * (second_box[3] - second_box[1]))
    union = first_area + second_area - intersection
    if not math.isfinite(union) or union <= 0.0:
        raise ValueError("box union must be finite and positive")
    return intersection / union


def detection_iou_matrix(predictions: ArrayLike, targets: ArrayLike) -> NDArray[np.float64]:
    """Return pairwise IoU with shape ``(len(predictions), len(targets))``.

    Empty inputs yield an all-empty ``(0, 0)``-shaped corner: ``(N, 0)`` when
    only targets are empty and ``(0, M)`` when only predictions are empty.
    """

    def _is_empty(values: ArrayLike) -> bool:
        try:
            return np.asarray(values, dtype=np.float64).size == 0
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("box arrays must contain real numeric XYXY coordinates") from exc

    pred_empty = _is_empty(predictions)
    target_empty = _is_empty(targets)
    if pred_empty and target_empty:
        return np.empty((0, 0), dtype=np.float64)
    if pred_empty:
        truth = _box_array(targets, "targets", batched=True)
        return np.empty((0, truth.shape[0]), dtype=np.float64)
    if target_empty:
        pred_only = _box_array(predictions, "predictions", batched=True)
        return np.empty((pred_only.shape[0], 0), dtype=np.float64)
    pred = _box_array(predictions, "predictions", batched=True)
    truth = _box_array(targets, "targets", batched=True)
    with np.errstate(all="ignore"):
        x_min = np.maximum(pred[:, None, 0], truth[None, :, 0])
        y_min = np.maximum(pred[:, None, 1], truth[None, :, 1])
        x_max = np.minimum(pred[:, None, 2], truth[None, :, 2])
        y_max = np.minimum(pred[:, None, 3], truth[None, :, 3])
        intersection = np.maximum(0.0, x_max - x_min) * np.maximum(0.0, y_max - y_min)
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        truth_area = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
        union = pred_area[:, None] + truth_area[None, :] - intersection
        iou = intersection / union
        if not np.all(np.isfinite(iou)):
            raise ValueError("box IoU matrix must be finite")
    return np.ascontiguousarray(iou, dtype=np.float64)


__all__ = [
    "detection_box_iou",
    "detection_iou_matrix",
    "mean_iou",
    "semantic_confusion_matrix",
    "semantic_iou",
    "woodscape_semantic_scores",
]
