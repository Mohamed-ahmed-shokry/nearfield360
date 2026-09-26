"""Deterministic, JSON-serializable evaluation over many labelled samples.

Aggregation happens on pooled counts (one summed confusion matrix, one set of
class-pooled detection matches), so a report is independent of the order and
batch size used to feed samples in. The same inputs always yield the same
metrics, which makes reruns comparable even before a model is stored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from nearfield360 import __version__ as _nearfield360_version
from nearfield360.data.detection import (
    WOODSCAPE_DETECTION_CLASSES,
    DetectionAnnotation,
)
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    validate_semantic_mask,
)
from nearfield360.perception.metrics import (
    detection_average_precision,
    semantic_confusion_matrix,
    semantic_iou,
    woodscape_detection_scores,
)

_CLASS_BY_ID = {item.class_id: item.name for item in WOODSCAPE_DETECTION_CLASSES}


class EvaluationError(ValueError):
    """Raised when evaluation inputs are empty, misaligned, or malformed."""


def environment_metadata() -> dict[str, str]:
    """Describe the runtime that produced a reproducible report."""
    import platform

    import numpy

    return {
        "nearfield360_version": _nearfield360_version,
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "platform": platform.platform(),
    }


def _confidence_scores(values: Iterable[float]) -> list[float]:
    scores = [float(value) for value in values]
    if not scores:
        return []
    if not all(np.isfinite(score) and 0.0 <= score <= 1.0 for score in scores):
        raise EvaluationError("Detection scores must be finite and lie in [0, 1]")
    return scores


@dataclass(frozen=True)
class SemanticEvaluation:
    """Aggregated semantic-segmentation scores over an image set."""

    image_count: int
    pixel_count: int
    per_class_iou: Mapping[str, float | None]
    mean_iou: float
    pixel_accuracy: float
    target_pixel_counts: Mapping[str, int]
    confusion: npt.NDArray[np.int64]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report (NaN becomes None)."""
        classes = {
            name: {
                "iou": iou,
                "target_pixels": self.target_pixel_counts[name],
            }
            for name, iou in self.per_class_iou.items()
        }
        return {
            "image_count": self.image_count,
            "pixel_count": self.pixel_count,
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "classes": classes,
            "confusion": self.confusion.tolist(),
        }


def evaluate_semantic(
    predictions_and_targets: Iterable[tuple[Any, Any]],
) -> SemanticEvaluation:
    """Aggregate segmentation pairs into one identity-independent report.

    Each pair is ``(prediction_labels, target_labels)``. Masks must use the
    WoodScape semantic label space and share the same shape. Per-image
    confusion matrices are summed before computing metrics.
    """
    pairs = tuple(predictions_and_targets)
    if not pairs:
        raise EvaluationError("Semantic evaluation requires at least one image pair")

    total_confusion: npt.NDArray[np.int64] | None = None
    pixel_total = 0
    class_totals: Counter[str] = Counter()
    for prediction, target in pairs:
        pred = validate_semantic_mask(np.asarray(prediction))
        truth = validate_semantic_mask(np.asarray(target))
        if pred.shape != truth.shape:
            raise EvaluationError(
                f"Prediction shape {pred.shape} must match target shape {truth.shape}"
            )
        classes = len(WOODSCAPE_SEMANTIC_CLASSES)
        confusion = semantic_confusion_matrix(pred, truth, num_classes=classes)
        total_confusion = confusion if total_confusion is None else total_confusion + confusion
        pixel_total += int(confusion.sum())
        for semantic_class, count in zip(
            (item for item in WOODSCAPE_SEMANTIC_CLASSES),
            confusion.sum(axis=1),
            strict=True,
        ):
            class_totals[semantic_class.name] += int(count)

    if total_confusion is None:
        raise EvaluationError("Semantic evaluation produced no confusion data")
    iou_values = semantic_iou(total_confusion)
    per_class = {
        item.name: (None if np.isnan(float(score)) else float(score))
        for item, score in zip(WOODSCAPE_SEMANTIC_CLASSES, iou_values, strict=True)
    }
    if pixel_total == 0:
        average = float("nan")
        accuracy = float("nan")
    else:
        finite = iou_values[np.isfinite(iou_values)]
        average = float(np.mean(finite)) if finite.size else float("nan")
        accuracy = float(np.diag(total_confusion).sum()) / pixel_total
    return SemanticEvaluation(
        image_count=len(pairs),
        pixel_count=pixel_total,
        per_class_iou=MappingProxyType(per_class),
        mean_iou=average,
        pixel_accuracy=accuracy,
        target_pixel_counts=MappingProxyType(dict(class_totals)),
        confusion=np.ascontiguousarray(total_confusion, dtype=np.int64),
    )


@dataclass(frozen=True)
class DetectionEvaluation:
    """Aggregated detection average precisions over an image set."""

    image_count: int
    iou_threshold: float
    per_class_ap: Mapping[str, float | None]
    mean_average_precision: float
    prediction_counts: Mapping[str, int]
    target_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report (NaN becomes None)."""
        classes = {
            name: {
                "average_precision": ap,
                "predictions": self.prediction_counts[name],
                "targets": self.target_counts[name],
            }
            for name, ap in self.per_class_ap.items()
        }
        return {
            "image_count": self.image_count,
            "iou_threshold": self.iou_threshold,
            "mean_average_precision": self.mean_average_precision,
            "classes": classes,
        }


DetectionBatch = tuple[Any, Any, Any]


def _pool_detection_batches(
    predictions: Sequence[DetectionBatch],
    targets: Sequence[Sequence[DetectionAnnotation]],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Validate per-image batches and pool them into aligned evaluation arrays.

    Returns ``(boxes, scores, pred_classes, target_boxes, target_classes)``
    with empty (0, 4)/(0,) arrays for missing predictions or targets.
    """
    if len(predictions) != len(targets):
        raise EvaluationError(
            f"Prediction batches {len(predictions)} must match target batches {len(targets)}"
        )
    if len(predictions) == 0:
        raise EvaluationError("Detection evaluation requires at least one image batch")

    pooled_boxes: list[npt.NDArray[np.float64]] = []
    pooled_scores: list[float] = []
    pooled_pred_classes: list[int] = []
    pooled_target_boxes: list[float] = []
    pooled_target_classes: list[int] = []
    for (image_boxes, image_scores, image_classes), target_batch in zip(
        predictions, targets, strict=True
    ):
        image_scores = _confidence_scores(image_scores)
        pooled_scores.extend(image_scores)
        image_boxes = np.asarray(image_boxes, dtype=np.float64)
        try:
            image_classes = np.asarray(image_classes, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError("Prediction classes must be integer WoodScape IDs") from exc
        if image_boxes.ndim != 2 or image_boxes.shape[1] != 4:
            raise EvaluationError(
                f"Prediction boxes must have shape (N, 4), got {image_boxes.shape}"
            )
        if len(image_boxes) != len(image_scores) or len(image_scores) != len(image_classes):
            raise EvaluationError("Prediction boxes, scores, and classes must align per batch")
        pooled_boxes.append(image_boxes)
        pooled_pred_classes.extend(int(value) for value in image_classes)

        for annotation in target_batch:
            if not isinstance(annotation, DetectionAnnotation):
                raise EvaluationError("Target batches must contain DetectionAnnotation objects")
            pooled_target_boxes.append(annotation.x_min)
            pooled_target_boxes.append(annotation.y_min)
            pooled_target_boxes.append(annotation.x_max)
            pooled_target_boxes.append(annotation.y_max)
            pooled_target_classes.append(annotation.class_id)

    if not pooled_boxes:
        all_boxes = np.empty((0, 4), dtype=np.float64)
        all_scores = np.empty((0,), dtype=np.float64)
        all_pred_classes = np.empty((0,), dtype=np.int64)
    else:
        all_boxes = np.concatenate(pooled_boxes, axis=0)
        all_scores = np.asarray(pooled_scores, dtype=np.float64)
        all_pred_classes = np.asarray(pooled_pred_classes, dtype=np.int64)
    all_target_boxes = (
        np.empty((0, 4), dtype=np.float64)
        if not pooled_target_boxes
        else np.asarray(pooled_target_boxes, dtype=np.float64).reshape(-1, 4)
    )
    all_target_classes = np.asarray(pooled_target_classes, dtype=np.int64)
    return all_boxes, all_scores, all_pred_classes, all_target_boxes, all_target_classes


def evaluate_detection(
    predictions: Sequence[DetectionBatch],
    targets: Sequence[Sequence[DetectionAnnotation]],
    *,
    iou_threshold: float = 0.5,
) -> DetectionEvaluation:
    """Aggregate per-image prediction batches against official annotations.

    Every prediction batch is ``(XYXY_boxes, scores, WoodScape detection
    class_ids)``; every target batch is a tuple of validated annotations.
    Matches are pooled per class across all images before one precision-recall
    curve per class is computed.
    """
    all_boxes, all_scores, all_pred_classes, all_target_boxes, all_target_classes = (
        _pool_detection_batches(predictions, targets)
    )

    per_class = woodscape_detection_scores(
        all_boxes,
        all_scores,
        all_pred_classes,
        all_target_boxes,
        all_target_classes,
        iou_threshold=iou_threshold,
    )
    map_score = detection_average_precision(
        all_boxes,
        all_scores,
        all_pred_classes,
        all_target_boxes,
        all_target_classes,
        iou_threshold=iou_threshold,
    )
    prediction_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for class_id in all_pred_classes:
        prediction_counts[detection_class_name(int(class_id))] += 1
    for class_id in all_target_classes:
        target_counts[detection_class_name(int(class_id))] += 1
    for item in WOODSCAPE_DETECTION_CLASSES:
        prediction_counts.setdefault(item.name, 0)
        target_counts.setdefault(item.name, 0)

    return DetectionEvaluation(
        image_count=len(predictions),
        iou_threshold=float(iou_threshold),
        per_class_ap=MappingProxyType(per_class),
        mean_average_precision=float(map_score),
        prediction_counts=MappingProxyType(dict(prediction_counts)),
        target_counts=MappingProxyType(dict(target_counts)),
    )


def detection_class_name(class_id: int) -> str:
    """Return the official class name for a validated detection class ID."""
    try:
        return _CLASS_BY_ID[class_id]
    except KeyError as exc:
        raise EvaluationError(f"Unknown WoodScape detection class_id: {class_id}") from exc


__all__ = [
    "DetectionBatch",
    "DetectionEvaluation",
    "EvaluationError",
    "SemanticEvaluation",
    "environment_metadata",
    "evaluate_detection",
    "evaluate_semantic",
]
