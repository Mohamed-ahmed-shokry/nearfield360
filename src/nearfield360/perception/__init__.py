"""Label-space perception metrics without model or accelerator dependencies."""

from nearfield360.perception.evaluation import (
    DetectionEvaluation,
    EvaluationError,
    SemanticEvaluation,
    environment_metadata,
    evaluate_detection,
    evaluate_semantic,
)
from nearfield360.perception.metrics import (
    detection_average_precision,
    detection_box_iou,
    detection_iou_matrix,
    mean_iou,
    semantic_confusion_matrix,
    semantic_iou,
    woodscape_detection_scores,
    woodscape_semantic_scores,
)

__all__ = [
    "DetectionEvaluation",
    "EvaluationError",
    "SemanticEvaluation",
    "detection_average_precision",
    "detection_box_iou",
    "detection_iou_matrix",
    "environment_metadata",
    "evaluate_detection",
    "evaluate_semantic",
    "mean_iou",
    "semantic_confusion_matrix",
    "semantic_iou",
    "woodscape_detection_scores",
    "woodscape_semantic_scores",
]
