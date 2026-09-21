"""Fisheye 2D object detection inference engine."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from nearfield360.data.detection import (
    DetectionAnnotation,
    DetectionPrediction,
    detection_class,
)
from nearfield360.perception.inference.backend import InferenceBackend, InferenceError
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
)


class ObjectDetectionEngine:
    """End-to-end inference engine for automotive fisheye 2D object detection."""

    def __init__(
        self,
        backend: InferenceBackend,
        preprocessor: FisheyeImagePreprocessor | None = None,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        num_classes: int = 5,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            msg = f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            raise ValueError(msg)
        if not 0.0 <= nms_threshold <= 1.0:
            msg = f"nms_threshold must be in [0, 1], got {nms_threshold}"
            raise ValueError(msg)
        if num_classes <= 0:
            msg = f"num_classes must be positive, got {num_classes}"
            raise ValueError(msg)

        self._backend = backend
        self._preprocessor = preprocessor or FisheyeImagePreprocessor()
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._num_classes = num_classes

    @property
    def backend(self) -> InferenceBackend:
        """Return the underlying inference backend."""
        return self._backend

    @property
    def preprocessor(self) -> FisheyeImagePreprocessor:
        """Return the image preprocessor."""
        return self._preprocessor

    @property
    def confidence_threshold(self) -> float:
        """Return minimum confidence score threshold."""
        return self._confidence_threshold

    @property
    def nms_threshold(self) -> float:
        """Return IoU suppression threshold for NMS."""
        return self._nms_threshold

    @property
    def num_classes(self) -> int:
        """Return number of detection classes."""
        return self._num_classes

    def predict(self, image: np.ndarray) -> tuple[DetectionPrediction, ...]:
        """Run object detection inference on a single RGB fisheye image.

        Args:
            image: Raw input image array of shape (H, W, 3) and dtype uint8.

        Returns:
            Tuple of validated DetectionPrediction instances sorted by score descending.
        """
        blob, transform = self._preprocessor.preprocess(image)
        outputs = self._backend.forward(blob)
        raw_preds = outputs[0] if isinstance(outputs, tuple) else outputs

        if raw_preds.ndim == 3:
            if raw_preds.shape[0] != 1:
                msg = f"Expected single-item batch, got shape {raw_preds.shape}"
                raise InferenceError(msg)
            # Handle transposed YOLO format (1, C+4, N)
            if (
                raw_preds.shape[1] == 4 + self._num_classes
                and raw_preds.shape[2] != 4 + self._num_classes
            ):
                raw_preds = np.transpose(raw_preds, (0, 2, 1))
            preds = raw_preds[0]
        elif raw_preds.ndim == 2:
            preds = raw_preds
        else:
            msg = f"Expected 2D or 3D detection tensor, got shape {raw_preds.shape}"
            raise InferenceError(msg)

        if preds.shape[1] < 4 + self._num_classes:
            msg = (
                f"Expected output dimension at least {4 + self._num_classes}, got {preds.shape[1]}"
            )
            raise InferenceError(msg)

        if preds.shape[0] == 0:
            return ()

        boxes = preds[:, :4]
        class_scores = preds[:, 4 : 4 + self._num_classes]

        target_h, target_w = transform.target_height, transform.target_width
        # Determine if coordinates are normalized [0, 1] or in pixel space
        if np.all(boxes[:, :4] <= 1.0 + 1e-4) and np.any(boxes[:, :4] > 0):
            cx = boxes[:, 0] * target_w
            cy = boxes[:, 1] * target_h
            w = boxes[:, 2] * target_w
            h = boxes[:, 3] * target_h
        else:
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_xyxy_model = np.stack([x1, y1, x2, y2], axis=1)
        boxes_xyxy_orig = FisheyeImagePreprocessor.unscale_boxes(boxes_xyxy_model, transform)

        all_predictions: list[DetectionPrediction] = []

        for c in range(self._num_classes):
            scores_c = class_scores[:, c]
            mask = scores_c >= self._confidence_threshold
            if not np.any(mask):
                continue

            class_boxes = boxes_xyxy_orig[mask]
            class_scores_filtered = scores_c[mask]

            nms_boxes = [
                [
                    float(box[0]),
                    float(box[1]),
                    float(max(1e-2, box[2] - box[0])),
                    float(max(1e-2, box[3] - box[1])),
                ]
                for box in class_boxes
            ]
            nms_scores = [float(s) for s in class_scores_filtered]

            indices = cv2.dnn.NMSBoxes(
                nms_boxes,
                nms_scores,
                self._confidence_threshold,
                self._nms_threshold,
            )
            if len(indices) == 0:
                continue

            indices_flat = (
                indices.flatten().tolist()
                if hasattr(indices, "flatten")
                else [int(idx) for idx in indices]
            )

            class_meta = detection_class(c)
            for idx in indices_flat:
                b = class_boxes[idx]
                score = float(class_scores_filtered[idx])
                x_min, y_min, x_max, y_max = (
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]),
                )
                if x_max <= x_min or y_max <= y_min:
                    continue

                prediction = DetectionPrediction(
                    class_id=c,
                    class_name=class_meta.name,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    score=min(1.0, max(0.0, score)),
                )
                all_predictions.append(prediction)

        # Sort by confidence descending
        all_predictions.sort(key=lambda p: p.score, reverse=True)
        return tuple(all_predictions)

    @staticmethod
    def to_annotations(
        predictions: Sequence[DetectionPrediction],
    ) -> tuple[DetectionAnnotation, ...]:
        """Convert a sequence of DetectionPrediction into DetectionAnnotation instances."""
        return tuple(
            DetectionAnnotation(
                class_id=p.class_id,
                class_name=p.class_name,
                x_min=p.x_min,
                y_min=p.y_min,
                x_max=p.x_max,
                y_max=p.y_max,
            )
            for p in predictions
        )

    def predict_annotations(self, image: np.ndarray) -> tuple[DetectionAnnotation, ...]:
        """Run object detection inference and return DetectionAnnotation instances."""
        return self.to_annotations(self.predict(image))


__all__ = [
    "ObjectDetectionEngine",
]
