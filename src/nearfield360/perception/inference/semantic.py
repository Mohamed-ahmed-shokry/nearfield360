"""Fisheye semantic segmentation inference engine."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nearfield360.perception.inference.backend import InferenceBackend, InferenceError
from nearfield360.perception.inference.preprocessor import (
    FisheyeImagePreprocessor,
)


class SemanticSegmentationEngine:
    """End-to-end inference engine for automotive fisheye semantic segmentation."""

    def __init__(
        self,
        backend: InferenceBackend,
        preprocessor: FisheyeImagePreprocessor | None = None,
        num_classes: int | None = None,
    ) -> None:
        self._backend = backend
        self._preprocessor = preprocessor or FisheyeImagePreprocessor()
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
    def num_classes(self) -> int | None:
        """Return expected number of semantic classes, if configured."""
        return self._num_classes

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run semantic segmentation inference on a single RGB fisheye image.

        Args:
            image: Raw input image array of shape (H, W, 3) and dtype uint8.

        Returns:
            Tuple of:
                class_mask: 2D uint8 array of shape (H, W) with discrete class IDs.
                confidence_map: 2D float32 array of shape (H, W) with softmax confidences.
        """
        blob, transform = self._preprocessor.preprocess(image)
        outputs = self._backend.forward(blob)
        raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if raw_logits.ndim == 4:
            if raw_logits.shape[0] != 1:
                msg = f"Expected single-item batch in logits, got batch size {raw_logits.shape[0]}"
                raise InferenceError(msg)
            logits = raw_logits[0]
        elif raw_logits.ndim == 3:
            logits = raw_logits
        else:
            msg = f"Expected 3D or 4D segmentation logits tensor, got shape {raw_logits.shape}"
            raise InferenceError(msg)

        if self._num_classes is not None and logits.shape[0] != self._num_classes:
            msg = f"Expected {self._num_classes} classes, got {logits.shape[0]}"
            raise InferenceError(msg)

        # Numerically stable softmax along class axis 0
        logits_max = np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

        class_mask_target = np.argmax(probabilities, axis=0).astype(np.uint8)
        conf_map_target = np.max(probabilities, axis=0).astype(np.float32)

        class_mask = FisheyeImagePreprocessor.unscale_mask(class_mask_target, transform)
        confidence_map = FisheyeImagePreprocessor.unscale_continuous_map(conf_map_target, transform)

        return class_mask, confidence_map

    def predict_batch(self, images: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run batched semantic segmentation inference on a sequence of RGB images."""
        if not images:
            return []

        blob, transforms = self._preprocessor.preprocess_batch(images)
        outputs = self._backend.forward(blob)
        raw_logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if raw_logits.ndim != 4 or raw_logits.shape[0] != len(transforms):
            msg = (
                f"Expected 4D logits tensor with batch size {len(transforms)}, "
                f"got shape {raw_logits.shape}"
            )
            raise InferenceError(msg)

        if self._num_classes is not None and raw_logits.shape[1] != self._num_classes:
            msg = f"Expected {self._num_classes} classes, got {raw_logits.shape[1]}"
            raise InferenceError(msg)

        results: list[tuple[np.ndarray, np.ndarray]] = []
        for i, transform in enumerate(transforms):
            logits_i = raw_logits[i]
            logits_max = np.max(logits_i, axis=0, keepdims=True)
            exp_logits = np.exp(logits_i - logits_max)
            probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

            mask_target = np.argmax(probabilities, axis=0).astype(np.uint8)
            conf_target = np.max(probabilities, axis=0).astype(np.float32)

            mask = FisheyeImagePreprocessor.unscale_mask(mask_target, transform)
            conf = FisheyeImagePreprocessor.unscale_continuous_map(conf_target, transform)
            results.append((mask, conf))

        return results


__all__ = [
    "SemanticSegmentationEngine",
]
