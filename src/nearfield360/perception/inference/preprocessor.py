"""Fisheye image preprocessing, normalization, letterboxing, and spatial inverse mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessTransform:
    """Transformation parameters used to map coordinates between original and model space."""

    orig_height: int
    orig_width: int
    target_height: int
    target_width: int
    scale: float
    pad_left: int
    pad_top: int


class PreprocessorError(ValueError):
    """Raised when image preprocessing inputs are malformed or invalid."""


class FisheyeImagePreprocessor:
    """Preprocesses raw automotive fisheye imagery for neural network inference."""

    def __init__(
        self,
        target_size: tuple[int, int] = (480, 640),  # (height, width)
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        preserve_aspect_ratio: bool = True,
        pad_value: int = 114,
    ) -> None:
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            msg = f"target_size must be positive (height, width), got {target_size}"
            raise PreprocessorError(msg)
        if any(s <= 0.0 for s in std):
            raise PreprocessorError(f"std must be strictly positive, got {std}")

        self._target_height, self._target_width = target_size
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._preserve_aspect_ratio = preserve_aspect_ratio
        self._pad_value = pad_value

    @property
    def target_size(self) -> tuple[int, int]:
        return self._target_height, self._target_width

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, PreprocessTransform]:
        """Preprocess a single RGB image into an NCHW float32 tensor."""
        if not isinstance(image, np.ndarray):
            raise PreprocessorError(f"image must be a numpy ndarray, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise PreprocessorError(f"image must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise PreprocessorError(f"image must have dtype uint8, got {image.dtype}")

        orig_h, orig_w = image.shape[:2]

        if not self._preserve_aspect_ratio:
            resized = cv2.resize(
                image,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            transform = PreprocessTransform(
                orig_height=orig_h,
                orig_width=orig_w,
                target_height=self._target_height,
                target_width=self._target_width,
                scale=1.0,
                pad_left=0,
                pad_top=0,
            )
        else:
            scale = min(self._target_height / orig_h, self._target_width / orig_w)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))

            resized_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full(
                (self._target_height, self._target_width, 3),
                self._pad_value,
                dtype=np.uint8,
            )

            pad_top = (self._target_height - new_h) // 2
            pad_left = (self._target_width - new_w) // 2

            canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_scaled
            resized = canvas

            transform = PreprocessTransform(
                orig_height=orig_h,
                orig_width=orig_w,
                target_height=self._target_height,
                target_width=self._target_width,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
            )

        # Normalize to [0, 1] then standardize
        normalized = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        # Transpose (H, W, C) -> (C, H, W) and add batch dim -> (1, C, H, W)
        chw = np.transpose(normalized, (2, 0, 1))
        blob = np.ascontiguousarray(np.expand_dims(chw, axis=0), dtype=np.float32)
        return blob, transform

    def preprocess_batch(
        self, images: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, list[PreprocessTransform]]:
        """Preprocess a sequence of RGB images into a single batched NCHW tensor."""
        if not images:
            raise PreprocessorError("images sequence must not be empty")

        blobs = []
        transforms = []
        for img in images:
            blob, trans = self.preprocess(img)
            blobs.append(blob)
            transforms.append(trans)

        batch_blob = np.ascontiguousarray(np.concatenate(blobs, axis=0), dtype=np.float32)
        return batch_blob, transforms

    @staticmethod
    def unscale_boxes(
        boxes_xyxy: np.ndarray,
        transform: PreprocessTransform,
    ) -> np.ndarray:
        """Map bounding boxes from model input space back to original image pixels.

        Args:
            boxes_xyxy: Array of shape (N, 4) with coordinates [x_min, y_min, x_max, y_max].
            transform: Transformation parameters recorded during preprocessing.

        Returns:
            Unscaled coordinates clipped to [0, orig_width] and [0, orig_height].
        """
        if boxes_xyxy.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        if transform.scale == 1.0 and transform.pad_left == 0 and transform.pad_top == 0:
            # Direct resize without padding
            scale_x = transform.orig_width / transform.target_width
            scale_y = transform.orig_height / transform.target_height
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        else:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_left) / transform.scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_top) / transform.scale

        # Clip to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(transform.orig_width))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(transform.orig_height))
        return boxes

    @staticmethod
    def unscale_mask(
        mask: np.ndarray,
        transform: PreprocessTransform,
    ) -> np.ndarray:
        """Map a 2D integer class mask from model input space back to original image pixels."""
        if mask.ndim != 2:
            raise PreprocessorError(f"mask must have shape (H, W), got {mask.shape}")

        if transform.scale == 1.0 and transform.pad_left == 0 and transform.pad_top == 0:
            return np.asarray(
                cv2.resize(
                    mask,
                    (transform.orig_width, transform.orig_height),
                    interpolation=cv2.INTER_NEAREST,
                ),
                dtype=mask.dtype,
            )

        scaled_h = max(1, round(transform.orig_height * transform.scale))
        scaled_w = max(1, round(transform.orig_width * transform.scale))

        cropped = mask[
            transform.pad_top : transform.pad_top + scaled_h,
            transform.pad_left : transform.pad_left + scaled_w,
        ]
        return np.asarray(
            cv2.resize(
                cropped,
                (transform.orig_width, transform.orig_height),
                interpolation=cv2.INTER_NEAREST,
            ),
            dtype=mask.dtype,
        )

    @staticmethod
    def unscale_continuous_map(
        heatmap: np.ndarray,
        transform: PreprocessTransform,
    ) -> np.ndarray:
        """Map a 2D float heatmap from model input space back to original image pixels."""
        if heatmap.ndim != 2:
            raise PreprocessorError(f"heatmap must have shape (H, W), got {heatmap.shape}")

        if transform.scale == 1.0 and transform.pad_left == 0 and transform.pad_top == 0:
            return np.asarray(
                cv2.resize(
                    heatmap,
                    (transform.orig_width, transform.orig_height),
                    interpolation=cv2.INTER_LINEAR,
                ),
                dtype=np.float32,
            )

        scaled_h = max(1, round(transform.orig_height * transform.scale))
        scaled_w = max(1, round(transform.orig_width * transform.scale))

        cropped = heatmap[
            transform.pad_top : transform.pad_top + scaled_h,
            transform.pad_left : transform.pad_left + scaled_w,
        ]
        return np.asarray(
            cv2.resize(
                cropped,
                (transform.orig_width, transform.orig_height),
                interpolation=cv2.INTER_LINEAR,
            ),
            dtype=np.float32,
        )


__all__ = [
    "FisheyeImagePreprocessor",
    "PreprocessTransform",
    "PreprocessorError",
]
