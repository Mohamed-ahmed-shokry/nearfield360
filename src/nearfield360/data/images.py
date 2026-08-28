"""Bounded OpenCV decoding at the external-dataset trust boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt


class ImageReadError(ValueError):
    """Raised when an image is missing, oversized, malformed, or has an invalid shape."""


@dataclass(frozen=True)
class ImageLimits:
    """Resource limits checked around image decoding."""

    max_file_bytes: int = 64 * 1024 * 1024
    max_pixels: int = 50_000_000

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")


DEFAULT_IMAGE_LIMITS = ImageLimits()


def _decode(path: Path, flags: int, limits: ImageLimits) -> npt.NDArray[np.generic]:
    if not path.is_file():
        raise ImageReadError(f"Image file does not exist: {path}")
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ImageReadError(f"Unable to inspect image file {path}: {exc}") from exc
    if file_size > limits.max_file_bytes:
        raise ImageReadError(
            f"Image file exceeds the {limits.max_file_bytes}-byte safety limit: {path}"
        )

    try:
        image = cv2.imread(str(path), flags)
    except cv2.error as exc:
        raise ImageReadError(f"OpenCV could not decode image {path}: {exc}") from exc
    if image is None:
        raise ImageReadError(f"OpenCV could not decode image: {path}")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ImageReadError(f"Decoded image has an invalid shape {image.shape}: {path}")

    pixels = int(image.shape[0]) * int(image.shape[1])
    if pixels > limits.max_pixels:
        raise ImageReadError(
            f"Decoded image has {pixels} pixels, exceeding the {limits.max_pixels}-pixel limit: "
            f"{path}"
        )
    return np.ascontiguousarray(image)


def load_rgb_image(path: Path, limits: ImageLimits = DEFAULT_IMAGE_LIMITS) -> npt.NDArray[np.uint8]:
    """Decode an image as contiguous RGB uint8, independent of OpenCV's BGR convention."""
    bgr = _decode(path, cv2.IMREAD_COLOR, limits)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ImageReadError(f"Expected an 8-bit three-channel image, got {bgr.shape}/{bgr.dtype}")
    bgr_uint8 = cast(npt.NDArray[np.uint8], bgr)
    return np.asarray(cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def load_label_image(
    path: Path, limits: ImageLimits = DEFAULT_IMAGE_LIMITS
) -> npt.NDArray[np.generic]:
    """Decode an integer, single-channel label image without changing label values."""
    labels = _decode(path, cv2.IMREAD_UNCHANGED, limits)
    if labels.ndim != 2:
        raise ImageReadError(
            f"Expected a single-channel label image, got shape {labels.shape}: {path}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ImageReadError(f"Expected integer labels, got dtype {labels.dtype}: {path}")
    return labels


__all__ = [
    "DEFAULT_IMAGE_LIMITS",
    "ImageLimits",
    "ImageReadError",
    "load_label_image",
    "load_rgb_image",
]
