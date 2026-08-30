"""Bounded PNG/JPEG decoding at the external-dataset trust boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from nearfield360.data.image_headers import ImageHeaderError, read_image_header


class ImageReadError(ValueError):
    """Raised when an image is missing, oversized, malformed, or has an invalid shape."""


@dataclass(frozen=True)
class ImageLimits:
    """File-byte and pixel limits enforced before image decompression."""

    max_file_bytes: int = 64 * 1024 * 1024
    max_pixels: int = 50_000_000

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive and an integer")


DEFAULT_IMAGE_LIMITS = ImageLimits()


def _decode(path: Path, flags: int, limits: ImageLimits) -> npt.NDArray[np.generic]:
    try:
        if not path.is_file():
            raise ImageReadError(f"Image file does not exist: {path}")
        with path.open("rb") as stream:
            data = stream.read(limits.max_file_bytes + 1)
    except OSError as exc:
        raise ImageReadError(f"Unable to read image file {path}: {exc}") from exc
    if len(data) > limits.max_file_bytes:
        raise ImageReadError(
            f"Image file exceeds the {limits.max_file_bytes}-byte safety limit: {path}"
        )

    try:
        header = read_image_header(data)
    except ImageHeaderError as exc:
        raise ImageReadError(f"Decoder could not decode image header {path}: {exc}") from exc
    pixels = header.width * header.height
    if pixels > limits.max_pixels:
        raise ImageReadError(
            f"Image header declares {pixels} pixels, exceeding the {limits.max_pixels}-pixel "
            f"limit: {path}"
        )

    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    except cv2.error as exc:
        raise ImageReadError(f"OpenCV could not decode image {path}: {exc}") from exc
    if image is None:
        raise ImageReadError(f"OpenCV could not decode image: {path}")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ImageReadError(f"Decoded image has an invalid shape {image.shape}: {path}")

    if image.shape[:2] != (header.height, header.width):
        raise ImageReadError(
            f"Decoded image shape {image.shape[:2]} does not match stored header dimensions "
            f"{(header.height, header.width)}: {path}"
        )
    return np.ascontiguousarray(image)


def load_rgb_image(path: Path, limits: ImageLimits = DEFAULT_IMAGE_LIMITS) -> npt.NDArray[np.uint8]:
    """Decode PNG/JPEG as RGB uint8, preserving stored geometry regardless of EXIF."""
    bgr = _decode(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION, limits)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ImageReadError(f"Expected an 8-bit three-channel image, got {bgr.shape}/{bgr.dtype}")
    bgr_uint8 = cast(npt.NDArray[np.uint8], bgr)
    return np.asarray(cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def load_label_image(
    path: Path, limits: ImageLimits = DEFAULT_IMAGE_LIMITS
) -> npt.NDArray[np.generic]:
    """Decode single-channel PNG/JPEG integer labels without changing stored values."""
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
