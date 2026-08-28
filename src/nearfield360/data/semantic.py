"""WoodScape semantic classes and validated mask utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt

from nearfield360.data.images import DEFAULT_IMAGE_LIMITS, ImageLimits, load_label_image


class SemanticMaskError(ValueError):
    """Raised when semantic labels violate the WoodScape class contract."""


@dataclass(frozen=True)
class SemanticClass:
    """One train/evaluation label and its official OpenCV BGR display color."""

    label_id: int
    name: str
    color_bgr: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not 0 <= self.label_id <= 255:
            raise ValueError("label_id must fit in an unsigned 8-bit mask")
        if not self.name:
            raise ValueError("semantic class name must not be empty")
        if len(self.color_bgr) != 3 or any(not 0 <= channel <= 255 for channel in self.color_bgr):
            raise ValueError("color_bgr must contain three unsigned 8-bit channels")


WOODSCAPE_SEMANTIC_CLASSES = (
    SemanticClass(0, "void", (0, 0, 0)),
    SemanticClass(1, "road", (255, 0, 255)),
    SemanticClass(2, "lanemarks", (255, 0, 0)),
    SemanticClass(3, "curb", (0, 255, 0)),
    SemanticClass(4, "person", (0, 0, 255)),
    SemanticClass(5, "rider", (255, 255, 255)),
    SemanticClass(6, "vehicles", (255, 255, 0)),
    SemanticClass(7, "bicycle", (0, 255, 255)),
    SemanticClass(8, "motorcycle", (128, 128, 255)),
    SemanticClass(9, "traffic_sign", (0, 128, 128)),
)

_CLASS_BY_ID = MappingProxyType(
    {semantic_class.label_id: semantic_class for semantic_class in WOODSCAPE_SEMANTIC_CLASSES}
)
_PALETTE_BGR = np.asarray(
    [semantic_class.color_bgr for semantic_class in WOODSCAPE_SEMANTIC_CLASSES],
    dtype=np.uint8,
)


def semantic_class(label_id: int) -> SemanticClass:
    """Return class metadata with a useful error for unknown IDs."""
    try:
        return _CLASS_BY_ID[label_id]
    except KeyError as exc:
        raise SemanticMaskError(f"Unknown WoodScape semantic label: {label_id}") from exc


def validate_semantic_mask(mask: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
    """Validate shape, dtype, and label values, returning a compact contiguous mask."""
    if mask.ndim != 2 or mask.size == 0:
        raise SemanticMaskError(f"Semantic mask must be a non-empty 2D array, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise SemanticMaskError(f"Semantic mask must contain integer labels, got {mask.dtype}")

    unique_labels = np.unique(mask)
    invalid_labels = [int(label) for label in unique_labels if int(label) not in _CLASS_BY_ID]
    if invalid_labels:
        rendered = ", ".join(str(label) for label in invalid_labels[:10])
        remainder = " ..." if len(invalid_labels) > 10 else ""
        raise SemanticMaskError(f"Semantic mask contains unknown labels: {rendered}{remainder}")
    return np.ascontiguousarray(mask, dtype=np.uint8)


def load_semantic_mask(
    path: Path, limits: ImageLimits = DEFAULT_IMAGE_LIMITS
) -> npt.NDArray[np.uint8]:
    """Decode and validate an official single-channel WoodScape semantic mask."""
    return validate_semantic_mask(load_label_image(path, limits=limits))


def colorize_semantic_mask(
    mask: npt.NDArray[np.generic],
    color_order: Literal["rgb", "bgr"] = "rgb",
) -> npt.NDArray[np.uint8]:
    """Map label IDs to the official palette in an explicit channel order."""
    labels = validate_semantic_mask(mask)
    colors_bgr = _PALETTE_BGR[labels]
    if color_order == "bgr":
        return np.ascontiguousarray(colors_bgr)
    if color_order == "rgb":
        return np.ascontiguousarray(colors_bgr[..., ::-1])
    raise ValueError(f"Unsupported color order: {color_order}")


def semantic_histogram(mask: npt.NDArray[np.generic]) -> dict[str, int]:
    """Count pixels for every class, including classes absent from this image."""
    labels = validate_semantic_mask(mask)
    counts = np.bincount(labels.ravel(), minlength=len(WOODSCAPE_SEMANTIC_CLASSES))
    return {
        semantic_class.name: int(counts[semantic_class.label_id])
        for semantic_class in WOODSCAPE_SEMANTIC_CLASSES
    }


__all__ = [
    "WOODSCAPE_SEMANTIC_CLASSES",
    "SemanticClass",
    "SemanticMaskError",
    "colorize_semantic_mask",
    "load_semantic_mask",
    "semantic_class",
    "semantic_histogram",
    "validate_semantic_mask",
]
