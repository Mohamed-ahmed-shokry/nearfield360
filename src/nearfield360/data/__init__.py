"""Dataset discovery, validation, and loading interfaces."""

from nearfield360.data.images import (
    DEFAULT_IMAGE_LIMITS,
    ImageLimits,
    ImageReadError,
    load_label_image,
    load_rgb_image,
)
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticClass,
    SemanticMaskError,
    colorize_semantic_mask,
    load_semantic_mask,
    semantic_class,
    semantic_histogram,
    validate_semantic_mask,
)
from nearfield360.data.woodscape import (
    CameraId,
    DatasetLayoutError,
    SampleKey,
    WoodScapeDataset,
    WoodScapeSample,
    locate_dataset_root,
    parse_sample_key,
)

__all__ = [
    "DEFAULT_IMAGE_LIMITS",
    "WOODSCAPE_SEMANTIC_CLASSES",
    "CameraId",
    "DatasetLayoutError",
    "ImageLimits",
    "ImageReadError",
    "SampleKey",
    "SemanticClass",
    "SemanticMaskError",
    "WoodScapeDataset",
    "WoodScapeSample",
    "colorize_semantic_mask",
    "load_label_image",
    "load_rgb_image",
    "load_semantic_mask",
    "locate_dataset_root",
    "parse_sample_key",
    "semantic_class",
    "semantic_histogram",
    "validate_semantic_mask",
]
