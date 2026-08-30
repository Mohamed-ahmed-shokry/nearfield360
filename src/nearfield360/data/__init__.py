"""Dataset discovery, validation, and loading interfaces."""

from nearfield360.data.calibration import (
    MAX_CALIBRATION_BYTES,
    CalibrationError,
    CameraCalibration,
    ExtrinsicParameters,
    IntrinsicParameters,
    load_calibration,
)
from nearfield360.data.images import (
    DEFAULT_IMAGE_LIMITS,
    ImageLimits,
    ImageReadError,
    load_label_image,
    load_rgb_image,
)
from nearfield360.data.integrity import (
    DatasetValidationReport,
    ValidationIssue,
    ValidationPolicy,
    ValidationSeverity,
    validate_dataset,
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
from nearfield360.data.splits import (
    DatasetSplit,
    DatasetSplits,
    SplitError,
    SplitRatios,
    create_splits,
)
from nearfield360.data.statistics import (
    DatasetStatistics,
    DatasetStatisticsError,
    Resolution,
    compute_dataset_statistics,
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
    "MAX_CALIBRATION_BYTES",
    "WOODSCAPE_SEMANTIC_CLASSES",
    "CalibrationError",
    "CameraCalibration",
    "CameraId",
    "DatasetLayoutError",
    "DatasetSplit",
    "DatasetSplits",
    "DatasetStatistics",
    "DatasetStatisticsError",
    "DatasetValidationReport",
    "ExtrinsicParameters",
    "ImageLimits",
    "ImageReadError",
    "IntrinsicParameters",
    "Resolution",
    "SampleKey",
    "SemanticClass",
    "SemanticMaskError",
    "SplitError",
    "SplitRatios",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationSeverity",
    "WoodScapeDataset",
    "WoodScapeSample",
    "colorize_semantic_mask",
    "compute_dataset_statistics",
    "create_splits",
    "load_calibration",
    "load_label_image",
    "load_rgb_image",
    "load_semantic_mask",
    "locate_dataset_root",
    "parse_sample_key",
    "semantic_class",
    "semantic_histogram",
    "validate_dataset",
    "validate_semantic_mask",
]
