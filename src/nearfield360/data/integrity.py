"""Structured integrity checks for discovered WoodScape samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from nearfield360.data.calibration import CalibrationError, load_calibration
from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import SemanticMaskError, load_semantic_mask
from nearfield360.data.woodscape import CameraId, SampleKey, WoodScapeDataset, WoodScapeSample


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable integrity finding with optional sample context."""

    severity: ValidationSeverity
    code: str
    message: str
    sample_key: SampleKey | None = None


@dataclass(frozen=True)
class ValidationPolicy:
    """Control required annotations and the depth of local validation."""

    require_previous_images: bool = False
    require_semantic_masks: bool = False
    require_calibrations: bool = False
    validate_previous_images: bool = True
    validate_semantic_masks: bool = True
    validate_calibrations: bool = True
    max_issues: int = 1000

    def __post_init__(self) -> None:
        if self.max_issues <= 0:
            raise ValueError("max_issues must be positive")


@dataclass(frozen=True)
class DatasetValidationReport:
    """Integrity summary suitable for both CLI rendering and JSON serialization."""

    sample_count: int
    camera_counts: Mapping[CameraId, int]
    previous_image_count: int
    semantic_mask_count: int
    calibration_count: int
    issues: tuple[ValidationIssue, ...]
    issues_truncated: bool = False

    @property
    def error_count(self) -> int:
        return sum(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is ValidationSeverity.WARNING for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0 and not self.issues_truncated


class _IssueCollector:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.issues: list[ValidationIssue] = []
        self.truncated = False

    def error(self, code: str, message: str, sample: WoodScapeSample) -> None:
        if len(self.issues) < self._limit:
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=code,
                    message=message,
                    sample_key=sample.key,
                )
            )
        else:
            self.truncated = True


def _load_rgb(sample: WoodScapeSample, collector: _IssueCollector) -> npt.NDArray[np.uint8] | None:
    try:
        return load_rgb_image(sample.image_path)
    except ImageReadError as exc:
        collector.error("invalid_rgb_image", str(exc), sample)
        return None


def _check_required_files(
    sample: WoodScapeSample,
    policy: ValidationPolicy,
    collector: _IssueCollector,
) -> None:
    requirements = (
        (
            policy.require_previous_images,
            sample.previous_image_path,
            "missing_previous_image",
            "Previous image is required but missing",
        ),
        (
            policy.require_semantic_masks,
            sample.semantic_mask_path,
            "missing_semantic_mask",
            "Semantic mask is required but missing",
        ),
        (
            policy.require_calibrations,
            sample.calibration_path,
            "missing_calibration",
            "Calibration is required but missing",
        ),
    )
    for required, path, code, message in requirements:
        if required and path is None:
            collector.error(code, message, sample)


def _validate_previous_image(
    sample: WoodScapeSample,
    rgb: npt.NDArray[np.uint8] | None,
    collector: _IssueCollector,
) -> None:
    if sample.previous_image_path is None:
        return
    try:
        previous = load_rgb_image(sample.previous_image_path)
    except ImageReadError as exc:
        collector.error("invalid_previous_image", str(exc), sample)
        return
    if rgb is not None and previous.shape[:2] != rgb.shape[:2]:
        collector.error(
            "previous_image_size_mismatch",
            f"Previous image shape {previous.shape[:2]} does not match RGB shape {rgb.shape[:2]}",
            sample,
        )


def _validate_semantic_mask(
    sample: WoodScapeSample,
    rgb: npt.NDArray[np.uint8] | None,
    collector: _IssueCollector,
) -> None:
    if sample.semantic_mask_path is None:
        return
    try:
        mask = load_semantic_mask(sample.semantic_mask_path)
    except (ImageReadError, SemanticMaskError) as exc:
        collector.error("invalid_semantic_mask", str(exc), sample)
        return
    if rgb is not None and mask.shape != rgb.shape[:2]:
        collector.error(
            "semantic_mask_size_mismatch",
            f"Semantic mask shape {mask.shape} does not match RGB shape {rgb.shape[:2]}",
            sample,
        )


def _validate_calibration(
    sample: WoodScapeSample,
    rgb: npt.NDArray[np.uint8] | None,
    collector: _IssueCollector,
) -> None:
    if sample.calibration_path is None:
        return
    try:
        calibration = load_calibration(sample.calibration_path)
    except CalibrationError as exc:
        collector.error("invalid_calibration", str(exc), sample)
        return
    if calibration.name is not sample.key.camera:
        collector.error(
            "calibration_camera_mismatch",
            f"Calibration identifies {calibration.name.value}, expected {sample.key.camera.value}",
            sample,
        )
    expected_shape = (calibration.intrinsic.height, calibration.intrinsic.width)
    if rgb is not None and expected_shape != rgb.shape[:2]:
        collector.error(
            "calibration_size_mismatch",
            f"Calibration size {expected_shape} does not match RGB shape {rgb.shape[:2]}",
            sample,
        )


def validate_dataset(
    dataset: WoodScapeDataset,
    policy: ValidationPolicy | None = None,
) -> DatasetValidationReport:
    """Validate every indexed sample without stopping at the first malformed file."""
    policy = ValidationPolicy() if policy is None else policy
    collector = _IssueCollector(policy.max_issues)
    camera_counts = dict.fromkeys(CameraId, 0)
    previous_count = 0
    semantic_count = 0
    calibration_count = 0

    for sample in dataset:
        camera_counts[sample.key.camera] += 1
        previous_count += sample.previous_image_path is not None
        semantic_count += sample.semantic_mask_path is not None
        calibration_count += sample.calibration_path is not None
        _check_required_files(sample, policy, collector)
        rgb = _load_rgb(sample, collector)
        if policy.validate_previous_images:
            _validate_previous_image(sample, rgb, collector)
        if policy.validate_semantic_masks:
            _validate_semantic_mask(sample, rgb, collector)
        if policy.validate_calibrations:
            _validate_calibration(sample, rgb, collector)

    return DatasetValidationReport(
        sample_count=len(dataset),
        camera_counts=MappingProxyType(camera_counts),
        previous_image_count=previous_count,
        semantic_mask_count=semantic_count,
        calibration_count=calibration_count,
        issues=tuple(collector.issues),
        issues_truncated=collector.truncated,
    )


__all__ = [
    "DatasetValidationReport",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationSeverity",
    "validate_dataset",
]
