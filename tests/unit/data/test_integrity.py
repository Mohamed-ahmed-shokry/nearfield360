import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from nearfield360.data import (
    CameraId,
    ValidationPolicy,
    WoodScapeDataset,
    validate_dataset,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _calibration_payload(
    *, camera: CameraId = CameraId.FRONT, width: int = 3, height: int = 2
) -> dict[str, Any]:
    return {
        "extrinsic": {
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 1.0],
        },
        "intrinsic": {
            "aspect_ratio": 1.0,
            "cx_offset": 0.0,
            "cy_offset": 0.0,
            "height": height,
            "k1": 100.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "model": "radial_poly",
            "poly_order": 4,
            "width": width,
        },
        "name": camera.value,
    }


def _complete_dataset(root: Path) -> WoodScapeDataset:
    _write_image(root / "rgb_images/00001_FV.png", np.zeros((2, 3, 3), dtype=np.uint8))
    _write_image(root / "previous_images/00001_FV_prev.png", np.zeros((2, 3, 3), dtype=np.uint8))
    _write_image(
        root / "semantic_annotations/gtLabels/00001_FV.png",
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8),
    )
    calibration_path = root / "calibration_data/00001_FV.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(json.dumps(_calibration_payload()), encoding="utf-8")
    return WoodScapeDataset.discover(root)


def test_validate_dataset_accepts_consistent_complete_sample(tmp_path: Path) -> None:
    report = validate_dataset(
        _complete_dataset(tmp_path),
        policy=ValidationPolicy(
            require_previous_images=True,
            require_semantic_masks=True,
            require_calibrations=True,
        ),
    )

    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.sample_count == 1
    assert report.camera_counts[CameraId.FRONT] == 1
    assert report.camera_counts[CameraId.REAR] == 0
    assert report.previous_image_count == 1
    assert report.semantic_mask_count == 1
    assert report.calibration_count == 1
    assert report.issues == ()


def test_validate_dataset_reports_required_missing_components(tmp_path: Path) -> None:
    _write_image(tmp_path / "rgb_images/00001_RV.png", np.zeros((2, 3, 3), dtype=np.uint8))
    dataset = WoodScapeDataset.discover(tmp_path)

    report = validate_dataset(
        dataset,
        policy=ValidationPolicy(
            require_previous_images=True,
            require_semantic_masks=True,
            require_calibrations=True,
        ),
    )

    assert not report.is_valid
    assert {issue.code for issue in report.issues} == {
        "missing_previous_image",
        "missing_semantic_mask",
        "missing_calibration",
    }


def test_validate_dataset_reports_cross_file_mismatches(tmp_path: Path) -> None:
    dataset = _complete_dataset(tmp_path)
    _write_image(
        tmp_path / "previous_images/00001_FV_prev.png", np.zeros((1, 3, 3), dtype=np.uint8)
    )
    _write_image(
        tmp_path / "semantic_annotations/gtLabels/00001_FV.png",
        np.zeros((2, 2), dtype=np.uint8),
    )
    calibration_path = tmp_path / "calibration_data/00001_FV.json"
    calibration_path.write_text(
        json.dumps(_calibration_payload(camera=CameraId.REAR, width=4)),
        encoding="utf-8",
    )

    report = validate_dataset(dataset)

    assert {issue.code for issue in report.issues} == {
        "previous_image_size_mismatch",
        "semantic_mask_size_mismatch",
        "calibration_camera_mismatch",
        "calibration_size_mismatch",
    }


def test_validate_dataset_continues_after_malformed_files(tmp_path: Path) -> None:
    dataset = _complete_dataset(tmp_path)
    (tmp_path / "rgb_images/00001_FV.png").write_text("broken", encoding="utf-8")
    (tmp_path / "previous_images/00001_FV_prev.png").write_text("broken", encoding="utf-8")
    _write_image(
        tmp_path / "semantic_annotations/gtLabels/00001_FV.png",
        np.array([[10]], dtype=np.uint8),
    )
    (tmp_path / "calibration_data/00001_FV.json").write_text("broken", encoding="utf-8")

    report = validate_dataset(dataset)

    assert {issue.code for issue in report.issues} == {
        "invalid_rgb_image",
        "invalid_previous_image",
        "invalid_semantic_mask",
        "invalid_calibration",
    }


def test_validation_policy_can_skip_optional_file_contents(tmp_path: Path) -> None:
    dataset = _complete_dataset(tmp_path)
    (tmp_path / "previous_images/00001_FV_prev.png").write_text("broken", encoding="utf-8")
    (tmp_path / "semantic_annotations/gtLabels/00001_FV.png").write_text("broken", encoding="utf-8")
    (tmp_path / "calibration_data/00001_FV.json").write_text("broken", encoding="utf-8")

    report = validate_dataset(
        dataset,
        policy=ValidationPolicy(
            validate_previous_images=False,
            validate_semantic_masks=False,
            validate_calibrations=False,
        ),
    )

    assert report.is_valid


def test_validation_report_marks_truncated_issues_invalid(tmp_path: Path) -> None:
    _write_image(tmp_path / "rgb_images/00001_FV.png", np.zeros((1, 1, 3), dtype=np.uint8))
    dataset = WoodScapeDataset.discover(tmp_path)

    report = validate_dataset(
        dataset,
        policy=ValidationPolicy(
            require_previous_images=True,
            require_semantic_masks=True,
            max_issues=1,
        ),
    )

    assert report.error_count == 1
    assert report.issues_truncated
    assert not report.is_valid


def test_validation_policy_rejects_non_positive_issue_limit() -> None:
    with pytest.raises(ValueError, match="max_issues must be positive"):
        ValidationPolicy(max_issues=0)
