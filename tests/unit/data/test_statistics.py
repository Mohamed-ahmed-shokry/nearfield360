from pathlib import Path

import cv2
import numpy as np
import pytest

from nearfield360.data import (
    CameraId,
    DatasetStatisticsError,
    WoodScapeDataset,
    compute_dataset_statistics,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_compute_dataset_statistics_measures_files_and_availability(tmp_path: Path) -> None:
    front_path = tmp_path / "rgb_images/00001_FV.png"
    rear_path = tmp_path / "rgb_images/00001_RV.png"
    _write_image(front_path, np.zeros((2, 3, 3), dtype=np.uint8))
    _write_image(rear_path, np.zeros((4, 5, 3), dtype=np.uint8))
    _write_image(
        tmp_path / "previous_images/00001_FV_prev.png",
        np.zeros((2, 3, 3), dtype=np.uint8),
    )

    statistics = compute_dataset_statistics(WoodScapeDataset.discover(tmp_path))

    assert statistics.sample_count == 2
    assert statistics.frame_count == 1
    assert statistics.camera_counts[CameraId.FRONT] == 1
    assert statistics.camera_counts[CameraId.REAR] == 1
    assert statistics.resolution_counts == {(2, 3): 1, (4, 5): 1}
    assert statistics.rgb_bytes == front_path.stat().st_size + rear_path.stat().st_size
    assert statistics.previous_image_count == 1
    assert statistics.semantic_mask_count == 0
    assert statistics.calibration_count == 0
    assert statistics.semantic_pixel_counts is None


def test_compute_dataset_statistics_aggregates_semantic_pixels(tmp_path: Path) -> None:
    _write_image(tmp_path / "rgb_images/00001_FV.png", np.zeros((2, 3, 3), dtype=np.uint8))
    _write_image(
        tmp_path / "semantic_annotations/gtLabels/00001_FV.png",
        np.array([[0, 1, 1], [3, 6, 9]], dtype=np.uint8),
    )

    statistics = compute_dataset_statistics(
        WoodScapeDataset.discover(tmp_path), include_semantic_pixels=True
    )

    assert statistics.semantic_pixel_counts is not None
    assert statistics.semantic_pixel_counts["road"] == 2
    assert statistics.semantic_pixel_counts["curb"] == 1
    assert statistics.semantic_pixel_counts["person"] == 0
    assert sum(statistics.semantic_pixel_counts.values()) == 6
    payload = statistics.as_dict()
    assert payload["resolutions"] == [{"height": 2, "width": 3, "count": 1}]
    assert payload["semantic_pixel_counts"]["vehicles"] == 1


def test_requested_semantic_counts_are_zero_when_no_masks_exist(tmp_path: Path) -> None:
    _write_image(tmp_path / "rgb_images/00001_FV.png", np.zeros((2, 3, 3), dtype=np.uint8))

    statistics = compute_dataset_statistics(
        WoodScapeDataset.discover(tmp_path), include_semantic_pixels=True
    )

    assert statistics.semantic_mask_count == 0
    assert statistics.semantic_pixel_counts is not None
    assert len(statistics.semantic_pixel_counts) == 10
    assert sum(statistics.semantic_pixel_counts.values()) == 0


def test_semantic_pixel_aggregation_rejects_shape_mismatch(tmp_path: Path) -> None:
    _write_image(tmp_path / "rgb_images/00001_FV.png", np.zeros((2, 3, 3), dtype=np.uint8))
    _write_image(
        tmp_path / "semantic_annotations/gtLabels/00001_FV.png",
        np.zeros((1, 3), dtype=np.uint8),
    )

    with pytest.raises(DatasetStatisticsError, match="does not match RGB shape"):
        compute_dataset_statistics(
            WoodScapeDataset.discover(tmp_path), include_semantic_pixels=True
        )


def test_statistics_wrap_malformed_rgb_and_semantic_files(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb_images/00001_FV.png"
    image_path.parent.mkdir()
    image_path.write_text("broken", encoding="utf-8")
    dataset = WoodScapeDataset.discover(tmp_path)

    with pytest.raises(DatasetStatisticsError, match="Unable to measure RGB sample"):
        compute_dataset_statistics(dataset)

    _write_image(image_path, np.zeros((1, 1, 3), dtype=np.uint8))
    semantic_path = tmp_path / "semantic_annotations/gtLabels/00001_FV.png"
    semantic_path.parent.mkdir(parents=True)
    semantic_path.write_text("broken", encoding="utf-8")
    dataset = WoodScapeDataset.discover(tmp_path)

    with pytest.raises(DatasetStatisticsError, match="Unable to measure semantic mask"):
        compute_dataset_statistics(dataset, include_semantic_pixels=True)
