"""Measured dataset summaries derived from discovered local files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from nearfield360.data.images import ImageReadError, load_rgb_image
from nearfield360.data.semantic import (
    WOODSCAPE_SEMANTIC_CLASSES,
    SemanticMaskError,
    load_semantic_mask,
    semantic_histogram,
)
from nearfield360.data.woodscape import CameraId, WoodScapeDataset

Resolution = tuple[int, int]


class DatasetStatisticsError(ValueError):
    """Raised when measured statistics cannot be computed from malformed files."""


@dataclass(frozen=True)
class DatasetStatistics:
    sample_count: int
    frame_count: int
    camera_counts: Mapping[CameraId, int]
    resolution_counts: Mapping[Resolution, int]
    rgb_bytes: int
    previous_image_count: int
    semantic_mask_count: int
    calibration_count: int
    semantic_pixel_counts: Mapping[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "sample_count": self.sample_count,
            "frame_count": self.frame_count,
            "camera_counts": {camera.value: count for camera, count in self.camera_counts.items()},
            "resolutions": [
                {"height": height, "width": width, "count": count}
                for (height, width), count in self.resolution_counts.items()
            ],
            "rgb_bytes": self.rgb_bytes,
            "available": {
                "previous_images": self.previous_image_count,
                "semantic_masks": self.semantic_mask_count,
                "calibrations": self.calibration_count,
            },
            "semantic_pixel_counts": (
                None if self.semantic_pixel_counts is None else dict(self.semantic_pixel_counts)
            ),
        }


def compute_dataset_statistics(
    dataset: WoodScapeDataset,
    *,
    include_semantic_pixels: bool = False,
) -> DatasetStatistics:
    """Decode local RGB files and optionally aggregate validated semantic labels."""
    camera_counts = dict.fromkeys(CameraId, 0)
    resolution_counts: Counter[Resolution] = Counter()
    semantic_counts = (
        Counter({item.name: 0 for item in WOODSCAPE_SEMANTIC_CLASSES})
        if include_semantic_pixels
        else None
    )
    frame_ids: set[str] = set()
    rgb_bytes = 0
    previous_count = 0
    semantic_count = 0
    calibration_count = 0

    for sample in dataset:
        frame_ids.add(sample.key.frame_id)
        camera_counts[sample.key.camera] += 1
        previous_count += sample.previous_image_path is not None
        semantic_count += sample.semantic_mask_path is not None
        calibration_count += sample.calibration_path is not None
        try:
            rgb_bytes += sample.image_path.stat().st_size
            image = load_rgb_image(sample.image_path)
        except (OSError, ImageReadError) as exc:
            raise DatasetStatisticsError(
                f"Unable to measure RGB sample {sample.key.stem}: {exc}"
            ) from exc
        resolution = (int(image.shape[0]), int(image.shape[1]))
        resolution_counts[resolution] += 1

        if semantic_counts is not None and sample.semantic_mask_path is not None:
            try:
                mask = load_semantic_mask(sample.semantic_mask_path)
            except (ImageReadError, SemanticMaskError) as exc:
                raise DatasetStatisticsError(
                    f"Unable to measure semantic mask {sample.key.stem}: {exc}"
                ) from exc
            if mask.shape != resolution:
                raise DatasetStatisticsError(
                    f"Semantic mask shape {mask.shape} does not match RGB shape {resolution} "
                    f"for {sample.key.stem}"
                )
            semantic_counts.update(semantic_histogram(mask))

    return DatasetStatistics(
        sample_count=len(dataset),
        frame_count=len(frame_ids),
        camera_counts=MappingProxyType(camera_counts),
        resolution_counts=MappingProxyType(dict(sorted(resolution_counts.items()))),
        rgb_bytes=rgb_bytes,
        previous_image_count=previous_count,
        semantic_mask_count=semantic_count,
        calibration_count=calibration_count,
        semantic_pixel_counts=(
            None if semantic_counts is None else MappingProxyType(dict(semantic_counts))
        ),
    )


__all__ = [
    "DatasetStatistics",
    "DatasetStatisticsError",
    "Resolution",
    "compute_dataset_statistics",
]
