"""Ground ray projection engine mapping 2D bounding boxes to 3D metric BEV footprints."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.geometry.ground import intersect_ground
from nearfield360.tracking.models import GroundFootprint

if TYPE_CHECKING:
    from nearfield360.data.detection import DetectionAnnotation
    from nearfield360.geometry.camera import CalibratedCamera

# Default automotive prior bounding dimensions (width, length) in metres
DEFAULT_CLASS_DIMENSIONS: MappingProxyType[int, tuple[float, float]] = MappingProxyType(
    {
        0: (1.8, 4.5),  # vehicles
        1: (0.6, 0.6),  # person
        2: (0.7, 1.8),  # bicycle
        3: (0.4, 0.4),  # traffic_light
        4: (0.5, 0.5),  # traffic_sign
    }
)


def project_detection_to_ground(
    box_xyxy: tuple[float, float, float, float],
    *,
    class_id: int,
    class_name: str,
    camera: CalibratedCamera,
    camera_name: str = "FV",
    ground_z: float = 0.0,
    max_distance: float = 20.0,
    confidence: float = 1.0,
    custom_dimensions: tuple[float, float] | None = None,
) -> GroundFootprint | None:
    """Project 2D bounding box ground-contact point into vehicle-frame metric coordinates.

    The ground contact point is estimated at the bottom-center of the 2D bounding box
    (x_center, y_max). The fisheye camera model unprojects this point into a 3D ray in the
    vehicle coordinate system, which is then intersected with the ground plane Z = ground_z.

    Args:
        box_xyxy: (x_min, y_min, x_max, y_max) in image pixels.
        class_id: WoodScape detection class identifier.
        class_name: Class name string.
        camera: CalibratedCamera model for the originating view.
        camera_name: Name of originating camera (e.g. 'FV', 'RV', 'MVL', 'MVR').
        ground_z: Metric ground plane height in vehicle coordinates (metres).
        max_distance: Maximum allowable projection distance in metres.
        confidence: Detection confidence score in [0.0, 1.0].
        custom_dimensions: Optional (width, length) in metres overriding prior defaults.

    Returns:
        GroundFootprint if ray successfully intersects ground ahead of sensor, else None.
    """
    x_min, _y_min, x_max, y_max = box_xyxy
    contact_x = (x_min + x_max) / 2.0
    contact_y = y_max

    pixel_coord = np.array([[contact_x, contact_y]], dtype=np.float64)
    rays = camera.pixel_rays_vehicle(pixel_coord)
    if not bool(rays.valid[0]):
        return None

    intersections = intersect_ground(
        rays.origins,
        rays.directions,
        ground_z=ground_z,
        max_distance=max_distance,
    )
    if not bool(intersections.valid[0]):
        return None

    center_x = float(intersections.points[0, 0])
    center_y = float(intersections.points[0, 1])

    if custom_dimensions is not None:
        width, length = custom_dimensions
    else:
        width, length = DEFAULT_CLASS_DIMENSIONS.get(class_id, (1.0, 1.0))

    return GroundFootprint(
        center_x=round(center_x, 3),
        center_y=round(center_y, 3),
        width=round(width, 2),
        length=round(length, 2),
        class_id=class_id,
        class_name=class_name,
        camera=camera_name,
        confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3),
    )


def project_detections(
    detections: Sequence[DetectionAnnotation],
    *,
    camera: CalibratedCamera,
    camera_name: str = "FV",
    ground_z: float = 0.0,
    max_distance: float = 20.0,
    default_confidence: float = 1.0,
) -> list[GroundFootprint]:
    """Project a sequence of 2D bounding boxes into vehicle-frame metric footprints."""
    footprints: list[GroundFootprint] = []
    for det in detections:
        footprint = project_detection_to_ground(
            det.xyxy,
            class_id=det.class_id,
            class_name=det.class_name,
            camera=camera,
            camera_name=camera_name,
            ground_z=ground_z,
            max_distance=max_distance,
            confidence=default_confidence,
        )
        if footprint is not None:
            footprints.append(footprint)
    return footprints


__all__ = [
    "DEFAULT_CLASS_DIMENSIONS",
    "project_detection_to_ground",
    "project_detections",
]
