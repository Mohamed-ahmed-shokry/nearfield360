"""Synthetic fisheye camera, parking-aisle map, and pixel detection generator."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from nearfield360.data.calibration import (
    CameraCalibration,
    ExtrinsicParameters,
    IntrinsicParameters,
)
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.selfcal.association import PolylineMap
from nearfield360.selfcal.models import RotationDelta


def demo_fisheye_camera(camera_id: CameraId = CameraId.FRONT) -> CalibratedCamera:
    """Return a deterministic 1280x966 calibrated fisheye camera for tests and CLI demos."""
    calibration = CameraCalibration(
        name=camera_id,
        extrinsic=ExtrinsicParameters(
            quaternion=(-0.5, 0.5, -0.5, 0.5),
            translation=(3.6, 0.0, 0.8),
        ),
        intrinsic=IntrinsicParameters(
            aspect_ratio=1.0,
            cx_offset=0.0,
            cy_offset=0.0,
            height=966,
            width=1280,
            k1=420.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            model="radial_poly",
            poly_order=4,
        ),
    )
    return CalibratedCamera.from_calibration(calibration, theta_max=2.2)


def demo_parking_aisle_polylines() -> list[list[tuple[float, float]]]:
    """Return vehicle-frame polylines for a parking aisle with cross-bars and bay dividers."""
    return [
        # Longitudinal aisle boundaries
        [(4.5, -2.2), (12.5, -2.2)],
        [(4.5, 2.2), (12.5, 2.2)],
        # Transverse stop bars / crosswalk edges across the aisle
        [(5.5, -2.2), (5.5, 2.2)],
        [(8.0, -2.2), (8.0, 2.2)],
        [(10.5, -2.2), (10.5, 2.2)],
        # Parking bay dividers flanking the aisle
        [(6.5, -4.5), (6.5, -2.2)],
        [(9.0, -4.5), (9.0, -2.2)],
        [(6.5, 2.2), (6.5, 4.5)],
        [(9.0, 2.2), (9.0, 4.5)],
    ]


def demo_parking_aisle_map() -> PolylineMap:
    """Return a PolylineMap of the demo parking-aisle markings."""
    return PolylineMap(demo_parking_aisle_polylines())


def synthesize_pixel_detections(
    camera: CalibratedCamera,
    polylines: Sequence[Sequence[tuple[float, float]]],
    true_delta: RotationDelta,
    *,
    points_per_segment: int = 16,
    pixel_noise_px: float = 1.0,
    clutter_fraction: float = 0.0,
    ground_z: float = 0.0,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Project ground markings through a drifted camera with pixel noise and clutter.

    ``true_delta`` is the camera's residual rotation from factory nominal to true
    vehicle-frame rays, matching ``ExtrinsicSelfCalibrator``'s convention.
    """
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be positive")
    if pixel_noise_px < 0.0:
        raise ValueError("pixel_noise_px must be non-negative")
    if not 0.0 <= clutter_fraction < 1.0:
        raise ValueError("clutter_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    ground_points: list[npt.NDArray[np.float64]] = []
    for polyline in polylines:
        for start, end in pairwise(polyline):
            t = rng.uniform(0.05, 0.95, size=points_per_segment)
            xs = start[0] + t * (end[0] - start[0])
            ys = start[1] + t * (end[1] - start[1])
            zs = np.full(points_per_segment, ground_z, dtype=np.float64)
            ground_points.append(np.column_stack([xs, ys, zs]))

    if not ground_points:
        return np.empty((0, 2), dtype=np.float64)

    all_points = np.vstack(ground_points)
    origin = camera.camera_to_vehicle.translation
    true_rays = all_points - origin
    nominal_rays = true_rays @ true_delta.matrix()  # R^T applied to row vectors
    projected = camera.project_vehicle(origin + nominal_rays, check_image_bounds=True)
    valid_pixels = projected.pixels[projected.valid].copy()

    if valid_pixels.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    if pixel_noise_px > 0.0:
        valid_pixels += rng.normal(0.0, pixel_noise_px, size=valid_pixels.shape)

    width = float(camera.model.intrinsics.width)
    height = float(camera.model.intrinsics.height)
    in_bounds = (
        (valid_pixels[:, 0] >= 0.0)
        & (valid_pixels[:, 0] < width)
        & (valid_pixels[:, 1] >= 0.0)
        & (valid_pixels[:, 1] < height)
    )
    valid_pixels = valid_pixels[in_bounds]

    if clutter_fraction > 0.0 and len(valid_pixels) > 0:
        num_clutter = round(len(valid_pixels) * clutter_fraction / (1.0 - clutter_fraction))
        if num_clutter > 0:
            clutter = np.column_stack(
                [
                    rng.uniform(0.0, width - 1.0, size=num_clutter),
                    rng.uniform(height * 0.5, height - 1.0, size=num_clutter),
                ]
            )
            valid_pixels = np.vstack([valid_pixels, clutter])

    return valid_pixels
