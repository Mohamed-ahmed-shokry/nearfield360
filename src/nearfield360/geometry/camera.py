"""WoodScape fisheye geometry in the metric vehicle coordinate frame.

Vehicle axes are X forward, Y left, and Z up. Camera axes are X right, Y down,
and Z forward. Calibration extrinsics map camera coordinates to vehicle
coordinates; their translation is the camera center in vehicle-frame metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.data.calibration import CameraCalibration
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.fisheye import ProjectionResult, RadialPolynomialFisheye
from nearfield360.geometry.transforms import RigidTransform


@dataclass(frozen=True, slots=True)
class PixelRayResult:
    """Vehicle-frame origins/directions ``(..., 3)`` and validity ``(...)``.

    Origins are camera centers in metres and directions have unit length.
    Both arrays contain NaNs for invalid pixels. A point along a valid ray is
    ``origin + distance * direction`` for a caller-supplied distance in metres;
    pixels alone do not determine that distance or any object's depth.
    """

    origins: NDArray[np.float64]
    directions: NDArray[np.float64]
    valid: NDArray[np.bool_]


def _point_array(values: ArrayLike) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError("points must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(array, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (..., 3), got {points.shape}")
    return points


@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    """A fisheye model and camera-to-vehicle extrinsics with explicit frames.

    Use :meth:`from_calibration` to build both components from the same record.
    An explicit ``theta_max`` in radians is required; no usable field of view
    is guessed from image dimensions. Geometric validity does not establish
    visibility, occlusion, calibration accuracy, or an object's distance.
    """

    name: CameraId
    model: RadialPolynomialFisheye
    camera_to_vehicle: RigidTransform
    vehicle_to_camera: RigidTransform = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vehicle_to_camera", self.camera_to_vehicle.inverse())

    @classmethod
    def from_calibration(cls, calibration: CameraCalibration, *, theta_max: float) -> Self:
        """Use WoodScape's XYZW camera-to-vehicle quaternion and metre translation."""
        return cls(
            name=calibration.name,
            model=RadialPolynomialFisheye(calibration.intrinsic, theta_max=theta_max),
            camera_to_vehicle=RigidTransform.from_quaternion(
                calibration.extrinsic.quaternion, calibration.extrinsic.translation
            ),
        )

    def project_vehicle(
        self, points: ArrayLike, *, check_image_bounds: bool = True
    ) -> ProjectionResult:
        """Project vehicle-frame points in metres, preserving shape ``(..., 3)``.

        Non-finite points and the exact camera center are invalid and produce
        NaN pixels. Other angular/image validity rules come from the fisheye
        model. Only finite nonzero displacements reach the rigid transform.
        """
        coordinates = _point_array(points)
        flat = coordinates.reshape(-1, 3)
        finite_indices = np.flatnonzero(np.isfinite(flat).all(axis=1))
        camera_directions = np.full(flat.shape, np.nan, dtype=np.float64)
        if finite_indices.size:
            finite_points = flat[finite_indices]
            center = self.camera_to_vehicle.translation
            with np.errstate(over="ignore", invalid="ignore"):
                relative = finite_points - center
            overflow = ~np.isfinite(relative).all(axis=1)
            if np.any(overflow):
                # A direction remains representable even if opposite extreme
                # coordinates overflow their metric displacement in float64.
                overflow_points = finite_points[overflow]
                scales = np.maximum(np.max(np.abs(overflow_points), axis=1), np.max(np.abs(center)))
                relative[overflow] = overflow_points / scales[:, None] - center / scales[:, None]
            scales = np.max(np.abs(relative), axis=1)
            nonzero = scales > 0.0
            if np.any(nonzero):
                # Projection is scale-invariant. Bounding each displacement
                # avoids overflowing a rotation of huge finite coordinates.
                directions = relative[nonzero] / scales[nonzero, None]
                camera_directions[finite_indices[nonzero]] = (
                    self.vehicle_to_camera.transform_directions(directions)
                )
        return self.model.project(
            camera_directions.reshape(coordinates.shape), check_image_bounds=check_image_bounds
        )

    def pixel_rays_vehicle(
        self, pixels: ArrayLike, *, check_image_bounds: bool = True
    ) -> PixelRayResult:
        """Return vehicle-frame rays for pixels ``(..., 2)`` without inferring depth.

        Translation sets each ray's origin; only rotation affects its direction.
        Invalid fisheye rays are never passed to the finite-only rigid transform.
        """
        camera_rays = self.model.unproject(pixels, check_image_bounds=check_image_bounds)
        flat_valid = camera_rays.valid.reshape(-1)
        flat_rays = camera_rays.rays.reshape(-1, 3)
        origins = np.full(flat_rays.shape, np.nan, dtype=np.float64)
        directions = np.full(flat_rays.shape, np.nan, dtype=np.float64)
        if np.any(flat_valid):
            rotated = self.camera_to_vehicle.transform_directions(flat_rays[flat_valid])
            directions[flat_valid] = rotated / np.linalg.norm(rotated, axis=1, keepdims=True)
            origins[flat_valid] = self.camera_to_vehicle.translation
        return PixelRayResult(
            origins=origins.reshape(camera_rays.rays.shape),
            directions=directions.reshape(camera_rays.rays.shape),
            valid=camera_rays.valid.copy(),
        )


__all__ = ["CalibratedCamera", "PixelRayResult"]
