"""A validated four-camera surround rig sharing the vehicle frame.

The rig keeps each camera's fisheye model and camera-to-vehicle extrinsics
together so multi-camera projection, ray generation, and ground footprints use
consistent frames. Vehicle axes are X forward, Y left, and Z up; camera axes
are X right, Y down, and Z forward.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.data.calibration import CameraCalibration
from nearfield360.data.woodscape import CameraId
from nearfield360.geometry.camera import CalibratedCamera, PixelRayResult
from nearfield360.geometry.fisheye import ProjectionResult
from nearfield360.geometry.ground import GroundIntersection, intersect_ground


@dataclass(frozen=True, slots=True)
class RigGroundFootprints:
    """Per-camera ground intersections keyed by camera identity."""

    footprints: Mapping[CameraId, GroundIntersection]

    def __getitem__(self, camera: CameraId) -> GroundIntersection:
        return self.footprints[camera]


@dataclass(frozen=True, slots=True)
class SurroundRig:
    """An immutable, non-empty set of calibrated cameras with unique names.

    The rig does not guess synchronization, overlap, or coverage: it only
    guarantees that every stored camera maps into the same vehicle frame.
    Per-camera operations preserve each camera's batch shapes; invalid inputs
    produce NaNs for that camera without affecting the others.
    """

    _cameras: tuple[CalibratedCamera, ...] = field(repr=False)
    _by_name: dict[CameraId, CalibratedCamera] = field(init=False, repr=False)

    def __init__(self, cameras: tuple[CalibratedCamera, ...] | list[CalibratedCamera]) -> None:
        items = tuple(cameras)
        if not items:
            raise ValueError("a surround rig requires at least one calibrated camera")
        if any(not isinstance(item, CalibratedCamera) for item in items):
            raise ValueError("a surround rig requires CalibratedCamera instances")
        names = [item.name for item in items]
        if len(set(names)) != len(names):
            raise ValueError("surround rig cameras must have unique names")
        object.__setattr__(self, "_cameras", items)
        object.__setattr__(self, "_by_name", {item.name: item for item in items})

    @classmethod
    def from_calibrations(
        cls,
        calibrations: Mapping[CameraId, CameraCalibration]
        | tuple[CameraCalibration, ...]
        | list[CameraCalibration],
        *,
        theta_max: float | Mapping[CameraId, float],
    ) -> Self:
        """Build one camera per record with explicit angular limits.

        ``theta_max`` is either a single limit applied to every camera or a
        per-camera mapping keyed by camera identity. Limits are required; no
        usable field of view is guessed from image dimensions.
        """
        records = (
            tuple(calibrations.values())
            if isinstance(calibrations, Mapping)
            else tuple(calibrations)
        )
        if not records:
            raise ValueError("a surround rig requires at least one calibration")
        if isinstance(theta_max, Mapping):
            limits = {
                record.name: theta_max[record.name]
                for record in records
                if record.name in theta_max
            }
            if set(limits) != {record.name for record in records}:
                raise ValueError("per-camera theta_max must cover every calibration exactly once")
            cameras = tuple(
                CalibratedCamera.from_calibration(record, theta_max=limits[record.name])
                for record in records
            )
        else:
            cameras = tuple(
                CalibratedCamera.from_calibration(record, theta_max=theta_max) for record in records
            )
        return cls(cameras)

    @property
    def names(self) -> tuple[CameraId, ...]:
        """Camera identities in construction order."""
        return tuple(camera.name for camera in self._cameras)

    def __len__(self) -> int:
        return len(self._cameras)

    def __iter__(self) -> Iterator[CalibratedCamera]:
        return iter(self._cameras)

    def get(self, camera: CameraId) -> CalibratedCamera:
        """Return the camera for an identity with a useful error for misses."""
        try:
            return self._by_name[camera]
        except KeyError as exc:
            expected = ", ".join(name.value for name in self.names)
            raise KeyError(f"Unknown rig camera: {camera!r}; expected one of {expected}") from exc

    def project_vehicle(
        self, points: ArrayLike, *, check_image_bounds: bool = True
    ) -> dict[CameraId, ProjectionResult]:
        """Project vehicle-frame points into every rig camera."""
        return {
            camera.name: camera.project_vehicle(points, check_image_bounds=check_image_bounds)
            for camera in self._cameras
        }

    def pixel_rays_vehicle(
        self,
        pixels_by_camera: Mapping[CameraId, ArrayLike],
        *,
        check_image_bounds: bool = True,
    ) -> dict[CameraId, PixelRayResult]:
        """Return vehicle-frame rays for per-camera pixel batches."""
        if set(pixels_by_camera) != set(self.names):
            raise ValueError(
                "pixel batches must cover every rig camera exactly once, got "
                f"{sorted(str(key) for key in pixels_by_camera)}"
            )
        return {
            name: self._by_name[name].pixel_rays_vehicle(
                pixels_by_camera[name], check_image_bounds=check_image_bounds
            )
            for name in self.names
        }

    def ground_footprints(
        self,
        pixels_by_camera: Mapping[CameraId, ArrayLike],
        *,
        ground_z: float = 0.0,
        max_distance: float | None = None,
        check_image_bounds: bool = True,
    ) -> RigGroundFootprints:
        """Unproject per-camera pixels and intersect the rays with the ground.

        Invalid fisheye rays never reach the intersection: they stay invalid
        footprints rather than being extrapolated. Use ``max_distance`` to keep
        far, uncertainty-dominated intersections unknown.
        """
        rays = self.pixel_rays_vehicle(pixels_by_camera, check_image_bounds=check_image_bounds)
        footprints: dict[CameraId, GroundIntersection] = {}
        for name in self.names:
            ray = rays[name]
            footprints[name] = intersect_ground(
                ray.origins, ray.directions, ground_z=ground_z, max_distance=max_distance
            )
            # Invalid rays already carry NaN origins/directions, which the
            # intersection marks invalid; mirror the fisheye validity so a
            # caller never sees a footprint where the pixel itself was invalid.
            combined_valid = ray.valid & footprints[name].valid
            points = np.where(combined_valid[..., None], footprints[name].points, np.nan)
            distances = np.where(combined_valid, footprints[name].distances, np.nan)
            footprints[name] = GroundIntersection(
                points=np.asarray(points, dtype=np.float64),
                distances=np.asarray(distances, dtype=np.float64),
                valid=np.asarray(combined_valid, dtype=np.bool_),
            )
        return RigGroundFootprints(footprints=footprints)

    def origins_array(self) -> dict[CameraId, NDArray[np.float64]]:
        """Return each camera center in vehicle-frame metres."""
        return {
            camera.name: camera.camera_to_vehicle.translation.copy() for camera in self._cameras
        }


__all__ = ["RigGroundFootprints", "SurroundRig"]
