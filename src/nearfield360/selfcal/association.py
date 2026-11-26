"""Associate fisheye pixel detections of painted lines with HD-map polylines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

from nearfield360.geometry.camera import CalibratedCamera
from nearfield360.geometry.ground import intersect_ground
from nearfield360.selfcal.models import GroundObservation, RotationDelta

if TYPE_CHECKING:
    from nearfield360.hdmap.models import HDVectorMap

_MIN_SEGMENT_M = 1e-9


@dataclass(frozen=True)
class NearestResult:
    """Nearest map points for a batch of query points.

    ``normals`` are unit vectors perpendicular to the nearest segment, so the
    point-to-line residual of a query ``p`` is ``dot(p - point, normal)``.
    """

    points: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]


class PolylineMap:
    """Vehicle-frame polylines supporting vectorized nearest-point queries."""

    def __init__(self, polylines: Sequence[Sequence[tuple[float, float]]]) -> None:
        starts: list[tuple[float, float]] = []
        ends: list[tuple[float, float]] = []
        for index, polyline in enumerate(polylines):
            if len(polyline) < 2:
                raise ValueError(f"polyline {index} needs at least 2 points")
            for x, y in polyline:
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"polyline {index} has non-finite coordinates")
            for first, second in pairwise(polyline):
                if math.hypot(second[0] - first[0], second[1] - first[1]) > _MIN_SEGMENT_M:
                    starts.append(first)
                    ends.append(second)
        if not starts:
            raise ValueError("map must contain at least one non-degenerate segment")
        self._starts = np.asarray(starts, dtype=np.float64)
        self._ends = np.asarray(ends, dtype=np.float64)
        self._vectors = self._ends - self._starts
        self._lengths_sq = np.sum(self._vectors**2, axis=1)
        lengths = np.sqrt(self._lengths_sq)
        self._normals = (
            np.column_stack([-self._vectors[:, 1], self._vectors[:, 0]]) / lengths[:, None]
        )

    @property
    def num_segments(self) -> int:
        """Number of non-degenerate segments in the map."""
        return int(self._starts.shape[0])

    def nearest(self, points_xy: npt.ArrayLike) -> NearestResult:
        """Return the nearest point, distance, and line normal for each query point."""
        queries = np.asarray(points_xy, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 2:
            raise ValueError(f"points_xy must have shape (M, 2), got {queries.shape}")
        offsets = queries[:, None, :] - self._starts[None, :, :]
        fraction = np.clip(
            np.sum(offsets * self._vectors[None, :, :], axis=2) / self._lengths_sq[None, :],
            0.0,
            1.0,
        )
        closest = self._starts[None, :, :] + fraction[:, :, None] * self._vectors[None, :, :]
        distances = np.linalg.norm(queries[:, None, :] - closest, axis=2)
        best = np.argmin(distances, axis=1)
        rows = np.arange(queries.shape[0])
        return NearestResult(
            points=closest[rows, best],
            distances=distances[rows, best],
            normals=self._normals[best],
        )

    @classmethod
    def from_hdmap(cls, hd_map: HDVectorMap) -> Self:
        """Build a polyline map from lanelet and parking-space boundaries in an HDVectorMap."""
        polylines: list[list[tuple[float, float]]] = []
        for tile in hd_map.tiles:
            for lanelet in tile.lanelets:
                polylines.append([(p.x, p.y) for p in lanelet.left_boundary.points])
                polylines.append([(p.x, p.y) for p in lanelet.right_boundary.points])
            for space in tile.parking_spaces:
                pts = [(p.x, p.y) for p in space.boundary.points]
                if space.boundary.is_closed and len(pts) >= 3:
                    pts.append(pts[0])
                polylines.append(pts)
        return cls(polylines)


def associate_pixel_detections(
    camera: CalibratedCamera,
    pixels: npt.ArrayLike,
    polyline_map: PolylineMap,
    *,
    current_delta: RotationDelta | None = None,
    gate_m: float = 0.6,
    ground_z: float = 0.0,
    max_distance: float | None = 25.0,
    check_image_bounds: bool = True,
) -> list[GroundObservation]:
    """Unproject fisheye pixel detections and associate them to nearest map polylines.

    Each returned ``GroundObservation`` keeps the nominal vehicle-frame ray from
    ``camera`` so the estimator solves directly for the total residual rotation.
    """
    if not math.isfinite(gate_m) or gate_m <= 0.0:
        raise ValueError("gate_m must be a positive finite distance")
    pixel_array = np.asarray(pixels, dtype=np.float64)
    if pixel_array.size == 0:
        return []
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pixel_array.shape}")

    ray_result = camera.pixel_rays_vehicle(pixel_array, check_image_bounds=check_image_bounds)
    valid_rays = ray_result.valid
    if not np.any(valid_rays):
        return []

    nominal_dirs = ray_result.directions[valid_rays]
    origins = ray_result.origins[valid_rays]
    if current_delta is not None:
        rotated_dirs = nominal_dirs @ current_delta.matrix().T
    else:
        rotated_dirs = nominal_dirs

    ground = intersect_ground(origins, rotated_dirs, ground_z=ground_z, max_distance=max_distance)
    if not np.any(ground.valid):
        return []

    hit_nominal = nominal_dirs[ground.valid]
    hit_xy = ground.points[ground.valid, :2]
    nearest = polyline_map.nearest(hit_xy)

    observations: list[GroundObservation] = []
    for idx in np.flatnonzero(nearest.distances <= gate_m):
        ray = hit_nominal[idx]
        ref = nearest.points[idx]
        normal = nearest.normals[idx]
        observations.append(
            GroundObservation(
                camera_id=camera.name,
                direction=(float(ray[0]), float(ray[1]), float(ray[2])),
                reference_x_m=float(ref[0]),
                reference_y_m=float(ref[1]),
                reference_normal=(float(normal[0]), float(normal[1])),
            )
        )
    return observations
