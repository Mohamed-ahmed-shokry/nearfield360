"""Associate fisheye pixel detections of painted lines with HD-map polylines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

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
