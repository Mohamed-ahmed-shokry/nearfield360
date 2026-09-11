"""A local bird's-eye-view grid around the vehicle for near-field fusion.

The grid covers ``[x_min, x_max) x [y_min, y_max)`` in vehicle-frame metres
(X forward, Y left) with square cells of ``resolution`` metres. Cell indices
use image-style ``(row, col)`` order where ``row`` runs along Y (left) and
``col`` runs along X (forward); the stored array shape is ``(height, width)``.
Bounds are half-open, matching the fisheye image-bounds convention: points on
the maximum edge belong to the next cell outside the grid and are invalid.

The grid stores no belief state itself. It only maps between metric positions
and discrete cells so occupancy, semantics, or uncertainty layers built on top
share one documented discretization. Unknown regions stay zero counts rather
than being presented as observed free space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class GridIndexResult:
    """Integer ``(..., 2)`` row/col indices and validity ``(...)``.

    Every invalid index is ``-1``. Valid indices always satisfy
    ``0 <= row < height`` and ``0 <= col < width``.
    """

    indices: NDArray[np.int64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BevGrid:
    """A uniform local occupancy discretization with an explicit extent."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_min", "y_max", "resolution"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite numeric value")
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite numeric value") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, number)
        if not self.x_max > self.x_min:
            raise ValueError("x_max must be strictly greater than x_min")
        if not self.y_max > self.y_min:
            raise ValueError("y_max must be strictly greater than y_min")
        if not self.resolution > 0.0:
            raise ValueError("resolution must be strictly positive")
        width = math.ceil((self.x_max - self.x_min) / self.resolution - 1e-9)
        height = math.ceil((self.y_max - self.y_min) / self.resolution - 1e-9)
        if width <= 0 or height <= 0 or width > 10_000 or height > 10_000:
            raise ValueError("BEV grid extent and resolution must yield 1..10000 cells per axis")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    width: int = 0
    height: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` matching the stored ``(row, col)`` order."""
        return (self.height, self.width)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, x_max, y_min, y_max)`` in metres."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def _positions(self, values: ArrayLike) -> NDArray[np.float64]:
        try:
            array = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("positions must be a real numeric array") from exc
        if array.dtype.kind not in "fiu":
            raise ValueError("positions must be a real numeric array")
        with np.errstate(over="ignore", invalid="ignore"):
            positions = np.asarray(array, dtype=np.float64)
        if positions.ndim == 0 or positions.shape[-1] != 2:
            raise ValueError(f"positions must have shape (..., 2), got {positions.shape}")
        return positions

    def world_to_grid(self, positions_xy: ArrayLike) -> GridIndexResult:
        """Map vehicle-frame ``(..., 2)`` XY positions to row/col cells.

        Non-finite positions and positions outside the half-open extent are
        invalid. No clipping is performed: out-of-grid points stay unknown.
        """
        positions = self._positions(positions_xy)
        flat = positions.reshape(-1, 2)
        with np.errstate(all="ignore"):
            finite = np.isfinite(flat).all(axis=1)
            col = np.floor((flat[:, 0] - self.x_min) / self.resolution).astype(np.int64)
            row = np.floor((flat[:, 1] - self.y_min) / self.resolution).astype(np.int64)
            valid = finite & (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
            indices = np.stack((row, col), axis=1)
            indices[~valid] = -1
        return GridIndexResult(
            indices=indices.reshape((*positions.shape[:-1], 2)),
            valid=valid.reshape(positions.shape[:-1]),
        )

    def grid_to_world(self, indices: ArrayLike) -> NDArray[np.float64]:
        """Return cell-center ``(..., 2)`` XY positions for integer row/col indices.

        Indices must be finite integers within the grid; out-of-range indices
        raise rather than being clipped, so programming errors surface instead
        of silently shifting occupancy.
        """
        try:
            array = np.asarray(indices)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("indices must be an integer array") from exc
        if array.dtype.kind not in "iu":
            raise ValueError("indices must contain integer row/col values")
        with np.errstate(over="ignore", invalid="ignore"):
            integer = np.asarray(array, dtype=np.int64)
        if integer.ndim == 0 or integer.shape[-1] != 2:
            raise ValueError(f"indices must have shape (..., 2), got {integer.shape}")
        flat = integer.reshape(-1, 2)
        if (
            np.any(flat[:, 0] < 0)
            or np.any(flat[:, 0] >= self.height)
            or np.any(flat[:, 1] < 0)
            or np.any(flat[:, 1] >= self.width)
        ):
            raise ValueError("grid indices are outside the BEV extent")
        with np.errstate(all="ignore"):
            centers = np.column_stack(
                (
                    (flat[:, 1].astype(np.float64) + 0.5) * self.resolution + self.x_min,
                    (flat[:, 0].astype(np.float64) + 0.5) * self.resolution + self.y_min,
                )
            )
        return np.ascontiguousarray(centers.reshape((*integer.shape[:-1], 2)), dtype=np.float64)

    def empty_counts(self) -> NDArray[np.int64]:
        """Return a zero ``(height, width)`` accumulator for rasterization."""
        return np.zeros((self.height, self.width), dtype=np.int64)

    def rasterize(
        self, positions_xy: ArrayLike, *, valid: ArrayLike | None = None
    ) -> NDArray[np.int64]:
        """Count valid XY positions per cell, ignoring unknown positions.

        An optional boolean ``valid`` mask with the broadcast position leading
        shape marks caller-known bad inputs (for example failed ground
        intersections) as unknown without raising. Shape mismatches raise.
        """
        positions = self._positions(positions_xy)
        leading = positions.shape[:-1]
        if valid is None:
            caller_valid = np.ones(leading, dtype=np.bool_)
        else:
            try:
                caller_valid = np.asarray(valid, dtype=np.bool_)
            except (TypeError, ValueError) as exc:
                raise ValueError("valid must be a boolean array") from exc
            if caller_valid.shape != leading:
                raise ValueError(
                    f"valid mask shape {caller_valid.shape} must match positions {leading}"
                )
        indexed = self.world_to_grid(positions)
        keep = indexed.valid & caller_valid.reshape(indexed.valid.shape)
        counts = self.empty_counts()
        if np.any(keep):
            rows = indexed.indices[keep][:, 0]
            cols = indexed.indices[keep][:, 1]
            np.add.at(counts, (rows, cols), 1)
        return counts


__all__ = ["BevGrid", "GridIndexResult"]
