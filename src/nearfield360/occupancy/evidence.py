"""Conservative camera evidence fused onto a local bird's-eye-view grid.

Camera pixels whose rays reach the ground plane contribute evidence about
the cells they land in: drivable-ground labels vote FREE, vehicles and other
objects vote OCCUPIED, and everything else stays unknown. Each pixel carries
a weight in ``[0, 1]`` (often a decreasing confidence in distance) so far,
uncertain measurements count for less. No belief is retro-inferred along the
ray between the camera and a footprint: occluded cells are never presented as
observed free space, matching the grid's zero-counts-mean-unknown convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.data.semantic import WOODSCAPE_SEMANTIC_CLASSES
from nearfield360.geometry.bev import BevGrid

NAME_BY_LABEL_ID = MappingProxyType(
    {semantic_class.label_id: semantic_class for semantic_class in WOODSCAPE_SEMANTIC_CLASSES}
)

_ID_BY_NAME = MappingProxyType(
    {semantic_class.name: semantic_class.label_id for semantic_class in WOODSCAPE_SEMANTIC_CLASSES}
)


class OccupancyPolicyError(ValueError):
    """Raised when an occupancy class policy is malformed or unknown."""


def _class_id_by_name(name: object, field_name: str) -> int:
    if not isinstance(name, str):
        raise OccupancyPolicyError(f"{field_name} classes must be names, got {name!r}")
    try:
        return _ID_BY_NAME[name]
    except KeyError as exc:
        expected = ", ".join(_ID_BY_NAME)
        raise OccupancyPolicyError(
            f"Unknown semantic class {name!r} in {field_name}; expected one of {expected}"
        ) from exc


@dataclass(frozen=True, slots=True)
class OccupancyPolicy:
    """Which label IDs are free evidence and which are occupied evidence.

    Classes must be disjoint so a single pixel never votes both ways; both
    sets may be empty only together, since a policy that treats nothing as
    known is a configuration error.
    """

    free_label_ids: frozenset[int]
    occupied_label_ids: frozenset[int]

    def __post_init__(self) -> None:
        overlap = self.free_label_ids & self.occupied_label_ids
        if overlap:
            rendered = ", ".join(str(label_id) for label_id in sorted(overlap))
            raise OccupancyPolicyError(f"label IDs cannot be free and occupied: {rendered}")
        for label_id in self.free_label_ids | self.occupied_label_ids:
            if label_id not in NAME_BY_LABEL_ID:
                raise OccupancyPolicyError(f"unknown WoodScape semantic label: {label_id}")
        if not self.free_label_ids and not self.occupied_label_ids:
            raise OccupancyPolicyError("an occupancy policy needs free or occupied classes")

    @classmethod
    def from_names(cls, free: Sequence[str], occupied: Sequence[str]) -> OccupancyPolicy:
        """Resolve semantic class names into disjoint, recognized label IDs."""
        free_ids = frozenset(_class_id_by_name(name, "free") for name in free)
        occupied_ids = frozenset(_class_id_by_name(name, "occupied") for name in occupied)
        return cls(free_label_ids=free_ids, occupied_label_ids=occupied_ids)


@dataclass(frozen=True, slots=True)
class OccupancyEvidence:
    """Weighted free/occupied evidence accumulated per BEV cell.

    ``observed`` counts the number of contributing pixels (of any class);
    ``occupied`` and ``free`` accumulate their weights. Zero-observation cells
    stay unknown and are never synthesized as free.
    """

    grid: BevGrid
    occupied: NDArray[np.float64]
    free: NDArray[np.float64]
    observed: NDArray[np.int64]

    def __post_init__(self) -> None:
        if not isinstance(self.grid, BevGrid):
            raise ValueError("OccupancyEvidence requires a BevGrid")
        shape = self.grid.shape
        for name in ("occupied", "free", "observed"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        occupied = np.ascontiguousarray(self.occupied, dtype=np.float64)
        free = np.ascontiguousarray(self.free, dtype=np.float64)
        observed = np.ascontiguousarray(self.observed, dtype=np.int64)
        if np.any(~np.isfinite(occupied)) or np.any(~np.isfinite(free)):
            raise ValueError("occupied and free evidence must be finite")
        if np.any(occupied < 0.0) or np.any(free < 0.0):
            raise ValueError("occupied and free evidence must be non-negative")
        if np.any(observed < 0):
            raise ValueError("observed counts must be non-negative")
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "free", free)
        object.__setattr__(self, "observed", observed)

    def occupancy(self) -> NDArray[np.float64]:
        """Return the occupied share of evidence per cell (NaN where unknown)."""
        total = self.occupied + self.free
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total > 0.0, self.occupied / total, np.nan)

    def uncertainty(self) -> NDArray[np.float64]:
        """Return the posterior standard deviation of occupancy per cell.

        Uses a uniform Beta(1, 1) prior over each cell's evidence, so the
        posterior is ``Beta(occupied + 1, free + 1)``. The standard deviation
        is ``sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)))``
        where ``alpha = occupied + 1`` and ``beta = free + 1``. Higher values
        indicate less certainty about the occupancy estimate.

        Returns NaN for cells with no evidence (unknown).
        """
        alpha = self.occupied + 1.0
        beta = self.free + 1.0
        total = alpha + beta
        total_sq = total * total
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = (alpha * beta) / (total_sq * (total + 1.0))
            std = np.sqrt(variance)
        has_evidence = (self.occupied + self.free) > 0.0
        return np.where(has_evidence, std, np.nan)

    def evidence_mass(self) -> NDArray[np.float64]:
        """Return the total weighted evidence per cell (zero where unknown)."""
        return self.occupied + self.free

    def add(self, other: OccupancyEvidence) -> OccupancyEvidence:
        """Fuse another frame into a new layer; the grids must match exactly."""
        if not _same_grid(self.grid, other.grid):
            raise ValueError("cannot fuse evidence computed on different BEV grids")
        return OccupancyEvidence(
            grid=self.grid,
            occupied=self.occupied + other.occupied,
            free=self.free + other.free,
            observed=self.observed + other.observed,
        )

    def scale(self, factor: float) -> OccupancyEvidence:
        """Return a new layer with weighted evidence multiplied by a non-negative scalar factor.

        Useful for health-aware confidence discounting where evidence from degraded
        or soiled cameras is attenuated before multi-camera fusion.
        """
        if not isinstance(factor, (int, float)) or isinstance(factor, bool):
            raise ValueError("factor must be a real numeric scalar")
        factor_val = float(factor)
        if not np.isfinite(factor_val) or factor_val < 0.0:
            raise ValueError("factor must be non-negative and finite")
        return OccupancyEvidence(
            grid=self.grid,
            occupied=self.occupied * factor_val,
            free=self.free * factor_val,
            observed=self.observed,
        )

    def __add__(self, other: OccupancyEvidence) -> OccupancyEvidence:
        return self.add(other)


def _same_grid(left: BevGrid, right: BevGrid) -> bool:
    return (
        left.x_min == right.x_min
        and left.x_max == right.x_max
        and left.y_min == right.y_min
        and left.y_max == right.y_max
        and left.resolution == right.resolution
    )


def _xy_points(values: ArrayLike) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError("points must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(array, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {points.shape}")
    return points


def _labels(values: ArrayLike, sizes: tuple[int, ...]) -> NDArray[np.int64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels must be an integer array") from exc
    if array.dtype.kind not in "iu":
        raise ValueError("labels must be an integer array")
    converted = np.asarray(array, dtype=np.int64)
    if converted.shape != sizes:
        raise ValueError(f"labels must have shape {sizes}, got {converted.shape}")
    return converted


def _coerced_weights(values: ArrayLike, sizes: tuple[int, ...]) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"weights must be an array with shape {sizes}") from exc
    if array.shape != sizes:
        raise ValueError(f"weights must have shape {sizes}, got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError("weights must be a real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64).reshape(-1)


def _coerced_valid(values: ArrayLike, sizes: tuple[int, ...]) -> NDArray[np.bool_]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"valid must be an array with shape {sizes}") from exc
    if array.shape != sizes:
        raise ValueError(f"valid must have shape {sizes}, got {array.shape}")
    if array.dtype.kind not in "biu":
        raise ValueError("valid must be a boolean array")
    return np.ascontiguousarray(array, dtype=np.bool_).reshape(-1)


def rasterize_occupancy(
    grid: BevGrid,
    points: ArrayLike,
    labels: ArrayLike,
    policy: OccupancyPolicy,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> OccupancyEvidence:
    """Rasterize per-pixel free/occupied evidence onto the grid cells.

    ``points`` are vehicle-frame ``(..., 2)`` footprints; ``labels`` are
    integer label IDs sharing the points' leading shape. A pixel contributes
    only when its footprint is inside the grid, the pixel is marked valid, and
    its label is a free or an occupied class. ``weights`` (an array in
    ``[0, 1]``, or None for unit weight) and ``valid`` are validated only on
    the points that actually vote, so NaN distances from invalid rays stay
    harmless as long as those points are excluded by ``valid``.
    """
    if not isinstance(policy, OccupancyPolicy):
        raise ValueError("policy must be an OccupancyPolicy")
    points_array = _xy_points(points)
    leading = points_array.shape[:-1]
    label_array = _labels(labels, leading).reshape(-1)
    weights_array = (
        np.ones(leading, dtype=np.float64).reshape(-1)
        if weights is None
        else _coerced_weights(weights, leading)
    )
    valid_array = (
        np.ones(leading, dtype=np.bool_).reshape(-1)
        if valid is None
        else _coerced_valid(valid, leading)
    )

    indexed = grid.world_to_grid(points_array)
    flat_indices = indexed.indices.reshape(-1, 2)
    flat_valid = indexed.valid.reshape(-1) & valid_array
    free_votes = np.isin(label_array, np.fromiter(policy.free_label_ids, dtype=np.int64))
    occupied_votes = np.isin(label_array, np.fromiter(policy.occupied_label_ids, dtype=np.int64))
    known = (free_votes | occupied_votes) & flat_valid

    rows = flat_indices[known, 0]
    cols = flat_indices[known, 1]
    weights_flat = weights_array.reshape(-1)
    kept_weights = weights_flat[known]
    if np.any(~np.isfinite(kept_weights)):
        raise ValueError("weights must be finite")
    if np.any((kept_weights < 0.0) | (kept_weights > 1.0)):
        raise ValueError("weights must be within [0, 1]")

    observed = np.zeros(grid.shape, dtype=np.int64)
    occupied = np.zeros(grid.shape, dtype=np.float64)
    free = np.zeros(grid.shape, dtype=np.float64)
    if np.any(known):
        np.add.at(observed, (rows, cols), 1)
        free_keep = known & free_votes
        occupied_keep = known & occupied_votes
        free_indices = flat_indices[free_keep]
        occupied_indices = flat_indices[occupied_keep]
        np.add.at(free, (free_indices[:, 0], free_indices[:, 1]), weights_flat[free_keep])
        np.add.at(
            occupied,
            (occupied_indices[:, 0], occupied_indices[:, 1]),
            weights_flat[occupied_keep],
        )
    return OccupancyEvidence(grid=grid, occupied=occupied, free=free, observed=observed)


def fuse_occupancy(layers: Iterable[OccupancyEvidence]) -> OccupancyEvidence:
    """Fuse any number of frames in order; identical grids are required."""
    iterator = iter(layers)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("fuse_occupancy requires at least one evidence layer") from exc
    result = first
    for layer in iterator:
        result = result.add(layer)
    return result


def distance_weights(distances: ArrayLike, slope: float = 0.0) -> NDArray[np.float64]:
    """Return ``1 / (1 + slope * distance)`` with NaN preserved for invalid inputs.

    A zero slope yields unit weights for every finite distance; negative or
    non-finite distances map to NaN and are ignored by rasterization.
    """
    if isinstance(slope, bool) or not isinstance(slope, (int, float)):
        raise ValueError("slope must be a finite non-negative real")
    try:
        slope_value = float(slope)
    except (OverflowError, ValueError) as exc:
        raise ValueError("slope must be a finite non-negative real") from exc
    if not np.isfinite(slope_value) or slope_value < 0.0:
        raise ValueError("slope must be a finite non-negative real")
    try:
        array = np.asarray(distances)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("distances must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError("distances must be a real numeric array")
    with np.errstate(all="ignore"):
        safe = np.where(np.isfinite(array) & (np.asarray(array) >= 0.0), array, np.nan)
        return 1.0 / (1.0 + slope_value * safe)


__all__ = [
    "NAME_BY_LABEL_ID",
    "OccupancyEvidence",
    "OccupancyPolicy",
    "OccupancyPolicyError",
    "distance_weights",
    "fuse_occupancy",
    "rasterize_occupancy",
]
