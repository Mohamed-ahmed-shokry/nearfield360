"""Vehicle-frame ray to ground-plane intersection without inferring depth.

Vehicle axes are X forward, Y left, and Z up. The ground is the horizontal
plane ``Z == ground_z`` (default zero). A footprint is ``origin + t*direction``
for the caller-visible distance ``t >= 0``; pixels alone never determine that
distance, and regions without a forward intersection stay unknown rather than
being presented as observed free space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class GroundIntersection:
    """Ground footprints ``(..., 3)``, distances ``(...)``, and validity ``(...)``.

    Every invalid point is NaN and every invalid distance is NaN. Distances are
    Euclidean ray lengths in the same units as the origins (metres for
    WoodScape), not depths along a single axis.
    """

    points: NDArray[np.float64]
    distances: NDArray[np.float64]
    valid: NDArray[np.bool_]


def _vector_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {result.shape}")
    return result


def _validate_plane(ground_z: float, max_distance: float | None) -> tuple[float, float | None]:
    if isinstance(ground_z, bool) or not isinstance(ground_z, (int, float)):
        raise ValueError("ground_z must be a finite numeric height")
    try:
        z = float(ground_z)
    except (OverflowError, ValueError) as exc:
        raise ValueError("ground_z must be a finite numeric height") from exc
    if not np.isfinite(z):
        raise ValueError("ground_z must be a finite numeric height")
    if max_distance is not None:
        if isinstance(max_distance, bool) or not isinstance(max_distance, (int, float)):
            raise ValueError("max_distance must be a positive finite distance")
        try:
            limit = float(max_distance)
        except (OverflowError, ValueError) as exc:
            raise ValueError("max_distance must be a positive finite distance") from exc
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("max_distance must be a positive finite distance")
        return z, limit
    return z, None


def intersect_ground(
    origins: ArrayLike,
    directions: ArrayLike,
    *,
    ground_z: float = 0.0,
    max_distance: float | None = None,
) -> GroundIntersection:
    """Intersect vehicle-frame rays with a horizontal plane.

    Origins and directions use shape ``(..., 3)`` with broadcastable leading
    dimensions; a single ``(3,)`` origin broadcasts against batched directions.
    Non-finite components, zero directions, rays parallel to the plane, and
    intersections behind the ray origin (``t < 0``) are invalid and produce
    NaNs rather than extrapolated footprints. An optional ``max_distance``
    marks intersections beyond that Euclidean ray length invalid.
    """
    plane_z, limit = _validate_plane(ground_z, max_distance)
    origin_array = _vector_array(origins, "origins")
    direction_array = _vector_array(directions, "directions")
    try:
        broadcast_shape = np.broadcast_shapes(origin_array.shape[:-1], direction_array.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "origins and directions must have broadcastable shapes, got "
            f"{origin_array.shape} and {direction_array.shape}"
        ) from exc
    shape = (*broadcast_shape, 3)
    try:
        flat_origins = np.broadcast_to(origin_array, shape).reshape(-1, 3)
        flat_directions = np.broadcast_to(direction_array, shape).reshape(-1, 3)
    except ValueError as exc:  # pragma: no cover - broadcast_shapes already validated
        raise ValueError("origins and directions must have broadcastable shapes") from exc

    with np.errstate(all="ignore"):
        finite = np.isfinite(flat_origins).all(axis=1) & np.isfinite(flat_directions).all(axis=1)
        nonzero = np.any(flat_directions != 0.0, axis=1)
        parallel = flat_directions[:, 2] == 0.0
        numerator = plane_z - flat_origins[:, 2]
        # Division by zero yields inf/nan, which the validity mask rejects below.
        parameters = numerator / flat_directions[:, 2]
        norms = np.linalg.norm(flat_directions, axis=1)
        distances = parameters * norms
        valid = (
            finite
            & nonzero
            & ~parallel
            & np.isfinite(parameters)
            & np.isfinite(distances)
            & (parameters >= 0.0)
        )
        if limit is not None:
            valid &= distances <= limit
        safe_parameters = np.where(valid, parameters, 0.0)
        safe_origins = np.where(valid[:, None], flat_origins, 0.0)
        safe_directions = np.where(valid[:, None], flat_directions, 0.0)
        points = safe_origins + safe_parameters[:, None] * safe_directions
        valid &= np.isfinite(points).all(axis=1)

    points_out = np.where(valid[:, None], points, np.nan).reshape(shape)
    distances_out = np.where(valid, distances, np.nan).reshape(broadcast_shape)
    valid_out = valid.reshape(broadcast_shape)
    return GroundIntersection(points=points_out, distances=distances_out, valid=valid_out)


__all__ = ["GroundIntersection", "intersect_ground"]
