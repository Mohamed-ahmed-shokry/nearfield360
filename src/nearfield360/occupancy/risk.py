"""Geometry-derived risk zones and occupancy statistics within them.

A zone is a boolean cell mask over one BEV grid (a driving corridor, a
circle around the vehicle, or any caller-supplied pattern). ``risk_report``
counts how confidently-observed, dangerously occupied cells fall inside each
zone so the fused occupancy layer can be reasoned about spatially.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.evidence import OccupancyEvidence


def _finite_number(value: object, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not np.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{name} must be a finite numeric value")
    return number


@dataclass(frozen=True, slots=True)
class RiskZone:
    """A named cell mask over a BEV grid."""

    name: str
    mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("zone name must be a non-empty string")
        if not isinstance(self.mask, np.ndarray) or self.mask.dtype.kind != "b":
            raise ValueError("zone mask must be a boolean array")
        if self.mask.ndim != 2 or self.mask.size == 0:
            raise ValueError(f"zone mask must be a non-empty 2D array, got {self.mask.shape}")
        object.__setattr__(self, "mask", np.ascontiguousarray(self.mask, dtype=np.bool_))


def validate_zone_mask(mask: ArrayLike, grid: BevGrid) -> NDArray[np.bool_]:
    """Coerce a boolean cell mask whose shape matches the grid exactly."""
    try:
        mask_array = np.asarray(mask)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("zone mask must be a boolean array") from exc
    if mask_array.dtype.kind not in "biu":
        raise ValueError("zone mask must be a boolean array")
    boolean = np.ascontiguousarray(mask_array, dtype=np.bool_)
    if boolean.shape != grid.shape:
        raise ValueError(f"zone mask shape {boolean.shape} must match grid {grid.shape}")
    return boolean


def _centers(grid: BevGrid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-cell-array ``(x, y)`` vehicle-frame center coordinate grids."""
    rows, cols = np.meshgrid(
        np.arange(grid.height, dtype=np.int64),
        np.arange(grid.width, dtype=np.int64),
        indexing="ij",
    )
    centers = grid.grid_to_world(np.stack((rows.ravel(), cols.ravel()), axis=1))
    return centers[:, 0].reshape(grid.shape), centers[:, 1].reshape(grid.shape)


def corridor_zone(
    grid: BevGrid,
    *,
    front_length: float,
    half_width: float,
    start: float = 0.0,
) -> RiskZone:
    """Build a forward driving corridor ``X`` in ``[start, start + front_length)``.

    The corridor includes the wedge ``|Y| <= half_width`` so it stays a
    collision-sensitive band centered on the vehicle's forward path.
    """
    front_length = _finite_number(front_length, "front_length", positive=True)
    half_width = _finite_number(half_width, "half_width", positive=False)
    start = _finite_number(start, "start", positive=False)
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative")
    x_centers, y_centers = _centers(grid)
    inside = (
        (x_centers >= start)
        & (x_centers < start + front_length)
        & (np.abs(y_centers) <= half_width)
    )
    return RiskZone(name="forward_corridor", mask=inside)


def rear_corridor_zone(
    grid: BevGrid,
    *,
    rear_length: float,
    half_width: float,
    start: float = 0.0,
) -> RiskZone:
    """Build a rear driving corridor ``X`` in ``(-(start + rear_length), -start]``.

    The corridor covers the backward collision path behind the vehicle
    within the lateral span ``|Y| <= half_width``.
    """
    rear_length = _finite_number(rear_length, "rear_length", positive=True)
    half_width = _finite_number(half_width, "half_width", positive=False)
    start = _finite_number(start, "start", positive=False)
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative")
    if start < 0.0:
        raise ValueError("start must be non-negative")
    x_centers, y_centers = _centers(grid)
    inside = (
        (x_centers > -(start + rear_length))
        & (x_centers <= -start)
        & (np.abs(y_centers) <= half_width)
    )
    return RiskZone(name="rear_corridor", mask=inside)


def lateral_clearance_zone(
    grid: BevGrid,
    side: Literal["left", "right"],
    *,
    width: float,
    x_min: float,
    x_max: float,
    start_y: float = 0.0,
) -> RiskZone:
    """Build a side clearance zone along the vehicle flank.

    ``side="left"`` monitors ``Y`` in ``[start_y, start_y + width)``;
    ``side="right"`` monitors ``Y`` in ``(-(start_y + width), -start_y]``.
    ``X`` is bounded in ``[x_min, x_max)``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    width = _finite_number(width, "width", positive=True)
    start_y = _finite_number(start_y, "start_y", positive=False)
    if start_y < 0.0:
        raise ValueError("start_y must be non-negative")
    x_min_val = _finite_number(x_min, "x_min", positive=False)
    x_max_val = _finite_number(x_max, "x_max", positive=False)
    if x_min_val >= x_max_val:
        raise ValueError("x_min must be strictly less than x_max")

    x_centers, y_centers = _centers(grid)
    x_inside = (x_centers >= x_min_val) & (x_centers < x_max_val)
    if side == "left":
        y_inside = (y_centers >= start_y) & (y_centers < start_y + width)
    else:
        y_inside = (y_centers > -(start_y + width)) & (y_centers <= -start_y)
    return RiskZone(name=f"{side}_clearance", mask=x_inside & y_inside)


def surround_parking_zones(
    grid: BevGrid,
    *,
    front_length: float = 3.0,
    rear_length: float = 3.0,
    half_width: float = 0.9,
    start_x: float = 0.0,
    rear_start_x: float = 0.0,
    lateral_width: float = 0.8,
    vehicle_x_min: float = -2.0,
    vehicle_x_max: float = 2.0,
    near_radius: float = 0.5,
    warning_radius: float = 1.5,
) -> tuple[RiskZone, ...]:
    """Construct a full 360-degree suite of explainable parking risk zones."""
    return (
        corridor_zone(grid, front_length=front_length, half_width=half_width, start=start_x),
        rear_corridor_zone(
            grid, rear_length=rear_length, half_width=half_width, start=rear_start_x
        ),
        lateral_clearance_zone(
            grid,
            "left",
            width=lateral_width,
            x_min=vehicle_x_min,
            x_max=vehicle_x_max,
            start_y=half_width,
        ),
        lateral_clearance_zone(
            grid,
            "right",
            width=lateral_width,
            x_min=vehicle_x_min,
            x_max=vehicle_x_max,
            start_y=half_width,
        ),
        circular_zone(grid, center_xy=(0.0, 0.0), radius=near_radius, name="near_circle"),
        circular_zone(grid, center_xy=(0.0, 0.0), radius=warning_radius, name="warning_circle"),
    )


def circular_zone(
    grid: BevGrid,
    center_xy: Sequence[float] | tuple[float, float],
    radius: float,
    *,
    name: str = "circular_zone",
) -> RiskZone:
    """Build a disk ``(X - cx)^2 + (Y - cy)^2 <= radius^2`` in metres."""
    if not isinstance(center_xy, Sequence) or len(center_xy) != 2:
        raise ValueError("center_xy must be an (x, y) pair of finite numbers")
    center = tuple(_finite_number(value, "center_xy", positive=False) for value in center_xy)
    radius = _finite_number(radius, "radius", positive=True)
    x_centers, y_centers = _centers(grid)
    inside = (x_centers - center[0]) ** 2 + (y_centers - center[1]) ** 2 <= radius**2
    return RiskZone(name=name, mask=inside)


@dataclass(frozen=True, slots=True)
class ZoneRisk:
    """Occupancy statistics restricted to one zone's cells."""

    name: str
    cells: int
    observed_cells: int
    occupied_cells: int
    area_m2: float
    observed_area_m2: float
    occupied_area_m2: float
    mean_occupancy: float | None
    max_occupancy: float | None
    mean_uncertainty: float | None
    max_uncertainty: float | None


def risk_report(
    zones: Sequence[RiskZone],
    evidence: OccupancyEvidence,
    *,
    min_evidence: int = 1,
    danger_occupancy: float = 0.5,
) -> tuple[ZoneRisk, ...]:
    """Summarize observed and dangerously occupied cells inside each zone.

    A cell is *confidently observed* when it received at least
    ``min_evidence`` observations and *dangerously occupied* when its
    occupancy share strictly exceeds ``danger_occupancy``. Occupancy shares
    (and their mean/max) are only reported over confidently observed cells;
    zones without any are left as ``None``.
    """
    if isinstance(min_evidence, bool) or not isinstance(min_evidence, int) or min_evidence < 1:
        raise ValueError("min_evidence must be a strictly positive integer")
    if (
        isinstance(danger_occupancy, bool)
        or not isinstance(danger_occupancy, (int, float))
        or not np.isfinite(danger_occupancy)
        or not 0.0 <= float(danger_occupancy) <= 1.0
    ):
        raise ValueError("danger_occupancy must be within [0, 1]")
    if not isinstance(evidence, OccupancyEvidence):
        raise ValueError("evidence must be an OccupancyEvidence layer")
    occupancy = evidence.occupancy()
    uncertainty = evidence.uncertainty()
    step = evidence.grid.resolution
    cell_area = step * step
    confident = evidence.observed >= min_evidence

    reports: list[ZoneRisk] = []
    for zone in zones:
        mask = validate_zone_mask(zone.mask, evidence.grid)
        cells = int(np.count_nonzero(mask))
        observed_mask = mask & confident
        observed_cells = int(np.count_nonzero(observed_mask))
        occupied_mask = observed_mask & (occupancy > danger_occupancy)
        occupied_cells = int(np.count_nonzero(occupied_mask))
        values = occupancy[observed_mask]
        mean = float(np.mean(values)) if values.size > 0 else None
        maximum = float(np.max(values)) if values.size > 0 else None
        unc_values = uncertainty[observed_mask]
        mean_unc = float(np.mean(unc_values)) if unc_values.size > 0 else None
        max_unc = float(np.max(unc_values)) if unc_values.size > 0 else None
        reports.append(
            ZoneRisk(
                name=zone.name,
                cells=cells,
                observed_cells=observed_cells,
                occupied_cells=occupied_cells,
                area_m2=cells * cell_area,
                observed_area_m2=observed_cells * cell_area,
                occupied_area_m2=occupied_cells * cell_area,
                mean_occupancy=mean,
                max_occupancy=maximum,
                mean_uncertainty=mean_unc,
                max_uncertainty=max_unc,
            )
        )
    return tuple(reports)


__all__ = [
    "RiskZone",
    "ZoneRisk",
    "circular_zone",
    "corridor_zone",
    "lateral_clearance_zone",
    "rear_corridor_zone",
    "risk_report",
    "surround_parking_zones",
    "validate_zone_mask",
]
