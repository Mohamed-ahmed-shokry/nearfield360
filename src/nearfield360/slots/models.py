"""Data models for 3D metric parking slot detection, occupancy, and corridor feasibility."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParkingSlotType(StrEnum):
    """Geometric categorization of parking slot layout."""

    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    SLANTED = "slanted"


class SlotOccupancyStatus(StrEnum):
    """Occupancy classification state for a delineated parking slot."""

    VACANT = "vacant"
    OCCUPIED = "occupied"
    UNCERTAIN = "uncertain"


class ParkingSlotCorner(BaseModel):
    """A metric 3D ground-level corner in the vehicle coordinate frame (X fwd, Y left, Z up)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    z: float = Field(default=0.0, allow_inf_nan=False)

    @property
    def xy(self) -> tuple[float, float]:
        """Return (x, y) 2D metric position."""
        return (self.x, self.y)


class SlotApproachPath(BaseModel):
    """Kinematic entry waypoint, maneuver orientation, and clearance feasibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_point: tuple[float, float]
    target_point: tuple[float, float]
    entry_heading_rad: float = Field(..., allow_inf_nan=False)
    maneuver_length_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    clearance_margin_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    is_feasible: bool

    @field_validator("entry_point", "target_point")
    @classmethod
    def validate_point(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("point must be a 2-tuple of (x, y)")
        if not (math.isfinite(value[0]) and math.isfinite(value[1])):
            raise ValueError("point coordinates must be finite floats")
        return value


class ParkingSlot(BaseModel):
    """A delineated 3D metric parking slot with occupancy status and approach path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str
    slot_type: ParkingSlotType
    corners: tuple[ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner]
    center: tuple[float, float]
    heading_rad: float = Field(..., allow_inf_nan=False)
    width_m: float = Field(..., gt=0.0, allow_inf_nan=False)
    length_m: float = Field(..., gt=0.0, allow_inf_nan=False)
    status: SlotOccupancyStatus = SlotOccupancyStatus.UNCERTAIN
    occupancy_ratio: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    uncertainty_ratio: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    approach_path: SlotApproachPath | None = None

    @field_validator("corners")
    @classmethod
    def validate_corners(
        cls,
        value: tuple[ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner],
    ) -> tuple[ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner, ParkingSlotCorner]:
        if len(value) != 4:
            raise ValueError("ParkingSlot must have exactly 4 corners")
        return value

    @field_validator("center")
    @classmethod
    def validate_center(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2 or not math.isfinite(value[0]) or not math.isfinite(value[1]):
            raise ValueError("center must be a 2-tuple of finite (x, y) coordinates")
        return value

    @property
    def polygon_xy(self) -> list[tuple[float, float]]:
        """Return list of (x, y) coordinates representing slot polygon corners."""
        return [c.xy for c in self.corners]

    @property
    def area_m2(self) -> float:
        """Compute shoelace polygon area in square metres."""
        poly = self.polygon_xy
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]
        return abs(area) * 0.5

    def contains_point(self, x: float, y: float) -> bool:
        """Check if an (x, y) coordinate lies inside the convex slot polygon."""
        poly = self.polygon_xy
        n = len(poly)
        sign = 0
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # 2D cross product: (p2.x - p1.x)*(y - p1.y) - (p2.y - p1.y)*(x - p1.x)
            cross = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
            if abs(cross) < 1e-9:
                continue
            cur_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = cur_sign
            elif sign != cur_sign:
                return False
        return True


class SlotDetectionSummary(BaseModel):
    """Aggregate statistics for detected parking slots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_slots: int = Field(default=0, ge=0)
    vacant_slots: int = Field(default=0, ge=0)
    occupied_slots: int = Field(default=0, ge=0)
    uncertain_slots: int = Field(default=0, ge=0)
    parallel_slots: int = Field(default=0, ge=0)
    perpendicular_slots: int = Field(default=0, ge=0)
    slanted_slots: int = Field(default=0, ge=0)
    feasible_approaches: int = Field(default=0, ge=0)


class SlotDetectionReport(BaseModel):
    """Complete diagnostic and detection report for parking slot perception."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str | None = None
    camera_sources: tuple[str, ...] = ()
    slots: list[ParkingSlot] = Field(default_factory=list)
    summary: SlotDetectionSummary
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ParkingSlot",
    "ParkingSlotCorner",
    "ParkingSlotType",
    "SlotApproachPath",
    "SlotDetectionReport",
    "SlotDetectionSummary",
    "SlotOccupancyStatus",
]
