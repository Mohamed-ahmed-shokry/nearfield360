"""Pydantic domain models and geometric primitives for HD vector maps and localization."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Point2D(BaseModel):
    """2D metric coordinate point in the facility frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)

    def distance_to(self, other: Point2D) -> float:
        """Euclidean distance to another Point2D."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[float, float]:
        """Return (x, y) tuple."""
        return (self.x, self.y)


class Polyline2D(BaseModel):
    """Sequence of 2D metric vertices forming a path or boundary polygon."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: list[Point2D] = Field(..., min_length=2)
    is_closed: bool = False

    @property
    def length_m(self) -> float:
        """Total cumulative arc length in meters."""
        dist = 0.0
        for i in range(len(self.points) - 1):
            dist += self.points[i].distance_to(self.points[i + 1])
        if self.is_closed and len(self.points) >= 3:
            dist += self.points[-1].distance_to(self.points[0])
        return dist

    def distance_to_point(self, pt: Point2D) -> float:
        """Minimum orthogonal distance from pt to any segment in this polyline."""
        min_dist = float("inf")
        pts = list(self.points)
        if self.is_closed:
            pts.append(pts[0])

        px, py = pt.x, pt.y
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-12:
                d = math.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                d = math.hypot(px - proj_x, py - proj_y)
            if d < min_dist:
                min_dist = d
        return min_dist


class LaneletType(StrEnum):
    """Categorization of lanelet purpose within parking facility."""

    DRIVING_LANE = "DRIVING_LANE"
    INTERSECTION = "INTERSECTION"
    RAMP = "RAMP"
    PARKING_AISLE = "PARKING_AISLE"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"


class TurnDirection(StrEnum):
    """Topological turning orientation along lanelet."""

    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    U_TURN = "U_TURN"


class Lanelet(BaseModel):
    """Atomic atomic driving corridor bound by left and right physical polylines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lanelet_id: str = Field(..., min_length=1)
    left_boundary: Polyline2D
    right_boundary: Polyline2D
    centerline: Polyline2D
    speed_limit_mps: float = Field(default=3.0, gt=0.0, le=20.0, allow_inf_nan=False)
    lanelet_type: LaneletType = LaneletType.PARKING_AISLE
    turn_direction: TurnDirection = TurnDirection.STRAIGHT
    predecessor_ids: list[str] = Field(default_factory=list)
    successor_ids: list[str] = Field(default_factory=list)


class ParkingSlotType(StrEnum):
    """Functional class of parking bay."""

    STANDARD = "STANDARD"
    EV_CHARGING = "EV_CHARGING"
    WPT_INDUCTIVE = "WPT_INDUCTIVE"
    ACCESSIBLE = "ACCESSIBLE"
    DROP_OFF = "DROP_OFF"


class ParkingSpaceFeature(BaseModel):
    """Metric geometric parking space feature with entrance orientation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str = Field(..., min_length=1)
    boundary: Polyline2D
    center: Point2D
    heading_rad: float = Field(default=0.0, allow_inf_nan=False)
    slot_type: ParkingSlotType = ParkingSlotType.STANDARD
    width_m: float = Field(default=2.4, gt=1.0, le=6.0, allow_inf_nan=False)
    length_m: float = Field(default=5.0, gt=2.0, le=12.0, allow_inf_nan=False)


class ObstacleZone(BaseModel):
    """Static architectural boundary (pillar, barrier wall, or high curb)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_id: str = Field(..., min_length=1)
    boundary: Polyline2D
    zone_type: str = Field(default="PILLAR")


class MapTile(BaseModel):
    """Spatially indexed metric bounding tile containing subset of map features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_id: str = Field(..., min_length=1)
    x_min: float = Field(..., allow_inf_nan=False)
    y_min: float = Field(..., allow_inf_nan=False)
    x_max: float = Field(..., allow_inf_nan=False)
    y_max: float = Field(..., allow_inf_nan=False)
    lanelets: list[Lanelet] = Field(default_factory=list)
    parking_spaces: list[ParkingSpaceFeature] = Field(default_factory=list)
    obstacles: list[ObstacleZone] = Field(default_factory=list)

    @field_validator("x_max")
    @classmethod
    def _validate_x_bounds(cls, v: float, info: Any) -> float:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be strictly greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def _validate_y_bounds(cls, v: float, info: Any) -> float:
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be strictly greater than y_min")
        return v

    def contains_point(self, pt: Point2D) -> bool:
        """Check whether pt is inside this tile's half-open bounding rectangle."""
        return self.x_min <= pt.x < self.x_max and self.y_min <= pt.y < self.y_max


class HDVectorMap(BaseModel):
    """Complete multi-tile vectorized facility map representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    map_id: str = Field(..., min_length=1)
    facility_id: str = Field(default="FAC-AVP-01")
    level_id: str = Field(default="L1")
    tiles: list[MapTile] = Field(default_factory=list)
    created_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PoseHypothesis(BaseModel):
    """Individual particle state in Monte Carlo Localizer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    yaw_rad: float = Field(..., allow_inf_nan=False)
    weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)


class LocalizationConvergence(StrEnum):
    """Convergence and tracking health of the particle filter localizer."""

    CONVERGED = "CONVERGED"
    TRACKING = "TRACKING"
    CONVERGING = "CONVERGING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


class LocalizationState(BaseModel):
    """Metric localization estimate, uncertainty covariance, and tracking health."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    estimated_x: float = Field(..., allow_inf_nan=False)
    estimated_y: float = Field(..., allow_inf_nan=False)
    estimated_yaw_rad: float = Field(..., allow_inf_nan=False)
    std_x: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    std_y: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    std_yaw_rad: float = Field(default=0.02, ge=0.0, allow_inf_nan=False)
    covariance: list[list[float]] = Field(default_factory=list)
    status: LocalizationConvergence = LocalizationConvergence.CONVERGING
    associated_features_count: int = Field(default=0, ge=0)
    active_tile_id: str = Field(default="")
