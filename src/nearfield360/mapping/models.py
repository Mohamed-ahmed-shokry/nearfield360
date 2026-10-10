"""Domain models for parking facility HD vector mapping, routing, and localization."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LaneDirection(StrEnum):
    """Directionality rule for vehicle movement along a facility driving lane."""

    ONE_WAY = "one_way"
    TWO_WAY = "two_way"


class WaypointType(StrEnum):
    """Topological function of a navigation waypoint in the parking facility."""

    ENTRY = "entry"
    EXIT = "exit"
    INTERSECTION = "intersection"
    LANE_NODE = "lane_node"
    SLOT_ACCESS = "slot_access"
    DROP_OFF = "drop_off"


class SlotBayType(StrEnum):
    """Geometric orientation layout of a pre-mapped parking bay."""

    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    SLANTED = "slanted"


class SlotReservationStatus(StrEnum):
    """Operational reservation or occupancy status for a parking slot."""

    VACANT = "vacant"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    BLOCKED = "blocked"


class FacilityBounds(BaseModel):
    """Metric spatial bounding box of the parking facility in meters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float = Field(..., allow_inf_nan=False)
    x_max: float = Field(..., allow_inf_nan=False)
    y_min: float = Field(..., allow_inf_nan=False)
    y_max: float = Field(..., allow_inf_nan=False)

    @field_validator("x_max")
    @classmethod
    def validate_x(cls, v: float, info: Any) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and v <= x_min:
            raise ValueError("x_max must be strictly greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y(cls, v: float, info: Any) -> float:
        y_min = info.data.get("y_min")
        if y_min is not None and v <= y_min:
            raise ValueError("y_max must be strictly greater than y_min")
        return v

    def contains(self, x: float, y: float) -> bool:
        """Check whether a 2D point lies inside the facility bounds."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class FacilityWaypoint(BaseModel):
    """Topological navigation node in the facility vector map."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    waypoint_id: str
    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float | None = Field(default=None, allow_inf_nan=False)
    waypoint_type: WaypointType = WaypointType.LANE_NODE
    connected_waypoints: list[str] = Field(default_factory=list)

    @property
    def xy(self) -> tuple[float, float]:
        """Return (x, y) coordinates."""
        return (self.x, self.y)


class FacilityLane(BaseModel):
    """Driving corridor centerline polyline and operational rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_id: str
    name: str = ""
    centerline_points: list[tuple[float, float]]
    width_m: float = Field(default=3.5, gt=0.0, allow_inf_nan=False)
    speed_limit_mps: float = Field(default=2.5, gt=0.0, allow_inf_nan=False)
    direction: LaneDirection = LaneDirection.ONE_WAY
    start_waypoint_id: str
    end_waypoint_id: str

    @field_validator("centerline_points")
    @classmethod
    def validate_points(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) < 2:
            raise ValueError("Lane centerline must contain at least 2 points")
        for p in pts:
            if len(p) != 2 or not (math.isfinite(p[0]) and math.isfinite(p[1])):
                raise ValueError("Centerline points must be finite 2-tuples of (x, y)")
        return pts

    @property
    def length_m(self) -> float:
        """Total polyline length along the lane centerline."""
        total = 0.0
        for i in range(len(self.centerline_points) - 1):
            p1 = self.centerline_points[i]
            p2 = self.centerline_points[i + 1]
            total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return total


class FacilitySlot(BaseModel):
    """Metric pre-mapped parking bay with 4-corner polygon and access point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str
    slot_type: SlotBayType
    corners: list[tuple[float, float]]
    center: tuple[float, float]
    heading_rad: float = Field(..., allow_inf_nan=False)
    width_m: float = Field(..., gt=0.0, allow_inf_nan=False)
    length_m: float = Field(..., gt=0.0, allow_inf_nan=False)
    access_lane_id: str
    access_waypoint_id: str
    status: SlotReservationStatus = SlotReservationStatus.VACANT

    @field_validator("corners")
    @classmethod
    def validate_corners(cls, corners: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(corners) != 4:
            raise ValueError("Slot must have exactly 4 corner points")
        for c in corners:
            if len(c) != 2 or not (math.isfinite(c[0]) and math.isfinite(c[1])):
                raise ValueError("Corners must be finite 2-tuples of (x, y)")
        return corners

    @field_validator("center")
    @classmethod
    def validate_center(cls, c: tuple[float, float]) -> tuple[float, float]:
        if len(c) != 2 or not (math.isfinite(c[0]) and math.isfinite(c[1])):
            raise ValueError("Center must be a finite 2-tuple of (x, y)")
        return c


class FacilityObstacle(BaseModel):
    """Structural boundary, pillar, curb, or perimeter wall in the facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obstacle_id: str
    obstacle_type: str = "wall"
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) < 2:
            raise ValueError("Obstacle polygon must contain at least 2 points")
        for p in pts:
            if len(p) != 2 or not (math.isfinite(p[0]) and math.isfinite(p[1])):
                raise ValueError("Polygon points must be finite 2-tuples of (x, y)")
        return pts


class RouteWaypoint(BaseModel):
    """Kinematic waypoint along an optimized global navigation route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float = Field(..., allow_inf_nan=False)
    target_speed_mps: float = Field(..., ge=0.0, allow_inf_nan=False)
    curvature: float = Field(default=0.0, allow_inf_nan=False)
    lane_id: str
    corridor_half_width_m: float = Field(default=1.75, gt=0.0, allow_inf_nan=False)
    s_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @property
    def xy(self) -> tuple[float, float]:
        """Return (x, y) 2D coordinate."""
        return (self.x, self.y)


class GlobalRoute(BaseModel):
    """Complete planned topological and metric route through the facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str
    start_pose: tuple[float, float, float]
    target_slot_id: str | None = None
    target_waypoint_id: str | None = None
    total_length_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    estimated_duration_s: float = Field(..., ge=0.0, allow_inf_nan=False)
    waypoints: list[RouteWaypoint] = Field(default_factory=list)
    lane_sequence: list[str] = Field(default_factory=list)


class PoseEstimate(BaseModel):
    """Estimated vehicle pose $(x, y, \\theta)$ with 3x3 covariance matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    heading_rad: float = Field(..., allow_inf_nan=False)
    covariance: list[list[float]] = Field(
        default_factory=lambda: [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.005]]
    )

    @field_validator("covariance")
    @classmethod
    def validate_covariance(cls, cov: list[list[float]]) -> list[list[float]]:
        if len(cov) != 3 or any(len(row) != 3 for row in cov):
            raise ValueError("Covariance matrix must be 3x3")
        for r in range(3):
            for c in range(3):
                if not math.isfinite(cov[r][c]):
                    raise ValueError("Covariance entries must be finite numbers")
            if cov[r][r] < 0.0:
                raise ValueError("Diagonal variance terms must be non-negative")
        return cov

    @property
    def position_uncertainty_m(self) -> float:
        """2D 1-sigma position standard deviation $\\sqrt{\\sigma_{xx} + \\sigma_{yy}}$."""
        return math.sqrt(max(0.0, self.covariance[0][0] + self.covariance[1][1]))

    @property
    def heading_uncertainty_rad(self) -> float:
        """1-sigma heading standard deviation $\\sqrt{\\sigma_{\\theta\\theta}}$."""
        return math.sqrt(max(0.0, self.covariance[2][2]))


class LandmarkObservation(BaseModel):
    """Metric landmark observation of a parking slot from perception."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str
    observed_center: tuple[float, float]
    observed_heading: float = Field(..., allow_inf_nan=False)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)


class LocalizationReport(BaseModel):
    """Quantitative summary of multi-sensor vehicle localization performance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_length_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    final_pose: PoseEstimate
    max_position_uncertainty_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    mean_position_error_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_position_error_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    mean_heading_error_rad: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_heading_error_rad: float = Field(..., ge=0.0, allow_inf_nan=False)
    total_landmark_updates: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)


class FacilityMap(BaseModel):
    """Complete High-Definition (HD) vector map representing a parking facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    map_id: str
    name: str
    facility_type: str = "indoor_garage"
    bounds: FacilityBounds
    lanes: list[FacilityLane] = Field(default_factory=list)
    slots: list[FacilitySlot] = Field(default_factory=list)
    obstacles: list[FacilityObstacle] = Field(default_factory=list)
    waypoints: list[FacilityWaypoint] = Field(default_factory=list)

    def get_slot(self, slot_id: str) -> FacilitySlot | None:
        """Find a parking slot by ID."""
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def get_waypoint(self, waypoint_id: str) -> FacilityWaypoint | None:
        """Find a waypoint by ID."""
        for w in self.waypoints:
            if w.waypoint_id == waypoint_id:
                return w
        return None

    def get_lane(self, lane_id: str) -> FacilityLane | None:
        """Find a driving lane by ID."""
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None

    def validate_integrity(self) -> list[str]:
        """Perform topological and metric integrity checks; returns list of issue strings."""
        issues: list[str] = []
        waypoint_ids = {w.waypoint_id for w in self.waypoints}
        lane_ids = {lane.lane_id for lane in self.lanes}

        for lane in self.lanes:
            if lane.start_waypoint_id not in waypoint_ids:
                issues.append(
                    f"Lane {lane.lane_id} start waypoint {lane.start_waypoint_id} not found"
                )
            if lane.end_waypoint_id not in waypoint_ids:
                issues.append(f"Lane {lane.lane_id} end waypoint {lane.end_waypoint_id} not found")
            issues.extend(
                f"Lane {lane.lane_id} point {p} exceeds facility bounds"
                for p in lane.centerline_points
                if not self.bounds.contains(p[0], p[1])
            )

        for s in self.slots:
            if s.access_lane_id not in lane_ids:
                issues.append(f"Slot {s.slot_id} access lane {s.access_lane_id} not found")
            if s.access_waypoint_id not in waypoint_ids:
                issues.append(f"Slot {s.slot_id} access waypoint {s.access_waypoint_id} not found")
            if not self.bounds.contains(s.center[0], s.center[1]):
                issues.append(f"Slot {s.slot_id} center {s.center} exceeds facility bounds")
            issues.extend(
                f"Slot {s.slot_id} corner {c} exceeds facility bounds"
                for c in s.corners
                if not self.bounds.contains(c[0], c[1])
            )

        for obs in self.obstacles:
            issues.extend(
                f"Obstacle {obs.obstacle_id} point {p} exceeds facility bounds"
                for p in obs.polygon
                if not self.bounds.contains(p[0], p[1])
            )

        return issues

    def to_json(self, indent: int = 2) -> str:
        """Serialize facility map to structured JSON."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> FacilityMap:
        """Deserialize facility map from structured JSON."""
        data = json.loads(json_str)
        return cls.model_validate(data)
