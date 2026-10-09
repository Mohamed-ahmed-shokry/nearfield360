"""Parking facility HD vector mapping, topological route planning, and localization."""

from __future__ import annotations

from nearfield360.mapping.builder import (
    FacilityBuilder,
    build_benchmark_garage,
    build_surface_lot,
)
from nearfield360.mapping.localization import (
    LocalizationSimulator,
    LocalizationStepRecord,
    PoseEstimator,
)
from nearfield360.mapping.models import (
    FacilityBounds,
    FacilityLane,
    FacilityMap,
    FacilityObstacle,
    FacilitySlot,
    FacilityWaypoint,
    GlobalRoute,
    LandmarkObservation,
    LaneDirection,
    LocalizationReport,
    PoseEstimate,
    RouteWaypoint,
    SlotBayType,
    SlotReservationStatus,
    WaypointType,
)
from nearfield360.mapping.router import (
    GlobalRouter,
    RoutingError,
    RoutingGraph,
    find_nearest_waypoint,
)
from nearfield360.mapping.viz import (
    render_facility_bev,
    render_localization_dashboard,
)

__all__ = [
    "FacilityBounds",
    "FacilityBuilder",
    "FacilityLane",
    "FacilityMap",
    "FacilityObstacle",
    "FacilitySlot",
    "FacilityWaypoint",
    "GlobalRoute",
    "GlobalRouter",
    "LandmarkObservation",
    "LaneDirection",
    "LocalizationReport",
    "LocalizationSimulator",
    "LocalizationStepRecord",
    "PoseEstimate",
    "PoseEstimator",
    "RouteWaypoint",
    "RoutingError",
    "RoutingGraph",
    "SlotBayType",
    "SlotReservationStatus",
    "WaypointType",
    "build_benchmark_garage",
    "build_surface_lot",
    "find_nearest_waypoint",
    "render_facility_bev",
    "render_localization_dashboard",
]
