"""Autonomous parking trajectory planning, Ackermann kinematics, and multi-stage maneuvers."""

from __future__ import annotations

from nearfield360.planning.collision import (
    SweptFootprintEvaluator,
    TrajectoryCollisionResult,
    point_to_polygon_distance,
)
from nearfield360.planning.kinematics import (
    AckermannVehicle,
    KinematicWaypoint,
    normalize_angle,
    plan_two_circle_reeds_shepp,
)
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingPlanReport,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.planning.planner import (
    ParkingTrajectoryPlanner,
    profile_segment_speed,
)
from nearfield360.planning.viz import render_parking_plan_bev_overlay

__all__ = [
    "AckermannVehicle",
    "KinematicWaypoint",
    "ManeuverGear",
    "ManeuverPhase",
    "ManeuverSegment",
    "ParkingPlanReport",
    "ParkingTrajectoryPlan",
    "ParkingTrajectoryPlanner",
    "PlanStatus",
    "SweptFootprintEvaluator",
    "TrajectoryCollisionResult",
    "TrajectoryWaypoint",
    "normalize_angle",
    "plan_two_circle_reeds_shepp",
    "point_to_polygon_distance",
    "profile_segment_speed",
    "render_parking_plan_bev_overlay",
]
