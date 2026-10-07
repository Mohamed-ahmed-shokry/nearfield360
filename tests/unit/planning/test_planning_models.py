from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingPlanReport,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import ParkingSlotType


def test_waypoint_properties_and_validation() -> None:
    wp = TrajectoryWaypoint(
        x=2.5,
        y=1.2,
        heading_rad=0.35,
        curvature=0.1,
        velocity=0.8,
        acceleration=0.2,
        gear=ManeuverGear.FORWARD,
        t=1.5,
        distance_m=2.0,
    )
    assert wp.pose == (2.5, 1.2, 0.35)
    assert wp.xy == (2.5, 1.2)
    assert wp.gear == ManeuverGear.FORWARD


def test_waypoint_rejects_nan_and_negative_time() -> None:
    with pytest.raises(ValidationError):
        TrajectoryWaypoint(x=float("nan"), y=1.0, heading_rad=0.0)

    with pytest.raises(ValidationError):
        TrajectoryWaypoint(x=1.0, y=1.0, heading_rad=0.0, t=-0.5)


def test_maneuver_segment_and_plan() -> None:
    wp1 = TrajectoryWaypoint(x=0.0, y=0.0, heading_rad=0.0, gear=ManeuverGear.REVERSE, t=0.0)
    wp2 = TrajectoryWaypoint(x=-1.0, y=0.5, heading_rad=0.2, gear=ManeuverGear.REVERSE, t=1.0)
    seg1 = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.STEER_IN,
        gear=ManeuverGear.REVERSE,
        length_m=1.2,
        duration_s=1.0,
        waypoints=[wp1, wp2],
    )
    plan = ParkingTrajectoryPlan(
        plan_id="plan_001",
        slot_id="slot_01",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.SUCCESS,
        total_length_m=1.2,
        total_duration_s=1.0,
        gear_switches=0,
        max_curvature=0.2,
        min_clearance_m=0.35,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(-1.0, 0.5, 0.2),
        is_executable=True,
        segments=[seg1],
    )
    assert len(plan.all_waypoints) == 2
    assert plan.all_waypoints[1].x == -1.0
    assert plan.is_executable is True


def test_plan_pose_validation() -> None:
    with pytest.raises(ValidationError, match="pose elements must be finite numbers"):
        ParkingTrajectoryPlan(
            plan_id="p1",
            slot_id="s1",
            slot_type=ParkingSlotType.PERPENDICULAR,
            status=PlanStatus.SUCCESS,
            total_length_m=1.0,
            total_duration_s=1.0,
            gear_switches=0,
            max_curvature=0.1,
            min_clearance_m=0.2,
            start_pose=(0.0, 0.0, float("nan")),
            target_pose=(1.0, 1.0, 0.0),
            is_executable=True,
        )


def test_plan_report_serialization() -> None:
    report = ParkingPlanReport(
        selected_slot_id="slot_01",
        candidate_slots_evaluated=3,
        status=PlanStatus.SUCCESS,
        total_length_m=5.4,
        total_duration_s=4.2,
        gear_switches=1,
        min_clearance_m=0.28,
        is_executable=True,
    )
    dumped = report.model_dump(mode="json")
    assert dumped["status"] == "success"
    assert dumped["candidate_slots_evaluated"] == 3
    assert dumped["is_executable"] is True
