"""Unit tests for the ManeuverExecutor and DynamicSafetyMonitor."""

from __future__ import annotations

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.executor import DynamicSafetyMonitor, ManeuverExecutor
from nearfield360.control.models import ExecutionStatus
from nearfield360.planning.kinematics import AckermannVehicle
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import ParkingSlotType
from nearfield360.tracking.models import TrackedObstacle, TrackState


def _create_simple_straight_plan() -> ParkingTrajectoryPlan:
    # 2.0 meter straight forward path
    waypoints = [
        TrajectoryWaypoint(
            x=round(float(i) * 0.1, 3),
            y=0.0,
            heading_rad=0.0,
            velocity=0.8,
            distance_m=round(float(i) * 0.1, 3),
            t=round(float(i) * 0.125, 3),
            gear=ManeuverGear.FORWARD,
        )
        for i in range(21)
    ]
    segment = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.APPROACH,
        gear=ManeuverGear.FORWARD,
        length_m=2.0,
        duration_s=2.5,
        waypoints=waypoints,
    )
    return ParkingTrajectoryPlan(
        plan_id="test-straight-plan",
        slot_id="slot-01",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.SUCCESS,
        total_length_m=2.0,
        total_duration_s=2.5,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(2.0, 0.0, 0.0),
        is_executable=True,
        segments=[segment],
    )


def test_safety_monitor_injected_obstacle() -> None:
    monitor = DynamicSafetyMonitor()
    vehicle = AckermannVehicle()
    footprint = vehicle.compute_footprint_polygon(0.0, 0.0, 0.0)

    # Obstacle far away
    is_haz, clearance, _ = monitor.evaluate_safety(footprint, injected_obstacle=(10.0, 10.0, 0.5))
    assert not is_haz
    assert clearance > 5.0

    # Obstacle right at the vehicle bumper
    is_haz2, clearance2, reason = monitor.evaluate_safety(
        footprint, injected_obstacle=(3.65, 0.0, 0.2)
    )
    assert is_haz2
    assert clearance2 < 0.15
    assert reason is not None and "collision" in reason.lower()


def test_safety_monitor_tracked_obstacle() -> None:
    monitor = DynamicSafetyMonitor()
    vehicle = AckermannVehicle()
    footprint = vehicle.compute_footprint_polygon(0.0, 0.0, 0.0)

    # Tracked obstacle inside collision boundary
    obs = TrackedObstacle(
        track_id=1,
        class_id=1,
        class_name="person",
        state=TrackState.CONFIRMED,
        position=(3.62, 0.1),
        velocity=(0.0, 0.0),
        speed=0.0,
        age=5,
        hits=5,
        time_since_update=0,
    )
    is_haz, clearance, reason = monitor.evaluate_safety(footprint, obstacles=[obs])
    assert is_haz
    assert clearance < 0.15
    assert reason is not None and "intrusion" in reason.lower()


def test_execute_straight_forward_plan() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.05)
    executor = ManeuverExecutor(control_config=ctl_cfg)
    plan = _create_simple_straight_plan()

    report = executor.execute_plan(plan)
    assert report.status == ExecutionStatus.COMPLETED
    assert report.total_steps > 10
    assert report.kpis.is_docked_successfully
    assert report.kpis.docking_distance_m < 0.15
    assert report.kpis.max_cross_track_error_m < 0.05


def test_execute_plan_with_emergency_brake() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.05)
    executor = ManeuverExecutor(control_config=ctl_cfg)
    plan = _create_simple_straight_plan()

    # Inject obstacle at step 10 directly on path
    report = executor.execute_plan(
        plan,
        injected_obstacle=(1.0, 0.0, 0.2),
        inject_obstacle_at_step=5,
    )
    assert report.status == ExecutionStatus.EMERGENCY_STOPPED
    assert not report.kpis.is_docked_successfully
    assert "collision" in report.message.lower()


def test_execute_plan_watchdog_abort() -> None:
    # Set a tiny max_cross_track_error threshold to trigger watchdog
    ctl_cfg = ParkingControlConfig(max_cross_track_error_m=0.10)
    plan_cfg = ParkingPlannerConfig()
    executor = ManeuverExecutor(control_config=ctl_cfg, planner_config=plan_cfg)

    # Start vehicle shifted laterally by 0.5m
    plan = _create_simple_straight_plan()
    # Plan has start_pose=(0.0, 0.5, 0.0) which exceeds 0.10m tolerance
    plan_deviated = plan.model_copy(update={"start_pose": (0.0, 0.5, 0.0)})

    report = executor.execute_plan(plan_deviated)
    assert report.status == ExecutionStatus.ABORTED_DEVIATION
    assert "exceeded tolerance" in report.message.lower()


def test_execute_empty_or_non_executable_plan() -> None:
    executor = ManeuverExecutor()
    empty_plan = ParkingTrajectoryPlan(
        plan_id="empty-plan",
        slot_id="slot-0",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.UNREACHABLE,
        total_length_m=0.0,
        total_duration_s=0.0,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=0.0,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(0.0, 0.0, 0.0),
        is_executable=False,
        segments=[],
    )
    report = executor.execute_plan(empty_plan)
    assert report.status == ExecutionStatus.ABORTED_DEVIATION
    assert report.total_steps == 0
