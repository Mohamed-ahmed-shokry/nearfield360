"""Unit tests for Autonomous Valet Parking (AVP) MissionExecutive."""

from nearfield360.config import MissionConfig, ParkingControlConfig
from nearfield360.mission.executive import MissionExecutive
from nearfield360.mission.models import MissionState, MissionTrigger
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)


def _create_sample_perpendicular_slot() -> ParkingSlot:
    corners = (
        ParkingSlotCorner(x=2.8, y=1.0, z=0.0),
        ParkingSlotCorner(x=2.8, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=1.0, z=0.0),
    )
    return ParkingSlot(
        slot_id="slot_perp_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=(4.0, 3.5),
        heading_rad=1.5708,
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.95,
    )


def _create_simple_straight_plan(
    slot_id: str = "slot_perp_01",
    start_pose: tuple[float, float, float] = (4.0, 0.5, 1.5708),
    target_pose: tuple[float, float, float] = (4.0, 3.5, 1.5708),
) -> ParkingTrajectoryPlan:
    """Create a short, simple forward docking trajectory for clean testing."""
    waypoints = [
        TrajectoryWaypoint(
            x=round(start_pose[0], 4),
            y=round(start_pose[1] + i * 0.15, 4),
            heading_rad=1.5708,
            curvature=0.0,
            velocity=0.4,
            acceleration=0.0,
            gear=ManeuverGear.FORWARD,
            t=round(i * 0.35, 3),
            distance_m=round(i * 0.15, 4),
        )
        for i in range(21)
    ]
    seg = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.DOCK,
        gear=ManeuverGear.FORWARD,
        length_m=3.0,
        duration_s=7.0,
        waypoints=waypoints,
    )
    return ParkingTrajectoryPlan(
        plan_id="test_plan_001",
        slot_id=slot_id,
        slot_type=ParkingSlotType.PERPENDICULAR,
        status=PlanStatus.SUCCESS,
        start_pose=start_pose,
        target_pose=target_pose,
        total_length_m=3.0,
        total_duration_s=7.0,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        is_executable=True,
        segments=[seg],
    )


def test_executive_initial_state_and_transition() -> None:
    executive = MissionExecutive()
    assert executive.current_state.value == "standby"
    assert len(executive.events) == 0

    executive.transition_to(
        MissionState.SEARCHING,
        MissionTrigger.ACTIVATE,
        t=0.5,
        description="Manual activation test",
    )
    assert executive.current_state.value == "searching"
    assert len(executive.events) == 1
    assert executive.events[0].trigger.value == "activate"


def test_executive_nominal_mission_with_precomputed_plan() -> None:
    mission_config = MissionConfig()
    control_config = ParkingControlConfig(dt=0.05)
    executive = MissionExecutive(mission_config=mission_config, control_config=control_config)

    slot = _create_sample_perpendicular_slot()
    plan = _create_simple_straight_plan(slot_id=slot.slot_id)

    report, steps = executive.execute_mission(
        slots=[slot],
        initial_plan=plan,
        target_slot=slot,
        mission_id="mission_nominal",
    )

    assert report.is_success is True
    assert report.final_state == MissionState.COMPLETED
    assert report.replan_count == 0
    assert len(steps) > 10

    # Ensure state progression events exist
    states_visited = [e.target_state for e in report.events]
    assert MissionState.SEARCHING in states_visited
    assert MissionState.SLOT_SELECTED in states_visited
    assert MissionState.PARKING_MANEUVER in states_visited
    assert MissionState.FINAL_ALIGNMENT in states_visited
    assert MissionState.COMPLETED in states_visited


def test_executive_transient_obstacle_yield_and_resume() -> None:
    mission_config = MissionConfig(hold_timeout_s=5.0)
    control_config = ParkingControlConfig(dt=0.05)
    executive = MissionExecutive(mission_config=mission_config, control_config=control_config)

    slot = _create_sample_perpendicular_slot()
    plan = _create_simple_straight_plan(slot_id=slot.slot_id)

    # Place an obstacle directly along the path during steps 10..22 (approx 0.6s)
    # Vehicle will enter OBSTACLE_HOLD, wait until step 23, then resume and complete!
    transient_obs = (4.0, 1.2, 0.4)
    transient_window = (10, 22)

    report, _steps = executive.execute_mission(
        slots=[slot],
        initial_plan=plan,
        target_slot=slot,
        transient_obstacle=transient_obs,
        transient_obstacle_window=transient_window,
        mission_id="mission_transient",
    )

    assert report.is_success is True
    assert report.final_state == MissionState.COMPLETED
    assert report.replan_count == 0

    # Verify that OBSTACLE_HOLD was entered and exited
    states_visited = [e.target_state for e in report.events]
    assert MissionState.OBSTACLE_HOLD in states_visited
    triggers_fired = [e.trigger for e in report.events]
    assert MissionTrigger.OBSTACLE_DETECTED in triggers_fired
    assert MissionTrigger.OBSTACLE_CLEARED in triggers_fired


def test_executive_persistent_obstacle_triggers_replan() -> None:
    mission_config = MissionConfig(hold_timeout_s=0.25, max_replans=3)
    control_config = ParkingControlConfig(dt=0.05)
    executive = MissionExecutive(mission_config=mission_config, control_config=control_config)

    slot = _create_sample_perpendicular_slot()
    plan = _create_simple_straight_plan(slot_id=slot.slot_id)

    # Obstacle appears at step 10 and stays
    persistent_obs = (4.0, 1.0, 0.25)

    report, _steps = executive.execute_mission(
        slots=[slot],
        initial_plan=plan,
        target_slot=slot,
        persistent_obstacle=persistent_obs,
        persistent_obstacle_step=10,
        mission_id="mission_persistent",
    )

    states_visited = [e.target_state for e in report.events]
    assert MissionState.OBSTACLE_HOLD in states_visited
    assert MissionState.REPLANNING in states_visited
    assert report.replan_count >= 1


def test_executive_no_vacant_slot_aborts() -> None:
    executive = MissionExecutive()
    report, _steps = executive.execute_mission(
        slots=[],
        initial_plan=None,
        target_slot=None,
    )

    assert report.is_success is False
    assert report.final_state == MissionState.ABORTED
    assert "no vacant slot" in report.message.lower()
