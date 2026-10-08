"""Unit tests for dynamic ParkingReplanner recovery maneuver generator."""

from nearfield360.config import MissionConfig, ParkingPlannerConfig
from nearfield360.mission.replanner import ParkingReplanner
from nearfield360.planning.models import ManeuverGear, ManeuverPhase
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)
from nearfield360.tracking.models import TrackedObstacle, TrackState


def _create_test_perpendicular_slot() -> ParkingSlot:
    # Slot with entrance along y = 1.0, depth up to y = 6.0, centered at x = 4.0
    corners = (
        ParkingSlotCorner(x=2.8, y=1.0, z=0.0),
        ParkingSlotCorner(x=2.8, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=1.0, z=0.0),
    )
    return ParkingSlot(
        slot_id="slot_perp_1",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=(4.0, 3.5),
        heading_rad=1.5708,  # Heading into slot (+Y)
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.95,
    )


def test_replanner_nominal_pull_out_and_dock() -> None:
    mission_config = MissionConfig(max_replans=3, replan_pull_out_dist_m=1.2)
    planner_config = ParkingPlannerConfig()
    replanner = ParkingReplanner(mission_config, planner_config)

    slot = _create_test_perpendicular_slot()
    # Stopped at (4.0, 0.5, 1.57) in reverse gear
    current_pose = (4.0, 0.5, 1.5708)

    recovery = replanner.replan_maneuver(
        current_pose=current_pose,
        current_gear=ManeuverGear.REVERSE,
        target_slot=slot,
        replan_count=1,
        trigger_reason="Transient safety halt",
    )

    assert recovery.is_successful is True
    assert recovery.plan is not None
    assert recovery.plan.is_executable is True
    assert len(recovery.plan.segments) >= 2
    # First segment should be FORWARD ALIGN, second REVERSE DOCK
    assert recovery.plan.segments[0].gear == ManeuverGear.FORWARD
    assert recovery.plan.segments[0].phase == ManeuverPhase.ALIGN
    assert recovery.plan.segments[1].gear == ManeuverGear.REVERSE


def test_replanner_exceeds_max_replans() -> None:
    mission_config = MissionConfig(max_replans=2)
    replanner = ParkingReplanner(mission_config)
    slot = _create_test_perpendicular_slot()

    recovery = replanner.replan_maneuver(
        current_pose=(4.0, 0.5, 1.5708),
        current_gear=ManeuverGear.REVERSE,
        target_slot=slot,
        replan_count=3,  # Exceeds max 2
    )

    assert recovery.is_successful is False
    assert recovery.plan is None
    assert "exceeded" in recovery.trigger_reason.lower()


def test_replanner_rejects_colliding_injected_obstacle() -> None:
    mission_config = MissionConfig(max_replans=3)
    replanner = ParkingReplanner(mission_config)
    slot = _create_test_perpendicular_slot()

    current_pose = (4.0, 0.5, 1.5708)
    # Place large obstacle right in front of the vehicle forward pull-out path
    injected_obstacle = (4.0, 1.2, 1.0)

    recovery = replanner.replan_maneuver(
        current_pose=current_pose,
        current_gear=ManeuverGear.REVERSE,
        target_slot=slot,
        replan_count=1,
        injected_obstacle=injected_obstacle,
    )

    # When obstacle blocks forward pull-out path and slot entry, replan fails safely
    assert recovery.is_successful is False
    assert recovery.plan is None


def test_replanner_handles_dynamic_obstacle_clearance() -> None:
    mission_config = MissionConfig(max_replans=3)
    replanner = ParkingReplanner(mission_config)
    slot = _create_test_perpendicular_slot()

    current_pose = (4.0, 0.5, 1.5708)
    # Distant dynamic obstacle (harmless at 15m)
    obstacle = TrackedObstacle(
        track_id=1,
        class_id=0,
        class_name="pedestrian",
        state=TrackState.CONFIRMED,
        position=(15.0, 15.0),
        velocity=(0.0, 0.0),
        speed=0.0,
        hits=5,
        age=5,
        time_since_update=0,
    )

    recovery = replanner.replan_maneuver(
        current_pose=current_pose,
        current_gear=ManeuverGear.REVERSE,
        target_slot=slot,
        replan_count=1,
        obstacles=[obstacle],
    )

    assert recovery.is_successful is True
    assert recovery.plan is not None
