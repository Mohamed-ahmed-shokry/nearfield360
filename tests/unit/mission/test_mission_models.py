"""Unit tests for AVP mission domain models."""

import pytest
from pydantic import ValidationError

from nearfield360.control.models import ControlPerformanceKPIs
from nearfield360.mission.models import (
    MissionEvent,
    MissionState,
    MissionSummaryReport,
    MissionTrigger,
    RecoveryManeuver,
    TrackedParkingSlot,
)
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)


def _create_sample_slot(slot_id: str = "slot_001") -> ParkingSlot:
    corners = (
        ParkingSlotCorner(x=2.0, y=1.0, z=0.0),
        ParkingSlotCorner(x=2.0, y=3.4, z=0.0),
        ParkingSlotCorner(x=7.0, y=3.4, z=0.0),
        ParkingSlotCorner(x=7.0, y=1.0, z=0.0),
    )
    return ParkingSlot(
        slot_id=slot_id,
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=(4.5, 2.2),
        heading_rad=0.0,
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.95,
    )


def test_mission_state_and_trigger_enums() -> None:
    assert MissionState.STANDBY.value == "standby"
    assert MissionState.SEARCHING.value == "searching"
    assert MissionState.SLOT_SELECTED.value == "slot_selected"
    assert MissionState.APPROACH.value == "approach"
    assert MissionState.PARKING_MANEUVER.value == "parking_maneuver"
    assert MissionState.OBSTACLE_HOLD.value == "obstacle_hold"
    assert MissionState.REPLANNING.value == "replanning"
    assert MissionState.FINAL_ALIGNMENT.value == "final_alignment"
    assert MissionState.COMPLETED.value == "completed"
    assert MissionState.ABORTED.value == "aborted"

    assert MissionTrigger.ACTIVATE.value == "activate"
    assert MissionTrigger.OBSTACLE_DETECTED.value == "obstacle_detected"
    assert MissionTrigger.TARGET_REACHED.value == "target_reached"


def test_mission_event_model() -> None:
    event = MissionEvent(
        t=1.25,
        source_state=MissionState.SEARCHING,
        target_state=MissionState.SLOT_SELECTED,
        trigger=MissionTrigger.SLOT_DISCOVERED,
        description="Target slot slot_001 confirmed and locked",
    )
    assert event.t == 1.25
    assert event.source_state == MissionState.SEARCHING
    assert event.target_state == MissionState.SLOT_SELECTED

    with pytest.raises(ValidationError):
        event.t = 2.0  # frozen model


def test_tracked_parking_slot_model() -> None:
    slot = _create_sample_slot()
    tracked = TrackedParkingSlot(
        track_id="track_1",
        slot=slot,
        first_observed_frame=1,
        last_observed_frame=3,
        hit_count=3,
        miss_count=0,
        stability_score=0.98,
        is_confirmed=True,
    )
    assert tracked.track_id == "track_1"
    assert tracked.hit_count == 3
    assert tracked.is_confirmed is True

    # Check stability score bounds
    with pytest.raises(ValidationError):
        TrackedParkingSlot(track_id="t2", slot=slot, stability_score=1.5)


def test_recovery_maneuver_validation() -> None:
    maneuver = RecoveryManeuver(
        replan_id="replan_1",
        trigger_reason="Dynamic obstacle blocking reverse corridor",
        start_pose=(1.0, 2.0, 0.5),
        target_slot_id="slot_001",
        is_successful=True,
        clearance_m=0.85,
    )
    assert maneuver.replan_id == "replan_1"
    assert maneuver.start_pose == (1.0, 2.0, 0.5)
    assert maneuver.is_successful is True

    # Invalid pose length
    with pytest.raises(ValidationError):
        RecoveryManeuver(
            replan_id="r2",
            trigger_reason="test",
            start_pose=(1.0, 2.0),  # type: ignore[arg-type]
            target_slot_id="slot_001",
            is_successful=False,
        )


def test_mission_summary_report_round_trip() -> None:
    kpis = ControlPerformanceKPIs(
        max_cross_track_error_m=0.04,
        mean_cross_track_error_m=0.015,
        rmse_cross_track_error_m=0.02,
        max_heading_error_rad=0.03,
        mean_heading_error_rad=0.01,
        max_lateral_accel_m_s2=0.2,
        max_jerk_m_s3=0.4,
        docking_error_x_m=0.05,
        docking_error_y_m=0.02,
        docking_error_heading_rad=0.01,
        docking_distance_m=0.054,
        is_docked_successfully=True,
    )
    event = MissionEvent(
        t=0.0,
        source_state=MissionState.STANDBY,
        target_state=MissionState.SEARCHING,
        trigger=MissionTrigger.ACTIVATE,
        description="Mission started",
    )
    report = MissionSummaryReport(
        mission_id="mission_test_001",
        final_state=MissionState.COMPLETED,
        target_slot_id="slot_001",
        total_duration_s=14.5,
        total_steps=290,
        replan_count=1,
        events=[event],
        final_kpis=kpis,
        is_success=True,
        message="Vehicle successfully parked and docked",
    )

    data = report.model_dump()
    assert data["mission_id"] == "mission_test_001"
    assert data["final_state"] == "completed"
    assert data["is_success"] is True

    loaded = MissionSummaryReport.model_validate(data)
    assert loaded == report
