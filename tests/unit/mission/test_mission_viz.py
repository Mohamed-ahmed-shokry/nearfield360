"""Unit tests for AVP visual mission dashboard and timeline renderer."""

from pathlib import Path

import numpy as np

from nearfield360.control.models import (
    ControlCommand,
    ControlPerformanceKPIs,
    ExecutionStatus,
    ManeuverExecutionStep,
    TrackingErrorState,
    VehicleSimState,
)
from nearfield360.geometry.bev import BevGrid
from nearfield360.mission.models import (
    MissionEvent,
    MissionState,
    MissionSummaryReport,
    MissionTrigger,
)
from nearfield360.mission.viz import (
    render_mission_dashboard_overlay,
    render_mission_timeline_chart,
)
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


def _make_sample_report_and_steps() -> tuple[MissionSummaryReport, list[ManeuverExecutionStep]]:
    kpis = ControlPerformanceKPIs(
        max_cross_track_error_m=0.03,
        mean_cross_track_error_m=0.01,
        rmse_cross_track_error_m=0.015,
        max_heading_error_rad=0.02,
        mean_heading_error_rad=0.01,
        max_lateral_accel_m_s2=0.1,
        max_jerk_m_s3=0.2,
        docking_error_x_m=0.04,
        docking_error_y_m=0.02,
        docking_error_heading_rad=0.01,
        docking_distance_m=0.045,
        is_docked_successfully=True,
    )
    events = [
        MissionEvent(
            t=0.0,
            source_state=MissionState.STANDBY,
            target_state=MissionState.SEARCHING,
            trigger=MissionTrigger.ACTIVATE,
            description="Mission started",
        ),
        MissionEvent(
            t=0.5,
            source_state=MissionState.SEARCHING,
            target_state=MissionState.SLOT_SELECTED,
            trigger=MissionTrigger.SLOT_DISCOVERED,
            description="Slot selected",
        ),
        MissionEvent(
            t=1.0,
            source_state=MissionState.SLOT_SELECTED,
            target_state=MissionState.PARKING_MANEUVER,
            trigger=MissionTrigger.MANEUVER_START,
            description="Tracking started",
        ),
        MissionEvent(
            t=4.0,
            source_state=MissionState.PARKING_MANEUVER,
            target_state=MissionState.FINAL_ALIGNMENT,
            trigger=MissionTrigger.TARGET_REACHED,
            description="Aligned in slot",
        ),
        MissionEvent(
            t=4.5,
            source_state=MissionState.FINAL_ALIGNMENT,
            target_state=MissionState.COMPLETED,
            trigger=MissionTrigger.ALIGNMENT_COMPLETE,
            description="Parked successfully",
        ),
    ]
    report = MissionSummaryReport(
        mission_id="mission_viz_test",
        final_state=MissionState.COMPLETED,
        target_slot_id="slot_01",
        total_duration_s=4.5,
        total_steps=90,
        replan_count=0,
        events=events,
        final_kpis=kpis,
        is_success=True,
        message="Mission succeeded",
    )
    steps = [
        ManeuverExecutionStep(
            t=round(i * 0.05, 3),
            step_index=i,
            segment_index=0,
            phase=ManeuverPhase.DOCK,
            vehicle_state=VehicleSimState(
                x=round(4.0, 3),
                y=round(0.5 + i * 0.03, 3),
                heading_rad=1.5708,
                velocity=0.4,
                steer_angle_rad=0.0,
                gear=ManeuverGear.FORWARD,
            ),
            command=ControlCommand(steering_angle_rad=0.0, target_velocity=0.4),
            error=TrackingErrorState(cross_track_error_m=0.01, heading_error_rad=0.005),
            nearest_obstacle_distance_m=3.0,
            status=ExecutionStatus.ACTIVE,
        )
        for i in range(90)
    ]
    return report, steps


def test_render_mission_dashboard_overlay_creates_valid_image(tmp_path: Path) -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.05)
    report, steps = _make_sample_report_and_steps()

    corners = (
        ParkingSlotCorner(x=2.8, y=1.0, z=0.0),
        ParkingSlotCorner(x=2.8, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=6.0, z=0.0),
        ParkingSlotCorner(x=5.2, y=1.0, z=0.0),
    )
    slot = ParkingSlot(
        slot_id="slot_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        corners=corners,
        center=(4.0, 3.5),
        heading_rad=1.5708,
        width_m=2.4,
        length_m=5.0,
        status=SlotOccupancyStatus.VACANT,
        confidence=0.9,
    )
    plan = ParkingTrajectoryPlan(
        plan_id="plan_01",
        slot_id="slot_01",
        slot_type=ParkingSlotType.PERPENDICULAR,
        status=PlanStatus.SUCCESS,
        start_pose=(4.0, 0.5, 1.5708),
        target_pose=(4.0, 3.5, 1.5708),
        total_length_m=3.0,
        total_duration_s=4.5,
        gear_switches=0,
        max_curvature=0.0,
        min_clearance_m=2.0,
        is_executable=True,
        segments=[
            ManeuverSegment(
                segment_index=0,
                phase=ManeuverPhase.DOCK,
                gear=ManeuverGear.FORWARD,
                length_m=3.0,
                duration_s=4.5,
                waypoints=[
                    TrajectoryWaypoint(
                        x=4.0,
                        y=0.5 + i * 0.1,
                        heading_rad=1.5708,
                        curvature=0.0,
                        velocity=0.4,
                        acceleration=0.0,
                        gear=ManeuverGear.FORWARD,
                        t=i * 0.15,
                        distance_m=i * 0.1,
                    )
                    for i in range(31)
                ],
            )
        ],
    )

    out_file = tmp_path / "dashboard.png"
    canvas = render_mission_dashboard_overlay(
        grid,
        report,
        steps,
        slots=[slot],
        plan=plan,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 500
    assert canvas.shape == (*grid.shape, 3)


def test_render_mission_timeline_chart_creates_valid_image(tmp_path: Path) -> None:
    report, steps = _make_sample_report_and_steps()
    out_file = tmp_path / "timeline.png"

    canvas = render_mission_timeline_chart(
        report,
        steps,
        width=800,
        height=500,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 500
    assert canvas.shape == (500, 800, 3)


def test_render_mission_timeline_chart_empty_fallback(tmp_path: Path) -> None:
    report = MissionSummaryReport(
        mission_id="empty_mission",
        final_state=MissionState.ABORTED,
        total_duration_s=0.0,
        total_steps=0,
        replan_count=0,
        events=[],
        final_kpis=None,
        is_success=False,
        message="Empty",
    )
    out_file = tmp_path / "empty_timeline.png"
    canvas = render_mission_timeline_chart(report, [], output_path=out_file)

    assert out_file.is_file()
    assert isinstance(canvas, np.ndarray)
