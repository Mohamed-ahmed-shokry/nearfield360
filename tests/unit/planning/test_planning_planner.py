from __future__ import annotations

import math

import numpy as np
import pytest

from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.kinematics import KinematicWaypoint
from nearfield360.planning.models import ManeuverGear, ManeuverPhase, PlanStatus
from nearfield360.planning.planner import ParkingTrajectoryPlanner, profile_segment_speed
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotApproachPath,
    SlotOccupancyStatus,
)


def _make_slot(
    slot_id: str,
    slot_type: ParkingSlotType,
    center: tuple[float, float],
    width_m: float = 2.4,
    length_m: float = 5.0,
    heading_rad: float = 0.0,
    status: SlotOccupancyStatus = SlotOccupancyStatus.VACANT,
) -> ParkingSlot:
    cx, cy = center
    hw = 0.5 * width_m
    hl = 0.5 * length_m
    cos_t = math.cos(heading_rad)
    sin_t = math.sin(heading_rad)

    # 4 corners around center
    local_pts = [
        (-hl, hw),
        (-hl, -hw),
        (hl, -hw),
        (hl, hw),
    ]
    corners = tuple(
        ParkingSlotCorner(
            x=round(cx + lx * cos_t - ly * sin_t, 3),
            y=round(cy + lx * sin_t + ly * cos_t, 3),
        )
        for lx, ly in local_pts
    )
    approach = SlotApproachPath(
        entry_point=(cx - 2.0 * cos_t, cy - 2.0 * sin_t),
        target_point=(cx, cy),
        entry_heading_rad=heading_rad,
        maneuver_length_m=2.0,
        clearance_margin_m=0.35,
        is_feasible=True,
    )
    return ParkingSlot(
        slot_id=slot_id,
        slot_type=slot_type,
        corners=corners,  # type: ignore[arg-type]
        center=center,
        heading_rad=heading_rad,
        width_m=width_m,
        length_m=length_m,
        status=status,
        approach_path=approach,
    )


def test_profile_segment_speed() -> None:
    wps = [
        KinematicWaypoint(x=0.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=0.0),
        KinematicWaypoint(x=1.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=1.0),
        KinematicWaypoint(x=2.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=2.0),
    ]
    seg, t_end = profile_segment_speed(
        wps,
        gear=ManeuverGear.FORWARD,
        phase=ManeuverPhase.APPROACH,
        segment_index=0,
        t_start=0.0,
        max_speed=1.5,
        max_accel=0.8,
    )
    assert seg.segment_index == 0
    assert seg.gear == ManeuverGear.FORWARD
    assert seg.length_m == 2.0
    assert seg.duration_s > 0.0
    assert len(seg.waypoints) == 3
    # Check start and end velocities are zero (stopped at boundaries)
    assert seg.waypoints[0].velocity == 0.0
    assert seg.waypoints[-1].velocity == 0.0
    assert t_end == pytest.approx(seg.duration_s, abs=1e-2)


def test_planner_no_vacant_slot() -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    slot = _make_slot(
        "slot_occ",
        ParkingSlotType.PARALLEL,
        center=(3.0, -2.5),
        status=SlotOccupancyStatus.OCCUPIED,
    )
    planner = ParkingTrajectoryPlanner()
    report = planner.plan_parking([slot], occupancy, grid)

    assert report.status == PlanStatus.NO_VACANT_SLOT
    assert report.is_executable is False
    assert report.plan is None


def test_planner_parallel_parking_success() -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    slot = _make_slot(
        "slot_par",
        ParkingSlotType.PARALLEL,
        center=(-1.0, -2.2),
        width_m=2.4,
        length_m=6.0,
        heading_rad=0.0,
        status=SlotOccupancyStatus.VACANT,
    )
    planner = ParkingTrajectoryPlanner()
    report = planner.plan_parking([slot], occupancy, grid, start_pose=(0.0, 0.0, 0.0))

    assert report.status == PlanStatus.SUCCESS
    assert report.is_executable is True
    assert report.selected_slot_id == "slot_par"
    assert report.plan is not None
    assert report.plan.gear_switches >= 1
    assert len(report.plan.segments) >= 2
    assert report.total_length_m > 1.0
    assert report.total_duration_s > 1.0


def test_planner_perpendicular_parking_success() -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    slot = _make_slot(
        "slot_perp",
        ParkingSlotType.PERPENDICULAR,
        center=(-2.0, -3.0),
        width_m=2.6,
        length_m=5.0,
        heading_rad=-math.pi / 2,
        status=SlotOccupancyStatus.VACANT,
    )
    planner = ParkingTrajectoryPlanner()
    report = planner.plan_parking([slot], occupancy, grid, start_pose=(0.0, 0.0, 0.0))

    assert report.status == PlanStatus.SUCCESS
    assert report.is_executable is True
    assert report.selected_slot_id == "slot_perp"
    assert report.plan is not None
    assert report.plan.slot_type == ParkingSlotType.PERPENDICULAR


def test_planner_slanted_parking_success() -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    slot = _make_slot(
        "slot_slant",
        ParkingSlotType.SLANTED,
        center=(-1.5, -2.8),
        width_m=2.5,
        length_m=5.2,
        heading_rad=-math.pi / 3,
        status=SlotOccupancyStatus.VACANT,
    )
    planner = ParkingTrajectoryPlanner()
    report = planner.plan_parking([slot], occupancy, grid, start_pose=(0.0, 0.0, 0.0))

    assert report.status == PlanStatus.SUCCESS
    assert report.is_executable is True
    assert report.selected_slot_id == "slot_slant"
    assert report.plan is not None
    assert report.plan.slot_type == ParkingSlotType.SLANTED


def test_planner_collision_rejection() -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    # Place a large wall of occupied cells right across the vehicle path
    occupancy[:, :] = 0.95

    slot = _make_slot(
        "slot_par",
        ParkingSlotType.PARALLEL,
        center=(-1.0, -2.2),
        width_m=2.4,
        length_m=6.0,
        status=SlotOccupancyStatus.VACANT,
    )
    planner = ParkingTrajectoryPlanner()
    report = planner.plan_parking([slot], occupancy, grid, start_pose=(0.0, 0.0, 0.0))

    assert report.status == PlanStatus.COLLISION_DETECTED
    assert report.is_executable is False
    assert report.plan is None
