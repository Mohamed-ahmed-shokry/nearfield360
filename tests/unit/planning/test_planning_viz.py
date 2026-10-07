from __future__ import annotations

from pathlib import Path

import numpy as np

from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.models import (
    ManeuverGear,
    ManeuverPhase,
    ManeuverSegment,
    ParkingTrajectoryPlan,
    PlanStatus,
    TrajectoryWaypoint,
)
from nearfield360.planning.viz import render_parking_plan_bev_overlay
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    ParkingSlotType,
    SlotOccupancyStatus,
)


def test_render_parking_plan_bev_overlay(tmp_path: Path) -> None:
    grid = BevGrid(x_min=-6.0, x_max=10.0, y_min=-6.0, y_max=6.0, resolution=0.1)
    wp1 = TrajectoryWaypoint(x=0.0, y=0.0, heading_rad=0.0, gear=ManeuverGear.REVERSE, t=0.0)
    wp2 = TrajectoryWaypoint(x=-2.0, y=-1.0, heading_rad=0.2, gear=ManeuverGear.REVERSE, t=1.5)
    seg1 = ManeuverSegment(
        segment_index=0,
        phase=ManeuverPhase.STEER_IN,
        gear=ManeuverGear.REVERSE,
        length_m=2.2,
        duration_s=1.5,
        waypoints=[wp1, wp2],
    )
    plan = ParkingTrajectoryPlan(
        plan_id="plan_01",
        slot_id="slot_01",
        slot_type=ParkingSlotType.PARALLEL,
        status=PlanStatus.SUCCESS,
        total_length_m=2.2,
        total_duration_s=1.5,
        gear_switches=0,
        max_curvature=0.2,
        min_clearance_m=0.35,
        start_pose=(0.0, 0.0, 0.0),
        target_pose=(-2.0, -1.0, 0.2),
        is_executable=True,
        segments=[seg1],
    )

    c0 = ParkingSlotCorner(x=-3.0, y=-2.0)
    c1 = ParkingSlotCorner(x=-1.0, y=-2.0)
    c2 = ParkingSlotCorner(x=-1.0, y=0.0)
    c3 = ParkingSlotCorner(x=-3.0, y=0.0)
    slot = ParkingSlot(
        slot_id="slot_01",
        slot_type=ParkingSlotType.PARALLEL,
        corners=(c0, c1, c2, c3),
        center=(-2.0, -1.0),
        heading_rad=0.0,
        width_m=2.0,
        length_m=2.0,
        status=SlotOccupancyStatus.VACANT,
    )

    out_file = tmp_path / "plan_bev.png"
    img = render_parking_plan_bev_overlay(
        grid=grid,
        plan=plan,
        slots=[slot],
        output_path=out_file,
    )

    assert isinstance(img, np.ndarray)
    assert img.shape == (*grid.shape, 3)
    assert img.dtype == np.uint8
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
