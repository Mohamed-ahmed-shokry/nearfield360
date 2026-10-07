from __future__ import annotations

import nearfield360.planning as planning


def test_planning_package_exports() -> None:
    assert hasattr(planning, "ParkingTrajectoryPlanner")
    assert hasattr(planning, "ParkingTrajectoryPlan")
    assert hasattr(planning, "AckermannVehicle")
    assert hasattr(planning, "SweptFootprintEvaluator")
    assert hasattr(planning, "render_parking_plan_bev_overlay")
    assert hasattr(planning, "ManeuverGear")
    assert hasattr(planning, "PlanStatus")
