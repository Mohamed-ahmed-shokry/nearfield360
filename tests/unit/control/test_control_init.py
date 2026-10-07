"""Unit tests for the control package init exports."""

from __future__ import annotations

import nearfield360.control as control


def test_control_package_exports() -> None:
    expected = {
        "ControlCommand",
        "ControlPerformanceKPIs",
        "DynamicSafetyMonitor",
        "ExecutionStatus",
        "ManeuverExecutionReport",
        "ManeuverExecutionStep",
        "ManeuverExecutor",
        "PathTrackingController",
        "SimulatorNoiseConfig",
        "TrackingErrorState",
        "VehicleSimState",
        "VehicleSimulator",
        "render_control_execution_bev_overlay",
        "render_control_telemetry_chart",
    }
    assert expected.issubset(set(control.__all__))
    for item in expected:
        assert hasattr(control, item)
