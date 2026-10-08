"""Unit tests for mission package __init__ exports."""

import nearfield360.mission as mission


def test_mission_package_exports() -> None:
    expected_exports = {
        "DynamicSafetyMonitor",
        "MissionEvent",
        "MissionExecutive",
        "MissionState",
        "MissionSummaryReport",
        "MissionTrigger",
        "ParkingReplanner",
        "RecoveryManeuver",
        "SlotTracker",
        "TrackedParkingSlot",
        "render_mission_dashboard_overlay",
        "render_mission_timeline_chart",
    }
    assert expected_exports.issubset(set(mission.__all__))
    for sym in expected_exports:
        assert hasattr(mission, sym)
