"""Unit tests for BEV teleoperation cockpit and MRM HUD visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2

from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.teleop.models import (
    MRMStage,
    TeleopMode,
    TeleopReport,
    TeleopSession,
)
from nearfield360.teleop.viz import render_teleop_dashboard


def test_render_teleop_dashboard_memory() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    s1 = TeleopSession(
        session_id="teleop-0001",
        vehicle_id="veh-01",
        operator_id="op-desk-A",
        active_mode=TeleopMode.TELE_GUIDANCE,
        started_at_s=10.0,
        commands_processed_count=4,
    )
    s2 = TeleopSession(
        session_id="teleop-0002",
        vehicle_id="veh-02",
        operator_id="op-desk-B",
        active_mode=TeleopMode.MRM_SAFE_STOP,
        started_at_s=15.0,
        mrm_stage=MRMStage.DECELERATING,
        commands_processed_count=1,
    )

    report = TeleopReport(
        facility_id=facility.map_id,
        timestamp_s=30.0,
        active_sessions_count=2,
        completed_sessions_count=0,
        total_interventions_count=2,
        average_rtt_ms=65.4,
        mrm_executions_count=1,
        sessions=[s1, s2],
    )

    img = render_teleop_dashboard(facility, report, canvas_size=800)
    assert img is not None
    assert img.shape == (800, 800 + 440, 3)
    assert img.dtype.name == "uint8"
    assert img.sum() > 0


def test_render_teleop_dashboard_file_export(tmp_path: Path) -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2)
    report = TeleopReport(
        facility_id=facility.map_id,
        timestamp_s=0.0,
        active_sessions_count=0,
        completed_sessions_count=0,
        total_interventions_count=0,
        average_rtt_ms=0.0,
        mrm_executions_count=0,
        sessions=[],
    )

    out_file = tmp_path / "teleop_hud.png"
    img = render_teleop_dashboard(facility, report, canvas_size=600, output_path=out_file)

    assert img.shape == (600, 600 + 440, 3)
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000

    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == (600, 600 + 440, 3)
