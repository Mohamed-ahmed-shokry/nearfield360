"""Unit tests for BEV curbside staging and pedestrian safety visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2

from nearfield360.curbside.models import (
    BayStatus,
    CurbsideBay,
    CurbsideReport,
    HandoverType,
    SafetyInterlockStatus,
)
from nearfield360.curbside.viz import render_curbside_dashboard
from nearfield360.mapping.builder import build_benchmark_garage


def test_render_curbside_dashboard_memory() -> None:
    facility = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    bay1 = CurbsideBay(
        bay_id="dopu-01",
        bay_type=HandoverType.DROPOFF,
        status=BayStatus.OCCUPIED_UNLOADING,
        occupied_vehicle_id="veh-42",
        occupancy_start_s=10.0,
        interlock_status=SafetyInterlockStatus.ENGAGED_DOOR_OPEN,
    )
    bay2 = CurbsideBay(
        bay_id="dopu-02",
        bay_type=HandoverType.PICKUP,
        status=BayStatus.VACANT,
    )
    report = CurbsideReport(
        facility_id=facility.map_id,
        timestamp_s=25.0,
        total_bays_count=2,
        vacant_bays_count=1,
        bays=[bay1, bay2],
        active_missions=[],
        completed_missions_count=0,
        active_interlocks_count=1,
        total_overstay_penalties_usd=0.0,
        average_dwell_duration_s=15.0,
    )

    img = render_curbside_dashboard(facility, report, canvas_size=800)
    assert img is not None
    assert img.shape == (800, 800 + 440, 3)
    assert img.dtype.name == "uint8"


def test_render_curbside_dashboard_file_export(tmp_path: Path) -> None:
    facility = build_benchmark_garage(aisle_count=1, slots_per_aisle=2)
    report = CurbsideReport(
        facility_id=facility.map_id,
        timestamp_s=0.0,
        total_bays_count=0,
        vacant_bays_count=0,
        bays=[],
        active_missions=[],
        completed_missions_count=0,
        active_interlocks_count=0,
        total_overstay_penalties_usd=0.0,
        average_dwell_duration_s=0.0,
    )

    out_file = tmp_path / "curbside_hud.png"
    img = render_curbside_dashboard(facility, report, canvas_size=600, output_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == img.shape
