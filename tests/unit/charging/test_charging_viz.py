"""Unit tests for BEV EV charging visualizer and dashboard renderer."""

from __future__ import annotations

from pathlib import Path

import cv2

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargerType,
    ChargingDispatchReport,
    ChargingSession,
    ChargingSessionStatus,
    FacilityPowerSummary,
)
from nearfield360.charging.viz import render_charging_dashboard
from nearfield360.mapping.builder import build_benchmark_garage


def test_render_charging_dashboard_exports_valid_image(tmp_path: Path) -> None:
    facility = build_benchmark_garage()

    bay1 = ChargerBay(
        bay_id="BAY_01",
        slot_id=facility.slots[0].slot_id,
        charger_type=ChargerType.DC_FAST_150KW,
        max_power_rating_kw=150.0,
        current_power_kw=80.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
        occupied_by_vehicle_id="EV_01",
    )
    bay2 = ChargerBay(
        bay_id="BAY_02",
        slot_id=facility.slots[1].slot_id,
        charger_type=ChargerType.DC_FAST_50KW,
        max_power_rating_kw=50.0,
        current_power_kw=0.0,
        status=ChargerPortStatus.AVAILABLE,
    )

    sess1 = ChargingSession(
        session_id="SESS_01",
        vehicle_id="EV_01",
        bay_id="BAY_01",
        start_time_s=10.0,
        last_update_time_s=50.0,
        start_soc=0.20,
        current_soc=0.65,
        target_soc=0.80,
        allocated_power_kw=80.0,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )

    power_summary = FacilityPowerSummary(
        facility_id="garage_test",
        grid_capacity_kw=250.0,
        total_requested_power_kw=80.0,
        total_allocated_power_kw=80.0,
        available_headroom_kw=170.0,
        active_charging_sessions=1,
        throttled_sessions=0,
        grid_utilization_ratio=0.32,
    )
    report = ChargingDispatchReport(
        facility_id="garage_test",
        timestamp_s=50.0,
        power_summary=power_summary,
        total_bays=2,
        available_bays=1,
        occupied_charging_bays=1,
        waiting_relocation_bays=0,
    )

    out_png = tmp_path / "charging_dashboard.png"
    img = render_charging_dashboard(
        facility=facility,
        bays=[bay1, bay2],
        sessions=[sess1],
        report=report,
        canvas_size=600,
        output_path=out_png,
    )

    assert img.shape[0] == 600
    assert img.shape[1] == 600 + 420
    assert out_png.exists()

    # Verify PNG header
    loaded = cv2.imread(str(out_png))
    assert loaded is not None
    assert loaded.shape == img.shape
