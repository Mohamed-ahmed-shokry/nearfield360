"""Unit tests for tactical V2G energy visualizer and PNG export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nearfield360.v2g.arbitrage import get_default_tariff_schedule
from nearfield360.v2g.microgrid import (
    FacilityMicrogridManager,
    create_default_demonstrator_fleet,
)
from nearfield360.v2g.models import VehicleBPTProfile
from nearfield360.v2g.viz import render_v2g_dashboard


def test_render_v2g_dashboard_array() -> None:
    manager = FacilityMicrogridManager()
    v1 = VehicleBPTProfile(
        vehicle_id="EV-TESLA-01",
        battery_capacity_kwh=75.0,
        current_soc=0.85,
        min_departure_soc=0.60,
        max_discharge_kw=22.0,
        max_charge_kw=40.0,
        arrival_hour=8.0,
        departure_hour=17.0,
    )
    bay = manager.connect_vehicle("BAY-01", v1)
    bay.inverter.convert_power(-18.0)  # Active discharging

    canvas = render_v2g_dashboard(manager, hour=15.0)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8
    # Canvas should not be entirely black
    assert np.mean(canvas) > 10.0


def test_render_v2g_dashboard_file_export(tmp_path: Path) -> None:
    manager = FacilityMicrogridManager()
    fleet = create_default_demonstrator_fleet()
    for i, v in enumerate(fleet):
        bay = manager.connect_vehicle(f"BAY-{i + 1:02d}", v)
        if i % 3 == 0:
            bay.inverter.convert_power(-20.0)  # V2G discharge
        elif i % 3 == 1:
            bay.inverter.convert_power(35.0)  # Charge

    tariff = get_default_tariff_schedule()[3]  # ON-PEAK
    out_png = tmp_path / "v2g_test_dashboard.png"

    canvas = render_v2g_dashboard(
        manager=manager,
        hour=16.0,
        tariff_window=tariff,
        output_path=out_png,
    )

    assert out_png.is_file()
    assert out_png.stat().st_size > 50_000  # PNG image should be substantial

    loaded = cv2.imread(str(out_png))
    assert loaded is not None
    assert loaded.shape == (720, 1280, 3)
    assert np.array_equal(canvas, loaded)


def test_render_v2g_dashboard_with_report(tmp_path: Path) -> None:
    manager = FacilityMicrogridManager()
    fleet = create_default_demonstrator_fleet()
    for i, v in enumerate(fleet):
        manager.connect_vehicle(f"BAY-{i + 1:02d}", v)

    report = manager.run_facility_simulation()
    out_png = tmp_path / "facility_report_dashboard.png"

    canvas = render_v2g_dashboard(
        manager=manager,
        report=report,
        hour=14.0,
        output_path=out_png,
    )

    assert canvas.shape == (720, 1280, 3)
    assert out_png.is_file()
    assert out_png.stat().st_size > 50_000
