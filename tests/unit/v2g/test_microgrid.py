"""Unit tests for FacilityMicrogridManager, peak shaving, and islanded backup."""

from __future__ import annotations

import pytest

from nearfield360.config import V2GConfig
from nearfield360.v2g.microgrid import (
    ConnectedBay,
    FacilityMicrogridManager,
    compute_facility_base_load_kw,
    compute_facility_solar_pv_kw,
    create_default_demonstrator_fleet,
)
from nearfield360.v2g.models import VehicleBPTProfile


def test_facility_base_load_and_solar_curves() -> None:
    # Base load is lowest at night (~80 kW) and highest at midday (~220 kW)
    p_night = compute_facility_base_load_kw(2.0)
    p_day = compute_facility_base_load_kw(13.0)
    assert 75.0 <= p_night <= 85.0
    assert 200.0 <= p_day <= 230.0

    # Solar PV: 0 at night (23:00, 04:00), peaks around midday (12:00)
    assert compute_facility_solar_pv_kw(23.0) == 0.0
    assert compute_facility_solar_pv_kw(4.0) == 0.0
    p_solar_noon = compute_facility_solar_pv_kw(12.0)
    assert 115.0 <= p_solar_noon <= 120.0


def test_connected_bay_capacity_and_flexibility() -> None:
    vehicle = VehicleBPTProfile(
        vehicle_id="EV-BAY-01",
        battery_capacity_kwh=100.0,
        current_soc=0.85,
        min_departure_soc=0.60,
        max_discharge_kw=22.0,
        max_charge_kw=40.0,
        arrival_hour=8.0,
        departure_hour=17.0,
    )
    bay = ConnectedBay(bay_id="BAY-01", vehicle=vehicle)

    assert bay.current_energy_kwh == 85.0
    # Flexible energy: (0.85 - 0.60) * 100 * 0.96 = 25 * 0.96 = 24.0 kWh
    assert bay.flexible_discharge_energy_kwh == pytest.approx(24.0, abs=0.1)


def test_microgrid_connect_disconnect() -> None:
    manager = FacilityMicrogridManager()
    v1 = VehicleBPTProfile(
        vehicle_id="EV-1",
        battery_capacity_kwh=75.0,
        current_soc=0.80,
        min_departure_soc=0.60,
        arrival_hour=8.0,
        departure_hour=16.0,
    )
    v2 = VehicleBPTProfile(
        vehicle_id="EV-2",
        battery_capacity_kwh=100.0,
        current_soc=0.90,
        min_departure_soc=0.50,
        arrival_hour=9.0,
        departure_hour=18.0,
    )

    manager.connect_vehicle("BAY-01", v1)
    manager.connect_vehicle("BAY-02", v2)

    assert len(manager.bays) == 2
    assert manager.get_aggregate_capacity_kwh() == 175.0
    assert manager.get_aggregate_flexible_energy_kwh() > 40.0

    # Disconnect
    removed = manager.disconnect_vehicle("BAY-01")
    assert removed is not None
    assert len(manager.bays) == 1
    assert manager.get_aggregate_capacity_kwh() == 100.0


def test_microgrid_peak_shaving_below_target() -> None:
    config = V2GConfig(peak_shaving_target_kw=350.0)
    manager = FacilityMicrogridManager(config=config)

    v = VehicleBPTProfile(
        vehicle_id="EV-TEST",
        battery_capacity_kwh=80.0,
        current_soc=0.80,
        min_departure_soc=0.60,
        arrival_hour=8.0,
        departure_hour=17.0,
    )
    manager.connect_vehicle("BAY-01", v)

    # Hour 2:00 base load is ~80 kW, well below 350 kW target
    dispatches = manager.dispatch_peak_shaving(hour=2.0, ev_unmanaged_charge_kw=0.0)
    assert dispatches["BAY-01"] == 0.0


def test_microgrid_peak_shaving_active() -> None:
    config = V2GConfig(peak_shaving_target_kw=300.0)
    manager = FacilityMicrogridManager(config=config)

    v1 = VehicleBPTProfile(
        vehicle_id="EV-1",
        battery_capacity_kwh=80.0,
        current_soc=0.90,
        min_departure_soc=0.50,
        max_discharge_kw=22.0,
        arrival_hour=8.0,
        departure_hour=18.0,
    )
    v2 = VehicleBPTProfile(
        vehicle_id="EV-2",
        battery_capacity_kwh=100.0,
        current_soc=0.85,
        min_departure_soc=0.60,
        max_discharge_kw=22.0,
        arrival_hour=8.0,
        departure_hour=18.0,
    )
    manager.connect_vehicle("BAY-01", v1)
    manager.connect_vehicle("BAY-02", v2)

    # At 14:00, net base (214.8 kW - 90 kW solar = 124.8 kW) + 250 kW EV charging = 374.8 kW
    # Excess above 300 kW target is ~74.8 kW
    dispatches = manager.dispatch_peak_shaving(hour=14.0, ev_unmanaged_charge_kw=250.0)

    assert "BAY-01" in dispatches
    assert "BAY-02" in dispatches
    assert dispatches["BAY-01"] < 0.0  # discharging
    assert dispatches["BAY-02"] < 0.0  # discharging

    total_shaved = abs(dispatches["BAY-01"]) + abs(dispatches["BAY-02"])
    assert total_shaved > 20.0
    assert abs(dispatches["BAY-01"]) <= 22.0
    assert abs(dispatches["BAY-02"]) <= 22.0


def test_microgrid_islanded_backup() -> None:
    manager = FacilityMicrogridManager()
    fleet = create_default_demonstrator_fleet()
    for i, v in enumerate(fleet):
        manager.connect_vehicle(f"BAY-{i + 1:02d}", v)

    # 10 vehicles aggregate significant flexible energy (> 100 kWh)
    flex_energy, hours, meets_target = manager.simulate_islanded_backup(critical_load_kw=25.0)
    assert flex_energy > 100.0
    assert hours >= 4.0
    assert meets_target is True


def test_microgrid_full_facility_simulation() -> None:
    manager = FacilityMicrogridManager()
    fleet = create_default_demonstrator_fleet()
    for i, v in enumerate(fleet):
        manager.connect_vehicle(f"BAY-{i + 1:02d}", v)

    report = manager.run_facility_simulation()
    assert report.total_bays_connected == 10
    assert report.total_battery_capacity_kwh > 700.0
    assert len(report.schedules) == 10
    assert report.peak_shaved_demand_kw <= report.baseline_facility_peak_kw
    assert 0.0 < report.transformer_utilization_percent < 100.0
    assert report.carbon_offset_kg >= 0.0
