"""Unit tests for EV charging domain models, serialization, and properties."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargerType,
    ChargingDispatchReport,
    ChargingSession,
    ChargingSessionStatus,
    EVBatteryState,
    FacilityPowerSummary,
    RelocationOrder,
)


def test_charger_bay_defaults_and_validation() -> None:
    bay = ChargerBay(
        bay_id="BAY_01",
        slot_id="SLOT_CHG_01",
        charger_type=ChargerType.DC_FAST_150KW,
        max_power_rating_kw=150.0,
        current_power_kw=50.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )
    assert bay.bay_id == "BAY_01"
    assert bay.slot_id == "SLOT_CHG_01"
    assert bay.charger_type == ChargerType.DC_FAST_150KW
    assert bay.max_power_rating_kw == 150.0
    assert bay.current_power_kw == 50.0
    assert bay.status == ChargerPortStatus.OCCUPIED_CHARGING
    assert bay.efficiency_rating == 0.95


def test_charger_bay_rejects_negative_power() -> None:
    with pytest.raises(ValidationError):
        ChargerBay(
            bay_id="BAY_BAD",
            slot_id="SLOT_01",
            max_power_rating_kw=-50.0,
        )


def test_ev_battery_state_properties() -> None:
    battery = EVBatteryState(
        vehicle_id="EV_01",
        battery_capacity_kwh=100.0,
        current_soc=0.20,
        target_soc=0.80,
        max_charge_acceptance_kw=150.0,
        degradation_factor=0.90,
    )
    # Energy = capacity * soc * degradation = 100 * 0.20 * 0.90 = 18.0 kWh
    assert pytest.approx(battery.current_energy_kwh) == 18.0
    # Energy needed = 100 * (0.80 - 0.20) * 0.90 = 54.0 kWh
    assert pytest.approx(battery.energy_needed_kwh) == 54.0
    assert not battery.is_critical_low

    low_batt = EVBatteryState(
        vehicle_id="EV_LOW",
        current_soc=0.15,
    )
    assert low_batt.is_critical_low


def test_charging_session_serialization_and_updates() -> None:
    session = ChargingSession(
        session_id="SESS_001",
        vehicle_id="EV_01",
        bay_id="BAY_01",
        start_time_s=100.0,
        last_update_time_s=100.0,
        start_soc=0.20,
        current_soc=0.20,
        target_soc=0.80,
        allocated_power_kw=50.0,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )
    assert session.status == ChargingSessionStatus.ACTIVE_CHARGING
    session.current_soc = 0.50
    session.energy_delivered_kwh = 25.0
    assert session.current_soc == 0.50

    json_str = session.to_json()
    data = json.loads(json_str)
    assert data["session_id"] == "SESS_001"
    assert data["vehicle_id"] == "EV_01"
    assert data["current_soc"] == 0.50


def test_facility_power_summary_and_dispatch_report() -> None:
    power_summary = FacilityPowerSummary(
        facility_id="garage_ev",
        grid_capacity_kw=250.0,
        total_requested_power_kw=180.0,
        total_allocated_power_kw=180.0,
        available_headroom_kw=70.0,
        active_charging_sessions=3,
        throttled_sessions=0,
        grid_utilization_ratio=0.72,
        is_curtailed=False,
    )
    reloc_order = RelocationOrder(
        order_id="RELOC_01",
        vehicle_id="EV_DONE",
        source_bay_id="BAY_01",
        target_slot_id="SLOT_STD_05",
        created_at_s=1200.0,
        reason="target_soc_reached",
        urgency_priority=2,
    )
    report = ChargingDispatchReport(
        facility_id="garage_ev",
        timestamp_s=1200.0,
        power_summary=power_summary,
        total_bays=4,
        available_bays=1,
        occupied_charging_bays=2,
        waiting_relocation_bays=1,
        pending_queue_count=1,
        relocation_orders=[reloc_order],
    )
    assert report.total_bays == 4
    assert len(report.relocation_orders) == 1
    serialized = json.loads(report.to_json())
    assert serialized["power_summary"]["grid_capacity_kw"] == 250.0
    assert serialized["relocation_orders"][0]["vehicle_id"] == "EV_DONE"
