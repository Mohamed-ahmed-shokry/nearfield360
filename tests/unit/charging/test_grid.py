"""Unit tests for GridPowerManager dynamic load balancing and peak shaving."""

from __future__ import annotations

from nearfield360.charging.grid import GridPowerManager
from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargerType,
    ChargingSession,
    ChargingSessionStatus,
    EVBatteryState,
)
from nearfield360.config import ChargingConfig


def test_grid_power_manager_unconstrained_allocation() -> None:
    config = ChargingConfig(facility_grid_power_kw=200.0)
    manager = GridPowerManager(config=config, facility_id="test_grid")

    bay1 = ChargerBay(
        bay_id="B1",
        slot_id="S1",
        charger_type=ChargerType.DC_FAST_50KW,
        max_power_rating_kw=50.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )
    bay2 = ChargerBay(
        bay_id="B2",
        slot_id="S2",
        charger_type=ChargerType.DC_FAST_50KW,
        max_power_rating_kw=50.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )

    sess1 = ChargingSession(
        session_id="SESS_1",
        vehicle_id="EV_1",
        bay_id="B1",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.20,
        current_soc=0.20,
        target_soc=0.80,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )
    sess2 = ChargingSession(
        session_id="SESS_2",
        vehicle_id="EV_2",
        bay_id="B2",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.40,
        current_soc=0.40,
        target_soc=0.80,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )

    bat1 = EVBatteryState(vehicle_id="EV_1", current_soc=0.20, max_charge_acceptance_kw=50.0)
    bat2 = EVBatteryState(vehicle_id="EV_2", current_soc=0.40, max_charge_acceptance_kw=50.0)

    summary = manager.balance_power([sess1, sess2], [bay1, bay2], [bat1, bat2])

    assert not summary.is_curtailed
    assert summary.active_charging_sessions == 2
    assert summary.total_allocated_power_kw == 100.0
    assert summary.available_headroom_kw == 100.0
    assert summary.throttled_sessions == 0
    assert sess1.allocated_power_kw == 50.0
    assert sess2.allocated_power_kw == 50.0
    assert bay1.current_power_kw == 50.0


def test_grid_power_manager_curtails_when_exceeding_capacity() -> None:
    # Capacity is 80 kW, but demand from two 50 kW bays is 100 kW
    config = ChargingConfig(
        facility_grid_power_kw=80.0,
        critical_soc_threshold=0.20,
    )
    manager = GridPowerManager(config=config, facility_id="test_grid")

    bay1 = ChargerBay(
        bay_id="B1",
        slot_id="S1",
        max_power_rating_kw=50.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )
    bay2 = ChargerBay(
        bay_id="B2",
        slot_id="S2",
        max_power_rating_kw=50.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )

    # EV_1 is critically depleted (SoC 0.10), EV_2 has normal SoC (0.50)
    sess1 = ChargingSession(
        session_id="SESS_1",
        vehicle_id="EV_1",
        bay_id="B1",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.10,
        current_soc=0.10,
        target_soc=0.80,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )
    sess2 = ChargingSession(
        session_id="SESS_2",
        vehicle_id="EV_2",
        bay_id="B2",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.50,
        current_soc=0.50,
        target_soc=0.80,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )

    bat1 = EVBatteryState(vehicle_id="EV_1", current_soc=0.10, max_charge_acceptance_kw=50.0)
    bat2 = EVBatteryState(vehicle_id="EV_2", current_soc=0.50, max_charge_acceptance_kw=50.0)

    summary = manager.balance_power([sess1, sess2], [bay1, bay2], [bat1, bat2])

    assert summary.is_curtailed
    assert summary.total_allocated_power_kw <= 80.0
    # EV_1 has higher weight and should receive more power than EV_2
    assert sess1.allocated_power_kw > sess2.allocated_power_kw
    assert summary.throttled_sessions > 0


def test_grid_power_manager_handles_zero_active_sessions() -> None:
    config = ChargingConfig(facility_grid_power_kw=100.0)
    manager = GridPowerManager(config=config)
    summary = manager.balance_power([], [], [])

    assert summary.active_charging_sessions == 0
    assert summary.total_allocated_power_kw == 0.0
    assert summary.available_headroom_kw == 100.0
    assert not summary.is_curtailed
