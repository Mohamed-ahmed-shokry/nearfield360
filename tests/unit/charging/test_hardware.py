"""Unit tests for ChargingHardwareSimulator CC/CV curve and robotic docking simulation."""

from __future__ import annotations

import pytest

from nearfield360.charging.hardware import ChargingHardwareSimulator
from nearfield360.charging.models import (
    ChargerBay,
    ChargerPortStatus,
    ChargingSession,
    ChargingSessionStatus,
    EVBatteryState,
)


def test_hardware_simulator_docking_dwell_transition() -> None:
    simulator = ChargingHardwareSimulator()

    bay = ChargerBay(
        bay_id="B1",
        slot_id="S1",
        docking_dwell_s=15.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )
    battery = EVBatteryState(vehicle_id="EV1", current_soc=0.20, target_soc=0.80)
    session: ChargingSession = ChargingSession(
        session_id="S1",
        vehicle_id="EV1",
        bay_id="B1",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.20,
        current_soc=0.20,
        target_soc=0.80,
        allocated_power_kw=50.0,
        status=ChargingSessionStatus.PENDING_DOCKING,
    )

    # Step 10 seconds: still pending docking
    energy = simulator.step_session(session, battery, bay, dt_s=10.0)
    assert energy == 0.0
    assert session.status.value == "pending_docking"

    # Step another 10 seconds (total 20s > 15s): transitions to ACTIVE_CHARGING
    energy = simulator.step_session(session, battery, bay, dt_s=10.0)
    assert session.status.value == "active_charging"


def test_hardware_simulator_cc_and_cv_charging_taper() -> None:
    simulator = ChargingHardwareSimulator(taper_factor=6.0)

    # In CC phase (SoC 0.50 <= 0.80)
    battery_cc = EVBatteryState(vehicle_id="EV_CC", current_soc=0.50, target_soc=0.90)
    power_cc = simulator.compute_instantaneous_power_acceptance(
        battery_cc, allocated_power_kw=100.0
    )
    assert power_cc == 100.0

    # In CV phase (SoC 0.90 > 0.80): power must taper down
    battery_cv = EVBatteryState(vehicle_id="EV_CV", current_soc=0.90, target_soc=1.00)
    power_cv = simulator.compute_instantaneous_power_acceptance(
        battery_cv, allocated_power_kw=100.0
    )
    assert power_cv < 100.0
    assert power_cv > 0.0

    # Fully charged
    battery_full = EVBatteryState(vehicle_id="EV_FULL", current_soc=0.80, target_soc=0.80)
    power_full = simulator.compute_instantaneous_power_acceptance(
        battery_full, allocated_power_kw=100.0
    )
    assert power_full == 0.0


def test_hardware_simulator_energy_delivery_and_soc_increase() -> None:
    simulator = ChargingHardwareSimulator()

    bay = ChargerBay(
        bay_id="B1",
        slot_id="S1",
        efficiency_rating=1.0,
        status=ChargerPortStatus.OCCUPIED_CHARGING,
    )
    # 100 kWh battery, starts at 0.20 SoC
    battery = EVBatteryState(
        vehicle_id="EV1",
        battery_capacity_kwh=100.0,
        current_soc=0.20,
        target_soc=0.80,
    )
    session = ChargingSession(
        session_id="S1",
        vehicle_id="EV1",
        bay_id="B1",
        start_time_s=0.0,
        last_update_time_s=0.0,
        start_soc=0.20,
        current_soc=0.20,
        target_soc=0.80,
        allocated_power_kw=100.0,
        status=ChargingSessionStatus.ACTIVE_CHARGING,
    )

    # Deliver 100 kW for 1800s (0.5 hour) = 50 kWh
    energy = simulator.step_session(session, battery, bay, dt_s=1800.0)
    assert pytest.approx(energy, rel=1e-3) == 50.0
    # 50 kWh on 100 kWh battery should increase SoC by 0.50 -> 0.70 SoC
    assert pytest.approx(battery.current_soc, rel=1e-3) == 0.70
    assert pytest.approx(session.current_soc, rel=1e-3) == 0.70
