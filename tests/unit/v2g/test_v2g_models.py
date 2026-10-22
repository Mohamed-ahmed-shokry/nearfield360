"""Unit tests for V2G domain models, validation constraints, and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.v2g.models import (
    GridTariffWindow,
    InverterPhase,
    InverterTelemetry,
    MicrogridFacilityReport,
    V2GArbitrageSchedule,
    V2GPowerMode,
    V2GTimeSlotDispatch,
    VehicleBPTProfile,
)


def test_v2g_power_mode_and_phase_enums() -> None:
    assert V2GPowerMode.STANDBY == "standby"
    assert V2GPowerMode.CHARGE == "charge"
    assert V2GPowerMode.DISCHARGE_PEAK_SHAVE == "discharge_peak_shave"
    assert V2GPowerMode.DISCHARGE_FREQUENCY_SUPPORT == "discharge_frequency_support"
    assert V2GPowerMode.ISLANDED_BACKUP == "islanded_backup"

    assert InverterPhase.SINGLE_PHASE == "single_phase"
    assert InverterPhase.THREE_PHASE == "three_phase"


def test_grid_tariff_window_validation() -> None:
    window = GridTariffWindow(
        window_id="PEAK-AFTERNOON",
        start_hour=14,
        end_hour=18,
        import_price_per_kwh=0.38,
        export_price_per_kwh=0.32,
        carbon_intensity_g_per_kwh=320.0,
    )
    assert window.window_id == "PEAK-AFTERNOON"
    assert window.import_price_per_kwh == 0.38
    assert window.export_price_per_kwh == 0.32

    # Verify JSON serialization round-trip
    dumped = window.model_dump_json()
    reloaded = GridTariffWindow.model_validate_json(dumped)
    assert reloaded == window

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        GridTariffWindow(
            window_id="INVALID",
            start_hour=0,
            end_hour=6,
            import_price_per_kwh=0.10,
            export_price_per_kwh=0.08,
            unknown_field=123,  # type: ignore[call-arg]
        )


def test_vehicle_bpt_profile_validation_and_constraints() -> None:
    profile = VehicleBPTProfile(
        vehicle_id="EV-TESLA-001",
        battery_capacity_kwh=75.0,
        current_soc=0.45,
        min_departure_soc=0.70,
        max_discharge_kw=22.0,
        max_charge_kw=50.0,
        arrival_hour=8.0,
        departure_hour=17.0,
        degradation_cost_factor=1.0,
    )
    assert profile.vehicle_id == "EV-TESLA-001"
    assert profile.battery_capacity_kwh == 75.0

    # Reject departure before or equal to arrival
    with pytest.raises(ValidationError, match="must be strictly after arrival_hour"):
        VehicleBPTProfile(
            vehicle_id="EV-INVALID-001",
            battery_capacity_kwh=60.0,
            current_soc=0.50,
            arrival_hour=12.0,
            departure_hour=10.0,
        )

    # Reject invalid SoC
    with pytest.raises(ValidationError):
        VehicleBPTProfile(
            vehicle_id="EV-INVALID-002",
            battery_capacity_kwh=60.0,
            current_soc=1.5,
            arrival_hour=8.0,
            departure_hour=12.0,
        )


def test_inverter_telemetry_serialization() -> None:
    telemetry = InverterTelemetry(
        inverter_id="INV-BAY-01",
        phase=InverterPhase.THREE_PHASE,
        active_power_kw=-18.5,  # discharging
        reactive_power_kvar=2.1,
        grid_frequency_hz=49.95,
        voltage_rms=401.2,
        dc_bus_voltage_v=805.0,
        efficiency_percent=97.2,
        power_factor=0.99,
        timestamp=100.0,
    )
    assert telemetry.active_power_kw == -18.5
    assert telemetry.grid_frequency_hz == 49.95

    json_str = telemetry.model_dump_json()
    reloaded = InverterTelemetry.model_validate_json(json_str)
    assert reloaded == telemetry


def test_v2g_arbitrage_schedule_and_facility_report() -> None:
    dispatch1 = V2GTimeSlotDispatch(
        slot_index=0,
        hour=14.0,
        power_mode=V2GPowerMode.DISCHARGE_PEAK_SHAVE,
        active_power_kw=-20.0,
        soc_end=0.42,
        import_tariff=0.35,
        export_tariff=0.30,
        cost_or_revenue_delta=1.50,
    )
    schedule = V2GArbitrageSchedule(
        vehicle_id="EV-001",
        dispatches=[dispatch1],
        total_energy_charged_kwh=0.0,
        total_energy_discharged_kwh=5.0,
        net_cost_dollars=0.0,
        net_revenue_dollars=1.50,
        net_profit_dollars=1.30,
        final_soc=0.72,
        min_departure_soc_met=True,
    )

    report = MicrogridFacilityReport(
        total_bays_connected=10,
        total_battery_capacity_kwh=650.0,
        available_flexible_energy_kwh=220.0,
        baseline_facility_peak_kw=420.0,
        peak_shaved_demand_kw=340.0,
        transformer_utilization_percent=68.0,
        net_arbitrage_profit_dollars=48.50,
        carbon_offset_kg=85.2,
        schedules=[schedule],
        is_islanded=False,
    )
    assert report.total_bays_connected == 10
    assert report.peak_shaved_demand_kw == 340.0
    assert len(report.schedules) == 1

    # Round trip JSON
    dumped = report.model_dump_json()
    parsed = MicrogridFacilityReport.model_validate_json(dumped)
    assert parsed.net_arbitrage_profit_dollars == 48.50
