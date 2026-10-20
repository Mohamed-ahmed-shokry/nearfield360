"""Unit tests for ArbitrageScheduler, economic evaluation, and departure SoC guarantees."""

from __future__ import annotations

from nearfield360.v2g.arbitrage import (
    ArbitrageScheduler,
    find_tariff_window,
    get_default_tariff_schedule,
    is_discharge_arbitrage_profitable,
)
from nearfield360.v2g.models import (
    GridTariffWindow,
    V2GPowerMode,
    VehicleBPTProfile,
)


def test_tariff_schedule_lookup() -> None:
    schedule = get_default_tariff_schedule()
    assert len(schedule) == 5

    # 02:00 -> OFF-PEAK-MORNING
    w_night = find_tariff_window(2.0, schedule)
    assert w_night.window_id == "OFF-PEAK-MORNING"
    assert w_night.import_price_per_kwh == 0.12

    # 10:00 -> MID-PEAK-DAY
    w_mid = find_tariff_window(10.0, schedule)
    assert w_mid.window_id == "MID-PEAK-DAY"

    # 16:00 -> ON-PEAK-EVENING
    w_peak = find_tariff_window(16.0, schedule)
    assert w_peak.window_id == "ON-PEAK-EVENING"
    assert w_peak.export_price_per_kwh == 0.36

    # 21:00 -> SHOULDER-NIGHT
    w_shoulder = find_tariff_window(21.0, schedule)
    assert w_shoulder.window_id == "SHOULDER-NIGHT"


def test_is_discharge_arbitrage_profitable() -> None:
    # Profitable: export $0.36, recharge $0.12, degradation $0.04
    # Recharge effective = 0.12 / 0.92 = 0.1304
    # Net spread = 0.36 - 0.04 - 0.1304 = 0.1896 > 0.01
    assert is_discharge_arbitrage_profitable(
        export_price=0.36,
        future_recharge_price=0.12,
        degradation_cost_per_kwh=0.04,
        round_trip_efficiency=0.92,
    )

    # Unprofitable: export $0.15, recharge $0.12, degradation $0.04
    # Net spread = 0.15 - 0.04 - 0.1304 = -0.0204 < 0.01
    assert not is_discharge_arbitrage_profitable(
        export_price=0.15,
        future_recharge_price=0.12,
        degradation_cost_per_kwh=0.04,
        round_trip_efficiency=0.92,
    )

    # Unprofitable due to high degradation cost
    assert not is_discharge_arbitrage_profitable(
        export_price=0.36,
        future_recharge_price=0.12,
        degradation_cost_per_kwh=0.25,
    )


def test_arbitrage_scheduler_departure_soc_guarantee() -> None:
    scheduler = ArbitrageScheduler()
    # Vehicle arrives at 8:00 with 30% battery, leaves at 16:00 requiring at least 70%
    profile = VehicleBPTProfile(
        vehicle_id="EV-DEP-001",
        battery_capacity_kwh=75.0,
        current_soc=0.30,
        min_departure_soc=0.70,
        max_discharge_kw=22.0,
        max_charge_kw=30.0,
        arrival_hour=8.0,
        departure_hour=16.0,
    )

    schedule = scheduler.schedule_vehicle(profile)
    assert schedule.vehicle_id == "EV-DEP-001"
    assert schedule.min_departure_soc_met
    assert schedule.final_soc >= 0.70
    assert schedule.total_energy_charged_kwh > 0.0
    assert len(schedule.dispatches) == 32  # 8 hours * 4 slots/hr


def test_arbitrage_scheduler_peak_discharge_and_arbitrage() -> None:
    scheduler = ArbitrageScheduler()
    # Arrives at 14:00 (peak start) with 90% battery, leaves at 22:00 requiring 60%
    profile = VehicleBPTProfile(
        vehicle_id="EV-ARB-001",
        battery_capacity_kwh=100.0,
        current_soc=0.90,
        min_departure_soc=0.60,
        max_discharge_kw=22.0,
        max_charge_kw=40.0,
        arrival_hour=14.0,
        departure_hour=22.0,
    )

    schedule = scheduler.schedule_vehicle(profile)
    assert schedule.min_departure_soc_met
    assert schedule.final_soc >= 0.60
    # During peak 14:00-20:00, vehicle should discharge to harvest revenue
    assert schedule.total_energy_discharged_kwh > 0.0
    assert schedule.net_revenue_dollars > 0.0

    # Ensure dispatches include DISCHARGE_PEAK_SHAVE
    discharge_slots = [
        d for d in schedule.dispatches if d.power_mode == V2GPowerMode.DISCHARGE_PEAK_SHAVE
    ]
    assert len(discharge_slots) > 0


def test_arbitrage_scheduler_prevents_unprofitable_discharge() -> None:
    # Set flat tariffs where export is always equal to or below import
    flat_schedule = [
        GridTariffWindow(
            window_id="FLAT",
            start_hour=0,
            end_hour=24,
            import_price_per_kwh=0.20,
            export_price_per_kwh=0.15,
            carbon_intensity_g_per_kwh=200.0,
        )
    ]
    scheduler = ArbitrageScheduler(tariff_schedule=flat_schedule)

    profile = VehicleBPTProfile(
        vehicle_id="EV-NO-ARB",
        battery_capacity_kwh=75.0,
        current_soc=0.85,
        min_departure_soc=0.60,
        max_discharge_kw=22.0,
        max_charge_kw=30.0,
        arrival_hour=9.0,
        departure_hour=17.0,
    )

    schedule = scheduler.schedule_vehicle(profile)
    # Should NOT discharge when export price is below import price
    assert schedule.total_energy_discharged_kwh == 0.0


def test_arbitrage_scheduler_short_dwell() -> None:
    scheduler = ArbitrageScheduler()
    profile = VehicleBPTProfile(
        vehicle_id="EV-SHORT",
        battery_capacity_kwh=60.0,
        current_soc=0.55,
        min_departure_soc=0.60,
        arrival_hour=10.0,
        departure_hour=10.5,  # 30 min = 2 slots
    )
    schedule = scheduler.schedule_vehicle(profile)
    assert len(schedule.dispatches) == 2
    assert schedule.final_soc >= 0.55
