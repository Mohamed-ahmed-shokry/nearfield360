"""Unit tests for SmartInverter power conversion, frequency droop, and voltage-VAR compensation."""

from __future__ import annotations

import pytest

from nearfield360.v2g.inverter import SmartInverter
from nearfield360.v2g.models import InverterPhase, V2GPowerMode


def test_inverter_initialization_and_properties() -> None:
    inverter = SmartInverter(
        inverter_id="INV-001",
        rated_power_kw=50.0,
        phase=InverterPhase.THREE_PHASE,
        nominal_voltage_rms=400.0,
        nominal_frequency_hz=50.0,
        peak_efficiency=0.975,
    )
    assert inverter.inverter_id == "INV-001"
    assert inverter.rated_power_kw == 50.0
    assert inverter.phase == InverterPhase.THREE_PHASE
    assert inverter.power_mode == V2GPowerMode.STANDBY
    assert inverter.active_power_kw == 0.0
    assert inverter.reactive_power_kvar == 0.0


def test_inverter_conversion_charging() -> None:
    inverter = SmartInverter(inverter_id="INV-CHG", rated_power_kw=50.0)
    effective_ac, battery_dc, loss_kw = inverter.convert_power(30.0)

    assert effective_ac == 30.0
    # Battery DC power is smaller than AC draw due to efficiency losses
    assert 0.0 < battery_dc < effective_ac
    assert loss_kw == pytest.approx(effective_ac - battery_dc, rel=1e-4)
    assert inverter.power_mode == V2GPowerMode.CHARGE
    assert inverter.active_power_kw == 30.0


def test_inverter_conversion_discharging() -> None:
    inverter = SmartInverter(inverter_id="INV-DIS", rated_power_kw=50.0)
    effective_ac, battery_dc, loss_kw = inverter.convert_power(-22.0)

    assert effective_ac == -22.0
    # Battery must supply more magnitude DC than AC delivered to grid
    assert battery_dc < effective_ac  # e.g. -22.8 kW < -22.0 kW
    assert loss_kw == pytest.approx(abs(battery_dc) - abs(effective_ac), rel=1e-4)
    assert inverter.power_mode == V2GPowerMode.DISCHARGE_PEAK_SHAVE
    assert inverter.active_power_kw == -22.0


def test_inverter_standby_and_clamping() -> None:
    inverter = SmartInverter(inverter_id="INV-CLAMP", rated_power_kw=40.0)

    # Standby
    ac, dc, loss = inverter.convert_power(0.0)
    assert ac == 0.0
    assert dc == 0.0
    assert loss == 0.0
    assert inverter.power_mode == V2GPowerMode.STANDBY

    # Clamping beyond rated power
    ac_clamped, _, _ = inverter.convert_power(85.0)
    assert ac_clamped == 40.0

    ac_clamped_neg, _, _ = inverter.convert_power(-100.0)
    assert ac_clamped_neg == -40.0


def test_frequency_droop_under_frequency() -> None:
    inverter = SmartInverter(inverter_id="INV-DROOP-UF", rated_power_kw=50.0)
    # Start at 0 kW standby
    inverter.convert_power(0.0)

    # Grid drops to 49.80 Hz (deficit of 0.20 Hz, exceeding deadband 0.036 Hz)
    adjusted_p = inverter.apply_frequency_droop(
        grid_frequency_hz=49.80, droop_gain_kw_per_hz=25.0, deadband_hz=0.036
    )

    # Deficit beyond deadband is (0.20 - 0.036) = 0.164 Hz
    # Expected injection delta P = -25.0 * 0.164 = -4.1 kW (discharging into grid)
    assert adjusted_p < 0.0
    assert adjusted_p == pytest.approx(-4.1, abs=0.01)
    assert inverter.power_mode == V2GPowerMode.DISCHARGE_FREQUENCY_SUPPORT


def test_frequency_droop_over_frequency() -> None:
    inverter = SmartInverter(inverter_id="INV-DROOP-OF", rated_power_kw=50.0)
    # Start discharging at -10 kW
    inverter.convert_power(-10.0)

    # Grid rises to 50.30 Hz (surplus of 0.30 Hz)
    adjusted_p = inverter.apply_frequency_droop(
        grid_frequency_hz=50.30, droop_gain_kw_per_hz=20.0, deadband_hz=0.036
    )

    # Overfrequency requires curtailing discharge (making active power more positive)
    assert adjusted_p > -10.0


def test_frequency_droop_within_deadband() -> None:
    inverter = SmartInverter(inverter_id="INV-DEADBAND", rated_power_kw=50.0)
    inverter.convert_power(-15.0)

    # 49.98 Hz is within 0.036 Hz deadband
    adjusted_p = inverter.apply_frequency_droop(
        grid_frequency_hz=49.98, droop_gain_kw_per_hz=25.0, deadband_hz=0.036
    )
    assert adjusted_p == -15.0


def test_voltage_var_compensation_and_telemetry() -> None:
    inverter = SmartInverter(inverter_id="INV-VAR", rated_power_kw=50.0, nominal_voltage_rms=400.0)
    inverter.convert_power(20.0)

    # Voltage sag: 385V (15V below nominal, 10V beyond 5V deadband)
    q = inverter.apply_voltage_var_compensation(
        voltage_rms=385.0, droop_gain_kvar_per_volt=0.50, deadband_volts=5.0
    )
    # 10V beyond deadband * 0.50 kVAR/V = +5.0 kVAR (capacitive)
    assert q == pytest.approx(5.0, abs=0.01)

    # Power factor
    pf = inverter.get_power_factor()
    assert 0.90 <= pf <= 1.0

    # Telemetry snapshot
    telemetry = inverter.measure_telemetry(timestamp=125.0)
    assert telemetry.inverter_id == "INV-VAR"
    assert telemetry.active_power_kw == 20.0
    assert telemetry.reactive_power_kvar == pytest.approx(5.0, abs=0.01)
    assert telemetry.voltage_rms == 385.0
    assert telemetry.timestamp == 125.0
    assert 90.0 <= telemetry.efficiency_percent <= 99.0
