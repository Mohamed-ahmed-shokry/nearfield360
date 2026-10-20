"""Smart bidirectional grid-tied inverter simulation and power conversion engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from nearfield360.v2g.models import InverterPhase, InverterTelemetry, V2GPowerMode

if TYPE_CHECKING:
    from nearfield360.config import V2GConfig


class SmartInverter:
    """Simulates a bidirectional AC/DC smart inverter conforming to IEEE 1547 and ISO 15118-20."""

    def __init__(
        self,
        inverter_id: str,
        rated_power_kw: float = 50.0,
        phase: InverterPhase = InverterPhase.THREE_PHASE,
        nominal_voltage_rms: float = 400.0,
        nominal_frequency_hz: float = 50.0,
        peak_efficiency: float = 0.975,
        config: V2GConfig | None = None,
    ) -> None:
        self.inverter_id = inverter_id
        self.rated_power_kw = max(1.0, float(rated_power_kw))
        self.phase = phase
        self.nominal_voltage_rms = max(100.0, float(nominal_voltage_rms))
        self.nominal_frequency_hz = max(40.0, float(nominal_frequency_hz))
        self.peak_efficiency = min(0.99, max(0.85, float(peak_efficiency)))

        if config is not None:
            self.nominal_frequency_hz = config.grid_frequency_nominal_hz

        # Internal state
        self._active_power_kw: float = 0.0  # + = charging (from AC), - = discharging (to AC)
        self._reactive_power_kvar: float = 0.0
        self._grid_frequency_hz: float = self.nominal_frequency_hz
        self._voltage_rms: float = self.nominal_voltage_rms
        self._dc_bus_voltage_v: float = 800.0
        self._power_mode: V2GPowerMode = V2GPowerMode.STANDBY

    @property
    def power_mode(self) -> V2GPowerMode:
        return self._power_mode

    @property
    def active_power_kw(self) -> float:
        return self._active_power_kw

    @property
    def reactive_power_kvar(self) -> float:
        return self._reactive_power_kvar

    def compute_efficiency(self, ac_power_kw: float) -> float:
        """Compute non-linear conversion efficiency based on fractional power loading."""
        loading = min(1.0, abs(ac_power_kw) / self.rated_power_kw)
        if loading < 0.01:
            return 0.88
        # Standard quadratic inverter loss model
        efficiency = self.peak_efficiency - 0.045 * ((1.0 - loading) ** 2)
        return max(0.85, min(self.peak_efficiency, efficiency))

    def convert_power(self, target_ac_power_kw: float) -> tuple[float, float, float]:
        """Convert power bidirectionally between AC grid and DC battery bus.

        Args:
            target_ac_power_kw: Commanded AC active power in kW.
                Positive (>0) = Charging from grid to battery.
                Negative (<0) = Discharging from battery to grid.
                Zero (0) = Standby.

        Returns:
            Tuple of (effective_ac_power_kw, battery_dc_power_kw, dissipation_loss_kw).
        """
        # Clamp to rated inverter capacity
        effective_ac_kw = max(-self.rated_power_kw, min(self.rated_power_kw, target_ac_power_kw))
        if abs(effective_ac_kw) < 1e-3:
            self._active_power_kw = 0.0
            self._power_mode = V2GPowerMode.STANDBY
            return 0.0, 0.0, 0.0

        eta = self.compute_efficiency(effective_ac_kw)

        if effective_ac_kw > 0.0:
            # Charging: AC grid draws power, battery receives DC power reduced by efficiency
            battery_dc_kw = effective_ac_kw * eta
            loss_kw = effective_ac_kw - battery_dc_kw
            self._power_mode = V2GPowerMode.CHARGE
        else:
            # Discharging: Battery must supply more DC power to deliver effective_ac_kw to AC grid
            # battery_dc_kw is negative (energy leaving battery)
            battery_dc_kw = effective_ac_kw / eta
            loss_kw = abs(battery_dc_kw) - abs(effective_ac_kw)
            self._power_mode = V2GPowerMode.DISCHARGE_PEAK_SHAVE

        self._active_power_kw = effective_ac_kw
        return effective_ac_kw, battery_dc_kw, loss_kw

    def apply_frequency_droop(
        self,
        grid_frequency_hz: float,
        droop_gain_kw_per_hz: float = 25.0,
        deadband_hz: float = 0.036,
    ) -> float:
        """Modulate active power according to IEEE 1547 Frequency-Watt droop characteristics.

        Args:
            grid_frequency_hz: Current grid frequency measurement.
            droop_gain_kw_per_hz: Frequency response gain in kW/Hz.
            deadband_hz: Permissible frequency deadband before droop intervention.

        Returns:
            Adjusted effective AC active power in kW.
        """
        self._grid_frequency_hz = max(40.0, min(70.0, float(grid_frequency_hz)))
        delta_f = self._grid_frequency_hz - self.nominal_frequency_hz

        if delta_f < -deadband_hz:
            # Under-frequency: Grid deficit -> Inject power (discharge) or reduce charging
            deficit = -delta_f - deadband_hz
            delta_p_kw = -float(droop_gain_kw_per_hz) * deficit
        elif delta_f > deadband_hz:
            # Over-frequency: Grid surplus -> Absorb power (charge) or reduce discharging
            surplus = delta_f - deadband_hz
            delta_p_kw = float(droop_gain_kw_per_hz) * surplus
        else:
            delta_p_kw = 0.0

        new_ac_power = max(
            -self.rated_power_kw,
            min(self.rated_power_kw, self._active_power_kw + delta_p_kw),
        )
        self._active_power_kw = new_ac_power

        if abs(new_ac_power) < 1e-3:
            self._power_mode = V2GPowerMode.STANDBY
        elif new_ac_power < 0.0:
            self._power_mode = V2GPowerMode.DISCHARGE_FREQUENCY_SUPPORT
        else:
            self._power_mode = V2GPowerMode.CHARGE

        return self._active_power_kw

    def apply_voltage_var_compensation(
        self,
        voltage_rms: float,
        droop_gain_kvar_per_volt: float = 0.20,
        deadband_volts: float = 5.0,
    ) -> float:
        """Modulate reactive power according to IEEE 1547 Voltage-VAR compensation.

        Args:
            voltage_rms: AC RMS line-to-line voltage in Volts.
            droop_gain_kvar_per_volt: Reactive power gain in kVAR/Volt.
            deadband_volts: Voltage deadband before reactive support intervention.

        Returns:
            Effective reactive power in kVAR (positive = capacitive, negative = inductive).
        """
        self._voltage_rms = max(50.0, float(voltage_rms))
        delta_v = self._voltage_rms - self.nominal_voltage_rms

        if delta_v < -deadband_volts:
            # Under-voltage: Inject capacitive VARs to boost voltage
            sag = -delta_v - deadband_volts
            q_target = float(droop_gain_kvar_per_volt) * sag
        elif delta_v > deadband_volts:
            # Over-voltage: Absorb inductive VARs to suppress voltage
            swell = delta_v - deadband_volts
            q_target = -float(droop_gain_kvar_per_volt) * swell
        else:
            q_target = 0.0

        # Maximum available reactive power given active power constraint (S^2 = P^2 + Q^2)
        s_rated = 1.10 * self.rated_power_kw
        q_limit = math.sqrt(max(0.0, (s_rated**2) - (self._active_power_kw**2)))
        self._reactive_power_kvar = max(-q_limit, min(q_limit, q_target))
        return self._reactive_power_kvar

    def get_power_factor(self) -> float:
        """Compute the instantaneous displacement power factor."""
        p = abs(self._active_power_kw)
        q = self._reactive_power_kvar
        apparent_power = math.hypot(p, q)
        if apparent_power < 1e-3:
            return 1.0
        return max(-1.0, min(1.0, self._active_power_kw / apparent_power))

    def measure_telemetry(self, timestamp: float = 0.0) -> InverterTelemetry:
        """Generate a complete telemetry measurement snapshot."""
        eta = self.compute_efficiency(self._active_power_kw)
        return InverterTelemetry(
            inverter_id=self.inverter_id,
            phase=self.phase,
            active_power_kw=round(self._active_power_kw, 3),
            reactive_power_kvar=round(self._reactive_power_kvar, 3),
            grid_frequency_hz=round(self._grid_frequency_hz, 3),
            voltage_rms=round(self._voltage_rms, 2),
            dc_bus_voltage_v=round(self._dc_bus_voltage_v, 2),
            efficiency_percent=round(eta * 100.0, 2),
            power_factor=round(self.get_power_factor(), 3),
            timestamp=max(0.0, float(timestamp)),
        )
