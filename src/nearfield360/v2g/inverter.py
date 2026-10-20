"""Smart bidirectional grid-tied inverter simulation and power conversion engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nearfield360.v2g.models import InverterPhase, V2GPowerMode

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
