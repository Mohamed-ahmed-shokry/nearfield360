"""Coil coupling factor (k) and electrical power transfer efficiency model for SAE J2954."""

import math

from nearfield360.config import WPTConfig
from nearfield360.wpt.models import (
    CoilOffset3D,
    FODStatus,
    WPTEfficiencyReport,
    WPTPowerClass,
    WPTZClass,
)


class WPTEfficiencyModel:
    """Computes magnetic coupling coefficient k, transfer efficiency, and power governor."""

    def __init__(
        self,
        config: WPTConfig | None = None,
        nominal_k0: float = 0.26,
        nominal_q: float = 280.0,
        inverter_efficiency: float = 0.95,
        radial_decay_sigma_mm: float = 135.0,
    ) -> None:
        self.config = config or WPTConfig()
        self.k0 = nominal_k0
        self.q_factor = nominal_q
        self.inverter_eff = inverter_efficiency
        self.sigma_r = radial_decay_sigma_mm

    def compute_coupling_factor(self, delta_r_mm: float, air_gap_mm: float) -> float:
        """Calculate mutual coupling coefficient k from radial offset and air gap."""
        r_decay = math.exp(-0.5 * (delta_r_mm / self.sigma_r) ** 2)
        z_ratio = (self.config.target_air_gap_mm / max(80.0, air_gap_mm)) ** 1.8
        k_val = float(self.k0 * r_decay * z_ratio)
        return float(round(max(0.01, min(0.40, k_val)), 3))

    def compute_transfer_efficiency(self, coupling_k: float) -> float:
        """Calculate end-to-end DC-to-DC wireless power transfer efficiency.

        Formula for resonant inductive link:
            eta_coil = (k^2 * Q^2) / (1 + sqrt(1 + k^2 * Q^2))^2
            eta_dc_dc = eta_coil * inverter_efficiency
        """
        k_q = coupling_k * self.q_factor
        k_q_sq = k_q**2
        coil_eff = k_q_sq / ((1.0 + math.sqrt(1.0 + k_q_sq)) ** 2)
        total_eff = coil_eff * self.inverter_eff
        return round(max(0.10, min(0.96, total_eff)), 3)

    def evaluate(
        self,
        offset: CoilOffset3D,
        fod_status: FODStatus | None = None,
        power_class: WPTPowerClass = WPTPowerClass.WPT3,
    ) -> WPTEfficiencyReport:
        """Evaluate power governor and calculate efficiency report."""
        coupling_k = self.compute_coupling_factor(offset.radial_offset_mm, offset.delta_z_mm)
        efficiency = self.compute_transfer_efficiency(coupling_k)

        # Determine Z-class
        if offset.delta_z_mm <= 150.0:
            z_class = WPTZClass.Z1
        elif offset.delta_z_mm <= 210.0:
            z_class = WPTZClass.Z2
        else:
            z_class = WPTZClass.Z3

        # Base rated power from power class
        rated_power_kw = (
            3.7
            if power_class == WPTPowerClass.WPT1
            else (
                7.7
                if power_class == WPTPowerClass.WPT2
                else (11.0 if power_class == WPTPowerClass.WPT3 else 22.0)
            )
        )

        # Power governor
        if fod_status and fod_status.power_inhibited:
            cmd_power = 0.0
        elif not offset.is_within_tolerance:
            # Low-power probing beacon during alignment
            cmd_power = 0.5
        elif offset.radial_offset_mm <= self.config.docked_radius_threshold_mm:
            # Full rated power when docked
            cmd_power = rated_power_kw
        else:
            # Scaled power within tolerance box
            scale = 1.0 - (0.4 * (offset.radial_offset_mm / self.config.max_longitudinal_offset_mm))
            cmd_power = round(rated_power_kw * max(0.5, scale), 2)

        delivered = round(cmd_power * efficiency, 2)
        thermal_loss = round(max(0.0, cmd_power - delivered), 2)

        return WPTEfficiencyReport(
            coupling_factor_k=coupling_k,
            transfer_efficiency=efficiency,
            commanded_power_kw=cmd_power,
            delivered_power_kw=delivered,
            thermal_loss_kw=thermal_loss,
            power_class=power_class,
            z_class=z_class,
        )
