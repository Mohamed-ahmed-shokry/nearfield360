"""4-wheel tire tread depth profiling, shoulder wear asymmetry, and TPMS pressure analyzer."""

from __future__ import annotations

from typing import Any

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import (
    TirePosition,
    TireTelemetry,
    TireWearStatus,
)


class TireProfiler:
    """Tire tread depth profiling, asymmetric wear detection, and cold pressure verification."""

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config or InspectionConfig()

    def evaluate_wheel(
        self,
        position: TirePosition,
        inner_depth_mm: float,
        center_depth_mm: float,
        outer_depth_mm: float,
        pressure_bar: float,
        temperature_c: float = 20.0,
    ) -> TireTelemetry:
        """Evaluate multi-groove tread depth profile and inflation pressure for a wheel."""
        mean_depth = round((inner_depth_mm + center_depth_mm + outer_depth_mm) / 3.0, 2)
        shoulder_diff = abs(inner_depth_mm - outer_depth_mm)
        center_diff = abs(center_depth_mm - ((inner_depth_mm + outer_depth_mm) / 2.0))

        uneven_wear = shoulder_diff >= self.config.max_tread_wear_variance_mm or center_diff >= 1.0

        # Check pressure deviation
        p_nominal = self.config.nominal_cold_pressure_bar
        p_tol = self.config.pressure_deviation_tolerance_bar
        pressure_fault = (pressure_bar < (p_nominal - p_tol)) or (
            pressure_bar > (p_nominal + p_tol)
        )

        # Status categorization
        if mean_depth <= self.config.min_legal_tread_depth_mm:
            status = TireWearStatus.CRITICAL_REPLACE
        elif mean_depth <= self.config.recommended_tread_depth_mm:
            status = TireWearStatus.WARNING_LOW_TREAD
        elif pressure_fault:
            status = TireWearStatus.PRESSURE_FAULT
        elif uneven_wear or mean_depth <= 4.5:
            status = TireWearStatus.ACCEPTABLE
        else:
            status = TireWearStatus.GOOD

        return TireTelemetry(
            position=position,
            inner_tread_depth_mm=round(inner_depth_mm, 2),
            center_tread_depth_mm=round(center_depth_mm, 2),
            outer_tread_depth_mm=round(outer_depth_mm, 2),
            mean_tread_depth_mm=mean_depth,
            pressure_bar=round(pressure_bar, 2),
            temperature_c=round(temperature_c, 1),
            status=status,
            uneven_wear_detected=uneven_wear,
        )

    def evaluate_four_wheels(
        self,
        wheel_data: dict[TirePosition, dict[str, Any]] | None = None,
    ) -> list[TireTelemetry]:
        """Profile all 4 vehicle wheels from telemetry readings or default nominal values."""
        results: list[TireTelemetry] = []

        for pos in (
            TirePosition.FRONT_LEFT,
            TirePosition.FRONT_RIGHT,
            TirePosition.REAR_LEFT,
            TirePosition.REAR_RIGHT,
        ):
            if wheel_data and pos in wheel_data:
                d = wheel_data[pos]
                t = self.evaluate_wheel(
                    position=pos,
                    inner_depth_mm=float(d.get("inner", 5.5)),
                    center_depth_mm=float(d.get("center", 5.6)),
                    outer_depth_mm=float(d.get("outer", 5.4)),
                    pressure_bar=float(d.get("pressure", self.config.nominal_cold_pressure_bar)),
                    temperature_c=float(d.get("temperature", 20.0)),
                )
            else:
                t = self.evaluate_wheel(
                    position=pos,
                    inner_depth_mm=6.0,
                    center_depth_mm=6.0,
                    outer_depth_mm=6.0,
                    pressure_bar=self.config.nominal_cold_pressure_bar,
                    temperature_c=20.0,
                )
            results.append(t)

        return results

    def compute_axle_variance(
        self,
        tires: list[TireTelemetry],
    ) -> dict[str, float]:
        """Compute cross-axle tread depth variance between front and rear axles."""
        by_pos = {t.position: t.mean_tread_depth_mm for t in tires}

        fl = by_pos.get(TirePosition.FRONT_LEFT, 6.0)
        fr = by_pos.get(TirePosition.FRONT_RIGHT, 6.0)
        rl = by_pos.get(TirePosition.REAR_LEFT, 6.0)
        rr = by_pos.get(TirePosition.REAR_RIGHT, 6.0)

        front_diff = round(abs(fl - fr), 2)
        rear_diff = round(abs(rl - rr), 2)
        axle_diff = round(abs(((fl + fr) / 2.0) - ((rl + rr) / 2.0)), 2)

        return {
            "front_cross_variance_mm": front_diff,
            "rear_cross_variance_mm": rear_diff,
            "axle_longitudinal_variance_mm": axle_diff,
        }
