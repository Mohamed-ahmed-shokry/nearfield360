"""Windshield and automotive safety glass integrity analyzer compliant with ECE R43."""

from __future__ import annotations

from typing import Any

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    GlassInspectionReport,
    OpticalAnomaly,
)


class GlassIntegrityAnalyzer:
    """Evaluates optical transmittance, crack lengths, and stone chips per ECE R43."""

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config or InspectionConfig()

    def evaluate_glass(
        self,
        optical_transmittance_pct: float = 85.0,
        raw_chips: int = 0,
        raw_cracks: list[dict[str, Any]] | None = None,
    ) -> GlassInspectionReport:
        """Evaluate glass structural integrity and optical clarity."""
        cracks: list[OpticalAnomaly] = []

        if raw_cracks:
            for idx, c in enumerate(raw_cracks):
                cid = str(c.get("anomaly_id", f"GLASS-{idx + 1:03d}"))
                length_mm = float(c.get("length_mm", 10.0))
                depth_mm = float(c.get("depth_mm", 1.2))
                location = str(c.get("location", "windshield_driver_zone_a"))
                bbox = tuple(c.get("bounding_box_norm", (0.3, 0.4, 0.5, 0.6)))

                # Severity based on crack length
                if length_mm >= (self.config.glass_crack_length_threshold_mm * 2.0):
                    sev = DamageSeverity.CRITICAL
                elif length_mm >= self.config.glass_crack_length_threshold_mm:
                    sev = DamageSeverity.SEVERE
                elif length_mm >= 5.0:
                    sev = DamageSeverity.MODERATE
                else:
                    sev = DamageSeverity.MINOR

                anomaly = OpticalAnomaly(
                    anomaly_id=cid,
                    camera_id="FV",
                    surface_location=location,
                    damage_type=DamageType.GLASS_CRACK,
                    severity=sev,
                    bounding_box_norm=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    estimated_area_cm2=round(length_mm * 0.1, 2),
                    depth_estimate_mm=depth_mm,
                    confidence=0.95,
                    description=f"{sev.value} glass fissure on {location} ({length_mm:.1f} mm)",
                )
                cracks.append(anomaly)

        # Windshield clear if no cracks exceeding threshold and chip count <= 2
        has_critical_crack = any(
            c.severity in (DamageSeverity.SEVERE, DamageSeverity.CRITICAL) for c in cracks
        )
        windshield_clear = (not has_critical_crack) and (raw_chips <= 2)

        # ECE R43 requires minimum 70.0% light transmittance for windshields
        ece_compliant = (optical_transmittance_pct >= 70.0) and (not has_critical_crack)

        return GlassInspectionReport(
            windshield_clear=windshield_clear,
            cracks_detected=cracks,
            chip_count=max(0, raw_chips),
            optical_transmittance_pct=round(optical_transmittance_pct, 1),
            ece_r43_compliant=ece_compliant,
        )

    def check_regulatory_compliance(
        self,
        report: GlassInspectionReport,
    ) -> tuple[bool, list[str]]:
        """Verify regulatory safety compliance against ECE R43 glazing standards."""
        violations: list[str] = []

        if report.optical_transmittance_pct < 70.0:
            violations.append(
                f"Windshield optical transmittance ({report.optical_transmittance_pct}%) "
                "below ECE R43 minimum (70.0%)"
            )

        severe_cracks = [
            c
            for c in report.cracks_detected
            if c.severity in (DamageSeverity.SEVERE, DamageSeverity.CRITICAL)
        ]
        if severe_cracks:
            violations.append(
                f"Detected {len(severe_cracks)} structural glass crack(s) exceeding limit"
            )

        if report.chip_count > 5:
            violations.append(
                f"Excessive stone chip count ({report.chip_count}) obstructing driver forward field"
            )

        return (len(violations) == 0), violations
