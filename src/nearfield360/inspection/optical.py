"""360-degree surround optical surface damage and anomaly detector."""

from __future__ import annotations

import math
from typing import Any

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    OpticalAnomaly,
)


class OpticalDamageDetector:
    """Multi-view surround fisheye optical damage assessment and surface scoring engine."""

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config or InspectionConfig()

    def classify_severity(
        self,
        damage_type: DamageType,
        area_cm2: float,
        depth_mm: float,
    ) -> DamageSeverity:
        """Classify damage severity tier based on physical area, depth, and anomaly type."""
        if damage_type == DamageType.GLASS_CRACK:
            if depth_mm >= 3.0 or area_cm2 >= 20.0:
                return DamageSeverity.CRITICAL
            if depth_mm >= 1.5 or area_cm2 >= 10.0:
                return DamageSeverity.SEVERE
            return DamageSeverity.MODERATE

        if damage_type == DamageType.DENT:
            if depth_mm >= 5.0 or area_cm2 >= 50.0:
                return DamageSeverity.SEVERE
            if depth_mm >= self.config.dent_depth_threshold_mm or area_cm2 >= 20.0:
                return DamageSeverity.MODERATE
            return DamageSeverity.MINOR

        # Scratches, chips, panel misalignments
        if area_cm2 >= 30.0:
            return DamageSeverity.SEVERE
        if area_cm2 >= 12.0 or depth_mm >= 1.0:
            return DamageSeverity.MODERATE
        if area_cm2 >= 3.0:
            return DamageSeverity.MINOR
        return DamageSeverity.NEGLIGIBLE

    def process_defect(
        self,
        anomaly_id: str,
        camera_id: str,
        surface_location: str,
        damage_type: DamageType,
        area_cm2: float,
        depth_mm: float,
        bounding_box_norm: tuple[float, float, float, float] = (0.2, 0.2, 0.4, 0.4),
        confidence: float = 0.92,
        description: str = "",
    ) -> OpticalAnomaly:
        """Construct validated OpticalAnomaly with computed severity and description."""
        severity = self.classify_severity(
            damage_type=damage_type,
            area_cm2=area_cm2,
            depth_mm=depth_mm,
        )

        desc = description or (
            f"{severity.value} {damage_type.value.lower().replace('_', ' ')} on {surface_location} "
            f"({area_cm2:.1f} cm2, depth {depth_mm:.1f} mm)"
        )

        return OpticalAnomaly(
            anomaly_id=anomaly_id,
            camera_id=camera_id,
            surface_location=surface_location,
            damage_type=damage_type,
            severity=severity,
            bounding_box_norm=bounding_box_norm,
            estimated_area_cm2=round(area_cm2, 2),
            depth_estimate_mm=round(depth_mm, 2),
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            description=desc,
        )

    def scan_surround(
        self,
        defects_raw: list[dict[str, Any]] | None = None,
    ) -> list[OpticalAnomaly]:
        """Process 360-degree surround optical observations into structured anomalies."""
        if not defects_raw:
            return []

        anomalies: list[OpticalAnomaly] = []
        for idx, d in enumerate(defects_raw):
            ano_id = str(d.get("anomaly_id", f"ANO-{idx + 1:03d}"))
            cam_id = str(d.get("camera_id", "FV"))
            loc = str(d.get("surface_location", "front_bumper"))
            raw_type = str(d.get("damage_type", "SCRATCH")).upper()
            dtype = (
                DamageType(raw_type)
                if raw_type in DamageType._value2member_map_
                else DamageType.SCRATCH
            )
            area = float(d.get("estimated_area_cm2", 5.0))
            depth = float(d.get("depth_estimate_mm", 0.5))
            bbox = tuple(d.get("bounding_box_norm", (0.1, 0.1, 0.3, 0.3)))
            conf = float(d.get("confidence", 0.90))
            desc = str(d.get("description", ""))

            anomaly = self.process_defect(
                anomaly_id=ano_id,
                camera_id=cam_id,
                surface_location=loc,
                damage_type=dtype,
                area_cm2=area,
                depth_mm=depth,
                bounding_box_norm=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                confidence=conf,
                description=desc,
            )
            anomalies.append(anomaly)

        return anomalies

    def compute_surface_score(self, anomalies: list[OpticalAnomaly]) -> float:
        """Calculate overall exterior condition score out of 100 based on damage severity."""
        penalty_weights: dict[DamageSeverity, float] = {
            DamageSeverity.NEGLIGIBLE: 0.5,
            DamageSeverity.MINOR: 2.5,
            DamageSeverity.MODERATE: 7.0,
            DamageSeverity.SEVERE: 15.0,
            DamageSeverity.CRITICAL: 30.0,
        }

        total_penalty = sum(penalty_weights.get(a.severity, 2.0) for a in anomalies)
        score = max(0.0, min(100.0, 100.0 - total_penalty))
        return round(score, 1)

    def analyze_gradient_patch(
        self,
        patch_contrast: float,
        gradient_variance: float,
    ) -> tuple[bool, float]:
        """Evaluate raw image high-frequency gradient metrics to detect micro-scratch presence."""
        # Significant edge variance and local contrast signify surface scratch or fracture
        severity_score = math.tanh(patch_contrast * 1.5) * min(1.0, gradient_variance / 40.0)
        is_damaged = severity_score >= self.config.scratch_severity_threshold
        return is_damaged, round(float(severity_score), 3)
