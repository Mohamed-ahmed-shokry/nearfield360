"""Tests for 360-degree optical surface damage and anomaly detector."""

from __future__ import annotations

from nearfield360.config import InspectionConfig
from nearfield360.inspection.models import DamageSeverity, DamageType
from nearfield360.inspection.optical import OpticalDamageDetector


def test_classify_severity_scratches_and_dents() -> None:
    detector = OpticalDamageDetector()

    # Small superficial scratch
    sev_small = detector.classify_severity(
        damage_type=DamageType.SCRATCH,
        area_cm2=2.0,
        depth_mm=0.2,
    )
    assert sev_small == DamageSeverity.NEGLIGIBLE

    # Moderate scratch
    sev_mod = detector.classify_severity(
        damage_type=DamageType.SCRATCH,
        area_cm2=15.0,
        depth_mm=0.5,
    )
    assert sev_mod == DamageSeverity.MODERATE

    # Severe dent
    sev_dent = detector.classify_severity(
        damage_type=DamageType.DENT,
        area_cm2=60.0,
        depth_mm=6.5,
    )
    assert sev_dent == DamageSeverity.SEVERE

    # Critical glass crack
    sev_glass = detector.classify_severity(
        damage_type=DamageType.GLASS_CRACK,
        area_cm2=25.0,
        depth_mm=3.5,
    )
    assert sev_glass == DamageSeverity.CRITICAL


def test_process_defect_and_description_generation() -> None:
    detector = OpticalDamageDetector()
    anomaly = detector.process_defect(
        anomaly_id="ANO-101",
        camera_id="MVL",
        surface_location="left_front_door",
        damage_type=DamageType.DENT,
        area_cm2=25.0,
        depth_mm=2.5,
    )

    assert anomaly.anomaly_id == "ANO-101"
    assert anomaly.camera_id == "MVL"
    assert anomaly.severity == DamageSeverity.MODERATE
    assert "MODERATE dent on left_front_door" in anomaly.description


def test_scan_surround_raw_defects_parsing() -> None:
    detector = OpticalDamageDetector()
    raw = [
        {
            "anomaly_id": "D1",
            "camera_id": "FV",
            "surface_location": "front_hood",
            "damage_type": "PAINT_CHIP",
            "estimated_area_cm2": 4.5,
            "depth_estimate_mm": 0.3,
        },
        {
            "anomaly_id": "D2",
            "camera_id": "RV",
            "surface_location": "rear_bumper",
            "damage_type": "SCRATCH",
            "estimated_area_cm2": 35.0,
            "depth_estimate_mm": 1.2,
        },
    ]

    anomalies = detector.scan_surround(raw)
    assert len(anomalies) == 2
    assert anomalies[0].damage_type == DamageType.PAINT_CHIP
    assert anomalies[0].severity == DamageSeverity.MINOR
    assert anomalies[1].severity == DamageSeverity.SEVERE


def test_compute_surface_score_pristine_and_degraded() -> None:
    detector = OpticalDamageDetector()

    # Pristine vehicle with zero defects
    assert detector.compute_surface_score([]) == 100.0

    # Vehicle with a couple anomalies
    d1 = detector.process_defect(
        "A1", "FV", "front", DamageType.SCRATCH, 2.0, 0.2
    )  # negligible: -0.5
    d2 = detector.process_defect(
        "A2", "RV", "rear", DamageType.SCRATCH, 15.0, 0.5
    )  # moderate: -7.0
    score = detector.compute_surface_score([d1, d2])
    assert score == 92.5


def test_analyze_gradient_patch_scratch_detection() -> None:
    config = InspectionConfig(scratch_severity_threshold=0.35)
    detector = OpticalDamageDetector(config=config)

    # Smooth pristine panel: low contrast, low variance
    is_damaged_clean, score_clean = detector.analyze_gradient_patch(
        patch_contrast=0.1,
        gradient_variance=5.0,
    )
    assert is_damaged_clean is False
    assert score_clean < 0.35

    # Scratched panel: high contrast, high gradient variance
    is_damaged_scratch, score_scratch = detector.analyze_gradient_patch(
        patch_contrast=0.8,
        gradient_variance=35.0,
    )
    assert is_damaged_scratch is True
    assert score_scratch >= 0.35
