"""Tests for windshield and automotive safety glass integrity analyzer (ECE R43)."""

from __future__ import annotations

from nearfield360.config import InspectionConfig
from nearfield360.inspection.glass import GlassIntegrityAnalyzer
from nearfield360.inspection.models import DamageSeverity, DamageType


def test_evaluate_glass_pristine_condition() -> None:
    analyzer = GlassIntegrityAnalyzer()
    report = analyzer.evaluate_glass(
        optical_transmittance_pct=88.5,
        raw_chips=0,
        raw_cracks=[],
    )

    assert report.windshield_clear is True
    assert report.ece_r43_compliant is True
    assert report.optical_transmittance_pct == 88.5
    assert len(report.cracks_detected) == 0

    compliant, violations = analyzer.check_regulatory_compliance(report)
    assert compliant is True
    assert len(violations) == 0


def test_evaluate_glass_minor_chips_within_allowance() -> None:
    analyzer = GlassIntegrityAnalyzer()
    report = analyzer.evaluate_glass(
        optical_transmittance_pct=82.0,
        raw_chips=2,
        raw_cracks=[],
    )

    assert report.windshield_clear is True
    assert report.ece_r43_compliant is True
    assert report.chip_count == 2


def test_evaluate_glass_severe_crack_failure() -> None:
    config = InspectionConfig(glass_crack_length_threshold_mm=15.0)
    analyzer = GlassIntegrityAnalyzer(config=config)

    cracks_data = [
        {
            "anomaly_id": "CRACK-01",
            "length_mm": 22.5,
            "depth_mm": 2.1,
            "location": "windshield_driver_zone_a",
        }
    ]

    report = analyzer.evaluate_glass(
        optical_transmittance_pct=85.0,
        raw_chips=1,
        raw_cracks=cracks_data,
    )

    assert report.windshield_clear is False
    assert report.ece_r43_compliant is False
    assert len(report.cracks_detected) == 1
    assert report.cracks_detected[0].severity == DamageSeverity.SEVERE
    assert report.cracks_detected[0].damage_type == DamageType.GLASS_CRACK

    compliant, violations = analyzer.check_regulatory_compliance(report)
    assert compliant is False
    assert any("structural glass crack" in v for v in violations)


def test_evaluate_glass_low_optical_transmittance_violation() -> None:
    analyzer = GlassIntegrityAnalyzer()
    # 65.0% transmittance fails the 70.0% legal minimum
    report = analyzer.evaluate_glass(
        optical_transmittance_pct=65.0,
        raw_chips=0,
    )

    assert report.ece_r43_compliant is False

    compliant, violations = analyzer.check_regulatory_compliance(report)
    assert compliant is False
    assert any("below ECE R43 minimum" in v for v in violations)
