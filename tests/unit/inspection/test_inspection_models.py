"""Tests for vehicle inspection domain models, damage enums, and schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    GlassInspectionReport,
    InspectionPhase,
    InspectionReport,
    LiabilityCertificate,
    LiabilityVerdict,
    OpticalAnomaly,
    TirePosition,
    TireTelemetry,
    TireWearStatus,
    VehicleConditionSnapshot,
)


def test_optical_anomaly_instantiation_and_validation() -> None:
    anomaly = OpticalAnomaly(
        anomaly_id="ANO-001",
        camera_id="cam_front",
        surface_location="front_bumper_center",
        damage_type=DamageType.SCRATCH,
        severity=DamageSeverity.MINOR,
        bounding_box_norm=(0.42, 0.55, 0.48, 0.62),
        estimated_area_cm2=12.5,
        depth_estimate_mm=0.8,
        confidence=0.94,
        description="Surface clear-coat scratch near center sensor",
    )

    assert anomaly.anomaly_id == "ANO-001"
    assert anomaly.damage_type == DamageType.SCRATCH
    assert anomaly.severity == DamageSeverity.MINOR
    assert anomaly.confidence == 0.94


def test_optical_anomaly_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        OpticalAnomaly(
            anomaly_id="ANO-ERR",
            camera_id="cam_rear",
            surface_location="rear_bumper",
            damage_type=DamageType.DENT,
            severity=DamageSeverity.MODERATE,
            bounding_box_norm=(0.1, 0.1, 0.2, 0.2),
            estimated_area_cm2=5.0,
            depth_estimate_mm=2.5,
            confidence=1.5,
            description="Invalid confidence",
        )


def test_tire_telemetry_evaluation() -> None:
    tire = TireTelemetry(
        position=TirePosition.FRONT_LEFT,
        inner_tread_depth_mm=4.5,
        center_tread_depth_mm=4.6,
        outer_tread_depth_mm=4.4,
        mean_tread_depth_mm=4.5,
        pressure_bar=2.42,
        temperature_c=22.5,
        status=TireWearStatus.GOOD,
        uneven_wear_detected=False,
    )

    assert tire.position == TirePosition.FRONT_LEFT
    assert tire.mean_tread_depth_mm == 4.5
    assert tire.status == TireWearStatus.GOOD
    assert tire.uneven_wear_detected is False


def test_glass_inspection_report_defaults() -> None:
    report = GlassInspectionReport()
    assert report.windshield_clear is True
    assert report.chip_count == 0
    assert report.optical_transmittance_pct == 85.0
    assert report.ece_r43_compliant is True
    assert len(report.cracks_detected) == 0


def test_vehicle_condition_snapshot_and_certificate() -> None:
    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=14250.0,
        overall_condition_score=96.5,
    )
    assert snapshot.vin == "WP0ZZZ99ZTS392001"
    assert snapshot.inspection_phase == InspectionPhase.PRE_DROPOFF
    assert snapshot.overall_condition_score == 96.5

    cert = LiabilityCertificate(
        certificate_id="CERT-2026-001",
        vin=snapshot.vin,
        session_id="SESS-AVP-88",
        phase=snapshot.inspection_phase,
        payload_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        signature_hmac="a" * 64,
        issuer_id="facility-gate-01",
        operator_id="avp-operator-alice",
        verdict=LiabilityVerdict.NO_NEW_DAMAGE,
    )
    assert cert.verdict == LiabilityVerdict.NO_NEW_DAMAGE
    assert cert.certificate_id == "CERT-2026-001"


def test_inspection_report_to_dict_serialization() -> None:
    snapshot = VehicleConditionSnapshot(
        vin="TEST-VIN-1234",
        inspection_phase=InspectionPhase.POST_PICKUP,
        mileage_km=500.0,
    )
    report = InspectionReport(
        vehicle_id="EV-101",
        facility_id="FAC-DOWNTOWN-01",
        bay_id="INSPECT-BAY-A",
        phase=InspectionPhase.POST_PICKUP,
        snapshot=snapshot,
    )

    data = report.to_dict()
    assert data["vehicle_id"] == "EV-101"
    assert data["phase"] == "POST_PICKUP"
    assert data["snapshot"]["vin"] == "TEST-VIN-1234"
    assert "generated_at_utc" in data
