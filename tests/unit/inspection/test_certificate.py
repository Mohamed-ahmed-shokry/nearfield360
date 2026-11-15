"""Tests for cryptographically signed liability certificate manager and comparator."""

from __future__ import annotations

from nearfield360.config import InspectionConfig
from nearfield360.inspection.certificate import LiabilityCertificateManager
from nearfield360.inspection.models import (
    DamageSeverity,
    DamageType,
    InspectionPhase,
    LiabilityVerdict,
    OpticalAnomaly,
    VehicleConditionSnapshot,
)


def test_issue_and_verify_valid_certificate() -> None:
    manager = LiabilityCertificateManager()
    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=12500.0,
        overall_condition_score=98.0,
    )

    cert = manager.issue_certificate(
        snapshot=snapshot,
        session_id="SESS-001",
        issuer_id="facility-gate-01",
    )

    assert cert.vin == snapshot.vin
    assert len(cert.payload_sha256) == 64
    assert len(cert.signature_hmac) == 64
    assert cert.verdict == LiabilityVerdict.NO_NEW_DAMAGE

    # Verification must pass for unaltered snapshot
    assert manager.verify_certificate(cert, snapshot) is True


def test_tampered_snapshot_fails_verification() -> None:
    manager = LiabilityCertificateManager()
    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=12500.0,
        overall_condition_score=98.0,
    )
    cert = manager.issue_certificate(snapshot, "SESS-002")

    # Tampered snapshot with altered mileage
    tampered_snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=18000.0,
        overall_condition_score=98.0,
    )

    assert manager.verify_certificate(cert, tampered_snapshot) is False


def test_tampered_hmac_secret_fails_verification() -> None:
    manager1 = LiabilityCertificateManager(
        InspectionConfig(certificate_signing_salt="secret-salt-one-2026")
    )
    manager2 = LiabilityCertificateManager(
        InspectionConfig(certificate_signing_salt="different-secret-salt-two")
    )

    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=12500.0,
    )
    cert = manager1.issue_certificate(snapshot, "SESS-003")

    # Verifying with manager using different salt must fail
    assert manager2.verify_certificate(cert, snapshot) is False


def test_compare_snapshots_no_new_damage() -> None:
    manager = LiabilityCertificateManager()
    pre = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=12500.0,
        overall_condition_score=100.0,
    )
    post = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.POST_PICKUP,
        mileage_km=12502.0,
        overall_condition_score=100.0,
    )

    comp = manager.compare_snapshots(pre, post)
    assert comp["verdict"] == LiabilityVerdict.NO_NEW_DAMAGE
    assert comp["new_damage_count"] == 0
    assert comp["score_delta"] == 0.0


def test_compare_snapshots_new_scratch_detected() -> None:
    manager = LiabilityCertificateManager()
    pre = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=12500.0,
        overall_condition_score=100.0,
    )

    new_scratch = OpticalAnomaly(
        anomaly_id="SCR-01",
        camera_id="MVL",
        surface_location="driver_door",
        damage_type=DamageType.SCRATCH,
        severity=DamageSeverity.MODERATE,
        bounding_box_norm=(0.3, 0.4, 0.6, 0.5),
        estimated_area_cm2=15.0,
        depth_estimate_mm=0.8,
        confidence=0.96,
        description="New scratch on driver door",
    )

    post = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.POST_PICKUP,
        mileage_km=12502.0,
        anomalies=[new_scratch],
        overall_condition_score=93.0,
    )

    comp = manager.compare_snapshots(pre, post)
    assert comp["verdict"] == LiabilityVerdict.NEW_DAMAGE_DETECTED
    assert comp["new_damage_count"] == 1
    assert comp["score_delta"] == -7.0
    assert comp["new_anomalies"][0]["anomaly_id"] == "SCR-01"
