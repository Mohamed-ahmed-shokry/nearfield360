"""Tests for tactical 1280x720 vehicle condition inspection HUD dashboard visualizer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

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
from nearfield360.inspection.viz import render_inspection_dashboard


def _create_sample_inspection_report() -> InspectionReport:
    """Helper creating representative InspectionReport fixture."""
    ano1 = OpticalAnomaly(
        anomaly_id="ANO-001",
        camera_id="MVL",
        surface_location="driver_door",
        damage_type=DamageType.SCRATCH,
        severity=DamageSeverity.MINOR,
        bounding_box_norm=(0.3, 0.4, 0.5, 0.6),
        estimated_area_cm2=8.5,
        depth_estimate_mm=0.6,
        confidence=0.95,
        description="Minor scratch on driver door",
    )
    ano2 = OpticalAnomaly(
        anomaly_id="ANO-002",
        camera_id="FV",
        surface_location="front_bumper",
        damage_type=DamageType.PAINT_CHIP,
        severity=DamageSeverity.NEGLIGIBLE,
        bounding_box_norm=(0.4, 0.2, 0.6, 0.3),
        estimated_area_cm2=2.1,
        depth_estimate_mm=0.2,
        confidence=0.91,
        description="Small paint chip on front bumper",
    )

    tires = [
        TireTelemetry(
            position=TirePosition.FRONT_LEFT,
            inner_tread_depth_mm=5.5,
            center_tread_depth_mm=5.6,
            outer_tread_depth_mm=5.4,
            mean_tread_depth_mm=5.5,
            pressure_bar=2.41,
            temperature_c=21.0,
            status=TireWearStatus.GOOD,
            uneven_wear_detected=False,
        ),
        TireTelemetry(
            position=TirePosition.FRONT_RIGHT,
            inner_tread_depth_mm=5.4,
            center_tread_depth_mm=5.5,
            outer_tread_depth_mm=5.3,
            mean_tread_depth_mm=5.4,
            pressure_bar=2.39,
            temperature_c=21.2,
            status=TireWearStatus.GOOD,
            uneven_wear_detected=False,
        ),
        TireTelemetry(
            position=TirePosition.REAR_LEFT,
            inner_tread_depth_mm=4.8,
            center_tread_depth_mm=4.9,
            outer_tread_depth_mm=4.7,
            mean_tread_depth_mm=4.8,
            pressure_bar=2.42,
            temperature_c=20.8,
            status=TireWearStatus.GOOD,
            uneven_wear_detected=False,
        ),
        TireTelemetry(
            position=TirePosition.REAR_RIGHT,
            inner_tread_depth_mm=4.7,
            center_tread_depth_mm=4.8,
            outer_tread_depth_mm=4.6,
            mean_tread_depth_mm=4.7,
            pressure_bar=2.40,
            temperature_c=20.9,
            status=TireWearStatus.GOOD,
            uneven_wear_detected=False,
        ),
    ]

    glass = GlassInspectionReport(
        windshield_clear=True,
        chip_count=1,
        optical_transmittance_pct=86.0,
        ece_r43_compliant=True,
    )

    snapshot = VehicleConditionSnapshot(
        vin="WP0ZZZ99ZTS392001",
        inspection_phase=InspectionPhase.PRE_DROPOFF,
        mileage_km=14200.0,
        anomalies=[ano1, ano2],
        tires=tires,
        glass=glass,
        overall_condition_score=97.0,
    )

    cert = LiabilityCertificate(
        certificate_id="CERT-20261115-A1B2C3D4",
        vin=snapshot.vin,
        session_id="SESS-VALET-99",
        phase=snapshot.inspection_phase,
        payload_sha256="a" * 64,
        signature_hmac="b" * 64,
        issuer_id="facility-gate-01",
        operator_id="avp-operator-alice",
        verdict=LiabilityVerdict.NO_NEW_DAMAGE,
    )

    return InspectionReport(
        vehicle_id="EV-101",
        facility_id="FAC-DOWNTOWN-01",
        bay_id="INSPECT-BAY-A",
        phase=InspectionPhase.PRE_DROPOFF,
        snapshot=snapshot,
        certificate=cert,
    )


def test_render_inspection_dashboard_dimensions_and_dtype() -> None:
    rep = _create_sample_inspection_report()
    canvas = render_inspection_dashboard(rep)

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8
    assert np.any(canvas > 0)


def test_render_inspection_dashboard_file_saving(tmp_path: Path) -> None:
    rep = _create_sample_inspection_report()
    out_file = tmp_path / "inspection_hud.png"

    canvas = render_inspection_dashboard(rep, output_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000

    loaded = cv2.imread(str(out_file))
    assert loaded is not None
    assert loaded.shape == (720, 1280, 3)
    assert np.array_equal(canvas, loaded)


def test_render_inspection_dashboard_without_certificate() -> None:
    rep = _create_sample_inspection_report()
    rep_no_cert = InspectionReport(
        vehicle_id=rep.vehicle_id,
        facility_id=rep.facility_id,
        bay_id=rep.bay_id,
        phase=rep.phase,
        snapshot=rep.snapshot,
        certificate=None,
    )

    canvas = render_inspection_dashboard(rep_no_cert)
    assert canvas.shape == (720, 1280, 3)
    assert np.any(canvas > 0)
