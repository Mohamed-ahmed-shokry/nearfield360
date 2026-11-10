"""Unit tests for ODD domain models, classification enums, and reports."""

import json

import pytest
from pydantic import ValidationError

from nearfield360.odd.models import (
    IlluminationCondition,
    ODDEvaluationResult,
    ODDMetrics,
    ODDReport,
    ODDState,
    PrecipitationCondition,
    SensorHealthStatus,
    SurfaceCondition,
    odd_state_severity,
)


def test_odd_state_severity_ordering() -> None:
    assert odd_state_severity(ODDState.NOMINAL) == 0
    assert odd_state_severity(ODDState.ADVISORY_DEGRADED) == 1
    assert odd_state_severity(ODDState.CRITICAL_RESTRICTED) == 2
    assert odd_state_severity(ODDState.ODD_BREACH_MRM) == 3


def test_sensor_health_status_bounds() -> None:
    status = SensorHealthStatus(
        sensor_id="FV",
        is_healthy=True,
        occlusion_ratio=0.15,
        lens_dirt_score=0.20,
        snr_db=32.5,
    )
    assert status.sensor_id == "FV"
    assert status.occlusion_ratio == 0.15

    with pytest.raises(ValidationError):
        SensorHealthStatus(sensor_id="FV", occlusion_ratio=1.5)

    with pytest.raises(ValidationError):
        SensorHealthStatus(sensor_id="FV", snr_db=-5.0)


def test_odd_metrics_defaults_and_validation() -> None:
    metrics = ODDMetrics()
    assert metrics.ambient_lux == 500.0
    assert metrics.glare_ratio == 0.05
    assert metrics.friction_coeff == 0.85

    with pytest.raises(ValidationError):
        ODDMetrics(glare_ratio=1.5)

    with pytest.raises(ValidationError):
        ODDMetrics(ambient_lux=-10.0)


def test_odd_evaluation_result_serialization() -> None:
    res = ODDEvaluationResult(
        current_state=ODDState.ADVISORY_DEGRADED,
        prior_state=ODDState.NOMINAL,
        illumination=IlluminationCondition.TWILIGHT,
        precipitation=PrecipitationCondition.LIGHT_RAIN,
        surface=SurfaceCondition.WET,
        active_limiters=["surface_wet", "low_lux"],
        governor_speed_limit_mps=2.5,
        safety_headway_multiplier=1.5,
        mrm_required=False,
        recommendations=["Increase follow distance", "Reduce max cruise speed"],
        metrics=ODDMetrics(ambient_lux=15.0, friction_coeff=0.55),
        timestamp_s=100.5,
    )
    data = res.model_dump()
    assert data["current_state"] == "advisory_degraded"
    assert data["governor_speed_limit_mps"] == 2.5
    assert "surface_wet" in data["active_limiters"]

    # Verify JSON round-trip
    serialized = res.model_dump_json()
    reloaded = ODDEvaluationResult.model_validate_json(serialized)
    assert reloaded == res


def test_odd_report_serialization() -> None:
    eval_res = ODDEvaluationResult()
    report = ODDReport(
        report_id="ODD-REP-001",
        vehicle_id="EGO-42",
        timestamp_iso="2026-11-09T16:00:00Z",
        latest_evaluation=eval_res,
        sensor_statuses={"FV": SensorHealthStatus(sensor_id="FV", is_healthy=True)},
        history=[eval_res],
        total_samples=1,
        mrm_triggered_count=0,
    )
    raw_json = report.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["report_id"] == "ODD-REP-001"
    assert parsed["vehicle_id"] == "EGO-42"
    assert "FV" in parsed["sensor_statuses"]
