"""Unit tests for tactical ODD HUD visualizer."""

from pathlib import Path

import cv2

from nearfield360.odd.models import (
    IlluminationCondition,
    ODDEvaluationResult,
    ODDMetrics,
    ODDReport,
    ODDState,
    PrecipitationCondition,
    SensorHealthStatus,
    SurfaceCondition,
)
from nearfield360.odd.viz import render_odd_dashboard


def test_render_odd_dashboard_nominal(tmp_path: Path) -> None:
    eval_res = ODDEvaluationResult(
        current_state=ODDState.NOMINAL,
        governor_speed_limit_mps=4.5,
        safety_headway_multiplier=1.0,
        mrm_required=False,
    )
    report = ODDReport(
        report_id="ODD-REP-NOMINAL",
        vehicle_id="AVP-01",
        timestamp_iso="2026-11-09T16:00:00Z",
        latest_evaluation=eval_res,
        sensor_statuses={
            "FV": SensorHealthStatus(sensor_id="FV", is_healthy=True, occlusion_ratio=0.02),
            "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.03),
            "MVL": SensorHealthStatus(sensor_id="MVL", is_healthy=True, occlusion_ratio=0.01),
            "MVR": SensorHealthStatus(sensor_id="MVR", is_healthy=True, occlusion_ratio=0.04),
        },
        history=[eval_res] * 5,
        total_samples=5,
        mrm_triggered_count=0,
    )

    out_file = tmp_path / "odd_nominal.png"
    result_path = render_odd_dashboard(report, out_file)
    assert result_path.is_file()
    assert result_path.stat().st_size > 5000

    img = cv2.imread(str(result_path))
    assert img is not None
    assert img.shape == (720, 1280, 3)


def test_render_odd_dashboard_mrm_breach(tmp_path: Path) -> None:
    eval_res = ODDEvaluationResult(
        current_state=ODDState.ODD_BREACH_MRM,
        illumination=IlluminationCondition.NIGHT,
        precipitation=PrecipitationCondition.HEAVY_RAIN,
        surface=SurfaceCondition.ICY_SNOW,
        active_limiters=["critical_low_friction_0.10", "sensor_FV_severe_occlusion_0.80"],
        governor_speed_limit_mps=0.0,
        safety_headway_multiplier=3.0,
        mrm_required=True,
        recommendations=["Controlled deceleration to halt", "Notify facility control tower"],
        metrics=ODDMetrics(friction_coeff=0.10, precipitation_rate_mmph=30.0),
    )
    report = ODDReport(
        report_id="ODD-REP-BREACH",
        vehicle_id="AVP-02",
        timestamp_iso="2026-11-09T16:30:00Z",
        latest_evaluation=eval_res,
        sensor_statuses={
            "FV": SensorHealthStatus(sensor_id="FV", is_healthy=False, occlusion_ratio=0.85),
            "RV": SensorHealthStatus(sensor_id="RV", is_healthy=True, occlusion_ratio=0.10),
            "MVL": SensorHealthStatus(sensor_id="MVL", is_healthy=True, occlusion_ratio=0.12),
            "MVR": SensorHealthStatus(sensor_id="MVR", is_healthy=True, occlusion_ratio=0.15),
        },
        history=[
            ODDEvaluationResult(current_state=ODDState.NOMINAL),
            ODDEvaluationResult(current_state=ODDState.ADVISORY_DEGRADED),
            ODDEvaluationResult(current_state=ODDState.CRITICAL_RESTRICTED),
            eval_res,
        ],
        total_samples=4,
        mrm_triggered_count=1,
    )

    out_file = tmp_path / "odd_breach.png"
    result_path = render_odd_dashboard(report, out_file)
    assert result_path.is_file()
    img = cv2.imread(str(result_path))
    assert img is not None
    assert img.shape == (720, 1280, 3)
