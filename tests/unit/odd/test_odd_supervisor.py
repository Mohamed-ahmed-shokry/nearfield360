"""Unit tests for ODD supervisor, actuation governance, and ASIL-D MRM interlocks."""

from unittest.mock import MagicMock

from nearfield360.config import ODDConfig
from nearfield360.odd.models import (
    ODDMetrics,
    ODDState,
    SensorHealthStatus,
)
from nearfield360.odd.supervisor import ODDSupervisor


def test_supervisor_nominal_governance() -> None:
    supervisor = ODDSupervisor(ODDConfig())
    nominal_metrics = ODDMetrics(friction_coeff=0.85, ambient_lux=600.0)

    eval_res = supervisor.update(nominal_metrics, timestamp_s=1.0)
    assert eval_res.current_state == ODDState.NOMINAL

    # Commanded 6.0 m/s should be capped at nominal speed limit (4.5 m/s)
    speed, headway, res = supervisor.govern_actuation(target_speed_mps=6.0, target_headway_m=5.0)
    assert speed == 4.5
    assert headway == 5.0
    assert res.current_state == ODDState.NOMINAL

    # Commanded 2.0 m/s is below limit, remains 2.0 m/s
    speed_low, _, _ = supervisor.govern_actuation(target_speed_mps=2.0, target_headway_m=5.0)
    assert speed_low == 2.0


def test_supervisor_advisory_governance() -> None:
    supervisor = ODDSupervisor(ODDConfig(degradation_confirm_frames=1))
    wet_metrics = ODDMetrics(friction_coeff=0.55)

    eval_res = supervisor.update(wet_metrics, timestamp_s=2.0)
    assert eval_res.current_state == ODDState.ADVISORY_DEGRADED

    # Commanded 5.0 m/s capped at 2.5 m/s; headway 5.0m scaled by 1.5x -> 7.5m
    speed, headway, _ = supervisor.govern_actuation(target_speed_mps=5.0, target_headway_m=5.0)
    assert speed == 2.5
    assert headway == 7.5


def test_supervisor_mrm_interlock_trigger() -> None:
    mock_safety = MagicMock()
    supervisor = ODDSupervisor(
        config=ODDConfig(enable_mrm_interlock=True),
        safety_supervisor=mock_safety,
    )
    ice_metrics = ODDMetrics(friction_coeff=0.10)  # black ice breach

    eval_res = supervisor.update(ice_metrics, timestamp_s=3.5)
    assert eval_res.current_state == ODDState.ODD_BREACH_MRM
    assert eval_res.mrm_required is True

    # Speed should be 0.0 m/s (safe halt)
    speed, headway, _ = supervisor.govern_actuation(target_speed_mps=3.0, target_headway_m=5.0)
    assert speed == 0.0
    assert headway == 15.0

    # Verify mock safety supervisor received emergency stop
    mock_safety.trigger_emergency_stop.assert_called_once()
    args, _ = mock_safety.trigger_emergency_stop.call_args
    assert "critical_low_friction" in args[0]
    assert args[1] == 3.5
    assert supervisor.mrm_trigger_count == 1


def test_supervisor_generate_report() -> None:
    supervisor = ODDSupervisor()
    sensors = {
        "FV": SensorHealthStatus(sensor_id="FV", occlusion_ratio=0.05),
        "RV": SensorHealthStatus(sensor_id="RV", occlusion_ratio=0.10),
    }
    metrics = ODDMetrics(ambient_lux=400.0)
    supervisor.update(metrics, sensors=sensors, timestamp_s=10.0)

    report = supervisor.generate_report(report_id="TEST-ODD-01", vehicle_id="EGO-TEST")
    assert report.report_id == "TEST-ODD-01"
    assert report.vehicle_id == "EGO-TEST"
    assert report.total_samples == 1
    assert "FV" in report.sensor_statuses
    assert "RV" in report.sensor_statuses
    assert len(report.history) == 1
