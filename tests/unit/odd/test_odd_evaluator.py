"""Unit tests for multi-factor ODD metrics evaluation."""

import numpy as np
import pytest

from nearfield360.config import ODDConfig
from nearfield360.odd.evaluator import ODDEvaluator, ODDStateMachine
from nearfield360.odd.models import (
    IlluminationCondition,
    ODDMetrics,
    ODDState,
    PrecipitationCondition,
    SensorHealthStatus,
    SurfaceCondition,
)


@pytest.fixture
def evaluator() -> ODDEvaluator:
    return ODDEvaluator(ODDConfig())


def test_estimate_frame_illumination_daylight(evaluator: ODDEvaluator) -> None:
    # 128x128 uniform bright frame
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    lux, glare = evaluator.estimate_frame_illumination(img)
    assert lux > 500.0
    assert glare == 0.0


def test_estimate_frame_illumination_glare(evaluator: ODDEvaluator) -> None:
    # Frame with 50% saturated pixels (255)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    img[:50, :, :] = 255
    lux, glare = evaluator.estimate_frame_illumination(img)
    assert glare >= 0.45


def test_estimate_frame_illumination_empty(evaluator: ODDEvaluator) -> None:
    img = np.empty((0, 0, 3), dtype=np.uint8)
    lux, glare = evaluator.estimate_frame_illumination(img)
    assert lux == 0.0
    assert glare == 0.0


def test_classify_illumination_cases(evaluator: ODDEvaluator) -> None:
    # Severe glare
    cond, state, lims = evaluator.classify_illumination(1000.0, 0.35)
    assert cond == IlluminationCondition.GLARE_DAZZLE
    assert state == ODDState.CRITICAL_RESTRICTED

    # Pitch black night
    cond, state, lims = evaluator.classify_illumination(0.05, 0.0)
    assert cond == IlluminationCondition.NIGHT
    assert state == ODDState.CRITICAL_RESTRICTED

    # Low lux night
    cond, state, lims = evaluator.classify_illumination(1.5, 0.0)
    assert cond == IlluminationCondition.NIGHT
    assert state == ODDState.ADVISORY_DEGRADED

    # Twilight
    cond, state, lims = evaluator.classify_illumination(15.0, 0.0)
    assert cond == IlluminationCondition.TWILIGHT
    assert state == ODDState.NOMINAL

    # Bright daylight
    cond, state, lims = evaluator.classify_illumination(2500.0, 0.02)
    assert cond == IlluminationCondition.DAYLIGHT
    assert state == ODDState.NOMINAL


def test_classify_precipitation_cases(evaluator: ODDEvaluator) -> None:
    # Clear
    cond, state, _ = evaluator.classify_precipitation(0.0)
    assert cond == PrecipitationCondition.CLEAR
    assert state == ODDState.NOMINAL

    # Light rain
    cond, state, _ = evaluator.classify_precipitation(4.0)
    assert cond == PrecipitationCondition.LIGHT_RAIN
    assert state == ODDState.ADVISORY_DEGRADED

    # Heavy rain
    cond, state, _ = evaluator.classify_precipitation(15.0)
    assert cond == PrecipitationCondition.HEAVY_RAIN
    assert state == ODDState.CRITICAL_RESTRICTED

    # Torrential storm breach
    cond, state, _ = evaluator.classify_precipitation(35.0)
    assert cond == PrecipitationCondition.HEAVY_RAIN
    assert state == ODDState.ODD_BREACH_MRM


def test_classify_surface_cases(evaluator: ODDEvaluator) -> None:
    # Dry asphalt
    cond, state, _ = evaluator.classify_surface(0.85)
    assert cond == SurfaceCondition.DRY
    assert state == ODDState.NOMINAL

    # Wet pavement
    cond, state, _ = evaluator.classify_surface(0.60)
    assert cond == SurfaceCondition.WET
    assert state == ODDState.ADVISORY_DEGRADED

    # Aquaplaning puddle
    cond, state, _ = evaluator.classify_surface(0.30)
    assert cond == SurfaceCondition.PUDDLE_AQUAPLANING
    assert state == ODDState.CRITICAL_RESTRICTED

    # Ice / black ice breach
    cond, state, _ = evaluator.classify_surface(0.12)
    assert cond == SurfaceCondition.ICY_SNOW
    assert state == ODDState.ODD_BREACH_MRM


def test_classify_connectivity_and_sensor_health(evaluator: ODDEvaluator) -> None:
    # Nominal connectivity
    st, _ = evaluator.classify_connectivity(45.0, 0.01)
    assert st == ODDState.NOMINAL

    # High latency
    st, _ = evaluator.classify_connectivity(200.0, 0.05)
    assert st == ODDState.CRITICAL_RESTRICTED

    # Critical dropout breach
    st, _ = evaluator.classify_connectivity(450.0, 0.40)
    assert st == ODDState.ODD_BREACH_MRM

    # Single sensor occluded
    sensors = {
        "FV": SensorHealthStatus(sensor_id="FV", occlusion_ratio=0.75),
        "RV": SensorHealthStatus(sensor_id="RV", occlusion_ratio=0.05),
    }
    st, _ = evaluator.classify_sensor_health(sensors)
    assert st == ODDState.CRITICAL_RESTRICTED

    # Multiple sensors occluded breach
    sensors["RV"] = SensorHealthStatus(sensor_id="RV", occlusion_ratio=0.80)
    st, _ = evaluator.classify_sensor_health(sensors)
    assert st == ODDState.ODD_BREACH_MRM


def test_evaluate_instantaneous_composite(evaluator: ODDEvaluator) -> None:
    metrics = ODDMetrics(
        ambient_lux=500.0,
        glare_ratio=0.01,
        friction_coeff=0.85,
        precipitation_rate_mmph=0.0,
        v2x_latency_ms=30.0,
        v2x_packet_loss_ratio=0.0,
    )
    st, lims, illum, precip, surf = evaluator.evaluate_instantaneous(metrics)
    assert st == ODDState.NOMINAL
    assert len(lims) == 0
    assert illum == IlluminationCondition.DAYLIGHT

    # Now inject wet surface -> ADVISORY_DEGRADED
    wet_metrics = metrics.model_copy(update={"friction_coeff": 0.55})
    st_wet, lims_wet, _, _, surf_wet = evaluator.evaluate_instantaneous(wet_metrics)
    assert st_wet == ODDState.ADVISORY_DEGRADED
    assert surf_wet == SurfaceCondition.WET
    assert any("wet_surface" in l for l in lims_wet)


def test_state_machine_debouncing_and_confirmation() -> None:
    sm = ODDStateMachine(ODDConfig(degradation_confirm_frames=3, restoration_confirm_frames=5))
    nominal = ODDMetrics(friction_coeff=0.85)
    degraded = ODDMetrics(friction_coeff=0.60)  # triggers advisory degraded

    # Frame 1: nominal
    res = sm.step(nominal, timestamp_s=0.1)
    assert res.current_state == ODDState.NOMINAL
    assert res.governor_speed_limit_mps == 4.5

    # Frame 2: transient glitch (1st degraded frame) -> still NOMINAL
    res = sm.step(degraded, timestamp_s=0.2)
    assert res.current_state == ODDState.NOMINAL
    assert sm.pending_count == 1

    # Frame 3: transient glitch goes away -> reset to nominal, pending count clears
    res = sm.step(nominal, timestamp_s=0.3)
    assert res.current_state == ODDState.NOMINAL
    assert sm.pending_count == 0

    # Frame 4, 5, 6: sustained degradation (3 consecutive frames) -> transitions
    sm.step(degraded, timestamp_s=0.4)
    sm.step(degraded, timestamp_s=0.5)
    res_deg = sm.step(degraded, timestamp_s=0.6)
    assert res_deg.current_state == ODDState.ADVISORY_DEGRADED
    assert res_deg.governor_speed_limit_mps == 2.5
    assert res_deg.safety_headway_multiplier == 1.5


def test_state_machine_restoration_damping() -> None:
    sm = ODDStateMachine(ODDConfig(degradation_confirm_frames=2, restoration_confirm_frames=4))
    nominal = ODDMetrics(friction_coeff=0.85)
    degraded = ODDMetrics(friction_coeff=0.60)

    # Move to degraded
    sm.step(degraded)
    sm.step(degraded)
    assert sm.current_state == ODDState.ADVISORY_DEGRADED

    # Improving to nominal needs 4 consecutive frames
    for _ in range(3):
        res = sm.step(nominal)
        assert res.current_state == ODDState.ADVISORY_DEGRADED  # not yet 4

    res = sm.step(nominal)
    assert res.current_state == ODDState.NOMINAL
    assert res.governor_speed_limit_mps == 4.5


def test_state_machine_emergency_mrm_immediate() -> None:
    sm = ODDStateMachine(ODDConfig(degradation_confirm_frames=5))
    nominal = ODDMetrics(friction_coeff=0.85)
    ice = ODDMetrics(friction_coeff=0.10)  # Critical low friction ice

    sm.step(nominal)
    assert sm.current_state == ODDState.NOMINAL

    # Immediate breach for emergency black ice in 1 frame
    res = sm.step(ice)
    assert res.current_state == ODDState.ODD_BREACH_MRM
    assert res.mrm_required is True
    assert res.governor_speed_limit_mps == 0.0

    # Test reset
    sm.reset()
    assert sm.current_state == ODDState.NOMINAL
    assert len(sm.history) == 0
