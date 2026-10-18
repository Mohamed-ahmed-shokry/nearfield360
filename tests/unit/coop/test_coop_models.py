"""Unit tests for cooperative perception domain models and schemas."""

import json

import pytest
from pydantic import ValidationError

from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    BlindSpotWarning,
    CollectivePerceptionMessage,
    CooperativeTrack,
    CoopTelemetryReport,
    CoopTrackSource,
    CPMPerceivedObject,
    CPMStationType,
    OcclusionState,
    PerceivedObjectClass,
    RPUSensorState,
    RPUSensorType,
    WarningAction,
)


def test_coop_enums_values() -> None:
    assert RPUSensorType.LIDAR == "lidar"
    assert RPUSensorType.FISHEYE_CAMERA == "fisheye_camera"
    assert CPMStationType.INFRASTRUCTURE_RPU == "infrastructure_rpu"
    assert CoopTrackSource.FUSED_COOPERATIVE == "fused_cooperative"
    assert BlindSpotRiskLevel.CRITICAL_BRAKE == "critical_brake"
    assert PerceivedObjectClass.VEHICLE == "vehicle"
    assert OcclusionState.FULLY_OCCLUDED == "fully_occluded"
    assert WarningAction.STOP_AND_YIELD == "stop_and_yield"


def test_rpu_sensor_state_valid() -> None:
    rpu = RPUSensorState(
        station_id="RPU-01",
        sensor_type=RPUSensorType.LIDAR,
        position_m=(15.0, 20.0, 3.8),
        yaw_deg=45.0,
        pitch_deg=-15.0,
        range_max_m=40.0,
        fov_azimuth_deg=180.0,
    )
    assert rpu.station_id == "RPU-01"
    assert rpu.position_m == (15.0, 20.0, 3.8)
    assert rpu.is_operational is True

    data = rpu.model_dump()
    assert data["sensor_type"] == "lidar"
    roundtrip = RPUSensorState.model_validate_json(rpu.model_dump_json())
    assert roundtrip == rpu


def test_rpu_sensor_state_validation() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        RPUSensorState(
            station_id="",
            sensor_type=RPUSensorType.RADAR,
            position_m=(0.0, 0.0, 3.0),
            yaw_deg=0.0,
        )

    with pytest.raises(ValidationError, match="greater than 0"):
        RPUSensorState(
            station_id="RPU-02",
            sensor_type=RPUSensorType.RADAR,
            position_m=(0.0, 0.0, 3.0),
            yaw_deg=0.0,
            range_max_m=-5.0,
        )

    with pytest.raises(ValidationError, match="less than or equal to 360"):
        RPUSensorState(
            station_id="RPU-02",
            sensor_type=RPUSensorType.RADAR,
            position_m=(0.0, 0.0, 3.0),
            yaw_deg=0.0,
            fov_azimuth_deg=400.0,
        )


def test_cpm_perceived_object_valid() -> None:
    obj = CPMPerceivedObject(
        object_id="OBJ-101",
        station_id="RPU-01",
        timestamp_s=100.25,
        category=PerceivedObjectClass.VEHICLE,
        x_m=12.5,
        y_m=8.2,
        vx_mps=2.0,
        vy_mps=-1.5,
        cov_xx=0.30,
        cov_yy=0.35,
        confidence=0.95,
        occlusion_state=OcclusionState.VISIBLE,
    )
    assert obj.object_id == "OBJ-101"
    assert obj.category == PerceivedObjectClass.VEHICLE
    assert obj.confidence == 0.95
    assert obj.vx_mps == 2.0


def test_cpm_message_envelope() -> None:
    obj = CPMPerceivedObject(
        object_id="OBJ-102",
        station_id="RPU-01",
        timestamp_s=100.25,
        category=PerceivedObjectClass.PEDESTRIAN,
        x_m=5.0,
        y_m=3.0,
    )
    msg = CollectivePerceptionMessage(
        station_id="RPU-01",
        sequence_number=42,
        timestamp_s=100.25,
        reference_x_m=15.0,
        reference_y_m=20.0,
        perceived_objects=[obj],
    )
    assert msg.sequence_number == 42
    assert len(msg.perceived_objects) == 1

    payload = json.loads(msg.model_dump_json())
    assert payload["station_id"] == "RPU-01"
    assert payload["perceived_objects"][0]["category"] == "pedestrian"


def test_cooperative_track_model() -> None:
    track = CooperativeTrack(
        track_id="COOP-001",
        source=CoopTrackSource.FUSED_COOPERATIVE,
        x_m=14.0,
        y_m=9.0,
        vx_mps=1.8,
        vy_mps=-1.2,
        cov_xx=0.15,
        cov_yy=0.18,
        category=PerceivedObjectClass.VEHICLE,
        confidence=0.92,
        onboard_track_id="TRK-05",
        rpu_station_id="RPU-01",
        rpu_object_id="OBJ-101",
    )
    assert track.track_id == "COOP-001"
    assert track.source == CoopTrackSource.FUSED_COOPERATIVE
    assert track.onboard_track_id == "TRK-05"


def test_blind_spot_warning_and_telemetry_report() -> None:
    warning = BlindSpotWarning(
        target_vehicle_id="AVP-V01",
        hazard_object_id="OBJ-999",
        conflict_point_m=(18.0, 12.0),
        ttc_s=1.8,
        risk_level=BlindSpotRiskLevel.CRITICAL_BRAKE,
        recommended_action=WarningAction.EMERGENCY_BRAKE,
        occlusion_source="structural_pillar_C3",
        timestamp_s=105.0,
    )
    assert warning.risk_level == BlindSpotRiskLevel.CRITICAL_BRAKE
    assert warning.ttc_s == 1.8

    report = CoopTelemetryReport(
        active_rpus=3,
        cpm_message_count=120,
        onboard_track_count=4,
        fused_track_count=6,
        blind_spot_hazard_count=1,
        warnings=[warning],
        mean_fusion_latency_ms=4.2,
        timestamp_s=105.0,
    )
    assert report.active_rpus == 3
    assert len(report.warnings) == 1
    assert report.mean_fusion_latency_ms == 4.2
