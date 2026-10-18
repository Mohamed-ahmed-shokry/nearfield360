"""Unit tests for tactical BEV visualizer and HUD rendering."""

from pathlib import Path

import cv2
import numpy as np

from nearfield360.coop.alignment import Pose2D
from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    BlindSpotWarning,
    CooperativeTrack,
    CoopTelemetryReport,
    CoopTrackSource,
    PerceivedObjectClass,
    WarningAction,
)
from nearfield360.coop.rpu import RPUSensorManager
from nearfield360.coop.viz import render_coop_dashboard


def test_render_coop_dashboard_shape_and_type() -> None:
    network = RPUSensorManager.create_default_demonstrator_network()
    rpus = network.list_stations()

    veh_pose = Pose2D(x_m=12.0, y_m=15.0, yaw_deg=0.0)
    tracks = [
        CooperativeTrack(
            track_id="TRK-01",
            source=CoopTrackSource.FUSED_COOPERATIVE,
            x_m=18.0,
            y_m=15.0,
            vx_mps=2.0,
            category=PerceivedObjectClass.VEHICLE,
        ),
        CooperativeTrack(
            track_id="TRK-02",
            source=CoopTrackSource.INFRASTRUCTURE_ONLY,
            x_m=22.0,
            y_m=30.0,
            vy_mps=-1.5,
            category=PerceivedObjectClass.PEDESTRIAN,
        ),
    ]
    warning = BlindSpotWarning(
        target_vehicle_id="EGO-01",
        hazard_object_id="TRK-02",
        conflict_point_m=(20.0, 15.0),
        ttc_s=1.8,
        risk_level=BlindSpotRiskLevel.WARNING,
        recommended_action=WarningAction.STOP_AND_YIELD,
        timestamp_s=100.0,
    )

    canvas = render_coop_dashboard(
        vehicle_pose=veh_pose,
        vehicle_speed_mps=3.5,
        rpus=rpus,
        tracks=tracks,
        warnings=[warning],
    )

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8

    # Left panel (BEV) has non-background pixels
    left_panel = canvas[:, :800]
    assert np.any(left_panel != 20)

    # Right panel (HUD) has non-background pixels
    right_panel = canvas[:, 800:]
    assert np.any(right_panel != 28)


def test_render_coop_dashboard_exports_valid_png(tmp_path: Path) -> None:
    network = RPUSensorManager.create_default_demonstrator_network()
    rpus = network.list_stations()
    output_png = tmp_path / "coop" / "dashboard.png"

    veh_pose = Pose2D(x_m=10.0, y_m=15.0, yaw_deg=45.0)
    warning = BlindSpotWarning(
        target_vehicle_id="EGO-01",
        hazard_object_id="HAZARD-01",
        conflict_point_m=(15.0, 15.0),
        ttc_s=0.9,
        risk_level=BlindSpotRiskLevel.CRITICAL_BRAKE,
        recommended_action=WarningAction.EMERGENCY_BRAKE,
        timestamp_s=50.0,
    )
    telemetry = CoopTelemetryReport(
        active_rpus=3,
        cpm_message_count=240,
        onboard_track_count=2,
        fused_track_count=4,
        blind_spot_hazard_count=1,
        warnings=[warning],
        mean_fusion_latency_ms=2.45,
        timestamp_s=50.0,
    )

    canvas = render_coop_dashboard(
        vehicle_pose=veh_pose,
        vehicle_speed_mps=4.2,
        rpus=rpus,
        tracks=[],
        warnings=[warning],
        telemetry=telemetry,
        output_path=output_png,
    )

    assert output_png.is_file()
    assert output_png.stat().st_size > 15000

    # Read back and verify dimensions
    loaded = cv2.imread(str(output_png))
    assert loaded is not None
    assert loaded.shape == canvas.shape


def test_render_coop_dashboard_nominal_no_warnings(tmp_path: Path) -> None:
    network = RPUSensorManager.create_default_demonstrator_network()
    rpus = network.list_stations()
    output_png = tmp_path / "nominal.png"

    veh_pose = Pose2D(x_m=5.0, y_m=15.0, yaw_deg=0.0)
    render_coop_dashboard(
        vehicle_pose=veh_pose,
        vehicle_speed_mps=0.0,
        rpus=rpus,
        tracks=[],
        warnings=[],
        output_path=output_png,
    )

    assert output_png.is_file()
    assert output_png.stat().st_size > 10000
