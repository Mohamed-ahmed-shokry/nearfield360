"""Unit tests for line-of-sight raycasting, pillar shadows, and blind-spot hazard assessor."""

from nearfield360.config import CoopConfig
from nearfield360.coop.alignment import Pose2D
from nearfield360.coop.blind_spot import (
    BlindSpotAssessor,
    compute_pillar_shadow_polygon,
    is_occluded_from_observer,
    segment_intersects_circle,
)
from nearfield360.coop.models import (
    BlindSpotRiskLevel,
    CooperativeTrack,
    CoopTrackSource,
    PerceivedObjectClass,
    WarningAction,
)


def test_segment_intersects_circle() -> None:
    circle_center = (10.0, 10.0)
    radius = 1.0

    # Line from (5, 10) to (15, 10) passes directly through center
    assert segment_intersects_circle((5.0, 10.0), (15.0, 10.0), circle_center, radius) is True

    # Line from (5, 5) to (15, 5) misses circle (at y=10)
    assert segment_intersects_circle((5.0, 5.0), (15.0, 5.0), circle_center, radius) is False

    # Line from (0, 10) to (5, 10) terminates before circle (at x=10)
    assert segment_intersects_circle((0.0, 10.0), (5.0, 10.0), circle_center, radius) is False


def test_is_occluded_from_observer() -> None:
    pillar = (10.0, 10.0, 1.0)
    observer = (0.0, 10.0)

    # Target directly behind pillar at (20, 10)
    assert is_occluded_from_observer(observer, (20.0, 10.0), [pillar]) is True

    # Target to the side with direct line of sight at (20, 20)
    assert is_occluded_from_observer(observer, (20.0, 20.0), [pillar]) is False


def test_compute_pillar_shadow_polygon() -> None:
    pillar = (15.0, 15.0, 0.8)
    observer = (10.0, 15.0)

    shadow = compute_pillar_shadow_polygon(observer, pillar, shadow_depth_m=20.0)
    assert len(shadow) == 4
    # All shadow polygon x coordinates should be beyond the pillar (x >= 15.0)
    assert all(pt[0] >= 14.0 for pt in shadow)


def test_blind_spot_assessor_detects_approaching_hazard() -> None:
    # Pillar located at (10.0, 17.5) with radius 1.0
    pillar = (10.0, 17.5, 1.0)
    assessor = BlindSpotAssessor(pillars=[pillar])
    cfg = CoopConfig(blind_spot_warning_ttc_s=3.0)

    # Vehicle at (5, 15) driving east (+X) at 4.0 m/s (yaw=0 deg)
    veh_pose = Pose2D(x_m=5.0, y_m=15.0, yaw_deg=0.0)
    veh_speed = 4.0

    # Conflict hazard at (15.0, 20.0) driving south (-Y) at 2.0 m/s
    # Hazard is occluded by the pillar at (10, 17.5) from observer at (5, 15)
    hazard_track = CooperativeTrack(
        track_id="HAZARD-01",
        source=CoopTrackSource.INFRASTRUCTURE_ONLY,
        x_m=15.0,
        y_m=20.0,
        vx_mps=0.0,
        vy_mps=-2.0,
        category=PerceivedObjectClass.VEHICLE,
        confidence=0.92,
    )

    warnings = assessor.assess_tracks(
        vehicle_id="EGO-01",
        vehicle_pose=veh_pose,
        vehicle_speed_mps=veh_speed,
        tracks=[hazard_track],
        timestamp_s=100.0,
        config=cfg,
    )

    assert len(warnings) == 1
    w = warnings[0]
    assert w.target_vehicle_id == "EGO-01"
    assert w.hazard_object_id == "HAZARD-01"
    assert w.ttc_s > 0.0
    assert w.ttc_s <= 3.0
    assert w.risk_level in (
        BlindSpotRiskLevel.CRITICAL_BRAKE,
        BlindSpotRiskLevel.WARNING,
        BlindSpotRiskLevel.ADVISORY,
    )
    assert w.recommended_action in (
        WarningAction.EMERGENCY_BRAKE,
        WarningAction.STOP_AND_YIELD,
        WarningAction.DECELERATE,
    )


def test_blind_spot_assessor_ignores_diverging_or_unoccluded() -> None:
    pillar = (15.0, 15.0, 1.0)
    assessor = BlindSpotAssessor(pillars=[pillar])

    veh_pose = Pose2D(x_m=5.0, y_m=15.0, yaw_deg=0.0)
    veh_speed = 3.0

    # Track 1: Moving away from vehicle (diverging)
    diverging = CooperativeTrack(
        track_id="TRK-DIV",
        source=CoopTrackSource.INFRASTRUCTURE_ONLY,
        x_m=25.0,
        y_m=15.0,
        vx_mps=3.0,
        vy_mps=0.0,
    )

    # Track 2: In plain sight (not occluded)
    clear_sight = CooperativeTrack(
        track_id="TRK-CLEAR",
        source=CoopTrackSource.ONBOARD_ONLY,
        x_m=5.0,
        y_m=20.0,
        vx_mps=0.0,
        vy_mps=0.0,
    )

    warnings = assessor.assess_tracks(
        vehicle_id="EGO-01",
        vehicle_pose=veh_pose,
        vehicle_speed_mps=veh_speed,
        tracks=[diverging, clear_sight],
        timestamp_s=100.0,
    )

    assert len(warnings) == 0
