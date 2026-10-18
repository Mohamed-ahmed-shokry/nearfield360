"""Unit tests for spatiotemporal alignment and latency extrapolation."""

import math

from nearfield360.coop.alignment import (
    Covariance2D,
    Pose2D,
    SpatiotemporalAligner,
)
from nearfield360.coop.models import (
    CPMPerceivedObject,
    PerceivedObjectClass,
)


def test_rpu_to_global_transform() -> None:
    # RPU at (10, 20) with yaw = 0 deg
    pose_zero = Pose2D(x_m=10.0, y_m=20.0, yaw_deg=0.0)
    gx, gy = SpatiotemporalAligner.rpu_to_global(5.0, 2.0, pose_zero)
    assert math.isclose(gx, 15.0, abs_tol=1e-5)
    assert math.isclose(gy, 22.0, abs_tol=1e-5)

    # RPU at (10, 20) with yaw = 90 deg
    pose_90 = Pose2D(x_m=10.0, y_m=20.0, yaw_deg=90.0)
    gx90, gy90 = SpatiotemporalAligner.rpu_to_global(5.0, 0.0, pose_90)
    assert math.isclose(gx90, 10.0, abs_tol=1e-5)
    assert math.isclose(gy90, 25.0, abs_tol=1e-5)


def test_global_to_vehicle_transform() -> None:
    # Vehicle at (10, 10) with yaw = 0 deg
    veh_zero = Pose2D(x_m=10.0, y_m=10.0, yaw_deg=0.0)
    vx, vy = SpatiotemporalAligner.global_to_vehicle(15.0, 12.0, veh_zero)
    assert math.isclose(vx, 5.0, abs_tol=1e-5)
    assert math.isclose(vy, 2.0, abs_tol=1e-5)

    # Vehicle at (10, 10) facing +Y (yaw = 90 deg)
    veh_90 = Pose2D(x_m=10.0, y_m=10.0, yaw_deg=90.0)
    # Point at (10, 15) is 5m directly ahead of vehicle
    vx90, vy90 = SpatiotemporalAligner.global_to_vehicle(10.0, 15.0, veh_90)
    assert math.isclose(vx90, 5.0, abs_tol=1e-5)
    assert math.isclose(vy90, 0.0, abs_tol=1e-5)


def test_rotate_velocity_and_covariance() -> None:
    # Rotate (2, 0) by 90 deg -> (0, 2)
    rx, ry = SpatiotemporalAligner.rotate_velocity(2.0, 0.0, 90.0)
    assert math.isclose(rx, 0.0, abs_tol=1e-5)
    assert math.isclose(ry, 2.0, abs_tol=1e-5)

    # Rotate diagonal covariance (1.0, 0.5, 0.0) by 90 deg -> (0.5, 1.0, 0.0)
    cov = Covariance2D(xx=1.0, yy=0.5, xy=0.0)
    rot_cov = SpatiotemporalAligner.rotate_covariance(cov, 90.0)
    assert math.isclose(rot_cov.xx, 0.5, abs_tol=1e-4)
    assert math.isclose(rot_cov.yy, 1.0, abs_tol=1e-4)
    assert math.isclose(rot_cov.xy, 0.0, abs_tol=1e-4)


def test_extrapolate_cpm_object_motion() -> None:
    obj = CPMPerceivedObject(
        object_id="CAR-10",
        station_id="RPU-01",
        timestamp_s=10.0,
        category=PerceivedObjectClass.VEHICLE,
        x_m=10.0,
        y_m=10.0,
        vx_mps=2.0,
        vy_mps=-1.0,
        cov_xx=0.20,
        cov_yy=0.20,
        confidence=0.90,
    )

    # Extrapolate by dt = 0.2s -> x = 10 + 2*0.2 = 10.4, y = 10 - 1*0.2 = 9.8
    extrap = SpatiotemporalAligner.extrapolate_cpm_object(obj, target_time_s=10.2)
    assert math.isclose(extrap.x_m, 10.4, abs_tol=1e-5)
    assert math.isclose(extrap.y_m, 9.8, abs_tol=1e-5)
    assert extrap.cov_xx > obj.cov_xx
    assert extrap.cov_yy > obj.cov_yy
    assert extrap.timestamp_s == 10.2

    # Zero or negative dt
    same = SpatiotemporalAligner.extrapolate_cpm_object(obj, target_time_s=10.0)
    assert same == obj


def test_extrapolate_cpm_object_excessive_latency() -> None:
    obj = CPMPerceivedObject(
        object_id="CAR-20",
        station_id="RPU-01",
        timestamp_s=10.0,
        x_m=5.0,
        y_m=5.0,
        vx_mps=1.0,
        confidence=0.90,
    )

    # 1.5s delay exceeds max_latency_s=0.50s
    stale = SpatiotemporalAligner.extrapolate_cpm_object(
        obj, target_time_s=11.5, max_latency_s=0.50
    )
    # Effective dt capped to 0.5s -> x = 5.0 + 1.0*0.5 = 5.5
    assert math.isclose(stale.x_m, 5.5, abs_tol=1e-5)
    assert stale.confidence < obj.confidence


def test_transform_cpm_object_to_vehicle() -> None:
    obj = CPMPerceivedObject(
        object_id="CAR-30",
        station_id="RPU-01",
        timestamp_s=20.0,
        x_m=15.0,
        y_m=10.0,
        vx_mps=0.0,
        vy_mps=2.0,
    )
    veh_pose = Pose2D(x_m=10.0, y_m=10.0, yaw_deg=90.0)

    # In vehicle frame (facing +Y):
    # Global point (15, 10) is 5m to vehicle's right (+X in global is -Y in vehicle body)
    obj_veh = SpatiotemporalAligner.transform_cpm_object_to_vehicle(obj, veh_pose)
    assert math.isclose(obj_veh.x_m, 0.0, abs_tol=1e-5)
    assert math.isclose(obj_veh.y_m, -5.0, abs_tol=1e-5)
    # Global velocity (0, 2) is aligned with vehicle heading -> forward velocity in body frame
    assert math.isclose(obj_veh.vx_mps, 2.0, abs_tol=1e-5)
    assert math.isclose(obj_veh.vy_mps, 0.0, abs_tol=1e-5)
