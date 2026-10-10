"""Unit tests for PoseEstimator EKF and LocalizationSimulator."""

from __future__ import annotations

import math

import pytest

from nearfield360.config import MappingConfig
from nearfield360.mapping.builder import build_benchmark_garage
from nearfield360.mapping.localization import (
    LocalizationSimulator,
    PoseEstimator,
)
from nearfield360.mapping.models import LandmarkObservation
from nearfield360.mapping.router import GlobalRouter


def test_pose_estimator_predict_odometry_straight() -> None:
    garage = build_benchmark_garage()
    estimator = PoseEstimator(garage, initial_pose=(0.0, 0.0, 0.0))

    initial_uncertainty = estimator.current_pose().position_uncertainty_m

    # Drive forward 5.0m with zero steering
    pose = estimator.predict_odometry(delta_s=5.0, steering_angle_rad=0.0)

    assert pose.x == pytest.approx(5.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading_rad == pytest.approx(0.0)

    # Odometry prediction without updates must increase uncertainty
    assert pose.position_uncertainty_m > initial_uncertainty


def test_pose_estimator_predict_odometry_curve() -> None:
    garage = build_benchmark_garage()
    estimator = PoseEstimator(garage, initial_pose=(0.0, 0.0, 0.0))

    # Steer 0.2 rad over 2.0m
    pose = estimator.predict_odometry(delta_s=2.0, steering_angle_rad=0.2)

    assert pose.x > 0.0
    assert pose.y > 0.0
    assert pose.heading_rad > 0.0


def test_pose_estimator_update_landmark_decreases_uncertainty() -> None:
    garage = build_benchmark_garage()
    slot = garage.slots[0]
    # Set estimator near the slot with artificial prior uncertainty
    est_pose = (slot.center[0] - 3.0, slot.center[1], 0.0)
    estimator = PoseEstimator(garage, initial_pose=est_pose)

    prior_unc = estimator.current_pose().position_uncertainty_m

    # Create observation in vehicle frame
    # True relative slot position is (3.0, 0.0) with slot heading
    obs = LandmarkObservation(
        slot_id=slot.slot_id,
        observed_center=(3.05, 0.02),
        observed_heading=slot.heading_rad,
        confidence=0.95,
    )

    applied = estimator.update_landmark(obs)
    assert applied is True

    post_unc = estimator.current_pose().position_uncertainty_m
    # Kalman measurement update must contract covariance
    assert post_unc < prior_unc


def test_pose_estimator_rejects_outlier_landmark() -> None:
    garage = build_benchmark_garage()
    slot = garage.slots[0]
    estimator = PoseEstimator(garage, initial_pose=(0.0, 0.0, 0.0))

    # Wild outlier observation 50 meters away
    outlier = LandmarkObservation(
        slot_id=slot.slot_id,
        observed_center=(50.0, 50.0),
        observed_heading=0.0,
        confidence=1.0,
    )

    applied = estimator.update_landmark(outlier)
    assert applied is False


def test_pose_estimator_spatial_association() -> None:
    garage = build_benchmark_garage()
    slot = garage.slots[0]
    # Vehicle placed 2m south of slot center, heading North (pi/2)
    estimator = PoseEstimator(
        garage, initial_pose=(slot.center[0], slot.center[1] - 2.0, math.pi / 2.0)
    )

    # Observation with arbitrary slot_id within association gate
    # In vehicle frame: X is forward, Y is left. So body_x = 2.0 (forward), body_y = 0.0
    obs = LandmarkObservation(
        slot_id="unknown_candidate",
        observed_center=(2.0, 0.0),
        observed_heading=slot.heading_rad - math.pi / 2.0,
        confidence=0.8,
    )

    applied = estimator.update_landmark(obs)
    assert applied is True


def test_localization_simulator_trajectory() -> None:
    garage = build_benchmark_garage(aisle_count=2, slots_per_aisle=4)
    router = GlobalRouter(garage)
    route = router.plan(
        start_pose=(0.0, 0.0, 0.0),
        target_slot_id="bay_a0_s0",
    )

    wp_coords = [(w.x, w.y, w.heading_rad) for w in route.waypoints]
    config = MappingConfig(
        odometry_noise_dist=0.04,
        odometry_noise_yaw=0.02,
        slot_observation_noise_pos=0.10,
        slot_observation_noise_yaw=0.03,
    )

    sim = LocalizationSimulator(
        facility=garage,
        config=config,
        sensor_range_m=15.0,
        observation_interval_steps=2,
        seed=123,
    )

    report, records = sim.simulate(wp_coords)

    assert report.step_count == len(records)
    assert report.trajectory_length_m > 10.0
    assert report.total_landmark_updates > 0
    # Average position error should remain tight with landmark corrections
    assert report.mean_position_error_m < 0.35
    assert report.max_position_uncertainty_m < 1.0


def test_localization_simulator_rejects_empty_waypoints() -> None:
    garage = build_benchmark_garage()
    sim = LocalizationSimulator(facility=garage)

    with pytest.raises(ValueError, match="at least 2 points"):
        sim.simulate([(0.0, 0.0, 0.0)])
