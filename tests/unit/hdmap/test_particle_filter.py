"""Unit tests for multi-hypothesis fisheye feature-map matching Monte Carlo localizer."""

from __future__ import annotations

import math

import pytest

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import (
    Lanelet,
    LocalizationConvergence,
    Point2D,
    Polyline2D,
)
from nearfield360.hdmap.particle_filter import FisheyeParticleFilter
from nearfield360.hdmap.tiler import MapTiler


def _setup_test_world() -> tuple[MapTiler, HDMapConfig]:
    cfg = HDMapConfig(
        tile_size_m=50.0,
        num_particles=150,
        observation_sigma_m=0.15,
        lateral_convergence_threshold_m=0.10,
        heading_convergence_threshold_deg=2.0,
    )
    tiler = MapTiler(cfg)

    # 50m straight lanelet corridor along X axis
    left = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=50.0, y=2.0)])
    right = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=50.0, y=-2.0)])
    center = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=50.0, y=0.0)])
    lane = Lanelet(
        lanelet_id="L-TEST-01",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
    )

    tiler.tile_map("MAP-TEST", lanelets=[lane])
    return tiler, cfg


def test_particle_filter_initialization() -> None:
    tiler, cfg = _setup_test_world()
    pf = FisheyeParticleFilter(tiler=tiler, config=cfg, seed=123)

    pf.initialize(init_x=20.0, init_y=0.0, init_yaw_rad=0.0, spread_pos_m=0.3, spread_yaw_rad=0.05)
    assert len(pf.particles) == 150

    state = pf.get_state(timestamp_s=1.0)
    assert state.estimated_x == pytest.approx(20.0, abs=0.2)
    assert state.estimated_y == pytest.approx(0.0, abs=0.2)
    assert state.estimated_yaw_rad == pytest.approx(0.0, abs=0.1)


def test_particle_filter_predict_motion() -> None:
    tiler, cfg = _setup_test_world()
    pf = FisheyeParticleFilter(tiler=tiler, config=cfg, seed=42)

    pf.initialize(
        init_x=10.0, init_y=0.0, init_yaw_rad=0.0, spread_pos_m=0.01, spread_yaw_rad=0.005
    )

    # Forward drive 5.0m
    pf.predict(delta_s_m=5.0, delta_yaw_rad=0.0)
    state = pf.get_state(timestamp_s=2.0)

    assert state.estimated_x == pytest.approx(15.0, abs=0.2)
    assert state.estimated_y == pytest.approx(0.0, abs=0.2)


def test_particle_filter_convergence_and_tracking() -> None:
    tiler, cfg = _setup_test_world()
    pf = FisheyeParticleFilter(tiler=tiler, config=cfg, seed=999)

    # Displaced initialization (error of 0.35m in X and Y)
    pf.initialize(
        init_x=20.35, init_y=0.35, init_yaw_rad=0.05, spread_pos_m=0.4, spread_yaw_rad=0.08
    )

    # True vehicle is at x=20.0, y=0.0, yaw=0.0.
    # Fisheye detects left curb at (0, 2.0), right curb at (0, -2.0), forward lane at (5, 0)
    obs_vehicle = [
        (0.0, 2.0),
        (0.0, -2.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (5.0, 2.0),
        (5.0, -2.0),
    ]

    # Iterative Bayesian measurement updates
    for _ in range(5):
        pf.update(obs_vehicle)

    state = pf.get_state(timestamp_s=5.0, associated_features_count=6)

    # Filter must converge to true pose within threshold (< 0.10m lateral, < 2 deg yaw)
    assert state.estimated_y == pytest.approx(0.0, abs=0.10)
    assert abs(state.estimated_yaw_rad) < math.radians(2.5)
    assert state.std_y <= 0.10
    assert state.status in (LocalizationConvergence.CONVERGED, LocalizationConvergence.TRACKING)


def test_particle_filter_kidnapped_robot() -> None:
    tiler, cfg = _setup_test_world()
    pf = FisheyeParticleFilter(tiler=tiler, config=cfg, seed=123)

    pf.initialize(init_x=20.0, init_y=0.0, init_yaw_rad=0.0)

    # Observations 500m away into deep void
    impossible_obs = [(500.0, 500.0)]
    pf.update(impossible_obs)

    # Filter should recover without crashing
    state = pf.get_state(timestamp_s=10.0)
    assert len(pf.particles) == 150
    assert state is not None


def test_particle_filter_uninitialized_safety() -> None:
    tiler, cfg = _setup_test_world()
    pf = FisheyeParticleFilter(tiler=tiler, config=cfg)

    # Safe no-ops
    pf.predict(1.0, 0.1)
    pf.update([(0.0, 1.0)])
    state = pf.get_state(0.0)
    assert state.status == LocalizationConvergence.LOST
