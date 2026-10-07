from __future__ import annotations

import numpy as np

from nearfield360.config import ParkingPlannerConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.planning.collision import (
    SweptFootprintEvaluator,
    point_to_polygon_distance,
)
from nearfield360.planning.kinematics import AckermannVehicle, KinematicWaypoint
from nearfield360.tracking.models import (
    TrackedObstacle,
    TrackState,
)


def test_point_to_polygon_distance() -> None:
    # A 2x2 square centered at (0, 0): [-1, 1] x [-1, 1]
    poly = [(-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)]
    # Point at (2, 0) is distance 1.0 from edge at x=1
    assert abs(point_to_polygon_distance(2.0, 0.0, poly) - 1.0) < 1e-4
    # Point at (0, 3) is distance 2.0 from edge at y=1
    assert abs(point_to_polygon_distance(0.0, 3.0, poly) - 2.0) < 1e-4


def test_swept_evaluator_clean_grid() -> None:
    config = ParkingPlannerConfig()
    vehicle = AckermannVehicle(config)
    evaluator = SweptFootprintEvaluator(vehicle, config)

    grid = BevGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    wps = [
        KinematicWaypoint(x=0.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=0.0),
        KinematicWaypoint(x=1.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=1.0),
    ]

    res = evaluator.evaluate_trajectory(wps, occupancy, grid)
    assert res.is_collision_free is True
    assert res.collision_waypoint_index is None
    assert res.min_clearance_m > 0.0


def test_swept_evaluator_occupancy_intrusion() -> None:
    config = ParkingPlannerConfig(collision_margin=0.1)
    vehicle = AckermannVehicle(config)
    evaluator = SweptFootprintEvaluator(vehicle, config)

    grid = BevGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    # Place an occupied obstacle at x=2.0, y=0.0 (in grid cells)
    col = round((2.0 - grid.x_min) / grid.resolution)
    row = round((0.0 - grid.y_min) / grid.resolution)
    occupancy[row - 2 : row + 3, col - 2 : col + 3] = 0.95

    wps = [
        KinematicWaypoint(x=0.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=0.0),
        KinematicWaypoint(x=1.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=1.0),
        KinematicWaypoint(x=2.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=2.0),
    ]

    res = evaluator.evaluate_trajectory(wps, occupancy, grid, danger_threshold=0.5)
    assert res.is_collision_free is False
    assert res.collision_waypoint_index is not None
    assert "Occupancy intrusion" in str(res.collision_reason)


def test_swept_evaluator_tracked_obstacle_collision() -> None:
    config = ParkingPlannerConfig(collision_margin=0.2)
    vehicle = AckermannVehicle(config)
    evaluator = SweptFootprintEvaluator(vehicle, config)

    grid = BevGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolution=0.1)
    occupancy = np.zeros(grid.shape, dtype=np.float64)

    # Obstacle placed right against the vehicle flank at (1.0, 1.0)
    # Vehicle half-width is 0.9, so flank is at y=0.9; obstacle at y=1.0 gives clearance 0.1m < 0.2m
    obs = TrackedObstacle(
        track_id=1,
        class_id=1,
        class_name="vehicles",
        state=TrackState.CONFIRMED,
        position=(1.0, 1.0),
        velocity=(0.0, 0.0),
        speed=0.0,
        age=10,
        hits=10,
        time_since_update=0,
    )

    wps = [
        KinematicWaypoint(x=0.0, y=0.0, heading_rad=0.0, curvature=0.0, distance_m=0.0),
    ]

    res = evaluator.evaluate_trajectory(wps, occupancy, grid, obstacles=[obs])
    assert res.is_collision_free is False
    assert "Tracked obstacle collision" in str(res.collision_reason)
