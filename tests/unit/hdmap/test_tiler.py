"""Unit tests for spatial metric map tiler and radius spatial query engine."""

from __future__ import annotations

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import (
    Lanelet,
    ObstacleZone,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
)
from nearfield360.hdmap.tiler import MapTiler


def test_tile_empty_map() -> None:
    tiler = MapTiler(HDMapConfig(tile_size_m=50.0))
    hd_map = tiler.tile_map(map_id="MAP-EMPTY")
    assert hd_map.map_id == "MAP-EMPTY"
    assert len(hd_map.tiles) == 1
    assert hd_map.tiles[0].tile_id == "TILE-0-0"


def test_tile_map_multiple_tiles() -> None:
    tiler = MapTiler(HDMapConfig(tile_size_m=50.0))

    # Lane spanning from x=10 to x=110 (crosses tile (0, 0), (1, 0), (2, 0))
    left = Polyline2D(points=[Point2D(x=10.0, y=10.0), Point2D(x=110.0, y=10.0)])
    right = Polyline2D(points=[Point2D(x=10.0, y=6.0), Point2D(x=110.0, y=6.0)])
    center = Polyline2D(points=[Point2D(x=10.0, y=8.0), Point2D(x=110.0, y=8.0)])
    lane = Lanelet(
        lanelet_id="L-MAIN",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
    )

    # Parking slot at (25, 20) in tile (0, 0)
    slot_poly = Polyline2D(
        points=[
            Point2D(x=25.0, y=20.0),
            Point2D(x=27.5, y=20.0),
            Point2D(x=27.5, y=25.0),
            Point2D(x=25.0, y=25.0),
        ],
        is_closed=True,
    )
    space = ParkingSpaceFeature(
        slot_id="BAY-1",
        boundary=slot_poly,
        center=Point2D(x=26.25, y=22.5),
    )

    hd_map = tiler.tile_map(
        map_id="MAP-MULTI",
        lanelets=[lane],
        parking_spaces=[space],
    )

    assert len(hd_map.tiles) >= 3

    # Check tile point lookup
    tile_near = tiler.get_tile_for_point(Point2D(x=25.0, y=20.0))
    assert tile_near is not None
    assert tile_near.contains_point(Point2D(x=25.0, y=20.0)) is True

    # Check get_tile by id
    found = tiler.get_tile(tile_near.tile_id)
    assert found is not None
    assert found.tile_id == tile_near.tile_id


def test_query_radius_and_features() -> None:
    tiler = MapTiler(HDMapConfig(tile_size_m=50.0))

    center1 = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=40.0, y=0.0)])
    lane1 = Lanelet(
        lanelet_id="L1",
        left_boundary=center1,
        right_boundary=center1,
        centerline=center1,
    )

    center2 = Polyline2D(points=[Point2D(x=0.0, y=80.0), Point2D(x=40.0, y=80.0)])
    lane2 = Lanelet(
        lanelet_id="L2",
        left_boundary=center2,
        right_boundary=center2,
        centerline=center2,
    )

    # Obstacle at (10, 5)
    obs_poly = Polyline2D(
        points=[
            Point2D(x=10.0, y=5.0),
            Point2D(x=11.0, y=5.0),
            Point2D(x=11.0, y=6.0),
            Point2D(x=10.0, y=6.0),
        ],
        is_closed=True,
    )
    obs = ObstacleZone(zone_id="PIL-1", boundary=obs_poly)

    tiler.tile_map("MAP-RADIUS", lanelets=[lane1, lane2], obstacles=[obs])

    # Query centered at (10, 0) with radius 10m
    tiles = tiler.query_radius(Point2D(x=10.0, y=0.0), radius_m=10.0)
    assert len(tiles) >= 1

    lanes, _spaces, obstacles = tiler.query_features_in_radius(
        Point2D(x=10.0, y=0.0),
        radius_m=10.0,
    )
    # L1 and PIL-1 should be included; L2 (at y=80m) must be excluded
    lane_ids = [lane.lanelet_id for lane in lanes]
    assert "L1" in lane_ids
    assert "L2" not in lane_ids
    assert len(obstacles) == 1
    assert obstacles[0].zone_id == "PIL-1"


def test_load_vector_map() -> None:
    tiler1 = MapTiler(HDMapConfig(tile_size_m=50.0))
    hd_map = tiler1.tile_map("MAP-ORIG")

    tiler2 = MapTiler(HDMapConfig(tile_size_m=50.0))
    tiler2.load_vector_map(hd_map)
    assert tiler2.get_tile("TILE-0-0") is not None
