"""Spatial metric map tiler and fast radius spatial query engine for HD vector maps."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    MapTile,
    ObstacleZone,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
)


def _polyline_bbox(polyline: Polyline2D) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box (min_x, min_y, max_x, max_y) of a polyline."""
    xs = [p.x for p in polyline.points]
    ys = [p.y for p in polyline.points]
    return (min(xs), min(ys), max(xs), max(ys))


def _boxes_overlap(
    b1: tuple[float, float, float, float],
    b2: tuple[float, float, float, float],
) -> bool:
    """Check whether two 2D bounding boxes overlap."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


class MapTiler:
    """Partitions continuous vector features into metric grid tiles for spatial queries."""

    def __init__(self, config: HDMapConfig | None = None) -> None:
        self.config = config or HDMapConfig()
        self.tile_size = self.config.tile_size_m
        self._tiles: dict[str, MapTile] = {}

    def tile_map(
        self,
        map_id: str,
        lanelets: Sequence[Lanelet] = (),
        parking_spaces: Sequence[ParkingSpaceFeature] = (),
        obstacles: Sequence[ObstacleZone] = (),
        facility_id: str = "FAC-AVP-01",
        level_id: str = "L1",
    ) -> HDVectorMap:
        """Partition input features into metric spatial tiles and return an HDVectorMap."""
        self._tiles.clear()

        # Find global bounding box
        all_polys: list[Polyline2D] = []
        for lane in lanelets:
            all_polys.extend([lane.left_boundary, lane.right_boundary, lane.centerline])
        all_polys.extend(space.boundary for space in parking_spaces)
        all_polys.extend(obs.boundary for obs in obstacles)

        if not all_polys:
            # Empty map with default single tile
            tile = MapTile(
                tile_id="TILE-0-0",
                x_min=0.0,
                y_min=0.0,
                x_max=self.tile_size,
                y_max=self.tile_size,
            )
            self._tiles[tile.tile_id] = tile
            return HDVectorMap(
                map_id=map_id,
                facility_id=facility_id,
                level_id=level_id,
                tiles=[tile],
            )

        min_x = min(min(p.x for p in poly.points) for poly in all_polys)
        min_y = min(min(p.y for p in poly.points) for poly in all_polys)
        max_x = max(max(p.x for p in poly.points) for poly in all_polys)
        max_y = max(max(p.y for p in poly.points) for poly in all_polys)

        # Snap to tile boundaries
        grid_x_min = math.floor(min_x / self.tile_size) * self.tile_size
        grid_y_min = math.floor(min_y / self.tile_size) * self.tile_size
        grid_x_max = math.ceil(max_x / self.tile_size) * self.tile_size
        grid_y_max = math.ceil(max_y / self.tile_size) * self.tile_size

        if grid_x_max <= grid_x_min:
            grid_x_max = grid_x_min + self.tile_size
        if grid_y_max <= grid_y_min:
            grid_y_max = grid_y_min + self.tile_size

        tiles_dict: dict[tuple[int, int], MapTile] = {}

        curr_x = grid_x_min
        col = round(grid_x_min / self.tile_size)
        while curr_x < grid_x_max:
            curr_y = grid_y_min
            row = round(grid_y_min / self.tile_size)
            while curr_y < grid_y_max:
                tile_id = f"TILE_{col}_{row}"
                tile = MapTile(
                    tile_id=tile_id,
                    x_min=curr_x,
                    y_min=curr_y,
                    x_max=curr_x + self.tile_size,
                    y_max=curr_y + self.tile_size,
                )
                tiles_dict[(col, row)] = tile
                curr_y += self.tile_size
                row += 1
            curr_x += self.tile_size
            col += 1

        # Distribute features to overlapping tiles
        for lane in lanelets:
            lane_box = _polyline_bbox(lane.centerline)
            for t in tiles_dict.values():
                tile_box = (t.x_min, t.y_min, t.x_max, t.y_max)
                if _boxes_overlap(lane_box, tile_box):
                    t.lanelets.append(lane)

        for space in parking_spaces:
            space_box = _polyline_bbox(space.boundary)
            for t in tiles_dict.values():
                tile_box = (t.x_min, t.y_min, t.x_max, t.y_max)
                if _boxes_overlap(space_box, tile_box):
                    t.parking_spaces.append(space)

        for obs in obstacles:
            obs_box = _polyline_bbox(obs.boundary)
            for t in tiles_dict.values():
                tile_box = (t.x_min, t.y_min, t.x_max, t.y_max)
                if _boxes_overlap(obs_box, tile_box):
                    t.obstacles.append(obs)

        # Store in internal index
        self._tiles = {t.tile_id: t for t in tiles_dict.values()}

        return HDVectorMap(
            map_id=map_id,
            facility_id=facility_id,
            level_id=level_id,
            tiles=list(self._tiles.values()),
        )

    def load_vector_map(self, vector_map: HDVectorMap) -> None:
        """Load an existing HDVectorMap into the tiler's spatial index."""
        self._tiles = {t.tile_id: t for t in vector_map.tiles}

    def get_tile(self, tile_id: str) -> MapTile | None:
        """Lookup tile by unique ID."""
        return self._tiles.get(tile_id)

    def get_tile_for_point(self, pt: Point2D) -> MapTile | None:
        """Retrieve the unique tile containing a metric 2D point."""
        for t in self._tiles.values():
            if t.contains_point(pt):
                return t
        return None

    def query_radius(self, center: Point2D, radius_m: float) -> list[MapTile]:
        """Retrieve all tiles overlapping a circular search region."""
        query_box = (
            center.x - radius_m,
            center.y - radius_m,
            center.x + radius_m,
            center.y + radius_m,
        )
        matched: list[MapTile] = []
        for t in self._tiles.values():
            tile_box = (t.x_min, t.y_min, t.x_max, t.y_max)
            if _boxes_overlap(query_box, tile_box):
                matched.append(t)
        return matched

    def query_features_in_radius(
        self,
        center: Point2D,
        radius_m: float,
    ) -> tuple[list[Lanelet], list[ParkingSpaceFeature], list[ObstacleZone]]:
        """Retrieve deduplicated map features whose distance to center is within radius_m."""
        candidate_tiles = self.query_radius(center, radius_m)

        seen_lanes: dict[str, Lanelet] = {}
        seen_spaces: dict[str, ParkingSpaceFeature] = {}
        seen_obstacles: dict[str, ObstacleZone] = {}

        for t in candidate_tiles:
            for lane in t.lanelets:
                if (
                    lane.lanelet_id not in seen_lanes
                    and lane.centerline.distance_to_point(center) <= radius_m
                ):
                    seen_lanes[lane.lanelet_id] = lane
            for space in t.parking_spaces:
                if (
                    space.slot_id not in seen_spaces
                    and space.boundary.distance_to_point(center) <= radius_m
                ):
                    seen_spaces[space.slot_id] = space
            for obs in t.obstacles:
                if (
                    obs.zone_id not in seen_obstacles
                    and obs.boundary.distance_to_point(center) <= radius_m
                ):
                    seen_obstacles[obs.zone_id] = obs

        return (
            list(seen_lanes.values()),
            list(seen_spaces.values()),
            list(seen_obstacles.values()),
        )
