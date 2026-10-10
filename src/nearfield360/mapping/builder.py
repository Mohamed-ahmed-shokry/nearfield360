"""Programmatic builder and synthetic benchmark generator for parking facilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

from nearfield360.mapping.models import (
    FacilityBounds,
    FacilityLane,
    FacilityMap,
    FacilityObstacle,
    FacilitySlot,
    FacilityWaypoint,
    LaneDirection,
    SlotBayType,
    SlotReservationStatus,
    WaypointType,
)


class FacilityBuilder:
    """Fluent builder for constructing and validating HD vector facility maps."""

    def __init__(
        self,
        bounds: FacilityBounds | None = None,
    ) -> None:
        self._bounds = bounds
        self._waypoints: dict[str, FacilityWaypoint] = {}
        self._lanes: dict[str, FacilityLane] = {}
        self._slots: dict[str, FacilitySlot] = {}
        self._obstacles: dict[str, FacilityObstacle] = {}

    def set_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> FacilityBuilder:
        """Set the spatial bounding box for the facility."""
        self._bounds = FacilityBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        return self

    def add_waypoint(
        self,
        waypoint_id: str,
        x: float,
        y: float,
        heading_rad: float | None = None,
        waypoint_type: WaypointType = WaypointType.LANE_NODE,
        connected_waypoints: Sequence[str] | None = None,
    ) -> FacilityBuilder:
        """Add a navigation waypoint to the facility."""
        self._waypoints[waypoint_id] = FacilityWaypoint(
            waypoint_id=waypoint_id,
            x=x,
            y=y,
            heading_rad=heading_rad,
            waypoint_type=waypoint_type,
            connected_waypoints=list(connected_waypoints or []),
        )
        return self

    def add_lane(
        self,
        lane_id: str,
        start_waypoint_id: str,
        end_waypoint_id: str,
        centerline_points: Sequence[tuple[float, float]],
        name: str = "",
        width_m: float = 3.5,
        speed_limit_mps: float = 2.5,
        direction: LaneDirection = LaneDirection.ONE_WAY,
    ) -> FacilityBuilder:
        """Add a driving lane corridor connecting two waypoints."""
        self._lanes[lane_id] = FacilityLane(
            lane_id=lane_id,
            name=name,
            centerline_points=list(centerline_points),
            width_m=width_m,
            speed_limit_mps=speed_limit_mps,
            direction=direction,
            start_waypoint_id=start_waypoint_id,
            end_waypoint_id=end_waypoint_id,
        )
        return self

    def add_slot(
        self,
        slot_id: str,
        slot_type: SlotBayType,
        center: tuple[float, float],
        heading_rad: float,
        width_m: float,
        length_m: float,
        access_lane_id: str,
        access_waypoint_id: str,
        status: SlotReservationStatus = SlotReservationStatus.VACANT,
        corners: Sequence[tuple[float, float]] | None = None,
    ) -> FacilityBuilder:
        """Add a parking slot with 4 corners computed from center, dimensions, and heading."""
        if corners is None:
            # Compute oriented rectangle corners from center, width, and length along heading
            cos_h = math.cos(heading_rad)
            sin_h = math.sin(heading_rad)
            dx_fwd = (length_m / 2.0) * cos_h
            dy_fwd = (length_m / 2.0) * sin_h
            dx_lat = -(width_m / 2.0) * sin_h
            dy_lat = (width_m / 2.0) * cos_h

            cx, cy = center
            computed_corners = [
                (cx - dx_fwd - dx_lat, cy - dy_fwd - dy_lat),  # Rear-right
                (cx - dx_fwd + dx_lat, cy - dy_fwd + dy_lat),  # Rear-left
                (cx + dx_fwd + dx_lat, cy + dy_fwd + dy_lat),  # Front-left
                (cx + dx_fwd - dx_lat, cy + dy_fwd - dy_lat),  # Front-right
            ]
        else:
            computed_corners = list(corners)

        self._slots[slot_id] = FacilitySlot(
            slot_id=slot_id,
            slot_type=slot_type,
            corners=computed_corners,
            center=center,
            heading_rad=heading_rad,
            width_m=width_m,
            length_m=length_m,
            access_lane_id=access_lane_id,
            access_waypoint_id=access_waypoint_id,
            status=status,
        )
        return self

    def add_obstacle(
        self,
        obstacle_id: str,
        polygon: Sequence[tuple[float, float]],
        obstacle_type: str = "wall",
    ) -> FacilityBuilder:
        """Add a structural obstacle (wall, pillar, curb) to the facility."""
        self._obstacles[obstacle_id] = FacilityObstacle(
            obstacle_id=obstacle_id,
            obstacle_type=obstacle_type,
            polygon=list(polygon),
        )
        return self

    def build(
        self,
        map_id: str,
        name: str,
        facility_type: str = "indoor_garage",
    ) -> FacilityMap:
        """Assemble the complete FacilityMap and verify geometric and topological integrity."""
        if self._bounds is None:
            # Automatically calculate bounding box with 5.0m margin from existing elements
            all_pts: list[tuple[float, float]] = [wp.xy for wp in self._waypoints.values()]
            for lane in self._lanes.values():
                all_pts.extend(lane.centerline_points)
            for s in self._slots.values():
                all_pts.extend(s.corners)
            for obs in self._obstacles.values():
                all_pts.extend(obs.polygon)

            if not all_pts:
                self._bounds = FacilityBounds(x_min=-10.0, x_max=50.0, y_min=-10.0, y_max=50.0)
            else:
                margin = 5.0
                xs = [p[0] for p in all_pts]
                ys = [p[1] for p in all_pts]
                self._bounds = FacilityBounds(
                    x_min=min(xs) - margin,
                    x_max=max(xs) + margin,
                    y_min=min(ys) - margin,
                    y_max=max(ys) + margin,
                )

        facility = FacilityMap(
            map_id=map_id,
            name=name,
            facility_type=facility_type,
            bounds=self._bounds,
            lanes=list(self._lanes.values()),
            slots=list(self._slots.values()),
            obstacles=list(self._obstacles.values()),
            waypoints=list(self._waypoints.values()),
        )

        issues = facility.validate_integrity()
        if issues:
            raise ValueError(f"Facility integrity errors ({len(issues)}): " + "; ".join(issues))

        return facility


def build_benchmark_garage(
    aisle_count: int = 2,
    slots_per_aisle: int = 4,
    occupied_slot_indices: Sequence[int] = (1, 3, 5),
) -> FacilityMap:
    """Synthesize a benchmark multi-aisle indoor parking garage with one-way circulation loops.

    Layout geometry:
    - Entry drop-off gate at (0, 0), moving along main entrance corridor (Y=0, X: 0 -> 40).
    - Aisles branch North (increasing Y) at regular X spacing.
    - Each aisle has parking bays on both sides (left & right).
    - Top corridor (Y=30) connects aisle exits back West towards exit gate (X=0, Y=30).
    - Perimeter walls enclose the garage with interior pillars between bays.
    """
    builder = FacilityBuilder()
    builder.set_bounds(x_min=-5.0, x_max=45.0, y_min=-5.0, y_max=35.0)

    # Perimeter walls
    builder.add_obstacle(
        "wall_south",
        [(-2.0, -2.0), (42.0, -2.0)],
        obstacle_type="wall",
    )
    builder.add_obstacle(
        "wall_north",
        [(-2.0, 32.0), (42.0, 32.0)],
        obstacle_type="wall",
    )
    builder.add_obstacle(
        "wall_west",
        [(-2.0, -2.0), (-2.0, 32.0)],
        obstacle_type="wall",
    )
    builder.add_obstacle(
        "wall_east",
        [(42.0, -2.0), (42.0, 32.0)],
        obstacle_type="wall",
    )

    # Waypoints: Entry gate and Main Entrance Corridor
    builder.add_waypoint(
        "wp_entry",
        x=0.0,
        y=0.0,
        heading_rad=0.0,
        waypoint_type=WaypointType.ENTRY,
        connected_waypoints=["wp_corridor_0"],
    )
    builder.add_waypoint(
        "wp_exit",
        x=0.0,
        y=30.0,
        heading_rad=math.pi,
        waypoint_type=WaypointType.EXIT,
    )

    # Main South corridor (Eastbound)
    prev_wp = "wp_entry"
    lane_idx = 1
    slot_global_idx = 0

    top_wps: list[str] = []

    for a in range(aisle_count):
        aisle_x = 15.0 + a * 15.0
        wp_corridor_bottom = f"wp_bottom_a{a}"
        builder.add_waypoint(
            wp_corridor_bottom,
            x=aisle_x,
            y=0.0,
            heading_rad=0.0,
            waypoint_type=WaypointType.INTERSECTION,
        )
        builder.add_lane(
            f"lane_bottom_{a}",
            start_waypoint_id=prev_wp,
            end_waypoint_id=wp_corridor_bottom,
            centerline_points=[(float(builder._waypoints[prev_wp].x), 0.0), (aisle_x, 0.0)],
            name=f"Main Entrance Segment {a}",
            direction=LaneDirection.ONE_WAY,
        )
        prev_wp = wp_corridor_bottom

        # Aisle (Northbound one-way)
        wp_aisle_start = f"wp_aisle_{a}_start"
        wp_aisle_mid = f"wp_aisle_{a}_mid"
        wp_aisle_top = f"wp_aisle_{a}_top"

        builder.add_waypoint(
            wp_aisle_start,
            x=aisle_x,
            y=5.0,
            heading_rad=math.pi / 2.0,
            waypoint_type=WaypointType.LANE_NODE,
        )
        builder.add_waypoint(
            wp_aisle_mid,
            x=aisle_x,
            y=15.0,
            heading_rad=math.pi / 2.0,
            waypoint_type=WaypointType.LANE_NODE,
        )
        builder.add_waypoint(
            wp_aisle_top,
            x=aisle_x,
            y=30.0,
            heading_rad=math.pi / 2.0,
            waypoint_type=WaypointType.INTERSECTION,
        )

        builder.add_lane(
            f"lane_turn_in_{a}",
            start_waypoint_id=wp_corridor_bottom,
            end_waypoint_id=wp_aisle_start,
            centerline_points=[(aisle_x, 0.0), (aisle_x, 5.0)],
            name=f"Aisle {a} Entry",
            direction=LaneDirection.ONE_WAY,
        )
        builder.add_lane(
            f"lane_aisle_{a}_lower",
            start_waypoint_id=wp_aisle_start,
            end_waypoint_id=wp_aisle_mid,
            centerline_points=[(aisle_x, 5.0), (aisle_x, 15.0)],
            name=f"Aisle {a} Lower",
            direction=LaneDirection.ONE_WAY,
        )
        builder.add_lane(
            f"lane_aisle_{a}_upper",
            start_waypoint_id=wp_aisle_mid,
            end_waypoint_id=wp_aisle_top,
            centerline_points=[(aisle_x, 15.0), (aisle_x, 30.0)],
            name=f"Aisle {a} Upper",
            direction=LaneDirection.ONE_WAY,
        )

        top_wps.append(wp_aisle_top)

        # Add slots along the aisle (left side and right side)
        for s in range(slots_per_aisle):
            slot_y = 7.0 + s * 4.5
            side = -1 if s % 2 == 0 else 1  # West or East
            slot_x = aisle_x + side * 4.5
            slot_heading = 0.0 if side > 0 else math.pi

            access_wp = wp_aisle_start if slot_y < 15.0 else wp_aisle_mid
            access_lane = f"lane_aisle_{a}_lower" if slot_y < 15.0 else f"lane_aisle_{a}_upper"

            is_occupied = slot_global_idx in occupied_slot_indices
            status = SlotReservationStatus.OCCUPIED if is_occupied else SlotReservationStatus.VACANT

            builder.add_slot(
                slot_id=f"bay_a{a}_s{s}",
                slot_type=SlotBayType.PERPENDICULAR,
                center=(slot_x, slot_y),
                heading_rad=slot_heading,
                width_m=2.5,
                length_m=5.0,
                access_lane_id=access_lane,
                access_waypoint_id=access_wp,
                status=status,
            )
            slot_global_idx += 1

    # Connect top corridor from Eastmost aisle back to Exit at (0, 30) (Westbound one-way)
    # Reverse top_wps so we go from east to west
    top_wps_east_to_west = list(reversed(top_wps))
    prev_top = top_wps_east_to_west[0]
    for twp in top_wps_east_to_west[1:]:
        builder.add_lane(
            f"lane_top_{lane_idx}",
            start_waypoint_id=prev_top,
            end_waypoint_id=twp,
            centerline_points=[
                (float(builder._waypoints[prev_top].x), 30.0),
                (float(builder._waypoints[twp].x), 30.0),
            ],
            name=f"Top Corridor segment {lane_idx}",
            direction=LaneDirection.ONE_WAY,
        )
        prev_top = twp
        lane_idx += 1

    # Final segment to exit gate
    builder.add_lane(
        "lane_to_exit",
        start_waypoint_id=prev_top,
        end_waypoint_id="wp_exit",
        centerline_points=[(float(builder._waypoints[prev_top].x), 30.0), (0.0, 30.0)],
        name="Exit Corridor",
        direction=LaneDirection.ONE_WAY,
    )

    return builder.build(
        map_id="benchmark_garage_avp",
        name="Benchmark Multi-Aisle Indoor Parking Garage",
        facility_type="indoor_garage",
    )


def build_surface_lot(
    bay_count: int = 6,
    vacant_count: int = 3,
) -> FacilityMap:
    """Synthesize a single-aisle outdoor surface parking lot with parking bays."""
    builder = FacilityBuilder()
    builder.set_bounds(x_min=-5.0, x_max=35.0, y_min=-5.0, y_max=25.0)

    # Perimeter curbs
    builder.add_obstacle("curb_south", [(-2.0, -1.0), (32.0, -1.0)], obstacle_type="curb")
    builder.add_obstacle("curb_north", [(-2.0, 21.0), (32.0, 21.0)], obstacle_type="curb")

    # Waypoints: Entry at (0, 10), Center drive at (15, 10), Exit at (30, 10)
    builder.add_waypoint(
        "wp_lot_entry", x=0.0, y=10.0, heading_rad=0.0, waypoint_type=WaypointType.ENTRY
    )
    builder.add_waypoint(
        "wp_lot_mid", x=15.0, y=10.0, heading_rad=0.0, waypoint_type=WaypointType.LANE_NODE
    )
    builder.add_waypoint(
        "wp_lot_exit", x=30.0, y=10.0, heading_rad=0.0, waypoint_type=WaypointType.EXIT
    )

    builder.add_lane(
        "lane_surface_1",
        start_waypoint_id="wp_lot_entry",
        end_waypoint_id="wp_lot_mid",
        centerline_points=[(0.0, 10.0), (15.0, 10.0)],
        direction=LaneDirection.TWO_WAY,
    )
    builder.add_lane(
        "lane_surface_2",
        start_waypoint_id="wp_lot_mid",
        end_waypoint_id="wp_lot_exit",
        centerline_points=[(15.0, 10.0), (30.0, 10.0)],
        direction=LaneDirection.TWO_WAY,
    )

    # Bays along North side (Y = 15) and South side (Y = 5)
    for i in range(bay_count):
        side = 1 if i % 2 == 0 else -1
        bx = 5.0 + (i // 2) * 8.0
        by = 10.0 + side * 5.0
        access_wp = "wp_lot_entry" if bx < 15.0 else "wp_lot_mid"
        access_lane = "lane_surface_1" if bx < 15.0 else "lane_surface_2"
        status = (
            SlotReservationStatus.VACANT if i < vacant_count else SlotReservationStatus.OCCUPIED
        )

        builder.add_slot(
            slot_id=f"surface_bay_{i}",
            slot_type=SlotBayType.PERPENDICULAR if side > 0 else SlotBayType.PARALLEL,
            center=(bx, by),
            heading_rad=math.pi / 2.0 if side > 0 else 0.0,
            width_m=2.5 if side > 0 else 2.2,
            length_m=5.0 if side > 0 else 6.0,
            access_lane_id=access_lane,
            access_waypoint_id=access_wp,
            status=status,
        )

    return builder.build(
        map_id="benchmark_surface_lot",
        name="Benchmark Surface Parking Lot",
        facility_type="surface_lot",
    )
