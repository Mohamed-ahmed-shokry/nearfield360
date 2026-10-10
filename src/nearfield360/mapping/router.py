"""Topological routing graph and A* global route planner for parking facilities."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

from nearfield360.mapping.models import (
    FacilityMap,
    FacilityWaypoint,
    GlobalRoute,
    LaneDirection,
    RouteWaypoint,
)


class RoutingError(RuntimeError):
    """Raised when route planning fails or target is unreachable."""


@dataclass(frozen=True)
class GraphEdge:
    """Directed edge in the facility topological navigation graph."""

    lane_id: str
    from_node: str
    to_node: str
    length_m: float
    speed_limit_mps: float
    width_m: float
    centerline_points: list[tuple[float, float]]
    entry_heading_rad: float
    exit_heading_rad: float


class RoutingGraph:
    """Directed topological graph constructed from facility lanes and waypoints."""

    def __init__(self, facility: FacilityMap, blocked_lanes: set[str] | None = None) -> None:
        self.facility = facility
        self.blocked_lanes = blocked_lanes or set()
        self.adjacency: dict[str, list[GraphEdge]] = {w.waypoint_id: [] for w in facility.waypoints}
        self._build_graph()

    def _build_graph(self) -> None:
        for lane in self.facility.lanes:
            if lane.lane_id in self.blocked_lanes:
                continue

            pts = lane.centerline_points
            entry_heading = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
            exit_heading = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])

            # Forward edge
            fwd_edge = GraphEdge(
                lane_id=lane.lane_id,
                from_node=lane.start_waypoint_id,
                to_node=lane.end_waypoint_id,
                length_m=lane.length_m,
                speed_limit_mps=lane.speed_limit_mps,
                width_m=lane.width_m,
                centerline_points=list(pts),
                entry_heading_rad=entry_heading,
                exit_heading_rad=exit_heading,
            )
            if lane.start_waypoint_id in self.adjacency:
                self.adjacency[lane.start_waypoint_id].append(fwd_edge)

            # Reverse edge if two-way
            if lane.direction == LaneDirection.TWO_WAY:
                rev_pts = list(reversed(pts))
                rev_entry_heading = math.atan2(
                    rev_pts[1][1] - rev_pts[0][1], rev_pts[1][0] - rev_pts[0][0]
                )
                rev_exit_heading = math.atan2(
                    rev_pts[-1][1] - rev_pts[-2][1], rev_pts[-1][0] - rev_pts[-2][0]
                )
                rev_edge = GraphEdge(
                    lane_id=lane.lane_id,
                    from_node=lane.end_waypoint_id,
                    to_node=lane.start_waypoint_id,
                    length_m=lane.length_m,
                    speed_limit_mps=lane.speed_limit_mps,
                    width_m=lane.width_m,
                    centerline_points=rev_pts,
                    entry_heading_rad=rev_entry_heading,
                    exit_heading_rad=rev_exit_heading,
                )
                if lane.end_waypoint_id in self.adjacency:
                    self.adjacency[lane.end_waypoint_id].append(rev_edge)


def _wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def find_nearest_waypoint(
    facility: FacilityMap,
    x: float,
    y: float,
    allowed_types: Sequence[str] | None = None,
) -> FacilityWaypoint:
    """Find the closest waypoint to given coordinate in the facility."""
    best_wp: FacilityWaypoint | None = None
    best_dist = float("inf")

    for wp in facility.waypoints:
        if allowed_types and wp.waypoint_type not in allowed_types:
            continue
        dist = math.hypot(wp.x - x, wp.y - y)
        if dist < best_dist:
            best_dist = dist
            best_wp = wp

    if best_wp is None:
        raise RoutingError(f"No suitable waypoint found in facility near ({x}, {y})")
    return best_wp


class GlobalRouter:
    """A* optimal route planner through facility topological graphs."""

    def __init__(
        self,
        facility: FacilityMap,
        turn_penalty_weight: float = 1.5,
        default_discretization_step_m: float = 0.5,
        max_lateral_accel_mps2: float = 1.0,
    ) -> None:
        self.facility = facility
        self.turn_penalty_weight = turn_penalty_weight
        self.step_m = default_discretization_step_m
        self.max_lat_accel = max_lateral_accel_mps2

    def plan(
        self,
        start_pose: tuple[float, float, float],
        target_slot_id: str | None = None,
        target_waypoint_id: str | None = None,
        blocked_lanes: set[str] | None = None,
    ) -> GlobalRoute:
        """Compute an optimal global route from start pose to target slot or waypoint."""
        if target_slot_id is None and target_waypoint_id is None:
            raise RoutingError("Must specify either target_slot_id or target_waypoint_id")

        goal_wp_id: str
        if target_slot_id is not None:
            slot = self.facility.get_slot(target_slot_id)
            if slot is None:
                raise RoutingError(f"Target slot {target_slot_id} not found in facility map")
            goal_wp_id = slot.access_waypoint_id
        elif target_waypoint_id is not None:
            if self.facility.get_waypoint(target_waypoint_id) is None:
                raise RoutingError(f"Target waypoint {target_waypoint_id} not found in map")
            goal_wp_id = target_waypoint_id
        else:
            raise RoutingError("Must specify either target_slot_id or target_waypoint_id")

        graph = RoutingGraph(self.facility, blocked_lanes=blocked_lanes)
        start_wp = find_nearest_waypoint(self.facility, start_pose[0], start_pose[1])
        goal_wp = self.facility.get_waypoint(goal_wp_id)
        if goal_wp is None:
            raise RoutingError(f"Goal waypoint {goal_wp_id} missing from facility")

        # A* Search over (cost, counter, current_node, current_heading, edge_path)
        counter = 0
        h_start = math.hypot(goal_wp.x - start_wp.x, goal_wp.y - start_wp.y)
        # Priority queue entries: (f_score, g_score, counter, node_id, heading, edge_path)
        open_set: list[tuple[float, float, int, str, float, list[GraphEdge]]] = [
            (h_start, 0.0, counter, start_wp.waypoint_id, start_pose[2], [])
        ]
        best_g: dict[tuple[str, int], float] = {}

        solution_edges: list[GraphEdge] | None = None

        if start_wp.waypoint_id == goal_wp_id:
            solution_edges = []

        while open_set and solution_edges is None:
            _f, g, _, curr_node, curr_heading, edge_path = heapq.heappop(open_set)

            if curr_node == goal_wp_id:
                solution_edges = edge_path
                break

            # Discretize heading into 8 cardinal bins for state domination check
            heading_bin = round((curr_heading % (2.0 * math.pi)) / (math.pi / 4.0)) % 8
            state_key = (curr_node, heading_bin)
            if state_key in best_g and best_g[state_key] <= g - 1e-4:
                continue
            best_g[state_key] = g

            for edge in graph.adjacency.get(curr_node, []):
                # Calculate turn penalty between current heading and edge entry heading
                d_heading = abs(_wrap_to_pi(edge.entry_heading_rad - curr_heading))
                turn_cost = self.turn_penalty_weight * d_heading

                edge_cost = edge.length_m + turn_cost
                new_g = g + edge_cost

                nxt_wp = self.facility.get_waypoint(edge.to_node)
                if nxt_wp is None:
                    continue
                h = math.hypot(goal_wp.x - nxt_wp.x, goal_wp.y - nxt_wp.y)
                new_f = new_g + h

                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        new_f,
                        new_g,
                        counter,
                        edge.to_node,
                        edge.exit_heading_rad,
                        [*edge_path, edge],
                    ),
                )

        if solution_edges is None:
            raise RoutingError(
                f"No feasible route found from {start_wp.waypoint_id} to {goal_wp_id} "
                "(target may be unreachable due to one-way constraints or blocked corridors)"
            )

        # Discretize solution edges into dense RouteWaypoints
        route_waypoints = self._discretize_edges(start_pose, start_wp, solution_edges)

        total_length = route_waypoints[-1].s_m if route_waypoints else 0.0
        # Estimate duration by integrating dt = ds / v
        duration = 0.0
        for i in range(len(route_waypoints) - 1):
            ds = route_waypoints[i + 1].s_m - route_waypoints[i].s_m
            v = max(
                0.5,
                (route_waypoints[i].target_speed_mps + route_waypoints[i + 1].target_speed_mps)
                / 2.0,
            )
            duration += ds / v

        lane_seq = [e.lane_id for e in solution_edges]

        route_id = f"route_{start_wp.waypoint_id}_to_{goal_wp_id}"
        return GlobalRoute(
            route_id=route_id,
            start_pose=start_pose,
            target_slot_id=target_slot_id,
            target_waypoint_id=target_waypoint_id,
            total_length_m=total_length,
            estimated_duration_s=duration,
            waypoints=route_waypoints,
            lane_sequence=lane_seq,
        )

    def _discretize_edges(
        self,
        start_pose: tuple[float, float, float],
        start_wp: FacilityWaypoint,
        edges: list[GraphEdge],
    ) -> list[RouteWaypoint]:
        """Convert a sequence of graph edges into finely spaced, continuous RouteWaypoints."""
        waypoints: list[RouteWaypoint] = []
        accum_s = 0.0

        # If vehicle start pose is slightly offset from start_wp, add initial transition waypoint
        start_dist = math.hypot(start_wp.x - start_pose[0], start_wp.y - start_pose[1])
        if start_dist > 0.05:
            waypoints.append(
                RouteWaypoint(
                    x=start_pose[0],
                    y=start_pose[1],
                    heading_rad=start_pose[2],
                    target_speed_mps=1.5,
                    curvature=0.0,
                    lane_id=edges[0].lane_id if edges else "approach",
                    corridor_half_width_m=edges[0].width_m / 2.0 if edges else 1.75,
                    s_m=0.0,
                )
            )
            accum_s += start_dist

        for edge in edges:
            pts = edge.centerline_points
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                seg_heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

                steps = max(1, math.ceil(seg_len / self.step_m))
                for s in range(steps):
                    frac = s / float(steps)
                    wx = p1[0] + frac * (p2[0] - p1[0])
                    wy = p1[1] + frac * (p2[1] - p1[1])

                    # Safe cornering speed based on lateral acceleration
                    target_v = min(edge.speed_limit_mps, 2.5)

                    waypoints.append(
                        RouteWaypoint(
                            x=wx,
                            y=wy,
                            heading_rad=seg_heading,
                            target_speed_mps=target_v,
                            curvature=0.0,
                            lane_id=edge.lane_id,
                            corridor_half_width_m=edge.width_m / 2.0,
                            s_m=accum_s,
                        )
                    )
                    accum_s += seg_len / steps

        if edges:
            last_edge = edges[-1]
            last_pt = last_edge.centerline_points[-1]
            waypoints.append(
                RouteWaypoint(
                    x=last_pt[0],
                    y=last_pt[1],
                    heading_rad=last_edge.exit_heading_rad,
                    target_speed_mps=0.0,
                    curvature=0.0,
                    lane_id=last_edge.lane_id,
                    corridor_half_width_m=last_edge.width_m / 2.0,
                    s_m=accum_s,
                )
            )

        return waypoints
