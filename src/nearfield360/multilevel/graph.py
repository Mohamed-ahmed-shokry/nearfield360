"""Multi-floor facility topological directed graph and 3D inter-floor routing engine."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import NamedTuple

from nearfield360.multilevel.models import (
    FloorLevel,
    MultiFloorRoute,
    PortalType,
    VerticalPortal,
    Waypoint3D,
)


class NodeData(NamedTuple):
    """Spatial coordinate of a node in a multi-level facility."""

    node_id: str
    floor_id: str
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class EdgeData:
    """Directed connection between two facility graph nodes."""

    target: str
    distance_m: float
    transit_time_s: float
    speed_mps: float
    portal_id: str | None = None
    portal_type: PortalType | None = None


class MultiFloorGraph:
    """Directed graph representing 3D connectivity across multiple garage floors."""

    def __init__(self) -> None:
        self._floors: dict[str, FloorLevel] = {}
        self._nodes: dict[str, NodeData] = {}
        self._adj: dict[str, list[EdgeData]] = {}
        self._portals: dict[str, VerticalPortal] = {}

    @property
    def floors(self) -> dict[str, FloorLevel]:
        """Registered garage floor levels."""
        return dict(self._floors)

    @property
    def nodes(self) -> dict[str, NodeData]:
        """Registered spatial graph nodes."""
        return dict(self._nodes)

    @property
    def portals(self) -> dict[str, VerticalPortal]:
        """Registered vertical transition portals."""
        return dict(self._portals)

    def add_floor(self, floor: FloorLevel) -> None:
        """Register a floor level in the facility graph."""
        self._floors[floor.floor_id] = floor

    def add_node(self, node_id: str, floor_id: str, x_m: float, y_m: float) -> None:
        """Register a spatial node on a specific floor level."""
        if floor_id not in self._floors:
            raise ValueError(f"Floor {floor_id} must be added before nodes can be placed on it.")
        z_m = self._floors[floor_id].elevation_m
        self._nodes[node_id] = NodeData(
            node_id=node_id, floor_id=floor_id, x_m=x_m, y_m=y_m, z_m=z_m
        )
        if node_id not in self._adj:
            self._adj[node_id] = []

    def add_intra_floor_edge(self, u: str, v: str, speed_mps: float = 3.0) -> None:
        """Connect two nodes on the same floor level with a bidirectional horizontal lane."""
        if u not in self._nodes or v not in self._nodes:
            raise KeyError(f"Both nodes ({u}, {v}) must exist in the facility graph.")
        node_u = self._nodes[u]
        node_v = self._nodes[v]
        if node_u.floor_id != node_v.floor_id:
            msg = (
                f"Intra-floor edges must connect nodes on same floor "
                f"({node_u.floor_id} != {node_v.floor_id})."
            )
            raise ValueError(msg)

        dist = math.hypot(node_v.x_m - node_u.x_m, node_v.y_m - node_u.y_m)
        transit_time = dist / max(speed_mps, 0.1)

        self._adj[u].append(
            EdgeData(target=v, distance_m=dist, transit_time_s=transit_time, speed_mps=speed_mps)
        )
        self._adj[v].append(
            EdgeData(target=u, distance_m=dist, transit_time_s=transit_time, speed_mps=speed_mps)
        )

    def add_portal(
        self,
        portal: VerticalPortal,
        entry_node: str,
        exit_node: str,
        door_dwell_s: float = 5.0,
        elevator_hoist_speed_mps: float = 1.2,
    ) -> None:
        """Connect two nodes on distinct floors across a vertical ramp or elevator."""
        if entry_node not in self._nodes or exit_node not in self._nodes:
            raise KeyError(f"Portal endpoints ({entry_node}, {exit_node}) must exist in graph.")
        u = self._nodes[entry_node]
        v = self._nodes[exit_node]

        if u.floor_id != portal.from_floor_id or v.floor_id != portal.to_floor_id:
            raise ValueError(
                f"Portal floor configuration ({portal.from_floor_id}->{portal.to_floor_id}) "
                f"mismatches node floors ({u.floor_id}->{v.floor_id})."
            )

        self._portals[portal.portal_id] = portal

        if portal.portal_type == PortalType.VEHICLE_ELEVATOR:
            dz = abs(v.z_m - u.z_m)
            transit_time = (2.0 * door_dwell_s) + (dz / max(elevator_hoist_speed_mps, 0.1))
            eff_dist = portal.length_m
            eff_speed = elevator_hoist_speed_mps
        else:
            eff_dist = portal.length_m
            eff_speed = portal.speed_limit_mps
            transit_time = eff_dist / max(eff_speed, 0.1)

        self._adj[entry_node].append(
            EdgeData(
                target=exit_node,
                distance_m=eff_dist,
                transit_time_s=transit_time,
                speed_mps=eff_speed,
                portal_id=portal.portal_id,
                portal_type=portal.portal_type,
            )
        )

        if portal.is_bidirectional:
            self._adj[exit_node].append(
                EdgeData(
                    target=entry_node,
                    distance_m=eff_dist,
                    transit_time_s=transit_time,
                    speed_mps=eff_speed,
                    portal_id=portal.portal_id,
                    portal_type=portal.portal_type,
                )
            )

    def find_route(
        self,
        origin_node: str,
        destination_node: str,
        vehicle_id: str = "veh-01",
    ) -> MultiFloorRoute:
        """Compute the optimal 3D path across floor aisles and vertical portals."""
        if origin_node not in self._nodes:
            raise KeyError(f"Origin node {origin_node} not found in graph.")
        if destination_node not in self._nodes:
            raise KeyError(f"Destination node {destination_node} not found in graph.")

        distances: dict[str, float] = {origin_node: 0.0}
        prev_nodes: dict[str, str | None] = {origin_node: None}
        prev_edges: dict[str, EdgeData | None] = {origin_node: None}
        queue: list[tuple[float, str]] = [(0.0, origin_node)]

        while queue:
            curr_cost, curr_node = heapq.heappop(queue)
            if curr_node == destination_node:
                break
            if curr_cost > distances[curr_node]:
                continue

            for edge in self._adj.get(curr_node, []):
                new_cost = curr_cost + edge.transit_time_s
                if edge.target not in distances or new_cost < distances[edge.target]:
                    distances[edge.target] = new_cost
                    prev_nodes[edge.target] = curr_node
                    prev_edges[edge.target] = edge
                    heapq.heappush(queue, (new_cost, edge.target))

        if destination_node not in distances:
            msg = (
                f"No navigable multi-floor route exists between "
                f"{origin_node} and {destination_node}."
            )
            raise ValueError(msg)

        path_nodes: list[str] = []
        curr: str | None = destination_node
        while curr is not None:
            path_nodes.append(curr)
            curr = prev_nodes[curr]
        path_nodes.reverse()

        portals_sequence: list[str] = []
        total_distance = 0.0
        total_time = 0.0
        waypoints: list[Waypoint3D] = []

        for i in range(len(path_nodes)):
            n_curr = self._nodes[path_nodes[i]]
            if i > 0:
                edge_opt = prev_edges.get(path_nodes[i])
                if edge_opt is not None:
                    total_distance += edge_opt.distance_m
                    total_time += edge_opt.transit_time_s
                    if edge_opt.portal_id is not None:
                        portals_sequence.append(edge_opt.portal_id)

            if i < len(path_nodes) - 1:
                n_next = self._nodes[path_nodes[i + 1]]
                dx = n_next.x_m - n_curr.x_m
                dy = n_next.y_m - n_curr.y_m
                dz = n_next.z_m - n_curr.z_m
                horiz_dist = math.hypot(dx, dy)
                heading = math.atan2(dy, dx)
                pitch_deg = (
                    math.degrees(math.atan2(dz, max(horiz_dist, 0.01))) if abs(dz) > 0.01 else 0.0
                )

                waypoints.append(
                    Waypoint3D(
                        x_m=round(n_curr.x_m, 2),
                        y_m=round(n_curr.y_m, 2),
                        z_m=round(n_curr.z_m, 2),
                        heading_rad=round(heading, 3),
                        pitch_deg=round(pitch_deg, 2),
                        target_speed_mps=2.0,
                    )
                )
            else:
                waypoints.append(
                    Waypoint3D(
                        x_m=round(n_curr.x_m, 2),
                        y_m=round(n_curr.y_m, 2),
                        z_m=round(n_curr.z_m, 2),
                        heading_rad=0.0,
                        pitch_deg=0.0,
                        target_speed_mps=0.0,
                    )
                )

        origin_floor = self._nodes[origin_node].floor_id
        dest_floor = self._nodes[destination_node].floor_id

        return MultiFloorRoute(
            route_id=f"mfr-{origin_node}-{destination_node}",
            vehicle_id=vehicle_id,
            origin_floor_id=origin_floor,
            destination_floor_id=dest_floor,
            portals_sequence=portals_sequence,
            waypoints=waypoints,
            total_distance_m=round(total_distance, 2),
            estimated_transit_time_s=round(total_time, 2),
        )


def build_default_facility_graph() -> MultiFloorGraph:
    """Construct a turnkey 3-level parking garage topology with straight ramps and car lift."""
    graph = MultiFloorGraph()

    # Floors
    f_b1 = FloorLevel(
        floor_id="B1",
        name="Basement Level 1",
        elevation_m=-3.5,
        bays_count=20,
        available_bays=5,
        is_basement=True,
    )
    f_l1 = FloorLevel(
        floor_id="L1",
        name="Ground Floor",
        elevation_m=0.0,
        bays_count=24,
        available_bays=8,
    )
    f_l2 = FloorLevel(
        floor_id="L2",
        name="Level 2 Upper Deck",
        elevation_m=3.5,
        bays_count=30,
        available_bays=14,
        is_rooftop=True,
    )

    graph.add_floor(f_b1)
    graph.add_floor(f_l1)
    graph.add_floor(f_l2)

    # Nodes on L1 (Ground)
    graph.add_node("L1-entry", "L1", x_m=0.0, y_m=0.0)
    graph.add_node("L1-aisle", "L1", x_m=20.0, y_m=0.0)
    graph.add_node("L1-ramp-up", "L1", x_m=30.0, y_m=5.0)
    graph.add_node("L1-ramp-down", "L1", x_m=10.0, y_m=-10.0)
    graph.add_node("L1-elevator", "L1", x_m=25.0, y_m=15.0)

    graph.add_intra_floor_edge("L1-entry", "L1-aisle", speed_mps=3.0)
    graph.add_intra_floor_edge("L1-aisle", "L1-ramp-up", speed_mps=2.5)
    graph.add_intra_floor_edge("L1-entry", "L1-ramp-down", speed_mps=2.5)
    graph.add_intra_floor_edge("L1-aisle", "L1-elevator", speed_mps=2.0)

    # Nodes on L2 (Upper)
    graph.add_node("L2-ramp-arrival", "L2", x_m=30.0, y_m=35.0)
    graph.add_node("L2-aisle", "L2", x_m=20.0, y_m=35.0)
    graph.add_node("L2-bay-42", "L2", x_m=10.0, y_m=35.0)
    graph.add_node("L2-elevator", "L2", x_m=25.0, y_m=15.0)

    graph.add_intra_floor_edge("L2-ramp-arrival", "L2-aisle", speed_mps=3.0)
    graph.add_intra_floor_edge("L2-aisle", "L2-bay-42", speed_mps=2.0)
    graph.add_intra_floor_edge("L2-aisle", "L2-elevator", speed_mps=2.0)

    # Nodes on B1 (Basement)
    graph.add_node("B1-ramp-arrival", "B1", x_m=10.0, y_m=-30.0)
    graph.add_node("B1-aisle", "B1", x_m=20.0, y_m=-30.0)
    graph.add_node("B1-bay-05", "B1", x_m=30.0, y_m=-30.0)

    graph.add_intra_floor_edge("B1-ramp-arrival", "B1-aisle", speed_mps=3.0)
    graph.add_intra_floor_edge("B1-aisle", "B1-bay-05", speed_mps=2.0)

    # Vertical Portals
    ramp_l1_l2 = VerticalPortal(
        portal_id="ramp-L1-to-L2",
        portal_type=PortalType.STRAIGHT_RAMP,
        from_floor_id="L1",
        to_floor_id="L2",
        length_m=32.0,
        grade_pct=11.0,
        is_bidirectional=True,
        speed_limit_mps=2.5,
    )
    graph.add_portal(ramp_l1_l2, entry_node="L1-ramp-up", exit_node="L2-ramp-arrival")

    ramp_l1_b1 = VerticalPortal(
        portal_id="ramp-L1-to-B1",
        portal_type=PortalType.SPIRAL_RAMP,
        from_floor_id="L1",
        to_floor_id="B1",
        length_m=25.0,
        grade_pct=14.0,
        curvature_radius_m=10.0,
        is_bidirectional=True,
        speed_limit_mps=2.0,
    )
    graph.add_portal(ramp_l1_b1, entry_node="L1-ramp-down", exit_node="B1-ramp-arrival")

    elevator_l1_l2 = VerticalPortal(
        portal_id="elevator-lift-01",
        portal_type=PortalType.VEHICLE_ELEVATOR,
        from_floor_id="L1",
        to_floor_id="L2",
        length_m=3.5,
        grade_pct=0.0,
        is_bidirectional=True,
        speed_limit_mps=1.2,
    )
    graph.add_portal(
        elevator_l1_l2,
        entry_node="L1-elevator",
        exit_node="L2-elevator",
        door_dwell_s=5.0,
        elevator_hoist_speed_mps=1.2,
    )

    return graph
