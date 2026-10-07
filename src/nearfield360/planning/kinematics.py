"""Ackermann steering kinematics, vehicle footprint geometry, and curve primitives."""

from __future__ import annotations

import math
from typing import NamedTuple

from nearfield360.config import ParkingPlannerConfig
from nearfield360.planning.models import ManeuverGear


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the interval [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class KinematicWaypoint(NamedTuple):
    """Raw spatial waypoint produced during kinematic integration."""

    x: float
    y: float
    heading_rad: float
    curvature: float
    distance_m: float


class AckermannVehicle:
    """Automotive kinematic bicycle model and bounding footprint generator."""

    def __init__(self, config: ParkingPlannerConfig | None = None) -> None:
        self.config = config or ParkingPlannerConfig()
        self.wheelbase = self.config.wheelbase
        self.front_overhang = self.config.front_overhang
        self.rear_overhang = self.config.rear_overhang
        self.width = self.config.vehicle_width
        self.max_steer_angle_rad = self.config.max_steer_angle_rad
        self.min_turn_radius = self.wheelbase / math.tan(self.max_steer_angle_rad)
        self.max_curvature = 1.0 / self.min_turn_radius

    def compute_footprint_polygon(
        self,
        x: float,
        y: float,
        heading_rad: float,
    ) -> list[tuple[float, float]]:
        """Compute the 4-corner metric bounding box of the vehicle at pose (x, y, heading).

        The reference frame is centered at the vehicle rear axle.
        Corners are returned in counter-clockwise order:
        [front-left, rear-left, rear-right, front-right].
        """
        cos_t = math.cos(heading_rad)
        sin_t = math.sin(heading_rad)

        half_w = 0.5 * self.width
        x_front = self.wheelbase + self.front_overhang
        x_rear = -self.rear_overhang

        # Vehicle-frame body coordinates: (x_body, y_body)
        body_corners = [
            (x_front, half_w),  # Front-Left
            (x_rear, half_w),  # Rear-Left
            (x_rear, -half_w),  # Rear-Right
            (x_front, -half_w),  # Front-Right
        ]

        # Rotate and translate into global/BEV frame
        world_corners: list[tuple[float, float]] = []
        for bx, by in body_corners:
            wx = x + bx * cos_t - by * sin_t
            wy = y + bx * sin_t + by * cos_t
            world_corners.append((round(wx, 4), round(wy, 4)))

        return world_corners

    def generate_straight(
        self,
        start_pose: tuple[float, float, float],
        length_m: float,
        gear: ManeuverGear,
        step_size: float = 0.1,
    ) -> list[KinematicWaypoint]:
        """Generate equidistant waypoints along a straight segment (curvature = 0)."""
        if length_m <= 1e-6:
            return [KinematicWaypoint(start_pose[0], start_pose[1], start_pose[2], 0.0, 0.0)]

        num_steps = max(1, math.ceil(length_m / step_size))
        actual_step = length_m / num_steps

        sign = 1.0 if gear == ManeuverGear.FORWARD else -1.0
        cos_t = math.cos(start_pose[2])
        sin_t = math.sin(start_pose[2])

        waypoints: list[KinematicWaypoint] = []
        for i in range(num_steps + 1):
            dist = i * actual_step
            # If reverse, vehicle moves opposite to heading orientation
            wx = start_pose[0] + sign * dist * cos_t
            wy = start_pose[1] + sign * dist * sin_t
            waypoints.append(
                KinematicWaypoint(
                    x=round(wx, 4),
                    y=round(wy, 4),
                    heading_rad=normalize_angle(start_pose[2]),
                    curvature=0.0,
                    distance_m=round(dist, 4),
                )
            )

        return waypoints

    def generate_arc(
        self,
        start_pose: tuple[float, float, float],
        curvature: float,
        arc_length_m: float,
        gear: ManeuverGear,
        step_size: float = 0.1,
    ) -> list[KinematicWaypoint]:
        """Generate equidistant waypoints along a constant curvature circular arc.

        Curvature > 0 turns left (counter-clockwise), Curvature < 0 turns right.
        """
        if abs(curvature) <= 1e-6:
            return self.generate_straight(start_pose, arc_length_m, gear, step_size)

        if arc_length_m <= 1e-6:
            return [KinematicWaypoint(start_pose[0], start_pose[1], start_pose[2], curvature, 0.0)]

        # Bound curvature by steering limits
        bounded_curvature = math.copysign(min(abs(curvature), self.max_curvature), curvature)

        num_steps = max(1, math.ceil(arc_length_m / step_size))
        actual_step = arc_length_m / num_steps

        # Kinematic integration
        sign = 1.0 if gear == ManeuverGear.FORWARD else -1.0
        waypoints: list[KinematicWaypoint] = []

        curr_x = start_pose[0]
        curr_y = start_pose[1]
        curr_theta = start_pose[2]

        waypoints.append(
            KinematicWaypoint(
                x=round(curr_x, 4),
                y=round(curr_y, 4),
                heading_rad=normalize_angle(curr_theta),
                curvature=round(bounded_curvature, 4),
                distance_m=0.0,
            )
        )

        for i in range(1, num_steps + 1):
            dist = i * actual_step
            # Midpoint/Euler integration step
            d_theta = sign * bounded_curvature * actual_step
            mid_theta = curr_theta + 0.5 * d_theta
            curr_x += sign * actual_step * math.cos(mid_theta)
            curr_y += sign * actual_step * math.sin(mid_theta)
            curr_theta += d_theta

            waypoints.append(
                KinematicWaypoint(
                    x=round(curr_x, 4),
                    y=round(curr_y, 4),
                    heading_rad=normalize_angle(curr_theta),
                    curvature=round(bounded_curvature, 4),
                    distance_m=round(dist, 4),
                )
            )

        return waypoints


def plan_two_circle_reeds_shepp(
    start_pose: tuple[float, float, float],
    target_pose: tuple[float, float, float],
    radius: float,
    vehicle: AckermannVehicle,
    step_size: float = 0.1,
) -> list[tuple[ManeuverGear, list[KinematicWaypoint]]] | None:
    """Plan an S-curve (inflection circle arcs) connecting two parallel or near-parallel poses.

    Typical for reverse parallel parking:
    Arc 1: Turn into the slot (e.g. reverse right)
    Arc 2: Inflection counter-turn to straighten (e.g. reverse left)
    """
    x0, y0, t0 = start_pose
    xf, yf, _tf = target_pose

    # Transform target pose into coordinate system aligned with start pose
    dx = xf - x0
    dy = yf - y0
    sin0 = math.sin(t0)
    cos0 = math.cos(t0)

    # Local lateral coordinate relative to start orientation
    ly = -dx * sin0 + dy * cos0

    # For reverse parallel parking, target is laterally displaced (ly != 0)
    # In symmetric S-turn: lateral offset = 2 * R * (1 - cos(alpha))
    turn_dir = -1.0 if ly < 0 else 1.0
    abs_ly = abs(ly)

    if abs_ly > 4.0 * radius:
        # Too far laterally for a simple 2-circle S-curve without intermediate straight
        return None

    cos_alpha = 1.0 - abs_ly / (2.0 * radius)
    if cos_alpha < -1.0 or cos_alpha > 1.0:
        return None

    alpha = math.acos(cos_alpha)
    if alpha < 1e-4:
        return None

    arc_length = radius * alpha
    curv1 = turn_dir / radius
    curv2 = -turn_dir / radius

    # Maneuver 1: Reverse with curv1
    wps1 = vehicle.generate_arc(start_pose, curv1, arc_length, ManeuverGear.REVERSE, step_size)
    if not wps1:
        return None

    mid_pose = (wps1[-1].x, wps1[-1].y, wps1[-1].heading_rad)

    # Maneuver 2: Reverse with curv2
    wps2 = vehicle.generate_arc(mid_pose, curv2, arc_length, ManeuverGear.REVERSE, step_size)
    if not wps2:
        return None

    return [(ManeuverGear.REVERSE, wps1), (ManeuverGear.REVERSE, wps2)]
