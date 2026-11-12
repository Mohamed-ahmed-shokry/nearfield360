"""3D UWB multilateration solver, AoA spatial tracking, and proximity zone classification."""

import math
from collections.abc import Sequence

import numpy as np

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.models import (
    DevicePosition3D,
    ProximityZone,
    UWBAnchor,
    UWBRangeMeasurement,
)


def get_default_uwb_anchors() -> list[UWBAnchor]:
    """Return automotive default 4-corner vehicle UWB anchor layout."""
    return [
        UWBAnchor(anchor_id="FL", x_m=1.8, y_m=0.8, z_m=0.7, channel=5, is_aoa_capable=True),
        UWBAnchor(anchor_id="FR", x_m=1.8, y_m=-0.8, z_m=0.7, channel=5, is_aoa_capable=True),
        UWBAnchor(anchor_id="RL", x_m=-1.8, y_m=0.8, z_m=0.7, channel=5, is_aoa_capable=False),
        UWBAnchor(anchor_id="RR", x_m=-1.8, y_m=-0.8, z_m=0.7, channel=5, is_aoa_capable=False),
    ]


class MultilaterationSolver:
    """Solves 3D device coordinates and determines proximity zones from UWB anchor ranges."""

    def __init__(
        self,
        anchors: Sequence[UWBAnchor] | None = None,
        config: DigitalKeyConfig | None = None,
    ) -> None:
        self.anchors = {a.anchor_id: a for a in (anchors or get_default_uwb_anchors())}
        self.config = config or DigitalKeyConfig()

    def is_inside_cabin(self, x: float, y: float, z: float) -> bool:
        """Check if 3D coordinate falls within the passenger cabin bounding envelope."""
        return (-1.2 <= x <= 1.2) and (-0.75 <= y <= 0.75) and (0.3 <= z <= 1.7)

    def classify_zone(self, distance_m: float, is_interior: bool) -> ProximityZone:
        """Categorize physical distance and interior status into a ProximityZone."""
        if is_interior:
            return ProximityZone.INTERIOR
        if distance_m <= self.config.touch_threshold_m:
            return ProximityZone.TOUCH
        if distance_m <= self.config.immediate_threshold_m:
            return ProximityZone.IMMEDIATE
        if distance_m <= self.config.near_threshold_m:
            return ProximityZone.NEAR
        if distance_m <= self.config.far_threshold_m:
            return ProximityZone.FAR
        return ProximityZone.OUT_OF_RANGE

    def solve(
        self,
        measurements: Sequence[UWBRangeMeasurement],
        timestamp_ns: int | None = None,
    ) -> DevicePosition3D:
        """Estimate 3D position using linear least-squares multilateration.

        Requires at least 3 matching anchor measurements for 2D/3D resolution.
        """
        matched: list[tuple[UWBAnchor, float]] = [
            (self.anchors[m.anchor_id], max(0.01, m.distance_m))
            for m in measurements
            if m.anchor_id in self.anchors
        ]

        ts = timestamp_ns or (measurements[0].timestamp_ns if measurements else 0)

        # Fallback if fewer than 3 anchors
        if len(matched) < 3:
            if matched:
                anchor0, dist0 = matched[0]
                x_est, y_est, z_est = anchor0.x_m, anchor0.y_m, anchor0.z_m
                dist_est = dist0
            else:
                x_est, y_est, z_est = 0.0, 0.0, 1.0
                dist_est = 10.0

            interior = self.is_inside_cabin(x_est, y_est, z_est)
            zone = self.classify_zone(dist_est, interior)
            return DevicePosition3D(
                x_m=round(x_est, 3),
                y_m=round(y_est, 3),
                z_m=round(z_est, 3),
                distance_m=round(dist_est, 3),
                zone=zone,
                is_interior=interior,
                residual_error_m=1.0,
                confidence=0.3,
                timestamp_ns=ts,
            )

        # Linear least-squares multilateration
        ref_anchor, ref_dist = matched[0]
        x0, y0, z0, d0 = ref_anchor.x_m, ref_anchor.y_m, ref_anchor.z_m, ref_dist

        a_rows: list[list[float]] = []
        b_rows: list[float] = []

        for anchor, dist in matched[1:]:
            xi, yi, zi, di = anchor.x_m, anchor.y_m, anchor.z_m, dist
            a_rows.append([2.0 * (xi - x0), 2.0 * (yi - y0), 2.0 * (zi - z0)])
            b_val = (d0**2 - di**2) + (xi**2 - x0**2) + (yi**2 - y0**2) + (zi**2 - z0**2)
            b_rows.append(b_val)

        matrix_a = np.array(a_rows, dtype=np.float64)
        vector_b = np.array(b_rows, dtype=np.float64)

        try:
            sol, _, _, _ = np.linalg.lstsq(matrix_a, vector_b, rcond=None)
            x_sol, y_sol, z_sol = float(sol[0]), float(sol[1]), float(sol[2])
        except Exception:
            x_sol, y_sol, z_sol = 0.0, 0.0, 1.0

        # Compute root-mean-square residual error
        residuals = []
        for anchor, dist in matched:
            dist_calc = math.sqrt(
                (x_sol - anchor.x_m) ** 2 + (y_sol - anchor.y_m) ** 2 + (z_sol - anchor.z_m) ** 2
            )
            residuals.append((dist_calc - dist) ** 2)

        rms_err = math.sqrt(sum(residuals) / len(residuals)) if residuals else 0.0
        confidence = max(0.1, min(1.0, 1.0 - (rms_err / 1.5)))

        dist_to_center = math.sqrt(x_sol**2 + y_sol**2)
        interior = self.is_inside_cabin(x_sol, y_sol, z_sol)
        zone = self.classify_zone(dist_to_center, interior)

        return DevicePosition3D(
            x_m=round(x_sol, 3),
            y_m=round(y_sol, 3),
            z_m=round(z_sol, 3),
            distance_m=round(dist_to_center, 3),
            zone=zone,
            is_interior=interior,
            residual_error_m=round(rms_err, 3),
            confidence=round(confidence, 3),
            timestamp_ns=ts,
        )

    def compute_aoa(
        self,
        target_pos: DevicePosition3D,
        anchor_id: str = "FL",
    ) -> tuple[float, float]:
        """Compute relative Angle-of-Arrival (azimuth, elevation in degrees) to an anchor."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return 0.0, 0.0

        dx = target_pos.x_m - anchor.x_m
        dy = target_pos.y_m - anchor.y_m
        dz = target_pos.z_m - anchor.z_m

        azimuth_deg = math.degrees(math.atan2(dy, dx))
        horizontal_dist = math.hypot(dx, dy)
        elevation_deg = math.degrees(math.atan2(dz, horizontal_dist))

        return round(azimuth_deg, 2), round(elevation_deg, 2)
