"""Multi-hypothesis fisheye feature-map matching Monte Carlo Particle Filter Localizer (MCL)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from nearfield360.config import HDMapConfig
from nearfield360.hdmap.models import (
    Lanelet,
    LocalizationConvergence,
    LocalizationState,
    MapTile,
    ParkingSpaceFeature,
    Point2D,
    PoseHypothesis,
)
from nearfield360.hdmap.tiler import MapTiler


def _wrap_to_pi(angle_rad: float) -> float:
    """Normalize angle to the interval [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class FisheyeParticleFilter:
    """Monte Carlo Localizer matching 4-camera fisheye features against HD map tiles."""

    def __init__(
        self,
        tiler: MapTiler,
        config: HDMapConfig | None = None,
        seed: int | None = 42,
    ) -> None:
        self.tiler = tiler
        self.config = config or HDMapConfig()
        self.rng = random.Random(seed)  # noqa: S311
        self.particles: list[PoseHypothesis] = []
        self._converged_frames = 0

    def initialize(
        self,
        init_x: float,
        init_y: float,
        init_yaw_rad: float,
        spread_pos_m: float = 0.50,
        spread_yaw_rad: float = 0.10,
    ) -> None:
        """Initialize particle cloud around a prior pose hypothesis."""
        n = self.config.num_particles
        self.particles = []
        init_weight = 1.0 / n

        for _ in range(n):
            px = self.rng.gauss(init_x, spread_pos_m)
            py = self.rng.gauss(init_y, spread_pos_m)
            pyaw = _wrap_to_pi(self.rng.gauss(init_yaw_rad, spread_yaw_rad))
            self.particles.append(PoseHypothesis(x=px, y=py, yaw_rad=pyaw, weight=init_weight))
        self._converged_frames = 0

    def predict(self, delta_s_m: float, delta_yaw_rad: float) -> None:
        """Propagate particles forward using kinematic vehicle odometry and process noise."""
        if not self.particles:
            return

        noise_pos = self.config.process_noise_pos_m
        noise_yaw = self.config.process_noise_yaw_rad

        updated: list[PoseHypothesis] = []
        for p in self.particles:
            ds = delta_s_m + self.rng.gauss(0.0, noise_pos)
            dyaw = delta_yaw_rad + self.rng.gauss(0.0, noise_yaw)

            # Circular arc motion integration
            mid_yaw = p.yaw_rad + 0.5 * dyaw
            nx = p.x + ds * math.cos(mid_yaw)
            ny = p.y + ds * math.sin(mid_yaw)
            nyaw = _wrap_to_pi(p.yaw_rad + dyaw)

            updated.append(PoseHypothesis(x=nx, y=ny, yaw_rad=nyaw, weight=p.weight))
        self.particles = updated

    def update(self, observed_points_vehicle: Sequence[tuple[float, float] | Point2D]) -> None:
        """Update particle weights by evaluating likelihood of fisheye observations."""
        if not self.particles or not observed_points_vehicle:
            return

        obs_pts = [
            Point2D(x=float(pt[0]), y=float(pt[1])) if isinstance(pt, tuple) else pt
            for pt in observed_points_vehicle
        ]

        sigma_obs = self.config.observation_sigma_m
        sigma_sq_2 = 2.0 * sigma_obs * sigma_obs
        min_like_floor = 1e-6

        # Evaluate each particle against map geometry
        total_weight = 0.0
        new_weights: list[float] = []

        for p in self.particles:
            cos_y = math.cos(p.yaw_rad)
            sin_y = math.sin(p.yaw_rad)

            # Retrieve local map features around particle
            tile = self.tiler.get_tile_for_point(Point2D(x=p.x, y=p.y))
            local_lanes: list[Lanelet] = []
            local_spaces: list[ParkingSpaceFeature] = []
            if tile:
                local_lanes = tile.lanelets
                local_spaces = tile.parking_spaces
            else:
                # Fallback to query
                lanes, spaces, _ = self.tiler.query_features_in_radius(
                    Point2D(x=p.x, y=p.y),
                    radius_m=15.0,
                )
                local_lanes = lanes
                local_spaces = spaces

            # Total log-likelihood
            log_prob = 0.0
            for opt in obs_pts:
                # Transform to map coordinate frame
                mx = p.x + opt.x * cos_y - opt.y * sin_y
                my = p.y + opt.x * sin_y + opt.y * cos_y
                m_pt = Point2D(x=mx, y=my)

                # Find closest map feature distance
                min_d = float("inf")
                for lane in local_lanes:
                    d_left = lane.left_boundary.distance_to_point(m_pt)
                    d_right = lane.right_boundary.distance_to_point(m_pt)
                    d_center = lane.centerline.distance_to_point(m_pt)
                    min_d = min(min_d, d_left, d_right, d_center)

                for sp in local_spaces:
                    d_space = sp.boundary.distance_to_point(m_pt)
                    min_d = min(min_d, d_space)

                if min_d == float("inf"):
                    min_d = 5.0  # Out of map penalty

                # Gaussian likelihood
                d_clamped = min(min_d, 3.0)
                log_prob -= (d_clamped * d_clamped) / sigma_sq_2

            likelihood = max(min_like_floor, math.exp(max(-50.0, log_prob)))
            weight = p.weight * likelihood
            new_weights.append(weight)
            total_weight += weight

        # Check for filter degeneration / kidnapped robot
        if total_weight < 1e-12:
            self._handle_kidnapped_robot()
            return

        # Normalize weights
        norm_particles: list[PoseHypothesis] = []
        for i, p in enumerate(self.particles):
            norm_w = new_weights[i] / total_weight
            norm_particles.append(PoseHypothesis(x=p.x, y=p.y, yaw_rad=p.yaw_rad, weight=norm_w))
        self.particles = norm_particles

        # Adaptive systematic resampling
        n_eff = 1.0 / sum(p.weight * p.weight for p in self.particles)
        if n_eff < len(self.particles) / 2.0:
            self._systematic_resample()

    def _handle_kidnapped_robot(self) -> None:
        """Inject uniform random particles across bounding active tiles when filter degenerates."""
        n = len(self.particles)
        if n == 0:
            return

        tiles = list(self.tiler._tiles.values())
        if not tiles:
            return

        ref_tile: MapTile = self.rng.choice(tiles)
        new_particles: list[PoseHypothesis] = []
        uniform_weight = 1.0 / n

        for _ in range(n):
            rx = self.rng.uniform(ref_tile.x_min, ref_tile.x_max)
            ry = self.rng.uniform(ref_tile.y_min, ref_tile.y_max)
            ryaw = self.rng.uniform(-math.pi, math.pi)
            new_particles.append(PoseHypothesis(x=rx, y=ry, yaw_rad=ryaw, weight=uniform_weight))
        self.particles = new_particles

    def _systematic_resample(self) -> None:
        """Low-variance systematic resampling to mitigate particle depletion."""
        n = len(self.particles)
        weights = [p.weight for p in self.particles]

        # Cumulative sum of weights
        cum_sum = [0.0] * n
        c = 0.0
        for i in range(n):
            c += weights[i]
            cum_sum[i] = c

        resampled: list[PoseHypothesis] = []
        u0 = self.rng.uniform(0.0, 1.0 / n)
        idx = 0
        weight_val = 1.0 / n

        for j in range(n):
            target = u0 + j * (1.0 / n)
            while target > cum_sum[idx] and idx < n - 1:
                idx += 1
            src = self.particles[idx]
            # Add micro-dither to avoid particle collapse
            dx = self.rng.gauss(0.0, 0.005)
            dy = self.rng.gauss(0.0, 0.005)
            dyaw = self.rng.gauss(0.0, 0.002)
            resampled.append(
                PoseHypothesis(
                    x=src.x + dx,
                    y=src.y + dy,
                    yaw_rad=_wrap_to_pi(src.yaw_rad + dyaw),
                    weight=weight_val,
                )
            )

        self.particles = resampled

    def get_state(
        self,
        timestamp_s: float = 0.0,
        associated_features_count: int = 0,
    ) -> LocalizationState:
        """Compute estimated pose, covariance, and convergence status from particle cloud."""
        if not self.particles:
            return LocalizationState(
                timestamp_s=timestamp_s,
                estimated_x=0.0,
                estimated_y=0.0,
                estimated_yaw_rad=0.0,
                status=LocalizationConvergence.LOST,
            )

        # Weighted arithmetic mean for (x, y) and circular mean for yaw
        mean_x = sum(p.x * p.weight for p in self.particles)
        mean_y = sum(p.y * p.weight for p in self.particles)
        sin_sum = sum(math.sin(p.yaw_rad) * p.weight for p in self.particles)
        cos_sum = sum(math.cos(p.yaw_rad) * p.weight for p in self.particles)
        mean_yaw = _wrap_to_pi(math.atan2(sin_sum, cos_sum))

        # Covariance elements
        var_x = sum(p.weight * (p.x - mean_x) ** 2 for p in self.particles)
        var_y = sum(p.weight * (p.y - mean_y) ** 2 for p in self.particles)
        cov_xy = sum(p.weight * (p.x - mean_x) * (p.y - mean_y) for p in self.particles)

        var_yaw = sum(p.weight * (_wrap_to_pi(p.yaw_rad - mean_yaw) ** 2) for p in self.particles)
        cov_xyaw = sum(
            p.weight * (p.x - mean_x) * _wrap_to_pi(p.yaw_rad - mean_yaw) for p in self.particles
        )
        cov_yyaw = sum(
            p.weight * (p.y - mean_y) * _wrap_to_pi(p.yaw_rad - mean_yaw) for p in self.particles
        )

        std_x = math.sqrt(max(0.0, var_x))
        std_y = math.sqrt(max(0.0, var_y))
        std_yaw = math.sqrt(max(0.0, var_yaw))

        cov_matrix = [
            [var_x, cov_xy, cov_xyaw],
            [cov_xy, var_y, cov_yyaw],
            [cov_xyaw, cov_yyaw, var_yaw],
        ]

        # Convergence evaluation
        lat_thresh = self.config.lateral_convergence_threshold_m
        yaw_thresh_rad = math.radians(self.config.heading_convergence_threshold_deg)

        active_tile = self.tiler.get_tile_for_point(Point2D(x=mean_x, y=mean_y))
        tile_id = active_tile.tile_id if active_tile else ""

        # Convergence evaluation: lateral cross-track error is critical lane-level gate
        is_converged = std_y <= lat_thresh and std_yaw <= yaw_thresh_rad and std_x <= 0.25
        if is_converged:
            self._converged_frames += 1
        else:
            self._converged_frames = max(0, self._converged_frames - 1)

        if self._converged_frames >= 3:
            status = LocalizationConvergence.CONVERGED
        elif self._converged_frames >= 1:
            status = LocalizationConvergence.TRACKING
        elif std_x <= 0.50 and std_y <= 0.50:
            status = LocalizationConvergence.CONVERGING
        else:
            status = LocalizationConvergence.DEGRADED

        return LocalizationState(
            timestamp_s=timestamp_s,
            estimated_x=mean_x,
            estimated_y=mean_y,
            estimated_yaw_rad=mean_yaw,
            std_x=std_x,
            std_y=std_y,
            std_yaw_rad=std_yaw,
            covariance=cov_matrix,
            status=status,
            associated_features_count=associated_features_count,
            active_tile_id=tile_id,
        )
