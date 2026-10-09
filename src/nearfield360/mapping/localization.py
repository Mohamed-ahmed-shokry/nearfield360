"""Multi-sensor vehicle pose estimation and landmark localization using EKF."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nearfield360.config import MappingConfig
from nearfield360.mapping.models import (
    FacilityMap,
    FacilitySlot,
    LandmarkObservation,
    LocalizationReport,
    PoseEstimate,
)


def _wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class LocalizationStepRecord:
    """Snapshot of ground-truth, dead-reckoning, and filtered state at one simulation step."""

    step: int
    t: float
    true_pose: tuple[float, float, float]
    dead_reckoning_pose: tuple[float, float, float]
    estimated_pose: tuple[float, float, float]
    covariance: list[list[float]]
    observed_landmarks: list[str] = field(default_factory=list)


class PoseEstimator:
    """Extended Kalman Filter (EKF) estimator fusing odometry dead-reckoning and slot landmarks."""

    def __init__(
        self,
        facility: FacilityMap,
        config: MappingConfig | None = None,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        wheelbase_m: float = 2.7,
    ) -> None:
        self.facility = facility
        self.config = config or MappingConfig()
        self.wheelbase = wheelbase_m

        self.x = float(initial_pose[0])
        self.y = float(initial_pose[1])
        self.theta = _wrap_angle(float(initial_pose[2]))

        # State covariance P (3x3)
        self.cov: np.ndarray = np.array(
            [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.01]],
            dtype=np.float64,
        )

    def current_pose(self) -> PoseEstimate:
        """Return current estimated pose and 3x3 covariance."""
        cov_list = [[float(v) for v in row] for row in self.cov]
        return PoseEstimate(
            x=self.x,
            y=self.y,
            heading_rad=self.theta,
            covariance=cov_list,
        )

    def reset(
        self,
        initial_pose: tuple[float, float, float],
        initial_covariance: list[list[float]] | None = None,
    ) -> None:
        """Reset the filter state to a known pose."""
        self.x = float(initial_pose[0])
        self.y = float(initial_pose[1])
        self.theta = _wrap_angle(float(initial_pose[2]))
        if initial_covariance is not None:
            self.cov = np.array(initial_covariance, dtype=np.float64)
        else:
            self.cov = np.array(
                [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.01]],
                dtype=np.float64,
            )

    def predict_odometry(
        self,
        delta_s: float,
        steering_angle_rad: float,
    ) -> PoseEstimate:
        """Propagate state and covariance forward using kinematic bicycle odometry."""
        # Kinematic bicycle equations
        delta_theta = (delta_s / self.wheelbase) * math.tan(steering_angle_rad)
        mid_theta = self.theta + delta_theta / 2.0

        dx = delta_s * math.cos(mid_theta)
        dy = delta_s * math.sin(mid_theta)

        # Update state
        self.x += dx
        self.y += dy
        self.theta = _wrap_angle(self.theta + delta_theta)

        # State transition Jacobian F
        f_matrix = np.array(
            [
                [1.0, 0.0, -delta_s * math.sin(mid_theta)],
                [0.0, 1.0, delta_s * math.cos(mid_theta)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance Q
        var_s = (self.config.odometry_noise_dist * abs(delta_s)) ** 2 + 1e-6
        var_yaw = (self.config.odometry_noise_yaw * abs(delta_s)) ** 2 + 1e-6

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        q_matrix = np.array(
            [
                [var_s * cos_t**2, var_s * cos_t * sin_t, 0.0],
                [var_s * cos_t * sin_t, var_s * sin_t**2, 0.0],
                [0.0, 0.0, var_yaw],
            ],
            dtype=np.float64,
        )

        # Covariance propagation
        self.cov = f_matrix @ self.cov @ f_matrix.T + q_matrix

        # Guarantee symmetry and positive diagonals
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.cov[0, 0] = max(1e-6, self.cov[0, 0])
        self.cov[1, 1] = max(1e-6, self.cov[1, 1])
        self.cov[2, 2] = max(1e-6, self.cov[2, 2])

        return self.current_pose()

    def update_landmark(self, observation: LandmarkObservation) -> bool:
        """Update filter state using perceived slot landmark relative measurement."""
        slot = self.facility.get_slot(observation.slot_id)
        if slot is None:
            # If slot_id is unknown, attempt spatial gating association across all mapped slots
            slot = self._associate_slot(observation.observed_center)
            if slot is None:
                return False

        # Mapped slot global landmark coordinates
        mx, my = slot.center
        m_theta = slot.heading_rad

        # Expected measurement in vehicle body frame:
        # z_rel = R(-theta) * (m - pose_xy)
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        dx_glob = mx - self.x
        dy_glob = my - self.y

        z_pred_x = cos_t * dx_glob + sin_t * dy_glob
        z_pred_y = -sin_t * dx_glob + cos_t * dy_glob
        z_pred_theta = _wrap_angle(m_theta - self.theta)

        # Actual observation in body frame
        z_act_x = observation.observed_center[0]
        z_act_y = observation.observed_center[1]
        z_act_theta = _wrap_angle(observation.observed_heading)

        # Innovation residual
        innov_x = z_act_x - z_pred_x
        innov_y = z_act_y - z_pred_y
        innov_theta = _wrap_angle(z_act_theta - z_pred_theta)
        y_innov = np.array([innov_x, innov_y, innov_theta], dtype=np.float64)

        # Measurement Jacobian H (3x3)
        h_matrix = np.array(
            [
                [-cos_t, -sin_t, -sin_t * dx_glob + cos_t * dy_glob],
                [sin_t, -cos_t, -cos_t * dx_glob - sin_t * dy_glob],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        # Measurement noise covariance R
        var_pos = (self.config.slot_observation_noise_pos / max(0.1, observation.confidence)) ** 2
        var_head = (self.config.slot_observation_noise_yaw / max(0.1, observation.confidence)) ** 2
        r_matrix = np.diag([var_pos, var_pos, var_head])

        # Innovation covariance S = H * P * H^T + R
        s_matrix = h_matrix @ self.cov @ h_matrix.T + r_matrix

        # Mahalanobis distance validation gate
        try:
            s_inv = np.linalg.inv(s_matrix)
        except np.linalg.LinAlgError:
            return False

        d_mahalanobis = float(math.sqrt(max(0.0, float(y_innov.T @ s_inv @ y_innov))))
        # Gate threshold (3-DOF chi-squared at 99% is ~3.38 standard deviations)
        if d_mahalanobis > 4.5:
            return False

        # Kalman Gain K = P * H^T * S^-1
        k_matrix = self.cov @ h_matrix.T @ s_inv

        # State update
        dx_update = k_matrix @ y_innov
        self.x += float(dx_update[0])
        self.y += float(dx_update[1])
        self.theta = _wrap_angle(self.theta + float(dx_update[2]))

        # Joseph form covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        eye = np.eye(3, dtype=np.float64)
        i_minus_kh = eye - k_matrix @ h_matrix
        self.cov = i_minus_kh @ self.cov @ i_minus_kh.T + k_matrix @ r_matrix @ k_matrix.T
        self.cov = 0.5 * (self.cov + self.cov.T)

        return True

    def _associate_slot(self, observed_body_pt: tuple[float, float]) -> FacilitySlot | None:
        """Find the closest mapped slot corresponding to a vehicle-relative observation."""
        # Convert body observation to global coordinate guess
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        bx, by = observed_body_pt
        gx = self.x + cos_t * bx - sin_t * by
        gy = self.y + sin_t * bx + cos_t * by

        best_slot: FacilitySlot | None = None
        best_dist = float("inf")
        gate_m = self.config.slot_association_gate_m

        for slot in self.facility.slots:
            dist = math.hypot(slot.center[0] - gx, slot.center[1] - gy)
            if dist < gate_m and dist < best_dist:
                best_dist = dist
                best_slot = slot

        return best_slot

    def update_landmarks(self, observations: Sequence[LandmarkObservation]) -> int:
        """Update with a batch of perceived landmark observations; returns count applied."""
        applied = 0
        for obs in observations:
            if self.update_landmark(obs):
                applied += 1
        return applied


class LocalizationSimulator:
    """Simulates vehicle trajectory tracking with noisy odometry and slot landmark updates."""

    def __init__(
        self,
        facility: FacilityMap,
        config: MappingConfig | None = None,
        sensor_range_m: float = 12.0,
        observation_interval_steps: int = 3,
        seed: int = 42,
    ) -> None:
        self.facility = facility
        self.config = config or MappingConfig()
        self.sensor_range = sensor_range_m
        self.obs_interval = observation_interval_steps
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        waypoints: Sequence[tuple[float, float, float]],
        step_dt_s: float = 0.1,
    ) -> tuple[LocalizationReport, list[LocalizationStepRecord]]:
        """Simulate vehicle motion along waypoints, applying noisy odometry and landmark fixes."""
        if len(waypoints) < 2:
            raise ValueError("Waypoints sequence must have at least 2 points")

        start_pose = waypoints[0]
        estimator = PoseEstimator(self.facility, self.config, initial_pose=start_pose)

        # Dead-reckoning pose tracking (unfiltered, accumulates drift)
        dr_x, dr_y, dr_theta = start_pose

        records: list[LocalizationStepRecord] = []
        total_dist = 0.0
        pos_errors: list[float] = []
        heading_errors: list[float] = []
        uncertainties: list[float] = []
        landmark_update_count = 0

        current_true_pose = list(start_pose)

        for step in range(len(waypoints) - 1):
            p_curr = waypoints[step]
            p_next = waypoints[step + 1]

            # Ground truth motion delta
            ds_true = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            dtheta_true = _wrap_angle(p_next[2] - p_curr[2])
            total_dist += ds_true

            # Update ground truth pose
            current_true_pose = [p_next[0], p_next[1], p_next[2]]

            # Synthesize noisy wheel odometry measurement
            noise_s = float(
                self.rng.normal(0.0, self.config.odometry_noise_dist * max(0.1, ds_true))
            )
            noise_yaw = float(
                self.rng.normal(0.0, self.config.odometry_noise_yaw * max(0.1, ds_true))
            )

            ds_noisy = max(0.0, ds_true + noise_s)
            dtheta_noisy = dtheta_true + noise_yaw

            # Equivalent steering angle for kinematic model
            steering_angle = math.atan2(dtheta_noisy * estimator.wheelbase, max(0.01, ds_noisy))

            # 1. Update Dead-reckoning
            dr_theta = _wrap_angle(dr_theta + dtheta_noisy)
            dr_x += ds_noisy * math.cos(dr_theta)
            dr_y += ds_noisy * math.sin(dr_theta)

            # 2. EKF Prediction step
            est_pose = estimator.predict_odometry(ds_noisy, steering_angle)

            # 3. Perception / Landmark observation step
            observed_slot_ids: list[str] = []
            if step % self.obs_interval == 0:
                observations = self._generate_slot_observations(current_true_pose)
                for obs in observations:
                    if estimator.update_landmark(obs):
                        observed_slot_ids.append(obs.slot_id)
                        landmark_update_count += 1

                est_pose = estimator.current_pose()

            # Record errors
            pos_err = math.hypot(
                est_pose.x - current_true_pose[0], est_pose.y - current_true_pose[1]
            )
            head_err = abs(_wrap_angle(est_pose.heading_rad - current_true_pose[2]))
            pos_errors.append(pos_err)
            heading_errors.append(head_err)
            uncertainties.append(est_pose.position_uncertainty_m)

            records.append(
                LocalizationStepRecord(
                    step=step,
                    t=step * step_dt_s,
                    true_pose=(current_true_pose[0], current_true_pose[1], current_true_pose[2]),
                    dead_reckoning_pose=(dr_x, dr_y, dr_theta),
                    estimated_pose=(est_pose.x, est_pose.y, est_pose.heading_rad),
                    covariance=est_pose.covariance,
                    observed_landmarks=observed_slot_ids,
                )
            )

        final_est = estimator.current_pose()
        report = LocalizationReport(
            trajectory_length_m=total_dist,
            final_pose=final_est,
            max_position_uncertainty_m=max(uncertainties) if uncertainties else 0.0,
            mean_position_error_m=float(np.mean(pos_errors)) if pos_errors else 0.0,
            max_position_error_m=max(pos_errors) if pos_errors else 0.0,
            mean_heading_error_rad=float(np.mean(heading_errors)) if heading_errors else 0.0,
            max_heading_error_rad=max(heading_errors) if heading_errors else 0.0,
            total_landmark_updates=landmark_update_count,
            step_count=len(records),
        )

        return report, records

    def _generate_slot_observations(
        self,
        vehicle_pose: list[float],
    ) -> list[LandmarkObservation]:
        """Synthesize noisy relative landmark observations for slots within sensor field of view."""
        vx, vy, v_theta = vehicle_pose
        observations: list[LandmarkObservation] = []

        for slot in self.facility.slots:
            sx, sy = slot.center
            dist = math.hypot(sx - vx, sy - vy)
            if dist > self.sensor_range:
                continue

            # Compute relative angle to slot
            angle_to_slot = math.atan2(sy - vy, sx - vx)
            rel_bearing = abs(_wrap_angle(angle_to_slot - v_theta))
            # Surround 4-camera perception covers 360 degrees (~pi)
            if rel_bearing > math.radians(160.0):
                continue

            # Transform mapped center into true body coordinates
            cos_t = math.cos(v_theta)
            sin_t = math.sin(v_theta)
            dx = sx - vx
            dy = sy - vy
            body_x = cos_t * dx + sin_t * dy
            body_y = -sin_t * dx + cos_t * dy
            body_heading = _wrap_angle(slot.heading_rad - v_theta)

            # Add sensor noise
            noise_x = float(self.rng.normal(0.0, self.config.slot_observation_noise_pos))
            noise_y = float(self.rng.normal(0.0, self.config.slot_observation_noise_pos))
            noise_head = float(self.rng.normal(0.0, self.config.slot_observation_noise_yaw))

            obs_center = (body_x + noise_x, body_y + noise_y)
            obs_head = _wrap_angle(body_heading + noise_head)

            observations.append(
                LandmarkObservation(
                    slot_id=slot.slot_id,
                    observed_center=obs_center,
                    observed_heading=obs_head,
                    confidence=max(0.3, 1.0 - (dist / self.sensor_range) * 0.4),
                )
            )

        return observations
