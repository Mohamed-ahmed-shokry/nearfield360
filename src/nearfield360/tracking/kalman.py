"""2D metric Kalman filter for tracking moving obstacles in vehicle coordinates."""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Discrete-time linear Kalman filter with a constant velocity motion model.

    State vector: [x, y, vx, vy]^T in vehicle coordinates (metres, metres/second).
    Measurement vector: [x, y]^T in vehicle coordinates (metres).
    """

    __slots__ = ("F", "H", "P", "Q", "R", "x")

    def __init__(
        self,
        initial_pos: tuple[float, float],
        initial_vel: tuple[float, float] = (0.0, 0.0),
        *,
        dt: float = 0.1,
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 1.0,
        measurement_noise: float = 0.5,
    ) -> None:
        self.x = np.array(
            [initial_pos[0], initial_pos[1], initial_vel[0], initial_vel[1]],
            dtype=np.float64,
        )

        # State transition matrix F
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Measurement matrix H
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Initial state covariance P
        self.P = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float64)

        # Process noise covariance Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q_p = process_noise_pos**2
        q_v = process_noise_vel**2
        self.Q = np.array(
            [
                [dt4 * q_p, 0.0, dt3 * q_p, 0.0],
                [0.0, dt4 * q_p, 0.0, dt3 * q_p],
                [dt3 * q_p, 0.0, dt2 * q_v, 0.0],
                [0.0, dt3 * q_p, 0.0, dt2 * q_v],
            ],
            dtype=np.float64,
        )

        # Measurement noise covariance R
        r_val = measurement_noise**2
        self.R = np.array(
            [
                [r_val, 0.0],
                [0.0, r_val],
            ],
            dtype=np.float64,
        )

    def predict(self) -> tuple[float, float]:
        """Propagate state and covariance forward by one time step dt."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """Incorporate new position measurement into the estimated state."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        y = z - self.H @ self.x  # Innovation
        s = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        k = self.P @ self.H.T @ np.linalg.solve(s, np.eye(2, dtype=np.float64))  # Kalman gain

        self.x = self.x + k @ y
        i = np.eye(4, dtype=np.float64)
        ikh = i - k @ self.H
        self.P = ikh @ self.P @ ikh.T + k @ self.R @ k.T  # Joseph form

        return float(self.x[0]), float(self.x[1])

    @property
    def position(self) -> tuple[float, float]:
        """Estimated (x, y) position in vehicle coordinates."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated (vx, vy) velocity vector in metres/second."""
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        """Estimated scalar speed in metres/second."""
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def position_uncertainty(self) -> float:
        """Root trace of position covariance submatrix."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


__all__ = ["KalmanFilter2D"]
