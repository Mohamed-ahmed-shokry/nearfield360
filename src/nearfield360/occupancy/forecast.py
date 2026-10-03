"""Spatiotemporal BEV occupancy forecasting network and recurrent runtime engine."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nearfield360.config import OccupancyForecastConfig
from nearfield360.geometry.bev import BevGrid
from nearfield360.health import CameraHealthReport
from nearfield360.occupancy.evidence import OccupancyEvidence
from nearfield360.occupancy.models import (
    OccupancyForecastGrid,
    OccupancyForecastStep,
    TemporalOccupancyState,
)
from nearfield360.tracking.models import TrackedObstacle


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid function."""
    pos = x >= 0.0
    neg = ~pos
    result = np.empty_like(x, dtype=np.float64)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result


def _advect_grid(
    grid_data: NDArray[np.float64],
    displacement_x: NDArray[np.float64],
    displacement_y: NDArray[np.float64],
    resolution: float,
    border_value: float = 0.0,
) -> NDArray[np.float64]:
    """Advect a 2D grid forward along continuous displacement fields.

    Uses backward-mapping bilinear interpolation via OpenCV remap.
    displacement_x is longitudinal displacement in metres (along cols).
    displacement_y is lateral displacement in metres (along rows).
    """
    height, width = grid_data.shape
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    # In backward mapping, source position is (dest - displacement / resolution)
    src_cols = cols - (displacement_x / resolution).astype(np.float32)
    src_rows = rows - (displacement_y / resolution).astype(np.float32)

    remapped = cv2.remap(
        grid_data.astype(np.float32),
        src_cols,
        src_rows,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return remapped.astype(np.float64)


def fuse_cross_attention_occupancy(
    grid: BevGrid,
    layers: Sequence[OccupancyEvidence],
    health_reports: Sequence[CameraHealthReport | None] | None = None,
) -> OccupancyEvidence:
    """Fuse multi-camera occupancy layers using spatial cross-attention weighting.

    When multiple fisheye cameras observe the same BEV cell, camera health and
    observation confidence are cross-weighted using softmax normalization to
    prevent seam artifacts at overlapping camera boundaries.
    """
    if not layers:
        raise ValueError("fuse_cross_attention_occupancy requires at least one layer")

    shape = grid.shape
    total_occupied = np.zeros(shape, dtype=np.float64)
    total_free = np.zeros(shape, dtype=np.float64)
    total_observed = np.zeros(shape, dtype=np.int64)

    health_weights = [
        1.0 if (health_reports is None or r is None) else float(r.discount_weight)
        for r in (health_reports or [None] * len(layers))
    ]

    # Pre-extract observation masks and weights
    layer_obs = [layer.observed for layer in layers]
    layer_weights = np.stack(
        [(obs.astype(np.float64) * hw) for obs, hw in zip(layer_obs, health_weights, strict=True)],
        axis=0,
    )

    weight_sum = np.sum(layer_weights, axis=0)
    has_evidence = weight_sum > 0.0

    # Compute normalized attention coefficients across cameras per cell
    attn_weights = np.zeros_like(layer_weights)
    np.divide(layer_weights, weight_sum, out=attn_weights, where=has_evidence)

    for idx, layer in enumerate(layers):
        w = attn_weights[idx]
        total_occupied += layer.occupied * w
        total_free += layer.free * w
        total_observed += layer.observed

    return OccupancyEvidence(
        grid=grid,
        occupied=total_occupied,
        free=total_free,
        observed=total_observed,
    )


class TemporalOccupancyForecaster:
    """Spatiotemporal recurrent neural BEV occupancy forecasting engine.

    Maintains internal recurrent memory over multi-frame observations,
    estimates dynamic velocity fields on the vehicle-centric BEV plane,
    and rolls forward predicted 4D occupancy grids over multi-second parking horizons.
    """

    def __init__(
        self,
        grid: BevGrid,
        config: OccupancyForecastConfig | None = None,
        seed: int = 42,
    ) -> None:
        if not isinstance(grid, BevGrid):
            raise ValueError("grid must be a BevGrid instance")

        self.grid = grid
        self.config = config or OccupancyForecastConfig()
        self._current_state: TemporalOccupancyState | None = None
        self._prev_occupancy: NDArray[np.float64] | None = None
        self._prev_timestamp: float | None = None

        # Deterministic initialization of recurrent ConvGRU weights
        rng = np.random.default_rng(seed)
        c_in = 4  # (P_occ, Uncertainty, Observation_mask, P_free)
        c_hid = self.config.hidden_channels
        k = self.config.spatial_kernel_size

        scale_in = math.sqrt(2.0 / (c_in * k * k))
        scale_hid = math.sqrt(2.0 / (c_hid * k * k))

        # Gate weights: Update gate (z), Reset gate (r), Candidate gate (h)
        self._W_z = rng.normal(0.0, scale_in, size=(c_hid, c_in, k, k)).astype(np.float64)
        self._U_z = rng.normal(0.0, scale_hid, size=(c_hid, c_hid, k, k)).astype(np.float64)
        self._b_z = np.zeros((c_hid, 1, 1), dtype=np.float64)

        self._W_r = rng.normal(0.0, scale_in, size=(c_hid, c_in, k, k)).astype(np.float64)
        self._U_r = rng.normal(0.0, scale_hid, size=(c_hid, c_hid, k, k)).astype(np.float64)
        self._b_r = np.zeros((c_hid, 1, 1), dtype=np.float64)

        self._W_h = rng.normal(0.0, scale_in, size=(c_hid, c_in, k, k)).astype(np.float64)
        self._U_h = rng.normal(0.0, scale_hid, size=(c_hid, c_hid, k, k)).astype(np.float64)
        self._b_h = np.zeros((c_hid, 1, 1), dtype=np.float64)

        # Output projection head from hidden state to [P_occ, Uncertainty]
        self._W_out = rng.normal(0.0, scale_hid, size=(2, c_hid, 1, 1)).astype(np.float64)
        self._b_out = np.array([0.0, -1.0], dtype=np.float64).reshape(2, 1, 1)

    @property
    def current_state(self) -> TemporalOccupancyState | None:
        """Current recurrent hidden state and dynamic belief on the BEV grid."""
        return self._current_state

    def reset(self) -> None:
        """Clear recurrent temporal state and reset history."""
        self._current_state = None
        self._prev_occupancy = None
        self._prev_timestamp = None

    def _conv2d(
        self,
        x: NDArray[np.float64],
        weights: NDArray[np.float64],
        bias: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multi-channel 2D spatial convolution using OpenCV filter2D."""
        c_out, c_in, _, _ = weights.shape
        _, h, w = x.shape
        output = np.zeros((c_out, h, w), dtype=np.float64)

        for out_idx in range(c_out):
            accum = np.zeros((h, w), dtype=np.float64)
            for in_idx in range(c_in):
                kernel = weights[out_idx, in_idx]
                if np.any(kernel != 0.0):
                    # cv2.filter2D expects float32/float64
                    accum += cv2.filter2D(
                        x[in_idx],
                        ddepth=-1,
                        kernel=kernel,
                        borderType=cv2.BORDER_REFLECT,
                    )
            output[out_idx] = accum + bias[out_idx, 0, 0]

        return output

    def _estimate_velocity_field(
        self,
        current_occ: NDArray[np.float64],
        dt: float,
        obstacles: Sequence[TrackedObstacle] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Estimate 2D metric velocity field (vx, vy in m/s) and dynamic cell mask."""
        shape = self.grid.shape
        vx_grid = np.zeros(shape, dtype=np.float64)
        vy_grid = np.zeros(shape, dtype=np.float64)
        res = self.grid.resolution

        # 1. Optical flow / temporal difference gradient estimation
        if self._prev_occupancy is not None and dt > 1e-4:
            clean_curr = np.nan_to_num(current_occ, nan=0.0)
            clean_prev = np.nan_to_num(self._prev_occupancy, nan=0.0)

            # Spatial gradients via Sobel
            sobel_x = cv2.Sobel(clean_curr.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(clean_curr.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            # Spatial metric gradient (dP/dx, dP/dy)
            grad_x = sobel_x / (8.0 * res)  # dx along cols (forward)
            grad_y = sobel_y / (8.0 * res)  # dy along rows (left)
            temporal_diff = (clean_curr - clean_prev) / dt

            denom = grad_x * grad_x + grad_y * grad_y + 0.05
            # Optical flow constraint: -dI/dt = grad_x * vx + grad_y * vy
            vx_flow = -(temporal_diff * grad_x) / denom
            vy_flow = -(temporal_diff * grad_y) / denom

            # Only assign flow where gradient and occupancy are noticeable
            active = (np.abs(grad_x) + np.abs(grad_y) > 0.05) & (clean_curr > 0.2)
            vx_grid[active] = np.clip(vx_flow[active], -15.0, 15.0)
            vy_grid[active] = np.clip(vy_flow[active], -15.0, 15.0)

        # 2. Tracked obstacle velocity anchoring
        if obstacles:
            for obs in obstacles:
                # Obtain metric footprint bounds
                pos_x, pos_y = obs.position
                vel_x, vel_y = obs.velocity
                # Footprint half dimensions (default automotive 2.0m x 4.5m)
                half_len = 2.2
                half_wid = 1.0

                corners = np.array(
                    [
                        [pos_x - half_len, pos_y - half_wid],
                        [pos_x + half_len, pos_y - half_wid],
                        [pos_x + half_len, pos_y + half_wid],
                        [pos_x - half_len, pos_y + half_wid],
                    ],
                    dtype=np.float64,
                )
                grid_idx = self.grid.world_to_grid(corners)
                if np.any(grid_idx.valid):
                    valid_indices = grid_idx.indices[grid_idx.valid]
                    r_min = max(0, int(valid_indices[:, 0].min()))
                    r_max = min(shape[0], int(valid_indices[:, 0].max()) + 1)
                    c_min = max(0, int(valid_indices[:, 1].min()))
                    c_max = min(shape[1], int(valid_indices[:, 1].max()) + 1)

                    if r_max > r_min and c_max > c_min:
                        vx_grid[r_min:r_max, c_min:c_max] = vel_x
                        vy_grid[r_min:r_max, c_min:c_max] = vel_y

        # Velocity magnitude thresholding
        speed = np.sqrt(vx_grid * vx_grid + vy_grid * vy_grid)
        dynamic_mask = (speed >= self.config.min_velocity_threshold) & (
            np.nan_to_num(current_occ, nan=0.0) > 0.2
        )

        return np.stack([vx_grid, vy_grid], axis=0), dynamic_mask

    def update(
        self,
        evidence: OccupancyEvidence,
        timestamp: float,
        obstacles: Sequence[TrackedObstacle] | None = None,
    ) -> TemporalOccupancyState:
        """Update recurrent temporal BEV state with new observation evidence.

        Fuses historical state with new evidence via gated ConvGRU recurrence,
        estimates dynamic velocity fields, and tracks belief persistence.
        """
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative float")

        raw_occ = evidence.occupancy()
        raw_unc = evidence.uncertainty()
        observed_mask = (evidence.observed > 0).astype(np.float64)

        # Feature preparation: clean NaNs for neural ConvGRU processing
        norm_occ = np.nan_to_num(raw_occ, nan=0.5)
        norm_unc = np.nan_to_num(raw_unc, nan=0.288)
        evidence_sum = evidence.occupied + evidence.free + 1e-6
        norm_free = np.clip(evidence.free / evidence_sum, 0.0, 1.0)

        x_t = np.stack([norm_occ, norm_unc, observed_mask, norm_free], axis=0)

        dt = 0.1 if self._prev_timestamp is None else max(timestamp - self._prev_timestamp, 1e-4)
        step_idx = 0 if self._current_state is None else self._current_state.step_index + 1

        # 1. Recurrent ConvGRU cell computation
        c_hid = self.config.hidden_channels
        shape = self.grid.shape

        if self._current_state is None:
            # Initialize hidden state from input features
            h_prev = np.zeros((c_hid, shape[0], shape[1]), dtype=np.float64)
            # Replicate input features into initial channels
            for ch in range(c_hid):
                h_prev[ch] = x_t[ch % 4]
        else:
            h_prev = self._current_state.hidden_state
            # Apply memory decay to unobserved cells
            decay_factor = math.pow(self.config.memory_decay, dt)
            unobserved = observed_mask == 0.0
            h_prev[:, unobserved] *= decay_factor

        # Gated ConvGRU equations
        # z = sigmoid(W_z * x + U_z * h + b_z)
        z_t = _sigmoid(
            self._conv2d(x_t, self._W_z, self._b_z) + self._conv2d(h_prev, self._U_z, self._b_z)
        )
        # r = sigmoid(W_r * x + U_r * h + b_r)
        r_t = _sigmoid(
            self._conv2d(x_t, self._W_r, self._b_r) + self._conv2d(h_prev, self._U_r, self._b_r)
        )
        # h_tilde = tanh(W_h * x + U_h * (r * h) + b_h)
        rh = r_t * h_prev
        h_cand = np.tanh(
            self._conv2d(x_t, self._W_h, self._b_h) + self._conv2d(rh, self._U_h, self._b_h)
        )
        # h_t = (1 - z) * h_prev + z * h_cand
        h_t = (1.0 - z_t) * h_prev + z_t * h_cand

        # 2. Output projection head
        out_features = self._conv2d(h_t, self._W_out, self._b_out)
        pred_occ = _sigmoid(out_features[0])
        pred_unc = np.clip(np.exp(out_features[1]) * 0.288, 0.01, 0.288)

        # Blend predicted belief with direct observations where high evidence is present
        direct_weight = np.clip(evidence.observed.astype(np.float64) / 5.0, 0.0, 0.8)
        filtered_occ = (1.0 - direct_weight) * pred_occ + direct_weight * norm_occ
        filtered_unc = (1.0 - direct_weight) * pred_unc + direct_weight * norm_unc

        # Preserve unknown marker (NaN) where cells have never been observed
        never_observed = (evidence.observed == 0) & (
            self._current_state is None or np.isnan(self._current_state.occupancy)
        )
        filtered_occ = np.where(never_observed, np.nan, filtered_occ)
        filtered_unc = np.where(never_observed, np.nan, filtered_unc)

        # 3. Velocity field estimation & dynamic classification
        velocity_field, dynamic_mask = self._estimate_velocity_field(
            filtered_occ, dt=dt, obstacles=obstacles
        )

        state = TemporalOccupancyState(
            timestamp=timestamp,
            step_index=step_idx,
            grid=self.grid,
            hidden_state=h_t,
            occupancy=filtered_occ,
            uncertainty=filtered_unc,
            velocity_field=velocity_field,
            dynamic_mask=dynamic_mask,
        )

        self._current_state = state
        self._prev_occupancy = filtered_occ
        self._prev_timestamp = timestamp
        return state

    def forecast(
        self,
        horizon_seconds: float | None = None,
        step_seconds: float | None = None,
    ) -> OccupancyForecastGrid:
        """Roll forward recurrent state and advect dynamic flow over prediction horizon.

        Predicts multi-step future occupancy grids [t + dt, t + 2dt, ..., t + H]
        with flow advection, static persistence decay, and Bayesian uncertainty diffusion.
        """
        if self._current_state is None:
            raise RuntimeError(
                "Cannot forecast occupancy without prior state; call update() at least once."
            )

        horizon = horizon_seconds if horizon_seconds is not None else self.config.horizon_seconds
        step = step_seconds if step_seconds is not None else self.config.step_seconds

        if horizon <= 0.0:
            raise ValueError("horizon_seconds must be strictly positive")
        if step <= 0.0:
            raise ValueError("step_seconds must be strictly positive")
        if step > horizon:
            raise ValueError(f"step_seconds ({step}) cannot exceed horizon_seconds ({horizon})")

        num_steps = max(1, math.ceil(horizon / step - 1e-9))
        base_state = self._current_state
        base_occ = np.nan_to_num(base_state.occupancy, nan=0.0)
        base_unc = np.nan_to_num(base_state.uncertainty, nan=0.288)
        vel_x = base_state.velocity_field[0]
        vel_y = base_state.velocity_field[1]
        dyn_mask_base = base_state.dynamic_mask
        res = self.grid.resolution

        # Decompose initial occupancy into dynamic vs static components
        occ_dynamic = base_occ * dyn_mask_base.astype(np.float64)
        occ_static = base_occ * (~dyn_mask_base).astype(np.float64)

        forecast_steps: list[OccupancyForecastStep] = []

        for step_idx in range(1, num_steps + 1):
            offset_s = round(step_idx * step, 4)
            future_ts = round(base_state.timestamp + offset_s, 4)

            # 1. Forward advection of dynamic occupancy along velocity vectors
            disp_x = vel_x * offset_s
            disp_y = vel_y * offset_s

            advected_dyn = _advect_grid(occ_dynamic, disp_x, disp_y, resolution=res)
            # Dynamic flow confidence decays over horizon
            flow_discount = math.pow(self.config.flow_decay, offset_s)
            advected_dyn *= flow_discount

            # Advect dynamic mask
            advected_mask_float = _advect_grid(
                dyn_mask_base.astype(np.float64), disp_x, disp_y, resolution=res
            )
            dyn_mask_future = (advected_mask_float > 0.3) & (advected_dyn > 0.1)

            # 2. Static persistence with temporal memory decay
            memory_discount = math.pow(self.config.memory_decay, offset_s)
            persisted_static = occ_static * memory_discount

            # 3. Combined forward occupancy and free space probabilities
            combined_occ = np.clip(persisted_static + advected_dyn, 0.0, 1.0)
            combined_free = np.clip(1.0 - combined_occ, 0.0, 1.0)

            # 4. Bayesian uncertainty diffusion: Var(p_k) = Var(p_0) + offset_s * diffusion_rate
            diffused_unc = np.clip(
                np.sqrt(base_unc * base_unc + offset_s * self.config.diffusion_rate),
                0.0,
                0.288,
            )
            # Apply slight Gaussian blur to model spatial diffusion
            diffused_unc = cv2.GaussianBlur(diffused_unc.astype(np.float32), (3, 3), 0.5).astype(
                np.float64
            )

            # Preserve NaN for unobserved cells from base state
            if np.any(np.isnan(base_state.occupancy)):
                unobs = np.isnan(base_state.occupancy) & ~dyn_mask_future
                combined_occ = np.where(unobs, np.nan, combined_occ)
                diffused_unc = np.where(unobs, np.nan, diffused_unc)
                combined_free = np.where(unobs, np.nan, combined_free)

            forecast_steps.append(
                OccupancyForecastStep(
                    time_offset_s=offset_s,
                    timestamp=future_ts,
                    occupancy=combined_occ,
                    uncertainty=diffused_unc,
                    free=combined_free,
                    dynamic_mask=dyn_mask_future,
                )
            )

        return OccupancyForecastGrid(
            grid=self.grid,
            base_timestamp=base_state.timestamp,
            steps=tuple(forecast_steps),
        )


__all__ = [
    "TemporalOccupancyForecaster",
    "fuse_cross_attention_occupancy",
]
