"""2D Cell-Averaging Constant False Alarm Rate (2D CA-CFAR) peak detection engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from nearfield360.sensor.models import RadarPointCloud, RadarTarget

if TYPE_CHECKING:
    from nearfield360.config import SensorConfig


class CACFARDetector:
    """2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector for Range-Doppler maps."""

    def __init__(
        self,
        guard_cells: int = 2,
        train_cells: int = 4,
        threshold_factor: float = 1.25,
        min_snr_db: float = 15.0,
    ) -> None:
        self.guard = guard_cells
        self.train = train_cells
        self.threshold_factor = threshold_factor
        self.min_snr_db = min_snr_db

    @classmethod
    def from_config(cls, config: SensorConfig) -> CACFARDetector:
        """Instantiate detector from SensorConfig parameters."""
        return cls(
            guard_cells=config.cfar_guard_cells,
            train_cells=config.cfar_train_cells,
            threshold_factor=config.cfar_threshold_factor,
        )

    def detect(
        self,
        power_db: np.ndarray,
        range_bins_m: np.ndarray | list[float],
        doppler_bins_mps: np.ndarray | list[float],
        timestamp_s: float = 0.0,
        sensor_id: str = "front_radar_77ghz",
    ) -> RadarPointCloud:
        """Run 2D CA-CFAR detection over a Range-Doppler power matrix.

        Args:
            power_db: 2D array [num_doppler_bins, num_range_bins] in dB.
            range_bins_m: 1D array of metric range bin centers.
            doppler_bins_mps: 1D array of Doppler radial velocity bin centers.
            timestamp_s: Timestamp of capture.
            sensor_id: Identifier of radar transceiver.

        Returns:
            RadarPointCloud containing detected point targets with SNR and RCS estimates.
        """
        power_db_arr = np.asarray(power_db, dtype=np.float32)
        range_bins = np.asarray(range_bins_m, dtype=np.float64)
        doppler_bins = np.asarray(doppler_bins_mps, dtype=np.float64)

        # Convert dB to linear power for unbiased cell averaging
        power_linear = np.power(10.0, power_db_arr / 10.0)

        # 2D Kernel dimensions (width along range, height along Doppler)
        g_w = 2 * self.guard + 1
        t_w = 2 * (self.guard + self.train) + 1

        # Use boxFilter to compute sliding 2D window sums
        outer_sum = cv2.boxFilter(
            power_linear,
            ddepth=-1,
            ksize=(t_w, t_w),
            borderType=cv2.BORDER_REFLECT,
            normalize=False,
        )
        guard_sum = cv2.boxFilter(
            power_linear,
            ddepth=-1,
            ksize=(g_w, g_w),
            borderType=cv2.BORDER_REFLECT,
            normalize=False,
        )

        train_cells_count = float(t_w * t_w - g_w * g_w)
        training_sum = np.maximum(outer_sum - guard_sum, 1e-12)
        noise_power = training_sum / train_cells_count
        cfar_threshold = noise_power * self.threshold_factor

        # Local peak detection via 3x3 morphological dilation
        kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(power_linear, kernel_3x3)
        is_local_max = power_linear >= dilated

        # Candidate detections: above CFAR threshold, is local peak, and above noise floor
        detection_mask = (power_linear > cfar_threshold) & is_local_max

        # Exclude border margins where filter window overlaps boundary
        margin = self.guard + self.train
        detection_mask[:margin, :] = False
        detection_mask[-margin:, :] = False
        detection_mask[:, :margin] = False
        detection_mask[:, -margin:] = False

        # Extract target coordinates
        doppler_indices, range_indices = np.where(detection_mask)

        detected_targets: list[RadarTarget] = []
        noise_floor_db = float(10.0 * np.log10(np.mean(noise_power)))

        for d_idx, r_idx in zip(doppler_indices, range_indices, strict=False):
            cell_p_db = float(power_db_arr[d_idx, r_idx])
            cell_noise_db = float(10.0 * np.log10(max(1e-12, noise_power[d_idx, r_idx])))
            snr_db = cell_p_db - cell_noise_db

            if snr_db < self.min_snr_db:
                continue

            r_m = float(range_bins[r_idx])
            vr_mps = float(doppler_bins[d_idx])

            # Radar cross section estimate based on SNR and range R^2
            rcs_est = float(np.clip(snr_db - 20.0 + 10.0 * np.log10(max(1.0, r_m)), -10.0, 40.0))

            detected_targets.append(
                RadarTarget(
                    range_m=r_m,
                    radial_velocity_mps=vr_mps,
                    azimuth_rad=0.0,
                    rcs_dbm2=rcs_est,
                    snr_db=round(snr_db, 2),
                )
            )

        # Sort targets by range
        detected_targets.sort(key=lambda t: t.range_m)

        return RadarPointCloud(
            points=detected_targets,
            timestamp_s=timestamp_s,
            sensor_id=sensor_id,
            noise_floor_db=round(noise_floor_db, 2),
        )
