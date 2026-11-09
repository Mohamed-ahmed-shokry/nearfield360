"""Multi-factor environmental and sensor health metrics evaluation engine."""

from __future__ import annotations

import numpy as np

from nearfield360.config import ODDConfig
from nearfield360.odd.models import (
    IlluminationCondition,
    ODDMetrics,
    ODDState,
    PrecipitationCondition,
    SensorHealthStatus,
    SurfaceCondition,
    odd_state_severity,
)


class ODDEvaluator:
    """Evaluates multi-modal environmental and sensor telemetry against ODD boundaries."""

    def __init__(self, config: ODDConfig | None = None) -> None:
        self.config = config or ODDConfig()

    def estimate_frame_illumination(
        self, image: np.ndarray
    ) -> tuple[float, float]:
        """Estimate ambient lux and saturation glare ratio from an image (H, W, 3) or (H, W).

        Returns:
            (estimated_lux, glare_ratio)
        """
        if image.size == 0:
            return 0.0, 0.0

        if image.ndim == 3 and image.shape[2] == 3:
            # Grayscale luminance Y = 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image.astype(float)

        mean_luma = float(np.mean(gray))
        glare_count = int(np.count_nonzero(gray >= 245.0))
        glare_ratio = float(glare_count / max(1, gray.size))

        # Nominal mapping from 8-bit luma to lux proxy [0.1, 10000.0]
        # Using quadratic optical scaling
        estimated_lux = float(max(0.01, (mean_luma / 255.0) ** 2 * 2500.0))
        return estimated_lux, glare_ratio

    def classify_illumination(
        self, lux: float, glare_ratio: float
    ) -> tuple[IlluminationCondition, ODDState, list[str]]:
        """Classify illumination condition and associated ODD tier."""
        limiters: list[str] = []
        if glare_ratio >= self.config.max_glare_ratio:
            limiters.append(f"severe_glare_ratio_{glare_ratio:.2f}")
            return IlluminationCondition.GLARE_DAZZLE, ODDState.CRITICAL_RESTRICTED, limiters

        if lux < self.config.min_ambient_lux:
            limiters.append(f"low_ambient_lux_{lux:.1f}")
            if lux < 0.1:
                return IlluminationCondition.NIGHT, ODDState.CRITICAL_RESTRICTED, limiters
            return IlluminationCondition.NIGHT, ODDState.ADVISORY_DEGRADED, limiters

        if lux < 25.0:
            return IlluminationCondition.TWILIGHT, ODDState.NOMINAL, limiters

        return IlluminationCondition.DAYLIGHT, ODDState.NOMINAL, limiters

    def classify_precipitation(
        self, rain_rate_mmph: float
    ) -> tuple[PrecipitationCondition, ODDState, list[str]]:
        """Classify atmospheric precipitation condition and associated ODD tier."""
        limiters: list[str] = []
        if rain_rate_mmph <= 0.1:
            return PrecipitationCondition.CLEAR, ODDState.NOMINAL, limiters

        if rain_rate_mmph <= 7.5:
            limiters.append(f"light_rain_{rain_rate_mmph:.1f}mmph")
            return PrecipitationCondition.LIGHT_RAIN, ODDState.ADVISORY_DEGRADED, limiters

        if rain_rate_mmph <= 25.0:
            limiters.append(f"heavy_rain_{rain_rate_mmph:.1f}mmph")
            return PrecipitationCondition.HEAVY_RAIN, ODDState.CRITICAL_RESTRICTED, limiters

        limiters.append(f"torrential_precipitation_{rain_rate_mmph:.1f}mmph")
        return PrecipitationCondition.HEAVY_RAIN, ODDState.ODD_BREACH_MRM, limiters

    def classify_surface(
        self, friction_coeff: float
    ) -> tuple[SurfaceCondition, ODDState, list[str]]:
        """Classify road/parking pavement condition and friction ODD tier."""
        limiters: list[str] = []
        if friction_coeff < self.config.critical_friction_coeff:
            limiters.append(f"critical_low_friction_{friction_coeff:.2f}")
            return SurfaceCondition.ICY_SNOW, ODDState.ODD_BREACH_MRM, limiters

        if friction_coeff < self.config.min_friction_coeff:
            limiters.append(f"aquaplaning_risk_friction_{friction_coeff:.2f}")
            return SurfaceCondition.PUDDLE_AQUAPLANING, ODDState.CRITICAL_RESTRICTED, limiters

        if friction_coeff < 0.70:
            limiters.append(f"wet_surface_friction_{friction_coeff:.2f}")
            return SurfaceCondition.WET, ODDState.ADVISORY_DEGRADED, limiters

        return SurfaceCondition.DRY, ODDState.NOMINAL, limiters

    def classify_connectivity(
        self, v2x_latency_ms: float, packet_loss_ratio: float
    ) -> tuple[ODDState, list[str]]:
        """Classify V2X infrastructure communications envelope."""
        limiters: list[str] = []
        if (
            v2x_latency_ms >= self.config.critical_v2x_latency_ms
            or packet_loss_ratio >= 0.35
        ):
            limiters.append(
                f"critical_v2x_dropout_lat_{v2x_latency_ms:.0f}ms_loss_{packet_loss_ratio:.2f}"
            )
            return ODDState.ODD_BREACH_MRM, limiters

        if (
            v2x_latency_ms >= self.config.max_v2x_latency_ms
            or packet_loss_ratio >= 0.12
        ):
            limiters.append(
                f"v2x_high_latency_{v2x_latency_ms:.0f}ms_loss_{packet_loss_ratio:.2f}"
            )
            return ODDState.CRITICAL_RESTRICTED, limiters

        return ODDState.NOMINAL, limiters

    def classify_sensor_health(
        self, sensors: dict[str, SensorHealthStatus]
    ) -> tuple[ODDState, list[str]]:
        """Classify multi-camera lens cleanliness, occlusion, and hardware health."""
        if not sensors:
            return ODDState.NOMINAL, []

        limiters: list[str] = []
        critical_count = 0
        advisory_count = 0

        for sid, s in sensors.items():
            if not s.is_healthy:
                critical_count += 1
                limiters.append(f"sensor_{sid}_faulted")
            elif s.occlusion_ratio >= self.config.critical_camera_occlusion_ratio:
                critical_count += 1
                limiters.append(f"sensor_{sid}_severe_occlusion_{s.occlusion_ratio:.2f}")
            elif s.occlusion_ratio >= self.config.max_camera_occlusion_ratio:
                advisory_count += 1
                limiters.append(f"sensor_{sid}_lens_soiled_{s.occlusion_ratio:.2f}")

        if critical_count >= 2:
            limiters.append(f"multiple_critical_sensors_{critical_count}")
            return ODDState.ODD_BREACH_MRM, limiters

        if critical_count == 1:
            return ODDState.CRITICAL_RESTRICTED, limiters

        if advisory_count >= 1:
            return ODDState.ADVISORY_DEGRADED, limiters

        return ODDState.NOMINAL, limiters

    def evaluate_instantaneous(
        self,
        metrics: ODDMetrics,
        sensors: dict[str, SensorHealthStatus] | None = None,
    ) -> tuple[
        ODDState,
        list[str],
        IlluminationCondition,
        PrecipitationCondition,
        SurfaceCondition,
    ]:
        """Compute instantaneous ODD tier and collect all active limiting factors."""
        limiters: list[str] = []
        candidate_states: list[ODDState] = [ODDState.NOMINAL]

        # 1. Illumination
        illum_cond, illum_state, illum_lims = self.classify_illumination(
            metrics.ambient_lux, metrics.glare_ratio
        )
        candidate_states.append(illum_state)
        limiters.extend(illum_lims)

        # 2. Precipitation
        precip_cond, precip_state, precip_lims = self.classify_precipitation(
            metrics.precipitation_rate_mmph
        )
        candidate_states.append(precip_state)
        limiters.extend(precip_lims)

        # 3. Surface Friction
        surf_cond, surf_state, surf_lims = self.classify_surface(metrics.friction_coeff)
        candidate_states.append(surf_state)
        limiters.extend(surf_lims)

        # 4. Connectivity
        conn_state, conn_lims = self.classify_connectivity(
            metrics.v2x_latency_ms, metrics.v2x_packet_loss_ratio
        )
        candidate_states.append(conn_state)
        limiters.extend(conn_lims)

        # 5. Sensor Health
        if sensors:
            sens_state, sens_lims = self.classify_sensor_health(sensors)
            candidate_states.append(sens_state)
            limiters.extend(sens_lims)

        # Composite severity: maximum over all factors
        worst_state = max(candidate_states, key=odd_state_severity)
        return worst_state, limiters, illum_cond, precip_cond, surf_cond
