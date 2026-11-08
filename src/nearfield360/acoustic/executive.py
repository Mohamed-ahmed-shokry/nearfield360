"""Acoustic sensing executive, Doppler velocity tracking, and emergency yield interlock."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from nearfield360.acoustic.detector import SirenDetector
from nearfield360.acoustic.doa import AcousticDoAEstimator
from nearfield360.acoustic.models import (
    AcousticReport,
    AudioVisualSectorMatch,
    CameraSector,
    DopplerState,
    EmergencyYieldState,
    SirenDetectionResult,
    SirenType,
    azimuth_to_camera_sector,
)

if TYPE_CHECKING:
    from nearfield360.config import AcousticConfig
    from nearfield360.safety.supervisor import SafetySupervisor


class AcousticExecutive:
    """Orchestrates acoustic siren detection, DoA estimation, and emergency vehicle yield."""

    def __init__(
        self,
        config: AcousticConfig,
        safety_supervisor: SafetySupervisor | None = None,
    ) -> None:
        self.config = config
        self.safety_supervisor = safety_supervisor
        self.detector = SirenDetector(config)
        self.doa_estimator = AcousticDoAEstimator(config)

        # Sliding history for Doppler velocity estimation: deque of (timestamp_s, fund_freq_hz)
        self._freq_history: deque[tuple[float, float]] = deque(maxlen=20)
        self._current_yield_state = EmergencyYieldState.CLEAR
        self._nominal_freq_anchor: float | None = None

    def process_audio(
        self,
        multichannel_audio: np.ndarray,
        timestamp_s: float,
        visual_camera_sectors: list[CameraSector] | None = None,
        ego_speed_mps: float = 0.0,
        estimated_distance_m: float = 25.0,
    ) -> AcousticReport:
        """Process multichannel audio buffer, update Doppler tracking, and evaluate yield state."""
        visual_sectors = visual_camera_sectors or []

        # 1. Siren detection on primary channel
        primary_audio = multichannel_audio[0] if multichannel_audio.ndim > 1 else multichannel_audio
        siren_res = self.detector.analyze_signal(primary_audio)

        # 2. Direction-of-Arrival (DoA) triangulation
        doa_res = None
        if siren_res.detected and multichannel_audio.ndim > 1 and multichannel_audio.shape[0] >= 2:
            doa_res = self.doa_estimator.estimate_doa(multichannel_audio)

        # 3. Doppler radial velocity tracking
        doppler_state, radial_speed_mps = self._update_doppler(
            siren_res.fundamental_freq_hz, timestamp_s
        )

        # 4. Audio-Visual sector correlation
        sector_match = None
        if doa_res is not None:
            camera_sec = azimuth_to_camera_sector(doa_res.azimuth_deg)
            has_visual = camera_sec in visual_sectors

            fused_conf = min(
                1.0,
                round(
                    doa_res.confidence * 0.50
                    + (0.35 if has_visual else 0.0)
                    + siren_res.confidence * 0.25,
                    4,
                ),
            )
            sector_match = AudioVisualSectorMatch(
                azimuth_deg=doa_res.azimuth_deg,
                camera_sector=camera_sec,
                visual_confirmation=has_visual,
                fused_confidence=fused_conf,
                yield_action_recommended=False,
            )

        # 5. Emergency Yield State Machine
        new_yield_state = self._evaluate_yield_state(
            siren_res=siren_res,
            doppler=doppler_state,
            sector_match=sector_match,
            distance_m=estimated_distance_m,
        )
        self._current_yield_state = new_yield_state

        if sector_match is not None:
            yield_rec = new_yield_state in (
                EmergencyYieldState.YIELD_PREPARED,
                EmergencyYieldState.EMERGENCY_STOP_ACTIVE,
            )
            sector_match = sector_match.model_copy(update={"yield_action_recommended": yield_rec})

        # 6. Safety supervisor interlock escalation
        if (
            new_yield_state == EmergencyYieldState.EMERGENCY_STOP_ACTIVE
            and self.config.enable_yield_interlock
            and self.safety_supervisor is not None
        ):
            self.safety_supervisor.trigger_emergency_stop(
                reason="Acoustic emergency vehicle yield interlock",
                timestamp_s=timestamp_s,
            )

        # 7. Formulate summary telemetry message
        if new_yield_state == EmergencyYieldState.EMERGENCY_STOP_ACTIVE:
            s_type = siren_res.siren_type.value.upper()
            sec_name = sector_match.camera_sector.value if sector_match else "UNKNOWN"
            msg = (
                f"EMERGENCY VEHICLE INTERLOCK ACTIVE: Approaching {s_type} siren on "
                f"sector {sec_name} at {radial_speed_mps:.1f} m/s. Safe stop commanded."
            )
        elif new_yield_state == EmergencyYieldState.YIELD_PREPARED:
            s_type = siren_res.siren_type.value.upper()
            msg = (
                f"EMERGENCY VEHICLE DETECTED: {s_type} siren at distance "
                f"{estimated_distance_m:.1f} m. Preparing yield."
            )
        elif new_yield_state == EmergencyYieldState.MONITORING:
            msg = f"Acoustic siren detected ({siren_res.siren_type.value}). Monitoring trajectory."
        else:
            msg = "Acoustic field clear. No emergency sirens detected."

        return AcousticReport(
            timestamp_s=timestamp_s,
            siren_detection=siren_res,
            doa=doa_res,
            doppler=doppler_state,
            estimated_radial_speed_mps=round(radial_speed_mps, 2),
            yield_state=new_yield_state,
            sector_match=sector_match,
            summary_message=msg,
        )

    def _update_doppler(
        self,
        current_freq_hz: float,
        timestamp_s: float,
    ) -> tuple[DopplerState, float]:
        """Estimate Doppler frequency shift and radial velocity."""
        if current_freq_hz <= 0.0:
            self._freq_history.clear()
            self._nominal_freq_anchor = None
            return DopplerState.UNKNOWN, 0.0

        self._freq_history.append((timestamp_s, current_freq_hz))
        if self._nominal_freq_anchor is None:
            self._nominal_freq_anchor = current_freq_hz

        if len(self._freq_history) < 3:
            return DopplerState.STATIONARY, 0.0

        # Frequency shift relative to baseline anchor
        # Positive delta means observed frequency higher than nominal -> approaching
        delta_f = current_freq_hz - self._nominal_freq_anchor
        c = self.config.sound_speed_mps
        # Doppler formula: v_radial = c * (delta_f / f0)
        radial_speed_mps = c * (delta_f / (self._nominal_freq_anchor + 1e-9))
        radial_speed_mps = max(-50.0, min(50.0, radial_speed_mps))

        if radial_speed_mps > 1.5:
            return DopplerState.APPROACHING, radial_speed_mps
        if radial_speed_mps < -1.5:
            return DopplerState.RECEDING, radial_speed_mps
        return DopplerState.STATIONARY, radial_speed_mps

    def _evaluate_yield_state(
        self,
        siren_res: SirenDetectionResult,
        doppler: DopplerState,
        sector_match: AudioVisualSectorMatch | None,
        distance_m: float,
    ) -> EmergencyYieldState:
        """Evaluate operational safety yield state transitions."""
        if not siren_res.detected or siren_res.siren_type == SirenType.NONE:
            return EmergencyYieldState.CLEAR

        has_visual = sector_match is not None and sector_match.visual_confirmation

        # Approaching or close emergency vehicle
        if doppler == DopplerState.APPROACHING:
            if distance_m <= self.config.yield_distance_threshold_m or has_visual:
                return EmergencyYieldState.EMERGENCY_STOP_ACTIVE
            return EmergencyYieldState.YIELD_PREPARED

        if doppler == DopplerState.STATIONARY:
            if distance_m <= self.config.yield_distance_threshold_m:
                return EmergencyYieldState.YIELD_PREPARED
            return EmergencyYieldState.MONITORING

        # Receding or unknown
        return EmergencyYieldState.MONITORING
