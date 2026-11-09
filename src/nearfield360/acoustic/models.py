"""Domain models, classifications, and telemetry schemas for acoustic sensing."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SirenType(StrEnum):
    """Categorization of emergency acoustic siren frequency modulation profiles."""

    NONE = "none"
    WAIL = "wail"
    YELP = "yelp"
    HI_LO = "hi_lo"
    HORN = "horn"


class DopplerState(StrEnum):
    """Radial Doppler approach or recession state of emergency acoustic source."""

    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    APPROACHING = "approaching"
    RECEDING = "receding"


class EmergencyYieldState(StrEnum):
    """Vehicle operational yield interlock states for emergency vehicle compliance."""

    CLEAR = "clear"
    MONITORING = "monitoring"
    YIELD_PREPARED = "yield_prepared"
    EMERGENCY_STOP_ACTIVE = "emergency_stop_active"


class CameraSector(StrEnum):
    """Automotive surround-view camera sector associated with acoustic bearing."""

    FRONT = "FV"
    RIGHT = "MVR"
    REAR = "RV"
    LEFT = "MVL"


def azimuth_to_camera_sector(azimuth_deg: float) -> CameraSector:
    """Map planar azimuth angle (-180 to +180 deg) to corresponding surround camera sector.

    Convention:
      - 0 deg is Vehicle Front (+X)
      - +90 deg is Vehicle Right (+Y)
      - -90 deg is Vehicle Left (-Y)
      - +/-180 deg is Vehicle Rear (-X)
    """
    # Normalize to [-180, 180)
    norm_deg = (azimuth_deg + 180.0) % 360.0 - 180.0
    if -45.0 <= norm_deg < 45.0:
        return CameraSector.FRONT
    if 45.0 <= norm_deg < 135.0:
        return CameraSector.RIGHT
    if -135.0 <= norm_deg < -45.0:
        return CameraSector.LEFT
    return CameraSector.REAR


class MicrophoneSpec(BaseModel):
    """Spatial 3D Cartesian coordinates of an acoustic microphone transducer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mic_id: str = Field(..., description="Unique microphone channel identifier.")
    x_m: float = Field(..., description="Forward offset in vehicle frame (meters).")
    y_m: float = Field(..., description="Lateral offset in vehicle frame (meters).")
    z_m: float = Field(default=0.0, description="Vertical offset in vehicle frame (meters).")


class SirenDetectionResult(BaseModel):
    """Acoustic frequency analysis and modulation classification result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detected: bool = Field(default=False, description="True if siren signature verified.")
    siren_type: SirenType = Field(default=SirenType.NONE)
    fundamental_freq_hz: float = Field(default=0.0, ge=0.0, le=10000.0)
    snr_db: float = Field(default=0.0, description="Estimated signal-to-noise ratio in dB.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    harmonic_energy_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    cycle_rate_hz: float = Field(default=0.0, ge=0.0, le=50.0)


class AcousticDoAResult(BaseModel):
    """Direction-of-Arrival (DoA) triangulation result from multi-microphone array."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    azimuth_deg: float = Field(
        ..., ge=-180.0, le=180.0, description="Planar azimuth (-180 to +180 deg)."
    )
    elevation_deg: float = Field(default=0.0, ge=-90.0, le=90.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sound_pressure_level_db: float = Field(default=60.0, ge=0.0, le=160.0)
    primary_pair: tuple[str, str] = Field(
        ..., description="Microphone pair providing highest cross-correlation."
    )
    time_diff_s: float = Field(
        default=0.0, description="Estimated Time Difference of Arrival (TDoA) in seconds."
    )


class AudioVisualSectorMatch(BaseModel):
    """Cross-modal correlation between acoustic bearing and surround camera sector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    azimuth_deg: float = Field(..., ge=-180.0, le=180.0)
    camera_sector: CameraSector
    visual_confirmation: bool = Field(default=False)
    fused_confidence: float = Field(..., ge=0.0, le=1.0)
    yield_action_recommended: bool = Field(default=False)


class AcousticReport(BaseModel):
    """Unified telemetry and safety audit report for in-vehicle acoustic sensing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(..., ge=0.0)
    siren_detection: SirenDetectionResult
    doa: AcousticDoAResult | None = None
    doppler: DopplerState = Field(default=DopplerState.UNKNOWN)
    estimated_radial_speed_mps: float = Field(default=0.0)
    yield_state: EmergencyYieldState = Field(default=EmergencyYieldState.CLEAR)
    sector_match: AudioVisualSectorMatch | None = None
    summary_message: str = Field(default="Acoustic monitoring clear.")

    def is_emergency_active(self) -> bool:
        """Return True if an approaching emergency siren requires yield intervention."""
        return (
            self.siren_detection.detected
            and self.doppler in (DopplerState.APPROACHING, DopplerState.STATIONARY)
            and self.yield_state
            in (EmergencyYieldState.YIELD_PREPARED, EmergencyYieldState.EMERGENCY_STOP_ACTIVE)
        )
