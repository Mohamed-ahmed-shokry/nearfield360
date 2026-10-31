"""Domain data models and schemas for synthetic radar and ultrasonic sensor simulation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SensorPose(BaseModel):
    """2D vehicle-frame mounting pose of an automotive sensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float = Field(..., allow_inf_nan=False)
    y_m: float = Field(..., allow_inf_nan=False)
    yaw_rad: float = Field(default=0.0, allow_inf_nan=False)


class RadarTarget(BaseModel):
    """Detected or simulated radar reflector target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    range_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    radial_velocity_mps: float = Field(default=0.0, allow_inf_nan=False)
    azimuth_rad: float = Field(default=0.0, allow_inf_nan=False)
    rcs_dbm2: float = Field(default=10.0, allow_inf_nan=False)
    snr_db: float = Field(default=15.0, allow_inf_nan=False)


class RadarPointCloud(BaseModel):
    """Collection of detected radar reflection point targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: list[RadarTarget] = Field(default_factory=list)
    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    sensor_id: str = Field(default="front_radar_77ghz")
    noise_floor_db: float = Field(default=-90.0, allow_inf_nan=False)


class RangeDopplerMap(BaseModel):
    """2D Range-Doppler power spectrum matrix generated via 2D FFT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matrix_db: list[list[float]] = Field(default_factory=list)
    range_bins_m: list[float] = Field(default_factory=list)
    doppler_bins_mps: list[float] = Field(default_factory=list)
    min_power_db: float = Field(default=-100.0, allow_inf_nan=False)
    max_power_db: float = Field(default=0.0, allow_inf_nan=False)


class UltrasonicEcho(BaseModel):
    """Acoustic reflection echo captured by an ultrasonic transducer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensor_id: str = Field(...)
    sensor_pose: SensorPose = Field(...)
    range_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    echo_amplitude: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    in_detection_cone: bool = True


class UltrasonicArrayFrame(BaseModel):
    """Perimeter ultrasonic transducer array echo frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    echoes: list[UltrasonicEcho] = Field(default_factory=list)
    ambient_temp_c: float = Field(default=20.0, ge=-40.0, le=60.0, allow_inf_nan=False)
    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class FusedSensorObstacle(BaseModel):
    """Cross-modal fused obstacle estimate combining radar, ultrasonic, and vision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obstacle_id: str = Field(...)
    x_m: float = Field(..., allow_inf_nan=False)
    y_m: float = Field(..., allow_inf_nan=False)
    radial_velocity_mps: float = Field(default=0.0, allow_inf_nan=False)
    min_range_m: float = Field(..., ge=0.0, allow_inf_nan=False)
    source_modalities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)


class SensorSimulationReport(BaseModel):
    """Aggregated report of synthetic sensor simulation and cross-modal fusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    radar_points_count: int = Field(default=0, ge=0)
    radar_point_cloud: RadarPointCloud = Field(...)
    ultrasonic_echoes_count: int = Field(default=0, ge=0)
    ultrasonic_frame: UltrasonicArrayFrame = Field(...)
    fused_obstacles_count: int = Field(default=0, ge=0)
    fused_obstacles: list[FusedSensorObstacle] = Field(default_factory=list)
