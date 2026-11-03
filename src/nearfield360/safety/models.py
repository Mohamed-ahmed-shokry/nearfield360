"""Domain data models and schemas for ISO 26262 ASIL-D functional safety supervisor."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SafetyState(StrEnum):
    """Operational state of the vehicle electronic safety architecture (ASIL-D)."""

    BOOT_SELF_TEST = "boot_self_test"
    NORMAL_OPERATION = "normal_operation"
    DEGRADED_PERCEPTION = "degraded_perception"
    EMERGENCY_SAFE_STOP = "emergency_safe_stop"
    FAIL_SILENT_LATCH = "fail_silent_latch"


class ECUNodeId(StrEnum):
    """Identifiers of monitored electronic control units and software pipelines."""

    PERCEPTION_SURROUND = "perception_surround"
    RADAR_TRANSCEIVER = "radar_transceiver"
    ULTRASONIC_ARRAY = "ultrasonic_array"
    TRAJECTORY_PLANNER = "trajectory_planner"
    CAN_GATEWAY = "can_gateway"
    VEHICLE_CONTROLLER = "vehicle_controller"
    TELEOP_SUPERVISOR = "teleop_supervisor"


class HeartbeatStatus(StrEnum):
    """Real-time health status of a monitored cyclic ECU heartbeat."""

    HEALTHY = "healthy"
    WARN_JITTER = "warn_jitter"
    MISSED = "missed"
    EXPIRED = "expired"


class ViolationSeverity(StrEnum):
    """Severity classification of sensor or kinematic plausibility violations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class HeartbeatPing(BaseModel):
    """Cyclic heartbeat ping transmitted by an ECU node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: ECUNodeId = Field(...)
    sequence_num: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)


class NodeHealthStatus(BaseModel):
    """Watchdog tracking record for a single ECU node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: ECUNodeId = Field(...)
    last_heartbeat_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    sequence_num: int = Field(default=0, ge=0)
    status: HeartbeatStatus = Field(default=HeartbeatStatus.HEALTHY)
    latency_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    jitter_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    consecutive_misses: int = Field(default=0, ge=0)


class PlausibilityViolation(BaseModel):
    """Kinematic or sensor data invariant failure detected at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    violation_type: str = Field(...)
    description: str = Field(...)
    measured_value: float = Field(..., allow_inf_nan=False)
    limit_value: float = Field(..., allow_inf_nan=False)
    severity: ViolationSeverity = Field(default=ViolationSeverity.CRITICAL)
    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class SafetyTransitionRecord(BaseModel):
    """Audit log entry of a state transition in the safety supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_state: SafetyState = Field(...)
    to_state: SafetyState = Field(...)
    reason: str = Field(...)
    timestamp_s: float = Field(..., ge=0.0, allow_inf_nan=False)


class ActuatorSafetyOverride(BaseModel):
    """Safe control actuation override applied when safety limits are breached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_override_active: bool = Field(default=False)
    target_deceleration_mps2: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    max_permitted_speed_mps: float = Field(default=10.0, ge=0.0, allow_inf_nan=False)
    steering_clamped: bool = Field(default=False)
    emergency_hazard_lights: bool = Field(default=False)


class SafetySupervisorReport(BaseModel):
    """Comprehensive ASIL-D safety supervisor audit report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    active_state: SafetyState = Field(default=SafetyState.NORMAL_OPERATION)
    node_healths: dict[str, NodeHealthStatus] = Field(default_factory=dict)
    active_violations: list[PlausibilityViolation] = Field(default_factory=list)
    transition_history: list[SafetyTransitionRecord] = Field(default_factory=list)
    actuator_override: ActuatorSafetyOverride = Field(default_factory=ActuatorSafetyOverride)
