from pathlib import Path

import pytest
from pydantic import ValidationError

from nearfield360.config import MAX_CONFIG_BYTES, ConfigurationError, load_config


def test_load_config_uses_typed_defaults() -> None:
    config = load_config()

    assert config.paths.dataset_root is None
    assert config.paths.output_root == Path("outputs")
    assert config.runtime.device == "auto"
    assert config.runtime.seed == 42
    assert config.logging.level == "INFO"
    assert config.logging.structured is False
    assert config.geometry.theta_max == 2.2
    assert config.geometry.ground_z == 0.0
    assert config.geometry.max_distance == 15.0
    assert (config.bev.x_min, config.bev.x_max) == (-6.0, 10.0)
    assert (config.bev.y_min, config.bev.y_max) == (-6.0, 6.0)
    assert config.bev.resolution == 0.05
    assert config.risk.front_length == 3.0
    assert config.risk.rear_length == 3.0
    assert config.risk.half_width == 0.9
    assert config.risk.start_x == 0.0
    assert config.risk.rear_start_x == 0.0
    assert config.risk.lateral_width == 0.8
    assert (config.risk.vehicle_x_min, config.risk.vehicle_x_max) == (-2.0, 2.0)
    assert config.risk.near_radius == 0.5
    assert config.risk.warning_radius == 1.5
    assert config.risk.danger_occupancy == 0.5
    assert config.health.soiling_threshold == 0.5
    assert config.health.blur_threshold == 100.0
    assert config.health.min_brightness == 15.0
    assert config.health.max_brightness == 240.0
    assert config.health.discount_factor == 0.5
    assert config.tracking.dt == 0.1
    assert config.tracking.max_distance == 2.5
    assert config.tracking.min_hits == 3
    assert config.tracking.max_age == 5
    assert config.tracking.forecast_horizon_s == 3.0


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "paths:\n  dataset_root: fixtures/woodscape\n"
        "runtime:\n  device: cpu\n  seed: 7\n"
        "logging:\n  level: DEBUG\n  structured: true\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.dataset_root == Path("fixtures/woodscape")
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 7
    assert config.logging.level == "DEBUG"
    assert config.logging.structured is True


def test_environment_overrides_nested_file_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("runtime:\n  device: cpu\n  seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_RUNTIME__DEVICE", "cuda")

    config = load_config(config_path)

    assert config.runtime.device == "cuda"
    assert config.runtime.seed == 7


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("- not\n- a\n- mapping\n", "root must be a mapping"),
        ("1: value\n", "keys must be strings"),
    ],
)
def test_load_config_rejects_invalid_document_roots(
    tmp_path: Path, contents: str, message: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text("surprise: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="surprise"):
        load_config(config_path)


def test_load_config_wraps_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("paths: [unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unable to read configuration file"):
        load_config(config_path)


def test_load_config_validates_seed_range(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-seed.yaml"
    config_path.write_text("runtime:\n  seed: -1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_config(config_path)


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_bounds_file_size(tmp_path: Path) -> None:
    config_path = tmp_path / "oversized.yaml"
    config_path.write_bytes(b"x" * (MAX_CONFIG_BYTES + 1))

    with pytest.raises(ConfigurationError, match="safety limit"):
        load_config(config_path)


def test_legacy_config_without_geometry_sections_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("runtime:\n  seed: 7\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.runtime.seed == 7
    assert config.geometry.theta_max == 2.2
    assert config.bev.resolution == 0.05


def test_geometry_and_bev_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("geometry:\n  theta_max: 2.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_GEOMETRY__THETA_MAX", "2.5")
    monkeypatch.setenv("NEARFIELD360_BEV__RESOLUTION", "0.1")

    config = load_config(config_path)

    assert config.geometry.theta_max == 2.5
    assert config.bev.resolution == 0.1


def test_geometry_config_rejects_nonpositive_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-bev.yaml"
    config_path.write_text("bev:\n  resolution: 0.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="resolution"):
        load_config(config_path)


def test_risk_config_validates_bounds(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-risk.yaml"
    config_path.write_text("risk:\n  rear_length: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="rear_length"):
        load_config(config_path)


def test_robustness_config_defaults() -> None:
    config = load_config()

    assert config.robustness.severities == (1, 2, 3, 4, 5)
    assert config.robustness.rotation_perturbations_deg == (0.5, 1.0, 2.0, 3.0, 5.0)
    assert config.robustness.translation_perturbations_m == (0.02, 0.05, 0.10)
    assert config.robustness.seed == 42


def test_robustness_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("robustness:\n  seed: 99\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_ROBUSTNESS__SEED", "123")

    config = load_config(config_path)

    assert config.robustness.seed == 123


@pytest.mark.parametrize(
    ("field", "bad_value", "pattern"),
    [
        ("severities", "[0, 1]", "severity 0"),
        ("severities", "[1, 6]", "severity 6"),
        ("severities", "[]", "severities must not be empty"),
        ("rotation_perturbations_deg", "[-0.5]", "must be non-negative"),
        ("translation_perturbations_m", "[-0.01]", "must be non-negative"),
    ],
)
def test_robustness_config_validates_fields(
    tmp_path: Path, field: str, bad_value: str, pattern: str
) -> None:
    config_path = tmp_path / "invalid-robustness.yaml"
    config_path.write_text(f"robustness:\n  {field}: {bad_value}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=pattern):
        load_config(config_path)


def test_health_config_validates_brightness_range(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-health.yaml"
    config_path.write_text(
        "health:\n  min_brightness: 200.0\n  max_brightness: 50.0\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match=r"min_brightness.*must be less than.*max_brightness"):
        load_config(config_path)


def test_tracking_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("tracking:\n  max_age: 10\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_TRACKING__MAX_AGE", "15")

    config = load_config(config_path)

    assert config.tracking.max_age == 15


def test_inference_config_defaults() -> None:
    config = load_config()
    assert config.inference.backend == "opencv"
    assert config.inference.device == "cpu"
    assert config.inference.precision == "fp32"
    assert config.inference.input_height == 480
    assert config.inference.input_width == 640
    assert config.inference.mean == (0.485, 0.456, 0.406)
    assert config.inference.std == (0.229, 0.224, 0.225)
    assert config.inference.confidence_threshold == 0.5
    assert config.inference.nms_threshold == 0.4
    assert config.inference.batch_size == 1


@pytest.mark.parametrize(
    ("field", "bad_value", "pattern"),
    [
        ("input_height", "0", "greater than 0"),
        ("input_width", "-10", "greater than 0"),
        ("confidence_threshold", "1.5", "less than or equal to 1"),
        ("nms_threshold", "-0.1", "greater than or equal to 0"),
        ("backend", "tensorflow", "Input should be 'opencv', 'onnxruntime' or 'tensorrt'"),
        ("precision", "int4", "Input should be 'fp32', 'fp16' or 'int8'"),
        ("tensorrt_workspace_mb", "0", "greater than 0"),
        ("device", "tpu", "Input should be 'cpu', 'cuda' or 'directml'"),
    ],
)
def test_inference_config_validates_fields(
    tmp_path: Path, field: str, bad_value: str, pattern: str
) -> None:
    config_path = tmp_path / "invalid-inference.yaml"
    config_path.write_text(f"inference:\n  {field}: {bad_value}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=pattern):
        load_config(config_path)


def test_inference_config_validates_positive_std(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-std.yaml"
    config_path.write_text("inference:\n  std: [0.2, -0.1, 0.2]\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must be strictly positive"):
        load_config(config_path)


def test_inference_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("inference:\n  input_width: 800\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_INFERENCE__INPUT_WIDTH", "1280")
    monkeypatch.setenv("NEARFIELD360_INFERENCE__DEVICE", "cuda")

    config = load_config(config_path)

    assert config.inference.input_width == 1280
    assert config.inference.device == "cuda"


def test_repository_default_config_deserializes_completely() -> None:
    """Verify configs/default.yaml loads into ProjectConfig without validation errors."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    if not config_path.exists():
        pytest.skip("configs/default.yaml not found")

    config = load_config(config_path)

    assert config.paths.dataset_root is None
    assert config.runtime.device == "auto"
    assert config.logging.level == "INFO"
    assert config.geometry.theta_max > 0
    assert config.bev.resolution > 0
    assert config.occupancy.confidence_slope > 0
    assert config.risk.danger_occupancy > 0
    assert config.health.blur_threshold > 0
    assert config.tracking.dt > 0
    assert config.inference.backend in ("opencv", "onnxruntime", "tensorrt")
    assert config.inference.device in ("cpu", "cuda", "directml")
    assert config.inference.precision in ("fp32", "fp16", "int8")
    assert config.inference.tensorrt_workspace_mb == 1024
    assert config.forecast.horizon_seconds == 3.0
    assert config.forecast.step_seconds == 0.5
    assert config.forecast.memory_decay == 0.9
    assert config.forecast.flow_decay == 0.85
    assert config.forecast.min_velocity_threshold == 0.1
    assert config.forecast.hidden_channels == 16
    assert config.slots.min_slot_width == 2.0
    assert config.slots.max_slot_width == 3.8
    assert config.slots.min_slot_length == 4.0
    assert config.slots.max_slot_length == 8.0
    assert config.slots.occupied_ratio_threshold == 0.20
    assert config.slots.uncertain_ratio_threshold == 0.35
    assert config.slots.min_confidence == 0.40
    assert config.slots.approach_lead_distance == 1.5
    assert config.slots.vehicle_width == 1.8
    assert config.slots.vehicle_length == 4.5
    assert config.slots.safety_margin == 0.25


def test_forecast_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("forecast:\n  horizon_seconds: 4.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_FORECAST__HORIZON_SECONDS", "5.0")
    monkeypatch.setenv("NEARFIELD360_FORECAST__STEP_SECONDS", "1.0")

    config = load_config(config_path)

    assert config.forecast.horizon_seconds == 5.0
    assert config.forecast.step_seconds == 1.0


def test_forecast_config_rejects_step_exceeding_horizon(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "forecast:\n  horizon_seconds: 1.0\n  step_seconds: 2.0\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="cannot exceed horizon_seconds"):
        load_config(config_path)


def test_forecast_config_rejects_even_kernel_size(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("forecast:\n  spatial_kernel_size: 4\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must be odd"):
        load_config(config_path)


def test_slots_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("slots:\n  min_slot_width: 2.2\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_SLOTS__MIN_SLOT_WIDTH", "2.5")
    monkeypatch.setenv("NEARFIELD360_SLOTS__MAX_SLOT_WIDTH", "4.0")

    config = load_config(config_path)

    assert config.slots.min_slot_width == 2.5
    assert config.slots.max_slot_width == 4.0


def test_slots_config_rejects_invalid_width(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "slots:\n  min_slot_width: 4.0\n  max_slot_width: 3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match=r"min_slot_width .* must be strictly less than max_slot_width",
    ):
        load_config(config_path)


def test_slots_config_rejects_invalid_length(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "slots:\n  min_slot_length: 9.0\n  max_slot_length: 5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match=r"min_slot_length .* must be strictly less than max_slot_length",
    ):
        load_config(config_path)


def test_planner_config_defaults() -> None:
    config = load_config()

    assert config.planner.wheelbase == 2.7
    assert config.planner.front_overhang == 0.9
    assert config.planner.rear_overhang == 0.9
    assert config.planner.vehicle_width == 1.8
    assert config.planner.max_steer_angle_rad == 0.65
    assert config.planner.max_speed == 1.5
    assert config.planner.max_acceleration == 0.8
    assert config.planner.step_size == 0.1
    assert config.planner.dt == 0.1
    assert config.planner.collision_margin == 0.15
    assert config.planner.danger_threshold == 0.50
    assert config.planner.uncertainty_threshold == 0.20
    assert abs(config.planner.vehicle_length - 4.5) < 1e-6
    assert config.planner.min_turn_radius == pytest.approx(3.55, rel=0.05)


def test_planner_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("planner:\n  wheelbase: 2.5\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_PLANNER__WHEELBASE", "2.8")
    monkeypatch.setenv("NEARFIELD360_PLANNER__MAX_SPEED", "2.0")

    config = load_config(config_path)

    assert config.planner.wheelbase == 2.8
    assert config.planner.max_speed == 2.0


def test_planner_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("planner:\n  max_speed: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_control_config_defaults() -> None:
    config = load_config()

    assert config.control.dt == 0.05
    assert config.control.lookahead_distance == 1.0
    assert config.control.stanley_k == 1.5
    assert config.control.stanley_soften_speed == 0.2
    assert config.control.kp_speed == 1.2
    assert config.control.ki_speed == 0.1
    assert config.control.max_steer_rate_rad_s == 0.8
    assert config.control.steer_time_constant_s == 0.15
    assert config.control.max_cross_track_error_m == 0.50
    assert config.control.max_heading_error_rad == 0.60
    assert config.control.emergency_brake_decel == 2.5
    assert config.control.collision_check_interval == 2
    assert config.control.terminal_dock_tol_xy == 0.15
    assert config.control.terminal_dock_tol_heading == 0.10


def test_control_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("control:\n  dt: 0.02\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_CONTROL__STANLEY_K", "2.0")
    monkeypatch.setenv("NEARFIELD360_CONTROL__LOOKAHEAD_DISTANCE", "1.5")

    config = load_config(config_path)

    assert config.control.dt == 0.02
    assert config.control.stanley_k == 2.0
    assert config.control.lookahead_distance == 1.5


def test_control_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("control:\n  dt: -0.05\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_mission_config_defaults() -> None:
    config = load_config()

    assert config.mission.hold_timeout_s == 5.0
    assert config.mission.max_replans == 3
    assert config.mission.replan_pull_out_dist_m == 1.5
    assert config.mission.approach_speed_m_s == 1.0
    assert config.mission.slot_tracking_distance_gate_m == 2.0
    assert config.mission.slot_confirm_frames == 2
    assert config.mission.slot_max_miss_frames == 5
    assert config.mission.docking_tolerance_x_m == 0.20
    assert config.mission.docking_tolerance_y_m == 0.15
    assert config.mission.docking_tolerance_heading_rad == 0.08
    assert config.mission.safety_dwell_steps == 10


def test_mission_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("mission:\n  hold_timeout_s: 3.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_MISSION__MAX_REPLANS", "5")
    monkeypatch.setenv("NEARFIELD360_MISSION__APPROACH_SPEED_M_S", "0.8")

    config = load_config(config_path)

    assert config.mission.hold_timeout_s == 3.0
    assert config.mission.max_replans == 5
    assert config.mission.approach_speed_m_s == 0.8


def test_mission_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("mission:\n  hold_timeout_s: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_mapping_config_defaults() -> None:
    config = load_config()

    assert config.mapping.default_lane_width_m == 3.5
    assert config.mapping.default_speed_limit_mps == 2.5
    assert config.mapping.turn_penalty_weight == 1.5
    assert config.mapping.odometry_noise_dist == 0.05
    assert config.mapping.odometry_noise_yaw == 0.02
    assert config.mapping.slot_association_gate_m == 2.5
    assert config.mapping.slot_observation_noise_pos == 0.20
    assert config.mapping.slot_observation_noise_yaw == 0.05
    assert config.mapping.max_localization_uncertainty_m == 1.5
    assert config.mapping.replan_reroute_on_blocked_lane is True


def test_mapping_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("mapping:\n  default_lane_width_m: 4.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_MAPPING__DEFAULT_SPEED_LIMIT_MPS", "3.0")
    monkeypatch.setenv("NEARFIELD360_MAPPING__SLOT_ASSOCIATION_GATE_M", "3.0")

    config = load_config(config_path)

    assert config.mapping.default_lane_width_m == 4.0
    assert config.mapping.default_speed_limit_mps == 3.0
    assert config.mapping.slot_association_gate_m == 3.0


def test_mapping_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("mapping:\n  default_lane_width_m: -2.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_fleet_config_defaults() -> None:
    config = load_config()

    assert config.fleet.heartbeat_interval_s == 0.5
    assert config.fleet.telematics_timeout_s == 3.0
    assert config.fleet.conflict_radius_m == 2.5
    assert config.fleet.temporal_headway_s == 2.0
    assert config.fleet.max_conflict_speed_mps == 1.5
    assert config.fleet.priority_policy == "first_come_first_served"
    assert config.fleet.reservation_lease_duration_s == 60.0
    assert config.fleet.communication_latency_s == 0.05
    assert config.fleet.packet_loss_rate == 0.02


def test_fleet_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("fleet:\n  conflict_radius_m: 3.5\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_FLEET__PRIORITY_POLICY", "distance_to_conflict")
    monkeypatch.setenv("NEARFIELD360_FLEET__TEMPORAL_HEADWAY_S", "4.0")

    config = load_config(config_path)

    assert config.fleet.conflict_radius_m == 3.5
    assert config.fleet.priority_policy == "distance_to_conflict"
    assert config.fleet.temporal_headway_s == 4.0


def test_fleet_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("fleet:\n  conflict_radius_m: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_fleet_config_rejects_invalid_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("fleet:\n  priority_policy: invalid_mode\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="Input should be"):
        load_config(config_path)


def test_charging_config_defaults() -> None:
    config = load_config()

    assert config.charging.facility_grid_power_kw == 250.0
    assert config.charging.peak_shaving_threshold_ratio == 0.90
    assert config.charging.default_target_soc == 0.80
    assert config.charging.critical_soc_threshold == 0.20
    assert config.charging.idle_penalty_per_min == 0.50
    assert config.charging.plug_dock_dwell_s == 15.0
    assert config.charging.default_charger_power_kw == 50.0
    assert config.charging.min_power_allocation_kw == 7.0


def test_charging_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("charging:\n  facility_grid_power_kw: 350.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_CHARGING__DEFAULT_TARGET_SOC", "0.95")
    monkeypatch.setenv("NEARFIELD360_CHARGING__PEAK_SHAVING_THRESHOLD_RATIO", "0.85")

    config = load_config(config_path)

    assert config.charging.facility_grid_power_kw == 350.0
    assert config.charging.default_target_soc == 0.95
    assert config.charging.peak_shaving_threshold_ratio == 0.85


def test_charging_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("charging:\n  facility_grid_power_kw: -10.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_charging_config_rejects_invalid_soc(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("charging:\n  default_target_soc: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_config(config_path)


def test_billing_config_defaults() -> None:
    config = load_config()

    assert config.billing.base_hourly_rate == 5.0
    assert config.billing.peak_multiplier == 1.5
    assert config.billing.kwh_tariff == 0.35
    assert config.billing.idle_overstay_per_min == 0.50
    assert config.billing.valet_service_fee == 10.0
    assert config.billing.grace_period_minutes == 15.0
    assert config.billing.tax_rate == 0.08
    assert config.billing.receipt_hmac_secret == "nearfield360-facility-hmac-secret-key-2026"  # noqa: S105


def test_billing_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("billing:\n  base_hourly_rate: 8.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_BILLING__PEAK_MULTIPLIER", "2.0")
    monkeypatch.setenv("NEARFIELD360_BILLING__KWH_TARIFF", "0.45")

    config = load_config(config_path)

    assert config.billing.base_hourly_rate == 8.0
    assert config.billing.peak_multiplier == 2.0
    assert config.billing.kwh_tariff == 0.45


def test_billing_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("billing:\n  base_hourly_rate: -2.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_config(config_path)


def test_billing_config_rejects_short_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("billing:\n  receipt_hmac_secret: abc\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="at least 8 characters"):
        load_config(config_path)


def test_tenant_config_defaults() -> None:
    config = load_config()

    assert config.tenant.default_quota_ratio == 0.25
    assert config.tenant.sla_max_dispatch_wait_s == 180.0
    assert config.tenant.sla_breach_penalty_per_min == 2.50
    assert config.tenant.vip_priority_multiplier == 2.0
    assert config.tenant.express_preemption_buffer_s == 15.0
    assert config.tenant.max_tenant_partitions == 8


def test_tenant_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("tenant:\n  default_quota_ratio: 0.50\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_TENANT__SLA_MAX_DISPATCH_WAIT_S", "120.0")
    monkeypatch.setenv("NEARFIELD360_TENANT__VIP_PRIORITY_MULTIPLIER", "3.0")

    config = load_config(config_path)

    assert config.tenant.default_quota_ratio == 0.50
    assert config.tenant.sla_max_dispatch_wait_s == 120.0
    assert config.tenant.vip_priority_multiplier == 3.0


def test_tenant_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("tenant:\n  default_quota_ratio: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_config(config_path)


def test_curbside_config_defaults() -> None:
    config = load_config()

    assert config.curbside.max_dropoff_dwell_s == 120.0
    assert config.curbside.max_pickup_dwell_s == 180.0
    assert config.curbside.pedestrian_safety_radius_m == 2.0
    assert config.curbside.interlock_clearance_hold_s == 3.0
    assert config.curbside.staging_capacity == 4
    assert config.curbside.overstay_penalty_per_min == 1.50


def test_curbside_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("curbside:\n  max_dropoff_dwell_s: 90.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_CURBSIDE__PEDESTRIAN_SAFETY_RADIUS_M", "2.5")
    monkeypatch.setenv("NEARFIELD360_CURBSIDE__STAGING_CAPACITY", "6")

    config = load_config(config_path)

    assert config.curbside.max_dropoff_dwell_s == 90.0
    assert config.curbside.pedestrian_safety_radius_m == 2.5
    assert config.curbside.staging_capacity == 6


def test_curbside_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("curbside:\n  staging_capacity: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        load_config(config_path)


def test_teleop_config_defaults_and_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("teleop:\n  heartbeat_timeout_s: 2.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_TELEOP__MAX_RTT_MS", "120.0")
    monkeypatch.setenv("NEARFIELD360_TELEOP__AUTO_MRM_ON_HEARTBEAT_LOSS", "false")

    config = load_config(config_path)

    assert config.teleop.heartbeat_timeout_s == 2.0
    assert config.teleop.max_rtt_ms == 120.0
    assert config.teleop.auto_mrm_on_heartbeat_loss is False
    assert config.teleop.critical_rtt_ms == 350.0
    assert config.teleop.video_target_fps == 20.0
    assert config.teleop.video_fallback_fps == 5.0


def test_teleop_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("teleop:\n  heartbeat_timeout_s: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_teleop_config_rejects_fallback_exceeding_target(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "teleop:\n  video_target_fps: 10.0\n  video_fallback_fps: 15.0\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="cannot exceed"):
        load_config(config_path)


def test_teleop_config_rejects_max_rtt_exceeding_critical(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "teleop:\n  max_rtt_ms: 400.0\n  critical_rtt_ms: 300.0\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="strictly less than"):
        load_config(config_path)


def test_floor_config_defaults() -> None:
    config = load_config()

    assert config.floor.default_floor_height_m == 3.5
    assert config.floor.max_ramp_grade_pct == 15.0
    assert config.floor.min_breakover_clearance_m == 0.05
    assert config.floor.ramp_speed_limit_mps == 2.5
    assert config.floor.elevator_hoist_speed_mps == 1.2
    assert config.floor.elevator_door_dwell_s == 5.0
    assert config.floor.elevator_leveling_tolerance_m == 0.015
    assert config.floor.ramp_token_timeout_s == 30.0
    assert config.floor.max_floors_supported == 16


def test_floor_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("floor:\n  default_floor_height_m: 4.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_FLOOR__MAX_RAMP_GRADE_PCT", "18.0")
    monkeypatch.setenv("NEARFIELD360_FLOOR__ELEVATOR_HOIST_SPEED_MPS", "1.5")
    monkeypatch.setenv("NEARFIELD360_FLOOR__MAX_FLOORS_SUPPORTED", "24")

    config = load_config(config_path)

    assert config.floor.default_floor_height_m == 4.0
    assert config.floor.max_ramp_grade_pct == 18.0
    assert config.floor.elevator_hoist_speed_mps == 1.5
    assert config.floor.max_floors_supported == 24


def test_floor_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("floor:\n  default_floor_height_m: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_floor_config_rejects_excessive_floors(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("floor:\n  max_floors_supported: 100\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 64"):
        load_config(config_path)


def test_emergency_config_defaults() -> None:
    config = load_config()

    assert config.emergency.preemption_radius_m == 35.0
    assert config.emergency.hazard_buffer_radius_m == 12.0
    assert config.emergency.evacuation_speed_limit_mps == 4.0
    assert config.emergency.fire_door_clearance_timeout_s == 15.0
    assert config.emergency.stagger_interval_s == 3.0
    assert config.emergency.emergency_heartbeat_timeout_s == 1.0
    assert config.emergency.thermal_runaway_temp_c == 70.0


def test_emergency_config_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("emergency:\n  preemption_radius_m: 45.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_EMERGENCY__HAZARD_BUFFER_RADIUS_M", "16.0")
    monkeypatch.setenv("NEARFIELD360_EMERGENCY__EVACUATION_SPEED_LIMIT_MPS", "5.0")
    monkeypatch.setenv("NEARFIELD360_EMERGENCY__THERMAL_RUNAWAY_TEMP_C", "85.0")

    config = load_config(config_path)

    assert config.emergency.preemption_radius_m == 45.0
    assert config.emergency.hazard_buffer_radius_m == 16.0
    assert config.emergency.evacuation_speed_limit_mps == 5.0
    assert config.emergency.thermal_runaway_temp_c == 85.0


def test_emergency_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("emergency:\n  preemption_radius_m: -10.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_emergency_config_rejects_excessive_temperature(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("emergency:\n  thermal_runaway_temp_c: 350.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 250"):
        load_config(config_path)


def test_gate_config_defaults() -> None:
    config = load_config()

    assert config.gate.portal_clearance_height_m == 2.10
    assert config.gate.portal_clearance_width_m == 2.60
    assert config.gate.ground_clearance_min_m == 0.12
    assert config.gate.barrier_arm_length_m == 3.0
    assert config.gate.barrier_opening_time_s == 1.8
    assert config.gate.barrier_holding_time_s == 5.0
    assert config.gate.barrier_closing_time_s == 2.2
    assert config.gate.alpr_confidence_threshold == 0.85
    assert config.gate.anti_tailgating_timeout_s == 2.0
    assert config.gate.safety_loop_debounce_ms == 50.0
    assert config.gate.uwb_rssi_threshold_dbm == -75.0


def test_gate_config_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("gate:\n  portal_clearance_height_m: 2.30\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_GATE__PORTAL_CLEARANCE_WIDTH_M", "3.0")
    monkeypatch.setenv("NEARFIELD360_GATE__BARRIER_ARM_LENGTH_M", "3.5")
    monkeypatch.setenv("NEARFIELD360_GATE__ALPR_CONFIDENCE_THRESHOLD", "0.90")

    config = load_config(config_path)

    assert config.gate.portal_clearance_height_m == 2.30
    assert config.gate.portal_clearance_width_m == 3.0
    assert config.gate.barrier_arm_length_m == 3.5
    assert config.gate.alpr_confidence_threshold == 0.90


def test_gate_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("gate:\n  portal_clearance_height_m: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_gate_config_rejects_excessive_clearance(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("gate:\n  portal_clearance_height_m: 15.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 10"):
        load_config(config_path)


def test_coop_config_defaults() -> None:
    config = load_config()

    assert config.coop.max_latency_compensation_s == 0.50
    assert config.coop.association_gate_mahalanobis == 3.0
    assert config.coop.min_cpm_confidence == 0.60
    assert config.coop.blind_spot_warning_ttc_s == 2.5
    assert config.coop.occlusion_grid_resolution_m == 0.25
    assert config.coop.rpu_broadcast_frequency_hz == 10.0
    assert config.coop.track_covariance_inflation_factor == 1.25


def test_coop_config_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("coop:\n  max_latency_compensation_s: 0.80\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_COOP__ASSOCIATION_GATE_MAHALANOBIS", "4.5")
    monkeypatch.setenv("NEARFIELD360_COOP__MIN_CPM_CONFIDENCE", "0.75")
    monkeypatch.setenv("NEARFIELD360_COOP__BLIND_SPOT_WARNING_TTC_S", "3.0")

    config = load_config(config_path)

    assert config.coop.max_latency_compensation_s == 0.80
    assert config.coop.association_gate_mahalanobis == 4.5
    assert config.coop.min_cpm_confidence == 0.75
    assert config.coop.blind_spot_warning_ttc_s == 3.0


def test_coop_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("coop:\n  max_latency_compensation_s: -1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_coop_config_rejects_excessive_latency(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("coop:\n  max_latency_compensation_s: 10.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 5"):
        load_config(config_path)


def test_v2g_config_defaults() -> None:
    config = load_config()

    assert config.v2g.facility_transformer_limit_kw == 500.0
    assert config.v2g.peak_shaving_target_kw == 350.0
    assert config.v2g.min_departure_reserve_soc == 0.60
    assert config.v2g.max_discharge_power_kw == 22.0
    assert config.v2g.max_charge_power_kw == 50.0
    assert config.v2g.battery_degradation_penalty_per_kwh == 0.04
    assert config.v2g.grid_frequency_nominal_hz == 50.0
    assert config.v2g.frequency_droop_gain_kw_per_hz == 25.0
    assert config.v2g.arbitrage_interval_minutes == 15


def test_v2g_config_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("v2g:\n  facility_transformer_limit_kw: 750.0\n", encoding="utf-8")
    monkeypatch.setenv("NEARFIELD360_V2G__PEAK_SHAVING_TARGET_KW", "420.0")
    monkeypatch.setenv("NEARFIELD360_V2G__MIN_DEPARTURE_RESERVE_SOC", "0.75")
    monkeypatch.setenv("NEARFIELD360_V2G__MAX_DISCHARGE_POWER_KW", "30.0")

    config = load_config(config_path)

    assert config.v2g.facility_transformer_limit_kw == 750.0
    assert config.v2g.peak_shaving_target_kw == 420.0
    assert config.v2g.min_departure_reserve_soc == 0.75
    assert config.v2g.max_discharge_power_kw == 30.0


def test_v2g_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("v2g:\n  facility_transformer_limit_kw: -50.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_v2g_config_rejects_invalid_soc(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("v2g:\n  min_departure_reserve_soc: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_config(config_path)


def test_operations_config_defaults() -> None:
    config = load_config()

    assert config.operations.day_duration_hours == 24.0
    assert config.operations.simulation_time_step_s == 300.0
    assert config.operations.peak_morning_arrival_hour == 8.5
    assert config.operations.peak_evening_arrival_hour == 17.5
    assert config.operations.peak_arrivals_per_hour == 45.0
    assert config.operations.base_arrivals_per_hour == 6.0
    assert config.operations.parking_duration_mean_hours == 3.0
    assert config.operations.charging_activation_share == 0.30
    assert config.operations.billing_grace_minutes == 15


def test_operations_config_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "operations:\n  peak_arrivals_per_hour: 60.0\n  billing_grace_minutes: 30\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.operations.peak_arrivals_per_hour == 60.0
    assert config.operations.billing_grace_minutes == 30
    assert config.operations.day_duration_hours == 24.0


def test_operations_config_rejects_invalid_arrival_rate(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("operations:\n  peak_arrivals_per_hour: -5.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_path)


def test_can_config_defaults() -> None:
    config = load_config()

    assert config.can.bus_bitrate_kbps == 500
    assert config.can.data_bitrate_kbps == 2000
    assert config.can.sample_point_percent == 80.0
    assert config.can.cycle_time_ms == 20.0
    assert config.can.fault_injection_rate == 0.0
    assert config.can.max_bus_load_percent == 85.0
    assert config.can.clock_jitter_max_us == 500.0


def test_can_config_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "can:\n  bus_bitrate_kbps: 250\n  fault_injection_rate: 0.05\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEARFIELD360_CAN__MAX_BUS_LOAD_PERCENT", "90.0")

    config = load_config(config_path)

    assert config.can.bus_bitrate_kbps == 250
    assert config.can.fault_injection_rate == 0.05
    assert config.can.max_bus_load_percent == 90.0
    assert config.can.cycle_time_ms == 20.0


def test_can_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("can:\n  bus_bitrate_kbps: 50\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 125"):
        load_config(config_path)


def test_can_config_rejects_invalid_fault_rate(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("can:\n  fault_injection_rate: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_config(config_path)


def test_sensor_config_defaults() -> None:
    config = load_config()

    assert config.sensor.carrier_frequency_ghz == 77.0
    assert config.sensor.sweep_bandwidth_mhz == 1500.0
    assert config.sensor.chirp_duration_us == 50.0
    assert config.sensor.num_chirps == 128
    assert config.sensor.samples_per_chirp == 256
    assert config.sensor.cfar_guard_cells == 2
    assert config.sensor.cfar_train_cells == 4
    assert config.sensor.cfar_threshold_factor == 1.25
    assert config.sensor.radar_max_range_m == 50.0
    assert config.sensor.ultrasonic_frequency_khz == 48.0
    assert config.sensor.ultrasonic_beam_angle_deg == 60.0
    assert config.sensor.ultrasonic_min_range_m == 0.15
    assert config.sensor.ultrasonic_max_range_m == 4.5
    assert config.sensor.speed_of_sound_mps == 343.0
    assert config.sensor.num_ultrasonic_sensors == 12


def test_sensor_config_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "sensor:\n  carrier_frequency_ghz: 79.0\n  radar_max_range_m: 80.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEARFIELD360_SENSOR__NUM_ULTRASONIC_SENSORS", "8")

    config = load_config(config_path)

    assert config.sensor.carrier_frequency_ghz == 79.0
    assert config.sensor.radar_max_range_m == 80.0
    assert config.sensor.num_ultrasonic_sensors == 8
    assert config.sensor.sweep_bandwidth_mhz == 1500.0


def test_sensor_config_rejects_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("sensor:\n  carrier_frequency_ghz: -77.0\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than 1"):
        load_config(config_path)


def test_sensor_config_rejects_invalid_num_sensors(tmp_path: Path) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("sensor:\n  num_ultrasonic_sensors: 2\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="greater than or equal to 4"):
        load_config(config_path)
