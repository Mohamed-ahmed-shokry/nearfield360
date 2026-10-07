"""Unit tests for the VehicleSimulator."""

from __future__ import annotations

from nearfield360.config import ParkingControlConfig, ParkingPlannerConfig
from nearfield360.control.models import ControlCommand
from nearfield360.control.simulator import SimulatorNoiseConfig, VehicleSimulator
from nearfield360.planning.models import ManeuverGear


def test_simulator_initial_state() -> None:
    sim = VehicleSimulator(initial_pose=(1.0, 2.0, 0.5))
    state = sim.state
    assert state.x == 1.0
    assert state.y == 2.0
    assert state.heading_rad == 0.5
    assert state.velocity == 0.0
    assert state.steer_angle_rad == 0.0


def test_simulator_straight_forward_motion() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.1)
    sim = VehicleSimulator(initial_pose=(0.0, 0.0, 0.0), control_config=ctl_cfg)

    cmd = ControlCommand(
        steering_angle_rad=0.0,
        acceleration_cmd=1.0,
        gear=ManeuverGear.FORWARD,
    )

    for _ in range(10):  # 1.0 s total
        sim.step(cmd)

    # After 1s with a=1.0, v reaches 1.0 m/s, distance x = 0.5 * a * t^2 ~ 0.55m
    assert sim.state.x > 0.4
    assert abs(sim.state.y) < 1e-3
    assert abs(sim.state.heading_rad) < 1e-3
    assert sim.state.velocity > 0.8


def test_simulator_straight_reverse_motion() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.1)
    sim = VehicleSimulator(initial_pose=(0.0, 0.0, 0.0), control_config=ctl_cfg)

    cmd = ControlCommand(
        steering_angle_rad=0.0,
        acceleration_cmd=1.0,
        gear=ManeuverGear.REVERSE,
    )

    for _ in range(10):
        sim.step(cmd)

    # In reverse gear, x decreases
    assert sim.state.x < -0.4
    assert abs(sim.state.y) < 1e-3
    assert sim.state.gear == ManeuverGear.REVERSE


def test_simulator_turning_motion() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.05, steer_time_constant_s=0.01)
    sim = VehicleSimulator(initial_pose=(0.0, 0.0, 0.0), control_config=ctl_cfg)

    cmd = ControlCommand(
        steering_angle_rad=0.3,
        acceleration_cmd=1.0,
        gear=ManeuverGear.FORWARD,
    )

    for _ in range(20):  # 1.0 s
        sim.step(cmd)

    # Vehicle should curve to the left (y > 0, heading > 0)
    assert sim.state.y > 0.01
    assert sim.state.heading_rad > 0.04


def test_simulator_actuator_lag() -> None:
    ctl_cfg = ParkingControlConfig(dt=0.05, steer_time_constant_s=0.20, max_steer_rate_rad_s=1.0)
    sim = VehicleSimulator(control_config=ctl_cfg)

    cmd = ControlCommand(steering_angle_rad=0.5, acceleration_cmd=0.0)
    sim.step(cmd)

    # Because of lag and rate limiting, steering angle should not jump immediately to 0.5
    assert 0.0 < sim.state.steer_angle_rad < 0.5


def test_simulator_noise_injection() -> None:
    noise_cfg = SimulatorNoiseConfig(pos_std_m=0.05, heading_std_rad=0.02)
    sim = VehicleSimulator(initial_pose=(5.0, 5.0, 1.0), noise_config=noise_cfg, seed=123)

    exact = sim.state
    measured = sim.get_measured_state()

    assert exact.x == 5.0
    assert exact.y == 5.0
    # Measured state deviates from exact state
    assert measured.x != exact.x or measured.y != exact.y


def test_simulator_footprint() -> None:
    plan_cfg = ParkingPlannerConfig(wheelbase=2.7, front_overhang=0.9, rear_overhang=0.9)
    sim = VehicleSimulator(initial_pose=(0.0, 0.0, 0.0), planner_config=plan_cfg)

    poly = sim.get_footprint_polygon()
    assert len(poly) == 4
