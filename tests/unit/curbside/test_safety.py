"""Unit tests for pedestrian safety interlock engine and clearance debounce."""

from __future__ import annotations

import pytest

from nearfield360.config import CurbsideConfig
from nearfield360.curbside.models import PedestrianDetection, SafetyInterlockStatus
from nearfield360.curbside.safety import InterlockViolationError, SafetyInterlockEngine


def test_safety_interlock_initial_state() -> None:
    engine = SafetyInterlockEngine()
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=10.0)
    assert permitted is True
    assert status == SafetyInterlockStatus.DISENGAGED_CLEAR
    engine.assert_motion_permitted("veh-01", current_time_s=10.0)


def test_safety_interlock_door_and_trunk() -> None:
    engine = SafetyInterlockEngine()

    # Door open
    engine.set_door_state("veh-01", is_open=True)
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=10.0)
    assert permitted is False
    assert status == SafetyInterlockStatus.ENGAGED_DOOR_OPEN

    with pytest.raises(InterlockViolationError, match="engaged_door_open"):
        engine.assert_motion_permitted("veh-01", current_time_s=10.0)

    # Door closed
    engine.set_door_state("veh-01", is_open=False)
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=10.0)
    assert permitted is True
    assert status == SafetyInterlockStatus.DISENGAGED_CLEAR

    # Trunk open
    engine.set_trunk_state("veh-01", is_open=True)
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=10.0)
    assert permitted is False
    assert status == SafetyInterlockStatus.ENGAGED_TRUNK_OPEN

    with pytest.raises(InterlockViolationError, match="engaged_trunk_open"):
        engine.assert_motion_permitted("veh-01", current_time_s=10.0)


def test_safety_interlock_pedestrian_proximity_and_clearance_hold() -> None:
    cfg = CurbsideConfig(
        pedestrian_safety_radius_m=2.0,
        interlock_clearance_hold_s=3.0,
    )
    engine = SafetyInterlockEngine(config=cfg)

    # Pedestrian at 3.5m (outside critical radius)
    peds_safe = [PedestrianDetection(pedestrian_id="p1", distance_m=3.5)]
    annotated = engine.update_pedestrians("veh-01", peds_safe, current_time_s=5.0)
    assert annotated[0].in_critical_zone is False

    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=5.0)
    assert permitted is True
    assert status == SafetyInterlockStatus.DISENGAGED_CLEAR

    # Pedestrian encroaches to 1.2m at t=10.0s
    peds_close = [PedestrianDetection(pedestrian_id="p1", distance_m=1.2)]
    annotated = engine.update_pedestrians("veh-01", peds_close, current_time_s=10.0)
    assert annotated[0].in_critical_zone is True

    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=10.0)
    assert permitted is False
    assert status == SafetyInterlockStatus.ENGAGED_PEDESTRIAN_PROXIMITY

    with pytest.raises(InterlockViolationError, match="engaged_pedestrian_proximity"):
        engine.assert_motion_permitted("veh-01", current_time_s=10.0)

    # At t=11.5s, pedestrian leaves (no pedestrians detected), but 1.5s < 3.0s hold
    engine.update_pedestrians("veh-01", [], current_time_s=11.5)
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=11.5)
    assert permitted is False
    assert status == SafetyInterlockStatus.ENGAGED_PEDESTRIAN_PROXIMITY

    # At t=13.1s, 3.1s > 3.0s hold has passed -> interlock clears
    permitted, status = engine.evaluate_interlock("veh-01", current_time_s=13.1)
    assert permitted is True
    assert status == SafetyInterlockStatus.DISENGAGED_CLEAR
    engine.assert_motion_permitted("veh-01", current_time_s=13.1)
