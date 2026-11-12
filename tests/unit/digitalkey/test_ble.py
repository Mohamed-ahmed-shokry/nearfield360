"""Unit tests for BLE proximity zone classification and RSSI distance estimation."""

from nearfield360.digitalkey.ble import BLEProximityEstimator
from nearfield360.digitalkey.models import ProximityZone


def test_ble_distance_estimation() -> None:
    estimator = BLEProximityEstimator(reference_rssi_1m=-59.0, path_loss_exponent=2.0)
    # At reference RSSI (-59 dBm), distance is exactly 1.0 m
    d_1m = estimator.estimate_distance_m(-59.0)
    assert abs(d_1m - 1.0) < 0.05

    # At -79 dBm (20 dB drop), with path_loss=2.0, distance is 10.0 m
    d_10m = estimator.estimate_distance_m(-79.0)
    assert abs(d_10m - 10.0) < 0.1


def test_classify_raw_zone() -> None:
    estimator = BLEProximityEstimator()
    assert estimator.classify_raw_zone(1.0) == ProximityZone.TOUCH
    assert estimator.classify_raw_zone(2.5) == ProximityZone.IMMEDIATE
    assert estimator.classify_raw_zone(5.0) == ProximityZone.NEAR
    assert estimator.classify_raw_zone(15.0) == ProximityZone.FAR
    assert estimator.classify_raw_zone(25.0) == ProximityZone.OUT_OF_RANGE


def test_ble_smoothing_filter() -> None:
    estimator = BLEProximityEstimator(alpha_filter=0.5)
    # First reading initializes filter
    dist1, _ = estimator.update(-60.0)
    assert estimator.filtered_rssi == -60.0

    # Second reading blends 50%
    dist2, _ = estimator.update(-80.0)
    assert estimator.filtered_rssi == -70.0
    assert dist2 > dist1


def test_ble_debounced_zone_transition() -> None:
    estimator = BLEProximityEstimator(debounce_count=3, alpha_filter=1.0)
    assert estimator.current_zone == ProximityZone.OUT_OF_RANGE

    # Send close reading (-50 dBm -> ~0.4m -> TOUCH)
    # Step 1: candidate count = 1, current remains OUT_OF_RANGE
    _, z1 = estimator.update(-50.0)
    assert z1 == ProximityZone.OUT_OF_RANGE

    # Step 2: candidate count = 2, current remains OUT_OF_RANGE
    _, z2 = estimator.update(-50.0)
    assert z2 == ProximityZone.OUT_OF_RANGE

    # Step 3: candidate count = 3 >= debounce threshold -> updates to TOUCH
    _, z3 = estimator.update(-50.0)
    assert z3 == ProximityZone.TOUCH
    assert estimator.current_zone == ProximityZone.TOUCH


def test_ble_reset() -> None:
    estimator = BLEProximityEstimator()
    estimator.update(-55.0)
    assert estimator.filtered_rssi is not None

    estimator.reset()
    assert estimator.filtered_rssi is None
    assert estimator.current_zone == ProximityZone.OUT_OF_RANGE
