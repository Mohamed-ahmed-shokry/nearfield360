"""Unit tests for UWB IEEE 802.15.4z STS distance bounding and anti-relay engine."""

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.models import UWBRangeMeasurement
from nearfield360.digitalkey.uwb import UWBEngine


def test_sds_twr_tof_calculation() -> None:
    engine = UWBEngine()
    # SDS-TWR symmetric timestamps
    t_round1 = 20.0
    t_reply1 = 10.0
    t_round2 = 20.0
    t_reply2 = 10.0
    tof = engine.compute_tof_from_sds_twr(t_round1, t_reply1, t_round2, t_reply2)
    # ((20*20) - (10*10)) / (20+20+10+10) = (400 - 100) / 60 = 300 / 60 = 5.0 ns
    assert abs(tof - 5.0) < 1e-4

    # Degenerate input handles cleanly
    zero_tof = engine.compute_tof_from_sds_twr(0.0, 0.0, 0.0, 0.0)
    assert zero_tof == 0.0


def test_distance_computation() -> None:
    engine = UWBEngine()
    tof_ns = 10.0
    distance_m = engine.compute_distance_m(tof_ns)
    # c * 10 ns = 0.299792458 * 10 ~ 2.9979 m
    assert abs(distance_m - 2.99792458) < 1e-5


def test_sts_cryptographic_generation_and_verification() -> None:
    engine = UWBEngine()
    nonce = "a1b2c3d4e5f67890"
    ts = 1700000000000000
    sts = engine.generate_sts_signature("FL", ts, nonce)

    assert isinstance(sts, str)
    assert len(sts) == 16
    assert engine.verify_sts("FL", ts, nonce, sts) is True
    # Tampered STS fails
    assert engine.verify_sts("FL", ts, nonce, "0000000000000000") is False
    assert engine.verify_sts("FR", ts, nonce, sts) is False


def test_relay_attack_detection_sts_failure() -> None:
    engine = UWBEngine()
    meas = UWBRangeMeasurement(
        anchor_id="FL",
        distance_m=1.5,
        timestamp_ns=1700000000000000,
        sts_valid=False,  # Attacker failed STS
        tof_ns=5.0,
        snr_db=20.0,
    )
    is_relay, _disc_ns, reason = engine.detect_relay_attack(meas)
    assert is_relay is True
    assert "STS verification failed" in reason


def test_relay_attack_detection_tof_discrepancy() -> None:
    cfg = DigitalKeyConfig(max_relay_tof_discrepancy_ns=2.0)
    engine = UWBEngine(cfg)

    # Injected fake distance that doesn't match physical ToF
    # Claimed distance = 1.0m, but ToF = 20.0 ns (~6.0m)
    meas = UWBRangeMeasurement(
        anchor_id="RR",
        distance_m=1.0,
        timestamp_ns=1700000000000000,
        sts_valid=True,
        tof_ns=20.0,
        snr_db=22.0,
    )
    is_relay, disc_ns, reason = engine.detect_relay_attack(meas)
    assert is_relay is True
    assert "exceeds threshold" in reason
    assert disc_ns > 2.0


def test_relay_attack_detection_low_snr_at_close_range() -> None:
    engine = UWBEngine()
    # At 2.0m, SNR of 3.0 dB indicates anomalous attenuated/reflected relay tunnel
    meas = UWBRangeMeasurement(
        anchor_id="FR",
        distance_m=2.0,
        timestamp_ns=1700000000000000,
        sts_valid=True,
        tof_ns=6.671,
        snr_db=3.0,
    )
    is_relay, _, reason = engine.detect_relay_attack(meas)
    assert is_relay is True
    assert "Abnormally low SNR" in reason


def test_batch_process_measurements() -> None:
    engine = UWBEngine()
    valid_meas = UWBRangeMeasurement(
        anchor_id="FL",
        distance_m=3.0,
        timestamp_ns=1700000000000000,
        sts_valid=True,
        tof_ns=10.007,
        snr_db=25.0,
    )
    attacked_meas = UWBRangeMeasurement(
        anchor_id="RL",
        distance_m=0.5,
        timestamp_ns=1700000000000000,
        sts_valid=True,
        tof_ns=25.0,  # ~7.5m ToF vs 0.5m claimed
        snr_db=20.0,
    )

    valid_list, any_attack, _max_disc, reasons = engine.process_measurements(
        [valid_meas, attacked_meas]
    )
    assert any_attack is True
    assert len(valid_list) == 1
    assert valid_list[0].anchor_id == "FL"
    assert len(reasons) == 1
    assert "[RL]" in reasons[0]
