"""IEEE 802.15.4z Ultra-Wideband (UWB) STS distance-bounding and anti-relay verification engine."""

import hashlib
import hmac
from typing import ClassVar

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.models import UWBRangeMeasurement


class UWBEngine:
    """UWB physical layer distance bounding and anti-relay attack defense engine.

    Implements IEEE 802.15.4z Scrambled Timestamp Sequence (STS) authentication,
    Time-of-Flight (ToF) physical law validation, and relay attack detection.
    """

    SPEED_OF_LIGHT_M_PER_NS: ClassVar[float] = 0.299792458

    def __init__(self, config: DigitalKeyConfig | None = None) -> None:
        self.config = config or DigitalKeyConfig()
        self._session_salt = self.config.session_salt.encode()

    def compute_tof_from_sds_twr(
        self,
        t_round1_ns: float,
        t_reply1_ns: float,
        t_round2_ns: float,
        t_reply2_ns: float,
    ) -> float:
        """Compute Symmetric Double-Sided Two-Way Ranging (SDS-TWR) Time-of-Flight.

        Formula:
            ToF = ((t_round1 * t_round2) - (t_reply1 * t_reply2)) /
                  (t_round1 + t_round2 + t_reply1 + t_reply2)
        """
        numerator = (t_round1_ns * t_round2_ns) - (t_reply1_ns * t_reply2_ns)
        denominator = t_round1_ns + t_round2_ns + t_reply1_ns + t_reply2_ns
        if denominator <= 0.0:
            return 0.0
        return max(0.0, numerator / denominator)

    def compute_distance_m(self, tof_ns: float) -> float:
        """Convert one-way Time-of-Flight in nanoseconds to distance in meters."""
        return max(0.0, tof_ns * self.SPEED_OF_LIGHT_M_PER_NS)

    def generate_sts_signature(self, anchor_id: str, timestamp_ns: int, nonce: str) -> str:
        """Generate expected cryptographic Scrambled Timestamp Sequence (STS) token."""
        msg = f"{anchor_id}:{timestamp_ns}:{nonce}".encode()
        return hmac.new(self._session_salt, msg, hashlib.sha256).hexdigest()[:16]

    def verify_sts(
        self,
        anchor_id: str,
        timestamp_ns: int,
        nonce: str,
        candidate_sts: str,
    ) -> bool:
        """Cryptographically verify the STS signature against tampering."""
        expected = self.generate_sts_signature(anchor_id, timestamp_ns, nonce)
        return hmac.compare_digest(expected, candidate_sts)

    def detect_relay_attack(
        self,
        measurement: UWBRangeMeasurement,
    ) -> tuple[bool, float, str]:
        """Examine physical layer measurement for signs of relay / distance manipulation attacks.

        Returns:
            (relay_detected, discrepancy_ns, reason)
        """
        # 1. Check STS validity
        if self.config.enable_uwb_sts and not measurement.sts_valid:
            return True, 999.0, "Cryptographic STS verification failed or sequence scrambled"

        # 2. Physics consistency check between distance and ToF
        expected_distance = self.compute_distance_m(measurement.tof_ns)
        discrepancy_m = abs(measurement.distance_m - expected_distance)
        discrepancy_ns = discrepancy_m / self.SPEED_OF_LIGHT_M_PER_NS

        if discrepancy_ns > self.config.max_relay_tof_discrepancy_ns:
            return (
                True,
                discrepancy_ns,
                f"Physical ToF discrepancy ({discrepancy_ns:.2f} ns) exceeds threshold "
                f"({self.config.max_relay_tof_discrepancy_ns:.2f} ns) - possible relay attack",
            )

        # 3. Channel impulse response (CIR) plausibility check
        # First Path Power significantly lower than SNR indicates reflected/amplified relay
        if measurement.snr_db < 6.0 and measurement.distance_m < 5.0:
            return (
                True,
                discrepancy_ns,
                (
                    f"Abnormally low SNR ({measurement.snr_db:.1f} dB) at close range "
                    "- potential relay"
                ),
            )

        return False, discrepancy_ns, "OK - Physical and cryptographic bounds verified"

    def process_measurements(
        self,
        measurements: list[UWBRangeMeasurement],
    ) -> tuple[list[UWBRangeMeasurement], bool, float, list[str]]:
        """Validate and filter a batch of anchor range measurements.

        Returns:
            (valid_measurements, any_relay_detected, max_discrepancy_ns, attack_reasons)
        """
        valid_list: list[UWBRangeMeasurement] = []
        any_attack = False
        max_discrepancy = 0.0
        reasons: list[str] = []

        for m in measurements:
            is_relay, disc_ns, reason = self.detect_relay_attack(m)
            if disc_ns > max_discrepancy:
                max_discrepancy = disc_ns

            if is_relay:
                any_attack = True
                reasons.append(f"[{m.anchor_id}] {reason}")
            else:
                valid_list.append(m)

        return valid_list, any_attack, max_discrepancy, reasons
