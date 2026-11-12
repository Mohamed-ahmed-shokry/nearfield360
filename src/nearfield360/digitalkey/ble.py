"""Bluetooth Low Energy (BLE) RSSI distance estimator and proximity zone classifier."""

from nearfield360.config import DigitalKeyConfig
from nearfield360.digitalkey.models import ProximityZone


class BLEProximityEstimator:
    """Estimates distance and classifies proximity zones using filtered BLE RSSI signals."""

    def __init__(
        self,
        config: DigitalKeyConfig | None = None,
        reference_rssi_1m: float = -59.0,
        path_loss_exponent: float = 2.4,
        alpha_filter: float = 0.35,
        debounce_count: int = 3,
    ) -> None:
        self.config = config or DigitalKeyConfig()
        self.ref_rssi_1m = reference_rssi_1m
        self.path_loss_exp = path_loss_exponent
        self.alpha = alpha_filter
        self.debounce_threshold = debounce_count

        self._filtered_rssi: float | None = None
        self._current_zone = ProximityZone.OUT_OF_RANGE
        self._candidate_zone: ProximityZone | None = None
        self._candidate_count: int = 0

    @property
    def current_zone(self) -> ProximityZone:
        """Currently active debounced proximity zone."""
        return self._current_zone

    @property
    def filtered_rssi(self) -> float | None:
        """Latest exponentially smoothed RSSI in dBm."""
        return self._filtered_rssi

    def estimate_distance_m(self, rssi_dbm: float) -> float:
        """Convert RSSI into an estimated distance in meters using log-distance path loss."""
        # Clamp RSSI to physically reasonable bounds
        clamped_rssi = min(-20.0, max(-105.0, rssi_dbm))
        exponent = (self.ref_rssi_1m - clamped_rssi) / (10.0 * self.path_loss_exp)
        return float(round(10.0**exponent, 2))

    def classify_raw_zone(self, distance_m: float) -> ProximityZone:
        """Map raw estimated distance into proximity zones according to configuration thresholds."""
        if distance_m <= self.config.touch_threshold_m:
            return ProximityZone.TOUCH
        if distance_m <= self.config.immediate_threshold_m:
            return ProximityZone.IMMEDIATE
        if distance_m <= self.config.near_threshold_m:
            return ProximityZone.NEAR
        if distance_m <= self.config.far_threshold_m:
            return ProximityZone.FAR
        return ProximityZone.OUT_OF_RANGE

    def update(self, raw_rssi_dbm: float) -> tuple[float, ProximityZone]:
        """Ingest a new raw RSSI reading, update the filter, and return (distance_m, zone)."""
        if self._filtered_rssi is None:
            self._filtered_rssi = raw_rssi_dbm
        else:
            self._filtered_rssi = (self.alpha * raw_rssi_dbm) + (
                (1.0 - self.alpha) * self._filtered_rssi
            )

        dist_m = self.estimate_distance_m(self._filtered_rssi)
        raw_zone = self.classify_raw_zone(dist_m)

        # Debounce zone transitions to prevent boundary flipping
        if raw_zone == self._current_zone:
            self._candidate_zone = None
            self._candidate_count = 0
        elif raw_zone == self._candidate_zone:
            self._candidate_count += 1
            if self._candidate_count >= self.debounce_threshold:
                self._current_zone = raw_zone
                self._candidate_zone = None
                self._candidate_count = 0
        else:
            self._candidate_zone = raw_zone
            self._candidate_count = 1

        return dist_m, self._current_zone

    def reset(self) -> None:
        """Reset internal filter and debouncer state."""
        self._filtered_rssi = None
        self._current_zone = ProximityZone.OUT_OF_RANGE
        self._candidate_zone = None
        self._candidate_count = 0
