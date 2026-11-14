"""Foreign Object Debris (FOD) and Living Object Detection (LOD) safety interlock engine."""

from datetime import UTC, datetime

from nearfield360.config import WPTConfig
from nearfield360.wpt.models import FODStatus, WPTInterlockState


class FODDetector:
    """Evaluates resonant circuit impedance metrics and underbody sensors for safety interlocks."""

    def __init__(
        self,
        config: WPTConfig | None = None,
        nominal_q_factor: float = 280.0,
    ) -> None:
        self.config = config or WPTConfig()
        self.nominal_frequency_khz = self.config.nominal_frequency_khz
        self.nominal_q_factor = nominal_q_factor

    def evaluate_resonance(
        self,
        measured_frequency_khz: float,
        measured_q_factor: float,
        living_object_present: bool = False,
        manual_emergency_trip: bool = False,
    ) -> FODStatus:
        """Evaluate circuit resonance shifts and perimeter sensors against safety thresholds."""
        freq_shift = round(abs(measured_frequency_khz - self.nominal_frequency_khz), 2)
        q_drop_percent = round(
            max(0.0, (self.nominal_q_factor - measured_q_factor) / self.nominal_q_factor * 100.0),
            1,
        )

        metal_detected = (
            q_drop_percent >= self.config.fod_q_drop_threshold_percent
            or freq_shift >= self.config.fod_freq_shift_threshold_khz
        )

        interlock_state = WPTInterlockState.NORMAL
        power_inhibited = False
        reason = "Clean primary pad; all safety perimeters clear"

        if manual_emergency_trip:
            interlock_state = WPTInterlockState.EMERGENCY_SHUTDOWN
            power_inhibited = True
            reason = "EMERGENCY SHUTDOWN: Manual emergency stop tripped"
        elif metal_detected:
            interlock_state = WPTInterlockState.FOD_FAULT
            power_inhibited = True
            reason = (
                f"FOD FAULT: Metallic debris detected on primary pad "
                f"(Q-drop: {q_drop_percent}%, freq shift: {freq_shift} kHz)"
            )
        elif living_object_present:
            interlock_state = WPTInterlockState.LOD_FAULT
            power_inhibited = True
            reason = "LOD FAULT: Living entity detected within vehicle underbody perimeter"

        return FODStatus(
            metal_object_detected=metal_detected,
            frequency_shift_khz=freq_shift,
            q_drop_percent=q_drop_percent,
            living_object_detected=living_object_present,
            interlock_state=interlock_state,
            power_inhibited=power_inhibited,
            reason=reason,
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
