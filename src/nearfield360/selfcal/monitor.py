"""Per-camera extrinsic drift tracking with EWMA smoothing and status gating."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from nearfield360.config import SelfCalConfig
from nearfield360.data.woodscape import CameraId
from nearfield360.selfcal.models import (
    CalibrationEstimate,
    DriftReport,
    DriftStatus,
    RotationDelta,
)

TRUSTED_STATUSES = frozenset({DriftStatus.NOMINAL, DriftStatus.WARNING})


@dataclass
class _CameraState:
    smoothed: npt.NDArray[np.float64] | None = None
    accepted: int = 0
    rejected: int = 0
    history: list[float] = field(default_factory=list)


class DriftMonitor:
    """Track each camera's smoothed misalignment and classify its severity.

    Only observable, converged estimates update the EWMA; low-confidence
    estimates are counted as rejected and leave the previous state untouched,
    so one bad frame can neither raise nor clear an alarm. A camera with no
    accepted estimate yet reports ``UNOBSERVABLE``.
    """

    def __init__(self, config: SelfCalConfig | None = None) -> None:
        self.config = config or SelfCalConfig()
        self._states: dict[CameraId, _CameraState] = {}

    def update(self, estimate: CalibrationEstimate) -> DriftReport:
        """Fold one estimate into the camera's state and return its report."""
        state = self._states.setdefault(estimate.camera_id, _CameraState())
        if estimate.observable and estimate.converged:
            vector = estimate.delta.as_vector()
            if state.smoothed is None:
                state.smoothed = vector
            else:
                alpha = self.config.ewma_alpha
                state.smoothed = alpha * vector + (1.0 - alpha) * state.smoothed
            state.accepted += 1
            state.history.append(float(np.degrees(np.linalg.norm(state.smoothed))))
        else:
            state.rejected += 1
        return self._report(estimate.camera_id, state)

    def report(self, camera_id: CameraId) -> DriftReport | None:
        """Return the current report for a camera, or None if never updated."""
        state = self._states.get(camera_id)
        return None if state is None else self._report(camera_id, state)

    def reports(self) -> dict[CameraId, DriftReport]:
        """Return reports for every camera seen so far."""
        return {camera: self._report(camera, state) for camera, state in self._states.items()}

    def history_deg(self, camera_id: CameraId) -> list[float]:
        """Smoothed misalignment magnitude (degrees) after each accepted update."""
        state = self._states.get(camera_id)
        return [] if state is None else list(state.history)

    def is_trusted(self, camera_id: CameraId) -> bool:
        """True when the camera is calibrated well enough to feed perception."""
        report = self.report(camera_id)
        return report is not None and report.status in TRUSTED_STATUSES

    def reset(self, camera_id: CameraId) -> None:
        """Forget a camera's history, e.g. after a successful recalibration."""
        self._states.pop(camera_id, None)

    def classify(self, magnitude_deg: float) -> DriftStatus:
        """Map a misalignment magnitude in degrees to a status."""
        if magnitude_deg >= self.config.fault_deg:
            return DriftStatus.FAULT
        if magnitude_deg >= self.config.recalibrate_deg:
            return DriftStatus.RECALIBRATE
        if magnitude_deg >= self.config.warning_deg:
            return DriftStatus.WARNING
        return DriftStatus.NOMINAL

    def _report(self, camera_id: CameraId, state: _CameraState) -> DriftReport:
        if state.smoothed is None:
            return DriftReport(
                camera_id=camera_id,
                status=DriftStatus.UNOBSERVABLE,
                smoothed_delta=RotationDelta(),
                magnitude_deg=0.0,
                accepted_updates=state.accepted,
                rejected_updates=state.rejected,
            )
        delta = RotationDelta.from_vector(state.smoothed)
        return DriftReport(
            camera_id=camera_id,
            status=self.classify(delta.magnitude_deg),
            smoothed_delta=delta,
            magnitude_deg=delta.magnitude_deg,
            accepted_updates=state.accepted,
            rejected_updates=state.rejected,
        )
