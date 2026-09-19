"""Multi-camera metric 2D obstacle tracker with Kalman filtering and association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nearfield360.config import TrackingConfig
from nearfield360.tracking.kalman import KalmanFilter2D
from nearfield360.tracking.models import (
    GroundFootprint,
    TrackedObstacle,
    TrackState,
)


class SingleTrack:
    """Internal mutable representation of an active obstacle track."""

    __slots__ = (
        "age",
        "class_id",
        "class_name",
        "filter",
        "history",
        "hits",
        "last_footprint",
        "state",
        "time_since_update",
        "track_id",
    )

    def __init__(
        self,
        track_id: int,
        footprint: GroundFootprint,
        config: TrackingConfig,
    ) -> None:
        self.track_id = track_id
        self.class_id = footprint.class_id
        self.class_name = footprint.class_name
        self.state = TrackState.CONFIRMED if config.min_hits <= 1 else TrackState.TENTATIVE
        self.filter = KalmanFilter2D(
            initial_pos=(footprint.center_x, footprint.center_y),
            dt=config.dt,
            process_noise_pos=config.process_noise_pos,
            process_noise_vel=config.process_noise_vel,
            measurement_noise=config.measurement_noise,
        )
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_footprint = footprint
        self.history: list[tuple[float, float]] = [(footprint.center_x, footprint.center_y)]

    def to_obstacle(self) -> TrackedObstacle:
        pos = self.filter.position
        vel = self.filter.velocity
        return TrackedObstacle(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            state=self.state,
            position=(round(pos[0], 3), round(pos[1], 3)),
            velocity=(round(vel[0], 3), round(vel[1], 3)),
            speed=round(self.filter.speed, 3),
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            history=tuple(self.history[-20:]),  # Keep recent history up to 20 frames
        )


class MultiObjectTracker:
    """Multi-camera metric 2D obstacle tracker operating in vehicle coordinates."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: dict[int, SingleTrack] = {}
        self._next_id = 1

    @property
    def active_tracks_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        footprints: Sequence[GroundFootprint],
    ) -> list[TrackedObstacle]:
        """Update tracker with ground footprints from current time step.

        Args:
            footprints: Sequence of GroundFootprint objects from any/all calibrated cameras.

        Returns:
            List of TrackedObstacle states representing current scene obstacles.
        """
        # Step 1: Predict state for all existing tracks
        for track in self._tracks.values():
            track.filter.predict()
            track.age += 1
            track.time_since_update += 1

        # Step 2: Associate tracks with measurements
        track_ids = list(self._tracks.keys())
        num_tracks = len(track_ids)
        num_measurements = len(footprints)

        matched_tracks: set[int] = set()
        matched_measurements: set[int] = set()
        matches: dict[int, int] = {}

        if num_tracks > 0 and num_measurements > 0:
            # Build cost matrix based on Euclidean distance on ground plane
            cost_matrix = np.full((num_tracks, num_measurements), np.inf, dtype=np.float64)
            for i, tid in enumerate(track_ids):
                t = self._tracks[tid]
                tx, ty = t.filter.position
                for j, det in enumerate(footprints):
                    # Forbid cross-class association
                    if t.class_id != det.class_id:
                        continue
                    dist = float(np.hypot(tx - det.center_x, ty - det.center_y))
                    if dist <= self.config.max_distance:
                        cost_matrix[i, j] = dist

            # Greedy minimum distance matching
            while True:
                min_val = float(np.min(cost_matrix))
                if np.isinf(min_val):
                    break
                row_idx, col_idx = np.unravel_index(
                    int(np.argmin(cost_matrix)), cost_matrix.shape
                )
                tid = track_ids[row_idx]
                col = int(col_idx)
                matches[tid] = col
                matched_tracks.add(tid)
                matched_measurements.add(col)

                # Invalidate row and column
                cost_matrix[row_idx, :] = np.inf
                cost_matrix[:, col_idx] = np.inf

        # Step 3: Update matched tracks
        for tid in matched_tracks:
            track = self._tracks[tid]
            det = footprints[matches[tid]]
            track.filter.update((det.center_x, det.center_y))
            track.hits += 1
            track.time_since_update = 0
            track.last_footprint = det
            pos = track.filter.position
            track.history.append((round(pos[0], 3), round(pos[1], 3)))
            if (track.state == TrackState.TENTATIVE and track.hits >= self.config.min_hits) or (
                track.state == TrackState.LOST
            ):
                track.state = TrackState.CONFIRMED

        # Step 4: Handle unmatched tracks
        unmatched_tids = set(track_ids) - matched_tracks
        for tid in unmatched_tids:
            track = self._tracks[tid]
            if track.time_since_update >= self.config.max_age:
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST

        # Step 5: Initialize new tracks for unmatched measurements
        unmatched_det_indices = set(range(num_measurements)) - matched_measurements
        for j in unmatched_det_indices:
            det = footprints[j]
            new_track = SingleTrack(self._next_id, det, self.config)
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # Step 6: Prune deleted tracks
        self._tracks = {
            tid: trk for tid, trk in self._tracks.items() if trk.state != TrackState.DELETED
        }

        # Step 7: Return all active obstacles (both CONFIRMED and TENTATIVE)
        return [trk.to_obstacle() for trk in self._tracks.values()]


__all__ = [
    "MultiObjectTracker",
    "SingleTrack",
]
