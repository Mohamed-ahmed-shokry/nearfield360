"""Spatial slot tracking and multi-frame temporal memory for automated parking."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from nearfield360.config import MissionConfig
from nearfield360.mission.models import TrackedParkingSlot
from nearfield360.planning.kinematics import normalize_angle
from nearfield360.slots.models import (
    ParkingSlot,
    ParkingSlotCorner,
    SlotOccupancyStatus,
)


def _transform_slot(
    slot: ParkingSlot,
    dx: float,
    dy: float,
    d_theta: float,
) -> ParkingSlot:
    """Transform slot corners, center, and heading under vehicle relative displacement."""
    cos_t = math.cos(-d_theta)
    sin_t = math.sin(-d_theta)

    def _transform_point(x: float, y: float) -> tuple[float, float]:
        tx = x - dx
        ty = y - dy
        return (cos_t * tx - sin_t * ty, sin_t * tx + cos_t * ty)

    new_center = _transform_point(slot.center[0], slot.center[1])
    new_corners = tuple(
        ParkingSlotCorner(
            x=p[0],
            y=p[1],
            z=corner.z,
        )
        for corner, p in [(c, _transform_point(c.x, c.y)) for c in slot.corners]
    )
    if len(new_corners) != 4:
        raise ValueError("Transformed slot must have exactly 4 corners")

    new_heading = normalize_angle(slot.heading_rad - d_theta)

    return ParkingSlot(
        slot_id=slot.slot_id,
        slot_type=slot.slot_type,
        corners=(new_corners[0], new_corners[1], new_corners[2], new_corners[3]),
        center=(round(new_center[0], 3), round(new_center[1], 3)),
        heading_rad=round(new_heading, 4),
        width_m=slot.width_m,
        length_m=slot.length_m,
        status=slot.status,
        occupancy_ratio=slot.occupancy_ratio,
        uncertainty_ratio=slot.uncertainty_ratio,
        confidence=slot.confidence,
        approach_path=slot.approach_path,
    )


class SlotTracker:
    """Maintains persistent tracked parking slots across multi-frame observation streams."""

    def __init__(self, config: MissionConfig | None = None) -> None:
        self.config = config or MissionConfig()
        self._tracks: dict[str, TrackedParkingSlot] = {}
        self._next_track_id: int = 1

    @property
    def tracks(self) -> dict[str, TrackedParkingSlot]:
        """Current active tracked slots indexed by track ID."""
        return self._tracks

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: Sequence[ParkingSlot],
        frame_index: int = 0,
        ego_displacement: tuple[float, float, float] | None = None,
    ) -> list[TrackedParkingSlot]:
        """Update tracker with new frame observations and ego motion displacement.

        ego_displacement: relative motion (dx, dy, d_theta) since last frame.
        """
        # 1. Transform existing tracks if ego vehicle moved
        if ego_displacement is not None:
            dx, dy, d_theta = ego_displacement
            if abs(dx) > 1e-4 or abs(dy) > 1e-4 or abs(d_theta) > 1e-4:
                updated_tracks = {}
                for tid, trk in self._tracks.items():
                    transformed_slot = _transform_slot(trk.slot, dx, dy, d_theta)
                    updated_tracks[tid] = TrackedParkingSlot(
                        track_id=trk.track_id,
                        slot=transformed_slot,
                        first_observed_frame=trk.first_observed_frame,
                        last_observed_frame=trk.last_observed_frame,
                        hit_count=trk.hit_count,
                        miss_count=trk.miss_count,
                        stability_score=trk.stability_score,
                        is_confirmed=trk.is_confirmed,
                    )
                self._tracks = updated_tracks

        # 2. Match detections to existing tracks
        matched_track_ids: set[str] = set()
        matched_detection_indices: set[int] = set()

        if self._tracks and detections:
            track_keys = list(self._tracks.keys())
            cost_matrix = np.full((len(track_keys), len(detections)), float("inf"))

            for i, tid in enumerate(track_keys):
                trk_slot = self._tracks[tid].slot
                for j, det_slot in enumerate(detections):
                    dist = math.hypot(
                        trk_slot.center[0] - det_slot.center[0],
                        trk_slot.center[1] - det_slot.center[1],
                    )
                    heading_diff = abs(normalize_angle(trk_slot.heading_rad - det_slot.heading_rad))
                    # Gate by distance and heading alignment
                    if (
                        dist <= self.config.slot_tracking_distance_gate_m
                        and heading_diff <= math.pi / 3
                    ):
                        cost_matrix[i, j] = dist + 0.5 * heading_diff

            # Greedy matching in order of lowest cost
            while True:
                min_val = np.min(cost_matrix)
                if min_val >= float("inf"):
                    break
                min_i, min_j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                cost_matrix[min_i, :] = float("inf")
                cost_matrix[:, min_j] = float("inf")

                tid = track_keys[int(min_i)]
                matched_track_ids.add(tid)
                matched_detection_indices.add(int(min_j))

                # Update matched track
                old_trk = self._tracks[tid]
                det = detections[int(min_j)]
                alpha = 0.4  # Smoothing factor for center

                smoothed_center = (
                    round((1.0 - alpha) * old_trk.slot.center[0] + alpha * det.center[0], 3),
                    round((1.0 - alpha) * old_trk.slot.center[1] + alpha * det.center[1], 3),
                )
                smoothed_heading = normalize_angle(
                    (1.0 - alpha) * old_trk.slot.heading_rad + alpha * det.heading_rad
                )
                new_hit_count = old_trk.hit_count + 1
                new_miss_count = 0
                new_stability = min(1.0, old_trk.stability_score + 0.1)
                is_conf = old_trk.is_confirmed or (new_hit_count >= self.config.slot_confirm_frames)

                # Prioritize occupied status if detected as occupied
                status = (
                    det.status
                    if det.status == SlotOccupancyStatus.OCCUPIED
                    else old_trk.slot.status
                )
                if det.confidence >= old_trk.slot.confidence:
                    status = det.status

                updated_slot = ParkingSlot(
                    slot_id=old_trk.slot.slot_id,
                    slot_type=det.slot_type,
                    corners=det.corners,
                    center=smoothed_center,
                    heading_rad=round(smoothed_heading, 4),
                    width_m=det.width_m,
                    length_m=det.length_m,
                    status=status,
                    occupancy_ratio=det.occupancy_ratio,
                    uncertainty_ratio=det.uncertainty_ratio,
                    confidence=max(old_trk.slot.confidence, det.confidence),
                    approach_path=det.approach_path or old_trk.slot.approach_path,
                )

                self._tracks[tid] = TrackedParkingSlot(
                    track_id=tid,
                    slot=updated_slot,
                    first_observed_frame=old_trk.first_observed_frame,
                    last_observed_frame=frame_index,
                    hit_count=new_hit_count,
                    miss_count=new_miss_count,
                    stability_score=round(new_stability, 3),
                    is_confirmed=is_conf,
                )

        # 3. Update unmatched tracks (miss)
        prune_ids: list[str] = []
        for tid, trk in self._tracks.items():
            if tid not in matched_track_ids:
                new_miss = trk.miss_count + 1
                new_stability = max(0.0, trk.stability_score - 0.2)
                if new_miss > self.config.slot_max_miss_frames:
                    prune_ids.append(tid)
                else:
                    self._tracks[tid] = TrackedParkingSlot(
                        track_id=trk.track_id,
                        slot=trk.slot,
                        first_observed_frame=trk.first_observed_frame,
                        last_observed_frame=trk.last_observed_frame,
                        hit_count=trk.hit_count,
                        miss_count=new_miss,
                        stability_score=round(new_stability, 3),
                        is_confirmed=trk.is_confirmed,
                    )

        for tid in prune_ids:
            del self._tracks[tid]

        # 4. Initialize new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detection_indices:
                tid = f"track_{self._next_track_id:03d}"
                self._next_track_id += 1
                is_conf = self.config.slot_confirm_frames <= 1
                self._tracks[tid] = TrackedParkingSlot(
                    track_id=tid,
                    slot=det,
                    first_observed_frame=frame_index,
                    last_observed_frame=frame_index,
                    hit_count=1,
                    miss_count=0,
                    stability_score=0.8,
                    is_confirmed=is_conf,
                )

        return list(self._tracks.values())

    def get_active_tracks(self, confirmed_only: bool = False) -> list[TrackedParkingSlot]:
        """Return list of current active slot tracks."""
        tracks = list(self._tracks.values())
        if confirmed_only:
            return [t for t in tracks if t.is_confirmed]
        return tracks

    def get_vacant_slots(self, confirmed_only: bool = True) -> list[TrackedParkingSlot]:
        """Return tracked slots classified as vacant."""
        active = self.get_active_tracks(confirmed_only=confirmed_only)
        return [t for t in active if t.slot.status == SlotOccupancyStatus.VACANT]

    def get_best_target_slot(
        self,
        current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        confirmed_only: bool = True,
    ) -> TrackedParkingSlot | None:
        """Select the highest scoring vacant parking slot for autonomous docking."""
        vacant = self.get_vacant_slots(confirmed_only=confirmed_only)
        if not vacant:
            return None

        cx, cy, _chead = current_pose

        def _score(tracked: TrackedParkingSlot) -> float:
            sx, sy = tracked.slot.center
            dist = math.hypot(sx - cx, sy - cy)
            # Favor slots ahead or slightly to the side over slots behind
            ahead_bonus = 2.0 if (sx - cx) >= 0.0 else -1.0
            # Higher confidence and stability increase score
            conf_score = tracked.slot.confidence * 3.0 + tracked.stability_score * 2.0
            # Feasibility bonus
            feasibility_bonus = (
                2.0
                if tracked.slot.approach_path and tracked.slot.approach_path.is_feasible
                else 0.0
            )
            # Penalty for distance
            dist_penalty = -0.5 * dist
            return ahead_bonus + conf_score + feasibility_bonus + dist_penalty

        return max(vacant, key=_score)
