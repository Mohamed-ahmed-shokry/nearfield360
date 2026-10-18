"""Track-to-track association and Covariance Intersection fusion engine."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from nearfield360.config import CoopConfig
from nearfield360.coop.alignment import Covariance2D, SpatiotemporalAligner
from nearfield360.coop.models import (
    CollectivePerceptionMessage,
    CooperativeTrack,
    CoopTrackSource,
    CPMPerceivedObject,
)


class MatchResult(NamedTuple):
    """Bipartite association result containing matched pairs and unmatched indices."""

    matched_pairs: list[tuple[int, int]]
    unmatched_first: list[int]
    unmatched_second: list[int]


def compute_mahalanobis_distance(
    pos1: tuple[float, float],
    cov1: Covariance2D,
    pos2: tuple[float, float],
    cov2: Covariance2D,
) -> float:
    """Compute the Mahalanobis distance between two 2D Gaussian spatial estimates.

    $D_M = \\sqrt{ (x_1 - x_2)^T (C_1 + C_2)^{-1} (x_1 - x_2) }$
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]

    # Combined innovation covariance S = C1 + C2
    s_xx = cov1.xx + cov2.xx
    s_yy = cov1.yy + cov2.yy
    s_xy = cov1.xy + cov2.xy

    det_s = s_xx * s_yy - s_xy**2
    if det_s <= 1e-9:
        # Singular or near-singular covariance fallback to Euclidean normalized distance
        norm_var = max(0.01, (s_xx + s_yy) / 2.0)
        return math.hypot(dx, dy) / math.sqrt(norm_var)

    # Inverse S^-1 = (1 / det) * [ s_yy, -s_xy; -s_xy, s_xx ]
    inv_xx = s_yy / det_s
    inv_yy = s_xx / det_s
    inv_xy = -s_xy / det_s

    # Quadratic form: d^T S^-1 d
    d_m_sq = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy)
    return math.sqrt(max(0.0, d_m_sq))


def associate_gated_greedy(
    first_positions: list[tuple[float, float]],
    first_covariances: list[Covariance2D],
    second_positions: list[tuple[float, float]],
    second_covariances: list[Covariance2D],
    gate_threshold: float = 3.0,
) -> MatchResult:
    """Perform greedy bipartite matching between two sets of tracks using Mahalanobis gating."""
    candidates: list[tuple[float, int, int]] = []

    for i, (p1, c1) in enumerate(zip(first_positions, first_covariances, strict=False)):
        for j, (p2, c2) in enumerate(zip(second_positions, second_covariances, strict=False)):
            d_m = compute_mahalanobis_distance(p1, c1, p2, c2)
            if d_m <= gate_threshold:
                candidates.append((d_m, i, j))

    # Sort candidates by increasing distance (greedy best-first)
    candidates.sort(key=lambda item: item[0])

    matched_pairs: list[tuple[int, int]] = []
    matched_first: set[int] = set()
    matched_second: set[int] = set()

    for _, i, j in candidates:
        if i not in matched_first and j not in matched_second:
            matched_pairs.append((i, j))
            matched_first.add(i)
            matched_second.add(j)

    unmatched_first = [i for i in range(len(first_positions)) if i not in matched_first]
    unmatched_second = [j for j in range(len(second_positions)) if j not in matched_second]

    return MatchResult(
        matched_pairs=matched_pairs,
        unmatched_first=unmatched_first,
        unmatched_second=unmatched_second,
    )


def invert_covariance_2d(cov: Covariance2D) -> tuple[float, float, float]:
    """Compute the matrix inverse of a 2x2 symmetric covariance matrix.

    Returns (inv_xx, inv_yy, inv_xy).
    """
    det = cov.xx * cov.yy - cov.xy**2
    safe_det = max(1e-9, det)
    return (cov.yy / safe_det, cov.xx / safe_det, -cov.xy / safe_det)


def fuse_covariance_intersection(
    pos1: tuple[float, float],
    cov1: Covariance2D,
    pos2: tuple[float, float],
    cov2: Covariance2D,
    omega: float | None = None,
    inflation_factor: float = 1.0,
) -> tuple[tuple[float, float], Covariance2D, float]:
    """Fuse two 2D Gaussian estimates with unknown cross-covariance using Covariance Intersection.

    Returns ((fused_x, fused_y), fused_cov, chosen_omega).
    """
    if omega is None:
        tr1 = cov1.xx + cov1.yy
        tr2 = cov2.xx + cov2.yy
        total_tr = tr1 + tr2
        raw_omega = tr2 / total_tr if total_tr > 1e-9 else 0.5
        chosen_omega = min(0.9, max(0.1, raw_omega))
    else:
        chosen_omega = min(0.99, max(0.01, omega))

    inv1_xx, inv1_yy, inv1_xy = invert_covariance_2d(cov1)
    inv2_xx, inv2_yy, inv2_xy = invert_covariance_2d(cov2)

    # P_F^-1 = omega * P1^-1 + (1 - omega) * P2^-1
    w1 = chosen_omega
    w2 = 1.0 - chosen_omega

    inv_f_xx = w1 * inv1_xx + w2 * inv2_xx
    inv_f_yy = w1 * inv1_yy + w2 * inv2_yy
    inv_f_xy = w1 * inv1_xy + w2 * inv2_xy

    det_inv_f = inv_f_xx * inv_f_yy - inv_f_xy**2
    safe_det = max(1e-9, det_inv_f)

    # Invert back to covariance P_F
    raw_f_xx = (inv_f_yy / safe_det) * inflation_factor
    raw_f_yy = (inv_f_xx / safe_det) * inflation_factor
    raw_f_xy = (-inv_f_xy / safe_det) * inflation_factor

    fused_cov = Covariance2D(
        xx=max(1e-4, raw_f_xx),
        yy=max(1e-4, raw_f_yy),
        xy=raw_f_xy,
    )

    # Information vector: y = omega * P1^-1 * x1 + (1 - omega) * P2^-1 * x2
    y_x = w1 * (inv1_xx * pos1[0] + inv1_xy * pos1[1]) + w2 * (
        inv2_xx * pos2[0] + inv2_xy * pos2[1]
    )
    y_y = w1 * (inv1_xy * pos1[0] + inv1_yy * pos1[1]) + w2 * (
        inv2_xy * pos2[0] + inv2_yy * pos2[1]
    )

    # Fused mean x_F = P_F * y (uninflated base covariance)
    uninflated_xx = inv_f_yy / safe_det
    uninflated_yy = inv_f_xx / safe_det
    uninflated_xy = -inv_f_xy / safe_det

    fused_x = uninflated_xx * y_x + uninflated_xy * y_y
    fused_y = uninflated_xy * y_x + uninflated_yy * y_y

    return ((fused_x, fused_y), fused_cov, chosen_omega)


def fuse_velocity_ci(
    v1: tuple[float, float],
    v2: tuple[float, float],
    omega: float,
) -> tuple[float, float]:
    """Fuse 2D velocities linearly using CI weighting factor omega."""
    w1 = omega
    w2 = 1.0 - omega
    return (w1 * v1[0] + w2 * v2[0], w1 * v1[1] + w2 * v2[1])


class CooperativeTrackManager:
    """Maintains cooperative track lifecycle and executes multi-sensor track-to-track fusion."""

    def __init__(self) -> None:
        self._tracks: dict[str, CooperativeTrack] = {}
        self._track_counter: int = 0
        self._last_timestamp_s: float = 0.0

    def get_active_tracks(self) -> list[CooperativeTrack]:
        """Return active cooperative tracks sorted by track ID."""
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def clear(self) -> None:
        """Clear all active tracks and reset counters."""
        self._tracks.clear()
        self._track_counter = 0

    def fuse_cycle(
        self,
        onboard_tracks: list[CooperativeTrack | dict[str, Any]],
        cpm_messages: list[CollectivePerceptionMessage],
        timestamp_s: float,
        config: CoopConfig | None = None,
    ) -> list[CooperativeTrack]:
        """Execute one complete cooperative track fusion cycle."""
        active_cfg = config or CoopConfig()
        self._last_timestamp_s = timestamp_s

        # 1. Normalize onboard tracks
        norm_onboard: list[CooperativeTrack] = []
        for item in onboard_tracks:
            if isinstance(item, CooperativeTrack):
                norm_onboard.append(item)
            elif isinstance(item, dict):
                norm_onboard.append(CooperativeTrack.model_validate(item))

        # 2. Extract and extrapolate valid CPM objects
        cpm_objects: list[CPMPerceivedObject] = []
        for msg in cpm_messages:
            for obj in msg.perceived_objects:
                if obj.confidence >= active_cfg.min_cpm_confidence:
                    extrap = SpatiotemporalAligner.extrapolate_cpm_object(
                        obj,
                        target_time_s=timestamp_s,
                        max_latency_s=active_cfg.max_latency_compensation_s,
                    )
                    cpm_objects.append(extrap)

        # 3. Associate onboard tracks with extrapolated CPM objects
        on_pos = [(t.x_m, t.y_m) for t in norm_onboard]
        on_cov = [Covariance2D(xx=t.cov_xx, yy=t.cov_yy, xy=t.cov_xy) for t in norm_onboard]
        cpm_pos = [(c.x_m, c.y_m) for c in cpm_objects]
        cpm_cov = [Covariance2D(xx=c.cov_xx, yy=c.cov_yy, xy=c.cov_xy) for c in cpm_objects]

        match_res = associate_gated_greedy(
            on_pos,
            on_cov,
            cpm_pos,
            cpm_cov,
            gate_threshold=active_cfg.association_gate_mahalanobis,
        )

        fused_tracks: list[CooperativeTrack] = []

        # 4. Handle matched pairs -> Fused Cooperative Tracks
        for i_on, j_cpm in match_res.matched_pairs:
            on_trk = norm_onboard[i_on]
            cpm_obj = cpm_objects[j_cpm]

            (fused_x, fused_y), fused_cov, omega = fuse_covariance_intersection(
                (on_trk.x_m, on_trk.y_m),
                Covariance2D(xx=on_trk.cov_xx, yy=on_trk.cov_yy, xy=on_trk.cov_xy),
                (cpm_obj.x_m, cpm_obj.y_m),
                Covariance2D(xx=cpm_obj.cov_xx, yy=cpm_obj.cov_yy, xy=cpm_obj.cov_xy),
                inflation_factor=active_cfg.track_covariance_inflation_factor,
            )
            fused_vx, fused_vy = fuse_velocity_ci(
                (on_trk.vx_mps, on_trk.vy_mps),
                (cpm_obj.vx_mps, cpm_obj.vy_mps),
                omega=omega,
            )

            combined_conf = min(0.99, max(on_trk.confidence, cpm_obj.confidence) + 0.05)
            best_cat = (
                on_trk.category if on_trk.confidence >= cpm_obj.confidence else cpm_obj.category
            )

            track = CooperativeTrack(
                track_id=on_trk.track_id,
                source=CoopTrackSource.FUSED_COOPERATIVE,
                x_m=fused_x,
                y_m=fused_y,
                vx_mps=fused_vx,
                vy_mps=fused_vy,
                cov_xx=fused_cov.xx,
                cov_yy=fused_cov.yy,
                cov_xy=fused_cov.xy,
                category=best_cat,
                confidence=combined_conf,
                age_s=on_trk.age_s + 0.1,
                hits=on_trk.hits + 1,
                onboard_track_id=on_trk.onboard_track_id or on_trk.track_id,
                rpu_station_id=cpm_obj.station_id,
                rpu_object_id=cpm_obj.object_id,
            )
            fused_tracks.append(track)
            self._tracks[track.track_id] = track

        # 5. Handle unmatched onboard tracks
        for i_on in match_res.unmatched_first:
            on_trk = norm_onboard[i_on]
            track = on_trk.model_copy(
                update={
                    "source": CoopTrackSource.ONBOARD_ONLY,
                    "age_s": on_trk.age_s + 0.1,
                }
            )
            fused_tracks.append(track)
            self._tracks[track.track_id] = track

        # 6. Handle unmatched CPM objects -> Infrastructure Only Tracks
        for j_cpm in match_res.unmatched_second:
            cpm_obj = cpm_objects[j_cpm]
            self._track_counter += 1
            tid = f"COOP-INFRA-{self._track_counter:04d}"
            track = CooperativeTrack(
                track_id=tid,
                source=CoopTrackSource.INFRASTRUCTURE_ONLY,
                x_m=cpm_obj.x_m,
                y_m=cpm_obj.y_m,
                vx_mps=cpm_obj.vx_mps,
                vy_mps=cpm_obj.vy_mps,
                cov_xx=cpm_obj.cov_xx,
                cov_yy=cpm_obj.cov_yy,
                cov_xy=cpm_obj.cov_xy,
                category=cpm_obj.category,
                confidence=cpm_obj.confidence,
                age_s=0.1,
                hits=1,
                rpu_station_id=cpm_obj.station_id,
                rpu_object_id=cpm_obj.object_id,
            )
            fused_tracks.append(track)
            self._tracks[track.track_id] = track

        return fused_tracks
