"""Track-to-track association and Covariance Intersection fusion engine."""

from __future__ import annotations

import math
from typing import NamedTuple

from nearfield360.coop.alignment import Covariance2D


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
