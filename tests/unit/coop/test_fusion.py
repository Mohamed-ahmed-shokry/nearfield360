"""Unit tests for track association, Covariance Intersection fusion, and track manager."""

import math

from nearfield360.config import CoopConfig
from nearfield360.coop.alignment import Covariance2D
from nearfield360.coop.fusion import (
    CooperativeTrackManager,
    associate_gated_greedy,
    compute_mahalanobis_distance,
    fuse_covariance_intersection,
    fuse_velocity_ci,
)
from nearfield360.coop.models import (
    CollectivePerceptionMessage,
    CooperativeTrack,
    CoopTrackSource,
    CPMPerceivedObject,
    PerceivedObjectClass,
)


def test_mahalanobis_distance() -> None:
    pos1 = (10.0, 10.0)
    cov1 = Covariance2D(xx=1.0, yy=1.0, xy=0.0)
    pos2 = (10.0, 10.0)
    cov2 = Covariance2D(xx=1.0, yy=1.0, xy=0.0)

    # Identical position -> distance 0
    d0 = compute_mahalanobis_distance(pos1, cov1, pos2, cov2)
    assert math.isclose(d0, 0.0, abs_tol=1e-5)

    # Offset by dx = 2.0. S = cov1 + cov2 = diag(2, 2). S^-1 = diag(0.5, 0.5)
    # d^T S^-1 d = 4 * 0.5 = 2.0 -> D_M = sqrt(2) approx 1.4142
    pos_offset = (12.0, 10.0)
    d_offset = compute_mahalanobis_distance(pos1, cov1, pos_offset, cov2)
    assert math.isclose(d_offset, math.sqrt(2.0), abs_tol=1e-4)


def test_associate_gated_greedy() -> None:
    first_pos = [(0.0, 0.0), (10.0, 10.0)]
    first_cov = [Covariance2D(xx=0.5, yy=0.5, xy=0.0) for _ in range(2)]

    second_pos = [(0.2, 0.1), (50.0, 50.0)]  # second[0] is close to first[0], second[1] is far
    second_cov = [Covariance2D(xx=0.5, yy=0.5, xy=0.0) for _ in range(2)]

    res = associate_gated_greedy(
        first_pos,
        first_cov,
        second_pos,
        second_cov,
        gate_threshold=3.0,
    )

    assert res.matched_pairs == [(0, 0)]
    assert res.unmatched_first == [1]
    assert res.unmatched_second == [1]


def test_covariance_intersection_symmetric() -> None:
    pos1 = (10.0, 5.0)
    cov1 = Covariance2D(xx=1.0, yy=1.0, xy=0.0)
    pos2 = (12.0, 5.0)
    cov2 = Covariance2D(xx=1.0, yy=1.0, xy=0.0)

    # Symmetric estimates should yield midpoint (11.0, 5.0) with omega = 0.5
    (fx, fy), fcov, omega = fuse_covariance_intersection(pos1, cov1, pos2, cov2, omega=0.5)
    assert math.isclose(fx, 11.0, abs_tol=1e-4)
    assert math.isclose(fy, 5.0, abs_tol=1e-4)
    assert math.isclose(omega, 0.5, abs_tol=1e-4)
    # Fused covariance should be less than or equal to individual
    assert fcov.xx <= cov1.xx
    assert fcov.yy <= cov1.yy


def test_covariance_intersection_with_inflation() -> None:
    pos1 = (0.0, 0.0)
    cov1 = Covariance2D(xx=0.5, yy=0.5, xy=0.0)
    pos2 = (0.0, 0.0)
    cov2 = Covariance2D(xx=0.5, yy=0.5, xy=0.0)

    _, fcov_base, _ = fuse_covariance_intersection(pos1, cov1, pos2, cov2, inflation_factor=1.0)
    _, fcov_inflated, _ = fuse_covariance_intersection(
        pos1, cov1, pos2, cov2, inflation_factor=1.25
    )

    assert fcov_inflated.xx > fcov_base.xx
    assert math.isclose(fcov_inflated.xx, fcov_base.xx * 1.25, abs_tol=1e-4)


def test_fuse_velocity_ci() -> None:
    v1 = (2.0, 0.0)
    v2 = (4.0, 2.0)
    fused_v = fuse_velocity_ci(v1, v2, omega=0.5)
    assert math.isclose(fused_v[0], 3.0, abs_tol=1e-5)
    assert math.isclose(fused_v[1], 1.0, abs_tol=1e-5)


def test_cooperative_track_manager_cycle() -> None:
    mgr = CooperativeTrackManager()
    cfg = CoopConfig()

    onboard_track = CooperativeTrack(
        track_id="TRK-ONB-1",
        source=CoopTrackSource.ONBOARD_ONLY,
        x_m=10.0,
        y_m=10.0,
        vx_mps=1.0,
        vy_mps=0.0,
        cov_xx=0.30,
        cov_yy=0.30,
        category=PerceivedObjectClass.VEHICLE,
        confidence=0.85,
    )

    # CPM object close to TRK-ONB-1 (at 10.1, 9.9) -> should match and fuse
    cpm_matched = CPMPerceivedObject(
        object_id="CPM-MATCHED",
        station_id="RPU-01",
        timestamp_s=5.0,
        category=PerceivedObjectClass.VEHICLE,
        x_m=10.1,
        y_m=9.9,
        vx_mps=1.1,
        vy_mps=0.1,
        confidence=0.90,
    )

    # CPM object far away (at 40.0, 40.0) -> should be unmatched and initialized
    # as INFRASTRUCTURE_ONLY
    cpm_unmatched = CPMPerceivedObject(
        object_id="CPM-UNMATCHED",
        station_id="RPU-01",
        timestamp_s=5.0,
        category=PerceivedObjectClass.PEDESTRIAN,
        x_m=40.0,
        y_m=40.0,
        vx_mps=0.0,
        vy_mps=1.0,
        confidence=0.88,
    )

    msg = CollectivePerceptionMessage(
        station_id="RPU-01",
        sequence_number=1,
        timestamp_s=5.0,
        perceived_objects=[cpm_matched, cpm_unmatched],
    )

    active_tracks = mgr.fuse_cycle(
        onboard_tracks=[onboard_track],
        cpm_messages=[msg],
        timestamp_s=5.0,
        config=cfg,
    )

    assert len(active_tracks) == 2
    fused = next(t for t in active_tracks if t.track_id == "TRK-ONB-1")
    assert fused.source == CoopTrackSource.FUSED_COOPERATIVE
    assert fused.hits == 2
    assert fused.rpu_object_id == "CPM-MATCHED"

    infra = next(t for t in active_tracks if t.source == CoopTrackSource.INFRASTRUCTURE_ONLY)
    assert infra.rpu_object_id == "CPM-UNMATCHED"
    assert infra.category == PerceivedObjectClass.PEDESTRIAN

    mgr.clear()
    assert len(mgr.get_active_tracks()) == 0
