"""Unit tests for camera health-aware evidence discounting and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from nearfield360.geometry.bev import BevGrid
from nearfield360.health.discount import apply_health_discount
from nearfield360.health.models import (
    CameraHealthMetrics,
    CameraHealthReport,
    CameraHealthStatus,
    HealthAnomaly,
)
from nearfield360.occupancy.evidence import OccupancyEvidence


def _make_evidence(
    grid: BevGrid,
    occ_val: float = 1.0,
    free_val: float = 0.0,
    obs_val: int = 1,
) -> OccupancyEvidence:
    return OccupancyEvidence(
        grid=grid,
        occupied=np.full(grid.shape, occ_val, dtype=np.float64),
        free=np.full(grid.shape, free_val, dtype=np.float64),
        observed=np.full(grid.shape, obs_val, dtype=np.int64),
    )


def test_occupancy_evidence_scale_multiplies_mass() -> None:
    grid = BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=1.0)
    ev = _make_evidence(grid, occ_val=2.0, free_val=1.0, obs_val=3)

    scaled = ev.scale(0.5)
    np.testing.assert_allclose(scaled.occupied, 1.0)
    np.testing.assert_allclose(scaled.free, 0.5)
    np.testing.assert_array_equal(scaled.observed, 3)

    zero_scaled = ev.scale(0.0)
    np.testing.assert_allclose(zero_scaled.occupied, 0.0)
    np.testing.assert_allclose(zero_scaled.free, 0.0)
    np.testing.assert_array_equal(zero_scaled.observed, 3)


@pytest.mark.parametrize("bad_factor", [-0.5, float("nan"), float("inf"), "0.5", None])
def test_occupancy_evidence_scale_rejects_invalid_factors(bad_factor: object) -> None:
    grid = BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=1.0)
    ev = _make_evidence(grid)

    with pytest.raises(ValueError):
        ev.scale(bad_factor)  # type: ignore[arg-type]


def test_apply_health_discount_with_float() -> None:
    grid = BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=1.0)
    ev = _make_evidence(grid, occ_val=4.0, free_val=2.0)

    discounted = apply_health_discount(ev, 0.25)
    np.testing.assert_allclose(discounted.occupied, 1.0)
    np.testing.assert_allclose(discounted.free, 0.5)


def test_apply_health_discount_with_report() -> None:
    grid = BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=1.0)
    ev = _make_evidence(grid, occ_val=10.0, free_val=0.0)

    report = CameraHealthReport(
        camera="RV",
        status=CameraHealthStatus.DEGRADED,
        anomalies=(HealthAnomaly.LENS_SOILING,),
        metrics=CameraHealthMetrics(
            soiling_score=0.4,
            blur_score=50.0,
            mean_brightness=100.0,
            contrast=20.0,
            blockage_ratio=0.0,
            confidence=0.5,
        ),
        discount_weight=0.2,
    )

    discounted = apply_health_discount(ev, report)
    np.testing.assert_allclose(discounted.occupied, 2.0)
    np.testing.assert_allclose(discounted.free, 0.0)


def test_health_discounting_mitigates_degraded_camera_corruption() -> None:
    grid = BevGrid(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=1.0)
    # Healthy camera sees free space with high confidence
    healthy_ev = _make_evidence(grid, occ_val=0.0, free_val=1.0)
    # Soiled / degraded camera falsely predicts occupied space due to mud
    soiled_ev = _make_evidence(grid, occ_val=1.0, free_val=0.0)

    # Without health discounting: unweighted fusion gives 50% occupancy (uncertain conflict)
    unweighted = healthy_ev.add(soiled_ev)
    np.testing.assert_allclose(unweighted.occupancy(), 0.5)

    # With health discounting: soiled camera is discounted to 0.1
    discounted_soiled = apply_health_discount(soiled_ev, 0.1)
    weighted_fusion = healthy_ev.add(discounted_soiled)

    # Occupancy is now 0.1 / (1.0 + 0.1) = ~0.09 (correctly dominated by healthy camera)
    expected_occ = 0.1 / 1.1
    np.testing.assert_allclose(weighted_fusion.occupancy(), expected_occ, rtol=1e-5)
