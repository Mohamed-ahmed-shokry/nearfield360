import numpy as np
import pytest

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.evidence import OccupancyEvidence
from nearfield360.occupancy.risk import (
    RiskZone,
    circular_zone,
    corridor_zone,
    risk_report,
    validate_zone_mask,
)


def _grid() -> BevGrid:
    return BevGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, resolution=0.5)


def test_corridor_zone_covers_the_forward_band() -> None:
    zone = corridor_zone(_grid(), front_length=1.0, half_width=0.25, start=0.0)

    assert zone.name == "forward_corridor"
    assert zone.mask.shape == (4, 4)
    assert zone.mask[0, 0] and zone.mask[0, 1]
    assert not zone.mask[0, 2]
    assert not zone.mask[1, 0]


def test_corridor_zone_is_half_open_on_the_forward_edge() -> None:
    zone = corridor_zone(_grid(), front_length=1.0, half_width=0.25, start=0.5)

    assert not zone.mask[0, 0]  # cell center x=0.25 < start
    assert zone.mask[0, 1]  # center 0.75 in [0.5, 1.5)
    assert zone.mask[0, 2]  # center 1.25 in [0.5, 1.5)
    assert not zone.mask[0, 3]  # center 1.75 >= 1.5


def test_corridor_zone_validates_parameters() -> None:
    with pytest.raises(ValueError, match="front_length"):
        corridor_zone(_grid(), front_length=0.0, half_width=0.5)
    with pytest.raises(ValueError, match="half_width"):
        corridor_zone(_grid(), front_length=1.0, half_width=-0.1)


def test_circular_zone_matches_cell_centers() -> None:
    zone = circular_zone(_grid(), center_xy=(0.25, 0.25), radius=0.3)

    assert zone.name == "circular_zone"
    assert zone.mask[0, 0]
    assert not zone.mask[0, 1]

    empty = circular_zone(_grid(), center_xy=(0.6, 0.6), radius=0.1)
    assert empty.mask.sum() == 0


def test_validate_zone_mask_enforces_grid_shape() -> None:
    zone = RiskZone(name="manual", mask=np.zeros((3, 3), dtype=np.bool_))

    with pytest.raises(ValueError, match="must match grid"):
        validate_zone_mask(zone.mask, _grid())

    with pytest.raises(ValueError, match="boolean array"):
        RiskZone(name="bad", mask=np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty string"):
        RiskZone(name="", mask=np.zeros((4, 4), dtype=np.bool_))


def _evidence(
    grid: BevGrid,
    observed: list[tuple[int, int]],
    *,
    occupied: list[tuple[int, int]] | None = None,
    free: list[tuple[int, int]] | None = None,
) -> OccupancyEvidence:
    occupied = occupied or []
    free = free or []
    occ = np.zeros(grid.shape, dtype=np.float64)
    fr = np.zeros(grid.shape, dtype=np.float64)
    obs = np.zeros(grid.shape, dtype=np.int64)
    for row, col in observed:
        obs[row, col] = 1
    for row, col in occupied:
        occ[row, col] = 1.0
    for row, col in free:
        fr[row, col] = 1.0
    return OccupancyEvidence(grid=grid, occupied=occ, free=fr, observed=obs)


def test_risk_report_counts_dangerous_cells_within_zones() -> None:
    grid = _grid()
    evidence = _evidence(
        grid,
        observed=[(0, 0), (0, 1), (2, 2)],
        occupied=[(0, 0), (2, 2)],
        free=[(0, 1)],
    )
    corridor = corridor_zone(grid, front_length=1.0, half_width=0.25, start=0.0)

    reports = risk_report([corridor], evidence)
    corridor_risk = reports[0]

    assert corridor_risk.name == "forward_corridor"
    assert corridor_risk.cells == 2
    assert corridor_risk.observed_cells == 2
    assert corridor_risk.occupied_cells == 1
    assert corridor_risk.area_m2 == pytest.approx(0.5)
    assert corridor_risk.occupied_area_m2 == pytest.approx(0.25)
    assert corridor_risk.mean_occupancy == pytest.approx(0.5)
    assert corridor_risk.max_occupancy == pytest.approx(1.0)


def test_risk_report_applies_min_evidence_and_danger_thresholds() -> None:
    grid = _grid()
    evidence = _evidence(grid, observed=[(0, 0)], occupied=[(0, 0)])
    corridor = corridor_zone(grid, front_length=1.0, half_width=0.25, start=0.0)

    strict_min = risk_report([corridor], evidence, min_evidence=2)[0]
    assert strict_min.observed_cells == 0
    assert strict_min.mean_occupancy is None

    high_bar = risk_report([corridor], evidence, danger_occupancy=1.0)[0]
    assert high_bar.occupied_cells == 0
    assert high_bar.max_occupancy == pytest.approx(1.0)


def test_risk_report_without_observed_zone_cells_uses_none() -> None:
    grid = _grid()
    evidence = _evidence(grid, observed=[(2, 2)], occupied=[(2, 2)])
    corridor = corridor_zone(grid, front_length=1.0, half_width=0.25, start=0.0)

    report = risk_report([corridor], evidence)[0]
    assert report.observed_cells == 0
    assert report.mean_occupancy is None
    assert report.max_occupancy is None
    assert report.mean_uncertainty is None
    assert report.max_uncertainty is None


def test_risk_report_computes_uncertainty_metrics() -> None:
    grid = _grid()
    evidence = _evidence(
        grid,
        observed=[(0, 0), (0, 1)],
        occupied=[(0, 0)],
        free=[(0, 1)],
    )
    corridor = corridor_zone(grid, front_length=1.0, half_width=0.25, start=0.0)

    report = risk_report([corridor], evidence)[0]
    assert report.mean_uncertainty is not None
    assert report.max_uncertainty is not None
    assert 0.0 < report.mean_uncertainty < 1.0
    assert 0.0 < report.max_uncertainty < 1.0
    assert report.max_uncertainty >= report.mean_uncertainty


def test_risk_report_validates_parameters() -> None:
    grid = _grid()
    evidence = _evidence(grid, observed=[(0, 0)], occupied=[(0, 0)])
    corridor = corridor_zone(grid, front_length=1.0, half_width=0.25)

    with pytest.raises(ValueError, match="min_evidence"):
        risk_report([corridor], evidence, min_evidence=0)
    with pytest.raises(ValueError, match="danger_occupancy"):
        risk_report([corridor], evidence, danger_occupancy=1.5)
