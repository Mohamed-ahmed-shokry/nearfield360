import numpy as np
import pytest

from nearfield360.geometry.bev import BevGrid
from nearfield360.occupancy.evidence import (
    OccupancyEvidence,
    OccupancyPolicy,
    OccupancyPolicyError,
    distance_weights,
    fuse_occupancy,
    rasterize_occupancy,
)


def _grid() -> BevGrid:
    return BevGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, resolution=0.5)


def test_policy_from_names_resolves_disjoint_classes() -> None:
    policy = OccupancyPolicy.from_names(["road", "curb"], ["vehicles", "person"])

    assert policy.free_label_ids == frozenset({1, 3})
    assert policy.occupied_label_ids == frozenset({6, 4})


def test_policy_rejects_unknown_overlapping_or_empty_classes() -> None:
    with pytest.raises(OccupancyPolicyError, match="Unknown semantic class 'sky'"):
        OccupancyPolicy.from_names(["road"], ["sky"])
    with pytest.raises(OccupancyPolicyError, match="cannot be free and occupied"):
        OccupancyPolicy(frozenset({1}), frozenset({1}))
    with pytest.raises(OccupancyPolicyError, match="unknown WoodScape semantic label: 99"):
        OccupancyPolicy(frozenset({99}), frozenset({6}))
    with pytest.raises(OccupancyPolicyError, match="needs free or occupied classes"):
        OccupancyPolicy(frozenset(), frozenset())


def test_rasterize_counts_free_and_occupied_votes() -> None:
    grid = _grid()
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))

    evidence = rasterize_occupancy(
        grid,
        points=np.array([[[0.2, 0.2], [1.4, 1.4]], [[0.8, 0.8], [1.8, 0.2]]]),
        labels=np.array([[1, 6], [1, 6]]),
        policy=policy,
    )

    assert evidence.observed[0, 0] == 1
    assert evidence.free[0, 0] == pytest.approx(1.0)
    assert evidence.occupied[2, 2] == pytest.approx(1.0)  # row=2 col=2
    assert evidence.observed.sum() == 4
    assert not np.isnan(evidence.occupancy()[0, 0])
    assert evidence.occupancy()[0, 0] == pytest.approx(0.0)
    assert evidence.occupancy()[2, 2] == pytest.approx(1.0)


def test_rasterize_mixed_votes_produce_ratio_and_unknown_cells_stay_nan() -> None:
    grid = _grid()
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))

    evidence = rasterize_occupancy(
        grid,
        points=np.array([[0.2, 0.2]]),
        labels=np.array([1]),
        policy=policy,
        weights=np.array([0.25]),
    )
    occupied_plus = rasterize_occupancy(
        grid,
        points=np.array([[0.2, 0.2]]),
        labels=np.array([6]),
        policy=policy,
        weights=np.array([0.75]),
    )

    fused = evidence.add(occupied_plus)
    assert fused.observed[0, 0] == 2
    assert fused.occupancy()[0, 0] == pytest.approx(0.75)
    others = fused.occupancy()
    assert np.isnan(others[1, 0]) and np.isnan(others[3, 3])


def test_rasterize_invalid_and_out_of_grid_votes_are_ignored() -> None:
    grid = _grid()
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))
    points = np.array([[0.2, 0.2], [5.0, 5.0], [np.nan, 0.2], [0.9, 0.9]])
    labels = np.array([1, 1, 1, 1])
    valid = np.array([True, True, True, False])

    evidence = rasterize_occupancy(grid, points=points, labels=labels, policy=policy, valid=valid)

    assert evidence.observed.sum() == 1
    assert evidence.free[0, 0] == pytest.approx(1.0)


def test_rasterize_validates_weights_only_on_voting_pixels() -> None:
    grid = _grid()
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))
    points = np.array([[0.2, 0.2], [0.9, 0.9]])
    labels = np.array([1, 1])
    valid = np.array([True, False])
    weights = np.array([0.5, np.nan])

    evidence = rasterize_occupancy(
        grid, points=points, labels=labels, policy=policy, valid=valid, weights=weights
    )

    assert evidence.free[0, 0] == pytest.approx(0.5)


def test_rasterize_rejects_out_of_range_weights_and_shape_mismatches() -> None:
    grid = _grid()
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))
    points = np.array([[0.2, 0.2]])
    labels = np.array([1])

    with pytest.raises(ValueError, match="weights must be finite"):
        rasterize_occupancy(grid, points, labels, policy, weights=np.array([np.nan]))
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        rasterize_occupancy(grid, points, labels, policy, weights=np.array([1.5]))
    with pytest.raises(ValueError, match="labels must be an integer array"):
        rasterize_occupancy(grid, points, labels.astype(float), policy)
    with pytest.raises(ValueError, match="weights must have shape"):
        rasterize_occupancy(grid, points, labels, policy, weights=np.array([0.5, 0.5]))


def test_fuse_occupancy_and_add_require_identical_grids() -> None:
    grid = _grid()
    other_grid = BevGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, resolution=0.25)
    policy = OccupancyPolicy(frozenset({1}), frozenset({6}))
    one = rasterize_occupancy(grid, points=[[0.2, 0.2]], labels=[1], policy=policy)
    two = rasterize_occupancy(grid, points=[[0.2, 0.2]], labels=[6], policy=policy)

    fused = fuse_occupancy([one, two, one])
    assert fused.observed[0, 0] == 3
    assert fused.occupied[0, 0] == pytest.approx(1.0)
    assert one.add(two).free[0, 0] == pytest.approx(1.0)

    mismatched = rasterize_occupancy(other_grid, points=[[0.2, 0.2]], labels=[1], policy=policy)
    with pytest.raises(ValueError, match="different BEV grids"):
        one.add(mismatched)
    with pytest.raises(ValueError, match="at least one evidence layer"):
        fuse_occupancy([])


def test_evidence_validates_arrays_and_rejects_bad_evidence() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="must have shape"):
        OccupancyEvidence(
            grid=grid,
            occupied=np.zeros((3, 3)),
            free=np.zeros((3, 3)),
            observed=np.zeros((3, 3), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="must be finite"):
        OccupancyEvidence(
            grid=grid,
            occupied=np.full(grid.shape, np.nan),
            free=np.zeros(grid.shape),
            observed=np.zeros(grid.shape, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        OccupancyEvidence(
            grid=grid,
            occupied=-np.ones(grid.shape),
            free=np.zeros(grid.shape),
            observed=np.zeros(grid.shape, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        OccupancyEvidence(
            grid=grid,
            occupied=np.zeros(grid.shape),
            free=np.zeros(grid.shape),
            observed=-np.ones(grid.shape, dtype=np.int64),
        )


def test_distance_weights_decay_with_distance_and_preserve_invalid() -> None:
    result = distance_weights(np.array([0.0, 2.0, 5.0, np.nan]), slope=0.5)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0 / 2.0)
    assert result[2] == pytest.approx(1.0 / 3.5)
    assert np.isnan(result[3])
    np.testing.assert_array_equal(distance_weights([0.0, 3.0]), np.ones(2))
    with pytest.raises(ValueError, match="slope"):
        distance_weights([1.0], slope=-1.0)
