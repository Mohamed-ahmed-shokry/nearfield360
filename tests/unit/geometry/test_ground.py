import math

import numpy as np
import pytest

from nearfield360.geometry.ground import intersect_ground


def test_downward_ray_hits_ground_with_euclidean_distance() -> None:
    result = intersect_ground([1.0, 2.0, 1.5], [0.0, 0.0, -1.0])

    assert result.points.shape == (3,)
    assert result.distances.shape == ()
    assert result.valid.shape == ()
    assert bool(result.valid)
    np.testing.assert_allclose(result.points, [1.0, 2.0, 0.0], atol=1e-15)
    assert float(result.distances) == pytest.approx(1.5)


def test_slanted_ray_and_custom_plane_height() -> None:
    result = intersect_ground(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
        [[1.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        ground_z=0.5,
    )

    assert result.valid.tolist() == [True, True]
    np.testing.assert_allclose(result.points[0], [0.5, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.points[1], [1.0, 1.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.distances, [0.5 * math.sqrt(2.0), 1.5], atol=1e-12)


def test_upward_and_parallel_rays_are_invalid() -> None:
    result = intersect_ground(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    assert not np.any(result.valid)
    assert np.isnan(result.points).all()
    assert np.isnan(result.distances).all()


def test_intersection_behind_origin_is_invalid() -> None:
    result = intersect_ground([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ground_z=0.0)

    assert not bool(result.valid)
    assert np.isnan(result.points).all()
    assert math.isnan(float(result.distances))


def test_origin_on_plane_with_downward_ray_is_valid_at_zero_distance() -> None:
    result = intersect_ground([3.0, -1.0, 0.0], [0.0, 0.0, -1.0])

    assert bool(result.valid)
    np.testing.assert_allclose(result.points, [3.0, -1.0, 0.0], atol=0.0)
    assert float(result.distances) == pytest.approx(0.0)


def test_nonfinite_and_zero_inputs_are_invalid_without_raising() -> None:
    result = intersect_ground(
        [[math.nan, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, -1.0], [math.inf, 0.0, -1.0], [0.0, 0.0, 0.0]],
    )

    assert not np.any(result.valid)
    assert np.isnan(result.points).all()


def test_max_distance_keeps_near_unknown_far() -> None:
    result = intersect_ground(
        [0.0, 0.0, 2.0], [[0.0, 0.0, -1.0], [3.0, 0.0, -1.0]], max_distance=2.5
    )

    assert result.valid.tolist() == [True, False]
    assert float(result.distances[0]) == pytest.approx(2.0)
    assert math.isnan(float(result.distances[1]))
    assert np.isnan(result.points[1]).all()


def test_broadcast_single_origin_against_many_directions() -> None:
    directions = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    result = intersect_ground([0.0, 0.0, 1.0], directions)

    assert result.points.shape == (2, 3)
    assert result.distances.shape == (2,)
    assert result.valid.shape == (2,)
    assert bool(np.all(result.valid))


def test_empty_batches_preserve_shapes() -> None:
    result = intersect_ground(np.empty((2, 0, 3)), np.empty((2, 0, 3)))

    assert result.points.shape == (2, 0, 3)
    assert result.distances.shape == (2, 0)
    assert result.valid.shape == (2, 0)


@pytest.mark.parametrize("value", [math.nan, math.inf, True, "0"])
def test_invalid_plane_parameters_raise(value: object) -> None:
    with pytest.raises(ValueError, match="ground_z"):
        intersect_ground([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], ground_z=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_distance"):
        intersect_ground(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            max_distance=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_nonpositive_max_distance_raises(value: float) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        intersect_ground([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], max_distance=value)


def test_mismatched_and_malformed_shapes_raise() -> None:
    with pytest.raises(ValueError, match="broadcastable"):
        intersect_ground(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"shape.*3"):
        intersect_ground([1.0, 2.0], [0.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="real numeric"):
        intersect_ground([1j, 0.0, 0.0], [0.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="real numeric"):
        intersect_ground([True, False, True], [0.0, 0.0, -1.0])


def test_inputs_are_not_mutated() -> None:
    origins = np.array([[0.0, 0.0, 1.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    origins.setflags(write=False)
    directions.setflags(write=False)
    original_origins = origins.copy()
    original_directions = directions.copy()

    intersect_ground(origins, directions)

    np.testing.assert_array_equal(origins, original_origins)
    np.testing.assert_array_equal(directions, original_directions)
