import numpy as np
import pytest

from nearfield360.geometry.bev import BevGrid


def _grid() -> BevGrid:
    return BevGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolution=0.5)


def test_shape_and_extent_follow_resolution() -> None:
    grid = _grid()

    assert grid.shape == (20, 20)
    assert grid.width == 20
    assert grid.height == 20
    assert grid.extent == (-5.0, 5.0, -5.0, 5.0)


def test_world_to_grid_uses_half_open_bounds() -> None:
    grid = BevGrid(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, resolution=0.5)

    result = grid.world_to_grid(
        [[0.0, 0.0], [0.49, 0.49], [0.5, 0.5], [0.99, 0.99], [1.0, 0.5], [0.5, 1.0]]
    )

    assert result.indices.tolist() == [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [-1, -1],
        [-1, -1],
    ]
    assert result.valid.tolist() == [True, True, True, True, False, False]


def test_world_to_grid_rejects_nonfinite_without_raising() -> None:
    grid = _grid()

    result = grid.world_to_grid([[float("nan"), 0.0], [0.0, float("inf")]])

    assert not np.any(result.valid)
    assert (result.indices == -1).all()


def test_world_to_grid_preserves_batch_shapes() -> None:
    grid = _grid()
    positions = np.zeros((2, 3, 2))

    result = grid.world_to_grid(positions)

    assert result.indices.shape == (2, 3, 2)
    assert result.valid.shape == (2, 3)
    assert bool(np.all(result.valid))
    # Origin maps to the middle cell of a symmetric grid.
    np.testing.assert_array_equal(result.indices[0, 0], [10, 10])


def test_grid_to_world_returns_cell_centers_and_roundtrip() -> None:
    grid = _grid()

    centers = grid.grid_to_world([[0, 0], [19, 19]])

    np.testing.assert_allclose(centers[0], [-4.75, -4.75], atol=1e-12)
    np.testing.assert_allclose(centers[1], [4.75, 4.75], atol=1e-12)
    roundtrip = grid.world_to_grid(centers)
    assert bool(np.all(roundtrip.valid))
    np.testing.assert_array_equal(roundtrip.indices, [[0, 0], [19, 19]])


def test_grid_to_world_rejects_out_of_range_and_non_integer() -> None:
    grid = _grid()

    with pytest.raises(ValueError, match="outside"):
        grid.grid_to_world([[20, 0]])
    with pytest.raises(ValueError, match="outside"):
        grid.grid_to_world([[-1, 0]])
    with pytest.raises(ValueError, match="integer"):
        grid.grid_to_world([[0.5, 0]])
    with pytest.raises(ValueError, match=r"shape.*2"):
        grid.grid_to_world([0, 0, 0])


def test_rasterize_counts_cells_and_ignores_unknown() -> None:
    grid = BevGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, resolution=1.0)

    counts = grid.rasterize([[0.2, 0.2], [0.8, 0.2], [1.5, 1.5], [5.0, 5.0]])

    assert counts.shape == (2, 2)
    assert counts.tolist() == [[2, 0], [0, 1]]


def test_rasterize_honors_caller_validity_mask() -> None:
    grid = BevGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0, resolution=1.0)

    counts = grid.rasterize([[0.2, 0.2], [0.8, 0.2], [1.5, 1.5]], valid=[True, False, True])

    assert counts.tolist() == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="must match"):
        grid.rasterize([[0.2, 0.2]], valid=[True, False])


def test_empty_counts_are_zeroed() -> None:
    grid = _grid()

    counts = grid.empty_counts()

    assert counts.shape == (20, 20)
    assert counts.dtype == np.int64
    assert not np.any(counts)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_min": 1.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": 0.5},
        {"x_min": 1.0, "x_max": 0.0, "y_min": 0.0, "y_max": 1.0, "resolution": 0.5},
        {"x_min": 0.0, "x_max": 1.0, "y_min": 1.0, "y_max": 0.0, "resolution": 0.5},
        {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": 0.0},
        {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": -0.5},
        {"x_min": float("nan"), "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": 0.5},
        {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": float("inf")},
        {"x_min": True, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "resolution": 0.5},
    ],
)
def test_invalid_grid_parameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BevGrid(**kwargs)  # type: ignore[arg-type]


def test_malformed_positions_raise() -> None:
    grid = _grid()

    with pytest.raises(ValueError, match="real numeric"):
        grid.world_to_grid([1j, 0.0])
    with pytest.raises(ValueError, match=r"shape.*2"):
        grid.world_to_grid([1.0, 2.0, 3.0])
