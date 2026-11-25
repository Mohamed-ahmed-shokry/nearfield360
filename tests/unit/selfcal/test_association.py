import numpy as np
import pytest

from nearfield360.selfcal.association import PolylineMap


def test_nearest_returns_exact_points_distances_and_normals() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (10.0, 0.0)], [(5.0, -5.0), (5.0, 5.0)]])

    result = polyline_map.nearest([[2.0, 1.0], [5.5, 3.0], [20.0, 0.0]])

    np.testing.assert_allclose(result.points, [[2.0, 0.0], [5.0, 3.0], [10.0, 0.0]])
    np.testing.assert_allclose(result.distances, [1.0, 0.5, 10.0])
    np.testing.assert_allclose(
        np.abs(result.normals), [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(result.normals, axis=1), 1.0)


def test_nearest_clamps_to_segment_endpoints() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (1.0, 0.0)]])

    result = polyline_map.nearest([[-3.0, 4.0]])

    np.testing.assert_allclose(result.points, [[0.0, 0.0]])
    np.testing.assert_allclose(result.distances, [5.0])


def test_multi_vertex_polylines_and_degenerate_segments() -> None:
    polyline_map = PolylineMap([[(0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]])

    assert polyline_map.num_segments == 2
    np.testing.assert_allclose(polyline_map.nearest([[4.5, 1.0]]).points, [[4.0, 1.0]])


def test_map_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        PolylineMap([[(0.0, 0.0)]])
    with pytest.raises(ValueError, match="non-finite"):
        PolylineMap([[(0.0, 0.0), (float("nan"), 1.0)]])
    with pytest.raises(ValueError, match="non-degenerate"):
        PolylineMap([[(1.0, 1.0), (1.0, 1.0)]])
    with pytest.raises(ValueError, match="non-degenerate"):
        PolylineMap([])
    with pytest.raises(ValueError, match="shape"):
        PolylineMap([[(0.0, 0.0), (1.0, 0.0)]]).nearest([1.0, 2.0])
