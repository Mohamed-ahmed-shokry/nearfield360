import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nearfield360.data.calibration import IntrinsicParameters
from nearfield360.geometry import fisheye
from nearfield360.geometry.fisheye import RadialPolynomialFisheye


def _intrinsics(
    *,
    coefficients: tuple[float, float, float, float] = (100.0, 0.0, 0.0, 0.0),
    aspect_ratio: float = 1.0,
    width: int = 1001,
    height: int = 1001,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> IntrinsicParameters:
    return IntrinsicParameters(
        aspect_ratio=aspect_ratio,
        cx_offset=cx_offset,
        cy_offset=cy_offset,
        height=height,
        k1=coefficients[0],
        k2=coefficients[1],
        k3=coefficients[2],
        k4=coefficients[3],
        model="radial_poly",
        poly_order=4,
        width=width,
    )


def test_explicit_angular_limit_is_required() -> None:
    with pytest.raises(TypeError, match="theta_max"):
        RadialPolynomialFisheye(_intrinsics())  # type: ignore[call-arg]


@pytest.mark.parametrize("theta_max", [0.0, -0.1, math.pi, math.inf, math.nan, True])
def test_invalid_angular_limit_is_rejected(theta_max: float) -> None:
    with pytest.raises(ValueError, match="theta_max"):
        RadialPolynomialFisheye(_intrinsics(), theta_max)


@pytest.mark.parametrize(
    ("coefficients", "theta_max"),
    [
        ((1.0, -1.0, 0.0, 0.0), 1.0),
        ((1.0, -0.5, 0.0, 0.0), 1.0),
        ((0.99, -1.0, 1.0 / 3.0, 0.0), 2.0),
        ((1.0, -3.0, 2.0, -0.25), 2.0),
    ],
)
def test_nonpositive_slope_is_rejected_even_between_endpoints(
    coefficients: tuple[float, float, float, float], theta_max: float
) -> None:
    with pytest.raises(ValueError, match="strictly positive derivative"):
        RadialPolynomialFisheye(_intrinsics(coefficients=coefficients), theta_max)


@pytest.mark.parametrize(
    "coefficients",
    [
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 1.5, -1.0, 0.25),
        (1.0, 0.0, 1.0, 0.0),
    ],
)
def test_monotonic_models_with_varied_derivative_extrema_are_accepted(
    coefficients: tuple[float, float, float, float],
) -> None:
    model = RadialPolynomialFisheye(_intrinsics(coefficients=coefficients), 2.0)
    expected = sum(coefficient * 2.0**order for order, coefficient in enumerate(coefficients, 1))
    assert model.radius_max == pytest.approx(expected)


def test_unrepresentable_radial_extent_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite, positive radius"):
        RadialPolynomialFisheye(_intrinsics(coefficients=(1e308, 0.0, 0.0, 0.0)), 2.0)


def test_projection_matches_analytic_equidistant_model() -> None:
    model = RadialPolynomialFisheye(_intrinsics(aspect_ratio=2.0), math.pi / 2.0)
    result = model.project([[1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [1.0, 1.0, 0.0]])
    radius_45 = 100.0 * math.pi / 4.0
    radius_90 = 100.0 * math.pi / 2.0
    assert_allclose(
        result.pixels,
        [
            [500.0 + radius_45, 500.0],
            [500.0, 500.0 - 2.0 * radius_45],
            [500.0 + radius_90 / math.sqrt(2.0), 500.0 + 2.0 * radius_90 / math.sqrt(2.0)],
        ],
        rtol=1e-14,
        atol=1e-12,
    )
    assert_array_equal(result.valid, [True, True, True])


def test_unprojection_matches_analytic_equidistant_model() -> None:
    model = RadialPolynomialFisheye(_intrinsics(aspect_ratio=2.0), 2.0)
    angle = math.pi / 4.0
    result = model.unproject([[500.0 + 100.0 * angle, 500.0], [500.0, 500.0 - 200.0 * angle]])
    assert_allclose(
        result.rays,
        [[math.sin(angle), 0.0, math.cos(angle)], [0.0, -math.sin(angle), math.cos(angle)]],
        atol=1e-14,
    )
    assert_array_equal(result.valid, [True, True])


def test_positive_optical_axis_uses_pixel_center_offsets() -> None:
    intrinsics = _intrinsics(width=201, height=101, cx_offset=1.25, cy_offset=-2.5)
    model = RadialPolynomialFisheye(intrinsics, 1.5)
    projected = model.project([0.0, 0.0, 2.0])
    assert projected.pixels.shape == (2,)
    assert projected.valid.shape == ()
    assert bool(projected.valid)
    assert_allclose(projected.pixels, [101.25, 47.5])
    unprojected = model.unproject(projected.pixels)
    assert unprojected.rays.shape == (3,)
    assert unprojected.valid.shape == ()
    assert bool(unprojected.valid)
    assert_array_equal(unprojected.rays, [0.0, 0.0, 1.0])


def test_negative_optical_axis_zero_and_nonfinite_vectors_are_invalid() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.5)
    result = model.project(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [math.nan, 1.0, 1.0], [1.0, math.inf, 1.0]],
        check_image_bounds=False,
    )
    assert not np.any(result.valid)
    assert np.isnan(result.pixels).all()


def test_negative_optical_axis_remains_invalid_with_limit_one_ulp_below_pi() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), float(np.nextafter(math.pi, 0.0)))
    result = model.project([0.0, 0.0, -1.0], check_image_bounds=False)
    assert not bool(result.valid)
    assert np.isnan(result.pixels).all()


def test_angular_boundary_is_inclusive_and_behind_camera_rays_are_supported() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.1)
    angles = np.array([math.pi / 2.0 + 0.1, model.theta_max, model.theta_max + 1e-6])
    rays = np.column_stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)))
    result = model.project(rays, check_image_bounds=False)
    assert_array_equal(result.valid, [True, True, False])
    assert np.isnan(result.pixels[2]).all()
    inverse = model.unproject(result.pixels[:2], check_image_bounds=False)
    assert np.all(inverse.valid)
    assert_allclose(inverse.rays, rays[:2], atol=1e-14)
    assert np.all(inverse.rays[:, 2] < 0.0)


def test_radial_boundary_is_inclusive_but_extrapolation_is_invalid() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.1)
    pixels = [[500.0 + model.radius_max, 500.0], [500.0 + model.radius_max + 1e-6, 500.0]]
    result = model.unproject(pixels, check_image_bounds=False)
    assert_array_equal(result.valid, [True, False])
    assert_allclose(result.rays[0], [math.sin(2.1), 0.0, math.cos(2.1)], atol=1e-14)
    assert np.isnan(result.rays[1]).all()


def test_image_bounds_are_half_open_and_optional_for_unprojection() -> None:
    model = RadialPolynomialFisheye(_intrinsics(width=101, height=101), 2.0)
    pixels = [[0.0, 0.0], [100.9, 100.9], [101.0, 50.0], [50.0, 101.0], [-0.01, 50.0]]
    filtered = model.unproject(pixels)
    unfiltered = model.unproject(pixels, check_image_bounds=False)
    assert_array_equal(filtered.valid, [True, True, False, False, False])
    assert np.isnan(filtered.rays[2:]).all()
    assert np.all(unfiltered.valid)


def test_image_bounds_are_optional_for_projection() -> None:
    model = RadialPolynomialFisheye(_intrinsics(width=101, height=101), 2.0)
    rays = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 0.0, 1.0]]
    filtered = model.project(rays)
    unfiltered = model.project(rays, check_image_bounds=False)
    assert not np.any(filtered.valid)
    assert np.isnan(filtered.pixels).all()
    assert np.all(unfiltered.valid)
    assert np.isfinite(unfiltered.pixels).all()


def test_fourth_order_roundtrip_preserves_direction_and_batch_shape() -> None:
    model = RadialPolynomialFisheye(
        _intrinsics(
            coefficients=(339.749, -31.988, 48.275, -7.201),
            aspect_ratio=1.2,
            cx_offset=3.942,
            cy_offset=-3.093,
            width=1280,
            height=966,
        ),
        2.2,
    )
    rng = np.random.default_rng(42)
    angles = rng.uniform(0.001, 2.199, size=(3, 7))
    azimuths = rng.uniform(-math.pi, math.pi, size=angles.shape)
    rays = np.stack(
        (np.sin(angles) * np.cos(azimuths), np.sin(angles) * np.sin(azimuths), np.cos(angles)),
        axis=-1,
    )
    projected = model.project(
        rays * rng.uniform(0.1, 100.0, size=(3, 7, 1)), check_image_bounds=False
    )
    unprojected = model.unproject(projected.pixels, check_image_bounds=False)
    assert projected.pixels.shape == (3, 7, 2)
    assert projected.valid.shape == (3, 7)
    assert unprojected.rays.shape == (3, 7, 3)
    assert unprojected.valid.shape == (3, 7)
    assert np.all(projected.valid)
    assert np.all(unprojected.valid)
    assert_allclose(unprojected.rays, rays, atol=2e-14)
    assert_allclose(np.linalg.norm(unprojected.rays, axis=-1), 1.0, atol=1e-14)


def test_projection_is_stable_for_huge_and_subnormal_directions() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    smallest = float(np.nextafter(0.0, 1.0))
    result = model.project([[1e308, 1e308, 1e308], [smallest, smallest, smallest]])
    reference = model.project([1.0, 1.0, 1.0])
    assert np.all(result.valid)
    assert_allclose(result.pixels, np.broadcast_to(reference.pixels, (2, 2)), atol=1e-13)


def test_unprojection_is_precise_near_optical_axis() -> None:
    model = RadialPolynomialFisheye(
        _intrinsics(coefficients=(1.0, 2.0, 3.0, 4.0), width=1, height=1), 2.0
    )
    result = model.unproject([[1e-200, 0.0], [0.0, 1e-30]], check_image_bounds=False)
    assert np.all(result.valid)
    assert_allclose(result.rays[:, 2], 1.0, atol=0.0)
    assert result.rays[0, 0] == pytest.approx(1e-200, rel=1e-13, abs=0.0)
    assert result.rays[1, 1] == pytest.approx(1e-30, rel=1e-13, abs=0.0)


def test_nonfinite_pixels_and_numeric_overflow_are_invalid_without_warnings() -> None:
    model = RadialPolynomialFisheye(_intrinsics(aspect_ratio=1e-300), 2.0)
    with np.errstate(all="raise"):
        result = model.unproject(
            [[math.nan, 500.0], [500.0, math.inf], [1e308, -1e308]], check_image_bounds=False
        )
    assert not np.any(result.valid)
    assert np.isnan(result.rays).all()


def test_projected_numeric_overflow_is_invalid_without_warnings() -> None:
    model = RadialPolynomialFisheye(_intrinsics(aspect_ratio=1e308), 2.0)
    with np.errstate(all="raise"):
        result = model.project([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], check_image_bounds=False)
    assert_array_equal(result.valid, [False, True])
    assert np.isnan(result.pixels[0]).all()


def test_nonconvergent_inverse_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RadialPolynomialFisheye(_intrinsics(coefficients=(100.0, 1.0, 2.0, 3.0)), 2.0)
    monkeypatch.setattr(fisheye, "_MAX_INVERSE_ITERATIONS", 0)
    result = model.unproject([[510.0, 500.0], [500.0, 500.0]])
    assert_array_equal(result.valid, [False, True])
    assert np.isnan(result.rays[0]).all()
    assert_array_equal(result.rays[1], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("values", [1.0, [], [[1.0, 2.0]], np.zeros((2, 4))])
def test_projection_rejects_wrong_coordinate_shape(values: object) -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    with pytest.raises(ValueError, match="shape"):
        model.project(values)  # type: ignore[arg-type]


@pytest.mark.parametrize("values", [1.0, [], [[1.0, 2.0, 3.0]], np.zeros((2, 4))])
def test_unprojection_rejects_wrong_coordinate_shape(values: object) -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    with pytest.raises(ValueError, match="shape"):
        model.unproject(values)  # type: ignore[arg-type]


def test_complex_coordinates_are_rejected() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    with pytest.raises(ValueError, match="real coordinates"):
        model.project([1.0 + 1.0j, 0.0, 1.0])
    with pytest.raises(ValueError, match="real coordinates"):
        model.unproject([1.0 + 1.0j, 0.0])


def test_empty_batches_preserve_shapes() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    projected = model.project(np.empty((2, 0, 3)))
    unprojected = model.unproject(np.empty((2, 0, 2)))
    assert projected.pixels.shape == (2, 0, 2)
    assert projected.valid.shape == (2, 0)
    assert unprojected.rays.shape == (2, 0, 3)
    assert unprojected.valid.shape == (2, 0)


def test_operations_do_not_mutate_input_arrays() -> None:
    model = RadialPolynomialFisheye(_intrinsics(), 2.0)
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, -1.0]])
    pixels = np.array([[501.0, 502.0], [-1000.0, 0.0]])
    original_points, original_pixels = points.copy(), pixels.copy()
    points.setflags(write=False)
    pixels.setflags(write=False)
    model.project(points)
    model.unproject(pixels)
    assert_array_equal(points, original_points)
    assert_array_equal(pixels, original_pixels)
