import math
from dataclasses import FrozenInstanceError

import numpy as np
import numpy.typing as npt
import pytest

from nearfield360.geometry.transforms import RigidTransform


def test_identity_preserves_points_and_directions() -> None:
    transform = RigidTransform.identity()
    points = np.array([[1, 2, 3], [-4, 5, 0]])

    np.testing.assert_array_equal(transform.rotation, np.eye(3))
    np.testing.assert_array_equal(transform.translation, np.zeros(3))
    np.testing.assert_array_equal(transform.transform_points(points), points)
    np.testing.assert_array_equal(transform.transform_directions(points), points)
    assert transform.transform_points(points).dtype == np.float64


@pytest.mark.parametrize(
    ("quaternion", "point", "expected"),
    [
        ((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (0, 1, 0), (0, 0, 1)),
        ((0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)), (0, 0, 1), (1, 0, 0)),
        ((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), (1, 0, 0), (0, 1, 0)),
    ],
)
def test_quaternion_uses_xyzw_and_right_handed_active_rotation(
    quaternion: npt.ArrayLike, point: npt.ArrayLike, expected: npt.ArrayLike
) -> None:
    transform = RigidTransform.from_quaternion(quaternion)

    np.testing.assert_allclose(
        transform.transform_points(point), np.asarray(expected, dtype=np.float64), atol=1e-15
    )
    assert np.linalg.det(transform.rotation) == pytest.approx(1.0)


def test_translation_applies_only_to_points() -> None:
    transform = RigidTransform.from_quaternion(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), (4.0, -5.0, 6.0)
    )

    np.testing.assert_allclose(transform.transform_points((2, 0, 0)), (4, -3, 6), atol=1e-15)
    np.testing.assert_allclose(transform.transform_directions((2, 0, 0)), (0, 2, 0), atol=1e-15)
    np.testing.assert_array_equal(transform.transform_directions((0, 0, 0)), (0, 0, 0))


def test_inverse_recovers_points_and_directions() -> None:
    quaternion = np.array([1.0, 2.0, 3.0, 4.0]) / math.sqrt(30.0)
    transform = RigidTransform.from_quaternion(quaternion, (3.75, -0.25, 0.66))
    values = np.random.default_rng(42).normal(size=(2, 4, 3))

    inverse = transform.inverse()

    np.testing.assert_allclose(
        inverse.transform_points(transform.transform_points(values)), values, atol=1e-14
    )
    np.testing.assert_allclose(
        inverse.transform_directions(transform.transform_directions(values)), values, atol=1e-14
    )
    np.testing.assert_allclose(inverse.inverse().rotation, transform.rotation, atol=1e-15)
    np.testing.assert_allclose(inverse.inverse().translation, transform.translation, atol=1e-15)
    np.testing.assert_allclose(transform.compose(inverse).rotation, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(transform.compose(inverse).translation, np.zeros(3), atol=1e-14)


def test_composition_applies_argument_then_receiver() -> None:
    source_to_middle = RigidTransform(np.eye(3), (2.0, 0.0, 0.0))
    middle_to_target = RigidTransform.from_quaternion(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), (0.0, 3.0, 0.0)
    )
    point = (1, 0, 0)

    combined = middle_to_target.compose(source_to_middle)

    np.testing.assert_allclose(combined.transform_points(point), (0, 6, 0), atol=1e-14)
    np.testing.assert_allclose(
        combined.transform_points(point),
        middle_to_target.transform_points(source_to_middle.transform_points(point)),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        source_to_middle.compose(middle_to_target).transform_points(point), (2, 4, 0), atol=1e-14
    )


def test_composition_preserves_order_of_noncommuting_rotations() -> None:
    source_to_middle = RigidTransform.from_quaternion(
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (1.0, 2.0, 3.0)
    )
    middle_to_target = RigidTransform.from_quaternion(
        (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)), (4.0, 5.0, 6.0)
    )
    points = np.eye(3)

    combined = middle_to_target.compose(source_to_middle)

    np.testing.assert_allclose(combined.transform_directions((0, 1, 0)), (1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(
        combined.transform_points(points),
        middle_to_target.transform_points(source_to_middle.transform_points(points)),
        atol=1e-14,
    )


@pytest.mark.parametrize("shape", [(3,), (1, 3), (2, 4, 3), (0, 3), (2, 0, 3)])
def test_transform_preserves_arbitrary_batch_dimensions(shape: tuple[int, ...]) -> None:
    transform = RigidTransform(np.eye(3), (1, 2, 3))
    values = np.ones(shape, dtype=np.float32)

    points = transform.transform_points(values)
    directions = transform.transform_directions(values)

    assert points.shape == shape
    assert directions.shape == shape
    np.testing.assert_array_equal(points, values + np.array([1, 2, 3]))
    np.testing.assert_array_equal(directions, values)


def test_transform_accepts_noncontiguous_input_without_mutating_it() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    original = values.copy()
    assert not values.flags.c_contiguous

    transformed = RigidTransform.identity().transform_points(values)
    transformed[0, 0] = -10.0

    np.testing.assert_array_equal(values, original)


def test_transform_storage_and_public_arrays_are_immutable_and_detached() -> None:
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    transform = RigidTransform(rotation, translation)
    rotation[0, 0] = 9.0
    translation[:] = 9.0

    np.testing.assert_array_equal(transform.rotation, np.eye(3))
    np.testing.assert_array_equal(transform.translation, (1, 2, 3))
    public_rotation = transform.rotation
    public_translation = transform.translation
    assert not np.shares_memory(public_rotation, transform.rotation)
    assert not np.shares_memory(public_translation, transform.translation)
    for array in (public_rotation, public_translation):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 9.0
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    assert transform.rotation.shape == (3, 3)
    assert transform.translation.shape == (3,)
    with pytest.raises(FrozenInstanceError):
        transform._rotation = np.eye(3)  # type: ignore[misc]


def test_quaternion_roundoff_is_normalized_but_arbitrary_scaling_is_rejected() -> None:
    quaternion = np.array([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])

    normalized = RigidTransform.from_quaternion(quaternion)
    near_unit = RigidTransform.from_quaternion(quaternion * 1.0005)
    equivalent_sign = RigidTransform.from_quaternion(-quaternion)

    np.testing.assert_allclose(near_unit.rotation, normalized.rotation, atol=1e-15)
    np.testing.assert_allclose(equivalent_sign.rotation, normalized.rotation, atol=1e-15)
    with pytest.raises(ValueError, match="unit norm"):
        RigidTransform.from_quaternion(quaternion * 1.002)


def test_float32_rotation_roundoff_is_accepted_without_repair() -> None:
    angle = math.pi / 7
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    transform = RigidTransform(rotation)

    np.testing.assert_array_equal(transform.rotation, rotation)


@pytest.mark.parametrize(
    ("rotation", "message"),
    [
        (np.eye(2), "shape"),
        (np.ones((3, 3, 1)), "shape"),
        (np.diag([1, 1, 2]), "orthonormal"),
        (np.diag([1, 1, -1]), "right-handed"),
        (np.full((3, 3), 1e308), "orthonormal"),
        (np.full((3, 3), math.nan), "finite"),
        (np.full((3, 3), math.inf), "finite"),
    ],
)
def test_constructor_rejects_invalid_rotations(rotation: npt.ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RigidTransform(rotation)


@pytest.mark.parametrize(
    ("translation", "message"),
    [
        ((1, 2), "shape"),
        ([[1, 2, 3]], "shape"),
        ((1, 2, math.nan), "finite"),
        ((1, math.inf, 3), "finite"),
        ((1, 2, 3j), "real numeric"),
        (("1", "2", "3"), "real numeric"),
    ],
)
def test_constructor_rejects_invalid_translations(translation: npt.ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RigidTransform(np.eye(3), translation)


@pytest.mark.parametrize(
    ("quaternion", "message"),
    [
        ((0, 0, 0), "shape"),
        ([[0, 0, 0, 1]], "shape"),
        ((0, 0, 0, 0), "unit norm"),
        ((0, 0, 0, 2), "unit norm"),
        ((1e308, 1e308, 1e308, 1e308), "unit norm"),
        ((0, 0, math.nan, 1), "finite"),
        ((0, 0, math.inf, 1), "finite"),
    ],
)
def test_constructor_rejects_invalid_quaternions(quaternion: npt.ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RigidTransform.from_quaternion(quaternion)


@pytest.mark.parametrize("method_name", ["transform_points", "transform_directions"])
@pytest.mark.parametrize(
    "values",
    [
        1.0,
        [1.0, 2.0],
        np.ones((3, 1)),
        [math.nan, 0.0, 0.0],
        [0.0, math.inf, 0.0],
        [1.0 + 1j, 0.0, 0.0],
        [True, False, True],
        ["1", "2", "3"],
        [[1, 2, 3], [4, 5]],
    ],
)
def test_vector_transforms_reject_malformed_or_nonfinite_input(
    method_name: str, values: npt.ArrayLike
) -> None:
    method = getattr(RigidTransform.identity(), method_name)

    with pytest.raises(ValueError):
        method(values)


def test_nonfinite_arithmetic_results_are_rejected() -> None:
    transform = RigidTransform(np.eye(3), (1e308, 0.0, 0.0))
    with pytest.raises(ValueError, match="transformed points exceed finite"):
        transform.transform_points((1e308, 0.0, 0.0))
    with pytest.raises(ValueError, match="translation must contain only finite"):
        transform.compose(transform)

    rotated = RigidTransform.from_quaternion(
        (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    )
    with pytest.raises(ValueError, match="transformed directions exceed finite"):
        rotated.transform_directions((1.7e308, 1.7e308, 0.0))

    inverse_overflow = RigidTransform(rotated.rotation, (1.7e308, 1.7e308, 0.0))
    with pytest.raises(ValueError, match="translation must contain only finite"):
        inverse_overflow.inverse()
