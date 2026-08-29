import json
import math
from pathlib import Path
from typing import Any

import pytest

from nearfield360.data import (
    MAX_CALIBRATION_BYTES,
    CalibrationError,
    CameraId,
    load_calibration,
)


def _valid_calibration() -> dict[str, Any]:
    return {
        "extrinsic": {
            "quaternion": [
                0.5941767906169857,
                -0.5878843193897473,
                0.3873184109007999,
                -0.3890121040340926,
            ],
            "translation": [3.7484, 0.0, 0.66017],
        },
        "intrinsic": {
            "aspect_ratio": 1.0,
            "cx_offset": 3.942,
            "cy_offset": -3.093,
            "height": 966.0,
            "k1": 339.749,
            "k2": -31.988,
            "k3": 48.275,
            "k4": -7.201,
            "model": "radial_poly",
            "poly_order": 4,
            "width": 1280.0,
        },
        "name": "FV",
    }


def _write_calibration(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_calibration_parses_official_schema(tmp_path: Path) -> None:
    path = tmp_path / "00001_FV.json"
    _write_calibration(path, _valid_calibration())

    calibration = load_calibration(path)

    assert calibration.name is CameraId.FRONT
    assert calibration.intrinsic.width == 1280
    assert calibration.intrinsic.height == 966
    assert calibration.intrinsic.coefficients == (339.749, -31.988, 48.275, -7.201)
    assert calibration.intrinsic.principal_point == pytest.approx((643.442, 479.407))
    assert calibration.extrinsic.translation == pytest.approx((3.7484, 0.0, 0.66017))


def test_load_calibration_rejects_missing_and_oversized_files(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="does not exist"):
        load_calibration(tmp_path / "missing.json")

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_CALIBRATION_BYTES + 1))
    with pytest.raises(CalibrationError, match="byte safety limit"):
        load_calibration(oversized)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("name", "SIDE"), "name"),
        (("intrinsic.model", "pinhole"), "intrinsic.model"),
        (("intrinsic.poly_order", 3), "intrinsic.poly_order"),
        (("intrinsic.width", 0), "intrinsic.width"),
        (("intrinsic.k1", math.nan), "non-finite JSON constant"),
        (("extrinsic.quaternion", [0.0, 0.0, 0.0, 0.0]), "unit norm"),
    ],
)
def test_load_calibration_rejects_invalid_fields(
    tmp_path: Path, mutation: tuple[str, Any], message: str
) -> None:
    payload = _valid_calibration()
    dotted_key, value = mutation
    if "." in dotted_key:
        section, key = dotted_key.split(".", maxsplit=1)
        payload[section][key] = value
    else:
        payload[dotted_key] = value
    path = tmp_path / "invalid.json"
    _write_calibration(path, payload)

    with pytest.raises(CalibrationError, match=message):
        load_calibration(path)


def test_load_calibration_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _valid_calibration()
    payload["undocumented"] = True
    path = tmp_path / "unknown.json"
    _write_calibration(path, payload)

    with pytest.raises(CalibrationError, match="Extra inputs are not permitted"):
        load_calibration(path)


def test_load_calibration_wraps_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text('{"name":', encoding="utf-8")

    with pytest.raises(CalibrationError, match="Invalid calibration file"):
        load_calibration(path)


def test_load_calibration_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"name":"FV","name":"RV"}', encoding="utf-8")

    with pytest.raises(CalibrationError, match="duplicate JSON key: name"):
        load_calibration(path)
