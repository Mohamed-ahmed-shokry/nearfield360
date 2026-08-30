import struct
import zlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from nearfield360.data import ImageLimits, ImageReadError, load_label_image, load_rgb_image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_load_rgb_image_converts_bgr_and_returns_contiguous_array(tmp_path: Path) -> None:
    path = tmp_path / "color.png"
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    _write_image(path, bgr)

    rgb = load_rgb_image(path)

    np.testing.assert_array_equal(rgb, np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8))
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_load_label_image_preserves_integer_values(tmp_path: Path, dtype: np.dtype) -> None:
    path = tmp_path / f"labels-{np.dtype(dtype).name}.png"
    expected = np.array([[0, 1], [8, 9]], dtype=dtype)
    _write_image(path, expected)

    actual = load_label_image(path)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_load_label_image_rejects_color_input(tmp_path: Path) -> None:
    path = tmp_path / "color-labels.png"
    _write_image(path, np.zeros((2, 3, 3), dtype=np.uint8))

    with pytest.raises(ImageReadError, match="single-channel"):
        load_label_image(path)


def test_image_loader_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ImageReadError, match="does not exist"):
        load_rgb_image(tmp_path / "missing.png")

    malformed = tmp_path / "malformed.png"
    malformed.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageReadError, match="could not decode"):
        load_rgb_image(malformed)


def test_image_loader_enforces_file_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized.png"
    path.write_bytes(b"x" * 17)
    decoder = Mock()
    monkeypatch.setattr(cv2, "imdecode", decoder)

    with pytest.raises(ImageReadError, match="byte safety limit"):
        load_rgb_image(path, limits=ImageLimits(max_file_bytes=16))
    decoder.assert_not_called()


def test_image_loader_caps_snapshot_read() -> None:
    path = MagicMock(spec=Path)
    path.is_file.return_value = True
    reader = Mock(wraps=BytesIO(b"x" * 100))
    path.open.return_value.__enter__.return_value = reader

    with pytest.raises(ImageReadError, match="byte safety limit"):
        load_rgb_image(path, limits=ImageLimits(max_file_bytes=16))

    reader.read.assert_called_once_with(17)


def test_image_loader_enforces_pixel_limit_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "image.png"
    _write_image(path, np.zeros((3, 4, 3), dtype=np.uint8))
    decoder = Mock()
    monkeypatch.setattr(cv2, "imdecode", decoder)

    with pytest.raises(ImageReadError, match="12 pixels"):
        load_rgb_image(path, limits=ImageLimits(max_pixels=11))
    decoder.assert_not_called()


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_huge_image_headers_never_reach_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    success, encoded = cv2.imencode(suffix, np.zeros((2, 3, 3), dtype=np.uint8))
    assert success
    data = bytearray(encoded.tobytes())
    if suffix == ".png":
        data[16:24] = struct.pack(">II", 100_000, 100_000)
        data[29:33] = zlib.crc32(data[12:29]).to_bytes(4, "big")
    else:
        frame_offset = data.index(b"\xff\xc0")
        data[frame_offset + 5 : frame_offset + 9] = struct.pack(">HH", 65_535, 65_535)
    path = tmp_path / f"huge{suffix}"
    path.write_bytes(data)
    decoder = Mock()
    monkeypatch.setattr(cv2, "imdecode", decoder)

    with pytest.raises(ImageReadError, match="pixel limit"):
        load_rgb_image(path)
    decoder.assert_not_called()


@pytest.mark.parametrize(
    "data", [b"", b"not an image", b"BM" + b"\x00" * 60, b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"]
)
def test_unsupported_or_truncated_headers_never_reach_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes
) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(data)
    decoder = Mock()
    monkeypatch.setattr(cv2, "imdecode", decoder)

    with pytest.raises(ImageReadError, match="could not decode"):
        load_rgb_image(path)
    decoder.assert_not_called()


@pytest.mark.parametrize("progressive", [False, True])
def test_load_rgb_image_supports_jpeg(tmp_path: Path, progressive: bool) -> None:
    path = tmp_path / "frame.jpg"
    source = np.full((4, 7, 3), (30, 60, 90), dtype=np.uint8)
    success, encoded = cv2.imencode(
        ".jpg", source, [cv2.IMWRITE_JPEG_PROGRESSIVE, int(progressive)]
    )
    assert success
    path.write_bytes(encoded.tobytes())

    image = load_rgb_image(path)

    assert image.shape == source.shape
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    np.testing.assert_allclose(image[0, 0], [90, 60, 30], atol=2)


def test_rgb_decoder_ignores_exif_orientation(tmp_path: Path) -> None:
    success, encoded = cv2.imencode(".jpg", np.zeros((3, 7, 3), dtype=np.uint8))
    assert success
    # One little-endian TIFF IFD entry: orientation = 6 (rotate 90 degrees).
    exif = (
        b"Exif\x00\x00II\x2a\x00\x08\x00\x00\x00"
        + struct.pack("<H", 1)
        + struct.pack("<HHI", 0x0112, 3, 1)
        + struct.pack("<H", 6)
        + b"\x00" * 6
    )
    app1 = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    data = encoded.tobytes()
    oriented = data[:2] + app1 + data[2:]
    path = tmp_path / "orientation.jpg"
    path.write_bytes(oriented)
    assert cv2.imdecode(np.frombuffer(oriented, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
        7,
        3,
        3,
    )

    assert load_rgb_image(path).shape == (3, 7, 3)


def test_decoding_uses_the_validated_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.png"
    source = np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8)
    _write_image(path, source)
    original_decode = cv2.imdecode

    def decode_snapshot(encoded: np.ndarray, flags: int) -> np.ndarray:
        path.write_bytes(b"changed after header validation")
        return original_decode(encoded, flags)

    monkeypatch.setattr(cv2, "imdecode", decode_snapshot)

    np.testing.assert_array_equal(load_rgb_image(path), source[..., ::-1])


@pytest.mark.parametrize("shape", [(3, 2, 3), (0, 3, 3)])
def test_decoded_shape_must_match_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    path = tmp_path / "shape.png"
    _write_image(path, np.zeros((2, 3, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "imdecode", Mock(return_value=np.zeros(shape, dtype=np.uint8)))

    with pytest.raises(ImageReadError, match="shape"):
        load_rgb_image(path)


@pytest.mark.parametrize("failed_result", [None, cv2.error("invalid compressed pixels")])
def test_decoder_failures_are_contextual_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_result: object
) -> None:
    path = tmp_path / "bad-body.png"
    _write_image(path, np.zeros((2, 3, 3), dtype=np.uint8))
    decoder = (
        Mock(side_effect=failed_result)
        if isinstance(failed_result, cv2.error)
        else Mock(return_value=failed_result)
    )
    monkeypatch.setattr(cv2, "imdecode", decoder)

    with pytest.raises(ImageReadError, match="OpenCV could not decode"):
        load_rgb_image(path)


def test_image_read_failures_are_contextual_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.png"
    path.touch()
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("not readable")))

    with pytest.raises(ImageReadError, match="Unable to read image file"):
        load_rgb_image(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_file_bytes": 0},
        {"max_pixels": 0},
        {"max_file_bytes": 3.5},
        {"max_pixels": float("nan")},
        {"max_pixels": True},
    ],
)
def test_image_limits_must_be_positive_integers(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ImageLimits(**kwargs)
