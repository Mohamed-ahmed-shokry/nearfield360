import struct
import zlib

import pytest

from nearfield360.data.image_headers import ImageHeader, ImageHeaderError, read_image_header


def _png_header(
    width: int = 7,
    height: int = 3,
    *,
    depth: int = 8,
    color: int = 2,
    compression: int = 0,
    filtering: int = 0,
    interlace: int = 0,
) -> bytes:
    chunk = b"IHDR" + struct.pack(
        ">IIBBBBB", width, height, depth, color, compression, filtering, interlace
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
    )


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg_frame(
    width: int = 7,
    height: int = 3,
    *,
    marker: int = 0xC0,
    precision: int = 8,
    components: int = 3,
) -> bytes:
    fields = struct.pack(">BHHB", precision, height, width, components)
    component_fields = b"".join(bytes((index + 1, 0x11, 0)) for index in range(components))
    return _segment(marker, fields + component_fields)


def _jpeg_scan(components: int = 3) -> bytes:
    fields = bytes((components,))
    component_fields = b"".join(bytes((index + 1, 0)) for index in range(components))
    return _segment(0xDA, fields + component_fields + b"\x00\x3f\x00")


@pytest.mark.parametrize(
    ("color", "depth"), [(0, 1), (0, 16), (2, 8), (2, 16), (3, 4), (4, 16), (6, 8)]
)
def test_png_dimensions_preserve_stored_pixels(color: int, depth: int) -> None:
    assert read_image_header(_png_header(color=color, depth=depth)) == ImageHeader(7, 3, "png")


@pytest.mark.parametrize("width,height", [(0, 3), (7, 0), (2**31, 3), (7, 2**31)])
def test_png_rejects_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(ImageHeaderError, match="dimensions"):
        read_image_header(_png_header(width, height))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": 1},
        {"color": 2, "depth": 4},
        {"color": 3, "depth": 16},
        {"compression": 1},
        {"filtering": 1},
        {"interlace": 2},
    ],
)
def test_png_rejects_invalid_encoding_fields(kwargs: dict[str, int]) -> None:
    with pytest.raises(ImageHeaderError):
        read_image_header(_png_header(**kwargs))


def test_png_accepts_adam7_header() -> None:
    assert read_image_header(_png_header(interlace=1)).width == 7


@pytest.mark.parametrize("length", [8, 16, 24, 32])
def test_png_rejects_truncated_ihdr(length: int) -> None:
    with pytest.raises(ImageHeaderError, match="truncated PNG"):
        read_image_header(_png_header()[:length])


def test_png_requires_first_chunk_to_be_ihdr() -> None:
    data = bytearray(_png_header())
    data[12:16] = b"IDAT"
    with pytest.raises(ImageHeaderError, match="must begin"):
        read_image_header(bytes(data))


def test_png_rejects_invalid_ihdr_checksum() -> None:
    data = bytearray(_png_header())
    data[32] ^= 0xFF
    with pytest.raises(ImageHeaderError, match="checksum"):
        read_image_header(bytes(data))


@pytest.mark.parametrize(
    ("marker", "precision"),
    [(0xC0, 8), (0xC1, 12), (0xC2, 8), (0xC3, 16), (0xC9, 8), (0xCA, 12), (0xCB, 2)],
)
def test_jpeg_reads_non_hierarchical_frames(marker: int, precision: int) -> None:
    data = b"\xff\xd8" + _jpeg_frame(marker=marker, precision=precision) + _jpeg_scan()
    assert read_image_header(data) == ImageHeader(7, 3, "jpeg")


def test_jpeg_skips_metadata_instead_of_reading_embedded_thumbnail_dimensions() -> None:
    thumbnail = b"\xff\xd8" + _jpeg_frame(100, 200) + _jpeg_scan()
    metadata = _segment(0xE1, thumbnail)
    data = b"\xff\xd8\xff\x01\xff" + metadata + _jpeg_frame() + _jpeg_scan()

    assert read_image_header(data) == ImageHeader(7, 3, "jpeg")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 0},
        {"components": 0},
        {"components": 5},
        {"marker": 0xC0, "precision": 12},
        {"marker": 0xC2, "precision": 16},
        {"marker": 0xC3, "precision": 1},
    ],
)
def test_jpeg_rejects_invalid_frame_fields(kwargs: dict[str, int]) -> None:
    data = b"\xff\xd8" + _jpeg_frame(**kwargs) + _jpeg_scan()
    with pytest.raises(ImageHeaderError):
        read_image_header(data)


@pytest.mark.parametrize(
    "suffix",
    [
        b"",
        b"not a marker",
        b"\xff",
        b"\xff\x00",
        b"\xff\xd0",
        b"\xff\xd8",
        b"\xff\xd9",
        b"\xff\xe0\x00",
        b"\xff\xe0\x00\x01",
        b"\xff\xe0\x00\x10\x00",
        _segment(0xC0, b"\x08\x00"),
        _segment(0xC0, b"\x08\x00\x03\x00\x07\x03"),
        _jpeg_scan(),
        _jpeg_frame(),
        _jpeg_frame() + _jpeg_frame() + _jpeg_scan(),
        _jpeg_frame() + _segment(0xDA, b""),
        _jpeg_frame() + _jpeg_scan(components=0),
        _jpeg_frame() + _segment(0xDA, b"\x01\x01\x00\x00\x3f\x00\x00"),
    ],
)
def test_jpeg_rejects_malformed_or_truncated_headers(suffix: bytes) -> None:
    with pytest.raises(ImageHeaderError):
        read_image_header(b"\xff\xd8" + suffix)


@pytest.mark.parametrize("marker", [0xC5, 0xCE, 0xDE, 0xDF])
def test_jpeg_rejects_hierarchical_images(marker: int) -> None:
    with pytest.raises(ImageHeaderError, match="hierarchical"):
        read_image_header(b"\xff\xd8" + _segment(marker, b""))


@pytest.mark.parametrize("data", [b"", b"\x89PNG", b"GIF89a", b"BM", b"\xff"])
def test_header_requires_supported_signature(data: bytes) -> None:
    with pytest.raises(ImageHeaderError, match="expected PNG or JPEG"):
        read_image_header(data)
