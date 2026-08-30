"""Inspect PNG/JPEG dimensions without allocating decoded pixel buffers.

This is a narrow preflight, not a replacement for the image decoder. PNG IHDR
and non-hierarchical JPEG frame headers provide dimensions before decompression;
JPEG files that defer their height to a DNL marker are deliberately unsupported.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Literal

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_DEPTHS = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}
_JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA, 0xCB})
_JPEG_LOSSLESS_MARKERS = frozenset({0xC3, 0xCB})
_JPEG_UNSUPPORTED_MARKERS = frozenset({0xC5, 0xC6, 0xC7, 0xCD, 0xCE, 0xCF, 0xDE, 0xDF})


class ImageHeaderError(ValueError):
    """Raised when a supported, bounded image header cannot be established."""


@dataclass(frozen=True)
class ImageHeader:
    """Stored dimensions, before any EXIF orientation transform."""

    width: int
    height: int
    format: Literal["png", "jpeg"]


def _png_header(data: bytes) -> ImageHeader:
    if len(data) < 33:
        raise ImageHeaderError("truncated PNG IHDR chunk")
    if data[8:16] != b"\x00\x00\x00\x0dIHDR":
        raise ImageHeaderError("PNG must begin with a 13-byte IHDR chunk")
    if zlib.crc32(data[12:29]) != int.from_bytes(data[29:33], "big"):
        raise ImageHeaderError("invalid PNG IHDR checksum")
    width, height, depth, color, compression, filtering, interlace = struct.unpack_from(
        ">IIBBBBB", data, 16
    )
    if not 0 < width <= 2**31 - 1 or not 0 < height <= 2**31 - 1:
        raise ImageHeaderError("invalid PNG dimensions")
    if depth not in _PNG_DEPTHS.get(color, frozenset()):
        raise ImageHeaderError("invalid PNG bit depth or color type")
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ImageHeaderError("unsupported PNG compression, filter, or interlace method")
    return ImageHeader(width=width, height=height, format="png")


def _jpeg_frame(data: bytes, offset: int, length: int, marker: int) -> ImageHeader:
    if length < 8:
        raise ImageHeaderError("truncated JPEG frame header")
    precision = data[offset + 2]
    height = int.from_bytes(data[offset + 3 : offset + 5], "big")
    width = int.from_bytes(data[offset + 5 : offset + 7], "big")
    components = data[offset + 7]
    if not 1 <= components <= 4 or length != 8 + 3 * components:
        raise ImageHeaderError("invalid or unsupported JPEG frame components")
    if width == 0 or height == 0:
        raise ImageHeaderError("JPEG dimensions must be positive and known before decoding")
    if marker == 0xC0:
        valid_precision = precision == 8
    elif marker in _JPEG_LOSSLESS_MARKERS:
        valid_precision = 2 <= precision <= 16
    else:
        valid_precision = precision in (8, 12)
    if not valid_precision:
        raise ImageHeaderError("invalid JPEG sample precision")
    return ImageHeader(width=width, height=height, format="jpeg")


def _jpeg_header(data: bytes) -> ImageHeader:
    offset = 2
    frame: ImageHeader | None = None
    while offset < len(data):
        if data[offset] != 0xFF:
            raise ImageHeaderError("expected a JPEG marker before the first scan")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ImageHeaderError("truncated JPEG marker")
        marker = data[offset]
        offset += 1
        if marker == 0x01:  # TEM is the only standalone marker allowed here.
            continue
        if marker in (0x00, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            raise ImageHeaderError("unexpected JPEG marker before the first scan")
        if marker in _JPEG_UNSUPPORTED_MARKERS:
            raise ImageHeaderError("hierarchical JPEG images are unsupported")
        if offset + 2 > len(data):
            raise ImageHeaderError("truncated JPEG segment length")
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            raise ImageHeaderError("invalid or truncated JPEG segment")
        if marker in _JPEG_FRAME_MARKERS:
            if frame is not None:
                raise ImageHeaderError("multiple JPEG frame headers are unsupported")
            frame = _jpeg_frame(data, offset, length, marker)
        elif marker == 0xDA:
            if frame is None:
                raise ImageHeaderError("JPEG scan appears before a supported frame header")
            if length < 6:
                raise ImageHeaderError("truncated JPEG scan header")
            components = data[offset + 2]
            if not 1 <= components <= 4 or length != 6 + 2 * components:
                raise ImageHeaderError("invalid JPEG scan components")
            return frame
        offset += length
    raise ImageHeaderError("JPEG is missing its first scan header")


def read_image_header(data: bytes) -> ImageHeader:
    """Read PNG or non-hierarchical JPEG dimensions from an already bounded snapshot."""
    if data.startswith(_PNG_SIGNATURE):
        return _png_header(data)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_header(data)
    raise ImageHeaderError("unsupported or truncated image signature; expected PNG or JPEG")


__all__ = ["ImageHeader", "ImageHeaderError", "read_image_header"]
