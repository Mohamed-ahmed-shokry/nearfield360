"""NearField360 automotive surround-view perception toolkit."""

from importlib.metadata import PackageNotFoundError, version
from typing import Final


def _distribution_version() -> str:
    try:
        return version("nearfield360")
    except PackageNotFoundError:  # pragma: no cover - only possible in an unpackaged source tree
        return "0+unknown"


__version__: Final = _distribution_version()

__all__ = ["__version__"]
