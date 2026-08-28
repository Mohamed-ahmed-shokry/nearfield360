from importlib.metadata import version

from nearfield360 import __version__


def test_package_version_matches_distribution_metadata() -> None:
    assert __version__ == version("nearfield360")
