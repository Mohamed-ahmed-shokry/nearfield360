"""Explicit fisheye projection and vehicle-frame geometry."""

from nearfield360.geometry.bev import BevGrid, GridIndexResult
from nearfield360.geometry.camera import CalibratedCamera, PixelRayResult
from nearfield360.geometry.fisheye import (
    ProjectionResult,
    RadialPolynomialFisheye,
    UnprojectionResult,
)
from nearfield360.geometry.ground import GroundIntersection, intersect_ground
from nearfield360.geometry.rig import RigGroundFootprints, SurroundRig
from nearfield360.geometry.transforms import RigidTransform

__all__ = [
    "BevGrid",
    "CalibratedCamera",
    "GridIndexResult",
    "GroundIntersection",
    "PixelRayResult",
    "ProjectionResult",
    "RadialPolynomialFisheye",
    "RigGroundFootprints",
    "RigidTransform",
    "SurroundRig",
    "UnprojectionResult",
    "intersect_ground",
]
