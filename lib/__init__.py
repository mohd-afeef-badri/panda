"""PANDA lib - Shared utilities for mesh, I/O, and boundary conditions."""

from .polygonal_mesh import PolygonalMesh, PolyhedralMesh
from . import io
from . import boundary_conditions

try:
    from . import med_io
except ModuleNotFoundError as error:
    if error.name != "medcoupling":
        raise
    med_io = None

from . import vtk_io

__all__ = ["PolygonalMesh", "PolyhedralMesh", "io", "boundary_conditions", "med_io", "vtk_io"]
