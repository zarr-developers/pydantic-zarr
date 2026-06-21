from __future__ import annotations

from zarr_metadata import RectilinearChunkGridMetadata, RegularChunkGridMetadata

from . import rectilinear, regular

# Re-export builders
regular_grid = regular.regular
rectilinear_grid = rectilinear.rectilinear

# Grid type aliases
_RegularChunkGrid = RegularChunkGridMetadata
_RectilinearChunkGrid = RectilinearChunkGridMetadata

# Module dispatch maps
_MODULES = {
    "regular": regular,
    "rectilinear": rectilinear,
}

GRID_NDIM_OF = {n: m.ndim_of for n, m in _MODULES.items()}
GRID_VALIDATE = {n: getattr(m, f"validate_{n}") for n, m in _MODULES.items()}

__all__ = [
    "GRID_NDIM_OF",
    "GRID_VALIDATE",
    "_RectilinearChunkGrid",
    "_RegularChunkGrid",
    "rectilinear_grid",
    "regular_grid",
]
