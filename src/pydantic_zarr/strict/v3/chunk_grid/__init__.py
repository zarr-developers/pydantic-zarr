from __future__ import annotations

from zarr_metadata import RectilinearChunkGridMetadata, RegularChunkGridMetadata

from pydantic_zarr.strict.v3._registry import element_name
from pydantic_zarr.strict.v3.chunk_grid._spec import GridSpec

from . import rectilinear, regular

# Re-export builders
regular_grid = regular.regular
rectilinear_grid = rectilinear.rectilinear

# Grid type aliases
_RegularChunkGrid = RegularChunkGridMetadata
_RectilinearChunkGrid = RectilinearChunkGridMetadata

_MODULES = (regular, rectilinear)
GRIDS: dict[str, GridSpec] = {m.SPEC.name: m.SPEC for m in _MODULES}


def grid_spec_for(g: object) -> GridSpec | None:
    name = element_name(g)
    return GRIDS.get(name) if name is not None else None


__all__ = [
    "GRIDS",
    "GridSpec",
    "_RectilinearChunkGrid",
    "_RegularChunkGrid",
    "grid_spec_for",
    "rectilinear_grid",
    "regular_grid",
]
