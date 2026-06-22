from __future__ import annotations

from typing import Any

from pydantic_zarr.strict.v3._registry import element_name
from pydantic_zarr.strict.v3.chunk_grid import grid_spec_for
from pydantic_zarr.strict.v3.codec import codec_spec_for


def check_array_consistency(
    *, shape: tuple[int, ...], chunk_grid: Any, codecs: tuple[Any, ...], dimension_names: Any
) -> list[str]:
    ndim = len(shape)
    errs: list[str] = []

    gspec = grid_spec_for(chunk_grid)
    if gspec is not None and isinstance(chunk_grid, dict):
        gnd = gspec.ndim_of(chunk_grid)
        if gnd is not None and gnd != ndim:
            errs.append(f"chunk_grid ndim {gnd} != array ndim {ndim}")

    for c in codecs or ():
        cspec = codec_spec_for(c)
        if cspec is not None and isinstance(c, dict):
            cnd = cspec.ndim_of(c)
            if cnd is not None and cnd != ndim:
                errs.append(f"codec {cspec.name} ndim {cnd} != array ndim {ndim}")

    if dimension_names is not None and len(dimension_names) != ndim:
        errs.append(f"dimension_names ndim {len(dimension_names)} != array ndim {ndim}")

    # sharding inner-divides-outer (only meaningful when the outer grid is regular + object form)
    if element_name(chunk_grid) == "regular" and isinstance(chunk_grid, dict):
        outer = tuple(chunk_grid["configuration"]["chunk_shape"])
        for c in codecs or ():
            if element_name(c) == "sharding_indexed" and isinstance(c, dict):
                inner = tuple(c["configuration"]["chunk_shape"])
                if len(inner) != len(outer):
                    errs.append(f"sharding inner ndim {len(inner)} != outer ndim {len(outer)}")
                elif any(o % i != 0 for o, i in zip(outer, inner, strict=False) if i):
                    errs.append(f"sharding inner {inner} does not evenly divide outer {outer}")
    return errs
