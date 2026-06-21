from __future__ import annotations

from typing import Any

from pydantic_zarr.strict.v3.chunk_grid import GRID_NDIM_OF
from pydantic_zarr.strict.v3.codec import CODEC_NDIM_OF


def _name(x: Any) -> str | None:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name")
    return None


def check_array_consistency(
    *, shape: tuple[int, ...], chunk_grid: Any, codecs: tuple[Any, ...], dimension_names: Any
) -> list[str]:
    ndim = len(shape)
    errs: list[str] = []

    gname = _name(chunk_grid)
    if gname in GRID_NDIM_OF:
        gnd = GRID_NDIM_OF[gname](chunk_grid)
        if gnd is not None and gnd != ndim:
            errs.append(f"chunk_grid ndim {gnd} != array ndim {ndim}")

    for c in codecs or ():
        cname = _name(c)
        if cname in CODEC_NDIM_OF and isinstance(c, dict):
            cnd = CODEC_NDIM_OF[cname](c)
            if cnd is not None and cnd != ndim:
                errs.append(f"codec {cname} ndim {cnd} != array ndim {ndim}")

    if dimension_names is not None and len(dimension_names) != ndim:
        errs.append(f"dimension_names ndim {len(dimension_names)} != array ndim {ndim}")

    # sharding inner-divides-outer
    if gname == "regular":
        outer = tuple(chunk_grid["configuration"]["chunk_shape"])
        for c in codecs or ():
            if _name(c) == "sharding_indexed" and isinstance(c, dict):
                inner = tuple(c["configuration"]["chunk_shape"])
                if len(inner) != len(outer):
                    errs.append(f"sharding inner ndim {len(inner)} != outer ndim {len(outer)}")
                elif any(o % i != 0 for o, i in zip(outer, inner, strict=False) if i):
                    errs.append(f"sharding inner {inner} does not evenly divide outer {outer}")
    return errs
