from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearChunkGridObject

from pydantic_zarr.strict.v3.chunk_grid._spec import GridSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata import RectilinearChunkGridMetadata


def rectilinear(chunk_shapes: Sequence[Any]) -> RectilinearChunkGridMetadata:
    meta: RectilinearChunkGridMetadata = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": tuple(chunk_shapes)},
    }
    validate_rectilinear(meta)
    return meta


def _leaf_ints(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    out: list[int] = []
    for item in spec:
        if isinstance(item, int):
            out.append(item)
        else:
            out.extend(int(x) for x in item)
    return out


def validate_rectilinear(meta: RectilinearChunkGridMetadata) -> None:
    for dim_spec in meta["configuration"]["chunk_shapes"]:
        if any(v <= 0 for v in _leaf_ints(dim_spec)):
            raise ValueError(f"rectilinear chunk_shapes dim spec {dim_spec!r} must be all positive")


def ndim_of(meta: RectilinearChunkGridMetadata) -> int | None:
    return len(meta["configuration"]["chunk_shapes"])


SPEC = GridSpec(
    name="rectilinear",
    metadata_type=RectilinearChunkGridObject,
    validate=validate_rectilinear,
    ndim_of=ndim_of,
)
