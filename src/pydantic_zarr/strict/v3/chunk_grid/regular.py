from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.v3.chunk_grid.regular import RegularChunkGridObject

from pydantic_zarr.strict.v3.chunk_grid._spec import GridSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata import RegularChunkGridMetadata


def regular(chunk_shape: Sequence[int]) -> RegularChunkGridMetadata:
    meta: RegularChunkGridMetadata = {
        "name": "regular",
        "configuration": {"chunk_shape": tuple(chunk_shape)},
    }
    validate_regular(meta)
    return meta


def validate_regular(meta: RegularChunkGridMetadata) -> None:
    cs = meta["configuration"]["chunk_shape"]
    if any(d <= 0 for d in cs):
        raise ValueError(f"regular chunk_shape {tuple(cs)} must be all positive")


def ndim_of(meta: RegularChunkGridMetadata) -> int | None:
    return len(meta["configuration"]["chunk_shape"])


SPEC = GridSpec(
    name="regular",
    metadata_type=RegularChunkGridObject,
    validate=validate_regular,
    ndim_of=ndim_of,
)
