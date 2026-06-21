from __future__ import annotations

from typing import TYPE_CHECKING

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
