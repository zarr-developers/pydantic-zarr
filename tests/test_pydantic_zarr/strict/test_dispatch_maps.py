from __future__ import annotations

from pydantic_zarr.strict.v3.chunk_grid import GRID_NDIM_OF
from pydantic_zarr.strict.v3.codec import (
    CODEC_DTYPE_OUT,
    CODEC_KIND,
    CODEC_NDIM_OF,
)


def test_codec_maps_cover_all() -> None:
    expected = {
        "bytes",
        "crc32c",
        "gzip",
        "zstd",
        "blosc",
        "transpose",
        "sharding_indexed",
        "scale_offset",
        "cast_value",
    }
    assert set(CODEC_NDIM_OF) == expected
    assert set(CODEC_KIND) == expected
    assert CODEC_KIND["transpose"] == "array_array"
    assert CODEC_KIND["bytes"] == "array_bytes"
    assert (
        CODEC_DTYPE_OUT["cast_value"](
            {"name": "cast_value", "configuration": {"data_type": "float64"}}, "int32"
        )
        == "float64"
    )


def test_grid_maps() -> None:
    assert set(GRID_NDIM_OF) == {"regular", "rectilinear"}
