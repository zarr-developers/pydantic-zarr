from __future__ import annotations

from pydantic_zarr.strict.v3.chunk_grid import GRIDS
from pydantic_zarr.strict.v3.codec import CODECS


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
    assert set(CODECS) == expected
    assert CODECS["transpose"].kind == "array_array"
    assert CODECS["bytes"].kind == "array_bytes"
    assert (
        CODECS["cast_value"].dtype_out(
            {"name": "cast_value", "configuration": {"data_type": "float64"}}, "int32"
        )
        == "float64"
    )


def test_grid_maps() -> None:
    assert set(GRIDS) == {"regular", "rectilinear"}
