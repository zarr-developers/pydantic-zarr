from __future__ import annotations

from typing import Literal, overload

import tensorstore as ts

from pydantic_zarr.base import ArrayV2Config, ArrayV3Config, NamedConfig


def zarrify_v3(array: ts.TensorStore) -> ArrayV3Config:
    chunk_grid: NamedConfig = {
        "name": "regular",
        "configuration": {"chunk_shape": array.schema.chunk_layout.read_chunk.shape},
    }
    chunk_key_encoding: NamedConfig = {"name": "default", "configuration": {"separator": "/"}}
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": array.shape,
        "data_type": array.dtype.to_json(),
        "attributes": {},
        "codecs": (),
        "chunk_grid": chunk_grid,
        "chunk_key_encoding": chunk_key_encoding,
        "fill_value": None,
        "dimension_names": None,
    }


def zarrify_v2(array: ts.TensorStore) -> ArrayV2Config:
    return {
        "zarr_format": 2,
        "shape": array.shape,
        "dtype": array.dtype.to_json(),
        "chunks": array.schema.chunk_layout.read_chunk.shape,
        "attributes": {},
        "filters": None,
        "compressor": None,
        "fill_value": None,
        "order": "C",
        "dimension_separator": "/",
    }


@overload
def zarrify(array: ts.TensorStore, zarr_format: Literal[2]) -> ArrayV2Config: ...


@overload
def zarrify(array: ts.TensorStore, zarr_format: Literal[3]) -> ArrayV3Config: ...


def zarrify(array: ts.TensorStore, zarr_format: Literal[2, 3]) -> ArrayV3Config | ArrayV2Config:
    """
    Generate a zarr array schema from an array-like object.
    """
    if zarr_format == 2:
        return zarrify_v2(array)
    if zarr_format == 3:
        return zarrify_v3(array)
    raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")
