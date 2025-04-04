from __future__ import annotations

from typing import Any, Literal, cast, overload

import numpy as np
import zarr

from pydantic_zarr.base import ArrayV2Config, ArrayV3Config, NamedConfig


def get_chunks(value: zarr.Array) -> tuple[int, ...]:
    return value.metadata.to_dict()["chunks"]


def get_shape(value: zarr.Array) -> tuple[int, ...]:
    return value.metadata.to_dict()["shape"]


def get_dtype(value: zarr.Array) -> str:
    if value.metadata.zarr_format == 3:
        return value.metadata.to_dict()["data_type"]
    return value.metadata.to_dict()["dtype"]


def get_attrs(value: zarr.Array | zarr.Group) -> dict[str, Any]:
    return value.attrs.asdict()


def get_filters(value: zarr.Array) -> list[dict[str, Any]] | None:
    if value.metadata.zarr_format == 3:
        raise ValueError("Filters are not defined for zarr v3 arrays")
    return value.metadata.to_dict()["filters"]


def get_compressor(value: zarr.Array) -> list[dict[str, Any]]:
    if value.metadata.zarr_format == 3:
        raise ValueError("Compressor is not defined for zarr v3 arrays")
    return value.metadata.to_dict()["compressor"]


def get_order(value: zarr.Array) -> Literal["C", "F"]:
    if value.metadata.zarr_format == 3:
        return "C"
    else:
        return value.metadata.to_dict()["order"]


def get_dimension_separator(value: zarr.Array) -> Literal[".", "/"]:
    if value.metadata.zarr_format == 3:
        return value.metadata.chunk_key_encoding.separator
    return value.metadata.to_dict()["dimension_separator"]


def get_fill_value(value: zarr.Array) -> Any:
    return value.metadata.to_dict()["fill_value"]


def zarrify_v3(array: zarr.Array) -> ArrayV3Config:
    # TODO: make this less brittle
    dtype: str | NamedConfig
    meta_dict: ArrayV2Config | ArrayV3Config = array.metadata.to_dict()  # type: ignore[assignment]

    attrs = array.attrs.asdict()
    shape = array.shape
    if meta_dict["zarr_format"] == 2:
        meta_dict = cast(ArrayV2Config, meta_dict)
        dtype = str(np.dtype(meta_dict["dtype"]))  # type: ignore[arg-type]
        bbc: NamedConfig
        if meta_dict["dtype"].startswith(">"):
            bbc = {"name": "bytes", "configuration": {"endian": "big"}}
        elif meta_dict["dtype"].startswith("<"):
            bbc = {"name": "bytes", "configuration": {"endian": "little"}}
        else:
            bbc = {"name": "bytes"}

        codecs: tuple[NamedConfig, ...] = (bbc,)
        chunk_grid: NamedConfig = {
            "name": "regular",
            "configuration": {"chunk_shape": meta_dict["chunks"]},
        }
        chunk_key_encoding: NamedConfig = {
            "name": "v2",
            "configuration": {"separator": meta_dict["dimension_separator"]},
        }
        fill_value = meta_dict["fill_value"]
        dimension_names = None

    elif meta_dict["zarr_format"] == 3:
        meta_dict = cast(ArrayV3Config, meta_dict)
        dtype = meta_dict["data_type"]
        codecs = meta_dict["codecs"]
        chunk_grid = meta_dict["chunk_grid"]
        chunk_key_encoding = meta_dict["chunk_key_encoding"]
        fill_value = meta_dict["fill_value"]
        dimension_names = meta_dict["dimension_names"]

    return {
        "node_type": "array",
        "zarr_format": 3,
        "shape": shape,
        "data_type": dtype,
        "attributes": attrs,
        "codecs": codecs,
        "chunk_grid": chunk_grid,
        "chunk_key_encoding": chunk_key_encoding,
        "fill_value": fill_value,
        "dimension_names": dimension_names,
    }


def zarrify_v2(array: zarr.Array) -> ArrayV2Config:
    # TODO: make this less brittle
    shape = array.shape
    meta_dict: ArrayV2Config | ArrayV3Config
    if array.metadata.zarr_format == 2:
        meta_dict = cast(ArrayV2Config, array.metadata.to_dict())
        dtype = meta_dict["dtype"]
        chunks = meta_dict["chunks"]
        fill_value = meta_dict["fill_value"]
        order = meta_dict["order"]
        dimension_separator = meta_dict["dimension_separator"]
        compressor = meta_dict["compressor"]
        filters = meta_dict["filters"]
        order = meta_dict["order"]
        attributes = meta_dict["attributes"]
    elif array.metadata.zarr_format == 3:
        meta_dict = cast(ArrayV3Config, array.metadata.to_dict())
        dtype = np.dtype(meta_dict["data_type"]).str
        chunks = array.chunks
        fill_value = meta_dict["fill_value"]
        dimension_separator = meta_dict["chunk_key_encoding"]["separator"]
        compressor = None
        filters = None
        order = "C"
        attributes = meta_dict["attributes"]

    return {
        "zarr_format": 2,
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
        "attributes": attributes,
        "filters": filters,
        "compressor": compressor,
        "fill_value": fill_value,
        "order": order,
        "dimension_separator": dimension_separator,
    }


@overload
def zarrify(array: zarr.Array, zarr_format: Literal[2]) -> ArrayV2Config: ...


@overload
def zarrify(array: zarr.Array, zarr_format: Literal[3]) -> ArrayV3Config: ...


def zarrify(array: zarr.Array, zarr_format: Literal[2, 3]) -> ArrayV3Config | ArrayV2Config:
    """
    Generate a zarr array schema from an array-like object.
    """
    if zarr_format == 2:
        return zarrify_v2(array)
    if zarr_format == 3:
        return zarrify_v3(array)
    raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")
