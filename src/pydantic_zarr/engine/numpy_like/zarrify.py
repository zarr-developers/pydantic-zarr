from collections.abc import Mapping
from typing import Literal, overload

from pydantic_zarr.base import (
    ArrayLike,
    ArrayV2Config,
    ArrayV3Config,
    NamedConfig,
    TShape,
    guess_chunks,
)


def get_attributes(array: ArrayLike[TShape]) -> Mapping[str, object]:
    if hasattr(array, "attrs"):
        return array.attrs  # type: ignore[no-any-return]
    elif hasattr(array, "attributes"):
        return array.attributes  # type: ignore[no-any-return]
    return {}


def get_chunks(array: ArrayLike[TShape]) -> tuple[int, ...]:
    if hasattr(array, "chunksize"):
        # this covers dask arrays
        return array.chunksize  # type: ignore[no-any-return]
    elif hasattr(array, "chunks"):
        # this covers h5py arrays
        return array.chunks  # type: ignore[no-any-return]
    # failing to find any chunks attributes, if the dtype permits it we guess chunks
    if hasattr(array.dtype, "itemsize"):
        return guess_chunks(array.shape, array.dtype.itemsize)
    else:
        return array.shape


def get_fill_value(array: ArrayLike[TShape]) -> object:
    if hasattr(array, "fill_value"):
        return array.fill_value
    return None


def get_order(array: ArrayLike[TShape]) -> Literal["C", "F"]:
    if hasattr(array, "order") and array.order in ("C", "F"):
        return array.order  # type: ignore[no-any-return]
    else:
        return "C"


def get_dtype_v2(array: ArrayLike[TShape]) -> str:
    # assumes a numpy dtype here
    # todo: work something out for structured dtypes
    return array.dtype.str  # type: ignore[no-any-return]


def get_dtype_v3(array: ArrayLike[TShape]) -> str:
    # assumes a numpy dtype here
    # todo: work something out for structured dtypes
    return str(array.dtype)  # type: ignore[no-any-return]


def zarrify_v2(array: ArrayLike[TShape]) -> ArrayV2Config:
    return {
        "shape": array.shape,
        "dtype": get_dtype_v2(array),
        "chunks": get_chunks(array),
        "attributes": get_attributes(array),
        "filters": None,
        "compressor": None,
        "fill_value": get_fill_value(array),
        "order": get_order(array),
        "dimension_separator": "/",
    }


def zarrify_v3(array: ArrayLike[TShape]) -> ArrayV3Config:
    fill_value = get_fill_value(array)
    attributes = get_attributes(array)
    codecs: tuple[NamedConfig, ...] = ({"name": "bytes"},)
    chunk_key_encoding: NamedConfig = {"name": "default", "configuration": {"separator": "/"}}
    chunk_grid: NamedConfig = {
        "name": "regular",
        "configuration": {"chunk_shape": get_chunks(array)},
    }
    dtype = get_dtype_v3(array)
    return {
        "shape": array.shape,
        "dtype": dtype,
        "fill_value": fill_value,
        "attributes": attributes,
        "codecs": codecs,
        "chunk_key_encoding": chunk_key_encoding,
        "chunk_grid": chunk_grid,
        "dimension_names": None,
    }


@overload
def zarrify(array: ArrayLike[TShape], zarr_format: Literal[2]) -> ArrayV2Config: ...


@overload
def zarrify(array: ArrayLike[TShape], zarr_format: Literal[3]) -> ArrayV3Config: ...


def zarrify(array: ArrayLike[TShape], zarr_format: Literal[2, 3]) -> ArrayV3Config | ArrayV2Config:
    """
    Generate a zarr array schema from an array-like object.
    """
    if zarr_format == 2:
        return zarrify_v2(array)
    if zarr_format == 3:
        return zarrify_v3(array)
    raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")
