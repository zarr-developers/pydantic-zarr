from typing import Any, Literal

import numpy as np
import tensorstore as ts

import pydantic_zarr._v2 as _v2

from ._models import DataType, ZarrDriver


def parse_dtype_name(dtype_name: str) -> DataType:
    return np.dtype(dtype_name).name


async def create_array(
    model: _v2.ArrayMetadataSpec,
    *,
    driver: Literal["file", "memory"],
    open: bool = True,
    create: bool = False,
    delete_existing: bool = False,
    assume_metadata: bool = False,
    assume_cached_metadata: bool = False,
) -> ts.TensorStore:
    spec = ZarrDriver(
        driver="zarr",
        kvstore={"driver": driver},
        metadata=model,
        open=open,
        create=create,
        delete_existing=delete_existing,
        assume_metadata=assume_metadata,
        assume_cached_metadata=assume_cached_metadata,
    )
    return await ts.open(spec.model_dump(exclude_none=True))
