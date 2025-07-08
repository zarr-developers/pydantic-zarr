import json

import numpy as np
import zarr

from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import (
    ArraySpec,
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingConfig,
    GroupSpec,
    NamedConfig,
    RegularChunking,
    RegularChunkingConfig,
)


def test_serialize_deserialize() -> None:
    array_attributes = {"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]}

    group_attributes = {"group": True}

    array_spec = ArraySpec(
        attributes=array_attributes,
        shape=[1000, 1000],
        dimension_names=["rows", "columns"],
        data_type="float64",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        codecs=[NamedConfig(name="GZip", configuration={"level": 1})],
        fill_value="NaN",
        storage_transformers=[],
    )

    GroupSpec(attributes=group_attributes, members={"array": array_spec})


def test_from_array() -> None:
    array_spec = ArraySpec.from_array(np.arange(10))
    assert array_spec == ArraySpec(
        zarr_format=3,
        node_type="array",
        attributes={},
        shape=(10,),
        data_type="int64",
        chunk_grid=RegularChunking(
            name="regular", configuration=RegularChunkingConfig(chunk_shape=[10])
        ),
        chunk_key_encoding=DefaultChunkKeyEncoding(
            name="default", configuration=DefaultChunkKeyEncodingConfig(separator="/")
        ),
        fill_value=0,
        codecs=[],
        storage_transformers=[],
        dimension_names=[None],
    )


def test_from_zarr() -> None:
    store_a = {}
    arr_a = zarr.create_array(store=store_a, shape=(10,), dtype="uint8")

    array_spec = ArraySpec.from_zarr(arr_a)
    assert array_spec.model_dump() == json.loads(
        store_a["zarr.json"].to_bytes(), object_hook=tuplify_json
    )
    store_b = {}
    arr_b = array_spec.to_zarr(store=store_b, path="")
    assert arr_b._async_array.metadata == arr_a._async_array.metadata
