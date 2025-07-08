import json
from dataclasses import asdict

import numpy as np
import pytest
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


def test_arrayspec_from_zarr() -> None:
    """
    Test that deserializing an ArraySpec from a zarr python store works as expected.
    """
    store = {}
    arr = zarr.create_array(store=store, shape=(10,), dtype="uint8")

    arr_spec = ArraySpec.from_zarr(arr)
    assert arr_spec.model_dump() == json.loads(
        store["zarr.json"].to_bytes(), object_hook=tuplify_json
    )


@pytest.mark.parametrize("path", ["", "foo"])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("config", [{}, {"write_empty_chunks": True, "order": "F"}])
def test_arrayspec_to_zarr(path: str, overwrite: bool, config: dict[str, object]) -> None:
    """
    Test that serializing an ArraySpec to a zarr python store works as expected.
    """
    store = {}
    arr_spec = ArraySpec(
        shape=(10,),
        data_type="uint8",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=({"name": "bytes", "configuration": {}},),
        fill_value=0,
        dimension_names=("x",),
    )
    arr = arr_spec.to_zarr(store=store, path=path, overwrite=overwrite, config=config)
    assert arr._async_array.metadata == arr._async_array.metadata
    for key, value in config.items():
        assert asdict(arr._async_array._config)[key] == value
