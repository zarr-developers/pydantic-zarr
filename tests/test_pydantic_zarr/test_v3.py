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
    TBaseAttr,
    TBaseItem,
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


def test_x():
    a = ArraySpec.from_array(np.arange(20), attributes={"level": 3, "type": "array"})
    assert a.model_dump()["chunk_grid"] == a.chunk_grid


def test_from_zarr_depth() -> None:
    codecs = ({"name": "bytes", "configuration": {}},)
    tree: dict[str, GroupSpec[TBaseAttr, TBaseItem] | ArraySpec[TBaseAttr]] = {
        "": GroupSpec(members=None, attributes={"level": 0, "type": "group"}),
        "/1": GroupSpec(members=None, attributes={"level": 1, "type": "group"}),
        "/1/2": GroupSpec(members=None, attributes={"level": 2, "type": "group"}),
        "/1/2/1": GroupSpec(members=None, attributes={"level": 3, "type": "group"}),
        "/1/2/2": ArraySpec.from_array(
            np.arange(20), attributes={"level": 3, "type": "array"}, codecs=codecs
        ),
    }
    store = zarr.storage.MemoryStore()
    group_out = GroupSpec.from_flat(tree).to_zarr(store, path="test")
    group_in_0 = GroupSpec.from_zarr(group_out, depth=0)  # type: ignore[var-annotated]
    assert group_in_0 == tree[""]

    group_in_1 = GroupSpec.from_zarr(group_out, depth=1)  # type: ignore[var-annotated]
    assert group_in_1.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_1.members is not None
    assert group_in_1.members["1"] == tree["/1"]

    group_in_2 = GroupSpec.from_zarr(group_out, depth=2)  # type: ignore[var-annotated]
    assert group_in_2.members is not None
    assert group_in_2.members["1"].members["2"] == tree["/1/2"]
    assert group_in_2.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_2.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]

    group_in_3 = GroupSpec.from_zarr(group_out, depth=3)  # type: ignore[var-annotated]
    assert group_in_3.members is not None
    assert group_in_3.members["1"].members["2"].members["1"] == tree["/1/2/1"]
    assert group_in_3.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_3.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]
    assert group_in_3.members["1"].members["2"].attributes == tree["/1/2"].attributes  # type: ignore[attr-defined]
