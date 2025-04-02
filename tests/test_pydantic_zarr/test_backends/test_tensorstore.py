import json

import numpy as np
import pytest
import tensorstore as ts

import pydantic_zarr.v2 as v2
from pydantic_zarr.backends._tensorstore import (
    SEPARATOR,
    V2_ARRAY_KEY,
    V2_GROUP_KEY,
    create_array_v2,
    create_group_v2,
    get_member_keys,
    read_array_v2,
    read_group_v2,
    read_members_v2,
)
from pydantic_zarr.backends._tensorstore.models import FileDriver, KVStore, MemoryDriver


@pytest.fixture
def kvstore(request, tmp_path):
    if request.param == "file":
        return FileDriver(path=str(tmp_path) + "/")
    elif request.param == "memory":
        return MemoryDriver(path="")
    raise ValueError(f"Invalid request: {request.param}")


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [None, {}, {"foo": "bar"}])
async def test_read_group_v2(kvstore: KVStore, attrs: dict[str, str] | None) -> None:
    store = await ts.KvStore.open(kvstore.model_dump(exclude_none=True))
    metadata = {"zarr_format": 2}
    _ = await store.write(".zgroup", json.dumps(metadata))
    if attrs is not None:
        _ = await store.write(".zattrs", json.dumps(attrs))
        attrs_expected = attrs
    else:
        attrs_expected = {}
    groupspec = await read_group_v2(store)
    assert groupspec.attributes == attrs_expected
    assert groupspec.zarr_format == 2


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [None, {}, {"foo": "bar"}])
async def test_read_array_v2(kvstore: KVStore, attrs: dict[str, str] | None) -> None:
    store = await ts.KvStore.open(kvstore.model_dump(exclude_none=True))
    metadata = v2.ArrayMetadataSpec(
        shape=(10,), chunks=(2,), dtype=">i2", order="C", zarr_format=2
    ).model_dump(exclude_none=True)
    _ = await store.write(".zarray", json.dumps(metadata))

    if attrs is not None:
        _ = await store.write(".zattrs", json.dumps(attrs))
        attrs_expected = attrs
    else:
        attrs_expected = {}
    array_spec = await read_array_v2(store)
    assert array_spec == v2.ArraySpec(**metadata, attributes=attrs_expected)


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
async def test_read_members_v2(kvstore: KVStore) -> None:
    """
    Test that reading the members of a group works as expected.
    """
    store = await ts.KvStore.open(kvstore.model_dump(exclude_none=True))
    g_metadata = {"zarr_format": 2}
    a_metadata = v2.ArrayMetadataSpec(
        shape=(10,), chunks=(2,), dtype=">i2", order="C", zarr_format=2
    ).model_dump(exclude_none=True)
    _ = store.write(".zgroup", json.dumps(g_metadata)).result()
    _ = store.write("array/.zarray", json.dumps(a_metadata)).result()
    print(store.path)
    result = await read_members_v2(store)
    assert result == {b"": v2.Group(**g_metadata), b"array": v2.ArraySpec(**a_metadata)}


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
async def test_create_array_v2(kvstore: KVStore, dtype: str, shape: tuple[int, ...]) -> None:
    """
    Test that creating an array with a given shape and dtype works as expected.
    """
    template = np.zeros(shape, dtype=dtype)
    model = v2.ArrayMetadataSpec.from_arraylike(template)
    ts = await create_array_v2(
        model, kvstore=kvstore, open=False, create=True, delete_existing=True
    )
    assert ts.shape == shape
    assert ts.dtype.numpy_dtype == np.dtype(dtype)
    assert ts.schema.chunk_layout.read_chunk.shape == model.chunks
    assert ts.schema.chunk_layout.write_chunk.shape == model.chunks


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [{}, {"foo": "bar"}])
async def test_create_group_v2(kvstore: KVStore, attrs: dict[str, str]):
    """
    Test that creating a group with attributes works as expected.
    """
    model = v2.Group(attributes=attrs)
    store = await ts.KvStore.open(kvstore.model_dump(exclude_none=True))
    _ = await create_group_v2(model, kvstore=store)
    assert await read_group_v2(store) == model


def test_get_member_keys():
    node_keys = (
        V2_GROUP_KEY,
        SEPARATOR.join([b"foo", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", b"wam", V2_ARRAY_KEY]),
        SEPARATOR.join([b"foo", b"baz", V2_GROUP_KEY]),
    )
    extra = (
        SEPARATOR.join([b"foo", b"bar", b"non_group", b"baz", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", b"wam", b"sub_array", V2_ARRAY_KEY]),
    )
    assert node_keys == get_member_keys(node_keys + extra)
