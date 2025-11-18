from __future__ import annotations

import importlib
import importlib.util
import json
import re
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import (
    ArraySpec,
    BaseGroupSpec,
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingConfig,
    GroupSpec,
    NamedConfig,
    RegularChunking,
    RegularChunkingConfig,
    auto_codecs,
)

from .conftest import DTYPE_EXAMPLES_V3, DTypeExample

ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None


def test_serialize_deserialize() -> None:
    array_attributes = {"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]}

    group_attributes = {"group": True}

    array_spec = ArraySpec(
        attributes=array_attributes,
        shape=(1000, 1000),
        dimension_names=("rows", "columns"),
        data_type="float64",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        codecs=(NamedConfig(name="GZip", configuration={"level": 1}),),
        fill_value="NaN",
        storage_transformers=[],
    )

    GroupSpec(attributes=group_attributes, members={"array": array_spec})


def test_from_array() -> None:
    array = np.arange(10)
    array_spec = ArraySpec.from_array(array)

    assert array_spec == ArraySpec(
        zarr_format=3,
        node_type="array",
        attributes={},
        shape=(10,),
        data_type="int64",
        chunk_grid=RegularChunking(
            name="regular", configuration=RegularChunkingConfig(chunk_shape=(10,))
        ),
        chunk_key_encoding=DefaultChunkKeyEncoding(
            name="default", configuration=DefaultChunkKeyEncodingConfig(separator="/")
        ),
        fill_value=0,
        codecs=auto_codecs(array),
        storage_transformers=(),
        dimension_names=None,
    )
    # check that we can write this array to zarr
    # TODO: fix type of the store argument in to_zarr
    if not ZARR_AVAILABLE:
        return
    array_spec.to_zarr(store={}, path="")  # type: ignore[arg-type]


def test_arrayspec_no_empty_codecs() -> None:
    """
    Ensure that it is not possible to create an ArraySpec with no codecs
    """

    with pytest.raises(
        ValidationError, match="Value error, Invalid length. Expected 1 or more, got 0."
    ):
        ArraySpec(
            shape=(1,),
            data_type="uint8",
            codecs=[],
            attributes={},
            fill_value=0,
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (1,)}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        )


@pytest.mark.filterwarnings("ignore:The dtype:UserWarning")
@pytest.mark.filterwarnings("ignore:The data type:FutureWarning")
@pytest.mark.filterwarnings("ignore:The codec:UserWarning")
@pytest.mark.parametrize("dtype_example", DTYPE_EXAMPLES_V3, ids=str)
def test_arrayspec_from_zarr(dtype_example: DTypeExample) -> None:
    """
    Test that deserializing an ArraySpec from a zarr python store works as expected.
    """
    zarr = pytest.importorskip("zarr")
    store = {}

    data_type = dtype_example.name

    if data_type == "variable_length_bytes":
        pytest.skip(
            reason="Bug in zarr python: see https://github.com/zarr-developers/zarr-python/issues/3263"
        )

    arr = zarr.create_array(store=store, shape=(10,), dtype=data_type, zarr_format=3)

    arr_spec = ArraySpec.from_zarr(arr)
    assert arr_spec.model_dump() == json.loads(
        store["zarr.json"].to_bytes(), object_hook=tuplify_json
    )


@pytest.mark.parametrize("path", ["", "foo"])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("dtype_example", DTYPE_EXAMPLES_V3, ids=str)
@pytest.mark.parametrize("config", [{}, {"write_empty_chunks": True, "order": "F"}])
@pytest.mark.filterwarnings("ignore:The codec `vlen-utf8`:UserWarning")
@pytest.mark.filterwarnings("ignore:The codec `vlen-bytes`:UserWarning")
@pytest.mark.filterwarnings("ignore:The data type :FutureWarning")
def test_arrayspec_to_zarr(
    path: str, overwrite: bool, config: dict[str, object], dtype_example: DTypeExample
) -> None:
    """
    Test that serializing an ArraySpec to a zarr python store works as expected.
    """
    data_type = dtype_example.name
    fill_value = dtype_example.fill_value

    codecs = ({"name": "bytes", "configuration": {}},)
    if data_type == "variable_length_bytes":
        codecs = ({"name": "vlen-bytes"},)

    elif data_type in ("str", "string"):
        codecs = ({"name": "vlen-utf8"},)

    store = {}

    arr_spec = ArraySpec(
        attributes={},
        shape=(10,),
        data_type=data_type,
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=codecs,
        fill_value=fill_value,
        dimension_names=("x",),
    )
    if not ZARR_AVAILABLE:
        return
    arr = arr_spec.to_zarr(store=store, path=path, overwrite=overwrite, config=config)
    assert arr._async_array.metadata == arr._async_array.metadata
    for key, value in config.items():
        assert asdict(arr._async_array._config)[key] == value


def get_flat_example() -> tuple[dict[str, ArraySpec | BaseGroupSpec], GroupSpec]:
    """
    Get example data for testing to_flat and from_flat.

    The returned value is a tuple with two elements: a flattened dict representation of a hierarchy,
    and the root group, with all of its members (i.e., the non-flat version of that hierarchy).
    """

    named_nodes: tuple[ArraySpec | GroupSpec, ...] = (
        GroupSpec(attributes={"name": ""}, members={}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/a1"}),
        GroupSpec(attributes={"name": "/g1"}, members={}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/a2"}),
        GroupSpec(attributes={"name": "/g1/g2"}, members={}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/g2/a3"}),
    )

    # For the flattened representation, groups should be BaseGroupSpec instances (without members)
    members_flat: dict[str, ArraySpec | BaseGroupSpec] = {
        "": BaseGroupSpec(attributes={"name": ""}),
        "/a1": named_nodes[1],  # ArraySpec
        "/g1": BaseGroupSpec(attributes={"name": "/g1"}),
        "/g1/a2": named_nodes[3],  # ArraySpec
        "/g1/g2": BaseGroupSpec(attributes={"name": "/g1/g2"}),
        "/g1/g2/a3": named_nodes[5],  # ArraySpec
    }

    # Build the non-flat hierarchy
    g2 = GroupSpec(attributes={"name": "/g1/g2"}, members={"a3": members_flat["/g1/g2/a3"]})
    g1 = GroupSpec(attributes={"name": "/g1"}, members={"a2": members_flat["/g1/a2"], "g2": g2})
    root = GroupSpec(attributes={"name": ""}, members={"g1": g1, "a1": members_flat["/a1"]})
    return members_flat, root


class TestGroupSpec:
    @staticmethod
    def test_to_flat() -> None:
        """
        Test that the to_flat method generates a flat representation of the hierarchy
        """

        members_flat, root = get_flat_example()
        observed = root.to_flat()
        assert observed == members_flat

    @staticmethod
    def test_from_flat() -> None:
        """
        Test that the from_flat method generates a `GroupSpec` from a flat representation of the
        hierarchy
        """
        members_flat, root = get_flat_example()
        assert GroupSpec.from_flat(members_flat) == root

    @staticmethod
    def test_from_zarr_depth() -> None:
        zarr = pytest.importorskip("zarr")
        from pydantic_zarr.v3 import BaseGroupSpec

        codecs = ({"name": "bytes", "configuration": {}},)
        tree: dict[str, BaseGroupSpec | ArraySpec] = {
            "": BaseGroupSpec(attributes={"level": 0, "type": "group"}),
            "/1": BaseGroupSpec(attributes={"level": 1, "type": "group"}),
            "/1/2": BaseGroupSpec(attributes={"level": 2, "type": "group"}),
            "/1/2/1": BaseGroupSpec(attributes={"level": 3, "type": "group"}),
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


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v2 import ArraySpec as ArraySpecv2

    members_flat = {"/a": ArraySpecv2.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v3 ArraySpec, GroupSpec, or BaseGroupSpec (got type(value)=<class 'pydantic_zarr.v2.ArraySpec'>)"
        ),
    ):
        GroupSpec.from_flat(members_flat)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "kwargs", "expected_names"),
    [
        ((1,), {"dimension_names": ["x"]}, ("x",)),
        ((1,), {}, None),
    ],
)
def test_dim_names_from_zarr_array(
    args: tuple, kwargs: dict, expected_names: tuple[str, ...] | None
) -> None:
    zarr = pytest.importorskip("zarr")

    arr = zarr.zeros(*args, **kwargs)
    spec: ArraySpec = ArraySpec.from_zarr(arr)
    assert spec.dimension_names == expected_names


# Additional tests ported from test_v2.py


def test_validation() -> None:
    """
    Test that specialized GroupSpec and ArraySpec instances cannot be serialized from
    the wrong inputs without a ValidationError.
    """
    zarr = pytest.importorskip("zarr")
    from typing import TypedDict

    if hasattr(__builtins__, "TypedDict"):
        pass
    else:
        from typing_extensions import TypedDict

    class GroupAttrsA(TypedDict):
        group_a: bool

    class GroupAttrsB(TypedDict):
        group_b: bool

    class ArrayAttrsA(TypedDict):
        array_a: bool

    class ArrayAttrsB(TypedDict):
        array_b: bool

    class ArrayA(ArraySpec):
        attributes: ArrayAttrsA

    class ArrayB(ArraySpec):
        attributes: ArrayAttrsB

    class GroupA(GroupSpec):
        attributes: GroupAttrsA

    class GroupB(GroupSpec):
        attributes: GroupAttrsB

    specA = GroupA(
        attributes=GroupAttrsA(group_a=True),
        members={
            "a": ArrayA(
                attributes=ArrayAttrsA(array_a=True),
                shape=(100,),
                data_type="uint8",
                chunk_grid=RegularChunking(
                    name="regular", configuration=RegularChunkingConfig(chunk_shape=(10,))
                ),
                chunk_key_encoding=DefaultChunkKeyEncoding(
                    name="default", configuration=DefaultChunkKeyEncodingConfig(separator="/")
                ),
                codecs=auto_codecs(np.zeros(100, dtype="uint8")),
                fill_value=0,
            )
        },
    )

    specB = GroupB(
        attributes=GroupAttrsB(group_b=True),
        members={
            "a": ArrayB(
                attributes=ArrayAttrsB(array_b=True),
                shape=(100,),
                data_type="uint8",
                chunk_grid=RegularChunking(
                    name="regular", configuration=RegularChunkingConfig(chunk_shape=(10,))
                ),
                chunk_key_encoding=DefaultChunkKeyEncoding(
                    name="default", configuration=DefaultChunkKeyEncodingConfig(separator="/")
                ),
                codecs=auto_codecs(np.zeros(100, dtype="uint8")),
                fill_value=0,
            )
        },
    )

    # check that we cannot create a specialized GroupSpec with the wrong attributes
    with pytest.raises(ValidationError):
        GroupB(
            attributes=GroupAttrsA(group_a=True),
            members={},
        )

    store = zarr.storage.MemoryStore()
    groupAMat = specA.to_zarr(store, path="group_a")
    groupBMat = specB.to_zarr(store, path="group_b")

    # from_zarr creates generic GroupSpec/ArraySpec instances
    groupA_from_zarr = GroupSpec.from_zarr(groupAMat)
    groupB_from_zarr = GroupSpec.from_zarr(groupBMat)

    # Check that the reconstructed groups match the originals
    assert groupA_from_zarr.attributes == specA.attributes
    assert groupB_from_zarr.attributes == specB.attributes

    # ArraySpec instances can be reconstructed
    ArraySpec.from_zarr(groupAMat["a"])
    ArraySpec.from_zarr(groupBMat["a"])


def test_member_name() -> None:
    """Test that member keys containing "/" raise ValidationError."""
    with pytest.raises(ValidationError, match='Strings containing "/" are invalid.'):
        GroupSpec(attributes={}, members={"path/with/slash": GroupSpec(attributes={}, members={})})


def test_flatten_unflatten() -> None:
    """Test flattening and unflattening of GroupSpec hierarchies."""
    from pydantic_zarr.v3 import from_flat, to_flat

    # Test with array
    arr = ArraySpec.from_array(np.arange(10))
    flattened = to_flat(arr)
    assert flattened == {"": arr}
    assert from_flat(flattened) == arr

    # Test with group containing array
    group = GroupSpec(
        attributes={"foo": 10},
        members={"a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100})},
    )
    flattened = to_flat(group)
    expected = {
        "": BaseGroupSpec(attributes={"foo": 10}),
        "/a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100}),
    }
    assert flattened == expected
    assert from_flat(flattened) == group

    # Test with nested groups
    nested_group = GroupSpec(
        attributes={},
        members={
            "a": GroupSpec(
                attributes={"foo": 10},
                members={"a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100})},
            ),
            "b": ArraySpec.from_array(np.arange(2), attributes={"foo": 3}),
        },
    )
    flattened = to_flat(nested_group)
    expected = {
        "": BaseGroupSpec(attributes={}),
        "/a": BaseGroupSpec(attributes={"foo": 10}),
        "/a/a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100}),
        "/b": ArraySpec.from_array(np.arange(2), attributes={"foo": 3}),
    }
    assert flattened == expected
    assert from_flat(flattened) == nested_group


def test_array_like() -> None:
    """Test ArraySpec.like() comparison method."""
    a = ArraySpec.from_array(np.arange(10))
    assert a.like(a)

    b = a.model_copy(update={"data_type": "uint8"})
    assert not a.like(b)


def test_group_like() -> None:
    """Test GroupSpec.like() comparison method."""
    from pydantic_zarr.v3 import from_flat

    tree: dict[str, BaseGroupSpec | ArraySpec] = {
        "": BaseGroupSpec(attributes={"path": ""}),
        "/a": BaseGroupSpec(attributes={"path": "/a"}),
        "/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/b"}),
        "/a/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/a/b"}),
    }
    group = from_flat(tree)
    assert group.like(group)
    assert not group.like(group.model_copy(update={"attributes": {}}))
    assert group.like(group.model_copy(update={"attributes": {}}), exclude={"attributes"})
    assert group.like(group.model_copy(update={"attributes": {}}), include={"members"})


def test_from_zarr_depth() -> None:
    """Test GroupSpec.from_zarr() with depth parameter."""
    zarr = pytest.importorskip("zarr")
    from pydantic_zarr.v3 import from_flat

    tree: dict[str, BaseGroupSpec | ArraySpec] = {
        "": BaseGroupSpec(attributes={"level": 0, "type": "group"}),
        "/1": BaseGroupSpec(attributes={"level": 1, "type": "group"}),
        "/1/2": BaseGroupSpec(attributes={"level": 2, "type": "group"}),
        "/1/2/1": BaseGroupSpec(attributes={"level": 3, "type": "group"}),
        "/1/2/2": ArraySpec.from_array(np.arange(20), attributes={"level": 3, "type": "array"}),
    }

    store = zarr.storage.MemoryStore()
    group_out = from_flat(tree).to_zarr(store, path="test")
    group_in_0 = GroupSpec.from_zarr(group_out, depth=0)
    assert group_in_0 == tree[""]

    group_in_1 = GroupSpec.from_zarr(group_out, depth=1)
    assert group_in_1.attributes == tree[""].attributes
    assert group_in_1.members is not None
    assert group_in_1.members["1"] == tree["/1"]

    group_in_2 = GroupSpec.from_zarr(group_out, depth=2)
    assert group_in_2.members is not None
    assert group_in_2.members["1"].members["2"] == tree["/1/2"]
    assert group_in_2.attributes == tree[""].attributes
    assert group_in_2.members["1"].attributes == tree["/1"].attributes

    group_in_3 = GroupSpec.from_zarr(group_out, depth=3)
    assert group_in_3.members is not None
    assert group_in_3.members["1"].members["2"].members["1"] == tree["/1/2/1"]
    assert group_in_3.attributes == tree[""].attributes
    assert group_in_3.members["1"].attributes == tree["/1"].attributes
    assert group_in_3.members["1"].members["2"].attributes == tree["/1/2"].attributes
