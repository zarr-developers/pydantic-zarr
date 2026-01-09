from __future__ import annotations

import json
import re
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_zarr.experimental.core import json_eq
from pydantic_zarr.experimental.v3 import (
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

from ..conftest import DTYPE_EXAMPLES_V3, ZARR_AVAILABLE, DTypeExample


@pytest.fixture
def groupspec(request: pytest.FixtureRequest) -> GroupSpec:
    """
    Fixture that returns a basic GroupSpec with default attributes and no members.
    """
    meta_request = getattr(request, "param", {})
    attributes = meta_request.get("attributes", {})
    members = meta_request.get("members", {})
    return GroupSpec(attributes=attributes, members=members)


@pytest.fixture
def arrayspec(request: pytest.FixtureRequest) -> ArraySpec:
    """
    Fixture that returns an ArraySpec. This fixture is parametrized by a dict with keys
    matching the fields of the ArraySpec. Any missing fields are filled with default values.
    """
    meta_request = getattr(request, "param", {})
    shape = meta_request.get("shape", (1,))
    data_type = meta_request.get("data_type", "uint8")
    chunk_grid = meta_request.get(
        "chunk_grid", {"name": "regular", "configuration": {"chunk_shape": shape}}
    )
    chunk_key_encoding = meta_request.get(
        "chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}}
    )
    fill_value = meta_request.get("fill_value", 0)
    codecs = meta_request.get("codecs", ({"name": "bytes"},))
    attributes = meta_request.get("attributes", {})
    return ArraySpec(
        shape=shape,
        data_type=data_type,  # type: ignore[arg-type]
        chunk_grid=chunk_grid,  # type: ignore[arg-type]
        chunk_key_encoding=chunk_key_encoding,  # type: ignore[arg-type]
        fill_value=fill_value,
        codecs=codecs,
        attributes=attributes,
    )


@pytest.mark.parametrize("invalid_dimension_names", [[], "hi", ["1", 2, None]], ids=str)
def test_dimension_names_validation(arrayspec: ArraySpec, invalid_dimension_names: object) -> None:
    """
    Test that the `dimension_names` attribute is rejected if any of the following are true:
    - it is a sequence with length different from the number of dimensions of the array
    - it is a sequence containing values other than strings or `None`.
    - it is neither a valid sequence nor the value `None`.
    """
    with pytest.raises(ValidationError):
        ArraySpec(**(arrayspec.model_dump() | {"dimension_names": invalid_dimension_names}))


def test_serialize_deserialize() -> None:
    array_attributes = {"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]}

    group_attributes = {"group": True}

    array_spec = ArraySpec(
        attributes=array_attributes,
        shape=(1000, 1000),
        dimension_names=("rows", "columns"),
        data_type="float64",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": (1000, 100)}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        codecs=(NamedConfig(name="GZip", configuration={"level": 1}),),
        fill_value="NaN",
        storage_transformers=(),
    )

    GroupSpec(attributes=group_attributes, members={"array": array_spec})


def test_from_array() -> None:
    array = np.arange(10)
    array_spec = ArraySpec.from_array(array)

    assert array_spec == ArraySpec(
        zarr_format=3,
        attributes={},
        shape=array.shape,
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
    arr_out = array_spec.to_zarr(store={}, path="")  # type: ignore[arg-type]
    arr_out[:] = array
    assert np.array_equal(arr_out[:], array)


def test_arrayspec_no_empty_codecs(arrayspec: ArraySpec) -> None:
    """
    Ensure that it is not possible to create an ArraySpec with no codecs
    """

    with pytest.raises(
        ValidationError, match="Value error, Invalid length. Expected 1 or more, got 0."
    ):
        ArraySpec(**(arrayspec.model_dump() | {"codecs": ()}))  # type: ignore[arg-type]


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
    assert json_eq(arr_spec.model_dump(), json.loads(store["zarr.json"].to_bytes()))


@pytest.mark.parametrize("path", ["", "foo"])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("dtype_example", DTYPE_EXAMPLES_V3, ids=str)
@pytest.mark.parametrize("config", [{}, {"write_empty_chunks": True, "order": "F"}])
@pytest.mark.filterwarnings("ignore:The codec `vlen-utf8`:UserWarning")
@pytest.mark.filterwarnings("ignore:The codec `vlen-bytes`:UserWarning")
@pytest.mark.filterwarnings("ignore:The data type :FutureWarning")
def test_arrayspec_to_zarr(
    arrayspec: ArraySpec,
    path: str,
    overwrite: bool,
    config: dict[str, object],
    dtype_example: DTypeExample,
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

    arr_spec = arrayspec.model_copy(
        update={"data_type": data_type, "fill_value": fill_value, "codecs": codecs}
    )

    if not ZARR_AVAILABLE:
        return

    arr = arr_spec.to_zarr(store=store, path=path, overwrite=overwrite, config=config)
    assert arr._async_array.metadata == arr._async_array.metadata
    for key, value in config.items():
        assert asdict(arr._async_array._config)[key] == value


@pytest.fixture
def flat_example(arrayspec: ArraySpec) -> tuple[dict[str, ArraySpec | GroupSpec], GroupSpec]:
    """
    Get example data for testing to_flat and from_flat.

    The returned value is a tuple with two elements: a flattened dict representation of a hierarchy,
    and the root group, with all of its members (i.e., the non-flat version of that hierarchy).
    """
    named_nodes: tuple[ArraySpec | BaseGroupSpec, ...] = (
        BaseGroupSpec(attributes={"name": ""}),
        arrayspec.with_attributes({"name": "/a1"}),
        BaseGroupSpec(attributes={"name": "/g1"}),
        arrayspec.with_attributes({"name": "/g1/a2"}),
        BaseGroupSpec(attributes={"name": "/g1/g2"}),
        arrayspec.with_attributes({"name": "/g1/g2/a3"}),
    )

    members_flat: dict[str, ArraySpec | BaseGroupSpec] = {
        a.attributes["name"]: a for a in named_nodes
    }
    g2 = GroupSpec(
        attributes=members_flat["/g1/g2"].attributes, members={"a3": members_flat["/g1/g2/a3"]}
    )
    g1 = GroupSpec(
        attributes=members_flat["/g1"].attributes, members={"a2": members_flat["/g1/a2"], "g2": g2}
    )
    root = GroupSpec(
        attributes=members_flat[""].attributes, members={"g1": g1, "a1": members_flat["/a1"]}
    )
    return members_flat, root


class TestGroupSpec:
    @staticmethod
    def test_to_flat(flat_example: tuple[dict[str, ArraySpec | BaseGroupSpec], GroupSpec]) -> None:
        """
        Test that the to_flat method generates a flat representation of the hierarchy
        """

        members_flat, root = flat_example
        observed = root.to_flat()
        assert observed == members_flat

    @staticmethod
    def test_from_flat(
        flat_example: tuple[dict[str, ArraySpec | BaseGroupSpec], GroupSpec],
    ) -> None:
        """
        Test that the from_flat method generates a `GroupSpec` from a flat representation of the
        hierarchy
        """
        members_flat, root = flat_example
        assert GroupSpec.from_flat(members_flat).attributes == root.attributes

    @staticmethod
    def test_from_zarr_depth(arrayspec) -> None:
        zarr = pytest.importorskip("zarr")
        tree: dict[str, BaseGroupSpec | ArraySpec] = {
            "": BaseGroupSpec(attributes={"level": 0, "type": "group"}),
            "/1": BaseGroupSpec(attributes={"level": 1, "type": "group"}),
            "/1/2": BaseGroupSpec(attributes={"level": 2, "type": "group"}),
            "/1/2/1": BaseGroupSpec(attributes={"level": 3, "type": "group"}),
            "/1/2/2": arrayspec.with_attributes({"level": 3, "type": "array"}),
        }
        store = zarr.storage.MemoryStore()
        group_out = GroupSpec.from_flat(tree).to_zarr(store, path="test")
        group_in_0 = GroupSpec.from_zarr(group_out, depth=0)  # type: ignore[var-annotated]
        assert group_in_0.attributes == tree[""].attributes

        group_in_1 = GroupSpec.from_zarr(group_out, depth=1)  # type: ignore[var-annotated]
        assert group_in_1.attributes == tree[""].attributes  # type: ignore[attr-defined]
        assert group_in_1.members is not None
        assert group_in_1.members["1"].attributes == tree["/1"].attributes

        group_in_2 = GroupSpec.from_zarr(group_out, depth=2)  # type: ignore[var-annotated]
        assert group_in_2.members is not None
        assert group_in_2.members["1"].members["2"].attributes == tree["/1/2"].attributes
        assert group_in_2.attributes == tree[""].attributes  # type: ignore[attr-defined]
        assert group_in_2.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]

        group_in_3 = GroupSpec.from_zarr(group_out, depth=3)  # type: ignore[var-annotated]
        assert group_in_3.members is not None
        assert (
            group_in_3.members["1"].members["2"].members["1"].attributes
            == tree["/1/2/1"].attributes
        )
        assert group_in_3.attributes == tree[""].attributes  # type: ignore[attr-defined]
        assert group_in_3.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]
        assert group_in_3.members["1"].members["2"].attributes == tree["/1/2"].attributes  # type: ignore[attr-defined]


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v2 import ArraySpec as ArraySpecv2

    members_flat = {"/a": ArraySpecv2.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v3 ArraySpec or BaseGroupSpec (got type(value)=<class 'pydantic_zarr.v2.ArraySpec'>)"
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


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr-python is not installed")
def test_typed_members(arrayspec) -> None:
    """
    Test GroupSpec creation with typed members
    """

    class DatasetMembers(TypedDict):
        x: ArraySpec
        y: ArraySpec

    class DatasetGroup(GroupSpec):
        members: DatasetMembers

    class ExpectedMembers(TypedDict):
        r10m: DatasetGroup
        r20m: DatasetGroup

    class ExpectedGroup(GroupSpec):
        members: ExpectedMembers

    flat = {
        "": BaseGroupSpec(attributes={}),
        "/r10m": BaseGroupSpec(attributes={}),
        "/r20m": BaseGroupSpec(attributes={}),
        "/r10m/x": arrayspec,
        "/r10m/y": arrayspec,
        "/r20m/x": arrayspec,
        "/r20m/y": arrayspec,
    }

    zg = GroupSpec.from_flat(flat).to_zarr({}, path="")
    ExpectedGroup.from_zarr(zg)


def test_arrayspec_with_methods(arrayspec) -> None:
    """
    Test that ArraySpec with_* methods create new validated copies
    """
    original = arrayspec

    # Test with_attributes
    new_attrs = original.with_attributes({"baz": "qux"})
    assert new_attrs.attributes == {"baz": "qux"}

    # Test with_shape
    new_shape = original.with_shape((20,))
    assert new_shape.shape == (20,)

    # Test with_data_type
    new_dtype = original.with_data_type("float32")
    assert new_dtype.data_type == "float32"

    # Test with_chunk_grid
    new_grid = original.with_chunk_grid({"name": "regular", "configuration": {"chunk_shape": (5,)}})
    assert new_grid.chunk_grid["configuration"]["chunk_shape"] == (5,)  # type: ignore[index]

    # Test with_chunk_key_encoding
    new_encoding = original.with_chunk_key_encoding(
        {"name": "default", "configuration": {"separator": "."}}
    )
    assert new_encoding.chunk_key_encoding["configuration"]["separator"] == "."  # type: ignore[index]

    # Test with_fill_value
    new_fill = original.with_fill_value(999)
    assert new_fill.fill_value == 999

    # Test with_codecs
    new_codecs = ({"name": "gzip", "configuration": {"level": 1}},)
    new_codecs_arr = original.with_codecs(new_codecs)
    assert new_codecs_arr.codecs == new_codecs

    # Test with_storage_transformers
    new_storage_transformers = ({"name": "foo", "configuration": {}},)
    new_transformers_arr = original.with_storage_transformers(new_storage_transformers)
    assert new_transformers_arr.storage_transformers == new_storage_transformers

    # Test with_dimension_names
    new_dims = original.with_dimension_names(("x",))
    assert new_dims.dimension_names == ("x",)


def test_arrayspec_with_methods_validation(arrayspec) -> None:
    """
    Test that ArraySpec with_* methods trigger validation
    """

    # Test that validation fails when dimension_names length doesn't match shape
    with pytest.raises(ValidationError, match="Invalid `dimension names` attribute"):
        arrayspec.with_dimension_names(("x", "y"))  # 2 names for 1D array

    # Test that validation fails with empty codecs
    with pytest.raises(ValidationError, match="Invalid length. Expected 1 or more, got 0"):
        arrayspec.with_codecs(())


def test_groupspec_with_methods(arrayspec) -> None:
    """
    Test that GroupSpec with_* methods create new validated copies
    """
    array_spec = arrayspec
    original = GroupSpec(attributes={"group": "attr"}, members={"arr": array_spec})

    # Test with_attributes
    new_attrs = original.with_attributes({"new": "attr"})
    assert new_attrs.attributes == {"new": "attr"}

    # Test with_members
    new_array = arrayspec.with_attributes({"another": "array"})
    new_members = original.with_members({"new_arr": new_array})
    assert new_members.members == {"new_arr": new_array}


def test_groupspec_with_members_validation(groupspec: GroupSpec) -> None:
    """
    Test that GroupSpec with_members triggers validation
    """

    # Test that validation fails with invalid member names
    with pytest.raises(ValidationError, match='Strings containing "/" are invalid'):
        groupspec.with_members({"a/b": ArraySpec.from_array(np.arange(10), attributes={})})


def test_allowed_extra(arrayspec: ArraySpec, groupspec: GroupSpec) -> None:
    """
    Test that an extra field which is a dict with must_understand=False is allowed
    """

    extra_field = {
        "name": "foo",
        "must_understand": False,
    }

    assert GroupSpec(**groupspec.model_dump(), foo=extra_field).foo == extra_field  # type: ignore[attr-defined]
    assert ArraySpec(**arrayspec.model_dump(), foo=extra_field).foo == extra_field  # type: ignore[attr-defined]


def test_disallowed_extra(arrayspec: ArraySpec, groupspec: GroupSpec) -> None:
    """
    Test that an extra field that is not a dict with must_understand=False causes a validation error.
    """
    extra_field = {
        "name": "foo",
        "must_understand": True,
    }

    with pytest.raises(ValidationError, match=r"foo.must_understand"):
        assert GroupSpec(**groupspec.model_dump(), foo=extra_field).foo == extra_field  # type: ignore[attr-defined]

    with pytest.raises(ValidationError, match=r"foo.must_understand"):
        assert ArraySpec(**arrayspec.model_dump(), foo=extra_field).foo == extra_field  # type: ignore[attr-defined]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_consolidated_metadata_to_from_zarr() -> None:
    """
    Test that GroupSpec.from_zarr picks up consolidated metadata.
    """
    zarr = pytest.importorskip("zarr")
    store: dict[str, object] = {}
    zarr.create_group(store)
    zg = zarr.consolidate_metadata(store)

    gspec = GroupSpec.from_zarr(zg)
    assert gspec.model_dump() == {
        "node_type": "group",
        "zarr_format": 3,
        "attributes": {},
        "members": {},
        "consolidated_metadata": {"kind": "inline", "metadata": {}, "must_understand": False},
    }
    store2: dict[str, object] = {}
    gspec.to_zarr(store2, path="")
    assert json.loads(store["zarr.json"].to_bytes()) == json.loads(store2["zarr.json"].to_bytes())
