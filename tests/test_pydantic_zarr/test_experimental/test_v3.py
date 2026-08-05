from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from packaging.version import Version
from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_zarr.experimental.core import json_eq
from pydantic_zarr.experimental.v3 import (
    AnyNamedConfig,
    ArraySpec,
    BaseGroupSpec,
    CodecLike,
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingConfig,
    GroupSpec,
    NamedConfig,
    RegularChunking,
    RegularChunkingConfig,
    auto_codecs,
    parse_dtype_v3,
)

from ..conftest import DTYPE_EXAMPLES_V3, ZARR_AVAILABLE, ZARR_PYTHON_VERSION, DTypeExample

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams


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
        data_type=data_type,
        chunk_grid=chunk_grid,
        chunk_key_encoding=chunk_key_encoding,
        fill_value=fill_value,
        codecs=codecs,
        attributes=attributes,
    )


@pytest.fixture
def flat_example(arrayspec: ArraySpec) -> tuple[dict[str, ArraySpec | BaseGroupSpec], GroupSpec]:
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
        cast("Mapping[str, str]", a.attributes)["name"]: a for a in named_nodes
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


@pytest.mark.filterwarnings("ignore:The data type :FutureWarning")
@pytest.mark.parametrize(
    ("dtype", "expected_codecs"),
    [
        (np.dtype("uint8"), ({"name": "bytes"},)),
        (np.dtype("bool"), ({"name": "bytes"},)),
        (np.dtype("int64"), ({"name": "bytes", "configuration": {"endian": "little"}},)),
        (np.dtype("float32"), ({"name": "bytes", "configuration": {"endian": "little"}},)),
        # for structured data types, itemsize is the total bytes per element, including padding
        (
            np.dtype([("a", "<i4"), ("b", "<f2")]),
            ({"name": "bytes", "configuration": {"endian": "little"}},),
        ),
        (
            np.dtype([("a", "i1"), ("b", "u1")]),
            ({"name": "bytes", "configuration": {"endian": "little"}},),
        ),
        (np.dtype([("a", "i1")]), ({"name": "bytes"},)),
        (
            np.dtype([("a", "i1"), ("b", "<f8")], align=True),
            ({"name": "bytes", "configuration": {"endian": "little"}},),
        ),
    ],
    ids=str,
)
def test_auto_codecs(dtype: np.dtype[Any], expected_codecs: tuple[CodecLike, ...]) -> None:
    """
    Test that auto_codecs emits a bytes codec with an explicit little-endian configuration
    exactly when the data type is multi-byte, including structured data types, and that zarr
    accepts the generated codecs when creating an array with that data type.
    """
    array = np.zeros((3,), dtype=dtype)
    assert auto_codecs(array) == expected_codecs

    if not ZARR_AVAILABLE or ZARR_PYTHON_VERSION < Version("3.1.0"):
        return
    from zarr.core.dtype import get_data_type_from_native_dtype

    zdt = get_data_type_from_native_dtype(dtype)
    spec = ArraySpec(
        attributes={},
        shape=array.shape,
        data_type=zdt.to_json(zarr_format=3),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": array.shape}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=auto_codecs(array),
        fill_value=zdt.to_json_scalar(zdt.default_scalar(), zarr_format=3),
    )
    spec.to_zarr(store={}, path="")  # type: ignore[arg-type]


def test_arrayspec_no_empty_codecs(arrayspec: ArraySpec) -> None:
    """
    Ensure that it is not possible to create an ArraySpec with no codecs
    """

    with pytest.raises(
        ValidationError, match=r"Value error, Invalid length\. Expected 1 or more, got 0\."
    ):
        ArraySpec(**(arrayspec.model_dump() | {"codecs": ()}))


@pytest.mark.filterwarnings("ignore:The dtype:UserWarning")
@pytest.mark.filterwarnings("ignore:The data type:FutureWarning")
@pytest.mark.filterwarnings("ignore:The codec:UserWarning")
@pytest.mark.parametrize("dtype_example", DTYPE_EXAMPLES_V3, ids=str)
def test_arrayspec_from_zarr(dtype_example: DTypeExample) -> None:
    """
    Test that deserializing an ArraySpec from a zarr python store works as expected.
    """
    zarr = pytest.importorskip("zarr")
    store: dict[str, Any] = {}

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

    codecs: tuple[CodecLike, ...] = ({"name": "bytes", "configuration": {"endian": "little"}},)
    if data_type == "variable_length_bytes":
        codecs = ({"name": "vlen-bytes"},)

    elif data_type in ("str", "string"):
        codecs = ({"name": "vlen-utf8"},)

    store: dict[str, Any] = {}

    arr_spec = arrayspec.model_copy(
        update={"data_type": data_type, "fill_value": fill_value, "codecs": codecs}
    )

    if not ZARR_AVAILABLE:
        return

    # zarr accepts a plain dict as a store at runtime, and `config` is a loosely-typed
    # parametrization of the `ArrayConfigParams` TypedDict.
    arr = arr_spec.to_zarr(
        store=cast("Store", store),
        path=path,
        overwrite=overwrite,
        config=cast("ArrayConfigParams", config),
    )
    assert arr._async_array.metadata == arr._async_array.metadata
    for key, value in config.items():
        assert asdict(arr._async_array._config)[key] == value


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
    def test_from_zarr_depth(arrayspec: ArraySpec) -> None:
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
        group_in_0 = GroupSpec.from_zarr(group_out, depth=0)
        assert group_in_0.attributes == tree[""].attributes

        group_in_1 = GroupSpec.from_zarr(group_out, depth=1)
        assert group_in_1.attributes == tree[""].attributes
        assert group_in_1.members is not None
        assert group_in_1.members["1"].attributes == tree["/1"].attributes

        group_in_2 = GroupSpec.from_zarr(group_out, depth=2)
        assert group_in_2.members is not None
        member_1 = group_in_2.members["1"]
        assert isinstance(member_1, GroupSpec)
        assert member_1.members["2"].attributes == tree["/1/2"].attributes
        assert group_in_2.attributes == tree[""].attributes
        assert member_1.attributes == tree["/1"].attributes

        group_in_3 = GroupSpec.from_zarr(group_out, depth=3)
        assert group_in_3.members is not None
        member_1 = group_in_3.members["1"]
        assert isinstance(member_1, GroupSpec)
        member_2 = member_1.members["2"]
        assert isinstance(member_2, GroupSpec)
        assert member_2.members["1"].attributes == tree["/1/2/1"].attributes
        assert group_in_3.attributes == tree[""].attributes
        assert member_1.attributes == tree["/1"].attributes
        assert member_2.attributes == tree["/1/2"].attributes


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v2 import ArraySpec as ArraySpecv2

    members_flat: dict[str, Any] = {"/a": ArraySpecv2.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v3 ArraySpec or BaseGroupSpec (got type(value)=<class 'pydantic_zarr.v2.ArraySpec'>)"
        ),
    ):
        GroupSpec.from_flat(members_flat)


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
def test_typed_members(arrayspec: ArraySpec) -> None:
    """
    Test GroupSpec creation with typed members
    """

    class DatasetMembers(TypedDict):
        x: ArraySpec
        y: ArraySpec

    class DatasetGroup(GroupSpec):
        # Narrowing `members` to a TypedDict is the point of this test; mypy treats it as an
        # incompatible override of the base `Mapping[str, ArraySpec | GroupSpec]` field.
        members: DatasetMembers  # type: ignore[assignment]

    class ExpectedMembers(TypedDict):
        r10m: DatasetGroup
        r20m: DatasetGroup

    class ExpectedGroup(GroupSpec):
        members: ExpectedMembers  # type: ignore[assignment]

    flat: dict[str, ArraySpec | BaseGroupSpec] = {
        "": BaseGroupSpec(attributes={}),
        "/r10m": BaseGroupSpec(attributes={}),
        "/r20m": BaseGroupSpec(attributes={}),
        "/r10m/x": arrayspec,
        "/r10m/y": arrayspec,
        "/r20m/x": arrayspec,
        "/r20m/y": arrayspec,
    }

    zg = GroupSpec.from_flat(flat).to_zarr(cast("Store", {}), path="")
    ExpectedGroup.from_zarr(zg)


def test_arrayspec_with_methods(arrayspec: ArraySpec) -> None:
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
    assert new_grid.chunk_grid["configuration"]["chunk_shape"] == (5,)

    # Test with_chunk_key_encoding
    new_encoding = original.with_chunk_key_encoding(
        {"name": "default", "configuration": {"separator": "."}}
    )
    assert new_encoding.chunk_key_encoding["configuration"]["separator"] == "."

    # Test with_fill_value
    new_fill = original.with_fill_value(999)
    assert new_fill.fill_value == 999

    # Test with_codecs
    new_codecs: tuple[CodecLike, ...] = ({"name": "gzip", "configuration": {"level": 1}},)
    new_codecs_arr = original.with_codecs(new_codecs)
    assert new_codecs_arr.codecs == new_codecs

    # Test with_storage_transformers
    new_storage_transformers: tuple[AnyNamedConfig, ...] = ({"name": "foo", "configuration": {}},)
    new_transformers_arr = original.with_storage_transformers(new_storage_transformers)
    assert new_transformers_arr.storage_transformers == new_storage_transformers

    # Test with_dimension_names
    new_dims = original.with_dimension_names(("x",))
    assert new_dims.dimension_names == ("x",)


def test_arrayspec_with_methods_validation(arrayspec: ArraySpec) -> None:
    """
    Test that ArraySpec with_* methods trigger validation
    """

    # Test that validation fails when dimension_names length doesn't match shape
    with pytest.raises(ValidationError, match="Invalid `dimension names` attribute"):
        arrayspec.with_dimension_names(("x", "y"))  # 2 names for 1D array

    # Test that validation fails with empty codecs
    with pytest.raises(ValidationError, match=r"Invalid length\. Expected 1 or more, got 0"):
        arrayspec.with_codecs(())


def test_groupspec_with_methods(arrayspec: ArraySpec) -> None:
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
    store: dict[str, Any] = {}
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
    store2: dict[str, Any] = {}
    gspec.to_zarr(cast("Store", store2), path="")
    assert json.loads(store["zarr.json"].to_bytes()) == json.loads(store2["zarr.json"].to_bytes())


def _make_array_spec_exp() -> ArraySpec:
    """Return a minimal ArraySpec (experimental) with dimension_names=None."""
    return ArraySpec(
        attributes={},
        shape=(4,),
        data_type="uint8",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=({"name": "bytes"},),
        fill_value=0,
    )


def test_exp_arrayspec_like_spec_vs_spec() -> None:
    """
    Regression test: experimental ArraySpec.like(other_spec) must not raise NameError.
    """
    spec = _make_array_spec_exp()
    assert spec.like(spec)


def test_exp_arrayspec_like_spec_vs_zarr_array() -> None:
    """
    Regression test: experimental ArraySpec.like(zarr_array) must not raise NameError.
    Previously zarr was only imported under TYPE_CHECKING so isinstance check crashed.
    """
    zarr = pytest.importorskip("zarr")
    arr = zarr.create_array(store={}, shape=(4,), dtype="uint8", zarr_format=3)
    spec = ArraySpec.from_zarr(arr)
    assert spec.like(arr)


def test_exp_from_zarr_array() -> None:
    """
    Regression test: experimental module-level from_zarr on a zarr array must not raise NameError.
    """
    zarr = pytest.importorskip("zarr")
    from pydantic_zarr.experimental.v3 import from_zarr

    arr = zarr.create_array(store={}, shape=(4,), dtype="uint8", zarr_format=3)
    result = from_zarr(arr)
    assert isinstance(result, ArraySpec)


def test_exp_from_zarr_group() -> None:
    """
    Regression test: experimental module-level from_zarr on a zarr group must not raise NameError.
    """
    zarr = pytest.importorskip("zarr")
    from pydantic_zarr.experimental.v3 import from_zarr

    grp = zarr.open_group(store={}, mode="w", zarr_format=3)
    result = from_zarr(grp)
    assert isinstance(result, GroupSpec)


def test_exp_model_dump_exclude_dimension_names() -> None:
    """
    Regression test: experimental model_dump(exclude={'dimension_names'}) must not raise KeyError.
    """
    spec = _make_array_spec_exp()
    d = spec.model_dump(exclude={"dimension_names"})
    assert "dimension_names" not in d


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.dtype("int8"), "int8"),
        (np.dtype("int16"), "int16"),
        (np.dtype("int32"), "int32"),
        (np.dtype("int64"), "int64"),
        (np.dtype("uint8"), "uint8"),
        (np.dtype("uint16"), "uint16"),
        (np.dtype("uint32"), "uint32"),
        (np.dtype("uint64"), "uint64"),
        (np.dtype("float16"), "float16"),
        (np.dtype("float32"), "float32"),
        (np.dtype("float64"), "float64"),
        (np.dtype("complex64"), "complex64"),
        (np.dtype("complex128"), "complex128"),
    ],
    ids=str,
)
def test_parse_dtype_v3_numpy(dtype: np.dtype, expected: str) -> None:
    """
    Regression test: parse_dtype_v3 must correctly handle all supported numpy dtypes.
    Previously, the float64 and complex64 match arms were copy-paste errors (using
    Float16DType and Float32DType respectively), making those dtypes unreachable and
    causing ValueError to be raised for float64 and complex64 inputs.
    """
    assert parse_dtype_v3(dtype) == expected


def test_v2_chunk_key_encoding() -> None:
    # Simple smoke test to make sure v2 chunk key encoding is allowed
    ArraySpec(
        attributes={},
        shape=[1000, 1000],
        dimension_names=["rows", "columns"],
        data_type="float64",
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": [1000, 100]}),
        chunk_key_encoding=NamedConfig(name="v2", configuration={"separator": "."}),
        codecs=[NamedConfig(name="GZip", configuration={"level": 1})],
        fill_value="NaN",
        storage_transformers=[],
    )


@pytest.mark.parametrize("separator", [".", "/"])
def test_v2_chunk_key_encoding_round_trip(separator: str) -> None:
    """
    Test that a zarr v3 array with a v2 chunk key encoding can be round-tripped through
    ArraySpec: from_zarr then to_zarr should yield structurally identical metadata.
    """
    zarr = pytest.importorskip("zarr")
    store: dict[str, Any] = {}
    arr = zarr.create_array(
        store=store,
        shape=(10,),
        dtype="float64",
        zarr_format=3,
        chunk_key_encoding={"name": "v2", "configuration": {"separator": separator}},
    )

    spec = ArraySpec.from_zarr(arr)
    assert spec.chunk_key_encoding == {
        "name": "v2",
        "configuration": {"separator": separator},
    }

    store_out: dict[str, Any] = {}
    spec.to_zarr(cast("Store", store_out), path="")
    assert json.loads(store["zarr.json"].to_bytes()) == json.loads(
        store_out["zarr.json"].to_bytes()
    )
