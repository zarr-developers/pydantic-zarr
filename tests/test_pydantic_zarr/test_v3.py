from __future__ import annotations

import importlib
import importlib.util
import json
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from packaging.version import Version
from pydantic import ValidationError

from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import (
    AnyArraySpec,
    AnyBaseGroupSpec,
    AnyGroupSpec,
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

from .conftest import DTYPE_EXAMPLES_V3, ZARR_PYTHON_VERSION, DTypeExample

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams

ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None


@pytest.mark.parametrize("invalid_dimension_names", [[], "hi", ["1", 2, None]], ids=str)
def test_dimension_names_validation(invalid_dimension_names: object) -> None:
    """
    Test that the `dimension_names` attribute is rejected if any of the following are true:
    - it is a sequence with length different from the number of dimensions of the array
    - it is a sequence containing values other than strings or `None`.
    - it is neither a valid sequence nor the value `None`.
    """
    base_array: AnyArraySpec = ArraySpec(
        shape=(1, 2, 3),
        data_type="int8",
        codecs=({"name": "bytes"},),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (1, 2, 3)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0,
        attributes={},
    )
    with pytest.raises(ValidationError):
        ArraySpec(**(base_array.model_dump() | {"dimension_names": invalid_dimension_names}))


def test_serialize_deserialize() -> None:
    array_attributes = {"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]}

    group_attributes = {"group": True}

    array_spec: AnyArraySpec = ArraySpec(
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
    array = np.arange(10)
    array_spec: AnyArraySpec = ArraySpec.from_array(array)

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
    spec: AnyArraySpec = ArraySpec(
        attributes={},
        shape=array.shape,
        data_type=zdt.to_json(zarr_format=3),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": array.shape}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=auto_codecs(array),
        fill_value=cast("Any", zdt.to_json_scalar(zdt.default_scalar(), zarr_format=3)),
    )
    spec.to_zarr(store={}, path="")  # type: ignore[arg-type]


def test_arrayspec_no_empty_codecs() -> None:
    """
    Ensure that it is not possible to create an ArraySpec with no codecs
    """

    with pytest.raises(
        ValidationError, match=r"Value error, Invalid length\. Expected 1 or more, got 0\."
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
    store: dict[str, Any] = {}

    data_type = dtype_example.name

    if data_type == "variable_length_bytes":
        pytest.skip(
            reason="Bug in zarr python: see https://github.com/zarr-developers/zarr-python/issues/3263"
        )

    arr = zarr.create_array(store=store, shape=(10,), dtype=data_type, zarr_format=3)

    arr_spec: AnyArraySpec = ArraySpec.from_zarr(arr)
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

    codecs: tuple[CodecLike, ...] = ({"name": "bytes", "configuration": {"endian": "little"}},)
    if data_type == "variable_length_bytes":
        codecs = ({"name": "vlen-bytes"},)

    elif data_type in ("str", "string"):
        codecs = ({"name": "vlen-utf8"},)

    store: dict[str, Any] = {}

    arr_spec: AnyArraySpec = ArraySpec(
        attributes={},
        shape=(10,),
        data_type=data_type,
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (10,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=codecs,
        fill_value=cast("Any", fill_value),
        dimension_names=("x",),
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


def get_flat_example() -> tuple[dict[str, AnyArraySpec | AnyBaseGroupSpec], AnyGroupSpec]:
    """
    Get example data for testing to_flat and from_flat.

    The returned value is a tuple with two elements: a flattened dict representation of a hierarchy,
    and the root group, with all of its members (i.e., the non-flat version of that hierarchy).
    """
    a1 = ArraySpec.from_array(np.arange(10), attributes={"name": "/a1"})
    a2 = ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/a2"})
    a3 = ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/g2/a3"})
    members_flat: dict[str, AnyArraySpec | AnyBaseGroupSpec] = {
        "": BaseGroupSpec(attributes={"name": ""}),
        "/a1": a1,
        "/g1": BaseGroupSpec(attributes={"name": "/g1"}),
        "/g1/a2": a2,
        "/g1/g2": BaseGroupSpec(attributes={"name": "/g1/g2"}),
        "/g1/g2/a3": a3,
    }
    g2 = GroupSpec(attributes={"name": "/g1/g2"}, members={"a3": a3})
    g1 = GroupSpec(attributes={"name": "/g1"}, members={"a2": a2, "g2": g2})
    root = GroupSpec(attributes={"name": ""}, members={"g1": g1, "a1": a1})
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
        codecs: tuple[CodecLike, ...] = ({"name": "bytes", "configuration": {"endian": "little"}},)
        tree: dict[str, AnyGroupSpec | AnyArraySpec] = {
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
        assert group_in_1.attributes == tree[""].attributes
        assert group_in_1.members is not None
        assert group_in_1.members["1"] == tree["/1"]

        group_in_2 = GroupSpec.from_zarr(group_out, depth=2)  # type: ignore[var-annotated]
        assert group_in_2.members is not None
        sub_2_1 = group_in_2.members["1"]
        assert isinstance(sub_2_1, GroupSpec) and sub_2_1.members is not None
        assert sub_2_1.members["2"] == tree["/1/2"]
        assert group_in_2.attributes == tree[""].attributes
        assert sub_2_1.attributes == tree["/1"].attributes

        group_in_3 = GroupSpec.from_zarr(group_out, depth=3)  # type: ignore[var-annotated]
        assert group_in_3.members is not None
        sub_3_1 = group_in_3.members["1"]
        assert isinstance(sub_3_1, GroupSpec) and sub_3_1.members is not None
        sub_3_2 = sub_3_1.members["2"]
        assert isinstance(sub_3_2, GroupSpec) and sub_3_2.members is not None
        assert sub_3_2.members["1"] == tree["/1/2/1"]
        assert group_in_3.attributes == tree[""].attributes
        assert sub_3_1.attributes == tree["/1"].attributes
        assert sub_3_2.attributes == tree["/1/2"].attributes


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v2 import ArraySpec as ArraySpecv2

    members_flat: dict[str, Any] = {"/a": ArraySpecv2.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v3 ArraySpec or GroupSpec (got type(value)=<class 'pydantic_zarr.v2.ArraySpec'>)"
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
    spec: AnyArraySpec = ArraySpec.from_zarr(arr)
    assert spec.dimension_names == expected_names


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


def _make_array_spec() -> AnyArraySpec:
    """Return a minimal ArraySpec with dimension_names=None for regression tests."""
    return ArraySpec(
        attributes={},
        shape=(4,),
        data_type="uint8",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        codecs=({"name": "bytes"},),
        fill_value=0,
    )


def test_arrayspec_like_spec_vs_spec() -> None:
    """
    Regression test: ArraySpec.like(other_spec) must not raise NameError.
    Previously crashed because `zarr` was only imported under TYPE_CHECKING.
    """
    spec = _make_array_spec()
    assert spec.like(spec)


def test_arrayspec_like_spec_vs_zarr_array() -> None:
    """
    Regression test: ArraySpec.like(zarr_array) must not raise NameError.
    Previously zarr was only imported under TYPE_CHECKING so isinstance check crashed.
    """
    zarr = pytest.importorskip("zarr")
    arr = zarr.create_array(store={}, shape=(4,), dtype="uint8", zarr_format=3)
    spec: AnyArraySpec = ArraySpec.from_zarr(arr)
    assert spec.like(arr)


def test_from_zarr_array() -> None:
    """
    Regression test: module-level from_zarr on a zarr array must not raise NameError.
    Previously the function body referenced `zarr.Array` without a runtime import.
    """
    zarr = pytest.importorskip("zarr")
    from pydantic_zarr.v3 import from_zarr

    arr = zarr.create_array(store={}, shape=(4,), dtype="uint8", zarr_format=3)
    result = from_zarr(arr)
    assert isinstance(result, ArraySpec)


def test_from_zarr_group() -> None:
    """
    Regression test: module-level from_zarr on a zarr group must not raise NameError.
    """
    zarr = pytest.importorskip("zarr")
    from pydantic_zarr.v3 import from_zarr

    grp = zarr.open_group(store={}, mode="w", zarr_format=3)
    result = from_zarr(grp)
    assert isinstance(result, GroupSpec)


def test_model_dump_exclude_dimension_names() -> None:
    """
    Regression test: model_dump(exclude={'dimension_names'}) must not raise KeyError.
    Previously the override did d["dimension_names"] unconditionally.
    """
    spec = _make_array_spec()
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


def _v3_array_kwargs(**overrides: object) -> dict[str, Any]:
    """Keyword arguments for a minimal valid v3 ArraySpec, with overrides."""
    kwargs: dict[str, Any] = {
        "attributes": {},
        "shape": (4, 4),
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "NaN",
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    kwargs.update(overrides)
    return kwargs


def test_typevar_defaults() -> None:
    """
    The generic type parameters carry defaults, so the bare class names are
    usable without parametrization, parametrized forms keep working, and a
    subclass can narrow a field annotation without generics.
    """
    from pydantic import BaseModel

    array = ArraySpec(**_v3_array_kwargs())
    group = GroupSpec(attributes={"a": 1}, members={"x": array})
    assert group.members is not None and tuple(group.members) == ("x",)

    class MyAttrs(BaseModel):
        scale: float

    typed = ArraySpec[MyAttrs](**_v3_array_kwargs(attributes=MyAttrs(scale=2.0)))
    assert typed.attributes.scale == 2.0

    class Narrow(ArraySpec):
        attributes: MyAttrs

    assert Narrow(**_v3_array_kwargs(attributes={"scale": 3.0})).attributes.scale == 3.0


@pytest.mark.parametrize(
    ("method", "value", "field"),
    [
        ("with_attributes", {"new": True}, "attributes"),
        ("with_shape", (8, 8), "shape"),
        ("with_data_type", "int16", "data_type"),
        (
            "with_chunk_grid",
            {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
            "chunk_grid",
        ),
        (
            "with_chunk_key_encoding",
            {"name": "v2", "configuration": {"separator": "."}},
            "chunk_key_encoding",
        ),
        ("with_fill_value", "Infinity", "fill_value"),
        ("with_codecs", ({"name": "bytes"},), "codecs"),
        ("with_storage_transformers", (), "storage_transformers"),
        ("with_dimension_names", ("y", "x"), "dimension_names"),
    ],
)
def test_with_builders(method: str, value: object, field: str) -> None:
    """Every `with_*` builder returns a revalidated copy with the one new field."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    built = getattr(spec, method)(value)
    assert getattr(built, field) == value
    # everything else is unchanged
    assert built.model_dump(exclude={field}) == spec.model_dump(exclude={field})


def test_with_builders_revalidate() -> None:
    """A `with_*` builder revalidates: an invalid new field is refused."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs(dimension_names=("a", "b")))
    with pytest.raises(ValidationError, match="dimension names"):
        spec.with_shape((1, 2, 3))


def test_base_group_spec() -> None:
    """
    `BaseGroupSpec` models a group without members: `to_flat` emits it for
    every group, `from_flat` accepts it, and `like` compares a group's own
    metadata, reducing a `GroupSpec` to its own metadata first.
    """
    from pydantic_zarr.v3 import BaseGroupSpec

    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    inner: AnyGroupSpec = GroupSpec(attributes={"b": 2}, members={"arr": array})
    outer: AnyGroupSpec = GroupSpec(attributes={"a": 1}, members={"sub": inner})

    flat = outer.to_flat()
    assert type(flat[""]) is BaseGroupSpec
    assert type(flat["/sub"]) is BaseGroupSpec
    assert type(flat["/sub/arr"]) is ArraySpec
    assert GroupSpec.from_flat(flat) == outer

    base = BaseGroupSpec(attributes={"a": 1})
    assert base.like(outer)  # members do not participate
    assert not base.like(GroupSpec(attributes={"a": 2}, members={}))
    assert base.with_attributes({"a": 2}).attributes == {"a": 2}


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_base_group_spec_zarr_round_trip() -> None:
    """`BaseGroupSpec.to_zarr` writes only the group's own metadata, and `from_zarr` reads it back."""
    import zarr.storage

    from pydantic_zarr.v3 import BaseGroupSpec

    store = zarr.storage.MemoryStore()
    base = BaseGroupSpec(attributes={"a": 1})
    group = base.to_zarr(store, path="g")
    assert group.attrs.asdict() == {"a": 1}
    assert BaseGroupSpec.from_zarr(group) == base
