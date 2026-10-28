"""
Testts for pydantic_zarr.v2.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_zarr.core import tuplify_json

from .conftest import DTYPE_EXAMPLES_V2, ZARR_PYTHON_VERSION, DTypeExample

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    # Imported under an alias so it is not shadowed by the function-local
    # `zarr = pytest.importorskip("zarr")` bindings used throughout this module.
    import zarr as zarr_mod

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from numcodecs.abc import Codec

from typing import TypedDict

import numpy as np
import numpy.typing as npt
from packaging.version import Version
from zarr_metadata.model import ZarrV2ArrayMetadata, ZarrV2GroupMetadata

from pydantic_zarr.v2 import (
    AnyArraySpec,
    AnyBaseGroupSpec,
    AnyGroupSpec,
    ArraySpec,
    BaseGroupSpec,
    GroupSpec,
    auto_attributes,
    auto_chunks,
    auto_compresser,
    auto_dimension_separator,
    auto_fill_value,
    auto_filters,
    auto_order,
    from_flat,
    from_zarr,
    from_zarr_flat,
    is_array_spec,
    is_group_spec,
    to_flat,
    to_zarr,
)

try:
    import numcodecs
except ImportError:
    numcodecs = None

with suppress(ImportError):
    from zarr.errors import ContainsArrayError, ContainsGroupError

ArrayMemoryOrder = Literal["C", "F"]
DimensionSeparator = Literal[".", "/"]


@pytest.fixture(params=("C", "F"), ids=["C", "F"])
def memory_order(request: pytest.FixtureRequest) -> ArrayMemoryOrder:
    """
    Fixture that returns either "C" or "F"
    """
    if request.param == "C":
        return "C"
    elif request.param == "F":
        return "F"
    msg = f"Invalid array memory order requested. Got {request.param}, expected one of (C, F)."
    raise ValueError(msg)


@pytest.fixture(params=("/", "."), ids=["/", "."])
def dimension_separator(request: pytest.FixtureRequest) -> DimensionSeparator:
    """
    Fixture that returns either "." or "/"
    """
    if request.param == ".":
        return "."
    elif request.param == "/":
        return "/"
    msg = f"Invalid dimension separator requested. Got {request.param}, expected one of (., /)."
    raise ValueError(msg)


@pytest.mark.parametrize("chunks", [(1,), (1, 2), ((1, 2, 3))])
@pytest.mark.parametrize("dtype", ["bool", "uint8", "float64"])
@pytest.mark.parametrize("compressor", [None, "LZMA", "GZip"])
@pytest.mark.parametrize(
    "filters", [(None,), ("delta",), ("scale_offset",), ("delta", "scale_offset")]
)
def test_array_spec(
    chunks: tuple[int, ...],
    memory_order: ArrayMemoryOrder,
    dtype: str,
    dimension_separator: DimensionSeparator,
    compressor: str | None,
    filters: tuple[str, ...] | None,
) -> None:
    zarr = pytest.importorskip("zarr")
    numcodecs = pytest.importorskip("numcodecs")

    if compressor is not None:
        compressor = getattr(numcodecs, compressor)()

    store = zarr.storage.MemoryStore()
    _filters: list[Codec] | None
    if filters is not None:
        _filters = []
        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype))
            if filter == "scale_offset":
                _filters.append(numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype))
    else:
        _filters = filters

    array = zarr.create(
        (100,) * len(chunks),
        path="foo",
        store=store,
        chunks=chunks,
        dtype=dtype,
        order=memory_order,
        dimension_separator=dimension_separator,
        compressor=compressor,
        filters=_filters,
        zarr_format=2,
    )
    attributes = {"foo": [100, 200, 300], "bar": "hello"}
    array.attrs.put(attributes)
    spec: AnyArraySpec = ArraySpec.from_zarr(array)

    assert spec.zarr_format == array.metadata.zarr_format
    assert spec.dtype == array.dtype
    assert spec.attributes == array.attrs.asdict()
    assert spec.chunks == array.chunks

    assert spec.dimension_separator == array.metadata.dimension_separator
    assert spec.shape == array.shape
    assert spec.fill_value == array.fill_value
    # this is a sign that nullability is being misused in zarr-python
    # the correct approach would be to use an empty list to express "no filters".
    if len(array.filters):
        assert spec.filters == [f.get_config() for f in array.filters]
    else:
        assert spec.filters is None

    if len(array.compressors):
        assert spec.compressor == array.compressors[0].get_config()
    else:
        assert spec.compressor is None

    assert spec.order == array.order

    array2 = spec.to_zarr(store, "foo2")

    assert spec.zarr_format == array2.metadata.zarr_format
    assert spec.dtype == array2.dtype
    assert spec.attributes == array2.attrs
    assert spec.chunks == array2.chunks

    if len(array2.compressors):
        assert spec.compressor == cast("Codec", array2.compressors[0]).get_config()
    else:
        assert spec.compressor is None

    if len(array2.filters):
        assert spec.filters == [cast("Codec", f).get_config() for f in array2.filters]
    else:
        assert spec.filters is None

    # a Zarr V2 array's metadata always carries a dimension_separator
    assert (
        spec.dimension_separator
        == array2.metadata.dimension_separator  # pyrefly: ignore[missing-attribute]
    )
    assert spec.shape == array2.shape
    assert spec.fill_value == array2.fill_value

    # test serialization
    store = zarr.storage.MemoryStore()
    stored = spec.to_zarr(store, path="foo")
    assert ArraySpec.from_zarr(stored) == spec

    # test that to_zarr is idempotent
    assert spec.to_zarr(store, path="foo") == stored

    # test that to_zarr raises if the extant array is different
    spec_2 = spec.model_copy(update={"attributes": {"baz": 10}})
    with pytest.raises(ContainsArrayError):
        spec_2.to_zarr(store, path="foo")

    # test that we can overwrite the dissimilar array
    stored_2 = spec_2.to_zarr(store, path="foo", overwrite=True)
    assert ArraySpec.from_zarr(stored_2) == spec_2

    assert spec_2.to_zarr(store, path="foo").read_only is False


@dataclass
class FakeArray:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass
class WithAttrs:
    attrs: dict[str, Any]


@dataclass
class WithChunksize:
    chunksize: tuple[int, ...]


@dataclass
class FakeDaskArray(FakeArray, WithChunksize): ...


@dataclass
class FakeXarray(FakeDaskArray, WithAttrs): ...


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((100), dtype="uint8"),
        FakeArray(shape=(11,), dtype=np.dtype("float64")),
        FakeDaskArray(shape=(22,), dtype=np.dtype("uint8"), chunksize=(11,)),
        FakeXarray(shape=(22,), dtype=np.dtype("uint8"), chunksize=(11,), attrs={"foo": "bar"}),
    ],
)
@pytest.mark.parametrize("chunks", ["omit", "auto", (10,)])
@pytest.mark.parametrize("attributes", ["omit", "auto", {"foo": 10}])
@pytest.mark.parametrize("fill_value", ["omit", "auto", 15])
@pytest.mark.parametrize("order", ["omit", "auto", "F"])
@pytest.mark.parametrize("filters", ["omit", "auto", []])
@pytest.mark.parametrize("dimension_separator", ["omit", "auto", "."])
@pytest.mark.parametrize("compressor", ["omit", "auto", {"id": "gzip", "level": 1}])
def test_array_spec_from_array(
    *,
    array: npt.NDArray[Any],
    chunks: str | tuple[int, ...],
    attributes: str | dict[str, object],
    fill_value: object,
    order: str,
    filters: str | list[Codec],
    dimension_separator: str,
    compressor: str | dict[str, object],
) -> None:
    auto_options = ("omit", "auto")
    kwargs_out: dict[str, Any] = {}

    kwargs_out["chunks"] = chunks
    kwargs_out["attributes"] = attributes
    kwargs_out["fill_value"] = fill_value
    kwargs_out["order"] = order
    kwargs_out["filters"] = filters
    kwargs_out["dimension_separator"] = dimension_separator
    kwargs_out["compressor"] = compressor

    # remove all the keyword arguments that should be defaulted
    kwargs_out = dict(filter(lambda kvp: kvp[1] != "omit", kwargs_out.items()))

    spec: AnyArraySpec = ArraySpec.from_array(array, **kwargs_out)
    # arrayspec should round-trip from_array with no arguments.
    # `from_array` is annotated for ndarray/zarr.Array but documents that it only needs
    # `shape` and `dtype`, which `ArraySpec` supplies.
    assert spec.from_array(spec) == spec  # type: ignore[arg-type]

    assert spec.dtype == array.dtype.str
    assert np.dtype(spec.dtype) == array.dtype

    assert spec.shape == array.shape

    if chunks in auto_options:
        assert spec.chunks == auto_chunks(array)
    else:
        assert spec.chunks == chunks

    if attributes in auto_options:
        assert spec.attributes == auto_attributes(array)
    else:
        assert spec.attributes == attributes

    if fill_value in auto_options:
        assert spec.fill_value == auto_fill_value(array)
    else:
        assert spec.fill_value == fill_value

    if order in auto_options:
        assert spec.order == auto_order(array)
    else:
        assert spec.order == order

    if filters in auto_options:
        assert spec.filters == auto_filters(array)
    else:
        assert spec.filters is None

    if dimension_separator in auto_options:
        assert spec.dimension_separator == auto_dimension_separator(array)
    else:
        assert spec.dimension_separator == dimension_separator

    if compressor in auto_options:
        assert spec.compressor == auto_compresser(array)
    else:
        assert spec.compressor == compressor


@pytest.mark.parametrize("chunks", [(1,), (1, 2), ((1, 2, 3))])
@pytest.mark.parametrize("dtype", ["bool", "uint8", np.dtype("uint8"), "float64"])
@pytest.mark.parametrize("dimension_separator", [".", "/"])
@pytest.mark.parametrize(
    "compressor",
    [{"id": "lzma", "format": 1, "check": -1, "preset": None, "filters": None}, "GZip"],
)
@pytest.mark.parametrize("filters", [(), ("delta",), ("scale_offset",), ("delta", "scale_offset")])
def test_serialize_deserialize_groupspec(
    chunks: tuple[int, ...],
    memory_order: ArrayMemoryOrder,
    dtype: str,
    dimension_separator: Literal[".", "/"],
    compressor: Any,
    filters: tuple[str, ...] | None,
) -> None:
    zarr = pytest.importorskip("zarr")
    numcodecs = pytest.importorskip("numcodecs")
    if isinstance(compressor, str):
        compressor = getattr(numcodecs, compressor)()

    _filters: list[Codec] | None
    if filters is not None:
        _filters = []
        for filter in filters:
            if filter == "delta":
                _filters.append(numcodecs.Delta(dtype))
            if filter == "scale_offset":
                _filters.append(numcodecs.FixedScaleOffset(0, 1.0, dtype=dtype))
    else:
        _filters = filters

    class RootAttrs(TypedDict):
        foo: int
        bar: list[int]

    class SubGroupAttrs(TypedDict):
        a: str
        b: float

    SubGroup = GroupSpec[SubGroupAttrs, Any]

    class ArrayAttrs(TypedDict):
        scale: list[float]

    store = zarr.storage.MemoryStore()

    spec = GroupSpec[RootAttrs, ArraySpec | SubGroup](
        attributes=RootAttrs(foo=10, bar=[0, 1, 2]),
        members={
            "s0": ArraySpec[ArrayAttrs](
                shape=(10,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=cast("Any", _filters),
                compressor=compressor,
                order=memory_order,
                dimension_separator=dimension_separator,
                attributes=ArrayAttrs(scale=[1.0]),
            ),
            "s1": ArraySpec[ArrayAttrs](
                shape=(5,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=cast("Any", _filters),
                compressor=compressor,
                order=memory_order,
                dimension_separator=dimension_separator,
                attributes=ArrayAttrs(scale=[2.0]),
            ),
            "subgroup": SubGroup(attributes=SubGroupAttrs(a="foo", b=1.0)),
        },
    )
    # check that the model round-trips dict representation
    assert spec == GroupSpec(**spec.model_dump())

    # materialize a zarr group, based on the spec
    group = to_zarr(spec, store, "/group_a")

    # parse the spec from that group
    observed = from_zarr(group)
    assert observed == spec

    # assert that we get the same group twice
    assert to_zarr(spec, store, "/group_a", overwrite=True) == group

    # check that we can't call to_zarr targeting the original group with a different spec
    spec_2 = spec.model_copy(update={"attributes": RootAttrs(foo=99, bar=[0, 1, 2])})
    with pytest.raises(ContainsGroupError):
        _ = to_zarr(spec_2, store, "/group_a")

    # check that we can't call to_zarr with the original spec if the group has changed
    group.attrs["foo"] = 100
    with pytest.raises(ContainsGroupError):
        _ = to_zarr(spec, store, "/group_a")
    group.attrs["foo"] = 10

    # materialize again with overwrite
    group2 = to_zarr(spec, store, "/group_a", overwrite=True)
    assert group2 == group

    # again with class methods
    group3 = spec.to_zarr(store, "/group_b")
    observed = spec.from_zarr(group3)
    assert observed == spec


@pytest.mark.parametrize("base", range(1, 5))
def test_shape_chunks(base: int) -> None:
    """
    Test that the length of the chunks and the shape match
    """
    with pytest.raises(ValidationError):
        ArraySpec(shape=(1,) * base, chunks=(1,) * (base + 1), dtype="uint8", attributes={})
    with pytest.raises(ValidationError):
        ArraySpec(shape=(1,) * (base + 1), chunks=(1,) * base, dtype="uint8", attributes={})


def test_validation() -> None:
    """
    Test that specialized GroupSpec and ArraySpec instances cannot be serialized from
    the wrong inputs without a ValidationError.
    """
    zarr = pytest.importorskip("zarr")

    class GroupAttrsA(TypedDict):
        group_a: bool

    class GroupAttrsB(TypedDict):
        group_b: bool

    class ArrayAttrsA(TypedDict):
        array_a: bool

    class ArrayAttrsB(TypedDict):
        array_b: bool

    ArrayA = ArraySpec[ArrayAttrsA]
    ArrayB = ArraySpec[ArrayAttrsB]
    GroupA = GroupSpec[GroupAttrsA, ArrayA]
    GroupB = GroupSpec[GroupAttrsB, ArrayB]

    store = zarr.storage.MemoryStore

    specA = GroupA(
        attributes=GroupAttrsA(group_a=True),
        members={
            "a": ArrayA(
                attributes=ArrayAttrsA(array_a=True),
                shape=(100,),
                dtype="uint8",
                chunks=(10,),
            )
        },
    )

    specB = GroupB(
        attributes=GroupAttrsB(group_b=True),
        members={
            "a": ArrayB(
                attributes=ArrayAttrsB(array_b=True),
                shape=(100,),
                dtype="uint8",
                chunks=(10,),
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

    GroupA.from_zarr(groupAMat)
    GroupB.from_zarr(groupBMat)

    memberA = cast("zarr_mod.Array", groupAMat["a"])
    memberB = cast("zarr_mod.Array", groupBMat["a"])

    ArrayA.from_zarr(memberA)
    ArrayB.from_zarr(memberB)

    with pytest.raises(ValidationError):
        ArrayA.from_zarr(memberB)

    with pytest.raises(ValidationError):
        ArrayB.from_zarr(memberA)

    with pytest.raises(ValidationError):
        GroupB.from_zarr(groupAMat)

    with pytest.raises(ValidationError):
        GroupA.from_zarr(groupBMat)


@pytest.mark.parametrize("shape", [(1,), (2, 2), (3, 4, 5)])
@pytest.mark.parametrize("dtype", [None, "uint8", "float32"])
def test_from_array(shape: tuple[int, ...], dtype: str | None) -> None:
    template = np.zeros(shape=shape, dtype=dtype)
    spec = ArraySpec.from_array(template)  # type: ignore[var-annotated]

    assert spec.shape == template.shape
    assert np.dtype(spec.dtype) == np.dtype(template.dtype)
    assert spec.chunks == template.shape
    assert spec.attributes == {}

    chunks = template.ndim * (1,)
    attrs = {"foo": 100}
    spec2 = ArraySpec.from_array(template, chunks=chunks, attributes=attrs)
    assert spec2.chunks == chunks
    assert spec2.attributes == attrs


@pytest.mark.parametrize("data", ["/", "a/b/c"])
def test_member_name(data: str) -> None:
    with pytest.raises(ValidationError, match=r'Strings containing "/" are invalid\.'):
        GroupSpec(attributes={}, members={data: GroupSpec(attributes={}, members={})})


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ArraySpec.from_array(np.arange(10)),
            {"": ArraySpec.from_array(np.arange(10))},
        ),
        (
            GroupSpec(
                attributes={"foo": 10},
                members={"a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100})},
            ),
            {
                "": BaseGroupSpec(attributes={"foo": 10}),
                "/a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100}),
            },
        ),
        (
            GroupSpec(
                attributes={},
                members={
                    "a": GroupSpec(
                        attributes={"foo": 10},
                        members={"a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100})},
                    ),
                    "b": ArraySpec.from_array(np.arange(2), attributes={"foo": 3}),
                },
            ),
            {
                "": BaseGroupSpec(attributes={}),
                "/a": BaseGroupSpec(attributes={"foo": 10}),
                "/a/a": ArraySpec.from_array(np.arange(5), attributes={"foo": 100}),
                "/b": ArraySpec.from_array(np.arange(2), attributes={"foo": 3}),
            },
        ),
    ],
)
def test_flatten_unflatten(
    data: ArraySpec | GroupSpec, expected: dict[str, ArraySpec | GroupSpec]
) -> None:
    flattened = to_flat(data)
    assert flattened == expected
    assert from_flat(flattened) == data


# todo: parametrize
def test_array_like() -> None:
    a = ArraySpec.from_array(np.arange(10))  # type: ignore[var-annotated]
    assert a.like(a)

    b = a.model_copy(update={"dtype": "uint8"})
    assert not a.like(b)
    assert a.like(b, exclude={"dtype"})
    assert a.like(b, include={"shape"})

    c = a.model_copy(update={"shape": (100, 100)})
    assert not a.like(c)
    assert a.like(c, exclude={"shape"})
    assert a.like(c, include={"dtype"})


def test_array_like_with_zarr() -> None:
    zarr = pytest.importorskip("zarr")
    arr: AnyArraySpec = ArraySpec(shape=(1,), dtype="uint8", chunks=(1,), attributes={})
    store = zarr.storage.MemoryStore()
    arr_stored = arr.to_zarr(store, path="arr")
    assert arr.like(arr_stored)


# todo: parametrize
def test_group_like() -> None:
    tree: dict[str, BaseGroupSpec | ArraySpec] = {
        "": BaseGroupSpec(attributes={"path": ""}),
        "/a": BaseGroupSpec(attributes={"path": "/a"}),
        "/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/b"}),
        "/a/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/a/b"}),
    }
    group = GroupSpec.from_flat(tree)  # type: ignore[var-annotated]
    assert group.like(group)
    assert not group.like(group.model_copy(update={"attributes": None}))
    assert group.like(group.model_copy(update={"attributes": None}), exclude={"attributes"})
    assert group.like(group.model_copy(update={"attributes": None}), include={"members"})


def _stored_hierarchy() -> tuple[zarr_mod.Group, dict[str, AnyArraySpec | AnyBaseGroupSpec]]:
    """Store a three-level hierarchy, and return it with its flat form as read back."""
    zarr = pytest.importorskip("zarr")
    array: AnyArraySpec = ArraySpec.from_array(np.arange(20), attributes={"level": 3})
    tree: dict[str, AnyArraySpec | AnyBaseGroupSpec] = {
        "": BaseGroupSpec(attributes={"level": 0}),
        "/1": BaseGroupSpec(attributes={"level": 1}),
        "/1/2": BaseGroupSpec(attributes={"level": 2}),
        "/1/2/1": BaseGroupSpec(attributes={"level": 3}),
        "/1/2/2": array,
    }
    stored = GroupSpec.from_flat(tree).to_zarr(zarr.storage.MemoryStore(), path="test")
    array_read = ArraySpec.from_zarr(cast("zarr_mod.Array", stored["1/2/2"]))
    return stored, {**tree, "/1/2/2": array_read}


@pytest.mark.parametrize(
    ("path", "depth", "expected_keys"),
    [
        ("", 0, [""]),
        ("", 1, ["", "/1"]),
        ("", 2, ["", "/1", "/1/2"]),
        ("", -1, ["", "/1", "/1/2", "/1/2/1", "/1/2/2"]),
        ("1", 1, ["/1", "/1/2"]),
        ("1/2/2", -1, ["/1/2/2"]),
    ],
)
def test_from_zarr_flat(path: str, depth: int, expected_keys: list[str]) -> None:
    """
    `from_zarr_flat` reads a node and `depth` levels of members under it,
    keyed by path relative to that node. Read in full, it is the flat form
    of what `from_zarr` reads.
    """
    stored, flat = _stored_hierarchy()
    node = stored[path] if path else stored
    prefix = f"/{path}" if path else ""
    expected = {key.removeprefix(prefix): flat[key] for key in expected_keys}
    observed = from_zarr_flat(node, depth=depth)
    assert observed == expected
    if depth == -1:
        assert from_flat(observed) == from_zarr(node)


def test_from_zarr_flat_invalid_depth() -> None:
    """`from_zarr_flat` refuses a depth below -1."""
    stored, _ = _stored_hierarchy()
    with pytest.raises(ValueError, match="Invalid value for depth"):
        from_zarr_flat(stored, depth=-2)


def test_members_none_refused() -> None:
    """`members` may not be `None`: a group without members is a `BaseGroupSpec`."""
    with pytest.raises(ValidationError, match="members"):
        GroupSpec(members=None)  # type: ignore[arg-type]


def test_base_group_member_refused() -> None:
    """A `GroupSpec` is a complete hierarchy, so a `BaseGroupSpec` is not a member."""
    with pytest.raises(ValidationError, match="members"):
        GroupSpec(members={"a": BaseGroupSpec()})  # type: ignore[dict-item]


@pytest.mark.parametrize(("dtype_example"), DTYPE_EXAMPLES_V2, ids=str)
def test_arrayspec_from_zarr(dtype_example: DTypeExample) -> None:
    """
    Test that deserializing an ArraySpec from a zarr python store works as expected.
    """
    zarr = pytest.importorskip("zarr")
    store: dict[str, Any] = {}
    data_type = dtype_example.name
    if ZARR_PYTHON_VERSION >= Version("3.1.0") and data_type == "|O":
        pytest.skip(reason="Data type inference with an object dtype will fail in zarr>=3.1.0")
    arr = zarr.create_array(store=store, shape=(10,), dtype=data_type, zarr_format=2)

    arr_spec: AnyArraySpec = ArraySpec.from_zarr(arr)

    observed = {"attributes": arr.attrs.asdict()} | json.loads(
        store[".zarray"].to_bytes(), object_hook=tuplify_json
    )
    if observed["filters"] is not None:
        observed["filters"] = list(observed["filters"])
    # this covers the case of the structured data type, which would otherwise be deserialized as a
    # tuple of tuples, but is stored on the arrayspec as a list of tuples.
    if isinstance(observed["dtype"], tuple):
        observed["dtype"] = list(observed["dtype"])

    assert arr_spec.model_dump() == observed


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v3 import ArraySpec as ArraySpecv3

    members_flat: dict[str, Any] = {"/a": ArraySpecv3.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v2 ArraySpec or GroupSpec (got type(value)=<class 'pydantic_zarr.v3.ArraySpec'>)"
        ),
    ):
        GroupSpec.from_flat(members_flat)


def test_typevar_defaults_v2() -> None:
    """The v2 generic type parameters carry defaults, so the bare class names are usable."""
    array = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    group = GroupSpec(attributes={"a": 1}, members={"x": array})
    assert tuple(group.members) == ("x",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attributes", {"new": True}),
        ("shape", (8,)),
        ("chunks", (4,)),
        ("fill_value", 1),
        ("order", "F"),
        ("filters", None),
        ("dimension_separator", "."),
        ("compressor", None),
    ],
)
def test_update_v2(field: str, value: object) -> None:
    """The v2 `update` returns a revalidated copy with the new field, everything else unchanged."""
    spec: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    built = spec.update(**{field: value})  # type: ignore[arg-type]
    assert getattr(built, field) == value
    assert built.model_dump(exclude={field}) == spec.model_dump(exclude={field})


def test_update_dtype_v2() -> None:
    """A dtype change routes through parse_dtype_v2, accepting numpy dtypes."""
    spec: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    assert spec.update(dtype=np.dtype("float64")).dtype == "<f8"


def test_base_group_spec_v2() -> None:
    """`BaseGroupSpec` models a v2 group without members, and `like` ignores members."""
    array: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    group: AnyGroupSpec = GroupSpec(attributes={"a": 1}, members={"x": array})
    base = BaseGroupSpec(attributes={"a": 1})
    assert base.like(group)
    assert group.like(base)
    assert not base.like(GroupSpec(attributes={"a": 2}, members={}))
    assert not group.like(BaseGroupSpec(attributes={"a": 2}))
    flat = group.to_flat()
    assert type(flat[""]) is BaseGroupSpec
    assert GroupSpec.from_flat(flat) == group


def test_narrowing_helpers_v2() -> None:
    """The v2 narrowing helpers tell flat-hierarchy values apart."""
    from pydantic_zarr.v2 import is_base_group_spec

    array: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    group: AnyGroupSpec = GroupSpec(attributes={}, members={"a": array})
    base = BaseGroupSpec(attributes={})
    assert is_array_spec(array)
    assert not is_array_spec(group)
    assert is_group_spec(group)
    assert not is_group_spec(base)
    assert is_base_group_spec(base)
    assert is_base_group_spec(group)
    assert not is_base_group_spec(array)


def test_member_name_validation_survives_narrowing_v2() -> None:
    """
    The v2 member-name check is a model validator, so a subclass that narrows
    the `members` annotation cannot drop it.
    """

    class Narrowed(GroupSpec):
        members: dict[str, ArraySpec]  # type: ignore[assignment]

    array: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,), attributes={})
    with pytest.raises(ValidationError, match="/"):
        Narrowed(attributes={}, members={"a/b": array})


def test_attributes_default() -> None:
    """
    `attributes` defaults to the empty dict on every node class, matching a
    stored node with no attributes document.
    """
    assert ArraySpec(shape=(4,), dtype="uint8", chunks=(2,)).attributes == {}
    assert GroupSpec().attributes == {}
    assert BaseGroupSpec().attributes == {}
    assert GroupSpec.model_validate({"zarr_format": 2}).attributes == {}


def test_attributes_default_narrowed_type_rejected() -> None:
    """
    The `attributes` default is validated, so a model whose attributes type
    does not accept the empty dict refuses construction without attributes
    instead of silently holding a value of the wrong type.
    """

    class RequiredAttrs(BaseModel):
        required: int

    with pytest.raises(ValidationError):
        GroupSpec[RequiredAttrs]()


def test_like_filters_keyword_only() -> None:
    """`like` takes its `include`/`exclude` filters as keyword arguments only."""
    group: AnyGroupSpec = GroupSpec()
    with pytest.raises(TypeError):
        group.like(group, {"attributes"})  # type: ignore[call-arg]


def test_resolve_path_v2() -> None:
    """
    `resolve_path` (and `group[path]`) returns the node at a `/`-separated
    relative path; a missing member raises `KeyError` naming the segment.
    """
    leaf: AnyArraySpec = ArraySpec(shape=(4,), dtype="uint8", chunks=(2,))
    root: AnyGroupSpec = GroupSpec(members={"a": GroupSpec(members={"b": leaf})})
    assert root.resolve_path("a/b") == leaf
    assert root["a/b"] == leaf
    with pytest.raises(KeyError, match="No member named 'zz' in a"):
        root.resolve_path("a/zz")


_TO_METADATA_JSON_ARRAY: AnyArraySpec = ArraySpec(
    attributes={"a": [1, 2]},
    shape=(4,),
    chunks=(2,),
    dtype="<f8",
    fill_value="NaN",
    filters=[{"id": "delta", "dtype": "<f8"}],
    compressor={"id": "zstd", "level": 1},
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            _TO_METADATA_JSON_ARRAY,
            {
                "zarr_format": 2,
                "attributes": {"a": [1, 2]},
                "shape": (4,),
                "chunks": (2,),
                "dtype": "<f8",
                "fill_value": "NaN",
                "order": "C",
                "filters": ({"id": "delta", "dtype": "<f8"},),
                "dimension_separator": "/",
                "compressor": {"id": "zstd", "level": 1},
            },
        ),
        (
            ArraySpec(shape=(4,), chunks=(2,), dtype=[("a", "<i4"), ("b", "<f8")], fill_value=None),
            {
                "zarr_format": 2,
                "attributes": {},
                "shape": (4,),
                "chunks": (2,),
                "dtype": (("a", "<i4"), ("b", "<f8")),
                "fill_value": None,
                "order": "C",
                "filters": None,
                "dimension_separator": "/",
                "compressor": None,
            },
        ),
        (
            ArraySpec(
                shape=(4,),
                chunks=(2,),
                dtype="<U1",
                fill_value="",
                filters=[{"id": "categorize", "labels": ["a", "b"], "dtype": "<U1"}],
            ),
            {
                "zarr_format": 2,
                "attributes": {},
                "shape": (4,),
                "chunks": (2,),
                "dtype": "<U1",
                "fill_value": "",
                "order": "C",
                "filters": ({"id": "categorize", "labels": ["a", "b"], "dtype": "<U1"},),
                "dimension_separator": "/",
                "compressor": None,
            },
        ),
        (
            ArraySpec.model_validate({**_TO_METADATA_JSON_ARRAY.model_dump(), "foo": [1]}),
            # extra fields are held, like other JSON values, with arrays as tuples
            {**_TO_METADATA_JSON_ARRAY.to_metadata_json(), "foo": (1,)},
        ),
        (
            BaseGroupSpec(attributes={"a": 1}),
            {"zarr_format": 2, "attributes": {"a": 1}},
        ),
        (
            GroupSpec(
                attributes={"a": 1}, members={"x": _TO_METADATA_JSON_ARRAY, "g": GroupSpec()}
            ),
            {"zarr_format": 2, "attributes": {"a": 1}},
        ),
    ],
    ids=[
        "array",
        "array-structured-dtype",
        "array-filter-config-list",
        "array-extra-field",
        "base-group",
        "group-with-members",
    ],
)
def test_to_metadata_json_v2(
    spec: AnyArraySpec | AnyBaseGroupSpec, expected: dict[str, object]
) -> None:
    """
    `to_metadata_json` returns the node's Zarr V2 metadata as one document holding the `.zarray`
    / `.zgroup` fields and, under `attributes`, the `.zattrs`, with the JSON arrays the type
    declares as tuples held as tuples (but not those inside a codec's configuration), extra
    fields as top-level members, and a group's members left out. (A v2 group holds no extra
    fields: `.zgroup` admits no other keys.) The result reads as a valid document in
    `zarr-metadata`.
    """
    document = spec.to_metadata_json()
    assert document == expected
    model = ZarrV2ArrayMetadata if is_array_spec(spec) else ZarrV2GroupMetadata
    # compared as JSON text, which equates tuples and lists; key order is not compared
    assert json.dumps(model.from_json(document).to_json(), sort_keys=True) == json.dumps(
        document, sort_keys=True
    )


# codecs whose configurations numcodecs reads back as written, so a stored
# layout read by zarr-python is the same spec
_STORED_LAYOUT_ARRAY: AnyArraySpec = ArraySpec(
    attributes={"a": [1, 2]},
    shape=(4,),
    chunks=(2,),
    dtype="<f8",
    fill_value="NaN",
    compressor={"id": "zlib", "level": 1},
)
_STORED_LAYOUT_ZARRAY = {
    "zarr_format": 2,
    "shape": (4,),
    "chunks": (2,),
    "dtype": "<f8",
    "fill_value": "NaN",
    "order": "C",
    "filters": None,
    "dimension_separator": "/",
    "compressor": {"id": "zlib", "level": 1},
}
_STORED_LAYOUT_GROUP = GroupSpec(
    attributes={"a": 1},
    members={
        "x": _STORED_LAYOUT_ARRAY,
        "g": GroupSpec(members={"y": _STORED_LAYOUT_ARRAY, "h": GroupSpec()}),
    },
)


@pytest.mark.parametrize(
    ("dtype", "fill_value", "expected"),
    [
        ("<f8", float("nan"), "NaN"),
        ("<f8", float("inf"), "Infinity"),
        ("<f8", float("-inf"), "-Infinity"),
        ("<f4", np.float32("nan"), "NaN"),
        ("<f8", np.float64("-inf"), "-Infinity"),
        ("<f8", "NaN", "NaN"),
        ("<f8", 1.5, 1.5),
        ("<f8", np.float32(1.5), 1.5),
        ("<i2", np.int16(3), 3),
        ("|b1", np.True_, True),
        ("<c16", complex(float("nan"), float("inf")), ("NaN", "Infinity")),
        ("<c8", np.complex64(complex(1, float("-inf"))), (1.0, "-Infinity")),
        ("<c16", (float("nan"), 1.0), ("NaN", 1.0)),
        ("<c16", ["Infinity", 0.0], ("Infinity", 0.0)),
        ("<f8", None, None),
    ],
)
def test_fill_value_stored(dtype: str, fill_value: object, expected: object) -> None:
    """
    A fill value is stored in the JSON form the Zarr V2 spec gives it: numpy
    scalars as Python scalars, complex numbers as pairs, and non-finite floats
    as the strings "NaN", "Infinity" and "-Infinity", so the metadata
    documents serialize as strict JSON.
    """
    spec = ArraySpec(shape=(4,), chunks=(2,), dtype=dtype, fill_value=cast("Any", fill_value))
    assert spec.fill_value == expected
    assert type(spec.fill_value) is type(expected)
    assert spec.to_metadata_json()["fill_value"] == expected
    json.dumps(spec.to_metadata_json(), allow_nan=False)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("<i4", 0),
        ("<f8", 0),
        ("|b1", None),
        ("<c16", None),
        ("|S3", None),
        ([("a", "<i4"), ("b", "<f8")], None),
    ],
)
def test_fill_value_default(dtype: object, expected: object) -> None:
    """
    A spec built without a fill value, or by `from_array` from an array with
    none, gets 0 when its dtype accepts 0 and None otherwise.
    """
    assert ArraySpec(shape=(4,), chunks=(2,), dtype=cast("Any", dtype)).fill_value == expected
    array = np.zeros((4,), dtype=np.dtype(cast("Any", dtype)))
    assert ArraySpec.from_array(array).fill_value == expected


class _NaNFillArraySpec(ArraySpec):
    fill_value: Literal["NaN"] = "NaN"


class _OneFillArraySpec(ArraySpec):
    fill_value: float = 1.0


@pytest.mark.parametrize(
    ("cls", "expected"), [(_NaNFillArraySpec, "NaN"), (_OneFillArraySpec, 1.0)], ids=str
)
def test_fill_value_default_subclass(cls: type[AnyArraySpec], expected: object) -> None:
    """
    A subclass that declares its own fill value default keeps it, instead of
    the dtype's default.
    """
    assert cls(shape=(4,), chunks=(2,), dtype="<f8").fill_value == expected


def test_update_dtype_keeps_fill_value() -> None:
    """
    A new dtype keeps the current fill value, so changing to a dtype that
    refuses it, without a matching fill value, is a validation error.
    """
    spec = ArraySpec(shape=(4,), chunks=(2,), dtype="<i4")
    assert spec.update(dtype="|b1", fill_value=False).fill_value is False
    with pytest.raises(ValidationError, match=r"fill_value: expected a boolean or null, got 0"):
        spec.update(dtype="|b1")


def test_fill_value_invalid_for_dtype() -> None:
    """
    A fill value the dtype does not accept is refused when the spec is
    validated as a Zarr V2 array metadata document.
    """
    with pytest.raises(ValidationError, match=r"Invalid Zarr V2 array metadata:\nfill_value: "):
        ArraySpec(shape=(4,), chunks=(2,), dtype="<i2", fill_value=1.5)


@pytest.mark.parametrize(
    ("dtype", "fill_value", "expected"),
    [
        ("<f8", np.nan, "NaN"),
        ("<f4", -np.inf, "-Infinity"),
        ("<c8", complex(np.nan, np.inf), ("NaN", "Infinity")),
    ],
)
def test_from_zarr_non_finite_fill_value(dtype: str, fill_value: object, expected: object) -> None:
    """
    A zarr-python v2 array with a non-finite fill value reads back as the spec
    that stores it, and writing that spec stores the same fill value.
    """
    zarr = pytest.importorskip("zarr")
    store = zarr.storage.MemoryStore()
    array = zarr.create_array(
        store, name="a", shape=(4,), chunks=(2,), dtype=dtype, fill_value=fill_value, zarr_format=2
    )
    spec = ArraySpec.from_zarr(array)
    assert spec.fill_value == expected
    assert spec.update(fill_value=cast("Any", fill_value)) == spec
    spec.to_zarr(store, path="b")
    assert ArraySpec.from_zarr(zarr.open_array(store, path="b")) == spec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            _STORED_LAYOUT_ARRAY,
            {".zarray": _STORED_LAYOUT_ZARRAY, ".zattrs": {"a": [1, 2]}},
        ),
        (
            _TO_METADATA_JSON_ARRAY,
            {
                ".zarray": {
                    **_STORED_LAYOUT_ZARRAY,
                    "filters": ({"id": "delta", "dtype": "<f8"},),
                    "compressor": {"id": "zstd", "level": 1},
                },
                ".zattrs": {"a": [1, 2]},
            },
        ),
        (
            ArraySpec.model_validate({**_STORED_LAYOUT_ARRAY.model_dump(), "foo": [1]}),
            # extra fields are members of `.zarray`
            {".zarray": {**_STORED_LAYOUT_ZARRAY, "foo": (1,)}, ".zattrs": {"a": [1, 2]}},
        ),
        (
            BaseGroupSpec(attributes={"a": 1}),
            {".zgroup": {"zarr_format": 2}, ".zattrs": {"a": 1}},
        ),
        (
            GroupSpec(),
            {".zgroup": {"zarr_format": 2}, ".zattrs": {}},
        ),
        (
            _STORED_LAYOUT_GROUP,
            {
                ".zgroup": {"zarr_format": 2},
                ".zattrs": {"a": 1},
                "x/.zarray": _STORED_LAYOUT_ZARRAY,
                "x/.zattrs": {"a": [1, 2]},
                "g/.zgroup": {"zarr_format": 2},
                "g/.zattrs": {},
                "g/y/.zarray": _STORED_LAYOUT_ZARRAY,
                "g/y/.zattrs": {"a": [1, 2]},
                "g/h/.zgroup": {"zarr_format": 2},
                "g/h/.zattrs": {},
            },
        ),
    ],
    ids=[
        "array",
        "array-normalized-codecs",
        "array-extra-field",
        "base-group",
        "group-empty",
        "group-nested",
    ],
)
def test_to_stored_metadata_json_v2(
    spec: AnyArraySpec | AnyBaseGroupSpec, expected: dict[str, object], tmp_path: Path
) -> None:
    """
    `to_stored_metadata_json` returns the Zarr V2 documents of the node and,
    for a `GroupSpec`, of each member at any depth, each at its store key
    prefixed by the member's path: `.zarray` / `.zgroup`, and `.zattrs`,
    written even when the attributes are empty. Read back from JSON, each
    node's documents are what `zarr-metadata`'s `to_key_value` writes for
    it, and the keys and `.zattrs` documents are the ones `to_zarr` writes
    (zarr-python normalizes codec configurations, so `.zarray` is not
    compared). `from_stored_metadata_json` reads the layout back as the same
    spec, and, unless zarr-python normalizes the codecs or drops the extra
    fields, so does `from_zarr` once the documents are written as JSON.
    """
    stored = spec.to_stored_metadata_json()
    assert stored == expected
    assert type(spec).from_stored_metadata_json(stored) == spec
    nodes = spec.to_flat() if isinstance(spec, GroupSpec) else {"": spec}
    for node in nodes.values():
        model = ZarrV2ArrayMetadata if is_array_spec(node) else ZarrV2GroupMetadata
        key_value = model.from_json(node.to_metadata_json()).to_key_value()
        assert {key: json.loads(value) for key, value in key_value.items()} == json.loads(
            json.dumps(node.to_stored_metadata_json())
        )
    as_read = json.loads(json.dumps(stored))
    zarr = pytest.importorskip("zarr")
    store = zarr.storage.MemoryStore()
    spec.to_zarr(store, path="")
    written = {key: json.loads(value.to_bytes()) for key, value in store._store_dict.items()}
    assert written.keys() == as_read.keys()
    assert {k: v for k, v in written.items() if k.endswith(".zattrs")} == {
        k: v for k, v in as_read.items() if k.endswith(".zattrs")
    }
    if spec is _TO_METADATA_JSON_ARRAY or (is_array_spec(spec) and spec.extra_fields):
        return
    for key, document in stored.items():
        (tmp_path / key).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / key).write_text(json.dumps(document))
    assert type(spec).from_zarr(zarr.open(tmp_path, mode="r", zarr_format=2)) == spec


def test_from_stored_metadata_json_v2_base_group_reads_root() -> None:
    """
    `BaseGroupSpec.from_stored_metadata_json` reads a hierarchy's root alone,
    and `GroupSpec` reads a `BaseGroupSpec`'s layout as a group with no
    members: a layout does not tell the two apart. A missing `.zattrs` reads
    as empty attributes.
    """
    assert BaseGroupSpec.from_stored_metadata_json(
        _STORED_LAYOUT_GROUP.to_stored_metadata_json()
    ) == BaseGroupSpec(attributes={"a": 1})
    assert GroupSpec.from_stored_metadata_json(
        BaseGroupSpec(attributes={"a": 1}).to_stored_metadata_json()
    ) == GroupSpec(attributes={"a": 1}, members={})
    assert GroupSpec.from_stored_metadata_json({".zgroup": {"zarr_format": 2}}) == GroupSpec()


def test_from_stored_metadata_json_v2_unrecognized_key() -> None:
    """A key other than a node's `.zarray`, `.zgroup` or `.zattrs` is refused."""
    stored = {**_STORED_LAYOUT_GROUP.to_stored_metadata_json(), ".zmetadata": {}}
    with pytest.raises(ValueError, match="Unrecognized store key '.zmetadata'"):
        GroupSpec.from_stored_metadata_json(stored)


def test_from_stored_metadata_json_v2_array_and_group() -> None:
    """A path holding both a `.zarray` and a `.zgroup` is refused."""
    stored = {**_STORED_LAYOUT_GROUP.to_stored_metadata_json(), "x/.zgroup": {"zarr_format": 2}}
    with pytest.raises(ValueError, match="both a .zarray and a .zgroup at 'x'"):
        GroupSpec.from_stored_metadata_json(stored)


def test_from_stored_metadata_json_v2_orphan_zattrs() -> None:
    """A `.zattrs` with neither a `.zarray` nor a `.zgroup` beside it is refused."""
    stored = {**_STORED_LAYOUT_GROUP.to_stored_metadata_json(), "z/.zattrs": {}}
    with pytest.raises(ValueError, match="a .zattrs at 'z', but no .zarray or .zgroup"):
        GroupSpec.from_stored_metadata_json(stored)


def test_from_stored_metadata_json_v2_no_root() -> None:
    """A layout without the root's `.zgroup` is refused."""
    stored = dict(_STORED_LAYOUT_GROUP.to_stored_metadata_json())
    del stored[".zgroup"], stored[".zattrs"]
    with pytest.raises(ValueError, match="no group's metadata at the root"):
        GroupSpec.from_stored_metadata_json(stored)


def test_from_stored_metadata_json_v2_root_of_other_type() -> None:
    """A layout whose root is not the node type of the class is refused."""
    with pytest.raises(ValueError, match="root is a group, not an array"):
        ArraySpec.from_stored_metadata_json(GroupSpec().to_stored_metadata_json())


def test_from_stored_metadata_json_v2_array_with_members() -> None:
    """An array's layout holding other nodes' documents is refused."""
    stored = {**_STORED_LAYOUT_ARRAY.to_stored_metadata_json(), "g/.zgroup": {"zarr_format": 2}}
    with pytest.raises(ValueError, match="An array has no members"):
        ArraySpec.from_stored_metadata_json(stored)


@pytest.mark.parametrize("parent", ["missing", "array"])
def test_from_stored_metadata_json_v2_parent_not_group(parent: str) -> None:
    """A node whose parent is not a group in the layout is refused."""
    stored: dict[str, object] = dict(_STORED_LAYOUT_GROUP.to_stored_metadata_json())
    if parent == "missing":
        del stored["g/.zgroup"], stored["g/.zattrs"]
        member = "g/y"
    else:
        stored["x/z/.zgroup"] = {"zarr_format": 2}
        member = "x/z"
    with pytest.raises(ValueError, match=f"documents at '{member}', but no group's"):
        GroupSpec.from_stored_metadata_json(stored)
