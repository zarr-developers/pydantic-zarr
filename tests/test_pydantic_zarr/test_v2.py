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
    assert group.members is not None
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
