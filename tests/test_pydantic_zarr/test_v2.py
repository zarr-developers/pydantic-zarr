"""
Testts for pydantic_zarr.v2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError
from typing_extensions import TypedDict
from zarr.errors import ContainsGroupError

from pydantic_zarr.base import ArrayV2Config
from pydantic_zarr.engine import zarrify_array_v2
from pydantic_zarr.zarr_v2.v2 import (
    ArraySpec,
    GroupSpec,
    from_flat,
    from_zarr,
    to_flat,
)

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
@pytest.mark.parametrize("compressor", [None, {"id": "lzma"}, {"id": "gzip"}])
@pytest.mark.parametrize(
    "filters", [(None,), ("delta",), ("scale_offset",), ("delta", "scale_offset")]
)
def test_array_spec(
    chunks: tuple[int, ...],
    memory_order: ArrayMemoryOrder,
    dtype: str,
    dimension_separator: DimensionSeparator,
    compressor: None | dict[str, object],
    filters: tuple[str, ...] | None,
) -> None:
    import numcodecs
    import zarr
    from numcodecs.abc import Codec  # noqa: TC002

    store = {}
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
    if compressor is not None:
        _compressor = numcodecs.get_codec(compressor)
    else:
        _compressor = None
    from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

    array = zarr.create_array(
        store=store,
        shape=(100,) * len(chunks),
        name="",
        chunks=chunks,
        dtype=dtype,
        order=memory_order,
        chunk_key_encoding=V2ChunkKeyEncoding(separator=dimension_separator),
        compressors=_compressor,
        filters=_filters,
        attributes={"foo": [100, 200, 300], "bar": "hello"},
        zarr_format=2,
    )

    spec = ArraySpec.from_array(array)

    observed = ArraySpec(
        **json.loads(store[".zarray"].to_bytes()),
        attributes=json.loads(store[".zattrs"].to_bytes()),
    ).model_dump()
    expected = spec.model_dump()
    assert expected == observed


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
@pytest.mark.parametrize("filters", ["omit", "auto", None])
@pytest.mark.parametrize("dimension_separator", ["omit", "auto", "."])
@pytest.mark.parametrize("compressor", ["omit", "auto", {"id": "gzip"}])
def test_array_spec_from_array(
    *,
    array: npt.NDArray[Any],
    chunks: str | tuple[int, ...],
    attributes: str | dict[str, object],
    fill_value: object,
    order: str,
    filters: str | list[object],
    dimension_separator: str,
    compressor: str | dict[str, object],
) -> None:
    auto_options = ("omit", "auto")
    kwargs_out: ArrayV2Config = {}

    kwargs_out["chunks"] = chunks
    kwargs_out["attributes"] = attributes
    kwargs_out["fill_value"] = fill_value
    kwargs_out["order"] = order
    kwargs_out["filters"] = filters
    kwargs_out["dimension_separator"] = dimension_separator
    kwargs_out["compressor"] = compressor

    # remove all the keyword arguments that should be defaulted
    kwargs_out = dict(filter(lambda kvp: kvp[1] != "omit", kwargs_out.items()))

    spec = ArraySpec.from_array(array, **kwargs_out)
    # arrayspec should round-trip from_array with no arguments
    assert spec.from_arraylike(spec) == spec
    meta_dict = zarrify_array_v2(array)

    assert spec.dtype == array.dtype.str
    assert np.dtype(spec.dtype) == array.dtype

    assert spec.shape == array.shape

    if chunks in auto_options:
        assert spec.chunks == meta_dict["chunks"]
    else:
        assert spec.chunks == chunks

    if attributes in auto_options:
        assert spec.attributes == meta_dict["attributes"]
    else:
        assert spec.attributes == attributes

    if fill_value in auto_options:
        assert spec.fill_value == meta_dict["fill_value"]
    else:
        assert spec.fill_value == fill_value

    if order in auto_options:
        assert spec.order == meta_dict["order"]
    else:
        assert spec.order == order

    if filters in auto_options:
        assert spec.filters == meta_dict["filters"]
    else:
        assert spec.filters == filters

    if dimension_separator in auto_options:
        assert spec.dimension_separator == meta_dict["dimension_separator"]
    else:
        assert spec.dimension_separator == dimension_separator

    if compressor in auto_options:
        assert spec.compressor == meta_dict["compressor"]
    else:
        assert spec.compressor == compressor


@pytest.mark.parametrize("chunks", [(1,), (1, 2), ((1, 2, 3))])
@pytest.mark.parametrize("dtype", ["bool", "uint8", np.dtype("uint8"), "float64"])
@pytest.mark.parametrize("dimension_separator", [".", "/"])
@pytest.mark.parametrize(
    "compressor",
    [
        {"id": "lzma", "format": 1, "check": -1, "preset": None, "filters": None},
        {"id": "gzip", "level": 1},
    ],
)
@pytest.mark.parametrize(
    "filters", [None, ("delta",), ("scale_offset",), ("delta", "scale_offset")]
)
def test_serialize_deserialize_groupspec(
    chunks: tuple[int, ...],
    memory_order: ArrayMemoryOrder,
    dtype: str,
    dimension_separator: Literal[".", "/"],
    compressor: Any,
    filters: tuple[str, ...] | None,
) -> None:
    import numcodecs
    from numcodecs.abc import Codec  # noqa: TC002

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

    store = {}

    spec = GroupSpec[RootAttrs, ArraySpec | SubGroup](
        attributes=RootAttrs(foo=10, bar=[0, 1, 2]),
        members={
            "s0": ArraySpec[ArrayAttrs](
                shape=(10,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
                compressor=compressor,
                order=memory_order,
                dimension_separator=dimension_separator,
                attributes=ArrayAttrs(scale=[1.0]),
            ),
            "s1": ArraySpec[ArrayAttrs](
                shape=(5,) * len(chunks),
                chunks=chunks,
                dtype=dtype,
                filters=_filters,
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
    assert to_zarr(spec, store, "/group_a") == group

    # check that we can't call to_zarr targeting the original group with a different spec
    spec_2 = spec.model_copy(update={"attributes": RootAttrs(foo=99, bar=[0, 1, 2])})
    with pytest.raises(ContainsGroupError):
        _ = to_zarr(spec_2, store, "/group_a")

    # check that we can't call to_zarr with the original spec if the group has changed
    group.attrs.put({"foo": 100})
    with pytest.raises(ContainsGroupError):
        _ = to_zarr(spec, store, "/group_a")

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
    import zarr

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

    store = zarr.MemoryStore

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

    store = zarr.MemoryStore()
    groupAMat = specA.to_zarr(store, path="group_a")
    groupBMat = specB.to_zarr(store, path="group_b")

    GroupA.from_zarr(groupAMat)
    GroupB.from_zarr(groupBMat)

    ArrayA.from_store(groupAMat["a"])
    ArrayB.from_store(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayA.from_store(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayB.from_store(groupAMat["a"])

    with pytest.raises(ValidationError):
        GroupB.from_zarr(groupAMat)

    with pytest.raises(ValidationError):
        GroupA.from_zarr(groupBMat)


@pytest.mark.parametrize("shape", [(1,), (2, 2), (3, 4, 5)])
@pytest.mark.parametrize("dtype", [None, "uint8", "float32"])
def test_from_array(shape: tuple[int, ...], dtype: str | None) -> None:
    template = np.zeros(shape=shape, dtype=dtype)
    spec = ArraySpec.from_array(template)

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
    with pytest.raises(ValidationError, match='Strings containing "/" are invalid.'):
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
                "": GroupSpec(attributes={"foo": 10}, members={}),
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
                "": GroupSpec(attributes={}, members={}),
                "/a": GroupSpec(attributes={"foo": 10}, members={}),
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
    a = ArraySpec.from_array(np.arange(10))
    assert a.like(a)

    b = a.model_copy(update={"dtype": "uint8"})
    assert not a.like(b)
    assert a.like(b, exclude={"dtype"})
    assert a.like(b, include={"shape"})

    c = a.model_copy(update={"shape": (100, 100)})
    assert not a.like(c)
    assert a.like(c, exclude={"shape"})
    assert a.like(c, include={"dtype"})


# todo: parametrize
def test_group_like() -> None:
    tree = {
        "": GroupSpec(attributes={"path": ""}, members=None),
        "/a": GroupSpec(attributes={"path": "/a"}, members=None),
        "/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/b"}),
        "/a/b": ArraySpec.from_array(np.arange(10), attributes={"path": "/a/b"}),
    }
    group = GroupSpec.from_flat(tree)
    assert group.like(group)
    assert not group.like(group.model_copy(update={"attributes": None}))
    assert group.like(group.model_copy(update={"attributes": None}), exclude={"attributes"})
    assert group.like(group.model_copy(update={"attributes": None}), include={"members"})


# todo: parametrize
def test_from_zarr_depth() -> None:
    tree = {
        "": GroupSpec(members=None, attributes={"level": 0, "type": "group"}),
        "/1": GroupSpec(members=None, attributes={"level": 1, "type": "group"}),
        "/1/2": GroupSpec(members=None, attributes={"level": 2, "type": "group"}),
        "/1/2/1": GroupSpec(members=None, attributes={"level": 3, "type": "group"}),
        "/1/2/2": ArraySpec.from_array(np.arange(20), attributes={"level": 3, "type": "array"}),
    }
    import zarr

    store = zarr.MemoryStore()
    group_out = GroupSpec.from_flat(tree).to_zarr(store, path="test")
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
