"""
Testts for pydantic_zarr.v2.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping  # noqa: TC003
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import dask.array as da
import pytest
import xarray as xr
from pydantic import ValidationError

from pydantic_zarr.core import tuplify_json
from pydantic_zarr.experimental.core import json_eq

from ..conftest import DTYPE_EXAMPLES_V2, ZARR_PYTHON_VERSION, DTypeExample

if TYPE_CHECKING:
    from numcodecs.abc import Codec

import numpy as np
import numpy.typing as npt
from packaging.version import Version

from pydantic_zarr.experimental.v2 import (
    DIMENSION_SEPARATOR,
    MEMORY_ORDER,
    ArraySpec,
    BaseGroupSpec,
    CodecDict,
    DimensionSeparator,
    GroupSpec,
    MemoryOrder,
    auto_attributes,
    auto_chunks,
    auto_compresser,
    auto_dimension_separator,
    auto_fill_value,
    auto_filters,
    auto_order,
    from_flat,
    from_zarr,
    to_flat,
    to_zarr,
)

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

try:
    import numcodecs
except ImportError:
    numcodecs = None

with suppress(ImportError):
    from zarr.errors import ContainsArrayError, ContainsGroupError


@pytest.mark.parametrize(("chunks", "shape"), [((1,), (10,)), ((1, 2, 3), (4, 5, 6))])
@pytest.mark.parametrize("dtype", ["bool", "float64", "|u1", np.float32])
@pytest.mark.parametrize("compressor", [None, {"id": "gzip", "level": 1}])
@pytest.mark.parametrize(
    "filters",
    [
        None,
        (),
        ({"id": "delta", "dtype": "uint8"},),
        ({"id": "delta", "dtype": "uint8"}, {"id": "gzip", "level": 1}),
    ],
)
@pytest.mark.parametrize("dimension_separator", DIMENSION_SEPARATOR)
@pytest.mark.parametrize("memory_order", MEMORY_ORDER)
@pytest.mark.parametrize("attributes", [{}, {"a": [100]}, {"b": ("e", "f")}])
def test_array_spec(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    memory_order: MemoryOrder,
    dtype: str,
    dimension_separator: DimensionSeparator,
    compressor: str | CodecDict,
    filters: tuple[str, ...] | None,
    attributes: dict[str, object],
) -> None:
    zarr = pytest.importorskip("zarr")
    import numcodecs

    if filters is not None:
        _filters = tuple(numcodecs.get_codec(f) for f in filters)
    else:
        _filters = None
    store = {}

    array = zarr.create_array(
        shape=shape,
        store=store,
        chunks=chunks,
        dtype=dtype,
        order=memory_order,
        chunk_key_encoding={"name": "v2", "configuration": {"separator": dimension_separator}},
        compressors=compressor,
        filters=_filters,
        zarr_format=2,
        attributes=attributes,
    )

    spec = ArraySpec.from_zarr(array)

    assert json_eq(
        spec.model_dump(), {**json.loads(store[".zarray"].to_bytes()), "attributes": attributes}
    )


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("path", ["", "foo"])
@pytest.mark.parametrize("config", [None, {}, {"order": "C", "write_empty_chunks": True}])
def test_arrayspec_to_zarr(overwrite: bool, path: str, config: dict[str, object] | None) -> None:
    """
    Test serializing an arrayspec to zarr and back again
    """
    zarr = pytest.importorskip("zarr")
    from zarr.core.array_spec import ArrayConfig

    spec = ArraySpec(
        shape=(10,),
        dtype="uint8",
        chunks=(1,),
        attributes={"a": 10},
    )

    # test serialization
    store = zarr.storage.MemoryStore()
    stored = spec.to_zarr(store, path=path, config=config)  # type: ignore[arg-type]

    if config not in (None, {}):
        assert stored._async_array._config == ArrayConfig(
            order=config["order"], write_empty_chunks=config["write_empty_chunks"]
        )

    assert json_eq(ArraySpec.from_zarr(stored).model_dump(), spec.model_dump())

    # test that to_zarr is idempotent when the arrays match
    assert json_eq(spec.to_zarr(store, path=path).metadata.to_dict(), stored.metadata.to_dict())

    # test that to_zarr raises if the extant array is different
    # unless overwrite is True
    spec_2 = spec.model_copy(update={"attributes": {"baz": 10}})
    if not overwrite:
        with pytest.raises(ContainsArrayError):
            spec_2.to_zarr(store, path=path, overwrite=overwrite)
    else:
        arr_2 = spec_2.to_zarr(store, path=path, overwrite=overwrite)
        assert json_eq(arr_2.attrs.asdict(), spec_2.attributes)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((100), dtype="uint8"),
        xr.DataArray(np.arange(10), attrs={"foo": 10}),
        xr.DataArray(da.arange(10), attrs={"foo": 10}),
        da.arange(10),
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
    kwargs_out: dict[str, object] = {}

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
    assert spec.from_array(spec) == spec

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


def test_serialize_deserialize_groupspec() -> None:
    zarr = pytest.importorskip("zarr")

    class RootAttrs(TypedDict):
        foo: int
        bar: list[int]

    class SubGroupAttrs(TypedDict):
        a: str
        b: float

    class SubGroup(GroupSpec):
        attributes: SubGroupAttrs

    class ArrayAttrs(TypedDict):
        scale: list[float]

    class MemberArray(ArraySpec):
        attributes: ArrayAttrs

    class RootGroup(GroupSpec):
        attributes: RootAttrs
        members: Mapping[str, MemberArray | SubGroup]

    store = zarr.storage.MemoryStore()

    spec = RootGroup(
        attributes=RootAttrs(foo=10, bar=[0, 1, 2]),
        members={
            "s0": MemberArray(
                shape=(10,),
                chunks=(1,),
                dtype="uint8",
                filters=None,
                compressor=None,
                order="C",
                dimension_separator="/",
                attributes=ArrayAttrs(scale=[1.0]),
            ),
            "s1": MemberArray(
                shape=(10,),
                chunks=(1,),
                dtype="uint8",
                filters=None,
                compressor=None,
                order="C",
                dimension_separator="/",
                attributes=ArrayAttrs(scale=[2.0]),
            ),
            "subgroup": SubGroup(attributes=SubGroupAttrs(a="foo", b=1.0), members={}),
        },
    )
    # check that the model round-trips dict representation
    assert spec.model_dump() == GroupSpec(**spec.model_dump()).model_dump()

    # materialize a zarr group, based on the spec
    group = to_zarr(spec, store, "/group_a")

    # parse the spec from that group
    observed = from_zarr(group)
    assert json_eq(observed.model_dump(), spec.model_dump())

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

    class ArrayA(ArraySpec):
        attributes: ArrayAttrsA

    class ArrayB(ArraySpec):
        attributes: ArrayAttrsB

    class GroupA(GroupSpec):
        attributes: GroupAttrsA
        members: Mapping[str, ArrayA]

    class GroupB(GroupSpec):
        attributes: GroupAttrsB
        members: Mapping[str, ArrayB]

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

    ArrayA.from_zarr(groupAMat["a"])
    ArrayB.from_zarr(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayA.from_zarr(groupBMat["a"])

    with pytest.raises(ValidationError):
        ArrayB.from_zarr(groupAMat["a"])

    with pytest.raises(ValidationError):
        GroupB.from_zarr(groupAMat)

    with pytest.raises(ValidationError):
        GroupA.from_zarr(groupBMat)


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
    arr = ArraySpec(shape=(1,), dtype="uint8", chunks=(1,), attributes={})
    store = zarr.storage.MemoryStore()
    arr_stored = arr.to_zarr(store, path="arr")
    assert arr.like(arr_stored)

    dissimilar_arr = arr.model_copy(update={"attributes": {"a": 10}}).to_zarr(store, path="arr_2")
    assert not arr.like(dissimilar_arr)
    assert arr.like(dissimilar_arr, exclude={"attributes"})


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
    assert not group.like(group.model_copy(update={"attributes": {}}))
    assert group.like(group.model_copy(update={"attributes": {}}), exclude={"attributes"})
    assert group.like(group.model_copy(update={"attributes": {}}), include={"members"})


# todo: parametrize
def test_from_zarr_depth() -> None:
    zarr = pytest.importorskip("zarr")
    tree: dict[str, BaseGroupSpec | ArraySpec] = {
        "": BaseGroupSpec(attributes={"level": 0, "type": "group"}),
        "/1": BaseGroupSpec(attributes={"level": 1, "type": "group"}),
        "/1/2": BaseGroupSpec(attributes={"level": 2, "type": "group"}),
        "/1/2/1": BaseGroupSpec(attributes={"level": 3, "type": "group"}),
        "/1/2/2": ArraySpec.from_array(np.arange(20), attributes={"level": 3, "type": "array"}),
    }

    store = zarr.storage.MemoryStore()
    group_out = GroupSpec.from_flat(tree).to_zarr(store, path="test")
    group_in_0 = GroupSpec.from_zarr(group_out, depth=0)  # type: ignore[var-annotated]
    assert group_in_0.attributes == tree[""].attributes

    group_in_1 = GroupSpec.from_zarr(group_out, depth=1)  # type: ignore[var-annotated]
    assert group_in_1.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_1.members["1"].attributes == tree["/1"].attributes

    group_in_2 = GroupSpec.from_zarr(group_out, depth=2)  # type: ignore[var-annotated]
    assert group_in_2.members["1"].members["2"].attributes == tree["/1/2"].attributes
    assert group_in_2.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_2.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]

    group_in_3 = GroupSpec.from_zarr(group_out, depth=3)  # type: ignore[var-annotated]
    assert group_in_3.members["1"].members["2"].members["1"].attributes == tree["/1/2/1"].attributes
    assert group_in_3.attributes == tree[""].attributes  # type: ignore[attr-defined]
    assert group_in_3.members["1"].attributes == tree["/1"].attributes  # type: ignore[attr-defined]
    assert group_in_3.members["1"].members["2"].attributes == tree["/1/2"].attributes  # type: ignore[attr-defined]


@pytest.mark.parametrize(("dtype_example"), DTYPE_EXAMPLES_V2, ids=str)
def test_arrayspec_from_zarr(dtype_example: DTypeExample) -> None:
    """
    Test that deserializing an ArraySpec from a zarr python store works as expected.
    """
    zarr = pytest.importorskip("zarr")
    store = {}
    data_type = dtype_example.name
    if ZARR_PYTHON_VERSION >= Version("3.1.0") and data_type == "|O":
        pytest.skip(reason="Data type inference with an object dtype will fail in zarr>=3.1.0")
    arr = zarr.create_array(store=store, shape=(10,), dtype=data_type, zarr_format=2)

    arr_spec = ArraySpec.from_zarr(arr)

    observed = {"attributes": arr.attrs.asdict()} | json.loads(
        store[".zarray"].to_bytes(), object_hook=tuplify_json
    )
    if observed["filters"] is not None:
        observed["filters"] = list(observed["filters"])
    # this covers the case of the structured data type, which would otherwise be deserialized as a
    # tuple of tuples, but is stored on the arrayspec as a list of tuples.
    if isinstance(observed["dtype"], tuple):
        observed["dtype"] = list(observed["dtype"])

    assert json_eq(arr_spec.model_dump(), observed)


def test_mix_v3_v2_fails() -> None:
    from pydantic_zarr.v3 import ArraySpec as ArraySpecv3

    members_flat = {"/a": ArraySpecv3.from_array(np.ones(1))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value at '/a' is not a v2 ArraySpec or GroupSpec (got type(value)=<class 'pydantic_zarr.v3.ArraySpec'>)"
        ),
    ):
        GroupSpec.from_flat(members_flat)  # type: ignore[arg-type]
