from __future__ import annotations

import importlib
import importlib.util
import json
import re
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

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

from ..conftest import DTYPE_EXAMPLES_V3, DTypeExample

ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None


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
    assert json_eq(arr_spec.model_dump(), json.loads(store["zarr.json"].to_bytes()))


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


def get_flat_example() -> tuple[dict[str, ArraySpec | GroupSpec], GroupSpec]:
    """
    Get example data for testing to_flat and from_flat.

    The returned value is a tuple with two elements: a flattened dict representation of a hierarchy,
    and the root group, with all of its members (i.e., the non-flat version of that hierarchy).
    """
    named_nodes: tuple[ArraySpec | BaseGroupSpec, ...] = (
        BaseGroupSpec(attributes={"name": ""}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/a1"}),
        BaseGroupSpec(attributes={"name": "/g1"}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/a2"}),
        BaseGroupSpec(attributes={"name": "/g1/g2"}),
        ArraySpec.from_array(np.arange(10), attributes={"name": "/g1/g2/a3"}),
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
        assert GroupSpec.from_flat(members_flat).attributes == root.attributes

    @staticmethod
    def test_from_zarr_depth() -> None:
        zarr = pytest.importorskip("zarr")
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
