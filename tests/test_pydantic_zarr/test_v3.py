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
from pydantic import BaseModel, ValidationError

from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import (
    _SINGLE_BYTE_DATA_TYPES,
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
    is_array_spec,
    is_group_spec,
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
        codecs=[
            NamedConfig(name="bytes", configuration={"endian": "little"}),
            NamedConfig(name="gzip", configuration={"level": 1}),
        ],
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
    expected = json.loads(store["zarr.json"].to_bytes(), object_hook=tuplify_json)
    # The spec canonicalizes two codec spellings zarr-python writes: an empty
    # configuration mapping (any codec), and the no-op endian configuration on
    # the bytes codec for single-byte data types (zarr-python 3.0 wrote it).
    # The rewrites mirror the validator's exact conditions, so any other
    # wrongly dropped configuration would still fail the comparison.
    expected["codecs"] = tuple(
        {k: v for k, v in codec.items() if k != "configuration"}
        if isinstance(codec, dict) and codec.get("configuration") == {}
        else codec
        for codec in expected["codecs"]
    )
    if data_type in _SINGLE_BYTE_DATA_TYPES:
        expected["codecs"] = tuple(
            {"name": "bytes"}
            if (
                isinstance(codec, dict)
                and codec.get("name") == "bytes"
                and isinstance(codec.get("configuration"), dict)
                and set(codec["configuration"]) == {"endian"}
            )
            else codec
            for codec in expected["codecs"]
        )
    assert arr_spec.model_dump() == expected


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
        assert is_group_spec(sub_2_1)
        assert sub_2_1.members is not None
        assert sub_2_1.members["2"] == tree["/1/2"]
        assert group_in_2.attributes == tree[""].attributes
        assert sub_2_1.attributes == tree["/1"].attributes

        group_in_3 = GroupSpec.from_zarr(group_out, depth=3)  # type: ignore[var-annotated]
        assert group_in_3.members is not None
        sub_3_1 = group_in_3.members["1"]
        assert is_group_spec(sub_3_1)
        assert sub_3_1.members is not None
        sub_3_2 = sub_3_1.members["2"]
        assert is_group_spec(sub_3_2)
        assert sub_3_2.members is not None
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
        codecs=[
            NamedConfig(name="bytes", configuration={"endian": "little"}),
            NamedConfig(name="gzip", configuration={"level": 1}),
        ],
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


@pytest.mark.parametrize(
    "overrides",
    [
        # a plain core document
        {},
        # integer dtype with an integer fill value
        {"data_type": "int8", "fill_value": 0},
        # a non-finite float fill value given as a float is canonicalized and accepted
        {"fill_value": float("nan")},
        # a hexadecimal float fill value
        {"fill_value": "0x7fc00000"},
        # a datetime dtype in its named-configuration form, with an integer fill value
        {
            "data_type": {
                "name": "numpy.datetime64",
                "configuration": {"unit": "ns", "scale_factor": 1},
            },
            "fill_value": -(2**63),
        },
        # a sharding codec holding a nested pipeline
        {
            "codecs": (
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": (1, 1),
                        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                        "index_codecs": (
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ),
                        "index_location": "end",
                    },
                },
            )
        },
        # a codec name nothing in the scope claims is left unjudged
        {
            "codecs": (
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "a-codec-nobody-claims", "configuration": {"whatever": True}},
            )
        },
    ],
    ids=str,
)
def test_metadata_document_validation_accepts(overrides: dict[str, object]) -> None:
    """
    ArraySpec accepts specs whose fields assemble into a valid Zarr V3 metadata
    document, across data types, fill value spellings, and codec pipelines.
    """
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs(**overrides))
    # the document round-trips through model validation unchanged
    assert ArraySpec(**spec.model_dump()) == spec


def test_metadata_document_validation_fill_value_dtype_mismatch() -> None:
    """A fill value that the named data type refuses is a validation error."""
    with pytest.raises(ValidationError, match="fill_value"):
        ArraySpec(**_v3_array_kwargs(data_type="int8", fill_value="NaN"))


def test_metadata_document_validation_chunk_grid_rank_mismatch() -> None:
    """A chunk shape with the wrong number of dimensions is a validation error."""
    with pytest.raises(ValidationError, match="chunk length per dimension"):
        ArraySpec(
            **_v3_array_kwargs(
                chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2, 2, 2)}}
            )
        )


def test_metadata_document_validation_pipeline_shape() -> None:
    """A codec pipeline with two array -> bytes codecs is a validation error."""
    with pytest.raises(ValidationError, match="array -> bytes"):
        ArraySpec(
            **_v3_array_kwargs(
                codecs=(
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "bytes", "configuration": {"endian": "little"}},
                )
            )
        )


def test_metadata_document_validation_invalid_extension_name() -> None:
    """A codec whose name is not a legal extension name is a validation error."""
    with pytest.raises(ValidationError, match="extension name"):
        ArraySpec(
            **_v3_array_kwargs(
                codecs=(
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "Not/A/Name"},
                )
            )
        )


def test_metadata_document_validation_claimed_codec_bad_config() -> None:
    """A codec the scope claims, with a configuration its definition refuses, is refused."""
    with pytest.raises(ValidationError, match=r"codecs\.1\.configuration\.level"):
        ArraySpec(
            **_v3_array_kwargs(
                codecs=(
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": "very"}},
                )
            )
        )


def test_metadata_document_validation_scope_from_context() -> None:
    """
    The validation scope comes from the pydantic validation context: in a scope
    that claims a codec name, an invalid configuration is refused; in one that
    does not, the same codec is left unjudged and the spec is accepted.
    """
    from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS

    from pydantic_zarr._metadata import CONTEXT_KEY

    document = {
        "zarr_format": 3,
        "node_type": "array",
        **_v3_array_kwargs(
            codecs=(
                {"name": "bytes", "configuration": {"endian": "little"}},
                # claimed by the zarr-extensions scope, with an unknown config key
                {"name": "scale_offset", "configuration": {"bogus": 1}},
            )
        ),
    }
    # CORE does not claim scale_offset, so the field is unjudged and the spec valid
    spec: AnyArraySpec = ArraySpec.model_validate(document, context=CORE)
    assert spec.codecs[-1]["name"] == "scale_offset"  # type: ignore[index]
    # the same scope travels under CONTEXT_KEY in a mapping context
    assert ArraySpec.model_validate(document, context={CONTEXT_KEY: CORE}) == spec
    # CORE_AND_EXTENSIONS claims it and refuses the configuration
    with pytest.raises(ValidationError, match="scale_offset|bogus"):
        ArraySpec.model_validate(document, context={CONTEXT_KEY: CORE_AND_EXTENSIONS})


def test_metadata_document_validation_context_key_type_error() -> None:
    """A validation context holding a non-scope under CONTEXT_KEY is a TypeError."""
    from pydantic_zarr._metadata import CONTEXT_KEY

    document = {"zarr_format": 3, "node_type": "array", **_v3_array_kwargs()}
    with pytest.raises(TypeError, match=CONTEXT_KEY):
        ArraySpec.model_validate(document, context={CONTEXT_KEY: "not a scope"})


def test_like_compares_documents_canonically() -> None:
    """
    With no include/exclude filters, `like` compares the metadata documents by
    meaning: two spellings of one float32 NaN fill value are alike, and specs
    that differ only in attributes are not.
    """
    a: AnyArraySpec = ArraySpec(**_v3_array_kwargs(fill_value="NaN"))
    b: AnyArraySpec = ArraySpec(**_v3_array_kwargs(fill_value="0x7fc00000"))
    assert a.like(b)
    assert b.like(a)
    # attributes still participate in the comparison
    assert not a.like(ArraySpec(**_v3_array_kwargs(fill_value="NaN", attributes={"x": 1})))
    # filtered comparisons keep the literal field semantics
    assert not a.like(b, exclude={"attributes"})


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.dtype("bool"), "bool"),
        (
            np.dtype("datetime64[ns]"),
            {"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 1}},
        ),
        (
            np.dtype("timedelta64[s]"),
            {"name": "numpy.timedelta64", "configuration": {"unit": "s", "scale_factor": 1}},
        ),
    ],
    ids=str,
)
def test_parse_dtype_v3_named_configs(
    dtype: np.dtype[Any], expected: str | dict[str, object]
) -> None:
    """The bool and numpy time dtypes map to their Zarr V3 forms."""
    assert parse_dtype_v3(dtype) == expected


def test_from_array_time_dtype() -> None:
    """`from_array` handles numpy time dtypes, emitting the named-configuration form."""
    spec: AnyArraySpec = ArraySpec.from_array(np.zeros((3,), dtype="datetime64[ns]"))
    assert spec.data_type == {
        "name": "numpy.datetime64",
        "configuration": {"unit": "ns", "scale_factor": 1},
    }
    assert spec.fill_value == -(2**63)


def test_from_array_bool_dtype() -> None:
    """`from_array` handles bool arrays: previously auto_fill_value checked the dtype
    char '?' where numpy reports kind 'b', so bool arrays always raised."""
    spec: AnyArraySpec = ArraySpec.from_array(np.zeros((3,), dtype=bool))
    assert spec.data_type == "bool"
    assert spec.fill_value is False


def test_group_member_names_accepted() -> None:
    """GroupSpec accepts member names the Zarr V3 node-name rules allow."""
    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    names = ("plain", "with.dots", "with spaces", "ünïcode", "_single_underscore", "0", "a..b")
    group: AnyGroupSpec = GroupSpec(attributes={}, members=dict.fromkeys(names, array))
    assert tuple(group.members or {}) == names


@pytest.mark.parametrize(
    "invalid_name",
    ["", "a/b", ".", "..", "...", "__reserved"],
    ids=repr,
)
def test_group_member_names_rejected(invalid_name: str) -> None:
    """
    GroupSpec rejects member names the Zarr V3 node-name rules refuse: the
    empty string (it names the root node), names holding "/", names composed
    of periods alone, and names starting with the reserved "__" prefix.
    """
    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    with pytest.raises(ValidationError, match="member name"):
        GroupSpec(attributes={}, members={invalid_name: array})


def test_typevar_defaults() -> None:
    """
    The generic type parameters carry defaults, so the bare class names are
    usable without parametrization, parametrized forms keep working, and a
    subclass can narrow a field annotation without generics.
    """
    from pydantic import BaseModel

    array = ArraySpec(**_v3_array_kwargs())
    group = GroupSpec(attributes={"a": 1}, members={"x": array})
    assert group.members is not None
    assert tuple(group.members) == ("x",)

    class MyAttrs(BaseModel):
        scale: float

    typed = ArraySpec[MyAttrs](**_v3_array_kwargs(attributes=MyAttrs(scale=2.0)))
    assert typed.attributes.scale == 2.0

    class Narrow(ArraySpec):
        attributes: MyAttrs

    assert Narrow(**_v3_array_kwargs(attributes={"scale": 3.0})).attributes.scale == 3.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attributes", {"new": True}),
        ("shape", (8, 8)),
        # float64 keeps the base spec's "NaN" fill value valid for the new dtype
        ("data_type", "float64"),
        ("chunk_grid", {"name": "regular", "configuration": {"chunk_shape": (4, 4)}}),
        ("chunk_key_encoding", {"name": "v2", "configuration": {"separator": "."}}),
        ("fill_value", "Infinity"),
        # the endian configuration keeps the bytes codec valid for a multi-byte dtype
        ("codecs", ({"name": "bytes", "configuration": {"endian": "big"}},)),
        ("storage_transformers", ()),
        ("dimension_names", ("y", "x")),
    ],
)
def test_update(field: str, value: object) -> None:
    """`update` returns a revalidated copy with the new field, everything else unchanged."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    built = spec.update(**{field: value})  # type: ignore[arg-type]
    assert getattr(built, field) == value
    assert built.model_dump(exclude={field}) == spec.model_dump(exclude={field})


def test_update_atomic() -> None:
    """Correlated fields change together in one validation pass."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    built = spec.update(
        shape=(8, 8, 8),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (2, 2, 2)}},
        dimension_names=("z", "y", "x"),
    )
    assert built.shape == (8, 8, 8)
    assert built.dimension_names == ("z", "y", "x")


def test_update_revalidates() -> None:
    """`update` revalidates: an invalid change is refused."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs(dimension_names=("a", "b")))
    with pytest.raises(ValidationError, match="dimension names"):
        spec.update(shape=(1, 2, 3))


def test_update_unknown_field() -> None:
    """A change naming a field the model does not have is refused."""
    spec: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    with pytest.raises(ValidationError, match="bogus"):
        spec.update(bogus=1)  # type: ignore[call-arg]


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
    assert base.update(attributes={"a": 2}).attributes == {"a": 2}


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


def test_narrowing_helpers() -> None:
    """
    `is_array_spec` and `is_group_spec` tell flat-hierarchy values apart, and
    `is_base_group_spec` holds for any group model, members or not.
    """
    from pydantic_zarr.v3 import is_base_group_spec

    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    group: AnyGroupSpec = GroupSpec(attributes={}, members={"a": array})
    base = BaseGroupSpec(attributes={})
    assert is_array_spec(array)
    assert not is_array_spec(group)
    assert is_group_spec(group)
    assert not is_group_spec(base)
    assert is_base_group_spec(base)
    assert is_base_group_spec(group)
    assert not is_base_group_spec(array)


def test_typed_members() -> None:
    """
    A subclass can declare its members as a closed `TypedDict` of specs, so
    the shape of the hierarchy is enforced: member values are validated into
    specs, an undeclared member name is refused, and a missing required
    member is refused.
    """
    from typing_extensions import TypedDict

    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())

    class Members(TypedDict, closed=True):
        r10m: ArraySpec
        r20m: ArraySpec

    class Measurements(GroupSpec):
        members: Members  # type: ignore[assignment]

    group = Measurements(attributes={}, members={"r10m": array, "r20m": array.model_dump()})
    assert isinstance(group.members["r20m"], ArraySpec)
    with pytest.raises(ValidationError, match="extra"):
        Measurements(attributes={}, members={"r10m": array, "r20m": array, "extra": array})
    with pytest.raises(ValidationError, match="r20m"):
        Measurements(attributes={}, members={"r10m": array})


def test_member_name_validation_survives_narrowing() -> None:
    """
    Member names are validated by a model validator, so a subclass that
    narrows the `members` annotation cannot drop the node-name rules.
    """

    class Narrowed(GroupSpec):
        members: dict[str, ArraySpec]  # type: ignore[assignment]

    array: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    with pytest.raises(ValidationError, match="member name"):
        Narrowed(attributes={}, members={"__reserved": array})


def test_group_like_canonical() -> None:
    """
    With no `include`/`exclude` filters, `GroupSpec.like` compares each member
    by that member's own `like`, so member arrays whose metadata documents
    differ only in spelling (a fill value written "NaN" against its
    hexadecimal byte form) are alike, at any depth, exactly as they are when
    compared directly. With filters, the comparison is the raw field
    comparison, where the spellings differ.
    """
    a_nan: AnyArraySpec = ArraySpec(**_v3_array_kwargs(fill_value="NaN"))
    a_hex: AnyArraySpec = ArraySpec(**_v3_array_kwargs(fill_value="0x7fc00000"))
    assert a_nan.like(a_hex)
    g_nan: AnyGroupSpec = GroupSpec(attributes={}, members={"x": a_nan})
    g_hex: AnyGroupSpec = GroupSpec(attributes={}, members={"x": a_hex})
    assert g_nan.like(g_hex)
    assert g_hex.like(g_nan)
    nested_nan: AnyGroupSpec = GroupSpec(attributes={}, members={"g": g_nan})
    nested_hex: AnyGroupSpec = GroupSpec(attributes={}, members={"g": g_hex})
    assert nested_nan.like(nested_hex)
    assert g_nan.like(g_hex, include={"attributes"})
    assert not g_nan.like(g_hex, include={"members"})


@pytest.mark.parametrize(
    "other_members",
    [
        {},
        None,
        {"y": ArraySpec(**_v3_array_kwargs())},
        {"x": GroupSpec(attributes={})},
        {"x": ArraySpec(**_v3_array_kwargs(shape=(8, 8)))},
    ],
    ids=["no_members", "none_members", "renamed_member", "kind_mismatch", "unalike_member"],
)
def test_group_like_canonical_unalike(other_members: dict[str, object] | None) -> None:
    """
    The canonical group comparison still distinguishes groups that differ in
    substance: a missing, renamed or unalike member, members of different node
    kinds, and unknown (`None`) against empty members are all not alike.
    """
    group: AnyGroupSpec = GroupSpec(attributes={}, members={"x": ArraySpec(**_v3_array_kwargs())})
    other = GroupSpec.model_validate(
        {"zarr_format": 3, "node_type": "group", "attributes": {}, "members": other_members}
    )
    assert not group.like(other)
    assert not other.like(group)


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_from_array_zarr_array_fidelity() -> None:
    """
    `from_array` on a `zarr.Array` reads the array's whole stored metadata
    document, so the chunk grid, chunk key encoding and attributes survive,
    and an explicit (non-"auto") keyword argument overrides the stored value.
    """
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(64, 64),
        dtype="float32",
        chunks=(16, 16),
        chunk_key_encoding={"name": "v2", "separator": "."},
        zarr_format=3,
    )
    arr.attrs.put({"a": 1})
    spec: AnyArraySpec = ArraySpec.from_array(arr)
    assert spec == ArraySpec.from_zarr(arr)
    assert spec.chunk_grid == {"name": "regular", "configuration": {"chunk_shape": (16, 16)}}
    assert spec.chunk_key_encoding == {"name": "v2", "configuration": {"separator": "."}}
    assert spec.attributes == {"a": 1}
    overridden: AnyArraySpec = ArraySpec.from_array(arr, attributes={"b": 2}, fill_value=1.5)
    assert overridden.attributes == {"b": 2}
    assert overridden.fill_value == 1.5
    assert overridden.chunk_grid == spec.chunk_grid


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_from_array_zarr_v2_array_rejected() -> None:
    """A Zarr V2 `zarr.Array` is refused by the v3 `from_array`."""
    import zarr

    arr = zarr.create_array(store={}, shape=(4,), dtype="uint8", chunks=(2,), zarr_format=2)
    with pytest.raises(ValueError, match="Only zarr v3 arrays"):
        ArraySpec.from_array(arr)


def test_attributes_default() -> None:
    """
    `attributes` defaults to the empty dict on every node class, and a stored
    metadata document without the `attributes` key (which the Zarr V3 spec
    makes optional) parses.
    """
    array_kwargs = _v3_array_kwargs()
    array_kwargs.pop("attributes")
    assert ArraySpec(**array_kwargs).attributes == {}
    assert GroupSpec().attributes == {}
    assert BaseGroupSpec().attributes == {}
    assert GroupSpec.model_validate({"zarr_format": 3, "node_type": "group"}).attributes == {}
    array_document = {"zarr_format": 3, "node_type": "array", **array_kwargs}
    assert ArraySpec.model_validate(array_document).attributes == {}


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


def test_dimension_names_omitted_when_none() -> None:
    """
    `dimension_names` is omitted from serialization when it is `None` (the
    Zarr V3 spec allows the key to be absent, not null), including for arrays
    nested as members in a group's dump and in JSON serialization, while set
    dimension names are kept at every depth.
    """
    unnamed: AnyArraySpec = ArraySpec(**_v3_array_kwargs())
    named: AnyArraySpec = ArraySpec(**_v3_array_kwargs(dimension_names=("x", "y")))
    assert "dimension_names" not in unnamed.model_dump()
    group: AnyGroupSpec = GroupSpec(members={"unnamed": unnamed, "named": named})
    dump = group.model_dump()
    assert "dimension_names" not in dump["members"]["unnamed"]
    assert dump["members"]["named"]["dimension_names"] == ("x", "y")
    json_dump = json.loads(group.model_dump_json())
    assert "dimension_names" not in json_dump["members"]["unnamed"]
    assert json_dump["members"]["named"]["dimension_names"] == ["x", "y"]


@pytest.mark.parametrize(
    ("data_type", "expected_fill", "expected_codecs"),
    [
        ("uint16", 0, ({"name": "bytes", "configuration": {"endian": "little"}},)),
        ("float32", "NaN", ({"name": "bytes", "configuration": {"endian": "little"}},)),
        ("bool", False, ({"name": "bytes"},)),
        ("uint8", 0, ({"name": "bytes"},)),
        (
            {"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 1}},
            -(2**63),
            ({"name": "bytes", "configuration": {"endian": "little"}},),
        ),
    ],
    ids=["uint16", "float32", "bool", "uint8", "datetime64"],
)
def test_from_shape_and_data_type(
    data_type: str | dict[str, Any], expected_fill: object, expected_codecs: tuple[Any, ...]
) -> None:
    """
    `from_shape_and_data_type` builds a valid `ArraySpec` from a shape and
    data type alone:
    chunks default to the whole shape, the chunk key encoding to the default
    `/` encoding, and fill value and codecs to the data type's derived
    defaults, while explicit keyword arguments and dimension names pass
    through.
    """
    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (10, 12), data_type, dimension_names=("y", "x")
    )
    assert spec.shape == (10, 12)
    assert spec.chunk_grid == {"name": "regular", "configuration": {"chunk_shape": (10, 12)}}
    assert spec.chunk_key_encoding == {"name": "default", "configuration": {"separator": "/"}}
    assert spec.fill_value == expected_fill
    assert spec.codecs == expected_codecs
    assert spec.dimension_names == ("y", "x")
    chunked: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (10, 12),
        data_type,
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (5, 6)}},
    )
    assert chunked.chunk_grid == {"name": "regular", "configuration": {"chunk_shape": (5, 6)}}
    explicit: AnyArraySpec = ArraySpec.from_shape_and_data_type((10, 12), "uint16", fill_value=7)
    assert explicit.fill_value == 7


def test_from_shape_and_data_type_underivable_rejected() -> None:
    """
    `from_shape_and_data_type` refuses to guess a fill value or codecs for a
    data type with
    no numpy equivalent, naming the arguments to pass explicitly.
    """
    with pytest.raises(ValueError, match="fill_value and codecs"):
        ArraySpec.from_shape_and_data_type((2,), "some_extension_dtype")


def test_bytes_codec_canonicalized_for_single_byte_dtypes() -> None:
    """
    For single-byte data types the `bytes` codec's no-op `endian`
    configuration is dropped at validation, matching the spelling zarr-python
    stores: specs constructed with either spelling are equal and alike, and
    multi-byte data types keep their endianness.
    """
    endian_codecs = ({"name": "bytes", "configuration": {"endian": "little"}},)
    for data_type in ("uint8", "int8", "bool"):
        with_endian: AnyArraySpec = ArraySpec.from_shape_and_data_type(
            (4,), data_type, codecs=endian_codecs
        )
        bare: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), data_type)
        assert with_endian.codecs == ({"name": "bytes"},)
        assert with_endian == bare
        assert with_endian.like(bare)
    assert (
        ArraySpec.from_shape_and_data_type((4,), "uint16", codecs=endian_codecs).codecs
        == endian_codecs
    )


def test_resolve_path() -> None:
    """
    `resolve_path` (and `group[path]`) returns the node at a `/`-separated
    relative path, ignoring leading and trailing separators.
    """
    leaf: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "uint8")
    inner: AnyGroupSpec = GroupSpec(attributes={"level": 2}, members={"b": leaf})
    root: AnyGroupSpec = GroupSpec(members={"a": inner, "arr": leaf})
    assert root.resolve_path("arr") == leaf
    assert root.resolve_path("a/b") == leaf
    assert root["a"] == inner
    assert root["/a/b/"] == leaf


@pytest.mark.parametrize(
    ("path", "exception", "match"),
    [
        ("a/zz", KeyError, "No member named 'zz' in a"),
        ("arr/x", KeyError, "arr is not a group"),
        pytest.param("", ValueError, "no member names", id="empty"),
        ("a//b", ValueError, "empty member name"),
    ],
    ids=["missing_member", "through_array", "empty", "empty_segment"],
)
def test_resolve_path_invalid(path: str, exception: type[Exception], match: str) -> None:
    """Failed resolution raises naming the failing segment, or rejects a malformed path."""
    leaf: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "uint8")
    root: AnyGroupSpec = GroupSpec(members={"a": GroupSpec(members={"b": leaf}), "arr": leaf})
    with pytest.raises(exception, match=match):
        root.resolve_path(path)


def test_resolve_path_unknown_members() -> None:
    """Resolution through a group whose members are unknown raises `KeyError`."""
    unknown: AnyGroupSpec = GroupSpec(members=None)
    with pytest.raises(KeyError, match="unknown"):
        unknown.resolve_path("x")


def test_attributes_serialized_by_alias() -> None:
    """
    Dumping a spec serializes attribute models with their aliases, so stored
    metadata keys that are not Python identifiers (`_FillValue`, `image-label`)
    come out under the spec's spelling, and unset `None` fields of an
    `AttributesBase` model are omitted while explicitly set ones are kept.
    """
    from pydantic import Field

    from pydantic_zarr.core import AttributesBase

    class CFAttrs(AttributesBase):
        fill: float | None = Field(default=None, alias="_FillValue")
        units: str | None = None

    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (4,), "float32", attributes=CFAttrs(units="K", _FillValue=1.5)
    )
    assert spec.model_dump()["attributes"] == {"_FillValue": 1.5, "units": "K"}
    unset: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "float32", attributes=CFAttrs())
    assert unset.model_dump()["attributes"] == {}
    explicit: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (4,), "float32", attributes=CFAttrs(units=None)
    )
    assert explicit.model_dump()["attributes"] == {"units": None}


def test_from_shape_and_data_type_guessed_chunks() -> None:
    """
    With no `chunk_grid` override, `from_shape_and_data_type` guesses a chunk
    shape from the array size the way zarr-python does, rather than defaulting
    to one whole-array chunk: a large array gets chunks smaller than its shape, in a sane byte
    range, while a small array keeps its whole shape.
    """
    big: AnyArraySpec = ArraySpec.from_shape_and_data_type((10980, 10980), "uint16")
    chunk_shape = big.chunk_grid["configuration"]["chunk_shape"]
    assert all(c <= s for c, s in zip(chunk_shape, big.shape, strict=True))
    nbytes = 2 * chunk_shape[0] * chunk_shape[1]
    assert 128 * 1024 <= nbytes <= 64 * 1024 * 1024
    small: AnyArraySpec = ArraySpec.from_shape_and_data_type((10, 12), "uint16")
    assert small.chunk_grid["configuration"]["chunk_shape"] == (10, 12)
    # an extension data type has no knowable element size: whole-shape chunks
    ext: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (10980, 10980),
        {"name": "example.ext"},
        fill_value=0,
        codecs=({"name": "example.serializer"},),
    )
    assert ext.chunk_grid["configuration"]["chunk_shape"] == (10980, 10980)


def test_get_path() -> None:
    """
    `get_path` resolves like `resolve_path` but returns the default instead of
    raising on an unresolvable path, while a malformed path still raises
    `ValueError`; `resolve_path`'s missing-member error lists the member names.
    """
    leaf: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "uint8")
    root: AnyGroupSpec = GroupSpec(members={"a": GroupSpec(members={"b": leaf})})
    assert root.get_path("a/b") == leaf
    assert root.get_path("a/zz") is None
    assert root.get_path("a/zz", default="sentinel") == "sentinel"
    with pytest.raises(ValueError, match="empty member name"):
        root.get_path("a//b")
    with pytest.raises(KeyError, match="Members: 'b'"):
        root.resolve_path("a/zz")


def test_group_like_base_group_member_projection() -> None:
    """
    A `BaseGroupSpec` compares as the projection it is — by the group's own
    metadata alone — at the top level and in member position alike, so the
    comparison does not depend on depth. A `GroupSpec` with unknown members
    (`members=None`) is a different state: it matches the same, or a
    projection, never a group with known members.
    """
    g_full: AnyGroupSpec = GroupSpec(attributes={"a": 1}, members={})
    g_base = BaseGroupSpec(attributes={"a": 1})
    assert g_full.like(g_base)
    outer_full: AnyGroupSpec = GroupSpec(members={"g": g_full})
    outer_base: AnyGroupSpec = GroupSpec(members={"g": g_base})
    members = outer_base.members
    assert members is not None
    assert isinstance(members["g"], BaseGroupSpec)
    assert not isinstance(members["g"], GroupSpec)
    assert outer_full.like(outer_base)
    assert outer_base.like(outer_full)
    # differing own metadata still distinguishes the projection
    other_attrs: AnyGroupSpec = GroupSpec(members={"g": BaseGroupSpec(attributes={"a": 2})})
    assert not outer_full.like(other_attrs)
    # members=None matches members=None or a projection, never known members
    outer_unknown: AnyGroupSpec = GroupSpec(
        members={"g": GroupSpec(attributes={"a": 1}, members=None)}
    )
    assert not outer_full.like(outer_unknown)
    assert outer_unknown.like(outer_unknown)
    assert outer_unknown.like(outer_base)


def test_bytes_codec_canonicalized_inside_sharding() -> None:
    """
    The single-byte `bytes` codec normalization reaches inside a sharding
    codec's inner pipeline (same data type), while the index codecs, which
    encode the uint64 index, keep their endianness.
    """

    def sharded(endian_config: bool) -> AnyArraySpec:
        inner: tuple[CodecLike, ...] = (
            {"name": "bytes", "configuration": {"endian": "little"}}
            if endian_config
            else {"name": "bytes"},
        )
        return ArraySpec.from_shape_and_data_type(
            (8,),
            "uint8",
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": (4,)}},
            codecs=(
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": (2,),
                        "codecs": inner,
                        "index_codecs": (
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ),
                        "index_location": "end",
                    },
                },
            ),
        )

    with_endian = sharded(True)
    bare = sharded(False)
    assert with_endian == bare
    assert with_endian.like(bare)
    outer = with_endian.codecs[0]
    assert isinstance(outer, dict)
    config = outer["configuration"]
    assert isinstance(config, dict)
    assert config["codecs"] == ({"name": "bytes"},)
    index_codecs = config["index_codecs"]
    assert isinstance(index_codecs, tuple)
    assert index_codecs[0] == {"name": "bytes", "configuration": {"endian": "little"}}


def test_bare_codec_rejected() -> None:
    """
    A single codec where a pipeline is expected is refused with a clear
    message, instead of being iterated into its keys or characters.
    """
    with pytest.raises(ValidationError, match="sequence of codecs"):
        ArraySpec.from_shape_and_data_type(
            (4,),
            "uint16",
            codecs={"name": "bytes"},  # type: ignore[typeddict-item]
        )
    with pytest.raises(ValidationError, match="sequence of codecs"):
        ArraySpec.from_shape_and_data_type((4,), "uint16", codecs="bytes")


def test_from_shape_and_data_type_numpy_style_name_suggestion() -> None:
    """
    A numpy-style data type string that is not a Zarr V3 name fails with a
    message suggesting the V3 spelling.
    """
    with pytest.raises(ValueError, match="Zarr V3 form is 'int64'"):
        ArraySpec.from_shape_and_data_type((4,), "i8")


@pytest.mark.parametrize(
    ("data_type", "expected_codecs"),
    [
        ("string", ({"name": "vlen-utf8"},)),
        ("bytes", ({"name": "vlen-bytes"},)),
        ("variable_length_bytes", ({"name": "vlen-bytes"},)),
    ],
    ids=str,
)
def test_from_shape_and_data_type_variable_length(
    data_type: str, expected_codecs: tuple[Any, ...]
) -> None:
    """
    The variable-length data types derive their defaults the way zarrs does:
    an empty fill value and the matching variable-length codec, with chunks
    guessed from a nominal element size.
    """
    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), data_type)
    assert spec.fill_value == ""
    assert spec.codecs == expected_codecs
    assert spec.chunk_grid == {"name": "regular", "configuration": {"chunk_shape": (4,)}}


def test_empty_codec_configuration_canonicalized() -> None:
    """
    An empty codec `configuration` mapping is dropped at validation, matching
    the spelling difference zarr-python writes, so specs constructed with
    either spelling compare equal.
    """
    with_empty: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (4,), "string", codecs=({"name": "vlen-utf8", "configuration": {}},)
    )
    bare: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "string")
    assert with_empty.codecs == ({"name": "vlen-utf8"},)
    assert with_empty == bare


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr is required for this test")
def test_variable_length_round_trip() -> None:
    """A derived variable-length string spec round-trips a store with plain equality."""
    import zarr

    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type((4,), "string")
    arr = spec.to_zarr(zarr.storage.MemoryStore(), path="x")
    assert ArraySpec.from_zarr(arr) == spec


def test_with_sharding_codec() -> None:
    """
    `with_sharding_codec` moves the existing codec pipeline inside shards:
    the chunk shape becomes the shard shape, the subchunk shape is the inner
    chunk shape, the index takes the default encoding at the requested
    location, and a compatible second application nests the shards.
    """
    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (16, 16),
        "uint16",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
    )
    sharded = spec.with_sharding_codec((4, 4))
    assert sharded.chunk_grid == spec.chunk_grid
    outer = sharded.codecs[0]
    assert isinstance(outer, dict)
    config = outer["configuration"]
    assert isinstance(config, dict)
    assert outer["name"] == "sharding_indexed"
    assert config["chunk_shape"] == (4, 4)
    assert config["codecs"] == spec.codecs
    assert config["index_location"] == "end"
    started = spec.with_sharding_codec((4, 4), index_location="start")
    started_outer = started.codecs[0]
    assert isinstance(started_outer, dict)
    started_config = started_outer["configuration"]
    assert isinstance(started_config, dict)
    assert started_config["index_location"] == "start"
    nested = sharded.with_sharding_codec((4, 4))
    nested_outer = nested.codecs[0]
    assert isinstance(nested_outer, dict)
    nested_config = nested_outer["configuration"]
    assert isinstance(nested_config, dict)
    nested_inner = nested_config["codecs"]
    assert isinstance(nested_inner, tuple)
    first_inner = nested_inner[0]
    assert isinstance(first_inner, dict)
    assert first_inner["name"] == "sharding_indexed"


def test_with_sharding_codec_non_dividing_rejected() -> None:
    """A subchunk shape that does not divide the chunk shape is refused."""
    spec: AnyArraySpec = ArraySpec.from_shape_and_data_type(
        (16, 16),
        "uint16",
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
    )
    with pytest.raises(ValidationError, match="divide"):
        spec.with_sharding_codec((3, 3))
