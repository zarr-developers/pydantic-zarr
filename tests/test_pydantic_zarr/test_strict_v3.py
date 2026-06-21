"""Tests for Core/Extra strict families in pydantic_zarr._strict_v3."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr._strict_fill import (
    StrictFloat64Fill,
    StrictInt8Fill,
    StrictUint8Fill,
    is_valid_fill,
)
from pydantic_zarr.v3 import AnyCoreArraySpec, AnyExtraArraySpec

CORE_ADAPTER = TypeAdapter(AnyCoreArraySpec)
EXTRA_ADAPTER = TypeAdapter(AnyExtraArraySpec)


def test_is_valid_fill_helper() -> None:
    assert is_valid_fill("float64", "NaN") is True
    assert is_valid_fill("float64", "garbage") is False
    assert is_valid_fill("int8", 5) is True
    assert is_valid_fill("int8", 999) is False
    assert is_valid_fill("r8", (1,)) is True
    assert is_valid_fill("unknown_dtype", 0) is False


_REGULAR_GRID = {"name": "regular", "configuration": {"chunk_shape": (4,)}}
_RECTILINEAR_GRID = {
    "name": "rectilinear",
    "configuration": {"kind": "inline", "chunk_shapes": ((2, 2),)},
}
_DEFAULT_CKE = {"name": "default", "configuration": {"separator": "/"}}
_BYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}

_COMMON = {
    "shape": (4,),
    "chunk_grid": _REGULAR_GRID,
    "chunk_key_encoding": _DEFAULT_CKE,
    "codecs": (_BYTES_CODEC,),
}


def _doc(data_type: str, fill_value: object, *, grid: dict | None = None) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": data_type,
        "shape": (4,),
        "chunk_grid": grid if grid is not None else _REGULAR_GRID,
        "chunk_key_encoding": _DEFAULT_CKE,
        "fill_value": fill_value,
        "codecs": (_BYTES_CODEC,),
        "attributes": {},
    }


# ---------------------------------------------------------------------------
# AnyCoreArraySpec union routing
# ---------------------------------------------------------------------------


def test_core_float64_accepts_nan_and_hex() -> None:
    CORE_ADAPTER.validate_python(_doc("float64", "NaN"))
    CORE_ADAPTER.validate_python(_doc("float64", "0x7ff8000000000000"))


def test_core_int64_rejects_nan_string() -> None:
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(_doc("int64", "NaN"))


def test_core_raw_dtype_routes_and_validates() -> None:
    result = CORE_ADAPTER.validate_python(_doc("r8", (1,)))
    from pydantic_zarr.v3 import CoreRawArraySpec

    assert isinstance(result, CoreRawArraySpec)


@pytest.mark.parametrize("bad_dtype", ["garbage", "float128", "bytes", "r"])
def test_core_union_rejects_unknown_data_type(bad_dtype: str) -> None:
    """Union must reject data_type values that are not known literal types or r<N>."""
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(_doc(bad_dtype, 0))


@pytest.mark.parametrize("bad_dtype", ["garbage", "float128", "bytes", "r"])
def test_extra_union_rejects_unknown_data_type(bad_dtype: str) -> None:
    """Union must reject data_type values that are not known literal types or r<N>."""
    with pytest.raises(ValidationError):
        EXTRA_ADAPTER.validate_python(_doc(bad_dtype, 0))


def test_core_union_accepts_raw_r8() -> None:
    """R8 with a bytes-tuple fill_value is accepted through the union."""
    from pydantic_zarr.v3 import CoreRawArraySpec

    result = CORE_ADAPTER.validate_python(_doc("r8", (1,)))
    assert isinstance(result, CoreRawArraySpec)


def test_extra_union_accepts_raw_r16() -> None:
    """R16 with a bytes-tuple fill_value is accepted through the extra union."""
    from pydantic_zarr.v3 import ExtraRawArraySpec

    result = EXTRA_ADAPTER.validate_python(_doc("r16", (0, 0)))
    assert isinstance(result, ExtraRawArraySpec)


def test_core_union_routes_to_float64_class() -> None:
    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    result = CORE_ADAPTER.validate_python(_doc("float64", "NaN"))
    assert isinstance(result, CoreFloat64ArraySpec)


# ---------------------------------------------------------------------------
# Codec string strictness (Core)
# ---------------------------------------------------------------------------


def test_core_accepts_known_codec_name_string() -> None:
    doc = _doc("int32", 0)
    # blosc is bytes->bytes; needs an array->bytes codec (bytes) to complete the pipeline
    doc["codecs"] = ({"name": "bytes", "configuration": {"endian": "little"}}, "blosc")
    CORE_ADAPTER.validate_python(doc)


def test_core_rejects_unknown_codec_name_string() -> None:
    doc = _doc("int32", 0)
    doc["codecs"] = ("made_up",)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


def test_core_rejects_unknown_codec_object() -> None:
    doc = _doc("int32", 0)
    doc["codecs"] = ({"name": "made_up_codec", "configuration": {}},)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


# ---------------------------------------------------------------------------
# Codec string strictness (Extra)
# ---------------------------------------------------------------------------


def test_extra_accepts_known_codec_name_string() -> None:
    doc = _doc("int32", 0)
    # blosc is bytes->bytes; needs an array->bytes codec (bytes) to complete the pipeline
    doc["codecs"] = ({"name": "bytes", "configuration": {"endian": "little"}}, "blosc")
    EXTRA_ADAPTER.validate_python(doc)


def test_extra_rejects_unknown_codec_name_string() -> None:
    doc = _doc("int32", 0)
    doc["codecs"] = ("made_up",)
    with pytest.raises(ValidationError):
        EXTRA_ADAPTER.validate_python(doc)


# ---------------------------------------------------------------------------
# Family difference: chunk_grid
# ---------------------------------------------------------------------------


def test_core_rejects_rectilinear_grid() -> None:
    doc = _doc("int32", 0, grid=_RECTILINEAR_GRID)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


def test_extra_accepts_rectilinear_grid() -> None:
    doc = _doc("int32", 0, grid=_RECTILINEAR_GRID)
    EXTRA_ADAPTER.validate_python(doc)


# ---------------------------------------------------------------------------
# Family difference: codecs
# ---------------------------------------------------------------------------


def test_core_rejects_scale_offset_codec() -> None:
    doc = _doc("float32", 0.0)
    doc["codecs"] = ({"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


def test_extra_accepts_scale_offset_codec() -> None:
    doc = _doc("float32", 0.0)
    # scale_offset is array->array; needs an array->bytes codec to complete the pipeline
    doc["codecs"] = (
        {"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    )
    EXTRA_ADAPTER.validate_python(doc)


def test_core_rejects_scale_offset_codec_name_string() -> None:
    doc = _doc("float32", 0.0)
    doc["codecs"] = ("scale_offset",)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


def test_extra_accepts_scale_offset_codec_name_string() -> None:
    doc = _doc("float32", 0.0)
    # scale_offset is array->array; needs an array->bytes codec to complete the pipeline
    doc["codecs"] = ("scale_offset", {"name": "bytes", "configuration": {"endian": "little"}})
    EXTRA_ADAPTER.validate_python(doc)


# ---------------------------------------------------------------------------
# CoreArraySpec single-class construction (runtime dtype/fill coupling)
# ---------------------------------------------------------------------------


def test_core_array_spec_construct_float64_nan() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec(data_type="float64", fill_value="NaN", **_COMMON)
    assert spec.data_type == "float64"
    assert spec.fill_value == "NaN"


def test_core_array_spec_construct_int64_nan_raises() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(data_type="int64", fill_value="NaN", **_COMMON)


def test_core_array_spec_construct_unknown_dtype_raises() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(data_type="float128", fill_value=0.0, **_COMMON)


def test_core_array_spec_construct_raw_bytes_ok() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec(data_type="r8", fill_value=(1,), **_COMMON)
    assert spec.data_type == "r8"


def test_core_array_spec_construct_raw_nan_raises() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(data_type="r8", fill_value="NaN", **_COMMON)


def test_core_array_spec_rejects_rectilinear_grid() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(
            data_type="float64",
            fill_value="NaN",
            shape=(4,),
            chunk_grid=_RECTILINEAR_GRID,
            chunk_key_encoding=_DEFAULT_CKE,
            codecs=(_BYTES_CODEC,),
        )


# ---------------------------------------------------------------------------
# ExtraArraySpec single-class construction
# ---------------------------------------------------------------------------


def test_extra_array_spec_construct_float64_nan() -> None:
    from pydantic_zarr.v3 import ExtraArraySpec

    spec = ExtraArraySpec(data_type="float64", fill_value="NaN", **_COMMON)
    assert spec.data_type == "float64"


def test_extra_array_spec_accepts_rectilinear_grid() -> None:
    from pydantic_zarr.v3 import ExtraArraySpec

    spec = ExtraArraySpec(
        data_type="float64",
        fill_value="NaN",
        shape=(4,),
        chunk_grid=_RECTILINEAR_GRID,
        chunk_key_encoding=_DEFAULT_CKE,
        codecs=(_BYTES_CODEC,),
    )
    assert spec.data_type == "float64"


# ---------------------------------------------------------------------------
# Public per-dtype classes
# ---------------------------------------------------------------------------


def test_core_float64_array_spec_construct_infinity() -> None:
    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    spec = CoreFloat64ArraySpec(data_type="float64", fill_value="Infinity", **_COMMON)
    assert spec.fill_value == "Infinity"


def test_extra_float64_array_spec_construct() -> None:
    from pydantic_zarr.v3 import ExtraFloat64ArraySpec

    spec = ExtraFloat64ArraySpec(data_type="float64", fill_value="NaN", **_COMMON)
    assert spec.data_type == "float64"


# ---------------------------------------------------------------------------
# Group spec recursion
# ---------------------------------------------------------------------------

_ARRAY_DOC_INT32 = {
    "zarr_format": 3,
    "node_type": "array",
    "data_type": "int32",
    "shape": (4,),
    "chunk_grid": _REGULAR_GRID,
    "chunk_key_encoding": _DEFAULT_CKE,
    "fill_value": 0,
    "codecs": (_BYTES_CODEC,),
    "attributes": {},
}


def test_core_group_accepts_nested_strict_members() -> None:
    from pydantic_zarr.v3 import CoreGroupSpec

    ta = TypeAdapter(CoreGroupSpec)
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {
            "sub": {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {},
                "members": {"arr": _ARRAY_DOC_INT32},
            },
        },
    }
    ta.validate_python(doc)


def test_core_group_rejects_nonstrict_member() -> None:
    from pydantic_zarr.v3 import CoreGroupSpec

    ta = TypeAdapter(CoreGroupSpec)
    bad_arr = {**_ARRAY_DOC_INT32, "fill_value": "NaN"}  # invalid for int32
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {"arr": bad_arr},
    }
    with pytest.raises(ValidationError):
        ta.validate_python(doc)


def test_core_group_rejects_rectilinear_member() -> None:
    from pydantic_zarr.v3 import CoreGroupSpec

    ta = TypeAdapter(CoreGroupSpec)
    recti_arr = {**_ARRAY_DOC_INT32, "chunk_grid": _RECTILINEAR_GRID}
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {"arr": recti_arr},
    }
    with pytest.raises(ValidationError):
        ta.validate_python(doc)


def test_extra_group_accepts_rectilinear_member() -> None:
    from pydantic_zarr.v3 import ExtraGroupSpec

    ta = TypeAdapter(ExtraGroupSpec)
    recti_arr = {**_ARRAY_DOC_INT32, "chunk_grid": _RECTILINEAR_GRID}
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {"arr": recti_arr},
    }
    ta.validate_python(doc)


# ---------------------------------------------------------------------------
# Attributes default and non-generic
# ---------------------------------------------------------------------------


def test_core_attributes_defaults_and_nongeneric() -> None:
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "shape": (4,),
        "chunk_grid": _REGULAR_GRID,
        "chunk_key_encoding": _DEFAULT_CKE,
        "fill_value": 0,
        "codecs": (_BYTES_CODEC,),
    }  # no "attributes" key
    result = CORE_ADAPTER.validate_python(doc)
    assert result.attributes == {}

    from pydantic_zarr.v3 import CoreGroupSpec

    g = TypeAdapter(CoreGroupSpec).validate_python(
        {"zarr_format": 3, "node_type": "group", "members": {}}
    )
    assert g.attributes == {}


# ---------------------------------------------------------------------------
# Drift-guard: shared base fields with loose ArraySpec
# ---------------------------------------------------------------------------


def test_loose_and_core_share_base_fields() -> None:
    """Loose and Core strict specs must share identical non-variant fields."""
    from pydantic_zarr.v3 import ArraySpec, CoreFloat64ArraySpec, _BaseArraySpec

    shared = set(_BaseArraySpec.model_fields)
    variant = {"data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs"}
    assert set(ArraySpec.model_fields) - variant == shared
    assert set(CoreFloat64ArraySpec.model_fields) - variant == shared


# ---------------------------------------------------------------------------
# Float fill wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [1.0, 5, "NaN", "Infinity", "-Infinity", "0x7ff8000000000000"])
def test_strict_float64_fill_accepts_valid(good: object) -> None:
    assert TypeAdapter(StrictFloat64Fill).validate_python(good) == good


@pytest.mark.parametrize("bad", ["garbage", "0xZZ", "0x123", "NotAFloat", "nan"])
def test_strict_float64_fill_rejects_bad_strings(bad: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(StrictFloat64Fill).validate_python(bad)


# ---------------------------------------------------------------------------
# Integer fill wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [5, 5.0, 127, -128, 127.0, 0])
def test_strict_int8_fill_accepts_valid(good: object) -> None:
    TypeAdapter(StrictInt8Fill).validate_python(good)


@pytest.mark.parametrize("bad", [5.5, True, False, "5", 999, -200, "garbage"])
def test_strict_int8_fill_rejects_bad(bad: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(StrictInt8Fill).validate_python(bad)


@pytest.mark.parametrize("bad", [-1, 256, 1.5, True])
def test_strict_uint8_fill_rejects_bad(bad: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(StrictUint8Fill).validate_python(bad)


# ---------------------------------------------------------------------------
# Bool fill wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("good", "bad"), [(True, 1), (False, "True")])
def test_strict_bool_fill(good: object, bad: object) -> None:
    from pydantic_zarr._strict_fill import StrictBoolFill

    TypeAdapter(StrictBoolFill).validate_python(good)
    with pytest.raises(ValidationError):
        TypeAdapter(StrictBoolFill).validate_python(bad)


# ---------------------------------------------------------------------------
# Complex fill wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [(1.0, 2.0), ("NaN", 1.0), (1.0, "Infinity")])
def test_strict_complex64_fill_accepts(good: object) -> None:
    from pydantic_zarr._strict_fill import StrictComplex64Fill

    TypeAdapter(StrictComplex64Fill).validate_python(good)


@pytest.mark.parametrize("bad", [("garbage", 1.0), 1.0, ("NaN",)])
def test_strict_complex64_fill_rejects(bad: object) -> None:
    from pydantic_zarr._strict_fill import StrictComplex64Fill

    with pytest.raises(ValidationError):
        TypeAdapter(StrictComplex64Fill).validate_python(bad)


# ---------------------------------------------------------------------------
# Raw bytes fill wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [(1, 2, 3), (0, 255), (), [1, 2], [0, 128, 255]])
def test_strict_raw_fill_accepts(good: object) -> None:
    from pydantic_zarr._strict_fill import StrictRawFill

    TypeAdapter(StrictRawFill).validate_python(good)


@pytest.mark.parametrize("bad", [(1, 999), (-1, 0), ("a",), [1, 999], [-1, 0]])
def test_strict_raw_fill_rejects(bad: object) -> None:
    from pydantic_zarr._strict_fill import StrictRawFill

    with pytest.raises(ValidationError):
        TypeAdapter(StrictRawFill).validate_python(bad)


# ---------------------------------------------------------------------------
# End-to-end fill rejection through per-dtype classes and single-class
# ---------------------------------------------------------------------------

_COMMON_E2E = {
    "shape": (4,),
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    "attributes": {},
}


def test_per_dtype_class_rejects_garbage_float_fill() -> None:
    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    with pytest.raises(ValidationError):
        CoreFloat64ArraySpec(data_type="float64", fill_value="garbage", **_COMMON_E2E)


def test_union_rejects_garbage_float_fill() -> None:
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "float64",
        "fill_value": "garbage",
        **_COMMON_E2E,
    }
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(doc)


def test_single_class_rejects_garbage_float_fill() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(data_type="float64", fill_value="garbage", **_COMMON_E2E)


def test_single_class_rejects_out_of_range_int_fill() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValidationError):
        CoreArraySpec(data_type="int8", fill_value=999, **_COMMON_E2E)


# ---------------------------------------------------------------------------
# AUTO sentinel / default constructors (Task 10)
# ---------------------------------------------------------------------------


def test_core_array_spec_bare_construct() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec.create()
    assert spec.data_type == "float64"  # default dtype
    assert spec.chunk_grid["name"] == "regular"
    assert spec.codecs[0]["name"] == "bytes"


def test_core_array_spec_shape_only_derives_grid() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec.create(shape=(10, 10))
    assert spec.chunk_grid["configuration"]["chunk_shape"] == (10, 10)


def test_core_float64_defaults_fill() -> None:
    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    spec = CoreFloat64ArraySpec.create(shape=(4,))
    assert spec.fill_value == "NaN"  # float64 default


# ---------------------------------------------------------------------------
# from_array constructors (Task 11)
# ---------------------------------------------------------------------------


def test_core_from_array_single_class() -> None:
    import numpy as np

    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec.from_array(np.zeros((4, 4), dtype="float64"))
    assert spec.data_type == "float64"
    assert spec.shape == (4, 4)
    assert spec.chunk_grid["configuration"]["chunk_shape"] == (4, 4)


def test_core_float64_from_array_returns_that_class() -> None:
    import numpy as np

    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    spec = CoreFloat64ArraySpec.from_array(np.zeros((2,), dtype="float64"))
    assert type(spec).__name__ == "CoreFloat64ArraySpec"


def test_core_float64_from_array_wrong_dtype_raises() -> None:
    import numpy as np

    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    with pytest.raises(ValidationError):
        CoreFloat64ArraySpec.from_array(np.zeros((2,), dtype="int32"))


# ---------------------------------------------------------------------------
# create_with_sharding constructors (Task 14)
# ---------------------------------------------------------------------------


def test_create_with_sharding_builds_sharding_codec() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec.create_with_sharding(
        outer_chunk_shape=(8, 8),
        inner_chunk_shape=(4, 4),
        shape=(8, 8),
        data_type="int32",
        fill_value=0,
    )
    assert spec.chunk_grid["configuration"]["chunk_shape"] == (8, 8)
    codec = spec.codecs[0]
    assert codec["name"] == "sharding_indexed"
    assert codec["configuration"]["chunk_shape"] == (4, 4)
    assert codec["configuration"]["index_codecs"][-1]["name"] == "crc32c"  # default index pipeline


def test_create_with_sharding_rejects_indivisible() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValueError):
        CoreArraySpec.create_with_sharding(
            outer_chunk_shape=(8, 8),
            inner_chunk_shape=(3, 3),
            shape=(8, 8),
            data_type="int32",
            fill_value=0,
        )


def test_create_with_sharding_rejects_rank_mismatch() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    with pytest.raises(ValueError):
        CoreArraySpec.create_with_sharding(
            outer_chunk_shape=(8, 8),
            inner_chunk_shape=(4,),
            shape=(8, 8),
            data_type="int32",
            fill_value=0,
        )


def test_create_with_sharding_per_dtype_class() -> None:
    from pydantic_zarr.v3 import CoreFloat64ArraySpec

    spec = CoreFloat64ArraySpec.create_with_sharding(
        outer_chunk_shape=(8,),
        inner_chunk_shape=(4,),
        shape=(8,),
        fill_value="NaN",
    )
    assert type(spec).__name__ == "CoreFloat64ArraySpec"
    assert spec.codecs[0]["name"] == "sharding_indexed"


def test_create_with_sharding_custom_inner_codecs() -> None:
    from pydantic_zarr.v3 import CoreArraySpec

    spec = CoreArraySpec.create_with_sharding(
        outer_chunk_shape=(8,),
        inner_chunk_shape=(4,),
        shape=(8,),
        data_type="int32",
        fill_value=0,
        inner_chunk_codecs=(
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 1}},
        ),
    )
    inner = spec.codecs[0]["configuration"]["codecs"]
    assert inner[-1]["name"] == "gzip"


def test_extra_create_with_sharding() -> None:
    from pydantic_zarr.v3 import ExtraArraySpec

    spec = ExtraArraySpec.create_with_sharding(
        outer_chunk_shape=(8,),
        inner_chunk_shape=(4,),
        shape=(8,),
        data_type="int32",
        fill_value=0,
    )
    assert spec.codecs[0]["name"] == "sharding_indexed"


# ---------------------------------------------------------------------------
# Internal validator rejection tests (Task 11)
# AfterValidator on codecs/chunk_grid must fire during pydantic validation of
# a parsed-from-dict strict spec, catching codec/grid configuration errors.
# ---------------------------------------------------------------------------


def test_core_union_rejects_non_permutation_transpose() -> None:
    """A transpose codec with a non-permutation order must be rejected via the union."""
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "fill_value": 0,
        "shape": (4, 4),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        # order [0, 0] is not a permutation of range(2)
        "codecs": ({"name": "transpose", "configuration": {"order": [0, 0]}},),
        "attributes": {},
    }
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(doc)


def test_core_union_rejects_blosc_out_of_range_clevel() -> None:
    """A blosc codec with clevel outside [0, 9] must be rejected via the union."""
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "fill_value": 0,
        "shape": (4,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": (
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 99,
                    "shuffle": "noshuffle",
                    "blocksize": 0,
                },
            },
        ),
        "attributes": {},
    }
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(doc)


def test_core_union_rejects_non_positive_chunk_shape() -> None:
    """A regular chunk_grid with a zero dimension must be rejected via the union."""
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "fill_value": 0,
        "shape": (4,),
        # chunk_shape with 0 dimension
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (0,)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "attributes": {},
    }
    with pytest.raises(ValidationError):
        TypeAdapter(AnyCoreArraySpec).validate_python(doc)
