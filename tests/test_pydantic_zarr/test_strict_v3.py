"""Tests for Core/Extra strict families in pydantic_zarr._strict_v3."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import AnyCoreArraySpec, AnyExtraArraySpec

CORE_ADAPTER = TypeAdapter(AnyCoreArraySpec)
EXTRA_ADAPTER = TypeAdapter(AnyExtraArraySpec)

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
    doc["codecs"] = ("blosc",)
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
    doc["codecs"] = ("blosc",)
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
    doc["codecs"] = ({"name": "scale_offset", "configuration": {"scale": 1.0, "offset": 0.0}},)
    EXTRA_ADAPTER.validate_python(doc)


def test_core_rejects_scale_offset_codec_name_string() -> None:
    doc = _doc("float32", 0.0)
    doc["codecs"] = ("scale_offset",)
    with pytest.raises(ValidationError):
        CORE_ADAPTER.validate_python(doc)


def test_extra_accepts_scale_offset_codec_name_string() -> None:
    doc = _doc("float32", 0.0)
    doc["codecs"] = ("scale_offset",)
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
