from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import AnyStrictArraySpec

ADAPTER = TypeAdapter(AnyStrictArraySpec)

_COMMON = {
    "shape": (4,),
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
}


def _doc(data_type: str, fill_value: object) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": data_type,
        "shape": (4,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
        "attributes": {},
    }


def test_float64_accepts_nan_and_hex() -> None:
    ADAPTER.validate_python(_doc("float64", "NaN"))
    ADAPTER.validate_python(_doc("float64", "0x7ff8000000000000"))


def test_int64_rejects_nan_string() -> None:
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(_doc("int64", "NaN"))


def test_raw_dtype_routes_and_validates() -> None:
    ADAPTER.validate_python(_doc("r8", (1,)))


def test_strict_rejects_unknown_codec() -> None:
    doc = _doc("int32", 0)
    doc["codecs"] = ({"name": "made_up_codec", "configuration": {}},)
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(doc)


def test_strict_group_accepts_nested_strict_members() -> None:
    from pydantic import TypeAdapter

    from pydantic_zarr.v3 import StrictGroupSpec

    ta = TypeAdapter(StrictGroupSpec)
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {
            "sub": {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {},
                "members": {
                    "arr": {
                        "zarr_format": 3,
                        "node_type": "array",
                        "data_type": "int32",
                        "shape": (4,),
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
                        "chunk_key_encoding": {
                            "name": "default",
                            "configuration": {"separator": "/"},
                        },
                        "fill_value": 0,
                        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                        "attributes": {},
                    },
                },
            },
        },
    }
    ta.validate_python(doc)


def test_strict_attributes_defaults_and_nongeneric() -> None:
    from pydantic import TypeAdapter

    from pydantic_zarr.v3 import AnyStrictArraySpec, StrictGroupSpec

    # array doc omitting attributes validates (default {})
    ta = TypeAdapter(AnyStrictArraySpec)
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "data_type": "int32",
        "shape": (4,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }  # no "attributes" key
    result = ta.validate_python(doc)
    assert result.attributes == {}
    # group doc omitting attributes validates too
    g = TypeAdapter(StrictGroupSpec).validate_python(
        {"zarr_format": 3, "node_type": "group", "members": {}}
    )
    assert g.attributes == {}


def test_strict_group_rejects_nonstrict_member() -> None:
    import pytest
    from pydantic import TypeAdapter, ValidationError

    from pydantic_zarr.v3 import StrictGroupSpec

    ta = TypeAdapter(StrictGroupSpec)
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
        "members": {
            "arr": {
                "zarr_format": 3,
                "node_type": "array",
                "data_type": "int32",
                "shape": (4,),
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": "NaN",  # invalid for int32 -> must propagate to a rejection
                "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                "attributes": {},
            },
        },
    }
    with pytest.raises(ValidationError):
        ta.validate_python(doc)


def test_loose_and_strict_share_base_fields() -> None:
    """Loose and strict specs must share identical non-codec/non-dtype fields."""
    from pydantic_zarr._strict_v3 import Float64ArraySpec
    from pydantic_zarr.v3 import ArraySpec, _BaseArraySpec

    shared = set(_BaseArraySpec.model_fields)
    variant = {"data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs"}
    assert set(ArraySpec.model_fields) - variant == shared
    assert set(Float64ArraySpec.model_fields) - variant == shared


# ---- Construction tests for StrictArraySpec (single constructible class) ----


def test_strict_array_spec_construct_float64_nan() -> None:
    """StrictArraySpec with float64 + 'NaN' fill_value constructs successfully."""
    from pydantic_zarr.v3 import StrictArraySpec

    spec = StrictArraySpec(data_type="float64", fill_value="NaN", **_COMMON)
    assert spec.data_type == "float64"
    assert spec.fill_value == "NaN"


def test_strict_array_spec_construct_int64_nan_raises() -> None:
    """StrictArraySpec with int64 + 'NaN' fill_value raises ValidationError."""
    from pydantic import ValidationError

    from pydantic_zarr.v3 import StrictArraySpec

    with pytest.raises(ValidationError):
        StrictArraySpec(data_type="int64", fill_value="NaN", **_COMMON)


def test_strict_array_spec_construct_unknown_dtype_raises() -> None:
    """StrictArraySpec with unrecognized data_type raises ValidationError."""
    from pydantic import ValidationError

    from pydantic_zarr.v3 import StrictArraySpec

    with pytest.raises(ValidationError):
        StrictArraySpec(data_type="float128", fill_value=0.0, **_COMMON)


def test_strict_array_spec_construct_raw_bytes_ok() -> None:
    """StrictArraySpec with r8 data_type and bytes-tuple fill_value constructs successfully."""
    from pydantic_zarr.v3 import StrictArraySpec

    spec = StrictArraySpec(data_type="r8", fill_value=(1,), **_COMMON)
    assert spec.data_type == "r8"


def test_strict_array_spec_construct_raw_nan_raises() -> None:
    """StrictArraySpec with r8 data_type and 'NaN' fill_value raises ValidationError."""
    from pydantic import ValidationError

    from pydantic_zarr.v3 import StrictArraySpec

    with pytest.raises(ValidationError):
        StrictArraySpec(data_type="r8", fill_value="NaN", **_COMMON)


# ---- Construction tests for public per-dtype classes ----


def test_float64_array_spec_construct_infinity() -> None:
    """Float64ArraySpec constructs directly with fill_value='Infinity'."""
    from pydantic_zarr.v3 import Float64ArraySpec

    spec = Float64ArraySpec(data_type="float64", fill_value="Infinity", **_COMMON)
    assert spec.fill_value == "Infinity"
