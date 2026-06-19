from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import StrictArraySpec

ADAPTER = TypeAdapter(StrictArraySpec)


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
