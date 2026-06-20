from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import AnyCoreArraySpec, AnyExtraArraySpec

from .strict_oracle import (
    is_valid_codec,
    is_valid_fill,
    is_valid_grid,
)


@pytest.mark.parametrize(
    ("dt", "val", "ok"),
    [
        ("float64", "NaN", True),
        ("float64", "0x7ff8000000000000", True),
        ("float64", "garbage", False),
        ("float64", 1.0, True),
        ("int8", 5, True),
        ("int8", 5.0, True),
        ("int8", 5.5, False),
        ("int8", 999, False),
        ("int8", True, False),
        ("int8", "5", False),
        ("bool", True, True),
        ("bool", 1, False),
        ("complex64", ("NaN", 1.0), True),
        ("complex64", ("garbage", 1.0), False),
        ("r8", (255,), True),
        ("r8", (256,), False),
        ("r8", [1], False),
    ],
)
def test_oracle_fill(dt: str, val: object, ok: bool) -> None:
    assert is_valid_fill(dt, val) is ok


@pytest.mark.parametrize(
    ("fam", "grid", "ok"),
    [
        ("core", {"name": "regular", "configuration": {"chunk_shape": (4,)}}, True),
        (
            "core",
            {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": ((1, 3),)}},
            False,
        ),
        (
            "extra",
            {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": ((1, 3),)}},
            True,
        ),
    ],
)
def test_oracle_grid(fam: str, grid: object, ok: bool) -> None:
    assert is_valid_grid(fam, grid) is ok


@pytest.mark.parametrize(
    ("fam", "codec", "ok"),
    [
        ("core", "blosc", True),
        ("core", "made_up", False),
        ("core", "scale_offset", False),
        ("extra", "scale_offset", True),
    ],
)
def test_oracle_codec(fam: str, codec: object, ok: bool) -> None:
    assert is_valid_codec(fam, codec) is ok


_DTYPES = ["bool", "int8", "uint8", "int64", "float16", "float64", "complex64", "r8", "r16"]
# candidate fill values spanning valid + adversarial inputs
_FILLS = st.one_of(
    st.booleans(),
    st.integers(min_value=-300, max_value=300),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.sampled_from(["NaN", "Infinity", "-Infinity", "garbage", "0x7ff8000000000000", "0xZZ"]),
    st.tuples(st.sampled_from(["NaN", 1.0, "garbage"]), st.sampled_from(["NaN", 2.0])),
    st.tuples(st.integers(min_value=-5, max_value=300)),
    st.tuples(st.integers(min_value=-5, max_value=300), st.integers(min_value=-5, max_value=300)),
)

_BASE = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4,),
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    "attributes": {},
}


def _accepts(adapter: TypeAdapter, data_type: str, fill: object) -> bool:
    try:
        adapter.validate_python({**_BASE, "data_type": data_type, "fill_value": fill})
    except ValidationError:
        return False
    else:
        return True


@given(data_type=st.sampled_from(_DTYPES), fill=_FILLS)
def test_core_fill_matches_oracle(data_type: str, fill: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    assert _accepts(adapter, data_type, fill) == is_valid_fill(data_type, fill)


@given(data_type=st.sampled_from(_DTYPES), fill=_FILLS)
def test_extra_fill_matches_oracle(data_type: str, fill: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyExtraArraySpec)
    assert _accepts(adapter, data_type, fill) == is_valid_fill(data_type, fill)
