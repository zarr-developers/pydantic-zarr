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


_DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "r8",
    "r16",
]
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


# Restrict to short-string (name-only) form so the only variable is name-vocabulary.
# Object forms like {"name": "blosc"} are structurally rejected when required config
# fields (e.g. cname/clevel for blosc) are absent — a rejection unrelated to the
# name-validity question the oracle tests.  Short strings avoid that noise.
_CODECS = st.sampled_from(
    ["blosc", "bytes", "zstd", "scale_offset", "cast_value", "made_up", "garbage"]
)
_GRIDS = st.sampled_from(
    [
        {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": ((1, 3),)}},
        {"name": "made_up", "configuration": {}},
    ]
)


def _accepts_field(adapter: TypeAdapter, **override: object) -> bool:
    try:
        adapter.validate_python({**_BASE, "data_type": "int64", "fill_value": 0, **override})
    except ValidationError:
        return False
    else:
        return True


@given(grid=_GRIDS)
def test_core_grid_matches_oracle(grid: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    assert _accepts_field(adapter, chunk_grid=grid) == is_valid_grid("core", grid)


@given(grid=_GRIDS)
def test_extra_grid_matches_oracle(grid: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyExtraArraySpec)
    assert _accepts_field(adapter, chunk_grid=grid) == is_valid_grid("extra", grid)


@given(codec=_CODECS)
def test_core_codec_matches_oracle(codec: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    # a bytes codec is always needed; put the candidate first
    bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    accepts = _accepts_field(adapter, codecs=(codec, bytes_codec))
    assert accepts == is_valid_codec("core", codec)


@given(codec=_CODECS)
def test_extra_codec_matches_oracle(codec: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyExtraArraySpec)
    # a bytes codec is always needed; put the candidate first
    bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    accepts = _accepts_field(adapter, codecs=(codec, bytes_codec))
    assert accepts == is_valid_codec("extra", codec)
