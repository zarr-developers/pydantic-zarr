from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from pydantic_zarr.v3 import AnyCoreArraySpec, AnyExtraArraySpec

from .strict_oracle import (
    is_valid_codec,
    is_valid_codec_internal,
    is_valid_fill,
    is_valid_grid,
    is_valid_ndim_match,
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
        ("r8", [1], True),
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
    # list variants mirror tuple variants — JSON arrays deserialise as lists
    st.lists(st.integers(min_value=-5, max_value=300), min_size=1, max_size=2),
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
    [
        "blosc",
        "bytes",
        "zstd",
        "scale_offset",
        "cast_value",
        "made_up",
        "garbage",
        "transpose",
        "sharding_indexed",
    ]
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


def _codec_pipeline_for(codec: object) -> tuple[object, ...]:
    """Build a minimal valid pipeline around *codec* for the pipeline-count check.

    If the candidate codec is itself an array->bytes codec (e.g. ``"bytes"``),
    the pipeline is just ``(codec,)``.  Otherwise a ``bytes`` codec is appended
    so that the pipeline has exactly one array->bytes step.
    """
    _ARRAY_BYTES_NAMES = {"bytes", "sharding_indexed"}
    name = (
        codec
        if isinstance(codec, str)
        else (codec.get("name") if isinstance(codec, dict) else None)
    )
    if name in _ARRAY_BYTES_NAMES:
        return (codec,)
    bytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
    return (codec, bytes_codec)


@given(codec=_CODECS)
def test_core_codec_matches_oracle(codec: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    accepts = _accepts_field(adapter, codecs=_codec_pipeline_for(codec))
    assert accepts == is_valid_codec("core", codec)


@given(codec=_CODECS)
def test_extra_codec_matches_oracle(codec: object) -> None:
    adapter: TypeAdapter = TypeAdapter(AnyExtraArraySpec)
    accepts = _accepts_field(adapter, codecs=_codec_pipeline_for(codec))
    assert accepts == is_valid_codec("extra", codec)


# ---------------------------------------------------------------------------
# Internal-validity property tests
# ---------------------------------------------------------------------------

# Strategy: generate lists of small ints that are sometimes valid permutations,
# sometimes not.  A pure permutation of range(n) is a specific subset — mixing
# in duplicates or out-of-range values exercises the non-permutation branch.
_SMALL_INTS = st.integers(min_value=0, max_value=5)
_ORDER_LISTS = st.one_of(
    # valid permutations of range(n) for n in 1..4
    st.permutations([0]),
    st.permutations([0, 1]),
    st.permutations([0, 1, 2]),
    st.permutations([0, 1, 2, 3]),
    # potentially-invalid: arbitrary short lists of small ints (may repeat / skip)
    st.lists(_SMALL_INTS, min_size=1, max_size=4),
)

_BYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}


@given(order=_ORDER_LISTS)
def test_transpose_internal_matches_oracle(order: list[int]) -> None:
    """TypeAdapter accepts transpose iff order is a valid permutation."""
    ndim = len(order)
    # shape and chunk_shape must match the transpose ndim
    shape = tuple(4 for _ in range(ndim))
    chunk_shape = tuple(4 for _ in range(ndim))
    chunk_grid = {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}
    transpose_codec = {"name": "transpose", "configuration": {"order": order}}
    # pipeline: transpose (array->array) then bytes (array->bytes)
    pipeline = (transpose_codec, _BYTES_CODEC)
    doc = {
        **_BASE,
        "shape": shape,
        "chunk_grid": chunk_grid,
        "data_type": "int64",
        "fill_value": 0,
        "codecs": pipeline,
    }
    try:
        TypeAdapter(AnyCoreArraySpec).validate_python(doc)
        accepted = True
    except ValidationError:
        accepted = False
    oracle_ok = is_valid_codec_internal("transpose", transpose_codec)
    assert accepted == oracle_ok, f"order={order}: accepted={accepted} but oracle says {oracle_ok}"


# Strategy: generate clevel values in and around the valid [0, 9] range
_BLOSC_CLEVELS = st.integers(min_value=-3, max_value=12)


@given(clevel=_BLOSC_CLEVELS)
def test_blosc_clevel_matches_oracle(clevel: int) -> None:
    """TypeAdapter accepts blosc iff clevel is in [0, 9]."""
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    blosc_codec = {
        "name": "blosc",
        "configuration": {
            "cname": "lz4",
            "clevel": clevel,
            "shuffle": "noshuffle",
            "blocksize": 0,
        },
    }
    # blosc is bytes_bytes; pipeline = bytes (array->bytes) then blosc (bytes->bytes)
    pipeline = (_BYTES_CODEC, blosc_codec)
    accepted = _accepts_field(adapter, codecs=pipeline)
    oracle_ok = is_valid_codec_internal("blosc", blosc_codec)
    assert accepted == oracle_ok, (
        f"clevel={clevel}: accepted={accepted} but oracle says {oracle_ok}"
    )


# ---------------------------------------------------------------------------
# Dimensionality-matching property tests
# ---------------------------------------------------------------------------

# We vary the chunk_shape ndim independently of the array shape ndim.
_NDIMS = st.integers(min_value=1, max_value=4)


@given(array_ndim=_NDIMS, chunk_ndim=_NDIMS)
def test_regular_grid_ndim_matches_oracle(array_ndim: int, chunk_ndim: int) -> None:
    """TypeAdapter accepts regular grid iff array ndim == chunk_grid ndim."""
    adapter: TypeAdapter = TypeAdapter(AnyCoreArraySpec)
    shape = tuple(4 for _ in range(array_ndim))
    chunk_shape = tuple(4 for _ in range(chunk_ndim))
    chunk_grid = {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}
    # codec must match the array ndim (use bytes which is ndim-agnostic)
    accepted = _accepts_field(adapter, shape=shape, chunk_grid=chunk_grid)
    oracle_ok = is_valid_ndim_match(shape, chunk_grid)
    assert accepted == oracle_ok, (
        f"array_ndim={array_ndim} chunk_ndim={chunk_ndim}: "
        f"accepted={accepted} but oracle says {oracle_ok}"
    )
