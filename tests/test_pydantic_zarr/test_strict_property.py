from __future__ import annotations

import pytest

from .strict_oracle import is_valid_codec, is_valid_fill, is_valid_grid


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
