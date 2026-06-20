"""Spec-derived reference predicates for strict v3 validation.

Encodes the Zarr v3 spec rules INDEPENDENTLY of pydantic-zarr's validators (its
own regexes, ranges, vocabularies), so the property tests are a true
differential check. Do NOT import from the pydantic-zarr package here.
"""

from __future__ import annotations

import re

_HEX_WIDTH = {"float16": 4, "float32": 8, "float64": 16}
_FLOAT_SPECIALS = {"NaN", "Infinity", "-Infinity"}
_INT_RANGE = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}
_COMPLEX_COMPONENT = {"complex64": "float32", "complex128": "float64"}
_RAW_RE = re.compile(r"^r(\d+)$")

_CORE_CODECS = {"blosc", "bytes", "crc32c", "gzip", "sharding_indexed", "transpose", "zstd"}
_EXTRA_CODECS = _CORE_CODECS | {"scale_offset", "cast_value"}


def _valid_float(dt: str, v: object) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        if v in _FLOAT_SPECIALS:
            return True
        return bool(re.fullmatch(rf"0x[0-9a-fA-F]{{{_HEX_WIDTH[dt]}}}", v))
    return False


def _valid_int(dt: str, v: object) -> bool:
    lo, hi = _INT_RANGE[dt]
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return lo <= v <= hi
    if isinstance(v, float):
        return v.is_integer() and lo <= int(v) <= hi
    return False


def is_valid_fill(data_type: str, value: object) -> bool:
    if data_type == "bool":
        return type(value) is bool
    if data_type in _INT_RANGE:
        return _valid_int(data_type, value)
    if data_type in _HEX_WIDTH:
        return _valid_float(data_type, value)
    if data_type in _COMPLEX_COMPONENT:
        comp = _COMPLEX_COMPONENT[data_type]
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and all(_valid_float(comp, c) for c in value)
        )
    m = _RAW_RE.fullmatch(data_type)
    if m:
        nbytes = int(m.group(1)) // 8
        return (
            isinstance(value, tuple)
            and len(value) == nbytes
            and all(isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= 255 for b in value)
        )
    return False  # unknown data_type


def _codec_name(codec: object) -> str | None:
    if isinstance(codec, str):
        return codec
    if isinstance(codec, dict) and isinstance(codec.get("name"), str):
        return codec["name"]
    return None


def is_valid_codec(family: str, codec: object) -> bool:
    name = _codec_name(codec)
    if name is None:
        return False
    if family == "core":
        return name in _CORE_CODECS
    if family == "extra":
        return name in _EXTRA_CODECS
    raise ValueError(f"unknown family: {family!r}")


def is_valid_grid(family: str, grid: object) -> bool:
    if not isinstance(grid, dict):
        return False
    name = grid.get("name")
    if name == "regular":
        return True
    if name == "rectilinear":
        return family == "extra"
    return False
