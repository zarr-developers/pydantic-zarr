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
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(_valid_float(comp, c) for c in value)
        )
    m = _RAW_RE.fullmatch(data_type)
    if m:
        nbytes = int(m.group(1)) // 8
        return (
            isinstance(value, (tuple, list))
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


def is_valid_codec_internal(name: str, meta: object) -> bool:
    """Return True iff the codec's own configuration is internally valid.

    Rules encoded from the spec, independently of pydantic-zarr source:
    - ``transpose``: ``order`` must be a permutation of ``range(len(order))``
    - ``blosc``: ``clevel`` must be in [0, 9]
    - ``gzip``: ``level`` must be in [0, 9]
    - ``regular`` (chunk grid) or ``sharding_indexed``: ``chunk_shape`` must be
      all-positive integers
    - All other recognised codec names: True (no extra internal rule)

    *meta* is expected to be a dict with at least a ``"configuration"`` key when
    the codec carries configuration; if the key is absent the function returns
    True (no configuration → nothing to validate internally).
    """
    if not isinstance(meta, dict):
        # bare string name — no configuration to validate
        return True
    config = meta.get("configuration")
    if not isinstance(config, dict):
        return True

    if name == "transpose":
        order = config.get("order")
        if not isinstance(order, (list, tuple)):
            return False
        order_list = list(order)
        return sorted(order_list) == list(range(len(order_list)))

    if name == "blosc":
        clevel = config.get("clevel")
        if not isinstance(clevel, int) or isinstance(clevel, bool):
            return False
        return 0 <= clevel <= 9

    if name == "gzip":
        level = config.get("level")
        if not isinstance(level, int) or isinstance(level, bool):
            return False
        return 0 <= level <= 9

    if name in ("sharding_indexed",):
        chunk_shape = config.get("chunk_shape")
        if not isinstance(chunk_shape, (list, tuple)):
            return False
        return all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in chunk_shape)

    # regular chunk_grid is not a codec but may be checked via the same predicate
    if name == "regular":
        chunk_shape = config.get("chunk_shape")
        if not isinstance(chunk_shape, (list, tuple)):
            return False
        return all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in chunk_shape)

    return True


def is_valid_ndim_match(shape: object, chunk_grid: object) -> bool:
    """Return True iff the array shape ndim equals the chunk_grid ndim.

    Only inspects the ``regular`` grid (the only one with an unambiguous
    per-dimension chunk_shape in the core spec).  For all other grid names
    the predicate returns True (nothing to check from this oracle).
    """
    if not isinstance(shape, (list, tuple)):
        return False
    ndim = len(shape)
    if not isinstance(chunk_grid, dict):
        return True  # can't determine — not our check
    name = chunk_grid.get("name")
    if name != "regular":
        return True
    config = chunk_grid.get("configuration")
    if not isinstance(config, dict):
        return True
    chunk_shape = config.get("chunk_shape")
    if not isinstance(chunk_shape, (list, tuple)):
        return True
    return len(chunk_shape) == ndim
