"""Tightened per-dtype fill-value types for strict v3 array specs.

zarr-metadata's ``*FillValue`` types are structurally permissive — their
``HexFloat<N>`` members are ``NewType(str)`` with no runtime pattern check, and
their integer/bool/raw members impose no range or type discipline. These
``AfterValidator``-wrapped aliases add the rejection the Zarr spec requires
*without changing the static annotation*, so per-dtype precision is preserved.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator
from zarr_metadata import (
    BoolFillValue,
    Complex64FillValue,
    Complex128FillValue,
    Float16FillValue,
    Float32FillValue,
    Float64FillValue,
    RawBytesFillValue,
)
from zarr_metadata.v3.data_type.float16 import (  # noqa: F401 (parity)
    Float16SpecialFillValue,
    hex_float16,
)
from zarr_metadata.v3.data_type.float32 import (  # noqa: F401 (parity)
    Float32SpecialFillValue,
    hex_float32,
)
from zarr_metadata.v3.data_type.float64 import (  # noqa: F401 (parity)
    Float64SpecialFillValue,
    hex_float64,
)

_FLOAT_SPECIALS = frozenset({"NaN", "Infinity", "-Infinity"})


def _make_float_before_check(hex_validator: Any) -> Any:
    """BeforeValidator: reject booleans and invalid hex strings before pydantic coerces."""

    def _check(value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"boolean is not a valid float fill value, got {value!r}")  # noqa: TRY004
        if isinstance(value, str):
            if value in _FLOAT_SPECIALS:
                return value
            hex_validator(value)  # raises ValueError if not a valid hex float
        return value  # int/float pass through to the base type

    return _check


StrictFloat16Fill = Annotated[
    Float16FillValue, BeforeValidator(_make_float_before_check(hex_float16))
]
StrictFloat32Fill = Annotated[
    Float32FillValue, BeforeValidator(_make_float_before_check(hex_float32))
]
StrictFloat64Fill = Annotated[
    Float64FillValue, BeforeValidator(_make_float_before_check(hex_float64))
]

# (min, max) per integer dtype — inlined fixed constants, no numpy needed at runtime
_INT_RANGE: dict[str, tuple[int, int]] = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}


def _make_int_check(lo: int, hi: int) -> Any:
    def _check(value: Any) -> Any:
        # BeforeValidator: sees the RAW input before int|float coercion.
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid integer fill value")  # noqa: TRY004
        if isinstance(value, int):
            if not (lo <= value <= hi):
                raise ValueError(f"integer fill {value} out of range [{lo}, {hi}]")
            return value
        if isinstance(value, float):
            if not value.is_integer() or not (lo <= int(value) <= hi):
                raise ValueError(f"float fill {value} is not a whole number in [{lo}, {hi}]")
            return value
        raise ValueError(f"invalid integer fill value: {value!r}")

    return _check


StrictInt8Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["int8"]))]
StrictInt16Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["int16"]))]
StrictInt32Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["int32"]))]
StrictInt64Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["int64"]))]
StrictUint8Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["uint8"]))]
StrictUint16Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["uint16"]))]
StrictUint32Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["uint32"]))]
StrictUint64Fill = Annotated[int | float, BeforeValidator(_make_int_check(*_INT_RANGE["uint64"]))]


def _check_bool(value: Any) -> Any:
    # BeforeValidator: reject anything that isn't a real bool BEFORE pydantic coerces 1 -> True.
    if type(value) is not bool:
        raise ValueError(f"bool fill must be a boolean, got {value!r}")
    return value


StrictBoolFill = Annotated[BoolFillValue, BeforeValidator(_check_bool)]


def _make_complex_check(component_check: Any) -> Any:
    def _check(value: Any) -> Any:
        # BeforeValidator: value is the raw input; require a 2-element sequence and validate each.
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"complex fill must be a 2-tuple, got {value!r}")
        for component in value:
            component_check(component)  # raises on a bad float component
        return value

    return _check


# reuse the float checks for each component
StrictComplex64Fill = Annotated[
    Complex64FillValue, BeforeValidator(_make_complex_check(_make_float_before_check(hex_float32)))
]
StrictComplex128Fill = Annotated[
    Complex128FillValue, BeforeValidator(_make_complex_check(_make_float_before_check(hex_float64)))
]


def _check_raw(value: Any) -> Any:
    # BeforeValidator: require a real tuple or list of in-range, non-bool ints on the RAW
    # input.  JSON arrays deserialise as lists (json.loads) so both forms are valid; pydantic
    # will coerce the list → tuple when it validates the underlying RawBytesFillValue type.
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"raw fill must be a tuple or list, got {value!r}")  # noqa: TRY004
    for b in value:
        if isinstance(b, bool) or not isinstance(b, int) or not (0 <= b <= 255):
            raise ValueError(f"raw fill byte out of range [0, 255]: {b!r}")
    return value


StrictRawFill = Annotated[RawBytesFillValue, BeforeValidator(_check_raw)]
