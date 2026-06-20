"""Tightened per-dtype fill-value types for strict v3 array specs.

zarr-metadata's ``*FillValue`` types are structurally permissive — their
``HexFloat<N>`` members are ``NewType(str)`` with no runtime pattern check, and
their integer/bool/raw members impose no range or type discipline. These
``AfterValidator``-wrapped aliases add the rejection the Zarr spec requires
*without changing the static annotation*, so per-dtype precision is preserved.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator
from zarr_metadata import (
    Float16FillValue,
    Float32FillValue,
    Float64FillValue,
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


def _make_float_check(hex_validator: Any) -> Any:
    def _check(value: Any) -> Any:
        if isinstance(value, str):
            if value in _FLOAT_SPECIALS:
                return value
            hex_validator(value)  # raises ValueError if not a valid hex float
            return value
        return value  # int/float already accepted by the base type

    return _check


StrictFloat16Fill = Annotated[Float16FillValue, AfterValidator(_make_float_check(hex_float16))]
StrictFloat32Fill = Annotated[Float32FillValue, AfterValidator(_make_float_check(hex_float32))]
StrictFloat64Fill = Annotated[Float64FillValue, AfterValidator(_make_float_check(hex_float64))]

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
