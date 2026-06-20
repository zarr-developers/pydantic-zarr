"""Tightened per-dtype fill-value types for strict v3 array specs.

zarr-metadata's ``*FillValue`` types are structurally permissive — their
``HexFloat<N>`` members are ``NewType(str)`` with no runtime pattern check, and
their integer/bool/raw members impose no range or type discipline. These
``AfterValidator``-wrapped aliases add the rejection the Zarr spec requires
*without changing the static annotation*, so per-dtype precision is preserved.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator  # noqa: F401 (used in later tasks)
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
