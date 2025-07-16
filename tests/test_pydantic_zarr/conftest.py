from __future__ import annotations

from typing import Any

from zarr.core.dtype import (
    DateTime64,
    FixedLengthUTF32,
    Float16,
    Int32,
    NullTerminatedBytes,
    RawBytes,
    Structured,
    TimeDelta64,
    ZDType,
    data_type_registry,
)

DTYPE_EXAMPLES: tuple[ZDType[Any, Any], ...] = ()

for dtype_cls in data_type_registry.contents.values():
    if dtype_cls in (DateTime64, TimeDelta64):
        DTYPE_EXAMPLES += (dtype_cls(unit="s", scale_factor=10),)
    elif dtype_cls in (FixedLengthUTF32, RawBytes, NullTerminatedBytes):
        DTYPE_EXAMPLES += (dtype_cls(length=10),)
    elif dtype_cls == Structured:
        DTYPE_EXAMPLES += (dtype_cls(fields=[("a", Int32()), ("b", Float16())]),)
    else:
        DTYPE_EXAMPLES += (dtype_cls(),)
