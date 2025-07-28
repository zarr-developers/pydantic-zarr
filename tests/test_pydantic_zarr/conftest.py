from __future__ import annotations

import warnings
from importlib.metadata import version
from typing import Any

from packaging.version import Version

ZARR_PYTHON_VERSION = Version(version("zarr"))

if ZARR_PYTHON_VERSION < Version("3.1.0"):
    DTYPE_EXAMPLES = ()
    DTYPE_EXAMPLES_V2 = [
        "|b1",
        "|i1",
        "<i2",
        "<i4",
        "<i8",
        "|u1",
        "<u2",
        "<u4",
        "<u8",
        "<f2",
        "<f4",
        "<f8",
        "<c8",
        "<c16",
        "<U10",
        "|O",
        "|V10",
        "|S10",
        [("a", "<i4"), ("b", "<f2")],
        "<M8[10s]",
        "<m8[10s]",
    ]
    DTYPE_EXAMPLES_V3 = [
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
        "str",
    ]
else:
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
    DTYPE_EXAMPLES_V2 = [d.to_json(zarr_format=2)["name"] for d in DTYPE_EXAMPLES]
    # Suppress the userwarning emitted when creating off-spec dtypes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        DTYPE_EXAMPLES_V3 = [d.to_json(zarr_format=3) for d in DTYPE_EXAMPLES]
