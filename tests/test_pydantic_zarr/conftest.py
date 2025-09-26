from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    ZARR_PYTHON_VERSION = Version(version("zarr"))
except PackageNotFoundError:
    ZARR_PYTHON_VERSION = Version("0.0.0")

DTYPE_EXAMPLES_V2: tuple[DTypeExample, ...]
DTYPE_EXAMPLES_V3: tuple[DTypeExample, ...]


@dataclass(frozen=True, slots=True)
class DTypeExample:
    name: object
    fill_value: object


if ZARR_PYTHON_VERSION < Version("3.1.0"):
    DTYPE_EXAMPLES_V2 = (
        DTypeExample("|b1", True),
        DTypeExample("|i1", -1),
        DTypeExample("|i2", -1),
        DTypeExample("|i4", -1),
        DTypeExample("|i8", -1),
        DTypeExample("|u1", 1),
        DTypeExample("<u2", 1),
        DTypeExample("<u4", 1),
        DTypeExample("<u8", 1),
        DTypeExample("<f2", 1.0),
        DTypeExample("<f4", 1.0),
        DTypeExample("<f8", 1.0),
        DTypeExample("<c8", [1.0, 1.0]),
        DTypeExample("<c16", [1.0, 10]),
        DTypeExample("<U10", "abcdefghij"),
        DTypeExample("|O", "hi"),
        DTypeExample("|V10", "AAAAAAAAAAAAAA=="),
        DTypeExample("|S10", "AAAAAAAAAAAAAA=="),
        DTypeExample([("a", "<i4"), ("b", "<f2")], "AAAAAAAA"),
        DTypeExample("<M8[10s]", "NaT"),
        DTypeExample("<m8[10s]", "NaT"),
    )
    DTYPE_EXAMPLES_V3 = (
        DTypeExample("bool", True),
        DTypeExample("int8", -1),
        DTypeExample("int16", -1),
        DTypeExample("int32", -1),
        DTypeExample("int64", -1),
        DTypeExample("uint8", 1),
        DTypeExample("uint16", 1),
        DTypeExample("uint32", 1),
        DTypeExample("uint64", 1),
        DTypeExample("float16", 1.0),
        DTypeExample("float32", 1.0),
        DTypeExample("float64", 1.0),
        DTypeExample("complex64", [1, 1]),
        DTypeExample("complex128", [1, 1]),
        DTypeExample("str", "hi"),
    )
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
        data_type_registry,
    )

    v2_examples: list[DTypeExample] = []
    v3_examples: list[DTypeExample] = []
    for dtype_cls in data_type_registry.contents.values():
        if dtype_cls in (DateTime64, TimeDelta64):
            dt = dtype_cls(unit="s", scale_factor=10)
        elif dtype_cls in (FixedLengthUTF32, RawBytes, NullTerminatedBytes):
            dt = dtype_cls(length=10)
        elif dtype_cls == Structured:
            dt = dtype_cls(fields=[("a", Int32()), ("b", Float16())])
        else:
            dt = dtype_cls()

        v2_examples.append(
            DTypeExample(
                dt.to_json(zarr_format=2)["name"],
                dt.to_json_scalar(dt.default_scalar(), zarr_format=2),
            )
        )
        # Suppress the userwarning emitted when creating off-spec dtypes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            v3_examples.append(
                DTypeExample(
                    dt.to_json(zarr_format=3), dt.to_json_scalar(dt.default_scalar(), zarr_format=3)
                )
            )

    DTYPE_EXAMPLES_V2 = tuple(v2_examples)
    DTYPE_EXAMPLES_V3 = tuple(v3_examples)
