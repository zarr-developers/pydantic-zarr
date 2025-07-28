from __future__ import annotations

import warnings
from importlib.metadata import version
from typing import Any

from packaging.version import Version

ZARR_PYTHON_VERSION = Version(version("zarr"))
DTYPE_EXAMPLES_V2: tuple[tuple[Any, Any], ...]
DTYPE_EXAMPLES_V3: tuple[tuple[Any, Any], ...]

if ZARR_PYTHON_VERSION < Version("3.1.0"):
    DTYPE_EXAMPLES_V2 = (
        ("|b1", True),
        ("|i1", -1),
        ("|i2", -1),
        ("|i4", -1),
        ("|i8", -1),
        ("|u1", 1),
        ("<u2", 1),
        ("<u4", 1),
        ("<u8", 1),
        ("<f2", 1.0),
        ("<f4", 1.0),
        ("<f8", 1.0),
        ("<c8", [1.0, 1.0]),
        ("<c16", [1.0, 10]),
        ("<U10", "abcdefghij"),
        ("|O", "hi"),
        ("|V10", "AAAAAAAAAAAAAA=="),
        ("|S10", "AAAAAAAAAAAAAA=="),
        ([("a", "<i4"), ("b", "<f2")], "AAAAAAAA"),
        ("<M8[10s]", "NaT"),
        ("<m8[10s]", "NaT"),
    )
    DTYPE_EXAMPLES_V3 = (
        ("bool", True),
        ("int8", -1),
        ("int16", -1),
        ("int32", -1),
        ("int64", -1),
        ("uint8", 1),
        ("uint16", 1),
        ("uint32", 1),
        ("uint64", 1),
        ("float16", 1.0),
        ("float32", 1.0),
        ("float64", 1.0),
        ("complex64", [1, 1]),
        ("complex128", [1, 1]),
        ("str", "hi"),
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

    v2_examples = []
    v3_examples = []
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
            (
                dt.to_json(zarr_format=2)["name"],
                dt.to_json_scalar(dt.default_scalar(), zarr_format=2),
            )
        )
        # Suppress the userwarning emitted when creating off-spec dtypes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            v3_examples.append(
                (dt.to_json(zarr_format=3), dt.to_json_scalar(dt.default_scalar(), zarr_format=3))
            )

    DTYPE_EXAMPLES_V2 = tuple(v2_examples)
    DTYPE_EXAMPLES_V3 = tuple(v3_examples)
