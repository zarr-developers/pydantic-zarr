"""Strict Zarr v3 array and group specs — Core and Extra families.

**Core family** (`CoreArraySpec`, `Core<Dtype>ArraySpec`, `AnyCoreArraySpec`, `CoreGroupSpec`):
  - Chunk grid: `RegularChunkGridMetadata` only.
  - Codecs: 7 known object types + corresponding name literals (no arbitrary str).

**Extra family** (`ExtraArraySpec`, `Extra<Dtype>ArraySpec`, `AnyExtraArraySpec`, `ExtraGroupSpec`):
  - Chunk grid: `RegularChunkGridMetadata | RectilinearChunkGridMetadata`.
  - Codecs: 9 known object types + corresponding name literals (adds ScaleOffset + CastValue).

Use `CoreArraySpec`/`ExtraArraySpec` for direct construction (runtime fill/dtype coupling).
Use `TypeAdapter(AnyCoreArraySpec)` / `TypeAdapter(AnyExtraArraySpec)` to validate documents
into the per-dtype public classes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Self, Union

from pydantic import AfterValidator, BeforeValidator, Field, TypeAdapter, model_validator
from zarr_metadata import (
    BoolDataTypeName,
    Complex64DataTypeName,
    Complex128DataTypeName,
    DefaultChunkKeyEncodingMetadata,
    Float16DataTypeName,
    Float32DataTypeName,
    Float64DataTypeName,
    Int8DataTypeName,
    Int16DataTypeName,
    Int32DataTypeName,
    Int64DataTypeName,
    JSONValue,
    RawBytesDataTypeName,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
    Uint8DataTypeName,
    Uint16DataTypeName,
    Uint32DataTypeName,
    Uint64DataTypeName,
    V2ChunkKeyEncodingMetadata,
)

from pydantic_zarr._strict_fill import (
    StrictBoolFill,
    StrictComplex64Fill,
    StrictComplex128Fill,
    StrictFloat16Fill,
    StrictFloat32Fill,
    StrictFloat64Fill,
    StrictInt8Fill,
    StrictInt16Fill,
    StrictInt32Fill,
    StrictInt64Fill,
    StrictRawFill,
    StrictUint8Fill,
    StrictUint16Fill,
    StrictUint32Fill,
    StrictUint64Fill,
)
from pydantic_zarr.core import ensure_key_no_path
from pydantic_zarr.strict.v3._cross_field import check_array_consistency
from pydantic_zarr.strict.v3._pipeline import validate_pipeline
from pydantic_zarr.strict.v3.chunk_grid import GRID_VALIDATE
from pydantic_zarr.strict.v3.codec import CODEC_VALIDATE, _CoreCodec, _ExtraCodec
from pydantic_zarr.v3 import NodeSpec, _BaseArraySpec

# ---------------------------------------------------------------------------
# AUTO sentinel for ergonomic default construction
# ---------------------------------------------------------------------------


class _Auto(Enum):
    token = 0


AUTO = _Auto.token

_DEFAULT_FILL_BY_DTYPE: dict[str, Any] = {
    "bool": False,
    "int8": 0,
    "int16": 0,
    "int32": 0,
    "int64": 0,
    "uint8": 0,
    "uint16": 0,
    "uint32": 0,
    "uint64": 0,
    "float16": "NaN",
    "float32": "NaN",
    "float64": "NaN",
    "complex64": ("NaN", "NaN"),
    "complex128": ("NaN", "NaN"),
}


_DEFAULT_INNER_CHUNK_CODECS: tuple[dict[str, Any], ...] = (
    {"name": "bytes", "configuration": {"endian": "little"}},
)
_DEFAULT_INDEX_CODECS: tuple[dict[str, Any], ...] = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "crc32c"},
)


def _check_divides(outer: tuple[int, ...], inner: tuple[int, ...]) -> None:
    """Raise ValueError if inner does not evenly divide outer (rank or remainder)."""
    if len(outer) != len(inner):
        raise ValueError(
            f"inner_chunk_shape {inner} has rank {len(inner)} but outer_chunk_shape "
            f"{outer} has rank {len(outer)}; ranks must match"
        )
    non_positive = [i for i in range(len(inner)) if inner[i] <= 0]
    if non_positive:
        raise ValueError(
            f"inner_chunk_shape {inner} contains non-positive dimension(s) at indices "
            f"{non_positive}; all dimensions must be positive"
        )
    bad = [i for i in range(len(outer)) if outer[i] % inner[i] != 0]
    if bad:
        raise ValueError(
            f"inner_chunk_shape {inner} does not evenly divide outer_chunk_shape {outer}"
        )


def _build_sharding_codec(
    inner_chunk_shape: tuple[int, ...],
    inner_chunk_codecs: Any,
    index_codecs: Any,
) -> dict[str, Any]:
    """Build a sharding_indexed codec dict from validated inputs."""
    inner_codecs_resolved: Any = (
        _DEFAULT_INNER_CHUNK_CODECS if inner_chunk_codecs is AUTO else inner_chunk_codecs
    )
    index_codecs_resolved: Any = _DEFAULT_INDEX_CODECS if index_codecs is AUTO else index_codecs
    return {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": tuple(inner_chunk_shape),
            "codecs": tuple(inner_codecs_resolved),
            "index_codecs": tuple(index_codecs_resolved),
        },
    }


def _resolve_strict_init(
    *,
    shape: Any,
    data_type: Any,
    fill_value: Any,
    chunk_grid: Any,
    chunk_key_encoding: Any,
    codecs: Any,
    attributes: Any,
    default_data_type: str,
) -> dict[str, Any]:
    dt = "float64" if data_type is AUTO else data_type
    if default_data_type:  # per-dtype subclass fixes the dtype
        dt = default_data_type
    shape_v = () if shape is AUTO else shape
    out: dict[str, Any] = {
        "shape": shape_v,
        "data_type": dt,
        "attributes": {} if attributes is AUTO else attributes,
        "fill_value": (_DEFAULT_FILL_BY_DTYPE.get(dt, 0) if fill_value is AUTO else fill_value),
        "chunk_grid": (
            {"name": "regular", "configuration": {"chunk_shape": shape_v}}
            if chunk_grid is AUTO
            else chunk_grid
        ),
        "chunk_key_encoding": (
            {"name": "default", "configuration": {"separator": "/"}}
            if chunk_key_encoding is AUTO
            else chunk_key_encoding
        ),
        "codecs": (
            ({"name": "bytes", "configuration": {"endian": "little"}},)
            if codecs is AUTO
            else codecs
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Internal validators: run per-element validate_<x> during pydantic parsing
# ---------------------------------------------------------------------------


def _validate_codec_internal(codec: Any) -> Any:
    if isinstance(codec, str):
        name: str | None = codec
    elif isinstance(codec, dict):
        raw = codec.get("name")
        name = raw if isinstance(raw, str) else None
    else:
        name = None
    if name is not None:
        fn = CODEC_VALIDATE.get(name)
        if fn is not None and isinstance(codec, dict):
            fn(codec)
    return codec


def _validate_grid_internal(grid: Any) -> Any:
    if isinstance(grid, dict):
        raw = grid.get("name")
        name: str | None = raw if isinstance(raw, str) else None
    else:
        name = None
    if name is not None:
        fn = GRID_VALIDATE.get(name)
        if fn is not None:
            fn(grid)
    return grid


# ---------------------------------------------------------------------------
# Codec vocabulary (imported from per-element package; _CoreCodec, _ExtraCodec
# defined there — identical membership, kept in sync)
# ---------------------------------------------------------------------------

_ChunkKeyEncoding = DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata

# ---------------------------------------------------------------------------
# Family base classes
# ---------------------------------------------------------------------------


class _CoreBase(_BaseArraySpec[Mapping[str, object]]):
    attributes: Mapping[str, object] = {}
    chunk_grid: Annotated[RegularChunkGridMetadata, AfterValidator(_validate_grid_internal)]
    chunk_key_encoding: _ChunkKeyEncoding
    codecs: tuple[Annotated[_CoreCodec, AfterValidator(_validate_codec_internal)], ...]

    @model_validator(mode="after")
    def _validate_array_consistency(self) -> Self:
        errs = check_array_consistency(
            shape=self.shape,
            chunk_grid=self.chunk_grid,
            codecs=self.codecs,
            dimension_names=self.dimension_names,
        )
        dt: Any = getattr(self, "data_type", "")
        errs += validate_pipeline(dt if isinstance(dt, str) else "", self.codecs)
        if errs:
            raise ValueError("; ".join(errs))
        return self

    @classmethod
    def create(
        cls,
        *,
        shape: tuple[int, ...] | _Auto = AUTO,
        data_type: Any = AUTO,
        fill_value: Any = AUTO,
        chunk_grid: Any = AUTO,
        chunk_key_encoding: Any = AUTO,
        codecs: Any = AUTO,
        attributes: Mapping[str, object] | _Auto = AUTO,
        **kwargs: Any,
    ) -> Self:
        resolved = _resolve_strict_init(
            shape=shape,
            data_type=data_type,
            fill_value=fill_value,
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            attributes=attributes,
            default_data_type=getattr(cls, "_default_data_type", ""),
        )
        return cls(**resolved, **kwargs)

    @classmethod
    def create_with_sharding(
        cls,
        *,
        outer_chunk_shape: tuple[int, ...],
        inner_chunk_shape: tuple[int, ...],
        inner_chunk_codecs: Any = AUTO,
        index_codecs: Any = AUTO,
        shape: tuple[int, ...] | _Auto = AUTO,
        data_type: Any = AUTO,
        fill_value: Any = AUTO,
        chunk_key_encoding: Any = AUTO,
        attributes: Mapping[str, object] | _Auto = AUTO,
        **kwargs: Any,
    ) -> Self:
        _check_divides(outer_chunk_shape, inner_chunk_shape)
        sharding_codec = _build_sharding_codec(inner_chunk_shape, inner_chunk_codecs, index_codecs)
        chunk_grid: dict[str, Any] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(outer_chunk_shape)},
        }
        return cls.create(
            shape=shape,
            data_type=data_type,
            fill_value=fill_value,
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            codecs=(sharding_codec,),
            attributes=attributes,
            **kwargs,
        )

    @classmethod
    def from_array(
        cls,
        array: Any,
        *,
        attributes: Any = "auto",
        chunk_grid: Any = "auto",
        chunk_key_encoding: Any = "auto",
        fill_value: Any = "auto",
        codecs: Any = "auto",
        dimension_names: Any = "auto",
    ) -> Self:
        from pydantic_zarr.v3 import (
            auto_attributes,
            auto_chunk_grid,
            auto_chunk_key_encoding,
            auto_codecs,
            auto_dimension_names,
            auto_fill_value,
            parse_dtype_v3,
        )

        doc: dict[str, Any] = {
            "shape": tuple(array.shape),
            "data_type": parse_dtype_v3(array.dtype),
            "attributes": auto_attributes(array) if attributes == "auto" else attributes,
            "chunk_grid": auto_chunk_grid(array) if chunk_grid == "auto" else chunk_grid,
            "chunk_key_encoding": (
                auto_chunk_key_encoding(array)
                if chunk_key_encoding == "auto"
                else chunk_key_encoding
            ),
            "fill_value": auto_fill_value(array) if fill_value == "auto" else fill_value,
            "codecs": auto_codecs(array) if codecs == "auto" else codecs,
            "dimension_names": (
                auto_dimension_names(array) if dimension_names == "auto" else dimension_names
            ),
        }
        return cls.model_validate(doc)


class _ExtraBase(_BaseArraySpec[Mapping[str, object]]):
    attributes: Mapping[str, object] = {}
    chunk_grid: Annotated[
        RegularChunkGridMetadata | RectilinearChunkGridMetadata,
        AfterValidator(_validate_grid_internal),
    ]
    chunk_key_encoding: _ChunkKeyEncoding
    codecs: tuple[Annotated[_ExtraCodec, AfterValidator(_validate_codec_internal)], ...]

    @model_validator(mode="after")
    def _validate_array_consistency(self) -> Self:
        errs = check_array_consistency(
            shape=self.shape,
            chunk_grid=self.chunk_grid,
            codecs=self.codecs,
            dimension_names=self.dimension_names,
        )
        dt: Any = getattr(self, "data_type", "")
        errs += validate_pipeline(dt if isinstance(dt, str) else "", self.codecs)
        if errs:
            raise ValueError("; ".join(errs))
        return self

    @classmethod
    def create(
        cls,
        *,
        shape: tuple[int, ...] | _Auto = AUTO,
        data_type: Any = AUTO,
        fill_value: Any = AUTO,
        chunk_grid: Any = AUTO,
        chunk_key_encoding: Any = AUTO,
        codecs: Any = AUTO,
        attributes: Mapping[str, object] | _Auto = AUTO,
        **kwargs: Any,
    ) -> Self:
        resolved = _resolve_strict_init(
            shape=shape,
            data_type=data_type,
            fill_value=fill_value,
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            attributes=attributes,
            default_data_type=getattr(cls, "_default_data_type", ""),
        )
        return cls(**resolved, **kwargs)

    @classmethod
    def create_with_sharding(
        cls,
        *,
        outer_chunk_shape: tuple[int, ...],
        inner_chunk_shape: tuple[int, ...],
        inner_chunk_codecs: Any = AUTO,
        index_codecs: Any = AUTO,
        shape: tuple[int, ...] | _Auto = AUTO,
        data_type: Any = AUTO,
        fill_value: Any = AUTO,
        chunk_key_encoding: Any = AUTO,
        attributes: Mapping[str, object] | _Auto = AUTO,
        **kwargs: Any,
    ) -> Self:
        _check_divides(outer_chunk_shape, inner_chunk_shape)
        sharding_codec = _build_sharding_codec(inner_chunk_shape, inner_chunk_codecs, index_codecs)
        chunk_grid: dict[str, Any] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(outer_chunk_shape)},
        }
        return cls.create(
            shape=shape,
            data_type=data_type,
            fill_value=fill_value,
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            codecs=(sharding_codec,),
            attributes=attributes,
            **kwargs,
        )

    @classmethod
    def from_array(
        cls,
        array: Any,
        *,
        attributes: Any = "auto",
        chunk_grid: Any = "auto",
        chunk_key_encoding: Any = "auto",
        fill_value: Any = "auto",
        codecs: Any = "auto",
        dimension_names: Any = "auto",
    ) -> Self:
        from pydantic_zarr.v3 import (
            auto_attributes,
            auto_chunk_grid,
            auto_chunk_key_encoding,
            auto_codecs,
            auto_dimension_names,
            auto_fill_value,
            parse_dtype_v3,
        )

        doc: dict[str, Any] = {
            "shape": tuple(array.shape),
            "data_type": parse_dtype_v3(array.dtype),
            "attributes": auto_attributes(array) if attributes == "auto" else attributes,
            "chunk_grid": auto_chunk_grid(array) if chunk_grid == "auto" else chunk_grid,
            "chunk_key_encoding": (
                auto_chunk_key_encoding(array)
                if chunk_key_encoding == "auto"
                else chunk_key_encoding
            ),
            "fill_value": auto_fill_value(array) if fill_value == "auto" else fill_value,
            "codecs": auto_codecs(array) if codecs == "auto" else codecs,
            "dimension_names": (
                auto_dimension_names(array) if dimension_names == "auto" else dimension_names
            ),
        }
        return cls.model_validate(doc)


# ---------------------------------------------------------------------------
# Raw dtype name validator
# ---------------------------------------------------------------------------

_RAW_DTYPE_RE = re.compile(r"^r\d+$")


def _ensure_raw_dtype_name(value: str) -> str:
    """Validate a raw ``r<N>`` data type name (N a positive integer)."""
    if not _RAW_DTYPE_RE.match(value):
        raise ValueError("raw data_type must match 'r<N>', got " + repr(value))
    return value


_RawDataTypeName = Annotated[RawBytesDataTypeName, BeforeValidator(_ensure_raw_dtype_name)]

# ---------------------------------------------------------------------------
# Core per-dtype classes (15 explicit)
# ---------------------------------------------------------------------------


class CoreBoolArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "bool"
    data_type: BoolDataTypeName = "bool"
    fill_value: StrictBoolFill


class CoreInt8ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "int8"
    data_type: Int8DataTypeName = "int8"
    fill_value: StrictInt8Fill


class CoreInt16ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "int16"
    data_type: Int16DataTypeName = "int16"
    fill_value: StrictInt16Fill


class CoreInt32ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "int32"
    data_type: Int32DataTypeName = "int32"
    fill_value: StrictInt32Fill


class CoreInt64ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "int64"
    data_type: Int64DataTypeName = "int64"
    fill_value: StrictInt64Fill


class CoreUint8ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "uint8"
    data_type: Uint8DataTypeName = "uint8"
    fill_value: StrictUint8Fill


class CoreUint16ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "uint16"
    data_type: Uint16DataTypeName = "uint16"
    fill_value: StrictUint16Fill


class CoreUint32ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "uint32"
    data_type: Uint32DataTypeName = "uint32"
    fill_value: StrictUint32Fill


class CoreUint64ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "uint64"
    data_type: Uint64DataTypeName = "uint64"
    fill_value: StrictUint64Fill


class CoreFloat16ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "float16"
    data_type: Float16DataTypeName = "float16"
    fill_value: StrictFloat16Fill


class CoreFloat32ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "float32"
    data_type: Float32DataTypeName = "float32"
    fill_value: StrictFloat32Fill


class CoreFloat64ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "float64"
    data_type: Float64DataTypeName = "float64"
    fill_value: StrictFloat64Fill


class CoreComplex64ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "complex64"
    data_type: Complex64DataTypeName = "complex64"
    fill_value: StrictComplex64Fill


class CoreComplex128ArraySpec(_CoreBase):
    _default_data_type: ClassVar[str] = "complex128"
    data_type: Complex128DataTypeName = "complex128"
    fill_value: StrictComplex128Fill


class CoreRawArraySpec(_CoreBase):
    data_type: _RawDataTypeName
    fill_value: StrictRawFill

    @model_validator(mode="after")
    def _validate_raw_fill_length(self) -> Self:
        nbytes = int(re.fullmatch(r"r(\d+)", self.data_type).group(1)) // 8  # type: ignore[union-attr]
        if len(self.fill_value) != nbytes:
            raise ValueError(
                f"fill_value for {self.data_type!r} must be a {nbytes}-element tuple, "
                f"got {len(self.fill_value)} elements"
            )
        return self


# ---------------------------------------------------------------------------
# Extra per-dtype classes (15 explicit)
# ---------------------------------------------------------------------------


class ExtraBoolArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "bool"
    data_type: BoolDataTypeName = "bool"
    fill_value: StrictBoolFill


class ExtraInt8ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "int8"
    data_type: Int8DataTypeName = "int8"
    fill_value: StrictInt8Fill


class ExtraInt16ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "int16"
    data_type: Int16DataTypeName = "int16"
    fill_value: StrictInt16Fill


class ExtraInt32ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "int32"
    data_type: Int32DataTypeName = "int32"
    fill_value: StrictInt32Fill


class ExtraInt64ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "int64"
    data_type: Int64DataTypeName = "int64"
    fill_value: StrictInt64Fill


class ExtraUint8ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "uint8"
    data_type: Uint8DataTypeName = "uint8"
    fill_value: StrictUint8Fill


class ExtraUint16ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "uint16"
    data_type: Uint16DataTypeName = "uint16"
    fill_value: StrictUint16Fill


class ExtraUint32ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "uint32"
    data_type: Uint32DataTypeName = "uint32"
    fill_value: StrictUint32Fill


class ExtraUint64ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "uint64"
    data_type: Uint64DataTypeName = "uint64"
    fill_value: StrictUint64Fill


class ExtraFloat16ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "float16"
    data_type: Float16DataTypeName = "float16"
    fill_value: StrictFloat16Fill


class ExtraFloat32ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "float32"
    data_type: Float32DataTypeName = "float32"
    fill_value: StrictFloat32Fill


class ExtraFloat64ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "float64"
    data_type: Float64DataTypeName = "float64"
    fill_value: StrictFloat64Fill


class ExtraComplex64ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "complex64"
    data_type: Complex64DataTypeName = "complex64"
    fill_value: StrictComplex64Fill


class ExtraComplex128ArraySpec(_ExtraBase):
    _default_data_type: ClassVar[str] = "complex128"
    data_type: Complex128DataTypeName = "complex128"
    fill_value: StrictComplex128Fill


class ExtraRawArraySpec(_ExtraBase):
    data_type: _RawDataTypeName
    fill_value: StrictRawFill

    @model_validator(mode="after")
    def _validate_raw_fill_length(self) -> Self:
        nbytes = int(re.fullmatch(r"r(\d+)", self.data_type).group(1)) // 8  # type: ignore[union-attr]
        if len(self.fill_value) != nbytes:
            raise ValueError(
                f"fill_value for {self.data_type!r} must be a {nbytes}-element tuple, "
                f"got {len(self.fill_value)} elements"
            )
        return self


# ---------------------------------------------------------------------------
# Union types (discriminated + hybrid for raw)
# ---------------------------------------------------------------------------

_CoreLiteralDtypes = Annotated[
    Union[
        CoreBoolArraySpec,
        CoreInt8ArraySpec,
        CoreInt16ArraySpec,
        CoreInt32ArraySpec,
        CoreInt64ArraySpec,
        CoreUint8ArraySpec,
        CoreUint16ArraySpec,
        CoreUint32ArraySpec,
        CoreUint64ArraySpec,
        CoreFloat16ArraySpec,
        CoreFloat32ArraySpec,
        CoreFloat64ArraySpec,
        CoreComplex64ArraySpec,
        CoreComplex128ArraySpec,
    ],
    Field(discriminator="data_type"),
]

AnyCoreArraySpec = Union[_CoreLiteralDtypes, CoreRawArraySpec]
"""Core strict Zarr v3 array spec union.

Validate into it with ``TypeAdapter(AnyCoreArraySpec).validate_python(doc)``.
``data_type`` and ``fill_value`` are coupled per type.  Codec strings must be
known Core codec names; ``chunk_grid`` must be ``RegularChunkGridMetadata``.
"""

_ExtraLiteralDtypes = Annotated[
    Union[
        ExtraBoolArraySpec,
        ExtraInt8ArraySpec,
        ExtraInt16ArraySpec,
        ExtraInt32ArraySpec,
        ExtraInt64ArraySpec,
        ExtraUint8ArraySpec,
        ExtraUint16ArraySpec,
        ExtraUint32ArraySpec,
        ExtraUint64ArraySpec,
        ExtraFloat16ArraySpec,
        ExtraFloat32ArraySpec,
        ExtraFloat64ArraySpec,
        ExtraComplex64ArraySpec,
        ExtraComplex128ArraySpec,
    ],
    Field(discriminator="data_type"),
]

AnyExtraArraySpec = Union[_ExtraLiteralDtypes, ExtraRawArraySpec]
"""Extra strict Zarr v3 array spec union.

Like ``AnyCoreArraySpec`` but also accepts ``RectilinearChunkGridMetadata``
and the additional ``ScaleOffsetCodecMetadata``/``CastValueCodecMetadata``
codec types (and their name literals).
"""

# ---------------------------------------------------------------------------
# Fill-value lookup (shared between Core and Extra)
# ---------------------------------------------------------------------------

_FILL_BY_DTYPE: dict[str, Any] = {
    "bool": StrictBoolFill,
    "int8": StrictInt8Fill,
    "int16": StrictInt16Fill,
    "int32": StrictInt32Fill,
    "int64": StrictInt64Fill,
    "uint8": StrictUint8Fill,
    "uint16": StrictUint16Fill,
    "uint32": StrictUint32Fill,
    "uint64": StrictUint64Fill,
    "float16": StrictFloat16Fill,
    "float32": StrictFloat32Fill,
    "float64": StrictFloat64Fill,
    "complex64": StrictComplex64Fill,
    "complex128": StrictComplex128Fill,
}

# ---------------------------------------------------------------------------
# Single constructible classes (runtime dtype/fill coupling)
# ---------------------------------------------------------------------------


class CoreArraySpec(_CoreBase):
    """Directly-constructible Core strict v3 array spec.

    ``fill_value`` is annotated loosely (``JSONValue``) but validated at runtime
    against the per-``data_type`` fill-value type.  Unrecognised ``data_type``
    values are rejected.  For static per-dtype ``fill_value`` typing, use the
    public per-dtype classes (e.g. ``CoreFloat64ArraySpec``) or validate into
    ``AnyCoreArraySpec``.

    ``chunk_grid`` must be ``RegularChunkGridMetadata``.
    ``codecs`` may only contain Core codec object types or known Core codec name
    literals (arbitrary strings are rejected).
    """

    data_type: str
    fill_value: JSONValue

    @model_validator(mode="after")
    def _validate_fill(self) -> Self:
        ft = _FILL_BY_DTYPE.get(self.data_type)
        if ft is not None:
            TypeAdapter(ft).validate_python(self.fill_value)
        elif _RAW_DTYPE_RE.match(self.data_type):
            TypeAdapter(StrictRawFill).validate_python(self.fill_value)
            nbytes = int(re.fullmatch(r"r(\d+)", self.data_type).group(1)) // 8  # type: ignore[union-attr]
            if len(self.fill_value) != nbytes:  # type: ignore[arg-type]
                raise ValueError(
                    f"fill_value for {self.data_type!r} must be a {nbytes}-element tuple, "
                    f"got {len(self.fill_value)} elements"  # type: ignore[arg-type]
                )
        else:
            raise ValueError(f"Unrecognized data_type: {self.data_type!r}")
        return self


class ExtraArraySpec(_ExtraBase):
    """Directly-constructible Extra strict v3 array spec.

    Like ``CoreArraySpec`` but ``chunk_grid`` additionally accepts
    ``RectilinearChunkGridMetadata`` and ``codecs`` additionally accept
    ``ScaleOffsetCodecMetadata``, ``CastValueCodecMetadata``, and their name
    literals.
    """

    data_type: str
    fill_value: JSONValue

    @model_validator(mode="after")
    def _validate_fill(self) -> Self:
        ft = _FILL_BY_DTYPE.get(self.data_type)
        if ft is not None:
            TypeAdapter(ft).validate_python(self.fill_value)
        elif _RAW_DTYPE_RE.match(self.data_type):
            TypeAdapter(StrictRawFill).validate_python(self.fill_value)
            nbytes = int(re.fullmatch(r"r(\d+)", self.data_type).group(1)) // 8  # type: ignore[union-attr]
            if len(self.fill_value) != nbytes:  # type: ignore[arg-type]
                raise ValueError(
                    f"fill_value for {self.data_type!r} must be a {nbytes}-element tuple, "
                    f"got {len(self.fill_value)} elements"  # type: ignore[arg-type]
                )
        else:
            raise ValueError(f"Unrecognized data_type: {self.data_type!r}")
        return self


# ---------------------------------------------------------------------------
# Group specs
# ---------------------------------------------------------------------------


class CoreGroupSpec(NodeSpec):
    """Zarr v3 group whose members are recursively Core-strict."""

    node_type: Literal["group"] = "group"
    attributes: Mapping[str, object] = {}
    members: Annotated[
        Mapping[str, Union[AnyCoreArraySpec, CoreGroupSpec]] | None,
        AfterValidator(ensure_key_no_path),
    ] = {}


CoreGroupSpec.model_rebuild()


class ExtraGroupSpec(NodeSpec):
    """Zarr v3 group whose members are recursively Extra-strict."""

    node_type: Literal["group"] = "group"
    attributes: Mapping[str, object] = {}
    members: Annotated[
        Mapping[str, Union[AnyExtraArraySpec, ExtraGroupSpec]] | None,
        AfterValidator(ensure_key_no_path),
    ] = {}


ExtraGroupSpec.model_rebuild()
