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
from typing import Annotated, Any, Literal, Self, Union

from pydantic import AfterValidator, BeforeValidator, Field, TypeAdapter, model_validator
from zarr_metadata import (
    BloscCodecMetadata,
    BloscCodecName,
    BoolDataTypeName,
    BytesCodecMetadata,
    BytesCodecName,
    CastValueCodecMetadata,
    CastValueCodecName,
    Complex64DataTypeName,
    Complex128DataTypeName,
    Crc32cCodecMetadata,
    Crc32cCodecName,
    DefaultChunkKeyEncodingMetadata,
    Float16DataTypeName,
    Float32DataTypeName,
    Float64DataTypeName,
    GzipCodecMetadata,
    GzipCodecName,
    Int8DataTypeName,
    Int16DataTypeName,
    Int32DataTypeName,
    Int64DataTypeName,
    JSONValue,
    RawBytesDataTypeName,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
    ShardingIndexedCodecMetadata,
    ShardingIndexedCodecName,
    TransposeCodecMetadata,
    TransposeCodecName,
    Uint8DataTypeName,
    Uint16DataTypeName,
    Uint32DataTypeName,
    Uint64DataTypeName,
    V2ChunkKeyEncodingMetadata,
    ZstdCodecMetadata,
    ZstdCodecName,
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
from pydantic_zarr.v3 import NodeSpec, _BaseArraySpec

# ---------------------------------------------------------------------------
# Codec vocabulary
# ---------------------------------------------------------------------------

_CoreCodec = (
    BloscCodecMetadata
    | BytesCodecMetadata
    | Crc32cCodecMetadata
    | GzipCodecMetadata
    | ShardingIndexedCodecMetadata
    | TransposeCodecMetadata
    | ZstdCodecMetadata
    | BloscCodecName
    | BytesCodecName
    | Crc32cCodecName
    | GzipCodecName
    | ShardingIndexedCodecName
    | TransposeCodecName
    | ZstdCodecName
)

_ExtraCodec = (
    BloscCodecMetadata
    | BytesCodecMetadata
    | CastValueCodecMetadata
    | Crc32cCodecMetadata
    | GzipCodecMetadata
    | ScaleOffsetCodecMetadata
    | ShardingIndexedCodecMetadata
    | TransposeCodecMetadata
    | ZstdCodecMetadata
    | BloscCodecName
    | BytesCodecName
    | CastValueCodecName
    | Crc32cCodecName
    | GzipCodecName
    | ScaleOffsetCodecName
    | ShardingIndexedCodecName
    | TransposeCodecName
    | ZstdCodecName
)

_ChunkKeyEncoding = DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata

# ---------------------------------------------------------------------------
# Family base classes
# ---------------------------------------------------------------------------


class _CoreBase(_BaseArraySpec[Mapping[str, object]]):
    attributes: Mapping[str, object] = {}
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: _ChunkKeyEncoding
    codecs: tuple[_CoreCodec, ...]


class _ExtraBase(_BaseArraySpec[Mapping[str, object]]):
    attributes: Mapping[str, object] = {}
    chunk_grid: RegularChunkGridMetadata | RectilinearChunkGridMetadata
    chunk_key_encoding: _ChunkKeyEncoding
    codecs: tuple[_ExtraCodec, ...]


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
    data_type: BoolDataTypeName
    fill_value: StrictBoolFill


class CoreInt8ArraySpec(_CoreBase):
    data_type: Int8DataTypeName
    fill_value: StrictInt8Fill


class CoreInt16ArraySpec(_CoreBase):
    data_type: Int16DataTypeName
    fill_value: StrictInt16Fill


class CoreInt32ArraySpec(_CoreBase):
    data_type: Int32DataTypeName
    fill_value: StrictInt32Fill


class CoreInt64ArraySpec(_CoreBase):
    data_type: Int64DataTypeName
    fill_value: StrictInt64Fill


class CoreUint8ArraySpec(_CoreBase):
    data_type: Uint8DataTypeName
    fill_value: StrictUint8Fill


class CoreUint16ArraySpec(_CoreBase):
    data_type: Uint16DataTypeName
    fill_value: StrictUint16Fill


class CoreUint32ArraySpec(_CoreBase):
    data_type: Uint32DataTypeName
    fill_value: StrictUint32Fill


class CoreUint64ArraySpec(_CoreBase):
    data_type: Uint64DataTypeName
    fill_value: StrictUint64Fill


class CoreFloat16ArraySpec(_CoreBase):
    data_type: Float16DataTypeName
    fill_value: StrictFloat16Fill


class CoreFloat32ArraySpec(_CoreBase):
    data_type: Float32DataTypeName
    fill_value: StrictFloat32Fill


class CoreFloat64ArraySpec(_CoreBase):
    data_type: Float64DataTypeName
    fill_value: StrictFloat64Fill


class CoreComplex64ArraySpec(_CoreBase):
    data_type: Complex64DataTypeName
    fill_value: StrictComplex64Fill


class CoreComplex128ArraySpec(_CoreBase):
    data_type: Complex128DataTypeName
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
    data_type: BoolDataTypeName
    fill_value: StrictBoolFill


class ExtraInt8ArraySpec(_ExtraBase):
    data_type: Int8DataTypeName
    fill_value: StrictInt8Fill


class ExtraInt16ArraySpec(_ExtraBase):
    data_type: Int16DataTypeName
    fill_value: StrictInt16Fill


class ExtraInt32ArraySpec(_ExtraBase):
    data_type: Int32DataTypeName
    fill_value: StrictInt32Fill


class ExtraInt64ArraySpec(_ExtraBase):
    data_type: Int64DataTypeName
    fill_value: StrictInt64Fill


class ExtraUint8ArraySpec(_ExtraBase):
    data_type: Uint8DataTypeName
    fill_value: StrictUint8Fill


class ExtraUint16ArraySpec(_ExtraBase):
    data_type: Uint16DataTypeName
    fill_value: StrictUint16Fill


class ExtraUint32ArraySpec(_ExtraBase):
    data_type: Uint32DataTypeName
    fill_value: StrictUint32Fill


class ExtraUint64ArraySpec(_ExtraBase):
    data_type: Uint64DataTypeName
    fill_value: StrictUint64Fill


class ExtraFloat16ArraySpec(_ExtraBase):
    data_type: Float16DataTypeName
    fill_value: StrictFloat16Fill


class ExtraFloat32ArraySpec(_ExtraBase):
    data_type: Float32DataTypeName
    fill_value: StrictFloat32Fill


class ExtraFloat64ArraySpec(_ExtraBase):
    data_type: Float64DataTypeName
    fill_value: StrictFloat64Fill


class ExtraComplex64ArraySpec(_ExtraBase):
    data_type: Complex64DataTypeName
    fill_value: StrictComplex64Fill


class ExtraComplex128ArraySpec(_ExtraBase):
    data_type: Complex128DataTypeName
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
