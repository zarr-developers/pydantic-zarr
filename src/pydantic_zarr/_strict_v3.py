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

from pydantic import AfterValidator, Field, TypeAdapter, model_validator
from zarr_metadata import (
    BloscCodecMetadata,
    BloscCodecName,
    BoolDataTypeName,
    BoolFillValue,
    BytesCodecMetadata,
    BytesCodecName,
    CastValueCodecMetadata,
    CastValueCodecName,
    Complex64DataTypeName,
    Complex64FillValue,
    Complex128DataTypeName,
    Complex128FillValue,
    Crc32cCodecMetadata,
    Crc32cCodecName,
    DefaultChunkKeyEncodingMetadata,
    Float16DataTypeName,
    Float16FillValue,
    Float32DataTypeName,
    Float32FillValue,
    Float64DataTypeName,
    Float64FillValue,
    GzipCodecMetadata,
    GzipCodecName,
    Int8DataTypeName,
    Int8FillValue,
    Int16DataTypeName,
    Int16FillValue,
    Int32DataTypeName,
    Int32FillValue,
    Int64DataTypeName,
    Int64FillValue,
    JSONValue,
    RawBytesDataTypeName,
    RawBytesFillValue,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
    ShardingIndexedCodecMetadata,
    ShardingIndexedCodecName,
    TransposeCodecMetadata,
    TransposeCodecName,
    Uint8DataTypeName,
    Uint8FillValue,
    Uint16DataTypeName,
    Uint16FillValue,
    Uint32DataTypeName,
    Uint32FillValue,
    Uint64DataTypeName,
    Uint64FillValue,
    V2ChunkKeyEncodingMetadata,
    ZstdCodecMetadata,
    ZstdCodecName,
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
# Core per-dtype classes (15 explicit)
# ---------------------------------------------------------------------------


class CoreBoolArraySpec(_CoreBase):
    data_type: BoolDataTypeName
    fill_value: BoolFillValue


class CoreInt8ArraySpec(_CoreBase):
    data_type: Int8DataTypeName
    fill_value: Int8FillValue


class CoreInt16ArraySpec(_CoreBase):
    data_type: Int16DataTypeName
    fill_value: Int16FillValue


class CoreInt32ArraySpec(_CoreBase):
    data_type: Int32DataTypeName
    fill_value: Int32FillValue


class CoreInt64ArraySpec(_CoreBase):
    data_type: Int64DataTypeName
    fill_value: Int64FillValue


class CoreUint8ArraySpec(_CoreBase):
    data_type: Uint8DataTypeName
    fill_value: Uint8FillValue


class CoreUint16ArraySpec(_CoreBase):
    data_type: Uint16DataTypeName
    fill_value: Uint16FillValue


class CoreUint32ArraySpec(_CoreBase):
    data_type: Uint32DataTypeName
    fill_value: Uint32FillValue


class CoreUint64ArraySpec(_CoreBase):
    data_type: Uint64DataTypeName
    fill_value: Uint64FillValue


class CoreFloat16ArraySpec(_CoreBase):
    data_type: Float16DataTypeName
    fill_value: Float16FillValue


class CoreFloat32ArraySpec(_CoreBase):
    data_type: Float32DataTypeName
    fill_value: Float32FillValue


class CoreFloat64ArraySpec(_CoreBase):
    data_type: Float64DataTypeName
    fill_value: Float64FillValue


class CoreComplex64ArraySpec(_CoreBase):
    data_type: Complex64DataTypeName
    fill_value: Complex64FillValue


class CoreComplex128ArraySpec(_CoreBase):
    data_type: Complex128DataTypeName
    fill_value: Complex128FillValue


class CoreRawArraySpec(_CoreBase):
    data_type: RawBytesDataTypeName
    fill_value: RawBytesFillValue


# ---------------------------------------------------------------------------
# Extra per-dtype classes (15 explicit)
# ---------------------------------------------------------------------------


class ExtraBoolArraySpec(_ExtraBase):
    data_type: BoolDataTypeName
    fill_value: BoolFillValue


class ExtraInt8ArraySpec(_ExtraBase):
    data_type: Int8DataTypeName
    fill_value: Int8FillValue


class ExtraInt16ArraySpec(_ExtraBase):
    data_type: Int16DataTypeName
    fill_value: Int16FillValue


class ExtraInt32ArraySpec(_ExtraBase):
    data_type: Int32DataTypeName
    fill_value: Int32FillValue


class ExtraInt64ArraySpec(_ExtraBase):
    data_type: Int64DataTypeName
    fill_value: Int64FillValue


class ExtraUint8ArraySpec(_ExtraBase):
    data_type: Uint8DataTypeName
    fill_value: Uint8FillValue


class ExtraUint16ArraySpec(_ExtraBase):
    data_type: Uint16DataTypeName
    fill_value: Uint16FillValue


class ExtraUint32ArraySpec(_ExtraBase):
    data_type: Uint32DataTypeName
    fill_value: Uint32FillValue


class ExtraUint64ArraySpec(_ExtraBase):
    data_type: Uint64DataTypeName
    fill_value: Uint64FillValue


class ExtraFloat16ArraySpec(_ExtraBase):
    data_type: Float16DataTypeName
    fill_value: Float16FillValue


class ExtraFloat32ArraySpec(_ExtraBase):
    data_type: Float32DataTypeName
    fill_value: Float32FillValue


class ExtraFloat64ArraySpec(_ExtraBase):
    data_type: Float64DataTypeName
    fill_value: Float64FillValue


class ExtraComplex64ArraySpec(_ExtraBase):
    data_type: Complex64DataTypeName
    fill_value: Complex64FillValue


class ExtraComplex128ArraySpec(_ExtraBase):
    data_type: Complex128DataTypeName
    fill_value: Complex128FillValue


class ExtraRawArraySpec(_ExtraBase):
    data_type: RawBytesDataTypeName
    fill_value: RawBytesFillValue


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

_RAW_DTYPE_RE = re.compile(r"^r\d+$")
_FILL_BY_DTYPE: dict[str, Any] = {
    "bool": BoolFillValue,
    "int8": Int8FillValue,
    "int16": Int16FillValue,
    "int32": Int32FillValue,
    "int64": Int64FillValue,
    "uint8": Uint8FillValue,
    "uint16": Uint16FillValue,
    "uint32": Uint32FillValue,
    "uint64": Uint64FillValue,
    "float16": Float16FillValue,
    "float32": Float32FillValue,
    "float64": Float64FillValue,
    "complex64": Complex64FillValue,
    "complex128": Complex128FillValue,
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
            TypeAdapter(RawBytesFillValue).validate_python(self.fill_value)
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
            TypeAdapter(RawBytesFillValue).validate_python(self.fill_value)
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
