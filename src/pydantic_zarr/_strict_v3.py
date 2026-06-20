"""Strict Zarr v3 array and group specs.

`AnyStrictArraySpec` is the discriminated-union validation target; validate into it
with `TypeAdapter`. `StrictArraySpec` is the directly-constructible single class.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal, Self, Union

from pydantic import AfterValidator, Field, TypeAdapter, model_validator
from zarr_metadata import (
    BloscCodecMetadata,
    BoolDataTypeName,
    BoolFillValue,
    BytesCodecMetadata,
    Complex64DataTypeName,
    Complex64FillValue,
    Complex128DataTypeName,
    Complex128FillValue,
    Crc32cCodecMetadata,
    DefaultChunkKeyEncodingMetadata,
    Float16DataTypeName,
    Float16FillValue,
    Float32DataTypeName,
    Float32FillValue,
    Float64DataTypeName,
    Float64FillValue,
    GzipCodecMetadata,
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
    RegularChunkGridMetadata,
    ScaleOffsetCodecMetadata,
    ShardingIndexedCodecMetadata,
    TransposeCodecMetadata,
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
)

from pydantic_zarr.core import ensure_key_no_path
from pydantic_zarr.v3 import NodeSpec, _BaseArraySpec

_StrictCodec = (
    BloscCodecMetadata
    | BytesCodecMetadata
    | Crc32cCodecMetadata
    | GzipCodecMetadata
    | ScaleOffsetCodecMetadata
    | ShardingIndexedCodecMetadata
    | TransposeCodecMetadata
    | ZstdCodecMetadata
    | str
)
_StrictChunkKeyEncoding = DefaultChunkKeyEncodingMetadata | V2ChunkKeyEncodingMetadata


class _StrictBase(_BaseArraySpec[Mapping[str, object]]):
    attributes: Mapping[str, object] = {}
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: _StrictChunkKeyEncoding
    codecs: tuple[_StrictCodec, ...]


class BoolArraySpec(_StrictBase):
    data_type: BoolDataTypeName
    fill_value: BoolFillValue


class Int8ArraySpec(_StrictBase):
    data_type: Int8DataTypeName
    fill_value: Int8FillValue


class Int16ArraySpec(_StrictBase):
    data_type: Int16DataTypeName
    fill_value: Int16FillValue


class Int32ArraySpec(_StrictBase):
    data_type: Int32DataTypeName
    fill_value: Int32FillValue


class Int64ArraySpec(_StrictBase):
    data_type: Int64DataTypeName
    fill_value: Int64FillValue


class Uint8ArraySpec(_StrictBase):
    data_type: Uint8DataTypeName
    fill_value: Uint8FillValue


class Uint16ArraySpec(_StrictBase):
    data_type: Uint16DataTypeName
    fill_value: Uint16FillValue


class Uint32ArraySpec(_StrictBase):
    data_type: Uint32DataTypeName
    fill_value: Uint32FillValue


class Uint64ArraySpec(_StrictBase):
    data_type: Uint64DataTypeName
    fill_value: Uint64FillValue


class Float16ArraySpec(_StrictBase):
    data_type: Float16DataTypeName
    fill_value: Float16FillValue


class Float32ArraySpec(_StrictBase):
    data_type: Float32DataTypeName
    fill_value: Float32FillValue


class Float64ArraySpec(_StrictBase):
    data_type: Float64DataTypeName
    fill_value: Float64FillValue


class Complex64ArraySpec(_StrictBase):
    data_type: Complex64DataTypeName
    fill_value: Complex64FillValue


class Complex128ArraySpec(_StrictBase):
    data_type: Complex128DataTypeName
    fill_value: Complex128FillValue


class RawArraySpec(_StrictBase):
    data_type: RawBytesDataTypeName
    fill_value: RawBytesFillValue


_LiteralDtypeSpecs = Annotated[
    Union[
        BoolArraySpec,
        Int8ArraySpec,
        Int16ArraySpec,
        Int32ArraySpec,
        Int64ArraySpec,
        Uint8ArraySpec,
        Uint16ArraySpec,
        Uint32ArraySpec,
        Uint64ArraySpec,
        Float16ArraySpec,
        Float32ArraySpec,
        Float64ArraySpec,
        Complex64ArraySpec,
        Complex128ArraySpec,
    ],
    Field(discriminator="data_type"),
]

AnyStrictArraySpec = Union[_LiteralDtypeSpecs, RawArraySpec]
"""Strict Zarr v3 array spec union: validate into it with TypeAdapter.

data_type and fill_value are coupled; codecs are validated per type.
"""

_RAW_DTYPE_RE = re.compile(r"^r\d+$")
_FILL_BY_DTYPE = {
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


class StrictArraySpec(_StrictBase):
    """A directly-constructible strict v3 array spec.

    `fill_value` is annotated loosely (`JSONValue`) but validated at runtime
    against the per-`data_type` fill-value type. An unrecognized `data_type`
    is rejected. For static per-dtype `fill_value` typing, use the public
    per-dtype classes (e.g. `Float64ArraySpec`) or validate into
    `AnyStrictArraySpec`.
    """

    data_type: str
    fill_value: JSONValue

    @model_validator(mode="after")
    def _validate_fill_matches_dtype(self) -> Self:
        ft = _FILL_BY_DTYPE.get(self.data_type)
        if ft is not None:
            TypeAdapter(ft).validate_python(self.fill_value)
        elif _RAW_DTYPE_RE.match(self.data_type):
            TypeAdapter(RawBytesFillValue).validate_python(self.fill_value)
        else:
            raise ValueError(f"Unrecognized strict data_type: {self.data_type!r}")
        return self


class StrictGroupSpec(NodeSpec):
    """A Zarr v3 group whose members are recursively strict (AnyStrictArraySpec/StrictGroupSpec)."""

    node_type: Literal["group"] = "group"
    attributes: Mapping[str, object] = {}
    members: Annotated[
        Mapping[str, Union[AnyStrictArraySpec, StrictGroupSpec]] | None,
        AfterValidator(ensure_key_no_path),
    ] = {}


StrictGroupSpec.model_rebuild()
