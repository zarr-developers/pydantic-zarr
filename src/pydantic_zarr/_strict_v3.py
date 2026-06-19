from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from typing import Annotated, Generic, Literal, Union

from pydantic import AfterValidator, Field
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
from pydantic_zarr.v3 import NodeSpec, TAttr, _BaseArraySpec

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


class _StrictBase(_BaseArraySpec[TAttr], Generic[TAttr]):
    attributes: Mapping[str, object] = {}  # type: ignore[assignment]
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: _StrictChunkKeyEncoding
    codecs: tuple[_StrictCodec, ...]


class _BoolArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: BoolDataTypeName
    fill_value: BoolFillValue


class _Int8ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Int8DataTypeName
    fill_value: Int8FillValue


class _Int16ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Int16DataTypeName
    fill_value: Int16FillValue


class _Int32ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Int32DataTypeName
    fill_value: Int32FillValue


class _Int64ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Int64DataTypeName
    fill_value: Int64FillValue


class _Uint8ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Uint8DataTypeName
    fill_value: Uint8FillValue


class _Uint16ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Uint16DataTypeName
    fill_value: Uint16FillValue


class _Uint32ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Uint32DataTypeName
    fill_value: Uint32FillValue


class _Uint64ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Uint64DataTypeName
    fill_value: Uint64FillValue


class _Float16ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Float16DataTypeName
    fill_value: Float16FillValue


class _Float32ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Float32DataTypeName
    fill_value: Float32FillValue


class _Float64ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Float64DataTypeName
    fill_value: Float64FillValue


class _Complex64ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Complex64DataTypeName
    fill_value: Complex64FillValue


class _Complex128ArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: Complex128DataTypeName
    fill_value: Complex128FillValue


class _RawArraySpec(_StrictBase[TAttr], Generic[TAttr]):
    data_type: RawBytesDataTypeName
    fill_value: RawBytesFillValue


_LiteralDtypeSpecs = Annotated[
    Union[
        _BoolArraySpec,
        _Int8ArraySpec,
        _Int16ArraySpec,
        _Int32ArraySpec,
        _Int64ArraySpec,
        _Uint8ArraySpec,
        _Uint16ArraySpec,
        _Uint32ArraySpec,
        _Uint64ArraySpec,
        _Float16ArraySpec,
        _Float32ArraySpec,
        _Float64ArraySpec,
        _Complex64ArraySpec,
        _Complex128ArraySpec,
    ],
    Field(discriminator="data_type"),
]

StrictArraySpec = Union[_LiteralDtypeSpecs, _RawArraySpec]
"""Strict Zarr v3 array spec: data_type and fill_value are coupled, codecs validated per type."""


class StrictGroupSpec(NodeSpec):
    """A Zarr v3 group whose members are recursively strict (StrictArraySpec/StrictGroupSpec)."""

    node_type: Literal["group"] = "group"
    attributes: Mapping[str, object] = {}
    members: Annotated[
        Mapping[str, Union[StrictArraySpec, StrictGroupSpec]] | None,
        AfterValidator(ensure_key_no_path),
    ] = {}


StrictGroupSpec.model_rebuild()
