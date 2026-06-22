from __future__ import annotations

from zarr_metadata import (
    BloscCodecMetadata,
    BloscCodecName,
    BytesCodecMetadata,
    BytesCodecName,
    CastValueCodecMetadata,
    CastValueCodecName,
    Crc32cCodecMetadata,
    Crc32cCodecName,
    GzipCodecMetadata,
    GzipCodecName,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
    ShardingIndexedCodecMetadata,
    ShardingIndexedCodecName,
    TransposeCodecMetadata,
    TransposeCodecName,
    ZstdCodecMetadata,
    ZstdCodecName,
)

from pydantic_zarr.strict.v3._registry import element_name
from pydantic_zarr.strict.v3.codec._spec import CodecSpec

from . import blosc, cast_value, crc32c, gzip, scale_offset, sharding_indexed, transpose, zstd
from . import bytes as _bytes

# Re-export builders
bytes_codec = _bytes.bytes_codec  # bytes module shadows builtin; use alias
crc32c_codec = crc32c.crc32c
gzip_codec = gzip.gzip
zstd_codec = zstd.zstd
blosc_codec = blosc.blosc
transpose_codec = transpose.transpose
sharding_indexed_codec = sharding_indexed.sharding_indexed
scale_offset_codec = scale_offset.scale_offset
cast_value_codec = cast_value.cast_value

# Per-codec spec registry (replaces the old CODEC_* parallel maps)
_MODULES = (
    _bytes,
    crc32c,
    gzip,
    zstd,
    blosc,
    transpose,
    sharding_indexed,
    scale_offset,
    cast_value,
)
CODECS: dict[str, CodecSpec] = {m.SPEC.name: m.SPEC for m in _MODULES}


def codec_spec_for(c: object) -> CodecSpec | None:
    name = element_name(c)
    return CODECS.get(name) if name is not None else None


# Unions MOVED from _strict_v3.py (identical membership)
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

__all__ = [
    "CODECS",
    "CodecSpec",
    "_CoreCodec",
    "_ExtraCodec",
    "blosc_codec",
    "bytes_codec",
    "cast_value_codec",
    "codec_spec_for",
    "crc32c_codec",
    "gzip_codec",
    "scale_offset_codec",
    "sharding_indexed_codec",
    "transpose_codec",
    "zstd_codec",
]
