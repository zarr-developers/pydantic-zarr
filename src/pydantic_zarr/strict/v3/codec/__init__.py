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

# Module dispatch map
_MODULES = {
    "bytes": _bytes,
    "crc32c": crc32c,
    "gzip": gzip,
    "zstd": zstd,
    "blosc": blosc,
    "transpose": transpose,
    "sharding_indexed": sharding_indexed,
    "scale_offset": scale_offset,
    "cast_value": cast_value,
}

CODEC_NDIM_OF = {n: m.ndim_of for n, m in _MODULES.items()}
CODEC_KIND = {n: m.kind for n, m in _MODULES.items()}
CODEC_DTYPE_OUT = {n: m.dtype_out for n, m in _MODULES.items()}
CODEC_VALIDATE = {n: getattr(m, f"validate_{n}") for n, m in _MODULES.items()}

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
    "CODEC_DTYPE_OUT",
    "CODEC_KIND",
    "CODEC_NDIM_OF",
    "CODEC_VALIDATE",
    "_CoreCodec",
    "_ExtraCodec",
    "blosc_codec",
    "bytes_codec",
    "cast_value_codec",
    "crc32c_codec",
    "gzip_codec",
    "scale_offset_codec",
    "sharding_indexed_codec",
    "transpose_codec",
    "zstd_codec",
]
