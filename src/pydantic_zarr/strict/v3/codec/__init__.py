from __future__ import annotations

from zarr_metadata.v3.codec.blosc import BloscCodecObject
from zarr_metadata.v3.codec.bytes import BytesCodecName, BytesCodecObject
from zarr_metadata.v3.codec.cast_value import CastValueCodecObject
from zarr_metadata.v3.codec.crc32c import Crc32cCodecName, Crc32cCodecObject
from zarr_metadata.v3.codec.gzip import GzipCodecObject
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodecName, ScaleOffsetCodecObject
from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecObject
from zarr_metadata.v3.codec.transpose import TransposeCodecObject
from zarr_metadata.v3.codec.zstd import ZstdCodecObject

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


# bare *CodecName members ONLY for config-optional codecs (config_required is False);
# every codec contributes its *CodecObject. Asserted against config_required in the tests.
_CoreCodec = (
    BytesCodecObject
    | Crc32cCodecObject
    | GzipCodecObject
    | ZstdCodecObject
    | BloscCodecObject
    | TransposeCodecObject
    | ShardingIndexedCodecObject
    | BytesCodecName  # bytes: config optional -> bare allowed
    | Crc32cCodecName  # crc32c: config optional -> bare allowed
)

_ExtraCodec = (
    BytesCodecObject
    | Crc32cCodecObject
    | GzipCodecObject
    | ZstdCodecObject
    | BloscCodecObject
    | TransposeCodecObject
    | ShardingIndexedCodecObject
    | ScaleOffsetCodecObject
    | CastValueCodecObject
    | BytesCodecName  # config optional -> bare allowed
    | Crc32cCodecName  # config optional -> bare allowed
    | ScaleOffsetCodecName  # config optional -> bare allowed
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
