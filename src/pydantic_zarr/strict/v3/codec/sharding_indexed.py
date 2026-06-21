from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata import ShardingIndexedCodecMetadata
    from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexLocation

from pydantic_zarr.strict.v3.codec.bytes import bytes_codec
from pydantic_zarr.strict.v3.codec.crc32c import crc32c

kind: Literal["array_bytes"] = "array_bytes"


def sharding_indexed(
    chunk_shape: Sequence[int],
    *,
    codecs: Sequence[Any] = (),
    index_codecs: Sequence[Any] = (),
    index_location: ShardingIndexLocation | None = None,
) -> ShardingIndexedCodecMetadata:
    """Builder for the sharding_indexed codec."""
    inner = tuple(codecs) if codecs else (bytes_codec(),)
    index = tuple(index_codecs) if index_codecs else (bytes_codec(), crc32c())
    config: dict = {
        "chunk_shape": tuple(chunk_shape),
        "codecs": inner,
        "index_codecs": index,
    }
    if index_location is not None:
        config["index_location"] = index_location
    meta: ShardingIndexedCodecMetadata = {"name": "sharding_indexed", "configuration": config}  # type: ignore[typeddict-item]
    validate_sharding_indexed(meta)
    return meta


def validate_sharding_indexed(meta: ShardingIndexedCodecMetadata) -> None:
    """Validate that chunk_shape contains all positive integers."""
    cs = meta["configuration"]["chunk_shape"]
    if any(d <= 0 for d in cs):
        raise ValueError(f"sharding chunk_shape {tuple(cs)} must be all positive")


def ndim_of(meta: ShardingIndexedCodecMetadata) -> int:
    """Return the number of dimensions from chunk_shape."""
    return len(meta["configuration"]["chunk_shape"])


def dtype_out(meta: ShardingIndexedCodecMetadata, input_dtype: str) -> str:
    """Identity transformation; array->bytes terminates dtype flow."""
    return input_dtype
