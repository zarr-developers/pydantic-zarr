from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zarr_metadata.v3.codec.bytes import BytesCodecObject

from pydantic_zarr.strict.v3.codec._spec import CodecSpec

if TYPE_CHECKING:
    from zarr_metadata import BytesCodecMetadata
    from zarr_metadata.v3.codec.bytes import Endianness

kind: Literal["array_bytes"] = "array_bytes"


def bytes_codec(endian: Endianness = "little") -> BytesCodecMetadata:
    """Builder for the `bytes` codec."""
    return {"name": "bytes", "configuration": {"endian": endian}}


def validate_bytes(meta: BytesCodecMetadata) -> None:
    """No intrinsic checks beyond the typed enum (endian is a Literal)."""
    return


def ndim_of(meta: BytesCodecMetadata) -> int | None:
    return None


def dtype_out(meta: BytesCodecMetadata, input_dtype: str) -> str:
    return input_dtype


SPEC = CodecSpec(
    name="bytes",
    kind="array_bytes",
    metadata_type=BytesCodecObject,
    has_dtype_dependent_config=False,
    validate=validate_bytes,
    ndim_of=ndim_of,
    dtype_out=dtype_out,
)
