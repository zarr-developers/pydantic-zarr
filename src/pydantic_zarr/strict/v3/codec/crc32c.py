from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zarr_metadata import Crc32cCodecMetadata

kind: Literal["bytes_bytes"] = "bytes_bytes"


def crc32c() -> Crc32cCodecMetadata:
    return {"name": "crc32c"}


def validate_crc32c(meta: Crc32cCodecMetadata) -> None:
    return None


def ndim_of(meta: Crc32cCodecMetadata) -> int | None:
    return None


def dtype_out(meta: Crc32cCodecMetadata, input_dtype: str) -> str:
    return input_dtype


from zarr_metadata.v3.codec.crc32c import Crc32cCodecObject  # noqa: E402

from pydantic_zarr.strict.v3.codec._spec import CodecSpec  # noqa: E402

SPEC = CodecSpec(
    name="crc32c",
    kind="bytes_bytes",
    metadata_type=Crc32cCodecObject,
    has_dtype_dependent_config=False,
    validate=validate_crc32c,
    ndim_of=ndim_of,
    dtype_out=dtype_out,
)
