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
