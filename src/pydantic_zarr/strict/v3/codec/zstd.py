from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zarr_metadata import ZstdCodecMetadata

kind: Literal["bytes_bytes"] = "bytes_bytes"


def zstd(level: int, checksum: bool = False) -> ZstdCodecMetadata:
    return {"name": "zstd", "configuration": {"level": level, "checksum": checksum}}


def validate_zstd(meta: ZstdCodecMetadata) -> None:
    """The Zarr spec states no hard bound on zstd `level`; no intrinsic check beyond type."""
    return


def ndim_of(meta: ZstdCodecMetadata) -> int | None:
    return None


def dtype_out(meta: ZstdCodecMetadata, input_dtype: str) -> str:
    return input_dtype
