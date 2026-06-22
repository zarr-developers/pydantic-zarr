from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zarr_metadata import GzipCodecMetadata

kind: Literal["bytes_bytes"] = "bytes_bytes"


def gzip(level: int) -> GzipCodecMetadata:
    meta: GzipCodecMetadata = {"name": "gzip", "configuration": {"level": level}}
    validate_gzip(meta)
    return meta


def validate_gzip(meta: GzipCodecMetadata) -> None:
    level = meta["configuration"]["level"]
    if not (0 <= level <= 9):
        raise ValueError(f"gzip level {level} out of range [0, 9]")


def ndim_of(meta: GzipCodecMetadata) -> int | None:
    return None


def dtype_out(meta: GzipCodecMetadata, input_dtype: str) -> str:
    return input_dtype


from zarr_metadata.v3.codec.gzip import GzipCodecObject  # noqa: E402

from pydantic_zarr.strict.v3.codec._spec import CodecSpec  # noqa: E402

SPEC = CodecSpec(
    name="gzip",
    kind="bytes_bytes",
    metadata_type=GzipCodecObject,
    has_dtype_dependent_config=False,
    validate=validate_gzip,
    ndim_of=ndim_of,
    dtype_out=dtype_out,
)
