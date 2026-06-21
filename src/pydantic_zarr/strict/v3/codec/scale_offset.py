from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zarr_metadata import JSONValue, ScaleOffsetCodecMetadata

kind: Literal["array_array"] = "array_array"


def scale_offset(scale: JSONValue = None, offset: JSONValue = None) -> ScaleOffsetCodecMetadata:
    config: dict = {}
    if scale is not None:
        config["scale"] = scale
    if offset is not None:
        config["offset"] = offset
    return {"name": "scale_offset", "configuration": config}  # type: ignore[typeddict-item]


def validate_scale_offset(meta: ScaleOffsetCodecMetadata) -> None:
    """Scale/offset are JSONValue per fill-value rules; checked in pipeline if needed."""
    return


def ndim_of(meta: ScaleOffsetCodecMetadata) -> int | None:
    return None


def dtype_out(meta: ScaleOffsetCodecMetadata, input_dtype: str) -> str:
    return input_dtype
