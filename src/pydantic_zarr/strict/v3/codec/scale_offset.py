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


from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodecObject  # noqa: E402

from pydantic_zarr.strict.v3.codec._spec import CodecSpec  # noqa: E402

SPEC = CodecSpec(
    name="scale_offset",
    kind="array_array",
    metadata_type=ScaleOffsetCodecObject,
    has_dtype_dependent_config=False,
    validate=validate_scale_offset,
    ndim_of=ndim_of,
    dtype_out=dtype_out,
)
