from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zarr_metadata import CastValueCodecMetadata, MetadataV3
    from zarr_metadata.v3.codec.cast_value import CastOutOfRangeMode, CastRoundingMode, ScalarMap

kind: Literal["array_array"] = "array_array"


def cast_value(
    data_type: MetadataV3,
    *,
    rounding: CastRoundingMode | None = None,
    out_of_range: CastOutOfRangeMode | None = None,
    scalar_map: ScalarMap | None = None,
) -> CastValueCodecMetadata:
    config: dict = {"data_type": data_type}
    if rounding is not None:
        config["rounding"] = rounding
    if out_of_range is not None:
        config["out_of_range"] = out_of_range
    if scalar_map is not None:
        config["scalar_map"] = scalar_map
    return {"name": "cast_value", "configuration": config}  # type: ignore[typeddict-item]


def validate_cast_value(meta: CastValueCodecMetadata) -> None:
    """Structural only here; scalar dtype-correctness is validated in the pipeline pass."""
    return


def ndim_of(meta: CastValueCodecMetadata) -> int | None:
    return None


def dtype_out(meta: CastValueCodecMetadata, input_dtype: str) -> str:
    dt = meta["configuration"]["data_type"]
    return dt if isinstance(dt, str) else input_dtype  # named-config target -> keep flowing dtype
