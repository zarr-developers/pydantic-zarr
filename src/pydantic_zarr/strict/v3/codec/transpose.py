from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata import TransposeCodecMetadata

kind: Literal["array_array"] = "array_array"


def transpose(order: Sequence[int]) -> TransposeCodecMetadata:
    order_t = tuple(order)
    meta: TransposeCodecMetadata = {"name": "transpose", "configuration": {"order": order_t}}
    validate_transpose(meta)
    return meta


def validate_transpose(meta: TransposeCodecMetadata) -> None:
    order = tuple(meta["configuration"]["order"])
    if sorted(order) != list(range(len(order))):
        raise ValueError(f"transpose order {order} is not a permutation of range({len(order)})")


def ndim_of(meta: TransposeCodecMetadata) -> int | None:
    return len(meta["configuration"]["order"])


def dtype_out(meta: TransposeCodecMetadata, input_dtype: str) -> str:
    return input_dtype
