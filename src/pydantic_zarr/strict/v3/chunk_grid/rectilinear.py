from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def rectilinear(chunk_shapes: Sequence[Any]) -> dict:
    meta: dict = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": tuple(chunk_shapes)},
    }
    validate_rectilinear(meta)
    return meta


def _leaf_ints(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    out: list[int] = []
    for item in spec:
        if isinstance(item, int):
            out.append(item)
        else:
            out.extend(int(x) for x in item)
    return out


def validate_rectilinear(meta: dict) -> None:
    for dim_spec in meta["configuration"]["chunk_shapes"]:
        if any(v <= 0 for v in _leaf_ints(dim_spec)):
            raise ValueError(f"rectilinear chunk_shapes dim spec {dim_spec!r} must be all positive")


def ndim_of(meta: dict) -> int | None:
    return len(meta["configuration"]["chunk_shapes"])
